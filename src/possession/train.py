"""Fixed, local research training on external weak proxy labels, not final inference."""
import argparse
import hashlib
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import yaml
import games
from .contracts import TaggingEvidence
from .quality import clean_tracking
from .features import build_features
from .model import SCHEMA, N_RAW, TAG_START, Normalizer, TemporalModel, flatten_features, swap_teams


def zero_tags(n):
    return TaggingEvidence(np.zeros((n,2),np.float32),np.zeros(n,np.float32),np.zeros(n,np.float32))


def match_features(match):
    clean = clean_tracking(match.tracking)
    features = build_features(clean)
    return features, zero_tags(len(match.labels))


def prepare_match(match, feature_version=1):
    features, tags = match_features(match)
    raw = flatten_features(features,tags)
    if feature_version == 2:
        from .model_v2 import enrich
        raw = enrich(raw, features.blocks, features.fps)
    elif feature_version != 1:
        raise ValueError('unknown feature version')
    labels = np.asarray(match.labels, dtype=np.int64)
    mask = np.asarray(match.supervision_mask, dtype=bool)
    dead_mask = np.asarray(getattr(match, 'dead_mask', np.zeros(len(raw), dtype=bool)), dtype=bool)
    boundary_mask = np.asarray(match.boundary_mask, dtype=bool)
    if labels.shape != (len(raw),) or mask.shape != (len(raw),):
        raise ValueError('match supervision shape mismatch')
    for name, coverage in (('dead', dead_mask), ('boundary', boundary_mask)):
        if coverage.shape != (len(raw),) or np.any(coverage & ~mask):
            raise ValueError(name + ' supervision must be inside known state supervision')
    if not np.isin(labels, [-1,0,1,2]).all() or np.any(mask & (labels < 0)):
        raise ValueError('invalid proxy labels or mask')
    return {'id': match.match_id, 'raw': raw, 'labels': labels, 'mask': mask,
            'dead_mask': dead_mask, 'boundary_mask': boundary_mask,
            'blocks': features.blocks, 'provenance': match.provenance}


def validate_split(train, validation):
    a = {str(m['id']) for m in train}; b = {str(m['id']) for m in validation}
    if not a or not b or a & b or len(a) != len(train) or len(b) != len(validation):
        raise ValueError('nonempty, unique and disjoint match IDs required')


def windows(match, center_frames, context_frames):
    """Yield (context_start, context_end, supervised_center_start, center_end)."""
    if center_frames <= 0 or context_frames < TemporalModel.receptive_radius:
        raise ValueError('context must cover the TCN receptive radius')
    n = len(match['raw'])
    if np.shape(match['labels']) != (n,) or np.shape(match['mask']) != (n,):
        raise ValueError('match supervision shape mismatch')
    previous_end = 0
    for start,end in match['blocks']:
        if (not isinstance(start, (int, np.integer)) or not isinstance(end, (int, np.integer))
                or start < previous_end or end <= start or end > n):
            raise ValueError('training blocks must be ordered nonoverlapping intervals within match')
        previous_end = end
        for left in range(start,end,center_frames):
            right=min(end,left+center_frames)
            if match['mask'][left:right].any():
                yield max(start,left-context_frames),min(end,right+context_frames),left,right


def augment(raw, labels, rng, cfg):
    x=raw.copy(); y=labels.copy()
    previous_ages = None
    if x.shape[1] != N_RAW and len(x):
        previous_ages = tuple(max(0., float(v)-.04) if np.isfinite(v) else np.nan
                              for v in x[0,N_RAW+3:N_RAW+5])
    if rng.random() < cfg['team_swap']:
        x,y=swap_teams(x,y)
    if rng.random() < cfg['tag_dropout']:
        x[:,TAG_START:N_RAW] = 0
    if len(x) and rng.random() < cfg['missing_block_probability']:
        first=int(rng.integers(len(x))); last=min(len(x),first+int(cfg['missing_block_frames']))
        offset=0
        unknown={'nearest_distance','ball_speed','boundary_distance'}
        zero={'control_support','nearby_player_count','ball_observed','ball_interpolated','ball_available','ball_rejected'}
        for name,width in SCHEMA:
            # Match build_features' missing-ball representation: distances/motion
            # unknown, proximity/counts zero, quality flags explicitly observed.
            if name in unknown: x[first:last,offset:offset+width]=np.nan
            if name in zero: x[first:last,offset:offset+width]=0
            offset += width
    if x.shape[1] != N_RAW:
        from .model_v2 import enrich
        x = enrich(x[:,:N_RAW], [(0,len(x))], previous_ages=previous_ages)
    return x,y


def masked_loss(dead_logits, team_logits, labels, mask, dead_head_trained, dead_mask=None):
    """Unsupervised rows never contribute; no D negatives if D positives absent."""
    mask=mask.bool() & (labels>=0)
    if not bool(mask.any()): return None
    live=mask & (labels<2)
    loss=team_logits.sum()*0
    if bool(live.any()):
        loss=loss+F.cross_entropy(team_logits[live],labels[live])
    if dead_head_trained:
        reliable_dead = mask & dead_mask.bool() if dead_mask is not None else torch.zeros_like(mask)
        if bool(reliable_dead.any()):
            loss=loss+F.binary_cross_entropy_with_logits(dead_logits[reliable_dead],(labels[reliable_dead]==2).float())
    return loss


def evaluate_live_proxy(model, normalizer, matches, device):
    """Evaluate H/A on labeled live frames of independently held-out matches."""
    confusion = np.zeros((2, 2), dtype=np.int64)
    model.eval()
    with torch.inference_mode():
        for match in matches:
            for start, end in match['blocks']:
                for left in range(start, end, 512):
                    right = min(left + 512, end)
                    lo, hi = max(start, left-TemporalModel.receptive_radius), min(end, right+TemporalModel.receptive_radius)
                    x = torch.from_numpy(normalizer.transform(match['raw'][lo:hi]).T[None]).to(device)
                    _, team = model(x)
                    scores = team[0, :, left-lo:right-lo].T
                    if not bool(torch.isfinite(scores).all()):
                        raise ValueError('nonfinite validation logits')
                    valid = match['mask'][left:right] & (match['labels'][left:right] < 2)
                    true = match['labels'][left:right][valid]
                    predicted = scores.argmax(dim=1).cpu().numpy()[valid]
                    np.add.at(confusion, (true, predicted), 1)
    support = confusion.sum(axis=1)
    f1 = [float(2*confusion[i,i]/(support[i]+confusion[:,i].sum()))
          if support[i]+confusion[:,i].sum() else None for i in range(2)]
    return {'scope': 'held-out labeled live H/A frames; label semantics depend on source', 'support': support.tolist(),
            'f1': {'H': f1[0], 'A': f1[1]},
            'macro_f1': float(np.mean(f1)) if all(value is not None for value in f1) else None,
            'D_f1': None, 'boundary_f1': None, 'spell_f1': None}


def fit(train, validation, config, device='cpu', epoch_callback=None):
    if validation:
        validate_split(train, validation)
    elif not config.get('fixed_epoch_final_training', False):
        raise ValueError('validation is required except for fixed-epoch final training')
    elif not train or len({str(m['id']) for m in train}) != len(train):
        raise ValueError('final training requires unique, nonempty training matches')
    if int(config['epochs']) < 1 or int(config['max_train_windows']) < 1:
        raise ValueError('epochs and max_train_windows must be positive')
    candidates=[(i,m,w) for i,m in enumerate(train) for w in windows(m,int(config['center_frames']),int(config['context_frames']))]
    if not candidates: raise ValueError('no supervised training windows')
    for m in train:
        if not np.isin(m['labels'], [-1,0,1,2]).all() or np.any(m['mask'] & (m['labels'] < 0)):
            raise ValueError('invalid proxy labels or mask')
        within = np.zeros(len(m['raw']), bool)
        for start,end in m['blocks']: within[start:end] = True
        if np.any(m['mask'] & ~within):
            raise ValueError('supervision outside training blocks')
        for key in ('dead_mask', 'boundary_mask'):
            if key in m and (np.shape(m[key]) != np.shape(m['mask']) or np.any(m[key] & ~m['mask'])):
                raise ValueError(key + ' must be inside known state supervision')
    seed=int(config['seed']); random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if device.startswith('cuda'): torch.cuda.manual_seed_all(seed)
    version=int(config.get('feature_version',1))
    if version == 2:
        from .model_v2 import NormalizerV2, TemporalModelV2
        normalizer=NormalizerV2.fit([m['raw'][start:end] for m in train for start,end in m['blocks']])
        model_class=TemporalModelV2
    elif version == 1:
        normalizer=Normalizer.fit([m['raw'][start:end] for m in train for start,end in m['blocks']])
        model_class=TemporalModel
    else: raise ValueError('unknown feature version')
    counts=np.zeros(3,dtype=np.int64)
    for m in train:
        counts += np.bincount(m['labels'][m['mask']],minlength=3)[:3]
    if counts[:2].sum()==0: raise ValueError('no supervised live training samples')
    dead_counts=np.zeros(2,dtype=np.int64)
    for m in train:
        reliable = m['mask'] & m.get('dead_mask', np.zeros_like(m['mask']))
        dead_counts[0] += int(np.count_nonzero(reliable & (m['labels'] < 2)))
        dead_counts[1] += int(np.count_nonzero(reliable & (m['labels'] == 2)))
    dead_head_trained=bool(np.all(dead_counts > 0))
    model=model_class(channels=int(config['channels'])).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=float(config['learning_rate']))
    rng=np.random.default_rng(seed)
    seen = [np.zeros(len(m['raw']), bool) for m in train]
    train_losses=[]; step=0; started=time.perf_counter()
    if device.startswith('cuda'):
        torch.cuda.reset_peak_memory_stats()
    model.train()
    epoch_losses=[]; validation_proxy=[]; epoch_metrics=[]
    for epoch in range(int(config['epochs'])):
        epoch_start = len(train_losses)
        for j in rng.permutation(len(candidates))[:int(config['max_train_windows'])]:
            match_index,m,(lo,hi,left,right)=candidates[j]
            raw,lab=augment(m['raw'][lo:hi],m['labels'][lo:hi],rng,config)
            x=torch.from_numpy(normalizer.transform(raw).T[None]).to(device)
            labels=torch.from_numpy(lab).to(device)
            mask=np.zeros(hi-lo,bool); mask[left-lo:right-lo]=m['mask'][left:right]
            supervised=torch.from_numpy(mask).to(device)
            optimizer.zero_grad(set_to_none=True)
            dead,team=model(x)
            dead_mask = torch.from_numpy(m.get('dead_mask', np.zeros_like(m['mask']))[lo:hi]).to(device)
            loss=masked_loss(dead[0],team[0].T,labels,supervised,dead_head_trained,dead_mask)
            if loss is None: continue
            if version == 2:
                # Adjacent state labels supervise a boundary only when both are reliable.
                known = m.get('boundary_mask', np.zeros_like(m['mask']))[lo:hi]
                valid = np.zeros(hi-lo, bool)
                valid[1:] = known[1:] & known[:-1] & mask[1:]
                target = np.zeros(hi-lo, np.float32)
                target[1:] = (lab[1:] != lab[:-1]).astype(np.float32)
                keep = torch.from_numpy(valid).to(device)
                if bool(keep.any()):
                    b = model.boundary_logits[0]
                    yb = torch.from_numpy(target).to(device)
                    loss = loss + float(config.get('boundary_loss_weight',.1))*F.binary_cross_entropy_with_logits(b[keep],yb[keep],pos_weight=torch.tensor(20.,device=device))
                    pair = keep & ~yb.bool()
                    if bool(pair.any()):
                        probs = torch.softmax(team[0].T,dim=1)
                        idx = torch.nonzero(pair,as_tuple=True)[0]
                        loss = loss + float(config.get('smoothness_weight',.05))*((probs[idx]-probs[idx-1])**2).sum(dim=1).mean()
            loss.backward(); optimizer.step(); step+=1; train_losses.append(float(loss.detach().cpu()))
            seen[match_index][left:right] |= m['mask'][left:right]
        epoch_losses.append(float(np.mean(train_losses[epoch_start:])))
        if validation:
            validation_proxy.append(evaluate_live_proxy(model, normalizer, validation, device))
        if epoch_callback is not None:
            epoch_metrics.append(epoch_callback(epoch + 1, model, normalizer, optimizer, dead_head_trained))
        model.train()
    if device.startswith('cuda'):
        torch.cuda.synchronize()
    return model,normalizer,optimizer,{'class_counts':counts.tolist(),'dead_head_trained':dead_head_trained,
           'dead_supervision_counts':{'negative_live':int(dead_counts[0]),'positive_D':int(dead_counts[1])},
           'boundary_supervision_frames':sum(int(np.count_nonzero(m.get('boundary_mask', np.zeros_like(m['mask'])))) for m in train),
           'candidate_windows':len(candidates),'epochs':int(config['epochs']),
           'windows_per_epoch':min(len(candidates),int(config['max_train_windows'])),
           'unique_supervised_samples_seen':sum(int(s.sum()) for s in seen),
           'supervision_coverage':sum(int(s.sum()) for s in seen)/int(counts.sum()),
           'epoch_train_loss':epoch_losses,
           'epoch_validation_proxy':validation_proxy,
           'epoch_metrics':epoch_metrics,
           'final_selection_eligible':False,
           'augmentation_version':'missing_ball_features_v2',
           'steps':step,'mean_train_loss':float(np.mean(train_losses)),'elapsed_seconds':time.perf_counter()-started,
           'peak_allocated_mib':torch.cuda.max_memory_allocated()/1048576 if device.startswith('cuda') else None}


def file_hash(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()


def load_checkpoint(path, device='cpu'):
    """Load a trusted local research checkpoint and reconstruct its fixed schema."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    schema=tuple(map(tuple, checkpoint['schema']))
    if schema == SCHEMA:
        model_class, norm_class, width = TemporalModel, Normalizer, N_RAW
    else:
        from .model_v2 import SCHEMA_V2, TemporalModelV2, NormalizerV2, WIDTH
        if schema != SCHEMA_V2: raise ValueError('checkpoint feature schema mismatch')
        model_class, norm_class, width = TemporalModelV2, NormalizerV2, WIDTH
    model = model_class(channels=int(checkpoint['config']['channels'])).to(device)
    model.load_state_dict(checkpoint['model'])
    norm = norm_class(np.asarray(checkpoint['normalizer']['mean'], dtype=np.float32),
                      np.asarray(checkpoint['normalizer']['scale'], dtype=np.float32))
    if norm.mean.shape != (width,) or norm.scale.shape != (width,) or not np.all(norm.scale > 0):
        raise ValueError('invalid checkpoint normalization')
    return model, norm, checkpoint


def main(argv=None):
    from .external_data import load_metrica_csv
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/temporal.yaml')
    parser.add_argument('--data-dir',required=True,help='directory containing Sample_Game_1 and Sample_Game_2')
    parser.add_argument('--output',default='data/runs/training/temporal.pt')
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    args=parser.parse_args(argv)
    with open(args.config,encoding='utf-8') as f: cfg=yaml.safe_load(f)
    root=Path(args.data_dir); files={}
    def load(game):
        d=root/('Sample_Game_'+str(game)); prefix='Sample_Game_'+str(game)
        paths=[d/(prefix+'_RawTrackingData_Home_Team.csv'),d/(prefix+'_RawTrackingData_Away_Team.csv'),d/(prefix+'_RawEventsData.csv')]
        files[str(game)]={p.name:file_hash(p) for p in paths}
        return prepare_match(load_metrica_csv(str(game),*paths))
    train=[load(g) for g in cfg['train_games']]; val=[load(g) for g in cfg['validation_games']]
    model,norm,opt,stats=fit(train,val,cfg,args.device)
    dest=Path(args.output).resolve(); allowed=Path('data/runs/training').resolve()
    if not dest.is_relative_to(allowed): raise ValueError('checkpoint must be under data/runs/training')
    dest.parent.mkdir(parents=True,exist_ok=True)
    source={p.name:file_hash(p) for p in sorted(Path(__file__).resolve().parent.glob('*.py'))}
    source['games.py'] = file_hash(games.__file__)
    project_root = Path(__file__).resolve().parents[2]
    for name in ('pyproject.toml', 'uv.lock'):
        path = project_root / name
        if path.is_file(): source[name] = file_hash(path)
    checkpoint={'model':model.state_dict(),'optimizer':opt.state_dict(),'normalizer':{'mean':norm.mean,'scale':norm.scale},
                'schema':SCHEMA,'config':cfg,'config_sha256':file_hash(args.config),'input_sha256':files,
                'source_sha256':source,'stats':stats,'train_matches':[m['id'] for m in train],
                'validation_matches':[m['id'] for m in val], 'device':args.device,'torch_version':torch.__version__}
    torch.save(checkpoint,dest)
    print(json.dumps({'checkpoint':str(dest),'checkpoint_sha256':file_hash(dest),**stats}))

if __name__=='__main__': main()
