"""Train on whole DFL matches and select by held-out decoded spellF1."""
import argparse
import copy
from dataclasses import replace
import json
from pathlib import Path
import numpy as np
import torch
import yaml
from possession.dfl_data import load_dfl
from possession.train import fit, prepare_match, zero_tags, file_hash, validate_split
from possession.model import SCHEMA, infer_evidence
from possession.quality import clean_tracking
from possession.features import build_features
from possession.teacher import build_evidence
from possession.decode import decode_labels
from possession.external_evaluation import evaluate_external
from possession.predict import load_config
from possession.external_data import load_metrica_csv


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', default='data/runs/idsse')
    p.add_argument('--train', nargs='+', required=True)
    p.add_argument('--validation', required=True)
    p.add_argument('--test', required=True, help='reserved match ID; never loaded here')
    p.add_argument('--metrica-dir', default='data/external/metrica-sample-data')
    p.add_argument('--metrica-train', nargs='*', default=[])
    p.add_argument('--output', required=True)
    p.add_argument('--epochs', type=int, default=6)
    p.add_argument('--learning-rate', type=float, default=.0003)
    p.add_argument('--feature-version', type=int, choices=(1,2), default=1)
    p.add_argument('--boundary-loss-weight', type=float, default=.1)
    p.add_argument('--smoothness-weight', type=float, default=.05)
    p.add_argument('--missing-block-probability', type=float, default=None)
    p.add_argument('--missing-block-frames', type=int, default=None)
    p.add_argument('--observation-pattern-data-dir', default=None,
                   help='add training copies masked by raw target observation availability; never reads target labels')
    p.add_argument('--device', choices=('cpu','cuda'), default='cuda')
    args = p.parse_args()
    ids = args.train + [args.validation, args.test]
    if (len(ids) != len(set(ids)) or args.epochs < 1 or args.learning_rate <= 0
            or args.boundary_loss_weight < 0 or args.smoothness_weight < 0
            or (args.missing_block_probability is not None and not 0 <= args.missing_block_probability <= 1)
            or (args.missing_block_frames is not None and args.missing_block_frames < 1)):
        raise ValueError('unique disjoint train/validation/test IDs and positive hyperparameters required')
    dest = Path(args.output).resolve()
    if not dest.is_relative_to(Path('data/runs/training').resolve()) or dest.exists():
        raise ValueError('output must be a new checkpoint under data/runs/training')
    root = Path(args.data_dir)
    pattern=None
    if args.observation_pattern_data_dir:
        from possession.io import load_tracking
        from possession.observation_masks import transfer_ball_mask
        pattern=load_tracking(args.observation_pattern_data_dir)
    def load(game):
        source = load_dfl(root / (game+'_positions.xml'), root / (game+'_info.xml'))
        if source.match_id != 'DFL-MAT-'+game: raise ValueError('requested match id does not match data')
        prepared = prepare_match(source, args.feature_version)
        prepared['dead_mask'] = prepared['mask'].copy()
        print(json.dumps({'loaded':game,'frames':len(source.labels)}), flush=True)
        return source, prepared
    train_sources, train = [], []
    for game in args.train:
        source, prepared = load(game)
        train_sources.append(source.provenance); train.append(prepared)
        if pattern is not None:
            masked,dropped=transfer_ball_mask(source.tracking,pattern,offset=len(train)*997)
            augmented=prepare_match(replace(source,tracking=masked,match_id=source.match_id+':observation_masked'),args.feature_version)
            augmented['dead_mask']=augmented['mask'].copy()
            train_sources.append(source.provenance); train.append(augmented)
            print(json.dumps({'masked_copy':game,'dropped':int(dropped.sum())}),flush=True)
        del source
    for game in args.metrica_train:
        prefix = 'Sample_Game_' + game
        folder = Path(args.metrica_dir) / prefix
        paths = [folder / (prefix+'_RawTrackingData_'+team+'_Team.csv') for team in ('Home','Away')]
        paths.append(folder / (prefix+'_RawEventsData.csv'))
        source = load_metrica_csv('metrica-'+game, *paths)
        prepared = prepare_match(source, args.feature_version)
        train_sources.append(source.provenance); train.append(prepared)
        print(json.dumps({'loaded':'metrica-'+game, 'frames':len(source.labels),
                          'supervised':int(source.supervision_mask.sum())}), flush=True)
        del source
    val_source, validation = load(args.validation)
    validate_split(train, [validation])
    rules = load_config('configs/rules.yaml')
    tags = zero_tags(len(val_source.labels))
    features = build_features(clean_tracking(val_source.tracking, rules.get('quality',{}), tags.restart), rules.get('features',{}))
    baseline = build_evidence(features, tags, rules)
    validation_views=[('clean',features,baseline)]
    if pattern is not None:
        tracking,_=transfer_ball_mask(val_source.tracking,pattern,offset=7919)
        masked_features=build_features(clean_tracking(tracking,rules.get('quality',{}),tags.restart),rules.get('features',{}))
        validation_views.append(('observation_masked',masked_features,build_evidence(masked_features,tags,rules)))
    cfg = yaml.safe_load(Path('configs/temporal.yaml').read_text(encoding='utf-8'))
    cfg.update(epochs=args.epochs, max_train_windows=100000, seed=2032, learning_rate=args.learning_rate,
               feature_version=args.feature_version, boundary_loss_weight=args.boundary_loss_weight,
               smoothness_weight=args.smoothness_weight)
    if args.missing_block_probability is not None:
        cfg['missing_block_probability'] = args.missing_block_probability
    if args.missing_block_frames is not None:
        cfg['missing_block_frames'] = args.missing_block_frames
    best = {'score':float('-inf')}
    def evaluate(epoch, model, norm, optimizer, dead_trained):
        view_metrics={}
        for name,view,base in validation_views:
            evidence=infer_evidence(model,view,tags,base,norm,args.device,dead_head_trained=dead_trained)
            view_metrics[name]=evaluate_external(val_source,decode_labels(evidence,view.blocks,rules))
        metrics=view_metrics['clean']
        spell = metrics['spell_f1']
        if spell is None: raise ValueError('validation does not support spellF1')
        score = float(np.mean([v['spell_f1']['mean_f1'] for v in view_metrics.values()]))
        if not np.isfinite(score): raise ValueError('nonfinite validation spellF1')
        if score > best['score']:
            best.update(score=score, epoch=epoch, model=copy.deepcopy(model.state_dict()),
                        optimizer=copy.deepcopy(optimizer.state_dict()))
        record = {'epoch':epoch, 'spell_f1':spell, 'macro_f1':metrics['macro_f1'],
                  'D':metrics['classes']['D'], 'boundary':metrics['boundary'],
                  'selection_score':score,'view_spell_f1':{k:v['spell_f1'] for k,v in view_metrics.items()}}
        print(json.dumps(record), flush=True)
        return record
    model, norm, optimizer, stats = fit(train, [validation], cfg, args.device, epoch_callback=evaluate)
    if args.feature_version == 2:
        from possession.model_v2 import SCHEMA_V2
        schema = SCHEMA_V2
    else: schema = SCHEMA
    checkpoint = {'model':best['model'], 'optimizer':best['optimizer'],
                  'normalizer':{'mean':norm.mean, 'scale':norm.scale}, 'schema':schema,
                  'config':cfg, 'stats':stats, 'train_matches':[m['id'] for m in train],
                  'validation_matches':[validation['id']], 'reserved_test_match':'DFL-MAT-'+args.test,
                  'selection':{'criterion':'highest mean decoded spellF1 across configured validation views, earliest tie',
                               'epoch':best['epoch'], 'spell_f1':best['score']},
                  'input_sha256':{m['id']:s['sha256'] for m,s in zip(train,train_sources)} |
                                 {validation['id']:val_source.provenance['sha256']},
                  'source_sha256':{name:file_hash(name) for name in
                      ('scripts/train_dfl_multi.py','src/possession/train.py','src/possession/dfl_data.py',
                       'src/possession/model.py','src/possession/model_v2.py','src/possession/decode.py','src/possession/teacher.py',
                       'src/possession/features.py','src/possession/external_evaluation.py','configs/rules.yaml')}}
    if pattern is not None:
        import games
        paths=games.asset_paths(data_dir=args.observation_pattern_data_dir)
        checkpoint['observation_pattern_sha256']={k:file_hash(paths[k]) for k in ('atd_meta','atd_raw')}
        checkpoint['source_sha256']['src/possession/observation_masks.py']=file_hash('src/possession/observation_masks.py')
    dest.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, dest)
    print(json.dumps({'checkpoint':str(dest),'sha256':file_hash(dest),'selection':checkpoint['selection'],
                      'stats':stats}), flush=True)


if __name__ == '__main__': main()
