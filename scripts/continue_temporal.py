"""Continue the fixed H/A proxy TCN experiment from its trusted local checkpoint."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from possession.external_data import load_metrica_csv
from possession.model import SCHEMA, N_RAW
from possession.predict import sha256, source_provenance
from possession.train import (augment, load_checkpoint, masked_loss, prepare_match,
                              validate_split, windows)


def load_match(root, game):
    prefix = f'Sample_Game_{game}'
    folder = root / prefix
    paths = [folder / f'{prefix}_RawTrackingData_{team}_Team.csv' for team in ('Home', 'Away')]
    paths.append(folder / f'{prefix}_RawEventsData.csv')
    hashes = {p.name: sha256(p) for p in paths}
    return prepare_match(load_metrica_csv(str(game), *paths)), hashes


def proxy_accuracy(model, normalizer, match, device):
    """Masked H/A pass-interior accuracy; this cannot assess D or spell quality."""
    model.eval()
    correct = count = 0
    with torch.inference_mode():
        for start, end in match['blocks']:
            for left in range(start, end, 512):
                right = min(left + 512, end)
                lo, hi = max(start, left-256), min(end, right+256)
                x = torch.from_numpy(normalizer.transform(match['raw'][lo:hi]).T[None]).to(device)
                _, team = model(x)
                logits = team[0, :, left-lo:right-lo].T
                if not bool(torch.isfinite(logits).all()):
                    raise ValueError('nonfinite validation logits')
                mask = match['mask'][left:right] & (match['labels'][left:right] < 2)
                prediction = logits.argmax(dim=1).cpu().numpy()
                correct += int(np.count_nonzero(prediction[mask] == match['labels'][left:right][mask]))
                count += int(mask.sum())
    if not count:
        raise ValueError('no held-out H/A proxy labels')
    return {'correct': correct, 'support': count, 'accuracy': correct/count,
            'scope': 'held-out pass-interior H/A proxy only'}


def continue_training(checkpoint_path, data_dir, output, epochs, device, seed, consistency_weight=0.):
    if epochs < 1 or not torch.cuda.is_available() and device == 'cuda':
        raise ValueError('positive epochs and an available device required')
    if not 0 <= consistency_weight <= 1:
        raise ValueError('consistency weight must be in [0, 1]')
    dest = Path(output).resolve()
    allowed = Path('data/runs/training').resolve()
    if not dest.is_relative_to(allowed) or dest.exists() or dest == Path(checkpoint_path).resolve():
        raise ValueError('new checkpoint must be a distinct unused path under data/runs/training')
    model, normalizer, previous = load_checkpoint(checkpoint_path, device)
    if previous['stats']['dead_head_trained'] or previous['stats']['class_counts'][2] != 0:
        raise ValueError('this continuation is only for the fixed H/A-only checkpoint')
    if tuple(map(tuple, previous['schema'])) != SCHEMA or normalizer.mean.shape != (N_RAW,):
        raise ValueError('checkpoint feature contract mismatch')
    cfg = previous['config']
    train, train_hashes = load_match(Path(data_dir), str(previous['train_matches'][0]))
    validation, val_hashes = load_match(Path(data_dir), str(previous['validation_matches'][0]))
    validate_split([train], [validation])
    for game, hashes in ((train['id'], train_hashes), (validation['id'], val_hashes)):
        if previous['input_sha256'][str(game)] != hashes:
            raise ValueError('external input differs from original checkpoint')
    candidates = list(windows(train, int(cfg['center_frames']), int(cfg['context_frames'])))
    if not candidates:
        raise ValueError('no supervised training windows')
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg['learning_rate']))
    optimizer.load_state_dict(previous['optimizer'])
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    seen = np.zeros(len(train['raw']), bool)
    stats = []
    started = time.perf_counter()
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    initial_validation = proxy_accuracy(model, normalizer, validation, device)
    for epoch in range(epochs):
        model.train()
        losses = []
        for index in rng.permutation(len(candidates)):
            lo, hi, left, right = candidates[index]
            raw, label = augment(train['raw'][lo:hi], train['labels'][lo:hi], rng, cfg)
            x = torch.from_numpy(normalizer.transform(raw).T[None]).to(device)
            y = torch.from_numpy(label).to(device)
            mask = np.zeros(hi-lo, bool)
            mask[left-lo:right-lo] = train['mask'][left:right]
            optimizer.zero_grad(set_to_none=True)
            dead, team = model(x)
            loss = masked_loss(dead[0], team[0].T, y, torch.from_numpy(mask).to(device), False)
            if loss is None:
                continue
            if consistency_weight:
                masked_raw, _ = augment(raw, label, rng, {**cfg, 'team_swap': 0.,
                    'tag_dropout': 0., 'missing_block_probability': 1., 'missing_block_frames': 100})
                masked_x = torch.from_numpy(normalizer.transform(masked_raw).T[None]).to(device)
                _, masked_team = model(masked_x)
                live = torch.from_numpy(mask & (label < 2)).to(device)
                if bool(live.any()):
                    original = team[0].T[live].detach()
                    occluded = masked_team[0].T[live]
                    loss = loss + consistency_weight * F.kl_div(
                        F.log_softmax(occluded, dim=1), F.softmax(original, dim=1), reduction='batchmean')
            if not bool(torch.isfinite(loss)):
                raise ValueError('nonfinite training loss')
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            seen[left:right] |= train['mask'][left:right]
        validation_proxy = proxy_accuracy(model, normalizer, validation, device)
        stats.append({'epoch': epoch+1, 'steps': len(losses), 'mean_train_loss': float(np.mean(losses)),
                      'validation_proxy': validation_proxy})
        print(json.dumps(stats[-1]), flush=True)
    if device == 'cuda':
        torch.cuda.synchronize()
    source = source_provenance()
    source['source_files_sha256']['scripts/continue_temporal.py'] = sha256(__file__)
    metadata = {'source_checkpoint_sha256': sha256(checkpoint_path), 'source_checkpoint_steps': previous['stats']['steps'],
                'continued_epochs': epochs, 'seed': seed, 'all_candidate_windows_per_epoch': len(candidates),
                'consistency_weight': consistency_weight,
                'consistency_missing_block_frames': 100 if consistency_weight else 0,
                'unique_supervised_frames_seen': int(seen.sum()), 'supervised_frames_total': int(train['mask'].sum()),
                'initial_validation_proxy': initial_validation, 'epochs': stats,
                'elapsed_seconds': time.perf_counter()-started,
                'peak_allocated_mib': torch.cuda.max_memory_allocated()/1048576 if device == 'cuda' else None,
                'device': device, 'D_head_trained': False, 'boundary_head_trained': False,
                'selection_policy': 'fixed final epoch; validation recorded, not used for checkpoint selection',
                'input_sha256': {str(train['id']): train_hashes, str(validation['id']): val_hashes},
                'source': source}
    dest.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = dict(previous)
    total_steps = int(previous['stats']['steps']) + sum(epoch['steps'] for epoch in stats)
    checkpoint['stats'] = {**previous['stats'], 'steps': total_steps,
                           'continued_steps': total_steps-int(previous['stats']['steps']),
                           'continued_epochs': epochs, 'dead_head_trained': False,
                           'mean_train_loss': stats[-1]['mean_train_loss']}
    checkpoint.update(model=model.state_dict(), optimizer=optimizer.state_dict(),
                      continuation=metadata, device=device)
    torch.save(checkpoint, dest)
    manifest = dest.with_suffix('.json')
    metadata['output_sha256'] = sha256(dest)
    manifest.write_text(json.dumps(metadata, indent=2, sort_keys=True)+'\n', encoding='utf-8')
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', default='data/runs/training/temporal.pt')
    parser.add_argument('--data-dir', default='data/external/metrica-sample-data')
    parser.add_argument('--output', default='data/runs/training/temporal_continued.pt')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--seed', type=int, default=2028)
    parser.add_argument('--consistency-weight', type=float, default=0.)
    args = parser.parse_args()
    continue_training(args.checkpoint, args.data_dir, args.output, args.epochs, args.device, args.seed,
                      args.consistency_weight)


if __name__ == '__main__':
    main()
