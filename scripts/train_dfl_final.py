"""Train a fixed-epoch TCN on all seven DFL matches, without target labels."""
import argparse
from dataclasses import replace
import json
from pathlib import Path

import torch
import yaml

from possession.dfl_data import load_dfl
from possession.external_data import load_metrica_csv
from possession.io import load_tracking
from possession.model_v2 import SCHEMA_V2
from possession.observation_masks import transfer_ball_mask
from possession.train import file_hash, fit, prepare_match

GAMES = ('J03WN1', 'J03WMX', 'J03WPY', 'J03WQQ', 'J03WOY', 'J03WOH', 'J03WR9')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', default='data/runs/idsse')
    parser.add_argument('--metrica-dir', default='data/external/metrica-sample-data')
    parser.add_argument('--observation-pattern-data-dir', default='data')
    parser.add_argument('--output', required=True)
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--learning-rate', type=float, default=0.0003)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    args = parser.parse_args()
    dest = Path(args.output).resolve()
    if not dest.is_relative_to(Path('data/runs/training').resolve()) or dest.exists():
        raise ValueError('output must be a new checkpoint under data/runs/training')
    if args.epochs < 1 or args.learning_rate <= 0:
        raise ValueError('epochs and learning rate must be positive')

    pattern = load_tracking(args.observation_pattern_data_dir)
    train, provenance = [], {}
    for game in GAMES:
        root = Path(args.data_dir)
        source = load_dfl(root / (game + '_positions.xml'), root / (game + '_info.xml'))
        if source.match_id != 'DFL-MAT-' + game:
            raise ValueError('requested match ID does not match data')
        item = prepare_match(source, 2)
        item['dead_mask'] = item['mask'].copy()
        train.append(item)
        provenance[item['id']] = source.provenance['sha256']
        print(json.dumps({'loaded': game, 'frames': len(source.labels)}), flush=True)
        masked, dropped = transfer_ball_mask(source.tracking, pattern, offset=len(train) * 997)
        item = prepare_match(replace(source, tracking=masked,
                                    match_id=source.match_id + ':observation_masked'), 2)
        item['dead_mask'] = item['mask'].copy()
        train.append(item)
        provenance[item['id']] = source.provenance['sha256']
        print(json.dumps({'masked_copy': game, 'dropped': int(dropped.sum())}), flush=True)

    for game in ('1', '2'):
        prefix = 'Sample_Game_' + game
        folder = Path(args.metrica_dir) / prefix
        paths = [folder / (prefix + '_RawTrackingData_' + team + '_Team.csv') for team in ('Home', 'Away')]
        paths.append(folder / (prefix + '_RawEventsData.csv'))
        source = load_metrica_csv('metrica-' + game, *paths)
        item = prepare_match(source, 2)
        train.append(item)
        provenance[item['id']] = source.provenance['sha256']
        print(json.dumps({'loaded': 'metrica-' + game, 'frames': len(source.labels),
                          'supervised': int(source.supervision_mask.sum())}), flush=True)

    cfg = yaml.safe_load(Path('configs/temporal.yaml').read_text(encoding='utf-8'))
    cfg.update(epochs=args.epochs, max_train_windows=100000, seed=2032,
               learning_rate=args.learning_rate, feature_version=2,
               boundary_loss_weight=0.1, smoothness_weight=0.05,
               fixed_epoch_final_training=True)

    def progress(epoch, model, norm, optimizer, dead_trained):
        print(json.dumps({'epoch_complete': epoch}), flush=True)
        return {'epoch': epoch}

    model, norm, optimizer, stats = fit(train, [], cfg, args.device, epoch_callback=progress)
    import games
    paths = games.asset_paths(data_dir=args.observation_pattern_data_dir)
    sources = ('scripts/train_dfl_final.py', 'src/possession/train.py',
               'src/possession/dfl_data.py', 'src/possession/external_data.py',
               'src/possession/model.py', 'src/possession/model_v2.py',
               'src/possession/observation_masks.py', 'src/possession/features.py',
               'src/possession/quality.py', 'configs/temporal.yaml')
    checkpoint = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                  'normalizer': {'mean': norm.mean, 'scale': norm.scale}, 'schema': SCHEMA_V2,
                  'config': cfg, 'stats': stats, 'train_matches': [m['id'] for m in train],
                  'validation_matches': [], 'reserved_test_match': None,
                  'selection': {'criterion': 'fixed epoch chosen before final training', 'epoch': args.epochs},
                  'input_sha256': provenance,
                  'observation_pattern_sha256': {k: file_hash(paths[k]) for k in ('atd_meta', 'atd_raw')},
                  'source_sha256': {name: file_hash(name) for name in sources}}
    dest.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, dest)
    print(json.dumps({'checkpoint': str(dest), 'sha256': file_hash(dest),
                      'selection': checkpoint['selection'], 'stats': stats}), flush=True)


if __name__ == '__main__':
    main()
