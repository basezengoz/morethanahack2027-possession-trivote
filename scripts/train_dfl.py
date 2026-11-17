"""Train a research TCN on one DFL match, validating on a different match."""
import argparse
import json
from pathlib import Path
import torch
import yaml
from possession.dfl_data import load_dfl
from possession.train import fit, prepare_match, file_hash
from possession.model import SCHEMA


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('train-positions', 'train-info', 'validation-positions', 'validation-info', 'output'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    args = parser.parse_args()
    dest = Path(args.output).resolve()
    if not dest.is_relative_to(Path('data/runs/training').resolve()) or dest.exists():
        raise ValueError('new private checkpoint path required')
    train_source = load_dfl(args.train_positions, args.train_info)
    validation_source = load_dfl(args.validation_positions, args.validation_info)
    train = prepare_match(train_source)
    validation = prepare_match(validation_source)
    train['dead_mask'] = train['mask'].copy()
    config = yaml.safe_load(Path('configs/temporal.yaml').read_text(encoding='utf-8'))
    config.update(epochs=args.epochs, max_train_windows=100000, seed=2031)
    model, norm, optimizer, stats = fit([train], [validation], config, args.device)
    checkpoint = {'model':model.state_dict(), 'optimizer':optimizer.state_dict(),
                  'normalizer':{'mean':norm.mean,'scale':norm.scale}, 'schema':SCHEMA,
                  'config':config, 'stats':stats, 'train_matches':[train['id']],
                  'validation_matches':[validation['id']], 'input_sha256':{
                      train['id']:train_source.provenance['sha256'],
                      validation['id']:validation_source.provenance['sha256']},
                  'source_sha256':{'train_dfl.py':file_hash(__file__),
                                   'train.py':file_hash('src/possession/train.py'),
                                   'dfl_data.py':file_hash('src/possession/dfl_data.py')}}
    dest.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, dest)
    print(json.dumps({'checkpoint_sha256':file_hash(dest), 'stats':stats}))


if __name__ == '__main__': main()
