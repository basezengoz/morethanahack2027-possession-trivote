"""Evaluate a single external Metrica match; no target-match policy or GT."""
import argparse
import json
import subprocess
from pathlib import Path
from possession.external_data import load_metrica_csv
from possession.external_evaluation import evaluate_external, geometry_predict, robustness
from possession.predict import load_config, sha256


ROOT = Path(__file__).resolve().parents[1]


def source_provenance():
    names = ['games.py', 'pyproject.toml', 'uv.lock', 'scripts/evaluate_external.py']
    names += [f'src/possession/{name}.py' for name in
              ('contracts', 'external_data', 'external_evaluation', 'quality', 'features', 'teacher', 'decode', 'predict')]
    hashes = {name: sha256(ROOT / name) for name in names}
    def git(*args):
        try:
            return subprocess.check_output(['git', '-c', f'safe.directory={ROOT.as_posix()}', '-C', str(ROOT), *args],
                                           text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    status = git('status', '--porcelain', '--untracked-files=all')
    return {'files_sha256': hashes, 'git_head': git('rev-parse', 'HEAD'),
            'git_dirty': None if status is None else bool(status)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--match-id', required=True)
    parser.add_argument('--home', required=True)
    parser.add_argument('--away', required=True)
    parser.add_argument('--events', required=True)
    parser.add_argument('--split', required=True, choices=['train', 'validation', 'test'])
    parser.add_argument('--config', default='configs/rules.yaml')
    parser.add_argument('--output', default='data/runs/external/report.json')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--drop-fraction', type=float, default=.1)
    parser.add_argument('--drop-frames', type=int, default=25)
    parser.add_argument('--boundary-tolerance', type=int, default=25)
    args = parser.parse_args(argv)
    source = source_provenance()
    config = load_config(args.config)
    match = load_metrica_csv(args.match_id, args.home, args.away, args.events)
    labels, states = geometry_predict(match.tracking, config)
    report = {
        'match_id': match.match_id, 'split': args.split, 'provenance': match.provenance,
        'config_sha256': sha256(args.config), 'source': source,
        'arguments': vars(args),
        'method': 'Rule-based geometry without external tags and without target-match kickoff policy',
        'rule_based_plus_tags': {'available': False, 'reason': 'external Smart Tagging is unavailable'},
        'A_plus_tags': {'available': False, 'reason': 'external Smart Tagging is unavailable'},
        'proxy_metrics': evaluate_external(match, labels, tolerance=args.boundary_tolerance),
        'robustness_not_accuracy': robustness(match.tracking, config, seed=args.seed,
                                               fraction=args.drop_fraction, block_frames=args.drop_frames,
                                               baseline=(labels, states)),
        'limitations': ['event-derived pass interiors are incomplete proxy supervision, not possession ground truth',
                        'masked-ball prediction changes do not measure accuracy',
                        'test matches must not be used for tuning or model selection'],
    }
    if source['files_sha256'] != source_provenance()['files_sha256']:
        raise RuntimeError('source files changed during evaluation')
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(f'Aggregate external proxy and robustness report written: {output}')


if __name__ == '__main__':
    main()
