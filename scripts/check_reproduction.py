"""Run final inference in two independent processes and compare validated CSVs."""
import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def source_bundle(config):
    paths = [ROOT / 'games.py', ROOT / 'pyproject.toml', ROOT / 'uv.lock']
    paths += sorted((ROOT / 'src' / 'possession').glob('*.py'))
    paths += sorted((ROOT / 'src' / 'possession' / 'rule_based').glob('*.py'))
    paths += sorted((ROOT / 'starter').glob('*.py'))
    paths += [ROOT / 'evaluation' / 'validate_submission.py',
              ROOT / 'evaluation' / 'possession_metrics.py', ROOT / 'scripts' / 'predict_rule_based_tcn.py',
              ROOT / 'data/runs/training/dfl_v6.pt',
              ROOT / 'configs/rule_based_tcn_frozen.json',
              Path(__file__).resolve()]
    paths += [Path(config).resolve()]
    return {str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else p.name: sha256(p) for p in paths}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/rules.yaml')
    parser.add_argument('--data-dir', default='data')
    parser.add_argument('--run-dir', default='data/runs/reproduction')
    parser.add_argument('--final-output', default='evaluation/results/Trivote.csv')
    args = parser.parse_args()
    os.chdir(ROOT)
    outdir = Path(args.run_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    before = source_bundle(args.config)
    hashes = []
    manifests = []
    for number in (1, 2):
        output = outdir / f'reproduction_{number}.csv'
        subprocess.run([sys.executable, '-m', 'possession.predict', '--config', args.config,
                        '--data-dir', args.data_dir, '--output', str(output)], check=True)
        subprocess.run([sys.executable, 'evaluation/validate_submission.py', str(output)], check=True)
        hashes.append(sha256(output))
        manifest = Path('data/runs') / f'{output.stem}.manifest.json'
        if not manifest.is_file():
            raise RuntimeError(f'missing inference manifest: {manifest}')
        manifests.append(json.loads(manifest.read_text(encoding='utf-8')))
    for key in ('config', 'inputs', 'source', 'python', 'packages', 'seed', 'device', 'checkpoint', 'experiment', 'method', 'tcn_model'):
        if manifests[0].get(key) != manifests[1].get(key):
            raise RuntimeError(f'inference manifest changed between runs: {key}')
    if before != source_bundle(args.config):
        raise RuntimeError('source bundle changed during reproduction')
    if hashes[0] != hashes[1]:
        raise SystemExit('CSV hashes differ')
    final = Path(args.final_output)
    final.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(outdir / 'reproduction_1.csv', final)
    subprocess.run([sys.executable, 'evaluation/validate_submission.py', str(final)], check=True)
    final_hash = sha256(final)
    if final_hash != hashes[0]:
        raise RuntimeError('final CSV differs from validated reproduction')
    report = {'final_path': str(final), 'final_sha256': final_hash,
              'final_source_manifest': 'data/runs/reproduction_1.manifest.json',
              'source_bundle_sha256': before, 'inference_manifests': manifests,
              'csv_sha256': hashes, 'identical': hashes[0] == hashes[1]}
    (outdir / 'reproduction.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print('CSV SHA256:', *hashes, 'equal:', report['identical'])
    if not report['identical']:
        raise SystemExit('CSV hashes differ')


if __name__ == '__main__':
    main()
