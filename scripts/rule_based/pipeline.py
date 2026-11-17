"""Generate a tactical possession CSV and VLC overlay. Default: rules + TCN v6."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / 'src'):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
from scripts.rule_based import overlay_subs  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--method', choices=('rule_based_tcn', 'rule_based', 'old_rule_based'), default='rule_based_tcn',
                   help='rule_based_tcn: kural tabanlı yöntem + TCN; rule_based: kural tabanlı yöntem; old_rule_based: previous rules')
    p.add_argument('--tcn-model', choices=('v6', 'v5', 'v4'), default='v6')
    p.add_argument('--checkpoint', default=None); p.add_argument('--checkpoint-sha256', default=None)
    p.add_argument('--data-dir', default=None)
    p.add_argument('--output', default=str(ROOT / 'data/runs/final_pipeline/Trivote.csv'))
    p.add_argument('--overlay', default=None, help='ASS output; defaults to tactical video subtitle path')
    args = p.parse_args(argv)
    output = str(Path(args.output).resolve())
    cmd = [sys.executable, '-m', 'possession.predict', '--method', args.method,
           '--tcn-model', args.tcn_model, '--output', output]
    if args.data_dir: cmd += ['--data-dir', str(Path(args.data_dir).resolve())]
    if args.checkpoint: cmd += ['--checkpoint', str(Path(args.checkpoint).resolve())]
    if args.checkpoint_sha256: cmd += ['--checkpoint-sha256', args.checkpoint_sha256]
    subprocess.run(cmd, cwd=ROOT, check=True)
    title = {'rule_based_tcn': 'Kural tabanlı + TCN ' + args.tcn_model,
             'rule_based': 'Kural tabanlı', 'old_rule_based': 'Eski kural tabanlı'}[args.method]
    if args.checkpoint and args.method == 'rule_based_tcn':
        title = 'Kural tabanlı + TCN ' + Path(args.checkpoint).stem
    overlay_args = ['--pred', output, '--title', title]
    if args.data_dir: overlay_args += ['--data-dir', str(Path(args.data_dir).resolve())]
    if args.overlay: overlay_args += ['--out', str(Path(args.overlay).resolve())]
    return overlay_subs.main(overlay_args)


if __name__ == '__main__':
    sys.exit(main())
