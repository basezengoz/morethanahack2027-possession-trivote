"""Count DFL ball-state coverage without exporting match frames."""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET


def audit(path):
    counts = collections.Counter()
    last = None
    for _, element in ET.iterparse(path, events=('end',)):
        a = element.attrib
        if 'BallStatus' in a or 'BallPossession' in a:
            status, possession = a.get('BallStatus', '<missing>'), a.get('BallPossession', '<missing>')
            counts[(status, possession)] += 1
            last = element.tag.split('}')[-1]
        element.clear()
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return {'sha256': digest.hexdigest(), 'element': last,
            'coverage': [{'ball_status': s, 'ball_possession': p, 'frames': n}
                         for (s, p), n in sorted(counts.items())]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to(Path('data/runs').resolve()):
        raise ValueError('output must be under data/runs')
    report = audit(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
