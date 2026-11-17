"""Freeze the protocol's DFL-development candidate before the separate test run."""
import argparse
import json
from pathlib import Path
import numpy as np


def select(reports):
    names = ('existing', 'pass', 'restart', 'pass_restart')
    sources = reports[0]['source_sha256']
    if any(r['source_sha256'] != sources for r in reports):
        raise ValueError('development reports use different source/config hashes')
    rows = {}
    for name in names:
        per_match = [np.mean([r['results'][v][name]['spell_f1']['mean_f1']
                              for v in ('clean', 'masked')]) for r in reports]
        rows[name] = {'per_match_mean_spell': [float(x) for x in per_match],
                      'mean_spell': float(np.mean(per_match))}
    baseline = np.asarray(rows['existing']['per_match_mean_spell'])
    eligible = [name for name in names[1:] if rows[name]['mean_spell'] > rows['existing']['mean_spell']
                and np.all(np.asarray(rows[name]['per_match_mean_spell']) >= baseline - .02)]
    winner = max(eligible, key=lambda x: rows[x]['mean_spell']) if eligible else 'existing'
    return {'selected': winner, 'eligible': eligible, 'development_results': rows,
            'match_ids': [r['match'] for r in reports], 'source_sha256': sources,
            'selection_rule': 'mean spell across clean/masked and matches; no match mean drop > 0.02'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reports', nargs=3, required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    output = Path(args.output).resolve()
    if not output.is_relative_to(Path('data/runs').resolve()) or output.exists():
        raise ValueError('selection output must be a new private file')
    reports = [json.loads(Path(p).read_text(encoding='utf-8')) for p in args.reports]
    if len({r['match'] for r in reports}) != len(reports):
        raise ValueError('development matches must be distinct')
    result = select(reports)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result['development_results'], indent=2))
    print('Frozen candidate:', result['selected'])


if __name__ == '__main__':
    main()
