"""Audit external proxy coverage without printing match events or frame labels."""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

from possession.external_data import load_metrica_csv
from possession.predict import sha256


def audit_match(root, game):
    prefix = f'Sample_Game_{game}'
    folder = root / prefix
    home = folder / f'{prefix}_RawTrackingData_Home_Team.csv'
    away = folder / f'{prefix}_RawTrackingData_Away_Team.csv'
    events = folder / f'{prefix}_RawEventsData.csv'
    match = load_metrica_csv(str(game), home, away, events)
    with events.open(encoding='utf-8-sig', newline='') as stream:
        event_types = Counter((row.get('Type') or '').strip().upper() for row in csv.DictReader(stream))
    mask = np.asarray(match.supervision_mask, bool)
    labels = np.asarray(match.labels)
    boundary = np.asarray(match.boundary_mask, bool)
    if np.any(boundary & ~mask):
        raise ValueError('boundary coverage outside state supervision')
    counts = {key: int(np.count_nonzero(mask & (labels == code)))
              for key, code in (('H', 0), ('A', 1), ('D', 2))}
    return {'match_id': str(game), 'frames': len(labels), 'supervised_state_frames': int(mask.sum()),
            'class_support': counts, 'reliable_boundary_frames': int(boundary.sum()),
            'boundary_metric_available': bool(boundary.any()),
            'spell_metric_available': False,
            'event_type_counts': dict(sorted(event_types.items())),
            'input_sha256': {name: sha256(path) for name, path in
                             (('home', home), ('away', away), ('events', events))},
            'interpretation': 'event counts and PASS interior labels are not complete possession ground truth; spell metric needs independently verified complete intervals'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', default='data/external/metrica-sample-data')
    parser.add_argument('--output', default='data/runs/external/supervision_audit.json')
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if not output.is_relative_to(Path('data/runs').resolve()):
        raise ValueError('audit output must be private under data/runs')
    report = {'matches': [audit_match(Path(args.data_dir), game) for game in (1, 2)],
              'decision': 'D and possession boundary supervision unavailable from the current conservative proxy; do not select a final model with H/A pass interiors alone',
              'source_sha256': sha256(__file__)}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True)+'\n', encoding='utf-8')
    print(json.dumps({'output': str(output), 'matches': [
        {key: item[key] for key in ('match_id','class_support','reliable_boundary_frames')}
        for item in report['matches']]}))


if __name__ == '__main__':
    main()
