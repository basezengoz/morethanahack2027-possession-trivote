"""Compare fixed rule methods with and without target Smart Tagging, without GT.

This audits prediction sensitivity, not correctness or model selection. All artifacts
and aggregate diagnostics are written under data/runs; no target GT is opened.
"""
import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from possession.contracts import TaggingEvidence
from possession.decode import decode_labels
from possession.features import build_features
from possession.io import load_tracking
from possession.predict import apply_published_dead_intervals, load_config, sha256, write_submission
from possession.quality import clean_tracking
from possession.rule_based import dead_ball, model
from possession.rule_based.loader import load_match, load_tracking as load_rule_tracking
from possession.tagging import load_tagging
from possession.teacher import build_evidence


def zero_tags(n):
    return TaggingEvidence(np.zeros((n, 2)), np.zeros(n), np.zeros(n))


def existing_predict(tracking, tags, cfg):
    clean = clean_tracking(tracking, cfg.get('quality', {}), tags.restart)
    features = build_features(clean, cfg.get('features', {}))
    result = decode_labels(build_evidence(features, tags, cfg), features.blocks, cfg)
    return apply_published_dead_intervals(result, features.blocks, cfg.get('published_dead_intervals', True))


def rule_predict(match):
    f = dead_ball.features(match)
    dead = dead_ball.detect_dead(match, dead_ball.dead_evidence(match, f), f)
    return model.build_labels(match, f, dead)[0]


def summarize(on, off, blocks):
    on, off = np.asarray(on), np.asarray(off)
    if on.shape != off.shape or on.ndim != 1 or not np.isin(on, list('HADX')).all() or not np.isin(off, list('HADX')).all():
        raise ValueError('labels must be aligned H/A/D/X vectors')
    n = len(on)
    mask = np.zeros(n, bool)
    previous = 0
    for start, end in blocks:
        if start < previous or end <= start or end > n:
            raise ValueError('invalid blocks')
        mask[start:end] = True
        previous = end

    def counts(labels):
        possessions = 0
        for a, b in blocks:
            span = labels[a:b]
            live = np.isin(span, ['H', 'A'])
            possessions += int(np.sum(live & np.r_[True, span[1:] != span[:-1]]))
        return {'labels': {label: int(np.sum((labels == label) & mask)) for label in 'HADX'},
                'possessions': possessions}

    changes = {}
    for a in 'HAD':
        for b in 'HAD':
            if a != b:
                changes[a + '_to_' + b] = int(np.sum(mask & (on == a) & (off == b)))
    changed = int(np.sum(mask & (on != off)))
    return {'graded_frames': int(mask.sum()), 'changed_frames': changed,
            'changed_fraction': changed / int(mask.sum()) if mask.any() else 0,
            'on_to_off': changes, 'tags_on': counts(on), 'tags_off': counts(off),
            'limitations': ['difference is not accuracy, improvement, or an error estimate',
                            'tag removal affects every tag-dependent module, including cleaning and restarts']}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--method', choices=('existing', 'rule_based'), required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--data-dir', default=None)
    args = p.parse_args()
    out = Path(args.out).resolve()
    if not out.is_relative_to(Path('data/runs').resolve()):
        raise ValueError('private output required')
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_config('configs/rules.yaml')
    tracking = load_tracking(args.data_dir)
    if args.method == 'existing':
        tags = load_tagging(args.data_dir, len(tracking.frame), tracking.fps, cfg.get('tagging', {}))
        on = existing_predict(tracking, tags, cfg)
        off = existing_predict(tracking, zero_tags(len(tracking.frame)), cfg)
    else:
        _, _, ids, _ = load_rule_tracking(data_dir=args.data_dir)
        if not np.array_equal(ids, np.arange(1, len(tracking.frame) + 1)):
            raise ValueError('kural tabanlı yöntem tracking IDs are not exactly ordered')
        m = load_match(data_dir=args.data_dir, with_homography=False)
        on = rule_predict(m)
        off = rule_predict(replace(m, tags=[]))
    on_path, off_path = out / (args.method + '_tags_on.csv'), out / (args.method + '_tags_off.csv')
    write_submission(on_path, on)
    write_submission(off_path, off)
    report = {'method': args.method, 'comparison': summarize(on, off, tracking.blocks),
              'hashes': {'tags_on_csv': sha256(on_path), 'tags_off_csv': sha256(off_path),
                         'script': sha256(Path(__file__)), 'rules_config': sha256('configs/rules.yaml')}}
    (out / (args.method + '.json')).write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report['comparison']))


if __name__ == '__main__':
    main()
