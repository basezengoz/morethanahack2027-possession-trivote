"""Cross the two rule systems' dead-ball and live-team decisions on DFL development matches.

Uses no target GT, no DFL labels as input, no tag fabrication, and no target-specific kickoff
override. DFL goalkeeper roster roles are used when present; speed is estimated from tracking.
"""
import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from possession.contracts import TaggingEvidence
from possession.decode import decode_labels
from possession.dfl_data import load_dfl
from possession.external_evaluation import evaluate_external
from possession.features import build_features
from possession.predict import load_config
from possession.quality import clean_tracking
from possession.rule_based import dead_ball, model
from possession.teacher import build_evidence
from scripts.compare_rule_methods import dfl_as_rule_based_match, dfl_goalkeepers


def live_blocks(dead, blocks):
    """Split live spans at stoppages and tracking block boundaries."""
    spans = []
    for start, end in blocks:
        edges = np.diff(np.r_[False, ~dead[start:end], False].astype(np.int8))
        spans.extend((start + int(a), start + int(b))
                     for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))
    return spans


def existing_team_for_rule_based_dead(evidence, dead, blocks, config):
    scores = np.asarray(evidence.state_scores).copy()
    scores[:, 2] = 0
    result = decode_labels(replace(evidence, state_scores=scores), live_blocks(dead, blocks), config)
    for start, end in blocks:
        result[start:end][dead[start:end]] = 'D'
    if np.any((result != 'H') & (result != 'A') & (result != 'D') & (result != 'X')):
        raise ValueError('invalid resulting label')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--positions', required=True)
    p.add_argument('--info', required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    out = Path(args.out).resolve()
    if not out.is_relative_to(Path('data/runs').resolve()):
        raise ValueError('private output required')
    match = load_dfl(args.positions, args.info)
    t = match.tracking
    cfg = load_config('configs/rules.yaml')
    tags = TaggingEvidence(np.zeros((len(t.frame), 2)), np.zeros(len(t.frame)), np.zeros(len(t.frame)))
    clean = clean_tracking(t, cfg.get('quality', {}), tags.restart)
    features = build_features(clean, cfg.get('features', {}))
    evidence = build_evidence(features, tags, cfg)
    existing = decode_labels(evidence, features.blocks, cfg)
    goalie = dfl_goalkeepers(args.positions, args.info, t)
    rule_based = np.full(len(t.frame), 'X', dtype='<U1')
    rule_based_team_existing_dead = rule_based.copy()
    for start, end in t.blocks:
        m = dfl_as_rule_based_match(t, start, end, goalie)
        f = dead_ball.features(m)
        llr = dead_ball.dead_evidence(m, f)
        d = dead_ball.detect_dead(m, llr, f)
        rule_based[start:end] = model.build_labels(m, f, d)[0]
        rule_based_team_existing_dead[start:end] = model.build_labels(m, f, existing[start:end] == 'D')[0]
    rule_based_dead = rule_based == 'D'
    existing_team_rule_based_dead = existing_team_for_rule_based_dead(evidence, rule_based_dead, t.blocks, cfg)
    if not np.array_equal(rule_based_team_existing_dead == 'D', existing == 'D'):
        raise RuntimeError('kural tabanlı yöntem team decoder did not preserve existing dead mask')
    if not np.array_equal(existing_team_rule_based_dead == 'D', rule_based_dead):
        raise RuntimeError('existing team decoder did not preserve kural tabanlı yöntem dead mask')
    predictions = {'existing': existing, 'rule_based': rule_based,
                   'rule_based_dead_existing_team': existing_team_rule_based_dead,
                   'existing_dead_rule_based_team': rule_based_team_existing_dead}
    report = {'match': match.match_id, 'provenance': match.provenance,
              'limitations': ['Development match only; not a held-out test.',
                              'No Smart Tagging; roster goalkeeper roles and derived speeds.',
                              'Different tracking domain, DFL reference semantics and target domain.'],
              'results': {name: evaluate_external(match, pred) for name, pred in predictions.items()}}
    out.mkdir(parents=True, exist_ok=True)
    (out / (match.match_id + '.json')).write_text(json.dumps(report, indent=2), encoding='utf-8')
    for name, metrics in report['results'].items():
        print(name, 'spell', metrics['spell_f1'], 'macro', metrics['macro_f1'],
              'D_F1', metrics['classes']['D']['f1'])


if __name__ == '__main__':
    main()
