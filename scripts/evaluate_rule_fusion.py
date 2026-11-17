"""Compare frozen rule candidates on one DFL match, clean and target-mask conditions.

DFL labels are read only for evaluation after predictions are produced. The target provides
only its ball-observation mask, not labels or coordinates to the DFL prediction.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from possession.contracts import TaggingEvidence
from possession.decode import decode_labels
from possession.dfl_data import load_dfl
from possession.external_evaluation import evaluate_external
from possession.features import build_features
from possession.io import load_tracking
from possession.observation_masks import transfer_ball_mask
from possession.predict import load_config, sha256
from possession.quality import clean_tracking
from possession.rule_based import dead_ball, model
from possession.rule_fusion import combine_evidence, cues_from_rule_features
from possession.teacher import build_evidence
from scripts.compare_rule_methods import dfl_as_rule_based_match, dfl_goalkeepers
from scripts.analyze_short_possessions import duration_recall


def predict(tracking, goalie, cfg):
    n = len(tracking.frame)
    tags = TaggingEvidence(np.zeros((n, 2)), np.zeros(n), np.zeros(n))
    clean = clean_tracking(tracking, cfg.get('quality', {}), tags.restart)
    features = build_features(clean, cfg.get('features', {}))
    base = build_evidence(features, tags, cfg)
    passes = np.zeros((n, 2), np.float32)
    restart = np.zeros(n, np.float32)
    rule_based = np.full(n, 'X', dtype='<U1')
    for start, end in tracking.blocks:
        m = dfl_as_rule_based_match(tracking, start, end, goalie)
        f = dead_ball.features(m)
        passes[start:end], restart[start:end] = cues_from_rule_features(f, end - start)
        d = dead_ball.detect_dead(m, dead_ball.dead_evidence(m, f), f)
        rule_based[start:end] = model.build_labels(m, f, d)[0]
    predictions = {'existing': decode_labels(base, tracking.blocks, cfg), 'rule_based': rule_based}
    for name, pass_on, restart_on in [('pass', True, False), ('restart', False, True),
                                      ('pass_restart', True, True)]:
        ev = combine_evidence(base, passes, restart, use_pass=pass_on, use_restart=restart_on)
        predictions[name] = decode_labels(ev, tracking.blocks, cfg)
    return predictions, {'pass_support_frames': int(np.any(passes > 0, axis=1).sum()),
                         'restart_support_frames': int((restart > 0).sum())}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('positions', 'info', 'output'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--pattern-data-dir', default='data')
    p.add_argument('--offset', type=int, default=7919)
    args = p.parse_args()
    output = Path(args.output).resolve()
    if not output.is_relative_to(Path('data/runs').resolve()):
        raise ValueError('private output required')
    match = load_dfl(args.positions, args.info)
    t = match.tracking
    goalies = dfl_goalkeepers(args.positions, args.info, t)
    pattern = load_tracking(args.pattern_data_dir)
    masked, dropped = transfer_ball_mask(t, pattern, offset=args.offset)
    del pattern
    cfg = load_config('configs/rules.yaml')
    # All predictions are fixed before any label is passed to a scoring function.
    predictions = {}
    stats = {}
    for name, tracking in [('clean', t), ('masked', masked)]:
        predictions[name], stats[name] = predict(tracking, goalies, cfg)
    results = {}
    for view, models in predictions.items():
        results[view] = {}
        for name, labels in models.items():
            metrics = evaluate_external(match, labels)
            metrics['duration_recall'] = duration_recall(match.labels, labels,
                match.supervision_mask & match.boundary_mask, t.blocks)
            results[view][name] = metrics
    sources = ['scripts/evaluate_rule_fusion.py', 'scripts/compare_rule_methods.py',
               'src/possession/rule_fusion.py', 'src/possession/rule_based/model.py',
               'src/possession/rule_based/dead_ball.py', 'configs/rules.yaml']
    report = {'match': match.match_id, 'reference': match.provenance,
              'source_sha256': {path: sha256(path) for path in sources},
              'observation_pattern': {'offset': args.offset, 'masked_frames': int(dropped.sum()),
                                      'mask_sha256': hashlib.sha256(dropped.tobytes()).hexdigest()},
              'cue_counts': stats, 'results': results,
              'limitations': ['DFL label semantics and tracking differ from target',
                              'no DFL Smart Tagging; player speed derived from coordinates',
                              'target ball availability transferred without coordinates or labels',
                              'mask experiment tests robustness, not target accuracy']}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({view: {k: {'spell': v['spell_f1']['mean_f1'],
                                 'macro': v['macro_f1'], 'D_f1': v['classes']['D']['f1'],
                                 'poss': v['spell_f1']['predicted']}
                            for k, v in models.items()} for view, models in results.items()}))


if __name__ == '__main__':
    main()
