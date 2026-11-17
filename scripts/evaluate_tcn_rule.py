"""Frozen TCN/kural tabanlı yöntem ablations on one DFL match, without target labels or positions."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from possession.dfl_data import load_dfl
from possession.external_evaluation import evaluate_external
from possession.io import load_tracking
from possession.observation_masks import transfer_ball_mask
from possession.predict import load_config, sha256
from possession.quality import clean_tracking
from possession.features import build_features
from possession.teacher import build_evidence
from possession.model import infer_evidence
from possession.train import load_checkpoint, zero_tags
from possession.decode import decode_labels
from possession.rule_based import dead_ball
from possession.rule_fusion import cues_from_rule_features
from possession.tcn_rule_fusion import fuse, keeper_support
from scripts.compare_rule_methods import dfl_as_rule_based_match, dfl_goalkeepers
from scripts.analyze_short_possessions import duration_recall


def predict(tracking, keepers, cfg, model, norm, trained_dead):
    n = len(tracking.frame)
    tags = zero_tags(n)
    features = build_features(clean_tracking(tracking, cfg.get('quality', {}), tags.restart), cfg.get('features', {}))
    base = build_evidence(features, tags, cfg)
    passes = np.zeros((n, 2), np.float32); restart = np.zeros(n, np.float32); keeper = np.zeros((n, 2), np.float32)
    for start, end in tracking.blocks:
        rule_based = dead_ball.features(dfl_as_rule_based_match(tracking, start, end, keepers))
        passes[start:end], restart[start:end] = cues_from_rule_features(rule_based, end-start)
        keeper[start:end] = keeper_support(rule_based, end-start)
    temporal = infer_evidence(model, features, tags, base, norm, dead_head_trained=trained_dead)
    candidates = {'tcn': temporal,
                  'pass': fuse(temporal, passes, restart, keeper, use_pass=True),
                  'restart': fuse(temporal, passes, restart, keeper, use_restart=True),
                  'keeper': fuse(temporal, passes, restart, keeper, use_keeper=True),
                  'combined': fuse(temporal, passes, restart, keeper, use_pass=True, use_restart=True, use_keeper=True),
                  'teacher_d': infer_evidence(model, features, tags, base, norm, dead_head_trained=False)}
    return {name: decode_labels(ev, tracking.blocks, cfg) for name, ev in candidates.items()}, {
        'pass_frames': int(np.any(passes, axis=1).sum()), 'restart_frames': int((restart > 0).sum()),
        'keeper_frames': int(np.any(keeper, axis=1).sum())}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('positions', 'info', 'checkpoint', 'output'): p.add_argument('--' + key, required=True)
    p.add_argument('--pattern-data-dir', default='data'); p.add_argument('--offset', type=int, default=7919)
    args = p.parse_args()
    output = Path(args.output).resolve()
    if not output.is_relative_to(Path('data/runs').resolve()): raise ValueError('private output required')
    model, norm, checkpoint = load_checkpoint(args.checkpoint, 'cpu')
    match = load_dfl(args.positions, args.info)
    if match.match_id in {str(x) for x in checkpoint.get('train_matches', [])}:
        raise ValueError('match used for checkpoint training')
    if not checkpoint['stats']['dead_head_trained']: raise ValueError('trained D head required for this ablation')
    tracking = match.tracking
    keepers = dfl_goalkeepers(args.positions, args.info, tracking)
    pattern = load_tracking(args.pattern_data_dir)
    masked, dropped = transfer_ball_mask(tracking, pattern, offset=args.offset)
    del pattern
    cfg = load_config('configs/rules.yaml')
    predictions, counts = {}, {}
    for view, data in [('clean', tracking), ('masked', masked)]:
        predictions[view], counts[view] = predict(data, keepers, cfg, model, norm, True)
    results = {}
    for view, candidates in predictions.items():
        results[view] = {}
        for name, labels in candidates.items():
            metrics = evaluate_external(match, labels)
            metrics['duration_recall'] = duration_recall(match.labels, labels, match.supervision_mask & match.boundary_mask, tracking.blocks)
            results[view][name] = metrics
    paths = ['scripts/evaluate_tcn_rule.py', 'src/possession/tcn_rule_fusion.py', 'src/possession/rule_fusion.py',
             'src/possession/rule_based/dead_ball.py', 'scripts/compare_rule_methods.py', 'configs/rules.yaml', 'docs/tcn_rule_experiment.md']
    report = {'match': match.match_id, 'reference': match.provenance, 'train_matches': checkpoint.get('train_matches', []),
              'checkpoint_sha256': sha256(args.checkpoint), 'input_sha256': {x: sha256(x) for x in [args.positions, args.info]},
              'source_sha256': {x: sha256(x) for x in paths}, 'mask_sha256': hashlib.sha256(dropped.tobytes()).hexdigest(),
              'mask_offset': args.offset, 'masked_frames': int(dropped.sum()), 'cue_counts': counts, 'results': results,
              'limitations': ['DFL labels and tracking differ from target', 'DFL has no Smart Tagging',
                              'target observation mask only; no target position or label used', 'previously used development data']}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({view: {name: round(m['spell_f1']['mean_f1'], 6) for name, m in items.items()} for view, items in results.items()}))


if __name__ == '__main__': main()
