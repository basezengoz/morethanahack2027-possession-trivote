"""Score fixed models against a DFL match; no target-match data are read."""
import argparse
import json
from pathlib import Path
import numpy as np
from possession.dfl_data import load_dfl
from possession.external_evaluation import evaluate_external
from possession.train import load_checkpoint, zero_tags
from possession.predict import load_config, sha256
from possession.quality import clean_tracking
from possession.features import build_features
from possession.teacher import build_evidence
from possession.decode import decode_labels
from possession.model import infer_evidence
from starter.baseline import smooth


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--positions', required=True)
    p.add_argument('--info', required=True)
    p.add_argument('--checkpoint', action='append', default=[])
    p.add_argument('--output', required=True)
    args = p.parse_args()
    output = Path(args.output).resolve()
    if not output.is_relative_to(Path('data/runs').resolve()): raise ValueError('private output required')
    match = load_dfl(args.positions, args.info)
    config = load_config('configs/rules.yaml')
    tags = zero_tags(len(match.labels))
    clean = clean_tracking(match.tracking, config.get('quality', {}), tags.restart)
    features = build_features(clean, config.get('features', {}))
    baseline = build_evidence(features, tags, config)
    pred = decode_labels(baseline, features.blocks, config)
    results = {'rule_geometry': evaluate_external(match, pred)}
    tracking = match.tracking
    distance = ((tracking.players_xy - tracking.ball_xy_raw[:, None, :]) * np.asarray(tracking.pitch)) ** 2
    distance = np.where(np.isfinite(distance).all(axis=2), distance.sum(axis=2), np.inf)
    nearest = distance.argmin(axis=1)
    codes = np.where(np.isfinite(distance.min(axis=1)), tracking.slot_team[nearest], -1)
    smooth_labels = np.full(len(codes), 'D', dtype='<U1')
    for start, end in tracking.blocks:
        smooth_labels[start:end] = smooth(codes[start:end], 50)
    results['baseline_smooth50'] = evaluate_external(match, smooth_labels)
    for path in args.checkpoint:
        model, norm, ckpt = load_checkpoint(path, 'cpu')
        temporal = infer_evidence(model, features, tags, baseline, norm,
                                   dead_head_trained=ckpt['stats']['dead_head_trained'])
        pred = decode_labels(temporal, features.blocks, config)
        results[Path(path).name] = {'checkpoint_sha256': sha256(path), **evaluate_external(match, pred)}
    report = {'match':match.match_id, 'provenance':match.provenance,
              'blocks':len(match.tracking.blocks), 'results':results,
              'limitations':['DFL possession semantics and tracking domain differ from target',
                             'complete spells at coverage edges excluded',
                             'no target match smart tags available for DFL']}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k:{'spell_f1':v['spell_f1'], 'macro_f1':v['macro_f1']} for k,v in results.items()}))


if __name__ == '__main__': main()
