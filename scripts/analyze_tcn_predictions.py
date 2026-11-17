"""Aggregate decoded confusion and team-swap symmetry on a separate DFL match."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from possession.dfl_data import load_dfl
from possession.train import load_checkpoint, zero_tags
from possession.model import N_RAW, flatten_features, infer_evidence, swap_teams
from possession.predict import load_config, sha256
from possession.quality import clean_tracking
from possession.features import build_features
from possession.teacher import build_evidence
from possession.decode import decode_labels


def swap_metrics(model, norm, raw, blocks, live):
    """Measure raw conditional H/A equivariance, only on reference-live frames."""
    count = disagree = 0
    total_difference = 0.0
    model.eval()
    with torch.inference_mode():
        for start, end in blocks:
            for left in range(start, end, 512):
                right = min(left + 512, end)
                lo, hi = max(start, left-256), min(end, right+256)
                x = raw[lo:hi]
                a = torch.from_numpy(norm.transform(x).T[None])
                swapped_x = swap_teams(x)
                if getattr(model, 'feature_version', 1) == 2:
                    swapped_x[:,N_RAW:N_RAW+2] *= -1
                b = torch.from_numpy(norm.transform(swapped_x).T[None])
                _, original = model(a)
                _, swapped = model(b)
                p = torch.softmax(original[0,:,left-lo:right-lo], dim=0).T.numpy()
                q = torch.softmax(swapped[0,:,left-lo:right-lo], dim=0).T.numpy()[:, ::-1]
                if not np.isfinite(p).all() or not np.isfinite(q).all():
                    raise ValueError('nonfinite model probabilities')
                mask = live[left:right]
                count += int(mask.sum())
                disagree += int(np.count_nonzero(p[mask].argmax(axis=1) != q[mask].argmax(axis=1)))
                total_difference += float(np.abs(p[mask, 0]-q[mask, 0]).sum())
    return {'live_frames':count, 'disagreement_fraction':disagree/count if count else None,
            'mean_abs_home_probability_difference':total_difference/count if count else None}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--positions', required=True); p.add_argument('--info', required=True)
    p.add_argument('--checkpoint', action='append', required=True); p.add_argument('--output', required=True)
    args = p.parse_args()
    out = Path(args.output).resolve()
    if not out.is_relative_to(Path('data/runs').resolve()): raise ValueError('private output required')
    match = load_dfl(args.positions, args.info)
    cfg = load_config('configs/rules.yaml')
    tags = zero_tags(len(match.labels))
    features = build_features(clean_tracking(match.tracking, cfg.get('quality', {}), tags.restart), cfg.get('features', {}))
    base = build_evidence(features, tags, cfg)
    raw = flatten_features(features, tags)
    live = match.supervision_mask & (match.labels < 2)
    results = {}
    for path in args.checkpoint:
        model, norm, ckpt = load_checkpoint(path, 'cpu')
        model_raw = raw
        if getattr(model, 'feature_version', 1) == 2:
            from possession.model_v2 import enrich
            model_raw = enrich(raw, features.blocks, features.fps)
        if Path(path).name in results: raise ValueError('checkpoint names must be unique')
        train_ids = {str(x) for x in ckpt.get('train_matches', [])}
        val_ids = {str(x) for x in ckpt.get('validation_matches', [])}
        match_role = ('train' if match.match_id in train_ids else
                      'validation' if match.match_id in val_ids else 'unlisted')
        evidence = infer_evidence(model, features, tags, base, norm, dead_head_trained=ckpt['stats']['dead_head_trained'])
        prediction = decode_labels(evidence, features.blocks, cfg)
        confusion = {truth:{pred:int(np.count_nonzero(match.supervision_mask & (match.labels == i) & (prediction == pred)))
                            for pred in 'HADX'} for i, truth in enumerate('HAD')}
        both_live = live & np.isin(prediction, ['H','A'])
        mistakes = ((match.labels == 0) & (prediction == 'A')) | ((match.labels == 1) & (prediction == 'H'))
        denominator = int(both_live.sum())
        results[Path(path).name] = {'sha256':sha256(path), 'recorded_match_role':match_role,
            'confusion':confusion,
            'team_error_on_both_live':int(np.count_nonzero(both_live & mistakes))/denominator if denominator else None,
            'both_live_support':denominator, 'team_swap':swap_metrics(model, norm, model_raw, features.blocks, live)}
    report = {'match':match.match_id, 'source':match.provenance, 'config_sha256':sha256('configs/rules.yaml'),
              'script_sha256':sha256(__file__), 'results':results,
              'scope':'DFL validation reference; zero tags; raw head symmetry and decoded confusion'}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(results))


if __name__ == '__main__': main()
