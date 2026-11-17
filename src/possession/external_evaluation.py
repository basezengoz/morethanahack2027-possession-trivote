"""Metrics for partially supervised external matches; never loads target GT."""
from dataclasses import replace
import numpy as np
from .contracts import TaggingEvidence
from .quality import clean_tracking
from .features import build_features
from .teacher import build_evidence
from .decode import decode_labels


def _runs(mask):
    edges = np.diff(np.r_[False, np.asarray(mask, bool), False].astype(np.int8))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def _ratio(n, d):
    return float(n / d) if d else None


def evaluate_external(match, prediction, tolerance=25):
    """Score known frame interiors; boundary coverage must include positives and negatives.

    Temporal metrics use only fully covered regions. Runs touching a region edge are
    excluded from spell scoring rather than turning a clipped interval into a spell.
    """
    y = np.asarray(match.labels)
    pred = np.asarray(prediction)
    n = len(y)
    supervision = np.asarray(match.supervision_mask, bool)
    boundary = np.asarray(match.boundary_mask, bool)
    if pred.shape != (n,) or supervision.shape != (n,) or boundary.shape != (n,):
        raise ValueError('incompatible external evaluation shapes')
    if tolerance < 0 or not np.isin(y, [-1, 0, 1, 2]).all() or not np.isin(pred, ['H', 'A', 'D', 'X']).all():
        raise ValueError('invalid labels or tolerance')
    if np.any(supervision & (y == -1)):
        raise ValueError('unknown labels cannot be supervised')
    p = np.full(n, 3, np.int8)
    for i, label in enumerate('HAD'):
        p[pred == label] = i
    valid = supervision & (y >= 0)
    f1 = []
    classes = {}
    for i, label in enumerate('HAD'):
        tp = int(np.sum(valid & (y == i) & (p == i)))
        fp = int(np.sum(valid & (y != i) & (p == i)))
        fn = int(np.sum(valid & (y == i) & (p != i)))
        precision, recall = (_ratio(tp, tp + fp), _ratio(tp, tp + fn)) if tp + fn else (None, None)
        score = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
        classes[label] = {'precision': precision, 'recall': recall, 'f1': score, 'support': tp + fn}
        f1.append(score)
    report = {'known_frames': int(valid.sum()), 'macro_f1': float(np.mean(f1)) if valid.any() else None,
              'live_macro_f1': float(np.mean(f1[:2])) if valid.any() else None,
              'macro_f1_definition': 'fixed H/A/D mean; absent classes contribute zero, so without D support the maximum is 2/3',
              'live_macro_f1_definition': 'fixed H/A mean; absent classes contribute zero',
              'classes': classes, 'boundary': None, 'spell_f1': None,
              'temporal_limit': 'requires fully supervised boundary coverage; spells touching coverage edges are excluded'}
    # Never bridge a half or an unknown/unverified gap.
    reliable = valid & boundary
    blocks = match.tracking.blocks or [(0, n)]
    regions = []
    for start, end in blocks:
        regions.extend((start+a, start+b) for a, b in _runs(reliable[start:end]))
    # Official temporal semantics: forward-fill X, then back-fill each region head.
    temporal_pred = p.copy()
    for start, end in regions:
        window = temporal_pred[start:end]
        known = np.flatnonzero(window != 3)
        if len(known):
            carry = np.maximum.accumulate(np.where(window != 3, np.arange(len(window)), -1))
            carry[carry < 0] = known[0]
            temporal_pred[start:end] = window[carry]
    true_boundary, pred_boundary = [], []
    gt_spells, pred_spells = [], []
    for region_id, (start, end) in enumerate(regions):
        # Full tolerance neighbourhood must be observable, including frame t-1.
        lo, hi = start + max(1, tolerance), end - tolerance
        if lo < hi:
            true_boundary.extend((region_id, int(x + lo)) for x in np.flatnonzero(y[lo:hi] != y[lo-1:hi-1]))
            pred_boundary.extend((region_id, int(x + lo)) for x in np.flatnonzero(temporal_pred[lo:hi] != temporal_pred[lo-1:hi-1]))
        for seq, dest in ((y, gt_spells), (temporal_pred, pred_spells)):
            for code in (0, 1):
                for a, b in _runs(seq[start:end] == code):
                    if a > 0 and b < end - start:
                        dest.append((start+a, start+b, code))
    if any(end-start > 2*tolerance+1 for start, end in regions):
        # Ordered two-pointer matching maximizes cardinality at a fixed tolerance.
        hits = 0
        for region_id, _ in enumerate(regions):
            truth = [pos for region, pos in true_boundary if region == region_id]
            predicted = [pos for region, pos in pred_boundary if region == region_id]
            i = j = 0
            while i < len(truth) and j < len(predicted):
                if predicted[j] < truth[i] - tolerance:
                    j += 1
                elif truth[i] < predicted[j] - tolerance:
                    i += 1
                else:
                    hits += 1; i += 1; j += 1
        report['boundary'] = {'tolerance_frames': tolerance, 'tp': hits,
                              'predicted': len(pred_boundary), 'reference': len(true_boundary),
                              'precision': _ratio(hits, len(pred_boundary)), 'recall': _ratio(hits, len(true_boundary))}
    if gt_spells or pred_spells:
        overlaps = sorted(((min(b, d)-max(a, c)) / (max(b, d)-min(a, c)), i, j)
                          for i, (a, b, k) in enumerate(gt_spells)
                          for j, (c, d, l) in enumerate(pred_spells) if k == l and min(b, d) > max(a, c))
        f1s = []
        for threshold in (.3, .4, .5, .6, .7):
            used_g, used_p = set(), set()
            for iou, i, j in reversed(overlaps):
                if iou >= threshold and i not in used_g and j not in used_p:
                    used_g.add(i); used_p.add(j)
            f1s.append(2 * len(used_g) / (len(gt_spells) + len(pred_spells)))
        report['spell_f1'] = {'mean_f1': float(np.mean(f1s)), 'reference': len(gt_spells), 'predicted': len(pred_spells)}
    return report


def mask_ball(tracking, seed=0, fraction=0.1, block_frames=25):
    """Copy raw ball arrays and remove deterministic block samples within periods."""
    if not 0 <= fraction <= 1 or block_frames < 1:
        raise ValueError('invalid dropout parameters')
    observed = np.asarray(tracking.ball_observed, bool)
    drop = np.zeros(len(observed), bool)
    rng = np.random.default_rng(seed)
    for start, end in tracking.blocks or [(0, len(observed))]:
        windows = [(i, min(i + block_frames, end)) for i in range(start, end, block_frames)]
        count = int(round(len(windows) * fraction))
        for idx in rng.choice(len(windows), size=count, replace=False):
            a, b = windows[idx]; drop[a:b] = observed[a:b]
    ball = tracking.ball_xy_raw.copy(); ball[drop] = np.nan
    mask = observed.copy(); mask[drop] = False
    return replace(tracking, ball_xy_raw=ball, ball_observed=mask), drop


def geometry_predict(tracking, config):
    """Run the existing rule-based pipeline without unavailable target-specific tags/policy."""
    n = len(tracking.frame)
    tags = TaggingEvidence(np.zeros((n, 2)), np.zeros(n), np.zeros(n))
    clean = clean_tracking(tracking, config.get('quality', {}), tags.restart)
    features = build_features(clean, config.get('features', {}))
    evidence = build_evidence(features, tags, config)
    return decode_labels(evidence, features.blocks, config), evidence.state_scores


def robustness(tracking, config, seed=0, fraction=0.1, block_frames=25, baseline=None):
    masked, dropped = mask_ball(tracking, seed, fraction, block_frames)
    base, base_state = geometry_predict(tracking, config) if baseline is None else baseline
    altered, altered_state = geometry_predict(masked, config)
    inside = np.zeros(len(base), bool)
    for a, b in tracking.blocks or [(0, len(base))]: inside[a:b] = True
    pair = inside[1:] & inside[:-1]
    for a, _ in (tracking.blocks or [(0, len(base))]):
        if a: pair[a-1] = False
    return {'seed': seed, 'requested_window_fraction': fraction, 'window_frames': block_frames,
            'observed_dropped': int(dropped.sum()), 'label_changed_frames': int(np.sum((base != altered) & inside)),
            'state_mean_absolute_change': float(np.mean(np.abs(base_state[inside] - altered_state[inside]))) if inside.any() else None,
            'boundary_changed_positions': int(np.sum(((base[1:] != base[:-1]) != (altered[1:] != altered[:-1])) & pair))}
