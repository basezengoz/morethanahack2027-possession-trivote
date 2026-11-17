"""Experimental bounded kural tabanlı yöntem evidence applied to TCN emissions."""
from dataclasses import replace
import numpy as np


def fuse(evidence, passes, restart, keeper, *, use_pass=False, use_restart=False, use_keeper=False):
    scores = np.asarray(evidence.state_scores)
    n = len(scores)
    arrays = [np.asarray(x, dtype=float) for x in (passes, restart, keeper)]
    if scores.shape != (n, 3) or [x.shape for x in arrays] != [(n, 2), (n,), (n, 2)]:
        raise ValueError('incompatible fusion evidence shapes')
    if not np.isfinite(scores).all() or np.any(scores < 0) or np.any(scores.sum(axis=1) <= 0):
        raise ValueError('invalid TCN scores')
    if any(not np.isfinite(x).all() or np.any((x < 0) | (x > 1)) for x in arrays):
        raise ValueError('fusion support must be finite and bounded')
    log_weight = np.zeros((n, 3), dtype=float)
    if use_pass:
        log_weight[:, :2] += .5 * arrays[0]
    if use_restart:
        log_weight[:, 2] += .6 * arrays[1]
    if use_keeper:
        log_weight[:, :2] += .5 * arrays[2]
    active = np.any(log_weight != 0, axis=1)
    out = scores.astype(np.float64, copy=True)
    out[active] *= np.exp(log_weight[active])
    out[active] /= out[active].sum(axis=1, keepdims=True)
    return replace(evidence, state_scores=out)


def keeper_support(features, n):
    """kural tabanlı yöntem's goalkeeper possession inference is team support, never a live label."""
    mask = np.asarray(features['keeper_poss'], dtype=bool)
    team = np.asarray(features['keeper_team'])
    if mask.shape != (n,) or team.shape != (n,) or np.any(mask & ~np.isin(team, [0, 1])):
        raise ValueError('invalid keeper features')
    result = np.zeros((n, 2), np.float32)
    for side in (0, 1):
        result[:, side] = mask & (team == side)
    return result
