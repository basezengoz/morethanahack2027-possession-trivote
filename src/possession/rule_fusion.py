"""Bounded kural tabanlı yöntem-derived evidence for the existing H/A/D decoder.

Experimental only: callers obtain all cues from tracking and automatic tags, not labels.
"""
from dataclasses import replace
import numpy as np


def combine_evidence(evidence, pass_support, restart_support, *, use_pass=True, use_restart=True):
    """Reweight H/A/D emissions using fixed bounded cues, keeping transitions unchanged.

    pass_support is (T, 2), restart_support is (T,); all values in [0, 1].
    A pass supports its receiving team; a confirmed out/restart cue supports D.
    Neither cue makes a state mandatory. Evidence weights are research priors.
    """
    scores = np.asarray(evidence.state_scores, dtype=np.float64)
    passes = np.asarray(pass_support, dtype=float)
    restart = np.asarray(restart_support, dtype=float)
    if scores.ndim != 2 or scores.shape[1] != 3:
        raise ValueError('state scores must have shape (T, 3)')
    if passes.shape != (len(scores), 2) or restart.shape != (len(scores),):
        raise ValueError('fusion cues have incompatible shape')
    if not np.isfinite(passes).all() or not np.isfinite(restart).all():
        raise ValueError('fusion cues must be finite')
    if np.any((passes < 0) | (passes > 1)) or np.any((restart < 0) | (restart > 1)):
        raise ValueError('fusion cues must be bounded in [0, 1]')
    out = scores.copy()
    if use_pass:
        out[:, :2] *= np.exp(0.5 * passes)
    if use_restart:
        out[:, 2] *= np.exp(0.6 * restart)
    total = out.sum(axis=1, keepdims=True)
    if not np.isfinite(out).all() or np.any(total <= 0):
        raise ValueError('state scores must be finite with positive sums')
    return replace(evidence, state_scores=out / total)


def cues_from_rule_features(features, n, hold_frames=25):
    """Build per-block support from kural tabanlı yöntem tracking-derived pass and out/set-piece cues."""
    if hold_frames < 1:
        raise ValueError('hold_frames must be positive')
    support = np.zeros((n, 2), dtype=np.float32)
    for _, reception, team in features['passes']:
        if 0 <= reception < n and team in (0, 1):
            end = min(n, reception + hold_frames)
            support[reception:end, team] = 1
    # Conflicting pass windows are neutral rather than voting for both sides.
    support[np.all(support > 0, axis=1)] = 0
    out = np.asarray(features['out'], dtype=bool)
    setpiece = np.asarray(features['setpiece'], dtype=float)
    if out.shape != (n,) or setpiece.shape != (n,):
        raise ValueError('kural tabanlı yöntem features have incompatible shape')
    return support, np.maximum(out.astype(np.float32), np.clip(setpiece, 0, 1))
