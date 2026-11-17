"""Bounded TCN evidence for the kural tabanlı yöntem method's own state decoders."""
import numpy as np


def tcn_cues(scores, observed):
    """Return signed H/A evidence and D log odds; missing raw ball gives zero.

    TCN outputs are normalized H/A/D probabilities, not ground truth.  Saturation
    bounds overconfident predictions; the H/A cue is discounted by P(D).
    """
    p = np.asarray(scores, dtype=float)
    observed = np.asarray(observed, dtype=bool)
    if p.ndim != 2 or p.shape[1] != 3 or observed.shape != (len(p),):
        raise ValueError('incompatible scores and observation mask')
    if not np.isfinite(p).all() or np.any(p < 0) or not np.allclose(p.sum(axis=1), 1, atol=1e-5):
        raise ValueError('scores must be normalized finite probabilities')
    clipped = np.clip(p, 1e-6, 1)
    team = .25 * np.clip(np.log(clipped[:, 0] / clipped[:, 1]), -2, 2) * (1-p[:, 2])
    dead = .25 * np.clip(np.log(clipped[:, 2] / np.maximum(1-p[:, 2], 1e-6)), -2, 2)
    return np.where(observed, team, 0.), np.where(observed, dead, 0.)
