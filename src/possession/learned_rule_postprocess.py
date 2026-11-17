"""Small learned TCN correction that keeps the kural tabanlı yöntem system as decision engine."""
import numpy as np


def inputs(scores, rule_labels, observed):
    p = np.asarray(scores, dtype=np.float64)
    rule_based = np.asarray(rule_labels)
    obs = np.asarray(observed, dtype=bool)
    if p.ndim != 2 or p.shape[1] != 3 or rule_based.shape != (len(p),) or obs.shape != (len(p),):
        raise ValueError('incompatible postprocessing inputs')
    if (not np.isfinite(p).all() or (p < 0).any() or
            not np.allclose(p.sum(axis=1), 1, atol=1e-5) or not np.isin(rule_based, ['H', 'A', 'D', 'X']).all()):
        raise ValueError('invalid score probabilities or kural tabanlı yöntem labels')
    p = np.clip(p, 1e-6, 1-1e-6)
    t = np.clip(np.log(p[:, 0] / p[:, 1]), -8, 8) / 8
    d = np.clip(np.log(p[:, 2] / (1-p[:, 2])), -8, 8) / 8
    rule_based_team = (rule_based == 'H').astype(float) - (rule_based == 'A').astype(float)
    # No intercept in team head: swapping both teams flips its prediction.
    team_x = np.column_stack((t, rule_based_team, t * obs, rule_based_team * obs))
    dead_x = np.column_stack((d, (rule_based == 'D').astype(float), obs, np.ones(len(p))))
    return team_x, dead_x


def tag_mask(match):
    """Suppress learned additions wherever kural tabanlı yöntem receives an automatic tag event."""
    mask = np.zeros(match.n_frames, dtype=bool)
    for event in match.tags:
        start = max(0, int(event['start_frame']))
        end = min(match.n_frames, int(event['end_frame']))
        if end > start:
            mask[start:end] = True
    return mask


def cues(scores, rule_labels, observed, weights, protected=None):
    team_x, dead_x = inputs(scores, rule_labels, observed)
    tw = np.asarray(weights['team'], dtype=float); dw = np.asarray(weights['dead'], dtype=float)
    if tw.shape != (4,) or dw.shape != (4,) or not np.isfinite(tw).all() or not np.isfinite(dw).all():
        raise ValueError('invalid fitted weights')
    protect = np.zeros(len(team_x), bool) if protected is None else np.asarray(protected, dtype=bool)
    if protect.shape != (len(team_x),): raise ValueError('invalid tag mask')
    team = .5 * np.tanh((team_x @ tw) / 2)
    dead = .5 * np.tanh((dead_x @ dw) / 2)
    team[protect] = 0; dead[protect] = 0
    return team, dead
