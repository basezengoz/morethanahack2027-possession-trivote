"""Deterministic, independent-block Viterbi decoding of H/A/D evidence."""
import numpy as np

DEFAULTS = {"takeover_cost": 4.0, "stop_cost": 3.0, "restart_cost": 3.0,
            "change_discount": 1.0, "stop_discount": 1.0, "restart_discount": 1.0,
            "score_floor": 1e-8}


def decode_labels(evidence, blocks, config=None):
    """Return one label per frame; blocks are nonoverlapping [start, end).

    Scores are normalized positive weights. Lower indexed states win exact ties.
    Transition costs are soft: no minimum possession duration is imposed.
    """
    cfg = {**DEFAULTS, **(config or {}).get("decode", {})}
    if any(not np.isfinite(v) or v < 0 for v in cfg.values()) or cfg["score_floor"] <= 0:
        raise ValueError("decoder costs must be finite and nonnegative; score_floor positive")
    scores = np.asarray(evidence.state_scores, dtype=float)
    if scores.ndim != 2 or scores.shape[1] != 3:
        raise ValueError("state_scores must have shape (T, 3)")
    n = len(scores)
    if not np.isfinite(scores).all() or (scores < 0).any() or (scores.max(axis=1) <= 0).any():
        raise ValueError("state_scores must be finite, nonnegative and have positive row sums")
    change = np.asarray(evidence.change, dtype=float)
    stop = np.asarray(evidence.stop, dtype=float)
    restart = np.asarray(evidence.restart, dtype=float)
    if change.shape != (n, 2) or stop.shape != (n,) or restart.shape != (n,):
        raise ValueError("transition evidence has incompatible shape")
    if not all(np.isfinite(a).all() for a in (change, stop, restart)):
        raise ValueError("transition evidence must be finite")
    change, stop, restart = (np.clip(a, 0, 1) for a in (change, stop, restart))
    scaled = scores / scores.max(axis=1, keepdims=True)
    emission = -np.log(np.maximum(scaled / scaled.sum(axis=1, keepdims=True), cfg["score_floor"]))
    neutral = ((np.ptp(emission, axis=1) == 0) & (change.max(axis=1) == 0)
               & (stop == 0) & (restart == 0))
    labels = np.full(n, "X", dtype="<U1")
    symbols = np.array(["H", "A", "D"])
    previous_end = 0
    for start, end in blocks:
        if not isinstance(start, (int, np.integer)) or not isinstance(end, (int, np.integer)) or start < previous_end or end <= start or end > n:
            raise ValueError("blocks must be ordered, nonoverlapping, nonempty and within T")
        previous_end = end
        back = np.zeros((end - start, 3), dtype=np.uint8)
        cost = emission[start].copy()
        for t in range(start + 1, end):
            to_h = max(0.0, cfg["takeover_cost"] - cfg["change_discount"] * change[t, 0])
            to_a = max(0.0, cfg["takeover_cost"] - cfg["change_discount"] * change[t, 1])
            to_d = max(0.0, cfg["stop_cost"] - cfg["stop_discount"] * stop[t])
            from_d = max(0.0, cfg["restart_cost"] - cfg["restart_discount"] * restart[t])
            transition = np.array([[0, to_a, to_d], [to_h, 0, to_d], [from_d, from_d, 0]])
            if neutral[t]:
                # No measurement locates a transition inside this frame. Without
                # this, equal-cost backtraces can move a later restart to the very
                # start of a missing-ball interval. Carry the established state.
                transition = np.full((3,3), np.inf)
                np.fill_diagonal(transition, 0)
            candidates = cost[:, None] + transition
            back[t - start] = np.argmin(candidates, axis=0)
            cost = candidates[back[t - start], np.arange(3)] + emission[t]
            cost -= cost.min()
        state = int(np.argmin(cost))
        for i in range(end - start - 1, -1, -1):
            labels[start + i] = symbols[state]
            state = int(back[i, state])
    return labels
