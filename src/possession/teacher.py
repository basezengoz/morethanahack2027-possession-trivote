"""Weak tracking and tagging evidence; scores are normalized weights, not probabilities."""
import numpy as np

from .contracts import Evidence

DEFAULTS = {
    "distance_scale_m": 3.0, "distance_weight": 1.5,
    "control_weight": 1.0, "tag_weight": 0.35,
    "dead_bias": -2.5, "dead_weight": 5.0,
    "slow_ball_mps": 1.0, "slow_team_mps": 1.0,
    "boundary_margin_m": 2.0,
}


def build_evidence(features, tags, config=None):
    """Build H/A/D weights and separate transition support from frame-aligned features.

    Missing ball measurements are neutral for dead/alive, allowing an established D
    to survive an occlusion. A restart tag, a slow team or a missing ball alone cannot
    establish D. Restart preparation and actual resumption are separate evidence.
    """
    cfg = {**DEFAULTS, **(config or {}).get("teacher", {})}
    if not all(np.isfinite(x) for x in cfg.values()) or any(cfg[k] <= 0 for k in ("distance_scale_m", "slow_ball_mps", "slow_team_mps", "boundary_margin_m")):
        raise ValueError("teacher distance, speed and boundary scales must be positive")
    v = features.values
    distance = np.asarray(v["nearest_distance"], dtype=float)
    if distance.ndim != 2 or distance.shape[1] != 2:
        raise ValueError("nearest_distance must have shape (T, 2)")
    n = len(distance)

    def array(name, shape):
        a = np.asarray(v[name], dtype=float)
        if a.shape != shape:
            raise ValueError(f"{name} must have shape {shape}")
        return a

    available = array("ball_available", (n,)).astype(bool)
    control = np.clip(np.nan_to_num(array("control_support", (n, 2)), nan=0.0), 0, 1)
    ball_speed = array("ball_speed", (n,))
    team_speed = array("team_speed", (n, 2))
    boundary = array("boundary_distance", (n,))
    visual = np.asarray(v.get('ball_visual', np.zeros(n, bool)), dtype=bool)
    if visual.shape != (n,):
        raise ValueError('ball_visual must have shape (T,)')
    team = np.asarray(tags.team_support, dtype=float)
    restart = np.asarray(tags.restart, dtype=float)
    conflict = np.asarray(tags.conflict, dtype=float)
    if team.shape != (n, 2) or restart.shape != (n,) or conflict.shape != (n,):
        raise ValueError("tag evidence has incompatible shape")
    team = np.clip(np.nan_to_num(team, nan=0.0), 0, 1)
    restart = np.clip(np.nan_to_num(restart, nan=0.0), 0, 1)
    conflict = np.clip(np.nan_to_num(conflict, nan=0.0), 0, 1)
    proximity = np.exp(-np.maximum(np.nan_to_num(distance, nan=np.inf), 0) / cfg["distance_scale_m"])
    proximity *= available[:, None]
    control *= available[:, None]
    live = cfg["distance_weight"] * proximity + cfg["control_weight"] * control
    live += cfg["tag_weight"] * team * (1 - conflict[:, None])
    slow_ball = np.where(np.isfinite(ball_speed) & available,
                         np.clip(1 - np.nan_to_num(ball_speed, nan=np.inf) / cfg["slow_ball_mps"], 0, 1), 0)
    # Both teams must have measured, low motion for the motion-based cue.
    finite_speeds = np.isfinite(team_speed).all(axis=1)
    max_speed = np.max(np.where(np.isfinite(team_speed), team_speed, np.inf), axis=1)
    slow_teams = np.where(finite_speeds, np.clip(1 - max_speed / cfg["slow_team_mps"], 0, 1), 0)
    # Visual ground projections of airborne balls are not reliable line crossings.
    boundary_available = np.isfinite(boundary) & available & ~visual
    near_boundary = np.where(boundary_available,
                             np.clip(1 - np.nan_to_num(boundary, nan=np.inf) / cfg["boundary_margin_m"], 0, 1), 0)
    outside = np.where(boundary_available,
                       np.clip(-np.nan_to_num(boundary, nan=0) / cfg['boundary_margin_m'], 0, 1), 0)
    # A stationary ball in a restart window can be at the taker's feet while the
    # other players still move. A rolling outside ball needs restart/collective cues.
    stop = np.maximum(slow_ball * np.maximum(near_boundary, restart),
                      outside * np.maximum(restart, slow_teams))
    # Confirmed preparation suppresses competing proximity to the restart taker.
    live *= (1 - stop[:, None])
    dead = cfg["dead_bias"] + cfg["dead_weight"] * stop
    # Previously every missing frame applied dead_bias again, immediately erasing
    # a supported stoppage. Without team tags all three states are now neutral;
    # with tags, D never gains an advantage over either supported live state.
    dead = np.where(available, dead, live.min(axis=1))
    logits = np.column_stack((live, dead))
    logits -= logits.max(axis=1, keepdims=True)
    scores = np.exp(logits)
    scores /= scores.sum(axis=1, keepdims=True)
    motion = np.where(np.isfinite(ball_speed) & available,
                      np.clip(np.nan_to_num(ball_speed, nan=0) / cfg['slow_ball_mps'], 0, 1), 0)
    resumption = restart * motion * control.max(axis=1) * (1-stop)
    return Evidence(scores.astype(np.float32), (proximity * control).astype(np.float32),
                    stop.astype(np.float32), resumption.astype(np.float32))
