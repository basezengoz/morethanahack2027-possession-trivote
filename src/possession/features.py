"""Metric features computed independently within each half."""
import numpy as np
from .contracts import FeatureData


def build_features(clean, config=None):
    cfg = config or {}
    t = clean.tracking
    n = len(t.frame)
    scale = np.asarray(t.pitch, dtype=np.float32)
    ball = clean.ball_xy * scale
    players = t.players_xy * scale
    available = np.isfinite(ball).all(axis=1)
    distances = np.linalg.norm(players - ball[:, None, :], axis=2)
    distances[~np.asarray(t.player_observed, bool)] = np.nan
    nearest = np.full((n, 2), np.nan, np.float32)
    for team in range(2):
        subset = distances[:, t.slot_team == team]
        if subset.shape[1]:
            mins = np.min(np.where(np.isfinite(subset), subset, np.inf), axis=1)
            nearest[:, team] = np.where(np.isfinite(mins), mins, np.nan)
    proximity = np.nan_to_num(np.exp(-nearest / float(cfg.get('control_distance_m', 3.0))), nan=0.0).astype(np.float32)
    support = proximity.copy()
    top3 = np.full((n, 2, 3), np.nan, np.float32)
    nearby = np.zeros((n, 2), np.int16)
    center = np.full((n, 2, 2), np.nan, np.float32)
    spread = np.full((n, 2), np.nan, np.float32)
    for team in range(2):
        select = t.slot_team == team
        d = distances[:, select]
        if not d.shape[1]:
            continue
        ordered = np.sort(np.where(np.isfinite(d), d, np.inf), axis=1)[:, :3]
        top3[:, team, :ordered.shape[1]] = np.where(np.isfinite(ordered), ordered, np.nan)
        nearby[:, team] = np.sum(d <= float(cfg.get('nearby_radius_m', 5.0)), axis=1)
        pts = players[:, select].copy()
        pts[~t.player_observed[:, select]] = np.nan
        valid = np.isfinite(pts).all(axis=2)
        count = valid.sum(axis=1)
        total = np.nansum(pts, axis=1)
        center[:, team] = np.divide(total, count[:, None], out=np.full((n,2), np.nan), where=count[:, None] > 0)
        radius = np.linalg.norm(pts - center[:, team, None, :], axis=2)
        spread[:, team] = np.divide(np.nansum(radius, axis=1), count, out=np.full(n, np.nan), where=count > 0)
    ball_speed = np.full(n, np.nan, np.float32)
    team_speed = np.full((n, 2), np.nan, np.float32)
    age = np.full(n, -1, np.int32)
    future = np.full(n, -1, np.int32)
    blocks = t.blocks or [(0, n)]
    max_player_speed = float(cfg.get('max_player_speed_mps', 12.0))
    max_ball_speed = float(cfg.get('max_ball_speed_mps', 45.0))
    usable_obs = clean.ball_observed & ~clean.ball_rejected
    for start, end in blocks:
        # Weak two-sided context does not erase short controls; instantaneous signal remains.
        radius = max(0, int(cfg.get('control_context_frames', 5)))
        if radius:
            ix = np.arange(end - start)
            low = np.maximum(0, ix - radius)
            high = np.minimum(end - start, ix + radius + 1)
            cumul = np.vstack((np.zeros((1, 2)), np.cumsum(proximity[start:end], axis=0)))
            mean = (cumul[high] - cumul[low]) / (high - low)[:, None]
            support[start:end] = 0.5 * proximity[start:end] + 0.5 * mean
        bs = np.linalg.norm(np.diff(ball[start:end], axis=0), axis=1) * t.fps
        bs[bs > max_ball_speed] = np.nan
        ball_speed[start+1:end] = bs
        ps = np.linalg.norm(np.diff(players[start:end], axis=0), axis=2) * t.fps
        valid = t.player_observed[start+1:end] & t.player_observed[start:end-1]
        ps[~valid | (ps > max_player_speed)] = np.nan
        for team in range(2):
            subset = ps[:, t.slot_team == team]
            if subset.shape[1]:
                # Median avoids a track reassignment dominating the team's motion.
                ordered = np.sort(np.where(np.isfinite(subset), subset, np.inf), axis=1)
                count = np.isfinite(subset).sum(axis=1)
                rows = np.arange(len(count)); lo = np.maximum(0, (count - 1) // 2); hi = np.maximum(0, count // 2)
                med = (ordered[rows, lo] + ordered[rows, hi]) / 2
                team_speed[start+1:end, team] = np.where(count > 0, med, np.nan)
        seen = np.flatnonzero(usable_obs[start:end]) + start
        if len(seen):
            ix = np.arange(start, end)
            prev = np.searchsorted(seen, ix, side='right') - 1
            nxt = np.searchsorted(seen, ix, side='left')
            has_prev = prev >= 0
            has_next = nxt < len(seen)
            age[ix[has_prev]] = ix[has_prev] - seen[prev[has_prev]]
            future[ix[has_next]] = seen[nxt[has_next]] - ix[has_next]
    boundary = np.min(np.column_stack((ball[:,0], scale[0]-ball[:,0], ball[:,1], scale[1]-ball[:,1])), axis=1)
    # Neighboring control context is not an observation of control on a missing frame.
    support[~available] = 0
    values = dict(nearest_distance=nearest, control_support=support, proximity_support=proximity,
                  team_top3_distance=top3, nearby_player_count=nearby, team_center=center, team_spread=spread, ball_speed=ball_speed,
                  team_speed=team_speed, ball_observed=clean.ball_observed.copy(),
                  ball_interpolated=clean.ball_interpolated.copy(), ball_rejected=clean.ball_rejected.copy(),
                  ball_visual=clean.ball_visual.copy(), ball_source=clean.ball_source.copy(),
                  ball_available=available, gap_age=age, time_to_next_observation=future,
                  boundary_distance=boundary.astype(np.float32), ball_xy=ball.astype(np.float32))
    return FeatureData(values, list(blocks), t.fps)
