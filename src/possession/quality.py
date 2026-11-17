"""Ball cleaning, preserving original observations and half boundaries."""
from dataclasses import dataclass
import numpy as np
from .contracts import TrackingData

@dataclass
class CleanTracking:
    tracking: TrackingData
    ball_xy: np.ndarray  # normalized
    ball_observed: np.ndarray  # original observation mask
    ball_interpolated: np.ndarray
    ball_rejected: np.ndarray
    ball_visual: np.ndarray | None = None
    ball_source: np.ndarray | None = None

    def __post_init__(self):
        n = len(self.ball_xy)
        if self.ball_visual is None:
            self.ball_visual = np.zeros(n, bool)
        if self.ball_source is None:
            self.ball_source = np.full(n, 'missing', dtype='<U64')
            self.ball_source[self.ball_observed & ~self.ball_rejected] = 'atd'
            self.ball_source[self.ball_interpolated] = 'interpolated'


def clean_tracking(tracking, config=None, restart=None, candidates=None):
    config = config or {}
    max_gap = int(config.get('max_gap_frames', 5))
    max_speed = float(config.get('max_ball_speed_mps', 45.0))
    if max_gap < 0 or max_speed <= 0:
        raise ValueError('invalid cleaning thresholds')
    xy = tracking.ball_xy_raw.copy()
    observed = np.asarray(tracking.ball_observed, bool) & np.isfinite(xy).all(axis=1)
    rejected = np.zeros(len(xy), bool)
    interp = np.zeros(len(xy), bool)
    restart = np.zeros(len(xy), bool) if restart is None else np.asarray(restart) > 0
    if restart.shape != (len(xy),):
        raise ValueError('restart shape mismatch')
    scale = np.asarray(tracking.pitch, dtype=np.float32)
    blocks = tracking.blocks or [(0, len(xy))]
    for start, end in blocks:
        # Reject an isolated implausible spike only when its neighbors agree with each other.
        p = xy[start:end] * scale
        if len(p) >= 3:
            dist1 = np.linalg.norm(p[1:-1] - p[:-2], axis=1) * tracking.fps
            dist2 = np.linalg.norm(p[2:] - p[1:-1], axis=1) * tracking.fps
            bridge = np.linalg.norm(p[2:] - p[:-2], axis=1) * tracking.fps / 2
            bad = ((dist1 > max_speed) & (dist2 > max_speed) & (bridge <= max_speed)
                   & observed[start:end-2] & observed[start+1:end-1] & observed[start+2:end])
            rejected[start+1:end-1] = bad
        valid = observed[start:end] & ~rejected[start:end]
        indices = np.flatnonzero(valid) + start
        for left, right in zip(indices[:-1], indices[1:]):
            gap = right - left - 1
            if 0 < gap <= max_gap and not restart[left:right+1].any():
                speed = np.linalg.norm((xy[right] - xy[left]) * scale) * tracking.fps / (gap + 1)
                if speed <= max_speed:
                    alpha = np.arange(1, gap + 1, dtype=np.float32)[:, None] / (gap + 1)
                    xy[left+1:right] = xy[left] + alpha * (xy[right] - xy[left])
                    interp[left+1:right] = True
    xy[~(observed & ~rejected) & ~interp] = np.nan
    clean = CleanTracking(tracking, xy, observed.copy(), interp, rejected)
    return fuse_ball_candidates(clean, candidates, config.get('fusion', {})) if candidates is not None else clean


def load_ball_candidates(path):
    """Read an offline NPZ candidate artifact without pickle or inference dependencies.

    Rows are candidates, not necessarily unique frames. Required arrays: frame (C,),
    pixel_xy (C,2), pitch_xy (C,2), score (C,), observed (C,), projection_valid (C,),
    source (C,). Pitch coordinates are normalized; frame is a zero-based video index.
    Geometry validity is supplied by the producer, including numerical checks.
    Scores are detector scores, not calibrated probabilities. Fusion validates rows.
    """
    with np.load(path, allow_pickle=False) as artifact:
        return {name: artifact[name].copy() for name in artifact.files}


def fuse_ball_candidates(clean, candidates, config=None):
    """Fill unavailable/rejected ATD with corroborated, unambiguous visual candidates.

    Reliable raw ATD always wins. Candidates need a consistent nearby ATD anchor or
    another candidate inside the same half. An isolated high score is insufficient.
    Original tracking, observation/rejection masks and input arrays are never edited.
    Visual points are ground projections; downstream code must not interpret them
    alone as proof that an airborne ball crossed a pitch boundary.
    """
    cfg = {'min_score': .8, 'ambiguity_margin': .1, 'agreement_m': 1.,
           'max_speed_mps': 45., 'context_frames': 5, **(config or {})}
    if (not all(np.isfinite(x) for x in cfg.values()) or not 0 <= cfg['min_score'] <= 1
            or not 0 <= cfg['ambiguity_margin'] <= 1 or cfg['agreement_m'] < 0
            or cfg['max_speed_mps'] <= 0 or cfg['context_frames'] < 1
            or int(cfg['context_frames']) != cfg['context_frames']):
        raise ValueError('invalid candidate fusion thresholds')
    required = ('frame', 'pixel_xy', 'pitch_xy', 'score', 'observed', 'projection_valid', 'source')
    if any(key not in candidates for key in required):
        raise ValueError('missing visual candidate arrays: ' + ', '.join(required))
    arrays = {key: np.asarray(candidates[key]) for key in required}
    frame = arrays['frame']
    if frame.ndim != 1 or frame.dtype.kind not in 'iu':
        raise ValueError('candidate frame must be an integer vector')
    count = len(frame)
    for key, array in arrays.items():
        shape = (count, 2) if key in ('pixel_xy', 'pitch_xy') else (count,)
        if array.shape != shape:
            raise ValueError(f'candidate {key} must have shape {shape}')
    if any(arrays[key].dtype.kind != 'b' for key in ('observed', 'projection_valid')):
        raise ValueError('candidate validity masks must be boolean')
    source = arrays['source']
    if source.dtype.kind != 'U' or np.any(np.isin(source, ['', 'atd', 'missing', 'interpolated'])) or np.any(np.char.str_len(source) > 64):
        raise ValueError('candidate source must be a nonempty visual source name of at most 64 characters')
    n = len(clean.ball_xy)
    if np.any(frame < 0) or np.any(frame >= n):
        raise ValueError('candidate frame outside tracking timeline')
    xy = np.asarray(arrays['pitch_xy'], dtype=np.float64)
    score = np.asarray(arrays['score'], dtype=np.float64)
    valid = (arrays['observed'] & arrays['projection_valid'] & np.isfinite(xy).all(axis=1)
             & np.isfinite(arrays['pixel_xy']).all(axis=1) & np.isfinite(score)
             & (score >= cfg['min_score']) & (score <= 1))
    reliable = clean.ball_observed & ~clean.ball_rejected & np.isfinite(clean.ball_xy).all(axis=1)
    selected = np.full(n, -1, np.int64)
    scale = np.asarray(clean.tracking.pitch)
    # Score order and original row number give reproducible ties.
    rows = np.flatnonzero(valid)
    rows = rows[np.lexsort((rows, -score[rows], frame[rows]))]
    groups = np.split(rows, np.flatnonzero(np.diff(frame[rows])) + 1) if len(rows) else []
    for group in groups:
        best = group[0]; f = int(frame[best])
        if reliable[f]:
            continue
        rivals = group[1:]
        distant = np.linalg.norm((xy[rivals] - xy[best]) * scale, axis=1) > cfg['agreement_m']
        if np.any(distant & (score[best] - score[rivals] < cfg['ambiguity_margin'])):
            continue
        selected[f] = best
    accepted = np.zeros(n, bool)
    context = int(cfg['context_frames'])
    for start, end in clean.tracking.blocks or [(0, n)]:
        anchors = np.flatnonzero(reliable[start:end]) + start
        proposals = np.flatnonzero(selected[start:end] >= 0) + start

        def consistent(f, g, other_xy):
            limit = cfg['agreement_m'] + cfg['max_speed_mps'] * abs(f - g) / clean.tracking.fps
            return np.linalg.norm((xy[selected[f]] - other_xy) * scale) <= limit

        # Reject a candidate conflicting with either nearby raw anchor first.
        viable = []
        anchored = set()
        for f in proposals:
            pos = np.searchsorted(anchors, f)
            neighbors = anchors[max(0, pos-1):pos+1]
            neighbors = neighbors[np.abs(neighbors-f) <= context]
            if all(consistent(f, g, clean.ball_xy[g]) for g in neighbors):
                viable.append(f)
                if len(neighbors):
                    anchored.add(f)
        for i, f in enumerate(viable):
            neighbors = viable[max(0, i-1):i] + viable[i+1:i+2]
            corroborated = any(abs(g-f) <= context and consistent(f, g, xy[selected[g]]) for g in neighbors)
            accepted[f] = f in anchored or corroborated
    ball = clean.ball_xy.copy()
    visual = clean.ball_visual.copy()
    sources = clean.ball_source.copy()
    interpolated = clean.ball_interpolated.copy()
    ball[accepted] = xy[selected[accepted]]
    visual[accepted] = True
    interpolated[accepted] = False
    sources[accepted] = source[selected[accepted]]
    return CleanTracking(clean.tracking, ball, clean.ball_observed.copy(), interpolated,
                         clean.ball_rejected.copy(), visual, sources)
