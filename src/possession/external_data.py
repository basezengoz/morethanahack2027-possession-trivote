"""Metrica sample CSVs and explicitly incomplete event-derived proxy supervision.

No event interval is ground-truth possession. Unlabelled frames and uncertain
boundaries are excluded; no event-gap is forward filled. Source files remain local.
"""
from dataclasses import dataclass
from pathlib import Path
import csv
import hashlib
import math
import numpy as np
from .contracts import TrackingData

UNKNOWN = -1

@dataclass
class ExternalMatch:
    match_id: str
    tracking: TrackingData
    labels: np.ndarray
    supervision_mask: np.ndarray
    boundary_mask: np.ndarray
    provenance: dict


def split_match_ids(match_ids, validation_ids):
    """Split whole, uniquely named matches; never split windows from one match."""
    ids = list(match_ids); val = set(validation_ids)
    if not ids or len(set(ids)) != len(ids) or not val or not val.issubset(ids):
        raise ValueError('match IDs must be unique and validation IDs nonempty and known')
    train = [x for x in ids if x not in val]
    validation = [x for x in ids if x in val]
    if not train:
        raise ValueError('training and validation both require at least one match')
    return train, validation


def _float(cell):
    if not cell.strip() or cell.strip().casefold() == 'nan':
        return np.nan
    result = float(cell)
    if not math.isfinite(result):
        raise ValueError('tracking coordinates must be finite or empty')
    if not -0.5 <= result <= 1.5:
        raise ValueError('tracking coordinates must use normalized pitch units')
    return result


def _tracking_rows(path):
    with open(path, newline='', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        team_header = next(reader); identity_header = next(reader); axis_header = next(reader)
        if len(axis_header) < 5 or len(axis_header[3:]) % 2:
            raise ValueError('tracking header must have paired x/y slots')
        if [x.strip().casefold() for x in axis_header[:3]] != ['period', 'frame', 'time [s]']:
            # The documented format places these names in the second header row.
            if [x.strip().casefold() for x in identity_header[:3]] != ['period', 'frame', 'time [s]']:
                raise ValueError('unrecognized Metrica tracking header')
        axes = [x.strip().casefold() for x in axis_header[3:]]
        pairs = [axes[i:i+2] for i in range(0,len(axes),2)]
        if all(pair == ['x', 'y'] for pair in pairs):
            names = [x.strip().casefold() for x in identity_header[3::2]]
        elif all((pair[0].startswith('player') or pair[0] == 'ball') and not pair[1] for pair in pairs):
            names = [pair[0] for pair in pairs]
        else:
            raise ValueError('unrecognized paired tracking slots')
        ball_indices = [i for i, name in enumerate(names) if name == 'ball']
        if len(ball_indices) != 1:
            raise ValueError('expected one ball slot')
        ball_idx = ball_indices[0]
        rows = []
        for row in reader:
            if len(row) != len(axis_header):
                raise ValueError('tracking row width mismatch')
            period, frame, timestamp = int(row[0]), int(row[1]), float(row[2])
            if period < 1 or frame < 0 or not math.isfinite(timestamp):
                raise ValueError('invalid tracking index or timestamp')
            vals = np.asarray([_float(v) for v in row[3:]], dtype=np.float32).reshape(-1,2)
            if np.any(np.isnan(vals[:,0]) != np.isnan(vals[:,1])):
                raise ValueError('incomplete tracking coordinate pair')
            rows.append((period,frame,timestamp, np.delete(vals, ball_idx, axis=0), vals[ball_idx]))
        if not rows:
            raise ValueError('empty tracking file')
        return rows


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def proxy_from_events(events, frame_lookup, n_frames, fps, boundary_margin_frames=5, max_event_frames=100):
    """Label interior of short, directed completed pass events only.

    Proxy: even a completed pass does not establish continuous possession. Boundary
    confidence is unavailable, so the separate boundary evaluation mask is all false.
    Conflicting team labels and event boundaries are excluded regardless of order.
    """
    if boundary_margin_frames < 0 or max_event_frames <= 0:
        raise ValueError('invalid proxy parameters')
    votes = np.zeros((n_frames,2), dtype=bool)
    uncertain = np.zeros(n_frames, dtype=bool)
    for event in events:
        try:
            period = int(event['Period']); start_frame = int(event['Start Frame']); end_frame = int(event['End Frame'])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError('invalid event index') from exc
        if end_frame < start_frame:
            # Malformed source events cannot supervise their ambiguous span.
            a = frame_lookup.get((period,end_frame)); b = frame_lookup.get((period,start_frame))
            if a is not None and b is not None and a <= b:
                uncertain[max(0,a-boundary_margin_frames):min(n_frames,b+boundary_margin_frames+1)] = True
            continue
        if (period,start_frame) not in frame_lookup or (period,end_frame) not in frame_lookup:
            continue
        start = frame_lookup[(period,start_frame)]; end = frame_lookup[(period,end_frame)] + 1
        if end <= start or end-start != end_frame-start_frame+1:
            continue
        margin = boundary_margin_frames
        if margin:
            uncertain[max(0,start-margin):min(n_frames,start+margin+1)] = True
            uncertain[max(0,end-margin):min(n_frames,end+margin)] = True
        code = (event.get('Type') or '').strip().casefold()
        subtype = (event.get('Subtype') or '').strip().casefold()
        team = (event.get('Team') or '').strip().casefold()
        receiver = (event.get('To') or '').strip()
        team_index = {'home':0, 'away':1}.get(team)
        if code != 'pass' or team_index is None or not receiver or end-start > max_event_frames:
            continue
        if any(x in subtype for x in ('lost', 'out', 'intercept', 'unsuccessful', 'incomplete')):
            continue
        lo, hi = start + margin, end - margin
        if lo < hi:
            votes[lo:hi,team_index] = True
    valid = (votes.sum(axis=1) == 1) & ~uncertain
    labels = np.full(n_frames, UNKNOWN, np.int8)
    labels[valid] = votes[valid].argmax(axis=1).astype(np.int8)
    return labels, valid, np.zeros(n_frames, dtype=bool)


def load_metrica_csv(match_id, home_path, away_path, events_path, fps=25.0,
                     boundary_margin_frames=5, max_event_frames=100):
    """Load a Metrica Game 1/2 match and conservative pass-interior proxies."""
    if not match_id or fps <= 0:
        raise ValueError('match ID and positive fps required')
    home = _tracking_rows(home_path); away = _tracking_rows(away_path)
    if len(home) != len(away):
        raise ValueError('home/away tracking length differs')
    n = len(home); blocks = []; start = 0; lookup = {}
    for i,(h,a) in enumerate(zip(home,away)):
        if h[:2] != a[:2] or not math.isclose(h[2], a[2], abs_tol=1e-5):
            raise ValueError('home/away timeline mismatch')
        key = h[:2]
        if key in lookup:
            raise ValueError('duplicate tracking frame')
        if i:
            prior = home[i-1]
            if h[0] < prior[0] or (h[0] == prior[0] and h[1] <= prior[1]):
                raise ValueError('tracking frame sequence out of order')
            if h[0] == prior[0] and abs((h[2] - prior[2]) - (h[1] - prior[1]) / fps) > 0.01:
                raise ValueError('tracking timestamps disagree with fps')
            if h[0] != prior[0] or h[1] != prior[1]+1:
                blocks.append((start,i)); start=i
        lookup[key] = i
    blocks.append((start,n))
    home_p = np.stack([row[3] for row in home]); away_p = np.stack([row[3] for row in away])
    ball_h = np.stack([row[4] for row in home]); ball_a = np.stack([row[4] for row in away])
    both = np.isfinite(ball_h).all(axis=1) & np.isfinite(ball_a).all(axis=1)
    if not np.allclose(ball_h[both], ball_a[both], atol=1e-5, rtol=0):
        raise ValueError('home and away ball observations disagree')
    ball = np.where(np.isfinite(ball_h), ball_h, ball_a).astype(np.float32)
    players = np.concatenate((home_p, away_p), axis=1).astype(np.float32)
    tracking = TrackingData(np.arange(n,dtype=np.int32), players, ball,
        np.isfinite(players).all(axis=2), np.isfinite(ball).all(axis=1),
        np.array([0]*home_p.shape[1]+[1]*away_p.shape[1],dtype=np.int8), fps, (105.0,68.0), blocks)
    with open(events_path, newline='', encoding='utf-8-sig') as f:
        events = list(csv.DictReader(f))
    labels, mask, boundary = proxy_from_events(events, lookup, n, fps, boundary_margin_frames, max_event_frames)
    provenance = {'source':'Metrica Sports sample-data', 'url':'https://github.com/metrica-sports/sample-data',
                  'license':'README requests attribution and responsible use; standard redistribution and derived checkpoint terms unspecified',
                  'label_kind':'event-derived proxy, not frame possession ground truth',
                  'sha256': {name:_sha256(path) for name,path in [('home',home_path),('away',away_path),('events',events_path)]},
                  'boundary_margin_frames':boundary_margin_frames, 'max_event_frames':max_event_frames}
    return ExternalMatch(match_id, tracking, labels, mask, boundary, provenance)
