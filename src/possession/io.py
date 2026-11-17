"""Streaming tactical ATD loading with strict per-record alignment checks."""
import xml.etree.ElementTree as ET
import numpy as np
from games import GAME, GAMES, require, source_frames, half_video_blocks
from .contracts import TrackingData


def _teams(meta_path):
    root = ET.parse(meta_path).getroot()
    teams = [t.get('id') for t in root.iter('Team')]
    if len(teams) != 2 or None in teams or len(set(teams)) != 2:
        raise ValueError('metadata must have two distinct team IDs')
    ids = [p.get('teamId') for p in root.iter('Player')]
    if not ids or any(x not in teams for x in ids):
        raise ValueError('unknown team ID or no player slots')
    return np.array([teams.index(x) for x in ids], dtype=np.int8)


def _xy(group):
    parts = group.strip().split(',')
    if len(parts) != 3:
        raise ValueError('observation must have x,y,speed')
    try:
        x, y, speed = map(float, parts)
    except ValueError as exc:
        raise ValueError('invalid observation') from exc
    if np.isnan(x) and np.isnan(y):
        return (np.nan, np.nan)
    if not np.isfinite(x) or not np.isfinite(y) or not (-0.5 <= x <= 1.5 and -0.5 <= y <= 1.5):
        raise ValueError('invalid normalized observation')
    if not (np.isfinite(speed) or np.isnan(speed)):
        raise ValueError('invalid speed')
    return x, y


def _read_atd(raw_path, slot_team, n_frames, fps, pitch, blocks):
    nslots = len(slot_team)
    players = np.full((n_frames, nslots, 2), np.nan, np.float32)
    ball = np.full((n_frames, 2), np.nan, np.float32)
    count = 0
    with open(raw_path, encoding='utf-8') as f:
        for count, line in enumerate(f, 1):
            parts = line.strip().split(':')
            if len(parts) != 3 or count > n_frames:
                raise ValueError('invalid ATD record count or structure')
            try:
                frame_id = int(parts[0])
            except ValueError as exc:
                raise ValueError('invalid frame ID') from exc
            if frame_id != count:
                raise ValueError('ATD frame IDs are not sequential from 1')
            groups = parts[1].split(';')
            if len(groups) != nslots:
                raise ValueError('ATD slot count does not match metadata')
            players[count - 1] = [_xy(group) for group in groups]
            ball[count - 1] = _xy(parts[2])
    if count != n_frames:
        raise ValueError('ATD record count does not match video')
    return TrackingData(np.arange(n_frames, dtype=np.int32), players, ball,
                        np.isfinite(players).all(axis=2), np.isfinite(ball).all(axis=1),
                        np.asarray(slot_team, dtype=np.int8), fps, pitch, blocks)


def load_tracking(data_dir=None):
    meta, raw = require(['atd_meta', 'atd_raw'], data_dir=data_dir)
    game = GAMES[GAME]
    blocks = [(start, end + 1) for _, start, end in half_video_blocks(GAME)]
    return _read_atd(raw, _teams(meta), source_frames(), game['fps'], game['pitch'], blocks)

