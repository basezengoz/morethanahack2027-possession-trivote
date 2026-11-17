"""Read a separate DFL match for independent reference evaluation."""
import xml.etree.ElementTree as ET
from datetime import datetime
import numpy as np
from .contracts import TrackingData
from .external_data import ExternalMatch, _sha256


def load_dfl(position_path, info_path):
    root = ET.parse(info_path).getroot()
    general = root.find('.//General')
    if general is None:
        raise ValueError('missing DFL match information')
    match_id = general.attrib['MatchId']
    teams = {general.attrib['HomeTeamId']: 0, general.attrib['GuestTeamId']: 1}
    records = []; slots = []; pitch = None; frame_set = None
    for event, element in ET.iterparse(position_path, events=('start', 'end')):
        if event == 'start' and element.tag == 'PitchSize':
            pitch = (float(element.attrib['X']), float(element.attrib['Y']))
        elif event == 'start' and element.tag == 'FrameSet':
            frame_set = dict(element.attrib)
            if frame_set['MatchId'] != match_id:
                raise ValueError('match id mismatch')
            if frame_set['TeamId'] in teams:
                key = (frame_set['TeamId'], frame_set['PersonId'])
                if key not in slots: slots.append(key)
        elif event == 'end' and element.tag == 'Frame' and frame_set['TeamId'] == 'BALL':
            a = element.attrib
            records.append((frame_set['GameSection'], int(a['N']), float(a['X']), float(a['Y']),
                            a.get('BallStatus'), a.get('BallPossession'),
                            datetime.fromisoformat(a['T'])))
        if event == 'end': element.clear()
    if not records or not pitch or min(pitch) <= 0 or not slots:
        raise ValueError('incomplete DFL tracking')
    keys = [(r[0], r[1]) for r in records]
    if len(set(keys)) != len(keys): raise ValueError('duplicate ball frame')
    lookup = {key: i for i, key in enumerate(keys)}
    n = len(records); blocks = []; start = 0
    for i in range(1, n):
        if keys[i][0] == keys[i-1][0]:
            elapsed = (records[i][6] - records[i-1][6]).total_seconds()
            if abs(elapsed - (keys[i][1] - keys[i-1][1]) / 25.) > .01:
                raise ValueError('DFL frame numbers and timestamps disagree')
        if keys[i][0] != keys[i-1][0] or keys[i][1] != keys[i-1][1]+1:
            if keys[i][0] == keys[i-1][0] and keys[i][1] <= keys[i-1][1]:
                raise ValueError('unordered frames')
            blocks.append((start, i)); start = i
    blocks.append((start, n))
    players = np.full((n, len(slots), 2), np.nan, np.float32)
    slot_lookup = {key: i for i, key in enumerate(slots)}
    frame_set = None
    for event, element in ET.iterparse(position_path, events=('start', 'end')):
        if event == 'start' and element.tag == 'FrameSet': frame_set = dict(element.attrib)
        elif event == 'end' and element.tag == 'Frame' and frame_set['TeamId'] in teams:
            a = element.attrib
            i = lookup.get((frame_set['GameSection'], int(a['N'])))
            if i is not None:
                j = slot_lookup[(frame_set['TeamId'], frame_set['PersonId'])]
                if np.isfinite(players[i,j]).any(): raise ValueError('duplicate player frame')
                players[i,j] = (float(a['X']) / pitch[0] + .5, float(a['Y']) / pitch[1] + .5)
        if event == 'end': element.clear()
    ball = np.array([(r[2]/pitch[0]+.5, r[3]/pitch[1]+.5) for r in records], np.float32)
    labels = np.array([2 if r[4] == '0' else {'1':0, '2':1}.get(r[5], -1)
                       if r[4] == '1' else -1 for r in records], np.int8)
    known = labels >= 0
    tracking = TrackingData(np.arange(n, dtype=np.int32), players, ball,
                            np.isfinite(players).all(axis=2), np.isfinite(ball).all(axis=1),
                            np.array([teams[k[0]] for k in slots], np.int8), 25., pitch, blocks)
    return ExternalMatch(match_id, tracking, labels, known, known.copy(),
                         {'source':'DFL IDSSE', 'sha256':{'positions':_sha256(position_path),
                                                          'info':_sha256(info_path)}})
