"""Smart Tagging as bounded, overlapping weak evidence, never labels."""
import unicodedata
import numpy as np
from starter.load_smart_tagging import load
from .contracts import TaggingEvidence

ATTACK = {'buildup', 'progression', 'attackingtransition', 'creatingchances',
          'longball', 'ballinfinalthird', 'ballinthebox', 'playersinthebox', 'goals'}
DEFENSE = {'defendingindefensivethird', 'defendinginmiddlethird',
           'defendinginattackingthird', 'defensivetransition'}


def _norm(value):
    text = unicodedata.normalize('NFKD', str(value or '')).casefold()
    return ''.join(c for c in text if c.isalnum())


def tagging_from_events(events, n_frames, fps, edge_frames=None):
    edge = int(round(fps)) if edge_frames is None else int(edge_frames)
    if edge < 0:
        raise ValueError('edge_frames must be nonnegative')
    support = np.zeros((n_frames, 2), np.float32)
    restarts = np.zeros(n_frames + 1, np.int32)
    for event in events:
        original_start = int(event['start_frame']); original_end = int(event['end_frame'])
        start = max(0, min(n_frames, original_start)); end = max(0, min(n_frames, original_end))
        if end <= start:
            continue
        code = _norm(event.get('code'))
        tags = {_norm(k): v for k, v in event.get('tags', {}).items()}
        if code == 'setpieces':
            restarts[start] += 1; restarts[end] -= 1
            continue
        if code not in ATTACK and code not in DEFENSE:
            continue
        name = _norm(tags.get('team'))
        team = 0 if name in ('barcelona', 'fcbarcelona') else 1 if name in ('atleticclub', 'athleticclub', 'athleticbilbao') else None
        if team is None:
            continue
        if code in DEFENSE:
            team = 1 - team
        ix = np.arange(start, end)
        taper = np.ones(end-start, np.float32) if edge == 0 else np.minimum(1, np.minimum(ix-original_start+1, original_end-ix) / edge)
        support[start:end, team] = np.maximum(support[start:end, team], 0.25*taper)
    conflict = np.all(support > 0, axis=1).astype(np.float32)
    return TaggingEvidence(support, (np.cumsum(restarts[:-1]) > 0).astype(np.float32), conflict)


def load_tagging(data_dir=None, n_frames=None, fps=25, config=None):
    if n_frames is None:
        from games import source_frames
        n_frames = source_frames()
    return tagging_from_events(load(data_dir=data_dir, fps=fps), n_frames, fps, (config or {}).get("edge_frames"))
