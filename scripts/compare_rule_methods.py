"""Fixed rule-method comparison on DFL or target tracking; never reads target GT.

DFL has no Smart Tagging in this adapter; goalkeeper roles come from the roster.
Player speeds are derived locally.
All outputs belong under data/runs. Public scoring is a separate operation.
"""
import argparse
import json
import xml.etree.ElementTree as ET
from dataclasses import replace
from pathlib import Path

import numpy as np

from possession.contracts import TaggingEvidence
from possession.decode import decode_labels
from possession.dfl_data import load_dfl
from possession.external_evaluation import evaluate_external
from possession.features import build_features
from possession.io import load_tracking
from possession.predict import apply_published_dead_intervals, load_config, write_submission
from possession.quality import clean_tracking
from possession.rule_based import dead_ball, model
from possession.rule_based.loader import Match, load_match, load_tracking as load_rule_tracking
from possession.tagging import load_tagging
from possession.teacher import build_evidence


def blend_control(evidence, features):
    """Add kural tabanlı yöntem's bounded speed/duel control to H/A log odds, preserving D mass."""
    d = features.values['nearest_distance']
    control, _, _, _ = model.control_evidence(d[:, 0], d[:, 1], features.values['ball_speed'])
    scores = np.asarray(evidence.state_scores, dtype=np.float64).copy()
    live_mass = scores[:, :2].sum(axis=1)
    scores[:, 0] *= np.exp(control)
    scores[:, 1] *= np.exp(-control)
    denominator = scores[:, :2].sum(axis=1)
    scores[:, :2] *= np.divide(live_mass, denominator, out=np.zeros_like(live_mass),
                                where=denominator > 0)[:, None]
    return replace(evidence, state_scores=scores)


def original_and_blended(tracking, tags, config, target=False):
    clean = clean_tracking(tracking, config.get('quality', {}), tags.restart)
    features = build_features(clean, config.get('features', {}))
    evidence = build_evidence(features, tags, config)
    result = {}
    for name, ev in [('existing', evidence), ('hybrid_control', blend_control(evidence, features))]:
        labels = decode_labels(ev, features.blocks, config)
        if target:
            apply_published_dead_intervals(labels, features.blocks, config.get('published_dead_intervals', True))
        result[name] = labels
    return result


def dfl_goalkeepers(positions, info, tracking):
    """Map documented TW roster roles onto the DFL loader's ordered player slots."""
    root = ET.parse(info).getroot()
    general = root.find('.//General')
    teams = {general.attrib['HomeTeamId']: 0, general.attrib['GuestTeamId']: 1}
    keepers = {player.attrib['PersonId'] for player in root.findall('.//Player')
               if player.get('PlayingPosition') == 'TW'}
    slots = []
    for event, element in ET.iterparse(positions, events=('start', 'end')):
        if event == 'start' and element.tag == 'FrameSet' and element.get('TeamId') in teams:
            key = (element.get('TeamId'), element.get('PersonId'))
            if key not in slots:
                slots.append(key)
        if event == 'end':
            element.clear()
    if len(slots) != len(tracking.slot_team) or any(
            teams[team] != tracking.slot_team[i] for i, (team, _) in enumerate(slots)):
        raise ValueError('DFL roster and tracking slot order do not match')
    flags = np.array([person in keepers for _, person in slots], dtype=bool)
    if not all(np.any(flags & (tracking.slot_team == team)) for team in (0, 1)):
        raise ValueError('DFL goalkeeper role unavailable for at least one team')
    return flags


def dfl_as_rule_based_match(tracking, start, end, goalkeepers):
    """One contiguous period as a kural tabanlı yöntem match, using documented goalkeeper roles."""
    if tracking.fps != 25:
        raise ValueError('kural tabanlı yöntem model requires 25 fps')
    xy = tracking.players_xy[start:end]
    n, slots, _ = xy.shape
    players = np.full((n, slots, 3), np.nan, np.float32)
    players[:, :, :2] = xy
    speed = np.linalg.norm(np.diff(xy, axis=0) * np.asarray(tracking.pitch), axis=2) * tracking.fps
    speed[(speed > 12) | ~np.isfinite(speed)] = np.nan
    players[1:, :, 2] = speed
    ball = np.full((n, 3), np.nan, np.float32)
    ball[:, :2] = tracking.ball_xy_raw[start:end]
    return Match('DFL', 'external', n, tracking.pitch, players, ball, tracking.slot_team,
                 goalkeepers, ('home', 'away'), (True, False),
                 np.ones(n, np.int8), tags=[])


def rule_predict(m):
    f = dead_ball.features(m)
    llr = dead_ball.dead_evidence(m, f)
    dead = dead_ball.detect_dead(m, llr, f)
    return model.build_labels(m, f, dead)[0]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dfl', nargs=2, metavar=('POSITIONS', 'INFO'))
    p.add_argument('--target', action='store_true')
    p.add_argument('--out', required=True)
    args = p.parse_args()
    if bool(args.dfl) == args.target:
        p.error('choose exactly one of --dfl or --target')
    out = Path(args.out).resolve()
    if not out.is_relative_to(Path('data/runs').resolve()):
        raise ValueError('outputs must be under data/runs')
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_config('configs/rules.yaml')
    if args.dfl:
        match = load_dfl(*args.dfl)
        t = match.tracking
        goalkeepers = dfl_goalkeepers(*args.dfl, t)
        tags = TaggingEvidence(np.zeros((len(t.frame), 2)), np.zeros(len(t.frame)), np.zeros(len(t.frame)))
        predictions = original_and_blended(t, tags, cfg)
        rule_based = np.full(len(t.frame), 'X', dtype='<U1')
        for start, end in t.blocks:
            rule_based[start:end] = rule_predict(dfl_as_rule_based_match(t, start, end, goalkeepers))
        predictions['rule_based_goalkeeper_no_tags'] = rule_based
        report = {'match': match.match_id, 'provenance': match.provenance,
                  'limitations': ['DFL domain and possession semantics differ from target',
                                  'kural tabanlı yöntem adapter has roster goalkeeper roles but no tags; speed derived from coordinates',
                                  'DFL match is development data, not an untouched final test'],
                  'results': {name: evaluate_external(match, labels) for name, labels in predictions.items()}}
        (out / (match.match_id + '.json')).write_text(json.dumps(report, indent=2), encoding='utf-8')
        for name, metrics in report['results'].items():
            print(name, 'spell', metrics['spell_f1'], 'macro', metrics['macro_f1'])
    else:
        t = load_tracking()
        tags = load_tagging(None, len(t.frame), t.fps, cfg.get('tagging', {}))
        predictions = original_and_blended(t, tags, cfg, target=True)
        _, _, frame_id, _ = load_rule_tracking()
        if not np.array_equal(frame_id, np.arange(1, len(t.frame) + 1)):
            raise ValueError('kural tabanlı yöntem tracking frame IDs must be exactly 1..N in order')
        predictions['rule_based'] = rule_predict(load_match(with_homography=False))
        for name, labels in predictions.items():
            write_submission(out / (name + '.csv'), labels)
        print('Wrote frozen target predictions:', ', '.join(predictions))


if __name__ == '__main__':
    main()
