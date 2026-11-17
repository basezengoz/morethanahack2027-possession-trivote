"""Synthetic checks through geometry, features, teacher and temporal decoder."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
from src.possession.contracts import TrackingData, TaggingEvidence
from src.possession.io import _read_atd
from src.possession.quality import clean_tracking
from src.possession.features import build_features
from src.possession.teacher import build_evidence
from src.possession.decode import decode_labels


def pipeline(owners, *, missing=(), dead=(), blocks=None):
    n = len(owners)
    ball = np.tile([.5, .5], (n, 1)).astype(np.float32)
    players = np.empty((n, 2, 2), np.float32)
    for i, owner in enumerate(owners):
        players[i] = [[.5, .5], [.8, .8]] if owner == 'H' else [[.8, .8], [.5, .5]]
    for i in dead:
        ball[i] = [.001, .5]
        players[i] = [[.2, .5], [.8, .5]]
    ball[list(missing)] = np.nan
    blocks = blocks or [(0, n)]
    t = TrackingData(np.arange(n), players, ball, np.ones((n, 2), bool),
                     np.isfinite(ball).all(axis=1), np.array([0, 1]), 25, (100., 50.), blocks)
    tags = TaggingEvidence(np.zeros((n, 2)), np.zeros(n), np.zeros(n))
    clean = clean_tracking(t, restart=tags.restart)
    features = build_features(clean)
    evidence = build_evidence(features, tags)
    return decode_labels(evidence, blocks), features, evidence


class IntegrationTest(unittest.TestCase):
    def test_geometry_units_and_half_reset(self):
        labels, f, _ = pipeline('H' * 40 + 'A' * 40, blocks=[(0, 40), (40, 80)])
        self.assertEqual(set(labels[:40]), {'H'})
        self.assertEqual(set(labels[40:]), {'A'})
        self.assertAlmostEqual(f.values['nearest_distance'][0, 1], np.hypot(30, 15), places=4)
        self.assertTrue(np.isnan(f.values['ball_speed'][40]))

    def test_short_touch_and_real_short_takeover(self):
        touch, _, _ = pipeline('H' * 30 + 'A' + 'H' * 30)
        takeover, _, _ = pipeline('H' * 30 + 'A' * 8 + 'H' * 30)
        self.assertEqual(touch[30], 'H')
        self.assertTrue(np.all(takeover[30:38] == 'A'))

    def test_dead_then_same_team_and_missing_ball(self):
        labels, _, evidence = pipeline('H' * 90, dead=range(30, 60))
        self.assertEqual(labels[15], 'H')
        self.assertEqual(labels[45], 'D')
        self.assertEqual(labels[75], 'H')
        labels, f, evidence = pipeline('H' * 90, missing=range(30, 60))
        self.assertFalse(f.values['ball_available'][30:60].any())
        self.assertFalse((labels[30:60] == 'D').any())
        self.assertFalse(evidence.stop[30:60].any())

    def test_sequential_ids_rejected(self):
        Path('data/runs').mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir='data/runs') as d:
            p = Path(d) / 'synthetic.txt'
            p.write_text('1:0.1,0.2,0;0.9,0.8,0:0.2,0.3,0\n3:0.1,0.2,0;0.9,0.8,0:0.2,0.3,0\n')
            with self.assertRaises(ValueError):
                _read_atd(p, np.array([0, 1]), 2, 25, (100, 50), [(0, 2)])
            p.write_text('1:10,20,0;0.9,0.8,0:0.2,0.3,0\n')
            with self.assertRaises(ValueError):
                _read_atd(p, np.array([0, 1]), 1, 25, (100, 50), [(0, 1)])

if __name__ == '__main__':
    unittest.main()
