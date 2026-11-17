import unittest
from types import SimpleNamespace
import numpy as np
from possession.rule_based_tcn_evidence import tcn_cues
from possession.rule_based import model


class TcnEvidenceTest(unittest.TestCase):
    def test_bounded_symmetric_and_missing(self):
        p = np.array([[.8, .1, .1], [.1, .8, .1], [.1, .1, .8], [.8, .1, .1]])
        team, dead = tcn_cues(p, [1, 1, 1, 0])
        self.assertAlmostEqual(team[0], -team[1]); self.assertEqual(team[2], 0)
        self.assertGreater(dead[2], 0); self.assertEqual(team[3], 0); self.assertEqual(dead[3], 0)
        self.assertTrue(np.all(np.abs(team) <= .5)); self.assertTrue(np.all(np.abs(dead) <= .5))

    def test_rule_decoder_uses_team_evidence(self):
        n = 120
        m = SimpleNamespace(n_frames=n, half=np.ones(n, dtype=int), tags=[])
        f = {'d_home': np.full(n, np.nan), 'd_away': np.full(n, np.nan),
             'ball_speed': np.full(n, np.nan), 'ball_x': np.full(n, np.nan),
             'keeper': np.zeros(n, bool), 'keeper_team': np.full(n, -1), 'setpiece_tags': []}
        dead = np.zeros(n, bool)
        base = model.build_labels(m, f, dead)[0]
        np.testing.assert_array_equal(base, model.build_labels(m, f, dead, np.zeros(n))[0])
        cues = np.r_[np.full(60, .5), np.full(60, -.5)]
        # Independent halves are supplied to kural tabanlı yöntem separately in the external adapter.
        result = model.build_labels(m, f, dead, cues)[0]
        self.assertIn('A', result)
        dead[25:40] = True
        self.assertTrue(np.all(model.build_labels(m, f, dead, cues)[0][25:40] == 'D'))

    def test_reject_invalid_probabilities(self):
        with self.assertRaises(ValueError): tcn_cues([[.5, .5, .5]], [True])
