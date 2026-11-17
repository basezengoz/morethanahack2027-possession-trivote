import unittest
import numpy as np
from possession.contracts import Evidence
from possession.tcn_rule_fusion import fuse, keeper_support
from possession.rule_fusion import cues_from_rule_features
from possession.decode import decode_labels


class FusionTest(unittest.TestCase):
    def test_no_support_preserves_scores_transitions_and_decode(self):
        scores = np.array([[.7, .2, .1], [.1, .1, .8], [.2, .7, .1]])
        ev = Evidence(scores, np.zeros((3, 2)), np.zeros(3), np.zeros(3))
        out = fuse(ev, np.zeros((3, 2)), np.zeros(3), np.zeros((3, 2)), use_pass=True, use_restart=True, use_keeper=True)
        np.testing.assert_array_equal(out.state_scores, scores)
        for name in ('change', 'stop', 'restart'):
            np.testing.assert_array_equal(getattr(out, name), getattr(ev, name))
        np.testing.assert_array_equal(decode_labels(out, [(0, 1), (1, 3)]), decode_labels(ev, [(0, 1), (1, 3)]))

    def test_cues_are_soft_and_teams_are_distinct(self):
        ev = Evidence(np.full((4, 3), 1/3), np.zeros((4, 2)), np.zeros(4), np.zeros(4))
        p = np.array([[1, 0], [0, 1], [0, 0], [0, 0]])
        k = keeper_support({'keeper_poss': [0, 0, 1, 0], 'keeper_team': [-1, -1, 1, -1]}, 4)
        out = fuse(ev, p, [0, 0, 0, 1], k, use_pass=True, use_restart=True, use_keeper=True).state_scores
        self.assertGreater(out[0, 0], out[0, 1]); self.assertGreater(out[1, 1], out[1, 0])
        self.assertGreater(out[2, 1], out[2, 0]); self.assertGreater(out[3, 2], out[3, 0])
        self.assertTrue(np.all(out > 0)); np.testing.assert_allclose(out.sum(axis=1), 1)

    def test_pass_windows_conflicts_and_half_end(self):
        f = {'passes': [(0, 8, 0), (0, 9, 1)], 'out': np.zeros(10), 'setpiece': np.zeros(10)}
        p, _ = cues_from_rule_features(f, 10, hold_frames=25)
        np.testing.assert_array_equal(p[8], [1, 0]); np.testing.assert_array_equal(p[9], [0, 0])
        self.assertEqual(len(p), 10)

    def test_short_turnover_and_dead_interval_remain_possible(self):
        scores = np.tile([.98, .01, .01], (90, 1))
        scores[30:35] = [.01, .98, .01]
        scores[55:65] = [.01, .01, .98]
        ev = Evidence(scores, np.zeros((90, 2)), np.zeros(90), np.zeros(90))
        support = np.zeros((90, 2)); support[30:35, 1] = 1
        out = fuse(ev, support, np.zeros(90), np.zeros((90, 2)), use_pass=True)
        labels = decode_labels(out, [(0, 45), (45, 90)])
        self.assertTrue(np.all(labels[30:35] == 'A'))
        self.assertTrue(np.all(labels[55:65] == 'D'))
        self.assertEqual(labels[65], 'H')

    def test_invalid_cues_rejected(self):
        ev = Evidence(np.array([[.5, .4, .1]]), np.zeros((1, 2)), np.zeros(1), np.zeros(1))
        with self.assertRaises(ValueError):
            fuse(ev, [[float('nan'), 0]], [0], [[0, 0]], use_pass=True)


if __name__ == '__main__': unittest.main()
