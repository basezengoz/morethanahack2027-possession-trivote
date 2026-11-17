import unittest
from types import SimpleNamespace
import numpy as np
from possession.learned_rule_postprocess import inputs, cues, tag_mask


class LearnedPostprocessTest(unittest.TestCase):
    def test_team_symmetry_bounds_and_tag_gate(self):
        p = np.array([[.8, .1, .1], [.1, .8, .1], [.2, .2, .6]])
        labels = np.array(['H', 'A', 'D']); observed = [True, True, False]
        tx, dx = inputs(p, labels, observed)
        np.testing.assert_allclose(tx[0], -tx[1]); np.testing.assert_allclose(dx[0], dx[1])
        weights = {'team': [1, 1, 1, 1], 'dead': [1, 1, 1, 1]}
        team, dead = cues(p, labels, observed, weights, [False, False, True])
        self.assertAlmostEqual(team[0], -team[1]); self.assertEqual(team[2], 0); self.assertEqual(dead[2], 0)
        self.assertTrue(np.all(np.abs(team) <= .5)); self.assertTrue(np.all(np.abs(dead) <= .5))

    def test_event_bounds_and_invalid_input(self):
        m = SimpleNamespace(n_frames=5, tags=[{'start_frame': -2, 'end_frame': 2},
                                               {'start_frame': 4, 'end_frame': 9}])
        np.testing.assert_array_equal(tag_mask(m), [1, 1, 0, 0, 1])
        with self.assertRaises(ValueError): inputs([[.5, .5, .5]], ['H'], [True])
