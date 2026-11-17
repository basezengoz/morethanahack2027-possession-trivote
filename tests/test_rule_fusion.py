"""Synthetic checks for the experimental rule-evidence connection."""
import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np

from possession.contracts import Evidence, FeatureData
from scripts.compare_rule_methods import blend_control, dfl_goalkeepers


class RuleFusionTest(unittest.TestCase):
    def test_goalkeeper_roles_follow_tracking_slot_order(self):
        with tempfile.TemporaryDirectory(dir='data/runs') as tmp:
            info = Path(tmp) / 'info.xml'
            positions = Path(tmp) / 'positions.xml'
            info.write_text('<Root><General HomeTeamId="H" GuestTeamId="A"/>'
                            '<Player PersonId="h" PlayingPosition="TW"/>'
                            '<Player PersonId="a" PlayingPosition="TW"/></Root>')
            positions.write_text('<Root><FrameSet TeamId="H" PersonId="x"/>'
                                 '<FrameSet TeamId="H" PersonId="h"/>'
                                 '<FrameSet TeamId="A" PersonId="a"/></Root>')
            tracking = SimpleNamespace(slot_team=np.array([0, 0, 1]))
            np.testing.assert_array_equal(dfl_goalkeepers(positions, info, tracking), [False, True, True])
            tracking.slot_team = np.array([1, 0, 1])
            with self.assertRaises(ValueError):
                dfl_goalkeepers(positions, info, tracking)

    def test_control_direction_dead_mass_and_unobserved_ball(self):
        evidence = Evidence(np.full((4, 3), 1 / 3), np.zeros((4, 2)),
                            np.zeros(4), np.zeros(4))
        features = FeatureData({'nearest_distance': np.array([[1, 5], [5, 1], [np.nan, np.nan], [1, 5]]),
                                'ball_speed': np.array([1., 1., np.nan, 12.])})
        out = blend_control(evidence, features)
        self.assertGreater(out.state_scores[0, 0], out.state_scores[0, 1])
        self.assertGreater(out.state_scores[1, 1], out.state_scores[1, 0])
        np.testing.assert_allclose(out.state_scores[2:], evidence.state_scores[2:])
        np.testing.assert_allclose(out.state_scores[:, 2], evidence.state_scores[:, 2])
        np.testing.assert_allclose(out.state_scores.sum(axis=1), 1.)
        np.testing.assert_allclose(evidence.state_scores, 1 / 3)


if __name__ == '__main__':
    unittest.main()
