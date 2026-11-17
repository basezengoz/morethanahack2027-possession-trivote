import unittest
import numpy as np
from tests.test_quality import sample
from src.possession.quality import clean_tracking
from src.possession.features import build_features

class FeaturesTest(unittest.TestCase):
    def test_units_and_slot_jump(self):
        t = sample([[.2,.1],[.2,.1]])
        t.players_xy[1,0] = [.9,.9]
        f = build_features(clean_tracking(t)).values
        self.assertAlmostEqual(f['nearest_distance'][0,0], 10, places=4)
        self.assertTrue(np.isnan(f['team_speed'][1,0]))
        self.assertAlmostEqual(f['boundary_distance'][0], 5, places=4)

    def test_missing_frame_has_no_observed_control_and_slots_are_permutable(self):
        t = sample([[.1,.1],[np.nan,np.nan],[.1,.1]])
        f = build_features(clean_tracking(t, {'max_gap_frames':0})).values
        self.assertTrue((f['control_support'][1] == 0).all())
        t.players_xy = t.players_xy[:,::-1].copy()
        t.player_observed = t.player_observed[:,::-1].copy()
        t.slot_team = t.slot_team[::-1].copy()
        other = build_features(clean_tracking(t, {'max_gap_frames':0})).values
        for name in ('nearest_distance','control_support','team_speed','nearby_player_count'):
            np.testing.assert_allclose(f[name], other[name], equal_nan=True)
