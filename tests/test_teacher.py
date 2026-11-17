import unittest
import numpy as np
from src.possession.contracts import FeatureData, TaggingEvidence
from src.possession.teacher import build_evidence
from src.possession.decode import decode_labels
from src.possession.contracts import TrackingData
from src.possession.quality import clean_tracking
from src.possession.features import build_features


def sample(n=8):
    values = dict(nearest_distance=np.tile([1., 12.], (n, 1)), control_support=np.tile([.8, .1], (n, 1)),
                  ball_available=np.ones(n, bool), ball_speed=np.full(n, 3.),
                  team_speed=np.ones((n, 2)), boundary_distance=np.full(n, 20.))
    return FeatureData(values, [(0, n)]), TaggingEvidence(np.zeros((n, 2)), np.zeros(n), np.zeros(n))


class TestTeacher(unittest.TestCase):
    def test_weak_team_and_neutral_missing_ball(self):
        f, tag = sample()
        result = build_evidence(f, tag)
        self.assertTrue(np.all(result.state_scores[:, 0] > result.state_scores[:, 1]))
        f.values['ball_available'][:] = False
        result = build_evidence(f, tag)
        np.testing.assert_allclose(result.state_scores[:, 0], result.state_scores[:, 1])
        self.assertTrue(np.all(result.stop == 0))
        np.testing.assert_allclose(result.state_scores.sum(axis=1), 1)

    def test_no_single_signal_is_dead(self):
        f, tag = sample()
        f.values['ball_available'][0] = False
        f.values['team_speed'][1] = 0
        tag.restart[2] = 1
        f.values['ball_speed'][3] = 0
        f.values['ball_speed'][4] = 0
        f.values['boundary_distance'][4] = 0
        self.assertTrue(np.all(build_evidence(f, tag).stop[:4] == 0))
        self.assertEqual(build_evidence(f, tag).stop[4], 1)

    def test_tag_conflict_zeros_team_evidence(self):
        f, tag = sample()
        f.values['ball_available'][:] = False
        tag.team_support[:, 1] = 1
        self.assertTrue(np.all(build_evidence(f, tag).state_scores[:, 1] > build_evidence(f, tag).state_scores[:, 0]))
        tag.conflict[:] = 1
        out = build_evidence(f, tag).state_scores
        np.testing.assert_allclose(out[:, 0], out[:, 1])

    def test_stoppage_survives_missing_ball_and_resumes_on_control(self):
        f, tag = sample(100)
        # Stationary restart taker, moving teammates: preparation is still a stop.
        f.values['ball_speed'][20:30] = 0
        tag.restart[20:80] = 1
        f.values['ball_available'][30:70] = False
        f.values['ball_speed'][30:70] = np.nan
        f.values['nearest_distance'][30:70] = np.nan
        f.values['control_support'][30:70] = 0
        ev = build_evidence(f, tag)
        labels = decode_labels(ev, f.blocks)
        self.assertEqual(''.join(labels), 'H'*20 + 'D'*50 + 'H'*30)
        self.assertFalse(ev.stop[30:70].any())
        self.assertFalse(ev.restart[20:70].any())
        self.assertTrue((ev.restart[70:80] > 0).all())

    def test_set_piece_window_does_not_force_dead_or_a_restart(self):
        f, tag = sample(60)
        tag.restart[:] = 1
        f.values['ball_speed'][20:30] = 0
        ev = build_evidence(f, tag)
        self.assertEqual(''.join(decode_labels(ev, f.blocks)), 'H'*20+'D'*10+'H'*30)
        self.assertFalse(ev.restart[20:30].any())
        self.assertTrue((ev.restart[:20] > 0).all())

    def test_rolling_outside_ball_with_restart_and_visual_projection_guard(self):
        f, tag = sample(40)
        f.values['boundary_distance'][10:30] = -3
        tag.restart[10:30] = 1
        ev = build_evidence(f, tag)
        self.assertEqual(''.join(decode_labels(ev, f.blocks)), 'H'*10+'D'*20+'H'*10)
        # Ground projections alone cannot establish line crossings of aerial balls.
        f.values['ball_visual'] = np.ones(40, bool)
        ev = build_evidence(f, tag)
        self.assertFalse(ev.stop.any())
        self.assertEqual(set(decode_labels(ev, f.blocks)), {'H'})

    def test_missing_ball_even_with_restart_does_not_create_dead(self):
        f, tag = sample(60)
        f.values['ball_available'][10:50] = False
        f.values['ball_speed'][10:50] = np.nan
        f.values['team_speed'][10:50] = 0
        tag.restart[10:50] = 1
        tag.team_support[10:30, 1] = .25
        tag.team_support[30:50, 0] = .25
        ev = build_evidence(f, tag)
        self.assertFalse(ev.stop.any())
        self.assertNotIn('D', decode_labels(ev, f.blocks))

    def test_missing_at_half_boundary_cannot_inherit_previous_dead(self):
        f, tag = sample(60)
        f.blocks = [(0,30),(30,60)]
        tag.restart[:30] = 1
        f.values['ball_speed'][:30] = 0
        f.values['ball_available'][30:50] = False
        f.values['ball_speed'][30:50] = np.nan
        labels = decode_labels(build_evidence(f, tag), f.blocks)
        self.assertEqual(''.join(labels), 'D'*30+'H'*30)

    def test_geometry_through_decoder_restart_and_occlusion(self):
        n=120
        ball=np.tile([.53,.5],(n,1)).astype(np.float32)
        ball[:30,0]=.5+np.arange(30)*.001
        ball[90:,0]=.53+np.arange(30)*.001
        players=np.stack((ball,ball+np.array([.2,0],np.float32)),axis=1)
        ball[50:90]=np.nan
        tracking=TrackingData(np.arange(n),players,ball,np.ones((n,2),bool),
                              np.isfinite(ball).all(axis=1),np.array([0,1]),25.,(100.,50.),[(0,n)])
        tags=TaggingEvidence(np.zeros((n,2)),np.zeros(n),np.zeros(n))
        tags.restart[20:110]=1
        features=build_features(clean_tracking(tracking,restart=tags.restart))
        labels=decode_labels(build_evidence(features,tags),features.blocks)
        # First still frame is 31, not the start of the broad restart window (20).
        self.assertEqual(''.join(labels),'H'*31+'D'*59+'H'*30)

if __name__ == '__main__': unittest.main()
