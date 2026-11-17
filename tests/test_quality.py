import unittest
import numpy as np
from src.possession.contracts import TrackingData
from src.possession.quality import clean_tracking, fuse_ball_candidates, load_ball_candidates
from src.possession.features import build_features

def sample(ball, blocks=None):
    b = np.array(ball, dtype=np.float32); n=len(b)
    p = np.tile(np.array([[[.1,.1],[.9,.9]]], np.float32), (n,1,1))
    return TrackingData(np.arange(n), p, b, np.ones((n,2),bool), np.isfinite(b).all(axis=1), np.array([0,1]), 25, (100, 50), blocks or [(0,n)])

class QualityTest(unittest.TestCase):
    def test_short_long_restart_half(self):
        b = [[.1,.1],[np.nan,np.nan],[.1,.1],[np.nan,np.nan],[.1,.1],[np.nan,np.nan],[.1,.1]]
        t = sample(b, [(0,5),(5,7)])
        c = clean_tracking(t, {'max_gap_frames':1}, np.array([0,0,0,1,0,0,0]))
        self.assertTrue(c.ball_interpolated[1]); self.assertFalse(c.ball_interpolated[3]); self.assertFalse(c.ball_interpolated[5])
        self.assertFalse(c.ball_observed[1])
    def test_spike(self):
        c = clean_tracking(sample([[.1,.1],[.9,.9],[.1,.1]]))
        self.assertTrue(c.ball_rejected[1]); self.assertTrue(c.ball_interpolated[1])

    def test_long_gap_stays_missing(self):
        b = [[.1,.1]] + [[np.nan,np.nan]] * 3 + [[.1,.1]]
        c = clean_tracking(sample(b), {'max_gap_frames': 2})
        self.assertFalse(c.ball_interpolated.any())
        self.assertTrue(np.isnan(c.ball_xy[1:4]).all())


def candidates(frames, xy=None, scores=None):
    n = len(frames)
    return dict(frame=np.array(frames, dtype=np.int64),
                pixel_xy=np.tile([960., 540.], (n, 1)),
                pitch_xy=np.tile([.1, .1], (n, 1)) if xy is None else np.array(xy),
                score=np.full(n, .95) if scores is None else np.array(scores),
                observed=np.ones(n, bool), projection_valid=np.ones(n, bool),
                source=np.full(n, 'wasb'))


class FusionTest(unittest.TestCase):
    def test_raw_preserved_gap_filled_and_provenance(self):
        t = sample([[.1,.1]] + [[np.nan,np.nan]]*8 + [[.1,.1]])
        raw = t.ball_xy_raw.copy()
        c = candidates([0,4,3,5], [[.9,.9],[.1,.1],[.1,.1],[.1,.1]])
        out = clean_tracking(t, candidates=c)
        np.testing.assert_array_equal(t.ball_xy_raw, raw)
        np.testing.assert_array_equal(out.ball_xy[0], raw[0])
        self.assertEqual(out.ball_source[0], 'atd')
        self.assertTrue(out.ball_visual[3:6].all())
        self.assertFalse(out.ball_observed[3:6].any())
        self.assertFalse(out.ball_interpolated[3:6].any())
        self.assertTrue((out.ball_source[3:6] == 'wasb').all())
        self.assertTrue(np.isnan(out.ball_xy[1]).all())
        features = build_features(out).values
        self.assertTrue(features['ball_available'][3:6].all())
        self.assertTrue(features['ball_visual'][3:6].all())
        self.assertEqual(features['gap_age'][4], 4)  # age of ATD, not of a visual fill

    def test_low_confidence_bad_projection_and_ambiguity(self):
        t = sample([[np.nan,np.nan]] * 6)
        c = candidates([0,1,2,2,3,4,5], scores=[.4,.95,.95,.94,.95,.95,.95])
        c['projection_valid'][1] = False
        c['pitch_xy'][3] = [.9,.9]
        c['observed'][4] = False
        c['pixel_xy'][5] = np.nan
        out = clean_tracking(t, candidates=c)
        # The final candidate has no corroborating, unambiguous neighbor.
        self.assertFalse(out.ball_visual.any())

    def test_physical_conflict_and_no_half_crossing(self):
        t = sample([[.1,.1],[np.nan,np.nan],[np.nan,np.nan],[np.nan,np.nan]], [(0,2),(2,4)])
        out = clean_tracking(t, candidates=candidates([1,2], [[.9,.9],[.9,.9]]))
        self.assertFalse(out.ball_visual.any())
        # Consistent points are accepted within a half even with no ATD anchor.
        out = clean_tracking(t, candidates=candidates([2,3]))
        self.assertTrue(out.ball_visual[2:].all())

    def test_rejected_atd_can_be_replaced_without_erasing_rejection(self):
        t = sample([[.1,.1],[.9,.9],[.1,.1]])
        out = clean_tracking(t, candidates=candidates([1]))
        self.assertTrue(out.ball_observed[1])
        self.assertTrue(out.ball_rejected[1])
        self.assertTrue(out.ball_visual[1])
        self.assertFalse(out.ball_interpolated[1])
        self.assertEqual(out.ball_source[1], 'wasb')

    def test_npz_adapter_and_bad_frame_contract(self):
        import tempfile
        from pathlib import Path
        root = Path('data/runs'); root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as directory:
            path = Path(directory) / 'synthetic.npz'
            np.savez(path, **candidates([0,1]))
            artifact = load_ball_candidates(path)
            self.assertTrue(clean_tracking(sample([[np.nan,np.nan]]*2), candidates=artifact).ball_visual.all())
        for frames in ([-1], [2]):
            with self.assertRaises(ValueError):
                clean_tracking(sample([[np.nan,np.nan]]*2), candidates=candidates(frames))
        c = candidates([0]); c['frame'] = np.array([.5])
        with self.assertRaises(ValueError):
            clean_tracking(sample([[np.nan,np.nan]]*2), candidates=c)
