import unittest
import numpy as np
from possession.contracts import TrackingData
from possession.observation_masks import transfer_ball_mask


def sample(observed,blocks):
    n=len(observed)
    return TrackingData(np.arange(n),np.zeros((n,2,2)),np.ones((n,2)),
                        np.ones((n,2),bool),np.array(observed,bool),np.array([0,1]),blocks=blocks)


class ObservationMaskTest(unittest.TestCase):
    def test_block_alignment_and_immutability(self):
        target=sample([1]*8,[(0,4),(4,8)])
        pattern=sample([1,0,0,1],[(0,2),(2,4)])
        changed,drop=transfer_ball_mask(target,pattern)
        np.testing.assert_array_equal(drop,[0,1,0,1,1,0,1,0])
        self.assertTrue(np.isfinite(target.ball_xy_raw).all())
        self.assertTrue(np.isnan(changed.ball_xy_raw[drop]).all())
        np.testing.assert_array_equal(changed.players_xy,target.players_xy)


if __name__=='__main__': unittest.main()
