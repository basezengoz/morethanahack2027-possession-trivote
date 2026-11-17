import unittest
import importlib.util
import numpy as np
HAS_TORCH=importlib.util.find_spec('torch') is not None
if HAS_TORCH:
    import torch
    from possession.model import N_RAW, SCHEMA, swap_teams
    from possession.model_v2 import enrich, NormalizerV2, TemporalModelV2, WIDTH
    from possession.train import augment


@unittest.skipUnless(HAS_TORCH, 'optional training dependency is not installed')
class ModelV2Test(unittest.TestCase):
    def test_context_resets_and_swap(self):
        x=np.ones((6,N_RAW),np.float32)
        offsets={}; i=0
        for name,w in SCHEMA: offsets[name]=i; i+=w
        x[:,offsets['ball_observed']]=[1,0,0,0,0,1]
        x[:,offsets['ball_available']]=[1,0,0,0,0,1]
        x[:,0]=2; x[:,1]=5
        a=enrich(x,[(0,3),(3,6)])
        self.assertAlmostEqual(a[2,N_RAW+3],.08)
        self.assertTrue(np.isnan(a[3,N_RAW+4]))
        self.assertAlmostEqual(a[3,N_RAW+3],.04)
        b=enrich(swap_teams(x),[(0,3),(3,6)])
        np.testing.assert_allclose(a[:,N_RAW],-b[:,N_RAW])
        norm=NormalizerV2.fit([a]); self.assertEqual(norm.transform(a).shape,(6,2*WIDTH))
        model=TemporalModelV2(channels=2)
        d,t=model(torch.from_numpy(norm.transform(a).T[None]))
        self.assertEqual(tuple(model.boundary_logits.shape),(1,6))
        self.assertEqual(tuple(t.shape),(1,2,6))

    def test_augmentation_recomputes_gap_features(self):
        x=enrich(np.ones((12,N_RAW),np.float32),[(0,12)])
        cfg=dict(team_swap=0.,tag_dropout=0.,missing_block_probability=1.,missing_block_frames=3)
        y,_=augment(x,np.zeros(12,np.int64),np.random.default_rng(1),cfg)
        self.assertTrue(np.any(y[:,N_RAW+3]>0))

    def test_window_retains_earlier_missing_context(self):
        x=np.ones((600,N_RAW),np.float32)
        offsets={}; pos=0
        for name,w in SCHEMA: offsets[name]=pos; pos+=w
        x[1:,offsets['ball_observed']]=0
        x[1:,offsets['ball_available']]=0
        full=enrich(x,[(0,len(x))])
        cfg=dict(team_swap=0.,tag_dropout=0.,missing_block_probability=0.,missing_block_frames=3)
        window,_=augment(full[100:500],np.zeros(400,np.int64),np.random.default_rng(1),cfg)
        np.testing.assert_allclose(window[:,N_RAW+3:N_RAW+5],full[100:500,N_RAW+3:N_RAW+5],atol=2e-4)


if __name__ == '__main__': unittest.main()
