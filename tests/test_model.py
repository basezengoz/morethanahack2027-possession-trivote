import unittest
import importlib.util
import numpy as np
from possession.contracts import FeatureData, TaggingEvidence, Evidence
HAS_TORCH = importlib.util.find_spec('torch') is not None
if HAS_TORCH:
    import torch
    from possession.model import SCHEMA, N_RAW, Normalizer, TemporalModel, flatten_features, infer_evidence, swap_teams


def sample(n=24):
    vals={}
    for name, width in SCHEMA:
        if not name.startswith('tag_'):
            vals[name]=np.ones((n,width),np.float32) if width==2 else np.ones(n,np.float32)
    vals['nearest_distance'][0,0]=np.nan
    return FeatureData(vals,[(0,n)]),TaggingEvidence(np.zeros((n,2),np.float32),np.zeros(n,np.float32),np.zeros(n,np.float32))

@unittest.skipUnless(HAS_TORCH, 'optional training dependency is not installed')
class TemporalModelTest(unittest.TestCase):
    def test_normalizer_mask_swap(self):
        f,t=sample(); raw=flatten_features(f,t); norm=Normalizer.fit([raw]); x=norm.transform(raw)
        self.assertEqual(x.shape,(24,2*N_RAW)); self.assertTrue(np.isfinite(x).all()); self.assertEqual(x[0,N_RAW],0)
        swapped=swap_teams(raw); np.testing.assert_allclose(norm.transform(swapped),swap_teams_normalized(x),atol=1e-6)
        _,labels=swap_teams(raw,np.array([0,1,2,-1]*6)); np.testing.assert_equal(labels[:4],[1,0,2,-1])

    def test_model_and_untrained_dead_head(self):
        torch.manual_seed(1); f,t=sample(); raw=flatten_features(f,t); norm=Normalizer.fit([raw]); model=TemporalModel(channels=4)
        base=Evidence(np.tile([.5,.2,.3],(24,1)).astype('float32'),np.full((24,2),.2),np.full(24,.3),np.full(24,.4))
        out=infer_evidence(model,f,t,base,norm,center_frames=8)
        np.testing.assert_array_equal(out.state_scores[:,2],base.state_scores[:,2]); np.testing.assert_allclose(out.state_scores.sum(axis=1),1,atol=1e-6)
        np.testing.assert_array_equal(out.stop,base.stop)
        with self.assertRaises(ValueError): infer_evidence(model,f,t,base,norm,context_frames=10)

    def test_no_context_across_blocks(self):
        torch.manual_seed(2); f,t=sample(); f.blocks=[(0,12),(12,24)]
        raw=flatten_features(f,t); norm=Normalizer.fit([raw]); model=TemporalModel(channels=4)
        base=Evidence(np.tile([.4,.4,.2],(24,1)).astype('float32'),np.zeros((24,2)),np.zeros(24),np.zeros(24))
        before=infer_evidence(model,f,t,base,norm).state_scores.copy()
        f.values['nearest_distance'][12:,:]=100
        after=infer_evidence(model,f,t,base,norm).state_scores
        np.testing.assert_array_equal(before[:12],after[:12])

    def test_unseen_constant_channels_and_invalid_normalization(self):
        raw = np.zeros((4,N_RAW), np.float32)
        raw[:,0] = np.nan
        norm = Normalizer.fit([raw])
        shifted = norm.transform(np.ones_like(raw))
        np.testing.assert_allclose(shifted[:,:N_RAW], 1)
        for mean,scale in ((np.full(N_RAW,np.nan),np.ones(N_RAW)),
                           (np.zeros(N_RAW),np.full(N_RAW,np.inf)),
                           (np.zeros(N_RAW),np.zeros(N_RAW))):
            with self.assertRaises(ValueError): Normalizer(mean,scale)

    def test_optional_control_gate_preserves_rule_scores_without_control(self):
        torch.manual_seed(3)
        f,t=sample(); norm=Normalizer.fit([flatten_features(f,t)])
        model=TemporalModel(channels=2)
        f.values['ball_available'][:8]=0
        f.values['control_support'][8:16]=0
        base=Evidence(np.tile([.7,.1,.2],(24,1)).astype(np.float32),np.zeros((24,2)),np.zeros(24),np.zeros(24))
        gated=infer_evidence(model,f,t,base,norm,control_gate=True)
        ordinary=infer_evidence(model,f,t,base,norm)
        np.testing.assert_array_equal(gated.state_scores[:16],base.state_scores[:16])
        np.testing.assert_allclose(gated.state_scores[16:],ordinary.state_scores[16:])
        np.testing.assert_array_equal(gated.state_scores[:,2],base.state_scores[:,2])
        np.testing.assert_array_equal(gated.change,base.change)

    def test_chunk_size_does_not_move_boundaries(self):
        torch.manual_seed(4)
        f,t=sample(600)
        f.values['nearest_distance'][:,0]=np.linspace(0,10,600)
        f.values['nearest_distance'][:,1]=np.linspace(10,0,600)
        norm=Normalizer.fit([flatten_features(f,t)]); model=TemporalModel(channels=2)
        base=Evidence(np.tile([.4,.4,.2],(600,1)).astype(np.float32),np.zeros((600,2)),np.zeros(600),np.zeros(600))
        full=infer_evidence(model,f,t,base,norm,center_frames=600)
        chunked=infer_evidence(model,f,t,base,norm,center_frames=80)
        np.testing.assert_allclose(full.state_scores,chunked.state_scores,atol=1e-6)

def swap_teams_normalized(x):
    from possession.model import TEAM_PAIRS
    y=x.copy()
    for offset in (0,N_RAW):
        for i in TEAM_PAIRS: y[:,[offset+i,offset+i+1]]=y[:,[offset+i+1,offset+i]]
    return y

if __name__=='__main__': unittest.main()
