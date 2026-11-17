import importlib.util
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import games

HAS_TORCH = importlib.util.find_spec('torch') is not None
if HAS_TORCH:
    from scripts import predict_temporal as target
    from possession.contracts import Evidence, FeatureData, TaggingEvidence


@unittest.skipUnless(HAS_TORCH, 'optional training dependency is not installed')
class TargetTemporalTest(unittest.TestCase):
    def test_teacher_real_tags_model_zero_tags_and_official_blocks(self):
        start = next(s for h,s,e in games.half_video_blocks(games.GAME) if h == 2)
        kickoff = games.anchor_video_frame(games.GAMES[games.GAME]['kickoff_2nd'])
        n = kickoff + 1
        tracking = SimpleNamespace(frame=np.arange(n))
        tags = TaggingEvidence(np.ones((n,2),np.float32), np.ones(n,np.float32), np.ones(n,np.float32))
        features = FeatureData({}, [(2,5),(start,n)])
        baseline = Evidence(np.tile([.9,.05,.05],(n,1)).astype(np.float32),
                            np.zeros((n,2),np.float32),np.zeros(n,np.float32),np.zeros(n,np.float32))
        def fake_infer(model, actual_features, model_tags, evidence, normalizer, device, dead_head_trained):
            self.assertIs(actual_features,features); self.assertIs(evidence,baseline)
            self.assertEqual(np.count_nonzero(model_tags.team_support),0)
            self.assertEqual(np.count_nonzero(model_tags.restart),0)
            self.assertEqual(np.count_nonzero(model_tags.conflict),0)
            self.assertFalse(dead_head_trained)
            return evidence
        with patch.object(target,'clean_tracking',return_value='clean') as cleaner, \
             patch.object(target,'build_features',return_value=features), \
             patch.object(target,'build_evidence',return_value=baseline) as teacher, \
             patch.object(target,'infer_evidence',side_effect=fake_infer):
            labels=target.run_temporal(tracking,tags,{},None,None,False)
        self.assertIs(cleaner.call_args.kwargs['restart'],tags.restart)
        self.assertIs(teacher.call_args.args[1],tags)
        self.assertEqual(labels[0],'X'); self.assertEqual(labels[2],'H'); self.assertEqual(labels[5],'X')
        self.assertEqual(labels[start],'D'); self.assertEqual(labels[kickoff-1],'D'); self.assertEqual(labels[kickoff],'H')

    def test_changed_dead_or_transition_rejected(self):
        tracking=SimpleNamespace(frame=np.arange(3))
        tags=TaggingEvidence(np.zeros((3,2)),np.zeros(3),np.zeros(3))
        feat=FeatureData({},[(0,3)])
        base=Evidence(np.tile([.5,.3,.2],(3,1)),np.zeros((3,2)),np.zeros(3),np.zeros(3))
        for key in ('dead','stop'):
            other=Evidence(base.state_scores.copy(),base.change.copy(),base.stop.copy(),base.restart.copy())
            if key=='dead': other.state_scores[:,2]=.4
            else: other.stop[0]=1
            with patch.object(target,'clean_tracking'),patch.object(target,'build_features',return_value=feat), \
                 patch.object(target,'build_evidence',return_value=base),patch.object(target,'infer_evidence',return_value=other):
                with self.assertRaises(ValueError): target.run_temporal(tracking,tags,{},None,None,False)

if __name__=='__main__': unittest.main()
