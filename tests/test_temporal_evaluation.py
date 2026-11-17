import unittest
import importlib.util
from unittest.mock import patch
import numpy as np
from tests.test_external_evaluation import fixture
HAS_TORCH = importlib.util.find_spec('torch') is not None
if HAS_TORCH:
    from scripts.evaluate_temporal import validate_split, compare
from src.possession.contracts import Evidence


@unittest.skipUnless(HAS_TORCH, 'optional torch extra not installed')
class TemporalEvaluationTest(unittest.TestCase):
    def test_split_validation_blocks_overlap_and_wrong_assignment(self):
        ckpt = {'train_matches': ['1'], 'validation_matches': ['2']}
        validate_split('2', 'validation', ckpt)
        validate_split('3', 'test', ckpt)
        for match_id, split in [('1', 'validation'), ('1', 'test'), ('3', 'train')]:
            with self.assertRaises(ValueError): validate_split(match_id, split, ckpt)
        with self.assertRaises(ValueError):
            validate_split('1', 'train', {'train_matches': ['1'], 'validation_matches': ['1']})

    def test_compare_preserves_d_and_transition_guards(self):
        m = fixture([0]*30 + [1]*30)
        checkpoint = {'stats': {'dead_head_trained': False}}
        def identity(model, features, tags, baseline, norm, device, dead_head_trained):
            return baseline
        with patch('scripts.evaluate_temporal.infer_evidence', side_effect=identity):
            r = compare(m, {}, None, None, checkpoint)
        self.assertEqual(r['labels_changed_frames'], 0)
        self.assertEqual(r['D_membership_changed_frames'], 0)
        self.assertEqual(r['A_geometry_proxy'], r['temporal_proxy'])
        self.assertEqual(r['rule_based_geometry_proxy'], r['A_geometry_proxy'])
        def tamper(model, features, tags, baseline, norm, device, dead_head_trained):
            scores = baseline.state_scores.copy(); scores[:,2] = 0
            return Evidence(scores, baseline.change, baseline.stop, baseline.restart)
        with patch('scripts.evaluate_temporal.infer_evidence', side_effect=tamper):
            with self.assertRaises(ValueError): compare(m, {}, None, None, checkpoint)

if __name__ == '__main__':
    unittest.main()
