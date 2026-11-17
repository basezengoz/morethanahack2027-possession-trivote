import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np

from possession import predict
from scripts.rule_based import pipeline, overlay_subs


class PipelineSelectionTest(unittest.TestCase):
    def test_pipeline_default_and_v4_forwarding(self):
        for arguments, model in [([], 'v6'), (['--tcn-model', 'v4'], 'v4'),
                                 (['--tcn-model', 'v5'], 'v5'),
                                 (['--method', 'old_rule_based'], 'v6')]:
            with patch.object(pipeline.subprocess, 'run') as run, patch.object(pipeline.overlay_subs, 'main', return_value=0) as overlay:
                self.assertEqual(pipeline.main(arguments), 0)
            command = run.call_args.args[0]
            self.assertEqual(command[command.index('--tcn-model') + 1], model)
            self.assertEqual(command[command.index('--method') + 1],
                             'old_rule_based' if '--method' in arguments else 'rule_based_tcn')
            self.assertEqual(overlay.call_args.args[0][0], '--pred')

    def test_overlay_never_reads_gt_by_default(self):
        with tempfile.TemporaryDirectory(dir='data/runs') as tmp:
            pred = Path(tmp) / 'test.csv'; out = Path(tmp) / 'test.ass'
            pred.write_text('frame,label\n0,H\n1,A\n2,D\n', encoding='utf-8')
            m = SimpleNamespace(n_frames=3, ball_seen=np.ones(3, bool))
            with patch.object(overlay_subs, 'truth_layer', side_effect=AssertionError('GT read')), \
                 patch.object(overlay_subs, 'panel_layer', return_value=['panel']):
                overlay_subs.write_overlay(m, [], [], pred, out)
            self.assertTrue(out.is_file())

    def test_overlay_rejects_shuffled_timeline(self):
        with tempfile.TemporaryDirectory(dir='data/runs') as tmp:
            pred = Path(tmp) / 'test.csv'
            pred.write_text('frame,label\n0,H\n2,A\n1,D\n', encoding='utf-8')
            with self.assertRaises(ValueError):
                overlay_subs.panel_layer(SimpleNamespace(n_frames=3), pred)

    def test_model_presets_have_distinct_hashes(self):
        self.assertNotEqual(predict.MODELS['v4'][1], predict.MODELS['v5'][1])
        self.assertIn('dfl_v4_missing100.pt', predict.MODELS['v4'][0])
        self.assertIn('dfl_v5_observation_pattern.pt', predict.MODELS['v5'][0])
        self.assertIn('dfl_v6.pt', predict.MODELS['v6'][0])
        self.assertNotEqual(predict.MODELS['v6'][1], predict.MODELS['v5'][1])

    def test_frozen_text_hash_tolerates_windows_line_endings(self):
        with tempfile.TemporaryDirectory(dir='data/runs') as tmp:
            a, b = Path(tmp) / 'lf', Path(tmp) / 'crlf'
            a.write_bytes(b'first\nsecond\n'); b.write_bytes(b'first\r\nsecond\r\n')
            self.assertEqual(predict.text_sha256(a), predict.text_sha256(b))


if __name__ == '__main__': unittest.main()
