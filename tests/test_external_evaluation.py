import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
from src.possession.contracts import TrackingData
from src.possession.external_evaluation import evaluate_external, mask_ball, robustness


def fixture(labels, supervised=None, boundary=None, blocks=None):
    y = np.asarray(labels, dtype=np.int8); n = len(y)
    ball = np.tile([.5, .5], (n, 1)).astype(np.float32)
    players = np.tile([[[.5, .5], [.8, .8]]], (n, 1, 1)).astype(np.float32)
    t = TrackingData(np.arange(n), players, ball, np.ones((n, 2), bool), np.ones(n, bool),
                     np.array([0, 1]), 25, (100, 50), blocks or [(0, n)])
    return SimpleNamespace(tracking=t, labels=y, supervision_mask=np.asarray(y >= 0 if supervised is None else supervised),
                           boundary_mask=np.asarray(np.zeros(n, bool) if boundary is None else boundary))


class ExternalEvaluationTest(unittest.TestCase):
    def test_masked_metrics_and_absent_temporal_truth(self):
        match = fixture([0, 0, 1, 1, 2, 2, -1])
        r = evaluate_external(match, list('HAAADDH'))
        self.assertAlmostEqual(r['macro_f1'], (2/3 + .8 + 1) / 3)
        self.assertEqual(r['classes']['D']['precision'], 1)
        self.assertEqual(r['known_frames'], 6)
        self.assertIsNone(r['boundary'])
        self.assertIsNone(r['spell_f1'])
        no_dead = evaluate_external(fixture([0, 0, 1, 1]), list('HHAA'))
        self.assertAlmostEqual(no_dead['macro_f1'], 2/3)
        self.assertEqual(no_dead['live_macro_f1'], 1)
        no_dead_with_prediction = evaluate_external(fixture([0, 0, 1, 1]), list('HDAA'))
        self.assertIsNone(no_dead_with_prediction['classes']['D']['precision'])
        self.assertIsNone(no_dead_with_prediction['classes']['D']['recall'])

    def test_complete_spells_only_and_one_to_one_boundary(self):
        match = fixture([2]*3 + [0]*4 + [2]*3 + [1]*4 + [2]*3, boundary=[1]*17)
        r = evaluate_external(match, list('DDDHHHHDDDAAAADDD'), tolerance=1)
        self.assertEqual(r['boundary']['tp'], 4)
        self.assertEqual(r['boundary']['precision'], 1)
        self.assertEqual(r['spell_f1']['reference'], 2)
        self.assertEqual(r['spell_f1']['mean_f1'], 1)
        clipped = fixture([0]*4 + [-1]*3 + [1]*4, boundary=[1]*11)
        r = evaluate_external(clipped, list('HHHHXXXAAAA'), tolerance=0)
        self.assertIsNone(r['spell_f1'])

    def test_temporal_abstention_fills_only_within_reliable_region(self):
        match = fixture([2, 0, 0, 0, 2], boundary=[1]*5)
        result = evaluate_external(match, list('DHXHD'), tolerance=0)
        self.assertEqual(result['spell_f1']['mean_f1'], 1)
        self.assertEqual(result['boundary']['tp'], 2)
        self.assertLess(result['macro_f1'], 2/3)
        all_x = evaluate_external(match, list('XXXXX'), tolerance=0)
        self.assertEqual(all_x['spell_f1']['mean_f1'], 0)
        self.assertEqual(all_x['boundary']['predicted'], 0)
        separated = fixture([2, 0, 0, 2, -1, 2, 1, 1, 2], boundary=[1]*9)
        result = evaluate_external(separated, list('DHXDXDAXD'), tolerance=0)
        self.assertEqual(result['spell_f1']['mean_f1'], 1)

    def test_reject_unknown_supervision_and_zero_coverage(self):
        with self.assertRaises(ValueError):
            evaluate_external(fixture([-1], supervised=[True]), ['H'])
        self.assertIsNone(evaluate_external(fixture([-1]), ['H'])['macro_f1'])

    def test_cli_outputs_only_aggregates_and_tags_unavailable(self):
        from scripts.evaluate_external import main
        m = fixture([0] * 50)
        m.match_id, m.provenance = 'synthetic', {'source': 'synthetic'}
        Path('data/runs').mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir='data/runs') as d:
            out = Path(d) / 'report.json'
            with patch('scripts.evaluate_external.load_metrica_csv', return_value=m):
                main(['--match-id', 'synthetic', '--home', 'h', '--away', 'a', '--events', 'e',
                      '--split', 'validation', '--output', str(out)])
            report = json.loads(out.read_text(encoding='utf-8'))
            self.assertEqual(report['proxy_metrics']['known_frames'], 50)
            self.assertFalse(report['A_plus_tags']['available'])
            self.assertEqual(report['rule_based_plus_tags'], report['A_plus_tags'])
            self.assertIn('robustness_not_accuracy', report)
            self.assertNotIn('predictions', report)
            self.assertIn('src/possession/teacher.py', report['source']['files_sha256'])
            self.assertEqual(report['arguments']['split'], 'validation')

    def test_mask_reproducible_and_nonmutating(self):
        m = fixture([0]*100, blocks=[(0, 50), (50, 100)])
        raw = m.tracking.ball_xy_raw.copy()
        first, drop = mask_ball(m.tracking, seed=9, fraction=.5, block_frames=10)
        second, drop2 = mask_ball(m.tracking, seed=9, fraction=.5, block_frames=10)
        np.testing.assert_array_equal(drop, drop2)
        np.testing.assert_array_equal(m.tracking.ball_xy_raw, raw)
        self.assertTrue(np.isnan(first.ball_xy_raw[drop]).all())
        self.assertFalse(first.ball_observed[drop].any())
        self.assertEqual(robustness(m.tracking, {}, seed=9, fraction=.5, block_frames=10),
                         robustness(m.tracking, {}, seed=9, fraction=.5, block_frames=10))

if __name__ == '__main__':
    unittest.main()
