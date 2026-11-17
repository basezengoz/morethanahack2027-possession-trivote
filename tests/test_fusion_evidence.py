import unittest
import numpy as np
from possession.contracts import Evidence
from possession.rule_fusion import combine_evidence, cues_from_rule_features
from scripts.select_rule_fusion import select


class FusionEvidenceTests(unittest.TestCase):
    def test_selection_rejects_large_single_match_regression(self):
        def report(index):
            values = {'existing': .5, 'pass': .7 if index != 2 else .4,
                      'restart': .51, 'pass_restart': .5}
            return {'match': str(index), 'source_sha256': {'source': 'same'},
                    'results': {view: {name: {'spell_f1': {'mean_f1': value}}
                                       for name, value in values.items()}
                                for view in ('clean', 'masked')}}
        result = select([report(i) for i in range(3)])
        self.assertEqual(result['selected'], 'restart')
        self.assertNotIn('pass', result['eligible'])

    def test_bounded_cues_favor_expected_state_without_mutation(self):
        base = Evidence(np.full((4, 3), 1 / 3), np.zeros((4, 2)), np.zeros(4), np.zeros(4))
        passes = np.array([[1, 0], [0, 1], [0, 0], [0, 0]])
        restart = np.array([0, 0, 1, 0])
        result = combine_evidence(base, passes, restart)
        self.assertGreater(result.state_scores[0, 0], result.state_scores[0, 1])
        self.assertGreater(result.state_scores[1, 1], result.state_scores[1, 0])
        self.assertGreater(result.state_scores[2, 2], result.state_scores[2, 0])
        np.testing.assert_allclose(result.state_scores[3], base.state_scores[3])
        np.testing.assert_allclose(result.state_scores.sum(axis=1), 1)
        np.testing.assert_allclose(base.state_scores, 1 / 3)
        with self.assertRaises(ValueError):
            combine_evidence(base, passes * 2, restart)

    def test_pass_windows_stop_at_block_end_and_conflict_is_neutral(self):
        f = {'passes': [(0, 1, 0), (0, 3, 1)],
             'out': np.array([0, 0, 0, 1, 0]), 'setpiece': np.zeros(5)}
        passes, restart = cues_from_rule_features(f, 5, hold_frames=3)
        np.testing.assert_array_equal(passes, [[0, 0], [1, 0], [1, 0], [0, 0], [0, 1]])
        np.testing.assert_array_equal(restart, [0, 0, 0, 1, 0])


if __name__ == '__main__':
    unittest.main()
