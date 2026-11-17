import unittest
import numpy as np

from possession.contracts import Evidence
from scripts.compare_rule_components import live_blocks, existing_team_for_rule_based_dead


class ComponentFusionTest(unittest.TestCase):
    def test_dead_mask_and_block_boundaries_are_preserved(self):
        dead = np.array([False, False, True, True, False, False, False, False])
        blocks = [(0, 5), (5, 7)]
        self.assertEqual(live_blocks(dead, blocks), [(0, 2), (4, 5), (5, 7)])
        scores = np.tile([.8, .1, .1], (len(dead), 1))
        evidence = Evidence(scores, np.zeros((len(dead), 2)), np.zeros(len(dead)), np.zeros(len(dead)))
        labels = existing_team_for_rule_based_dead(evidence, dead, blocks, {})
        np.testing.assert_array_equal(labels, ['H', 'H', 'D', 'D', 'H', 'H', 'H', 'X'])


if __name__ == '__main__':
    unittest.main()
