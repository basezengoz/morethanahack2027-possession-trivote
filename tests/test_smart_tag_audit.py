import unittest
import numpy as np
from scripts.audit_smart_tagging import summarize, zero_tags


class SmartTagAuditTest(unittest.TestCase):
    def test_counts_are_scored_per_half_and_differences_have_direction(self):
        on = np.array(list('XHHDHAX'))
        off = np.array(list('XHADHAX'))
        report = summarize(on, off, [(1, 4), (4, 6)])
        self.assertEqual(report['graded_frames'], 5)
        self.assertEqual(report['changed_frames'], 1)
        self.assertEqual(report['on_to_off']['H_to_A'], 1)
        self.assertEqual(report['tags_on']['possessions'], 3)
        self.assertEqual(report['tags_off']['possessions'], 4)

    def test_zero_tags_remove_team_restart_and_conflict(self):
        tags = zero_tags(3)
        self.assertEqual(tags.team_support.shape, (3, 2))
        self.assertFalse(tags.team_support.any() or tags.restart.any() or tags.conflict.any())


if __name__ == '__main__':
    unittest.main()
