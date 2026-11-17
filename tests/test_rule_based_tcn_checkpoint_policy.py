import unittest
from scripts.predict_rule_based_tcn import checkpoint_policy


class CheckpointPolicyTest(unittest.TestCase):
    def test_tested_checkpoint_and_explicit_alternative(self):
        self.assertEqual(checkpoint_policy('abc', 'abc', None), 'experiment_checkpoint')
        self.assertEqual(checkpoint_policy('def', 'abc', 'DEF'), 'user_selected_untested_combination')

    def test_reject_implicit_or_wrong_checkpoint(self):
        with self.assertRaises(ValueError): checkpoint_policy('def', 'abc', None)
        with self.assertRaises(ValueError): checkpoint_policy('def', 'abc', 'abc')
