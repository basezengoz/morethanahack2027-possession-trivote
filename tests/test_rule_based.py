import unittest

from src.possession.rule_based import dead_ball, model


class TestRuleBased(unittest.TestCase):
    """Synthetic checks of the rule-based model. No match data needed."""

    def test_dead_ball_selftest(self):
        dead_ball.selftest()

    def test_model_selftest(self):
        model.selftest()


if __name__ == "__main__":
    unittest.main()
