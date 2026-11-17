import unittest
import numpy as np
from src.possession.contracts import Evidence
from src.possession.decode import decode_labels


def evidence(rows):
    scores = np.asarray(rows, dtype=float)
    n = len(scores)
    return Evidence(scores, np.zeros((n, 2)), np.zeros(n), np.zeros(n))

H, A, D = [0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.05, 0.05, 0.9]

class TestDecode(unittest.TestCase):
    def test_h_d_h_and_first_supported_boundary(self):
        ev = evidence([H]*10 + [D]*8 + [H]*10)
        self.assertEqual(''.join(decode_labels(ev, [(0, 28)])), 'H'*10+'D'*8+'H'*10)

    def test_short_touch_and_short_takeover(self):
        ev = evidence([H]*12 + [A] + [H]*12)
        self.assertEqual(''.join(decode_labels(ev, [(0, 25)])), 'H'*25)
        ev = evidence([H]*12 + [A]*4 + [H]*12)
        self.assertEqual(''.join(decode_labels(ev, [(0, 28)])), 'H'*12+'A'*4+'H'*12)

    def test_missing_ball_neutral_preserves_state(self):
        ev = evidence([A]*10 + [[.49, .49, .02]]*50 + [A]*10)
        self.assertEqual(set(decode_labels(ev, [(0, 70)])), {'A'})

    def test_reset_unscored_and_tie(self):
        ev = evidence([A]*8 + [[1, 1, 1]]*4 + [H]*8)
        labels = decode_labels(ev, [(0, 8), (12, 20)])
        self.assertEqual(''.join(labels), 'A'*8+'X'*4+'H'*8)
        self.assertEqual(''.join(decode_labels(evidence([[1, 1, 1]]*4), [(0, 4)])), 'H'*4)

    def test_numerical_and_validation(self):
        ev = evidence([[1e-300, 1, 0]]*5)
        self.assertEqual(''.join(decode_labels(ev, [(0, 5)])), 'A'*5)
        with self.assertRaises(ValueError): decode_labels(ev, [(0, 4), (3, 5)])
        ev.state_scores[0, 0] = np.nan
        with self.assertRaises(ValueError): decode_labels(ev, [(0, 5)])

    def test_large_finite_weights_are_scale_invariant(self):
        ev = evidence([[3,4,3]]*50+[[4,3,3]]*50)
        expected = decode_labels(ev, [(0,100)])
        self.assertEqual(''.join(expected), 'A'*50+'H'*50)
        ev.state_scores *= 4e307
        with np.errstate(over='raise', invalid='raise'):
            np.testing.assert_array_equal(decode_labels(ev, [(0,100)]), expected)

    def test_uninformative_gap_cannot_move_later_restart_to_its_start(self):
        ev=evidence([D]*10+[[1,1,1]]*40+[H]*10)
        self.assertEqual(''.join(decode_labels(ev,[(0,60)])), 'D'*50+'H'*10)
        ev=evidence([H]*10+[[1,1,1]]*40+[A]*10)
        self.assertEqual(''.join(decode_labels(ev,[(0,60)])), 'H'*50+'A'*10)

if __name__ == '__main__': unittest.main()
