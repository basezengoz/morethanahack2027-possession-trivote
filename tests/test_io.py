import tempfile
import unittest
from pathlib import Path
import numpy as np
from src.possession.io import _read_atd, _teams

class IOTest(unittest.TestCase):
    def test_alignment_and_slots(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'atd.txt'
            p.write_text('1:0.1,0.2,0;NaN,NaN,NaN:0.2,0.3,0\n2:NaN,NaN,NaN;0.4,0.5,0:NaN,NaN,NaN\n')
            a = _read_atd(p, np.array([0, 1]), 2, 25, (105.3,68), [(0,2)])
            self.assertEqual(a.frame.tolist(), [0,1])
            self.assertEqual(a.player_observed.sum(), 2)
            for bad in ['1:0.1,0.2,0:0.2,0.3,0\n', '2:0.1,0.2,0;NaN,NaN,NaN:0.2,0.3,0\n']:
                p.write_text(bad)
                with self.assertRaises(ValueError): _read_atd(p, np.array([0,1]), 1, 25, (105,68), [(0,1)])
    def test_unknown_metadata_team(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'meta.xml'
            p.write_text('<r><Team id="a"/><Team id="b"/><Player teamId="c"/></r>')
            with self.assertRaises(ValueError): _teams(p)

    def test_outside_pitch_still_valid_normalized_scale(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'atd.txt'
            p.write_text('1:1.01,0.2,0;NaN,NaN,NaN:-0.01,0.3,0\n')
            a = _read_atd(p, np.array([0,1]), 1, 25, (105,68), [(0,1)])
            self.assertTrue(a.ball_observed[0])
            self.assertLess(a.ball_xy_raw[0,0], 0)

    def test_metre_coordinates_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'atd.txt'
            p.write_text('1:50,34,0;NaN,NaN,NaN:0.2,0.3,0\n')
            with self.assertRaises(ValueError): _read_atd(p, np.array([0,1]), 1, 25, (105,68), [(0,1)])
