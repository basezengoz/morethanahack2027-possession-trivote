import tempfile
import unittest
from pathlib import Path
import numpy as np
from possession.dfl_data import load_dfl


class DflDataTest(unittest.TestCase):
    def test_alignment_and_dead_ball_precedence(self):
        info = '<root><General MatchId="m" HomeTeamId="h" GuestTeamId="a"/></root>'
        positions = '''<root><PitchSize X="100" Y="60"/>
        <FrameSet MatchId="m" GameSection="firstHalf" TeamId="h" PersonId="p">
        <Frame N="10" X="0" Y="0"/><Frame N="12" X="10" Y="0"/></FrameSet>
        <FrameSet MatchId="m" GameSection="firstHalf" TeamId="a" PersonId="q">
        <Frame N="10" X="-10" Y="0"/></FrameSet>
        <FrameSet MatchId="m" GameSection="firstHalf" TeamId="BALL" PersonId="b">
        <Frame N="10" T="2022-01-01T00:00:00.000+00:00" X="0" Y="0" BallStatus="1" BallPossession="1"/>
        <Frame N="11" T="2022-01-01T00:00:00.040+00:00" X="1" Y="0" BallStatus="0" BallPossession="2"/>
        <Frame N="12" T="2022-01-01T00:00:00.080+00:00" X="2" Y="0" BallStatus="1" BallPossession="2"/></FrameSet></root>'''
        with tempfile.TemporaryDirectory(dir='data/runs') as d:
            i, p = Path(d)/'info.xml', Path(d)/'positions.xml'
            i.write_text(info); p.write_text(positions)
            match = load_dfl(p, i)
            p.write_text(positions.replace('00:00:00.080', '00:00:01.080'))
            with self.assertRaisesRegex(ValueError, 'timestamps disagree'):
                load_dfl(p, i)
        np.testing.assert_array_equal(match.labels, [0, 2, 1])
        self.assertEqual(match.tracking.blocks, [(0, 3)])
        self.assertTrue(np.isnan(match.tracking.players_xy[1]).all())
        np.testing.assert_allclose(match.tracking.players_xy[0,0], [.5,.5])
        np.testing.assert_array_equal(match.tracking.slot_team, [0,1])


if __name__ == '__main__': unittest.main()
