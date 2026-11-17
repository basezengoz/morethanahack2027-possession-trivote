import unittest
from src.possession.tagging import tagging_from_events

class TaggingTest(unittest.TestCase):
    def test_conflict_defense_restart_unknown(self):
        events = [dict(code='BUILD UP',start_frame=0,end_frame=3,tags={'Team':'Barcelona'}),
                  dict(code='DEFENDING IN MIDDLE THIRD',start_frame=1,end_frame=2,tags={'Team':'Barcelona'}),
                  dict(code='SET PIECES',start_frame=2,end_frame=4,tags={'Team':'N/A'})]
        x = tagging_from_events(events, 5, 25, edge_frames=0)
        self.assertEqual(x.conflict.tolist(), [0,1,0,0,0])
        self.assertEqual(x.restart.tolist(), [0,0,1,1,0])
        self.assertEqual(x.team_support[1].tolist(), [.25,.25])

    def test_edges_unknown_and_na(self):
        events = [dict(code='BUILD UP',start_frame=0,end_frame=5,tags={'Team':'Barcelona'}),
                  dict(code='UNKNOWN',start_frame=0,end_frame=5,tags={'Team':'Athletic Club'}),
                  dict(code='BUILD UP',start_frame=0,end_frame=5,tags={'Team':'N/A'})]
        x = tagging_from_events(events, 5, 25, edge_frames=2)
        self.assertLess(x.team_support[0,0], x.team_support[2,0])
        self.assertFalse(x.team_support[:,1].any())
