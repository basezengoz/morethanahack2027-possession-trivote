import csv
import tempfile
import unittest
from pathlib import Path
import numpy as np
from possession.external_data import load_metrica_csv, proxy_from_events, split_match_ids


def write_tracking(path, rows, ball_override=None):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['Team','','','Home','','Ball',''])
        w.writerow(['','','','1','','Ball',''])
        w.writerow(['Period','Frame','Time [s]','Player1','','Ball',''])
        for period, frame, time in rows:
            ball = ball_override or (.4,.5)
            w.writerow([period,frame,time,.1,.2,*ball])


def event(team='Home', code='PASS', subtype='', period=1, start=1, end=15, receiver='p'):
    return {'Team':team,'Type':code,'Subtype':subtype,'Period':str(period),
            'Start Frame':str(start),'End Frame':str(end),'To':receiver}

class ExternalDataTest(unittest.TestCase):
    def test_proxy_masks_unobserved_boundaries_and_conflicts(self):
        lookup = {(1,i):i-1 for i in range(1,41)}
        events = [event(),event('Away',start=8,end=18),event('Home','SHOT',start=25,end=30),
                  event('Away',start=32,end=36,receiver='')]
        labels, mask, boundary = proxy_from_events(events,lookup,40,25,boundary_margin_frames=2)
        self.assertEqual(labels[3],0)
        self.assertEqual(labels[9],-1)
        self.assertFalse(mask[0]); self.assertFalse(mask[24:30].any()); self.assertFalse(boundary.any())
        self.assertEqual(labels.dtype, np.int8)

    def test_long_gap_and_unknown_event_never_filled(self):
        lookup = {(1,i):i-1 for i in range(1,101)}
        events = [event(start=1,end=10),event(start=90,end=100),event('Away',start=20,end=80)]
        labels, mask, _ = proxy_from_events(events,lookup,100,25,2,20)
        self.assertTrue(mask.any()); self.assertFalse(mask[20:80].any()); self.assertTrue((labels[20:80] == -1).all())

    def test_split_matches_prevents_leakage(self):
        self.assertEqual(split_match_ids(['a','b','c'],['c']), (['a','b'],['c']))
        for matches, validation in [(['a','a'],['a']),(['a'],['a']),(['a','b'],['z']),(['a','b'],[])]:
            with self.assertRaises(ValueError): split_match_ids(matches,validation)

    def test_adapter_blocks_alignment_and_provenance(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d); rows=[(1,i,i/25) for i in range(1,21)]+[(2,i,i/25) for i in range(1,21)]
            write_tracking(p/'h.csv',rows); write_tracking(p/'a.csv',rows)
            with open(p/'e.csv','w',newline='',encoding='utf-8') as f:
                e=event(start=1,end=15); w=csv.DictWriter(f,fieldnames=e.keys()); w.writeheader(); w.writerow(e)
            result=load_metrica_csv('synthetic',p/'h.csv',p/'a.csv',p/'e.csv',boundary_margin_frames=2)
            self.assertEqual(result.tracking.blocks,[(0,20),(20,40)])
            self.assertEqual(result.tracking.slot_team.tolist(),[0,1])
            self.assertEqual(result.tracking.players_xy.shape,(40,2,2))
            self.assertTrue(result.supervision_mask.any()); self.assertFalse(result.boundary_mask.any())
            self.assertEqual(set(result.provenance['sha256']),{'home','away','events'})
            write_tracking(p/'a.csv',rows[:-1])
            with self.assertRaises(ValueError): load_metrica_csv('synthetic',p/'h.csv',p/'a.csv',p/'e.csv')

    def test_boundary_blocks_and_ball_conflict(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d); rows=[(1,1,.04),(1,3,.12),(2,1,.04)]
            write_tracking(p/'h.csv',rows); write_tracking(p/'a.csv',rows)
            (p/'e.csv').write_text('Team,Type,Subtype,Period,Start Frame,End Frame,To\n',encoding='utf-8')
            r=load_metrica_csv('x',p/'h.csv',p/'a.csv',p/'e.csv')
            self.assertEqual(r.tracking.blocks,[(0,1),(1,2),(2,3)])
            write_tracking(p/'a.csv',rows,(.8,.8))
            with self.assertRaises(ValueError): load_metrica_csv('x',p/'h.csv',p/'a.csv',p/'e.csv')

    def test_reversed_interval_and_missing_observation_are_safe(self):
        lookup = {(1,i):i-1 for i in range(1,31)}
        labels, mask, _ = proxy_from_events([event(start=1,end=20),event(start=14,end=8)],lookup,30,25,2)
        self.assertFalse(mask[7:14].any())
        self.assertTrue((labels[7:14] == -1).all())
        with tempfile.TemporaryDirectory() as d:
            p=Path(d); rows=[(1,1,.04),(1,2,.08)]
            write_tracking(p/'h.csv',rows); write_tracking(p/'a.csv',rows)
            text=(p/'h.csv').read_text(); text=text.replace('0.1,0.2','NaN,NaN')
            (p/'h.csv').write_text(text)
            (p/'e.csv').write_text('Team,Type,Subtype,Period,Start Frame,End Frame,To\n')
            m=load_metrica_csv('x',p/'h.csv',p/'a.csv',p/'e.csv')
            self.assertFalse(m.tracking.player_observed[:,0].any())

    def test_fetch_import_is_offline_and_revision_is_pinned(self):
        import importlib.util
        import json
        from unittest.mock import patch
        script = Path(__file__).resolve().parents[1] / 'scripts' / 'fetch_external_data.py'
        spec = importlib.util.spec_from_file_location('fetch_external_test', script)
        module = importlib.util.module_from_spec(spec)
        with patch('urllib.request.urlopen', side_effect=AssertionError('unexpected network')):
            spec.loader.exec_module(module)
        self.assertEqual(module.select_revision(), module.DEFAULT_REVISION)
        with tempfile.TemporaryDirectory() as d:
            manifest = Path(d) / 'manifest.json'
            manifest.write_text(json.dumps({'revision':'a'*40}),encoding='utf-8')
            self.assertEqual(module.select_revision(None, manifest), 'a'*40)
            self.assertEqual(module.select_revision('b'*40, manifest), 'b'*40)
            with self.assertRaises(ValueError): module.select_revision('master', manifest)
