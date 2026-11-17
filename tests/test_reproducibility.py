import tempfile
import unittest
from pathlib import Path
import numpy as np
from scripts.check_reproduction import sha256
from src.possession.diagnostics import summarize


class ReproductionSupportTest(unittest.TestCase):
    def test_sha256_changes_with_bytes(self):
        Path('data/runs').mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir='data/runs') as d:
            p = Path(d) / 'synthetic.csv'
            p.write_bytes(b'frame,label\n0,H\n')
            first = sha256(p)
            self.assertEqual(first, sha256(p))
            p.write_bytes(b'frame,label\n0,A\n')
            self.assertNotEqual(first, sha256(p))

    def test_aggregate_diagnostics(self):
        report = summarize(np.array(list('XHHDAA')), 2, {'missing': [0, 1, 0, 0, 1, 0]})
        self.assertEqual(report['counts'], {'H': 2, 'A': 2, 'D': 1, 'X': 1})
        self.assertEqual(report['seconds']['H'], 1)
        self.assertEqual(report['transitions'], 3)
        self.assertEqual(report['quality_counts']['missing'], 2)
        self.assertEqual(summarize([])['transitions'], 0)

if __name__ == '__main__':
    unittest.main()
