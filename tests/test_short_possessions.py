import unittest
import importlib.util
import numpy as np
HAS_TORCH=importlib.util.find_spec('torch') is not None
if HAS_TORCH:
    from scripts.analyze_short_possessions import duration_recall


@unittest.skipUnless(HAS_TORCH, 'optional training dependency is not installed')
class ShortPossessionTest(unittest.TestCase):
    def test_short_spells_and_region_edges(self):
        y=np.array([2]*5+[0]*25+[2]*5+[1]*75+[2]*5)
        p=np.array(['D']*5+['H']*25+['D']*5+['A']*75+['D']*5)
        result=duration_recall(y,p,np.ones(len(y),bool),[(0,len(y))])
        self.assertEqual(result['duration_bins']['under_2s'],{'support':1,'matched':1,'recall':1.})
        self.assertEqual(result['duration_bins']['2_to_5s'],{'support':1,'matched':1,'recall':1.})
        result=duration_recall(y,p,np.ones(len(y),bool),[(5,30)])
        self.assertEqual(result['reference_spells'],0)


if __name__=='__main__': unittest.main()
