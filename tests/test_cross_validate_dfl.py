import unittest
from scripts.cross_validate_dfl import build_folds


class CrossValidationTest(unittest.TestCase):
    def test_every_match_is_test_once_and_splits_are_disjoint(self):
        ids=('a','b','c','d')
        folds=build_folds(ids)
        self.assertEqual([f['test'] for f in folds],list(ids))
        for f in folds:
            self.assertEqual(set(f['train']) | {f['validation'],f['test']},set(ids))
            self.assertFalse(set(f['train']) & {f['validation'],f['test']})
            self.assertNotEqual(f['test'],f['validation'])
        with self.assertRaises(ValueError): build_folds(('a','a','b'))


if __name__=='__main__': unittest.main()
