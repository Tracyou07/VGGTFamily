"""Raw prediction diagnostics must retain signed, unrounded differences."""
import unittest
import numpy as np

from experiments.ours_v10.compare_raw import field_stats


class RawDifferenceTest(unittest.TestCase):
    def test_finite_difference_and_exact_are_separate(self):
        a=np.array([[1.,2.],[3.,4.]],dtype=np.float32)
        b=a.copy();b[0,0]+=0.125
        stats=field_stats(a,b)
        self.assertFalse(stats["exact"])
        self.assertTrue(stats["finite"])
        self.assertEqual(stats["max_abs"],0.125)
        self.assertEqual(stats["mean_abs"],0.03125)
        self.assertGreater(stats["relative_l2"],0)
        self.assertTrue(field_stats(a,a)["exact"])

    def test_shape_and_nonfinite_fail(self):
        with self.assertRaisesRegex(ValueError,"shape"):
            field_stats(np.ones(2),np.ones(3))
        with self.assertRaisesRegex(ValueError,"nonfinite"):
            field_stats(np.ones(2),np.array([1.,np.nan]))


if __name__=="__main__":unittest.main()
