"""Synthetic saved-array fixtures; never load real measurement files."""
import unittest
import numpy as np
from audit_evaluation import check_arrays


class Tests(unittest.TestCase):
    def fixture(self):
        rng = np.random.default_rng(71)
        h = rng.normal(size=(8, 1, 7)).astype(np.float32)
        u = rng.normal(size=(7, 3)).astype(np.float32)
        mean = rng.normal(size=7).astype(np.float64)
        scores = (h.astype(np.float64)-mean) @ u.astype(np.float64)
        return h, u, mean, scores, scores[::2]-scores[1::2]

    def test_valid_and_all_zero_ties(self):
        arrays = self.fixture(); self.assertEqual(check_arrays(*arrays)['pairs'], 4)
        h, u, mean, *_ = arrays
        h[:] = 0; mean[:] = 0
        self.assertEqual(check_arrays(h, u, mean, np.zeros((8, 1, 3)), np.zeros((4, 1, 3)))['axes'], 3)

    def test_score_and_gap_mutation_rejected(self):
        for index in (3, 4):
            values = list(self.fixture()); values[index][0, 0, 0] += .01
            with self.assertRaises(ValueError): check_arrays(*values)

    def test_wrong_dtype_shape_nonfinite_rejected(self):
        for kind in ('dtype', 'shape', 'nan', 'mean'):
            values = list(self.fixture())
            if kind == 'dtype': values[0] = values[0].astype(np.float64)
            elif kind == 'shape': values[4] = values[4][:-1]
            elif kind == 'nan': values[1][0, 0] = np.nan
            else: values[2] = values[2]+1
            with self.assertRaises(ValueError): check_arrays(*values)

    def test_gap_is_subtraction_of_rounded_scores(self):
        h = np.array([[[1.]], [[0.]]], dtype=np.float32)
        u = np.ones((1, 1), dtype=np.float32); mean = np.array([2.**55])
        scores = (h.astype(np.float64)-mean) @ u.astype(np.float64)
        gaps = scores[::2]-scores[1::2]
        self.assertEqual(gaps.item(), 0.)
        check_arrays(h, u, mean, scores, gaps)
        with self.assertRaises(ValueError): check_arrays(h, u, mean, scores, np.ones_like(gaps))


if __name__ == '__main__': unittest.main()
