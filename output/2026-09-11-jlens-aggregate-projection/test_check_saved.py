import unittest
import numpy as np
from analyze import estimate
from check_saved import check_geometry, close


class SavedChecks(unittest.TestCase):
    def test_fabricated_geometry_and_mutation(self):
        rng = np.random.default_rng(61)
        fit = rng.normal(size=(24, 6))+np.arange(6)[None, :]/7
        query = rng.normal(size=(4, 3, 6))
        refs = rng.normal(size=(2, 5, 6))
        _, values = estimate(fit, query, refs, 3)
        arrays = {'gradient_'+k: v for k, v in values.items()}
        error, projected, references = check_geometry(fit, query, refs, arrays, 'gradient')
        self.assertLess(error, 1e-10)
        np.testing.assert_allclose(projected['raw'], query.mean(axis=1))
        self.assertEqual(references.shape, (3, 6))
        arrays['gradient_centered_basis64'] = arrays['gradient_centered_basis64'].copy()
        arrays['gradient_centered_basis64'][0, 0] += .01
        with self.assertRaises(ValueError):
            check_geometry(fit, query, refs, arrays, 'gradient')

    def test_zero_and_nonzero_checks(self):
        self.assertEqual(close(np.zeros(3), np.zeros(3)), 0)
        with self.assertRaises(ValueError):
            close(np.ones(3), np.zeros(3))
        with self.assertRaises(ValueError):
            close(np.ones(2), np.ones(1))
        with self.assertRaises(ValueError):
            close(np.array([np.nan]), np.ones(1))
        with self.assertRaises(ValueError):
            close(np.ones(1), np.array([np.inf]))


if __name__ == '__main__': unittest.main()
