"""Fabricated-array tests only; never opens model or historical scientific data."""
import unittest
import numpy as np
from patterns import fit_pair_patterns


class PatternTests(unittest.TestCase):
    def setUp(self):
        self.h = np.array([[4, 2, 1], [0, 0, 0], [1, 4, 2], [0, 0, 0],
                           [-1, 2, 3], [2, 1, 1]], dtype=np.float32)
        self.u = np.eye(3, dtype=np.float32)[:, :2]
        self.pairs = [[0, 1], [2, 3], [4, 5]]

    def fit(self, h=None, u=None, pairs=None):
        return fit_pair_patterns(self.h if h is None else h,
                                 self.u if u is None else u,
                                 self.pairs if pairs is None else pairs)

    def test_independent_dense_identity(self):
        d = self.h[::2].astype(float)-self.h[1::2].astype(float)
        covariance = d.T @ d / len(d)
        u = self.u.astype(float)
        expected = (covariance @ u)/np.diag(u.T @ covariance @ u)[None, :]
        result = self.fit()
        np.testing.assert_allclose(result['patterns64'], expected, rtol=0, atol=1e-14)
        np.testing.assert_allclose(result['unit_score_identity64'], [1, 1], rtol=0, atol=1e-14)
        np.testing.assert_allclose(result['calibration_gaps64'], d @ u, rtol=0, atol=1e-14)

    def test_symmetrized_population(self):
        d = self.h[::2].astype(float)-self.h[1::2].astype(float)
        both = np.vstack([d, -d])
        covariance = np.cov(both, rowvar=False, bias=True)
        expected = covariance @ self.u/np.diag(self.u.T @ covariance @ self.u)[None, :]
        np.testing.assert_allclose(self.fit()['patterns64'], expected, rtol=0, atol=1e-14)

    def test_pair_sign_reversal(self):
        result = self.fit(pairs=[[1, 0], [2, 3], [5, 4]])
        original = self.fit()
        for key in ['patterns64', 'cross64', 'score_energy64', 'signed_inputs32']:
            np.testing.assert_array_equal(result[key], original[key])

    def test_pair_permutation(self):
        result = self.fit(pairs=self.pairs[::-1])
        np.testing.assert_allclose(result['patterns64'], self.fit()['patterns64'], rtol=0, atol=1e-14)

    def test_translation_and_common_rescaling(self):
        expected = self.fit()['patterns64']
        np.testing.assert_allclose(self.fit(h=self.h+np.float32(100))['patterns64'], expected)
        np.testing.assert_allclose(self.fit(h=self.h*np.float32(2))['patterns64'], expected)

    def test_original_score_orientation_not_new_pca(self):
        result = self.fit()
        signed = result['signed_inputs32']
        np.testing.assert_array_equal(signed[1::2], -signed[::2])
        self.assertEqual(signed.dtype, np.float32)
        np.testing.assert_allclose(np.linalg.norm(signed.astype(float), axis=1), 1, atol=1e-6)
        self.assertTrue(np.all(np.diag(self.u.T @ signed[::2].T) > 0))
        self.assertGreater(abs(float(result['patterns64'][:, 0] @ result['patterns64'][:, 1])), 0)

    def test_large_gap_dominance_is_reported(self):
        h = np.array([[10, 1], [0, 0], [1, 10], [0, 0]], dtype=np.float32)
        u = np.array([[1], [0]], dtype=np.float32)
        result = self.fit(h=h, u=u, pairs=[[0, 1], [2, 3]])
        self.assertAlmostEqual(result['max_single_pair_energy_share64'][0], 100/101)
        np.testing.assert_allclose(result['patterns64'][:, 0], [1, 20/101])

    def test_low_positive_energy_is_retained(self):
        h = np.array([[1e-20, 1], [0, 0]], dtype=np.float32)
        u = np.array([[1], [0]], dtype=np.float32)
        result = self.fit(h=h, u=u, pairs=[[0, 1]])
        self.assertGreater(result['score_energy64'][0], 0)
        self.assertLess(result['score_energy64'][0], 1e-30)
        self.assertAlmostEqual(result['unit_score_identity64'][0], 1)

    def test_any_zero_energy_axis_stops_entire_fit(self):
        h = self.h.copy(); h[:, 1] = 0
        with self.assertRaisesRegex(ValueError, 'zero score energy'):
            self.fit(h=h)

    def test_nonfinite_or_wrong_dtype(self):
        for invalid in [np.nan, np.inf]:
            h = self.h.copy(); h[0, 0] = invalid
            with self.assertRaises(ValueError): self.fit(h=h)
        with self.assertRaises(ValueError): self.fit(h=self.h.astype(np.float64))
        with self.assertRaises(ValueError): self.fit(u=self.u.astype(np.float64))

    def test_empty_shape_or_nonunit_basis(self):
        for h, u in [(self.h[:, :2], self.u), (self.h[:0], self.u), (self.h, self.u*2)]:
            with self.assertRaises(ValueError): self.fit(h=h, u=u)

    def test_pair_indices_must_cover_exactly_once(self):
        for pairs in [[], [[0, 1]], [[0, 1], [0, 2], [4, 5]], [[0, 0], [2, 3], [4, 5]],
                      [[True, 1], [2, 3], [4, 5]], [[0, 1], [2, 3], [4, 6]]]:
            with self.assertRaises(ValueError): self.fit(pairs=pairs)

    def test_no_input_mutation(self):
        old_h, old_u = self.h.copy(), self.u.copy()
        self.h.flags.writeable = False; self.u.flags.writeable = False
        self.fit()
        np.testing.assert_array_equal(self.h, old_h)
        np.testing.assert_array_equal(self.u, old_u)


if __name__ == '__main__':
    unittest.main()
