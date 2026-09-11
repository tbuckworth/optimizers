import unittest
import numpy as np
from analyze import estimate, top_basis, cosine


class AggregateTests(unittest.TestCase):
    def test_mean_can_be_lost_by_centering(self):
        fit = np.array([[-2., 3.], [2., 3.]])
        query = np.array([[[0., 3.], [0., 3.]]])
        ref = np.array([[[0., 3.]], [[0., 3.]]])
        result, values = estimate(fit, query, ref, 1)
        np.testing.assert_allclose(values['centered'], [[0., 0.]])
        np.testing.assert_allclose(values['second_moment'], [[0., 3.]])
        self.assertEqual(result['arms']['centered']['mean_raw_minus_arm_error'], [-9., -9., -9.])
        self.assertFalse(result['arms']['centered']['positive_both_reference_groups'])

    def test_full_rank_identity_and_coefficients(self):
        fit = np.array([[-2., 3.], [2., 3.], [1., 2.]])
        query = np.array([[[.4, 1.], [.8, 3.]], [[0., 3.], [4., 3.]]])
        ref = np.array([[[0., 3.]], [[1., 3.]]])
        _, values = estimate(fit, query, ref, 2)
        np.testing.assert_allclose(values['centered'], query.mean(axis=1), atol=1e-14)
        np.testing.assert_allclose(values['centered_coefficients64']@values['centered_basis64'].T, values['raw'], atol=1e-14)

    def test_shared_mean_noise_removal(self):
        fit = np.array([[-2., 0.], [4., 0.]])
        query = np.array([[[1., 1.]], [[1., -1.]]])
        ref = np.array([[[1., 0.]], [[1., 0.]]])
        result, _ = estimate(fit, query, ref, 1)
        self.assertEqual(result['arms']['centered']['mean_raw_minus_arm_error'], [1., 1., 1.])
        self.assertTrue(result['arms']['centered']['positive_both_reference_groups'])
        self.assertIsNone(cosine(np.zeros(2), np.ones(2)))


if __name__ == '__main__':
    unittest.main()
