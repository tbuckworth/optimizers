"""Fabricated-only arithmetic tests; no source text/model/measurement file access."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np

spec = importlib.util.spec_from_file_location('audit_fixture', Path(__file__).with_name('audit_measurement.py'))
a = importlib.util.module_from_spec(spec); spec.loader.exec_module(a)


class ArithmeticTests(unittest.TestCase):
    def test_fsum_exact_exports_and_corruption(self):
        rng = np.random.default_rng(831)
        h = rng.normal(size=(24, 1024)).astype(np.float32)
        u = rng.normal(size=(1024, 4)).astype(np.float32)
        mean = rng.normal(size=1024)
        scores = (h.astype(np.float64)-mean) @ u.astype(np.float64)
        gaps = scores[::2]-scores[1::2]
        result = a.arithmetic(h, u, mean, scores, gaps)
        self.assertEqual(96, result['scalar_fsum_checks'])
        self.assertEqual(48, result['ordering_sign_checks'])
        bad = scores.copy(); bad[0, 0] += 1
        with self.assertRaises(ValueError): a.arithmetic(h, u, mean, bad, gaps)
        bad_gap = gaps.copy(); bad_gap[1, 1] += 1e-8
        with self.assertRaises(ValueError): a.arithmetic(h, u, mean, scores, bad_gap)
        with self.assertRaises(ValueError): a.arithmetic(h.astype(np.float64), u, mean, scores, gaps)

    def test_exact_ties_retained(self):
        result = a.arithmetic(np.zeros((24, 1024), np.float32), np.zeros((1024, 4), np.float32),
                              np.zeros(1024), np.zeros((24, 4)), np.zeros((12, 4)))
        self.assertEqual(0, result['maximum_absolute_difference'])
        self.assertEqual(48, result['ordering_sign_checks'])


if __name__ == '__main__':
    unittest.main()
