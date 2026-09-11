"""Small synthetic checks for I18 scalar report corroboration helpers."""
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

import corroborate_tracking_report as check


class CorroborationTests(unittest.TestCase):
    def test_seed_statistics_and_signs(self):
        result = check.stats({1: -1.0, 2: 0.0, 3: 4.0}, seeds=(1, 2, 3))
        self.assertEqual(result["mean"], 1.0)
        self.assertEqual(result["positive_seeds"], 1)
        self.assertEqual(result["negative_seeds"], 1)
        self.assertEqual(result["zero_seeds"], 1)
        self.assertAlmostEqual(result["standard_error"], math.sqrt(7 / 3))

    def test_compare_seed_summary_rejects_mutation_and_missing_seed(self):
        values = {1: 1.0, 2: 3.0}
        expected = {"per_seed": {"1": 1.0, "2": 3.0}, "available": True,
                    "n_independent_seeds": 2, "mean": 2.0, "standard_error": 1.0,
                    "positive_seeds": 2, "negative_seeds": 0, "zero_seeds": 0,
                    "no_survivor_averaging": True}
        compare = check.Compare()
        check.compare_seed_summary(expected, values, compare, "fixture", seeds=(1, 2))
        expected["mean"] = 2.1
        with self.assertRaisesRegex(ValueError, "differs"):
            check.compare_seed_summary(expected, values, check.Compare(), "fixture", seeds=(1, 2))
        with self.assertRaisesRegex(ValueError, "incomplete"):
            check.stats({1: 1.0}, seeds=(1, 2))

    def test_npz_scalar_mse_is_independently_determined(self):
        signal = np.array([[0., 0.], [1., -1.], [2., -2.]])
        output = np.stack((signal, signal + np.array([1., 2.])), axis=1)
        arrays = {key: np.zeros(1, dtype=np.float64) for key in check.ARRAY_KEYS}
        arrays.update({"s": signal, "output": output})
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "fixture.npz"
            np.savez(path, **arrays)
            windows, whole = check.load_scalar_metrics(
                path, ("exact", "offset"), 3, (("first_two", 0, 2),))
        self.assertEqual(windows["first_two"], {"exact": 0.0, "offset": 5.0})
        self.assertEqual(whole, {"exact": 0.0, "offset": 5.0})

    def test_verified_file_rejects_hash_size_and_symlink(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / "value.bin"
            path.write_bytes(b"fixed")
            record = {"path": "value.bin", "size": 5, "sha256": check.sha(path)}
            self.assertEqual(check.verify_file(root, record, "value.bin"), path)
            wrong = dict(record, sha256="0" * 64)
            with self.assertRaisesRegex(ValueError, "differs"):
                check.verify_file(root, wrong, "value.bin")
            link = root / "link.bin"
            link.symlink_to(path)
            linked = {"path": "link.bin", "size": 5, "sha256": check.sha(path)}
            with self.assertRaisesRegex(ValueError, "differs"):
                check.verify_file(root, linked, "link.bin")


if __name__ == "__main__":
    unittest.main()
