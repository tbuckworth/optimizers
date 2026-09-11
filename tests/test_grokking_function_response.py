import unittest
import numpy as np

from experiments.grokking_function_response import (
    analyze_seed, centered, paired_summary, response, sum_projection,
)


class FunctionResponseTests(unittest.TestCase):
    def setUp(self):
        self.p = 3
        self.sums = np.array([(a + b) % 3 for a in range(3) for b in range(3)])
        self.rng = np.random.default_rng(8021)
        self.base = self.rng.normal(size=(9, 3))
        self.logits = {name: self.base + self.rng.normal(size=(9, 3)) * .02
                       for name in ("before", "raw", "trunc", "projected", "zero")}
        self.train, self.test = np.arange(4), np.arange(4, 9)

    def test_orthogonal_projector(self):
        x = centered(self.base)
        rx = sum_projection(x, self.sums)
        np.testing.assert_allclose(sum_projection(rx, self.sums), rx, atol=1e-15)
        self.assertAlmostEqual(float(np.sum(rx * (x-rx))), 0)
        np.testing.assert_allclose(sum_projection(x-rx, self.sums), 0, atol=1e-15)
        self.assertAlmostEqual(float(np.sum(x*x)), float(np.sum(rx*rx) + np.sum((x-rx)**2)))

    def test_class_shift_invariance(self):
        a = analyze_seed(self.logits, self.sums, self.train, self.test)
        shifted = {k: v + self.rng.normal(size=(9, 1)) for k, v in self.logits.items()}
        b = analyze_seed(shifted, self.sums, self.train, self.test)
        self.assertAlmostEqual(a["contrasts"]["raw_to_trunc"]["energy"]["total"],
                               b["contrasts"]["raw_to_trunc"]["energy"]["total"])

    def test_finite_utilities_and_telescoping(self):
        result = analyze_seed(self.logits, self.sums, self.train, self.test)
        for item in result["responses_from_before"].values():
            for split in ("train", "test"):
                self.assertGreaterEqual(item["ce_convex_remainder"][split], -1e-14)
                self.assertAlmostEqual(item["utility_additivity_residual"][split], 0)
        for values in result["finite_scalar_telescoping_residual"].values():
            for residual in values.values():
                self.assertAlmostEqual(residual, 0)
        for values in result["linear_utility_telescoping_residual"].values():
            for residual in values.values():
                self.assertAlmostEqual(residual, 0)
        for left, right in (("raw", "trunc"), ("trunc", "projected"), ("raw", "projected")):
            pair = result["contrasts"][left + "_to_" + right]
            for split in ("train", "test"):
                for component in ("total", "sum_consistent", "within_sum"):
                    self.assertAlmostEqual(pair["linear_ce_utility"][split][component],
                        result["responses_from_before"][right]["linear_ce_utility"][split][component] -
                        result["responses_from_before"][left]["linear_ce_utility"][split][component])

    def test_utility_matches_small_logit_derivative(self):
        delta = self.rng.normal(size=(9, 3))
        logs = {k: self.base.copy() for k in self.logits}
        logs["raw"] += 1e-5 * delta
        result = analyze_seed(logs, self.sums, self.train, self.test)
        value = result["responses_from_before"]["raw"]
        self.assertLess(abs(value["finite_improvement"]["test"]["ce"] -
                            value["linear_ce_utility"]["test"]["total"]), 1e-9)

    def test_sum_consistency_not_correctness(self):
        base = np.zeros((9, 3))
        wrong = np.eye(3)[(self.sums+1) % 3]
        logs = {k: base.copy() for k in self.logits}
        logs["raw"] = wrong
        result = analyze_seed(logs, self.sums, self.train, self.test)
        effect = result["responses_from_before"]["raw"]
        self.assertAlmostEqual(effect["energy"]["within_sum"], 0)
        self.assertLess(effect["linear_ce_utility"]["test"]["sum_consistent"], 0)
        self.assertLess(effect["finite_improvement"]["test"]["ce"], 0)

    def test_invalid_grids_splits_and_nonfinite(self):
        with self.assertRaises(ValueError):
            sum_projection(self.base[:8], self.sums[:8])
        with self.assertRaises(ValueError):
            analyze_seed(self.logits, self.sums, self.train, self.train)
        with self.assertRaises(ValueError):
            sum_projection(self.base, self.sums.astype(float))
        with self.assertRaises(ValueError):
            centered(np.full((9, 3), np.nan))

    def test_sample_se(self):
        item = paired_summary([{"x": i} for i in range(5)])["x"]
        self.assertEqual(item["mean"], 2)
        self.assertAlmostEqual(item["sample_se"], np.sqrt(.5))
        self.assertEqual(item["positive_count"], 4)


if __name__ == "__main__":
    unittest.main()
