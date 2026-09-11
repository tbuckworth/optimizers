import unittest

from summarize_results import finite_mean, leakage_events, paired_contrasts


class SummaryTests(unittest.TestCase):
    def test_nulls_not_zero(self):
        self.assertEqual(finite_mean([1., None, 3.]), (2., {"finite": 2, "null": 1}))
        self.assertEqual(finite_mean([None]), (None, {"finite": 0, "null": 1}))

    def test_nonfinite_rejected(self):
        with self.assertRaises(ValueError):
            finite_mean([float("nan")])

    def test_closure_can_disqualify_nominal_reversal(self):
        row = {"raw_gradient_dot_update": {"value": .1, "sign": 1, "tolerance": .001},
               "in_subspace_contribution": {"value": -.2, "sign": -1, "tolerance": .001},
               "outside_contribution": {"value": .3, "sign": 1, "tolerance": .001},
               "identity_closure_residual": 0.}
        self.assertTrue(leakage_events(row)["closure_robust_leakage_reversal"])
        row["identity_closure_residual"] = .11
        self.assertTrue(leakage_events(row)["nominal_leakage_reversal"])
        self.assertFalse(leakage_events(row)["closure_robust_leakage_reversal"])

    def test_pairs_are_equal_weighted_seeds(self):
        runs = [{"seed": seed, "replacement_probability": noise, "arm": arm,
                 "metrics": {"value": seed + offset, "null": None}, "counts": {}}
                for seed in (0, 1, 2) for noise in (0., .9)
                for arm, offset in (("adamw", 0), ("estimate32_project32", 2), ("estimate128_project32", 3))]
        contrasts = paired_contrasts(runs)
        self.assertEqual(contrasts[0]["mean"], 1.)
        self.assertEqual(len(contrasts[0]["paired_differences"]), 3)
        self.assertIsNone(contrasts[1]["mean"])
        self.assertEqual(contrasts[1]["unavailable_seeds"], [0, 1, 2])

    def test_equal_null_counts_different_steps_are_not_comparable(self):
        runs = [{"seed": seed, "replacement_probability": noise, "arm": arm,
                 "metrics": {"value": 1.},
                 "counts": {"value": {"finite": 38, "null": 1, "null_steps": [null_step]}}}
                for seed in (0, 1, 2) for noise in (0., .9)
                for arm, null_step in (("adamw", 101), ("estimate32_project32", 101),
                                       ("estimate128_project32", 150))]
        contrast = paired_contrasts(runs)[0]
        self.assertFalse(contrast["all_three_pairs_available"])
        self.assertIsNone(contrast["mean"])


if __name__ == "__main__":
    unittest.main()
