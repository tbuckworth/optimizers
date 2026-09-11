"""Pure scalar weighting and missingness tests for I13 report assets."""
from __future__ import annotations

from pathlib import Path
import sys
import unittest


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import report_assets as report


class ReportAssetTests(unittest.TestCase):
    def test_seed_first_weights_parents_then_seeds(self):
        values = {(100, 1500): 0.0, (100, 2000): 2.0,
                  (101, 1500): 10.0, (101, 2000): 14.0,
                  (102, 1500): -4.0, (102, 2000): 4.0}
        result = report.seed_first(values)
        self.assertEqual(result["all_three_seed_values"], [1.0, 12.0, 0.0])
        self.assertEqual(result["mean"], 13.0 / 3.0)
        self.assertTrue(result["available"] and result["no_survivor_averaging"])

    def test_missing_parent_disables_whole_aggregate(self):
        values = {(seed, parent): 1.0 for seed in report.SEEDS for parent in report.PARENTS}
        values[(102, 2000)] = None
        result = report.seed_first(values)
        self.assertFalse(result["available"])
        self.assertIsNone(result["mean"])
        self.assertIsNone(result["all_three_seed_values"][-1])
        self.assertFalse(result["per_seed"][-1]["available"])

    def test_leakage_is_ratio_within_branch_not_mean_step_ratio(self):
        def row(norm, fraction):
            squared = norm * norm
            leakage = {"reason": None, "fraction": fraction, "squared_norm": squared,
                       "outside_squared_norm": squared * fraction}
            return {"displacement": {"data_norm": norm, "data_leakage": {
                "current_basis": leakage}}}
        steps = [row(1.0, 1.0), row(3.0, 0.0)]
        self.assertAlmostEqual(report.branch_fraction(steps, "data", "current_basis"), .1)
        self.assertNotAlmostEqual(report.branch_fraction(steps, "data", "current_basis"), .5)


if __name__ == "__main__":
    unittest.main()
