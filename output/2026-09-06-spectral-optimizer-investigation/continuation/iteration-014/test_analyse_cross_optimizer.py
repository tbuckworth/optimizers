"""Pure scalar/unit tests for the independent I14 analyzer."""
from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyse_cross_optimizer as analysis


def point(horizon: int, validation_ce: float, validation_accuracy: float,
          auxiliary_ce: float = 0.5, auxiliary_accuracy: float = 0.8) -> dict:
    return {"horizon": horizon,
            "train": {"clean_ce": auxiliary_ce,
                      "clean_accuracy": auxiliary_accuracy,
                      "fixed_ce": auxiliary_ce, "soft_ce": auxiliary_ce,
                      "fixed_minus_soft_ce": 0.0, "fixed_accuracy": auxiliary_accuracy,
                      "mean_max_probability": 0.7},
            "validation": {"clean_ce": validation_ce,
                           "clean_accuracy": validation_accuracy},
            "auxiliary": {"clean_ce": auxiliary_ce,
                          "clean_accuracy": auxiliary_accuracy}}


def calibration_branch(seed: int, base: str, lr: float, ce: float,
                       accuracy: float = 0.9, complete: bool = True) -> dict:
    curve = [point(horizon, ce + (2000 - horizon) / 10000, accuracy)
             for horizon in analysis.HORIZONS]
    return {"seed": seed, "base": base, "lr": lr,
            "status": "complete" if complete else "numerical_failure",
            "curve_by_horizon": {row["horizon"]: row for row in curve}}


def confirmation_branches() -> dict:
    branches = {}
    for seed in analysis.CONFIRMATION_SEEDS:
        for base_index, base in enumerate(analysis.BASES):
            for target in analysis.TARGETS:
                for policy in analysis.POLICIES:
                    offset = 0.02 if policy == "current32" else 0.0
                    curve = [point(horizon, 0.8 - horizon / 10000,
                                   0.6 + horizon / 10000,
                                   auxiliary_ce=0.7 - offset,
                                   auxiliary_accuracy=0.75 + offset)
                             for horizon in analysis.HORIZONS]
                    branches[(seed, base, target, policy)] = {
                        "status": "complete", "curve_by_horizon": {
                            row["horizon"]: row for row in curve},
                        "independent_selection": analysis.selection(curve),
                        "diagnostic_summary": {
                            "energy_weighted_current_basis_leakage": {
                                "data": {"fraction": 0.1 * (base_index + 1)},
                                "total": {"fraction": 0.2 * (base_index + 1)},
                            },
                            "filtering_active_window": {
                                "energy_weighted_current_basis_leakage": {
                                    "data": {"fraction": 0.01 * (base_index + 1)},
                                    "total": {"fraction": 0.02 * (base_index + 1)},
                                },
                                "arithmetic_step_means": {
                                    "raw_gradient_norm": 1.0,
                                    "applied_gradient_norm": 0.8,
                                    "native_gradient_norm": 0.8,
                                    "total_displacement_norm": 0.1,
                                    "data_displacement_norm": 0.09,
                                    "nominal_decay_norm": 0.01,
                                    "raw_gradient_dot_data_displacement": -0.09,
                                },
                            },
                            "arithmetic_step_means": {
                                "raw_gradient_norm": 1.0,
                                "applied_gradient_norm": 0.9,
                                "native_gradient_norm": 0.9,
                                "total_displacement_norm": 0.1,
                                "data_displacement_norm": 0.09,
                                "nominal_decay_norm": 0.01,
                                "raw_gradient_dot_data_displacement": -0.09,
                            }},
                    }
    return branches


class AnalyzerTests(unittest.TestCase):
    def test_selection_uses_nonzero_validation_only_and_earliest_ties(self):
        curve = [point(0, 0.01, 0.99), point(100, 0.5, 0.7),
                 point(250, 0.4, 0.8), point(500, 0.4, 0.8)]
        self.assertEqual(analysis.selection(curve), {
            "minimum_validation_ce": 250,
            "maximum_validation_accuracy": 250,
        })

    def test_rate_selection_requires_both_seeds_and_breaks_exact_tie_by_small_rate(self):
        rows = {}
        for base in analysis.BASES:
            for seed in analysis.CALIBRATION_SEEDS:
                for index, lr in enumerate(analysis.RATES[base]):
                    # First two rates tie and beat the third.
                    rows[(seed, base, lr)] = calibration_branch(
                        seed, base, lr, 0.2 if index < 2 else 0.3)
        selected = analysis.choose_rates(rows)
        self.assertTrue(selected["ready_for_confirmation"])
        self.assertEqual(selected["selected_rates"], {
            base: analysis.RATES[base][0] for base in analysis.BASES})

        # One weak seed invalidates that rate; the other seed cannot survive-average it.
        first = analysis.RATES["sgd"][0]
        rows[(190, "sgd", first)] = calibration_branch(190, "sgd", first, 0.1, 0.849)
        selected = analysis.choose_rates(rows)
        self.assertEqual(selected["selected_rates"]["sgd"], analysis.RATES["sgd"][1])

        rows[(190, "sgd", first)]["status"] = "numerical_failure"
        selected = analysis.choose_rates(rows)
        candidate = next(row for row in selected["candidates"]
                         if row["base"] == "sgd" and row["lr"] == first)
        self.assertFalse(candidate["finite_complete"])
        self.assertEqual(candidate["validation_by_seed"][0],
                         rows[(190, "sgd", first)]["curve_by_horizon"][2000]["validation"])

    def test_endpoint_effect_signs_and_missing_pair_disable_only_bound_estimand(self):
        branches = confirmation_branches()
        effects = analysis.endpoint_effects(branches, 2000)
        for row in effects:
            self.assertTrue(row["effect"]["available"])
            self.assertAlmostEqual(row["effect"]["mean"], 0.02)
        branches[(201, "sgd", "clean", "current32")]["status"] = "numerical_failure"
        del branches[(201, "sgd", "clean", "current32")]["curve_by_horizon"][2000]
        effects = analysis.endpoint_effects(branches, 2000)
        missing = next(row for row in effects if row["base"] == "sgd"
                       and row["target"] == "clean" and row["metric"] == "ce")
        retained = next(row for row in effects if row["base"] == "sgdm"
                        and row["target"] == "clean" and row["metric"] == "ce")
        self.assertFalse(missing["effect"]["available"])
        self.assertIsNone(missing["effect"]["mean"])
        self.assertTrue(retained["effect"]["available"])

    def test_selector_effects_report_both_metrics_and_arm_specific_horizons(self):
        rows = analysis.selector_effects(confirmation_branches())
        self.assertEqual(len(rows), 3 * 2 * 2 * 2)
        self.assertEqual({row["auxiliary_metric"] for row in rows}, {"ce", "accuracy"})
        self.assertTrue(all(row["effect"]["available"] for row in rows))

    def test_leakage_is_branch_ratio_then_equal_seed_mean(self):
        branches = confirmation_branches()
        rows = analysis.trajectory_leakage_seed_means(branches)
        row = next(item for item in rows if item["base"] == "sgdm"
                   and item["target"] == "fixed" and item["policy"] == "raw"
                   and item["window"] == "full_trajectory"
                   and item["displacement"] == "data")
        self.assertEqual(row["aggregation"],
                         "energy ratio within trajectory, then equal seeds")
        self.assertAlmostEqual(row["equal_seed_mean"], 0.2)

        branches[(201, "sgdm", "fixed", "raw")]["status"] = "numerical_failure"
        rows = analysis.trajectory_leakage_seed_means(branches)
        row = next(item for item in rows if item["base"] == "sgdm"
                   and item["target"] == "fixed" and item["policy"] == "raw"
                   and item["window"] == "full_trajectory"
                   and item["displacement"] == "data")
        self.assertFalse(row["available"])
        self.assertIsNone(row["equal_seed_mean"])

    def test_progress_keeps_raw_ce_and_names_positive_good_reduction(self):
        rows = analysis.progress_rows(confirmation_branches())
        row = next(item for item in rows if item["seed"] == 200
                   and item["base"] == "sgd" and item["policy"] == "raw"
                   and item["horizon"] == 2000)
        self.assertAlmostEqual(row["values"]["auxiliary_clean_ce"], 0.7)
        self.assertAlmostEqual(row["values"]["auxiliary_clean_ce_reduction_from_h0"], 0.0)
        self.assertNotIn("auxiliary_ce", row["values"])

    def test_failure_fingerprint_is_tensor_bit_sensitive_and_nonfinite_safe(self):
        value = {"state": torch.tensor([1.0, float("nan")]),
                 "failure": {"value": float("inf")}, "curve": [], "steps": []}
        first = analysis.failure_fingerprint(value)
        second = analysis.failure_fingerprint(value)
        self.assertEqual(first, second)
        changed = {**value, "state": torch.tensor([2.0, float("nan")])}
        self.assertNotEqual(first, analysis.failure_fingerprint(changed))

    def test_consumed_membership_and_corruption_counts(self):
        audit = analysis.Audit()
        analysis.require_member({"bound.json": {}}, "bound.json", audit, "test")
        analysis.require_member({"bound.json": {}}, "unlisted.json", audit, "test")
        self.assertEqual(len(audit.errors), 1)

        labels = np.arange(60000, dtype=np.int64) % 10
        indices = list(range(5000))
        mask = [False] * 5000
        digits = [0] * 5000
        mask[0] = mask[1] = True
        digits[0] = int(labels[0])       # replacement, but not incorrect
        digits[1] = int((labels[1] + 1) % 10)
        counts = analysis.corruption_counts({"train_indices": indices,
            "replacement_mask": mask, "replacement_digits": digits}, labels)
        self.assertEqual(counts, {"replaced_count": 2, "incorrect_count": 1,
                                  "train_count": 5000})


if __name__ == "__main__":
    unittest.main()
