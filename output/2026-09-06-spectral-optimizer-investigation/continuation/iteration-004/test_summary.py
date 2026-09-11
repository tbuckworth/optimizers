#!/usr/bin/env python3
"""Synthetic-only checks of selectors, aggregation, nulls and completion gates."""
import copy
import unittest
import summarize_results as s


def evaluation(accuracy=.4, ce=2., count=5000):
    return {"accuracy": accuracy, "cross_entropy": ce, "count": count}


def fixture(seed=3, arm="scalar32_norm"):
    trajectory = [{"step": step, **evaluation()} for step in range(0, 2001, 100)]
    trajectory[1] = {"step": 100, **evaluation(.5, 1.9)}
    trajectory[2] = {"step": 200, **evaluation(.6, 1.8)}
    trajectory[3] = {"step": 300, **evaluation(.6, 1.7)}
    trajectory[4] = {"step": 400, **evaluation(.55, 1.7)}
    update = {"squared_norm": .01,
              "raw_gradient_dot_update": {"value": -.1, "tolerance": .00001, "sign": -1},
              "applied_gradient_dot_update": {"value": -.05, "tolerance": .00001, "sign": -1},
              "raw_gradient_update_cosine": -.5, "applied_gradient_update_cosine": -.5}
    rows = []
    for step in range(1, 2001):
        rows.append({"step": step, "raw_squared_norm": 4., "candidate_squared_norm": 1.,
                     "applied_squared_norm": 1., "alpha": .5, "raw_applied_cosine": 1.,
                     "scalar_collinearity_relative_error": 0., "norm_matching_absolute_error": 0.,
                     "total": copy.deepcopy(update), "decay_subtracted": copy.deepcopy(update)})
    return {"seed": seed, "arm": arm, "replacement_probability": .9, "steps": 2000,
            "steps_raw": rows, "all_invariant_gates_passed": True, "warmup_checks_passed": True,
            "validation_trajectory": trajectory,
            "checkpoint_steps": {"final": 2000, "min_val_ce": 300, "max_val_accuracy": 200, "warmup100": 100},
            "test": {name: evaluation(.5 if name == "warmup100" else .6, 1.8, 10000) for name in s.CHECKPOINTS},
            "final_training_clean": evaluation(), "final_training_noisy": evaluation(),
            "final_validation": trajectory[-1], "realized_incorrect_fraction": .81,
            "realized_replacement_fraction": .9}


class SummaryTests(unittest.TestCase):
    def test_independent_earliest_selectors_and_step_zero(self):
        result = fixture()
        s.validate_selectors(result)
        result["checkpoint_steps"]["max_val_accuracy"] = 300
        with self.assertRaises(ValueError):
            s.validate_selectors(result)
        result["checkpoint_steps"]["max_val_accuracy"] = 200
        result["validation_trajectory"][0].update(accuracy=.9, cross_entropy=.1)
        result["checkpoint_steps"].update(min_val_ce=0, max_val_accuracy=0)
        s.validate_selectors(result)

    def test_norms_windows_and_warmup_difference(self):
        result = s.summarize_run(fixture())
        self.assertEqual(result["metrics"]["gradient.all.raw_norm"], 2.)
        self.assertEqual(result["metrics"]["gradient.all.applied_raw_norm_ratio"], .5)
        self.assertAlmostEqual(result["metrics"]["test.max_val_accuracy.minus_warmup100.accuracy"], .1)
        self.assertEqual(result["counts"]["gradient.all.alpha"]["finite"], 1900)
        self.assertEqual(result["counts"]["gradient.early.alpha"]["finite"], 400)
        self.assertEqual(result["counts"]["gradient.late.alpha"]["finite"], 500)

    def test_zero_cosine_null_and_all_step_frequency_denominator(self):
        fixture_result = fixture()
        row = fixture_result["steps_raw"][100]
        row["raw_squared_norm"] = 0.
        row["raw_applied_cosine"] = None
        row["total"]["raw_gradient_dot_update"] = {"value": .1, "tolerance": .01, "sign": 1}
        result = s.summarize_run(fixture_result)
        self.assertEqual(result["counts"]["gradient.all.candidate_raw_norm_ratio"]["null_steps"], [101])
        self.assertEqual(result["metrics"]["update.all.total.raw_gradient_ascent_frequency"], 1 / 1900)

    def test_validation_training_and_sign_corruption_rejected(self):
        for mutation in (lambda r: r["steps_raw"].pop(),
                         lambda r: r["validation_trajectory"].pop(),
                         lambda r: r["test"]["final"].update(count=9999),
                         lambda r: r["steps_raw"][100]["total"]["raw_gradient_dot_update"].update(sign=1)):
            result = fixture()
            mutation(result)
            with self.assertRaises(ValueError):
                s.summarize_run(result)

    def test_finite_values_required(self):
        with self.assertRaises(ValueError):
            s.finite_mean([1., float("nan")], [1, 2])
        result = fixture()
        result["validation_trajectory"][1]["cross_entropy"] = float("inf")
        with self.assertRaises(ValueError):
            s.summarize_run(result)

    def test_three_pairs_and_exact_null_masks(self):
        runs = []
        for seed in s.SEEDS:
            for arm in s.ARMS:
                runs.append({"seed": seed, "arm": arm, "metrics": {s.PRIMARY: seed + s.ARMS.index(arm), "nullable": 1.},
                             "counts": {"nullable": {"finite": 1, "null": 1, "null_steps": [101]}}})
        contrasts = s.paired_contrasts(runs)
        primary = [r for r in contrasts if r["co_primary"]]
        self.assertEqual(len(primary), 2)
        self.assertEqual(primary[0]["mean"], 1.)
        self.assertEqual(primary[1]["mean"], -2.)
        self.assertEqual([r["seed"] for r in primary[0]["paired_differences"]], [3, 4, 5])
        runs[1]["counts"]["nullable"]["null_steps"] = [102]
        null_contrast = next(r for r in s.paired_contrasts(runs) if r["treatment"] == s.ARMS[1] and r["control"] == s.ARMS[0] and r["metric"] == "nullable")
        self.assertIsNone(null_contrast["mean"])
        self.assertEqual(null_contrast["unavailable_seeds"], [3])
        with self.assertRaises(ValueError):
            s.paired_contrasts(runs[:-1])

    def test_complete_execution_and_test_order_gate(self):
        record = {"mode": "confirmatory", "status": "complete", "completed_runs": 12,
                  "test_evaluations": 48, "warmup_checks_passed": True,
                  "all_training_completed_utc": "2026-09-06T12:00:00Z",
                  "test_first_loaded_utc": "2026-09-06T12:00:01Z"}
        s.validate_execution(record)
        record["test_first_loaded_utc"] = "2026-09-06T11:59:59Z"
        with self.assertRaises(ValueError):
            s.validate_execution(record)
        record["test_first_loaded_utc"] = "2026-09-06T12:00:01Z"
        record["completed_runs"] = 11
        with self.assertRaises(ValueError):
            s.validate_execution(record)


if __name__ == "__main__":
    unittest.main(verbosity=2)
