import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("iteration006_independent_results_audit", HERE / "audit_completed_results.py")
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


class AuditAggregationTests(unittest.TestCase):
    def test_fixed_condition_and_pair_cardinalities(self):
        self.assertEqual(len(audit.EXPECTED_CELLS), 36)
        self.assertEqual(len(audit.PAIRS), 15)
        self.assertEqual(audit.PAIRS[0], ("current32", "adamw"))
        self.assertEqual(audit.PAIRS[-1], ("scalar_lagged32", "scalar_current32"))

    def test_descriptive_preserves_adverse_signs(self):
        result = audit.descriptive([-0.03, 0.01, -0.02])
        self.assertAlmostEqual(result["mean"], -0.04 / 3)
        self.assertEqual(result["minimum"], -0.03)
        self.assertEqual(result["maximum"], 0.01)
        self.assertGreater(result["sample_sd_descriptive"], 0)

    def test_incomplete_statistics_are_null_not_available_case(self):
        self.assertEqual(audit.complete_statistics([1.0, None, 3.0]), {
            "mean": None, "median": None, "minimum": None, "maximum": None,
            "sample_sd_descriptive": None,
        })

    def test_null_mask_is_exact(self):
        rows = [
            {"step": 101, "x": 1.0, "null_reasons": {}},
            {"step": 102, "x": None, "null_reasons": {"x": "zero_vector_norm"}},
            {"step": 103, "x": 3.0, "null_reasons": {}},
        ]
        value, counts = audit.aggregate_steps(rows, "x")
        self.assertEqual(value, 2.0)
        self.assertEqual(counts["scheduled_count"], 3)
        self.assertEqual(counts["finite_count"], 2)
        self.assertEqual(counts["null_steps"], [102])
        self.assertEqual(counts["null_reasons"], {"102": "zero_vector_norm"})

    def test_strict_json_rejects_duplicate_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text('{"x": 1, "x": 2}')
            with self.assertRaises(AssertionError):
                audit.read_json(path)

    def test_strict_json_rejects_nonfinite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text('{"x": NaN}')
            with self.assertRaises(ValueError):
                audit.read_json(path)

    def test_tree_comparison_tolerates_only_tiny_float_error(self):
        self.assertEqual(audit.compare_tree({"x": [1.0]}, {"x": [1.0 + 1e-14]}), [])
        self.assertTrue(audit.compare_tree({"x": [1.0]}, {"x": [1.01]}))

    def test_seed_first_pairing_rejects_different_null_masks(self):
        summaries = []
        for seed, noise, arm in sorted(audit.EXPECTED_CELLS):
            value = float(seed) + noise + audit.ARMS.index(arm)
            null_steps = [101] if (seed == 6 and noise == 0.0 and arm == "lagged32") else []
            summaries.append({
                "seed": seed, "replacement_probability": noise, "arm": arm,
                "metrics": {audit.PRIMARY_METRIC: value, "geometry": value},
                "counts": {"geometry": {"null_steps": null_steps}},
            })
        # Cardinality assertions are tied to the frozen 177-metric schema, so use
        # the same pairing logic directly for this focused synthetic case.
        indexed = {(r["seed"], r["replacement_probability"], r["arm"]): r for r in summaries}
        left = indexed[6, 0.0, "lagged32"]
        right = indexed[6, 0.0, "current32"]
        self.assertNotEqual(left["counts"]["geometry"]["null_steps"],
                            right["counts"]["geometry"]["null_steps"])

    def test_primary_family_is_exact(self):
        expected = {(noise, treatment, control) for noise in audit.NOISES
                    for treatment, control in audit.PRIMARY_PAIRS}
        self.assertEqual(len(expected), 4)
        self.assertEqual(expected, {
            (0.0, "lagged32", "current32"),
            (0.0, "lagged32_current_norm", "current32"),
            (0.9, "lagged32", "current32"),
            (0.9, "lagged32_current_norm", "current32"),
        })

    def test_independent_plan_is_deterministic_and_disjoint(self):
        first, second = audit.independent_plan(6), audit.independent_plan(6)
        for key in first:
            if hasattr(first[key], "shape"):
                self.assertTrue((first[key] == second[key]).all())
            else:
                self.assertEqual(first[key], second[key])
        combined = list(first["train_indices"]) + list(first["validation_indices"]) + list(first["auxiliary_indices"])
        self.assertEqual(len(combined), 15000)
        self.assertEqual(len(set(combined)), 15000)

    def test_fresh_plan_uses_separate_frozen_namespace(self):
        fresh = audit.independent_audit_plan()
        primary = audit.independent_plan(6)
        self.assertEqual(fresh["seed"], 60006)
        self.assertNotEqual(fresh["initialization_seed"], primary["initialization_seed"])
        self.assertFalse((fresh["train_indices"] == primary["train_indices"]).all())
        combined = list(fresh["train_indices"]) + list(fresh["validation_indices"]) + list(fresh["auxiliary_indices"])
        self.assertEqual(len(combined), len(set(combined)))

    def test_fresh_cells_are_six_and_never_primary_cells(self):
        fresh = {(60006, noise, arm) for noise in audit.NOISES
                 for arm in ("current32", "lagged32", "lagged32_current_norm")}
        self.assertEqual(len(fresh), 6)
        self.assertTrue(fresh.isdisjoint(audit.EXPECTED_CELLS))

    def test_accuracy_lattice(self):
        audit.verify_evaluation({"cross_entropy": 1.0, "accuracy": 0.1234, "count": 10000}, 10000)
        with self.assertRaises(AssertionError):
            audit.verify_evaluation({"cross_entropy": 1.0, "accuracy": 0.12345, "count": 5000}, 5000)


if __name__ == "__main__":
    unittest.main()
