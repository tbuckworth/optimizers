"""Synthetic prospective tests; no experiment or dataset access."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import summarize_results as s


def fixture():
    execution = {"schema_version": 1, "mode": "full", "status": "complete", "all_gates_passed": True,
                 "official_test_opened": False, "new_accuracy_or_checkpoint_selection_computed": False,
                 "completed_replay_seeds": list(s.SEEDS),
                 "completed_snapshots": [{"seed": seed, "step": step} for seed in s.SEEDS for step in s.STEPS]}
    replays = []
    for seed in s.SEEDS:
        replays.append({"schema_version": 1, "seed": seed, "steps": 2000, "instrumented": True,
                        "all_historical_gates_passed": True, "all_state_gates_passed": True,
                        "historical_scalar_comparison_steps": 2000, "snapshot_count": 4, "probe_state_check_count": 4,
                        "observer_means_bitwise_equal_steps": 2000, "delivered_raw_equal_steps": 2000,
                        "historical_warmup_hash_steps": 100, "no_observer_resets": True,
                        "trajectory_parameter_sha256": ["a" * 64] * 2000,
                        "raw_gradient_sha256": ["b" * 64] * 2000, "innovation_sha256": ["c" * 64] * 2000,
                        "warmup_trajectory_hashes": [{"step": step, "parameters": "a" * 64,
                                                     "raw_gradient": "b" * 64, "applied_gradient": "b" * 64}
                                                    for step in range(1, 101)],
                        "checkpoint_comparisons": [{"name": name, "step": step, "bitwise_equal": True}
                                                   for name, step in (("final", 2000), ("min_val_ce", 500),
                                                                      ("max_val_accuracy", 200), ("warmup100", 100))]})
    rows = []
    for seed in s.SEEDS:
        for index, step in enumerate(s.STEPS):
            ref = {"metrics": {name: 1. for name in s.REFERENCE_METRICS}, "null_reasons": {},
                   "diagnostics": {"reference_projector_valid": True, "positive_rank": 100, "boundary_relative_gap": .01},
                   "energies": {probe: {"input_squared_norm": index + 1., "output_squared_norm": index + 1.}
                                for probe in s.PROBES}}
            ref["metrics"]["clean_minus_corruption_retention"] = 0.
            observers = {}
            for width, size in zip(s.WIDTHS, (32, 128)):
                value = .5 if size == 32 else .5 + (seed - 2) / 10
                metrics = {name: value for name in s.OBSERVER_METRICS}
                for probe in s.PROBES:
                    metrics[f"native_{probe}_retention"] = .1 * (index + 1)
                metrics["native_clean_minus_corruption_retention"] = 0.
                metrics["self_inclusion_retention_increment"] = 0.
                observers[width] = {"metrics": metrics, "estimation_width": size, "actual_rank": size,
                                    "null_reasons": {}, "diagnostics": {},
                                    "energies": {"probes": {probe: {"input_squared_norm": index + 1.,
                                                                     "output_squared_norm": .1 * (index + 1) ** 2}
                                                             for probe in s.PROBES}, "joint": {},
                                                 "reference_squared_frobenius": 1., "estimator_squared_frobenius": 1.,
                                                 "covariance_inner_product": 1 - value ** 2 / 2,
                                                 "covariance_squared_error_raw": value ** 2,
                                                 "span_captured_energy": value,
                                                 "represented_operator_output_energy": value}}
            rows.append({"schema_version": 1, "seed": seed, "step": step,
                         "state_timing": "pre_adam_after_current_gradient_observation",
                         "reference": ref, "observers": observers, "all_numerical_gates_passed": True,
                         "artifacts": []})
    return execution, rows, replays


def make_null(row, source, name, reason="synthetic_unavailable"):
    block = row["reference"] if source == "reference" else row["observers"][source]
    block["metrics"][name] = None
    block["null_reasons"][name] = reason


def deficient_observer(row, width):
    row["observers"][width]["actual_rank"] = 31
    make_null(row, width, s.PRIMARY, "observer_rank_below_32")
    make_null(row, width, "span_projector_distance", "observer_rank_below_32")


class SummaryTests(unittest.TestCase):
    def test_primary_direction_all_pairs_sample_sd(self):
        result = s.build_summary(*fixture())
        primary = result["primary"]
        self.assertEqual(primary["valid_seed_mask"], [3, 4, 5])
        for actual, expected in zip(primary["individual_values"], (.1, .2, .3)):
            self.assertAlmostEqual(actual["value"], expected)
        self.assertAlmostEqual(primary["complete_case_statistics"]["mean"], .2)
        self.assertAlmostEqual(primary["complete_case_statistics"]["sample_sd"], .1)
        self.assertNotIn("secondary_available_case_statistics", primary)
        self.assertEqual(len(result["raw_snapshot_records"]), 12)

    def test_null_primary_does_not_substitute_available_case(self):
        execution, rows, replays = fixture()
        deficient_observer(rows[-1], "width128")
        result = s.build_summary(execution, rows, replays)
        self.assertIsNone(result["primary"]["complete_case_statistics"])
        self.assertEqual(result["primary"]["valid_seed_mask"], [3, 4])
        secondary = result["fixed_step_secondary"]["2000"]["width128_minus_width32"][s.PRIMARY]
        self.assertAlmostEqual(secondary["secondary_available_case_statistics"]["mean"], .15)

    def test_boundary_tie_does_not_null_primary(self):
        execution, rows, replays = fixture()
        for row in rows:
            row["reference"]["diagnostics"]["reference_projector_valid"] = False
            row["reference"]["diagnostics"]["boundary_relative_gap"] = 0.
            for width in s.WIDTHS:
                make_null(row, width, "span_projector_distance", "boundary_tie")
            for probe in s.PROBES:
                make_null(row, "reference", f"{probe}_retention", "boundary_tie")
                row["reference"]["energies"][probe]["output_squared_norm"] = None
            make_null(row, "reference", "clean_minus_corruption_retention", "boundary_tie")
        result = s.build_summary(execution, rows, replays)
        self.assertIsNotNone(result["primary"]["complete_case_statistics"])
        self.assertIsNone(result["complete_four_snapshot_means"]["reference"]["clean_retention"]["complete_case_statistics"])

    def test_four_snapshot_requires_all_and_matching_pairs(self):
        execution, rows, replays = fixture()
        deficient_observer(rows[0], "width32")
        deficient_observer(rows[1], "width128")
        result = s.build_summary(execution, rows, replays)
        pooled = result["complete_four_snapshot_means"]["width128_minus_width32"][s.PRIMARY]
        self.assertIsNone(pooled["complete_case_statistics"])
        self.assertEqual(pooled["individual_values"][0]["valid_step_mask"], [1000, 2000])
        self.assertIsNone(pooled["individual_values"][0]["value"])
        self.assertIsNotNone(result["primary"]["complete_case_statistics"])

    def test_arithmetic_and_energy_weighted_retention_distinct(self):
        result = s.build_summary(*fixture())
        arithmetic = result["complete_four_snapshot_means"]["width32"]["native_clean_retention"]
        weighted = result["secondary_energy_weighted_four_snapshot_ratios"]["width32"]["clean"]
        self.assertAlmostEqual(arithmetic["complete_case_statistics"]["mean"], .25)
        self.assertAlmostEqual(weighted["complete_case_statistics"]["mean"], .3)
        self.assertAlmostEqual(weighted["individual_values"][0]["summed_input_squared_norm"], 10)

    def test_zero_input_does_not_become_zero_retention(self):
        execution, rows, replays = fixture()
        for row in rows:
            for width in s.WIDTHS:
                block = row["observers"][width]
                block["energies"]["probes"]["clean"] = {"input_squared_norm": 0., "output_squared_norm": 0.}
                make_null(row, width, "native_clean_retention", "zero_input_norm")
                make_null(row, width, "native_clean_minus_corruption_retention", "zero_input_norm")
            row["reference"]["energies"]["clean"] = {"input_squared_norm": 0., "output_squared_norm": 0.}
            make_null(row, "reference", "clean_retention", "zero_input_norm")
            make_null(row, "reference", "clean_minus_corruption_retention", "zero_input_norm")
        result = s.build_summary(execution, rows, replays)
        weighted = result["secondary_energy_weighted_four_snapshot_ratios"]["width32"]["clean"]
        self.assertIsNone(weighted["complete_case_statistics"])

    def test_incomplete_duplicate_failed_or_wrong_phase_rejected(self):
        changes = (
            lambda e, r, p: e.update(status="reference"),
            lambda e, r, p: e.update(mode="pilot"),
            lambda e, r, p: e.update(official_test_opened=True),
            lambda e, r, p: e["completed_snapshots"].pop(),
            lambda e, r, p: r.__setitem__(0, copy.deepcopy(r[1])),
            lambda e, r, p: r[0].update(all_numerical_gates_passed=False),
            lambda e, r, p: r[0].update(state_timing="post_adam"),
            lambda e, r, p: p[0].update(all_historical_gates_passed=False),
            lambda e, r, p: p[0].update(snapshot_count=3),
            lambda e, r, p: p[0]["raw_gradient_sha256"].pop(),
        )
        for change in changes:
            with self.subTest(change=change):
                args = fixture()
                change(*args)
                with self.assertRaises(ValueError):
                    s.build_summary(*args)

    def test_schema_null_nonfinite_and_energy_inconsistency_rejected(self):
        changes = (
            lambda b: b["metrics"].update(unknown=1),
            lambda b: b["metrics"].update(span_energy_fraction=None),
            lambda b: b["metrics"].update(span_energy_fraction=True),
            lambda b: b["metrics"].update(span_energy_fraction=float("nan")),
            lambda b: b["diagnostics"].update(bad=float("inf")),
            lambda b: b["metrics"].update(native_clean_retention=.999),
            lambda b: b["energies"]["probes"]["clean"].update(output_squared_norm=None),
        )
        for change in changes:
            args = fixture()
            change(args[1][0]["observers"]["width32"])
            with self.assertRaises(ValueError):
                s.build_summary(*args)

    def test_no_input_mutation(self):
        args = fixture()
        before = copy.deepcopy(args)
        s.build_summary(*args)
        self.assertEqual(args, before)

    def test_invalid_scientific_validity_is_rejected(self):
        changes = (
            lambda r: r["observers"]["width32"].update(actual_rank=31),
            lambda r: r["reference"]["diagnostics"].update(positive_rank=31),
            lambda r: r["reference"]["metrics"].update(optimal_rank32_energy=0),
            lambda r: r["reference"]["diagnostics"].update(reference_projector_valid=False),
            lambda r: r["reference"]["diagnostics"].update(boundary_relative_gap=0),
            lambda r: r["observers"]["width128"]["metrics"].update(span_energy_fraction=1.1),
        )
        for change in changes:
            args = fixture()
            change(args[1][0])
            with self.assertRaises(ValueError):
                s.build_summary(*args)

    def test_source_bindings_must_be_complete_exact_and_valid(self):
        valid = {name: "a" * 64 for name in s.expected_source_paths()}
        s.validate_source_map(valid)
        for invalid in ({}, {**valid, "unexpected": "a" * 64},
                        {key: value for key, value in valid.items() if key != next(iter(valid))},
                        {**valid, next(iter(valid)): "not-a-hash"}):
            with self.assertRaises(ValueError):
                s.validate_source_map(invalid)

    def test_incomplete_execution_is_rejected_before_reading_outcomes(self):
        for change in ({"status": "reference"}, {"mode": "pilot"}):
            execution, _, _ = fixture()
            execution.update(change)
            with mock.patch.object(s, "read_json", return_value=execution) as reader:
                with self.assertRaises(ValueError):
                    s.summarize_directory(Path("/synthetic/no-outcomes"))
                reader.assert_called_once_with(Path("/synthetic/no-outcomes/execution.json"))

    def test_raw_energies_derived_metrics_and_anchors_are_strict(self):
        changes = (
            lambda e, r, p: r[0]["observers"]["width32"]["energies"].pop("span_captured_energy"),
            lambda e, r, p: r[0]["observers"]["width32"]["energies"].update(span_captured_energy=.99),
            lambda e, r, p: r[0]["observers"]["width32"]["metrics"].update(self_inclusion_retention_increment=.1),
            lambda e, r, p: r[0]["observers"]["width32"]["metrics"].update(native_clean_minus_corruption_retention=.1),
            lambda e, r, p: p[0]["checkpoint_comparisons"][0].update(step=0),
            lambda e, r, p: p[0]["warmup_trajectory_hashes"].__setitem__(0, {}),
        )
        for change in changes:
            args = fixture()
            change(*args)
            with self.assertRaises(ValueError):
                s.build_summary(*args)
        args = fixture()
        block = args[1][0]["observers"]["width32"]
        block["energies"]["probes"]["clean"] = {"input_squared_norm": 0., "output_squared_norm": 1.}
        args[1][0]["reference"]["energies"]["clean"]["input_squared_norm"] = 0.
        make_null(args[1][0], "width32", "native_clean_retention")
        make_null(args[1][0], "width32", "native_clean_minus_corruption_retention")
        with self.assertRaisesRegex(ValueError, "Zero input"):
            s.build_summary(*args)

    def test_common_probe_and_reference_energy_are_required(self):
        for key in ("reference", "probe"):
            args = fixture()
            energy = args[1][0]["observers"]["width32"]["energies"]
            if key == "reference":
                energy["reference_squared_frobenius"] = 2.
            else:
                energy["probes"]["clean"]["input_squared_norm"] = 2.
                energy["probes"]["clean"]["output_squared_norm"] = .2
            with self.assertRaisesRegex(ValueError, "do not share"):
                s.build_summary(*args)

    def test_duplicate_json_and_overwrite_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "summary.json"
            s.write_new(path, {"safe": 1})
            with self.assertRaises(FileExistsError):
                s.write_new(path, {"changed": 2})
            self.assertEqual(s.read_json(path), {"safe": 1})
        with self.assertRaises(ValueError):
            json.loads('{"a": 1, "a": 2}', object_pairs_hook=s.reject_duplicates)


if __name__ == "__main__":
    unittest.main()
