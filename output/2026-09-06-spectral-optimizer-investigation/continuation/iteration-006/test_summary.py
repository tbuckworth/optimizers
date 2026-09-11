"""Adversarial CPU/stdlib-only fixtures; no experiment or dataset imports."""
import copy
import json
import math
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import summarize_results as summary


def evaluation(accuracy=.5, ce=1.0, count=5000):
    return {"accuracy": accuracy, "cross_entropy": ce, "count": count}


def selector_fixture():
    rows = [dict(step=step, **evaluation(.5, 1.0)) for step in range(0, 2001, 100)]
    # Accuracy ties at 100/200 despite different CE; CE ties at 200/300 despite accuracy.
    rows[1].update(accuracy=.8, cross_entropy=.8)
    rows[2].update(accuracy=.8, cross_entropy=.5)
    rows[3].update(accuracy=.7, cross_entropy=.5)
    return {"validation_trajectory": rows,
            "checkpoint_steps": {"final": 2000, "warmup100": 100,
                                 "max_val_accuracy": 100, "min_val_ce": 200}}


def step_fixture(arm="current32", step=101, *, zero=False, lagged_length=4.):
    """Independent two-dimensional scalar fixture; no policy/torch import."""
    raw = (0., 0.) if zero else (3., 4.)
    observing, active = arm != "adamw", arm != "adamw" and step > 100
    current = None if not active else ((0., 0.) if zero else (3., 0.))
    lagged = None if not active else ((0., 0.) if zero else (0., lagged_length))
    dot = lambda a, b: math.fsum(x * y for x, y in zip(a, b))
    norm = lambda a: math.sqrt(dot(a, a))
    if not active:
        direction, target, name, operator = raw, norm(raw), "raw", "identity"
    elif arm == "current32":
        direction, target, name, operator = current, norm(current), "current", "native_current"
    elif arm == "lagged32":
        direction, target, name, operator = lagged, norm(lagged), "lagged", "native_lagged"
    elif arm == "lagged32_current_norm":
        direction, target, name, operator = lagged, norm(current), "lagged", "scaled_native_lagged"
    else:
        direction, name, operator = raw, "raw", "scalar_identity"
        target = norm(current if arm == "scalar_current32" else lagged)
    scaled = active and arm in ("lagged32_current_norm", "scalar_current32", "scalar_lagged32")
    scale = (0. if target == 0 else target / norm(direction)) if scaled else None
    applied = direction if scale is None else tuple(x * scale for x in direction)
    row = {"step": step, "null_reasons": {}}
    unavailable = "baseline_no_observer" if not observing else "warmup_candidates_not_measured"
    vectors = {"raw": raw, "current": current, "lagged": lagged, "applied": applied}

    def put(key, value, reason=None):
        row[key] = value
        if value is None:
            row["null_reasons"][key] = reason

    for component, vector in vectors.items():
        put(component + "_norm", None if vector is None else norm(vector), unavailable)
        put(component + "_squared_norm", None if vector is None else dot(vector, vector), unavailable)
    for component in ("current", "lagged", "applied"):
        vector = vectors[component]
        reason = unavailable if vector is None else "zero_raw_norm"
        put(component + "_raw_norm_ratio", None if vector is None or zero else norm(vector) / norm(raw), reason)
        if component != "applied":
            put(component + "_energy_retention", None if vector is None or zero else dot(vector, vector) / dot(raw, raw), reason)
    put("current_minus_lagged_retention", None if not active or zero else
        row["current_energy_retention"] - row["lagged_energy_retention"], unavailable if not active else "zero_raw_norm")
    for key, left, right in (("current_lagged_cosine", current, lagged), ("raw_applied_cosine", raw, applied)):
        missing = left is None or right is None
        denominator = 0. if missing else norm(left) * norm(right)
        put(key, None if denominator == 0 else dot(left, right) / denominator,
            unavailable if missing else "zero_vector_norm")
    put("policy_scale", scale, "unscaled_policy")
    put("norm_matching_relative_error", 0. if target == 0 else abs(norm(applied) - target) / target)
    put("direction_relative_error", None if target == 0 else 0., "zero_target_and_delivery")
    rank = min(max(step - 1, 0), 32)
    previous_rank = min(max(step - 2, 0), 32)
    row["policy"] = {
        "arm": arm, "observing_step": step if observing else None, "policy_active": active,
        "active_policy": arm if active else "identity_warmup" if observing else "identity_baseline",
        "current_basis_rank": rank if observing else None,
        "lagged_basis_rank": previous_rank if observing else None,
        "current_basis_missing": (rank == 0) if observing else None,
        "lagged_basis_missing": (previous_rank == 0) if observing else None,
        "current_candidate_operator": "native_basis" if active else None,
        "lagged_candidate_operator": "native_basis" if active else None,
        "scale": scale, "scale_direction": name, "target_norm": target, "delivery_operator": operator}
    for kind in ("total", "decay_subtracted"):
        update_vector = (-.03, -.04)
        update = {"norm": norm(update_vector), "squared_norm": dot(update_vector, update_vector), "null_reasons": {}}
        for component in ("raw", "applied"):
            product = norm(vectors[component]) * norm(update_vector)
            value = dot(vectors[component], update_vector)
            tolerance = 1e-6 * product + 1e-14
            update[component + "_gradient_dot_update"] = {
                "value": value, "tolerance": tolerance,
                "sign": 1 if value > tolerance else -1 if value < -tolerance else 0}
            key = component + "_gradient_update_cosine"
            update[key] = None if product == 0 else value / product
            if product == 0:
                update["null_reasons"][key] = "zero_vector_norm"
        row[kind] = update
    return row


def run_fixture(seed=6, noise=.9, arm="current32"):
    result = selector_fixture()
    key = (seed, noise, arm)
    result.update(schema_version=1, run_key=summary.run_key(key), seed=seed,
                  replacement_probability=noise, arm=arm, steps=2000, instrumented=True,
                  all_invariant_gates_passed=True, warmup_checks_passed=True,
                  delivery_gate_steps=2000, measurement_state_checks=22,
                  realized_incorrect_fraction=.81 if noise else 0.,
                  realized_replacement_fraction=.9 if noise else 0., plan_sha256="a" * 64)
    result["steps_raw"] = [step_fixture(arm, step) for step in range(1, 2001)]
    result["trajectory_parameter_sha256"] = ["b" * 64] * 2000
    result["estimation_rank_by_step"] = [min(max(step - 1, 0), 32) if arm != "adamw" else 0 for step in range(1, 2001)]
    result["repair_count_by_step"] = [step // 100 if arm != "adamw" else 0 for step in range(1, 2001)]
    result["warmup_trajectory_hashes"] = [{"step": step, "parameters": "b" * 64,
                                          "raw_gradient": "c" * 64, "applied_gradient": "c" * 64}
                                         for step in range(1, 101)]
    result["warmup_core_sha256"] = "d" * 64
    result["warmup_observer_hashes"] = ["e" * 64] * 100 if arm != "adamw" else []
    result["warmup_observer_sha256"] = "e" * 64 if arm != "adamw" else None
    result["checkpoint_sha256"] = {name: "f" * 64 for name in summary.CHECKPOINTS}
    result["checkpoint_path"] = "/synthetic/unused-checkpoint.pt"
    result["test"] = {name: evaluation(count=10000) for name in summary.CHECKPOINTS}
    result["final_training_clean"] = evaluation()
    result["final_training_noisy"] = evaluation()
    result["final_validation"] = evaluation()
    result["step_elapsed_seconds"] = [.001] * 2000
    result["elapsed_seconds"] = 2.
    result["resources"] = {}
    return result


def normalized_fixture():
    runs = []
    for seed in summary.SEEDS:
        for noise in summary.NOISES:
            for index, arm in enumerate(summary.ARMS):
                primary = .5 + ((-1 if seed == 6 else 1) * index * .001)
                counts = {"geometry": {"null_steps": []}, "candidate": {"null_steps": [101] if arm == "adamw" else []}}
                runs.append({"seed": seed, "replacement_probability": noise, "arm": arm,
                             "metrics": {summary.PRIMARY_METRIC: primary, "geometry": index / 10,
                                         "candidate": None if arm == "adamw" else index / 20},
                             "counts": counts})
    return runs


def execution_fixture():
    cells = [{"seed": seed, "replacement_probability": noise, "arm": arm,
              "run_key": summary.run_key((seed, noise, arm))}
             for seed in summary.SEEDS for noise in summary.NOISES for arm in summary.ARMS]
    return {"schema_version": 1, "mode": "full", "status": "complete", "all_gates_passed": True,
            "completed_runs": 36, "test_evaluations": 144, "warmup_checks_passed": True,
            "official_test_opened": True, "completed_cells": cells,
            "started_utc": "2026-09-06T14:00:00+00:00",
            "all_training_completed_utc": "2026-09-06T14:02:00+00:00",
            "test_first_opened_utc": "2026-09-06T14:02:01+00:00",
            "completed_utc": "2026-09-06T14:02:03+00:00",
            "runs": copy.deepcopy(cells), "training_runs": copy.deepcopy(cells), "checkpoints": copy.deepcopy(cells),
            "plans": [{"seed": seed} for seed in summary.SEEDS], "artifact_total_bytes": 100,
            "resources": {"elapsed_seconds": 123., "peak_rss_bytes": 1000,
                          "peak_gpu_allocated_bytes": 2000, "peak_gpu_reserved_bytes": 3000},
            "environment": {"python": "synthetic", "numpy": "synthetic", "torch": "synthetic", "cuda": "synthetic",
                            "gpu": "synthetic", "cpu_threads": 1, "deterministic_algorithms": True,
                            "tf32": False, "cudnn_benchmark": False, "cublas_workspace": ":4096:8",
                            "foreach": False, "fused": False}}


def pilot_fixture(directory):
    """Tiny synthetic provenance files; no outcomes, tensors, dataset or Git commit."""
    directory = Path(directory)

    def bind(name, value):
        path = directory / name
        path.write_text(json.dumps(value))
        return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": summary.sha(path)}

    reports = []
    for arm in summary.ARMS:
        observing = arm != "adamw"
        warmup = [{"step": step, "parameters": "a" * 64, "raw_gradient": "b" * 64,
                   "applied_gradient": "b" * 64} for step in range(1, 101)]
        report = {"arm": arm, "trajectory_bitwise_identical": True,
                  "final_state_bitwise_identical": True, "warmup_checks_passed": True}
        for name, enabled in (("uninstrumented", False), ("instrumented", True)):
            report[name] = {"seed": 9880, "replacement_probability": .9, "arm": arm,
                            "steps": 220, "instrumented": enabled, "delivery_gate_steps": 220,
                            "measurement_state_checks": 5 if enabled else 0,
                            "all_invariant_gates_passed": True, "warmup_checks_passed": True,
                            "trajectory_parameter_sha256": ["a" * 64] * 220,
                            "warmup_trajectory_hashes": copy.deepcopy(warmup),
                            "warmup_core_sha256": "c" * 64,
                            "warmup_observer_hashes": ["d" * 64] * 100 if observing else [],
                            "warmup_observer_sha256": "d" * 64 if observing else None,
                            "estimation_rank_by_step": [min(step - 1, 32) if observing else 0 for step in range(1, 221)],
                            "repair_count_by_step": [step // 100 if observing else 0 for step in range(1, 221)],
                            "step_elapsed_seconds": [.001] * 220}
        reports.append(report)
    full = execution_fixture()
    full.update(source_sha256={"synthetic": "a" * 64}, training_data_artifacts=[])
    plan = bind("pilot-plan.json", {"synthetic": True})
    plan["seed"] = 9880
    pilot = {"mode": "pilot", "status": "complete_passed", "completed_traces": 12,
             "all_gates_passed": True, "warmup_checks_passed": True,
             "official_test_opened": False, "validation_or_accuracy_computed": False,
             "source_sha256": full["source_sha256"], "environment": full["environment"],
             "training_data_artifacts": [], "repository_revision": "a" * 40,
             "started_utc": "2026-09-06T13:59:00+00:00", "completed_utc": "2026-09-06T13:59:01+00:00",
             "resources": dict(full["resources"], elapsed_seconds=1.),
             "plans": [plan], "bulk_root": str(directory),
             "timing_and_invariants": bind("pilot-timings.json", reports)}
    full["passing_pilot"] = bind("pilot-execution.json", pilot)
    return full, pilot, reports, bind


class UtilityTests(unittest.TestCase):
    def test_strict_json(self):
        with tempfile.TemporaryDirectory(prefix="iteration006-summary-test-") as directory:
            path = Path(directory) / "small.json"
            for payload in ('{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}', '{"x":1e999}'):
                path.write_text(payload)
                with self.assertRaises(ValueError):
                    summary.read_json(path)
            path.write_text('{"x":null,"y":0}')
            self.assertEqual(summary.read_json(path), {"x": None, "y": 0})

    def test_output_is_exclusive_and_strict(self):
        with tempfile.TemporaryDirectory(prefix="iteration006-summary-test-") as directory:
            path = Path(directory) / "summary.json"
            summary.write_exclusive(path, {"value": 1.0})
            before = path.read_bytes()
            with self.assertRaises(FileExistsError):
                summary.write_exclusive(path, {"value": 2.0})
            self.assertEqual(path.read_bytes(), before)
            with self.assertRaises(ValueError):
                summary.write_exclusive(Path(directory) / "bad.json", {"bad": float("nan")})
            self.assertFalse((Path(directory) / "bad.json").exists())

    def test_counts_and_null_reasons(self):
        value, counts = summary.step_aggregate([2., None, 4.], [101, 102, 103],
                                               [None, "zero_raw_norm", None])
        self.assertEqual(value, 3.)
        self.assertEqual(counts["null_steps"], [102])
        self.assertEqual(counts["null_reasons"], {"102": "zero_raw_norm"})
        self.assertEqual(counts["finite_count"], 2)
        with self.assertRaises(ValueError):
            summary.step_aggregate([None], [101], [None])
        with self.assertRaises(ValueError):
            summary.step_aggregate([1.], [101], ["fake_null"])

    def test_complete_statistics_not_available_case(self):
        self.assertEqual(summary.descriptive([1., 2., 3.])["sample_sd_descriptive"], 1.)
        self.assertIsNone(summary.complete_statistics([1., None, 3.])["mean"])
        self.assertIsNone(summary.complete_statistics([1., 2.])["mean"])
        self.assertEqual(summary.complete_statistics([-1., -2., 0.])["mean"], -1.)

    def test_selector_earliest_ties_are_independent(self):
        good = selector_fixture()
        summary.validate_selectors(good)
        for selector, wrong_step in (("max_val_accuracy", 200), ("min_val_ce", 300), ("final", 1900)):
            bad = copy.deepcopy(good)
            bad["checkpoint_steps"][selector] = wrong_step
            with self.assertRaises(ValueError):
                summary.validate_selectors(bad)
        bad = copy.deepcopy(good)
        bad["validation_trajectory"].pop()
        with self.assertRaises(ValueError):
            summary.validate_selectors(bad)

    def test_cell_and_numeric_type_validation(self):
        self.assertEqual(len(summary.EXPECTED_CELLS), 36)
        self.assertEqual(len(summary.PAIRS), 15)
        for field, value in (("seed", 3), ("replacement_probability", .8), ("arm", "scalar32")):
            bad = {"seed": 6, "replacement_probability": .9, "arm": "adamw"}
            bad[field] = value
            with self.assertRaises(ValueError):
                summary.condition_key(bad)
        for value in (True, float("inf"), "1"):
            with self.assertRaises(ValueError):
                summary.number(value, "fixture")

    def test_evaluation_domain(self):
        summary.validate_evaluation(evaluation(), 5000)
        for field, value in (("accuracy", float("nan")), ("accuracy", 1.1),
                             ("accuracy", .50001), ("cross_entropy", -1), ("count", 4999)):
            bad = evaluation()
            bad[field] = value
            with self.assertRaises(ValueError):
                summary.validate_evaluation(bad, 5000)


class PolicyRecordTests(unittest.TestCase):
    def test_all_arms_active_warmup_and_zero(self):
        for arm in summary.ARMS:
            for step in (1, 100, 101, 200):
                for zero in (False, True):
                    row = step_fixture(arm, step, zero=zero)
                    rank = min(max(step - 1, 0), 32) if arm != "adamw" else 0
                    old = min(max(step - 2, 0), 32) if arm != "adamw" else 0
                    summary.validate_step(row, arm, rank, old)

    def test_restoration_is_not_scalar_contraction(self):
        row = step_fixture("lagged32_current_norm", lagged_length=.5)
        self.assertEqual(row["policy_scale"], 6.)
        summary.validate_step(row, "lagged32_current_norm", 32, 32)

    def test_reject_candidate_null_rank_operator_and_gate_confusion(self):
        good = step_fixture()
        changes = (("current_norm", None), ("policy_scale", 1.),
                   ("current_energy_retention", .99), ("norm_matching_relative_error", 1e-3),
                   ("direction_relative_error", 1e-3))
        for field, value in changes:
            bad = copy.deepcopy(good)
            bad[field] = value
            with self.assertRaises(ValueError):
                summary.validate_step(bad, "current32", 32, 32)
        for field, value in (("delivery_operator", "scalar_identity"), ("current_basis_rank", 31),
                             ("target_norm", 5.), ("observing_step", 100)):
            bad = copy.deepcopy(good)
            bad["policy"][field] = value
            with self.assertRaises(ValueError):
                summary.validate_step(bad, "current32", 32, 32)
        baseline = step_fixture("adamw")
        baseline["current_norm"] = baseline["raw_norm"]
        with self.assertRaises(ValueError):
            summary.validate_step(baseline, "adamw", 0, 0)

    def test_sign_tolerance_is_rederived_not_accepted(self):
        for field, value in (("tolerance", .1), ("sign", 1), ("value", 1.)):
            row = step_fixture()
            row["decay_subtracted"]["raw_gradient_dot_update"][field] = value
            with self.assertRaises(ValueError):
                summary.validate_step(row, "current32", 32, 32)

    def test_full_run_aggregation_exposure_and_phase_counts(self):
        run = run_fixture()
        result = summary.summarize_run(run)
        self.assertEqual(result["metrics"]["checkpoint.max_val_accuracy.postwarmup_updates"], 0)
        self.assertEqual(result["metrics"]["checkpoint.max_val_accuracy.at_or_before_warmup"], 1)
        self.assertEqual(result["counts"]["phase.scheduled_repair.current_lagged_cosine"]["scheduled_count"], 19)
        self.assertEqual(result["counts"]["phase.other_steps.current_lagged_cosine"]["scheduled_count"], 1881)
        self.assertEqual(result["counts"]["update.all.total.raw_gradient_ascent_frequency"]["scheduled_count"], 1900)
        # An unscheduled extra repair is not falsely excluded from the other-steps group.
        run["repair_count_by_step"] = [count + int(index >= 149) for index, count in enumerate(run["repair_count_by_step"])]
        summary.validate_run(run)

    def test_missing_warmup_and_incomplete_histories_fail(self):
        for field in ("warmup_trajectory_hashes", "warmup_observer_hashes", "steps_raw"):
            run = run_fixture()
            run[field].pop()
            with self.assertRaises(ValueError):
                summary.validate_run(run)

    def test_float32_realized_fractions_are_preserved(self):
        run = run_fixture()
        rate = struct.unpack("f", struct.pack("f", .81))[0]
        self.assertNotEqual(rate, .81)
        run["realized_incorrect_fraction"] = rate
        summary.validate_run(run)
        self.assertEqual(run["realized_incorrect_fraction"], rate)
        run["realized_incorrect_fraction"] = .8101
        with self.assertRaises(ValueError):
            summary.validate_run(run)

    def test_zero_cosines_do_not_change_sign_denominators(self):
        run = run_fixture()
        run["steps_raw"][100] = step_fixture(step=101, zero=True)
        result = summary.summarize_run(run)
        self.assertEqual(result["counts"]["gradient.all.current_lagged_cosine"]["null_steps"], [101])
        count = result["counts"]["update.all.decay_subtracted.raw_gradient_ascent_frequency"]
        self.assertEqual(count["finite_count"], 1900)
        self.assertEqual(count["null_count"], 0)


class ExecutionTests(unittest.TestCase):
    def test_exact_execution_and_strict_test_access_order(self):
        good = execution_fixture()
        summary.validate_execution(good)
        for field, value in (("mode", "pilot"), ("status", "training"), ("completed_runs", 35),
                             ("test_evaluations", 143), ("all_gates_passed", False),
                             ("warmup_checks_passed", False), ("official_test_opened", False),
                             ("test_first_opened_utc", good["all_training_completed_utc"]),
                             ("test_first_opened_utc", "2026-09-06T14:01:00+00:00")):
            bad = copy.deepcopy(good)
            bad[field] = value
            with self.assertRaises(ValueError):
                summary.validate_execution(bad)
        for family in ("completed_cells", "runs", "training_runs", "checkpoints"):
            bad = copy.deepcopy(good)
            bad[family][-1] = bad[family][0]
            with self.assertRaises(ValueError):
                summary.validate_execution(bad)
        bad = copy.deepcopy(good)
        bad["completed_cells"][-1]["replacement_probability"] = .8
        with self.assertRaises(ValueError):
            summary.validate_execution(bad)

    def test_hash_bound_artifacts_reject_changed_bytes_and_path_escape(self):
        with tempfile.TemporaryDirectory(prefix="iteration006-summary-test-") as directory:
            path = Path(directory) / "small.json"
            path.write_text("{}")
            binding = {"path": str(path), "sha256": summary.sha(path), "size_bytes": path.stat().st_size}
            summary.verify_binding(binding, bulk_root=directory)
            with self.assertRaises(ValueError):
                summary.verify_binding(binding, bulk_root=Path(directory) / "other")
            path.write_text("[]")
            with self.assertRaises(ValueError):
                summary.verify_binding(binding)

    def test_exact_source_membership_current_and_committed_bytes(self):
        with tempfile.TemporaryDirectory(prefix="iteration006-summary-test-") as directory:
            root = Path(directory)
            mapping = {}
            for name in summary.SOURCE_PATHS:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"synthetic source\n")
                mapping[name] = summary.sha(path)
            with mock.patch.object(summary.subprocess, "run", return_value=SimpleNamespace(stdout=b"synthetic source\n")):
                summary.verify_sources(mapping, "a" * 40, root=root)
                bad = dict(mapping)
                bad.pop(next(iter(bad)))
                with self.assertRaises(ValueError):
                    summary.verify_sources(bad, "a" * 40, root=root)
                path = root / next(iter(mapping))
                path.write_bytes(b"changed source\n")
                with self.assertRaises(ValueError):
                    summary.verify_sources(mapping, "a" * 40, root=root)
                path.write_bytes(b"synthetic source\n")
            with mock.patch.object(summary.subprocess, "run", return_value=SimpleNamespace(stdout=b"different committed source\n")):
                with self.assertRaises(ValueError):
                    summary.verify_sources(mapping, "a" * 40, root=root)

    def test_shared_warmup_is_within_condition_and_requires_observer_history(self):
        full = normalized_fixture()
        for run in full:
            witness = f'{run["seed"]}-{run["replacement_probability"]}'
            run.update(warmup_trajectory_hashes=[witness] * 100, warmup_core_sha256=witness,
                       warmup_observer_hashes=[witness] * 100 if run["arm"] != "adamw" else [],
                       warmup_observer_sha256=witness if run["arm"] != "adamw" else None,
                       plan_sha256=str(run["seed"]), realized_incorrect_fraction=run["replacement_probability"],
                       realized_replacement_fraction=run["replacement_probability"],
                       test={"warmup100": evaluation()}, checkpoint_sha256={"warmup100": witness})
        summary.validate_warmup_pairs(full)
        for key, changed in (("warmup_core_sha256", "other"), ("warmup_trajectory_hashes", []),
                             ("warmup_observer_hashes", []), ("plan_sha256", "other")):
            bad = copy.deepcopy(full)
            bad[1][key] = changed
            with self.assertRaises(ValueError):
                summary.validate_warmup_pairs(bad)

    def test_pilot_artifacts_scope_and_warmup_are_independently_checked(self):
        with tempfile.TemporaryDirectory(prefix="iteration006-summary-test-") as directory:
            full, pilot, reports, bind = pilot_fixture(directory)
            with mock.patch.object(summary, "verify_sources"):
                summary.verify_pilot(full)
                for field, value in (("completed_traces", 10), ("validation_or_accuracy_computed", True),
                                     ("official_test_opened", True), ("status", "failed")):
                    bad = copy.deepcopy(pilot)
                    bad[field] = value
                    changed = dict(full, passing_pilot=bind("changed-pilot.json", bad))
                    with self.assertRaises(ValueError):
                        summary.verify_pilot(changed)
                for kind in ("bad_hash", "missing_rank", "outcome", "warmup"):
                    bad_reports = copy.deepcopy(reports)
                    trace = bad_reports[1]["instrumented"]
                    if kind == "bad_hash":
                        trace["trajectory_parameter_sha256"][-1] = "e" * 64
                    elif kind == "missing_rank":
                        trace["estimation_rank_by_step"].pop()
                    elif kind == "outcome":
                        trace["test"] = {}
                    else:
                        trace["warmup_core_sha256"] = "e" * 64
                    bad = dict(pilot, timing_and_invariants=bind("changed-timing.json", bad_reports))
                    changed = dict(full, passing_pilot=bind("changed-pilot.json", bad))
                    with self.assertRaises(ValueError):
                        summary.verify_pilot(changed)

    def test_pilot_source_and_environment_must_equal_full(self):
        with tempfile.TemporaryDirectory(prefix="iteration006-summary-test-") as directory:
            full, _, _, _ = pilot_fixture(directory)
            for field in ("source_sha256", "environment", "training_data_artifacts"):
                bad = copy.deepcopy(full)
                bad[field] = {} if field != "training_data_artifacts" else [{"changed": True}]
                with self.assertRaises(ValueError), mock.patch.object(summary, "verify_sources"):
                    summary.verify_pilot(bad)


class GroupTests(unittest.TestCase):
    def test_exact_four_primary_groups_preserve_adverse_seeds(self):
        result = summary.aggregate_seed_summaries(normalized_fixture())
        self.assertEqual(len(result["primary"]), 4)
        self.assertEqual(len(result["paired_contrasts"]), 15 * 2 * 3)
        for primary in result["primary"]:
            self.assertLess(primary["paired_differences"][0]["difference"], 0)
            self.assertGreater(primary["paired_differences"][1]["difference"], 0)

    def test_incomplete_duplicate_and_wrong_conditions_fail(self):
        for alteration in ("missing", "duplicate", "wrong_noise", "missing_primary"):
            runs = normalized_fixture()
            if alteration == "missing":
                runs.pop()
            elif alteration == "duplicate":
                runs[-1] = runs[0]
            elif alteration == "wrong_noise":
                runs[-1]["replacement_probability"] = .8
            else:
                runs[-1]["metrics"][summary.PRIMARY_METRIC] = None
            with self.assertRaises(ValueError):
                summary.aggregate_seed_summaries(runs)

    def test_mismatched_step_masks_invalidate_group_not_other_pairs(self):
        runs = normalized_fixture()
        for run in runs:
            if (run["seed"], run["replacement_probability"], run["arm"]) == (6, .9, "lagged32"):
                run["counts"]["geometry"]["null_steps"] = [102]
        result = summary.aggregate_seed_summaries(runs)
        contrast = next(row for row in result["paired_contrasts"] if row["replacement_probability"] == .9
                        and row["treatment"] == "lagged32" and row["control"] == "current32" and row["metric"] == "geometry")
        self.assertEqual(contrast["unavailable_seeds"], [6])
        self.assertIsNone(contrast["mean"])
        self.assertIsNotNone(contrast["paired_differences"][1]["difference"])
        baseline = next(row for row in result["arm_groups"] if row["arm"] == "adamw" and row["metric"] == "candidate")
        self.assertIsNone(baseline["mean"])


if __name__ == "__main__":
    unittest.main()
