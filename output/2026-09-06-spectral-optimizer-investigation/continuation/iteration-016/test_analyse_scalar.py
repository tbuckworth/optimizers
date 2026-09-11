"""Synthetic CPU tests for the prospective I16 scalar analyzer."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch

import analyse_scalar as analysis


def _threads() -> None:
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        if torch.get_num_interop_threads() != 1:
            raise


def _leakage(norm: float, fraction: float = .25) -> dict:
    energy = norm * norm
    return {"squared_norm": energy, "outside_squared_norm": energy * fraction,
            "fraction": fraction, "reason": None}


def _component(norm: float = 2.0, fraction: float = .25) -> dict:
    return {"norm": norm, "squared_energy": norm * norm,
            "current_basis_leakage": _leakage(norm, fraction)}


def _evaluation(offset: float = 0.0) -> dict:
    fixed, soft = 1.2 + offset, 1.0 + offset
    return {
        "train": {"clean_ce": .8 + offset, "soft_ce": soft,
            "clean_accuracy": .6, "mean_max_probability": .7,
            "mean_true_label_probability": .5, "fixed_ce": fixed,
            "fixed_accuracy": .4, "fixed_minus_soft_ce": fixed - soft},
        "validation": {"clean_ce": .9 + offset, "soft_ce": 1.1 + offset,
            "clean_accuracy": .55, "mean_max_probability": .65,
            "mean_true_label_probability": .45},
        "auxiliary": {"clean_ce": .85 + offset, "soft_ce": 1.05 + offset,
            "clean_accuracy": .57, "mean_max_probability": .66,
            "mean_true_label_probability": .46},
    }


def _step(step: int, k: float) -> dict:
    label = next(name for name, value in analysis.K_BY_LABEL.items() if value == k)
    components = {name: _component() for name in (
        "raw_gradient", "post_mean", "native_projection_diagnostic", "applied_delivery",
        "old_momentum_buffer", "scaled_old_momentum_buffer", "new_momentum_buffer")}
    data_norm = 2.0
    return {"step": step, "relative_step": step - 100,
        "path_statistics": {"data_step_squared_energy": data_norm * data_norm,
            "data_path_length_increment": data_norm, "raw_gradient_dot_data_step": -1.0},
        "schema": "i16_scalar_step_v1", "base": "sgdm", "lr": .03, "k": k,
        "loss": 1.0,
        "observer": {"step_before": step - 1, "step_after": step,
            "filtering_active": True, "used": True, "basis_rank": 2},
        "gradient_filter_applied": False, "native_projection_used_for_delivery": False,
        "gradient": {"raw_norm": 2.0, "post_mean_norm": 2.0,
            "native_projection_norm": 2.0, "applied_norm": 2.0},
        "components": components,
        "temporal_control": {"old_history_scale": k, "raw_gradient_weight": k,
            "post_mean_weight": 1.0 - k, "effective_old_history_coefficient": .9 * k,
            "delivery_literal": "post_mean" if k == 0 else None,
            "history_literal": "zero" if k == 0 else None,
            "expected_buffer_norm": 2.0, "empirical_ideal_data_step_defect_norm": 1e-7},
        "displacement": {"total_norm": 2.0, "data_norm": data_norm,
            "nominal_decay_norm": .1, "total_current_basis_leakage": _leakage(2.0),
            "data_current_basis_leakage": _leakage(data_norm),
            "raw_gradient_dot_data_delta": -1.0,
            "post_mean_dot_data_delta": -.5, "applied_gradient_dot_data_delta": -.75},
        "decay": {"coefficient": .01, "factor": .9997,
            "manual_before_optimizer_step": True, "manual_actual_norm": .1,
            "manual_minus_nominal_norm": 1e-8},
        "algebra_residuals": {name: {"norm": 0.0, "homogeneous_error_bound": 1e-5}
            for name in ("post_mean_recurrence", "delivery_definition",
                         "scaled_old_buffer_definition", "momentum_buffer_recurrence")},
        "digests": ({name: hashlib.sha256((label + name).encode()).hexdigest()
            for name in ("raw_gradient", "post_observer", "old_momentum_buffer",
                         "applied_gradient", "new_momentum_buffer")}
            if step == 101 else None)}


def _branch(seed: int, target: str, label: str, *, complete: bool = True) -> dict:
    k = analysis.K_BY_LABEL[label]
    steps = [_step(step, k) for step in range(101, 111)]
    curve = [{"horizon": 100, **_evaluation()}, {"horizon": 110, **_evaluation(-.1)}]
    checkpoints = [{"horizon": 110, "relative_horizon": 10,
        "full_state": {"name": "state.pt", "bytes": 0, "sha256": "0" * 64},
        "full_state_digest": "1" * 64}]
    failure = None
    if not complete:
        steps = steps[:-1]
        curve = curve[:1]
        checkpoints = []
        failure = {"type": "NumericalFailure", "message": "synthetic nonfinite",
            "attempted_step": 110, "completed_step": 109, "last_valid_horizon": 100,
            "state_artifact": {"name": "failed.pt", "bytes": 0, "sha256": "2" * 64}}
    return {"schema": "i16_scalar_branch_v1", "id": f"s{seed}-sgdm-{target}-{label}",
        "seed": seed, "target": target, "k": k, "k_label": label,
        "base": "sgdm", "lr": .03, "parent_horizon": 100,
        "parent_state_digest": "3" * 64, "parent_evaluation_digest": "4" * 64,
        "status": "complete" if complete else "numerical_failure",
        "requested_updates": 10, "completed_updates": len(steps),
        "last_completed_horizon": 100 + len(steps), "curve": curve,
        "scalar_update_seconds": .5, "steps": steps, "checkpoints": checkpoints,
        "failure": failure, "first_step_digests": steps[0]["digests"] if steps else None,
        "selected_horizons": analysis._selection(curve) if complete else None}


def _policy_branch(seed: int, target: str, policy: str,
                   validation_ce: float, validation_accuracy: float,
                   auxiliary_ce: float, auxiliary_accuracy: float) -> dict:
    curve = []
    for horizon in analysis.HORIZONS:
        point = {"horizon": horizon, **_evaluation()}
        point["validation"]["clean_ce"] = validation_ce
        point["validation"]["clean_accuracy"] = validation_accuracy
        point["auxiliary"]["clean_ce"] = auxiliary_ce
        point["auxiliary"]["clean_accuracy"] = auxiliary_accuracy
        curve.append(point)
    return {"seed": seed, "target": target, "policy": policy,
            "status": "complete", "curve": curve, "steps": [], "failure": None}


def _record(path: Path) -> dict:
    return {"name": path.name, "bytes": path.stat().st_size,
            "sha256": analysis.sha256(path)}


def _summary_values(point: dict) -> dict:
    return {"train_clean_ce": point["train"]["clean_ce"],
        "train_soft_ce": point["train"]["soft_ce"],
        "train_clean_accuracy": point["train"]["clean_accuracy"],
        "train_fixed_ce": point["train"]["fixed_ce"],
        "train_fixed_minus_soft_ce": point["train"]["fixed_minus_soft_ce"],
        "train_fixed_accuracy": point["train"]["fixed_accuracy"],
        "train_confidence": point["train"]["mean_max_probability"],
        "train_true_label_probability": point["train"]["mean_true_label_probability"],
        "validation_clean_ce": point["validation"]["clean_ce"],
        "validation_clean_accuracy": point["validation"]["clean_accuracy"],
        "validation_confidence": point["validation"]["mean_max_probability"],
        "validation_true_label_probability": point["validation"]["mean_true_label_probability"],
        "auxiliary_clean_ce": point["auxiliary"]["clean_ce"],
        "auxiliary_clean_accuracy": point["auxiliary"]["clean_accuracy"],
        "auxiliary_confidence": point["auxiliary"]["mean_max_probability"],
        "auxiliary_true_label_probability": point["auxiliary"]["mean_true_label_probability"]}


class ScalarAnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _threads()

    def test_step_and_smoke_branch_schema(self) -> None:
        audit = analysis.Audit()
        value = analysis.validate_branch(_branch(216, "clean", "k0p5"),
                                         "smoke", audit, "synthetic")
        self.assertIsNotNone(value)
        self.assertEqual(audit.errors, [])

    def test_step_rejects_residual_and_accepts_nonideal_action_fraction(self) -> None:
        row = _step(101, .5)
        row["components"]["raw_gradient"]["current_basis_leakage"] = _leakage(2.0, 1.2)
        audit = analysis.Audit()
        analysis.validate_step(row, 101, .5, audit, "nonideal")
        self.assertEqual(audit.errors, [])
        bad = copy.deepcopy(row)
        bad["algebra_residuals"]["delivery_definition"]["norm"] = 1.0
        audit = analysis.Audit()
        analysis.validate_step(bad, 101, .5, audit, "bad")
        self.assertTrue(any("exceeds saved bound" in error for error in audit.errors))

    def test_evaluation_stage_failure_is_retained(self) -> None:
        branch = _branch(216, "fixed", "k0")
        branch["status"] = "numerical_failure"
        branch["curve"] = branch["curve"][:1]
        branch["checkpoints"] = []
        branch["failure"] = {"type": "NumericalFailure", "message": "evaluation nonfinite",
            "attempted_step": 110, "completed_step": 110, "last_valid_horizon": 100,
            "state_artifact": {"name": "failed.pt", "bytes": 0, "sha256": "2" * 64}}
        branch["selected_horizons"] = None
        audit = analysis.Audit()
        analysis.validate_branch(branch, "smoke", audit, "evaluation failure")
        self.assertEqual(audit.errors, [])

    def test_joint_selector_ties_then_missing_member_refuses_survivors(self) -> None:
        branches = {}
        for seed in analysis.SEEDS:
            for target in analysis.TARGETS:
                for label, ce, accuracy, aux_ce, aux_accuracy in (
                        ("k0", 1.0, .8, .7, .4), ("k0p5", 1.0, .8, .6, .5),
                        ("k0p9", 1.2, .7, .5, .6), ("k1", 1.3, .6, .4, .7),
                        (analysis.SPECTRAL, .9, .9, .2, .9)):
                    branches[seed, target, label] = _policy_branch(
                        seed, target, label, ce, accuracy, aux_ce, aux_accuracy)
        choices, primary, per_k = analysis.selection_results(branches)
        self.assertTrue(all(row["scalar"]["k"] == 0.0 for row in choices))
        self.assertTrue(all(row["scalar"]["horizon"] == 100 for row in choices))
        ce_rows = [row for row in primary if row["auxiliary_metric"] == "ce"]
        accuracy_rows = [row for row in primary if row["auxiliary_metric"] == "accuracy"]
        self.assertTrue(all(all(abs(value - .5) < 1e-12
                                for value in row["effect"]["all_three_seed_values"])
                            for row in ce_rows))
        self.assertTrue(all(all(abs(value - .5) < 1e-12
                                for value in row["effect"]["all_three_seed_values"])
                            for row in accuracy_rows))
        self.assertEqual(len(primary), 8)
        self.assertEqual(len(per_k), 32)
        branches[200, "clean", "k0p9"]["status"] = "numerical_failure"
        _, broken, fixed = analysis.selection_results(branches)
        broken_row = next(row for row in broken if row["target"] == "clean")
        self.assertFalse(broken_row["effect"]["available"])
        self.assertIsNone(broken_row["effect"]["mean"])
        fixed_row = next(row for row in fixed if row["target"] == "clean"
                         and row["k_label"] == "k0")
        self.assertTrue(fixed_row["effect"]["available"])

    def test_geometry_handles_legacy_k1_without_new_scalar_dots(self) -> None:
        branches = {}
        for seed in analysis.SEEDS:
            for target in analysis.TARGETS:
                for label in analysis.NEW_LABELS:
                    branches[seed, target, label] = {
                        "status": "complete", "steps": [_step(101, analysis.K_BY_LABEL[label])]}
                branches[seed, target, analysis.SPECTRAL] = {
                    "status": "complete", "steps": [_step(101, .5)]}
                old_rows = []
                for step in range(101, 111):
                    old = _step(step, .5)
                    old["displacement"].pop("post_mean_dot_data_delta")
                    old["displacement"].pop("applied_gradient_dot_data_delta")
                    old_rows.append(old)
                branches[seed, target, "k1"] = {"status": "complete", "steps": old_rows}
        paths, components = analysis.geometry_results(branches)
        legacy = next(row for row in paths if row["target"] == "clean"
                      and row["k_label"] == "k1" and row["window"] == "steps101_2000")
        self.assertIsNone(legacy["equal_seed_mean"]["mean_post_mean_dot_data_step"])
        self.assertEqual(legacy["equal_seed_mean"]["sum_raw_gradient_dot_data_step"], -10.0)
        self.assertEqual(legacy["equal_seed_mean"]["mean_raw_gradient_dot_data_step"], -1.0)
        spectral = next(row for row in paths if row["target"] == "clean"
                        and row["policy"] == analysis.SPECTRAL
                        and row["window"] == "steps101_2000")
        self.assertTrue(spectral["available"])
        self.assertTrue(components)

    def test_direct_i15_spectral_branch_and_checkpoint_binding(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            outer = Path(temporary)
            i15 = outer / "iteration-015"
            evidence = i15 / "analysis-001"
            collection_dir = i15 / "raw-results-001"
            root = outer / "root"
            confirmation = root / "confirmation"
            evidence.mkdir(parents=True)
            collection_dir.mkdir()
            confirmation.mkdir(parents=True)
            artifacts, spectral_rows, trajectories, raw_rows, old = [], [], [], [], {}
            for seed in analysis.SEEDS:
                for target in analysis.TARGETS:
                    curve = [{"horizon": horizon, **_evaluation(horizon / 100000)}
                             for horizon in analysis.HORIZONS]
                    old[seed, target, "raw"] = {"seed": seed, "target": target,
                        "policy": "raw", "status": "complete", "curve": curve,
                        "steps": [], "failure": None}
                    raw_rows.append({"seed": seed, "target": target,
                        "curve": {"curve": [{"horizon": 0, **_evaluation()}] + curve}})
                    checkpoints, projected = [], []
                    for horizon in analysis.HORIZONS[1:]:
                        kind = "full_state" if horizon == 2000 else "model_state"
                        path = confirmation / f"{kind}-s{seed}-{target}-h{horizon}.pt"
                        path.write_bytes(f"{seed}-{target}-{horizon}".encode())
                        record = _record(path)
                        artifacts.append(record)
                        checkpoints.append({"horizon": horizon, "relative_horizon": horizon - 100,
                            kind: record, **({"full_state_digest": "f" * 64}
                                              if kind == "full_state" else {})})
                        projected.append({"horizon": horizon, kind: record})
                    parent_eval = analysis.tree_digest(curve[0])
                    branch = {"schema": "i15_history_branch_v1", "seed": seed,
                        "target": target, "policy": analysis.SPECTRAL, "base": "sgdm",
                        "lr": .03, "status": "complete", "failure": None,
                        "completed_updates": 1900, "parent_state_digest": "a" * 64,
                        "parent_evaluation_digest": parent_eval, "curve": curve,
                        "checkpoints": checkpoints}
                    branch_path = confirmation / f"branch-s{seed}-{target}.json"
                    branch_path.write_text(json.dumps(branch))
                    branch_record = _record(branch_path)
                    artifacts.append(branch_record)
                    spectral_rows.append({"seed": seed, "target": target,
                        "curve_artifact": branch_record, "checkpoint_records": projected,
                        "parent_state_digest": "a" * 64,
                        "parent_evaluation_digest": parent_eval})
                    trajectories.append({"seed": seed, "target": target,
                        "policy": analysis.SPECTRAL, "status": "complete", "failure": None,
                        "curve": [{"horizon": point["horizon"],
                                   "values": _summary_values(point)} for point in curve]})
            trajectories.extend({"policy": "filler", "ordinal": index} for index in range(24))
            completion = {"schema": "i15_completion_v1", "status": "complete",
                "phase": "confirmation",
                "frozen_commit": "2bccbc6a883f1c4c11950965f9801d2c03d4350b",
                "branches": 18, "numerical_failures": 0, "artifacts": artifacts}
            completion_path = confirmation / "completion.json"
            completion_path.write_text(json.dumps(completion))
            completion_sha = analysis.sha256(completion_path)
            summary_path = evidence / "summary.json"
            summary_path.write_text(json.dumps({"schema": "i15_history_analysis_summary_v1",
                "audit_status": "pass", "artifact_root": str(root),
                "counts": {"new_confirmation_branches": 18,
                           "confirmation_numerical_failures": 0},
                "all_policy_trajectories": trajectories}))
            audit_path = evidence / "audit.json"
            audit_path.write_text(json.dumps({"schema": "i15_history_analysis_audit_v1",
                "status": "pass", "artifact_root": str(root),
                "phase_completion_sha256": {"confirmation": completion_sha}}))
            audit_sha = analysis.sha256(audit_path)
            collection_path = collection_dir / "collection.json"
            collection_path.write_text(json.dumps({"schema": "i15_scalar_collection_v1",
                "status": "complete", "source_root": str(root),
                "analysis_audit": {"sha256": audit_sha},
                "phase_completion_sha256": {"confirmation": completion_sha}}))
            binding = {"schema": "i16_parent_and_reference_binding_v1",
                "i14": {"parents": [{"seed": seed, "target": target,
                    "state_digest": "a" * 64} for seed in analysis.SEEDS
                    for target in analysis.TARGETS]},
                "i15": {"artifact_root": str(root),
                    "acquisition_commit": "2bccbc6a883f1c4c11950965f9801d2c03d4350b",
                    "audit_sha256": audit_sha,
                    "summary_sha256": analysis.sha256(summary_path),
                    "collection_sha256": analysis.sha256(collection_path),
                    "confirmation_completion_sha256": completion_sha},
                "raw_references": raw_rows, "spectral_references": spectral_rows,
                "source_replayed": False}
            patches = (mock.patch.object(analysis, "I15", i15),
                mock.patch.object(analysis, "I15_ROOT", root),
                mock.patch.object(analysis, "I15_AUDIT_SHA", audit_sha),
                mock.patch.object(analysis, "I15_SUMMARY_SHA", analysis.sha256(summary_path)),
                mock.patch.object(analysis, "I15_COLLECTION_SHA", analysis.sha256(collection_path)),
                mock.patch.object(analysis, "I15_COMPLETION_SHA", completion_sha),
                mock.patch.object(analysis.base, "load_i14_references", return_value=old),
                mock.patch.object(analysis.base, "validate_branch",
                    side_effect=lambda branch, phase, audit, label: {
                        "curve": branch["curve"], "steps": branch.get("steps", [])}),
                mock.patch.object(analysis, "validate_legacy_raw_steps", return_value=None))
            with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], \
                    patches[6], patches[7], patches[8]:
                audit = analysis.Audit()
                loaded = analysis.load_references(binding, audit)
                self.assertEqual(audit.errors, [])
                self.assertEqual(len(loaded), 12)
                bad = copy.deepcopy(binding)
                bad["spectral_references"][0]["checkpoint_records"][0]["horizon"] = 500
                audit = analysis.Audit()
                analysis.load_references(bad, audit)
                self.assertTrue(any("checkpoint" in error for error in audit.errors))

    def test_cross_phase_forecast_and_completion_accounting(self) -> None:
        smoke_seconds = 1.0
        expected = smoke_seconds / 60 * 34200 * 1.5 + 60
        phases = {"smoke": ({"smoke_based_seconds_forecast": None},
                             {"scalar_update_seconds": smoke_seconds}, {}),
                  "confirmation": ({"smoke_based_seconds_forecast": expected}, {}, {})}
        audit = analysis.Audit()
        analysis.validate_cross_phase_timing(phases, audit)
        entries = [{"completed_updates": 10, "status": "complete",
                    "scalar_update_seconds": .25},
                   {"completed_updates": 9, "status": "numerical_failure",
                    "scalar_update_seconds": .5}]
        completion = {"completed_training_updates": 219, "numerical_failures": 1,
                      "scalar_update_seconds": .75}
        analysis.validate_phase_aggregate(completion, entries, 200, audit, "synthetic")
        self.assertEqual(audit.errors, [])
        phases["confirmation"][0]["smoke_based_seconds_forecast"] += 1
        analysis.validate_cross_phase_timing(phases, audit)
        self.assertTrue(any("forecast" in error for error in audit.errors))

    def test_phase_resource_limits(self) -> None:
        completion = {"scalar_update_seconds": 1.0, "elapsed_seconds": 99.0,
                      "peak_torch_gpu_bytes": 4 * 1024**3,
                      "shared_artifact_bytes_before_completion": analysis.ARTIFACT_CAP}
        audit = analysis.Audit()
        analysis.validate_phase_resources(completion, "smoke", audit)
        self.assertEqual(audit.errors, [])
        completion["elapsed_seconds"] = 100.01
        completion["peak_torch_gpu_bytes"] += 1
        analysis.validate_phase_resources(completion, "smoke", audit)
        self.assertTrue(any("wall limit" in error for error in audit.errors))
        self.assertTrue(any("resource counters" in error for error in audit.errors))

    def test_checkpoint_complete_state_is_cpu_tree_digested(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            state = {"vector": torch.arange(3), "scalar": torch.tensor(1.)}
            path = directory / "state.pt"
            torch.save(state, path)
            record = {"name": path.name, "bytes": path.stat().st_size,
                      "sha256": analysis.sha256(path)}
            branch = _branch(216, "clean", "k0")
            branch["checkpoints"][0]["full_state"] = record
            branch["checkpoints"][0]["full_state_digest"] = analysis.tree_digest(state)
            audit = analysis.Audit()
            analysis.validate_checkpoints(branch, {path.name: record}, directory,
                                          audit, "synthetic")
            self.assertEqual(audit.errors, [])
            self.assertEqual(audit.tree_digests, 1)

    def test_runtime_inventory_is_exact_and_rejects_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "runtime" / "nested" / "empty").mkdir(parents=True)
            (root / "runtime" / "nested" / "file").write_bytes(b"abc")
            for phase in analysis.PHASES:
                (root / phase).mkdir()
                (root / f"attempt-{phase}.json").write_text("{}")
            audit = analysis.Audit()
            result = analysis.runtime_inventory(root, audit)
            self.assertEqual(audit.errors, [])
            self.assertIn({"path": "nested/empty", "kind": "directory", "bytes": 0,
                           "sha256": None}, result["entries"])
            os.symlink(root / "runtime" / "nested", root / "runtime" / "link")
            audit = analysis.Audit()
            analysis.runtime_inventory(root, audit)
            self.assertTrue(any("not a real directory" in error for error in audit.errors))


if __name__ == "__main__":
    unittest.main()
