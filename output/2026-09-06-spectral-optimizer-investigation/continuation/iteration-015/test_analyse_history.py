"""Synthetic CPU tests for the prospective I15 integrity analyzer."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch

import analyse_history as analysis


def _threads() -> None:
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        if torch.get_num_interop_threads() != 1:
            raise


def _leakage(norm: float, fraction: float = .25) -> dict:
    squared = norm * norm
    return {"squared_norm": squared, "outside_squared_norm": squared * fraction,
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


def _step(step: int, policy: str) -> dict:
    component = _component()
    selected = copy.deepcopy(component)
    native = copy.deepcopy(component)
    before = copy.deepcopy(component)
    return {
        "step": step, "relative_step": step - 100,
        "schema": "i15_history_step_v1", "base": "sgdm", "lr": .03,
        "policy": policy, "loss": 1.0,
        "observer": {"step_before": step - 1, "step_after": step,
                     "filtering_active": True, "used": True, "basis_rank": 2},
        "gradient_filter_applied": True,
        "gradient": {"raw_norm": 2.0, "applied_norm": 2.0,
                     "native_norm": 2.0, "mean_complement_norm": 2.0},
        "components": {name: copy.deepcopy(component) for name in (
            "raw_gradient", "native_projection", "post_mean",
            "native_action_post_mean", "post_mean_complement", "mean_delivery",
            "applied_delivery")},
        "displacement": {"total_norm": 2.0, "data_norm": 2.0,
            "nominal_decay_norm": .1, "total_current_basis_leakage": _leakage(2.0),
            "data_current_basis_leakage": _leakage(2.0),
            "raw_gradient_dot_data_delta": -1.0,
            "mean_complement_dot_data_delta": -.5,
            "projected_old_history_dot_data_delta": -.25,
            "removed_old_history_dot_data_delta": .1},
        "decay": {"coefficient": .01, "factor": 1 - .03 * .01,
                  "manual_before_optimizer_step": True, "manual_actual_norm": .1,
                  "manual_minus_nominal_norm": 0.0},
        "momentum_history": {"history_projected": policy.endswith("projected_history"),
            "buffer_before": before, "native_action_old_buffer": native,
            "removed_old_buffer": copy.deepcopy(component),
            "selected_old_buffer": (copy.deepcopy(native) if policy.endswith("projected_history")
                                    else copy.deepcopy(before)),
            "buffer_after": copy.deepcopy(component), "expected_buffer_after_norm": 2.0,
            "native_action_idempotence_defect": {"norm": 0.0,
                "homogeneous_roundoff_reference": 1e-5, "enforced": False},
            "current_basis_orthogonality_error": 1e-6,
            "native_action_old_dot_removed": 0.0,
            "removed_old_dot_mean_complement": .2,
            "empirical_ideal_data_step_defect_norm": .01},
        "algebra_residuals": {name: {"norm": 0.0, "homogeneous_error_bound": 1e-5}
                              for name in ("native_minus_actual_action_raw",
                                  "post_mean_recurrence", "mean_delivery_definition",
                                  "mean_alternate_definition", "momentum_buffer_recurrence")},
        "digests": ({name: hashlib.sha256(name.encode()).hexdigest() for name in (
            "raw_gradient", "post_observer", "applied_gradient", "old_momentum_buffer",
            "native_action_old_buffer", "new_momentum_buffer")} if step == 101 else None),
    }


def _branch(seed: int, target: str, policy: str, *, complete: bool = True) -> dict:
    steps = [_step(step, policy) for step in range(101, 111)]
    curve = [{"horizon": 100, **_evaluation()}, {"horizon": 110, **_evaluation(-.1)}]
    failure = None
    checkpoints = [{"horizon": 110, "relative_horizon": 10,
                    "full_state": {"name": "state.pt", "bytes": 0, "sha256": "0" * 64},
                    "full_state_digest": "1" * 64}]
    if not complete:
        steps = steps[:-1]
        curve = curve[:1]
        checkpoints = []
        failure = {"type": "NumericalFailure", "message": "synthetic finite failure",
                   "attempted_step": 110, "completed_step": 109,
                   "last_valid_horizon": 100,
                   "state_artifact": {"name": "failed.pt", "bytes": 0,
                                      "sha256": "2" * 64}}
    return {"schema": "i15_history_branch_v1",
            "id": f"s{seed}-sgdm-{target}-{policy}", "seed": seed, "target": target,
            "policy": policy, "base": "sgdm", "lr": .03, "parent_horizon": 100,
            "parent_state_digest": "3" * 64, "parent_evaluation_digest": "4" * 64,
            "status": "complete" if complete else "numerical_failure",
            "requested_updates": 10, "completed_updates": len(steps),
            "last_completed_horizon": 100 + len(steps), "curve": curve,
            "history_update_seconds": .5, "steps": steps, "checkpoints": checkpoints,
            "failure": failure, "first_step_digests": steps[0]["digests"] if steps else None,
            "selected_horizons": analysis._selection(curve) if complete else None}


class HistoryAnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _threads()

    def test_step_and_branch_schema_pass(self) -> None:
        audit = analysis.Audit()
        branch = _branch(215, "clean", "mean_projected_history")
        value = analysis.validate_branch(branch, "smoke", audit, "synthetic")
        self.assertIsNotNone(value)
        self.assertEqual(audit.errors, [])

    def test_step_mutations_and_nonideal_fraction(self) -> None:
        row = _step(101, "mean_native")
        row["components"]["raw_gradient"]["current_basis_leakage"] = _leakage(2.0, 1.2)
        audit = analysis.Audit()
        analysis.validate_step(row, 101, "mean_native", audit, "nonideal")
        self.assertEqual(audit.errors, [])
        bad = copy.deepcopy(row)
        bad["algebra_residuals"]["post_mean_recurrence"]["norm"] = 1.0
        audit = analysis.Audit()
        analysis.validate_step(bad, 101, "mean_native", audit, "bad")
        self.assertTrue(any("exceeds saved bound" in error for error in audit.errors))

    def test_evaluation_stage_terminal_failure_is_valid(self) -> None:
        branch = _branch(215, "fixed", "current_projected_history")
        branch["status"] = "numerical_failure"
        branch["failure"] = {"type": "NumericalFailure", "message": "evaluation nonfinite",
                             "attempted_step": 110, "completed_step": 110,
                             "last_valid_horizon": 100,
                             "state_artifact": {"name": "failed.pt", "bytes": 0,
                                                "sha256": "2" * 64}}
        branch["curve"] = branch["curve"][:1]
        branch["checkpoints"] = []
        branch["selected_horizons"] = None
        audit = analysis.Audit()
        analysis.validate_branch(branch, "smoke", audit, "evaluation failure")
        self.assertEqual(audit.errors, [])

    def test_primary_missing_only_required_contribution(self) -> None:
        points = {name: {"auxiliary": {"clean_ce": value,
                                        "clean_accuracy": 1 - value / 10}}
                  for name, value in zip(analysis.POLICIES, (.8, .7, .6, .5, .4))}
        points["mean_native"] = None
        self.assertIsNotNone(analysis._family_value({}, 200, "clean", "ce",
            "H_history_under_current", points))
        self.assertIsNotNone(analysis._family_value({}, 200, "clean", "ce",
            "M_mean_under_projected_history", points))
        self.assertIsNone(analysis._family_value({}, 200, "clean", "ce",
            "S_mean_by_history_interaction", points))

    def test_exact_primary_signs_means_and_validation_selectors(self) -> None:
        ce = {"current_native": .5, "current_projected_history": .8,
              "mean_native": .6, "mean_projected_history": .4, "raw": .9}
        accuracy = {"current_native": .7, "current_projected_history": .5,
                    "mean_native": .75, "mean_projected_history": .8, "raw": .4}
        branches = {}
        for seed in analysis.SEEDS:
            for target in analysis.TARGETS:
                for policy in analysis.POLICIES:
                    point = {"horizon": 2000, **_evaluation()}
                    point["auxiliary"]["clean_ce"] = ce[policy]
                    point["auxiliary"]["clean_accuracy"] = accuracy[policy]
                    branches[seed, target, policy] = {
                        "status": "complete", "curve": [point]}
        rows = analysis.primary_effects(branches)
        expected = {("H_history_under_current", "ce"): .3,
                    ("M_mean_under_projected_history", "ce"): .4,
                    ("S_mean_by_history_interaction", "ce"): .5,
                    ("H_history_under_current", "accuracy"): .2,
                    ("M_mean_under_projected_history", "accuracy"): .3,
                    ("S_mean_by_history_interaction", "accuracy"): .25}
        for row in rows:
            value = expected[row["family"], row["metric"]]
            self.assertTrue(row["effect"]["available"])
            self.assertTrue(all(abs(item - value) < 1e-12
                                for item in row["effect"]["all_three_seed_values"]))
            self.assertAlmostEqual(row["effect"]["mean"], value)
        broken = copy.deepcopy(branches)
        broken[200, "clean", "mean_native"]["status"] = "numerical_failure"
        broken_rows = analysis.primary_effects(broken)
        broken_s = next(row for row in broken_rows if row["family"].startswith("S_")
                        and row["target"] == "clean" and row["metric"] == "ce")
        self.assertFalse(broken_s["effect"]["available"])
        self.assertIsNone(broken_s["effect"]["mean"])

        curve = []
        for horizon, validation_ce, validation_accuracy, auxiliary_ce in (
                (100, 2.0, .2, .01), (250, 1.0, .3, 9.0),
                (500, 1.0, .8, .02), (1000, 1.2, .8, .03),
                (1500, 1.3, .7, .04), (2000, 1.4, .6, .05)):
            point = {"horizon": horizon, **_evaluation()}
            point["validation"]["clean_ce"] = validation_ce
            point["validation"]["clean_accuracy"] = validation_accuracy
            point["auxiliary"]["clean_ce"] = auxiliary_ce
            curve.append(point)
        self.assertEqual(analysis._selection(curve), {
            "minimum_validation_ce": 250, "maximum_validation_accuracy": 500})

    def test_checkpoint_full_state_tree_digest_and_model_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            state = {"vector": torch.arange(3), "scalar": torch.tensor(1.)}
            state_path = directory / "state.pt"
            torch.save(state, state_path)
            record = {"name": state_path.name, "bytes": state_path.stat().st_size,
                      "sha256": analysis.sha256(state_path)}
            branch = _branch(215, "clean", "mean_native")
            branch["checkpoints"][0]["full_state"] = record
            branch["checkpoints"][0]["full_state_digest"] = analysis.tree_digest(state)
            audit = analysis.Audit()
            analysis._validate_new_checkpoints(branch, {record["name"]: record}, directory,
                                                 audit, "synthetic")
            self.assertEqual(audit.errors, [])
            self.assertEqual(audit.tree_digests, 1)

    def test_runtime_inventory_lists_nested_empty_directory_and_rejects_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runtime = root / "runtime"
            (runtime / "outer" / "empty").mkdir(parents=True)
            (runtime / "outer" / "payload.bin").write_bytes(b"abc")
            for phase in analysis.PHASES:
                (root / phase).mkdir()
                (root / f"attempt-{phase}.json").write_text("{}")
            audit = analysis.Audit()
            result = analysis.runtime_inventory(root, audit)
            self.assertEqual(audit.errors, [])
            self.assertIn({"path": "outer/empty", "kind": "directory", "bytes": 0,
                           "sha256": None}, result["entries"])
            os.symlink(runtime / "outer", runtime / "link")
            audit = analysis.Audit()
            analysis.runtime_inventory(root, audit)
            self.assertTrue(any("not a real directory" in error for error in audit.errors))

    def test_energy_aggregation_is_within_branch_then_equal_seed(self) -> None:
        branches = {}
        for seed, fraction in zip(analysis.SEEDS, (.1, .2, .9)):
            for target in analysis.TARGETS:
                for policy in analysis.NEW_POLICIES:
                    branch = {"status": "complete", "steps": [_step(101, policy)]}
                    branch["steps"][0]["components"]["raw_gradient"][
                        "current_basis_leakage"] = _leakage(2.0, fraction)
                    zero = _step(102, policy)
                    zero["components"]["raw_gradient"] = {
                        "norm": 0.0, "squared_energy": 0.0,
                        "current_basis_leakage": {"squared_norm": 0.0,
                            "outside_squared_norm": 0.0, "fraction": None,
                            "reason": "zero_displacement"}}
                    branch["steps"].append(zero)
                    branches[seed, target, policy] = branch
        rows = analysis.component_energy_results(branches)
        row = next(value for value in rows if value["target"] == "clean"
                   and value["policy"] == "mean_native"
                   and value["window"] == "steps101_2000"
                   and value["component"] == "raw_gradient")
        self.assertAlmostEqual(row["equal_seed_mean_of_branch_outside_energy_fractions"], .4)


if __name__ == "__main__":
    unittest.main()
