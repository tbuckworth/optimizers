"""Independent synthetic CPU tests for the I16 scalar-control runner."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import torch

import run_scalar_controls as runner


def _threads():
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        if torch.get_num_interop_threads() != 1:
            raise


def _data():
    generator = torch.Generator().manual_seed(16101)
    x = torch.rand((29, 4), generator=generator)
    labels = torch.randint(2, (29,), generator=generator)
    return {"x": x[:16].clone(), "clean": labels[:16].clone(),
            "noisy": (1 - labels[:16]).clone(),
            "vx": x[16:22].clone(), "vy": labels[16:22].clone(),
            "ax": x[22:].clone(), "ay": labels[22:].clone()}


def _plan(seed=216):
    batches = np.empty((110, 64), dtype=np.int64)
    for row in range(110):
        batches[row].fill(row % 16)
    return {"seed": seed, "training_batches": batches}


def _parent(data, plan, target="clean"):
    runner.old.seed_all(16102)
    model = runner.i9.make_model(16102, "cpu", input_dim=4, width=3, classes=2)
    optimizer = runner.c14.make_optimizer(model, "sgdm", .03)
    tracker = runner.i9.make_tracker(model, optimizer)
    labels = data["clean"] if target == "clean" else data["noisy"]
    batches = torch.as_tensor(plan["training_batches"], dtype=torch.long)
    for step in range(100):
        indices = batches[step]
        runner.c14.train_step(model, optimizer, tracker, data["x"][indices],
                              labels[indices], "raw", "sgdm")
    state = runner.c14.snapshot(model, optimizer, tracker, "sgdm", .03)
    expected = {"horizon": 100, **runner.old.evaluate(model, data)}
    return state, expected


def _summary_values(evaluation):
    return {
        "train_clean_ce": evaluation["train"]["clean_ce"],
        "train_soft_ce": evaluation["train"]["soft_ce"],
        "train_clean_accuracy": evaluation["train"]["clean_accuracy"],
        "train_fixed_ce": evaluation["train"]["fixed_ce"],
        "train_fixed_minus_soft_ce": evaluation["train"]["fixed_minus_soft_ce"],
        "train_fixed_accuracy": evaluation["train"]["fixed_accuracy"],
        "train_confidence": evaluation["train"]["mean_max_probability"],
        "train_true_label_probability":
            evaluation["train"]["mean_true_label_probability"],
        "validation_clean_ce": evaluation["validation"]["clean_ce"],
        "validation_clean_accuracy": evaluation["validation"]["clean_accuracy"],
        "validation_confidence": evaluation["validation"]["mean_max_probability"],
        "validation_true_label_probability":
            evaluation["validation"]["mean_true_label_probability"],
        "auxiliary_clean_ce": evaluation["auxiliary"]["clean_ce"],
        "auxiliary_clean_accuracy": evaluation["auxiliary"]["clean_accuracy"],
        "auxiliary_confidence": evaluation["auxiliary"]["mean_max_probability"],
        "auxiliary_true_label_probability":
            evaluation["auxiliary"]["mean_true_label_probability"]}


def _spectral_reference(seed, target, expected):
    curve = [{"horizon": horizon,
              "values": copy.deepcopy(_summary_values(expected))}
             for horizon in runner.HORIZONS]
    return {"seed": seed, "target": target, "policy": "mean_projected_history",
            "status": "complete", "failure": None, "curve": curve}


class _NoWriteRun:
    def __init__(self):
        self.saved = 0
        self.started = 0.0

    def check(self):
        return None

    def save(self, *_args, **_kwargs):
        self.saved += 1
        raise AssertionError("invalid input reached artifact creation")


class _CaptureRun:
    def __init__(self):
        self.saved = []

    def check(self):
        return None

    def save(self, name, value, tensor=False):
        self.saved.append((name, value, tensor))
        return {"name": name, "bytes": 1, "sha256": "a" * 64}


class ScalarRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _threads()
        cls.data = _data()
        cls.plan = _plan()
        cls.clean_parent, cls.clean_start = _parent(cls.data, cls.plan, "clean")
        cls.fixed_parent, cls.fixed_start = _parent(cls.data, cls.plan, "fixed")

    def run_one(self, root, phase, k, target="clean"):
        state, expected = ((self.clean_parent, self.clean_start)
                           if target == "clean"
                           else (self.fixed_parent, self.fixed_start))
        run = runner.old.Run(root, phase, 60)
        entry = runner.run_branch(run, state, self.data, self.plan, target, k,
                                  expected, smoke=True, end=110,
                                  horizons=(100, 110))
        result = json.loads((run.path / entry["artifact"]["name"]).read_text())
        return run, entry, result

    def test_invalid_k_dimensions_parent_data_and_batches_reject_before_write(self):
        cases = []
        cases.append((self.clean_parent, self.data, self.plan, "clean", 1.0, 110,
                      (100, 110)))
        cases.append((self.clean_parent, self.data, self.plan, "clean", 0, 110,
                      (100, 110)))
        cases.append((self.clean_parent, self.data, self.plan, "other", .5, 110,
                      (100, 110)))
        bad_parent = copy.deepcopy(self.clean_parent)
        bad_parent["tracker"]["step_count"] = 99
        cases.append((bad_parent, self.data, self.plan, "clean", .5, 110,
                      (100, 110)))
        cases.append((self.clean_parent, self.data, self.plan, "clean", .5, 111,
                      (100, 111)))
        cases.append((self.clean_parent, self.data, self.plan, "clean", .5, 110,
                      (100, 109, 110)))
        bad_data = dict(self.data)
        bad_data["vx"] = bad_data["vx"][:, :3]
        cases.append((self.clean_parent, bad_data, self.plan, "clean", .5, 110,
                      (100, 110)))
        bad_shape = copy.deepcopy(self.plan)
        bad_shape["training_batches"] = bad_shape["training_batches"][:-1]
        cases.append((self.clean_parent, self.data, bad_shape, "clean", .5, 110,
                      (100, 110)))
        bad_dtype = copy.deepcopy(self.plan)
        bad_dtype["training_batches"] = bad_dtype["training_batches"].astype(np.int32)
        cases.append((self.clean_parent, self.data, bad_dtype, "clean", .5, 110,
                      (100, 110)))
        bad_index = copy.deepcopy(self.plan)
        bad_index["training_batches"][100, 0] = len(self.data["x"])
        cases.append((self.clean_parent, self.data, bad_index, "clean", .5, 110,
                      (100, 110)))

        for state, data, plan, target, k, end, horizons in cases:
            fake = _NoWriteRun()
            with self.subTest(target=target, k=k, end=end):
                with self.assertRaises((ValueError, runner.core.ScalarCoreError)):
                    runner.run_branch(fake, state, data, plan, target, k,
                                      self.clean_start, smoke=True, end=end,
                                      horizons=horizons)
                self.assertEqual(fake.saved, 0)

    def test_true_h100_to_h110_fork_rows_and_path_statistics(self):
        parent_before = runner.i9.tree_digest(self.clean_parent)
        expected_batches = [self.data["x"][self.plan["training_batches"][row]]
                            for row in range(100, 110)]
        used = []
        original = runner.core.scalar_step

        def recording_step(model, optimizer, tracker, x, target, k,
                           *, capture_digests=False):
            used.append(x.detach().clone())
            return original(model, optimizer, tracker, x, target, k,
                            capture_digests=capture_digests)

        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.object(runner.core, "scalar_step", side_effect=recording_step):
            _, entry, result = self.run_one(Path(directory), "fork", .5)
        self.assertEqual(entry["completed_updates"], 10)
        self.assertEqual(result["status"], "complete")
        self.assertEqual([row["step"] for row in result["steps"]], list(range(101, 111)))
        self.assertEqual([row["relative_step"] for row in result["steps"]],
                         list(range(1, 11)))
        self.assertEqual([row["horizon"] for row in result["curve"]], [100, 110])
        self.assertEqual(len(used), 10)
        self.assertTrue(all(torch.equal(actual, expected)
                            for actual, expected in zip(used, expected_batches)))
        for row in result["steps"]:
            path = row["path_statistics"]
            self.assertEqual(path["data_step_squared_energy"],
                             path["data_path_length_increment"] ** 2)
            self.assertEqual(path["raw_gradient_dot_data_step"],
                             row["displacement"]["raw_gradient_dot_data_delta"])
        checkpoint = result["checkpoints"][0]
        self.assertEqual((checkpoint["horizon"], checkpoint["relative_horizon"]),
                         (110, 10))
        self.assertIn("full_state", checkpoint)
        self.assertNotIn("model_state", checkpoint)
        self.assertEqual(runner.i9.tree_digest(self.clean_parent), parent_before)

    def test_parent_evaluation_is_state_and_rng_neutral(self):
        model, optimizer, tracker = runner.c14.restore(self.clean_parent, "cpu")
        before = runner.i9.tree_digest(
            runner.c14.snapshot(model, optimizer, tracker, "sgdm", .03))
        rng = runner.i9._rng_state()
        measured = {"horizon": 100, **runner.old.evaluate(model, self.data)}
        after = runner.i9.tree_digest(
            runner.c14.snapshot(model, optimizer, tracker, "sgdm", .03))
        self.assertEqual(measured, self.clean_start)
        self.assertEqual(before, after)
        self.assertTrue(runner.i9.equal_tree(rng, runner.i9._rng_state()))

    def test_first_step_pairing_checks_only_policy_independent_digests(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entries = [self.run_one(root, "pair-" + runner.K_NAMES[k], k)[1]
                       for k in runner.core.REAL_K]
        checks = runner.validate_first_step_pairs(entries)
        self.assertEqual(checks, [{"seed": 216, "target": "clean",
                                   "available_first_steps": 3, "status": "pass"}])
        for key in ("raw_gradient", "post_observer", "old_momentum_buffer"):
            self.assertEqual(len({row["first_step_digests"][key]
                                  for row in entries}), 1)
        deliberately_different = copy.deepcopy(entries)
        for index, row in enumerate(deliberately_different):
            row["first_step_digests"]["applied_gradient"] = f"different-{index}"
        self.assertEqual(runner.validate_first_step_pairs(deliberately_different)[0][
            "status"], "pass")
        altered = copy.deepcopy(entries)
        altered[0]["first_step_digests"]["raw_gradient"] = "wrong"
        with self.assertRaisesRegex(ValueError, "common tensor"):
            runner.validate_first_step_pairs(altered)
        partial = copy.deepcopy(entries)
        partial[0]["first_step_digests"] = None
        partial[0]["status"] = "numerical_failure"
        self.assertEqual(runner.validate_first_step_pairs(partial)[0]["status"],
                         "partial_numerical_evidence")

    def test_k1_test_only_step_is_exact_i14_raw_state_parity(self):
        indices = torch.as_tensor(self.plan["training_batches"][100], dtype=torch.long)
        left = runner.c14.restore(self.clean_parent, "cpu")
        right = runner.c14.restore(self.clean_parent, "cpu")
        runner.c14.train_step(*left, self.data["x"][indices], self.data["clean"][indices],
                              "raw", "sgdm")
        runner.core.scalar_step(*right, self.data["x"][indices],
                                self.data["clean"][indices], 1.0)
        left_state = runner.c14.snapshot(*left, "sgdm", .03)
        right_state = runner.c14.snapshot(*right, "sgdm", .03)
        self.assertTrue(runner.i9.equal_tree(left_state, right_state))

    def test_nonfinite_retains_partial_branch_but_structural_error_aborts(self):
        original = runner.core.scalar_step
        calls = 0

        def fail_second(model, optimizer, tracker, x, target, k,
                        *, capture_digests=False):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise runner.core.NumericalFailure("synthetic nonfinite")
            return original(model, optimizer, tracker, x, target, k,
                            capture_digests=capture_digests)

        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.object(runner.core, "scalar_step", side_effect=fail_second):
            run, entry, result = self.run_one(Path(directory), "numeric", .5)
            self.assertEqual((entry["status"], entry["completed_updates"]),
                             ("numerical_failure", 1))
            self.assertEqual(result["failure"]["attempted_step"], 102)
            self.assertEqual([row["step"] for row in result["steps"]], [101])
            self.assertTrue((run.path /
                result["failure"]["state_artifact"]["name"]).is_file())

        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.object(runner.core, "scalar_step",
                                  side_effect=runner.core.ScalarCoreError(
                                      "synthetic structural failure")):
            root = Path(directory)
            with self.assertRaisesRegex(runner.core.ScalarCoreError, "structural"):
                self.run_one(root, "structural", .5)
            self.assertFalse(any((root / "structural").glob("branch-*.json")))
            self.assertFalse(any((root / "structural").glob("failed-state-*.pt")))

    def test_evaluation_nonfinite_seals_branch_but_generic_error_aborts(self):
        original = runner.old.evaluate
        calls = 0

        def numerical_on_second(model, data):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise runner.old.core.NumericalFailure(
                    "synthetic evaluation nonfinite")
            return original(model, data)

        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.object(runner.old, "evaluate",
                                  side_effect=numerical_on_second):
            run, entry, result = self.run_one(Path(directory), "eval-numeric", .5)
            self.assertEqual((entry["status"], entry["completed_updates"]),
                             ("numerical_failure", 10))
            self.assertEqual(result["failure"]["attempted_step"], 110)
            self.assertEqual(result["failure"]["last_valid_horizon"], 100)
            self.assertEqual([row["step"] for row in result["steps"]],
                             list(range(101, 111)))
            self.assertTrue((run.path /
                result["failure"]["state_artifact"]["name"]).is_file())

        calls = 0

        def generic_on_second(model, data):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("synthetic generic evaluation failure")
            return original(model, data)

        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.object(runner.old, "evaluate", side_effect=generic_on_second):
            root = Path(directory)
            with self.assertRaisesRegex(RuntimeError, "generic evaluation"):
                self.run_one(root, "eval-generic", .5)
            self.assertFalse(any((root / "eval-generic").glob("branch-*.json")))
            self.assertFalse(any((root / "eval-generic").glob("failed-state-*.pt")))

    def test_bound_inputs_pins_i15_without_mutating_previous_globals(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            i15_dir = root / "iteration-015"
            analysis = i15_dir / "analysis-001"
            collected = i15_dir / "raw-results-001"
            source = root / "source"
            confirmation = source / "confirmation"
            analysis.mkdir(parents=True)
            collected.mkdir()
            confirmation.mkdir(parents=True)
            curves = {}
            trajectories = []
            parents = {}
            saved_plans = {}
            declared = []
            branch_entries = []

            def save_artifact(name, value=None):
                path = confirmation / name
                if value is None:
                    path.write_bytes(name.encode("ascii"))
                else:
                    path.write_text(json.dumps(value))
                record = {"name": name, "bytes": path.stat().st_size,
                          "sha256": runner.old.digest(path)}
                declared.append(record)
                return record

            for seed in runner.SEEDS:
                saved_plans[seed] = _plan(seed)
                for target in runner.TARGETS:
                    expected = self.clean_start if target == "clean" else self.fixed_start
                    parent = self.clean_parent if target == "clean" else self.fixed_parent
                    parents[seed, target] = {"artifact": {"name": "parent.pt",
                        "bytes": 1, "sha256": "a" * 64},
                        "state_digest": runner.i9.tree_digest(parent)}
                    curve = [{"horizon": 0}, copy.deepcopy(expected)]
                    for policy in ("raw", "current32"):
                        curves[seed, target, policy] = {
                            "status": "complete", "curve": copy.deepcopy(curve)}
                    trajectories.append(_spectral_reference(seed, target, expected))
                    for policy in runner.previous.core.REAL_POLICIES:
                        identity = f"s{seed}-sgdm-{target}-{policy}"
                        entry = {"id": identity, "seed": seed, "target": target,
                                 "policy": policy, "status": "complete",
                                 "completed_updates": 1900,
                                 "parent_state_digest":
                                     runner.i9.tree_digest(parent),
                                 "parent_evaluation_digest":
                                     runner.i9.tree_digest(expected)}
                        if policy == "mean_projected_history":
                            checkpoints = []
                            for horizon in runner.HORIZONS[1:]:
                                kind = "full_state" if horizon == 2000 else "model_state"
                                record = save_artifact(
                                    f"{kind}-{identity}-h{horizon}.pt")
                                checkpoints.append({"horizon": horizon,
                                                    "relative_horizon": horizon - 100,
                                                    kind: record})
                            direct_curve = [{**copy.deepcopy(expected), "horizon": horizon}
                                            for horizon in runner.HORIZONS]
                            branch = {"schema": "i15_history_branch_v1", **entry,
                                      "base": "sgdm", "lr": .03,
                                      "parent_horizon": 100,
                                      "requested_updates": 1900,
                                      "last_completed_horizon": 2000,
                                      "curve": direct_curve, "steps": [],
                                      "checkpoints": checkpoints, "failure": None}
                            entry["artifact"] = save_artifact(
                                f"branch-{identity}.json", branch)
                        branch_entries.append(entry)
            branches_record = save_artifact(
                "branches.json", {"entries": branch_entries,
                                  "first_step_pair_checks": []})
            audit = {"schema": "i15_history_analysis_audit_v1", "status": "pass",
                     "artifact_root": str(source),
                     "phase_completion_sha256": {"smoke": "b" * 64}}
            summary = {"schema": "i15_history_analysis_summary_v1", "audit_status": "pass",
                       "artifact_root": str(source),
                       "counts": {"new_confirmation_branches": 18,
                                  "confirmation_numerical_failures": 0},
                       "all_policy_trajectories": trajectories + [
                           {"policy": "other"} for _ in range(24)]}
            completion = {"schema": "i15_completion_v1", "status": "complete",
                          "phase": "confirmation",
                          "frozen_commit": runner.I15_ACQUISITION_COMMIT,
                          "branches": 18, "numerical_failures": 0,
                          "all_requested_endpoints_present": True,
                          "artifacts": declared}
            (confirmation / "completion.json").write_text(json.dumps(completion))
            confirmation_sha = runner.old.digest(confirmation / "completion.json")
            audit["phase_completion_sha256"]["confirmation"] = confirmation_sha
            (analysis / "audit.json").write_text(json.dumps(audit))
            (analysis / "summary.json").write_text(json.dumps(summary))
            audit_sha = runner.old.digest(analysis / "audit.json")
            collection = {"schema": "i15_scalar_collection_v1", "status": "complete",
                          "source_root": str(source),
                          "analysis_audit": {"sha256": audit_sha},
                          "phase_completion_sha256": {"confirmation":
                              confirmation_sha}}
            (collected / "collection.json").write_text(json.dumps(collection))
            previous_root = runner.previous.SOURCE_ROOT
            with mock.patch.object(runner, "I15", i15_dir), \
                    mock.patch.object(runner, "I15_ROOT", source), \
                    mock.patch.object(runner, "I15_AUDIT_SHA", audit_sha), \
                    mock.patch.object(runner, "I15_SUMMARY_SHA",
                                      runner.old.digest(analysis / "summary.json")), \
                    mock.patch.object(runner, "I15_COLLECTION_SHA",
                                      runner.old.digest(collected / "collection.json")), \
                    mock.patch.object(runner, "I15_CONFIRMATION_SHA",
                                      confirmation_sha), \
                    mock.patch.object(runner.previous, "bound_inputs", return_value=(
                        {"corruption_counts": {}}, parents, curves, saved_plans)):
                binding, actual_parents, actual_curves, plans, spectral = (
                    runner.bound_inputs())
                first_spectral = next(row for row in branch_entries
                                      if row["policy"] == "mean_projected_history")
                changed = confirmation / first_spectral["artifact"]["name"]
                changed.write_text(changed.read_text() + " ")
                with self.assertRaisesRegex(ValueError, "Artifact content differs"):
                    runner.bound_inputs()
            self.assertEqual(binding["schema"], "i16_parent_and_reference_binding_v1")
            self.assertEqual(len(binding["raw_references"]), 6)
            self.assertEqual(len(binding["spectral_references"]), 6)
            self.assertEqual((actual_parents, actual_curves, plans),
                             (parents, curves, saved_plans))
            self.assertEqual(set(spectral), {(seed, target) for seed in runner.SEEDS
                                             for target in runner.TARGETS})
            self.assertEqual(runner.previous.SOURCE_ROOT, previous_root)

    def test_confirmation_loads_and_checks_all_six_parents_before_first_branch(self):
        parents, references, spectral, states = {}, {}, {}, {}
        saved_plans = {seed: _plan(seed) for seed in runner.SEEDS}
        corruption = {"replaced_count": 2, "incorrect_count": 1, "train_count": 16}
        binding = {"i14": {"corruption_counts": {
            str(seed): corruption for seed in runner.SEEDS}}}
        for seed in runner.SEEDS:
            for target in runner.TARGETS:
                state, expected = ((self.clean_parent, self.clean_start)
                                   if target == "clean"
                                   else (self.fixed_parent, self.fixed_start))
                states[f"p-{seed}-{target}.pt"] = state
                parents[seed, target] = {
                    "artifact": {"name": f"p-{seed}-{target}.pt",
                                 "bytes": 1, "sha256": "a" * 64},
                    "state_digest": runner.i9.tree_digest(state)}
                for policy in ("raw", "current32"):
                    references[seed, target, policy] = {
                        "status": "complete", "curve": [copy.deepcopy(expected)]}
                spectral[seed, target] = _spectral_reference(seed, target, expected)
        bound = (binding, parents, references, saved_plans, spectral)
        loads = []
        branches = []

        def fake_checked(_directory, record):
            return Path(record["name"])

        def fake_load(path, **_kwargs):
            loads.append(path.name)
            return copy.deepcopy(states[path.name])

        def fake_branch(_run, state, _data_value, plan, target, k, _expected, **_kwargs):
            self.assertEqual(len(loads), 6)
            branches.append((plan["seed"], target, k))
            return {"id": "fake", "seed": plan["seed"], "target": target,
                    "k": k, "k_label": runner.K_NAMES[k], "status": "complete",
                    "completed_updates": 1900,
                    "parent_state_digest": runner.i9.tree_digest(state),
                    "parent_evaluation_digest": "d" * 64,
                    "first_step_digests": {"raw_gradient": "e" * 64,
                        "post_observer": "f" * 64, "old_momentum_buffer": "0" * 64,
                        "applied_gradient": str(k), "new_momentum_buffer": str(k)},
                    "scalar_update_seconds": 1.,
                    "artifact": {"name": "fake.json", "bytes": 1,
                                 "sha256": "a" * 64}}

        run = _CaptureRun()
        restore = runner.c14.restore
        with mock.patch.object(runner, "bound_inputs", return_value=bound), \
                mock.patch.object(runner.plans, "read_training", return_value=(None, None)), \
                mock.patch.object(runner.plans, "data_for_plan",
                                  return_value=(self.data, corruption)), \
                mock.patch.object(runner.previous, "checked_artifact",
                                  side_effect=fake_checked), \
                mock.patch.object(runner.torch, "load", side_effect=fake_load), \
                mock.patch.object(runner.c14, "restore",
                                  side_effect=lambda state, _device:
                                      restore(state, "cpu")), \
                mock.patch.object(runner, "run_branch", side_effect=fake_branch), \
                mock.patch.object(runner, "validate_first_step_pairs", return_value=[]):
            entries, warmup = runner.confirmation(run)
        self.assertEqual(warmup, 0)
        self.assertEqual(len(loads), 6)
        self.assertEqual(len(set(loads)), 6)
        self.assertEqual(len(branches), 18)
        self.assertEqual(len(entries), 18)
        seam = next(value for name, value, _tensor in run.saved
                    if name == "parent-seams.json")
        self.assertEqual(seam["parents_loaded"], 6)
        self.assertEqual(seam["scientific_updates_before_admission"], 0)

    def test_fixed_limits_and_inherited_shared_cap(self):
        self.assertEqual(runner.LIMITS, {"smoke": 100, "confirmation": 1800})
        self.assertEqual(runner.CONFIRMATION_UPDATES, 18 * 1900)
        self.assertEqual(runner.old.ARTIFACT_CAP, 3 * 1024**3)
        self.assertEqual(runner.core.REAL_K, (0.0, .5, .9))
        self.assertEqual(runner.core.TEST_ONLY_K, (1.0,))


if __name__ == "__main__":
    unittest.main()
