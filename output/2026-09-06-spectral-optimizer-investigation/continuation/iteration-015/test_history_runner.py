"""Independent synthetic CPU tests for the I15 history-branch runner."""
from __future__ import annotations

import copy
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import torch

import run_history_branches as runner


def _threads() -> None:
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        if torch.get_num_interop_threads() != 1:
            raise


def _data() -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(15101)
    x = torch.rand((29, 4), generator=generator)
    labels = torch.randint(2, (29,), generator=generator)
    return {
        "x": x[:16].clone(),
        "clean": labels[:16].clone(),
        "noisy": (1 - labels[:16]).clone(),
        "vx": x[16:22].clone(),
        "vy": labels[16:22].clone(),
        "ax": x[22:].clone(),
        "ay": labels[22:].clone(),
    }


def _plan() -> dict:
    # Every row selects one distinct, identifiable training example repeatedly.
    batches = np.empty((110, 64), dtype=np.int64)
    for row in range(110):
        batches[row].fill(row % 16)
    return {"seed": 215, "training_batches": batches}


def _parent(data: dict[str, torch.Tensor], plan: dict, target: str = "clean"):
    runner.old.seed_all(15102)
    model = runner.i9.make_model(15102, "cpu", input_dim=4, width=3, classes=2)
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


def _save_json(path: Path, value) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return {"name": path.name, "bytes": path.stat().st_size,
            "sha256": runner.old.digest(path)}


def _fake_i14_inputs(root: Path):
    """Create small structurally complete provenance inputs, never tensor-loaded."""
    i14 = root / "iteration-014"
    analysis = i14 / "analysis-001"
    source = root / "source"
    confirmation = source / "confirmation"
    data_dir = root / "mnist"
    analysis.mkdir(parents=True)
    confirmation.mkdir(parents=True)
    data_dir.mkdir()
    (analysis / "audit.json").write_text('{"status":"fail"}')
    (analysis / "summary.json").write_text('{"schema":"summary"}')
    for name, contents in (("train-images-idx3-ubyte", b"images"),
                           ("train-labels-idx1-ubyte", b"labels")):
        (data_dir / name).write_bytes(contents)

    declared: list[dict] = []
    entries = []
    for seed in runner.SEEDS:
        for target in runner.TARGETS:
            shared_curve = {h: {"horizon": h, "shared": f"{seed}-{target}-{h}"}
                            for h in (0, 100)}
            for policy in ("raw", "current32"):
                identity = f"s{seed}-sgdm-lr0.03-{target}-{policy}"
                checkpoints = []
                for horizon in runner.plans.HORIZONS:
                    kind = "full_state" if horizon in (0, 100, 2000) else "model_state"
                    name = f"{kind}-{identity}-h{horizon}.pt"
                    path = confirmation / name
                    path.write_bytes(f"{identity}:{horizon}".encode())
                    record = {"name": name, "bytes": path.stat().st_size,
                              "sha256": runner.old.digest(path)}
                    declared.append(record)
                    checkpoint = {"horizon": horizon, kind: record}
                    if kind == "full_state":
                        checkpoint["full_state_digest"] = (
                            f"warmup-{seed}-{target}" if horizon == 100
                            else f"state-{identity}-{horizon}")
                    checkpoints.append(checkpoint)
                curve_rows = [copy.deepcopy(shared_curve[h]) if h in shared_curve else
                              {"horizon": h, "policy": policy}
                              for h in runner.plans.HORIZONS]
                curve = {"schema": "i14_trajectory_v1", "id": identity,
                         "seed": seed, "base": "sgdm", "lr": .03,
                         "target": target, "policy": policy, "status": "complete",
                         "completed_steps": 2000,
                         "warmup_full_state_digest": f"warmup-{seed}-{target}",
                         "warmup_evaluation_digest": f"warmup-eval-{seed}-{target}",
                         "curve": curve_rows, "checkpoints": checkpoints}
                curve_record = _save_json(confirmation / f"curve-{identity}.json", curve)
                declared.append(curve_record)
                entries.append({key: curve[key] for key in (
                    "id", "seed", "base", "lr", "target", "policy", "status",
                    "completed_steps", "warmup_full_state_digest",
                    "warmup_evaluation_digest")} | {"artifact": curve_record})

    trajectories = _save_json(confirmation / "trajectories.json", {"entries": entries})
    declared.append(trajectories)
    for seed in runner.SEEDS:
        declared.append(_save_json(confirmation / f"plan-confirmation-s{seed}.json",
                                   {"seed": seed}))
        declared.append(_save_json(confirmation / f"corruption-s{seed}.json",
                                   {"replaced_count": 4500 + seed,
                                    "incorrect_count": 4000 + seed,
                                    "train_count": 5000}))
    dataset_inputs = {name: runner.old.digest(data_dir / name) for name in
                      ("train-images-idx3-ubyte", "train-labels-idx1-ubyte")}
    declared.append(_save_json(confirmation / "calibration-binding.json",
                               {"selection": {"dataset_inputs": dataset_inputs}}))
    completion = {"status": "complete", "numerical_failures": 0,
                  "frozen_commit": runner.I14_COMMIT, "artifacts": declared}
    completion_path = confirmation / "completion.json"
    completion_path.write_text(json.dumps(completion))
    supplement_payload = {
        "status": "accepted_with_empty_runtime_directory_exception",
        "original_audit_status": "fail", "artifact_root": str(source),
        "original_audit_sha256": runner.old.digest(analysis / "audit.json"),
        "original_summary_sha256": runner.old.digest(analysis / "summary.json"),
        "bound_terminal_and_attempt_sha256": {
            "confirmation/completion.json": runner.old.digest(completion_path)},
    }
    supplement = analysis / "runtime-directory-supplement.json"
    supplement.write_text(json.dumps(supplement_payload))
    return i14, source, data_dir, supplement


class _NoWriteRun:
    def __init__(self):
        self.saved = 0
        self.started = 0.0

    def check(self):
        return None

    def save(self, *_args, **_kwargs):
        self.saved += 1
        raise AssertionError("invalid input reached artifact creation")


class HistoryRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _threads()
        cls.data = _data()
        cls.plan = _plan()
        cls.clean_parent, cls.clean_start = _parent(cls.data, cls.plan, "clean")

    def run_one(self, root: Path, phase: str, policy: str,
                *, target: str = "clean"):
        state, expected = ((self.clean_parent, self.clean_start) if target == "clean"
                           else _parent(self.data, self.plan, "fixed"))
        run = runner.old.Run(root, phase, 60)
        entry = runner.run_branch(run, state, self.data, self.plan, target, policy,
                                  expected, smoke=True, end=110,
                                  horizons=(100, 110))
        result = json.loads((run.path / entry["artifact"]["name"]).read_text())
        return run, entry, result

    def test_invalid_arm_dimensions_parent_and_batches_reject_before_write(self):
        cases = []
        cases.append((self.clean_parent, self.plan, "clean", "current_native", 110,
                      (100, 110)))
        cases.append((self.clean_parent, self.plan, "other", "mean_native", 110,
                      (100, 110)))
        bad_parent = copy.deepcopy(self.clean_parent)
        bad_parent["tracker"]["step_count"] = 99
        cases.append((bad_parent, self.plan, "clean", "mean_native", 110,
                      (100, 110)))
        cases.append((self.clean_parent, self.plan, "clean", "mean_native", 111,
                      (100, 111)))
        cases.append((self.clean_parent, self.plan, "clean", "mean_native", 110,
                      (100, 109, 110)))
        bad_shape = copy.deepcopy(self.plan)
        bad_shape["training_batches"] = bad_shape["training_batches"][:-1]
        cases.append((self.clean_parent, bad_shape, "clean", "mean_native", 110,
                      (100, 110)))
        bad_dtype = copy.deepcopy(self.plan)
        bad_dtype["training_batches"] = bad_dtype["training_batches"].astype(np.int32)
        cases.append((self.clean_parent, bad_dtype, "clean", "mean_native", 110,
                      (100, 110)))
        bad_index = copy.deepcopy(self.plan)
        bad_index["training_batches"][100, 0] = len(self.data["x"])
        cases.append((self.clean_parent, bad_index, "clean", "mean_native", 110,
                      (100, 110)))

        for state, plan, target, policy, end, horizons in cases:
            fake = _NoWriteRun()
            with self.subTest(target=target, policy=policy, end=end,
                              shape=plan["training_batches"].shape):
                with self.assertRaises((ValueError, runner.core.HistoryCoreError)):
                    runner.run_branch(fake, state, self.data, plan, target, policy,
                                      self.clean_start, smoke=True, end=end,
                                      horizons=horizons)
                self.assertEqual(fake.saved, 0)

    def test_true_h100_to_h110_fork_uses_rows_100_through_109(self):
        parent_before = runner.i9.tree_digest(self.clean_parent)
        expected_batches = [self.data["x"][self.plan["training_batches"][row]]
                            for row in range(100, 110)]
        used: list[torch.Tensor] = []
        original = runner.core.history_step

        def recording_step(model, optimizer, tracker, x, target, policy,
                           *, capture_digests=False):
            used.append(x.detach().clone())
            return original(model, optimizer, tracker, x, target, policy,
                            capture_digests=capture_digests)

        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.object(runner.core, "history_step", side_effect=recording_step):
            _, entry, result = self.run_one(Path(directory), "fork", "mean_native")

        self.assertEqual(entry["completed_updates"], 10)
        self.assertEqual(result["status"], "complete")
        self.assertEqual([row["step"] for row in result["steps"]], list(range(101, 111)))
        self.assertEqual([row["relative_step"] for row in result["steps"]],
                         list(range(1, 11)))
        self.assertEqual([row["horizon"] for row in result["curve"]], [100, 110])
        self.assertEqual(result["checkpoints"][0]["horizon"], 110)
        self.assertEqual(result["checkpoints"][0]["relative_horizon"], 10)
        self.assertIn("full_state", result["checkpoints"][0])
        self.assertNotIn("model_state", result["checkpoints"][0])
        self.assertEqual(len(used), 10)
        self.assertTrue(all(torch.equal(actual, expected)
                            for actual, expected in zip(used, expected_batches)))
        self.assertFalse(torch.equal(used[0],
                                     self.data["x"][self.plan["training_batches"][99]]))
        self.assertEqual(runner.i9.tree_digest(self.clean_parent), parent_before)

    def test_h100_evaluation_is_state_and_rng_neutral(self):
        model, optimizer, tracker = runner.c14.restore(self.clean_parent, "cpu")
        before_state = runner.i9.tree_digest(
            runner.c14.snapshot(model, optimizer, tracker, "sgdm", .03))
        before_rng = runner.i9._rng_state()
        measured = {"horizon": 100, **runner.old.evaluate(model, self.data)}
        after_state = runner.i9.tree_digest(
            runner.c14.snapshot(model, optimizer, tracker, "sgdm", .03))
        self.assertEqual(measured, self.clean_start)
        self.assertEqual(before_state, after_state)
        self.assertTrue(runner.i9.equal_tree(before_rng, runner.i9._rng_state()))

    def test_first_step_digest_pairing_is_enforced(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entries = [self.run_one(root, "pair-" + policy, policy)[1]
                       for policy in runner.core.REAL_POLICIES]
        checks = runner.validate_first_step_pairs(entries)
        self.assertEqual(checks, [{"seed": 215, "target": "clean",
                                   "available_first_steps": 3, "status": "pass"}])
        common = ("raw_gradient", "post_observer", "old_momentum_buffer",
                  "native_action_old_buffer")
        for key in common:
            self.assertEqual(len({row["first_step_digests"][key] for row in entries}), 1)
        means = [row for row in entries if row["policy"].startswith("mean_")]
        self.assertEqual(len({row["first_step_digests"]["applied_gradient"]
                              for row in means}), 1)

        altered = copy.deepcopy(entries)
        altered[0]["first_step_digests"]["raw_gradient"] = "different"
        with self.assertRaisesRegex(ValueError, "common tensor"):
            runner.validate_first_step_pairs(altered)
        incomplete = copy.deepcopy(entries)
        incomplete[0]["first_step_digests"] = None
        incomplete[0]["status"] = "numerical_failure"
        self.assertEqual(runner.validate_first_step_pairs(incomplete)[0]["status"],
                         "partial_numerical_evidence")
        incomplete[0]["status"] = "complete"
        with self.assertRaisesRegex(ValueError, "lacks first-step"):
            runner.validate_first_step_pairs(incomplete)

    def test_nonfinite_is_partial_evidence_but_structural_error_propagates(self):
        original = runner.core.history_step
        calls = 0

        def fail_second(model, optimizer, tracker, x, target, policy,
                        *, capture_digests=False):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise runner.c14.NumericalFailure("synthetic nonfinite")
            return original(model, optimizer, tracker, x, target, policy,
                            capture_digests=capture_digests)

        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.object(runner.core, "history_step", side_effect=fail_second):
            run, entry, result = self.run_one(Path(directory), "numeric", "mean_native")
            self.assertEqual((entry["status"], entry["completed_updates"]),
                             ("numerical_failure", 1))
            self.assertEqual(result["failure"]["attempted_step"], 102)
            self.assertEqual(result["failure"]["last_valid_horizon"], 100)
            self.assertEqual([row["step"] for row in result["steps"]], [101])
            self.assertIsNotNone(entry["first_step_digests"])
            self.assertTrue((run.path /
                result["failure"]["state_artifact"]["name"]).is_file())

        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.object(runner.core, "history_step",
                                  side_effect=runner.core.HistoryCoreError(
                                      "synthetic structural failure")):
            root = Path(directory)
            with self.assertRaisesRegex(runner.core.HistoryCoreError, "structural"):
                self.run_one(root, "structural", "mean_native")
            path = root / "structural"
            self.assertFalse(any(path.glob("branch-*.json")))
            self.assertFalse(any(path.glob("failed-state-*.pt")))

    def test_checked_artifact_rejects_escape_symlink_size_and_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            file = root / "good.json"
            file.write_bytes(b"evidence\n")
            record = {"name": file.name, "bytes": file.stat().st_size,
                      "sha256": runner.old.digest(file)}
            self.assertEqual(runner.checked_artifact(root, record), file)

            for changed in (
                record | {"name": "../good.json"},
                record | {"bytes": record["bytes"] + 1},
                record | {"sha256": "0" * 64},
                record | {"extra": True},
            ):
                with self.subTest(record=changed):
                    with self.assertRaises(ValueError):
                        runner.checked_artifact(root, changed)
            link = root / "link.json"
            link.symlink_to(file)
            with self.assertRaisesRegex(ValueError, "nonsymlink"):
                runner.checked_artifact(root, record | {"name": link.name})

    def test_fake_bound_inputs_rejects_unpinned_or_wrong_supplement(self):
        with tempfile.TemporaryDirectory() as directory:
            i14 = Path(directory) / "iteration-014"
            analysis = i14 / "analysis-001"
            analysis.mkdir(parents=True)
            supplement = analysis / "runtime-directory-supplement.json"
            supplement.write_text("{}\n")
            with mock.patch.object(runner, "I14", i14), \
                    mock.patch.object(runner, "SOURCE_ROOT", Path(directory) / "source"), \
                    mock.patch.object(runner, "SUPPLEMENT_SHA", "0" * 64):
                with self.assertRaisesRegex(ValueError, "supplement differs"):
                    runner.bound_inputs()

            payload = {"status": "accepted_with_empty_runtime_directory_exception",
                       "original_audit_status": "fail",
                       "artifact_root": str(Path(directory) / "wrong-source")}
            supplement.write_text(json.dumps(payload))
            with mock.patch.object(runner, "I14", i14), \
                    mock.patch.object(runner, "SOURCE_ROOT", Path(directory) / "source"), \
                    mock.patch.object(runner, "SUPPLEMENT_SHA", runner.old.digest(supplement)):
                with self.assertRaisesRegex(ValueError, "accepted provenance"):
                    runner.bound_inputs()

    def test_fake_bound_inputs_accepts_closed_fixture_and_detects_change(self):
        with tempfile.TemporaryDirectory() as directory:
            i14, source, data_dir, supplement = _fake_i14_inputs(Path(directory))

            def parse_plan(payload):
                return {"seed": payload["seed"],
                        "training_batches": np.zeros((2000, 64), dtype=np.int64)}

            patches = (
                mock.patch.object(runner, "I14", i14),
                mock.patch.object(runner, "SOURCE_ROOT", source),
                mock.patch.object(runner, "SUPPLEMENT_SHA",
                                  runner.old.digest(supplement)),
                mock.patch.object(runner.plans, "DATA", data_dir),
                mock.patch.object(runner.plans, "plan_from_json",
                                  side_effect=parse_plan),
            )
            with patches[0], patches[1], patches[2], patches[3], patches[4]:
                binding, parents, curves, saved_plans = runner.bound_inputs()

            self.assertEqual(binding["schema"], "i15_i14_inputs_v1")
            self.assertEqual(len(parents), 6)
            self.assertEqual(len(curves), 12)
            self.assertEqual(set(saved_plans), set(runner.SEEDS))
            self.assertEqual(len(binding["parents"]), 6)
            self.assertEqual(len(binding["references"]), 12)
            self.assertEqual(set(binding["corruption_counts"]),
                             {str(seed) for seed in runner.SEEDS})
            self.assertIn("trajectories.json", binding["source_records"])
            self.assertEqual(
                {record["horizon"] for reference in binding["references"]
                 for record in reference["checkpoint_records"]},
                set(runner.plans.HORIZONS))

            changed_plan = source / "confirmation" / "plan-confirmation-s200.json"
            changed_plan.write_text('{"seed":999}')
            patches = (
                mock.patch.object(runner, "I14", i14),
                mock.patch.object(runner, "SOURCE_ROOT", source),
                mock.patch.object(runner, "SUPPLEMENT_SHA",
                                  runner.old.digest(supplement)),
                mock.patch.object(runner.plans, "DATA", data_dir),
                mock.patch.object(runner.plans, "plan_from_json",
                                  side_effect=parse_plan),
            )
            with patches[0], patches[1], patches[2], patches[3], patches[4]:
                with self.assertRaisesRegex(ValueError, "Artifact content differs"):
                    runner.bound_inputs()

    def test_inherited_writer_counts_runtime_and_enforces_output_cap(self):
        buffer = io.BytesIO()
        writer = runner.old.BudgetWriter(buffer, 3)
        self.assertEqual(writer.write(b"abc"), 3)
        with self.assertRaisesRegex(RuntimeError, "budget"):
            writer.write(b"d")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime = root / "runtime"
            runtime.mkdir()
            (runtime / "cache").write_bytes(b"runtime")
            run = runner.old.Run(root, "bounded", 30)
            self.assertGreaterEqual(run.used(), 7)
            record = run.save("small.json", {"ok": True})
            self.assertGreater(record["bytes"], 0)
            with self.assertRaises(ValueError):
                run.save("../escape.json", {"ok": True})
            with self.assertRaises(FileExistsError):
                run.save("small.json", {"ok": False})
            used = run.used()
            with mock.patch.object(runner.old, "ARTIFACT_CAP",
                                   used + runner.old.RESERVE + 3):
                with self.assertRaisesRegex(RuntimeError, "ceiling"):
                    run.save("too-large.json", {"more": "than three bytes"})
                self.assertFalse((run.path / "too-large.json").exists())
            with mock.patch.object(runner.old, "ARTIFACT_CAP", used + 10), \
                    mock.patch.object(runner.old, "RESERVE", 4):
                with self.assertRaisesRegex(RuntimeError, "terminal metadata"):
                    run.terminal("completion.json", {"not": "small"})


if __name__ == "__main__":
    unittest.main()
