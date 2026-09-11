"""Synthetic-only runner/state/receipt fixtures; never load scientific inputs."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from experiments import grokking_raw_direction as runner
from experiments import run_grokking_raw_direction_batch as batch
from experiments.grokking_action_intervention import snapshot, scientific_hash
from experiments.grokking_action_policy import LegacyActionPolicyFilter
from experiments.grokking_confirmation import (
    ADAMW, FILTER, FILTER_MUTABLE, FILTER_STATIC, filter_configuration,
    filter_state, parameter_identity, tensor_set_identity,
)
from experiments.grokking_raw_direction_policy import RawDirectionPolicyFilter


def synthetic_state():
    torch.manual_seed(619)
    model = torch.nn.Linear(3, 2, bias=False)
    optimizer = torch.optim.AdamW(model.parameters(), **ADAMW)
    tracker = LegacyActionPolicyFilter(model, optimizer, action_policy="native",
                                       **FILTER, stable_update=False)
    for _ in range(3):
        model.weight.grad = torch.randn_like(model.weight)
        optimizer.step()
    tracker.V = torch.tensor([[1., .1], [.2, .6], [0., .1], [.2, 0.], [.1, .3], [.2, .1]])
    tracker.S = torch.tensor([.3, .2])
    tracker.grad_mean = torch.ones(6) * .01
    tracker.step_count = runner.FORK_STEP
    data = (torch.randn(4, 3), torch.tensor([0, 1, 1, 0]))
    contract = {"config": {"seed": 100, "synthetic": True},
                "source_identity": {"synthetic": True},
                "split_identity": tensor_set_identity((*data, *data)),
                "parameter_identity": parameter_identity(model)}
    return snapshot(model, optimizer, tracker, contract, "cpu"), data


class RawDirectionRunnerTests(unittest.TestCase):
    def test_full_state_restores_without_configuration_change(self):
        state, _ = synthetic_state()
        with patch.object(runner, "GrokkingTransformer",
                          side_effect=lambda **kwargs: torch.nn.Linear(3, 2, bias=False)):
            model, optimizer, tracker, actual = runner.restored(state, "cpu")
        self.assertIsInstance(tracker, RawDirectionPolicyFilter)
        self.assertTrue(tracker.retain_action_basis)
        self.assertEqual(actual, scientific_hash(state))
        self.assertEqual(scientific_hash(snapshot(model, optimizer, tracker, state, "cpu")), actual)
        self.assertEqual(set(filter_state(tracker)), set(FILTER_MUTABLE))
        self.assertEqual(set(filter_configuration(tracker)), set(FILTER_STATIC))
        self.assertEqual(filter_configuration(tracker), state["filter_configuration"])
        self.assertTrue(torch.equal(torch.get_rng_state(), state["torch_cpu_rng_state"]))
        before = state["optimizer_state"]["state"][0]
        after = optimizer.state[model.weight]
        for key in ("step", "exp_avg", "exp_avg_sq"):
            self.assertTrue(torch.equal(before[key], after[key]))

    def test_one_training_step_and_full_first_tensors(self):
        state, data = synthetic_state()
        with patch.object(runner, "GrokkingTransformer",
                          side_effect=lambda **kwargs: torch.nn.Linear(3, 2, bias=False)):
            model, optimizer, tracker, _ = runner.restored(state, "cpu")
        full_before = snapshot(model, optimizer, tracker, state, "cpu")
        with patch.object(runner, "train_step", wraps=runner.train_step) as train:
            row, diagnostic, seconds, diagnostic_seconds = runner.step_with_diagnostic(
                model, optimizer, tracker, *data, "cpu", 1501)
        self.assertEqual(train.call_count, 1)
        self.assertEqual(tracker.step_count, 1501)
        self.assertEqual(int(optimizer.state[model.weight]["step"]), 4)
        self.assertEqual(row["step"], 1501)
        self.assertEqual(row["adam"], diagnostic["summary"])
        self.assertGreaterEqual(seconds, 0)
        self.assertGreaterEqual(diagnostic_seconds, 0)
        self.assertLess(row["adam"]["decomposition_residual_max_abs"], 1e-7)
        self.assertEqual(set(tracker.last_action_result["actions"]),
                         {"native", "orthogonal", "norm_matched", runner.POLICY})
        self.assertIn("Q", tracker.last_action_result)
        self.assertEqual(tracker.last_raw_gradient.numel(), 6)
        self.assertTrue(torch.equal(model.weight.grad.flatten(),
                                   tracker.last_action_result["actions"][runner.POLICY]))
        self.assertEqual(scientific_hash(full_before), scientific_hash(state))
        tracker.retain_action_basis = False
        row, _, _, _ = runner.step_with_diagnostic(model, optimizer, tracker, *data, "cpu", 1502)
        self.assertEqual(row["step"], 1502)
        self.assertNotIn("Q", tracker.last_action_result)

    def test_fixed_commands_cannot_invoke_old_runner(self):
        parent = Path("/synthetic/no-execution")
        self.assertEqual(runner.expected_roster(), [["raw_norm_matched", 1501],
                                                   ["raw_norm_matched", 2000],
                                                   ["raw_norm_matched", 2500]])
        for seed in range(100, 105):
            command = batch.command_for(seed, parent)
            self.assertEqual(Path(command[1]).name, "grokking_raw_direction.py")
            self.assertNotIn("grokking_action_intervention.py", " ".join(command))
            self.assertEqual(command[command.index("--seed") + 1], str(seed))
            self.assertEqual("--admission-dir" in command, seed != 100)
            if seed != 100:
                self.assertEqual(command[-1], str(parent / "seed100"))
        with self.assertRaises(ValueError):
            batch.command_for(105, parent)
        with patch.object(runner, "load_checkpoint", side_effect=AssertionError("must not load")):
            with self.assertRaises(ValueError):
                runner.run(99, parent)
            with self.assertRaises(ValueError):
                runner.run(101, parent)

    def test_batch_deadline_and_no_automatic_retry(self):
        with tempfile.TemporaryDirectory() as temporary:
            storage = Path(temporary)
            parent = storage / "new-batch"
            parent.mkdir()
            with patch.object(batch, "STORAGE", storage), \
                    patch.object(batch, "source_pins", return_value={"synthetic": "x"}), \
                    patch.object(batch.subprocess, "check_output", return_value="synthetic\n"), \
                    patch.object(batch.time, "monotonic", side_effect=[0., runner.BATCH_SECONDS, runner.BATCH_SECONDS]), \
                    patch.object(batch.subprocess, "run") as launch:
                with self.assertRaises(TimeoutError):
                    batch.run(parent)
            launch.assert_not_called()
            self.assertTrue((parent / "batch-failure.json").is_file())

    def test_writer_exclusive_json_and_atomic_tensor(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            writer = runner.BoundedWriter(root)
            path = root / "value.json"
            receipt = writer.json(path, {"synthetic": True})
            runner.verify_receipt(receipt, root)
            with self.assertRaises(ValueError):
                writer.json(path, {"replacement": True})
            self.assertEqual(json.loads(path.read_text()), {"synthetic": True})
            state = {"synthetic": torch.arange(3)}
            tensor_receipt = writer.checkpoint(root / "tiny.pt", state)
            runner.verify_receipt(tensor_receipt, root)
            self.assertFalse(any(path.name.endswith(".incomplete") for path in root.iterdir()))
            with self.assertRaises(ValueError):
                writer.checkpoint(root / "nonfinite.pt", {"x": torch.tensor(float("nan"))})
            with self.assertRaises(ValueError):
                writer.reserve(root.parent / "escape.json", 1)

    def test_writer_budget_and_free_space_are_prospective(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            writer = runner.BoundedWriter(root)
            with patch.object(runner, "MAX_BYTES", 4):
                with self.assertRaises(RuntimeError):
                    writer.json(root / "too-big.json", {"synthetic": True})
            self.assertEqual(list(root.iterdir()), [])
            usage = type("Usage", (), {"free": runner.FREE_RESERVE})()
            with patch.object(runner.shutil, "disk_usage", return_value=usage):
                with self.assertRaises(RuntimeError):
                    writer.json(root / "no-reserve.json", {"x": 1})
            self.assertEqual(list(root.iterdir()), [])
            with patch.object(runner.time, "monotonic", return_value=runner.SEED_SECONDS):
                with self.assertRaises(TimeoutError):
                    runner.check_budget(writer, 0)

    @staticmethod
    def completed_fixture(directory, seed=100):
        writer = runner.BoundedWriter(directory.parent)
        pins, env = {"synthetic": "x"}, {"device": "synthetic"}
        artifacts = [writer.json(directory / name, {"synthetic": True}) for name in
                     ("manifest.json", "first-step-tensors.pt", "first-step-summary.json")]
        checkpoints = []
        for step in runner.CAPTURE_STEPS:
            receipt = writer.json(directory / f"{runner.POLICY}-step-{step:06d}.pt",
                                  {"synthetic": True})
            receipt.update({"seed": seed, "policy": runner.POLICY, "step": step})
            checkpoints.append(receipt)
            artifacts.append(receipt)
            artifacts.append(writer.json(directory / f"{runner.POLICY}-through-{step:06d}.json", {
                "schema": runner.SCHEMA, "seed": seed, "policy": runner.POLICY,
                "source_sha256": pins, "step": step, "checkpoint": receipt,
                "action_history": [{"step": value} for value in range(1501, step + 1)]}))
        completion = {"schema": runner.SCHEMA, "status": "complete", "seed": seed,
                      "policy": runner.POLICY, "source_sha256": pins, "environment": env,
                      "accepted_roster": runner.expected_roster(), "completed_updates": 1000,
                      "history_steps": list(range(1501, 2501)), "elapsed_seconds": 10.,
                      "artifact_receipts": artifacts, "checkpoints": checkpoints}
        writer.json(directory / "complete.json", completion)
        return completion, pins, env

    def test_complete_admission_binds_every_history_and_artifact(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "seed100"
            directory.mkdir()
            expected, pins, env = self.completed_fixture(directory)
            self.assertEqual(runner.verify_completion(directory, 100, pins, env), expected)
            with self.assertRaises(ValueError):
                runner.verify_completion(directory, 101, pins, env)
            with self.assertRaises(ValueError):
                runner.verify_completion(directory, 100, {"changed": "x"}, env)
            with self.assertRaises(ValueError):
                runner.verify_completion(directory, 100, pins, {"changed": "x"})
            history = directory / f"{runner.POLICY}-through-002500.json"
            # A changed receipt must fail even if its metric values look favorable.
            with patch.object(runner, "file_hash", side_effect=lambda path:
                              "changed" if path == history else runner_hash(path)):
                with self.assertRaises(ValueError):
                    runner.verify_completion(directory, 100, pins, env)

    def test_parent_binding_checks_exact_accepted_fork(self):
        with tempfile.TemporaryDirectory() as temporary:
            archive = Path(temporary)
            writer = runner.BoundedWriter(archive)
            parent = {"path": "/synthetic/no-load.pt", "sha256": "synthetic"}
            accepted = []
            for seed in runner.SEEDS:
                directory = archive / f"seed{seed}"
                directory.mkdir()
                receipt = writer.json(directory / "complete.json", {
                    "status": "complete", "seed": seed, "parent_checkpoint": parent,
                    "parent_metrics_sha256": "metrics", "fork_scientific_state_sha256": "fork",
                    "source_sha256": {"synthetic": "unchanged"}})
                accepted.append({"seed": seed, **receipt})
            receipt = writer.json(archive / "batch-complete.json", {
                "status": "complete", "seeds": list(runner.SEEDS), "accepted": accepted})
            with patch.object(runner, "ARCHIVED_BATCH", archive), \
                    patch.object(runner, "ARCHIVED_BATCH_SHA256", receipt["sha256"]), \
                    patch.object(runner, "legacy_source_pins", return_value={"synthetic": "unchanged"}), \
                    patch.object(runner, "load_checkpoint", side_effect=AssertionError("no loading")):
                result = runner.archived_parent(100, parent, "metrics")
                self.assertEqual(result["fork_scientific_state_sha256"], "fork")
                with self.assertRaises(ValueError):
                    runner.archived_parent(100, {**parent, "sha256": "changed"}, "metrics")
                with self.assertRaises(ValueError):
                    runner.archived_parent(100, parent, "changed")

    def test_batch_runs_exact_roster_without_metric_gate(self):
        with tempfile.TemporaryDirectory() as temporary:
            storage = Path(temporary)
            parent = storage / "new-batch"
            parent.mkdir()
            launched = []

            def fake_run(command, *, check, cwd, timeout):
                seed = int(command[command.index("--seed") + 1])
                launched.append(seed)
                self.assertTrue(check)
                self.assertLessEqual(timeout, runner.SEED_SECONDS)
                self.assertEqual(Path(command[1]).name, "grokking_raw_direction.py")
                directory = parent / f"seed{seed}"
                directory.mkdir()
                self.completed_fixture(directory, seed)

            with patch.object(batch, "STORAGE", storage), \
                    patch.object(batch, "source_pins", return_value={"synthetic": "x"}), \
                    patch.object(batch.subprocess, "check_output", return_value="synthetic\n"), \
                    patch.object(batch.subprocess, "run", side_effect=fake_run):
                batch.run(parent)
            self.assertEqual(launched, [100, 101, 102, 103, 104])
            complete = json.loads((parent / "batch-complete.json").read_text())
            self.assertEqual([item["seed"] for item in complete["accepted"]], launched)
            self.assertEqual(complete["status"], "complete")
            self.assertEqual(complete["paid_spend_usd"], 0)


runner_hash = runner.file_hash


if __name__ == "__main__":
    unittest.main()
