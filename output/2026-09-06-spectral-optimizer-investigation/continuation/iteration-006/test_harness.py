"""Synthetic CPU tests only. Run with CUDA_VISIBLE_DEVICES=''."""
import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import random
import tempfile
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("iteration006_harness_tested", HERE / "delivery_order_harness.py")
d = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d)
torch, np = d.torch, d.np


def tiny_model(seed, device):
    random.seed(seed)
    torch.manual_seed(seed)
    return torch.nn.Linear(3, 2).to(device)


def tiny_fixture(seed=9880, steps=220):
    generator = torch.Generator().manual_seed(771)
    x = torch.randn(8, 3, generator=generator)
    clean = torch.tensor([0, 1, 0, 1, 1, 0, 1, 0])
    noisy = torch.tensor([0, 0, 1, 1, 0, 1, 1, 0])
    plan = {"seed": seed, "initialization_seed": 223,
            "training_batches": np.random.default_rng(119).integers(0, 8, (steps, 4))}
    data = {"x": x, "clean": clean, "noisy": noisy, "vx": x.clone(), "vy": clean.clone(),
            "realized_replacement_fraction": .5, "realized_incorrect_fraction": .5}
    return plan, data


def fixture_completion(store):
    results, bindings = [], []
    rows = [{"step": step} for step in range(1, 2001)]
    grid = [{"step": step, "accuracy": step / 2000., "cross_entropy": 3 - step / 2000., "count": 5000}
            for step in range(0, 2001, 100)]
    for identity in d.expected_cells():
        states = {name: {"weight": torch.tensor([100. if name == "warmup100" else 2000.])} for name in d.CHECKPOINTS}
        item = store.checkpoints(identity["run_key"] + ".pt", states, **identity)
        result = {**identity, "steps": 2000, "delivery_gate_steps": 2000,
                  "all_invariant_gates_passed": True, "warmup_checks_passed": True,
                  "steps_raw": rows, "validation_trajectory": grid,
                  "checkpoint_steps": {"final": 2000, "warmup100": 100, "min_val_ce": 2000, "max_val_accuracy": 2000},
                  "checkpoint_sha256": {k: d.tree_hash(v) for k, v in states.items()},
                  "checkpoint_path": item["path"]}
        results.append(result)
        bindings.append(item)
    return results, bindings


class HarnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
            raise RuntimeError("Synthetic harness tests require CUDA_VISIBLE_DEVICES=''")
        torch.set_num_threads(1)
        if torch.cuda.is_initialized():
            raise RuntimeError("These tests must never initialize CUDA")

    def test_exact_cell_identity_and_source_membership(self):
        cells = d.expected_cells()
        self.assertEqual(len(cells), 36)
        self.assertEqual(len({x["run_key"] for x in cells}), 36)
        self.assertEqual({x["replacement_probability"] for x in cells}, {0., .9})
        self.assertEqual({x["seed"] for x in cells}, {6, 7, 8})
        self.assertEqual(len(d.source_paths()), 15)
        self.assertNotIn("iteration-004", " ".join(str(p) for p in d.source_paths()))

    def test_selector_earliest_independent_and_cpu_storage(self):
        selector = d.Selectors()
        original = {"weight": torch.tensor([1., 2.])}
        score = {"cross_entropy": 2., "accuracy": .5, "count": 5000}
        selector.consider(0, score, original, 5000)
        self.assertNotEqual(selector.states["min_val_ce"]["weight"].data_ptr(), original["weight"].data_ptr())
        self.assertNotEqual(selector.states["min_val_ce"]["weight"].data_ptr(), selector.states["max_val_accuracy"]["weight"].data_ptr())
        original["weight"].add_(4)
        selector.consider(100, score, original, 5000)
        self.assertEqual(selector.steps, {"min_val_ce": 0, "max_val_accuracy": 0})
        self.assertEqual(selector.states["min_val_ce"]["weight"].tolist(), [1., 2.])
        selector.consider(200, {**score, "cross_entropy": 1.}, original, 5000)
        selector.consider(300, {**score, "accuracy": .6}, original, 5000)
        self.assertEqual(selector.steps, {"min_val_ce": 200, "max_val_accuracy": 300})

    def test_selector_rejects_nonfinite_before_context_append(self):
        selector = d.Selectors()
        state = {"weight": torch.ones(1)}
        for value in (float("nan"), float("inf")):
            with self.assertRaises(AssertionError):
                selector.consider(0, {"cross_entropy": value, "accuracy": .5, "count": 8}, state, 8)
            self.assertEqual(selector.validations, [])
            json.dumps(selector.validations, allow_nan=False)

    def test_numpy_state_is_captured_and_compared(self):
        model = tiny_model(41, "cpu")
        optimizer = d.h.make_optimizer(model)
        initial = np.random.get_state()
        try:
            a = d.complete_snapshot(model, optimizer, None)
            np.random.random()
            b = d.complete_snapshot(model, optimizer, None)
            self.assertFalse(d.equal_tree(a, b))
            self.assertNotEqual(d.tree_hash(a), d.tree_hash(b))
            np.random.set_state(initial)
            self.assertTrue(d.equal_tree(a, d.complete_snapshot(model, optimizer, None)))
        finally:
            np.random.set_state(initial)

    def test_update_zero_cosines_and_positive_sign_denominator_inputs(self):
        value = d.update_metrics(torch.zeros(3), torch.zeros(3), torch.ones(3))
        self.assertIsNone(value["raw_gradient_update_cosine"])
        self.assertEqual(value["null_reasons"]["raw_gradient_update_cosine"], "zero_vector_norm")
        self.assertEqual(value["raw_gradient_dot_update"]["sign"], 0)
        self.assertEqual(value["raw_gradient_dot_update"]["tolerance"], 1e-14)
        self.assertEqual(d.update_metrics(torch.ones(3), torch.ones(3), torch.ones(3))["raw_gradient_dot_update"]["sign"], 1)

    def test_training_label_metrics_share_logits(self):
        class CountModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.calls = 0
            def forward(self, values):
                self.calls += 1
                return torch.stack((values[:, 0], -values[:, 0]), 1)
        model = CountModel()
        data = {"x": torch.tensor([[1.], [-1.], [2.], [-2.]]),
                "clean": torch.tensor([0, 1, 0, 1]), "noisy": torch.tensor([1, 1, 1, 1])}
        clean, noisy = d.training_evaluations(model, data)
        self.assertEqual(model.calls, 1)
        self.assertEqual(clean["accuracy"], 1.)
        self.assertEqual(noisy["accuracy"], .5)
        self.assertEqual(clean["count"], 4)

    def test_float32_fraction_contract_is_not_exact_count_division(self):
        mask = torch.zeros(5000)
        mask[:4054] = 1
        rate = float(mask.mean())
        self.assertNotEqual(rate, 4054 / 5000)
        self.assertLessEqual(abs(rate - round(rate * 5000) / 5000), 3e-8)
        self.assertGreater(abs(rate * 5000 - round(rate * 5000)), 1e-8)

    def test_plan_dtype_and_bounds_and_condition_rate_gate(self):
        plan = d.h.make_plan(6, 2)
        d.validate_plan(plan, 2)
        bad = {**plan, "training_batches": plan["training_batches"].astype(float)}
        with self.assertRaises(AssertionError):
            d.validate_plan(bad, 2)
        bad = {**plan, "training_batches": np.full((2, 64), 5000, dtype=np.int64)}
        with self.assertRaises(AssertionError):
            d.validate_plan(bad, 2)
        clean = torch.zeros(5000, dtype=torch.int64)
        noisy = clean.clone()
        noisy[:4054] = 1
        mask = np.r_[np.zeros(4054), np.full(946, .99)]
        rate = float((noisy != clean).float().mean())
        data = {"x": torch.zeros(5000, 1), "vx": torch.zeros(5000, 1), "ax": torch.zeros(5000, 1),
                "clean": clean, "noisy": noisy, "realized_incorrect_fraction": rate,
                "realized_replacement_fraction": rate}
        d.validate_data(data, .9, {"replacement_uniforms": mask})
        self.assertEqual(data["realized_incorrect_fraction"], rate)
        with self.assertRaises(AssertionError):
            d.validate_data(data, 0., {"replacement_uniforms": mask})

    def test_all_six_policies_synthetic_on_off_and_warmup(self):
        plan, data = tiny_fixture()
        references, observer_references = {}, {}
        with contextlib.redirect_stdout(io.StringIO()):
            for arm in d.ARMS:
                traces, finals = {}, {}
                for instrumented in (False, True):
                    with mock.patch.object(d.policy, "delivery_metrics", wraps=d.policy.delivery_metrics) as delivery:
                        result, checkpoint, final, warm = d.train_cell(
                            plan, data, .9, arm, 220, instrumented, True,
                            d.storage.Guard(True, "cpu"), {}, device="cpu", model_factory=tiny_model)
                    self.assertEqual(delivery.call_count, 220)
                    self.assertEqual(checkpoint, {})
                    self.assertNotIn("steps_raw", result)
                    self.assertNotIn("validation_trajectory", result)
                    self.assertNotIn("test", result)
                    self.assertEqual(result["measurement_state_checks"], 5 if instrumented else 0)
                    if arm == "adamw":
                        references[instrumented] = (result, warm)
                    else:
                        d.verify_warmup(*references[instrumented], result, warm)
                        if arm == "current32":
                            observer_references[instrumented] = (result, warm)
                        else:
                            d.verify_warmup(*observer_references[instrumented], result, warm, observer=True)
                    traces[instrumented], finals[instrumented] = result, final
                self.assertEqual(traces[False]["trajectory_parameter_sha256"], traces[True]["trajectory_parameter_sha256"])
                self.assertTrue(d.equal_tree(finals[False], finals[True]))

    def test_full_cell_retains_grid_four_checkpoints_and_real_delivery(self):
        plan, data = tiny_fixture(6, 200)
        with contextlib.redirect_stdout(io.StringIO()):
            result, states, _, _ = d.train_cell(plan, data, .9, "lagged32_current_norm", 200, True, False,
                                               d.storage.Guard(False, "cpu"), {}, device="cpu", model_factory=tiny_model)
        self.assertEqual(set(states), set(d.CHECKPOINTS))
        self.assertEqual([v["step"] for v in result["validation_trajectory"]], [0, 100, 200])
        self.assertEqual(len(result["steps_raw"]), 200)
        self.assertIsNone(result["steps_raw"][99]["current_norm"])
        self.assertIsNotNone(result["steps_raw"][100]["current_norm"])
        self.assertEqual(result["steps_raw"][100]["policy"]["observing_step"], 101)
        self.assertEqual(result["final_training_clean"]["count"], 8)
        self.assertNotIn("test", result)

    def test_actual_delivery_corruption_rejected_even_logging_off(self):
        plan, data = tiny_fixture()
        original = d.h.set_grad
        def broken(model, value):
            original(model, value * 0)
        with mock.patch.object(d.h, "set_grad", side_effect=broken):
            with self.assertRaises(AssertionError):
                d.train_cell(plan, data, .9, "current32", 220, False, True,
                             d.storage.Guard(True, "cpu"), {}, device="cpu", model_factory=tiny_model)

    def test_full_completion_gate_checks_identity_states_before_loader(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = d.storage.Store("test", synthetic_parent=tmp)
            results, bindings = fixture_completion(store)
            loader = mock.Mock(return_value=("synthetic_images", "synthetic_labels"))
            record = {"all_training_completed_utc": "2026-01-01T00:00:00+00:00"}
            for bad_results, bad_bindings in ((results[:-1], bindings[:-1]),
                                             (results[:-1] + [results[0]], bindings)):
                with self.assertRaises(AssertionError):
                    d.load_test_with_gate(bad_results, bad_bindings, record, loader)
            loader.assert_not_called()
            self.assertEqual(d.load_test_with_gate(results, bindings, record, loader), ("synthetic_images", "synthetic_labels"))
            loader.assert_called_once()
            self.assertGreater(record["test_first_opened_utc"], record["all_training_completed_utc"])
            self.assertTrue(record["official_test_opened"])
            original = results[0]["checkpoint_sha256"]["final"]
            results[0]["checkpoint_sha256"]["final"] = "0" * 64
            with self.assertRaises(AssertionError):
                d.complete_training_gate(results, bindings)
            results[0]["checkpoint_sha256"]["final"] = original
            results[0]["checkpoint_steps"]["min_val_ce"] = 100
            with self.assertRaises(AssertionError):
                d.complete_training_gate(results, bindings)
            results[0]["checkpoint_steps"]["min_val_ce"] = 2000
            changed = [dict(row) for row in results[0]["validation_trajectory"]]
            changed[3]["cross_entropy"] = float("nan")
            results[0]["validation_trajectory"] = changed
            with self.assertRaises(AssertionError):
                d.complete_training_gate(results, bindings)

    def test_storage_exclusivity_escape_and_tensor_only_checkpoints(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = d.storage.Store("test", synthetic_parent=tmp)
            store.json("record.json", {"finite": 1.})
            with self.assertRaises(FileExistsError):
                store.json("record.json", {})
            with self.assertRaises(FileExistsError):
                store.json("../escape.json", {})
            with self.assertRaises(TypeError):
                store.checkpoints("bad.pt", {"state": {"rng": np.ones(3)}})
            with mock.patch.object(store, "bytes_used", return_value=d.storage.GIB):
                with self.assertRaises(RuntimeError):
                    store.json("too-much.json", {"x": 1})

    def test_failure_preserved_with_nonfinite_context_and_partial_states(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = d.storage.Store("test", synthetic_parent=tmp)
            metadata = d.storage.Metadata(Path(tmp) / "metadata")
            manifest = {"status": "training", "completed_cells": [{"run_key": "already-complete"}]}
            metadata.write("execution.json", manifest)
            context = {"step": 17, "phase": "synthetic_failure", "rejected": float("nan"),
                       "partial_rows": [{"step": 1}], "partial_validations": [],
                       "partial_checkpoint_tensors": {"min_val_ce": {"weight": torch.ones(1)}}}
            d.storage.preserve_failure(metadata, manifest, context, RuntimeError("synthetic"), store)
            saved = json.loads((metadata.root / "execution.json").read_text())
            self.assertEqual(saved["status"], "failed")
            self.assertEqual(saved["completed_cells"], [{"run_key": "already-complete"}])
            failure = json.loads((metadata.root / "failure.json").read_text())
            self.assertEqual(failure["partial_row_count"], 1)
            self.assertIn("partial_checkpoints", failure)
            self.assertEqual(failure["rejected"], {"rejected_nonfinite": "nan"})
            with self.assertRaises(FileExistsError):
                d.storage.Metadata(metadata.root)

    def test_resource_timeout_is_fatal(self):
        guard = d.storage.Guard(True, "cpu")
        guard.limit = -1
        with self.assertRaises(TimeoutError):
            guard.check()

    def test_full_launch_binds_pilot_manifest_not_last_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = d.storage.Store("test", synthetic_parent=tmp)
            training = [store.json(f"sentinel-{i}.json", {"synthetic": i}) for i in (0, 1)]
            plan = store.plan("development-plan.npz", {"seed": 9880})
            reports = [{"arm": arm, "trajectory_bitwise_identical": True,
                        "final_state_bitwise_identical": True, "warmup_checks_passed": True} for arm in d.ARMS]
            timing = store.json("timing-and-invariants.json", reports)
            payload = b"synthetic-scientific-source"
            digest = hashlib.sha256(payload).hexdigest()
            sources = {f"source-{i}.py": digest for i in range(14)}
            sources["spectral_filter.py"] = digest  # Exposes the original loop-variable shadowing bug.
            numerical = {"synthetic_environment": True}
            pilot = {"mode": "pilot", "status": "complete_passed", "completed_traces": 12,
                     "all_gates_passed": True, "warmup_checks_passed": True,
                     "official_test_opened": False, "validation_or_accuracy_computed": False,
                     "source_sha256": sources, "environment": numerical,
                     "training_data_artifacts": training, "plans": [plan],
                     "timing_and_invariants": timing, "repository_revision": "synthetic-revision"}
            metadata = d.storage.Metadata(Path(tmp) / "pilot")
            expected = metadata.write("execution.json", pilot)
            with mock.patch.object(d, "HERE", Path(tmp)), mock.patch.object(d.subprocess, "check_output", return_value=payload):
                self.assertEqual(d.full_launch_gate(sources, numerical, training), expected)
                with self.assertRaises(AssertionError):
                    d.full_launch_gate({**sources, "spectral_filter.py": "0" * 64}, numerical, training)
                with self.assertRaises(AssertionError):
                    d.full_launch_gate(sources, {"changed": True}, training)

    def test_occupancy_allows_own_pid_not_foreign_compute(self):
        gpu = "NVIDIA GeForce RTX 3090, 20000\n"
        own = f"{os.getpid()}, python3, 200 MiB\n1, /usr/libexec/gnome-remote-desktop-daemon, 100 MiB\n"
        with mock.patch.object(d.subprocess, "check_output", side_effect=[gpu, own]):
            self.assertEqual(d.occupancy_gate()["own_pid"], os.getpid())
        with mock.patch.object(d.subprocess, "check_output", side_effect=[gpu, "987654, python3, 200 MiB\n"]):
            with self.assertRaises(AssertionError):
                d.occupancy_gate()

    def test_missing_go_flags_exit_without_launch(self):
        for argv in (["harness", "--pilot"], ["harness", "--full"],
                     ["harness", "--pilot", "--development-go", "--confirmatory-go"]):
            with mock.patch.object(d.sys, "argv", argv), mock.patch.object(d, "launch") as launch:
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    d.main()
                launch.assert_not_called()


if __name__ == "__main__":
    unittest.main()
