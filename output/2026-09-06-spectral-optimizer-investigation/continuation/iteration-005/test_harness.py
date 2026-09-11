#!/usr/bin/env python3
"""Synthetic CPU tests only. No MNIST loading, replay, GPU or worst-size pilot."""
from pathlib import Path
import json
import math
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch
from torch import nn
import artifact_store as disk
import reference_math as r
import replay_harness as launch


class ReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)

    def test_weights_delayed_start_and_startup_overweight(self):
        innovations = torch.zeros(8, 5, dtype=torch.float32)
        innovations[3:] = torch.arange(25).reshape(5, 5) / 10
        for beta in (.9, .99, .999):
            x, weights = r.weighted_columns(innovations, 4, beta)
            direct = torch.outer(innovations[3].double(), innovations[3].double())
            for row in innovations[4:]:
                direct = beta * direct + (1 - beta) * torch.outer(row.double(), row.double())
            self.assertTrue(torch.allclose(x @ x.T, direct, atol=1e-12, rtol=1e-12))
            self.assertAlmostEqual(float(weights.sum()), 1., places=12)
            self.assertEqual(float(weights[0]), beta ** 4)
            regular = weights.clone()
            regular[0] *= 1 - beta
            self.assertGreater(abs(float(weights[0] - regular[0])), .5)
        with self.assertRaises(AssertionError):
            r.weighted_columns(innovations, 5)

    def test_dense_dual_orientation_and_residuals(self):
        generator = torch.Generator().manual_seed(43)
        innovations = torch.randn((22, 8), generator=generator).float()
        innovations[0] = 0
        for beta in (.9, .99, .999):
            x, _ = r.weighted_columns(innovations, 2, beta)
            ref, tensors = r.reference(x, 2, rank=3)
            dense_vals, dense_vecs = torch.linalg.eigh(x @ x.T)
            self.assertTrue(torch.allclose(tensors["eigenvalues"][:8], dense_vals.flip(0), atol=1e-12, rtol=1e-10))
            self.assertTrue(torch.allclose(tensors["mapped_vectors"][:, :3] @ tensors["mapped_vectors"][:, :3].T,
                                           dense_vecs[:, -3:] @ dense_vecs[:, -3:].T, atol=1e-10, rtol=1e-10))
            self.assertTrue(ref["diagnostics"]["reference_projector_valid"])

    def test_tie_preserves_energy_but_nulls_projector(self):
        x = torch.diag(torch.tensor([3., 2., 2., 1.], dtype=torch.float64))
        ref, tensors = r.reference(x, 1, rank=2)
        self.assertEqual(ref["metrics"]["optimal_rank32_energy"], 13.)
        self.assertFalse(ref["diagnostics"]["reference_projector_valid"])
        self.assertIsNone(tensors["projector_basis"])
        probes = self.probes(4)
        state = {"V": torch.eye(4, dtype=torch.float32)[:, :2], "S": torch.tensor([3., 2.], dtype=torch.float64)}
        row = r.observer_metrics(x, ref, tensors, state, state["V"], probes, probes["noisy"], 2, rank=2)
        self.assertAlmostEqual(row["metrics"]["span_energy_fraction"], 1.)
        self.assertIsNone(row["metrics"]["span_projector_distance"])
        self.assertIsNotNone(row["metrics"]["native_clean_retention"])
        r.add_reference_probes(ref, tensors, probes)
        self.assertIsNone(ref["metrics"]["clean_retention"])

    def test_near_tie_gate_does_not_drop_primary(self):
        x = torch.diag(torch.tensor([3., 2. + 1e-8, 2., 1.], dtype=torch.float64))
        ref, tensors = r.reference(x, 1, rank=2)
        self.assertLess(ref["diagnostics"]["boundary_relative_gap"], 1e-6)
        self.assertGreater(ref["metrics"]["optimal_rank32_energy"], 0)

    def test_zero_rank_deficiency_and_native_identity(self):
        innovation = torch.zeros((7, 4), dtype=torch.float32)
        x, _ = r.weighted_columns(innovation, None)
        ref, tensors = r.reference(x, None, rank=2)
        probes = self.probes(4)
        row = r.observer_metrics(x, ref, tensors, {"V": None, "S": None}, None, probes, probes["noisy"], 2, rank=2)
        self.assertEqual(row["metrics"]["native_clean_retention"], 1.)
        self.assertEqual(row["metrics"]["self_inclusion_retention_increment"], 0.)
        self.assertIsNone(row["metrics"]["span_energy_fraction"])
        self.assertIsNone(row["metrics"]["relative_covariance_error"])
        self.assertEqual(row["metrics"]["covariance_estimator_trace"], 0.)
        x = torch.diag(torch.tensor([2., 0., 0., 0.], dtype=torch.float64))
        ref, tensors = r.reference(x, 1, rank=2)
        self.assertEqual(ref["diagnostics"]["positive_rank"], 1)
        self.assertIsNone(tensors["projector_basis"])
        self.assertEqual(ref["metrics"]["optimal_rank32_energy"], 4.)

    def test_nonorthogonal_representation_matches_dense(self):
        x = torch.tensor([[2., 0., .2], [0., 1., 0.], [.1, 0., .3], [0., .2, 0.]], dtype=torch.float64)
        ref, tensors = r.reference(x, 1, rank=2)
        v = torch.eye(4, dtype=torch.float32)[:, :2]
        v[0, 0] *= 1.0005
        v[0, 1] += .0003
        singular = torch.tensor([1.3, .7], dtype=torch.float64)
        probes = self.probes(4)
        row = r.observer_metrics(x, ref, tensors, {"V": v, "S": singular}, v, probes, probes["noisy"], 2, rank=2)
        c = x @ x.T
        chat = (v.double() * singular.square()) @ v.double().T
        p = v.double() @ v.double().T
        self.assertAlmostEqual(row["metrics"]["relative_covariance_error"], r.norm(c - chat) / r.norm(c), places=12)
        self.assertAlmostEqual(row["energies"]["represented_operator_output_energy"], float(torch.trace(p @ c @ p)), places=12)
        self.assertAlmostEqual(row["metrics"]["trace_P_C"], float(torch.trace(p @ c)), places=12)
        self.assertNotAlmostEqual(float(torch.trace(p @ c @ p)), float(torch.trace(p @ c)), places=6)

    def test_cancellation_bound_and_failure(self):
        self.assertEqual(r.cancellation_square(1., 1., 1. + 1e-13)[0], 0.)
        self.assertLess(r.cancellation_square(1., 1., 1. + 1e-13)[1], 0.)
        with self.assertRaises(AssertionError):
            r.cancellation_square(1., 1., 1.01)

    def test_probe_cross_terms_zero_handling_and_self_inclusion(self):
        probes = self.probes(4)
        row = r.probe_metrics(probes, torch.eye(4, dtype=torch.float32)[:, :2])
        self.assertLess(row["energies"]["joint"]["energy_closure_relative_error"], 1e-5)
        zero = {name: torch.zeros(4) for name in r.PROBES}
        empty = r.probe_metrics(zero, None)
        self.assertIsNone(empty["metrics"]["native_clean_retention"])
        self.assertIsNone(empty["metrics"]["native_clean_corruption_cosine"])
        self.assertEqual(set(k for k, v in empty["metrics"].items() if v is None), set(empty["null_reasons"]))
        x = torch.diag(torch.tensor([3., 2., 1., .5], dtype=torch.float64))
        ref, tensors = r.reference(x, 1, rank=2)
        basis = torch.eye(4, dtype=torch.float32)[:, :2]
        result = r.observer_metrics(x, ref, tensors, {"V": basis, "S": torch.ones(2).double()},
                                    None, probes, torch.tensor([0., 0., 1., 0.]), 2, rank=2)
        self.assertEqual(result["metrics"]["self_inclusion_retention_increment"], -1.)

    def test_nonfinite_and_bad_orthogonality_rejected(self):
        bad = torch.eye(4, dtype=torch.float64)
        bad[0, 0] = float("nan")
        with self.assertRaises(AssertionError):
            r.reference(bad, 1, rank=2)
        probes = self.probes(4)
        probes["clean"][0] = float("nan")
        with self.assertRaises(AssertionError):
            r.probe_metrics(probes, None)
        x = torch.eye(4, dtype=torch.float64)
        ref, tensors = r.reference(x, 1, rank=2)
        with self.assertRaises(AssertionError):
            r.observer_metrics(x, ref, tensors, {"V": torch.ones(4, 2), "S": torch.ones(2)},
                               None, self.probes(4), torch.ones(4), 2, rank=2)

    @staticmethod
    def probes(dimension):
        clean = torch.linspace(.1, 1., dimension).float()
        noisy = clean.flip(0) * .4
        return {"clean": clean, "noisy": noisy, "corruption_residual": noisy - clean,
                "auxiliary_clean": clean.roll(1)}


class ReplayAndStorageTests(unittest.TestCase):
    def setUp(self):
        self.cuda = mock.patch.object(torch.cuda, "is_available", return_value=False)
        self.cuda.start()
        launch.h.configure()

    def tearDown(self):
        self.cuda.stop()

    def test_canonical_mean_and_observer_passivity(self):
        model = nn.Linear(3, 2)
        optimizer = launch.h.make_optimizer(model)
        trackers = {width: launch.h.make_tracker(model, optimizer, width) for width in (2, 4)}
        mean, first, expected_mean = None, None, None
        generator = torch.Generator().manual_seed(12)
        for step in range(1, 12):
            raw = torch.ones(8) if step <= 3 else torch.randn(8, generator=generator)
            launch.h.set_grad(model, raw.clone())
            before = launch.h.snapshot(model, optimizer, None)
            mean, innovation, first = launch.observe_common(trackers, raw, mean, first, step)
            if expected_mean is None:
                expected_mean = raw.clone()
            else:
                expected_mean.mul_(.99).add_(raw, alpha=1 - .99)
            self.assertTrue(torch.equal(mean, expected_mean))
            self.assertTrue(torch.equal(innovation, raw - expected_mean))
            self.assertTrue(launch.h.equal_tree(before, launch.h.snapshot(model, optimizer, None)))
        self.assertEqual(first, 4)

    def test_small_cpu_training_matches_baseline_core_and_actual_gradient(self):
        generator = torch.Generator().manual_seed(11)
        x, y = torch.randn(8, 4, generator=generator), torch.arange(8) % 2
        traces, finals = [], []
        for measured in (False, True):
            torch.manual_seed(18)
            model = nn.Linear(4, 2)
            optimizer = launch.h.make_optimizer(model)
            observers = {width: launch.h.make_tracker(model, optimizer, width) for width in (2, 4)} if measured else {}
            mean, first, hashes = None, None, []
            for step in range(1, 9):
                optimizer.zero_grad(set_to_none=True)
                launch.h.F.cross_entropy(model(x[step % 4:step % 4 + 4]), y[step % 4:step % 4 + 4]).backward()
                raw = launch.h.flat_grad(model)
                if measured:
                    mean, _, first = launch.observe_common(observers, raw, mean, first, step)
                else:
                    # Historical baseline assigns a cloned flattened raw gradient.
                    launch.h.set_grad(model, raw.clone())
                self.assertTrue(torch.equal(raw, launch.h.flat_grad(model)))
                optimizer.step()
                hashes.append(launch.h.tensor_hash(launch.h.flat_params(model)))
            traces.append(hashes)
            finals.append(launch.core_snapshot(model, optimizer))
        self.assertEqual(traces[0], traces[1])
        self.assertTrue(launch.h.equal_tree(finals[0], finals[1]))

    def test_all_zero_observation_and_underflow_convention(self):
        model = nn.Linear(3, 2)
        optimizer = launch.h.make_optimizer(model)
        trackers = {width: launch.h.make_tracker(model, optimizer, width) for width in (2, 4)}
        mean, first = None, None
        for step in range(1, 4):
            mean, c, first = launch.observe_common(trackers, torch.zeros(8), mean, first, step)
        self.assertIsNone(first)
        with self.assertRaises(AssertionError):
            launch.observe_common(trackers, torch.full((8,), 1e-30), mean, first, 4)

    def test_nonfinite_startup_gradient_rejected_before_observer_update(self):
        model = nn.Linear(3, 2)
        optimizer = launch.h.make_optimizer(model)
        tracker = launch.h.make_tracker(model, optimizer, 2)
        with self.assertRaisesRegex(AssertionError, "Non-finite"):
            launch.observe_common({2: tracker}, torch.full((8,), float("nan")), None, None, 1)
        self.assertEqual(tracker.step_count, 0)
        self.assertIsNone(tracker.grad_mean)

    def test_established_basis_reset_is_a_failure(self):
        model = nn.Linear(3, 2)
        optimizer = launch.h.make_optimizer(model)
        tracker = launch.h.make_tracker(model, optimizer, 2)
        mean, _, first = launch.observe_common({2: tracker}, torch.zeros(8), None, None, 1)
        mean, _, first = launch.observe_common({2: tracker}, torch.ones(8), mean, first, 2)
        raw = torch.ones(8)
        expected = mean.clone().mul_(.99).add_(raw, alpha=1 - .99)
        def reset(_):
            tracker.grad_mean = expected
            tracker.V = None
        with mock.patch.object(tracker, "_update_svd", side_effect=reset):
            with self.assertRaisesRegex(AssertionError, "reset"):
                launch.observe_common({2: tracker}, raw, mean, first, 3)

    def test_probe_and_omitted_evaluation_preserve_complete_state(self):
        model = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 2))
        optimizer = launch.h.make_optimizer(model)
        data = {"x": torch.randn(10, 4), "clean": torch.arange(10) % 2,
                "noisy": (torch.arange(10) + 1) % 2, "ax": torch.randn(10, 4), "ay": torch.arange(10) % 2}
        launch.h.F.cross_entropy(model(data["x"]), data["noisy"]).backward()
        trackers = {width: launch.h.make_tracker(model, optimizer, width) for width in (2, 4)}
        before = launch.complete_snapshot(model, optimizer, trackers)
        probes = launch.make_probes(model, optimizer, trackers, data, torch.arange(5), torch.arange(5, 10))
        self.assertEqual(set(probes), set(r.PROBES))
        launch.h.evaluate(model, data["ax"], data["ay"])
        self.assertTrue(launch.h.equal_tree(before, launch.complete_snapshot(model, optimizer, trackers)))
        np.random.random()
        self.assertFalse(launch.h.equal_tree(before, launch.complete_snapshot(model, optimizer, trackers)))

    def test_new_core_snapshot_captures_numpy_but_old_warmup_hash_is_unchanged(self):
        model = nn.Linear(4, 2)
        optimizer = launch.h.make_optimizer(model)
        before = launch.core_snapshot(model, optimizer)
        historical_before = launch.old.tree_hash(launch.old.warmup_snapshot(model, optimizer, None)["core"])
        np.random.random()
        self.assertFalse(launch.h.equal_tree(before, launch.core_snapshot(model, optimizer)))
        historical_after = launch.old.tree_hash(launch.old.warmup_snapshot(model, optimizer, None)["core"])
        self.assertEqual(historical_before, historical_after)

    def test_checkpoint_clone_and_all_named_anchors(self):
        model = nn.Linear(3, 2)
        saved = launch.old.cpu_tree(model.state_dict())
        prior = launch.old.tree_hash(saved)
        with torch.no_grad():
            model.weight.add_(1)
        self.assertEqual(launch.old.tree_hash(saved), prior)
        historical = {"checkpoint_steps": {"final": 7, "warmup100": 7}}
        checkpoints = {"final": saved, "warmup100": saved}
        with self.assertRaises(AssertionError):
            launch.checkpoint_gate(model, 7, historical, checkpoints)
        model.load_state_dict(saved)
        self.assertEqual(len(launch.checkpoint_gate(model, 7, historical, checkpoints)), 2)

    def test_large_volume_path_guard_and_finite_failure_serialization(self):
        with self.assertRaises(RuntimeError):
            disk.verify_large_mount(launch.HERE)
        # Tiny synthetic artifacts still use the real verified large volume.
        store = disk.Store("cpu-unit-test")
        stream = store.stream("tiny.npy", (3, 2))
        store.append("tiny.npy", 0, np.array([1., 2.], dtype=np.float32))
        store.flush()
        artifact = store.stream_artifacts()[0]
        self.assertEqual(artifact["completed_rows"], 1)
        self.assertEqual(artifact["shape"], [3, 2])
        with self.assertRaises(RuntimeError):
            store.stream("tiny.npy", (3, 2))
        with self.assertRaises(RuntimeError):
            store.path("../../escape.npy")
        state = {"parameters": torch.arange(4).float(), "numpy_rng": torch.arange(4).to(torch.uint32)}
        bundle = store.tensor_bundle("tiny-state.pt", state)
        loaded = torch.load(bundle["path"], map_location="cpu", weights_only=True)
        self.assertTrue(launch.h.equal_tree(state, loaded))
        self.assertEqual(bundle["tensor_inventory"]["root.numpy_rng"]["dtype"], "torch.uint32")
        output = Path(tempfile.mkdtemp(prefix="metadata-", dir=store.root))
        manifest = {"status": "replay"}
        disk.preserve_failure(output, manifest, {"step": 2, "rejected": float("nan")}, ValueError("bad finite gate"), store)
        result = json.loads((output / "failure.json").read_text())
        self.assertEqual(result["completed_stream_rows"]["tiny.npy"], 1)
        self.assertEqual(result["context"]["rejected"], {"nonfinite_rejected": "nan"})
        self.assertEqual(json.loads((output / "execution.json").read_text())["status"], "failed")

    def test_launch_cli_does_not_default_to_execution(self):
        with mock.patch("sys.argv", ["replay_harness.py"]):
            with self.assertRaises(SystemExit):
                launch.main()
        with mock.patch("sys.argv", ["replay_harness.py", "--full"]):
            with self.assertRaises(SystemExit):
                launch.main()

    def test_gpu_process_gate_excludes_only_own_python_pid(self):
        launch.check_compute_processes("123, python3\n77, stremio\n", own_pid=123)
        launch.check_compute_processes("", own_pid=123)
        with self.assertRaisesRegex(AssertionError, "PID 124"):
            launch.check_compute_processes("123, python3\n124, python3\n", own_pid=123)

    def test_full_launch_rejects_changed_pilot_sources(self):
        pilot = {"status": "complete", "mode": "pilot", "completed_development_traces": 2,
                 "all_gates_passed": True, "worst_size_resource_check_passed": True,
                 "source_sha256": {"one": "original"}}
        with mock.patch.object(launch, "source_hashes", return_value={"one": "original"}):
            with mock.patch.object(Path, "read_text", return_value=json.dumps(pilot)):
                self.assertEqual(launch.full_gate()["status"], "complete")
            pilot["source_sha256"] = {"one": "changed"}
            with mock.patch.object(Path, "read_text", return_value=json.dumps(pilot)):
                with self.assertRaisesRegex(AssertionError, "Sources changed"):
                    launch.full_gate()
            pilot["source_sha256"], pilot["all_gates_passed"] = {"one": "original"}, False
            with mock.patch.object(Path, "read_text", return_value=json.dumps(pilot)):
                with self.assertRaisesRegex(AssertionError, "every required gate"):
                    launch.full_gate()

    def test_elapsed_cap_is_enforced(self):
        with mock.patch.object(disk.time, "perf_counter", side_effect=[0., 601.]):
            guard = disk.Guard(True)
            with self.assertRaises(TimeoutError):
                guard.check()


if __name__ == "__main__":
    unittest.main()
