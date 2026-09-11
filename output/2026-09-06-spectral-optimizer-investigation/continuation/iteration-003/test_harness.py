#!/usr/bin/env python3
"""CPU unit tests; no MNIST access and no confirmatory training."""
import unittest
import json
from pathlib import Path
import tempfile
import numpy as np
import torch
from torch import nn
import neural_harness as h


class HarnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        h.configure()

    def test_warmup_width_and_projection_rank_are_distinct(self):
        torch.manual_seed(12)
        model = nn.Linear(8, 8).double()
        optimizer = h.make_optimizer(model)
        tracker = h.make_tracker(model, optimizer, 40)
        for _ in range(110):
            raw = torch.randn(72, dtype=torch.float64)
            applied, basis = h.observe_and_project(tracker, raw)
            if tracker.step_count <= 100:
                self.assertIsNone(basis)
                self.assertTrue(torch.equal(raw, applied))
            else:
                self.assertEqual(basis.shape, (72, 32))
                self.assertEqual(tracker.V.shape, (72, 40))
                self.assertTrue(torch.allclose(applied, basis @ (basis.T @ raw)))

    def test_width32_matches_canonical_filter(self):
        torch.manual_seed(21)
        first, second = nn.Linear(8, 8).double(), nn.Linear(8, 8).double()
        first_opt, second_opt = h.make_optimizer(first), h.make_optimizer(second)
        a, b = h.make_tracker(first, first_opt, 32), h.make_tracker(second, second_opt, 32)
        for _ in range(110):
            raw = torch.randn(72, dtype=torch.float64)
            applied, _ = h.observe_and_project(a, raw)
            h.set_grad(second, raw.clone())
            b.filter_grad()
            self.assertTrue(torch.equal(applied, h.flat_grad(second)))
            self.assertTrue(h.equal_tree(h.tracker_state(a), h.tracker_state(b)))

    def test_probe_state_and_corruption_decomposition(self):
        torch.manual_seed(9)
        model = nn.Linear(4, 3).double()
        optimizer = h.make_optimizer(model)
        tracker = h.make_tracker(model, optimizer, 2)
        data = {"x": torch.randn(20, 4, dtype=torch.float64), "clean": torch.arange(20) % 3,
                "noisy": (torch.arange(20) + 1) % 3,
                "ax": torch.randn(20, 4, dtype=torch.float64), "ay": torch.arange(20) % 3}
        optimizer.zero_grad()
        torch.nn.functional.cross_entropy(model(data["x"]), data["noisy"]).backward()
        h.observe_and_project(tracker, h.flat_grad(model))
        optimizer.step()
        before = h.snapshot(model, optimizer, tracker)
        probes = h.independent_probes(model, optimizer, tracker, data, torch.arange(10), torch.arange(10))
        self.assertTrue(h.equal_tree(before, h.snapshot(model, optimizer, tracker)))
        self.assertTrue(torch.allclose(probes["noisy"], probes["clean"] + probes["corruption_residual"]))
        basis = torch.linalg.qr(torch.randn(15, 2, dtype=torch.float64)).Q
        joint = h.joint_probe_geometry(probes, basis)
        self.assertLess(abs(joint["energy_closure_residual"]), 1e-12)
        self.assertLess(abs(joint["projected_energy_closure_residual"]), 1e-12)
        data["noisy"] = data["clean"]
        clean_probes = h.independent_probes(model, optimizer, tracker, data, torch.arange(10), torch.arange(10))
        self.assertEqual(float(clean_probes["corruption_residual"].norm()), 0.)
        self.assertIsNone(h.retention(clean_probes["corruption_residual"], None)["squared_norm_retention"])

    def test_projection_identity_and_harmful_leakage(self):
        g = torch.tensor([1., 1.], dtype=torch.float64)
        basis = torch.tensor([[1.], [0.]], dtype=torch.float64)
        applied = h.project(g, basis)
        displacement = torch.tensor([-.1, .5], dtype=torch.float64)
        metrics = h.displacement_metrics(g, applied, displacement, basis)
        self.assertAlmostEqual(metrics["identity_closure_residual"], 0.)
        self.assertEqual(metrics["raw_gradient_dot_update"]["sign"], 1)
        self.assertEqual(metrics["in_subspace_contribution"]["sign"], -1)
        self.assertEqual(metrics["outside_contribution"]["sign"], 1)
        baseline = h.displacement_metrics(g, g, displacement, None)
        self.assertEqual(baseline["leakage_squared_fraction"], 0.)
        self.assertEqual(baseline["raw_gradient_dot_update"]["sign"], 1)

    def test_decay_subtraction(self):
        theta = torch.tensor([1., -2.], dtype=torch.float64)
        data_step = torch.tensor([.2, .3], dtype=torch.float64)
        total = data_step - h.LR * h.WD * theta
        self.assertTrue(torch.allclose(h.decay_subtracted_update(total, theta), data_step, atol=1e-15, rtol=0))

    def test_checkpoint_tie(self):
        self.assertTrue(h.choose_checkpoint(1., .9))
        self.assertFalse(h.choose_checkpoint(1., 1.))
        self.assertFalse(h.choose_checkpoint(1., 1.1))

    def test_stream_pairing_and_window_counts(self):
        short, long = h.make_plan(9876, 180), h.make_plan(9876, 2000)
        for key in ("training_batches", "primary_probe_batches", "auxiliary_probe_batches"):
            self.assertTrue(np.array_equal(short[key], long[key][:180]))
        self.assertFalse(np.array_equal(short["training_batches"][0], short["primary_probe_batches"][0, :64]))
        self.assertEqual(len(set(short["train_indices"]) & set(short["validation_indices"])), 0)
        self.assertEqual(len(set(short["train_indices"]) & set(short["auxiliary_indices"])), 0)
        self.assertEqual(sum(h.is_probe_step(s) for s in range(101, 2001)), 39)
        self.assertEqual(sum(h.is_probe_step(s) for s in range(101, 501)), 9)
        self.assertEqual(sum(h.is_probe_step(s) for s in range(1501, 2001)), 10)
        self.assertEqual(h.finite_average([None, 1., 0.])["mean"], .5)
        self.assertIsNone(h.finite_average([None])["mean"])

    def test_failure_context_preserves_partial_records(self):
        with tempfile.TemporaryDirectory(prefix="failure-test-", dir=h.HERE) as directory:
            output = Path(directory)
            record = {"status": "running"}
            context = {"seed": 9876, "arm": "test_arm", "step": 3, "phase": "update_measurements",
                       "steps_raw": [{"step": 1}, {"step": 2}], "probes_raw": [{"step": 1}],
                       "validation_trajectory": []}
            h.preserve_failure(output, record, context, RuntimeError("injected test failure"))
            saved = json.loads((output / "execution.json").read_text())
            partial = json.loads((output / "partial-failure.json").read_text())
            self.assertEqual(saved["status"], "failed_preserved")
            self.assertEqual(saved["failure_context"]["step"], 3)
            self.assertEqual(partial["steps_raw"], [{"step": 1}, {"step": 2}])
            self.assertEqual(h.file_hash(output / "partial-failure.json"), saved["partial_failure_artifact"]["sha256"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
