"""Synthetic CPU tests for the narrow I10 branch adapter."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
import sys
import unittest

import numpy as np
import torch
from torch.nn import functional as F


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))

import branch_core as core


def fixture():
    model = core.i9.make_model(23, "cpu", input_dim=4, width=5, classes=3)
    optimizer = core.i9.make_optimizer(model)
    tracker = core.i9.make_tracker(model, optimizer)
    x = torch.arange(41 * 4, dtype=torch.float32).reshape(41, 4).remainder(19).div(19)
    clean = torch.arange(41, dtype=torch.long).remainder(3)
    fixed = (clean + (torch.arange(41) % 2)).remainder(3)
    data = {"x": x, "clean": clean, "noisy": fixed,
            "vx": x.flip(0).clone(), "vy": clean.flip(0).clone(),
            "ax": x.roll(7, 0).clone(), "ay": clean.roll(7, 0).clone()}
    n = core.i9.flat_params(model).numel()
    tracker.V = torch.eye(n)[:, :2].contiguous()
    tracker.S = torch.ones(2, dtype=torch.float64)
    tracker.grad_mean = torch.linspace(-0.1, 0.2, n)
    tracker.step_count = 100
    return model, optimizer, tracker, data, tracker.V.detach().clone()


class BranchCoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            if torch.get_num_interop_threads() != 1:
                raise
        expected = "9280c7d360aef4c775baf0d781a5afd56e016e9cfb93ef829c3ae4b8cc3df943"
        if hashlib.sha256((ROOT / "spectral_filter.py").read_bytes()).hexdigest() != expected:
            raise AssertionError("canonical spectral filter differs")

    def test_raw_is_plain_adamw_while_observer_advances(self):
        model, optimizer, tracker, data, basis = fixture()
        parent = core.i9.snapshot(model, optimizer, tracker)
        plain, plain_optimizer, _ = core.i9.restore(parent, "cpu")
        plain_optimizer.zero_grad(set_to_none=True)
        core.i9._objective_loss(plain, data["x"][:8], data["noisy"][:8]).backward()
        plain_optimizer.step()

        actual, actual_optimizer, actual_tracker = core.i9.restore(parent, "cpu")
        row = core.branch_step(actual, actual_optimizer, actual_tracker,
                               data["x"][:8], data["noisy"][:8], "raw", basis)
        self.assertEqual(row["observer"]["step_after"], 101)
        self.assertTrue(row["observer"]["used"])
        self.assertTrue(torch.equal(core.i9.flat_params(plain), core.i9.flat_params(actual)))
        self.assertTrue(core.i9.equal_tree(core.i9._cpu_clone(plain_optimizer.state_dict()),
                                           core.i9._cpu_clone(actual_optimizer.state_dict())))
        json.dumps(row, allow_nan=False)

    def test_current_delegates_exactly_to_i9(self):
        model, optimizer, tracker, data, basis = fixture()
        parent = core.i9.snapshot(model, optimizer, tracker)
        reference = core.i9.restore(parent, "cpu")
        actual = core.i9.restore(parent, "cpu")
        expected_row = core.i9.train_step(*reference, data["x"][:8], data["noisy"][:8], "current32")
        row = core.branch_step(*actual, data["x"][:8], data["noisy"][:8], "current32", basis)
        self.assertEqual(row["gradient"]["raw_norm"], expected_row["raw_norm"])
        self.assertEqual(row["gradient"]["applied_norm"], expected_row["native_norm"])
        self.assertTrue(core.i9.equal_tree(core.i9.snapshot(*reference),
                                           core.i9.snapshot(*actual)))

    def test_frozen_projects_exact_basis_and_never_mutates_observer(self):
        model, optimizer, tracker, data, _ = fixture()
        # Seed nonzero Adam moments; freeze the observer's resulting anchor basis.
        core.i9.train_step(model, optimizer, tracker, data["x"][:8], data["noisy"][:8], "raw")
        frozen = core.i9._basis(tracker)
        self.assertIsNotNone(frozen)
        parent = core.i9.snapshot(model, optimizer, tracker)
        reference_model, reference_optimizer, reference_tracker = core.i9.restore(parent, "cpu")
        reference_optimizer.zero_grad(set_to_none=True)
        objective = core.i9._objective_loss(reference_model, data["x"][8:16], data["clean"][8:16])
        objective.backward()
        raw = core.i9.flat_grad(reference_model)
        core.i9.set_grad(reference_model, frozen @ (frozen.T @ raw))
        reference_optimizer.step()

        actual_model, actual_optimizer, actual_tracker = core.i9.restore(parent, "cpu")
        tracker_before = parent["tracker"]
        row = core.branch_step(actual_model, actual_optimizer, actual_tracker,
                               data["x"][8:16], data["clean"][8:16], "frozen32", frozen)
        self.assertFalse(row["observer"]["used"])
        self.assertTrue(row["gradient_filter_applied"])
        self.assertEqual(row["observer"]["step_before"], row["observer"]["step_after"])
        self.assertTrue(torch.equal(core.i9.flat_params(reference_model),
                                    core.i9.flat_params(actual_model)))
        self.assertTrue(core.i9.equal_tree(tracker_before,
                                           core.i9.snapshot(actual_model, actual_optimizer,
                                                            actual_tracker)["tracker"]))
        # Adam moments and weight decay can deliver motion outside the gradient basis.
        self.assertGreater(row["displacement"]["total_leakage"]["frozen_basis"]
                           ["outside_squared_norm"], 0.0)

    def test_soft_target_branch_and_restored_moments(self):
        model, optimizer, tracker, data, basis = fixture()
        core.i9.train_step(model, optimizer, tracker, data["x"][:8], data["noisy"][:8], "raw")
        parent = core.i9.snapshot(model, optimizer, tracker)
        restored = core.i9.restore(parent, "cpu")
        self.assertTrue(core.i9.equal_tree(parent, core.i9.snapshot(*restored)))
        labels = data["clean"][8:16]
        soft = 0.1 * F.one_hot(labels, num_classes=3).to(torch.float64) + 0.9 / 3
        row = core.branch_step(*restored, data["x"][8:16], soft, "raw", basis)
        self.assertTrue(row["loss"] > 0)
        self.assertFalse(row["gradient_filter_applied"])

    def test_evaluation_is_neutral_and_chunk_mean_consistent(self):
        model, optimizer, tracker, data, _ = fixture()
        model.train(False)
        for parameter in model.parameters():
            parameter.grad = torch.arange(parameter.numel(), dtype=parameter.dtype).reshape_as(parameter)
        random.seed(7)
        np.random.seed(8)
        torch.manual_seed(9)
        before = core.i9.snapshot(model, optimizer, tracker)
        small = core.evaluate(model, data, chunk_size=7)
        self.assertTrue(core.i9.equal_tree(before, core.i9.snapshot(model, optimizer, tracker)))
        whole = core.evaluate(model, data, chunk_size=10_000)
        self.assertTrue(core.i9.equal_tree(before, core.i9.snapshot(model, optimizer, tracker)))
        for section in ("train", "auxiliary", "validation"):
            for key in small[section]:
                self.assertAlmostEqual(small[section][key], whole[section][key], places=14)
        self.assertAlmostEqual(small["train"]["fixed_minus_soft_ce"],
                               small["train"]["fixed_ce"] - small["train"]["soft_ce"],
                               places=15)
        self.assertIn("soft_ce", small["validation"])
        for section in ("train", "auxiliary", "validation"):
            self.assertIn("mean_max_probability", small[section])
            self.assertIn("mean_true_label_probability", small[section])
        json.dumps(small, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
