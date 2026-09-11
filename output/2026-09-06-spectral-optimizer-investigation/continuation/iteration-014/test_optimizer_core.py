"""Synthetic CPU tests for the I14 cross-optimizer core."""
from __future__ import annotations

from pathlib import Path
import sys
import unittest

import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import optimizer_core as core


def fixture(base: str, lr: float = 0.03):
    generator = torch.Generator().manual_seed(14014)
    x = torch.rand((64, 4), generator=generator)
    y = torch.randint(0, 3, (64,), generator=generator)
    model = core.i9.make_model(14014, "cpu", input_dim=4, width=5, classes=3)
    optimizer = core.make_optimizer(model, base, lr)
    tracker = core.i9.make_tracker(model, optimizer)
    return model, optimizer, tracker, x, y


def flat_momentum(model, optimizer):
    return torch.cat([optimizer.state[parameter]["momentum_buffer"].reshape(-1)
                      for parameter in model.parameters()]).clone()


class OptimizerCoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            if torch.get_num_interop_threads() != 1:
                raise

    def test_exact_optimizer_configs_and_manual_sgd_decay_formula(self):
        for base in core.BASES:
            model, optimizer, tracker, x, y = fixture(base)
            core._validate_optimizer(model, optimizer, base, .03)
            core._validate_tracker(model, optimizer, tracker)
            group = optimizer.param_groups[0]
            if base in ("sgd", "sgdm"):
                self.assertEqual(group["weight_decay"], 0)
                self.assertEqual(group["momentum"], 0 if base == "sgd" else .9)
            else:
                self.assertEqual(group["weight_decay"], .01)

        model, optimizer, tracker, x, y = fixture("sgd")
        before = core.i9.flat_params(model)
        row = core.train_step(model, optimizer, tracker, x, y, "raw", "sgd")
        gradient = core.i9.flat_grad(model)
        after = core.i9.flat_params(model)
        expected = before * (1 - .03 * core.WD) - .03 * gradient
        self.assertTrue(torch.allclose(after, expected, rtol=1e-6, atol=2e-8))
        nominal = -.03 * core.WD * before
        data = after - before - nominal
        self.assertAlmostEqual(row["displacement"]["nominal_decay_norm"],
                               float(torch.linalg.vector_norm(nominal.double())), places=12)
        self.assertAlmostEqual(row["displacement"]["data_norm"],
                               float(torch.linalg.vector_norm(data.double())), places=12)
        self.assertTrue(row["decay"]["manual_before_optimizer_step"])
        self.assertLess(row["decay"]["manual_minus_nominal_norm"], 1e-6)

    def test_sgdm_first_buffer_and_multiple_step_recurrence(self):
        model, optimizer, tracker, x, y = fixture("sgdm")
        before = core.i9.flat_params(model)
        core.train_step(model, optimizer, tracker, x, y, "raw", "sgdm")
        gradient1 = core.i9.flat_grad(model)
        buffer1 = flat_momentum(model, optimizer)
        after1 = core.i9.flat_params(model)
        self.assertTrue(torch.equal(buffer1, gradient1))
        self.assertTrue(torch.allclose(after1, before * (1 - .03 * core.WD) - .03 * gradient1,
                                       rtol=1e-6, atol=2e-8))

        before2 = after1.clone()
        core.train_step(model, optimizer, tracker, x.flip(0), y.flip(0), "raw", "sgdm")
        gradient2 = core.i9.flat_grad(model)
        buffer2 = flat_momentum(model, optimizer)
        expected_buffer = .9 * buffer1 + gradient2
        expected_after = before2 * (1 - .03 * core.WD) - .03 * expected_buffer
        self.assertTrue(torch.allclose(buffer2, expected_buffer, rtol=1e-6, atol=2e-8))
        self.assertTrue(torch.allclose(core.i9.flat_params(model), expected_after,
                                       rtol=1e-6, atol=2e-8))

    def test_raw_native_are_exact_through_warmup_and_plain_sgd_confines_step(self):
        left = fixture("sgd", .02)
        right = fixture("sgd", .02)
        model_l, optimizer_l, tracker_l, x, y = left
        model_r, optimizer_r, tracker_r, _, _ = right
        for _ in range(100):
            raw = core.train_step(model_l, optimizer_l, tracker_l, x, y, "raw", "sgd")
            native = core.train_step(model_r, optimizer_r, tracker_r, x, y, "current32", "sgd")
            self.assertFalse(raw["observer"]["filtering_active"])
            self.assertFalse(native["observer"]["filtering_active"])
        self.assertTrue(core.i9.equal_tree(
            core.snapshot(model_l, optimizer_l, tracker_l, "sgd", .02),
            core.snapshot(model_r, optimizer_r, tracker_r, "sgd", .02)))

        row = core.train_step(model_r, optimizer_r, tracker_r, x, y, "current32", "sgd")
        self.assertTrue(row["observer"]["filtering_active"])
        leakage = row["displacement"]["data_current_basis_leakage"]
        self.assertIsNone(leakage["reason"])
        self.assertLess(leakage["fraction"], 2e-8)

    def test_snapshot_roundtrip_and_one_step_continuation_every_base(self):
        for base in core.BASES:
            model, optimizer, tracker, x, y = fixture(base)
            for offset in range(3):
                core.train_step(model, optimizer, tracker, x.roll(offset, 0), y.roll(offset, 0),
                                "raw", base)
            saved = core.snapshot(model, optimizer, tracker, base, .03)
            clone_model, clone_optimizer, clone_tracker = core.restore(saved, "cpu")
            self.assertTrue(core.i9.equal_tree(
                saved, core.snapshot(clone_model, clone_optimizer, clone_tracker, base, .03)))
            core.train_step(model, optimizer, tracker, x, y, "current32", base)
            expected = core.snapshot(model, optimizer, tracker, base, .03)
            core.train_step(clone_model, clone_optimizer, clone_tracker, x, y, "current32", base)
            actual = core.snapshot(clone_model, clone_optimizer, clone_tracker, base, .03)
            self.assertTrue(core.i9.equal_tree(expected, actual), base)

    def test_typed_nonfinite_and_structural_or_generic_failures(self):
        model, optimizer, tracker, x, y = fixture("sgd")
        bad_x = x.clone()
        bad_x[0, 0] = float("nan")
        with self.assertRaisesRegex(core.NumericalFailure, "objective loss"):
            core.train_step(model, optimizer, tracker, bad_x, y, "raw", "sgd")

        model, optimizer, tracker, x, y = fixture("sgd")
        hook = next(model.parameters()).register_hook(
            lambda gradient: gradient * torch.tensor(float("nan")))
        with self.assertRaisesRegex(core.NumericalFailure, "gradient"):
            core.train_step(model, optimizer, tracker, x, y, "raw", "sgd")
        hook.remove()

        model, optimizer, tracker, x, y = fixture("adamw")
        with torch.no_grad():
            next(model.parameters()).view(-1)[0] = float("nan")
        with self.assertRaisesRegex(core.NumericalFailure, "nonfinite"):
            core.train_step(model, optimizer, tracker, x, y, "raw", "adamw")
        failed = core.snapshot(model, optimizer, tracker, "adamw", .03)
        self.assertTrue(torch.isnan(next(iter(failed["model_state"].values()))).any())

        model, optimizer, tracker, x, y = fixture("sgd")
        with self.assertRaises(core.OptimizerCoreError):
            core.train_step(model, optimizer, tracker, x, y, "unknown", "sgd")
        with self.assertRaises(core.OptimizerCoreError):
            core.train_step(model, optimizer, tracker, x, y, "raw", "sgdm")
        saved = core.snapshot(model, optimizer, tracker, "sgd", .03)
        saved["parameter_topology"][0]["shape"] = [999]
        with self.assertRaisesRegex(core.OptimizerCoreError, "parameter topology"):
            core.restore(saved, "cpu")

        model, optimizer, tracker, _, _ = fixture("sgd")
        saved = core.snapshot(model, optimizer, tracker, "sgd", .03)
        saved["optimizer"]["param_groups"][0]["params"] = list(reversed(
            saved["optimizer"]["param_groups"][0]["params"]))
        with self.assertRaisesRegex(core.OptimizerCoreError, "IDs/order"):
            core.restore(saved, "cpu")

        model, optimizer, tracker, x, y = fixture("sgd")
        tracker.filter_grad = lambda: (_ for _ in ()).throw(RuntimeError("synthetic eigensolver"))
        with self.assertRaisesRegex(RuntimeError, "synthetic eigensolver"):
            core.train_step(model, optimizer, tracker, x, y, "raw", "sgd")

        model, optimizer, tracker, x, y = fixture("adamw")
        core.train_step(model, optimizer, tracker, x, y, "raw", "adamw")
        optimizer.state[next(model.parameters())]["exp_avg"].view(-1)[0] = float("inf")
        with self.assertRaisesRegex(core.NumericalFailure, "nonfinite"):
            core.train_step(model, optimizer, tracker, x, y, "raw", "adamw")


if __name__ == "__main__":
    unittest.main()
