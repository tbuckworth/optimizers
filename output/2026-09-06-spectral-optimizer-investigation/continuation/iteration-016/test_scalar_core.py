"""Small CPU-only tests for the I16 scalar temporal controls."""
from __future__ import annotations

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import math
from pathlib import Path
import sys
import unittest

import torch


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import scalar_core as core


def _threads() -> None:
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        if torch.get_num_interop_threads() != 1:
            raise


class ScalarCoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _threads()

    def fixture(self):
        model = core.i9.make_model(1601, "cpu", input_dim=4, width=3, classes=2)
        optimizer = core.i14.make_optimizer(model, "sgdm", core.LR)
        tracker = core.i9.make_tracker(model, optimizer)
        count = sum(parameter.numel() for parameter in model.parameters())
        old = torch.linspace(-.37, .43, count, dtype=torch.float32)
        position = 0
        for parameter in model.parameters():
            size = parameter.numel()
            optimizer.state[parameter]["momentum_buffer"] = (
                old[position:position + size].reshape_as(parameter).clone())
            position += size
        tracker.step_count = 100
        tracker.grad_mean = torch.linspace(.21, -.17, count, dtype=torch.float32)
        basis = torch.zeros((count, 2), dtype=torch.float32)
        basis[0, 0] = 1.0
        basis[1, 1] = 1.0
        tracker.V = basis.contiguous()
        tracker.S = torch.ones(2, dtype=torch.float32)
        tracker.proj_k = None
        x = torch.tensor([
            [.2, -.1, .5, .7], [-.4, .3, .1, -.2],
            [.8, .2, -.6, .4], [.1, .9, -.3, -.5],
            [-.7, .6, .4, .2], [.3, -.8, .9, .1],
        ], dtype=torch.float32)
        target = torch.tensor([0, 1, 0, 1, 1, 0], dtype=torch.long)
        return model, optimizer, tracker, x, target

    def state(self):
        model, optimizer, tracker, x, target = self.fixture()
        state = core.i14.snapshot(model, optimizer, tracker, "sgdm", core.LR)
        return state, x, target

    def expected(self, state, x, target, k):
        model, optimizer, tracker = core.i14.restore(state, "cpu")
        old = core.i15._buffer(model, optimizer)
        pre_mean = tracker.grad_mean.detach().clone()
        optimizer.zero_grad(set_to_none=True)
        core.i9._objective_loss(model, x, target).backward()
        raw = core.i14._flat_grad(model)
        tracker.filter_grad()
        native = core.i14._flat_grad(model)
        post_mean = tracker.grad_mean.detach().clone()
        applied = post_mean.clone() if k == 0.0 else (
            raw.clone() if k == 1.0 else k * raw + (1.0 - k) * post_mean)
        scaled = torch.zeros_like(old) if k == 0.0 else (
            old.clone() if k == 1.0 else k * old)
        new = core.MOMENTUM * scaled + applied
        return old, pre_mean, raw, native, post_mean, applied, scaled, new

    def test_k1_is_exact_unchanged_i14_raw_sgdm_step(self) -> None:
        state, x, target = self.state()
        expected_live = core.i14.restore(state, "cpu")
        observed_live = core.i14.restore(state, "cpu")
        old_record = core.i14.train_step(*expected_live, x, target, "raw", "sgdm")
        record = core.scalar_step(*observed_live, x, target, 1.0, capture_digests=True)
        expected_state = core.i14.snapshot(*expected_live, "sgdm", core.LR)
        observed_state = core.i14.snapshot(*observed_live, "sgdm", core.LR)
        self.assertTrue(core.i9.equal_tree(expected_state, observed_state))
        self.assertEqual(record["loss"], old_record["loss"])
        self.assertEqual(record["displacement"]["data_norm"],
                         old_record["displacement"]["data_norm"])
        self.assertEqual(record["temporal_control"]["delivery_literal"], "raw_gradient")
        self.assertEqual(record["temporal_control"]["history_literal"], "old_buffer")

    def test_registered_k_values_match_independent_sgdm_formula(self) -> None:
        state, x, target = self.state()
        for k in core.REAL_K:
            with self.subTest(k=k):
                expected = self.expected(state, x, target, k)
                model, optimizer, tracker = core.i14.restore(state, "cpu")
                record = core.scalar_step(model, optimizer, tracker, x, target, k)
                actual = core.i15._buffer(model, optimizer)
                self.assertTrue(torch.allclose(actual, expected[-1], rtol=2e-6, atol=2e-7))
                self.assertEqual(record["observer"]["step_before"], 100)
                self.assertEqual(record["observer"]["step_after"], 101)
                self.assertFalse(record["gradient_filter_applied"])
                self.assertFalse(record["native_projection_used_for_delivery"])
                for residual in record["algebra_residuals"].values():
                    self.assertLessEqual(residual["norm"], residual["homogeneous_error_bound"])

    def test_k0_uses_literal_post_mean_and_erases_old_history(self) -> None:
        state, x, target = self.state()
        expected = self.expected(state, x, target, 0.0)
        self.assertFalse(torch.equal(expected[3], expected[4]))
        model, optimizer, tracker = core.i14.restore(state, "cpu")
        record = core.scalar_step(model, optimizer, tracker, x, target, 0.0,
                                  capture_digests=True)
        actual = core.i15._buffer(model, optimizer)
        self.assertTrue(torch.equal(actual, expected[4]))
        self.assertEqual(record["temporal_control"]["delivery_literal"], "post_mean")
        self.assertEqual(record["temporal_control"]["history_literal"], "zero")
        self.assertEqual(record["components"]["scaled_old_momentum_buffer"][
            "squared_energy"], 0.0)
        self.assertNotEqual(record["digests"]["applied_gradient"],
                            core.i9.tree_digest(expected[3]))

    def test_first_step_digests_are_deterministic_and_complete(self) -> None:
        state, x, target = self.state()
        records = []
        for _ in range(2):
            model, optimizer, tracker = core.i14.restore(state, "cpu")
            records.append(core.scalar_step(model, optimizer, tracker, x, target, .5,
                                             capture_digests=True))
        self.assertEqual(records[0]["digests"], records[1]["digests"])
        self.assertEqual(tuple(records[0]["digests"]), (
            "raw_gradient", "post_observer", "old_momentum_buffer",
            "applied_gradient", "new_momentum_buffer"))
        model, optimizer, tracker = core.i14.restore(state, "cpu")
        self.assertIsNone(core.scalar_step(model, optimizer, tracker, x, target, .5)["digests"])

    def test_soft_target_and_displacement_diagnostics(self) -> None:
        state, x, target = self.state()
        soft = .1 * torch.nn.functional.one_hot(target, num_classes=2).float() + .45
        model, optimizer, tracker = core.i14.restore(state, "cpu")
        record = core.scalar_step(model, optimizer, tracker, x, soft, .9)
        self.assertTrue(math.isfinite(record["loss"]))
        self.assertGreater(record["displacement"]["data_norm"], 0)
        self.assertTrue(math.isfinite(record["displacement"][
            "raw_gradient_dot_data_delta"]))
        self.assertEqual(set(record["components"]), {
            "raw_gradient", "post_mean", "native_projection_diagnostic",
            "applied_delivery", "old_momentum_buffer",
            "scaled_old_momentum_buffer", "new_momentum_buffer"})

    def test_parent_binding_and_configuration_rejections(self) -> None:
        state, x, target = self.state()
        binding = core.validate_parent_snapshot(state)
        self.assertEqual(binding["observer_step"], 100)
        bad = dict(state)
        bad["lr"] = .031
        with self.assertRaises((core.ScalarCoreError, core.i15.HistoryCoreError)):
            core.validate_parent_snapshot(bad)
        model, optimizer, tracker = core.i14.restore(state, "cpu")
        with self.assertRaisesRegex(core.ScalarCoreError, "exact registered"):
            core.scalar_step(model, optimizer, tracker, x, target, 0)
        model, optimizer, tracker = core.i14.restore(state, "cpu")
        del optimizer.state[next(iter(model.parameters()))]
        with self.assertRaises((core.ScalarCoreError, core.i15.HistoryCoreError,
                                core.i14.OptimizerCoreError)):
            core.scalar_step(model, optimizer, tracker, x, target, .5)
        model, optimizer, tracker = core.i14.restore(state, "cpu")
        optimizer.param_groups[0]["momentum"] = .8
        with self.assertRaises(core.i14.OptimizerCoreError):
            core.scalar_step(model, optimizer, tracker, x, target, .5)

    def test_nonfinite_state_is_typed_numerical_failure(self) -> None:
        state, x, target = self.state()
        model, optimizer, tracker = core.i14.restore(state, "cpu")
        with torch.no_grad():
            next(iter(model.parameters())).view(-1)[0] = float("inf")
        with self.assertRaises(core.NumericalFailure):
            core.scalar_step(model, optimizer, tracker, x, target, .5)


if __name__ == "__main__":
    unittest.main()
