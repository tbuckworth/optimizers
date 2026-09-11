"""Small CPU-only tests for the I15 SGDm history interventions."""
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
import history_core as core


def _threads() -> None:
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        if torch.get_num_interop_threads() != 1:
            raise


class HistoryCoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _threads()

    def fixture(self):
        model = core.i9.make_model(17, "cpu", input_dim=4, width=3, classes=2)
        optimizer = core.i14.make_optimizer(model, "sgdm", core.LR)
        tracker = core.i9.make_tracker(model, optimizer)
        size = sum(parameter.numel() for parameter in model.parameters())
        old = torch.linspace(-0.31, 0.47, size, dtype=torch.float32)
        position = 0
        for parameter in model.parameters():
            count = parameter.numel()
            optimizer.state[parameter]["momentum_buffer"] = (
                old[position:position + count].reshape_as(parameter).clone())
            position += count
        tracker.step_count = 100
        tracker.grad_mean = torch.linspace(0.19, -0.23, size, dtype=torch.float32)
        first = torch.zeros(size, dtype=torch.float32)
        first[0] = 1.0
        tracker.V = first[:, None].contiguous()
        tracker.S = torch.ones(1, dtype=torch.float32)
        tracker.proj_k = None
        x = torch.tensor([
            [0.2, -0.1, 0.5, 0.7], [-0.4, 0.3, 0.1, -0.2],
            [0.8, 0.2, -0.6, 0.4], [0.1, 0.9, -0.3, -0.5],
            [-0.7, 0.6, 0.4, 0.2], [0.3, -0.8, 0.9, 0.1],
        ], dtype=torch.float32)
        target = torch.tensor([0, 1, 0, 1, 1, 0], dtype=torch.long)
        return model, optimizer, tracker, x, target

    def state(self):
        model, optimizer, tracker, x, target = self.fixture()
        return core.i14.snapshot(model, optimizer, tracker, "sgdm", core.LR), x, target

    def observe_expected(self, state, x, target, mean):
        model, optimizer, tracker = core.i14.restore(state, "cpu")
        old = core._buffer(model, optimizer)
        pre_basis = core.i9._basis(tracker)
        optimizer.zero_grad(set_to_none=True)
        loss = core.i9._objective_loss(model, x, target)
        loss.backward()
        raw = core.i14._flat_grad(model)
        tracker.filter_grad()
        native = core.i14._flat_grad(model)
        basis = core.i9._basis(tracker)
        post_mean = tracker.grad_mean.detach().clone()
        action = lambda value: tracker._project_gradient(value.detach().clone()).detach().clone()
        delivered = native + (post_mean - action(post_mean)) if mean else native
        return old, pre_basis, raw, native, basis, post_mean, action, delivered

    def test_current_native_is_exact_i14_sgdm_step(self):
        state, x, target = self.state()
        left = core.i14.restore(state, "cpu")
        right = core.i14.restore(state, "cpu")
        expected = core.i14.train_step(*left, x, target, "current32", "sgdm")
        observed = core.history_step(*right, x, target, "current_native",
                                     capture_digests=True)
        left_state = core.i14.snapshot(*left, "sgdm", core.LR)
        right_state = core.i14.snapshot(*right, "sgdm", core.LR)
        self.assertTrue(core.i9.equal_tree(left_state, right_state))
        self.assertEqual(observed["loss"], expected["loss"])
        self.assertEqual(observed["gradient"]["applied_norm"],
                         expected["gradient"]["applied_norm"])
        self.assertIsNotNone(observed["digests"])

    def test_projected_history_matches_explicit_sgdm_formula(self):
        state, x, target = self.state()
        old, _, _, native, basis, _, action, _ = self.observe_expected(
            state, x, target, mean=False)
        model, optimizer, tracker = core.i14.restore(state, "cpu")
        record = core.history_step(model, optimizer, tracker, x, target,
                                   "current_projected_history")
        new = core._buffer(model, optimizer)
        expected = 0.9 * action(old) + native
        self.assertTrue(torch.allclose(new, expected, rtol=2e-6, atol=2e-7))
        self.assertTrue(record["momentum_history"]["history_projected"])
        self.assertLessEqual(record["algebra_residuals"]["momentum_buffer_recurrence"]["norm"],
                             record["algebra_residuals"]["momentum_buffer_recurrence"]
                             ["homogeneous_error_bound"])
        self.assertEqual(tuple(basis.shape), (old.numel(), 2))

    def test_moving_basis_changes_old_history(self):
        state, x, target = self.state()
        old, pre_basis, _, _, post_basis, _, action, _ = self.observe_expected(
            state, x, target, mean=False)
        pre = pre_basis @ (pre_basis.T @ old)
        post = action(old)
        self.assertGreater(float(torch.linalg.vector_norm(post - pre)), 1e-5)
        model, optimizer, tracker = core.i14.restore(state, "cpu")
        record = core.history_step(model, optimizer, tracker, x, target,
                                   "current_projected_history")
        self.assertGreater(record["momentum_history"]["removed_old_buffer"]["norm"], 0)
        self.assertTrue(math.isfinite(
            record["momentum_history"]["native_action_idempotence_defect"]["norm"]))
        self.assertFalse(record["momentum_history"]
                         ["native_action_idempotence_defect"]["enforced"])
        self.assertEqual(set(record["components"]), {
            "raw_gradient", "native_projection", "post_mean",
            "native_action_post_mean", "post_mean_complement",
            "mean_delivery", "applied_delivery"})

    def test_mean_complement_is_not_erased_by_projected_history(self):
        state, x, target = self.state()
        old, _, _, _, basis, post_mean, action, delivered = self.observe_expected(
            state, x, target, mean=True)
        complement = post_mean - action(post_mean)
        self.assertGreater(float(torch.linalg.vector_norm(complement)), 1e-6)
        model, optimizer, tracker = core.i14.restore(state, "cpu")
        core.history_step(model, optimizer, tracker, x, target,
                          "mean_projected_history")
        new = core._buffer(model, optimizer)
        expected = 0.9 * action(old) + delivered
        self.assertTrue(torch.allclose(new, expected, rtol=2e-6, atol=2e-7))
        # With an empirical nearly-projective action, the new outside component
        # is the newly delivered mean complement, not a projected whole buffer.
        outside_new = new - action(new)
        self.assertTrue(torch.allclose(outside_new, complement,
                                       rtol=5e-5, atol=5e-6))

    def test_fixed_action_mean_history_has_factor_ten_dc_gain(self):
        # Exact fixed-P real-arithmetic distinction: native history accumulates
        # a constant outside-mean delivery; projected history deletes old Q-state.
        q = torch.tensor([0.0, 2.0], dtype=torch.float64)
        old_steady = q / (1.0 - core.MOMENTUM)
        projected_old = torch.tensor([old_steady[0], 0.0], dtype=torch.float64)
        native_after = core.MOMENTUM * old_steady + q
        projected_after = core.MOMENTUM * projected_old + q
        self.assertTrue(torch.equal(native_after, old_steady))
        self.assertTrue(math.isclose(float(native_after[1] / projected_after[1]),
                                     10.0, rel_tol=2e-15, abs_tol=0.0))

    def test_parent_binding_and_state_continuity(self):
        state, x, target = self.state()
        binding = core.validate_parent_snapshot(state)
        self.assertEqual(binding["observer_step"], 100)
        model, optimizer, tracker = core.i14.restore(state, "cpu")
        old = core._buffer(model, optimizer)
        record = core.history_step(model, optimizer, tracker, x, target, "mean_native")
        self.assertEqual(record["observer"]["step_before"], 100)
        self.assertEqual(record["observer"]["step_after"], 101)
        self.assertFalse(record["momentum_history"]["history_projected"])
        self.assertFalse(torch.equal(old, core._buffer(model, optimizer)))
        bad = dict(state)
        bad["lr"] = 0.031
        with self.assertRaisesRegex(core.HistoryCoreError, "SGDm/.03"):
            core.validate_parent_snapshot(bad)
        model, optimizer, tracker = core.i14.restore(state, "cpu")
        del optimizer.state[next(iter(model.parameters()))]
        with self.assertRaises((core.HistoryCoreError, core.i14.OptimizerCoreError)):
            core.history_step(model, optimizer, tracker, x, target, "mean_native")
        with self.assertRaisesRegex(core.HistoryCoreError, "unknown"):
            model, optimizer, tracker = core.i14.restore(state, "cpu")
            core.history_step(model, optimizer, tracker, x, target, "raw")

    def test_nonfinite_state_has_typed_failure(self):
        state, x, target = self.state()
        model, optimizer, tracker = core.i14.restore(state, "cpu")
        with torch.no_grad():
            next(iter(model.parameters())).view(-1)[0] = float("inf")
        with self.assertRaises(core.NumericalFailure):
            core.history_step(model, optimizer, tracker, x, target, "mean_native")


if __name__ == "__main__":
    unittest.main()
