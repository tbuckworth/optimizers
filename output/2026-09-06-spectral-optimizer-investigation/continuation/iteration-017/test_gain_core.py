"""CPU-only synthetic tests for I17 gain-normalized temporal controls."""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import math
from pathlib import Path
import sys
import unittest

import torch


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gain_core as core


def _threads() -> None:
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        if torch.get_num_interop_threads() != 1:
            raise


class GainCoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _threads()

    def fixture(self):
        model = core.i9.make_model(1701, "cpu", input_dim=4, width=3, classes=2)
        optimizer = core.i14.make_optimizer(model, "sgdm", core.LR)
        tracker = core.i9.make_tracker(model, optimizer)
        count = sum(parameter.numel() for parameter in model.parameters())
        old = torch.linspace(-.41, .37, count, dtype=torch.float32)
        position = 0
        for parameter in model.parameters():
            size = parameter.numel()
            optimizer.state[parameter]["momentum_buffer"] = (
                old[position:position + size].reshape_as(parameter).clone())
            position += size
        tracker.step_count = 100
        tracker.grad_mean = torch.linspace(.23, -.19, count, dtype=torch.float32)
        basis = torch.zeros((count, 2), dtype=torch.float32)
        basis[0, 0] = 1.0
        basis[1, 0] = .04
        basis[1, 1] = .97
        tracker.V = basis.contiguous()
        tracker.S = torch.tensor([1.2, .8], dtype=torch.float32)
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

    def observed_inputs(self, state, x, target):
        model, optimizer, tracker = core.i14.restore(state, "cpu")
        old = core.i15._buffer(model, optimizer)
        optimizer.zero_grad(set_to_none=True)
        core.i9._objective_loss(model, x, target).backward()
        raw = core.i14._flat_grad(model)
        tracker.filter_grad()
        native = core.i14._flat_grad(model)
        post_mean = tracker.grad_mean.detach().clone()
        action = lambda value: core.i15._action(tracker, value)
        return old, raw, native, post_mean, tracker, action

    def torch_buffer_reference(self, selected_old, applied):
        parameter = torch.nn.Parameter(torch.zeros_like(selected_old))
        optimizer = torch.optim.SGD(
            [parameter], lr=core.LR, momentum=core.MOMENTUM, dampening=0.0,
            weight_decay=0.0, nesterov=False, maximize=False, foreach=False,
            differentiable=False, fused=False)
        optimizer.state[parameter]["momentum_buffer"] = selected_old.clone()
        parameter.grad = applied.clone()
        optimizer.step()
        return optimizer.state[parameter]["momentum_buffer"].detach().clone()

    def test_policy_contract_and_parent_binding(self) -> None:
        self.assertEqual(core.REAL_POLICIES, (
            "scalar_k0p5", "scalar_k0p9", "scalar_k1",
            "spectral_mean_projected"))
        self.assertEqual(core.TEST_ONLY_POLICIES, ("scalar_k0",))
        state, x, target = self.state()
        binding = core.validate_parent_snapshot(state)
        self.assertEqual(binding["schema"], "i17_parent_binding_v1")
        self.assertEqual(binding["observer_step"], 100)
        model, optimizer, tracker = core.i14.restore(state, "cpu")
        with self.assertRaisesRegex(core.GainCoreError, "unknown"):
            core.gain_step(model, optimizer, tracker, x, target, "scalar_.5")
        bad = dict(state)
        bad["lr"] = .031
        with self.assertRaises((core.GainCoreError, core.i16.ScalarCoreError,
                                core.i15.HistoryCoreError)):
            core.validate_parent_snapshot(bad)

    def test_k0_is_bitwise_full_state_parity_with_i16(self) -> None:
        state, x, target = self.state()
        expected_live = core.i14.restore(state, "cpu")
        observed_live = core.i14.restore(state, "cpu")
        for index in range(3):
            core.i16.scalar_step(*expected_live, x, target, 0.0,
                                 capture_digests=(index == 0))
            record = core.gain_step(*observed_live, x, target, "scalar_k0",
                                    capture_digests=(index == 0))
        expected_state = core.i14.snapshot(*expected_live, "sgdm", core.LR)
        observed_state = core.i14.snapshot(*observed_live, "sgdm", core.LR)
        self.assertTrue(core.i9.equal_tree(expected_state, observed_state))
        self.assertTrue(core.i9.equal_tree(expected_state["gradients"],
                                           observed_state["gradients"]))
        self.assertTrue(core.i9.equal_tree(expected_state["tracker"],
                                           observed_state["tracker"]))
        self.assertTrue(core.i9.equal_tree(expected_state["rng"],
                                           observed_state["rng"]))
        self.assertEqual(record["normalized_delivery"]["scalar_gain_correction"], 1.0)
        self.assertEqual(record["normalized_delivery"]["delivery_literal"], "post_mean")
        self.assertEqual(record["normalized_delivery"]["history_literal"], "zero")

    def test_scalar_buffers_match_native_torch_mul_then_add(self) -> None:
        state, x, target = self.state()
        old, raw, _, post_mean, _, _ = self.observed_inputs(state, x, target)
        policies = {"scalar_k0p5": .5, "scalar_k0p9": .9, "scalar_k1": 1.0}
        for policy, k in policies.items():
            with self.subTest(policy=policy):
                applied = k * raw + (1.0 - k) * post_mean
                selected_old = k * old
                expected_buffer = self.torch_buffer_reference(selected_old, applied)
                model, optimizer, tracker = core.i14.restore(state, "cpu")
                before = core.i9.flat_params(model)
                record = core.gain_step(model, optimizer, tracker, x, target, policy)
                actual_buffer = core.i15._buffer(model, optimizer)
                self.assertTrue(torch.equal(actual_buffer, expected_buffer))
                gain = 1.0 - core.MOMENTUM * k
                after = core.i9.flat_params(model)
                delivered = gain * expected_buffer
                expected_after = before.clone()
                expected_after.mul_(core.MANUAL_SHRINKAGE)
                expected_after.add_(delivered, alpha=-core.LR)
                self.assertTrue(torch.equal(after, expected_after))
                core.i14._validate_optimizer(model, optimizer, "sgdm", core.LR)
                self.assertEqual(optimizer.param_groups[0]["lr"], core.LR)
                self.assertEqual(record["normalized_delivery"]["scalar_gain_correction"], gain)
                self.assertTrue(record["normalized_delivery"]["unnormalized_buffer_retained"])
                for residual in record["algebra_residuals"].values():
                    self.assertLessEqual(residual["norm"], residual["homogeneous_error_bound"])

    def test_spectral_buffer_matches_i15_and_actual_action_delivery(self) -> None:
        state, x, target = self.state()
        reference_live = core.i14.restore(state, "cpu")
        observed_live = core.i14.restore(state, "cpu")
        core.i15.history_step(*reference_live, x, target, "mean_projected_history")
        before = core.i9.flat_params(observed_live[0])
        record = core.gain_step(*observed_live, x, target,
                                "spectral_mean_projected", capture_digests=True)
        expected_buffer = core.i15._buffer(reference_live[0], reference_live[1])
        observed_buffer = core.i15._buffer(observed_live[0], observed_live[1])
        self.assertTrue(torch.equal(observed_buffer, expected_buffer))
        action_new = core.i15._action(observed_live[2], observed_buffer)
        delivered = observed_buffer - core.MOMENTUM * action_new
        after = core.i9.flat_params(observed_live[0])
        expected_after = before * core.MANUAL_SHRINKAGE - core.LR * delivered
        self.assertTrue(torch.allclose(after, expected_after, rtol=0.0, atol=3e-8))
        core.i14._validate_optimizer(observed_live[0], observed_live[1],
                                     "sgdm", core.LR)
        self.assertEqual(observed_live[1].param_groups[0]["lr"], core.LR)
        self.assertEqual(record["normalized_delivery"]["operator"], "(I-rho*A_t)*b_t")
        self.assertTrue(record["gradient_filter_applied"])
        self.assertTrue(record["native_projection_used_for_recurrence"])
        self.assertGreater(record["action_diagnostics"]
                           ["current_basis_orthogonality_error"], 0.0)

    def test_first_step_digests_are_deterministic_and_complete(self) -> None:
        state, x, target = self.state()
        records = []
        for _ in range(2):
            model, optimizer, tracker = core.i14.restore(state, "cpu")
            records.append(core.gain_step(model, optimizer, tracker, x, target,
                                          "scalar_k0p5", capture_digests=True))
        self.assertEqual(records[0]["digests"], records[1]["digests"])
        self.assertEqual(tuple(records[0]["digests"]), (
            "raw_gradient", "post_observer", "old_momentum_buffer",
            "unnormalized_new_momentum_buffer", "delivered_buffer"))
        model, optimizer, tracker = core.i14.restore(state, "cpu")
        self.assertIsNone(core.gain_step(model, optimizer, tracker, x, target,
                                        "scalar_k0p5")["digests"])

    def test_soft_target_and_actual_ideal_step_diagnostics(self) -> None:
        state, x, target = self.state()
        soft = .1 * torch.nn.functional.one_hot(target, num_classes=2).float() + .45
        model, optimizer, tracker = core.i14.restore(state, "cpu")
        record = core.gain_step(model, optimizer, tracker, x, soft,
                                "spectral_mean_projected")
        self.assertTrue(math.isfinite(record["loss"]))
        self.assertGreater(record["displacement"]["data_norm"], 0.0)
        self.assertGreater(record["displacement"]["ideal_data_norm"], 0.0)
        self.assertTrue(math.isfinite(record["displacement"]
                                     ["actual_minus_ideal_relative"]))
        self.assertEqual(record["decay"]["factor"], .9997)
        self.assertTrue(record["decay"]["manual_before_data_step"])

    def test_nonfinite_and_structural_failures_remain_distinct(self) -> None:
        state, x, target = self.state()
        model, optimizer, tracker = core.i14.restore(state, "cpu")
        with torch.no_grad():
            next(iter(model.parameters())).view(-1)[0] = float("inf")
        with self.assertRaises(core.NumericalFailure):
            core.gain_step(model, optimizer, tracker, x, target, "scalar_k0p5")
        model, optimizer, tracker = core.i14.restore(state, "cpu")
        del optimizer.state[next(iter(model.parameters()))]
        with self.assertRaises((core.GainCoreError, core.i15.HistoryCoreError,
                                core.i14.OptimizerCoreError)):
            core.gain_step(model, optimizer, tracker, x, target, "scalar_k0p5")


if __name__ == "__main__":
    unittest.main()
