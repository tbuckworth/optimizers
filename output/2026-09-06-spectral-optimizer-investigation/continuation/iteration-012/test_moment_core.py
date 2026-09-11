"""Synthetic CPU tests for the isolated I12 moment intervention core."""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
import sys
import unittest
from unittest import mock

import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import moment_core as core


def fixture():
    generator = torch.Generator().manual_seed(12012)
    x = torch.rand((128, 4), generator=generator)
    clean = torch.randint(0, 3, (128,), generator=generator)
    noisy = (clean + (torch.arange(128) % 2)).remainder(3)
    data = {"x": x, "clean": clean, "noisy": noisy,
            "vx": x.flip(0).clone(), "vy": clean.flip(0).clone(),
            "ax": x.roll(11, 0).clone(), "ay": clean.roll(11, 0).clone()}
    model = core.i9.make_model(12012, "cpu", input_dim=4, width=5, classes=3)
    optimizer = core.i9.make_optimizer(model)
    tracker = core.i9.make_tracker(model, optimizer)
    for offset in range(4):
        indices = torch.arange(64).add(13 * offset).remainder(128)
        core.i9.train_step(model, optimizer, tracker, x[indices], noisy[indices], "raw")
    count = core.i9.flat_params(model).numel()
    tracker.V = torch.eye(count)[:, :2].contiguous()
    tracker.S = torch.ones(2, dtype=torch.float64)
    tracker.grad_mean = torch.linspace(-0.05, 0.05, count)
    tracker.step_count = 100
    state = core.i9.snapshot(model, optimizer, tracker)
    plan = {"batches": torch.arange(3 * 64).reshape(3, 64).remainder(128),
            "redraw_mask": torch.zeros((3, 64), dtype=torch.bool),
            "redraw_digits": torch.zeros((3, 64), dtype=torch.long)}
    return state, data, plan


class MomentCoreTests(unittest.TestCase):
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

    def test_exact_mutation_masks_parent_neutrality_and_topology(self):
        parent, _, _ = fixture()
        parent_copy = core.i9._cpu_clone(parent)
        mappings = {
            "zero_m": {"exp_avg"}, "zero_v": {"exp_avg_sq"},
            "zero_mv": {"exp_avg", "exp_avg_sq"},
            "fresh_adam": {"step", "exp_avg", "exp_avg_sq"},
        }
        for arm, selected in mappings.items():
            edited, audit = core.edit_moments(parent, arm)
            self.assertTrue(core.i9.equal_tree(parent, parent_copy))
            self.assertTrue(audit["parent_unchanged"])
            self.assertTrue(audit["unmodified_fields_equal"])
            self.assertEqual(set(audit["zeroed_fields"]), selected)
            self.assertEqual(tuple(parent), tuple(edited))
            self.assertTrue(core.i9.equal_tree(parent["rng"], edited["rng"]))
            self.assertTrue(core.i9.equal_tree(parent["tracker"], edited["tracker"]))
            for parameter_id, before in parent["optimizer"]["state"].items():
                after = edited["optimizer"]["state"][parameter_id]
                self.assertEqual(tuple(before), tuple(after))
                for key in core._STATE_KEYS:
                    self.assertEqual(before[key].dtype, after[key].dtype)
                    self.assertEqual(tuple(before[key].shape), tuple(after[key].shape))
                    if key in selected:
                        self.assertTrue(bool((after[key] == 0).all()))
                    else:
                        self.assertTrue(torch.equal(before[key], after[key]))
            self.assertEqual(audit["counter_after"], 0 if arm == "fresh_adam" else 4)

        with self.assertRaisesRegex(core.MomentCoreError, "synthetic fixture"):
            core.edit_moments(parent, "inherited")
        inherited, audit = core.edit_moments(parent, "inherited", synthetic_fixture=True)
        self.assertTrue(core.i9.equal_tree(parent, inherited))
        self.assertEqual(audit["zeroed_fields"], [])

        malformed = core.i9._cpu_clone(parent)
        malformed["optimizer"]["param_groups"][0]["amsgrad"] = True
        with self.assertRaisesRegex(core.MomentCoreError, "configuration"):
            core.edit_moments(malformed, "zero_m")

    def test_first_adam_data_step_matches_closed_form_and_old_counter_amplitude(self):
        parent, _, _ = fixture()
        beta1, beta2, epsilon = 0.9, 0.999, 1e-8
        for old_counter in (4, 1500, 2000):
            counter_parent = core.i9._cpu_clone(parent)
            for row in counter_parent["optimizer"]["state"].values():
                row["step"].fill_(old_counter)
            zero_mv, audit = core.edit_moments(counter_parent, "zero_mv")
            fresh, _ = core.edit_moments(counter_parent, "fresh_adam")
            self.assertEqual(audit["counter_before"], old_counter)
            amplitudes = {}
            for name, state in (("old", zero_mv), ("fresh", fresh)):
                model, optimizer, _ = core.i9.restore(state, "cpu")
                before = core.i9.flat_params(model)
                for parameter in model.parameters():
                    parameter.grad = torch.ones_like(parameter)
                optimizer.step()
                after = core.i9.flat_params(model)
                data_delta = after - before + core.i9.LR * core.i9.WD * before
                amplitudes[name] = -float(data_delta.to(torch.float64).mean().item())
                self.assertLess(float(data_delta.to(torch.float64).std().item()), 2e-8)

            expected_old = core.i9.LR * ((1 - beta1) / (1 - beta1 ** (old_counter + 1))) / (
                math.sqrt((1 - beta2) / (1 - beta2 ** (old_counter + 1))) + epsilon)
            expected_fresh = core.i9.LR / (1 + epsilon)
            # The optimizer and decay-adjustment arithmetic are float32; tolerate
            # the final few ulps introduced by reconstructing the data-only delta.
            self.assertTrue(math.isclose(amplitudes["old"], expected_old,
                                         rel_tol=2e-5, abs_tol=1e-8))
            self.assertTrue(math.isclose(amplitudes["fresh"], expected_fresh,
                                         rel_tol=2e-5, abs_tol=1e-8))
            self.assertTrue(math.isclose(amplitudes["old"] / amplitudes["fresh"],
                                         expected_old / expected_fresh,
                                         rel_tol=2e-5, abs_tol=1e-8))

    def test_healthy_inherited_branch_is_exact_i10_for_all_policies(self):
        parent, data, plan = fixture()
        inherited, _ = core.edit_moments(parent, "inherited", synthetic_fixture=True)
        for policy in core.i10.POLICIES:
            actual = core.run_branch(inherited, data, plan, "fixed", policy,
                                     steps=3, horizons=(0, 1, 3))
            model, optimizer, tracker = core.i9.restore(inherited, "cpu")
            basis = core.i9._basis(tracker)
            expected_curve = [{"horizon": 0, **core.i10.evaluate(model, data)}]
            expected_steps = []
            batches = core.i9._indices(plan["batches"], (3, 64), len(data["x"]),
                                       torch.device("cpu"))
            for horizon in range(1, 4):
                indices = batches[horizon - 1]
                row = core.i10.branch_step(model, optimizer, tracker, data["x"][indices],
                                           data["noisy"][indices], policy, basis)
                expected_steps.append({"horizon": horizon, **row})
                if horizon in (1, 3):
                    expected_curve.append({"horizon": horizon,
                                           **core.i10.evaluate(model, data)})
            expected_final = core.i9.snapshot(model, optimizer, tracker)
            self.assertEqual(actual["status"], "complete")
            self.assertEqual(actual["curve"], expected_curve)
            self.assertEqual(actual["steps"], expected_steps)
            self.assertTrue(core.i9.equal_tree(actual["terminal_state"], expected_final))
            self.assertEqual(actual["completed_steps"], 3)
            self.assertEqual(actual["attempted_step"], 3)
            self.assertIsNone(actual["numerical_failure"])
            self.assertIsInstance(actual["first_step_applied_gradient_digest"], str)

    def test_first_delivered_gradient_is_identical_across_moment_surgeries(self):
        parent, data, plan = fixture()
        digests = []
        for arm in core.ARMS:
            edited, _ = core.edit_moments(parent, arm)
            result = core.run_branch(edited, data, {key: value[:1] for key, value in plan.items()},
                                     "fixed", "current32", steps=1, horizons=(0, 1))
            digests.append(result["first_step_applied_gradient_digest"])
        self.assertEqual(len(set(digests)), 1)

    def test_initial_seam_mismatch_refuses_before_first_step(self):
        parent, data, plan = fixture()
        model, _, _ = core.i9.restore(parent, "cpu")
        expected = core.i10.evaluate(model, data)
        expected["auxiliary"]["clean_ce"] += 1.0
        calls = []

        def check():
            calls.append(len(calls))
            if len(calls) > 1:
                raise AssertionError("a training callback was reached")

        with self.assertRaisesRegex(core.MomentCoreError, "horizon-zero"):
            core.run_branch(parent, data, plan, "fixed", "raw", steps=3,
                            horizons=(0, 3), check=check,
                            expected_initial_evaluation=expected)
        self.assertEqual(len(calls), 1)

    def test_numeric_failure_seals_partial_state_but_callback_errors_propagate(self):
        parent, data, plan = fixture()
        original = core.i10.branch_step
        calls = 0

        def fail_second(*args, **kwargs):
            nonlocal calls
            calls += 1
            row = original(*args, **kwargs)
            if calls == 2:
                with torch.no_grad():
                    next(args[0].parameters()).reshape(-1)[0] = float("inf")
                raise core.i9.NeuralCoreError("updated parameters must be a finite dense tensor")
            return row

        with mock.patch.object(core.i10, "branch_step", side_effect=fail_second):
            result = core.run_branch(parent, data, plan, "fixed", "raw", steps=3,
                                     horizons=(0, 3))
        self.assertEqual(result["status"], "numerical_failure")
        self.assertEqual(result["completed_steps"], 1)
        self.assertEqual(result["attempted_step"], 2)
        self.assertEqual(result["last_evaluated_horizon"], 0)
        self.assertIsInstance(result["first_step_applied_gradient_digest"], str)
        self.assertEqual(len(result["steps"]), 1)
        self.assertTrue(core.i9.equal_tree(result["last_evaluated_state"], parent))
        self.assertTrue(bool(torch.isinf(next(iter(result["terminal_state"]["model_state"].values()))).any()))

        callback_calls = 0

        def generic_failure():
            nonlocal callback_calls
            callback_calls += 1
            if callback_calls == 2:
                raise RuntimeError("synthetic resource callback failure")

        with self.assertRaisesRegex(RuntimeError, "resource callback"):
            core.run_branch(parent, data, plan, "fixed", "raw", steps=3,
                            horizons=(0, 3), check=generic_failure)

    def test_nonfinite_loss_with_finite_gradient_is_narrowly_sealed(self):
        parent, data, plan = fixture()

        def nan_value_finite_gradient(model, _x, _target):
            zero_with_gradient = sum(parameter.sum() * 0 for parameter in model.parameters())
            return zero_with_gradient + torch.tensor(float("nan"))

        short_plan = {key: value[:1] for key, value in plan.items()}
        with mock.patch.object(core.i9, "_objective_loss",
                               side_effect=nan_value_finite_gradient):
            result = core.run_branch(parent, data, short_plan, "fixed", "raw",
                                     steps=1, horizons=(0, 1))
        self.assertEqual(result["status"], "numerical_failure")
        self.assertEqual(result["completed_steps"], 0)
        self.assertEqual(result["attempted_step"], 1)
        self.assertEqual(result["numerical_failure"]["exception_type"], "ValueError")
        self.assertIn(result["numerical_failure"]["message"],
                      core._JSON_NONFINITE_MESSAGES)
        self.assertIsNone(result["first_step_applied_gradient_digest"])
        model, optimizer, tracker = core.i9.restore(result["terminal_state"], "cpu")
        core._finite_live(model, optimizer, tracker, "sealed_terminal")

        self.assertFalse(core._is_known_numeric(ValueError("another JSON error")))

    def test_live_scan_sees_parameter_nan_and_nondense_is_structural(self):
        parent, _, _ = fixture()
        model, optimizer, tracker = core.i9.restore(parent, "cpu")
        with torch.no_grad():
            next(model.parameters()).reshape(-1)[0] = float("nan")
        with self.assertRaises(core._NumericalFailure):
            core._finite_live(model, optimizer, tracker, "test")
        sparse = torch.tensor([1.0, 0.0]).to_sparse()
        with self.assertRaisesRegex(core.MomentCoreError, "nondense"):
            core._finite_record({"bad": sparse})


if __name__ == "__main__":
    unittest.main()
