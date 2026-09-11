"""Synthetic CPU tests for I13 mean-preserving gradient delivery."""
from __future__ import annotations

import hashlib
from pathlib import Path
import sys
import unittest

import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import mean_core as core


def fixture(steps: int = 3):
    generator = torch.Generator().manual_seed(13013)
    x = torch.rand((128, 4), generator=generator)
    clean = torch.randint(0, 10, (128,), generator=generator)
    noisy = (clean + (torch.arange(128) % 3)).remainder(10)
    data = {"x": x, "clean": clean, "noisy": noisy,
            "vx": x.flip(0).clone(), "vy": clean.flip(0).clone(),
            "ax": x.roll(9, 0).clone(), "ay": clean.roll(9, 0).clone()}
    model = core.i9.make_model(13013, "cpu", input_dim=4, width=5, classes=10)
    optimizer = core.i9.make_optimizer(model)
    tracker = core.i9.make_tracker(model, optimizer)
    for offset in range(4):
        indices = torch.arange(64).add(11 * offset).remainder(128)
        core.i9.train_step(model, optimizer, tracker, x[indices], noisy[indices], "raw")
    count = core.i9.flat_params(model).numel()
    tracker.V = torch.eye(count)[:, :4].contiguous()
    tracker.S = torch.ones(4, dtype=torch.float64)
    tracker.grad_mean = torch.linspace(-0.08, 0.06, count)
    tracker.step_count = 100
    state = core.i9.snapshot(model, optimizer, tracker)
    plan = {"batches": torch.arange(steps * 64).reshape(steps, 64).remainder(128),
            "redraw_mask": torch.arange(steps * 64).reshape(steps, 64).remainder(4) != 0,
            "redraw_digits": torch.arange(steps * 64).reshape(steps, 64).remainder(10)}
    return state, data, plan


class MeanCoreTests(unittest.TestCase):
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

    def test_observe_once_algebra_delivery_and_no_adam_step(self):
        parent, data, _ = fixture()
        rows = {}
        for policy in core.POLICIES:
            model, optimizer, tracker = core.i9.restore(parent, "cpu")
            parameter_before = core.i9.flat_params(model)
            optimizer_before = core.i9.tree_digest(core.i9._cpu_clone(optimizer.state_dict()))
            counter_before = core.i12._optimizer_topology(parent)[1]
            indices = torch.arange(64)
            record, tensors = core.observe_delivery(
                model, optimizer, tracker, data["x"][indices], data["noisy"][indices], policy)
            self.assertEqual(tracker.step_count, 101)
            self.assertEqual(core.i12._optimizer_topology(
                core.i9.snapshot(model, optimizer, tracker))[1], counter_before)
            self.assertTrue(torch.equal(parameter_before, core.i9.flat_params(model)))
            self.assertEqual(optimizer_before,
                             core.i9.tree_digest(core.i9._cpu_clone(optimizer.state_dict())))
            basis = tensors["post_basis"]
            project = lambda value: basis @ (basis.T @ value)
            raw, native = tensors["raw_gradient"], tensors["native_projection"]
            mean, leak = tensors["mean32_delivery"], tensors["leak01_32_delivery"]
            expected_mean = tensors["post_mean"] + project(raw - tensors["post_mean"])
            expected_leak = native + 0.01 * (raw - native)
            expected_delta = 0.99 * (tensors["pre_mean"] - project(tensors["pre_mean"]))
            self.assertTrue(torch.allclose(native, project(raw), rtol=2e-6, atol=2e-7))
            self.assertTrue(torch.allclose(mean, expected_mean, rtol=2e-6, atol=2e-7))
            self.assertTrue(torch.allclose(leak, expected_leak, rtol=2e-6, atol=2e-7))
            self.assertTrue(torch.allclose(mean - leak, expected_delta, rtol=2e-6, atol=2e-7))
            expected_delivery = {"raw": raw, "current32": native,
                                 "mean32": mean, "leak01_32": leak}[policy]
            self.assertTrue(torch.equal(tensors["delivered_gradient"], expected_delivery))
            self.assertTrue(torch.equal(core.i9.flat_grad(model), expected_delivery))
            for component in record["components"].values():
                self.assertAlmostEqual(component["norm"] ** 2,
                                       component["squared_energy"], places=8)
            self.assertEqual(set(record["algebra_residual_norms"]),
                             set(record["algebra_error_bounds"]))
            for name, residual in record["algebra_residual_norms"].items():
                self.assertLessEqual(residual, record["algebra_error_bounds"][name])
            rows[policy] = record
        self.assertEqual(rows["mean32"]["raw_gradient_digest"],
                         rows["leak01_32"]["raw_gradient_digest"])
        self.assertEqual(rows["mean32"]["post_observer_digest"],
                         rows["leak01_32"]["post_observer_digest"])
        self.assertNotEqual(rows["mean32"]["applied_gradient_digest"],
                            rows["leak01_32"]["applied_gradient_digest"])

    def test_branch_step_matches_explicit_observe_then_adam(self):
        parent, data, _ = fixture()
        reference_model, reference_optimizer, reference_tracker = core.i9.restore(parent, "cpu")
        basis = core.i9._basis(reference_tracker)
        indices = torch.arange(64)
        core.observe_delivery(reference_model, reference_optimizer, reference_tracker,
                              data["x"][indices], data["noisy"][indices], "mean32")
        reference_optimizer.step()
        expected = core.i9.snapshot(reference_model, reference_optimizer, reference_tracker)

        model, optimizer, tracker = core.i9.restore(parent, "cpu")
        row = core.branch_step(model, optimizer, tracker, data["x"][indices],
                               data["noisy"][indices], "mean32", basis)
        self.assertTrue(core.i9.equal_tree(expected, core.i9.snapshot(model, optimizer, tracker)))
        self.assertEqual(row["observer"]["step_after"], 101)
        self.assertTrue(row["gradient_filter_applied"])
        self.assertEqual(row["gradient"]["applied_norm"],
                         row["mean_diagnostics"]["components"]["mean32_delivery"]["norm"])

    def test_run_branch_all_targets_and_two_new_policies(self):
        parent, data, plan = fixture(steps=3)
        parent_digest = core.i9.tree_digest(parent)
        results = {}
        for objective in core.OBJECTIVES:
            for policy in ("mean32", "leak01_32"):
                model, _, _ = core.i9.restore(parent, "cpu")
                expected = core.i10.evaluate(model, data)
                calls = []

                def probe(snapshot, horizon):
                    calls.append(horizon)
                    return {"schema": "synthetic_probe_v1", "horizon": horizon,
                            "state_digest": core.i9.tree_digest(snapshot)}

                result = core.run_branch(parent, data, plan, objective, policy, steps=3,
                    horizons=(0, 1, 3), expected_initial_evaluation=expected, probe=probe)
                self.assertEqual(result["status"], "complete")
                self.assertEqual([row["horizon"] for row in result["curve"]], [0, 1, 3])
                self.assertEqual([row["horizon"] for row in result["probes"]], [0, 1, 3])
                self.assertEqual(calls, [0, 1, 3])
                self.assertEqual(result["completed_steps"], 3)
                self.assertEqual(core.i12._optimizer_topology(result["terminal_state"])[1], 7)
                self.assertEqual(result["terminal_state"]["tracker"]["step_count"], 103)
                self.assertEqual(core.i9.tree_digest(parent), parent_digest)
                results[(objective, policy)] = result
            left, right = results[(objective, "mean32")], results[(objective, "leak01_32")]
            self.assertEqual(left["first_step_raw_gradient_digest"],
                             right["first_step_raw_gradient_digest"])
            self.assertEqual(left["first_step_observer_digest"],
                             right["first_step_observer_digest"])
            self.assertNotEqual(left["first_step_applied_gradient_digest"],
                                right["first_step_applied_gradient_digest"])

    def test_raw_and_current_paths_equal_frozen_i10_for_all_targets(self):
        parent, data, plan = fixture(steps=3)
        for objective in core.OBJECTIVES:
            for policy in ("raw", "current32"):
                model, optimizer, tracker = core.i9.restore(parent, "cpu")
                basis = core.i9._basis(tracker)
                reference_curve = [{"horizon": 0, **core.i10.evaluate(model, data)}]
                reference_steps = []
                redraw_mask = torch.as_tensor(plan["redraw_mask"])
                redraw_digits = torch.as_tensor(plan["redraw_digits"])
                for horizon in range(1, 4):
                    indices = torch.as_tensor(plan["batches"][horizon - 1])
                    target = core.i12._target(data, indices, objective, redraw_mask,
                                              redraw_digits, horizon - 1)
                    row = core.i10.branch_step(model, optimizer, tracker,
                                               data["x"][indices], target, policy, basis)
                    reference_steps.append({"horizon": horizon, **row})
                    if horizon in (1, 3):
                        reference_curve.append({"horizon": horizon,
                                                **core.i10.evaluate(model, data)})
                reference_state = core.i9.snapshot(model, optimizer, tracker)

                result = core.run_branch(parent, data, plan, objective, policy, steps=3,
                                         horizons=(0, 1, 3))
                self.assertEqual(result["curve"], reference_curve)
                self.assertTrue(core.i9.equal_tree(result["terminal_state"], reference_state))
                self.assertEqual(len(result["steps"]), len(reference_steps))
                for actual, expected in zip(result["steps"], reference_steps):
                    actual_common = {key: value for key, value in actual.items()
                                     if key not in ("schema", "mean_diagnostics")}
                    expected_common = {key: value for key, value in expected.items()
                                       if key != "schema"}
                    self.assertEqual(actual_common, expected_common)

    def test_probe_and_seam_failures_are_structural_and_preupdate(self):
        parent, data, plan = fixture(steps=1)
        model, _, _ = core.i9.restore(parent, "cpu")
        expected = core.i10.evaluate(model, data)
        wrong = core.i9._cpu_clone(expected)
        wrong["auxiliary"]["clean_ce"] += 1
        checks = []

        def check():
            checks.append(1)
            if len(checks) > 1:
                raise AssertionError("update callback reached")

        with self.assertRaisesRegex(core.MeanCoreError, "horizon-zero"):
            core.run_branch(parent, data, plan, "fixed", "mean32", steps=1,
                            horizons=(0, 1), check=check,
                            expected_initial_evaluation=wrong)
        self.assertEqual(len(checks), 1)

        def resource_failure(_state, _horizon):
            raise RuntimeError("synthetic probe resource failure")

        with self.assertRaisesRegex(RuntimeError, "probe resource"):
            core.run_branch(parent, data, plan, "fixed", "mean32", steps=1,
                            horizons=(0, 1), expected_initial_evaluation=expected,
                            probe=resource_failure)

        def mutate_snapshot(snapshot, _horizon):
            snapshot["model_state"]["0.bias"].zero_()
            return {"ok": True}

        with self.assertRaisesRegex(core.MeanCoreError, "mutated its snapshot"):
            core.run_branch(parent, data, plan, "fixed", "mean32", steps=1,
                            horizons=(0, 1), expected_initial_evaluation=expected,
                            probe=mutate_snapshot)

    def test_exact_nonfinite_probe_gradient_is_sealed_only_after_horizon_zero(self):
        parent, data, plan = fixture(steps=1)
        model, _, _ = core.i9.restore(parent, "cpu")
        expected = core.i10.evaluate(model, data)

        def nonfinite_gradient(_snapshot, _horizon):
            raise core.i9.NeuralCoreError(core._PROBE_GRADIENT_NONFINITE_MESSAGE)

        with self.assertRaisesRegex(core.i12._NumericalFailure, "probe gradient"):
            core.run_branch(parent, data, plan, "fixed", "mean32", steps=1,
                            horizons=(0, 1), expected_initial_evaluation=expected,
                            probe=nonfinite_gradient)

        def late_nonfinite_gradient(_snapshot, horizon):
            if horizon == 1:
                raise core.i9.NeuralCoreError(core._PROBE_GRADIENT_NONFINITE_MESSAGE)
            return {"schema": "finite_horizon_zero_probe_v1"}

        result = core.run_branch(parent, data, plan, "fixed", "mean32", steps=1,
                                 horizons=(0, 1), expected_initial_evaluation=expected,
                                 probe=late_nonfinite_gradient)
        self.assertEqual(result["status"], "numerical_failure")
        self.assertEqual(result["numerical_failure"]["phase"], "probe")
        self.assertEqual(result["completed_steps"], 1)
        self.assertEqual(result["last_evaluated_horizon"], 1)
        self.assertEqual([row["horizon"] for row in result["curve"]], [0, 1])
        self.assertEqual([row["horizon"] for row in result["probes"]], [0])
        self.assertTrue(core.i9.equal_tree(result["last_evaluated_state"],
                                           result["terminal_state"]))


if __name__ == "__main__":
    unittest.main()
