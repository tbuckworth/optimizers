#!/usr/bin/env python3
"""Synthetic CPU unit tests only; no MNIST training or data loading."""
import json
from pathlib import Path
import tempfile
import unittest
import torch
from torch import nn
import norm_control_harness as n


class NormControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        n.h.configure()

    def test_scalar_candidate_norm_direction_and_actual_delivery(self):
        raw = torch.tensor([3., 4.])
        candidate = torch.tensor([3., 0.])
        scalar, alpha = n.scalar_gradient(raw, candidate, True)
        self.assertAlmostEqual(alpha, .6)
        model = nn.Linear(1, 1)
        n.h.set_grad(model, scalar)
        delivered = n.h.flat_grad(model)
        row = n.delivered_gradient_metrics("scalar32_norm", raw, candidate, delivered,
                                          {"alpha": alpha, "operator": "scalar_identity"})
        self.assertAlmostEqual(row["applied_squared_norm"], 9., places=5)
        self.assertAlmostEqual(row["raw_applied_cosine"], 1., places=6)
        self.assertEqual(row["scalar_collinearity_relative_error"], 0.)
        wrong_delivery = torch.tensor([0., 3.])
        with self.assertRaises(AssertionError):
            n.delivered_gradient_metrics("scalar32_norm", raw, candidate, wrong_delivery,
                                         {"alpha": alpha, "operator": "scalar_identity"})

    def test_scalar_zero_and_warmup_cases(self):
        zero, nonzero = torch.zeros(2), torch.tensor([1., 2.])
        applied, alpha = n.scalar_gradient(zero, zero, True)
        self.assertEqual(alpha, 0.)
        self.assertTrue(torch.equal(applied, zero))
        applied, alpha = n.scalar_gradient(nonzero, zero, True)
        self.assertEqual(alpha, 0.)
        row = n.delivered_gradient_metrics("scalar32_norm", nonzero, zero, applied, {"alpha": alpha})
        self.assertIsNone(row["raw_applied_cosine"])
        applied, alpha = n.scalar_gradient(zero, zero, False)
        self.assertEqual(alpha, 1.)
        self.assertTrue(torch.equal(applied, zero))
        applied, alpha = n.scalar_gradient(nonzero, zero, False)
        self.assertEqual(alpha, 1.)
        self.assertTrue(torch.equal(applied, nonzero))

    def test_scalar_approximate_contraction_gate_no_clamp(self):
        raw = torch.tensor([1., 0.], dtype=torch.float64)
        _, alpha = n.scalar_gradient(raw, raw * 1.002, True)
        self.assertEqual(alpha, 1.002)
        with self.assertRaises(AssertionError):
            n.scalar_gradient(raw, raw * 1.006, True)
        with self.assertRaises(AssertionError):
            n.scalar_gradient(raw, torch.tensor([float("nan"), 0.]), True)
        with self.assertRaises(AssertionError):
            n.scalar_gradient(torch.zeros_like(raw), torch.tensor([float("nan"), 0.]), True)

    def test_checkpoint_survives_cpu_live_parameter_mutation(self):
        model = nn.Linear(2, 2).double().cpu()
        checkpoint = n.cpu_tree(model.state_dict())
        before = n.tree_hash(checkpoint)
        for name, parameter in model.named_parameters():
            self.assertNotEqual(checkpoint[name].data_ptr(), parameter.data_ptr())
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.add_(17.)
        self.assertEqual(n.tree_hash(checkpoint), before)
        self.assertNotEqual(n.tree_hash(model.state_dict()), before)
        model.load_state_dict(checkpoint)
        self.assertEqual(n.tree_hash(model.state_dict()), before)

    def test_zero_gradient_does_not_skip_adamw_moments(self):
        model = nn.Linear(1, 1, bias=False).double()
        optimizer = n.h.make_optimizer(model)
        model.weight.grad = torch.ones_like(model.weight)
        optimizer.step()
        before = model.weight.detach().clone()
        model.weight.grad = torch.zeros_like(model.weight)
        optimizer.step()
        self.assertFalse(torch.equal(before, model.weight))
        self.assertEqual(int(optimizer.state[model.weight]["step"]), 2)
        self.assertGreater(float(optimizer.state[model.weight]["exp_avg"].abs().max()), 0.)

    def test_independent_selectors_earliest_exact_ties(self):
        result = n.strict_selector_updates(1., .5, {"cross_entropy": 1., "accuracy": .5})
        self.assertEqual(result, {"min_val_ce": False, "max_val_accuracy": False})
        result = n.strict_selector_updates(1., .5, {"cross_entropy": .9, "accuracy": .4})
        self.assertEqual(result, {"min_val_ce": True, "max_val_accuracy": False})
        result = n.strict_selector_updates(1., .5, {"cross_entropy": 1.1, "accuracy": .6})
        self.assertEqual(result, {"min_val_ce": False, "max_val_accuracy": True})
        with self.assertRaises(AssertionError):
            n.strict_selector_updates(1., .5, {"cross_entropy": float("nan"), "accuracy": .6})

    def test_schema_has_no_projector_identity_for_scalar(self):
        raw = torch.tensor([1., 2.], dtype=torch.float64)
        applied, delta = raw * .3, torch.tensor([-.01, -.01], dtype=torch.float64)
        metrics = n.update_metrics(raw, applied, delta)
        self.assertNotIn("leakage_squared_fraction", metrics)
        self.assertNotIn("identity_closure_residual", metrics)
        self.assertNotIn("in_subspace_contribution", metrics)
        self.assertAlmostEqual(metrics["applied_gradient_dot_update"]["value"],
                               .3 * metrics["raw_gradient_dot_update"]["value"])

    def test_synthetic_warmup_core_and_observer_equality(self):
        torch.manual_seed(7)
        initial = nn.Linear(3, 3).double().state_dict()
        gradients = [torch.randn(12, dtype=torch.float64) for _ in range(6)]
        outputs = {}
        for arm in n.ARMS:
            model = nn.Linear(3, 3).double()
            model.load_state_dict(initial)
            optimizer = n.h.make_optimizer(model)
            tracker = n.h.make_tracker(model, optimizer, n.WIDTHS[arm])
            hashes, observer_hashes = [], []
            for step, raw in enumerate(gradients, 1):
                applied, _, _ = n.transform_gradient(arm, tracker, raw)
                self.assertTrue(torch.equal(applied, raw))
                n.h.set_grad(model, applied)
                optimizer.step()
                hashes.append({"step": step, "parameters": n.h.tensor_hash(n.h.flat_params(model)),
                               "raw_gradient": n.h.tensor_hash(raw), "applied_gradient": n.h.tensor_hash(n.h.flat_grad(model))})
                if n.WIDTHS[arm] == 32:
                    observer_hashes.append(n.tree_hash(n.h.tracker_state(tracker)))
            state = n.warmup_snapshot(model, optimizer, tracker)
            # Fixture model construction consumes RNG; compare the training state
            # itself here, while the MNIST pilot also checks identically seeded RNGs.
            state["core"].pop("torch_rng")
            state["core"].pop("cuda_rng")
            state["core"].pop("python_rng")
            result = {"warmup_trajectory_hashes": hashes, "warmup_observer_hashes": observer_hashes}
            outputs[arm] = (result, state)
        for arm in n.ARMS[1:]:
            n.verify_warmup_pair(*outputs["adamw"], *outputs[arm])
        n.verify_warmup_pair(*outputs["estimate32_project32"], *outputs["scalar32_norm"], compare_observer=True)

    def test_measurements_do_not_mutate_state(self):
        model = nn.Linear(1, 1).double()
        optimizer = n.h.make_optimizer(model)
        raw = torch.tensor([3., 4.], dtype=torch.float64)
        candidate = torch.tensor([3., 0.], dtype=torch.float64)
        applied, alpha = n.scalar_gradient(raw, candidate, True)
        n.h.set_grad(model, applied)
        optimizer.step()
        before = n.h.snapshot(model, optimizer, None)
        n.delivered_gradient_metrics("scalar32_norm", raw, candidate, n.h.flat_grad(model), {"alpha": alpha})
        n.update_metrics(raw, applied, torch.tensor([.2, -.3], dtype=torch.float64))
        self.assertTrue(n.h.equal_tree(before, n.h.snapshot(model, optimizer, None)))

    def test_partial_failure_preserves_context(self):
        with tempfile.TemporaryDirectory(prefix="failure-test-", dir=n.HERE) as directory:
            output, record = Path(directory), {}
            context = {"seed": 9877, "arm": "scalar32_norm", "step": 101,
                       "phase": "measurement", "steps_raw": [{"step": 100}], "validation_trajectory": []}
            n.h.preserve_failure(output, record, context, RuntimeError("synthetic failure"))
            saved = json.loads((output / "execution.json").read_text())
            self.assertEqual(saved["status"], "failed_preserved")
            self.assertEqual(saved["failure_context"]["step"], 101)
            self.assertEqual(json.loads((output / "partial-failure.json").read_text())["steps_raw"], [{"step": 100}])

    def test_nonfinite_validation_does_not_poison_failure_record(self):
        for invalid in (float("nan"), float("inf")):
            with tempfile.TemporaryDirectory(prefix="nonfinite-test-", dir=n.HERE) as directory:
                output, record, validations = Path(directory), {}, []
                context = {"seed": 9877, "arm": "scalar32_norm", "step": 100,
                           "phase": "validation_selection", "validation_trajectory": validations}
                try:
                    n.append_validated_evaluation(validations, 100,
                                                  {"cross_entropy": invalid, "accuracy": .5, "count": 5000},
                                                  1., .5)
                except AssertionError as exc:
                    n.h.preserve_failure(output, record, context, exc)
                else:
                    self.fail("Non-finite validation was not rejected")
                self.assertEqual(validations, [])
                saved = json.loads((output / "execution.json").read_text())
                self.assertEqual(saved["status"], "failed_preserved")
                self.assertEqual(saved["failure_context"]["step"], 100)
                self.assertEqual(json.loads((output / "partial-failure.json").read_text())["validation_trajectory"], [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
