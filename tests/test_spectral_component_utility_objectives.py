"""Fabricated CPU tensors only; no neural models, datasets or saved science."""
import importlib
import json
import math
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as functional

from experiments import spectral_component_utility_objectives as obj


def fixture_grid(n=4, views=3, classes=3):
    return ((torch.arange(n * views * classes, dtype=torch.float32).reshape(n, views, classes) * 7) % 13 - 6) / 2


def scalar_reference(grid, true, assigned):
    """Separate scalar loops/shifted-exp formula, not producer helpers."""
    values = grid.detach().numpy().astype(np.float64)
    true, assigned = true.tolist(), assigned.tolist()
    n, views, k = values.shape

    def lse(row):
        largest = max(row)
        return largest + math.log(math.fsum(math.exp(float(v) - largest) for v in row))

    rows = []
    for i in range(n):
        mean = [math.fsum(float(values[i, v, c]) for v in range(views)) / views for c in range(k)]
        mean_lse = lse(mean)
        st = mean_lse - mean[true[i]]
        su = mean_lse - math.fsum(mean) / k
        q = [.9 / k + (.1 if c == true[i] else 0) for c in range(k)]
        epsilon = [(1 if c == assigned[i] else 0) - q[c] for c in range(k)]
        rows.append({"S": math.fsum(q[c] * (mean_lse - mean[c]) for c in range(k)),
                     "F": -math.fsum(epsilon[c] * mean[c] for c in range(k)),
                     "C": math.fsum(lse(row) for row in values[i]) / views - mean_lse,
                     "L": math.fsum(lse(row) - row[assigned[i]] for row in values[i]) / views,
                     "S_true": st, "S_uniform": su})
    return {key: math.fsum(row[key] for row in rows) / n for key in obj.OBJECTIVE_KEYS}


class ObjectiveFixtures(unittest.TestCase):
    def setUp(self):
        self.grid = fixture_grid()
        self.true = torch.tensor([0, 1, 2, 0])
        self.assigned = torch.tensor([2, 1, 0, 2])

    def test_import_has_no_computation_or_environment_side_effect(self):
        with patch.object(torch, "load", side_effect=AssertionError("read")), \
                patch.object(torch, "manual_seed", side_effect=AssertionError("seed")), \
                patch.object(torch.cuda, "is_available", side_effect=AssertionError("GPU query")), \
                patch.object(torch, "logsumexp", side_effect=AssertionError("work on import")):
            importlib.reload(obj)

    def test_direct_scalar_identity_independent_reference_and_exact_keys(self):
        result = obj.objective_tensors(self.grid, self.true, self.assigned)
        expected = scalar_reference(self.grid, self.true, self.assigned)
        self.assertEqual(tuple(result), obj.OBJECTIVE_KEYS)
        for name, value in result.items():
            self.assertEqual(value.dtype, torch.float64)
            self.assertEqual(value.ndim, 0)
            self.assertAlmostEqual(value.item(), expected[name], delta=1e-10)
        self.assertAlmostEqual((result["S"] + result["F"] + result["C"]).item(), result["L"].item(), delta=1e-10)
        self.assertEqual(result["S"].item(), (.1 * result["S_true"] + .9 * result["S_uniform"]).item())

    def test_parameter_gradients_direct_ce_identity_and_sentinel_preservation(self):
        parameter = torch.linspace(-.5, .5, 7, dtype=torch.float32, requires_grad=True)
        coefficient = torch.arange(36 * 7, dtype=torch.float32).reshape(36, 7) / 300
        grid = (coefficient @ parameter.sin()).reshape(4, 3, 3) + self.grid
        parameter.grad = torch.full_like(parameter, 8.25)
        sentinel = parameter.grad
        initial_values = parameter.detach().clone()
        metadata = (grid.shape, grid.stride(), grid.device, grid.dtype, grid._version)
        objectives = obj.objective_tensors(grid, self.true, self.assigned)
        gradients = obj.flat_objective_gradients({k: objectives[k] for k in ("S", "F", "C", "L")},
                                                (parameter,), retain_graph=True)
        direct = functional.cross_entropy(grid.double().reshape(-1, 3), self.assigned[:, None].expand(-1, 3).reshape(-1))
        independent = torch.autograd.grad(direct, parameter)[0]
        torch.testing.assert_close(gradients["L"], independent, atol=1e-6, rtol=5e-5)
        residual = (gradients["S"].double() + gradients["F"].double() + gradients["C"].double() - independent.double()).norm()
        self.assertLessEqual(residual.item(), 1e-6 + 5e-5 * independent.double().norm().item())
        self.assertIs(parameter.grad, sentinel)
        self.assertTrue(torch.equal(parameter.grad, torch.full_like(parameter, 8.25)))
        self.assertTrue(torch.equal(parameter.detach(), initial_values))
        self.assertEqual(metadata, (grid.shape, grid.stride(), grid.device, grid.dtype, grid._version))
        self.assertTrue(all(v.dtype == torch.float32 and not v.requires_grad for v in gradients.values()))

    def test_logit_gradient_formula_includes_moving_mean_and_label_force(self):
        grid = self.grid.clone().requires_grad_()
        grid.grad = torch.full_like(grid, -7.)
        result = obj.objective_tensors(grid, self.true, self.assigned)
        gradients = obj.flat_objective_gradients({k: result[k] for k in ("S", "F", "C", "L")}, (grid,))
        values = self.grid.double()
        mean_p = values.mean(dim=1).softmax(dim=-1)
        view_p = values.softmax(dim=-1)
        q = .1 * functional.one_hot(self.true, 3).double() + .9 / 3
        target = functional.one_hot(self.assigned, 3).double()
        expected = {
            "S": ((mean_p - q)[:, None, :] / 12).expand(-1, 3, -1),
            "F": ((q - target)[:, None, :] / 12).expand(-1, 3, -1),
            "C": (view_p - mean_p[:, None, :]) / 12,
            "L": (view_p - target[:, None, :]) / 12,
        }
        for key, value in expected.items():
            torch.testing.assert_close(gradients[key].double(), value.reshape(-1), atol=1e-7, rtol=5e-5)
        self.assertGreater(gradients["C"].norm().item(), .01)
        self.assertTrue(torch.equal(grid.grad, torch.full_like(grid, -7.)))

    def test_one_view_consistency_value_and_gradient_are_zero(self):
        for k in (2, 3, 10):
            grid = fixture_grid(2, 1, k).requires_grad_()
            true, assigned = torch.tensor([0, 1]), torch.tensor([1, 0])
            result = obj.objective_tensors(grid, true, assigned)
            gradient = obj.flat_objective_gradients({"C": result["C"]}, (grid,))["C"]
            self.assertEqual(result["C"].item(), 0.)
            self.assertEqual(torch.count_nonzero(gradient).item(), 0)

    def test_consistency_is_label_independent_in_value_and_gradient(self):
        grid = self.grid.clone().requires_grad_()
        first = obj.objective_tensors(grid, self.true, self.assigned)
        second = obj.objective_tensors(grid, (self.true + 1) % 3, (self.assigned + 2) % 3)
        self.assertTrue(torch.equal(first["C"], second["C"]))
        g1 = torch.autograd.grad(first["C"], grid, retain_graph=True)[0]
        g2 = torch.autograd.grad(second["C"], grid)[0]
        self.assertTrue(torch.equal(g1, g2))
        self.assertNotEqual(first["F"].item(), second["F"].item())

    def test_large_per_view_common_offsets_are_removed_before_averaging(self):
        grid = self.grid * 2  # Integer differences survive the chosen FP32 offsets.
        offsets = torch.tensor([[2**20, -2**21, 2**19], [-2**20, 2**21, 2**18],
                                [2**22, -2**22, 2**20], [2**18, 2**19, -2**20]], dtype=torch.float32)
        shifted = grid + offsets[:, :, None]
        first = obj.objective_tensors(grid, self.true, self.assigned)
        second = obj.objective_tensors(shifted, self.true, self.assigned)
        for key in first:
            self.assertTrue(torch.equal(first[key], second[key]), key)
        self.assertEqual(obj.metric_report(grid, self.true, self.assigned, original_view_index=1),
                         obj.metric_report(shifted, self.true, self.assigned, original_view_index=1))

    def test_r_losses_are_original_and_per_view_not_mean_logit_ce(self):
        r_grid = torch.tensor([[[4., 0.], [0., 8.], [0., 8.]],
                               [[0., 4.], [6., 0.], [6., 0.]]])
        labels = torch.tensor([0, 1])
        result = obj.objective_tensors(r_grid, labels, labels, r_grid[:, 0], r_grid, labels)
        direct = functional.cross_entropy(r_grid.double().reshape(-1, 2), labels[:, None].expand(-1, 3).reshape(-1))
        mean_logit = functional.cross_entropy(r_grid.double().mean(dim=1), labels)
        self.assertAlmostEqual(result["H_T"].item(), direct.item(), delta=1e-10)
        self.assertGreater(abs(result["H_T"].item() - mean_logit.item()), .1)
        self.assertAlmostEqual(result["H_O"].item(), functional.cross_entropy(r_grid[:, 0].double(), labels).item(), delta=1e-10)

    def test_flat_gradients_keep_parameter_order_and_reject_disconnected(self):
        a = torch.tensor([1., 2.], requires_grad=True)
        b = torch.tensor([3., 4., 5.], requires_grad=True)
        value = (a.double().square().sum() + 3 * b.double().sum())
        gradient = obj.flat_objective_gradients({"fixture": value}, (b, a))["fixture"]
        self.assertEqual(gradient.tolist(), [3., 3., 3., 2., 4.])
        unused = torch.tensor([0.], requires_grad=True)
        with self.assertRaises(RuntimeError):
            obj.flat_objective_gradients({"fixture": a.double().sum()}, (a, unused))

    def test_exact_25_view_10_class_objectives_and_all_six_gradients(self):
        parameter = torch.tensor([.125, -.25], requires_grad=True)
        coefficient = torch.linspace(-1, 1, 750).reshape(3, 25, 10)
        class_scale = torch.linspace(-.5, .5, 10)
        grid = fixture_grid(3, 25, 10) + parameter[0] * coefficient + parameter[1] * class_scale
        r_grid = fixture_grid(2, 25, 10) + parameter[0] * coefficient[:2].flip(1) + parameter[1] * class_scale
        true, assigned, r_true = torch.tensor([0, 5, 9]), torch.tensor([4, 5, 2]), torch.tensor([1, 7])
        values = obj.objective_tensors(grid, true, assigned, r_grid[:, 12], r_grid, r_true)
        expected = scalar_reference(grid, true, assigned)
        for key in obj.OBJECTIVE_KEYS:
            self.assertAlmostEqual(values[key].item(), expected[key], delta=1e-10)
        direct = {
            "L": functional.cross_entropy(grid.double().reshape(-1, 10), assigned[:, None].expand(-1, 25).reshape(-1)),
            "H_O": functional.cross_entropy(r_grid[:, 12].double(), r_true),
            "H_T": functional.cross_entropy(r_grid.double().reshape(-1, 10), r_true[:, None].expand(-1, 25).reshape(-1)),
        }
        references = {key: torch.autograd.grad(value, parameter, retain_graph=True)[0] for key, value in direct.items()}
        gradients = obj.flat_objective_gradients({key: values[key] for key in obj.GRADIENT_KEYS}, (parameter,))
        self.assertEqual(tuple(gradients), obj.GRADIENT_KEYS)
        for key, reference in references.items():
            torch.testing.assert_close(gradients[key], reference, atol=1e-6, rtol=5e-5)
        torch.testing.assert_close(gradients["S"] + gradients["F"] + gradients["C"],
                                   references["L"], atol=1e-6, rtol=5e-5)
        self.assertIsNone(parameter.grad)

    def test_invalid_objective_inputs_are_rejected_without_mutation(self):
        for grid in (self.grid.double(), self.grid[:, 0], self.grid[:0], self.grid[:, :0],
                     self.grid[:, :, :1], torch.full_like(self.grid, float("nan"))):
            with self.assertRaises(ValueError):
                obj.objective_tensors(grid, self.true, self.assigned)
        for labels in (self.true.float(), self.true[:-1], torch.tensor([0, 1, 2, 3])):
            with self.assertRaises(ValueError):
                obj.objective_tensors(self.grid, labels, self.assigned)
        with self.assertRaisesRegex(ValueError, "supplied together"):
            obj.objective_tensors(self.grid, self.true, self.assigned, r_true=self.true)
        for index in (True, -1, 3):
            with self.assertRaises(ValueError):
                obj.metric_report(self.grid, self.true, self.assigned, original_view_index=index)
        leaf = torch.ones(2, requires_grad=True)
        for parameters in ((leaf, leaf), (leaf.double(),), ()):
            with self.assertRaises(ValueError):
                obj.flat_objective_gradients({"x": leaf.double().sum()}, parameters)


class MetricFixtures(unittest.TestCase):
    def setUp(self):
        self.grid = torch.tensor([[[4., 0.], [0., 8.], [0., 8.]],
                                  [[0., 4.], [6., 0.], [6., 0.]]])
        self.true = torch.tensor([0, 1])
        self.assigned = torch.tensor([1, 1])

    def test_original_view_average_and_wrong_subset_denominators(self):
        result = obj.metric_report(self.grid, self.true, self.assigned, original_view_index=0,
                                   r_original=self.grid[:, 0], r_grid=self.grid, r_true=self.true)
        self.assertEqual(result["I"]["original"]["true"]["accuracy"], 1.)
        self.assertEqual(result["I"]["per_view"]["true"]["accuracy"], 1 / 3)
        self.assertEqual(result["I"]["per_view"]["true"]["count"], 6)
        self.assertEqual(result["I"]["per_view"]["true"]["example_count"], 2)
        self.assertEqual(result["I"]["wrong_original"]["assigned"]["accuracy"], 0.)
        self.assertEqual(result["I"]["wrong_per_view"]["assigned"]["accuracy"], 2 / 3)
        self.assertEqual(result["I"]["wrong_per_view"]["assigned"]["count"], 3)
        self.assertEqual(result["metadata"]["I_wrong_examples"], 1)
        self.assertEqual(result["R"]["per_view"]["true"], result["I"]["per_view"]["true"])
        for panel in result["I"].values():
            for metric in panel.values():
                self.assertEqual(metric["ce"], metric["ce_sum"] / metric["count"])
        json.dumps(result, allow_nan=False)

    def test_empty_wrong_subset_has_no_invented_mean(self):
        result = obj.metric_report(self.grid, self.true, self.true, original_view_index=0)
        self.assertNotIn("R", result)
        for panel in ("wrong_original", "wrong_per_view"):
            for metric in result["I"][panel].values():
                self.assertFalse(metric["available"])
                self.assertEqual((metric["example_count"], metric["count"], metric["correct"], metric["ce_sum"]), (0, 0, 0, 0.))
                self.assertIsNone(metric["accuracy"])
                self.assertIsNone(metric["ce"])
        json.dumps(result, allow_nan=False)

    def test_noncontiguous_inputs_and_metadata_are_preserved(self):
        grid = fixture_grid(2, 3, 4)[:, :, ::2]
        before = grid.clone()
        metadata = (grid.shape, grid.stride(), grid.data_ptr(), grid._version, grid.requires_grad)
        obj.metric_report(grid, self.true, self.assigned, original_view_index=2)
        self.assertTrue(torch.equal(grid, before))
        self.assertEqual(metadata, (grid.shape, grid.stride(), grid.data_ptr(), grid._version, grid.requires_grad))


class PathFixtures(unittest.TestCase):
    def test_full_endpoint_preserved_when_reconstruction_would_round(self):
        parent, endpoint = torch.tensor([float(2**25)]), torch.tensor([1.])
        self.assertEqual((parent + (endpoint - parent)).item(), 0.)
        point = obj.materialize_path(parent, endpoint, 1.)
        self.assertTrue(torch.equal(point, endpoint))
        self.assertNotEqual(point.data_ptr(), endpoint.data_ptr())
        self.assertEqual(obj.actual_delta(parent, point).item(), 1 - 2**25)
        self.assertNotEqual(obj.actual_delta(parent, point).item(), (point - parent).double().item())

    def test_fraction_uses_actual_materialized_fp32_delta_not_scaled_full_delta(self):
        parent = torch.tensor([float(2**25), 1., -1.])
        endpoint = torch.tensor([float(2**25 + 4), 1. + 2**-23, -1. + 2**-24])
        point = obj.materialize_path(parent, endpoint, .1)
        self.assertTrue(torch.equal(point, parent))
        self.assertTrue(torch.equal(obj.actual_delta(parent, point), torch.zeros(3, dtype=torch.float64)))
        self.assertGreater((.1 * obj.actual_delta(parent, endpoint)).norm().item(), 0)
        expected = parent.numpy() + np.float32(.1) * (endpoint.numpy() - parent.numpy())
        np.testing.assert_array_equal(point.numpy(), expected)

    def test_fraction_matched_decay_and_additive_deltas(self):
        parent = torch.tensor([1., -3., 9.])
        endpoint = torch.tensor([1.005, -3.006, 8.995])
        decay = parent * (1 - .001 * .01)
        originals = [value.clone() for value in (parent, endpoint, decay)]
        for fraction in (1., .1):
            record = obj.path_accounting(parent, endpoint, decay, fraction)
            self.assertTrue(torch.equal(record["total_delta"], record["data_delta"] + record["decay_delta"]))
            self.assertTrue(torch.equal(record["data_delta"], record["point"].double() - record["decay_point"].double()))
            self.assertEqual(record["total_norm"], record["total_delta"].norm().item())
        for value, before in zip((parent, endpoint, decay), originals):
            self.assertTrue(torch.equal(value, before))

    def test_signed_dot_fp64_products_compensated_sum_and_orientation(self):
        gradient = torch.tensor([float(2**60), 1., -float(2**60)])
        delta = torch.ones(3, dtype=torch.float64)
        self.assertEqual(obj.signed_utility(gradient, delta), -1.)
        self.assertEqual(obj.signed_utility(torch.tensor([2., -3.]), torch.tensor([-1., 2.], dtype=torch.float64)), 8.)

    def test_path_and_dot_invalids(self):
        parent = torch.tensor([1., 2.])
        for fraction in (True, -1, 1.1, float("nan")):
            with self.assertRaises(ValueError):
                obj.materialize_path(parent, parent, fraction)
        for endpoint in (parent.double(), parent[:1], torch.full_like(parent, float("inf"))):
            with self.assertRaises(ValueError):
                obj.actual_delta(parent, endpoint)
        with self.assertRaises(ValueError):
            obj.signed_utility(parent, parent)
        with self.assertRaises(ValueError):
            obj.materialize_path(torch.tensor([torch.finfo(torch.float32).max]),
                                 torch.tensor([-torch.finfo(torch.float32).max]), .1)


if __name__ == "__main__":
    unittest.main()
