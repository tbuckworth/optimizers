"""Tiny synthetic CPU fixtures only; no saved scientific state/data or GPU."""

import copy
import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments import spectral_augmentation_state_core as core


def tiny_parent(zero=False):
    with torch.random.fork_rng(devices=[]):
        model = torch.nn.Linear(2, 2)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[.2, -.1], [-.3, .4]]))
        model.bias.copy_(torch.tensor([.05, -.02]))
        if zero:
            for parameter in model.parameters():
                parameter.zero_()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01,
                                  betas=(.9, .999), eps=1e-8, foreach=False, fused=False)
    for parameter in model.parameters():
        optimizer.state[parameter] = {"step": torch.tensor(100.),
                                      "exp_avg": torch.full_like(parameter, 0 if zero else .02),
                                      "exp_avg_sq": torch.full_like(parameter, .1)}
        parameter.grad = torch.full_like(parameter, 7.)
    return model, optimizer, TinyTracker(model, optimizer)


class TinyTracker:
    """Each private observer must begin at100; candidate input changes its span."""
    def __init__(self, model, optimizer):
        self.model, self.base_optimizer = model, optimizer
        self.param_list = list(model.parameters())
        self.V = torch.eye(sum(parameter.numel() for parameter in self.param_list), 2)
        self.proj_k, self.step_count = None, 100

    def filter_grad(self):
        if self.step_count != 100:
            raise AssertionError("observer reused across candidates")
        self.step_count += 1
        gradient = torch.cat([parameter.grad.flatten() for parameter in self.param_list])
        if gradient[0] < 0:
            self.V = torch.zeros_like(self.V)
            self.V[2, 0], self.V[3, 1] = 1, 1
        applied = self.V @ (self.V.T @ gradient)
        position = 0
        for parameter in self.param_list:
            parameter.grad = applied[position:position + parameter.numel()].reshape_as(parameter).clone()
            position += parameter.numel()


def tree_equal(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(tree_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(tree_equal(x, y) for x, y in zip(left, right))
    return left == right


class GradientAndGeometryTests(unittest.TestCase):
    def test_gradients_match_linear_softmax_oracle_without_mutation(self):
        model, optimizer, tracker = tiny_parent()
        views = torch.tensor([[[.2, .7], [.4, .5], [.3, .4], [.1, .8], [.5, .2]],
                              [[.8, .1], [.7, .2], [.5, .3], [.6, .4], [.9, .0]],
                              [[.3, .3], [.2, .4], [.4, .2], [.5, .1], [.1, .5]]])
        labels = torch.tensor([0, 1, 0])
        before_model, before_optimizer = copy.deepcopy(model.state_dict()), copy.deepcopy(optimizer.state_dict())
        before_grads = [parameter.grad.clone() for parameter in model.parameters()]
        before_rng = torch.get_rng_state().clone()
        result = core.collect_gradients(model, views, labels)
        self.assertEqual(result["per_example"].shape, (3, 5, 6))
        self.assertEqual(result["batch"].shape, (5, 6))
        weight, bias = model.weight.detach().numpy(), model.bias.detach().numpy()
        for example in range(3):
            for view in range(5):
                x = views[example, view].numpy().astype(np.float64)
                logits = weight.astype(np.float64) @ x + bias
                probability = np.exp(logits - logits.max())
                probability /= probability.sum()
                probability[int(labels[example])] -= 1
                expected = np.concatenate((np.outer(probability, x).ravel(), probability))
                np.testing.assert_allclose(result["per_example"][example, view], expected, atol=1e-7, rtol=1e-6)
        np.testing.assert_allclose(result["batch"], result["per_example"].mean(axis=0), atol=1e-7, rtol=1e-6)
        self.assertTrue(tree_equal(model.state_dict(), before_model))
        self.assertTrue(tree_equal(optimizer.state_dict(), before_optimizer))
        self.assertEqual(tracker.step_count, 100)
        for actual, previous in zip(model.parameters(), before_grads):
            self.assertTrue(torch.equal(actual.grad, previous))
        self.assertTrue(torch.equal(before_rng, torch.get_rng_state()))

    def test_finite_grid_identity_and_explicit_zero_denominator(self):
        gradients = np.zeros((2, 5, 3), dtype=np.float32)
        gradients[:, 0, 2] = 2
        gradients[0, 1:, 0] = 1
        gradients[1, 1:, 0] = -1
        gradients[:, 1:, 1] = [1, -1, 1, -1]
        q = np.array([[1], [0], [0]], dtype=np.float32)
        result = core.empirical_geometry(gradients, q)
        self.assertEqual(result["denominators"], {"total": 8, "between": 2, "within": 8})
        self.assertEqual((result["total"]["trace"], result["between"]["trace"], result["within"]["trace"]), (2, 1, 1))
        self.assertEqual(result["total"]["coordinate_retention"], .5)
        self.assertEqual(result["within"]["coordinate_trace"], 0)
        self.assertEqual(result["between"]["coordinate_trace"], 1)
        self.assertIsNone(result["original_centered"]["coordinate_retention"])
        self.assertIsNone(result["translated_mean"]["coordinate_retention"])
        self.assertEqual(result["original_mean"]["trace"], 4)
        self.assertTrue(result["identity_pass"])
        # Distinguish Q^T coordinate energy from actual QQ^T energy when Q is
        # deliberately nonorthogonal. Neither quantity is clipped into [0,1].
        imperfect = core.empirical_geometry(gradients, 2 * q)
        self.assertEqual(imperfect["between"]["coordinate_trace"], 4)
        self.assertEqual(imperfect["between"]["operator_trace"], 16)
        self.assertEqual(imperfect["basis_gram_max_abs_error"], 3)
        self.assertTrue(imperfect["identity_pass"])

    def test_invalid_gradient_grid_rejected(self):
        for value in (np.zeros((0, 5, 3)), np.zeros((2, 1, 3)), np.full((2, 5, 3), np.nan)):
            with self.assertRaises(ValueError):
                core.empirical_geometry(value, np.zeros((3, 1)))


class ActionTests(unittest.TestCase):
    def test_private_actual_adam_endpoints_matching_and_fractional_path(self):
        model, optimizer, tracker = tiny_parent()
        gradients = np.array([[1, -2, 3, -4, 5, -6], [-3, 2, -1, 4, -2, 1]], dtype=np.float32)
        before_model, before_optimizer = copy.deepcopy(model.state_dict()), copy.deepcopy(optimizer.state_dict())
        before_basis = tracker.V.clone()
        before_grads = [p.grad.clone() for p in model.parameters()]
        result = core.propose_actions(model, optimizer, tracker, gradients)
        self.assertEqual(result["after_parameters"].shape, (2, 5, 2, 6))
        self.assertEqual(result["basis_after"].shape, (2, 6, 32))
        np.testing.assert_array_equal(result["adam_steps"], [100, 100])
        np.testing.assert_array_equal(result["param_sizes"], [4, 2])
        np.testing.assert_array_equal(result["applied_grads"][:, 0], gradients)
        np.testing.assert_array_equal(result["applied_grads"][0, 1], [1, -2, 0, 0, 0, 0])
        np.testing.assert_array_equal(result["applied_grads"][1, 1], [0, 0, -1, 4, 0, 0])
        theta = torch.from_numpy(result["theta"])
        decay = theta * (1 - .001 * .01)
        for view in range(2):
            # Independent direct reference for raw's full actual Adam endpoint.
            direct_model, direct_optimizer = copy.deepcopy((model, optimizer))
            offset = 0
            for parameter in direct_model.parameters():
                parameter.grad = torch.tensor(gradients[view, offset:offset + parameter.numel()]).reshape_as(parameter)
                offset += parameter.numel()
            direct_optimizer.step()
            actual = torch.cat([p.detach().flatten() for p in direct_model.parameters()]).numpy()
            np.testing.assert_array_equal(result["after_parameters"][view, 0, 0], actual)
            np.testing.assert_array_equal(result["after_parameters"][view, 4, 0], decay.numpy())
            for action in range(5):
                full = torch.from_numpy(result["after_parameters"][view, action, 0])
                np.testing.assert_array_equal(result["after_parameters"][view, action, 1],
                                              (theta + .1 * (full - theta)).numpy())
            for action, target in ((2, 1), (3, 0)):
                source_norm = np.linalg.norm((result["after_parameters"][view, action, 0] - decay.numpy()).astype(np.float64))
                target_norm = np.linalg.norm(result["planned_data"][view, target].astype(np.float64))
                self.assertLessEqual(abs(source_norm - target_norm), core.MATCH_ATOL + core.MATCH_RTOL * target_norm)
        self.assertTrue(tree_equal(model.state_dict(), before_model))
        self.assertTrue(tree_equal(optimizer.state_dict(), before_optimizer))
        self.assertTrue(torch.equal(tracker.V, before_basis))
        self.assertEqual(tracker.step_count, 100)
        for parameter, grad in zip(model.parameters(), before_grads):
            self.assertTrue(torch.equal(parameter.grad, grad))

    def test_zero_norm_matches_unavailable_without_suppressing_actual_actions(self):
        model, optimizer, tracker = tiny_parent(zero=True)
        result = core.propose_actions(model, optimizer, tracker, np.zeros((1, 6), dtype=np.float32))
        np.testing.assert_array_equal(result["valid"], [[True, True, False, False, True]])
        self.assertTrue(np.all(result["after_parameters"][:, 2:4] == 0))
        self.assertTrue(np.all(result["scales"][:, 2:4] == 0))

    def test_rescale_cap_is_unavailable_not_clipped(self):
        model, optimizer, tracker = tiny_parent(zero=True)
        gradients = np.array([[1e-4, 0, 1, 1, 1, 1]], dtype=np.float32)
        result = core.propose_actions(model, optimizer, tracker, gradients)
        self.assertTrue(result["valid"][0, 2])
        self.assertFalse(result["valid"][0, 3])
        self.assertEqual(result["scales"][0, 3], 0)
        self.assertGreater(np.linalg.norm(result["planned_data"][0, 0].astype(np.float64)) /
                           np.linalg.norm(result["planned_data"][0, 1].astype(np.float64)), 100)

    def test_decay_only_is_not_zero_gradient_adam_with_carried_moments(self):
        model, optimizer, tracker = tiny_parent()
        result = core.propose_actions(model, optimizer, tracker, np.zeros((1, 6), dtype=np.float32))
        self.assertFalse(np.array_equal(result["after_parameters"][0, 0, 0], result["after_parameters"][0, 4, 0]))

    def test_missing_adam_state_does_not_mutate_parent(self):
        model, optimizer, tracker = tiny_parent()
        optimizer.state.clear()
        with self.assertRaisesRegex(ValueError, "populated canonical Adam state"):
            core.propose_actions(model, optimizer, tracker, np.zeros((1, 6), dtype=np.float32))
        self.assertEqual(len(optimizer.state), 0)

    def test_each_cloned_adam_clock_is_checked_against_its_parent(self):
        model, optimizer, tracker = tiny_parent()
        parameters = list(model.parameters())
        optimizer.state[parameters[1]]["step"].fill_(103)
        gradients = np.ones((1, 6), dtype=np.float32)
        result = core.propose_actions(model, optimizer, tracker, gradients)
        np.testing.assert_array_equal(result["adam_steps"], [100, 103])
        actual_step = torch.optim.AdamW.step
        for corrupt_call in (1, 2):  # Independently exercise raw and native clones.
            calls = 0

            def wrong_clock(clone, *args, **kwargs):
                nonlocal calls
                calls += 1
                result = actual_step(clone, *args, **kwargs)
                if calls == corrupt_call:
                    parameter = clone.param_groups[0]["params"][1]
                    clone.state[parameter]["step"].add_(1)
                return result

            with self.subTest(corrupt_call=corrupt_call), \
                 patch.object(torch.optim.AdamW, "step", new=wrong_clock):
                with self.assertRaisesRegex(ValueError, "each cloned Adam counter must advance exactly once"):
                    core.propose_actions(model, optimizer, tracker, gradients)
        self.assertEqual([int(optimizer.state[p]["step"].item()) for p in parameters], [100, 103])
        self.assertEqual(tracker.step_count, 100)

    def test_objective_logits_utilities_and_parent_preservation(self):
        model, optimizer, tracker = tiny_parent()
        before = copy.deepcopy(model.state_dict())
        views = torch.tensor([[[.2, .7], [.8, .1], [.3, .3]],
                              [[.4, .5], [.7, .2], [.2, .4]]])
        labels = torch.tensor([0, 1, 0])
        proposals = core.propose_actions(model, optimizer, tracker,
                                         np.array([[1, -2, 3, -4, 5, -6]], dtype=np.float32))
        result = core.measure_actions(model, proposals, views, labels)
        self.assertEqual(result["after_logits"].shape, (1, 5, 2, 2, 3, 2))
        self.assertEqual(result["objective_grads"].shape, (2, 6))
        theta = proposals["theta"].astype(np.float64)
        for action in range(5):
            for scale in range(2):
                after = proposals["after_parameters"][0, action, scale]
                utility = -result["objective_grads"].astype(np.float64) @ (after.astype(np.float64) - theta)
                np.testing.assert_allclose(result["derivative_utility"][0, action, scale], utility, atol=1e-15)
                for objective in range(2):
                    expected = views[objective].numpy() @ after[:4].reshape(2, 2).T + after[4:]
                    np.testing.assert_allclose(result["after_logits"][0, action, scale, objective], expected, atol=1e-7)
        np.testing.assert_array_equal(result["data_derivative_utility"][:, 4], 0)
        self.assertTrue(tree_equal(before, model.state_dict()))
        with self.assertRaisesRegex(ValueError, "fixed full/.1"):
            core.measure_actions(model, proposals, views, labels, scales=(1., .2))


if __name__ == "__main__":
    unittest.main()
