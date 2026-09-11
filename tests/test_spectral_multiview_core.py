"""Tiny fabricated CPU models only; no MNIST, checkpoint, CUDA or real run."""

import copy
import importlib
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

from experiments import spectral_multiview_core as core
from spectral_filter import SpectralGradientFilter


def tiny_parent(policy="native1", warmup=2):
    # Deliberately tiny width/rank and fixture warmup, not changed run constants.
    model = core.neural_core.make_model(17, "cpu", width=3)
    optimizer = core.neural_core.make_optimizer(model)
    tracker = None if policy in ("raw1", "raw4") else SpectralGradientFilter(
        model, optimizer, rank=2, warmup=warmup, stabilize_every=2,
        stable_update=True, decay=.99)
    return model, optimizer, tracker


def inputs():
    values = torch.linspace(0, 1, 4 * 3 * 784, dtype=torch.float32).reshape(4, 3, 784)
    return values, torch.tensor([0, 1, 2], dtype=torch.int64)


def state_equal(left, right):
    return core.neural_core.equal_tree(left, right)


def manual_gradient(model, value, labels):
    model.zero_grad(set_to_none=True)
    loss = F.cross_entropy(model(value), labels)
    loss.backward()
    return core.neural_core.flat_grad(model), float(loss.detach())


class MultiviewUpdateTests(unittest.TestCase):
    def assert_learning_equal(self, left, right):
        self.assertTrue(state_equal(dict(left[0].state_dict()), dict(right[0].state_dict())))
        self.assertTrue(state_equal(left[1].state_dict(), right[1].state_dict()))
        self.assertTrue(torch.equal(core.neural_core.flat_grad(left[0]),
                                    core.neural_core.flat_grad(right[0])))

    def test_import_is_inert(self):
        with patch.object(core.neural_core, "make_model", side_effect=AssertionError("model creation")), \
                patch.object(torch, "load", side_effect=AssertionError("checkpoint read")), \
                patch.object(torch.cuda, "is_available", side_effect=AssertionError("GPU inspection")):
            importlib.reload(core)

    def test_native1_bit_exact_with_canonical_backward_filter_adam(self):
        actual = tiny_parent()
        reference = copy.deepcopy(actual)
        views, labels = inputs()
        for step in range(1, 7):
            view = views[(step - 1) % 4: (step - 1) % 4 + 1]
            gradient, loss = manual_gradient(reference[0], view[0], labels)
            reference[2].filter_grad()
            reference[1].step()
            record = core.training_update(*actual, view, labels, "native1")
            self.assert_learning_equal(actual, reference)
            self.assertTrue(state_equal(core.neural_core._tracker_state(actual[2]),
                                       core.neural_core._tracker_state(reference[2])))
            self.assertEqual(record["losses"][0], loss)
            self.assertEqual(record["observation_norm"], float(gradient.double().norm()))
            self.assertEqual((record["adam_steps"], record["observer_steps"]), (step, step))
        self.assertGreater(actual[2].stabilization_count, 0)

    def test_raw1_exact_with_canonical_backward_adam_and_no_observer(self):
        actual = tiny_parent("raw1")
        reference = copy.deepcopy(actual)
        views, labels = inputs()
        for step in range(1, 4):
            manual_gradient(reference[0], views[0], labels)
            reference[1].step()
            record = core.training_update(*actual, views[:1], labels, "raw1")
            self.assert_learning_equal(actual, reference)
            self.assertEqual(record["adam_steps"], step)
            self.assertEqual((record["observer_steps"], record["basis_rank"],
                              record["observation_norm"]), (0, 0, 0.0))

    def test_observer4_matches_one_canonical_ingest_of_mean_then_g1_projection(self):
        actual = tiny_parent("observer4")
        reference = copy.deepcopy(actual)
        views, labels = inputs()
        for step in range(1, 5):
            gradients, losses = zip(*(manual_gradient(reference[0], view, labels) for view in views))
            mean = gradients[0].clone()
            for gradient in gradients[1:]:
                mean.add_(gradient)
            mean.div_(4)
            core.neural_core.set_grad(reference[0], mean)
            reference[2].filter_grad()  # Canonical ingest on mean; its delivery is discarded.
            applied = reference[2]._project_gradient(gradients[0]) if step > 2 else gradients[0]
            core.neural_core.set_grad(reference[0], applied)
            reference[1].step()
            with patch.object(actual[2], "filter_grad", side_effect=AssertionError("second ingest")):
                record = core.training_update(*actual, views, labels, "observer4")
            self.assert_learning_equal(actual, reference)
            self.assertTrue(state_equal(core.neural_core._tracker_state(actual[2]),
                                       core.neural_core._tracker_state(reference[2])))
            np.testing.assert_array_equal(record["losses"], np.asarray(losses, dtype=np.float64))
            self.assertEqual(record["observation_norm"], float(mean.double().norm()))
            self.assertEqual(record["delivery_norm"], float(gradients[0].double().norm()))
            self.assertEqual(record["observer_steps"], step)

    def test_raw4_exact_separate_view_mean_and_single_adam_update(self):
        actual = tiny_parent("raw4")
        reference = copy.deepcopy(actual)
        views, labels = inputs()
        for step in range(1, 3):
            grads = [manual_gradient(reference[0], view, labels)[0] for view in views]
            mean = grads[0].clone()
            for gradient in grads[1:]:
                mean.add_(gradient)
            mean.div_(4)
            core.neural_core.set_grad(reference[0], mean)
            reference[1].step()
            record = core.training_update(*actual, views, labels, "raw4")
            self.assert_learning_equal(actual, reference)
            self.assertEqual(record["applied_norm"], float(mean.double().norm()))
            self.assertEqual(record["gradient_evaluations"], 4)
            self.assertEqual(record["adam_steps"], step)

    def test_warmup_delivery_identity_for_three_arms_with_distinct_observer(self):
        parents = {policy: tiny_parent(policy) for policy in ("raw1", "native1", "observer4", "raw4")}
        views, labels = inputs()
        for step in range(2):
            for policy, parent in parents.items():
                selected = views if policy in ("observer4", "raw4") else views[:1]
                core.training_update(*parent, selected, labels, policy)
            self.assert_learning_equal(parents["raw1"], parents["native1"])
            self.assert_learning_equal(parents["raw1"], parents["observer4"])
        self.assertFalse(torch.equal(parents["native1"][2].grad_mean,
                                     parents["observer4"][2].grad_mean))
        self.assertFalse(state_equal(parents["raw1"][1].state_dict(), parents["raw4"][1].state_dict()))

    def test_sequential_fp32_sum_no_grad_accumulation_or_midview_update(self):
        views, labels = inputs()
        for policy in ("observer4", "raw4"):
            parent = tiny_parent(policy)
            before = core.neural_core.flat_params(parent[0])
            values = [2.0**24, 1.0, -(2.0**24), 4.0]
            visited = []

            def artificial_grad(loss, parameters, **kwargs):
                self.assertEqual(kwargs, {"retain_graph": False, "create_graph": False, "allow_unused": False})
                self.assertTrue(all(parameter.grad is None for parameter in parameters))
                self.assertTrue(torch.equal(before, core.neural_core.flat_params(parent[0])))
                value = values[len(visited)]
                visited.append(value)
                return tuple(torch.full_like(parameter, value) for parameter in parameters)

            with patch.object(torch.autograd, "grad", side_effect=artificial_grad), \
                    patch.object(torch.Tensor, "backward", side_effect=AssertionError("backward accumulation")):
                record = core.training_update(*parent, views, labels, policy)
            self.assertEqual(len(visited), 4)
            expected = torch.full_like(before, 1.0 if policy == "raw4" else 2.0**24)
            self.assertTrue(torch.equal(core.neural_core.flat_grad(parent[0]), expected))
            if parent[2] is not None:
                self.assertTrue(torch.equal(parent[2].grad_mean, torch.ones_like(before)))
            self.assertEqual(record["adam_steps"], 1)

    def test_event_order_four_gradients_observe_project_then_adam(self):
        parent = tiny_parent("observer4", warmup=0)
        views, labels = inputs()
        events = []
        gradient_call, observe_call = torch.autograd.grad, parent[2]._update_svd
        projection_call, step_call = parent[2]._project_gradient, parent[1].step

        def gradient(*args, **kwargs):
            events.append("gradient")
            return gradient_call(*args, **kwargs)

        def observe(value):
            events.append("observe")
            return observe_call(value)

        def project(value):
            events.append("project")
            return projection_call(value)

        def step():
            events.append("Adam")
            return step_call()

        with patch.object(torch.autograd, "grad", side_effect=gradient), \
                patch.object(parent[2], "_update_svd", side_effect=observe), \
                patch.object(parent[2], "_project_gradient", side_effect=project), \
                patch.object(parent[1], "step", side_effect=step):
            core.training_update(*parent, views, labels, "observer4")
        self.assertEqual(events, ["gradient"] * 4 + ["observe", "project", "Adam"])

    def test_record_schema_zero_fill_and_actual_rounded_decay_data_norm(self):
        views, labels = inputs()
        for policy in core.POLICIES:
            parent = tiny_parent(policy)
            theta = core.neural_core.flat_params(parent[0])
            decay = theta.clone().mul_(1 - .001 * .01)
            selected = views if policy in ("observer4", "raw4") else views[:1]
            before_views, before_labels = selected.clone(), labels.clone()
            before_rng = torch.get_rng_state().clone()
            record = core.training_update(*parent, selected, labels, policy)
            self.assertEqual(tuple(record), core.STREAM_KEYS)
            self.assertEqual((record["losses"].shape, record["losses"].dtype), ((4,), np.dtype("float64")))
            self.assertTrue(np.isfinite(record["losses"]).all())
            self.assertTrue((record["losses"][:len(selected)] > 0).all())
            self.assertTrue((record["losses"][len(selected):] == 0).all())
            for key in ("gradient_evaluations", "observer_steps", "adam_steps", "basis_rank"):
                self.assertIs(type(record[key]), int)
            for key in ("observation_norm", "delivery_norm", "applied_norm", "data_step_norm"):
                self.assertIs(type(record[key]), float)
                self.assertTrue(np.isfinite(record[key]) and record[key] >= 0)
            actual_data = core.neural_core.flat_params(parent[0]) - decay
            self.assertEqual(record["data_step_norm"], float(actual_data.double().norm()))
            self.assertTrue(torch.equal(before_views, selected) and torch.equal(before_labels, labels))
            self.assertTrue(torch.equal(before_rng, torch.get_rng_state()))

    def test_zero_gradient_has_valid_used_loss_and_zero_data_norm(self):
        parent = tiny_parent("raw1")
        views, labels = inputs()
        zeros = tuple(torch.zeros_like(parameter) for parameter in parent[0].parameters())
        with patch.object(torch.autograd, "grad", return_value=zeros):
            record = core.training_update(*parent, views[:1], labels, "raw1")
        self.assertEqual(record["adam_steps"], 1)
        self.assertEqual(record["gradient_evaluations"], 1)
        self.assertGreater(record["losses"][0], 0)
        self.assertEqual(record["data_step_norm"], 0.)
        self.assertEqual(record["applied_norm"], 0.)

    def test_rejects_missing_or_double_adam_step_and_mismatched_clocks(self):
        views, labels = inputs()
        parent = tiny_parent("raw1")
        with patch.object(parent[1], "step", return_value=None), self.assertRaisesRegex(ValueError, "exactly once"):
            core.training_update(*parent, views[:1], labels, "raw1")
        parent = tiny_parent("raw1")
        normal_step = parent[1].step

        def twice():
            normal_step()
            normal_step()

        with patch.object(parent[1], "step", side_effect=twice), self.assertRaisesRegex(ValueError, "exactly once"):
            core.training_update(*parent, views[:1], labels, "raw1")
        parent = tiny_parent("native1")
        parent[2].step_count = 1
        with self.assertRaisesRegex(ValueError, "clocks differ"):
            core.training_update(*parent, views[:1], labels, "native1")

    def test_rejects_invalid_input_policy_and_tracker_before_update(self):
        views, labels = inputs()
        cases = [(views, labels, "raw1"), (views[:1], labels, "raw4"),
                 (views[:1].double(), labels, "raw1"),
                 (views[:1, :, :-1], labels, "raw1"),
                 (torch.full_like(views[:1], float("nan")), labels, "raw1"),
                 (torch.full_like(views[:1], 2.), labels, "raw1"),
                 (views[:1], labels.float(), "raw1"),
                 (views[:1], torch.tensor([0, -1, 2]), "raw1"),
                 (views[:1], labels, "unknown"), (views[:1], labels, "native1")]
        for value, target, policy in cases:
            parent = tiny_parent("raw1")
            before = core.neural_core.flat_params(parent[0])
            with self.subTest(policy=policy), self.assertRaises(ValueError):
                core.training_update(*parent, value, target, policy)
            self.assertTrue(torch.equal(before, core.neural_core.flat_params(parent[0])))
            self.assertEqual(len(parent[1].state), 0)

    def test_rejects_nonfinite_gradient_and_unsupported_tracker(self):
        parent = tiny_parent("raw1")
        views, labels = inputs()
        values = tuple(torch.full_like(parameter, float("inf")) for parameter in parent[0].parameters())
        with patch.object(torch.autograd, "grad", return_value=values), self.assertRaisesRegex(ValueError, "batch gradient"):
            core.training_update(*parent, views[:1], labels, "raw1")
        self.assertEqual(len(parent[1].state), 0)
        parent = tiny_parent()
        parent[2].filter_strength = .5
        with self.assertRaisesRegex(ValueError, "hard-projection"):
            core.training_update(*parent, views[:1], labels, "native1")


if __name__ == "__main__":
    unittest.main()
