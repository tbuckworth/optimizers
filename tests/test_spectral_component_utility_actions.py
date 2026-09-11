"""Fabricated CPU strong-model states; no saved scientific parent or dataset."""

import copy
import importlib
import unittest
from unittest.mock import patch

import torch
from torch import nn

from experiments import spectral_component_utility_actions as actions


def fabricated_parent(step=100):
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(7301)
        model = nn.Sequential(nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 128),
                              nn.ReLU(), nn.Linear(128, 10))
    model._strong_augmentation_spec = dict(actions.core.MODEL_SPEC)
    optimizer = actions.core.make_optimizer(model)
    tracker = actions.core.make_tracker(model, optimizer)
    for parameter in model.parameters():
        optimizer.state[parameter] = {"step": torch.tensor(float(step)),
            "exp_avg": torch.full_like(parameter, .001),
            "exp_avg_sq": torch.full_like(parameter, .01)}
        parameter.grad = torch.full_like(parameter, .003)
    tracker.step_count = step
    tracker.grad_mean = torch.linspace(-.001, .001, actions.core.PARAMETER_COUNT)
    tracker.V = torch.zeros(actions.core.PARAMETER_COUNT, 2)
    tracker.V[0, 0], tracker.V[1, 1] = 1, 1
    tracker.S = torch.tensor([.03, .02], dtype=torch.float64)
    return model, optimizer, tracker


def state(parent):
    model, optimizer, tracker = parent
    return {"model": dict(model.state_dict()), "optimizer": optimizer.state_dict(),
            "gradients": [p.grad for p in model.parameters()],
            "tracker": actions.core.neural_core._tracker_state(tracker),
            "modes": [(name, module.training) for name, module in model.named_modules()]}


class ActionTests(unittest.TestCase):
    def test_import_is_inert(self):
        with patch.object(torch, "load", side_effect=AssertionError("read")), \
                patch.object(nn, "Linear", side_effect=AssertionError("model creation")), \
                patch.object(torch.cuda, "is_available", side_effect=AssertionError("GPU inspect")):
            importlib.reload(actions)

    def test_literal_canonical_step_and_all_saved_tensors(self):
        for step in (100, 56304):
            parent = fabricated_parent(step)
            gradient = torch.linspace(-.02, .02, actions.core.PARAMETER_COUNT)
            gradient_before = gradient.clone()
            before = actions.core.tree_digest(state(parent))
            rng = torch.get_rng_state().clone()
            with patch.object(torch.cuda, "_lazy_init", side_effect=AssertionError("GPU init")):
                records = actions.paired_actions(parent, gradient)
            self.assertEqual([record["policy"] for record in records], ["raw", "native"])
            for policy, record in zip(("raw", "native"), records):
                model, optimizer, tracker = copy.deepcopy(parent)
                theta = actions.core.flat_params(model)
                actions.core.set_grad(model, gradient)
                if policy == "native":
                    tracker.filter_grad()
                delivered = actions.core.flat_grad(model)
                optimizer.step()
                self.assertTrue(torch.equal(record["theta_before"], theta))
                self.assertTrue(torch.equal(record["theta_after"], actions.core.flat_params(model)))
                self.assertTrue(torch.equal(record["gradient_delivered"], delivered))
                self.assertTrue(torch.equal(record["gradient_raw"], gradient))
                self.assertTrue(torch.equal(record["decay_endpoint"], theta.clone().mul_(1 - .001*.01)))
                for key, source in (("m_after", "exp_avg"), ("v_after", "exp_avg_sq")):
                    expected = torch.cat([optimizer.state[p][source].flatten() for p in model.parameters()])
                    self.assertTrue(torch.equal(record[key], expected))
                self.assertEqual(record["adam_steps_before"].tolist(), [step] * 6)
                self.assertEqual(record["adam_steps_after"].tolist(), [step + 1] * 6)
                self.assertEqual(record["observer_steps_after"], step + (policy == "native"))
                if policy == "native":
                    self.assertTrue(torch.equal(record["post_basis"], tracker.V))
                    self.assertTrue(torch.equal(record["post_singular_values"], tracker.S))
                else:
                    self.assertIsNone(record["post_basis"])
                for value in record.values():
                    if isinstance(value, torch.Tensor):
                        self.assertEqual(value.device.type, "cpu")
                        self.assertFalse(value.requires_grad)
            self.assertEqual(before, actions.core.tree_digest(state(parent)))
            self.assertTrue(torch.equal(gradient, gradient_before))
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            for record in records:
                for value in record.values():
                    if isinstance(value, torch.Tensor):
                        value.fill_(7)
            self.assertEqual(before, actions.core.tree_digest(state(parent)))
            self.assertTrue(torch.equal(gradient, gradient_before))

    def test_repeated_pair_is_not_continuation(self):
        parent = fabricated_parent()
        gradient = torch.linspace(-.02, .02, actions.core.PARAMETER_COUNT)
        first, second = actions.paired_actions(parent, gradient), actions.paired_actions(parent, gradient)
        self.assertEqual(actions.core.tree_digest(first), actions.core.tree_digest(second))

    def test_invalid_gradient_and_parent_clock_rejected(self):
        parent = fabricated_parent()
        gradient = torch.zeros(actions.core.PARAMETER_COUNT)
        for value in (gradient.double(), gradient[:-1], gradient.fill_(float("nan"))):
            with self.assertRaises(ValueError):
                actions.paired_actions(parent, value)
        with self.assertRaises(ValueError):
            actions.paired_actions(fabricated_parent(99), torch.zeros(actions.core.PARAMETER_COUNT))
        parent[2].step_count += 1
        with self.assertRaises(ValueError):
            actions.paired_actions(parent, torch.zeros(actions.core.PARAMETER_COUNT))

    def test_changed_activation_rejected_despite_identical_parameter_names(self):
        parent = fabricated_parent()
        parent[0][1] = nn.Tanh()
        with self.assertRaisesRegex(ValueError, "topology"):
            actions.paired_actions(parent, torch.zeros(actions.core.PARAMETER_COUNT))

    def test_composed_restore_action_and_objective_accounting(self):
        from experiments import spectral_component_utility_restore as adapter
        from experiments import spectral_component_utility_objectives as objectives

        parent = fabricated_parent()
        with patch.object(torch.cuda, "is_available", return_value=False):
            saved = actions.core.snapshot(*parent)
        restored = adapter.restore(saved, expected_step=100)
        original_digest = actions.core.tree_digest(state(restored))
        model = restored[0]
        action_x = torch.linspace(-.2, .4, 64 * 784).reshape(64, 784)
        action_y = torch.arange(64) % 10
        loss = torch.nn.functional.cross_entropy(model(action_x), action_y)
        raw = torch.cat([g.detach().flatten() for g in torch.autograd.grad(loss, tuple(model.parameters()))])
        i_x = torch.linspace(-.1, .3, 3 * 25 * 784).reshape(3 * 25, 784)
        r_x = torch.linspace(-.3, .2, 2 * 25 * 784).reshape(2 * 25, 784)
        true, assigned, r_true = torch.tensor([0, 1, 2]), torch.tensor([2, 1, 9]), torch.tensor([4, 7])

        def evaluate(candidate):
            i_grid = candidate(i_x).reshape(3, 25, 10)
            r_grid = candidate(r_x).reshape(2, 25, 10)
            return objectives.objective_tensors(i_grid, true, assigned, r_grid[:, 12], r_grid, r_true)

        baseline = evaluate(model)
        gradients = objectives.flat_objective_gradients(
            {key: baseline[key] for key in objectives.GRADIENT_KEYS}, tuple(model.parameters()))
        baseline_values = {key: value.detach().clone() for key, value in baseline.items()}
        del baseline
        records = actions.paired_actions(restored, raw)
        for record in records:
            for fraction in (1., .1):
                path = objectives.path_accounting(record["theta_before"], record["theta_after"],
                                                 record["decay_endpoint"], fraction)
                endpoint = copy.deepcopy(model)
                actions.core.set_params(endpoint, path["point"])
                with torch.no_grad():
                    after = evaluate(endpoint)
                changes = {key: baseline_values[key] - after[key] for key in baseline_values}
                self.assertAlmostEqual((changes["S"] + changes["F"] + changes["C"]).item(),
                                       changes["L"].item(), delta=1e-10)
                for gradient in gradients.values():
                    self.assertTrue(torch.isfinite(torch.tensor(
                        objectives.signed_utility(gradient, path["total_delta"]))))
        self.assertEqual(original_digest, actions.core.tree_digest(state(restored)))


if __name__ == "__main__":
    unittest.main()
