"""Synthetic-only fork and Adam checks; no scientific checkpoint acquisition."""
import copy
import unittest

import torch

from experiments.grokking_action_intervention import (
    adam_diagnostic, cosine, restore_filter, scientific_hash, snapshot, tree_hash,
)
from experiments.grokking_confirmation import (
    FILTER, filter_state, parameter_identity, restore_state, tensor_set_identity,
)
from experiments.grokking_action_policy import compute_actions, LegacyActionPolicyFilter


class ActionInterventionTests(unittest.TestCase):
    def test_tree_hash_types_values_and_order(self):
        one = {"a": torch.tensor([1., 2.]), "b": [3, None]}
        other = {"b": [3, None], "a": one["a"].clone()}
        self.assertEqual(tree_hash(one), tree_hash(other))
        other["a"][0] = 2
        self.assertNotEqual(tree_hash(one), tree_hash(other))
        self.assertNotEqual(tree_hash([1, 2]), tree_hash((1, 2)))
        self.assertNotEqual(tree_hash(torch.ones(2)), tree_hash(torch.ones(2).double()))

    def test_adam_decomposition_carried_moments(self):
        model = torch.nn.Linear(2, 1, bias=False).double()
        optimizer = torch.optim.AdamW(model.parameters(), lr=.003, betas=(.9, .98),
                                      weight_decay=1., eps=1e-8, foreach=False)
        for gradient in ([.3, -.1], [-.2, .6], [.02, -.7]):
            before = [p.detach().clone() for p in model.parameters()]
            model.weight.grad = torch.tensor([gradient], dtype=torch.float64)
            optimizer.step()
            result = adam_diagnostic(model, optimizer, before)
            self.assertLess(result["summary"]["decomposition_residual_max_abs"], 1e-15)
            self.assertEqual(float(result["m"][0]), float(optimizer.state[model.weight]["exp_avg"][0, 0]))
        self.assertIsNone(cosine(torch.zeros(2), torch.ones(2)))
        self.assertAlmostEqual(cosine(torch.ones(2), torch.ones(2)), 1.)

    def test_exact_common_fork_and_unchanged_estimator(self):
        torch.manual_seed(101)
        model = torch.nn.Linear(3, 1, bias=False)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001, betas=(.9, .98), weight_decay=1.)
        tracker = LegacyActionPolicyFilter(model, optimizer, action_policy="native",
                                          **{**FILTER, "rank": 2}, stable_update=False)
        tracker.V = torch.tensor([[2., 0.], [0., .5], [.1, .1]])
        tracker.S = torch.tensor([1., .8])
        tracker.grad_mean = torch.zeros(3)
        tracker.step_count = 1500
        # Initialize authentic carried moments, then snapshot the common fork.
        model.weight.grad = torch.ones_like(model.weight)
        optimizer.step()
        split = tensor_set_identity((torch.ones(2, 3), torch.ones(2), torch.ones(2, 3), torch.ones(2)))
        contract = {"config": {"seed": 101}, "source_identity": {"synthetic": True},
                    "split_identity": split, "parameter_identity": parameter_identity(model)}
        state = snapshot(model, optimizer, tracker, contract, "cpu")
        raw = torch.tensor([.1, .2, -.1])
        tracker.step_count += 1
        tracker._update_svd(raw)
        shared = filter_state(tracker)
        actions = compute_actions(tracker.V, raw, retain_basis=True)
        updated_models = []
        for policy in ("native", "orthogonal", "norm_matched"):
            new_model = torch.nn.Linear(3, 1, bias=False)
            new_optimizer = torch.optim.AdamW(new_model.parameters(), lr=.001, betas=(.9, .98), weight_decay=1.)
            new_tracker = LegacyActionPolicyFilter(new_model, new_optimizer, action_policy=policy,
                                                   **{**FILTER, "rank": 2}, stable_update=False)
            restore_state(state, new_model, new_optimizer, new_tracker,
                          config=state["config"], sources=state["source_identity"],
                          split=split, parameters=parameter_identity(new_model), device="cpu")
            self.assertEqual(scientific_hash(state), scientific_hash(snapshot(
                new_model, new_optimizer, new_tracker, state, "cpu")))
            restore_filter(new_tracker, shared, "cpu")
            new_tracker._set_flat_grad(actions["actions"][policy].clone())
            new_optimizer.step()
            self.assertEqual(tree_hash(filter_state(new_tracker)), tree_hash(shared))
            self.assertEqual(new_tracker.step_count, 1501)
            updated_models.append(new_model.weight.detach().clone())
        self.assertFalse(torch.equal(updated_models[0], updated_models[1]))


if __name__ == "__main__":
    unittest.main()
