"""Fabricated CPU-only states and inputs; no real data, GPU or scientific run."""

import copy
import importlib
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from experiments import spectral_strong_augmentation_core as core


def tiny_parent(policy="native200", step=0):
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(41)
        model = nn.Linear(784, 10, device="cpu", dtype=torch.float32)
    optimizer = core.make_optimizer(model)
    tracker = core.make_tracker(model, optimizer) if policy == "native200" else None
    if step:
        for parameter in model.parameters():
            optimizer.state[parameter] = {"step": torch.tensor(float(step)),
                                          "exp_avg": torch.full_like(parameter, .001),
                                          "exp_avg_sq": torch.full_like(parameter, .01)}
        if tracker is not None:
            tracker.step_count = step
            count = sum(parameter.numel() for parameter in model.parameters())
            tracker.grad_mean = torch.linspace(-.01, .01, count)
            tracker.V = torch.zeros(count, 2)
            tracker.V[0, 0], tracker.V[1, 1] = 1, 1
            tracker.S = torch.tensor([.3, .2], dtype=torch.float64)
    return model, optimizer, tracker


def batch(count=3):
    inputs = torch.linspace(-.4, 2.8, count * 784, dtype=torch.float32).reshape(count, 784)
    return inputs, torch.arange(count, dtype=torch.int64) % 10


def reference_step(parent, inputs, targets):
    model, optimizer, tracker = parent
    optimizer.zero_grad(set_to_none=True)
    loss = F.cross_entropy(model(inputs), targets)
    loss.backward()
    raw = core.flat_grad(model)
    if tracker is not None:
        tracker.filter_grad()
    optimizer.step()
    return float(loss.detach()), raw


class StrongFactoryTests(unittest.TestCase):
    def test_import_is_inert(self):
        with patch.object(torch, "manual_seed", side_effect=AssertionError("seed on import")), \
                patch.object(nn, "Linear", side_effect=AssertionError("model on import")), \
                patch.object(torch, "load", side_effect=AssertionError("checkpoint read")), \
                patch.object(torch.cuda, "is_available", side_effect=AssertionError("GPU inspection")):
            importlib.reload(core)

    def test_fixed_model_initialization_matches_literal_cpu_reference(self):
        for seed in (202609171, 202609172, 202609173):
            # Blocking CUDA initialization ensures these are CPU fixtures even
            # though literal manual_seed also registers accelerator seed values.
            with patch.object(torch.cuda, "_lazy_init", side_effect=AssertionError("CUDA initialization")):
                model = core.make_model(seed, "cpu")
                torch.manual_seed(seed)
                expected = nn.Sequential(nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 128),
                                         nn.ReLU(), nn.Linear(128, 10))
            self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), 235146)
            self.assertTrue(core.neural_core.equal_tree(dict(model.state_dict()), dict(expected.state_dict())))
            self.assertEqual(model._strong_augmentation_spec,
                             {"input_dim": 784, "hidden_dims": (256, 128), "classes": 10})
            self.assertFalse(hasattr(model, "_i9_spec"))
            self.assertTrue(all(parameter.device.type == "cpu" and parameter.dtype == torch.float32
                                and parameter.requires_grad for parameter in model.parameters()))
        for seed in (True, -1, 2**63, 1.5):
            with self.assertRaises(ValueError):
                core.make_model(seed, "cpu")

    def test_exact_optimizer_and_tracker_settings_no_factory_overrides(self):
        model, optimizer, tracker = tiny_parent()
        group = optimizer.param_groups[0]
        self.assertEqual((group["lr"], group["betas"], group["eps"], group["weight_decay"]),
                         (.001, (.9, .999), 1e-8, .01))
        self.assertTrue(all(group[key] is False for key in ("amsgrad", "foreach", "fused", "maximize", "capturable", "differentiable")))
        expected = {"rank": 200, "decay": .99, "warmup": 100, "filter_strength": 1.0,
                    "energy_threshold": None, "adaptive": "none", "normalize": "none", "weighting": "hard",
                    "alpha": 1.0, "soft_residual": True, "stable_update": True,
                    "relative_eig_tol": 1e-8, "absolute_eig_floor": 0.0, "stabilize_every": 100}
        self.assertEqual({key: getattr(tracker, key) for key in expected}, expected)
        self.assertIs(tracker.model, model)
        self.assertIs(tracker.base_optimizer, optimizer)
        with self.assertRaises(TypeError):
            core.make_tracker(model, optimizer, rank=32)
        with self.assertRaises(TypeError):
            core.make_model(1, "cpu", width=3)

    def test_shape_agnostic_flat_helpers_and_truthful_snapshot(self):
        model = core.make_model(7, "cpu")
        optimizer = core.make_optimizer(model)
        tracker = core.make_tracker(model, optimizer)
        original = core.flat_params(model)
        core.set_params(model, original.clone())
        self.assertTrue(torch.equal(original, core.flat_params(model)))
        core.training_update(model, optimizer, tracker, *batch(2), "native200")
        before = core.tree_digest({"model": dict(model.state_dict()), "optimizer": optimizer.state_dict(),
                                   "tracker": core.neural_core._tracker_state(tracker)})
        fake_rng = {"fixture": "CPU-only", "torch_cpu": torch.get_rng_state().clone()}
        with patch.object(core.neural_core, "_rng_state", return_value=fake_rng), \
                patch.object(core.neural_core, "snapshot", side_effect=AssertionError("wrong snapshot schema")), \
                patch.object(torch.cuda, "_lazy_init", side_effect=AssertionError("CUDA initialization")):
            saved = core.snapshot(model, optimizer, tracker)
        self.assertEqual(saved["schema"], "spectral_strong_augmentation_snapshot_v1")
        self.assertEqual(saved["model_spec"]["hidden_dims"], (256, 128))
        self.assertEqual(set(saved["model_state"]), {"0.weight", "0.bias", "2.weight", "2.bias", "4.weight", "4.bias"})
        self.assertEqual(len(saved["gradients"]), 6)
        self.assertEqual(len(core.tree_digest(saved)), 64)
        for value in saved["model_state"].values():
            value.fill_(99)
        saved["optimizer"]["state"][0]["exp_avg"].fill_(99)
        saved["tracker"]["grad_mean"].fill_(99)
        saved["gradients"][0].fill_(99)
        after = core.tree_digest({"model": dict(model.state_dict()), "optimizer": optimizer.state_dict(),
                                  "tracker": core.neural_core._tracker_state(tracker)})
        self.assertEqual(before, after)
        with self.assertRaises(ValueError):
            core.snapshot(*tiny_parent())


class StrongUpdateTests(unittest.TestCase):
    def assert_equal_state(self, actual, expected):
        self.assertTrue(core.neural_core.equal_tree(dict(actual[0].state_dict()), dict(expected[0].state_dict())))
        self.assertTrue(core.neural_core.equal_tree(actual[1].state_dict(), expected[1].state_dict()))
        self.assertTrue(torch.equal(core.flat_grad(actual[0]), core.flat_grad(expected[0])))
        if actual[2] is not None:
            self.assertTrue(core.neural_core.equal_tree(core.neural_core._tracker_state(actual[2]),
                                                        core.neural_core._tracker_state(expected[2])))

    def test_native_bit_exact_at_warmup_boundary_and_canonical_repair(self):
        actual = tiny_parent(step=99)
        expected = copy.deepcopy(actual)
        inputs, targets = batch()
        for step in (100, 101, 102):
            expected_loss, raw = reference_step(expected, inputs, targets)
            result = core.training_update(*actual, inputs, targets, "native200")
            self.assert_equal_state(actual, expected)
            self.assertEqual(result["loss"], expected_loss)
            self.assertEqual(result["raw_norm"], float(raw.double().norm()))
            self.assertEqual((result["observer_steps"], result["adam_steps"]), (step, step))
        self.assertGreater(actual[2].stabilization_count, 0)

    def test_raw_native_model_adam_pairing_during_warmup(self):
        raw, native = tiny_parent("raw"), tiny_parent()
        inputs, targets = batch()
        for step in range(1, 4):
            raw_row = core.training_update(*raw, inputs, targets, "raw")
            native_row = core.training_update(*native, inputs, targets, "native200")
            self.assertTrue(core.neural_core.equal_tree(dict(raw[0].state_dict()), dict(native[0].state_dict())))
            self.assertTrue(core.neural_core.equal_tree(raw[1].state_dict(), native[1].state_dict()))
            self.assertEqual(raw_row["loss"], native_row["loss"])
            self.assertEqual(raw_row["observer_steps"], 0)
            self.assertEqual(native_row["observer_steps"], step)

    def test_final16_mean_loss_and_full64_batch_match_reference(self):
        for count in (16, 64):
            actual = tiny_parent("raw")
            expected = copy.deepcopy(actual)
            inputs, targets = batch(count)
            expected_loss, _ = reference_step(expected, inputs, targets)
            with patch.object(core.SpectralGradientFilter, "filter_grad", side_effect=AssertionError("raw observer")):
                row = core.training_update(*actual, inputs, targets, "raw")
            self.assertEqual(row["loss"], expected_loss)
            self.assert_equal_state(actual, expected)
            self.assertEqual((row["adam_steps"], row["observer_steps"], row["basis_rank"]), (1, 0, 0))

    def test_exact_scalar_schema_and_rounded_decay_data_norm(self):
        for policy in core.POLICIES:
            parent = tiny_parent(policy, step=100)
            inputs, targets = batch(16)
            before_inputs, before_targets = inputs.clone(), targets.clone()
            decay_base = core.flat_params(parent[0]).mul_(1 - .001 * .01)
            row = core.training_update(*parent, inputs, targets, policy)
            self.assertEqual(tuple(row), core.STREAM_KEYS)
            for key in ("loss", "raw_norm", "applied_norm", "data_step_norm"):
                self.assertIs(type(row[key]), float)
                self.assertTrue(np.isfinite(row[key]) and row[key] >= 0)
            for key in ("observer_steps", "adam_steps", "basis_rank"):
                self.assertIs(type(row[key]), int)
            self.assertEqual(row["data_step_norm"], float((core.flat_params(parent[0]) - decay_base).double().norm()))
            self.assertTrue(torch.equal(before_inputs, inputs) and torch.equal(before_targets, targets))

    def test_update_order_one_backward_filter_and_adam(self):
        parent = tiny_parent(step=100)
        inputs, targets = batch()
        events = []
        original_backward, original_filter, original_step = torch.Tensor.backward, parent[2].filter_grad, parent[1].step

        def backward(tensor, *args, **kwargs):
            events.append("backward")
            return original_backward(tensor, *args, **kwargs)

        def filter_grad():
            events.append("filter")
            return original_filter()

        def step():
            events.append("Adam")
            return original_step()

        with patch.object(torch.Tensor, "backward", new=backward), \
                patch.object(parent[2], "filter_grad", side_effect=filter_grad), \
                patch.object(parent[1], "step", side_effect=step):
            core.training_update(*parent, inputs, targets, "native200")
        self.assertEqual(events, ["backward", "filter", "Adam"])

    def test_zero_gradient_data_norm_excludes_actual_decay(self):
        parent = tiny_parent("raw")
        inputs, targets = batch()

        def zero_backward(tensor, *args, **kwargs):
            for parameter in parent[0].parameters():
                parameter.grad = torch.zeros_like(parameter)

        with patch.object(torch.Tensor, "backward", new=zero_backward):
            row = core.training_update(*parent, inputs, targets, "raw")
        self.assertGreater(row["loss"], 0)
        self.assertEqual((row["raw_norm"], row["applied_norm"], row["data_step_norm"], row["adam_steps"]), (0., 0., 0., 1))

    def test_rejects_skipped_doubled_and_inconsistent_clocks(self):
        inputs, targets = batch()
        parent = tiny_parent("raw")
        with patch.object(parent[1], "step", return_value=None), self.assertRaisesRegex(ValueError, "exactly once"):
            core.training_update(*parent, inputs, targets, "raw")
        parent = tiny_parent("raw")
        normal = parent[1].step

        def twice():
            normal()
            normal()

        with patch.object(parent[1], "step", side_effect=twice), self.assertRaisesRegex(ValueError, "exactly once"):
            core.training_update(*parent, inputs, targets, "raw")
        parent = tiny_parent()
        parent[2].step_count = 1
        with self.assertRaisesRegex(ValueError, "clocks differ"):
            core.training_update(*parent, inputs, targets, "native200")

    def test_rejects_invalid_inputs_and_changed_scientific_settings(self):
        inputs, targets = batch()
        bad_inputs = (inputs.double(), inputs[:, :-1], torch.empty(0, 784),
                      torch.zeros(65, 784), torch.full_like(inputs, float("nan")))
        for value in bad_inputs:
            parent = tiny_parent("raw")
            with self.assertRaises(ValueError):
                core.training_update(*parent, value, targets, "raw")
            self.assertEqual(len(parent[1].state), 0)
        for target in (targets.float(), targets[:-1], torch.tensor([0, 10, -1])):
            with self.assertRaises(ValueError):
                core.training_update(*tiny_parent("raw"), inputs, target, "raw")
        parent = tiny_parent()
        parent[2].rank = 32
        with self.assertRaisesRegex(ValueError, "scientific settings"):
            core.training_update(*parent, inputs, targets, "native200")
        parent = tiny_parent("raw")
        parent[1].param_groups[0]["lr"] = .002
        with self.assertRaisesRegex(ValueError, "scientific settings"):
            core.training_update(*parent, inputs, targets, "raw")


if __name__ == "__main__":
    unittest.main()
