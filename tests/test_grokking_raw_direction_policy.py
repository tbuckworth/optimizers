import json
import unittest

import torch
from torch import nn

from experiments.grokking_raw_direction_policy import (
    POLICY, RawDirectionPolicyFilter, compute_raw_direction_actions, match_raw_norm,
)
from spectral_filter import SpectralGradientFilter


class FlatModel(nn.Module):
    def __init__(self, size=4):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))


COMMON = dict(rank=3, decay=.9, warmup=0, stable_update=False,
              weighting="hard", normalize="none", filter_strength=1., adaptive="none")


class RawDirectionGeometryTest(unittest.TestCase):
    def test_direction_norm_purity_and_old_native_order(self):
        V = torch.tensor([[2., 0.], [0., .5], [1., 1.], [0., 0.]])
        g = torch.tensor([1., -2., .5, 3.])
        saved_V, saved_g, rng = V.clone(), g.clone(), torch.get_rng_state().clone()
        result = compute_raw_direction_actions(V, g, retain_basis=True)
        native, raw, projected = (result["actions"][key] for key in ("native", POLICY, "norm_matched"))
        self.assertTrue(torch.equal(native, V @ (V.T @ g)))
        target = native.double().norm()
        self.assertTrue(torch.equal(raw, (g.double() * (target / g.double().norm())).float()))
        torch.testing.assert_close(raw.double().norm(), target, rtol=2e-6, atol=0)
        torch.testing.assert_close(raw.double().norm(), projected.double().norm(), rtol=2e-6, atol=0)
        self.assertGreater(float((raw.double() - result["Q"] @ (result["Q"].T @ raw.double())).norm()), .1)
        rho = result["diagnostics"]["projected_over_raw_norm"]
        self.assertAlmostEqual(result["diagnostics"]["pairwise_cosines"]["norm_matched_" + POLICY], rho, places=6)
        self.assertTrue(torch.equal(V, saved_V))
        self.assertTrue(torch.equal(g, saved_g))
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        self.assertEqual(result["coefficients"][POLICY].shape, (2,))

    def test_zero_target_nonzero_raw_keeps_zero_tensor(self):
        result = compute_raw_direction_actions(torch.zeros(4, 2), torch.ones(4))
        self.assertTrue(torch.equal(result["actions"][POLICY], torch.zeros(4)))
        self.assertEqual(result["diagnostics"]["raw_norm_match_target_norm"], 0)
        self.assertTrue(result["diagnostics"]["raw_norm_match_exact"])
        self.assertIsNone(result["diagnostics"]["pairwise_cosines"]["raw_" + POLICY])
        self.assertNotIn(POLICY, result["coefficients"])

    def test_two_zeros_and_positive_target_zero_raw(self):
        out, diag = match_raw_norm(torch.zeros(3), torch.zeros(3))
        self.assertTrue(torch.equal(out, torch.zeros(3)))
        self.assertTrue(diag["raw_norm_match_exact"])
        self.assertTrue(diag["raw_norm_match_degenerate"])
        self.assertTrue(diag["raw_norm_match_denominator_clamped"])
        with self.assertRaisesRegex(FloatingPointError, "zero raw direction"):
            match_raw_norm(torch.zeros(3), torch.ones(3))

    def test_subfloor_is_explicit_and_uses_frozen_formula(self):
        g = torch.tensor([1e-32, 0.], dtype=torch.float64)
        native = torch.tensor([1e-31, 0.], dtype=torch.float64)
        out, diag = match_raw_norm(g, native)
        self.assertTrue(torch.equal(out, g * (native.norm().item() / 1e-30)))
        self.assertTrue(diag["raw_norm_match_denominator_clamped"])
        self.assertFalse(diag["raw_norm_match_exact"])
        self.assertGreater(diag["raw_norm_match_relative_mismatch"], 1e-3)

    def test_rank_deficient_and_zero_gradient_domains(self):
        V = torch.tensor([[1., 1.], [0., 0.], [1., 1.]], dtype=torch.float64)
        result = compute_raw_direction_actions(V, torch.zeros(3, dtype=torch.float64), retain_basis=True)
        self.assertEqual(result["numerical_rank"], 1)
        self.assertFalse(result["full_column_rank"])
        self.assertIsNone(result["diagnostics"]["projected_over_raw_norm"])
        self.assertTrue(torch.equal(result["actions"][POLICY], torch.zeros(3, dtype=torch.float64)))

    def test_bad_inputs_and_overflow_rejected(self):
        with self.assertRaises(TypeError):
            match_raw_norm(None, torch.ones(2))
        for a, b in [(torch.ones(2), torch.ones(3)), (torch.ones(2), torch.ones(2).double()),
                     (torch.ones(2), torch.tensor([float("nan"), 1.]))]:
            with self.assertRaises(ValueError):
                match_raw_norm(a, b)
        with self.assertRaises(FloatingPointError):
            match_raw_norm(torch.ones(2, dtype=torch.float64), torch.full((2,), 1.7e308, dtype=torch.float64))


class RawDirectionEstimatorTest(unittest.TestCase):
    def test_shared_input_updates_identical_legacy_estimator_once(self):
        models = [FlatModel(), FlatModel()]
        trackers = [cls(model, torch.optim.SGD(model.parameters(), lr=0), **COMMON)
                    for cls, model in zip((SpectralGradientFilter, RawDirectionPolicyFilter), models)]
        for tracker in trackers:
            tracker.V = torch.tensor([[1., 0.], [0., .8], [.2, .1], [0., .2]])
            tracker.S = torch.tensor([1.5, .7])
            tracker.grad_mean = torch.tensor([.1, -.2, .3, 0.])
            tracker.step_count = 12
        g = torch.tensor([.5, -1.2, .7, .4])
        for model in models:
            model.weight.grad = g.clone()
        ordinary = trackers[0].filter_grad()
        transformed = trackers[1].filter_grad()
        self.assertEqual(ordinary, transformed)
        for key in ("V", "S", "grad_mean"):
            self.assertTrue(torch.equal(getattr(trackers[0], key), getattr(trackers[1], key)))
        self.assertEqual(trackers[1].step_count, 13)
        target = models[0].weight.grad.double().norm()
        expected = (g.double() * (target / g.double().norm())).float()
        self.assertTrue(torch.equal(models[1].weight.grad, expected))
        self.assertTrue(torch.equal(trackers[1].last_raw_gradient, g))
        json.dumps(trackers[1].last_action_metrics, allow_nan=False)

    def test_zero_action_preserves_carried_adam_and_decay_step(self):
        model = FlatModel(size=2)
        opt = torch.optim.AdamW(model.parameters(), lr=.01, betas=(.9, .98), weight_decay=.1)
        model.weight.grad = torch.full((2,), 2.)
        opt.step()
        tracker = RawDirectionPolicyFilter(model, opt, **COMMON)
        tracker.V, tracker.S = torch.tensor([[1.], [0.]]), torch.ones(1)
        tracker.grad_mean = torch.tensor([0., 1.])
        tracker.step_count = 12
        model.weight.grad = tracker.grad_mean.clone()
        tracker.filter_grad()
        self.assertTrue(torch.equal(model.weight.grad, torch.zeros(2)))
        before = model.weight.detach().clone()
        m, v = opt.state[model.weight]["exp_avg"].clone(), opt.state[model.weight]["exp_avg_sq"].clone()
        opt.step()
        self.assertEqual(int(opt.state[model.weight]["step"]), 2)
        torch.testing.assert_close(opt.state[model.weight]["exp_avg"], .9 * m)
        torch.testing.assert_close(opt.state[model.weight]["exp_avg_sq"], .98 * v)
        self.assertFalse(torch.equal(model.weight, before))
        expected = before.double() * (1 - .01*.1) - .01 * ((.9*m.double())/(1-.9**2)) / ((.98*v.double()/(1-.98**2)).sqrt() + 1e-8)
        torch.testing.assert_close(model.weight.double(), expected, rtol=2e-7, atol=2e-7)

    def test_no_basis_preserves_legacy_identity_and_rejects_stable(self):
        model = FlatModel()
        opt = torch.optim.SGD(model.parameters(), lr=0)
        tracker = RawDirectionPolicyFilter(model, opt, **COMMON)
        raw = torch.arange(4, dtype=torch.float32)
        self.assertTrue(torch.equal(tracker._project_gradient(raw), raw))
        self.assertTrue(tracker.last_action_metrics["diagnostics"]["no_basis_identity"])
        self.assertIsNone(tracker.last_action_result)
        with self.assertRaisesRegex(ValueError, "stable_update"):
            RawDirectionPolicyFilter(model, opt, stable_update=True)


if __name__ == "__main__":
    unittest.main()
