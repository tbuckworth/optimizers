import unittest

import torch
import torch.nn as nn

from experiments.grokking_action_policy import (
    LegacyActionPolicyFilter,
    compute_actions,
)
from spectral_filter import SpectralGradientFilter


class FlatModel(nn.Module):
    def __init__(self, size=5):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(size, dtype=torch.float32))


def make_filter(cls, model, policy=None):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    common = dict(
        rank=3,
        decay=0.9,
        warmup=0,
        stable_update=False,
        weighting="hard",
        normalize="none",
        filter_strength=1.0,
        adaptive="none",
    )
    if cls is LegacyActionPolicyFilter:
        return cls(model, optimizer, action_policy=policy, **common)
    return cls(model, optimizer, **common)


class ActionComputationTest(unittest.TestCase):
    def test_native_exact_and_orthogonal_geometry(self):
        V = torch.tensor(
            [[2.0, 0.0], [0.0, 0.5], [1.0, 1.0], [0.0, 0.0]],
            dtype=torch.float32,
        )
        g = torch.tensor([1.0, -2.0, 0.5, 3.0], dtype=torch.float32)
        result = compute_actions(V, g, retain_basis=True)
        native = V @ (V.T @ g)
        self.assertTrue(torch.equal(result["actions"]["native"], native))
        self.assertEqual(result["actions"]["native"].dtype, g.dtype)
        self.assertEqual(result["numerical_rank"], 2)
        self.assertEqual((result["P"], result["k"]), V.shape)
        self.assertGreater(result["sigma_max"], 0.0)
        self.assertGreaterEqual(result["elapsed_seconds"], 0.0)
        self.assertTrue(result["full_column_rank"])

        Q = result["Q"]
        torch.testing.assert_close(
            Q.T @ Q, torch.eye(2, dtype=torch.float64), rtol=1e-12, atol=1e-12
        )
        self.assertLess(
            result["geometry_diagnostics"]["q_orthogonality_max_abs"], 1e-12
        )
        self.assertLess(
            result["geometry_diagnostics"][
                "v_reconstruction_relative_frobenius_residual"
            ],
            1e-12,
        )
        orthogonal = result["actions"]["orthogonal"]
        twice = compute_actions(V, orthogonal, selected="orthogonal")["actions"][
            "orthogonal"
        ]
        torch.testing.assert_close(twice, orthogonal, rtol=2e-6, atol=2e-6)
        self.assertNotAlmostEqual(native.norm().item(), orthogonal.norm().item())
        torch.testing.assert_close(
            result["actions"]["norm_matched"].double().norm(),
            native.double().norm(),
            rtol=2e-6,
            atol=2e-6,
        )
        self.assertLess(
            result["diagnostics"]["norm_match_relative_mismatch"], 2e-6
        )
        self.assertFalse(
            result["diagnostics"]["norm_match_denominator_clamped"]
        )
        self.assertTrue(result["diagnostics"]["norm_match_exact"])
        expected_scale = (
            native.double().norm()
            / result["actions"]["orthogonal"].double().norm()
        ).item()
        self.assertEqual(
            result["diagnostics"]["norm_match_scale"], expected_scale
        )
        self.assertTrue(
            torch.equal(
                result["actions"]["norm_matched"],
                (result["actions"]["orthogonal"].double() * expected_scale).float(),
            )
        )

    def test_rank_deficiency_and_span_invariance(self):
        base = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, -1.0]],
            dtype=torch.float64,
        )
        deficient = torch.column_stack((base[:, 0], base[:, 1], base[:, 0]))
        g = torch.tensor([0.4, -1.0, 2.0, 0.3], dtype=torch.float64)
        result = compute_actions(deficient, g, retain_basis=True)
        self.assertEqual(result["numerical_rank"], 2)
        self.assertFalse(result["full_column_rank"])
        self.assertEqual(len(result["singular_values"]), 3)
        self.assertGreaterEqual(
            result["svd_truncation_relative_frobenius_residual"], 0.0
        )

        transform = torch.tensor([[2.0, 1.0], [-0.5, 3.0]], dtype=torch.float64)
        transformed = compute_actions(base @ transform, g)
        torch.testing.assert_close(
            transformed["actions"]["orthogonal"],
            compute_actions(base, g)["actions"]["orthogonal"],
            rtol=1e-12,
            atol=1e-12,
        )

    def test_zero_inputs_are_finite_pure_and_rng_free(self):
        V = torch.zeros((5, 3), dtype=torch.float32)
        g = torch.tensor([1.0, -2.0, 3.0, -4.0, 5.0], dtype=torch.float32)
        V_before, g_before = V.clone(), g.clone()
        rng_before = torch.random.get_rng_state().clone()
        result = compute_actions(V, g, retain_basis=True)
        self.assertTrue(torch.equal(torch.random.get_rng_state(), rng_before))
        self.assertTrue(torch.equal(V, V_before))
        self.assertTrue(torch.equal(g, g_before))
        self.assertEqual(result["numerical_rank"], 0)
        self.assertFalse(result["full_column_rank"])
        self.assertEqual(result["Q"].shape, (5, 0))
        for action in result["actions"].values():
            self.assertTrue(torch.equal(action, torch.zeros_like(g)))
            self.assertTrue(bool(torch.isfinite(action).all()))
        self.assertTrue(result["diagnostics"]["norm_match_degenerate"])
        self.assertEqual(
            result["diagnostics"]["norm_match_absolute_mismatch"], 0.0
        )

    def test_subfloor_denominator_clamps_and_exact_zero_fails_closed(self):
        V = torch.tensor([[1e20], [0.0]], dtype=torch.float32)
        g = torch.tensor([1e-40, 1.0], dtype=torch.float32)
        clamped = compute_actions(V, g, selected="norm_matched")
        self.assertTrue(
            clamped["diagnostics"]["norm_match_denominator_clamped"]
        )
        self.assertFalse(clamped["diagnostics"]["norm_match_exact"])

        min_subnormal = torch.finfo(torch.float32).tiny * torch.finfo(
            torch.float32
        ).eps
        V = torch.tensor([[1e20], [1e20]], dtype=torch.float32)
        g = torch.tensor([min_subnormal, 0.0], dtype=torch.float32)
        native = compute_actions(V, g, selected="native")["actions"]["native"]
        self.assertGreater(native.double().norm().item(), 0.0)
        with self.assertRaisesRegex(FloatingPointError, "cannot norm-match"):
            compute_actions(V, g, selected="norm_matched")

    def test_malformed_inputs_and_policy_are_rejected(self):
        V = torch.ones((3, 2), dtype=torch.float32)
        g = torch.ones(3, dtype=torch.float32)
        with self.assertRaisesRegex(ValueError, "shape"):
            compute_actions(V, g[:2])
        with self.assertRaisesRegex(ValueError, "same dtype"):
            compute_actions(V.double(), g)
        with self.assertRaisesRegex(ValueError, "finite"):
            compute_actions(V, torch.tensor([1.0, float("nan"), 2.0]))
        with self.assertRaisesRegex(ValueError, "selected"):
            compute_actions(V, g, selected="unknown")


class LegacyWrapperTest(unittest.TestCase):
    def test_estimator_recurrence_and_inherited_diagnostics_match(self):
        baseline_model = FlatModel()
        policy_model = FlatModel()
        baseline = make_filter(SpectralGradientFilter, baseline_model)
        policy = make_filter(LegacyActionPolicyFilter, policy_model, "orthogonal")

        initial_V = torch.tensor(
            [[1.0, 0.0], [0.0, 0.8], [0.2, 0.1], [0.0, 0.2], [0.1, -0.1]],
            dtype=torch.float32,
        )
        initial_S = torch.tensor([1.5, 0.7], dtype=torch.float32)
        initial_mean = torch.tensor([0.1, -0.2, 0.3, 0.0, -0.1])
        for tracker in (baseline, policy):
            tracker.V = initial_V.clone()
            tracker.S = initial_S.clone()
            tracker.grad_mean = initial_mean.clone()
            tracker.step_count = 12

        raw = torch.tensor([0.5, -1.2, 0.7, 0.1, 0.4])
        baseline_model.weight.grad = raw.clone()
        policy_model.weight.grad = raw.clone()
        baseline_diagnostics = baseline.filter_grad()
        policy_diagnostics = policy.filter_grad()

        self.assertEqual(baseline_diagnostics, policy_diagnostics)
        self.assertEqual(baseline.step_count, policy.step_count)
        self.assertTrue(torch.equal(baseline.S, policy.S))
        self.assertTrue(torch.equal(baseline.grad_mean, policy.grad_mean))
        self.assertTrue(torch.equal(baseline.V, policy.V))
        self.assertIsNotNone(policy.last_action_metrics)
        self.assertEqual(policy.last_action_metrics["step"], 13)
        self.assertEqual(
            policy.last_action_metrics["basis_column_count"], policy.V.shape[1]
        )
        self.assertIn("singular_values", policy.last_action_metrics)
        self.assertIn("coefficients", policy.last_action_metrics)
        self.assertIsNone(policy.last_action_basis)
        self.assertEqual(set(policy.last_action_metrics["coefficients"]), {"orthogonal"})
        expected = compute_actions(policy.V, raw)["actions"]["orthogonal"]
        self.assertTrue(torch.equal(policy_model.weight.grad, expected))

    def test_wrapper_rejects_nonlegacy_configuration(self):
        model = FlatModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
        with self.assertRaisesRegex(ValueError, "stable_update"):
            LegacyActionPolicyFilter(
                model,
                optimizer,
                action_policy="orthogonal",
                stable_update=True,
            )


if __name__ == "__main__":
    unittest.main()
