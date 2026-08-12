import unittest

import torch
import torch.nn as nn

from spectral_filter import SpectralGradientFilter


class FlatModel(nn.Module):
    def __init__(self, size, dtype=torch.float64):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(size, dtype=dtype))


def make_filter(size, rank, dtype=torch.float64, **kwargs):
    model = FlatModel(size, dtype=dtype)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    kwargs.setdefault("relative_eig_tol", 1e-12)
    kwargs.setdefault("stabilize_every", 7)
    kwargs.setdefault("stable_update", True)
    filt = SpectralGradientFilter(
        model,
        optimizer,
        rank=rank,
        decay=0.9,
        warmup=0,
        weighting="hard",
        **kwargs,
    )
    return model, filt


def feed(model, filt, gradient):
    model.weight.grad = gradient.clone()
    filt.step_count += 1
    filt._update_svd(gradient)


def represented_covariance(filt):
    return (
        filt.V * filt.S.to(filt.V).square().unsqueeze(0)
    ) @ filt.V.T


class SpectralFilterNumericsTest(unittest.TestCase):
    def test_exact_covariance_without_truncation(self):
        torch.manual_seed(1)
        model, filt = make_filter(9, 9)
        mean = None
        exact = None
        for gradient in torch.randn(40, 9, dtype=torch.float64):
            if mean is None:
                mean = gradient.clone()
            else:
                mean = 0.9 * mean + 0.1 * gradient
            centered = gradient - mean
            feed(model, filt, gradient)
            if exact is None and centered.norm() > 0:
                exact = torch.outer(centered, centered)
            elif exact is not None:
                exact = 0.9 * exact + 0.1 * torch.outer(centered, centered)
            if exact is not None:
                torch.testing.assert_close(
                    represented_covariance(filt), exact, rtol=2e-9, atol=2e-10
                )
                self.assertLess(filt.orthogonality_error(), 1e-9)

    def test_hard_projection_is_nonexpansive_and_idempotent(self):
        torch.manual_seed(2)
        model, filt = make_filter(30, 12)
        for gradient in torch.randn(80, 30, dtype=torch.float64):
            feed(model, filt, gradient)
        for gradient in torch.randn(20, 30, dtype=torch.float64):
            projected = filt._project_gradient(gradient)
            projected_twice = filt._project_gradient(projected)
            self.assertLessEqual(projected.norm(), gradient.norm() * (1 + 1e-10))
            torch.testing.assert_close(
                projected_twice, projected, rtol=1e-9, atol=1e-10
            )

    def test_gradient_scale_does_not_change_subspace(self):
        torch.manual_seed(3)
        stream = torch.randn(70, 24, dtype=torch.float64)
        filters = []
        for scale in (1e-5, 1.0, 1e5):
            model, filt = make_filter(24, 10)
            for gradient in stream:
                feed(model, filt, scale * gradient)
            filters.append(filt)
        projectors = [filt.V @ filt.V.T for filt in filters]
        for projector in projectors[1:]:
            torch.testing.assert_close(
                projector, projectors[0], rtol=2e-8, atol=2e-9
            )
        self.assertEqual(len({filt.V.shape[1] for filt in filters}), 1)

    def test_fp32_long_stream_remains_a_projection(self):
        torch.manual_seed(6)
        model, filt = make_filter(
            256,
            64,
            dtype=torch.float32,
            relative_eig_tol=1e-8,
            stabilize_every=100,
        )
        for gradient in torch.randn(600, 256, dtype=torch.float32):
            feed(model, filt, gradient)
        self.assertEqual(filt.V.shape[1], 64)
        self.assertLess(filt.orthogonality_error(), 5e-4)
        for gradient in torch.randn(10, 256, dtype=torch.float32):
            projected = filt._project_gradient(gradient)
            self.assertLessEqual(projected.norm(), gradient.norm() * (1 + 2e-5))

    def test_reset_clears_numerical_diagnostics(self):
        model, filt = make_filter(8, 4)
        for gradient in torch.randn(10, 8, dtype=torch.float64):
            feed(model, filt, gradient)
        filt.reset()
        self.assertIsNone(filt.V)
        self.assertIsNone(filt.S)
        self.assertEqual(filt.stabilization_count, 0)
        self.assertEqual(filt.max_orthogonality_error, 0.0)

    def test_invalid_numerical_controls_are_rejected(self):
        model = FlatModel(4)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
        with self.assertRaisesRegex(ValueError, "non-negative"):
            SpectralGradientFilter(model, optimizer, relative_eig_tol=-1)
        with self.assertRaisesRegex(ValueError, "positive or None"):
            SpectralGradientFilter(model, optimizer, stabilize_every=0)

    def test_stable_update_is_default_and_legacy_is_available(self):
        model = FlatModel(4)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
        filt = SpectralGradientFilter(model, optimizer)
        self.assertTrue(filt.stable_update)
        self.assertTrue(filt._diagnostics()["stable_update"])
        legacy = SpectralGradientFilter(model, optimizer, stable_update=False)
        self.assertFalse(legacy.stable_update)


if __name__ == "__main__":
    unittest.main()
