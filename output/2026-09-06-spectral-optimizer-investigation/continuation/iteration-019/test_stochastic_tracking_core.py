#!/usr/bin/env python3
"""Deterministic CPU fixtures for the import-safe I19 tracking core."""
from __future__ import annotations

import math
import unittest
from unittest import mock

import numpy as np
import torch

import stochastic_tracking_core as core


class StochasticTrackingCoreTests(unittest.TestCase):
    def test_registered_policy_order_and_analytic_constants(self):
        self.assertEqual(core.policy_names(0.0), (
            "raw", "ema_q0p9", "ema_q0p99", "ema_q0p999",
            "dema_q0p9", "dema_q0p99", "dema_q0p999",
            "common_kalman", "useful_oracle_kalman",
            "scalar_k0", "scalar_k0p5", "scalar_k0p9", "scalar_k1",
            "oracle_useful", "oracle_nuisance", "native_i17", "native_cp",
            "native_cp_star", "oracle_useful_star"))
        self.assertEqual(len(core.policy_names(0.1)), 20)
        self.assertEqual(core.policy_names(0.1)[7], "ema_common_steady_oracle")
        self.assertIsNone(core.optimal_ema_decay(0.0))
        for process in (0.01, 0.1):
            decay = core.optimal_ema_decay(process)
            self.assertGreater(decay, 0.0)
            self.assertLess(decay, 1.0)
            self.assertLess(abs(process * decay - 5.0 * (1.0 - decay) ** 2), 2e-15)
        expected_star = (2.1 - math.sqrt(.41)) / 2.0
        self.assertEqual(core.RHO_STAR, expected_star)

    def test_invalid_inputs_rejected_before_observer_use(self):
        valid = torch.zeros((3, 3), dtype=torch.float64)
        cases = [torch.zeros((3, 3), dtype=torch.float32),
                 torch.zeros((3, 2), dtype=torch.float64),
                 torch.tensor([[0.0, 0.0, float("nan")]], dtype=torch.float64)]
        with mock.patch.object(core.SpectralGradientFilter, "filter_grad",
                               side_effect=AssertionError("observer used")):
            for bad in cases:
                with self.subTest(shape=tuple(bad.shape), dtype=str(bad.dtype)):
                    with self.assertRaises(core.StochasticTrackingCoreError):
                        core.run_stream(bad, 0.0, 0.0)
            with self.assertRaises(core.StochasticTrackingCoreError):
                core.run_stream(valid, 0.02, 0.0)
            with self.assertRaises(core.StochasticTrackingCoreError):
                core.run_stream(valid, 0.0, 0.1)
            with self.assertRaises(core.StochasticTrackingCoreError):
                core.run_stream(valid, 0.0, 0.0, check_budget=3)

    def test_stream_construction_preserves_unused_process_entry_and_rotation(self):
        canonical = torch.tensor([
            [1.0, -2.0, 99.0], [2.0, 1.0, 1.0], [-1.0, 3.0, -2.0],
            [0.5, -0.25, 4.0]], dtype=torch.float64)
        before = canonical.clone()
        base, metadata0 = core.run_stream(canonical, 0.1, 0.0)
        expected_signal = math.sqrt(.1) * np.array([0.0, 1.0, -1.0, 3.0])
        np.testing.assert_array_equal(base["canonical"], canonical.numpy())
        np.testing.assert_array_equal(canonical.numpy(), before.numpy())
        np.testing.assert_array_equal(base["noise"], canonical[:, :2].numpy() * (1.0, 2.0))
        np.testing.assert_allclose(base["s"][:, 0], expected_signal, rtol=0, atol=1e-15)
        np.testing.assert_array_equal(base["s"][:, 1], 0.0)
        np.testing.assert_array_equal(base["g"], base["s"] + base["epsilon"])

        rotated, metadata1 = core.run_stream(canonical, 0.1, math.pi / 4.0)
        angle = math.pi / 4.0
        rotation = np.array([[math.cos(angle), -math.sin(angle)],
                             [math.sin(angle), math.cos(angle)]])
        for key in ("epsilon", "g", "s"):
            np.testing.assert_allclose(rotated[key], base[key] @ rotation.T,
                                       rtol=1e-14, atol=1e-14)
        for name in (
                "raw", "ema_q0p9", "ema_q0p99", "ema_q0p999",
                "dema_q0p9", "dema_q0p99", "dema_q0p999",
                "ema_common_steady_oracle", "common_kalman",
                "useful_oracle_kalman", "scalar_k0", "scalar_k0p5",
                "scalar_k0p9", "scalar_k1", "oracle_useful",
                "oracle_nuisance", "oracle_useful_star"):
            left = rotated["output"][:, metadata1["policy_names"].index(name)]
            right = base["output"][:, metadata0["policy_names"].index(name)] @ rotation.T
            np.testing.assert_allclose(left, right, rtol=2e-13, atol=2e-13)

    def test_observer_and_full_moment_follow_i18_conventions(self):
        canonical = torch.tensor([
            [1., -2., 7.], [2., 1., 1.], [-1., 3., -2.],
            [.5, -.25, 4.], [4., 2., -1.]], dtype=torch.float64)
        arrays, metadata = core.run_stream(canonical, 0.01, 0.0)
        self.assertEqual(metadata["schema"], "i19_stochastic_tracking_stream_metadata_v1")
        self.assertEqual(metadata["array_order"], list(arrays))
        mean = arrays["g"][0].copy()
        full = np.zeros((2, 2))
        initialized = False
        for index in range(len(canonical)):
            if index:
                mean = .99 * mean + .01 * arrays["g"][index]
            np.testing.assert_allclose(arrays["mu"][index], mean, rtol=0, atol=1e-15)
            residual = arrays["g"][index] - mean
            outer = np.outer(residual, residual)
            if not initialized and np.count_nonzero(residual):
                full = outer
                initialized = True
            elif initialized:
                full = .99 * full + .01 * outer
            np.testing.assert_allclose(arrays["full_moment"][index], full,
                                       rtol=1e-14, atol=1e-15)
            np.testing.assert_allclose(arrays["native_delivery"][index],
                                       arrays["A"][index] @ arrays["g"][index],
                                       rtol=1e-12, atol=1e-14)
        self.assertFalse(arrays["basis_present"][0])
        np.testing.assert_array_equal(arrays["A"][0], np.eye(2))

    def test_zero_process_kalman_controls_are_prefix_means(self):
        canonical = torch.tensor([
            [1., -2., 8.], [2., 1., -7.], [-1., 3., 6.],
            [.5, -.25, -5.], [4., 2., 4.]], dtype=torch.float64)
        arrays, metadata = core.run_stream(canonical, 0.0, math.pi / 4.0)
        outputs = {name: arrays["output"][:, position]
                   for position, name in enumerate(metadata["policy_names"])}
        prefix = np.cumsum(arrays["g"], axis=0) / np.arange(1, 6)[:, None]
        np.testing.assert_allclose(outputs["common_kalman"], prefix,
                                   rtol=2e-15, atol=2e-15)
        np.testing.assert_allclose(outputs["useful_oracle_kalman"], prefix,
                                   rtol=2e-15, atol=2e-15)
        np.testing.assert_allclose(arrays["common_kalman_posterior_variance"],
                                   5.0 / np.arange(1, 6), rtol=2e-15, atol=2e-15)
        expected_variance = np.column_stack((1.0 / np.arange(1, 6),
                                             4.0 / np.arange(1, 6)))
        np.testing.assert_allclose(
            arrays["useful_oracle_kalman_posterior_variance"], expected_variance,
            rtol=2e-15, atol=2e-15)
        np.testing.assert_array_equal(arrays["common_kalman_prior_weight"][0], 0.0)
        np.testing.assert_array_equal(arrays["useful_oracle_kalman_prior_weight"][0],
                                      np.zeros(2))

    def test_kalman_second_step_and_all_policy_recurrences(self):
        canonical = torch.tensor([
            [1., 0., 9.], [0., 2., 1.], [-1., 1., -1.],
            [3., -2., 2.], [.5, .25, -2.], [-.75, 1.5, 3.]],
            dtype=torch.float64)
        arrays, metadata = core.run_stream(canonical, 0.1, 0.0)
        names = metadata["policy_names"]
        output = {name: arrays["output"][:, names.index(name)] for name in names}
        for name in names:
            np.testing.assert_array_equal(output[name][0], arrays["g"][0])
        self.assertEqual(arrays["common_kalman_prior_weight"][1], 50.0 / 101.0)
        self.assertEqual(arrays["common_kalman_posterior_variance"][1], 255.0 / 101.0)
        np.testing.assert_allclose(arrays["useful_oracle_kalman_prior_weight"][1],
                                   (10.0 / 21.0, .5), rtol=0, atol=1e-16)
        np.testing.assert_allclose(arrays["useful_oracle_kalman_posterior_variance"][1],
                                   (11.0 / 21.0, 2.0), rtol=0, atol=1e-16)

        np.testing.assert_allclose(output["scalar_k0"], output["ema_q0p99"],
                                   rtol=2e-15, atol=2e-15)
        np.testing.assert_allclose(output["scalar_k1"], output["ema_q0p9"],
                                   rtol=2e-15, atol=2e-15)
        eye = np.eye(2)
        for index in range(1, len(canonical)):
            action = arrays["A"][index]
            h = arrays["native_h"][index]
            expected_i17 = (.9 * action @ arrays["native_i17_buffer"][index - 1] + h)
            np.testing.assert_allclose(arrays["native_i17_buffer"][index], expected_i17,
                                       rtol=2e-15, atol=2e-15)
            expected_cp = (.9 * action @ arrays["native_cp_buffer"][index - 1]
                           + (eye - .9 * action) @ h)
            np.testing.assert_allclose(arrays["native_cp_buffer"][index], expected_cp,
                                       rtol=2e-15, atol=2e-15)
            expected_cp_star = (core.RHO_STAR * action
                @ arrays["native_cp_star_buffer"][index - 1]
                + (eye - core.RHO_STAR * action) @ h)
            np.testing.assert_allclose(arrays["native_cp_star_buffer"][index],
                                       expected_cp_star, rtol=2e-15, atol=2e-15)
        for position, decay in enumerate(core.EMA_DECAYS):
            first = arrays["dema_first_state"][:, position]
            second = arrays["dema_second_state"][:, position]
            np.testing.assert_allclose(output[("dema_q0p9", "dema_q0p99",
                "dema_q0p999")[position]], 2.0 * first - second,
                rtol=2e-15, atol=2e-15)
        for key in (
                "ema_fixed_response_residual", "dema_first_response_residual",
                "dema_second_response_residual", "ema_common_response_residual",
                "common_kalman_response_residual", "common_kalman_variance_residual",
                "useful_oracle_kalman_response_residual",
                "useful_oracle_kalman_variance_residual",
                "oracle_useful_fast_response_residual",
                "oracle_useful_slow_response_residual",
                "oracle_useful_star_fast_response_residual", "scalar_response_residual",
                "native_i17_response_residual", "native_cp_response_residual",
                "native_cp_star_response_residual", "full_moment_residual"):
            self.assertLessEqual(float(np.max(np.abs(arrays[key]))), 5e-15, key)

    def test_external_rng_input_and_dummy_parameter_are_unchanged(self):
        canonical = torch.tensor([[1.25, -2.5, 4.0], [3.0, 4.0, -2.0],
                                  [-.5, .75, 1.0]], dtype=torch.float64)
        before_input = canonical.clone()
        before_rng = torch.get_rng_state().clone()
        with mock.patch.object(torch.optim.SGD, "step",
                               side_effect=AssertionError("optimizer.step called")):
            arrays, _ = core.run_stream(canonical, 0.01, 0.0)
        np.testing.assert_array_equal(canonical.numpy(), before_input.numpy())
        self.assertTrue(torch.equal(torch.get_rng_state(), before_rng))
        np.testing.assert_array_equal(arrays["g"], arrays["s"] + arrays["epsilon"])

    def test_budget_callback_and_numeric_schema(self):
        calls = []
        arrays, metadata = core.run_stream(
            torch.zeros((101, 3), dtype=torch.float64), 0.0, 0.0,
            check_budget=lambda: calls.append(len(calls)))
        self.assertEqual(len(calls), 4)
        self.assertEqual(metadata["array_shapes"],
                         {name: list(value.shape) for name, value in arrays.items()})
        self.assertEqual(metadata["array_dtypes"],
                         {name: str(value.dtype) for name, value in arrays.items()})
        for name, value in arrays.items():
            self.assertNotEqual(value.dtype, np.dtype("O"), name)
            if np.issubdtype(value.dtype, np.number):
                self.assertTrue(np.isfinite(value).all(), name)


if __name__ == "__main__":
    unittest.main()
