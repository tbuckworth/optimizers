"""Handcrafted CPU tests for the import-safe I18 tracking core."""
from __future__ import annotations

import math
import unittest
from unittest import mock

import numpy as np
import torch

import tracking_core as core


class TrackingCoreTests(unittest.TestCase):
    def test_registered_policy_order_and_optimal_decay(self):
        self.assertEqual(core.policy_names(0.0), (
            "raw", "ema_q0p9", "ema_q0p99", "ema_q0p999",
            "scalar_k0", "scalar_k0p5", "scalar_k0p9", "scalar_k1",
            "oracle_useful", "oracle_nuisance", "native_i17", "native_cp"))
        self.assertEqual(len(core.policy_names(0.03)), 13)
        self.assertEqual(core.policy_names(0.03)[4], "ema_optimal_oracle")
        self.assertIsNone(core.optimal_ema_decay(0.0))
        for drift in (0.01, 0.03):
            decay = core.optimal_ema_decay(drift)
            lag = decay / (1.0 - decay)
            residual = 2 * drift * drift * lag - 10 / (2 * lag + 1) ** 2
            self.assertLess(abs(residual), 1e-13)

    def test_invalid_inputs_are_rejected_before_observer_use(self):
        valid = torch.zeros((3, 2), dtype=torch.float64)
        cases = [torch.zeros((3, 2), dtype=torch.float32),
                 torch.zeros((3, 3), dtype=torch.float64),
                 torch.tensor([[float("nan"), 0.0]], dtype=torch.float64)]
        for bad in cases:
            with self.subTest(shape=tuple(bad.shape), dtype=str(bad.dtype)):
                with self.assertRaises(core.TrackingCoreError):
                    core.run_stream(bad, 0.0, 0.0)
        with self.assertRaises(core.TrackingCoreError):
            core.run_stream(valid, 0.02, 0.0)
        with self.assertRaises(core.TrackingCoreError):
            core.run_stream(valid, 0.0, 0.1)
        with self.assertRaises(core.TrackingCoreError):
            core.run_stream(valid, 0.0, 0.0, check_budget=3)

    def test_saved_stream_observer_and_full_moment_recurrences(self):
        noise = torch.tensor([[1.0, -2.0], [2.0, 1.0], [-1.0, 3.0],
                              [0.5, -0.25], [4.0, 2.0]], dtype=torch.float64)
        arrays, metadata = core.run_stream(noise, 0.03, 0.0)
        self.assertEqual(metadata["schema"], "i18_tracking_stream_metadata_v1")
        self.assertEqual(metadata["array_order"], list(arrays))
        np.testing.assert_array_equal(arrays["g"], arrays["s"] + arrays["epsilon"])
        np.testing.assert_array_equal(arrays["noise"], noise.numpy())
        np.testing.assert_array_equal(arrays["s"][:, 0], np.arange(5) * .03)
        np.testing.assert_array_equal(arrays["s"][:, 1], 0.0)

        mean = arrays["g"][0].copy()
        full = np.zeros((2, 2))
        initialized = False
        for index in range(5):
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
        self.assertEqual(arrays["native_useful_squared_alignment"][0], 0.0)
        self.assertTrue(np.all(np.isfinite(arrays["full_gap"])))

    def test_all_response_rules_reconstruct_and_duplicate_controls_match(self):
        noise = torch.tensor([[1., 0.], [0., 2.], [-1., 1.], [3., -2.],
                              [.5, .25], [-.75, 1.5]], dtype=torch.float64)
        arrays, metadata = core.run_stream(noise, 0.01, math.pi / 4)
        names = metadata["policy_names"]
        output = {name: arrays["output"][:, names.index(name)] for name in names}
        np.testing.assert_allclose(output["scalar_k0"], output["ema_q0p99"],
                                   rtol=2e-15, atol=2e-15)
        np.testing.assert_allclose(output["scalar_k1"], output["ema_q0p9"],
                                   rtol=2e-15, atol=2e-15)
        for name in names:
            np.testing.assert_array_equal(output[name][0], arrays["g"][0])

        eye = np.eye(2)
        for index in range(1, len(noise)):
            action, h = arrays["A"][index], arrays["native_h"][index]
            expected_b = .9 * action @ arrays["native_i17_buffer"][index - 1] + h
            np.testing.assert_allclose(arrays["native_i17_buffer"][index], expected_b,
                                       rtol=2e-15, atol=2e-15)
            np.testing.assert_allclose(output["native_i17"][index],
                                       (eye - .9 * action) @ expected_b,
                                       rtol=2e-15, atol=2e-15)
            expected_cp = (.9 * action @ output["native_cp"][index - 1]
                           + (eye - .9 * action) @ h)
            np.testing.assert_allclose(output["native_cp"][index], expected_cp,
                                       rtol=2e-15, atol=2e-15)
            for position, k in enumerate(core.SCALAR_K):
                expected = (.9 * k * arrays["scalar_buffer"][index - 1, position]
                            + k * arrays["g"][index] + (1 - k) * arrays["mu"][index])
                np.testing.assert_allclose(arrays["scalar_buffer"][index, position], expected,
                                           rtol=2e-15, atol=2e-15)
                np.testing.assert_allclose(output[("scalar_k0", "scalar_k0p5",
                    "scalar_k0p9", "scalar_k1")[position]][index],
                    (1 - .9 * k) * expected, rtol=2e-15, atol=2e-15)
        for key in ("native_action_raw_error", "native_i17_response_residual",
                    "native_cp_response_residual", "scalar_response_residual",
                    "full_moment_residual"):
            self.assertLessEqual(float(np.max(np.abs(arrays[key]))), 4e-15)

    def test_rotation_transforms_complete_stream_and_oracle_directions(self):
        noise = torch.tensor([[1., 2.], [-3., .5], [.25, -1.]], dtype=torch.float64)
        base, metadata0 = core.run_stream(noise, 0.03, 0.0)
        rotated, metadata1 = core.run_stream(noise, 0.03, math.pi / 4)
        angle = math.pi / 4
        q = np.array([[math.cos(angle), -math.sin(angle)],
                      [math.sin(angle), math.cos(angle)]])
        np.testing.assert_array_equal(rotated["noise"], base["noise"])
        for key in ("epsilon", "g", "s"):
            np.testing.assert_allclose(rotated[key], base[key] @ q.T,
                                       rtol=1e-14, atol=1e-14)
        for policy in ("raw", "ema_q0p9", "ema_q0p99", "ema_q0p999",
                       "ema_optimal_oracle", "scalar_k0", "scalar_k0p5",
                       "scalar_k0p9", "scalar_k1", "oracle_useful",
                       "oracle_nuisance"):
            left = rotated["output"][:, metadata1["policy_names"].index(policy)]
            right = base["output"][:, metadata0["policy_names"].index(policy)] @ q.T
            np.testing.assert_allclose(left, right, rtol=1e-12, atol=1e-13)

    def test_budget_callback_runs_at_start_every_fifty_and_end(self):
        calls = []
        core.run_stream(torch.zeros((101, 2), dtype=torch.float64), 0.0, 0.0,
                        check_budget=lambda: calls.append(len(calls)))
        self.assertEqual(len(calls), 4)  # t=1, t=51, t=101, final

    def test_input_and_external_rng_are_unchanged_and_dummy_parameter_is_not_stepped(self):
        noise = torch.tensor([[1.25, -2.5], [3.0, 4.0], [-.5, .75]],
                             dtype=torch.float64)
        before_noise = noise.clone()
        before_rng = torch.get_rng_state().clone()
        with mock.patch.object(torch.optim.SGD, "step",
                               side_effect=AssertionError("optimizer.step called")):
            arrays, _ = core.run_stream(noise, 0.01, 0.0)
        np.testing.assert_array_equal(noise.numpy(), before_noise.numpy())
        self.assertTrue(torch.equal(torch.get_rng_state(), before_rng))
        # The saved gradients are the exogenous stream itself; no model update
        # can feed a parameter-dependent term back into later observations.
        np.testing.assert_array_equal(arrays["g"], arrays["s"] + arrays["epsilon"])


if __name__ == "__main__":
    unittest.main()
