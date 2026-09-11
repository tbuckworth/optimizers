#!/usr/bin/env python3
"""Deterministic unit tests for the bounded I20 NumPy response core."""
from __future__ import annotations

import math
import unittest

import numpy as np

import response_core as core


def _projector(vector: tuple[float, float]) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64)
    value = value / np.linalg.norm(value)
    return np.outer(value, value)


def _saved_mean(g: np.ndarray) -> np.ndarray:
    mean = np.empty_like(g)
    mean[0] = g[0]
    for index in range(1, len(g)):
        mean[index] = 0.99 * mean[index - 1] + 0.01 * g[index]
    return mean


def _fixture() -> tuple[np.ndarray, ...]:
    g = np.array([
        [2.0, -1.0],
        [3.0, 5.0],
        [-2.0, 7.0],
        [11.0, 1.0],
        [-4.0, -3.0],
    ], dtype=np.float64)
    mu = _saved_mean(g)
    p0 = _projector((1.0, 0.0))
    p1 = _projector((3.0, 4.0))
    p2 = _projector((4.0, -3.0))
    native = np.stack((np.eye(2), p1, p2, p0, p1)).astype(np.float64)
    native_present = np.array((False, True, True, True, True), dtype=np.bool_)
    full = np.stack((np.zeros((2, 2)), p0, p0, p0, p0)).astype(np.float64)
    full_present = np.array((False, True, True, True, True), dtype=np.bool_)
    useful_axis = np.array((3.0 / 5.0, 4.0 / 5.0), dtype=np.float64)
    return g, mu, native, native_present, full, full_present, useful_axis


def _run_fixture() -> dict[str, np.ndarray]:
    return core.reconstruct(*_fixture())


def _policy(arrays: dict[str, np.ndarray], name: str) -> np.ndarray:
    return arrays["output"][:, core.POLICIES.index(name)]


class ResponseCoreTests(unittest.TestCase):
    def test_constants_and_exact_estimator_major_policy_order(self):
        self.assertEqual(core.ESTIMATORS, (
            "native", "full_legacy", "ew99", "ew999",
            "oracle_useful", "oracle_nuisance"))
        self.assertEqual(core.RHO_LABELS, ("r0p9", "rstar"))
        self.assertEqual(core.RHOS, (
            0.9, (2.1 - math.sqrt(0.41)) / 2.0))
        self.assertEqual(core.RESPONSES, ("cp", "rec99", "rec9"))
        expected = tuple(
            f"{estimator}/{rho}/{response}"
            for estimator in core.ESTIMATORS
            for rho in core.RHO_LABELS
            for response in core.RESPONSES)
        self.assertEqual(core.POLICIES, expected)
        self.assertEqual(len(core.POLICIES), 36)

    def test_exact_array_schema_dtypes_and_bytes(self):
        arrays = _run_fixture()
        self.assertEqual(tuple(arrays), core.ARRAY_ORDER)
        self.assertEqual(
            {name: value.shape for name, value in arrays.items()},
            core.expected_shapes(5))
        for name, value in arrays.items():
            expected_dtype = np.dtype(np.bool_) if name == "direction_present" \
                else np.dtype(np.float64)
            self.assertEqual(value.dtype, expected_dtype, name)
            self.assertNotEqual(value.dtype, np.dtype("O"), name)
        self.assertEqual(sum(value.nbytes for value in arrays.values()), 1062 * 5)
        shapes_4000 = core.expected_shapes(4000)
        expected_bytes = sum(
            math.prod(shape) * (1 if name == "direction_present" else 8)
            for name, shape in shapes_4000.items())
        self.assertEqual(expected_bytes, 4_248_000)

    def test_zero_start_ew_mass_moments_ties_and_masks(self):
        g, mu, native, native_mask, full, full_mask, axis = _fixture()
        arrays = core.reconstruct(
            g, mu, native, native_mask, full, full_mask, axis)
        np.testing.assert_array_equal(arrays["weighted_moment"][0], 0.0)
        np.testing.assert_allclose(
            arrays["weight_mass"][0], (1.0 - 0.99, 1.0 - 0.999),
            rtol=0, atol=0)
        np.testing.assert_array_equal(arrays["normalized_moment"][0], 0.0)
        np.testing.assert_array_equal(arrays["weighted_eigenvalues"][0], 0.0)
        np.testing.assert_array_equal(arrays["weighted_gap"][0], 0.0)
        np.testing.assert_array_equal(
            arrays["direction_present"][0],
            np.array((False, False, False, False, True, True)))
        np.testing.assert_array_equal(arrays["actions"][0, :4],
                                      np.broadcast_to(np.eye(2), (4, 2, 2)))
        np.testing.assert_array_equal(arrays["useful_squared_alignment"][0, :4], 0.0)

        residual = g[1] - mu[1]
        outer = np.outer(residual, residual)
        np.testing.assert_allclose(
            arrays["weighted_moment"][1, 0], 0.01 * outer,
            rtol=2e-15, atol=2e-15)
        np.testing.assert_allclose(
            arrays["weighted_moment"][1, 1], 0.001 * outer,
            rtol=2e-15, atol=2e-15)
        expected_mass = np.array((1.0 - 0.99, 1.0 - 0.999))
        expected_mass = (np.array((0.99, 0.999)) * expected_mass
                         + np.array((1.0 - 0.99, 1.0 - 0.999)))
        np.testing.assert_array_equal(arrays["weight_mass"][1], expected_mass)

        constant = np.broadcast_to(g[:1], g.shape).copy()
        tied = core.reconstruct(
            constant, constant.copy(), native, native_mask,
            full, full_mask, axis)
        np.testing.assert_array_equal(tied["direction_present"][:, 2:4], False)
        np.testing.assert_array_equal(
            tied["actions"][:, 2:4],
            np.broadcast_to(np.eye(2), (len(g), 2, 2, 2)))

    def test_first_output_and_equal_decay_control_are_exact(self):
        arrays = _run_fixture()
        g = _fixture()[0]
        np.testing.assert_array_equal(
            arrays["output"][0], np.broadcast_to(g[0], (36, 2)))
        ema = np.empty_like(g)
        ema[0] = g[0]
        for index in range(1, len(g)):
            ema[index] = (0.9 * ema[index - 1]
                          + (1.0 - 0.9) * g[index])
        for estimator in core.ESTIMATORS:
            np.testing.assert_array_equal(
                _policy(arrays, f"{estimator}/r0p9/rec9"), ema)

    def test_fixed_oracles_close_the_three_parent_formulas(self):
        arrays = _run_fixture()
        g, mu, *_rest, axis = _fixture()
        useful = np.outer(axis, axis)
        nuisance = np.eye(2) - useful

        fast = {rho: np.empty_like(g) for rho in core.RHOS}
        for rho in core.RHOS:
            fast[rho][0] = g[0]
            for index in range(1, len(g)):
                fast[rho][index] = (rho * fast[rho][index - 1]
                                    + (1.0 - rho) * g[index])
        expected_useful = fast[0.9] @ useful + mu @ nuisance
        expected_useful_star = fast[core.RHOS[1]] @ useful + mu @ nuisance
        expected_nuisance = fast[0.9] @ nuisance + mu @ useful
        np.testing.assert_allclose(
            _policy(arrays, "oracle_useful/r0p9/cp"), expected_useful,
            rtol=3e-15, atol=3e-15)
        np.testing.assert_allclose(
            _policy(arrays, "oracle_useful/rstar/cp"), expected_useful_star,
            rtol=3e-15, atol=3e-15)
        np.testing.assert_allclose(
            _policy(arrays, "oracle_nuisance/r0p9/cp"), expected_nuisance,
            rtol=3e-15, atol=3e-15)

    def test_native_cp_uses_literal_parent_recurrence_for_both_rhos(self):
        arrays = _run_fixture()
        g, mu, native, native_mask, *_ = _fixture()
        eye = np.eye(2)
        actions = np.where(native_mask[:, None, None], native, eye)
        for rho_label, rho in zip(core.RHO_LABELS, core.RHOS):
            expected = np.empty_like(g)
            expected[0] = g[0]
            for index in range(1, len(g)):
                projection = actions[index]
                h = (projection @ g[index] + mu[index]
                     - projection @ mu[index])
                expected[index] = (rho * projection @ expected[index - 1]
                                   + (eye - rho * projection) @ h)
            np.testing.assert_allclose(
                _policy(arrays, f"native/{rho_label}/cp"), expected,
                rtol=2e-15, atol=2e-15)

    def test_fixed_cp_matches_rec99_but_moving_projection_can_differ(self):
        arrays = _run_fixture()
        for rho_label in core.RHO_LABELS:
            np.testing.assert_allclose(
                _policy(arrays, f"oracle_useful/{rho_label}/cp"),
                _policy(arrays, f"oracle_useful/{rho_label}/rec99"),
                rtol=3e-15, atol=3e-15)
            np.testing.assert_allclose(
                _policy(arrays, f"full_legacy/{rho_label}/cp"),
                _policy(arrays, f"full_legacy/{rho_label}/rec99"),
                rtol=3e-15, atol=3e-15)
        moving_difference = (
            _policy(arrays, "native/r0p9/cp")
            - _policy(arrays, "native/r0p9/rec99"))
        self.assertGreater(float(np.max(np.abs(moving_difference))), 1e-8)

    def test_action_diagnostics_follow_masks_and_projector_geometry(self):
        arrays = _run_fixture()
        _g, _mu, _native, native_mask, _full, full_mask, axis = _fixture()
        np.testing.assert_array_equal(arrays["direction_present"][:, 0], native_mask)
        np.testing.assert_array_equal(arrays["direction_present"][:, 1], full_mask)
        np.testing.assert_array_equal(arrays["action_change_norm"][0], 0.0)
        expected_change = np.linalg.norm(
            arrays["actions"][1:] - arrays["actions"][:-1], axis=(2, 3))
        np.testing.assert_allclose(arrays["action_change_norm"][1:], expected_change,
                                   rtol=0, atol=0)
        np.testing.assert_allclose(
            arrays["useful_squared_alignment"][:, 4], 1.0,
            rtol=0, atol=3e-16)
        np.testing.assert_allclose(
            arrays["useful_squared_alignment"][:, 5], 0.0,
            rtol=0, atol=3e-16)
        for projection in arrays["actions"].reshape(-1, 2, 2):
            np.testing.assert_allclose(projection, projection.T, rtol=0, atol=2e-15)
            np.testing.assert_allclose(projection @ projection, projection,
                                       rtol=0, atol=2e-15)

    def test_inputs_are_not_modified(self):
        values = _fixture()
        before = tuple(value.copy() for value in values)
        core.reconstruct(*values)
        for actual, expected in zip(values, before):
            np.testing.assert_array_equal(actual, expected)

    def test_nonoracle_outputs_do_not_use_privileged_axis(self):
        values = list(_fixture())
        first = core.reconstruct(*values)
        values[-1] = np.array((1.0, 0.0), dtype=np.float64)
        second = core.reconstruct(*values)
        for name in ('actions', 'direction_present'):
            np.testing.assert_array_equal(first[name][:, :4], second[name][:, :4])
        for name in ('weighted_moment', 'weight_mass', 'normalized_moment',
                     'weighted_eigenvalues', 'weighted_gap'):
            np.testing.assert_array_equal(first[name], second[name])
        np.testing.assert_array_equal(first['output'][:, :24], second['output'][:, :24])
        self.assertFalse(np.array_equal(first['actions'][:, 4:], second['actions'][:, 4:]))

    def test_native_action_is_copied_and_absent_identity_is_enforced(self):
        values = _fixture()
        arrays = core.reconstruct(*values)
        np.testing.assert_array_equal(arrays["actions"][:, 0], values[2])
        malformed = list(values)
        malformed_native = values[2].copy()
        malformed_native[0] = 0.0
        malformed[2] = malformed_native
        with self.assertRaises(core.ResponseCoreError):
            core.reconstruct(*malformed)

    def test_malformed_numeric_mask_and_geometry_inputs_are_rejected(self):
        base = _fixture()

        def rejected(position: int, replacement: object) -> None:
            values = list(base)
            values[position] = replacement
            with self.assertRaises(core.ResponseCoreError):
                core.reconstruct(*values)

        rejected(0, base[0].astype(np.float32))
        rejected(1, base[1][:-1].copy())
        nonfinite = base[2].copy()
        nonfinite[2, 0, 0] = np.nan
        rejected(2, nonfinite)
        rejected(3, base[3].astype(np.int64))
        rejected(5, np.zeros((len(base[0]), 1), dtype=np.bool_))
        rejected(6, np.array((1.0, 1.0), dtype=np.float64))
        non_projector = base[2].copy()
        non_projector[1] = np.array(((1.0, 1.0), (0.0, 0.0)))
        rejected(2, non_projector)
        with self.assertRaises(core.ResponseCoreError):
            core.expected_shapes(True)
        with self.assertRaises(core.ResponseCoreError):
            core.expected_shapes(0)


if __name__ == "__main__":
    unittest.main()
