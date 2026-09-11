#!/usr/bin/env python3
"""Pure NumPy reconstruction core for the frozen I20 response study.

The module has no file I/O, randomness, native-observer dependency, or hidden
state.  ``reconstruct`` operates only on the supplied saved stream fields.
"""
from __future__ import annotations

import math

import numpy as np


DIMENSION = 2
BETA = 0.99
EW_DECAYS = (0.99, 0.999)
GAP_RELATIVE_TOLERANCE = 1e-10

ESTIMATORS = (
    "native",
    "full_legacy",
    "ew99",
    "ew999",
    "oracle_useful",
    "oracle_nuisance",
)
RHO_LABELS = ("r0p9", "rstar")
RHOS = (0.9, (2.1 - math.sqrt(0.41)) / 2.0)
RESPONSES = ("cp", "rec99", "rec9")
POLICIES = tuple(
    f"{estimator}/{rho_label}/{response}"
    for estimator in ESTIMATORS
    for rho_label in RHO_LABELS
    for response in RESPONSES
)
ARRAY_ORDER = (
    "actions",
    "direction_present",
    "weighted_moment",
    "weight_mass",
    "normalized_moment",
    "weighted_eigenvalues",
    "weighted_gap",
    "output",
    "useful_squared_alignment",
    "action_change_norm",
)

_PROJECTOR_TOLERANCE = 1e-10


class ResponseCoreError(ValueError):
    """Raised when supplied stream fields do not match the frozen I20 API."""


def expected_shapes(horizon: int) -> dict[str, tuple[int, ...]]:
    """Return the exact I20 array schema for a positive integer horizon."""
    if type(horizon) is not int or horizon < 1:
        raise ResponseCoreError("horizon must be a positive int")
    return {
        "actions": (horizon, 6, 2, 2),
        "direction_present": (horizon, 6),
        "weighted_moment": (horizon, 2, 2, 2),
        "weight_mass": (horizon, 2),
        "normalized_moment": (horizon, 2, 2, 2),
        "weighted_eigenvalues": (horizon, 2, 2),
        "weighted_gap": (horizon, 2),
        "output": (horizon, 36, 2),
        "useful_squared_alignment": (horizon, 6),
        "action_change_norm": (horizon, 6),
    }


def _float64_array(value: object, shape: tuple[int, ...], label: str) -> np.ndarray:
    if type(value) is not np.ndarray or value.dtype != np.dtype(np.float64) \
            or value.shape != shape:
        raise ResponseCoreError(
            f"{label} must be a float64 ndarray with shape {shape}")
    if not np.isfinite(value).all():
        raise ResponseCoreError(f"{label} must be finite")
    return value


def _bool_array(value: object, shape: tuple[int, ...], label: str) -> np.ndarray:
    if type(value) is not np.ndarray or value.dtype != np.dtype(np.bool_) \
            or value.shape != shape:
        raise ResponseCoreError(
            f"{label} must be a bool ndarray with shape {shape}")
    return value


def _validate_present_projectors(
        actions: np.ndarray, present: np.ndarray, label: str) -> None:
    """Require each available saved direction to be a rank-one projector."""
    active = actions[present]
    if active.shape[0] == 0:
        return
    transposed = np.swapaxes(active, -1, -2)
    products = active @ active
    if (np.max(np.abs(active - transposed)) > _PROJECTOR_TOLERANCE
            or np.max(np.abs(products - active)) > _PROJECTOR_TOLERANCE
            or np.max(np.abs(np.trace(active, axis1=1, axis2=2) - 1.0))
            > _PROJECTOR_TOLERANCE):
        raise ResponseCoreError(
            f"{label} entries marked present must be rank-one orthogonal projectors")


def _validate_native_absent_identity(
        actions: np.ndarray, present: np.ndarray) -> None:
    """The saved I19 native action already contains its identity fallback."""
    inactive = actions[~present]
    if inactive.shape[0] and np.max(
            np.abs(inactive - np.eye(DIMENSION, dtype=np.float64))) \
            > _PROJECTOR_TOLERANCE:
        raise ResponseCoreError(
            "native_action entries marked absent must be identity")


def _validate_inputs(
        g: object,
        mu: object,
        native_action: object,
        native_present: object,
        full_action: object,
        full_present: object,
        useful_axis: object,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
           np.ndarray, np.ndarray, np.ndarray, int]:
    if type(g) is not np.ndarray or g.dtype != np.dtype(np.float64) \
            or g.ndim != 2 or g.shape[0] < 1 or g.shape[1] != DIMENSION:
        raise ResponseCoreError("g must be a nonempty float64 ndarray with shape (T,2)")
    horizon = int(g.shape[0])
    g = _float64_array(g, (horizon, DIMENSION), "g")
    mu = _float64_array(mu, (horizon, DIMENSION), "mu")
    native_action = _float64_array(
        native_action, (horizon, DIMENSION, DIMENSION), "native_action")
    native_present = _bool_array(native_present, (horizon,), "native_present")
    full_action = _float64_array(
        full_action, (horizon, DIMENSION, DIMENSION), "full_action")
    full_present = _bool_array(full_present, (horizon,), "full_present")
    useful_axis = _float64_array(useful_axis, (DIMENSION,), "useful_axis")

    if not np.isclose(np.dot(useful_axis, useful_axis), 1.0,
                      rtol=0.0, atol=_PROJECTOR_TOLERANCE):
        raise ResponseCoreError("useful_axis must have unit Euclidean norm")
    _validate_present_projectors(native_action, native_present, "native_action")
    _validate_present_projectors(full_action, full_present, "full_action")
    _validate_native_absent_identity(native_action, native_present)
    return (g, mu, native_action, native_present, full_action, full_present,
            useful_axis, horizon)


def reconstruct(
        g: np.ndarray,
        mu: np.ndarray,
        native_action: np.ndarray,
        native_present: np.ndarray,
        full_action: np.ndarray,
        full_present: np.ndarray,
        useful_axis: np.ndarray,
) -> dict[str, np.ndarray]:
    """Reconstruct all six I20 estimators and 36 causal responses.

    ``mu`` is the saved post-ingest I19 mean and is deliberately not recomputed.
    Saved native actions are copied exactly, including their already-materialized
    identity fallback.  Unavailable full-legacy entries are replaced by identity.
    """
    (g, mu, native_action, native_present, full_action, full_present,
     useful_axis, horizon) = _validate_inputs(
        g, mu, native_action, native_present, full_action, full_present,
        useful_axis)

    eye = np.eye(DIMENSION, dtype=np.float64)
    shapes = expected_shapes(horizon)
    arrays: dict[str, np.ndarray] = {
        name: np.empty(shape, dtype=(np.bool_ if name == "direction_present"
                                     else np.float64))
        for name, shape in shapes.items()
    }

    actions = arrays["actions"]
    direction_present = arrays["direction_present"]
    actions[:, 0] = native_action
    actions[:, 1] = np.where(full_present[:, None, None], full_action, eye)
    direction_present[:, 0] = native_present
    direction_present[:, 1] = full_present

    weighted_moment = arrays["weighted_moment"]
    weight_mass = arrays["weight_mass"]
    normalized_moment = arrays["normalized_moment"]
    current_moment = np.zeros((len(EW_DECAYS), DIMENSION, DIMENSION),
                              dtype=np.float64)
    current_mass = np.zeros(len(EW_DECAYS), dtype=np.float64)
    one_minus_decay = 1.0 - np.asarray(EW_DECAYS, dtype=np.float64)
    decay_array = np.asarray(EW_DECAYS, dtype=np.float64)
    for index in range(horizon):
        residual = g[index] - mu[index]
        residual_outer = np.outer(residual, residual)
        current_moment = (decay_array[:, None, None] * current_moment
                          + one_minus_decay[:, None, None] * residual_outer)
        current_mass = decay_array * current_mass + one_minus_decay
        weighted_moment[index] = current_moment
        weight_mass[index] = current_mass
        normalized_moment[index] = current_moment / current_mass[:, None, None]

    eigenvalues, eigenvectors = np.linalg.eigh(normalized_moment)
    arrays["weighted_eigenvalues"][:] = eigenvalues
    gaps = eigenvalues[:, :, 1] - eigenvalues[:, :, 0]
    arrays["weighted_gap"][:] = gaps
    ew_present = gaps > (GAP_RELATIVE_TOLERANCE
                         * np.maximum(1.0, np.abs(eigenvalues[:, :, 1])))
    direction_present[:, 2:4] = ew_present
    for estimator_offset in range(len(EW_DECAYS)):
        vectors = eigenvectors[:, estimator_offset, :, 1]
        projectors = vectors[:, :, None] * vectors[:, None, :]
        actions[:, estimator_offset + 2] = np.where(
            ew_present[:, estimator_offset, None, None], projectors, eye)

    useful_projector = np.outer(useful_axis, useful_axis)
    nuisance_projector = eye - useful_projector
    actions[:, 4] = useful_projector
    actions[:, 5] = nuisance_projector
    direction_present[:, 4:6] = True

    alignments = np.einsum(
        "i,teij,j->te", useful_axis, actions, useful_axis, optimize=False)
    arrays["useful_squared_alignment"][:] = np.where(
        direction_present, alignments, 0.0)
    action_change = arrays["action_change_norm"]
    action_change[0] = 0.0
    if horizon > 1:
        action_change[1:] = np.linalg.norm(
            actions[1:] - actions[:-1], axis=(2, 3))

    # State layout is estimator, rho, response, coordinate.  The assignment to
    # output below preserves the frozen estimator-major/rho/response order.
    state = np.broadcast_to(g[0], (len(ESTIMATORS), len(RHOS),
                                   len(RESPONSES), DIMENSION)).copy()
    output = arrays["output"].reshape(
        horizon, len(ESTIMATORS), len(RHOS), len(RESPONSES), DIMENSION)
    output[0] = state
    for index in range(1, horizon):
        projection = actions[index]
        gradient = g[index]
        mean = mu[index]
        projected_gradient = np.einsum(
            "eij,j->ei", projection, gradient, optimize=False)
        projected_mean = np.einsum(
            "eij,j->ei", projection, mean, optimize=False)
        h = projected_gradient + mean - projected_mean

        next_state = np.empty_like(state)
        for rho_index, rho in enumerate(RHOS):
            # Constant-preserving response.  Keep the literal frozen formula:
            # rho P d_(t-1) + (I-rho P) h.
            previous_cp = state[:, rho_index, 0]
            next_state[:, rho_index, 0] = (
                rho * np.einsum(
                    "eij,ej->ei", projection, previous_cp, optimize=False)
                + np.einsum("eij,ej->ei", eye - rho * projection, h,
                            optimize=False))

            previous_rec99 = state[:, rho_index, 1]
            rec99_decay = (rho * projection
                           + BETA * (eye - projection))
            next_state[:, rho_index, 1] = (
                np.einsum("eij,ej->ei", rec99_decay, previous_rec99,
                          optimize=False)
                + np.einsum("eij,j->ei", eye - rec99_decay, gradient,
                            optimize=False))

            previous_rec9 = state[:, rho_index, 2]
            if rho == 0.9:
                # The registered equal-decay control is exactly scalar EMA .9.
                next_state[:, rho_index, 2] = (
                    rho * previous_rec9 + (1.0 - rho) * gradient)
            else:
                rec9_decay = (rho * projection
                              + 0.9 * (eye - projection))
                next_state[:, rho_index, 2] = (
                    np.einsum("eij,ej->ei", rec9_decay, previous_rec9,
                              optimize=False)
                    + np.einsum("eij,j->ei", eye - rec9_decay, gradient,
                                optimize=False))
        state = next_state
        output[index] = state

    for name, value in arrays.items():
        if value.shape != shapes[name] or value.dtype not in (
                np.dtype(np.float64), np.dtype(np.bool_)):
            raise AssertionError("internal I20 array schema violation: " + name)
        if value.dtype == np.dtype(np.float64) and not np.isfinite(value).all():
            raise FloatingPointError("nonfinite reconstructed array: " + name)
    return arrays


__all__ = [
    "ResponseCoreError",
    "DIMENSION",
    "BETA",
    "EW_DECAYS",
    "GAP_RELATIVE_TOLERANCE",
    "ESTIMATORS",
    "RHO_LABELS",
    "RHOS",
    "RESPONSES",
    "POLICIES",
    "ARRAY_ORDER",
    "expected_shapes",
    "reconstruct",
]
