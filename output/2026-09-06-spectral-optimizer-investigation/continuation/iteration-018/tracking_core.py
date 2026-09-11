#!/usr/bin/env python3
"""Pure CPU stream mechanism for the prospective I18 tracking experiment.

Importing this module does not generate a stream or read an experimental file.
"""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path
from typing import Callable

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
_SPEC = importlib.util.spec_from_file_location(
    "_i18_canonical_spectral_filter", REPO / "spectral_filter.py")
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover
    raise ImportError("cannot load canonical spectral_filter.py")
_native = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_native)
SpectralGradientFilter = _native.SpectralGradientFilter

HORIZON = 4000
DIMENSION = 2
DRIFTS = (0.0, 0.01, 0.03)
ROTATIONS = (0.0, math.pi / 4.0)
RHO = 0.9
BETA = 0.99
EMA_DECAYS = (0.9, 0.99, 0.999)
SCALAR_K = (0.0, 0.5, 0.9, 1.0)
FULL_GAP_RELATIVE_TOLERANCE = 1e-10
FILTER_CONFIGURATION = {
    "rank": 1, "decay": BETA, "warmup": 0, "filter_strength": 1.0,
    "adaptive": "none", "normalize": "none", "weighting": "hard",
    "stable_update": True, "relative_eig_tol": 1e-8,
    "absolute_eig_floor": 0.0, "stabilize_every": 100,
}


class TrackingCoreError(ValueError):
    """Raised when a stream or registered configuration is malformed."""


class _DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.vector = torch.nn.Parameter(torch.zeros(DIMENSION, dtype=torch.float64))


def optimal_ema_decay(drift: float) -> float | None:
    """Return the registered analytic shared-EMA control for nonzero drift."""
    if type(drift) is not float or drift not in DRIFTS:
        raise TrackingCoreError("drift must be one registered float value")
    if drift == 0.0:
        return None
    low, high = 0.0, 10000.0
    for _ in range(100):
        middle = (low + high) / 2.0
        value = 2.0 * drift * drift * middle - 10.0 / (2.0 * middle + 1.0) ** 2
        if value > 0.0:
            high = middle
        else:
            low = middle
    lag = (low + high) / 2.0
    return lag / (lag + 1.0)


def policy_names(drift: float) -> tuple[str, ...]:
    """Return the frozen output-column order for one drift cell."""
    optimum = optimal_ema_decay(drift)
    names = ["raw", "ema_q0p9", "ema_q0p99", "ema_q0p999"]
    if optimum is not None:
        names.append("ema_optimal_oracle")
    names.extend(("scalar_k0", "scalar_k0p5", "scalar_k0p9", "scalar_k1",
                  "oracle_useful", "oracle_nuisance", "native_i17", "native_cp"))
    return tuple(names)


def _rotation(angle: float) -> torch.Tensor:
    cosine, sine = math.cos(angle), math.sin(angle)
    return torch.tensor([[cosine, -sine], [sine, cosine]], dtype=torch.float64)


def _finite_tensor(value: torch.Tensor, label: str) -> torch.Tensor:
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError(label + " is nonfinite")
    return value


def _to_numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy().copy()


def _validate_noise(noise: torch.Tensor) -> int:
    if type(noise) is not torch.Tensor or noise.device.type != "cpu" \
            or noise.dtype != torch.float64 or noise.layout != torch.strided \
            or noise.ndim != 2 or noise.shape[1] != DIMENSION or noise.shape[0] < 1:
        raise TrackingCoreError("noise must be a nonempty dense CPU float64 tensor [H,2]")
    if not bool(torch.isfinite(noise).all()):
        raise TrackingCoreError("noise must be finite")
    return int(noise.shape[0])


def run_stream(noise: torch.Tensor, drift: float, rotation: float, *,
               check_budget: Callable[[], None] | None = None) \
        -> tuple[dict[str, np.ndarray], dict]:
    """Run one shared exogenous stream through all registered response rules."""
    horizon = _validate_noise(noise)
    if type(rotation) is not float or rotation not in ROTATIONS:
        raise TrackingCoreError("rotation must be one registered float value")
    names = policy_names(drift)
    optimum = optimal_ema_decay(drift)
    if check_budget is not None and not callable(check_budget):
        raise TrackingCoreError("check_budget must be callable or None")
    check = check_budget if check_budget is not None else (lambda: None)

    q = _rotation(rotation)
    useful = q[:, :1] @ q[:, :1].T
    nuisance = torch.eye(DIMENSION, dtype=torch.float64) - useful
    times = torch.arange(horizon, dtype=torch.float64)
    signal_base = torch.stack((drift * times, torch.zeros_like(times)), dim=1)
    signal = signal_base @ q.T
    rotated_noise = noise @ q.T
    gradients = _finite_tensor(signal + rotated_noise, "gradient stream")

    model = _DummyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    tracker = SpectralGradientFilter(model, optimizer, **FILTER_CONFIGURATION)

    p = len(names)
    arrays = {
        "noise": np.empty((horizon, 2), dtype=np.float64),
        "epsilon": np.empty((horizon, 2), dtype=np.float64),
        "g": np.empty((horizon, 2), dtype=np.float64),
        "s": np.empty((horizon, 2), dtype=np.float64),
        "mu": np.empty((horizon, 2), dtype=np.float64),
        "A": np.empty((horizon, 2, 2), dtype=np.float64),
        "native_delivery": np.empty((horizon, 2), dtype=np.float64),
        "native_h": np.empty((horizon, 2), dtype=np.float64),
        "full_moment": np.empty((horizon, 2, 2), dtype=np.float64),
        "full_action": np.zeros((horizon, 2, 2), dtype=np.float64),
        "full_gap": np.empty(horizon, dtype=np.float64),
        "full_basis_present": np.zeros(horizon, dtype=np.bool_),
        "full_basis_V": np.zeros((horizon, 2), dtype=np.float64),
        "basis_present": np.zeros(horizon, dtype=np.bool_),
        "basis_V": np.zeros((horizon, 2), dtype=np.float64),
        "basis_S": np.zeros(horizon, dtype=np.float64),
        "output": np.empty((horizon, p, 2), dtype=np.float64),
        "native_i17_buffer": np.empty((horizon, 2), dtype=np.float64),
        "scalar_buffer": np.empty((horizon, len(SCALAR_K), 2), dtype=np.float64),
        "native_action_raw_error": np.empty((horizon, 2), dtype=np.float64),
        "native_i17_response_residual": np.zeros((horizon, 2), dtype=np.float64),
        "native_cp_response_residual": np.zeros((horizon, 2), dtype=np.float64),
        "scalar_response_residual": np.zeros((horizon, len(SCALAR_K), 2), dtype=np.float64),
        "full_moment_residual": np.zeros((horizon, 2, 2), dtype=np.float64),
        "native_action_idempotence_error": np.empty(horizon, dtype=np.float64),
        "native_basis_orthogonality_error": np.empty(horizon, dtype=np.float64),
        "native_useful_squared_alignment": np.empty(horizon, dtype=np.float64),
        "full_useful_squared_alignment": np.zeros(horizon, dtype=np.float64),
    }
    # Preserve both the shared draw whose digest is bound by the runner and the
    # rotated disturbance actually added to the signal in this cell.
    arrays["noise"][:] = _to_numpy(noise)
    arrays["epsilon"][:] = _to_numpy(rotated_noise)
    arrays["g"][:] = _to_numpy(gradients)
    arrays["s"][:] = _to_numpy(signal)

    ema_states: dict[float, torch.Tensor] = {}
    scalar_states: dict[float, torch.Tensor] = {}
    full = torch.zeros((2, 2), dtype=torch.float64)
    full_initialized = False
    previous_i17 = previous_cp = None
    eye = torch.eye(2, dtype=torch.float64)

    for index in range(horizon):
        if index % 50 == 0:
            check()
        gradient = gradients[index].clone()
        model.vector.grad = gradient.clone()
        tracker.filter_grad()
        mu = tracker.grad_mean.detach().clone()
        if tracker.V is None:
            action = eye.clone()
            basis = None
        else:
            basis = tracker.V[:, 0].detach().clone()
            action = torch.outer(basis, basis)
        native = model.vector.grad.detach().clone()
        action_raw = action @ gradient
        h = native + (mu - action @ mu)

        residual = gradient - mu
        outer = torch.outer(residual, residual)
        expected_full = (outer if not full_initialized and bool(torch.count_nonzero(residual))
                         else BETA * full + (1.0 - BETA) * outer if full_initialized
                         else full)
        full = expected_full.clone()
        if not full_initialized and bool(torch.count_nonzero(residual)):
            full_initialized = True
        eigenvalues, eigenvectors = torch.linalg.eigh((full + full.T) * 0.5)
        gap = eigenvalues[-1] - eigenvalues[-2]
        full_available = bool(gap > FULL_GAP_RELATIVE_TOLERANCE *
                              max(1.0, abs(float(eigenvalues[-1]))))
        full_vector = eigenvectors[:, -1] if full_available else None

        outputs = {}
        if index == 0:
            for decay in EMA_DECAYS:
                ema_states[decay] = gradient.clone()
            if optimum is not None:
                ema_states[optimum] = gradient.clone()
            for k in SCALAR_K:
                scalar_states[k] = gradient / (1.0 - RHO * k)
            previous_i17 = torch.linalg.solve(eye - RHO * action, gradient)
            previous_cp = gradient.clone()
        else:
            for decay in tuple(ema_states):
                ema_states[decay] = decay * ema_states[decay] + (1.0 - decay) * gradient
            for k in SCALAR_K:
                prior = scalar_states[k]
                expected = RHO * k * prior + k * gradient + (1.0 - k) * mu
                scalar_states[k] = expected
            prior_i17 = previous_i17
            previous_i17 = RHO * (action @ prior_i17) + h
            prior_cp = previous_cp
            previous_cp = RHO * (action @ prior_cp) + (eye - RHO * action) @ h

        outputs["raw"] = gradient
        outputs["ema_q0p9"] = ema_states[0.9]
        outputs["ema_q0p99"] = ema_states[0.99]
        outputs["ema_q0p999"] = ema_states[0.999]
        if optimum is not None:
            outputs["ema_optimal_oracle"] = ema_states[optimum]
        for label, k in zip(("scalar_k0", "scalar_k0p5", "scalar_k0p9", "scalar_k1"), SCALAR_K):
            outputs[label] = (1.0 - RHO * k) * scalar_states[k]
        outputs["oracle_useful"] = useful @ ema_states[0.9] + nuisance @ ema_states[0.99]
        outputs["oracle_nuisance"] = nuisance @ ema_states[0.9] + useful @ ema_states[0.99]
        outputs["native_i17"] = (eye - RHO * action) @ previous_i17
        outputs["native_cp"] = previous_cp
        if index == 0:
            # The registered initialization defines every delivered response
            # literally as g_1; the unnormalized internal states remain as above.
            for name in outputs:
                outputs[name] = gradient

        arrays["mu"][index] = _to_numpy(mu)
        arrays["A"][index] = _to_numpy(action)
        arrays["native_delivery"][index] = _to_numpy(native)
        arrays["native_h"][index] = _to_numpy(h)
        arrays["full_moment"][index] = _to_numpy(full)
        arrays["full_gap"][index] = float(gap)
        arrays["output"][index] = np.stack([_to_numpy(outputs[name]) for name in names])
        arrays["native_i17_buffer"][index] = _to_numpy(previous_i17)
        arrays["scalar_buffer"][index] = np.stack([_to_numpy(scalar_states[k]) for k in SCALAR_K])
        arrays["native_action_raw_error"][index] = _to_numpy(native - action_raw)
        arrays["native_action_idempotence_error"][index] = float(torch.linalg.matrix_norm(action @ action - action))
        arrays["native_basis_orthogonality_error"][index] = (0.0 if basis is None else
            abs(float(torch.dot(basis, basis)) - 1.0))
        arrays["native_useful_squared_alignment"][index] = (0.0 if basis is None else
            float(torch.dot(basis, q[:, 0]).square()))
        if basis is not None:
            arrays["basis_present"][index] = True
            arrays["basis_V"][index] = _to_numpy(basis)
            arrays["basis_S"][index] = float(tracker.S[0])
        if full_vector is not None:
            arrays["full_basis_present"][index] = True
            arrays["full_basis_V"][index] = _to_numpy(full_vector)
            arrays["full_action"][index] = _to_numpy(torch.outer(full_vector, full_vector))
            arrays["full_useful_squared_alignment"][index] = float(torch.dot(full_vector, q[:, 0]).square())

        if index > 0:
            expected_i17 = RHO * (action @ prior_i17) + h
            expected_cp = RHO * (action @ prior_cp) + (eye - RHO * action) @ h
            arrays["native_i17_response_residual"][index] = _to_numpy(previous_i17 - expected_i17)
            arrays["native_cp_response_residual"][index] = _to_numpy(previous_cp - expected_cp)
            for position, k in enumerate(SCALAR_K):
                # Reconstruct the previous unnormalized state from its saved row.
                prior = torch.from_numpy(arrays["scalar_buffer"][index - 1, position].copy())
                expected = RHO * k * prior + k * gradient + (1.0 - k) * mu
                arrays["scalar_response_residual"][index, position] = _to_numpy(scalar_states[k] - expected)
        arrays["full_moment_residual"][index] = _to_numpy(full - expected_full)

    check()
    for name, value in arrays.items():
        if value.dtype == np.dtype("O") or (np.issubdtype(value.dtype, np.number)
                                             and not np.isfinite(value).all()):
            raise FloatingPointError("saved array is invalid: " + name)

    metadata = {
        "schema": "i18_tracking_stream_metadata_v1",
        "array_schema": "i18_tracking_arrays_v1",
        "horizon": horizon, "dimension": DIMENSION, "drift": drift,
        "rotation_radians": rotation, "rho": RHO, "beta": BETA,
        "policy_names": list(names), "policy_count": p,
        "optimal_ema_decay": optimum,
        "full_gap_relative_tolerance": FULL_GAP_RELATIVE_TOLERANCE,
        "filter_configuration": dict(FILTER_CONFIGURATION),
        "initialization": {
            "all_policy_outputs_at_t1_equal_g1": True,
            "native_i17_buffer": "solve(I-rho*A_1,g_1)",
            "native_cp_delivery": "g_1",
            "scalar_buffer": "g_1/(1-rho*k)",
            "uniform_ema": "g_1",
            "full_moment": "zero until first nonzero post-ingest residual, then z*z^T unscaled",
        },
        "native_absent_basis_action": "identity",
        "full_absent_basis_arrays": "zero-filled with full_basis_present=false",
        "array_order": list(arrays),
        "array_shapes": {name: list(value.shape) for name, value in arrays.items()},
        "array_dtypes": {name: str(value.dtype) for name, value in arrays.items()},
        "noise_generated_by_core": False,
        "noise_array_semantics": "unrotated shared input draw",
        "epsilon_array_semantics": "rotation-applied disturbance used in g=s+epsilon",
    }
    return arrays, metadata


__all__ = ["TrackingCoreError", "HORIZON", "DIMENSION", "DRIFTS", "ROTATIONS",
           "RHO", "BETA", "EMA_DECAYS", "SCALAR_K", "FILTER_CONFIGURATION",
           "FULL_GAP_RELATIVE_TOLERANCE", "optimal_ema_decay", "policy_names",
           "run_stream"]
