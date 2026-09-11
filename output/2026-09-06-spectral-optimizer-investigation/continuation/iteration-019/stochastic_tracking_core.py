#!/usr/bin/env python3
"""Pure CPU stream mechanism for the prospective I19 tracking study.

Importing this module does not draw randomness, read artifacts, or run a stream.
The caller supplies the complete canonical standard-normal tensor.
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
    "_i19_canonical_spectral_filter", REPO / "spectral_filter.py")
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover
    raise ImportError("cannot load canonical spectral_filter.py")
_native = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_native)
SpectralGradientFilter = _native.SpectralGradientFilter

HORIZON = 4000
DIMENSION = 2
CANONICAL_DIMENSION = 3
PROCESS_VARIANCES = (0.0, 0.01, 0.1)
ROTATIONS = (0.0, math.pi / 4.0)
RHO = 0.9
BETA = 0.99
RHO_STAR = (2.0 + 0.1 - math.sqrt(0.1 ** 2 + 4.0 * 0.1)) / 2.0
EMA_DECAYS = (0.9, 0.99, 0.999)
SCALAR_K = (0.0, 0.5, 0.9, 1.0)
FULL_GAP_RELATIVE_TOLERANCE = 1e-10
FILTER_CONFIGURATION = {
    "rank": 1, "decay": BETA, "warmup": 0, "filter_strength": 1.0,
    "adaptive": "none", "normalize": "none", "weighting": "hard",
    "stable_update": True, "relative_eig_tol": 1e-8,
    "absolute_eig_floor": 0.0, "stabilize_every": 100,
}


class StochasticTrackingCoreError(ValueError):
    """Raised when a stream or registered configuration is malformed."""


class _DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.vector = torch.nn.Parameter(torch.zeros(DIMENSION, dtype=torch.float64))


def optimal_ema_decay(process_variance: float) -> float | None:
    """Return the registered common steady-state EMA for nonzero Q1."""
    if type(process_variance) is not float or process_variance not in PROCESS_VARIANCES:
        raise StochasticTrackingCoreError(
            "process_variance must be one registered float value")
    if process_variance == 0.0:
        return None
    error = (math.sqrt(process_variance ** 2 + 20.0 * process_variance)
             - process_variance) / 2.0
    return 1.0 - error / 5.0


def policy_names(process_variance: float) -> tuple[str, ...]:
    """Return the frozen output-column order for one process-variance cell."""
    optimum = optimal_ema_decay(process_variance)
    names = ["raw", "ema_q0p9", "ema_q0p99", "ema_q0p999",
             "dema_q0p9", "dema_q0p99", "dema_q0p999"]
    if optimum is not None:
        names.append("ema_common_steady_oracle")
    names.extend(("common_kalman", "useful_oracle_kalman"))
    names.extend(("scalar_k0", "scalar_k0p5", "scalar_k0p9", "scalar_k1",
                  "oracle_useful", "oracle_nuisance", "native_i17", "native_cp",
                  "native_cp_star", "oracle_useful_star"))
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


def _validate_canonical(canonical: torch.Tensor) -> int:
    if type(canonical) is not torch.Tensor or canonical.device.type != "cpu" \
            or canonical.dtype != torch.float64 or canonical.layout != torch.strided \
            or canonical.ndim != 2 or canonical.shape[1] != CANONICAL_DIMENSION \
            or canonical.shape[0] < 1:
        raise StochasticTrackingCoreError(
            "canonical must be a nonempty dense CPU float64 tensor [H,3]")
    if not bool(torch.isfinite(canonical).all()):
        raise StochasticTrackingCoreError("canonical must be finite")
    return int(canonical.shape[0])


def run_stream(canonical: torch.Tensor, process_variance: float, rotation: float, *,
               check_budget: Callable[[], None] | None = None) \
        -> tuple[dict[str, np.ndarray], dict]:
    """Route one supplied exogenous random-walk stream through all policies."""
    horizon = _validate_canonical(canonical)
    if type(rotation) is not float or rotation not in ROTATIONS:
        raise StochasticTrackingCoreError("rotation must be one registered float value")
    names = policy_names(process_variance)
    optimum = optimal_ema_decay(process_variance)
    if check_budget is not None and not callable(check_budget):
        raise StochasticTrackingCoreError("check_budget must be callable or None")
    check = check_budget if check_budget is not None else (lambda: None)

    q = _rotation(rotation)
    useful = q[:, :1] @ q[:, :1].T
    nuisance = torch.eye(DIMENSION, dtype=torch.float64) - useful
    measurement_noise = canonical[:, :2] * torch.tensor((1.0, 2.0), dtype=torch.float64)
    signal_base = torch.zeros((horizon, DIMENSION), dtype=torch.float64)
    if horizon > 1 and process_variance > 0.0:
        # canonical[0,2] is deliberately retained but unused by the process.
        signal_base[1:, 0] = (math.sqrt(process_variance)
                              * torch.cumsum(canonical[1:, 2], dim=0))
    signal = signal_base @ q.T
    epsilon = measurement_noise @ q.T
    gradients = _finite_tensor(signal + epsilon, "gradient stream")

    model = _DummyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    tracker = SpectralGradientFilter(model, optimizer, **FILTER_CONFIGURATION)

    policy_count = len(names)
    arrays = {
        "canonical": np.empty((horizon, 3), dtype=np.float64),
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
        "output": np.empty((horizon, policy_count, 2), dtype=np.float64),
        "ema_fixed_state": np.empty((horizon, len(EMA_DECAYS), 2), dtype=np.float64),
        "dema_first_state": np.empty((horizon, len(EMA_DECAYS), 2), dtype=np.float64),
        "dema_second_state": np.empty((horizon, len(EMA_DECAYS), 2), dtype=np.float64),
        "ema_common_state": np.zeros((horizon, 2), dtype=np.float64),
        "common_kalman_state": np.empty((horizon, 2), dtype=np.float64),
        "common_kalman_prior_weight": np.zeros(horizon, dtype=np.float64),
        "common_kalman_posterior_variance": np.empty(horizon, dtype=np.float64),
        "useful_oracle_kalman_state_base": np.empty((horizon, 2), dtype=np.float64),
        "useful_oracle_kalman_prior_weight": np.zeros((horizon, 2), dtype=np.float64),
        "useful_oracle_kalman_posterior_variance": np.empty((horizon, 2), dtype=np.float64),
        "oracle_useful_fast_state": np.empty((horizon, 2), dtype=np.float64),
        "oracle_useful_slow_state": np.empty((horizon, 2), dtype=np.float64),
        "oracle_useful_star_fast_state": np.empty((horizon, 2), dtype=np.float64),
        "scalar_buffer": np.empty((horizon, len(SCALAR_K), 2), dtype=np.float64),
        "native_i17_buffer": np.empty((horizon, 2), dtype=np.float64),
        "native_cp_buffer": np.empty((horizon, 2), dtype=np.float64),
        "native_cp_star_buffer": np.empty((horizon, 2), dtype=np.float64),
        "native_action_raw_error": np.empty((horizon, 2), dtype=np.float64),
        "ema_fixed_response_residual": np.zeros((horizon, len(EMA_DECAYS), 2), dtype=np.float64),
        "dema_first_response_residual": np.zeros((horizon, len(EMA_DECAYS), 2), dtype=np.float64),
        "dema_second_response_residual": np.zeros((horizon, len(EMA_DECAYS), 2), dtype=np.float64),
        "ema_common_response_residual": np.zeros((horizon, 2), dtype=np.float64),
        "common_kalman_response_residual": np.zeros((horizon, 2), dtype=np.float64),
        "common_kalman_variance_residual": np.zeros(horizon, dtype=np.float64),
        "useful_oracle_kalman_response_residual": np.zeros((horizon, 2), dtype=np.float64),
        "useful_oracle_kalman_variance_residual": np.zeros((horizon, 2), dtype=np.float64),
        "oracle_useful_fast_response_residual": np.zeros((horizon, 2), dtype=np.float64),
        "oracle_useful_slow_response_residual": np.zeros((horizon, 2), dtype=np.float64),
        "oracle_useful_star_fast_response_residual": np.zeros((horizon, 2), dtype=np.float64),
        "scalar_response_residual": np.zeros((horizon, len(SCALAR_K), 2), dtype=np.float64),
        "native_i17_response_residual": np.zeros((horizon, 2), dtype=np.float64),
        "native_cp_response_residual": np.zeros((horizon, 2), dtype=np.float64),
        "native_cp_star_response_residual": np.zeros((horizon, 2), dtype=np.float64),
        "full_moment_residual": np.zeros((horizon, 2, 2), dtype=np.float64),
        "native_action_idempotence_error": np.empty(horizon, dtype=np.float64),
        "native_basis_orthogonality_error": np.empty(horizon, dtype=np.float64),
        "native_useful_squared_alignment": np.empty(horizon, dtype=np.float64),
        "full_useful_squared_alignment": np.zeros(horizon, dtype=np.float64),
    }
    arrays["canonical"][:] = _to_numpy(canonical)
    arrays["noise"][:] = _to_numpy(measurement_noise)
    arrays["epsilon"][:] = _to_numpy(epsilon)
    arrays["g"][:] = _to_numpy(gradients)
    arrays["s"][:] = _to_numpy(signal)

    ema_states: dict[float, torch.Tensor] = {}
    dema_first: dict[float, torch.Tensor] = {}
    dema_second: dict[float, torch.Tensor] = {}
    scalar_states: dict[float, torch.Tensor] = {}
    common_state: torch.Tensor | None = None
    common_kalman_state = None
    common_kalman_variance = 5.0
    oracle_kalman_state_base = None
    oracle_kalman_variance = torch.tensor((1.0, 4.0), dtype=torch.float64)
    useful_fast = useful_slow = useful_star_fast = None
    previous_i17 = previous_cp = previous_cp_star = None
    full = torch.zeros((2, 2), dtype=torch.float64)
    full_initialized = False
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
        if not full_initialized and bool(torch.count_nonzero(residual)):
            expected_full = outer
            full_initialized = True
        elif full_initialized:
            expected_full = BETA * full + (1.0 - BETA) * outer
        else:
            expected_full = full
        full = expected_full.clone()
        eigenvalues, eigenvectors = torch.linalg.eigh((full + full.T) * 0.5)
        gap = eigenvalues[-1] - eigenvalues[-2]
        full_available = bool(gap > FULL_GAP_RELATIVE_TOLERANCE
                              * max(1.0, abs(float(eigenvalues[-1]))))
        full_vector = eigenvectors[:, -1] if full_available else None

        if index == 0:
            for decay in EMA_DECAYS:
                ema_states[decay] = gradient.clone()
                dema_first[decay] = gradient.clone()
                dema_second[decay] = gradient.clone()
            if optimum is not None:
                common_state = gradient.clone()
            common_kalman_state = gradient.clone()
            oracle_kalman_state_base = q.T @ gradient
            useful_fast = useful_slow = useful_star_fast = gradient.clone()
            for k in SCALAR_K:
                scalar_states[k] = gradient / (1.0 - RHO * k)
            previous_i17 = torch.linalg.solve(eye - RHO * action, gradient)
            previous_cp = gradient.clone()
            previous_cp_star = gradient.clone()
        else:
            prior_ema = {decay: state.clone() for decay, state in ema_states.items()}
            prior_dema_first = {decay: state.clone() for decay, state in dema_first.items()}
            prior_dema_second = {decay: state.clone() for decay, state in dema_second.items()}
            for decay in EMA_DECAYS:
                ema_states[decay] = decay * ema_states[decay] + (1.0 - decay) * gradient
                dema_first[decay] = decay * dema_first[decay] + (1.0 - decay) * gradient
                dema_second[decay] = (decay * dema_second[decay]
                                      + (1.0 - decay) * dema_first[decay])
            if optimum is not None:
                prior_common = common_state.clone()
                common_state = optimum * common_state + (1.0 - optimum) * gradient
            prior_common_kalman = common_kalman_state.clone()
            prior_common_variance = common_kalman_variance
            common_weight = 5.0 / (prior_common_variance + process_variance + 5.0)
            common_kalman_state = (common_weight * common_kalman_state
                                   + (1.0 - common_weight) * gradient)
            common_kalman_variance = (5.0 * (prior_common_variance + process_variance)
                                      / (prior_common_variance + process_variance + 5.0))
            prior_oracle_kalman = oracle_kalman_state_base.clone()
            prior_oracle_variance = oracle_kalman_variance.clone()
            process_diagonal = torch.tensor((process_variance, 0.0), dtype=torch.float64)
            measurement_diagonal = torch.tensor((1.0, 4.0), dtype=torch.float64)
            oracle_weights = (measurement_diagonal
                              / (prior_oracle_variance + process_diagonal
                                 + measurement_diagonal))
            gradient_base = q.T @ gradient
            oracle_kalman_state_base = (oracle_weights * oracle_kalman_state_base
                                        + (1.0 - oracle_weights) * gradient_base)
            oracle_kalman_variance = (measurement_diagonal
                * (prior_oracle_variance + process_diagonal)
                / (prior_oracle_variance + process_diagonal + measurement_diagonal))
            prior_useful_fast = useful_fast.clone()
            prior_useful_slow = useful_slow.clone()
            prior_useful_star_fast = useful_star_fast.clone()
            useful_fast = RHO * useful_fast + (1.0 - RHO) * gradient
            useful_slow = BETA * useful_slow + (1.0 - BETA) * gradient
            useful_star_fast = (RHO_STAR * useful_star_fast
                                + (1.0 - RHO_STAR) * gradient)
            prior_scalar = {k: state.clone() for k, state in scalar_states.items()}
            for k in SCALAR_K:
                scalar_states[k] = (RHO * k * scalar_states[k] + k * gradient
                                    + (1.0 - k) * mu)
            prior_i17 = previous_i17
            previous_i17 = RHO * (action @ previous_i17) + h
            prior_cp = previous_cp
            previous_cp = RHO * (action @ previous_cp) + (eye - RHO * action) @ h
            prior_cp_star = previous_cp_star
            previous_cp_star = (RHO_STAR * (action @ previous_cp_star)
                                + (eye - RHO_STAR * action) @ h)

        outputs = {
            "raw": gradient,
            "ema_q0p9": ema_states[0.9],
            "ema_q0p99": ema_states[0.99],
            "ema_q0p999": ema_states[0.999],
            "dema_q0p9": 2.0 * dema_first[0.9] - dema_second[0.9],
            "dema_q0p99": 2.0 * dema_first[0.99] - dema_second[0.99],
            "dema_q0p999": 2.0 * dema_first[0.999] - dema_second[0.999],
        }
        if optimum is not None:
            outputs["ema_common_steady_oracle"] = common_state
        outputs["common_kalman"] = common_kalman_state
        outputs["useful_oracle_kalman"] = q @ oracle_kalman_state_base
        for label, k in zip(("scalar_k0", "scalar_k0p5", "scalar_k0p9", "scalar_k1"),
                            SCALAR_K):
            outputs[label] = (1.0 - RHO * k) * scalar_states[k]
        outputs.update({
            "oracle_useful": useful @ useful_fast + nuisance @ useful_slow,
            "oracle_nuisance": nuisance @ useful_fast + useful @ useful_slow,
            "native_i17": (eye - RHO * action) @ previous_i17,
            "native_cp": previous_cp,
            "native_cp_star": previous_cp_star,
            "oracle_useful_star": useful @ useful_star_fast + nuisance @ useful_slow,
        })
        if index == 0:
            for name in outputs:
                outputs[name] = gradient

        arrays["mu"][index] = _to_numpy(mu)
        arrays["A"][index] = _to_numpy(action)
        arrays["native_delivery"][index] = _to_numpy(native)
        arrays["native_h"][index] = _to_numpy(h)
        arrays["full_moment"][index] = _to_numpy(full)
        arrays["full_gap"][index] = float(gap)
        arrays["output"][index] = np.stack([_to_numpy(outputs[name]) for name in names])
        arrays["ema_fixed_state"][index] = np.stack([_to_numpy(ema_states[d]) for d in EMA_DECAYS])
        arrays["dema_first_state"][index] = np.stack([_to_numpy(dema_first[d]) for d in EMA_DECAYS])
        arrays["dema_second_state"][index] = np.stack([_to_numpy(dema_second[d]) for d in EMA_DECAYS])
        if common_state is not None:
            arrays["ema_common_state"][index] = _to_numpy(common_state)
        arrays["common_kalman_state"][index] = _to_numpy(common_kalman_state)
        arrays["common_kalman_posterior_variance"][index] = common_kalman_variance
        arrays["useful_oracle_kalman_state_base"][index] = _to_numpy(
            oracle_kalman_state_base)
        arrays["useful_oracle_kalman_posterior_variance"][index] = _to_numpy(
            oracle_kalman_variance)
        if index > 0:
            arrays["common_kalman_prior_weight"][index] = common_weight
            arrays["useful_oracle_kalman_prior_weight"][index] = _to_numpy(
                oracle_weights)
        arrays["oracle_useful_fast_state"][index] = _to_numpy(useful_fast)
        arrays["oracle_useful_slow_state"][index] = _to_numpy(useful_slow)
        arrays["oracle_useful_star_fast_state"][index] = _to_numpy(useful_star_fast)
        arrays["scalar_buffer"][index] = np.stack([_to_numpy(scalar_states[k]) for k in SCALAR_K])
        arrays["native_i17_buffer"][index] = _to_numpy(previous_i17)
        arrays["native_cp_buffer"][index] = _to_numpy(previous_cp)
        arrays["native_cp_star_buffer"][index] = _to_numpy(previous_cp_star)
        arrays["native_action_raw_error"][index] = _to_numpy(native - action_raw)
        arrays["native_action_idempotence_error"][index] = float(
            torch.linalg.matrix_norm(action @ action - action))
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
            arrays["full_useful_squared_alignment"][index] = float(
                torch.dot(full_vector, q[:, 0]).square())

        if index > 0:
            for position, decay in enumerate(EMA_DECAYS):
                expected_ema = decay * prior_ema[decay] + (1.0 - decay) * gradient
                expected_first = (decay * prior_dema_first[decay]
                                  + (1.0 - decay) * gradient)
                expected_second = (decay * prior_dema_second[decay]
                                   + (1.0 - decay) * dema_first[decay])
                arrays["ema_fixed_response_residual"][index, position] = _to_numpy(
                    ema_states[decay] - expected_ema)
                arrays["dema_first_response_residual"][index, position] = _to_numpy(
                    dema_first[decay] - expected_first)
                arrays["dema_second_response_residual"][index, position] = _to_numpy(
                    dema_second[decay] - expected_second)
            if optimum is not None:
                expected_common = optimum * prior_common + (1.0 - optimum) * gradient
                arrays["ema_common_response_residual"][index] = _to_numpy(
                    common_state - expected_common)
            expected_common_kalman = (common_weight * prior_common_kalman
                                      + (1.0 - common_weight) * gradient)
            expected_common_variance = (5.0 * (prior_common_variance + process_variance)
                / (prior_common_variance + process_variance + 5.0))
            arrays["common_kalman_response_residual"][index] = _to_numpy(
                common_kalman_state - expected_common_kalman)
            arrays["common_kalman_variance_residual"][index] = (
                common_kalman_variance - expected_common_variance)
            expected_oracle_kalman = (oracle_weights * prior_oracle_kalman
                + (1.0 - oracle_weights) * (q.T @ gradient))
            expected_oracle_variance = (measurement_diagonal
                * (prior_oracle_variance + process_diagonal)
                / (prior_oracle_variance + process_diagonal + measurement_diagonal))
            arrays["useful_oracle_kalman_response_residual"][index] = _to_numpy(
                oracle_kalman_state_base - expected_oracle_kalman)
            arrays["useful_oracle_kalman_variance_residual"][index] = _to_numpy(
                oracle_kalman_variance - expected_oracle_variance)
            expected_fast = RHO * prior_useful_fast + (1.0 - RHO) * gradient
            expected_slow = BETA * prior_useful_slow + (1.0 - BETA) * gradient
            expected_star_fast = (RHO_STAR * prior_useful_star_fast
                                  + (1.0 - RHO_STAR) * gradient)
            arrays["oracle_useful_fast_response_residual"][index] = _to_numpy(
                useful_fast - expected_fast)
            arrays["oracle_useful_slow_response_residual"][index] = _to_numpy(
                useful_slow - expected_slow)
            arrays["oracle_useful_star_fast_response_residual"][index] = _to_numpy(
                useful_star_fast - expected_star_fast)
            for position, k in enumerate(SCALAR_K):
                expected_scalar = (RHO * k * prior_scalar[k] + k * gradient
                                   + (1.0 - k) * mu)
                arrays["scalar_response_residual"][index, position] = _to_numpy(
                    scalar_states[k] - expected_scalar)
            expected_i17 = RHO * (action @ prior_i17) + h
            expected_cp = RHO * (action @ prior_cp) + (eye - RHO * action) @ h
            expected_cp_star = (RHO_STAR * (action @ prior_cp_star)
                                + (eye - RHO_STAR * action) @ h)
            arrays["native_i17_response_residual"][index] = _to_numpy(
                previous_i17 - expected_i17)
            arrays["native_cp_response_residual"][index] = _to_numpy(
                previous_cp - expected_cp)
            arrays["native_cp_star_response_residual"][index] = _to_numpy(
                previous_cp_star - expected_cp_star)
        arrays["full_moment_residual"][index] = _to_numpy(full - expected_full)

    check()
    for name, value in arrays.items():
        if value.dtype == np.dtype("O") or (np.issubdtype(value.dtype, np.number)
                                             and not np.isfinite(value).all()):
            raise FloatingPointError("saved array is invalid: " + name)

    metadata = {
        "schema": "i19_stochastic_tracking_stream_metadata_v1",
        "array_schema": "i19_stochastic_tracking_arrays_v1",
        "horizon": horizon, "dimension": DIMENSION,
        "canonical_dimension": CANONICAL_DIMENSION,
        "process_variance": process_variance, "rotation_radians": rotation,
        "rho": RHO, "beta": BETA, "rho_star": RHO_STAR,
        "policy_names": list(names), "policy_count": policy_count,
        "optimal_common_ema_decay": optimum,
        "full_gap_relative_tolerance": FULL_GAP_RELATIVE_TOLERANCE,
        "filter_configuration": dict(FILTER_CONFIGURATION),
        "initialization": {
            "all_policy_outputs_at_t1_equal_g1": True,
            "signal_at_t1": "zero",
            "process_increments": "sqrt(Q1)*canonical[1:,2]; canonical[0,2] unused",
            "native_i17_buffer": "solve(I-rho*A_1,g_1)",
            "native_cp_delivery": "g_1",
            "native_cp_star_delivery": "g_1",
            "scalar_buffer": "g_1/(1-rho*k)",
            "uniform_ema_and_dema_states": "g_1",
            "common_kalman": "d_1=g_1,P_1=5; prior-weight q_1 sentinel is zero",
            "useful_oracle_kalman": "base-coordinate d_1=rotation_matrix^T*g_1,P_1=(1,4); prior-weight q_1 sentinel is zero",
            "full_moment": "zero until first nonzero post-ingest residual, then z*z^T unscaled",
        },
        "native_absent_basis_action": "identity",
        "full_absent_basis_arrays": "zero-filled with full_basis_present=false",
        "array_order": list(arrays),
        "array_shapes": {name: list(value.shape) for name, value in arrays.items()},
        "array_dtypes": {name: str(value.dtype) for name, value in arrays.items()},
        "canonical_generated_by_core": False,
        "canonical_array_semantics": "unmodified supplied [measurement0,measurement1,process] draw",
        "noise_array_semantics": "unrotated canonical[:,:2] scaled by measurement std (1,2)",
        "epsilon_array_semantics": "rotation-applied measurement disturbance used in g=s+epsilon",
        "common_ema_label_semantics": "theoretical steady state; not finite-horizon optimal or learned",
        "kalman_prior_weight_semantics": "q_t multiplies prior state; index zero is a sentinel with no update",
        "kalman_control_semantics": "matched-first-observation model-based controls, not unrestricted finite-horizon optima",
        "oracle_star_semantics": "fixed strong-cell rho_star used in every process-variance cell",
    }
    return arrays, metadata


__all__ = ["StochasticTrackingCoreError", "HORIZON", "DIMENSION",
           "CANONICAL_DIMENSION", "PROCESS_VARIANCES", "ROTATIONS", "RHO",
           "BETA", "RHO_STAR", "EMA_DECAYS", "SCALAR_K", "FILTER_CONFIGURATION",
           "FULL_GAP_RELATIVE_TOLERANCE", "optimal_ema_decay", "policy_names",
           "run_stream"]
