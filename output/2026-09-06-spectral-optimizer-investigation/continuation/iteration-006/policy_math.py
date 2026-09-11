"""Six prospective raw-gradient policies; no dataset, training or GPU launch.

The caller must assign the returned gradient to every model parameter and call
delivery_metrics on the actual flattened .grad, even with instrumentation off.
This module does not call AdamW or change model parameters. Missing bases have
the canonical identity action, never the empty-basis zero action.
"""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
HELPER_PATH = HERE.parent / "iteration-003" / "neural_harness.py"
_spec = importlib.util.spec_from_file_location("iteration006_policy_helpers003", HELPER_PATH)
h = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(h)
if Path(h.__file__).resolve() != HELPER_PATH.resolve():
    raise RuntimeError("Unexpected immutable helper import")
torch = h.torch

ARMS = ("adamw", "current32", "lagged32", "lagged32_current_norm",
        "scalar_current32", "scalar_lagged32")
WARMUP, WIDTH = 100, 32
NORM_TOL, DIRECTION_TOL, SCALAR_MAX = 1e-6, 1e-6, 1.005
METRIC_KEYS = (
    "raw_norm", "raw_squared_norm", "current_norm", "current_squared_norm",
    "lagged_norm", "lagged_squared_norm", "applied_norm", "applied_squared_norm",
    "current_raw_norm_ratio", "lagged_raw_norm_ratio", "applied_raw_norm_ratio",
    "current_energy_retention", "lagged_energy_retention", "current_minus_lagged_retention",
    "current_lagged_cosine", "raw_applied_cosine", "policy_scale",
    "norm_matching_relative_error", "direction_relative_error",
)


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _vector(value, name, reference=None):
    _require(isinstance(value, torch.Tensor), f"{name}: expected tensor")
    _require(value.layout == torch.strided and value.ndim == 1 and value.numel() > 0,
             f"{name}: expected nonempty dense flat vector")
    _require(value.dtype == torch.float32, f"{name}: frozen native dtype must be float32")
    if reference is not None:
        _require(value.shape == reference.shape and value.device == reference.device,
                 f"{name}: shape/device mismatch")
    _require(bool(torch.isfinite(value).all()), f"{name}: nonfinite input")


def _basis(value, raw, name):
    if value is None:
        return 0
    _require(isinstance(value, torch.Tensor) and value.ndim == 2,
             f"{name}: expected matrix or None")
    _require(value.shape[0] == raw.numel() and 0 < value.shape[1] <= WIDTH,
             f"{name}: invalid stored width")
    _require(value.dtype == raw.dtype and value.device == raw.device,
             f"{name}: native dtype/device mismatch")
    _require(bool(torch.isfinite(value).all()), f"{name}: nonfinite input")
    # This is the stored column count, not a separate numerical-rank estimate.
    return int(value.shape[1])


def _policy_spec(arm, active, raw, current, lagged):
    """Return direction, target norm, scale, direction name and operator label."""
    _require(arm in ARMS, "Unknown policy arm")
    _vector(raw, "raw")
    if current is not None:
        _vector(current, "current", raw)
    if lagged is not None:
        _vector(lagged, "lagged", raw)
    if not active:
        _require(current is None and lagged is None, "Inactive candidates must be null")
        return raw, h.norm(raw), None, "raw", "identity"
    _require(arm != "adamw" and current is not None and lagged is not None,
             "Active observing policy requires both candidates")
    if arm == "current32":
        return current, h.norm(current), None, "current", "native_current"
    if arm == "lagged32":
        return lagged, h.norm(lagged), None, "lagged", "native_lagged"
    if arm == "lagged32_current_norm":
        direction, target, name, operator = lagged, h.norm(current), "lagged", "scaled_native_lagged"
    else:
        candidate = current if arm == "scalar_current32" else lagged
        direction, target, name, operator = raw, h.norm(candidate), "raw", "scalar_identity"
    direction_norm = h.norm(direction)
    _require(math.isfinite(target) and math.isfinite(direction_norm), "Nonfinite norm")
    _require(target == 0 or direction_norm > 0, "Positive target with zero direction")
    scale = 0.0 if target == 0 else target / direction_norm
    _require(math.isfinite(scale) and scale >= 0, "Nonfinite or negative rescaling")
    if operator == "scalar_identity":
        _require(scale <= SCALAR_MAX, "Scalar approximate-contraction bound exceeded")
    return direction, target, scale, name, operator


def _delivery_errors(direction, actual, target):
    """Strict norm and unit-direction gates: no absolute underflow escape."""
    _vector(actual, "actual_delivered", direction)
    actual_norm = h.norm(actual)
    if target == 0:
        _require(actual_norm == 0, "Zero target must deliver exact zero")
        return 0.0, None
    _require(actual_norm > 0, "Positive target lost through complete underflow")
    norm_error = abs(actual_norm - target) / target
    _require(norm_error <= NORM_TOL, f"Delivered relative norm error {norm_error}")
    direction_norm = h.norm(direction)
    _require(direction_norm > 0, "Positive target with zero direction")
    error_vector = actual.double() / actual_norm - direction.double() / direction_norm
    direction_error = float(torch.linalg.vector_norm(error_vector).item())
    _require(math.isfinite(direction_error) and direction_error <= DIRECTION_TOL,
             f"Delivered normalized direction error {direction_error}")
    return norm_error, direction_error


@torch.no_grad()
def transform_gradient(arm, tracker, raw):
    """Observe raw once and return (applied, current, lagged, scalar metadata).

    Baseline and steps 1..100 have null candidates. Observing arms nevertheless
    update their canonical tracker during warmup. Stored previous V is cloned
    before ingestion, independently of the helper's warmup-null return value.
    """
    _require(arm in ARMS, "Unknown policy arm")
    _vector(raw, "raw")
    current = lagged = None
    step = current_rank = lagged_rank = None
    current_missing = lagged_missing = None
    current_operator = lagged_operator = None
    active = False
    if arm == "adamw":
        _require(tracker is None, "AdamW baseline must not have an observer")
    else:
        _require(tracker is not None, "Observing policy requires tracker")
        _require(tracker.rank == WIDTH and tracker.warmup == WARMUP,
                 "Observer width/warmup differ from frozen policy")
        old_step = tracker.step_count
        _require(type(old_step) is int and old_step >= 0, "Invalid observer step")
        lagged_rank = _basis(tracker.V, raw, "previous basis")
        previous = None if tracker.V is None else tracker.V.detach().clone()
        observed, _ = h.observe_and_project(tracker, raw)
        step = tracker.step_count
        _require(step == old_step + 1, "Observer must ingest exactly once")
        current_rank = _basis(tracker.V, raw, "current basis")
        current_missing, lagged_missing = tracker.V is None, previous is None
        active = step > WARMUP
        if active:
            current = observed
            lagged = h.project(raw, previous)
            current_operator = "identity_missing_basis" if current_missing else "native_basis"
            lagged_operator = "identity_missing_basis" if lagged_missing else "native_basis"
    direction, target, scale, direction_name, operator = _policy_spec(
        arm, active, raw, current, lagged)
    applied = (direction.detach().clone() if scale is None
               else (direction.double() * scale).to(dtype=raw.dtype))
    _delivery_errors(direction, applied, target)
    metadata = {
        "arm": arm, "observing_step": step, "policy_active": active,
        "active_policy": arm if active else ("identity_baseline" if arm == "adamw" else "identity_warmup"),
        "current_basis_rank": current_rank, "lagged_basis_rank": lagged_rank,
        "current_basis_missing": current_missing, "lagged_basis_missing": lagged_missing,
        "current_candidate_operator": current_operator, "lagged_candidate_operator": lagged_operator,
        "scale": scale, "scale_direction": direction_name, "target_norm": target,
        "delivery_operator": operator,
    }
    return applied, current, lagged, metadata


@torch.no_grad()
def delivery_metrics(raw, current, lagged, actual_delivered, policy_metadata):
    """Validate actual .grad and return flat finite-or-null metrics plus reasons.

    Norm-matching errors apply to the prescribed target for every policy, not
    just scalar controls. Direction error is distance between normalized vectors,
    not projector leakage. Candidate nulls mean baseline/warmup; cosine and ratio
    nulls additionally distinguish zero norms. No clipping of reported cosines.
    """
    arm, active = policy_metadata["arm"], policy_metadata["policy_active"]
    _require(type(active) is bool, "Invalid policy_active metadata")
    step = policy_metadata["observing_step"]
    _require((arm == "adamw" and step is None and not active)
             or (arm != "adamw" and type(step) is int and step > 0 and active == (step > WARMUP)),
             "Inconsistent observation/activation metadata")
    direction, target, scale, name, operator = _policy_spec(arm, active, raw, current, lagged)
    for key, expected in (("scale", scale), ("scale_direction", name),
                          ("target_norm", target), ("delivery_operator", operator)):
        _require(policy_metadata[key] == expected, f"Policy metadata mismatch: {key}")
    norm_error, direction_error = _delivery_errors(direction, actual_delivered, target)
    values, reasons = {}, {}
    unavailable = "baseline_no_observer" if arm == "adamw" else "warmup_candidates_not_measured"

    def record(key, value, reason=None):
        _require(key in METRIC_KEYS, f"Unexpected metric {key}")
        if value is None:
            _require(isinstance(reason, str) and bool(reason), "Null metric requires reason")
            reasons[key] = reason
        else:
            _require(math.isfinite(value), f"Nonfinite output metric {key}")
        values[key] = value

    norms, energies = {}, {}
    for name, vector in (("raw", raw), ("current", current), ("lagged", lagged),
                         ("applied", actual_delivered)):
        energy = None if vector is None else h.dot(vector, vector)
        norm = None if energy is None else math.sqrt(energy)
        energies[name], norms[name] = energy, norm
        record(name + "_squared_norm", energy, unavailable)
        record(name + "_norm", norm, unavailable)
    for name in ("current", "lagged", "applied"):
        reason = unavailable if norms[name] is None else "zero_raw_norm"
        value = None if norms[name] is None or norms["raw"] == 0 else norms[name] / norms["raw"]
        record(name + "_raw_norm_ratio", value, reason)
    for name in ("current", "lagged"):
        reason = unavailable if energies[name] is None else "zero_raw_norm"
        value = None if energies[name] is None or energies["raw"] == 0 else energies[name] / energies["raw"]
        record(name + "_energy_retention", value, reason)
    a, b = values["current_energy_retention"], values["lagged_energy_retention"]
    record("current_minus_lagged_retention", None if a is None or b is None else a - b,
           unavailable if current is None else "zero_raw_norm")
    for key, a_name, b_name, a_vec, b_vec in (
        ("current_lagged_cosine", "current", "lagged", current, lagged),
        ("raw_applied_cosine", "raw", "applied", raw, actual_delivered),
    ):
        missing = a_vec is None or b_vec is None
        denominator = 0 if missing else norms[a_name] * norms[b_name]
        record(key, None if denominator == 0 else h.dot(a_vec, b_vec) / denominator,
               unavailable if missing else "zero_vector_norm")
    record("policy_scale", scale, "unscaled_policy")
    record("norm_matching_relative_error", norm_error)
    record("direction_relative_error", direction_error, "zero_target_and_delivery")
    _require(set(values) == set(METRIC_KEYS), "Incomplete policy metric schema")
    values["null_reasons"] = reasons
    return values
