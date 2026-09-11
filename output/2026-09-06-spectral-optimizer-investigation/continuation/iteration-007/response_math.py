"""Inert I7 delivered-gradient arithmetic; no data, optimizer or launch code.

construct(raw, current, lagged) builds six owned native-float32 gradients and
leverage metadata. contrasts() accepts an exact six-branch scalar map (None
means an already validated domain-null cell). primary_summary() never uses an
available-case subset. These helpers do not validate full scientific artifacts.
"""
from __future__ import annotations

import itertools
import math

import torch

BRANCHES = ("raw", "current", "lagged", "restored", "reciprocal", "zero")
PRIMARY = (71001, 71002, 71003)
TOL = 1e-6
COEFFICIENTS = {
    "ordering": {"lagged": 1, "current": -1},
    "direction_at_current_norm": {"restored": 1, "current": -1},
    "direction_at_lagged_norm": {"lagged": 1, "reciprocal": -1},
    "norm_at_current_direction": {"current": 1, "reciprocal": -1},
    "norm_at_lagged_direction": {"restored": 1, "lagged": -1},
    "interaction": {"restored": 1, "lagged": -1, "current": -1, "reciprocal": 1},
    **{f"{name}_minus_raw": {name: 1, "raw": -1}
       for name in ("current", "lagged", "restored", "reciprocal")},
    **{f"{name}_minus_zero": {name: 1, "zero": -1}
       for name in BRANCHES[:-1]},
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def number(value):
    require(type(value) in (float, int), "Expected a real scalar, not bool or object")
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError("Scalar exceeds finite float range") from exc
    require(math.isfinite(result), "Nonfinite scalar")
    return result


def vector(value, reference=None, native=False):
    require(type(value) is torch.Tensor, "Expected a plain tensor")
    require(value.layout == torch.strided and value.ndim == 1 and value.numel() > 0,
            "Expected a nonempty dense flat vector")
    permitted = (torch.float32,) if native else (torch.float32, torch.float64)
    require(value.dtype in permitted and not value.requires_grad,
            "Wrong dtype or attached autograd graph")
    require(bool(torch.isfinite(value).all()), "Nonfinite vector")
    if reference is not None:
        require(value.shape == reference.shape and value.device == reference.device,
                "Vector shape/device mismatch")
    return value


def norm(value):
    vector(value)
    result = float(torch.linalg.vector_norm(value.double()).item())
    return number(result)


def pair_geometry(left, right):
    vector(left)
    vector(right, left)
    a, b = norm(left), norm(right)
    distance = norm(left.double() - right.double())
    if a == 0 or b == 0:
        return {"distance": distance, "cosine": None, "cosine_reason": "zero_norm"}
    cosine = number(float(torch.dot(left.double() / a, right.double() / b).item()))
    return {"distance": distance, "cosine": cosine, "cosine_reason": None}


def _delivery(direction, target, *, scaled=False):
    target = number(target)
    require(target >= 0, "Negative target norm")
    direction_norm = norm(direction)
    require(target == 0 or direction_norm > 0, "Undefined positive-target zero direction")
    if not scaled:
        applied = direction.detach().clone()
    elif target == 0:
        applied = torch.zeros_like(direction)
    else:
        applied = ((direction.double() / direction_norm) * target).to(torch.float32)
    vector(applied, direction, native=True)
    actual_norm = norm(applied)
    if target == 0:
        require(actual_norm == 0, "Zero target did not produce zeros")
        norm_error, direction_error = 0.0, None
    else:
        require(actual_norm > 0, "Positive target lost to native underflow")
        norm_error = abs(actual_norm - target) / target
        direction_error = norm(applied.double() / actual_norm - direction.double() / direction_norm)
        require(norm_error <= TOL and direction_error <= TOL,
                "Native delivery exceeded relative norm/unit-direction gate")
    return {"status": "defined", "reason": None, "gradient": applied,
            "target_norm": target, "actual_norm": actual_norm,
            "norm_error": norm_error, "direction_error": direction_error}


def construct(raw, current, lagged):
    """c/l must already come from the single shared observer transition."""
    vector(raw, native=True)
    vector(current, raw, native=True)
    vector(lagged, raw, native=True)
    nc, nl = norm(current), norm(lagged)
    rows = {
        "raw": _delivery(raw, norm(raw)),
        "current": _delivery(current, nc),
        "lagged": _delivery(lagged, nl),
    }
    for key, direction, target, source_norm, reason in (
        ("restored", lagged, nc, nl, "positive_current_norm_zero_lagged_direction"),
        ("reciprocal", current, nl, nc, "positive_lagged_norm_zero_current_direction"),
    ):
        if target > 0 and source_norm == 0:
            rows[key] = {"status": "undefined", "reason": reason, "gradient": None,
                         "target_norm": target, "actual_norm": None,
                         "norm_error": None, "direction_error": None}
        else:
            rows[key] = _delivery(direction, target, scaled=True)
    rows["zero"] = _delivery(raw, 0.0, scaled=True)
    separation = None if max(nc, nl) == 0 else abs(nc - nl) / max(nc, nl)
    direction_distance = (None if nc == 0 or nl == 0 else
                          norm(current.double() / nc - lagged.double() / nl))
    leverage = {
        "current_norm": nc, "lagged_norm": nl, "signed_norm_difference": nc - nl,
        "relative_norm_separation": separation,
        "current_lagged_norm_ratio": None if nl == 0 else nc / nl,
        "ratio_reason": "zero_lagged_norm" if nl == 0 else None,
        "unit_direction_distance": direction_distance,
        "unit_direction_cosine": pair_geometry(current, lagged)["cosine"],
        "direction_reason": "zero_norm" if direction_distance is None else None,
        "norm_leverage": separation is not None and separation > 2 * TOL,
        "direction_leverage": direction_distance is not None and direction_distance > 2 * TOL,
    }
    pairs = {}
    for left, right in itertools.combinations(BRANCHES, 2):
        a, b = rows[left]["gradient"], rows[right]["gradient"]
        pairs[f"{left}__{right}"] = (
            {"distance": None, "cosine": None, "cosine_reason": "undefined_branch"}
            if a is None or b is None else pair_geometry(a, b))
    return {"branches": rows, "leverage": leverage, "delivered_pairs": pairs}


def contrasts(values):
    """For CE use six AFTER losses: shared before loss cancels exactly algebraically."""
    require(type(values) is dict and set(values) == set(BRANCHES), "Wrong branch membership")
    numeric = {name: None if values[name] is None else number(values[name]) for name in BRANCHES}
    result = {}
    for name, coefficients in COEFFICIENTS.items():
        components = {branch: numeric[branch] for branch in coefficients}
        missing = [branch for branch, value in components.items() if value is None]
        result[name] = {
            "coefficients": dict(coefficients), "components": components,
            "defined_mask": {branch: value is not None for branch, value in components.items()},
            "reason": "undefined_required_branch" if missing else None,
            "value": None if missing else number(math.fsum(
                coefficients[branch] * value for branch, value in components.items())),
        }
    return result


def primary_summary(values):
    require(type(values) is dict and set(values) == set(PRIMARY), "Incomplete or unexpected primary bundles")
    ordered = [None if values[bundle] is None else number(values[bundle]) for bundle in PRIMARY]
    mask = [value is not None for value in ordered]
    complete = all(mask)
    return {"bundles": list(PRIMARY), "values": ordered, "defined_mask": mask,
            "mean": number(math.fsum(ordered) / 3) if complete else None,
            "min": min(ordered) if complete else None,
            "max": max(ordered) if complete else None,
            "reason": None if complete else "incomplete_predeclared_primary_mask"}
