"""Inert component objectives and actual-displacement accounting.

Callers supply FP32 logits [image, view, class] and ordered FP32 parameter
tensors. This module never constructs a model, reads files, samples data,
inspects a GPU, restores an optimizer, or mutates caller tensors/.grad.
Production roster, 25 shifts, 10 classes and original-view index 12 belong
to the future runner; smaller caller-supplied grids support fabricated tests.
"""
from __future__ import annotations

import math
from collections.abc import Mapping

import torch


OBJECTIVE_KEYS = ("S", "F", "C", "L", "S_true", "S_uniform")
GRADIENT_KEYS = ("S", "F", "C", "L", "H_O", "H_T")
SCALAR_ATOL = SCALAR_RTOL = 1e-10
GRADIENT_ATOL, GRADIENT_RTOL = 1e-6, 5e-5


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _finite(value, name):
    _need(bool(torch.isfinite(value).all()), f"nonfinite {name}")


def _logits(value, ndim, name):
    _need(isinstance(value, torch.Tensor) and value.dtype == torch.float32
          and value.ndim == ndim and all(n > 0 for n in value.shape)
          and value.shape[-1] >= 2, f"{name}: expected nonempty FP32 logits")
    _finite(value, name)


def _labels(value, count, classes, device, name):
    _need(isinstance(value, torch.Tensor) and value.dtype == torch.int64
          and value.shape == (count,) and value.device == device, f"{name}: shape/dtype/device")
    _need(bool(((value >= 0) & (value < classes)).all()), f"{name}: class range")


def _gauge(logits):
    # Differentiable common-class offset removal. No detached mean or teacher.
    values = logits.to(dtype=torch.float64)
    return values - values[..., :1]


def _selected(values, labels):
    if values.ndim == 3:
        index = labels[:, None, None].expand(-1, values.shape[1], 1)
    else:
        index = labels[:, None]
    return values.gather(-1, index).squeeze(-1)


def _view_ce(values, labels):
    return torch.logsumexp(values, dim=-1) - _selected(values, labels)


def _check_inputs(i_grid, true_labels, assigned_labels, r_original, r_grid, r_true):
    _logits(i_grid, 3, "I grid")
    n, _, k = i_grid.shape
    _labels(true_labels, n, k, i_grid.device, "I true labels")
    _labels(assigned_labels, n, k, i_grid.device, "I assigned labels")
    supplied = (r_original is not None, r_grid is not None, r_true is not None)
    _need(all(supplied) or not any(supplied), "R inputs must be supplied together")
    if all(supplied):
        _logits(r_original, 2, "R original")
        _logits(r_grid, 3, "R grid")
        _need(r_original.shape == (r_grid.shape[0], k)
              and r_grid.shape[-1] == k and r_original.device == r_grid.device == i_grid.device,
              "R shape/classes/device differ")
        _labels(r_true, r_grid.shape[0], k, i_grid.device, "R true labels")


def objective_tensors(i_grid, true_labels, assigned_labels, r_original=None,
                      r_grid=None, r_true=None):
    """Return differentiable FP64 scalar S/F/C/direct-L and optional H_O/H_T.

    Every reduction gives equal weight to images and views. The fixed target
    law is .1*true_one_hot + .9*uniform, regardless of the realized wrong mask.
    The class-0 gauge is removed after conversion to FP64, before averaging.
    L is computed directly from assigned per-view CE, never from S+F+C.
    C is returned without nonnegative clamping, preserving numerical evidence.
    """
    _check_inputs(i_grid, true_labels, assigned_labels, r_original, r_grid, r_true)
    values = _gauge(i_grid)
    mean_logits = values.mean(dim=1)
    mean_log_normalizer = torch.logsumexp(mean_logits, dim=-1)
    true_logit = _selected(mean_logits, true_labels)
    assigned_logit = _selected(mean_logits, assigned_labels)
    uniform_logit = mean_logits.mean(dim=-1)
    s_true = (mean_log_normalizer - true_logit).mean()
    s_uniform = (mean_log_normalizer - uniform_logit).mean()
    result = {
        "S": .1 * s_true + .9 * s_uniform,
        "F": (.1 * true_logit + .9 * uniform_logit - assigned_logit).mean(),
        "C": (torch.logsumexp(values, dim=-1).mean(dim=1) - mean_log_normalizer).mean(),
        "L": _view_ce(values, assigned_labels).mean(),
        "S_true": s_true,
        "S_uniform": s_uniform,
    }
    if r_original is not None:
        result["H_O"] = _view_ce(_gauge(r_original), r_true).mean()
        result["H_T"] = _view_ce(_gauge(r_grid), r_true).mean()
    for name, value in result.items():
        _finite(value, name)
    return result


def _stats(grid, labels):
    """Sufficient stats; count means image-view outcomes, not image count."""
    n, views, _ = grid.shape
    count = n * views
    if count == 0:
        return {"available": False, "example_count": 0, "view_count": views,
                "count": 0, "correct": 0, "accuracy": None, "ce_sum": 0.0, "ce": None}
    loss = _view_ce(_gauge(grid), labels)
    ce_sum = math.fsum(loss.detach().cpu().reshape(-1).tolist())
    correct = int((grid.argmax(dim=-1) == labels[:, None]).sum().item())
    return {"available": True, "example_count": n, "view_count": views,
            "count": count, "correct": correct, "accuracy": correct / count,
            "ce_sum": ce_sum, "ce": ce_sum / count}


@torch.no_grad()
def metric_report(i_grid, true_labels, assigned_labels, *, original_view_index,
                  r_original=None, r_grid=None, r_true=None):
    """Pure JSON-compatible values with explicit original/per-view reductions.

    I.{original,per_view,wrong_original,wrong_per_view}.{true,assigned} contain
    sufficient stats. Wrong means assigned != true, fixed across all views.
    R.{original,per_view}.true is optional. No mean-logit accuracy is reported.
    Empty wrong subsets have zero counts/sums and unavailable mean scores.
    """
    objectives = objective_tensors(i_grid, true_labels, assigned_labels,
                                   r_original, r_grid, r_true)
    _need(type(original_view_index) is int and 0 <= original_view_index < i_grid.shape[1],
          "original_view_index outside grid")
    original = i_grid[:, original_view_index:original_view_index + 1]
    wrong = assigned_labels != true_labels
    panels = {"original": (original, true_labels, assigned_labels),
              "per_view": (i_grid, true_labels, assigned_labels),
              "wrong_original": (original[wrong], true_labels[wrong], assigned_labels[wrong]),
              "wrong_per_view": (i_grid[wrong], true_labels[wrong], assigned_labels[wrong])}
    result = {
        "objectives": {name: float(value.item()) for name, value in objectives.items()},
        "I": {name: {"true": _stats(grid, true), "assigned": _stats(grid, assigned)}
              for name, (grid, true, assigned) in panels.items()},
        "metadata": {"grid_layout": "image,view,class", "original_view_index": original_view_index,
                     "I_examples": i_grid.shape[0], "I_views": i_grid.shape[1],
                     "I_wrong_examples": int(wrong.sum().item()), "classes": i_grid.shape[2]},
    }
    if r_original is not None:
        result["R"] = {"original": {"true": _stats(r_original[:, None, :], r_true)},
                       "per_view": {"true": _stats(r_grid, r_true)}}
        result["metadata"].update(R_examples=r_grid.shape[0], R_views=r_grid.shape[1])
    return result


def flat_objective_gradients(objectives, parameters, *, retain_graph=False):
    """Differentiate an ordered scalar mapping without accumulating .grad.

    Parameters must be distinct trainable FP32 leaves on one device. Returned
    flattened FP32 tensors are detached copies in the caller's parameter order.
    Unused parameters are errors. Internal graph retention supports the several
    objectives. Unless retain_graph=True, the last call releases the graph
    reachable from its objective, not necessarily separate earlier graphs.
    After saving detached values, callers must drop all original objective and
    forward-logit graph references before proceeding to private actions.
    """
    _need(isinstance(objectives, Mapping) and len(objectives) > 0, "empty objective mapping")
    params = tuple(parameters)
    _need(params and all(isinstance(p, torch.Tensor) and p.dtype == torch.float32
                        and p.is_leaf and p.requires_grad and p.numel() > 0 for p in params),
          "parameters must be trainable nonempty FP32 leaves")
    _need(len({id(p) for p in params}) == len(params) and len({p.device for p in params}) == 1,
          "duplicate parameters or differing devices")
    _need(type(retain_graph) is bool, "retain_graph must be bool")
    for name, value in objectives.items():
        _need(type(name) is str and isinstance(value, torch.Tensor) and value.ndim == 0
              and value.dtype == torch.float64 and value.device == params[0].device
              and value.requires_grad, "differentiable FP64 scalar objective required")
        _finite(value, name)
    result = {}
    for index, (name, value) in enumerate(objectives.items()):
        parts = torch.autograd.grad(value, params, create_graph=False, allow_unused=False,
                                    retain_graph=retain_graph or index < len(objectives) - 1)
        flat = torch.cat([part.detach().reshape(-1) for part in parts]).clone()
        _finite(flat, f"{name} gradient")
        result[name] = flat
    return result


def _vector(value, dtype, name):
    _need(isinstance(value, torch.Tensor) and value.dtype == dtype
          and value.ndim == 1 and value.numel() > 0, f"{name}: vector shape/dtype")
    _finite(value, name)


@torch.no_grad()
def materialize_path(parent, endpoint, fraction):
    """FP32 subtract, multiply, add; fraction=1 preserves endpoint bytes."""
    _vector(parent, torch.float32, "parent")
    _vector(endpoint, torch.float32, "endpoint")
    _need(parent.shape == endpoint.shape and parent.device == endpoint.device, "endpoint mismatch")
    _need(type(fraction) in (int, float) and math.isfinite(fraction) and 0 <= fraction <= 1,
          "fraction must be finite in [0,1]")
    if fraction == 1:
        point = endpoint.detach().clone()
    elif fraction == 0:
        point = parent.detach().clone()
    else:
        difference = endpoint - parent
        scaled = difference * fraction
        point = parent + scaled
    _finite(point, "materialized point")
    return point


@torch.no_grad()
def actual_delta(parent, point):
    """Subtract actual stored FP32 endpoints AFTER conversion to FP64."""
    _vector(parent, torch.float32, "parent")
    _vector(point, torch.float32, "point")
    _need(parent.shape == point.shape and parent.device == point.device, "point mismatch")
    return point.to(torch.float64) - parent.to(torch.float64)


@torch.no_grad()
def path_accounting(parent, endpoint, decay_endpoint, fraction):
    """Return actual path points and total/decay/data FP64 deltas and norms."""
    point = materialize_path(parent, endpoint, fraction)
    decay_point = materialize_path(parent, decay_endpoint, fraction)
    deltas = {"total_delta": actual_delta(parent, point),
              "decay_delta": actual_delta(parent, decay_point),
              "data_delta": actual_delta(decay_point, point)}
    return {"point": point, "decay_point": decay_point, **deltas,
            **{name.replace("delta", "norm"): float(value.norm().item()) for name, value in deltas.items()}}


@torch.no_grad()
def signed_utility(gradient, displacement):
    """-grad·actual_delta: FP64 products, CPU ordered math.fsum, Python float."""
    _vector(gradient, torch.float32, "gradient")
    _vector(displacement, torch.float64, "displacement")
    _need(gradient.shape == displacement.shape and gradient.device == displacement.device,
          "dot vector mismatch")
    products = gradient.to(torch.float64) * displacement
    _finite(products, "dot products")
    result = -math.fsum(products.detach().cpu().tolist())
    _need(math.isfinite(result), "nonfinite signed utility")
    return result
