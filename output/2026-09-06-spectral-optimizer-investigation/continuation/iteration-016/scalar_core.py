"""I16 scalar temporal controls; importing this module performs no work."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import torch


HERE = Path(__file__).resolve().parent
_I15_PATH = HERE.parent / "iteration-015" / "history_core.py"
_I15_SPEC = importlib.util.spec_from_file_location("_i16_frozen_i15_history_core",
                                                    _I15_PATH)
if _I15_SPEC is None or _I15_SPEC.loader is None:  # pragma: no cover
    raise ImportError("cannot load frozen I15 history core")
i15 = importlib.util.module_from_spec(_I15_SPEC)
_I15_SPEC.loader.exec_module(i15)
i14, i9 = i15.i14, i15.i9


LR = 0.03
MOMENTUM = 0.9
DECAY = 0.99
REAL_K = (0.0, 0.5, 0.9)
TEST_ONLY_K = (1.0,)
ALL_K = REAL_K + TEST_ONLY_K


class ScalarCoreError(ValueError):
    """Structural, configuration, or scalar-intervention contract error."""


NumericalFailure = i14.NumericalFailure


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise ScalarCoreError(message)


def _k(value: Any) -> float:
    _need(type(value) is float and value in ALL_K,
          "k must be one exact registered scalar-control float")
    return value


def validate_parent_snapshot(value: Any) -> dict[str, Any]:
    """Bind an exact audited I14 SGDm/.03 h100 parent without restoring it."""
    bound = i15.validate_parent_snapshot(value)
    return {"schema": "i16_parent_binding_v1", "base": bound["base"],
            "lr": bound["lr"], "observer_step": bound["observer_step"],
            "snapshot_digest": bound["snapshot_digest"]}


def _component(value: torch.Tensor, basis: torch.Tensor) -> dict[str, Any]:
    return i15._component(value, basis)


def _residual(value: torch.Tensor, scale: float, label: str) -> dict[str, float]:
    return i15._residual(value, scale, label)


def _literal_delivery(raw: torch.Tensor, post_mean: torch.Tensor,
                      k: float) -> tuple[torch.Tensor, str | None]:
    if k == 0.0:
        result = post_mean.detach().clone()
        _need(torch.equal(result, post_mean), "k=0 must deliver the literal post-mean")
        return result, "post_mean"
    if k == 1.0:
        result = raw.detach().clone()
        _need(torch.equal(result, raw), "k=1 must deliver the literal raw gradient")
        return result, "raw_gradient"
    return i14._finite_tensor(k * raw + (1.0 - k) * post_mean,
                              "scalar mixed delivery"), None


def _literal_history(old: torch.Tensor, k: float) -> tuple[torch.Tensor, str | None]:
    if k == 0.0:
        result = torch.zeros_like(old)
        _need(bool(torch.count_nonzero(result) == 0),
              "k=0 must install a literal zero old buffer")
        return result, "zero"
    if k == 1.0:
        result = old.detach().clone()
        _need(torch.equal(result, old), "k=1 must retain the literal old buffer")
        return result, "old_buffer"
    return i14._finite_tensor(k * old, "scaled old momentum buffer"), None


def scalar_step(model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                tracker: Any, x: torch.Tensor, target: torch.Tensor,
                k: float, *, capture_digests: bool = False) -> dict[str, Any]:
    """Apply one registered isotropic temporal-control SGDm step.

    The canonical observer ingests the raw gradient exactly once, but its
    projected gradient is diagnostic only. ``k=1`` is exposed solely for
    synthetic equality tests and must never be scheduled by a real runner.
    """
    k = _k(k)
    _need(type(capture_digests) is bool, "capture_digests must be an exact bool")
    i14._validate_optimizer(model, optimizer, "sgdm", LR)
    i14._validate_tracker(model, optimizer, tracker)
    _need(float(tracker.decay) == DECAY, "observer decay differs")
    i14._finite_live(model, optimizer, tracker, "pre_scalar_step")
    before = i9.flat_params(model)
    old_buffer = i15._buffer(model, optimizer)
    observer_before = int(tracker.step_count)
    _need(observer_before >= 100 and tracker.grad_mean is not None,
          "I16 requires an initialized post-warmup observer")
    pre_mean = i14._finite_tensor(tracker.grad_mean.detach().clone(),
                                  "pre-ingest mean")

    optimizer.zero_grad(set_to_none=True)
    loss = i9._objective_loss(model, x, target)
    _need(type(loss) is torch.Tensor and loss.ndim == 0,
          "objective loss must be a scalar tensor")
    if not bool(torch.isfinite(loss)):
        raise NumericalFailure("objective loss is nonfinite")
    loss.backward()
    raw = i14._flat_grad(model)
    raw_digest = i9.tree_digest(raw) if capture_digests else None
    observer_diagnostic = tracker.filter_grad()
    native = i14._flat_grad(model)
    observer_after = int(tracker.step_count)
    _need(observer_after == observer_before + 1
          and int(observer_diagnostic["step"]) == observer_after
          and bool(observer_diagnostic["filtering_active"]),
          "observer must advance once with active filtering")
    basis = i9._basis(tracker)
    _need(basis is not None and basis.ndim == 2 and basis.shape[0] == raw.numel(),
          "post-ingest native basis is unavailable or malformed")
    basis = i14._finite_tensor(basis, "post-ingest basis")
    post_mean = i14._finite_tensor(tracker.grad_mean.detach().clone(),
                                   "post-ingest mean")
    applied, delivery_literal = _literal_delivery(raw, post_mean, k)
    scaled_old, history_literal = _literal_history(old_buffer, k)
    i15._set_buffer(model, optimizer, scaled_old)
    i9.set_grad(model, applied)

    nominal_decay = -LR * i14.WD * before
    i14._manual_decay(model, 1.0 - LR * i14.WD)
    after_decay = i9.flat_params(model)
    manual_decay = i14._finite_tensor(after_decay - before,
                                       "manual decay displacement")
    optimizer.step()
    after = i9.flat_params(model)
    i14._finite_tensor(after, "updated parameters")
    i14._finite_live(model, optimizer, tracker, "post_scalar_step")
    new_buffer = i15._buffer(model, optimizer)
    expected_buffer = i14._finite_tensor(MOMENTUM * scaled_old + applied,
                                         "expected scalar-control buffer")
    total_delta = i14._finite_tensor(after - before, "total displacement")
    data_delta = i14._finite_tensor(total_delta - nominal_decay,
                                    "nominal-decay-adjusted data displacement")
    ideal_data_delta = i14._finite_tensor(-LR * expected_buffer,
                                          "ideal scalar-control data displacement")

    expected_mean = DECAY * pre_mean + (1.0 - DECAY) * raw
    expected_delivery = k * raw + (1.0 - k) * post_mean
    expected_scaled_old = k * old_buffer
    residual_values = {
        "post_mean_recurrence": post_mean - expected_mean,
        "delivery_definition": applied - expected_delivery,
        "scaled_old_buffer_definition": scaled_old - expected_scaled_old,
        "momentum_buffer_recurrence": new_buffer - expected_buffer,
    }
    residual_scales = {
        "post_mean_recurrence": (i14._norm(post_mean, "post mean")
            + DECAY * i14._norm(pre_mean, "pre mean")
            + (1.0 - DECAY) * i14._norm(raw, "raw gradient")),
        "delivery_definition": (i14._norm(applied, "applied delivery")
            + k * i14._norm(raw, "raw gradient")
            + (1.0 - k) * i14._norm(post_mean, "post mean")),
        "scaled_old_buffer_definition": (i14._norm(scaled_old, "scaled old buffer")
            + k * i14._norm(old_buffer, "old buffer")),
        "momentum_buffer_recurrence": (i14._norm(new_buffer, "new buffer")
            + MOMENTUM * i14._norm(scaled_old, "scaled old buffer")
            + i14._norm(applied, "applied delivery")),
    }
    residuals = {name: _residual(value, residual_scales[name], name)
                 for name, value in residual_values.items()}
    manual_error = i14._norm(manual_decay - nominal_decay,
                             "manual decay realization error")
    ideal_defect = i14._norm(data_delta - ideal_data_delta,
                             "empirical ideal data-step defect")

    digests = None
    if capture_digests:
        digests = {
            "raw_gradient": raw_digest,
            "post_observer": i9.tree_digest(i9._tracker_state(tracker)),
            "old_momentum_buffer": i9.tree_digest(old_buffer),
            "applied_gradient": i9.tree_digest(applied),
            "new_momentum_buffer": i9.tree_digest(new_buffer),
        }
    record = {
        "schema": "i16_scalar_step_v1", "base": "sgdm", "lr": LR, "k": k,
        "loss": float(loss.detach().item()),
        "observer": {"step_before": observer_before, "step_after": observer_after,
                     "filtering_active": True, "used": True,
                     "basis_rank": int(basis.shape[1])},
        "gradient_filter_applied": False,
        "native_projection_used_for_delivery": False,
        "gradient": {"raw_norm": i14._norm(raw, "raw gradient"),
                     "post_mean_norm": i14._norm(post_mean, "post mean"),
                     "native_projection_norm": i14._norm(native, "native projection"),
                     "applied_norm": i14._norm(applied, "applied delivery")},
        "components": {
            "raw_gradient": _component(raw, basis),
            "post_mean": _component(post_mean, basis),
            "native_projection_diagnostic": _component(native, basis),
            "applied_delivery": _component(applied, basis),
            "old_momentum_buffer": _component(old_buffer, basis),
            "scaled_old_momentum_buffer": _component(scaled_old, basis),
            "new_momentum_buffer": _component(new_buffer, basis),
        },
        "temporal_control": {
            "old_history_scale": k,
            "raw_gradient_weight": k,
            "post_mean_weight": 1.0 - k,
            "effective_old_history_coefficient": MOMENTUM * k,
            "delivery_literal": delivery_literal,
            "history_literal": history_literal,
            "expected_buffer_norm": i14._norm(expected_buffer, "expected buffer"),
            "empirical_ideal_data_step_defect_norm": ideal_defect,
        },
        "displacement": {
            "total_norm": i14._norm(total_delta, "total displacement"),
            "data_norm": i14._norm(data_delta, "data displacement"),
            "nominal_decay_norm": i14._norm(nominal_decay, "nominal decay"),
            "total_current_basis_leakage": i14._leakage(total_delta, basis,
                                                          "total leakage"),
            "data_current_basis_leakage": i14._leakage(data_delta, basis,
                                                         "data leakage"),
            "raw_gradient_dot_data_delta": i14._dot(raw, data_delta,
                                                       "raw/data displacement"),
            "post_mean_dot_data_delta": i14._dot(post_mean, data_delta,
                                                   "post-mean/data displacement"),
            "applied_gradient_dot_data_delta": i14._dot(
                applied, data_delta, "applied/data displacement"),
        },
        "decay": {"coefficient": i14.WD, "factor": 1.0 - LR * i14.WD,
                  "manual_before_optimizer_step": True,
                  "manual_actual_norm": i14._norm(manual_decay, "manual decay"),
                  "manual_minus_nominal_norm": manual_error},
        "algebra_residuals": residuals,
        "digests": digests,
    }
    i14._scan_finite(record, "I16 scalar step")
    json.dumps(record, ensure_ascii=True, allow_nan=False, separators=(",", ":"))
    return record


__all__ = ["ScalarCoreError", "NumericalFailure", "LR", "MOMENTUM", "DECAY",
           "REAL_K", "TEST_ONLY_K", "ALL_K", "i15", "i14", "i9",
           "validate_parent_snapshot", "scalar_step"]
