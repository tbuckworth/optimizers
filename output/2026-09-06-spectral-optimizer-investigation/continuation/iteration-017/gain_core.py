"""I17 gain-normalized scalar and spectral controls; import performs no work."""
from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
from typing import Any

import torch


HERE = Path(__file__).resolve().parent
_I16_PATH = HERE.parent / "iteration-016" / "scalar_core.py"
_I16_SPEC = importlib.util.spec_from_file_location("_i17_frozen_i16_scalar_core",
                                                   _I16_PATH)
if _I16_SPEC is None or _I16_SPEC.loader is None:  # pragma: no cover
    raise ImportError("cannot load frozen I16 scalar core")
i16 = importlib.util.module_from_spec(_I16_SPEC)
_I16_SPEC.loader.exec_module(i16)
i15, i14, i9 = i16.i15, i16.i14, i16.i9


LR = 0.03
MOMENTUM = 0.9
POST_MEAN_DECAY = 0.99
MANUAL_SHRINKAGE = 0.9997
REAL_POLICIES = ("scalar_k0p5", "scalar_k0p9", "scalar_k1",
                 "spectral_mean_projected")
TEST_ONLY_POLICIES = ("scalar_k0",)
POLICIES = REAL_POLICIES + TEST_ONLY_POLICIES
_SCALAR_K = {"scalar_k0": 0.0, "scalar_k0p5": 0.5,
             "scalar_k0p9": 0.9, "scalar_k1": 1.0}


class GainCoreError(ValueError):
    """Structural, configuration, or normalized-delivery contract error."""


NumericalFailure = i14.NumericalFailure


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise GainCoreError(message)


def validate_parent_snapshot(value: Any) -> dict[str, Any]:
    """Bind an exact audited I14 SGDm/.03 h100 parent without restoring it."""
    bound = i16.validate_parent_snapshot(value)
    return {"schema": "i17_parent_binding_v1", "base": bound["base"],
            "lr": bound["lr"], "observer_step": bound["observer_step"],
            "snapshot_digest": bound["snapshot_digest"]}


def _component(value: torch.Tensor, basis: torch.Tensor) -> dict[str, Any]:
    return i15._component(value, basis)


def _residual(value: torch.Tensor, scale: float, label: str) -> dict[str, float]:
    return i15._residual(value, scale, label)


def _scalar_inputs(raw: torch.Tensor, post_mean: torch.Tensor,
                   old_buffer: torch.Tensor, k: float
                   ) -> tuple[torch.Tensor, torch.Tensor, str | None, str | None]:
    applied, delivery_literal = i16._literal_delivery(raw, post_mean, k)
    selected_old, history_literal = i16._literal_history(old_buffer, k)
    return applied, selected_old, delivery_literal, history_literal


@torch.no_grad()
def _advance_buffer(model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                    applied: torch.Tensor) -> torch.Tensor:
    """Advance retained SGDm buffers with Torch's in-place mul-then-add order."""
    applied = i14._finite_tensor(applied, "unnormalized recurrence input")
    position = 0
    flattened = []
    for parameter in model.parameters():
        buffer = optimizer.state[parameter]["momentum_buffer"]
        count = parameter.numel()
        piece = applied[position:position + count].reshape_as(parameter)
        _need(piece.dtype == buffer.dtype and piece.device == buffer.device,
              "recurrence input dtype/device differs")
        buffer.mul_(MOMENTUM).add_(piece)
        flattened.append(buffer.detach().reshape(-1))
        position += count
    _need(position == applied.numel(), "recurrence input length differs")
    return i14._finite_tensor(torch.cat(flattened).clone(),
                              "unnormalized new momentum buffer")


@torch.no_grad()
def _apply_data_step(model: torch.nn.Module, delivered_buffer: torch.Tensor) -> None:
    """Apply ``-LR * delivered_buffer`` without changing optimizer state."""
    delivered_buffer = i14._finite_tensor(delivered_buffer, "delivered buffer")
    position = 0
    for parameter in model.parameters():
        count = parameter.numel()
        piece = delivered_buffer[position:position + count].reshape_as(parameter)
        _need(piece.dtype == parameter.dtype and piece.device == parameter.device,
              "delivered-buffer dtype/device differs")
        parameter.add_(piece, alpha=-LR)
        position += count
    _need(position == delivered_buffer.numel(), "delivered-buffer length differs")


def gain_step(model: torch.nn.Module, optimizer: torch.optim.Optimizer,
              tracker: Any, x: torch.Tensor, target: torch.Tensor,
              policy: str, *, capture_digests: bool = False) -> dict[str, Any]:
    """Apply one normalized-response SGDm step from a post-warmup parent.

    The observer ingests the raw gradient exactly once. The optimizer stores the
    unnormalized recurrent buffer; only the data displacement is gain-corrected.
    ``scalar_k0`` exists solely for synthetic bitwise parity with I16.
    """
    _need(type(policy) is str and policy in POLICIES, "unknown I17 gain policy")
    _need(type(capture_digests) is bool, "capture_digests must be an exact bool")
    i14._validate_optimizer(model, optimizer, "sgdm", LR)
    i14._validate_tracker(model, optimizer, tracker)
    _need(float(tracker.decay) == POST_MEAN_DECAY, "observer decay differs")
    i14._finite_live(model, optimizer, tracker, "pre_gain_step")

    before = i9.flat_params(model)
    old_buffer = i15._buffer(model, optimizer)
    observer_before = int(tracker.step_count)
    _need(observer_before >= 100 and tracker.grad_mean is not None,
          "I17 requires an initialized post-warmup observer")
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

    action_raw = i15._action(tracker, raw)
    action_mean = i15._action(tracker, post_mean)
    action_old = i15._action(tracker, old_buffer)
    mean_complement = i14._finite_tensor(post_mean - action_mean,
                                         "post-mean action complement")
    mean_delivery = i14._finite_tensor(native + mean_complement,
                                        "spectral mean delivery")

    family = "scalar" if policy in _SCALAR_K else "spectral"
    k = _SCALAR_K.get(policy)
    if family == "scalar":
        assert k is not None
        applied, selected_old, delivery_literal, history_literal = _scalar_inputs(
            raw, post_mean, old_buffer, k)
    else:
        applied = mean_delivery
        selected_old = action_old
        delivery_literal = None
        history_literal = "native_action_old_buffer"

    i15._set_buffer(model, optimizer, selected_old)
    i9.set_grad(model, applied)
    new_buffer = _advance_buffer(model, optimizer, applied)
    action_new = i15._action(tracker, new_buffer)
    if family == "scalar":
        assert k is not None
        gain_correction = 1.0 - MOMENTUM * k
        delivered_buffer = i14._finite_tensor(gain_correction * new_buffer,
                                               "gain-normalized scalar buffer")
        delivery_operator = "(1-rho*k)*b_t"
    else:
        gain_correction = None
        delivered_buffer = i14._finite_tensor(
            new_buffer - MOMENTUM * action_new,
            "gain-normalized spectral buffer")
        delivery_operator = "(I-rho*A_t)*b_t"

    nominal_decay = -LR * i14.WD * before
    i14._manual_decay(model, MANUAL_SHRINKAGE)
    after_decay = i9.flat_params(model)
    manual_decay = i14._finite_tensor(after_decay - before,
                                       "manual decay displacement")
    _apply_data_step(model, delivered_buffer)
    after = i9.flat_params(model)
    i14._finite_tensor(after, "updated parameters")
    i14._finite_live(model, optimizer, tracker, "post_gain_step")
    retained_buffer = i15._buffer(model, optimizer)
    _need(torch.equal(retained_buffer, new_buffer),
          "gain-corrected delivery must not replace recurrent optimizer state")

    expected_mean = POST_MEAN_DECAY * pre_mean + (1.0 - POST_MEAN_DECAY) * raw
    if family == "scalar":
        assert k is not None
        expected_applied = k * raw + (1.0 - k) * post_mean
        expected_selected_old = k * old_buffer
        expected_delivered = (1.0 - MOMENTUM * k) * new_buffer
    else:
        expected_applied = native + (post_mean - action_mean)
        expected_selected_old = action_old
        expected_delivered = new_buffer - MOMENTUM * action_new
    expected_buffer = MOMENTUM * selected_old + applied

    residual_values = {
        "native_minus_actual_action_raw": native - action_raw,
        "post_mean_recurrence": post_mean - expected_mean,
        "applied_input_definition": applied - expected_applied,
        "selected_old_buffer_definition": selected_old - expected_selected_old,
        "unnormalized_buffer_recurrence": new_buffer - expected_buffer,
        "delivered_buffer_definition": delivered_buffer - expected_delivered,
    }
    residual_scales = {
        "native_minus_actual_action_raw": i14._norm(native, "native gradient")
            + i14._norm(action_raw, "action raw gradient"),
        "post_mean_recurrence": i14._norm(post_mean, "post mean")
            + POST_MEAN_DECAY * i14._norm(pre_mean, "pre mean")
            + (1.0 - POST_MEAN_DECAY) * i14._norm(raw, "raw gradient"),
        "applied_input_definition": i14._norm(applied, "applied input")
            + i14._norm(expected_applied, "expected applied input"),
        "selected_old_buffer_definition": i14._norm(selected_old, "selected old")
            + i14._norm(expected_selected_old, "expected selected old"),
        "unnormalized_buffer_recurrence": i14._norm(new_buffer, "new buffer")
            + MOMENTUM * i14._norm(selected_old, "selected old")
            + i14._norm(applied, "applied input"),
        "delivered_buffer_definition": i14._norm(delivered_buffer, "delivered buffer")
            + i14._norm(expected_delivered, "expected delivered buffer"),
    }
    residuals = {name: _residual(value, residual_scales[name], name)
                 for name, value in residual_values.items()}

    total_delta = i14._finite_tensor(after - before, "total displacement")
    data_delta = i14._finite_tensor(total_delta - nominal_decay,
                                    "nominal-decay-adjusted data displacement")
    ideal_data_delta = i14._finite_tensor(-LR * delivered_buffer,
                                          "ideal normalized data displacement")
    ideal_defect = i14._norm(data_delta - ideal_data_delta,
                             "empirical ideal data-step defect")
    ideal_norm = i14._norm(ideal_data_delta, "ideal data displacement")
    manual_error = i14._norm(manual_decay - nominal_decay,
                             "manual decay realization error")

    twice_action_new = i15._action(tracker, action_new)
    action_idempotence = i14._norm(twice_action_new - action_new,
                                   "new-buffer action idempotence defect")
    action_idempotence_reference = 128.0 * torch.finfo(new_buffer.dtype).eps * (
        i14._norm(twice_action_new, "twice-action new buffer")
        + i14._norm(action_new, "action new buffer"))
    gram = basis.T @ basis
    eye = torch.eye(gram.shape[0], dtype=gram.dtype, device=gram.device)
    orthogonality_error = i14._finite_scalar(
        torch.linalg.matrix_norm(gram - eye, ord=2).item(),
        "current basis orthogonality error")

    digests = None
    if capture_digests:
        digests = {
            "raw_gradient": raw_digest,
            "post_observer": i9.tree_digest(i9._tracker_state(tracker)),
            "old_momentum_buffer": i9.tree_digest(old_buffer),
            "unnormalized_new_momentum_buffer": i9.tree_digest(new_buffer),
            "delivered_buffer": i9.tree_digest(delivered_buffer),
        }

    record = {
        "schema": "i17_gain_step_v1", "base": "sgdm", "lr": LR,
        "policy": policy, "family": family, "k": k,
        "loss": float(loss.detach().item()),
        "observer": {"step_before": observer_before, "step_after": observer_after,
                     "filtering_active": True, "used": True,
                     "basis_rank": int(basis.shape[1])},
        "gradient_filter_applied": family == "spectral",
        "native_projection_used_for_recurrence": family == "spectral",
        "normalized_delivery": {
            "operator": delivery_operator,
            "scalar_gain_correction": gain_correction,
            "manual_parameter_update": True,
            "unnormalized_buffer_retained": True,
            "delivery_literal": delivery_literal,
            "history_literal": history_literal,
            "unnormalized_buffer_norm": i14._norm(new_buffer, "new buffer"),
            "delivered_buffer_norm": i14._norm(delivered_buffer, "delivered buffer"),
        },
        "gradient": {
            "raw_norm": i14._norm(raw, "raw gradient"),
            "native_projection_norm": i14._norm(native, "native projection"),
            "post_mean_norm": i14._norm(post_mean, "post mean"),
            "recurrence_input_norm": i14._norm(applied, "recurrence input"),
        },
        "components": {
            "raw_gradient": _component(raw, basis),
            "native_projection": _component(native, basis),
            "post_mean": _component(post_mean, basis),
            "native_action_post_mean": _component(action_mean, basis),
            "post_mean_action_complement": _component(mean_complement, basis),
            "recurrence_input": _component(applied, basis),
            "old_momentum_buffer": _component(old_buffer, basis),
            "selected_old_momentum_buffer": _component(selected_old, basis),
            "unnormalized_new_momentum_buffer": _component(new_buffer, basis),
            "native_action_new_momentum_buffer": _component(action_new, basis),
            "delivered_buffer": _component(delivered_buffer, basis),
        },
        "displacement": {
            "total_norm": i14._norm(total_delta, "total displacement"),
            "data_norm": i14._norm(data_delta, "data displacement"),
            "ideal_data_norm": ideal_norm,
            "nominal_decay_norm": i14._norm(nominal_decay, "nominal decay"),
            "actual_minus_ideal_data_norm": ideal_defect,
            "actual_minus_ideal_relative": (ideal_defect / ideal_norm
                                              if ideal_norm > 0 else None),
            "total_current_basis_leakage": i14._leakage(total_delta, basis,
                                                          "total leakage"),
            "data_current_basis_leakage": i14._leakage(data_delta, basis,
                                                         "data leakage"),
            "ideal_data_current_basis_leakage": i14._leakage(
                ideal_data_delta, basis, "ideal data leakage"),
            "raw_gradient_dot_data_delta": i14._dot(
                raw, data_delta, "raw/data displacement"),
            "post_mean_dot_data_delta": i14._dot(
                post_mean, data_delta, "post-mean/data displacement"),
            "delivered_buffer_dot_data_delta": i14._dot(
                delivered_buffer, data_delta, "delivered-buffer/data displacement"),
        },
        "action_diagnostics": {
            "current_basis_orthogonality_error": orthogonality_error,
            "new_buffer_action_idempotence_defect": {
                "norm": action_idempotence,
                "homogeneous_roundoff_reference": action_idempotence_reference,
                "enforced": False,
            },
        },
        "decay": {
            "coefficient": i14.WD, "factor": MANUAL_SHRINKAGE,
            "manual_before_data_step": True,
            "manual_actual_norm": i14._norm(manual_decay, "manual decay"),
            "manual_minus_nominal_norm": manual_error,
        },
        "algebra_residuals": residuals,
        "digests": digests,
    }
    i14._scan_finite(record, "I17 gain step")
    json.dumps(record, ensure_ascii=True, allow_nan=False, separators=(",", ":"))
    return record


__all__ = ["GainCoreError", "NumericalFailure", "LR", "MOMENTUM",
           "POST_MEAN_DECAY", "MANUAL_SHRINKAGE", "REAL_POLICIES",
           "TEST_ONLY_POLICIES", "POLICIES", "i16", "i15", "i14", "i9",
           "validate_parent_snapshot", "gain_step"]
