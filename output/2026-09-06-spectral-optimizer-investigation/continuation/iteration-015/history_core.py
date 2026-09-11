"""I15 SGDm delivery/history interventions; import performs no work."""
from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
from typing import Any

import torch


HERE = Path(__file__).resolve().parent
_I14_PATH = HERE.parent / "iteration-014" / "optimizer_core.py"
_I14_SPEC = importlib.util.spec_from_file_location("_i15_frozen_i14_optimizer_core", _I14_PATH)
if _I14_SPEC is None or _I14_SPEC.loader is None:  # pragma: no cover
    raise ImportError("cannot load frozen I14 optimizer core")
i14 = importlib.util.module_from_spec(_I14_SPEC)
_I14_SPEC.loader.exec_module(i14)
i9 = i14.i9


LR = 0.03
MOMENTUM = 0.9
DECAY = 0.99
REAL_POLICIES = ("current_projected_history", "mean_native", "mean_projected_history")
TEST_ONLY_POLICIES = ("current_native",)
POLICIES = REAL_POLICIES + TEST_ONLY_POLICIES


class HistoryCoreError(ValueError):
    """Structural, configuration, or intervention-contract error."""


NumericalFailure = i14.NumericalFailure


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise HistoryCoreError(message)


def _action(tracker: Any, value: torch.Tensor) -> torch.Tensor:
    """Apply the observer's actual post-ingest native action without mutation."""
    result = tracker._project_gradient(value.detach().clone())
    _need(type(result) is torch.Tensor and result.shape == value.shape
          and result.dtype == value.dtype and result.device == value.device,
          "native action changed tensor topology")
    return i14._finite_tensor(result.detach().clone(), "native action")


def _buffer(model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> torch.Tensor:
    values = []
    for parameter in model.parameters():
        state = optimizer.state.get(parameter)
        _need(type(state) is dict and tuple(state) == ("momentum_buffer",),
              "every SGDm parameter must have exactly one initialized momentum buffer")
        value = state["momentum_buffer"]
        _need(type(value) is torch.Tensor and value.shape == parameter.shape
              and value.dtype == parameter.dtype and value.device == parameter.device,
              "momentum-buffer topology differs")
        values.append(i14._finite_tensor(value, "momentum buffer").detach().reshape(-1))
    _need(bool(values), "model has no momentum buffers")
    return i14._finite_tensor(torch.cat(values).clone(), "flat momentum buffer")


@torch.no_grad()
def _set_buffer(model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                value: torch.Tensor) -> None:
    value = i14._finite_tensor(value, "replacement momentum buffer")
    position = 0
    for parameter in model.parameters():
        buffer = optimizer.state[parameter]["momentum_buffer"]
        size = parameter.numel()
        _need(value.device == buffer.device, "replacement buffer device differs")
        buffer.copy_(value[position:position + size].reshape_as(buffer))
        position += size
    _need(position == value.numel(), "replacement buffer length differs")


def _component(value: torch.Tensor, basis: torch.Tensor) -> dict[str, Any]:
    return {"norm": i14._norm(value, "component"),
            "squared_energy": i14._dot(value, value, "component"),
            "current_basis_leakage": i14._leakage(value, basis, "component")}


def _residual(value: torch.Tensor, scale: float, label: str) -> dict[str, float]:
    norm = i14._norm(value, label)
    bound = 128.0 * torch.finfo(value.dtype).eps * scale
    _need(norm <= bound if scale > 0 else norm == 0, label + " algebra invariant differs")
    return {"norm": norm, "homogeneous_error_bound": bound}


def validate_parent_snapshot(value: Any) -> dict[str, Any]:
    """Pure structural binding for the audited I14 SGDm h100 parent."""
    keys = ("schema", "base", "lr", "model_spec", "parameter_topology", "model_state",
            "gradients", "model_modes", "optimizer", "tracker", "rng")
    _need(type(value) is dict and tuple(value) == keys
          and value["schema"] == "i14_optimizer_snapshot_v1"
          and value["base"] == "sgdm" and value["lr"] == LR,
          "I15 parent must be an exact I14 SGDm/.03 snapshot")
    tracker = value.get("tracker")
    _need(type(tracker) is dict and tracker.get("step_count") == 100
          and type(tracker.get("grad_mean")) is torch.Tensor
          and type(tracker.get("V")) is torch.Tensor,
          "I15 parent must contain an active h100 observer state")
    saved = value.get("optimizer")
    _need(type(saved) is dict and tuple(saved) == ("state", "param_groups")
          and type(saved["state"]) is dict and type(saved["param_groups"]) is list
          and len(saved["param_groups"]) == 1,
          "I15 parent optimizer topology differs")
    identifiers = saved["param_groups"][0].get("params")
    _need(type(identifiers) is list and set(saved["state"]) == set(identifiers)
          and all(type(row) is dict and tuple(row) == ("momentum_buffer",)
                  for row in saved["state"].values()),
          "I15 parent momentum state is incomplete")
    return {"schema": "i15_parent_binding_v1", "base": "sgdm", "lr": LR,
            "observer_step": 100, "snapshot_digest": i9.tree_digest(value)}


def history_step(model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                 tracker: Any, x: torch.Tensor, target: torch.Tensor,
                 policy: str, *, capture_digests: bool = False) -> dict[str, Any]:
    """Take one SGDm step after changing delivery and/or only the old buffer.

    ``current_native`` is exposed solely for synthetic parity testing. A real
    runner must restrict membership to ``REAL_POLICIES``.
    """
    _need(type(policy) is str and policy in POLICIES, "unknown I15 history policy")
    _need(type(capture_digests) is bool, "capture_digests must be an exact bool")
    i14._validate_optimizer(model, optimizer, "sgdm", LR)
    i14._validate_tracker(model, optimizer, tracker)
    _need(float(tracker.decay) == DECAY, "observer decay differs")
    i14._finite_live(model, optimizer, tracker, "pre_history_step")
    before = i9.flat_params(model)
    old_buffer = _buffer(model, optimizer)
    observer_before = int(tracker.step_count)
    _need(observer_before >= 100 and tracker.grad_mean is not None,
          "I15 requires an initialized post-warmup observer")
    pre_mean = i14._finite_tensor(tracker.grad_mean.detach().clone(), "pre-ingest mean")

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
    post_mean = i14._finite_tensor(tracker.grad_mean.detach().clone(), "post-ingest mean")

    action_raw = _action(tracker, raw)
    action_mean = _action(tracker, post_mean)
    action_old = _action(tracker, old_buffer)
    mean_complement = i14._finite_tensor(post_mean - action_mean,
                                         "post-mean complement")
    mean_delivery = i14._finite_tensor(native + mean_complement, "mean delivery")
    use_mean = policy.startswith("mean_")
    project_history = policy.endswith("projected_history")
    applied = mean_delivery if use_mean else native
    selected_old = action_old if project_history else old_buffer
    removed_history = i14._finite_tensor(old_buffer - selected_old, "removed history")
    _set_buffer(model, optimizer, selected_old)
    i9.set_grad(model, applied)

    nominal_decay = -LR * i14.WD * before
    i14._manual_decay(model, 1.0 - LR * i14.WD)
    after_decay = i9.flat_params(model)
    manual_decay = i14._finite_tensor(after_decay - before, "manual decay displacement")
    optimizer.step()
    after = i9.flat_params(model)
    i14._finite_tensor(after, "updated parameters")
    i14._finite_live(model, optimizer, tracker, "post_history_step")
    new_buffer = _buffer(model, optimizer)
    expected_buffer = i14._finite_tensor(MOMENTUM * selected_old + applied,
                                         "expected momentum buffer")
    total_delta = i14._finite_tensor(after - before, "total displacement")
    data_delta = i14._finite_tensor(total_delta - nominal_decay,
                                    "nominal-decay-adjusted data displacement")
    ideal_data_delta = i14._finite_tensor(-LR * expected_buffer,
                                          "ideal SGDm data displacement")

    residual_scales = {
        "native_minus_actual_action_raw": i14._norm(native, "native")
            + i14._norm(action_raw, "action raw"),
        "post_mean_recurrence": i14._norm(post_mean, "post mean")
            + DECAY * i14._norm(pre_mean, "pre mean")
            + (1.0 - DECAY) * i14._norm(raw, "raw"),
        "mean_delivery_definition": i14._norm(mean_delivery, "mean delivery")
            + i14._norm(native, "native") + i14._norm(mean_complement, "mean complement"),
        "mean_alternate_definition": i14._norm(mean_delivery, "mean delivery")
            + i14._norm(post_mean, "post mean")
            + i14._norm(_action(tracker, raw - post_mean), "centered-gradient action"),
        "momentum_buffer_recurrence": i14._norm(new_buffer, "new buffer")
            + MOMENTUM * i14._norm(selected_old, "selected old buffer")
            + i14._norm(applied, "applied"),
    }
    residual_values = {
        "native_minus_actual_action_raw": native - action_raw,
        "post_mean_recurrence": post_mean - (DECAY * pre_mean + (1.0 - DECAY) * raw),
        "mean_delivery_definition": mean_delivery - (native + mean_complement),
        "mean_alternate_definition": mean_delivery
            - (post_mean + _action(tracker, raw - post_mean)),
        "momentum_buffer_recurrence": new_buffer - expected_buffer,
    }
    residuals = {name: _residual(value, residual_scales[name], name)
                 for name, value in residual_values.items()}
    ideal_defect = i14._norm(data_delta - ideal_data_delta,
                             "empirical ideal data-step defect")
    twice_action_old = _action(tracker, action_old)
    action_idempotence = i14._norm(twice_action_old - action_old,
                                   "native-action idempotence defect")
    action_idempotence_bound = 128.0 * torch.finfo(old_buffer.dtype).eps * (
        i14._norm(twice_action_old, "twice-projected old buffer")
        + i14._norm(action_old, "projected old buffer"))
    gram = basis.T @ basis
    eye = torch.eye(gram.shape[0], dtype=gram.dtype, device=gram.device)
    basis_orthogonality_error = i14._finite_scalar(
        torch.linalg.matrix_norm(gram - eye, ord=2).item(),
        "current basis orthogonality error")
    manual_error = i14._norm(manual_decay - nominal_decay,
                             "manual decay realization error")

    digests = None
    if capture_digests:
        digests = {
            "raw_gradient": raw_digest,
            "post_observer": i9.tree_digest(i9._tracker_state(tracker)),
            "applied_gradient": i9.tree_digest(applied),
            "old_momentum_buffer": i9.tree_digest(old_buffer),
            "native_action_old_buffer": i9.tree_digest(action_old),
            "new_momentum_buffer": i9.tree_digest(new_buffer),
        }
    record = {
        "schema": "i15_history_step_v1", "base": "sgdm", "lr": LR,
        "policy": policy, "loss": float(loss.detach().item()),
        "observer": {"step_before": observer_before, "step_after": observer_after,
                     "filtering_active": True, "used": True,
                     "basis_rank": int(basis.shape[1])},
        "gradient_filter_applied": True,
        "gradient": {"raw_norm": i14._norm(raw, "raw gradient"),
                     "applied_norm": i14._norm(applied, "applied gradient"),
                     "native_norm": i14._norm(native, "native gradient"),
                     "mean_complement_norm": i14._norm(mean_complement,
                                                        "mean complement")},
        "components": {
            "raw_gradient": _component(raw, basis),
            "native_projection": _component(native, basis),
            "post_mean": _component(post_mean, basis),
            "native_action_post_mean": _component(action_mean, basis),
            "post_mean_complement": _component(mean_complement, basis),
            "mean_delivery": _component(mean_delivery, basis),
            "applied_delivery": _component(applied, basis),
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
            "mean_complement_dot_data_delta": i14._dot(
                mean_complement, data_delta, "mean-complement/data displacement"),
            "projected_old_history_dot_data_delta": i14._dot(
                action_old, data_delta, "projected-history/data displacement"),
            "removed_old_history_dot_data_delta": i14._dot(
                old_buffer - action_old, data_delta, "removed-history/data displacement"),
        },
        "decay": {"coefficient": i14.WD, "factor": 1.0 - LR * i14.WD,
                  "manual_before_optimizer_step": True,
                  "manual_actual_norm": i14._norm(manual_decay, "manual decay"),
                  "manual_minus_nominal_norm": manual_error},
        "momentum_history": {
            "history_projected": project_history,
            "buffer_before": _component(old_buffer, basis),
            "native_action_old_buffer": _component(action_old, basis),
            "removed_old_buffer": _component(old_buffer - action_old, basis),
            "selected_old_buffer": _component(selected_old, basis),
            "buffer_after": _component(new_buffer, basis),
            "expected_buffer_after_norm": i14._norm(expected_buffer, "expected buffer"),
            "native_action_idempotence_defect": {
                "norm": action_idempotence,
                "homogeneous_roundoff_reference": action_idempotence_bound,
                "enforced": False,
            },
            "current_basis_orthogonality_error": basis_orthogonality_error,
            "native_action_old_dot_removed": i14._dot(
                action_old, old_buffer - action_old, "action/removed history"),
            "removed_old_dot_mean_complement": i14._dot(
                old_buffer - action_old, mean_complement, "removed history/mean complement"),
            "empirical_ideal_data_step_defect_norm": ideal_defect,
        },
        "algebra_residuals": residuals,
        "digests": digests,
    }
    i14._scan_finite(record, "I15 history step")
    json.dumps(record, ensure_ascii=True, allow_nan=False, separators=(",", ":"))
    return record


__all__ = ["HistoryCoreError", "NumericalFailure", "LR", "MOMENTUM", "DECAY",
           "REAL_POLICIES", "TEST_ONLY_POLICIES", "POLICIES", "i14", "i9",
           "validate_parent_snapshot", "history_step"]
