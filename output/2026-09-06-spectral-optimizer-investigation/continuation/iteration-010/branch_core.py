"""Narrow I10 branch-step and evaluation adapter over the frozen I9 core."""
from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F


_I9_PATH = Path(__file__).resolve().parent.parent / "iteration-009" / "neural_core.py"
_I9_SPEC = importlib.util.spec_from_file_location("_i10_frozen_i9_neural_core", _I9_PATH)
if _I9_SPEC is None or _I9_SPEC.loader is None:  # pragma: no cover - import machinery failure
    raise ImportError("cannot load frozen I9 neural core")
i9 = importlib.util.module_from_spec(_I9_SPEC)
_I9_SPEC.loader.exec_module(i9)


POLICIES = ("raw", "current32", "frozen32")
DATA_KEYS = i9.DATA_KEYS


class BranchCoreError(ValueError):
    pass


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise BranchCoreError(message)


def _finite(value: torch.Tensor, label: str) -> torch.Tensor:
    _need(type(value) is torch.Tensor and value.layout == torch.strided
          and bool(torch.isfinite(value).all()), label + " must be a finite dense tensor")
    return value


def _basis(value: torch.Tensor, parameter_count: int, device: torch.device,
           dtype: torch.dtype) -> torch.Tensor:
    value = _finite(value, "frozen basis")
    _need(value.ndim == 2 and value.shape[0] == parameter_count
          and 0 < value.shape[1] <= 32, "frozen basis shape invalid")
    _need(value.device == device and value.dtype == dtype,
          "frozen basis device or dtype differs")
    return value.detach()


def _optimizer_is_i9(optimizer: torch.optim.Optimizer, model: torch.nn.Module) -> None:
    _need(type(optimizer) is torch.optim.AdamW, "optimizer must be exact AdamW")
    expected = list(model.parameters())
    seen = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    _need(len(seen) == len(expected) and all(left is right for left, right in zip(seen, expected)),
          "optimizer parameters/order differ")
    for group in optimizer.param_groups:
        _need(group["lr"] == i9.LR and group["weight_decay"] == i9.WD
              and tuple(group["betas"]) == (0.9, 0.999) and group["eps"] == 1e-8
              and group["foreach"] is False and group["fused"] is False,
              "optimizer configuration differs from I9")


def _leakage(value: torch.Tensor, basis: torch.Tensor | None) -> dict[str, Any]:
    return i9._leakage(value, basis)


def branch_step(model: torch.nn.Module, optimizer: torch.optim.Optimizer, tracker: Any,
                x: torch.Tensor, target: torch.Tensor, policy: str,
                frozen_basis: torch.Tensor) -> dict[str, Any]:
    """Execute one branch update and return a bounded JSON diagnostic."""
    _need(type(policy) is str and policy in POLICIES, "unknown branch policy")
    _optimizer_is_i9(optimizer, model)
    before = i9.flat_params(model)
    basis = _basis(frozen_basis, before.numel(), before.device, before.dtype)
    observer_before = int(tracker.step_count)

    if policy in ("raw", "current32"):
        inherited = i9.train_step(model, optimizer, tracker, x, target, policy)
        raw_norm = float(inherited["raw_norm"])
        applied = i9.flat_grad(model)
        applied_norm = i9._norm(applied)
        loss = float(inherited["loss"])
        observer_after = int(tracker.step_count)
        filtering_active = bool(inherited["filtering_active"])
        gradient_filter_applied = policy == "current32" and filtering_active
        _need(observer_after == observer_before + 1, "observer did not advance once")
    else:
        tracker_basis = i9._basis(tracker)
        _need(tracker_basis is not None and torch.equal(tracker_basis, basis),
              "frozen basis differs from the unused parent observer basis")
        optimizer.zero_grad(set_to_none=True)
        objective = i9._objective_loss(model, x, target)
        objective.backward()
        raw = i9.flat_grad(model)
        applied = basis @ (basis.T @ raw)
        _finite(applied, "frozen projected gradient")
        i9.set_grad(model, applied)
        optimizer.step()
        raw_norm, applied_norm = i9._norm(raw), i9._norm(applied)
        loss = float(objective.detach().item())
        observer_after = int(tracker.step_count)
        filtering_active = False
        gradient_filter_applied = True
        _need(observer_after == observer_before, "frozen policy changed observer")

    after = i9.flat_params(model)
    total_delta = _finite(after - before, "total displacement")
    # Preserve the I9 convention: remove nominal decoupled AdamW decay.
    data_delta = _finite(total_delta + i9.LR * i9.WD * before,
                         "decay-adjusted data displacement")
    current_basis = i9._basis(tracker)
    row = {
        "schema": "i10_branch_step_v1", "policy": policy, "loss": loss,
        "observer": {"used": policy != "frozen32", "step_before": observer_before,
                     "step_after": observer_after, "filtering_active": filtering_active},
        "gradient_filter_applied": gradient_filter_applied,
        "gradient": {"raw_norm": raw_norm, "applied_norm": applied_norm},
        "displacement": {
            "total_norm": i9._norm(total_delta), "data_norm": i9._norm(data_delta),
            "total_leakage": {"frozen_basis": _leakage(total_delta, basis),
                              "current_basis": _leakage(total_delta, current_basis)},
            "data_leakage": {"frozen_basis": _leakage(data_delta, basis),
                             "current_basis": _leakage(data_delta, current_basis)},
        },
    }
    json.dumps(row, ensure_ascii=True, allow_nan=False, separators=(",", ":"))
    return row


def _dataset(data: dict[str, torch.Tensor], x_name: str, y_name: str,
             chunk_size: int) -> dict[str, float]:
    x, labels = data[x_name], data[y_name]
    _need(type(x) is torch.Tensor and type(labels) is torch.Tensor
          and x.ndim >= 2 and labels.dtype == torch.long and labels.ndim == 1
          and len(x) == len(labels) and len(x) > 0, x_name + " dataset invalid")
    _need(x.device == labels.device, x_name + " device differs")
    total, clean_loss, soft_loss, correct = 0, 0.0, 0.0, 0
    max_probability, true_probability = 0.0, 0.0
    for start in range(0, len(x), chunk_size):
        stop = min(start + chunk_size, len(x))
        logits = data["_model"](x[start:stop])
        _finite(logits, "evaluation logits")
        _need(logits.dtype == torch.float32, "evaluation logits must be float32")
        logits64 = logits.to(torch.float64)
        chunk_labels = labels[start:stop]
        _need(logits64.ndim == 2 and logits64.shape[0] == len(chunk_labels)
              and logits64.shape[1] >= 2 and bool((chunk_labels >= 0).all())
              and bool((chunk_labels < logits64.shape[1]).all()), "evaluation labels invalid")
        log_prob = F.log_softmax(logits64, dim=-1)
        rows = torch.arange(len(chunk_labels), device=logits.device)
        nll = -log_prob[rows, chunk_labels]
        uniform = -log_prob.mean(dim=-1)
        probability = log_prob.exp()
        clean_loss += float(nll.sum().item())
        soft_loss += float((0.1 * nll + 0.9 * uniform).sum().item())
        correct += int((logits.argmax(dim=-1) == chunk_labels).sum().item())
        max_probability += float(probability.max(dim=-1).values.sum().item())
        true_probability += float(probability[rows, chunk_labels].sum().item())
        total += len(chunk_labels)
    values = {
        "clean_ce": clean_loss / total, "soft_ce": soft_loss / total,
        "clean_accuracy": correct / total,
        "mean_max_probability": max_probability / total,
        "mean_true_label_probability": true_probability / total,
    }
    _need(all(math.isfinite(value) for value in values.values()), "nonfinite evaluation summary")
    return values


def evaluate(model: torch.nn.Module, data: dict[str, torch.Tensor], *,
             chunk_size: int = 512) -> dict[str, Any]:
    """Evaluate full train/aux/validation tensors without changing model, grads, modes or RNG."""
    _need(type(data) is dict and tuple(data) == DATA_KEYS, "evaluation data keys/order differ")
    _need(type(chunk_size) is int and chunk_size > 0, "invalid evaluation chunk size")
    device = next(model.parameters()).device
    _need(all(type(value) is torch.Tensor and value.device == device for value in data.values()),
          "evaluation tensors must share model device")
    modes = tuple(module.training for module in model.modules())
    rng = i9._rng_state()
    gradients = tuple(None if parameter.grad is None else parameter.grad.detach().clone()
                      for parameter in model.parameters())
    parameters = i9.flat_params(model)
    buffers = tuple(buffer.detach().clone() for buffer in model.buffers())
    try:
        model.eval()
        view = dict(data)
        view["_model"] = model
        with torch.no_grad():
            train_clean = _dataset(view, "x", "clean", chunk_size)
            train_fixed = _dataset(view, "x", "noisy", chunk_size)
            auxiliary = _dataset(view, "ax", "ay", chunk_size)
            validation = _dataset(view, "vx", "vy", chunk_size)
        row = {
            "schema": "i10_branch_evaluation_v1",
            "train": {
                "clean_ce": train_clean["clean_ce"],
                "fixed_ce": train_fixed["clean_ce"],
                "soft_ce": train_clean["soft_ce"],
                "clean_accuracy": train_clean["clean_accuracy"],
                "fixed_accuracy": train_fixed["clean_accuracy"],
                "fixed_minus_soft_ce": train_fixed["clean_ce"] - train_clean["soft_ce"],
                "mean_max_probability": train_clean["mean_max_probability"],
                "mean_true_label_probability": train_clean["mean_true_label_probability"],
            },
            "auxiliary": {key: auxiliary[key] for key in
                          ("clean_ce", "soft_ce", "clean_accuracy",
                           "mean_max_probability", "mean_true_label_probability")},
            "validation": {key: validation[key] for key in
                           ("clean_ce", "soft_ce", "clean_accuracy",
                            "mean_max_probability", "mean_true_label_probability")},
        }
        _need(all(math.isfinite(value) for group in row.values() if type(group) is dict
                  for value in group.values() if type(value) in (int, float)),
              "nonfinite evaluation record")
        json.dumps(row, ensure_ascii=True, allow_nan=False, separators=(",", ":"))
        return row
    finally:
        for module, training in zip(model.modules(), modes):
            module.training = training
        i9._restore_rng(rng)
        _need(torch.equal(parameters, i9.flat_params(model)), "evaluation changed parameters")
        current_buffers = tuple(model.buffers())
        _need(len(buffers) == len(current_buffers)
              and all(torch.equal(before, after) for before, after in zip(buffers, current_buffers)),
              "evaluation changed buffers")
        for parameter, gradient in zip(model.parameters(), gradients):
            _need((gradient is None and parameter.grad is None)
                  or (gradient is not None and parameter.grad is not None
                      and torch.equal(gradient, parameter.grad)), "evaluation changed gradients")


def main() -> int:
    print("branch_core: inert library; no branch or evaluation executed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
