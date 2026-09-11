"""I14 cross-optimizer adapter with one canonical spectral observer."""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys
from typing import Any

import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
I9 = HERE.parent / "iteration-009"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(I9))
import neural_core as i9


BASES = ("sgd", "sgdm", "adamw")
POLICIES = ("raw", "current32")
WD = 0.01


class OptimizerCoreError(ValueError):
    """Structural, configuration or provenance error."""


class NumericalFailure(FloatingPointError):
    """Explicitly detected nonfinite branch arithmetic."""


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise OptimizerCoreError(message)


def _rate(lr: Any) -> float:
    _need(type(lr) is float and math.isfinite(lr) and lr > 0 and lr * WD < 1,
          "learning rate must be a finite positive float with lr*WD<1")
    return lr


def make_optimizer(model: torch.nn.Module, base: str,
                   lr: float) -> torch.optim.Optimizer:
    """Construct one exact I14 base optimizer at an externally chosen rate."""
    _need(type(base) is str and base in BASES, "unknown optimizer base")
    _rate(lr)
    parameters = list(model.parameters())
    _need(parameters and all(parameter.requires_grad for parameter in parameters),
          "model must have trainable parameters")
    if base in ("sgd", "sgdm"):
        return torch.optim.SGD(
            parameters, lr=lr, momentum=0.0 if base == "sgd" else 0.9,
            dampening=0.0, weight_decay=0.0, nesterov=False, maximize=False,
            foreach=False, differentiable=False, fused=False)
    return torch.optim.AdamW(
        parameters, lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=WD,
        amsgrad=False, maximize=False, foreach=False, capturable=False,
        differentiable=False, fused=False)


def _parameter_topology(model: torch.nn.Module) -> list[dict[str, Any]]:
    return [{"name": name, "shape": list(parameter.shape), "dtype": str(parameter.dtype),
             "requires_grad": bool(parameter.requires_grad)}
            for name, parameter in model.named_parameters()]


def _validate_optimizer(model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                        base: str, lr: float) -> None:
    _need(type(base) is str and base in BASES, "unknown optimizer base")
    _rate(lr)
    expected_parameters = list(model.parameters())
    _need(len(optimizer.param_groups) == 1, "optimizer must have one parameter group")
    group = optimizer.param_groups[0]
    _need(len(group["params"]) == len(expected_parameters)
          and all(left is right for left, right in zip(group["params"], expected_parameters)),
          "optimizer parameter identity/order differs")
    if base in ("sgd", "sgdm"):
        _need(type(optimizer) is torch.optim.SGD
              and group["lr"] == lr and group["momentum"] == (0.0 if base == "sgd" else 0.9)
              and group["dampening"] == 0.0 and group["weight_decay"] == 0.0
              and group["nesterov"] is False and group["maximize"] is False
              and group["foreach"] is False and group["differentiable"] is False
              and group["fused"] is False, "SGD optimizer configuration differs")
    else:
        _need(type(optimizer) is torch.optim.AdamW and group["lr"] == lr
              and tuple(group["betas"]) == (0.9, 0.999) and group["eps"] == 1e-8
              and group["weight_decay"] == WD and group["amsgrad"] is False
              and group["maximize"] is False and group["foreach"] is False
              and group["capturable"] is False and group["differentiable"] is False
              and group["fused"] is False
              and group.get("decoupled_weight_decay", True) is True,
              "AdamW optimizer configuration differs")
    states = optimizer.state
    _need(set(states).issubset(set(expected_parameters)),
          "optimizer state contains an unknown parameter")
    if base == "sgd":
        _need(not states, "plain SGD must have no optimizer state")
    elif states:
        _need(set(states) == set(expected_parameters),
              "initialized optimizer state must cover every parameter")
        fields = ("momentum_buffer",) if base == "sgdm" else ("step", "exp_avg", "exp_avg_sq")
        for parameter in expected_parameters:
            row = states[parameter]
            _need(tuple(row) == fields, "optimizer state fields/order differ")
            for name, value in row.items():
                _need(type(value) is torch.Tensor and value.layout == torch.strided
                      and not value.is_sparse, "optimizer state tensor topology differs")
                expected_shape = () if name == "step" else tuple(parameter.shape)
                _need(tuple(value.shape) == expected_shape,
                      "optimizer state tensor shape differs")
                if name != "step":
                    _need(value.dtype == parameter.dtype and value.device == parameter.device,
                          "optimizer state tensor dtype/device differs")


def _validate_tracker(model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                      tracker: Any) -> None:
    _need(tracker is not None and tracker.model is model
          and tracker.base_optimizer is optimizer
          and len(tracker.param_list) == len(tuple(model.parameters()))
          and all(left is right for left, right in zip(tracker.param_list, model.parameters())),
          "tracker model/optimizer/parameter binding differs")
    expected = {"rank": 32, "decay": 0.99, "warmup": 100,
                "filter_strength": 1.0, "weighting": "hard", "alpha": 1.0,
                "soft_residual": True, "energy_threshold": None,
                "normalize": "none", "adaptive": "none", "stable_update": True,
                "relative_eig_tol": 1e-8, "absolute_eig_floor": 0.0,
                "stabilize_every": 100}
    _need(all(getattr(tracker, name) == value for name, value in expected.items()),
          "tracker configuration differs from canonical I14 observer")
    _need(tracker.n_params == sum(parameter.numel() for parameter in model.parameters()),
          "tracker parameter count differs")


def _finite_tensor(value: Any, label: str) -> torch.Tensor:
    _need(isinstance(value, torch.Tensor) and value.layout == torch.strided and not value.is_sparse,
          label + " must be a dense tensor")
    if not bool(torch.isfinite(value).all()):
        raise NumericalFailure(label + " contains a nonfinite value")
    return value


def _finite_scalar(value: Any, label: str) -> float:
    _need(type(value) in (int, float), label + " must be a scalar")
    result = float(value)
    if not math.isfinite(result):
        raise NumericalFailure(label + " is nonfinite")
    return result


def _flat_grad(model: torch.nn.Module) -> torch.Tensor:
    values = []
    for name, parameter in model.named_parameters():
        _need(parameter.grad is not None, "missing gradient for " + name)
        values.append(_finite_tensor(parameter.grad, "gradient." + name).detach().reshape(-1))
    _need(bool(values), "model has no gradients")
    return _finite_tensor(torch.cat(values).clone(), "flat gradient")


def _scan_finite(value: Any, label: str) -> None:
    if isinstance(value, torch.Tensor):
        _finite_tensor(value, label)
    elif type(value) is dict:
        for key, item in value.items():
            _scan_finite(item, f"{label}.{key}")
    elif type(value) in (list, tuple):
        for index, item in enumerate(value):
            _scan_finite(item, f"{label}[{index}]")
    elif type(value) is float and not math.isfinite(value):
        raise NumericalFailure(label + " is nonfinite")


def _finite_live(model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                 tracker: Any, label: str) -> None:
    for name, parameter in model.named_parameters():
        _finite_tensor(parameter, f"{label}.parameter.{name}")
        if parameter.grad is not None:
            _finite_tensor(parameter.grad, f"{label}.gradient.{name}")
    _scan_finite(optimizer.state_dict(), label + ".optimizer")
    tracker_values = {key: value for key, value in vars(tracker).items()
                      if key not in ("model", "base_optimizer", "param_list")}
    _scan_finite(tracker_values, label + ".tracker")


def _norm(value: torch.Tensor, label: str) -> float:
    result = float(torch.linalg.vector_norm(value.to(torch.float64)).item())
    if not math.isfinite(result):
        raise NumericalFailure(label + " norm is nonfinite")
    return result


def _dot(left: torch.Tensor, right: torch.Tensor, label: str) -> float:
    result = float(torch.dot(left.to(torch.float64), right.to(torch.float64)).item())
    if not math.isfinite(result):
        raise NumericalFailure(label + " dot product is nonfinite")
    return result


def _leakage(value: torch.Tensor, basis: torch.Tensor | None,
             label: str) -> dict[str, Any]:
    squared = _dot(value, value, label)
    if basis is None:
        return {"squared_norm": squared, "outside_squared_norm": None,
                "fraction": None, "reason": "basis_unavailable"}
    _finite_tensor(basis, label + " basis")
    _need(basis.ndim == 2 and basis.shape[0] == value.numel(),
          label + " basis shape differs")
    if squared == 0:
        return {"squared_norm": 0.0, "outside_squared_norm": 0.0,
                "fraction": None, "reason": "zero_displacement"}
    outside = value - basis @ (basis.T @ value)
    outside_squared = _dot(outside, outside, label + " outside")
    return {"squared_norm": squared, "outside_squared_norm": outside_squared,
            "fraction": outside_squared / squared, "reason": None}


@torch.no_grad()
def _manual_decay(model: torch.nn.Module, factor: float) -> None:
    for parameter in model.parameters():
        parameter.mul_(factor)


def train_step(model: torch.nn.Module, optimizer: torch.optim.Optimizer,
               tracker: Any, x: torch.Tensor, target: torch.Tensor,
               policy: str, base: str) -> dict[str, Any]:
    """Observe one raw gradient, deliver raw/native, decay, and take one base step."""
    _need(type(policy) is str and policy in POLICIES, "unknown filter policy")
    saved_lr = optimizer.param_groups[0].get("lr")
    _need(type(saved_lr) is float, "optimizer learning rate must be an exact float")
    lr = saved_lr
    _validate_optimizer(model, optimizer, base, lr)
    _validate_tracker(model, optimizer, tracker)
    _finite_live(model, optimizer, tracker, "pre_step")
    before = i9.flat_params(model)
    observer_before = int(tracker.step_count)

    optimizer.zero_grad(set_to_none=True)
    loss = i9._objective_loss(model, x, target)
    _need(type(loss) is torch.Tensor and loss.ndim == 0, "objective loss must be a scalar tensor")
    if not bool(torch.isfinite(loss)):
        raise NumericalFailure("objective loss is nonfinite")
    loss.backward()
    raw = _flat_grad(model)
    observer_diagnostic = tracker.filter_grad()
    applied_native = _flat_grad(model)
    if policy == "raw":
        i9.set_grad(model, raw)
    applied = _flat_grad(model)
    observer_after = int(tracker.step_count)
    _need(observer_after == observer_before + 1
          and int(observer_diagnostic["step"]) == observer_after,
          "observer did not advance exactly once")
    filtering_active = bool(observer_diagnostic["filtering_active"])

    nominal_decay = -lr * WD * before
    manual_decay = None
    if base in ("sgd", "sgdm"):
        _manual_decay(model, 1.0 - lr * WD)
        after_decay = i9.flat_params(model)
        manual_decay = after_decay - before
        _finite_tensor(manual_decay, "manual decay displacement")
    optimizer.step()
    after = i9.flat_params(model)
    _finite_tensor(after, "updated parameters")
    _finite_live(model, optimizer, tracker, "post_step")
    total_delta = _finite_tensor(after - before, "total displacement")
    data_delta = _finite_tensor(total_delta - nominal_decay,
                                "nominal-decay-adjusted data displacement")
    basis = i9._basis(tracker)
    manual_error = None if manual_decay is None else _norm(
        manual_decay - nominal_decay, "manual decay realization error")
    record = {
        "schema": "i14_optimizer_step_v1", "base": base, "lr": lr,
        "policy": policy, "loss": float(loss.detach().item()),
        "observer": {"step_before": observer_before, "step_after": observer_after,
                     "filtering_active": filtering_active, "used": True},
        "gradient_filter_applied": policy == "current32" and filtering_active,
        "gradient": {"raw_norm": _norm(raw, "raw gradient"),
                     "applied_norm": _norm(applied, "applied gradient"),
                     "native_norm": _norm(applied_native, "native gradient")},
        "displacement": {
            "total_norm": _norm(total_delta, "total displacement"),
            "data_norm": _norm(data_delta, "data displacement"),
            "nominal_decay_norm": _norm(nominal_decay, "nominal decay"),
            "total_current_basis_leakage": _leakage(total_delta, basis, "total leakage"),
            "data_current_basis_leakage": _leakage(data_delta, basis, "data leakage"),
            "raw_gradient_dot_data_delta": _dot(raw, data_delta, "raw/data displacement"),
        },
        "decay": {"coefficient": WD, "factor": 1.0 - lr * WD,
                  "manual_before_optimizer_step": base in ("sgd", "sgdm"),
                  "manual_actual_norm": (None if manual_decay is None else
                                           _norm(manual_decay, "manual decay")),
                  "manual_minus_nominal_norm": manual_error},
    }
    _scan_finite(record, "step diagnostic")
    json.dumps(record, ensure_ascii=True, allow_nan=False, separators=(",", ":"))
    return record


def snapshot(model: torch.nn.Module, optimizer: torch.optim.Optimizer,
             tracker: Any, base: str, lr: float) -> dict[str, Any]:
    """Capture complete resumable state; nonfinite tensors remain serializable evidence."""
    _validate_optimizer(model, optimizer, base, lr)
    _validate_tracker(model, optimizer, tracker)
    spec = getattr(model, "_i9_spec", None)
    _need(type(spec) is dict and tuple(spec) == ("input_dim", "width", "classes"),
          "model was not made by I9 neural_core")
    return {
        "schema": "i14_optimizer_snapshot_v1", "base": base, "lr": lr,
        "model_spec": dict(spec), "parameter_topology": _parameter_topology(model),
        "model_state": {key: value.detach().cpu().clone()
                        for key, value in model.state_dict().items()},
        "gradients": [None if parameter.grad is None else parameter.grad.detach().cpu().clone()
                      for parameter in model.parameters()],
        "model_modes": [(name, bool(module.training)) for name, module in model.named_modules()],
        "optimizer": i9._cpu_clone(optimizer.state_dict()),
        "tracker": i9._tracker_state(tracker), "rng": i9._rng_state(),
    }


def _validate_saved_optimizer_topology(value: dict[str, Any],
                                       expected_identifiers: list[Any]) -> None:
    _need(type(value) is dict and tuple(value) == ("state", "param_groups")
          and type(value["state"]) is dict and type(value["param_groups"]) is list
          and len(value["param_groups"]) == 1, "saved optimizer topology differs")
    identifiers = value["param_groups"][0].get("params")
    _need(type(identifiers) is list and identifiers == expected_identifiers
          and len(set(identifiers)) == len(expected_identifiers)
          and set(value["state"]).issubset(set(identifiers)),
          "saved optimizer parameter IDs/order differ")


def restore(value: dict[str, Any], device: str | torch.device):
    """Restore and validate exact model/optimizer/tracker/RNG topology."""
    keys = ("schema", "base", "lr", "model_spec", "parameter_topology", "model_state",
            "gradients", "model_modes", "optimizer", "tracker", "rng")
    _need(type(value) is dict and tuple(value) == keys
          and value["schema"] == "i14_optimizer_snapshot_v1",
          "invalid I14 optimizer snapshot")
    base, lr = value["base"], value["lr"]
    _need(type(base) is str and base in BASES, "snapshot optimizer base differs")
    _rate(lr)
    target = torch.device(device)
    spec = value["model_spec"]
    _need(type(spec) is dict and tuple(spec) == ("input_dim", "width", "classes"),
          "snapshot model specification differs")
    model = i9.make_model(0, target, **spec)
    _need(value["parameter_topology"] == _parameter_topology(model),
          "snapshot parameter topology differs")
    model.load_state_dict(i9._cpu_clone(value["model_state"]), strict=True)
    modes = value["model_modes"]
    _need(type(modes) is list and [name for name, _ in modes]
          == [name for name, _ in model.named_modules()], "model mode topology differs")
    for (name, module), (_, training) in zip(model.named_modules(), modes):
        _need(type(training) is bool, "invalid model mode")
        module.train(training)
    gradients = value["gradients"]
    parameters = list(model.parameters())
    _need(type(gradients) is list and len(gradients) == len(parameters),
          "gradient topology differs")
    for parameter, gradient in zip(parameters, gradients):
        if gradient is None:
            parameter.grad = None
        else:
            _need(type(gradient) is torch.Tensor and gradient.layout == torch.strided
                  and tuple(gradient.shape) == tuple(parameter.shape)
                  and gradient.dtype == parameter.dtype,
                  "saved gradient tensor topology differs")
            parameter.grad = gradient.to(target).clone()
    optimizer = make_optimizer(model, base, lr)
    expected_identifiers = optimizer.state_dict()["param_groups"][0]["params"]
    _validate_saved_optimizer_topology(value["optimizer"], expected_identifiers)
    optimizer.load_state_dict(i9._cpu_clone(value["optimizer"]))
    _validate_optimizer(model, optimizer, base, lr)
    tracker_value = value["tracker"]
    _need(type(tracker_value) is dict, "snapshot tracker state is missing")
    tracker = i9.make_tracker(model, optimizer)
    expected_tracker = {key for key in vars(tracker)
                        if key not in ("model", "base_optimizer", "param_list")}
    _need(set(tracker_value) == expected_tracker, "tracker attribute topology differs")
    for key, item in tracker_value.items():
        setattr(tracker, key, i9._move_tracker_value(key, item, target))
    tracker.model, tracker.base_optimizer = model, optimizer
    tracker.param_list = list(model.parameters())
    _validate_tracker(model, optimizer, tracker)
    i9._restore_rng(value["rng"])
    return model, optimizer, tracker


def main() -> int:
    print("optimizer_core: inert library; no training or data access")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
