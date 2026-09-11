"""Compact I9 neural mechanism core; import has no execution side effects."""
from __future__ import annotations

import hashlib
import json
import math
import random
import copy
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from spectral_filter import SpectralGradientFilter


LR, WD = 1e-3, 1e-2
MAX_RESCALE = 1_000_000.0
FILTER_SHA256 = "9280c7d360aef4c775baf0d781a5afd56e016e9cfb93ef829c3ae4b8cc3df943"
DATA_KEYS = ("x", "clean", "noisy", "vx", "vy", "ax", "ay")
UTILITY_NAMES = ("train_clean", "train_soft", "train_noisy", "aux_clean", "aux_soft")
INTERVENTION_NAMES = (
    "raw_gradient", "native_gradient", "raw_gradient_to_native_norm",
    "native_gradient_to_raw_norm", "raw_data_delta_to_native_norm",
    "native_data_delta_to_raw_norm",
)


class NeuralCoreError(ValueError):
    pass


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise NeuralCoreError(message)


def _finite_tensor(value: torch.Tensor, label: str) -> torch.Tensor:
    _need(type(value) is torch.Tensor and value.layout == torch.strided
          and not value.is_sparse and bool(torch.isfinite(value).all()),
          label + " must be a finite dense tensor")
    return value


def _cpu_clone(value: Any) -> Any:
    if type(value) is torch.Tensor:
        return value.detach().cpu().clone()
    if type(value) is dict:
        return {_cpu_clone(k): _cpu_clone(v) for k, v in value.items()}
    if type(value) is list:
        return [_cpu_clone(v) for v in value]
    if type(value) is tuple:
        return tuple(_cpu_clone(v) for v in value)
    if value is None or type(value) in (bool, int, float, str):
        return value
    raise NeuralCoreError("snapshot contains an unsupported value")


def equal_tree(left: Any, right: Any) -> bool:
    if type(left) is torch.Tensor:
        return (type(right) is torch.Tensor and left.dtype == right.dtype
                and tuple(left.shape) == tuple(right.shape) and torch.equal(left, right))
    if type(left) is not type(right):
        return False
    if type(left) is dict:
        return (tuple(left) == tuple(right)
                and all(equal_tree(left[k], right[k]) for k in left))
    if type(left) in (list, tuple):
        return len(left) == len(right) and all(equal_tree(x, y) for x, y in zip(left, right))
    return left == right


def _digest_node(value: Any) -> Any:
    if type(value) is torch.Tensor:
        tensor = value.detach().cpu().contiguous()
        raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
        return ["tensor", str(tensor.dtype), list(tensor.shape), hashlib.sha256(raw).hexdigest()]
    if type(value) is dict:
        return ["dict", [[_digest_node(k), _digest_node(v)] for k, v in value.items()]]
    if type(value) is list:
        return ["list", [_digest_node(v) for v in value]]
    if type(value) is tuple:
        return ["tuple", [_digest_node(v) for v in value]]
    if value is None or type(value) in (bool, int, str):
        return [type(value).__name__, value]
    if type(value) is float:
        _need(math.isfinite(value), "nonfinite tree float")
        return ["float", value.hex()]
    raise NeuralCoreError("unsupported digest value")


def tree_digest(value: Any) -> str:
    raw = json.dumps(_digest_node(value), ensure_ascii=True, allow_nan=False,
                     separators=(",", ":")).encode("ascii")
    return hashlib.sha256(b"i9_neural_tree_v1\n" + raw).hexdigest()


def make_model(seed: int, device: str | torch.device, input_dim: int = 784,
               width: int = 64, classes: int = 10) -> nn.Module:
    _need(type(seed) is int and 0 <= seed < 2**63, "invalid model seed")
    _need(all(type(v) is int and v > 0 for v in (input_dim, width, classes))
          and classes >= 2, "invalid model dimensions")
    # Initialization is deliberately CPU-only and does not advance caller RNG.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        model = nn.Sequential(nn.Linear(input_dim, width), nn.ReLU(), nn.Linear(width, classes))
    model._i9_spec = {"input_dim": input_dim, "width": width, "classes": classes}
    return model.to(torch.device(device))


def make_optimizer(model: nn.Module) -> torch.optim.AdamW:
    return torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD,
                             betas=(0.9, 0.999), eps=1e-8,
                             foreach=False, fused=False)


def make_tracker(model: nn.Module, optimizer: torch.optim.Optimizer) -> SpectralGradientFilter:
    return SpectralGradientFilter(
        model, optimizer, rank=32, decay=0.99, warmup=100,
        stable_update=True, stabilize_every=100, relative_eig_tol=1e-8,
        absolute_eig_floor=0.0, weighting="hard", normalize="none", adaptive="none")


def flat_params(model: nn.Module) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()]).clone()


@torch.no_grad()
def set_params(model: nn.Module, value: torch.Tensor) -> None:
    value = _finite_tensor(value, "parameter vector")
    position = 0
    for parameter in model.parameters():
        size = parameter.numel()
        _need(value.device == parameter.device, "parameter vector device differs")
        parameter.copy_(value[position:position + size].reshape_as(parameter))
        position += size
    _need(position == value.numel(), "parameter vector length differs")


def flat_grad(model: nn.Module) -> torch.Tensor:
    parameters = tuple(model.parameters())
    _need(parameters and all(p.grad is not None for p in parameters), "missing gradient")
    value = torch.cat([p.grad.detach().reshape(-1) for p in parameters]).clone()
    return _finite_tensor(value, "gradient")


def set_grad(model: nn.Module, value: torch.Tensor) -> None:
    value = _finite_tensor(value, "gradient vector")
    position = 0
    for parameter in model.parameters():
        size = parameter.numel()
        _need(value.device == parameter.device, "gradient vector device differs")
        parameter.grad = value[position:position + size].reshape_as(parameter).clone()
        position += size
    _need(position == value.numel(), "gradient vector length differs")


def _rng_state() -> dict[str, Any]:
    numpy_state = np.random.get_state()
    cuda = [state.detach().cpu().clone() for state in torch.cuda.get_rng_state_all()] \
        if torch.cuda.is_available() else []
    return {
        "python": _cpu_clone(random.getstate()),
        "numpy": {"algorithm": numpy_state[0],
                  "state": torch.tensor(numpy_state[1].tolist(), dtype=torch.uint32),
                  "position": int(numpy_state[2]), "has_gauss": int(numpy_state[3]),
                  "cached_gaussian": float(numpy_state[4])},
        "torch_cpu": torch.get_rng_state().detach().cpu().clone(),
        "torch_cuda": cuda,
    }


def _restore_rng(value: dict[str, Any]) -> None:
    _need(type(value) is dict and tuple(value) == ("python", "numpy", "torch_cpu", "torch_cuda"),
          "invalid RNG snapshot")
    random.setstate(value["python"])
    row = value["numpy"]
    _need(type(row) is dict and tuple(row) ==
          ("algorithm", "state", "position", "has_gauss", "cached_gaussian"),
          "invalid NumPy RNG snapshot")
    np.random.set_state((row["algorithm"],
                         row["state"].detach().cpu().numpy().astype(np.uint32, copy=True),
                         row["position"], row["has_gauss"], row["cached_gaussian"]))
    torch.set_rng_state(value["torch_cpu"].detach().cpu())
    cuda_states = value["torch_cuda"]
    if cuda_states:
        _need(torch.cuda.is_available() and len(cuda_states) == torch.cuda.device_count(),
              "CUDA RNG topology differs")
        torch.cuda.set_rng_state_all([state.detach().cpu() for state in cuda_states])


def _tracker_state(tracker: SpectralGradientFilter | None) -> Any:
    if tracker is None:
        return None
    aliases = {"model", "base_optimizer", "param_list"}
    return {key: _cpu_clone(value) for key, value in vars(tracker).items() if key not in aliases}


def snapshot(model: nn.Module, optimizer: torch.optim.Optimizer,
             tracker: SpectralGradientFilter | None) -> dict[str, Any]:
    spec = getattr(model, "_i9_spec", None)
    _need(type(spec) is dict and tuple(spec) == ("input_dim", "width", "classes"),
          "model was not made by neural_core")
    return {
        "schema": "i9_neural_snapshot_v1",
        "model_spec": dict(spec),
        "model_state": {key: value.detach().cpu().clone()
                        for key, value in model.state_dict().items()},
        "gradients": [None if p.grad is None else p.grad.detach().cpu().clone()
                      for p in model.parameters()],
        "model_modes": [(name, bool(module.training)) for name, module in model.named_modules()],
        "optimizer": _cpu_clone(optimizer.state_dict()),
        "tracker": _tracker_state(tracker),
        "rng": _rng_state(),
    }


def _move_tracker_value(key: str, value: Any, device: torch.device) -> Any:
    value = _cpu_clone(value)
    if type(value) is torch.Tensor and key in ("V", "grad_mean"):
        return value.to(device)
    return value


def restore(value: dict[str, Any], device: str | torch.device):
    _need(type(value) is dict and tuple(value) ==
          ("schema", "model_spec", "model_state", "gradients", "model_modes",
           "optimizer", "tracker", "rng") and value["schema"] == "i9_neural_snapshot_v1",
          "invalid neural snapshot")
    target = torch.device(device)
    spec = value["model_spec"]
    model = make_model(0, target, **spec)
    model.load_state_dict(_cpu_clone(value["model_state"]), strict=True)
    modes = value["model_modes"]
    _need(type(modes) is list and [name for name, _ in modes] ==
          [name for name, _ in model.named_modules()], "model mode topology differs")
    for (name, module), (_, training) in zip(model.named_modules(), modes):
        _need(type(training) is bool, "invalid model mode")
        module.train(training)
    gradients = value["gradients"]
    _need(type(gradients) is list and len(gradients) == len(tuple(model.parameters())),
          "gradient topology differs")
    for parameter, gradient_value in zip(model.parameters(), gradients):
        parameter.grad = None if gradient_value is None else gradient_value.to(target).clone()
    optimizer = make_optimizer(model)
    optimizer.load_state_dict(_cpu_clone(value["optimizer"]))
    tracker_value = value["tracker"]
    tracker = None if tracker_value is None else make_tracker(model, optimizer)
    if tracker is not None:
        _need(type(tracker_value) is dict, "invalid tracker snapshot")
        expected = {key for key in vars(tracker) if key not in ("model", "base_optimizer", "param_list")}
        _need(set(tracker_value) == expected, "tracker attribute topology differs")
        for key, item in tracker_value.items():
            setattr(tracker, key, _move_tracker_value(key, item, target))
        tracker.model, tracker.base_optimizer = model, optimizer
        tracker.param_list = list(model.parameters())
    _restore_rng(value["rng"])
    return model, optimizer, tracker


def _objective_loss(model: nn.Module, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    logits = model(x)
    if target.dtype.is_floating_point:
        _need(target.shape == logits.shape and bool(torch.isfinite(target).all())
              and bool((target >= 0).all()), "invalid soft target")
        sums = target.to(torch.float64).sum(dim=-1)
        _need(bool(torch.allclose(sums, torch.ones_like(sums), atol=1e-12, rtol=1e-12)),
              "soft target rows must sum to one")
        return -(target.to(logits) * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
    _need(target.dtype == torch.long and target.ndim == 1 and len(target) == len(logits),
          "invalid hard target")
    return F.cross_entropy(logits, target, reduction="mean")


def gradient(model: nn.Module, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Return a gradient without accumulating `.grad` or advancing any RNG."""
    before_rng = _rng_state()
    try:
        clone = copy.deepcopy(model)
        loss = _objective_loss(clone, x, target)
        values = torch.autograd.grad(loss, tuple(clone.parameters()), create_graph=False,
                                     retain_graph=False, allow_unused=False)
        result = torch.cat([item.detach().reshape(-1) for item in values]).clone()
        return _finite_tensor(result, "probe gradient")
    finally:
        _restore_rng(before_rng)


def _soft(labels: torch.Tensor, classes: int) -> torch.Tensor:
    _need(labels.dtype == torch.long and labels.ndim == 1, "soft-label source invalid")
    return 0.1 * F.one_hot(labels, num_classes=classes).to(torch.float64) + 0.9 / classes


def _norm(value: torch.Tensor) -> float:
    result = float(torch.linalg.vector_norm(value.to(torch.float64)).item())
    _need(math.isfinite(result), "nonfinite norm")
    return result


def _dot(left: torch.Tensor, right: torch.Tensor) -> float:
    result = float(torch.dot(left.to(torch.float64), right.to(torch.float64)).item())
    _need(math.isfinite(result), "nonfinite dot product")
    return result


def _project(value: torch.Tensor, basis: torch.Tensor | None) -> torch.Tensor | None:
    if basis is None:
        return None
    return basis @ (basis.T @ value)


def _pair_geometry(first: torch.Tensor, second: torch.Tensor,
                   mean: torch.Tensor | None, basis: torch.Tensor | None) -> dict[str, Any]:
    first64, second64 = first.to(torch.float64), second.to(torch.float64)
    difference = first64 - second64
    fresh = 0.5 * _dot(difference, difference)
    result: dict[str, Any] = {"fresh": fresh, "surprise": None, "innovation": None,
                              "mean_reason": None if mean is not None else "mean_unavailable"}
    if mean is not None:
        mean64 = mean.to(torch.float64)
        left, right = first64 - mean64, second64 - mean64
        result.update(surprise=_dot(left, right),
                      innovation=0.5 * (_dot(left, left) + _dot(right, right)))
    if basis is None:
        result["after_projection"] = None
        result["projection_reason"] = "basis_unavailable"
    else:
        projected_mean = basis @ (basis.T @ mean) if mean is not None else None
        result["after_projection"] = _pair_geometry(
            basis @ (basis.T @ first), basis @ (basis.T @ second), projected_mean, None)
        result["after_projection"].pop("after_projection")
        result["after_projection"].pop("projection_reason")
        result["projection_reason"] = None
    return result


def _retention(value: torch.Tensor, basis: torch.Tensor | None) -> dict[str, Any]:
    squared = _dot(value, value)
    if basis is None:
        return {"squared_norm": squared, "projected_squared_norm": None,
                "fraction": None, "reason": "basis_unavailable"}
    projected_squared = _dot(_project(value, basis), _project(value, basis))
    return {"squared_norm": squared, "projected_squared_norm": projected_squared,
            "fraction": None if squared == 0 else projected_squared / squared,
            "reason": "zero_vector" if squared == 0 else None}


def rescale_to_norm(value: torch.Tensor, target_norm: float) -> tuple[torch.Tensor | None, dict[str, Any]]:
    _need(type(target_norm) is float and math.isfinite(target_norm) and target_norm >= 0,
          "invalid target norm")
    source = _norm(value)
    if source == 0:
        return None, {"scale": None, "source_norm": source,
                      "target_norm": target_norm, "achieved_norm": None,
                      "reason": "zero_source_norm"}
    scale = target_norm / source
    if not math.isfinite(scale) or scale > MAX_RESCALE:
        return None, {"scale": None, "source_norm": source,
                      "target_norm": target_norm, "achieved_norm": None,
                      "reason": "scale_limit"}
    result = value * scale
    achieved = _norm(result)
    _need(math.isclose(achieved, target_norm, rel_tol=2e-6, abs_tol=2e-12),
          "rescaled norm differs after float32 application")
    return result, {"scale": scale, "source_norm": source,
                    "target_norm": target_norm, "achieved_norm": achieved, "reason": None}


def train_step(model: nn.Module, optimizer: torch.optim.Optimizer,
               tracker: SpectralGradientFilter, x: torch.Tensor,
               noisy: torch.Tensor, source: str) -> dict[str, Any]:
    _need(source in ("raw", "current32"), "unknown gradient source")
    optimizer.zero_grad(set_to_none=True)
    loss = _objective_loss(model, x, noisy)
    loss.backward()
    raw = flat_grad(model)
    diagnostics = tracker.filter_grad()
    native = flat_grad(model)
    if source == "raw":
        set_grad(model, raw)
    optimizer.step()
    _finite_tensor(flat_params(model), "updated parameters")
    return {"loss": float(loss.detach().item()), "source": source,
            "raw_norm": _norm(raw), "native_norm": _norm(native),
            "observer_step": int(diagnostics["step"]),
            "filtering_active": bool(diagnostics["filtering_active"])}


def _indices(value: Any, shape: tuple[int, ...], maximum: int, device: torch.device) -> torch.Tensor:
    raw = torch.as_tensor(value)
    _need(raw.dtype in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64),
          "probe indices must be integer-valued")
    tensor = raw.to(device=device, dtype=torch.long)
    _need(tuple(tensor.shape) == shape and maximum > 0
          and bool((tensor >= 0).all()) and bool((tensor < maximum).all()),
          "probe indices invalid")
    return tensor


def _basis(tracker: SpectralGradientFilter) -> torch.Tensor | None:
    if tracker.V is None:
        return None
    k = tracker.V.shape[1] if tracker.proj_k is None else min(tracker.proj_k, tracker.V.shape[1])
    return tracker.V[:, :k].detach().clone()


def _adam_delta(state: dict[str, Any], device: torch.device,
                applied: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    model, optimizer, _ = restore(state, device)
    before = flat_params(model)
    set_grad(model, applied.to(device))
    optimizer.step()
    after = flat_params(model)
    return _finite_tensor(after - before, "Adam displacement"), after


def _direct_delta(state: dict[str, Any], device: torch.device, theta: torch.Tensor,
                  planned: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    model, _, _ = restore(state, device)
    set_params(model, theta + planned)
    after = flat_params(model)
    return _finite_tensor(after - theta, "direct displacement"), after


@torch.no_grad()
def _loss_value(model: nn.Module, x: torch.Tensor, target: torch.Tensor) -> float:
    logits = model(x).to(torch.float64)
    if target.dtype.is_floating_point:
        _need(target.shape == logits.shape, "evaluation soft target shape differs")
        loss = -(target.to(torch.float64) * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
    else:
        loss = F.cross_entropy(logits, target, reduction="mean")
    result = float(loss.item())
    _need(math.isfinite(result), "nonfinite evaluation loss")
    return result


def _leakage(value: torch.Tensor, basis: torch.Tensor | None) -> dict[str, Any]:
    squared = _dot(value, value)
    if basis is None:
        return {"squared_norm": squared, "outside_squared_norm": None,
                "fraction": None, "reason": "basis_unavailable"}
    if squared == 0:
        return {"squared_norm": squared, "outside_squared_norm": 0.0,
                "fraction": None, "reason": "zero_displacement"}
    outside = value - _project(value, basis)
    outside_squared = _dot(outside, outside)
    return {"squared_norm": squared, "outside_squared_norm": outside_squared,
            "fraction": outside_squared / squared, "reason": None}


def probe_anchor(model: nn.Module, optimizer: torch.optim.Optimizer,
                 tracker: SpectralGradientFilter, data: dict[str, torch.Tensor],
                 plans: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Measure one frozen anchor without changing the supplied live state."""
    _need(type(data) is dict and tuple(data) == DATA_KEYS, "data keys/order differ")
    _need(type(plans) is dict and tuple(plans) ==
          ("pairs", "update", "loss_train", "loss_aux", "utility_aux"),
          "plan keys/order differ")
    anchor = snapshot(model, optimizer, tracker)
    device = next(model.parameters()).device
    try:
        work_model, work_optimizer, work_tracker = restore(anchor, device)
        _need(work_tracker is not None, "probe requires a tracker")
        for key in DATA_KEYS:
            _need(type(data[key]) is torch.Tensor and data[key].device == device,
                  "probe data must be tensors on the model device")
        train_n, aux_n = len(data["x"]), len(data["ax"])
        pairs = _indices(plans["pairs"], (32, 2, 64), train_n, device)
        update = _indices(plans["update"], (64,), train_n, device)
        loss_train = _indices(plans["loss_train"], (1024,), train_n, device)
        loss_aux = _indices(plans["loss_aux"], (1024,), aux_n, device)
        utility_aux = _indices(plans["utility_aux"], (1024,), aux_n, device)
        classes = anchor["model_spec"]["classes"]

        old_basis = _basis(work_tracker)
        old_mean = None if work_tracker.grad_mean is None else work_tracker.grad_mean.detach().clone()
        pair_rows = []
        paired_sums = {name: torch.zeros_like(flat_params(work_model))
                       for name in ("train_clean", "train_soft", "train_noisy")}
        for pair_index, row in enumerate(pairs):
            first = gradient(work_model, data["x"][row[0]], data["noisy"][row[0]])
            second = gradient(work_model, data["x"][row[1]], data["noisy"][row[1]])
            pair_rows.append({"pair": pair_index,
                              **_pair_geometry(first, second, old_mean, old_basis)})
            for indices, noisy_gradient in zip(row, (first, second)):
                batch_x = data["x"][indices]
                batch_clean = data["clean"][indices]
                paired_sums["train_clean"].add_(gradient(work_model, batch_x, batch_clean))
                paired_sums["train_soft"].add_(
                    gradient(work_model, batch_x, _soft(batch_clean, classes)))
                paired_sums["train_noisy"].add_(noisy_gradient)

        utility_inputs = {
            "train_clean": (data["x"][loss_train], data["clean"][loss_train]),
            "train_soft": (data["x"][loss_train], _soft(data["clean"][loss_train], classes)),
            "train_noisy": (data["x"][loss_train], data["noisy"][loss_train]),
            "aux_clean": (data["ax"][loss_aux], data["ay"][loss_aux]),
            "aux_soft": (data["ax"][loss_aux], _soft(data["ay"][loss_aux], classes)),
        }
        utilities = {name: paired_sums[name] / 64.0
                     for name in ("train_clean", "train_soft", "train_noisy")}
        utilities["aux_clean"] = gradient(
            work_model, data["ax"][utility_aux], data["ay"][utility_aux])
        utilities["aux_soft"] = gradient(
            work_model, data["ax"][utility_aux], _soft(data["ay"][utility_aux], classes))
        components = dict(utilities)
        components["soft_minus_clean"] = utilities["train_soft"] - utilities["train_clean"]
        components["fixed_minus_soft"] = utilities["train_noisy"] - utilities["train_soft"]
        component_names = (*UTILITY_NAMES, "soft_minus_clean", "fixed_minus_soft")
        gram = []
        for left_index, left in enumerate(component_names):
            for right in component_names[left_index:]:
                value = _dot(components[left], components[right])
                gram.append({"left": left, "right": right, "value": value,
                             "sign": 1 if value > 0 else (-1 if value < 0 else 0)})

        g0 = gradient(work_model, data["x"][update], data["noisy"][update])
        set_grad(work_model, g0)
        observer = work_tracker.filter_grad()
        pg = flat_grad(work_model)
        current_basis = _basis(work_tracker)
        post_observe = snapshot(work_model, work_optimizer, work_tracker)
        theta = flat_params(work_model)

        raw_to_native, raw_scale = rescale_to_norm(g0, _norm(pg))
        native_to_raw, native_scale = rescale_to_norm(pg, _norm(g0))
        applied = {"raw_gradient": g0, "native_gradient": pg,
                   "raw_gradient_to_native_norm": raw_to_native,
                   "native_gradient_to_raw_norm": native_to_raw}
        totals, after_parameters = {}, {}
        for name, vector in applied.items():
            if vector is not None:
                totals[name], after_parameters[name] = _adam_delta(post_observe, device, vector)
            else:
                totals[name] = after_parameters[name] = None
        decay = LR * WD * theta
        data_deltas = {name: None if total is None else total + decay
                       for name, total in totals.items()}
        raw_data_scaled, raw_data_scale = rescale_to_norm(
            data_deltas["raw_gradient"], _norm(data_deltas["native_gradient"]))
        native_data_scaled, native_data_scale = rescale_to_norm(
            data_deltas["native_gradient"], _norm(data_deltas["raw_gradient"]))
        for name, scaled in (("raw_data_delta_to_native_norm", raw_data_scaled),
                             ("native_data_delta_to_raw_norm", native_data_scaled)):
            if scaled is None:
                totals[name] = data_deltas[name] = after_parameters[name] = None
            else:
                totals[name], after_parameters[name] = _direct_delta(
                    anchor, device, theta, scaled - decay)
                data_deltas[name] = totals[name] + decay
        scales = {"raw_gradient_to_native_norm": raw_scale,
                  "native_gradient_to_raw_norm": native_scale,
                  "raw_data_delta_to_native_norm": raw_data_scale,
                  "native_data_delta_to_raw_norm": native_data_scale}

        base_model, _, _ = restore(anchor, device)
        before_losses = {name: _loss_value(base_model, *utility_inputs[name])
                         for name in UTILITY_NAMES}
        summaries = []
        for name in INTERVENTION_NAMES:
            delta, data_delta = totals[name], data_deltas[name]
            scale_record = scales.get(name)
            if delta is None:
                summaries.append({"name": name, "status": "undefined",
                                  "reason": scale_record["reason"], "gradient_scale": scale_record,
                                  "total_norm": None, "data_norm": None, "loss_change": None,
                                  "negative_gradient_dot_data_delta": None,
                                  "total_leakage": None, "data_leakage": None})
                continue
            if name == "raw_data_delta_to_native_norm":
                target = _norm(data_deltas["native_gradient"])
            elif name == "native_data_delta_to_raw_norm":
                target = _norm(data_deltas["raw_gradient"])
            else:
                target = None
            if target is not None:
                _need(math.isclose(_norm(data_delta), target, rel_tol=5e-5, abs_tol=1e-10),
                      "direct data norm differs after parameter quantization")
            candidate, _, _ = restore(anchor, device)
            set_params(candidate, after_parameters[name])
            changes = {utility: _loss_value(candidate, *utility_inputs[utility])
                       - before_losses[utility] for utility in UTILITY_NAMES}
            predicted = {}
            for utility in UTILITY_NAMES:
                value = -_dot(utilities[utility], data_delta)
                predicted[utility] = {"value": value,
                                      "sign": 1 if value > 0 else (-1 if value < 0 else 0)}
            summaries.append({
                "name": name, "status": "defined", "reason": None,
                "gradient_scale": scale_record,
                "total_norm": _norm(delta), "data_norm": _norm(data_delta),
                "loss_change": changes, "negative_gradient_dot_data_delta": predicted,
                "total_leakage": {"old_basis": _leakage(delta, old_basis),
                                  "current_basis": _leakage(delta, current_basis)},
                "data_leakage": {"old_basis": _leakage(data_delta, old_basis),
                                 "current_basis": _leakage(data_delta, current_basis)},
            })

        tensor_payload = {
            "g0": g0.detach().cpu().clone(), "pg": pg.detach().cpu().clone(),
            "current_V": None if current_basis is None else current_basis.detach().cpu().clone(),
            "mean_utility_gradients": {name: utilities[name].detach().cpu().clone()
                                       for name in UTILITY_NAMES},
            "interventions": {name: {
                "total_delta": None if totals[name] is None else totals[name].detach().cpu().clone(),
                "data_delta": None if data_deltas[name] is None else data_deltas[name].detach().cpu().clone()}
                              for name in INTERVENTION_NAMES},
        }
        record = {
            "schema": "i9_neural_anchor_probe_v1", "status": "complete",
            "anchor_snapshot_sha256": tree_digest(anchor),
            "observer_step_before": int(anchor["tracker"]["step_count"]),
            "observer_step_after": int(observer["step"]),
            "designated_update": {"raw_gradient_norm": _norm(g0),
                                  "native_gradient_norm": _norm(pg),
                                  "previous_basis_rank": 0 if old_basis is None else old_basis.shape[1],
                                  "current_basis_rank": 0 if current_basis is None else current_basis.shape[1]},
            "pair_rows": pair_rows, "utility_order": list(component_names),
            "signed_utility_gram": gram,
            "utility_retention": {
                name: {"old_basis": _retention(components[name], old_basis),
                       "current_basis": _retention(components[name], current_basis)}
                for name in component_names},
            "interventions": summaries,
            "tensor_payload_sha256": tree_digest(tensor_payload),
        }
        json.dumps(record, ensure_ascii=True, allow_nan=False, separators=(",", ":"))
        return record, tensor_payload
    finally:
        _restore_rng(anchor["rng"])
        after = snapshot(model, optimizer, tracker)
        _need(equal_tree(anchor, after), "probe changed parent state or RNG")


def main() -> int:
    print("neural_core: inert library; no training or probe executed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
