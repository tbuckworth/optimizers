#!/usr/bin/env python3
"""Strict iteration-007 training-state core; importing this module is inert.

Public API:
  capture_core(model, optimizer, observer, profile, completed_updates)
      Capture an owned tensor/primitive pre-forward state. All parameter grads
      must be None and optimizer/observer counters must equal completed_updates.
  validate_core(core)
      Fail closed on unknown/missing keys, types, order, devices, or counters.
  restore_core(core, model_factory, optimizer_factory, observer_factory)
      Construct fresh live objects, restore their state, restore RNG *after*
      constructors, verify RNG continuation without advancing it, and return
      {'model', 'optimizer', 'observer'}.
  save_core(path, core) / load_core(path)
      Atomic no-replace tensor/primitive serialization and restricted weights-
      only loading for trusted local artifacts. Full external hash, size, path,
      provenance, and resource enforcement belongs to the artifact controller.

This component contains no dataset, runner, launch, or full artifact bindings.
The tiny CPU profile exercises engineering paths only; it cannot certify the
scientific CUDA/MNIST profile.
"""
from __future__ import annotations

import copy
import os
from pathlib import Path
import random
import tempfile
from typing import Any, Callable

import numpy as np
import torch

from cuda_identity import canonical_cuda_uuid, is_canonical_cuda_uuid


SCIENTIFIC_PROFILE = "scientific_mnist_current32_v1"
FIXTURE_PROFILE = "fixture_tiny_cpu_v1"
MLP_FIXTURE_PROFILE = "fixture_tiny_mlp_cpu_v1"
PROFILES = (SCIENTIFIC_PROFILE, FIXTURE_PROFILE, MLP_FIXTURE_PROFILE)
ROOT_KEYS = (
    "schema_name", "schema_version", "profile", "phase", "anchor_update",
    "state_completed_updates", "model", "optimizer", "observer", "rng",
)
PARAMETER_ENTRY_KEYS = (
    "index", "name", "shape", "requires_grad", "native_dtype",
    "native_device", "value",
)
NATIVE_TENSOR_KEYS = ("value", "shape", "native_dtype", "native_device")
OPTIMIZER_KEYS = (
    "class_name", "state_completed_updates", "parameter_order", "param_groups", "state",
)
GROUP_KEYS = (
    "lr", "betas", "eps", "weight_decay", "amsgrad", "maximize", "foreach",
    "capturable", "differentiable", "fused", "decoupled_weight_decay", "param_indices",
)
OPT_STATE_KEYS = ("parameter_index", "parameter_name", "step", "exp_avg", "exp_avg_sq")
OBSERVER_KEYS = (
    "class_name", "state_completed_observations", "excluded_live_aliases", "config", "state",
)
OBSERVER_CONFIG_KEYS = (
    "rank", "decay", "warmup", "filter_strength", "energy_threshold", "adaptive",
    "normalize", "weighting", "alpha", "soft_residual", "stable_update",
    "relative_eig_tol", "absolute_eig_floor", "stabilize_every", "n_params",
)
OBSERVER_STATE_KEYS = (
    "V", "S", "proj_k", "step_count", "grad_mean",
    "stabilization_count", "max_orthogonality_error",
)
RNG_KEYS = ("python", "numpy", "torch_cpu", "torch_cuda", "continuation_witness")
PYTHON_RNG_KEYS = ("version", "internal", "gauss_next")
NUMPY_RNG_KEYS = ("algorithm", "keys", "position", "has_gauss", "cached_gaussian")
CUDA_RNG_KEYS = ("device_index", "name", "uuid", "state")
WITNESS_KEYS = ("spec", "python", "numpy", "torch_cpu", "torch_cuda")

PROFILE = {
    SCIENTIFIC_PROFILE: {
        "architecture": "Sequential(Linear(784,64),ReLU(),Linear(64,10))",
        "parameters": (("0.weight", (64, 784)), ("0.bias", (64,)),
                       ("2.weight", (10, 64)), ("2.bias", (10,))),
        "modules": ("", "0", "1", "2"), "rank": 32, "warmup": 100,
        "n_params": 50890, "device_kind": "cuda",
    },
    FIXTURE_PROFILE: {
        "architecture": "Sequential(Linear(3,2))",
        "parameters": (("0.weight", (2, 3)), ("0.bias", (2,))),
        "modules": ("", "0"), "rank": 2, "warmup": 2,
        "n_params": 8, "device_kind": "cpu",
    },
    MLP_FIXTURE_PROFILE: {
        "architecture": "Sequential(Linear(3,4),ReLU(),Linear(4,2))",
        "parameters": (("0.weight", (4, 3)), ("0.bias", (4,)),
                       ("2.weight", (2, 4)), ("2.bias", (2,))),
        "modules": ("", "0", "1", "2"), "rank": 2, "warmup": 2,
        "n_params": 26, "device_kind": "cpu",
    },
}
EXPECTED_GROUP = {
    "lr": 0.001, "betas": (0.9, 0.999), "eps": 1e-8,
    "weight_decay": 0.01, "amsgrad": False, "maximize": False,
    "foreach": False, "capturable": False, "differentiable": False,
    "fused": False, "decoupled_weight_decay": True,
}
OBSERVER_ALIASES = ("model", "base_optimizer", "param_list")


def _fail(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _exact_keys(value: Any, keys: tuple[str, ...], where: str) -> None:
    _fail(type(value) is dict, f"{where} must be a dict")
    _fail(tuple(value.keys()) == keys, f"{where} keys/order differ: {tuple(value.keys())}")


def _integer(value: Any, where: str, minimum: int = 0) -> int:
    _fail(type(value) is int and value >= minimum, f"{where} must be int >= {minimum}")
    return value


def _finite_float(value: Any, where: str) -> float:
    _fail(type(value) is float and np.isfinite(value), f"{where} must be a finite float")
    return value


def _device_string(tensor: torch.Tensor) -> str:
    return str(tensor.device)


def _device_kind(value: str, where: str) -> str:
    try:
        return torch.device(value).type
    except (RuntimeError, ValueError) as error:
        raise ValueError(f"{where} invalid device") from error


def _native_tensor(tensor: torch.Tensor) -> dict[str, Any]:
    _fail(type(tensor) is torch.Tensor, "native value must be a Tensor")
    return {
        "value": tensor.detach().cpu().contiguous().clone(),
        "shape": list(tensor.shape),
        "native_dtype": str(tensor.dtype),
        "native_device": _device_string(tensor),
    }


def _validate_native(record: Any, where: str, shape: tuple[int, ...] | None = None,
                     dtype: str | None = None, device_kind: str | None = None) -> None:
    _exact_keys(record, NATIVE_TENSOR_KEYS, where)
    tensor = record["value"]
    _fail(type(tensor) is torch.Tensor and tensor.device.type == "cpu" and tensor.is_contiguous(),
          f"{where}.value must be contiguous CPU Tensor")
    _fail(type(record["shape"]) is list and all(type(x) is int and x >= 0 for x in record["shape"]),
          f"{where}.shape invalid")
    _fail(tuple(record["shape"]) == tuple(tensor.shape), f"{where}.shape mismatch")
    _fail(record["native_dtype"] == str(tensor.dtype), f"{where}.native_dtype mismatch")
    _fail(type(record["native_device"]) is str, f"{where}.native_device invalid")
    if shape is not None:
        _fail(tuple(tensor.shape) == shape, f"{where} wrong shape")
    if dtype is not None:
        _fail(record["native_dtype"] == dtype, f"{where} wrong dtype")
    if device_kind is not None:
        _fail(_device_kind(record["native_device"], where) == device_kind, f"{where} wrong device")
        if device_kind == "cpu":
            _fail(record["native_device"] == "cpu", f"{where} noncanonical CPU device")
        else:
            parts = record["native_device"].split(":")
            _fail(len(parts) == 2 and parts[0] == "cuda" and parts[1].isdigit(),
                  f"{where} requires indexed CUDA device")
    if tensor.is_floating_point():
        _fail(bool(torch.isfinite(tensor).all()), f"{where} nonfinite")


def _parameter_spec(profile: str) -> tuple[tuple[str, tuple[int, ...]], ...]:
    return PROFILE[profile]["parameters"]


def _validate_live_model_structure(model: torch.nn.Module, profile: str) -> None:
    _fail(type(model) is torch.nn.Sequential, "model must be exact nn.Sequential")
    children = list(model.children())
    if profile in (SCIENTIFIC_PROFILE, MLP_FIXTURE_PROFILE):
        _fail(len(children) == 3 and type(children[0]) is torch.nn.Linear and
              type(children[1]) is torch.nn.ReLU and type(children[2]) is torch.nn.Linear,
              "two-layer module classes/order mismatch")
        first, relu, second = children
        width, hidden, classes = ((784, 64, 10) if profile == SCIENTIFIC_PROFILE else (3, 4, 2))
        _fail((first.in_features, first.out_features, first.bias is not None) == (width, hidden, True) and
              relu.inplace is False and
              (second.in_features, second.out_features, second.bias is not None) == (hidden, classes, True),
              "two-layer module configuration mismatch")
    else:
        _fail(len(children) == 1 and type(children[0]) is torch.nn.Linear,
              "fixture module class/order mismatch")
        layer = children[0]
        _fail((layer.in_features, layer.out_features, layer.bias is not None) == (3, 2, True),
              "fixture module configuration mismatch")


def _capture_model(model: torch.nn.Module, profile: str) -> dict[str, Any]:
    spec = PROFILE[profile]
    _validate_live_model_structure(model, profile)
    named = list(model.named_parameters())
    _fail(tuple((n, tuple(p.shape)) for n, p in named) == spec["parameters"], "parameter order/shape mismatch")
    _fail(list(model.named_buffers()) == [], "profile requires no model buffers")
    _fail(tuple(n for n, _ in model.named_modules()) == spec["modules"], "module order mismatch")
    parameters = []
    gradients = []
    for index, (name, parameter) in enumerate(named):
        _fail(parameter.requires_grad, f"{name} must require grad")
        _fail(parameter.dtype == torch.float32, f"{name} must be float32")
        _fail(parameter.device.type == spec["device_kind"], f"{name} wrong native device")
        _fail(parameter.grad is None, f"{name} gradient must be None at capture")
        parameters.append({"index": index, "name": name, "shape": list(parameter.shape),
                           "requires_grad": True, "native_dtype": str(parameter.dtype),
                           "native_device": str(parameter.device),
                           "value": parameter.detach().cpu().contiguous().clone()})
        gradients.append(None)
    return {
        "architecture": spec["architecture"],
        "parameter_order": [n for n, _ in spec["parameters"]],
        "parameters": parameters, "buffer_order": [], "buffers": [],
        "module_order": list(spec["modules"]),
        "training_modes": [bool(m.training) for _, m in model.named_modules()],
        "gradient_null_mask": [True] * len(named), "gradients": gradients,
    }


def _validate_model(value: Any, profile: str) -> None:
    keys = ("architecture", "parameter_order", "parameters", "buffer_order", "buffers",
            "module_order", "training_modes", "gradient_null_mask", "gradients")
    _exact_keys(value, keys, "model")
    spec = PROFILE[profile]
    names = [n for n, _ in spec["parameters"]]
    _fail(value["architecture"] == spec["architecture"], "architecture mismatch")
    _fail(value["parameter_order"] == names, "parameter_order mismatch")
    _fail(type(value["parameters"]) is list and len(value["parameters"]) == len(names), "parameters invalid")
    for index, (entry, (name, shape)) in enumerate(zip(value["parameters"], spec["parameters"])):
        _exact_keys(entry, PARAMETER_ENTRY_KEYS, f"parameters[{index}]")
        _fail(entry["index"] == index and type(entry["index"]) is int, "parameter index mismatch")
        _fail(entry["name"] == name and entry["shape"] == list(shape), "parameter metadata mismatch")
        _fail(entry["requires_grad"] is True, "requires_grad mismatch")
        record = {k: entry[k] for k in NATIVE_TENSOR_KEYS}
        _validate_native(record, f"parameters[{index}]", shape, "torch.float32", spec["device_kind"])
    _fail(value["buffer_order"] == [] and value["buffers"] == [], "buffers must be empty")
    _fail(value["module_order"] == list(spec["modules"]), "module_order mismatch")
    _fail(type(value["training_modes"]) is list and
          all(type(x) is bool for x in value["training_modes"]) and
          len(value["training_modes"]) == len(spec["modules"]), "training_modes invalid")
    _fail(type(value["gradient_null_mask"]) is list and
          len(value["gradient_null_mask"]) == len(names) and
          all(type(x) is bool and x is True for x in value["gradient_null_mask"]),
          "gradient null mask invalid")
    _fail(type(value["gradients"]) is list and len(value["gradients"]) == len(names) and
          all(x is None for x in value["gradients"]), "gradients must all be None")


def _capture_optimizer(optimizer: torch.optim.Optimizer, model: torch.nn.Module,
                       profile: str, completed: int) -> dict[str, Any]:
    params = list(model.parameters())
    _fail(type(optimizer).__name__ == "AdamW", "optimizer must be AdamW")
    group_params = optimizer.param_groups[0]["params"] if len(optimizer.param_groups) == 1 else []
    _fail(len(group_params) == len(params) and all(a is b for a, b in zip(group_params, params)),
          "optimizer parameter group order mismatch")
    state_dict = optimizer.state_dict()
    group = state_dict["param_groups"][0]
    _fail(set(group) == set(GROUP_KEYS[:-1] + ("params",)), "installed AdamW group keys mismatch")
    encoded_group = {k: copy.deepcopy(group[k]) for k in GROUP_KEYS if k != "param_indices"}
    encoded_group["param_indices"] = list(group["params"])
    _fail(_tree_equal(encoded_group, {**EXPECTED_GROUP, "param_indices": list(range(len(params)))}),
          "AdamW options mismatch")
    states = []
    names = [n for n, _ in _parameter_spec(profile)]
    for index, (name, parameter) in enumerate(zip(names, params)):
        raw = optimizer.state.get(parameter)
        _fail(type(raw) is dict and tuple(raw.keys()) == ("step", "exp_avg", "exp_avg_sq"),
              f"optimizer state fields invalid for {name}")
        step = raw["step"]
        _fail(type(step) is torch.Tensor and step.device.type == "cpu" and step.dtype == torch.float32 and
              step.shape == torch.Size([]) and float(step.item()) == completed, f"bad step for {name}")
        states.append({"parameter_index": index, "parameter_name": name,
                       "step": step.detach().cpu().contiguous().clone(),
                       "exp_avg": _native_tensor(raw["exp_avg"]),
                       "exp_avg_sq": _native_tensor(raw["exp_avg_sq"])})
    return {"class_name": "torch.optim.AdamW", "state_completed_updates": completed,
            "parameter_order": names, "param_groups": [encoded_group], "state": states}


def _validate_optimizer(value: Any, profile: str, completed: int, native_device: str) -> None:
    _exact_keys(value, OPTIMIZER_KEYS, "optimizer")
    names = [n for n, _ in _parameter_spec(profile)]
    _fail(value["class_name"] == "torch.optim.AdamW", "optimizer class mismatch")
    _fail(type(value["state_completed_updates"]) is int and
          value["state_completed_updates"] == completed, "optimizer counter mismatch")
    _fail(value["parameter_order"] == names, "optimizer parameter order mismatch")
    _fail(type(value["param_groups"]) is list and len(value["param_groups"]) == 1, "param_groups invalid")
    group = value["param_groups"][0]
    _exact_keys(group, GROUP_KEYS, "optimizer group")
    _fail(_tree_equal(group, {**EXPECTED_GROUP, "param_indices": list(range(len(names)))}),
          "optimizer group mismatch")
    _fail(type(value["state"]) is list and len(value["state"]) == len(names), "optimizer state count")
    spec = PROFILE[profile]
    for index, (entry, (name, shape)) in enumerate(zip(value["state"], _parameter_spec(profile))):
        _exact_keys(entry, OPT_STATE_KEYS, f"optimizer state {index}")
        _fail(type(entry["parameter_index"]) is int and entry["parameter_index"] == index and
              entry["parameter_name"] == name, "optimizer state binding mismatch")
        step = entry["step"]
        _fail(type(step) is torch.Tensor and step.device.type == "cpu" and step.dtype == torch.float32 and
              step.shape == torch.Size([]) and float(step.item()) == completed, "optimizer step mismatch")
        _validate_native(entry["exp_avg"], f"exp_avg[{name}]", shape, "torch.float32", spec["device_kind"])
        _validate_native(entry["exp_avg_sq"], f"exp_avg_sq[{name}]", shape, "torch.float32", spec["device_kind"])
        _fail(bool((entry["exp_avg_sq"]["value"] >= 0).all()), f"negative exp_avg_sq for {name}")
        _fail(entry["exp_avg"]["native_device"] == native_device and
              entry["exp_avg_sq"]["native_device"] == native_device, "moment device binding mismatch")


def _expected_observer_config(profile: str) -> dict[str, Any]:
    spec = PROFILE[profile]
    return {"rank": spec["rank"], "decay": 0.99, "warmup": spec["warmup"],
            "filter_strength": 1.0, "energy_threshold": None, "adaptive": "none",
            "normalize": "none", "weighting": "hard", "alpha": 1.0,
            "soft_residual": True, "stable_update": True, "relative_eig_tol": 1e-8,
            "absolute_eig_floor": 0.0, "stabilize_every": 100, "n_params": spec["n_params"]}


def _capture_observer(observer: Any, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                      profile: str, completed: int) -> dict[str, Any]:
    _fail(type(observer).__name__ == "SpectralGradientFilter", "wrong observer class")
    params = list(model.parameters())
    _fail(observer.model is model and observer.base_optimizer is optimizer and
          type(observer.param_list) is list and len(observer.param_list) == len(params) and
          all(a is b for a, b in zip(observer.param_list, params)),
          "observer live aliases/binding mismatch")
    expected_vars = set(OBSERVER_ALIASES + OBSERVER_CONFIG_KEYS + OBSERVER_STATE_KEYS)
    _fail(set(vars(observer)) == expected_vars, "observer field membership mismatch")
    config = {key: copy.deepcopy(getattr(observer, key)) for key in OBSERVER_CONFIG_KEYS}
    _fail(_tree_equal(config, _expected_observer_config(profile)), "observer config mismatch")
    _fail(observer.step_count == completed, "observer counter mismatch")
    state = {"V": None if observer.V is None else _native_tensor(observer.V),
             "S": None if observer.S is None else _native_tensor(observer.S),
             "proj_k": observer.proj_k, "step_count": observer.step_count,
             "grad_mean": None if observer.grad_mean is None else _native_tensor(observer.grad_mean),
             "stabilization_count": observer.stabilization_count,
             "max_orthogonality_error": float(observer.max_orthogonality_error)}
    return {"class_name": "SpectralGradientFilter", "state_completed_observations": completed,
            "excluded_live_aliases": list(OBSERVER_ALIASES), "config": config, "state": state}


def _validate_observer(value: Any, profile: str, completed: int, native_device: str) -> None:
    _exact_keys(value, OBSERVER_KEYS, "observer")
    spec = PROFILE[profile]
    _fail(value["class_name"] == "SpectralGradientFilter", "observer class mismatch")
    _fail(type(value["state_completed_observations"]) is int and
          value["state_completed_observations"] == completed, "observer counter mismatch")
    _fail(value["excluded_live_aliases"] == list(OBSERVER_ALIASES), "observer aliases mismatch")
    _exact_keys(value["config"], OBSERVER_CONFIG_KEYS, "observer config")
    _fail(_tree_equal(value["config"], _expected_observer_config(profile)), "observer config mismatch")
    state = value["state"]
    _exact_keys(state, OBSERVER_STATE_KEYS, "observer state")
    _fail(type(state["step_count"]) is int and state["step_count"] == completed, "observer step mismatch")
    _fail(type(state["stabilization_count"]) is int and state["stabilization_count"] >= 0,
          "stabilization_count invalid")
    _finite_float(state["max_orthogonality_error"], "max_orthogonality_error")
    _fail(state["max_orthogonality_error"] >= 0, "negative orthogonality error")
    _fail(state["proj_k"] is None, "proj_k must be None")
    _fail((state["V"] is None) == (state["S"] is None), "V/S None coupling invalid")
    if state["V"] is not None:
        _validate_native(state["V"], "observer.V", dtype="torch.float32", device_kind=spec["device_kind"])
        k = state["V"]["shape"][1] if len(state["V"]["shape"]) == 2 else -1
        _fail(state["V"]["shape"] == [spec["n_params"], k] and 1 <= k <= spec["rank"], "V shape invalid")
        _validate_native(state["S"], "observer.S", (k,), "torch.float64", "cpu")
        _fail(bool((state["S"]["value"] >= 0).all()), "observer S must be nonnegative")
        _fail(state["V"]["native_device"] == native_device, "observer V device binding mismatch")
    if completed == 0:
        _fail(state["grad_mean"] is None, "initial grad_mean must be None")
    else:
        _validate_native(state["grad_mean"], "observer.grad_mean", (spec["n_params"],),
                         "torch.float32", spec["device_kind"])
        _fail(state["grad_mean"]["native_device"] == native_device, "grad_mean device binding mismatch")


def _encode_python_rng(state: tuple[Any, ...]) -> dict[str, Any]:
    version, internal, gauss_next = state
    return {"version": int(version), "internal": [int(x) for x in internal],
            "gauss_next": None if gauss_next is None else float(gauss_next)}


def _decode_python_rng(value: dict[str, Any]) -> tuple[Any, ...]:
    return value["version"], tuple(value["internal"]), value["gauss_next"]


def _encode_numpy_rng(state: tuple[Any, ...]) -> dict[str, Any]:
    algorithm, keys, position, has_gauss, cached = state
    return {"algorithm": algorithm,
            "keys": torch.from_numpy(np.asarray(keys, dtype=np.uint32).copy()).contiguous(),
            "position": int(position), "has_gauss": int(has_gauss),
            "cached_gaussian": float(cached)}


def _decode_numpy_rng(value: dict[str, Any]) -> tuple[Any, ...]:
    keys = value["keys"].cpu().numpy().astype(np.uint32, copy=True)
    return value["algorithm"], keys, value["position"], value["has_gauss"], value["cached_gaussian"]


def _cuda_identity(index: int) -> tuple[str, str]:
    props = torch.cuda.get_device_properties(index)
    return str(props.name), canonical_cuda_uuid(getattr(props, "uuid", None))


def _raw_rng_state() -> dict[str, Any]:
    cuda = []
    if torch.cuda.is_available():
        for index, state in enumerate(torch.cuda.get_rng_state_all()):
            name, uuid = _cuda_identity(index)
            cuda.append({"device_index": index, "name": name, "uuid": uuid,
                         "state": state.detach().cpu().contiguous().clone()})
    return {"python": _encode_python_rng(random.getstate()),
            "numpy": _encode_numpy_rng(np.random.get_state()),
            "torch_cpu": torch.get_rng_state().detach().cpu().contiguous().clone(),
            "torch_cuda": cuda}


def _set_rng_state(value: dict[str, Any]) -> None:
    random.setstate(_decode_python_rng(value["python"]))
    np.random.set_state(_decode_numpy_rng(value["numpy"]))
    torch.set_rng_state(value["torch_cpu"].clone())
    if value["torch_cuda"]:
        states = [entry["state"].clone() for entry in value["torch_cuda"]]
        torch.cuda.set_rng_state_all(states)


def _draw_witness() -> dict[str, Any]:
    cuda_draws = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            cuda_draws.append(torch.rand(4, device=f"cuda:{index}", dtype=torch.float32).cpu())
    return {"spec": "four_draws_v1", "python": [random.random() for _ in range(4)],
            "numpy": [float(x) for x in np.random.random_sample(4)],
            "torch_cpu": torch.rand(4, dtype=torch.float64), "torch_cuda": cuda_draws}


def _capture_rng() -> dict[str, Any]:
    saved = _raw_rng_state()
    try:
        witness = _draw_witness()
    finally:
        _set_rng_state(saved)
    return {**saved, "continuation_witness": witness}


def _validate_rng(value: Any, profile: str) -> None:
    _exact_keys(value, RNG_KEYS, "rng")
    _exact_keys(value["python"], PYTHON_RNG_KEYS, "python rng")
    py = value["python"]
    _fail(type(py["version"]) is int and py["version"] == 3 and type(py["internal"]) is list and
          len(py["internal"]) == 625 and
          all(type(x) is int and 0 <= x <= 0xFFFFFFFF for x in py["internal"][:624]) and
          type(py["internal"][624]) is int and 0 <= py["internal"][624] <= 624 and
          (py["gauss_next"] is None or
           (type(py["gauss_next"]) is float and np.isfinite(py["gauss_next"]))), "python rng invalid")
    _exact_keys(value["numpy"], NUMPY_RNG_KEYS, "numpy rng")
    nr = value["numpy"]
    _fail(nr["algorithm"] == "MT19937" and type(nr["keys"]) is torch.Tensor and
          nr["keys"].device.type == "cpu" and nr["keys"].dtype == torch.uint32 and
          nr["keys"].shape == (624,) and nr["keys"].is_contiguous(), "numpy rng invalid")
    _integer(nr["position"], "numpy position")
    _fail(nr["position"] <= 624 and type(nr["has_gauss"]) is int and nr["has_gauss"] in (0, 1),
          "numpy rng scalar invalid")
    _finite_float(nr["cached_gaussian"], "numpy cached gaussian")
    current_cpu_rng_size = torch.get_rng_state().numel()
    _fail(type(value["torch_cpu"]) is torch.Tensor and value["torch_cpu"].device.type == "cpu" and
          value["torch_cpu"].dtype == torch.uint8 and value["torch_cpu"].ndim == 1 and
          value["torch_cpu"].numel() == current_cpu_rng_size and value["torch_cpu"].numel() > 0 and
          value["torch_cpu"].is_contiguous(), "torch CPU RNG invalid")
    _fail(type(value["torch_cuda"]) is list, "torch_cuda must be list")
    if profile in (FIXTURE_PROFILE, MLP_FIXTURE_PROFILE):
        _fail(value["torch_cuda"] == [], "fixture must not contain CUDA RNG")
    else:
        _fail(len(value["torch_cuda"]) > 0, "scientific profile requires CUDA RNG")
    for index, entry in enumerate(value["torch_cuda"]):
        _exact_keys(entry, CUDA_RNG_KEYS, f"cuda rng {index}")
        _fail(entry["device_index"] == index and type(entry["device_index"]) is int, "CUDA order invalid")
        _fail(type(entry["name"]) is str and bool(entry["name"]) and
              is_canonical_cuda_uuid(entry["uuid"]), "CUDA identity invalid")
        state = entry["state"]
        _fail(type(state) is torch.Tensor and state.device.type == "cpu" and state.dtype == torch.uint8 and
              state.ndim == 1 and state.numel() > 0 and state.is_contiguous(), "CUDA RNG tensor invalid")
    witness = value["continuation_witness"]
    _exact_keys(witness, WITNESS_KEYS, "RNG continuation witness")
    _fail(witness["spec"] == "four_draws_v1" and type(witness["python"]) is list and
          len(witness["python"]) == 4 and
          all(type(x) is float and np.isfinite(x) and 0 <= x < 1 for x in witness["python"]), "witness invalid")
    _fail(type(witness["numpy"]) is list and len(witness["numpy"]) == 4 and
          all(type(x) is float and np.isfinite(x) and 0 <= x < 1 for x in witness["numpy"]), "NumPy witness invalid")
    _fail(type(witness["torch_cpu"]) is torch.Tensor and witness["torch_cpu"].shape == (4,) and
          witness["torch_cpu"].device.type == "cpu" and witness["torch_cpu"].dtype == torch.float64 and
          witness["torch_cpu"].is_contiguous() and bool(torch.isfinite(witness["torch_cpu"]).all()) and
          bool(((witness["torch_cpu"] >= 0) & (witness["torch_cpu"] < 1)).all()),
          "Torch witness invalid")
    _fail(type(witness["torch_cuda"]) is list and len(witness["torch_cuda"]) == len(value["torch_cuda"]),
          "CUDA witness count invalid")
    _fail(all(type(x) is torch.Tensor and x.device.type == "cpu" and x.dtype == torch.float32 and
              x.shape == (4,) and x.is_contiguous() and bool(torch.isfinite(x).all())
              and bool(((x >= 0) & (x < 1)).all())
              for x in witness["torch_cuda"]), "CUDA witness invalid")


def _assert_no_tensor_aliases(value: Any) -> None:
    spans: list[tuple[int, int, int]] = []
    def visit(item: Any) -> None:
        if type(item) is torch.Tensor and item.numel() > 0:
            storage = item.untyped_storage()
            span = (storage.data_ptr(), storage.nbytes())
            _fail(span not in spans, "serialized tensors share storage")
            spans.append(span)
        elif type(item) is dict:
            for child in item.values():
                visit(child)
        elif type(item) in (list, tuple):
            for child in item:
                visit(child)
    visit(value)


def capture_core(model: torch.nn.Module, optimizer: torch.optim.Optimizer, observer: Any,
                 *, profile: str, completed_updates: int) -> dict[str, Any]:
    """Capture an owned, strict pre-forward core without advancing any RNG."""
    _fail(profile in PROFILES, "unknown profile")
    completed = _integer(completed_updates, "completed_updates", 1)
    before_rng = _raw_rng_state()
    core = {"schema_name": "i7_state_core", "schema_version": 1, "profile": profile,
            "phase": "pre_forward_pre_observe", "anchor_update": completed + 1,
            "state_completed_updates": completed,
            "model": _capture_model(model, profile),
            "optimizer": _capture_optimizer(optimizer, model, profile, completed),
            "observer": _capture_observer(observer, model, optimizer, profile, completed),
            "rng": _capture_rng()}
    after_rng = _raw_rng_state()
    _fail(_tree_equal(before_rng, after_rng), "capture changed caller RNG")
    validate_core(core)
    _assert_no_tensor_aliases(core)
    return core


def validate_core(core: Any) -> None:
    """Validate exact state-core schema and cross-field invariants."""
    _exact_keys(core, ROOT_KEYS, "core")
    _fail(core["schema_name"] == "i7_state_core" and type(core["schema_name"]) is str and
          type(core["schema_version"]) is int and core["schema_version"] == 1, "schema mismatch")
    profile = core["profile"]
    _fail(profile in PROFILES, "unknown profile")
    completed = _integer(core["state_completed_updates"], "state_completed_updates", 1)
    _fail(type(core["phase"]) is str and core["phase"] == "pre_forward_pre_observe", "wrong phase")
    _fail(type(core["anchor_update"]) is int and core["anchor_update"] == completed + 1, "anchor counter mismatch")
    _validate_model(core["model"], profile)
    native_devices = [entry["native_device"] for entry in core["model"]["parameters"]]
    _fail(len(set(native_devices)) == 1, "model parameters span devices")
    native_device = native_devices[0]
    _validate_optimizer(core["optimizer"], profile, completed, native_device)
    _validate_observer(core["observer"], profile, completed, native_device)
    _validate_rng(core["rng"], profile)
    if profile == SCIENTIFIC_PROFILE:
        _fail(int(native_device.split(":")[1]) < len(core["rng"]["torch_cuda"]),
              "model CUDA device absent from RNG state")
    _assert_no_tensor_aliases(core)


def _restore_native(record: dict[str, Any]) -> torch.Tensor:
    return record["value"].detach().clone().to(torch.device(record["native_device"]))


def _verify_rng_continuation(rng: dict[str, Any]) -> None:
    if rng["torch_cuda"]:
        _fail(torch.cuda.device_count() == len(rng["torch_cuda"]), "CUDA device count mismatch")
        for entry in rng["torch_cuda"]:
            _fail(_cuda_identity(entry["device_index"]) == (entry["name"], entry["uuid"]),
                  "CUDA stable identity mismatch")
    _set_rng_state(rng)
    try:
        actual = _draw_witness()
        _fail(_tree_equal(actual, rng["continuation_witness"]), "RNG continuation mismatch")
    finally:
        _set_rng_state(rng)


def restore_core(core: dict[str, Any], model_factory: Callable[[], torch.nn.Module],
                 optimizer_factory: Callable[[torch.nn.Module], torch.optim.Optimizer],
                 observer_factory: Callable[[torch.nn.Module, torch.optim.Optimizer], Any]) -> dict[str, Any]:
    """Restore fresh live objects; success leaves RNG at the captured state.

    Constructor/validation failure restores the caller's entry RNG state.
    """
    caller_rng = _raw_rng_state()
    try:
        return _restore_core_impl(core, model_factory, optimizer_factory, observer_factory)
    except BaseException:
        _set_rng_state(caller_rng)
        raise


def _restore_core_impl(core: dict[str, Any], model_factory: Callable[[], torch.nn.Module],
                       optimizer_factory: Callable[[torch.nn.Module], torch.optim.Optimizer],
                       observer_factory: Callable[[torch.nn.Module, torch.optim.Optimizer], Any]) -> dict[str, Any]:
    validate_core(core)
    profile = core["profile"]
    model = model_factory()
    optimizer = optimizer_factory(model)
    observer = observer_factory(model, optimizer)
    # Constructors may reseed/consume RNG. State restoration deliberately follows them.
    _validate_live_model_structure(model, profile)
    _fail(tuple((n, tuple(p.shape)) for n, p in model.named_parameters()) == _parameter_spec(profile),
          "restored model factory order mismatch")
    state_dict = {entry["name"]: entry["value"].clone().to(entry["native_device"])
                  for entry in core["model"]["parameters"]}
    model.load_state_dict(state_dict, strict=True)
    for mode, (_, module) in zip(core["model"]["training_modes"], model.named_modules()):
        module.train(mode)
    params = list(model.parameters())
    opt_state = {}
    for entry, parameter in zip(core["optimizer"]["state"], params):
        opt_state[entry["parameter_index"]] = {
            "step": entry["step"].clone(), "exp_avg": _restore_native(entry["exp_avg"]),
            "exp_avg_sq": _restore_native(entry["exp_avg_sq"]),
        }
    group = copy.deepcopy(core["optimizer"]["param_groups"][0])
    group["params"] = group.pop("param_indices")
    optimizer.load_state_dict({"state": opt_state, "param_groups": [group]})
    restored_params = optimizer.param_groups[0]["params"]
    _fail(len(restored_params) == len(params) and all(a is b for a, b in zip(restored_params, params)),
          "restored optimizer binding mismatch")
    expected_config = _expected_observer_config(profile)
    _fail(_tree_equal({k: getattr(observer, k) for k in OBSERVER_CONFIG_KEYS}, expected_config),
          "restored observer factory config mismatch")
    state = core["observer"]["state"]
    observer.V = None if state["V"] is None else _restore_native(state["V"])
    observer.S = None if state["S"] is None else _restore_native(state["S"])
    observer.proj_k = state["proj_k"]
    observer.step_count = state["step_count"]
    observer.grad_mean = None if state["grad_mean"] is None else _restore_native(state["grad_mean"])
    observer.stabilization_count = state["stabilization_count"]
    observer.max_orthogonality_error = state["max_orthogonality_error"]
    _fail(observer.model is model and observer.base_optimizer is optimizer and
          len(observer.param_list) == len(params) and
          all(a is b for a, b in zip(observer.param_list, params)),
          "observer live aliases/binding mismatch")
    optimizer.zero_grad(set_to_none=True)
    _verify_rng_continuation(core["rng"])
    restored = {"model": model, "optimizer": optimizer, "observer": observer}
    # Re-capture validates values, devices, options, counters, modes, null grads and ownership.
    check = capture_core(model, optimizer, observer, profile=profile,
                         completed_updates=core["state_completed_updates"])
    _fail(_tree_equal(check, core), "restored core differs from source")
    _set_rng_state(core["rng"])
    return restored


def save_core(path: str | os.PathLike[str], core: dict[str, Any]) -> None:
    """Atomically publish a validated trusted-local core without replacement."""
    validate_core(core)
    target = Path(path)
    _fail(target.parent.is_dir(), "save parent must already exist")
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"refusing to overwrite immutable core: {target}")
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    os.close(fd)
    try:
        torch.save(core, temporary)
        with open(temporary, "rb") as handle:
            os.fsync(handle.fileno())
        # A same-directory hard link is atomic and fails with EEXIST; unlike
        # os.replace it cannot overwrite a target created after the early check.
        os.link(temporary, target)
        os.unlink(temporary)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_core(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Load a trusted local core with weights-only CPU decoding and validation.

    This is not the external artifact envelope: callers must separately enforce
    expected path, regular-file status, byte size, SHA-256, provenance, and caps.
    """
    core = torch.load(Path(path), map_location="cpu", weights_only=True)
    validate_core(core)
    return core


def _tree_equal(left: Any, right: Any) -> bool:
    if type(left) is torch.Tensor:
        if type(right) is not torch.Tensor or left.dtype != right.dtype or left.shape != right.shape:
            return False
        a = left.detach().cpu().contiguous().numpy().tobytes(order="C")
        b = right.detach().cpu().contiguous().numpy().tobytes(order="C")
        return a == b
    if type(left) is not type(right):
        return False
    if type(left) is dict:
        return tuple(left.keys()) == tuple(right.keys()) and all(_tree_equal(left[k], right[k]) for k in left)
    if type(left) in (list, tuple):
        return len(left) == len(right) and all(_tree_equal(a, b) for a, b in zip(left, right))
    return left == right


def clone_tree(value: Any) -> Any:
    """Return an owned clone of a validated tensor/primitive tree."""
    if type(value) is torch.Tensor:
        return value.detach().cpu().contiguous().clone()
    if type(value) is dict:
        return {key: clone_tree(child) for key, child in value.items()}
    if type(value) is list:
        return [clone_tree(child) for child in value]
    if type(value) is tuple:
        return tuple(clone_tree(child) for child in value)
    _fail(value is None or type(value) in (bool, int, float, str), "non-primitive tree leaf")
    return copy.deepcopy(value)
