"""Strict, inert restoration of the strong bridge's native snapshot schema.

No checkpoint read, model construction, RNG mutation or accelerator inspection
occurs on import. Callers explicitly admit any real load/device use. Recorded
RNG is validated and preserved as payload; restore never installs it globally.
This module does not restore I9 snapshots or change any canonical source.
"""

from __future__ import annotations

import hashlib
import math
import os
import random
import re
import stat
from pathlib import Path

import numpy as np
import torch
from torch import nn

from experiments import spectral_strong_augmentation_core as strong


TOP_KEYS = ("schema", "model_spec", "model_state", "gradients", "model_modes",
            "optimizer", "tracker", "rng")
PARAMETERS = (("0.weight", (256, 784)), ("0.bias", (256,)),
              ("2.weight", (128, 256)), ("2.bias", (128,)),
              ("4.weight", (10, 128)), ("4.bias", (10,)))
MODEL_MODES = ("", "0", "1", "2", "3", "4")
GROUP_CONFIG = {"lr": .001, "betas": (.9, .999), "eps": 1e-8,
                "weight_decay": .01, "amsgrad": False, "maximize": False,
                "foreach": False, "capturable": False, "differentiable": False,
                "fused": False, "decoupled_weight_decay": True}
TRACKER_CONFIG = {"rank": 200, "decay": .99, "warmup": 100,
                  "filter_strength": 1.0, "weighting": "hard", "alpha": 1.0,
                  "soft_residual": True, "energy_threshold": None,
                  "normalize": "none", "adaptive": "none", "stable_update": True,
                  "relative_eig_tol": 1e-8, "absolute_eig_floor": 0.0,
                  "stabilize_every": 100, "n_params": strong.PARAMETER_COUNT}
TRACKER_KEYS = (*TRACKER_CONFIG, "V", "S", "proj_k", "step_count", "grad_mean",
                "stabilization_count", "max_orthogonality_error")
MAX_CHECKPOINT_BYTES = 256 * 1024**2
MAX_STEP = 56305  # includes the one proposed private post-final action


class RestoreError(ValueError):
    """Malformed or noncanonical snapshot; no fallback is attempted."""


def _need(condition, message):
    if not condition:
        raise RestoreError(message)


def _mapping(value, keys, label):
    expected = tuple(keys)
    _need(type(value) is dict and len(value) == len(expected)
          and all(type(a) is type(b) and a == b for a, b in zip(value, expected)),
          label + " keys/order differ")


def _literal(left, right):
    return type(left) is type(right) and (
        all(_literal(x, y) for x, y in zip(left, right)) and len(left) == len(right)
        if type(right) is tuple else left == right)


def _tensor(value, shape, dtype, label):
    _need(type(value) is torch.Tensor and value.device.type == "cpu"
          and value.layout == torch.strided and value.dtype == dtype
          and tuple(value.shape) == tuple(shape) and value.is_contiguous()
          and not value.requires_grad, label + " tensor shape/dtype/layout/device differs")
    _need(bool(torch.isfinite(value).all()), label + " is nonfinite")


def _step(value):
    _need(type(value) is int and 0 <= value <= MAX_STEP, "invalid expected_step")


def validate_rng(value):
    """Validate bounded CPU RNG payload without touching any global generator.

    Python, NumPy and CPU torch states are also admitted by private generators.
    CUDA bytes are structurally checked only, never queried/installed; their
    device-specific semantic validity is outside a CPU-only restoration check.
    """
    _mapping(value, ("python", "numpy", "torch_cpu", "torch_cuda"), "RNG")
    py = value["python"]
    _need(type(py) is tuple and len(py) == 3 and type(py[0]) is int and py[0] == 3
          and type(py[1]) is tuple and len(py[1]) == 625, "Python RNG structure differs")
    _need(all(type(x) is int and 0 <= x < 2**32 for x in py[1][:-1])
          and type(py[1][-1]) is int and 0 <= py[1][-1] <= 624
          and (py[2] is None or type(py[2]) is float and math.isfinite(py[2])),
          "Python RNG values differ")
    row = value["numpy"]
    _mapping(row, ("algorithm", "state", "position", "has_gauss", "cached_gaussian"), "NumPy RNG")
    _need(type(row["algorithm"]) is str and row["algorithm"] == "MT19937"
          and type(row["position"]) is int and 0 <= row["position"] <= 624
          and type(row["has_gauss"]) is int and row["has_gauss"] in (0, 1)
          and type(row["cached_gaussian"]) is float and math.isfinite(row["cached_gaussian"]),
          "NumPy RNG values differ")
    _tensor(row["state"], (624,), torch.uint32, "NumPy RNG state")
    _tensor(value["torch_cpu"], tuple(torch.get_rng_state().shape), torch.uint8, "torch CPU RNG")
    cuda = value["torch_cuda"]
    _need(type(cuda) is list and len(cuda) <= 8, "CUDA RNG topology envelope differs")
    for item in cuda:
        _need(type(item) is torch.Tensor and item.ndim == 1 and 1 <= item.numel() <= 65536,
              "CUDA RNG byte envelope differs")
        _tensor(item, tuple(item.shape), torch.uint8, "CUDA RNG bytes")
    try:
        random.Random(0).setstate(py)
        np.random.RandomState(0).set_state((row["algorithm"], row["state"].numpy().copy(),
                                          row["position"], row["has_gauss"], row["cached_gaussian"]))
        torch.Generator(device="cpu").set_state(value["torch_cpu"].clone())
    except (ValueError, TypeError, RuntimeError, OverflowError) as exc:
        raise RestoreError("private generator rejected recorded RNG") from exc


def validate_snapshot(value, *, expected_step):
    """Validate the exact ordered native schema; never construct a model."""
    _step(expected_step)
    _mapping(value, TOP_KEYS, "snapshot")
    _need(value["schema"] == strong.SNAPSHOT_SCHEMA and type(value["schema"]) is str,
          "wrong strong snapshot schema")
    spec = value["model_spec"]
    _mapping(spec, ("input_dim", "hidden_dims", "classes"), "model specification")
    _need(all(_literal(spec[k], v) for k, v in strong.MODEL_SPEC.items()), "model specification differs")
    _mapping(value["model_state"], [name for name, _ in PARAMETERS], "model parameter")
    for name, shape in PARAMETERS:
        _tensor(value["model_state"][name], shape, torch.float32, name)
    gradients = value["gradients"]
    _need(type(gradients) is list and len(gradients) == 6, "gradient topology differs")
    for (name, shape), gradient in zip(PARAMETERS, gradients):
        if gradient is None:
            _need(expected_step == 0, "positive-step snapshot is missing a gradient")
        else:
            _tensor(gradient, shape, torch.float32, name + " gradient")
    modes = value["model_modes"]
    _need(type(modes) is list and len(modes) == len(MODEL_MODES), "model modes differ")
    for item, name in zip(modes, MODEL_MODES):
        _need(type(item) is tuple and len(item) == 2 and type(item[0]) is str
              and item[0] == name and type(item[1]) is bool, "model mode topology/type differs")
    opt = value["optimizer"]
    _mapping(opt, ("state", "param_groups"), "optimizer")
    _need(type(opt["param_groups"]) is list and len(opt["param_groups"]) == 1,
          "exactly one optimizer group required")
    group = opt["param_groups"][0]
    _mapping(group, (*GROUP_CONFIG, "params"), "optimizer group")
    _need(all(_literal(group[k], v) for k, v in GROUP_CONFIG.items()), "optimizer configuration differs")
    _need(type(group["params"]) is list and len(group["params"]) == 6
          and all(type(i) is int and i == j for j, i in enumerate(group["params"])),
          "optimizer parameter ID order differs")
    _mapping(opt["state"], range(6) if expected_step else (), "optimizer state")
    for index, (_, shape) in enumerate(PARAMETERS):
        if not expected_step:
            break
        state = opt["state"][index]
        _mapping(state, ("step", "exp_avg", "exp_avg_sq"), "Adam state")
        _tensor(state["step"], (), torch.float32, "Adam counter")
        _need(float(state["step"].item()) == expected_step, "Adam counter differs")
        for key in ("exp_avg", "exp_avg_sq"):
            _tensor(state[key], shape, torch.float32, "Adam " + key)
        _need(bool((state["exp_avg_sq"] >= 0).all()), "negative Adam second moment")
    tracker = value["tracker"]
    _mapping(tracker, TRACKER_KEYS, "native tracker")
    _need(all(_literal(tracker[k], v) for k, v in TRACKER_CONFIG.items()), "tracker configuration differs")
    _need(type(tracker["step_count"]) is int and tracker["step_count"] == expected_step,
          "observer counter differs")
    _need(tracker["proj_k"] is None, "unexpected adaptive projection rank")
    _need(type(tracker["stabilization_count"]) is int
          and 0 <= tracker["stabilization_count"] <= 2 * expected_step,
          "invalid stabilization counter")
    error = tracker["max_orthogonality_error"]
    _need(type(error) is float and math.isfinite(error) and error >= 0, "invalid orthogonality diagnostic")
    if expected_step == 0:
        _need(tracker["grad_mean"] is None, "initial observer has a mean")
    else:
        _tensor(tracker["grad_mean"], (strong.PARAMETER_COUNT,), torch.float32, "gradient mean")
    if tracker["V"] is None:
        _need(tracker["S"] is None, "absent basis has singular values")
    else:
        basis = tracker["V"]
        _need(type(basis) is torch.Tensor and basis.ndim == 2
              and 1 <= basis.shape[1] <= min(200, max(0, expected_step - 1)), "basis rank differs")
        _tensor(basis, (strong.PARAMETER_COUNT, basis.shape[1]), torch.float32, "basis")
        singular = tracker["S"]
        _tensor(singular, (basis.shape[1],), torch.float64, "singular values")
        _need(bool((singular > 0).all()) and bool((singular[:-1] >= singular[1:]).all()),
              "singular values must be positive descending")
    validate_rng(value["rng"])


def snapshot_with_rng(model, optimizer, tracker, rng):
    """Copy live state with an explicitly supplied recorded RNG payload.

    This is not a fresh capture of ambient RNG and does not set any generator.
    It permits an honest round-trip check while keeping restoration RNG-inert.
    """
    validate_rng(rng)
    _need(type(model) is nn.Sequential and len(model) == 5
          and [type(m) for m in model] == [nn.Linear, nn.ReLU, nn.Linear, nn.ReLU, nn.Linear]
          and [(model[i].in_features, model[i].out_features) for i in (0, 2, 4)]
          == [(784, 256), (256, 128), (128, 10)]
          and getattr(model, "_strong_augmentation_spec", None) == strong.MODEL_SPEC,
          "live model topology differs")
    _need(tracker is not None, "native tracker required")
    strong._validate_ownership(model, optimizer, tracker)
    result = {"schema": strong.SNAPSHOT_SCHEMA, "model_spec": dict(strong.MODEL_SPEC),
              "model_state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
              "gradients": [None if p.grad is None else p.grad.detach().cpu().clone() for p in model.parameters()],
              "model_modes": [(name, bool(m.training)) for name, m in model.named_modules()],
              "optimizer": strong.neural_core._cpu_clone(optimizer.state_dict()),
              "tracker": strong.neural_core._tracker_state(tracker),
              "rng": strong.neural_core._cpu_clone(rng)}
    validate_snapshot(result, expected_step=tracker.step_count)
    return result


def restore(value, device="cpu", *, expected_step):
    """Reconstruct private model/Adam/native state, preserving caller CPU RNG.

    CPU work makes no CUDA calls. Explicit non-CPU targets permit only indexed
    CUDA devices; those transfers are for a separately admitted acquisition,
    not exercised by this module's CPU fixtures. No recorded RNG is installed.
    """
    validate_snapshot(value, expected_step=expected_step)
    target = torch.device(device)
    _need(target.type == "cpu" and target.index is None
          or target.type == "cuda" and target.index is not None, "explicit CPU or indexed CUDA target required")
    # Avoid strong.make_model's torch.manual_seed(), which also schedules CUDA
    # seeding. Literal CPU construction advances only the forked CPU generator.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(0)
        model = nn.Sequential(nn.Linear(784, 256, device="cpu", dtype=torch.float32), nn.ReLU(),
                              nn.Linear(256, 128, device="cpu", dtype=torch.float32), nn.ReLU(),
                              nn.Linear(128, 10, device="cpu", dtype=torch.float32))
    model._strong_augmentation_spec = dict(strong.MODEL_SPEC)
    model.load_state_dict(strong.neural_core._cpu_clone(value["model_state"]), strict=True)
    model.to(target)
    for (_, module), (_, training) in zip(model.named_modules(), value["model_modes"]):
        module.training = training
    for parameter, gradient in zip(model.parameters(), value["gradients"]):
        parameter.grad = None if gradient is None else gradient.to(target).clone()
    optimizer = strong.make_optimizer(model)
    optimizer.load_state_dict(strong.neural_core._cpu_clone(value["optimizer"]))
    tracker = strong.make_tracker(model, optimizer)
    _need(tuple(k for k in vars(tracker) if k not in ("model", "base_optimizer", "param_list"))
          == TRACKER_KEYS, "canonical tracker source topology changed")
    for key, item in value["tracker"].items():
        cloned = strong.neural_core._cpu_clone(item)
        if key in ("V", "grad_mean") and cloned is not None:
            cloned = cloned.to(target)
        setattr(tracker, key, cloned)
    strong._validate_ownership(model, optimizer, tracker)
    observed = snapshot_with_rng(model, optimizer, tracker, value["rng"])
    _need(strong.neural_core.equal_tree(value, observed), "restoration round-trip differs")
    return model, optimizer, tracker


def load_snapshot(path, *, expected_size, expected_sha256, expected_step):
    """Explicit pinned, bounded, restricted CPU load; no permissive fallback.

    The same open regular-file descriptor is hashed before and after loading.
    Cgroup memory/time limits remain required: restricted loading is not a
    sandbox for an untrusted or resource-malicious serialized file.
    """
    _step(expected_step)
    _need(type(expected_size) is int and 0 < expected_size <= MAX_CHECKPOINT_BYTES,
          "checkpoint byte envelope differs")
    _need(type(expected_sha256) is str and re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is not None,
          "invalid checkpoint hash")
    candidate = Path(path)
    _need(candidate.is_absolute(), "checkpoint path must be absolute")
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
    descriptor = os.open(candidate, flags)
    with os.fdopen(descriptor, "rb") as handle:
        before = os.fstat(handle.fileno())
        _need(stat.S_ISREG(before.st_mode) and before.st_size == expected_size,
              "checkpoint must be a regular file of the pinned size")

        def hash_file():
            handle.seek(0)
            digest = hashlib.sha256()
            remaining = expected_size
            while remaining:
                chunk = handle.read(min(1024**2, remaining))
                _need(bool(chunk), "checkpoint truncated while hashing")
                digest.update(chunk)
                remaining -= len(chunk)
            _need(not handle.read(1), "checkpoint grew while hashing")
            return digest.hexdigest()

        _need(hash_file() == expected_sha256, "checkpoint hash differs before load")
        handle.seek(0)
        value = torch.load(handle, map_location="cpu", weights_only=True)
        after = os.fstat(handle.fileno())
        _need((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
              == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns),
              "checkpoint metadata changed while loading")
        _need(hash_file() == expected_sha256, "checkpoint hash differs after load")
        final = os.fstat(handle.fileno())
        _need((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
              == (final.st_dev, final.st_ino, final.st_size, final.st_mtime_ns, final.st_ctime_ns),
              "checkpoint metadata changed during final hash")
    validate_snapshot(value, expected_step=expected_step)
    return value
