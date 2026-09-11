"""Inert model/update/snapshot helpers for the fixed rank-200 bridge.

No model creation, checkpoint/data reads, GPU inspection or training on import.
Only explicit factories, training_update and snapshot touch caller-owned state.
The snapshot truthfully records two hidden layers; no I9 restore is provided.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from experiments.spectral_general_augmentation import core as neural_core
from spectral_filter import SpectralGradientFilter


LR, WD, BETAS, EPS = .001, .01, (.9, .999), 1e-8
PARAMETER_COUNT = 235146
MODEL_SPEC = {"input_dim": 784, "hidden_dims": (256, 128), "classes": 10}
SNAPSHOT_SCHEMA = "spectral_strong_augmentation_snapshot_v1"
POLICIES = ("raw", "native200")
STREAM_KEYS = ("loss", "raw_norm", "applied_norm", "data_step_norm",
               "observer_steps", "adam_steps", "basis_rank")
flat_params, flat_grad = neural_core.flat_params, neural_core.flat_grad
set_params, set_grad = neural_core.set_params, neural_core.set_grad
tree_digest = neural_core.tree_digest


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def make_model(seed, device):
    """Literal manual_seed then ordinary CPU FP32 Linear initialization.

    This explicit factory resets torch's seed as specified by the protocol;
    unlike import, it is not RNG-inert. No layer/dimension overrides are exposed.
    """
    _need(type(seed) is int and 0 <= seed < 2**63, "invalid model seed")
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(784, 256, device="cpu", dtype=torch.float32), nn.ReLU(),
                          nn.Linear(256, 128, device="cpu", dtype=torch.float32), nn.ReLU(),
                          nn.Linear(128, 10, device="cpu", dtype=torch.float32))
    model._strong_augmentation_spec = dict(MODEL_SPEC)
    _need(sum(parameter.numel() for parameter in model.parameters()) == PARAMETER_COUNT,
          "model parameter count differs")
    return model.to(torch.device(device))


def make_optimizer(model):
    return torch.optim.AdamW(model.parameters(), lr=LR, betas=BETAS, eps=EPS,
                             weight_decay=WD, amsgrad=False, foreach=False, fused=False,
                             maximize=False, capturable=False, differentiable=False)


def make_tracker(model, optimizer):
    return SpectralGradientFilter(model, optimizer, rank=200, decay=.99, warmup=100,
                                  filter_strength=1.0, energy_threshold=None,
                                  adaptive="none", normalize="none", weighting="hard",
                                  alpha=1.0, soft_residual=True, stable_update=True,
                                  relative_eig_tol=1e-8, absolute_eig_floor=0.0,
                                  stabilize_every=100)


def _counts(optimizer, parameters):
    result = []
    for parameter in parameters:
        state = optimizer.state.get(parameter)
        if not state:
            result.append(0)
            continue
        _need(set(state) == {"step", "exp_avg", "exp_avg_sq"}, "noncanonical Adam state")
        _need(isinstance(state["step"], torch.Tensor) and state["step"].numel() == 1,
              "Adam counter must be a scalar tensor")
        step = float(state["step"].item())
        _need(math.isfinite(step) and step >= 0 and step.is_integer(), "invalid Adam counter")
        result.append(int(step))
        for name in ("exp_avg", "exp_avg_sq"):
            value = state[name]
            _need(isinstance(value, torch.Tensor) and value.dtype == torch.float32
                  and value.shape == parameter.shape and value.device == parameter.device,
                  "Adam moment shape/dtype/device differs")
    _need(len(set(result)) == 1, "Adam clocks differ across parameters")
    return result


def _validate_ownership(model, optimizer, tracker):
    parameters = tuple(model.parameters())
    _need(parameters and all(p.requires_grad and p.dtype == torch.float32 for p in parameters)
          and len({p.device for p in parameters}) == 1, "trainable single-device FP32 parameters required")
    _need(isinstance(optimizer, torch.optim.AdamW) and len(optimizer.param_groups) == 1,
          "one-group AdamW required")
    group = optimizer.param_groups[0]
    _need(len(group["params"]) == len(parameters)
          and all(left is right for left, right in zip(group["params"], parameters)),
          "optimizer parameter order differs")
    _need(group["lr"] == LR and group["weight_decay"] == WD and group["betas"] == BETAS
          and group["eps"] == EPS, "Adam scientific settings differ")
    _need(not any(group.get(key, False) for key in
                  ("amsgrad", "foreach", "fused", "maximize", "capturable", "differentiable")),
          "unsupported Adam option")
    counts = _counts(optimizer, parameters)
    if tracker is not None:
        _need(tracker.model is model and tracker.base_optimizer is optimizer
              and len(tracker.param_list) == len(parameters)
              and all(left is right for left, right in zip(tracker.param_list, parameters)),
              "tracker ownership differs")
        expected = {"rank": 200, "decay": .99, "warmup": 100, "filter_strength": 1.0,
                    "energy_threshold": None, "adaptive": "none", "normalize": "none",
                    "weighting": "hard", "alpha": 1.0, "soft_residual": True,
                    "stable_update": True, "relative_eig_tol": 1e-8,
                    "absolute_eig_floor": 0.0, "stabilize_every": 100}
        _need(all(getattr(tracker, key) == value for key, value in expected.items()),
              "tracker scientific settings differ")
        _need(type(tracker.step_count) is int and tracker.step_count == counts[0],
              "observer and Adam clocks differ")
    return parameters, counts


def _norm(value):
    result = float(torch.linalg.vector_norm(value.detach().double()).item())
    _need(math.isfinite(result), "nonfinite norm")
    return result


def training_update(model, optimizer, tracker, inputs, targets, policy):
    """One FP32 mean-CE backward, canonical optional filter, and actual AdamW.

    inputs[B,784] are already standardized FP32, with 1<=B<=64 (the production
    tail has B=16). targets[B] are assigned int64 labels, not heldout labels.
    Returns exactly STREAM_KEYS: Python float loss/norms and int counters/rank.
    Both raw/natively filtered actual data norms use FP32 after minus rounded
    FP32 theta*(1-lr*wd), then an FP64 norm. No hypothetical zero-input Adam,
    clipping, normalization, moment reset or update rescaling is performed.
    """
    _need(policy in POLICIES and ((policy == "raw") == (tracker is None)), "policy/tracker mismatch")
    parameters, before_counts = _validate_ownership(model, optimizer, tracker)
    _need(isinstance(inputs, torch.Tensor) and inputs.dtype == torch.float32
          and inputs.ndim == 2 and inputs.shape[1] == 784 and 1 <= inputs.shape[0] <= 64
          and inputs.device == parameters[0].device and bool(torch.isfinite(inputs).all()),
          "finite standardized FP32[B,784] inputs required, 1<=B<=64")
    _need(isinstance(targets, torch.Tensor) and targets.dtype == torch.int64
          and targets.shape == (len(inputs),) and targets.device == inputs.device
          and bool(((targets >= 0) & (targets < 10)).all()), "assigned int64 digit labels required")
    theta = flat_params(model)
    _need(bool(torch.isfinite(theta).all()), "nonfinite parameters before update")
    decay_base = theta.clone().mul_(1 - LR * WD)
    optimizer.zero_grad(set_to_none=True)
    logits = model(inputs)
    _need(logits.dtype == torch.float32 and logits.shape == (len(targets), 10)
          and bool(torch.isfinite(logits).all()), "finite FP32[B,10] logits required")
    loss = F.cross_entropy(logits, targets, reduction="mean")
    _need(bool(torch.isfinite(loss)), "nonfinite loss")
    loss.backward()
    raw_norm = _norm(flat_grad(model))
    if tracker is not None:
        tracker.filter_grad()
        _need(tracker.step_count == before_counts[0] + 1, "observer must advance exactly once")
    applied_norm = _norm(flat_grad(model))
    optimizer.step()
    after_counts = _counts(optimizer, parameters)
    _need(all(after == before + 1 for before, after in zip(before_counts, after_counts)),
          "every Adam counter must advance exactly once")
    finite_moments = torch.stack([torch.isfinite(optimizer.state[p][name]).all()
                                  for p in parameters for name in ("exp_avg", "exp_avg_sq")]).all()
    _need(bool(finite_moments), "nonfinite Adam moments")
    after = flat_params(model)
    _need(bool(torch.isfinite(after).all()), "nonfinite parameters after update")
    return {"loss": float(loss.detach().item()), "raw_norm": raw_norm, "applied_norm": applied_norm,
            "data_step_norm": _norm(after - decay_base),
            "observer_steps": 0 if tracker is None else int(tracker.step_count),
            "adam_steps": int(after_counts[0]),
            "basis_rank": 0 if tracker is None or tracker.V is None else int(tracker.V.shape[1])}


def snapshot(model, optimizer, tracker):
    """Fresh CPU copies under a truthful two-hidden-layer snapshot schema.

    Uses only shape-agnostic canonical clone, tracker, RNG and digest helpers;
    neither canonical snapshot nor its one-hidden-layer restore is called.
    This explicit operation captures available CUDA RNG through the accepted
    RNG helper in an admitted run; it performs no forward, backward or update.
    """
    _need(type(model) is nn.Sequential and len(model) == 5
          and [type(module) for module in model] == [nn.Linear, nn.ReLU, nn.Linear, nn.ReLU, nn.Linear]
          and [(model[i].in_features, model[i].out_features) for i in (0, 2, 4)]
          == [(784, 256), (256, 128), (128, 10)]
          and getattr(model, "_strong_augmentation_spec", None) == MODEL_SPEC
          and sum(p.numel() for p in model.parameters()) == PARAMETER_COUNT,
          "snapshot requires the exact two-hidden-layer strong model")
    _validate_ownership(model, optimizer, tracker)
    return {"schema": SNAPSHOT_SCHEMA, "model_spec": dict(MODEL_SPEC),
            "model_state": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
            "gradients": [None if p.grad is None else p.grad.detach().cpu().clone() for p in model.parameters()],
            "model_modes": [(name, bool(module.training)) for name, module in model.named_modules()],
            "optimizer": neural_core._cpu_clone(optimizer.state_dict()),
            "tracker": neural_core._tracker_state(tracker), "rng": neural_core._rng_state()}
