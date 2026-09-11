"""Inert update helper for one fixed clean multiview experiment.

Only explicit training_update calls change caller-owned states. This module
does not create models, load data/checkpoints, inspect GPUs or launch work.
Accepted neural_core helpers and the canonical filter are not modified.
"""

import math

import numpy as np
import torch
from torch.nn import functional as F

from experiments.spectral_general_augmentation import core as neural_core
from experiments.spectral_multiview_data import POLICIES


STREAM_KEYS = ("losses", "gradient_evaluations", "observer_steps", "adam_steps",
               "basis_rank", "observation_norm", "delivery_norm", "applied_norm",
               "data_step_norm")


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _norm(value):
    result = float(torch.linalg.vector_norm(value.detach().double()).item())
    _need(math.isfinite(result), "nonfinite vector norm")
    return result


def _adam_counts(optimizer, parameters):
    counts = []
    for parameter in parameters:
        state = optimizer.state.get(parameter)
        if not state:
            counts.append(0)
            continue
        _need(set(state) == {"step", "exp_avg", "exp_avg_sq"}, "noncanonical Adam state")
        counter = state["step"]
        _need(isinstance(counter, torch.Tensor) and counter.numel() == 1,
              "Adam counter must be a scalar tensor")
        step = float(counter.item())
        _need(math.isfinite(step) and step >= 0 and step.is_integer(), "invalid Adam counter")
        counts.append(int(step))
        for key in ("exp_avg", "exp_avg_sq"):
            value = state[key]
            _need(value.shape == parameter.shape and value.dtype == torch.float32
                  and value.device == parameter.device and bool(torch.isfinite(value).all()),
                  "invalid or nonfinite Adam moment")
    _need(len(set(counts)) == 1, "Adam parameter clocks differ")
    return counts


def _validate(model, optimizer, tracker, views, targets, policy):
    _need(policy in POLICIES, "unknown policy")
    raw = policy in ("raw1", "raw4")
    _need(raw == (tracker is None), "tracker/policy mismatch")
    expected_views = 4 if policy in ("observer4", "raw4") else 1
    _need(isinstance(views, torch.Tensor) and views.dtype == torch.float32
          and views.ndim == 3 and views.shape[0] == expected_views
          and views.shape[1] > 0 and views.shape[2] == 784,
          "views must be float32[V,B,784], V=1 or 4 according to policy")
    _need(bool(torch.isfinite(views).all()) and bool(((views >= 0) & (views <= 1)).all()),
          "views must be finite in [0,1]")
    _need(isinstance(targets, torch.Tensor) and targets.dtype == torch.int64
          and targets.shape == (views.shape[1],) and targets.device == views.device,
          "targets must be int64[B] on the views device")
    _need(bool(((targets >= 0) & (targets < 10)).all()), "targets must be digits 0..9")
    parameters = tuple(model.parameters())
    _need(parameters and all(p.requires_grad and p.dtype == torch.float32
                            and p.device == views.device for p in parameters),
          "all model parameters must be trainable FP32 on the views device")
    _need(isinstance(optimizer, torch.optim.AdamW) and len(optimizer.param_groups) == 1,
          "one-group AdamW required")
    group = optimizer.param_groups[0]
    _need(len(group["params"]) == len(parameters)
          and all(left is right for left, right in zip(group["params"], parameters)),
          "optimizer parameter order differs from model")
    _need(not any(group.get(key, False) for key in
                  ("amsgrad", "maximize", "capturable", "differentiable", "foreach", "fused")),
          "unsupported AdamW option")
    _need(all(isinstance(group[key], (int, float)) and math.isfinite(group[key])
              for key in ("lr", "weight_decay")), "finite scalar AdamW lr/decay required")
    counts = _adam_counts(optimizer, parameters)
    if tracker is not None:
        _need(tracker.model is model and tracker.base_optimizer is optimizer
              and len(tracker.param_list) == len(parameters)
              and all(left is right for left, right in zip(tracker.param_list, parameters)),
              "tracker ownership or parameter order differs")
        _need(tracker.stable_update and tracker.weighting == "hard"
              and tracker.normalize == "none" and tracker.adaptive == "none"
              and tracker.energy_threshold is None and tracker.filter_strength == 1.0,
              "canonical stable hard-projection tracker required")
        _need(type(tracker.step_count) is int and tracker.step_count == counts[0],
              "observer and Adam clocks differ before update")
    return parameters, group, counts


def training_update(model, optimizer, tracker, views, targets, policy):
    """Execute one update, returning exactly STREAM_KEYS (no vector archive).

    The caller supplies a deterministic stateless MLP and FP32 views[V,B,784].
    Each view is separately differentiated at the SAME parameters using FP32
    mean CE and autograd.grad (no .grad accumulation or graph retention).
    For four views, average by sequential FP32 (((g1+g2)+g3)+g4)/4.

    raw1/native1/observer4 deliver g1 before filtering; raw4 delivers the mean
    from update 1. native1 uses canonical filter_grad verbatim. observer4
    advances its canonical covariance update once on the mean, then projects
    g1 only when step_count>warmup, without a second ingest. AdamW advances
    exactly once, with no moment, scale, normalization or decay changes.

    losses is np.float64[4] containing FP32-loss values, with unused slots zero;
    gradient_evaluations=1/4 determines valid slots. Raw observer_steps,
    basis_rank and observation_norm are zero. Other counts are Python ints;
    all norms are Python floats evaluated in FP64 from their FP32 vectors.
    delivery_norm is BEFORE projection, applied_norm is the Adam input AFTER
    projection. data_step_norm uses actual FP32 after - FP32(theta*(1-lr*wd)),
    not a zero-gradient Adam proposal. Existing .grad values are replaced.
    """
    parameters, group, before_counts = _validate(model, optimizer, tracker, views, targets, policy)
    theta = neural_core.flat_params(model)
    _need(bool(torch.isfinite(theta).all()), "nonfinite parameters before update")
    decay_base = theta.clone().mul_(1 - group["lr"] * group["weight_decay"])
    optimizer.zero_grad(set_to_none=True)
    losses = np.zeros(4, dtype=np.float64)
    first, accumulated = None, None
    for view_index, inputs in enumerate(views):
        logits = model(inputs)
        _need(logits.dtype == torch.float32 and logits.shape == (len(targets), 10)
              and bool(torch.isfinite(logits).all()), "finite FP32[B,10] logits required")
        loss = F.cross_entropy(logits, targets, reduction="mean")
        _need(bool(torch.isfinite(loss)), "nonfinite training loss")
        gradients = torch.autograd.grad(loss, parameters, retain_graph=False,
                                        create_graph=False, allow_unused=False)
        flat = torch.cat([gradient.detach().reshape(-1) for gradient in gradients])
        _need(flat.dtype == torch.float32 and bool(torch.isfinite(flat).all()),
              "nonfinite or non-FP32 batch gradient")
        losses[view_index] = float(loss.detach().item())
        if view_index == 0:
            first, accumulated = flat, flat.clone()
        else:
            accumulated.add_(flat)
        del gradients, flat, loss, logits
    if len(views) == 4:
        accumulated.div_(4)
    _need(bool(torch.isfinite(accumulated).all()), "nonfinite averaged gradient")
    delivery = accumulated if policy == "raw4" else first
    delivery_norm = _norm(delivery)
    observation_norm = 0.0
    if policy == "native1":
        observation_norm = _norm(first)
        neural_core.set_grad(model, delivery)
        tracker.filter_grad()
    elif policy == "observer4":
        observation_norm = _norm(accumulated)
        tracker.step_count += 1
        tracker._update_svd(accumulated)
        applied = tracker._project_gradient(delivery) if tracker.step_count > tracker.warmup else delivery
        neural_core.set_grad(model, applied)
    else:
        neural_core.set_grad(model, delivery)
    if tracker is not None:
        _need(tracker.step_count == before_counts[0] + 1, "observer must advance exactly once")
    applied_norm = _norm(neural_core.flat_grad(model))
    optimizer.step()
    after_counts = _adam_counts(optimizer, parameters)
    _need(all(after == before + 1 for before, after in zip(before_counts, after_counts)),
          "every Adam counter must advance exactly once")
    after = neural_core.flat_params(model)
    _need(bool(torch.isfinite(after).all()), "nonfinite parameters after update")
    return {
        "losses": losses,
        "gradient_evaluations": len(views),
        "observer_steps": 0 if tracker is None else int(tracker.step_count),
        "adam_steps": int(after_counts[0]),
        "basis_rank": 0 if tracker is None or tracker.V is None else int(tracker.V.shape[1]),
        "observation_norm": observation_norm,
        "delivery_norm": delivery_norm,
        "applied_norm": applied_norm,
        "data_step_norm": _norm(after - decay_base),
    }
