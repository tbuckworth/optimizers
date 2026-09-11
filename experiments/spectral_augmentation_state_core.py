"""Inert fixed-state measurements; no files, acquisition, or import-time work.

The caller supplies private restored models and enforces the six-parent study
roster. Gradients use autograd.grad without touching .grad. Candidate optimizer
and observer changes occur only on jointly deep-copied objects, then discarded.
All returned arrays are CPU NumPy arrays. The finite-view geometry is empirical,
not a population signal/noise identification or a trajectory mechanism claim.
"""
from __future__ import annotations

import copy

import numpy as np
import torch
from torch.nn import functional as F


CANDIDATES = ("raw", "native", "raw_to_native", "native_to_raw", "decay")
STEP_SCALES = (1.0, 0.1)
MAX_RESCALE = 100.0
BASIS_CAP = 32
GEOMETRY_ATOL = 1e-10
GEOMETRY_RTOL = 1e-9
GRADIENT_ATOL, GRADIENT_RTOL = 1e-6, 5e-5
MATCH_ATOL, MATCH_RTOL = 1e-7, 5e-5


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _cpu(value):
    result = value.detach().cpu().contiguous().numpy().copy()
    _need(np.isfinite(result).all(), "nonfinite returned array")
    return result


def _parameters(model):
    parameters = tuple(model.parameters())
    _need(parameters and all(p.requires_grad and p.dtype == torch.float32 for p in parameters),
          "all parameters must be trainable float32")
    _need(len({p.device for p in parameters}) == 1, "parameter devices differ")
    return parameters


def _flat(values):
    return torch.cat([value.detach().reshape(-1) for value in values]).clone()


@torch.no_grad()
def _set_parameters(model, flat):
    parameters = _parameters(model)
    _need(flat.ndim == 1 and flat.numel() == sum(p.numel() for p in parameters), "parameter vector shape")
    offset = 0
    for parameter in parameters:
        parameter.copy_(flat[offset:offset + parameter.numel()].reshape_as(parameter).to(parameter))
        offset += parameter.numel()


def _set_gradient(model, flat):
    parameters = _parameters(model)
    _need(flat.ndim == 1 and flat.numel() == sum(p.numel() for p in parameters), "gradient vector shape")
    offset = 0
    for parameter in parameters:
        parameter.grad = flat[offset:offset + parameter.numel()].reshape_as(parameter).to(parameter).clone()
        offset += parameter.numel()


def _gradient(model, inputs, labels, objective_float64=False):
    logits = model(inputs)
    _need(logits.ndim == 2 and logits.shape[0] == len(labels) and bool(torch.isfinite(logits).all()), "finite logit schema")
    loss = F.cross_entropy(logits.double() if objective_float64 else logits, labels)
    gradient = torch.autograd.grad(loss, _parameters(model), retain_graph=False,
                                   create_graph=False, allow_unused=False)
    return _cpu(_flat(gradient)), _cpu(logits)


def collect_gradients(model, views, labels, check=None):
    """views[B,V,D], labels[B] -> per_example[B,V,P], exact batch[V,P].

    Slot zero is the original input; remaining slots are translated views.
    Each batch gradient is independently differentiated with the same float32
    mean CE as training. Floating-point reduction can differ from the average
    of separately differentiated per-example gradients; return those residuals.
    The frozen stateless MLP is assumed: stateful/stochastic layers are outside
    this helper's experiment contract. Existing parameter .grad values survive.
    """
    check = check or (lambda: None)
    _need(isinstance(views, torch.Tensor) and views.dtype == torch.float32
          and views.ndim == 3 and views.shape[0] > 0 and views.shape[1] >= 2, "views must be float32[B,V,D]")
    _need(isinstance(labels, torch.Tensor) and labels.dtype == torch.int64
          and labels.shape == (views.shape[0],), "label shape/dtype")
    _need(bool(torch.isfinite(views).all()), "nonfinite views")
    p = sum(parameter.numel() for parameter in _parameters(model))
    b, v = views.shape[:2]
    individual = np.empty((b, v, p), dtype=np.float32)
    batch = np.empty((v, p), dtype=np.float32)
    for view in range(v):
        check()
        batch[view], _ = _gradient(model, views[:, view], labels)
        for example in range(b):
            check()
            individual[example, view], _ = _gradient(model, views[example:example + 1, view], labels[example:example + 1])
    error = np.linalg.norm(individual.astype(np.float64).mean(axis=0) - batch, axis=1)
    norms = np.linalg.norm(batch.astype(np.float64), axis=1)
    relative = np.divide(error, norms, out=np.zeros_like(error), where=norms > 0)
    _need(np.all(error <= GRADIENT_ATOL + GRADIENT_RTOL * norms), "per-example/batch gradient disagreement")
    return {"per_example": individual, "batch": batch, "mean_error_norm": error,
            "mean_relative_error": relative}


def objective_readouts(model, views, labels, check=None):
    """Return objective_grads[O,P] and baseline_logits[O,N,C], both float32.

    Objectives use mean CE computed in float64 from the model's float32 logits,
    then differentiate through that cast. The returned parameter gradients are
    float32; finite CE readouts use the same float64-logit objective definition.
    """
    check = check or (lambda: None)
    _need(isinstance(views, torch.Tensor) and views.dtype == torch.float32
          and views.ndim == 3 and all(size > 0 for size in views.shape), "objective views shape/dtype")
    _need(labels.dtype == torch.int64 and labels.shape == (views.shape[1],), "objective labels")
    gradients, logits = [], []
    for inputs in views:
        check()
        gradient, values = _gradient(model, inputs, labels, objective_float64=True)
        gradients.append(gradient)
        logits.append(values)
    return {"objective_grads": np.stack(gradients), "baseline_logits": np.stack(logits)}


def empirical_geometry(per_example, basis):
    """Exact finite translated-grid total=between+within trace accounting.

    Input is [B,V,P], original slot0 excluded from the R=V-1 translated grid.
    Denominators are B*R, B, and B*R, respectively (not unbiased estimates).
    ``basis`` is recorded stable V, not a newly orthogonalized matrix. Report
    the prespecified Q^T coordinate energies, plus separate actual QQ^T operator
    energies using the Gram matrix. Both identities hold for fixed linear maps,
    even with small recorded orthogonality error; they are not silently equated.
    """
    values = np.asarray(per_example, dtype=np.float64)
    q = np.asarray(basis, dtype=np.float64)
    _need(values.ndim == 3 and values.shape[0] > 0 and values.shape[1] >= 2
          and values.shape[2] > 0 and np.isfinite(values).all(), "gradient grid schema")
    _need(q.ndim == 2 and q.shape[0] == values.shape[2] and np.isfinite(q).all(), "basis schema")
    b, v, p = values.shape
    r = v - 1
    translated = values[:, 1:]
    example_means = translated.mean(axis=1)
    grand_mean = example_means.mean(axis=0)
    original_mean = values[:, 0].mean(axis=0)
    gram = q.T @ q

    def energy(vectors):
        matrix = vectors.reshape(-1, p)
        coordinates = matrix @ q
        total = float(np.square(matrix).sum(axis=1).mean())
        coordinate = float(np.square(coordinates).sum(axis=1).mean())
        kept = float(np.einsum("nk,kl,nl->n", coordinates, gram, coordinates).mean())
        return {"trace": total, "coordinate_trace": coordinate,
                "coordinate_retention": None if total == 0 else coordinate / total,
                "operator_trace": kept,
                "operator_retention": None if total == 0 else kept / total}

    total = energy(translated - grand_mean)
    between = energy(example_means - grand_mean)
    within = energy(translated - example_means[:, None])
    residual = total["trace"] - between["trace"] - within["trace"]
    coordinate_residual = total["coordinate_trace"] - between["coordinate_trace"] - within["coordinate_trace"]
    kept_residual = total["operator_trace"] - between["operator_trace"] - within["operator_trace"]
    passed = all(abs(error) <= GEOMETRY_ATOL + GEOMETRY_RTOL * abs(reference)
                 for error, reference in ((residual, total["trace"]),
                                          (coordinate_residual, total["coordinate_trace"]),
                                          (kept_residual, total["operator_trace"])))
    return {"examples": b, "translated_views": r,
            "denominators": {"total": b * r, "between": b, "within": b * r},
            "basis_rank": q.shape[1],
            "basis_gram_max_abs_error": float(np.max(np.abs(gram - np.eye(q.shape[1])), initial=0)),
            "total": total, "between": between, "within": within,
            "original_centered": energy(values[:, 0] - original_mean),
            "original_mean": energy(original_mean), "translated_mean": energy(grand_mean),
            "mean_translation_change": energy(grand_mean - original_mean),
            "identity_residual": residual, "operator_identity_residual": kept_residual,
            "coordinate_identity_residual": coordinate_residual,
            "identity_pass": bool(passed)}


def _basis(tracker, p):
    result = np.zeros((p, BASIS_CAP), dtype=np.float32)
    value = tracker.V
    if value is None:
        return result, 0
    k = value.shape[1] if tracker.proj_k is None else min(value.shape[1], tracker.proj_k)
    _need(value.ndim == 2 and value.shape[0] == p and k <= BASIS_CAP, "basis exceeds fixed cap")
    result[:, :k] = _cpu(value[:, :k])
    return result, k


def _adam_state(model, optimizer):
    _need(isinstance(optimizer, torch.optim.AdamW) and len(optimizer.param_groups) == 1, "one-group AdamW required")
    parameters = _parameters(model)
    group = optimizer.param_groups[0]
    _need(len(group["params"]) == len(parameters)
          and all(left is right for left, right in zip(group["params"], parameters)), "optimizer parameter order mismatch")
    _need(not any(group.get(key, False) for key in ("amsgrad", "maximize", "capturable", "differentiable", "foreach", "fused")),
          "unsupported AdamW option")
    moments, variances, steps = [], [], []
    for parameter in parameters:
        state = optimizer.state.get(parameter)
        _need(isinstance(state, dict) and set(state) == {"step", "exp_avg", "exp_avg_sq"},
              "populated canonical Adam state required")
        step = float(state["step"].item())
        _need(step >= 0 and step.is_integer(), "integer Adam counter required")
        moments.append(state["exp_avg"])
        variances.append(state["exp_avg_sq"])
        steps.append(int(step))
    return parameters, group, _flat(moments), _flat(variances), np.array(steps, dtype=np.int64)


def propose_actions(model, optimizer, tracker, batch_gradients, check=None):
    """Private raw/native Adam candidates and post-Adam data-norm controls.

    Every native clone observes its supplied gradient once, starting from the
    identical parent tracker; raw clones never observe. Actual FP32 decay base
    is theta*(1-lr*wd), not an Adam zero-gradient update. data=after-decay_base.
    Rescaling uses float64 norms/multiplication then casts to FP32. Zero source
    or target norm, or scale>100, makes only that matched candidate unavailable.
    Invalid slots are finite zero fillers and must be excluded via valid.

    Raw/native full-scale after_parameters are the exact optimizer outputs.
    Matched full-scale parameters are FP32 decay_base+planned_data. The .1 path
    is FP32 theta+.1*(full_after-theta), scaling the TOTAL realized displacement
    (including decay), not another optimizer step. Source objects are untouched.
    """
    check = check or (lambda: None)
    parameters, group, m, second, steps = _adam_state(model, optimizer)
    theta_device = _flat(parameters)
    p = theta_device.numel()
    gradients = np.asarray(batch_gradients)
    _need(gradients.dtype == np.float32 and gradients.ndim == 2 and gradients.shape[1] == p
          and gradients.shape[0] > 0 and np.isfinite(gradients).all(), "batch gradient schema")
    _need(tracker.model is model and tracker.base_optimizer is optimizer, "tracker ownership mismatch")
    theta = _cpu(theta_device)
    with torch.no_grad():
        decay_base = torch.cat([(parameter.detach().clone().mul_(1 - group["lr"] * group["weight_decay"]))
                                .reshape(-1) for parameter in parameters])
    v = gradients.shape[0]
    old_basis, old_rank = _basis(tracker, p)
    output = {"theta": theta, "adam_m": _cpu(m), "adam_v": _cpu(second), "adam_steps": steps,
              "param_sizes": np.array([parameter.numel() for parameter in parameters], dtype=np.int64),
              "basis_before": old_basis, "basis_before_rank": np.array(old_rank, dtype=np.int64),
              "basis_after": np.zeros((v, p, BASIS_CAP), dtype=np.float32),
              "basis_after_ranks": np.zeros(v, dtype=np.int64),
              "applied_grads": np.zeros((v, 2, p), dtype=np.float32),
              "planned_data": np.zeros((v, len(CANDIDATES), p), dtype=np.float32),
              "after_parameters": np.zeros((v, len(CANDIDATES), len(STEP_SCALES), p), dtype=np.float32),
              "valid": np.ones((v, len(CANDIDATES)), dtype=np.bool_),
              "scales": np.ones((v, len(CANDIDATES)), dtype=np.float64),
              "target_norm": np.zeros((v, len(CANDIDATES)), dtype=np.float64),
              "match_norm_error": np.zeros((v, len(CANDIDATES)), dtype=np.float64)}
    output["scales"][:, 4] = 0.
    for view, gradient in enumerate(gradients):
        check()
        after = []
        for action in range(2):
            check()
            if action == 0:
                cloned_model, cloned_optimizer = copy.deepcopy((model, optimizer))
                cloned_tracker = None
            else:
                cloned_model, cloned_optimizer, cloned_tracker = copy.deepcopy((model, optimizer, tracker))
                _need(cloned_tracker.model is cloned_model and cloned_tracker.base_optimizer is cloned_optimizer,
                      "private clone aliases differ")
            _set_gradient(cloned_model, torch.from_numpy(gradient).to(theta_device))
            if cloned_tracker is not None:
                previous = cloned_tracker.step_count
                cloned_tracker.filter_grad()
                _need(cloned_tracker.step_count == previous + 1, "observer must advance exactly once")
                output["basis_after"][view], output["basis_after_ranks"][view] = _basis(cloned_tracker, p)
            output["applied_grads"][view, action] = _cpu(_flat([parameter.grad for parameter in cloned_model.parameters()]))
            cloned_optimizer.step()
            _need(all(float(cloned_optimizer.state[parameter]["step"].item()) == int(parent_step) + 1
                      for parameter, parent_step in zip(_parameters(cloned_model), steps)),
                  "each cloned Adam counter must advance exactly once")
            after.append(_flat(_parameters(cloned_model)))
            del cloned_model, cloned_optimizer, cloned_tracker
        actual_data = [value - decay_base for value in after]
        norms = [float(torch.linalg.vector_norm(value.double()).item()) for value in actual_data]
        full = after + [None, None, decay_base]
        for action in range(2):
            output["planned_data"][view, action] = _cpu(actual_data[action])
            output["target_norm"][view, action] = norms[action]
        for action, source, target in ((2, 0, 1), (3, 1, 0)):
            output["target_norm"][view, action] = norms[target]
            scale = None if norms[source] == 0 or norms[target] == 0 else norms[target] / norms[source]
            if scale is None or not np.isfinite(scale) or scale > MAX_RESCALE:
                output["valid"][view, action] = False
                output["scales"][view, action] = 0.
                continue
            output["scales"][view, action] = scale
            planned = (actual_data[source].double() * scale).to(dtype=torch.float32)
            output["planned_data"][view, action] = _cpu(planned)
            full[action] = decay_base + planned
            realized_norm = float(torch.linalg.vector_norm((full[action] - decay_base).double()).item())
            output["match_norm_error"][view, action] = realized_norm - norms[target]
            _need(abs(realized_norm - norms[target]) <= MATCH_ATOL + MATCH_RTOL * norms[target],
                  "materialized data norm mismatch")
        for action, value in enumerate(full):
            if value is None:
                continue
            output["after_parameters"][view, action, 0] = _cpu(value)
            output["after_parameters"][view, action, 1] = _cpu(theta_device + 0.1 * (value - theta_device))
    return output


def _stats(logits, labels):
    values = np.asarray(logits, dtype=np.float64)
    maximum = values.max(axis=-1)
    loss = maximum + np.log(np.exp(values - maximum[..., None]).sum(axis=-1)) - values[np.arange(len(labels)), labels]
    return float(loss.mean()), float((values.argmax(axis=-1) == labels).mean())


def measure_actions(model, proposals, objective_views, labels, scales=STEP_SCALES, check=None):
    """Measure private after-parameter models; all utility uses realized motion.

    Returned after_logits[V,A,S,O,N,C], objective_grads[O,P], baseline_logits,
    CE/accuracy/derivative_utility[V,A,S,O], and baseline CE/accuracy[O] are
    independently auditable. Invalid actions have finite zero readout fillers;
    proposals['valid'] controls availability. Neither inference nor set-parameter
    operations touch the supplied parent model. No optimizer state is updated.
    """
    check = check or (lambda: None)
    _need(tuple(scales) == STEP_SCALES, "only the fixed full/.1 total-displacement paths are allowed")
    private_model = copy.deepcopy(model)
    baseline = objective_readouts(private_model, objective_views, labels, check)
    target = _cpu(labels)
    parameters = np.asarray(proposals["after_parameters"])
    valid = np.asarray(proposals["valid"])
    _need(parameters.dtype == np.float32 and parameters.ndim == 4
          and parameters.shape[1:3] == (len(CANDIDATES), len(STEP_SCALES))
          and valid.shape == parameters.shape[:2] and valid.dtype == np.bool_, "proposal shape")
    v, a, s, p = parameters.shape
    o, n, c = baseline["baseline_logits"].shape
    after_logits = np.zeros((v, a, s, o, n, c), dtype=np.float32)
    ce = np.zeros((v, a, s, o), dtype=np.float64)
    accuracy, utility, data_utility = np.zeros_like(ce), np.zeros_like(ce), np.zeros_like(ce)
    baseline_ce, baseline_accuracy = zip(*(_stats(values, target) for values in baseline["baseline_logits"]))
    theta = np.asarray(proposals["theta"], dtype=np.float64)
    q = baseline["objective_grads"].astype(np.float64)
    for view in range(v):
        for action in range(a):
            if not valid[view, action]:
                continue
            for scale in range(s):
                check()
                _set_parameters(private_model, torch.from_numpy(parameters[view, action, scale]))
                delivered = parameters[view, action, scale].astype(np.float64) - theta
                utility[view, action, scale] = -(q @ delivered)
                data_delivered = (parameters[view, action, scale].astype(np.float64)
                                  - parameters[view, 4, scale].astype(np.float64))
                data_utility[view, action, scale] = -(q @ data_delivered)
                with torch.no_grad():
                    for objective, inputs in enumerate(objective_views):
                        check()
                        values = _cpu(private_model(inputs))
                        after_logits[view, action, scale, objective] = values
                        ce[view, action, scale, objective], accuracy[view, action, scale, objective] = _stats(values, target)
    return {**baseline, "after_logits": after_logits, "ce": ce, "accuracy": accuracy,
            "derivative_utility": utility, "data_derivative_utility": data_utility,
            "baseline_ce": np.asarray(baseline_ce),
            "baseline_accuracy": np.asarray(baseline_accuracy)}
