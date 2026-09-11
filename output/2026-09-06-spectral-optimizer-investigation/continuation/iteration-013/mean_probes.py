"""I13 isolated alignment and first-step probes; import performs no work."""
from __future__ import annotations

import json
import math
from typing import Any, Callable

import torch

import mean_core


i9 = mean_core.i9
i10 = mean_core.i10
OBJECTIVES = ("fixed", "soft", "redraw")
DELIVERY_POLICIES = ("current32", "mean32", "leak01_32")
PROBE_PLAN_KEYS = ("loss_train", "loss_aux", "utility_aux")
BRANCH_PLAN_KEYS = ("seed", "parent_step", "batches", "redraw_mask", "redraw_digits")
LOSS_NAMES = ("train_fixed", "train_soft", "train_clean", "aux_clean", "aux_soft")


class MeanProbeError(ValueError):
    pass


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise MeanProbeError(message)


def _tick(check: Callable[[], None] | None) -> None:
    _need(check is None or callable(check), "check must be callable or None")
    if check is not None:
        check()


def _cpu(value: torch.Tensor) -> torch.Tensor:
    return value.detach().cpu().clone()


def _same_tensor(left: torch.Tensor, right: torch.Tensor) -> bool:
    return (type(left) is torch.Tensor and type(right) is torch.Tensor
            and left.dtype == right.dtype and tuple(left.shape) == tuple(right.shape)
            and torch.equal(left, right))


def _validate_data(state: dict[str, Any], data: dict[str, torch.Tensor]) -> torch.device:
    _need(type(data) is dict and tuple(data) == i9.DATA_KEYS, "data keys/order differ")
    _need(type(state) is dict and type(state.get("model_spec")) is dict,
          "state model specification is missing")
    device = data["x"].device if type(data["x"]) is torch.Tensor else None
    _need(type(device) is torch.device and all(type(value) is torch.Tensor
          and value.device == device for value in data.values()),
          "data must be tensors on one device")
    _need(data["x"].ndim >= 2 and len(data["x"]) > 0
          and data["clean"].dtype == torch.long and data["noisy"].dtype == torch.long
          and data["clean"].ndim == data["noisy"].ndim == 1
          and len(data["clean"]) == len(data["noisy"]) == len(data["x"]),
          "training data topology differs")
    _need(data["ax"].ndim >= 2 and len(data["ax"]) > 0
          and data["ay"].dtype == torch.long and data["ay"].ndim == 1
          and len(data["ax"]) == len(data["ay"]), "auxiliary data topology differs")
    classes = state["model_spec"].get("classes")
    _need(type(classes) is int and classes >= 2
          and bool((data["clean"] >= 0).all()) and bool((data["clean"] < classes).all())
          and bool((data["noisy"] >= 0).all()) and bool((data["noisy"] < classes).all())
          and bool((data["ay"] >= 0).all()) and bool((data["ay"] < classes).all()),
          "labels or model class count are invalid")
    return device


def _probe_indices(probe_plan: dict[str, Any], data: dict[str, torch.Tensor],
                   device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _need(type(probe_plan) is dict and tuple(probe_plan) == PROBE_PLAN_KEYS,
          "probe plan keys/order differ")
    train = i9._indices(probe_plan["loss_train"], (1024,), len(data["x"]), device)
    auxiliary_loss = i9._indices(
        probe_plan["loss_aux"], (1024,), len(data["ax"]), device)
    auxiliary_utility = i9._indices(
        probe_plan["utility_aux"], (1024,), len(data["ax"]), device)
    return train, auxiliary_loss, auxiliary_utility


def _cosine(left: torch.Tensor, right: torch.Tensor) -> dict[str, Any]:
    left_norm, right_norm = i9._norm(left), i9._norm(right)
    dot = i9._dot(left, right)
    if left_norm == 0 or right_norm == 0:
        return {"gradient_norm": right_norm, "dot": dot, "cosine": None,
                "reason": "zero_complement_mean" if left_norm == 0 else "zero_gradient"}
    cosine = dot / (left_norm * right_norm)
    _need(math.isfinite(cosine), "nonfinite alignment cosine")
    return {"gradient_norm": right_norm, "dot": dot, "cosine": cosine, "reason": None}


def _tensor_digests(values: dict[str, Any]) -> dict[str, str]:
    result = {}
    for key, value in values.items():
        result[key] = i9.tree_digest(value)
    return result


def alignment_probe(state: dict[str, Any], data: dict[str, torch.Tensor],
                    probe_plan: dict[str, Any], *,
                    check: Callable[[], None] | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    """Measure frozen incumbent outside-mean alignment without observing a gradient."""
    state_digest = i9.tree_digest(state)
    caller_rng = i9._rng_state()
    device = _validate_data(state, data)
    try:
        _tick(check)
        model, optimizer, tracker = i9.restore(state, device)
        _need(tracker is not None and tracker.grad_mean is not None,
              "alignment probe requires an initialized observer mean")
        basis = i9._basis(tracker)
        _need(basis is not None and basis.ndim == 2 and basis.shape[0] == i9.flat_params(model).numel(),
              "alignment probe requires an incumbent basis")
        mean = tracker.grad_mean.detach().clone()
        mean_projection = i9._project(mean, basis)
        _need(mean_projection is not None, "mean projection is unavailable")
        complement = mean - mean_projection
        train, _, auxiliary = _probe_indices(probe_plan, data, device)
        classes = state["model_spec"]["classes"]
        inputs = {
            "train_fixed": (data["x"][train], data["noisy"][train]),
            "train_soft": (data["x"][train], i9._soft(data["clean"][train], classes)),
            "train_clean": (data["x"][train], data["clean"][train]),
            "aux_clean": (data["ax"][auxiliary], data["ay"][auxiliary]),
        }
        gradients = {}
        for name, arguments in inputs.items():
            _tick(check)
            gradients[name] = i9.gradient(model, *arguments)
            _tick(check)
        gradients["fixed_minus_soft"] = gradients["train_fixed"] - gradients["train_soft"]

        tensors = {
            "incumbent_mean": _cpu(mean),
            "mean_projection": _cpu(mean_projection),
            "complement_mean": _cpu(complement),
            "gradients": {name: _cpu(value) for name, value in gradients.items()},
            "loss_train_indices": _cpu(train),
            "utility_aux_indices": _cpu(auxiliary),
        }
        alignments = {name: _cosine(complement, value) for name, value in gradients.items()}
        record = {
            "schema": "i13_alignment_probe_v1",
            "status": "complete",
            "state_sha256": state_digest,
            "observer_step": int(tracker.step_count),
            "basis_rank": int(basis.shape[1]),
            "complement_mean_norm": i9._norm(complement),
            "alignments": alignments,
            "tensor_digests": {
                "incumbent_mean": i9.tree_digest(tensors["incumbent_mean"]),
                # Bind the complete incumbent basis without duplicating ~6.5 MiB
                # in every alignment artifact. The runner retains full bound
                # states only at selected horizons; a digest-only intermediate
                # record is therefore not independently projection-replayable.
                "incumbent_basis": i9.tree_digest(_cpu(basis)),
                "mean_projection": i9.tree_digest(tensors["mean_projection"]),
                "complement_mean": i9.tree_digest(tensors["complement_mean"]),
                "gradients": _tensor_digests(tensors["gradients"]),
            },
            "tensor_payload_sha256": i9.tree_digest(tensors),
        }
        json.dumps(record, ensure_ascii=True, allow_nan=False, separators=(",", ":"))
        _tick(check)
        return record, tensors
    finally:
        i9._restore_rng(caller_rng)
        _need(i9.tree_digest(state) == state_digest, "alignment probe changed caller state")


def _first_target(state: dict[str, Any], data: dict[str, torch.Tensor],
                  branch_plan: dict[str, Any], objective: str,
                  device: torch.device) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    _need(type(branch_plan) is dict and tuple(branch_plan) == BRANCH_PLAN_KEYS,
          "branch plan keys/order differ")
    _need(type(branch_plan["seed"]) is int and type(branch_plan["parent_step"]) is int,
          "branch plan identity is invalid")
    batches = torch.as_tensor(branch_plan["batches"])
    masks = torch.as_tensor(branch_plan["redraw_mask"])
    digits = torch.as_tensor(branch_plan["redraw_digits"])
    _need(tuple(batches.shape) == (500, 64)
          and tuple(masks.shape) == (500, 64) and tuple(digits.shape) == (500, 64)
          and masks.dtype == torch.bool
          and batches.dtype in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
          and digits.dtype in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64),
          "branch plan tensor topology differs")
    _need(bool((batches >= 0).all()) and bool((batches < len(data["x"])).all()),
          "branch plan batches are out of range")
    classes = state["model_spec"]["classes"]
    _need(bool((digits >= 0).all()) and bool((digits < classes).all()),
          "branch plan redraw digits are out of range")
    indices = batches[0].to(device=device, dtype=torch.long)
    _need(bool((indices >= 0).all()) and bool((indices < len(data["x"])).all()),
          "first branch batch is out of range")
    clean = data["clean"][indices]
    if objective == "fixed":
        target = data["noisy"][indices]
    elif objective == "soft":
        target = i9._soft(clean, state["model_spec"]["classes"])
    else:
        first_mask = masks[0].to(device=device)
        first_digits = digits[0].to(device=device, dtype=torch.long)
        target = torch.where(first_mask, first_digits, clean)
    identity = {"seed": branch_plan["seed"], "parent_step": branch_plan["parent_step"],
                "first_batch_indices_sha256": i9.tree_digest(_cpu(indices)),
                "first_target_sha256": i9.tree_digest(_cpu(target))}
    return indices, target, identity


def _loss_inputs(state: dict[str, Any], data: dict[str, torch.Tensor],
                 train: torch.Tensor, auxiliary: torch.Tensor) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    classes = state["model_spec"]["classes"]
    return {
        "train_fixed": (data["x"][train], data["noisy"][train]),
        "train_soft": (data["x"][train], i9._soft(data["clean"][train], classes)),
        "train_clean": (data["x"][train], data["clean"][train]),
        "aux_clean": (data["ax"][auxiliary], data["ay"][auxiliary]),
        "aux_soft": (data["ax"][auxiliary], i9._soft(data["ay"][auxiliary], classes)),
    }


def _evaluate_at(state: dict[str, Any], device: torch.device, parameters: torch.Tensor,
                 inputs: dict[str, tuple[torch.Tensor, torch.Tensor]],
                 check: Callable[[], None] | None) -> dict[str, float]:
    _tick(check)
    model, _, _ = i9.restore(state, device)
    i9.set_params(model, parameters)
    values = {name: i9._loss_value(model, *inputs[name]) for name in LOSS_NAMES}
    _tick(check)
    return values


def _changes(values: dict[str, float], baseline: dict[str, float]) -> dict[str, float]:
    result = {name: values[name] - baseline[name] for name in LOSS_NAMES}
    _need(all(math.isfinite(value) for value in result.values()), "nonfinite loss change")
    return result


def paired_step_probe(state: dict[str, Any], data: dict[str, torch.Tensor],
                      branch_plan: dict[str, Any], probe_plan: dict[str, Any],
                      objective: str, *,
                      check: Callable[[], None] | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    """Compare three inherited-Adam proposals and reciprocal direct norm controls at h0."""
    _need(type(objective) is str and objective in OBJECTIVES, "unknown objective")
    state_digest = i9.tree_digest(state)
    caller_rng = i9._rng_state()
    device = _validate_data(state, data)
    try:
        train, auxiliary, _ = _probe_indices(probe_plan, data, device)
        indices, target, plan_identity = _first_target(state, data, branch_plan, objective, device)
        loss_inputs = _loss_inputs(state, data, train, auxiliary)
        base_model, _, _ = i9.restore(state, device)
        theta = i9.flat_params(base_model)
        baseline = {name: i9._loss_value(base_model, *loss_inputs[name]) for name in LOSS_NAMES}

        observations: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
        post_states, totals, after_parameters = {}, {}, {}
        for policy in DELIVERY_POLICIES:
            _tick(check)
            model, optimizer, tracker = i9.restore(state, device)
            before_parameters = i9.flat_params(model)
            before_optimizer = i9._cpu_clone(optimizer.state_dict())
            observation = mean_core.observe_delivery(
                model, optimizer, tracker, data["x"][indices], target, policy)
            _need(type(observation) is tuple and len(observation) == 2,
                  "observe_delivery result topology differs")
            observation_record, observation_tensors = observation
            _need(i9.equal_tree(before_parameters, i9.flat_params(model))
                  and i9.equal_tree(before_optimizer, i9._cpu_clone(optimizer.state_dict())),
                  "observe_delivery changed parameters or Adam state")
            post_states[policy] = i9.snapshot(model, optimizer, tracker)
            observations[policy] = (observation_record, observation_tensors)
            delivered = observation_tensors["delivered_gradient"]
            totals[policy], after_parameters[policy] = i9._adam_delta(
                post_states[policy], device, delivered.to(device))
            _tick(check)

        common_keys = ("pre_mean", "raw_gradient", "native_projection", "post_mean", "post_basis",
                       "mean32_delivery", "leak01_32_delivery")
        reference_tensors = observations["current32"][1]
        for policy in DELIVERY_POLICIES[1:]:
            candidate = observations[policy][1]
            _need(all(_same_tensor(reference_tensors[key], candidate[key]) for key in common_keys),
                  "delivery policies did not share an exact observation")
        _need(_same_tensor(reference_tensors["native_projection"],
                           observations["current32"][1]["delivered_gradient"]),
              "current32 delivery differs from native projection")

        decay = i9.LR * i9.WD * theta
        data_deltas = {name: totals[name] + decay for name in DELIVERY_POLICIES}
        direct_specs = (
            ("mean_to_current_data_norm", "mean32", "current32"),
            ("current_to_mean_data_norm", "current32", "mean32"),
            ("leak_to_current_data_norm", "leak01_32", "current32"),
            ("current_to_leak_data_norm", "current32", "leak01_32"),
        )
        scales, direct_names = {}, []
        for name, source, target_name in direct_specs:
            direct_names.append(name)
            scaled, scale = i9.rescale_to_norm(data_deltas[source], i9._norm(data_deltas[target_name]))
            scales[name] = scale
            if scaled is None:
                totals[name] = data_deltas[name] = after_parameters[name] = None
                continue
            totals[name], after_parameters[name] = i9._direct_delta(
                state, device, theta, scaled - decay)
            data_deltas[name] = totals[name] + decay
            _need(math.isclose(i9._norm(data_deltas[name]), i9._norm(data_deltas[target_name]),
                               rel_tol=5e-5, abs_tol=1e-10),
                  "direct data norm differs after parameter quantization")

        intervention_records = {}
        for name in (*DELIVERY_POLICIES, *direct_names):
            if totals[name] is None:
                intervention_records[name] = {
                    "status": "undefined", "reason": scales[name]["reason"],
                    "scale": scales[name], "total_norm": None, "data_norm": None,
                    "losses": None, "loss_changes": None,
                }
                continue
            values = _evaluate_at(state, device, after_parameters[name], loss_inputs, check)
            intervention_records[name] = {
                "status": "defined", "reason": None, "scale": scales.get(name),
                "total_norm": i9._norm(totals[name]), "data_norm": i9._norm(data_deltas[name]),
                "losses": values, "loss_changes": _changes(values, baseline),
            }

        tensors = {
            "first_batch_indices": _cpu(indices), "first_target": _cpu(target),
            "pre_mean": _cpu(reference_tensors["pre_mean"]),
            "raw_gradient": _cpu(reference_tensors["raw_gradient"]),
            "native_projection": _cpu(reference_tensors["native_projection"]),
            "post_mean": _cpu(reference_tensors["post_mean"]),
            "post_basis": _cpu(reference_tensors["post_basis"]),
            "delivered_gradients": {
                policy: _cpu(observations[policy][1]["delivered_gradient"])
                for policy in DELIVERY_POLICIES},
            "interventions": {name: {
                "total_delta": None if totals[name] is None else _cpu(totals[name]),
                "data_delta": None if data_deltas[name] is None else _cpu(data_deltas[name]),
            } for name in (*DELIVERY_POLICIES, *direct_names)},
        }
        record = {
            "schema": "i13_paired_step_probe_v1", "status": "complete",
            "state_sha256": state_digest, "objective": objective,
            "plan": plan_identity, "baseline_losses": baseline,
            "delivery_observations": {policy: observations[policy][0]
                                      for policy in DELIVERY_POLICIES},
            "interventions": intervention_records,
            "tensor_digests": {
                "raw_gradient": i9.tree_digest(tensors["raw_gradient"]),
                "post_basis": i9.tree_digest(tensors["post_basis"]),
                "delivered_gradients": _tensor_digests(tensors["delivered_gradients"]),
                "interventions": {name: i9.tree_digest(value)
                                  for name, value in tensors["interventions"].items()},
            },
            "tensor_payload_sha256": i9.tree_digest(tensors),
        }
        json.dumps(record, ensure_ascii=True, allow_nan=False, separators=(",", ":"))
        _tick(check)
        return record, tensors
    finally:
        i9._restore_rng(caller_rng)
        _need(i9.tree_digest(state) == state_digest, "paired step probe changed caller state")
