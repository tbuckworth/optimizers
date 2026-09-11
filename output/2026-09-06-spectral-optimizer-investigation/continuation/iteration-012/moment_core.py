"""Isolated I12 Adam-moment interventions over the frozen I9/I10 helpers."""
from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
import sys
from typing import Any, Callable, Iterable

import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
_I10_PATH = HERE.parent / "iteration-010" / "branch_core.py"
_I10_SPEC = importlib.util.spec_from_file_location("_i12_frozen_i10_branch_core", _I10_PATH)
if _I10_SPEC is None or _I10_SPEC.loader is None:  # pragma: no cover
    raise ImportError("cannot load frozen I10 branch core")
i10 = importlib.util.module_from_spec(_I10_SPEC)
_I10_SPEC.loader.exec_module(i10)
i9 = i10.i9


ARMS = ("zero_m", "zero_v", "zero_mv", "fresh_adam")
OBJECTIVES = ("fixed", "soft", "redraw")
DEFAULT_HORIZONS = (0, 1, 10, 50, 100, 250, 500)
_SYNTHETIC_INHERITED_SPEC = {"input_dim": 4, "width": 5, "classes": 3}
_STATE_KEYS = ("step", "exp_avg", "exp_avg_sq")
_NUMERIC_MESSAGES = {
    "gradient must be a finite dense tensor",
    "gradient vector must be a finite dense tensor",
    "updated parameters must be a finite dense tensor",
    "evaluation logits must be a finite dense tensor",
    "frozen projected gradient must be a finite dense tensor",
    "total displacement must be a finite dense tensor",
    "decay-adjusted data displacement must be a finite dense tensor",
    "nonfinite norm",
    "nonfinite dot product",
    "nonfinite evaluation loss",
    "nonfinite evaluation summary",
    "nonfinite evaluation record",
}
_JSON_NONFINITE_MESSAGES = frozenset({
    "Out of range float values are not JSON compliant",
    "Out of range float values are not JSON compliant: nan",
    "Out of range float values are not JSON compliant: inf",
    "Out of range float values are not JSON compliant: -inf",
})


class MomentCoreError(ValueError):
    pass


class _NumericalFailure(FloatingPointError):
    pass


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise MomentCoreError(message)


def _expected_parameter_shapes(spec: dict[str, Any]) -> tuple[tuple[int, ...], ...]:
    _need(type(spec) is dict and tuple(spec) == ("input_dim", "width", "classes"),
          "model specification topology differs")
    values = tuple(spec[key] for key in ("input_dim", "width", "classes"))
    _need(all(type(value) is int and value > 0 for value in values) and values[2] >= 2,
          "model specification is invalid")
    input_dim, width, classes = values
    return ((width, input_dim), (width,), (classes, width), (classes,))


def _optimizer_topology(state: dict[str, Any]) -> tuple[list[Any], int]:
    """Validate the exact initialized I9 AdamW state represented in a snapshot."""
    _need(type(state) is dict and tuple(state) ==
          ("schema", "model_spec", "model_state", "gradients", "model_modes",
           "optimizer", "tracker", "rng") and state["schema"] == "i9_neural_snapshot_v1",
          "state must be a complete I9-schema snapshot")
    expected_shapes = _expected_parameter_shapes(state["model_spec"])
    _need(tuple(state["model_state"]) == ("0.weight", "0.bias", "2.weight", "2.bias")
          and tuple(tuple(state["model_state"][name].shape) for name in state["model_state"])
          == expected_shapes, "model parameter topology differs")
    _need(type(state["gradients"]) is list and len(state["gradients"]) == len(expected_shapes),
          "gradient topology differs")

    optimizer = state["optimizer"]
    _need(type(optimizer) is dict and tuple(optimizer) == ("state", "param_groups"),
          "optimizer snapshot topology differs")
    groups = optimizer["param_groups"]
    _need(type(groups) is list and len(groups) == 1 and type(groups[0]) is dict,
          "optimizer must have exactly one parameter group")
    group = groups[0]
    required = {"params", "lr", "weight_decay", "betas", "eps", "amsgrad", "maximize",
                "foreach", "capturable", "differentiable", "fused"}
    _need(required.issubset(group), "AdamW parameter-group flags are incomplete")
    _need(group["lr"] == i9.LR and group["weight_decay"] == i9.WD
          and tuple(group["betas"]) == (0.9, 0.999) and group["eps"] == 1e-8
          and group["amsgrad"] is False and group["maximize"] is False
          and group["foreach"] is False and group["capturable"] is False
          and group["differentiable"] is False and group["fused"] is False,
          "optimizer configuration differs from I9 AdamW")
    if "decoupled_weight_decay" in group:
        _need(group["decoupled_weight_decay"] is True,
              "AdamW decoupled-weight-decay flag differs")
    parameter_ids = group["params"]
    _need(type(parameter_ids) is list and len(parameter_ids) == len(expected_shapes)
          and len(set(parameter_ids)) == len(parameter_ids),
          "optimizer parameter identifiers differ")
    states = optimizer["state"]
    _need(type(states) is dict and tuple(states) == tuple(parameter_ids),
          "optimizer state order/coverage differs")
    counters: list[int] = []
    for parameter_id, expected_shape in zip(parameter_ids, expected_shapes):
        row = states[parameter_id]
        _need(type(row) is dict and tuple(row) == _STATE_KEYS,
              "Adam parameter-state topology differs")
        step, first, second = (row[key] for key in _STATE_KEYS)
        _need(type(step) is torch.Tensor and step.layout == torch.strided
              and step.numel() == 1 and bool(torch.isfinite(step).all()),
              "Adam step must be one finite dense tensor scalar")
        scalar = float(step.item())
        _need(scalar >= 0 and scalar == round(scalar), "Adam step is not a nonnegative integer")
        counters.append(int(scalar))
        _need(type(first) is torch.Tensor and type(second) is torch.Tensor
              and first.layout == second.layout == torch.strided
              and tuple(first.shape) == tuple(second.shape) == expected_shape
              and first.dtype == second.dtype
              and bool(torch.isfinite(first).all()) and bool(torch.isfinite(second).all()),
              "Adam moment tensor topology or finiteness differs")
    _need(len(set(counters)) == 1, "Adam parameter counters differ")
    return parameter_ids, counters[0]


def _masked_digest(state: dict[str, Any], keys: frozenset[str]) -> str:
    masked = i9._cpu_clone(state)
    for row in masked["optimizer"]["state"].values():
        for key in keys:
            row[key] = "__I12_AUTHORIZED_MOMENT_FIELD__"
    return i9.tree_digest(masked)


def edit_moments(state: dict[str, Any], arm: str, *,
                 synthetic_fixture: bool = False) -> tuple[dict[str, Any], dict[str, Any]]:
    """Copy an I9 snapshot and zero only the Adam fields authorized by ``arm``."""
    parameter_ids, counter_before = _optimizer_topology(state)
    _need(type(arm) is str and (arm in ARMS or arm == "inherited"), "unknown moment arm")
    if arm == "inherited":
        _need(synthetic_fixture is True and state["model_spec"] == _SYNTHETIC_INHERITED_SPEC,
              "inherited is restricted to the explicit synthetic fixture")
    else:
        _need(synthetic_fixture is False,
              "synthetic_fixture is permitted only for the inherited test arm")
    zeroed = {
        "zero_m": frozenset(("exp_avg",)),
        "zero_v": frozenset(("exp_avg_sq",)),
        "zero_mv": frozenset(("exp_avg", "exp_avg_sq")),
        "fresh_adam": frozenset(("step", "exp_avg", "exp_avg_sq")),
        "inherited": frozenset(),
    }[arm]
    parent_digest = i9.tree_digest(state)
    unmodified_before = _masked_digest(state, zeroed)
    edited = i9._cpu_clone(state)
    for parameter_id in parameter_ids:
        for key in zeroed:
            edited["optimizer"]["state"][parameter_id][key].zero_()

    _, counter_after = _optimizer_topology(edited)
    _need(i9.tree_digest(state) == parent_digest, "moment edit mutated its parent")
    unmodified_after = _masked_digest(edited, zeroed)
    _need(unmodified_after == unmodified_before, "moment edit changed an unauthorized field")
    for parameter_id in parameter_ids:
        for key in zeroed:
            value = edited["optimizer"]["state"][parameter_id][key]
            _need(bool((value == 0).all()), "authorized Adam field was not exactly zeroed")
    audit = {
        "schema": "i12_moment_edit_audit_v1", "arm": arm,
        "parameter_states": len(parameter_ids), "zeroed_fields": sorted(zeroed),
        "targeted_tensor_fields": len(parameter_ids) * len(zeroed),
        "counter_before": counter_before, "counter_after": counter_after,
        "parent_digest": parent_digest, "edited_digest": i9.tree_digest(edited),
        "parent_unchanged": True, "unmodified_fields_digest": unmodified_after,
        "unmodified_fields_equal": True,
    }
    json.dumps(audit, ensure_ascii=True, allow_nan=False, separators=(",", ":"))
    return edited, audit


def _horizons(values: Iterable[int], steps: int) -> tuple[int, ...]:
    result = tuple(values)
    _need(result and all(type(value) is int for value in result)
          and result == tuple(sorted(set(result))) and result[0] == 0
          and result[-1] == steps and all(0 <= value <= steps for value in result),
          "evaluation horizons must be unique, increasing, start at zero and end at steps")
    return result


def _finite_live(model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                 tracker: Any, phase: str) -> None:
    def scan(value: Any, label: str) -> None:
        if isinstance(value, torch.Tensor):
            if value.layout != torch.strided:
                raise MomentCoreError(f"{phase}: nondense {label}")
            if not bool(torch.isfinite(value).all()):
                raise _NumericalFailure(f"{phase}: nonfinite {label}")
        elif type(value) is dict:
            for key, item in value.items():
                scan(item, f"{label}.{key}")
        elif type(value) in (list, tuple):
            for index, item in enumerate(value):
                scan(item, f"{label}[{index}]")
        elif type(value) is float and not math.isfinite(value):
            raise _NumericalFailure(f"{phase}: nonfinite {label}")

    for name, parameter in model.named_parameters():
        scan(parameter, "model." + name)
        if parameter.grad is not None:
            scan(parameter.grad, "gradient." + name)
    for name, buffer in model.named_buffers():
        scan(buffer, "buffer." + name)
    scan(optimizer.state_dict(), "optimizer")
    tracker_values = {key: value for key, value in vars(tracker).items()
                      if key not in ("model", "base_optimizer", "param_list")}
    scan(tracker_values, "tracker")


def _finite_record(value: Any, label: str = "diagnostic") -> None:
    if isinstance(value, torch.Tensor):
        if value.layout != torch.strided:
            raise MomentCoreError(label + " contains a nondense tensor")
        if not bool(torch.isfinite(value).all()):
            raise _NumericalFailure(label + " contains a nonfinite tensor")
    elif type(value) is dict:
        for key, item in value.items():
            _finite_record(item, f"{label}.{key}")
    elif type(value) in (list, tuple):
        for index, item in enumerate(value):
            _finite_record(item, f"{label}[{index}]")
    elif type(value) is float and not math.isfinite(value):
        raise _NumericalFailure(label + " contains a nonfinite scalar")


def _is_known_numeric(exc: BaseException) -> bool:
    return isinstance(exc, _NumericalFailure) or (
        isinstance(exc, (i9.NeuralCoreError, i10.BranchCoreError))
        and str(exc) in _NUMERIC_MESSAGES
    ) or (type(exc) is ValueError and str(exc) in _JSON_NONFINITE_MESSAGES)


def _target(data: dict[str, torch.Tensor], indices: torch.Tensor, objective: str,
            redraw_mask: torch.Tensor | None, redraw_digits: torch.Tensor | None,
            position: int) -> torch.Tensor:
    if objective == "fixed":
        return data["noisy"][indices]
    clean = data["clean"][indices]
    if objective == "soft":
        return i9._soft(clean, 10)
    assert redraw_mask is not None and redraw_digits is not None
    return torch.where(redraw_mask[position], redraw_digits[position], clean)


def run_branch(
    state: dict[str, Any],
    data: dict[str, torch.Tensor],
    plan: dict[str, Any],
    objective: str,
    policy: str,
    *,
    steps: int = 500,
    horizons: Iterable[int] = DEFAULT_HORIZONS,
    check: Callable[[], None] | None = None,
    expected_initial_evaluation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run one edited-state branch, sealing only explicitly detected nonfinites."""
    _optimizer_topology(state)
    _need(type(steps) is int and steps > 0, "steps must be positive")
    eval_horizons = _horizons(horizons, steps)
    _need(type(objective) is str and objective in OBJECTIVES, "unknown objective")
    _need(type(policy) is str and policy in i10.POLICIES, "unknown branch policy")
    _need(type(data) is dict and tuple(data) == i9.DATA_KEYS
          and all(type(value) is torch.Tensor for value in data.values()),
          "data topology differs")
    device = data["x"].device
    _need(all(value.device == device for value in data.values()),
          "all data tensors must share one device")
    _need(type(plan) is dict and "batches" in plan, "branch plan lacks batches")
    batches = i9._indices(plan["batches"], (steps, 64), len(data["x"]), device)
    redraw_mask = redraw_digits = None
    if objective == "redraw":
        _need("redraw_mask" in plan and "redraw_digits" in plan,
              "redraw plan fields are missing")
        redraw_mask = torch.as_tensor(plan["redraw_mask"], device=device)
        redraw_digits = i9._indices(plan["redraw_digits"], (steps, 64), 10, device)
        _need(redraw_mask.dtype == torch.bool and tuple(redraw_mask.shape) == (steps, 64),
              "redraw mask must be a boolean steps-by-64 tensor")

    parent_digest = i9.tree_digest(state)
    model, optimizer, tracker = i9.restore(state, device)
    _need(tracker is not None, "moment branch requires native observer state")
    _need(i9.tree_digest(i9.snapshot(model, optimizer, tracker)) == parent_digest,
          "restored starting state differs")
    i10._optimizer_is_i9(optimizer, model)
    frozen_basis = i9._basis(tracker)
    _need(frozen_basis is not None, "moment branch requires an initialized frozen basis")
    parent_counter = _optimizer_topology(i9.snapshot(model, optimizer, tracker))[1]
    parent_observer = int(tracker.step_count)
    parent_tracker_digest = i9.tree_digest(state["tracker"])

    curve: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    last_evaluated_state: dict[str, Any] | None = None
    last_evaluated_horizon: int | None = None
    first_step_applied_gradient_digest: str | None = None

    def failure(exc: BaseException, phase: str, attempted: int) -> dict[str, Any]:
        _need(i9.tree_digest(state) == parent_digest, "numeric branch mutated its parent")
        terminal = i9.snapshot(model, optimizer, tracker)
        return {
            "schema": "i12_moment_branch_v1", "status": "numerical_failure",
            "curve": curve, "steps": diagnostics, "terminal_state": terminal,
            "completed_steps": len(diagnostics), "attempted_step": attempted,
            "last_evaluated_state": last_evaluated_state,
            "last_evaluated_horizon": last_evaluated_horizon,
            "first_step_applied_gradient_digest": first_step_applied_gradient_digest,
            "numerical_failure": {"phase": phase, "exception_type": type(exc).__name__,
                                  "message": str(exc)},
        }

    if check is not None:
        check()
    # A nonfinite horizon-zero state is a broken bound parent, not an outcome of
    # this moment intervention; it must abort the encompassing run.
    _finite_live(model, optimizer, tracker, "initial_state")
    before_eval = i9.snapshot(model, optimizer, tracker)
    baseline = i10.evaluate(model, data)
    _finite_record(baseline, "initial_evaluation")
    _need(i9.equal_tree(before_eval, i9.snapshot(model, optimizer, tracker)),
          "horizon-zero evaluation changed state")
    if expected_initial_evaluation is not None:
        _need(type(expected_initial_evaluation) is dict,
              "expected initial evaluation must be a dictionary")
        _need(baseline == expected_initial_evaluation,
              "horizon-zero evaluation differs from the bound parent endpoint")
    curve.append({"horizon": 0, **baseline})
    last_evaluated_state, last_evaluated_horizon = before_eval, 0

    for horizon in range(1, steps + 1):
        if check is not None:
            check()
        indices = batches[horizon - 1]
        target = _target(data, indices, objective, redraw_mask, redraw_digits, horizon - 1)
        try:
            _finite_live(model, optimizer, tracker, f"pre_step_{horizon}")
            row = i10.branch_step(model, optimizer, tracker, data["x"][indices], target,
                                  policy, frozen_basis)
            _finite_live(model, optimizer, tracker, f"post_step_{horizon}")
            _finite_record(row)
            if horizon == 1:
                first_step_applied_gradient_digest = i9.tree_digest(i9.flat_grad(model))
        except BaseException as exc:
            if _is_known_numeric(exc):
                return failure(exc, "step", horizon)
            raise
        diagnostics.append({"horizon": horizon, **row})
        if horizon in eval_horizons:
            if check is not None:
                check()
            try:
                before_eval = i9.snapshot(model, optimizer, tracker)
                evaluation = i10.evaluate(model, data)
                _finite_record(evaluation, "evaluation")
            except BaseException as exc:
                if _is_known_numeric(exc):
                    return failure(exc, "evaluation", horizon)
                raise
            _need(i9.equal_tree(before_eval, i9.snapshot(model, optimizer, tracker)),
                  f"horizon-{horizon} evaluation changed state")
            curve.append({"horizon": horizon, **evaluation})
            last_evaluated_state, last_evaluated_horizon = before_eval, horizon

    final = i9.snapshot(model, optimizer, tracker)
    final_counter = _optimizer_topology(final)[1]
    _need(final_counter == parent_counter + steps,
          "final Adam counter is not parent plus branch steps")
    expected_observer = parent_observer if policy == "frozen32" else parent_observer + steps
    _need(int(final["tracker"]["step_count"]) == expected_observer,
          "final observer counter differs")
    if policy == "frozen32":
        _need(i9.tree_digest(final["tracker"]) == parent_tracker_digest,
              "frozen observer changed during branch")
    _need(i9.tree_digest(state) == parent_digest, "moment branch mutated its parent")
    _need([row["horizon"] for row in curve] == list(eval_horizons),
          "returned curve horizons differ")
    _need(len(diagnostics) == steps, "returned diagnostic count differs")
    return {
        "schema": "i12_moment_branch_v1", "status": "complete",
        "curve": curve, "steps": diagnostics, "terminal_state": final,
        "completed_steps": steps, "attempted_step": steps,
        "first_step_applied_gradient_digest": first_step_applied_gradient_digest,
        "numerical_failure": None,
    }


def main() -> int:
    print("moment_core: inert library; no moment branch executed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
