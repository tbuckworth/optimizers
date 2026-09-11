"""I13 mean-preserving and matched one-percent-leak gradient deliveries."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from typing import Any, Callable, Iterable

import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
_I12_PATH = HERE.parent / "iteration-012" / "moment_core.py"
_I12_SPEC = importlib.util.spec_from_file_location("_i13_frozen_i12_moment_core", _I12_PATH)
if _I12_SPEC is None or _I12_SPEC.loader is None:  # pragma: no cover
    raise ImportError("cannot load frozen I12 moment core")
i12 = importlib.util.module_from_spec(_I12_SPEC)
_I12_SPEC.loader.exec_module(i12)
i10, i9 = i12.i10, i12.i9


POLICIES = ("mean32", "leak01_32", "current32", "raw")
OBJECTIVES = i12.OBJECTIVES
DEFAULT_HORIZONS = i12.DEFAULT_HORIZONS
DECAY = 0.99
LEAK = 1.0 - DECAY
_PROBE_NONFINITE_MESSAGES = frozenset(("nonfinite alignment cosine",
                                       "nonfinite loss change"))
_PROBE_GRADIENT_NONFINITE_MESSAGE = "probe gradient must be a finite dense tensor"


class MeanCoreError(ValueError):
    pass


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise MeanCoreError(message)


def _project(value: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return basis @ (basis.T @ value)


def _component(value: torch.Tensor) -> dict[str, float]:
    norm = i9._norm(value)
    energy = i9._dot(value, value)
    return {"norm": norm, "squared_energy": energy}


def observe_delivery(model: torch.nn.Module, optimizer: torch.optim.Optimizer, tracker: Any,
                     x: torch.Tensor, target: torch.Tensor,
                     policy: str) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    """Observe one raw gradient once and leave the selected delivery in ``.grad``.

    Returned tensors are detached clones on the model's device. ``post_basis`` is
    the selected post-ingest basis, also on that device. No optimizer step occurs.
    """
    _need(type(policy) is str and policy in POLICIES, "unknown mean-filter policy")
    i10._optimizer_is_i9(optimizer, model)
    _need(tracker is not None and tracker.grad_mean is not None,
          "mean delivery requires an initialized observer mean")
    _need(float(tracker.decay) == DECAY, "observer decay differs from I13 definition")
    before_parameters = i9.flat_params(model)
    before_optimizer = i9.tree_digest(i9._cpu_clone(optimizer.state_dict()))
    observer_before = int(tracker.step_count)
    pre_mean = i9._finite_tensor(tracker.grad_mean.detach().clone(), "pre-ingest mean")

    optimizer.zero_grad(set_to_none=True)
    objective = i9._objective_loss(model, x, target)
    objective.backward()
    raw = i9.flat_grad(model)
    raw_digest = i9.tree_digest(raw)
    observer_diagnostic = tracker.filter_grad()
    native = i9.flat_grad(model)
    observer_after = int(tracker.step_count)
    _need(observer_after == observer_before + 1,
          "observer did not advance exactly once")
    _need(bool(observer_diagnostic["filtering_active"]),
          "mean delivery requires active rank filtering")
    post_mean = i9._finite_tensor(tracker.grad_mean.detach().clone(), "post-ingest mean")
    post_basis = i9._basis(tracker)
    _need(post_basis is not None and post_basis.ndim == 2 and post_basis.shape[0] == raw.numel(),
          "post-ingest basis is unavailable or malformed")
    post_basis = i9._finite_tensor(post_basis, "post-ingest basis")

    explicit_pg = _project(raw, post_basis)
    outside_raw = raw - native
    outside_post_mean = post_mean - _project(post_mean, post_basis)
    outside_pre_mean = pre_mean - _project(pre_mean, post_basis)
    mean_delivery = native + outside_post_mean
    leak_delivery = native + LEAK * outside_raw
    selected = {"raw": raw, "current32": native,
                "mean32": mean_delivery, "leak01_32": leak_delivery}[policy]
    for name, value in (("native projection", native), ("outside raw", outside_raw),
                        ("outside post mean", outside_post_mean),
                        ("outside pre mean", outside_pre_mean),
                        ("mean delivery", mean_delivery), ("leak delivery", leak_delivery),
                        ("selected delivery", selected)):
        i9._finite_tensor(value, name)
    i9.set_grad(model, selected)

    residuals = {
        "native_minus_explicit_post_projection": native - explicit_pg,
        "post_mean_recurrence": post_mean - (DECAY * pre_mean + LEAK * raw),
        "mean_definition": mean_delivery - (post_mean + _project(raw - post_mean, post_basis)),
        "leak_definition": leak_delivery - (native + LEAK * (raw - native)),
        "mean_minus_leak_historical_term":
            mean_delivery - leak_delivery - DECAY * outside_pre_mean,
    }
    residual_scales = {
        "native_minus_explicit_post_projection": i9._norm(native) + i9._norm(explicit_pg),
        "post_mean_recurrence": (i9._norm(post_mean) + DECAY * i9._norm(pre_mean)
                                 + LEAK * i9._norm(raw)),
        "mean_definition": (i9._norm(mean_delivery) + i9._norm(post_mean)
                            + i9._norm(_project(raw - post_mean, post_basis))),
        "leak_definition": (i9._norm(leak_delivery) + i9._norm(native)
                            + LEAK * i9._norm(raw - native)),
        "mean_minus_leak_historical_term": (i9._norm(mean_delivery) + i9._norm(leak_delivery)
                                             + DECAY * i9._norm(outside_pre_mean)),
    }
    residual_norms = {name: i9._norm(value) for name, value in residuals.items()}
    # Homogeneous backward-error checks: no arbitrary absolute norm floor or
    # division by a clamped component norm.
    algebra_bounds = {name: 128 * torch.finfo(raw.dtype).eps * scale
                      for name, scale in residual_scales.items()}
    for name in residuals:
        _need(residual_norms[name] <= algebra_bounds[name]
              if residual_scales[name] > 0 else residual_norms[name] == 0,
              f"{name} algebra invariant differs")
    record = {
        "schema": "i13_observe_delivery_v1", "policy": policy,
        "loss": float(objective.detach().item()),
        "observer": {"step_before": observer_before, "step_after": observer_after,
                     "filtering_active": True,
                     "basis_rank": int(post_basis.shape[1])},
        "components": {
            "raw_gradient": _component(raw),
            "native_projection": _component(native),
            "outside_raw_gradient": _component(outside_raw),
            "outside_post_mean": _component(outside_post_mean),
            "outside_pre_mean": _component(outside_pre_mean),
            "mean32_delivery": _component(mean_delivery),
            "leak01_32_delivery": _component(leak_delivery),
            "selected_delivery": _component(selected),
        },
        "algebra_residual_norms": residual_norms,
        "algebra_error_bounds": algebra_bounds,
        "raw_gradient_digest": raw_digest,
        "post_observer_digest": i9.tree_digest(i9._tracker_state(tracker)),
        "applied_gradient_digest": i9.tree_digest(selected),
    }
    _need(torch.equal(before_parameters, i9.flat_params(model)),
          "observation changed model parameters")
    _need(before_optimizer == i9.tree_digest(i9._cpu_clone(optimizer.state_dict())),
          "observation changed Adam state")
    i12._finite_record(record, "mean observation")
    json.dumps(record, ensure_ascii=True, allow_nan=False, separators=(",", ":"))
    tensors = {
        "pre_mean": pre_mean.detach().clone(),
        "raw_gradient": raw.detach().clone(),
        "native_projection": native.detach().clone(),
        "post_mean": post_mean.detach().clone(),
        "post_basis": post_basis.detach().clone(),
        "delivered_gradient": selected.detach().clone(),
        "mean32_delivery": mean_delivery.detach().clone(),
        "leak01_32_delivery": leak_delivery.detach().clone(),
    }
    return record, tensors


def branch_step(model: torch.nn.Module, optimizer: torch.optim.Optimizer, tracker: Any,
                x: torch.Tensor, target: torch.Tensor, policy: str,
                original_basis: torch.Tensor) -> dict[str, Any]:
    """Execute one I13 update with the standard I10 displacement diagnostics."""
    before = i9.flat_params(model)
    basis = i10._basis(original_basis, before.numel(), before.device, before.dtype)
    observation, _ = observe_delivery(model, optimizer, tracker, x, target, policy)
    applied = i9.flat_grad(model)
    optimizer.step()
    after = i9.flat_params(model)
    i9._finite_tensor(after, "updated parameters")
    total_delta = i10._finite(after - before, "total displacement")
    data_delta = i10._finite(total_delta + i9.LR * i9.WD * before,
                             "decay-adjusted data displacement")
    current_basis = i9._basis(tracker)
    row = {
        "schema": "i13_mean_branch_step_v1", "policy": policy,
        "loss": observation["loss"],
        "observer": {"used": True,
                     "step_before": observation["observer"]["step_before"],
                     "step_after": observation["observer"]["step_after"],
                     "filtering_active": observation["observer"]["filtering_active"]},
        "gradient_filter_applied": policy != "raw",
        "gradient": {"raw_norm": observation["components"]["raw_gradient"]["norm"],
                     "applied_norm": i9._norm(applied)},
        "displacement": {
            "total_norm": i9._norm(total_delta), "data_norm": i9._norm(data_delta),
            "total_leakage": {"frozen_basis": i10._leakage(total_delta, basis),
                              "current_basis": i10._leakage(total_delta, current_basis)},
            "data_leakage": {"frozen_basis": i10._leakage(data_delta, basis),
                             "current_basis": i10._leakage(data_delta, current_basis)},
        },
        "mean_diagnostics": observation,
    }
    i12._finite_live(model, optimizer, tracker, "post_mean_branch_step")
    i12._finite_record(row, "mean branch step")
    json.dumps(row, ensure_ascii=True, allow_nan=False, separators=(",", ":"))
    return row


def _probe(probe: Callable[[dict[str, Any], int], dict[str, Any]] | None,
           model: torch.nn.Module, optimizer: torch.optim.Optimizer, tracker: Any,
           horizon: int) -> dict[str, Any] | None:
    if probe is None:
        return None
    state = i9.snapshot(model, optimizer, tracker)
    state_digest = i9.tree_digest(state)
    rng = i9._rng_state()
    live_digest = state_digest
    try:
        result = probe(state, horizon)
    except ValueError as exc:
        if type(exc).__name__ == "MeanProbeError" and str(exc) in _PROBE_NONFINITE_MESSAGES:
            raise i12._NumericalFailure("probe: " + str(exc)) from exc
        if type(exc) is i9.NeuralCoreError and str(exc) == _PROBE_GRADIENT_NONFINITE_MESSAGE:
            raise i12._NumericalFailure("probe: " + str(exc)) from exc
        raise
    _need(type(result) is dict, "probe callback must return a dictionary")
    _need(i9.tree_digest(state) == state_digest, "probe callback mutated its snapshot input")
    _need(i9.equal_tree(rng, i9._rng_state()), "probe callback changed global RNG state")
    _need(i9.tree_digest(i9.snapshot(model, optimizer, tracker)) == live_digest,
          "probe callback changed live branch state")
    i12._finite_record(result, "probe record")
    json.dumps(result, ensure_ascii=True, allow_nan=False, separators=(",", ":"))
    return result


def run_branch(
    state: dict[str, Any], data: dict[str, torch.Tensor], plan: dict[str, Any],
    objective: str, policy: str, *, steps: int = 500,
    horizons: Iterable[int] = DEFAULT_HORIZONS,
    check: Callable[[], None] | None = None,
    expected_initial_evaluation: dict[str, Any] | None = None,
    probe: Callable[[dict[str, Any], int], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run one inherited-state I13 branch and retain neutral horizon probes."""
    i12._optimizer_topology(state)
    _need(type(policy) is str and policy in POLICIES, "unknown mean-filter policy")
    _need(type(objective) is str and objective in OBJECTIVES, "unknown objective")
    _need(type(steps) is int and steps > 0, "steps must be positive")
    eval_horizons = i12._horizons(horizons, steps)
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
    _need(tracker is not None and tracker.grad_mean is not None,
          "mean branch requires initialized observer state")
    _need(i9.tree_digest(i9.snapshot(model, optimizer, tracker)) == parent_digest,
          "restored starting state differs")
    original_basis = i9._basis(tracker)
    _need(original_basis is not None, "mean branch requires initialized original basis")
    original_basis_digest = i9.tree_digest(original_basis)
    parent_counter = i12._optimizer_topology(state)[1]
    parent_observer = int(tracker.step_count)

    curve: list[dict[str, Any]] = []
    probes: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    last_evaluated_state: dict[str, Any] | None = None
    last_evaluated_horizon: int | None = None
    first_raw_gradient_digest: str | None = None
    first_post_observer_digest: str | None = None
    first_applied_gradient_digest: str | None = None

    def failure(exc: BaseException, phase: str, attempted: int) -> dict[str, Any]:
        _need(i9.tree_digest(state) == parent_digest, "numeric mean branch mutated its parent")
        return {
            "schema": "i13_mean_branch_v1", "status": "numerical_failure",
            "curve": curve, "probes": probes, "steps": diagnostics,
            "terminal_state": i9.snapshot(model, optimizer, tracker),
            "completed_steps": len(diagnostics), "attempted_step": attempted,
            "last_evaluated_state": last_evaluated_state,
            "last_evaluated_horizon": last_evaluated_horizon,
            "first_step_raw_gradient_digest": first_raw_gradient_digest,
            "first_step_observer_digest": first_post_observer_digest,
            "first_step_applied_gradient_digest": first_applied_gradient_digest,
            "numerical_failure": {"phase": phase, "exception_type": type(exc).__name__,
                                  "message": str(exc)},
        }

    if check is not None:
        check()
    i12._finite_live(model, optimizer, tracker, "initial_state")
    before_eval = i9.snapshot(model, optimizer, tracker)
    baseline = i10.evaluate(model, data)
    i12._finite_record(baseline, "initial_evaluation")
    _need(i9.equal_tree(before_eval, i9.snapshot(model, optimizer, tracker)),
          "horizon-zero evaluation changed state")
    if expected_initial_evaluation is not None:
        _need(type(expected_initial_evaluation) is dict
              and baseline == expected_initial_evaluation,
              "horizon-zero evaluation differs from the bound I10 endpoint")
    probe_record = None
    if probe is not None:
        if check is not None:
            check()
        probe_record = _probe(probe, model, optimizer, tracker, 0)
    curve.append({"horizon": 0, **baseline})
    if probe_record is not None:
        probes.append({"horizon": 0, "record": probe_record})
    last_evaluated_state, last_evaluated_horizon = before_eval, 0

    for horizon in range(1, steps + 1):
        if check is not None:
            check()
        indices = batches[horizon - 1]
        target = i12._target(data, indices, objective, redraw_mask, redraw_digits, horizon - 1)
        try:
            i12._finite_live(model, optimizer, tracker, f"pre_step_{horizon}")
            row = branch_step(model, optimizer, tracker, data["x"][indices], target,
                              policy, original_basis)
            if horizon == 1:
                first_raw_gradient_digest = row["mean_diagnostics"]["raw_gradient_digest"]
                first_post_observer_digest = row["mean_diagnostics"]["post_observer_digest"]
                first_applied_gradient_digest = row["mean_diagnostics"]["applied_gradient_digest"]
        except BaseException as exc:
            if i12._is_known_numeric(exc):
                return failure(exc, "step", horizon)
            raise
        diagnostics.append({"horizon": horizon, **row})
        if horizon in eval_horizons:
            if check is not None:
                check()
            try:
                before_eval = i9.snapshot(model, optimizer, tracker)
                evaluation = i10.evaluate(model, data)
                i12._finite_record(evaluation, "evaluation")
            except BaseException as exc:
                if i12._is_known_numeric(exc):
                    return failure(exc, "evaluation", horizon)
                raise
            _need(i9.equal_tree(before_eval, i9.snapshot(model, optimizer, tracker)),
                  f"horizon-{horizon} evaluation changed state")
            # Evaluation is independently successful evidence. Preserve it and
            # its exact state even if the following optional probe is the
            # operation that encounters an explicit numerical failure.
            curve.append({"horizon": horizon, **evaluation})
            last_evaluated_state, last_evaluated_horizon = before_eval, horizon
            probe_record = None
            if probe is not None:
                if check is not None:
                    check()
                try:
                    probe_record = _probe(probe, model, optimizer, tracker, horizon)
                except BaseException as exc:
                    if i12._is_known_numeric(exc):
                        return failure(exc, "probe", horizon)
                    raise
            if probe_record is not None:
                probes.append({"horizon": horizon, "record": probe_record})

    final = i9.snapshot(model, optimizer, tracker)
    _need(i12._optimizer_topology(final)[1] == parent_counter + steps,
          "final Adam counter is not parent plus branch steps")
    _need(int(final["tracker"]["step_count"]) == parent_observer + steps,
          "final observer counter differs")
    _need(i9.tree_digest(state) == parent_digest, "mean branch mutated its parent")
    _need(i9.tree_digest(original_basis) == original_basis_digest,
          "original diagnostic basis was mutated")
    _need([row["horizon"] for row in curve] == list(eval_horizons)
          and len(diagnostics) == steps, "returned horizon/diagnostic coverage differs")
    _need(probe is None or [row["horizon"] for row in probes] == list(eval_horizons),
          "returned probe horizons differ")
    return {
        "schema": "i13_mean_branch_v1", "status": "complete",
        "curve": curve, "probes": probes, "steps": diagnostics,
        "terminal_state": final, "completed_steps": steps, "attempted_step": steps,
        "first_step_raw_gradient_digest": first_raw_gradient_digest,
        "first_step_observer_digest": first_post_observer_digest,
        "first_step_applied_gradient_digest": first_applied_gradient_digest,
        "numerical_failure": None,
    }


def main() -> int:
    print("mean_core: inert library; no branch or probe executed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
