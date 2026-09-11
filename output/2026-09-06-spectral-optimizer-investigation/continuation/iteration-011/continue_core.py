"""Thin fixed-label continuation helper over the frozen I9/I10 state machinery."""
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
_I10_SPEC = importlib.util.spec_from_file_location("_i11_frozen_i10_branch_core", _I10_PATH)
if _I10_SPEC is None or _I10_SPEC.loader is None:  # pragma: no cover
    raise ImportError("cannot load frozen I10 branch core")
i10 = importlib.util.module_from_spec(_I10_SPEC)
_I10_SPEC.loader.exec_module(i10)
i9 = i10.i9


DEFAULT_HORIZONS = (0, 100, 500, 1000, 1500)


class ContinuationError(ValueError):
    pass


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise ContinuationError(message)


def _optimizer_counter(snapshot: dict[str, Any]) -> int:
    states = snapshot["optimizer"]["state"]
    _need(type(states) is dict and bool(states), "optimizer state is empty")
    values = []
    for state in states.values():
        step = state.get("step")
        scalar = float(step.item()) if type(step) is torch.Tensor else float(step)
        _need(math.isfinite(scalar) and scalar == round(scalar),
              "optimizer counter is not a finite integer")
        values.append(int(scalar))
    _need(len(set(values)) == 1, "optimizer parameter counters differ")
    return values[0]


def _horizons(values: Iterable[int], steps: int) -> tuple[int, ...]:
    result = tuple(values)
    _need(result and all(type(value) is int for value in result),
          "evaluation horizons must be integers")
    _need(result == tuple(sorted(set(result))) and result[0] == 0
          and result[-1] == steps and all(0 <= value <= steps for value in result),
          "evaluation horizons must be unique, increasing, start at zero and end at steps")
    return result


def continue_fixed(
    state: dict[str, Any],
    data: dict[str, torch.Tensor],
    plan_batches: Any,
    policy: str,
    original_frozen_basis: torch.Tensor,
    *,
    steps: int = 1500,
    eval_horizons: Iterable[int] = DEFAULT_HORIZONS,
    check: Callable[[], None] | None = None,
    expected_initial_evaluation: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Continue an I10 final state on its fixed labels without mutating inputs.

    Horizons in the returned curve and diagnostics are local to this continuation.
    The caller owns any offset relative to the preceding 500-step I10 branch.
    ``original_frozen_basis`` must be the I9-parent action used by the I10 branch;
    it is never inferred or reset at this continuation boundary.
    """
    _need(type(state) is dict and state.get("schema") == "i9_neural_snapshot_v1",
          "state must be a complete I9-schema snapshot")
    _need(type(steps) is int and steps > 0, "steps must be positive")
    _need(type(policy) is str and policy in i10.POLICIES, "unknown continuation policy")
    _need(type(data) is dict and tuple(data) == i9.DATA_KEYS, "data keys/order differ")
    _need(all(type(value) is torch.Tensor for value in data.values()),
          "data values must be tensors")
    device = data["x"].device
    _need(all(value.device == device for value in data.values()),
          "all data tensors must share one device")
    horizons = _horizons(eval_horizons, steps)
    batches = i9._indices(plan_batches, (steps, 64), len(data["x"]), device)
    _need(type(original_frozen_basis) is torch.Tensor and original_frozen_basis.ndim == 2,
          "original frozen basis must be a matrix")

    input_digest = i9.tree_digest(state)
    original_basis_digest = i9.tree_digest(original_frozen_basis)
    model, optimizer, tracker = i9.restore(state, device)
    _need(tracker is not None, "continuation requires native observer state")
    restored = i9.snapshot(model, optimizer, tracker)
    _need(i9.tree_digest(restored) == input_digest, "restored starting state differs")
    parameter = next(model.parameters())
    frozen_basis = original_frozen_basis.detach().to(
        device=device, dtype=parameter.dtype).clone()
    _need(frozen_basis.shape[0] == sum(item.numel() for item in model.parameters())
          and 0 < frozen_basis.shape[1] <= 32, "original frozen basis shape differs")

    parent_optimizer_step = _optimizer_counter(restored)
    parent_observer_step = int(tracker.step_count)
    parent_tracker_digest = i9.tree_digest(restored["tracker"])

    if check is not None:
        check()
    before_eval = i9.snapshot(model, optimizer, tracker)
    baseline = i10.evaluate(model, data)
    _need(i9.equal_tree(before_eval, i9.snapshot(model, optimizer, tracker)),
          "horizon-zero evaluation changed state")
    if expected_initial_evaluation is not None:
        _need(type(expected_initial_evaluation) is dict,
              "expected initial evaluation must be a dictionary")
        _need(baseline == expected_initial_evaluation,
              "horizon-zero evaluation differs from the bound I10 endpoint")
    curve = [{"horizon": 0, **baseline}]
    diagnostics: list[dict[str, Any]] = []

    for horizon in range(1, steps + 1):
        if check is not None:
            check()
        indices = batches[horizon - 1]
        row = i10.branch_step(model, optimizer, tracker, data["x"][indices],
                              data["noisy"][indices], policy, frozen_basis)
        diagnostics.append({"horizon": horizon, **row})
        if horizon in horizons:
            if check is not None:
                check()
            before_eval = i9.snapshot(model, optimizer, tracker)
            curve.append({"horizon": horizon, **i10.evaluate(model, data)})
            _need(i9.equal_tree(before_eval, i9.snapshot(model, optimizer, tracker)),
                  f"horizon-{horizon} evaluation changed state")

    final = i9.snapshot(model, optimizer, tracker)
    _need(_optimizer_counter(final) == parent_optimizer_step + steps,
          "final Adam counter is not parent plus continuation steps")
    expected_observer = (parent_observer_step if policy == "frozen32"
                         else parent_observer_step + steps)
    _need(int(final["tracker"]["step_count"]) == expected_observer,
          "final observer counter differs")
    if policy == "frozen32":
        _need(i9.tree_digest(final["tracker"]) == parent_tracker_digest,
              "frozen observer changed during continuation")
    _need(i9.tree_digest(state) == input_digest, "input state was mutated")
    _need(i9.tree_digest(original_frozen_basis) == original_basis_digest,
          "original frozen basis input was mutated")
    _need([row["horizon"] for row in curve] == list(horizons),
          "returned curve horizons differ")
    _need(len(diagnostics) == steps, "returned diagnostic count differs")
    json.dumps(curve, ensure_ascii=True, allow_nan=False, separators=(",", ":"))
    json.dumps(diagnostics, ensure_ascii=True, allow_nan=False, separators=(",", ":"))
    return curve, diagnostics, final


def main() -> int:
    print("continue_core: inert library; no continuation executed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
