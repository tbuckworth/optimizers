#!/usr/bin/env python3
"""CPU-only integrity audit and predeclared analysis of completed I12 branches."""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
_A11_PATH = HERE.parent / "iteration-011" / "analyse_continuation.py"
_A11_SPEC = importlib.util.spec_from_file_location("_i12_analysis_helpers", _A11_PATH)
if _A11_SPEC is None or _A11_SPEC.loader is None:  # pragma: no cover
    raise ImportError("cannot load independent I11 analysis helpers")
a11 = importlib.util.module_from_spec(_A11_SPEC)
_A11_SPEC.loader.exec_module(a11)
a10 = a11.a10


SEEDS = (100, 101, 102)
ANCHORS = (1500, 2000)
OBJECTIVES = ("soft", "redraw")
POLICIES = ("raw", "current32")
NEW_ARMS = ("zero_m", "zero_v", "zero_mv", "fresh_adam")
ARMS = ("inherited",) + NEW_ARMS
HORIZONS = (0, 1, 10, 50, 100, 250, 500)
EDIT_FIELDS = {
    "zero_m": frozenset(("exp_avg",)),
    "zero_v": frozenset(("exp_avg_sq",)),
    "zero_mv": frozenset(("exp_avg", "exp_avg_sq")),
    "fresh_adam": frozenset(("step", "exp_avg", "exp_avg_sq")),
}
ENTRY_META = (
    "id", "seed", "parent_step", "source", "objective", "policy", "arm",
    "parent_artifact", "parent_complete_digest", "edited_parent_artifact",
    "edited_parent_complete_digest", "baseline_branch_artifact", "baseline_final_artifact",
    "plan_file", "initial_evaluation_exact", "status", "completed_steps",
    "attempted_step", "requested_horizon",
)


def expected_new_keys() -> set[tuple[Any, ...]]:
    return {(seed, anchor, objective, policy, arm) for seed in SEEDS for anchor in ANCHORS
            for objective in OBJECTIVES for policy in POLICIES for arm in NEW_ARMS}


def expected_baseline_keys() -> set[tuple[Any, ...]]:
    return {(seed, anchor, objective, policy) for seed in SEEDS for anchor in ANCHORS
            for objective in OBJECTIVES for policy in POLICIES}


def same_tree(left: Any, right: Any) -> bool:
    try:
        return a10.tree_digest(left) == a10.tree_digest(right)
    except Exception:
        return False


def external_hash(path: Path, expected: str, audit: Any, label: str) -> None:
    audit.require(path.is_file(), f"{label}: file is missing")
    if path.is_file():
        audit.require(a10.sha256(path) == expected, f"{label}: SHA256 differs")
        audit.hash_files += 1
        audit.hash_bytes += path.stat().st_size


def curve_map(curve: list[dict[str, Any]], audit: Any, label: str,
              allowed: tuple[int, ...] = HORIZONS) -> dict[int, dict[str, Any]]:
    horizons = [row.get("horizon") for row in curve]
    audit.require(horizons == sorted(set(horizons)), f"{label}: curve horizons differ/order repeats")
    audit.require(all(horizon in allowed for horizon in horizons),
                  f"{label}: curve contains an undeclared horizon")
    result = {}
    for row in curve:
        a10.validate_evaluation(row, audit, f"{label} h{row.get('horizon')}")
        result[row["horizon"]] = row
    return result


def masked_digest(state: dict[str, Any], fields: frozenset[str]) -> str:
    # Independent equivalent of the acquisition mutation proof: authorized
    # values are replaced by a fixed sentinel before digesting.
    value = {key: item for key, item in state.items()}
    optimizer = {key: item for key, item in state["optimizer"].items()}
    optimizer["state"] = {parameter_id: dict(row)
                          for parameter_id, row in state["optimizer"]["state"].items()}
    value["optimizer"] = optimizer
    for row in optimizer["state"].values():
        for field in fields:
            row[field] = "__I12_AUTHORIZED_MOMENT_FIELD__"
    return a10.tree_digest(value)


def validate_edit(parent: dict[str, Any], edited: dict[str, Any], arm: str,
                  mutation: dict[str, Any], audit: Any, label: str) -> None:
    fields = EDIT_FIELDS[arm]
    audit.require(tuple(parent) == tuple(edited), f"{label}: snapshot topology differs")
    for key in parent:
        if key != "optimizer":
            audit.require(same_tree(parent[key], edited[key]), f"{label}: changed {key}")
    before_opt, after_opt = parent["optimizer"], edited["optimizer"]
    audit.require(tuple(before_opt) == tuple(after_opt) == ("state", "param_groups"),
                  f"{label}: optimizer topology differs")
    audit.require(same_tree(before_opt["param_groups"], after_opt["param_groups"]),
                  f"{label}: optimizer parameter groups changed")
    before_states, after_states = before_opt["state"], after_opt["state"]
    audit.require(tuple(before_states) == tuple(after_states), f"{label}: parameter order differs")
    for parameter_id in before_states:
        before, after = before_states[parameter_id], after_states[parameter_id]
        audit.require(tuple(before) == tuple(after) == ("step", "exp_avg", "exp_avg_sq"),
                      f"{label}: Adam state fields differ")
        for field in before:
            left, right = before[field], after[field]
            audit.require(type(left) is torch.Tensor and type(right) is torch.Tensor
                          and left.dtype == right.dtype and tuple(left.shape) == tuple(right.shape),
                          f"{label}: {parameter_id}/{field} tensor metadata differs")
            if field in fields:
                audit.require(bool((right == 0).all()),
                              f"{label}: {parameter_id}/{field} was not zeroed")
            else:
                audit.require(torch.equal(left, right),
                              f"{label}: {parameter_id}/{field} changed unexpectedly")
    parent_digest, edited_digest = a10.tree_digest(parent), a10.tree_digest(edited)
    audit.require(mutation.get("schema") == "i12_moment_edit_audit_v1"
                  and mutation.get("arm") == arm, f"{label}: mutation audit schema/arm differs")
    audit.require(mutation.get("parent_digest") == parent_digest
                  and mutation.get("edited_digest") == edited_digest,
                  f"{label}: mutation digest proof differs")
    audit.require(mutation.get("zeroed_fields") == sorted(fields),
                  f"{label}: mutation field declaration differs")
    audit.require(mutation.get("parameter_states") == len(before_states)
                  and mutation.get("targeted_tensor_fields") == len(before_states) * len(fields),
                  f"{label}: mutation field counts differ")
    audit.require(mutation.get("counter_before") == a10.optimizer_counter(parent)
                  and mutation.get("counter_after") == a10.optimizer_counter(edited),
                  f"{label}: mutation counter audit differs")
    audit.require(mutation.get("parent_unchanged") is True
                  and mutation.get("unmodified_fields_equal") is True
                  and mutation.get("unmodified_fields_digest") == masked_digest(parent, fields)
                  == masked_digest(edited, fields), f"{label}: unmodified-field proof differs")


def value_at(branch: dict[str, Any], horizon: int, metric: str) -> float | None:
    level = branch["curve_by_horizon"].get(horizon)
    if level is None:
        return None
    if metric == "ce":
        return -float(level["auxiliary"]["clean_ce"])
    if metric == "accuracy":
        return float(level["auxiliary"]["clean_accuracy"])
    raise ValueError(metric)


def aggregate(cells: list[dict[str, Any]], field: str) -> dict[str, Any]:
    per_seed = []
    for seed in SEEDS:
        rows = [row for row in cells if row["seed"] == seed]
        rows.sort(key=lambda row: row["parent_step"])
        values = [row[field] for row in rows]
        available = len(rows) == len(ANCHORS) and all(value is not None for value in values)
        per_seed.append({"seed": seed,
                         "parent_values": {str(row["parent_step"]): row[field] for row in rows},
                         "available": available,
                         "seed_first_parent_mean": (float(np.mean(values)) if available else None)})
    seed_values = [row["seed_first_parent_mean"] for row in per_seed]
    available = len(per_seed) == len(SEEDS) and all(value is not None for value in seed_values)
    return {"per_seed": per_seed, "available": available,
            "all_three_seed_values": seed_values,
            "mean": float(np.mean(seed_values)) if available else None,
            "no_survivor_averaging": True}


def benefit_cells(branches: dict[tuple[Any, ...], dict[str, Any]], objective: str,
                  metric: str, arm: str, horizon: int) -> list[dict[str, Any]]:
    result = []
    for seed in SEEDS:
        for anchor in ANCHORS:
            raw = value_at(branches[(seed, anchor, objective, "raw", arm)], horizon, metric)
            current = value_at(branches[(seed, anchor, objective, "current32", arm)], horizon, metric)
            result.append({"seed": seed, "parent_step": anchor,
                           "raw_utility": raw, "current32_utility": current,
                           "benefit": None if raw is None or current is None else current - raw,
                           "available": raw is not None and current is not None})
    return result


def all_benefits(branches: dict[tuple[Any, ...], dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for objective in OBJECTIVES:
        for metric in ("ce", "accuracy"):
            for arm in ARMS:
                for horizon in HORIZONS:
                    cells = benefit_cells(branches, objective, metric, arm, horizon)
                    inherited = ({(r["seed"], r["parent_step"]): r for r in
                                  benefit_cells(branches, objective, metric, "inherited", horizon)})
                    for cell in cells:
                        reference = inherited[(cell["seed"], cell["parent_step"])]["benefit"]
                        cell["benefit_minus_inherited"] = (
                            None if cell["benefit"] is None or reference is None
                            else cell["benefit"] - reference)
                    rows.append({"objective": objective, "metric": metric, "arm": arm,
                                 "horizon": horizon, "individual_parent_cells": cells,
                                 "benefit_seed_first": aggregate(cells, "benefit"),
                                 "benefit_minus_inherited_seed_first":
                                     aggregate(cells, "benefit_minus_inherited")})
    return rows


def factorial_values(benefits: dict[str, float | None]) -> dict[str, float | None]:
    definitions = {
        "delta_v_minus_m": (("zero_v", "zero_m"),
                            lambda b: b["zero_v"] - b["zero_m"]),
        "m_removal_marginal": (("zero_m", "zero_mv", "inherited", "zero_v"),
                               lambda b: (b["zero_m"] + b["zero_mv"]
                                          - b["inherited"] - b["zero_v"]) / 2),
        "v_removal_marginal": (("zero_v", "zero_mv", "inherited", "zero_m"),
                               lambda b: (b["zero_v"] + b["zero_mv"]
                                          - b["inherited"] - b["zero_m"]) / 2),
        "m_by_v_interaction": (("zero_mv", "zero_m", "zero_v", "inherited"),
                               lambda b: (b["zero_mv"] - b["zero_m"]
                                          - b["zero_v"] + b["inherited"])),
        "counter_effect_empty_moments": (("fresh_adam", "zero_mv"),
                                         lambda b: b["fresh_adam"] - b["zero_mv"]),
    }
    return {name: (None if any(benefits[arm] is None for arm in required)
                   else function(benefits))
            for name, (required, function) in definitions.items()}


def factorial_rows(branches: dict[tuple[Any, ...], dict[str, Any]]) -> list[dict[str, Any]]:
    names = ("delta_v_minus_m", "m_removal_marginal", "v_removal_marginal",
             "m_by_v_interaction", "counter_effect_empty_moments")
    result = []
    for objective in OBJECTIVES:
        for metric in ("ce", "accuracy"):
            for horizon in HORIZONS:
                by_arm = {arm: {(row["seed"], row["parent_step"]): row["benefit"]
                                for row in benefit_cells(branches, objective, metric, arm, horizon)}
                          for arm in ARMS}
                cells = []
                for seed in SEEDS:
                    for anchor in ANCHORS:
                        benefits = {arm: by_arm[arm][(seed, anchor)] for arm in ARMS}
                        cell: dict[str, Any] = {"seed": seed, "parent_step": anchor,
                                               "benefits": benefits}
                        cell.update(factorial_values(benefits))
                        cells.append(cell)
                result.append({"objective": objective, "metric": metric, "horizon": horizon,
                               "individual_parent_cells": cells,
                               "seed_first": {name: aggregate(cells, name)
                                              for name in names}})
    return result


def primary(factorials: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for objective in OBJECTIVES:
        for metric in ("ce", "accuracy"):
            source = next(row for row in factorials if row["objective"] == objective
                          and row["metric"] == metric and row["horizon"] == 500)
            estimate = source["seed_first"]["delta_v_minus_m"]
            rows.append({"objective": objective, "metric": metric, "horizon": 500,
                         "estimand": "B_zero_v - B_zero_m",
                         "positive_means": "v deletion improves current-vs-raw utility gap more than m deletion",
                         **estimate})
    return rows


def absolute_progress(branches: dict[tuple[Any, ...], dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for objective in OBJECTIVES:
        for metric in ("ce", "accuracy"):
            for policy in POLICIES:
                for arm in ARMS:
                    for horizon in HORIZONS:
                        cells = []
                        for seed in SEEDS:
                            for anchor in ANCHORS:
                                branch = branches[(seed, anchor, objective, policy, arm)]
                                start = value_at(branch, 0, metric)
                                level = value_at(branch, horizon, metric)
                                inherited = branches[(seed, anchor, objective, policy, "inherited")]
                                inherited_start = value_at(inherited, 0, metric)
                                inherited_level = value_at(inherited, horizon, metric)
                                progress = None if start is None or level is None else level - start
                                inherited_progress = (None if inherited_start is None
                                                      or inherited_level is None
                                                      else inherited_level - inherited_start)
                                cells.append({
                                    "seed": seed, "parent_step": anchor, "utility_h0": start,
                                    "utility": level, "progress_from_h0": progress,
                                    "utility_minus_inherited": (None if level is None
                                                                 or inherited_level is None
                                                                 else level - inherited_level),
                                    "progress_minus_inherited_progress":
                                        (None if progress is None or inherited_progress is None
                                         else progress - inherited_progress),
                                })
                        fields = ("progress_from_h0", "utility_minus_inherited",
                                  "progress_minus_inherited_progress")
                        rows.append({"objective": objective, "metric": metric,
                                     "policy": policy, "arm": arm, "horizon": horizon,
                                     "individual_parent_cells": cells,
                                     "seed_first": {field: aggregate(cells, field)
                                                    for field in fields}})
    return rows


def flatten_diagnostic(branch: dict[str, Any]) -> dict[str, Any]:
    diagnostic = branch["diagnostic_summary"]
    values: dict[str, Any] = {"completed_steps": branch["completed_steps"]}
    for name, value in diagnostic["arithmetic_step_means"].items():
        values["mean/" + name] = value
    for name, value in diagnostic["energy_weighted_leakage"].items():
        values["energy_leakage/" + name] = value["fraction"]
    first = branch.get("first_step")
    if first is not None:
        values.update({
            "first/loss": first["loss"],
            "first/raw_gradient_norm": first["gradient"]["raw_norm"],
            "first/applied_gradient_norm": first["gradient"]["applied_norm"],
            "first/total_step_norm": first["displacement"]["total_norm"],
            "first/data_step_norm": first["displacement"]["data_norm"],
        })
    return values


def diagnostic_cohorts(branches: dict[tuple[Any, ...], dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for objective in OBJECTIVES:
        for policy in POLICIES:
            for arm in ARMS:
                cells = []
                for seed in SEEDS:
                    for anchor in ANCHORS:
                        branch = branches[(seed, anchor, objective, policy, arm)]
                        cells.append({"seed": seed, "parent_step": anchor,
                                      "status": branch["status"],
                                      "values": flatten_diagnostic(branch)})
                complete = all(cell["status"] == "complete" for cell in cells)
                names = sorted(set.intersection(*[set(cell["values"]) for cell in cells]))
                aggregates = {}
                for name in names:
                    flat = [{"seed": cell["seed"], "parent_step": cell["parent_step"],
                             "value": cell["values"][name]} for cell in cells]
                    aggregates[name] = aggregate(flat, "value") if complete else {
                        "available": False, "mean": None, "no_survivor_averaging": True,
                        "reason": "one or more branches did not complete 500 updates"}
                result.append({"objective": objective, "policy": policy, "arm": arm,
                               "all_six_branches_complete": complete,
                               "individual_parent_cells": cells,
                               "equal_parent_then_seed_aggregates": aggregates})
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.artifacts.resolve(strict=True)
    if not root.is_dir():
        raise SystemExit("--artifacts must be a directory")
    args.output.parent.resolve(strict=True)
    args.output.mkdir(exist_ok=False)
    torch.set_num_threads(1)
    audit = a10.Audit()

    manifest = a10.read_json(root / "manifest.json")
    completion = a10.read_json(root / "completion.json")
    audit.require(manifest.get("mode") == "full" and manifest.get("planned_branches") == 96
                  and manifest.get("baseline_count") == 24
                  and manifest.get("edited_parent_count") == 24,
                  "manifest mode/counts differ")
    audit.require(tuple(manifest.get("arms", ())) == NEW_ARMS
                  and tuple(manifest.get("objectives", ())) == OBJECTIVES
                  and tuple(manifest.get("policies", ())) == POLICIES
                  and tuple(manifest.get("horizons", ())) == HORIZONS,
                  "manifest arms/objectives/policies/horizons differ")
    audit.require(completion.get("status") == "complete" and completion.get("mode") == "full"
                  and completion.get("attempted_branches") == 96,
                  "completion status/mode/attempt count differs")
    records = completion.get("artifacts", [])
    completion_index = {record["name"]: record for record in records}
    audit.require(len(completion_index) == len(records), "completion artifact names repeat")
    for record in records:
        a10.verify_record(root, record, audit, "I12 completion")
    for name in ("manifest.json", "parent-inputs.json", "baseline-references.json", "branches.json"):
        audit.require(name in completion_index, f"completion omits {name}")

    for path, expected in manifest.get("source_hashes", {}).items():
        external_hash(ROOT / path, expected, audit, f"frozen source {path}")

    parent_inputs = a10.read_json(root / "parent-inputs.json")
    i10_root = Path(parent_inputs["i10_directory"]).resolve(strict=True)
    i9_root = Path(parent_inputs["i9_directory"]).resolve(strict=True)
    audit.require(i10_root.is_dir() and i9_root.is_dir(), "I10/I9 input root missing")
    for parent_root, hashes, label in ((i10_root, parent_inputs["i10_hashes"], "I10"),
                                       (i9_root, parent_inputs["i9_hashes"], "I9")):
        for name, expected in hashes.items():
            external_hash(parent_root / name, expected, audit, f"{label} input {name}")
    audit.require(parent_inputs.get("baseline_count") == 24
                  and parent_inputs.get("parent_count") == 6,
                  "parent-input declared coverage differs")
    audit.require(parent_inputs.get("baseline_source_hashes") == {
        key: manifest["source_hashes"][key] for key in parent_inputs["baseline_source_hashes"]},
        "I10 baseline source hashes are not bound into I12 manifest")

    parent_cache: dict[tuple[int, int], dict[str, Any]] = {}
    parent_names = set()
    for seed in SEEDS:
        for anchor in ANCHORS:
            name = f"anchor-s{seed}-current32-t{anchor}.pt"
            parent_names.add(name)
            binding = torch.load(i9_root / name, weights_only=True, map_location="cpu")
            state = binding["state"]
            digest = a10.tree_digest(state)
            audit.tree_digests += 1
            audit.require((binding["seed"], binding["source"], binding["completed_step"])
                          == (seed, "current32", anchor), f"I9 parent {name}: metadata differs")
            audit.require(digest == binding["complete_state_digest"],
                          f"I9 parent {name}: complete-state digest differs")
            audit.require(a10.optimizer_counter(state) == anchor,
                          f"I9 parent {name}: optimizer counter differs")
            parent_cache[(seed, anchor)] = {
                "state": state, "digest": digest,
                "observer": int(state["tracker"]["step_count"]),
                "artifact_name": name,
            }

    baseline_refs = a10.read_json(root / "baseline-references.json")
    audit.require(baseline_refs.get("count") == 24
                  and baseline_refs.get("new_baseline_training_updates") == 0
                  and Path(baseline_refs.get("directory", "")).resolve() == i10_root,
                  "baseline-reference declaration differs")
    audit.require(len(baseline_refs.get("entries", [])) == 24,
                  "baseline-reference entry count differs")
    baselines: dict[tuple[Any, ...], dict[str, Any]] = {}
    baseline_summaries = []
    for entry in baseline_refs.get("entries", []):
        identity = entry.get("id", "unknown-baseline")
        try:
            key = (entry["seed"], entry["parent_step"], entry["objective"], entry["policy"])
            audit.require(key in expected_baseline_keys() and key not in baselines,
                          f"{identity}: baseline key invalid/duplicated")
            audit.require(entry["physical_source"] == "current32"
                          and entry["logical_sources"] == ["current32"],
                          f"{identity}: baseline source differs")
            branch_record, final_record = entry["branch_artifact"], entry["final_artifact"]
            audit.require(parent_inputs["i10_hashes"].get(branch_record["name"])
                          == branch_record["sha256"], f"{identity}: baseline branch hash unbound")
            audit.require(parent_inputs["i10_hashes"].get(final_record["name"])
                          == final_record["sha256"], f"{identity}: baseline final hash unbound")
            branch = a10.read_json(i10_root / branch_record["name"])
            final_binding = torch.load(i10_root / final_record["name"], weights_only=True,
                                       map_location="cpu")
            for name in ("id", "seed", "parent_step", "physical_source", "logical_sources",
                         "objective", "policy", "parent_artifacts", "parent_complete_digest",
                         "plan_file", "completed_horizon", "final_artifact"):
                audit.require(branch.get(name) == entry.get(name),
                              f"{identity}: inherited branch metadata differs for {name}")
            final_digest = a10.tree_digest(final_binding["state"])
            audit.tree_digests += 1
            audit.require(final_digest == final_binding["complete_state_digest"]
                          == branch["final_complete_digest"],
                          f"{identity}: inherited final digest differs")
            curve = curve_map(branch["curve"], audit, identity)
            audit.require(tuple(curve) == HORIZONS, f"{identity}: inherited curve incomplete")
            parent = parent_cache[(entry["seed"], entry["parent_step"])]
            diagnostic = a11.validate_steps(branch["steps"], count=500,
                policy=entry["policy"], parent_observer=parent["observer"], cumulative_offset=0,
                require_cumulative_field=False, audit=audit, label=identity)
            audit.require(a10.optimizer_counter(final_binding["state"])
                          == a10.optimizer_counter(parent["state"]) + 500,
                          f"{identity}: inherited final counter differs")
            row = {"status": "complete", "completed_steps": 500,
                   "curve_by_horizon": curve, "curve": branch["curve"],
                   "diagnostic_summary": diagnostic, "first_step": branch["steps"][0],
                   "branch_artifact": branch_record["name"],
                   "terminal_artifact": final_record["name"]}
            baselines[key] = row
            baseline_summaries.append({"seed": key[0], "parent_step": key[1],
                "objective": key[2], "policy": key[3], "curve": branch["curve"],
                "diagnostic_summary": diagnostic, "branch_artifact": branch_record["name"],
                "terminal_artifact": final_record["name"]})
        except Exception as exc:
            audit.errors.append(f"{identity}: baseline analysis failed: {type(exc).__name__}: {exc}")
    audit.require(set(baselines) == expected_baseline_keys(), "inherited baseline coverage differs")

    branch_index = a10.read_json(root / "branches.json")
    entries = branch_index.get("entries", [])
    edits = branch_index.get("edited_parents", [])
    audit.require(len(entries) == 96 and len(edits) == 24
                  and branch_index.get("attempted_branches") == 96
                  and branch_index.get("baseline_count") == 24
                  and branch_index.get("parent_hashes_unchanged") is True,
                  "branches index count/binding declaration differs")
    declared_failures = int(branch_index.get("numerical_failures", -1))
    audit.require(completion.get("numerical_failures") == declared_failures
                  and completion.get("science_complete") is (declared_failures == 0)
                  and branch_index.get("science_complete") is (declared_failures == 0),
                  "numerical-failure/science-complete declarations differ")

    edited_cache: dict[tuple[int, int, str], dict[str, Any]] = {}
    for entry in edits:
        label = f"edited-s{entry.get('seed')}-t{entry.get('parent_step')}-{entry.get('arm')}"
        try:
            key = (entry["seed"], entry["parent_step"], entry["arm"])
            audit.require(key not in edited_cache and entry["arm"] in NEW_ARMS,
                          f"{label}: edited key invalid/duplicated")
            record = entry["artifact"]
            audit.require(completion_index.get(record["name"]) == record,
                          f"{label}: completion record differs")
            binding = torch.load(root / record["name"], weights_only=True, map_location="cpu")
            audit.require((binding["seed"], binding["parent_step"], binding["arm"]) == key,
                          f"{label}: edited binding metadata differs")
            parent = parent_cache[(entry["seed"], entry["parent_step"])]
            audit.require(binding["parent_artifact"]["name"] == parent["artifact_name"]
                          and binding["parent_artifact"]["sha256"]
                          == parent_inputs["i9_hashes"][parent["artifact_name"]]
                          and binding["parent_complete_digest"] == parent["digest"],
                          f"{label}: parent binding differs")
            digest = a10.tree_digest(binding["state"])
            audit.tree_digests += 1
            audit.require(digest == binding["complete_state_digest"]
                          == entry["complete_state_digest"],
                          f"{label}: edited complete-state digest differs")
            validate_edit(parent["state"], binding["state"], entry["arm"],
                          binding["mutation_audit"], audit, label)
            expected_counter = 0 if entry["arm"] == "fresh_adam" else entry["parent_step"]
            audit.require(a10.optimizer_counter(binding["state"]) == expected_counter,
                          f"{label}: edited counter differs")
            edited_cache[key] = {"state": binding["state"], "digest": digest,
                                 "artifact": record}
        except Exception as exc:
            audit.errors.append(f"{label}: edit analysis failed: {type(exc).__name__}: {exc}")
    expected_edits = {(seed, anchor, arm) for seed in SEEDS for anchor in ANCHORS
                      for arm in NEW_ARMS}
    audit.require(set(edited_cache) == expected_edits, "edited-parent coverage differs")

    branches: dict[tuple[Any, ...], dict[str, Any]] = {
        (seed, anchor, objective, policy, "inherited"): row
        for (seed, anchor, objective, policy), row in baselines.items()
    }
    new_summaries, failure_rows = [], []
    first_gradient_groups: dict[tuple[Any, ...], list[tuple[str, str | None, int]]] = {}
    for entry in entries:
        identity = entry.get("id", "unknown-branch")
        try:
            key = (entry["seed"], entry["parent_step"], entry["objective"],
                   entry["policy"], entry["arm"])
            audit.require(key in expected_new_keys() and key not in branches,
                          f"{identity}: new branch key invalid/duplicated")
            expected_id = (f"s{key[0]}-current32-t{key[1]}-{key[2]}-{key[3]}-{key[4]}")
            audit.require(identity == expected_id and entry["source"] == "current32",
                          f"{identity}: ID/source differs")
            branch_record, terminal_record = entry["branch_artifact"], entry["terminal_artifact"]
            audit.require(completion_index.get(branch_record["name"]) == branch_record
                          and completion_index.get(terminal_record["name"]) == terminal_record,
                          f"{identity}: completion artifact record differs")
            branch = a10.read_json(root / branch_record["name"])
            terminal = torch.load(root / terminal_record["name"], weights_only=True,
                                  map_location="cpu")
            for name in ENTRY_META:
                audit.require(branch.get(name) == entry.get(name) == terminal.get(name),
                              f"{identity}: metadata differs for {name}")
            edited = edited_cache[(entry["seed"], entry["parent_step"], entry["arm"])]
            parent = parent_cache[(entry["seed"], entry["parent_step"])]
            baseline = baselines[(entry["seed"], entry["parent_step"],
                                  entry["objective"], entry["policy"])]
            audit.require(entry["edited_parent_artifact"] == edited["artifact"]
                          and entry["edited_parent_complete_digest"] == edited["digest"],
                          f"{identity}: edited-parent binding differs")
            audit.require(entry["parent_artifact"]["name"] == parent["artifact_name"]
                          and entry["parent_complete_digest"] == parent["digest"],
                          f"{identity}: original-parent binding differs")
            audit.require(entry["baseline_branch_artifact"]["name"]
                          == baseline["branch_artifact"]
                          and entry["baseline_final_artifact"]["name"]
                          == baseline["terminal_artifact"],
                          f"{identity}: inherited baseline binding differs")
            audit.require(entry["plan_file"] == f"plan-s{entry['seed']}-t{entry['parent_step']}.json"
                          and entry["initial_evaluation_exact"] is True,
                          f"{identity}: plan/seam declaration differs")
            copied_plan = root / entry["plan_file"]
            audit.require(completion_index.get(entry["plan_file"], {}).get("sha256")
                          == parent_inputs["i10_hashes"].get(entry["plan_file"]),
                          f"{identity}: copied plan hash differs from I10")

            status = branch["status"]
            audit.require(status in ("complete", "numerical_failure")
                          and status == entry["status"], f"{identity}: status differs")
            completed, attempted = int(branch["completed_steps"]), int(branch["attempted_step"])
            audit.require(completed == len(branch["steps"])
                          and 0 <= completed <= attempted <= 500
                          and attempted - completed in (0, 1),
                          f"{identity}: completed/attempted counters differ")
            curves = curve_map(branch["curve"], audit, identity)
            audit.require(0 in curves and branch["curve"][0] == baseline["curve"][0],
                          f"{identity}: h0 seam differs from inherited reference")
            if status == "complete":
                audit.require(completed == attempted == 500 and tuple(curves) == HORIZONS
                              and branch.get("numerical_failure") is None
                              and branch.get("last_evaluated_artifact") is None
                              and branch.get("last_evaluated_horizon") is None,
                              f"{identity}: complete branch fields differ")
            else:
                audit.require(branch.get("numerical_failure") is not None
                              and tuple(curves) != HORIZONS,
                              f"{identity}: numerical failure has a fabricated full endpoint")
                failure_rows.append({"id": identity, "completed_steps": completed,
                    "attempted_step": attempted, "last_evaluated_horizon":
                    branch.get("last_evaluated_horizon"),
                    "numerical_failure": branch.get("numerical_failure")})

            diagnostic = a11.validate_steps(branch["steps"], count=completed,
                policy=entry["policy"], parent_observer=parent["observer"], cumulative_offset=0,
                require_cumulative_field=False, audit=audit, label=identity)
            terminal_state = terminal["state"]
            declared_digest = terminal.get("complete_state_digest")
            try:
                actual_digest = a10.tree_digest(terminal_state)
                audit.tree_digests += 1
                audit.require(actual_digest == declared_digest
                              == branch.get("terminal_complete_digest"),
                              f"{identity}: terminal state digest differs")
                audit.require(terminal.get("complete_state_digest_unavailable_reason") is None
                              and branch.get("terminal_digest_unavailable_reason") is None,
                              f"{identity}: terminal digest reason is unexpectedly populated")
            except ValueError as exc:
                audit.require(status == "numerical_failure" and declared_digest is None
                              and branch.get("terminal_complete_digest") is None
                              and terminal.get("complete_state_digest_unavailable_reason")
                              == branch.get("terminal_digest_unavailable_reason")
                              and terminal.get("complete_state_digest_unavailable_reason") is not None
                              and "non-finite float" in str(exc),
                              f"{identity}: unavailable terminal digest is unjustified")
            start_counter = a10.optimizer_counter(edited["state"])
            terminal_counter = a10.optimizer_counter(terminal_state)
            if status == "complete":
                audit.require(terminal_counter == start_counter + 500,
                              f"{identity}: complete terminal counter differs")
            else:
                audit.require(start_counter + completed <= terminal_counter
                              <= start_counter + attempted,
                              f"{identity}: failed terminal counter outside attempted range")
            observer_delta = int(terminal_state["tracker"]["step_count"]) - parent["observer"]
            audit.require((observer_delta == 500 if status == "complete"
                           else completed <= observer_delta <= attempted),
                          f"{identity}: terminal observer counter differs")

            last_record = branch.get("last_evaluated_artifact")
            if status == "numerical_failure":
                audit.require(entry.get("last_evaluated_artifact") == last_record
                              and last_record is not None
                              and completion_index.get(last_record["name"]) == last_record,
                              f"{identity}: last-evaluated record differs")
                last = torch.load(root / last_record["name"], weights_only=True, map_location="cpu")
                for name in ENTRY_META:
                    audit.require(last.get(name) == entry.get(name),
                                  f"{identity}: last-evaluated metadata differs for {name}")
                last_horizon = branch.get("last_evaluated_horizon")
                audit.require(last_horizon == max(curves) == last.get("last_evaluated_horizon"),
                              f"{identity}: last-evaluated horizon differs")
                last_digest = a10.tree_digest(last["state"])
                audit.tree_digests += 1
                audit.require(last_digest == last["complete_state_digest"],
                              f"{identity}: last-evaluated state digest differs")
                audit.require(a10.optimizer_counter(last["state"]) == start_counter + last_horizon
                              and int(last["state"]["tracker"]["step_count"])
                              == parent["observer"] + last_horizon,
                              f"{identity}: last-evaluated counters differ")

            first_digest = branch.get("first_step_applied_gradient_digest")
            audit.require((completed == 0 and first_digest is None)
                          or (completed > 0 and type(first_digest) is str
                              and len(first_digest) == 64),
                          f"{identity}: first-gradient digest availability differs")
            first_gradient_groups.setdefault(key[:4], []).append(
                (entry["arm"], first_digest, completed))
            row = {"status": status, "completed_steps": completed,
                   "attempted_step": attempted, "curve": branch["curve"],
                   "curve_by_horizon": curves, "diagnostic_summary": diagnostic,
                   "first_step": branch["steps"][0] if completed else None,
                   "first_step_applied_gradient_digest": first_digest,
                   "branch_artifact": branch_record["name"],
                   "terminal_artifact": terminal_record["name"]}
            branches[key] = row
            new_summaries.append({"id": identity, "seed": key[0], "parent_step": key[1],
                "objective": key[2], "policy": key[3], "arm": key[4], "status": status,
                "completed_steps": completed, "attempted_step": attempted,
                "curve": branch["curve"], "diagnostic_summary": diagnostic,
                "first_step": row["first_step"],
                "first_step_applied_gradient_digest": first_digest,
                "branch_artifact": branch_record["name"],
                "terminal_artifact": terminal_record["name"],
                "last_evaluated_artifact": None if last_record is None else last_record["name"]})
        except Exception as exc:
            audit.errors.append(f"{identity}: branch analysis failed: {type(exc).__name__}: {exc}")
    audit.require(set(branches) == {(seed, anchor, objective, policy, arm)
                  for seed in SEEDS for anchor in ANCHORS for objective in OBJECTIVES
                  for policy in POLICIES for arm in ARMS}, "combined branch coverage differs")
    audit.require(len(failure_rows) == declared_failures,
                  "observed numerical-failure count differs")

    gradient_checks = []
    for key in sorted(expected_baseline_keys()):
        values = first_gradient_groups.get(key, [])
        audit.require(len(values) == 4 and {arm for arm, _, _ in values} == set(NEW_ARMS),
                      f"first-gradient group {key}: arm coverage differs")
        available = all(completed > 0 for _, _, completed in values)
        equal = available and len({digest for _, digest, _ in values}) == 1
        if available:
            audit.require(equal, f"first-gradient group {key}: digests differ across arms")
        gradient_checks.append({"seed": key[0], "parent_step": key[1],
            "objective": key[2], "policy": key[3], "available": available,
            "equal_across_four_edits": equal,
            "digests": {arm: digest for arm, digest, _ in values}})

    benefit_rows = all_benefits(branches)
    factorials = factorial_rows(branches)
    primaries = primary(factorials)
    any_primary_missing = any(not row["available"] for row in primaries)
    expected_primary_missing = any(entry["status"] == "numerical_failure"
                                   and entry["arm"] in ("zero_m", "zero_v")
                                   for entry in entries)
    audit.require(any_primary_missing is expected_primary_missing,
        "primary availability does not match relevant missing endpoints")

    summary = {
        "schema": "i12_moment_analysis_summary_v1",
        "scientific_scope": [
            "Adaptive three-seed saved-state interventions; not an independent replication.",
            "Positive utility benefits favor current32 over raw under the same moment arm.",
            "Primary Delta_v_minus_m is B_zero_v minus B_zero_m at h500, separately by objective/metric.",
            "Missing paired endpoints make the affected primary unavailable; no survivor mean or imputation.",
            "Moment interventions are dynamic treatment effects, not a mediation fraction or Adam recommendation.",
        ],
        "artifact_directory": str(root),
        "counts": {"new_branches": len(entries), "inherited_references": len(baselines),
                   "edited_parents": len(edited_cache), "original_parents": len(parent_cache),
                   "numerical_failures": len(failure_rows),
                   "science_complete": declared_failures == 0},
        "utility_definitions": {
            "ce": "negative auxiliary clean CE (higher is better)",
            "accuracy": "auxiliary clean accuracy (higher is better)",
            "benefit": "utility(current32, arm) - utility(raw, arm)",
            "primary": "benefit(zero_v) - benefit(zero_m)",
        },
        "primary_four_h500_delta_v_minus_m": primaries,
        "all_horizon_arm_benefits_and_relative_inherited": benefit_rows,
        "all_horizon_factorials": factorials,
        "absolute_policy_progress_and_relative_inherited": absolute_progress(branches),
        "diagnostic_cohorts": diagnostic_cohorts(branches),
        "first_step_gradient_cross_arm_checks": gradient_checks,
        "numerical_failures": failure_rows,
        "new_branch_curves_and_diagnostics": new_summaries,
        "inherited_branch_curves_and_diagnostics": baseline_summaries,
        "original_references": {
            "parent_inputs": parent_inputs,
            "baseline_reference_artifact": "baseline-references.json",
            "branches_artifact": "branches.json",
        },
    }
    audit_payload = {
        "schema": "i12_moment_analysis_audit_v1",
        "status": "pass" if not audit.errors else "fail",
        "checks": audit.checks, "errors": audit.errors, "warnings": audit.warnings,
        "completion_and_external_hash_files_verified": audit.hash_files,
        "hash_bytes_streamed": audit.hash_bytes,
        "complete_state_tree_digests_verified": audit.tree_digests,
        "maximum_train_residual_identity_error": audit.max_residual_error,
        "maximum_leakage_energy_identity_error": audit.max_leakage_identity_error,
        "coverage": {"new_branches": len(entries), "inherited_references": len(baselines),
                     "edited_parents": len(edited_cache), "original_parents": len(parent_cache),
                     "first_gradient_groups": len(gradient_checks)},
        "numerical_failures_are_retained_outcomes": len(failure_rows),
        "science_complete": declared_failures == 0,
        "energy_weighted_leakage_definition":
            "sum outside squared step energy / sum total squared step energy within branch, then equal parent and seed weights",
        "audit_limits": [
            "Recorded finite evaluations were schema/identity checked, not replayed from model/data forwards.",
            "Per-step leakage used saved scalar norm/outside-energy identities; displacement vectors were not saved.",
            "First-step gradient equality uses acquisition digests; gradient tensors were not stored for recomputation.",
        ],
    }
    for name, payload in (("summary.json", summary), ("audit.json", audit_payload)):
        with (args.output / name).open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, allow_nan=False)
            handle.write("\n")
    print(json.dumps({"status": audit_payload["status"], "new": len(entries),
                      "failures": len(failure_rows), "output": str(args.output)},
                     allow_nan=False))
    return 0 if not audit.errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
