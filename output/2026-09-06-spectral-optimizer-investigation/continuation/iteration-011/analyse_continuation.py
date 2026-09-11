#!/usr/bin/env python3
"""CPU-only audit and predeclared analysis of completed I11 continuations."""
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
_A10_PATH = HERE.parent / "iteration-010" / "analyse_branches.py"
_A10_SPEC = importlib.util.spec_from_file_location("_i11_analysis_helpers", _A10_PATH)
if _A10_SPEC is None or _A10_SPEC.loader is None:  # pragma: no cover
    raise ImportError("cannot load independent I10 analysis helpers")
a10 = importlib.util.module_from_spec(_A10_SPEC)
_A10_SPEC.loader.exec_module(a10)

SEEDS = (100, 101, 102)
SOURCES = ("raw", "current32")
ANCHORS = (100, 500, 1500, 2000)
POLICIES = ("raw", "current32", "frozen32")
I10_HORIZONS = (0, 1, 10, 50, 100, 250, 500)
I11_LOCAL_HORIZONS = (0, 100, 500, 1000, 1500)
CUMULATIVE_HORIZONS = (0, 1, 10, 50, 100, 250, 500, 600, 1000, 1500, 2000)
META_KEYS = ("id", "seed", "i9_parent_step", "physical_source", "logical_sources",
             "objective", "policy", "i10_branch_artifact", "i10_final_artifact",
             "i10_complete_digest", "i9_parent_artifact", "i9_parent_complete_digest",
             "frozen_reference", "plan_file", "completed_horizon",
             "completed_continuation_horizon", "seam_evaluation_exact")


def metadata(value: dict[str, Any]) -> dict[str, Any]:
    return {key: value[key] for key in META_KEYS}


def expected_physical_keys() -> set[tuple[int, str, int, str]]:
    return {(seed, source, anchor, policy) for seed in SEEDS for anchor in ANCHORS
            for source in (("raw",) if anchor == 100 else SOURCES) for policy in POLICIES}


def evaluation_without_horizons(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items()
            if key not in ("horizon", "continuation_horizon")}


def selected_basis(tracker: dict[str, Any]) -> torch.Tensor:
    value = tracker.get("V")
    if type(value) is not torch.Tensor or value.ndim != 2:
        raise ValueError("I9 parent has no matrix basis")
    proj_k = tracker.get("proj_k")
    k = value.shape[1] if proj_k is None else min(int(proj_k), value.shape[1])
    if not 0 < k <= 32:
        raise ValueError("I9 selected basis rank is outside 1..32")
    return value[:, :k].detach().cpu()


def validate_steps(steps: list[dict[str, Any]], *, count: int, policy: str,
                   parent_observer: int, cumulative_offset: int,
                   require_cumulative_field: bool, audit: Any, label: str) -> dict[str, Any]:
    audit.require(len(steps) == count, f"{label}: expected {count} step diagnostics")
    sums = {"objective_loss": 0.0, "raw_gradient_norm": 0.0,
            "applied_gradient_norm": 0.0, "total_step_norm": 0.0,
            "data_step_norm": 0.0}
    leakage = {(kind, basis): [0.0, 0.0] for kind in ("total", "data")
               for basis in ("frozen_basis", "current_basis")}
    for index, row in enumerate(steps, start=1):
        prefix = f"{label} step {index}"
        audit.require(row.get("horizon") == index, f"{prefix}: local horizon differs")
        if require_cumulative_field:
            audit.require(row.get("continuation_horizon") == cumulative_offset + index,
                          f"{prefix}: cumulative horizon differs")
        audit.require(row.get("schema") == "i10_branch_step_v1", f"{prefix}: schema differs")
        audit.require(row.get("policy") == policy, f"{prefix}: policy differs")
        observer = row["observer"]
        expected_before = parent_observer if policy == "frozen32" else parent_observer + index - 1
        expected_after = expected_before if policy == "frozen32" else expected_before + 1
        audit.require(observer["step_before"] == expected_before and
                      observer["step_after"] == expected_after,
                      f"{prefix}: observer counters differ")
        audit.require(observer["used"] is (policy != "frozen32"),
                      f"{prefix}: observer-used flag differs")
        audit.require(row["gradient_filter_applied"] is (policy != "raw"),
                      f"{prefix}: filtering flag differs")
        if policy == "current32":
            audit.require(observer["filtering_active"] is True,
                          f"{prefix}: current filtering inactive")
        elif policy == "frozen32":
            audit.require(observer["filtering_active"] is False,
                          f"{prefix}: frozen observer active")
        values = {
            "objective_loss": a10.finite(row["loss"], prefix + " loss"),
            "raw_gradient_norm": a10.finite(row["gradient"]["raw_norm"], prefix + " raw norm"),
            "applied_gradient_norm": a10.finite(row["gradient"]["applied_norm"],
                                                  prefix + " applied norm"),
            "total_step_norm": a10.finite(row["displacement"]["total_norm"],
                                            prefix + " total norm"),
            "data_step_norm": a10.finite(row["displacement"]["data_norm"],
                                           prefix + " data norm"),
        }
        for name, value in values.items():
            audit.require(value >= 0, f"{prefix}: {name} is negative")
            sums[name] += value
        for kind, norm_name in (("total", "total_step_norm"), ("data", "data_step_norm")):
            expected_squared = values[norm_name] ** 2
            for basis in ("frozen_basis", "current_basis"):
                saved = a10.leakage_energy(row["displacement"][kind + "_leakage"][basis],
                                           expected_squared, audit,
                                           f"{prefix} {kind}/{basis}")
                if saved is not None:
                    leakage[(kind, basis)][0] += saved[0]
                    leakage[(kind, basis)][1] += saved[1]
    result = {
        "step_count": len(steps),
        "arithmetic_step_means": {name: value / len(steps) if steps else None
                                   for name, value in sums.items()},
        "arithmetic_step_sums": sums,
        "energy_weighted_leakage": {},
        "observer": {"parent": parent_observer,
                     "final": parent_observer if policy == "frozen32"
                     else parent_observer + len(steps)},
    }
    for (kind, basis), (outside, total) in leakage.items():
        result["energy_weighted_leakage"][kind + "_against_" + basis] = {
            "outside_energy": outside, "total_energy": total,
            "fraction": None if total == 0 else outside / total,
        }
    return result


def combine_diagnostics(first: dict[str, Any], second: dict[str, Any]) -> dict[str, Any]:
    count = first["step_count"] + second["step_count"]
    sums = {name: first["arithmetic_step_sums"][name] + second["arithmetic_step_sums"][name]
            for name in first["arithmetic_step_sums"]}
    leakage = {}
    for name in first["energy_weighted_leakage"]:
        outside = (first["energy_weighted_leakage"][name]["outside_energy"]
                   + second["energy_weighted_leakage"][name]["outside_energy"])
        total = (first["energy_weighted_leakage"][name]["total_energy"]
                 + second["energy_weighted_leakage"][name]["total_energy"])
        leakage[name] = {"outside_energy": outside, "total_energy": total,
                         "fraction": None if total == 0 else outside / total}
    return {"step_count": count,
            "arithmetic_step_means": {name: value / count for name, value in sums.items()},
            "arithmetic_step_sums": sums, "energy_weighted_leakage": leakage}


def numeric_changes(level: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    result = {}
    for group in ("train", "auxiliary", "validation"):
        result[group] = {key: level[group][key] - reference[group][key]
                         for key in level[group]}
    return result


def benefit(branches: dict[tuple[Any, ...], dict[str, Any]], seed: int, source: str,
            anchor: int, horizon: int, policy: str, metric: str) -> float:
    raw = branches[(seed, source, anchor, "raw")]["stitched_by_horizon"][horizon]
    selected = branches[(seed, source, anchor, policy)]["stitched_by_horizon"][horizon]
    if metric == "ce":
        return raw["auxiliary"]["clean_ce"] - selected["auxiliary"]["clean_ce"]
    if metric == "accuracy":
        return selected["auxiliary"]["clean_accuracy"] - raw["auxiliary"]["clean_accuracy"]
    raise ValueError(metric)


def primary(branches: dict[tuple[Any, ...], dict[str, Any]]) -> dict[str, Any]:
    per_seed = []
    for seed in SEEDS:
        parent_values = {str(anchor): {
            "ce_benefit": benefit(branches, seed, "current32", anchor, 2000,
                                  "current32", "ce"),
            "accuracy_benefit": benefit(branches, seed, "current32", anchor, 2000,
                                        "current32", "accuracy"),
        } for anchor in (1500, 2000)}
        means = {name: float(np.mean([parent_values[str(anchor)][name]
                                     for anchor in (1500, 2000)]))
                 for name in ("ce_benefit", "accuracy_benefit")}
        per_seed.append({"seed": seed, "parent_values": parent_values,
                         "seed_first_parent_mean": means})
    across = {}
    for name in ("ce_benefit", "accuracy_benefit"):
        values = [row["seed_first_parent_mean"][name] for row in per_seed]
        across[name] = {"all_three_seed_values": values, "mean": float(np.mean(values))}
    return {"source": "current32", "parents": [1500, 2000], "cumulative_horizon": 2000,
            "positive_favors": "current32", "per_seed": per_seed, "across_seeds": across}


def mandatory_benefits(branches: dict[tuple[Any, ...], dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for source in SOURCES:
        for anchor in ANCHORS:
            for horizon in CUMULATIVE_HORIZONS:
                for policy in ("current32", "frozen32"):
                    rows = [{"seed": seed,
                             "ce_benefit": benefit(branches, seed, source, anchor,
                                                   horizon, policy, "ce"),
                             "accuracy_benefit": benefit(branches, seed, source, anchor,
                                                         horizon, policy, "accuracy")}
                            for seed in SEEDS]
                    result.append({"source": source, "anchor": anchor,
                                   "cumulative_horizon": horizon, "policy_vs_raw": policy,
                                   "step100_source_alias": anchor == 100,
                                   "per_seed": rows,
                                   "seed_means": {
                                       "ce_benefit": float(np.mean([row["ce_benefit"] for row in rows])),
                                       "accuracy_benefit": float(np.mean(
                                           [row["accuracy_benefit"] for row in rows])),
                                   }})
    return result


def hindsight_raw(branches: dict[tuple[Any, ...], dict[str, Any]]) -> dict[str, Any]:
    per_seed = []
    for seed in SEEDS:
        grid = []
        for horizon in CUMULATIVE_HORIZONS:
            raw_rows = [branches[(seed, "current32", anchor, "raw")]
                        ["stitched_by_horizon"][horizon] for anchor in (1500, 2000)]
            grid.append({"horizon": horizon,
                         "raw_parent_mean_clean_ce": float(np.mean(
                             [row["auxiliary"]["clean_ce"] for row in raw_rows])),
                         "raw_parent_mean_clean_accuracy": float(np.mean(
                             [row["auxiliary"]["clean_accuracy"] for row in raw_rows]))})
        # Python min/max retain the first item on exact ties; grid is horizon-sorted.
        best_ce = min(grid, key=lambda row: row["raw_parent_mean_clean_ce"])
        best_accuracy = max(grid, key=lambda row: row["raw_parent_mean_clean_accuracy"])
        current_rows = [branches[(seed, "current32", anchor, "current32")]
                        ["stitched_by_horizon"][2000] for anchor in (1500, 2000)]
        current_ce = float(np.mean([row["auxiliary"]["clean_ce"] for row in current_rows]))
        current_accuracy = float(np.mean(
            [row["auxiliary"]["clean_accuracy"] for row in current_rows]))
        per_seed.append({
            "seed": seed, "raw_grid_after_parent_mean": grid,
            "raw_min_clean_ce": {"value": best_ce["raw_parent_mean_clean_ce"],
                                 "earliest_exact_tie_horizon": best_ce["horizon"]},
            "raw_max_clean_accuracy": {"value": best_accuracy["raw_parent_mean_clean_accuracy"],
                                       "earliest_exact_tie_horizon": best_accuracy["horizon"]},
            "current32_cumulative2000": {"clean_ce": current_ce,
                                         "clean_accuracy": current_accuracy},
            "signed_gaps_positive_favors_current32": {
                "raw_min_ce_minus_current32_ce": best_ce["raw_parent_mean_clean_ce"] - current_ce,
                "current32_accuracy_minus_raw_max_accuracy":
                    current_accuracy - best_accuracy["raw_parent_mean_clean_accuracy"],
            },
        })
    names = ("raw_min_ce_minus_current32_ce", "current32_accuracy_minus_raw_max_accuracy")
    return {"selection_scope": "within seed after averaging raw anchors 1500/2000 at each grid point",
            "grid": list(CUMULATIVE_HORIZONS), "same_auxiliary_data_hindsight": True,
            "per_seed": per_seed,
            "mean_signed_gaps": {name: float(np.mean([
                row["signed_gaps_positive_favors_current32"][name] for row in per_seed]))
                for name in names}}


def diagnostic_cohorts(branches: dict[tuple[Any, ...], dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for source in SOURCES:
        for anchor in ANCHORS:
            for policy in POLICIES:
                seed_rows = []
                for seed in SEEDS:
                    branch = branches[(seed, source, anchor, policy)]
                    row: dict[str, Any] = {"seed": seed}
                    for window in ("i11_500_to2000", "combined_0_to2000"):
                        diagnostic = branch["diagnostics"][window]
                        for name, value in diagnostic["arithmetic_step_means"].items():
                            row[f"{window}/mean/{name}"] = value
                        for name, value in diagnostic["energy_weighted_leakage"].items():
                            row[f"{window}/energy_leakage/{name}"] = value["fraction"]
                    seed_rows.append(row)
                names = [name for name in seed_rows[0] if name != "seed"]
                result.append({"source": source, "anchor": anchor, "policy": policy,
                               "step100_source_alias": anchor == 100,
                               "per_seed": seed_rows,
                               "equal_seed_means_of_within_branch_values": {
                                   name: float(np.mean([row[name] for row in seed_rows]))
                                   for name in names}})
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
    audit.require(manifest.get("mode") == "full" and manifest.get("unique_branches") == 63
                  and manifest.get("logical_cells") == 72,
                  "I11 manifest mode/counts differ")
    audit.require(manifest.get("horizons") == list(I11_LOCAL_HORIZONS)
                  and manifest.get("continuation_horizons") == [500, 600, 1000, 1500, 2000],
                  "I11 manifest horizons differ")
    audit.require(completion.get("status") == "complete" and completion.get("mode") == "full",
                  "I11 completion status/mode differs")
    completion_records = completion["artifacts"]
    completion_index = {record["name"]: record for record in completion_records}
    audit.require(len(completion_index) == len(completion_records),
                  "I11 completion artifact names are duplicated")
    for record in completion_records:
        a10.verify_record(root, record, audit, "I11 completion")

    parent_inputs = a10.read_json(root / "parent-inputs.json")
    i10_root = Path(parent_inputs["i10_directory"]).resolve(strict=True)
    i9_root = Path(parent_inputs["i9_directory"]).resolve(strict=True)
    audit.require(i10_root.is_dir() and i9_root.is_dir(), "I10/I9 parent directories missing")
    for parent_root, hashes, label in ((i10_root, parent_inputs["i10_hashes"], "I10"),
                                       (i9_root, parent_inputs["i9_hashes"], "I9")):
        for name, expected in hashes.items():
            path = parent_root / name
            audit.require(path.is_file(), f"{label} parent missing {name}")
            if path.is_file():
                audit.require(a10.sha256(path) == expected, f"{label} parent hash differs {name}")
                audit.hash_files += 1
                audit.hash_bytes += path.stat().st_size

    branch_index = a10.read_json(root / "branches.json")
    audit.require(branch_index.get("unique_branches") == 63
                  and branch_index.get("logical_cells") == 72
                  and branch_index.get("parent_hashes_unchanged") is True,
                  "I11 branches index status/counts differ")
    entries = branch_index["entries"]
    audit.require(len(entries) == 63, "I11 branches index does not contain 63 entries")

    i10_json_cache: dict[str, dict[str, Any]] = {}
    i10_state_cache: dict[str, dict[str, Any]] = {}
    i9_state_cache: dict[str, dict[str, Any]] = {}
    plan_cache: dict[str, dict[str, Any]] = {}
    physical: dict[tuple[Any, ...], dict[str, Any]] = {}
    logical: dict[tuple[Any, ...], dict[str, Any]] = {}
    physical_summaries = []

    for entry in entries:
        identity = entry.get("id", "unknown-entry")
        try:
            expected_id = (f"s{entry['seed']}-{entry['physical_source']}-"
                           f"t{entry['i9_parent_step']}-fixed-{entry['policy']}")
            audit.require(identity == expected_id, f"{identity}: ID differs from metadata")
            audit.require(entry["physical_source"] in SOURCES and entry["policy"] in POLICIES,
                          f"{identity}: source/policy differs")
            audit.require(metadata(entry)["objective"] == "fixed", f"{identity}: objective differs")
            expected_sources = (["raw", "current32"] if entry["i9_parent_step"] == 100
                                else [entry["physical_source"]])
            audit.require(entry["logical_sources"] == expected_sources,
                          f"{identity}: logical-source aliases differ")
            audit.require(entry["completed_horizon"] == 1500
                          and entry["completed_continuation_horizon"] == 2000
                          and entry["seam_evaluation_exact"] is True,
                          f"{identity}: completion/seam metadata differs")
            audit.require(entry["frozen_reference"] ==
                          "original I9-parent basis, unchanged across I10/I11",
                          f"{identity}: frozen reference convention differs")

            branch_record, final_record = entry["branch_artifact"], entry["final_artifact"]
            audit.require(completion_index.get(branch_record["name"]) == branch_record,
                          f"{identity}: branch record differs from completion")
            audit.require(completion_index.get(final_record["name"]) == final_record,
                          f"{identity}: final record differs from completion")
            branch = a10.read_json(root / branch_record["name"])
            final_binding = torch.load(root / final_record["name"], weights_only=True,
                                       map_location="cpu")
            audit.require(metadata(branch) == metadata(entry), f"{identity}: branch metadata differs")
            audit.require(metadata(final_binding) == metadata(entry),
                          f"{identity}: final metadata differs")
            audit.require(branch["final_artifact"] == final_record,
                          f"{identity}: final record binding differs")
            final_digest = a10.tree_digest(final_binding["state"])
            audit.tree_digests += 1
            audit.require(final_binding["state"].get("schema") == "i9_neural_snapshot_v1",
                          f"{identity}: I11 final snapshot schema differs")
            audit.require(final_digest == final_binding["complete_state_digest"]
                          == branch["final_complete_digest"],
                          f"{identity}: I11 final digest differs")

            plan_name = entry["plan_file"]
            audit.require(plan_name == f"plan-s{entry['seed']}-t{entry['i9_parent_step']}.json",
                          f"{identity}: plan filename differs")
            audit.require(plan_name in completion_index, f"{identity}: plan absent from completion")
            if plan_name not in plan_cache:
                plan_cache[plan_name] = a10.read_json(root / plan_name)
            audit.require((plan_cache[plan_name]["seed"], plan_cache[plan_name]["i9_parent_step"])
                          == (entry["seed"], entry["i9_parent_step"]),
                          f"{identity}: plan metadata differs")

            i10_branch_record = entry["i10_branch_artifact"]
            i10_final_record = entry["i10_final_artifact"]
            for record in (i10_branch_record, i10_final_record):
                audit.require(parent_inputs["i10_hashes"].get(record["name"]) == record["sha256"],
                              f"{identity}: I10 hash binding differs for {record['name']}")
            if i10_branch_record["name"] not in i10_json_cache:
                i10_json_cache[i10_branch_record["name"]] = a10.read_json(
                    i10_root / i10_branch_record["name"])
            i10_branch = i10_json_cache[i10_branch_record["name"]]
            if i10_final_record["name"] not in i10_state_cache:
                loaded = torch.load(i10_root / i10_final_record["name"], weights_only=True,
                                    map_location="cpu")
                loaded_digest = a10.tree_digest(loaded["state"])
                audit.tree_digests += 1
                audit.require(loaded_digest == loaded["complete_state_digest"],
                              f"I10 {i10_final_record['name']}: state digest differs")
                i10_state_cache[i10_final_record["name"]] = loaded
            i10_final = i10_state_cache[i10_final_record["name"]]
            audit.require(a10.metadata(i10_final) == a10.metadata(i10_branch),
                          f"{identity}: I10 branch/final metadata differs")
            audit.require(i10_branch["final_artifact"] == i10_final_record,
                          f"{identity}: I10 final artifact binding differs")
            audit.require(i10_final["state"].get("schema") == "i9_neural_snapshot_v1",
                          f"{identity}: I10 final snapshot schema differs")
            audit.require(entry["i10_complete_digest"] == i10_final["complete_state_digest"]
                          == i10_branch["final_complete_digest"],
                          f"{identity}: I10 complete digest binding differs")
            audit.require((i10_branch["seed"], i10_branch["physical_source"],
                           i10_branch["parent_step"], i10_branch["objective"],
                           i10_branch["policy"])
                          == (entry["seed"], entry["physical_source"],
                              entry["i9_parent_step"], "fixed", entry["policy"]),
                          f"{identity}: I10 branch metadata differs")

            i9_record = entry["i9_parent_artifact"]
            audit.require(parent_inputs["i9_hashes"].get(i9_record["name"]) == i9_record["sha256"],
                          f"{identity}: I9 parent hash binding differs")
            if i9_record["name"] not in i9_state_cache:
                loaded = torch.load(i9_root / i9_record["name"], weights_only=True,
                                    map_location="cpu")
                loaded_digest = a10.tree_digest(loaded["state"])
                audit.tree_digests += 1
                audit.require(loaded_digest == loaded["complete_state_digest"],
                              f"I9 {i9_record['name']}: state digest differs")
                audit.require((loaded["seed"], loaded["source"], loaded["completed_step"])
                              == (entry["seed"], entry["physical_source"],
                                  entry["i9_parent_step"]),
                              f"I9 {i9_record['name']}: metadata differs")
                basis = selected_basis(loaded["state"]["tracker"])
                i9_state_cache[i9_record["name"]] = {
                    "binding": loaded, "basis_digest": a10.tree_digest(basis),
                    "basis_rank": basis.shape[1],
                    "tracker_digest": a10.tree_digest(loaded["state"]["tracker"]),
                }
            i9_info = i9_state_cache[i9_record["name"]]
            i9_binding = i9_info["binding"]
            audit.require(i9_binding["state"].get("schema") == "i9_neural_snapshot_v1",
                          f"{identity}: I9 parent snapshot schema differs")
            audit.require(i9_binding["plan_file"] == f"plan-s{entry['seed']}.json",
                          f"{identity}: I9 plan binding differs")
            audit.require(entry["i9_parent_complete_digest"] ==
                          i9_binding["complete_state_digest"] ==
                          i10_branch["parent_complete_digest"],
                          f"{identity}: I9 parent digest binding differs")

            i10_curve = i10_branch["curve"]
            audit.require([row["horizon"] for row in i10_curve] == list(I10_HORIZONS),
                          f"{identity}: I10 curve horizons differ")
            for row in i10_curve:
                a10.validate_evaluation(row, audit, f"{identity} I10 h{row['horizon']}")
            i11_curve = branch["curve"]
            audit.require([row["horizon"] for row in i11_curve] == list(I11_LOCAL_HORIZONS),
                          f"{identity}: I11 local horizons differ")
            audit.require([row["continuation_horizon"] for row in i11_curve]
                          == [500, 600, 1000, 1500, 2000],
                          f"{identity}: I11 cumulative horizons differ")
            for row in i11_curve:
                a10.validate_evaluation(row, audit, f"{identity} I11 h{row['horizon']}")
            audit.require(evaluation_without_horizons(i11_curve[0]) ==
                          evaluation_without_horizons(i10_curve[-1]),
                          f"{identity}: I10/I11 seam evaluation differs")

            i9_state = i9_binding["state"]
            i10_state = i10_final["state"]
            i11_state = final_binding["state"]
            i9_observer = int(i9_state["tracker"]["step_count"])
            i10_observer = int(i10_state["tracker"]["step_count"])
            i11_observer = int(i11_state["tracker"]["step_count"])
            expected_i10_observer = i9_observer if entry["policy"] == "frozen32" else i9_observer + 500
            expected_i11_observer = i10_observer if entry["policy"] == "frozen32" else i10_observer + 1500
            audit.require(i10_observer == expected_i10_observer,
                          f"{identity}: I10 observer counter differs")
            audit.require(i11_observer == expected_i11_observer,
                          f"{identity}: I11 observer counter differs")
            audit.require(a10.optimizer_counter(i10_state) == a10.optimizer_counter(i9_state) + 500,
                          f"{identity}: I10 Adam counter differs")
            audit.require(a10.optimizer_counter(i11_state) == a10.optimizer_counter(i10_state) + 1500,
                          f"{identity}: I11 Adam counter differs")
            if entry["policy"] == "frozen32":
                audit.require(a10.tree_digest(i10_state["tracker"]) ==
                              a10.tree_digest(i11_state["tracker"]) ==
                              i9_info["tracker_digest"],
                              f"{identity}: frozen observer differs from original I9 parent")

            i10_diagnostics = validate_steps(
                i10_branch["steps"], count=500, policy=entry["policy"],
                parent_observer=i9_observer, cumulative_offset=0,
                require_cumulative_field=False, audit=audit, label=identity + " I10")
            i11_diagnostics = validate_steps(
                branch["steps"], count=1500, policy=entry["policy"],
                parent_observer=i10_observer, cumulative_offset=500,
                require_cumulative_field=True, audit=audit, label=identity + " I11")
            combined = combine_diagnostics(i10_diagnostics, i11_diagnostics)

            stitched = [*i10_curve, *i11_curve[1:]]
            stitched_by_horizon = {}
            for row in stitched:
                horizon = row["horizon"] if "continuation_horizon" not in row else row["continuation_horizon"]
                stitched_by_horizon[horizon] = evaluation_without_horizons(row)
            audit.require(tuple(stitched_by_horizon) == CUMULATIVE_HORIZONS,
                          f"{identity}: stitched cumulative horizons differ")
            parent_level, seam_level = stitched_by_horizon[0], stitched_by_horizon[500]
            changes = []
            for horizon in (500, 600, 1000, 1500, 2000):
                level = stitched_by_horizon[horizon]
                changes.append({"cumulative_horizon": horizon,
                                "change_from_i9_parent_h0": numeric_changes(level, parent_level),
                                "change_from_own_i10_endpoint_h500": numeric_changes(level, seam_level)})

            row = {"id": identity, "seed": entry["seed"],
                   "physical_source": entry["physical_source"],
                   "logical_sources": entry["logical_sources"],
                   "i9_parent_step": entry["i9_parent_step"], "policy": entry["policy"],
                   "i9_basis_rank": i9_info["basis_rank"],
                   "i9_basis_digest": i9_info["basis_digest"],
                   "stitched_curve": [{"cumulative_horizon": horizon,
                                       **stitched_by_horizon[horizon]}
                                      for horizon in CUMULATIVE_HORIZONS],
                   "stitched_by_horizon": stitched_by_horizon,
                   "absolute_changes": changes,
                   "diagnostics": {"i10_0_to500": i10_diagnostics,
                                   "i11_500_to2000": i11_diagnostics,
                                   "combined_0_to2000": combined},
                   "branch_artifact": branch_record["name"],
                   "final_artifact": final_record["name"]}
            pkey = (entry["seed"], entry["physical_source"],
                    entry["i9_parent_step"], entry["policy"])
            audit.require(pkey not in physical, f"duplicate physical key {pkey}")
            physical[pkey] = row
            for source in entry["logical_sources"]:
                lkey = (entry["seed"], source, entry["i9_parent_step"], entry["policy"])
                audit.require(lkey not in logical, f"duplicate logical key {lkey}")
                logical[lkey] = row
            physical_summaries.append({key: value for key, value in row.items()
                                       if key not in ("stitched_by_horizon",)})
        except Exception as exc:
            audit.errors.append(f"{identity}: analysis failed: {type(exc).__name__}: {exc}")

    audit.require(set(physical) == expected_physical_keys(), "physical branch coverage differs")
    expected_logical = {(seed, source, anchor, policy) for seed in SEEDS for source in SOURCES
                        for anchor in ANCHORS for policy in POLICIES}
    audit.require(set(logical) == expected_logical, "logical cell coverage differs")
    logical_references = [{"seed": seed, "source": source, "i9_parent_step": anchor,
                           "policy": policy, "physical_branch_id": row["id"],
                           "is_step100_alias": anchor == 100 and source != row["physical_source"]}
                          for (seed, source, anchor, policy), row in sorted(logical.items())]

    summary = {
        "schema": "i11_continuation_analysis_summary_v1",
        "scientific_scope": [
            "Cumulative contrasts compare policy regimens whose I10 lineages had already diverged.",
            "Positive CE and accuracy benefits favor filtering; absolute changes distinguish preservation.",
            "The raw-grid comparator uses the same auxiliary data in hindsight and is not a stopping policy.",
            "Step-100 source aliases are references to identical physical branches, not replications.",
        ],
        "artifact_directory": str(root),
        "counts": {"physical_branches": len(physical), "logical_cells": len(logical),
                   "step100_alias_cells": sum(row["is_step100_alias"] for row in logical_references)},
        "primary_two_effects": primary(logical),
        "hindsight_raw_sampled_grid": hindsight_raw(logical),
        "mandatory_policy_benefits": mandatory_benefits(logical),
        "diagnostic_cohorts": diagnostic_cohorts(logical),
        "physical_stitched_curves_changes_and_diagnostics": physical_summaries,
        "logical_cell_references": logical_references,
        "aggregation": {
            "primary": "average parents 1500/2000 within seed, retain three seeds, then mean",
            "diagnostic_leakage": ("sum outside and total squared step energy within each branch; "
                                   "cohort summaries equally average resulting branch ratios"),
        },
    }
    audit_payload = {
        "schema": "i11_continuation_analysis_audit_v1",
        "status": "pass" if not audit.errors else "fail",
        "checks": audit.checks, "errors": audit.errors, "warnings": audit.warnings,
        "hash_files_verified": audit.hash_files, "hash_bytes_streamed": audit.hash_bytes,
        "complete_state_tree_digests_verified": audit.tree_digests,
        "maximum_train_residual_identity_error": audit.max_residual_error,
        "maximum_leakage_energy_identity_error": audit.max_leakage_identity_error,
        "coverage": {"physical": len(physical), "logical": len(logical),
                     "i10_final_states": len(i10_state_cache),
                     "i9_parent_states": len(i9_state_cache)},
        "audit_limits": [
            "Recorded finite evaluations were schema/identity checked, not replayed from MNIST/model forwards.",
            "Per-step leakage was checked from saved scalar norm/outside-energy identities; vectors were not saved.",
            "Raw/current diagnostic use of the original I9 basis is source/hash bound, not tensor-recomputed.",
        ],
    }
    with (args.output / "summary.json").open("x", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, allow_nan=False)
        handle.write("\n")
    with (args.output / "audit.json").open("x", encoding="utf-8") as handle:
        json.dump(audit_payload, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"status": audit_payload["status"], "physical": len(physical),
                      "logical": len(logical), "output": str(args.output)}, allow_nan=False))
    return 0 if not audit.errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
