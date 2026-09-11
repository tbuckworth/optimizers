#!/usr/bin/env python3
"""CPU-only integrity audit and predeclared analysis of I10 branch artifacts."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch


SEEDS = (100, 101, 102)
SOURCES = ("raw", "current32")
ANCHORS = (100, 500, 1500, 2000)
OBJECTIVES = ("fixed", "soft", "redraw")
POLICIES = ("raw", "current32", "frozen32")
HORIZONS = (0, 1, 10, 50, 100, 250, 500)
META_KEYS = ("id", "seed", "parent_step", "physical_source", "logical_sources",
             "objective", "policy", "parent_artifacts", "parent_complete_digest",
             "plan_file", "completed_horizon")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle, parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON constant {value} in {path.name}")))


def digest_node(value: Any) -> Any:
    """Independent copy of the frozen I9 complete-state digest format."""
    if type(value) is torch.Tensor:
        tensor = value.detach().cpu().contiguous()
        raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
        return ["tensor", str(tensor.dtype), list(tensor.shape), hashlib.sha256(raw).hexdigest()]
    if type(value) is dict:
        return ["dict", [[digest_node(k), digest_node(v)] for k, v in value.items()]]
    if type(value) is list:
        return ["list", [digest_node(item) for item in value]]
    if type(value) is tuple:
        return ["tuple", [digest_node(item) for item in value]]
    if value is None or type(value) in (bool, int, str):
        return [type(value).__name__, value]
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("non-finite float in complete state")
        return ["float", value.hex()]
    raise TypeError(f"unsupported complete-state value {type(value)!r}")


def tree_digest(value: Any) -> str:
    payload = json.dumps(digest_node(value), ensure_ascii=True, allow_nan=False,
                         separators=(",", ":")).encode("ascii")
    return hashlib.sha256(b"i9_neural_tree_v1\n" + payload).hexdigest()


def finite(value: Any, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} is non-finite")
    return result


class Audit:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.checks = 0
        self.hash_files = 0
        self.hash_bytes = 0
        self.tree_digests = 0
        self.max_residual_error = 0.0
        self.max_leakage_identity_error = 0.0

    def require(self, condition: bool, message: str) -> None:
        self.checks += 1
        if not condition:
            self.errors.append(message)


def verify_record(root: Path, record: dict[str, Any], audit: Audit, label: str) -> Path | None:
    try:
        name = record["name"]
        if type(name) is not str or Path(name).name != name:
            raise ValueError("record name is not a basename")
        path = root / name
        audit.require(path.is_file(), f"{label}: missing {name}")
        if not path.is_file():
            return None
        size = path.stat().st_size
        audit.require(size == int(record["bytes"]), f"{label}: byte count differs for {name}")
        audit.require(sha256(path) == record["sha256"], f"{label}: SHA256 differs for {name}")
        audit.hash_files += 1
        audit.hash_bytes += size
        return path
    except Exception as exc:
        audit.errors.append(f"{label}: malformed record: {exc}")
        return None


def metadata(value: dict[str, Any]) -> dict[str, Any]:
    return {key: value[key] for key in META_KEYS}


def optimizer_counter(snapshot: dict[str, Any]) -> int:
    states = snapshot["optimizer"]["state"]
    if not states:
        raise ValueError("optimizer has no parameter states")
    counters = []
    for row in states.values():
        value = row["step"]
        scalar = float(value.item()) if type(value) is torch.Tensor else float(value)
        if not math.isfinite(scalar) or scalar != round(scalar):
            raise ValueError("Adam step counter is not a finite integer")
        counters.append(int(scalar))
    if len(set(counters)) != 1:
        raise ValueError(f"Adam parameter counters differ: {counters}")
    return counters[0]


def validate_evaluation(row: dict[str, Any], audit: Audit, label: str) -> None:
    audit.require(row.get("schema") == "i10_branch_evaluation_v1", f"{label}: bad schema")
    expected = {
        "train": ("clean_ce", "fixed_ce", "soft_ce", "clean_accuracy", "fixed_accuracy",
                  "fixed_minus_soft_ce", "mean_max_probability", "mean_true_label_probability"),
        "auxiliary": ("clean_ce", "soft_ce", "clean_accuracy", "mean_max_probability",
                      "mean_true_label_probability"),
        "validation": ("clean_ce", "soft_ce", "clean_accuracy", "mean_max_probability",
                       "mean_true_label_probability"),
    }
    for group, keys in expected.items():
        audit.require(tuple(row[group]) == keys, f"{label}: {group} keys/order differ")
        for key in keys:
            value = finite(row[group][key], f"{label} {group}/{key}")
            if "accuracy" in key or "probability" in key:
                audit.require(0 <= value <= 1, f"{label}: {group}/{key} outside [0,1]")
            elif key != "fixed_minus_soft_ce":
                audit.require(value >= 0, f"{label}: {group}/{key} is negative")
    discrepancy = abs(row["train"]["fixed_minus_soft_ce"]
                      - (row["train"]["fixed_ce"] - row["train"]["soft_ce"]))
    audit.max_residual_error = max(audit.max_residual_error, discrepancy)
    audit.require(discrepancy <= 2e-12, f"{label}: fixed-soft residual identity differs")


def leakage_energy(row: dict[str, Any], expected_squared: float, audit: Audit,
                   label: str) -> tuple[float, float] | None:
    reason, value = row.get("reason"), row.get("fraction")
    squared = finite(row["squared_norm"], label + " squared norm")
    audit.require(math.isclose(squared, expected_squared, rel_tol=2e-6, abs_tol=1e-14),
                  f"{label}: saved squared norm differs from step norm squared")
    if reason is not None:
        audit.require(value is None, f"{label}: reason has non-null fraction")
        if reason == "zero_displacement":
            audit.require(squared == 0 and row.get("outside_squared_norm") == 0,
                          f"{label}: zero displacement energies differ")
        return None
    result = finite(value, label)
    outside = finite(row["outside_squared_norm"], label + " outside squared norm")
    discrepancy = abs(outside - squared * result)
    audit.max_leakage_identity_error = max(audit.max_leakage_identity_error, discrepancy)
    audit.require(math.isclose(outside, squared * result, rel_tol=2e-12, abs_tol=1e-18),
                  f"{label}: leakage energy identity differs")
    audit.require(-1e-7 <= result <= 1.000001, f"{label}: leakage outside numerical [0,1]")
    return outside, squared


def validate_steps(steps: list[dict[str, Any]], policy: str, parent_observer: int,
                   audit: Audit, label: str) -> dict[str, Any]:
    audit.require(len(steps) == 500, f"{label}: expected 500 step diagnostics")
    sums = {"objective_loss": 0.0, "raw_gradient_norm": 0.0, "applied_gradient_norm": 0.0,
            "total_step_norm": 0.0, "data_step_norm": 0.0}
    leakage = {(kind, basis): [0.0, 0.0] for kind in ("total", "data")
               for basis in ("frozen_basis", "current_basis")}
    for index, row in enumerate(steps, start=1):
        prefix = f"{label} step {index}"
        audit.require(row.get("horizon") == index, f"{prefix}: horizon differs")
        audit.require(row.get("schema") == "i10_branch_step_v1", f"{prefix}: bad schema")
        audit.require(row.get("policy") == policy, f"{prefix}: policy differs")
        observer = row["observer"]
        expected_before = parent_observer if policy == "frozen32" else parent_observer + index - 1
        expected_after = expected_before if policy == "frozen32" else expected_before + 1
        audit.require(observer["step_before"] == expected_before and
                      observer["step_after"] == expected_after,
                      f"{prefix}: observer counter differs")
        audit.require(observer["used"] is (policy != "frozen32"),
                      f"{prefix}: observer-used flag differs")
        audit.require(row["gradient_filter_applied"] is (policy != "raw"),
                      f"{prefix}: filter-applied flag differs")
        if policy == "current32":
            audit.require(observer["filtering_active"] is True,
                          f"{prefix}: current filter unexpectedly inactive")
        elif policy == "frozen32":
            audit.require(observer["filtering_active"] is False,
                          f"{prefix}: frozen observer unexpectedly active")
        values = {
            "objective_loss": finite(row["loss"], prefix + " loss"),
            "raw_gradient_norm": finite(row["gradient"]["raw_norm"], prefix + " raw norm"),
            "applied_gradient_norm": finite(row["gradient"]["applied_norm"], prefix + " applied norm"),
            "total_step_norm": finite(row["displacement"]["total_norm"], prefix + " total norm"),
            "data_step_norm": finite(row["displacement"]["data_norm"], prefix + " data norm"),
        }
        for name, value in values.items():
            audit.require(value >= 0, f"{prefix}: {name} negative")
            sums[name] += value
        for kind, norm_name in (("total", "total_step_norm"), ("data", "data_step_norm")):
            energy = values[norm_name] ** 2
            for basis in ("frozen_basis", "current_basis"):
                saved_energy = leakage_energy(
                    row["displacement"][kind + "_leakage"][basis], energy, audit,
                    f"{prefix} {kind}/{basis}")
                if saved_energy is not None:
                    leakage[(kind, basis)][0] += saved_energy[0]
                    leakage[(kind, basis)][1] += saved_energy[1]
    count = len(steps)
    result = {"step_count": count,
              "arithmetic_step_means": {key: value / count if count else None
                                         for key, value in sums.items()},
              "energy_weighted_leakage": {}}
    for (kind, basis), (outside_energy, total_energy) in leakage.items():
        result["energy_weighted_leakage"][kind + "_against_" + basis] = {
            "definition": "sum(step_norm_squared * per_step_leakage) / sum(step_norm_squared)",
            "outside_energy": outside_energy, "total_energy": total_energy,
            "fraction": None if total_energy == 0 else outside_energy / total_energy,
        }
    result["observer"] = {"parent": parent_observer,
                          "final": parent_observer if policy == "frozen32"
                          else parent_observer + count}
    return result


def expected_physical_keys() -> set[tuple[Any, ...]]:
    return {(seed, source, anchor, objective, policy)
            for seed in SEEDS for anchor in ANCHORS
            for source in (("raw",) if anchor == 100 else SOURCES)
            for objective in OBJECTIVES for policy in POLICIES}


def curve_at(branch: dict[str, Any], horizon: int) -> dict[str, Any]:
    return branch["curve_by_horizon"][horizon]


def benefit(branches: dict[tuple[Any, ...], dict[str, Any]], seed: int, source: str,
            anchor: int, horizon: int, policy: str, objective: str, metric: str) -> float:
    raw = branches[(seed, source, anchor, objective, "raw")]
    selected = branches[(seed, source, anchor, objective, policy)]
    if metric == "ce":
        return (curve_at(raw, horizon)["auxiliary"]["clean_ce"]
                - curve_at(selected, horizon)["auxiliary"]["clean_ce"])
    if metric == "accuracy":
        return (curve_at(selected, horizon)["auxiliary"]["clean_accuracy"]
                - curve_at(raw, horizon)["auxiliary"]["clean_accuracy"])
    raise ValueError(metric)


def six_effects(branches: dict[tuple[Any, ...], dict[str, Any]], seed: int, source: str,
                anchor: int, horizon: int, policy: str) -> dict[str, float]:
    result = {}
    for metric in ("ce", "accuracy"):
        values = {objective: benefit(branches, seed, source, anchor, horizon,
                                     policy, objective, metric)
                  for objective in OBJECTIVES}
        result[f"{metric}_fixed"] = values["fixed"]
        result[f"{metric}_fixed_minus_soft"] = values["fixed"] - values["soft"]
        result[f"{metric}_fixed_minus_redraw"] = values["fixed"] - values["redraw"]
    return result


def primary_effects(branches: dict[tuple[Any, ...], dict[str, Any]]) -> dict[str, Any]:
    per_seed = []
    for seed in SEEDS:
        anchors = {str(anchor): six_effects(branches, seed, "current32", anchor, 500, "current32")
                   for anchor in (1500, 2000)}
        names = tuple(next(iter(anchors.values())))
        means = {name: float(np.mean([anchors[str(anchor)][name]
                                     for anchor in (1500, 2000)])) for name in names}
        per_seed.append({"seed": seed, "anchor_values": anchors,
                         "seed_first_late_anchor_mean": means})
    across = {}
    for name in per_seed[0]["seed_first_late_anchor_mean"]:
        values = [row["seed_first_late_anchor_mean"][name] for row in per_seed]
        across[name] = {"all_three_seed_values": values, "mean": float(np.mean(values))}
    return {"source": "current32", "policy_vs_raw": "current32", "horizon": 500,
            "late_anchors_averaged_within_seed": [1500, 2000],
            "positive_direction": "benefit favors current32",
            "per_seed": per_seed, "across_seeds": across}


def secondary_effects(branches: dict[tuple[Any, ...], dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for source in SOURCES:
        for anchor in ANCHORS:
            for horizon in HORIZONS:
                for policy in ("current32", "frozen32"):
                    seed_rows = [{"seed": seed, **six_effects(
                        branches, seed, source, anchor, horizon, policy)} for seed in SEEDS]
                    names = [name for name in seed_rows[0] if name != "seed"]
                    rows.append({"source": source, "anchor": anchor, "horizon": horizon,
                                 "policy_vs_raw": policy,
                                 "step100_source_alias": anchor == 100,
                                 "per_seed": seed_rows,
                                 "seed_means": {name: float(np.mean([row[name] for row in seed_rows]))
                                                for name in names}})
    return rows


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
    audit = Audit()

    manifest = read_json(root / "manifest.json")
    completion = read_json(root / "completion.json")
    audit.require(manifest.get("mode") == "full", "manifest is not a full acquisition")
    audit.require(manifest.get("unique_branches") == 189 and
                  manifest.get("logical_cells") == 216 and
                  manifest.get("horizons") == list(HORIZONS),
                  "manifest branch counts/horizons differ")
    audit.require(completion.get("status") == "complete" and completion.get("mode") == "full",
                  "completion status/mode differs")
    records = completion["artifacts"]
    completion_index = {record["name"]: record for record in records}
    audit.require(len(completion_index) == len(records), "duplicate completion artifact names")
    for record in records:
        verify_record(root, record, audit, "I10 completion")

    index_path = root / "branches.json"
    branch_index = read_json(index_path)
    audit.require(branch_index.get("unique_branches") == 189 and
                  branch_index.get("logical_cells") == 216 and
                  branch_index.get("parent_hashes_unchanged") is True,
                  "branches.json declared counts/status differ")
    entries = branch_index["entries"]
    audit.require(len(entries) == 189, "branches.json does not contain 189 entries")

    parent_inputs = read_json(root / "parent-inputs.json")
    parent_root = Path(parent_inputs["directory"]).resolve(strict=True)
    audit.require(parent_root.is_dir(), "parent input directory missing")
    for name, expected_hash in parent_inputs["hashes"].items():
        path = parent_root / name
        audit.require(path.is_file(), f"parent input missing: {name}")
        if path.is_file():
            size = path.stat().st_size
            audit.require(sha256(path) == expected_hash, f"parent input SHA256 differs: {name}")
            audit.hash_files += 1
            audit.hash_bytes += size

    parent_cache: dict[str, dict[str, Any]] = {}
    plan_cache: dict[str, dict[str, Any]] = {}
    physical: dict[tuple[Any, ...], dict[str, Any]] = {}
    logical: dict[tuple[Any, ...], dict[str, Any]] = {}
    baselines: dict[tuple[int, str, int], list[tuple[str, dict[str, Any]]]] = {}
    physical_summaries = []

    for entry in entries:
        identity = entry["id"]
        try:
            expected_id = (f"s{entry['seed']}-{entry['physical_source']}-t{entry['parent_step']}-"
                           f"{entry['objective']}-{entry['policy']}")
            audit.require(identity == expected_id, f"{identity}: ID differs from metadata")
            audit.require(entry["completed_horizon"] == 500, f"{identity}: horizon differs")
            audit.require(entry["objective"] in OBJECTIVES and entry["policy"] in POLICIES,
                          f"{identity}: objective/policy invalid")
            expected_sources = (["raw", "current32"] if entry["parent_step"] == 100
                                else [entry["physical_source"]])
            audit.require(entry["logical_sources"] == expected_sources,
                          f"{identity}: logical-source aliases differ")

            branch_record = entry["branch_artifact"]
            final_record = entry["final_artifact"]
            audit.require(completion_index.get(branch_record["name"]) == branch_record,
                          f"{identity}: branch record differs from completion")
            audit.require(completion_index.get(final_record["name"]) == final_record,
                          f"{identity}: final record differs from completion")
            branch_path = root / branch_record["name"]
            final_path = root / final_record["name"]
            branch = read_json(branch_path)
            final_binding = torch.load(final_path, weights_only=True, map_location="cpu")
            audit.require(metadata(branch) == metadata(entry), f"{identity}: branch metadata differs")
            audit.require(metadata(final_binding) == metadata(entry), f"{identity}: final metadata differs")
            audit.require(branch["final_artifact"] == final_record,
                          f"{identity}: branch final record differs")
            audit.require(branch["final_complete_digest"] == final_binding["complete_state_digest"],
                          f"{identity}: final digest bindings differ")
            final_digest = tree_digest(final_binding["state"])
            audit.tree_digests += 1
            audit.require(final_digest == final_binding["complete_state_digest"],
                          f"{identity}: final complete-state digest differs")

            plan_name = entry["plan_file"]
            audit.require(plan_name == f"plan-s{entry['seed']}-t{entry['parent_step']}.json",
                          f"{identity}: plan filename differs")
            audit.require(plan_name in completion_index, f"{identity}: plan absent from completion")
            if plan_name not in plan_cache:
                plan_cache[plan_name] = read_json(root / plan_name)
            plan = plan_cache[plan_name]
            audit.require((plan["seed"], plan["parent_step"]) ==
                          (entry["seed"], entry["parent_step"]),
                          f"{identity}: plan binding differs")

            parent_names = [record["name"] for record in entry["parent_artifacts"]]
            expected_parent_names = [f"anchor-s{entry['seed']}-{source}-t{entry['parent_step']}.pt"
                                     for source in entry["logical_sources"]]
            audit.require(parent_names == expected_parent_names,
                          f"{identity}: parent artifact bindings differ")
            for parent_record, logical_source in zip(entry["parent_artifacts"],
                                                     entry["logical_sources"]):
                name = parent_record["name"]
                audit.require(parent_inputs["hashes"].get(name) == parent_record["sha256"],
                              f"{identity}: parent hash binding differs for {name}")
                if name not in parent_cache:
                    binding = torch.load(parent_root / name, weights_only=True, map_location="cpu")
                    audit.require(binding["state"].get("schema") == "i9_neural_snapshot_v1",
                                  f"parent {name}: snapshot schema differs")
                    audit.require((binding["seed"], binding["source"], binding["completed_step"])
                                  == (entry["seed"], logical_source, entry["parent_step"]),
                                  f"parent {name}: binding metadata differs")
                    audit.require(binding["plan_file"] == f"plan-s{entry['seed']}.json",
                                  f"parent {name}: plan binding differs")
                    state_digest = tree_digest(binding["state"])
                    audit.tree_digests += 1
                    audit.require(state_digest == binding["complete_state_digest"],
                                  f"parent {name}: complete-state digest differs")
                    parent_cache[name] = {
                        "complete_digest": state_digest,
                        "observer": int(binding["state"]["tracker"]["step_count"]),
                        "optimizer": optimizer_counter(binding["state"]),
                        "tracker_digest": tree_digest(binding["state"]["tracker"]),
                    }
                audit.require(parent_cache[name]["complete_digest"] ==
                              entry["parent_complete_digest"],
                              f"{identity}: aliased parent complete digest differs for {name}")
            physical_parent = parent_cache[parent_names[0]]
            audit.require(entry["parent_complete_digest"] == physical_parent["complete_digest"],
                          f"{identity}: parent complete digest differs")

            curve = branch["curve"]
            audit.require([row["horizon"] for row in curve] == list(HORIZONS),
                          f"{identity}: curve horizons differ")
            curve_by_horizon = {}
            for row in curve:
                validate_evaluation(row, audit, f"{identity} horizon {row['horizon']}")
                curve_by_horizon[row["horizon"]] = row
            baselines.setdefault((entry["seed"], entry["physical_source"], entry["parent_step"]),
                                 []).append((identity, curve[0]))
            diagnostic_summary = validate_steps(branch["steps"], entry["policy"],
                                                physical_parent["observer"], audit, identity)

            final_state = final_binding["state"]
            audit.require(final_state.get("schema") == "i9_neural_snapshot_v1",
                          f"{identity}: final snapshot schema differs")
            audit.require(optimizer_counter(final_state) == physical_parent["optimizer"] + 500,
                          f"{identity}: final Adam counter is not parent+500")
            final_observer = int(final_state["tracker"]["step_count"])
            expected_observer = (physical_parent["observer"] if entry["policy"] == "frozen32"
                                 else physical_parent["observer"] + 500)
            audit.require(final_observer == expected_observer,
                          f"{identity}: final observer counter differs")
            if entry["policy"] == "frozen32":
                audit.require(tree_digest(final_state["tracker"]) == physical_parent["tracker_digest"],
                              f"{identity}: frozen observer differs from parent")

            row = {**metadata(entry), "curve": curve, "curve_by_horizon": curve_by_horizon,
                   "diagnostic_summary": diagnostic_summary,
                   "branch_artifact": branch_record["name"], "final_artifact": final_record["name"]}
            pkey = (entry["seed"], entry["physical_source"], entry["parent_step"],
                    entry["objective"], entry["policy"])
            audit.require(pkey not in physical, f"duplicate physical branch key {pkey}")
            physical[pkey] = row
            for source in entry["logical_sources"]:
                lkey = (entry["seed"], source, entry["parent_step"],
                        entry["objective"], entry["policy"])
                audit.require(lkey not in logical, f"duplicate logical branch key {lkey}")
                logical[lkey] = row
            physical_summaries.append({key: row[key] for key in META_KEYS
                                       if key not in ("parent_artifacts",)} | {
                "branch_artifact": row["branch_artifact"], "curve": curve,
                "diagnostic_summary": diagnostic_summary})
        except Exception as exc:
            audit.errors.append(f"{identity}: analysis failed: {type(exc).__name__}: {exc}")

    audit.require(set(physical) == expected_physical_keys(), "physical branch key coverage differs")
    expected_logical = {(seed, source, anchor, objective, policy)
                        for seed in SEEDS for source in SOURCES for anchor in ANCHORS
                        for objective in OBJECTIVES for policy in POLICIES}
    audit.require(set(logical) == expected_logical, "logical branch key coverage differs")
    for parent_key, rows in baselines.items():
        audit.require(len(rows) == 9, f"parent {parent_key}: expected nine baselines")
        if rows:
            reference = rows[0][1]
            for identity, value in rows[1:]:
                audit.require(value == reference, f"{identity}: horizon-zero baseline differs")

    logical_references = []
    for key, row in sorted(logical.items()):
        seed, source, anchor, objective, policy = key
        logical_references.append({"seed": seed, "source": source, "anchor": anchor,
                                   "objective": objective, "policy": policy,
                                   "physical_branch_id": row["id"],
                                   "is_step100_alias": anchor == 100 and source != row["physical_source"]})

    summary = {
        "schema": "i10_branch_analysis_summary_v1",
        "scientific_scope": [
            "Positive CE/accuracy benefits favor the selected filtering policy over raw continuation.",
            "Effects are conditional branches from inherited I9 states, not end-to-end optimizer effects.",
            "Step-100 raw/current parents are identical aliases and are never independent evidence.",
            "Signed interactions and residuals are retained; no composite or selected horizon is used.",
        ],
        "artifact_directory": str(root),
        "counts": {"physical_branches": len(physical), "logical_cells": len(logical),
                   "step100_alias_cells": sum(row["is_step100_alias"] for row in logical_references)},
        "effect_definitions": {
            "ce_benefit": "auxiliary clean CE(raw) - auxiliary clean CE(policy)",
            "accuracy_benefit": "auxiliary clean accuracy(policy) - auxiliary clean accuracy(raw)",
            "interaction": "benefit(fixed) - benefit(comparator objective)",
        },
        "primary_six_effects": primary_effects(logical),
        "mandatory_secondary_effects": secondary_effects(logical),
        "physical_branch_curves_and_diagnostics": physical_summaries,
        "logical_cell_references": logical_references,
    }
    audit_payload = {
        "schema": "i10_branch_analysis_audit_v1",
        "status": "pass" if not audit.errors else "fail",
        "checks": audit.checks, "errors": audit.errors, "warnings": audit.warnings,
        "completion_hash_files_verified": audit.hash_files,
        "hash_bytes_streamed": audit.hash_bytes,
        "complete_state_tree_digests_verified": audit.tree_digests,
        "maximum_train_residual_identity_error": audit.max_residual_error,
        "maximum_leakage_energy_identity_error": audit.max_leakage_identity_error,
        "coverage": {"physical": len(physical), "logical": len(logical),
                     "baseline_parent_groups": len(baselines)},
        "energy_weighted_leakage_definition":
            "sum(step_norm^2 * saved per-step leakage fraction) / sum(step_norm^2)",
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
