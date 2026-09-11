#!/usr/bin/env python3
"""CPU-only integrity audit and predeclared scalar analysis of I13 artifacts."""
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
_A10_PATH = HERE.parent / "iteration-010" / "analyse_branches.py"
_SPEC = importlib.util.spec_from_file_location("_i13_independent_i10_analysis", _A10_PATH)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover
    raise ImportError("cannot load independent I10 analysis helpers")
a10 = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(a10)


SEEDS = (100, 101, 102)
ANCHORS = (1500, 2000)
OBJECTIVES = ("fixed", "soft", "redraw")
NEW_POLICIES = ("mean32", "leak01_32")
REFERENCE_POLICIES = ("raw", "current32")
POLICIES = REFERENCE_POLICIES + NEW_POLICIES
HORIZONS = (0, 1, 10, 50, 100, 250, 500)
DATA_ROOT = Path("data/MNIST/raw")
ALIGNMENT_NAMES = ("train_fixed", "train_soft", "train_clean", "aux_clean",
                   "fixed_minus_soft")
COMPONENT_NAMES = ("raw_gradient", "native_projection", "outside_raw_gradient",
                   "outside_post_mean", "outside_pre_mean", "mean32_delivery",
                   "leak01_32_delivery", "selected_delivery")
ALGEBRA_NAMES = ("native_minus_explicit_post_projection", "post_mean_recurrence",
                 "mean_definition", "leak_definition",
                 "mean_minus_leak_historical_term")


def expected_new_keys() -> set[tuple[Any, ...]]:
    return {(seed, anchor, objective, policy) for seed in SEEDS for anchor in ANCHORS
            for objective in OBJECTIVES for policy in NEW_POLICIES}


def expected_reference_keys() -> set[tuple[Any, ...]]:
    return {(seed, anchor, objective, policy) for seed in SEEDS for anchor in ANCHORS
            for objective in OBJECTIVES for policy in REFERENCE_POLICIES}


def expected_probe_horizons(curve_horizons: list[int], status: str,
                            numerical_failure: dict[str, Any] | None) -> list[int]:
    """Return the retained-probe prefix, preserving an eval before probe failure."""
    result = [horizon for horizon in HORIZONS if horizon <= max(curve_horizons)]
    if status == "numerical_failure" and numerical_failure is not None \
            and numerical_failure.get("phase") == "probe":
        result = result[:-1]
    return result


def fingerprint_check(values: list[dict[str, Any]]) -> dict[str, bool]:
    available = len(values) == 2 and all(row["completed"] > 0 for row in values)
    return {
        "available": available,
        "raw_equal": available and len({row["raw"] for row in values}) == 1,
        "observer_equal": available and len({row["observer"] for row in values}) == 1,
        # Equality here is scientifically meaningful: it means the historical
        # complement term was exactly zero for this first delivery.
        "applied_different": available and len({row["applied"] for row in values}) == 2,
    }


def external_hash(path: Path, expected: str, audit: Any, label: str) -> None:
    audit.require(path.is_file(), f"{label}: file is missing")
    if path.is_file():
        audit.require(a10.sha256(path) == expected, f"{label}: SHA256 differs")
        audit.hash_files += 1
        audit.hash_bytes += path.stat().st_size


def curve_map(curve: list[dict[str, Any]], audit: Any, label: str) -> dict[int, dict[str, Any]]:
    horizons = [row.get("horizon") for row in curve]
    audit.require(horizons == sorted(set(horizons)), f"{label}: curve horizons/order differ")
    audit.require(all(horizon in HORIZONS for horizon in horizons),
                  f"{label}: curve has an undeclared horizon")
    result = {}
    for row in curve:
        a10.validate_evaluation(row, audit, f"{label} h{row.get('horizon')}")
        result[row["horizon"]] = row
    return result


def finite(value: Any, label: str) -> float:
    return a10.finite(value, label)


def validate_mean_steps(steps: list[dict[str, Any]], policy: str, parent_observer: int,
                        audit: Any, label: str) -> dict[str, Any]:
    sums: dict[str, float] = {"loss": 0.0, "raw_gradient_norm": 0.0,
                              "applied_gradient_norm": 0.0,
                              "total_step_norm": 0.0, "data_step_norm": 0.0}
    component_sums = {name: 0.0 for name in COMPONENT_NAMES}
    leakage = {(kind, basis): [0.0, 0.0] for kind in ("total", "data")
               for basis in ("frozen_basis", "current_basis")}
    max_algebra_ratio = 0.0
    for index, row in enumerate(steps, start=1):
        prefix = f"{label} step {index}"
        audit.require(row.get("horizon") == index and row.get("schema")
                      == "i13_mean_branch_step_v1", f"{prefix}: schema/horizon differs")
        audit.require(row.get("policy") == policy and row.get("gradient_filter_applied") is True,
                      f"{prefix}: policy/filter flag differs")
        observer = row["observer"]
        audit.require(observer == {"used": True, "step_before": parent_observer + index - 1,
                                    "step_after": parent_observer + index,
                                    "filtering_active": True},
                      f"{prefix}: observer fields differ")
        observation = row["mean_diagnostics"]
        audit.require(observation.get("schema") == "i13_observe_delivery_v1"
                      and observation.get("policy") == policy
                      and observation.get("observer", {}).get("step_before")
                      == parent_observer + index - 1
                      and observation.get("observer", {}).get("step_after")
                      == parent_observer + index,
                      f"{prefix}: observation schema/counter differs")
        audit.require(row.get("loss") == observation.get("loss"),
                      f"{prefix}: loss differs from observation")
        audit.require(tuple(observation.get("components", {})) == COMPONENT_NAMES,
                      f"{prefix}: component topology differs")
        for name, component in observation.get("components", {}).items():
            norm = finite(component["norm"], f"{prefix} {name} norm")
            energy = finite(component["squared_energy"], f"{prefix} {name} energy")
            audit.require(norm >= 0 and energy >= 0
                          and math.isclose(energy, norm * norm, rel_tol=2e-6, abs_tol=1e-14),
                          f"{prefix}: {name} norm/energy identity differs")
            component_sums[name] += norm
        residuals, bounds = (observation.get("algebra_residual_norms", {}),
                             observation.get("algebra_error_bounds", {}))
        audit.require(tuple(residuals) == ALGEBRA_NAMES and tuple(bounds) == ALGEBRA_NAMES,
                      f"{prefix}: algebra topology differs")
        for name in ALGEBRA_NAMES:
            residual, bound = finite(residuals[name], prefix + " residual"), finite(
                bounds[name], prefix + " bound")
            audit.require(residual >= 0 and bound >= 0 and residual <= bound,
                          f"{prefix}: {name} exceeds prospective error bound")
            max_algebra_ratio = max(max_algebra_ratio,
                                    0.0 if bound == 0 else residual / bound)
        for digest_name in ("raw_gradient_digest", "post_observer_digest",
                            "applied_gradient_digest"):
            value = observation.get(digest_name)
            audit.require(type(value) is str and len(value) == 64,
                          f"{prefix}: {digest_name} malformed")
        values = {
            "loss": finite(row["loss"], prefix + " loss"),
            "raw_gradient_norm": finite(row["gradient"]["raw_norm"], prefix + " raw norm"),
            "applied_gradient_norm": finite(row["gradient"]["applied_norm"], prefix + " applied norm"),
            "total_step_norm": finite(row["displacement"]["total_norm"], prefix + " total norm"),
            "data_step_norm": finite(row["displacement"]["data_norm"], prefix + " data norm"),
        }
        audit.require(values["raw_gradient_norm"]
                      == observation["components"]["raw_gradient"]["norm"]
                      and values["applied_gradient_norm"]
                      == observation["components"]["selected_delivery"]["norm"],
                      f"{prefix}: outer gradient norms differ from observation")
        for name, value in values.items():
            audit.require(value >= 0, f"{prefix}: {name} is negative")
            sums[name] += value
        for kind, norm_name in (("total", "total_step_norm"), ("data", "data_step_norm")):
            energy = values[norm_name] ** 2
            for basis in ("frozen_basis", "current_basis"):
                saved = a10.leakage_energy(row["displacement"][kind + "_leakage"][basis],
                                           energy, audit, f"{prefix} {kind}/{basis}")
                if saved is not None:
                    leakage[(kind, basis)][0] += saved[0]
                    leakage[(kind, basis)][1] += saved[1]
    count = len(steps)
    return {
        "step_count": count,
        "arithmetic_step_means": {name: value / count if count else None
                                  for name, value in sums.items()},
        "component_norm_means": {name: value / count if count else None
                                 for name, value in component_sums.items()},
        "maximum_algebra_residual_to_bound_ratio": max_algebra_ratio,
        "energy_weighted_leakage": {
            kind + "_against_" + basis: {
                "outside_energy": values[0], "total_energy": values[1],
                "fraction": None if values[1] == 0 else values[0] / values[1],
                "definition": "sum outside squared energy / sum total squared energy within branch",
            } for (kind, basis), values in leakage.items()
        },
    }


def utility(branch: dict[str, Any], horizon: int, split: str, metric: str) -> float | None:
    row = branch["curve_by_horizon"].get(horizon)
    if row is None:
        return None
    group = row[split]
    if metric == "ce":
        return -float(group["clean_ce"])
    if metric == "accuracy":
        return float(group["clean_accuracy"])
    raise ValueError(metric)


def aggregate(cells: list[dict[str, Any]], field: str) -> dict[str, Any]:
    per_seed = []
    for seed in SEEDS:
        rows = sorted((row for row in cells if row["seed"] == seed),
                      key=lambda row: row["parent_step"])
        values = [row.get(field) for row in rows]
        available = len(rows) == len(ANCHORS) and all(value is not None for value in values)
        per_seed.append({"seed": seed,
                         "parent_values": {str(row["parent_step"]): row.get(field) for row in rows},
                         "available": available,
                         "seed_first_parent_mean": float(np.mean(values)) if available else None})
    values = [row["seed_first_parent_mean"] for row in per_seed]
    available = len(per_seed) == len(SEEDS) and all(value is not None for value in values)
    return {"per_seed": per_seed, "available": available,
            "all_three_seed_values": values,
            "mean": float(np.mean(values)) if available else None,
            "no_survivor_averaging": True}


def contrast_rows(branches: dict[tuple[Any, ...], dict[str, Any]]) -> list[dict[str, Any]]:
    pairs = (("mean32", "current32"), ("mean32", "leak01_32"),
             ("mean32", "raw"), ("leak01_32", "current32"),
             ("leak01_32", "raw"), ("current32", "raw"))
    result = []
    for objective in OBJECTIVES:
        for split in ("auxiliary", "validation"):
            for metric in ("ce", "accuracy"):
                for left, right in pairs:
                    for horizon in HORIZONS:
                        cells = []
                        for seed in SEEDS:
                            for anchor in ANCHORS:
                                lhs = utility(branches[(seed, anchor, objective, left)],
                                              horizon, split, metric)
                                rhs = utility(branches[(seed, anchor, objective, right)],
                                              horizon, split, metric)
                                cells.append({"seed": seed, "parent_step": anchor,
                                              "left_utility": lhs, "right_utility": rhs,
                                              "contrast": (None if lhs is None or rhs is None
                                                           else lhs - rhs)})
                        result.append({"objective": objective, "split": split,
                                       "metric": metric, "left": left, "right": right,
                                       "horizon": horizon, "individual_parent_cells": cells,
                                       "seed_first": aggregate(cells, "contrast")})
    return result


def primary(contrasts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"objective": objective, "metric": metric, "split": "auxiliary",
             "horizon": 500, "estimand": "U(mean32)-U(current32)",
             "positive_means": "mean32 has higher clean utility than current32",
             **next(row for row in contrasts if row["objective"] == objective
                    and row["split"] == "auxiliary" and row["metric"] == metric
                    and row["left"] == "mean32" and row["right"] == "current32"
                    and row["horizon"] == 500)["seed_first"]}
            for objective in ("soft", "redraw") for metric in ("ce", "accuracy")]


def absolute_progress(branches: dict[tuple[Any, ...], dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for objective in OBJECTIVES:
        for split in ("auxiliary", "validation"):
            for metric in ("ce", "accuracy"):
                for policy in POLICIES:
                    for horizon in HORIZONS:
                        cells = []
                        for seed in SEEDS:
                            for anchor in ANCHORS:
                                branch = branches[(seed, anchor, objective, policy)]
                                start = utility(branch, 0, split, metric)
                                end = utility(branch, horizon, split, metric)
                                cells.append({"seed": seed, "parent_step": anchor,
                                              "h0_utility": start, "utility": end,
                                              "progress": None if start is None or end is None
                                              else end - start})
                        result.append({"objective": objective, "split": split,
                                       "metric": metric, "policy": policy,
                                       "horizon": horizon, "individual_parent_cells": cells,
                                       "seed_first": aggregate(cells, "progress")})
    return result


def tensor_close(left: torch.Tensor, right: torch.Tensor, *, rtol: float = 3e-6,
                 atol: float = 1e-8) -> tuple[bool, float]:
    if type(left) is not torch.Tensor or type(right) is not torch.Tensor \
            or left.shape != right.shape:
        return False, math.inf
    difference = float((left.to(torch.float64) - right.to(torch.float64)).abs().max().item())
    return bool(torch.allclose(left, right, rtol=rtol, atol=atol)), difference


def load_probe(root: Path, descriptor: dict[str, Any], completion_index: dict[str, Any],
               audit: Any, label: str) -> tuple[dict[str, Any], dict[str, Any]] | None:
    try:
        artifact = descriptor["tensor_artifact"]
        audit.require(completion_index.get(artifact["name"]) == artifact,
                      f"{label}: artifact is not completion-bound")
        payload = torch.load(root / artifact["name"], weights_only=True, map_location="cpu")
        for key, value in descriptor.items():
            if key != "tensor_artifact":
                audit.require(payload.get(key) == value, f"{label}: payload {key} differs")
        actual = a10.tree_digest(payload["tensors"])
        audit.tree_digests += 1
        audit.require(actual == payload["tensor_digest"] == descriptor["tensor_digest"],
                      f"{label}: tensor payload digest differs")
        if "tensor_payload_sha256" in payload["record"]:
            audit.require(payload["record"]["tensor_payload_sha256"] == actual,
                          f"{label}: record tensor-payload digest differs")
        return payload["record"], payload["tensors"]
    except Exception as exc:
        audit.errors.append(f"{label}: probe load failed: {type(exc).__name__}: {exc}")
        return None


def alignment_arithmetic(record: dict[str, Any], tensors: dict[str, Any], audit: Any,
                         label: str) -> dict[str, Any]:
    audit.require(record.get("schema") == "i13_alignment_probe_v1"
                  and record.get("status") == "complete", f"{label}: record schema/status differs")
    mean, projected, complement = (tensors["incumbent_mean"], tensors["mean_projection"],
                                   tensors["complement_mean"])
    ok, error = tensor_close(mean, projected + complement, rtol=2e-6, atol=2e-7)
    audit.require(ok, f"{label}: mean decomposition differs")
    gradients = tensors["gradients"]
    audit.require(tuple(gradients) == ALIGNMENT_NAMES, f"{label}: gradient topology differs")
    ok_residual, residual_error = tensor_close(
        gradients["fixed_minus_soft"], gradients["train_fixed"] - gradients["train_soft"],
        rtol=2e-6, atol=2e-7)
    audit.require(ok_residual, f"{label}: fixed-minus-soft gradient identity differs")
    audit.require(record.get("complement_mean_norm") is not None
                  and math.isclose(float(record["complement_mean_norm"]),
                                   float(torch.linalg.vector_norm(complement.to(torch.float64)).item()),
                                   rel_tol=2e-6, abs_tol=1e-10),
                  f"{label}: complement norm differs")
    recomputed = {}
    left_norm = float(torch.linalg.vector_norm(complement.to(torch.float64)).item())
    for name, gradient in gradients.items():
        right_norm = float(torch.linalg.vector_norm(gradient.to(torch.float64)).item())
        dot = float(torch.dot(complement.to(torch.float64), gradient.to(torch.float64)).item())
        saved = record["alignments"][name]
        audit.require(math.isclose(saved["gradient_norm"], right_norm, rel_tol=2e-6, abs_tol=1e-10)
                      and math.isclose(saved["dot"], dot, rel_tol=3e-6, abs_tol=1e-10),
                      f"{label}: {name} signed norm/dot differs")
        cosine = None if left_norm == 0 or right_norm == 0 else dot / (left_norm * right_norm)
        if cosine is None:
            audit.require(saved["cosine"] is None and saved["reason"] is not None,
                          f"{label}: {name} zero-vector case differs")
        else:
            audit.require(saved["reason"] is None and math.isclose(saved["cosine"], cosine,
                          rel_tol=4e-6, abs_tol=2e-8), f"{label}: {name} cosine differs")
        recomputed[name] = {"gradient_norm": right_norm, "dot": dot, "cosine": cosine,
                            "reason": saved["reason"]}
    digests = record["tensor_digests"]
    for name in ("incumbent_mean", "mean_projection", "complement_mean"):
        audit.require(digests[name] == a10.tree_digest(tensors[name]),
                      f"{label}: {name} digest differs")
        audit.tree_digests += 1
    for name, gradient in gradients.items():
        audit.require(digests["gradients"][name] == a10.tree_digest(gradient),
                      f"{label}: gradient {name} digest differs")
        audit.tree_digests += 1
    return {"state_digest": record["state_sha256"], "observer_step": record["observer_step"],
            "basis_rank": record["basis_rank"], "alignments": recomputed,
            "mean_decomposition_max_abs_error": error,
            "fixed_minus_soft_max_abs_error": residual_error}


def flat_state_parameters(state: dict[str, Any]) -> torch.Tensor:
    values = list(state["model_state"].values())
    if len(values) != 4 or not all(type(value) is torch.Tensor for value in values):
        raise ValueError("I13 MLP model-state topology differs")
    return torch.cat([value.detach().cpu().reshape(-1) for value in values])


def paired_arithmetic(record: dict[str, Any], tensors: dict[str, Any], parent: dict[str, Any],
                      audit: Any, label: str) -> dict[str, Any]:
    audit.require(record.get("schema") == "i13_paired_step_probe_v1"
                  and record.get("status") == "complete", f"{label}: paired schema/status differs")
    raw, native, pre, post, basis = (tensors["raw_gradient"], tensors["native_projection"],
                                    tensors["pre_mean"], tensors["post_mean"],
                                    tensors["post_basis"])
    mean = tensors["delivered_gradients"]["mean32"]
    leak = tensors["delivered_gradients"]["leak01_32"]
    current = tensors["delivered_gradients"]["current32"]
    project = lambda value: basis @ (basis.T @ value)
    identities = {
        "native_projection": (native, project(raw)),
        "current_delivery": (current, native),
        "post_mean_recurrence": (post, .99 * pre + .01 * raw),
        "mean_delivery": (mean, native + post - project(post)),
        "leak_delivery": (leak, native + .01 * (raw - native)),
        "mean_minus_leak": (mean - leak, .99 * (pre - project(pre))),
    }
    errors = {}
    for name, (left, right) in identities.items():
        ok, error = tensor_close(left, right, rtol=4e-5, atol=3e-7)
        audit.require(ok, f"{label}: {name} vector identity differs")
        errors[name] = error
    digests = record["tensor_digests"]
    audit.require(digests["raw_gradient"] == a10.tree_digest(raw)
                  and digests["post_basis"] == a10.tree_digest(basis),
                  f"{label}: raw/basis tensor digests differ")
    audit.tree_digests += 2
    for name, delivered in tensors["delivered_gradients"].items():
        audit.require(digests["delivered_gradients"][name] == a10.tree_digest(delivered),
                      f"{label}: delivered gradient {name} digest differs")
        audit.tree_digests += 1
    theta = flat_state_parameters(parent)
    interventions = tensors["interventions"]
    summaries = {}
    for name, saved in record["interventions"].items():
        vectors = interventions[name]
        audit.require(digests["interventions"][name] == a10.tree_digest(vectors),
                      f"{label}: intervention {name} tensor digest differs")
        audit.tree_digests += 1
        if saved["status"] == "undefined":
            audit.require(vectors["total_delta"] is None and vectors["data_delta"] is None
                          and saved["losses"] is None and saved["loss_changes"] is None,
                          f"{label}: undefined intervention has values")
            summaries[name] = {"status": "undefined", "reason": saved["reason"]}
            continue
        total, data = vectors["total_delta"], vectors["data_delta"]
        ok, error = tensor_close(data, total + 1e-5 * theta, rtol=2e-5, atol=2e-8)
        audit.require(ok, f"{label}: {name} decay-adjusted displacement differs")
        total_norm = float(torch.linalg.vector_norm(total.to(torch.float64)).item())
        data_norm = float(torch.linalg.vector_norm(data.to(torch.float64)).item())
        audit.require(math.isclose(saved["total_norm"], total_norm, rel_tol=2e-6, abs_tol=1e-10)
                      and math.isclose(saved["data_norm"], data_norm, rel_tol=2e-6, abs_tol=1e-10),
                      f"{label}: {name} vector/scalar norms differ")
        for loss_name, value in saved["losses"].items():
            expected_change = value - record["baseline_losses"][loss_name]
            audit.require(math.isclose(saved["loss_changes"][loss_name], expected_change,
                                       rel_tol=1e-10, abs_tol=1e-12),
                          f"{label}: {name}/{loss_name} change identity differs")
        summaries[name] = {"status": "defined", "total_norm": total_norm,
                           "data_norm": data_norm, "loss_changes": saved["loss_changes"],
                           "decay_identity_max_abs_error": error}
    reciprocal = (("mean_to_current_data_norm", "current32"),
                  ("current_to_mean_data_norm", "mean32"),
                  ("leak_to_current_data_norm", "current32"),
                  ("current_to_leak_data_norm", "leak01_32"))
    for controlled, target in reciprocal:
        if summaries[controlled]["status"] == "defined":
            audit.require(math.isclose(summaries[controlled]["data_norm"],
                                       summaries[target]["data_norm"],
                                       rel_tol=5e-5, abs_tol=1e-10),
                          f"{label}: {controlled} reciprocal data norm differs")
    return {"objective": record["objective"], "vector_identity_max_abs_errors": errors,
            "interventions": summaries}


def add_descriptor(store: dict[str, dict[str, Any]], descriptor: dict[str, Any],
                   audit: Any, label: str) -> None:
    name = descriptor.get("tensor_artifact", {}).get("name")
    audit.require(type(name) is str, f"{label}: descriptor artifact name malformed")
    if type(name) is str and name in store:
        audit.require(store[name] == descriptor, f"{label}: repeated descriptor differs")
    elif type(name) is str:
        store[name] = descriptor


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
    audit.require(manifest.get("mode") == "full" and manifest.get("planned_branches") == 36
                  and manifest.get("baseline_count") == 36 and manifest.get("parent_count") == 6
                  and manifest.get("paired_step_groups") == 18
                  and manifest.get("physical_alignment_records_if_complete") == 258,
                  "manifest mode/counts differ")
    audit.require(tuple(manifest.get("policies", ())) == NEW_POLICIES
                  and tuple(manifest.get("objectives", ())) == OBJECTIVES
                  and tuple(manifest.get("horizons", ())) == HORIZONS,
                  "manifest policy/objective/horizon declarations differ")
    audit.require(completion.get("status") == "complete" and completion.get("mode") == "full"
                  and completion.get("attempted_branches") == 36,
                  "completion status/mode/attempt count differs")
    records = completion.get("artifacts", [])
    completion_index = {record["name"]: record for record in records}
    audit.require(len(completion_index) == len(records), "completion artifact names repeat")
    for record in records:
        a10.verify_record(root, record, audit, "I13 completion")
    for name in ("manifest.json", "parent-inputs.json", "baseline-references.json", "branches.json"):
        audit.require(name in completion_index, f"completion omits {name}")
    for path, expected in manifest.get("source_hashes", {}).items():
        external_hash(ROOT / path, expected, audit, f"frozen source {path}")

    inputs = a10.read_json(root / "parent-inputs.json")
    i10_root = Path(inputs["i10_directory"]).resolve(strict=True)
    i9_root = Path(inputs["i9_directory"]).resolve(strict=True)
    for external_root, hashes, label in ((i10_root, inputs["i10_hashes"], "I10"),
                                          (i9_root, inputs["i9_hashes"], "I9")):
        for name, expected in hashes.items():
            external_hash(external_root / name, expected, audit, f"{label} input {name}")
    for name, expected in inputs.get("data_hashes", {}).items():
        external_hash(DATA_ROOT / name, expected, audit, f"bound MNIST input {name}")
    audit.require(inputs.get("baseline_count") == 36 and inputs.get("parent_count") == 6,
                  "parent-input counts differ")
    audit.require(inputs.get("baseline_source_hashes") == {
        key: manifest["source_hashes"][key] for key in inputs.get("baseline_source_hashes", {})},
        "I10 source hashes are not bound into I13 manifest")

    parents: dict[tuple[int, int], dict[str, Any]] = {}
    states_by_digest: dict[str, dict[str, Any]] = {}
    for seed in SEEDS:
        for anchor in ANCHORS:
            name = f"anchor-s{seed}-current32-t{anchor}.pt"
            binding = torch.load(i9_root / name, weights_only=True, map_location="cpu")
            state, state_digest = binding["state"], a10.tree_digest(binding["state"])
            audit.tree_digests += 1
            audit.require((binding["seed"], binding["source"], binding["completed_step"])
                          == (seed, "current32", anchor)
                          and state_digest == binding["complete_state_digest"]
                          and a10.optimizer_counter(state) == anchor,
                          f"I9 parent {name}: metadata/state/counter differs")
            parents[(seed, anchor)] = {"state": state, "digest": state_digest,
                                       "observer": int(state["tracker"]["step_count"]),
                                       "artifact": name}
            states_by_digest[state_digest] = state
            copied_plan = root / f"plan-s{seed}-t{anchor}.json"
            audit.require(a10.sha256(copied_plan) == inputs["i10_hashes"][copied_plan.name],
                          f"copied plan {copied_plan.name}: hash differs")
            probe_plan = a10.read_json(root / f"probe-plan-s{seed}-t{anchor}.json")
            audit.require((probe_plan["seed"], probe_plan["parent_step"])
                          == (seed, anchor)
                          and probe_plan.get("source_plan") == f"plan-s{seed}.json"
                          and probe_plan.get("source_sha256")
                          == inputs["i9_hashes"][f"plan-s{seed}.json"],
                          f"probe plan s{seed}/t{anchor}: source identity differs")
            source_plan = a10.read_json(i9_root / f"plan-s{seed}.json")
            source_anchor = source_plan["anchors"][str(anchor)]
            audit.require(tuple(probe_plan["indices"])
                          == ("loss_train", "loss_aux", "utility_aux"),
                          f"probe plan s{seed}/t{anchor}: key order differs")
            for key, value in probe_plan["indices"].items():
                array = np.asarray(value)
                audit.require(array.shape == (1024,) and array.dtype.kind in "iu"
                              and bool(((array >= 0) & (array < 5000)).all())
                              and np.array_equal(array, np.asarray(source_anchor[key])),
                              f"probe plan s{seed}/t{anchor}: {key} indices/provenance differ")

    references_doc = a10.read_json(root / "baseline-references.json")
    audit.require(references_doc.get("count") == 36
                  and references_doc.get("new_baseline_training_updates") == 0
                  and Path(references_doc.get("directory", "")).resolve() == i10_root,
                  "baseline-reference declaration differs")
    references: dict[tuple[Any, ...], dict[str, Any]] = {}
    reference_summaries = []
    for entry in references_doc.get("entries", []):
        identity = entry.get("id", "unknown-reference")
        try:
            key = (entry["seed"], entry["parent_step"], entry["objective"], entry["policy"])
            audit.require(key in expected_reference_keys() and key not in references,
                          f"{identity}: reference key differs/repeats")
            branch_record, final_record = entry["branch_artifact"], entry["final_artifact"]
            audit.require(inputs["i10_hashes"].get(branch_record["name"]) == branch_record["sha256"]
                          and inputs["i10_hashes"].get(final_record["name"]) == final_record["sha256"],
                          f"{identity}: reference hashes are unbound")
            branch = a10.read_json(i10_root / branch_record["name"])
            terminal = torch.load(i10_root / final_record["name"], weights_only=True,
                                  map_location="cpu")
            digest = a10.tree_digest(terminal["state"])
            audit.tree_digests += 1
            audit.require(digest == terminal["complete_state_digest"]
                          == branch["final_complete_digest"],
                          f"{identity}: reference final state digest differs")
            parent = parents[(entry["seed"], entry["parent_step"])]
            for field in ("id", "seed", "parent_step", "physical_source", "logical_sources",
                          "objective", "policy", "parent_artifacts", "parent_complete_digest",
                          "plan_file", "completed_horizon", "final_artifact"):
                audit.require(branch.get(field) == entry.get(field),
                              f"{identity}: reference metadata differs for {field}")
            audit.require(entry["parent_complete_digest"] == parent["digest"]
                          and entry["completed_horizon"] == 500,
                          f"{identity}: reference parent/horizon differs")
            curve = curve_map(branch["curve"], audit, identity)
            audit.require(tuple(curve) == HORIZONS and len(branch["steps"]) == 500,
                          f"{identity}: reference curve/steps incomplete")
            diagnostic = a10.validate_steps(branch["steps"], entry["policy"], parent["observer"],
                                            audit, identity)
            references[key] = {"status": "complete", "curve": branch["curve"],
                               "curve_by_horizon": curve, "diagnostic_summary": diagnostic,
                               "branch_artifact": branch_record["name"],
                               "branch_record": branch_record,
                               "terminal_artifact": final_record["name"],
                               "terminal_digest": digest}
            states_by_digest[digest] = terminal["state"]
            reference_summaries.append({"id": identity, "seed": key[0], "parent_step": key[1],
                "objective": key[2], "policy": key[3], "curve": branch["curve"],
                "diagnostic_summary": diagnostic})
        except Exception as exc:
            audit.errors.append(f"{identity}: reference analysis failed: {type(exc).__name__}: {exc}")
    audit.require(set(references) == expected_reference_keys(), "reference coverage differs")

    index = a10.read_json(root / "branches.json")
    entries = index.get("entries", [])
    failures_declared = int(index.get("numerical_failures", -1))
    audit.require(len(entries) == 36 and index.get("attempted_branches") == 36
                  and index.get("baseline_count") == 36
                  and index.get("parent_hashes_unchanged") is True,
                  "branch-index coverage/binding differs")
    audit.require(completion.get("numerical_failures") == failures_declared
                  and completion.get("science_complete") is (failures_declared == 0)
                  and index.get("science_complete") is (failures_declared == 0),
                  "failure/science-complete declarations differ")

    branches: dict[tuple[Any, ...], dict[str, Any]] = dict(references)
    new_summaries, failure_rows = [], []
    first_groups: dict[tuple[int, int, str], list[dict[str, Any]]] = {}
    alignment_descriptors: dict[str, dict[str, Any]] = {}
    paired_descriptors: dict[str, dict[str, Any]] = {}
    common_map = {}
    common_records = index.get("common_h0_alignments", [])
    audit.require(len(common_records) == 6, "common h0 alignment count differs")
    for descriptor in common_records:
        key = (descriptor.get("seed"), descriptor.get("parent_step"))
        audit.require(key in parents and key not in common_map and descriptor.get("horizon") == 0
                      and descriptor.get("state_digest") == parents[key]["digest"],
                      f"common alignment {key}: membership/state differs")
        common_map[key] = descriptor
        add_descriptor(alignment_descriptors, descriptor, audit, "common alignment")
    audit.require(set(common_map) == set(parents), "common h0 alignment membership differs")
    baseline_map = {}
    baseline_records = index.get("baseline_h500_alignments", [])
    audit.require(len(baseline_records) == 36, "baseline h500 alignment count differs")
    for descriptor in baseline_records:
        key = (descriptor.get("seed"), descriptor.get("parent_step"),
               descriptor.get("objective"), descriptor.get("policy"))
        reference = references.get(key)
        audit.require(key in expected_reference_keys() and key not in baseline_map
                      and descriptor.get("horizon") == 500 and reference is not None
                      and descriptor.get("state_digest") == reference["terminal_digest"],
                      f"baseline alignment {key}: membership/state differs")
        baseline_map[key] = descriptor
        add_descriptor(alignment_descriptors, descriptor, audit, "baseline alignment")
    audit.require(set(baseline_map) == expected_reference_keys(),
                  "baseline h500 alignment membership differs")
    for descriptor in index.get("paired_step_probes", []):
        add_descriptor(paired_descriptors, descriptor, audit, "paired probe")

    for entry in entries:
        identity = entry.get("id", "unknown-branch")
        try:
            key = (entry["seed"], entry["parent_step"], entry["objective"], entry["policy"])
            audit.require(key in expected_new_keys() and key not in branches,
                          f"{identity}: new branch key differs/repeats")
            audit.require(identity == f"s{key[0]}-current32-t{key[1]}-{key[2]}-{key[3]}"
                          and entry["source"] == "current32", f"{identity}: ID/source differs")
            branch_record, terminal_record = entry["branch_artifact"], entry["terminal_artifact"]
            audit.require(completion_index.get(branch_record["name"]) == branch_record
                          and completion_index.get(terminal_record["name"]) == terminal_record,
                          f"{identity}: branch/terminal completion record differs")
            branch = a10.read_json(root / branch_record["name"])
            terminal = torch.load(root / terminal_record["name"], weights_only=True,
                                  map_location="cpu")
            for field in ("id", "seed", "parent_step", "source", "objective", "policy",
                          "parent_artifact", "parent_complete_digest", "plan_file",
                          "initial_evaluation_exact", "baseline_branches", "paired_step_probe",
                          "alignment_probes", "first_step_raw_gradient_digest",
                          "first_step_observer_digest", "status", "completed_steps",
                          "attempted_step", "requested_horizon"):
                audit.require(entry.get(field) == branch.get(field) == terminal.get(field),
                              f"{identity}: metadata differs for {field}")
            parent = parents[(entry["seed"], entry["parent_step"])]
            audit.require(entry["parent_artifact"]["name"] == parent["artifact"]
                          and entry["parent_artifact"]["sha256"]
                          == inputs["i9_hashes"][parent["artifact"]]
                          and entry["parent_complete_digest"] == parent["digest"]
                          and entry["plan_file"] == f"plan-s{entry['seed']}-t{entry['parent_step']}.json"
                          and entry["initial_evaluation_exact"] is True,
                          f"{identity}: parent/plan/seam binding differs")
            for policy in REFERENCE_POLICIES:
                expected = references[(entry["seed"], entry["parent_step"],
                                       entry["objective"], policy)]["branch_record"]
                audit.require(entry["baseline_branches"][policy] == expected,
                              f"{identity}: {policy} baseline binding differs")
            status, completed, attempted = (branch["status"], int(branch["completed_steps"]),
                                            int(branch["attempted_step"]))
            audit.require(status in ("complete", "numerical_failure")
                          and completed == len(branch["steps"])
                          and 0 <= completed <= attempted <= 500
                          and attempted - completed in (0, 1),
                          f"{identity}: status/step counters differ")
            curve = curve_map(branch["curve"], audit, identity)
            seam = references[(entry["seed"], entry["parent_step"],
                               entry["objective"], "current32")]["curve_by_horizon"][0]
            audit.require(0 in curve and curve[0] == seam, f"{identity}: h0 seam differs")
            probe_horizons = [item["horizon"] for item in entry["alignment_probes"]]
            expected_probes = expected_probe_horizons(list(curve), status,
                                                       branch["numerical_failure"])
            audit.require(probe_horizons == expected_probes,
                          f"{identity}: alignment-probe prefix differs")
            for wrapper in entry["alignment_probes"]:
                audit.require(wrapper["record"]["horizon"] == wrapper["horizon"],
                              f"{identity}: wrapped alignment horizon differs")
                descriptor = wrapper["record"]
                if wrapper["horizon"] == 0:
                    audit.require(descriptor == common_map[(entry["seed"], entry["parent_step"])],
                                  f"{identity}: h0 wrapper is not the exact common descriptor")
                else:
                    audit.require(descriptor.get("id") == identity
                                  and descriptor.get("horizon") == wrapper["horizon"]
                                  and type(descriptor.get("state_digest")) is str
                                  and len(descriptor["state_digest"]) == 64,
                                  f"{identity}: nonzero alignment metadata differs")
                add_descriptor(alignment_descriptors, wrapper["record"], audit,
                               f"{identity} alignment")
            add_descriptor(paired_descriptors, entry["paired_step_probe"], audit,
                           f"{identity} paired probe")
            if status == "complete":
                audit.require(completed == attempted == 500 and tuple(curve) == HORIZONS
                              and branch["numerical_failure"] is None,
                              f"{identity}: complete endpoint differs")
            else:
                failure_rows.append({"id": identity, "completed_steps": completed,
                                     "attempted_step": attempted,
                                     "last_evaluated_horizon": branch["last_evaluated_horizon"],
                                     "numerical_failure": branch["numerical_failure"]})
            diagnostic = validate_mean_steps(branch["steps"], entry["policy"],
                                             parent["observer"], audit, identity)
            terminal_state = terminal["state"]
            terminal_digest = None
            try:
                terminal_digest = a10.tree_digest(terminal_state)
                audit.tree_digests += 1
                audit.require(terminal_digest == terminal["complete_state_digest"]
                              == branch["terminal_complete_digest"]
                              and terminal.get("complete_state_digest_unavailable_reason") is None
                              and branch.get("terminal_digest_unavailable_reason") is None,
                              f"{identity}: terminal state digest differs")
            except ValueError as exc:
                audit.require(status == "numerical_failure"
                              and terminal.get("complete_state_digest") is None
                              and branch.get("terminal_complete_digest") is None
                              and terminal.get("complete_state_digest_unavailable_reason")
                              == branch.get("terminal_digest_unavailable_reason")
                              and "non-finite float" in str(exc),
                              f"{identity}: unavailable terminal digest is unjustified")
            terminal_counter = a10.optimizer_counter(terminal_state)
            terminal_observer = int(terminal_state["tracker"]["step_count"])
            if status == "complete":
                audit.require(terminal_counter == entry["parent_step"] + 500
                              and terminal_observer == parent["observer"] + 500,
                              f"{identity}: complete terminal counters differ")
            else:
                audit.require(entry["parent_step"] + completed <= terminal_counter
                              <= entry["parent_step"] + attempted
                              and parent["observer"] + completed <= terminal_observer
                              <= parent["observer"] + attempted,
                              f"{identity}: failed terminal counters outside attempted range")
            if status == "complete":
                states_by_digest[terminal_digest] = terminal_state
                terminal_probe = next(item for item in entry["alignment_probes"]
                                      if item["horizon"] == 500)
                audit.require(terminal_probe["record"]["state_digest"] == terminal_digest,
                              f"{identity}: h500 alignment is not terminal-state bound")
            else:
                last_record = branch["last_evaluated_artifact"]
                audit.require(completion_index.get(last_record["name"]) == last_record,
                              f"{identity}: last-evaluated artifact unbound")
                last = torch.load(root / last_record["name"], weights_only=True, map_location="cpu")
                last_digest = a10.tree_digest(last["state"])
                audit.tree_digests += 1
                audit.require(last_digest == last["complete_state_digest"]
                              and last["last_evaluated_horizon"] == max(curve)
                              and a10.optimizer_counter(last["state"])
                              == entry["parent_step"] + max(curve)
                              and int(last["state"]["tracker"]["step_count"])
                              == parent["observer"] + max(curve),
                              f"{identity}: last-evaluated state/horizon differs")
                states_by_digest[last_digest] = last["state"]
            first = {"policy": entry["policy"], "completed": completed,
                     "raw": branch["first_step_raw_gradient_digest"],
                     "observer": branch["first_step_observer_digest"],
                     "applied": branch["first_step_applied_gradient_digest"]}
            if completed:
                observation = branch["steps"][0]["mean_diagnostics"]
                audit.require((first["raw"], first["observer"], first["applied"])
                              == (observation["raw_gradient_digest"],
                                  observation["post_observer_digest"],
                                  observation["applied_gradient_digest"]),
                              f"{identity}: first-step fingerprints differ")
            first_groups.setdefault(key[:3], []).append(first)
            branches[key] = {"status": status, "curve": branch["curve"],
                             "curve_by_horizon": curve, "diagnostic_summary": diagnostic,
                             "branch_artifact": branch_record["name"],
                             "terminal_artifact": terminal_record["name"]}
            new_summaries.append({"id": identity, "seed": key[0], "parent_step": key[1],
                "objective": key[2], "policy": key[3], "status": status,
                "completed_steps": completed, "curve": branch["curve"],
                "diagnostic_summary": diagnostic})
        except Exception as exc:
            audit.errors.append(f"{identity}: branch analysis failed: {type(exc).__name__}: {exc}")
    audit.require(set(branches) == {(seed, anchor, objective, policy) for seed in SEEDS
                  for anchor in ANCHORS for objective in OBJECTIVES for policy in POLICIES},
                  "combined branch coverage differs")
    audit.require(len(failure_rows) == failures_declared, "numerical-failure count differs")

    first_checks = []
    for key in sorted((seed, anchor, objective) for seed in SEEDS for anchor in ANCHORS
                      for objective in OBJECTIVES):
        values = first_groups.get(key, [])
        audit.require(len(values) == 2 and {row["policy"] for row in values} == set(NEW_POLICIES),
                      f"first-step group {key}: policy coverage differs")
        check = fingerprint_check(values)
        if check["available"]:
            audit.require(check["raw_equal"] and check["observer_equal"],
                          f"first-step group {key}: raw/observer fingerprints differ")
        first_checks.append({"seed": key[0], "parent_step": key[1], "objective": key[2],
                             **check})

    audit.require(len(paired_descriptors) == 18, "physical paired-probe count differs")
    paired_summaries = []
    paired_by_group = {}
    for name, descriptor in sorted(paired_descriptors.items()):
        loaded = load_probe(root, descriptor, completion_index, audit, f"paired {name}")
        if loaded is None:
            continue
        record, tensors = loaded
        key = (descriptor["seed"], descriptor["parent_step"], descriptor["objective"])
        audit.require(key not in paired_by_group and record["objective"] == key[2]
                      and (record["plan"]["seed"], record["plan"]["parent_step"]) == key[:2]
                      and descriptor.get("state_digest") == parents[key[:2]]["digest"]
                      and record["state_sha256"] == parents[key[:2]]["digest"],
                      f"paired {name}: identity/state differs")
        summary = paired_arithmetic(record, tensors, parents[key[:2]]["state"], audit,
                                    f"paired {name}")
        plan = a10.read_json(root / f"plan-s{key[0]}-t{key[1]}.json")
        expected_indices = torch.as_tensor(plan["batches"][0], dtype=torch.long)
        audit.require(torch.equal(tensors["first_batch_indices"], expected_indices)
                      and record["plan"]["first_batch_indices_sha256"]
                      == a10.tree_digest(expected_indices)
                      and record["plan"]["first_target_sha256"]
                      == a10.tree_digest(tensors["first_target"])
                      and (record["plan"]["seed"], record["plan"]["parent_step"])
                      == key[:2], f"paired {name}: first plan/target binding differs")
        audit.tree_digests += 2
        paired_by_group[key] = (record, summary)
        paired_summaries.append({"seed": key[0], "parent_step": key[1],
                                 "objective": key[2], **summary})
    audit.require(set(paired_by_group) == set(first_groups), "paired-probe group coverage differs")
    for key, values in first_groups.items():
        if key not in paired_by_group:
            continue
        paired_record = paired_by_group[key][0]
        for value in values:
            if value["completed"]:
                observation = paired_record["delivery_observations"][value["policy"]]
                audit.require(value["raw"] == observation["raw_gradient_digest"]
                              and value["observer"] == observation["post_observer_digest"]
                              and value["applied"] == observation["applied_gradient_digest"],
                              f"paired/branch first fingerprints differ for {key}/{value['policy']}")

    audit.require(len(alignment_descriptors) == index.get("physical_alignment_records"),
                  "deduplicated physical alignment count differs")
    if failures_declared == 0:
        audit.require(len(alignment_descriptors) == 258,
                      "healthy physical alignment count differs from 258")
    alignment_summaries = []
    basis_replay_count = 0
    for name, descriptor in sorted(alignment_descriptors.items()):
        loaded = load_probe(root, descriptor, completion_index, audit, f"alignment {name}")
        if loaded is None:
            continue
        record, tensors = loaded
        summary = alignment_arithmetic(record, tensors, audit, f"alignment {name}")
        audit.require(descriptor.get("state_digest") == record["state_sha256"],
                      f"alignment {name}: descriptor/state digest differs")
        full_state = states_by_digest.get(record["state_sha256"])
        replayable = full_state is not None
        if replayable:
            basis = full_state["tracker"]["V"]
            proj_k = full_state["tracker"].get("proj_k")
            if proj_k is not None:
                basis = basis[:, :min(int(proj_k), basis.shape[1])]
            audit.require(record["tensor_digests"]["incumbent_basis"] == a10.tree_digest(basis),
                          f"alignment {name}: endpoint basis digest differs")
            audit.tree_digests += 1
            audit.require(torch.equal(tensors["incumbent_mean"],
                                      full_state["tracker"]["grad_mean"]),
                          f"alignment {name}: endpoint mean differs")
            ok, _ = tensor_close(tensors["mean_projection"],
                                 basis @ (basis.T @ tensors["incumbent_mean"]),
                                 rtol=4e-5, atol=3e-7)
            audit.require(ok, f"alignment {name}: endpoint projection replay differs")
            basis_replay_count += 1
        alignment_summaries.append({"artifact": name,
                                    "metadata": {key: descriptor[key] for key in descriptor
                                                 if key not in ("record", "tensor_digest",
                                                                "tensor_artifact")},
                                    "basis_replayable_from_retained_full_state": replayable,
                                    **summary})

    contrasts = contrast_rows(branches)
    primaries = primary(contrasts)
    expected_primary_missing = any(branches[(seed, anchor, objective, "mean32")]
                                   ["curve_by_horizon"].get(500) is None
                                   for seed in SEEDS for anchor in ANCHORS
                                   for objective in ("soft", "redraw"))
    audit.require(any(not row["available"] for row in primaries) is expected_primary_missing,
                  "primary availability does not match missing required mean32 endpoint")

    summary = {
        "schema": "i13_mean_analysis_summary_v1",
        "scientific_scope": [
            "Three reused seed bundles and two parent states per seed; conditional saved-state evidence.",
            "Positive clean-CE utility means lower clean CE; all CE and accuracy effects remain separate.",
            "Missing required endpoints make the whole affected seed-first estimand unavailable; no survivor mean.",
            "Alignment is signed local gradient geometry, not proof of a useful Adam update or long-run mediation.",
            "Intermediate bases are digest-bound only; only retained full-state endpoints permit projection replay.",
        ],
        "counts": {"new_branches": len(entries), "inherited_references": len(references),
                   "original_parents": len(parents), "paired_step_probes": len(paired_descriptors),
                   "physical_alignment_records": len(alignment_descriptors),
                   "alignment_records_with_full_state_basis_replay": basis_replay_count,
                   "numerical_failures": len(failure_rows),
                   "science_complete": failures_declared == 0},
        "utility_definitions": {"ce": "negative clean CE (higher is better)",
                                "accuracy": "clean accuracy (higher is better)",
                                "contrast": "left policy utility minus right policy utility"},
        "primary_four_h500_mean32_minus_current32": primaries,
        "all_horizon_policy_contrasts": contrasts,
        "absolute_clean_progress_all_policies": absolute_progress(branches),
        "first_step_mean_vs_leak_fingerprint_checks": first_checks,
        "paired_actual_and_norm_control_probes": paired_summaries,
        "signed_alignment_probes": alignment_summaries,
        "new_branch_curves_and_diagnostics": new_summaries,
        "inherited_reference_curves_and_diagnostics": reference_summaries,
        "numerical_failures": failure_rows,
        "original_references": {"parent_inputs": inputs,
                                "baseline_reference_artifact": "baseline-references.json",
                                "branches_artifact": "branches.json"},
    }
    audit_payload = {
        "schema": "i13_mean_analysis_audit_v1",
        "status": "pass" if not audit.errors else "fail",
        "checks": audit.checks, "errors": audit.errors, "warnings": audit.warnings,
        "completion_and_external_hash_files_verified": audit.hash_files,
        "hash_bytes_streamed": audit.hash_bytes,
        "complete_state_and_tensor_tree_digests_verified": audit.tree_digests,
        "maximum_train_residual_identity_error": audit.max_residual_error,
        "maximum_leakage_energy_identity_error": audit.max_leakage_identity_error,
        "coverage": summary["counts"],
        "energy_weighted_leakage_definition":
            "within branch sum outside squared step energy / sum total squared step energy; cohort summaries retain equal parent then seed weighting",
        "audit_limits": [
            "Saved finite evaluations and loss changes were identity/schema checked; no model/data forward was replayed.",
            "Per-step displacement leakage was audited from retained scalar energies; step vectors were not saved.",
            "Intermediate alignment bases are digest-only, so their projections were not replayed; signed retained-vector arithmetic was recomputed.",
            "Common h0 and retained final-state bases were digest checked and their saved mean projection replayed on CPU within declared tolerance.",
            "Actual Adam proposals were checked through retained displacement vectors and scalar identities, not recomputed by an optimizer step.",
        ],
        "analysis_source_sha256": a10.sha256(Path(__file__)),
    }
    for name, payload in (("summary.json", summary), ("audit.json", audit_payload)):
        with (args.output / name).open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, allow_nan=False)
            handle.write("\n")
    print(json.dumps({"status": audit_payload["status"], "new": len(entries),
                      "references": len(references), "alignments": len(alignment_descriptors),
                      "failures": len(failure_rows), "output": str(args.output)}, allow_nan=False))
    return 0 if not audit.errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
