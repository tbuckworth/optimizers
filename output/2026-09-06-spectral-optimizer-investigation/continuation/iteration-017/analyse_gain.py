#!/usr/bin/env python3
"""Independent CPU integrity audit and scalar reductions for prospective I17.

No acquisition module is imported, no model is constructed, and no forward or
optimizer is replayed. Only hash-bound full states are loaded, on CPU with
weights_only=True, for a deterministic tree digest. Import itself reads no
experimental artifact. Reused I16 functions are scalar audit/reduction helpers,
not the already-consumed analysis entrypoint.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import stat
import subprocess
from typing import Any

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
I16 = HERE.parent / "iteration-016"
_SPEC = importlib.util.spec_from_file_location("_i17_frozen_i16_analysis", I16 / "analyse_scalar.py")
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover
    raise ImportError("cannot load frozen I16 scalar analysis helpers")
old = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(old)
base = old.base
Audit, sha256, tree_digest = old.Audit, old.sha256, old.tree_digest
verify_record, verify_file = old.verify_record, old.verify_file
validate_evaluation, validate_component = old.validate_evaluation, old.validate_component
validate_leakage, _number = old.validate_leakage, old._number
runtime_inventory, validate_checkpoints = old.runtime_inventory, old.validate_checkpoints
SEEDS, TARGETS, HORIZONS = old.SEEDS, old.TARGETS, old.HORIZONS
SELECTORS, METRICS, METRIC_PATHS = old.SELECTORS, old.METRICS, old.METRIC_PATHS
PHASES = ("smoke", "confirmation")
K_BY_LABEL = {"scalar_k0": 0.0, "scalar_k0p5": .5, "scalar_k0p9": .9, "scalar_k1": 1.0}
SCALAR_LABELS = tuple(K_BY_LABEL)
SPECTRAL = "spectral_mean_projected"
NEW_POLICIES = SCALAR_LABELS[1:] + (SPECTRAL,)
POLICIES = SCALAR_LABELS + (SPECTRAL,)
ARTIFACT_CAP = 3 * 1024**3
SHA_RE = re.compile(r"[0-9a-f]{64}")
I16_ROOT = Path("/tmp/spectral-experiment-artifacts/spectral-i16-001.NdCmZ1")
I16_COMMIT = "f8c99aa82d8de106148cde90e6814069d692d953"
I16_COMPLETION_SHA = "eed6d1497f6a3672a6a59bd8962b5c9e4d33e8b97521ba2f499df0f7a7ec4cb4"
PINS = {
    "analysis-001/audit.json": "ace6596d8dd79ca419d0193ca9239864d7ad62f26bcf2be2fe85627e88f1b473",
    "analysis-001/summary.json": "11da55533e66c371132f939ce7e4a3a4495efc0fc5f5b202164d496bb438e401",
    "analysis-001/report-audit.json": "87be955c4f6f01514486969d51414d58b6ac583c0fb68f21cfb74de53ef6e96f",
    "raw-results-001/collection.json": "cc112411fd23c43f0c87d876491fc8a91ecd8e018fd58488668a037420a0687a",
}
ACQUISITION_SOURCES = old.ACQUISITION_SOURCES + tuple(str((HERE / name).relative_to(REPO))
    for name in ("gain_core.py", "test_gain_core.py", "run_gain_controls.py",
                 "test_gain_runner.py", "protocol.md", "predictions.md"))
# The original scientific closure remains immutable. This is the one explicit,
# pre-confirmation resource-only lineage in resource-amendment.md, not a general
# permission to accept arbitrary source drift or resampled smoke timings.
CONFIRMATION_SOURCES = ACQUISITION_SOURCES + tuple(str((HERE / name).relative_to(REPO))
    for name in ("run_gain_confirmation_v2.py", "test_confirmation_resource.py", "resource-amendment.md"))
SMOKE_COMMIT = "14678e2a7a6fdc3aacc25a320516f1c07a037002"
SMOKE_COMPLETION_SHA = "4985e7e1b5280f7e01f8ef4412518639e9aea7618ad9b068129a76dc1e55849d"
ARTIFACT_ROOT = Path("/tmp/spectral-experiment-artifacts/spectral-i17-001.k4VKcx")
ORIGINAL_CONFIRMATION_LIMIT = 1800
AMENDED_CONFIRMATION_LIMIT = 2400
COMPONENTS = ("raw_gradient", "native_projection", "post_mean", "native_action_post_mean",
    "post_mean_action_complement", "recurrence_input", "old_momentum_buffer",
    "selected_old_momentum_buffer", "unnormalized_new_momentum_buffer",
    "native_action_new_momentum_buffer", "delivered_buffer")
RESIDUALS = ("native_minus_actual_action_raw", "post_mean_recurrence", "applied_input_definition",
    "selected_old_buffer_definition", "unnormalized_buffer_recurrence", "delivered_buffer_definition")
DIGESTS = ("raw_gradient", "post_observer", "old_momentum_buffer",
           "unnormalized_new_momentum_buffer", "delivered_buffer")


def read_json(path: Path) -> Any:
    """Reject nonfinite constants and duplicate keys before schema validation."""
    def pairs(items):
        value = {}
        for key, item in items:
            if key in value:
                raise ValueError(f"duplicate JSON key {key!r} in {path.name}")
            value[key] = item
        return value
    with path.open(encoding="utf-8") as handle:
        return json.load(handle, object_pairs_hook=pairs, parse_constant=lambda value:
            (_ for _ in ()).throw(ValueError(f"nonfinite JSON constant {value} in {path.name}")))


def _shape(value, keys, audit, label):
    valid = type(value) is dict and tuple(value) == tuple(keys)
    audit.require(valid, label + " topology differs")
    return type(value) is dict


def _close(left, right, audit, label, *, rtol=2e-6, atol=1e-14):
    a, b = _number(left, audit, label), _number(right, audit, label)
    if a is not None and b is not None:
        audit.require(math.isclose(a, b, rel_tol=rtol, abs_tol=atol), label + " identity differs")


def validate_step(row: Any, expected_step: int, policy: str, audit: Audit, label: str) -> None:
    keys = ("step", "relative_step", "path_statistics", "schema", "base", "lr", "policy",
        "family", "k", "loss", "observer", "gradient_filter_applied",
        "native_projection_used_for_recurrence", "normalized_delivery", "gradient", "components",
        "displacement", "action_diagnostics", "decay", "algebra_residuals", "digests")
    if not _shape(row, keys, audit, label + " step"):
        return
    scalar, k = policy in K_BY_LABEL, K_BY_LABEL.get(policy)
    audit.require(policy in NEW_POLICIES and row.get("policy") == policy
        and row.get("step") == expected_step and row.get("relative_step") == expected_step - 100
        and row.get("schema") == "i17_gain_step_v1" and row.get("base") == "sgdm"
        and row.get("lr") == .03 and row.get("family") == ("scalar" if scalar else "spectral")
        and row.get("k") == k and (row.get("k") is None or type(row["k"]) is float),
        label + " step identity differs")
    observer = row.get("observer")
    if _shape(observer, ("step_before", "step_after", "filtering_active", "used", "basis_rank"),
              audit, label + " observer"):
        audit.require(observer.get("step_before") == expected_step - 1
            and observer.get("step_after") == expected_step and observer.get("filtering_active") is True
            and observer.get("used") is True and type(observer.get("basis_rank")) is int
            and 1 <= observer["basis_rank"] <= 32, label + " observer identity differs")
    audit.require(row.get("gradient_filter_applied") is (not scalar)
        and row.get("native_projection_used_for_recurrence") is (not scalar),
        label + " native recurrence declaration differs")
    _number(row.get("loss"), audit, label + " loss", nonnegative=True)
    components = row.get("components")
    if _shape(components, COMPONENTS, audit, label + " components"):
        for name in COMPONENTS:
            validate_component(components.get(name), audit, label + " " + name)
    gradient = row.get("gradient")
    mapping = {"raw_norm": "raw_gradient", "native_projection_norm": "native_projection",
               "post_mean_norm": "post_mean", "recurrence_input_norm": "recurrence_input"}
    if _shape(gradient, mapping, audit, label + " gradient") and type(components) is dict:
        for name, component in mapping.items():
            audit.require(gradient.get(name) == components.get(component, {}).get("norm"),
                          label + " gradient/component norm differs")
    normalized = row.get("normalized_delivery")
    if _shape(normalized, ("operator", "scalar_gain_correction", "manual_parameter_update",
            "unnormalized_buffer_retained", "delivery_literal", "history_literal",
            "unnormalized_buffer_norm", "delivered_buffer_norm"), audit, label + " normalization"):
        correction = 1.0 - .9 * k if scalar else None
        audit.require(normalized.get("operator") == ("(1-rho*k)*b_t" if scalar else "(I-rho*A_t)*b_t")
            and normalized.get("scalar_gain_correction") == correction
            and normalized.get("manual_parameter_update") is True
            and normalized.get("unnormalized_buffer_retained") is True
            and normalized.get("delivery_literal") == ("raw_gradient" if k == 1 else None)
            and normalized.get("history_literal") == ("old_buffer" if k == 1 else
                None if scalar else "native_action_old_buffer"), label + " normalization declaration differs")
        for field, component in (("unnormalized_buffer_norm", "unnormalized_new_momentum_buffer"),
                                 ("delivered_buffer_norm", "delivered_buffer")):
            _number(normalized.get(field), audit, label + " " + field, nonnegative=True)
            if type(components) is dict:
                audit.require(normalized.get(field) == components.get(component, {}).get("norm"),
                              label + " normalized/component norm differs")
        if scalar and type(normalized.get("unnormalized_buffer_norm")) in (int, float):
            _close(normalized.get("delivered_buffer_norm"),
                   correction * normalized["unnormalized_buffer_norm"], audit,
                   label + " scalar delivered norm")
    if scalar and type(components) is dict:
        old_norm = components.get("old_momentum_buffer", {}).get("norm")
        if type(old_norm) in (int, float):
            _close(components.get("selected_old_momentum_buffer", {}).get("norm"),
                   k * old_norm, audit, label + " scalar selected-old norm")
    displacement = row.get("displacement")
    displacement_keys = ("total_norm", "data_norm", "ideal_data_norm", "nominal_decay_norm",
        "actual_minus_ideal_data_norm", "actual_minus_ideal_relative", "total_current_basis_leakage",
        "data_current_basis_leakage", "ideal_data_current_basis_leakage", "raw_gradient_dot_data_delta",
        "post_mean_dot_data_delta", "delivered_buffer_dot_data_delta")
    if _shape(displacement, displacement_keys, audit, label + " displacement"):
        norms = {name: _number(displacement.get(name), audit, label + " " + name, nonnegative=True)
                 for name in displacement_keys[:5]}
        for prefix in ("total", "data", "ideal_data"):
            norm = norms[prefix + "_norm"]
            validate_leakage(displacement.get(prefix + "_current_basis_leakage"),
                None if norm is None else norm * norm, audit, label + " " + prefix)
        for name in displacement_keys[9:]:
            _number(displacement.get(name), audit, label + " " + name)
        ideal, defect = norms["ideal_data_norm"], norms["actual_minus_ideal_data_norm"]
        if ideal is not None and defect is not None:
            audit.require(displacement.get("actual_minus_ideal_relative") ==
                          (defect / ideal if ideal > 0 else None), label + " relative defect differs")
        if type(normalized) is dict and type(normalized.get("delivered_buffer_norm")) in (int, float):
            _close(ideal, .03 * normalized["delivered_buffer_norm"], audit, label + " ideal-data norm")
    path = row.get("path_statistics")
    if _shape(path, ("data_step_squared_energy", "data_path_length_increment", "raw_gradient_dot_data_step"),
              audit, label + " path") and type(displacement) is dict:
        norm = _number(displacement.get("data_norm"), audit, label + " path data norm", nonnegative=True)
        audit.require(norm is not None and path.get("data_step_squared_energy") == norm * norm
            and path.get("data_path_length_increment") == norm
            and path.get("raw_gradient_dot_data_step") == displacement.get("raw_gradient_dot_data_delta"),
            label + " path statistic identity differs")
    action = row.get("action_diagnostics")
    if _shape(action, ("current_basis_orthogonality_error", "new_buffer_action_idempotence_defect"),
              audit, label + " action diagnostics"):
        _number(action.get("current_basis_orthogonality_error"), audit, label + " orthogonality", nonnegative=True)
        defect = action.get("new_buffer_action_idempotence_defect")
        if _shape(defect, ("norm", "homogeneous_roundoff_reference", "enforced"), audit, label + " idempotence"):
            audit.require(defect.get("enforced") is False, label + " idempotence unexpectedly enforced")
            for field in ("norm", "homogeneous_roundoff_reference"):
                _number(defect.get(field), audit, label + " " + field, nonnegative=True)
    decay = row.get("decay")
    if _shape(decay, ("coefficient", "factor", "manual_before_data_step", "manual_actual_norm",
                       "manual_minus_nominal_norm"), audit, label + " decay"):
        audit.require(decay.get("coefficient") == .01 and decay.get("factor") == .9997
            and decay.get("manual_before_data_step") is True, label + " shrinkage declaration differs")
        for field in ("manual_actual_norm", "manual_minus_nominal_norm"):
            _number(decay.get(field), audit, label + " " + field, nonnegative=True)
    residuals = row.get("algebra_residuals")
    if _shape(residuals, RESIDUALS, audit, label + " residuals"):
        for name in RESIDUALS:
            value = residuals.get(name)
            if _shape(value, ("norm", "homogeneous_error_bound"), audit, label + " " + name):
                norm = _number(value.get("norm"), audit, label + " residual norm", nonnegative=True)
                bound = _number(value.get("homogeneous_error_bound"), audit, label + " residual bound", nonnegative=True)
                if norm is not None:
                    audit.max_algebra_residual = max(audit.max_algebra_residual, norm)
                if norm is not None and bound is not None:
                    audit.require(norm <= bound, label + " residual exceeds saved bound: " + name)
    digests = row.get("digests")
    if expected_step == 101:
        audit.require(type(digests) is dict and tuple(digests) == DIGESTS
            and all(type(value) is str and SHA_RE.fullmatch(value) for value in digests.values()),
            label + " first-step digests differ")
    else:
        audit.require(digests is None, label + " unexpected first-step digests")


def validate_branch(branch: Any, phase: str, audit: Audit, label: str) -> dict[str, Any] | None:
    keys = ("schema", "id", "seed", "target", "policy", "base", "lr", "parent_horizon",
        "parent_state_digest", "parent_evaluation_digest", "status", "requested_updates", "completed_updates",
        "last_completed_horizon", "curve", "steps", "update_seconds", "checkpoints", "failure",
        "first_step_digests", "selected_horizons")
    if not _shape(branch, keys, audit, label + " branch"):
        return None
    seed, target, policy = branch.get("seed"), branch.get("target"), branch.get("policy")
    count, horizons = (10, (100, 110)) if phase == "smoke" else (1900, HORIZONS)
    audit.require(branch.get("schema") == "i17_gain_branch_v1" and policy in NEW_POLICIES
        and seed in ((217,) if phase == "smoke" else SEEDS) and target in TARGETS
        and branch.get("id") == f"s{seed}-sgdm-{target}-{policy}"
        and branch.get("base") == "sgdm" and branch.get("lr") == .03
        and branch.get("parent_horizon") == 100 and branch.get("requested_updates") == count,
        label + " branch identity differs")
    for name in ("parent_state_digest", "parent_evaluation_digest"):
        audit.require(type(branch.get(name)) is str and SHA_RE.fullmatch(branch[name]) is not None,
                      label + " " + name + " malformed")
    steps, curve = branch.get("steps"), branch.get("curve")
    audit.require(type(steps) is list and branch.get("completed_updates") == len(steps),
                  label + " completed-step count differs")
    steps = steps if type(steps) is list else []
    for step, row in enumerate(steps, 101):
        validate_step(row, step, policy, audit, f"{label} step {step}")
    audit.require(type(curve) is list and bool(curve), label + " curve missing")
    curve = curve if type(curve) is list else []
    actual = [row.get("horizon") for row in curve if type(row) is dict]
    audit.require(actual == list(horizons[:len(actual)]) and bool(actual), label + " curve is not scheduled prefix")
    for point in curve:
        if _shape(point, ("horizon", "train", "validation", "auxiliary"), audit, label + " point"):
            validate_evaluation({name: point.get(name) for name in ("train", "validation", "auxiliary")},
                                audit, f"{label} h{point.get('horizon')}")
    if curve:
        audit.require(branch.get("parent_evaluation_digest") == tree_digest(curve[0]),
                      label + " h100 evaluation digest differs")
    status = branch.get("status")
    audit.require(status in ("complete", "numerical_failure"), label + " status differs")
    if status == "complete":
        audit.require(len(steps) == count and actual == list(horizons)
            and branch.get("last_completed_horizon") == horizons[-1] and branch.get("failure") is None
            and branch.get("selected_horizons") == old._selection(curve), label + " complete coverage differs")
    elif status == "numerical_failure":
        failure = branch.get("failure")
        if _shape(failure, ("type", "message", "attempted_step", "completed_step", "last_valid_horizon",
                            "state_artifact"), audit, label + " failure"):
            completed, attempted = failure.get("completed_step"), failure.get("attempted_step")
            position = type(completed) is int and (attempted == completed + 1 or
                (attempted == completed and completed in horizons[1:] and completed not in actual))
            audit.require(failure.get("type") == "NumericalFailure" and type(failure.get("message")) is str
                and 0 < len(failure["message"]) <= 2000 and completed == 100 + len(steps)
                and position and actual and failure.get("last_valid_horizon") == actual[-1]
                and branch.get("last_completed_horizon") == completed and len(steps) <= count
                and branch.get("selected_horizons") is None, label + " numerical failure identity differs")
    _number(branch.get("update_seconds"), audit, label + " update seconds", nonnegative=True)
    first = steps[0].get("digests") if steps else None
    audit.require(branch.get("first_step_digests") == first, label + " first-step copy differs")
    return {"seed": seed, "target": target, "policy": policy, "status": status, "curve": curve,
        "steps": steps, "failure": branch.get("failure"), "checkpoints": branch.get("checkpoints"),
        "parent_state_digest": branch.get("parent_state_digest"),
        "parent_evaluation_digest": branch.get("parent_evaluation_digest"), "first_step_digests": first}


def _best_scalar(branches, seed, target, selector):
    members = [branches.get((seed, target, policy)) for policy in SCALAR_LABELS]
    if any(row is None or row.get("status") != "complete" for row in members):
        return None
    candidates = []
    field = "clean_ce" if selector == SELECTORS[0] else "clean_accuracy"
    sign = 1 if selector == SELECTORS[0] else -1
    for policy, branch in zip(SCALAR_LABELS, members):
        for point in branch["curve"]:
            candidates.append(((sign * point["validation"][field], point["horizon"], K_BY_LABEL[policy]),
                               point, K_BY_LABEL[policy]))
    _, point, k = min(candidates, key=lambda row: row[0])
    return point, k


def selection_results(branches):
    choices, primaries, per_k, selected = [], [], [], {}
    for seed in SEEDS:
        for target in TARGETS:
            for selector in SELECTORS:
                pair = _best_scalar(branches, seed, target, selector)
                scalar, k = pair if pair is not None else (None, None)
                spectral = old._best_policy(branches.get((seed, target, SPECTRAL)), selector)
                selected[seed, target, selector] = scalar, spectral
                def choice(point, *, k=None):
                    if point is None:
                        return None
                    return {**({"k": k} if k is not None else {"policy": SPECTRAL}),
                        "horizon": point["horizon"], "validation_value": point["validation"][
                            "clean_ce" if selector == SELECTORS[0] else "clean_accuracy"],
                        "auxiliary_clean_ce": point["auxiliary"]["clean_ce"],
                        "auxiliary_clean_accuracy": point["auxiliary"]["clean_accuracy"]}
                choices.append({"seed": seed, "target": target, "selector": selector,
                                "scalar": choice(scalar, k=k), "spectral": choice(spectral)})
    for target in TARGETS:
        for selector in SELECTORS:
            for metric in METRICS:
                values = {}
                for seed in SEEDS:
                    scalar, spectral = selected[seed, target, selector]
                    values[str(seed)] = None if scalar is None or spectral is None else (
                        old._utility(spectral, metric) - old._utility(scalar, metric))
                primaries.append({"target": target, "selector": selector,
                    "auxiliary_metric": metric, "effect": old._effect(values)})
                for policy in SCALAR_LABELS:
                    fixed = {}
                    for seed in SEEDS:
                        spectral = selected[seed, target, selector][1]
                        scalar = old._best_policy(branches.get((seed, target, policy)), selector)
                        fixed[str(seed)] = None if scalar is None or spectral is None else (
                            old._utility(spectral, metric) - old._utility(scalar, metric))
                    per_k.append({"k": K_BY_LABEL[policy], "k_label": policy, "target": target,
                        "selector": selector, "auxiliary_metric": metric, "effect": old._effect(fixed)})
    return choices, primaries, per_k


def endpoint_effects(branches):
    results = []
    for policy in SCALAR_LABELS:
        for target in TARGETS:
            for metric in METRICS:
                values = {}
                for seed in SEEDS:
                    scalar = old._point(branches.get((seed, target, policy)), 2000)
                    spectral = old._point(branches.get((seed, target, SPECTRAL)), 2000)
                    values[str(seed)] = None if scalar is None or spectral is None else (
                        old._utility(spectral, metric) - old._utility(scalar, metric))
                results.append({"k": K_BY_LABEL[policy], "k_label": policy, "target": target,
                                "metric": metric, "horizon": 2000, "effect": old._effect(values)})
    return results


trajectory_rows = old.trajectory_rows


def scheduled_aggregates(branches):
    results = []
    for target in TARGETS:
        for policy in POLICIES:
            for horizon in HORIZONS:
                for metric, (split, field) in METRIC_PATHS.items():
                    values, changes = {}, {}
                    for seed in SEEDS:
                        point = old._point(branches.get((seed, target, policy)), horizon)
                        start = old._point(branches.get((seed, target, policy)), 100)
                        values[str(seed)] = None if point is None else point[split][field]
                        changes[str(seed)] = None if point is None or start is None else point[split][field] - start[split][field]
                    effect, progress = old._effect(values), old._effect(changes)
                    results.append({"target": target, "policy": policy, "horizon": horizon,
                        "metric": metric, "per_seed": values, "equal_seed_mean": effect["mean"],
                        "per_seed_change_from_h100": changes,
                        "equal_seed_mean_change_from_h100": progress["mean"],
                        "available": effect["available"], "no_survivor_averaging": True})
    return results


def _equal_seed_tree(values):
    if any(value is None for value in values):
        return None
    if type(values[0]) is dict:
        return {key: _equal_seed_tree([value[key] for value in values]) for key in values[0]}
    return sum(values) / len(SEEDS)


def geometry_results(branches):
    paths, components = [], []
    for target in TARGETS:
        for policy in POLICIES:
            for window, (low, high) in {"steps101_2000": (101, 2000), "steps1001_2000": (1001, 2000)}.items():
                seed_steps = {}
                for seed in SEEDS:
                    branch = branches.get((seed, target, policy))
                    rows = [row for row in branch["steps"] if low <= row["step"] <= high] if (
                        branch is not None and branch.get("status") == "complete") else []
                    seed_steps[seed] = rows if [row["step"] for row in rows] == list(range(low, high + 1)) else []
                per_seed = {}
                for seed, rows in seed_steps.items():
                    if not rows:
                        per_seed[str(seed)] = None
                        continue
                    value = {}
                    for name in ("data", "total") + (() if policy == "scalar_k0" else ("ideal_data",)):
                        norms = [row["displacement"][name + "_norm"] for row in rows]
                        energy = sum(norm * norm for norm in norms)
                        leaks = [row["displacement"][name + "_current_basis_leakage"] for row in rows]
                        value[name] = {"squared_energy_sum": energy, "path_length_sum": sum(norms),
                            "current_action_complement_energy_fraction": (sum(row["outside_squared_norm"]
                                for row in leaks) / energy if energy > 0 and all(row.get("reason") in
                                (None, "zero_displacement") for row in leaks) else None)}
                    for name in ("raw_gradient", "post_mean"):
                        dots = [row["displacement"][name + "_dot_data_delta"] for row in rows]
                        value["sum_" + name + "_dot_data_step"] = sum(dots)
                        value["mean_" + name + "_dot_data_step"] = sum(dots) / len(rows)
                    per_seed[str(seed)] = value
                paths.append({"target": target, "policy": policy, "k": K_BY_LABEL.get(policy),
                    "window": window, "per_seed": per_seed,
                    "equal_seed_mean": _equal_seed_tree(list(per_seed.values())),
                    "available": all(value is not None for value in per_seed.values()),
                    "aggregation": "within branch, then equal seeds; exact complete window required"})
                names = COMPONENTS if policy in NEW_POLICIES else (
                    "raw_gradient", "post_mean", "native_projection_diagnostic", "applied_delivery",
                    "old_momentum_buffer", "scaled_old_momentum_buffer", "new_momentum_buffer")
                for name in names:
                    values = {}
                    for seed, rows in seed_steps.items():
                        if not rows:
                            values[str(seed)] = None
                            continue
                        energy = sum(row["components"][name]["squared_energy"] for row in rows)
                        leaks = [row["components"][name]["current_basis_leakage"] for row in rows]
                        values[str(seed)] = {"mean_squared_energy": energy / len(rows),
                            "current_action_complement_energy_fraction": (sum(row["outside_squared_norm"]
                                for row in leaks) / energy if energy > 0 and all(row.get("reason") in
                                (None, "zero_displacement") for row in leaks) else None)}
                    mean = _equal_seed_tree(list(values.values()))
                    components.append({"target": target, "policy": policy, "window": window,
                        "component": name, "per_seed": values, "equal_seed_mean": mean,
                        "available": all(value is not None for value in values.values()),
                        "aggregation": "within branch, then equal seeds; exact complete window required"})
    return paths, components


def numerical_diagnostics(branches):
    results = []
    for target in TARGETS:
        for policy in NEW_POLICIES:
            values = {}
            for seed in SEEDS:
                branch = branches.get((seed, target, policy))
                if branch is None or branch.get("status") != "complete" or not branch.get("steps"):
                    values[str(seed)] = None
                    continue
                steps = branch["steps"]
                residuals = [value for row in steps for value in row["algebra_residuals"].values()]
                relative = [row["displacement"]["actual_minus_ideal_relative"] for row in steps]
                values[str(seed)] = {
                    "maximum_algebra_residual_norm": max(value["norm"] for value in residuals),
                    "maximum_residual_to_bound_ratio": max(value["norm"] / value["homogeneous_error_bound"]
                        if value["homogeneous_error_bound"] > 0 else 0.0 for value in residuals),
                    "maximum_empirical_ideal_data_step_defect_norm": max(row["displacement"]["actual_minus_ideal_data_norm"] for row in steps),
                    "maximum_empirical_ideal_relative_defect": max((value for value in relative if value is not None), default=None),
                    "maximum_manual_decay_realization_error_norm": max(row["decay"]["manual_minus_nominal_norm"] for row in steps),
                    "maximum_basis_orthogonality_error": max(row["action_diagnostics"]["current_basis_orthogonality_error"] for row in steps),
                    "maximum_new_buffer_action_idempotence_defect": max(row["action_diagnostics"]["new_buffer_action_idempotence_defect"]["norm"] for row in steps),
                }
            results.append({"target": target, "policy": policy, "per_seed": values,
                            "all_seeds_available": all(value is not None for value in values.values())})
    return results


def validate_phase_resources(completion, phase, audit):
    for name in ("update_seconds", "elapsed_seconds"):
        _number(completion.get(name), audit, phase + " " + name, nonnegative=True)
    elapsed = completion.get("elapsed_seconds")
    audit.require(type(elapsed) in (int, float) and math.isfinite(elapsed)
        and elapsed <= (100 if phase == "smoke" else AMENDED_CONFIRMATION_LIMIT),
        phase + " exceeds cooperative wall limit")
    audit.require(type(completion.get("peak_torch_gpu_bytes")) is int
        and 0 <= completion["peak_torch_gpu_bytes"] <= 4 * 1024**3
        and type(completion.get("shared_artifact_bytes_before_completion")) is int
        and 0 <= completion["shared_artifact_bytes_before_completion"] <= ARTIFACT_CAP,
        phase + " resource counters differ")


def phase_records(root, phase, audit):
    directory = root / phase
    try:
        audit.require(directory.is_dir() and not directory.is_symlink(), phase + " directory differs")
        manifest, completion = read_json(directory / "manifest.json"), read_json(directory / "completion.json")
        _shape(manifest, ("schema", "phase", "frozen_commit", "source_hashes", "torch_version",
            "numpy_version", "python_version", "gpu", "real_policies", "test_only_policies", "horizons",
            "smoke_forecast", "timing_safety_factor", "cloud_spend_usd", "authorized_budget_usd",
            "shared_artifact_cap_bytes", "wall_limit_seconds", "runtime_directory", "runtime_bytes_count_in_cap"),
            audit, phase + " manifest")
        _shape(completion, ("schema", "status", "phase", "source_hashes", "frozen_commit", "branches",
            "numerical_failures", "all_requested_endpoints_present", "completed_training_updates", "update_seconds",
            "elapsed_seconds", "peak_torch_gpu_bytes", "shared_artifact_bytes_before_completion", "artifacts"),
            audit, phase + " completion")
        audit.require(manifest.get("schema") == "i17_manifest_v1" and manifest.get("phase") == phase
            and completion.get("schema") == "i17_completion_v1" and completion.get("status") == "complete"
            and completion.get("phase") == phase, phase + " schema/status differs")
        audit.require(manifest.get("frozen_commit") == completion.get("frozen_commit")
            and manifest.get("source_hashes") == completion.get("source_hashes"), phase + " source binding differs")
        audit.require(manifest.get("real_policies") == list(NEW_POLICIES)
            and manifest.get("test_only_policies") == ["scalar_k0"] and manifest.get("horizons") == list(HORIZONS)
            and manifest.get("timing_safety_factor") == 1.5 and manifest.get("cloud_spend_usd") == 0
            and manifest.get("authorized_budget_usd") == 100 and manifest.get("shared_artifact_cap_bytes") == ARTIFACT_CAP
            and manifest.get("wall_limit_seconds") == (100 if phase == "smoke" else AMENDED_CONFIRMATION_LIMIT)
            and manifest.get("runtime_directory") == "runtime" and manifest.get("runtime_bytes_count_in_cap") is True,
            phase + " protocol declaration differs")
        count = 8 if phase == "smoke" else 24
        failures = completion.get("numerical_failures")
        audit.require(completion.get("branches") == count and type(failures) is int and 0 <= failures <= count
            and completion.get("all_requested_endpoints_present") is (failures == 0), phase + " count/failure differs")
        if phase == "smoke":
            audit.require(failures == 0, "smoke did not qualify with zero failures")
        validate_phase_resources(completion, phase, audit)
        records = completion.get("artifacts")
        audit.require(type(records) is list and all(type(row) is dict for row in records), phase + " artifact list differs")
        records = records if type(records) is list else []
        index = {row.get("name"): row for row in records if type(row) is dict}
        audit.require(len(index) == len(records), phase + " duplicate/malformed artifact name")
        audit.require({path.name for path in directory.iterdir()} == set(index) | {"completion.json"},
                      phase + " unlisted, partial or missing physical artifact")
        for name, record in index.items():
            verify_record(directory, record, audit, f"{phase}/{name}")
        path = directory / "completion.json"
        info = path.lstat()
        audit.require(stat.S_ISREG(info.st_mode) and not path.is_symlink(), phase + " completion is not regular")
        if stat.S_ISREG(info.st_mode) and not path.is_symlink():
            audit.phase_completion_sha256[phase] = sha256(path)
            audit.hash_files += 1
            audit.hash_bytes += info.st_size
            if phase == "smoke":
                audit.require(audit.phase_completion_sha256[phase] == SMOKE_COMPLETION_SHA
                    and manifest.get("frozen_commit") == SMOKE_COMMIT,
                    "consumed smoke completion/commit pin differs")
        return manifest, completion, index
    except Exception as exc:
        audit.errors.append(f"{phase} loading failed: {type(exc).__name__}: {exc}")
        return {}, {}, {}


def verify_sources(manifests, audit):
    audit.require(type(manifests) is list and len(manifests) == 2,
                  "source admission requires exactly smoke and confirmation manifests")
    lineage = {}
    for position, (phase, closure) in enumerate((("smoke", ACQUISITION_SOURCES),
                                               ("confirmation", CONFIRMATION_SOURCES))):
        manifest = manifests[position] if type(manifests) is list and len(manifests) > position else {}
        commit, sources = manifest.get("frozen_commit"), manifest.get("source_hashes")
        audit.require(manifest.get("phase") == phase, phase + " source-manifest phase differs")
        audit.require(type(commit) is str and re.fullmatch(r"[0-9a-f]{40}", commit) is not None,
                      phase + " acquisition full commit differs")
        audit.require(commit == SMOKE_COMMIT if phase == "smoke" else commit != SMOKE_COMMIT,
                      phase + " amended lineage commit differs")
        audit.require(type(sources) is dict and tuple(sources) == closure
            and all(type(value) is str and SHA_RE.fullmatch(value) for value in sources.values()),
            phase + f" exact {len(closure)}-file source closure differs")
        verified = []
        if type(commit) is str and type(sources) is dict:
            for name, expected in sources.items():
                if name not in closure:
                    continue
                try:
                    path = REPO / name
                    audit.require(path.is_file() and not path.is_symlink() and sha256(path) == expected,
                                  phase + " worktree source differs: " + name)
                    frozen = subprocess.check_output(["git", "show", commit + ":" + name], cwd=REPO)
                    audit.require(hashlib.sha256(frozen).hexdigest() == expected,
                                  phase + " committed source differs: " + name)
                    verified.append(name)
                except Exception as exc:
                    audit.errors.append(f"{phase} source {name}: {type(exc).__name__}: {exc}")
        lineage[phase] = {"frozen_commit": commit, "source_hashes": sources, "verified_files": verified}
    smoke_sources, confirmation_sources = (lineage[phase]["source_hashes"] for phase in PHASES)
    shared = (type(smoke_sources) is dict and type(confirmation_sources) is dict
        and tuple(smoke_sources) == ACQUISITION_SOURCES
        and list(confirmation_sources.items())[:len(ACQUISITION_SOURCES)] == list(smoke_sources.items()))
    audit.require(shared, "confirmation must retain the identical ordered 24-file smoke prefix")
    return {**lineage["confirmation"], "smoke": lineage["smoke"],
        "shared_original_24_source_hashes_unchanged": shared,
        "resource_only_amendment_sources": list(CONFIRMATION_SOURCES[len(ACQUISITION_SOURCES):]),
        "consumed_smoke_completion_sha256": SMOKE_COMPLETION_SHA}


def verify_analysis_sources(commit, audit):
    audit.require(type(commit) is str and re.fullmatch(r"[0-9a-f]{40}", commit) is not None,
                  "analysis full commit differs")
    hashes = {}
    for path in (HERE / "analyse_gain.py", HERE / "test_analyse_gain.py", I16 / "analyse_scalar.py",
                 HERE.parent / "iteration-015/analyse_history.py"):
        name, current = str(path.relative_to(REPO)), sha256(path)
        try:
            frozen = subprocess.check_output(["git", "show", commit + ":" + name], cwd=REPO)
            audit.require(hashlib.sha256(frozen).hexdigest() == current, "analysis source not frozen: " + name)
        except Exception as exc:
            audit.errors.append(f"analysis source {name}: {type(exc).__name__}: {exc}")
        hashes[name] = current
    return {"frozen_commit": commit, "source_hashes": hashes}


def validate_phase_aggregate(completion, entries, warmup, audit, label):
    audit.require(type(entries) is list and all(type(row) is dict for row in entries), label + " entries differ")
    entries = entries if type(entries) is list else []
    times, updates = [], []
    for row in entries:
        if type(row) is not dict:
            continue
        seconds = _number(row.get("update_seconds"), audit, label + " branch seconds", nonnegative=True)
        if seconds is not None:
            times.append(seconds)
        count = row.get("completed_updates")
        audit.require(type(count) is int and count >= 0, label + " completed count differs")
        if type(count) is int:
            updates.append(count)
    audit.require(len(times) == len(entries) and len(updates) == len(entries)
        and completion.get("update_seconds") == sum(times)
        and completion.get("completed_training_updates") == warmup + sum(updates)
        and completion.get("numerical_failures") == sum(row.get("status") == "numerical_failure"
            for row in entries if type(row) is dict), label + " completion aggregate differs")


def validate_cross_phase_timing(phases, entries, audit):
    rates = {}
    audit.require(len(entries) == 8 and {(row.get("seed"), row.get("target"), row.get("policy")) for row in entries}
        == {(217, target, policy) for target in TARGETS for policy in NEW_POLICIES}, "smoke forecast roster differs")
    for policy in NEW_POLICIES:
        group = [row for row in entries if row.get("policy") == policy]
        good = len(group) == 2 and all(row.get("status") == "complete" and row.get("completed_updates") == 10
            and type(row.get("update_seconds")) is float and math.isfinite(row["update_seconds"])
            and row["update_seconds"] > 0 for row in group)
        audit.require(good, "smoke timing policy not qualified: " + policy)
        if good:
            rates[policy] = sum(row["update_seconds"] for row in group) / 20
    if len(rates) != 4:
        return
    expected = {"policy_seconds_per_update": rates, "confirmation_seconds": 1.5 * max(rates.values()) * 45600 + 60}
    audit.require(phases["smoke"][0].get("smoke_forecast") is None
        and phases["confirmation"][0].get("smoke_forecast") == expected
        and ORIGINAL_CONFIRMATION_LIMIT < expected["confirmation_seconds"] <= AMENDED_CONFIRMATION_LIMIT,
        "confirmation forecast differs or amended lineage/gate does not qualify")
    return {"schema": "i17_resource_only_amendment_audit_v1",
        "smoke_frozen_commit": SMOKE_COMMIT, "smoke_completion_sha256": SMOKE_COMPLETION_SHA,
        "formula": "1.5 * max_policy(sum(two policy branch seconds)/20) * 45600 + 60",
        "saved_smoke_forecast": expected,
        "original_confirmation_limit_seconds": ORIGINAL_CONFIRMATION_LIMIT,
        "original_gate_passed": expected["confirmation_seconds"] <= ORIGINAL_CONFIRMATION_LIMIT,
        "amended_confirmation_limit_seconds": AMENDED_CONFIRMATION_LIMIT,
        "amended_gate_passed": expected["confirmation_seconds"] <= AMENDED_CONFIRMATION_LIMIT,
        "new_smoke_or_scientific_design_change": False}


def load_references(binding, audit):
    """Load only accepted k0 scalar records; hash, but never load, old terminals."""
    result, initial_errors = {}, len(audit.errors)
    if not _shape(binding, ("schema", "inherited_i16", "i16", "k0_references", "source_replayed"),
                  audit, "parent/reference binding"):
        return result
    audit.require(binding.get("schema") == "i17_parent_and_reference_binding_v1"
        and binding.get("source_replayed") is False, "parent/reference identity differs")
    audit.require(binding.get("i16") == {"artifact_root": str(I16_ROOT), "acquisition_commit": I16_COMMIT,
        "pinned_sha256": PINS, "confirmation_sha256": I16_COMPLETION_SHA}, "I16 pin set differs")
    pinned = {}
    for name, digest in PINS.items():
        path = I16 / name
        if verify_file(path, path.stat().st_size, digest, audit, "I16 " + name):
            pinned[name] = read_json(path)
    path = I16_ROOT / "confirmation/completion.json"
    if not verify_file(path, path.stat().st_size, I16_COMPLETION_SHA, audit, "I16 completion"):
        return result
    completion = read_json(path)
    accepted_audit = pinned.get("analysis-001/audit.json", {})
    summary = pinned.get("analysis-001/summary.json", {})
    report = pinned.get("analysis-001/report-audit.json", {})
    collection = pinned.get("raw-results-001/collection.json", {})
    audit.require(accepted_audit.get("schema") == "i16_scalar_analysis_audit_v1"
        and accepted_audit.get("status") == "pass" and accepted_audit.get("errors") == []
        and accepted_audit.get("artifact_root") == str(I16_ROOT)
        and accepted_audit.get("phase_completion_sha256", {}).get("confirmation") == I16_COMPLETION_SHA
        and summary.get("schema") == "i16_scalar_analysis_summary_v1" and summary.get("audit_status") == "pass"
        and summary.get("artifact_root") == str(I16_ROOT) and report.get("status") == "pass"
        and report.get("errors") == [] and report.get("maximum_absolute_numeric_difference") == 0
        and report.get("inputs", {}).get("i16_summary", {}).get("sha256") == PINS["analysis-001/summary.json"]
        and collection.get("status") == "complete" and collection.get("source_root") == str(I16_ROOT)
        and collection.get("analysis_audit", {}).get("sha256") == PINS["analysis-001/audit.json"]
        and collection.get("phase_completion_sha256", {}).get("confirmation") == I16_COMPLETION_SHA,
        "I16 accepted evidence status/binding differs")
    audit.require(completion.get("schema") == "i16_completion_v1" and completion.get("status") == "complete"
        and completion.get("phase") == "confirmation" and completion.get("frozen_commit") == I16_COMMIT
        and completion.get("branches") == 18 and completion.get("numerical_failures") == 0
        and completion.get("completed_training_updates") == 34200
        and completion.get("all_requested_endpoints_present") is True, "I16 completion membership differs")
    records = completion.get("artifacts", [])
    index = {row.get("name"): row for row in records if type(row) is dict}
    audit.require(type(records) is list and len(records) == len(index), "I16 artifact index differs")
    directory = I16_ROOT / "confirmation"
    original_path = verify_record(directory, index.get("parent-inputs.json"), audit, "I16 original parent binding")
    if original_path is not None:
        audit.require(binding.get("inherited_i16") == read_json(original_path), "transitive I16 parent binding differs")
    if len(audit.errors) != initial_errors:
        return result  # Never load a parent through unaccepted provenance.
    # This hashes transitive I14/I15 references and CPU-digests six used h100
    # parents only. It does not call any historical analyze()/main() entrypoint.
    inherited = old.load_references(binding["inherited_i16"], audit)
    parent_index = {(row["seed"], row["target"]): row for row in binding["inherited_i16"]["i14"]["parents"]}
    trajectories = summary.get("all_policy_trajectories", [])
    summaries = {(row.get("seed"), row.get("target")): row for row in trajectories
                 if type(row) is dict and row.get("policy") == "k0"}
    expected = {(seed, target) for seed in SEEDS for target in TARGETS}
    audit.require(type(trajectories) is list and len(trajectories) == 30 and set(summaries) == expected,
                  "I16 k0 summary membership differs")
    branch_index_path = verify_record(directory, index.get("branches.json"), audit, "I16 branch index")
    entries = read_json(branch_index_path).get("entries", []) if branch_index_path else []
    entry_index = {(row.get("seed"), row.get("target"), row.get("k_label")): row for row in entries if type(row) is dict}
    audit.require(len(entries) == len(entry_index) == 18 and set(entry_index)
        == {(seed, target, policy) for seed in SEEDS for target in TARGETS for policy in ("k0", "k0p5", "k0p9")},
        "I16 original branch roster differs")
    supplied = binding.get("k0_references")
    audit.require(type(supplied) is list and len(supplied) == 6, "I16 k0 supplied reference count differs")
    observed = set()
    for reference in supplied if type(supplied) is list else []:
        if not _shape(reference, ("seed", "target", "curve_artifact", "checkpoint_records",
                                 "parent_state_digest", "parent_evaluation_digest"), audit, "I16 k0 reference"):
            continue
        key = reference.get("seed"), reference.get("target")
        audit.require(key in expected and key not in observed, "I16 k0 reference identity duplicate/unknown")
        observed.add(key)
        artifact = reference.get("curve_artifact")
        entry = entry_index.get(key + ("k0",), {})
        audit.require(type(artifact) is dict and artifact == index.get(artifact.get("name"))
            and artifact == entry.get("artifact"), "I16 k0 branch artifact binding differs")
        path = verify_record(directory, artifact, audit, "I16 k0 branch " + str(key))
        if path is None:
            continue
        branch = read_json(path)
        identity_fields = ("id", "seed", "target", "k", "k_label", "status", "completed_updates",
                           "parent_state_digest", "parent_evaluation_digest", "first_step_digests", "scalar_update_seconds")
        audit.require(all(branch.get(name) == entry.get(name) for name in identity_fields)
            and branch.get("k") == 0.0 and branch.get("k_label") == "k0"
            and branch.get("status") == "complete" and branch.get("failure") is None
            and branch.get("checkpoints") == reference.get("checkpoint_records")
            and branch.get("parent_state_digest") == reference.get("parent_state_digest") == parent_index.get(key, {}).get("state_digest"),
            "I16 k0 direct branch binding differs")
        value = old.validate_branch(branch, "confirmation", audit, "accepted I16 k0 " + str(key))
        start = old._point(inherited.get(key + ("k1",)), 100)
        audit.require(start is not None and branch.get("curve", [None])[0] == start
            and branch.get("parent_evaluation_digest") == reference.get("parent_evaluation_digest") == tree_digest(start),
            "I16 k0 parent seam differs")
        checkpoints = branch.get("checkpoints", [])
        audit.require([row.get("horizon") for row in checkpoints] == list(HORIZONS[1:]), "I16 k0 checkpoint roster differs")
        for checkpoint in checkpoints:
            horizon = checkpoint.get("horizon")
            kind = "full_state" if horizon == 2000 else "model_state"
            record = checkpoint.get(kind)
            audit.require(type(record) is dict and record == index.get(record.get("name")), "I16 k0 checkpoint binding differs")
            verify_record(directory, record, audit, f"I16 k0 checkpoint {key}/h{horizon}")
        summary_row = summaries.get(key, {})
        audit.require(summary_row.get("status") == "complete" and summary_row.get("failure") is None
            and [row.get("horizon") for row in summary_row.get("curve", [])] == list(HORIZONS),
            "I16 k0 summary coverage differs")
        for saved, point in zip(summary_row.get("curve", []), branch.get("curve", [])):
            audit.require(saved.get("horizon") == point.get("horizon")
                and old._summary_values_match(saved.get("values"), point), "I16 k0 summary/direct values differ")
        if value is not None:
            result[key + ("scalar_k0",)] = {**value, "policy": "scalar_k0", "source": "accepted_i16_k0_reused_not_replayed"}
    audit.require(observed == expected and set(result) == {key + ("scalar_k0",) for key in expected},
                  "I16 k0 complete logical membership differs")
    return result


def _phase_branches(root, phase, phase_data, audit):
    directory, index = root / phase, phase_data[2]
    path = verify_record(directory, index.get("branches.json"), audit, phase + " branch index")
    envelope = read_json(path) if path else {}
    expected_keys = (("entries", "first_step_pair_checks", "synthetic_warmup_updates", "new_gain_updates")
                     if phase == "smoke" else ("entries", "first_step_pair_checks", "new_gain_branches", "reused_i16_k0"))
    _shape(envelope, expected_keys, audit, phase + " branch index")
    audit.require((phase == "smoke" and envelope.get("synthetic_warmup_updates") == 200 and envelope.get("new_gain_updates") == 80)
        or (phase == "confirmation" and envelope.get("new_gain_branches") == 24 and envelope.get("reused_i16_k0") == 6),
        phase + " branch index declarations differ")
    entries = envelope.get("entries", [])
    seeds = (217,) if phase == "smoke" else SEEDS
    expected = {(seed, target, policy) for seed in seeds for target in TARGETS for policy in NEW_POLICIES}
    audit.require(type(entries) is list and len(entries) == len(expected), phase + " branch count differs")
    branches, observed = {}, set()
    used = {"manifest.json", "branches.json"} | ({"synthetic-parent-clean.pt", "synthetic-parent-fixed.pt"}
        if phase == "smoke" else {"parent-inputs.json", "parent-seams.json"}
        | {f"{prefix}-s{seed}.json" for seed in SEEDS for prefix in ("plan", "corruption")})
    for entry in entries if type(entries) is list else []:
        if not _shape(entry, ("id", "seed", "target", "policy", "status", "completed_updates",
                             "parent_state_digest", "parent_evaluation_digest", "first_step_digests", "update_seconds", "artifact"),
                      audit, phase + " index entry"):
            continue
        key = entry.get("seed"), entry.get("target"), entry.get("policy")
        audit.require(key in expected and key not in observed, phase + " unknown/duplicate branch identity")
        observed.add(key)
        artifact = entry.get("artifact")
        audit.require(type(artifact) is dict and artifact == index.get(artifact.get("name"))
            and artifact.get("name") == f"branch-{entry.get('id')}.json", phase + " indexed branch artifact differs")
        path = verify_record(directory, artifact, audit, phase + " branch " + str(key))
        if path is None:
            continue
        used.add(path.name)
        branch = read_json(path)
        audit.require(all(branch.get(name) == value for name, value in entry.items() if name != "artifact"),
                      phase + " branch/index scalar binding differs")
        value = validate_branch(branch, phase, audit, phase + " " + str(key))
        validate_checkpoints(branch, index, directory, audit, phase + " " + str(key))
        for checkpoint in branch.get("checkpoints", []):
            horizon = checkpoint.get("horizon")
            kind = "full_state" if "full_state" in checkpoint else "model_state"
            record = checkpoint.get(kind, {})
            expected_name = f"{'state' if kind == 'full_state' else 'model'}-{entry.get('id')}-h{horizon}.pt"
            audit.require(record.get("name") == expected_name, phase + " checkpoint filename association differs")
            used.add(record.get("name"))
        if type(branch.get("failure")) is dict:
            name = branch["failure"].get("state_artifact", {}).get("name")
            audit.require(name == f"failed-state-{entry.get('id')}.pt", phase + " failure filename differs")
            used.add(name)
        if value is not None:
            branches[key] = value
    audit.require(observed == expected and set(branches) == expected, phase + " exact branch roster differs")
    audit.require(set(index) == used, phase + " declared artifact role membership differs")
    validate_phase_aggregate(phase_data[1], entries, 200 if phase == "smoke" else 0, audit, phase)
    pairs = []
    for seed in seeds:
        for target in TARGETS:
            group = [branches.get((seed, target, policy)) for policy in NEW_POLICIES]
            if any(row is None for row in group):
                continue
            for name in ("parent_state_digest", "parent_evaluation_digest"):
                audit.require(len({row[name] for row in group}) == 1, phase + " common parent differs: " + name)
            available = [row for row in group if type(row["first_step_digests"]) is dict]
            audit.require(all(row["status"] == "numerical_failure" for row in group if row not in available),
                          phase + " completed branch lacks first step")
            for name in DIGESTS[:3]:
                audit.require(len({row["first_step_digests"].get(name) for row in available}) <= 1,
                              phase + " first-step common digest differs: " + name)
            pairs.append({"seed": seed, "target": target, "available_first_steps": len(available),
                          "status": "pass" if len(available) == 4 else "partial_numerical_evidence"})
            if phase == "smoke":
                record = index.get(f"synthetic-parent-{target}.pt")
                parent_path = verify_record(directory, record, audit, "synthetic parent " + target)
                if parent_path is not None:
                    base._load_state(parent_path, group[0]["parent_state_digest"], audit, "synthetic parent " + target)
    audit.require(envelope.get("first_step_pair_checks") == pairs, phase + " saved first-step checks differ")
    return branches, entries


def analyze(root: Path, analysis_commit: str, audit: Audit):
    audit.artifact_root = str(root)
    runtime = runtime_inventory(root, audit)
    audit.runtime_inventory = runtime
    total_bytes = base.total_regular_bytes(root, audit)
    phases = {phase: phase_records(root, phase, audit) for phase in PHASES}
    acquisition = verify_sources([phases[phase][0] for phase in PHASES], audit)
    analysis = verify_analysis_sources(analysis_commit, audit)
    for phase in PHASES:
        path = root / f"attempt-{phase}.json"
        info = path.lstat()
        audit.require(stat.S_ISREG(info.st_mode) and not path.is_symlink(), phase + " attempt not regular")
        attempt = read_json(path)
        phase_acquisition = acquisition["smoke"] if phase == "smoke" else acquisition
        _shape(attempt, ("schema", "phase", "frozen_commit", "source_hashes", "pid", "restart"), audit, phase + " attempt")
        audit.require(attempt.get("schema") == "i17_attempt_v1" and attempt.get("phase") == phase
            and attempt.get("frozen_commit") == phase_acquisition["frozen_commit"]
            and attempt.get("source_hashes") == phase_acquisition["source_hashes"]
            and type(attempt.get("pid")) is int and attempt["pid"] > 0 and attempt.get("restart") == "forbidden",
            phase + " attempt identity differs")
        audit.attempt_sha256[phase] = sha256(path)
        audit.hash_files += 1
        audit.hash_bytes += info.st_size
    if audit.errors:
        raise ValueError("root/phase/source admission failed; no state loading admitted")
    directory, index = root / "confirmation", phases["confirmation"][2]
    path = verify_record(directory, index.get("parent-inputs.json"), audit, "parent inputs")
    binding = read_json(path) if path else {}
    branches = load_references(binding, audit)
    if audit.errors:
        raise ValueError("accepted reference admission failed; no new state loading admitted")
    seam_path = verify_record(directory, index.get("parent-seams.json"), audit, "parent seams")
    seams = read_json(seam_path) if seam_path else {}
    _shape(seams, ("parents", "parents_loaded", "scientific_updates_before_admission"), audit, "parent seams")
    audit.require(seams.get("parents_loaded") == 6 and seams.get("scientific_updates_before_admission") == 0
        and type(seams.get("parents")) is list and len(seams["parents"]) == 6, "parent seam count differs")
    seam_index = {}
    for row in seams.get("parents", []):
        _shape(row, ("seed", "target", "state_digest", "evaluation_digest", "rng_neutral", "status"), audit, "parent seam row")
        key = row.get("seed"), row.get("target")
        audit.require(key not in seam_index and row.get("rng_neutral") is True and row.get("status") == "pass",
                      "parent seam row identity differs")
        seam_index[key] = row
    expected_parents = {(seed, target) for seed in SEEDS for target in TARGETS}
    audit.require(set(seam_index) == expected_parents, "parent seam roster differs")
    i14_binding = binding["inherited_i16"]["i14"]
    parents = {(row["seed"], row["target"]): row for row in i14_binding["parents"]}
    plan_bindings = []
    for seed in SEEDS:
        for prefix in ("plan", "corruption"):
            name = f"{prefix}-s{seed}.json"
            path = verify_record(directory, index.get(name), audit, "I17 " + name)
            if path is None:
                continue
            value = read_json(path)
            if prefix == "plan":
                old_name = f"plan-confirmation-s{seed}.json"
                old_path = verify_record(base.I14_ROOT / "confirmation", i14_binding["source_records"].get(old_name), audit, "I14 " + old_name)
                if old_path is not None:
                    equal = value == read_json(old_path)
                    audit.require(equal, f"seed {seed} saved plan differs")
                    plan_bindings.append({"seed": seed, "i17_sha256": sha256(path), "i14_sha256": sha256(old_path), "exact_json_tree_equal": equal})
            else:
                audit.require(value == i14_binding["corruption_counts"].get(str(seed)), f"seed {seed} corruption differs")
    smoke, smoke_entries = _phase_branches(root, "smoke", phases["smoke"], audit)
    new, _ = _phase_branches(root, "confirmation", phases["confirmation"], audit)
    resource_amendment = validate_cross_phase_timing(phases, smoke_entries, audit)
    branches.update(new)
    for seed, target in sorted(expected_parents):
        start = old._point(branches.get((seed, target, "scalar_k0")), 100)
        seam, parent = seam_index.get((seed, target), {}), parents.get((seed, target), {})
        for policy in NEW_POLICIES:
            row = branches.get((seed, target, policy), {})
            audit.require(row.get("parent_state_digest") == seam.get("state_digest") == parent.get("state_digest"), "parent complete-state seam differs")
            audit.require(row.get("parent_evaluation_digest") == seam.get("evaluation_digest") == tree_digest(start)
                and row.get("curve", [None])[0] == start, "h100 evaluation seam differs")
    audit.require(set(branches) == {(seed, target, policy) for seed in SEEDS for target in TARGETS for policy in POLICIES},
                  "complete five-policy roster differs")
    # Malformed scalar input must never produce a apparently valid selected mean.
    if audit.errors:
        raise ValueError("branch/seam scalar integrity failed; scalar results unavailable")
    choices, primaries, per_k = selection_results(branches)
    paths, components = geometry_results(branches)
    return {"schema": "i17_gain_analysis_summary_v1", "artifact_root": str(root),
        "scope": ["Six accepted I16 k0 references are reused without training, forwards, or old terminal-state loads.",
            "The original 1800-second synthetic-forecast gate failed; a documented pre-scientific resource-only amendment admits 2400 seconds without rerunning smoke or changing the scientific closure.",
            "Six used I14 parents, two synthetic parents, and each new complete/failure state are CPU tree-digested; other state files are hash-checked only.",
            "All four scalar policies are required for joint selection; failed branches and missing seeds never permit survivor averaging.",
            "Stationary gain and common shrinkage do not imply equal finite-time dose, fixed kernels, or identical effective regularization.",
            "Action-complement geometry and bounded local algebra residuals are not semantic utility or nonlinear trajectory-error bounds."],
        "counts": {"new_smoke_branches": len(smoke), "new_confirmation_branches": len(new), "reused_i16_k0": 6,
                   "confirmation_numerical_failures": phases["confirmation"][1]["numerical_failures"]},
        "input_provenance": {"acquisition": acquisition, "analysis": analysis, "parent_reference_binding": binding,
            "plan_bindings": plan_bindings, "runtime_inventory": runtime, "resource_amendment": resource_amendment,
            "artifact_root_regular_file_bytes": total_bytes, "artifact_cap_bytes": ARTIFACT_CAP},
        "primary_validation_selected_spectral_minus_scalar": primaries, "validation_selected_choices": choices,
        "validation_selected_spectral_minus_each_k": per_k, "fixed_k_endpoint_spectral_minus_scalar": endpoint_effects(branches),
        "all_policy_trajectories": trajectory_rows(branches), "scheduled_metric_equal_seed_aggregates": scheduled_aggregates(branches),
        "path_and_displacement_geometry": paths, "component_energy_and_action_complement": components,
        "numerical_diagnostics": numerical_diagnostics(branches)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frozen-commit", required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise SystemExit("analysis requires explicit CUDA_VISIBLE_DEVICES= (CPU-only)")
    if any(os.environ.get(name) != "1" for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")):
        raise SystemExit("analysis requires numerical thread environment limits of one")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    root = args.root.resolve(strict=True)
    if args.root.absolute() != root or root.is_symlink() or root != ARTIFACT_ROOT:
        raise SystemExit("artifact root must be the single pinned canonical I17 root")
    if subprocess.check_output(["findmnt", "-n", "-o", "SOURCE", "-T", str(root)], text=True).strip() != "/dev/RECONFIGURE_FOR_LOCAL_STORAGE":
        raise SystemExit("artifact root is not on the declared volume")
    output = args.output
    if output.name != "analysis-001" or output.parent.resolve(strict=True) != HERE or output.is_symlink():
        raise SystemExit("only a new iteration-017/analysis-001 is admitted")
    output.mkdir(exist_ok=False)
    audit = Audit()
    audit.artifact_root = str(root)
    try:
        summary = analyze(root, args.frozen_commit, audit)
    except BaseException as exc:
        audit.errors.append(f"analysis aborted: {type(exc).__name__}: {exc}")
        summary = {"schema": "i17_gain_analysis_summary_v1", "artifact_root": str(root),
                   "status": "unavailable_due_to_analysis_error"}
    summary["audit_status"] = "pass" if not audit.errors else "fail"
    payload = {"schema": "i17_gain_analysis_audit_v1", "status": summary["audit_status"], "artifact_root": audit.artifact_root,
        "phase_completion_sha256": audit.phase_completion_sha256, "attempt_sha256": audit.attempt_sha256,
        "runtime_inventory": audit.runtime_inventory, "checks": audit.checks, "errors": audit.errors, "warnings": audit.warnings,
        "hash_files_verified": audit.hash_files, "hash_bytes_streamed": audit.hash_bytes,
        "complete_state_tree_digests_verified": audit.tree_digests, "maximum_saved_algebra_residual": audit.max_algebra_residual,
        "maximum_leakage_identity_error": audit.max_leakage_identity_error, "failure_state_tree_digests": audit.failure_state_tree_digests}
    for name, value in (("summary.json", summary), ("audit.json", payload)):
        with (output / name).open("x", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, allow_nan=False, ensure_ascii=True)
            handle.write("\n")
    print(json.dumps({"status": payload["status"], "checks": audit.checks, "errors": len(audit.errors),
                      "hash_files": audit.hash_files, "tree_digests": audit.tree_digests}))
    return 0 if not audit.errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
