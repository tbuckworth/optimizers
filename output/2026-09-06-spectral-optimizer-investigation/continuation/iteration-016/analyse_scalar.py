#!/usr/bin/env python3
"""Independent CPU audit and scalar analysis for the fixed I16 study."""
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
I15 = HERE.parent / "iteration-015"
I15_ROOT = Path("/tmp/spectral-experiment-artifacts/spectral-i15-001.RTFjXq")
I15_AUDIT_SHA = "53f9c65d7e189567d017bf5b8a4488629e27308d02d90cc1426b000f802837ac"
I15_SUMMARY_SHA = "71cb51a6933b52594fb97f29e322b3dd3b46489cc28f4eab442c209dfb15a754"
I15_COLLECTION_SHA = "b50bd33762ac58b77fc0755418934432e503f3a92343724f33cef6cfffa700fa"
I15_COMPLETION_SHA = "2e28e146ef84d934bc33d41f30dfa92cfbcbe7cd19a9b506aec83a4ef7635be6"
_BASE_SPEC = importlib.util.spec_from_file_location(
    "_i16_immutable_i15_analysis", I15 / "analyse_history.py")
if _BASE_SPEC is None or _BASE_SPEC.loader is None:  # pragma: no cover
    raise ImportError("cannot load immutable I15 analysis helpers")
base = importlib.util.module_from_spec(_BASE_SPEC)
_BASE_SPEC.loader.exec_module(base)

PHASES = ("smoke", "confirmation")
SEEDS = (200, 201, 202)
TARGETS = ("clean", "fixed")
K_BY_LABEL = {"k0": 0.0, "k0p5": 0.5, "k0p9": 0.9, "k1": 1.0}
NEW_LABELS = ("k0", "k0p5", "k0p9")
SCALAR_LABELS = tuple(K_BY_LABEL)
SPECTRAL = "mean_projected_history"
POLICIES = SCALAR_LABELS + (SPECTRAL,)
HORIZONS = (100, 250, 500, 1000, 1500, 2000)
SELECTORS = ("minimum_validation_ce", "maximum_validation_accuracy")
METRICS = ("ce", "accuracy")
ARTIFACT_CAP = 3 * 1024**3
SHA_RE = re.compile(r"[0-9a-f]{64}")
ACQUISITION_SOURCES = (
    "spectral_filter.py",
    "output/2026-09-06-spectral-optimizer-investigation/continuation/iteration-009/neural_core.py",
    "output/2026-09-06-spectral-optimizer-investigation/continuation/iteration-014/optimizer_core.py",
    "output/2026-09-06-spectral-optimizer-investigation/continuation/iteration-014/data_plan.py",
    "output/2026-09-06-spectral-optimizer-investigation/continuation/iteration-014/run_cross_optimizer.py",
    "output/2026-09-06-spectral-optimizer-investigation/continuation/iteration-013/mean_core.py",
    "output/2026-09-06-spectral-optimizer-investigation/continuation/iteration-015/history_core.py",
    "output/2026-09-06-spectral-optimizer-investigation/continuation/iteration-015/test_history_core.py",
    "output/2026-09-06-spectral-optimizer-investigation/continuation/iteration-015/run_history_branches.py",
    "output/2026-09-06-spectral-optimizer-investigation/continuation/iteration-015/test_history_runner.py",
    "output/2026-09-06-spectral-optimizer-investigation/continuation/iteration-015/protocol.md",
    "output/2026-09-06-spectral-optimizer-investigation/continuation/iteration-015/predictions.md",
    "output/2026-09-06-spectral-optimizer-investigation/continuation/iteration-016/scalar_core.py",
    "output/2026-09-06-spectral-optimizer-investigation/continuation/iteration-016/test_scalar_core.py",
    "output/2026-09-06-spectral-optimizer-investigation/continuation/iteration-016/run_scalar_controls.py",
    "output/2026-09-06-spectral-optimizer-investigation/continuation/iteration-016/test_scalar_runner.py",
    "output/2026-09-06-spectral-optimizer-investigation/continuation/iteration-016/protocol.md",
    "output/2026-09-06-spectral-optimizer-investigation/continuation/iteration-016/predictions.md",
)

Audit = base.Audit
sha256 = base.sha256
read_json = base.read_json
tree_digest = base.tree_digest
finite = base.finite
verify_file = base.verify_file
verify_record = base.verify_record
validate_evaluation = base.validate_evaluation
validate_leakage = base.validate_leakage
validate_component = base.validate_component


def _safe_relative(name: Any) -> bool:
    return (type(name) is str and name not in ("", ".", "..")
            and not Path(name).is_absolute() and ".." not in Path(name).parts)


def _number(value: Any, audit: Audit, label: str, *, nonnegative: bool = False) -> float | None:
    try:
        result = finite(value, label)
        audit.require(not nonnegative or result >= 0, label + " is negative")
        return result
    except Exception as exc:
        audit.errors.append(str(exc))
        return None


def runtime_inventory(root: Path, audit: Audit) -> dict[str, Any]:
    expected = set(PHASES) | {f"attempt-{phase}.json" for phase in PHASES} | {"runtime"}
    try:
        audit.require({path.name for path in root.iterdir()} == expected,
                      "artifact root membership differs")
    except Exception as exc:
        audit.errors.append(f"root membership failed: {type(exc).__name__}: {exc}")
    return base.runtime_inventory(root, audit)


def validate_phase_resources(completion: dict[str, Any], phase: str, audit: Audit) -> None:
    limit = 100 if phase == "smoke" else 1800
    for field in ("scalar_update_seconds", "elapsed_seconds"):
        _number(completion.get(field), audit, phase + " " + field, nonnegative=True)
    elapsed = completion.get("elapsed_seconds")
    audit.require(type(elapsed) in (int, float) and math.isfinite(float(elapsed))
        and elapsed <= limit, phase + " exceeds its cooperative wall limit")
    audit.require(type(completion.get("peak_torch_gpu_bytes")) is int
        and 0 <= completion["peak_torch_gpu_bytes"] <= 4 * 1024**3
        and type(completion.get("shared_artifact_bytes_before_completion")) is int
        and 0 <= completion["shared_artifact_bytes_before_completion"] <= ARTIFACT_CAP,
        phase + " resource counters differ")


def phase_records(root: Path, phase: str, audit: Audit) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    directory = root / phase
    try:
        manifest = read_json(directory / "manifest.json")
        completion = read_json(directory / "completion.json")
        manifest_keys = ("schema", "phase", "frozen_commit", "source_hashes",
            "torch_version", "numpy_version", "python_version", "gpu", "real_k",
            "test_only_k", "horizons", "smoke_based_seconds_forecast",
            "timing_safety_factor", "cloud_spend_usd", "authorized_budget_usd",
            "shared_artifact_cap_bytes", "wall_limit_seconds", "runtime_directory",
            "runtime_bytes_count_in_cap")
        completion_keys = ("schema", "status", "phase", "source_hashes", "frozen_commit",
            "branches", "numerical_failures", "all_requested_endpoints_present",
            "completed_training_updates", "scalar_update_seconds", "elapsed_seconds",
            "peak_torch_gpu_bytes", "shared_artifact_bytes_before_completion", "artifacts")
        audit.require(type(manifest) is dict and tuple(manifest) == manifest_keys
            and manifest.get("schema") == "i16_manifest_v1" and manifest.get("phase") == phase,
            phase + " manifest differs")
        audit.require(type(completion) is dict and tuple(completion) == completion_keys
            and completion.get("schema") == "i16_completion_v1"
            and completion.get("status") == "complete" and completion.get("phase") == phase,
            phase + " completion differs")
        audit.require(manifest.get("frozen_commit") == completion.get("frozen_commit")
            and manifest.get("source_hashes") == completion.get("source_hashes"),
            phase + " source binding differs")
        audit.require(manifest.get("real_k") == [0.0, 0.5, 0.9]
            and manifest.get("test_only_k") == [1.0]
            and tuple(manifest.get("horizons", ())) == HORIZONS
            and manifest.get("timing_safety_factor") == 1.5
            and manifest.get("cloud_spend_usd") == 0
            and manifest.get("authorized_budget_usd") == 100
            and manifest.get("shared_artifact_cap_bytes") == ARTIFACT_CAP
            and manifest.get("wall_limit_seconds") == (100 if phase == "smoke" else 1800)
            and manifest.get("runtime_directory") == "runtime"
            and manifest.get("runtime_bytes_count_in_cap") is True,
            phase + " protocol declaration differs")
        count = 6 if phase == "smoke" else 18
        audit.require(completion.get("branches") == count, phase + " branch count differs")
        validate_phase_resources(completion, phase, audit)
        records = completion.get("artifacts")
        audit.require(type(records) is list, phase + " artifact list missing")
        records = records if type(records) is list else []
        index = {row.get("name"): row for row in records if type(row) is dict}
        audit.require(len(index) == len(records), phase + " artifact names duplicate/malformed")
        audit.require({path.name for path in directory.iterdir()} == set(index) | {"completion.json"},
                      phase + " contains an unlisted, partial, or missing entry")
        for name, record in index.items():
            verify_record(directory, record, audit, f"{phase}/{name}")
        completion_path = directory / "completion.json"
        info = completion_path.lstat()
        audit.require(stat.S_ISREG(info.st_mode) and not completion_path.is_symlink(),
                      phase + " completion is not a regular nonsymlink file")
        if stat.S_ISREG(info.st_mode) and not completion_path.is_symlink():
            audit.phase_completion_sha256[phase] = sha256(completion_path)
            audit.hash_files += 1
            audit.hash_bytes += info.st_size
        failures = completion.get("numerical_failures")
        audit.require(type(failures) is int and 0 <= failures <= count
            and completion.get("all_requested_endpoints_present") is (failures == 0),
            phase + " failure aggregate differs")
        return manifest, completion, index
    except Exception as exc:
        audit.errors.append(f"{phase}: phase loading failed: {type(exc).__name__}: {exc}")
        return {}, {}, {}


def verify_sources(manifests: list[dict[str, Any]], audit: Audit) -> dict[str, Any]:
    commit = manifests[0].get("frozen_commit") if manifests else None
    sources = manifests[0].get("source_hashes") if manifests else None
    audit.require(type(commit) is str and re.fullmatch(r"[0-9a-f]{40}", commit or "") is not None,
                  "acquisition commit is not a full hash")
    audit.require(type(sources) is dict and tuple(sources) == ACQUISITION_SOURCES
        and all(type(value) is str and SHA_RE.fullmatch(value) for value in sources.values()),
        "acquisition source map differs from the exact 18-file closure")
    for manifest in manifests[1:]:
        audit.require(manifest.get("frozen_commit") == commit
            and manifest.get("source_hashes") == sources, "phase source maps differ")
    verified = []
    if type(commit) is str and type(sources) is dict:
        for relative, expected in sources.items():
            try:
                if not _safe_relative(relative):
                    raise ValueError("unsafe source path")
                path = REPO / relative
                audit.require(path.is_file() and sha256(path) == expected,
                              "worktree source differs: " + relative)
                frozen = subprocess.check_output(["git", "show", commit + ":" + relative], cwd=REPO)
                audit.require(hashlib.sha256(frozen).hexdigest() == expected,
                              "committed source differs: " + relative)
                verified.append(relative)
            except Exception as exc:
                audit.errors.append(f"source failed {relative}: {type(exc).__name__}: {exc}")
    return {"frozen_commit": commit, "source_hashes": sources, "verified_files": verified}


def validate_cross_phase_timing(phases: dict[str, tuple[dict[str, Any], dict[str, Any],
                                                        dict[str, Any]]], audit: Audit) -> None:
    smoke_seconds = phases["smoke"][1].get("scalar_update_seconds")
    recorded = phases["confirmation"][0].get("smoke_based_seconds_forecast")
    smoke_forecast = phases["smoke"][0].get("smoke_based_seconds_forecast")
    if type(smoke_seconds) not in (int, float) or not math.isfinite(float(smoke_seconds)):
        audit.errors.append("smoke scalar-update time is invalid for forecast")
        return
    expected = smoke_seconds / 60 * 34200 * 1.5 + 60
    audit.require(smoke_forecast is None and type(recorded) is float
        and recorded == expected and recorded <= 1800,
        "confirmation timing forecast differs from the fixed smoke formula")


def validate_phase_aggregate(completion: dict[str, Any], entries: Any,
                             warmup_updates: int, audit: Audit, label: str) -> None:
    rows = entries if type(entries) is list else []
    completed = sum(row.get("completed_updates", 0) for row in rows
                    if type(row) is dict and type(row.get("completed_updates")) is int)
    failures = sum(row.get("status") == "numerical_failure" for row in rows
                   if type(row) is dict)
    times = [row.get("scalar_update_seconds") for row in rows if type(row) is dict]
    audit.require(type(entries) is list
        and all(type(value) in (int, float) and math.isfinite(float(value)) and value >= 0
                for value in times)
        and completion.get("completed_training_updates") == completed + warmup_updates
        and completion.get("numerical_failures") == failures
        and completion.get("scalar_update_seconds") == sum(times),
        label + " completion aggregate differs")


def verify_analysis_sources(commit: str, audit: Audit) -> dict[str, Any]:
    audit.require(type(commit) is str and re.fullmatch(r"[0-9a-f]{40}", commit or "") is not None,
                  "analysis commit malformed")
    hashes = {}
    for path in (Path(__file__).resolve(), HERE / "test_analyse_scalar.py",
                 I15 / "analyse_history.py"):
        relative = str(path.relative_to(REPO))
        current = sha256(path)
        try:
            frozen = subprocess.check_output(["git", "show", commit + ":" + relative], cwd=REPO)
            audit.require(hashlib.sha256(frozen).hexdigest() == current,
                          "analysis source is not frozen: " + relative)
        except Exception as exc:
            audit.errors.append(f"analysis source failed {relative}: {type(exc).__name__}: {exc}")
        hashes[relative] = current
    return {"frozen_commit": commit, "source_hashes": hashes}


def validate_step(row: Any, expected_step: int, expected_k: float,
                  audit: Audit, label: str) -> dict[str, float]:
    keys = ("step", "relative_step", "path_statistics", "schema", "base", "lr", "k",
            "loss", "observer", "gradient_filter_applied",
            "native_projection_used_for_delivery", "gradient", "components",
            "temporal_control", "displacement", "decay", "algebra_residuals", "digests")
    audit.require(type(row) is dict and tuple(row) == keys, label + " step topology differs")
    if type(row) is not dict:
        return {}
    audit.require(row.get("step") == expected_step and row.get("relative_step") == expected_step - 100
        and row.get("schema") == "i16_scalar_step_v1" and row.get("base") == "sgdm"
        and row.get("lr") == .03 and row.get("k") == expected_k,
        label + " step identity differs")
    observer = row.get("observer")
    audit.require(type(observer) is dict and tuple(observer) == (
        "step_before", "step_after", "filtering_active", "used", "basis_rank")
        and observer.get("step_before") == expected_step - 1
        and observer.get("step_after") == expected_step
        and observer.get("filtering_active") is True and observer.get("used") is True
        and type(observer.get("basis_rank")) is int and 1 <= observer["basis_rank"] <= 32,
        label + " observer differs")
    audit.require(row.get("gradient_filter_applied") is False
        and row.get("native_projection_used_for_delivery") is False,
        label + " native-delivery declaration differs")
    _number(row.get("loss"), audit, label + " loss")
    component_names = ("raw_gradient", "post_mean", "native_projection_diagnostic",
                       "applied_delivery", "old_momentum_buffer",
                       "scaled_old_momentum_buffer", "new_momentum_buffer")
    components = row.get("components")
    audit.require(type(components) is dict and tuple(components) == component_names,
                  label + " component membership differs")
    if type(components) is dict:
        for name in component_names:
            validate_component(components.get(name), audit, label + " " + name)
    gradient = row.get("gradient")
    gradient_map = {"raw_norm": "raw_gradient", "post_mean_norm": "post_mean",
                    "native_projection_norm": "native_projection_diagnostic",
                    "applied_norm": "applied_delivery"}
    audit.require(type(gradient) is dict and tuple(gradient) == tuple(gradient_map),
                  label + " gradient topology differs")
    if type(gradient) is dict and type(components) is dict:
        for name, component in gradient_map.items():
            value = _number(gradient.get(name), audit, label + " " + name, nonnegative=True)
            if value is not None:
                audit.require(value == components[component].get("norm"),
                              label + " gradient/component norm differs")
    temporal = row.get("temporal_control")
    temporal_keys = ("old_history_scale", "raw_gradient_weight", "post_mean_weight",
                     "effective_old_history_coefficient", "delivery_literal", "history_literal",
                     "expected_buffer_norm", "empirical_ideal_data_step_defect_norm")
    audit.require(type(temporal) is dict and tuple(temporal) == temporal_keys
        and temporal.get("old_history_scale") == expected_k
        and temporal.get("raw_gradient_weight") == expected_k
        and temporal.get("post_mean_weight") == 1.0 - expected_k
        and temporal.get("effective_old_history_coefficient") == .9 * expected_k
        and temporal.get("delivery_literal") == ("post_mean" if expected_k == 0 else None)
        and temporal.get("history_literal") == ("zero" if expected_k == 0 else None),
        label + " scalar-control declaration differs")
    if type(temporal) is dict:
        for name in ("expected_buffer_norm", "empirical_ideal_data_step_defect_norm"):
            _number(temporal.get(name), audit, label + " " + name, nonnegative=True)
    displacement = row.get("displacement")
    displacement_keys = ("total_norm", "data_norm", "nominal_decay_norm",
        "total_current_basis_leakage", "data_current_basis_leakage",
        "raw_gradient_dot_data_delta", "post_mean_dot_data_delta",
        "applied_gradient_dot_data_delta")
    audit.require(type(displacement) is dict and tuple(displacement) == displacement_keys,
                  label + " displacement topology differs")
    signed = {}
    if type(displacement) is dict:
        norms = {name: _number(displacement.get(name), audit, label + " " + name,
                               nonnegative=True) for name in displacement_keys[:3]}
        validate_leakage(displacement.get("total_current_basis_leakage"),
            None if norms["total_norm"] is None else norms["total_norm"] ** 2,
            audit, label + " total displacement")
        validate_leakage(displacement.get("data_current_basis_leakage"),
            None if norms["data_norm"] is None else norms["data_norm"] ** 2,
            audit, label + " data displacement")
        for name in displacement_keys[5:]:
            value = _number(displacement.get(name), audit, label + " " + name)
            if value is not None:
                signed[name] = value
    path = row.get("path_statistics")
    audit.require(type(path) is dict and tuple(path) == (
        "data_step_squared_energy", "data_path_length_increment", "raw_gradient_dot_data_step"),
        label + " path-statistic topology differs")
    if type(path) is dict and type(displacement) is dict:
        data_norm = displacement.get("data_norm")
        audit.require(path.get("data_step_squared_energy") == data_norm * data_norm
            and path.get("data_path_length_increment") == data_norm
            and path.get("raw_gradient_dot_data_step")
                == displacement.get("raw_gradient_dot_data_delta"),
            label + " path-statistic identity differs")
    decay = row.get("decay")
    audit.require(type(decay) is dict and tuple(decay) == (
        "coefficient", "factor", "manual_before_optimizer_step",
        "manual_actual_norm", "manual_minus_nominal_norm")
        and decay.get("coefficient") == .01 and decay.get("factor") == .9997
        and decay.get("manual_before_optimizer_step") is True,
        label + " decay differs")
    if type(decay) is dict:
        _number(decay.get("manual_actual_norm"), audit, label + " manual decay", nonnegative=True)
        _number(decay.get("manual_minus_nominal_norm"), audit,
                label + " decay realization error", nonnegative=True)
    residuals = row.get("algebra_residuals")
    residual_names = ("post_mean_recurrence", "delivery_definition",
                      "scaled_old_buffer_definition", "momentum_buffer_recurrence")
    audit.require(type(residuals) is dict and tuple(residuals) == residual_names,
                  label + " residual membership differs")
    if type(residuals) is dict:
        for name in residual_names:
            value = residuals.get(name)
            audit.require(type(value) is dict and tuple(value) == (
                "norm", "homogeneous_error_bound"), label + " residual topology differs")
            if type(value) is dict:
                norm = _number(value.get("norm"), audit, label + " residual norm", nonnegative=True)
                bound = _number(value.get("homogeneous_error_bound"), audit,
                                label + " residual bound", nonnegative=True)
                if norm is not None:
                    audit.max_algebra_residual = max(audit.max_algebra_residual, norm)
                if norm is not None and bound is not None:
                    audit.require(norm <= bound, label + " residual exceeds saved bound")
    digests = row.get("digests")
    if expected_step == 101:
        digest_names = ("raw_gradient", "post_observer", "old_momentum_buffer",
                        "applied_gradient", "new_momentum_buffer")
        audit.require(type(digests) is dict and tuple(digests) == digest_names
            and all(type(value) is str and SHA_RE.fullmatch(value) for value in digests.values()),
            label + " first-step digest bundle differs")
    else:
        audit.require(digests is None, label + " unexpected digest bundle")
    return signed


def _selection(curve: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "minimum_validation_ce": min(
            curve, key=lambda row: (row["validation"]["clean_ce"], row["horizon"]))["horizon"],
        "maximum_validation_accuracy": min(
            curve, key=lambda row: (-row["validation"]["clean_accuracy"], row["horizon"]))["horizon"],
    }


def validate_branch(branch: Any, phase: str, audit: Audit,
                    label: str) -> dict[str, Any] | None:
    if type(branch) is not dict:
        audit.errors.append(label + " branch is not an object")
        return None
    keys = ("schema", "id", "seed", "target", "k", "k_label", "base", "lr",
        "parent_horizon", "parent_state_digest", "parent_evaluation_digest", "status",
        "requested_updates", "completed_updates", "last_completed_horizon", "curve",
        "scalar_update_seconds", "steps", "checkpoints", "failure", "first_step_digests",
        "selected_horizons")
    audit.require(tuple(branch) == keys, label + " branch topology differs")
    seed, target, k, k_label = (branch.get("seed"), branch.get("target"),
                                branch.get("k"), branch.get("k_label"))
    expected_seeds = (216,) if phase == "smoke" else SEEDS
    audit.require(branch.get("schema") == "i16_scalar_branch_v1"
        and seed in expected_seeds and target in TARGETS
        and type(k) is float and k in (0.0, .5, .9) and K_BY_LABEL.get(k_label) == k
        and branch.get("id") == f"s{seed}-sgdm-{target}-{k_label}"
        and branch.get("base") == "sgdm" and branch.get("lr") == .03
        and branch.get("parent_horizon") == 100
        and type(branch.get("parent_state_digest")) is str
        and SHA_RE.fullmatch(branch["parent_state_digest"])
        and type(branch.get("parent_evaluation_digest")) is str
        and SHA_RE.fullmatch(branch["parent_evaluation_digest"]),
        label + " identity differs")
    expected_updates = 10 if phase == "smoke" else 1900
    expected_horizons = (100, 110) if phase == "smoke" else HORIZONS
    audit.require(branch.get("requested_updates") == expected_updates,
                  label + " requested updates differ")
    status = branch.get("status")
    audit.require(status in ("complete", "numerical_failure"), label + " status differs")
    steps = branch.get("steps")
    audit.require(type(steps) is list and branch.get("completed_updates") == len(steps),
                  label + " completed-step count differs")
    steps = steps if type(steps) is list else []
    signed_steps = [validate_step(row, step, k, audit, f"{label} step {step}")
                    for step, row in enumerate(steps, start=101)]
    curve = branch.get("curve")
    audit.require(type(curve) is list and bool(curve), label + " curve missing")
    curve = curve if type(curve) is list else []
    actual_horizons = [row.get("horizon") for row in curve if type(row) is dict]
    audit.require(actual_horizons == list(expected_horizons[:len(actual_horizons)])
                  and bool(actual_horizons), label + " curve is not a scheduled prefix")
    for point in curve:
        if type(point) is dict:
            audit.require(tuple(point) == ("horizon", "train", "validation", "auxiliary"),
                          label + " evaluation topology differs")
            validate_evaluation({name: point.get(name)
                                 for name in ("train", "validation", "auxiliary")},
                                audit, label + f" h{point.get('horizon')}")
    if status == "complete":
        audit.require(len(steps) == expected_updates and actual_horizons == list(expected_horizons)
            and branch.get("last_completed_horizon") == expected_horizons[-1]
            and branch.get("failure") is None
            and branch.get("selected_horizons") == _selection(curve),
            label + " complete coverage differs")
    else:
        failure = branch.get("failure")
        attempted = failure.get("attempted_step") if type(failure) is dict else None
        completed = failure.get("completed_step") if type(failure) is dict else None
        valid_position = (attempted == completed + 1 if type(completed) is int else False) or (
            attempted == completed and completed in expected_horizons[1:]
            and completed not in actual_horizons)
        audit.require(type(failure) is dict and tuple(failure) == (
            "type", "message", "attempted_step", "completed_step", "last_valid_horizon",
            "state_artifact") and failure.get("type") == "NumericalFailure"
            and type(failure.get("message")) is str and 0 < len(failure["message"]) <= 2000
            and completed == 100 + len(steps) and valid_position
            and failure.get("last_valid_horizon") == actual_horizons[-1]
            and branch.get("last_completed_horizon") == completed
            and branch.get("selected_horizons") is None and len(steps) <= expected_updates,
            label + " numerical failure differs")
    _number(branch.get("scalar_update_seconds"), audit, label + " update time", nonnegative=True)
    first = branch.get("first_step_digests")
    audit.require(first == (steps[0].get("digests") if steps else None),
                  label + " first-step digest copy differs")
    return {"seed": seed, "target": target, "policy": k_label, "k": k, "status": status,
        "curve": curve, "steps": steps, "signed_steps": signed_steps,
        "parent_state_digest": branch.get("parent_state_digest"),
        "parent_evaluation_digest": branch.get("parent_evaluation_digest"),
        "first_step_digests": first, "checkpoints": branch.get("checkpoints", []),
        "failure": branch.get("failure")}


def _point(branch: dict[str, Any] | None, horizon: int) -> dict[str, Any] | None:
    if branch is None or branch.get("status") != "complete":
        return None
    return next((row for row in branch["curve"] if row["horizon"] == horizon), None)


def _utility(point: dict[str, Any], metric: str) -> float:
    return (-point["auxiliary"]["clean_ce"] if metric == "ce"
            else point["auxiliary"]["clean_accuracy"])


def _effect(values: dict[str, float | None]) -> dict[str, Any]:
    ordered = [values[str(seed)] for seed in SEEDS]
    available = all(value is not None for value in ordered)
    return {"per_seed": values, "all_three_seed_values": ordered, "available": available,
            "mean": sum(ordered) / 3 if available else None,
            "no_survivor_averaging": True}


def _best_scalar(branches: dict[tuple[int, str, str], dict[str, Any]], seed: int,
                 target: str, selector: str) -> tuple[dict[str, Any], float] | None:
    members = [branches.get((seed, target, label)) for label in SCALAR_LABELS]
    if any(row is None or row.get("status") != "complete" for row in members):
        return None
    candidates = []
    for label, branch in zip(SCALAR_LABELS, members):
        for point in branch["curve"]:
            value = point["validation"]["clean_ce" if selector == SELECTORS[0]
                                                else "clean_accuracy"]
            order = (value, point["horizon"], K_BY_LABEL[label]) if selector == SELECTORS[0] \
                else (-value, point["horizon"], K_BY_LABEL[label])
            candidates.append((order, point, K_BY_LABEL[label]))
    _, point, k = min(candidates, key=lambda row: row[0])
    return point, k


def _best_policy(branch: dict[str, Any] | None, selector: str) -> dict[str, Any] | None:
    if branch is None or branch.get("status") != "complete":
        return None
    horizon = _selection(branch["curve"])[selector]
    return _point(branch, horizon)


def selection_results(branches: dict[tuple[int, str, str], dict[str, Any]]) \
        -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    choices, primary, per_k = [], [], []
    selected: dict[tuple[int, str, str], tuple[dict[str, Any] | None, float | None,
                                               dict[str, Any] | None]] = {}
    for seed in SEEDS:
        for target in TARGETS:
            for selector in SELECTORS:
                scalar = _best_scalar(branches, seed, target, selector)
                scalar_point, k = scalar if scalar is not None else (None, None)
                spectral = _best_policy(branches.get((seed, target, SPECTRAL)), selector)
                selected[seed, target, selector] = scalar_point, k, spectral
                choices.append({"seed": seed, "target": target, "selector": selector,
                    "scalar": None if scalar_point is None else {
                        "k": k, "horizon": scalar_point["horizon"],
                        "validation_value": scalar_point["validation"][
                            "clean_ce" if selector == SELECTORS[0] else "clean_accuracy"],
                        "auxiliary_clean_ce": scalar_point["auxiliary"]["clean_ce"],
                        "auxiliary_clean_accuracy": scalar_point["auxiliary"]["clean_accuracy"]},
                    "spectral": None if spectral is None else {
                        "policy": SPECTRAL, "horizon": spectral["horizon"],
                        "validation_value": spectral["validation"][
                            "clean_ce" if selector == SELECTORS[0] else "clean_accuracy"],
                        "auxiliary_clean_ce": spectral["auxiliary"]["clean_ce"],
                        "auxiliary_clean_accuracy": spectral["auxiliary"]["clean_accuracy"]}})
    for target in TARGETS:
        for selector in SELECTORS:
            for metric in METRICS:
                values = {}
                for seed in SEEDS:
                    scalar, _, spectral = selected[seed, target, selector]
                    values[str(seed)] = (None if scalar is None or spectral is None else
                                         _utility(spectral, metric) - _utility(scalar, metric))
                primary.append({"target": target, "selector": selector,
                                "auxiliary_metric": metric, "effect": _effect(values)})
                for label in SCALAR_LABELS:
                    fixed = {}
                    for seed in SEEDS:
                        spectral = selected[seed, target, selector][2]
                        point = _best_policy(branches.get((seed, target, label)), selector)
                        fixed[str(seed)] = (None if spectral is None or point is None else
                                            _utility(spectral, metric) - _utility(point, metric))
                    per_k.append({"k": K_BY_LABEL[label], "k_label": label,
                        "target": target, "selector": selector, "auxiliary_metric": metric,
                        "effect": _effect(fixed)})
    return choices, primary, per_k


def endpoint_effects(branches: dict[tuple[int, str, str], dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for label in SCALAR_LABELS:
        for target in TARGETS:
            for metric in METRICS:
                values = {}
                for seed in SEEDS:
                    scalar = _point(branches.get((seed, target, label)), 2000)
                    spectral = _point(branches.get((seed, target, SPECTRAL)), 2000)
                    values[str(seed)] = (None if scalar is None or spectral is None else
                                         _utility(spectral, metric) - _utility(scalar, metric))
                rows.append({"k": K_BY_LABEL[label], "k_label": label, "target": target,
                             "metric": metric, "horizon": 2000, "effect": _effect(values)})
    return rows


METRIC_PATHS = {
    "train_clean_ce": ("train", "clean_ce"), "train_soft_ce": ("train", "soft_ce"),
    "train_fixed_ce": ("train", "fixed_ce"),
    "train_fixed_minus_soft_ce": ("train", "fixed_minus_soft_ce"),
    "train_clean_accuracy": ("train", "clean_accuracy"),
    "train_fixed_accuracy": ("train", "fixed_accuracy"),
    "train_confidence": ("train", "mean_max_probability"),
    "train_true_label_probability": ("train", "mean_true_label_probability"),
    "validation_clean_ce": ("validation", "clean_ce"),
    "validation_clean_accuracy": ("validation", "clean_accuracy"),
    "validation_confidence": ("validation", "mean_max_probability"),
    "validation_true_label_probability": ("validation", "mean_true_label_probability"),
    "auxiliary_clean_ce": ("auxiliary", "clean_ce"),
    "auxiliary_clean_accuracy": ("auxiliary", "clean_accuracy"),
    "auxiliary_confidence": ("auxiliary", "mean_max_probability"),
    "auxiliary_true_label_probability": ("auxiliary", "mean_true_label_probability"),
}


def trajectory_rows(branches: dict[tuple[int, str, str], dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for key in sorted(branches):
        branch = branches[key]
        start = next((point for point in branch["curve"] if point["horizon"] == 100), None)
        curve = []
        for point in branch["curve"]:
            values = {name: point[split][field]
                      for name, (split, field) in METRIC_PATHS.items()}
            values["absolute_progress_from_h100"] = None if start is None else {
                name + "_change": point[split][field] - start[split][field]
                for name, (split, field) in METRIC_PATHS.items()}
            curve.append({"horizon": point["horizon"], "values": values})
        rows.append({"seed": key[0], "target": key[1], "policy": key[2],
                     "status": branch["status"], "failure": branch.get("failure"),
                     "curve": curve})
    return rows


def scheduled_aggregates(branches: dict[tuple[int, str, str], dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for target in TARGETS:
        for policy in POLICIES:
            for horizon in HORIZONS:
                for metric, (split, field) in METRIC_PATHS.items():
                    values, changes = {}, {}
                    for seed in SEEDS:
                        point = _point(branches.get((seed, target, policy)), horizon)
                        start = _point(branches.get((seed, target, policy)), 100)
                        values[str(seed)] = None if point is None else point[split][field]
                        changes[str(seed)] = (None if point is None or start is None else
                                              point[split][field] - start[split][field])
                    available = all(value is not None for value in values.values())
                    rows.append({"target": target, "policy": policy, "horizon": horizon,
                        "metric": metric, "per_seed": values,
                        "equal_seed_mean": sum(values.values()) / 3 if available else None,
                        "per_seed_change_from_h100": changes,
                        "equal_seed_mean_change_from_h100":
                            sum(changes.values()) / 3 if available else None,
                        "available": available, "no_survivor_averaging": True})
    return rows


def geometry_results(branches: dict[tuple[int, str, str], dict[str, Any]]) \
        -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    windows = {"steps101_2000": (101, 2000), "steps1001_2000": (1001, 2000)}
    paths, components = [], []
    for target in TARGETS:
        for label in POLICIES:
            for window, (low, high) in windows.items():
                per_seed = {}
                for seed in SEEDS:
                    branch = branches.get((seed, target, label))
                    rows = ([row for row in branch["steps"] if low <= row["step"] <= high]
                            if branch is not None and branch.get("status") == "complete" else [])
                    if not rows:
                        per_seed[str(seed)] = None
                        continue
                    result = {}
                    for displacement in ("data", "total"):
                        norm_name = displacement + "_norm"
                        leakage_name = displacement + "_current_basis_leakage"
                        energy = sum(row["displacement"][norm_name] ** 2 for row in rows)
                        outside = []
                        for row in rows:
                            leak = row["displacement"][leakage_name]
                            if leak.get("reason") in (None, "zero_displacement"):
                                outside.append(leak["outside_squared_norm"])
                        result[displacement] = {
                            "squared_energy_sum": energy,
                            "path_length_sum": sum(row["displacement"][norm_name] for row in rows),
                            "current_action_complement_energy_fraction":
                                (sum(outside) / energy if len(outside) == len(rows) and energy > 0
                                 else None)}
                    for prefix, source_name in (
                            ("raw_gradient_dot_data_step", "raw_gradient_dot_data_delta"),
                            ("post_mean_dot_data_step", "post_mean_dot_data_delta"),
                            ("applied_gradient_dot_data_step",
                             "applied_gradient_dot_data_delta")):
                        values = [row["displacement"].get(source_name) for row in rows]
                        valid = all(type(value) in (int, float) and math.isfinite(value)
                                    for value in values)
                        result["sum_" + prefix] = sum(values) if valid else None
                        result["mean_" + prefix] = (sum(values) / len(values) if valid else None)
                    per_seed[str(seed)] = result
                available = all(value is not None for value in per_seed.values())
                paths.append({"target": target, "policy": label,
                    "k": K_BY_LABEL.get(label),
                    "k_label": label if label in SCALAR_LABELS else None,
                    "window": window, "per_seed": per_seed,
                    "equal_seed_mean": (None if not available else {
                        name: ({
                            field: (sum(value[name][field] for value in per_seed.values()) / 3
                                    if all(value[name][field] is not None
                                           for value in per_seed.values()) else None)
                            for field in per_seed[str(SEEDS[0])][name]}
                            if type(per_seed[str(SEEDS[0])][name]) is dict else
                            (sum(value[name] for value in per_seed.values()) / 3
                             if all(value[name] is not None for value in per_seed.values())
                             else None))
                        for name in per_seed[str(SEEDS[0])]}),
                    "available": available,
                    "aggregation": "within branch, then equal seeds"})
                if label not in NEW_LABELS:
                    continue
                for component in ("raw_gradient", "post_mean", "native_projection_diagnostic",
                                  "applied_delivery", "old_momentum_buffer",
                                  "scaled_old_momentum_buffer", "new_momentum_buffer"):
                    values = {}
                    for seed in SEEDS:
                        branch = branches.get((seed, target, label))
                        rows = ([row for row in branch["steps"] if low <= row["step"] <= high]
                                if branch is not None and branch.get("status") == "complete" else [])
                        energy = sum(row["components"][component]["squared_energy"] for row in rows)
                        valid = [row["components"][component]["current_basis_leakage"]
                                 for row in rows]
                        ratio = (sum(row["outside_squared_norm"] for row in valid) / energy
                                 if rows and energy > 0 and all(row.get("reason") in
                                     (None, "zero_displacement") for row in valid) else None)
                        values[str(seed)] = None if not rows else {
                            "mean_squared_energy": energy / len(rows),
                            "current_action_complement_energy_fraction": ratio}
                    okay = all(value is not None for value in values.values())
                    ratios = okay and all(value["current_action_complement_energy_fraction"]
                                            is not None for value in values.values())
                    components.append({"target": target, "k": K_BY_LABEL[label],
                        "k_label": label, "window": window, "component": component,
                        "per_seed": values,
                        "equal_seed_mean_squared_energy": (sum(value["mean_squared_energy"]
                            for value in values.values()) / 3 if okay else None),
                        "equal_seed_mean_of_branch_complement_fractions": (sum(
                            value["current_action_complement_energy_fraction"]
                            for value in values.values()) / 3 if ratios else None),
                        "available": okay, "aggregation": "within branch, then equal seeds"})
    return paths, components


def numerical_diagnostics(branches: dict[tuple[int, str, str], dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for target in TARGETS:
        for label in NEW_LABELS:
            per_seed = {}
            for seed in SEEDS:
                branch = branches.get((seed, target, label))
                if branch is None or branch.get("status") != "complete":
                    per_seed[str(seed)] = None
                    continue
                steps = branch["steps"]
                per_seed[str(seed)] = {
                    "maximum_algebra_residual_norm": max(
                        value["norm"] for row in steps
                        for value in row["algebra_residuals"].values()),
                    "maximum_residual_to_bound_ratio": max(
                        (value["norm"] / value["homogeneous_error_bound"]
                         if value["homogeneous_error_bound"] > 0 else 0.0)
                        for row in steps for value in row["algebra_residuals"].values()),
                    "maximum_empirical_ideal_data_step_defect_norm": max(
                        row["temporal_control"]["empirical_ideal_data_step_defect_norm"]
                        for row in steps),
                    "maximum_manual_decay_realization_error_norm": max(
                        row["decay"]["manual_minus_nominal_norm"] for row in steps),
                }
            rows.append({"target": target, "k": K_BY_LABEL[label], "k_label": label,
                         "per_seed": per_seed,
                         "all_seeds_available": all(value is not None for value in per_seed.values())})
    return rows


def validate_legacy_raw_steps(rows: Any, audit: Audit, label: str) -> None:
    """Validate the I14 fields used for k=1 geometry without replaying I14."""
    audit.require(type(rows) is list and len(rows) == 1900,
                  label + " legacy raw step count differs")
    for expected_step, row in enumerate(rows if type(rows) is list else [], start=101):
        audit.require(type(row) is dict and row.get("step") == expected_step
            and row.get("schema") == "i14_optimizer_step_v1" and row.get("base") == "sgdm"
            and row.get("lr") == .03 and row.get("policy") == "raw",
            label + f" legacy step {expected_step} identity differs")
        if type(row) is not dict:
            continue
        displacement = row.get("displacement")
        keys = ("total_norm", "data_norm", "nominal_decay_norm",
                "total_current_basis_leakage", "data_current_basis_leakage",
                "raw_gradient_dot_data_delta")
        audit.require(type(displacement) is dict and tuple(displacement) == keys,
                      label + f" legacy step {expected_step} displacement differs")
        if type(displacement) is dict:
            total = _number(displacement.get("total_norm"), audit, label + " total norm",
                            nonnegative=True)
            data = _number(displacement.get("data_norm"), audit, label + " data norm",
                           nonnegative=True)
            _number(displacement.get("nominal_decay_norm"), audit, label + " decay norm",
                    nonnegative=True)
            _number(displacement.get("raw_gradient_dot_data_delta"), audit,
                    label + " raw/data dot")
            validate_leakage(displacement.get("total_current_basis_leakage"),
                None if total is None else total * total, audit, label + " total leakage")
            validate_leakage(displacement.get("data_current_basis_leakage"),
                None if data is None else data * data, audit, label + " data leakage")


def _summary_values_match(values: Any, point: Any) -> bool:
    """Compare every metric retained by the accepted I15 scalar summary."""
    if type(values) is not dict or type(point) is not dict:
        return False
    expected = {
        "train_clean_ce": point["train"]["clean_ce"],
        "train_soft_ce": point["train"]["soft_ce"],
        "train_clean_accuracy": point["train"]["clean_accuracy"],
        "train_fixed_ce": point["train"]["fixed_ce"],
        "train_fixed_minus_soft_ce": point["train"]["fixed_minus_soft_ce"],
        "train_fixed_accuracy": point["train"]["fixed_accuracy"],
        "train_confidence": point["train"]["mean_max_probability"],
        "train_true_label_probability": point["train"]["mean_true_label_probability"],
        "validation_clean_ce": point["validation"]["clean_ce"],
        "validation_clean_accuracy": point["validation"]["clean_accuracy"],
        "validation_confidence": point["validation"]["mean_max_probability"],
        "validation_true_label_probability": point["validation"][
            "mean_true_label_probability"],
        "auxiliary_clean_ce": point["auxiliary"]["clean_ce"],
        "auxiliary_clean_accuracy": point["auxiliary"]["clean_accuracy"],
        "auxiliary_confidence": point["auxiliary"]["mean_max_probability"],
        "auxiliary_true_label_probability": point["auxiliary"][
            "mean_true_label_probability"],
    }
    return all(values.get(name) == value for name, value in expected.items())


def load_references(binding: Any, audit: Audit) -> dict[tuple[int, str, str], dict[str, Any]]:
    result = {}
    keys = ("schema", "i14", "i15", "raw_references", "spectral_references", "source_replayed")
    audit.require(type(binding) is dict and tuple(binding) == keys
        and binding.get("schema") == "i16_parent_and_reference_binding_v1"
        and binding.get("source_replayed") is False, "parent/reference binding differs")
    if type(binding) is not dict:
        return result
    i15 = binding.get("i15")
    expected_i15 = {"artifact_root": str(I15_ROOT),
        "acquisition_commit": "2bccbc6a883f1c4c11950965f9801d2c03d4350b",
        "audit_sha256": I15_AUDIT_SHA, "summary_sha256": I15_SUMMARY_SHA,
        "collection_sha256": I15_COLLECTION_SHA,
        "confirmation_completion_sha256": I15_COMPLETION_SHA}
    audit.require(i15 == expected_i15, "I15 evidence pin set differs")
    pinned = {
        "audit": (I15 / "analysis-001/audit.json", I15_AUDIT_SHA),
        "summary": (I15 / "analysis-001/summary.json", I15_SUMMARY_SHA),
        "collection": (I15 / "raw-results-001/collection.json", I15_COLLECTION_SHA),
        "completion": (I15_ROOT / "confirmation/completion.json", I15_COMPLETION_SHA),
    }
    loaded = {}
    for name, (path, expected) in pinned.items():
        audit.require(path.is_file() and not path.is_symlink() and sha256(path) == expected,
                      "I15 pinned evidence differs: " + name)
        if path.is_file() and not path.is_symlink() and sha256(path) == expected:
            loaded[name] = read_json(path)
    audit.require(loaded.get("audit", {}).get("schema") == "i15_history_analysis_audit_v1"
        and loaded.get("audit", {}).get("status") == "pass"
        and loaded.get("audit", {}).get("artifact_root") == str(I15_ROOT)
        and loaded.get("audit", {}).get("phase_completion_sha256", {}).get("confirmation")
            == I15_COMPLETION_SHA
        and loaded.get("summary", {}).get("schema") == "i15_history_analysis_summary_v1"
        and loaded.get("summary", {}).get("audit_status") == "pass"
        and loaded.get("summary", {}).get("artifact_root") == str(I15_ROOT)
        and loaded.get("summary", {}).get("counts", {}).get("new_confirmation_branches") == 18
        and loaded.get("summary", {}).get("counts", {}).get(
            "confirmation_numerical_failures") == 0
        and loaded.get("collection", {}).get("schema") == "i15_scalar_collection_v1"
        and loaded.get("collection", {}).get("status") == "complete"
        and loaded.get("collection", {}).get("source_root") == str(I15_ROOT)
        and loaded.get("collection", {}).get("analysis_audit", {}).get("sha256")
            == I15_AUDIT_SHA
        and loaded.get("collection", {}).get("phase_completion_sha256", {}).get(
            "confirmation") == I15_COMPLETION_SHA
        and loaded.get("completion", {}).get("schema") == "i15_completion_v1"
        and loaded.get("completion", {}).get("status") == "complete"
        and loaded.get("completion", {}).get("phase") == "confirmation"
        and loaded.get("completion", {}).get("frozen_commit")
            == "2bccbc6a883f1c4c11950965f9801d2c03d4350b"
        and loaded.get("completion", {}).get("branches") == 18
        and loaded.get("completion", {}).get("numerical_failures") == 0,
        "I15 accepted evidence status differs")
    old = base.load_i14_references(binding.get("i14"), audit)
    i14_parents = {(row.get("seed"), row.get("target")): row
                   for row in binding.get("i14", {}).get("parents", [])
                   if type(row) is dict}
    raws = binding.get("raw_references")
    audit.require(type(raws) is list and len(raws) == 6, "raw reference count differs")
    for row in raws if type(raws) is list else []:
        seed, target = row.get("seed"), row.get("target")
        reference = old.get((seed, target, "raw"))
        audit.require(type(row) is dict and tuple(row) == ("seed", "target", "curve")
            and seed in SEEDS and target in TARGETS and reference is not None
            and row.get("curve", {}).get("curve", [])[1:] == reference["curve"],
            "I14 raw reference binding differs")
        if reference is not None:
            validate_legacy_raw_steps(reference.get("steps"), audit,
                                      f"I14 raw {seed}/{target}")
            result[seed, target, "k1"] = {**reference, "policy": "k1"}
    trajectories = loaded.get("summary", {}).get("all_policy_trajectories", [])
    audit.require(type(trajectories) is list and len(trajectories) == 30,
                  "I15 accepted trajectory-summary membership differs")
    expected_spectral = {(row.get("seed"), row.get("target")): row for row in trajectories
        if type(row) is dict and row.get("policy") == SPECTRAL}
    audit.require(set(expected_spectral) == {(seed, target) for seed in SEEDS
                                              for target in TARGETS},
                  "I15 spectral summary membership differs")
    supplied = binding.get("spectral_references")
    audit.require(type(supplied) is list and len(supplied) == 6,
                  "spectral reference count differs")
    completion_records = loaded.get("completion", {}).get("artifacts", [])
    completion_index = {row.get("name"): row for row in completion_records
                        if type(row) is dict}
    audit.require(type(completion_records) is list
        and len(completion_index) == len(completion_records),
        "I15 completion artifact map differs")
    for row in supplied if type(supplied) is list else []:
        key = row.get("seed"), row.get("target")
        audit.require(type(row) is dict and tuple(row) == ("seed", "target", "curve_artifact",
            "checkpoint_records", "parent_state_digest", "parent_evaluation_digest")
            and key in expected_spectral, "I15 spectral reference identity differs")
        artifact = row.get("curve_artifact") if type(row) is dict else None
        audit.require(type(artifact) is dict and artifact == completion_index.get(
            artifact.get("name") if type(artifact) is dict else None),
            "I15 spectral curve membership differs")
        path = verify_record(I15_ROOT / "confirmation", artifact, audit,
                             "I15 spectral branch " + str(key))
        branch = read_json(path) if path is not None else {}
        audit.require(branch.get("schema") == "i15_history_branch_v1"
            and branch.get("seed") == key[0] and branch.get("target") == key[1]
            and branch.get("policy") == SPECTRAL and branch.get("base") == "sgdm"
            and branch.get("lr") == .03 and branch.get("status") == "complete"
            and branch.get("failure") is None and branch.get("completed_updates") == 1900
            and branch.get("parent_state_digest") == row.get("parent_state_digest")
            and branch.get("parent_evaluation_digest") == row.get("parent_evaluation_digest"),
            "I15 spectral branch binding differs")
        raw_start = _point(result.get(key + ("k1",)), 100)
        audit.require(branch.get("parent_state_digest")
            == i14_parents.get(key, {}).get("state_digest")
            and raw_start is not None
            and branch.get("parent_evaluation_digest") == tree_digest(raw_start)
            and branch.get("curve") and branch["curve"][0] == raw_start,
            "I15 spectral branch does not bind the accepted I14 h100 parent")
        validated = base.validate_branch(branch, "confirmation", audit,
                                         "accepted I15 spectral " + str(key))
        curve = (validated.get("curve", []) if type(validated) is dict else
                 branch.get("curve", []) if type(branch) is dict else [])
        audit.require([point.get("horizon") for point in curve if type(point) is dict]
            == list(HORIZONS), "spectral curve coverage differs")
        for point in curve:
            audit.require(type(point) is dict and tuple(point) == (
                "horizon", "train", "validation", "auxiliary"),
                "I15 spectral point topology differs")
            if type(point) is dict:
                validate_evaluation({name: point.get(name)
                                     for name in ("train", "validation", "auxiliary")},
                                    audit, f"spectral {key} h{point.get('horizon')}")
        checkpoint_rows = row.get("checkpoint_records") if type(row) is dict else None
        audit.require(type(checkpoint_rows) is list
            and [item.get("horizon") for item in checkpoint_rows
                 if type(item) is dict] == list(HORIZONS[1:]),
                      "I15 spectral checkpoint count differs")
        direct_projection = []
        for checkpoint in branch.get("checkpoints", []) if type(branch) is dict else []:
            horizon = checkpoint.get("horizon") if type(checkpoint) is dict else None
            kind = "full_state" if horizon == 2000 else "model_state"
            direct_projection.append({"horizon": horizon, kind: checkpoint.get(kind)})
        audit.require(checkpoint_rows == direct_projection,
                      "I15 spectral checkpoint projection differs")
        for checkpoint in checkpoint_rows if type(checkpoint_rows) is list else []:
            horizon = checkpoint.get("horizon") if type(checkpoint) is dict else None
            kind = "full_state" if horizon == 2000 else "model_state"
            record = checkpoint.get(kind) if type(checkpoint) is dict else None
            audit.require(type(checkpoint) is dict and tuple(checkpoint) == ("horizon", kind)
                and horizon in HORIZONS[1:] and record == completion_index.get(
                    record.get("name") if type(record) is dict else None),
                "I15 spectral checkpoint binding differs")
            verify_record(I15_ROOT / "confirmation", record, audit,
                          f"I15 spectral checkpoint {key}/h{horizon}")
        summary = expected_spectral.get(key, {})
        summary_curve = summary.get("curve", []) if type(summary) is dict else []
        audit.require([item.get("horizon") for item in summary_curve] == list(HORIZONS)
            and summary.get("status") == "complete" and summary.get("failure") is None,
            "I15 spectral summary coverage differs")
        for summary_point, direct_point in zip(summary_curve, curve):
            audit.require(summary_point.get("horizon") == direct_point.get("horizon")
                          and _summary_values_match(summary_point.get("values"), direct_point),
                          "I15 summary/direct spectral curve differs")
        result[key + (SPECTRAL,)] = {"seed": key[0], "target": key[1],
            "policy": SPECTRAL, "status": "complete", "curve": curve,
            "steps": validated.get("steps", []) if type(validated) is dict else [],
            "failure": None, "source": "accepted_i15_original_branch_and_summary"}
    audit.require(set(result) == {(seed, target, policy) for seed in SEEDS
        for target in TARGETS for policy in ("k1", SPECTRAL)},
        "accepted reference membership differs")
    return result


def _load_state(path: Path, expected: str | None, audit: Audit, label: str) -> str | None:
    return base._load_state(path, expected, audit, label)


def validate_checkpoints(branch: dict[str, Any], index: dict[str, Any], directory: Path,
                         audit: Audit, label: str) -> None:
    checkpoints = branch.get("checkpoints")
    if type(checkpoints) is not list:
        audit.errors.append(label + " checkpoint list missing")
        return
    expected = [point.get("horizon") for point in branch.get("curve", [])[1:]]
    audit.require([row.get("horizon") for row in checkpoints if type(row) is dict] == expected,
                  label + " checkpoint/curve coverage differs")
    for row in checkpoints:
        horizon = row.get("horizon") if type(row) is dict else None
        if branch.get("status") == "complete" and horizon == branch.get("last_completed_horizon"):
            audit.require(type(row) is dict and tuple(row) == (
                "horizon", "relative_horizon", "full_state", "full_state_digest"),
                label + " terminal checkpoint topology differs")
            record = row.get("full_state") if type(row) is dict else None
            audit.require(type(record) is dict and record == index.get(record.get("name")),
                          label + " terminal state membership differs")
            path = verify_record(directory, record, audit, label + " terminal state")
            if path is not None:
                _load_state(path, row.get("full_state_digest"), audit, label + " terminal state")
        else:
            audit.require(type(row) is dict and tuple(row) == (
                "horizon", "relative_horizon", "model_state"),
                label + " intermediate checkpoint topology differs")
            record = row.get("model_state") if type(row) is dict else None
            audit.require(type(record) is dict and record == index.get(record.get("name")),
                          label + " intermediate state membership differs")
            verify_record(directory, record, audit, label + " intermediate state")
        audit.require(type(horizon) is int and row.get("relative_horizon") == horizon - 100,
                      label + " relative checkpoint horizon differs")
    if branch.get("status") == "numerical_failure" and type(branch.get("failure")) is dict:
        record = branch["failure"].get("state_artifact")
        audit.require(type(record) is dict and record == index.get(record.get("name")),
                      label + " failure state membership differs")
        path = verify_record(directory, record, audit, label + " failure state")
        if path is not None:
            digest = _load_state(path, None, audit, label + " failure state")
            if digest is not None:
                audit.failure_state_tree_digests[label] = digest


def analyze(root: Path, analysis_commit: str, audit: Audit) -> dict[str, Any]:
    audit.artifact_root = str(root)
    runtime = runtime_inventory(root, audit)
    audit.runtime_inventory = runtime
    total_bytes = base.total_regular_bytes(root, audit)
    phases = {phase: phase_records(root, phase, audit) for phase in PHASES}
    validate_cross_phase_timing(phases, audit)
    acquisition = verify_sources([phases[phase][0] for phase in PHASES], audit)
    analysis = verify_analysis_sources(analysis_commit, audit)
    for phase in PHASES:
        path = root / f"attempt-{phase}.json"
        info = path.lstat()
        audit.require(stat.S_ISREG(info.st_mode) and not path.is_symlink(),
                      phase + " attempt is not a regular nonsymlink file")
        attempt = read_json(path)
        audit.require(type(attempt) is dict and tuple(attempt) == (
            "schema", "phase", "frozen_commit", "source_hashes", "pid", "restart")
            and attempt.get("schema") == "i16_attempt_v1" and attempt.get("phase") == phase
            and attempt.get("frozen_commit") == acquisition["frozen_commit"]
            and attempt.get("source_hashes") == acquisition["source_hashes"]
            and type(attempt.get("pid")) is int and attempt["pid"] > 0
            and attempt.get("restart") == "forbidden", phase + " attempt differs")
        audit.attempt_sha256[phase] = sha256(path)
        audit.hash_files += 1
        audit.hash_bytes += info.st_size
    confirmation_dir = root / "confirmation"
    confirmation_index = phases["confirmation"][2]
    parent_record = confirmation_index.get("parent-inputs.json")
    parent_path = verify_record(confirmation_dir, parent_record, audit, "parent inputs")
    binding = read_json(parent_path) if parent_path is not None else {}
    branches = load_references(binding, audit)
    seams_record = confirmation_index.get("parent-seams.json")
    seams_path = verify_record(confirmation_dir, seams_record, audit, "parent seams")
    seams = read_json(seams_path) if seams_path is not None else {}
    audit.require(type(seams) is dict and tuple(seams) == (
        "parents", "parents_loaded", "scientific_updates_before_admission")
        and seams.get("parents_loaded") == 6 and seams.get("scientific_updates_before_admission") == 0
        and type(seams.get("parents")) is list and len(seams["parents"]) == 6,
        "parent seam envelope differs")
    seam_index = {}
    for row in seams.get("parents", []) if type(seams) is dict else []:
        audit.require(type(row) is dict and tuple(row) == (
            "seed", "target", "state_digest", "evaluation_digest", "rng_neutral", "status")
            and row.get("seed") in SEEDS and row.get("target") in TARGETS
            and row.get("rng_neutral") is True and row.get("status") == "pass"
            and type(row.get("state_digest")) is str and SHA_RE.fullmatch(row["state_digest"])
            and type(row.get("evaluation_digest")) is str and SHA_RE.fullmatch(row["evaluation_digest"]),
            "parent seam row differs")
        seam_index[row.get("seed"), row.get("target")] = row
    audit.require(set(seam_index) == {(seed, target) for seed in SEEDS for target in TARGETS},
                  "parent seam membership differs")
    i14_binding = binding.get("i14", {}) if type(binding) is dict else {}
    records = i14_binding.get("source_records", {}) if type(i14_binding) is dict else {}
    parents = {(row.get("seed"), row.get("target")): row
               for row in i14_binding.get("parents", []) if type(row) is dict}
    plan_bindings = []
    for seed in SEEDS:
        for name in (f"plan-s{seed}.json", f"corruption-s{seed}.json"):
            path = verify_record(confirmation_dir, confirmation_index.get(name), audit,
                                 "I16 " + name)
            old_name = f"plan-confirmation-s{seed}.json" if name.startswith("plan") else None
            if path is not None and old_name is not None and old_name in records:
                old_path = verify_record(base.I14_ROOT / "confirmation", records[old_name], audit,
                                         "I14 " + old_name)
                if old_path is not None:
                    audit.require(read_json(path) == read_json(old_path),
                                  f"seed {seed} plan differs from I14")
                    plan_bindings.append({"seed": seed, "i16_sha256": sha256(path),
                                          "i14_sha256": sha256(old_path),
                                          "exact_json_tree_equal": read_json(path) == read_json(old_path)})
            if path is not None and name.startswith("corruption"):
                value = read_json(path)
                audit.require(type(value) is dict and tuple(value) == (
                    "replaced_count", "incorrect_count", "train_count")
                    and value == i14_binding.get("corruption_counts", {}).get(str(seed)),
                    f"seed {seed} corruption count differs")
    for phase in PHASES:
        directory, index = root / phase, phases[phase][2]
        index_path = verify_record(directory, index.get("branches.json"), audit,
                                   phase + " branch index")
        if index_path is None:
            continue
        envelope = read_json(index_path)
        expected_keys = (("entries", "first_step_pair_checks", "synthetic_warmup_updates",
                          "new_scalar_updates", "k1_parity_scope") if phase == "smoke" else
                         ("entries", "first_step_pair_checks", "new_scalar_branches",
                          "reused_i14_raw_k1", "reused_i15_mean_projected"))
        audit.require(type(envelope) is dict and tuple(envelope) == expected_keys,
                      phase + " branch-index topology differs")
        audit.require((phase != "smoke" or (envelope.get("synthetic_warmup_updates") == 200
            and envelope.get("new_scalar_updates") == 60
            and envelope.get("k1_parity_scope") == "synthetic_unit_test_only"))
            and (phase != "confirmation" or (envelope.get("new_scalar_branches") == 18
            and envelope.get("reused_i14_raw_k1") == 6
            and envelope.get("reused_i15_mean_projected") == 6)),
            phase + " branch-index declaration differs")
        entries = envelope.get("entries")
        expected_members = ({(216, target, label) for target in TARGETS for label in NEW_LABELS}
            if phase == "smoke" else
            {(seed, target, label) for seed in SEEDS for target in TARGETS for label in NEW_LABELS})
        audit.require(type(entries) is list and len(entries) == len(expected_members),
                      phase + " branch-index count differs")
        observed, validated = set(), []
        for entry in entries if type(entries) is list else []:
            audit.require(type(entry) is dict and tuple(entry) == (
                "id", "seed", "target", "k", "k_label", "status", "completed_updates",
                "parent_state_digest", "parent_evaluation_digest", "first_step_digests",
                "scalar_update_seconds", "artifact"), phase + " index entry differs")
            key = entry.get("seed"), entry.get("target"), entry.get("k_label")
            observed.add(key)
            artifact = entry.get("artifact")
            audit.require(type(artifact) is dict and artifact == index.get(artifact.get("name")),
                          phase + " branch artifact membership differs")
            path = verify_record(directory, artifact, audit, phase + " branch " + str(key))
            if path is None:
                continue
            branch = read_json(path)
            audit.require(all(branch.get(name) == entry.get(name) for name in (
                "id", "seed", "target", "k", "k_label", "status", "completed_updates",
                "parent_state_digest", "parent_evaluation_digest", "first_step_digests",
                "scalar_update_seconds")), phase + " branch-index binding differs")
            row = validate_branch(branch, phase, audit, phase + "/" + str(key))
            if row is not None:
                validated.append(row)
                if phase == "confirmation":
                    audit.require(key not in branches, "duplicate confirmation branch")
                    branches[key] = row
                validate_checkpoints(branch, index, directory, audit, phase + "/" + str(key))
        audit.require(observed == expected_members, phase + " exact branch membership differs")
        completion = phases[phase][1]
        validate_phase_aggregate(completion, entries, 200 if phase == "smoke" else 0,
                                 audit, phase)
        pair_rows = []
        for seed, target in sorted({(row["seed"], row["target"]) for row in validated}):
            group = [row for row in validated if (row["seed"], row["target"]) == (seed, target)]
            audit.require(len(group) == 3 and {row["policy"] for row in group} == set(NEW_LABELS),
                          phase + " first-step membership differs")
            for name in ("parent_state_digest", "parent_evaluation_digest"):
                audit.require(len({row[name] for row in group}) == 1,
                              phase + " common parent differs: " + name)
            available = [row for row in group if type(row["first_step_digests"]) is dict]
            for name in ("raw_gradient", "post_observer", "old_momentum_buffer"):
                audit.require(len({row["first_step_digests"][name] for row in available}) <= 1,
                              phase + " first-step common digest differs: " + name)
            pair_rows.append({"seed": seed, "target": target, "available_first_steps": len(available),
                "status": "pass" if len(available) == 3 else "partial_numerical_evidence"})
            if phase == "confirmation":
                seam = seam_index.get((seed, target), {})
                parent = parents.get((seed, target), {})
                raw_start = _point(branches.get((seed, target, "k1")), 100)
                for row in group:
                    audit.require(row["parent_state_digest"] == seam.get("state_digest")
                        == parent.get("state_digest"), "parent full-state seam differs")
                    audit.require(row["parent_evaluation_digest"] == seam.get("evaluation_digest")
                        == tree_digest(raw_start), "parent evaluation seam differs")
                    audit.require(row["curve"] and row["curve"][0] == raw_start,
                                  "h100 curve seam differs")
        audit.require(envelope.get("first_step_pair_checks") == pair_rows,
                      phase + " saved first-step checks differ")
    expected = {(seed, target, policy) for seed in SEEDS for target in TARGETS
                for policy in POLICIES}
    audit.require(set(branches) == expected, "complete five-policy membership differs")
    choices, primary, per_k = selection_results(branches)
    path_geometry, component_geometry = geometry_results(branches)
    return {"schema": "i16_scalar_analysis_summary_v1", "artifact_root": str(root),
        "scope": [
            "I14 raw and I15 spectral curves are accepted hash-bound references; neither was rerun.",
            "Six used old parents and every new terminal/failure state were CPU tree-digested; intermediate model-only states were hash-checked only.",
            "Joint scalar selection requires all four k members; missing contributions are unavailable without survivor reselection.",
            "Current-action complement energies are descriptive geometry, not mediation fractions.",
        ],
        "counts": {"new_smoke_branches": phases["smoke"][1].get("branches"),
                   "new_confirmation_branches": phases["confirmation"][1].get("branches"),
                   "confirmation_numerical_failures": phases["confirmation"][1].get(
                       "numerical_failures")},
        "input_provenance": {"acquisition": acquisition, "analysis": analysis,
            "parent_reference_binding": binding, "plan_bindings": plan_bindings,
            "runtime_inventory": runtime, "artifact_root_regular_file_bytes": total_bytes,
            "artifact_cap_bytes": ARTIFACT_CAP},
        "primary_validation_selected_spectral_minus_scalar": primary,
        "validation_selected_choices": choices,
        "validation_selected_spectral_minus_each_k": per_k,
        "fixed_k_endpoint_spectral_minus_scalar": endpoint_effects(branches),
        "all_policy_trajectories": trajectory_rows(branches),
        "scheduled_metric_equal_seed_aggregates": scheduled_aggregates(branches),
        "path_and_displacement_geometry": path_geometry,
        "component_energy_and_action_complement": component_geometry,
        "numerical_diagnostics": numerical_diagnostics(branches),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--frozen-commit", required=True)
    args = parser.parse_args()
    audit = Audit()
    try:
        unresolved = args.root.absolute()
        root = args.root.resolve(strict=True)
        audit.require(not unresolved.is_symlink() and unresolved == root
            and root.parent == Path("/tmp/spectral-experiment-artifacts")
            and root.name.startswith("spectral-i16-001."),
            "artifact root must be the direct canonical I16 path")
    except Exception as exc:
        raise SystemExit(f"invalid artifact root: {exc}")
    output = args.output
    output.parent.resolve(strict=True)
    if output.parent.resolve() != HERE or not output.name.startswith("analysis-"):
        raise SystemExit("output must be a new analysis-* directory in iteration-016")
    try:
        output.resolve().relative_to(root)
        raise SystemExit("analysis output must not be inside the artifact root")
    except ValueError:
        pass
    output.mkdir(exist_ok=False)
    try:
        summary = analyze(root, args.frozen_commit, audit)
    except BaseException as exc:
        audit.errors.append(f"analysis aborted: {type(exc).__name__}: {exc}")
        summary = {"schema": "i16_scalar_analysis_summary_v1", "artifact_root": str(root),
                   "status": "unavailable_due_to_analysis_error"}
    summary["audit_status"] = "pass" if not audit.errors else "fail"
    audit_payload = {"schema": "i16_scalar_analysis_audit_v1",
        "status": "pass" if not audit.errors else "fail", "artifact_root": audit.artifact_root,
        "phase_completion_sha256": audit.phase_completion_sha256,
        "attempt_sha256": audit.attempt_sha256, "runtime_inventory": audit.runtime_inventory,
        "checks": audit.checks, "errors": audit.errors, "warnings": audit.warnings,
        "hash_files_verified": audit.hash_files, "hash_bytes_streamed": audit.hash_bytes,
        "complete_state_tree_digests_verified": audit.tree_digests,
        "maximum_saved_algebra_residual": audit.max_algebra_residual,
        "maximum_leakage_identity_error": audit.max_leakage_identity_error,
        "failure_state_tree_digests": audit.failure_state_tree_digests}
    for name, value in (("summary.json", summary), ("audit.json", audit_payload)):
        with (output / name).open("x", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, ensure_ascii=True, allow_nan=False)
            handle.write("\n")
    print(json.dumps({"status": audit_payload["status"], "checks": audit.checks,
        "errors": len(audit.errors), "hash_files": audit.hash_files,
        "tree_digests": audit.tree_digests}, allow_nan=False))
    return 0 if not audit.errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
