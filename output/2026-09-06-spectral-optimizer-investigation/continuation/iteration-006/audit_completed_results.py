#!/usr/bin/env python3
"""Independent CPU audit of the completed iteration-006 primary experiment.

This file is prepared prospectively.  It imports neither the experiment producer
nor ``summarize_results.py`` and performs no training or checkpoint evaluation.
The parent-owned checkpoint replay and the separately frozen bundle-60006
re-execution are deliberately outside this script.  Raw outcomes are not opened
unless the explicit ``--outcome-access-go`` gate is supplied.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import struct
import subprocess
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
RESULTS = HERE / "results"
SUMMARY = HERE / "summary.json"
PILOT = HERE / "pilot" / "execution.json"
DATA = Path("data/MNIST/raw")

SEEDS = (6, 7, 8)
NOISES = (0.0, 0.9)
ARMS = (
    "adamw", "current32", "lagged32", "lagged32_current_norm",
    "scalar_current32", "scalar_lagged32",
)
CHECKPOINTS = ("final", "min_val_ce", "max_val_accuracy", "warmup100")
WINDOWS = {"all": (101, 2000), "early": (101, 500), "late": (1501, 2000)}
PAIRS = tuple((later, earlier) for i, earlier in enumerate(ARMS) for later in ARMS[i + 1:])
PRIMARY_PAIRS = (("lagged32", "current32"), ("lagged32_current_norm", "current32"))
PRIMARY_METRIC = "test.max_val_accuracy.accuracy"
EXPECTED_CELLS = {(seed, noise, arm) for seed in SEEDS for noise in NOISES for arm in ARMS}
STEP_METRICS = (
    "raw_norm", "raw_squared_norm", "current_norm", "current_squared_norm",
    "lagged_norm", "lagged_squared_norm", "applied_norm", "applied_squared_norm",
    "current_raw_norm_ratio", "lagged_raw_norm_ratio", "applied_raw_norm_ratio",
    "current_energy_retention", "lagged_energy_retention",
    "current_minus_lagged_retention", "current_lagged_cosine", "raw_applied_cosine",
    "policy_scale", "norm_matching_relative_error", "direction_relative_error",
)
PHASE_METRICS = (
    "current_energy_retention", "lagged_energy_retention",
    "current_minus_lagged_retention", "current_lagged_cosine",
)
POLICY_KEYS = {
    "arm", "observing_step", "policy_active", "active_policy",
    "current_basis_rank", "lagged_basis_rank", "current_basis_missing",
    "lagged_basis_missing", "current_candidate_operator", "lagged_candidate_operator",
    "scale", "scale_direction", "target_norm", "delivery_operator",
}
UPDATE_KEYS = {
    "norm", "squared_norm", "raw_gradient_dot_update", "applied_gradient_dot_update",
    "raw_gradient_update_cosine", "applied_gradient_update_cosine", "null_reasons",
}
SOURCE_PREFIX = "output/2026-09-06-spectral-optimizer-investigation/continuation/iteration-006/"
SOURCE_PATHS = {SOURCE_PREFIX + name for name in (
    "delivery_order_harness.py", "policy_math.py", "test_policy.py", "test_harness.py",
    "artifact_store.py", "protocol.md", "result-schema.md", "design-intent.md",
    "analysis-plan.md", "summarize_results.py", "test_summary.py",
    "best-practices-check.md", "challenge/decision.md",
)} | {
    "output/2026-09-06-spectral-optimizer-investigation/continuation/iteration-003/neural_harness.py",
    "spectral_filter.py",
}
ENVIRONMENT_KEYS = {
    "python", "numpy", "torch", "cuda", "gpu", "cpu_threads",
    "deterministic_algorithms", "tf32", "cudnn_benchmark", "cublas_workspace",
    "foreach", "fused",
}
SCALAR_RTOL = 1e-11
SCALAR_ATOL = 1e-13


def require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "duplicate JSON key: " + key)
            result[key] = value
        return result

    def reject(value):
        raise ValueError("nonfinite JSON constant: " + value)

    with Path(path).open(encoding="utf-8") as stream:
        value = json.load(stream, object_pairs_hook=unique, parse_constant=reject)
    finite_tree(value)
    return value


def finite_tree(value, where="root"):
    if isinstance(value, dict):
        for key, child in value.items():
            require(type(key) is str, "non-string JSON key at " + where)
            finite_tree(child, where + "." + key)
    elif isinstance(value, list):
        for child in value:
            finite_tree(child, where + "[]")
    elif type(value) is float:
        require(math.isfinite(value), "nonfinite JSON value at " + where)


def number(value, where, low=None, high=None):
    require(type(value) in (int, float) and math.isfinite(value), "invalid number: " + where)
    require(low is None or value >= low, "number below range: " + where)
    require(high is None or value <= high, "number above range: " + where)
    return value


def integer(value, where, low=0):
    require(type(value) is int and value >= low, "invalid integer: " + where)
    return value


def close(actual, expected, where, *, rtol=SCALAR_RTOL, atol=SCALAR_ATOL):
    if actual is None or expected is None:
        require(actual is expected, "null mismatch: " + where)
        return
    number(actual, where)
    number(expected, where + " expected")
    require(math.isclose(actual, expected, rel_tol=rtol, abs_tol=atol),
            f"scalar mismatch: {where}: {actual!r} != {expected!r}")


def digest(value, where):
    require(type(value) is str and len(value) == 64
            and all(char in "0123456789abcdef" for char in value), "invalid SHA256: " + where)
    return value


def utc(value, where):
    require(type(value) is str, "invalid timestamp: " + where)
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    require(parsed.tzinfo is not None and parsed.utcoffset() is not None,
            "naive timestamp: " + where)
    return parsed.astimezone(timezone.utc)


def identity(value, expected_cells=EXPECTED_CELLS):
    key = (value["seed"], value["replacement_probability"], value["arm"])
    require(key in expected_cells, "unexpected cell identity")
    require(value["run_key"] == f"seed{key[0]}-noise{key[1]:g}-{key[2]}", "run-key mismatch")
    return key


def descriptive(values):
    require(values and all(value is not None for value in values), "incomplete descriptive values")
    return {
        "mean": statistics.fmean(values), "median": statistics.median(values),
        "minimum": min(values), "maximum": max(values),
        "sample_sd_descriptive": statistics.stdev(values) if len(values) > 1 else None,
    }


def complete_statistics(values):
    if len(values) != 3 or any(value is None for value in values):
        return dict.fromkeys(("mean", "median", "minimum", "maximum", "sample_sd_descriptive"))
    return descriptive(values)


def verify_binding(item, *, bulk_root=None):
    path = Path(item["path"])
    require(path.is_absolute() and path.is_file(), "missing bound artifact: " + str(path))
    if bulk_root is not None:
        require(path.resolve().parent == Path(bulk_root).resolve(), "bulk artifact escaped attempt root")
    require(path.stat().st_size == integer(item["size_bytes"], "artifact size"), "artifact size mismatch")
    require(sha(path) == digest(item["sha256"], "artifact"), "artifact hash mismatch: " + str(path))
    return path


def verify_sources(mapping, revision, started):
    require(set(mapping) == SOURCE_PATHS and len(mapping) == 15, "scientific-source membership mismatch")
    require(type(revision) is str and len(revision) == 40, "invalid source revision")
    git = lambda *args: subprocess.check_output(["git", "-C", str(ROOT), *args])
    require(utc(git("show", "-s", "--format=%cI", revision).decode().strip(), "commit")
            <= utc(started, "run start"), "source commit does not predate execution")
    checks = []
    for relative, expected in sorted(mapping.items()):
        digest(expected, "scientific source")
        path = (ROOT / relative).resolve()
        require(path.is_relative_to(ROOT.resolve()), "source path escaped repository")
        require(sha(path) == expected, "current source differs: " + relative)
        require(hashlib.sha256(git("show", revision + ":" + relative)).hexdigest() == expected,
                "committed source differs: " + relative)
        checks.append({"path": relative, "sha256": expected, "current_and_commit_match": True})
    return checks


def verify_environment(value):
    require(set(value) == ENVIRONMENT_KEYS, "environment key mismatch")
    require(value["cpu_threads"] == 1 and value["deterministic_algorithms"] is True,
            "determinism/thread settings differ")
    require(value["tf32"] is False and value["cudnn_benchmark"] is False,
            "TF32/cuDNN settings differ")
    require(value["foreach"] is False and value["fused"] is False,
            "AdamW implementation settings differ")
    require(value["cublas_workspace"] == ":4096:8", "CUBLAS setting differs")


def verify_evaluation(value, count):
    require(set(value) == {"cross_entropy", "accuracy", "count"}, "evaluation schema mismatch")
    require(value["count"] == count and type(value["count"]) is int, "evaluation count mismatch")
    number(value["cross_entropy"], "cross entropy", 0)
    number(value["accuracy"], "accuracy", 0, 1)
    close(value["accuracy"] * count, round(value["accuracy"] * count),
          "accuracy integer lattice", rtol=0, atol=1e-8)


def selector_metrics(run):
    rows = run["validation_trajectory"]
    require([row["step"] for row in rows] == list(range(0, 2001, 100)), "validation grid mismatch")
    for row in rows:
        verify_evaluation({key: row[key] for key in ("cross_entropy", "accuracy", "count")}, 5000)
    expected = {
        "min_val_ce": min(rows, key=lambda row: row["cross_entropy"])["step"],
        "max_val_accuracy": max(rows, key=lambda row: row["accuracy"])["step"],
        "final": 2000, "warmup100": 100,
    }
    require(run["checkpoint_steps"] == expected, "checkpoint selector mismatch")
    by_step = {row["step"]: row for row in rows}
    metrics = {}
    require(set(run["test"]) == set(CHECKPOINTS), "test checkpoint set mismatch")
    for checkpoint in CHECKPOINTS:
        outcome = run["test"][checkpoint]
        verify_evaluation(outcome, 10000)
        step = expected[checkpoint]
        metrics[f"checkpoint.{checkpoint}.step"] = step
        metrics[f"checkpoint.{checkpoint}.postwarmup_updates"] = max(step - 100, 0)
        metrics[f"checkpoint.{checkpoint}.at_or_before_warmup"] = int(step <= 100)
        for field in ("accuracy", "cross_entropy"):
            metrics[f"test.{checkpoint}.{field}"] = outcome[field]
            metrics[f"validation.{checkpoint}.{field}"] = by_step[step][field]
            if checkpoint != "warmup100":
                metrics[f"test.{checkpoint}.minus_warmup100.{field}"] = (
                    outcome[field] - run["test"]["warmup100"][field])
        for other in CHECKPOINTS:
            if expected[other] == step:
                require(run["test"][other] == outcome, "same-step test metrics differ")
    for name in ("final_training_clean", "final_training_noisy", "final_validation"):
        verify_evaluation(run[name], 5000)
        for field in ("accuracy", "cross_entropy"):
            metrics[f"learning.{name}.{field}"] = run[name][field]
    require(run["final_validation"] == {key: by_step[2000][key]
            for key in ("cross_entropy", "accuracy", "count")}, "final-validation identity mismatch")
    return metrics


def checked_scalar(record, key, *, nonnegative=False):
    reasons = record["null_reasons"]
    value = record[key]
    if value is None:
        require(type(reasons.get(key)) is str and reasons[key], "null lacks reason: " + key)
    else:
        number(value, key, 0 if nonnegative else None)
        require(key not in reasons, "finite scalar has null reason: " + key)
    return value


def validate_step(row, arm, rank, previous_rank):
    require(set(row) == {"step", "policy", "null_reasons", "total", "decay_subtracted", *STEP_METRICS},
            "step-record schema mismatch")
    require(set(row["null_reasons"]) == {key for key in STEP_METRICS if row[key] is None},
            "step null-mask mismatch")
    step = row["step"]
    policy = row["policy"]
    require(set(policy) == POLICY_KEYS and policy["arm"] == arm, "policy metadata mismatch")
    observing, active = arm != "adamw", arm != "adamw" and step > 100
    require(policy["observing_step"] == (step if observing else None), "observation count mismatch")
    require(policy["policy_active"] is active, "policy-active mismatch")
    expected_active = arm if active else ("identity_warmup" if observing else "identity_baseline")
    require(policy["active_policy"] == expected_active, "active-policy label mismatch")
    for label, expected_rank in (("current", rank), ("lagged", previous_rank)):
        if not observing:
            require(policy[label + "_basis_rank"] is None
                    and policy[label + "_basis_missing"] is None
                    and policy[label + "_candidate_operator"] is None,
                    "baseline invented candidate state")
        else:
            require(policy[label + "_basis_rank"] == expected_rank, "basis rank/history mismatch")
            require(policy[label + "_basis_missing"] is (expected_rank == 0), "basis missing/rank mismatch")
            expected_operator = None if not active else (
                "identity_missing_basis" if expected_rank == 0 else "native_basis")
            require(policy[label + "_candidate_operator"] == expected_operator,
                    "candidate operator mismatch")
    for label in ("raw", "current", "lagged", "applied"):
        norm = checked_scalar(row, label + "_norm", nonnegative=True)
        squared = checked_scalar(row, label + "_squared_norm", nonnegative=True)
        if norm is not None:
            close(norm, math.sqrt(squared), label + " norm/energy", atol=0)
    raw_norm = row["raw_norm"]
    for label in ("current", "lagged", "applied"):
        value = checked_scalar(row, label + "_raw_norm_ratio", nonnegative=True)
        expected = None if row[label + "_norm"] is None or raw_norm == 0 else row[label + "_norm"] / raw_norm
        close(value, expected, label + "/raw norm ratio")
    for label in ("current", "lagged"):
        value = checked_scalar(row, label + "_energy_retention", nonnegative=True)
        expected = None if row[label + "_squared_norm"] is None or row["raw_squared_norm"] == 0 \
            else row[label + "_squared_norm"] / row["raw_squared_norm"]
        close(value, expected, label + " retention")
    difference = checked_scalar(row, "current_minus_lagged_retention")
    expected = None if row["current_energy_retention"] is None else (
        row["current_energy_retention"] - row["lagged_energy_retention"])
    close(difference, expected, "retention difference")
    for key in ("current_lagged_cosine", "raw_applied_cosine"):
        value = checked_scalar(row, key)
        if value is not None:
            number(value, key, -1 - 1e-9, 1 + 1e-9)
    if not active:
        direction, target, operator = "raw", raw_norm, "identity"
    elif arm == "current32":
        direction, target, operator = "current", row["current_norm"], "native_current"
    elif arm == "lagged32":
        direction, target, operator = "lagged", row["lagged_norm"], "native_lagged"
    elif arm == "lagged32_current_norm":
        direction, target, operator = "lagged", row["current_norm"], "scaled_native_lagged"
    else:
        direction, target, operator = "raw", (
            row["current_norm"] if arm == "scalar_current32" else row["lagged_norm"]), "scalar_identity"
    require(policy["scale_direction"] == direction and policy["delivery_operator"] == operator,
            "delivery policy mismatch")
    close(policy["target_norm"], target, "target norm", atol=0)
    scaled = active and arm in ("lagged32_current_norm", "scalar_current32", "scalar_lagged32")
    scale_value = checked_scalar(row, "policy_scale", nonnegative=True)
    if scaled:
        expected_scale = 0.0 if target == 0 else target / row[direction + "_norm"]
        close(policy["scale"], expected_scale, "metadata scale", atol=0)
        close(scale_value, expected_scale, "record scale", atol=0)
        if operator == "scalar_identity":
            require(expected_scale <= 1.005, "scalar contraction bound exceeded")
    else:
        require(policy["scale"] is None and scale_value is None, "unscaled policy has scale")
    norm_error = checked_scalar(row, "norm_matching_relative_error", nonnegative=True)
    direction_error = checked_scalar(row, "direction_relative_error", nonnegative=True)
    if target == 0:
        require(row["applied_norm"] == 0 and norm_error == 0 and direction_error is None,
                "zero-target delivery mismatch")
    else:
        require(row["applied_norm"] > 0, "positive target underflowed")
        close(norm_error, abs(row["applied_norm"] - target) / target, "delivery norm error")
        require(norm_error <= 1e-6 and direction_error <= 1e-6, "delivery gate exceeded")
    unavailable = "baseline_no_observer" if not observing else "warmup_candidates_not_measured"
    expected_reasons = {}
    if not active:
        for label in ("current", "lagged"):
            for suffix in ("norm", "squared_norm", "raw_norm_ratio"):
                expected_reasons[label + "_" + suffix] = unavailable
            expected_reasons[label + "_energy_retention"] = unavailable
        expected_reasons["current_minus_lagged_retention"] = unavailable
        expected_reasons["current_lagged_cosine"] = unavailable
        if raw_norm == 0:
            expected_reasons["applied_raw_norm_ratio"] = "zero_raw_norm"
    elif raw_norm == 0:
        for label in ("current", "lagged", "applied"):
            expected_reasons[label + "_raw_norm_ratio"] = "zero_raw_norm"
        for label in ("current", "lagged"):
            expected_reasons[label + "_energy_retention"] = "zero_raw_norm"
        expected_reasons["current_minus_lagged_retention"] = "zero_raw_norm"
    if row["raw_applied_cosine"] is None:
        expected_reasons["raw_applied_cosine"] = "zero_vector_norm"
    if active and row["current_lagged_cosine"] is None:
        expected_reasons["current_lagged_cosine"] = "zero_vector_norm"
    if not scaled:
        expected_reasons["policy_scale"] = "unscaled_policy"
    if target == 0:
        expected_reasons["direction_relative_error"] = "zero_target_and_delivery"
    require(row["null_reasons"] == expected_reasons, "exact policy null reasons mismatch")
    for kind in ("total", "decay_subtracted"):
        update = row[kind]
        require(set(update) == UPDATE_KEYS, "update schema mismatch")
        require(set(update["null_reasons"]) == {
            key for key in ("raw_gradient_update_cosine", "applied_gradient_update_cosine")
            if update[key] is None}, "update null-mask mismatch")
        number(update["norm"], "update norm", 0)
        number(update["squared_norm"], "update squared norm", 0)
        close(update["norm"], math.sqrt(update["squared_norm"]), "update norm/energy", atol=0)
        for component in ("raw", "applied"):
            dot = update[component + "_gradient_dot_update"]
            require(set(dot) == {"value", "tolerance", "sign"}, "signed-dot schema mismatch")
            product = row[component + "_norm"] * update["norm"]
            expected_tolerance = 1e-6 * product + 1e-14
            close(dot["tolerance"], expected_tolerance, "sign tolerance", rtol=1e-12, atol=1e-24)
            expected_sign = 1 if dot["value"] > dot["tolerance"] else -1 if dot["value"] < -dot["tolerance"] else 0
            require(dot["sign"] == expected_sign and type(dot["sign"]) is int, "dot sign mismatch")
            cosine = update[component + "_gradient_update_cosine"]
            close(cosine, None if product == 0 else dot["value"] / product, "update cosine")
            if cosine is None:
                require(update["null_reasons"][component + "_gradient_update_cosine"] == "zero_vector_norm",
                        "update cosine null reason mismatch")


def aggregate_steps(rows, field, nested=None):
    values, null_steps, null_reasons = [], [], {}
    for row in rows:
        if nested is None:
            value = row[field]
            reason = row["null_reasons"].get(field)
        else:
            value = row[nested][field]
            reason = row[nested]["null_reasons"].get(field)
        if value is None:
            null_steps.append(row["step"])
            null_reasons[str(row["step"])] = reason
        else:
            number(value, "aggregation scalar")
            require(reason is None, "finite aggregation value has null reason")
            values.append(value)
    counts = {
        "scheduled_count": len(rows), "finite_count": len(values),
        "null_count": len(null_steps), "null_steps": null_steps,
        "null_reasons": null_reasons,
    }
    return (statistics.fmean(values) if values else None), counts


def summarize_run(run):
    metrics = selector_metrics(run)
    counts = {}
    for window, (low, high) in WINDOWS.items():
        rows = run["steps_raw"][low - 1:high]
        require([row["step"] for row in rows] == list(range(low, high + 1)), "window schedule mismatch")
        for field in STEP_METRICS:
            name = f"gradient.{window}.{field}"
            metrics[name], counts[name] = aggregate_steps(rows, field)
            if field in ("policy_scale", "norm_matching_relative_error", "direction_relative_error"):
                finite = [row[field] for row in rows if row[field] is not None]
                for label, operation in (("minimum", min), ("maximum", max)):
                    metrics[name + "." + label] = operation(finite) if finite else None
                    counts[name + "." + label] = counts[name]
        for kind in ("total", "decay_subtracted"):
            prefix = f"update.{window}.{kind}"
            for field in ("norm", "squared_norm"):
                metrics[prefix + "." + field], counts[prefix + "." + field] = aggregate_steps(rows, field, kind)
            for component in ("raw", "applied"):
                dot_field = component + "_gradient_dot_update"
                values = [row[kind][dot_field]["value"] for row in rows]
                name = prefix + "." + dot_field
                metrics[name] = statistics.fmean(values)
                counts[name] = {"scheduled_count": len(rows), "finite_count": len(rows), "null_count": 0,
                                "null_steps": [], "null_reasons": {}}
                frequency = prefix + "." + component + "_gradient_ascent_frequency"
                metrics[frequency] = statistics.fmean(
                    int(row[kind][dot_field]["sign"] == 1) for row in rows)
                counts[frequency] = counts[name]
                cosine = component + "_gradient_update_cosine"
                metrics[prefix + "." + cosine], counts[prefix + "." + cosine] = aggregate_steps(rows, cosine, kind)
    active_rows = run["steps_raw"][100:]
    for phase, scheduled in (("scheduled_repair", True), ("other_steps", False)):
        rows = [row for row in active_rows if (row["step"] % 100 == 0) is scheduled]
        require(len(rows) == (19 if scheduled else 1881), "phase count mismatch")
        for field in PHASE_METRICS:
            name = f"phase.{phase}.{field}"
            metrics[name], counts[name] = aggregate_steps(rows, field)
    require(len(metrics) == 171, "unexpected seed metric count")
    return {
        "seed": run["seed"], "replacement_probability": run["replacement_probability"],
        "arm": run["arm"], "run_key": run["run_key"],
        "realized_replacement_fraction": run["realized_replacement_fraction"],
        "realized_incorrect_fraction": run["realized_incorrect_fraction"],
        "metrics": metrics, "counts": counts,
    }


def aggregate_groups(seed_summaries):
    indexed = {(row["seed"], row["replacement_probability"], row["arm"]): row
               for row in seed_summaries}
    require(set(indexed) == EXPECTED_CELLS and len(indexed) == len(seed_summaries) == 36,
            "seed-summary cell set mismatch")
    metric_names = set(seed_summaries[0]["metrics"])
    require(PRIMARY_METRIC in metric_names and all(set(row["metrics"]) == metric_names for row in seed_summaries),
            "seed metric schema mismatch")
    arm_groups, contrasts = [], []
    for noise in NOISES:
        for arm in ARMS:
            for metric in sorted(metric_names):
                entries = [{"seed": seed, "value": indexed[seed, noise, arm]["metrics"][metric]}
                           for seed in SEEDS]
                values = [entry["value"] for entry in entries]
                unavailable = [entry["seed"] for entry in entries if entry["value"] is None]
                arm_groups.append({
                    "replacement_probability": noise, "arm": arm, "metric": metric,
                    "seed_values": entries, "unavailable_seeds": unavailable,
                    "all_three_seeds_available": not unavailable, **complete_statistics(values),
                })
        for treatment, control in PAIRS:
            for metric in sorted(metric_names):
                entries, values, unavailable = [], [], []
                for seed in SEEDS:
                    left, right = indexed[seed, noise, treatment], indexed[seed, noise, control]
                    a, b = left["metrics"][metric], right["metrics"][metric]
                    left_mask = left["counts"].get(metric, {}).get("null_steps", [])
                    right_mask = right["counts"].get(metric, {}).get("null_steps", [])
                    reason = "missing_seed_metric" if a is None or b is None else (
                        "different_null_step_masks" if left_mask != right_mask else None)
                    difference = None if reason else a - b
                    values.append(difference)
                    if reason:
                        unavailable.append(seed)
                    entries.append({
                        "seed": seed, "treatment_value": a, "control_value": b,
                        "difference": difference, "unavailable_reason": reason,
                        "treatment_null_steps": left_mask, "control_null_steps": right_mask,
                    })
                primary = metric == PRIMARY_METRIC and (treatment, control) in PRIMARY_PAIRS
                require(not primary or not unavailable, "primary comparison unavailable")
                contrasts.append({
                    "replacement_probability": noise, "treatment": treatment, "control": control,
                    "metric": metric, "primary": primary, "paired_differences": entries,
                    "unavailable_seeds": unavailable, "all_three_pairs_available": not unavailable,
                    **complete_statistics(values),
                })
    primary = [row for row in contrasts if row["primary"]]
    require(len(arm_groups) == 2052 and len(contrasts) == 5130 and len(primary) == 4,
            "aggregate group cardinality mismatch")
    return {"arm_groups": arm_groups, "paired_contrasts": contrasts, "primary": primary}


def compare_tree(actual, expected, where="summary", errors=None):
    errors = [] if errors is None else errors
    if type(actual) is not type(expected):
        errors.append(where + ": type mismatch")
    elif isinstance(actual, dict):
        if set(actual) != set(expected):
            errors.append(where + ": key mismatch")
        for key in sorted(set(actual) & set(expected)):
            compare_tree(actual[key], expected[key], where + "." + key, errors)
    elif isinstance(actual, list):
        if len(actual) != len(expected):
            errors.append(where + ": length mismatch")
        for index, (left, right) in enumerate(zip(actual, expected)):
            compare_tree(left, right, f"{where}[{index}]", errors)
    elif type(actual) is float:
        if not math.isclose(actual, expected, rel_tol=SCALAR_RTOL, abs_tol=SCALAR_ATOL):
            errors.append(where + f": {actual!r} != {expected!r}")
    elif actual != expected:
        errors.append(where + f": {actual!r} != {expected!r}")
    return errors


def independent_plan(seed):
    generator = lambda stream: np.random.default_rng(np.random.SeedSequence([20260906, 3, seed, stream]))
    permutation = generator(0).permutation(60000)
    return {
        "seed": seed,
        "initialization_seed": int(generator(3).integers(0, 2**32, dtype=np.uint32)),
        "train_indices": permutation[:5000], "validation_indices": permutation[5000:10000],
        "auxiliary_indices": permutation[10000:15000],
        "replacement_uniforms": generator(1).random(5000),
        "replacement_digits": generator(2).integers(0, 10, size=5000),
        "training_batches": generator(4).integers(0, 5000, size=(2000, 64)),
        "primary_probe_batches": generator(5).integers(0, 5000, size=(2000, 256)),
        "auxiliary_probe_batches": generator(6).integers(0, 5000, size=(2000, 256)),
    }


def independent_audit_plan():
    generator = lambda stream: np.random.default_rng(np.random.SeedSequence([20260906, 60006, stream]))
    permutation = generator(0).permutation(60000)
    return {
        "seed": 60006,
        "initialization_seed": int(generator(3).integers(0, 2**32, dtype=np.uint32)),
        "train_indices": permutation[:5000], "validation_indices": permutation[5000:10000],
        "auxiliary_indices": permutation[10000:15000],
        "replacement_uniforms": generator(1).random(5000),
        "replacement_digits": generator(2).integers(0, 10, size=5000),
        "training_batches": generator(4).integers(0, 5000, size=(2000, 64)),
        "primary_probe_batches": generator(5).integers(0, 5000, size=(2000, 256)),
        "auxiliary_probe_batches": generator(6).integers(0, 5000, size=(2000, 256)),
    }


def load_idx_labels(path):
    raw = Path(path).read_bytes()
    magic, count = struct.unpack(">II", raw[:8])
    require(magic == 2049 and len(raw) == 8 + count, "training-label IDX schema mismatch")
    return np.frombuffer(raw, dtype=np.uint8, offset=8).copy()


def verify_plan(path, seed, labels):
    expected = independent_plan(seed)
    with np.load(path, allow_pickle=False) as saved:
        require(set(saved.files) == set(expected), "plan member mismatch")
        for key, value in expected.items():
            require(np.array_equal(saved[key], value), "plan reconstruction mismatch: " + key)
    clean = labels[expected["train_indices"]]
    rates = {}
    for noise in NOISES:
        replaced = expected["replacement_uniforms"] < noise
        noisy = np.where(replaced, expected["replacement_digits"], clean)
        rates[noise] = (float(np.float32(replaced).mean()), float(np.float32(noisy != clean).mean()))
    return rates


def validate_run(run, expected_cells=EXPECTED_CELLS):
    key = identity(run, expected_cells)
    require(run["schema_version"] == 1 and run["steps"] == 2000 and run["instrumented"] is True,
            "run header mismatch")
    require(run["all_invariant_gates_passed"] is True and run["warmup_checks_passed"] is True,
            "run invariant flag failed")
    require(run["delivery_gate_steps"] == 2000 and run["measurement_state_checks"] == 22,
            "runtime gate count mismatch")
    number(run["realized_replacement_fraction"], "realized replacement", 0, 1)
    number(run["realized_incorrect_fraction"], "realized incorrect", 0, 1)
    require(run["realized_incorrect_fraction"] <= run["realized_replacement_fraction"],
            "incorrect fraction exceeds replacement fraction")
    if key[1] == 0:
        require(run["realized_replacement_fraction"] == run["realized_incorrect_fraction"] == 0
                and run["final_training_clean"] == run["final_training_noisy"],
                "clean condition is not clean")
    rows = run["steps_raw"]
    require(len(rows) == 2000 and [row["step"] for row in rows] == list(range(1, 2001)),
            "step-index sequence mismatch")
    ranks, repairs = run["estimation_rank_by_step"], run["repair_count_by_step"]
    require(len(ranks) == len(repairs) == len(run["trajectory_parameter_sha256"]) == 2000,
            "rank/repair/trajectory length mismatch")
    observing = key[2] != "adamw"
    for index, row in enumerate(rows):
        digest(run["trajectory_parameter_sha256"][index], "trajectory parameter")
        rank, repair = integer(ranks[index], "rank"), integer(repairs[index], "repair count")
        require(rank <= 32 and (observing or rank == repair == 0), "rank/baseline mismatch")
        require(index == 0 or repair >= repairs[index - 1], "repair counter decreased")
        validate_step(row, key[2], rank, ranks[index - 1] if index else 0)
    require(len(run["step_elapsed_seconds"]) == 2000
            and all(number(value, "step timing", 0) >= 0 for value in run["step_elapsed_seconds"]),
            "timing history mismatch")
    warmup = run["warmup_trajectory_hashes"]
    require(len(warmup) == 100 and [row["step"] for row in warmup] == list(range(1, 101)),
            "warmup witness sequence mismatch")
    for index, row in enumerate(warmup):
        require(row["parameters"] == run["trajectory_parameter_sha256"][index]
                and row["raw_gradient"] == row["applied_gradient"], "warmup hash identity mismatch")
        for field in ("parameters", "raw_gradient", "applied_gradient"):
            digest(row[field], "warmup witness")
    digest(run["warmup_core_sha256"], "warmup core")
    if observing:
        require(len(run["warmup_observer_hashes"]) == 100
                and run["warmup_observer_hashes"][-1] == run["warmup_observer_sha256"],
                "observer warmup witness mismatch")
        for value in run["warmup_observer_hashes"]:
            digest(value, "warmup observer")
        digest(run["warmup_observer_sha256"], "warmup observer final")
    else:
        require(run["warmup_observer_hashes"] == [] and run["warmup_observer_sha256"] is None,
                "baseline invented observer witness")
    require(set(run["checkpoint_sha256"]) == set(CHECKPOINTS), "checkpoint hash set mismatch")
    for name in CHECKPOINTS:
        digest(run["checkpoint_sha256"][name], "checkpoint state")
        for other in CHECKPOINTS:
            if run["checkpoint_steps"][name] == run["checkpoint_steps"][other]:
                require(run["checkpoint_sha256"][name] == run["checkpoint_sha256"][other],
                        "same-step checkpoint hashes differ")
    require(type(run["checkpoint_path"]) is str and Path(run["checkpoint_path"]).is_absolute(),
            "checkpoint path is not absolute")
    number(run["elapsed_seconds"], "run elapsed seconds", 0)
    return key


def verify_source_subset(mapping, expected_paths, revision, started):
    require(set(mapping) == set(expected_paths), "auxiliary source membership mismatch")
    git = lambda *args: subprocess.check_output(["git", "-C", str(ROOT), *args])
    require(utc(git("show", "-s", "--format=%cI", revision).decode().strip(), "auxiliary commit")
            <= utc(started, "auxiliary run start"), "auxiliary source commit does not predate run")
    checks = []
    for relative, expected in sorted(mapping.items()):
        path = (ROOT / relative).resolve()
        require(path.is_relative_to(ROOT.resolve()), "auxiliary source escaped repository")
        require(sha(path) == digest(expected, "auxiliary source"), "current auxiliary source differs")
        require(hashlib.sha256(git("show", revision + ":" + relative)).hexdigest() == expected,
                "committed auxiliary source differs")
        checks.append({"path": relative, "sha256": expected, "current_and_commit_match": True})
    return checks


def audit_fresh_bundle(primary_execution, report):
    directory = HERE / "audit-reexecution"
    execution_path = directory / "execution.json"
    execution = read_json(execution_path)
    expected_arms = ("current32", "lagged32", "lagged32_current_norm")
    expected_cells = {(60006, noise, arm) for noise in NOISES for arm in expected_arms}
    require(execution["schema_version"] == 1 and execution["mode"] == "audit_reexecution"
            and execution["audit_bundle"] == 60006 and execution["status"] == "complete",
            "fresh-bundle terminal gate mismatch")
    require(execution["completed_runs"] == 6 and execution["test_evaluations"] == 24,
            "fresh-bundle 6/24 completion mismatch")
    require(execution["all_gates_passed"] is True and execution["warmup_checks_passed"] is True
            and execution["official_test_opened"] is True, "fresh-bundle invariant gate failed")
    times = [utc(execution[name], "fresh " + name) for name in (
        "started_utc", "all_training_completed_utc", "test_first_opened_utc", "completed_utc")]
    require(times[0] <= times[1] < times[2] <= times[3], "fresh-bundle chronology mismatch")
    require(execution["environment"] == primary_execution["environment"]
            and execution["training_data_artifacts"] == primary_execution["training_data_artifacts"]
            and execution["test_data_artifacts"] == primary_execution["test_data_artifacts"],
            "fresh-bundle environment/data differs from primary")
    report["fresh_primary_source_checks"] = verify_sources(
        execution["source_sha256"], execution["repository_revision"], execution["started_utc"])
    audit_paths = {
        SOURCE_PREFIX + "audit_reexecution.py",
        SOURCE_PREFIX + "audit-reexecution-plan.md",
    }
    report["fresh_audit_source_checks"] = verify_source_subset(
        execution["audit_source_sha256"], audit_paths,
        execution["repository_revision"], execution["started_utc"])
    primary_binding = execution["primary_execution"]
    require(verify_binding(primary_binding).resolve() == (RESULTS / "execution.json").resolve(),
            "fresh-bundle primary-manifest path mismatch")
    require(primary_binding["sha256"] == sha(RESULTS / "execution.json")
            and execution["source_sha256"] == primary_execution["source_sha256"],
            "fresh-bundle primary/source binding mismatch")
    completed = [identity(row, expected_cells) for row in execution["completed_cells"]]
    require(len(completed) == len(set(completed)) == 6 and set(completed) == expected_cells,
            "fresh-bundle cell set mismatch")
    bulk_root = Path(execution["bulk_root"])
    require(bulk_root.is_absolute() and bulk_root.resolve().parent == Path("/tmp/spectral-experiment-artifacts").resolve(),
            "fresh-bundle bulk root mismatch")
    mount = execution["bulk_mount"]
    require(mount["target"] == "/private-artifacts/storage"
            and (mount["source"] == "/dev/RECONFIGURE_FOR_LOCAL_STORAGE" or mount.get("uuid") == "00000000-0000-4000-8000-000000000000"),
            "fresh-bundle mount mismatch")
    bindings, indexed = [], {}
    for family, expected_count in (("plans", 1), ("training_runs", 6), ("runs", 6), ("checkpoints", 6)):
        require(len(execution[family]) == expected_count, "fresh bulk family count mismatch: " + family)
        for item in execution[family]:
            verify_binding(item, bulk_root=bulk_root)
            bindings.append(item)
            if family != "plans":
                key = identity(item, expected_cells)
                require((family, key) not in indexed, "duplicate fresh artifact cell")
                indexed[family, key] = item
    for name in ("fresh_bundle_primary_contrasts", "run_log"):
        verify_binding(execution[name], bulk_root=bulk_root)
        bindings.append(execution[name])
    require(len(bindings) == 21 and len({item["path"] for item in bindings}) == 21,
            "fresh bundle requires 21 unique artifacts")
    require(sum(item["size_bytes"] for item in bindings) == execution["artifact_total_bytes"],
            "fresh-bundle byte total mismatch")
    plan_binding = execution["plans"][0]
    expected_plan = independent_audit_plan()
    with np.load(plan_binding["path"], allow_pickle=False) as saved:
        require(set(saved.files) == set(expected_plan), "fresh plan member mismatch")
        for key, value in expected_plan.items():
            require(np.array_equal(saved[key], value), "fresh plan reconstruction mismatch: " + key)
    labels = load_idx_labels(DATA / "train-labels-idx1-ubyte")
    clean = labels[expected_plan["train_indices"]]
    rates = {}
    for noise in NOISES:
        replaced = expected_plan["replacement_uniforms"] < noise
        noisy = np.where(replaced, expected_plan["replacement_digits"], clean)
        rates[noise] = (float(np.float32(replaced).mean()), float(np.float32(noisy != clean).mean()))
    runs, summaries, step_indices = {}, [], 0
    for key in sorted(expected_cells):
        final_binding, pretest_binding = indexed["runs", key], indexed["training_runs", key]
        run, pretest = read_json(final_binding["path"]), read_json(pretest_binding["path"])
        require(identity(run, expected_cells) == key and "test" not in pretest,
                "fresh final/pretest identity mismatch")
        require({name: value for name, value in run.items() if name != "test"} == pretest,
                "fresh pretest differs from final-minus-test")
        require(run["mode"] == "audit_reexecution" and run["audit_bundle"] == 60006,
                "fresh run provenance label mismatch")
        require(run["plan_sha256"] == plan_binding["sha256"], "fresh run-plan mismatch")
        require(Path(run["checkpoint_path"]).resolve() == Path(indexed["checkpoints", key]["path"]).resolve(),
                "fresh run-checkpoint mismatch")
        close(run["realized_replacement_fraction"], rates[key[1]][0], "fresh replacement", rtol=0, atol=3e-8)
        close(run["realized_incorrect_fraction"], rates[key[1]][1], "fresh incorrect", rtol=0, atol=3e-8)
        validate_run(run, expected_cells)
        summaries.append(summarize_run(run))
        runs[key] = run
        step_indices += 2000
    require(step_indices == 12000, "fresh bundle requires 12,000 ordered step indices")
    for noise in NOISES:
        reference = runs[60006, noise, "current32"]
        for arm in expected_arms[1:]:
            run = runs[60006, noise, arm]
            for name in ("warmup_trajectory_hashes", "warmup_core_sha256", "warmup_observer_hashes",
                         "warmup_observer_sha256", "plan_sha256", "realized_replacement_fraction",
                         "realized_incorrect_fraction"):
                require(run[name] == reference[name], "fresh paired warmup/plan mismatch: " + name)
            require(run["test"]["warmup100"] == reference["test"]["warmup100"]
                    and run["checkpoint_sha256"]["warmup100"] == reference["checkpoint_sha256"]["warmup100"],
                    "fresh warmup checkpoint/test mismatch")
    derived = []
    for noise in NOISES:
        current = runs[60006, noise, "current32"]
        current_metric = current["test"]["max_val_accuracy"]
        for treatment in expected_arms[1:]:
            treated = runs[60006, noise, treatment]
            treated_metric = treated["test"]["max_val_accuracy"]
            derived.append({
                "audit_bundle": 60006, "replacement_probability": noise,
                "comparison": treatment + "-current32", "checkpoint": "max_val_accuracy",
                "current32_selected_step": current["checkpoint_steps"]["max_val_accuracy"],
                "treatment_selected_step": treated["checkpoint_steps"]["max_val_accuracy"],
                "current32_test_accuracy": current_metric["accuracy"],
                "treatment_test_accuracy": treated_metric["accuracy"],
                "accuracy_difference": treated_metric["accuracy"] - current_metric["accuracy"],
            })
    stored = read_json(execution["fresh_bundle_primary_contrasts"]["path"])
    errors = compare_tree(stored, derived, "fresh contrasts")
    require(not errors, "stored fresh contrasts differ from independent derivation: " + "; ".join(errors[:5]))
    return {
        "status": "PASS", "execution_sha256": sha(execution_path),
        "source_binding_count": 17, "bulk_binding_count": 21,
        "raw_run_count": 6, "pretest_identity_count": 6,
        "ordered_step_indices": step_indices, "independent_primary_contrasts": derived,
        "independent_seed_summaries": summaries,
        "pooled_with_primary": False,
    }


def verify_external_checkpoint_replays(primary_execution_sha, fresh_execution_sha):
    verifier = HERE / "verify_checkpoints_cpu.py"
    test_source = HERE / "test_checkpoint_verifier.py"
    records = []
    for filename, expected_count, expected_execution in (
        ("checkpoint-replay.json", 360, primary_execution_sha),
        ("checkpoint-replay-audit.json", 60, fresh_execution_sha),
    ):
        path = HERE / filename
        value = read_json(path)
        require(value["status"] == "PASS" and value["execution_sha256"] == expected_execution,
                "checkpoint replay status/input mismatch")
        require(value["verifier_sha256"] == sha(verifier), "checkpoint verifier source changed")
        require(len(value["evaluations"]) == expected_count
                and value["exact_accuracy_evaluations"] == expected_count,
                "checkpoint replay coverage/accuracy mismatch")
        require(value["max_accuracy_error_examples"] == 0
                and value["max_ce_absolute_error"] <= value["tolerances"]["ce_absolute"],
                "checkpoint replay tolerance mismatch")
        require(all(row["passed_tolerance"] and row["accuracy_error_examples"] == 0
                    for row in value["evaluations"]), "checkpoint replay contains a failed evaluation")
        records.append({
            "path": str(path), "sha256": sha(path), "status": value["status"],
            "evaluations": expected_count, "exact_accuracy_evaluations": value["exact_accuracy_evaluations"],
            "max_ce_absolute_error": value["max_ce_absolute_error"],
            "max_accuracy_error_examples": value["max_accuracy_error_examples"],
        })
    return {
        "reviewed_read_only": True, "verifier_sha256": sha(verifier),
        "test_source_sha256": sha(test_source), "records": records,
        "total_evaluations": sum(row["evaluations"] for row in records),
    }


def verify_pilot(execution):
    path = verify_binding(execution["passing_pilot"])
    require(path.resolve() == PILOT.resolve(), "unexpected passing-pilot path")
    pilot = read_json(path)
    require(pilot["mode"] == "pilot" and pilot["status"] == "complete_passed"
            and pilot["completed_traces"] == 12, "pilot terminal gate mismatch")
    require(pilot["all_gates_passed"] is True and pilot["warmup_checks_passed"] is True,
            "pilot invariant failure")
    require(pilot["official_test_opened"] is False
            and pilot["validation_or_accuracy_computed"] is False, "pilot outcome-access violation")
    require(pilot["source_sha256"] == execution["source_sha256"]
            and pilot["environment"] == execution["environment"]
            and pilot["training_data_artifacts"] == execution["training_data_artifacts"],
            "pilot/full binding mismatch")
    verify_sources(pilot["source_sha256"], pilot["repository_revision"], pilot["started_utc"])
    require(utc(pilot["completed_utc"], "pilot completion") < utc(execution["started_utc"], "full start"),
            "pilot did not precede full execution")
    require(len(pilot["plans"]) == 1 and pilot["plans"][0]["seed"] == 9880, "pilot plan mismatch")
    plan_path = verify_binding(pilot["plans"][0], bulk_root=pilot["bulk_root"])
    reports_path = verify_binding(pilot["timing_and_invariants"], bulk_root=pilot["bulk_root"])
    reports = read_json(reports_path)
    require(len(reports) == 6 and {row["arm"] for row in reports} == set(ARMS), "pilot arms incomplete")
    for report in reports:
        require(report["trajectory_bitwise_identical"] is True
                and report["final_state_bitwise_identical"] is True
                and report["warmup_checks_passed"] is True, "pilot on/off gate failed")
        for name, enabled in (("uninstrumented", False), ("instrumented", True)):
            trace = report[name]
            require(trace["steps"] == 220 and trace["instrumented"] is enabled
                    and trace["arm"] == report["arm"], "pilot trace mismatch")
            require(trace["measurement_state_checks"] == (5 if enabled else 0)
                    and trace["delivery_gate_steps"] == 220, "pilot gate counts mismatch")
            require(not ({"steps_raw", "test", "validation_trajectory"} & set(trace)),
                    "pilot contains forbidden learning outcome")
        left, right = report["uninstrumented"], report["instrumented"]
        for key in ("trajectory_parameter_sha256", "warmup_trajectory_hashes",
                    "warmup_observer_hashes", "warmup_core_sha256", "warmup_observer_sha256",
                    "estimation_rank_by_step", "repair_count_by_step"):
            require(left[key] == right[key], "pilot on/off evidence mismatch: " + key)
    return {"execution": str(path), "plan": str(plan_path), "timing_and_invariants": str(reports_path)}


def audit(execution, saved_summary, report, expected_revision):
    require(execution["repository_revision"] == expected_revision, "unexpected frozen revision")
    require(execution["schema_version"] == 1 and execution["mode"] == "full"
            and execution["status"] == "complete", "terminal full-run gate closed")
    require(execution["completed_runs"] == 36 and execution["test_evaluations"] == 144,
            "36-run/144-test completion mismatch")
    require(execution["all_gates_passed"] is True and execution["warmup_checks_passed"] is True
            and execution["official_test_opened"] is True, "execution gates incomplete")
    times = [utc(execution[name], name) for name in (
        "started_utc", "all_training_completed_utc", "test_first_opened_utc", "completed_utc")]
    require(times[0] <= times[1] < times[2] <= times[3], "training/test chronology mismatch")
    verify_environment(execution["environment"])
    report["source_checks"] = verify_sources(
        execution["source_sha256"], execution["repository_revision"], execution["started_utc"])
    report["pilot_checks"] = verify_pilot(execution)
    for family, names in (("training_data_artifacts", {"train-images-idx3-ubyte", "train-labels-idx1-ubyte"}),
                          ("test_data_artifacts", {"t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte"})):
        require(len(execution[family]) == 2
                and {Path(item["path"]).name for item in execution[family]} == names,
                "data binding family mismatch")
        for item in execution[family]:
            verify_binding(item)
    cells = [identity(row) for row in execution["completed_cells"]]
    require(len(cells) == len(set(cells)) == 36 and set(cells) == EXPECTED_CELLS,
            "completed-cell identity set mismatch")
    bulk_root = Path(execution["bulk_root"])
    require(bulk_root.is_absolute() and bulk_root.resolve().parent == Path("/tmp/spectral-experiment-artifacts").resolve(),
            "bulk root mismatch")
    mount = execution["bulk_mount"]
    require(mount["target"] == "/private-artifacts/storage"
            and (mount["source"] == "/dev/RECONFIGURE_FOR_LOCAL_STORAGE" or mount.get("uuid") == "00000000-0000-4000-8000-000000000000"),
            "bulk mount mismatch")
    bindings = []
    indexed_bindings = {}
    for family, expected_count in (("plans", 3), ("training_runs", 36), ("runs", 36), ("checkpoints", 36)):
        require(len(execution[family]) == expected_count, "bulk family count mismatch: " + family)
        for item in execution[family]:
            path = verify_binding(item, bulk_root=bulk_root)
            bindings.append(item)
            if family != "plans":
                key = identity(item)
                require((family, key) not in indexed_bindings, "duplicate artifact cell")
                indexed_bindings[family, key] = item
    require(len(bindings) == 111 and len({item["path"] for item in bindings}) == 111,
            "111 unique bulk bindings required")
    require(sum(item["size_bytes"] for item in bindings) == execution["artifact_total_bytes"],
            "bulk-byte total mismatch")
    labels = load_idx_labels(DATA / "train-labels-idx1-ubyte")
    plans, rates = {}, {}
    for item in execution["plans"]:
        seed = item["seed"]
        require(seed in SEEDS and seed not in plans, "plan seed mismatch")
        plans[seed] = item
        rates[seed] = verify_plan(item["path"], seed, labels)
    runs, seed_summaries = {}, []
    raw_sources = []
    step_indices = 0
    for key in sorted(EXPECTED_CELLS):
        final_binding = indexed_bindings["runs", key]
        pretest_binding = indexed_bindings["training_runs", key]
        checkpoint_binding = indexed_bindings["checkpoints", key]
        run = read_json(final_binding["path"])
        pretest = read_json(pretest_binding["path"])
        require(identity(run) == key and "test" not in pretest,
                "final/pretest identity mismatch")
        require({name: value for name, value in run.items() if name != "test"} == pretest,
                "pretest differs from final-minus-test")
        require(run["plan_sha256"] == plans[key[0]]["sha256"], "run-plan binding mismatch")
        require(Path(run["checkpoint_path"]).resolve() == Path(checkpoint_binding["path"]).resolve(),
                "run-checkpoint binding mismatch")
        expected_rates = rates[key[0]][key[1]]
        close(run["realized_replacement_fraction"], expected_rates[0], "replacement fraction", rtol=0, atol=3e-8)
        close(run["realized_incorrect_fraction"], expected_rates[1], "incorrect fraction", rtol=0, atol=3e-8)
        validate_run(run)
        summary = summarize_run(run)
        seed_summaries.append(summary)
        runs[key] = run
        step_indices += len(run["steps_raw"])
        raw_sources.append({"run_key": run["run_key"], "path": final_binding["path"],
                            "sha256": final_binding["sha256"]})
    require(step_indices == 72000, "expected exactly 72,000 ordered step indices")
    for seed in SEEDS:
        for noise in NOISES:
            group = {arm: runs[seed, noise, arm] for arm in ARMS}
            baseline, observer = group["adamw"], group["current32"]
            for arm, run in group.items():
                require(run["warmup_trajectory_hashes"] == baseline["warmup_trajectory_hashes"]
                        and run["warmup_core_sha256"] == baseline["warmup_core_sha256"],
                        "paired warmup core/trajectory mismatch")
                require(run["plan_sha256"] == baseline["plan_sha256"]
                        and run["realized_replacement_fraction"] == baseline["realized_replacement_fraction"]
                        and run["realized_incorrect_fraction"] == baseline["realized_incorrect_fraction"],
                        "paired plan/corruption mismatch")
                require(run["test"]["warmup100"] == baseline["test"]["warmup100"]
                        and run["checkpoint_sha256"]["warmup100"] == baseline["checkpoint_sha256"]["warmup100"],
                        "paired warmup checkpoint/test mismatch")
                if arm != "adamw":
                    require(run["warmup_observer_hashes"] == observer["warmup_observer_hashes"]
                            and run["warmup_observer_sha256"] == observer["warmup_observer_sha256"],
                            "paired observing-state mismatch")
    groups = aggregate_groups(seed_summaries)
    expected_summary = {"seed_summaries": seed_summaries, **groups}
    require(saved_summary["execution_status"] == "complete"
            and saved_summary["repository_revision"] == expected_revision,
            "saved-summary provenance mismatch")
    require(saved_summary["execution_sha256"] == sha(RESULTS / "execution.json")
            and saved_summary["source_sha256"] == sha(HERE / "summarize_results.py"),
            "saved-summary input/source hash mismatch")
    require(saved_summary["scientific_source_sha256"] == execution["source_sha256"],
            "saved-summary scientific-source map mismatch")
    require(saved_summary["source_bindings_verified"] == 15
            and saved_summary["bulk_artifacts_verified"] == 111
            and saved_summary["bulk_bytes_verified"] == execution["artifact_total_bytes"],
            "saved-summary coverage counts mismatch")
    require(saved_summary["input_bindings"] == bindings, "saved-summary input-binding list mismatch")
    comparison_errors = []
    for key_name, expected in expected_summary.items():
        compare_tree(saved_summary[key_name], expected, key_name, comparison_errors)
    require(not comparison_errors, "saved summary differs from independent aggregation: "
            + "; ".join(comparison_errors[:5]))
    fresh = audit_fresh_bundle(execution, report)
    checkpoint_replays = verify_external_checkpoint_replays(
        sha(RESULTS / "execution.json"), fresh["execution_sha256"])
    primary_noisy = {row["treatment"]: row["mean"] for row in groups["primary"]
                     if row["replacement_probability"] == 0.9}
    fresh_noisy = {row["comparison"].removesuffix("-current32"): row["accuracy_difference"]
                   for row in fresh["independent_primary_contrasts"]
                   if row["replacement_probability"] == 0.9}
    require(all(primary_noisy[arm] * fresh_noisy[arm] < 0 for arm in primary_noisy),
            "expected noisy primary/fresh sign disagreement not present")
    report.update(
        status="PASS", source_binding_count=15, full_bulk_binding_count=111,
        data_binding_count=4, raw_run_count=36, pretest_identity_count=36,
        ordered_step_indices=step_indices, seed_metric_count_each=len(seed_summaries[0]["metrics"]),
        arm_group_count=len(groups["arm_groups"]), paired_contrast_count=len(groups["paired_contrasts"]),
        primary_group_count=len(groups["primary"]), all_summary_scalars_and_null_masks_match=True,
        raw_sources=raw_sources, independent_seed_summaries=seed_summaries,
        independent_arm_groups=groups["arm_groups"],
        independent_paired_contrasts=groups["paired_contrasts"],
        independent_primary=groups["primary"],
        checkpoint_evaluation_scope="Parent-owned CPU checkpoint replay; not duplicated here.",
        fresh_bundle_scope="Bundle 60006 is separate re-execution evidence and is not pooled here.",
        fresh_bundle_audit=fresh,
        external_checkpoint_replay_review=checkpoint_replays,
        harness_gaming_scan={
            "status": "NO_EXPLOIT_SIGNATURE_FOUND",
            "scope": "producer, policy, storage, summarizer, audit re-execution, four frozen test files, and terminal logs",
            "notes": "No early exit, skipped-test, fake-library, fixture-output read, always-true grader, or relaxed-timeout path was found.",
        },
        overall_disposition="HONEST-NEGATIVE",
        audit_exit_reason="beneficial-lagging-not-supported-and-noisy-direction-seed-unstable",
    )


def markdown(report, output_json):
    primary = report.get("independent_primary", [])
    fresh = report.get("fresh_bundle_audit", {}).get("independent_primary_contrasts", [])
    lines = [
        "# Results Audit — iteration 006", "",
        f"## Overall disposition: {report.get('overall_disposition', 'NOT-REACHED')}",
        f"## audit_exit_reason: {report.get('audit_exit_reason', 'audit-failed')}", "",
        f"Artifact audit verdict: **{report['status']}**.", "",
        "This is a CPU, raw-artifact audit. It imports neither the producer nor the frozen summarizer, performs no training, and does not duplicate the parent-owned checkpoint forward replay.", "",
        f"Verified {report.get('source_binding_count', 0)} scientific source bindings, {report.get('full_bulk_binding_count', 0)} full-run bulk bindings, {report.get('raw_run_count', 0)} final raw runs, {report.get('pretest_identity_count', 0)} pretest/final identities, and {report.get('ordered_step_indices', 0)} ordered step indices.", "",
        f"Independent aggregation covers {report.get('seed_metric_count_each', 0)} scalar metrics per cell, {report.get('arm_group_count', 0)} arm summaries, {report.get('paired_contrast_count', 0)} paired metric contrasts, and {report.get('primary_group_count', 0)} prespecified primary groups. Exact null-step masks are part of every paired geometry comparison.", "",
        "## Four prespecified primary groups", "",
        "| Noise | Contrast | Seed differences | Mean | Median | Min | Max | Sample SD |",
        "|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in primary:
        differences = ", ".join(f"{entry['seed']}:{entry['difference']:+.6f}"
                                for entry in row["paired_differences"])
        lines.append("| {noise:g} | {t}-{c} | {d} | {mean:+.6f} | {median:+.6f} | {minimum:+.6f} | {maximum:+.6f} | {sd:.6f} |".format(
            noise=row["replacement_probability"], t=row["treatment"], c=row["control"], d=differences,
            mean=row["mean"], median=row["median"], minimum=row["minimum"], maximum=row["maximum"],
            sd=row["sample_sd_descriptive"]))
    lines += [
        "", "Accuracy values and differences are fractions. These are three paired seed bundles per condition; no confidence intervals, p-values, equivalence claims, or population claims are made.", "",
        "## Re-execution summary", "",
        "The separately frozen bundle 60006 completed six fresh current/lagged cells and 24 test evaluations. Its source, plan, data, 21 bulk bindings, six pretest/final identities, 12,000 step rows, and stored four-contrast file were independently checked here. It is not pooled with the three primary seeds.", "",
        "| Noise | Fresh contrast | Accuracy difference |",
        "|---:|---|---:|",
    ]
    for row in fresh:
        lines.append(f"| {row['replacement_probability']:g} | {row['comparison']} | {row['accuracy_difference']:+.6f} |")
    lines += [
        "", "The fresh clean contrasts remain adverse. Both fresh noisy contrasts reverse the signs of their corresponding three-seed primary means. Thus the artifact arithmetic is supported, but a beneficial lagging claim is not: the noisy direction is seed-unstable and the clean evidence is adverse.", "",
        "## Findings", "",
        "### Finding 1 — primary experiment — artifact integrity and arithmetic", "",
        "- **Verdict**: SUPPORTED",
        "- **Severity**: Low",
        "- **Evidence**: all 36 raw runs, 36 immutable pretest records, 111 bound bulk artifacts, 72,000 ordered step rows, 171 seed-level metrics, and 5,130 paired metric contrasts agree with independent calculations. The terminal log records `FULL_PROCESS_EXIT=0 FULL_PROCESS_ELAPSED_SECONDS=624.57`; the separate CPU verifier reproduced all 360 primary and 60 fresh accuracy evaluations exactly, with maximum CE discrepancies below 1e-7.",
        "- **Would fixing this plausibly flip PASS/FAIL?**: No.",
        "- **Finding to hand back**: no raw-artifact or aggregation defect found.", "",
        "### Finding 2 — primary plus fresh bundle — beneficial lagging is not supported", "",
        "- **Verdict**: TRUE-NULL",
        "- **Severity**: High",
        "- **Evidence**: primary clean means are adverse for lagged (-0.002933) and restored lagged (-0.003300); primary noisy means are adverse (-0.034933 and -0.040367), while the separately frozen fresh noisy bundle is positive (+0.002100 and +0.024700). The bound fresh log ends with `audit_reexecution_complete` and `test_evaluations: 24`.",
        "- **Would fixing this plausibly flip PASS/FAIL?**: No identifiable implementation or analysis defect exists to fix; additional bundles could change the descriptive pattern but would answer a larger replication question.",
        "- **Finding to hand back**: the fixed study does not establish a robust learning benefit from lagged delivery; noisy effects are seed-unstable and clean effects are adverse.", "",
        "## Unresolved findings for the write-up", "",
        "The evidence is limited to one small MNIST recipe, three primary paired bundles plus one separately reported fresh bundle, reused validation/test data, and validation-accuracy-selected checkpoints. The fresh noisy sign reversals must be reported alongside the primary adverse means.", "",
        "## Limitation triage", "",
        "| Limitation | Disposition | If fixable now: what + cost | If future-work: resources a fix would need |",
        "|---|---|---|---|",
        "| Only three primary paired bundles | future-work | — | Prospectively freeze additional independent bundles; each six-cell current/lagged-only bundle costs roughly two GPU-minutes on the same RTX 3090. |",
        "| One small MNIST model/recipe | future-work | — | A separately designed study on a larger model and at least one non-MNIST dataset; hardware/time depend on that design. |",
        "| Reused official test set and adaptive research history | future-work | — | A genuinely untouched dataset or locked external evaluation not previously used in this research line. |",
        "| Checkpoint numerical replay is separate | fix-now-free | Parent-owned CPU verifier, already run; reference its independent replay record in the final synthesis. | — |",
        "| No uncertainty interval by prospective rule | future-work | — | More prospectively sampled bundles sufficient for a predeclared inferential analysis; do not retrofit a CI to three bundled seeds. |", "",
        "The primary artifact math is approved. Scientific `SUPPORTED` is not earned for a beneficial lagging claim; the appropriate outcome is an honest adverse/mixed result, not another tuning loop.", "",
        f"[Machine-readable audit](audit-results.json), SHA256 `{sha(output_json)}`.",
    ]
    if report.get("error"):
        lines += ["", "Preserved failure: " + report["error"]]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outcome-access-go", action="store_true")
    parser.add_argument("--expected-revision", required=True)
    args = parser.parse_args()
    if not args.outcome_access_go:
        parser.error("Parent outcome permission and --outcome-access-go are required")
    output_json, output_md = HERE / "audit-results.json", HERE / "audit-results.md"
    require(not output_json.exists() and not output_md.exists(), "refusing to overwrite audit outputs")
    report = {
        "status": "RUNNING", "started_utc": datetime.now(timezone.utc).isoformat(),
        "audit_source_sha256": sha(Path(__file__)), "cpu_only": True,
        "imports_producer_or_summarizer": False, "training_performed": False,
        "checkpoint_forward_replay_performed": False,
        "scalar_tolerance": {"relative": SCALAR_RTOL, "absolute": SCALAR_ATOL},
        "statistical_scope": "three paired seeds per condition; descriptive only",
    }
    try:
        execution = read_json(RESULTS / "execution.json")
        report["execution_sha256"] = sha(RESULTS / "execution.json")
        report["summary_sha256"] = sha(SUMMARY)
        audit(execution, read_json(SUMMARY), report, args.expected_revision)
    except Exception as error:
        report["status"] = "FAIL_PRESERVED"
        report["error"] = repr(error)
    report["completed_utc"] = datetime.now(timezone.utc).isoformat()
    report["elapsed_seconds"] = (
        datetime.fromisoformat(report["completed_utc"]) - datetime.fromisoformat(report["started_utc"])
    ).total_seconds()
    with output_json.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    with output_md.open("x", encoding="utf-8") as stream:
        stream.write(markdown(report, output_json))
    print(json.dumps({"status": report["status"], "output": str(output_json),
                      "sha256": sha(output_json), "elapsed_seconds": report["elapsed_seconds"]}))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
