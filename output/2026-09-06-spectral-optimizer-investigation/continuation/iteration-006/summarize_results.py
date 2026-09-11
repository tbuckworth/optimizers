#!/usr/bin/env python3
"""Independent stdlib-only iteration006 aggregation; never imports training.

Strict JSON rejects duplicate keys and nonfinite numbers, unlike default Python
JSON decoding (https://docs.python.org/3/library/json.html). Evidence is read-only;
output creation is exclusive. Only the exact completed frozen study is accepted.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SEEDS = (6, 7, 8)
NOISES = (0.0, 0.9)
ARMS = ("adamw", "current32", "lagged32", "lagged32_current_norm",
        "scalar_current32", "scalar_lagged32")
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
    "current_energy_retention", "lagged_energy_retention", "current_minus_lagged_retention",
    "current_lagged_cosine", "raw_applied_cosine", "policy_scale",
    "norm_matching_relative_error", "direction_relative_error")
PHASE_METRICS = ("current_energy_retention", "lagged_energy_retention",
                 "current_minus_lagged_retention", "current_lagged_cosine")
POLICY_KEYS = {"arm", "observing_step", "policy_active", "active_policy",
               "current_basis_rank", "lagged_basis_rank", "current_basis_missing",
               "lagged_basis_missing", "current_candidate_operator", "lagged_candidate_operator",
               "scale", "scale_direction", "target_norm", "delivery_operator"}
UPDATE_KEYS = {"norm", "squared_norm", "raw_gradient_dot_update", "applied_gradient_dot_update",
               "raw_gradient_update_cosine", "applied_gradient_update_cosine", "null_reasons"}
SOURCE_PREFIX = "output/2026-09-06-spectral-optimizer-investigation/continuation/iteration-006/"
SOURCE_PATHS = {SOURCE_PREFIX + name for name in (
    "delivery_order_harness.py", "policy_math.py", "test_policy.py", "test_harness.py",
    "artifact_store.py", "protocol.md", "result-schema.md", "design-intent.md", "analysis-plan.md",
    "summarize_results.py", "test_summary.py", "best-practices-check.md", "challenge/decision.md")}
SOURCE_PATHS |= {"output/2026-09-06-spectral-optimizer-investigation/continuation/iteration-003/neural_harness.py",
                 "spectral_filter.py"}
ENVIRONMENT_KEYS = {"python", "numpy", "torch", "cuda", "gpu", "cpu_threads",
                    "deterministic_algorithms", "tf32", "cudnn_benchmark", "cublas_workspace",
                    "foreach", "fused"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def number(value, where, *, minimum=None, maximum=None):
    require(type(value) in (int, float) and math.isfinite(value), f"Nonfinite/non-numeric {where}")
    require(minimum is None or value >= minimum, f"Below range {where}")
    require(maximum is None or value <= maximum, f"Above range {where}")
    return value


def integer(value, where, *, minimum=0):
    require(type(value) is int and value >= minimum, f"Invalid integer {where}")
    return value


def finite_tree(value, where="root"):
    if isinstance(value, dict):
        for key, item in value.items():
            require(type(key) is str, f"Nonstring JSON key {where}")
            finite_tree(item, where + "." + key)
    elif isinstance(value, list):
        for item in value:
            finite_tree(item, where + "[]")
    elif type(value) is float:
        require(math.isfinite(value), f"Nonfinite JSON value {where}")


def read_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "Duplicate JSON key: " + key)
            result[key] = value
        return result

    def reject_constant(value):
        raise ValueError("Nonfinite JSON constant: " + value)

    with Path(path).open(encoding="utf-8") as handle:
        result = json.load(handle, object_pairs_hook=unique, parse_constant=reject_constant)
    finite_tree(result)
    return result


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def digest_string(value, where):
    require(type(value) is str and len(value) == 64
            and all(c in "0123456789abcdef" for c in value), "Invalid SHA256 " + where)
    return value


def utc_time(value, where):
    require(type(value) is str, "Invalid timestamp " + where)
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    require(parsed.tzinfo is not None and parsed.utcoffset() is not None,
            "Naive timestamp " + where)
    return parsed.astimezone(timezone.utc)


def close(actual, expected, where, *, relative=1e-10, absolute=1e-14):
    number(actual, where)
    number(expected, where + " expected")
    require(math.isclose(actual, expected, rel_tol=relative, abs_tol=absolute),
            "Inconsistent " + where)


def descriptive(values):
    require(len(values) > 0, "Empty descriptive sample")
    for value in values:
        number(value, "descriptive sample")
    return {"mean": statistics.fmean(values), "median": statistics.median(values),
            "minimum": min(values), "maximum": max(values),
            "sample_sd_descriptive": statistics.stdev(values) if len(values) > 1 else None}


def complete_statistics(values):
    if len(values) != len(SEEDS) or any(value is None for value in values):
        return dict.fromkeys(("mean", "median", "minimum", "maximum", "sample_sd_descriptive"))
    return descriptive(values)


def step_aggregate(values, steps, reasons):
    require(len(values) == len(steps) == len(reasons) and len(set(steps)) == len(steps)
            and len(steps) > 0, "Invalid step aggregation schedule")
    present, nulls = [], {}
    for step, value, reason in zip(steps, values, reasons):
        integer(step, "aggregate step", minimum=1)
        if value is None:
            require(type(reason) is str and bool(reason), "Missing null reason")
            nulls[str(step)] = reason
        else:
            number(value, "step metric")
            require(reason is None, "Null reason attached to finite value")
            present.append(value)
    counts = {"scheduled_count": len(steps), "finite_count": len(present),
              "null_count": len(nulls), "null_steps": [step for step in steps if str(step) in nulls],
              "null_reasons": nulls}
    return (statistics.fmean(present) if present else None), counts


def write_exclusive(path, value):
    finite_tree(value)
    payload = json.dumps(value, indent=2, allow_nan=False) + "\n"
    with Path(path).open("x", encoding="utf-8") as handle:
        handle.write(payload)


def condition_key(result):
    seed = integer(result["seed"], "seed")
    noise = number(result["replacement_probability"], "replacement_probability")
    arm = result["arm"]
    key = (seed, noise, arm)
    require(key in EXPECTED_CELLS, "Unexpected experimental condition")
    return key


def run_key(key):
    seed, noise, arm = key
    return f"seed{seed}-noise{noise:g}-{arm}"


def validate_evaluation(evaluation, count):
    require(integer(evaluation["count"], "evaluation count") == count, "Incomplete evaluation")
    number(evaluation["accuracy"], "accuracy", minimum=0, maximum=1)
    number(evaluation["cross_entropy"], "cross entropy", minimum=0)
    close(evaluation["accuracy"] * count, round(evaluation["accuracy"] * count),
          "accuracy/count lattice", relative=0, absolute=1e-8)


def validate_selectors(result):
    trajectory = result["validation_trajectory"]
    require([row["step"] for row in trajectory] == list(range(0, 2001, 100)),
            "Unexpected validation schedule")
    for row in trajectory:
        integer(row["step"], "validation step")
        validate_evaluation(row, 5000)
    expected = {"min_val_ce": min(trajectory, key=lambda row: row["cross_entropy"])["step"],
                "max_val_accuracy": max(trajectory, key=lambda row: row["accuracy"])["step"],
                "final": 2000, "warmup100": 100}
    require(result["checkpoint_steps"] == expected,
            "Checkpoint violates earliest strict independent selector")
    return {row["step"]: row for row in trajectory}


def checked_scalar(record, key, expected=None, *, null_reason=None, nonnegative=False):
    value = record[key]
    reasons = record["null_reasons"]
    if null_reason is not None:
        require(value is None and reasons.get(key) == null_reason, "Invalid null semantics " + key)
        return
    number(value, key, minimum=0 if nonnegative else None)
    require(key not in reasons, "Spurious null reason " + key)
    if expected is not None:
        close(value, expected, key, relative=1e-10, absolute=0)


def validate_step(row, arm, current_rank, previous_rank):
    step = integer(row["step"], "step", minimum=1)
    require(set(row) == {"step", "policy", "null_reasons", "total", "decay_subtracted", *STEP_METRICS},
            "Unexpected flat step schema")
    require(type(row["null_reasons"]) is dict and set(row["null_reasons"]) ==
            {key for key in STEP_METRICS if row[key] is None}, "Invalid scalar null-reason mask")
    policy = row["policy"]
    require(set(policy) == POLICY_KEYS and policy["arm"] == arm, "Invalid policy metadata schema")
    observing, active = arm != "adamw", arm != "adamw" and step > 100
    require(type(policy["policy_active"]) is bool and policy["policy_active"] == active,
            "Wrong policy activation")
    require(policy["observing_step"] == (step if observing else None), "Observation count mismatch")
    if observing:
        integer(policy["observing_step"], "observing step", minimum=1)
    expected_active = arm if active else ("identity_warmup" if observing else "identity_baseline")
    require(policy["active_policy"] == expected_active, "Wrong active policy")
    for name, rank in (("current", current_rank), ("lagged", previous_rank)):
        actual_rank, missing = policy[name + "_basis_rank"], policy[name + "_basis_missing"]
        operator = policy[name + "_candidate_operator"]
        if not observing:
            require(actual_rank is None and missing is None and operator is None,
                    "Baseline invented candidate basis")
        else:
            integer(actual_rank, "stored basis rank")
            require(actual_rank == rank and 0 <= actual_rank <= 32 and type(missing) is bool
                    and missing == (rank == 0), "Basis rank/missing/history disagreement")
            expected_operator = ("identity_missing_basis" if missing else "native_basis") if active else None
            require(operator == expected_operator, "Candidate operator/rank confusion")
    unavailable = "baseline_no_observer" if not observing else "warmup_candidates_not_measured"
    for name in ("raw", "current", "lagged", "applied"):
        missing = name in ("current", "lagged") and not active
        for suffix in ("norm", "squared_norm"):
            checked_scalar(row, name + "_" + suffix, null_reason=unavailable if missing else None,
                           nonnegative=True)
        if not missing:
            energy, norm = row[name + "_squared_norm"], row[name + "_norm"]
            close(norm, math.sqrt(energy), name + " norm/energy", absolute=0)
            require((energy == 0) == (norm == 0), "Zero norm/energy disagreement")
            if active and name in ("current", "lagged"):
                if row["raw_norm"] == 0:
                    require(norm == 0, "Linear candidate nonzero for zero raw input")
                if policy[name + "_basis_missing"]:
                    require(norm == row["raw_norm"] and energy == row["raw_squared_norm"],
                            "Missing basis must act as identity")
    raw_norm = row["raw_norm"]
    for name in ("current", "lagged", "applied"):
        norm = row[name + "_norm"]
        reason = unavailable if norm is None else "zero_raw_norm" if raw_norm == 0 else None
        checked_scalar(row, name + "_raw_norm_ratio", None if reason else norm / raw_norm,
                       null_reason=reason, nonnegative=True)
        if name != "applied":
            checked_scalar(row, name + "_energy_retention",
                           None if reason else row[name + "_squared_norm"] / row["raw_squared_norm"],
                           null_reason=reason, nonnegative=True)
    reason = unavailable if not active else "zero_raw_norm" if raw_norm == 0 else None
    checked_scalar(row, "current_minus_lagged_retention",
                   None if reason else row["current_energy_retention"] - row["lagged_energy_retention"],
                   null_reason=reason)
    for key, names in (("current_lagged_cosine", ("current", "lagged")),
                       ("raw_applied_cosine", ("raw", "applied"))):
        norms = [row[name + "_norm"] for name in names]
        reason = unavailable if None in norms else "zero_vector_norm" if 0 in norms else None
        checked_scalar(row, key, null_reason=reason)
        if reason is None:
            number(row[key], key, minimum=-1 - 1e-10, maximum=1 + 1e-10)
    if not active:
        direction, target, operator = "raw", raw_norm, "identity"
    elif arm == "current32":
        direction, target, operator = "current", row["current_norm"], "native_current"
    elif arm == "lagged32":
        direction, target, operator = "lagged", row["lagged_norm"], "native_lagged"
    elif arm == "lagged32_current_norm":
        direction, target, operator = "lagged", row["current_norm"], "scaled_native_lagged"
    else:
        direction = "raw"
        target = row["current_norm"] if arm == "scalar_current32" else row["lagged_norm"]
        operator = "scalar_identity"
    close(policy["target_norm"], target, "policy target norm", absolute=0)
    require(policy["scale_direction"] == direction and policy["delivery_operator"] == operator,
            "Delivery direction/operator mismatch")
    scaled = active and arm in ("lagged32_current_norm", "scalar_current32", "scalar_lagged32")
    if scaled:
        direction_norm = row[direction + "_norm"]
        require(target == 0 or direction_norm > 0, "Positive target with zero direction")
        scale = 0.0 if target == 0 else target / direction_norm
        close(policy["scale"], scale, "policy scale", absolute=0)
        checked_scalar(row, "policy_scale", scale, nonnegative=True)
        if operator == "scalar_identity":
            require(scale <= 1.005, "Scalar bound exceeded")
    else:
        require(policy["scale"] is None, "Unscaled policy has a scale")
        checked_scalar(row, "policy_scale", null_reason="unscaled_policy")
    applied_norm = row["applied_norm"]
    if target == 0:
        require(applied_norm == 0, "Zero target has nonzero delivery")
        checked_scalar(row, "norm_matching_relative_error", 0.)
        checked_scalar(row, "direction_relative_error", null_reason="zero_target_and_delivery")
    else:
        require(applied_norm > 0, "Positive target lost in delivery")
        expected_error = abs(applied_norm - target) / target
        close(row["norm_matching_relative_error"], expected_error, "relative delivery norm error",
              relative=1e-9, absolute=1e-15)
        number(row["norm_matching_relative_error"], "norm gate", minimum=0, maximum=1e-6)
        checked_scalar(row, "direction_relative_error", nonnegative=True)
        require(row["direction_relative_error"] <= 1e-6, "Direction gate failed")
    if not scaled:
        require(applied_norm == target and row["norm_matching_relative_error"] == 0,
                "Unscaled delivery differs from prescribed norm")
        require(row["direction_relative_error"] == (0.0 if target > 0 else None),
                "Unscaled delivery has direction error")
    if direction == "raw" and target > 0:
        close(row["raw_applied_cosine"], 1.0, "Raw-direction delivery cosine", absolute=2e-12)
    for kind in ("total", "decay_subtracted"):
        update = row[kind]
        require(set(update) == UPDATE_KEYS, "Unexpected update schema")
        number(update["norm"], "update norm", minimum=0)
        number(update["squared_norm"], "update energy", minimum=0)
        close(update["norm"], math.sqrt(update["squared_norm"]), "update norm/energy", absolute=0)
        expected_nulls = set()
        for component in ("raw", "applied"):
            dot = update[component + "_gradient_dot_update"]
            require(set(dot) == {"value", "tolerance", "sign"}, "Unexpected signed-dot schema")
            value = number(dot["value"], "gradient/update dot")
            product = row[component + "_norm"] * update["norm"]
            tolerance = 1e-6 * product + 1e-14
            close(dot["tolerance"], tolerance, "sign tolerance", relative=1e-12, absolute=1e-24)
            expected_sign = 1 if value > tolerance else -1 if value < -tolerance else 0
            require(type(dot["sign"]) is int and dot["sign"] == expected_sign, "Inconsistent dot sign")
            key = component + "_gradient_update_cosine"
            if product == 0:
                require(value == 0, "Zero vector has nonzero inner product")
                expected_nulls.add(key)
                checked_scalar(update, key, null_reason="zero_vector_norm")
            else:
                require(abs(value) <= product * (1 + 1e-10), "Dot violates Cauchy-Schwarz")
                checked_scalar(update, key, value / product)
        require(set(update["null_reasons"]) == expected_nulls, "Update null mask mismatch")


def validate_run(result):
    key = condition_key(result)
    finite_tree(result)
    require(result["schema_version"] == 1 and type(result["schema_version"]) is int,
            "Wrong run schema version")
    require(result["run_key"] == run_key(key), "Run key/content mismatch")
    require(result["steps"] == 2000 and type(result["steps"]) is int, "Incomplete training length")
    for flag in ("instrumented", "all_invariant_gates_passed", "warmup_checks_passed"):
        require(result[flag] is True, "Failed run flag " + flag)
    require(integer(result["delivery_gate_steps"], "delivery gate count") == 2000,
            "Incomplete delivery gates")
    require(integer(result["measurement_state_checks"], "state check count") == 22,
            "Incomplete measurement isolation checks")
    incorrect = number(result["realized_incorrect_fraction"], "realized incorrect", minimum=0, maximum=1)
    replaced = number(result["realized_replacement_fraction"], "realized replaced", minimum=0, maximum=1)
    require(incorrect <= replaced, "Incorrect fraction exceeds replacement fraction")
    for value in (incorrect, replaced):
        # The unchanged data helper returns float32 means, not Python count/count.
        # Preserve those raw values; half an f32 ULP near one is below 3e-8.
        close(value, round(5000 * value) / 5000, "realized fraction/count",
              relative=0, absolute=3e-8)
    if key[1] == 0:
        require(incorrect == replaced == 0, "Clean condition contains corruption")
        require(result["final_training_clean"] == result["final_training_noisy"],
                "Clean-condition training label evaluations disagree")
    digest_string(result["plan_sha256"], "run plan")
    rows = result["steps_raw"]
    require([row["step"] for row in rows] == list(range(1, 2001)), "Incomplete/unordered step records")
    trajectory = result["trajectory_parameter_sha256"]
    ranks, repairs = result["estimation_rank_by_step"], result["repair_count_by_step"]
    require(len(trajectory) == len(ranks) == len(repairs) == 2000, "Incomplete hash/rank/repair histories")
    observing = key[2] != "adamw"
    for index, row in enumerate(rows):
        digest_string(trajectory[index], "trajectory parameter")
        rank = integer(ranks[index], "estimation rank")
        repair = integer(repairs[index], "repair count")
        require(rank <= 32 and (observing or rank == repair == 0), "Baseline/rank/counter confusion")
        require(index == 0 or repair >= repairs[index - 1], "Repair counter moved backwards")
        validate_step(row, key[2], rank, ranks[index - 1] if index else 0)
    warmup = result["warmup_trajectory_hashes"]
    require(len(warmup) == 100 and [row["step"] for row in warmup] == list(range(1, 101)),
            "Missing warmup trajectory witnesses")
    for index, witness in enumerate(warmup):
        require(set(witness) == {"step", "parameters", "raw_gradient", "applied_gradient"},
                "Unexpected warmup witness schema")
        for field in ("parameters", "raw_gradient", "applied_gradient"):
            digest_string(witness[field], "warmup " + field)
        require(witness["parameters"] == trajectory[index], "Warmup parameter hash link failed")
        require(witness["raw_gradient"] == witness["applied_gradient"], "Warmup gradient not identity")
    digest_string(result["warmup_core_sha256"], "warmup full core")
    observer_hashes = result["warmup_observer_hashes"]
    if observing:
        require(len(observer_hashes) == 100, "Missing warmup observer witnesses")
        for value in observer_hashes:
            digest_string(value, "warmup observer")
        require(digest_string(result["warmup_observer_sha256"], "warmup observer final") == observer_hashes[-1],
                "Warmup observer final hash link failed")
    else:
        require(observer_hashes == [] and result["warmup_observer_sha256"] is None,
                "Baseline invented observer witness")
    require(set(result["checkpoint_sha256"]) == set(CHECKPOINTS), "Missing checkpoint state hashes")
    for checkpoint in CHECKPOINTS:
        digest_string(result["checkpoint_sha256"][checkpoint], "checkpoint state")
        for other in CHECKPOINTS:
            if result["checkpoint_steps"][checkpoint] == result["checkpoint_steps"][other]:
                require(result["checkpoint_sha256"][checkpoint] == result["checkpoint_sha256"][other],
                        "Same-step checkpoint state hashes disagree")
    # State-dict hash and flat parameter hash have different encodings; do not equate them.
    require(type(result["checkpoint_path"]) is str and Path(result["checkpoint_path"]).is_absolute(),
            "Invalid checkpoint path")
    elapsed = result["step_elapsed_seconds"]
    require(len(elapsed) == 2000, "Incomplete per-step timings")
    for value in elapsed:
        number(value, "step elapsed seconds", minimum=0)
    number(result["elapsed_seconds"], "run elapsed seconds", minimum=0)
    return key


def summarize_run(result):
    key = validate_run(result)
    metrics, counts = learning_metrics(result), {}

    def add(name, rows, values, reasons):
        metrics[name], counts[name] = step_aggregate(values, [row["step"] for row in rows], reasons)

    for window, (low, high) in WINDOWS.items():
        rows = result["steps_raw"][low - 1:high]
        for field in STEP_METRICS:
            name = f"gradient.{window}.{field}"
            values = [row[field] for row in rows]
            add(name, rows, values, [row["null_reasons"].get(field) for row in rows])
            if field in ("policy_scale", "norm_matching_relative_error", "direction_relative_error"):
                present = [value for value in values if value is not None]
                for label, operation in (("minimum", min), ("maximum", max)):
                    extra = name + "." + label
                    metrics[extra] = operation(present) if present else None
                    counts[extra] = counts[name]
        for kind in ("total", "decay_subtracted"):
            prefix = f"update.{window}.{kind}"
            for field in ("norm", "squared_norm"):
                add(prefix + "." + field, rows, [row[kind][field] for row in rows], [None] * len(rows))
            for component in ("raw", "applied"):
                field = component + "_gradient_dot_update"
                add(prefix + "." + field, rows, [row[kind][field]["value"] for row in rows],
                    [None] * len(rows))
                # All scheduled steps, including zero/near-zero classifications, enter this denominator.
                add(prefix + "." + component + "_gradient_ascent_frequency", rows,
                    [int(row[kind][field]["sign"] == 1) for row in rows], [None] * len(rows))
                cosine = component + "_gradient_update_cosine"
                add(prefix + "." + cosine, rows, [row[kind][cosine] for row in rows],
                    [row[kind]["null_reasons"].get(cosine) for row in rows])
    for phase, scheduled in (("scheduled_repair", True), ("other_steps", False)):
        rows = [row for row in result["steps_raw"][100:] if (row["step"] % 100 == 0) == scheduled]
        require(len(rows) == (19 if scheduled else 1881), "Unexpected scheduled phase count")
        for field in PHASE_METRICS:
            add(f"phase.{phase}.{field}", rows, [row[field] for row in rows],
                [row["null_reasons"].get(field) for row in rows])
    return {"seed": key[0], "replacement_probability": key[1], "arm": key[2],
            "run_key": run_key(key), "realized_replacement_fraction": result["realized_replacement_fraction"],
            "realized_incorrect_fraction": result["realized_incorrect_fraction"],
            "metrics": metrics, "counts": counts}


def learning_metrics(result):
    validation = validate_selectors(result)
    require(set(result["test"]) == set(CHECKPOINTS), "Incomplete checkpoint test set")
    metrics = {}
    for checkpoint in CHECKPOINTS:
        evaluation = result["test"][checkpoint]
        validate_evaluation(evaluation, 10000)
        step = result["checkpoint_steps"][checkpoint]
        integer(step, "selected step")
        metrics[f"checkpoint.{checkpoint}.step"] = step
        metrics[f"checkpoint.{checkpoint}.postwarmup_updates"] = max(step - 100, 0)
        metrics[f"checkpoint.{checkpoint}.at_or_before_warmup"] = int(step <= 100)
        for field in ("accuracy", "cross_entropy"):
            metrics[f"test.{checkpoint}.{field}"] = evaluation[field]
            metrics[f"validation.{checkpoint}.{field}"] = validation[step][field]
            if checkpoint != "warmup100":
                metrics[f"test.{checkpoint}.minus_warmup100.{field}"] = (
                    evaluation[field] - result["test"]["warmup100"][field])
        for earlier in CHECKPOINTS:
            if result["checkpoint_steps"][earlier] == step:
                require(result["test"][earlier] == evaluation,
                        "Same-step checkpoint test evaluations disagree")
    for name in ("final_training_clean", "final_training_noisy", "final_validation"):
        validate_evaluation(result[name], 5000)
        for field in ("accuracy", "cross_entropy"):
            metrics[f"learning.{name}.{field}"] = result[name][field]
    for field in ("accuracy", "cross_entropy", "count"):
        require(result["final_validation"][field] == validation[2000][field],
                "Final validation disagrees with scheduled final observation")
    return metrics


def aggregate_seed_summaries(runs):
    """Only complete three-seed summaries; preserve null-aware paired records."""
    indexed = {condition_key(run): run for run in runs}
    require(len(runs) == 36 and len(indexed) == 36 and set(indexed) == EXPECTED_CELLS,
            "Expected exactly 36 unique conditions")
    metric_keys = set(runs[0]["metrics"])
    require(PRIMARY_METRIC in metric_keys, "Missing primary metric")
    for run in runs:
        require(set(run["metrics"]) == metric_keys, "Metric schema differs across runs")
        require(set(run["counts"]).issubset(metric_keys), "Unknown metric mask")
        for metric, value in run["metrics"].items():
            if value is not None:
                number(value, "seed summary " + metric)
        require(run["metrics"][PRIMARY_METRIC] is not None, "Missing primary observation")
    arm_groups, contrasts = [], []
    for noise in NOISES:
        for arm in ARMS:
            for metric in sorted(metric_keys):
                entries = [{"seed": seed, "value": indexed[seed, noise, arm]["metrics"][metric]}
                           for seed in SEEDS]
                values = [entry["value"] for entry in entries]
                unavailable = [entry["seed"] for entry in entries if entry["value"] is None]
                arm_groups.append({"replacement_probability": noise, "arm": arm, "metric": metric,
                                   "seed_values": entries, "unavailable_seeds": unavailable,
                                   "all_three_seeds_available": not unavailable,
                                   **complete_statistics(values)})
        for treatment, control in PAIRS:
            for metric in sorted(metric_keys):
                entries, differences, unavailable = [], [], []
                for seed in SEEDS:
                    left, right = indexed[seed, noise, treatment], indexed[seed, noise, control]
                    a, b = left["metrics"][metric], right["metrics"][metric]
                    left_mask = left["counts"].get(metric, {}).get("null_steps", [])
                    right_mask = right["counts"].get(metric, {}).get("null_steps", [])
                    reason = ("missing_seed_metric" if a is None or b is None else
                              "different_null_step_masks" if left_mask != right_mask else None)
                    difference = a - b if reason is None else None
                    if reason is not None:
                        unavailable.append(seed)
                    differences.append(difference)
                    entries.append({"seed": seed, "treatment_value": a, "control_value": b,
                                    "difference": difference, "unavailable_reason": reason,
                                    "treatment_null_steps": left_mask, "control_null_steps": right_mask})
                is_primary = metric == PRIMARY_METRIC and (treatment, control) in PRIMARY_PAIRS
                require(not is_primary or not unavailable, "Incomplete primary comparison")
                contrasts.append({"replacement_probability": noise, "treatment": treatment,
                                  "control": control, "metric": metric, "primary": is_primary,
                                  "paired_differences": entries, "unavailable_seeds": unavailable,
                                  "all_three_pairs_available": not unavailable,
                                  **complete_statistics(differences)})
    primary = [entry for entry in contrasts if entry["primary"]]
    require(len(primary) == 4 and {(p["replacement_probability"], p["treatment"], p["control"])
            for p in primary} == {(n, t, c) for n in NOISES for t, c in PRIMARY_PAIRS},
            "Incomplete four-group primary family")
    return {"arm_groups": arm_groups, "paired_contrasts": contrasts, "primary": primary}


def validate_environment(environment):
    require(set(environment) == ENVIRONMENT_KEYS, "Unexpected numerical environment schema")
    require(type(environment["cpu_threads"]) is int and environment["cpu_threads"] == 1,
            "Wrong CPU thread count")
    for key, expected in (("deterministic_algorithms", True), ("tf32", False),
                          ("cudnn_benchmark", False), ("foreach", False), ("fused", False)):
        require(environment[key] is expected, "Changed numerical setting " + key)
    require(environment["cublas_workspace"] == ":4096:8", "Changed CUBLAS workspace")
    for key in ("python", "numpy", "torch", "cuda", "gpu"):
        require(type(environment[key]) is str and bool(environment[key]), "Missing environment " + key)


def validate_resources(resources, seconds):
    number(resources["elapsed_seconds"], "elapsed resource guard", minimum=0, maximum=seconds)
    for key, cap in (("peak_rss_bytes", 12 * 1024**3), ("peak_gpu_allocated_bytes", 8 * 1024**3)):
        require(integer(resources[key], key) <= cap, "Resource cap exceeded " + key)
    integer(resources["peak_gpu_reserved_bytes"], "peak reservation")


def validate_execution(execution):
    finite_tree(execution)
    require(type(execution["schema_version"]) is int and execution["schema_version"] == 1,
            "Wrong execution schema")
    require(execution["mode"] == "full" and execution["status"] == "complete",
            "Only completed full execution can be summarized")
    require(integer(execution["completed_runs"], "completed runs") == 36
            and integer(execution["test_evaluations"], "test count") == 144,
            "Incomplete 36-run/144-test execution")
    for flag in ("all_gates_passed", "warmup_checks_passed", "official_test_opened"):
        require(execution[flag] is True, "Incomplete execution gate " + flag)
    times = [utc_time(execution[key], key) for key in (
        "started_utc", "all_training_completed_utc", "test_first_opened_utc", "completed_utc")]
    require(times[0] <= times[1] < times[2] <= times[3], "Official test accessed before all training/selection")
    cells = execution["completed_cells"]
    keys = [condition_key(cell) for cell in cells]
    require(len(keys) == 36 and len(set(keys)) == 36 and set(keys) == EXPECTED_CELLS,
            "Completed cell set is not exact")
    for cell, key in zip(cells, keys):
        require(cell["run_key"] == run_key(key), "Completed cell identity mismatch")
    for family in ("runs", "training_runs", "checkpoints"):
        bindings = execution[family]
        keys = [condition_key(item) for item in bindings]
        require(len(keys) == 36 and len(set(keys)) == 36 and set(keys) == EXPECTED_CELLS,
                "Incomplete/duplicate artifact family " + family)
        for item, key in zip(bindings, keys):
            require(item["run_key"] == run_key(key), "Artifact identity mismatch")
    plans = execution["plans"]
    require(len(plans) == 3 and {integer(item["seed"], "plan seed") for item in plans} == set(SEEDS),
            "Missing/duplicate RNG plans")
    validate_environment(execution["environment"])
    validate_resources(execution["resources"], 900)
    require(integer(execution["artifact_total_bytes"], "artifact byte total") <= 1024**3,
            "Artifact byte budget exceeded")


def verify_binding(item, *, bulk_root=None):
    path = Path(item["path"])
    require(path.is_absolute() and path.is_file(), "Artifact missing or not absolute: " + str(path))
    if bulk_root is not None:
        require(path.resolve().parent == Path(bulk_root).resolve(), "Artifact escaped its exclusive bulk root")
    digest_string(item["sha256"], "artifact")
    require(path.stat().st_size == integer(item["size_bytes"], "artifact size")
            and sha(path) == item["sha256"], "Artifact hash/size mismatch: " + str(path))
    return path


def verify_sources(mapping, revision, *, root=ROOT, started_utc=None):
    require(set(mapping) == SOURCE_PATHS, "Scientific source membership mismatch")
    require(type(revision) is str and len(revision) == 40
            and all(char in "0123456789abcdef" for char in revision), "Invalid source revision")
    if started_utc is not None:
        committed_time = subprocess.run(["git", "show", "-s", "--format=%cI", revision], cwd=root,
                                        check=True, capture_output=True, text=True).stdout.strip()
        require(utc_time(committed_time, "source commit") <= utc_time(started_utc, "execution start"),
                "Scientific commit does not precede execution")
    for name, digest in mapping.items():
        digest_string(digest, "scientific source")
        require(sha(Path(root) / name) == digest, "Current scientific source mismatch: " + name)
        committed = subprocess.run(["git", "show", f"{revision}:{name}"], cwd=root,
                                   check=True, capture_output=True).stdout
        require(hashlib.sha256(committed).hexdigest() == digest, "Committed scientific source mismatch: " + name)


def verify_data_bindings(items, *, test):
    expected = ({"t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte"} if test else
                {"train-images-idx3-ubyte", "train-labels-idx1-ubyte"})
    require(len(items) == 2 and {Path(item["path"]).name for item in items} == expected,
            "Unexpected data artifact set")
    for item in items:
        verify_binding(item)


def validate_warmup_pairs(runs, *, pilot=False):
    grouped = {}
    for result in runs:
        key = (result["seed"], result["replacement_probability"])
        arms = grouped.setdefault(key, {})
        require(result["arm"] not in arms, "Duplicate warmup cell")
        arms[result["arm"]] = result
    expected = {(9880, .9)} if pilot else {(seed, noise) for seed in SEEDS for noise in NOISES}
    require(set(grouped) == expected, "Wrong warmup seed/condition groups")
    for arms in grouped.values():
        require(set(arms) == set(ARMS), "Missing paired warmup arm")
        baseline, observer = arms["adamw"], arms["current32"]
        for arm, result in arms.items():
            for key in ("warmup_trajectory_hashes", "warmup_core_sha256"):
                require(result[key] == baseline[key], "Paired warmup witness mismatch: " + key)
            if arm != "adamw":
                for key in ("warmup_observer_hashes", "warmup_observer_sha256"):
                    require(result[key] == observer[key], "Paired observer warmup mismatch: " + key)
            if not pilot:
                for key in ("plan_sha256", "realized_incorrect_fraction", "realized_replacement_fraction"):
                    require(result[key] == baseline[key], "Paired data plan/fractions differ: " + key)
                require(result["test"]["warmup100"] == baseline["test"]["warmup100"]
                        and result["checkpoint_sha256"]["warmup100"] == baseline["checkpoint_sha256"]["warmup100"],
                        "Paired warmup checkpoint/test witnesses differ")


def verify_pilot(execution, *, root=ROOT):
    path = verify_binding(execution["passing_pilot"])
    pilot = read_json(path)
    require(pilot["mode"] == "pilot" and pilot["status"] == "complete_passed"
            and integer(pilot["completed_traces"], "pilot traces") == 12, "No complete passing pilot")
    for flag in ("all_gates_passed", "warmup_checks_passed"):
        require(pilot[flag] is True, "Pilot invariant failed")
    for flag in ("official_test_opened", "validation_or_accuracy_computed"):
        require(pilot[flag] is False, "Pilot outcome-access violation")
    require(pilot["source_sha256"] == execution["source_sha256"]
            and pilot["environment"] == execution["environment"]
            and pilot["training_data_artifacts"] == execution["training_data_artifacts"],
            "Source/environment/training data changed since pilot")
    verify_sources(pilot["source_sha256"], pilot["repository_revision"], root=root,
                   started_utc=pilot["started_utc"])
    require(utc_time(pilot["started_utc"], "pilot start") < utc_time(pilot["completed_utc"], "pilot complete")
            < utc_time(execution["started_utc"], "full start"), "Pilot does not precede full execution")
    validate_resources(pilot["resources"], 180)
    require(len(pilot["plans"]) == 1 and pilot["plans"][0]["seed"] == 9880, "Wrong development plan")
    verify_binding(pilot["plans"][0], bulk_root=pilot["bulk_root"])
    reports = read_json(verify_binding(pilot["timing_and_invariants"], bulk_root=pilot["bulk_root"]))
    require(len(reports) == 6 and {report["arm"] for report in reports} == set(ARMS), "Incomplete pilot arm reports")
    variants = {"uninstrumented": [], "instrumented": []}
    for report in reports:
        for flag in ("trajectory_bitwise_identical", "final_state_bitwise_identical", "warmup_checks_passed"):
            require(report[flag] is True, "Failed pilot measurement-invariance witness")
        for name, enabled in (("uninstrumented", False), ("instrumented", True)):
            trace = report[name]
            require(trace["seed"] == 9880 and trace["replacement_probability"] == .9
                    and trace["arm"] == report["arm"] and trace["steps"] == 220
                    and trace["instrumented"] is enabled, "Wrong pilot trace identity")
            require(trace["delivery_gate_steps"] == 220 and trace["measurement_state_checks"] == (5 if enabled else 0),
                    "Incomplete pilot mandatory/optional gates")
            require(trace["all_invariant_gates_passed"] is True and trace["warmup_checks_passed"] is True,
                    "Failed pilot trace gates")
            require(len(trace["trajectory_parameter_sha256"]) == 220
                    and len(trace["warmup_trajectory_hashes"]) == 100, "Incomplete pilot trajectory witnesses")
            require(not ({"steps_raw", "test", "validation_trajectory", "final_training_clean", "final_training_noisy",
                          "final_validation", "checkpoint_steps", "checkpoint_sha256"} & set(trace)),
                    "Forbidden pilot scientific outcomes")
            ranks, repairs = trace["estimation_rank_by_step"], trace["repair_count_by_step"]
            require(len(ranks) == len(repairs) == len(trace["step_elapsed_seconds"]) == 220,
                    "Incomplete pilot rank/repair/timing history")
            for index, (rank, repair, seconds) in enumerate(zip(ranks, repairs, trace["step_elapsed_seconds"])):
                integer(rank, "pilot rank")
                integer(repair, "pilot repairs")
                require(rank <= 32 and (trace["arm"] != "adamw" or rank == repair == 0),
                        "Pilot baseline/rank confusion")
                require(index == 0 or repair >= repairs[index - 1], "Pilot repair count decreased")
                number(seconds, "pilot step timing", minimum=0)
            for value in trace["trajectory_parameter_sha256"]:
                digest_string(value, "pilot trajectory")
            for index, witness in enumerate(trace["warmup_trajectory_hashes"]):
                require(witness["step"] == index + 1 and witness["parameters"] == trace["trajectory_parameter_sha256"][index]
                        and witness["raw_gradient"] == witness["applied_gradient"], "Pilot warmup hash link failed")
                for key in ("parameters", "raw_gradient", "applied_gradient"):
                    digest_string(witness[key], "pilot warmup")
            digest_string(trace["warmup_core_sha256"], "pilot warmup core")
            if trace["arm"] == "adamw":
                require(trace["warmup_observer_hashes"] == [] and trace["warmup_observer_sha256"] is None,
                        "Pilot baseline has fake observer")
            else:
                require(len(trace["warmup_observer_hashes"]) == 100
                        and trace["warmup_observer_hashes"][-1] == trace["warmup_observer_sha256"],
                        "Pilot observer warmup witness incomplete")
                for value in trace["warmup_observer_hashes"]:
                    digest_string(value, "pilot observer")
            variants[name].append(trace)
        left, right = report["uninstrumented"], report["instrumented"]
        for key in ("trajectory_parameter_sha256", "warmup_trajectory_hashes", "warmup_observer_hashes",
                    "warmup_core_sha256", "warmup_observer_sha256", "estimation_rank_by_step", "repair_count_by_step"):
            require(left[key] == right[key], "Pilot on/off witness mismatch " + key)
    for traces in variants.values():
        validate_warmup_pairs(traces, pilot=True)
    return pilot


def load_and_summarize(execution_path, *, root=ROOT):
    execution_path = Path(execution_path)
    execution = read_json(execution_path)
    validate_execution(execution)
    verify_sources(execution["source_sha256"], execution["repository_revision"], root=root,
                   started_utc=execution["started_utc"])
    verify_data_bindings(execution["training_data_artifacts"], test=False)
    verify_data_bindings(execution["test_data_artifacts"], test=True)
    pilot = verify_pilot(execution, root=root)
    bulk_root = Path(execution["bulk_root"])
    require(bulk_root.is_absolute() and bulk_root.resolve().parent == Path("/tmp/spectral-experiment-artifacts").resolve(),
            "Wrong bulk storage root")
    mount = execution["bulk_mount"]
    require(mount["target"] == "/private-artifacts/storage" and (mount["source"] == "/dev/RECONFIGURE_FOR_LOCAL_STORAGE"
            or mount.get("uuid") == "00000000-0000-4000-8000-000000000000"), "Unverified bulk mount")
    all_bindings = [item for family in ("plans", "training_runs", "runs", "checkpoints") for item in execution[family]]
    require(len({item["path"] for item in all_bindings}) == len(all_bindings), "Aliased bulk artifact bindings")
    for item in all_bindings:
        verify_binding(item, bulk_root=bulk_root)
    recorded_bytes = sum(item["size_bytes"] for item in all_bindings)
    require(recorded_bytes == execution["artifact_total_bytes"], "Unbound or inconsistent total bulk bytes")
    plans = {item["seed"]: item for item in execution["plans"]}
    checkpoints = {condition_key(item): item for item in execution["checkpoints"]}
    pretest = {condition_key(item): item for item in execution["training_runs"]}
    runs, summaries = [], []
    for binding in execution["runs"]:
        result = read_json(binding["path"])
        key = condition_key(result)
        require(key == condition_key(binding) and result["run_key"] == binding["run_key"], "Run file/binding identity mismatch")
        require(result["plan_sha256"] == plans[key[0]]["sha256"], "Run plan binding mismatch")
        require(Path(result["checkpoint_path"]).resolve() == Path(checkpoints[key]["path"]).resolve(),
                "Run checkpoint binding mismatch")
        before_test = read_json(pretest[key]["path"])
        require("test" not in before_test and {k: v for k, v in result.items() if k != "test"} == before_test,
                "Final run differs from immutable pre-test training evidence")
        summaries.append(summarize_run(result))
        runs.append(result)
    validate_warmup_pairs(runs)
    groups = aggregate_seed_summaries(summaries)
    return {"schema_version": 1, "execution_status": "complete", "source_sha256": sha(HERE / "summarize_results.py"),
            "execution_sha256": sha(execution_path), "scientific_source_sha256": execution["source_sha256"],
            "passing_pilot_sha256": execution["passing_pilot"]["sha256"],
            "repository_revision": execution["repository_revision"], "pilot_revision": pilot["repository_revision"],
            "scope": "Three paired seeds per condition on reused MNIST/test data; descriptive whole-policy comparisons. Accuracy is a fraction.",
            "primary_metric": PRIMARY_METRIC, "primary_update": "decay_subtracted; total mandatory",
            "source_bindings_verified": 15, "bulk_artifacts_verified": len(all_bindings),
            "bulk_bytes_verified": recorded_bytes, "input_bindings": all_bindings,
            "seed_summaries": summaries, **groups}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=HERE / "results")
    parser.add_argument("--output", type=Path, default=HERE / "summary.json")
    args = parser.parse_args()
    require(not args.output.exists(), "Refusing to overwrite existing summary")
    result = load_and_summarize(args.results / "execution.json")
    write_exclusive(args.output, result)
    print(json.dumps({"output": str(args.output), "sha256": sha(args.output), "runs": 36,
                      "primary_groups": len(result["primary"]), "paired_metric_contrasts": len(result["paired_contrasts"])}))


if __name__ == "__main__":
    main()
