#!/usr/bin/env python3
"""Aggregate only the complete, prospectively specified iteration004 results."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
SEEDS = (3, 4, 5)
ARMS = ("adamw", "estimate32_project32", "estimate128_project32", "scalar32_norm")
CHECKPOINTS = ("final", "min_val_ce", "max_val_accuracy", "warmup100")
WINDOWS = {"all": (101, 2000), "early": (101, 500), "late": (1501, 2000)}
CONTRASTS = ((ARMS[1], ARMS[0]), (ARMS[1], ARMS[3]), (ARMS[2], ARMS[1]),
             (ARMS[2], ARMS[0]), (ARMS[2], ARMS[3]), (ARMS[3], ARMS[0]))
PRIMARY = "test.max_val_accuracy.accuracy"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finite_mean(values, steps):
    present = [float(value) for value in values if value is not None]
    if not all(math.isfinite(value) for value in present):
        raise ValueError("Non-finite observed value")
    return (statistics.fmean(present) if present else None,
            {"finite": len(present), "null": len(values) - len(present),
             "null_steps": [step for step, value in zip(steps, values) if value is None]})


def validate_evaluation(evaluation, count):
    if evaluation["count"] != count:
        raise ValueError("Incomplete evaluation")
    accuracy, ce = evaluation["accuracy"], evaluation["cross_entropy"]
    if not math.isfinite(accuracy) or not 0 <= accuracy <= 1 or not math.isfinite(ce) or ce < 0:
        raise ValueError("Invalid evaluation metric")


def validate_selectors(result):
    trajectory = result["validation_trajectory"]
    if [row["step"] for row in trajectory] != list(range(0, 2001, 100)):
        raise ValueError("Unexpected validation schedule")
    for row in trajectory:
        validate_evaluation(row, 5000)
    expected = {"min_val_ce": min(trajectory, key=lambda r: r["cross_entropy"])["step"],
                "max_val_accuracy": max(trajectory, key=lambda r: r["accuracy"])["step"],
                "final": 2000, "warmup100": 100}
    if result["checkpoint_steps"] != expected:
        raise ValueError("Checkpoint does not follow independent earliest strict selector")
    return {row["step"]: row for row in trajectory}


def summarize_run(result):
    if result["seed"] not in SEEDS or result["arm"] not in ARMS or result["replacement_probability"] != .9:
        raise ValueError("Unexpected experimental condition")
    if result["steps"] != 2000 or [r["step"] for r in result["steps_raw"]] != list(range(1, 2001)):
        raise ValueError("Incomplete or unordered training records")
    if not result["all_invariant_gates_passed"] or not result["warmup_checks_passed"]:
        raise ValueError("Failed training invariants")
    validation = validate_selectors(result)
    if set(result["test"]) != set(CHECKPOINTS):
        raise ValueError("Unexpected checkpoint set")
    metrics, counts = {}, {}

    def add(name, values, steps):
        metrics[name], counts[name] = finite_mean(values, steps)

    for checkpoint in CHECKPOINTS:
        evaluation = result["test"][checkpoint]
        validate_evaluation(evaluation, 10000)
        step = result["checkpoint_steps"][checkpoint]
        metrics[f"checkpoint.{checkpoint}.step"] = step
        for field in ("accuracy", "cross_entropy"):
            metrics[f"test.{checkpoint}.{field}"] = evaluation[field]
            metrics[f"validation.{checkpoint}.{field}"] = validation[step][field]
            if checkpoint != "warmup100":
                metrics[f"test.{checkpoint}.minus_warmup100.{field}"] = (
                    evaluation[field] - result["test"]["warmup100"][field])
    for name in ("final_training_clean", "final_training_noisy", "final_validation"):
        validate_evaluation(result[name], 5000)
        for field in ("accuracy", "cross_entropy"):
            metrics[f"learning.{name}.{field}"] = result[name][field]

    for window, (low, high) in WINDOWS.items():
        rows = [r for r in result["steps_raw"] if low <= r["step"] <= high]
        steps = [r["step"] for r in rows]
        prefix = f"gradient.{window}"
        for component in ("raw", "candidate", "applied"):
            energies = [r[f"{component}_squared_norm"] for r in rows]
            add(f"{prefix}.{component}_squared_norm", energies, steps)
            add(f"{prefix}.{component}_norm", [math.sqrt(v) for v in energies], steps)
        for component in ("candidate", "applied"):
            add(f"{prefix}.{component}_raw_norm_ratio",
                [math.sqrt(r[f"{component}_squared_norm"] / r["raw_squared_norm"])
                 if r["raw_squared_norm"] > 0 else None for r in rows], steps)
        for field in ("alpha", "raw_applied_cosine", "scalar_collinearity_relative_error",
                      "norm_matching_absolute_error"):
            add(f"{prefix}.{field}", [r[field] for r in rows], steps)
        metrics[f"{prefix}.max_norm_matching_absolute_error"] = max(r["norm_matching_absolute_error"] for r in rows)
        collinear = [r["scalar_collinearity_relative_error"] for r in rows
                     if r["scalar_collinearity_relative_error"] is not None]
        metrics[f"{prefix}.max_scalar_collinearity_relative_error"] = max(collinear) if collinear else None
        for kind in ("total", "decay_subtracted"):
            prefix = f"update.{window}.{kind}"
            entries = [r[kind] for r in rows]
            add(prefix + ".squared_norm", [r["squared_norm"] for r in entries], steps)
            add(prefix + ".norm", [math.sqrt(r["squared_norm"]) for r in entries], steps)
            for component in ("raw", "applied"):
                field = f"{component}_gradient_dot_update"
                for r in entries:
                    sign = r[field]
                    expected = 1 if sign["value"] > sign["tolerance"] else (-1 if sign["value"] < -sign["tolerance"] else 0)
                    if not math.isfinite(sign["value"]) or not math.isfinite(sign["tolerance"]) or sign["tolerance"] < 0 or sign["sign"] != expected:
                        raise ValueError("Inconsistent tolerance-qualified direction")
                add(prefix + "." + field, [r[field]["value"] for r in entries], steps)
                add(prefix + f".{component}_gradient_ascent_frequency", [int(r[field]["sign"] == 1) for r in entries], steps)
                field = f"{component}_gradient_update_cosine"
                add(prefix + "." + field, [r[field] for r in entries], steps)
    return {"seed": result["seed"], "arm": result["arm"], "replacement_probability": .9,
            "realized_incorrect_fraction": result["realized_incorrect_fraction"],
            "realized_replacement_fraction": result["realized_replacement_fraction"],
            "metrics": metrics, "counts": counts}


def paired_contrasts(runs):
    indexed = {(r["seed"], r["arm"]): r for r in runs}
    if len(indexed) != 12 or len(runs) != 12 or set(indexed) != {(s, a) for s in SEEDS for a in ARMS}:
        raise ValueError("Expected exactly twelve unique conditions")
    contrasts = []
    for treatment, control in CONTRASTS:
        for metric in runs[0]["metrics"]:
            values, unavailable = [], []
            for seed in SEEDS:
                left, right = indexed[seed, treatment], indexed[seed, control]
                a, b = left["metrics"][metric], right["metrics"][metric]
                if a is None or b is None or left["counts"].get(metric) != right["counts"].get(metric):
                    unavailable.append(seed)
                else:
                    values.append({"seed": seed, "difference": a - b})
            differences = [v["difference"] for v in values]
            complete = len(differences) == 3
            contrasts.append({"treatment": treatment, "control": control, "metric": metric,
                              "co_primary": metric == PRIMARY and (treatment, control) in CONTRASTS[:2],
                              "paired_differences": values, "unavailable_seeds": unavailable,
                              "all_three_pairs_available": complete,
                              "mean": statistics.fmean(differences) if complete else None,
                              "median": statistics.median(differences) if complete else None,
                              "minimum": min(differences) if complete else None,
                              "maximum": max(differences) if complete else None,
                              "sample_sd_descriptive": statistics.stdev(differences) if complete else None})
    return contrasts


def validate_execution(execution):
    if (execution["mode"], execution["status"], execution["completed_runs"], execution["test_evaluations"]) != ("confirmatory", "complete", 12, 48):
        raise ValueError("Only the completed twelve-run confirmation can be summarized")
    if not execution["warmup_checks_passed"] or execution["all_training_completed_utc"] > execution["test_first_loaded_utc"]:
        raise ValueError("Warmup or test-access ordering gate failed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=HERE / "results")
    parser.add_argument("--output", type=Path, default=HERE / "summary.json")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Refusing to overwrite a summary")
    execution_path = args.results / "execution.json"
    execution = json.loads(execution_path.read_text())
    validate_execution(execution)
    runs, inputs, warmup = [], [{"path": str(execution_path), "sha256": sha(execution_path)}], {}
    for seed in SEEDS:
        for arm in ARMS:
            path = args.results / f"seed{seed}-{arm}.json"
            result = json.loads(path.read_text())
            if (result["seed"], result["arm"]) != (seed, arm):
                raise ValueError("Filename/content condition mismatch")
            if arm == "adamw":
                warmup[seed] = result["test"]["warmup100"]
            elif result["test"]["warmup100"] != warmup[seed]:
                raise ValueError("Common warmup checkpoint test evaluations differ")
            runs.append(summarize_run(result))
            inputs.append({"path": str(path), "sha256": sha(path)})
    summary = {"source_sha256": sha(Path(__file__)), "inputs": inputs,
               "scope": "Three paired seed bundles on reused MNIST/test data; descriptive only. Accuracy is a fraction.",
               "primary_update": "decay_subtracted; total mandatory",
               "seed_summaries": runs, "paired_contrasts": paired_contrasts(runs)}
    with args.output.open("x") as handle:
        json.dump(summary, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"output": str(args.output), "sha256": sha(args.output), "runs": len(runs),
                      "paired_metric_contrasts": len(summary["paired_contrasts"])}))


if __name__ == "__main__":
    main()
