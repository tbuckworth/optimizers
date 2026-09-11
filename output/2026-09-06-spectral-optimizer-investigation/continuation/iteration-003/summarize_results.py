#!/usr/bin/env python3
"""Summarize the complete frozen experiment; never consume development results."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
ARMS = ("adamw", "estimate32_project32", "estimate128_project32")
WINDOWS = {"all": (101, 2000), "early": (101, 500), "late": (1501, 2000)}
PROBE_COUNTS = {"all": 39, "early": 9, "late": 10}
UPDATE_COUNTS = {"all": 1900, "early": 400, "late": 500}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finite_mean(values):
    present = [float(value) for value in values if value is not None]
    if not all(math.isfinite(value) for value in present):
        raise ValueError("Non-finite observed value")
    return (statistics.fmean(present) if present else None,
            {"finite": len(present), "null": len(values) - len(present)})


def leakage_events(metrics):
    """Descriptive signs plus a stricter, closure-margin-qualified reversal."""
    total = metrics["raw_gradient_dot_update"]
    inside = metrics["in_subspace_contribution"]
    outside = metrics["outside_contribution"]
    closure = abs(metrics["identity_closure_residual"])
    robust = (total["value"] > total["tolerance"] + closure
              and inside["value"] < -inside["tolerance"] - closure
              and outside["value"] > outside["tolerance"] + closure)
    return {
        "current_gradient_ascent": total["sign"] == 1,
        "positive_outside_contribution": outside["sign"] == 1,
        "nominal_leakage_reversal": (inside["sign"] == -1
                                     and outside["sign"] == 1 and total["sign"] == 1),
        "closure_robust_leakage_reversal": robust,
        "reversal_unresolved_by_closure": total["sign"] == 1 and inside["sign"] == -1 and not robust,
    }


def summarize_run(result):
    if result["steps"] != 2000 or len(result["steps_raw"]) != 2000:
        raise ValueError("Incomplete training records")
    if [row["step"] for row in result["steps_raw"]] != list(range(1, 2001)):
        raise ValueError("Duplicate, missing or unordered training step")
    expected_probes = [step for step in range(1, 2001)
                       if step in (1, 101) or step % 50 == 0]
    if [row["step"] for row in result["probes_raw"]] != expected_probes:
        raise ValueError("Unexpected probe schedule")
    metrics, counts, energy_sums = {}, {}, {}

    def add(name, values, steps):
        metrics[name], counts[name] = finite_mean(values)
        counts[name]["null_steps"] = [step for step, value in zip(steps, values) if value is None]

    for window, (low, high) in WINDOWS.items():
        probes = [row for row in result["probes_raw"] if low <= row["step"] <= high]
        updates = [row for row in result["steps_raw"] if low <= row["step"] <= high]
        assert len(probes) == PROBE_COUNTS[window]
        assert len(updates) == UPDATE_COUNTS[window]
        probe_steps = [row["step"] for row in probes]
        update_steps = [row["step"] for row in updates]
        for component in ("clean", "corruption_residual", "auxiliary_clean", "noisy"):
            name = f"probe.{window}.{component}_retention"
            entries = [row["retention"][component] for row in probes]
            add(name, [entry["squared_norm_retention"] for entry in entries], probe_steps)
            denominator = sum(entry["squared_norm"] for entry in entries)
            numerator = sum(entry["projected_squared_norm"] for entry in entries)
            metrics[name + ".secondary_energy_weighted"] = (numerator / denominator
                                                           if denominator else None)
            counts[name + ".secondary_energy_weighted"] = counts[name].copy()
            energy_sums[name + ".secondary_energy_weighted"] = {
                "summed_original_squared_norm": denominator,
                "summed_projected_squared_norm": numerator,
                "scheduled_observations": len(entries),
            }
        add(f"probe.{window}.clean_minus_corruption_retention",
            [row["clean_minus_corruption_retention"] for row in probes], probe_steps)
        for name in ("clean_dot_corruption", "projected_clean_dot_corruption",
                     "noisy_energy", "projected_noisy_energy"):
            add(f"probe.{window}.joint.{name}", [row["joint_geometry"][name] for row in probes], probe_steps)
        for kind in ("total", "decay_subtracted"):
            prefix = f"update.{window}.{kind}"
            entries = [row[kind] for row in updates]
            add(prefix + ".leakage_squared_fraction", [v["leakage_squared_fraction"] for v in entries], update_steps)
            add(prefix + ".cosine_raw_gradient_update", [v["cosine_raw_gradient_update"] for v in entries], update_steps)
            for term in ("raw_gradient_dot_update", "in_subspace_contribution", "outside_contribution"):
                add(prefix + "." + term, [v[term]["value"] for v in entries], update_steps)
            events = [leakage_events(v) for v in entries]
            for name in events[0]:
                add(prefix + "." + name + "_frequency", [int(v[name]) for v in events], update_steps)
            metrics[prefix + ".max_identity_relative_closure"] = max(
                v["identity_relative_closure"] for v in entries)
        add(f"probe.{window}.finite_training_batch_loss_increase_frequency",
            [int(row["same_training_batch_loss"]["sign"] == 1) for row in probes], probe_steps)
        add(f"probe.{window}.finite_training_batch_loss_change",
            [row["same_training_batch_loss"]["change"] for row in probes], probe_steps)

    for checkpoint in ("final", "validation_selected"):
        if result["test"][checkpoint]["count"] != 10000:
            raise ValueError("Incomplete test evaluation")
        for name in ("accuracy", "cross_entropy"):
            metrics[f"test.{checkpoint}.{name}"] = result["test"][checkpoint][name]
    for name in ("final_training_clean", "final_training_noisy", "final_validation"):
        for field in ("accuracy", "cross_entropy"):
            metrics[f"learning.{name}.{field}"] = result[name][field]
    metrics["learning.validation_selected_step"] = result["validation_selected_step"]
    return {"seed": result["seed"], "arm": result["arm"],
            "replacement_probability": result["replacement_probability"],
            "realized_incorrect_fraction": result["realized_incorrect_fraction"],
            "metrics": metrics, "counts": counts, "energy_sums": energy_sums}


def paired_contrasts(runs):
    indexed = {(run["seed"], run["replacement_probability"], run["arm"]): run for run in runs}
    contrasts = []
    for noise in (0., .9):
        for treatment, control in ((ARMS[2], ARMS[1]), (ARMS[1], ARMS[0]), (ARMS[2], ARMS[0])):
            rows = [(indexed[(seed, noise, treatment)], indexed[(seed, noise, control)]) for seed in (0, 1, 2)]
            for name in rows[0][0]["metrics"]:
                values, unavailable = [], []
                for seed, (treated, baseline) in enumerate(rows):
                    left, right = treated["metrics"][name], baseline["metrics"][name]
                    finite_counts = (treated["counts"].get(name), baseline["counts"].get(name))
                    if left is None or right is None or finite_counts[0] != finite_counts[1]:
                        unavailable.append(seed)
                    else:
                        values.append({"seed": seed, "difference": left - right})
                differences = [value["difference"] for value in values]
                complete = len(differences) == 3
                contrasts.append({
                    "replacement_probability": noise, "treatment": treatment, "control": control,
                    "metric": name, "paired_differences": values, "unavailable_seeds": unavailable,
                    "all_three_pairs_available": complete,
                    "mean": statistics.fmean(differences) if complete else None,
                    "median": statistics.median(differences) if complete else None,
                    "minimum": min(differences) if complete else None,
                    "maximum": max(differences) if complete else None,
                    "sample_sd_descriptive": statistics.stdev(differences) if complete else None,
                })
    return contrasts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=HERE / "results")
    parser.add_argument("--output", type=Path, default=HERE / "summary.json")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Refusing to overwrite a summary")
    execution_path = args.results / "execution.json"
    execution = json.loads(execution_path.read_text())
    if execution["mode"] != "confirmatory" or execution["status"] != "complete" or execution["completed_runs"] != 18:
        raise SystemExit("Only the complete 18-run confirmatory experiment can be summarized")
    runs, inputs = [], [{"path": str(execution_path), "sha256": sha(execution_path)}]
    for seed in (0, 1, 2):
        for noise in (0., .9):
            for arm in ARMS:
                path = args.results / f"seed{seed}-noise{noise:g}-{arm}.json"
                result = json.loads(path.read_text())
                if (result["seed"], result["replacement_probability"], result["arm"]) != (seed, noise, arm):
                    raise ValueError("Filename/content condition mismatch")
                runs.append(summarize_run(result))
                inputs.append({"path": str(path), "sha256": sha(path)})
    summary = {"source_sha256": sha(Path(__file__)), "inputs": inputs,
               "scope": "Three paired seed-level contrasts; no independent-step inference or significance claims. Accuracy is a fraction.",
               "primary_update": "decay_subtracted; total also fully reported",
               "ratio_aggregation": "Arithmetic mean of finite per-step ratios; secondary ratio of summed energies separately named.",
               "seed_summaries": runs, "paired_contrasts": paired_contrasts(runs)}
    args.output.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "sha256": sha(args.output), "runs": len(runs),
                      "paired_metric_contrasts": len(summary["paired_contrasts"])}))


if __name__ == "__main__":
    main()
