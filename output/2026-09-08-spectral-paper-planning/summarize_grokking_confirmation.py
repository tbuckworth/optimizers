"""Collect the entire registered five-seed confirmation and plot its trajectories."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics

ARMS = ("adamw", "legacy", "stable")
SEEDS = tuple(range(100, 105))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def threshold(rows):
    crossings = [r["step"] for r in rows if r["test"]["accuracy"] >= .9]
    first = min(crossings) if crossings else None
    sustained = next((r["step"] for i, r in enumerate(rows)
                      if all(later["test"]["accuracy"] >= .9 for later in rows[i:])), None)
    return first, sustained


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-roots", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    results, evidence = {}, []
    source_pins = None
    for root in args.batch_roots:
        completion = json.loads((root / "complete.json").read_text())
        if source_pins is None:
            source_pins = completion["source_sha256"]
        elif completion["source_sha256"] != source_pins:
            raise ValueError("Batches used different frozen scientific sources.")
        for entry in completion["accepted_results"]:
            key = (entry["seed"], entry["arm"])
            if key in results:
                raise ValueError("Duplicate scientific result, not another replicate.")
            path = Path(entry["metrics_path"])
            if path.resolve().parent.parent.parent != root.resolve():
                raise ValueError("Metrics outside declared batch root.")
            if digest(path) != entry["metrics_sha256"]:
                raise ValueError("Metrics changed since batch completion.")
            result = json.loads(path.read_text())
            rows = result["evaluation_rows"]
            if (result["status"] != "complete" or result["completed_steps"] != 6000
                    or [r["step"] for r in rows] != list(range(0, 6001, 50))
                    or (result["config"]["seed"], result["config"]["arm"]) != key):
                raise ValueError("Result does not match fixed roster/grid.")
            first, sustained = threshold(rows)
            if (first != result["test_accuracy_90"]["upper_step_inclusive"]
                    or sustained != result["test_accuracy_90"]["sustained_attainment_step"]):
                raise ValueError("Independent threshold reconstruction disagrees.")
            results[key] = result
            evidence.append({"seed": key[0], "arm": key[1], "source": str(path),
                             "sha256": digest(path)})
    if set(results) != {(seed, arm) for seed in SEEDS for arm in ARMS}:
        raise ValueError("All 15 registered outcomes are required; no selected-subset summary.")
    reference = results[100, "adamw"]
    common_config = {k: v for k, v in reference["config"].items()
                     if k not in ("seed", "arm", "filter")}
    legacy_filter = dict(results[100, "legacy"]["config"]["filter"])
    legacy_filter.pop("stable_update")
    for (seed, arm), result in results.items():
        config = result["config"]
        common = {k: v for k, v in config.items() if k not in ("seed", "arm", "filter")}
        if (common != common_config or result["environment"] != reference["environment"]
                or result["source_identity"]["files"] != reference["source_identity"]["files"]
                or result["parameter_identity"] != reference["parameter_identity"]):
            raise ValueError("Recipe, software/hardware, parameter layout or scientific source mismatch.")
        expected_filter = None if arm == "adamw" else {**legacy_filter, "stable_update": arm == "stable"}
        if config["filter"] != expected_filter:
            raise ValueError("Unexpected within-roster filter configuration difference.")
    for seed in SEEDS:
        splits = {results[seed, arm]["split_identity"]["sha256"] for arm in ARMS}
        if len(splits) != 1:
            raise ValueError("Within-seed data mismatch.")

    summary = {"design": "prospective within-benchmark confirmation, five paired seeds",
               "steps": 6000, "evaluation_grid": 50, "seed_results": [], "arms": {},
               "paired_first_crossing_differences": {}, "evidence": evidence}
    for seed in SEEDS:
        for arm in ARMS:
            result = results[seed, arm]
            first, sustained = threshold(result["evaluation_rows"])
            row = {"seed": seed, "arm": arm, "first_90_recorded_step": first,
                   "sustained_90_recorded_step": sustained,
                   "final_test_accuracy": result["evaluation_rows"][-1]["test"]["accuracy"],
                   "final_test_loss": result["evaluation_rows"][-1]["test"]["loss"],
                   "training_seconds": result["timing"]["training_seconds"],
                   "end_to_end_seconds": result["timing"]["end_to_end_seconds_before_final_json"]}
            summary["seed_results"].append(row)
    for arm in ARMS:
        rows = [r for r in summary["seed_results"] if r["arm"] == arm]
        uncensored = all(r["first_90_recorded_step"] is not None for r in rows)
        summary["arms"][arm] = {
            "n": 5, "crossings": sum(r["first_90_recorded_step"] is not None for r in rows),
            "mean_first_90_recorded_step": statistics.mean(r["first_90_recorded_step"] for r in rows)
            if uncensored else None,
            "mean_final_test_accuracy": statistics.mean(r["final_test_accuracy"] for r in rows),
            "mean_training_seconds": statistics.mean(r["training_seconds"] for r in rows),
            "mean_end_to_end_seconds": statistics.mean(r["end_to_end_seconds"] for r in rows)}
    for control in ("adamw", "legacy"):
        differences = []
        for seed in SEEDS:
            treatment = threshold(results[seed, "stable"]["evaluation_rows"])[0]
            reference = threshold(results[seed, control]["evaluation_rows"])[0]
            differences.append(None if treatment is None or reference is None else treatment-reference)
        complete = all(d is not None for d in differences)
        summary["paired_first_crossing_differences"]["stable_minus_" + control] = {
            "seeds": SEEDS, "differences_steps": differences,
            "mean_steps": statistics.mean(differences) if complete else None,
            "standard_error_steps": statistics.stdev(differences)/math.sqrt(5) if complete else None,
            "interpretation": "negative is earlier stable crossing; 50-step observation grid, n=5"}

    output = args.output_dir.resolve()
    output.mkdir(parents=False, exist_ok=False)
    raw = output / "raw"
    raw.mkdir()
    for entry in evidence:
        filename = f"seed{entry['seed']}-{entry['arm']}.json"
        shutil.copyfile(entry["source"], raw / filename)
        if digest(raw / filename) != entry["sha256"]:
            raise ValueError("Copy hash mismatch.")
        entry["local_copy"] = "raw/" + filename
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"adamw": "#52657b", "legacy": "#cc7a29", "stable": "#087d78"}
    labels = {"adamw": "AdamW", "legacy": "Legacy filter", "stable": "Stable filter"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for axis, split in zip(axes, ("train", "test")):
        for arm in ARMS:
            curves = [[r[split]["accuracy"] for r in results[seed, arm]["evaluation_rows"]]
                      for seed in SEEDS]
            steps = list(range(0, 6001, 50))
            for curve in curves:
                axis.plot(steps, curve, color=colors[arm], alpha=.22, linewidth=.8)
            means = [statistics.mean(point) for point in zip(*curves)]
            axis.plot(steps, means, color=colors[arm], linewidth=2.2, label=labels[arm])
        axis.set(title="Training pairs" if split == "train" else "Held-out pairs",
                 xlabel="Optimizer updates", ylim=(-.02, 1.03))
        axis.grid(alpha=.16)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Accuracy")
    axes[1].axhline(.9, color="#8694a3", linestyle=":", linewidth=1)
    axes[1].legend(frameon=False, loc="lower right")
    fig.suptitle("Modular addition: current stable vs legacy vs AdamW", fontsize=14)
    fig.text(.5, .01, "Five paired seeds; thin lines = individual runs, thick lines = means. "
             "Fixed recipe and horizon; not a wall-clock benchmark.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .055, 1, .95))
    fig.savefig(output / "learning-curves.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for axis, split in zip(axes, ("train", "test")):
        for arm in ARMS:
            curves = [[r[split]["loss"] for r in results[seed, arm]["evaluation_rows"]]
                      for seed in SEEDS]
            for curve in curves:
                axis.plot(steps, curve, color=colors[arm], alpha=.22, linewidth=.8)
            axis.plot(steps, [statistics.mean(point) for point in zip(*curves)],
                      color=colors[arm], linewidth=2.2, label=labels[arm])
        axis.set(title="Training pairs" if split == "train" else "Held-out pairs",
                 xlabel="Optimizer updates")
        axis.set_yscale("symlog", linthresh=1e-6)
        axis.grid(alpha=.16)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Cross-entropy loss")
    axes[1].legend(frameon=False)
    fig.suptitle("Modular addition: training and generalization losses", fontsize=14)
    fig.text(.5, .01, "Five paired seeds; thin lines = individual runs, thick lines = means. "
             "Loss scale is logarithmic above 0.000001.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .055, 1, .95))
    fig.savefig(output / "loss-curves.png", dpi=160)
    plt.close(fig)
    print(json.dumps({"output": str(output), "arms": summary["arms"],
                      "paired": summary["paired_first_crossing_differences"]}, indent=2))


if __name__ == "__main__":
    main()
