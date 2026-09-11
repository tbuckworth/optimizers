#!/usr/bin/env python3
"""Independent scalar report tables, source/archive checks and I12 CE chart."""
import gzip
import hashlib
import json
from pathlib import Path
import statistics
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
I10 = HERE.parent / "iteration-010/raw-results"
ARMS = ("inherited", "zero_m", "zero_v", "zero_mv", "fresh_adam")
SEEDS, PARENTS = (100, 101, 102), (1500, 2000)
HORIZONS = (0, 1, 10, 50, 100, 250, 500)
OBJECTIVES, POLICIES = ("soft", "redraw"), ("raw", "current32")


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read(name, directory=None):
    directory = directory or HERE / "raw-results"
    return json.loads(gzip.decompress((directory / (name + ".gz")).read_bytes()))


def seed_first(values):
    result = {}
    for seed in SEEDS:
        pair = [values.get((seed, parent)) for parent in PARENTS]
        result[str(seed)] = None if None in pair else statistics.mean(pair)
    return {"seed_values": result, "mean": None if None in result.values()
            else statistics.mean(result.values())}


def main():
    audit = json.loads((HERE / "analysis-001/audit.json").read_text())
    assert audit["status"] == "pass"
    collection = json.loads((HERE / "raw-results/collection.json").read_text())
    for row in collection["files"]:
        raw = (HERE / "raw-results" / row["archive"]).read_bytes()
        assert sha(raw) == row["archive_sha256"] and len(raw) == row["archive_bytes"]
        original = gzip.decompress(raw)
        assert sha(original) == row["original_sha256"] and len(original) == row["original_bytes"]
    manifest = read("manifest.json")
    assert manifest["git_commit"] == "6c40f99" or manifest["git_commit"].startswith("6c40f99")
    for name, expected in manifest["source_hashes"].items():
        assert sha((ROOT / name).read_bytes()) == expected
        committed = subprocess.check_output(["git", "show", "6c40f99:" + name], cwd=ROOT)
        assert sha(committed) == expected
    completion = read("completion.json")
    assert completion["attempted_branches"] == 96
    branches = {}
    for row in read("branches.json")["entries"]:
        key = (row["seed"], row["parent_step"], row["objective"], row["policy"], row["arm"])
        assert key not in branches
        branches[key] = read(row["branch_artifact"]["name"])
    for row in read("baseline-references.json")["entries"]:
        key = (row["seed"], row["parent_step"], row["objective"], row["policy"], "inherited")
        assert key not in branches
        branches[key] = read(row["branch_artifact"]["name"], I10)
    assert len(branches) == 120

    def value(seed, parent, objective, policy, arm, horizon, split, metric):
        curve = branches[(seed, parent, objective, policy, arm)]["curve"]
        row = next((r for r in curve if r["horizon"] == horizon), None)
        return None if row is None else row[split][metric]

    levels, benefits, primaries, first_steps = [], [], [], []
    for objective in OBJECTIVES:
        for arm in ARMS:
            for policy in POLICIES:
                for horizon in HORIZONS:
                    for split in ("train", "auxiliary", "validation"):
                        metrics = branches[(100, 1500, objective, policy, arm)]["curve"][0][split]
                        for metric in metrics:
                            points = {(s, p): value(s, p, objective, policy, arm, horizon, split, metric)
                                      for s in SEEDS for p in PARENTS}
                            levels.append({"objective": objective, "arm": arm, "policy": policy,
                                "horizon": horizon, "split": split, "metric": metric,
                                **seed_first(points)})
                points = {}
                for s in SEEDS:
                    for p in PARENTS:
                        rows = branches[(s, p, objective, policy, arm)]["steps"]
                        points[(s, p)] = rows[0]["displacement"]["data_norm"] if rows else None
                first_steps.append({"objective": objective, "arm": arm, "policy": policy,
                                    **seed_first(points)})
            for horizon in HORIZONS:
                for metric, sign in (("clean_ce", -1), ("clean_accuracy", 1)):
                    points = {}
                    for s in SEEDS:
                        for p in PARENTS:
                            raw, current = [value(s, p, objective, policy, arm, horizon,
                                                  "auxiliary", metric) for policy in POLICIES]
                            points[(s, p)] = None if raw is None or current is None else sign * (current - raw)
                    benefits.append({"objective": objective, "arm": arm, "horizon": horizon,
                                     "metric": metric, **seed_first(points)})
        for metric, sign in (("clean_ce", -1), ("clean_accuracy", 1)):
            points = {}
            for s in SEEDS:
                for p in PARENTS:
                    terms = [value(s, p, objective, policy, arm, 500, "auxiliary", metric)
                             for arm in ("zero_v", "zero_m") for policy in POLICIES]
                    points[(s, p)] = None if None in terms else sign * (
                        terms[1] - terms[0] - terms[3] + terms[2])
            primaries.append({"objective": objective, "metric": metric, **seed_first(points)})

    tables = {"status": "complete", "science_complete": completion["science_complete"],
              "numerical_failures": completion["numerical_failures"], "primaries": primaries,
              "levels": levels, "benefits": benefits, "first_data_step_norms": first_steps}
    with (HERE / "analysis-001/report-tables.json").open("x") as handle:
        json.dump(tables, handle, indent=2, allow_nan=False)
        handle.write("\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = dict(zip(ARMS, ("#111827", "#e69f00", "#009e73", "#cc79a7", "#0072b2")))
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7), sharex=True)
    fig.patch.set_facecolor("white")
    for row, objective in enumerate(OBJECTIVES):
        for col, policy in enumerate(POLICIES):
            ax = axes[row, col]
            for arm in ARMS:
                selected = [r for r in levels if (r["objective"], r["policy"], r["arm"],
                    r["split"], r["metric"]) == (objective, policy, arm, "auxiliary", "clean_ce")]
                ax.plot([r["horizon"] for r in selected], [r["mean"] for r in selected],
                        label=arm, color=colors[arm], linewidth=1.8)
            ax.set_title(f"{objective} targets · {policy}")
            ax.grid(alpha=.2)
            # Outcome-driven presentation only: retain the very large finite
            # zero-v losses without making all remaining curves invisible.
            ax.set_yscale("log")
            ax.set_ylabel("Clean CE (log scale) ↓")
            if row == 1:
                ax.set_xlabel("New branch updates")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False)
    fig.suptitle("Adam history interventions: absolute learning, not just relative gaps", fontsize=13)
    fig.tight_layout(rect=(0, .055, 1, .96))
    fig.savefig(HERE / "moment-curves.png", dpi=150, facecolor="white")
    plt.close(fig)
    report_audit = {"status": "pass", "source_files": len(manifest["source_hashes"]),
        "frozen_commit": "6c40f99", "archive_roundtrips": collection["json_count"],
        "scalar_branches": len(branches), "independent_primary_aggregation": primaries,
        "scope": "Source/commit and archive-byte checks; direct scalar aggregation, no independent forward or vector replay."}
    with (HERE / "analysis-001/report-audit.json").open("x") as handle:
        json.dump(report_audit, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(report_audit))


if __name__ == "__main__":
    main()
