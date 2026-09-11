#!/usr/bin/env python3
"""Aggregate both I11 source histories and audit archived JSON/source provenance."""
import gzip
import hashlib
import io
import json
from pathlib import Path
from statistics import mean
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    here = Path(__file__).resolve().parent
    repo = here.parents[3]
    analysis = here / "analysis-001"
    assert json.loads((analysis / "audit.json").read_text())["status"] == "pass"
    summary = json.loads((analysis / "summary.json").read_text())
    branches = summary["physical_stitched_curves_changes_and_diagnostics"]
    assert len(branches) == 63
    horizons = summary["hindsight_raw_sampled_grid"]["grid"]
    rows = []
    for source in ("raw", "current32"):
        for policy in ("raw", "current32", "frozen32"):
            seeds = []
            for seed in (100, 101, 102):
                parents = [r for r in branches if r["seed"] == seed
                           and r["physical_source"] == source
                           and r["i9_parent_step"] in (1500, 2000)
                           and r["policy"] == policy]
                assert len(parents) == 2
                values = {}
                for horizon in horizons:
                    levels = [next(x for x in parent["stitched_curve"]
                                   if x["cumulative_horizon"] == horizon)
                              for parent in parents]
                    for split in ("train", "auxiliary", "validation"):
                        for metric in levels[0][split]:
                            values[f"{split}/{metric}/h{horizon}"] = mean(
                                level[split][metric] for level in levels)
                for window in ("i11_500_to2000", "combined_0_to2000"):
                    for metric in parents[0]["diagnostics"][window]["energy_weighted_leakage"]:
                        values[f"{window}/mean_branch_energy_fraction/{metric}"] = mean(
                            p["diagnostics"][window]["energy_weighted_leakage"][metric]["fraction"]
                            for p in parents)
                seeds.append({"seed": seed, "values": values})
            rows.append({"source": source, "policy": policy, "per_seed": seeds,
                         "mean": {k: mean(s["values"][k] for s in seeds)
                                  for k in seeds[0]["values"]}})
    archive = here / "raw-results"
    collection = json.loads((archive / "collection.json").read_text())
    assert collection["json_count"] == len(collection["files"])
    assert len({r["original"] for r in collection["files"]}) == len(collection["files"])
    for row in collection["files"]:
        compressed = (archive / row["archive"]).read_bytes()
        assert len(compressed) == row["archive_bytes"] and sha(compressed) == row["archive_sha256"]
        raw = gzip.decompress(compressed)
        assert len(raw) == row["original_bytes"] and sha(raw) == row["original_sha256"]
    manifest = json.loads(gzip.decompress((archive / "manifest.json.gz").read_bytes()))
    completion = gzip.decompress((archive / "completion.json.gz").read_bytes())
    assert sha(completion) == collection["source_completion_sha256"]
    sources = []
    for path, expected in manifest["source_hashes"].items():
        current = sha((repo / path).read_bytes())
        frozen = sha(subprocess.check_output(["git", "show", manifest["git_commit"] + ":" + path], cwd=repo))
        assert current == frozen == expected, path
        sources.append({"path": path, "sha256": current})
    assert len(sources) == 16
    tables = {"scope": "Both source histories, parent1500/2000 means within seed, then three-seed mean",
              "leakage_aggregation": "Energy-weighted within branch, then equal parent and seed weights",
              "horizons": horizons, "rows": rows}
    audit = {"status": "pass", "source_commit": manifest["git_commit"],
             "source_files_match_worktree_and_commit": sources,
             "lossless_json_archives_checked": len(collection["files"]),
             "finite_loss_replay": False,
             "limitation": "Recorded evaluations and scalar diagnostics; no independent forward/vector replay"}
    for name, payload in (("report-tables.json", tables), ("report-audit.json", audit)):
        with (analysis / name).open("x") as handle:
            json.dump(payload, handle, indent=2, allow_nan=False)
            handle.write("\n")

    colors = {"raw": "#B35820", "current32": "#007F86", "frozen32": "#6655A4"}
    labels = {"raw": "Raw AdamW", "current32": "Current filter", "frozen32": "Frozen filter"}
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, sharey="row")
    for col, source in enumerate(("current32", "raw")):
        for row_index, metric in enumerate(("clean_ce", "clean_accuracy")):
            ax = axes[row_index, col]
            scale = 100 if metric == "clean_accuracy" else 1
            for policy in colors:
                group = next(r for r in rows if r["source"] == source and r["policy"] == policy)
                for seed in group["per_seed"]:
                    ax.plot(horizons, [scale * seed["values"][f"auxiliary/{metric}/h{h}"]
                                       for h in horizons], color=colors[policy], alpha=.22, linewidth=1)
                ax.plot(horizons, [scale * group["mean"][f"auxiliary/{metric}/h{h}"]
                                  for h in horizons], color=colors[policy], linewidth=2.4,
                        marker="o", markersize=3, label=labels[policy])
            ax.axvline(500, color="#444444", linewidth=.8, linestyle="--")
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", alpha=.18)
            if col == 0:
                ax.set_ylabel("Clean CE (lower is better)" if metric == "clean_ce"
                              else "Clean accuracy (%)")
            if row_index == 0:
                ax.set_title("Previously filtered states (primary)" if source == "current32"
                             else "Previously raw-trained states (secondary)")
            else:
                ax.set_xlabel("Cumulative new updates since the I9 parent")
    fig.suptitle("Does protection become useful continued learning?", fontsize=16, y=.98)
    fig.text(.5, .925, "Fixed corrupted labels; dashed boundary is the exact I10-to-I11 resume",
             ha="center", fontsize=10)
    handles, names = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, names, loc="lower center", ncol=3, bbox_to_anchor=(.5, .035), frameon=False)
    fig.text(.5, .012, "Thin individual seeds, thick mean; parent steps 1500/2000 averaged within seed.\n"
             "Different policies already have different states at the 500-update boundary.",
             ha="center", va="bottom", fontsize=9, color="#444444")
    fig.tight_layout(rect=(0, .10, 1, .91))
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=160)
    with (here / "duration-curves.png").open("xb") as handle:
        handle.write(buffer.getvalue())
    plt.close(fig)
    print(json.dumps({"status": "pass", "groups": len(rows), "sources": len(sources),
                      "archives": len(collection["files"])}))


if __name__ == "__main__":
    main()
