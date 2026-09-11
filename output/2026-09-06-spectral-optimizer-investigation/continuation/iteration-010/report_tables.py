#!/usr/bin/env python3
"""Deterministic I10 report tables and separate frozen-source/archive checks."""
import gzip
import hashlib
import json
from pathlib import Path
from statistics import mean
import subprocess


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    here = Path(__file__).resolve().parent
    repo = here.parents[3]
    summary = json.loads((here / "analysis-001/summary.json").read_text())
    assert json.loads((here / "analysis-001/audit.json").read_text())["status"] == "pass"
    branches = summary["physical_branch_curves_and_diagnostics"]
    all_rows = []
    for objective in ("fixed", "soft", "redraw"):
        for policy in ("raw", "current32", "frozen32"):
            seeds = []
            for seed in (100, 101, 102):
                parents = [r for r in branches if r["seed"] == seed
                           and r["physical_source"] == "current32"
                           and r["parent_step"] in (1500, 2000)
                           and r["objective"] == objective and r["policy"] == policy]
                assert len(parents) == 2
                values = {}
                for split in ("train", "auxiliary", "validation"):
                    for metric in parents[0]["curve"][0][split]:
                        for horizon in (0, 500):
                            values[f"{split}/{metric}/h{horizon}"] = mean(
                                next(x for x in parent["curve"] if x["horizon"] == horizon)[split][metric]
                                for parent in parents)
                for metric in parents[0]["diagnostic_summary"]["arithmetic_step_means"]:
                    values["steps/mean/" + metric] = mean(
                        p["diagnostic_summary"]["arithmetic_step_means"][metric] for p in parents)
                for metric in parents[0]["diagnostic_summary"]["energy_weighted_leakage"]:
                    # Ratio within each branch, then equal parent weighting, then seed mean.
                    values["steps/mean_branch_energy_fraction/" + metric] = mean(
                        p["diagnostic_summary"]["energy_weighted_leakage"][metric]["fraction"]
                        for p in parents)
                seeds.append({"seed": seed, "values": values})
            all_rows.append({"objective": objective, "policy": policy, "per_seed": seeds,
                             "mean": {k: mean(s["values"][k] for s in seeds) for k in seeds[0]["values"]}})
    archive = here / "raw-results"
    collection = json.loads((archive / "collection.json").read_text())
    for row in collection["files"]:
        compressed = (archive / row["archive"]).read_bytes()
        assert len(compressed) == row["archive_bytes"] and sha(compressed) == row["archive_sha256"]
        raw = gzip.decompress(compressed)
        assert len(raw) == row["original_bytes"] and sha(raw) == row["original_sha256"]
    manifest = json.loads(gzip.decompress((archive / "manifest.json.gz").read_bytes()))
    sources = []
    for path, expected in manifest["source_hashes"].items():
        current = sha((repo / path).read_bytes())
        frozen = sha(subprocess.check_output(["git", "show", manifest["git_commit"] + ":" + path], cwd=repo))
        assert current == frozen == expected, path
        sources.append({"path": path, "sha256": current})
    output = {"scope": "Current-source parents 1500/2000, averaged within seed before three-seed mean",
              "leakage_aggregation": "Energy-weighted within branch; equal-weight parent ratios then seeds, not a pooled cross-branch energy ratio",
              "rows": all_rows}
    audit = {"status": "pass", "source_commit": manifest["git_commit"],
             "source_files_match_worktree_and_commit": sources,
             "lossless_json_archives_checked": len(collection["files"]),
             "finite_loss_replay": False,
             "limitation": "Tables aggregate recorded evaluations; no independent forward pass or per-step tensor-vector recomputation."}
    for name, payload in (("report-tables.json", output), ("report-audit.json", audit)):
        with (here / "analysis-001" / name).open("x") as handle:
            json.dump(payload, handle, indent=2, allow_nan=False)
            handle.write("\n")
    print(json.dumps({"status": "pass", "groups": len(all_rows), "sources": len(sources),
                      "archives": len(collection["files"])}))


if __name__ == "__main__":
    main()
