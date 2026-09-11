#!/usr/bin/env python3
"""Render the disclosed I19 summary view from accepted aggregate JSON only."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


HERE = Path(__file__).resolve().parent
SEEDS = tuple(range(19000, 19032))
PROCESS_VARIANCES = (0.0, 0.01, 0.1)
WINDOWS = ("whole", "startup", "transition", "late")
SHA256 = re.compile(r"[0-9a-f]{64}")
DISPLAY_POLICIES = (
    ("ema_q0p99", "EMA q=.99", "#377eb8"),
    ("ema_q0p9", "EMA q=.9", "#6baed6"),
    ("dema_q0p99", "DEMA q=.99", "#ff9f1c"),
    ("common_kalman", "Common Kalman\n(model-based)", "#4daf4a"),
    ("native_cp", "Native CP\nρ=.9", "#7b2cbf"),
    ("native_cp_star", "Native CP*\nρ=.730", "#b5179e"),
    ("oracle_useful_star", "Fixed route ρ*\n(direction oracle)", "#8c6d31"),
    ("useful_oracle_kalman", "Directional Kalman\n(model-based oracle)", "#d62728"),
)


def policy_names(process_variance: float) -> tuple[str, ...]:
    names = ["raw", "ema_q0p9", "ema_q0p99", "ema_q0p999",
             "dema_q0p9", "dema_q0p99", "dema_q0p999"]
    if process_variance != 0.0:
        names.append("ema_common_steady_oracle")
    names.extend(("common_kalman", "useful_oracle_kalman",
                  "scalar_k0", "scalar_k0p5", "scalar_k0p9", "scalar_k1",
                  "oracle_useful", "oracle_nuisance", "native_i17", "native_cp",
                  "native_cp_star", "oracle_useful_star"))
    return tuple(names)


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def read_json(path: Path):
    def pairs(items):
        value = {}
        for key, item in items:
            if key in value:
                raise ValueError("duplicate JSON key: " + key)
            value[key] = item
        return value
    with path.open(encoding="utf-8") as handle:
        return json.load(handle, object_pairs_hook=pairs, parse_constant=lambda token:
            (_ for _ in ()).throw(ValueError("nonfinite JSON constant: " + token)))


def finite_number(value) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def expected_keys() -> set[tuple[int, int, str, str]]:
    return {(process_index, rotation_index, window, policy)
            for process_index, process_variance in enumerate(PROCESS_VARIANCES)
            for rotation_index in (0, 1)
            for window in WINDOWS
            for policy in policy_names(process_variance)}


def validate_summary(summary: dict, audit: dict, expected_hashes: dict[str, str]):
    if not (summary.get("schema") == "i19_tracking_summary_v1"
            and audit.get("schema") == "i19_tracking_audit_v1"
            and summary.get("audit_status") == audit.get("status") == "pass"
            and audit.get("errors") == []
            and audit.get("summary_sha256") == expected_hashes["summary"]
            and summary.get("all_scientific_streams_complete") is True
            and summary.get("completed_streams") == summary.get("expected_streams") == 192
            and summary.get("missing_streams") == []
            and audit.get("artifact_root") == summary.get("artifact_root")
            and audit.get("attempt_sha256") == summary.get("attempt_sha256")
            and audit.get("completion_sha256") == summary.get("completion_sha256")
            and audit.get("input_provenance") == summary.get("input_provenance")):
        raise ValueError("complete accepted I19 evidence is required")

    means = summary.get("equal_seed_mse_summaries")
    if type(means) is not list or len(means) != 472:
        raise ValueError("exactly 472 aggregate MSE summaries are required")
    index = {}
    seed_keys = {str(seed) for seed in SEEDS}
    for row in means:
        if type(row) is not dict:
            raise ValueError("aggregate summary row must be an object")
        try:
            key = (row["process_index"], row["rotation_index"],
                   row["window"], row["policy"])
            value = row["mse"]
        except KeyError as exc:
            raise ValueError("aggregate summary row is incomplete") from exc
        if key in index:
            raise ValueError("duplicate aggregate MSE summary")
        if not (type(key[0]) is int and type(key[1]) is int
                and type(key[2]) is str and type(key[3]) is str):
            raise ValueError("invalid aggregate MSE key types")
        if type(value) is not dict or value.get("available") is not True \
                or value.get("n_independent_seeds") != 32 \
                or value.get("no_survivor_averaging") is not True \
                or set(value.get("per_seed", {})) != seed_keys:
            raise ValueError("complete equal-seed MSE summary is required")
        ordered = [value["per_seed"][str(seed)] for seed in SEEDS]
        if not all(finite_number(item) and item >= 0.0 for item in ordered):
            raise ValueError("finite nonnegative per-seed MSE values are required")
        mean = statistics.fmean(ordered)
        standard_error = statistics.stdev(ordered) / math.sqrt(len(ordered))
        if not (finite_number(value.get("mean"))
                and finite_number(value.get("standard_error"))
                and math.isclose(value["mean"], mean, rel_tol=2e-12, abs_tol=5e-15)
                and math.isclose(value["standard_error"], standard_error,
                                 rel_tol=2e-12, abs_tol=5e-15)):
            raise ValueError("aggregate MSE mean or seed SE does not reconstruct")
        index[key] = row
    if set(index) != expected_keys():
        raise ValueError("aggregate MSE roster differs from the frozen 472-row design")
    return means, index


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-sha256", required=True)
    parser.add_argument("--audit-sha256", required=True)
    args = parser.parse_args()
    if SHA256.fullmatch(args.summary_sha256) is None \
            or SHA256.fullmatch(args.audit_sha256) is None:
        raise SystemExit("full lowercase SHA-256 pins are required")

    paths = {"summary": HERE / "analysis-001" / "summary.json",
             "audit": HERE / "analysis-001" / "audit.json"}
    expected_hashes = {"summary": args.summary_sha256, "audit": args.audit_sha256}
    for name, path in paths.items():
        if not path.is_file() or path.is_symlink() or digest(path) != expected_hashes[name]:
            raise SystemExit("input file/hash differs: " + name)
    summary = read_json(paths["summary"])
    audit = read_json(paths["audit"])
    try:
        means, index = validate_summary(summary, audit, expected_hashes)
    except (KeyError, TypeError, ValueError, statistics.StatisticsError) as exc:
        raise SystemExit(str(exc)) from exc

    output = HERE / "plots-001"
    output.mkdir(exist_ok=False)
    csv_path = output / "all-policy-window-means.csv"
    with csv_path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("process_variance", "process_index", "rotation_index", "window",
                         "policy", "policy_class", "mean_mse", "seed_standard_error",
                         "independent_seeds"))
        for row in means:
            value = row["mse"]
            writer.writerow((PROCESS_VARIANCES[row["process_index"]], row["process_index"],
                             row["rotation_index"], row["window"], row["policy"],
                             row["policy_class"], value["mean"], value["standard_error"],
                             value["n_independent_seeds"]))

    plt.rcParams.update({"font.size": 10.5, "axes.titleweight": "bold"})
    figure, axes = plt.subplots(1, 3, figsize=(16.8, 6.5), sharey=False)
    figure.patch.set_facecolor("white")
    plotted = []
    positions = list(range(len(DISPLAY_POLICIES)))
    for process_index, (axis, process_variance) in enumerate(
            zip(axes, PROCESS_VARIANCES)):
        whole = [index[(process_index, 0, "whole", name)]["mse"]
                 for name, _, _ in DISPLAY_POLICIES]
        late = [index[(process_index, 0, "late", name)]["mse"]
                for name, _, _ in DISPLAY_POLICIES]
        colors = [color for _, _, color in DISPLAY_POLICIES]
        axis.bar(positions, [row["mean"] for row in whole],
                 yerr=[row["standard_error"] for row in whole],
                 color=colors, alpha=.8, edgecolor="white", linewidth=.7,
                 error_kw={"ecolor": "#333333", "elinewidth": 1.0, "capsize": 2.5},
                 zorder=2)
        axis.errorbar(positions, [row["mean"] for row in late],
                      yerr=[row["standard_error"] for row in late],
                      fmt="o", markersize=5.5, markerfacecolor="white",
                      markeredgecolor="#111111", markeredgewidth=1.2,
                      color="#111111", elinewidth=1.1, capsize=3, zorder=4)
        for window, rows in (("whole", whole), ("late", late)):
            plotted.extend({"process_index": process_index,
                            "process_variance": process_variance,
                            "rotation_index": 0, "window": window,
                            "policy": name, "mean_mse": row["mean"],
                            "seed_standard_error": row["standard_error"],
                            "n_independent_seeds": row["n_independent_seeds"]}
                           for (name, _, _), row in zip(DISPLAY_POLICIES, rows))
        axis.set_title(f"Process variance Q₁ = {process_variance:g}", loc="left")
        axis.set_xticks(positions, [label for _, label, _ in DISPLAY_POLICIES],
                        rotation=34, ha="right", rotation_mode="anchor")
        axis.set_ylabel("Tracking MSE (lower is better)")
        axis.grid(axis="y", alpha=.22, zorder=0)
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_axisbelow(True)

    axes[0].legend(handles=(
        Patch(facecolor="#777777", alpha=.8, label="Whole horizon 1–4000 — primary"),
        Line2D([], [], color="#111111", marker="o", markerfacecolor="white",
               linestyle="none", label="Late window 1001–4000 — secondary")),
        frameon=False, loc="upper left", fontsize=9)
    figure.suptitle("I19 stochastic tracking: whole-horizon and late-window MSE",
                   x=.055, ha="left", fontsize=15, weight="bold")
    figure.text(.055, .035,
                "Identity rotation; 32 independent canonical seeds. Bars and dots show means ±1 seed-level SE (not confidence intervals). "
                "Late-window results are secondary. Model-based and direction-oracle controls are labeled explicitly.\n"
                "EMA q=.9 was added after the accepted audit as a presentation control; all 472 registered summaries remain in the CSV.",
                fontsize=9, color="#444444")
    figure.tight_layout(rect=(0.025, .12, 1, .93), w_pad=2.2)
    png_path = output / "stochastic-tracking.png"
    figure.savefig(png_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(figure)

    manifest = {
        "schema": "i19_stochastic_tracking_plot_v1",
        "source_sha256": digest(Path(__file__)),
        "input_sha256": {str(path.relative_to(HERE)): expected_hashes[name]
                         for name, path in paths.items()},
        "output_sha256": {png_path.name: digest(png_path),
                          csv_path.name: digest(csv_path)},
        "plot_file": png_path.name,
        "csv_file": csv_path.name,
        "plotted_aggregate_points": plotted,
        "plotted_point_count": len(plotted),
        "csv_rows": len(means),
        "complete_summary_rows_validated": 472,
        "display_policy_selection": (
            "seven policies were requested before I19 outcome access; EMA q=.9 was added "
            "after the accepted audit as a transparent post-outcome presentation control, "
            "without changing the complete 472-row CSV"),
        "scope": "accepted aggregate summary only; no scientific arrays, observer, or RNG replay",
        "uncertainty": "plus/minus one standard error across 32 independent seeds; not a confidence interval",
        "primary_secondary_boundary": "whole horizon is primary; late window is secondary",
    }
    manifest_path = output / "manifest.json"
    with manifest_path.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"output": str(output), "plotted_points": len(plotted),
                      "csv_rows": len(means)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
