#!/usr/bin/env python3
"""One exclusive plot/table rendering from accepted I18 summary JSON only."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
POLICIES = (
    ("native_cp", "Spectral: delivery state", "#087f8c", "o"),
    ("native_i17", "Spectral: normalized buffer", "#c75d32", "s"),
    ("ema_q0p99", "Uniform EMA, q=.99", "#5158a5", "D"),
    ("ema_q0p9", "Uniform EMA, q=.9", "#777777", "^"),
    ("ema_optimal_oracle", "Optimal steady-state EMA (oracle)", "#a85a91", "v"),
    ("oracle_useful", "Fixed e1 route (oracle)", "#478652", "P"),
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-sha256", required=True)
    parser.add_argument("--audit-sha256", required=True)
    args = parser.parse_args()
    paths = {"summary": HERE / "analysis-001/summary.json", "audit": HERE / "analysis-001/audit.json"}
    expected = {"summary": args.summary_sha256, "audit": args.audit_sha256}
    for name, path in paths.items():
        if digest(path) != expected[name]:
            raise SystemExit("Input hash differs: " + name)
    summary, audit = (json.loads(paths[name].read_text()) for name in ("summary", "audit"))
    if not (audit["schema"] == "i18_tracking_audit_v1" and summary["schema"] == "i18_tracking_summary_v1"
            and audit["status"] == summary["audit_status"] == "pass" and audit["errors"] == []
            and audit["summary_sha256"] == expected["summary"]
            and audit["artifact_root"] == summary["artifact_root"]
            and audit["input_provenance"] == summary["input_provenance"]
            and summary["all_scientific_streams_complete"] and summary["completed_streams"] == 192
            and audit["completion_sha256"] == summary["completion_sha256"]
            and audit["attempt_sha256"] == summary["attempt_sha256"]):
        raise SystemExit("Complete accepted evidence required")
    output = HERE / "plots-001"
    output.mkdir(exist_ok=False)
    means = summary["equal_seed_mse_summaries"]
    alignment = summary["equal_seed_alignment_summaries"]
    index = {(r["drift_index"], r["rotation_index"], r["window"], r["policy"]): r["mse"] for r in means}
    drift = (0., .01, .03)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [1.5, 1]})
    fig.patch.set_facecolor("#ffffff")
    point_records = []
    for name, label, color, marker in POLICIES:
        ds = (1, 2) if name == "ema_optimal_oracle" else (0, 1, 2)
        rows = [index[(di, 0, "late", name)] for di in ds]
        if not all(row["available"] and row["n_independent_seeds"] == 32 for row in rows):
            raise SystemExit("Incomplete plotted seed summary")
        xs, ys, errors = [drift[di] for di in ds], [r["mean"] for r in rows], [r["standard_error"] for r in rows]
        axes[0].errorbar(xs, ys, yerr=errors, marker=marker, color=color,
                         label=label, linewidth=1.7, markersize=5, capsize=3)
        point_records.extend({"panel": "mse", "policy": name, "drift_index": di,
                              "mean": row["mean"], "standard_error": row["standard_error"]}
                             for di, row in zip(ds, rows))
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Tracking MSE — logarithmic scale; lower is better")
    axes[0].set_title("Tracking the latent signal", loc="left", fontsize=12, weight="bold")
    axes[0].legend(frameon=False, fontsize=8, loc="best")
    for field, label, color, marker in (
            ("native_mean_squared_alignment_when_present", "Native rank-one observer", "#087f8c", "o"),
            ("full_mean_squared_alignment_when_available", "Full-moment diagnostic", "#777777", "s")):
        rows = [next(row for row in alignment if row["drift_index"] == di
                     and row["rotation_index"] == 0 and row["window"] == "late")[field] for di in range(3)]
        if not all(row["available"] for row in rows):
            raise SystemExit("Unavailable learned directions cannot be plotted as zeros")
        axes[1].errorbar(drift, [r["mean"] for r in rows], yerr=[r["standard_error"] for r in rows],
                         color=color, marker=marker, linewidth=1.7, capsize=3, label=label)
        point_records.extend({"panel": "alignment", "field": field, "drift_index": di,
                              "mean": row["mean"], "standard_error": row["standard_error"]}
                             for di, row in enumerate(rows))
    axes[1].axhline(.5, color="#c7c7c7", linestyle="--", linewidth=1)
    axes[1].set_ylim(-.02, 1.02)
    axes[1].set_ylabel("Squared alignment with generating axis e1")
    axes[1].set_title("Leading-direction alignment", loc="left", fontsize=12, weight="bold")
    axes[1].legend(frameon=False, fontsize=8, loc="best")
    for ax in axes:
        ax.set_xticks(drift, ["None\n0", "Weak\n0.01", "Strong\n0.03"])
        ax.set_xlabel("Signal drift per observation")
        ax.grid(axis="y", alpha=.2)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Spectral Optimizer Investigation | native tracking test", x=.06, ha="left", fontsize=14, weight="bold")
    fig.text(.06, .02, "32 paired seeds; identity rotation; steps 1001–4000. Bars: ±1 seed-level SE, not confidence intervals.\n"
             "Six prechosen display policies; every registered control/window/rotation is in the accompanying table.\n"
             "At zero drift the steady-state EMA-risk optimum is not attained for q<1; e1 is only an axis. Toy tracking ≠ neural optimization.",
             fontsize=8, color="#444444")
    fig.tight_layout(rect=(0, .14, 1, .94))
    png = output / "native-tracking.png"
    fig.savefig(png, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    table = output / "all-policy-window-means.csv"
    with table.open("x", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("drift", "rotation_index", "window", "policy", "mean_mse", "seed_standard_error", "independent_seeds"))
        for row in means:
            value = row["mse"]
            writer.writerow((drift[row["drift_index"]], row["rotation_index"], row["window"], row["policy"],
                             value["mean"], value["standard_error"], value["n_independent_seeds"]))
    manifest = {"schema": "i18_tracking_plot_v1", "source_sha256": digest(Path(__file__)),
                "input_sha256": {str(path.relative_to(HERE)): expected[name] for name, path in paths.items()},
                "output_sha256": {path.name: digest(path) for path in (png, table)},
                "points": point_records, "table_rows": len(means), "scientific_arrays_replayed": False,
                "display_policy_selection": "six policies specified in source while acquisition was live, before outcome reads",
                "uncertainty": "plus/minus one standard error across 32 paired seeds; not time-independent errors or CIs"}
    with (output / "manifest.json").open("x") as handle:
        json.dump(manifest, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"output": str(output), "plotted_points": len(point_records), "table_rows": len(means)}))


if __name__ == "__main__":
    main()
