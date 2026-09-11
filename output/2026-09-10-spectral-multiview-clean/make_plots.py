#!/usr/bin/env python3
"""Render scalar-only figures AFTER the independent audit has passed.

Import and default CLI are inert. Explicit --execute reads only the audit.json
beside this script, never its referenced archive, logits, plans or checkpoints.
Outputs are exclusive new files in this same report directory. This is not a
training, acquisition, audit-replay or numerical-remediation entry point.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path


HERE = Path(__file__).resolve().parent
DIRECTORY_NAME = "2026-09-10-spectral-multiview-clean"
AUDIT_SCHEMA = "spectral_multiview_clean_audit_v1"
SEEDS = (202609161, 202609162, 202609163)
POLICIES = ("raw1", "native1", "observer4", "raw4")
STEPS = (0, 100) + tuple(range(200, 4001, 200))
PANELS = ("heldout", "heldout_translated")
COLORS = ("#0072B2", "#D55E00", "#009E73", "#9467BD")
LABELS = ("Raw AdamW: 1 view", "Native spectral: 1 view",
          "4-view observer: 1-view update", "Raw AdamW: 4-view average")
SHORT_LABELS = ("Raw AdamW\n1 view", "Native spectral\n1 view",
                "4-view observer\n1-view update", "Raw AdamW\n4-view average")
SEED_MARKERS = ("o", "s", "^")
CURVES_NAME = "primary-learning-curves.png"
ENDPOINTS_NAME = "paired-endpoints.png"
MANIFEST_NAME = "plots-manifest.json"
INPUT_CAP = 32 * 1024**2
PNG_CAP = 8 * 1024**2


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(payload):
    return hashlib.sha256(payload).hexdigest()


def finite_number(value, name):
    require(type(value) in (int, float) and math.isfinite(value), "nonfinite/invalid " + name)
    return float(value)


def validate_audit(audit):
    """Validate only the audited scalar layout used by these two figures."""
    require(type(audit) is dict and audit.get("schema") == AUDIT_SCHEMA, "unexpected audit schema")
    require(audit.get("status") == "PASS", "plotting requires independent audit PASS")
    require(audit.get("trajectories") == 12
            and audit.get("logical_evaluation_records") == 12 * len(STEPS), "incomplete audit roster")
    summary = audit.get("summary", {})
    require(summary.get("seeds") == list(SEEDS) and summary.get("endpoint_step") == 4000
            and summary.get("warmup_step") == 100, "unexpected summary seeds/endpoint/warmup")
    branches = audit.get("branches")
    require(type(branches) is list and len(branches) == 12, "expected 12 audited scalar branches")
    indexed = {}
    for branch in branches:
        require(type(branch) is dict and type(branch.get("seed")) is int, "invalid branch identity")
        branch_id = (branch["seed"], branch.get("policy"))
        require(branch_id[0] in SEEDS and branch_id[1] in POLICIES and branch_id not in indexed,
                "duplicate/unknown branch")
        rows = branch.get("metrics")
        require(type(rows) is list and len(rows) == len(STEPS), "expected all 22 scheduled states")
        for step, row in zip(STEPS, rows):
            require(type(row) is dict and type(row.get("step")) is int and row["step"] == step,
                    "changed evaluation schedule")
            for panel in PANELS:
                stats = row.get(panel)
                require(type(stats) is dict and stats.get("count") == 5000, "heldout scalar panel schema")
                accuracy = finite_number(stats.get("accuracy"), "accuracy")
                ce = finite_number(stats.get("ce"), "CE")
                require(0 <= accuracy <= 1 and ce >= 0, "metric outside its domain")
        indexed[branch_id] = rows
    require(set(indexed) == {(seed, policy) for seed in SEEDS for policy in POLICIES}, "incomplete paired roster")
    return indexed


def series(indexed, policy, panel, metric):
    return [[float(row[panel][metric]) for row in indexed[seed, policy]] for seed in SEEDS]


def png_bytes(figure, audit_sha, title):
    target = io.BytesIO()
    figure.savefig(target, format="png", dpi=150, facecolor="white", edgecolor="white",
                   metadata={"Title": title, "AuditSHA256": audit_sha,
                             "Description": "Audited scalar records; all three paired seeds; fixed scheduled states."})
    payload = target.getvalue()
    require(len(payload) <= PNG_CAP, "figure exceeds 8 MiB cap")
    return payload


def render(indexed, audit_sha):
    """No archive imports or reads; arithmetic means of audited scalars only."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import numpy as np

    style = {"font.family": "DejaVu Sans", "font.size": 11,
             "axes.titlesize": 13, "axes.labelsize": 11,
             "axes.facecolor": "white", "figure.facecolor": "white",
             "axes.spines.top": False, "axes.spines.right": False,
             "axes.edgecolor": "#B0B7BF", "axes.labelcolor": "#26313C",
             "xtick.color": "#46515D", "ytick.color": "#46515D",
             "axes.axisbelow": True, "savefig.transparent": False}
    with plt.rc_context(style):
        fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.3))
        fig.subplots_adjust(left=.075, right=.985, bottom=.18, top=.73, wspace=.22)
        for axis, metric, title, ylabel in zip(
                axes, ("accuracy", "ce"), ("Accuracy — higher is better", "Cross-entropy — lower is better"),
                ("Heldout accuracy (%)", "Heldout cross-entropy (nats)")):
            for policy, color, label in zip(POLICIES, COLORS, LABELS):
                values = np.asarray(series(indexed, policy, "heldout", metric), dtype=np.float64)
                if metric == "accuracy":
                    values = values * 100
                for seed_curve in values:
                    axis.plot(STEPS, seed_curve, color=color, alpha=.22, linewidth=1.05)
                axis.plot(STEPS, values.mean(axis=0), color=color, linewidth=2.5,
                          marker="o", markersize=2.5, label=label)
            axis.axvline(100, color="#697580", linestyle="--", linewidth=1.1, alpha=.8)
            axis.set(xlim=(0, 4000), xlabel="Optimizer updates", ylabel=ylabel, title=title)
            axis.set_xticks((0, 1000, 2000, 3000, 4000))
            axis.grid(axis="y", color="#CBD2DA", alpha=.55, linewidth=.65)
            if metric == "accuracy":
                axis.set_ylim(0, 100)
            else:
                axis.set_ylim(bottom=0)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.53, .925),
                   ncol=2, frameon=False, columnspacing=2.2, handlelength=2.7, fontsize=10.5)
        fig.suptitle("Clean multiview training: original heldout images", y=.99, fontsize=17)
        fig.text(.5, .055, "Faint lines: 3 paired seeds   •   Solid lines: means   •   Dashed line: warmup ends at update 100",
                 ha="center", color="#56616D", fontsize=10)
        curves = png_bytes(fig, audit_sha, "Original-heldout learning curves, all 22 states")
        plt.close(fig)

        fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.9), sharey="row")
        fig.subplots_adjust(left=.075, right=.985, bottom=.14, top=.84, hspace=.34, wspace=.15)
        xpos = np.arange(len(POLICIES), dtype=np.float64)
        offsets = (-.15, -.03, .09)
        for row_index, metric in enumerate(("accuracy", "ce")):
            row_values = []
            for column, panel in enumerate(PANELS):
                axis = axes[row_index, column]
                values = np.asarray([[indexed[seed, policy][-1][panel][metric]
                                      for policy in POLICIES] for seed in SEEDS], dtype=np.float64)
                if metric == "accuracy":
                    values = values * 100
                row_values.extend(values.ravel().tolist())
                for seed_index, (offset, marker) in enumerate(zip(offsets, SEED_MARKERS)):
                    axis.plot(xpos + offset, values[seed_index], color="#7C8793", alpha=.42,
                              linewidth=.9, zorder=1)
                    for method_index, color in enumerate(COLORS):
                        axis.scatter(xpos[method_index] + offset, values[seed_index, method_index],
                                     marker=marker, s=37, color=color, edgecolors="white", linewidths=.5,
                                     zorder=3)
                axis.scatter(xpos + .25, values.mean(axis=0), color="#1F2933", marker="D", s=40,
                             edgecolors="white", linewidths=.5, zorder=4)
                axis.set_xticks(xpos, SHORT_LABELS, fontsize=9.5)
                axis.set_xlim(-.40, 3.50)
                axis.grid(axis="y", color="#CBD2DA", alpha=.55, linewidth=.65)
                if row_index == 0:
                    axis.set_title("Original heldout images" if column == 0 else "Translated heldout images", pad=10)
                if column == 0:
                    axis.set_ylabel("Accuracy (%) — higher is better" if metric == "accuracy"
                                    else "Cross-entropy (nats) — lower is better")
            # Identical y-limits across the two readouts for each metric; neither
            # axis is inverted. Dot plots may zoom without implying a bar origin.
            minimum, maximum = min(row_values), max(row_values)
            padding = max((maximum - minimum) * .13, 1.0 if metric == "accuracy" else .02)
            axes[row_index, 0].set_ylim(max(0., minimum - padding),
                                       min(100., maximum + padding) if metric == "accuracy" else maximum + padding)
        seed_handles = [Line2D([], [], marker=marker, linestyle="none", color="#697580", markersize=6,
                               label="Seed " + str(seed)[-3:]) for seed, marker in zip(SEEDS, SEED_MARKERS)]
        mean_handle = Line2D([], [], marker="D", linestyle="none", color="#1F2933", markersize=6,
                             label="3-seed mean")
        fig.legend(handles=seed_handles + [mean_handle], loc="upper center", bbox_to_anchor=(.52, .93),
                   ncol=4, frameon=False, columnspacing=2.5, fontsize=10.5)
        fig.suptitle("Fixed endpoint: update 4,000", y=.985, fontsize=17)
        fig.text(.5, .045, "Thin lines join the same seed   •   Diamonds: arithmetic means   •   Endpoint axes zoomed; no epoch selection",
                 ha="center", color="#56616D", fontsize=10)
        endpoints = png_bytes(fig, audit_sha, "Paired update-4000 original and translated heldout endpoints")
        plt.close(fig)
    return {CURVES_NAME: curves, ENDPOINTS_NAME: endpoints}, matplotlib.__version__


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="read this directory's PASS audit and create new PNGs")
    parser.add_argument("--audit-sha256", help="optional exact SHA-256 of the approved audit.json")
    args = parser.parse_args(argv)
    if not args.execute:
        print("Prepared only. No input read or output written; use --execute only after audit PASS.")
        return 0
    require(HERE.name == DIRECTORY_NAME, "script must stay in its assigned report directory")
    targets = [HERE / name for name in (CURVES_NAME, ENDPOINTS_NAME, MANIFEST_NAME)]
    require(all(not path.exists() and not path.is_symlink() for path in targets),
            "exclusive outputs required; an existing plot or manifest must not be overwritten")
    audit_path = HERE / "audit.json"
    require(audit_path.is_file() and not audit_path.is_symlink(), "regular local audit.json required")
    require(audit_path.stat().st_size <= INPUT_CAP, "audit exceeds 32 MiB input cap")
    with audit_path.open("rb") as handle:
        payload = handle.read(INPUT_CAP + 1)
    require(len(payload) <= INPUT_CAP, "audit grew beyond input cap")
    audit_sha = sha256(payload)
    if args.audit_sha256 is not None:
        require(args.audit_sha256 == audit_sha, "approved audit digest differs")
    audit = json.loads(payload)
    indexed = validate_audit(audit)
    figures, version = render(indexed, audit_sha)
    manifest = {"schema": "spectral_multiview_scalar_plots_v1", "status": "complete",
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "input": {"path": "audit.json", "size_bytes": len(payload), "sha256": audit_sha,
                          "schema": audit["schema"], "status": audit["status"]},
                "input_results_sha256": audit.get("input_results_sha256"),
                "source_commit": audit.get("source_commit"),
                "script_sha256": sha256(Path(__file__).read_bytes()),
                "matplotlib_version": version, "dpi": 150,
                "seeds": list(SEEDS), "policies": list(POLICIES), "steps": list(STEPS),
                "outputs": [{"path": name, "size_bytes": len(value), "sha256": sha256(value)}
                            for name, value in figures.items()],
                "scope": "Audited scalar plots only; no training, model/array reads, or audit replay."}
    # Render both images in memory before writing. Files are never overwritten;
    # a failed partial write remains visible and does not trigger an auto-retry.
    for name, value in figures.items():
        with (HERE / name).open("xb") as handle:
            handle.write(value)
    with (HERE / MANIFEST_NAME).open("xb") as handle:
        handle.write((json.dumps(manifest, indent=2, allow_nan=False) + "\n").encode())
    print(json.dumps({"status": "complete", "input_sha256": audit_sha,
                      "outputs": [item["path"] for item in manifest["outputs"]]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
