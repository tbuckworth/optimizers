#!/usr/bin/env python3
"""Plot every saved-state trajectory from the verified representation summary."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ARMS = ("adamw", "legacy", "stable")
LABELS = {"adamw": "AdamW", "legacy": "Legacy filter", "stable": "Stable filter"}
COLORS = {"adamw": "#54616e", "legacy": "#c46b24", "stable": "#157e81"}


def field(row, path):
    for key in path.split("."):
        row = row[key]
    return np.nan if row is None else float(row)


def curves(ax, rows, path, *, ylim=None, log=False):
    steps = sorted({r["step"] for r in rows})
    for arm in ARMS:
        matrix = np.array([[field(next(r for r in rows if
                           (r["seed"], r["arm"], r["step"]) == (seed, arm, step)), path)
                           for step in steps] for seed in range(100, 105)])
        mean = np.nanmean(matrix, axis=0)
        count = np.sum(np.isfinite(matrix), axis=0)
        se = np.nanstd(matrix, axis=0, ddof=1) / np.sqrt(count)
        ax.plot(steps, mean, marker="o", markersize=3, color=COLORS[arm], label=LABELS[arm])
        ax.fill_between(steps, mean - se, mean + se, color=COLORS[arm], alpha=.12)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if log:
        ax.set_yscale("log")
    ax.set_xlabel("Training updates (saved checkpoints)")
    ax.grid(alpha=.15)
    ax.spines[["top", "right"]].set_visible(False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Use a new plot directory; do not overwrite a prior rendering.")
    summary = json.loads(args.summary.read_text())
    rows = summary["rows"]
    keys = [(r["seed"], r["arm"], r["step"]) for r in rows]
    steps = (0, 100, 500, 1000, 1500, 2000, 2500, 3000, 4000, 6000)
    expected = {(seed, arm, step) for seed in range(100, 105) for arm in ARMS for step in steps}
    if len(keys) != 150 or set(keys) != expected:
        raise ValueError("Plot only the complete 150-state roster.")
    args.output.mkdir(parents=True)
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 12, "figure.facecolor": "white"})

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    panels = [
        ("probes.final_hidden.selected_eval_mean_r2", "Readable modular-sum information", "Held-out probe R²", (-.12, 1.05), False),
        ("behavior.test.accuracy", "Actual held-out predictions", "Accuracy", (-.03, 1.05), False),
        ("behavior.test.correct_class_margin_mean", "Correct-class margin", "True logit minus largest alternative", None, False),
        ("behavior.test.loss", "Held-out cross-entropy", "Cross-entropy (log scale)", None, True),
    ]
    for ax, (path, title, ylabel, ylim, log) in zip(axes.flat, panels):
        curves(ax, rows, path, ylim=ylim, log=log)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
    axes[0, 0].legend(frameon=False, fontsize=9)
    fig.suptitle("Rule information and behavior along the completed grokking runs\n"
                 "Five paired seeds · mean ± standard error · observational, not causal", fontsize=14)
    fig.savefig(args.output / "representation-and-behavior.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 5, figsize=(16, 6), sharex=True, sharey="row", constrained_layout=True)
    for column, seed in enumerate(range(100, 105)):
        for arm in ARMS:
            selected = sorted((r for r in rows if (r["seed"], r["arm"]) == (seed, arm)), key=lambda r:r["step"])
            for i, path in enumerate(("probes.final_hidden.selected_eval_mean_r2", "behavior.test.accuracy")):
                axes[i, column].plot([r["step"] for r in selected], [field(r,path) for r in selected],
                                     color=COLORS[arm], marker="o", markersize=3, label=LABELS[arm])
        axes[0, column].set_title(f"Seed {seed}")
        axes[1, column].set_xlabel("Training updates")
        for ax in axes[:, column]:
            ax.set_ylim(-.12, 1.05)
            ax.grid(alpha=.15)
            ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].set_ylabel("Held-out probe R²")
    axes[1, 0].set_ylabel("Held-out accuracy")
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("All five paired trajectories; no seed selection\n"
                 "Lines connect recorded states; sparse checkpoints cannot locate an exact transition", fontsize=14)
    fig.savefig(args.output / "individual-seeds.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for ax, path, title, ylabel in [
        (axes[0,0], "probes.final_hidden.fixed_panel_eval_mean_r2", "Same five calibration-selected frequencies", "Held-out probe R²"),
        (axes[0,1], "symmetry.heldout_shift_pooled.correct.value", "Correct-shift output symmetry", "Normalized defect (lower = more equivariant)"),
        (axes[1,0], "symmetry.training_membership_pooled.excess", "Training-membership symmetry difference", "Matched TH defect − HH defect"),
        (axes[1,1], "symmetry.centered_logit_rms.test", "Held-out centered-logit magnitude", "Logit RMS"),
    ]:
        curves(ax, rows, path)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
    axes[0,0].legend(frameon=False, fontsize=9)
    fig.suptitle("Supporting diagnostics and their limits\n"
                 "Symmetry can change with temperature; it is not a memorization-circuit measurement", fontsize=14)
    fig.savefig(args.output / "supporting-diagnostics.png", dpi=160)
    plt.close(fig)

    # Transparent calibration controls: true vs shuffled targets at every state.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    curves(axes[0], rows, "probes.pre_attention.selected_eval_mean_r2")
    curves(axes[1], rows, "probes.final_hidden.null_max_eval_mean_r2")
    axes[0].set_title("Additive pre-attention control")
    axes[1].set_title("Largest of 20 shuffled-row probe scores")
    for ax in axes:
        ax.set_ylabel("Held-out probe R² (negative values retained)")
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle("Readout controls · five-seed means ± standard errors", fontsize=14)
    fig.savefig(args.output / "probe-controls.png", dpi=160)
    plt.close(fig)
    receipt = {"summary_path": str(args.summary.resolve()),
               "summary_sha256": hashlib.sha256(args.summary.read_bytes()).hexdigest(),
               "plotter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "plots": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(args.output.glob("*.png"))},
               "description": "All 150 states, five-seed mean/sample SE and all-seed panels; no interpolation claim."}
    (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
