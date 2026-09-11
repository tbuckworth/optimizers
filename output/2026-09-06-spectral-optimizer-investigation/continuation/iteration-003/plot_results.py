#!/usr/bin/env python3
"""Descriptive plots of every scheduled validation point; no selection changes."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np

HERE = Path(__file__).resolve().parent


def main():
    execution = json.loads((HERE / "results/execution.json").read_text())
    assert execution["status"] == "complete" and execution["completed_runs"] == 18
    arms = [("adamw", "AdamW", "#343f4c"),
            ("estimate32_project32", "Estimate 32 / project 32", "#28759b"),
            ("estimate128_project32", "Estimate 128 / project 32", "#c66e32")]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 7.0), sharex=True)
    for column, noise in enumerate((0., .9)):
        for arm, label, color in arms:
            runs = [json.loads((HERE / "results" / f"seed{seed}-noise{noise:g}-{arm}.json").read_text())
                    for seed in (0, 1, 2)]
            steps = [row["step"] for row in runs[0]["validation_trajectory"]]
            assert steps == list(range(0, 2001, 100))
            for row, field in enumerate(("accuracy", "cross_entropy")):
                values = np.array([[v[field] for v in run["validation_trajectory"]] for run in runs])
                axis = axes[row, column]
                axis.fill_between(steps, values.min(0), values.max(0), color=color, alpha=.13, linewidth=0)
                axis.plot(steps, values.mean(0), color=color, linewidth=2, label=label)
                axis.grid(alpha=.15)
                axis.axvline(100, color="#888888", linestyle=":", linewidth=.8)
        axes[0, column].set_title("Clean training labels" if noise == 0 else "90% uniform label replacement")
        axes[0, column].yaxis.set_major_formatter(PercentFormatter(1.))
        axes[1, column].set_xlabel("Optimizer steps")
    axes[0, 0].set_ylabel("Clean validation accuracy")
    axes[1, 0].set_ylabel("Clean validation cross-entropy")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, .025), ncol=3, frameon=False)
    fig.suptitle("Frozen MNIST measurement: clean-data cost and noisy-label preservation", y=.98)
    fig.text(.5, .008, "Three paired seeds; bands show seed ranges, not confidence intervals. Dotted line: end of warmup.",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .08, 1, .94))
    fig.savefig(HERE / "validation-curves.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
