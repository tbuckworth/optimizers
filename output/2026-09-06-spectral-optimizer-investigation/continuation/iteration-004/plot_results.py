#!/usr/bin/env python3
"""Display all scheduled validation observations; never change selectors."""
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
    assert execution["status"] == "complete" and execution["completed_runs"] == 12
    output = HERE / "validation-curves.png"
    if output.exists():
        raise SystemExit("Refusing to overwrite a figure")
    arms = [("adamw", "AdamW", "#343f4c"),
            ("estimate32_project32", "Estimate 32 / project 32", "#28759b"),
            ("estimate128_project32", "Estimate 128 / project 32", "#c66e32"),
            ("scalar32_norm", "Scalar norm control", "#80649a")]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.9), sharex=True)
    for arm, label, color in arms:
        runs = [json.loads((HERE / "results" / f"seed{seed}-{arm}.json").read_text()) for seed in (3, 4, 5)]
        steps = [row["step"] for row in runs[0]["validation_trajectory"]]
        assert steps == list(range(0, 2001, 100))
        for axis, field in zip(axes, ("accuracy", "cross_entropy")):
            values = np.array([[v[field] for v in run["validation_trajectory"]] for run in runs])
            axis.fill_between(steps, values.min(0), values.max(0), color=color, alpha=.12, linewidth=0)
            axis.plot(steps, values.mean(0), color=color, linewidth=2, label=label)
    for axis in axes:
        axis.grid(alpha=.15)
        axis.axvline(100, color="#888888", linestyle=":", linewidth=1)
        axis.set_xlabel("Optimizer steps")
    axes[0].set_ylabel("Clean validation accuracy")
    axes[0].yaxis.set_major_formatter(PercentFormatter(1.))
    axes[1].set_ylabel("Clean validation cross-entropy")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, .045), ncol=2, frameon=False)
    fig.suptitle("Fresh-seed MNIST comparison: 90% uniform label replacement", y=.98)
    fig.text(.5, .012, "Three paired seeds; bands are seed ranges, not confidence intervals. Dotted line: shared warmup ends.",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .15, 1, .94))
    fig.savefig(output, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
