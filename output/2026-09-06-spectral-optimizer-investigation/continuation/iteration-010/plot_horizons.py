#!/usr/bin/env python3
"""Plot all predeclared I10 late-parent horizons, without selecting a winner."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.summary.open() as handle:
        summary = json.load(handle)
    assert summary["counts"]["physical_branches"] == 189
    rows = summary["mandatory_secondary_effects"]
    horizons = [0, 1, 10, 50, 100, 250, 500]
    colors = {"fixed": "#007F86", "soft": "#AC4E11", "redraw": "#6655A4"}
    labels = {"fixed": "Fixed corrupted labels", "soft": "Expected soft targets",
              "redraw": "Fresh label draws"}
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, sharey="row")
    for col, policy in enumerate(("current32", "frozen32")):
        for row_index, metric in enumerate(("ce", "accuracy")):
            ax = axes[row_index, col]
            for objective, color in colors.items():
                by_seed = []
                for seed in (100, 101, 102):
                    values = []
                    for horizon in horizons:
                        parents = [r for r in rows if r["source"] == "current32"
                                   and r["anchor"] in (1500, 2000)
                                   and r["horizon"] == horizon
                                   and r["policy_vs_raw"] == policy]
                        assert len(parents) == 2
                        effects = []
                        for parent in parents:
                            value = next(r for r in parent["per_seed"] if r["seed"] == seed)
                            effect = value[metric + "_fixed"]
                            if objective != "fixed":
                                effect -= value[metric + "_fixed_minus_" + objective]
                            effects.append(effect)
                        values.append(np.mean(effects) * (100 if metric == "accuracy" else 1))
                    by_seed.append(values)
                    ax.plot(horizons, values, color=color, alpha=.25, linewidth=1)
                ax.plot(horizons, np.mean(by_seed, axis=0), color=color, linewidth=2.4,
                        marker="o", markersize=3, label=labels[objective])
            ax.axhline(0, color="#444444", linewidth=.8, linestyle="--")
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", alpha=.18)
            if col == 0:
                ax.set_ylabel("Clean CE benefit" if metric == "ce" else "Clean accuracy benefit (pp)")
            if row_index == 0:
                ax.set_title("Continuously updated filter" if policy == "current32" else "Frozen parent subspace")
            else:
                ax.set_xlabel("New updates from inherited state")
    fig.suptitle("Does filtering help as learning continues?", fontsize=17, x=.5, y=.98)
    fig.text(.5, .925, "Above zero favors filtering over raw AdamW continuation", ha="center", fontsize=11)
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="lower center", ncol=3, bbox_to_anchor=(.5, .035), frameon=False)
    fig.text(.5, .012, "Three seeds: faint individual traces, bold mean; parent steps 1500/2000 averaged within seed.\n"
             "All policies inherit current-filter-trained weights and Adam state. Frozen panels are secondary.",
             ha="center", va="bottom", fontsize=9, color="#444444")
    fig.tight_layout(rect=(0, .10, 1, .91))
    if args.output.exists():
        raise FileExistsError(args.output)
    fig.savefig(args.output, dpi=160)
    plt.close(fig)
    print(str(args.output))


if __name__ == "__main__":
    main()
