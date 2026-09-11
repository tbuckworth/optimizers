#!/usr/bin/env python3
"""Plot all eight retained order seeds; no training or checkpoint reselection."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent


def main():
    target = HERE / "learning-curves-full-scale.png"
    if target.exists():
        raise SystemExit("Plot exists; do not overwrite this report artifact.")
    records = [json.loads(line) for line in (HERE / "artifacts/trajectories.jsonl").read_text().splitlines()]
    assert len(records) == 512
    assert (HERE / "artifacts/completion.json").is_file()
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    colors = {"raw": "#555555", "live": "#0072B2", "frozen": "#009E73", "oracle": "#CC79A7"}
    panels = [("sgd", "useful", 30), ("sgd", "nuisance", 30),
              ("adam", "useful", 0), ("adam", "useful", 30)]
    with np.load(HERE / "artifacts/clean_risk_curves.npz", allow_pickle=False) as curves:
        for ax, (opt, design, angle) in zip(axes.flat, panels):
            maximum_risk = .5
            for arm in colors:
                paths = np.stack([curves[f"{opt}-{design}-a{angle}-s{seed}-{arm}"] for seed in range(8)])
                x = np.arange(paths.shape[1])
                maximum_risk = max(maximum_risk, float(paths.max()))
                ax.plot(x, paths.mean(0), color=colors[arm], label=arm, lw=1.6)
                ax.fill_between(x, paths.min(0), paths.max(0), color=colors[arm], alpha=.12, linewidth=0)
                if arm == "raw":
                    ax.axhline(paths.min(1).mean(), color=colors[arm], lw=.8, ls=":",
                               label="mean of raw best stops")
            ax.axvline(100, color="#AAAAAA", lw=.7, ls="--")
            ax.set_title(f"{opt.upper()} | {design} variation | {angle}° coordinates")
            ax.set_ylim(0, maximum_risk * 1.08)
            ax.grid(alpha=.16)
            ax.set_xlabel("Training steps")
            ax.set_ylabel("Clean fixed-design risk (lower is better)")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False)
    fig.suptitle("Actual-filter selective learning: favorable construction and failure controls")
    fig.text(.5, .065, "Lines: means; bands: full range of 8 batch-order seeds. Oracle knows the useful direction.",
             ha="center", fontsize=9)
    fig.patch.set_facecolor("white")
    fig.tight_layout(rect=(0, .09, 1, .95))
    fig.savefig(target, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(target)


if __name__ == "__main__":
    main()
