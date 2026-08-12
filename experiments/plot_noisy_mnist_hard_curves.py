#!/usr/bin/env python3
"""Plot matched optimizer curves for the 90%-label-noise MNIST run."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


RUNS = (
    ("AdamW", "adamw_n90_s42.json", "#5B6472"),
    (
        "Global spectral (hard, r=200)",
        "global_stable_hard_r200_n90_s42.json",
        "#246EB9",
    ),
    (
        "Per-matrix spectral (hard, r=64)",
        "matrix_stable_hard_r64_n90_s42.json",
        "#E07A2D",
    ),
)


def load_run(path):
    with path.open() as handle:
        return json.load(handle)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/noisy_mnist_hard_curves/final"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/noisy_mnist_hard_curves/noisy_mnist_hard_curves.png"),
    )
    args = parser.parse_args()

    runs = [(label, load_run(args.results_dir / filename), color)
            for label, filename, color in RUNS]

    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(1, 2, figsize=(13.2, 5.1), sharex=True)
    for label, run, color in runs:
        epochs = [entry["epoch"] + 1 for entry in run["metrics"]]
        axes[0].plot(
            epochs,
            [entry["train_acc"] for entry in run["metrics"]],
            label=label,
            color=color,
            linewidth=2.2,
        )
        axes[1].plot(
            epochs,
            [entry["test_acc"] for entry in run["metrics"]],
            label=label,
            color=color,
            linewidth=2.2,
        )

    axes[0].set_title("Training accuracy (90% randomly relabeled)")
    axes[1].set_title("Test accuracy (clean labels)")
    for axis in axes:
        axis.set_xlabel("Epochs completed")
        axis.set_ylabel("Accuracy")
        axis.set_xlim(0, 60)
        axis.set_ylim(0, 0.9)
        axis.legend(loc="best", frameon=True)

    figure.suptitle(
        "Noisy MNIST: stable hard spectral filtering resists memorization",
        fontsize=15,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.01,
        "Seed 42 · AdamW lr=1e-3, weight decay=0.01 · decay=0.99 · warmup=100 steps",
        ha="center",
        fontsize=9,
        color="#4B5563",
    )
    figure.tight_layout(rect=(0, 0.04, 1, 0.94))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180, bbox_inches="tight")
    print(args.output)


if __name__ == "__main__":
    main()
