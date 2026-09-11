#!/usr/bin/env python3
"""Render fixed plots from audited scalar summaries; never trains or reads tensors."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CELLS = ("clean", "shared", "sham")
MODES = ("none", "random", "targeted", "opposite")
COLORS = {"raw": "#778394", "native32": "#176B9B"}
LABELS = {"raw": "AdamW", "native32": "Spectral + AdamW"}


def traces(ax, rows, multiplier=1):
    for policy, records in rows.items():
        values = np.array([record["values"] for record in records], dtype=float) * multiplier
        if values.shape != (4, 3) or not np.isfinite(values).all():
            raise ValueError("four modes and three finite paired seed values required")
        x = np.arange(4)
        ax.plot(x, values.mean(1), "o-", color=COLORS[policy], lw=2,
                markersize=5, label=LABELS[policy])
        for seed, offset in enumerate((-.06, 0, .06)):
            ax.scatter(x + offset, values[:, seed], color=COLORS[policy],
                       s=15, alpha=.5, zorder=3)
    ax.set_xticks(range(4), ["None", "Random", "Targeted", "Opposite"], fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#DCE2E8", lw=.7)
    ax.set_axisbelow(True)


def save(fig, output, name, caption):
    fig.patch.set_facecolor("white")
    fig.text(.02, .012, caption, fontsize=9, color="#435164")
    path = output / name
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {"path": name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "size_bytes": path.stat().st_size}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    payload = args.summary.read_bytes()
    summary = json.loads(payload)
    if summary["schema"] != "spectral_augmentation_analysis_v1" or summary["endpoint_step"] != 2000:
        raise ValueError("unexpected analysis schema or endpoint")
    args.output_dir.mkdir(exist_ok=False)
    receipts = []
    for measure, title, filename, scale in (
        ("accuracy", "Does masking improve useful recognition?", "recognition.png", 100),
        ("ce", "What changes beneath the accuracy threshold?", "cross-entropy.png", 1),
    ):
        fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.7), constrained_layout=False)
        fig.subplots_adjust(top=.81, bottom=.14, hspace=.42, wspace=.29)
        fig.suptitle(title, x=.02, ha="left", fontsize=19, fontweight="bold", y=.98)
        fig.text(.02, .925, "Held-out original images · fixed step 2,000 · mean lines and all three seed points",
                 fontsize=11, color="#435164")
        for column, cell in enumerate(CELLS):
            for row, group in enumerate(("majority_macro", "rare")):
                ax = axes[row, column]
                key = f"heldout_unpatched/{group}/{measure}"
                traces(ax, {p: [summary["groups"][f"{cell}/{p}/{a}"][key] for a in MODES]
                            for p in COLORS}, scale)
                ax.set_title(cell.capitalize(), fontsize=13)
                prefix = "Common digits" if row == 0 else "Rare digit (8)"
                ax.set_ylabel(prefix + (" accuracy (%)" if measure == "accuracy" else " CE (lower is better)"), fontsize=9)
                ax.set_ylim(bottom=0)
                if measure == "accuracy":
                    ax.set_ylim(0, 100)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(.012, .905), ncol=2, frameon=False)
        receipts.append(save(fig, args.output_dir, filename,
            "Three paired seeds, not 72 independent replicates. Targeted uses the known cue location; evaluation is never masked."))
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.4))
    fig.subplots_adjust(top=.70, bottom=.22, wspace=.29)
    fig.suptitle("Does masking weaken the learned misleading cue?", x=.02, ha="left", fontsize=18,
                 fontweight="bold", y=.98)
    fig.text(.02, .885, "Same held-out images, with and without a visible cue · lower excess is less cue reliance",
             fontsize=10, color="#435164")
    key = "cue/majority_nonzero/patch_excess"
    for ax, cell in zip(axes[:2], ("shared", "sham")):
        traces(ax, {p: [summary["groups"][f"{cell}/{p}/{a}"][key] for a in MODES] for p in COLORS}, 100)
        ax.set_title(cell.capitalize() + ": patch excess", fontsize=12)
        ax.set_ylabel("Percentage points", fontsize=9)
        ax.axhline(0, color="#46566A", lw=.7)
    traces(axes[2], {p: [summary["cue_association"]["Q"][f"{p}/{a}"] for a in MODES] for p in COLORS}, 100)
    axes[2].set_title("Shared − Sham (Q)", fontsize=12)
    axes[2].set_ylabel("Percentage points", fontsize=9)
    axes[2].axhline(0, color="#46566A", lw=.7)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(.012, .84), ncol=2, frameon=False)
    receipts.append(save(fig, args.output_dir, "cue-reliance.png",
        "Primary cue set: true digits 1–7 and 9. Lower cue reliance is useful only alongside retained competence."))
    manifest = {"summary_path": str(args.summary), "summary_sha256": hashlib.sha256(payload).hexdigest(),
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "plots": receipts, "scope": "fixed endpoint scalar rendering; no acquisition or new inferential statistics"}
    with (args.output_dir / "manifest.json").open("x") as handle:
        json.dump(manifest, handle, indent=2)
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
