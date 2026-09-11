#!/usr/bin/env python3
"""Descriptive display of all frozen snapshots; no new analysis or selection."""
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
SUMMARY_SHA = "f7a26b44d98f97ce5bc7347155b63b9239dbb46eab68ad7a79517aa547026157"


def render(summary):
    rows = {(r["seed"], r["step"]): r for r in summary["raw_snapshot_records"]}
    seeds, steps = (3, 4, 5), (200, 500, 1000, 2000)
    assert len(rows) == 12 and summary["execution_status"] == "complete"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(11.6, 8.0), sharex=True)
    widths = (("width32", "Store 32 / inspect 32", "#28759b"),
              ("width128", "Store 128 / inspect 32", "#c66e32"))
    panels = ((axes[0, 0], "span_energy_fraction", lambda x: 100 * (1 - x)),
              (axes[0, 1], "relative_covariance_error", lambda x: x),
              (axes[1, 0], "native_clean_minus_corruption_retention", lambda x: 100 * x))
    for width, label, color in widths:
        for axis, metric, transform in panels:
            values = transform(np.array([[rows[seed, step]["observers"][width]["metrics"][metric]
                                          for step in steps] for seed in seeds]))
            axis.fill_between(steps, values.min(0), values.max(0), color=color, alpha=.12, linewidth=0)
            axis.plot(steps, values.mean(0), marker="o", color=color, linewidth=2, label=label)
        for metric, style, marker, phase in (("native_current_gradient_retention", "-", "o", "after observing current g"),
                                             ("native_previous_current_gradient_retention", "--", "s", "previous basis")):
            values = np.array([[rows[seed, step]["observers"][width]["metrics"][metric]
                                for step in steps] for seed in seeds]) * 100
            axes[1, 1].plot(steps, values.mean(0), style, marker=marker, color=color,
                           linewidth=2, label=f"{width[5:]}: {phase}")
    axes[0, 0].set(title="Matched rank: missed optimal covariance energy", ylabel="100 × (1 − captured / optimal rank-32 energy)")
    axes[0, 0].set_ylim(bottom=0)
    axes[0, 1].set(title="Full covariance: unequal storage capacity", ylabel="Relative Frobenius error")
    axes[0, 1].set_ylim(bottom=0)
    axes[1, 0].set(title="Independent probes: clean minus fixed corruption", ylabel="Retention-energy gap (percentage points)")
    axes[1, 0].axhline(0, color="#888888", linewidth=.8)
    axes[1, 1].set(title="Current gradient: before / after its inclusion", ylabel="Current-gradient energy retained (%)", ylim=(0, 100))
    axes[1, 1].legend(loc="lower right", fontsize=8, frameon=False)
    for axis in axes.flat:
        axis.grid(alpha=.15)
        axis.set_xticks(steps)
        axis.set_xlabel("AdamW step (same passive replay)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, .038), ncol=2, frameon=False)
    fig.suptitle("Same gradients, different covariance estimates — three reused streams", y=.99)
    fig.text(.5, .018, "All four prescribed snapshots. Lines: seed means; bands: seed ranges, not confidence intervals. No new test evaluation.",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .09, 1, .955))
    return fig


def main():
    source = HERE / "summary.json"
    assert hashlib.sha256(source.read_bytes()).hexdigest() == SUMMARY_SHA
    output = HERE / "common-stream-diagnostics.png"
    manifest = HERE / "figure-manifest.json"
    if output.exists() or manifest.exists():
        raise SystemExit("Refusing to overwrite an existing figure/manifest")
    fig = render(json.loads(source.read_text()))
    fig.savefig(output, dpi=160)
    plt.close(fig)
    data = {"summary_sha256": SUMMARY_SHA, "plot_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "output": output.name, "size_bytes": output.stat().st_size,
            "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            "scope": "Descriptive rendering of every frozen snapshot; no new metric selection or inference."}
    with manifest.open("x") as handle:
        json.dump(data, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(data))


if __name__ == "__main__":
    main()
