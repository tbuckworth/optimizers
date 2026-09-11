#!/usr/bin/env python3
"""Render precomputed JSON summaries only; no scientific arrays or reanalysis."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import time

START = time.monotonic()
HERE = Path(__file__).resolve().parent
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["MPLCONFIGDIR"] = str(HERE / "figure-mplconfig")
resource.setrlimit(resource.RLIMIT_AS, (1024**3, 1024**3))
resource.setrlimit(resource.RLIMIT_CPU, (120, 120))
os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
signal.alarm(120)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

METRICS_SHA = "f849489f3d96752033500b66e1606a89caa91552a5a08ebe34d872ed183ee7ec"
CELLS = ("clean", "diffuse", "shared", "sham")
POLICIES = ("raw", "native32", "norm_raw")
COLORS = {"raw": "#087f8c", "native32": "#c45a20", "norm_raw": "#6867ad"}
LABELS = {"raw": "Raw AdamW", "native32": "Native spectral", "norm_raw": "Own-norm raw"}
SEED_MARKERS = ("o", "s", "^")
SEED_OFFSETS = (-0.045, 0.0, 0.045)
POLICY_OFFSETS = {"raw": -0.24, "native32": 0.0, "norm_raw": 0.24}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    input_names = ("metrics.json", "provenance.json", "main-readback.json")
    inputs = {name: json.loads((HERE / name).read_text()) for name in input_names}
    assert digest(HERE / "metrics.json") == METRICS_SHA
    assert inputs["provenance.json"]["overall_status"] == "PARTIAL"
    assert inputs["provenance.json"]["terminal_status"] == "RESOURCE_FOOTER_FAILED"
    assert inputs["main-readback.json"]["status"] == "TABLE_READBACK_PASS"
    assert inputs["main-readback.json"]["metrics_sha256"] == METRICS_SHA
    summaries = {(r["cell"], r["policy"], r["step"], r["view"]): r["metrics"]
                 for r in inputs["metrics.json"]["summaries"]}

    def saved(cell, policy, metric, step=2000, view="original"):
        return summaries[(cell, policy, step, view)][metric]

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "figure.facecolor": "white", "axes.facecolor": "white",
                         "savefig.facecolor": "white", "axes.spines.top": False,
                         "axes.spines.right": False, "axes.edgecolor": "#a4aab0",
                         "axes.labelcolor": "#26313a", "text.color": "#26313a",
                         "xtick.color": "#46525c", "ytick.color": "#46525c"})

    # Figure 1: no uncertainty calculation; every displayed value is saved JSON.
    fig, ax = plt.subplots(figsize=(10.0, 5.3))
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.21, top=0.73)
    fig.text(0.085, 0.95, "Rare-class ranking improves in three cells", fontsize=17, weight="bold")
    fig.text(0.085, 0.902, "Saved unpatched predictions · endpoints at update 2000 · three paired seeds", fontsize=10.5)
    warmup = saved("clean", "raw", "rare_auroc", step=100)
    ax.axhline(warmup["mean"], color="#78818a", linestyle="--", linewidth=1.2, zorder=1)
    ax.axhline(0.5, color="#c9ced3", linestyle=":", linewidth=1.0, zorder=1)
    for k, (value, marker) in enumerate(zip(warmup["seed_values"], SEED_MARKERS)):
        ax.scatter(SEED_OFFSETS[k], value, marker=marker, s=45, color="#5e6872", edgecolors="white", linewidths=0.6, zorder=4)
    ax.plot([-0.095, 0.095], [warmup["mean"]] * 2, color="#454f59", linewidth=2.5, zorder=3)
    for cell_i, cell in enumerate(CELLS, start=1):
        for policy in POLICIES:
            point = saved(cell, policy, "rare_auroc")
            x = cell_i + POLICY_OFFSETS[policy]
            for k, (value, marker) in enumerate(zip(point["seed_values"], SEED_MARKERS)):
                ax.scatter(x + SEED_OFFSETS[k], value, marker=marker, s=45,
                           color=COLORS[policy], edgecolors="white", linewidths=0.6, zorder=4)
            ax.plot([x - 0.095, x + 0.095], [point["mean"]] * 2,
                    color=COLORS[policy], linewidth=2.5, zorder=3)
    ax.set(xlim=(-0.45, 4.5), ylim=(0, 1.025), ylabel="Rare digit 8 vs rest: AUROC",
           xticks=range(5), xticklabels=["Warmup\nupdate 100", "Clean", "Diffuse", "Shared", "Sham"])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.grid(axis="y", color="#edf0f2", linewidth=0.8)
    ax.set_axisbelow(True)
    policy_handles = [Line2D([0], [0], color=COLORS[p], marker="o", linewidth=2, markersize=5, label=LABELS[p]) for p in POLICIES]
    policy_handles.append(Line2D([0], [0], color="#78818a", linestyle="--", label="Warmup mean"))
    fig.legend(handles=policy_handles, loc="upper left", bbox_to_anchor=(0.078, 0.86),
               ncol=4, frameon=False, fontsize=9.5, handlelength=1.8, columnspacing=1.8)
    seed_handles = [Line2D([0], [0], color="#596570", marker=m, linestyle="none", markersize=5,
                           label=f"Seed {s}") for s, m in zip(inputs["metrics.json"]["seed_order"], SEED_MARKERS)]
    fig.legend(handles=seed_handles, loc="upper left", bbox_to_anchor=(0.078, 0.805),
               ncol=3, frameon=False, fontsize=8.8, columnspacing=1.6)
    fig.text(0.085, 0.115, "Short bars: saved means. The fixed +log(11) shift leaves every AUROC unchanged.", fontsize=9.5)
    fig.text(0.085, 0.062, "POST-HOC · PARTIAL / RESOURCE_FOOTER_FAILED · original process resource certification missing", fontsize=8.7, color="#77551d")
    first = HERE / "rare-auroc.png"
    assert not first.exists()
    fig.savefig(first, dpi=150)
    plt.close(fig)

    # Figure 2: native seed pairs plus SAME-adjusted control means; no fitted values.
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.1), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.17, top=0.755, hspace=0.30, wspace=0.18)
    fig.text(0.09, 0.955, "Boosting digit 8 changes the rare/common tradeoff", fontsize=16.5, weight="bold")
    fig.text(0.09, 0.913, "Native seed pairs: original → fixed +log(11). Control means use the same adjustment.", fontsize=10)
    handles = [Line2D([0], [0], marker="o", color=COLORS["native32"], markerfacecolor="white", linestyle="none", markersize=6, label="Native: original"),
               Line2D([0], [0], marker="o", color=COLORS["native32"], linestyle="none", markersize=6, label="Native: +log(11)"),
               Line2D([0], [0], marker="D", color=COLORS["raw"], linestyle="none", markersize=6, label="Adjusted raw mean"),
               Line2D([0], [0], marker="x", color=COLORS["norm_raw"], linestyle="none", markersize=7, markeredgewidth=1.8, label="Adjusted own-norm raw mean")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.083, 0.864),
               ncol=2, frameon=False, fontsize=9.1, columnspacing=2.0)
    for ax, cell in zip(axes.flat, CELLS):
        original_x = saved(cell, "native32", "common_accuracy")["seed_values"]
        original_y = saved(cell, "native32", "rare_accuracy")["seed_values"]
        adjusted_x = saved(cell, "native32", "common_accuracy", view="plus_log11")["seed_values"]
        adjusted_y = saved(cell, "native32", "rare_accuracy", view="plus_log11")["seed_values"]
        for before_x, before_y, after_x, after_y, marker in zip(original_x, original_y, adjusted_x, adjusted_y, SEED_MARKERS):
            before = (100 * before_x, 100 * before_y)
            after = (100 * after_x, 100 * after_y)
            ax.annotate("", xy=after, xytext=before,
                        arrowprops={"arrowstyle": "->", "color": COLORS["native32"], "lw": 1.15, "alpha": 0.75}, zorder=2)
            ax.scatter(*before, marker=marker, s=42, facecolor="white", edgecolor=COLORS["native32"], linewidth=1.1, zorder=4, clip_on=False)
            ax.scatter(*after, marker=marker, s=42, color=COLORS["native32"], edgecolor="white", linewidth=0.5, zorder=4)
        for policy, marker in (("raw", "D"), ("norm_raw", "x")):
            x = 100 * saved(cell, policy, "common_accuracy", view="plus_log11")["mean"]
            y = 100 * saved(cell, policy, "rare_accuracy", view="plus_log11")["mean"]
            ax.scatter(x, y, s=65 if marker == "D" else 90, marker=marker,
                       color=COLORS[policy], linewidths=1.8 if marker == "x" else 0.7, zorder=5)
        ax.set_title(cell.title(), loc="left", fontsize=12, weight="bold", pad=7)
        ax.set(xlim=(0, 100), ylim=(0, 100))
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.grid(color="#edf0f2", linewidth=0.8)
        ax.set_axisbelow(True)
    for ax in axes[:, 0]:
        ax.set_ylabel("Rare accuracy (%)")
    for ax in axes[1, :]:
        ax.set_xlabel("Common accuracy (%)")
    fig.text(0.09, 0.076, "Post-hoc readout change; no fitted threshold, new training, calibrated correction or causal conclusion.", fontsize=9.1)
    fig.text(0.09, 0.038, "PARTIAL / RESOURCE_FOOTER_FAILED · original process resource certification missing", fontsize=8.7, color="#77551d")
    second = HERE / "rare-common-tradeoffs.png"
    assert not second.exists()
    fig.savefig(second, dpi=150)
    plt.close(fig)

    proc = {}
    for line in Path("/proc/self/status").read_text().splitlines():
        key, _, value = line.partition(":")
        if key in ("VmPeak", "VmSize", "VmHWM", "VmRSS", "Threads"):
            proc[key] = value.strip()
    assert int(proc["VmHWM"].split()[0]) * 1024 < 1024**3
    assert proc["Threads"] == "1"
    assert time.monotonic() - START < 120
    assert sum(p.stat().st_size for p in HERE.rglob("*") if p.is_file()) < 50 * 1024**2
    receipt = {"status": "RENDER_COMPLETE", "overall_scientific_status": "PARTIAL",
               "scientific_terminal_status": "RESOURCE_FOOTER_FAILED", "scientific_status_changed": False,
               "input_sha256": {name: digest(HERE / name) for name in input_names},
               "render_source_sha256": digest(__file__),
               "html_fragment": {"path": "email-fragment.html", "sha256": digest(HERE / "email-fragment.html")},
               "cid_mapping": {"rare-score-auroc": first.name, "rare-score-tradeoffs": second.name},
               "outputs": [{"path": p.name, "sha256": digest(p), "size_bytes": p.stat().st_size,
                            "dpi": 150, "background": "white"} for p in (first, second)],
               "elapsed_seconds": time.monotonic() - START, "render_proc_status": proc,
               "cpu_affinity": sorted(os.sched_getaffinity(0)),
               "matplotlib_version": matplotlib.__version__,
               "scope": "Rendering saved summary means/seed values only; no scientific source or logit reads, metric recomputation or scientific retry",
               "email_sent": False, "commit_created_by_renderer": False}
    with (HERE / "figure-receipt.json").open("x") as handle:
        json.dump(receipt, handle, indent=2)
        handle.write("\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
