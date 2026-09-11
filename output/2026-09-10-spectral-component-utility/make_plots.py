#!/usr/bin/env python3
"""Render two descriptive figures from one pinned terminal audit JSON only.

No acquisition directory, NPZ, model, checkpoint or old plotter is opened.
Scalar endpoint means corroborate the audit summary used for plotting; this is
not a new scientific audit. Import is inert; matplotlib is imported on render.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
AUDIT_SHA256 = "2318d563e458e07251e822820fa7dd43d07f1edf6a9bc6e1f7d06bdb9f36f47e"
SEEDS = (202609171, 202609172, 202609173)
MODES = ("none", "translate")
STEPS = (100, 56304)
FRACTIONS = (("full", 1.0), ("tenth", 0.1))
OBJECTIVES = ("H_O", "C")
TINY = 1e-8
FILENAMES = ("primary-action-decreases.png", "all-parent-contrasts.png")
RAW_COLOR, NATIVE_COLOR = "#21808D", "#AD5267"
FULL_COLOR, TENTH_COLOR = "#3C628D", "#CD782A"


def need(condition, message):
    if not condition:
        raise ValueError(message)


def scalar(value):
    need(type(value) in (int, float) and math.isfinite(value), "finite plain scalar required")
    return float(value)


def load_pinned_audit():
    path = HERE / "audit.json"
    need(path.is_file() and not path.is_symlink() and path.stat().st_size <= 2 * 1024**2,
         "bounded regular audit JSON required")
    payload = path.read_bytes()
    need(hashlib.sha256(payload).hexdigest() == AUDIT_SHA256, "audit hash mismatch")
    return json.loads(payload)


def plot_data(audit):
    """Copy all selected audit scalars, corroborating their two-draw means."""
    need(audit["schema"] == "spectral_component_utility_audit_v1"
         and audit["status"] == "PASS" and audit["errors"] == [], "terminal PASS audit required")
    parents = audit["checked_parents"]
    expected = [(seed, mode, step) for seed in SEEDS for mode in MODES for step in STEPS]
    need([(p["seed"], p["augmentation"], p["step"]) for p in parents] == expected,
         "exact ordered twelve-parent audit roster")
    by_parent = {}
    for p in parents:
        key = (p["seed"], p["augmentation"], p["step"])
        need(p["parent_id"] == f"s{key[0]}-{key[1]}-h{key[2]:05d}", "parent ID binding")
        endpoints = p["endpoints"]
        expected_endpoints = [(batch, policy, fid, fraction)
                              for batch in (0, 1) for policy in ("raw", "native", "decay")
                              for fid, fraction in FRACTIONS]
        need([(e["batch"], e["policy"], e["fraction_id"], e["fraction"])
              for e in endpoints] == expected_endpoints, "endpoint batch/policy/path roster")
        by_parent[key] = {(e["batch"], e["policy"], e["fraction_id"]): e for e in endpoints}

    summary = audit["independent_summary"]
    cells = summary["cells"]
    expected_cells = [(mode, step, fid, fraction) for mode in MODES for step in STEPS
                      for fid, fraction in FRACTIONS]
    need([(c["augmentation"], c["step"], c["fraction_id"], c["fraction"])
          for c in cells] == expected_cells, "exact ordered eight summary cells")
    selected = {}
    count = 0
    for cell_index, cell in enumerate(cells):
        need([row["seed"] for row in cell["seed_rows"]] == list(SEEDS), "three separate ordered seeds")
        for seed_index, row in enumerate(cell["seed_rows"]):
            key = (row["seed"], cell["augmentation"], cell["step"])
            endpoints = by_parent[key]
            objectives = {}
            for objective in OBJECTIVES:
                values = row["objectives"][objective]
                raw_draws = [scalar(endpoints[b, "raw", cell["fraction_id"]]["effects"][objective]["finite"])
                             for b in (0, 1)]
                native_draws = [scalar(endpoints[b, "native", cell["fraction_id"]]["effects"][objective]["finite"])
                                for b in (0, 1)]
                contrasts = [native_draws[b] - raw_draws[b] for b in (0, 1)]
                rebuilt = {"raw_finite": math.fsum(raw_draws) / 2,
                           "native_finite": math.fsum(native_draws) / 2,
                           "contrast_finite": math.fsum(contrasts) / 2}
                for field, expected_value in rebuilt.items():
                    actual = scalar(values[field])
                    need(actual == expected_value, "audit endpoint/summary scalar mismatch")
                    count += 1
                objectives[objective] = {**rebuilt, "raw_draws": raw_draws,
                                         "native_draws": native_draws,
                                         "tiny_contrast": abs(rebuilt["contrast_finite"]) <= TINY}
            selected[key + (cell["fraction_id"],)] = {
                "fraction": cell["fraction"], "fraction_id": cell["fraction_id"],
                "source_pointer": f"/independent_summary/cells/{cell_index}/seed_rows/{seed_index}",
                "objectives": objectives}

    primary = summary["primary"]
    need({key: primary[key] for key in ("augmentation", "step", "fraction_id", "fraction")}
         == {"augmentation": "translate", "step": 56304, "fraction_id": "full", "fraction": 1.0},
         "registered primary cell")
    need([r["seed"] for r in primary["seed_rows"]] == list(SEEDS), "primary seed roster")
    primary_rows = []
    for row in primary["seed_rows"]:
        picked = selected[row["seed"], "translate", 56304, "full"]
        for objective in OBJECTIVES:
            for field in ("raw_finite", "native_finite", "contrast_finite"):
                need(scalar(row["objectives"][objective][field]) == picked["objectives"][objective][field],
                     "primary differs from full translated-final cell")
                count += 1
        primary_rows.append({"seed": row["seed"], "parent_id": f's{row["seed"]}-translate-h56304',
                             **copy.deepcopy(picked)})
    all_rows = [{"seed": seed, "augmentation": mode, "step": step,
                 "parent_id": f"s{seed}-{mode}-h{step:05d}",
                 "paths": {fid: copy.deepcopy(selected[seed, mode, step, fid]) for fid, _ in FRACTIONS}}
                for mode in MODES for step in STEPS for seed in SEEDS]
    return {"primary": primary_rows, "all_parents": all_rows,
            "scalar_mean_and_primary_checks": count,
            "tiny_finite_threshold_nats": TINY,
            "aggregation": "Arithmetic mean of the two draws within each parent; no across-parent or across-seed pool.",
            "definitions": {
                "decrease": "E_J = J(parent) - J(endpoint); positive means the named objective decreases.",
                "contrast": "D_J = E_J(native) - E_J(raw); positive means a larger decrease than raw, not necessarily an absolute decrease.",
                "H_O": "Original-image true-label cross-entropy on the fixed reporting panel.",
                "C": "View-inconsistency Jensen gap on the fixed training evaluation panel; not clean skill.",
                "tenth": "The actual materialized FP32 tenth path, evaluated separately; not full-path effects divided by ten.",
                "tiny": "Absolute finite contrast <= 1e-8 nats is visibly flagged, not rounded to zero or declared equivalent."}}


def _plotting():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter
    return plt, Line2D, FuncFormatter


def _style(ax):
    ax.set_facecolor("white")
    ax.spines[["top", "right"]].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#B8C0C7")
    ax.tick_params(labelsize=10, colors="#34424C")
    ax.set_axisbelow(True)


def render_primary(data):
    plt, _, _ = _plotting()
    fig, axes = plt.subplots(2, 2, figsize=(11.7, 8.0), dpi=150)
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0.085, right=0.975, bottom=0.15, top=0.81, wspace=0.27, hspace=0.48)
    fig.suptitle("One action at translated final states", x=0.085, y=0.985, ha="left", fontsize=20, weight="bold")
    fig.text(0.085, 0.929, "Three seeds • Full endpoints • Two paired action draws averaged within each seed", fontsize=11, color="#52616C")
    fig.text(0.085, 0.889, "Positive = objective decrease. The lower row isolates native − raw, on its own nats scale.", fontsize=11)
    titles = {"H_O": "Original-image clean CE  (H_O)", "C": "View inconsistency  (C)"}
    for col, objective in enumerate(OBJECTIVES):
        top, bottom = axes[0, col], axes[1, col]
        raw = [r["objectives"][objective]["raw_finite"] for r in data["primary"]]
        native = [r["objectives"][objective]["native_finite"] for r in data["primary"]]
        contrasts = [r["objectives"][objective]["contrast_finite"] for r in data["primary"]]
        top.bar([i - 0.18 for i in range(3)], raw, width=0.34, color=RAW_COLOR, label="Raw (filter bypassed)")
        top.bar([i + 0.18 for i in range(3)], native, width=0.34, color=NATIVE_COLOR, label="Native (filter retained)")
        top.set_title(titles[objective], fontsize=13, loc="left", pad=14)
        top.set_ylabel("Absolute objective decrease (nats)", fontsize=10)
        top.legend(frameon=False, fontsize=9, loc="upper left" if objective == "H_O" else "lower left")
        bottom.bar(range(3), contrasts, width=0.5, color=NATIVE_COLOR)
        bottom.axhspan(-TINY, TINY, color="#E9EBED", zorder=0)
        bottom.set_title("Native − raw: difference in decreases", fontsize=12, loc="left", pad=14)
        bottom.set_ylabel("Difference (nats)", fontsize=10)
        bound = max(abs(v) for v in contrasts) * 1.48 or TINY * 2
        bottom.set_ylim(-bound, bound)
        for i, value in enumerate(contrasts):
            bottom.annotate(f"{value:+.3e}", xy=(i, value), xytext=(0, 7 if value >= 0 else -7),
                            textcoords="offset points", ha="center", va="bottom" if value >= 0 else "top",
                            fontsize=10, color="#34424C")
            if abs(value) <= TINY:
                bottom.scatter([i], [value], marker="x", color="#17232B", s=45, zorder=5)
        for ax in (top, bottom):
            _style(ax)
            ax.axhline(0, color="#657581", linewidth=0.8)
            ax.set_xticks(range(3), [str(seed) for seed in SEEDS])
            ax.set_xlabel("Seed", fontsize=10)
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
            ax.grid(axis="y", alpha=0.16)
    fig.text(0.085, 0.068, "Negative C decreases mean C worsens — even when the native − raw contrast is positive.", fontsize=11)
    fig.text(0.085, 0.036, "C is not clean skill. Descriptive common-state actions; no across-seed pool, equivalence test, or training-speed claim.",
             fontsize=10, color="#52616C")
    return fig, axes


def render_all_parents(data):
    plt, Line2D, FuncFormatter = _plotting()
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 9.0), dpi=150)
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0.15, right=0.97, top=0.795, bottom=0.155, hspace=0.48, wspace=0.33)
    fig.suptitle("Native − raw across all 12 saved parents", x=0.075, y=0.982, ha="left", fontsize=20, weight="bold")
    fig.text(0.075, 0.935, "Positive = a larger objective decrease than raw. Every seed is separate; two draws averaged per parent.",
             fontsize=10.5, color="#52616C")
    legend = [Line2D([], [], marker="o", linestyle="none", color=FULL_COLOR, markersize=7, label="Full endpoint"),
              Line2D([], [], marker="D", linestyle="none", markerfacecolor="white", color=TENTH_COLOR, markersize=6,
                     label="Actual tenth endpoint"),
              Line2D([], [], marker="x", linestyle="none", color="#17232B", markersize=7,
                     label="Tiny: |finite contrast| ≤ 1e−8 nats")]
    fig.legend(handles=legend, loc="upper left", bbox_to_anchor=(0.071, 0.92), frameon=False, ncol=3, fontsize=10)
    labels = [f'{"Warmup 100" if step == 100 else "Final 56304"} · {str(seed)[-3:]}'
              for step in STEPS for seed in SEEDS]
    for row_index, mode in enumerate(MODES):
        rows = [r for r in data["all_parents"] if r["augmentation"] == mode]
        for col, objective in enumerate(OBJECTIVES):
            ax = axes[row_index, col]
            _style(ax)
            ax.axvspan(-TINY, TINY, color="#E9EBED", zorder=0)
            ax.axvline(0, color="#657581", linewidth=0.8)
            ax.axhline(2.5, color="#CAD0D5", linewidth=0.8)
            for index, parent in enumerate(rows):
                values = [parent["paths"][fid]["objectives"][objective]["contrast_finite"] for fid, _ in FRACTIONS]
                limit = 2e-4 if objective == "H_O" else 1e-5
                need(all(-limit < value < limit for value in values), "plot limits would hide a point")
                ys = [index - 0.10, index + 0.10]
                ax.plot(values, ys, color="#C7CED3", linewidth=0.8, zorder=1)
                for value, y, fid, color, marker in zip(values, ys, ("full", "tenth"),
                                                       (FULL_COLOR, TENTH_COLOR), ("o", "D")):
                    ax.scatter([value], [y], color=color, facecolors=color if fid == "full" else "white",
                               marker=marker, s=44, linewidths=1.2, zorder=3)
                    if abs(value) <= TINY:
                        ax.scatter([value], [y], marker="x", color="#17232B", s=65, linewidths=1.2, zorder=5)
            ax.set_xscale("symlog", linthresh=TINY, linscale=0.7)
            limit = 2e-4 if objective == "H_O" else 1e-5
            ax.set_xlim(-limit, limit)
            ticks = [-1e-4, -1e-6, -1e-8, 0, 1e-8, 1e-6, 1e-4] if objective == "H_O" else [-1e-6, -1e-8, 0, 1e-8, 1e-6]
            ax.set_xticks(ticks)
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: "0" if x == 0 else f"{x:.0e}"))
            ax.set_yticks(range(6), labels)
            ax.set_ylim(5.6, -0.6)
            ax.grid(axis="x", alpha=0.17)
            mode_label = "No augmentation" if mode == "none" else "Translation"
            objective_label = "Clean CE (H_O)" if objective == "H_O" else "Inconsistency (C)"
            ax.set_title(f"Prior training: {mode_label.lower()}\n{objective_label}", fontsize=11.5, loc="left", pad=12)
            ax.set_xlabel("Native − raw difference (nats; signed-log axis)", fontsize=10)
    fig.text(0.075, 0.082, "Gray band: linear axis segment from −1e−8 to +1e−8 nats. Outside it, signed logarithmic spacing.", fontsize=10.5)
    fig.text(0.075, 0.054, "Seed suffixes 171 / 172 / 173 denote 202609171 / 202609172 / 202609173. Tiny points are retained, not zeroed.",
             fontsize=10, color="#52616C")
    fig.text(0.075, 0.026, "C is not clean skill. Tenth paths were evaluated separately, not obtained by dividing full-path effects by ten.",
             fontsize=10, color="#52616C")
    return fig, axes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replace-own-output", action="store_true", help="Regenerate only this plotter's two PNGs and manifest.")
    args = parser.parse_args()
    outputs = [HERE / name for name in (*FILENAMES, "plots-manifest.json")]
    need(all(not path.is_symlink() and (not path.exists() or path.is_file()) for path in outputs),
         "only regular own plot outputs may be regenerated")
    need(args.replace_own_output or not any(path.exists() for path in outputs), "plot outputs already exist")
    data = plot_data(load_pinned_audit())
    plt, _, _ = _plotting()
    figures = []
    for name, render in zip(FILENAMES, (render_primary, render_all_parents)):
        fig, _ = render(data)
        path = HERE / name
        fig.savefig(path, dpi=150, facecolor="white", metadata={"Description": "Pinned audit " + AUDIT_SHA256})
        width, height = fig.canvas.get_width_height()
        plt.close(fig)
        figures.append({"path": name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "size_bytes": path.stat().st_size, "width_px": width, "height_px": height,
                        "dpi": 150, "background": "white"})
    figures[0]["axes"] = {"columns": ["H_O", "C"], "rows": ["absolute finite decreases", "native-minus-raw finite contrasts"],
                           "scale": "linear", "units": "nats"}
    figures[1]["axes"] = {"columns": ["H_O", "C"], "rows": ["none", "translate"],
                           "scale": "symlog", "linear_threshold_nats": TINY, "linear_scale": 0.7,
                           "limits_nats": {"H_O": [-2e-4, 2e-4], "C": [-1e-5, 1e-5]}, "units": "nats"}
    manifest = {"schema": "spectral_component_utility_plots_v1", "audit_path": "audit.json",
                "audit_sha256": AUDIT_SHA256, "plotter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "figures": figures, "data": data,
                "boundary": "Audit JSON scalar summary rendering and two-draw mean corroboration only; no scientific input replay or new audit."}
    (HERE / "plots-manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"figures": figures, "scalar_checks": data["scalar_mean_and_primary_checks"]}, indent=2))


if __name__ == "__main__":
    main()
