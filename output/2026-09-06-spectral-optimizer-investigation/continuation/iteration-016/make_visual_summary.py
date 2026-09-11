#!/usr/bin/env python3
"""Render the audited I16 visual summary without replaying training or analysis."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


HERE = Path(__file__).resolve().parent
SUMMARY = HERE / "analysis-001" / "summary.json"
REPORT_AUDIT = HERE / "analysis-001" / "report-audit.json"
OUTPUT = HERE / "visual-summary"
MANIFEST = OUTPUT / "manifest.json"

SUMMARY_SHA256 = "11da55533e66c371132f939ce7e4a3a4495efc0fc5f5b202164d496bb438e401"
REPORT_AUDIT_SHA256 = "87be955c4f6f01514486969d51414d58b6ac583c0fb68f21cfb74de53ef6e96f"
HORIZONS = (100, 250, 500, 1000, 1500, 2000)
SEEDS = (200, 201, 202)
TARGETS = ("clean", "fixed")
POLICIES = ("k0", "k0p5", "k0p9", "k1", "mean_projected_history")
SELECTORS = ("minimum_validation_ce", "maximum_validation_accuracy")
METRICS = ("ce", "accuracy")

POLICY_STYLE = {
    "k0": ("EMA-only", "#159D91", "o"),
    "k0p5": ("Scalar .5", "#E69F00", "s"),
    "k0p9": ("Scalar .9", "#D55E00", "^"),
    "k1": ("Raw momentum", "#2474B5", "D"),
    "mean_projected_history": ("Spectral", "#7756A4", "P"),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} is not numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} is not finite")
    return result


def _validate_inputs() -> dict[str, Any]:
    if _sha256(SUMMARY) != SUMMARY_SHA256:
        raise ValueError("I16 summary SHA-256 mismatch")
    if _sha256(REPORT_AUDIT) != REPORT_AUDIT_SHA256:
        raise ValueError("I16 report-audit SHA-256 mismatch")

    summary = _load_json(SUMMARY)
    audit = _load_json(REPORT_AUDIT)
    if summary.get("schema") != "i16_scalar_analysis_summary_v1":
        raise ValueError("unexpected I16 summary schema")
    if summary.get("audit_status") != "pass":
        raise ValueError("I16 summary is not audit-passing")
    if audit.get("schema") != "i16_report_corroboration_v1" or audit.get("status") != "pass":
        raise ValueError("I16 report corroboration is not passing")
    if audit.get("errors") != []:
        raise ValueError("I16 report corroboration contains errors")
    if audit.get("maximum_absolute_numeric_difference") != 0:
        raise ValueError("I16 report corroboration is not exact")
    audit_summary = audit.get("inputs", {}).get("i16_summary", {})
    if audit_summary.get("sha256") != SUMMARY_SHA256:
        raise ValueError("report corroboration does not bind the pinned summary")
    return summary


def _extract_curve_data(summary: dict[str, Any]) -> dict[str, Any]:
    rows = summary.get("all_policy_trajectories")
    if not isinstance(rows, list) or len(rows) != 30:
        raise ValueError("expected exactly 30 complete policy trajectories")

    indexed: dict[tuple[str, str, int], dict[int, dict[str, float]]] = {}
    for row in rows:
        if row.get("status") != "complete" or row.get("failure") is not None:
            raise ValueError("visual summary requires all 30 complete trajectories")
        target, policy, seed = row.get("target"), row.get("policy"), row.get("seed")
        key = (target, policy, seed)
        if target not in TARGETS or policy not in POLICIES or seed not in SEEDS or key in indexed:
            raise ValueError(f"unexpected or duplicate trajectory key: {key}")
        curve = row.get("curve")
        if not isinstance(curve, list) or tuple(point.get("horizon") for point in curve) != HORIZONS:
            raise ValueError(f"trajectory has wrong horizons: {key}")
        points: dict[int, dict[str, float]] = {}
        for point in curve:
            horizon = point["horizon"]
            values = point.get("values", {})
            points[horizon] = {
                "auxiliary_clean_ce": _finite_number(values.get("auxiliary_clean_ce"), f"{key}/{horizon}/ce"),
                "auxiliary_clean_accuracy": _finite_number(
                    values.get("auxiliary_clean_accuracy"), f"{key}/{horizon}/accuracy"
                ),
            }
        indexed[key] = points

    expected = {(target, policy, seed) for target in TARGETS for policy in POLICIES for seed in SEEDS}
    if set(indexed) != expected:
        raise ValueError("trajectory membership mismatch")

    data: dict[str, Any] = {"horizons": list(HORIZONS), "seed_count": 3, "targets": {}}
    for target in TARGETS:
        target_data: dict[str, Any] = {}
        for policy in POLICIES:
            means = []
            for horizon in HORIZONS:
                ces = [indexed[(target, policy, seed)][horizon]["auxiliary_clean_ce"] for seed in SEEDS]
                accs = [indexed[(target, policy, seed)][horizon]["auxiliary_clean_accuracy"] for seed in SEEDS]
                means.append(
                    {
                        "horizon": horizon,
                        "auxiliary_clean_ce_mean": sum(ces) / 3.0,
                        "auxiliary_clean_accuracy_percent_mean": 100.0 * sum(accs) / 3.0,
                    }
                )
            target_data[policy] = means
        data["targets"][target] = target_data
    return data


def _extract_effect_data(summary: dict[str, Any]) -> dict[str, Any]:
    rows = summary.get("primary_validation_selected_spectral_minus_scalar")
    if not isinstance(rows, list) or len(rows) != 8:
        raise ValueError("expected exactly eight primary effects")
    indexed: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        key = (row.get("target"), row.get("selector"), row.get("auxiliary_metric"))
        if key in indexed or key[0] not in TARGETS or key[1] not in SELECTORS or key[2] not in METRICS:
            raise ValueError(f"unexpected or duplicate primary key: {key}")
        effect = row.get("effect", {})
        if effect.get("available") is not True or effect.get("no_survivor_averaging") is not True:
            raise ValueError(f"primary effect unavailable or survivor-averaged: {key}")
        per_seed = effect.get("per_seed")
        if not isinstance(per_seed, dict) or set(per_seed) != {str(seed) for seed in SEEDS}:
            raise ValueError(f"primary seed membership mismatch: {key}")
        values = [_finite_number(per_seed[str(seed)], f"{key}/{seed}") for seed in SEEDS]
        reported_values = effect.get("all_three_seed_values")
        if reported_values != values:
            raise ValueError(f"primary seed order mismatch: {key}")
        mean = _finite_number(effect.get("mean"), f"{key}/mean")
        if not math.isclose(mean, sum(values) / 3.0, rel_tol=0.0, abs_tol=1e-15):
            raise ValueError(f"primary mean mismatch: {key}")
        scale = 100.0 if key[2] == "accuracy" else 1.0
        indexed[key] = {
            "target": key[0],
            "selector": key[1],
            "metric": key[2],
            "seed_order": list(SEEDS),
            "seed_effects": [scale * value for value in values],
            "mean_effect": scale * mean,
            "orientation": "spectral_minus_scalar_utility; negative favors scalar",
        }

    expected = {(target, selector, metric) for target in TARGETS for selector in SELECTORS for metric in METRICS}
    if set(indexed) != expected:
        raise ValueError("primary effect membership mismatch")
    return {
        "seed_count": 3,
        "row_order": [
            {"target": target, "selector": selector}
            for target in TARGETS
            for selector in SELECTORS
        ],
        "effects": [indexed[(target, selector, metric)] for metric in METRICS for target in TARGETS for selector in SELECTORS],
    }


def _setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 11,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _render_curves(data: dict[str, Any], path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 7.2), sharex=True, constrained_layout=False)
    for row_index, target in enumerate(TARGETS):
        for column_index, metric in enumerate(("accuracy", "ce")):
            axis = axes[row_index][column_index]
            for policy in POLICIES:
                label, color, marker = POLICY_STYLE[policy]
                points = data["targets"][target][policy]
                key = "auxiliary_clean_accuracy_percent_mean" if metric == "accuracy" else "auxiliary_clean_ce_mean"
                axis.plot(
                    HORIZONS,
                    [point[key] for point in points],
                    label=label,
                    color=color,
                    marker=marker,
                    linewidth=2.0,
                    markersize=5,
                )
            target_label = "Clean-label training" if target == "clean" else "Corrupted-label training"
            metric_label = "Accuracy (%)" if metric == "accuracy" else "CE (lower is better)"
            axis.set_title(f"{target_label}\n{metric_label}")
            axis.set_ylabel("Accuracy (%)" if metric == "accuracy" else "Cross-entropy")
            axis.set_xticks((100, 500, 1000, 1500, 2000))
            axis.grid(axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.8)
            if row_index == 1:
                axis.set_xlabel("Cumulative updates")

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.875))
    fig.suptitle("I16 clean auxiliary learning curves", fontsize=15, fontweight="bold", y=0.99)
    fig.text(0.5, 0.94, "Three-seed means; descriptive, not confidence intervals", ha="center", fontsize=10.5)
    fig.text(0.5, 0.015, "All outcomes use the disjoint clean auxiliary set; rows identify the training target.", ha="center", fontsize=9.5)
    fig.subplots_adjust(top=0.75, bottom=0.11, left=0.11, right=0.98, hspace=0.40, wspace=0.25)
    fig.savefig(path, dpi=180, metadata={"Software": "I16 make_visual_summary.py"})
    plt.close(fig)


def _render_effects(data: dict[str, Any], path: Path) -> None:
    row_labels = ["Clean / min val CE", "Clean / max val accuracy", "Corrupted / min val CE", "Corrupted / max val accuracy"]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 6.4), sharey=True)
    jitters = (-0.15, 0.0, 0.15)
    for axis, metric in zip(axes, METRICS):
        rows = [row for row in data["effects"] if row["metric"] == metric]
        all_values = [value for row in rows for value in row["seed_effects"]] + [row["mean_effect"] for row in rows]
        limit = 1.18 * max(abs(value) for value in all_values)
        for row_index, row in enumerate(rows):
            for jitter, value in zip(jitters, row["seed_effects"]):
                color = "#159D91" if value < 0 else "#7756A4"
                axis.scatter(value, row_index + jitter, s=38, color=color, alpha=0.82, edgecolor="white", linewidth=0.5, zorder=3)
            mean = row["mean_effect"]
            mean_color = "#159D91" if mean < 0 else "#7756A4"
            axis.scatter(mean, row_index, marker="D", s=74, color=mean_color, edgecolor="#202020", linewidth=0.8, zorder=4)
            axis.text(mean, row_index + 0.24, f"{mean:+.3f}", ha="center", va="top", fontsize=9.5, color="#303030")
        axis.axvline(0.0, color="#404040", linewidth=1.1)
        axis.set_xlim(-limit, limit)
        axis.set_yticks(range(4), row_labels)
        axis.grid(axis="x", color="#D9D9D9", linewidth=0.8, alpha=0.8)
        axis.set_title("Accuracy effect (pp)" if metric == "accuracy" else "CE utility effect", pad=10)
        axis.set_xlabel("Spectral − selected scalar utility\n← favors scalar        favors spectral →")

    axes[0].set_ylim(3.55, -0.45)

    legend = [
        Line2D([0], [0], marker="o", linestyle="", color="#159D91", label="Seed effect favoring scalar", markersize=7),
        Line2D([0], [0], marker="o", linestyle="", color="#7756A4", label="Seed effect favoring spectral", markersize=7),
        Line2D([0], [0], marker="D", linestyle="", markerfacecolor="white", markeredgecolor="#202020", label="Three-seed mean", markersize=7),
    ]
    fig.legend(handles=legend, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.85))
    fig.suptitle("I16 validation-selected primary effects", fontsize=15, fontweight="bold", y=0.99)
    fig.text(0.5, 0.94, "Three seeds plus mean diamonds; dots are not confidence intervals", ha="center", fontsize=10.5)
    fig.text(0.5, 0.015, "CE utility is −CE, so its plotted value is scalar CE minus spectral CE.", ha="center", fontsize=9.5)
    fig.subplots_adjust(top=0.72, bottom=0.21, left=0.29, right=0.98, wspace=0.18)
    fig.savefig(path, dpi=180, metadata={"Software": "I16 make_visual_summary.py"})
    plt.close(fig)


def _png_dimensions(path: Path) -> tuple[int, int]:
    with path.open("rb") as handle:
        signature = handle.read(24)
    if signature[:8] != b"\x89PNG\r\n\x1a\n" or signature[12:16] != b"IHDR":
        raise ValueError(f"not a valid PNG: {path}")
    return struct.unpack(">II", signature[16:24])


def _data_bundle(summary: dict[str, Any]) -> dict[str, Any]:
    return {"learning_curves": _extract_curve_data(summary), "selection_effects": _extract_effect_data(summary)}


def _verify_existing(summary: dict[str, Any]) -> None:
    manifest = _load_json(MANIFEST)
    if manifest.get("schema") != "i16_visual_summary_v1" or manifest.get("status") != "complete":
        raise ValueError("visual manifest is not complete")
    if manifest.get("source", {}).get("summary_sha256") != SUMMARY_SHA256:
        raise ValueError("visual manifest summary pin mismatch")
    if manifest.get("source", {}).get("report_audit_sha256") != REPORT_AUDIT_SHA256:
        raise ValueError("visual manifest audit pin mismatch")
    expected_data = _data_bundle(summary)
    if manifest.get("plotted_data") != expected_data:
        raise ValueError("manifest plotted data does not exactly match the pinned summary")
    for name, record in manifest.get("figures", {}).items():
        path = OUTPUT / name
        if _sha256(path) != record.get("sha256") or path.stat().st_size != record.get("bytes"):
            raise ValueError(f"figure integrity mismatch: {name}")
        if list(_png_dimensions(path)) != record.get("pixel_dimensions"):
            raise ValueError(f"figure dimensions mismatch: {name}")
    print("Visual summary verification passed: plotted data exactly matches pinned I16 summary; PNG hashes match manifest.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true", help="verify existing figures and manifest against pinned inputs")
    args = parser.parse_args()
    summary = _validate_inputs()
    if args.verify_only:
        _verify_existing(summary)
        return

    OUTPUT.mkdir(exist_ok=True)
    data = _data_bundle(summary)
    _setup_style()
    figures = {
        "learning-curves.png": data["learning_curves"],
        "selection-effects.png": data["selection_effects"],
    }
    _render_curves(figures["learning-curves.png"], OUTPUT / "learning-curves.png")
    _render_effects(figures["selection-effects.png"], OUTPUT / "selection-effects.png")

    figure_records = {}
    for name in figures:
        path = OUTPUT / name
        width, height = _png_dimensions(path)
        if min(width, height) < 900:
            raise ValueError(f"figure resolution too small: {name}")
        figure_records[name] = {
            "sha256": _sha256(path),
            "bytes": path.stat().st_size,
            "pixel_dimensions": [width, height],
            "dpi": 180,
        }
    manifest = {
        "schema": "i16_visual_summary_v1",
        "status": "complete",
        "generated_date": "2026-09-08",
        "statement": "Visualization of accepted audited I16 scalar results; no new experiment, training, forward pass, or scientific reanalysis.",
        "source": {
            "summary": str(SUMMARY.relative_to(HERE)),
            "summary_sha256": SUMMARY_SHA256,
            "report_audit": str(REPORT_AUDIT.relative_to(HERE)),
            "report_audit_sha256": REPORT_AUDIT_SHA256,
            "report_audit_status": "pass",
        },
        "scope": {
            "seeds": list(SEEDS),
            "seed_count": 3,
            "horizons": list(HORIZONS),
            "learning_curve_axis": "actual numeric cumulative-update horizon; the h250 point is retained at x=250 but is not a labeled major tick",
            "learning_curve_aggregation": "equal arithmetic mean over the three seeds; descriptive only",
            "selection_effects": "all three seed effects plus arithmetic mean; no confidence interval or equivalence claim",
            "evaluation": "all plotted outcomes use the disjoint clean auxiliary data; rows distinguish clean-label versus corrupted-label training targets",
        },
        "figures": figure_records,
        "plotted_data": data,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    _verify_existing(summary)


if __name__ == "__main__":
    main()
