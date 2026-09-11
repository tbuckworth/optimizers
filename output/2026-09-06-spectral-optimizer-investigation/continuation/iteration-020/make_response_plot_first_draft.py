#!/usr/bin/env python3
"""Render the hash-bound, audited I20 presentation subset and full mean CSV.

This module performs no work at import.  Rendering is admitted only after the
supplied summary and audit hashes, audit binding, and complete mean rosters
have been verified.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import stat

import numpy as np


HERE = Path(__file__).resolve().parent
SEEDS = tuple(range(19000, 19032))
PROCESS_VARIANCES = (0.0, 0.01, 0.1)
ROTATIONS = (0.0, math.pi / 4.0)
WINDOWS = ("whole", "startup", "transition", "late")
ESTIMATORS = (
    "native", "full_legacy", "ew99", "ew999",
    "oracle_useful", "oracle_nuisance",
)
RHO_LABELS = ("r0p9", "rstar")
RESPONSES = ("cp", "rec99", "rec9")
POLICIES = tuple(
    f"{estimator}/{rho}/{response}"
    for estimator in ESTIMATORS
    for rho in RHO_LABELS
    for response in RESPONSES
)
SHA256 = re.compile(r"[0-9a-f]{64}")

PLOT_POLICIES = (
    ("parent", "ema_q0p9", "EMA .9 (parent)"),
    ("parent", "common_kalman", "Kalman (parent)"),
    ("new", "native/rstar/cp", "Native: old CP"),
    ("new", "native/rstar/rec99", "Native: carried-state response"),
    ("new", "native/rstar/rec9", "Native: shorter complement"),
    ("new", "ew99/rstar/rec9", "EW .99: shorter complement"),
    ("new", "ew999/rstar/rec9", "EW .999: shorter complement"),
    ("new", "oracle_useful/rstar/rec9", "Useful-direction oracle (privileged)"),
)
ALIGNMENT_ESTIMATORS = (
    ("native", "Native"),
    ("full_legacy", "Full legacy"),
    ("ew99", "EW .99"),
    ("ew999", "EW .999"),
)


class PlotError(ValueError):
    """Raised before rendering when an input or roster is not admissible."""


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def _regular_input(path: Path, label: str) -> Path:
    if path.is_symlink():
        raise PlotError(label + " must not be a symlink")
    resolved = path.resolve(strict=True)
    info = resolved.lstat()
    if not stat.S_ISREG(info.st_mode):
        raise PlotError(label + " must be a regular file")
    return resolved


def _read_json(path: Path) -> dict:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise PlotError("duplicate JSON key: " + key)
            result[key] = value
        return result

    with path.open(encoding="utf-8") as handle:
        value = json.load(
            handle,
            object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                PlotError("nonfinite JSON constant: " + token)),
        )
    if type(value) is not dict:
        raise PlotError("JSON root must be an object: " + str(path))
    return value


def _numeric(value: object, label: str) -> float:
    if type(value) not in (int, float) or isinstance(value, bool) \
            or not math.isfinite(value):
        raise PlotError(label + " must be a finite number")
    return float(value)


def _parent_policy_names(process_variance: float) -> tuple[str, ...]:
    names = [
        "raw", "ema_q0p9", "ema_q0p99", "ema_q0p999",
        "dema_q0p9", "dema_q0p99", "dema_q0p999",
    ]
    if process_variance != 0.0:
        names.append("ema_common_steady_oracle")
    names.extend((
        "common_kalman", "useful_oracle_kalman",
        "scalar_k0", "scalar_k0p5", "scalar_k0p9", "scalar_k1",
        "oracle_useful", "oracle_nuisance", "native_i17", "native_cp",
        "native_cp_star", "oracle_useful_star",
    ))
    return tuple(names)


def _validate_seed_summary(value: object, label: str) -> tuple[float, float]:
    keys = {
        "per_seed", "available", "n_independent_seeds", "mean",
        "standard_error", "positive_seeds", "negative_seeds", "zero_seeds",
        "no_survivor_averaging",
    }
    if type(value) is not dict or set(value) != keys:
        raise PlotError(label + " seed-summary schema differs")
    if value["available"] is not True or value["n_independent_seeds"] != 32 \
            or value["no_survivor_averaging"] is not True:
        raise PlotError(label + " must contain all 32 independent reused seeds")
    per_seed = value["per_seed"]
    if type(per_seed) is not dict or set(per_seed) != {str(seed) for seed in SEEDS}:
        raise PlotError(label + " per-seed roster differs")
    samples = np.asarray(
        [_numeric(per_seed[str(seed)], label + " per-seed value") for seed in SEEDS],
        dtype=np.float64,
    )
    mean = _numeric(value["mean"], label + " mean")
    standard_error = _numeric(value["standard_error"], label + " SE")
    if standard_error < 0.0:
        raise PlotError(label + " SE must be nonnegative")
    expected_mean = float(samples.mean())
    expected_se = float(samples.std(ddof=1) / math.sqrt(32))
    if not math.isclose(mean, expected_mean, rel_tol=2e-14, abs_tol=2e-14) \
            or not math.isclose(standard_error, expected_se,
                                rel_tol=2e-14, abs_tol=2e-14):
        raise PlotError(label + " mean/SE does not close from its seed values")
    signs = (
        int(np.sum(samples > 0.0)),
        int(np.sum(samples < 0.0)),
        int(np.sum(samples == 0.0)),
    )
    if tuple(value[key] for key in
             ("positive_seeds", "negative_seeds", "zero_seeds")) != signs:
        raise PlotError(label + " sign counts differ")
    return mean, standard_error


def _validate_mean_rows(
        rows: object, *, parent: bool,
) -> dict[tuple[int, int, str, str], dict]:
    label = "reused parent" if parent else "I20 new"
    expected = []
    for process_index, process_variance in enumerate(PROCESS_VARIANCES):
        policies = _parent_policy_names(process_variance) if parent else POLICIES
        for rotation_index in range(2):
            for window in WINDOWS:
                for policy in policies:
                    expected.append((process_index, rotation_index, window, policy))
    expected_count = 472 if parent else 864
    if type(rows) is not list or len(rows) != expected_count:
        raise PlotError(f"{label} mean row count is not exactly {expected_count}")
    observed = []
    index = {}
    for position, row in enumerate(rows):
        row_keys = {"process_index", "rotation_index", "window", "policy", "mse"}
        if parent:
            row_keys.add("policy_class")
        if type(row) is not dict or set(row) != row_keys:
            raise PlotError(f"{label} mean row {position} schema differs")
        if parent and (type(row["policy_class"]) is not str
                       or not row["policy_class"]):
            raise PlotError(f"{label} mean row {position} policy class differs")
        key = (row["process_index"], row["rotation_index"],
               row["window"], row["policy"])
        observed.append(key)
        _validate_seed_summary(row["mse"], f"{label} {key}")
        index[key] = row
    if observed != expected or len(index) != expected_count:
        raise PlotError(f"{label} mean roster/order differs")
    return index


def _validate_direction_rows(rows: object) -> dict[tuple[int, int, str, str], dict]:
    expected = [
        (process_index, rotation_index, window, estimator)
        for process_index in range(3)
        for rotation_index in range(2)
        for window in WINDOWS
        for estimator in ESTIMATORS
    ]
    if type(rows) is not list or len(rows) != 144:
        raise PlotError("equal-seed direction summary count is not exactly 144")
    observed = []
    index = {}
    required = {
        "process_index", "rotation_index", "window", "estimator",
        "mean_useful_squared_alignment_when_present", "mean_action_change_norm",
        "total_direction_present_observations",
        "total_direction_absent_observations",
    }
    for position, row in enumerate(rows):
        if type(row) is not dict or set(row) != required:
            raise PlotError(f"direction summary row {position} schema differs")
        key = (row["process_index"], row["rotation_index"],
               row["window"], row["estimator"])
        observed.append(key)
        index[key] = row
    if observed != expected or len(index) != 144:
        raise PlotError("equal-seed direction summary roster/order differs")
    for window in ("whole", "late"):
        for estimator, _ in ALIGNMENT_ESTIMATORS:
            key = (2, 0, window, estimator)
            _validate_seed_summary(
                index[key]["mean_useful_squared_alignment_when_present"],
                "strong-cell alignment " + str(key),
            )
    return index


def _admit(
        summary_path: Path,
        expected_summary_hash: str,
        audit_path: Path,
        expected_audit_hash: str,
) -> tuple[dict, dict, dict, dict, dict]:
    if SHA256.fullmatch(expected_summary_hash) is None \
            or SHA256.fullmatch(expected_audit_hash) is None:
        raise PlotError("both supplied hashes must be lowercase SHA-256 values")
    summary_path = _regular_input(summary_path, "summary")
    audit_path = _regular_input(audit_path, "audit")
    if summary_path.name != "summary.json" or audit_path.name != "audit.json" \
            or summary_path.parent != audit_path.parent:
        raise PlotError("summary.json and audit.json must be sibling analysis outputs")
    observed_summary_hash = _sha256(summary_path)
    observed_audit_hash = _sha256(audit_path)
    if observed_summary_hash != expected_summary_hash:
        raise PlotError("summary SHA-256 differs from supplied hash")
    if observed_audit_hash != expected_audit_hash:
        raise PlotError("audit SHA-256 differs from supplied hash")

    summary = _read_json(summary_path)
    audit = _read_json(audit_path)
    if summary.get("schema") != "i20_response_summary_v1" \
            or summary.get("audit_status") != "pass" \
            or summary.get("completed_streams") != 192 \
            or summary.get("expected_streams") != 192 \
            or summary.get("all_scientific_streams_complete") is not True:
        raise PlotError("summary is not a complete passing I20 response summary")
    if audit.get("schema") != "i20_response_audit_v1" \
            or audit.get("status") != "pass" or audit.get("errors") != []:
        raise PlotError("audit status is not pass")
    if audit.get("summary_sha256") != observed_summary_hash:
        raise PlotError("audit does not bind the supplied summary")
    for key in ("artifact_root", "attempt_sha256", "completion_sha256",
                "input_provenance"):
        if audit.get(key) != summary.get(key):
            raise PlotError("audit/summary binding differs: " + key)
    if type(summary.get("registered_contrasts")) is not list \
            or len(summary["registered_contrasts"]) != 2160 \
            or type(summary.get("primary_identity_whole_contrasts")) is not list \
            or len(summary["primary_identity_whole_contrasts"]) != 270:
        raise PlotError("registered 2,160/270 scientific contrast rosters differ")

    new_index = _validate_mean_rows(
        summary.get("equal_seed_mse_summaries"), parent=False)
    parent_index = _validate_mean_rows(
        summary.get("reused_parent_equal_seed_mse_summaries"), parent=True)
    direction_index = _validate_direction_rows(
        summary.get("equal_seed_direction_summaries"))
    records = {
        "summary": {"path": str(summary_path), "size": summary_path.stat().st_size,
                    "sha256": observed_summary_hash},
        "audit": {"path": str(audit_path), "size": audit_path.stat().st_size,
                  "sha256": observed_audit_hash},
    }
    return summary, new_index, parent_index, direction_index, records


def _mean_record(
        row: dict, collection: str, process_index: int, rotation_index: int,
        window: str, policy: str, role: str,
) -> dict:
    mean, standard_error = _validate_seed_summary(
        row["mse"], f"plotted MSE {(process_index, rotation_index, window, policy)}")
    return {
        "source_collection": collection,
        "source_keys": {
            "process_index": process_index,
            "rotation_index": rotation_index,
            "window": window,
            "policy": policy,
        },
        "Q": PROCESS_VARIANCES[process_index],
        "rotation": ROTATIONS[rotation_index],
        "n_independent_reused_seed_bundles": 32,
        "mean": mean,
        "standard_error": standard_error,
        "visual_role": role,
    }


def _alignment_record(
        row: dict, process_index: int, rotation_index: int,
        window: str, estimator: str,
) -> dict:
    metric = "mean_useful_squared_alignment_when_present"
    mean, standard_error = _validate_seed_summary(
        row[metric], f"plotted alignment {(process_index, rotation_index, window, estimator)}")
    return {
        "source_collection": "equal_seed_direction_summaries",
        "source_keys": {
            "process_index": process_index,
            "rotation_index": rotation_index,
            "window": window,
            "estimator": estimator,
            "metric": metric,
        },
        "Q": PROCESS_VARIANCES[process_index],
        "rotation": ROTATIONS[rotation_index],
        "n_independent_reused_seed_bundles": 32,
        "mean": mean,
        "standard_error": standard_error,
    }


def _write_full_csv(
        path: Path, new_index: dict, parent_index: dict,
) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("origin", "policy", "Q", "rotation", "window",
                        "n32", "mean", "SE"),
        )
        writer.writeheader()
        for origin, index in (("i20_new", new_index),
                              ("reused_parent", parent_index)):
            for (process_index, rotation_index, window, policy), row in index.items():
                mean, standard_error = _validate_seed_summary(
                    row["mse"], f"CSV {origin} {(process_index, rotation_index, window, policy)}")
                writer.writerow({
                    "origin": origin,
                    "policy": policy,
                    "Q": format(PROCESS_VARIANCES[process_index], ".17g"),
                    "rotation": format(ROTATIONS[rotation_index], ".17g"),
                    "window": window,
                    "n32": 32,
                    "mean": format(mean, ".17g"),
                    "SE": format(standard_error, ".17g"),
                })


def _render_mse(
        path: Path, new_index: dict, parent_index: dict, plt,
) -> list[dict]:
    colors = (
        "#767676", "#303030", "#3B6FB6", "#72A0D4",
        "#A7C5EB", "#D9872B", "#B65818", "#7651A8",
    )
    labels = [label for _, _, label in PLOT_POLICIES]
    y = np.arange(len(PLOT_POLICIES), dtype=np.float64)
    panel_titles = (
        "No process change\nQ = 0",
        "Weak process change\nQ = 0.01",
        "Strong process change\nQ = 0.1",
    )
    figure, axes = plt.subplots(1, 3, figsize=(10.5, 6.8), sharey=True)
    plotted = []
    for process_index, axis in enumerate(axes):
        whole_means, whole_ses, late_means, late_ses = [], [], [], []
        for origin, policy, _label in PLOT_POLICIES:
            index = parent_index if origin == "parent" else new_index
            collection = ("reused_parent_equal_seed_mse_summaries"
                          if origin == "parent" else "equal_seed_mse_summaries")
            for window, means, errors, role in (
                    ("whole", whole_means, whole_ses, "whole MSE bar"),
                    ("late", late_means, late_ses, "late MSE diamond")):
                row = index[(process_index, 0, window, policy)]
                record = _mean_record(
                    row, collection, process_index, 0, window, policy, role)
                record["origin"] = origin
                record["plain_label"] = _label
                means.append(record["mean"])
                errors.append(record["standard_error"])
                plotted.append(record)
        axis.barh(
            y, whole_means, xerr=whole_ses, color=colors, height=0.68,
            edgecolor="white", linewidth=0.7, capsize=2.5,
            error_kw={"elinewidth": 1.0},
        )
        axis.errorbar(
            late_means, y, xerr=late_ses, fmt="D", color="black",
            markerfacecolor="white", markersize=5, capsize=2.5,
            linewidth=1.0, label="Late (steps 1001–4000)",
        )
        axis.set_title(panel_titles[process_index], fontsize=11)
        axis.set_xlabel("Mean squared vector error\n(lower is better)")
        axis.grid(axis="x", color="#DDDDDD", linewidth=0.7)
        axis.set_axisbelow(True)
        axis.tick_params(axis="both", labelsize=10)
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    for axis in axes[1:]:
        axis.tick_params(axis="y", labelleft=False)
    axes[0].plot([], [], color="#777777", linewidth=8, label="Whole (steps 1–4000)")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, legend_labels, loc="upper center", ncol=2,
                  bbox_to_anchor=(0.5, 0.855), frameon=False, fontsize=11)
    figure.suptitle(
        "I20 identity-rotation responses — 32 REUSED independent seed bundles\n"
        "Exploratory, outcome-informed comparison; bars and diamonds show mean ± SE",
        fontsize=13, y=0.985,
    )
    figure.text(
        0.5, 0.002,
        "Eight-policy pre-outcome presentation subset; the accompanying CSV retains all "
        "864 new and 472 reused-parent mean rows. The useful-direction oracle is privileged.",
        ha="center", va="bottom", fontsize=9.5,
    )
    figure.subplots_adjust(left=0.30, right=0.985, bottom=0.16, top=0.73,
                           wspace=0.12)
    with path.open("xb") as handle:
        figure.savefig(handle, format="png", dpi=150, facecolor="white")
    plt.close(figure)
    return plotted


def _render_alignment(path: Path, direction_index: dict, plt) -> list[dict]:
    colors = ("#3B6FB6", "#767676", "#D9872B", "#B65818")
    labels = [label for _, label in ALIGNMENT_ESTIMATORS]
    x = np.arange(len(ALIGNMENT_ESTIMATORS), dtype=np.float64)
    figure, axes = plt.subplots(1, 2, figsize=(11, 6), constrained_layout=True)
    plotted = []
    for axis, window, title in zip(
            axes, ("whole", "late"),
            ("Whole: steps 1–4000", "Late: steps 1001–4000"), strict=True):
        means, errors = [], []
        for estimator, _label in ALIGNMENT_ESTIMATORS:
            row = direction_index[(2, 0, window, estimator)]
            record = _alignment_record(row, 2, 0, window, estimator)
            record["plain_label"] = _label
            means.append(record["mean"])
            errors.append(record["standard_error"])
            plotted.append(record)
        axis.bar(x, means, yerr=errors, color=colors, width=0.68,
                 edgecolor="white", capsize=3)
        axis.set_xticks(x, labels, rotation=25, ha="right")
        axis.set_ylim(0.0, 1.0)
        axis.set_title(title)
        axis.set_ylabel("Useful squared alignment when direction present")
        axis.grid(axis="y", color="#DDDDDD", linewidth=0.7)
        axis.set_axisbelow(True)
    figure.suptitle(
        "I20 strong cell (Q = 0.1, identity rotation) — 32 REUSED seed bundles\n"
        "Mean ± SE; exploratory direction diagnostic",
        fontsize=14,
    )
    figure.text(
        0.5, 0.005,
        "Alignment is diagnostic and outcome-inspected; it is not a prediction plug-in, "
        "native-learning guarantee, or fresh confirmation.",
        ha="center", va="bottom", fontsize=9,
    )
    with path.open("xb") as handle:
        figure.savefig(handle, format="png", dpi=150, facecolor="white")
    plt.close(figure)
    return plotted


def render(
        summary_path: Path,
        summary_hash: str,
        audit_path: Path,
        audit_hash: str,
        output_directory: Path,
) -> dict:
    """Validate inputs, then exclusively create the fixed I20 plot bundle."""
    summary, new_index, parent_index, direction_index, inputs = _admit(
        summary_path, summary_hash, audit_path, audit_hash)

    output_directory = Path(output_directory)
    if output_directory.name != "plots-001" \
            or output_directory.parent.resolve(strict=True) != HERE \
            or output_directory.exists() or output_directory.is_symlink():
        raise PlotError("output must be the absent exclusive iteration-020/plots-001")

    # Delay the plotting import and every write until all scientific inputs pass.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.dpi": 150,
        "font.size": 11,
    })

    output_directory.mkdir(mode=0o775, exist_ok=False)
    mse_path = output_directory / "response_mse.png"
    alignment_path = output_directory / "direction_alignment.png"
    csv_path = output_directory / "all_equal_seed_means.csv"
    manifest_path = output_directory / "plot_manifest.json"

    mse_values = _render_mse(mse_path, new_index, parent_index, plt)
    alignment_values = _render_alignment(alignment_path, direction_index, plt)
    _write_full_csv(csv_path, new_index, parent_index)
    generated = {
        path.name: {"size": path.stat().st_size, "sha256": _sha256(path)}
        for path in (mse_path, alignment_path, csv_path)
    }
    manifest = {
        "schema": "i20_response_plot_manifest_v1",
        "created_utc": _utc(),
        "source_sha256": _sha256(Path(__file__).resolve()),
        "input_files": inputs,
        "summary_audit_status": summary["audit_status"],
        "presentation_scope": {
            "study": "exploratory outcome-informed counterfactual",
            "replication": "32 reused independent I19 seed bundles; not fresh confirmation",
            "mse_direction": "lower is better",
            "rotation_for_figures": "identity",
            "oracle": "useful-direction oracle is privileged",
            "subset_disclosure": (
                "Figure 1 uses the exact eight-policy pre-outcome presentation subset; "
                "all 864 I20 and 472 reused-parent mean rows are retained in the CSV."
            ),
            "scientific_comparisons": (
                "No registered comparison was removed from the bound summary; the figures "
                "are presentation subsets and do not define a new selector."
            ),
            "alignment_caveat": (
                "Strong-cell alignment is diagnostic, not a prediction plug-in claim."
            ),
        },
        "figure_1_policy_roster": [
            {"origin": origin, "policy": policy, "plain_label": label}
            for origin, policy, label in PLOT_POLICIES
        ],
        "figure_1_plotted_values": mse_values,
        "figure_2_estimator_roster": [
            {"estimator": estimator, "plain_label": label}
            for estimator, label in ALIGNMENT_ESTIMATORS
        ],
        "figure_2_plotted_values": alignment_values,
        "full_csv": {
            "path": csv_path.name,
            "columns": ["origin", "policy", "Q", "rotation", "window",
                        "n32", "mean", "SE"],
            "i20_new_rows": 864,
            "reused_parent_rows": 472,
            "total_data_rows": 1336,
        },
        "generated_files": generated,
        "rendering": {"backend": "Agg", "background": "white", "dpi": 150},
    }
    with manifest_path.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, allow_nan=False)
        handle.write("\n")
    return {
        "status": "pass",
        "output_directory": str(output_directory.resolve()),
        "manifest_sha256": _sha256(manifest_path),
        "files": 4,
        "csv_data_rows": 1336,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Render hash-bound audited I20 response figures and full mean CSV")
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--summary-sha256", required=True)
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument("--audit-sha256", required=True)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    result = render(
        args.summary, args.summary_sha256,
        args.audit, args.audit_sha256,
        args.out,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
