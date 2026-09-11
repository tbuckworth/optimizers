#!/usr/bin/env python3
"""Package audited raw-direction JSON into plots/tables; never rerun analysis.

All five expected input hashes must be supplied by the main agent after its
review. This script reads only those five JSONs. It neither opens a model/NPZ
nor computes a mean, standard deviation, standard error or paired difference.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
import time

REPO = Path(__file__).resolve().parents[1]
BATCH = Path("/tmp/spectral-experiment-artifacts/spectral-grokking-raw-direction-20260909.KzGtkl")
OUT = REPO / "output/2026-09-09-spectral-raw-direction/results"
SEEDS = tuple(range(100, 105))
STEPS = (2000, 2500)
POLICY = "raw_norm_matched"
ARCHIVED_SHA = "dc79ddbc9261607468c6adb5c8ff48b3e1249803f4651397aca134bcbc02a7a1"
INPUTS = {
    "summary": (BATCH / "analysis-001/summary.json", "summary.json"),
    "complete": (BATCH / "analysis-001/complete.json", "complete.json"),
    "manifest": (BATCH / "analysis-001/manifest.json", "manifest.json"),
    "tensor_audit": (BATCH / "tensor-history-audit-001/result.json", "tensor-history-audit.json"),
    "readout_audit": (BATCH / "readout-paired-audit-001/result.json", "readout-paired-audit.json"),
}
CONTRASTS = (("raw_minus_norm_matched", "norm_matched", "primary"),
             ("raw_minus_native", "native", "secondary"),
             ("raw_minus_orthogonal", "orthogonal", "secondary"))
METRICS = (
    ("heldout_cross_entropy", ("behavior", "test", "loss"),
     "Held-out cross-entropy", "Δ nats / example", "lower"),
    ("heldout_correct_margin_mean", ("behavior", "test", "correct_class_margin_mean"),
     "Correct-class margin", "Δ logit", "higher"),
    ("final_hidden_selected_five_heldout_r2", ("probes", "final_hidden", "selected_eval_mean_r2"),
     "Selected-five held-out probe R²", "Δ R²", "higher"),
)
ALL_METRICS = tuple(item[0] for item in METRICS) + (
    "heldout_accuracy", "final_hidden_fixed_panel_heldout_r2",
    "final_hidden_null_max_heldout_r2", "pre_attention_selected_five_heldout_r2",
    "pre_attention_fixed_panel_heldout_r2", "pre_attention_null_max_heldout_r2",
    "heldout_correct_shift_defect", "heldout_wrong_shift_defect", "exchange_defect",
    "training_membership_excess", "centered_logit_rms_heldout")
LABELS = {POLICY: "Raw direction", "norm_matched": "Projected direction (archived)",
          "native": "Native (archived)", "orthogonal": "Unscaled projection (archived)"}
MAX_INPUT_BYTES = 80 * 1024**2
MAX_OUTPUT_BYTES = 100 * 1024**2
FREE_RESERVE = 1024**3


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _finite(value, label, *, optional=False):
    if optional and value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise ValueError(f"invalid numeric display value: {label}")


def _read(path, expected):
    if re.fullmatch(r"[0-9a-f]{64}", expected) is None:
        raise ValueError("expected hash must be a main-supplied lowercase SHA256")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(descriptor, "rb") as handle:
        info = os.fstat(handle.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size > MAX_INPUT_BYTES:
            raise ValueError("input must be a small singly-linked regular JSON file")
        raw = handle.read(MAX_INPUT_BYTES + 1)
    if len(raw) > MAX_INPUT_BYTES or sha(raw) != expected:
        raise ValueError(f"expected input hash/size mismatch: {path}")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("input JSON must be an object")
    return {"value": value, "raw": raw,
            "receipt": {"path": str(path.resolve()), "sha256": expected, "size_bytes": len(raw)}}


def _receipt_identity(receipt):
    return {key: receipt.get(key) for key in ("path", "sha256", "size_bytes")}


def _roster(rows, wanted, label):
    identities = [(row.get("seed"), row.get("policy"), row.get("step")) for row in rows]
    if identities != wanted or len(set(identities)) != len(identities):
        raise ValueError(f"{label} exact ordered roster mismatch")


def _audit_input(audit, expected):
    matching = [row for row in audit.get("input_receipts", [])
                if row.get("path") == expected["path"]]
    if len(matching) != 1 or _receipt_identity(matching[0]) != expected:
        raise ValueError("audit does not bind this exact packaged input")


def admit(inputs):
    """Schema/receipt admission only. The independent audit owns arithmetic."""
    summary, complete, manifest, tensor, readout = (
        inputs[key]["value"] for key in ("summary", "complete", "manifest", "tensor_audit", "readout_audit"))
    if (summary.get("schema") != "grokking_raw_direction_analysis_summary_v1"
            or summary.get("new_state_count") != 15
            or complete.get("schema") != "grokking_raw_direction_analysis_complete_v1"
            or complete.get("status") != "complete" or complete.get("new_state_count") != 15
            or complete.get("contrast_count") != 6
            or manifest.get("schema") != "grokking_raw_direction_analysis_v1"
            or complete.get("summary") != inputs["summary"]["receipt"]
            or complete.get("manifest") != inputs["manifest"]["receipt"]
            or not summary.get("analysis_source_sha256")
            or summary["analysis_source_sha256"] != complete.get("analysis_source_sha256")
            or summary["analysis_source_sha256"] != manifest.get("analysis_source_sha256")
            or manifest.get("contrast_steps") != list(STEPS)
            or manifest.get("contrasts") != [list(row) for row in CONTRASTS]
            or manifest.get("archived_summary", {}).get("sha256") != ARCHIVED_SHA
            or summary.get("input_receipts", {}).get("archived_summary", {}).get("sha256") != ARCHIVED_SHA):
        raise ValueError("completed paired-analysis schema/source/input binding mismatch")
    for audit in (tensor, readout):
        if (audit.get("status") != "PASS" or audit.get("errors") != []
                or isinstance(audit.get("checks"), bool)
                or not isinstance(audit.get("checks"), int) or audit["checks"] <= 0):
            raise ValueError("both independent audits must be PASS with positive check counts and no errors")
    if (readout.get("analysis_sha256") != inputs["complete"]["receipt"]["sha256"]
            or readout.get("measurement_sha256") != manifest.get("measurement", {}).get("sha256")
            or readout.get("measurement_sha256")
               != summary.get("input_receipts", {}).get("measurement_completion", {}).get("sha256")
            or not isinstance(tensor.get("batch_completion_sha256"), str)
            or re.fullmatch(r"[0-9a-f]{64}", tensor["batch_completion_sha256"]) is None
            or readout.get("batch_sha256") != tensor["batch_completion_sha256"]
            or [row.get("seed") for row in tensor.get("rows", [])] != list(SEEDS)
            or any(row.get("history_rows") != 1000 for row in tensor["rows"])):
        raise ValueError("audit completion/batch/seed binding mismatch")
    for name in ("summary", "complete", "manifest"):
        _audit_input(readout, inputs[name]["receipt"])
    raw_roster = [(seed, POLICY, step) for seed in SEEDS for step in (1501, 2000, 2500)]
    _roster(summary.get("new_state_rows", []), raw_roster, "new state")
    if readout.get("states") != [list(row) for row in raw_roster]:
        raise ValueError("readout audit is not the fixed 15 states")
    old_roster = [(seed, policy, step) for seed in SEEDS
                  for policy in ("native", "orthogonal", "norm_matched")
                  for step in ((1501,) if policy == "native" else (1501, 2000, 2500))]
    _roster(summary.get("archived_action_reference_rows", []), old_roster, "archived action")
    _roster(summary.get("archived_native_reference_rows", []),
            [(seed, "native", step) for seed in SEEDS for step in (1500, 2000, 2500)], "archived native")
    contrasts = summary.get("paired_endpoint_contrasts", [])
    if [(row.get("step"), row.get("contrast")) for row in contrasts] != [
            (step, name) for step in STEPS for name, _, _ in CONTRASTS]:
        raise ValueError("not the exact six ordered endpoint contrasts")
    for contrast in contrasts:
        _, comparator, role = next(row for row in CONTRASTS if row[0] == contrast["contrast"])
        if (contrast.get("left_policy") != POLICY or contrast.get("right_policy") != comparator
                or contrast.get("contrast_role") != role
                or contrast.get("right_origin") != "accepted_archived_reference"
                or contrast.get("endpoint_role") != ("primary" if contrast["step"] == 2500 else "fixed_earlier")
                or set(contrast.get("metrics", {})) != set(ALL_METRICS)):
            raise ValueError("contrast policy/role/metric schema mismatch")
        for name, metric in contrast["metrics"].items():
            if (metric.get("metric") != name or metric.get("n_total_seeds") != 5
                    or metric.get("difference_definition") != "left_minus_right"
                    or [row.get("seed") for row in metric.get("paired", [])] != list(SEEDS)):
                raise ValueError("metric pair schema mismatch")
            optional = metric.get("complete_five_seed_aggregate") is False
            if name in {row[0] for row in METRICS} and optional:
                raise ValueError("primary metric cannot be undefined")
            for key in ("mean_difference", "sample_sd", "sample_se"):
                _finite(metric.get(key), name + ":" + key, optional=optional)
            for pair in metric["paired"]:
                for key in ("left", "right", "difference"):
                    _finite(pair.get(key), name + ":" + key, optional=optional)
    return summary


def value(row, path):
    for key in path:
        row = row[key]
    _finite(row, ".".join(path))
    return row


def number(value, *, signed=False):
    if value is None:
        return "undefined"
    _finite(value, "table")
    return format(value, "+.6g" if signed else ".6g")


def _cell(metric):
    if not metric["complete_five_seed_aggregate"]:
        return "undefined (incomplete five-seed metric)"
    return (f"{number(metric['mean_difference'], signed=True)} ± {number(metric['sample_se'])}; "
            f"{metric['favorable_count']}/5 favorable")


def tables(summary):
    """Format stored values, with percent conversions explicitly labeled."""
    contrasts = summary["paired_endpoint_contrasts"]
    primary = ["# Raw minus archived norm-matched projected direction", "",
               "Stored five-seed mean difference ± sample SE; no arithmetic is re-estimated here.", "",
               "| Update | Δ CE (lower favorable) | Δ margin (higher favorable) | Δ selected R² (higher favorable) |",
               "|---|---|---|---|"]
    for contrast in contrasts:
        if contrast["contrast"] == "raw_minus_norm_matched":
            primary.append(f"| {contrast['step']} | " + " | ".join(
                _cell(contrast["metrics"][name]) for name, *_ in METRICS) + " |")
    primary += ["", "Update 2,500 is primary; update 2,000 is the fixed earlier endpoint.",
                "Positive values favor raw for margin/R²; negative values favor raw for CE.",
                "These are archived-reference comparisons. Probe readability is not task performance."]
    detailed = ["# All six fixed endpoint contrasts", "",
                "All numbers below are stored analyzer outputs, not recomputed aggregates.",
                "Each seed cell is left / right / difference. Left is raw_norm_matched.",
                "Undefined means undefined, not zero or a reduced-roster estimate. Accuracy remains a fraction."]
    for contrast in contrasts:
        detailed += ["", f"## Update {contrast['step']}: {contrast['contrast']} ({contrast['contrast_role']})", "",
                     "| Metric (unit; favorable direction) | Mean Δ | Sample SD | Sample SE | Signs − / 0 / + | Favorable | "
                     + " | ".join(f"Seed {seed}: L / R / Δ" for seed in SEEDS) + " |",
                     "|" + "---|" * 11]
        for name in ALL_METRICS:
            metric = contrast["metrics"][name]
            fields = [f"{name} ({metric['unit']}; {metric['favorable_direction']})",
                      number(metric["mean_difference"], signed=True), number(metric["sample_sd"]),
                      number(metric["sample_se"]),
                      " / ".join(str(metric[key]) if metric[key] is not None else "undefined"
                                 for key in ("negative_count", "zero_count", "positive_count")),
                      "descriptive" if metric["favorable_direction"] == "descriptive" else
                      ("undefined" if metric["favorable_count"] is None else f"{metric['favorable_count']}/5")]
            fields += [" / ".join(number(pair[key], signed=key == "difference")
                                  for key in ("left", "right", "difference")) for pair in metric["paired"]]
            detailed.append("| " + " | ".join(fields) + " |")
    rows = summary["new_state_rows"] + summary["archived_action_reference_rows"] + summary["archived_native_reference_rows"]
    lookup = {(row["seed"], row["policy"], row["step"]): row for row in rows}
    exact = ["# Every seed at the fixed endpoints", "",
             "Raw step 1,501 is retained as descriptive context, not an extra primary endpoint.",
             "Accuracy below is percent; source JSON uses fractions."]
    for step in (1501, 2000, 2500):
        exact += ["", f"## Update {step}", "",
                  "| Seed | Policy / origin | CE (nats/example) | Margin (logit) | Selected probe R² | Accuracy (%) |",
                  "|---|---|---|---|---|---|"]
        policies = (POLICY,) if step == 1501 else (POLICY, "norm_matched", "native", "orthogonal")
        for seed in SEEDS:
            for policy in policies:
                row = lookup[seed, policy, step]
                fields = [str(seed), LABELS[policy]] + [number(value(row, path)) for _, path, *_ in METRICS]
                fields.append(number(100 * value(row, ("behavior", "test", "accuracy"))))
                exact.append("| " + " | ".join(fields) + " |")
    accuracy = ["# Secondary held-out accuracy contrasts", "",
                "Stored fraction differences and sample SE multiplied by 100 for percentage-point display.",
                "Higher favors raw. This metric is secondary; it is not a composite success criterion.", "",
                "| Update | Contrast | Mean Δ accuracy (percentage points) | Sample SE (percentage points) | Favorable seeds |",
                "|---|---|---|---|---|"]
    for contrast in contrasts:
        metric = contrast["metrics"]["heldout_accuracy"]
        accuracy.append(f"| {contrast['step']} | {contrast['contrast']} | "
                        f"{number(100 * metric['mean_difference'], signed=True)} | "
                        f"{number(100 * metric['sample_se'])} | {metric['favorable_count']}/5 |")
    return {"primary-paired-table.md": primary, "all-contrast-tables.md": detailed,
            "all-seed-table.md": exact, "secondary-accuracy-table.md": accuracy}


def paired_plot(summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    primary = {row["step"]: row for row in summary["paired_endpoint_contrasts"]
               if row["contrast"] == "raw_minus_norm_matched"}
    colors = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#806300")
    with plt.rc_context({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}):
        fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.8))
        for axis, (name, _, title, unit, direction) in zip(axes, METRICS):
            for position, step in enumerate(STEPS):
                metric = primary[step]["metrics"][name]
                for index, pair in enumerate(metric["paired"]):
                    axis.scatter(position + (index - 2) * .035, pair["difference"],
                                 color=colors[index], s=38, alpha=.9, zorder=3)
                # Separate the summary marker from every seed dot; otherwise
                # the central seed can be hidden beneath an identical mean.
                axis.errorbar(position + .18, metric["mean_difference"], yerr=metric["sample_se"],
                              fmt="D", color="#202124", markerfacecolor="white", markersize=6,
                              capsize=5, linewidth=1.8, zorder=4)
            axis.axhline(0, color="#686868", linewidth=1, linestyle="--", zorder=1)
            axis.set_xticks((0, 1), ("2,000", "2,500"))
            axis.set_xlim(-.3, 1.3)
            axis.set_xlabel("Optimizer update")
            axis.set_ylabel(unit)
            axis.set_title(title + "\n" + ("Below zero favors raw" if direction == "lower" else "Above zero favors raw"), fontsize=10)
            axis.grid(axis="y", alpha=.18, zorder=0)
        handles = [Line2D([], [], color=color, marker="o", linestyle="none", label=f"Seed {seed}")
                   for seed, color in zip(SEEDS, colors)]
        handles.append(Line2D([], [], color="#202124", marker="D", markerfacecolor="white",
                              linestyle="none", label="Stored mean ± sample SE"))
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .075),
                   ncol=6, frameon=False, fontsize=9)
        fig.suptitle("Raw direction − archived norm-matched projected direction", fontsize=14, y=.98)
        fig.text(.5, .017, "Five paired seeds · 2,500 primary; 2,000 fixed earlier · "
                 "same norm function, not identical scale histories · archived-reference limits apply",
                 ha="center", fontsize=8.5, color="#4c4c4c")
        fig.subplots_adjust(left=.065, right=.985, top=.79, bottom=.27, wspace=.34)
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=150, facecolor="white")
        plt.close(fig)
    return buffer.getvalue()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in INPUTS:
        parser.add_argument("--" + name.replace("_", "-") + "-sha256", required=True)
    args = parser.parse_args(argv)
    expected = {name: getattr(args, name + "_sha256") for name in INPUTS}
    started = time.monotonic()
    if OUT.exists() or OUT.is_symlink() or not OUT.parent.is_dir():
        raise ValueError("packaging output must be absent; no overwrite or automatic retry")
    source_raw = Path(__file__).read_bytes()
    inputs = {name: _read(path, expected[name]) for name, (path, _) in INPUTS.items()}
    if sum(len(item["raw"]) for item in inputs.values()) > MAX_INPUT_BYTES:
        raise ValueError("combined small-JSON input cap exceeded")
    summary = admit(inputs)
    if shutil.disk_usage(OUT.parent).free < MAX_OUTPUT_BYTES + FREE_RESERVE:
        raise ValueError("insufficient output space plus reserve")
    OUT.mkdir(mode=0o700)
    artifacts, written = {}, 0

    def write(name, raw):
        nonlocal written
        if (Path(name).name != name or written + len(raw) > MAX_OUTPUT_BYTES
                or shutil.disk_usage(OUT).free < len(raw) + FREE_RESERVE):
            raise ValueError("packaging path/output/free-space bound exceeded")
        path = OUT / name
        with path.open("xb") as handle:
            handle.write(raw)
        written += len(raw)
        artifacts[name] = {"sha256": sha(raw), "size_bytes": len(raw)}

    try:
        for name, (_, copied_name) in INPUTS.items():
            write(copied_name, inputs[name]["raw"])
        write("primary-paired-differences.png", paired_plot(summary))
        for name, lines in tables(summary).items():
            write(name, ("\n".join(lines) + "\n").encode())
        for name, (path, _) in INPUTS.items():
            if _read(path, expected[name])["raw"] != inputs[name]["raw"]:
                raise ValueError("packaging input changed")
        if Path(__file__).read_bytes() != source_raw:
            raise ValueError("report builder changed during packaging")
        manifest = {
            "schema": "grokking_raw_direction_report_packaging_v1", "status": "complete",
            "operation": "Display/copy audited stored JSON only; no numerical aggregation or model/array execution",
            "builder": {"path": str(Path(__file__).resolve()), "sha256": sha(source_raw)},
            "inputs": {name: item["receipt"] for name, item in inputs.items()},
            "expected_sha256": expected, "artifacts": dict(artifacts),
            "seeds": list(SEEDS), "primary_contrast": "raw_minus_norm_matched",
            "primary_endpoint": 2500, "fixed_earlier_endpoint": 2000,
            "plot": "All five stored differences and stored mean ± sample SE; no inferential intervals",
            "bounds": {"combined_input_bytes": MAX_INPUT_BYTES, "output_bytes": MAX_OUTPUT_BYTES,
                       "free_reserve_bytes": FREE_RESERVE},
            "elapsed_seconds": time.monotonic() - started,
            "interpretation_limits": ["Archived references; known CUDA sensitivity.",
                "Same scale function does not mean the same numerical scales or optimizer histories.",
                "Probe readability is not task performance or evidence of a rule circuit.",
                "No equivalence, causal mediation, safety, pre-fork formation or general-speed claim."]}
        write("packaging-manifest.json", (json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n").encode())
        print(json.dumps({"output": str(OUT), "packaging_manifest_sha256":
                          artifacts["packaging-manifest.json"]["sha256"], "output_bytes": written}))
    except BaseException as error:
        try:
            write("packaging-failure.json", (json.dumps({"status": "failed_preserved",
                  "type": type(error).__name__, "message": str(error), "artifacts": dict(artifacts)},
                  indent=2, allow_nan=False) + "\n").encode())
        except BaseException:
            pass
        raise


if __name__ == "__main__":
    main()
