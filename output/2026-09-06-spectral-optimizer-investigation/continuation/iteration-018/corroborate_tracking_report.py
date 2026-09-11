#!/usr/bin/env python3
"""Independent I18 report arithmetic from saved output/s arrays; no replay."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re

import numpy as np


HERE = Path(__file__).resolve().parent
SUMMARY = HERE / "analysis-001/summary.json"
AUDIT = HERE / "analysis-001/audit.json"
PLOT_MANIFEST = HERE / "plots-001/manifest.json"
PLOT_CSV = HERE / "plots-001/all-policy-window-means.csv"
PLOT_PNG = HERE / "plots-001/native-tracking.png"
PLOT_SOURCE = HERE / "make_tracking_plot.py"
OUTPUT = HERE / "report-check-001"
SUMMARY_SHA = "ce05b26c60a1f9f35656d021341b82494b901b1ce50fee94d2fee3e18a508df7"
AUDIT_SHA = "23866c4fd47230f36b0c0d82349f3c017abf2c9862276534a44f14c1728ba55a"
PLOT_MANIFEST_SHA = "d88fe4e98e565297e3594603fb7e566bc928b91d02e25e8b833dbcdf6bf929cd"
SEEDS = tuple(range(18000, 18032))
DRIFTS = (0.0, .01, .03)
ROTATIONS = (0, 1)
HORIZON = 4000
WINDOWS = (("startup", 0, 100), ("transition", 100, 1000), ("late", 1000, 4000))
ARRAY_KEYS = {"noise", "epsilon", "g", "s", "mu", "A", "native_delivery", "native_h",
    "full_moment", "full_action", "full_gap", "full_basis_present", "full_basis_V",
    "basis_present", "basis_V", "basis_S", "output", "native_i17_buffer", "scalar_buffer",
    "native_action_raw_error", "native_i17_response_residual", "native_cp_response_residual",
    "scalar_response_residual", "full_moment_residual", "native_action_idempotence_error",
    "native_basis_orthogonality_error", "native_useful_squared_alignment",
    "full_useful_squared_alignment"}


def policies(drift):
    result = ["raw", "ema_q0p9", "ema_q0p99", "ema_q0p999"]
    if drift:
        result.append("ema_optimal_oracle")
    return result + ["scalar_k0", "scalar_k0p5", "scalar_k0p9", "scalar_k1",
                     "oracle_useful", "oracle_nuisance", "native_i17", "native_cp"]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 ** 2), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    def pairs(items):
        out = {}
        for key, value in items:
            if key in out:
                raise ValueError("duplicate JSON key: " + key)
            out[key] = value
        return out
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle, object_pairs_hook=pairs,
                         parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))


def need(condition, message):
    if not condition:
        raise ValueError(message)


class Compare:
    def __init__(self):
        self.count = 0
        self.maximum = 0.0

    def number(self, actual, expected, label):
        need(type(actual) in (int, float) and type(expected) in (int, float)
             and math.isfinite(actual) and math.isfinite(expected), label + ": nonfinite")
        difference = abs(float(actual) - float(expected))
        self.maximum = max(self.maximum, difference)
        self.count += 1
        need(difference <= 5e-13 * max(1.0, abs(float(expected))), label + ": differs")

    def exact(self, actual, expected, label):
        self.count += 1
        need(actual == expected, label + ": differs")


def stats(per_seed, seeds=SEEDS):
    need(set(per_seed) == set(seeds), "incomplete seed roster")
    values = np.asarray([per_seed[seed] for seed in seeds], dtype=np.float64)
    need(np.isfinite(values).all(), "nonfinite seed metric")
    return {"mean": float(values.mean()),
            "standard_error": float(values.std(ddof=1) / math.sqrt(len(values))),
            "positive_seeds": int((values > 0).sum()),
            "negative_seeds": int((values < 0).sum()), "zero_seeds": int((values == 0).sum())}


def verify_file(root, record, expected):
    need(type(record) is dict and set(record) == {"path", "size", "sha256"}
         and record["path"] == expected and re.fullmatch(r"[0-9a-f]{64}", record["sha256"]),
         "invalid inventory record: " + expected)
    path = root / expected
    need(path.is_file() and not path.is_symlink() and path.stat().st_size == record["size"]
         and sha(path) == record["sha256"], "artifact differs: " + expected)
    return path


def compare_seed_summary(observed, values, compare, label, seeds=SEEDS):
    expected = stats(values, seeds)
    compare.exact(observed["available"], True, label + " available")
    compare.exact(observed["n_independent_seeds"], len(seeds), label + " n")
    compare.exact(observed["no_survivor_averaging"], True, label + " pooling")
    saved = observed["per_seed"]
    compare.exact(set(saved), {str(seed) for seed in seeds}, label + " seed keys")
    for seed in seeds:
        compare.number(values[seed], saved[str(seed)], label + f" seed {seed}")
    for key, value in expected.items():
        (compare.number if key in ("mean", "standard_error") else compare.exact)(
            observed[key], value, label + " " + key)


def load_scalar_metrics(array_path, names, horizon, windows):
    """Load only output and s from a numeric NPZ and derive scalar MSEs."""
    with np.load(array_path, allow_pickle=False) as archive:
        need(set(archive.files) == ARRAY_KEYS, "NPZ membership differs")
        output, signal = archive["output"], archive["s"]
        need(output.shape == (horizon, len(names), 2) and output.dtype == np.float64
             and signal.shape == (horizon, 2) and signal.dtype == np.float64
             and np.isfinite(output).all() and np.isfinite(signal).all(), "loaded scalar arrays differ")
        result = {}
        for window, first, last in windows:
            need(0 <= first < last <= horizon, "invalid scalar window")
            error = output[first:last] - signal[first:last, None, :]
            values = np.mean(np.sum(error * error, axis=2), axis=0)
            result[window] = {policy: float(values[column]) for column, policy in enumerate(names)}
        full_error = output - signal[:, None, :]
        values = np.mean(np.sum(full_error * full_error, axis=2), axis=0)
        whole = {policy: float(values[column]) for column, policy in enumerate(names)}
    return result, whole


def corroborate(root, summary_path=SUMMARY, audit_path=AUDIT, manifest_path=PLOT_MANIFEST,
                csv_path=PLOT_CSV, png_path=PLOT_PNG, *, seeds=SEEDS, drifts=DRIFTS,
                rotations=ROTATIONS, horizon=HORIZON, windows=WINDOWS):
    root = Path(root)
    need(root.is_absolute() and root.resolve() == root and not root.is_symlink(), "unsafe root")
    summary, audit, manifest = map(read_json, (summary_path, audit_path, manifest_path))
    need(sha(summary_path) == SUMMARY_SHA and sha(audit_path) == AUDIT_SHA
         and sha(manifest_path) == PLOT_MANIFEST_SHA, "accepted input hash differs")
    need(summary["schema"] == "i18_tracking_summary_v1" and audit["schema"] == "i18_tracking_audit_v1"
         and summary["audit_status"] == audit["status"] == "pass" and audit["errors"] == []
         and audit["summary_sha256"] == SUMMARY_SHA and summary["artifact_root"] == str(root)
         and audit["artifact_root"] == str(root)
         and summary["input_provenance"] == audit["input_provenance"]
         and summary["all_scientific_streams_complete"] and summary["completed_streams"] == 192,
         "accepted evidence envelope differs")
    attempt_path, completion_path = root / "attempt.json", root / "completion.json"
    need(sha(attempt_path) == summary["attempt_sha256"] == audit["attempt_sha256"]
         and sha(completion_path) == summary["completion_sha256"] == audit["completion_sha256"],
         "terminal pin differs")
    attempt, completion = read_json(attempt_path), read_json(completion_path)
    records = completion["streams"]
    ids = [f"{seed}-d{di}-r{ri}" for seed in seeds for di in range(len(drifts)) for ri in rotations]
    need(attempt["root"] == str(root) and attempt["seeds"] == list(seeds)
         and attempt["drifts"] == list(drifts) and attempt["rotations"] == [0.0, math.pi / 4]
         and attempt["horizon"] == horizon
         and completion["schema"] == "i18_tracking_completion_v1" and completion["status"] == "complete"
         and completion["failure"] is None and completion["expected_streams"] == len(ids)
         and completion["completed_streams"] == len(ids)
         and completion["completed_observations"] == len(ids) * horizon
         and [row["id"] for row in records] == ids,
         "acquisition roster differs")
    expected_files = {"attempt.json", "completion.json"}
    raw, whole_raw = {}, {}
    for row in records:
        identity = row["id"]
        match = re.fullmatch(r"([0-9]+)-d([0-9]+)-r([0-9]+)", identity)
        seed, di, ri = map(int, match.groups())
        array_name, meta_name = f"streams/{identity}.npz", f"streams/{identity}.json"
        array_path = verify_file(root, row["array"], array_name)
        meta_path = verify_file(root, row["metadata"], meta_name)
        expected_files.update((array_name, meta_name))
        metadata = read_json(meta_path)
        names = policies(drifts[di])
        need(metadata["id"] == identity and metadata["seed"] == seed
             and metadata["drift_index"] == di and metadata["rotation_index"] == ri
             and metadata["core"]["policy_names"] == names, "stream metadata differs")
        metric_values, whole_values = load_scalar_metrics(array_path, names, horizon, windows)
        for window, values in metric_values.items():
            for policy, value in values.items():
                raw[(seed, di, ri, window, policy)] = value
        if ri == 0:
            for policy in ("native_cp", "ema_optimal_oracle"):
                if policy in names:
                    whole_raw[(seed, di, policy)] = whole_values[policy]

    actual_files, actual_dirs = set(), set()
    for current, directories, files in os.walk(root, followlinks=False):
        for name in directories:
            path = Path(current) / name
            need(not path.is_symlink(), "symlink directory")
            actual_dirs.add(str(path.relative_to(root)))
        for name in files:
            path = Path(current) / name
            need(path.is_file() and not path.is_symlink(), "nonregular artifact")
            actual_files.add(str(path.relative_to(root)))
    need(actual_files == expected_files and actual_dirs == {"streams"},
         "physical root membership differs")

    compare = Compare()
    metric_index = {(r["seed"], r["drift_index"], r["rotation_index"], r["window"], r["policy"]): r
                    for r in summary["per_seed_window_metrics"]}
    need(len(metric_index) == len(summary["per_seed_window_metrics"]) and set(metric_index) == set(raw),
         "per-seed summary membership differs")
    for key, value in raw.items():
        compare.number(value, metric_index[key]["mse"], "per-seed MSE " + str(key))

    grouped = {}
    for (seed, di, ri, window, policy), value in raw.items():
        grouped.setdefault((di, ri, window, policy), {})[seed] = value
    mean_index = {(r["drift_index"], r["rotation_index"], r["window"], r["policy"]): r["mse"]
                  for r in summary["equal_seed_mse_summaries"]}
    need(len(mean_index) == len(summary["equal_seed_mse_summaries"]) and set(mean_index) == set(grouped),
         "aggregate membership differs")
    for key, values in grouped.items():
        compare_seed_summary(mean_index[key], values, compare, "aggregate " + str(key), seeds)

    for section, reference in (("paired_cp_contrasts", "native_cp"),
                               ("paired_useful_oracle_contrasts", "oracle_useful")):
        observed = summary[section]
        expected_comparison = ("comparator_mse_minus_native_cp_mse" if reference == "native_cp"
                               else "comparator_mse_minus_useful_oracle_mse")
        need(all(row["comparison"] == expected_comparison
                 and row["positive_favors"] == reference for row in observed),
             section + " sign semantics differ")
        expected_keys = {(di, ri, window, policy) for di, ri, window, policy in grouped
                         if (policy != reference if reference == "native_cp" else
                             policy == "raw" or policy.startswith(("ema_", "scalar_")))}
        index = {(r["drift_index"], r["rotation_index"], r["window"], r["comparator"]): r["effect"]
                 for r in observed}
        need(len(index) == len(observed) and set(index) == expected_keys, section + " membership differs")
        for key in expected_keys:
            di, ri, window, policy = key
            differences = {seed: grouped[key][seed] - grouped[(di, ri, window, reference)][seed]
                           for seed in seeds}
            compare_seed_summary(index[key], differences, compare, section + " " + str(key), seeds)

    need(sha(PLOT_SOURCE) == manifest["source_sha256"] and len(manifest["points"]) == 23
         and manifest["table_rows"] == len(grouped) == 228
         and manifest["input_sha256"] == {"analysis-001/summary.json": SUMMARY_SHA,
                                           "analysis-001/audit.json": AUDIT_SHA}
         and sha(csv_path) == manifest["output_sha256"][Path(csv_path).name]
         and sha(png_path) == manifest["output_sha256"][Path(png_path).name], "plot envelope differs")
    alignment = {(r["drift_index"], r["rotation_index"], r["window"]): r
                 for r in summary["equal_seed_alignment_summaries"]}
    for point in manifest["points"]:
        if point["panel"] == "mse":
            value = mean_index[(point["drift_index"], 0, "late", point["policy"])]
        else:
            value = alignment[(point["drift_index"], 0, "late")][point["field"]]
        compare.number(point["mean"], value["mean"], "plot mean")
        compare.number(point["standard_error"], value["standard_error"], "plot SE")

    with Path(csv_path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        need(reader.fieldnames == ["drift", "rotation_index", "window", "policy", "mean_mse",
                                   "seed_standard_error", "independent_seeds"], "CSV header differs")
        rows = list(reader)
    need(len(rows) == 228, "CSV row count differs")
    csv_keys = {(float(row["drift"]), int(row["rotation_index"]), row["window"], row["policy"])
                for row in rows}
    need(len(csv_keys) == len(rows), "duplicate CSV row")
    for row in rows:
        di = drifts.index(float(row["drift"]))
        value = mean_index[(di, int(row["rotation_index"]), row["window"], row["policy"])]
        compare.number(float(row["mean_mse"]), value["mean"], "CSV mean")
        compare.number(float(row["seed_standard_error"]), value["standard_error"], "CSV SE")
        compare.exact(int(row["independent_seeds"]), len(seeds), "CSV n")

    whole = []
    for di, drift in enumerate(drifts):
        cp = {seed: whole_raw[(seed, di, "native_cp")] for seed in seeds}
        optimum = None if drift == 0 else {
            seed: whole_raw[(seed, di, "ema_optimal_oracle")] for seed in seeds}
        row = {"drift": drift, "window": "descriptive_whole_horizon_1_4000",
               "native_cp_mse": stats(cp, seeds), "ema_optimal_oracle_mse": None,
               "optimal_minus_native_cp": None, "posthoc_not_primary": True}
        if optimum is not None:
            row["ema_optimal_oracle_mse"] = stats(optimum, seeds)
            row["optimal_minus_native_cp"] = stats({seed: optimum[seed] - cp[seed] for seed in seeds}, seeds)
            row["positive_favors"] = "native_cp"
        whole.append(row)
    return {"schema": "i18_tracking_report_corroboration_v1", "status": "pass",
            "input_sha256": {"summary": sha(summary_path), "audit": sha(audit_path),
                "plot_manifest": sha(manifest_path), "plot_csv": sha(csv_path), "plot_png": sha(png_path)},
            "root": str(root), "streams": len(records), "registered_scalar_comparisons": compare.count,
            "maximum_absolute_difference": compare.maximum, "whole_horizon_descriptive": whole,
            "scope": ["NumPy/stdlib scalar recomputation from saved output and s arrays; no observer, RNG, optimizer or tensor replay.",
                      "MSE plot points and all CSV rows are checked against raw aggregates; alignment plot points are checked against the accepted summary.",
                      "The whole-horizon rows are post-hoc descriptions and do not replace the three fixed windows."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    need(OUTPUT.parent == HERE and OUTPUT.name == "report-check-001" and not OUTPUT.exists(),
         "exclusive fixed report output required")
    result = corroborate(args.root)
    OUTPUT.mkdir(mode=0o700)
    content = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with (OUTPUT / "report-audit.json").open("x", encoding="utf-8") as handle:
        handle.write(content)
    print(json.dumps({"status": "pass", "output": str(OUTPUT),
                      "comparisons": result["registered_scalar_comparisons"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
