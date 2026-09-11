#!/usr/bin/env python3
"""Strict paired aggregation of completed saved grokking-action measurements.

This program never loads a checkpoint, model, or NPZ member.  Raw-array files
are hash-checked as receipts; all numerical inputs come from accepted JSON.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import shutil
import time
from typing import Any, Iterable

REPO = Path(__file__).resolve().parents[1]
SEEDS = tuple(range(100, 105))
POLICIES = ("native", "orthogonal", "norm_matched")
ENDPOINT_STEPS = (2000, 2500)
REFERENCE_STEPS = (1500, 2000, 2500)
FIXED_PANEL = [9, 33, 32, 49, 11]
PRIOR_SUMMARY = REPO / "output/2026-09-09-spectral-grokking-mechanism/results/summary.json"
PRIOR_SUMMARY_SHA256 = "9ac881c1bbca72a9dfd5525c226e05938ea4b0553e1e924bafb59ffd8a9a7c0d"
OUTPUT_LIMIT = 100 * 1024**2
OUTPUT_RESERVE = 1024**3
MEMORY_LIMIT = 2 * 1024**3
COOPERATIVE_SECONDS = 300


def expected_new_roster() -> list[tuple[int, str, int]]:
    return [(seed, policy, step) for seed in SEEDS for policy in POLICIES
            for step in ((1501,) if policy == "native" else (1501, 2000, 2500))]


def expected_prior_roster() -> list[tuple[int, str, int]]:
    return [(seed, arm, step) for seed in SEEDS
            for arm in ("adamw", "legacy", "stable")
            for step in (0, 100, 500, 1000, 1500, 2000, 2500, 3000, 4000, 6000)]


METRICS: dict[str, dict[str, Any]] = {
    "heldout_cross_entropy": {
        "path": ("behavior", "test", "loss"), "role": "primary",
        "unit": "nats_per_example", "favorable_direction": "lower"},
    "heldout_correct_margin_mean": {
        "path": ("behavior", "test", "correct_class_margin_mean"), "role": "primary",
        "unit": "logit", "favorable_direction": "higher"},
    "final_hidden_selected_five_heldout_r2": {
        "path": ("probes", "final_hidden", "selected_eval_mean_r2"), "role": "primary",
        "unit": "R2", "favorable_direction": "higher"},
    "heldout_accuracy": {
        "path": ("behavior", "test", "accuracy"), "role": "secondary",
        "unit": "fraction", "favorable_direction": "higher"},
    "final_hidden_fixed_panel_heldout_r2": {
        "path": ("probes", "final_hidden", "fixed_panel_eval_mean_r2"),
        "role": "secondary", "unit": "R2", "favorable_direction": "higher"},
    "final_hidden_null_max_heldout_r2": {
        "path": ("probes", "final_hidden", "null_max_eval_mean_r2"),
        "role": "secondary", "unit": "R2", "favorable_direction": "descriptive"},
    "pre_attention_selected_five_heldout_r2": {
        "path": ("probes", "pre_attention", "selected_eval_mean_r2"),
        "role": "secondary", "unit": "R2", "favorable_direction": "descriptive"},
    "pre_attention_fixed_panel_heldout_r2": {
        "path": ("probes", "pre_attention", "fixed_panel_eval_mean_r2"),
        "role": "secondary", "unit": "R2", "favorable_direction": "descriptive"},
    "pre_attention_null_max_heldout_r2": {
        "path": ("probes", "pre_attention", "null_max_eval_mean_r2"),
        "role": "secondary", "unit": "R2", "favorable_direction": "descriptive"},
    "heldout_correct_shift_defect": {
        "path": ("symmetry", "heldout_shift_pooled", "correct", "value"),
        "role": "secondary", "unit": "normalized_squared_defect",
        "favorable_direction": "lower", "allow_undefined": True},
    "heldout_wrong_shift_defect": {
        "path": ("symmetry", "heldout_shift_pooled", "wrong_shift", "value"),
        "role": "secondary", "unit": "normalized_squared_defect",
        "favorable_direction": "descriptive", "allow_undefined": True},
    "exchange_defect": {
        "path": ("symmetry", "exchange", "correct", "value"),
        "role": "secondary", "unit": "normalized_squared_defect",
        "favorable_direction": "lower", "allow_undefined": True},
    "training_membership_excess": {
        "path": ("symmetry", "training_membership_pooled", "excess"),
        "role": "secondary", "unit": "normalized_squared_defect_difference",
        "favorable_direction": "descriptive", "allow_undefined": True},
    "centered_logit_rms_heldout": {
        "path": ("symmetry", "centered_logit_rms", "test"),
        "role": "secondary", "unit": "logit", "favorable_direction": "descriptive"},
}

CONTRASTS = (
    ("orthogonal_minus_native", "orthogonal", "native"),
    ("norm_matched_minus_native", "norm_matched", "native"),
    ("norm_matched_minus_orthogonal", "norm_matched", "orthogonal"),
)


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> tuple[dict[str, Any], bytes, dict[str, Any]]:
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_nlink != 1:
        raise ValueError(f"not a singly-linked regular JSON file: {path}")
    raw = path.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"JSON root is not an object: {path}")
    return value, raw, {"path": str(path.resolve()), "sha256": file_hash(path),
                        "size_bytes": len(raw)}


def _verify_receipt(receipt: dict[str, Any], parent: Path, *, name: str | None = None,
                    hash_contents: bool = True) -> dict[str, Any]:
    if (not isinstance(receipt, dict)
            or not {"path", "sha256", "size_bytes"}.issubset(receipt)
            or not isinstance(receipt["sha256"], str)):
        raise ValueError("malformed receipt")
    path = Path(receipt["path"])
    if (path.is_symlink() or not path.is_file() or path.stat().st_nlink != 1
            or path.resolve().parent != Path(parent).resolve()
            or (name is not None and path.name != name)):
        raise ValueError("receipt path/type/parent mismatch")
    size = path.stat().st_size
    if receipt["size_bytes"] != size:
        raise ValueError("receipt size mismatch")
    if hash_contents and file_hash(path) != receipt["sha256"]:
        raise ValueError("receipt hash mismatch")
    return {"path": str(path.resolve()), "sha256": receipt["sha256"],
            "size_bytes": size}


def _verify_source_map(pins: Any) -> dict[str, str]:
    if not isinstance(pins, dict) or not pins:
        raise ValueError("empty or malformed source hash map")
    normalized = {}
    for relative, expected in pins.items():
        if (not isinstance(relative, str) or Path(relative).is_absolute()
                or ".." in Path(relative).parts or not isinstance(expected, str)
                or len(expected) != 64):
            raise ValueError("malformed source hash binding")
        path = REPO / relative
        if not path.is_file() or path.is_symlink() or file_hash(path) != expected:
            raise ValueError(f"source hash mismatch: {relative}")
        normalized[relative] = expected
    return normalized


def analysis_sources() -> dict[str, str]:
    paths = (
        "experiments/analyze_grokking_action_results.py",
        "tests/test_grokking_action_analysis.py",
        "experiments/measure_grokking_action_states.py",
        "experiments/analyze_grokking_representations.py",
        "output/2026-09-09-spectral-grokking-action/protocol.md",
        "output/2026-09-09-spectral-grokking-action/measurement-protocol.md",
        "output/2026-09-09-spectral-grokking-action/paired-analysis-protocol.md",
    )
    return {path: file_hash(REPO / path) for path in paths}


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"missing/non-numeric required metric: {label}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"nonfinite required metric: {label}")
    return result


def _nested(row: dict[str, Any], path: tuple[str, ...], label: str,
            *, allow_undefined: bool = False) -> float | None:
    value: Any = row
    try:
        for key in path:
            value = value[key]
    except (KeyError, TypeError) as error:
        raise ValueError(f"missing required metric: {label}") from error
    if value is None and allow_undefined:
        return None
    return _finite_number(value, label)


def validate_metric_row(row: dict[str, Any]) -> None:
    for name, definition in METRICS.items():
        _nested(row, definition["path"], name,
                allow_undefined=definition.get("allow_undefined", False))
    for split in ("train", "test"):
        count = row.get("behavior", {}).get(split, {}).get("count")
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError("behavior count is missing or invalid")
    for feature in ("final_hidden", "pre_attention"):
        probe = row.get("probes", {}).get(feature, {})
        if (probe.get("fixed_panel_frequencies") != FIXED_PANEL
                or len(probe.get("selected_frequencies", [])) != 5):
            raise ValueError("probe panel/selection schema mismatch")


def _paired_summary(left_rows: dict[int, dict], right_rows: dict[int, dict],
                    metric: str) -> dict[str, Any]:
    definition = METRICS[metric]
    paired = []
    for seed in SEEDS:
        left = _nested(left_rows[seed], definition["path"], metric,
                       allow_undefined=definition.get("allow_undefined", False))
        right = _nested(right_rows[seed], definition["path"], metric,
                        allow_undefined=definition.get("allow_undefined", False))
        difference = None if left is None or right is None else left - right
        if difference is not None and not math.isfinite(difference):
            raise ValueError("nonfinite paired difference")
        paired.append({"seed": seed, "left": left, "right": right,
                       "difference": difference})
    values = [row["difference"] for row in paired if row["difference"] is not None]
    complete = len(values) == len(SEEDS)
    mean = math.fsum(values) / len(values) if complete else None
    variance = (math.fsum((value - mean) ** 2 for value in values) / (len(values) - 1)
                if complete else None)
    sd = math.sqrt(variance) if variance is not None else None
    se = sd / math.sqrt(len(values)) if sd is not None else None
    return {"metric": metric, "role": definition["role"], "unit": definition["unit"],
            "favorable_direction": definition["favorable_direction"],
            "difference_definition": "left_minus_right", "n_total_seeds": len(SEEDS),
            "defined_count": len(values), "complete_five_seed_aggregate": complete,
            "paired": paired, "mean_difference": mean, "sample_sd": sd,
            "sample_se": se,
            "positive_count": sum(value > 0 for value in values) if complete else None,
            "negative_count": sum(value < 0 for value in values) if complete else None,
            "zero_count": sum(value == 0 for value in values) if complete else None}


def paired_contrasts(new_rows: Iterable[dict], reference_rows: Iterable[dict]) -> list[dict]:
    new_list, reference_list = list(new_rows), list(reference_rows)
    new = {(row["seed"], row["policy"], row["step"]): row for row in new_list}
    native = {(row["seed"], row["step"]): row for row in reference_list}
    needed_new = {(seed, policy, step) for seed in SEEDS
                  for policy in ("orthogonal", "norm_matched")
                  for step in ENDPOINT_STEPS}
    needed_native = {(seed, step) for seed in SEEDS for step in ENDPOINT_STEPS}
    if (not needed_new.issubset(new) or not needed_native.issubset(native)
            or len(new) != len(new_list) or len(native) != len(reference_list)):
        raise ValueError("paired input is duplicate or lacks a fixed endpoint")
    output = []
    for step in ENDPOINT_STEPS:
        groups = {
            "native": {seed: native[seed, step] for seed in SEEDS},
            "orthogonal": {seed: new[seed, "orthogonal", step] for seed in SEEDS},
            "norm_matched": {seed: new[seed, "norm_matched", step] for seed in SEEDS},
        }
        for name, left, right in CONTRASTS:
            output.append({
                "step": step, "contrast": name, "left_policy": left,
                "right_policy": right, "reference_for_native": "archived_legacy",
                "metrics": {metric: _paired_summary(groups[left], groups[right], metric)
                            for metric in METRICS},
            })
    return output


def verify_measurement(path: Path, *, hash_raw: bool = True) -> dict[str, Any]:
    path = Path(path).resolve()
    if (path / "failure.json").exists():
        raise ValueError("complete measurement also contains a failure marker")
    manifest, _, manifest_receipt = _read_json(path / "manifest.json")
    complete, _, complete_receipt = _read_json(path / "complete.json")
    roster = expected_new_roster()
    accepted = complete.get("accepted_results", [])
    identities = [(row.get("seed"), row.get("policy"), row.get("step"))
                  for row in accepted]
    if (manifest.get("schema") != "grokking_action_measurement_v1"
            or complete.get("schema") != "grokking_action_measurement_complete_v1"
            or complete.get("status") != "complete" or complete.get("state_count") != 35
            or [tuple(row) for row in manifest.get("roster", [])] != roster
            or identities != roster or len(set(identities)) != 35
            or complete.get("manifest", {}).get("sha256") != manifest_receipt["sha256"]
            or complete.get("prior_summary", {}).get("sha256") != PRIOR_SUMMARY_SHA256
            or manifest.get("fixed_frequency_panel", {}).get("frequencies") != FIXED_PANEL
            or manifest.get("measurement_source_sha256")
               != complete.get("measurement_source_sha256")
            or manifest.get("action_source_sha256") != complete.get("action_source_sha256")):
        raise ValueError("measurement manifest/completion/roster contract mismatch")
    _verify_receipt(complete["manifest"], path, name="manifest.json")
    prior_receipts = (manifest.get("prior_recipe", {}).get("summary"),
                      complete.get("prior_summary"))
    for prior_receipt in prior_receipts:
        if (not isinstance(prior_receipt, dict)
                or Path(prior_receipt.get("path", "")).resolve() != PRIOR_SUMMARY.resolve()
                or prior_receipt.get("sha256") != PRIOR_SUMMARY_SHA256):
            raise ValueError("measurement does not bind the accepted prior summary")
    action_batch = manifest.get("action_batch", {})
    if (complete.get("input_batch_completion") != action_batch.get("completion")
            or not isinstance(action_batch.get("path"), str)):
        raise ValueError("measurement action-batch receipt binding mismatch")
    batch_parent = Path(action_batch["path"])
    _verify_receipt(action_batch.get("manifest"), batch_parent, name="batch-manifest.json")
    _verify_receipt(action_batch.get("completion"), batch_parent, name="batch-complete.json")
    _verify_source_map(manifest["measurement_source_sha256"])
    _verify_source_map(manifest["action_source_sha256"])
    rows, receipts, scalar_copies, state_copies = [], [], [], []
    for accepted_row in accepted:
        seed, policy, step = (accepted_row[key] for key in ("seed", "policy", "step"))
        stem = f"seed{seed}-{policy}-step{step:06d}"
        raw = _verify_receipt(accepted_row["raw"], path / "raw",
                              name=stem + ".npz", hash_contents=hash_raw)
        scalar_receipt = _verify_receipt(accepted_row["scalar"], path / "scalars",
                                         name=stem + ".json")
        state_receipt = _verify_receipt(accepted_row["analyzed_state"], path / "states",
                                        name=stem + ".json")
        scalar, scalar_bytes, reread_scalar = _read_json(Path(scalar_receipt["path"]))
        state, state_bytes, reread_state = _read_json(Path(state_receipt["path"]))
        identity = (seed, policy, step)
        provenance = state.get("checkpoint_provenance", {})
        if (reread_scalar["sha256"] != scalar_receipt["sha256"]
                or reread_state["sha256"] != state_receipt["sha256"]
                or scalar.get("schema") != "grokking_action_measurement_state_v1"
                or (scalar.get("seed"), scalar.get("policy"), scalar.get("step")) != identity
                or (state.get("seed"), state.get("policy"), state.get("step")) != identity
                or state.get("arm") != policy or scalar.get("arm") != policy
                or scalar.get("raw_activations", {}).get("sha256") != raw["sha256"]
                or scalar.get("source_sha256") != manifest["measurement_source_sha256"]
                or state.get("source", {}).get("stage") != "action"
                or state.get("source", {}).get("source_scalar_sha256") != scalar_receipt["sha256"]
                or state.get("source", {}).get("source_npz_sha256") != raw["sha256"]
                or state.get("checkpoint_provenance") != scalar.get("checkpoint_provenance")
                or (provenance.get("outer_seed"), provenance.get("outer_policy"),
                    provenance.get("outer_step")) != identity
                or provenance.get("action_checkpoint") != accepted_row.get("checkpoint")
                or state.get("source", {}).get("action_checkpoint_sha256")
                   != accepted_row.get("checkpoint", {}).get("sha256")):
            raise ValueError("new-state scalar/analyzed/provenance binding mismatch")
        validate_metric_row(state)
        summary_row = {key: value for key, value in state.items() if key != "full_symmetry"}
        summary_row["raw_receipt"] = raw
        summary_row["source_scalar_receipt"] = scalar_receipt
        summary_row["source_state_receipt"] = state_receipt
        rows.append(summary_row)
        receipts.extend((raw, scalar_receipt, state_receipt))
        scalar_copies.append((stem, scalar_bytes, scalar_receipt))
        state_copies.append((stem, state_bytes, state_receipt))
    return {"path": str(path), "manifest": manifest, "complete": complete,
            "manifest_receipt": manifest_receipt,
            "complete_receipt": complete_receipt, "rows": rows, "receipts": receipts,
            "scalar_copies": scalar_copies, "state_copies": state_copies}


def load_prior_summary(path: Path = PRIOR_SUMMARY,
                       expected_sha256: str = PRIOR_SUMMARY_SHA256) -> dict[str, Any]:
    summary, _, receipt = _read_json(path)
    rows = summary.get("rows", [])
    identities = [(row.get("seed"), row.get("arm"), row.get("step")) for row in rows]
    if (receipt["sha256"] != expected_sha256
            or summary.get("schema") != "grokking_representation_analysis_summary_v1"
            or summary.get("fixed_frequency_panel", {}).get("frequencies") != FIXED_PANEL
            or identities != expected_prior_roster() or len(set(identities)) != 150):
        raise ValueError("archived native summary identity/roster/hash mismatch")
    raw_map = {}
    for stage in ("calibration", "remaining"):
        for raw in summary.get("input_receipts", {}).get(stage, {}).get("raw_receipts", []):
            key = (raw.get("seed"), raw.get("arm"), raw.get("step"))
            if key in raw_map:
                raise ValueError("duplicate archived raw receipt")
            raw_map[key] = raw
    if set(raw_map) != set(expected_prior_roster()):
        raise ValueError("archived raw-receipt roster mismatch")
    reference_rows = []
    for row in rows:
        key = (row["seed"], row["arm"], row["step"])
        if row.get("source", {}).get("source_npz_sha256") != raw_map[key].get("sha256"):
            raise ValueError("archived row/raw receipt binding mismatch")
        if row["arm"] == "legacy" and row["step"] in REFERENCE_STEPS:
            validate_metric_row(row)
            copied = dict(row)
            copied["policy"] = "native"
            copied["reference_origin"] = "accepted_archived_legacy"
            copied["raw_receipt"] = raw_map[key]
            reference_rows.append(copied)
    expected_reference = [(seed, step) for seed in SEEDS for step in REFERENCE_STEPS]
    if [(row["seed"], row["step"]) for row in reference_rows] != expected_reference:
        raise ValueError("archived native reference roster mismatch")
    return {"summary_receipt": receipt, "rows": reference_rows,
            "analysis_source_sha256": summary.get("analysis_source_sha256")}


class Output:
    def __init__(self, path: Path):
        storage = Path("/tmp/spectral-experiment-artifacts").resolve()
        target = Path(path).resolve()
        if (target == storage or not target.is_relative_to(storage) or target.exists()
                or not target.parent.is_dir()):
            raise ValueError("output must be an absent descendant of /tmp/spectral-experiment-artifacts")
        if shutil.disk_usage(target.parent).free < OUTPUT_LIMIT + OUTPUT_RESERVE:
            raise ValueError("insufficient output capacity plus reserve")
        target.mkdir(mode=0o700)
        (target / "source-scalars").mkdir()
        (target / "source-states").mkdir()
        self.path, self.bytes = target, 0

    def write_bytes(self, relative: Path | str, raw: bytes) -> dict[str, Any]:
        if self.bytes + len(raw) > OUTPUT_LIMIT:
            raise RuntimeError("analysis output bound exhausted")
        if shutil.disk_usage(self.path).free < len(raw) + OUTPUT_RESERVE:
            raise RuntimeError("large-volume free-space reserve exhausted")
        path = self.path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(path, flags, 0o600)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(raw)
        except BaseException:
            try:
                os.close(descriptor)
            except OSError:
                pass
            raise
        self.bytes += path.stat().st_size
        return {"path": str(path.resolve()), "sha256": file_hash(path),
                "size_bytes": path.stat().st_size}

    def write_json(self, relative: Path | str, value: Any) -> dict[str, Any]:
        raw = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
        return self.write_bytes(relative, raw)


def _resource_check(started: float) -> None:
    if time.monotonic() - started > COOPERATIVE_SECONDS:
        raise TimeoutError("five-minute analysis deadline exceeded")
    # Linux ru_maxrss is KiB; VmSwap is the process's current swap residency.
    if int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024 > MEMORY_LIMIT:
        raise MemoryError("2-GiB process memory bound exceeded")
    status = Path("/proc/self/status")
    if status.is_file():
        swap_lines = [line for line in status.read_text().splitlines()
                      if line.startswith("VmSwap:")]
        if len(swap_lines) != 1 or int(swap_lines[0].split()[1]) != 0:
            raise MemoryError("analysis process has nonzero or unreadable swap residency")


def _recheck(receipts: Iterable[dict[str, Any]], started: float) -> None:
    for receipt in receipts:
        _resource_check(started)
        path = Path(receipt["path"])
        if (path.is_symlink() or not path.is_file() or path.stat().st_nlink != 1
                or path.stat().st_size != receipt["size_bytes"]
                or file_hash(path) != receipt["sha256"]):
            raise ValueError("input changed during analysis")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurement-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    started = time.monotonic()
    _resource_check(started)
    pins = analysis_sources()
    measurement = verify_measurement(args.measurement_dir)
    prior = load_prior_summary()
    if (measurement["manifest"].get("prior_recipe", {}).get("summary")
            != prior["summary_receipt"]
            or measurement["complete"].get("prior_summary") != prior["summary_receipt"]):
        raise ValueError("measurement prior receipt differs from accepted summary receipt")
    output = Output(args.output_dir)
    manifest_receipt = output.write_json("manifest.json", {
        "schema": "grokking_action_analysis_v1",
        "measurement": measurement["complete_receipt"],
        "prior_summary": prior["summary_receipt"],
        "analysis_source_sha256": pins,
        "new_state_roster": expected_new_roster(),
        "contrast_steps": list(ENDPOINT_STEPS),
        "contrasts": [name for name, _, _ in CONTRASTS],
        "metric_definitions": METRICS,
        "bounds": {"cooperative_seconds": COOPERATIVE_SECONDS,
                   "cpu_threads": 1, "memory_bytes": MEMORY_LIMIT, "swap_bytes": 0,
                   "output_bytes": OUTPUT_LIMIT, "free_reserve_bytes": OUTPUT_RESERVE},
    })
    copies = []
    try:
        for stem, raw, source in measurement["scalar_copies"]:
            _resource_check(started)
            copied = output.write_bytes(Path("source-scalars") / f"{stem}.json", raw)
            if copied["sha256"] != source["sha256"]:
                raise RuntimeError("source scalar copy differs")
            copies.append({"kind": "scalar", "seed_policy_step": stem, **copied})
        for stem, raw, source in measurement["state_copies"]:
            _resource_check(started)
            copied = output.write_bytes(Path("source-states") / f"{stem}.json", raw)
            if copied["sha256"] != source["sha256"]:
                raise RuntimeError("source state copy differs")
            copies.append({"kind": "analyzed_state", "seed_policy_step": stem, **copied})
        contrasts = paired_contrasts(measurement["rows"], prior["rows"])
        summary = {
            "schema": "grokking_action_analysis_summary_v1",
            "new_state_count": 35,
            "new_state_rows": measurement["rows"],
            "archived_native_reference_rows": prior["rows"],
            "paired_endpoint_contrasts": contrasts,
            "source_copies": copies,
            "input_receipts": {
                "measurement_manifest": measurement["manifest_receipt"],
                "measurement_completion": measurement["complete_receipt"],
                "measurement_artifacts": measurement["receipts"],
                "archived_summary": prior["summary_receipt"],
                "archived_raw_receipts": [row["raw_receipt"] for row in prior["rows"]],
            },
            "analysis_source_sha256": pins,
            "interpretation": (
                "Fixed-endpoint paired descriptive contrasts; seed is the replicate. "
                "Differences are left minus right with sample SE; there are no p-values, "
                "equivalence claims, checkpoint/frequency pseudo-replicates, or AUC."),
        }
        summary_receipt = output.write_json("summary.json", summary)
        _resource_check(started)
        if analysis_sources() != pins:
            raise RuntimeError("analysis source changed during execution")
        _verify_source_map(measurement["manifest"]["measurement_source_sha256"])
        _verify_source_map(measurement["manifest"]["action_source_sha256"])
        _recheck(measurement["receipts"] + [measurement["manifest_receipt"],
                                            measurement["complete_receipt"]], started)
        if file_hash(prior["summary_receipt"]["path"]) != PRIOR_SUMMARY_SHA256:
            raise ValueError("archived summary changed during analysis")
        output.write_json("complete.json", {
            "schema": "grokking_action_analysis_complete_v1", "status": "complete",
            "new_state_count": 35, "contrast_count": len(contrasts),
            "summary": summary_receipt, "manifest": manifest_receipt,
            "analysis_source_sha256": pins,
            "elapsed_seconds": time.monotonic() - started,
            "output_bytes_before_completion": output.bytes,
        })
    except BaseException as error:
        try:
            output.write_json("failure.json", {
                "schema": "grokking_action_analysis_failure_v1",
                "status": "failed_preserved", "type": type(error).__name__,
                "message": str(error), "manifest": manifest_receipt,
                "source_copies": copies, "analysis_source_sha256": pins,
                "elapsed_seconds": time.monotonic() - started,
                "output_bytes_before_failure": output.bytes,
            })
        except BaseException:
            pass
        raise


if __name__ == "__main__":
    main()
