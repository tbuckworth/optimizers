#!/usr/bin/env python3
"""Strict JSON-only paired analysis of the one-arm raw-direction continuation.

Checkpoint and NPZ contents are streamed only for receipt verification. This
module imports no training/measurement module and never loads array members.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from experiments.analyze_grokking_action_results import (
    COOPERATIVE_SECONDS, ENDPOINT_STEPS, FIXED_PANEL, MEMORY_LIMIT, METRICS,
    OUTPUT_LIMIT, OUTPUT_RESERVE, Output, SEEDS, _paired_summary, _read_json,
    _recheck, _resource_check, _verify_receipt, _verify_source_map, file_hash,
    expected_new_roster as old_roster, validate_metric_row,
)

POLICY = "raw_norm_matched"
CAPTURE_STEPS = (1501, 2000, 2500)
SCHEMA = "grokking_raw_direction_analysis_v1"
ARCHIVED_SUMMARY = REPO / "output/2026-09-09-spectral-grokking-action/results/summary.json"
ARCHIVED_SUMMARY_SHA256 = "dc79ddbc9261607468c6adb5c8ff48b3e1249803f4651397aca134bcbc02a7a1"
PRIOR_SUMMARY_SHA256 = "9ac881c1bbca72a9dfd5525c226e05938ea4b0553e1e924bafb59ffd8a9a7c0d"
OLD_MEASUREMENT_SHA256 = "fe11e1404e8ab20c792d8d441477f325f48a2bb9fdc4c868620a3ffeeafbb9ee"
OLD_BATCH_SHA256 = "22bb866319fa2fa85771bd156eb406523a66abee4d0bc789cc6104d40ec86e45"
PROTOCOL = "output/2026-09-09-spectral-raw-direction/paired-analysis-protocol.md"
CONTRASTS = (("raw_minus_norm_matched", "norm_matched", "primary"),
             ("raw_minus_native", "native", "secondary"),
             ("raw_minus_orthogonal", "orthogonal", "secondary"))
ACQUISITION_PATHS = (
    "experiments/grokking_raw_direction.py", "experiments/run_grokking_raw_direction_batch.py",
    "experiments/grokking_raw_direction_policy.py", "tests/test_grokking_raw_direction.py",
    "tests/test_grokking_raw_direction_policy.py", "tests/test_grokking_raw_direction_guard.py",
    "output/2026-09-09-spectral-raw-direction/guarded_launch.py",
    "experiments/grokking_action_intervention.py", "experiments/grokking_action_policy.py",
    "experiments/grokking_confirmation.py", "experiments/grokking_model.py",
    "experiments/measure_grokking_representations.py", "experiments/grokking_representation.py",
    "spectral_filter.py", "output/2026-09-09-spectral-raw-direction/protocol.md")
MEASUREMENT_PATHS = (
    "experiments/measure_grokking_raw_direction_states.py",
    "tests/test_grokking_raw_direction_measurement.py",
    "output/2026-09-09-spectral-raw-direction/guarded_measurement.py",
    "tests/test_grokking_raw_direction_measurement_guard.py",
    "output/2026-09-09-spectral-raw-direction/measurement-protocol.md")


def expected_roster():
    return [(seed, POLICY, step) for seed in SEEDS for step in CAPTURE_STEPS]


def analysis_sources():
    paths = ("experiments/analyze_grokking_raw_direction_results.py",
             "tests/test_grokking_raw_direction_analysis.py",
             "experiments/analyze_grokking_action_results.py", PROTOCOL)
    return {path: file_hash(REPO / path) for path in paths}


def _identities(rows):
    return [(row.get("seed"), row.get("policy"), row.get("step")) for row in rows]


def _roster(rows, wanted, label):
    identities = _identities(rows)
    if identities != list(wanted) or len(set(identities)) != len(identities):
        raise ValueError(f"{label} exact ordered roster mismatch")


def _receipt_json(receipt, parent, name, receipts):
    verified = _verify_receipt(receipt, parent, name=name)
    value, _, reread = _read_json(Path(verified["path"]))
    if verified != reread:
        raise ValueError("JSON changed during receipt admission")
    receipts.append(verified)
    return value


def _archived_seed_receipt_json(entry, parent, seed, receipts):
    """Normalize only the pinned old batch's size-optional seed receipts.

    The historical entry itself remains unchanged for raw archived_reference
    identity. Its observed size is a present admission receipt, not a size
    recorded during the old acquisition. New receipts still use _receipt_json.
    """
    required = {"path", "seed", "sha256"}
    if (not isinstance(entry, dict)
            or set(entry) not in (required, required | {"size_bytes"})
            or entry.get("seed") != seed or seed not in SEEDS
            or not isinstance(entry.get("path"), str)
            or re.fullmatch(r"[0-9a-f]{64}", str(entry.get("sha256"))) is None):
        raise ValueError("malformed archived seed receipt")
    path = Path(entry["path"])
    if (not path.is_absolute() or path.is_symlink() or not path.is_file()
            or path.stat().st_nlink != 1 or path.name != "complete.json"
            or path.resolve().parent != Path(parent).resolve()):
        raise ValueError("archived seed receipt path/type/parent mismatch")
    if "size_bytes" in entry and (isinstance(entry["size_bytes"], bool)
            or not isinstance(entry["size_bytes"], int) or entry["size_bytes"] < 0):
        raise ValueError("invalid archived seed optional size")
    value, raw, observed = _read_json(path)
    if (observed["sha256"] != entry["sha256"]
            or hashlib.sha256(raw).hexdigest() != entry["sha256"]):
        raise ValueError("archived seed receipt hash mismatch")
    if "size_bytes" in entry and entry["size_bytes"] != observed["size_bytes"]:
        raise ValueError("archived seed receipt size mismatch")
    # Reuse strict checks after the old-schema-specific normalization. Do not
    # insert size_bytes into entry: the new acquisition binds its original form.
    verified = _verify_receipt(observed, parent, name="complete.json")
    receipts.append(verified)
    return value


def _identity_receipt(receipt):
    """Receipt identity without loading a hash-bound archived model/array."""
    if (not isinstance(receipt, dict) or not isinstance(receipt.get("path"), str)
            or not Path(receipt["path"]).is_absolute()
            or re.fullmatch(r"[0-9a-f]{64}", str(receipt.get("sha256"))) is None
            or isinstance(receipt.get("size_bytes"), bool)
            or not isinstance(receipt.get("size_bytes"), int)
            or receipt["size_bytes"] < 0):
        raise ValueError("malformed bound receipt identity")
    return {key: receipt[key] for key in ("path", "sha256", "size_bytes")}


def paired_contrasts(new_rows, archived_rows, native_rows):
    """All five seeds; missing secondary ratios invalidate the full aggregate."""
    new_rows, archived_rows, native_rows = map(list, (new_rows, archived_rows, native_rows))
    _roster(new_rows, expected_roster(), "new state")
    _roster(archived_rows, old_roster(), "archived action")
    _roster(native_rows, [(seed, "native", step) for seed in SEEDS
                          for step in (1500, 2000, 2500)], "archived native")
    for row in new_rows + archived_rows + native_rows:
        validate_metric_row(row)
    new = {(r["seed"], r["step"]): r for r in new_rows}
    old = {(r["seed"], r["policy"], r["step"]): r for r in archived_rows}
    old.update({(r["seed"], "native", r["step"]): r for r in native_rows})
    result = []
    for step in ENDPOINT_STEPS:
        left = {seed: new[seed, step] for seed in SEEDS}
        for name, comparator, role in CONTRASTS:
            right = {seed: old[seed, comparator, step] for seed in SEEDS}
            metrics = {}
            for metric, definition in METRICS.items():
                value = _paired_summary(left, right, metric)
                sign_key = {"lower": "negative_count", "higher": "positive_count"}.get(
                    definition["favorable_direction"])
                value["favorable_count"] = value[sign_key] if sign_key else None
                # The reused helper rejects nonfinite inputs/differences; also
                # reject arithmetic overflow in its aggregate computations.
                if any(value[key] is not None and not math.isfinite(value[key])
                       for key in ("mean_difference", "sample_sd", "sample_se")):
                    raise ValueError("nonfinite paired aggregate")
                metrics[metric] = value
            result.append({"step": step, "contrast": name, "contrast_role": role,
                           "endpoint_role": "primary" if step == 2500 else "fixed_earlier",
                           "left_policy": POLICY, "right_policy": comparator,
                           "right_origin": "accepted_archived_reference", "metrics": metrics})
    return result


def load_archived_summary(path=ARCHIVED_SUMMARY, expected_sha256=ARCHIVED_SUMMARY_SHA256):
    """Admit trusted saved rows; do not repeat old aggregation or inference."""
    summary, _, receipt = _read_json(Path(path))
    if (receipt["sha256"] != expected_sha256
            or summary.get("schema") != "grokking_action_analysis_summary_v1"
            or summary.get("new_state_count") != 35):
        raise ValueError("accepted archived summary hash/schema mismatch")
    rows = summary.get("new_state_rows", [])
    native = summary.get("archived_native_reference_rows", [])
    _roster(rows, old_roster(), "archived action")
    _roster(native, [(seed, "native", step) for seed in SEEDS
                     for step in (1500, 2000, 2500)], "archived native")
    inputs = summary["input_receipts"]
    prior = _identity_receipt(inputs["archived_summary"])
    if prior["sha256"] != PRIOR_SUMMARY_SHA256:
        raise ValueError("archived summary lacks accepted original representation binding")
    for row in rows + native:
        validate_metric_row(row)
        raw = _identity_receipt(row["raw_receipt"])
        if row.get("source", {}).get("source_npz_sha256") != raw["sha256"]:
            raise ValueError("archived row/raw receipt binding mismatch")
    for row in rows:
        source = row.get("source", {})
        provenance = row.get("checkpoint_provenance", {})
        if (source.get("source_scalar_sha256") != row["source_scalar_receipt"]["sha256"]
                or source.get("action_checkpoint_sha256")
                   != provenance.get("action_checkpoint", {}).get("sha256")
                or (provenance.get("outer_seed"), provenance.get("outer_policy"),
                    provenance.get("outer_step")) != (row["seed"], row["policy"], row["step"])):
            raise ValueError("archived scalar/checkpoint binding mismatch")
    if any(row.get("arm") != "legacy"
           or row.get("reference_origin") != "accepted_archived_legacy" for row in native):
        raise ValueError("archived native reference origin mismatch")
    receipts = [receipt]
    old_path = Path(inputs["measurement_completion"]["path"]).parent
    complete = _receipt_json(inputs["measurement_completion"], old_path, "complete.json", receipts)
    manifest = _receipt_json(inputs["measurement_manifest"], old_path, "manifest.json", receipts)
    if (inputs["measurement_completion"]["sha256"] != OLD_MEASUREMENT_SHA256
            or complete.get("schema") != "grokking_action_measurement_complete_v1"
            or complete.get("status") != "complete" or complete.get("state_count") != 35
            or complete.get("manifest") != inputs["measurement_manifest"]
            or complete.get("prior_summary") != prior
            or manifest.get("schema") != "grokking_action_measurement_v1"
            or manifest.get("prior_recipe", {}).get("summary") != prior
            or complete.get("measurement_source_sha256") != manifest.get("measurement_source_sha256")
            or complete.get("action_source_sha256") != manifest.get("action_source_sha256")
            or manifest.get("fixed_frequency_panel", {}).get("frequencies") != FIXED_PANEL):
        raise ValueError("accepted old measurement completion/source binding mismatch")
    _verify_source_map(summary["analysis_source_sha256"])
    _verify_source_map(manifest["measurement_source_sha256"])
    _verify_source_map(manifest["action_source_sha256"])
    old_batch = manifest["action_batch"]
    old_parent = Path(old_batch["path"])
    batch = _receipt_json(old_batch["completion"], old_parent, "batch-complete.json", receipts)
    if (old_batch["completion"]["sha256"] != OLD_BATCH_SHA256
            or complete.get("input_batch_completion") != old_batch["completion"]
            or batch.get("status") != "complete"
            or [entry.get("seed") for entry in batch.get("accepted", [])] != list(SEEDS)):
        raise ValueError("accepted archived acquisition binding mismatch")
    old_seeds = {}
    for seed, entry in zip(SEEDS, batch["accepted"]):
        old_seed = _archived_seed_receipt_json(entry, old_parent / f"seed{seed}", seed, receipts)
        if (old_seed.get("status") != "complete" or old_seed.get("seed") != seed
                or old_seed.get("source_sha256") != manifest["action_source_sha256"]):
            raise ValueError("archived acquisition seed/source mismatch")
        old_seeds[seed] = {"complete": old_seed, "completion": entry}
    return {"summary_receipt": receipt, "rows": rows, "native_rows": native,
            "prior_summary": prior, "old_manifest": manifest, "old_complete": complete,
            "receipts": receipts, "old_completion_receipt": inputs["measurement_completion"],
            "old_manifest_receipt": inputs["measurement_manifest"], "old_seeds": old_seeds}


def _verify_batch(manifest, complete, parent, sources, receipts, started):
    shared = ("schema", "seeds", "policy", "source_sha256", "source_commit",
              "first_seed_is_resource_only_admission", "paid_spend_usd",
              "batch_seconds_limit", "seed_seconds_limit")
    if (any(manifest.get(key) != complete.get(key) for key in shared)
            or complete.get("schema") != "grokking_raw_direction_batch_v1"
            or complete.get("status") != "complete" or complete.get("seeds") != list(SEEDS)
            or complete.get("policy") != POLICY or complete.get("source_sha256") != sources
            or complete.get("first_seed_is_resource_only_admission") is not True
            or complete.get("paid_spend_usd") != 0
            or complete.get("batch_seconds_limit") != 43200
            or complete.get("seed_seconds_limit") != 10800
            or not 0 <= complete.get("elapsed_seconds", math.inf) <= 43200
            or re.fullmatch(r"[0-9a-f]{40}", str(complete.get("source_commit"))) is None
            or [row.get("seed") for row in complete.get("accepted", [])] != list(SEEDS)
            or (parent / "batch-failure.json").exists()):
        raise ValueError("raw batch completion/source/roster mismatch")
    checkpoints, seeds, admission = {}, {}, None
    for seed, accepted in zip(SEEDS, complete["accepted"]):
        _resource_check(started)
        directory = parent / f"seed{seed}"
        seed_complete = _receipt_json(accepted, directory, "complete.json", receipts)
        seed_manifest, _, seed_manifest_receipt = _read_json(directory / "manifest.json")
        receipts.append(seed_manifest_receipt)
        if (any(seed_complete.get(key) != value for key, value in seed_manifest.items())
                or seed_manifest.get("schema") != "grokking_raw_direction_acquisition_v1"
                or seed_complete.get("status") != "complete"
                or seed_complete.get("seed") != seed or seed_complete.get("policy") != POLICY
                or seed_complete.get("source_sha256") != sources
                or seed_complete.get("environment") != complete.get("environment")
                or seed_complete.get("seed100_admission") != admission
                or seed_complete.get("capture_steps") != list(CAPTURE_STEPS)
                or seed_complete.get("accepted_roster") != [[POLICY, step] for step in CAPTURE_STEPS]
                or seed_complete.get("completed_updates") != 1000
                or seed_complete.get("history_steps") != list(range(1501, 2501))
                or seed_complete.get("no_reference_policy_execution") is not True
                or not 0 <= seed_complete.get("elapsed_seconds", math.inf) <= 10800
                or (directory / "failure.json").exists()):
            raise ValueError("raw seed completion/source/roster mismatch")
        if seed == 100:
            admission = _identity_receipt(accepted)
        expected_names = {"manifest.json", "first-step-tensors.pt", "first-step-summary.json"}
        expected_names.update(f"{POLICY}-step-{step:06d}.pt" for step in CAPTURE_STEPS)
        expected_names.update(f"{POLICY}-through-{step:06d}.json" for step in CAPTURE_STEPS)
        artifacts = seed_complete.get("artifact_receipts", [])
        names = [Path(item["path"]).name for item in artifacts]
        if (len(names) != len(expected_names) or set(names) != expected_names
                or {path.name for path in directory.iterdir()} != expected_names | {"complete.json"}):
            raise ValueError("raw seed artifact roster mismatch")
        by_name = {}
        for item in artifacts:
            _resource_check(started)
            name = Path(item["path"]).name
            verified = _verify_receipt(item, directory, name=name)
            receipts.append(verified)
            by_name[name] = item
        if (_identity_receipt(by_name["manifest.json"]) != seed_manifest_receipt
                or seed_complete.get("checkpoints") != [by_name[f"{POLICY}-step-{step:06d}.pt"]
                                                         for step in CAPTURE_STEPS]):
            raise ValueError("raw checkpoint/manifest receipt binding mismatch")
        for step in CAPTURE_STEPS:
            item = by_name[f"{POLICY}-step-{step:06d}.pt"]
            if (item.get("seed"), item.get("policy"), item.get("step")) != (seed, POLICY, step):
                raise ValueError("raw checkpoint receipt identity mismatch")
            checkpoints[seed, POLICY, step] = _identity_receipt(item)
        seeds[seed] = {"complete": seed_complete, "completion": _identity_receipt(accepted),
                       "manifest": seed_manifest_receipt}
    return checkpoints, seeds


def verify_measurement(path, archived, started=None):
    started = time.monotonic() if started is None else started
    path = Path(path).resolve()
    if (path / "failure.json").exists():
        raise ValueError("measurement has failure marker")
    manifest, _, manifest_receipt = _read_json(path / "manifest.json")
    complete, _, complete_receipt = _read_json(path / "complete.json")
    accepted = complete.get("accepted_results", [])
    _roster(accepted, expected_roster(), "measured state")
    if (manifest.get("schema") != "grokking_raw_direction_measurement_v1"
            or complete.get("schema") != "grokking_raw_direction_measurement_complete_v1"
            or complete.get("status") != "complete" or complete.get("state_count") != 15
            or [tuple(row) for row in manifest.get("roster", [])] != expected_roster()
            or complete.get("manifest") != manifest_receipt
            or manifest.get("fixed_frequency_panel", {}).get("frequencies") != FIXED_PANEL
            or complete.get("prior_summary") != archived["prior_summary"]
            or manifest.get("prior_recipe") != archived["old_manifest"].get("prior_recipe")
            or manifest.get("recipe") != archived["old_manifest"].get("recipe")
            or manifest.get("measurement_source_sha256") != complete.get("measurement_source_sha256")
            or manifest.get("acquisition_source_sha256") != complete.get("acquisition_source_sha256")):
        raise ValueError("measurement completion/recipe/source contract mismatch")
    old_sources = manifest.get("accepted_old_measurement_sources", {})
    if (old_sources.get("completion") != archived["old_completion_receipt"]
            or old_sources.get("manifest") != archived["old_manifest_receipt"]
            or old_sources.get("measurement_source_sha256")
               != archived["old_manifest"]["measurement_source_sha256"]):
        raise ValueError("measurement does not bind accepted old measurement")
    sources = _verify_source_map(manifest["acquisition_source_sha256"])
    measurement_sources = _verify_source_map(manifest["measurement_source_sha256"])
    for key, sha in archived["old_manifest"]["action_source_sha256"].items():
        if sources.get(key) != sha:
            raise ValueError("acquisition lost original scientific source binding")
    expected_acquisition = {**archived["old_manifest"]["action_source_sha256"],
                            **{key: file_hash(REPO / key) for key in ACQUISITION_PATHS}}
    if sources != expected_acquisition:
        raise ValueError("incomplete raw acquisition scientific source map")
    required_measurement = {**old_sources["measurement_source_sha256"], **sources}
    for key in MEASUREMENT_PATHS:
        required_measurement[key] = file_hash(REPO / key)
    if measurement_sources != required_measurement:
        raise ValueError("measurement source map is not the full recipe/acquisition union")
    batch = manifest.get("raw_batch", {})
    parent = Path(batch["path"]).resolve()
    receipts = [manifest_receipt, complete_receipt]
    batch_manifest = _receipt_json(batch["manifest"], parent, "batch-manifest.json", receipts)
    batch_complete = _receipt_json(batch["completion"], parent, "batch-complete.json", receipts)
    if (complete.get("input_batch_completion") != batch.get("completion")
            or batch_complete.get("batch_manifest") != batch.get("manifest")):
        raise ValueError("measurement/raw-batch receipt mismatch")
    checkpoints, seeds = _verify_batch(batch_manifest, batch_complete, parent, sources, receipts, started)
    rows, scalar_copies, state_copies = [], [], []
    old_by_seed = {row["seed"]: row for row in archived["rows"] if row["policy"] == "native"}
    for item in accepted:
        _resource_check(started)
        identity = tuple(item[key] for key in ("seed", "policy", "step"))
        seed, policy, step = identity
        if item.get("checkpoint") != checkpoints[identity]:
            raise ValueError("measurement checkpoint not bound to completed raw batch")
        stem = f"seed{seed}-{policy}-step{step:06d}"
        raw = _verify_receipt(item["raw"], path / "raw", name=stem + ".npz")
        scalar_receipt = _verify_receipt(item["scalar"], path / "scalars", name=stem + ".json")
        state_receipt = _verify_receipt(item["analyzed_state"], path / "states", name=stem + ".json")
        scalar, scalar_bytes, scalar_reread = _read_json(Path(scalar_receipt["path"]))
        state, state_bytes, state_reread = _read_json(Path(state_receipt["path"]))
        provenance = state.get("checkpoint_provenance", {})
        seed_info = seeds[seed]
        seed_complete = seed_info["complete"]
        old_seed = archived["old_seeds"][seed]
        old_provenance = old_by_seed[seed]["checkpoint_provenance"]
        recipe = manifest["prior_recipe"]["seeds"][str(seed)]
        if (scalar_reread != scalar_receipt or state_reread != state_receipt
                or scalar.get("schema") != "grokking_raw_direction_measurement_state_v1"
                or state.get("schema") != "grokking_raw_direction_analyzed_state_v1"
                or _identities([scalar, state]) != [identity, identity]
                or scalar.get("arm") != policy or state.get("arm") != policy
                or scalar.get("raw_activations") != raw
                or scalar.get("source_sha256") != measurement_sources
                or state.get("source", {}).get("stage") != "raw_direction"
                or state.get("source", {}).get("source_scalar_sha256") != scalar_receipt["sha256"]
                or state.get("source", {}).get("source_npz_sha256") != raw["sha256"]
                or state.get("source", {}).get("raw_direction_checkpoint_sha256") != item["checkpoint"]["sha256"]
                or scalar.get("checkpoint_provenance") != provenance
                or (provenance.get("outer_seed"), provenance.get("outer_policy"), provenance.get("outer_step")) != identity
                or provenance.get("raw_direction_checkpoint") != item["checkpoint"]
                or provenance.get("raw_direction_seed_completion") != seed_info["completion"]
                or provenance.get("raw_direction_seed_manifest") != seed_info["manifest"]
                or provenance.get("acquisition_source_sha256") != sources
                or provenance.get("parent_legacy_checkpoint") != seed_complete.get("parent_checkpoint")
                or provenance.get("parent_legacy_checkpoint") != old_provenance.get("parent_legacy_checkpoint")
                or provenance.get("parent_metrics_sha256") != seed_complete.get("parent_metrics_sha256")
                or provenance.get("parent_metrics_sha256") != old_provenance.get("parent_metrics_sha256")
                or seed_complete.get("archived_reference", {}).get("batch_completion_sha256") != OLD_BATCH_SHA256
                or seed_complete.get("archived_reference", {}).get("seed_completion") != old_seed["completion"]
                or seed_complete.get("fork_scientific_state_sha256") != old_seed["complete"].get("fork_scientific_state_sha256")
                or seed_complete.get("archived_reference", {}).get("fork_scientific_state_sha256")
                   != seed_complete.get("fork_scientific_state_sha256")
                or scalar.get("prior_recipe_contract") != recipe or state.get("prior_recipe_contract") != recipe
                or scalar.get("probe_split_sha256") != recipe.get("probe_split_sha256")):
            raise ValueError("new scalar/analyzed/checkpoint/parent/recipe binding mismatch")
        for split in ("train", "test"):
            for key in ("loss", "accuracy", "count"):
                if state.get("behavior", {}).get(split, {}).get(key) != scalar.get("behavior", {}).get(split, {}).get(key):
                    raise ValueError("analyzed row changes scalar behavioral measurements")
        for feature in ("final_hidden", "pre_attention"):
            probe = scalar.get("probes", {}).get(feature, {})
            observed = probe.get("observed", {})
            analyzed_probe = state.get("probes", {}).get(feature, {})
            if (analyzed_probe.get("selected_frequencies") != observed.get("selected_frequencies")
                    or analyzed_probe.get("selected_eval_mean_r2") != observed.get("selected_eval_mean_r2")
                    or analyzed_probe.get("null_max_eval_mean_r2") != probe.get("null", {}).get("selected_eval_mean_r2_max")):
                raise ValueError("analyzed row changes scalar selected/null probe measurements")
        validate_metric_row(state)
        copied = {key: value for key, value in state.items() if key != "full_symmetry"}
        copied.update(raw_receipt=raw, source_scalar_receipt=scalar_receipt,
                      source_state_receipt=state_receipt)
        rows.append(copied)
        receipts.extend((raw, scalar_receipt, state_receipt))
        scalar_copies.append((stem, scalar_bytes, scalar_receipt))
        state_copies.append((stem, state_bytes, state_receipt))
    return {"manifest": manifest, "manifest_receipt": manifest_receipt,
            "complete_receipt": complete_receipt, "rows": rows,
            "receipts": list({item["path"]: item for item in receipts}.values()),
            "scalar_copies": scalar_copies, "state_copies": state_copies}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurement-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    started = time.monotonic()
    _resource_check(started)
    pins = analysis_sources()
    archived = load_archived_summary()
    measured = verify_measurement(args.measurement_dir, archived, started)
    _resource_check(started)
    output = Output(args.output_dir)
    manifest_receipt = output.write_json("manifest.json", {
        "schema": SCHEMA, "measurement": measured["complete_receipt"],
        "archived_summary": archived["summary_receipt"], "analysis_source_sha256": pins,
        "new_state_roster": expected_roster(), "metric_definitions": METRICS,
        "contrast_steps": list(ENDPOINT_STEPS), "contrasts": CONTRASTS,
        "bounds": {"cooperative_seconds": COOPERATIVE_SECONDS, "cpu_threads": 1,
                   "memory_bytes": MEMORY_LIMIT, "swap_bytes": 0,
                   "output_bytes": OUTPUT_LIMIT, "free_reserve_bytes": OUTPUT_RESERVE}})
    copies = []
    try:
        for kind, directory, entries in (
                ("scalar", "source-scalars", measured["scalar_copies"]),
                ("analyzed_state", "source-states", measured["state_copies"])):
            for stem, raw, source in entries:
                _resource_check(started)
                copied = output.write_bytes(Path(directory) / f"{stem}.json", raw)
                if copied["sha256"] != source["sha256"]:
                    raise ValueError("new-state source copy differs")
                copies.append({"kind": kind, "seed_policy_step": stem, **copied})
        contrasts = paired_contrasts(measured["rows"], archived["rows"], archived["native_rows"])
        summary = output.write_json("summary.json", {
            "schema": "grokking_raw_direction_analysis_summary_v1", "new_state_count": 15,
            "new_state_rows": measured["rows"], "archived_action_reference_rows": archived["rows"],
            "archived_native_reference_rows": archived["native_rows"],
            "paired_endpoint_contrasts": contrasts, "source_copies": copies,
            "input_receipts": {"measurement_manifest": measured["manifest_receipt"],
                               "measurement_completion": measured["complete_receipt"],
                               "measurement_and_acquisition_artifacts": measured["receipts"],
                               "archived_summary": archived["summary_receipt"],
                               "archived_binding_receipts": archived["receipts"]},
            "analysis_source_sha256": pins,
            "interpretation": "Five paired seeds, left minus archived right, sample SE. "
                "Same functional norm law, not identical numeric scale, observer or Adam histories. "
                "All references are archived and CUDA-sensitive. No p-values, equivalence, "
                "mediation, pre-fork formation, semantic usefulness, safety or speed conclusion. "
                "Independent first-action tensor and saved-array arithmetic audit still required."})
        _resource_check(started)
        if analysis_sources() != pins:
            raise ValueError("analysis source changed")
        for key in ("measurement_source_sha256", "acquisition_source_sha256"):
            _verify_source_map(measured["manifest"][key])
        _recheck(measured["receipts"] + archived["receipts"], started)
        output.write_json("complete.json", {
            "schema": "grokking_raw_direction_analysis_complete_v1", "status": "complete",
            "new_state_count": 15, "contrast_count": len(contrasts), "summary": summary,
            "manifest": manifest_receipt, "analysis_source_sha256": pins,
            "elapsed_seconds": time.monotonic() - started,
            "output_bytes_before_completion": output.bytes})
    except BaseException as error:
        try:
            output.write_json("failure.json", {
                "schema": "grokking_raw_direction_analysis_failure_v1", "status": "failed_preserved",
                "type": type(error).__name__, "message": str(error), "manifest": manifest_receipt,
                "source_copies": copies, "elapsed_seconds": time.monotonic() - started})
        except BaseException:
            pass
        raise


if __name__ == "__main__":
    main()
