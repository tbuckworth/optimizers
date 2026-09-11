#!/usr/bin/env python3
"""Read only the 15 newly acquired raw-direction checkpoints, exactly once."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from experiments.grokking_raw_direction import (
    ARCHIVED_BATCH, BATCH_SECONDS, CAPTURE_STEPS, CHECKPOINT_SCHEMA as OUTER_SCHEMA,
    POLICY, SCHEMA as ACQUISITION_SCHEMA, SEEDS, SEED_SECONDS, STORAGE,
    archived_parent, source_pins as acquisition_source_pins, verify_completion,
)
from experiments.measure_grokking_action_states import (
    CHECKPOINT_SCHEMA, COOPERATIVE_SECONDS, FILTER_MUTABLE, FIXED_PANEL,
    INNER_KEYS, MEMORY_LIMIT, MODEL, OUTPUT_LIMIT, OUTPUT_RESERVE, RECIPE,
    Output, _array_hash, _check_resources, _parent_receipt, _peak_rss_bytes,
    _read_json, _recheck, _structural_arrays, _verify_receipt, analyze_state,
    behavior, canonical_hash, corpus, current_cuda_environments,
    evaluate_fourier_probes, extract_activations, file_hash, finite_tree,
    GrokkingTransformer, load_checkpoint, load_prior_recipe_contracts,
    measurement_sources as original_measurement_sources,
    parameter_identity, prior_analysis_sources, prior_measurement_sources,
    tensor_set_identity, validate_environment_contract,
)

SCHEMA = "grokking_raw_direction_measurement_v1"
STATE_SCHEMA = "grokking_raw_direction_measurement_state_v1"
COMPLETE_SCHEMA = "grokking_raw_direction_measurement_complete_v1"
PROTOCOL = "output/2026-09-09-spectral-raw-direction/measurement-protocol.md"
OLD_MEASUREMENT = ARCHIVED_BATCH / "measurement-001"
OLD_MEASUREMENT_COMPLETE_SHA256 = "fe11e1404e8ab20c792d8d441477f325f48a2bb9fdc4c868620a3ffeeafbb9ee"
OUTER_KEYS = {"schema", "policy", "seed", "step", "parent_checkpoint", "source_sha256",
              "first_step_tensors", "native_compatible_state", "note"}


def expected_roster():
    return [(seed, POLICY, step) for seed in SEEDS for step in CAPTURE_STEPS]


def measurement_sources():
    paths = ("experiments/measure_grokking_raw_direction_states.py",
             "tests/test_grokking_raw_direction_measurement.py",
             "tests/test_grokking_raw_direction_measurement_guard.py",
             "output/2026-09-09-spectral-raw-direction/guarded_measurement.py", PROTOCOL)
    return {**original_measurement_sources(), **acquisition_source_pins(),
            **{name: file_hash(REPO / name) for name in paths}}


def verify_old_measurement_sources():
    """Check old receipt/source JSON only, without loading old models or logits."""
    complete, receipt = _read_json(OLD_MEASUREMENT / "complete.json")
    if (receipt["sha256"] != OLD_MEASUREMENT_COMPLETE_SHA256
            or complete.get("schema") != "grokking_action_measurement_complete_v1"
            or complete.get("status") != "complete" or complete.get("state_count") != 35):
        raise ValueError("Accepted action-measurement completion changed")
    manifest_receipt = _verify_receipt(complete["manifest"], OLD_MEASUREMENT,
                                       exact_name="manifest.json")
    manifest, reread = _read_json(OLD_MEASUREMENT / "manifest.json")
    current = original_measurement_sources()
    if (reread != manifest_receipt or manifest.get("schema") != "grokking_action_measurement_v1"
            or manifest.get("measurement_source_sha256") != current
            or complete.get("measurement_source_sha256") != current
            or manifest.get("recipe") != RECIPE
            or manifest.get("fixed_frequency_panel") != FIXED_PANEL):
        raise ValueError("Accepted action-measurement source/recipe binding changed")
    return {"completion": receipt, "manifest": manifest_receipt,
            "measurement_source_sha256": current}, [receipt, manifest_receipt]


def verify_raw_batch(parent, records, expected_sources):
    """Admit only the complete new batch, with every history and tensor bound."""
    parent = Path(parent).resolve()
    if parent.parent != STORAGE.resolve() or not parent.is_dir():
        raise ValueError("Raw batch must be directly under large-volume tmp")
    manifest, manifest_receipt = _read_json(parent / "batch-manifest.json")
    complete, complete_receipt = _read_json(parent / "batch-complete.json")
    batch_keys = ("schema", "seeds", "policy", "source_sha256", "source_commit",
                  "first_seed_is_resource_only_admission", "paid_spend_usd",
                  "batch_seconds_limit", "seed_seconds_limit")
    if (any(complete.get(key) != manifest.get(key) for key in batch_keys)
            or manifest.get("schema") != "grokking_raw_direction_batch_v1"
            or manifest.get("seeds") != list(SEEDS) or manifest.get("policy") != POLICY
            or manifest.get("source_sha256") != expected_sources
            or manifest.get("first_seed_is_resource_only_admission") is not True
            or manifest.get("paid_spend_usd") != 0
            or manifest.get("batch_seconds_limit") != BATCH_SECONDS
            or manifest.get("seed_seconds_limit") != SEED_SECONDS
            or complete.get("status") != "complete"
            or complete.get("elapsed_seconds", float("inf")) > BATCH_SECONDS
            or re.fullmatch(r"[0-9a-f]{40}", str(manifest.get("source_commit"))) is None
            or [item.get("seed") for item in complete.get("accepted", [])] != list(SEEDS)
            or _verify_receipt(complete["batch_manifest"], parent,
                               exact_name="batch-manifest.json") != manifest_receipt):
        raise ValueError("Raw batch manifest/completion/source contract mismatch")
    if (parent / "batch-failure.json").exists():
        raise ValueError("Completed raw batch also has a failure marker")
    states, receipts, seed_commits = [], [manifest_receipt, complete_receipt], {}
    env = complete["environment"]
    seed100_receipt = None
    common_keys = ("schema", "seed", "policy", "parent_checkpoint", "parent_metrics_sha256",
                   "source_sha256", "source_commit", "environment",
                   "fork_scientific_state_sha256", "archived_reference", "seed100_admission",
                   "capture_steps", "no_reference_policy_execution")
    for seed, accepted in zip(SEEDS, complete["accepted"]):
        directory = parent / f"seed{seed}"
        completion_receipt = _verify_receipt(accepted, directory, exact_name="complete.json")
        seed_complete = verify_completion(directory, seed, expected_sources, env)
        reread, reread_receipt = _read_json(directory / "complete.json")
        if seed_complete != reread or completion_receipt != reread_receipt:
            raise ValueError("Seed completion changed during admission")
        seed_manifest, seed_manifest_receipt = _read_json(directory / "manifest.json")
        if (any(seed_complete.get(key) != seed_manifest.get(key) for key in common_keys)
                or seed_manifest.get("schema") != ACQUISITION_SCHEMA
                or seed_manifest.get("capture_steps") != list(CAPTURE_STEPS)
                or seed_manifest.get("no_reference_policy_execution") is not True
                or re.fullmatch(r"[0-9a-f]{40}", str(seed_manifest.get("source_commit"))) is None):
            raise ValueError("Raw seed manifest/source identity mismatch")
        if seed == 100:
            if seed_manifest.get("seed100_admission") is not None:
                raise ValueError("Seed100 cannot inherit another admission")
            seed100_receipt = completion_receipt
        elif seed_manifest.get("seed100_admission") != seed100_receipt:
            raise ValueError("Remaining seed is not bound to this batch's seed100 admission")
        record, metrics_hash = records[seed, "legacy"]
        parent_checkpoint = _parent_receipt(record)
        reference = archived_parent(seed, parent_checkpoint, metrics_hash)
        if (seed_manifest.get("parent_checkpoint") != parent_checkpoint
                or seed_manifest.get("parent_metrics_sha256") != metrics_hash
                or seed_manifest.get("archived_reference") != reference
                or seed_manifest.get("fork_scientific_state_sha256")
                   != reference["fork_scientific_state_sha256"]):
            raise ValueError("Raw seed differs from the accepted original legacy fork")
        receipts.extend((completion_receipt, seed_manifest_receipt))
        by_name = {}
        for artifact in seed_complete["artifact_receipts"]:
            name = Path(artifact["path"]).name
            by_name[name] = artifact
            receipts.append(_verify_receipt(artifact, directory, exact_name=name))
        for checkpoint in seed_complete["checkpoints"]:
            step = checkpoint["step"]
            if checkpoint.get("seed") != seed or checkpoint.get("policy") != POLICY:
                raise ValueError("Raw checkpoint receipt identity mismatch")
            states.append({"seed": seed, "policy": POLICY, "step": step,
                           "checkpoint": _verify_receipt(checkpoint, directory,
                               exact_name=f"{POLICY}-step-{step:06d}.pt"),
                           "first_step_tensors": by_name["first-step-tensors.pt"],
                           "seed_completion": completion_receipt,
                           "seed_manifest": seed_manifest_receipt,
                           "completion": seed_complete, "record": record,
                           "parent_metrics_sha256": metrics_hash})
        seed_commits[seed] = seed_manifest["source_commit"]
    if [(item["seed"], item["policy"], item["step"]) for item in states] != expected_roster():
        raise ValueError("Raw input is not the exact ordered 15-state roster")
    unique_receipts = {item["path"]: item for item in receipts}
    return {"path": str(parent), "states": states, "manifest_receipt": manifest_receipt,
            "completion_receipt": complete_receipt, "input_receipts": list(unique_receipts.values()),
            "environment": env, "source_sha256": expected_sources,
            "source_commits": {"batch_start": manifest["source_commit"], "seeds": seed_commits}}


def validate_envelope(envelope, item, split_identity):
    seed, policy, step = item["seed"], item["policy"], item["step"]
    completion, record = item["completion"], item["record"]
    if (not isinstance(envelope, dict) or set(envelope) != OUTER_KEYS
            or envelope["schema"] != OUTER_SCHEMA
            or policy != POLICY or seed not in SEEDS or step not in CAPTURE_STEPS
            or (envelope["seed"], envelope["policy"], envelope["step"]) != (seed, policy, step)
            or envelope["source_sha256"] != completion["source_sha256"]
            or envelope["parent_checkpoint"] != completion["parent_checkpoint"]
            or envelope["first_step_tensors"] != item["first_step_tensors"]):
        raise ValueError("Outer raw-direction envelope identity mismatch")
    inner = envelope["native_compatible_state"]
    if not isinstance(inner, dict) or set(inner) != INNER_KEYS or not finite_tree(inner):
        raise ValueError("Invalid inherited legacy state")
    if (inner["schema"] != CHECKPOINT_SCHEMA or inner["step"] != step
            or inner["config"] != record["config"]
            or inner["config_sha256"] != canonical_hash(record["config"])
            or inner["source_identity"] != record["source_identity"]
            or inner["source_identity_sha256"] != canonical_hash(record["source_identity"])
            or inner["split_identity"] != split_identity
            or inner["split_identity"] != record["split_identity"]
            or inner["parameter_identity"] != record["parameter_identity"]
            or (inner["config"]["seed"], inner["config"]["arm"]) != (seed, "legacy")
            or inner["device_type"] != completion["environment"]["device"]
            or inner["device_type"] != "cuda"
            or inner["cuda_device_count"] != len(inner["torch_cuda_rng_states"])
            or inner["cuda_device_count"] < 1):
        raise ValueError("Inherited config/source/split/device identity mismatch")
    filter_state = inner["filter_state"]
    parameters = sum(math.prod(row["shape"]) for row in inner["parameter_identity"])
    expected_filter = {**record["config"]["filter"], "n_params": parameters}
    if (not isinstance(filter_state, dict) or set(filter_state) != set(FILTER_MUTABLE)
            or filter_state["step_count"] != step
            or inner["filter_configuration"] != expected_filter
            or expected_filter.get("stable_update") is not False):
        raise ValueError("Inherited legacy filter configuration/counter mismatch")
    evaluations = [] if step == 1501 else list(range(1550, step + 1, 50))
    if [row.get("step") for row in inner["evaluation_rows"]] != evaluations:
        raise ValueError("Raw continuation evaluation prefix mismatch")
    return inner


def check_training_grid(behavior_record, evaluations, step):
    matching = [row for row in evaluations if row["step"] == step]
    if step == 1501:
        if matching:
            raise ValueError("Step1501 must not have an extra training-grid evaluation")
        return
    if len(matching) != 1:
        raise ValueError("Fixed endpoint lacks its original training-grid behavior")
    for name in ("train", "test"):
        if (abs(behavior_record[name]["loss"] - matching[0][name]["loss"]) > 1e-4
                or abs(behavior_record[name]["accuracy"] - matching[0][name]["accuracy"]) > 1e-6):
            raise ValueError("Saved raw state fails original training-grid reproduction tolerance")


def measure_one(item, prior_seed, all_pairs, all_sums, output, pins):
    """One new model extraction; remaining calculations reuse its saved arrays."""
    seed, policy, step = item["seed"], item["policy"], item["step"]
    if (seed, policy, step) not in expected_roster():
        raise ValueError("Cannot measure an old or unregistered state")
    structural, recipe = _structural_arrays(seed, all_pairs, all_sums)
    if (recipe["split"]["sha256"] != prior_seed["probe_split_sha256"]
            or any(_array_hash(name, array) != prior_seed["structural_array_sha256"][name]
                   for name, array in structural.items())):
        raise ValueError("Generated recipe differs from accepted native structural arrays")
    envelope = load_checkpoint(Path(item["checkpoint"]["path"]), item["checkpoint"]["sha256"])
    inner = validate_envelope(envelope, item, tensor_set_identity(recipe["data"]))
    model = GrokkingTransformer(**MODEL).to("cuda")
    model.load_state_dict(inner["model_state"], strict=True)
    if (parameter_identity(model) != inner["parameter_identity"]
            or any(not bool(torch.isfinite(value).all()) for value in model.state_dict().values()
                   if value.is_floating_point())):
        raise ValueError("Loaded new model parameter layout/finiteness mismatch")
    rng_before = (torch.get_rng_state().clone(),
                  [value.clone() for value in torch.cuda.get_rng_state_all()])
    activations = extract_activations(model, all_pairs, batch_size=RECIPE["batch_size"])
    cuda_after = torch.cuda.get_rng_state_all()
    if (not torch.equal(rng_before[0], torch.get_rng_state())
            or len(rng_before[1]) != len(cuda_after)
            or any(not torch.equal(a, b) for a, b in zip(rng_before[1], cuda_after))):
        raise ValueError("New-state extraction changed global RNG")
    if any(not bool(torch.isfinite(value).all()) for value in activations.values()):
        raise ValueError("Nonfinite new-state activations")
    arrays = {name: value.numpy() for name, value in activations.items()}
    arrays.update(structural)
    stem = f"seed{seed}-{policy}-step{step:06d}"
    raw_receipt = output.write_npz(Path("raw") / f"{stem}.npz", arrays)
    behavior_record = {name: behavior(activations["logits"], all_sums, ids)
                       for name, ids in (("train", recipe["train_ids"]),
                                         ("test", recipe["test_ids"]))}
    check_training_grid(behavior_record, inner["evaluation_rows"], step)
    probes = {feature: evaluate_fourier_probes(
        activations[feature][recipe["test_ids"]], recipe["y"],
        recipe["split"]["fit_indices"], recipe["split"]["eval_indices"],
        p=RECIPE["p"], ridge=RECIPE["ridge"], top_k=RECIPE["top_k"],
        null_seed=RECIPE["null_seed_offset"] + seed, n_nulls=RECIPE["n_nulls"],
        null_permutations=recipe["permutations"]) for feature in RECIPE["features"]}
    provenance = {"raw_direction_checkpoint": item["checkpoint"],
                  "raw_direction_seed_completion": item["seed_completion"],
                  "raw_direction_seed_manifest": item["seed_manifest"],
                  "parent_legacy_checkpoint": envelope["parent_checkpoint"],
                  "parent_metrics_sha256": item["parent_metrics_sha256"],
                  "acquisition_source_sha256": envelope["source_sha256"],
                  "outer_policy": policy, "outer_seed": seed, "outer_step": step}
    result = {"schema": STATE_SCHEMA, "seed": seed, "arm": policy, "policy": policy,
              "step": step, "checkpoint_provenance": provenance, "source_sha256": pins,
              "probe_split_sha256": recipe["split"]["sha256"],
              "prior_recipe_contract": prior_seed, "raw_activations": raw_receipt,
              "behavior": behavior_record, "probes": probes}
    if not finite_tree(result):
        raise ValueError("Nonfinite new-state scalar measurement")
    scalar_receipt = output.write_json(Path("scalars") / f"{stem}.json", result)
    record = {"stage": "raw_direction", "result": result,
              "result_path": Path(scalar_receipt["path"]),
              "result_sha256": scalar_receipt["sha256"],
              "result_bytes": Path(scalar_receipt["path"]).read_bytes(),
              "raw_path": Path(raw_receipt["path"]), "raw_sha256": raw_receipt["sha256"],
              "raw_size_bytes": raw_receipt["size_bytes"]}
    analyzed = analyze_state(record, FIXED_PANEL)
    analyzed.update({"schema": "grokking_raw_direction_analyzed_state_v1",
                     "policy": policy, "checkpoint_provenance": provenance,
                     "prior_recipe_contract": prior_seed})
    analyzed["source"]["raw_direction_checkpoint_sha256"] = item["checkpoint"]["sha256"]
    if not finite_tree(analyzed):
        raise ValueError("Nonfinite new-state analysis")
    state_receipt = output.write_json(Path("states") / f"{stem}.json", analyzed)
    return {"seed": seed, "policy": policy, "step": step, "checkpoint": item["checkpoint"],
            "raw": raw_receipt, "scalar": scalar_receipt, "analyzed_state": state_receipt}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-batch-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError("Accepted CUDA measurement environment unavailable")
    started = time.monotonic()
    records = corpus()
    acquisition_pins = acquisition_source_pins()
    batch = verify_raw_batch(args.raw_batch_dir, records, acquisition_pins)
    old_measurement, old_receipts = verify_old_measurement_sources()
    all_pairs = torch.cartesian_prod(torch.arange(RECIPE["p"]), torch.arange(RECIPE["p"]))
    all_sums = all_pairs.sum(-1) % RECIPE["p"]
    prior, prior_receipts = load_prior_recipe_contracts(all_pairs, all_sums)
    current_prior, current_batch = current_cuda_environments()
    validate_environment_contract(prior["environments"]["calibration"],
                                  prior["environments"]["remaining"], batch["environment"],
                                  current_prior, current_batch)
    pins = measurement_sources()
    output = Output(args.output_dir)
    manifest = {"schema": SCHEMA, "roster": expected_roster(),
                "raw_batch": {"path": batch["path"], "manifest": batch["manifest_receipt"],
                              "completion": batch["completion_receipt"],
                              "source_commits": batch["source_commits"]},
                "acquisition_source_sha256": acquisition_pins, "measurement_source_sha256": pins,
                "source_commit": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
                "accepted_old_measurement_sources": old_measurement,
                "prior_recipe": prior, "recipe": RECIPE, "fixed_frequency_panel": FIXED_PANEL,
                "environment": current_prior, "acquisition_environment": current_batch,
                "bounds": {"cooperative_seconds": COOPERATIVE_SECONDS, "cpu_threads": 1,
                           "memory_bytes": MEMORY_LIMIT, "output_bytes": OUTPUT_LIMIT,
                           "free_reserve_bytes": OUTPUT_RESERVE},
                "interpretation": "new-state-only measurement; no paired causal aggregation"}
    manifest_receipt = output.write_json("manifest.json", manifest)
    accepted = []
    try:
        for item in batch["states"]:
            _check_resources(started, output)
            accepted.append(measure_one(item, prior["seeds"][item["seed"]], all_pairs,
                                        all_sums, output, pins))
            print(json.dumps({"measured": [item["seed"], item["policy"], item["step"]],
                              "output_bytes": output.bytes}), flush=True)
        if [(item["seed"], item["policy"], item["step"]) for item in accepted] != expected_roster():
            raise ValueError("Completed measurement differs from fixed 15-state roster")
        _check_resources(started, output)
        if (measurement_sources() != pins or acquisition_source_pins() != acquisition_pins
                or original_measurement_sources() != old_measurement["measurement_source_sha256"]
                or prior_measurement_sources() != prior["acquisition_source_sha256"]
                or prior_analysis_sources() != prior["analysis_source_sha256"]):
            raise ValueError("Scientific sources changed during new-state measurement")
        _recheck(batch["input_receipts"] + prior_receipts + old_receipts)
        output.write_json("complete.json", {
            "schema": COMPLETE_SCHEMA, "status": "complete", "state_count": len(accepted),
            "accepted_results": accepted, "manifest": manifest_receipt,
            "measurement_source_sha256": pins, "acquisition_source_sha256": acquisition_pins,
            "input_batch_completion": batch["completion_receipt"], "prior_summary": prior["summary"],
            "elapsed_seconds": time.monotonic() - started, "peak_rss_bytes": _peak_rss_bytes(),
            "output_bytes_before_completion": output.bytes,
            "interpretation": "15 new per-state measurements; no aggregate causal conclusion"})
    except BaseException as error:
        failure = {"schema": "grokking_raw_direction_measurement_failure_v1",
                   "status": "failed_preserved", "type": type(error).__name__,
                   "message": str(error), "accepted_results": accepted, "manifest": manifest_receipt,
                   "elapsed_seconds": time.monotonic() - started, "peak_rss_bytes": _peak_rss_bytes(),
                   "output_bytes_before_failure": output.bytes}
        print(json.dumps(failure, allow_nan=False), flush=True)
        try:
            output.write_json("failure.json", failure)
        except Exception as receipt_error:
            print(f"Failure JSON preserved in raw stdout: {receipt_error}", flush=True)
        raise


if __name__ == "__main__":
    main()
