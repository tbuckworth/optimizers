#!/usr/bin/env python3
"""Observe the frozen 150-checkpoint corpus; never train or resume a model.

Calibration and remainder have disjoint fixed rosters. Raw activations and
probe scalars are exclusive-create and hash-bound to original checkpoints.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from experiments.grokking_confirmation import (
    CHECKPOINT_SCHEMA, CHECKPOINT_STEPS, MODEL, atomic_exclusive, canonical_hash,
    file_hash, load_checkpoint, parameter_identity, tensor_set_identity, write_json,
)
from experiments.grokking_model import GrokkingTransformer, get_modular_addition_data
from experiments.grokking_representation import (
    evaluate_fourier_probes, extract_activations, make_probe_split, make_row_permutations,
)

SOURCE_DIR = REPO / "output/2026-09-08-spectral-paper-planning/grokking-confirmation-results"
SUMMARY_SHA256 = "7c408285a6895adda9a423be2c6ddedb99120ea17b73275c7a6ce3f89cb873ab"
ARMS = ("adamw", "legacy", "stable")
SEEDS = tuple(range(100, 105))
RECIPE = {"ridge": .001, "top_k": 5, "n_nulls": 20, "p": 113,
          "split_seed_offset": 20260909, "null_seed_offset": 20261909,
          "batch_size": 1024, "threads": 1,
          "features": ["final_hidden", "pre_attention"],
          "max_output_bytes": 10 * 1024**3, "cooperative_seconds": 1740}


def sources():
    names = ("experiments/measure_grokking_representations.py",
             "experiments/grokking_representation.py",
             "experiments/grokking_model.py", "experiments/grokking_confirmation.py",
             "spectral_filter.py", "tests/test_grokking_representation.py",
             "tests/test_grokking_representation_runner.py",
             "output/2026-09-09-spectral-grokking-mechanism/protocol.md")
    return {name: file_hash(REPO / name) for name in names}


def roster(stage):
    if stage not in ("calibration", "remaining"):
        raise ValueError("Unknown acquisition stage.")
    result = []
    for seed in SEEDS:
        for arm in ARMS:
            for step in CHECKPOINT_STEPS:
                calibration = seed == 100 and step in (0, 6000)
                if calibration == (stage == "calibration"):
                    result.append((seed, arm, step))
    return result


def validate_output_location(path):
    storage = Path("/tmp/spectral-experiment-artifacts").resolve()
    output = path.resolve()
    if output == storage or not output.is_relative_to(storage) or output.exists():
        raise ValueError("Output must be a new descendant of /tmp/spectral-experiment-artifacts.")
    if not output.parent.is_dir():
        raise ValueError("Exclusive parent directory must already exist.")
    if shutil.disk_usage(output.parent).free < RECIPE["max_output_bytes"] + 1024**3:
        raise ValueError("Insufficient free space for the bounded observation bundle.")
    return output


def verify_calibration(path, pins, environment):
    if path is None:
        raise ValueError("Remaining stage requires the completed calibration directory.")
    manifest = json.loads((path / "manifest.json").read_text())
    completion = json.loads((path / "complete.json").read_text())
    if (manifest["stage"] != "calibration" or completion["stage"] != "calibration"
            or completion["status"] != "complete" or manifest["source_sha256"] != pins
            or completion["source_sha256"] != pins or manifest["recipe"] != RECIPE
            or manifest["environment"] != environment):
        raise ValueError("Calibration and remaining acquisition are not comparable.")
    accepted = completion["accepted_results"]
    if (len(accepted) != 6 or
            sorted((r["seed"], r["arm"], r["step"]) for r in accepted) != sorted(roster("calibration"))):
        raise ValueError("Calibration did not complete the exact six-state roster.")
    results = []
    for receipt in accepted:
        result_path = Path(receipt["path"])
        if result_path.resolve().parent != path.resolve() or file_hash(result_path) != receipt["sha256"]:
            raise ValueError("Calibration result receipt mismatch.")
        result = json.loads(result_path.read_text())
        if (result["seed"], result["arm"], result["step"]) != (receipt["seed"], receipt["arm"], receipt["step"]):
            raise ValueError("Calibration result identity mismatch.")
        raw = result["raw_activations"]
        if (Path(raw["path"]).resolve().parent != path.resolve()
                or file_hash(Path(raw["path"])) != raw["sha256"]):
            raise ValueError("Calibration activations changed.")
        results.append(result)
    gate = calibration_gate(results)
    if not gate["admitted"] or completion.get("calibration_gate") != gate:
        raise ValueError("Calibration did not pass the frozen construct-suitability gate.")
    return {"path": str(path.resolve()), "completion_sha256": file_hash(path / "complete.json")}


def calibration_gate(results):
    by_key = {(r["seed"], r["arm"], r["step"]): r for r in results}
    if len(results) != 6 or set(by_key) != set(roster("calibration")):
        raise ValueError("Construct gate requires the exact six-state calibration roster.")
    arms = {}
    for arm in ARMS:
        first, last = by_key[100, arm, 0], by_key[100, arm, 6000]
        final = last["probes"]["final_hidden"]
        score = final["observed"]["selected_eval_mean_r2"]
        gaps = {"final_score": score,
                "above_null_max": score - final["null"]["selected_eval_mean_r2_max"],
                "above_initial": score - first["probes"]["final_hidden"]["observed"]["selected_eval_mean_r2"],
                "above_final_pre_attention": score - last["probes"]["pre_attention"]["observed"]["selected_eval_mean_r2"]}
        gaps["passed"] = gaps["final_score"] >= .5 and all(gaps[k] >= .2 for k in
                            ("above_null_max", "above_initial", "above_final_pre_attention"))
        arms[arm] = gaps
    return {"admitted": all(row["passed"] for row in arms.values()), "arms": arms,
            "interpretation": "construct suitability only, not a circuit or optimizer-success test"}


def corpus():
    summary_path = SOURCE_DIR / "summary.json"
    if file_hash(summary_path) != SUMMARY_SHA256:
        raise ValueError("Confirmation summary differs from audited source.")
    summary = json.loads(summary_path.read_text())
    result = {}
    for entry in summary["evidence"]:
        key = (entry["seed"], entry["arm"])
        path = SOURCE_DIR / entry["local_copy"]
        if file_hash(path) != entry["sha256"] or key in result:
            raise ValueError("Changed or duplicate raw source.")
        record = json.loads(path.read_text())
        if (record["status"] != "complete" or record["completed_steps"] != 6000
                or (record["config"]["seed"], record["config"]["arm"]) != key):
            raise ValueError("Incomplete or mismatched scientific run.")
        result[key] = (record, entry["sha256"])
    if set(result) != {(seed, arm) for seed in SEEDS for arm in ARMS}:
        raise ValueError("Not the complete 15-run corpus.")
    return result


def validate_checkpoint(state, record, seed, arm, step, data):
    if (state["schema"] != CHECKPOINT_SCHEMA or state["step"] != step
            or state["config"] != record["config"]
            or state["config_sha256"] != canonical_hash(record["config"])
            or state["source_identity"] != record["source_identity"]
            or state["source_identity_sha256"] != canonical_hash(record["source_identity"])
            or state["split_identity"] != tensor_set_identity(data)
            or state["split_identity"] != record["split_identity"]
            or state["parameter_identity"] != record["parameter_identity"]
            or (state["config"]["seed"], state["config"]["arm"]) != (seed, arm)):
        raise ValueError("Checkpoint schema/config/step/split/source mismatch.")
    expected_rows = [r for r in record["evaluation_rows"] if r["step"] <= step]
    if (state["evaluation_rows"] != expected_rows
            or [r["step"] for r in expected_rows] != list(range(0, step + 1, 50))):
        raise ValueError("Saved checkpoint evaluation-prefix mismatch.")
    paths = {"model": REPO / "experiments/grokking_model.py",
             "harness": REPO / "experiments/grokking_confirmation.py",
             "filter": REPO / "spectral_filter.py"}
    for key, path in paths.items():
        if file_hash(path) != state["source_identity"]["files"][key]["sha256"]:
            raise ValueError("Original scientific implementation has changed.")


def behavior(logits, targets, indices):
    values = logits[indices].double()
    labels = targets[indices]
    return {"loss": float(F.cross_entropy(values, labels)),
            "accuracy": float((values.argmax(-1) == labels).double().mean()),
            "count": len(indices)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("calibration", "remaining"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--calibration-dir", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA is unavailable; do not silently change device.")
    records = corpus()
    pins = sources()
    output = validate_output_location(args.output_dir)
    started = time.monotonic()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    manifest = {"schema": "grokking_representation_measurement_v1", "stage": args.stage,
                "roster": roster(args.stage), "recipe": RECIPE, "source_sha256": pins,
                "source_commit": commit, "parent_summary_sha256": SUMMARY_SHA256,
                "environment": {"python": platform.python_version(), "torch": str(torch.__version__),
                                "numpy": np.__version__, "device": args.device,
                                "gpu": torch.cuda.get_device_name() if args.device == "cuda" else None}}
    if args.stage == "remaining":
        manifest["calibration"] = verify_calibration(args.calibration_dir, pins, manifest["environment"])
    elif args.calibration_dir is not None:
        raise ValueError("Calibration stage cannot inherit a calibration result.")
    output.mkdir(exist_ok=False)
    write_json(output / "manifest.json", manifest)
    receipts, calibration_results = [], []
    total_bytes = (output / "manifest.json").stat().st_size
    all_pairs = torch.cartesian_prod(torch.arange(113), torch.arange(113))
    all_sums = all_pairs.sum(-1) % 113
    try:
        for seed, arm, step in roster(args.stage):
            if time.monotonic() - started > RECIPE["cooperative_seconds"]:
                raise TimeoutError("Cooperative deadline; preserve partial observations.")
            item_started = time.monotonic()
            record, record_hash = records[seed, arm]
            if str(torch.__version__) != record["environment"]["torch"]:
                raise ValueError("PyTorch differs from the checkpoint acquisition environment.")
            data = get_modular_addition_data(p=113, train_frac=.3, seed=seed)
            train_ids = torch.sort(data[0][:, 0] * 113 + data[0][:, 1]).values
            test_ids = torch.sort(data[2][:, 0] * 113 + data[2][:, 1]).values
            y = all_sums[test_ids]
            split = make_probe_split(y, seed=RECIPE["split_seed_offset"] + seed)
            permutations = make_row_permutations(len(y), RECIPE["null_seed_offset"] + seed, RECIPE["n_nulls"])
            matching = [r for r in record["checkpoints"]
                        if Path(r["path"]).name == f"checkpoint-step-{step:06d}.pt"]
            if len(matching) != 1:
                raise ValueError("Missing or ambiguous checkpoint receipt.")
            checkpoint = matching[0]
            state = load_checkpoint(Path(checkpoint["path"]), checkpoint["sha256"])
            validate_checkpoint(state, record, seed, arm, step, data)
            model = GrokkingTransformer(**MODEL).to(args.device)
            model.load_state_dict(state["model_state"], strict=True)
            if parameter_identity(model) != record["parameter_identity"]:
                raise ValueError("Model parameter layout changed.")
            expected = next(r for r in record["evaluation_rows"] if r["step"] == step)
            del state
            activations = extract_activations(model, all_pairs, batch_size=RECIPE["batch_size"])
            if any(not bool(torch.isfinite(tensor).all()) for tensor in activations.values()):
                raise ValueError("Nonfinite extracted activation.")
            stem = f"seed{seed}-{arm}-step{step:06d}"
            arrays = {name: tensor.numpy() for name, tensor in activations.items()}
            arrays.update({"pairs": all_pairs.numpy(), "sums": all_sums.numpy(),
                           "train_ids": train_ids.numpy(), "test_ids": test_ids.numpy(),
                           "probe_fit_indices": np.asarray(split["fit_indices"], dtype=np.int64),
                           "probe_eval_indices": np.asarray(split["eval_indices"], dtype=np.int64),
                           "null_permutations": np.asarray([r["permutation"] for r in permutations], dtype=np.int64)})
            # NPZ may slightly expand incompressible arrays; leave a full MiB
            # for ZIP/NPY overhead before creating this bounded artifact.
            raw_upper_bound = sum(array.nbytes for array in arrays.values()) + 1024**2
            if total_bytes + raw_upper_bound + 1024**2 > RECIPE["max_output_bytes"]:
                raise RuntimeError("Insufficient remaining result budget for the next state.")
            if shutil.disk_usage(output).free < raw_upper_bound + 1024**3:
                raise RuntimeError("Large-volume free-space reserve exhausted.")
            raw_path = output / (stem + ".npz")
            atomic_exclusive(raw_path, lambda handle: np.savez_compressed(handle, **arrays))
            total_bytes += raw_path.stat().st_size
            behavior_record = {name: behavior(activations["logits"], all_sums, ids)
                               for name, ids in (("train", train_ids), ("test", test_ids))}
            for name in ("train", "test"):
                if (abs(behavior_record[name]["loss"] - expected[name]["loss"]) > 1e-4
                        or abs(behavior_record[name]["accuracy"] - expected[name]["accuracy"]) > 1e-6):
                    raise ValueError("Observed model does not reproduce checkpoint behavior within fixed tolerance.")
            probes = {}
            for feature in RECIPE["features"]:
                probes[feature] = evaluate_fourier_probes(
                    activations[feature][test_ids], y, split["fit_indices"], split["eval_indices"],
                    p=113, ridge=RECIPE["ridge"], top_k=RECIPE["top_k"],
                    null_seed=RECIPE["null_seed_offset"] + seed, n_nulls=RECIPE["n_nulls"],
                    null_permutations=permutations)
            result = {"seed": seed, "arm": arm, "step": step,
                      "checkpoint": checkpoint, "parent_metrics_sha256": record_hash,
                      "source_sha256": pins, "probe_split_sha256": split["sha256"],
                      "raw_activations": {"path": str(raw_path), "sha256": file_hash(raw_path),
                                          "size_bytes": raw_path.stat().st_size},
                      "behavior": behavior_record, "probes": probes,
                      "elapsed_seconds": time.monotonic() - item_started}
            result_path = output / (stem + ".json")
            write_json(result_path, result)
            total_bytes += result_path.stat().st_size
            receipts.append({"seed": seed, "arm": arm, "step": step,
                             "path": str(result_path), "sha256": file_hash(result_path)})
            if args.stage == "calibration":
                calibration_results.append(result)
            print(json.dumps({"completed": [seed, arm, step], "seconds": result["elapsed_seconds"],
                              "raw_bytes": raw_path.stat().st_size}), flush=True)
            del model, activations, arrays, probes, result
            if total_bytes > RECIPE["max_output_bytes"]:
                raise RuntimeError("Result storage bound exceeded; preserve partial observations.")
        if sources() != pins:
            raise ValueError("Measurement source changed while running.")
        complete = {"status": "complete", "stage": args.stage,
                    "source_sha256": pins, "accepted_results": receipts,
                    "elapsed_seconds": time.monotonic() - started, "output_bytes": total_bytes}
        if args.stage == "calibration":
            complete["calibration_gate"] = calibration_gate(calibration_results)
        write_json(output / "complete.json", complete)
    except Exception as error:
        write_json(output / "failure.json", {"status": "failed", "type": type(error).__name__,
                   "message": str(error), "accepted_results": receipts,
                   "elapsed_seconds": time.monotonic() - started})
        raise


if __name__ == "__main__":
    main()
