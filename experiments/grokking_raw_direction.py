#!/usr/bin/env python3
"""Acquire only the fixed raw-direction policy from five archived legacy forks.

The previous action runner is imported for unchanged state/Adam helpers, never
invoked. No old branch, training step or measurement is repeated by this module.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.grokking_action_intervention import (
    adam_diagnostic, scientific_hash, snapshot, tensor_bytes,
    validate_environment, source_pins as legacy_source_pins,
)
from experiments.grokking_confirmation import (
    ADAMW, FILTER, MODEL, clone_cpu, evaluate, file_hash, finite_tree,
    load_checkpoint, parameter_identity, restore_state, save_checkpoint,
    train_step, write_json as atomic_write_json,
)
from experiments.grokking_raw_direction_policy import RawDirectionPolicyFilter
from experiments.grokking_model import GrokkingTransformer, get_modular_addition_data
from experiments.measure_grokking_representations import corpus, validate_checkpoint

SCHEMA = "grokking_raw_direction_acquisition_v1"
CHECKPOINT_SCHEMA = "grokking_raw_direction_checkpoint_v1"
POLICY = "raw_norm_matched"
SEEDS = tuple(range(100, 105))
FORK_STEP, END_STEP = 1500, 2500
CAPTURE_STEPS = (1501, 2000, 2500)
MAX_BYTES, FREE_RESERVE = 20 * 1024**3, 1024**3
SEED_SECONDS, BATCH_SECONDS = 3 * 3600, 12 * 3600
STORAGE = Path("/tmp/spectral-experiment-artifacts")
PROTOCOL = "output/2026-09-09-spectral-raw-direction/protocol.md"
ARCHIVED_BATCH = STORAGE / "spectral-grokking-action-20260909.ghvEvD"
ARCHIVED_BATCH_SHA256 = "22bb866319fa2fa85771bd156eb406523a66abee4d0bc789cc6104d40ec86e45"


def source_pins():
    paths = (
        "experiments/grokking_raw_direction.py",
        "experiments/run_grokking_raw_direction_batch.py",
        "experiments/grokking_raw_direction_policy.py",
        "tests/test_grokking_raw_direction.py",
        "tests/test_grokking_raw_direction_policy.py",
        "tests/test_grokking_raw_direction_guard.py",
        "output/2026-09-09-spectral-raw-direction/guarded_launch.py",
        "experiments/grokking_action_intervention.py",
        "experiments/grokking_action_policy.py",
        "experiments/grokking_confirmation.py",
        "experiments/grokking_model.py",
        "experiments/measure_grokking_representations.py",
        "experiments/grokking_representation.py",
        "spectral_filter.py", PROTOCOL,
    )
    return {**legacy_source_pins(),
            **{path: file_hash(ROOT / path) for path in paths}}


def expected_roster():
    return [[POLICY, step] for step in CAPTURE_STEPS]


class BoundedWriter:
    """One batch-wide prospective budget for every JSON and tensor write."""

    def __init__(self, root):
        self.root = root.resolve()

    def size(self):
        paths = list(self.root.rglob("*"))
        if any(path.is_symlink() for path in paths):
            raise ValueError("Output tree contains a symlink")
        return sum(path.stat().st_size for path in paths if path.is_file())

    def reserve(self, path, estimate):
        if (estimate < 0 or path.is_symlink() or path.exists()
                or not path.parent.is_dir()
                or not path.resolve().is_relative_to(self.root)):
            raise ValueError("Need a new contained output file and nonnegative estimate")
        if (self.size() + estimate > MAX_BYTES
                or shutil.disk_usage(self.root).free < estimate + FREE_RESERVE):
            raise RuntimeError("Prospective output/free-space budget exhausted")

    def json(self, path, value):
        raw_size = len(json.dumps(value, indent=2, sort_keys=True,
                                  allow_nan=False).encode()) + 1
        self.reserve(path, raw_size)
        atomic_write_json(path, value)
        return self.receipt(path)

    def checkpoint(self, path, value):
        if not finite_tree(value):
            raise ValueError("Nonfinite tensor artifact")
        self.reserve(path, tensor_bytes(value) + 1024**2)
        result = save_checkpoint(path, value)
        if self.size() > MAX_BYTES or shutil.disk_usage(self.root).free < FREE_RESERVE:
            raise RuntimeError("Actual tensor serialization exceeded the output budget")
        return result

    @staticmethod
    def receipt(path):
        return {"path": str(path.resolve()), "sha256": file_hash(path),
                "size_bytes": path.stat().st_size}


def check_budget(writer, started):
    if time.monotonic() - started > SEED_SECONDS - 120:
        raise TimeoutError("Three-hour seed deadline; preserve partial acquisition")
    if writer.size() >= MAX_BYTES or shutil.disk_usage(writer.root).free < FREE_RESERVE:
        raise RuntimeError("Output/free-space budget exhausted")


def validate_output(output):
    output = output.resolve()
    if (output.exists() or output.parent.parent != STORAGE.resolve()
            or not output.parent.is_dir()):
        raise ValueError("Need a new seed child of an exclusive large-volume batch")
    if shutil.disk_usage(output.parent).free < MAX_BYTES + FREE_RESERVE:
        raise ValueError("Insufficient free disk for the bounded acquisition")
    return output


def verify_receipt(receipt, directory):
    path = Path(receipt["path"])
    if (path.is_symlink() or path.resolve().parent != directory.resolve()
            or not path.is_file() or path.stat().st_size != receipt["size_bytes"]
            or file_hash(path) != receipt["sha256"]):
        raise ValueError("Artifact receipt is missing, changed or outside its seed directory")


def archived_parent(seed, parent, metrics_sha256):
    """Read only fixed receipt JSON; never load any archived branch tensor."""
    batch_path = ARCHIVED_BATCH / "batch-complete.json"
    if file_hash(batch_path) != ARCHIVED_BATCH_SHA256:
        raise ValueError("Accepted reference batch changed")
    batch = json.loads(batch_path.read_text())
    if (batch["status"] != "complete" or batch["seeds"] != list(SEEDS)
            or [item["seed"] for item in batch["accepted"]] != list(SEEDS)):
        raise ValueError("Accepted reference batch roster mismatch")
    entry = next(item for item in batch["accepted"] if item["seed"] == seed)
    path = Path(entry["path"])
    if (path.resolve() != (ARCHIVED_BATCH / f"seed{seed}/complete.json").resolve()
            or file_hash(path) != entry["sha256"]):
        raise ValueError("Accepted reference seed receipt changed")
    reference = json.loads(path.read_text())
    if (reference["status"] != "complete" or reference["seed"] != seed
            or reference["parent_checkpoint"] != parent
            or reference["parent_metrics_sha256"] != metrics_sha256
            or reference["source_sha256"] != legacy_source_pins()):
        raise ValueError("Raw branch does not use the accepted action batch's exact parent")
    return {"batch_completion_path": str(batch_path),
            "batch_completion_sha256": ARCHIVED_BATCH_SHA256,
            "seed_completion": entry,
            "fork_scientific_state_sha256": reference["fork_scientific_state_sha256"]}


def verify_completion(directory, seed, pins, env):
    completion_path = directory / "complete.json"
    record = json.loads(completion_path.read_text())
    if (record["schema"] != SCHEMA or record["status"] != "complete"
            or record["seed"] != seed or record["policy"] != POLICY
            or record["source_sha256"] != pins or record["environment"] != env
            or record["accepted_roster"] != expected_roster()
            or record["completed_updates"] != END_STEP - FORK_STEP
            or record["history_steps"] != list(range(FORK_STEP + 1, END_STEP + 1))
            or record["elapsed_seconds"] > SEED_SECONDS):
        raise ValueError("Completed seed failed source/resource/fixed-roster admission")
    expected_names = {"manifest.json", "first-step-tensors.pt", "first-step-summary.json"}
    expected_names |= {f"{POLICY}-step-{step:06d}.pt" for step in CAPTURE_STEPS}
    expected_names |= {f"{POLICY}-through-{step:06d}.json" for step in CAPTURE_STEPS}
    artifacts = record["artifact_receipts"]
    if (len(artifacts) != len(expected_names)
            or {Path(item["path"]).name for item in artifacts} != expected_names
            or {path.name for path in directory.iterdir()} != expected_names | {"complete.json"}):
        raise ValueError("Completed artifact roster is incomplete or contains extras")
    for item in artifacts:
        verify_receipt(item, directory)
    by_name = {Path(item["path"]).name: item for item in artifacts}
    for step in CAPTURE_STEPS:
        history_receipt = by_name[f"{POLICY}-through-{step:06d}.json"]
        history = json.loads(Path(history_receipt["path"]).read_text())
        checkpoint_receipt = by_name[f"{POLICY}-step-{step:06d}.pt"]
        if (history["schema"] != SCHEMA or history["seed"] != seed
                or history["policy"] != POLICY or history["step"] != step
                or history["source_sha256"] != pins
                or history["checkpoint"] != checkpoint_receipt
                or [row["step"] for row in history["action_history"]]
                != list(range(FORK_STEP + 1, step + 1))
                or not finite_tree(history)):
            raise ValueError("Bound history/checkpoint identity or update roster mismatch")
    if record["checkpoints"] != [by_name[f"{POLICY}-step-{step:06d}.pt"]
                                  for step in CAPTURE_STEPS]:
        raise ValueError("Checkpoint list differs from bound artifact roster")
    return record


def restored(state, device):
    model = GrokkingTransformer(**MODEL).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), **ADAMW)
    tracker = RawDirectionPolicyFilter(model, optimizer, retain_action_basis=True,
                                       **FILTER, stable_update=False)
    restore_state(state, model, optimizer, tracker, config=state["config"],
                  sources=state["source_identity"], split=state["split_identity"],
                  parameters=parameter_identity(model), device=device)
    actual = scientific_hash(snapshot(model, optimizer, tracker, state, device))
    if actual != scientific_hash(state):
        raise ValueError("Restored scientific state is not bitwise identical to fork")
    return model, optimizer, tracker, actual


def step_with_diagnostic(model, optimizer, tracker, inputs, targets, device, step):
    """Exactly one original training step, followed only by an Adam observation."""
    before = [parameter.detach().cpu().clone() for parameter in model.parameters()]
    loss, _, training_seconds = train_step(model, optimizer, tracker, inputs, targets, device)
    if tracker.step_count != step or tracker.last_action_metrics is None:
        raise ValueError("Action and training counters disagree")
    diagnostic_started = time.perf_counter()
    diagnostic = adam_diagnostic(model, optimizer, before)
    row = {**tracker.last_action_metrics, "step": step,
           "training_loss_before_update": loss, "adam": diagnostic["summary"]}
    if not finite_tree(row):
        raise ValueError("Nonfinite step diagnostic")
    return row, diagnostic, training_seconds, time.perf_counter() - diagnostic_started


def run(seed, output, admission=None):
    if seed not in SEEDS:
        raise ValueError("Seed outside fixed roster")
    if (seed == 100) != (admission is None):
        raise ValueError("Only seed100 can run without completed seed100 admission")
    output = validate_output(output)
    if output.name != f"seed{seed}":
        raise ValueError("Seed output name does not match fixed roster")
    if admission is not None and admission.resolve() != output.parent / "seed100":
        raise ValueError("Admission must be seed100 from this same batch")
    device = torch.device("cuda")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError("Original CUDA device unavailable")
    record, record_hash = corpus()[seed, "legacy"]
    pins = source_pins()
    env = validate_environment(record, device)
    admission_receipt = None
    if admission is not None:
        verify_completion(admission, 100, pins, env)
        admission_receipt = BoundedWriter.receipt(admission / "complete.json")
    matching = [item for item in record["checkpoints"]
                if Path(item["path"]).name == "checkpoint-step-001500.pt"]
    if len(matching) != 1:
        raise ValueError("Missing or ambiguous original checkpoint")
    parent = matching[0]
    reference = archived_parent(seed, parent, record_hash)
    state = load_checkpoint(Path(parent["path"]), parent["sha256"])
    cpu_data = get_modular_addition_data(p=113, train_frac=.3, seed=seed)
    validate_checkpoint(state, record, seed, "legacy", FORK_STEP, cpu_data)
    if scientific_hash(state) != reference["fork_scientific_state_sha256"]:
        raise ValueError("Scientific parent differs from the accepted action fork")
    data = tuple(value.to(device) for value in cpu_data)
    output.mkdir(mode=0o700)
    writer = BoundedWriter(output.parent)
    started = time.monotonic()
    current_step, accepted, checkpoints, artifacts = FORK_STEP, [], [], []
    metadata = {"schema": SCHEMA, "seed": seed, "policy": POLICY,
                "parent_checkpoint": parent, "parent_metrics_sha256": record_hash,
                "source_sha256": pins, "source_commit": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "environment": env, "fork_scientific_state_sha256": scientific_hash(state),
                "archived_reference": reference, "seed100_admission": admission_receipt,
                "capture_steps": list(CAPTURE_STEPS), "no_reference_policy_execution": True}
    try:
        artifacts.append(writer.json(output / "manifest.json", metadata))
        model, optimizer, tracker, restored_hash = restored(state, device)
        rows, history = [], []
        timing = {"training_seconds": 0.0, "evaluation_seconds": 0.0,
                  "adam_diagnostic_seconds": 0.0}
        full_before = snapshot(model, optimizer, tracker, state, device)
        first_receipt = None
        for step in range(FORK_STEP + 1, END_STEP + 1):
            check_budget(writer, started)
            row, diagnostic, seconds, diagnostic_seconds = step_with_diagnostic(
                model, optimizer, tracker, data[0], data[1], device, step)
            current_step = step
            timing["training_seconds"] += seconds
            timing["adam_diagnostic_seconds"] += diagnostic_seconds
            history.append(row)
            if step == FORK_STEP + 1:
                action_result = tracker.last_action_result
                if (action_result is None or "Q" not in action_result
                        or set(action_result["actions"]) !=
                        {"native", "orthogonal", "norm_matched", POLICY}
                        or tracker.last_raw_gradient is None):
                    raise ValueError("First-step tensor diagnostic is incomplete")
                first_receipt = writer.checkpoint(output / "first-step-tensors.pt", {
                    "schema": SCHEMA, "seed": seed, "policy": POLICY, "step": step,
                    "parent_checkpoint": parent, "source_sha256": pins,
                    "restored_scientific_state_sha256": restored_hash,
                    "full_before": full_before,
                    "full_after": snapshot(model, optimizer, tracker, state, device, rows, timing),
                    "raw_gradient": clone_cpu(tracker.last_raw_gradient),
                    "actions": clone_cpu(action_result), "adam": diagnostic,
                    "training_loss_before_update": row["training_loss_before_update"],
                    "note": "Only raw_norm_matched was delivered. Other actions are local counterfactual tensors, not extra optimizer steps."})
                artifacts.append(first_receipt)
                artifacts.append(writer.json(output / "first-step-summary.json", {
                    "schema": SCHEMA, "seed": seed, "policy": POLICY,
                    "source_sha256": pins, "first_step_tensors": first_receipt,
                    "step_diagnostic": row}))
                tracker.retain_action_basis = False
                tracker.last_action_result = None
                tracker.last_raw_gradient = None
                del action_result, full_before
            del diagnostic
            if step % 50 == 0:
                evaluation_rng = (torch.get_rng_state().clone(),
                                  [value.clone() for value in torch.cuda.get_rng_state_all()])
                evaluation, seconds = evaluate(model, data, step, device)
                if (not torch.equal(evaluation_rng[0], torch.get_rng_state())
                        or any(not torch.equal(a, b) for a, b in zip(
                            evaluation_rng[1], torch.cuda.get_rng_state_all()))):
                    raise ValueError("Original-grid evaluation changed training RNG")
                rows.append(evaluation)
                timing["evaluation_seconds"] += seconds
                print(json.dumps({"seed": seed, "policy": POLICY, "step": step,
                                  "elapsed_seconds": time.monotonic() - started}), flush=True)
            if step in CAPTURE_STEPS:
                value = snapshot(model, optimizer, tracker, state, device, rows, timing)
                envelope = {"schema": CHECKPOINT_SCHEMA, "policy": POLICY,
                            "seed": seed, "step": step, "parent_checkpoint": parent,
                            "source_sha256": pins, "first_step_tensors": first_receipt,
                            "native_compatible_state": value,
                            "note": "Inner config/source fields describe the inherited native state contract; outer policy/source fields identify the new execution."}
                checkpoint = writer.checkpoint(output / f"{POLICY}-step-{step:06d}.pt", envelope)
                checkpoint.update({"policy": POLICY, "seed": seed, "step": step})
                checkpoints.append(checkpoint)
                artifacts.append(checkpoint)
                accepted.append([POLICY, step])
                artifacts.append(writer.json(output / f"{POLICY}-through-{step:06d}.json", {
                    "schema": SCHEMA, "policy": POLICY, "seed": seed, "step": step,
                    "source_sha256": pins, "checkpoint": checkpoint,
                    "evaluation_rows": rows, "action_history": history, "timing": timing}))
                del value, envelope
        if source_pins() != pins:
            raise ValueError("Frozen scientific source changed during acquisition")
        check_budget(writer, started)
        writer.json(output / "complete.json", {
            **metadata, "status": "complete", "accepted_roster": accepted,
            "checkpoints": checkpoints, "artifact_receipts": artifacts,
            "completed_updates": END_STEP - FORK_STEP,
            "history_steps": [row["step"] for row in history],
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
            "batch_output_bytes_before_completion": writer.size(),
            "elapsed_seconds": time.monotonic() - started})
    except BaseException as error:
        failure = {**metadata, "status": "failed_preserved", "last_completed_step": current_step,
                   "accepted_roster": accepted, "checkpoints": checkpoints,
                   "artifact_receipts": artifacts, "error_type": type(error).__name__,
                   "error": str(error)}
        print(json.dumps(failure, allow_nan=False), flush=True)
        try:
            writer.json(output / "failure.json", failure)
        except Exception as receipt_error:
            print(f"Failure receipt unavailable; raw stdout preserves it: {receipt_error}", flush=True)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--admission-dir", type=Path)
    args = parser.parse_args()
    run(args.seed, args.output_dir, args.admission_dir)
