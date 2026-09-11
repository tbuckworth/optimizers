#!/usr/bin/env python3
"""Two fixed new policies from existing legacy states; never repeat native runs."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.grokking_confirmation import (
    ADAMW, FILTER, FILTER_MUTABLE, MODEL, checkpoint_state, clone_cpu,
    environment, evaluate, file_hash, filter_state, finite_tree, load_checkpoint,
    make_optimizer, parameter_identity, restore_state, save_checkpoint,
    tensor_set_identity, train_step, write_json as _write_json,
)
from experiments.grokking_action_policy import compute_actions, LegacyActionPolicyFilter
from experiments.grokking_model import GrokkingTransformer, get_modular_addition_data
from experiments.measure_grokking_representations import corpus, validate_checkpoint
from spectral_filter import SpectralGradientFilter

SCHEMA = "grokking_action_intervention_v1"
POLICIES = ("native", "orthogonal", "norm_matched")
FORK_STEP, END_STEP = 1500, 2500
CAPTURE_STEPS = (1501, 2000, 2500)
MAX_BYTES = 20 * 1024**3
STATE_KEYS = ("model_state", "optimizer_state", "filter_state", "filter_configuration",
              "torch_cpu_rng_state", "torch_cuda_rng_states", "device_type",
              "cuda_device_count", "config", "config_sha256", "split_identity",
              "parameter_identity")
PROTOCOL = "output/2026-09-09-spectral-grokking-action/protocol.md"


def _reserve(path, estimated_bytes):
    root = path.parent.parent
    used = sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
    if used + estimated_bytes > MAX_BYTES or shutil.disk_usage(path.parent).free < estimated_bytes + 1024**3:
        raise RuntimeError("Prospective output/free-space budget exhausted")


def write_json(path, value):
    estimated = len(json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode()) + 1
    _reserve(path, estimated)
    _write_json(path, value)


def tensor_bytes(value):
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size() + 4096
    if isinstance(value, dict):
        return 4096 + sum(tensor_bytes(k) + tensor_bytes(v) for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return 4096 + sum(tensor_bytes(v) for v in value)
    return len(str(value).encode()) + 256


def bounded_checkpoint(path, value):
    _reserve(path, tensor_bytes(value) + 1024**2)
    return save_checkpoint(path, value)


def source_pins():
    paths = ("experiments/grokking_action_intervention.py",
             "experiments/run_grokking_action_batch.py",
             "experiments/grokking_action_policy.py",
             "tests/test_grokking_action_policy.py",
             "tests/test_grokking_action_intervention.py",
             "experiments/grokking_confirmation.py", "experiments/grokking_model.py",
             "experiments/measure_grokking_representations.py", "spectral_filter.py",
             PROTOCOL)
    return {path: file_hash(ROOT / path) for path in paths}


def tree_hash(value):
    """Type/shape/value-bound hash, independent of torch pickle/storage IDs."""
    digest = hashlib.sha256()

    def visit(item):
        if isinstance(item, torch.Tensor):
            data = item.detach().cpu().contiguous()
            digest.update(b"tensor\0" + str(data.dtype).encode() + b"\0")
            digest.update(json.dumps(list(data.shape)).encode() + b"\0")
            digest.update(data.numpy().tobytes())
        elif isinstance(item, dict):
            digest.update(b"dict\0")
            for key in sorted(item, key=lambda key: (type(key).__name__, str(key))):
                visit(key)
                visit(item[key])
            digest.update(b"enddict\0")
        elif isinstance(item, (tuple, list)):
            digest.update(type(item).__name__.encode() + b"\0")
            for child in item:
                visit(child)
            digest.update(b"endsequence\0")
        else:
            digest.update(type(item).__name__.encode() + b"\0")
            digest.update(json.dumps(item, allow_nan=False).encode() + b"\0")
    visit(value)
    return digest.hexdigest()


def scientific_hash(state):
    return tree_hash({key: state[key] for key in STATE_KEYS})


def restore_filter(tracker, saved, device):
    if set(saved) != set(FILTER_MUTABLE):
        raise ValueError("Filter state schema changed")
    for key, value in saved.items():
        if isinstance(value, torch.Tensor):
            value = value.clone().to("cpu" if key == "S" else device)
        setattr(tracker, key, value)


def snapshot(model, optimizer, tracker, state, device, rows=(), timing=None):
    return checkpoint_state(
        model, optimizer, tracker, step=tracker.step_count, rows=list(rows),
        timing=timing or {}, config=state["config"], sources=state["source_identity"],
        split=state["split_identity"], parameters=state["parameter_identity"], device=device)


def restored(state, policy, device):
    model = GrokkingTransformer(**MODEL).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), **ADAMW)
    tracker = LegacyActionPolicyFilter(model, optimizer, action_policy=policy,
                                      **FILTER, stable_update=False)
    restore_state(state, model, optimizer, tracker, config=state["config"],
                  sources=state["source_identity"], split=state["split_identity"],
                  parameters=parameter_identity(model), device=device)
    actual = scientific_hash(snapshot(model, optimizer, tracker, state, device))
    if actual != scientific_hash(state):
        raise ValueError("Restored scientific state is not bitwise identical to fork")
    return model, optimizer, tracker, actual


def cosine(left, right):
    a, b = left.double().reshape(-1), right.double().reshape(-1)
    product = float(a.norm() * b.norm())
    return float(a.dot(b)) / product if product > 0 else None


@torch.no_grad()
def adam_diagnostic(model, optimizer, before):
    """Actual post-step moments plus mathematical AdamW decomposition in fp64."""
    m, v, directions, decay, adaptive, displacement = [], [], [], [], [], []
    old_by_parameter = {id(p): value for p, value in zip(model.parameters(), before)}
    for group in optimizer.param_groups:
        if group["amsgrad"] or group.get("maximize", False):
            raise ValueError("Unsupported non-original AdamW mode")
        beta1, beta2 = group["betas"]
        for parameter in group["params"]:
            state = optimizer.state[parameter]
            step = int(state["step"])
            first, second = state["exp_avg"].double(), state["exp_avg_sq"].double()
            direction = (first / (1 - beta1**step)) / (
                (second / (1 - beta2**step)).sqrt() + group["eps"])
            old = old_by_parameter[id(parameter)].to(parameter.device).double()
            m.append(first.flatten().cpu())
            v.append(second.flatten().cpu())
            directions.append(direction.flatten().cpu())
            decay.append((-group["lr"] * group["weight_decay"] * old).flatten().cpu())
            adaptive.append((-group["lr"] * direction).flatten().cpu())
            displacement.append((parameter.double() - old).flatten().cpu())
    result = {name: torch.cat(values) for name, values in
              (("m", m), ("v", v), ("adaptive_direction", directions),
               ("decay_movement", decay), ("adaptive_movement", adaptive),
               ("actual_displacement", displacement))}
    residual = result["actual_displacement"] - result["decay_movement"] - result["adaptive_movement"]
    result["summary"] = {
        **{name + "_norm": float(value.norm()) for name, value in result.items()},
        "decomposition_residual_norm": float(residual.norm()),
        "decomposition_residual_max_abs": float(residual.abs().max()),
    }
    if not finite_tree(result):
        raise ValueError("Nonfinite Adam diagnostic")
    return result


def validate_environment(record, device):
    env = environment(device, 1)
    for name in ("python", "torch", "cuda_runtime", "cudnn", "device",
                 "cuda_device_name", "cuda_capability", "torch_cpu_threads"):
        if env.get(name) != record["environment"].get(name):
            raise ValueError(f"Original environment mismatch: {name}")
    env["current_backend_flags"] = {
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
    }
    return env


def check_budget(output, started):
    if time.monotonic() - started > 3 * 3600 - 120:
        raise TimeoutError("Three-hour seed deadline; preserve partial branch")
    size = sum(p.stat().st_size for p in output.parent.rglob("*") if p.is_file())
    if size >= MAX_BYTES or shutil.disk_usage(output).free < 1024**3:
        raise RuntimeError("Output/free-space bound exhausted")


def run(seed, output, admission=None):
    if seed not in range(100, 105):
        raise ValueError("Seed outside fixed roster")
    device = torch.device("cuda")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError("Original CUDA device unavailable")
    output = output.resolve()
    root = Path("/tmp/spectral-experiment-artifacts").resolve()
    if output.exists() or root not in output.parents or output.parent == root:
        raise ValueError("Need a new seed subdirectory of an exclusive large-volume parent")
    if not output.parent.is_dir() or shutil.disk_usage(output.parent).free < MAX_BYTES + 1024**3:
        raise ValueError("Missing parent or insufficient free disk")
    record, record_hash = corpus()[seed, "legacy"]
    pins = source_pins()
    env = validate_environment(record, device)
    if seed != 100:
        if admission is None:
            raise ValueError("Remaining fixed seeds require completed seed100 admission")
        prior = json.loads((admission / "complete.json").read_text())
        if (prior["status"] != "complete" or prior["seed"] != 100
                or prior["source_sha256"] != pins or prior["environment"] != env
                or prior["accepted_roster"] != [[p, s] for p in POLICIES
                    for s in ((1501,) if p == "native" else CAPTURE_STEPS)]):
            raise ValueError("Seed100 resource/source/roster admission failed")
        for receipt in prior["checkpoints"]:
            if file_hash(Path(receipt["path"])) != receipt["sha256"]:
                raise ValueError("Admitted checkpoint receipt changed")
    elif admission is not None:
        raise ValueError("Seed100 must not inherit another acquisition")
    matching = [r for r in record["checkpoints"]
                if Path(r["path"]).name == "checkpoint-step-001500.pt"]
    if len(matching) != 1:
        raise ValueError("Missing/ambiguous original checkpoint")
    parent = matching[0]
    state = load_checkpoint(Path(parent["path"]), parent["sha256"])
    cpu_data = get_modular_addition_data(p=113, train_frac=.3, seed=seed)
    validate_checkpoint(state, record, seed, "legacy", FORK_STEP, cpu_data)
    data = tuple(value.to(device) for value in cpu_data)
    output.mkdir(mode=0o700)
    started = time.monotonic()
    accepted, receipts = [], []
    current_policy, current_step = None, FORK_STEP
    metadata = {"schema": SCHEMA, "seed": seed, "parent_checkpoint": parent,
                "parent_metrics_sha256": record_hash, "source_sha256": pins,
                "source_commit": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "environment": env, "fork_scientific_state_sha256": scientific_hash(state),
                "capture_steps": list(CAPTURE_STEPS), "policies": list(POLICIES),
                "native_policy_is_one_step_diagnostic_only": True}
    write_json(output / "manifest.json", metadata)
    try:
        # One shared gradient and post-estimator basis for all first actions.
        model, optimizer, tracker, shared_hash = restored(state, "native", device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(data[0]), data[1])
        loss.backward()
        g = tracker._get_flat_grad().detach().clone()
        if not bool(torch.isfinite(g).all()) or not bool(torch.isfinite(loss)):
            raise ValueError("Nonfinite shared first gradient")
        tracker.step_count += 1
        tracker._update_svd(g)
        shared_filter = filter_state(tracker)
        actions = compute_actions(tracker.V, g, retain_basis=True)
        if not torch.equal(actions["actions"]["native"], SpectralGradientFilter._project_gradient(tracker, g)):
            raise ValueError("Native action differs from frozen original implementation")
        rng = {"cpu": torch.get_rng_state().clone(),
               "cuda": [v.clone() for v in torch.cuda.get_rng_state_all()]}
        shared_receipt = bounded_checkpoint(output / "shared-first-action.pt", {
            "schema": SCHEMA, "seed": seed, "parent_checkpoint": parent,
            "raw_gradient": g.cpu(), "post_estimator_sha256": tree_hash(shared_filter),
            "actions": clone_cpu(actions), "rng_after_shared_forward": clone_cpu(rng),
            "restored_scientific_state_sha256": shared_hash,
            "training_loss_before_update": float(loss.detach()),
        })
        del model, optimizer, tracker, g, loss
        first_step_diagnostics = {}
        for policy in POLICIES:
            current_policy, current_step = policy, FORK_STEP
            check_budget(output, started)
            model, optimizer, tracker, restored_hash = restored(state, policy, device)
            restore_filter(tracker, shared_filter, device)
            if tree_hash(filter_state(tracker)) != tree_hash(shared_filter):
                raise ValueError("Shared post-estimator state changed at fork")
            torch.set_rng_state(rng["cpu"])
            torch.cuda.set_rng_state_all(rng["cuda"])
            before = [p.detach().cpu().clone() for p in model.parameters()]
            optimizer.zero_grad(set_to_none=True)
            tracker._set_flat_grad(actions["actions"][policy].clone())
            optimizer.step()
            current_step = 1501
            first_step_diagnostics[policy] = adam_diagnostic(model, optimizer, before)
            first_step_diagnostics[policy]["restored_scientific_state_sha256"] = restored_hash
            bounded_checkpoint(output / f"{policy}-first-step-adam.pt", first_step_diagnostics[policy])
            del before
            rows, history = [], []
            timing = {"training_seconds": 0.0, "evaluation_seconds": 0.0}

            def capture(step):
                value = snapshot(model, optimizer, tracker, state, device, rows, timing)
                envelope = {"schema": "grokking_action_checkpoint_v1", "policy": policy,
                            "seed": seed, "step": step, "parent_checkpoint": parent,
                            "source_sha256": pins, "shared_first_action": shared_receipt,
                            "native_compatible_state": value,
                            "note": "Inner config/source fields describe inherited native contract; outer policy and source_sha256 identify new execution."}
                receipt = bounded_checkpoint(output / f"{policy}-step-{step:06d}.pt", envelope)
                receipt.update({"policy": policy, "seed": seed, "step": step})
                receipts.append(receipt)
                accepted.append([policy, step])
                write_json(output / f"{policy}-through-{step:06d}.json", {
                    "schema": SCHEMA, "policy": policy, "seed": seed, "step": step,
                    "source_sha256": pins, "checkpoint": receipt,
                    "evaluation_rows": rows, "action_history": history, "timing": timing})

            # Capture before any new readout. Later inference uses loaded copies;
            # the live continuation only retains the original 50-step eval grid.
            capture(1501)
            if policy != "native":
                for step in range(1502, END_STEP + 1):
                    loss, diagnostics, seconds = train_step(
                        model, optimizer, tracker, data[0], data[1], device)
                    current_step = step
                    timing["training_seconds"] += seconds
                    history.append({"step": step, **tracker.last_action_metrics})
                    if step % 50 == 0:
                        check_budget(output, started)
                        evaluation_rng = (torch.get_rng_state().clone(),
                                          [x.clone() for x in torch.cuda.get_rng_state_all()])
                        row, seconds = evaluate(model, data, step, device)
                        if (not torch.equal(evaluation_rng[0], torch.get_rng_state())
                                or any(not torch.equal(a, b) for a, b in zip(
                                    evaluation_rng[1], torch.cuda.get_rng_state_all()))):
                            raise ValueError("Original-grid evaluation changed training RNG")
                        rows.append(row)
                        timing["evaluation_seconds"] += seconds
                        print(json.dumps({"seed": seed, "policy": policy, "step": step,
                                          "elapsed_seconds": time.monotonic() - started}), flush=True)
                    if step in CAPTURE_STEPS:
                        capture(step)
            del model, optimizer, tracker
        for policy, diagnostic in first_step_diagnostics.items():
            diagnostic["summary"]["pairwise"] = {
                other: {name + "_cosine": cosine(diagnostic[name], target[name])
                        for name in ("adaptive_direction", "actual_displacement")}
                for other, target in first_step_diagnostics.items()}
        first_receipt = bounded_checkpoint(output / "first-step-adam-diagnostics.pt", first_step_diagnostics)
        write_json(output / "first-step-summary.json", {
            "seed": seed, "source_sha256": pins,
            "shared_first_action": shared_receipt, "adam_diagnostics": first_receipt,
            "policies": {p: x["summary"] for p, x in first_step_diagnostics.items()}})
        if source_pins() != pins:
            raise ValueError("Frozen scientific source changed during acquisition")
        check_budget(output, started)
        write_json(output / "complete.json", {
            **metadata, "status": "complete", "accepted_roster": accepted,
            "checkpoints": receipts, "shared_first_action": shared_receipt,
            "first_step_adam_diagnostics": first_receipt,
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
            "elapsed_seconds": time.monotonic() - started})
    except BaseException as error:
        _write_json(output / "failure.json", {
            **metadata, "status": "failed_preserved", "policy": current_policy,
            "last_completed_step": current_step, "accepted_roster": accepted,
            "checkpoints": receipts, "error_type": type(error).__name__, "error": str(error)})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=range(100, 105), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--admission-dir", type=Path)
    args = parser.parse_args()
    run(args.seed, args.output_dir, args.admission_dir)
