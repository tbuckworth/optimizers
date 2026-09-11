#!/usr/bin/env python3
"""Fixed current-stable modular-addition confirmation; one arm/seed per run."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import stat
import subprocess
import sys
import time
from typing import Any

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _directory in (str(HERE), str(ROOT)):
    if _directory not in sys.path:
        sys.path.insert(0, _directory)

from grokking_model import GrokkingTransformer, get_modular_addition_data  # noqa: E402
from spectral_filter import SpectralGradientFilter  # noqa: E402

SCHEMA = "grokking_confirmation_v1"
CHECKPOINT_SCHEMA = "grokking_confirmation_checkpoint_v1"
ARMS = ("adamw", "legacy", "stable")
SEEDS = tuple(range(100, 105))
CHECKPOINT_STEPS = (0, 100, 500, 1000, 1500, 2000, 2500, 3000, 4000, 6000)
MODEL = {"p": 113, "d_model": 128, "n_heads": 4, "d_mlp": 512}
ADAMW = {"lr": 1e-3, "weight_decay": 1.0, "betas": (0.9, 0.98),
         "eps": 1e-8, "amsgrad": False}
FILTER = {"rank": 200, "decay": 0.99, "warmup": 100,
          "filter_strength": 1.0, "weighting": "hard", "normalize": "none",
          "adaptive": "none", "energy_threshold": None, "alpha": 1.0,
          "soft_residual": True, "relative_eig_tol": 1e-8,
          "absolute_eig_floor": 0.0, "stabilize_every": 100}
FILTER_MUTABLE = ("V", "S", "proj_k", "step_count", "grad_mean",
                  "stabilization_count", "max_orthogonality_error")
FILTER_STATIC = ("rank", "decay", "warmup", "filter_strength", "energy_threshold",
                 "adaptive", "normalize", "weighting", "alpha", "soft_residual",
                 "stable_update", "relative_eig_tol", "absolute_eig_floor",
                 "stabilize_every", "n_params")


class ConfirmationError(RuntimeError):
    pass


def canonical_hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"),
                     allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_identity() -> dict[str, Any]:
    paths = {"harness": Path(__file__).resolve(),
             "model": HERE / "grokking_model.py", "filter": ROOT / "spectral_filter.py"}
    try:
        commit = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], check=True,
            capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise ConfirmationError("cannot resolve repository commit") from error
    return {"repository_commit": commit,
            "files": {name: {"path": str(path), "sha256": file_hash(path)}
                      for name, path in paths.items()}}


def resolved_config(arm: str, seed: int, steps=6000, threads=1) -> dict[str, Any]:
    if arm not in ARMS or seed not in SEEDS:
        raise ValueError("arm or prospective seed is outside the fixed roster")
    if steps != 6000 or threads != 1:
        raise ValueError("confirmation requires 6000 steps and one CPU thread")
    return {"arm": arm, "seed": seed, "steps": steps, "threads": threads,
            "model": MODEL, "data": {"p": 113, "train_frac": 0.3,
                                      "full_batch": True},
            "optimizer": {**ADAMW, "betas": list(ADAMW["betas"])},
            "filter": None if arm == "adamw" else
                      {**FILTER, "stable_update": arm == "stable"},
            "evaluation": {"initial_step": 0, "every_steps": 50,
                           "phase": "post_update", "threshold": 0.9},
            "checkpoint_steps": list(CHECKPOINT_STEPS)}


def tensor_set_identity(values) -> dict[str, Any]:
    names = ("train_X", "train_y", "test_X", "test_y")
    digest = hashlib.sha256()
    layout = []
    for name, value in zip(names, values):
        tensor = value.detach().cpu().contiguous()
        shape, dtype = list(tensor.shape), str(tensor.dtype)
        digest.update(name.encode() + b"\0" + dtype.encode() + b"\0")
        digest.update(json.dumps(shape).encode() + b"\0" + tensor.numpy().tobytes())
        layout.append({"name": name, "shape": shape, "dtype": dtype})
    return {"sha256": digest.hexdigest(), "layout": layout,
            "train_count": len(values[0]), "test_count": len(values[2])}


def parameter_identity(model) -> list[dict[str, Any]]:
    return [{"name": name, "shape": list(value.shape), "dtype": str(value.dtype)}
            for name, value in model.named_parameters()]


def make_optimizer(model, arm):
    optimizer = torch.optim.AdamW(model.parameters(), **ADAMW)
    tracker = None if arm == "adamw" else SpectralGradientFilter(
        model, optimizer, **FILTER, stable_update=arm == "stable")
    return optimizer, tracker


def clone_cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: clone_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clone_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(clone_cpu(item) for item in value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"unsupported checkpoint type: {type(value).__name__}")


def filter_state(tracker):
    return None if tracker is None else {
        name: clone_cpu(getattr(tracker, name)) for name in FILTER_MUTABLE}


def filter_configuration(tracker):
    return None if tracker is None else {
        name: getattr(tracker, name) for name in FILTER_STATIC}


def finite_tree(value) -> bool:
    if isinstance(value, torch.Tensor):
        return (not value.is_floating_point()) or bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(finite_tree(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_tree(item) for item in value)
    return not isinstance(value, float) or math.isfinite(value)


def checkpoint_state(model, optimizer, tracker, *, step, rows, timing, config,
                     sources, split, parameters, device):
    if tracker is not None and tracker.step_count != step:
        raise ConfirmationError("filter and training step counters differ")
    state = {"schema": CHECKPOINT_SCHEMA, "step": step,
             "model_state": clone_cpu(dict(model.state_dict())),
             "optimizer_state": clone_cpu(optimizer.state_dict()),
             "filter_state": filter_state(tracker),
             "filter_configuration": filter_configuration(tracker),
             "torch_cpu_rng_state": torch.get_rng_state().cpu().clone(),
             "torch_cuda_rng_states": [item.cpu().clone()
                                       for item in torch.cuda.get_rng_state_all()]
             if torch.device(device).type == "cuda" else [],
             "device_type": torch.device(device).type,
             "cuda_device_count": torch.cuda.device_count()
             if torch.device(device).type == "cuda" else 0,
             "config": config, "config_sha256": canonical_hash(config),
             "source_identity": sources,
             "source_identity_sha256": canonical_hash(sources),
             "split_identity": split, "parameter_identity": parameters,
             "evaluation_rows": clone_cpu(rows), "timing": dict(timing)}
    if not finite_tree(state):
        raise ConfirmationError("checkpoint contains non-finite state")
    return state


def restore_state(state, model, optimizer, tracker, *, config, sources, split,
                  parameters, device):
    required = {"schema", "step", "model_state", "optimizer_state", "filter_state",
                "filter_configuration",
                "torch_cpu_rng_state", "torch_cuda_rng_states", "device_type",
                "cuda_device_count", "config", "config_sha256", "source_identity",
                "source_identity_sha256", "split_identity", "parameter_identity",
                "evaluation_rows", "timing"}
    if not isinstance(state, dict) or set(state) != required or not finite_tree(state):
        raise ConfirmationError("invalid checkpoint root")
    expected = ((state["schema"], CHECKPOINT_SCHEMA),
                (state["config"], config),
                (state["config_sha256"], canonical_hash(config)),
                (state["source_identity"], sources),
                (state["source_identity_sha256"], canonical_hash(sources)),
                (state["split_identity"], split),
                (state["parameter_identity"], parameters),
                (state["device_type"], torch.device(device).type))
    if any(actual != wanted for actual, wanted in expected):
        raise ConfirmationError("checkpoint identity mismatch")
    model.load_state_dict(state["model_state"], strict=True)
    optimizer.load_state_dict(clone_cpu(state["optimizer_state"]))
    saved_filter = state["filter_state"]
    if (tracker is None) != (saved_filter is None):
        raise ConfirmationError("checkpoint filter-arm mismatch")
    if tracker is not None:
        if state["filter_configuration"] != filter_configuration(tracker):
            raise ConfirmationError("checkpoint filter configuration mismatch")
        if set(saved_filter) != set(FILTER_MUTABLE):
            raise ConfirmationError("checkpoint filter-state mismatch")
        for name, value in saved_filter.items():
            if isinstance(value, torch.Tensor):
                value = value.detach().clone()
                value = value.cpu() if name == "S" else value.to(device)
            setattr(tracker, name, value)
        if tracker.step_count != state["step"]:
            raise ConfirmationError("checkpoint filter step counter mismatch")
    torch.set_rng_state(state["torch_cpu_rng_state"].clone())
    if torch.device(device).type == "cuda":
        if (state["cuda_device_count"] != torch.cuda.device_count()
                or len(state["torch_cuda_rng_states"]) != torch.cuda.device_count()):
            raise ConfirmationError("CUDA RNG topology mismatch")
        torch.cuda.set_rng_state_all(
            [item.clone() for item in state["torch_cuda_rng_states"]])


def atomic_exclusive(path: Path, writer):
    temporary = path.with_name("." + path.name + ".incomplete")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(temporary, flags, 0o600)
    try:
        with os.fdopen(fd, "wb") as handle:
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path, follow_symlinks=False)  # fails rather than replaces
        os.unlink(temporary)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def save_checkpoint(path: Path, state) -> dict[str, Any]:
    started = time.perf_counter()
    atomic_exclusive(path, lambda handle: torch.save(state, handle))
    return {"path": str(path.resolve()), "sha256": file_hash(path),
            "size_bytes": path.stat().st_size,
            "write_seconds": time.perf_counter() - started}


def load_checkpoint(path: Path, sha256: str):
    if len(sha256) != 64 or any(character not in "0123456789abcdef"
                                for character in sha256):
        raise ConfirmationError("checkpoint hash must be lowercase SHA-256")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    with os.fdopen(fd, "rb") as handle:
        info = os.fstat(handle.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise ConfirmationError("checkpoint must be a singly-linked regular file")
        digest = hashlib.sha256()
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
        if digest.hexdigest() != sha256:
            raise ConfirmationError("checkpoint hash mismatch")
        handle.seek(0)
        return torch.load(handle, map_location="cpu", weights_only=True)


def write_json(path: Path, value):
    raw = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    atomic_exclusive(path, lambda handle: handle.write(raw))


def sync(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(torch.device(device))


@torch.no_grad()
def evaluate(model, data, step, device):
    sync(device)
    started = time.perf_counter()
    model.eval()
    result = {"step": step, "phase": "initial" if step == 0 else "post_update"}
    for label, inputs, targets in (("train", *data[:2]), ("test", *data[2:])):
        logits = model(inputs)
        loss = float(F.cross_entropy(logits, targets))
        if not math.isfinite(loss) or not bool(torch.isfinite(logits).all()):
            raise ConfirmationError("non-finite evaluation")
        result[label] = {"loss": loss,
                         "accuracy": float((logits.argmax(-1) == targets).float().mean()),
                         "count": len(targets)}
    model.train()
    sync(device)
    return result, time.perf_counter() - started


def train_step(model, optimizer, tracker, inputs, targets, device):
    sync(device)
    started = time.perf_counter()
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss = F.cross_entropy(model(inputs), targets)
    if not bool(torch.isfinite(loss)):
        raise ConfirmationError("non-finite training loss")
    loss.backward()
    if any(p.grad is None or not bool(torch.isfinite(p.grad).all())
           for p in model.parameters()):
        raise ConfirmationError("missing or non-finite gradient")
    diagnostics = tracker.filter_grad() if tracker is not None else None
    optimizer.step()
    if any(not bool(torch.isfinite(p).all()) for p in model.parameters()):
        raise ConfirmationError("non-finite parameter")
    if tracker is not None:
        # Check the O(p) mean and small spectrum/counters every step.  The large
        # p-by-k basis is checked on the 50-step measurement grid and at capture.
        for name in ("S", "grad_mean"):
            value = getattr(tracker, name)
            if value is not None and not bool(torch.isfinite(value).all()):
                raise ConfirmationError(f"non-finite filter {name}")
        if not math.isfinite(float(tracker.max_orthogonality_error)):
            raise ConfirmationError("non-finite filter diagnostic")
    sync(device)
    return float(loss.detach()), diagnostics, time.perf_counter() - started


def threshold_summary(rows):
    previous = None
    first = None
    for row in rows:
        if row["test"]["accuracy"] >= 0.9 and first is None:
            first = {"lower_step_exclusive": previous,
                     "upper_step_inclusive": row["step"]}
        previous = row["step"]
    if first is None:
        first = {"lower_step_exclusive": rows[-1]["step"],
                 "upper_step_inclusive": None}
    sustained = None
    for index, row in enumerate(rows):
        if row["test"]["accuracy"] >= 0.9 and all(
                later["test"]["accuracy"] >= 0.9 for later in rows[index:]):
            sustained = row["step"]
            break
    return {"threshold": 0.9,
            "status": "right_censored" if first["upper_step_inclusive"] is None
                      else "interval_censored",
            **first, "sustained_attainment_step": sustained}


def environment(device, threads):
    result = {"python": platform.python_version(), "platform": platform.platform(),
              "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
              "cudnn": torch.backends.cudnn.version(), "device": str(device),
              "torch_cpu_threads": threads}
    if torch.device(device).type == "cuda":
        index = torch.device(device).index or torch.cuda.current_device()
        result.update({"cuda_device_name": torch.cuda.get_device_name(index),
                       "cuda_capability": list(torch.cuda.get_device_capability(index))})
    return result


def run(args):
    wall_start = time.perf_counter()
    checkpoints = tuple(int(item) for item in args.checkpoint_steps.split(","))
    if checkpoints != CHECKPOINT_STEPS:
        raise ValueError(f"checkpoint steps must be {CHECKPOINT_STEPS}")
    config = resolved_config(args.arm, args.seed, args.steps, args.threads)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ConfirmationError("CUDA requested but unavailable")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(args.seed)
    sources = source_identity()
    cpu_data = get_modular_addition_data(p=113, train_frac=0.3, seed=args.seed)
    split = tensor_set_identity(cpu_data)
    data = tuple(value.to(device) for value in cpu_data)
    model = GrokkingTransformer(**MODEL).to(device)
    parameters = parameter_identity(model)
    optimizer, tracker = make_optimizer(model, args.arm)

    output = Path(args.output_dir).resolve()
    large_root = Path("/tmp/spectral-experiment-artifacts").resolve()
    if large_root not in output.parents:
        raise ConfirmationError("output must be a new child of /tmp/spectral-experiment-artifacts")
    output.mkdir(mode=0o700)  # exclusive; parents must already exist
    env = environment(device, args.threads)
    write_json(output / "run.json", {"schema": SCHEMA, "status": "started",
               "config": config, "config_sha256": canonical_hash(config),
               "source_identity": sources,
               "source_identity_sha256": canonical_hash(sources),
               "split_identity": split, "parameter_identity": parameters,
               "environment": env})

    rows, receipts = [], []
    timing = {"training_seconds": 0.0, "evaluation_seconds": 0.0,
              "checkpoint_capture_and_write_seconds": 0.0}
    completed_step = 0
    try:
        row, seconds = evaluate(model, data, 0, device)
        timing["evaluation_seconds"] += seconds
        row.update({"training_loss_before_update": None,
                    "cumulative_training_seconds": 0.0,
                    "cumulative_evaluation_seconds": timing["evaluation_seconds"]})
        rows.append(row)
        checkpoint_started = time.perf_counter()
        state = checkpoint_state(model, optimizer, tracker, step=0, rows=rows,
                                 timing=timing, config=config, sources=sources,
                                 split=split, parameters=parameters, device=device)
        receipt = save_checkpoint(output / "checkpoint-step-000000.pt", state)
        receipt["capture_and_write_seconds"] = time.perf_counter() - checkpoint_started
        timing["checkpoint_capture_and_write_seconds"] += receipt[
            "capture_and_write_seconds"]
        receipts.append(receipt)

        for step in range(1, args.steps + 1):
            loss, diagnostics, seconds = train_step(
                model, optimizer, tracker, data[0], data[1], device)
            timing["training_seconds"] += seconds
            completed_step = step
            if step % 50 == 0:
                if tracker is not None and tracker.V is not None and not bool(
                        torch.isfinite(tracker.V).all()):
                    raise ConfirmationError("non-finite filter basis")
                row, seconds = evaluate(model, data, step, device)
                timing["evaluation_seconds"] += seconds
                row.update({"training_loss_before_update": loss,
                            "cumulative_training_seconds": timing["training_seconds"],
                            "cumulative_evaluation_seconds": timing["evaluation_seconds"]})
                if diagnostics is not None:
                    row["filter"] = {key: diagnostics.get(key) for key in
                                     ("filtering_active", "basis_rank", "effective_rank",
                                      "stable_update", "stabilization_count",
                                      "max_orthogonality_error")}
                rows.append(row)
                print(json.dumps({"arm": args.arm, "seed": args.seed, "step": step,
                                  "train_accuracy": row["train"]["accuracy"],
                                  "test_accuracy": row["test"]["accuracy"],
                                  "training_seconds": timing["training_seconds"],
                                  "evaluation_seconds": timing["evaluation_seconds"]}),
                      flush=True)
            if step in CHECKPOINT_STEPS:
                checkpoint_started = time.perf_counter()
                state = checkpoint_state(model, optimizer, tracker, step=step,
                                         rows=rows, timing=timing, config=config,
                                         sources=sources, split=split,
                                         parameters=parameters, device=device)
                receipt = save_checkpoint(
                    output / f"checkpoint-step-{step:06d}.pt", state)
                receipt["capture_and_write_seconds"] = (
                    time.perf_counter() - checkpoint_started)
                timing["checkpoint_capture_and_write_seconds"] += receipt[
                    "capture_and_write_seconds"]
                receipts.append(receipt)

        if source_identity()["files"] != sources["files"]:
            raise ConfirmationError("scientific source file changed during run")
        end_to_end_seconds = time.perf_counter() - wall_start
        accounted_seconds = sum(timing.values())
        result = {"schema": SCHEMA, "status": "complete", "config": config,
                  "config_sha256": canonical_hash(config), "source_identity": sources,
                  "source_identity_sha256": canonical_hash(sources),
                  "split_identity": split, "parameter_identity": parameters,
                  "environment": env, "completed_steps": args.steps,
                  "evaluation_rows": rows, "test_accuracy_90": threshold_summary(rows),
                  "timing": {**timing,
                             "end_to_end_seconds_before_final_json":
                                 end_to_end_seconds,
                             "setup_checks_progress_and_other_seconds":
                                 max(0.0, end_to_end_seconds - accounted_seconds),
                             "training_includes":
                                 "synchronization+forward+backward+filter_if_any+AdamW_update+finite_checks",
                             "evaluation": "common synchronized post-update phase"},
                  "checkpoints": receipts}
        write_json(output / "metrics.json", result)
        return result
    except BaseException as error:
        try:
            write_json(output / "failure.json", {"schema": SCHEMA,
                       "status": "failed_preserved", "error_type": type(error).__name__,
                       "error": str(error), "completed_step": completed_step})
        except BaseException:
            pass
        raise


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=ARMS)
    parser.add_argument("--seed", required=True, type=int, choices=SEEDS)
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint-steps", default=",".join(map(str, CHECKPOINT_STEPS)))
    parser.add_argument("--threads", type=int, default=1)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
