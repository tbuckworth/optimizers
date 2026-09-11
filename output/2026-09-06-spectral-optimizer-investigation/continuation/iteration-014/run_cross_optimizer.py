#!/usr/bin/env python3
"""I14 bounded fresh-source calibration/confirmation; exclusive, no restarts."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import subprocess
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(name, "1")

import numpy as np
import torch
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(HERE))
import data_plan as plans
import optimizer_core as core

BASES = ("sgd", "sgdm", "adamw")
POLICIES = ("raw", "current32")
TARGETS = ("clean", "fixed")
RATES = {"sgd": (.03, .1, .3), "sgdm": (.003, .01, .03),
         "adamw": (.0003, .001, .003)}
COMPETENCE_FLOOR = .85
ARTIFACT_CAP = 3 * 1024**3
RESERVE = 1024**2
LIMITS = {"smoke": 100, "calibration": 1800, "confirmation": 3600}
TIMING_SAFETY_FACTOR = 1.5
CANONICAL_SHA = "9280c7d360aef4c775baf0d781a5afd56e016e9cfb93ef829c3ae4b8cc3df943"


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024**2), b""):
            result.update(chunk)
    return result.hexdigest()


def json_bytes(value):
    return (json.dumps(plans.json_tree(value), indent=2, allow_nan=False) + "\n").encode()


def failure_fingerprint(value):
    """Semantic fingerprint retaining nonfinite tensor/float bits for startup parity."""
    def encode(item):
        if type(item) is torch.Tensor:
            tensor = item.detach().cpu().contiguous()
            raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
            return ["tensor", str(tensor.dtype), list(tensor.shape), hashlib.sha256(raw).hexdigest()]
        if type(item) is dict:
            return ["dict", [[encode(key), encode(val)] for key, val in item.items()]]
        if type(item) in (list, tuple):
            return [type(item).__name__, [encode(val) for val in item]]
        if type(item) is float:
            return ["float", item.hex()]
        if item is None or type(item) in (bool, int, str):
            return [type(item).__name__, item]
        raise ValueError("Unsupported startup-failure evidence type")
    return hashlib.sha256(json.dumps(encode(value), separators=(",", ":")).encode()).hexdigest()


class BudgetWriter:
    def __init__(self, handle, remaining):
        self.handle, self.remaining = handle, remaining

    def write(self, value):
        if len(value) > self.remaining:
            raise RuntimeError("I14 shared artifact budget exhausted")
        count = self.handle.write(value)
        self.remaining -= count
        return count

    def flush(self):
        return self.handle.flush()


class Run:
    def __init__(self, root, phase, limit):
        self.root, self.phase = Path(root), phase
        self.path = self.root / phase
        self.path.mkdir(exist_ok=False)
        self.started = time.monotonic()
        self.deadline = self.started + limit
        self.artifacts = []

    def used(self):
        return sum(path.stat().st_size for path in self.root.rglob("*") if path.is_file())

    def check(self):
        if time.monotonic() >= self.deadline:
            raise RuntimeError("I14 cooperative phase wall limit exceeded")
        if torch.cuda.is_available() and torch.cuda.max_memory_allocated() > 4 * 1024**3:
            raise RuntimeError("I14 PyTorch GPU allocation ceiling exceeded")
        if shutil.disk_usage(self.root).free < 1024**3:
            raise RuntimeError("I14 output volume has less than1GiB free")

    def save(self, name, value, tensor=False):
        self.check()
        if Path(name).name != name:
            raise ValueError("Artifact name must be a single filename")
        path = self.path / name
        allowance = ARTIFACT_CAP - self.used() - RESERVE
        if tensor:
            with path.open("xb") as handle:
                torch.save(value, BudgetWriter(handle, allowance))
        else:
            raw = json_bytes(value)
            if len(raw) > allowance:
                raise RuntimeError("I14 JSON exceeds shared artifact ceiling")
            with path.open("xb") as handle:
                handle.write(raw)
        record = {"name": name, "bytes": path.stat().st_size, "sha256": digest(path)}
        self.artifacts.append(record)
        self.check()
        return record

    def terminal(self, name, value):
        raw = json_bytes(value)
        if len(raw) > RESERVE or len(raw) + self.used() > ARTIFACT_CAP:
            raise RuntimeError("I14 terminal metadata exceeds reserved capacity")
        with (self.path / name).open("xb") as handle:
            handle.write(raw)


def source_manifest(commit):
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("A full frozen commit hash is required")
    files = [ROOT / "spectral_filter.py", HERE.parent / "iteration-009/neural_core.py",
             HERE / "optimizer_core.py", HERE / "data_plan.py", Path(__file__).resolve(),
             HERE / "test_optimizer_core.py", HERE / "test_data_plan.py",
             HERE / "test_cross_optimizer_runner.py", HERE / "protocol.md"]
    assert digest(files[0]) == CANONICAL_SHA
    hashes = {}
    for path in files:
        name = str(path.relative_to(ROOT))
        current = digest(path)
        frozen = subprocess.check_output(["git", "show", commit + ":" + name], cwd=ROOT)
        assert hashlib.sha256(frozen).hexdigest() == current, "Frozen source differs: " + name
        hashes[name] = current
    return hashes


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure():
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    if not torch.cuda.is_available():
        raise RuntimeError("The I14 real/smoke CLI requires the admitted local GPU")
    memory = {line.split(":", 1)[0]: int(line.split()[1]) * 1024
              for line in Path("/proc/meminfo").read_text().splitlines()}
    if memory["MemAvailable"] < 16 * 1024**3:
        raise RuntimeError("Less than16GiB available host RAM")
    occupancy = subprocess.check_output(["nvidia-smi",
        "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"], text=True)
    for line in occupancy.splitlines():
        pid, name, _ = [part.strip() for part in line.split(",", 2)]
        if int(pid) != os.getpid() and name not in {
                "/usr/libexec/gnome-remote-desktop-daemon", "stremio"}:
            raise RuntimeError("Foreign GPU compute process; leaving it untouched")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    free, total = torch.cuda.mem_get_info()
    if free < 8 * 1024**3:
        raise RuntimeError("Less than8GiB free GPU memory")
    torch.cuda.set_per_process_memory_fraction(4 * 1024**3 / total)
    seed_all(2026090714)


def finite(value, label):
    if not bool(torch.isfinite(value).all()):
        raise core.NumericalFailure("nonfinite " + label)


@torch.no_grad()
def evaluate(model, data, calibration=False):
    """FP32 forward/FP64 reductions, no caller mutation or official-test data."""
    modes = [module.training for module in model.modules()]
    result = {}
    groups = (("train", "x", "clean"), ("validation", "vx", "vy"))
    if not calibration:
        groups += (("auxiliary", "ax", "ay"),)
    try:
        model.eval()
        for group, xkey, ykey in groups:
            x, labels = data[xkey], data[ykey]
            if len(x) == 0 or len(x) != len(labels):
                raise ValueError("Invalid evaluation dataset")
            sums = {"clean_ce": 0., "soft_ce": 0., "clean_accuracy": 0.,
                    "mean_max_probability": 0., "mean_true_label_probability": 0.}
            if group == "train":
                sums.update(fixed_ce=0., fixed_accuracy=0.)
            for start in range(0, len(x), 512):
                stop = min(start + 512, len(x))
                logits = model(x[start:stop])
                finite(logits, "evaluation logits")
                if logits.dtype != torch.float32:
                    raise ValueError("Evaluation requires float32 forward")
                logp = F.log_softmax(logits.double(), dim=-1)
                rows = torch.arange(stop - start, device=logits.device)
                y = labels[start:stop]
                nll = -logp[rows, y]
                probs = logp.exp()
                sums["clean_ce"] += nll.sum().item()
                sums["soft_ce"] += (.1 * nll - .9 * logp.mean(dim=-1)).sum().item()
                sums["clean_accuracy"] += (logits.argmax(-1) == y).sum().item()
                sums["mean_max_probability"] += probs.max(-1).values.sum().item()
                sums["mean_true_label_probability"] += probs[rows, y].sum().item()
                if group == "train":
                    noisy = data["noisy"][start:stop]
                    sums["fixed_ce"] += (-logp[rows, noisy]).sum().item()
                    sums["fixed_accuracy"] += (logits.argmax(-1) == noisy).sum().item()
            values = {key: value / len(x) for key, value in sums.items()}
            if group == "train":
                values["fixed_minus_soft_ce"] = values["fixed_ce"] - values["soft_ce"]
            if not all(math.isfinite(value) for value in values.values()):
                raise core.NumericalFailure("nonfinite evaluation summary")
            result[group] = values
    finally:
        for module, training in zip(model.modules(), modes):
            module.training = training
    return result


def selection(curve):
    candidates = [row for row in curve if row["horizon"] > 0]
    if not candidates:
        raise ValueError("No selectable evaluation")
    by_ce = min(candidates, key=lambda row: (row["validation"]["clean_ce"], row["horizon"]))
    by_acc = min(candidates, key=lambda row: (-row["validation"]["clean_accuracy"], row["horizon"]))
    return {"minimum_validation_ce": by_ce["horizon"],
            "maximum_validation_accuracy": by_acc["horizon"]}


def run_trajectory(run, data, plan, base, lr, policy, target, *, steps=2000,
                   horizons=plans.HORIZONS, model_spec=None):
    # Validate before model construction, data forward or artifact creation.
    if base not in BASES or policy not in POLICIES or target not in TARGETS \
            or type(lr) is not float or not math.isfinite(lr) or lr <= 0 \
            or type(steps) is not int or steps <= 0 \
            or type(horizons) not in (tuple, list) or not horizons \
            or any(type(h) is not int for h in horizons) \
            or list(horizons) != sorted(set(horizons)) \
            or horizons[0] != 0 or horizons[-1] != steps:
        raise ValueError("Invalid trajectory configuration")
    if type(plan) is not dict or plan.get("phase") not in ("calibration", "confirmation", "smoke"):
        raise ValueError("Invalid trajectory plan phase")
    if plan["phase"] != "smoke":
        plans.validate_plan(plan)
        if steps != plans.STEPS or tuple(horizons) != plans.HORIZONS or model_spec is not None:
            raise ValueError("Real trajectory differs from registered dimensions/horizons")
        if plan["phase"] == "calibration" and (policy != "raw" or target != "clean"):
            raise ValueError("Calibration is clean/raw only")
    elif type(plan.get("seed")) is not int or type(plan.get("initialization_seed")) is not int \
            or not 0 <= plan["initialization_seed"] < 2**31 or steps > 110:
        raise ValueError("Invalid synthetic trajectory plan")
    batches_raw = plan["training_batches"]
    if type(data) is not dict or tuple(data) != plans.DATA_KEYS \
            or any(type(value) is not torch.Tensor or value.layout != torch.strided
                   for value in data.values()):
        raise ValueError("Invalid data tensor layout")
    spec = {"input_dim": 784, "width": 64, "classes": 10}
    if model_spec is not None:
        if type(model_spec) is not dict or not set(model_spec).issubset(spec):
            raise ValueError("Invalid model specification")
        spec.update(model_spec)
    if any(type(value) is not int or value <= 0 for value in spec.values()) or spec["classes"] < 2:
        raise ValueError("Invalid model dimensions")
    if any(value.device != data["x"].device for value in data.values()):
        raise ValueError("Evaluation/training data devices differ")
    for xkey, ykey in (("x", "clean"), ("x", "noisy"), ("vx", "vy"), ("ax", "ay")):
        xvalue, yvalue = data[xkey], data[ykey]
        unused_aux = plan["phase"] == "calibration" and xkey == "ax"
        if xvalue.dtype != torch.float32 or xvalue.ndim != 2 \
                or xvalue.shape[1] != spec["input_dim"] or yvalue.dtype != torch.long \
                or yvalue.ndim != 1 or len(xvalue) != len(yvalue) \
                or (len(xvalue) == 0 and not unused_aux):
            raise ValueError("Invalid training/evaluation data topology or values")
        if not unused_aux and (not bool(torch.isfinite(xvalue).all())
                or not bool(((yvalue >= 0) & (yvalue < spec["classes"])).all())):
            raise ValueError("Invalid training/evaluation data topology or values")
    if type(batches_raw) is not np.ndarray or batches_raw.dtype != np.int64 \
            or batches_raw.ndim != 2 or batches_raw.shape[0] < steps \
            or batches_raw.shape[1] != plans.BATCH \
            or not bool(((batches_raw >= 0) & (batches_raw < len(data["x"]))).all()):
        raise ValueError("Invalid batch plan or data layout")
    seed_all(plan["initialization_seed"])
    device = data["x"].device
    model = core.i9.make_model(plan["initialization_seed"], device, **(model_spec or {}))
    optimizer = core.make_optimizer(model, base, lr)
    tracker = core.i9.make_tracker(model, optimizer)
    identity = f"s{plan['seed']}-{base}-lr{lr:.8g}-{target}-{policy}"
    calibration = plan["phase"] == "calibration"
    initial = core.snapshot(model, optimizer, tracker, base, lr)
    initial_digest = core.i9.tree_digest(initial)
    initial_record = run.save("state-" + identity + "-h0.pt", initial, tensor=True)
    curve, diagnostics, checkpoints = [], [], []
    # Invalid h0 aborts the phase: it is not a post-start scientific failure.
    curve.append({"horizon": 0, **evaluate(model, data, calibration)})
    checkpoints.append({"horizon": 0, "full_state": initial_record,
                        "full_state_digest": initial_digest})
    last_valid_horizon, completed = 0, 0
    warmup_digest, warmup_evaluation_digest, failure = None, None, None
    startup_failure_signature = None
    batches = torch.as_tensor(plan["training_batches"], dtype=torch.long, device=device)
    target_values = data["clean"] if target == "clean" else data["noisy"]
    for step in range(1, steps + 1):
        if step == 1 or step % 25 == 0:
            run.check()
        try:
            batch = batches[step - 1]
            diagnostic = core.train_step(model, optimizer, tracker,
                data["x"][batch], target_values[batch], policy, base)
            diagnostics.append({"step": step, **diagnostic})
            completed = step
            if step in horizons:
                measured = evaluate(model, data, calibration)
                curve.append({"horizon": step, **measured})
                last_valid_horizon = step
                checkpoint = {"horizon": step}
                if step in (100, steps):
                    state = core.snapshot(model, optimizer, tracker, base, lr)
                    state_digest = core.i9.tree_digest(state)
                    checkpoint.update(full_state=run.save(
                        "state-" + identity + f"-h{step}.pt", state, tensor=True),
                        full_state_digest=state_digest)
                    if step == 100:
                        warmup_digest = state_digest
                        warmup_evaluation_digest = core.i9.tree_digest(curve[-1])
                else:
                    checkpoint["model_state"] = run.save(
                        "model-" + identity + f"-h{step}.pt",
                        {key: value.detach().cpu().clone()
                         for key, value in model.state_dict().items()}, tensor=True)
                checkpoints.append(checkpoint)
        except core.NumericalFailure as exc:
            failure = {"type": type(exc).__name__, "message": str(exc),
                       "attempted_step": step, "completed_steps": completed,
                       "last_valid_horizon": last_valid_horizon}
            # Retain nonfinite tensors without asking a finite-tree digest to accept them.
            failed = core.snapshot(model, optimizer, tracker, base, lr)
            if warmup_digest is None:
                startup_failure_signature = failure_fingerprint({
                    "state": failed, "failure": failure, "curve": curve,
                    "steps": [{key: value for key, value in row.items() if key != "policy"}
                              for row in diagnostics]})
            failure["state_artifact"] = run.save("failed-state-" + identity + ".pt", failed, tensor=True)
            break
    result = {"schema": "i14_trajectory_v1", "id": identity,
              "phase": plan["phase"], "seed": plan["seed"], "base": base,
              "lr": lr, "policy": policy, "target": target,
              "status": "numerical_failure" if failure else "complete",
              "requested_steps": steps, "completed_steps": completed,
              "initial_full_state_digest": initial_digest,
              "initial_model_digest": core.i9.tree_digest(initial["model_state"]),
              "warmup_full_state_digest": warmup_digest,
              "initial_evaluation_digest": core.i9.tree_digest(curve[0]),
              "warmup_evaluation_digest": warmup_evaluation_digest,
              "startup_failure_signature": startup_failure_signature,
              "curve": curve, "steps": diagnostics, "checkpoints": checkpoints,
              "failure": failure,
              "selected_horizons": selection(curve) if failure is None else None}
    artifact = run.save("curve-" + identity + ".json", result)
    print(json.dumps({"trajectory": identity, "status": result["status"],
                      "elapsed_seconds": time.monotonic() - run.started}), flush=True)
    return {key: result[key] for key in ("id", "phase", "seed", "base", "lr", "policy",
        "target", "status", "completed_steps", "initial_model_digest",
        "initial_full_state_digest", "warmup_full_state_digest", "initial_evaluation_digest",
        "warmup_evaluation_digest", "startup_failure_signature")} | {"artifact": artifact}, result


def choose_rates(results):
    for row in results:
        if row["phase"] != "calibration" or row["policy"] != "raw" or row["target"] != "clean" \
                or row["requested_steps"] != plans.STEPS \
                or type(row["completed_steps"]) is not int \
                or not 0 <= row["completed_steps"] <= plans.STEPS \
                or row["status"] not in ("complete", "numerical_failure"):
            raise ValueError("Invalid calibration trajectory identity/status")
        actual = [point["horizon"] for point in row["curve"]]
        if not actual or actual != list(plans.HORIZONS[:len(actual)]) \
                or actual[-1] > row["completed_steps"]:
            raise ValueError("Calibration curve horizons differ")
        if row["status"] == "complete" and (row["completed_steps"] != plans.STEPS
                or actual != list(plans.HORIZONS)):
            raise ValueError("Complete calibration curve is incomplete")
        for point in row["curve"]:
            val = point["validation"]
            if any(type(val[key]) is not float or not math.isfinite(val[key])
                   for key in ("clean_ce", "clean_accuracy")) \
                    or val["clean_ce"] < 0 or not 0 <= val["clean_accuracy"] <= 1:
                raise ValueError("Invalid calibration validation metric")
    expected = {(seed, base, lr) for seed in plans.CALIBRATION_SEEDS
                for base in BASES for lr in RATES[base]}
    indexed = {(r["seed"], r["base"], r["lr"]): r for r in results}
    if set(indexed) != expected or len(indexed) != len(results):
        raise ValueError("Calibration membership differs")
    choices, candidates = {}, []
    for base in BASES:
        eligible = []
        for lr in RATES[base]:
            rows = [indexed[(seed, base, lr)] for seed in plans.CALIBRATION_SEEDS]
            finals = [next((point["validation"] for point in row["curve"]
                            if point["horizon"] == plans.STEPS), None) for row in rows]
            finite_complete = all(row["status"] == "complete" and final is not None
                and all(math.isfinite(final[key]) for key in ("clean_ce", "clean_accuracy"))
                for row, final in zip(rows, finals))
            competent = finite_complete and all(final["clean_accuracy"] >= COMPETENCE_FLOOR
                                                for final in finals)
            mean_ce = sum(final["clean_ce"] for final in finals) / len(finals) if finite_complete else None
            candidate = {"base": base, "lr": lr, "finite_complete": finite_complete,
                "eligible": bool(competent), "validation_by_seed": finals,
                "mean_final_validation_ce": mean_ce,
                "competence_floor_each_seed": COMPETENCE_FLOOR}
            candidates.append(candidate)
            if competent:
                eligible.append((mean_ce, lr))
        choices[base] = min(eligible)[1] if eligible else None
    return {"schema": "i14_calibration_selection_v1", "selected_rates": choices,
            "ready_for_confirmation": all(lr is not None for lr in choices.values()),
            "candidates": candidates, "confirmation_outcomes_used": False}


def verify_phase(path, sources):
    path = Path(path)
    completion = json.loads((path / "completion.json").read_text())
    if completion["status"] != "complete" or completion["source_hashes"] != sources:
        raise ValueError("Prerequisite phase/source is not complete")
    for record in completion["artifacts"]:
        file = path / record["name"]
        if Path(record["name"]).name != record["name"] or file.stat().st_size != record["bytes"] \
                or digest(file) != record["sha256"]:
            raise ValueError("Prerequisite artifact mismatch")
    return completion


def validate_pairs(entries, expected_seeds=plans.CONFIRMATION_SEEDS):
    indexed = {(r["seed"], r["base"], r["target"], r["policy"]): r for r in entries}
    expected = {(seed, base, target, policy) for seed in expected_seeds for base in BASES
                for target in TARGETS for policy in POLICIES}
    if len(indexed) != len(entries) or set(indexed) != expected:
        raise ValueError("Paired trajectory membership differs")
    checked, missing = 0, 0
    for seed in expected_seeds:
        initial = {r["initial_model_digest"] for r in entries if r["seed"] == seed}
        assert len(initial) == 1, "Initial weights differ within seed"
        for base in BASES:
            for target in TARGETS:
                raw, current = (indexed[(seed, base, target, policy)] for policy in POLICIES)
                assert raw["initial_full_state_digest"] == current["initial_full_state_digest"]
                assert raw["initial_evaluation_digest"] == current["initial_evaluation_digest"]
                for row in (raw, current):
                    if row["status"] == "complete":
                        assert row["warmup_full_state_digest"] is not None
                        assert row["warmup_evaluation_digest"] is not None
                if raw["warmup_full_state_digest"] is not None and current["warmup_full_state_digest"] is not None:
                    assert raw["warmup_full_state_digest"] == current["warmup_full_state_digest"], "Warmup diverged"
                    assert raw["warmup_evaluation_digest"] == current["warmup_evaluation_digest"], "Warmup evaluation differs"
                    checked += 1
                else:
                    if raw["status"] != "numerical_failure" or current["status"] != "numerical_failure" \
                            or raw["startup_failure_signature"] is None \
                            or raw["startup_failure_signature"] != current["startup_failure_signature"]:
                        raise ValueError("Pre-warmup paired behavior differs before treatment")
                    missing += 1
    return {"expected_pairs": len(expected) // 2, "available_pairs_checked_exact": checked,
            "pairs_unavailable_due_to_numerical_failure": missing,
            "all_pairs_reached_warmup": missing == 0}


def synthetic_data(device):
    gen = torch.Generator().manual_seed(21414)
    x = torch.rand((384, 784), generator=gen)
    y = torch.randint(10, (384,), generator=gen)
    noisy = torch.randint(10, (128,), generator=gen)
    data = {"x": x[:128].to(device), "clean": y[:128].to(device), "noisy": noisy.to(device),
            "vx": x[128:256].to(device), "vy": y[128:256].to(device),
            "ax": x[256:].to(device), "ay": y[256:].to(device)}
    plan = {"phase": "smoke", "seed": 21414, "initialization_seed": 21414,
            "training_batches": torch.randint(128, (110, 64), generator=gen).numpy()}
    return data, plan


def smoke(run):
    data, plan = synthetic_data("cuda")
    entries, roundtrips = [], []
    for base in BASES:
        for target in TARGETS:
            for policy in POLICIES:
                entry, result = run_trajectory(run, data, plan, base, RATES[base][1],
                    policy, target, steps=110, horizons=(0, 100, 110))
                assert entry["status"] == "complete"
                final = result["checkpoints"][-1]
                state = torch.load(run.path / final["full_state"]["name"], weights_only=True, map_location="cpu")
                model, optimizer, tracker = core.restore(state, "cuda")
                restored = core.snapshot(model, optimizer, tracker, base, RATES[base][1])
                assert core.i9.equal_tree(state, restored)
                roundtrips.append(final["full_state_digest"])
                entries.append(entry)
                del model, optimizer, tracker, restored, state, result
    pair_checks = validate_pairs(entries, expected_seeds=(21414,))
    run.save("smoke-checks.json", {"status": "pass", "cells": len(entries),
        "warmup_pairs": 6, "pair_checks": pair_checks, "full_state_roundtrips": len(roundtrips),
        "new_training_updates": 1320, "real_data_read": False, "entries": entries})
    return entries


def calibration(run):
    saved_plans = {}
    for phase, seeds in (("calibration", plans.CALIBRATION_SEEDS),
                         ("confirmation", plans.CONFIRMATION_SEEDS)):
        for seed in seeds:
            plan = plans.make_plan(seed, phase)
            run.save(f"plan-{phase}-s{seed}.json", plan)
            saved_plans[(phase, seed)] = plan
    inputs = {name: digest(plans.DATA / name) for name in
              ("train-images-idx3-ubyte", "train-labels-idx1-ubyte")}
    run.save("dataset-inputs.json", inputs)
    x, y = plans.read_training()
    entries, results = [], []
    for seed in plans.CALIBRATION_SEEDS:
        plan = saved_plans[("calibration", seed)]
        data, corruption = plans.data_for_plan(x, y, plan, "cuda")
        run.save(f"corruption-s{seed}.json", corruption)
        for base in BASES:
            for lr in RATES[base]:
                entry, result = run_trajectory(run, data, plan, base, lr, "raw", "clean")
                entries.append(entry)
                # Only retain the fields needed for rate selection in process memory.
                results.append({key: result[key] for key in ("seed", "base", "lr", "status", "curve",
                    "phase", "policy", "target", "requested_steps", "completed_steps")})
                del result
        del data
    decision = choose_rates(results)
    decision["dataset_inputs"] = inputs
    run.save("selection.json", decision)
    run.save("trajectories.json", {"entries": entries, "expected": 18})
    assert {name: digest(plans.DATA / name) for name in inputs} == inputs
    return entries


def confirmation(run, sources):
    previous = run.root / "calibration"
    verify_phase(previous, sources)
    index = json.loads((previous / "trajectories.json").read_text())
    rows = [json.loads((previous / row["artifact"]["name"]).read_text()) for row in index["entries"]]
    independent = choose_rates(rows)
    selected = json.loads((previous / "selection.json").read_text())
    assert {key: selected[key] for key in independent} == independent
    if not selected["ready_for_confirmation"]:
        raise ValueError("No competent complete calibration rate for every base")
    inputs = selected["dataset_inputs"]
    assert {name: digest(plans.DATA / name) for name in inputs} == inputs
    run.save("calibration-binding.json", {"selection_sha256": digest(previous / "selection.json"),
        "completion_sha256": digest(previous / "completion.json"), "selection": selected})
    x, y = plans.read_training()
    entries = []
    for seed in plans.CONFIRMATION_SEEDS:
        source_plan = previous / f"plan-confirmation-s{seed}.json"
        plan = plans.plan_from_json(json.loads(source_plan.read_text()))
        copied = run.save(f"plan-confirmation-s{seed}.json", plan)
        assert copied["sha256"] == digest(source_plan)
        data, corruption = plans.data_for_plan(x, y, plan, "cuda")
        run.save(f"corruption-s{seed}.json", corruption)
        for base in BASES:
            for target in TARGETS:
                for policy in POLICIES:
                    entry, result = run_trajectory(run, data, plan, base,
                        selected["selected_rates"][base], policy, target)
                    entries.append(entry)
                    del result
        del data
    pair_checks = validate_pairs(entries)
    expected = {(s, b, t, p) for s in plans.CONFIRMATION_SEEDS for b in BASES
                for t in TARGETS for p in POLICIES}
    assert {(r["seed"], r["base"], r["target"], r["policy"]) for r in entries} == expected
    run.save("trajectories.json", {"entries": entries, "expected": 36,
        "initial_weights_shared": True, "warmup_pair_checks": pair_checks})
    assert {name: digest(plans.DATA / name) for name in inputs} == inputs
    return entries


def main():
    if not __debug__:
        raise RuntimeError("I14 must not run with Python assertion checks disabled")
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--phase", required=True, choices=tuple(LIMITS))
    parser.add_argument("--frozen-commit", required=True)
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    if root.parent != Path("/tmp/spectral-experiment-artifacts") or not root.name.startswith("spectral-i14-001."):
        raise ValueError("I14 requires its exclusive validated large-volume root")
    device = subprocess.check_output(["findmnt", "-n", "-o", "SOURCE", "-T", str(root)], text=True).strip()
    if device != "/dev/RECONFIGURE_FOR_LOCAL_STORAGE":
        raise RuntimeError("I14 root is not on the intended large volume")
    sources = source_manifest(args.frozen_commit)
    if args.phase != "smoke":
        smoke_completion = verify_phase(root / "smoke", sources)
        smoke_updates = smoke_completion["completed_training_updates"]
        if smoke_updates != 1320 or smoke_completion["numerical_failures"] != 0:
            raise ValueError("Synthetic smoke coverage differs")
        forecast = {phase: smoke_completion["elapsed_seconds"] / smoke_updates * updates
                    * TIMING_SAFETY_FACTOR
                    for phase, updates in (("calibration", 36000), ("confirmation", 72000))}
        if any(estimate >= LIMITS[phase] for phase, estimate in forecast.items()):
            raise RuntimeError("Smoke timing projection exceeds the frozen real-phase ceiling")
    else:
        forecast = None
    if args.phase == "confirmation":
        verify_phase(root / "calibration", sources)
        selected = json.loads((root / "calibration/selection.json").read_text())
        if not selected["ready_for_confirmation"]:
            raise ValueError("Calibration does not qualify confirmation")
    with (root / ("attempt-" + args.phase + ".json")).open("xb") as handle:
        handle.write(json_bytes({"phase": args.phase, "frozen_commit": args.frozen_commit,
            "pid": os.getpid(), "source_hashes": sources, "restart": "forbidden"}))
    run = Run(root, args.phase, LIMITS[args.phase])
    try:
        configure()
        run.save("manifest.json", {"schema": "i14_manifest_v1", "phase": args.phase,
            "frozen_commit": args.frozen_commit, "source_hashes": sources,
            "torch_version": str(torch.__version__), "numpy_version": np.__version__,
            "python_version": sys.version, "gpu": torch.cuda.get_device_name(),
            "rates": RATES, "competence_floor": COMPETENCE_FLOOR,
            "smoke_based_seconds_forecast": forecast,
            "timing_safety_factor": TIMING_SAFETY_FACTOR,
            "cloud_spend_usd": 0, "authorized_budget_usd": 100,
            "shared_artifact_cap_bytes": ARTIFACT_CAP, "wall_limit_seconds": LIMITS[args.phase]})
        entries = smoke(run) if args.phase == "smoke" else calibration(run) \
            if args.phase == "calibration" else confirmation(run, sources)
        assert source_manifest(args.frozen_commit) == sources
        failures = sum(row["status"] == "numerical_failure" for row in entries)
        run.terminal("completion.json", {"schema": "i14_completion_v1", "status": "complete",
            "phase": args.phase, "source_hashes": sources, "frozen_commit": args.frozen_commit,
            "trajectories": len(entries), "numerical_failures": failures,
            "all_requested_endpoints_present": failures == 0,
            "completed_training_updates": sum(row["completed_steps"] for row in entries),
            "elapsed_seconds": time.monotonic() - run.started,
            "peak_torch_gpu_bytes": torch.cuda.max_memory_allocated(),
            "shared_artifact_bytes_before_completion": run.used(), "artifacts": run.artifacts})
        print(json.dumps({"phase": args.phase, "status": "complete", "failures": failures}), flush=True)
    except BaseException as exc:
        run.terminal("failure.json", {"schema": "i14_phase_failure_v1", "status": "failed",
            "phase": args.phase, "exception_type": type(exc).__name__, "message": str(exc)[:2000],
            "source_hashes": sources, "elapsed_seconds": time.monotonic() - run.started,
            "artifacts": run.artifacts, "restart": "forbidden"})
        raise


if __name__ == "__main__":
    main()
