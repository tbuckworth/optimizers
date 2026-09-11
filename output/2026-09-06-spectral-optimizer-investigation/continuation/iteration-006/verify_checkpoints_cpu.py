#!/usr/bin/env python3
"""Independent CPU checkpoint/data audit; imports no experimental modules.

Tolerances were fixed in audit-reexecution-decision.md before outcome access.
This verifies saved models, not the unsaved training vectors or Adam moments.
No training, GPU initialization, source modification or output overwrite.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import time
import traceback

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
CHECKPOINTS = ("final", "min_val_ce", "max_val_accuracy", "warmup100")
ARMS = ("adamw", "current32", "lagged32", "lagged32_current_norm",
        "scalar_current32", "scalar_lagged32")
SHAPES = {"0.weight": (64, 784), "0.bias": (64,), "2.weight": (10, 64), "2.bias": (10,)}
CE_ATOL, ACCURACY_EXAMPLES = 5e-5, 1


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "Duplicate JSON key")
            result[key] = value
        return result
    def invalid(value):
        raise ValueError("Nonfinite JSON constant: " + value)
    return json.loads(Path(path).read_bytes(), object_pairs_hook=pairs, parse_constant=invalid)


def binding(item):
    path = Path(item["path"])
    require(path.is_absolute() and path.is_file(), "Missing absolute artifact")
    require(path.stat().st_size == item["size_bytes"] and sha(path) == item["sha256"],
            "Artifact binding mismatch: " + str(path))
    return path


def idx(path, images, count):
    raw = Path(path).read_bytes()
    if images:
        require(struct.unpack(">IIII", raw[:16]) == (2051, count, 28, 28)
                and len(raw) == 16 + count * 784, "Invalid image IDX")
        return torch.from_numpy(np.frombuffer(raw, np.uint8, offset=16).copy().reshape(count, 784)).float() / 255
    require(struct.unpack(">II", raw[:8]) == (2049, count) and len(raw) == 8 + count,
            "Invalid label IDX")
    labels = torch.from_numpy(np.frombuffer(raw, np.uint8, offset=8).copy()).long()
    require(bool(((labels >= 0) & (labels < 10)).all()), "Invalid digit label")
    return labels


def independent_plan(seed, audit=False):
    def rng(stream):
        namespace = [20260906, seed, stream] if audit else [20260906, 3, seed, stream]
        return np.random.default_rng(np.random.SeedSequence(namespace))
    permutation = rng(0).permutation(60000)
    return {
        "seed": seed, "initialization_seed": int(rng(3).integers(0, 2**32, dtype=np.uint32)),
        "train_indices": permutation[:5000], "validation_indices": permutation[5000:10000],
        "auxiliary_indices": permutation[10000:15000],
        "replacement_uniforms": rng(1).random(5000),
        "replacement_digits": rng(2).integers(0, 10, size=5000),
        "training_batches": rng(4).integers(0, 5000, size=(2000, 64)),
        "primary_probe_batches": rng(5).integers(0, 5000, size=(2000, 256)),
        "auxiliary_probe_batches": rng(6).integers(0, 5000, size=(2000, 256)),
    }


def selectors(run):
    rows = run["validation_trajectory"]
    require([row["step"] for row in rows] == list(range(0, 2001, 100)), "Incomplete validation grid")
    for row in rows:
        valid_metric(row, 5000)
    # Python min/max preserve the first item on ties.
    steps = {"final": 2000, "warmup100": 100,
             "min_val_ce": min(rows, key=lambda row: row["cross_entropy"])["step"],
             "max_val_accuracy": max(rows, key=lambda row: row["accuracy"])["step"]}
    require(steps == run["checkpoint_steps"], "Incorrect strict earliest selectors")
    return {row["step"]: row for row in rows}


def valid_metric(row, count):
    require(row["count"] == count, "Evaluation cardinality")
    require(math.isfinite(row["cross_entropy"]) and row["cross_entropy"] >= 0, "Invalid CE")
    require(math.isfinite(row["accuracy"]) and 0 <= row["accuracy"] <= 1
            and abs(row["accuracy"] * count - round(row["accuracy"] * count)) <= 1e-8,
            "Accuracy is not a valid example count")


def state_hash(state):
    require(set(state) == set(SHAPES), "MLP checkpoint schema")
    digest = hashlib.sha256(b"dict")
    for key in sorted(state, key=repr):
        value = state[key]
        require(isinstance(value, torch.Tensor) and value.device.type == "cpu"
                and value.dtype == torch.float32 and tuple(value.shape) == SHAPES[key]
                and bool(torch.isfinite(value).all()), "Invalid checkpoint tensor")
        digest.update(repr(key).encode() + b"\0")
        digest.update(b"tensor" + str((str(value.dtype), tuple(value.shape))).encode())
        digest.update(value.contiguous().numpy().tobytes())
    return digest.hexdigest()


@torch.no_grad()
def evaluate(state, x, y):
    losses, correct = [], 0
    for start in range(0, len(y), 512):
        xb, yb = x[start:start + 512], y[start:start + 512]
        hidden = torch.relu(torch.nn.functional.linear(xb, state["0.weight"], state["0.bias"]))
        logits = torch.nn.functional.linear(hidden, state["2.weight"], state["2.bias"])
        # Re-derive CE from normalized log probabilities, not the producer evaluator.
        log_prob = torch.log_softmax(logits, dim=1)
        losses.append(float((-log_prob.gather(1, yb[:, None])).sum(dtype=torch.float64)))
        correct += int((logits.argmax(dim=1) == yb).sum())
    return {"cross_entropy": math.fsum(losses) / len(y), "accuracy": correct / len(y),
            "correct": correct, "count": len(y)}


def compare(observed, reference):
    valid_metric(reference, observed["count"])
    ce_error = abs(observed["cross_entropy"] - reference["cross_entropy"])
    count_error = abs(observed["correct"] - round(reference["accuracy"] * observed["count"]))
    return {"observed": observed, "recorded": reference, "ce_absolute_error": ce_error,
            "accuracy_error_examples": count_error,
            "passed_tolerance": ce_error <= CE_ATOL and count_error <= ACCURACY_EXAMPLES}


def audit(execution_path, report):
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and not torch.cuda.is_initialized(),
            "CUDA must be hidden and uninitialized")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    manifest = read_json(execution_path)
    is_audit = manifest["mode"] == "audit_reexecution"
    require(manifest["mode"] in ("full", "audit_reexecution"), "Unexpected execution mode")
    seeds, arms = ((60006,), ARMS[1:4]) if is_audit else ((6, 7, 8), ARMS)
    expected = {(seed, noise, arm) for seed in seeds for noise in (0., .9) for arm in arms}
    require(manifest["status"] == "complete" and manifest["all_gates_passed"]
            and manifest["completed_runs"] == len(expected)
            and manifest["test_evaluations"] == 4 * len(expected), "Execution not complete")
    require(datetime.fromisoformat(manifest["test_first_opened_utc"]) >=
            datetime.fromisoformat(manifest["all_training_completed_utc"]), "Early test access")
    report.update(mode=manifest["mode"], execution_sha256=sha(execution_path),
                  numerical_environment={"torch": torch.__version__, "numpy": np.__version__, "device": "cpu"})
    artifacts = [item for group in ("training_data_artifacts", "test_data_artifacts", "plans", "runs", "checkpoints")
                 for item in manifest[group]]
    for item in artifacts:
        binding(item)
    data = {Path(item["path"]).name: Path(item["path"]) for item in
            manifest["training_data_artifacts"] + manifest["test_data_artifacts"]}
    train_x = idx(data["train-images-idx3-ubyte"], True, 60000)
    train_y = idx(data["train-labels-idx1-ubyte"], False, 60000)
    test_x = idx(data["t10k-images-idx3-ubyte"], True, 10000)
    test_y = idx(data["t10k-labels-idx1-ubyte"], False, 10000)
    plans = {}
    for item in manifest["plans"]:
        with np.load(item["path"], allow_pickle=False) as saved:
            seed = int(saved["seed"])
            require(seed in seeds and seed not in plans, "Unexpected or duplicated plan")
            plan = independent_plan(seed, is_audit)
            require(set(saved.files) == set(plan) and all(np.array_equal(saved[k], v) for k, v in plan.items()),
                    "Plan differs from independent seeded reconstruction")
        plans[seed] = (plan, item["sha256"])
    require(set(plans) == set(seeds), "Missing plan")
    cp_bindings = {item["path"]: item for item in manifest["checkpoints"]}
    require(len(cp_bindings) == len(expected), "Checkpoint count")
    seen, warmup = set(), {}
    for item in manifest["runs"]:
        run = read_json(item["path"])
        key = (run["seed"], run["replacement_probability"], run["arm"])
        require(key in expected and key not in seen, "Unexpected/duplicated run")
        seen.add(key)
        plan, plan_sha = plans[key[0]]
        require(run["plan_sha256"] == plan_sha and run["checkpoint_path"] in cp_bindings,
                "Run plan/checkpoint binding")
        validation = selectors(run)
        clean = train_y[plan["train_indices"]]
        mask = torch.from_numpy(plan["replacement_uniforms"] < key[1])
        noisy = torch.where(mask, torch.from_numpy(plan["replacement_digits"]), clean)
        require(float(mask.float().mean()) == run["realized_replacement_fraction"]
                and float((noisy != clean).float().mean()) == run["realized_incorrect_fraction"],
                "Corruption metadata differs from reconstructed fixed labels")
        vx, vy = train_x[plan["validation_indices"]], train_y[plan["validation_indices"]]
        bundle = torch.load(run["checkpoint_path"], map_location="cpu", weights_only=True)
        require(set(bundle) == set(CHECKPOINTS), "Missing checkpoint")
        warm_hash = state_hash(bundle["warmup100"])
        require(warmup.setdefault(key[:2], warm_hash) == warm_hash, "Cross-arm warmup tensors differ")
        for name in CHECKPOINTS:
            state, step = bundle[name], run["checkpoint_steps"][name]
            digest = state_hash(state)
            require(digest == run["checkpoint_sha256"][name], "Checkpoint state fingerprint differs")
            flat_hash = hashlib.sha256(b"".join(state[k].contiguous().numpy().tobytes() for k in SHAPES)).hexdigest()
            if step:
                require(flat_hash == run["trajectory_parameter_sha256"][step - 1], "Checkpoint does not match step trajectory")
            report["checkpoint_links"].append({"run_key": run["run_key"], "checkpoint": name,
                                               "step": step, "state_sha256": digest, "flat_parameter_sha256": flat_hash})
            datasets = [("test", test_x, test_y, run["test"][name]),
                        ("validation", vx, vy, validation[step])]
            if name == "final":
                datasets += [("training_clean", train_x[plan["train_indices"]], clean, run["final_training_clean"]),
                             ("training_noisy", train_x[plan["train_indices"]], noisy, run["final_training_noisy"])]
            for label, x, y, reference in datasets:
                record = compare(evaluate(state, x, y), reference)
                report["evaluations"].append({"run_key": run["run_key"], "checkpoint": name,
                                               "dataset": label, **record})
        print(json.dumps({"verified_cell": run["run_key"], "evaluations": len(report["evaluations"])}), flush=True)
    require(seen == expected and len(report["evaluations"]) == 10 * len(expected), "Incomplete audit coverage")
    for item in artifacts:
        binding(item)
    require(report["execution_sha256"] == sha(execution_path), "Execution manifest changed during audit")
    require(not torch.cuda.is_initialized(), "Unexpected GPU initialization")
    report.update(artifact_bindings_verified=len(artifacts), independent_plans_verified=len(plans),
                  exact_accuracy_evaluations=sum(row["accuracy_error_examples"] == 0 for row in report["evaluations"]),
                  max_ce_absolute_error=max(row["ce_absolute_error"] for row in report["evaluations"]),
                  max_accuracy_error_examples=max(row["accuracy_error_examples"] for row in report["evaluations"]))
    report["status"] = "PASS" if all(row["passed_tolerance"] for row in report["evaluations"]) else "FAIL_TOLERANCE"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    start = time.monotonic()
    report = {"status": "INCOMPLETE", "started_utc": datetime.now(timezone.utc).isoformat(),
              "verifier_sha256": sha(__file__), "tolerances": {"ce_absolute": CE_ATOL, "accuracy_examples": ACCURACY_EXAMPLES},
              "checkpoint_links": [], "evaluations": []}
    with args.output.open("x") as output:
        try:
            audit(args.execution, report)
        except BaseException:
            report.update(status="ERROR", traceback=traceback.format_exc())
            raise
        finally:
            report["elapsed_seconds"] = time.monotonic() - start
            json.dump(report, output, indent=2, allow_nan=False)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
    require(report["status"] == "PASS", "Checkpoint audit failed; retain discrepancies")
    print(json.dumps({key: report[key] for key in ("status", "exact_accuracy_evaluations", "max_ce_absolute_error",
                                                   "max_accuracy_error_examples", "elapsed_seconds")}))


if __name__ == "__main__":
    main()
