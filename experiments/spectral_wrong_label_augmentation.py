#!/usr/bin/env python3
"""Fixed wrong-label augmentation factorial; inert until explicit --execute.

This is a new acquisition, not a resume or alteration of the completed clean
study. Shared numerical/training/resource primitives are imported, not patched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import resource
import shutil
import subprocess
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments import spectral_general_augmentation as ordinary
from experiments import spectral_wrong_label_augmentation_data as data

core = ordinary.core
DOCS = ROOT / "output/2026-09-10-spectral-wrong-label-augmentation"
SCHEMA = "spectral_wrong_label_augmentation_v1"
UNIT = "spectral-wrong-label-augmentation-001.service"
SEEDS, STEPS, BATCH = data.SEEDS, data.STEPS, data.BATCH
POLICIES, AUGMENTATIONS, EVAL_STEPS = ordinary.POLICIES, ordinary.AUGMENTATIONS, ordinary.EVAL_STEPS
DATA, DATA_PINS = ordinary.DATA, ordinary.DATA_PINS
MAX_BYTES, RESERVE_BYTES, DEADLINE_SECONDS = ordinary.MAX_BYTES, ordinary.RESERVE_BYTES, ordinary.DEADLINE_SECONDS
Run, CappedWriter = ordinary.Run, ordinary.CappedWriter
need, digest, array_digest = ordinary.need, ordinary.digest, ordinary.array_digest
byte_inventory = ordinary.byte_inventory
validate_bounds, validate_gpu_clients = ordinary.validate_bounds, ordinary.validate_gpu_clients
CORRUPTION = {"kind": "fixed_guaranteed_wrong", "fraction": 0.8, "per_true_class": 400,
              "selection_stream": 3, "offset_stream": 4}
FROZEN_ORDINARY = {
    "experiments/spectral_general_augmentation.py":
        "abffd85aa30f350cc2a6c857758a315d878c27de170c1a4a2adee2f39101538e",
    "experiments/spectral_general_augmentation_data.py":
        "4404d59fe79e294abc946cd3a58963d1b6e84d65e47d29c33bcebbde7003396a",
    "experiments/spectral_general_augmentation_audit.py":
        "9f2c38c48a9d0dc14c733930a1022fe094caf623cb10a524bf1a262612d2cb15",
}


def branch_roster():
    rows = []
    for si, seed in enumerate(SEEDS):
        for ai, augmentation in enumerate(AUGMENTATIONS):
            offset = (si + ai) % 2
            for policy in POLICIES[offset:] + POLICIES[:offset]:
                rows.append({"seed": seed, "policy": policy, "augmentation": augmentation})
    return rows


def source_pins():
    paths = [ROOT / "spectral_filter.py", Path(core.__file__), Path(__file__), Path(data.__file__),
             *(ROOT / name for name in FROZEN_ORDINARY),
             DOCS / "protocol.md", DOCS / "implementation-check.md", DOCS / "visual-review.md",
             ROOT / "experiments/spectral_wrong_label_augmentation_audit.py",
             ROOT / "tests/test_spectral_wrong_label_augmentation.py",
             ROOT / "tests/test_spectral_wrong_label_augmentation_data.py",
             ROOT / "tests/test_spectral_wrong_label_augmentation_audit.py"]
    pins = {str(path.resolve().relative_to(ROOT)): digest(path) for path in paths}
    need(pins["spectral_filter.py"] == core.FILTER_SHA256, "canonical filter changed")
    need(pins[str(Path(core.__file__).resolve().relative_to(ROOT))] == ordinary.CORE_SHA,
         "accepted I9 core changed")
    need(all(pins[name] == sha for name, sha in FROZEN_ORDINARY.items()), "frozen ordinary dependency changed")
    return pins


def evaluation_row(step, train_logits, heldout_logits, plan):
    """All readouts use original, unaugmented images and explicit label sets.

    train_clean/train_assigned cover all 5,000 training IDs. wrong_true and
    wrong_target cover the identical 4,000 corrupted IDs with different labels.
    The independent auditor reconstructs these from the saved full logits.
    """
    mask = plan["corruption_mask"]
    need(isinstance(mask, np.ndarray) and mask.dtype == np.bool_ and mask.ndim == 1
         and mask.shape == (len(train_logits),) and bool(mask.any()), "corruption mask schema")
    stats = ordinary.classification_stats
    return {"step": step,
            "train_clean": stats(train_logits, plan["train_labels"]),
            "train_assigned": stats(train_logits, plan["assigned_labels"]),
            "wrong_true": stats(train_logits[mask], plan["train_labels"][mask]),
            "wrong_target": stats(train_logits[mask], plan["assigned_labels"][mask]),
            "heldout": stats(heldout_logits, plan["eval_labels"])}


def update_occurrences(model, optimizer, tracker, train_images, assigned_labels,
                       batch, shifts, policy, augmentation, device):
    """Use each source example's fixed assigned label for every transformed view."""
    need(augmentation in AUGMENTATIONS, "unknown augmentation")
    need(isinstance(batch, np.ndarray) and batch.dtype == np.int64 and batch.ndim == 1
         and bool(np.all((batch >= 0) & (batch < len(train_images)))), "batch schema")
    need(isinstance(assigned_labels, np.ndarray) and assigned_labels.dtype == np.int64
         and assigned_labels.shape == (len(train_images),)
         and bool(np.all((assigned_labels >= 0) & (assigned_labels < 10))), "assigned label schema")
    values = train_images[batch]
    need(values.dtype == np.float32 and values.shape == (len(batch), 784)
         and bool(np.isfinite(values).all()) and bool(np.all((values >= 0) & (values <= 1))), "input schema")
    augmentation_seconds = 0.
    if augmentation == "translate":
        started = time.monotonic()
        values = data.translate(values, shifts)
        augmentation_seconds = time.monotonic() - started
    x = torch.from_numpy(values).to(device)
    targets = torch.as_tensor(assigned_labels[batch], device=device)
    loss, diagnostic = ordinary.training_update(model, optimizer, tracker, x, targets, policy)
    return loss, diagnostic, augmentation_seconds


def acquire(run):
    images, labels = ordinary.read_training()
    branches, identities, warmups = [], {}, {}
    roster = branch_roster()
    for seed in SEEDS:
        plan = data.make_plan(labels, seed)
        run.save(f"plan-s{seed}.npz", plan, "npz")
        run.save(f"plan-s{seed}.json", {"seed": seed, "corruption": CORRUPTION,
                 "array_hashes": {key: array_digest(value) for key, value in plan.items()},
                 "numpy_version": np.__version__})
        cpu = images[plan["train_ids"]].astype(np.float32) / np.float32(255)
        train_x = torch.from_numpy(cpu).to(run.device)
        eval_x = torch.from_numpy(images[plan["eval_ids"]].astype(np.float32) / np.float32(255)).to(run.device)
        for row in (r for r in roster if r["seed"] == seed):
            run.check()
            name = f"s{seed}-{row['policy']}-{row['augmentation']}"
            ordinary.sync(run.device)
            started = time.monotonic()
            model = core.make_model(seed, run.device)
            optimizer = core.make_optimizer(model)
            tracker = core.make_tracker(model, optimizer) if row["policy"] == "native32" else None
            need(sum(p.numel() for p in model.parameters()) == 50890, "model parameter count")
            model_sha = core.tree_digest(dict(model.state_dict()))
            if seed not in identities:
                identities[seed] = model_sha
                run.save(f"initial-s{seed}.pt", core.snapshot(model, optimizer, tracker), "tensor")
            need(model_sha == identities[seed] and len(optimizer.state) == 0, "fresh paired initialization required")
            metrics, train_logits, heldout_logits = [], [], []
            loss_history = np.empty(STEPS, dtype=np.float64)
            training_seconds = augmentation_seconds = evaluation_seconds = 0.
            warmup_sha, warmup_receipt = None, None

            def evaluate(step):
                nonlocal evaluation_seconds
                ordinary.sync(run.device)
                before = time.monotonic()
                tr, ev = ordinary.predict(model, train_x), ordinary.predict(model, eval_x)
                metrics.append(evaluation_row(step, tr, ev, plan))
                train_logits.append(tr)
                heldout_logits.append(ev)
                ordinary.sync(run.device)
                evaluation_seconds += time.monotonic() - before

            evaluate(0)
            for step, batch in enumerate(plan["occurrences"], 1):
                run.check()
                ordinary.sync(run.device)
                before = time.monotonic()
                loss, diagnostic, aug_seconds = update_occurrences(
                    model, optimizer, tracker, cpu, plan["assigned_labels"], batch,
                    plan["shifts"][step - 1], row["policy"], row["augmentation"], run.device)
                loss_history[step - 1] = loss
                augmentation_seconds += aug_seconds
                ordinary.sync(run.device)
                training_seconds += time.monotonic() - before
                if step == 100:
                    warmup_sha = ordinary.learning_digest(model, optimizer)
                    key = (seed, row["augmentation"])
                    if key in warmups:
                        need(warmup_sha == warmups[key], "within-condition warmup learning state differs")
                    warmups[key] = warmup_sha
                    warmup_receipt = run.save("warmup-" + name + ".pt", core.snapshot(model, optimizer, tracker), "tensor")
                if step in EVAL_STEPS:
                    evaluate(step)
            need(all(bool(torch.isfinite(p).all()) for p in model.parameters()), "nonfinite final parameters")
            need(np.isfinite(loss_history).all() and len(metrics) == len(EVAL_STEPS), "complete finite stream required")
            final_receipt = run.save("final-" + name + ".pt", core.snapshot(model, optimizer, tracker), "tensor")
            logits_receipt = run.save("logits-" + name + ".npz", {
                "steps": np.array(EVAL_STEPS, dtype=np.int64),
                "train": np.stack(train_logits), "heldout": np.stack(heldout_logits)}, "npz")
            stream_receipt = run.save("stream-" + name + ".npz", {"loss": loss_history}, "npz")
            rank_summary = None if tracker is None else {
                "basis_rank": 0 if tracker.S is None else int(tracker.S.numel()),
                "stabilization_count": tracker.stabilization_count,
                "max_orthogonality_error": tracker.max_orthogonality_error,
                "step_count": tracker.step_count}
            branch = {**row, "name": name, "initial_model_sha256": model_sha,
                "warmup_learning_sha256": warmup_sha, "warmup_receipt": warmup_receipt,
                "final_receipt": final_receipt, "logits_receipt": logits_receipt, "stream_receipt": stream_receipt,
                "metrics": metrics, "training_seconds": training_seconds, "augmentation_seconds": augmentation_seconds,
                "evaluation_seconds": evaluation_seconds, "wall_seconds": time.monotonic() - started,
                "rank_summary": rank_summary}
            branches.append(branch)
            print(json.dumps({"completed": name, "of": len(roster), "count": len(branches),
                              "heldout": metrics[-1]["heldout"], "seconds": branch["wall_seconds"]}), flush=True)
            del model, optimizer, tracker
    need([{key: row[key] for key in ("seed", "policy", "augmentation")} for row in branches] == roster,
         "complete ordered branch roster required")
    return branches


def configure():
    """Admission for this new service; shared bound/client validators stay frozen."""
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        need(os.environ.get(name) == "1", "set " + name + "=1 before Python")
    need(os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8", "deterministic workspace required")
    group = next(line.split(":", 2)[2] for line in Path("/proc/self/cgroup").read_text().splitlines()
                 if line.startswith("0::"))
    need(Path(group).name == UNIT, "unexpected acquisition unit")
    root = Path("/sys/fs/cgroup") / group.lstrip("/")
    effective = {key: (root / key).read_text().strip() for key in ("memory.max", "memory.swap.max", "cpu.max")}
    props = subprocess.check_output(["systemctl", "--user", "show", UNIT, "--property=Type",
        "--property=RuntimeMaxUSec", "--property=Restart", "--property=KillMode"], text=True)
    service = dict(line.split("=", 1) for line in props.splitlines())
    validate_bounds(effective, service)
    clients = validate_gpu_clients(subprocess.check_output(["nvidia-smi",
        "--query-compute-apps=pid,used_gpu_memory", "--format=csv,noheader,nounits"], text=True))
    need(torch.__version__ == "2.11.0+cu128" and np.__version__ == "1.26.4", "unexpected framework version")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    need(torch.cuda.is_available() and torch.cuda.get_device_name() == "NVIDIA GeForce RTX 3090", "local RTX3090 required")
    free, total = torch.cuda.mem_get_info()
    need(free >= 8 * 1024**3, "8 GiB free GPU required")
    torch.cuda.set_per_process_memory_fraction(8 * 1024**3 / total)
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    return {"cgroup": group, "effective": effective, "service": service, "preexisting_gpu_clients": clients}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    need(args.execute, "No acquisition without explicit --execute")
    parent = args.output_dir.parent.resolve(strict=True)
    need(parent.is_relative_to(Path("/tmp/spectral-experiment-artifacts")) and not list(parent.iterdir()), "unused large-volume parent required")
    need(args.output_dir == parent / "acquisition-001" and not args.output_dir.exists()
         and not args.output_dir.is_symlink(), "exclusive acquisition-001 required")
    mount = subprocess.check_output(["findmnt", "-n", "-o", "TARGET,SOURCE", "--target", str(parent)], text=True).split()
    need(mount == ["/private-artifacts/storage", "/dev/RECONFIGURE_FOR_LOCAL_STORAGE"], "wrong output mount identity")
    need(shutil.disk_usage(parent).free > MAX_BYTES + 1024**3, "cap plus free-disk reserve required")
    pins, inventory = source_pins(), byte_inventory()
    commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    for path, sha in pins.items():
        blob = subprocess.check_output(["git", "-C", str(ROOT), "show", commit + ":" + path])
        need(hashlib.sha256(blob).hexdigest() == sha, "uncommitted source: " + path)
    need("PASS" in (DOCS / "visual-review.md").read_text(), "inherited visual review not passed")
    attempt = {"schema": SCHEMA, "commit": commit, "source_pins": pins, "data_pins": DATA_PINS,
               "corruption": CORRUPTION, "output_dir": str(args.output_dir), "unit": UNIT,
               "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    with (DOCS / "attempt.json").open("x") as handle:
        json.dump(attempt, handle, indent=2)
        handle.write("\n")
    args.output_dir.mkdir()
    run = Run(args.output_dir)
    try:
        guards = configure()
        run.save("provenance.json", {**attempt, "guards": guards, "inventory": inventory,
            "python": sys.version, "torch": torch.__version__, "numpy": np.__version__})
        branches = acquire(run)
        need(source_pins() == pins and {name: digest(DATA / name) for name in DATA_PINS} == DATA_PINS,
             "source or data changed during acquisition")
        result = {"schema": SCHEMA, "status": "complete", "source_pins": pins, "data_pins": DATA_PINS,
                  "corruption": CORRUPTION, "roster": branch_roster(), "eval_steps": EVAL_STEPS, "branches": branches,
                  "seconds": time.monotonic() - run.started, "receipts": list(run.receipts),
                  "gpu_max_allocated_bytes": torch.cuda.max_memory_allocated(),
                  "host_high_water_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
        receipt = run.save("results.json", result)
        print(json.dumps({"status": "complete", "result": receipt, "bytes": run.used}), flush=True)
    except BaseException as exc:
        failure = {"status": "failed", "error_type": type(exc).__name__, "message": str(exc)[:2000],
                   "seconds": time.monotonic() - run.started}
        with (args.output_dir / "failure.json").open("x") as handle:
            json.dump(failure, handle)
        raise


if __name__ == "__main__":
    main()
