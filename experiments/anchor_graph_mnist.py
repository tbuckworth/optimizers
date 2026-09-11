#!/usr/bin/env python3
"""Fixed, exclusive neural feasibility test of the frozen anchor-graph action.

Import is inert: only main reads MNIST or runs training. No resume/search mode.
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
import struct
import subprocess
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
I9 = ROOT / "output/2026-09-06-spectral-optimizer-investigation/continuation/iteration-009"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(I9))
import neural_core as core
from experiments import anchor_graph_pilot as graph_core

DOCS = ROOT / "output/2026-09-09-spectral-clustering-mnist"
DATA = Path("data/MNIST/raw")
DATA_FILES = ("train-images-idx3-ubyte", "train-labels-idx1-ubyte")
DATA_SHA256 = {
    "train-images-idx3-ubyte": "ba891046e6505d7aadcbbe25680a0738ad16aec93bde7f9b65e87a2fc25776db",
    "train-labels-idx1-ubyte": "65a50cbbf4e906d70832878ad85ccda5333a97f0f4c3dd2ef09a8a9eef7101c5",
}
SEEDS = (202609091, 202609092, 202609093)
CONDITIONS = ("clean", "fixed_uniform_0p9")
ARMS = ("adamw", "hard32", "cluster32", "mixed32")
STEPS, WARMUP, CADENCE = 2000, 100, 100
RANK, ANCHORS, CLUSTERS, SIGMA = 32, 64, 32, .35
EVAL_STEPS = tuple(range(0, STEPS + 1, 100))
MAX_BYTES, DEADLINE_SECONDS = 2 * 1024**3, 1740
CANONICAL_SHA = "9280c7d360aef4c775baf0d781a5afd56e016e9cfb93ef829c3ae4b8cc3df943"
PILOT_SHA = "2581010abe529d1b3da2af0a71106710dcc59b81ec03e58b93b56f2f4790f89e"


def need(condition, message):
    if not condition:
        raise RuntimeError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def source_pins():
    paths = [ROOT / "spectral_filter.py", Path(graph_core.__file__),
             Path(core.__file__), Path(__file__),
             ROOT / "tests/test_anchor_graph_mnist.py", DOCS / "protocol.md"]
    pins = {str(path.relative_to(ROOT)): digest(path) for path in paths}
    need(pins["spectral_filter.py"] == CANONICAL_SHA, "canonical filter changed")
    need(pins["experiments/anchor_graph_pilot.py"] == PILOT_SHA, "frozen graph changed")
    return pins


def make_plan(seed, count=60000, train_count=5000, heldout_count=5000,
              steps=STEPS, batch_size=64):
    def rng(stream):
        return np.random.default_rng(np.random.SeedSequence([20260909, 31, seed, stream]))
    need(count >= train_count + heldout_count, "insufficient disjoint examples")
    order = rng(0).permutation(count)
    return {"train_indices": order[:train_count],
            "heldout_indices": order[train_count:train_count + heldout_count],
            "initialization_seed": np.array(int(rng(1).integers(0, 2**31))),
            "replacement_mask": rng(2).random(train_count) < .9,
            "replacement_digits": rng(3).integers(0, 10, train_count),
            "training_batches": rng(4).integers(0, train_count, (steps, batch_size))}


def fixed_targets(clean, plan, condition):
    need(condition in CONDITIONS, "unknown condition")
    return (clean.copy() if condition == "clean" else
            np.where(plan["replacement_mask"], plan["replacement_digits"], clean))


def array_digest(value):
    value = np.ascontiguousarray(value)
    metadata = json.dumps([value.dtype.str, list(value.shape)], separators=(",", ":"))
    return hashlib.sha256(metadata.encode() + b"\n" + value.tobytes()).hexdigest()


def covariance_factor(tracker):
    if tracker.V is None:
        return np.empty((tracker.n_params, 0), dtype=np.float64)
    # S stores singular values, not covariance eigenvalues. Do not sqrt again.
    return tracker.V.detach().double().cpu().numpy() * tracker.S.double().numpy()[None, :]


def cluster_diagnostics(labels, previous=None):
    assigned = labels >= 0
    counts = np.bincount(labels[assigned])
    counts = counts[counts > 0]
    isolates = int((~assigned).sum())
    result = {"nonempty_clusters": int(counts.size), "cluster_sizes": counts.tolist(),
              "isolates": isolates, "isolate_fraction": isolates / labels.size,
              "effective_projector_rank": int(counts.size) + isolates,
              "largest_cluster_fraction": float(counts.max() / labels.size) if counts.size else 0.,
              "stability_ari_common_assigned": None, "common_assigned_count": 0,
              "isolate_status_changed_fraction": None}
    if previous is not None:
        common = assigned & (previous >= 0)
        result["common_assigned_count"] = int(common.sum())
        result["isolate_status_changed_fraction"] = float(np.mean(assigned != (previous >= 0)))
        if common.sum() >= 2:
            result["stability_ari_common_assigned"] = graph_core.recovery_metrics(
                previous[common], labels[common])["ari"]
    return result


class ClusterAction:
    def __init__(self, seed):
        self.seed, self.labels, self.refreshes, self.label_history = seed, None, [], []

    def apply(self, tracker, gradient, step, arm, check=lambda: None):
        need(arm in ("cluster32", "mixed32"), "unknown cluster arm")
        if self.labels is None or (step - WARMUP - 1) % CADENCE == 0:
            check()
            started = time.monotonic()
            factor = covariance_factor(tracker)
            graph = graph_core.anchor_graph(factor, ANCHORS, SIGMA, self.seed)
            labels, values, iterations = graph_core.cluster_graph(
                graph, CLUSTERS, self.seed, tol=1e-10, max_iter=30)
            row = {"step": step, **cluster_diagnostics(labels, self.labels),
                   "factor_rank": factor.shape[1], "graph_eigenvalues": values.tolist(),
                   "kmeans_iterations": iterations, "actual_anchors": graph.anchor_indices.size,
                   "labels_sha256": array_digest(labels),
                   "retained_array_bytes": factor.nbytes + graph.state_bytes() + labels.nbytes,
                   "refresh_seconds": time.monotonic() - started}
            self.labels = labels
            self.refreshes.append(row)
            self.label_history.append(labels.copy())
            check()
        raw = gradient.detach().double().cpu().numpy()
        projected = graph_core.cluster_mean_action(raw, self.labels)
        projection_energy = float(projected @ projected)
        raw_energy = float(raw @ raw)
        tolerance = 1e-10 * max(raw_energy, 1e-30)
        need(projection_energy <= raw_energy + tolerance, "cluster projector expanded gradient")
        need(abs(float(raw @ projected) - projection_energy) <= tolerance,
             "cluster projector orthogonality identity violated")
        applied = projected if arm == "cluster32" else .5 * raw + .5 * projected
        return torch.from_numpy(applied).to(gradient)


def observe(tracker, raw):
    tracker.step_count += 1
    tracker._update_svd(raw)


def training_step(model, optimizer, tracker, x, targets, arm, step,
                  cluster=None, check=lambda: None):
    check()
    optimizer.zero_grad(set_to_none=True)
    loss = F.cross_entropy(model(x), targets)
    need(bool(torch.isfinite(loss)), "nonfinite training loss")
    loss.backward()
    raw = core.flat_grad(model)
    if tracker is not None:
        observe(tracker, raw)
        need(tracker.step_count == step, "observer/trajectory step mismatch")
    first_action_binding = {}
    if step == WARMUP + 1:
        first_action_binding["raw_gradient_sha256"] = core.tree_digest(raw.detach().cpu())
        if tracker is not None:
            first_action_binding["post_observe_tracker_sha256"] = core.tree_digest(core._tracker_state(tracker))
    applied = raw
    if step > WARMUP and arm != "adamw":
        if arm == "hard32":
            applied = tracker._project_gradient(raw)
        else:
            applied = cluster.apply(tracker, raw, step, arm, check)
        core.set_grad(model, applied)
    need(bool(torch.isfinite(applied).all()), "nonfinite applied gradient")
    r2 = float(raw.double().square().sum())
    a2 = float(applied.double().square().sum())
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return {"step": step, "training_batch_ce": float(loss.detach()), **first_action_binding,
            "raw_squared_norm": r2, "applied_squared_norm": a2,
            "applied_to_raw_norm_ratio": math.sqrt(a2 / r2) if r2 else None}


def sufficient_statistics(logits, clean, fixed, heldout_logits, heldout):
    train = torch.as_tensor(logits, dtype=torch.float64)
    val = torch.as_tensor(heldout_logits, dtype=torch.float64)
    need(bool(torch.isfinite(train).all() and torch.isfinite(val).all()), "nonfinite eval logits")
    pred, vpred = train.argmax(1).numpy(), val.argmax(1).numpy()
    changed = fixed != clean
    return {"train_count": len(clean), "heldout_count": len(heldout),
            "train_clean_correct": int((pred == clean).sum()),
            "train_fixed_correct": int((pred == fixed).sum()),
            "train_corrupted_count": int(changed.sum()),
            "train_corrupted_correct": int((pred[changed] == fixed[changed]).sum()),
            "heldout_clean_correct": int((vpred == heldout).sum()),
            "train_clean_ce_sum": float(F.cross_entropy(train, torch.as_tensor(clean), reduction="sum")),
            "train_fixed_ce_sum": float(F.cross_entropy(train, torch.as_tensor(fixed), reduction="sum")),
            "heldout_clean_ce_sum": float(F.cross_entropy(val, torch.as_tensor(heldout), reduction="sum"))}


def metrics_from_statistics(stats):
    train, val, corrupted = (stats[k] for k in ("train_count", "heldout_count", "train_corrupted_count"))
    return {"heldout_clean_accuracy": stats["heldout_clean_correct"] / val,
            "heldout_clean_ce": stats["heldout_clean_ce_sum"] / val,
            "train_clean_accuracy": stats["train_clean_correct"] / train,
            "train_fixed_accuracy": stats["train_fixed_correct"] / train,
            "train_corrupted_accuracy": stats["train_corrupted_correct"] / corrupted if corrupted else None,
            "train_clean_ce": stats["train_clean_ce_sum"] / train,
            "train_fixed_ce": stats["train_fixed_ce_sum"] / train}


@torch.no_grad()
def evaluate(model, data, labels, step):
    was_training = model.training
    model.eval()
    train = model(data["x"]).detach().cpu().numpy()
    val = model(data["vx"]).detach().cpu().numpy()
    model.train(was_training)
    stats = sufficient_statistics(train, labels["clean"], labels["fixed"], val, labels["heldout"])
    return {"step": step, "sufficient_statistics": stats,
            "metrics": metrics_from_statistics(stats)}, train, val


class CappedWriter:
    def __init__(self, handle, allowance):
        self.handle, self.allowance = handle, allowance

    def write(self, value):
        need(len(value) <= self.allowance, "output cap exhausted")
        written = self.handle.write(value)
        self.allowance -= written
        return written

    def flush(self):
        self.handle.flush()

    def read(self, *args):
        # NumPy1.26 identifies file-like objects by `read`; ZIP writing never
        # reads this write-only handle. All bytes still pass through write cap.
        return self.handle.read(*args)


class Run:
    def __init__(self, path, device):
        self.path, self.device = path, device
        self.started = time.monotonic()
        self.checked = 0.
        self.receipts, self.used = [], 0

    def check(self):
        now = time.monotonic()
        need(now - self.started < DEADLINE_SECONDS, "cooperative 29-minute deadline exceeded")
        if now - self.checked > 1:
            need(shutil.disk_usage(self.path).free > 1024**3, "less than 1GiB disk reserve")
            if self.device == "cuda":
                need(torch.cuda.max_memory_allocated() < 8 * 1024**3, "GPU allocation cap exceeded")
            self.checked = now

    def save(self, name, value, kind="json"):
        self.check()
        target = self.path / name
        need(target.parent == self.path, "artifact must be a direct child")
        with target.open("xb") as handle:
            capped = CappedWriter(handle, MAX_BYTES - self.used - 1024**2)
            if kind == "tensor":
                torch.save(value, capped)
            elif kind == "npz":
                np.savez(capped, **value)
            else:
                capped.write((json.dumps(value, indent=2, allow_nan=False) + "\n").encode())
        receipt = {"path": name, "size_bytes": target.stat().st_size, "sha256": digest(target)}
        self.used += receipt["size_bytes"]
        self.receipts.append(receipt)
        self.check()
        return receipt


def read_training():
    raw = (DATA / DATA_FILES[0]).read_bytes()
    need(struct.unpack(">IIII", raw[:16]) == (2051, 60000, 28, 28)
         and len(raw) == 16 + 60000 * 784, "unexpected MNIST training image IDX")
    images = np.frombuffer(raw, np.uint8, offset=16).reshape(60000, 784).copy()
    raw = (DATA / DATA_FILES[1]).read_bytes()
    need(struct.unpack(">II", raw[:8]) == (2049, 60000) and len(raw) == 60008,
         "unexpected MNIST training label IDX")
    labels = np.frombuffer(raw, np.uint8, offset=8).astype(np.int64)
    need(bool(((labels >= 0) & (labels < 10)).all()), "invalid MNIST digit")
    return images, labels


def sync(device):
    if device == "cuda":
        torch.cuda.synchronize()


def acquire(run):
    images, all_labels = read_training()
    curves = []
    for seed in SEEDS:
        plan = make_plan(seed)
        clean = all_labels[plan["train_indices"]]
        heldout = all_labels[plan["heldout_indices"]]
        plan_receipt = run.save(f"plan-s{seed}.npz", {**plan, "train_clean_labels": clean,
                                "heldout_clean_labels": heldout}, "npz")
        plan_hashes = {name: array_digest(value) for name, value in plan.items()}
        data = {"x": torch.as_tensor(images[plan["train_indices"]], device=run.device).float() / 255,
                "vx": torch.as_tensor(images[plan["heldout_indices"]], device=run.device).float() / 255}
        for condition in CONDITIONS:
            run.check()
            labels = {"clean": clean, "heldout": heldout, "fixed": fixed_targets(clean, plan, condition)}
            target = torch.as_tensor(labels["fixed"], device=run.device)
            model = core.make_model(int(plan["initialization_seed"]), run.device)
            need(sum(p.numel() for p in model.parameters()) == 50890, "wrong model size")
            optimizer = core.make_optimizer(model)
            tracker = core.make_tracker(model, optimizer)
            initial_hash = core.tree_digest(dict(model.state_dict()))
            initial_eval = evaluate(model, data, labels, 0)
            sync(run.device)
            start = time.monotonic()
            for step in range(1, WARMUP + 1):
                index = torch.as_tensor(plan["training_batches"][step - 1], device=run.device)
                training_step(model, optimizer, tracker, data["x"][index], target[index],
                              "adamw", step, check=run.check)
            sync(run.device)
            warmup_seconds = time.monotonic() - start
            warmup_eval = evaluate(model, data, labels, WARMUP)
            warmup = core.snapshot(model, optimizer, tracker)
            warmup_hash = core.tree_digest(warmup)
            stem = f"s{seed}-{condition}"
            warmup_receipt = run.save(f"warmup-{stem}.pt", warmup, "tensor")
            binding = {"seed": seed, "condition": condition, "plan": plan_receipt,
                       "plan_array_hashes": plan_hashes, "initial_model_sha256": initial_hash,
                       "warmup": warmup_receipt, "warmup_state_sha256": warmup_hash,
                       "fixed_targets_sha256": array_digest(labels["fixed"]),
                       "actually_changed_mask_sha256": array_digest(labels["fixed"] != clean),
                       "selected_replacement_count": int(plan["replacement_mask"].sum()) if condition != "clean" else 0,
                       "actually_changed_count": int((labels["fixed"] != clean).sum()),
                       "warmup_seconds": warmup_seconds}
            run.save(f"binding-{stem}.json", binding)
            del model, optimizer, tracker
            # Rotate a priori to spread order-dependent wall-clock effects.
            offset = (SEEDS.index(seed) + CONDITIONS.index(condition)) % len(ARMS)
            order = ARMS[offset:] + ARMS[:offset]
            first_raw_hash = first_observer_hash = None
            for arm in order:
                run.check()
                model, optimizer, tracker = core.restore(warmup, run.device)
                need(core.tree_digest(core.snapshot(model, optimizer, tracker)) == warmup_hash,
                     "fork model/Adam/observer/RNG differs from common warmup")
                if arm == "adamw":
                    tracker = None  # Actual base-optimizer overhead, not a dummy observer.
                cluster = ClusterAction(seed)
                rows = [initial_eval[0], warmup_eval[0]]
                train_logits = [initial_eval[1], warmup_eval[1]]
                heldout_logits = [initial_eval[2], warmup_eval[2]]
                actions = []
                sync(run.device)
                start = time.monotonic()
                for step in range(WARMUP + 1, STEPS + 1):
                    index = torch.as_tensor(plan["training_batches"][step - 1], device=run.device)
                    actions.append(training_step(model, optimizer, tracker, data["x"][index],
                                                  target[index], arm, step, cluster, run.check))
                    if step == WARMUP + 1:
                        raw_hash = actions[-1]["raw_gradient_sha256"]
                        if first_raw_hash is None:
                            first_raw_hash = raw_hash
                        need(raw_hash == first_raw_hash, "first fork raw gradient differs")
                        if arm != "adamw":
                            observer_hash = actions[-1]["post_observe_tracker_sha256"]
                            if first_observer_hash is None:
                                first_observer_hash = observer_hash
                            need(observer_hash == first_observer_hash, "first post-observe fork differs")
                    if step % 100 == 0:
                        row, train, val = evaluate(model, data, labels, step)
                        rows.append(row)
                        train_logits.append(train)
                        heldout_logits.append(val)
                        print(json.dumps({"seed": seed, "condition": condition, "arm": arm,
                                          "step": step, "metrics": row["metrics"],
                                          "batch_elapsed_seconds": time.monotonic() - run.started}), flush=True)
                sync(run.device)
                branch_seconds = time.monotonic() - start
                final_state = core.snapshot(model, optimizer, tracker)
                identifier = f"{stem}-{arm}"
                final_receipt = run.save(f"final-{identifier}.pt", final_state, "tensor")
                logits_receipt = run.save(f"logits-{identifier}.npz", {
                    "steps": np.array(EVAL_STEPS), "train_logits": np.stack(train_logits),
                    "heldout_logits": np.stack(heldout_logits)}, "npz")
                action_receipt = run.save(f"actions-{identifier}.json", actions)
                cluster_receipt = run.save(f"clusters-{identifier}.json", cluster.refreshes)
                labels_receipt = None
                if cluster.labels is not None:
                    labels_receipt = run.save(f"labels-{identifier}.npz", {
                        "steps": np.array([row["step"] for row in cluster.refreshes]),
                        "labels": np.stack(cluster.label_history)}, "npz")
                record = {**binding, "arm": arm, "arm_order": list(order), "curve": rows,
                          "branch_seconds_including_eval_excluding_save": branch_seconds,
                          "warmup_plus_branch_seconds": warmup_seconds + branch_seconds,
                          "first_action_binding": {key: value for key, value in actions[0].items() if key.endswith("sha256")},
                          "final_state_sha256": core.tree_digest(final_state), "final_state": final_receipt,
                          "logits": logits_receipt, "actions": action_receipt, "clusters": cluster_receipt,
                          "cluster_labels": labels_receipt}
                curve_receipt = run.save(f"curve-{identifier}.json", record)
                curves.append({"seed": seed, "condition": condition, "arm": arm,
                               "curve_receipt": curve_receipt, "endpoint": rows[-1]["metrics"],
                               "warmup_plus_branch_seconds": warmup_seconds + branch_seconds})
                del model, optimizer, tracker, final_state, train_logits, heldout_logits, actions
            del warmup
        del data
    need(len(curves) == 24, "incomplete fixed roster")
    return run.save("results.json", {"schema": "anchor_graph_mnist_results_v1", "rows": curves,
                                    "selection": "fixed_step_2000_no_checkpoint_selection"})


def configure(device):
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        need(os.environ.get(name) == "1", f"set {name}=1 before Python")
    need(os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8", "set cuBLAS determinism before Python")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    need(torch.__version__ == "2.11.0+cu128", "unexpected PyTorch version")
    if device == "cuda":
        need(torch.cuda.is_available() and torch.cuda.get_device_name() == "NVIDIA GeForce RTX 3090",
             "expected local RTX3090")
        free, total = torch.cuda.mem_get_info()
        need(free >= 8 * 1024**3, "less than 8GiB free GPU")
        torch.cuda.set_per_process_memory_fraction(8 * 1024**3 / total)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    parent = args.output_dir.parent.resolve(strict=True)
    need(parent.is_relative_to(Path("/tmp/spectral-experiment-artifacts")), "output must be on large temporary volume")
    mount = subprocess.check_output(["findmnt", "-n", "-o", "TARGET,SOURCE", "--target", str(parent)], text=True).split()
    need(mount == ["/private-artifacts/storage", "/dev/RECONFIGURE_FOR_LOCAL_STORAGE"], "large-volume mount identity differs")
    need(args.output_dir.name == "acquisition-001", "fixed exclusive acquisition name required")
    args.output_dir.mkdir(exist_ok=False)
    run = Run(args.output_dir, args.device)
    try:
        configure(args.device)
        pins = source_pins()
        data_pins = {name: digest(DATA / name) for name in DATA_FILES}
        need(data_pins == DATA_SHA256, "MNIST bytes differ from accepted I9 data")
        manifest = {"schema": "anchor_graph_mnist_manifest_v1", "source_pins": pins,
                    "data_pins": data_pins, "data_directory": str(DATA), "git_commit": subprocess.check_output(
                        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                    "started_unix": time.time(), "pid": os.getpid(),
                    "invocation_id": os.environ.get("INVOCATION_ID"), "device": args.device,
                    "torch": torch.__version__, "numpy": np.__version__, "python": sys.version,
                    "seeds": list(SEEDS), "conditions": list(CONDITIONS), "arms": list(ARMS),
                    "steps": STEPS, "warmup": WARMUP, "eval_steps": list(EVAL_STEPS),
                    "rank": RANK, "anchors": ANCHORS, "clusters": CLUSTERS, "sigma": SIGMA,
                    "refresh_steps": list(range(101, 2001, 100)), "mixed_identity_weight": .5,
                    "max_output_bytes": MAX_BYTES, "cooperative_seconds": DEADLINE_SECONDS,
                    "cloud_spend_usd": 0}
        run.save("manifest.json", manifest)
        results = acquire(run)
        need(source_pins() == pins, "source changed during acquisition")
        need({name: digest(DATA / name) for name in DATA_FILES} == data_pins, "data changed during acquisition")
        run.save("complete.json", {"schema": "anchor_graph_mnist_completion_v1", "status": "complete",
                                  "results": results, "source_pins": pins, "data_pins": data_pins,
                                  "finished_unix": time.time(), "wall_seconds": time.monotonic() - run.started,
                                  "process_high_water_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                                  "gpu_max_allocated_bytes": torch.cuda.max_memory_allocated() if args.device == "cuda" else 0,
                                  "receipts": list(run.receipts), "artifact_bytes_before_completion": run.used})
    except BaseException as exc:
        failure = {"status": "failed", "type": type(exc).__name__, "message": str(exc),
                   "finished_unix": time.time(), "wall_seconds": time.monotonic() - run.started,
                   "completed_receipts": run.receipts}
        # Keep a bounded failure record even after a cooperative deadline.
        with (args.output_dir / "failed.json").open("x") as handle:
            json.dump(failure, handle, indent=2, allow_nan=False)
        raise


if __name__ == "__main__":
    main()
