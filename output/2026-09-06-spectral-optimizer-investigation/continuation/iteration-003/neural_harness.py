#!/usr/bin/env python3
"""Prospective MNIST diagnostic. Defaults to no execution; see protocol.md."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import struct
import subprocess
import sys
import time
from datetime import datetime, timezone

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO))
from spectral_filter import SpectralGradientFilter

DATA = Path("data/MNIST/raw")
ARMS = {"adamw": None, "estimate32_project32": 32, "estimate128_project32": 128}
LR, WD, WARMUP, PROJECTION_RANK = .001, .01, 100, 32
SEEDS, STEPS, PILOT_SEED, PILOT_STEPS = (0, 1, 2), 2000, 9876, 220
ANALYSIS_FILES = ("summarize_results.py", "test_summary.py", "analysis-plan.md")
TRAINING_FILES = ("train-images-idx3-ubyte", "train-labels-idx1-ubyte")


def utc():
    return datetime.now(timezone.utc).isoformat()


def digest_bytes(value):
    return hashlib.sha256(value).hexdigest()


def file_hash(path):
    return digest_bytes(Path(path).read_bytes())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def artifact(path):
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": file_hash(path)}


def configure():
    if os.environ["CUBLAS_WORKSPACE_CONFIG"] != ":4096:8":
        raise RuntimeError("CUBLAS_WORKSPACE_CONFIG must match frozen :4096:8")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def occupancy():
    commands = [
        ["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu", "--format=csv"],
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv"],
    ]
    return [subprocess.check_output(cmd, text=True).strip() for cmd in commands]


def manifest(mode):
    return {
        "mode": mode, "started_utc": utc(), "status": "running",
        "repository_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "source_sha256": file_hash(__file__), "protocol_sha256": file_hash(HERE / "protocol.md"),
        "tests_sha256": file_hash(HERE / "test_harness.py"),
        "canonical_source_sha256": file_hash(REPO / "spectral_filter.py"),
        "analysis_sources_sha256": {name: file_hash(HERE / name) for name in ANALYSIS_FILES},
        "python": sys.version, "numpy": np.__version__, "torch": torch.__version__,
        "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
        "occupancy_before": occupancy(), "precision": "float32 model/basis; float64 scalar reductions and small eigensolve",
        "torch_threads": torch.get_num_threads(), "cublas_workspace": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "tf32": False, "cudnn_benchmark": False,
    }


def rng(seed, stream):
    return np.random.default_rng(np.random.SeedSequence([20260906, 3, seed, stream]))


def make_plan(seed, steps):
    perm = rng(seed, 0).permutation(60000)
    return {
        "seed": seed, "initialization_seed": int(rng(seed, 3).integers(0, 2**32, dtype=np.uint32)),
        "train_indices": perm[:5000], "validation_indices": perm[5000:10000],
        "auxiliary_indices": perm[10000:15000],
        "replacement_uniforms": rng(seed, 1).random(5000),
        "replacement_digits": rng(seed, 2).integers(0, 10, size=5000),
        "training_batches": rng(seed, 4).integers(0, 5000, size=(steps, 64)),
        "primary_probe_batches": rng(seed, 5).integers(0, 5000, size=(steps, 256)),
        "auxiliary_probe_batches": rng(seed, 6).integers(0, 5000, size=(steps, 256)),
    }


def read_idx(path, images):
    raw = path.read_bytes()
    if images:
        magic, count, rows, cols = struct.unpack(">IIII", raw[:16])
        assert magic == 2051 and rows == cols == 28 and len(raw) == 16 + count * 784
        return torch.from_numpy(np.frombuffer(raw, dtype=np.uint8, offset=16).copy().reshape(count, 784)).float().div_(255)
    magic, count = struct.unpack(">II", raw[:8])
    assert magic == 2049 and len(raw) == 8 + count
    return torch.from_numpy(np.frombuffer(raw, dtype=np.uint8, offset=8).copy()).long()


def load_training():
    x = read_idx(DATA / "train-images-idx3-ubyte", True)
    y = read_idx(DATA / "train-labels-idx1-ubyte", False)
    assert len(x) == len(y) == 60000
    return x, y


def load_test_after_training():
    x = read_idx(DATA / "t10k-images-idx3-ubyte", True)
    y = read_idx(DATA / "t10k-labels-idx1-ubyte", False)
    assert len(x) == len(y) == 10000
    return x, y


def dataset_for_plan(x, y, plan, noise, device):
    index = torch.from_numpy(plan["train_indices"])
    clean = y[index]
    mask = torch.from_numpy(plan["replacement_uniforms"] < noise)
    noisy = torch.where(mask, torch.from_numpy(plan["replacement_digits"]), clean)
    val = torch.from_numpy(plan["validation_indices"])
    aux = torch.from_numpy(plan["auxiliary_indices"])
    return {
        "x": x[index].to(device), "clean": clean.to(device), "noisy": noisy.to(device),
        "vx": x[val].to(device), "vy": y[val].to(device),
        "ax": x[aux].to(device), "ay": y[aux].to(device),
        "realized_replacement_fraction": float(mask.float().mean()),
        "realized_incorrect_fraction": float((noisy != clean).float().mean()),
    }


def make_model(seed, device):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = nn.Sequential(nn.Linear(784, 64), nn.ReLU(), nn.Linear(64, 10)).to(device)
    assert sum(p.numel() for p in model.parameters()) == 50890
    return model


def make_optimizer(model):
    return torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD,
                             betas=(.9, .999), eps=1e-8, foreach=False, fused=False)


def make_tracker(model, optimizer, width):
    if width is None:
        return None
    return SpectralGradientFilter(model, optimizer, rank=width, decay=.99,
                                  warmup=WARMUP, stable_update=True, stabilize_every=100,
                                  relative_eig_tol=1e-8, absolute_eig_floor=0,
                                  weighting="hard", normalize="none", adaptive="none")


def flat_params(model):
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()])


def flat_grad(model):
    params = list(model.parameters())
    assert all(p.grad is not None for p in params), "Missing parameter gradient"
    value = torch.cat([p.grad.detach().reshape(-1) for p in params])
    assert bool(torch.isfinite(value).all()), "Non-finite training gradient"
    return value


def set_grad(model, value):
    pos = 0
    for p in model.parameters():
        p.grad = value[pos:pos + p.numel()].reshape_as(p)
        pos += p.numel()
    assert pos == value.numel()


def project(value, basis):
    return value.clone() if basis is None else basis @ (basis.T @ value)


def observe_and_project(tracker, raw):
    if tracker is None:
        return raw.clone(), None
    tracker.step_count += 1
    tracker._update_svd(raw)
    if tracker.step_count <= WARMUP or tracker.V is None:
        return raw.clone(), None
    basis = tracker.V[:, :min(PROJECTION_RANK, tracker.V.shape[1])]
    assert basis.shape[1] <= PROJECTION_RANK and tracker.V.shape[1] <= tracker.rank
    return project(raw, basis), basis


def clone_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, dict):
        return {k: clone_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clone_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(clone_tree(v) for v in value)
    return copy.deepcopy(value)


def equal_tree(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(equal_tree(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(equal_tree(x, y) for x, y in zip(left, right))
    return left == right


def tracker_state(tracker):
    if tracker is None:
        return None
    # Exclude aliases to the model/optimizer only; capture all estimator state.
    return {k: clone_tree(v) for k, v in vars(tracker).items()
            if k not in ("model", "base_optimizer", "param_list")}


def snapshot(model, optimizer, tracker):
    return {
        "parameters": clone_tree(list(model.parameters())),
        "gradients": [None if p.grad is None else p.grad.detach().clone() for p in model.parameters()],
        "optimizer": clone_tree(optimizer.state_dict()), "tracker": tracker_state(tracker),
        "model_modes": [m.training for m in model.modules()],
        "torch_rng": torch.get_rng_state().clone(),
        "cuda_rng": [v.clone() for v in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else [],
        "python_rng": random.getstate(),
    }


def gradient(model, x, y):
    loss = F.cross_entropy(model(x), y)
    values = torch.autograd.grad(loss, tuple(model.parameters()), create_graph=False,
                                 retain_graph=False, allow_unused=False)
    result = torch.cat([g.detach().reshape(-1) for g in values])
    assert bool(torch.isfinite(result).all()), "Non-finite probe gradient"
    return result


def independent_probes(model, optimizer, tracker, data, primary, auxiliary):
    before = snapshot(model, optimizer, tracker)
    clean = gradient(model, data["x"][primary], data["clean"][primary])
    noisy = gradient(model, data["x"][primary], data["noisy"][primary])
    heldout = gradient(model, data["ax"][auxiliary], data["ay"][auxiliary])
    after = snapshot(model, optimizer, tracker)
    assert equal_tree(before, after), "Probes changed training state"
    return {"clean": clean, "noisy": noisy, "corruption_residual": noisy - clean, "auxiliary_clean": heldout}


def dot(x, y):
    return float(torch.dot(x.double(), y.double()).item())


def norm(x):
    return dot(x, x) ** .5


def sign_class(value, x, y):
    tolerance = 1e-6 * norm(x) * norm(y) + 1e-14
    return {"value": value, "tolerance": tolerance,
            "sign": 1 if value > tolerance else (-1 if value < -tolerance else 0)}


def ratio(numerator, denominator):
    return None if denominator == 0 else numerator / denominator


def retention(value, basis):
    projected = project(value, basis)
    return {"squared_norm": dot(value, value), "projected_squared_norm": dot(projected, projected),
            "squared_norm_retention": ratio(dot(projected, projected), dot(value, value))}


def joint_probe_geometry(probes, basis):
    c, r, n = probes["clean"], probes["corruption_residual"], probes["noisy"]
    pc, pr, pn = project(c, basis), project(r, basis), project(n, basis)
    return {
        "clean_dot_corruption": dot(c, r), "projected_clean_dot_corruption": dot(pc, pr),
        "noisy_energy": dot(n, n), "projected_noisy_energy": dot(pn, pn),
        "energy_closure_residual": dot(n, n) - dot(c, c) - dot(r, r) - 2 * dot(c, r),
        "projected_energy_closure_residual": dot(pn, pn) - dot(pc, pc) - dot(pr, pr) - 2 * dot(pc, pr),
    }


def displacement_metrics(raw, applied, displacement, basis):
    inside = project(displacement, basis)
    outside = displacement - inside
    residual = raw - applied
    total_term, inside_term, outside_term = dot(raw, displacement), dot(applied, inside), dot(residual, outside)
    scale = norm(raw) * norm(displacement)
    closure = total_term - inside_term - outside_term
    relative_closure = abs(closure) / max(scale, 1e-30)
    assert relative_closure <= 1e-4, f"Projection identity closure failure {relative_closure}"
    signed_total = sign_class(total_term, raw, displacement)
    signed_inside = sign_class(inside_term, applied, inside)
    signed_outside = sign_class(outside_term, residual, outside)
    robust_reversal = (total_term > signed_total["tolerance"] + abs(closure)
                       and inside_term < -signed_inside["tolerance"] - abs(closure)
                       and outside_term > signed_outside["tolerance"] + abs(closure))
    return {
        "squared_norm": dot(displacement, displacement),
        "raw_gradient_dot_update": signed_total,
        "cosine_raw_gradient_update": ratio(total_term, scale),
        "leakage_squared_fraction": ratio(dot(outside, outside), dot(displacement, displacement)),
        "in_subspace_contribution": signed_inside,
        "outside_contribution": signed_outside,
        "identity_closure_residual": closure, "identity_relative_closure": relative_closure,
        "closure_robust_leakage_reversal": robust_reversal,
        "reversal_unresolved_by_closure": (signed_total["sign"] == 1 and signed_inside["sign"] == -1 and not robust_reversal),
    }


def decay_subtracted_update(displacement, theta):
    return displacement + LR * WD * theta


def is_probe_step(step):
    return step in (1, 101) or step % 50 == 0


def choose_checkpoint(best_loss, candidate_loss):
    # Strict improvement preserves the earliest checkpoint on an exact tie.
    return candidate_loss < best_loss


@torch.no_grad()
def evaluate(model, x, y):
    loss, correct = 0., 0
    for start in range(0, len(x), 512):
        logits = model(x[start:start + 512])
        loss += float(F.cross_entropy(logits, y[start:start + 512], reduction="sum"))
        correct += int((logits.argmax(-1) == y[start:start + 512]).sum())
    return {"cross_entropy": loss / len(x), "accuracy": correct / len(x), "count": len(x)}


def tensor_hash(value):
    return digest_bytes(value.detach().cpu().contiguous().numpy().tobytes())


def synchronize(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


def train_arm(plan, data, arm, steps, instrumented, pilot, device="cuda", active_context=None):
    model = make_model(plan["initialization_seed"], device)
    optimizer = make_optimizer(model)
    tracker = make_tracker(model, optimizer, ARMS[arm])
    rows, probes, validations, trajectory_hashes, step_seconds = [], [], [], [], []
    if active_context is None:
        active_context = {}
    active_context.update({"seed": plan["seed"], "arm": arm, "step": 0,
                           "instrumented": instrumented, "pilot": pilot,
                           "phase": "initialization", "steps_raw": rows,
                           "probes_raw": probes, "validation_trajectory": validations})
    training_batches = torch.as_tensor(plan["training_batches"], device=device)
    primary_batches = torch.as_tensor(plan["primary_probe_batches"], device=device)
    auxiliary_batches = torch.as_tensor(plan["auxiliary_probe_batches"], device=device)
    best_loss, best_step, best_state = float("inf"), 0, None
    if not pilot:
        initial = evaluate(model, data["vx"], data["vy"])
        validations.append({"step": 0, **initial})
        best_loss, best_state = initial["cross_entropy"], clone_tree(model.state_dict())
    if str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    synchronize(device)
    start = time.perf_counter()
    ranks = []
    for step in range(1, steps + 1):
        active_context.update({"step": step, "phase": "training_gradient_and_covariance"})
        synchronize(device)
        step_start = time.perf_counter()
        batch = training_batches[step - 1]
        x, y = data["x"][batch], data["noisy"][batch]
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        raw = flat_grad(model)
        theta = flat_params(model)
        applied, basis = observe_and_project(tracker, raw)
        set_grad(model, applied)
        selected_probes = None
        if instrumented and is_probe_step(step):
            active_context["phase"] = "independent_probes"
            selected_probes = independent_probes(model, optimizer, tracker, data,
                                                 primary_batches[step - 1], auxiliary_batches[step - 1])
        active_context["phase"] = "optimizer_update"
        optimizer.step()
        after = flat_params(model)
        assert bool(torch.isfinite(after).all()), "Non-finite parameters"
        current_rank = 0 if tracker is None or tracker.V is None else tracker.V.shape[1]
        ranks.append(current_rank)
        if tracker is not None and is_probe_step(step):
            assert tracker.orthogonality_error() <= 5e-3, "Nonorthogonal covariance basis"
        if instrumented:
            active_context["phase"] = "update_measurements"
            delta = after - theta
            data_delta = decay_subtracted_update(delta, theta)
            row = {
                "step": step, "filtering_active": basis is not None,
                "estimation_rank": current_rank, "applied_rank": len(raw) if basis is None else basis.shape[1],
                "raw_squared_norm": dot(raw, raw), "applied_squared_norm": dot(applied, applied),
                "total": displacement_metrics(raw, applied, delta, basis),
                "decay_subtracted": displacement_metrics(raw, applied, data_delta, basis),
            }
            if not pilot:
                rows.append(row)
            if selected_probes is not None:
                probe_row = {"step": step, "retention": {k: retention(v, basis) for k, v in selected_probes.items()}}
                probe_row["joint_geometry"] = joint_probe_geometry(selected_probes, basis)
                clean_fraction = probe_row["retention"]["clean"]["squared_norm_retention"]
                corruption_fraction = probe_row["retention"]["corruption_residual"]["squared_norm_retention"]
                probe_row["clean_minus_corruption_retention"] = (None if clean_fraction is None or corruption_fraction is None
                                                                else clean_fraction - corruption_fraction)
                probe_row["gradient_dot_updates"] = {
                    k: {"total": sign_class(dot(v, delta), v, delta),
                        "decay_subtracted": sign_class(dot(v, data_delta), v, data_delta)}
                    for k, v in selected_probes.items()
                }
                with torch.no_grad():
                    after_loss = float(F.cross_entropy(model(x), y))
                before_loss = float(loss.detach())
                loss_change = after_loss - before_loss
                loss_tolerance = 1e-6 * max(1., abs(before_loss))
                probe_row["same_training_batch_loss"] = {
                    "before": before_loss, "after": after_loss, "change": loss_change,
                    "tolerance": loss_tolerance,
                    "sign": 1 if loss_change > loss_tolerance else (-1 if loss_change < -loss_tolerance else 0),
                }
                if not pilot:
                    probes.append(probe_row)
        if pilot:
            trajectory_hashes.append(tensor_hash(after))
        if not pilot and step % 100 == 0:
            active_context["phase"] = "validation_evaluation"
            validation = evaluate(model, data["vx"], data["vy"])
            validations.append({"step": step, **validation})
            if choose_checkpoint(best_loss, validation["cross_entropy"]):
                best_loss, best_step = validation["cross_entropy"], step
                best_state = clone_tree(model.state_dict())
        synchronize(device)
        step_seconds.append(time.perf_counter() - step_start)
        if step % 50 == 0:
            print(json.dumps({"arm": arm, "step": step, "pilot": pilot, "instrumented": instrumented,
                              "estimation_rank": current_rank, "elapsed_seconds": time.perf_counter() - start}), flush=True)
    synchronize(device)
    elapsed = time.perf_counter() - start
    state = snapshot(model, optimizer, tracker)
    active_context["phase"] = "final_state_and_evaluation"
    result = {
        "arm": arm, "seed": plan["seed"], "steps": steps, "instrumented": instrumented,
        "elapsed_seconds": elapsed, "step_elapsed_seconds": step_seconds,
        "steady_steps129_to220_mean_seconds": float(np.mean(step_seconds[128:220])),
        "steady_steps129_to220_median_seconds": float(np.median(step_seconds[128:220])),
        "scheduled_repair_step200_seconds": step_seconds[199] if len(step_seconds) >= 200 else None,
        "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated() if str(device).startswith("cuda") else 0,
        "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved() if str(device).startswith("cuda") else 0,
        "estimation_rank_by_step": ranks, "trajectory_parameter_sha256": trajectory_hashes,
        "probe_state_checks": sum(is_probe_step(s) for s in range(1, steps + 1)) if instrumented else 0,
        "all_invariant_gates_passed": True,
    }
    if pilot and ARMS[arm] == 128:
        assert ranks[159] == 128, "Wide pilot did not reach steady width128 by step160"
    if not pilot:
        result.update({"steps_raw": rows, "probes_raw": probes, "validation_trajectory": validations,
                       "validation_selected_step": best_step,
                       "final_training_clean": evaluate(model, data["x"], data["clean"]),
                       "final_training_noisy": evaluate(model, data["x"], data["noisy"]),
                       "final_validation": validations[-1],
                       "realized_replacement_fraction": data["realized_replacement_fraction"],
                       "realized_incorrect_fraction": data["realized_incorrect_fraction"]})
    checkpoints = {} if pilot else {"final": clone_tree(model.state_dict()), "validation_selected": best_state}
    return result, state, checkpoints


def finite_average(values):
    valid = [v for v in values if v is not None]
    return {"mean": None if not valid else float(np.mean(valid)),
            "defined_count": len(valid), "undefined_count": len(values) - len(valid)}


def save_plan(path, plan):
    np.savez_compressed(path, **plan)
    return artifact(path)


def preserve_failure(output, record, active_context, exc):
    record.update({"status": "failed_preserved", "failed_utc": utc(), "error": repr(exc)})
    partial_path = output / "partial-failure.json"
    write_json(partial_path, active_context)
    record["failure_context"] = {k: v for k, v in active_context.items()
                                 if k not in ("steps_raw", "probes_raw", "validation_trajectory")}
    record["partial_failure_artifact"] = artifact(partial_path)
    write_json(output / "execution.json", record)


def verify_committed_sources():
    for path in (Path(__file__), HERE / "protocol.md", HERE / "test_harness.py", REPO / "spectral_filter.py",
                 *(HERE / name for name in ANALYSIS_FILES)):
        relative = path.relative_to(REPO).as_posix()
        committed = subprocess.check_output(["git", "show", f"HEAD:{relative}"], cwd=REPO)
        assert digest_bytes(committed) == file_hash(path), f"Uncommitted source mismatch: {relative}"


def run_pilot(output):
    output.mkdir(exist_ok=False)
    record = manifest("development_pilot")
    write_json(output / "execution.json", record)
    active_context = {"phase": "loading_training_data", "replacement_probability": .9}
    try:
        record["training_data_artifacts"] = [artifact(DATA / name) for name in TRAINING_FILES]
        write_json(output / "execution.json", record)
        x, y = load_training()
        plan = make_plan(PILOT_SEED, PILOT_STEPS)
        record["plans"] = [save_plan(output / "development-plan.npz", plan)]
        data = dataset_for_plan(x, y, plan, .9, "cuda")
        reports = []
        for arm in ARMS:
            active_context.update({"seed": PILOT_SEED, "arm": arm, "step": 0,
                                   "instrumented": False, "phase": "model_initialization"})
            bare, bare_state, _ = train_arm(plan, data, arm, PILOT_STEPS, False, True,
                                           active_context=active_context)
            # Store the first final state on CPU to avoid inflating second-run GPU peak.
            def cpu_tree(v):
                if isinstance(v, torch.Tensor):
                    return v.cpu()
                if isinstance(v, dict):
                    return {k: cpu_tree(w) for k, w in v.items()}
                if isinstance(v, list):
                    return [cpu_tree(w) for w in v]
                if isinstance(v, tuple):
                    return tuple(cpu_tree(w) for w in v)
                return v
            bare_state = cpu_tree(bare_state)
            torch.cuda.empty_cache()
            active_context.update({"step": 0, "instrumented": True, "phase": "model_initialization"})
            measured, measured_state, _ = train_arm(plan, data, arm, PILOT_STEPS, True, True,
                                                   active_context=active_context)
            active_context["phase"] = "paired_trajectory_and_final_state_gate"
            assert bare["trajectory_parameter_sha256"] == measured["trajectory_parameter_sha256"], "Instrumented trajectory differs"
            assert equal_tree(bare_state, cpu_tree(measured_state)), "Instrumented final training state differs"
            del measured_state, bare_state
            reports.append({"arm": arm, "trajectory_bitwise_identical": True,
                            "final_state_bitwise_identical": True, "uninstrumented": bare, "instrumented": measured})
            write_json(output / "timing-and-invariants.json", reports)
            print(json.dumps({"arm_completed": arm, "trajectory_bitwise_identical": True}), flush=True)
        record.update({"status": "complete_passed", "completed_utc": utc(), "arms": len(reports),
                       "official_test_loaded": False, "validation_or_accuracy_computed": False})
    except Exception as exc:
        preserve_failure(output, record, active_context, exc)
        raise
    write_json(output / "execution.json", record)


def run_full(output):
    verify_committed_sources()
    pilot_manifest = json.loads((HERE / "pilot" / "execution.json").read_text())
    assert pilot_manifest["status"] == "complete_passed", "Development gates have not passed"
    assert pilot_manifest["source_sha256"] == file_hash(__file__), "Source changed since passing pilot"
    assert pilot_manifest["protocol_sha256"] == file_hash(HERE / "protocol.md"), "Protocol changed since passing pilot"
    assert pilot_manifest["tests_sha256"] == file_hash(HERE / "test_harness.py"), "Tests changed since passing pilot"
    assert pilot_manifest["canonical_source_sha256"] == file_hash(REPO / "spectral_filter.py"), "Canonical source changed since passing pilot"
    assert pilot_manifest["analysis_sources_sha256"] == {name: file_hash(HERE / name) for name in ANALYSIS_FILES}, "Analysis source changed since passing pilot"
    for data_artifact in pilot_manifest["training_data_artifacts"]:
        assert file_hash(data_artifact["path"]) == data_artifact["sha256"], "Training data bytes changed since passing pilot"
    output.mkdir(exist_ok=False)
    record = manifest("confirmatory")
    write_json(output / "execution.json", record)
    active_context = {"phase": "loading_training_data"}
    try:
        x, y = load_training()
        record["data_artifacts"] = [artifact(DATA / "train-images-idx3-ubyte"), artifact(DATA / "train-labels-idx1-ubyte")]
        record["plans"], record["checkpoints"] = [], []
        all_results = []
        for seed in SEEDS:
            plan = make_plan(seed, STEPS)
            record["plans"].append(save_plan(output / f"plan-seed{seed}.npz", plan))
            for noise in (0., .9):
                data = dataset_for_plan(x, y, plan, noise, "cuda")
                for arm in ARMS:
                    tag = f"seed{seed}-noise{noise:g}-{arm}"
                    active_context = {"tag": tag, "seed": seed, "arm": arm, "step": 0,
                                      "phase": "model_initialization", "replacement_probability": noise}
                    result, state, checkpoints = train_arm(plan, data, arm, STEPS, True, False,
                                                           active_context=active_context)
                    result["replacement_probability"] = noise
                    checkpoint_path = output / f"{tag}-checkpoints.pt"
                    torch.save({name: {k: v.cpu() for k, v in ckpt.items()} for name, ckpt in checkpoints.items()}, checkpoint_path)
                    record["checkpoints"].append(artifact(checkpoint_path))
                    result["checkpoint_path"] = str(checkpoint_path)
                    write_json(output / f"{tag}.json", result)
                    all_results.append((tag, result))
                    del state, checkpoints
        # Test is first opened here: all 18 training/validation decisions are over.
        record["all_training_completed_utc"] = utc()
        active_context = {"phase": "loading_official_test_after_all_training"}
        tx, ty = load_test_after_training()
        tx, ty = tx.cuda(), ty.cuda()
        record["test_first_loaded_utc"] = utc()
        record["data_artifacts"] += [artifact(DATA / "t10k-images-idx3-ubyte"), artifact(DATA / "t10k-labels-idx1-ubyte")]
        for tag, result in all_results:
            active_context = {"phase": "test_evaluation", "tag": tag, "seed": result["seed"],
                              "arm": result["arm"], "replacement_probability": result["replacement_probability"]}
            model = make_model(0, "cuda")
            checkpoints = torch.load(result["checkpoint_path"], map_location="cuda", weights_only=True)
            result["test"] = {}
            for name, state in checkpoints.items():
                active_context["checkpoint"] = name
                model.load_state_dict(state)
                result["test"][name] = evaluate(model, tx, ty)
            write_json(output / f"{tag}.json", result)
        record.update({"status": "complete", "completed_utc": utc(), "completed_runs": len(all_results)})
    except Exception as exc:
        preserve_failure(output, record, active_context, exc)
        raise
    write_json(output / "execution.json", record)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--pilot", action="store_true")
    mode.add_argument("--full", action="store_true")
    parser.add_argument("--confirmatory-go", action="store_true", help="Only after separate parent approval and source commit")
    args = parser.parse_args()
    if args.full and not args.confirmatory_go:
        parser.error("Full run requires separate approval and --confirmatory-go")
    configure()
    if not torch.cuda.is_available():
        raise RuntimeError("This protocol requires the available local GPU; no CPU/training fallback")
    if args.pilot:
        run_pilot(HERE / "pilot")
    else:
        run_full(HERE / "results")


if __name__ == "__main__":
    main()
