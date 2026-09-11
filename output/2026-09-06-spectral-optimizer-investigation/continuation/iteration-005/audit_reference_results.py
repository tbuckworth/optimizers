#!/usr/bin/env python3
"""Independent saved-tensor audit; no producer/helper imports, data, or training.

Freeze this file's SHA256 before --run. One attempt, 300 seconds, 8 GiB CUDA
allocated and 12 GiB RSS. Failure is preserved; no retries/tolerance adaptation.
Full-file byte provenance belongs to the separately executed raw-summary audit.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import numpy as np
import torch

BASE = Path(__file__).resolve().parent
SEEDS, STEPS, P, BETA, RANK = (3, 4, 5), (200, 500, 1000, 2000), 50890, .99, 32
PHASE = "pre_adam_after_current_gradient_observation"
PROBES = ("clean", "noisy", "corruption_residual", "auxiliary_clean")
ATOL64, RTOL64, ATOL32 = 1e-11, 1e-9, 1e-6
GIB = 1024 ** 3
RECORD = {}
START = 0.


def utc():
    return datetime.now(timezone.utc).isoformat()


def require(value, message):
    if not bool(value):
        raise AssertionError(f"{RECORD.get('active')}: {message}")


def norm(value):
    return float(torch.linalg.vector_norm(value.double()))


def square(value):
    value = value.double()
    return float((value * value).sum())


def dot(left, right):
    return float((left.double() * right.double()).sum())


def fraction(a, b):
    return None if b == 0 else a / b


def difference(a, b):
    return None if a is None or b is None else a - b


def compare(label, observed, expected, native=False):
    """Compare independently computed values to saved values, including nulls."""
    group = "native_float32" if native else "float64"
    stats = RECORD["comparisons"][group]
    if observed is None or expected is None:
        require(observed is None and expected is None, f"{label}: null mismatch")
        stats["null_checks"] += 1
        return
    a = torch.as_tensor(observed, dtype=torch.float64, device="cpu")
    b = torch.as_tensor(expected, dtype=torch.float64, device="cpu")
    require(a.shape == b.shape, f"{label}: shape mismatch {a.shape}/{b.shape}")
    require(torch.isfinite(a).all() and torch.isfinite(b).all(), f"{label}: nonfinite")
    err = (a - b).abs()
    limit = ATOL32 + b.abs() * 0 if native else ATOL64 + RTOL64 * b.abs()
    maximum = float(err.max()) if err.numel() else 0.
    scaled = float((err / limit).max()) if err.numel() else 0.
    stats["checks"] += 1
    stats["elements"] += a.numel()
    if maximum > stats["max_absolute_error"]:
        stats["max_absolute_error"] = maximum
        stats["max_absolute_error_location"] = label
    if scaled > stats["max_tolerance_fraction"]:
        stats["max_tolerance_fraction"] = scaled
        stats["max_tolerance_fraction_location"] = label
    require(scaled <= 1., f"{label}: max error {maximum}, tolerance fraction {scaled}")


def gate(label, value, limit):
    require(math.isfinite(value) and value <= limit, f"{label}: {value} > {limit}")
    stats = RECORD["residual_gates"].setdefault(label, {"count": 0, "maximum": 0., "limit": limit})
    stats["count"] += 1
    stats["maximum"] = max(stats["maximum"], value)


def finite_tree(value):
    if isinstance(value, torch.Tensor):
        require(torch.isfinite(value).all(), "Nonfinite saved tensor")
    elif isinstance(value, dict):
        for item in value.values():
            finite_tree(item)


def tensor_sha(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def same_float32_bits(left, right):
    require(left.dtype == right.dtype == torch.float32, "Bitwise comparison dtype mismatch")
    return torch.equal(left.view(torch.int32), right.view(torch.int32))


def resources():
    torch.cuda.synchronize()
    result = {"elapsed_seconds": time.perf_counter() - START,
              "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
              "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
              "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved()}
    RECORD["resources"] = result
    require(result["elapsed_seconds"] <= 300., "Audit wall-time budget exceeded")
    require(result["peak_rss_bytes"] <= 12 * GIB, "Audit RSS budget exceeded")
    require(result["peak_gpu_allocated_bytes"] <= 8 * GIB, "Audit GPU allocation budget exceeded")


def artifact(snapshot, role):
    matches = [a for a in snapshot["artifacts"] if a.get("role") == role]
    if role == "state":
        matches = [a for a in snapshot["artifacts"] if a["path"].endswith("-state.pt")]
    elif role == "weights":
        matches = [a for a in snapshot["artifacts"] if a["path"].endswith("-weights.npy")]
    require(len(matches) == 1, f"Expected one {role} artifact")
    item = matches[0]
    path = Path(item["path"]).resolve()
    require(path.is_relative_to(Path(RECORD["bulk_root"]).resolve()), "Bulk path escapes bound root")
    require(path.stat().st_size == item["size_bytes"], "Artifact size changed")
    return item


def array(snapshot, role):
    item = artifact(snapshot, role)
    value = np.load(item["path"], allow_pickle=False)
    require(list(value.shape) == item["shape"] and str(value.dtype) == item["dtype"], "Array metadata mismatch")
    require(np.isfinite(value).all(), "Nonfinite saved array")
    return torch.from_numpy(value)


def native_action(basis, value):
    return value.clone() if basis is None else basis @ (basis.T @ value)


def verify_observer(x, eigenvalues, mapped, valid, ref2, state, snapshot, width):
    name = f"seed{snapshot['seed']}/step{snapshot['step']}/width{width}"
    saved = snapshot["observers"][f"width{width}"]
    obs = state["observers"][str(width)]
    v = None if obs["V"] is None else obs["V"].to("cuda")
    rank = 0 if v is None else v.shape[1]
    require(saved["actual_rank"] == rank <= width and saved["estimation_width"] == width, "Wrong width/rank")
    b32 = None if v is None else v[:, :RANK]
    b64 = None if b32 is None else b32.double()
    previous = state["previous_bases"][str(width)]
    previous = None if previous is None else previous.to("cuda")
    metrics, energies, diagnostics = {}, {}, {}
    probes = {key: state["probes"][key].to("cuda") for key in PROBES}
    require(same_float32_bits(probes["noisy"] - probes["clean"], probes["corruption_residual"]), "Rounded probe residual mismatch")
    projected = {key: native_action(b32, value) for key, value in probes.items()}
    action_errors = []
    for key in PROBES:
        n2, o2 = square(probes[key]), square(projected[key])
        metrics[f"native_{key}_retention"] = fraction(o2, n2)
        compare(f"{name}/input/{key}", n2, saved["energies"]["probes"][key]["input_squared_norm"])
        compare(f"{name}/output/{key}", o2, saved["energies"]["probes"][key]["output_squared_norm"], native=True)
        full = probes[key].double() if b64 is None else b64 @ (b64.T @ probes[key].double())
        action_errors.append(norm(projected[key].double() - full) / max(math.sqrt(n2), 1e-30))
    joint = {}
    for prefix, values in (("", probes), ("projected_", projected)):
        clean, residual, noisy = (values[key] for key in ("clean", "corruption_residual", "noisy"))
        nc, cc, rr, cr = square(noisy), square(clean), square(residual), dot(clean, residual)
        closure = nc - cc - rr - 2 * cr
        relative = abs(closure) / max(abs(nc) + abs(cc) + abs(rr) + 2 * abs(cr), 1e-30)
        gate("probe_energy_closure", relative, 1e-5)
        joint[prefix + "clean_dot_corruption"] = cr
        joint[prefix + "energy_closure_residual"] = closure
        joint[prefix + "energy_closure_relative_error"] = relative
        joint[prefix + "noisy_energy"] = nc
        metric_name = "native_" + prefix + "clean_corruption_cosine"
        metrics[metric_name] = fraction(cr, math.sqrt(cc) * math.sqrt(rr))
    for key, value in joint.items():
        compare(f"{name}/joint/{key}", value, saved["energies"]["joint"][key], native=key.startswith("projected_"))
    metrics["native_clean_minus_corruption_retention"] = difference(metrics["native_clean_retention"], metrics["native_corruption_residual_retention"])
    raw = state["raw"].to("cuda")
    current_out, previous_out = native_action(b32, raw), native_action(previous, raw)
    for basis, output in ((b32, current_out), (previous, previous_out)):
        represented = raw.double() if basis is None else basis.double() @ (basis.double().T @ raw.double())
        action_errors.append(norm(output.double() - represented) / max(norm(raw), 1e-30))
    diagnostics["native_represented_action_error_max"] = max(action_errors)
    gate("native_represented_action", max(action_errors), 1e-5)
    metrics["native_current_gradient_retention"] = fraction(square(current_out), square(raw))
    metrics["native_previous_current_gradient_retention"] = fraction(square(previous_out), square(raw))
    metrics["self_inclusion_retention_increment"] = difference(metrics["native_current_gradient_retention"], metrics["native_previous_current_gradient_retention"])

    ref_energy = float(eigenvalues[:RANK].clamp_min(0).sum())
    estimated2 = inner = estimated_trace = orth = 0.
    if v is not None:
        singular = obs["S"].to(device="cuda", dtype=torch.float64)
        require(singular.numel() == rank and (singular > 0).all(), "Invalid saved singular values")
        v64 = v.double()
        variances = singular.square()
        gram_v = v64.T @ v64
        orth = norm(gram_v - torch.eye(rank, device="cuda", dtype=torch.float64))
        gate("native_full_basis_orthogonality", orth, 5e-3)
        # Factor-product identities independently preserve nonorthogonal V.
        factor = v64 * singular
        estimated2 = square(factor.T @ factor)
        inner = square(factor.T @ x)
        estimated_trace = square(factor)
    raw_error = ref2 + estimated2 - 2 * inner
    cancel = 1e-12 * (ref2 + estimated2 + 2 * abs(inner))
    require(raw_error >= -cancel, "Covariance error cancellation outside frozen gate")
    metrics["relative_covariance_error"] = fraction(math.sqrt(max(0., raw_error)), math.sqrt(ref2))
    metrics["covariance_estimator_trace"] = estimated_trace
    energies.update(reference_squared_frobenius=ref2, estimator_squared_frobenius=estimated2,
                    covariance_inner_product=inner, covariance_squared_error_raw=raw_error)
    diagnostics.update(native_basis_orthogonality_error=orth, cancellation_tolerance=cancel)
    z = x if b64 is None else b64.T @ x
    trace_pc = square(z)
    output_energy = trace_pc if b64 is None else float((z * ((b64.T @ b64) @ z)).sum())
    metrics["trace_P_C"] = trace_pc
    metrics["represented_operator_energy_fraction"] = fraction(output_energy, ref_energy)
    energies["represented_operator_output_energy"] = output_energy
    span_fraction = distance = captured = None
    if rank >= RANK:
        sv = torch.linalg.svdvals(b64)
        condition = float(sv[-1] / sv[0])
        require(condition > 1e-8, "Rank-deficient QR span")
        q, _ = torch.linalg.qr(b64, mode="reduced")
        q_orth = norm(q.T @ q - torch.eye(RANK, device="cuda", dtype=torch.float64))
        gate("qr_span_orthogonality", q_orth, 1e-10)
        diagnostics.update(qr_orthogonality_error=q_orth, basis_singular_value_ratio=condition)
        captured = square(q.T @ x)
        if int((eigenvalues > eigenvalues[0] * 1e-10).sum()) >= RANK and ref_energy > 0:
            span_fraction = captured / ref_energy
            require(-1e-6 <= span_fraction <= 1 + 1e-6, "Invalid span-energy bounds")
        if valid:
            overlap = square(q.T @ mapped[:, :RANK])
            energies["reference_span_overlap_squared"] = overlap
            distance = 1 - overlap / RANK
            require(-1e-6 <= distance <= 1 + 1e-6, "Invalid span-distance bounds")
    metrics.update(span_energy_fraction=span_fraction, span_projector_distance=distance)
    energies["span_captured_energy"] = captured
    require(set(metrics) == set(saved["metrics"]), "Observer metric coverage mismatch")
    for key, value in metrics.items():
        compare(f"{name}/metric/{key}", value, saved["metrics"][key], native=key.startswith("native_") or key.startswith("self_"))
        require(value is not None or isinstance(saved["null_reasons"].get(key), str), "Missing observer null reason")
    for key, value in energies.items():
        compare(f"{name}/energy/{key}", value, saved["energies"][key])
    for key, value in diagnostics.items():
        compare(f"{name}/diagnostic/{key}", value, saved["diagnostics"][key], native=key.startswith("native_represented"))
    RECORD["observer_metric_values_checked"] += len(metrics)


def verify_reference(innovations, state, snapshot):
    step, first = snapshot["step"], state["first"]
    tag = f"seed{snapshot['seed']}/step{step}"
    RECORD["active"] = tag + "/weighted_reference"
    # Formula implemented independently on CPU: weights are covariance weights.
    w = np.array([(1. if j == first else 1 - BETA) * BETA ** (step - j)
                  for j in range(first, step + 1)], dtype=np.float64)
    compare(tag + "/weights", w, array(snapshot, "weights"))
    w_item = artifact(snapshot, "weights")
    require(w_item["first_innovation_step"] == first and w_item["last_innovation_step"] == step
            and w_item["beta"] == BETA, "Weight index metadata mismatch")
    x = innovations[first - 1:step].double().T.contiguous()
    x *= torch.from_numpy(np.sqrt(w)).to("cuda")[None, :]
    require(torch.isfinite(x).all(), "Nonfinite weighted columns")
    gram_gpu = x.T @ x
    symmetry = norm(gram_gpu - gram_gpu.T) / max(norm(gram_gpu), 1e-300)
    gate("gram_symmetry", symmetry, 1e-12)
    gram = ((gram_gpu + gram_gpu.T) / 2).cpu()
    del gram_gpu
    compare(tag + "/Gram", gram, array(snapshot, "gram"))
    eigenvalues, dual = torch.linalg.eigh(gram)
    eigenvalues, dual = eigenvalues.flip(0), dual.flip(1)
    compare(tag + "/full_spectrum", eigenvalues, array(snapshot, "eigenvalues"))
    require(eigenvalues[0] > 0 and eigenvalues[-1] >= -1e-10 * eigenvalues[0], "Invalid reference spectrum")
    lead = float(eigenvalues[0])
    threshold = lead * 1e-10
    positive = int((eigenvalues > threshold).sum())
    require(positive >= 33, "This completed-evidence audit expects 33 positive saved modes")
    gap = float(eigenvalues[31] - eigenvalues[32].clamp_min(0)) / lead
    valid = positive >= RANK and gap > 1e-6
    orth_dual = norm(dual.T @ dual - torch.eye(len(dual), dtype=torch.float64))
    gate("full_rebuilt_dual_orthogonality", orth_dual, 1e-8)
    stored_dual = array(snapshot, "dual_vectors")
    require(stored_dual.shape == (len(eigenvalues), 33), "Saved dual shape mismatch")
    residual = float(torch.linalg.vector_norm(gram @ stored_dual - stored_dual * eigenvalues[:33], dim=0).max()) / lead
    gate("saved_dual_eigen_residual", residual, 1e-8)
    gate("saved_dual_orthogonality", norm(stored_dual.T @ stored_dual - torch.eye(33, dtype=torch.float64)), 1e-8)
    # Orientation-invariant comparison; do not require arbitrary eigensolver signs.
    compare(tag + "/leading32_dual_overlap", square(dual[:, :32].T @ stored_dual[:, :32]), 32.)
    mapped = (x @ dual[:, :33].to("cuda")) / eigenvalues[:33].to("cuda").sqrt()
    saved_mapped = array(snapshot, "mapped_vectors").to("cuda")
    expected_saved_mapping = (x @ stored_dual.to("cuda")) / eigenvalues[:33].to("cuda").sqrt()
    compare(tag + "/saved_dual_to_mapped", expected_saved_mapping.cpu(), saved_mapped.cpu())
    compare(tag + "/leading32_covariance_overlap", square(mapped[:, :32].T @ saved_mapped[:, :32]), 32.)
    for label, u in (("rebuilt", mapped), ("saved", saved_mapped)):
        orth = norm(u.T @ u - torch.eye(33, device="cuda", dtype=torch.float64))
        cov_res = float(torch.linalg.vector_norm(x @ (x.T @ u) - u * eigenvalues[:33].to("cuda"), dim=0).max()) / lead
        gate(label + "_mapped_orthogonality", orth, 1e-6)
        gate(label + "_mapped_covariance_residual", cov_res, 1e-8)
    ref2, trace = square(gram), square(x)
    gate("spectral_trace", abs(float(eigenvalues.sum()) - trace) / max(trace, 1e-300), 1e-9)
    gate("spectral_frobenius", abs(norm(eigenvalues) - math.sqrt(ref2)) / max(math.sqrt(ref2), 1e-300), 1e-9)
    metrics = {"optimal_rank32_energy": float(eigenvalues[:32].clamp_min(0).sum()),
               "covariance_trace": trace, "covariance_squared_frobenius": ref2}
    reference = snapshot["reference"]
    rd = reference["diagnostics"]
    require(rd["positive_rank"] == positive and rd["reference_projector_valid"] == valid
            and rd["first_accepted_innovation_step"] == first, "Reference rank/gap validity mismatch")
    compare(tag + "/positive_threshold", threshold, rd["positive_threshold"])
    compare(tag + "/boundary_gap", gap, rd["boundary_relative_gap"])
    for key, value in {"gram_symmetry_error": symmetry,
                       "dual_orthogonality_error": orth_dual,
                       "dual_eigen_residual": residual,
                       "mapped_orthogonality_error": orth,
                       "covariance_eigen_residual": cov_res,
                       "spectral_trace_error": abs(float(eigenvalues.sum()) - trace) / max(trace, 1e-300),
                       "spectral_frobenius_error": abs(norm(eigenvalues) - math.sqrt(ref2)) / max(math.sqrt(ref2), 1e-300)}.items():
        compare(tag + "/reference_diagnostic/" + key, value, rd[key])
    for key in PROBES:
        probe = state["probes"][key].to(device="cuda", dtype=torch.float64)
        input2 = square(probe)
        out2 = square(mapped[:, :32] @ (mapped[:, :32].T @ probe)) if valid else None
        metrics[key + "_retention"] = fraction(out2, input2) if out2 is not None else None
        compare(tag + f"/reference_input/{key}", input2, reference["energies"][key]["input_squared_norm"])
        compare(tag + f"/reference_output/{key}", out2, reference["energies"][key]["output_squared_norm"])
    metrics["clean_minus_corruption_retention"] = difference(metrics["clean_retention"], metrics["corruption_residual_retention"])
    require(set(metrics) == set(reference["metrics"]), "Reference metric coverage mismatch")
    for key, value in metrics.items():
        compare(tag + "/reference_metric/" + key, value, reference["metrics"][key])
        require(value is not None or isinstance(reference["null_reasons"].get(key), str), "Missing reference null reason")
    for width in (32, 128):
        RECORD["active"] = tag + f"/observer{width}"
        verify_observer(x, eigenvalues, mapped, valid, ref2, state, snapshot, width)
    RECORD["reference_metric_values_checked"] += len(metrics)
    RECORD["completed_snapshots"].append({"seed": snapshot["seed"], "step": step,
                                           "positive_rank": positive, "boundary_relative_gap": gap})
    resources()
    print(json.dumps({"event": "snapshot_pass", "seed": snapshot["seed"], "step": step,
                      "elapsed_seconds": RECORD["resources"]["elapsed_seconds"]}), flush=True)


def verify_seed(seed):
    RECORD["active"] = f"seed{seed}/load"
    replay = json.loads((BASE / "results" / f"replay-seed{seed}.json").read_text())
    require(replay["steps"] == 2000 and replay["all_historical_gates_passed"] and replay["all_state_gates_passed"], "Incomplete replay")
    streams = {}
    for key, suffix in (("raw", "-raw.npy"), ("innovations", "-innovations.npy")):
        matches = [a for a in replay["stream_artifacts"] if a["path"].endswith(suffix)]
        require(len(matches) == 1, "Ambiguous stream artifact")
        item = matches[0]
        path = Path(item["path"]).resolve()
        require(path.is_relative_to(Path(RECORD["bulk_root"]).resolve()), "Stream outside bound root")
        require(path.stat().st_size == item["size_bytes"] and item["completed_rows"] == 2000, "Incomplete stream artifact")
        rows = np.load(path, allow_pickle=False, mmap_mode="r")
        require(rows.shape == (2000, P) and rows.dtype == np.float32, "Stream shape/dtype mismatch")
        hashes = replay["raw_gradient_sha256" if key == "raw" else "innovation_sha256"]
        require(len(hashes) == 2000, "Row hash count mismatch")
        for index, row in enumerate(rows):
            require(np.isfinite(row).all() and hashlib.sha256(row.tobytes()).hexdigest() == hashes[index], "Row hash mismatch")
        RECORD["row_hashes_checked"] += 2000
        streams[key] = torch.from_numpy(np.array(rows, copy=True)).to("cuda")
        del rows
        resources()
    snapshots, states = {}, {}
    for step in STEPS:
        snapshot = json.loads((BASE / "results" / f"snapshot-seed{seed}-step{step}.json").read_text())
        require(snapshot["seed"] == seed and snapshot["step"] == step and snapshot["state_timing"] == PHASE
                and snapshot["all_numerical_gates_passed"], "Snapshot identity/gate mismatch")
        state = torch.load(artifact(snapshot, "state")["path"], map_location="cpu", weights_only=True)
        finite_tree(state)
        require(state["seed"] == seed and state["step"] == step and state["state_timing"] == PHASE, "State phase mismatch")
        require(state["parameter_hash"] == snapshot["parameter_hash"] == replay["trajectory_parameter_sha256"][step - 2], "Pre-Adam parameter hash link mismatch")
        snapshots[step], states[step] = snapshot, state
    mean, first = None, None
    for index, raw in enumerate(streams["raw"]):
        step = index + 1
        RECORD["active"] = f"seed{seed}/sequential_mean/step{step}"
        if mean is None:
            mean = raw.clone()
        else:
            mean.mul_(BETA).add_(raw, alpha=1 - BETA)
        innovation = raw - mean
        require(same_float32_bits(innovation, streams["innovations"][index]), "Bitwise reconstructed innovation mismatch")
        if first is None:
            if float(innovation.norm()) > 0:
                first = step
            else:
                require(torch.count_nonzero(innovation) == 0, "Nonzero underflowed innovation before acceptance")
        if step in STEPS:
            state = states[step]
            require(state["first"] == first, "Snapshot startup mismatch")
            require(same_float32_bits(state["mean"], mean.cpu()), "Snapshot mean mismatch")
            require(tensor_sha(state["raw"]) == replay["raw_gradient_sha256"][index], "Snapshot raw hash mismatch")
            require(tensor_sha(state["innovation"]) == replay["innovation_sha256"][index], "Snapshot innovation hash mismatch")
            for width in (32, 128):
                obs = state["observers"][str(width)]
                require(same_float32_bits(obs["grad_mean"], mean.cpu()), "Saved observer mean mismatch")
                require(obs["step_count"] == step and obs["rank"] == width and obs["decay"] == BETA, "Observer timing/configuration mismatch")
                actual = 0 if obs["V"] is None else obs["V"].shape[1]
                previous = state["previous_bases"][str(width)]
                previous_rank = 0 if previous is None else previous.shape[1]
                require(actual == replay["ranks_by_step"][index][str(width)], "Snapshot actual rank mismatch")
                require(previous_rank == min(32, replay["ranks_by_step"][index - 1][str(width)]), "Previous rank timing mismatch")
                require(obs["stabilization_count"] == replay["repair_counts_by_step"][index][str(width)], "Snapshot repair counter mismatch")
            RECORD["snapshot_hash_phase_mean_checks"] += 1
        RECORD["sequential_mean_innovation_steps_checked"] += 1
        if step % 200 == 0:
            resources()
    require(first == replay["first_accepted_innovation_step"], "Stream startup mismatch")
    del streams["raw"], raw, innovation, mean
    gc.collect()
    for step in STEPS:
        verify_reference(streams["innovations"], states[step], snapshots[step])
        gc.collect()
        torch.cuda.empty_cache()
    RECORD["completed_seeds"].append(seed)


def timeout_handler(_signal, _frame):
    raise TimeoutError("Frozen audit wall-time budget of 300 seconds exceeded")


def run():
    global START
    START = time.perf_counter()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    gpu_info = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.free", "--format=csv,noheader,nounits"], text=True)
    apps = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory", "--format=csv,noheader"], text=True)
    for line in apps.splitlines():
        fields = [item.strip() for item in line.split(",")]
        require(len(fields) >= 2 and Path(fields[1]).name in {"gnome-remote-desktop-daemon", "stremio", "Xorg", "gnome-shell"}, "Another GPU compute process is present")
    require(len(gpu_info.splitlines()) == 1 and "RTX 3090" in gpu_info, "Unexpected GPU")
    require(float(gpu_info.rsplit(",", 1)[1]) * 1024 ** 2 >= 8 * GIB, "Insufficient free GPU memory")
    mem = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    available = int(mem["MemAvailable"].split()[0]) * 1024
    require(available >= 16 * GIB, "Insufficient available host RAM")
    RECORD["occupancy_before"] = {"gpu": gpu_info.strip(), "compute_processes": apps.strip(), "available_ram_bytes": available}
    execution = json.loads((BASE / "results" / "execution.json").read_text())
    require(execution["mode"] == "full" and execution["status"] == "complete" and execution["all_gates_passed"], "Incomplete original execution")
    require(execution["completed_replay_seeds"] == list(SEEDS) and len(execution["completed_snapshots"]) == 12, "Incomplete seed/snapshot count")
    require(execution["official_test_opened"] is False and execution["new_accuracy_or_checkpoint_selection_computed"] is False, "Original execution scope gate failed")
    require(execution["torch"] == torch.__version__ and execution["cuda"] == torch.version.cuda
            and execution["numpy"] == np.__version__ and execution["python"] == sys.version, "Numerical environment changed")
    RECORD["bulk_root"] = execution["bulk_root"]
    RECORD["original_execution_sha256"] = hashlib.sha256((BASE / "results" / "execution.json").read_bytes()).hexdigest()
    RECORD["environment"] = {"python": sys.version, "torch": torch.__version__, "numpy": np.__version__,
                             "cuda": torch.version.cuda, "cpu_threads": torch.get_num_threads(),
                             "gpu": torch.cuda.get_device_name(0), "TF32": False, "deterministic": True}
    provenance = BASE / "raw-summary-audit.json"
    require(provenance.is_file(), "Missing separate byte-binding audit")
    RECORD["separate_byte_binding_audit"] = {"path": str(provenance), "sha256": hashlib.sha256(provenance.read_bytes()).hexdigest()}
    torch.cuda.reset_peak_memory_stats()
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.setitimer(signal.ITIMER_REAL, 300.)
    for seed in SEEDS:
        verify_seed(seed)
        gc.collect()
        torch.cuda.empty_cache()
    resources()
    require(RECORD["row_hashes_checked"] == 12000 and RECORD["sequential_mean_innovation_steps_checked"] == 6000,
            "Incomplete stream audit")
    require(len(RECORD["completed_snapshots"]) == 12 and RECORD["observer_metric_values_checked"] == 384
            and RECORD["reference_metric_values_checked"] == 96, "Incomplete metric audit")
    RECORD["status"] = "passed"


def main():
    global RECORD
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--expected-source-sha256", required=True)
    args = parser.parse_args()
    digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if not args.run or digest != args.expected_source_sha256:
        raise SystemExit("Explicit --run and the frozen current source SHA256 are required")
    output = BASE / "reference-results-audit.json"
    RECORD = {"status": "started", "started_utc": utc(), "source_sha256": digest,
              "scope": "Saved tensors only; no producer/helper imports, dataset access, optimizer replay, training, or new scientific snapshot.",
              "limits": "Historical unsaved parameter/gradient vectors and previous-basis ancestry are not independently replayed. Snapshot phase is checked against saved hash/counter links. Full bulk-file hashes are separately audited.",
              "tolerances": {"float64_absolute": ATOL64, "float64_relative": RTOL64,
                             "native_float32_absolute": ATOL32, "mean_and_row_hashes": "bitwise",
                             "wall_seconds": 300, "gpu_allocated_bytes": 8 * GIB, "rss_bytes": 12 * GIB},
              "comparisons": {key: {"checks": 0, "elements": 0, "null_checks": 0,
                                     "max_absolute_error": 0., "max_tolerance_fraction": 0.}
                              for key in ("float64", "native_float32")},
              "residual_gates": {}, "completed_seeds": [], "completed_snapshots": [],
              "row_hashes_checked": 0, "sequential_mean_innovation_steps_checked": 0,
              "snapshot_hash_phase_mean_checks": 0, "observer_metric_values_checked": 0,
              "reference_metric_values_checked": 0}
    with output.open("x") as stream:
        try:
            print(json.dumps({"event": "audit_start", "source_sha256": digest, "utc": RECORD["started_utc"]}), flush=True)
            with torch.no_grad():
                run()
        except BaseException as exc:
            RECORD["status"] = "failed"
            RECORD["error"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0.)
            RECORD["finished_utc"] = utc()
            if START:
                RECORD.setdefault("resources", {})["final_elapsed_seconds"] = time.perf_counter() - START
            json.dump(RECORD, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    print(json.dumps({"event": "audit_finished", "status": RECORD["status"], "output": str(output),
                      "resources": RECORD.get("resources"), "error": RECORD.get("error")}), flush=True)
    if RECORD["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
