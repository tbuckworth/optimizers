"""Finite-history reference and diagnostics; no datasets or experiment launcher."""
from __future__ import annotations

import math
import time
import torch

RANK = 32
PROBES = ("clean", "noisy", "corruption_residual", "auxiliary_clean")


def require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def finite(*values):
    for value in values:
        require(torch.isfinite(value).all() if isinstance(value, torch.Tensor)
                else math.isfinite(value), "Non-finite reference/measurement input")


def dot(x, y):
    return float(torch.dot(x.double().reshape(-1), y.double().reshape(-1)))


def norm(x):
    return math.sqrt(dot(x, x))


def ratio(a, b):
    return None if b == 0 else a / b


def difference(a, b):
    return None if a is None or b is None else a - b


def metric(row, name, value, reason="zero_input_norm"):
    finite(value) if value is not None else None
    row["metrics"][name] = value
    if value is None:
        row["null_reasons"][name] = reason


def empty_row():
    return {"metrics": {}, "null_reasons": {}, "energies": {}, "diagnostics": {}}


def weights_for(step, first, beta=.99, device="cpu"):
    require(0 < beta < 1, "Invalid covariance decay")
    if first is None:
        return torch.zeros(0, dtype=torch.float64, device=device)
    require(1 <= first <= step, "Invalid first innovation index")
    age = torch.arange(step - first, -1, -1, dtype=torch.float64, device=device)
    weights = (1 - beta) * torch.pow(beta, age)
    weights[0] = beta ** (step - first)
    return weights


def weighted_columns(innovations, first, beta=.99, device="cpu"):
    """Input rows are chronological unweighted float32 innovations."""
    finite(innovations)
    count, dimension = innovations.shape
    if first is None:
        require(not bool(torch.count_nonzero(innovations)), "Unaccepted nonzero innovation")
        return torch.zeros((dimension, 0), dtype=torch.float64, device=device), weights_for(count, first, beta, device)
    require(not bool(torch.count_nonzero(innovations[:first - 1])), "Nonzero innovation before initialization")
    weights = weights_for(count, first, beta, device)
    x = innovations[first - 1:].to(device=device, dtype=torch.float64).T.contiguous()
    x.mul_(weights.sqrt().unsqueeze(0))
    finite(x, weights)
    return x, weights


def cancellation_square(ref2, est2, inner):
    raw = ref2 + est2 - 2 * inner
    tolerance = 1e-12 * (ref2 + est2 + 2 * abs(inner))
    require(raw >= -tolerance, f"Materially negative covariance squared error: {raw}")
    return max(raw, 0.), raw, tolerance


def synchronize(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


@torch.no_grad()
def reference(x, first, rank=RANK):
    """Return checked scalar metadata and tensors; no p-by-p allocation."""
    finite(x)
    require(x.dtype == torch.float64 and x.ndim == 2, "Reference requires float64 columns")
    device = x.device
    synchronize(device)
    start = time.perf_counter()
    gram_device = x.T @ x
    finite(gram_device)
    gram_norm = norm(gram_device)
    symmetry = norm(gram_device - gram_device.T) / max(gram_norm, 1e-300)
    require(symmetry <= 1e-12, f"Gram symmetry error {symmetry} exceeds 1e-12")
    gram = ((gram_device + gram_device.T) * .5).cpu()
    del gram_device
    synchronize(device)
    gram_seconds = time.perf_counter() - start
    start = time.perf_counter()
    vals, dual = torch.linalg.eigh(gram)
    vals, dual = vals.flip(0), dual.flip(1)
    finite(vals, dual)
    solve_seconds = time.perf_counter() - start
    ref2, trace = dot(gram, gram), dot(x, x)
    positive, threshold, gap = 0, 0., None
    mapped = x.new_zeros((x.shape[0], 0))
    residual_dual = residual_cov = orth_mapped = 0.
    all_dual_orth = norm(dual.T @ dual - torch.eye(len(vals), dtype=torch.float64))
    require(all_dual_orth <= 1e-8, f"Dual orthogonality error {all_dual_orth} exceeds 1e-8")
    start = time.perf_counter()
    if trace > 0:
        require(len(vals) > 0 and vals[0] > 0, "Nonzero reference lacks positive eigenvalue")
        top = float(vals[0])
        require(float(vals[-1]) >= -1e-10 * top, f"Minimum eigenvalue {float(vals[-1])} below {-1e-10 * top}")
        threshold = 1e-10 * top
        positive = int((vals > threshold).sum())
        keep = min(rank + 1, len(vals))
        saved_dual = dual[:, :keep]
        residual_dual = float(torch.linalg.vector_norm(gram @ saved_dual - saved_dual * vals[:keep], dim=0).max()) / top
        require(residual_dual <= 1e-8, f"Dual eigen-residual {residual_dual} exceeds 1e-8")
        mapped_count = min(rank + 1, positive)
        mapped_vals = vals[:mapped_count].to(device)
        mapped = (x @ dual[:, :mapped_count].to(device)) / mapped_vals.sqrt()
        residual_cov = float(torch.linalg.vector_norm(x @ (x.T @ mapped) - mapped * mapped_vals, dim=0).max()) / top
        orth_mapped = norm(mapped.T @ mapped - torch.eye(mapped_count, dtype=torch.float64, device=device))
        require(residual_cov <= 1e-8 and orth_mapped <= 1e-6,
                f"Mapped covariance residual/orthogonality {residual_cov}/{orth_mapped} exceed 1e-8/1e-6")
        if positive >= rank:
            next_value = max(float(vals[rank]), 0.) if len(vals) > rank else 0.
            gap = (float(vals[rank - 1]) - next_value) / top
    else:
        require(not bool(torch.count_nonzero(x)) and not bool(torch.count_nonzero(gram)), "Invalid zero reference")
    spectral_trace_error = abs(float(vals.sum()) - trace) / max(trace, 1e-300)
    spectral_frobenius_error = abs(norm(vals) - math.sqrt(ref2)) / max(math.sqrt(ref2), 1e-300)
    require(spectral_trace_error <= 1e-9 and spectral_frobenius_error <= 1e-9,
            f"Spectral trace/Frobenius errors {spectral_trace_error}/{spectral_frobenius_error} exceed 1e-9")
    synchronize(device)
    mapping_seconds = time.perf_counter() - start
    valid = positive >= rank and gap is not None and gap > 1e-6
    row = empty_row()
    row["metrics"].update(optimal_rank32_energy=float(vals[:rank].clamp_min(0).sum()),
                          covariance_trace=trace, covariance_squared_frobenius=ref2)
    row["diagnostics"].update(first_accepted_innovation_step=first, positive_rank=positive,
                             positive_threshold=threshold, boundary_relative_gap=gap,
                             reference_projector_valid=valid, gram_symmetry_error=symmetry,
                             dual_orthogonality_error=all_dual_orth, dual_eigen_residual=residual_dual,
                             mapped_orthogonality_error=orth_mapped, covariance_eigen_residual=residual_cov,
                             spectral_trace_error=spectral_trace_error,
                             spectral_frobenius_error=spectral_frobenius_error,
                             gram_seconds=gram_seconds, solve_seconds=solve_seconds, mapping_seconds=mapping_seconds)
    tensors = {"gram": gram, "eigenvalues": vals, "dual_vectors": dual[:, :rank + 1],
               "mapped_vectors": mapped, "projector_basis": mapped[:, :rank] if valid else None}
    return row, tensors


def apply_native(value, basis):
    return value.clone() if basis is None else basis @ (basis.T @ value)


def probe_metrics(probes, basis):
    row = empty_row()
    finite(*probes.values())
    if basis is not None:
        finite(basis)
    output = {key: apply_native(value, basis) for key, value in probes.items()}
    finite(*output.values())
    roundoff = 0.
    for key, value in probes.items():
        input2, output2 = dot(value, value), dot(output[key], output[key])
        row["energies"][key] = {"input_squared_norm": input2, "output_squared_norm": output2}
        metric(row, f"native_{key}_retention", ratio(output2, input2))
        represented = value.double() if basis is None else basis.double() @ (basis.double().T @ value.double())
        error = norm(output[key].double() - represented) / max(math.sqrt(input2), 1e-30)
        roundoff = max(roundoff, error)
    require(roundoff <= 1e-5, f"Native/represented action error {roundoff} exceeds 1e-5")
    row["diagnostics"]["native_represented_action_error_max"] = roundoff
    c, r, n = (probes[k] for k in ("clean", "corruption_residual", "noisy"))
    pc, pr, pn = (output[k] for k in ("clean", "corruption_residual", "noisy"))
    joint = {"clean_dot_corruption": dot(c, r), "projected_clean_dot_corruption": dot(pc, pr)}
    for prefix, nc, cc, rr in (("", n, c, r), ("projected_", pn, pc, pr)):
        terms = [dot(nc, nc), dot(cc, cc), dot(rr, rr), 2 * dot(cc, rr)]
        closure = terms[0] - terms[1] - terms[2] - terms[3]
        relative = abs(closure) / max(sum(abs(v) for v in terms), 1e-30)
        require(relative <= 1e-5, f"Probe {prefix}energy closure error {relative} exceeds 1e-5")
        joint[f"{prefix}energy_closure_residual"] = closure
        joint[f"{prefix}energy_closure_relative_error"] = relative
        joint[f"{prefix}noisy_energy"] = terms[0]
    row["energies"]["joint"] = joint
    metric(row, "native_clean_corruption_cosine", ratio(dot(c, r), norm(c) * norm(r)))
    metric(row, "native_projected_clean_corruption_cosine", ratio(dot(pc, pr), norm(pc) * norm(pr)), "zero_projected_component_norm")
    metric(row, "native_clean_minus_corruption_retention", difference(row["metrics"]["native_clean_retention"], row["metrics"]["native_corruption_residual_retention"]))
    return row


@torch.no_grad()
def observer_metrics(x, ref, tensors, state, previous_basis, probes, raw, width, rank=RANK):
    row = empty_row()
    v, singular = state["V"], state["S"]
    actual_rank = 0 if v is None else v.shape[1]
    require(actual_rank <= width, "Observer exceeds estimation width")
    native_basis = None if v is None else v[:, :rank]
    row.update(estimation_width=width, actual_rank=actual_rank)
    native = probe_metrics(probes, native_basis)
    row["metrics"].update(native["metrics"])
    row["null_reasons"].update(native["null_reasons"])
    row["diagnostics"].update(native["diagnostics"])
    row["energies"]["probes"] = {k: native["energies"][k] for k in PROBES}
    row["energies"]["joint"] = native["energies"]["joint"]
    g2 = dot(raw, raw)
    current, previous = apply_native(raw, native_basis), apply_native(raw, previous_basis)
    for basis, output in ((native_basis, current), (previous_basis, previous)):
        represented = raw.double() if basis is None else basis.double() @ (basis.double().T @ raw.double())
        action_error = norm(output.double() - represented) / max(math.sqrt(g2), 1e-30)
        require(action_error <= 1e-5, f"Current-gradient native action error {action_error} exceeds 1e-5")
        row["diagnostics"]["native_represented_action_error_max"] = max(
            action_error, row["diagnostics"]["native_represented_action_error_max"])
    now, before = ratio(dot(current, current), g2), ratio(dot(previous, previous), g2)
    metric(row, "native_current_gradient_retention", now)
    metric(row, "native_previous_current_gradient_retention", before)
    metric(row, "self_inclusion_retention_increment", difference(now, before))
    ref2 = ref["metrics"]["covariance_squared_frobenius"]
    energy = ref["metrics"]["optimal_rank32_energy"]
    est2 = inner = est_trace = 0.
    orth = 0.
    b = None
    if v is not None:
        finite(v, singular)
        require(singular.numel() == actual_rank and bool((singular > 0).all()), "Invalid observer singular spectrum")
        v64, d = v.double(), singular.to(device=x.device, dtype=torch.float64).square()
        g = v64.T @ v64
        orth = norm(g - torch.eye(actual_rank, dtype=torch.float64, device=x.device))
        require(orth <= 5e-3, f"Native basis orthogonality error {orth} exceeds 5e-3")
        a = v64.T @ x
        est2 = float((d[:, None] * d[None, :] * g.square()).sum())
        inner = float((d * a.square().sum(dim=1)).sum())
        est_trace = float((d * g.diag()).sum())
        b = v64[:, :rank]
    error2, raw_error, cancellation_tol = cancellation_square(ref2, est2, inner)
    metric(row, "relative_covariance_error", ratio(math.sqrt(error2), math.sqrt(ref2)), "zero_reference_covariance")
    metric(row, "covariance_estimator_trace", est_trace)
    row["energies"].update(reference_squared_frobenius=ref2, estimator_squared_frobenius=est2,
                           covariance_inner_product=inner, covariance_squared_error_raw=raw_error)
    row["diagnostics"].update(native_basis_orthogonality_error=orth,
                              cancellation_tolerance=cancellation_tol)
    # No-basis native map is identity, not an empty-basis zero operator.
    z = x if b is None else b.T @ x
    trace_pc = dot(z, z)
    output_energy = trace_pc if b is None else float((z * ((b.T @ b) @ z)).sum())
    metric(row, "trace_P_C", trace_pc)
    metric(row, "represented_operator_energy_fraction", ratio(output_energy, energy), "zero_optimal_reference_energy")
    row["energies"]["represented_operator_output_energy"] = output_energy
    captured = span_fraction = distance = None
    rank_reason = "observer_rank_below_32"
    if actual_rank >= rank:
        sv = torch.linalg.svdvals(b)
        condition_ratio = float(sv[-1] / sv[0])
        require(condition_ratio > 1e-8, "Rank-deficient observer leading columns")
        q, _ = torch.linalg.qr(b, mode="reduced")
        q_orth = norm(q.T @ q - torch.eye(rank, dtype=torch.float64, device=x.device))
        require(q_orth <= 1e-10, "QR span not orthonormal")
        row["diagnostics"].update(qr_orthogonality_error=q_orth, basis_singular_value_ratio=condition_ratio)
        captured = dot(q.T @ x, q.T @ x)
        rank_reason = "reference_positive_rank_below_32_or_zero_energy"
        if ref["diagnostics"]["positive_rank"] >= rank and energy > 0:
            span_fraction = captured / energy
            require(-1e-6 <= span_fraction <= 1 + 1e-6, "Invalid orthonormal span energy")
        if tensors["projector_basis"] is not None:
            overlap = dot(q.T @ tensors["projector_basis"], q.T @ tensors["projector_basis"])
            distance = 1 - overlap / rank
            require(-1e-6 <= distance <= 1 + 1e-6, "Invalid projector distance")
            row["energies"]["reference_span_overlap_squared"] = overlap
    row["energies"]["span_captured_energy"] = captured
    metric(row, "span_energy_fraction", span_fraction, rank_reason)
    metric(row, "span_projector_distance", distance,
           "observer_rank_below_32" if actual_rank < rank else "reference_rank_or_boundary_gap_invalid")
    return row


@torch.no_grad()
def add_reference_probes(row, tensors, probes):
    u = tensors["projector_basis"]
    for name, value in probes.items():
        input2 = dot(value, value)
        out2 = None if u is None else dot(u @ (u.T @ value.double()), u @ (u.T @ value.double()))
        result = None if out2 is None else ratio(out2, input2)
        metric(row, f"{name}_retention", result,
               "reference_rank_or_boundary_gap_invalid" if u is None else "zero_input_norm")
        row["energies"][name] = {"input_squared_norm": input2, "output_squared_norm": out2}
    metric(row, "clean_minus_corruption_retention",
           difference(row["metrics"]["clean_retention"], row["metrics"]["corruption_residual_retention"]),
           "reference_rank_or_boundary_gap_invalid" if u is None else "zero_component_norm")
