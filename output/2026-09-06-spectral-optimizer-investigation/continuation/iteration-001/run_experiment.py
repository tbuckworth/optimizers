#!/usr/bin/env python3
"""Prospective CPU-only axis-switch experiment; requires explicit --run.

The adjacent protocol.md defines the estimands and analysis. Outputs are fresh
derived artifacts in results/; no production optimizer file is modified.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

# Bound BLAS thread creation before importing NumPy or Torch.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
DIMENSION = 8
RANKS = (1, 2, 4, 8)
METHODS = ("rank1", "rank2", "rank4", "rank8", "exact_finite")
DECAY_WINDOWS = ((.9, 50, 50, 5), (.99, 500, 500, 20))
BACKGROUNDS = (0., .1)
INITIALIZATIONS = ("current_overweight", "regular_weight")
SEEDS = tuple(range(20))
RELATIVE_EIG_TOL = 1e-8


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def draw_stream(seed, n_pre, n_post, background):
    pre_rng = np.random.default_rng(np.random.SeedSequence([20260906, seed, 0]))
    post_rng = np.random.default_rng(np.random.SeedSequence([20260906, seed, 1]))
    pre_normals = pre_rng.standard_normal((n_pre, 9))
    post_normals = post_rng.standard_normal((n_post, 9))
    pre = np.sqrt(background) * pre_normals[:, 1:]
    post = np.sqrt(background) * post_normals[:, 1:]
    pre[:, 0] += pre_normals[:, 0]
    post[:, 1] += post_normals[:, 0]
    return np.concatenate((pre, post), axis=0)


def draw_probe_covariances(seed, n_post):
    rng = np.random.default_rng(np.random.SeedSequence([20260906, seed, 2]))
    probes = rng.standard_normal((n_post, 256, DIMENSION))
    return np.einsum("tbi,tbj->tij", probes, probes) / 256


def truncate(matrix, rank):
    matrix = (matrix + matrix.T) * .5
    values, vectors = np.linalg.eigh(matrix)
    floor = RELATIVE_EIG_TOL * max(float(values[-1]), 0.)
    keep = np.flatnonzero(values > floor)[-rank:]
    if keep.size == 0:
        return np.zeros_like(matrix)
    basis = vectors[:, keep]
    return (basis * values[keep]) @ basis.T


def simulate(raw, decay, initialization):
    """Return every dense covariance state, including the zero first state."""
    histories = np.zeros((len(raw), len(METHODS), DIMENSION, DIMENSION))
    states = histories[0].copy()
    mean = raw[0].copy()
    initial_weight = 1. if initialization == "current_overweight" else 1. - decay
    for step in range(1, len(raw)):
        mean = decay * mean + (1. - decay) * raw[step]
        centered = raw[step] - mean
        outer = np.outer(centered, centered)
        if step == 1:
            candidate = initial_weight * outer
            for method, rank in enumerate(RANKS):
                states[method] = truncate(candidate, rank)
            states[-1] = candidate
        else:
            for method, rank in enumerate(RANKS):
                candidate = decay * states[method] + (1. - decay) * outer
                states[method] = truncate(candidate, rank)
            states[-1] = decay * states[-1] + (1. - decay) * outer
        histories[step] = states
    return histories


def population_ema(decay, background, n_pre, n_post, initialization):
    """Exact E[C] for the finite centered EMA, with random initial running mean."""
    pre_covariance = background * np.eye(DIMENSION)
    post_covariance = background * np.eye(DIMENSION)
    pre_covariance[0, 0] += 1.
    post_covariance[1, 1] += 1.
    mean_second_moment = pre_covariance.copy()
    expected_covariance = np.zeros((DIMENSION, DIMENSION))
    states = np.zeros((n_pre + n_post, DIMENSION, DIMENSION))
    initial_weight = 1. if initialization == "current_overweight" else 1. - decay
    for step in range(1, len(states)):
        covariance = pre_covariance if step < n_pre else post_covariance
        innovation_second_moment = decay ** 2 * (covariance + mean_second_moment)
        mean_second_moment = (
            decay ** 2 * mean_second_moment + (1. - decay) ** 2 * covariance
        )
        if step == 1:
            expected_covariance = initial_weight * innovation_second_moment
        else:
            expected_covariance = (
                decay * expected_covariance + (1. - decay) * innovation_second_moment
            )
        states[step] = expected_covariance
    return states


def leading_geometry(matrices):
    values, vectors = np.linalg.eigh((matrices + matrices.swapaxes(-1, -2)) * .5)
    leading = vectors[..., -1]
    top_values = values[..., -1]
    relative_gap = (top_values - values[..., -2]) / np.maximum(top_values, 1e-30)
    tie_tolerance = np.maximum(1e-14, 1e-10 * np.maximum(top_values, 0.))
    tied = values >= top_values[..., None] - tie_tolerance[..., None]
    top_space = np.einsum("...ij,...j,...kj->...ik", vectors, tied, vectors)
    return {
        "values": values,
        "vectors": leading,
        "relative_gap": relative_gap,
        "top_space": top_space,
        "top_dimension": tied.sum(axis=-1),
    }


def probe_action_error(vectors, reference, sample_covariances):
    """Held-out mean ||(vv'-ww')z||² without materializing projected probes."""
    vrv = np.einsum("tmi,tij,tmj->tm", vectors, sample_covariances, vectors)
    wrw = np.einsum("ti,tij,tj->t", reference, sample_covariances, reference)
    inner = np.einsum("tmi,ti->tm", vectors, reference)
    vrw = np.einsum("tmi,tij,tj->tm", vectors, sample_covariances, reference)
    return np.maximum(vrv + wrw[:, None] - 2. * inner * vrw, 0.)


def sustained_adaptation(overlap, window):
    censoring_step = len(overlap) - window + 1
    for offset in range(censoring_step):
        if bool(np.all(overlap[offset:offset + window] >= .9)):
            step = offset + 1
            return {
                "adaptation_step": step, "censored": False,
                "censoring_step": censoring_step, "restricted_adaptation_step": step,
            }
    return {
        "adaptation_step": None, "censored": True,
        "censoring_step": censoring_step,
        "restricted_adaptation_step": censoring_step,
    }


def analyze(histories, expected, probe_covariances, n_pre, window):
    post = histories[n_pre:]
    actual_geometry = leading_geometry(post)
    expected_geometry = leading_geometry(expected[n_pre:])
    vectors = actual_geometry["vectors"]
    exact_vectors = vectors[:, -1]
    population_vectors = expected_geometry["vectors"]
    axis_vectors = np.zeros((len(post), DIMENSION))
    axis_vectors[:, 1] = 1.
    axis_overlap = np.clip(vectors[:, :, 1] ** 2, 0., 1.)
    finite_overlap = np.clip(np.einsum(
        "tmi,tij,tmj->tm", vectors, actual_geometry["top_space"][:, -1], vectors
    ), 0., 1.)
    population_overlap = np.clip(np.einsum(
        "tmi,tij,tmj->tm", vectors, expected_geometry["top_space"], vectors
    ), 0., 1.)
    exact_rankone_inner = np.einsum("tmi,ti->tm", vectors, exact_vectors)
    population_rankone_inner = np.einsum("tmi,ti->tm", vectors, population_vectors)
    analytic_axis_probe = 2. * (1. - axis_overlap)
    covariance_error = np.linalg.norm(post - post[:, -1, None], axis=(-2, -1))
    covariance_error /= np.maximum(
        np.linalg.norm(post[:, -1], axis=(-2, -1)), 1e-12
    )[:, None]
    trace = {
        "axis_overlap": axis_overlap,
        "finite_eigenspace_overlap": finite_overlap,
        "population_eigenspace_overlap": population_overlap,
        "heldout_axis_mse": probe_action_error(vectors, axis_vectors, probe_covariances),
        "heldout_finite_mse": probe_action_error(vectors, exact_vectors, probe_covariances),
        "heldout_population_mse": probe_action_error(vectors, population_vectors, probe_covariances),
        "analytic_axis_mse": analytic_axis_probe,
        "analytic_finite_rankone_mse": 2. * (1. - np.clip(exact_rankone_inner ** 2, 0., 1.)),
        "analytic_population_rankone_mse": 2. * (1. - np.clip(population_rankone_inner ** 2, 0., 1.)),
        "leading_vectors": vectors,
        "leading_values": actual_geometry["values"][..., -1],
        "retained_rank": (actual_geometry["values"] > RELATIVE_EIG_TOL *
                          actual_geometry["values"][..., -1, None]).sum(axis=-1),
        "relative_covariance_error": covariance_error,
        "finite_relative_gap": actual_geometry["relative_gap"][:, -1],
        "population_relative_gap": expected_geometry["relative_gap"],
        "finite_top_dimension": actual_geometry["top_dimension"][:, -1],
        "population_top_dimension": expected_geometry["top_dimension"],
        "population_axis_overlap": population_vectors[:, 1] ** 2,
    }
    if not all(np.isfinite(value).all() for value in trace.values()):
        raise ValueError("Nonfinite post-switch metric")
    if np.max(np.abs(np.linalg.norm(vectors, axis=-1) - 1.)) > 1e-10:
        raise ValueError("Delivered leading vector not unit length")
    symmetry = np.linalg.norm(histories - histories.swapaxes(-1, -2), axis=(-2, -1))
    covariance_norm = np.linalg.norm(histories, axis=(-2, -1))
    if np.max(symmetry / np.maximum(covariance_norm, 1e-12)) > 1e-10:
        raise ValueError("Covariance symmetry invariant failed")
    values = actual_geometry["values"]
    covariance_trace = np.trace(post, axis1=-2, axis2=-1)
    if np.any(values[..., 0] < -1e-10 * np.maximum(covariance_trace, 1.)):
        raise ValueError("Significantly negative covariance eigenvalue")
    descriptive_window = 10 if len(post) == 50 else 100
    summaries = []
    for method, name in enumerate(METHODS):
        error = 1. - axis_overlap[:, method]
        summary = {
            "method": name,
            "integrated_axis_error": float(error.sum()),
            "mean_axis_error": float(error.mean()),
            "early_mean_axis_error": float(error[:descriptive_window].mean()),
            "late_mean_axis_error": float(error[-descriptive_window:].mean()),
            "mean_finite_eigenspace_error": float((1. - finite_overlap[:, method]).mean()),
            "mean_population_eigenspace_error": float((1. - population_overlap[:, method]).mean()),
            "mean_heldout_axis_mse": float(trace["heldout_axis_mse"][:, method].mean()),
            "mean_heldout_finite_mse": float(trace["heldout_finite_mse"][:, method].mean()),
            "mean_heldout_population_mse": float(trace["heldout_population_mse"][:, method].mean()),
            "max_relative_covariance_error": float(covariance_error[:, method].max()),
            "final_axis_overlap": float(axis_overlap[-1, method]),
            **sustained_adaptation(axis_overlap[:, method], window),
        }
        summaries.append(summary)
    population_summary = sustained_adaptation(trace["population_axis_overlap"], window)
    population_summary["mean_axis_error"] = float((1. - trace["population_axis_overlap"]).mean())
    return trace, summaries, population_summary


def validate_against_stable(raw, histories, decay, initialization, n_pre):
    """Use the existing canonical class; correction is only a harness S rescale."""
    import torch
    sys.path.insert(0, str(REPO))
    from spectral_filter import SpectralGradientFilter

    torch.set_num_threads(1)
    reports = []
    for method, rank in enumerate(RANKS):
        model = torch.nn.Linear(DIMENSION, 1, bias=False, dtype=torch.float64)
        optimizer = torch.optim.SGD(model.parameters(), lr=.1)
        filt = SpectralGradientFilter(
            model, optimizer, rank=rank, decay=decay, warmup=0,
            stable_update=True, relative_eig_tol=RELATIVE_EIG_TOL,
            absolute_eig_floor=0., stabilize_every=100,
        )
        initialized = False
        max_covariance_error = 0.
        max_projector_error = 0.
        near_tied_post_steps = 0
        covariance_errors = []
        projector_errors = []
        for step, gradient in enumerate(raw):
            filt.step_count += 1
            filt._update_svd(torch.as_tensor(gradient, dtype=torch.float64))
            if not initialized and filt.V is not None:
                if initialization == "regular_weight":
                    filt.S.mul_(np.sqrt(1. - decay))
                initialized = True
            if filt.V is None:
                represented = np.zeros((DIMENSION, DIMENSION))
            else:
                represented = (filt.V @ torch.diag(filt.S ** 2) @ filt.V.T).numpy()
            dense = histories[step, method]
            error = float(np.linalg.norm(represented - dense) / max(np.linalg.norm(dense), 1e-12))
            max_covariance_error = max(max_covariance_error, error)
            covariance_errors.append(error)
            if step >= n_pre:
                dense_values, dense_vectors = np.linalg.eigh(dense)
                gap = (dense_values[-1] - dense_values[-2]) / max(dense_values[-1], 1e-30)
                if gap > 1e-5 and filt.V is not None:
                    v = dense_vectors[:, -1]
                    w = filt.V[:, 0].numpy()
                    projector_error = float(np.linalg.norm(np.outer(v, v) - np.outer(w, w)))
                    max_projector_error = max(max_projector_error, projector_error)
                    projector_errors.append(projector_error)
                else:
                    near_tied_post_steps += 1
                    projector_errors.append(None)
        reports.append({
            "rank": rank,
            "max_relative_covariance_error": max_covariance_error,
            "max_nondegenerate_projector_error": max_projector_error,
            "near_tied_post_steps": near_tied_post_steps,
            "covariance_errors_every_observation": covariance_errors,
            "projector_errors_post_switch": projector_errors,
            "passed": max_covariance_error < 1e-6 and max_projector_error < 1e-4,
        })
    return reports


def paired_summaries(records):
    output = []
    metric_names = (
        "mean_axis_error", "integrated_axis_error", "mean_finite_eigenspace_error",
        "mean_population_eigenspace_error", "restricted_adaptation_step",
    )
    for decay, _, _, _ in DECAY_WINDOWS:
        for background in BACKGROUNDS:
            for initialization in INITIALIZATIONS:
                cell = [r for r in records if r["decay"] == decay and
                        r["background_variance"] == background and
                        r["initialization"] == initialization]
                by_seed_method = {(r["seed"], r["method"]): r for r in cell}
                comparisons = [(method, "rank1") for method in METHODS[1:]]
                comparisons += [(method, "exact_finite") for method in METHODS[:-1]]
                for candidate, reference in comparisons:
                    metrics = {}
                    for metric in metric_names:
                        differences = [
                            by_seed_method[(seed, candidate)][metric] -
                            by_seed_method[(seed, reference)][metric] for seed in SEEDS
                        ]
                        array = np.asarray(differences, dtype=np.float64)
                        metrics[metric] = {
                            "seed_order": list(SEEDS), "differences": differences,
                            "mean": float(array.mean()), "median": float(np.median(array)),
                            "sample_sd": float(array.std(ddof=1)),
                            "negative": int((array < -1e-12).sum()),
                            "positive": int((array > 1e-12).sum()),
                            "near_zero": int((np.abs(array) <= 1e-12).sum()),
                        }
                    output.append({
                        "decay": decay, "background_variance": background,
                        "initialization": initialization,
                        "candidate": candidate, "reference": reference,
                        "candidate_censored": sum(by_seed_method[(s, candidate)]["censored"] for s in SEEDS),
                        "reference_censored": sum(by_seed_method[(s, reference)]["censored"] for s in SEEDS),
                        "metrics": metrics,
                    })
    return output


def run(output_dir):
    import torch

    # Refuse overwrites, even when an earlier experiment ended unsuccessfully.
    output_dir.mkdir(parents=False, exist_ok=False)
    started = timestamp()
    start_clock = time.monotonic()
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
    ).strip()
    metadata = {
        "started_utc": started, "status": "running", "device": "CPU",
        "repository_revision_at_execution": revision,
        "protocol_sha256": sha256(HERE / "protocol.md"),
        "script_sha256": sha256(Path(__file__).resolve()),
        "canonical_source_sha256": sha256(REPO / "spectral_filter.py"),
        "python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
        "seeds": list(SEEDS), "methods": list(METHODS),
        "array_dtype": "float64 except explicit integer rank/dimension arrays",
        "metric_index_order": "post-switch step (one-based when reported), method",
        "rank_eight_has_relative_eigenvalue_floor": True,
    }
    write_json(output_dir / "execution.json", metadata)
    arrays = {}
    records = []
    validations = []
    population_records = []
    try:
        for decay, n_pre, n_post, window in DECAY_WINDOWS:
            for background in BACKGROUNDS:
                prefix = f"d{int(round(decay * 100)):02d}_n{int(round(background * 100)):02d}"
                expected_by_init = {}
                for initialization in INITIALIZATIONS:
                    expected = population_ema(decay, background, n_pre, n_post, initialization)
                    expected_by_init[initialization] = expected
                    arrays[f"{prefix}_{initialization}_population_covariances"] = expected
                for seed in SEEDS:
                    raw = draw_stream(seed, n_pre, n_post, background)
                    probe_covariances = draw_probe_covariances(seed, n_post)
                    stream_key = f"{prefix}_s{seed:02d}"
                    arrays[f"{stream_key}_raw_gradients"] = raw
                    arrays[f"{stream_key}_independent_probe_covariances"] = probe_covariances
                    for initialization in INITIALIZATIONS:
                        run_id = f"{stream_key}_{initialization}"
                        histories = simulate(raw, decay, initialization)
                        trace, summaries, population_summary = analyze(
                            histories, expected_by_init[initialization], probe_covariances, n_pre, window
                        )
                        for name, value in trace.items():
                            arrays[f"{run_id}_{name}"] = value
                        common = {
                            "trace_id": run_id, "seed": seed, "decay": decay,
                            "background_variance": background, "initialization": initialization,
                            "n_pre": n_pre, "n_post": n_post, "sustained_window": window,
                        }
                        records.extend([{**common, **summary} for summary in summaries])
                        if seed == 0:
                            population_records.append({
                                "decay": decay, "background_variance": background,
                                "initialization": initialization, **population_summary,
                            })
                            check = validate_against_stable(raw, histories, decay, initialization, n_pre)
                            validations.extend([{**common, **item} for item in check])
                    if seed % 5 == 4:
                        print(f"Completed decay={decay}, background={background}, seeds 0–{seed}; "
                              f"elapsed {time.monotonic() - start_clock:.1f}s", flush=True)
        if len(records) != 800 or len(validations) != 32:
            raise ValueError("Unexpected number of condition or validation records")
        passed = all(check["passed"] for check in validations)
        np.savez_compressed(output_dir / "traces.npz", **arrays)
        write_json(output_dir / "per-seed-results.json", records)
        write_json(output_dir / "paired-summary.json", paired_summaries(records))
        write_json(output_dir / "population-references.json", population_records)
        write_json(output_dir / "validation.json", {"passed": passed, "traces": validations})
        metadata.update({
            "status": "complete_validated" if passed else "complete_invalid_validation",
            "completed_utc": timestamp(), "elapsed_seconds": time.monotonic() - start_clock,
            "condition_records": len(records), "validation_traces": len(validations),
            "array_count": len(arrays), "validation_passed": passed,
        })
        write_json(output_dir / "execution.json", metadata)
        print(json.dumps({key: metadata[key] for key in (
            "status", "elapsed_seconds", "condition_records", "validation_traces", "validation_passed"
        )}), flush=True)
        return 0 if passed else 2
    except Exception as error:
        np.savez_compressed(output_dir / "partial-traces.npz", **arrays)
        write_json(output_dir / "partial-per-seed-results.json", records)
        write_json(output_dir / "partial-validation.json", validations)
        metadata.update({
            "status": "failed", "completed_utc": timestamp(),
            "elapsed_seconds": time.monotonic() - start_clock,
            "error_type": type(error).__name__, "error": str(error),
        })
        write_json(output_dir / "execution.json", metadata)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="Explicitly execute the frozen experiment")
    args = parser.parse_args()
    if not args.run:
        parser.error("Execution requires --run and prior parent GO; no stream has been generated.")
    raise SystemExit(run(HERE / "results"))
