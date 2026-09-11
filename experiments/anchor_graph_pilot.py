"""CPU-only anchor graph feasibility prototype; never constructs a P x P graph.

This is independent of the repository optimizer. CLI reads a frozen synthetic
recipe and runs once. NumPy is the only third-party dependency.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import resource
import time
import tracemalloc

import numpy as np


def truncate_factor(factor: np.ndarray, rank: int) -> np.ndarray:
    """Best rank-r factor of F F.T, without forming that matrix."""
    if factor.ndim != 2 or rank < 1 or not np.isfinite(factor).all():
        raise ValueError("finite matrix and positive rank required")
    if factor.shape[1] == 0:
        return factor.copy()
    q, small = np.linalg.qr(factor, mode="reduced")
    u, singular, _ = np.linalg.svd(small, full_matrices=False)
    width = min(rank, singular.size)
    return (q @ u[:, :width]) * singular[:width]


def rank_one_update(factor: np.ndarray, h: np.ndarray, beta: float,
                    rank: int) -> np.ndarray:
    """Truncate beta F F.T + (1-beta) h h.T; applies first weight correctly."""
    if not 0 <= beta <= 1 or h.shape != (factor.shape[0],):
        raise ValueError("invalid beta or observation shape")
    augmented = np.column_stack((np.sqrt(beta) * factor, np.sqrt(1-beta) * h))
    return truncate_factor(augmented, rank)


def normalized_rows(x: np.ndarray, tol: float = 1e-12):
    if x.ndim != 2 or not np.isfinite(x).all() or tol < 0:
        raise ValueError("finite matrix and nonnegative tolerance required")
    norms = np.linalg.norm(x, axis=1)
    active = np.flatnonzero(norms > tol)
    return x[active] / norms[active, None], active


def farthest_indices(x: np.ndarray, count: int, seed: int) -> np.ndarray:
    """O(n*d*count) selection storing indices, with deterministic tie breaks."""
    if count < 1 or x.shape[0] == 0:
        return np.empty(0, dtype=np.int64)
    count = min(count, x.shape[0])
    selected = [int(np.random.default_rng(seed).integers(x.shape[0]))]
    nearest = np.full(x.shape[0], np.inf)
    for _ in range(count-1):
        distance = np.sum((x - x[selected[-1]]) ** 2, axis=1)
        np.minimum(nearest, distance, out=nearest)
        nearest[selected] = -np.inf
        selected.append(int(np.argmax(nearest)))
    return np.array(selected, dtype=np.int64)


@dataclass
class AnchorGraph:
    """Graph on active rows only. Isolate adjacency is zero; action passes through."""
    size: int
    active: np.ndarray
    anchor_indices: np.ndarray
    z: np.ndarray
    mass: np.ndarray
    r: np.ndarray

    @classmethod
    def from_weights(cls, size, active, anchor_indices, weights):
        z = np.array(weights, dtype=np.float64, copy=True)
        active = np.asarray(active, dtype=np.int64)
        anchor_indices = np.asarray(anchor_indices, dtype=np.int64)
        if z.shape != (active.size, anchor_indices.size):
            raise ValueError("weight shape mismatch")
        if (not np.isfinite(z).all() or np.any(z < 0)
                or np.any(z.sum(axis=1) <= 0)):
            raise ValueError("nonnegative weights with nonzero row mass required")
        if (size < 0 or np.unique(active).size != active.size
                or np.any(active < 0) or np.any(active >= size)
                or not np.isin(anchor_indices, active).all()):
            raise ValueError("invalid active rows or anchors")
        z /= z.sum(axis=1, keepdims=True)
        mass = z.sum(axis=0)
        keep = mass > 0
        z, mass = z[:, keep], mass[keep]
        return cls(size, active, anchor_indices[keep], z, mass,
                   z / np.sqrt(mass)[None, :])

    def matvec(self, x):
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 1 or x.size != self.size:
            raise ValueError("vector shape mismatch")
        out = np.zeros_like(x)
        out[self.active] = self.r @ (self.r.T @ x[self.active])
        return out

    def eigenvectors(self, count, tol=1e-10):
        """Return leading positive adjacency eigenpairs on active rows."""
        if count < 1 or tol <= 0:
            raise ValueError("positive count and tolerance required")
        values, vectors = np.linalg.eigh(self.r.T @ self.r)
        selected = np.flatnonzero(values > tol)[::-1][:count]
        values = values[selected]
        u = (self.r @ vectors[:, selected]) / np.sqrt(values)[None, :]
        return values, u

    def state_bytes(self):
        return sum(a.nbytes for a in
                   (self.active, self.anchor_indices, self.z, self.mass, self.r))


def anchor_graph(factor, anchors=16, sigma=.35, seed=0, zero_tol=1e-12,
                 anchor_indices=None):
    """Re-read anchor profiles from current factor: no stale basis coordinates."""
    if not np.isfinite(sigma) or sigma <= 0 or anchors < 1:
        raise ValueError("positive finite sigma and anchor count required")
    profiles, active = normalized_rows(factor, zero_tol)
    if active.size == 0:
        return AnchorGraph.from_weights(factor.shape[0], active, [],
                                        np.empty((0, 0)))
    if anchor_indices is None:
        local = farthest_indices(profiles, anchors, seed)
    else:
        # Persist row IDs only; discard IDs that have become isolates.
        ids = np.unique(np.asarray(anchor_indices, dtype=np.int64))
        ids = ids[np.isin(ids, active)]
        local = np.searchsorted(active, ids)
        if local.size == 0:
            local = farthest_indices(profiles, anchors, seed)
    distances = 2 - 2 * (profiles @ profiles[local].T)
    np.maximum(distances, 0, out=distances)
    # Subtract row minima before division, so narrow kernels do not all underflow.
    distances -= distances.min(axis=1, keepdims=True)
    with np.errstate(over="ignore", under="ignore"):
        weights = np.exp(-.5 * (distances / sigma) / sigma)
    return AnchorGraph.from_weights(factor.shape[0], active, active[local], weights)


def kmeans(x, count, seed=0, max_iter=30):
    if count < 1 or max_iter < 1 or x.ndim != 2:
        raise ValueError("invalid k-means parameters")
    if x.shape[0] == 0:
        return np.empty(0, dtype=np.int64), 0
    centers = x[farthest_indices(x, count, seed)].copy()
    previous = None
    for iteration in range(1, max_iter+1):
        distance = np.sum(x*x, axis=1)[:, None] - 2*x @ centers.T
        distance += np.sum(centers*centers, axis=1)[None, :]
        labels = np.argmin(distance, axis=1)
        if previous is not None and np.array_equal(labels, previous):
            return labels, iteration
        previous = labels.copy()
        for group in range(centers.shape[0]):
            members = x[labels == group]
            if members.size:
                centers[group] = members.mean(axis=0)
        # Empty clusters keep their center. Do not split identical rows arbitrarily.
    return labels, max_iter


def cluster_graph(graph, count, seed=0, tol=1e-10, max_iter=30):
    values, u = graph.eigenvectors(count, tol)
    labels = np.full(graph.size, -1, dtype=np.int64)
    if u.shape[1] == 0:
        return labels, values, 0
    norms = np.linalg.norm(u, axis=1)
    valid = norms > 0
    assigned, iterations = kmeans(u[valid] / norms[valid, None], count, seed, max_iter)
    labels[graph.active[valid]] = assigned
    return labels, values, iterations


def cluster_mean_action(x, labels):
    """Disjoint group projection in O(P+groups); -1 isolates pass through."""
    x = np.asarray(x, dtype=np.float64)
    labels = np.asarray(labels)
    if (x.ndim != 1 or labels.shape != x.shape
            or not np.issubdtype(labels.dtype, np.integer)
            or np.any(labels < -1) or np.any(labels >= x.size)):
        raise ValueError("labels must be integers in [-1, P)")
    active = labels >= 0
    out = x.copy()
    sums = np.bincount(labels[active], weights=x[active])
    counts = np.bincount(labels[active])
    out[active] = sums[labels[active]] / counts[labels[active]]
    return out


def recovery_metrics(truth, labels):
    """Permutation-invariant ARI and each true group's best whole-cluster F1."""
    _, left = np.unique(truth, return_inverse=True)
    _, right = np.unique(labels, return_inverse=True)
    table = np.zeros((left.max()+1, right.max()+1), dtype=np.int64)
    np.add.at(table, (left, right), 1)
    choose2 = lambda a: np.sum(a * (a-1) / 2)
    pair_count = len(truth) * (len(truth)-1) / 2
    row_pairs, col_pairs = choose2(table.sum(1)), choose2(table.sum(0))
    expected = row_pairs * col_pairs / pair_count if pair_count else 0
    maximum = .5 * (row_pairs + col_pairs)
    ari = ((choose2(table)-expected) / (maximum-expected)
           if maximum != expected else 1.)
    f1 = (2*table / (table.sum(1)[:, None]+table.sum(0)[None, :])).max(1)
    return {"ari": float(ari), "group_best_f1": f1.tolist(),
            "predicted_clusters": int(table.shape[1])}


def moment_error(factor, exact):
    """Relative Frobenius error via small cross-Grams, not dense moments."""
    norm_sq = np.sum((exact.T @ exact)**2)
    error_sq = (np.sum((factor.T @ factor)**2) + norm_sq
                - 2*np.sum((factor.T @ exact)**2))
    return float(np.sqrt(max(0., error_sq) / norm_sq)) if norm_sq else 0.


def synthetic_profiles(size, pattern, seed, recipe):
    rng = np.random.default_rng(seed)
    dim = recipe["latent_dim"]
    if pattern == "balanced":
        labels = np.arange(size) % 4
        centers = np.eye(dim)[:4]
    elif pattern == "rare":
        rare = max(1, int(size * recipe["rare_fraction"]))
        labels = np.concatenate((np.arange(size-rare) % 8, np.full(rare, 8)))
        centers = np.eye(dim)[:9]
    elif pattern == "signed":
        labels = np.arange(size) % 8
        centers = np.concatenate((np.eye(dim)[:4], -np.eye(dim)[:4]))
    else:
        raise ValueError(pattern)
    rng.shuffle(labels)
    profiles = centers[labels] + recipe["noise_sd"] * rng.standard_normal((size, dim))
    observations = rng.standard_normal((recipe["observations"], dim))
    return profiles, labels, observations


def run_cell(size, pattern, seed, recipe):
    profiles, truth, observations = synthetic_profiles(size, pattern, seed, recipe)
    tracemalloc.start()
    started = time.perf_counter()
    factor = np.empty((size, 0))
    latent = np.zeros((recipe["latent_dim"], recipe["latent_dim"]))
    beta = recipe["beta"]
    for observation in observations:
        factor = rank_one_update(factor, profiles @ observation, beta, recipe["rank"])
        latent *= beta
        latent += (1-beta) * np.outer(observation, observation)
    sketch_seconds = time.perf_counter() - started
    eigen, basis = np.linalg.eigh(latent)
    exact = profiles @ (basis * np.sqrt(np.maximum(eigen, 0)))
    batch = truncate_factor(exact, recipe["rank"])
    methods = {}
    for name, current in (("exact", exact), ("batch_rank8", batch),
                          ("stream_rank8", factor)):
        t0 = time.perf_counter()
        graph = anchor_graph(current, recipe["anchors"], recipe["sigma"], seed,
                             recipe["zero_row_tol"])
        labels, values, iterations = cluster_graph(
            graph, int(truth.max()+1), seed, recipe["eigenvalue_tol"],
            recipe["kmeans_max_iter"])
        scores = recovery_metrics(truth, labels)
        scores.update({"refresh_seconds": time.perf_counter()-t0,
                       "trace_ratio": float(np.sum(current**2)/np.sum(exact**2)),
                       "moment_relative_frobenius_error": moment_error(current, exact),
                       "state_bytes": current.nbytes + graph.state_bytes() + labels.nbytes,
                       "graph_shape": list(graph.r.shape),
                       "eigenvalues": values.tolist(), "kmeans_iterations": iterations,
                       "row_sum_max_error": float(np.max(np.abs(graph.matvec(np.ones(size))-1)))})
        if pattern == "rare":
            scores["rare_best_f1"] = scores["group_best_f1"][-1]
        methods[name] = scores
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {"size": size, "pattern": pattern, "seed": seed, "methods": methods,
            "sketch_seconds": sketch_seconds,
            "cell_seconds": time.perf_counter()-started,
            "tracemalloc_peak_bytes": peak,
            "process_high_water_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        if os.environ.get(key) != "1":
            raise RuntimeError(f"set {key}=1 before Python startup")
    recipe_bytes = args.recipe.read_bytes()
    recipe = json.loads(recipe_bytes)
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = {"recipe": recipe, "recipe_sha256": hashlib.sha256(recipe_bytes).hexdigest(),
                "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "numpy": np.__version__, "pid": os.getpid(),
                "started_unix": time.time(), "invocation_id": os.environ.get("INVOCATION_ID"),
                "threads": {k: os.environ.get(k) for k in
                            ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}}
    with (args.output / "acquisition-start.json").open("x") as f:
        json.dump(metadata, f, indent=2)
    results = []
    with (args.output / "cells.jsonl").open("x") as journal:
        for size in recipe["sizes"]:
            for pattern in recipe["patterns"]:
                for seed in recipe["seeds"]:
                    cell = run_cell(size, pattern, seed, recipe)
                    results.append(cell)
                    journal.write(json.dumps(cell, allow_nan=False)+"\n")
                    journal.flush()
                    print(f"P={size} {pattern} seed={seed}: " +
                          ", ".join(f"{m} ARI={v['ari']:.4f}" for m, v in cell["methods"].items()),
                          flush=True)
    metadata.update({"finished_unix": time.time(), "cells": results})
    with (args.output / "results.json").open("x") as f:
        json.dump(metadata, f, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
