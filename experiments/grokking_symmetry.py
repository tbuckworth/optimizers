"""Pure diagnostics for modular-addition logits on the lexicographic full grid.

This module never evaluates or mutates a model.  Rows use ``id = a * p + b``.
All edge selection depends only on the supplied train/test partition.
"""
from __future__ import annotations

import hashlib
import math
from typing import Iterable

import numpy as np


DEFAULT_DELTAS = (1, 2, 4, 8, 16, 32)
SCHEMA = "grokking_symmetry_v1"
MATCH_NAMESPACE = "grokking-symmetry-match-v1"


def _int_array_hash(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values, dtype="<i8")
    digest = hashlib.sha256()
    digest.update(b"little-endian-int64-v1\0")
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _validate_ids(values, name: str, limit: int) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"{name} must be a one-dimensional integer array")
    result = np.asarray(array, dtype=np.int64)
    if len(np.unique(result)) != len(result):
        raise ValueError(f"{name} contains duplicate ids")
    if np.any(result < 0) or np.any(result >= limit):
        raise ValueError(f"{name} contains an out-of-range id")
    return np.sort(result)


def _edge_destination(source: np.ndarray, axis: str, delta: int, p: int) -> np.ndarray:
    a, b = source // p, source % p
    if axis == "a":
        a = (a + delta) % p
    elif axis == "b":
        b = (b + delta) % p
    else:
        raise ValueError("axis must be 'a' or 'b'")
    return a * p + b


def _metric(
    logits: np.ndarray,
    centered: np.ndarray,
    source: np.ndarray,
    destination: np.ndarray,
    output_shift: int,
    p: int,
    energy_floor: float,
) -> dict:
    if source.shape != destination.shape:
        raise ValueError("source and destination edge arrays differ in shape")
    count = int(len(source))
    if count == 0:
        return {
            "count": 0,
            "numerator": 0.0,
            "denominator": 0.0,
            "centered_rms": 0.0,
            "raw_rms": 0.0,
            "value": None,
            "energy_defined": False,
        }
    left = centered[source]
    right = centered[destination]
    shifted = np.roll(left, shift=int(output_shift) % p, axis=1)
    difference = right - shifted
    numerator = float(np.einsum("ij,ij->", difference, difference))
    denominator = float(
        np.einsum("ij,ij->", left, left) + np.einsum("ij,ij->", right, right)
    )
    raw_left, raw_right = logits[source], logits[destination]
    raw_energy = float(
        np.einsum("ij,ij->", raw_left, raw_left)
        + np.einsum("ij,ij->", raw_right, raw_right)
    )
    centered_rms = math.sqrt(max(0.0, denominator) / (2 * count * p))
    raw_rms = math.sqrt(max(0.0, raw_energy) / (2 * count * p))
    defined = bool(centered_rms > energy_floor and denominator > 0.0)
    return {
        "count": count,
        "numerator": numerator,
        "denominator": denominator,
        "centered_rms": centered_rms,
        "raw_rms": raw_rms,
        "value": numerator / denominator if defined else None,
        "energy_defined": defined,
    }


def _hash_key(
    namespace: str, role: str, axis: str, delta: int, source_sum: int,
    source: int, destination: int,
) -> bytes:
    text = (
        f"{namespace}|{role}|{axis}|{delta}|{source_sum}|"
        f"{source}|{destination}"
    )
    return hashlib.sha256(text.encode("ascii")).digest()


def _select_by_hash(
    edges: np.ndarray,
    count: int,
    namespace: str,
    role: str,
    axis: str,
    delta: int,
    source_sum: int,
) -> np.ndarray:
    if count == 0:
        return np.empty((0, 2), dtype=np.int64)
    order = sorted(
        range(len(edges)),
        key=lambda index: (
            _hash_key(
                namespace, role, axis, delta, source_sum,
                int(edges[index, 0]), int(edges[index, 1]),
            ),
            int(edges[index, 0]),
            int(edges[index, 1]),
        ),
    )
    return np.asarray(edges[order[:count]], dtype=np.int64)


def _pooled_metric(
    logits: np.ndarray,
    centered: np.ndarray,
    edge_blocks: Iterable[np.ndarray],
    shifts: Iterable[int],
    p: int,
    energy_floor: float,
) -> dict:
    metrics = []
    for edges, shift in zip(edge_blocks, shifts):
        metrics.append(
            _metric(
                logits, centered, edges[:, 0], edges[:, 1], shift, p,
                energy_floor,
            )
        )
    count = sum(item["count"] for item in metrics)
    numerator = sum(item["numerator"] for item in metrics)
    denominator = sum(item["denominator"] for item in metrics)
    if count:
        centered_rms = math.sqrt(max(0.0, denominator) / (2 * count * p))
        raw_energy = sum(
            (item["raw_rms"] ** 2) * (2 * item["count"] * p)
            for item in metrics
        )
        raw_rms = math.sqrt(max(0.0, raw_energy) / (2 * count * p))
    else:
        centered_rms = raw_rms = 0.0
    defined = bool(centered_rms > energy_floor and denominator > 0.0)
    return {
        "count": count,
        "numerator": float(numerator),
        "denominator": float(denominator),
        "centered_rms": centered_rms,
        "raw_rms": raw_rms,
        "value": float(numerator / denominator) if defined else None,
        "energy_defined": defined,
    }


def symmetry_diagnostics(
    logits,
    train_ids,
    test_ids,
    *,
    p: int = 113,
    deltas: Iterable[int] = DEFAULT_DELTAS,
    wrong_shift_offset: int = 37,
    energy_floor: float = 1e-12,
    match_namespace: str = MATCH_NAMESPACE,
) -> dict:
    """Measure held-out equivariance and matched train-specific excess.

    ``logits`` must contain every pair in lexicographic order.  The returned
    object contains only Python scalars/containers and is JSON serializable.
    """
    if not isinstance(p, int) or p < 3:
        raise ValueError("p must be an integer at least three")
    if not math.isfinite(energy_floor) or energy_floor < 0:
        raise ValueError("energy_floor must be finite and non-negative")
    values = np.asarray(logits)
    if values.shape != (p * p, p) or not np.issubdtype(values.dtype, np.number):
        raise ValueError("logits must have shape (p*p, p) and numeric dtype")
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("logits must be finite")
    train = _validate_ids(train_ids, "train_ids", p * p)
    test = _validate_ids(test_ids, "test_ids", p * p)
    if len(np.intersect1d(train, test)):
        raise ValueError("train_ids and test_ids overlap")
    if len(train) + len(test) != p * p or not np.array_equal(
        np.sort(np.concatenate((train, test))), np.arange(p * p)
    ):
        raise ValueError("train_ids and test_ids must partition the full grid")
    delta_values = tuple(int(delta) for delta in deltas)
    if (
        not delta_values
        or any(delta <= 0 or delta >= p for delta in delta_values)
        or len(set(delta_values)) != len(delta_values)
    ):
        raise ValueError("deltas must be distinct integers in [1,p)")
    wrong_offset = int(wrong_shift_offset) % p
    if wrong_offset == 0:
        raise ValueError("wrong_shift_offset must be nonzero modulo p")
    if not isinstance(match_namespace, str) or not match_namespace:
        raise ValueError("match_namespace must be a nonempty string")

    # A private copy ensures callers cannot observe mutation through a view.
    values = values.copy()
    centered = values - values.mean(axis=1, keepdims=True)
    train_mask = np.zeros(p * p, dtype=bool)
    test_mask = np.zeros(p * p, dtype=bool)
    train_mask[train] = True
    test_mask[test] = True

    heldout = {"a": {}, "b": {}}
    for axis in ("a", "b"):
        for delta in delta_values:
            destination = _edge_destination(test, axis, delta, p)
            keep = test_mask[destination]
            source_edges, destination_edges = test[keep], destination[keep]
            correct = _metric(
                values, centered, source_edges, destination_edges, delta, p,
                energy_floor,
            )
            wrong_shift = (delta + wrong_offset) % p
            wrong = _metric(
                values, centered, source_edges, destination_edges, wrong_shift,
                p, energy_floor,
            )
            pairs = np.column_stack((source_edges, destination_edges))
            heldout[axis][str(delta)] = {
                "input_shift": delta,
                "correct_output_shift": delta,
                "wrong_output_shift": wrong_shift,
                "edge_ids_sha256": _int_array_hash(pairs),
                "correct": correct,
                "wrong_shift": wrong,
            }

    a, b = test // p, test % p
    exchange_keep = (a < b) & test_mask[b * p + a]
    exchange_source = test[exchange_keep]
    exchange_destination = (exchange_source % p) * p + exchange_source // p
    exchange_pairs = np.column_stack((exchange_source, exchange_destination))
    exchange = {
        "unordered": True,
        "edge_ids_sha256": _int_array_hash(exchange_pairs),
        "correct": _metric(
            values, centered, exchange_source, exchange_destination, 0, p,
            energy_floor,
        ),
        "wrong_shift": _metric(
            values, centered, exchange_source, exchange_destination,
            wrong_offset, p, energy_floor,
        ),
        "wrong_output_shift": wrong_offset,
    }

    cleanup = {"a": {}, "b": {}}
    pooled_th, pooled_hh, pooled_shifts = [], [], []
    for axis in ("a", "b"):
        for delta in delta_values:
            group_metadata = []
            selected_th, selected_hh = [], []
            train_destination = _edge_destination(train, axis, delta, p)
            test_destination = _edge_destination(test, axis, delta, p)
            train_sums = ((train // p) + (train % p)) % p
            test_sums = ((test // p) + (test % p)) % p
            for source_sum in range(p):
                th_keep = (train_sums == source_sum) & test_mask[train_destination]
                hh_keep = (test_sums == source_sum) & test_mask[test_destination]
                th_candidates = np.column_stack((
                    train[th_keep], train_destination[th_keep]
                )).astype(np.int64, copy=False)
                hh_candidates = np.column_stack((
                    test[hh_keep], test_destination[hh_keep]
                )).astype(np.int64, copy=False)
                matched_count = min(len(th_candidates), len(hh_candidates))
                th_edges = _select_by_hash(
                    th_candidates, matched_count, match_namespace, "TH", axis,
                    delta, source_sum,
                )
                hh_edges = _select_by_hash(
                    hh_candidates, matched_count, match_namespace, "HH", axis,
                    delta, source_sum,
                )
                selected_th.append(th_edges)
                selected_hh.append(hh_edges)
                group_metadata.append({
                    "source_sum": source_sum,
                    "th_candidates": int(len(th_candidates)),
                    "hh_candidates": int(len(hh_candidates)),
                    "matched_count": int(matched_count),
                    "th_edge_ids_sha256": _int_array_hash(th_edges),
                    "hh_edge_ids_sha256": _int_array_hash(hh_edges),
                })
            th_all = np.concatenate(selected_th, axis=0)
            hh_all = np.concatenate(selected_hh, axis=0)
            th_metric = _metric(
                values, centered, th_all[:, 0], th_all[:, 1], delta, p,
                energy_floor,
            )
            hh_metric = _metric(
                values, centered, hh_all[:, 0], hh_all[:, 1], delta, p,
                energy_floor,
            )
            excess = (
                th_metric["value"] - hh_metric["value"]
                if th_metric["value"] is not None and hh_metric["value"] is not None
                else None
            )
            cleanup[axis][str(delta)] = {
                "input_shift": delta,
                "matched_count": int(len(th_all)),
                "th_edge_ids_sha256": _int_array_hash(th_all),
                "hh_edge_ids_sha256": _int_array_hash(hh_all),
                "groups": group_metadata,
                "train_to_heldout": th_metric,
                "heldout_to_heldout": hh_metric,
                "excess": excess,
            }
            pooled_th.append(th_all)
            pooled_hh.append(hh_all)
            pooled_shifts.append(delta)

    pooled_th_metric = _pooled_metric(
        values, centered, pooled_th, pooled_shifts, p, energy_floor
    )
    pooled_hh_metric = _pooled_metric(
        values, centered, pooled_hh, pooled_shifts, p, energy_floor
    )
    pooled_excess = (
        pooled_th_metric["value"] - pooled_hh_metric["value"]
        if pooled_th_metric["value"] is not None
        and pooled_hh_metric["value"] is not None
        else None
    )
    cleanup["pooled"] = {
        "matched_count": pooled_th_metric["count"],
        "th_edge_ids_sha256": _int_array_hash(np.concatenate(pooled_th, axis=0)),
        "hh_edge_ids_sha256": _int_array_hash(np.concatenate(pooled_hh, axis=0)),
        "train_to_heldout": pooled_th_metric,
        "heldout_to_heldout": pooled_hh_metric,
        "excess": pooled_excess,
    }

    return {
        "schema": SCHEMA,
        "p": p,
        "deltas": list(delta_values),
        "wrong_shift_offset": wrong_offset,
        "energy_floor": energy_floor,
        "partition": {
            "train_count": int(len(train)),
            "test_count": int(len(test)),
            "train_ids_sha256": _int_array_hash(train),
            "test_ids_sha256": _int_array_hash(test),
        },
        "matching": {
            "namespace": match_namespace,
            "order": "sha256(namespace|role|axis|delta|source_sum|source|destination),then-ids",
            "group": "axis,delta,source_sum",
            "count_rule": "min(th_candidates,hh_candidates)",
            "outcome_independent": True,
        },
        "heldout_symmetry": heldout,
        "exchange": exchange,
        "cleanup": cleanup,
    }


__all__ = [
    "DEFAULT_DELTAS", "MATCH_NAMESPACE", "SCHEMA", "symmetry_diagnostics",
]
