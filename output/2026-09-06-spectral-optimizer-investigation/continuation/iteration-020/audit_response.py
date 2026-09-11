#!/usr/bin/env python3
"""Independent I20 saved-array algebra, provenance, and registered-risk audit.

NumPy/stdlib only.  This module does not import the I20 producer core, execute
the native observer, replay randomness, or reconstruct the I19 latent process.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import stat
import subprocess
import time
import zipfile

import numpy as np


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
PARENT_ROOT = Path("/tmp/spectral-experiment-artifacts/spectral-i19-001.wnMmy7")
I19 = HERE.parent / "iteration-019"
SEEDS = tuple(range(19000, 19032))
PROCESS_VARIANCES = (0.0, 0.01, 0.1)
ROTATIONS = (0.0, math.pi / 4)
HORIZON = 4000
BETA = 0.99
EW_DECAYS = (0.99, 0.999)
RHO_STAR = (2.1 - math.sqrt(0.41)) / 2
RHOS = (0.9, RHO_STAR)
ESTIMATORS = ("native", "full_legacy", "ew99", "ew999", "oracle_useful", "oracle_nuisance")
RHO_LABELS = ("r0p9", "rstar")
RESPONSES = ("cp", "rec99", "rec9")
POLICIES = tuple(f"{e}/{r}/{response}" for e in ESTIMATORS for r in RHO_LABELS for response in RESPONSES)
ARRAY_ORDER = ("actions", "direction_present", "weighted_moment", "weight_mass", "normalized_moment",
               "weighted_eigenvalues", "weighted_gap", "output", "useful_squared_alignment", "action_change_norm")
WINDOWS = (("whole", 1, 4000), ("startup", 1, 100), ("transition", 101, 1000), ("late", 1001, 4000))
INITIALIZATION = ("all outputs g1; zero weighted moments/masses before t1; parent post-ingest mu held fixed; "
                  "unavailable full/weighted directions use identity")
SOURCE_NAMES = ("response_core.py", "test_response_core.py", "run_response.py", "test_run_response.py",
                "protocol.md", "best-practices-check.md")
ANALYSIS_NAMES = ("audit_response.py", "test_audit_response.py")
PINS = (
    (PARENT_ROOT / "attempt.json", "3ed5bfe9d66e94a0193e7881d7aa6bd11b61e80eb0d477aeb9cf89c3cfb2a1c1"),
    (PARENT_ROOT / "completion.json", "6a69b9f96f81dcd1066954f53f703280794c88ea29c32beed0c01fa329b4740e"),
    (I19 / "analysis-001/audit.json", "eb35695b43bb26ff2c68a4be4ed4cb5fa6ddb61b999a95e0e5d060d72a077e6c"),
    (I19 / "analysis-001/summary.json", "5d70c8297708af99903be61e554b43d19c0447eb2c97c48bde746712edbef1b7"),
    (I19 / "report-check-001/report-audit.json", "dcc36b3b230d0f08c73852a6fc78062c1c20208ed952589e8a5eede5a0ce1304"),
)
RTOL, ATOL = 2e-11, 5e-12
ARRAY_LIMIT, ROOT_LIMIT = 2 * 1024**3, 3 * 1024**3
SHA = re.compile(r"[0-9a-f]{64}")


class AuditError(ValueError):
    pass


def _utc():
    return datetime.now(timezone.utc).isoformat()


class Checks:
    def __init__(self):
        self.checks = 0
        self.numeric_values = 0
        self.maximum_absolute_error = {}
        self.hash_files = 0
        self.hash_bytes = 0

    def need(self, condition, message):
        self.checks += 1
        if not condition:
            raise AuditError(message)

    def close(self, observed, expected, label, *, rtol=RTOL, atol=ATOL):
        left, right = np.asarray(observed), np.asarray(expected)
        self.need(left.shape == right.shape, label + ": shape differs")
        self.numeric_values += left.size
        self.need(np.isfinite(left).all() and np.isfinite(right).all(), label + ": nonfinite")
        delta = np.abs(left - right)
        self.maximum_absolute_error[label] = max(self.maximum_absolute_error.get(label, 0.0),
                                                 float(np.max(delta, initial=0.0)))
        self.need(np.all(delta <= atol + rtol * np.abs(right)), label + ": numerical closure differs")


def shapes(horizon):
    return {
        "actions": (horizon, 6, 2, 2), "direction_present": (horizon, 6),
        "weighted_moment": (horizon, 2, 2, 2), "weight_mass": (horizon, 2),
        "normalized_moment": (horizon, 2, 2, 2), "weighted_eigenvalues": (horizon, 2, 2),
        "weighted_gap": (horizon, 2), "output": (horizon, 36, 2),
        "useful_squared_alignment": (horizon, 6), "action_change_norm": (horizon, 6),
    }


def parent_policy_names(process_variance):
    names = ["raw", "ema_q0p9", "ema_q0p99", "ema_q0p999", "dema_q0p9", "dema_q0p99", "dema_q0p999"]
    if process_variance != 0:
        names.append("ema_common_steady_oracle")
    return names + ["common_kalman", "useful_oracle_kalman", "scalar_k0", "scalar_k0p5", "scalar_k0p9",
                    "scalar_k1", "oracle_useful", "oracle_nuisance", "native_i17", "native_cp",
                    "native_cp_star", "oracle_useful_star"]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise AuditError("duplicate JSON key: " + key)
            result[key] = value
        return result
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle, object_pairs_hook=pairs, parse_constant=lambda value:
                         (_ for _ in ()).throw(AuditError("nonfinite JSON constant: " + value)))


def _regular(path, checks):
    path = Path(path)
    info = path.lstat()
    checks.need(path.is_absolute() and path.resolve() == path and stat.S_ISREG(info.st_mode) and not path.is_symlink(),
                "not a resolved regular nonsymlink file: " + str(path))
    return info


def _file_record(path, checks, expected_hash=None):
    info = _regular(path, checks)
    digest = sha256(path)
    checks.hash_files += 1
    checks.hash_bytes += info.st_size
    if expected_hash is not None:
        checks.need(digest == expected_hash, "pinned hash differs: " + str(path))
    return {"path": str(path), "size": info.st_size, "sha256": digest}


def _record(root, record, expected_path, checks):
    checks.need(type(record) is dict and set(record) == {"path", "size", "sha256"}
                and record.get("path") == expected_path and type(record.get("size")) is int
                and record["size"] >= 0 and type(record.get("sha256")) is str
                and SHA.fullmatch(record["sha256"]) is not None, "malformed file record: " + expected_path)
    observed = _file_record(root / expected_path, checks)
    checks.need(observed["size"] == record["size"] and observed["sha256"] == record["sha256"],
                "file record binding differs: " + expected_path)
    return root / expected_path


def _rotation_axis(ri):
    if ri not in (0, 1):
        raise AuditError("unregistered rotation index")
    return np.array([math.cos(ROTATIONS[ri]), math.sin(ROTATIONS[ri])], dtype=np.float64)


def _validate_arrays(arrays, parent, ri, checks):
    """Independent equation audit. Every recurrence uses the saved predecessor."""
    checks.need(type(parent) is dict and set(parent) >= {"g", "mu", "A", "basis_present", "full_action",
                                                           "full_basis_present", "output"}, "parent fields missing")
    g, mu = parent["g"], parent["mu"]
    checks.need(type(g) is np.ndarray and g.dtype == np.float64 and g.ndim == 2 and g.shape[1] == 2
                and len(g) > 0 and type(mu) is np.ndarray and mu.shape == g.shape and mu.dtype == np.float64,
                "invalid parent g/mu")
    horizon = len(g)
    expected_shapes = shapes(horizon)
    checks.need(type(arrays) is dict and list(arrays) == list(ARRAY_ORDER), "array membership/order differs")
    for name, expected_shape in expected_shapes.items():
        value = arrays[name]
        dtype = np.dtype(np.bool_ if name == "direction_present" else np.float64)
        checks.need(type(value) is np.ndarray and value.shape == expected_shape and value.dtype == dtype,
                    "array shape/dtype differs: " + name)
        if dtype == np.float64:
            checks.need(np.isfinite(value).all(), "nonfinite array: " + name)
    for name in ("A", "full_action"):
        checks.need(type(parent[name]) is np.ndarray and parent[name].shape == (horizon, 2, 2)
                    and parent[name].dtype == np.float64 and np.isfinite(parent[name]).all(), "invalid parent " + name)
    for name in ("basis_present", "full_basis_present"):
        checks.need(type(parent[name]) is np.ndarray and parent[name].shape == (horizon,)
                    and parent[name].dtype == np.bool_, "invalid parent " + name)
    checks.need(type(parent["output"]) is np.ndarray and parent["output"].shape[0] == horizon
                and parent["output"].shape[2:] == (2,) and parent["output"].dtype == np.float64,
                "invalid parent output")

    eye = np.eye(2, dtype=np.float64)
    axis = _rotation_axis(ri)
    useful = np.outer(axis, axis)
    expected_actions = np.empty_like(arrays["actions"])
    expected_present = np.empty_like(arrays["direction_present"])
    # Native is an exact saved-array copy.  The accepted parent already stores
    # identity at absent-basis positions; assert that fact rather than silently
    # repairing it here.
    expected_actions[:, 0] = parent["A"]
    checks.close(parent["A"][~parent["basis_present"]],
                 np.broadcast_to(eye, parent["A"][~parent["basis_present"]].shape),
                 "parent native absent-basis identity")
    expected_actions[:, 1] = np.where(parent["full_basis_present"][:, None, None], parent["full_action"], eye)
    expected_present[:, 0] = parent["basis_present"]
    expected_present[:, 1] = parent["full_basis_present"]

    z = g - mu
    outer = np.einsum("ti,tj->tij", z, z)
    expected_weighted = np.empty((horizon, 2, 2, 2), dtype=np.float64)
    expected_mass = np.empty((horizon, 2), dtype=np.float64)
    prior_moment = np.zeros((2, 2, 2), dtype=np.float64)
    prior_mass = np.zeros(2, dtype=np.float64)
    decays = np.asarray(EW_DECAYS, dtype=np.float64)
    for t in range(horizon):
        prior_moment = decays[:, None, None] * prior_moment + (1 - decays)[:, None, None] * outer[t]
        prior_mass = decays * prior_mass + (1 - decays)
        expected_weighted[t], expected_mass[t] = prior_moment, prior_mass
    expected_normalized = expected_weighted / expected_mass[:, :, None, None]
    values, vectors = np.linalg.eigh((expected_normalized + np.swapaxes(expected_normalized, -1, -2)) / 2)
    gaps = values[:, :, 1] - values[:, :, 0]
    available = gaps > 1e-10 * np.maximum(1.0, np.abs(values[:, :, 1]))
    expected_present[:, 2:4] = available
    for offset in range(2):
        v = vectors[:, offset, :, 1]
        projector = np.einsum("ti,tj->tij", v, v)
        expected_actions[:, offset + 2] = np.where(available[:, offset, None, None], projector, eye)
    expected_actions[:, 4], expected_actions[:, 5] = useful, eye - useful
    expected_present[:, 4:6] = True

    checks.close(arrays["weighted_moment"], expected_weighted, "zero-start weighted moment recurrence")
    checks.close(arrays["weight_mass"], expected_mass, "zero-start weight-mass recurrence")
    checks.close(arrays["normalized_moment"], expected_normalized, "normalized weighted moments")
    checks.close(arrays["weighted_eigenvalues"], values, "ascending normalized-moment eigenvalues")
    checks.close(arrays["weighted_gap"], gaps, "normalized-moment eigengaps")
    checks.need(np.array_equal(arrays["direction_present"], expected_present), "direction availability differs")
    checks.close(arrays["actions"], expected_actions, "saved action semantics")
    checks.close(arrays["actions"], np.swapaxes(arrays["actions"], -1, -2), "action symmetry")
    checks.close(arrays["actions"] @ arrays["actions"], arrays["actions"], "action idempotence")

    expected_alignment = np.einsum("i,teij,j->te", axis, expected_actions, axis)
    expected_alignment[~expected_present] = 0.0
    checks.close(arrays["useful_squared_alignment"], expected_alignment, "masked useful squared alignment")
    expected_change = np.zeros((horizon, 6), dtype=np.float64)
    if horizon > 1:
        expected_change[1:] = np.linalg.norm(expected_actions[1:] - expected_actions[:-1], axis=(2, 3))
    checks.close(arrays["action_change_norm"], expected_change, "action-change Frobenius norm")
    checks.need(np.array_equal(arrays["action_change_norm"][0], np.zeros(6)), "first action change not exact zero")

    output = arrays["output"].reshape(horizon, 6, 2, 3, 2)
    checks.need(np.array_equal(output[0], np.broadcast_to(g[0], output[0].shape)), "first outputs not exactly g1")
    if horizon > 1:
        P = arrays["actions"][1:]
        h = np.einsum("teij,tj->tei", P, g[1:]) + mu[1:, None, :] \
            - np.einsum("teij,tj->tei", P, mu[1:])
        for rj, rho in enumerate(RHOS):
            previous = output[:-1, :, rj]
            cp = rho * np.einsum("teij,tej->tei", P, previous[:, :, 0]) \
                 + np.einsum("teij,tej->tei", eye - rho * P, h)
            B99 = rho * P + BETA * (eye - P)
            rec99 = np.einsum("teij,tej->tei", B99, previous[:, :, 1]) \
                    + np.einsum("teij,tj->tei", eye - B99, g[1:])
            B9 = rho * P + 0.9 * (eye - P)
            rec9 = np.einsum("teij,tej->tei", B9, previous[:, :, 2]) \
                   + np.einsum("teij,tj->tei", eye - B9, g[1:])
            checks.close(output[1:, :, rj, 0], cp, "literal CP saved-predecessor recurrence")
            checks.close(output[1:, :, rj, 1], rec99, "rec99 saved-predecessor recurrence")
            checks.close(output[1:, :, rj, 2], rec9, "rec9 saved-predecessor recurrence")

    process_variance = parent.get("process_variance")
    parent_names = parent_policy_names(process_variance)
    checks.need(parent["output"].shape == (horizon, len(parent_names), 2), "parent policy/output width differs")
    old = {name: parent["output"][:, i] for i, name in enumerate(parent_names)}
    new = {name: arrays["output"][:, i] for i, name in enumerate(POLICIES)}
    closures = (("native/r0p9/cp", "native_cp"), ("native/rstar/cp", "native_cp_star"),
                ("oracle_useful/r0p9/cp", "oracle_useful"),
                ("oracle_useful/rstar/cp", "oracle_useful_star"),
                ("oracle_nuisance/r0p9/cp", "oracle_nuisance"))
    for current, accepted in closures:
        checks.close(new[current], old[accepted], "accepted parent CP closure: " + current)
    for estimator in ESTIMATORS:
        checks.close(new[f"{estimator}/r0p9/rec9"], old["ema_q0p9"],
                     "rho=.9 rec9 parent EMA closure: " + estimator)
    return {"policy_outputs": new, "checks": checks.checks, "numeric_values": checks.numeric_values}


def validate_arrays(arrays, parent, ri):
    """Validate the ten saved arrays against selected saved I19 arrays; raise on failure."""
    return _validate_arrays(arrays, parent, ri, Checks())


def _load_npz(path, names, expected_shapes, dtypes, checks, *, archive_names=None,
              expanded_cap=32 * 1024**2):
    archive_names = tuple(names) if archive_names is None else tuple(archive_names)
    with zipfile.ZipFile(path) as archive:
        records = archive.infolist()
        checks.need([r.filename for r in records] == [name + ".npy" for name in archive_names],
                    "NPZ member membership/order differs")
        checks.need(all(r.compress_type == zipfile.ZIP_STORED for r in records), "compressed NPZ member rejected")
        checks.need(all(0 <= r.file_size <= 16 * 1024**2 for r in records)
                    and sum(r.file_size for r in records) <= expanded_cap, "NPZ expanded size rejected")
    with np.load(path, allow_pickle=False, max_header_size=10000) as archive:
        arrays = {name: archive[name] for name in names}
    for name in names:
        value = arrays[name]
        checks.need(type(value) is np.ndarray and value.shape == tuple(expected_shapes[name])
                    and value.dtype == np.dtype(dtypes[name])
                    and (value.dtype == np.bool_ or np.isfinite(value).all()), "NPZ array invalid: " + name)
    return arrays


def _source_manifest(commit, names, checks):
    checks.need(type(commit) is str and re.fullmatch(r"[0-9a-f]{40}", commit) is not None, "invalid frozen commit")
    result = []
    for name in names:
        path = HERE / name
        record = _file_record(path, checks)
        relative = str(path.relative_to(REPO))
        frozen = subprocess.check_output(["git", "show", commit + ":" + relative], cwd=REPO)
        checks.need(hashlib.sha256(frozen).hexdigest() == record["sha256"], "source not frozen: " + relative)
        result.append({"path": relative, "size": record["size"], "sha256": record["sha256"]})
    return result


def _verify_source_records(records, checks):
    checks.need(type(records) is list and len({r.get("path") for r in records if type(r) is dict}) == len(records),
                "duplicate/malformed parent sources")
    for row in records:
        checks.need(type(row) is dict and set(row) == {"path", "size", "sha256", "frozen_commit"},
                    "parent source record schema differs")
        relative = Path(row["path"])
        checks.need(not relative.is_absolute() and ".." not in relative.parts, "unsafe parent source path")
        observed = _file_record(REPO / relative, checks)
        checks.need(observed["size"] == row["size"] and observed["sha256"] == row["sha256"], "parent source changed")
        frozen = subprocess.check_output(["git", "show", row["frozen_commit"] + ":" + row["path"]], cwd=REPO)
        checks.need(hashlib.sha256(frozen).hexdigest() == row["sha256"], "parent source freeze differs")


def _expected_ids():
    return [f"{seed}-p{pi}-r{ri}" for seed in SEEDS for pi in range(3) for ri in range(2)]


def _verify_parent(checks):
    records = [_file_record(path, checks, digest) for path, digest in PINS]
    attempt, completion, parent_audit, parent_summary, report = [read_json(path) for path, _ in PINS]
    checks.need(completion.get("schema") == "i19_tracking_completion_v1" and completion.get("status") == "complete"
                and completion.get("failure") is None and completion.get("completed_streams") == 192
                and completion.get("completed_observations") == 768000 and completion.get("attempt_sha256") == PINS[0][1],
                "accepted parent completion differs")
    checks.need(parent_audit.get("status") == "pass" and parent_audit.get("errors") == []
                and parent_summary.get("audit_status") == "pass" and report.get("status") == "pass",
                "accepted parent reports no longer pass")
    checks.need([r.get("id") for r in completion.get("streams", [])] == _expected_ids(), "parent stream roster differs")
    checks.need({p.name for p in PARENT_ROOT.iterdir()} == {"attempt.json", "completion.json", "streams"}
                and not (PARENT_ROOT / "streams").is_symlink(), "parent root membership differs")
    expected_files = set()
    for row in completion["streams"]:
        checks.need(type(row) is dict and set(row) == {"id", "array", "metadata"}, "parent inventory record differs")
        for role, suffix in (("array", ".npz"), ("metadata", ".json")):
            expected = "streams/" + row["id"] + suffix
            _record(PARENT_ROOT, row[role], expected, checks)
            expected_files.add(Path(expected).name)
    checks.need({p.name for p in (PARENT_ROOT / "streams").iterdir()} == expected_files,
                "parent physical stream inventory differs")
    provenance = parent_audit["input_provenance"]
    report_provenance = report["input_provenance"]
    groups = ((attempt["frozen_commit"], attempt["sources"]),
              (provenance["analysis_commit"], provenance["analysis_sources"]),
              (report_provenance["report_commit"], report_provenance["report_sources"]))
    sources = [dict(row, frozen_commit=commit) for commit, rows in groups for row in rows]
    checks.need(len(sources) == 11, "parent source closure count differs")
    _verify_source_records(sources, checks)
    return {"root": str(PARENT_ROOT), "files": records, "sources": sources}, completion, parent_summary


def _seed_summary(values):
    complete = set(values) == {str(seed) for seed in SEEDS} and all(type(v) in (int, float)
                and not isinstance(v, bool) and math.isfinite(v) for v in values.values())
    if not complete:
        return {"per_seed": values, "available": False, "n_independent_seeds": 32, "mean": None,
                "standard_error": None, "positive_seeds": None, "negative_seeds": None, "zero_seeds": None,
                "no_survivor_averaging": True}
    a = np.asarray([values[str(seed)] for seed in SEEDS], dtype=np.float64)
    return {"per_seed": values, "available": True, "n_independent_seeds": 32, "mean": float(a.mean()),
            "standard_error": float(a.std(ddof=1) / math.sqrt(32)), "positive_seeds": int(np.sum(a > 0)),
            "negative_seeds": int(np.sum(a < 0)), "zero_seeds": int(np.sum(a == 0)),
            "no_survivor_averaging": True}


def _mse(delivery, signal):
    error = delivery - signal
    return float(np.mean(np.sum(error * error, axis=1)))


def _stream_rows(arrays, parent, seed, pi, ri):
    metrics, diagnostics = [], []
    for window, first, last in WINDOWS:
        sl = slice(first - 1, last)
        for column, policy in enumerate(POLICIES):
            metrics.append({"seed": seed, "process_index": pi, "rotation_index": ri, "window": window,
                            "policy": policy, "observations": last - first + 1,
                            "mse": _mse(arrays["output"][sl, column], parent["s"][sl])})
        for ei, estimator in enumerate(ESTIMATORS):
            present = arrays["direction_present"][sl, ei]
            diagnostics.append({"seed": seed, "process_index": pi, "rotation_index": ri, "window": window,
                "estimator": estimator, "observations": last - first + 1,
                "direction_present_observations": int(present.sum()),
                "direction_absent_observations": int((~present).sum()),
                "mean_useful_squared_alignment_when_present":
                    float(arrays["useful_squared_alignment"][sl, ei][present].mean()) if present.any() else None,
                "mean_action_change_norm": float(arrays["action_change_norm"][sl, ei].mean())})
    return metrics, diagnostics


def _validate_parent_means(summary, checks):
    rows = summary.get("equal_seed_mse_summaries")
    expected = [(pi, ri, window, policy) for pi, q in enumerate(PROCESS_VARIANCES) for ri in range(2)
                for window, _, _ in WINDOWS for policy in parent_policy_names(q)]
    checks.need(type(rows) is list and len(rows) == 472
                and [(r.get("process_index"), r.get("rotation_index"), r.get("window"), r.get("policy")) for r in rows]
                == expected, "parent 472-mean roster differs")
    for row in rows:
        observed, recomputed = row["mse"], _seed_summary(row["mse"]["per_seed"])
        checks.need(set(observed) == set(recomputed), "parent seed-summary schema differs")
        for key in ("mean", "standard_error"):
            checks.close(observed[key], recomputed[key], "parent mean arithmetic: " + key)
        for key in ("available", "n_independent_seeds", "positive_seeds", "negative_seeds", "zero_seeds",
                    "no_survivor_averaging"):
            checks.need(observed[key] == recomputed[key], "parent mean metadata differs: " + key)
    return rows


def _aggregate(metrics, diagnostics, parent_controls):
    index = {(r["seed"], r["process_index"], r["rotation_index"], r["window"], r["policy"]): r["mse"]
             for r in metrics}
    if len(index) != 27648:
        raise AuditError("per-seed metric roster is not exactly 27,648")
    means = []
    for pi in range(3):
        for ri in range(2):
            for window, _, _ in WINDOWS:
                for policy in POLICIES:
                    means.append({"process_index": pi, "rotation_index": ri, "window": window, "policy": policy,
                                  "mse": _seed_summary({str(seed): index[(seed, pi, ri, window, policy)] for seed in SEEDS})})
    controls = {(r["seed"], r["process_index"], r["rotation_index"], r["window"], r["policy"]): r["mse"]
                for r in parent_controls if r["policy"] in ("ema_q0p9", "common_kalman")}
    contrast_specs = []
    nonoracle = ESTIMATORS[:4]
    for estimator in nonoracle:
        for rho in RHO_LABELS:
            for response in RESPONSES:
                target = f"{estimator}/{rho}/{response}"
                contrast_specs.extend((target, comparator) for comparator in ("parent/ema_q0p9", "parent/common_kalman"))
    for target_estimator, comparator_estimator in (("ew999", "ew99"), ("ew99", "full_legacy"),
                                                    ("full_legacy", "native")):
        for rho in RHO_LABELS:
            for response in RESPONSES:
                contrast_specs.append((f"{target_estimator}/{rho}/{response}",
                                       f"{comparator_estimator}/{rho}/{response}"))
    for estimator in ESTIMATORS:
        for rho in RHO_LABELS:
            contrast_specs.append((f"{estimator}/{rho}/rec99", f"{estimator}/{rho}/cp"))
    for estimator in ESTIMATORS:
        for rho in RHO_LABELS:
            contrast_specs.append((f"{estimator}/{rho}/rec9", f"{estimator}/{rho}/rec99"))
    if len(contrast_specs) != 90 or len(set(contrast_specs)) != 90:
        raise AssertionError("internal contrast roster")
    contrasts = []
    for pi in range(3):
        for ri in range(2):
            for window, _, _ in WINDOWS:
                for target, comparator in contrast_specs:
                    values = {}
                    for seed in SEEDS:
                        target_value = index[(seed, pi, ri, window, target)]
                        if comparator.startswith("parent/"):
                            comparator_value = controls[(seed, pi, ri, window, comparator.split("/", 1)[1])]
                        else:
                            comparator_value = index[(seed, pi, ri, window, comparator)]
                        values[str(seed)] = comparator_value - target_value
                    contrasts.append({"process_index": pi, "rotation_index": ri, "window": window,
                        "comparison": "comparator_mse_minus_target_mse", "target": target, "comparator": comparator,
                        "positive_favors": target, "effect": _seed_summary(values)})
    diagnostic_index = {(r["seed"], r["process_index"], r["rotation_index"], r["window"], r["estimator"]): r
                        for r in diagnostics}
    if len(diagnostic_index) != 4608:
        raise AuditError("per-seed direction diagnostic roster is not exactly 4,608")
    diagnostic_means = []
    for pi in range(3):
        for ri in range(2):
            for window, _, _ in WINDOWS:
                for estimator in ESTIMATORS:
                    rows = [diagnostic_index[(seed, pi, ri, window, estimator)] for seed in SEEDS]
                    diagnostic_means.append({"process_index": pi, "rotation_index": ri, "window": window,
                        "estimator": estimator,
                        "mean_useful_squared_alignment_when_present": _seed_summary({str(seed):
                            rows[i]["mean_useful_squared_alignment_when_present"] for i, seed in enumerate(SEEDS)}),
                        "mean_action_change_norm": _seed_summary({str(seed): rows[i]["mean_action_change_norm"]
                                                                   for i, seed in enumerate(SEEDS)}),
                        "total_direction_present_observations": sum(r["direction_present_observations"] for r in rows),
                        "total_direction_absent_observations": sum(r["direction_absent_observations"] for r in rows)})
    primary = [r for r in contrasts if r["rotation_index"] == 0 and r["window"] == "whole"]
    strong = {(r["estimator"]): r for r in diagnostic_means
              if r["process_index"] == 2 and r["rotation_index"] == 0 and r["window"] == "late"}
    ew999, ew99 = strong["ew999"]["mean_useful_squared_alignment_when_present"], strong["ew99"]["mean_useful_squared_alignment_when_present"]
    prediction = {"scope": "process=.1, identity rotation, late window, all 32 reused independent seeds",
        "available": ew999["available"] and ew99["available"],
        "ew999_mean_alignment_exceeds_ew99": ew999["mean"] > ew99["mean"] if ew999["available"] and ew99["available"] else None,
        "difference_ew999_minus_ew99": ew999["mean"] - ew99["mean"] if ew999["available"] and ew99["available"] else None,
        "whole_risk_improvement_was_not_predicted": True, "not_a_multiplicity_adjusted_significance_test": True}
    return means, contrasts, primary, diagnostic_means, prediction


def audit(root, frozen_commit, outputdir, analysis_commit):
    """Audit one complete exclusive I20 root and create an exclusive summary/audit directory."""
    checks = Checks()
    started = time.monotonic()
    root, outputdir = Path(root), Path(outputdir)
    checks.need(root.is_absolute() and root.resolve(strict=True) == root and root.parent == Path("/tmp/spectral-experiment-artifacts")
                and root.name.startswith("spectral-i20-001.") and not root.is_symlink(), "direct canonical I20 root required")
    checks.need(os.path.ismount("/private-artifacts/storage") and root.stat().st_dev == Path("/private-artifacts/storage").stat().st_dev,
                "I20 root is not on intended mounted volume")
    checks.need(outputdir.parent.resolve(strict=True) == HERE and outputdir.name == "analysis-001"
                and not outputdir.exists() and not outputdir.is_symlink(), "exclusive iteration-020/analysis-001 required")

    expected_parent, parent_completion, parent_summary = _verify_parent(checks)
    attempt_path, completion_path = root / "attempt.json", root / "completion.json"
    attempt_record, completion_record = _file_record(attempt_path, checks), _file_record(completion_path, checks)
    attempt, completion = read_json(attempt_path), read_json(completion_path)
    attempt_keys = {"schema", "created_utc", "root", "frozen_commit", "sources", "parent", "seeds",
        "process_variances", "rotations", "horizon", "cooperative_seconds", "array_limit_bytes", "root_limit_bytes",
        "pid", "service_invocation_id", "python", "numpy", "platform", "cuda_visible_devices", "paid_spend_usd",
        "paid_reserved_usd"}
    checks.need(set(attempt) == attempt_keys and attempt.get("schema") == "i20_response_attempt_v1"
                and attempt.get("root") == str(root) and attempt.get("frozen_commit") == frozen_commit
                and attempt.get("sources") == _source_manifest(frozen_commit, SOURCE_NAMES, checks)
                and attempt.get("parent") == expected_parent and attempt.get("seeds") == list(SEEDS)
                and attempt.get("process_variances") == list(PROCESS_VARIANCES) and attempt.get("rotations") == list(ROTATIONS)
                and attempt.get("horizon") == HORIZON and attempt.get("cooperative_seconds") == 540
                and attempt.get("array_limit_bytes") == ARRAY_LIMIT and attempt.get("root_limit_bytes") == ROOT_LIMIT
                and type(attempt.get("pid")) is int and attempt["pid"] > 0 and attempt.get("cuda_visible_devices") == ""
                and attempt.get("paid_spend_usd") == 0 and attempt.get("paid_reserved_usd") == 0,
                "I20 attempt schema/configuration/provenance differs")
    acquisition_sources = _source_manifest(frozen_commit, SOURCE_NAMES, checks)
    analysis_sources = _source_manifest(analysis_commit, ANALYSIS_NAMES, checks)
    checks.need(attempt["sources"] == acquisition_sources, "acquisition source manifest changed during admission")
    completion_keys = {"schema", "created_utc", "status", "failure", "frozen_commit", "attempt_sha256",
        "expected_streams", "completed_streams", "completed_observations", "elapsed_seconds", "max_rss_kib", "array_bytes",
        "root_bytes_before_completion", "streams"}
    records = completion.get("streams")
    checks.need(set(completion) == completion_keys and completion.get("schema") == "i20_response_completion_v1"
                and completion.get("status") == "complete" and completion.get("failure") is None
                and completion.get("frozen_commit") == frozen_commit and completion.get("attempt_sha256") == attempt_record["sha256"]
                and completion.get("expected_streams") == 192 and completion.get("completed_streams") == 192
                and completion.get("completed_observations") == 768000 and type(records) is list
                and [r.get("id") for r in records] == _expected_ids(), "I20 completion/complete roster differs")
    checks.need(type(completion.get("elapsed_seconds")) in (int, float) and 0 <= completion["elapsed_seconds"] <= 605
                and type(completion.get("max_rss_kib")) is int and 0 <= completion["max_rss_kib"] <= 4 * 1024**2,
                "I20 time/memory envelope differs")
    expected_files = {"attempt.json", "completion.json"}
    total_bytes = attempt_record["size"] + completion_record["size"]
    for row in records:
        checks.need(type(row) is dict and set(row) == {"id", "array", "metadata"}, "I20 stream record differs")
        for role, suffix in (("array", ".npz"), ("metadata", ".json")):
            name = "streams/" + row["id"] + suffix
            path = _record(root, row[role], name, checks)
            expected_files.add(name)
            total_bytes += path.stat().st_size
    actual_files = set()
    for current, dirs, files in os.walk(root, followlinks=False):
        for name in dirs:
            path = Path(current) / name
            checks.need(not path.is_symlink() and path == root / "streams", "unexpected/symlink directory in I20 root")
        for name in files:
            actual_files.add(str((Path(current) / name).relative_to(root)))
    checks.need({p.name for p in root.iterdir()} == {"attempt.json", "completion.json", "streams"}
                and actual_files == expected_files, "I20 physical file roster differs")
    checks.need(completion["array_bytes"] == sum(r["array"]["size"] for r in records)
                and completion["array_bytes"] <= ARRAY_LIMIT and total_bytes <= ROOT_LIMIT
                and completion["root_bytes_before_completion"] == total_bytes - completion_record["size"],
                "I20 storage accounting differs")

    parent_rows = parent_completion["streams"]
    metrics, diagnostics, parent_controls, stream_checks = [], [], [], []
    parent_metric_index = {(r["seed"], r["process_index"], r["rotation_index"], r["window"], r["policy"]): r
                           for r in parent_summary["per_seed_window_metrics"]}
    for row, parent_row in zip(records, parent_rows, strict=True):
        checks.need(time.monotonic() - started < 280, "audit cooperative time limit reached")
        match = re.fullmatch(r"(190(?:0[0-9]|1[0-9]|2[0-9]|3[01]))-p([0-2])-r([01])", row["id"])
        checks.need(match is not None, "invalid stream id")
        seed, pi, ri = map(int, match.groups())
        parent_array_path = PARENT_ROOT / parent_row["array"]["path"]
        parent_meta = read_json(PARENT_ROOT / parent_row["metadata"]["path"])
        parent_core = parent_meta["core"]
        selected = ("g", "mu", "A", "basis_present", "full_action", "full_basis_present", "s", "output")
        parent_arrays = _load_npz(parent_array_path, selected,
            {name: parent_core["array_shapes"][name] for name in selected},
            {name: parent_core["array_dtypes"][name] for name in selected}, checks,
            archive_names=parent_core["array_order"], expanded_cap=96 * 1024**2)
        parent_arrays["process_variance"] = PROCESS_VARIANCES[pi]
        metadata = read_json(root / row["metadata"]["path"])
        expected_meta = {"schema": "i20_response_stream_v1", "id": row["id"], "seed": seed, "process_index": pi,
            "rotation_index": ri, "process_variance": PROCESS_VARIANCES[pi], "rotation": ROTATIONS[ri], "horizon": HORIZON,
            "frozen_commit": frozen_commit, "parent_array_sha256": parent_row["array"]["sha256"],
            "parent_metadata_sha256": parent_row["metadata"]["sha256"], "policy_names": list(POLICIES),
            "estimator_names": list(ESTIMATORS), "array_order": list(ARRAY_ORDER),
            "array_shapes": {name: list(shape) for name, shape in shapes(HORIZON).items()},
            "array_dtypes": {name: "bool" if name == "direction_present" else "float64" for name in ARRAY_ORDER},
            "initialization": INITIALIZATION}
        elapsed = metadata.pop("elapsed_seconds", None)
        checks.need(metadata == expected_meta and type(elapsed) in (int, float) and math.isfinite(elapsed)
                    and 0 <= elapsed <= completion["elapsed_seconds"], "I20 stream metadata differs")
        arrays = _load_npz(root / row["array"]["path"], ARRAY_ORDER, shapes(HORIZON), expected_meta["array_dtypes"], checks)
        before = checks.checks
        _validate_arrays(arrays, parent_arrays, ri, checks)
        new_rows, new_diagnostics = _stream_rows(arrays, parent_arrays, seed, pi, ri)
        metrics.extend(new_rows)
        diagnostics.extend(new_diagnostics)
        old_names = parent_policy_names(PROCESS_VARIANCES[pi])
        for window, first, last in WINDOWS:
            for policy in ("ema_q0p9", "common_kalman"):
                recomputed = _mse(parent_arrays["output"][first - 1:last, old_names.index(policy)],
                                  parent_arrays["s"][first - 1:last])
                accepted = parent_metric_index[(seed, pi, ri, window, policy)]
                checks.close(recomputed, accepted["mse"], "parent control MSE binding: " + policy)
                parent_controls.append(dict(accepted, mse=recomputed))
        stream_checks.append({"id": row["id"], "status": "pass", "arrays": 10, "observations": HORIZON,
                              "checks": checks.checks - before})
        checks.need(sha256(root / row["array"]["path"]) == row["array"]["sha256"]
                    and sha256(root / row["metadata"]["path"]) == row["metadata"]["sha256"]
                    and sha256(parent_array_path) == parent_row["array"]["sha256"], "array changed during audit")
    parent_means = _validate_parent_means(parent_summary, checks)
    means, contrasts, primary, diagnostic_means, prediction = _aggregate(metrics, diagnostics, parent_controls)
    checks.need(len(means) == 864 and len(contrasts) == 2160 and len(primary) == 270
                and len(diagnostic_means) == 144, "registered aggregate counts differ")
    checks.need(sha256(attempt_path) == attempt_record["sha256"] and sha256(completion_path) == completion_record["sha256"],
                "I20 terminal record changed during audit")
    checks.need(_verify_parent(checks)[0] == expected_parent, "parent changed during audit")
    checks.need(_source_manifest(frozen_commit, SOURCE_NAMES, checks) == acquisition_sources,
                "acquisition sources changed during audit")
    checks.need(_source_manifest(analysis_commit, ANALYSIS_NAMES, checks) == analysis_sources,
                "analysis sources changed during audit")
    elapsed = time.monotonic() - started
    checks.need(elapsed < 280, "audit cooperative time limit reached after aggregation")
    created_utc = _utc()
    summary = {"schema": "i20_response_summary_v1", "created_utc": created_utc,
        "audit_status": "pass", "artifact_root": str(root),
        "completed_streams": 192, "expected_streams": 192, "all_scientific_streams_complete": True,
        "attempt_sha256": attempt_record["sha256"], "completion_sha256": completion_record["sha256"],
        "input_provenance": {"acquisition_commit": frozen_commit, "acquisition_sources": attempt["sources"],
                             "analysis_commit": analysis_commit, "analysis_sources": analysis_sources,
                             "parent": expected_parent},
        "array_audit_streams": stream_checks, "per_seed_window_metrics": metrics,
        "equal_seed_mse_summaries": means, "registered_contrasts": contrasts,
        "primary_identity_whole_contrasts": primary, "per_seed_direction_diagnostics": diagnostics,
        "equal_seed_direction_summaries": diagnostic_means, "strong_cell_late_alignment_prediction": prediction,
        "reused_parent_equal_seed_mse_summaries": parent_means,
        "scope": ["Outcome-informed counterfactual on 32 reused I19 seed bundles; not fresh replication or neural evidence.",
                  "Saved-array NumPy/stdlib audit only; no native observer, RNG replay, optimizer, Torch, or latent-process reconstruction.",
                  "SE and exact signs use 32 seeds; rotations and observations are not independent replications.",
                  "All 90 registered contrasts per process/rotation/window are retained; positive means comparator MSE minus target MSE.",
                  "rho=.9 rec9 is an implementation control equal to EMA .9, not a novel scientific result."]}
    outputdir.mkdir(exist_ok=False)
    summary_path = outputdir / "summary.json"
    with summary_path.open("x", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, allow_nan=False)
        handle.write("\n")
    payload = {"schema": "i20_response_audit_v1", "created_utc": created_utc,
        "status": "pass", "artifact_root": str(root),
        "attempt_sha256": attempt_record["sha256"], "completion_sha256": completion_record["sha256"],
        "summary_sha256": sha256(summary_path), "input_provenance": summary["input_provenance"],
        "checks": checks.checks, "numeric_values_checked": checks.numeric_values, "errors": [],
        "maximum_absolute_error_by_relation": checks.maximum_absolute_error,
        "hash_files_verified": checks.hash_files, "hash_bytes_streamed": checks.hash_bytes,
        "analysis_runtime": {"python": platform.python_version(), "numpy": np.__version__, "platform": platform.platform(),
            "elapsed_seconds": elapsed, "cooperative_limit_seconds": 280, "service_hard_limit_seconds": 300,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "threads": {name: os.environ.get(name) for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}},
        "scope": "independent saved-array algebra/risk audit; no producer numerical import"}
    with (outputdir / "audit.json").open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write("\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--frozen-commit", required=True)
    parser.add_argument("--analysis-commit", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or any(os.environ.get(name) != "1" for name in
            ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")):
        raise SystemExit("CPU-only one-thread audit environment required")
    try:
        summary = audit(args.root, args.frozen_commit, args.out, args.analysis_commit)
    except Exception as exc:
        print(json.dumps({"status": "fail", "error_type": type(exc).__name__, "error": str(exc)}))
        return 1
    print(json.dumps({"status": "pass", "streams": summary["completed_streams"],
                      "per_seed_metrics": len(summary["per_seed_window_metrics"]),
                      "contrasts": len(summary["registered_contrasts"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
