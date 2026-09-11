#!/usr/bin/env python3
"""Once-only, bounded CPU analysis of two saved unpatched logit slices per run."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import signal
import struct
import subprocess
import sys
import time
import zipfile

START = time.monotonic()
THREAD_KEYS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
for key in THREAD_KEYS:
    os.environ[key] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
resource.setrlimit(resource.RLIMIT_AS, (1024**3, 1024**3))
resource.setrlimit(resource.RLIMIT_CPU, (120, 120))
resource.setrlimit(resource.RLIMIT_FSIZE, (50 * 1024**2, 50 * 1024**2))
os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
signal.alarm(120)

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ACQUISITION = Path("/tmp/spectral-experiment-artifacts/spectral-selectivity-boundary-20260910.2IruKP/acquisition-001")
COMPLETION_SHA = "de66b0331bec90c9c24585ab9874af447de17cc653661984c0237fb65da66ceb"
AUDIT_SHA = "d1b3a47371d8eed4d8fe46ec4b0c1cd3586f788a69cd003b264a7c0b39dd3907"
SEEDS = (202609111, 202609112, 202609113)
CELLS = ("clean", "diffuse", "shared", "sham")
POLICIES = ("raw", "native32", "norm_raw")
STEPS = (100, 2000)
COMMON = tuple(d for d in range(10) if d != 8)
SHIFT = math.log(550 / 50)
MAX_OUTPUT = 50 * 1024**2


def need(condition, explanation):
    if not condition:
        raise RuntimeError(explanation)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_write(name, value):
    encoded = json.dumps(value, indent=2, allow_nan=False) + "\n"
    used = sum(p.stat().st_size for p in HERE.rglob("*") if p.is_file())
    need(used + len(encoded.encode()) < MAX_OUTPUT, "total output budget")
    with (HERE / name).open("x") as handle:
        handle.write(encoded)


def auc_rank(scores, positive):
    scores = np.asarray(scores, dtype=np.float64)
    positive = np.asarray(positive, dtype=bool)
    need(scores.ndim == 1 and scores.shape == positive.shape, "AUROC shape")
    need(np.isfinite(scores).all(), "AUROC finite scores")
    npos, nneg = int(positive.sum()), int((~positive).sum())
    need(npos > 0 and nneg > 0, "AUROC requires both classes")
    order = np.argsort(scores, kind="stable")
    ordered = scores[order]
    starts = np.r_[0, np.flatnonzero(ordered[1:] != ordered[:-1]) + 1]
    ends = np.r_[starts[1:], len(scores)]
    ranks = np.repeat((starts + ends + 1) / 2, ends - starts)
    rank_sum = float(ranks[positive[order]].sum())
    return (rank_sum - npos * (npos + 1) / 2) / (npos * nneg)


def auc_pairwise_fixture(scores, positive):
    """Independent literal pair enumeration, used only on fabricated fixtures."""
    positives = [float(s) for s, p in zip(scores, positive) if p]
    negatives = [float(s) for s, p in zip(scores, positive) if not p]
    return sum(1 if p > n else 0.5 if p == n else 0
               for p in positives for n in negatives) / (len(positives) * len(negatives))


def logsumexp(values):
    maxima = values.max(axis=1)
    return maxima + np.log(np.exp(values - maxima[:, None]).sum(axis=1))


def score_and_metrics(logits, labels):
    z = np.asarray(logits, dtype=np.float64)
    need(z.shape == (len(labels), 10) and np.isfinite(z).all(), "logit shape/domain")
    scores = z[:, 8] - logsumexp(z[:, COMMON])
    predictions = z.argmax(axis=1)
    ce = logsumexp(z) - z[np.arange(len(labels)), labels]
    per_class = []
    for digit in range(10):
        mask = labels == digit
        need(mask.any(), "missing class")
        per_class.append({"accuracy": float((predictions[mask] == digit).mean()),
                          "ce": float(ce[mask].mean())})
    result = {"rare_auroc": float(auc_rank(scores, labels == 8)),
              "rare_accuracy": per_class[8]["accuracy"],
              "rare_ce": per_class[8]["ce"],
              "common_accuracy": float(np.mean([per_class[d]["accuracy"] for d in COMMON])),
              "common_ce": float(np.mean([per_class[d]["ce"] for d in COMMON])),
              "balanced_accuracy": float(np.mean([p["accuracy"] for p in per_class])),
              "balanced_ce": float(np.mean([p["ce"] for p in per_class])),
              "common_predicted_8_rate": float((predictions[labels != 8] == 8).mean())}
    return scores, result


def fixtures():
    cases = [
        ("perfect", [0, 1, 2, 3], [False, False, True, True], 1.0),
        ("reversed", [3, 2, 1, 0], [False, False, True, True], 0.0),
        ("all_tied", [7, 7, 7, 7], [False, False, True, True], 0.5),
        ("mixed_ties", [0, 1, 1, 2, 2, 3], [False, False, True, False, True, True], 7 / 9),
        ("mixed_ties_permuted", [3, 1, 2, 0, 2, 1], [True, True, False, False, True, False], 7 / 9),
    ]
    checked = []
    for name, scores, positive, expected in cases:
        rank, pairwise = auc_rank(scores, positive), auc_pairwise_fixture(scores, positive)
        need(abs(rank - expected) < 1e-14 and abs(pairwise - expected) < 1e-14, name)
        need(auc_rank(np.array(scores) + SHIFT, positive) == rank, name + " shift")
        checked.append({"fixture": name, "scores": scores, "positive": positive,
                        "expected": expected, "rank": rank, "pairwise": pairwise})
    # Balanced fabricated ten-class panel: uniform predictions and a sole class shift.
    labels = np.arange(10, dtype=np.int64)
    z = np.zeros((10, 10), dtype=np.float64)
    _, before = score_and_metrics(z, labels)
    z[:, 8] += SHIFT
    _, after = score_and_metrics(z, labels)
    need(abs(before["rare_ce"] - math.log(10)) < 1e-14, "uniform CE")
    need(before["rare_accuracy"] == 0 and abs(before["common_accuracy"] - 1 / 9) < 1e-14, "argmax ties")
    need(after["rare_accuracy"] == 1 and after["common_accuracy"] == 0, "fixed shift argmax")
    need(abs(after["rare_ce"] - math.log(20 / 11)) < 1e-14, "shift rare CE")
    need(abs(after["common_ce"] - math.log(20)) < 1e-14, "shift common CE")
    need(before["rare_auroc"] == after["rare_auroc"] == 0.5, "uniform AUROC")
    # Large common offset must preserve finite CE and log-odds to float64 precision.
    s, m = score_and_metrics(z, labels)
    offset_s, offset_m = score_and_metrics(z + 10000, labels)
    need(np.max(abs(s - offset_s)) < 1e-11, "stable log-odds")
    need(max(abs(m[k] - offset_m[k]) for k in m) < 1e-11, "stable metrics")
    return {"status": "PASS", "rank_pairwise": checked,
            "analytical_multiclass_before": before, "analytical_multiclass_after": after,
            "large_offset_invariance": "PASS"}


def array_header(handle):
    version = np.lib.format.read_magic(handle)
    need(version in ((1, 0), (2, 0)), "unsupported NPY version")
    reader = (np.lib.format.read_array_header_1_0 if version == (1, 0)
              else np.lib.format.read_array_header_2_0)
    shape, fortran, dtype = reader(handle)
    need(not fortran and not dtype.hasobject, "NPY storage mode")
    return shape, dtype


def selected_logits(path):
    """Read exact rows from a ZIP_STORED NPY member, not other saved evaluations."""
    with zipfile.ZipFile(path) as archive:
        with archive.open("steps.npy") as member:
            steps = np.lib.format.read_array(member, allow_pickle=False)
        need(np.array_equal(steps, np.arange(0, 2001, 100)), "evaluation roster")
        info = archive.getinfo("heldout_unpatched.npy")
        need(info.compress_type == zipfile.ZIP_STORED, "uncompressed logits required")
        need(not (info.flag_bits & 1), "encrypted archive")
    with path.open("rb") as handle:
        handle.seek(info.header_offset)
        header = struct.unpack("<4s5H3I2H", handle.read(30))
        need(header[0] == b"PK\x03\x04", "ZIP local signature")
        filename = handle.read(header[9])
        need(filename == b"heldout_unpatched.npy", "ZIP member binding")
        handle.seek(header[10], 1)
        shape, dtype = array_header(handle)
        need(shape == (21, 5000, 10) and dtype == np.dtype("float32"), "saved logits shape/dtype")
        data_start = handle.tell()
        row_bytes = 5000 * 10 * dtype.itemsize
        result = {}
        for step in STEPS:
            index = int(np.flatnonzero(steps == step)[0])
            handle.seek(data_start + index * row_bytes)
            data = handle.read(row_bytes)
            need(len(data) == row_bytes, "complete selected slice")
            result[step] = np.frombuffer(data, dtype=dtype).reshape(5000, 10).copy()
    return result


def summary(values):
    values = [float(v) for v in values]
    return {"seed_values": values, "mean": float(np.mean(values)),
            "sample_sd": float(np.std(values, ddof=1)), "n_seeds": len(values)}


def execute():
    json_write("run-started.json", {"started_unix": time.time(), "pid": os.getpid(),
                                  "scientific_attempt": 1, "retry_permitted": False})
    fixture_result = fixtures()
    json_write("fixtures.json", fixture_result)
    completion_path = ACQUISITION / "complete.json"
    audit_path = ACQUISITION.parent / "audit-001/result.json"
    need(sha256(completion_path) == COMPLETION_SHA, "completion SHA")
    need(sha256(audit_path) == AUDIT_SHA, "accepted audit SHA")
    completion = json.loads(completion_path.read_text())
    audit = json.loads(audit_path.read_text())
    need(completion["status"] == "complete" and audit["status"] == "PASS", "accepted completion status")
    source_name = "experiments/spectral_selectivity_boundary.py"
    source_sha = sha256(ROOT / source_name)
    need(source_sha == completion["source_pins"][source_name], "frozen acquisition source")
    receipts = {r["path"]: r for r in completion["receipts"]}
    need(len(receipts) == len(completion["receipts"]), "unique completion receipt paths")
    names = [f"plan-s{seed}.npz" for seed in SEEDS]
    names += [f"logits-s{seed}-{cell}-{policy}.npz"
              for seed in SEEDS for cell in CELLS for policy in POLICIES]
    verified = []
    for name in names:
        path = ACQUISITION / name
        expected = receipts[name]
        actual_sha, actual_size = sha256(path), path.stat().st_size
        need(actual_sha == expected["sha256"] and actual_size == expected["size_bytes"], name + " receipt")
        verified.append({"path": str(path), "size_bytes": actual_size, "sha256": actual_sha})
    # No numerical array is opened before ALL input hashes have passed.
    pre_read = {"status": "ALL_INPUT_HASHES_PASS_BEFORE_ARRAY_READ", "unix": time.time(),
                "completion_sha256": COMPLETION_SHA, "audit_sha256": AUDIT_SHA,
                "design_sha256": sha256(HERE / "design.md"), "analysis_source_sha256": sha256(__file__),
                "acquisition_source_sha256": source_sha, "verified_inputs": verified}
    json_write("input-verification.json", pre_read)
    rows, warmups, label_provenance = [], {}, []
    index = {}
    for seed in SEEDS:
        with np.load(ACQUISITION / f"plan-s{seed}.npz", allow_pickle=False) as plan:
            labels = plan["heldout_true"]
            ids = plan["heldout_ids"]
        need(labels.shape == (5000,) and labels.dtype.kind in "iu", "label shape/dtype")
        need(np.array_equal(np.bincount(labels, minlength=10), np.full(10, 500)), "balanced label counts")
        need(ids.shape == (5000,) and len(np.unique(ids)) == 5000, "heldout ID integrity")
        label_provenance.append({"seed": seed, "class_counts": np.bincount(labels).tolist(),
                                 "labels_bytes_sha256": hashlib.sha256(labels.tobytes()).hexdigest(),
                                 "ids_bytes_sha256": hashlib.sha256(ids.tobytes()).hexdigest()})
        for cell in CELLS:
            for policy in POLICIES:
                name = f"logits-s{seed}-{cell}-{policy}.npz"
                selected = selected_logits(ACQUISITION / name)
                if seed not in warmups:
                    warmups[seed] = selected[100].copy()
                need(np.array_equal(warmups[seed], selected[100]), "identical within-seed warmup")
                for step in STEPS:
                    original = selected[step].astype(np.float64)
                    adjusted = original.copy()
                    adjusted[:, 8] += SHIFT
                    original_scores, original_metrics = score_and_metrics(original, labels)
                    adjusted_scores, adjusted_metrics = score_and_metrics(adjusted, labels)
                    need(np.max(abs(adjusted_scores - original_scores - SHIFT)) < 1e-12, "fixed score shift")
                    need(original_metrics["rare_auroc"] == adjusted_metrics["rare_auroc"], "AUROC shift invariance")
                    for view, metrics in (("original", original_metrics), ("plus_log11", adjusted_metrics)):
                        row = {"seed": seed, "cell": cell, "policy": policy, "step": step,
                               "view": view, "metrics": metrics, "source": name}
                        rows.append(row)
                        index[(seed, cell, policy, step, view)] = metrics
    need(len(rows) == 144, "full logical row roster")
    metric_names = tuple(rows[0]["metrics"])
    summaries, changes, contrasts = [], [], []
    for cell in CELLS:
        for policy in POLICIES:
            for view in ("original", "plus_log11"):
                for step in STEPS:
                    summaries.append({"cell": cell, "policy": policy, "step": step, "view": view,
                        "metrics": {m: summary([index[(s, cell, policy, step, view)][m] for s in SEEDS])
                                    for m in metric_names}})
                changes.append({"cell": cell, "policy": policy, "view": view,
                    "contrast": "endpoint2000_minus_warmup100",
                    "metrics": {m: summary([index[(s, cell, policy, 2000, view)][m] -
                                            index[(s, cell, policy, 100, view)][m] for s in SEEDS])
                                for m in metric_names}})
        for control in ("raw", "norm_raw"):
            for view in ("original", "plus_log11"):
                contrasts.append({"cell": cell, "view": view, "step": 2000,
                    "contrast": "native32_minus_" + control,
                    "metrics": {m: summary([index[(s, cell, "native32", 2000, view)][m] -
                                            index[(s, cell, control, 2000, view)][m] for s in SEEDS])
                                for m in metric_names}})
    result = {"schema": "spectral_rare_saved_logit_posthoc_v1", "status": "COMPLETE",
              "evidence": "POST-HOC saved-logit analysis; three paired seeds; no new trained policy",
              "design_sha256": pre_read["design_sha256"], "seed_order": list(SEEDS),
              "fixed_adjustment": {"class": 8, "additive_logit_shift": SHIFT,
                                   "multiplier": 11, "basis": "550/50 true-label counts, balanced heldout"},
              "rows": rows, "summaries": summaries, "endpoint_minus_warmup": changes,
              "native_minus_controls": contrasts}
    json_write("metrics.json", result)
    elapsed = time.monotonic() - START
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    need(elapsed < 120 and rss <= 1024**3, "runtime/memory envelope")
    receipt = {"status": "COMPLETE", "scientific_attempts": 1,
               "elapsed_seconds": elapsed, "peak_rss_bytes": rss,
               "address_space_limit_bytes": 1024**3, "wall_alarm_seconds": 120,
               "cpu_affinity": sorted(os.sched_getaffinity(0)),
               "thread_environment": {k: os.environ[k] for k in THREAD_KEYS},
               "numpy_version": np.__version__, "python_version": sys.version,
               "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
               "source_commit_created": False, "acquisition": str(ACQUISITION),
               "completion_sha256": COMPLETION_SHA, "accepted_audit_sha256": AUDIT_SHA,
               "analysis_source_sha256": sha256(__file__), "design_sha256": pre_read["design_sha256"],
               "metrics_sha256": sha256(HERE / "metrics.json"),
               "fixture_sha256": sha256(HERE / "fixtures.json"),
               "input_verification_sha256": sha256(HERE / "input-verification.json"),
               "heldout_label_provenance": label_provenance,
               "logical_trajectories": 36, "saved_logit_slices": 72,
               "metric_rows_including_two_views": 144,
               "warmup_check": "all 12 slices within each seed exactly equal",
               "array_members_read": ["steps.npy", "heldout_unpatched.npy rows for 100 and 2000 only",
                                      "heldout_true.npy", "heldout_ids.npy"],
               "new_output_bytes_before_receipt": sum(p.stat().st_size for p in HERE.rglob("*") if p.is_file()),
               "limitations": ["post-hoc analysis on existing heldout data",
                               "fixed transform is not a trained policy or fitted/calibrated correction",
                               "no causal identification of training-prior mismatch",
                               "noise, finite training and model misspecification remain",
                               "three seeds and one rare digit with absent-warmup exposure"]}
    json_write("provenance.json", receipt)
    print(json.dumps({"status": "COMPLETE", "metrics": str(HERE / "metrics.json"),
                      "elapsed_seconds": elapsed, "peak_rss_bytes": rss,
                      "metrics_sha256": receipt["metrics_sha256"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--fixtures-only", action="store_true")
    mode.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.fixtures_only:
        print(json.dumps(fixtures(), indent=2))
    else:
        execute()
