#!/usr/bin/env python3
"""Exclusive, bounded I19 acquisition. No CLI resume or membership overrides."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import re
import resource
import shutil
import subprocess
import sys
import time

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
SEEDS = tuple(range(19000, 19032))
PROCESS_VARIANCES = (0.0, .01, .1)
ROTATIONS = (0.0, math.pi / 4)
HORIZON = 4000
SECONDS = 1600
ARRAY_LIMIT = 2 * 1024 ** 3
ROOT_LIMIT = 3 * 1024 ** 3
SOURCE_PATHS = (
    REPO / "spectral_filter.py", HERE / "stochastic_tracking_core.py",
    HERE / "test_stochastic_tracking_core.py", HERE / "run_stochastic_tracking.py",
    HERE / "test_stochastic_tracking_runner.py", HERE / "protocol.md",
    HERE / "best-practices-check.md",
)


def need(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_once(path, value):
    # Validate serialization before exclusive creation; never emit NaN/Infinity.
    content = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with Path(path).open("x") as handle:
        handle.write(content)


def utc():
    return datetime.now(timezone.utc).isoformat()


def source_manifest(commit):
    need(re.fullmatch(r"[0-9a-f]{40}", commit or "") is not None,
         "Full frozen commit required")
    records = []
    for path in SOURCE_PATHS:
        name = str(path.relative_to(REPO))
        frozen = subprocess.check_output(["git", "show", commit + ":" + name], cwd=REPO)
        digest = sha(path)
        need(hashlib.sha256(frozen).hexdigest() == digest, "Changed source: " + name)
        records.append({"path": name, "size": path.stat().st_size, "sha256": digest})
    return records


def configure():
    need(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CUDA must be hidden")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        need(os.environ.get(name) == "1", "One numerical thread required: " + name)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)


def load_core():
    sys.path.insert(0, str(REPO))
    spec = importlib.util.spec_from_file_location("i19_tracking_core", HERE / "stochastic_tracking_core.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    import spectral_filter
    need(Path(spectral_filter.__file__).resolve() == REPO / "spectral_filter.py",
         "Native source import differs")
    return module


def cell_id(seed, process_index, rotation_index):
    return f"{seed}-p{process_index}-r{rotation_index}"


def expected_cells():
    return [(seed, di, ri) for seed in SEEDS for di in range(3) for ri in range(2)]


def write_stream(root, identity, arrays, metadata):
    need(re.fullmatch(r"[0-9]+-p[0-2]-r[0-1]", identity) is not None,
         "Invalid stream identity")
    need(isinstance(arrays, dict) and arrays, "Missing numeric arrays")
    for key, value in arrays.items():
        need(re.fullmatch(r"[a-zA-Z][a-zA-Z0-9_]*", key) is not None, "Invalid array key")
        need(isinstance(value, np.ndarray) and value.dtype.kind in "fibu"
             and np.isfinite(value).all(), "Nonfinite/non-numeric array: " + key)
    directory = root / "streams"
    npz = directory / (identity + ".npz")
    meta = directory / (identity + ".json")
    need(not npz.exists() and not meta.exists(), "Consumed stream identity: " + identity)
    # Validate metadata before writing the array file. Both files remain exclusive.
    json.dumps(metadata, allow_nan=False)
    with npz.open("xb") as handle:
        np.savez(handle, **arrays)
    json_once(meta, metadata)
    return {"id": identity, "array": {"path": str(npz.relative_to(root)),
            "size": npz.stat().st_size, "sha256": sha(npz)},
            "metadata": {"path": str(meta.relative_to(root)),
            "size": meta.stat().st_size, "sha256": sha(meta)}}


def preflight_root(root):
    need(root.is_absolute() and root.resolve() == root and not root.is_symlink(),
         "Absolute, resolved, nonsymlink root required")
    need(root.parent == Path("/tmp/spectral-experiment-artifacts")
         and root.name.startswith("spectral-i19-001."), "Unexpected root")
    need(os.path.ismount("/private-artifacts/storage"), "Large volume not mounted")
    need(root.stat().st_dev == Path("/private-artifacts/storage").stat().st_dev,
         "Root not on intended mounted volume")
    need(root.is_dir() and not any(root.iterdir()), "Root must exist and be empty")
    need(shutil.disk_usage(root).free >= 10 * 1024 ** 3, "Less than 10GiB free")


def acquire(root, commit):
    started = time.monotonic()
    sources = source_manifest(commit)
    configure()
    core = load_core()
    preflight_root(root)
    attempt = {"schema": "i19_tracking_attempt_v1", "created_utc": utc(),
               "root": str(root), "frozen_commit": commit, "sources": sources,
               "seeds": list(SEEDS), "process_variances": list(PROCESS_VARIANCES),
               "rotations": list(ROTATIONS), "horizon": HORIZON,
               "cooperative_seconds": SECONDS, "array_limit_bytes": ARRAY_LIMIT,
               "root_limit_bytes": ROOT_LIMIT, "pid": os.getpid(),
               "service_invocation_id": os.environ.get("INVOCATION_ID"),
               "python": platform.python_version(), "numpy": np.__version__,
               "torch": torch.__version__, "cpu_threads": torch.get_num_threads(),
               "platform": platform.platform(), "cuda_visible_devices": "",
               "paid_spend_usd": 0, "paid_reserved_usd": 0}
    json_once(root / "attempt.json", attempt)
    inventory = []
    failure = None
    array_bytes = 0
    total_bytes = (root / "attempt.json").stat().st_size

    def check_budget():
        need(time.monotonic() - started < SECONDS, "Cooperative time limit reached")

    try:
        (root / "streams").mkdir()
        for seed in SEEDS:
            check_budget()
            generator = torch.Generator(device="cpu").manual_seed(seed)
            canonical = torch.randn((HORIZON, 3), generator=generator, dtype=torch.float64)
            canonical_sha = hashlib.sha256(canonical.numpy().tobytes(order="C")).hexdigest()
            for di, process_variance in enumerate(PROCESS_VARIANCES):
                for ri, rotation in enumerate(ROTATIONS):
                    identity = cell_id(seed, di, ri)
                    cell_start = time.monotonic()
                    arrays, metadata = core.run_stream(canonical, process_variance, rotation,
                                                       check_budget=check_budget)
                    check_budget()
                    need(hashlib.sha256(canonical.numpy().tobytes(order="C")).hexdigest()
                         == canonical_sha, "Shared canonical input mutated")
                    predicted_bytes = sum(value.nbytes for value in arrays.values())
                    need(array_bytes + predicted_bytes + 1024 * 1024 < ARRAY_LIMIT,
                         "Array storage cap reached")
                    metadata = {"schema": "i19_tracking_stream_v1", "id": identity,
                                "seed": seed, "process_index": di, "rotation_index": ri,
                                "process_variance": process_variance, "rotation": rotation, "horizon": HORIZON,
                                "canonical_sha256": canonical_sha, "frozen_commit": commit,
                                "elapsed_seconds": time.monotonic() - cell_start,
                                "core": metadata}
                    row = write_stream(root, identity, arrays, metadata)
                    inventory.append(row)
                    array_bytes += row["array"]["size"]
                    total_bytes += row["array"]["size"] + row["metadata"]["size"]
                    need(array_bytes <= ARRAY_LIMIT and total_bytes < ROOT_LIMIT - 1024 ** 2,
                         "Actual storage cap reached")
                    print(json.dumps({"event": "stream_complete", "id": identity,
                                      "completed": len(inventory),
                                      "elapsed_seconds": time.monotonic() - started}), flush=True)
        # Final source pin catches edits during acquisition. No source replay.
        need(source_manifest(commit) == sources, "Source changed during acquisition")
    except Exception as exc:
        failure = {"type": type(exc).__name__, "message": str(exc)}
    completion = {"schema": "i19_tracking_completion_v1",
                  "status": "complete" if failure is None else "failed",
                  "created_utc": utc(), "frozen_commit": commit,
                  "attempt_sha256": sha(root / "attempt.json"),
                  "expected_streams": len(expected_cells()), "completed_streams": len(inventory),
                  "completed_observations": len(inventory) * HORIZON,
                  "elapsed_seconds": time.monotonic() - started,
                  "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  "array_bytes": array_bytes, "root_bytes_before_completion": total_bytes,
                  "failure": failure, "streams": inventory}
    json_once(root / "completion.json", completion)
    print(json.dumps({key: value for key, value in completion.items() if key != "streams"}), flush=True)
    return 0 if failure is None else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--frozen-commit", required=True)
    args = parser.parse_args()
    return acquire(args.root, args.frozen_commit)


if __name__ == "__main__":
    raise SystemExit(main())
