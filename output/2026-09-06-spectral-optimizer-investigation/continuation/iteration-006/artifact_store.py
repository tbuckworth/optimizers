"""Iteration006 exclusive storage, finite metadata and cooperative resource guards."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import resource
import shutil
import subprocess
import tempfile
import time

import numpy as np
import torch

GIB = 1024 ** 3
BIG = Path("/tmp/spectral-experiment-artifacts")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 ** 2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path, **metadata):
    path = Path(path).resolve()
    return {"path": str(path), "size_bytes": path.stat().st_size,
            "sha256": sha256(path), **metadata}


def verify_artifact(item):
    path = Path(item["path"])
    if path.stat().st_size != item["size_bytes"] or sha256(path) != item["sha256"]:
        raise AssertionError(f"Artifact changed: {path}")


def safe_context(value):
    if isinstance(value, torch.Tensor):
        return {"tensor_omitted": True, "shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, np.ndarray):
        return {"array_omitted": True, "shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, float) and not math.isfinite(value):
        return {"rejected_nonfinite": repr(value)}
    if isinstance(value, dict):
        return {str(k): safe_context(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_context(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def json_bytes(value):
    return (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()


def verify_large_mount():
    resolved = BIG.resolve()
    mount = json.loads(subprocess.check_output(
        ["findmnt", "-J", "-T", str(resolved), "-o", "TARGET,SOURCE,UUID"], text=True))["filesystems"][0]
    if mount["target"] != "/private-artifacts/storage" or not (
            mount["source"] == "/dev/RECONFIGURE_FOR_LOCAL_STORAGE" or mount.get("uuid") == "00000000-0000-4000-8000-000000000000"):
        raise RuntimeError("Bulk directory is not on the documented large-volume mount")
    return mount


class Metadata:
    """Only replace progress files created by this object in its new directory."""
    def __init__(self, directory):
        self.root = Path(directory).resolve()
        self.root.mkdir(exist_ok=False)
        self.owned = set()

    def write(self, name, value):
        path = (self.root / name).resolve()
        if path.parent != self.root or (path.exists() and name not in self.owned):
            raise FileExistsError("Metadata path escape or unowned overwrite")
        payload = json_bytes(value)
        current = sum(p.stat().st_size for p in self.root.iterdir() if p.is_file())
        if current + len(payload) > 20 * 1024 ** 2:
            raise RuntimeError("Small metadata budget exceeded")
        pending = path.with_name(path.name + ".pending")
        with pending.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        pending.replace(path)
        self.owned.add(name)
        return artifact(path)


class Store:
    def __init__(self, mode, *, synthetic_parent=None):
        # The override is for tiny CPU storage tests only; no launcher exposes it.
        if synthetic_parent is None:
            self.mount = verify_large_mount()
            parent = BIG
            if shutil.disk_usage(parent).free < 2 * GIB:
                raise RuntimeError("Insufficient large-volume disk headroom")
        else:
            parent = Path(synthetic_parent)
            self.mount = {"synthetic_cpu_test": True}
        self.root = Path(tempfile.mkdtemp(prefix=f"spectral-iteration006-{mode}-", dir=parent)).resolve()
        self.artifacts = []

    def bytes_used(self):
        return sum(p.stat().st_size for p in self.root.iterdir() if p.is_file())

    def capacity(self, additional=0):
        if self.bytes_used() + additional > GIB:
            raise RuntimeError("One-GiB artifact budget exceeded")
        if shutil.disk_usage(self.root).free < GIB + additional:
            raise RuntimeError("Bulk free-space headroom failed")

    def path(self, name, estimate=0):
        path = (self.root / name).resolve()
        if path.parent != self.root or path.exists():
            raise FileExistsError("Bulk path escape or overwrite")
        self.capacity(estimate)
        return path

    def finish(self, path, **metadata):
        self.capacity()
        item = artifact(path, **metadata)
        self.artifacts.append(item)
        return item

    def json(self, name, value, **metadata):
        payload = json_bytes(value)
        path = self.path(name, len(payload))
        with path.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        return self.finish(path, **metadata)

    def plan(self, name, plan):
        values = {k: np.asarray(v) for k, v in plan.items()}
        if any(v.dtype.hasobject for v in values.values()):
            raise TypeError("Plan must contain numerical arrays only")
        path = self.path(name, sum(v.nbytes for v in values.values()) + 16384)
        with path.open("xb") as stream:
            np.savez_compressed(stream, **values)
            stream.flush()
            os.fsync(stream.fileno())
        return self.finish(path, seed=int(plan["seed"]))

    def checkpoints(self, name, states, **metadata):
        total = 0
        for state in states.values():
            if not isinstance(state, dict) or not state:
                raise TypeError("Checkpoint needs a nonempty tensor state dictionary")
            for key, value in state.items():
                if not isinstance(key, str) or not isinstance(value, torch.Tensor) or value.device.type != "cpu":
                    raise TypeError("Checkpoint bundles must contain CPU tensor state dictionaries only")
                if not bool(torch.isfinite(value).all()):
                    raise ValueError("Nonfinite checkpoint")
                total += value.numel() * value.element_size()
        path = self.path(name, total + 2 * 1024 ** 2)
        with path.open("xb") as stream:
            torch.save(states, stream)
            stream.flush()
            os.fsync(stream.fileno())
        return self.finish(path, **metadata)


class Guard:
    def __init__(self, pilot, device="cuda"):
        self.started = time.perf_counter()
        self.limit = 180 if pilot else 900
        self.device = str(device)

    def observe(self):
        cuda = self.device.startswith("cuda") and torch.cuda.is_initialized()
        return {"elapsed_seconds": time.perf_counter() - self.started,
                "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated() if cuda else 0,
                "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved() if cuda else 0}

    def check(self):
        row = self.observe()
        if row["elapsed_seconds"] > self.limit:
            raise TimeoutError("Prospective cooperative wall-time limit exceeded")
        if row["peak_rss_bytes"] > 12 * GIB:
            raise MemoryError("Prospective 12-GiB RSS limit exceeded")
        if row["peak_gpu_allocated_bytes"] > 8 * GIB:
            raise MemoryError("Prospective 8-GiB allocated GPU limit exceeded")
        return row


def preserve_failure(metadata, manifest, context, error, store=None):
    """Keep completed evidence and a bounded, finite failure record, never retry."""
    partial = {}
    if store is not None:
        try:
            partial["partial_cell"] = store.json("partial-cell.json", safe_context({
                k: v for k, v in context.items() if k != "partial_checkpoint_tensors"}))
            checkpoints = context.get("partial_checkpoint_tensors", {})
            if checkpoints:
                partial["partial_checkpoints"] = store.checkpoints("partial-checkpoints.pt", checkpoints)
        except Exception as secondary:
            partial["preservation_error"] = repr(secondary)
    compact = {k: v for k, v in context.items() if not k.startswith("partial_")}
    compact.update(partial, error_type=type(error).__name__, error=str(error),
                   bulk_root=None if store is None else str(store.root),
                   partial_row_count=len(context.get("partial_rows", [])),
                   partial_validation_count=len(context.get("partial_validations", [])))
    failure = metadata.write("failure.json", safe_context(compact))
    manifest.update(status="failed", all_gates_passed=False, failure=failure)
    metadata.write("execution.json", safe_context(manifest))
