"""Scoped large-volume storage and finite, recoverable progress records."""
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

BIG = Path("/tmp/spectral-experiment-artifacts")
GIB = 1024 ** 3


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 ** 2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_safe(value):
    """Failure context cannot itself fail because its rejected scalar was NaN."""
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite_rejected": repr(value)}
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path, value):
    path = Path(path)
    # Atomic replace only our own progress file, never another attempt directory.
    payload = json.dumps(value, indent=2, allow_nan=False) + "\n"
    if sum(p.stat().st_size for p in path.parent.glob("*.json")) + len(payload) > 20 * 1024 ** 2:
        raise RuntimeError("Metadata budget exceeded")
    pending = path.with_name(path.name + ".pending")
    with pending.open("w") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    pending.replace(path)


def verify_large_mount(path=BIG):
    resolved = Path(path).resolve()
    if not resolved.is_relative_to(BIG.resolve()):
        raise RuntimeError("Bulk path must be inside verified big/tmp")
    mounted = json.loads(subprocess.check_output(
        ["findmnt", "-J", "-T", str(resolved), "-o", "TARGET,SOURCE,UUID"], text=True))["filesystems"][0]
    if mounted["target"] != "/private-artifacts/storage" or not (
            mounted["source"] == "/dev/RECONFIGURE_FOR_LOCAL_STORAGE" or mounted.get("uuid") == "00000000-0000-4000-8000-000000000000"):
        raise RuntimeError("Bulk path is not on the documented large-volume mount")
    return mounted


def artifact(path, **metadata):
    path = Path(path)
    return {"path": str(path.resolve()), "size_bytes": path.stat().st_size,
            "sha256": sha256(path), **metadata}


def tensor_inventory(value, prefix="root"):
    """No tensor values in metadata; nested leaves retain shape/dtype semantics."""
    if isinstance(value, torch.Tensor):
        return {prefix: {"shape": list(value.shape), "dtype": str(value.dtype)}}
    if isinstance(value, dict):
        return {key: item for name, child in value.items()
                for key, item in tensor_inventory(child, f"{prefix}.{name}").items()}
    if isinstance(value, (list, tuple)):
        return {key: item for i, child in enumerate(value)
                for key, item in tensor_inventory(child, f"{prefix}[{i}]").items()}
    return {}


class Store:
    def __init__(self, mode):
        self.mount = verify_large_mount()
        if shutil.disk_usage(BIG).free < 10 * GIB:
            raise RuntimeError("Insufficient free space on large volume")
        self.root = Path(tempfile.mkdtemp(prefix=f"spectral-iteration005-{mode}-", dir=BIG))
        self.maps = {}
        self.completed_rows = {}
        self.durable_rows = {}

    def path(self, name, additional_bytes=0):
        path = (self.root / name).resolve()
        if not path.is_relative_to(self.root) or path.exists():
            raise RuntimeError("Attempt overwrite or path escape")
        self.check_capacity(additional_bytes)
        return path

    def check_capacity(self, additional_bytes=0):
        size = sum(p.stat().st_size for p in self.root.iterdir() if p.is_file())
        if size + additional_bytes > 6 * GIB:
            raise RuntimeError("Persistent bulk artifact budget exceeded")
        if shutil.disk_usage(self.root).free < 10 * GIB + additional_bytes:
            raise RuntimeError("Large-volume headroom gate failed")

    def stream(self, name, shape):
        path = self.path(name, int(np.prod(shape)) * 4 + 256)
        value = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=shape)
        self.maps[name], self.completed_rows[name] = value, 0
        self.durable_rows[name] = 0
        return value

    def append(self, name, row, value):
        if row != self.completed_rows[name]:
            raise RuntimeError("Nonsequential stream append")
        self.maps[name][row] = value
        self.completed_rows[name] = row + 1

    def flush(self):
        for name, value in self.maps.items():
            value.flush()
            self.durable_rows[name] = self.completed_rows[name]

    def stream_artifacts(self):
        self.flush()
        return [artifact(self.root / name, shape=list(value.shape), dtype=str(value.dtype),
                         axes=["chronological_step", "flat_parameter"], completed_rows=self.completed_rows[name])
                for name, value in self.maps.items()]

    def array(self, name, value, **metadata):
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().contiguous().numpy()
        path = self.path(name, value.nbytes + 256)
        with path.open("xb") as stream:
            np.save(stream, value, allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        return artifact(path, shape=list(value.shape), dtype=str(value.dtype), **metadata)

    def tensor_bundle(self, name, value, expected_bytes=0, **metadata):
        path = self.path(name, expected_bytes)
        with path.open("xb") as stream:
            torch.save(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
        self.check_capacity()
        return artifact(path, tensor_inventory=tensor_inventory(value), **metadata)


class Guard:
    def __init__(self, pilot):
        self.started = time.perf_counter()
        self.limit = 600 if pilot else 900
        self.pilot = pilot

    def check(self):
        elapsed = time.perf_counter() - self.started
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        if elapsed > self.limit:
            raise TimeoutError("Prospective elapsed-time cap exceeded")
        if rss > 12 * GIB:
            raise MemoryError("Prospective peak RSS gate exceeded")
        gpu = torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else 0
        if gpu > 8 * GIB:
            raise MemoryError("Prospective allocated GPU-memory gate exceeded")
        return {"elapsed_seconds": elapsed, "peak_rss_bytes": rss,
                "peak_gpu_allocated_bytes": gpu,
                "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved() if torch.cuda.is_initialized() else 0}


def preserve_failure(output, manifest, context, error, store=None):
    flushing_error = None
    if store is not None:
        try:
            store.flush()
        except Exception as exc:
            flushing_error = repr(exc)
    failed = {"error": repr(error), "context": json_safe(context), "flush_error": flushing_error,
              "bulk_root": None if store is None else str(store.root),
              "completed_stream_rows": {} if store is None else dict(store.completed_rows),
              "durable_stream_rows": {} if store is None else dict(store.durable_rows)}
    write_json(Path(output) / "failure.json", failed)
    # Do not duplicate a large partial context in both failure and execution JSON.
    manifest.update(status="failed", all_gates_passed=False,
                    failure=artifact(Path(output) / "failure.json"))
    write_json(Path(output) / "execution.json", json_safe(manifest))
