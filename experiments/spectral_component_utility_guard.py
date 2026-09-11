"""Inert runtime/IO controls for the one component-utility acquisition.

Importing this module does not configure torch, inspect resources, or touch an
archive. The caller owns admission, source/input binding, and output creation.
CPU mode is solely for fabricated tests, not an alternate scientific runner.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import resource
import shutil
import stat
import subprocess
import time
import zipfile

import numpy as np
import torch

UNIT = "spectral-component-utility-001.service"
MAX_BYTES = 8 * 1024**3
RESERVE_BYTES = 1024**2
JSON_MAX_BYTES = 16 * 1024**2
NPZ_MAX_BYTES = 256 * 1024**2
DEADLINE_SECONDS = 1200
HOST_MAX_BYTES = 8 * 1024**3
GPU_MAX_BYTES = 4 * 1024**3
GPU_MIN_FREE_BYTES = 8 * 1024**3
DISK_MIN_FREE_BYTES = 1024**3
THREAD_ENV = dict.fromkeys(
    ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"), "1"
)
SERVICE_KEYS = (
    "Type", "RuntimeMaxUSec", "Restart", "KillMode", "MainPID",
    "InvocationID", "ActiveState", "SubState", "ControlGroup",
)


class GuardError(RuntimeError):
    pass


def need(condition, message):
    if not condition:
        raise GuardError(message)


def _direct_name(name, suffix):
    need(type(name) is str and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,159}", name)
         is not None and name.endswith(suffix), "bounded direct-child artifact name/extension")


def _json_value(value, depth=0, ancestors=None):
    """Reject coercions, nonfinite scalars and cycles before starting a file."""
    need(depth <= 40, "JSON nesting limit")
    if value is None or type(value) in (bool, int):
        if type(value) is int:
            need(value.bit_length() <= 256, "JSON integer size")
        return
    if type(value) is float:
        need(math.isfinite(value), "finite JSON float required")
        return
    if type(value) is str:
        need(len(value) <= JSON_MAX_BYTES, "JSON string size")
        return
    need(type(value) in (dict, list), "plain JSON values required")
    ancestors = set() if ancestors is None else ancestors
    need(id(value) not in ancestors, "cyclic JSON")
    need(len(value) <= 1_000_000, "JSON container size")
    ancestors.add(id(value))
    try:
        if type(value) is dict:
            for key, item in value.items():
                need(type(key) is str and len(key) <= 1024, "plain bounded JSON keys")
                _json_value(item, depth + 1, ancestors)
        else:
            for item in value:
                _json_value(item, depth + 1, ancestors)
    finally:
        ancestors.remove(id(value))


def _npz_value(value):
    need(type(value) is dict and 1 <= len(value) <= 64, "bounded plain NPZ dictionary")
    size = 0
    for key, array in value.items():
        need(type(key) is str and re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,63}", key)
             is not None and key != "file", "plain NPZ key")
        need(type(array) is np.ndarray and array.dtype.fields is None
             and array.dtype.subdtype is None and array.dtype.metadata is None and array.dtype.kind in "biuf"
             and array.ndim <= 8, "plain numeric NumPy array required")
        size += array.nbytes
        need(size <= NPZ_MAX_BYTES, "NPZ uncompressed payload limit")
        if array.dtype.kind == "f":
            need(bool(np.isfinite(array).all()), "finite NPZ values required")


class _BudgetWriter:
    """Unseekable ZIP-compatible stream; every physical byte consumes budget."""
    def __init__(self, handle, allowance, check, account):
        self.handle, self.allowance = handle, allowance
        self.check, self.account = check, account

    def write(self, value):
        self.check()
        size = memoryview(value).nbytes
        need(size <= self.allowance, "artifact or whole-archive byte cap")
        count = self.handle.write(value)
        need(type(count) is int and 0 <= count <= size, "invalid write count")
        self.account(count)
        self.allowance -= count
        need(count == size, "short artifact write")
        return count

    def flush(self):
        return self.handle.flush()

    def read(self, *args):
        # NumPy's file-like recognition checks for read and write attributes.
        return self.handle.read(*args)


class Run:
    """Exclusive bounded writer over a caller-created, empty archive directory."""
    def __init__(self, path, device="cuda:0"):
        self.path = Path(path)
        need(device in ("cuda:0", "cpu"), "explicit cuda:0 or fabricated CPU mode")
        need(self.path.is_absolute() and self.path.resolve() == self.path
             and not self.path.is_symlink() and self.path.is_dir(), "absolute real output directory")
        self.device = device
        self._fd = os.open(self.path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            self._identity = os.fstat(self._fd)
            need(not os.listdir(self._fd), "output directory must be unused and empty")
        except BaseException:
            os.close(self._fd)
            self._fd = None
            raise
        self.started = time.monotonic()
        self._last_check = None
        self._used = 0
        self._receipts = []
        self.serialization_seconds = 0.0
        self._failed = False
        self._failure_attempted = False

    @property
    def used(self):
        return self._used

    @property
    def receipts(self):
        return copy.deepcopy(self._receipts)

    def close(self):
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def __del__(self):
        if getattr(self, "_fd", None) is not None:
            os.close(self._fd)

    def _directory_ok(self):
        need(self._fd is not None, "closed archive")
        current = self.path.lstat()
        need(stat.S_ISDIR(current.st_mode)
             and (current.st_dev, current.st_ino) == (self._identity.st_dev, self._identity.st_ino),
             "output directory identity changed")

    def check(self):
        try:
            self._check()
        except BaseException:
            self._failed = True
            raise

    def _check(self):
        need(not self._failed, "failed attempt cannot resume")
        self._directory_ok()
        now = time.monotonic()
        need(0 <= now - self.started < DEADLINE_SECONDS, "1200-second cooperative deadline")
        if self._last_check is None or now - self._last_check >= 1:
            need(shutil.disk_usage(self.path).free >= DISK_MIN_FREE_BYTES, "1GiB free-disk reserve")
            need(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 <= HOST_MAX_BYTES,
                 "8GiB host high-water RSS cap")
            if self.device == "cuda:0":
                need(torch.cuda.max_memory_allocated(device=self.device) <= GPU_MAX_BYTES,
                     "4GiB PyTorch allocator high-water cap")
            self._last_check = now

    def _account(self, count):
        self._used += count

    def _measure(self):
        entries = {}
        for name in os.listdir(self._fd):
            info = os.stat(name, dir_fd=self._fd, follow_symlinks=False)
            need(stat.S_ISREG(info.st_mode) and info.st_nlink == 1,
                 "archive contains a nonregular or aliased file")
            entries[name] = info.st_size
        self._used = sum(entries.values())
        return entries

    def _write(self, name, value, kind, allowance, check):
        initial_used = self._used
        fd = os.open(name, os.O_CREAT | os.O_EXCL | os.O_RDWR | os.O_NOFOLLOW, 0o600,
                     dir_fd=self._fd)
        with os.fdopen(fd, "w+b", buffering=0) as handle:
            writer = _BudgetWriter(handle, allowance, check, self._account)
            if kind == "npz":
                # Explicit contexts close partial ZIPs on error. np.savez in
                # NumPy 1.26 leaves its ZIP object unclosed when writing raises.
                with zipfile.ZipFile(writer, mode="w", compression=zipfile.ZIP_STORED,
                                     allowZip64=True) as archive:
                    for key, array in value.items():
                        with archive.open(key + ".npy", mode="w", force_zip64=True) as member:
                            np.lib.format.write_array(member, array, allow_pickle=False)
            else:
                for part in json.JSONEncoder(allow_nan=False, ensure_ascii=True,
                                             separators=(",", ":")).iterencode(value):
                    writer.write(part.encode("ascii"))
                writer.write(b"\n")
            writer.flush()
            os.fsync(handle.fileno())
            check()
            info = os.fstat(handle.fileno())
            linked = os.stat(name, dir_fd=self._fd, follow_symlinks=False)
            need(stat.S_ISREG(linked.st_mode) and info.st_nlink == 1
                 and (info.st_dev, info.st_ino) == (linked.st_dev, linked.st_ino)
                 and info.st_size == self._used - initial_used <= allowance,
                 "artifact identity changed")
            handle.seek(0)
            checksum = hashlib.sha256()
            while chunk := handle.read(1024**2):
                check()
                checksum.update(chunk)
        return {"path": name, "size_bytes": info.st_size, "sha256": checksum.hexdigest()}

    def save(self, name, value, kind="json"):
        began = time.monotonic()
        try:
            self.check()
            need(kind in ("json", "npz"), "JSON or numeric NPZ artifacts only")
            _direct_name(name, "." + kind)
            need(name != "failure.json", "failure.json is reserved")
            (_json_value if kind == "json" else _npz_value)(value)
            entries = self._measure()
            need(entries == {row["path"]: row["size_bytes"] for row in self._receipts},
                 "archive inventory changed outside this writer")
            allowance = min(JSON_MAX_BYTES if kind == "json" else NPZ_MAX_BYTES,
                            MAX_BYTES - RESERVE_BYTES - self._used)
            need(allowance > 0, "whole-archive byte cap")
            receipt = self._write(name, value, kind, allowance, self.check)
            self._receipts.append(receipt)
            return copy.deepcopy(receipt)
        except BaseException:
            self._failed = True
            raise
        finally:
            self.serialization_seconds += time.monotonic() - began

    def write_failure(self, exc):
        """Best-effort exclusive footer; never deletes partial files or resumes.

        Deliberately bypasses the expired cooperative/resource checks, but not
        directory identity, byte cap, or exclusivity. Hard kills/OOM/disk failure
        can still prevent this footer. No exception repr/traceback is persisted.
        """
        self._failed = True
        need(not self._failure_attempted, "failure footer already attempted")
        self._failure_attempted = True
        self._directory_ok()
        began = time.monotonic()
        try:
            self._measure()  # Include any partial write, even if flush raised.
            value = {"status": "failed", "error_type": type(exc).__name__[:128],
                     "message": str(exc)[:4096], "seconds": max(0.0, began - self.started),
                     "serialization_seconds": self.serialization_seconds,
                     "archive_bytes_before_footer": self._used, "receipts": self.receipts}
            _json_value(value)
            receipt = self._write("failure.json", value, "json",
                                  min(RESERVE_BYTES, MAX_BYTES - self._used), self._directory_ok)
            self._receipts.append(receipt)
            return copy.deepcopy(receipt)
        finally:
            self.serialization_seconds += time.monotonic() - began


def _output(command):
    return subprocess.check_output(command, text=True, timeout=10)


def _gpu_clients(text, pid):
    rows = []
    seen = set()
    for line in text.splitlines():
        if not line.strip():
            continue
        cells = [part.strip() for part in line.split(",")]
        need(len(cells) == 2 and all(re.fullmatch(r"[0-9]+", part) for part in cells),
             "GPU client query schema")
        other, mib = map(int, cells)
        need(other > 0 and other not in seen, "duplicate/invalid GPU client PID")
        seen.add(other)
        caps = {2101: 512, 8861: 128, pid: 4096}
        need(other in caps and mib <= caps[other], "unknown/oversized GPU client")
        rows.append({"pid": other, "memory_mib": mib})
    return sorted(rows, key=lambda row: row["pid"])


def configure(device="cuda:0"):
    """Fail closed on the exact admitted runtime; no launch/credential actions.

    Called explicitly only by the future admitted runner. Tests replace all
    process/cgroup/service/device interfaces with fabricated observations.
    """
    need(device == "cuda:0", "scientific runtime requires explicit cuda:0")
    need(str(torch.__version__) == "2.11.0+cu128" and np.__version__ == "1.26.4",
         "pinned torch/NumPy versions")
    need(all(os.environ.get(key) == value for key, value in THREAD_ENV.items()), "pinned thread environment")
    need(os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8", "pinned CUBLAS workspace")
    pid = os.getpid()
    invocation = os.environ.get("INVOCATION_ID", "")
    need(re.fullmatch(r"[0-9a-f]{32}", invocation) is not None, "systemd invocation ID")
    lines = Path("/proc/self/cgroup").read_text().splitlines()
    need(len(lines) == 1 and lines[0].startswith("0::/"), "unified cgroup v2 required")
    group = lines[0][3:]
    need(group.startswith("/") and "//" not in group
         and all(part not in (".", "..") for part in group.split("/"))
         and Path(group).name == UNIT, "exact service cgroup")
    cgroup = Path("/sys/fs/cgroup") / group.lstrip("/")
    effective = {key: (cgroup / key).read_text().strip()
                 for key in ("memory.max", "memory.swap.max", "cpu.max")}
    need(effective == {"memory.max": str(HOST_MAX_BYTES), "memory.swap.max": "0",
                       "cpu.max": "100000 100000"}, "exact 8GiB/no-swap/one-CPU cgroup caps")
    props = _output(["systemctl", "--user", "show", UNIT]
                    + ["--property=" + key for key in SERVICE_KEYS])
    service = {}
    for line in props.splitlines():
        key, sep, value = line.partition("=")
        need(sep == "=" and key in SERVICE_KEYS and key not in service, "systemd property schema")
        service[key] = value
    expected = {"Type": "exec", "RuntimeMaxUSec": "30min", "Restart": "no",
                "KillMode": "control-group", "MainPID": str(pid), "InvocationID": invocation,
                "ActiveState": "active", "SubState": "running", "ControlGroup": group}
    need(service == expected, "exact current service identity/properties")
    clients = _gpu_clients(_output(["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory",
                                   "--format=csv,noheader,nounits", "--id=0"]), pid)
    need(torch.cuda.is_available(), "CUDA availability")
    name = torch.cuda.get_device_name(device)
    need(name == "NVIDIA GeForce RTX 3090", "exact RTX 3090 target")
    free, total = torch.cuda.mem_get_info(device)
    need(type(free) is int and type(total) is int and total >= free >= GPU_MIN_FREE_BYTES,
         "at least 8GiB GPU free")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    fraction = GPU_MAX_BYTES / total
    torch.cuda.memory.set_per_process_memory_fraction(fraction, device=device)
    deterministic = {"thread_env": {key: os.environ[key] for key in THREAD_ENV},
                     "cublas_workspace": os.environ["CUBLAS_WORKSPACE_CONFIG"],
                     "intraop_threads": torch.get_num_threads(),
                     "interop_threads": torch.get_num_interop_threads(),
                     "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                     "cudnn_benchmark": torch.backends.cudnn.benchmark,
                     "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                     "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32}
    need(deterministic == {"thread_env": THREAD_ENV, "cublas_workspace": ":4096:8",
                           "intraop_threads": 1, "interop_threads": 1,
                           "deterministic_algorithms": True, "cudnn_benchmark": False,
                           "matmul_allow_tf32": False, "cudnn_allow_tf32": False},
         "deterministic settings readback")
    return {"unit": UNIT, "pid": pid, "invocation_id": invocation, "cgroup": group,
            "effective": effective, "service": service, "gpu_clients": clients,
            "preexisting_gpu_clients": [row for row in clients if row["pid"] != pid],
            "device": device, "device_name": name, "gpu_free_bytes_at_configure": free,
            "gpu_total_bytes": total, "gpu_allocator_limit_bytes": GPU_MAX_BYTES,
            "gpu_allocator_fraction": fraction, "torch_version": str(torch.__version__),
            "numpy_version": np.__version__, "deterministic": deterministic}
