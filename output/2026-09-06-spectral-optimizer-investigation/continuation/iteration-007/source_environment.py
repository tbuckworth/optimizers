"""Strict read-only source/runtime bindings; no launch or provenance proof.

``validate_sources`` and ``validate_environment`` are pure consistency checks.
The collectors observe a clean Git checkout or the current process without
changing RNG/settings. ``validate_state_environment`` compares saved core and
environment values only; it does not inspect or certify foreign hardware.
"""
from __future__ import annotations

import hashlib
import math
import os
import platform
from pathlib import Path
import random
import selectors
import stat
import subprocess
import sys
import time
from typing import Any

import numpy as np
import torch

import artifact_store as storage
from cuda_identity import canonical_cuda_uuid, is_canonical_cuda_uuid
# Re-export the existing schema API; collectors still import Torch, while the
# controller imports only source_environment_schema for the same validators.
from source_environment_schema import (
    SourceEnvironmentError, I7, SOURCE_FILE_CAP, SOURCE_TOTAL_CAP, METADATA_CAP,
    COMMAND_TIMEOUT, SOURCE_KEYS, SOURCE_FILE_KEYS, ENVIRONMENT_KEYS, PYTHON_KEYS,
    LIBRARY_KEYS, OS_KEYS, SAFE_ENV_KEYS, TORCH_KEYS, CUDA_KEYS, CUDA_DEVICE_KEYS,
    RNG_KEYS, PY_RNG_KEYS, NP_RNG_KEYS, TORCH_RNG_KEYS, CUDA_RNG_KEYS, PROFILES,
    FIXTURE_ROLES, SCIENTIFIC_ROLES, _ROLES, _SHA256, _OID, _require, _keys, _string,
    _integer, _canonical_absolute, _membership, _source_path, validate_sources,
    _optional_string, _bools, validate_environment,
)


def _signature(info: os.stat_result) -> tuple[int, ...]:
    return (info.st_dev, info.st_ino, info.st_mode, info.st_nlink, info.st_size,
            info.st_mtime_ns, info.st_ctime_ns)


def _directory_identity(info: os.stat_result) -> tuple[int, int, int]:
    """Stable identity only; child creation legitimately changes other stat fields."""
    return info.st_dev, info.st_ino, info.st_mode


def _canonical_root(path: Any) -> tuple[str, tuple[tuple[str, tuple[int, int, int]], ...]]:
    raw = os.fspath(path)
    _require(type(raw) is str and raw.isascii() and not raw.startswith("//"),
             "repository root must be a canonical ASCII path")
    absolute = os.path.abspath(raw)
    _require(absolute != "/" and not absolute.startswith("//") and
             os.path.normpath(absolute) == absolute,
             "repository root must be canonical and non-root")
    rows = []
    current = "/"
    for part in Path(absolute).parts[1:]:
        current = os.path.join(current, part)
        try:
            info = os.lstat(current)
        except OSError as exc:
            raise SourceEnvironmentError("repository path inspection failed") from exc
        _require(stat.S_ISDIR(info.st_mode) and not stat.S_ISLNK(info.st_mode),
                 "repository path component is not a real directory")
        rows.append((current, _directory_identity(info)))
    return absolute, tuple(rows)


def _check_root_snapshot(rows: tuple[tuple[str, tuple[int, int, int]], ...]) -> None:
    for path, expected in rows:
        try:
            actual = os.lstat(path)
        except OSError as exc:
            raise SourceEnvironmentError("repository path changed") from exc
        _require(stat.S_ISDIR(actual.st_mode) and not stat.S_ISLNK(actual.st_mode) and
                 _directory_identity(actual) == expected, "repository path changed")


def _run(command: list[str], *, cwd: str, cap: int) -> bytes:
    _require(type(cap) is int and 0 <= cap <= SOURCE_FILE_CAP, "invalid command output cap")
    env = dict(os.environ)
    env.update(LC_ALL="C", GIT_OPTIONAL_LOCKS="0")
    process = subprocess.Popen(command, cwd=cwd, stdin=subprocess.DEVNULL,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
    selector = selectors.DefaultSelector()
    buffers = {process.stdout: bytearray(), process.stderr: bytearray()}
    selector.register(process.stdout, selectors.EVENT_READ)
    selector.register(process.stderr, selectors.EVENT_READ)
    deadline = time.monotonic() + COMMAND_TIMEOUT
    try:
        while selector.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise SourceEnvironmentError("bounded command timed out")
            events = selector.select(remaining)
            if not events and process.poll() is None:
                continue
            for key, _ in events:
                chunk = os.read(key.fileobj.fileno(), 65536)
                if not chunk:
                    selector.unregister(key.fileobj)
                    continue
                buffer = buffers[key.fileobj]
                buffer.extend(chunk)
                limit = cap if key.fileobj is process.stdout else 65536
                if len(buffer) > limit:
                    raise SourceEnvironmentError("bounded command output exceeded cap")
        returncode = process.wait(timeout=max(0.0, deadline - time.monotonic()))
        _require(returncode == 0, "bounded command failed")
        return bytes(buffers[process.stdout])
    except BaseException:
        if process.poll() is None:
            process.kill()
        process.wait()
        raise
    finally:
        selector.close()
        process.stdout.close()
        process.stderr.close()


def _git(root: str, arguments: list[str], *, cap: int = METADATA_CAP) -> bytes:
    return _run(["git", *arguments], cwd=root, cap=cap)


def _line(data: bytes, where: str) -> str:
    try:
        value = data.decode("ascii")
    except UnicodeDecodeError as exc:
        raise SourceEnvironmentError(where + " returned non-ASCII output") from exc
    _require(value.endswith("\n") and "\n" not in value[:-1], where + " returned malformed output")
    return value[:-1]


def _read_source(rootfd: int, path: str) -> tuple[bytes, tuple[tuple[tuple[int, int, int], ...],
                                                               tuple[int, ...]]]:
    parts = path.split("/")
    fd = os.dup(rootfd)
    directories = []
    try:
        for part in parts[:-1]:
            next_fd = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = next_fd
            directories.append(_directory_identity(os.fstat(fd)))
        # O_NONBLOCK prevents a substituted FIFO from hanging before fstat can
        # reject its nonregular type. It has no effect on regular-file reads.
        filefd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd)
        try:
            before = os.fstat(filefd)
            _require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1 and
                     before.st_size <= SOURCE_FILE_CAP, "unsafe or oversized source file")
            result = bytearray()
            while len(result) <= before.st_size:
                chunk = os.read(filefd, min(65536, before.st_size + 1 - len(result)))
                if not chunk:
                    break
                result.extend(chunk)
            after = os.fstat(filefd)
            _require(_signature(after) == _signature(before) and len(result) == before.st_size,
                     "source file changed during read")
            return bytes(result), (tuple(directories), _signature(before))
        finally:
            os.close(filefd)
    except OSError as exc:
        raise SourceEnvironmentError("source file open failed") from exc
    finally:
        os.close(fd)


def _tracked_file(root: str, path: str) -> tuple[str, str]:
    output = _git(root, ["ls-files", "--stage", "--", path])
    line = _line(output, "git ls-files")
    try:
        metadata, found_path = line.split("\t", 1)
        mode, oid, stage = metadata.split(" ")
    except ValueError as exc:
        raise SourceEnvironmentError("malformed Git index membership") from exc
    _require(found_path == path and mode in ("100644", "100755") and stage == "0",
             "source is not one exact regular stage-zero Git path")
    return mode, oid


def collect_verified_sources(repo_root, *, profile):
    """Read and bind exact clean worktree bytes to the current Git commit."""
    membership = _membership(profile)
    root, root_snapshot = _canonical_root(repo_root)
    _require(_line(_git(root, ["rev-parse", "--show-toplevel"]), "Git top-level") == root,
             "supplied root is not the Git top-level")
    object_format = _line(_git(root, ["rev-parse", "--show-object-format"]), "Git object format")
    _require(object_format in _OID, "unsupported Git object format")
    revision = _line(_git(root, ["rev-parse", "--verify", "HEAD^{commit}"]), "Git revision")
    _require(_OID[object_format].fullmatch(revision) is not None, "invalid Git revision")
    status_args = ["status", "--porcelain=v1", "-z", "--untracked-files=all", "--ignored=no"]
    _require(_git(root, status_args) == b"", "Git worktree is not clean including untracked files")
    _require(_git(root, ["submodule", "status", "--recursive"]) == b"", "submodules are forbidden")
    rootfd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    signatures = {}
    records = []
    total = 0
    try:
        for role, path in membership:
            mode, oid = _tracked_file(root, path)
            _require(_OID[object_format].fullmatch(oid) is not None, "invalid Git blob object ID")
            size_text = _line(_git(root, ["cat-file", "-s", oid]), "Git blob size")
            _require(size_text.isdecimal(), "invalid Git blob size")
            blob_size = int(size_text)
            _require(blob_size <= SOURCE_FILE_CAP, "Git blob exceeds source cap")
            blob = _git(root, ["cat-file", "blob", oid], cap=SOURCE_FILE_CAP)
            _require(len(blob) == blob_size, "Git blob length mismatch")
            current, signature = _read_source(rootfd, path)
            _require(current == blob, "worktree bytes differ from committed blob")
            signatures[path] = signature
            total += len(current)
            _require(total <= SOURCE_TOTAL_CAP, "source set exceeds total cap")
            records.append({"path": path, "role": role, "git_mode": mode,
                            "size_bytes": len(current),
                            "sha256": hashlib.sha256(current).hexdigest(),
                            "git_blob_oid": oid})
        _require(_line(_git(root, ["rev-parse", "--verify", "HEAD^{commit}"]),
                       "Git revision") == revision, "Git revision changed during collection")
        _require(_git(root, status_args) == b"", "Git worktree changed during collection")
        _require(_git(root, ["submodule", "status", "--recursive"]) == b"",
                 "submodule state changed during collection")
        for _, path in membership:
            _, signature = _read_source(rootfd, path)
            _require(signature == signatures[path], "source file changed after collection")
        _check_root_snapshot(root_snapshot)
        _require(_directory_identity(os.fstat(rootfd)) == root_snapshot[-1][1],
                 "repository root changed")
    finally:
        os.close(rootfd)
    result = {"schema": "i7_source_binding_v1", "profile": profile,
              "repository_revision": revision, "git_object_format": object_format,
              "repository_root_realpath": root,
              "worktree_status": "clean_including_untracked_v1", "files": records}
    validate_sources(result, profile=profile)
    return result


def _same_numpy_rng(left, right) -> bool:
    return (left[0] == right[0] and np.array_equal(left[1], right[1]) and
            left[2:] == right[2:])


def _uuid(properties) -> str:
    try:
        return canonical_cuda_uuid(getattr(properties, "uuid", None))
    except ValueError as exc:
        raise SourceEnvironmentError("CUDA device has no valid stable UUID") from exc


def _driver_version(root: str, uuid: str) -> str:
    _require(is_canonical_cuda_uuid(uuid), "driver lookup requires canonical CUDA UUID")
    output = _run(["nvidia-smi", "--query-gpu=uuid,driver_version",
                   "--format=csv,noheader,nounits"], cwd=root, cap=METADATA_CAP)
    try:
        rows = [tuple(item.strip() for item in line.split(",", 1))
                for line in output.decode("ascii").splitlines() if line.strip()]
    except UnicodeDecodeError as exc:
        raise SourceEnvironmentError("non-ASCII NVIDIA driver output") from exc
    _require(all(len(row) == 2 for row in rows), "malformed NVIDIA driver row")
    try:
        normalized = [(canonical_cuda_uuid(found_uuid), version)
                      for found_uuid, version in rows]
    except ValueError as exc:
        raise SourceEnvironmentError("malformed NVIDIA driver UUID") from exc
    matches = [version for found_uuid, version in normalized if found_uuid == uuid and version]
    _require(len(matches) == 1, "CUDA UUID has no unique driver-version row")
    return matches[0]


def _rng_snapshot(*, native: bool):
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    numpy_copy = (numpy_state[0], numpy_state[1].copy(), *numpy_state[2:])
    cpu = torch.get_rng_state().detach().cpu().clone()
    cuda = [item.detach().cpu().clone() for item in torch.cuda.get_rng_state_all()] if native else []
    return python_state, numpy_copy, cpu, cuda


def _rng_equal(left, right) -> bool:
    return (left[0] == right[0] and _same_numpy_rng(left[1], right[1]) and
            torch.equal(left[2], right[2]) and len(left[3]) == len(right[3]) and
            all(torch.equal(a, b) for a, b in zip(left[3], right[3])))


def collect_runtime_environment(repo_root, *, profile, runtime_role):
    """Observe whitelisted process state without changing settings or RNG."""
    _membership(profile)
    allowed = ("native_source", "cpu_audit", "storage_crosscheck_cpu") if profile == storage.SCIENTIFIC else ("fixture_cpu",)
    _require(type(runtime_role) is str and runtime_role in allowed, "runtime role/profile mismatch")
    root, root_snapshot = _canonical_root(repo_root)
    initialized = torch.cuda.is_initialized()
    native = runtime_role == "native_source"
    _require(initialized == native, "CUDA initialization state differs from runtime role")
    before = _rng_snapshot(native=native)
    try:
        libc_name, libc_version = platform.libc_ver()
        result = {
            "schema": "i7_environment_binding_v1", "profile": profile,
            "runtime_role": runtime_role, "repository_root_realpath": root,
            "python": {
                "implementation": platform.python_implementation(),
                "version": platform.python_version(),
                "version_info": list(sys.version_info[:3]),
                "cache_tag": str(sys.implementation.cache_tag),
                "executable_realpath": os.path.realpath(sys.executable),
            },
            "libraries": {
                "numpy_version": str(np.__version__), "torch_version": str(torch.__version__),
                "torch_git_revision": str(torch.version.git_version),
                "torch_cuda_build": None if torch.version.cuda is None else str(torch.version.cuda),
                # The public helper calls torch.cuda.is_available(); this loaded
                # extension query returns the same integer without probing CUDA.
                "cudnn_version": int(torch._C._cudnn.getVersionInt()),
            },
            "operating_system": {
                "sys_platform": sys.platform, "system": platform.system(),
                "release": platform.release(), "machine": platform.machine(),
                "libc_name": libc_name, "libc_version": libc_version,
            },
            "safe_environment": {name: os.environ.get(name) for name in SAFE_ENV_KEYS},
            "torch_settings": {
                "num_threads": torch.get_num_threads(),
                "num_interop_threads": torch.get_num_interop_threads(),
                "default_dtype": str(torch.get_default_dtype()),
                "default_device": str(torch.get_default_device()),
                "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
                "grad_enabled": torch.is_grad_enabled(),
                "inference_mode_enabled": torch.is_inference_mode_enabled(),
                "autocast_cpu_enabled": torch.is_autocast_enabled("cpu"),
                "autocast_cuda_enabled": torch.is_autocast_enabled("cuda"),
                "autocast_cpu_dtype": str(torch.get_autocast_dtype("cpu")),
                "autocast_cuda_dtype": str(torch.get_autocast_dtype("cuda")),
                "float32_matmul_precision": torch.get_float32_matmul_precision(),
                "cudnn_enabled": torch.backends.cudnn.enabled,
                "cudnn_benchmark": torch.backends.cudnn.benchmark,
                "cudnn_deterministic": torch.backends.cudnn.deterministic,
                "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                "allow_fp16_reduced_precision_reduction":
                    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
                "allow_bf16_reduced_precision_reduction":
                    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
            },
            "cuda": {"initialized": initialized, "visible_device_count": None,
                     "current_device": None, "driver_version": None, "devices": []},
            "rng_layout": {
                "python": {"state_version": before[0][0], "internal_length": len(before[0][1])},
                "numpy": {"algorithm": before[1][0], "keys_dtype": str(before[1][1].dtype),
                          "keys_length": int(before[1][1].size)},
                "torch_cpu": {"dtype": str(before[2].dtype), "state_length": before[2].numel()},
                "torch_cuda": [],
            },
        }
        if native:
            count = torch.cuda.device_count()
            current = torch.cuda.current_device()
            devices, rng_rows = [], []
            _require(len(before[3]) == count, "CUDA RNG/device count mismatch")
            for index, state in enumerate(before[3]):
                props = torch.cuda.get_device_properties(index)
                uuid = _uuid(props)
                devices.append({"index": index, "name": str(props.name), "uuid": uuid,
                                "total_memory_bytes": int(props.total_memory),
                                "capability_major": int(props.major),
                                "capability_minor": int(props.minor)})
                rng_rows.append({"index": index, "name": str(props.name), "uuid": uuid,
                                 "dtype": str(state.dtype), "state_length": state.numel()})
            result["cuda"] = {"initialized": True, "visible_device_count": count,
                              "current_device": current,
                              "driver_version": _driver_version(root, devices[current]["uuid"]),
                              "devices": devices}
            result["rng_layout"]["torch_cuda"] = rng_rows
        validate_environment(result, profile=profile)
        _check_root_snapshot(root_snapshot)
        return result
    finally:
        after = _rng_snapshot(native=native)
        _require(_rng_equal(before, after), "runtime collection changed RNG state")
        _require(torch.cuda.is_initialized() == initialized,
                 "runtime collection changed CUDA initialization state")


def validate_state_environment(core, environment, *, profile):
    """Pure cross-check of a previously validated core and environment snapshot."""
    validate_environment(environment, profile=profile)
    try:
        _require(type(core) is dict and core["profile"] == profile, "core profile mismatch")
        role = environment["runtime_role"]
        if profile == storage.SCIENTIFIC:
            _require(role == "native_source", "scientific core requires native-source environment")
            native_device = "cuda:" + str(environment["cuda"]["current_device"])
        else:
            _require(role == "fixture_cpu", "fixture core requires fixture environment")
            native_device = "cpu"
        parameters = core["model"]["parameters"]
        _require(type(parameters) is list and parameters and
                 all(entry["native_device"] == native_device for entry in parameters),
                 "model device differs from environment")
        for entry in core["optimizer"]["state"]:
            _require(entry["exp_avg"]["native_device"] == native_device and
                     entry["exp_avg_sq"]["native_device"] == native_device,
                     "optimizer device differs from environment")
        observer = core["observer"]["state"]
        for name in ("V", "grad_mean"):
            if observer[name] is not None:
                _require(observer[name]["native_device"] == native_device,
                         "observer device differs from environment")
        rng, layout = core["rng"], environment["rng_layout"]
        _require(rng["python"]["version"] == layout["python"]["state_version"] and
                 len(rng["python"]["internal"]) == layout["python"]["internal_length"],
                 "Python RNG layout differs")
        _require(rng["numpy"]["algorithm"] == layout["numpy"]["algorithm"] and
                 str(rng["numpy"]["keys"].dtype).removeprefix("torch.") ==
                 layout["numpy"]["keys_dtype"] and
                 rng["numpy"]["keys"].numel() == layout["numpy"]["keys_length"],
                 "NumPy RNG layout differs")
        _require(str(rng["torch_cpu"].dtype) == layout["torch_cpu"]["dtype"] and
                 rng["torch_cpu"].numel() == layout["torch_cpu"]["state_length"],
                 "Torch CPU RNG layout differs")
        _require(len(rng["torch_cuda"]) == len(layout["torch_cuda"]) ==
                 len(environment["cuda"]["devices"]), "CUDA RNG/device count differs")
        for saved, expected, device in zip(rng["torch_cuda"], layout["torch_cuda"],
                                           environment["cuda"]["devices"]):
            _require((saved["device_index"], saved["name"], saved["uuid"],
                      str(saved["state"].dtype), saved["state"].numel()) ==
                     (expected["index"], expected["name"], expected["uuid"],
                      expected["dtype"], expected["state_length"]) and
                     (expected["index"], expected["name"], expected["uuid"]) ==
                     (device["index"], device["name"], device["uuid"]),
                     "CUDA RNG identity/layout differs")
    except SourceEnvironmentError:
        raise
    except (KeyError, TypeError, AttributeError, IndexError):
        raise SourceEnvironmentError("malformed core/environment cross-binding") from None
