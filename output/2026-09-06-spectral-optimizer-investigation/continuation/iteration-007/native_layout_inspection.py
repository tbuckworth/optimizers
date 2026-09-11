"""One-shot engineering-only native layout inspector.

Import, help, and the default CLI are inert: Torch and the scientific source
collector are imported only by the controller-bound worker.  This file
does not enable a scientific runner or constitute launch approval.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import resource
import selectors
import signal
import stat
import subprocess
import sys
import time
from typing import Any

import source_environment_schema as environment_schema


SCHEMA = "i7_native_layout_inspection_v2"
REPO_ROOT = "/private-artifacts/repositories/optimizers-launch-investigation"
RELATIVE_PATH = (
    "output/2026-09-06-spectral-optimizer-investigation/continuation/"
    "iteration-007/native_layout_inspection.py"
)
ATTEMPT_PARENT = "/tmp/spectral-experiment-artifacts"
ATTEMPT_ROOT = ATTEMPT_PARENT + "/spectral-i7-native-layout-inspection-001"
EXPECTED_MOUNT = {
    "target": "/private-artifacts/storage",
    "source": "/dev/RECONFIGURE_FOR_LOCAL_STORAGE",
    "uuid": "00000000-0000-4000-8000-000000000000",
}

MARKER_NAME = "inspection-marker.json"
RESULT_NAME = "inspection-result.json"
FAILURE_NAME = "inspection-failure.json"
TOTAL_OUTPUT_CAP = 2 << 20
FAILURE_RESERVE = 256 << 10
NORMAL_OUTPUT_CAP = TOTAL_OUTPUT_CAP - FAILURE_RESERVE
COMBINED_CHILD_OUTPUT_CAP = 1 << 20
FREE_SPACE_MINIMUM = 1 << 30
RSS_CAP = 2 << 30
CUDA_ALLOCATED_CAP = 64 << 20
GPU_FREE_MINIMUM_MIB = 8 << 10
DEADLINE_SECONDS = 45.0
TERMINATION_GRACE_SECONDS = 5.0
KILL_DISAPPEAR_SECONDS = 1.0
COMMAND_OUTPUT_CAP = 256 << 10

EXPECTED_GPU_NAME = "NVIDIA GeForce RTX 3090"
EXPECTED_SAFE_ENVIRONMENT = {
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "CUDA_VISIBLE_DEVICES": None,  # Replaced by the predeclared UUID.
    "MKL_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "PYTHONHASHSEED": "0",
}
EXPECTED_TORCH_SETTINGS = {
    "num_threads": 1,
    "num_interop_threads": 1,
    "default_dtype": "torch.float32",
    "default_device": "cpu",
    "deterministic_algorithms": True,
    "deterministic_warn_only": False,
    "grad_enabled": True,
    "inference_mode_enabled": False,
    "autocast_cpu_enabled": False,
    "autocast_cuda_enabled": False,
    "autocast_cpu_dtype": "torch.bfloat16",
    "autocast_cuda_dtype": "torch.float16",
    "float32_matmul_precision": "highest",
    "cudnn_enabled": True,
    "cudnn_benchmark": False,
    "cudnn_deterministic": True,
    "cuda_matmul_allow_tf32": False,
    "cudnn_allow_tf32": False,
    "allow_fp16_reduced_precision_reduction": True,
    "allow_bf16_reduced_precision_reduction": True,
}

# These are the two explicitly observed, unrelated desktop applications.  The
# kernel's /proc/PID/exe link text is compared directly; in particular, the
# Flatpak namespace path need not exist in the host mount namespace.
ALLOWED_COMPUTE_PROCESSES = {
    "/usr/libexec/gnome-remote-desktop-daemon": {
        "comms": frozenset({"gnome-remote-desktop-daemon", "gnome-remote-de"}),
        "reported_names": frozenset(
            {"/usr/libexec/gnome-remote-desktop-daemon", "gnome-remote-desktop-daemon"}
        ),
    },
    "/app/opt/stremio/stremio": {
        "comms": frozenset({"stremio"}),
        "reported_names": frozenset({"stremio"}),
    },
}

_COMMIT = re.compile(r"[0-9a-f]{40}\Z", re.ASCII)
_SHA256 = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)
_GPU_UUID = re.compile(
    r"GPU-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-"
    r"[0-9a-f]{4}-[0-9a-f]{12}\Z",
    re.ASCII,
)
_COMMANDS_INHERIT_WORKER_GROUP = False
WORKER_STAGES = (
    "worker_invocation_validation", "marker_validation", "worker_environment_validation",
    "helper_before",
    "torch_import", "torch_configuration", "source_before", "cuda_initialization",
    "resource_after_cuda_initialization", "environment_collection",
    "resource_after_environment", "source_after", "resource_after_source",
    "source_comparison", "helper_after", "binding_validation", "resource_final",
    "result_assembly",
)
CUDA_INITIALIZATION_STATES = (
    "not_attempted", "attempted_outcome_unknown", "initialized", "unknown",
)
WORKER_FAILURE_CLASSES = (
    "inspection_error", "source_environment_error", "memory_error",
    "system_error", "runtime_validation_error", "unexpected_exception",
)
WORKER_FAILURE_REASONS = (
    "invalid_cuda_uuid", "driver_uuid_unmatched", "inspection_guard_failure",
    "source_environment_failure", "memory_exhausted", "system_failure",
    "runtime_validation_failure", "unexpected_failure",
)
WORKER_REASONS_BY_CLASS = {
    "inspection_error": frozenset({"inspection_guard_failure"}),
    "source_environment_error": frozenset({
        "invalid_cuda_uuid", "driver_uuid_unmatched", "source_environment_failure"}),
    "memory_error": frozenset({"memory_exhausted"}),
    "system_error": frozenset({"system_failure"}),
    "runtime_validation_error": frozenset({"runtime_validation_failure"}),
    "unexpected_exception": frozenset({"unexpected_failure"}),
}
WORKER_FAILURE_KEYS = (
    "schema", "status", "failure_stage", "failure_class", "failure_reason",
    "inspector_cuda_initialization_state", "report_complete",
)
WORKER_SUCCESS_KEYS = (
    "schema", "status", "expected_commit", "expected_gpu_uuid", "source_binding",
    "native_environment_binding", "helper_binding", "resources", "side_effects",
    "certificates",
)
WORKER_RESOURCE_KEYS = (
    "limits", "wall_seconds", "cpu_seconds", "peak_rss_bytes",
    "torch_peak_allocated_bytes", "torch_peak_reserved_bytes", "own_nvml_gpu_bytes",
    "own_nvml_observable", "rss_scope", "cpu_scope", "torch_allocator_scope",
    "nvml_scope", "memory_cap_scope", "caps_are_cooperative_post_call_observations",
)
WORKER_RESOURCE_LIMITS = {
    "wall_seconds": DEADLINE_SECONDS, "peak_rss_bytes": RSS_CAP,
    "torch_peak_allocated_bytes": CUDA_ALLOCATED_CAP,
}
WORKER_RESOURCE_SCOPES = {
    "rss_scope": "worker RUSAGE_SELF process-lifetime peak on Linux",
    "cpu_scope": "worker process_time only; helper subprocess CPU is unmeasured",
    "torch_allocator_scope": "process allocator peaks; excludes CUDA context/driver",
    "nvml_scope": "own PID current compute-process footprint; null if not observable",
    "memory_cap_scope": "controller and worker checked separately; not aggregate tree RSS",
}
FAILURE_DETAIL_KEYS = (
    "origin", "classification", "reason", "stage",
    "inspector_cuda_initialization_state",
    "worker_report_complete",
)
CONTROLLER_STAGES = (
    "marker_publication", "gpu_preflight", "worker_process", "worker_report_decode",
    "worker_result_validation", "gpu_exit_check", "helper_after",
    "controller_resource_check", "result_publication",
)


class InspectionError(RuntimeError):
    """A terminal, non-retriable inspection failure."""


class WorkerReportedFailure(InspectionError):
    def __init__(self, report: dict[str, Any]):
        super().__init__("worker reported a structured failure")
        self.report = report


class WorkerProtocolFailure(InspectionError):
    def __init__(self, classification: str):
        _require(classification in {
            "malformed_worker_report", "worker_terminated_without_report",
            "worker_exit_status_disagreement",
        }, "invalid worker protocol failure classification")
        super().__init__("worker protocol failure")
        self.classification = classification


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise InspectionError(message)


def _canonical_json(value: Any, *, cap: int) -> bytes:
    try:
        payload = json.dumps(
            # Collector validators treat insertion order as schema.  Preserve
            # it recursively so a persisted source/environment document can be
            # validated directly after json.loads.
            value, sort_keys=False, separators=(",", ":"), ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii") + b"\n"
    except (TypeError, ValueError, UnicodeError) as exc:
        raise InspectionError("record is not bounded canonical JSON") from exc
    _require(len(payload) <= cap, "canonical JSON exceeds its byte cap")
    return payload


def _strict_json_loads(payload: bytes, *, cap: int) -> Any:
    """Decode one canonical JSON value, rejecting duplicates and nonfinite numbers."""
    _require(type(payload) is bytes and 0 < len(payload) <= cap,
             "worker JSON payload has invalid size")
    try:
        text = payload.decode("ascii")
    except UnicodeDecodeError as exc:
        raise InspectionError("worker JSON payload is not ASCII") from exc

    def reject_constant(_token: str):
        raise ValueError("nonfinite JSON constant")

    def finite_float(token: str) -> float:
        value = float(token)
        if not math.isfinite(value):
            raise ValueError("nonfinite JSON number")
        return value

    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    try:
        value = json.loads(text, object_pairs_hook=unique_object,
                           parse_constant=reject_constant, parse_float=finite_float)
    except (json.JSONDecodeError, UnicodeError, ValueError, RecursionError) as exc:
        raise InspectionError("worker JSON payload failed strict decoding") from exc
    _require(_canonical_json(value, cap=cap) == payload,
             "worker JSON payload is not canonical")
    return value


def _worker_failure_class(exc: BaseException) -> str:
    if isinstance(exc, InspectionError):
        return "inspection_error"
    source_module = sys.modules.get("source_environment")
    source_error = (getattr(source_module, "SourceEnvironmentError", None)
                    if source_module is not None else None)
    if isinstance(source_error, type) and isinstance(exc, source_error):
        return "source_environment_error"
    if isinstance(exc, MemoryError):
        return "memory_error"
    if isinstance(exc, (OSError, subprocess.SubprocessError)):
        return "system_error"
    if isinstance(exc, (ValueError, TypeError, RuntimeError)):
        return "runtime_validation_error"
    return "unexpected_exception"


def _worker_failure_report(progress: dict[str, str], exc: BaseException) -> dict[str, Any]:
    stage = progress.get("failure_stage")
    cuda_state = progress.get("inspector_cuda_initialization_state")
    _require(stage in WORKER_STAGES and cuda_state in CUDA_INITIALIZATION_STATES and
             cuda_state != "unknown",
             "invalid worker failure progress")
    failure_class = _worker_failure_class(exc)
    source_module = sys.modules.get("source_environment")
    source_error = (getattr(source_module, "SourceEnvironmentError", None)
                    if source_module is not None else None)
    trusted_source_error = isinstance(source_error, type) and isinstance(exc, source_error)
    if trusted_source_error and str(exc) in {
            "CUDA device has no stable UUID", "CUDA device has no valid stable UUID"}:
        reason = "invalid_cuda_uuid"
    elif trusted_source_error and str(exc) == "CUDA UUID has no unique driver-version row":
        reason = "driver_uuid_unmatched"
    else:
        reason = {
            "inspection_error": "inspection_guard_failure",
            "source_environment_error": "source_environment_failure",
            "memory_error": "memory_exhausted",
            "system_error": "system_failure",
            "runtime_validation_error": "runtime_validation_failure",
            "unexpected_exception": "unexpected_failure",
        }[failure_class]
    return {
        "schema": SCHEMA, "status": "worker_failed", "failure_stage": stage,
        "failure_class": failure_class, "failure_reason": reason,
        "inspector_cuda_initialization_state": cuda_state, "report_complete": True,
    }


def _safe_error(exc: BaseException) -> dict[str, str]:
    """Return only a closed CLI classification, never exception text or type names."""
    if isinstance(exc, InspectionError):
        classification = "inspection_refused"
    elif isinstance(exc, MemoryError):
        classification = "memory_exhausted"
    elif isinstance(exc, (OSError, subprocess.SubprocessError)):
        classification = "system_failure"
    else:
        classification = "unexpected_failure"
    return {"classification": classification}


def _path_snapshot(path: str) -> tuple[tuple[str, tuple[int, int, int]], ...]:
    _require(path.startswith("/") and not path.startswith("//") and
             path != "/" and os.path.normpath(path) == path,
             "path must be canonical, absolute, and non-root")
    rows = []
    current = "/"
    for part in Path(path).parts[1:]:
        current = os.path.join(current, part)
        try:
            info = os.lstat(current)
        except OSError as exc:
            raise InspectionError("path component inspection failed") from exc
        _require(stat.S_ISDIR(info.st_mode) and not stat.S_ISLNK(info.st_mode),
                 "path contains a symlink or non-directory component")
        rows.append((current, (info.st_dev, info.st_ino, info.st_mode)))
    return tuple(rows)


def _check_path_snapshot(rows: tuple[tuple[str, tuple[int, int, int]], ...]) -> None:
    for path, expected in rows:
        try:
            info = os.lstat(path)
        except OSError as exc:
            raise InspectionError("path identity changed") from exc
        actual = (info.st_dev, info.st_ino, info.st_mode)
        _require(stat.S_ISDIR(info.st_mode) and not stat.S_ISLNK(info.st_mode) and
                 actual == expected, "path identity changed")


def _live_group_members(pgid: int) -> tuple[int, ...]:
    """Return live Linux PIDs in a group; dead orphan zombies are not live work."""
    _require(sys.platform.startswith("linux") and type(pgid) is int and pgid > 1,
             "live process-group inspection requires Linux and a valid group")
    members = []
    try:
        entries = os.scandir("/proc")
    except OSError as exc:
        raise InspectionError("cannot inspect owned process group") from exc
    with entries:
        for entry in entries:
            if not entry.name.isdecimal():
                continue
            try:
                with open(entry.path + "/stat", "r", encoding="ascii") as handle:
                    row = handle.read(4096)
            except (FileNotFoundError, ProcessLookupError):
                continue
            except (OSError, UnicodeError) as exc:
                raise InspectionError("cannot inspect owned process group member") from exc
            end = row.rfind(")")
            fields = row[end + 2:].split() if end >= 0 else []
            _require(len(fields) >= 4 and fields[2].lstrip("-").isdecimal(),
                     "malformed process-group metadata")
            state, found_group = fields[0], int(fields[2])
            if found_group == pgid and state not in {"Z", "X"}:
                members.append(int(entry.name))
    return tuple(sorted(members))


def _terminate_own_group(process: subprocess.Popen, *, grace: float) -> None:
    """TERM/KILL and reap only a child known to lead its own new session."""
    pgid = process.pid  # Guaranteed by the Popen(start_new_session=True) call.
    if not _live_group_members(pgid):
        process.wait()
        return
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        process.wait()
        return
    grace_deadline = time.monotonic() + grace
    while time.monotonic() < grace_deadline and _live_group_members(pgid):
        process.poll()
        time.sleep(min(0.01, max(0.0, grace_deadline - time.monotonic())))
    if _live_group_members(pgid):
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    process.wait()
    kill_deadline = time.monotonic() + KILL_DISAPPEAR_SECONDS
    while time.monotonic() < kill_deadline and _live_group_members(pgid):
        time.sleep(min(0.01, max(0.0, kill_deadline - time.monotonic())))
    _require(not _live_group_members(pgid),
             "live owned process-group member remained after SIGKILL")


def _terminate_direct_child(process: subprocess.Popen, *, grace: float) -> None:
    """Stop only a helper inheriting the worker's controller-owned group."""
    if process.poll() is not None:
        process.wait()
        return
    process.terminate()
    try:
        process.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def _capture_process(process: subprocess.Popen, *, deadline: float, cap: int,
                     grace: float = TERMINATION_GRACE_SECONDS,
                     owns_group: bool = True) -> tuple[int, bytes]:
    """Read one combined stream without allowing it to exceed ``cap``."""
    _require(process.stdout is not None, "bounded child has no output pipe")
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)
    output = bytearray()
    try:
        while selector.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise InspectionError("child deadline exceeded")
            events = selector.select(remaining)
            if not events and process.poll() is None:
                continue
            for key, _ in events:
                chunk = os.read(key.fileobj.fileno(), min(65536, cap + 1 - len(output)))
                if not chunk:
                    selector.unregister(key.fileobj)
                    continue
                output.extend(chunk)
                if len(output) > cap:
                    raise InspectionError("combined child output exceeded cap")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise InspectionError("child deadline exceeded")
        code = process.wait(timeout=remaining)
        if owns_group and _live_group_members(process.pid):
            _terminate_own_group(process, grace=grace)
            raise InspectionError("child exited while owned descendants remained")
        return code, bytes(output)
    except BaseException:
        if owns_group:
            _terminate_own_group(process, grace=grace)
        else:
            _terminate_direct_child(process, grace=grace)
        raise
    finally:
        selector.close()
        process.stdout.close()


def _run_command(command: list[str], *, cwd: str, deadline: float,
                 cap: int = COMMAND_OUTPUT_CAP) -> bytes:
    """Run a read-only helper command with bounded combined output."""
    env = dict(os.environ)
    env.update(LC_ALL="C", GIT_OPTIONAL_LOCKS="0", PYTHONDONTWRITEBYTECODE="1")
    owns_group = not _COMMANDS_INHERIT_WORKER_GROUP
    process = subprocess.Popen(
        command, cwd=cwd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, env=env, start_new_session=owns_group,
    )
    code, output = _capture_process(process, deadline=deadline, cap=cap,
                                    owns_group=owns_group)
    _require(code == 0, "bounded helper command failed")
    return output


def _one_ascii_line(data: bytes, where: str) -> str:
    try:
        text = data.decode("ascii")
    except UnicodeDecodeError as exc:
        raise InspectionError(where + " returned non-ASCII output") from exc
    _require(text.endswith("\n") and "\n" not in text[:-1],
             where + " returned malformed output")
    return text[:-1]


def _git(root: str, args: list[str], *, deadline: float,
         cap: int = COMMAND_OUTPUT_CAP) -> bytes:
    return _run_command(["git", *args], cwd=root, deadline=deadline, cap=cap)


def _bind_helper(root: str, expected_commit: str, *, deadline: float) -> dict[str, Any]:
    """Bind this helper's committed bytes separately from SCIENTIFIC_ROLES."""
    _require(_COMMIT.fullmatch(expected_commit) is not None, "invalid expected commit")
    root = os.path.abspath(root)
    _require(root == REPO_ROOT and os.path.realpath(root) == root,
             "repository root is not the fixed physical root")
    snapshot = _path_snapshot(root)
    top = _one_ascii_line(_git(root, ["rev-parse", "--show-toplevel"], deadline=deadline),
                          "Git top-level")
    head = _one_ascii_line(_git(root, ["rev-parse", "--verify", "HEAD^{commit}"],
                                deadline=deadline), "Git revision")
    _require(top == root and head == expected_commit, "expected clean commit mismatch")
    status = _git(root, ["status", "--porcelain=v1", "-z", "--untracked-files=all",
                         "--ignored=no"], deadline=deadline)
    _require(status == b"", "Git worktree is not clean including untracked files")
    _require(_git(root, ["submodule", "status", "--recursive"], deadline=deadline) == b"",
             "submodules are forbidden")
    line = _one_ascii_line(
        _git(root, ["ls-files", "--stage", "--", RELATIVE_PATH], deadline=deadline),
        "Git helper membership",
    )
    try:
        metadata, found_path = line.split("\t", 1)
        mode, oid, stage = metadata.split(" ")
    except ValueError as exc:
        raise InspectionError("malformed helper Git membership") from exc
    _require(found_path == RELATIVE_PATH and mode in ("100644", "100755") and stage == "0" and
             _COMMIT.fullmatch(oid) is not None,
             "helper is not one committed regular stage-zero file")
    blob = _git(root, ["cat-file", "blob", oid], deadline=deadline,
                cap=COMBINED_CHILD_OUTPUT_CAP)
    helper_path = os.path.join(root, RELATIVE_PATH)
    parentfd = os.open(os.path.dirname(helper_path), os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        filefd = os.open(os.path.basename(helper_path), os.O_RDONLY | os.O_NOFOLLOW, dir_fd=parentfd)
        try:
            before = os.fstat(filefd)
            _require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1 and
                     before.st_size <= COMBINED_CHILD_OUTPUT_CAP,
                     "helper file is unsafe or oversized")
            chunks = bytearray()
            while len(chunks) <= before.st_size:
                chunk = os.read(filefd, min(65536, before.st_size + 1 - len(chunks)))
                if not chunk:
                    break
                chunks.extend(chunk)
            after = os.fstat(filefd)
            signature = lambda row: (row.st_dev, row.st_ino, row.st_mode, row.st_nlink,
                                     row.st_size, row.st_mtime_ns, row.st_ctime_ns)
            _require(signature(before) == signature(after) and len(chunks) == before.st_size and
                     bytes(chunks) == blob, "helper bytes differ from committed blob")
        finally:
            os.close(filefd)
    finally:
        os.close(parentfd)
    _require(_one_ascii_line(_git(root, ["rev-parse", "--verify", "HEAD^{commit}"],
                                  deadline=deadline), "Git revision") == expected_commit and
             _git(root, ["status", "--porcelain=v1", "-z", "--untracked-files=all",
                         "--ignored=no"], deadline=deadline) == b"",
             "source tree changed during helper binding")
    _check_path_snapshot(snapshot)
    return {
        "path": RELATIVE_PATH, "role": "engineering_native_layout_inspector",
        "repository_revision": expected_commit, "git_mode": mode,
        "size_bytes": len(blob), "sha256": hashlib.sha256(blob).hexdigest(),
        "git_blob_oid": oid,
    }


def _verify_attempt_parent(parent: str, *, deadline: float) -> tuple[dict[str, Any], tuple]:
    """Verify fixed mount identity, physical path, and free space without writes."""
    _require(parent == ATTEMPT_PARENT, "attempt parent is not the fixed path")
    snapshot = _path_snapshot(parent)
    output = _run_command(
        ["findmnt", "--noheadings", "--target", parent,
         "--output", "TARGET,SOURCE,UUID,MAJ:MIN"],
        cwd=REPO_ROOT, deadline=deadline,
    )
    fields = _one_ascii_line(output, "findmnt").split()
    _require(len(fields) == 4, "findmnt did not return one exact four-field row")
    target, source, uuid, device = fields
    _require((target, source, uuid) == (EXPECTED_MOUNT["target"], EXPECTED_MOUNT["source"],
                                        EXPECTED_MOUNT["uuid"]),
             "wrong mount target/source/UUID")
    try:
        major_text, minor_text = device.split(":", 1)
        _require(all(part.isascii() and part.isdecimal() and str(int(part)) == part
                     for part in (major_text, minor_text)), "invalid mount device number")
        mount_device = (int(major_text), int(minor_text))
    except (ValueError, InspectionError) as exc:
        raise InspectionError("invalid mount device number") from exc
    info = os.stat(parent)
    _require((os.major(info.st_dev), os.minor(info.st_dev)) == mount_device,
             "stat device disagrees with findmnt")
    space = os.statvfs(parent)
    free = int(space.f_bavail) * int(space.f_frsize)
    _require(free >= FREE_SPACE_MINIMUM, "less than 1 GiB free on inspection volume")
    _check_path_snapshot(snapshot)
    return ({"target": target, "source": source, "uuid": uuid,
             "major": mount_device[0], "minor": mount_device[1],
             "free_bytes_before_creation": free}, snapshot)


def _root_identity(rootfd: int, root: str) -> tuple[int, int, int]:
    by_fd = os.fstat(rootfd)
    by_path = os.lstat(root)
    expected = (by_fd.st_dev, by_fd.st_ino, by_fd.st_mode)
    actual = (by_path.st_dev, by_path.st_ino, by_path.st_mode)
    _require(stat.S_ISDIR(by_fd.st_mode) and not stat.S_ISLNK(by_path.st_mode) and
             expected == actual, "inspection directory identity changed")
    return expected


def _create_attempt_root(root: str | None = None) -> tuple[int, tuple[int, int, int]]:
    if root is None:
        root = ATTEMPT_ROOT
    _require(root == ATTEMPT_ROOT, "attempt root is not the fixed singleton")
    try:
        os.mkdir(root, 0o700)
    except FileExistsError as exc:
        raise InspectionError("fixed inspection directory already exists; retry refused") from exc
    rootfd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        return rootfd, _root_identity(rootfd, root)
    except BaseException:
        os.close(rootfd)
        raise


def _owned_file_total(rootfd: int) -> int:
    total = 0
    for name in os.listdir(rootfd):
        _require(name in {MARKER_NAME, RESULT_NAME, FAILURE_NAME},
                 "unexpected inspection-directory entry")
        info = os.stat(name, dir_fd=rootfd, follow_symlinks=False)
        _require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1,
                 "unsafe inspection output entry")
        total += info.st_size
    return total


def _write_new_json(rootfd: int, root: str, identity: tuple[int, int, int], name: str,
                    value: Any, *, cap: int, normal: bool) -> int:
    _require(name in {MARKER_NAME, RESULT_NAME, FAILURE_NAME}, "invalid output filename")
    _require(_root_identity(rootfd, root) == identity, "inspection root changed before write")
    payload = _canonical_json(value, cap=cap)
    current = _owned_file_total(rootfd)
    limit = NORMAL_OUTPUT_CAP if normal else TOTAL_OUTPUT_CAP
    _require(current + len(payload) <= limit, "inspection output budget exhausted")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
    fd = os.open(name, flags, 0o600, dir_fd=rootfd)
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(fd, payload[offset:])
            _require(written > 0, "output write made no progress")
            offset += written
        os.fsync(fd)
        info = os.fstat(fd)
        _require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1 and
                 info.st_size == len(payload), "published output file is unsafe")
    finally:
        os.close(fd)
    os.fsync(rootfd)
    _require(_root_identity(rootfd, root) == identity and
             _owned_file_total(rootfd) <= limit, "inspection output identity/budget changed")
    return len(payload)


def _read_marker(rootfd: int) -> dict[str, Any]:
    fd = os.open(MARKER_NAME, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=rootfd)
    try:
        info = os.fstat(fd)
        _require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1 and
                 0 < info.st_size <= FAILURE_RESERVE, "unsafe inspection marker")
        data = bytearray()
        while len(data) <= info.st_size:
            chunk = os.read(fd, min(65536, info.st_size + 1 - len(data)))
            if not chunk:
                break
            data.extend(chunk)
        _require(len(data) == info.st_size, "inspection marker changed during read")
    finally:
        os.close(fd)
    try:
        value = json.loads(bytes(data))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise InspectionError("inspection marker is not valid JSON") from exc
    _require(_canonical_json(value, cap=FAILURE_RESERVE) == bytes(data),
             "inspection marker is not canonical JSON")
    return value


def _proc_identity(pid: int) -> tuple[str, str, int] | None:
    """Observe executable link text, comm, and real UID; None means it exited."""
    base = f"/proc/{pid}"
    try:
        executable = os.readlink(base + "/exe")
        with open(base + "/comm", "r", encoding="ascii") as handle:
            comm = handle.read(80)
        with open(base + "/status", "r", encoding="ascii") as handle:
            status_text = handle.read(64 << 10)
    except FileNotFoundError:
        return None
    except (OSError, UnicodeError) as exc:
        raise InspectionError("could not verify observed GPU process identity") from exc
    _require(comm.endswith("\n") and "\n" not in comm[:-1],
             "malformed observed GPU process comm")
    uid_rows = [line for line in status_text.splitlines() if line.startswith("Uid:\t")]
    _require(len(uid_rows) == 1, "malformed observed GPU process UID")
    fields = uid_rows[0].split()[1:]
    _require(len(fields) == 4 and all(field.isdecimal() for field in fields),
             "malformed observed GPU process UID")
    uids = [int(field) for field in fields]
    _require(len(set(uids)) == 1, "observed GPU process changes UID domains")
    return executable, comm[:-1], uids[0]


def _parse_compute_rows(output: bytes) -> list[tuple[str, int, str, int | None]]:
    try:
        text = output.decode("ascii")
    except UnicodeDecodeError as exc:
        raise InspectionError("NVIDIA process query returned non-ASCII output") from exc
    rows = []
    for line in text.splitlines():
        if not line.strip() or line.strip().startswith("No running processes found"):
            continue
        fields = [field.strip() for field in line.split(",")]
        unavailable = {"N/A", "[N/A]", "Not Supported"}
        _require(len(fields) == 4 and fields[1].isdecimal() and
                 (fields[3].isdecimal() or fields[3] in unavailable),
                 "malformed NVIDIA process row")
        used = int(fields[3]) if fields[3].isdecimal() else None
        rows.append((fields[0], int(fields[1]), fields[2], used))
    return rows


def _gpu_preflight(expected_uuid: str, *, deadline: float) -> dict[str, Any]:
    _require(_GPU_UUID.fullmatch(expected_uuid) is not None, "invalid expected GPU UUID")
    gpu_output = _run_command(
        ["nvidia-smi", "--query-gpu=index,uuid,name,memory.free,driver_version",
         "--format=csv,noheader,nounits"], cwd=REPO_ROOT, deadline=deadline,
    )
    try:
        lines = [line for line in gpu_output.decode("ascii").splitlines() if line.strip()]
    except UnicodeDecodeError as exc:
        raise InspectionError("NVIDIA GPU query returned non-ASCII output") from exc
    _require(len(lines) == 1, "expected exactly one physical NVIDIA GPU")
    fields = [field.strip() for field in lines[0].split(",")]
    _require(len(fields) == 5 and fields[0] == "0" and fields[3].isdecimal(),
             "malformed NVIDIA GPU row")
    index, uuid, name, free_text, driver = fields
    free_mib = int(free_text)
    _require(uuid == expected_uuid and name == EXPECTED_GPU_NAME,
             "predeclared GPU UUID/name mismatch")
    _require(free_mib >= GPU_FREE_MINIMUM_MIB, "less than 8 GiB free GPU memory")
    _require(bool(driver) and driver.isascii(), "missing NVIDIA driver observation")

    process_output = _run_command(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
         "--format=csv,noheader,nounits"], cwd=REPO_ROOT, deadline=deadline,
    )
    tolerated = []
    for found_uuid, pid, reported_name, used_mib in _parse_compute_rows(process_output):
        _require(found_uuid == expected_uuid, "compute process observed on unexpected GPU")
        observed = _proc_identity(pid)
        if observed is None:
            continue
        executable, comm, uid = observed
        allowed = ALLOWED_COMPUTE_PROCESSES.get(executable)
        _require(allowed is not None and comm in allowed["comms"] and
                 reported_name in allowed["reported_names"] and uid == os.getuid(),
                 "research or unknown compute process occupies the GPU")
        tolerated.append({"kind": "known_unrelated_desktop_application",
                          "pid": pid, "used_gpu_memory_mib": used_mib})
    return {"index": int(index), "uuid": uuid, "name": name,
            "free_memory_mib": free_mib, "driver_version_observed": driver,
            "driver_version_pinned": False,
            "occupancy_snapshot_scope": "compute-apps query; not a continuous monitor",
            "occupancy_exclusive": False, "tolerated_processes": tolerated}


def _worker_environment(expected_uuid: str) -> dict[str, str]:
    env = dict(os.environ)
    expected = dict(EXPECTED_SAFE_ENVIRONMENT)
    expected["CUDA_VISIBLE_DEVICES"] = expected_uuid
    env.update(expected)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def _torch_settings(torch) -> dict[str, Any]:
    return {
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
    }


def _configure_and_assert_torch(torch) -> dict[str, Any]:
    """Set every recorded controllable value to an independently fixed literal."""
    _require(not torch.cuda.is_initialized(), "CUDA initialized before numerical configuration")
    _require(not torch.is_inference_mode_enabled(), "worker entered ambient inference mode")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_default_dtype(torch.float32)
    torch.set_default_device("cpu")
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.set_grad_enabled(True)
    torch.set_autocast_enabled("cpu", False)
    torch.set_autocast_enabled("cuda", False)
    torch.set_autocast_dtype("cpu", torch.bfloat16)
    torch.set_autocast_dtype("cuda", torch.float16)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    observed = _torch_settings(torch)
    _require(observed == EXPECTED_TORCH_SETTINGS, "explicit Torch settings mismatch")
    _require(not torch.cuda.is_initialized(), "numerical configuration initialized CUDA")
    return observed


def _peak_rss_bytes() -> int:
    _require(sys.platform.startswith("linux"), "RSS observation requires Linux KiB semantics")
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
    _require(value >= 0, "invalid process RSS observation")
    return value


def _own_nvml_footprint(expected_uuid: str, *, deadline: float) -> int | None:
    output = _run_command(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
         "--format=csv,noheader,nounits"], cwd=REPO_ROOT, deadline=deadline,
    )
    matches = [used for uuid, pid, _name, used in _parse_compute_rows(output)
               if uuid == expected_uuid and pid == os.getpid()]
    _require(len(matches) <= 1, "own NVIDIA process footprint is ambiguous")
    return None if not matches or matches[0] is None else matches[0] * (1 << 20)


def _check_worker_resources(torch, *, expected_uuid: str, deadline: float,
                            observe_nvml: bool) -> dict[str, Any]:
    rss = _peak_rss_bytes()
    allocated = int(torch.cuda.max_memory_allocated(0))
    reserved = int(torch.cuda.max_memory_reserved(0))
    _require(0 <= allocated <= reserved, "invalid Torch CUDA allocator observations")
    own_nvml = (_own_nvml_footprint(expected_uuid, deadline=deadline)
                if observe_nvml else None)
    _require(rss <= RSS_CAP, "worker RSS cap exceeded")
    _require(allocated <= CUDA_ALLOCATED_CAP, "worker Torch allocated-memory cap exceeded")
    return {"peak_rss_bytes": rss, "torch_peak_allocated_bytes": allocated,
            "torch_peak_reserved_bytes": reserved, "own_nvml_gpu_bytes": own_nvml,
            "own_nvml_observable": observe_nvml and own_nvml is not None}


def _worker(marker_digest: str, cli_expected_commit: str, cli_expected_uuid: str,
            cli_parent_pid: int, progress: dict[str, str]) -> dict[str, Any]:
    """Private worker; caller has already parsed the fixed marker and parent PID."""
    global _COMMANDS_INHERIT_WORKER_GROUP
    _COMMANDS_INHERIT_WORKER_GROUP = True
    progress["failure_stage"] = "marker_validation"
    rootfd = os.open(ATTEMPT_ROOT, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        identity = _root_identity(rootfd, ATTEMPT_ROOT)
        marker = _read_marker(rootfd)
        _require(hashlib.sha256(_canonical_json(marker, cap=FAILURE_RESERVE)).hexdigest() ==
                 marker_digest, "worker marker digest mismatch")
        required = {"schema", "status", "attempt_root", "controller_pid", "expected_commit",
                    "expected_gpu_uuid", "helper_binding_digest", "deadline_monotonic", "limits"}
        _require(type(marker) is dict and set(marker) == required and
                 marker["schema"] == SCHEMA and marker["status"] == "started" and
                 marker["attempt_root"] == ATTEMPT_ROOT and
                 type(marker["controller_pid"]) is int and marker["controller_pid"] > 1 and
                 os.getppid() == marker["controller_pid"] == cli_parent_pid and
                 marker["expected_commit"] == cli_expected_commit and
                 marker["expected_gpu_uuid"] == cli_expected_uuid,
                 "worker parent/marker binding failed")
        expected_limits = {"total_output_bytes": TOTAL_OUTPUT_CAP,
                           "failure_reserve_bytes": FAILURE_RESERVE,
                           "combined_child_output_bytes": COMBINED_CHILD_OUTPUT_CAP,
                           "wall_seconds": DEADLINE_SECONDS,
                           "termination_grace_seconds": TERMINATION_GRACE_SECONDS,
                           "kill_disappear_seconds": KILL_DISAPPEAR_SECONDS,
                           "peak_rss_bytes": RSS_CAP,
                           "torch_peak_allocated_bytes": CUDA_ALLOCATED_CAP}
        _require(marker["limits"] == expected_limits and
                 type(marker["helper_binding_digest"]) is str and
                 _SHA256.fullmatch(marker["helper_binding_digest"]) is not None,
                 "worker marker limits/helper digest mismatch")
        expected_commit = marker["expected_commit"]
        expected_uuid = marker["expected_gpu_uuid"]
        deadline = marker["deadline_monotonic"]
        _require(type(deadline) is float and math.isfinite(deadline) and
                 time.monotonic() < deadline <= time.monotonic() + DEADLINE_SECONDS,
                 "worker marker deadline is invalid")
        progress["failure_stage"] = "worker_environment_validation"
        expected_env = dict(EXPECTED_SAFE_ENVIRONMENT)
        expected_env["CUDA_VISIBLE_DEVICES"] = expected_uuid
        _require({key: os.environ.get(key) for key in expected_env} == expected_env and
                 os.environ.get("PYTHONDONTWRITEBYTECODE") == "1",
                 "worker numerical environment mismatch")
        progress["failure_stage"] = "helper_before"
        helper_before = _bind_helper(REPO_ROOT, expected_commit, deadline=deadline)
        helper_digest = hashlib.sha256(
            _canonical_json(helper_before, cap=FAILURE_RESERVE)).hexdigest()
        _require(helper_digest == marker["helper_binding_digest"],
                 "worker helper binding mismatch")
        _require(_root_identity(rootfd, ATTEMPT_ROOT) == identity,
                 "inspection root changed before worker import")
    finally:
        os.close(rootfd)

    wall_start, cpu_start = time.monotonic(), time.process_time()
    progress["failure_stage"] = "torch_import"
    import torch
    import artifact_store as storage
    import source_environment

    progress["failure_stage"] = "torch_configuration"
    _configure_and_assert_torch(torch)
    _require(_peak_rss_bytes() <= RSS_CAP, "worker RSS cap exceeded after Torch import")
    progress["failure_stage"] = "source_before"
    source_before = source_environment.collect_verified_sources(
        REPO_ROOT, profile=storage.SCIENTIFIC)
    _require(not torch.cuda.is_initialized(),
             "pre-initialization scientific source collection initialized CUDA")
    _require(_peak_rss_bytes() <= RSS_CAP,
             "worker RSS cap exceeded after pre-initialization source collection")
    progress["failure_stage"] = "cuda_initialization"
    progress["inspector_cuda_initialization_state"] = "attempted_outcome_unknown"
    torch.cuda.init()
    _require(torch.cuda.is_initialized(), "explicit CUDA initialization failed")
    progress["inspector_cuda_initialization_state"] = "initialized"
    progress["failure_stage"] = "resource_after_cuda_initialization"
    resource_rows = [_check_worker_resources(
        torch, expected_uuid=expected_uuid, deadline=deadline, observe_nvml=False)]
    progress["failure_stage"] = "environment_collection"
    environment = source_environment.collect_runtime_environment(
        REPO_ROOT, profile=storage.SCIENTIFIC, runtime_role="native_source")
    progress["failure_stage"] = "resource_after_environment"
    resource_rows.append(_check_worker_resources(
        torch, expected_uuid=expected_uuid, deadline=deadline, observe_nvml=False))
    progress["failure_stage"] = "source_after"
    source_after = source_environment.collect_verified_sources(
        REPO_ROOT, profile=storage.SCIENTIFIC)
    progress["failure_stage"] = "resource_after_source"
    resource_rows.append(_check_worker_resources(
        torch, expected_uuid=expected_uuid, deadline=deadline, observe_nvml=False))
    progress["failure_stage"] = "source_comparison"
    _require(source_before == source_after, "scientific source binding changed during inspection")
    progress["failure_stage"] = "helper_after"
    helper_after = _bind_helper(REPO_ROOT, expected_commit, deadline=deadline)
    _require(helper_after == helper_before, "helper binding changed during inspection")
    progress["failure_stage"] = "binding_validation"
    _require(source_before["repository_revision"] == expected_commit,
             "scientific source revision differs from expected commit")
    device = environment["cuda"]["devices"][0]
    _require(device["uuid"] == expected_uuid and device["name"] == EXPECTED_GPU_NAME,
             "native collector differs from predeclared GPU identity")
    expected_env = dict(EXPECTED_SAFE_ENVIRONMENT)
    expected_env["CUDA_VISIBLE_DEVICES"] = expected_uuid
    _require(environment["safe_environment"] == expected_env,
             "native collector differs from frozen numerical environment")
    _require(environment["torch_settings"] == EXPECTED_TORCH_SETTINGS,
             "native collector differs from explicit Torch settings")
    progress["failure_stage"] = "resource_final"
    final_resources = _check_worker_resources(
        torch, expected_uuid=expected_uuid, deadline=deadline, observe_nvml=True)
    resource_rows.append(final_resources)
    progress["failure_stage"] = "result_assembly"
    resources = {
        "limits": dict(WORKER_RESOURCE_LIMITS),
        "wall_seconds": time.monotonic() - wall_start,
        "cpu_seconds": time.process_time() - cpu_start,
        "peak_rss_bytes": max(row["peak_rss_bytes"] for row in resource_rows),
        "torch_peak_allocated_bytes": max(
            row["torch_peak_allocated_bytes"] for row in resource_rows),
        "torch_peak_reserved_bytes": max(
            row["torch_peak_reserved_bytes"] for row in resource_rows),
        "own_nvml_gpu_bytes": final_resources["own_nvml_gpu_bytes"],
        "own_nvml_observable": final_resources["own_nvml_observable"],
        **WORKER_RESOURCE_SCOPES,
        "caps_are_cooperative_post_call_observations": True,
    }
    _require(resources["wall_seconds"] <= DEADLINE_SECONDS,
             "worker cooperative wall deadline exceeded")
    return {
        "schema": SCHEMA, "status": "worker_pass",
        "expected_commit": expected_commit, "expected_gpu_uuid": expected_uuid,
        "source_binding": source_before, "native_environment_binding": environment,
        "helper_binding": helper_before, "resources": resources,
        "side_effects": {"cuda_context_initialized": True,
                         "cuda_initialization_is_not_read_only": True,
                         "random_draws_performed": False,
                         "rng_state_restoration_performed": False},
        "certificates": {"scientific_execution": False, "pilot": False,
                         "full_fit": False, "capture_neutrality": False,
                         "continuation_restore": False},
    }


def _exact_bool_mapping(value: Any, expected: dict[str, bool]) -> bool:
    return (type(value) is dict and tuple(value) == tuple(expected) and
            all(type(value[name]) is bool for name in expected) and value == expected)


def _decode_worker_output(output: bytes) -> dict[str, Any]:
    """Validate the wrapper envelope and delegate complete collector-doc schemas."""
    value = _strict_json_loads(output, cap=COMBINED_CHILD_OUTPUT_CAP)
    _require(type(value) is dict and value.get("schema") == SCHEMA and
             type(value.get("status")) is str,
             "worker returned wrong schema/status")
    if value["status"] == "worker_failed":
        stage = value.get("failure_stage")
        cuda_state = value.get("inspector_cuda_initialization_state")
        pre_cuda = WORKER_STAGES[:WORKER_STAGES.index("cuda_initialization")]
        post_cuda = WORKER_STAGES[WORKER_STAGES.index(
            "resource_after_cuda_initialization"):]
        _require(tuple(value) == WORKER_FAILURE_KEYS and
                 stage in WORKER_STAGES and
                 value["failure_class"] in WORKER_FAILURE_CLASSES and
                 value["failure_reason"] in WORKER_FAILURE_REASONS and
                 value["failure_reason"] in
                    WORKER_REASONS_BY_CLASS[value["failure_class"]] and
                 ((cuda_state == "not_attempted" and stage in pre_cuda) or
                  (cuda_state == "attempted_outcome_unknown" and
                   stage == "cuda_initialization") or
                  (cuda_state == "initialized" and stage in post_cuda)) and
                 value["report_complete"] is True,
                 "worker failure report has wrong exact shape")
        return value
    _require(value["status"] == "worker_pass" and tuple(value) == WORKER_SUCCESS_KEYS,
             "worker success report has wrong exact shape")
    _require(type(value["expected_commit"]) is str and
             _COMMIT.fullmatch(value["expected_commit"]) is not None and
             type(value["expected_gpu_uuid"]) is str and
             _GPU_UUID.fullmatch(value["expected_gpu_uuid"]) is not None,
             "worker success identity fields are invalid")
    source = value["source_binding"]
    environment = value["native_environment_binding"]
    helper = value["helper_binding"]
    try:
        environment_schema.validate_sources(
            source, profile=environment_schema.SCIENTIFIC)
        environment_schema.validate_environment(
            environment, profile=environment_schema.SCIENTIFIC)
    except environment_schema.SourceEnvironmentError as exc:
        raise InspectionError("worker collector document failed exact schema validation") from exc
    _require(source["repository_revision"] == value["expected_commit"] and
             source["repository_root_realpath"] == REPO_ROOT and
             environment["repository_root_realpath"] == REPO_ROOT and
             environment["cuda"]["devices"][0].get("uuid") ==
                value["expected_gpu_uuid"] and
             environment["safe_environment"] == {
                 **EXPECTED_SAFE_ENVIRONMENT,
                 "CUDA_VISIBLE_DEVICES": value["expected_gpu_uuid"]} and
             environment["torch_settings"] == EXPECTED_TORCH_SETTINGS,
             "worker collector document differs from wrapper bindings")
    _require(type(helper) is dict and tuple(helper) == (
                 "path", "role", "repository_revision", "git_mode", "size_bytes", "sha256",
                 "git_blob_oid"),
             "worker helper binding has wrong exact shape")
    _require(helper["path"] == RELATIVE_PATH and
             helper["role"] == "engineering_native_layout_inspector" and
             helper["repository_revision"] == value["expected_commit"] and
             helper["git_mode"] in ("100644", "100755") and
             type(helper["size_bytes"]) is int and helper["size_bytes"] >= 0 and
             type(helper["sha256"]) is str and
             _SHA256.fullmatch(helper["sha256"]) is not None and
             type(helper["git_blob_oid"]) is str and
             _COMMIT.fullmatch(helper["git_blob_oid"]) is not None,
             "worker helper binding has invalid fixed fields")
    _require(type(value["resources"]) is dict and
             tuple(value["resources"]) == WORKER_RESOURCE_KEYS,
             "worker resource report has wrong exact shape")
    resources = value["resources"]
    byte_fields = ("peak_rss_bytes", "torch_peak_allocated_bytes",
                   "torch_peak_reserved_bytes")
    limits = resources["limits"]
    _require(type(limits) is dict and tuple(limits) == tuple(WORKER_RESOURCE_LIMITS) and
             type(limits["wall_seconds"]) is float and
             all(type(limits[name]) is int for name in (
                 "peak_rss_bytes", "torch_peak_allocated_bytes")) and
             limits == WORKER_RESOURCE_LIMITS and
             all(type(resources[name]) is float and resources[name] >= 0.0
                 for name in ("wall_seconds", "cpu_seconds")) and
             all(type(resources[name]) is int and resources[name] >= 0
                 for name in byte_fields) and
             resources["wall_seconds"] <= DEADLINE_SECONDS and
             resources["peak_rss_bytes"] <= RSS_CAP and
             resources["torch_peak_allocated_bytes"] <= CUDA_ALLOCATED_CAP and
             resources["torch_peak_allocated_bytes"] <=
                resources["torch_peak_reserved_bytes"] and
             (resources["own_nvml_gpu_bytes"] is None or
              (type(resources["own_nvml_gpu_bytes"]) is int and
               resources["own_nvml_gpu_bytes"] >= 0)) and
             type(resources["own_nvml_observable"]) is bool and
             resources["own_nvml_observable"] is
                (resources["own_nvml_gpu_bytes"] is not None) and
             all(resources[name] == expected
                 for name, expected in WORKER_RESOURCE_SCOPES.items()) and
             resources["caps_are_cooperative_post_call_observations"] is True,
             "worker resource report has invalid fields")
    expected_side_effects = {
                 "cuda_context_initialized": True,
                 "cuda_initialization_is_not_read_only": True,
                 "random_draws_performed": False,
                 "rng_state_restoration_performed": False}
    expected_certificates = {
                 "scientific_execution": False, "pilot": False, "full_fit": False,
                 "capture_neutrality": False, "continuation_restore": False}
    _require(_exact_bool_mapping(value["side_effects"], expected_side_effects) and
             _exact_bool_mapping(value["certificates"], expected_certificates),
             "worker success flags are invalid")
    return value


def _interpret_worker_exit(returncode: int, output: bytes) -> dict[str, Any]:
    _require(type(returncode) is int, "worker return code is invalid")
    if not output:
        if returncode != 0:
            raise WorkerProtocolFailure("worker_terminated_without_report")
        raise WorkerProtocolFailure("malformed_worker_report")
    try:
        report = _decode_worker_output(output)
    except Exception as exc:
        raise WorkerProtocolFailure("malformed_worker_report") from exc
    if returncode == 0 and report["status"] == "worker_pass":
        return report
    if returncode != 0 and report["status"] == "worker_failed":
        raise WorkerReportedFailure(report)
    raise WorkerProtocolFailure("worker_exit_status_disagreement")


def _worker_process(marker_digest: str, expected_commit: str, expected_uuid: str,
                    *, deadline: float) -> tuple[int, bytes, int]:
    command = [sys.executable, os.path.join(REPO_ROOT, RELATIVE_PATH), "--_worker",
               "--parent-pid", str(os.getpid()), "--marker-sha256", marker_digest,
               "--expected-commit", expected_commit, "--expected-gpu-uuid", expected_uuid]
    process = subprocess.Popen(
        command, cwd=REPO_ROOT, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, env=_worker_environment(expected_uuid),
        start_new_session=True,
    )
    try:
        code, output = _capture_process(
            process, deadline=deadline, cap=COMBINED_CHILD_OUTPUT_CAP,
            grace=TERMINATION_GRACE_SECONDS,
        )
    except BaseException as exc:
        # The controller uses this only to verify that its terminated child no
        # longer occupies the GPU.  It never signals a PID after this point.
        setattr(exc, "inspection_worker_pid", process.pid)
        raise
    return code, output, process.pid


def _assert_pid_absent_from_gpu(pid: int, expected_uuid: str, *, deadline: float) -> bool:
    output = _run_command(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
         "--format=csv,noheader,nounits"], cwd=REPO_ROOT, deadline=deadline,
    )
    return not any(uuid == expected_uuid and found_pid == pid
                   for uuid, found_pid, _name, _used in _parse_compute_rows(output))


def _failure_detail(exc: BaseException, *, controller_stage: str,
                    worker_started: bool) -> dict[str, Any]:
    _require(controller_stage in CONTROLLER_STAGES, "invalid controller failure stage")
    if isinstance(exc, WorkerReportedFailure):
        report = exc.report
        _require(tuple(report) == WORKER_FAILURE_KEYS,
                 "unvalidated structured worker failure")
        detail = {
            "origin": "worker", "classification": report["failure_class"],
            "reason": report["failure_reason"], "stage": report["failure_stage"],
            "inspector_cuda_initialization_state":
                report["inspector_cuda_initialization_state"],
            "worker_report_complete": True,
        }
    elif isinstance(exc, WorkerProtocolFailure):
        detail = {
            "origin": "worker_protocol", "classification": exc.classification,
            "reason": exc.classification, "stage": "worker_report_decode",
            "inspector_cuda_initialization_state": "unknown",
            "worker_report_complete": False,
        }
    elif getattr(exc, "inspection_worker_pid", None) is not None:
        classification = {
            "child deadline exceeded": "worker_process_timeout",
            "combined child output exceeded cap": "worker_output_limit",
        }.get(str(exc), "worker_process_cleanup_failure")
        detail = {
            "origin": "worker_process", "classification": classification,
            "reason": classification, "stage": "worker_process",
            "inspector_cuda_initialization_state": "unknown",
            "worker_report_complete": False,
        }
    else:
        detail = {
            "origin": "controller", "classification": "controller_guard_failure",
            "reason": "controller_guard_failure", "stage": controller_stage,
            "inspector_cuda_initialization_state":
                ("unknown" if worker_started else "not_attempted"),
            "worker_report_complete": False,
        }
    _require(tuple(detail) == FAILURE_DETAIL_KEYS,
             "invalid internal failure-detail shape")
    return detail


def _failure_record(detail: dict[str, Any], *, expected_commit: str,
                    expected_uuid: str, worker_pid: int | None,
                    worker_gpu_process_absent: bool | None) -> dict[str, Any]:
    _require(tuple(detail) == FAILURE_DETAIL_KEYS, "invalid failure detail")
    return {
        "schema": SCHEMA, "status": "failed", "scope": "engineering_native_layout_only",
        "expected_commit": expected_commit, "expected_gpu_uuid": expected_uuid,
        "failure": detail, "worker_pid": worker_pid,
        "worker_gpu_process_absent_after_exit": worker_gpu_process_absent,
        "automatic_retry": False, "backend_fallback": False, "cap_change": False,
        "certificates": {"scientific_execution": False, "pilot": False,
                         "full_fit": False, "capture_neutrality": False,
                         "continuation_restore": False},
    }


def run_controller(expected_commit: str, expected_uuid: str) -> int:
    """Run the fixed one-shot controller.  This is never called by default."""
    _require(_COMMIT.fullmatch(expected_commit) is not None, "invalid expected commit")
    _require(_GPU_UUID.fullmatch(expected_uuid) is not None, "invalid expected GPU UUID")
    _require("torch" not in sys.modules,
             "controller must be a fresh process with Torch unimported")
    started = time.monotonic()
    cpu_started = time.process_time()
    deadline = started + DEADLINE_SECONDS
    helper = _bind_helper(REPO_ROOT, expected_commit, deadline=deadline)
    _require(_peak_rss_bytes() <= RSS_CAP, "controller RSS cap exceeded during source binding")
    mount, parent_snapshot = _verify_attempt_parent(ATTEMPT_PARENT, deadline=deadline)
    _require(_peak_rss_bytes() <= RSS_CAP, "controller RSS cap exceeded during mount check")
    _require(not os.path.lexists(ATTEMPT_ROOT),
             "fixed inspection directory already exists; retry refused")
    rootfd, identity = _create_attempt_root()
    worker_pid = None
    gpu_absent = None
    worker_started = False
    controller_stage = "marker_publication"
    try:
        _check_path_snapshot(parent_snapshot)
        marker = {
            "schema": SCHEMA, "status": "started", "attempt_root": ATTEMPT_ROOT,
            "controller_pid": os.getpid(), "expected_commit": expected_commit,
            "expected_gpu_uuid": expected_uuid,
            "helper_binding_digest": hashlib.sha256(
                _canonical_json(helper, cap=FAILURE_RESERVE)).hexdigest(),
            "deadline_monotonic": float(deadline),
            "limits": {"total_output_bytes": TOTAL_OUTPUT_CAP,
                       "failure_reserve_bytes": FAILURE_RESERVE,
                       "combined_child_output_bytes": COMBINED_CHILD_OUTPUT_CAP,
                       "wall_seconds": DEADLINE_SECONDS, "termination_grace_seconds":
                           TERMINATION_GRACE_SECONDS,
                       "kill_disappear_seconds": KILL_DISAPPEAR_SECONDS,
                       "peak_rss_bytes": RSS_CAP,
                       "torch_peak_allocated_bytes": CUDA_ALLOCATED_CAP},
        }
        marker_bytes = _canonical_json(marker, cap=FAILURE_RESERVE)
        _write_new_json(rootfd, ATTEMPT_ROOT, identity, MARKER_NAME, marker,
                        cap=FAILURE_RESERVE, normal=True)
        marker_digest = hashlib.sha256(marker_bytes).hexdigest()
        controller_stage = "gpu_preflight"
        preflight = _gpu_preflight(expected_uuid, deadline=deadline)
        _require(_peak_rss_bytes() <= RSS_CAP, "controller RSS cap exceeded during GPU preflight")
        controller_stage = "worker_process"
        worker_started = True
        code, output, worker_pid = _worker_process(
            marker_digest, expected_commit, expected_uuid, deadline=deadline)
        controller_stage = "worker_report_decode"
        worker = _interpret_worker_exit(code, output)
        controller_stage = "worker_result_validation"
        _require(worker.get("expected_commit") == expected_commit and
                 worker.get("expected_gpu_uuid") == expected_uuid and
                 worker.get("helper_binding") == helper,
                 "native worker binding differs from controller")
        native_driver = worker["native_environment_binding"]["cuda"]["driver_version"]
        _require(native_driver == preflight["driver_version_observed"],
                 "preflight/native observed driver versions differ")
        preflight["native_driver_observation_consistent"] = True
        controller_stage = "gpu_exit_check"
        gpu_absent = _assert_pid_absent_from_gpu(worker_pid, expected_uuid,
                                                deadline=deadline)
        _require(gpu_absent, "worker GPU process remained after terminal exit")
        controller_stage = "helper_after"
        helper_after = _bind_helper(REPO_ROOT, expected_commit, deadline=deadline)
        _require(helper_after == helper, "source tree/helper changed across controller")
        _check_path_snapshot(parent_snapshot)
        controller_stage = "controller_resource_check"
        controller_rss = _peak_rss_bytes()
        _require(controller_rss <= RSS_CAP, "controller RSS cap exceeded")
        worker["resources"]["controller_wall_seconds"] = time.monotonic() - started
        worker["resources"]["controller_cpu_seconds"] = time.process_time() - cpu_started
        worker["resources"]["controller_peak_rss_bytes"] = controller_rss
        worker["resources"]["controller_timing_scope"] = (
            "controller process_time only; worker and helper subprocess CPU are excluded"
        )
        _require(worker["resources"]["controller_wall_seconds"] <= DEADLINE_SECONDS,
                 "controller wall deadline exceeded")
        result = {
            "schema": SCHEMA, "status": "pass", "scope": "engineering_native_layout_only",
            "attempt_root": ATTEMPT_ROOT, "mount_verification": mount,
            "gpu_preflight": preflight, "expected_commit": expected_commit,
            "expected_gpu_uuid": expected_uuid,
            "source_binding": worker["source_binding"],
            "native_environment_binding": worker["native_environment_binding"],
            "helper_binding": worker["helper_binding"],
            "resources": worker["resources"], "side_effects": worker["side_effects"],
            "worker_gpu_process_absent_after_exit": True,
            "certificates": worker["certificates"],
        }
        controller_stage = "result_publication"
        _write_new_json(rootfd, ATTEMPT_ROOT, identity, RESULT_NAME, result,
                        cap=NORMAL_OUTPUT_CAP, normal=True)
        print(json.dumps({"status": "pass", "result": ATTEMPT_ROOT + "/" + RESULT_NAME},
                         sort_keys=True, separators=(",", ":")))
        return 0
    except BaseException as exc:
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        failed_pid = getattr(exc, "inspection_worker_pid", None)
        if worker_pid is None and type(failed_pid) is int:
            worker_pid = failed_pid
        if worker_pid is not None and gpu_absent is None:
            try:
                # This is terminal cleanup after the 45-second work deadline,
                # analogous to the explicitly separate termination grace.
                gpu_absent = _assert_pid_absent_from_gpu(
                    worker_pid, expected_uuid, deadline=time.monotonic() + 5.0)
            except BaseException:
                gpu_absent = None
        detail = _failure_detail(exc, controller_stage=controller_stage,
                                 worker_started=worker_started)
        record = _failure_record(detail, expected_commit=expected_commit,
                                 expected_uuid=expected_uuid, worker_pid=worker_pid,
                                 worker_gpu_process_absent=gpu_absent)
        failure_published = False
        try:
            _write_new_json(rootfd, ATTEMPT_ROOT, identity, FAILURE_NAME, record,
                            cap=FAILURE_RESERVE, normal=False)
            failure_published = True
        except BaseException:
            # Existing outputs are never removed or overwritten to make room.
            pass
        print(json.dumps({"status": "failed", "failure_published": failure_published,
                         "failure": (ATTEMPT_ROOT + "/" + FAILURE_NAME
                                     if failure_published else None)},
                         sort_keys=True, separators=(",", ":")))
        return 1
    finally:
        os.close(rootfd)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inert by default; one-shot engineering native-layout inspector."
    )
    parser.add_argument("--inspect-native-layout", action="store_true")
    parser.add_argument("--expected-commit")
    parser.add_argument("--expected-gpu-uuid")
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--parent-pid", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--marker-sha256", help=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args._worker:
        progress = {"failure_stage": "worker_invocation_validation",
                    "inspector_cuda_initialization_state": "not_attempted"}
        try:
            _require(not args.inspect_native_layout and args.parent_pid is not None and
                     args.parent_pid == os.getppid() and
                     type(args.marker_sha256) is str and
                     _SHA256.fullmatch(args.marker_sha256) is not None and
                     type(args.expected_commit) is str and
                     type(args.expected_gpu_uuid) is str,
                     "invalid private worker invocation")
            value = _worker(args.marker_sha256, args.expected_commit,
                            args.expected_gpu_uuid, args.parent_pid, progress)
            _require(value["expected_commit"] == args.expected_commit and
                     value["expected_gpu_uuid"] == args.expected_gpu_uuid,
                     "private worker argument/marker mismatch")
            sys.stdout.buffer.write(_canonical_json(value, cap=COMBINED_CHILD_OUTPUT_CAP))
            return 0
        except BaseException as exc:
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            failure = _worker_failure_report(progress, exc)
            sys.stdout.buffer.write(_canonical_json(failure, cap=FAILURE_RESERVE))
            return 1
    if not args.inspect_native_layout:
        if not (args.expected_commit is None and args.expected_gpu_uuid is None and
                args.parent_pid is None and args.marker_sha256 is None):
            print(json.dumps({"status": "refused",
                              "failure": {"error_type": "InspectionError",
                                          "message": "inspection parameters require "
                                                     "--inspect-native-layout"}},
                             sort_keys=True, separators=(",", ":")))
            return 2
        print('{"status":"inert"}')
        return 0
    try:
        _require(args.parent_pid is None and args.marker_sha256 is None and
                 type(args.expected_commit) is str and type(args.expected_gpu_uuid) is str,
                 "native inspection requires expected commit and GPU UUID")
        return run_controller(args.expected_commit, args.expected_gpu_uuid)
    except BaseException as exc:
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        print(json.dumps({"status": "refused", "failure": _safe_error(exc)},
                         sort_keys=True, separators=(",", ":")))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
