"""Inert, one-shot outer process supervision for the fixed I7 native phases.

Import/default CLI perform no I/O and import no Torch-bearing project module.
The explicit native API persists one request before launching one fresh process,
observes its real exit, verifies a closed handoff, and never retries.
"""
from __future__ import annotations

from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import selectors
import signal
import stat
import subprocess
import sys
import time


class SupervisionError(RuntimeError):
    pass


REQUEST_SCHEMA = "i7_native_process_request_v2"
WORKER_REPORT_SCHEMA = "i7_native_phase_worker_report_v1"
EXIT_SCHEMA = "i7_native_process_exit_v1"
ATTEMPT_ID = "i7-native-development-attempt-001"
PHASES = ("development", "primary", "sensitivity", "audit")
SCOPES = {
    "development":"frozen_current_policy_development_pilot",
    "primary":"frozen_current32_primary_three_bundle_and_four_plan_preparation",
    "sensitivity":"frozen_current32_sensitivity_b71901",
    "audit":"frozen_independent_cpu_audit_16_groups",
}
PHASE_RECORDS = {"development":10, "primary":44, "sensitivity":56, "audit":90}
PHASE_WALL_SECONDS = {"development":180.0, "primary":600.0,
                      "sensitivity":240.0, "audit":600.0}
PHASE_CPU_SECONDS = {"development":None, "primary":None,
                     "sensitivity":None, "audit":600.0}

SUPERVISION_ROOT = "/tmp/spectral-experiment-artifacts/i7-native-development-attempt-001-supervision"
REPOSITORY_ROOT = "/private-artifacts/repositories/optimizers-i7-native-frozen-002"
I7_RELATIVE = ("output/2026-09-06-spectral-optimizer-investigation/"
               "continuation/iteration-007/")
WORKER_RELATIVE_PATH = I7_RELATIVE + "phase_worker.py"
WORKER_PATH = REPOSITORY_ROOT + "/" + WORKER_RELATIVE_PATH
PYTHON_EXECUTABLE = "/usr/bin/python3.12"
DATA_DIRECTORY = "data/MNIST/raw"
STORAGE_ADMISSION_PATH = (
    "" + I7_RELATIVE
    + "native-storage-admission.json")
LAYOUT_EVIDENCE_PATH = (
    "" + I7_RELATIVE
    + "native-max-layout-measurement-attempt-002.json")
EXPECTED_FILES = {
    "training_images":{"sha256":
        "ba891046e6505d7aadcbbe25680a0738ad16aec93bde7f9b65e87a2fc25776db",
        "size_bytes":47040016},
    "training_labels":{"sha256":
        "65a50cbbf4e906d70832878ad85ccda5333a97f0f4c3dd2ef09a8a9eef7101c5",
        "size_bytes":60008},
}

REQUEST_MAX = 8 << 10
EXIT_MAX = 16 << 10
WORKER_OUTPUT_MAX = 64 << 10
SUPERVISION_TOTAL_MAX = 4 * (REQUEST_MAX + EXIT_MAX)
SHARED_FAILURE_RESERVE = 1 << 20
ROOT_BUDGET = 1 << 30
MIN_FREE_BYTES = 1 << 30
STORAGE_ADMISSION_MAX = 64 << 10
RSS_CAP = 12 << 30
CUDA_ALLOCATED_CAP = 8 << 30
TERMINATION_GRACE_SECONDS = 5.0
KILL_DISAPPEAR_SECONDS = 1.0

PIN_FIELDS = ("path", "size_bytes", "sha256")
ADMISSION_FIELDS = ("schema", "status", "attempt_id", "expected_commit",
    "expected_source_set_sha256", "root_budget_bytes",
    "shared_failure_reserve_bytes", "supervision_total_bytes",
    "layout_evidence_pin", "decision", "retry_allowed",
    "execution_authorized", "scientific_execution_certified")
DATA_PIN_FIELDS = ("sha256", "size_bytes")
WORKER_BINDING_FIELDS = ("repository_root", "worker_path", "worker_relative_path",
    "python_executable", "repository_revision", "git_mode", "size_bytes", "sha256",
    "git_blob_oid", "data_directory", "expected_files")
LIMIT_FIELDS = ("wall_seconds", "cpu_seconds", "worker_output_bytes",
    "termination_grace_seconds", "kill_disappear_seconds", "peak_rss_bytes",
    "peak_cuda_allocated_bytes", "request_bytes", "exit_bytes",
    "supervision_total_bytes")
REQUEST_FIELDS = ("schema", "status", "attempt_id", "phase", "scope",
    "evidence_kind", "retry_allowed", "created_utc", "supervisor_pid",
    "supervisor_wall_origin", "deadline_monotonic", "expected_commit",
    "expected_source_set_sha256", "expected_environment_sha256",
    "expected_gpu_uuid", "authority_kind", "development_permission_ascii",
    "development_pins", "phase_permission_pin", "prior_boundary",
    "storage_admission_pin",
    "worker_binding", "limits")
RESOURCE_FIELDS = ("worker_entry_wall_origin", "worker_entry_cpu_origin",
    "worker_entry_process_id", "wall_seconds", "cpu_seconds", "peak_rss_bytes",
    "peak_cuda_allocated_bytes", "peak_cuda_reserved_bytes", "cuda_initialized")
FAILURE_FIELDS = ("stage", "reason", "retention")
WORKER_REPORT_FIELDS = ("schema", "status", "attempt_id", "phase",
    "request_sha256", "worker_pid", "handoff", "failure", "resources",
    "execution_authorized", "scientific_execution_certified")
EXIT_FIELDS = ("schema", "status", "attempt_id", "phase", "request_pin",
    "worker_binding", "supervisor_pid", "child_pid", "process_group_id",
    "observed_returncode", "exit_classification", "term_sent", "kill_sent",
    "group_disappearance_observed", "captured_output_bytes", "worker_report",
    "verified_handoff", "supervisor_wall_seconds", "retry_allowed",
    "execution_authorized", "scientific_execution_certified")
PRIOR_FIELDS = ("phase", "phase_records", "root_binding", "journal_pin",
                "boundary_ref", "inspection")
WORKER_STAGES = ("request_validation", "runtime_initialization",
    "metadata_collection", "storage_admission", "bootstrap", "acquire",
    "controller", "boundary", "worker_final")
WORKER_FAILURE_REASONS = ("worker_stage_failed", "resource_limit")
RETENTION_STATUSES = ("consumed_request_no_retry",)
EXIT_CLASSIFICATIONS = ("boundary_verified", "structured_worker_failure",
    "timeout", "output_limit", "signaled", "nonzero_without_valid_failure",
    "malformed_report", "exit_status_disagreement", "handoff_invalid",
    "launch_failed", "ownership_unverified", "cleanup_unverified")

_SHA = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)
_COMMIT = re.compile(r"[0-9a-f]{40}\Z", re.ASCII)
_BLOB = re.compile(r"[0-9a-f]{40}\Z", re.ASCII)
_UUID = re.compile(r"GPU-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}\Z", re.ASCII)
_UTC = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z\Z",
                  re.ASCII)


def _need(condition, message):
    if not condition:
        raise SupervisionError(message)


def request_name(phase):
    _need(type(phase) is str and phase in PHASES, "invalid supervised phase")
    return f"native-{phase}-process-request.json"


def exit_name(phase):
    _need(type(phase) is str and phase in PHASES, "invalid supervised phase")
    return f"native-{phase}-process-exit.json"


def supervision_names():
    return frozenset(name for phase in PHASES
                     for name in (request_name(phase), exit_name(phase)))


def fixed_limits(phase):
    _need(type(phase) is str and phase in PHASES, "invalid supervised phase")
    return {"wall_seconds":PHASE_WALL_SECONDS[phase],
        "cpu_seconds":PHASE_CPU_SECONDS[phase],
        "worker_output_bytes":WORKER_OUTPUT_MAX,
        "termination_grace_seconds":TERMINATION_GRACE_SECONDS,
        "kill_disappear_seconds":KILL_DISAPPEAR_SECONDS,
        "peak_rss_bytes":RSS_CAP,
        "peak_cuda_allocated_bytes":CUDA_ALLOCATED_CAP,
        "request_bytes":REQUEST_MAX, "exit_bytes":EXIT_MAX,
        "supervision_total_bytes":SUPERVISION_TOTAL_MAX}


def _keys(value, fields, label):
    _need(type(value) is dict and tuple(value) == fields
          and all(type(key) is str for key in value), label + ": exact ordered fields")


def _hash(value, pattern, label):
    _need(type(value) is str and pattern.fullmatch(value) is not None,
          label + ": invalid digest")


def _finite(value, label):
    _need(type(value) is float and math.isfinite(value) and value >= 0.0,
          label + ": invalid finite nonnegative float")


def _integer(value, label, *, positive=False):
    _need(type(value) is int and type(value) is not bool
          and value >= (1 if positive else 0), label + ": invalid integer")


def _utc(value):
    _need(type(value) is str and _UTC.fullmatch(value) is not None,
          "request UTC invalid")
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as exc:
        raise SupervisionError("request UTC invalid") from exc
    _need(parsed.strftime("%Y-%m-%dT%H:%M:%SZ") == value, "request UTC invalid")


def _tree(value, active=None):
    active = set() if active is None else active
    if value is None or type(value) in (bool, int, str):
        if type(value) is str:
            _need(not any(0xD800 <= ord(char) <= 0xDFFF for char in value),
                  "surrogate JSON string")
        return
    if type(value) is float:
        _need(math.isfinite(value), "nonfinite JSON number")
        return
    _need(type(value) in (list, dict) and id(value) not in active,
          "unsupported or cyclic JSON value")
    active.add(id(value))
    rows = value.items() if type(value) is dict else enumerate(value)
    for key, item in rows:
        if type(value) is dict:
            _need(type(key) is str, "JSON key must be string")
        _tree(item, active)
    active.remove(id(value))


def json_bytes(value, *, maximum):
    _tree(value)
    try:
        raw = (json.dumps(value, ensure_ascii=True, allow_nan=False, sort_keys=False,
                          separators=(",", ":")) + "\n").encode("ascii")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise SupervisionError("record is not canonical JSON") from exc
    _need(type(maximum) is int and 0 < len(raw) <= maximum,
          "record exceeds byte ceiling")
    return raw


def json_loads(raw, *, maximum):
    _need(type(raw) is bytes and 0 < len(raw) <= maximum, "bounded JSON required")
    try:
        text = raw.decode("ascii")
    except UnicodeDecodeError as exc:
        raise SupervisionError("JSON is not ASCII") from exc
    def pairs(rows):
        result = {}
        for key, value in rows:
            _need(key not in result, "duplicate JSON key")
            result[key] = value
        return result
    def constant(_):
        raise SupervisionError("nonfinite JSON constant")
    try:
        value = json.loads(text, object_pairs_hook=pairs, parse_constant=constant)
    except (json.JSONDecodeError, UnicodeError, ValueError, RecursionError) as exc:
        if isinstance(exc, SupervisionError):
            raise
        raise SupervisionError("malformed JSON") from exc
    _need(json_bytes(value, maximum=maximum) == raw, "JSON is not canonical")
    return value


def _pin(value, label, maximum):
    _keys(value, PIN_FIELDS, label)
    path = value["path"]
    _need(type(path) is str and path.startswith("/") and path != "/"
          and os.path.normpath(path) == path, label + ": invalid path")
    _integer(value["size_bytes"], label + " size", positive=True)
    _need(value["size_bytes"] <= maximum, label + ": size exceeds ceiling")
    _hash(value["sha256"], _SHA, label)


def load_pinned_bytes(path, size_bytes, sha256, *, maximum):
    """Read one exact regular file through a canonical nofollow descriptor walk."""
    pin = {"path":path, "size_bytes":size_bytes, "sha256":sha256}
    _pin(pin, "pinned file", maximum)
    parts = path.split("/")[1:]
    dirfd = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    fd = None
    try:
        for part in parts[:-1]:
            _need(part not in ("", ".", ".."), "invalid pinned path component")
            nextfd = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                             dir_fd=dirfd)
            os.close(dirfd)
            dirfd = nextfd
        fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                     dir_fd=dirfd)
        before = os.fstat(fd)
        _need(stat.S_ISREG(before.st_mode) and before.st_nlink == 1
              and before.st_size == size_bytes, "pinned file identity differs")
        chunks, remaining = [], size_bytes
        while remaining:
            part = os.read(fd, min(1 << 20, remaining))
            _need(bool(part), "pinned file short read")
            chunks.append(part)
            remaining -= len(part)
        _need(os.read(fd, 1) == b"", "pinned file grew")
        after = os.fstat(fd)
        at_path = os.stat(parts[-1], dir_fd=dirfd, follow_symlinks=False)
        signature = lambda row:(row.st_dev, row.st_ino, row.st_mode, row.st_nlink,
                                row.st_size, row.st_mtime_ns, row.st_ctime_ns)
        _need(signature(before) == signature(after) == signature(at_path),
              "pinned file changed during read")
    finally:
        if fd is not None:
            os.close(fd)
        os.close(dirfd)
    raw = b"".join(chunks)
    _need(hashlib.sha256(raw).hexdigest() == sha256, "pinned file hash differs")
    return raw


def _load_existing(path, *, maximum):
    """Pin and read one existing file without a path-following preliminary read."""
    _need(type(path) is str and path.startswith("/"), "existing path invalid")
    parts = path.split("/")[1:]
    dirfd = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    fd = None
    try:
        for part in parts[:-1]:
            _need(part not in ("", ".", ".."), "existing path component invalid")
            nextfd = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                             dir_fd=dirfd)
            os.close(dirfd)
            dirfd = nextfd
        fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                     dir_fd=dirfd)
        info = os.fstat(fd)
        _need(stat.S_ISREG(info.st_mode) and info.st_nlink == 1
              and 0 < info.st_size <= maximum, "existing file invalid")
        pin = {"path":path, "size_bytes":info.st_size, "sha256":"0" * 64}
        remaining, chunks = info.st_size, []
        while remaining:
            part = os.read(fd, min(1 << 20, remaining))
            _need(bool(part), "existing file short read")
            chunks.append(part)
            remaining -= len(part)
        _need(os.read(fd, 1) == b"", "existing file grew")
        after = os.fstat(fd)
        at_path = os.stat(parts[-1], dir_fd=dirfd, follow_symlinks=False)
        signature = lambda row:(row.st_dev, row.st_ino, row.st_mode, row.st_nlink,
                                row.st_size, row.st_mtime_ns, row.st_ctime_ns)
        _need(signature(info) == signature(after) == signature(at_path),
              "existing file changed during read")
    finally:
        if fd is not None:
            os.close(fd)
        os.close(dirfd)
    raw = b"".join(chunks)
    pin["sha256"] = hashlib.sha256(raw).hexdigest()
    return raw, pin


def validate_storage_admission(raw, pin, request, *, fixture=False):
    """Validate the reviewed storage gate; it is authority, not an execution result."""
    _pin(pin, "storage admission pin", STORAGE_ADMISSION_MAX)
    if not fixture:
        _need(pin["path"] == STORAGE_ADMISSION_PATH,
              "native storage admission path differs")
    _need(hashlib.sha256(raw).hexdigest() == pin["sha256"]
          and len(raw) == pin["size_bytes"], "storage admission pin differs")
    value = json_loads(raw, maximum=STORAGE_ADMISSION_MAX)
    if fixture:
        _need(type(value) is dict, "fixture storage admission must be an object")
        return value
    import native_storage_authority as authority
    try:
        return authority.validate_structural_admission(
            raw, pin, expected_commit=request["expected_commit"],
            expected_source_set_sha256=request["expected_source_set_sha256"])
    except authority.StorageAuthorityError as exc:
        raise SupervisionError("storage admission structure invalid") from exc


def _validate_expected_files(value, *, fixture):
    _need(type(value) is dict and tuple(value) == ("training_images", "training_labels"),
          "expected IDX files have wrong exact fields")
    sizes = (47040016, 60008) if not fixture else None
    for index, key in enumerate(value):
        pin = value[key]
        _keys(pin, DATA_PIN_FIELDS, "IDX pin")
        _hash(pin["sha256"], _SHA, "IDX pin")
        _integer(pin["size_bytes"], "IDX size", positive=True)
        if sizes is not None:
            _need(pin["size_bytes"] == sizes[index], "IDX size differs")
    if not fixture:
        _need(value == EXPECTED_FILES, "IDX pins differ from reviewed evidence")


def _validate_worker_binding(value, request, *, fixture):
    _keys(value, WORKER_BINDING_FIELDS, "worker binding")
    for key in ("repository_root", "worker_path", "worker_relative_path",
                "python_executable", "data_directory"):
        _need(type(value[key]) is str and value[key].isascii(),
              "worker path binding invalid")
    _hash(value["repository_revision"], _COMMIT, "worker revision")
    _need(value["repository_revision"] == request["expected_commit"],
          "worker revision differs from request")
    _need(value["git_mode"] == "100644", "worker git mode differs")
    _integer(value["size_bytes"], "worker size", positive=True)
    _hash(value["sha256"], _SHA, "worker")
    _hash(value["git_blob_oid"], _BLOB, "worker blob")
    _validate_expected_files(value["expected_files"], fixture=fixture)
    if fixture:
        _need(value["python_executable"] == PYTHON_EXECUTABLE,
              "fixture interpreter differs")
    else:
        _need(value["repository_root"] == REPOSITORY_ROOT
              and value["worker_path"] == WORKER_PATH
              and value["worker_relative_path"] == WORKER_RELATIVE_PATH
              and value["python_executable"] == PYTHON_EXECUTABLE
              and value["data_directory"] == DATA_DIRECTORY,
              "native worker/data path differs")


def _authenticate_worker_binding(value, request):
    """Bind the frozen tree pre-launch; explicit use may import CPU-only Torch."""
    _need(os.path.realpath(PYTHON_EXECUTABLE) == PYTHON_EXECUTABLE,
          "native interpreter is not its canonical path")
    python_info = os.stat(PYTHON_EXECUTABLE, follow_symlinks=False)
    _need(stat.S_ISREG(python_info.st_mode) and python_info.st_nlink == 1
          and os.access(PYTHON_EXECUTABLE, os.X_OK), "native interpreter invalid")
    existing_torch = sys.modules.get("torch")
    _need(existing_torch is None or not existing_torch.cuda.is_initialized(),
          "supervisor CUDA was initialized before source authentication")
    import identity_codec as codec
    import source_environment as environment
    import torch
    _need(not torch.cuda.is_initialized(),
          "source authentication initialized supervisor CUDA")
    sources = environment.collect_verified_sources(
        value["repository_root"], profile=environment.SCIENTIFIC)
    _need(not torch.cuda.is_initialized(),
          "source collection initialized supervisor CUDA")
    _need(sources["repository_revision"] == request["expected_commit"]
          and sources["repository_root_realpath"] == REPOSITORY_ROOT
          and codec.tree_digest(sources) == request["expected_source_set_sha256"],
          "actual frozen source binding differs")
    matches = [row for row in sources["files"]
               if row["path"] == WORKER_RELATIVE_PATH]
    _need(len(matches) == 1, "phase worker absent from exact source membership")
    row = matches[0]
    _need(row["git_mode"] == value["git_mode"]
          and row["size_bytes"] == value["size_bytes"]
          and row["sha256"] == value["sha256"]
          and row["git_blob_oid"] == value["git_blob_oid"],
          "actual phase worker bytes differ")


def _validate_limits(value, phase, *, fixture):
    _keys(value, LIMIT_FIELDS, "supervision limits")
    expected = fixed_limits(phase)
    for key in ("wall_seconds", "termination_grace_seconds",
                "kill_disappear_seconds"):
        _finite(value[key], key)
    _need(value["cpu_seconds"] is None
          or (type(value["cpu_seconds"]) is float
              and math.isfinite(value["cpu_seconds"])
              and value["cpu_seconds"] >= 0.0), "CPU limit invalid")
    for key in ("worker_output_bytes", "peak_rss_bytes",
                "peak_cuda_allocated_bytes", "request_bytes", "exit_bytes",
                "supervision_total_bytes"):
        _integer(value[key], key, positive=True)
    if fixture:
        for key in ("wall_seconds", "termination_grace_seconds",
                    "kill_disappear_seconds"):
            _finite(value[key], "fixture " + key)
            _need(value[key] > 0.0 and value[key] <= expected[key],
                  "fixture time limit outside native ceiling")
        _need(value["cpu_seconds"] is None or
              (type(value["cpu_seconds"]) is float and value["cpu_seconds"] >= 0.0),
              "fixture CPU limit invalid")
        for key in LIMIT_FIELDS[2:3] + LIMIT_FIELDS[5:]:
            _integer(value[key], "fixture " + key, positive=True)
        _need(value["worker_output_bytes"] <= WORKER_OUTPUT_MAX
              and value["peak_rss_bytes"] <= RSS_CAP
              and value["peak_cuda_allocated_bytes"] <= CUDA_ALLOCATED_CAP
              and value["request_bytes"] == REQUEST_MAX
              and value["exit_bytes"] == EXIT_MAX
              and value["supervision_total_bytes"] == SUPERVISION_TOTAL_MAX,
              "fixture byte/resource ceiling differs")
    else:
        _need(value == expected, "native supervision limits differ")


def validate_request(raw, *, expected_sha256, expected_phase=None, fixture=False,
                     expected_supervisor_pid=None, authenticate=True):
    """Validate one exact persisted request; optionally authenticate its authority."""
    _need(type(fixture) is bool and type(authenticate) is bool, "request mode invalid")
    _hash(expected_sha256, _SHA, "expected request")
    _need(hashlib.sha256(raw).hexdigest() == expected_sha256,
          "request hash differs from reviewed pin")
    value = json_loads(raw, maximum=REQUEST_MAX)
    _keys(value, REQUEST_FIELDS, "process request")
    phase = value["phase"]
    _need(type(phase) is str and phase in PHASES
          and (expected_phase is None or phase == expected_phase),
          "request phase differs")
    evidence = "synthetic_contract_fixture" if fixture else "native_producer_attestation"
    _need(value["schema"] == REQUEST_SCHEMA and value["status"] == "authorized_once"
          and value["attempt_id"] == ATTEMPT_ID and value["scope"] == SCOPES[phase]
          and value["evidence_kind"] == evidence and value["retry_allowed"] is False,
          "request scope/authority differs")
    _utc(value["created_utc"])
    _integer(value["supervisor_pid"], "supervisor PID", positive=True)
    if expected_supervisor_pid is not None:
        _need(value["supervisor_pid"] == expected_supervisor_pid,
              "request supervisor PID differs")
    _finite(value["supervisor_wall_origin"], "supervisor wall origin")
    _finite(value["deadline_monotonic"], "supervisor deadline")
    _validate_limits(value["limits"], phase, fixture=fixture)
    _need(value["deadline_monotonic"]
          == value["supervisor_wall_origin"] + value["limits"]["wall_seconds"],
          "request deadline differs from entry origin")
    _hash(value["expected_commit"], _COMMIT, "expected commit")
    _hash(value["expected_source_set_sha256"], _SHA, "expected source set")
    if phase == "development":
        _need(value["expected_environment_sha256"] is None
              and type(value["expected_gpu_uuid"]) is str
              and _UUID.fullmatch(value["expected_gpu_uuid"]) is not None,
              "development environment/GPU expectation differs")
    else:
        _hash(value["expected_environment_sha256"], _SHA,
              "expected phase environment")
        if phase == "audit":
            _need(value["expected_gpu_uuid"] is None,
                  "audit request must omit GPU UUID")
        else:
            _need(type(value["expected_gpu_uuid"]) is str
                  and _UUID.fullmatch(value["expected_gpu_uuid"]) is not None,
                  "phase GPU UUID invalid")
    _validate_worker_binding(value["worker_binding"], value, fixture=fixture)
    _pin(value["storage_admission_pin"], "storage admission pin",
         STORAGE_ADMISSION_MAX)

    if fixture:
        _need(value["authority_kind"] == "fixture"
              and value["development_permission_ascii"] is None
              and value["development_pins"] is None
              and value["phase_permission_pin"] is None,
              "fixture request carries native authority")
        if phase == "development":
            _need(value["prior_boundary"] is None,
                  "development fixture has prior boundary")
        else:
            _keys(value["prior_boundary"], PRIOR_FIELDS, "fixture prior boundary")
            previous = PHASES[PHASES.index(phase) - 1]
            _need(value["prior_boundary"]["phase"] == previous
                  and value["prior_boundary"]["phase_records"]
                      == PHASE_RECORDS[previous],
                  "fixture prior boundary differs")
        return value

    if phase == "development":
        _need(value["authority_kind"] == "development_launcher"
              and type(value["development_permission_ascii"]) is str
              and value["development_permission_ascii"].isascii()
              and value["phase_permission_pin"] is None
              and value["prior_boundary"] is None,
              "development authority fields differ")
        permission_raw = value["development_permission_ascii"].encode("ascii")
        _need(0 < len(permission_raw) <= REQUEST_MAX, "development permission size invalid")
        if authenticate:
            import native_control as control
            development_pins = control._validate_pins(value["development_pins"])
            _need(development_pins["attempt_directory"]
                      == control.DEVELOPMENT_ATTEMPT_PATH,
                  "development attempt directory differs from fixed path")
            permission = control._permission(
                permission_raw, development_pins, fixture=False)
            _need(permission["expected_commit"] == value["expected_commit"]
                  and permission["expected_source_set_sha256"]
                      == value["expected_source_set_sha256"]
                  and permission["expected_gpu_uuid"] == value["expected_gpu_uuid"]
                  and permission["storage_admission_pin"]
                      == value["storage_admission_pin"],
                  "development permission differs from request")
    else:
        _need(value["authority_kind"] == "phase_permission_pin"
              and value["development_permission_ascii"] is None
              and value["development_pins"] is None,
              "later phase authority fields differ")
        _pin(value["phase_permission_pin"], "phase permission pin", REQUEST_MAX)
        _keys(value["prior_boundary"], PRIOR_FIELDS, "prior boundary")
        if authenticate:
            import phase_transition as transition
            _raw, permission = transition._authenticate_actual(
                value["phase_permission_pin"], phase, value["prior_boundary"],
                "native_producer_attestation")
            _need(permission["expected_commit"] == value["expected_commit"]
                  and permission["expected_source_set_sha256"]
                      == value["expected_source_set_sha256"]
                  and permission["expected_environment_sha256"]
                      == value["expected_environment_sha256"]
                  and permission["expected_gpu_uuid"] == value["expected_gpu_uuid"]
                  and permission["storage_admission_pin"]
                      == value["storage_admission_pin"],
                  "phase permission differs from request")
    if authenticate:
        admission_raw = load_pinned_bytes(**value["storage_admission_pin"],
                                          maximum=STORAGE_ADMISSION_MAX)
        validate_storage_admission(admission_raw, value["storage_admission_pin"], value)
    return value


def load_pinned_request(path, size_bytes, sha256, *, expected_phase=None,
                        fixture=False, expected_supervisor_pid=None,
                        authenticate=True):
    raw = load_pinned_bytes(path, size_bytes, sha256, maximum=REQUEST_MAX)
    value = validate_request(raw, expected_sha256=sha256,
        expected_phase=expected_phase, fixture=fixture,
        expected_supervisor_pid=expected_supervisor_pid,
        authenticate=authenticate)
    if not fixture:
        _need(path == str(Path(SUPERVISION_ROOT) / request_name(value["phase"])),
              "native request path differs")
    return raw, value


def encode_request(value, *, fixture=False):
    """Encode a complete predeclared request; this does not consume its namespace."""
    raw = json_bytes(value, maximum=REQUEST_MAX)
    validate_request(raw, expected_sha256=hashlib.sha256(raw).hexdigest(),
                     expected_supervisor_pid=value.get("supervisor_pid"),
                     fixture=fixture, authenticate=False)
    return raw


def sanitized_environment(request):
    """Return a fresh allowlisted worker environment; never copy ambient values."""
    cuda = "" if request["phase"] == "audit" else request["expected_gpu_uuid"]
    return {"PATH":"/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "LANG":"C.UTF-8", "LC_ALL":"C.UTF-8", "TZ":"UTC",
        "TMPDIR":"/tmp/spectral-experiment-artifacts", "PYTHONDONTWRITEBYTECODE":"1",
        "PYTHONHASHSEED":"0", "OMP_NUM_THREADS":"1", "MKL_NUM_THREADS":"1",
        "OPENBLAS_NUM_THREADS":"1", "NUMEXPR_NUM_THREADS":"1",
        "CUBLAS_WORKSPACE_CONFIG":":4096:8", "CUDA_VISIBLE_DEVICES":cuda}


def _resources(value, request, *, child_pid, enforce_limits):
    _keys(value, RESOURCE_FIELDS, "worker resources")
    for key in ("worker_entry_wall_origin", "worker_entry_cpu_origin",
                "wall_seconds", "cpu_seconds"):
        _finite(value[key], "worker " + key)
    _integer(value["worker_entry_process_id"], "worker entry PID", positive=True)
    _need(value["worker_entry_process_id"] == child_pid,
          "worker resource PID differs")
    for key in ("peak_rss_bytes", "peak_cuda_allocated_bytes",
                "peak_cuda_reserved_bytes"):
        _integer(value[key], "worker " + key)
    _need(value["peak_cuda_allocated_bytes"] <= value["peak_cuda_reserved_bytes"]
          and type(value["cuda_initialized"]) is bool,
          "worker resource relationship differs")
    limits = request["limits"]
    if enforce_limits:
        _need(value["wall_seconds"] <= limits["wall_seconds"]
              and value["peak_rss_bytes"] <= limits["peak_rss_bytes"]
              and value["peak_cuda_allocated_bytes"] <= limits["peak_cuda_allocated_bytes"],
              "worker resource cap exceeded")
        cpu_cap = limits["cpu_seconds"]
        _need(cpu_cap is None or value["cpu_seconds"] <= cpu_cap,
              "worker CPU cap exceeded")
        expected_cuda = (request["evidence_kind"] != "synthetic_contract_fixture"
                         and request["phase"] != "audit")
        _need(value["cuda_initialized"] is expected_cuda,
              "successful worker CUDA lifecycle differs")


def validate_worker_report(raw, request, *, child_pid):
    value = json_loads(raw, maximum=request["limits"]["worker_output_bytes"])
    _keys(value, WORKER_REPORT_FIELDS, "worker report")
    _integer(child_pid, "observed child PID", positive=True)
    _need(value["schema"] == WORKER_REPORT_SCHEMA
          and value["attempt_id"] == ATTEMPT_ID
          and value["phase"] == request["phase"]
          and value["request_sha256"] == hashlib.sha256(
              json_bytes(request, maximum=REQUEST_MAX)).hexdigest()
          and value["worker_pid"] == child_pid
          and value["execution_authorized"] is False
          and value["scientific_execution_certified"] is False,
          "worker report binding differs")
    if value["status"] == "boundary_closed":
        _need(value["failure"] is None and value["resources"] is not None,
              "successful worker report has failure/missing resources")
        _resources(value["resources"], request, child_pid=child_pid,
                   enforce_limits=True)
        _keys(value["handoff"], PRIOR_FIELDS, "worker handoff")
        _need(value["handoff"]["phase"] == request["phase"]
              and value["handoff"]["phase_records"] == PHASE_RECORDS[request["phase"]],
              "worker handoff phase/count differs")
    else:
        _need(value["status"] == "worker_failed" and value["handoff"] is None,
              "worker report status differs")
        if value["resources"] is not None:
            _resources(value["resources"], request, child_pid=child_pid,
                       enforce_limits=False)
        _keys(value["failure"], FAILURE_FIELDS, "worker failure")
        _need(value["failure"]["stage"] in WORKER_STAGES
              and value["failure"]["reason"] in WORKER_FAILURE_REASONS
              and value["failure"]["retention"] in RETENTION_STATUSES,
              "worker failure declaration differs")
    return value


def encode_worker_report(value, request):
    """Worker-side exact encoder; validation still occurs in the supervisor."""
    raw = json_bytes(value, maximum=request["limits"]["worker_output_bytes"])
    validate_worker_report(raw, request, child_pid=value.get("worker_pid"))
    return raw


def validate_exit_record(raw, request):
    """Validate one complete persisted process-exit observation."""
    value = json_loads(raw, maximum=EXIT_MAX)
    _keys(value, EXIT_FIELDS, "process exit")
    _need(value["schema"] == EXIT_SCHEMA and value["attempt_id"] == ATTEMPT_ID
          and value["phase"] == request["phase"]
          and value["worker_binding"] == request["worker_binding"]
          and value["supervisor_pid"] == request["supervisor_pid"]
          and value["retry_allowed"] is False
          and value["execution_authorized"] is False
          and value["scientific_execution_certified"] is False,
          "process exit binding differs")
    _pin(value["request_pin"], "exit request pin", REQUEST_MAX)
    _integer(value["captured_output_bytes"], "captured output bytes")
    _need(value["captured_output_bytes"]
          <= request["limits"]["worker_output_bytes"] + 1,
          "captured output observation exceeds bounded read")
    _finite(value["supervisor_wall_seconds"], "supervisor wall elapsed")
    _need(value["exit_classification"] in EXIT_CLASSIFICATIONS
          and type(value["term_sent"]) is bool
          and type(value["kill_sent"]) is bool
          and (value["group_disappearance_observed"] is None
               or type(value["group_disappearance_observed"]) is bool)
          and type(value["verified_handoff"]) is bool,
          "process exit observation invalid")
    for key in ("child_pid", "process_group_id"):
        _need(value[key] is None or (type(value[key]) is int
              and type(value[key]) is not bool and value[key] > 0),
              "process identity observation invalid")
    _need(value["observed_returncode"] is None
          or (type(value["observed_returncode"]) is int
              and type(value["observed_returncode"]) is not bool),
          "process return code invalid")
    if value["worker_report"] is not None:
        _need(value["child_pid"] is not None, "worker report lacks child PID")
        encoded = json_bytes(value["worker_report"],
                             maximum=request["limits"]["worker_output_bytes"])
        validate_worker_report(encoded, request, child_pid=value["child_pid"])
    if value["status"] == "boundary_verified":
        _need(value["exit_classification"] == "boundary_verified"
              and value["observed_returncode"] == 0
              and value["process_group_id"] == value["child_pid"]
              and value["group_disappearance_observed"] is True
              and value["term_sent"] is False and value["kill_sent"] is False
              and value["worker_report"] is not None
              and value["worker_report"]["status"] == "boundary_closed"
              and value["verified_handoff"] is True,
              "successful exit evidence differs")
    else:
        _need(value["status"] == "terminal_failure"
              and value["exit_classification"] != "boundary_verified"
              and value["verified_handoff"] is False,
              "failure exit evidence differs")
        classification = value["exit_classification"]
        report = value["worker_report"]
        code = value["observed_returncode"]
        if classification == "structured_worker_failure":
            _need(code == 1 and report is not None
                  and report["status"] == "worker_failed"
                  and value["group_disappearance_observed"] is True,
                  "structured failure exit differs")
        elif classification == "signaled":
            _need(type(code) is int and code < 0 and report is None,
                  "signaled exit differs")
        elif classification == "nonzero_without_valid_failure":
            _need(type(code) is int and code > 0 and report is None,
                  "unstructured nonzero exit differs")
        elif classification == "malformed_report":
            _need(code == 0 and report is None, "malformed-report exit differs")
        elif classification == "launch_failed":
            _need(value["child_pid"] is None and code is None and report is None,
                  "launch-failed exit differs")
        elif classification == "ownership_unverified":
            _need(value["child_pid"] is not None
                  and value["process_group_id"] is None and report is None,
                  "ownership-unverified exit differs")
        elif classification == "cleanup_unverified":
            _need(value["child_pid"] is not None
                  and value["group_disappearance_observed"] is not True,
                  "cleanup-unverified exit differs")
        elif classification == "handoff_invalid":
            _need(code == 0 and report is not None
                  and report["status"] == "boundary_closed",
                  "invalid-handoff exit differs")
    return value


def _write_new(dirfd, name, raw, maximum):
    _need(type(name) is str and name in supervision_names(), "invalid supervision filename")
    _need(type(raw) is bytes and 0 < len(raw) <= maximum, "supervision write size invalid")
    fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                 0o600, dir_fd=dirfd)
    try:
        view = memoryview(raw)
        while view:
            count = os.write(fd, view)
            _need(count > 0, "short supervision write")
            view = view[count:]
        os.fsync(fd)
        info = os.fstat(fd)
        _need(stat.S_ISREG(info.st_mode) and info.st_nlink == 1
              and info.st_size == len(raw), "new supervision file invalid")
    finally:
        os.close(fd)
    os.fsync(dirfd)
    return {"path":"", "size_bytes":len(raw),
            "sha256":hashlib.sha256(raw).hexdigest()}


def _namespace(path, request, *, fixture):
    namespace = Path(path)
    parent = namespace.parent
    if not fixture:
        _need(str(namespace) == SUPERVISION_ROOT, "native supervision namespace differs")
    import native_control as control
    control._verify_big_parent(parent)
    _need(os.statvfs(parent).f_bavail * os.statvfs(parent).f_frsize >= MIN_FREE_BYTES,
          "insufficient supervision filesystem headroom")
    phase_index = PHASES.index(request["phase"])
    if phase_index == 0:
        parent_info = os.lstat(parent)
        parentfd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            opened_parent = os.fstat(parentfd)
            _need((opened_parent.st_dev, opened_parent.st_ino)
                  == (parent_info.st_dev, parent_info.st_ino),
                  "supervision parent changed during open")
            try:
                os.mkdir(namespace.name, 0o700, dir_fd=parentfd)
            except FileExistsError as exc:
                raise SupervisionError(
                    "development supervision namespace already consumed") from exc
            os.fsync(parentfd)
        finally:
            os.close(parentfd)
    info = os.lstat(namespace)
    _need(stat.S_ISDIR(info.st_mode) and not stat.S_ISLNK(info.st_mode)
          and info.st_nlink >= 2, "supervision namespace invalid")
    dirfd = os.open(namespace, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    opened = os.fstat(dirfd)
    try:
        _need((opened.st_dev, opened.st_ino) == (info.st_dev, info.st_ino),
              "supervision namespace changed during open")
        names = {entry.name for entry in os.scandir(dirfd)}
        expected = {name for prior in PHASES[:phase_index]
                    for name in (request_name(prior), exit_name(prior))}
        _need(names == expected, "supervision phase membership differs")
        previous_report = None
        for prior in PHASES[:phase_index]:
            prior_request_path = str(namespace / request_name(prior))
            prior_request_raw, prior_request_pin = _load_existing(
                prior_request_path, maximum=REQUEST_MAX)
            prior_request = validate_request(prior_request_raw,
                expected_sha256=prior_request_pin["sha256"], expected_phase=prior,
                fixture=fixture, authenticate=False)
            _need(prior_request["expected_commit"] == request["expected_commit"]
                  and prior_request["expected_source_set_sha256"]
                      == request["expected_source_set_sha256"]
                  and prior_request["storage_admission_pin"]
                      == request["storage_admission_pin"]
                  and prior_request["worker_binding"] == request["worker_binding"],
                  "prior supervised request source/admission binding differs")
            prior_exit_path = str(namespace / exit_name(prior))
            prior_exit_raw, _ = _load_existing(prior_exit_path, maximum=EXIT_MAX)
            row = validate_exit_record(prior_exit_raw, prior_request)
            _need(row["request_pin"] == prior_request_pin,
                  "prior exit does not pin its actual request")
            previous_report = row["worker_report"]
        if phase_index:
            _need(previous_report is not None
                  and request["prior_boundary"] == previous_report["handoff"],
                  "current request does not bind prior supervised handoff")
        return namespace, dirfd
    except BaseException:
        os.close(dirfd)
        raise


def _proc_identity(pid):
    try:
        raw = Path(f"/proc/{pid}/stat").read_text("ascii")
        close = raw.rfind(")")
        fields = raw[close + 2:].split()
        if close <= 0 or len(fields) < 20:
            raise SupervisionError("process stat malformed")
        return {"state":fields[0], "parent":int(fields[1]), "group":int(fields[2]),
                "session":int(fields[3]), "start":int(fields[19])}
    except (FileNotFoundError, ProcessLookupError):
        return None
    except (OSError, UnicodeError, ValueError, IndexError) as exc:
        raise SupervisionError("process identity could not be verified") from exc


def _live_group_members(group):
    result = []
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        row = _proc_identity(int(name))
        if (row is not None and row["group"] == group
                and row["session"] == group and row["state"] != "Z"):
            result.append(int(name))
    return result


def _waitid_exited(pidfd):
    try:
        return os.waitid(os.P_PIDFD, pidfd, os.WEXITED | os.WNOHANG | os.WNOWAIT)
    except ChildProcessError:
        return None


def _capture_owned(process, *, deadline, output_cap, grace, disappear):
    pid, pidfd, selector = process.pid, None, None
    output = bytearray()
    term_sent = kill_sent = False
    eof = exited = False
    ownership = False
    classification = None
    returncode = None
    group_gone = None
    reaped = False
    try:
        try:
            pidfd = os.pidfd_open(pid, 0)
            identity = _proc_identity(pid)
            ownership = (identity is not None and identity["parent"] == os.getpid()
                         and identity["group"] == pid
                         and identity["session"] == pid and os.getpgid(pid) == pid
                         and os.getsid(pid) == pid)
        except (OSError, SupervisionError):
            ownership = False
        if not ownership:
            classification = "ownership_unverified"
            if pidfd is not None:
                try:
                    signal.pidfd_send_signal(pidfd, signal.SIGKILL)
                    kill_sent = True
                except OSError:
                    pass
        else:
            selector = selectors.DefaultSelector()
            selector.register(process.stdout, selectors.EVENT_READ)
            os.set_blocking(process.stdout.fileno(), False)
            while classification is None:
                exited = _waitid_exited(pidfd) is not None
                room = output_cap + 1 - len(output)
                if room > 0:
                    try:
                        chunk = os.read(process.stdout.fileno(), min(1 << 14, room))
                    except BlockingIOError:
                        chunk = None
                    if chunk == b"":
                        eof = True
                    elif chunk:
                        output.extend(chunk)
                        if len(output) > output_cap:
                            classification = "output_limit"
                            break
                if exited and eof and not _live_group_members(pid):
                    break
                if time.monotonic() >= deadline:
                    classification = "timeout"
                    break
                selector.select(min(0.05, max(0.0, deadline - time.monotonic())))

            members = _live_group_members(pid)
            if members:
                try:
                    os.killpg(pid, signal.SIGTERM)
                    term_sent = True
                except ProcessLookupError:
                    pass
                grace_end = time.monotonic() + grace
                while _live_group_members(pid) and time.monotonic() < grace_end:
                    exited = _waitid_exited(pidfd) is not None
                    time.sleep(min(0.02, max(0.0, grace_end - time.monotonic())))
                if _live_group_members(pid):
                    try:
                        os.killpg(pid, signal.SIGKILL)
                        kill_sent = True
                    except ProcessLookupError:
                        pass
                    kill_end = time.monotonic() + disappear
                    while _live_group_members(pid) and time.monotonic() < kill_end:
                        time.sleep(min(0.02, max(0.0, kill_end - time.monotonic())))

        group_gone = ownership and not _live_group_members(pid)
        if ownership and not group_gone:
            classification = "cleanup_unverified"
        if ownership:
            room = output_cap + 1 - len(output)
            while room > 0:
                try:
                    chunk = os.read(process.stdout.fileno(), min(1 << 14, room))
                except BlockingIOError:
                    break
                if not chunk:
                    break
                output.extend(chunk)
                room = output_cap + 1 - len(output)
        if pidfd is not None:
            wait_end = time.monotonic() + max(1.0, disappear)
            while _waitid_exited(pidfd) is None and time.monotonic() < wait_end:
                time.sleep(0.01)
            if _waitid_exited(pidfd) is not None:
                returncode = process.wait(timeout=0)
                reaped = True
            else:
                classification = ("cleanup_unverified" if ownership
                                  else "ownership_unverified")
        if len(output) > output_cap and classification is None:
            classification = "output_limit"
        return {"output":bytes(output[:output_cap]), "output_bytes":len(output),
                "returncode":returncode, "classification":classification,
                "term_sent":term_sent, "kill_sent":kill_sent,
                "group_gone":group_gone, "pid":pid,
                "pgid":pid if ownership else None}
    except Exception:
        if ownership and not reaped:
            try:
                os.killpg(pid, signal.SIGKILL)
                kill_sent = True
            except OSError:
                pass
        elif pidfd is not None:
            try:
                signal.pidfd_send_signal(pidfd, signal.SIGKILL)
                kill_sent = True
            except OSError:
                pass
        if pidfd is not None:
            wait_end = time.monotonic() + max(1.0, disappear)
            while _waitid_exited(pidfd) is None and time.monotonic() < wait_end:
                time.sleep(0.01)
            if _waitid_exited(pidfd) is not None:
                try:
                    returncode = process.wait(timeout=0)
                    reaped = True
                except (OSError, subprocess.SubprocessError):
                    returncode = None
        if ownership and not reaped:
            group_gone = not _live_group_members(pid)
        return {"output":bytes(output[:output_cap]), "output_bytes":len(output),
                "returncode":returncode, "classification":"cleanup_unverified",
                "term_sent":term_sent, "kill_sent":kill_sent,
                "group_gone":group_gone, "pid":pid,
                "pgid":pid if ownership else None}
    finally:
        if selector is not None:
            selector.close()
        if pidfd is not None:
            os.close(pidfd)
        if process.stdout is not None:
            process.stdout.close()


def _verify_native_handoff(request, report):
    import artifact_store as storage
    import native_control as control
    handoff = report["handoff"]
    inspected = control.inspect_native_attempt(
        handoff["root_binding"]["path"], handoff["journal_pin"],
        expected_store_profile=storage.SCIENTIFIC,
        expected_evidence_kind="native_producer_attestation")
    expected_status = "complete" if request["phase"] == "audit" else "sealed_boundary"
    _need(inspected == handoff["inspection"] and inspected["status"] == expected_status
          and inspected["phase_records"] == PHASE_RECORDS[request["phase"]]
          and inspected["eligible_for_writable_reopen"]
              is (request["phase"] != "audit"),
          "worker closed handoff failed independent inspection")
    if request["phase"] != "development":
        _need(handoff["root_binding"] == request["prior_boundary"]["root_binding"]
              and handoff["journal_pin"] == request["prior_boundary"]["journal_pin"],
              "worker handoff changed root/journal")
    return True


def _exit_record(request, request_pin, outcome, report, classification, verified, elapsed):
    return {"schema":EXIT_SCHEMA,
        "status":"boundary_verified" if classification == "boundary_verified"
                 else "terminal_failure",
        "attempt_id":ATTEMPT_ID, "phase":request["phase"],
        "request_pin":request_pin, "worker_binding":request["worker_binding"],
        "supervisor_pid":request["supervisor_pid"],
        "child_pid":outcome.get("pid"), "process_group_id":outcome.get("pgid"),
        "observed_returncode":outcome.get("returncode"),
        "exit_classification":classification,
        "term_sent":outcome.get("term_sent", False),
        "kill_sent":outcome.get("kill_sent", False),
        "group_disappearance_observed":outcome.get("group_gone"),
        "captured_output_bytes":outcome.get("output_bytes", 0),
        "worker_report":report, "verified_handoff":verified,
        "supervisor_wall_seconds":elapsed, "retry_allowed":False,
        "execution_authorized":False, "scientific_execution_certified":False}


def _supervise(request_raw, request, *, namespace, argv, environment,
               verify_handoff, fixture):
    namespace, dirfd = _namespace(namespace, request, fixture=fixture)
    try:
        request_pin = _write_new(dirfd, request_name(request["phase"]),
                                 request_raw, REQUEST_MAX)
        request_pin["path"] = str(namespace / request_name(request["phase"]))
        outcome = {"pid":None, "pgid":None, "returncode":None,
            "classification":"launch_failed", "term_sent":False, "kill_sent":False,
            "group_gone":None, "output":b"", "output_bytes":0}
        try:
            if not fixture:
                _authenticate_worker_binding(request["worker_binding"], request)
            _need(time.monotonic() < request["deadline_monotonic"],
                  "supervision deadline expired before process launch")
            process = subprocess.Popen(argv, cwd=request["worker_binding"]["repository_root"],
                env=environment, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, close_fds=True, start_new_session=True)
            outcome = _capture_owned(process,
                deadline=request["deadline_monotonic"],
                output_cap=request["limits"]["worker_output_bytes"],
                grace=request["limits"]["termination_grace_seconds"],
                disappear=request["limits"]["kill_disappear_seconds"])
        except Exception:
            pass

        report = None
        classification = outcome["classification"]
        verified = False
        if classification is None:
            try:
                report = validate_worker_report(
                    outcome["output"], request, child_pid=outcome["pid"])
            except (SupervisionError, KeyError, TypeError, RecursionError):
                returncode = outcome["returncode"]
                if type(returncode) is int and returncode < 0:
                    classification = "signaled"
                elif type(returncode) is int and returncode != 0:
                    classification = "nonzero_without_valid_failure"
                else:
                    classification = "malformed_report"
            else:
                returncode = outcome["returncode"]
                if report["status"] == "worker_failed":
                    classification = ("structured_worker_failure" if returncode == 1
                                      else "exit_status_disagreement")
                elif returncode != 0:
                    classification = ("signaled" if type(returncode) is int
                                      and returncode < 0 else "exit_status_disagreement")
                else:
                    try:
                        verified = bool(verify_handoff(request, report))
                    except BaseException as exc:
                        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                            raise
                        verified = False
                    classification = "boundary_verified" if verified else "handoff_invalid"
        elif classification not in EXIT_CLASSIFICATIONS:
            classification = "cleanup_unverified"
        elapsed = max(0.0, time.monotonic() - request["supervisor_wall_origin"])
        record = _exit_record(request, request_pin, outcome, report,
                              classification, verified, elapsed)
        raw = json_bytes(record, maximum=EXIT_MAX)
        validate_exit_record(raw, request)
        _write_new(dirfd, exit_name(request["phase"]), raw, EXIT_MAX)
        return record
    finally:
        os.close(dirfd)


def supervise_phase(request_bytes, *, expected_request_sha256):
    """Launch exactly one native phase; there is no retry or fallback API."""
    request = validate_request(request_bytes,
        expected_sha256=expected_request_sha256, expected_supervisor_pid=os.getpid(),
        fixture=False, authenticate=True)
    _need(time.monotonic() >= request["supervisor_wall_origin"]
          and time.monotonic() < request["deadline_monotonic"],
          "supervisor request is already expired or from the future")
    binding = request["worker_binding"]
    argv = [PYTHON_EXECUTABLE, WORKER_PATH, "--run-native-phase",
        "--request-path", str(Path(SUPERVISION_ROOT) / request_name(request["phase"])),
        "--request-size", str(len(request_bytes)), "--request-sha256",
        expected_request_sha256, "--supervisor-pid", str(os.getpid())]
    return _supervise(request_bytes, request, namespace=SUPERVISION_ROOT,
        argv=argv, environment=sanitized_environment(request),
        verify_handoff=_verify_native_handoff, fixture=False)


def _supervise_fixture(request_bytes, *, expected_request_sha256, namespace,
                       argv, verify_handoff):
    """Private CPU-test seam; synthetic requests carry no native authority."""
    request = validate_request(request_bytes,
        expected_sha256=expected_request_sha256, expected_supervisor_pid=os.getpid(),
        fixture=True, authenticate=False)
    _need(type(argv) is list and len(argv) >= 2 and argv[0] == PYTHON_EXECUTABLE
          and all(type(item) is str for item in argv)
          and callable(verify_handoff), "fixture launch seam invalid")
    environment = sanitized_environment(request)
    return _supervise(request_bytes, request, namespace=namespace, argv=argv,
        environment=environment, verify_handoff=verify_handoff, fixture=True)


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    _need(type(args) is list and not args, "process supervision has no launch CLI")
    print('{"status":"inert"}')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
