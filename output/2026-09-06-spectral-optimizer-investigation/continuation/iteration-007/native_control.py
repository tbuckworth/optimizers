"""Inert native bootstrap component and read-only attempt inspector.

Import/default CLI perform no I/O and do not import Torch.  There is deliberately
no native launch CLI.  Explicit APIs trust their controller callbacks; they are
not hostile-same-user authentication or execution authority by themselves.
"""
from __future__ import annotations

import fcntl
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import resource
import stat
import subprocess
import sys
import time


class ControlError(RuntimeError):
    pass


class _BootstrapResourceError(ControlError):
    pass


PERMISSION_SCHEMA = "i7_native_launcher_permission_v2"
JOURNAL_SCHEMA = "i7_native_launcher_journal_v2"
FAILURE_SCHEMA_V1 = "i7_native_launcher_failure_v1"
FAILURE_SCHEMA = "i7_native_launcher_failure_v2"
INSPECTION_SCHEMA = "i7_native_attempt_inspection_v1"
SCOPE = "frozen_current_policy_development_pilot"
DEVELOPMENT_ATTEMPT_ID = "i7-native-development-attempt-001"
DEVELOPMENT_ATTEMPT_PATH = "/tmp/spectral-experiment-artifacts/i7-native-development-attempt-001"
EVIDENCE_KINDS = ("synthetic_contract_fixture", "native_producer_attestation")
PHASE_PREFIX = "native-phase-"
ROOT_JOURNAL_NAME = "native-launch-journal.json"
SOURCE_NAME = "native-sources.json"
ENVIRONMENT_NAME = "native-development-environment.json"
EXTERNAL_JOURNAL_NAME = "native-launch-journal.json"
EXTERNAL_FAILURE_NAME = "native-launcher-failure.json"
EXTERNAL_JOURNAL_MAX = 8 << 10
EXTERNAL_FAILURE_MAX = 8 << 10
EXTERNAL_PHASE_FILE_MAX = 8 << 10
EXTERNAL_TOTAL_MAX = (EXTERNAL_JOURNAL_MAX + EXTERNAL_FAILURE_MAX
                      + 3 * 2 * EXTERNAL_PHASE_FILE_MAX)
SHARED_FAILURE_RESERVE = 1 << 20
MIN_FREE_BYTES = 1 << 30
MAX_PHASE_RECORDS = 90
PRE_TRANSITION_RECORDS = 10
TRANSITION_PHASES = ("primary", "sensitivity", "audit")
EXTERNAL_PHASE_NAMES = frozenset(
    f"native-{phase}-{kind}.json"
    for phase in TRANSITION_PHASES for kind in ("permission", "transition"))
BIG_VOLUME = {"target":"/private-artifacts/storage", "source":"/dev/RECONFIGURE_FOR_LOCAL_STORAGE",
              "uuid":"00000000-0000-4000-8000-000000000000"}

PERMISSION_KEYS = ("schema", "attempt_id", "phase", "scope", "expected_commit",
                   "expected_source_set_sha256", "expected_gpu_uuid",
                   "storage_admission_pin", "authorized_utc", "decision",
                   "retry_allowed")
JOURNAL_KEYS = ("schema", "status", "attempt_id", "phase", "scope",
                "permission_sha256", "expected_commit", "expected_source_set_sha256",
                "expected_gpu_uuid", "storage_admission_pin", "authorized_utc",
                "retry_allowed",
                "launcher_entry_wall_origin", "launcher_entry_cpu_origin",
                "bootstrap_entry_wall_observed", "bootstrap_entry_cpu_observed")
PIN_KEYS = ("path", "size_bytes", "sha256")
FAILURE_KEYS_V1 = ("schema", "status", "attempt_id", "stage", "reason",
                   "retry_allowed", "root_binding")
FAILURE_KEYS = FAILURE_KEYS_V1 + ("partial_root", "resource_snapshot")  # gitleaks:allow
PARTIAL_ROOT_KEYS = ("path", "device", "inode", "header_status", "header_sha256")
INSPECTION_KEYS = ("schema", "status", "reason", "phase_records",
                   "completed_operations", "last_event", "pending_operation",
                   "store_terminal", "store_failure_count", "metadata_complete",
                   "actual_artifacts_verified", "root_identity", "header_sha256",
                   "journal_pin", "eligible_for_writable_reopen",
                   "can_resume_incomplete", "execution_authorized",
                   "scientific_execution_certified")
CLOCK_ORIGIN_KEYS = ("wall_origin", "cpu_origin")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)
_COMMIT = re.compile(r"[0-9a-f]{40}\Z", re.ASCII)
_PHASE = re.compile(r"native-phase-([0-9]{3})\.json\Z", re.ASCII)
_UTC = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z\Z", re.ASCII)
_GPU_UUID = re.compile(r"GPU-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\Z",
                       re.ASCII)
DEVELOPMENT_WALL_SECONDS = 180.0
BOOTSTRAP_RSS_BYTES = 12 << 30
BOOTSTRAP_CUDA_ALLOCATED_BYTES = 8 << 30
RESOURCE_KEYS = ("stage", "wall_seconds", "cpu_seconds", "peak_rss_bytes",
                 "peak_cuda_allocated_bytes", "peak_cuda_reserved_bytes")
LIMITATIONS = ("cooperative_checks_do_not_preempt_blocking_calls",
               "caught_interrupt_has_no_external_failure_guarantee",
               "future_phase_environment_collection_not_integrated")
FAILURE_REASONS = {"journal_creation":"journal_creation_failed",
    "native_import":"native_import_failed",
    "runtime_initialization":"runtime_initialization_failed",
    "metadata_collection":"metadata_collection_failed",
    "metadata_validation":"metadata_validation_failed",
    "go_formation":"go_formation_failed",
    "root_creation":"root_creation_failed_identity_may_be_unavailable",
    "phase_zero_write":"phase_zero_write_failed",
    "metadata_copy":"metadata_copy_failed",
    "return_check":"return_resource_check_failed",
    "resource_check":"bootstrap_resource_check_failed",
    "boundary_close":"boundary_close_failed"}


def _need(condition, message):
    if not condition:
        raise ControlError(message)


def _keys(value, expected, label):
    _need(type(value) is dict and tuple(value) == expected
          and all(type(key) is str for key in value), label + ": exact ordered keys")


def _sha(value, label):
    _need(type(value) is str and _SHA256.fullmatch(value) is not None,
          label + ": invalid SHA-256")


def _finite(value, label):
    _need(type(value) is float and math.isfinite(value) and value >= 0.0,
          label + ": invalid clock")


def _exact_nonnegative_int(value, label):
    _need(type(value) is int and type(value) is not bool and value >= 0,
          label + ": invalid nonnegative integer")
    return value


def _native_rss():
    _need(sys.platform.startswith("linux"), "bootstrap RSS requires Linux")
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)


def _torch_is_imported():
    return "torch" in sys.modules


class _BootstrapChecks:
    """One-process cumulative checks local to the permission bootstrap."""

    def __init__(self, origins, entry_process_id, first_wall, first_cpu, *,
                 wall_clock, cpu_clock, rss_probe, cuda_probe, fixture):
        _keys(origins, CLOCK_ORIGIN_KEYS, "bootstrap origins")
        _finite(origins["wall_origin"], "launcher wall origin")
        _finite(origins["cpu_origin"], "launcher CPU origin")
        _finite(first_wall, "bootstrap wall observation")
        _finite(first_cpu, "bootstrap CPU observation")
        _need(origins["wall_origin"] <= first_wall
              and origins["cpu_origin"] <= first_cpu,
              "launcher clock origin follows bootstrap entry")
        _need(type(entry_process_id) is int and type(entry_process_id) is not bool
              and entry_process_id > 0 and entry_process_id == os.getpid(),
              "launcher origins belong to a different process")
        self.origins = (origins["wall_origin"], origins["cpu_origin"])
        self.process_id = entry_process_id
        self.wall, self.cpu = wall_clock, cpu_clock
        self.rss, self.cuda = rss_probe, cuda_probe
        self.fixture = fixture
        self.last = (first_wall, first_cpu)
        self.peaks = [0, 0, 0]
        self.snapshots = []

    def check(self, stage, torch_module=None, *, enforce=True):
        _need(type(stage) is str and stage.isascii() and 0 < len(stage) <= 48,
              "bootstrap resource stage invalid")
        try:
            _need(os.getpid() == self.process_id,
                  "bootstrap process identity changed")
            wall_now, cpu_now = self.wall(), self.cpu()
            _finite(wall_now, "bootstrap wall observation")
            _finite(cpu_now, "bootstrap CPU observation")
            _need(wall_now >= self.last[0] and cpu_now >= self.last[1],
                  "bootstrap clock regressed")
            rss_bytes = self.rss()
            _exact_nonnegative_int(rss_bytes, "bootstrap RSS")
            if self.fixture:
                cuda = self.cuda()
            elif torch_module is None or not torch_module.cuda.is_initialized():
                cuda = (0, 0)
            else:
                cuda = (int(torch_module.cuda.max_memory_allocated(0)),
                        int(torch_module.cuda.max_memory_reserved(0)))
            _need(type(cuda) is tuple and len(cuda) == 2,
                  "bootstrap CUDA probe shape differs")
            allocated = _exact_nonnegative_int(cuda[0], "bootstrap CUDA allocation")
            reserved = _exact_nonnegative_int(cuda[1], "bootstrap CUDA reservation")
            _need(allocated <= reserved, "bootstrap CUDA allocation exceeds reservation")
        except (KeyboardInterrupt, SystemExit):
            raise
        except ControlError:
            raise _BootstrapResourceError(
                "bootstrap resource check failed at " + stage) from None
        except Exception:
            raise _BootstrapResourceError(
                "bootstrap resource check failed at " + stage) from None
        self.last = (wall_now, cpu_now)
        self.peaks[0] = max(self.peaks[0], rss_bytes)
        self.peaks[1] = max(self.peaks[1], allocated)
        self.peaks[2] = max(self.peaks[2], reserved)
        row = {"stage":stage,
            "wall_seconds":wall_now - self.origins[0],
            "cpu_seconds":cpu_now - self.origins[1],
            "peak_rss_bytes":self.peaks[0],
            "peak_cuda_allocated_bytes":self.peaks[1],
            "peak_cuda_reserved_bytes":self.peaks[2]}
        _keys(row, RESOURCE_KEYS, "bootstrap resource snapshot")
        self.snapshots.append(row)
        if enforce:
            if row["wall_seconds"] > DEVELOPMENT_WALL_SECONDS:
                raise _BootstrapResourceError("bootstrap development wall cap exceeded")
            if row["peak_rss_bytes"] > BOOTSTRAP_RSS_BYTES:
                raise _BootstrapResourceError("bootstrap RSS cap exceeded")
            if row["peak_cuda_allocated_bytes"] > BOOTSTRAP_CUDA_ALLOCATED_BYTES:
                raise _BootstrapResourceError("bootstrap CUDA allocation cap exceeded")
        return dict(row)


def _utc(value, label):
    _need(type(value) is str and _UTC.fullmatch(value) is not None, label + ": invalid UTC")
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as exc:
        raise ControlError(label + ": invalid UTC") from exc
    _need(parsed.strftime("%Y-%m-%dT%H:%M:%SZ") == value, label + ": noncanonical UTC")


def _json_tree(value, active=None):
    active = set() if active is None else active
    if value is None or type(value) in (bool, int, str):
        if type(value) is str:
            _need(not any(0xD800 <= ord(char) <= 0xDFFF for char in value),
                  "JSON string contains surrogate")
        return
    if type(value) is float:
        _need(math.isfinite(value), "nonfinite JSON float")
        return
    _need(type(value) in (list, dict), "unsupported JSON value")
    _need(id(value) not in active, "cyclic JSON value")
    active.add(id(value))
    if type(value) is dict:
        for key, item in value.items():
            _need(type(key) is str, "JSON key must be a string")
            _json_tree(key, active)
            _json_tree(item, active)
    else:
        for item in value:
            _json_tree(item, active)
    active.remove(id(value))


def _json_bytes(value):
    _json_tree(value)
    return (json.dumps(value, ensure_ascii=True, allow_nan=False, sort_keys=False,
                       separators=(",", ":")) + "\n").encode("ascii")


def _json_loads(raw, maximum):
    _need(type(raw) is bytes and 0 < len(raw) <= maximum, "bounded JSON bytes required")
    try:
        text = raw.decode("ascii")
    except UnicodeDecodeError as exc:
        raise ControlError("JSON must be ASCII") from exc
    def pairs(rows):
        result = {}
        for key, value in rows:
            _need(key not in result, "duplicate JSON key")
            result[key] = value
        return result
    def constant(_):
        raise ControlError("nonfinite JSON constant")
    try:
        value = json.loads(text, object_pairs_hook=pairs, parse_constant=constant)
    except (json.JSONDecodeError, RecursionError, ValueError) as exc:
        if isinstance(exc, ControlError):
            raise
        raise ControlError("malformed JSON") from exc
    _need(_json_bytes(value) == raw, "JSON bytes are not canonical")
    return value


def phase_name(sequence):
    _need(type(sequence) is int and 0 <= sequence < MAX_PHASE_RECORDS,
          "invalid phase sequence")
    return f"{PHASE_PREFIX}{sequence:03d}.json"


def cumulative_clocks(origins, *, wall_clock=time.monotonic, cpu_clock=time.process_time):
    """Return cumulative controller clocks from the pre-launcher origins."""
    wall_now = wall_clock()
    cpu_now = cpu_clock()
    _keys(origins, CLOCK_ORIGIN_KEYS, "clock origins")
    _finite(origins["wall_origin"], "wall origin")
    _finite(origins["cpu_origin"], "CPU origin")
    _finite(wall_now, "wall observation")
    _finite(cpu_now, "CPU observation")
    _need(wall_now >= origins["wall_origin"] and cpu_now >= origins["cpu_origin"],
          "cumulative clock moved backwards")
    return {"wall_seconds":wall_now - origins["wall_origin"],
            "cpu_seconds":cpu_now - origins["cpu_origin"]}


def _modules():
    import artifact_store as storage
    import cuda_identity
    import identity_codec as codec
    import native_phase_policy as policy
    import runtime_guard as runtime
    import source_environment_schema as environment
    return storage, cuda_identity, codec, policy, runtime, environment


def _permission(raw, pins, *, fixture=False):
    _need(type(raw) is bytes and 0 < len(raw) <= EXTERNAL_JOURNAL_MAX,
          "permission must be bounded bytes")
    value = _json_loads(raw, EXTERNAL_JOURNAL_MAX)
    _keys(value, PERMISSION_KEYS, "permission")
    _need(value["schema"] == PERMISSION_SCHEMA and value["phase"] == "development"
          and value["scope"] == SCOPE and value["decision"] == "go"
          and value["retry_allowed"] is False, "permission scope/decision differs")
    _need(type(value["attempt_id"]) is str
          and value["attempt_id"] == DEVELOPMENT_ATTEMPT_ID,
          "invalid attempt identity")
    _need(type(value["expected_commit"]) is str
          and _COMMIT.fullmatch(value["expected_commit"]), "invalid expected commit")
    _sha(value["expected_source_set_sha256"], "expected source set")
    _need(type(value["expected_gpu_uuid"]) is str
          and _GPU_UUID.fullmatch(value["expected_gpu_uuid"]),
          "invalid expected GPU UUID")
    import native_storage_authority as authority
    try:
        authority.validate_storage_admission_pin(
            value["storage_admission_pin"], fixture=fixture)
    except authority.StorageAuthorityError as exc:
        raise ControlError("invalid storage admission pin") from exc
    _utc(value["authorized_utc"], "authorization")
    expected = (pins["attempt_id"], pins["expected_commit"],
                pins["expected_source_set_sha256"], pins["expected_gpu_uuid"])
    actual = (value["attempt_id"], value["expected_commit"],
              value["expected_source_set_sha256"], value["expected_gpu_uuid"])
    _need(actual == expected, "permission differs from exact launcher pins")
    _sha(pins["permission_sha256"], "expected permission")
    _need(hashlib.sha256(raw).hexdigest() == pins["permission_sha256"],
          "permission byte hash differs")
    return value


def _validate_journal(value, *, fixture=False):
    _keys(value, JOURNAL_KEYS, "journal")
    _need(value["schema"] == JOURNAL_SCHEMA and value["status"] == "consumed"
          and value["phase"] == "development" and value["scope"] == SCOPE
          and value["retry_allowed"] is False, "journal status/scope differs")
    _need(type(value["attempt_id"]) is str
          and value["attempt_id"] == DEVELOPMENT_ATTEMPT_ID,
          "journal attempt invalid")
    _sha(value["permission_sha256"], "journal permission")
    _need(type(value["expected_commit"]) is str
          and _COMMIT.fullmatch(value["expected_commit"]), "journal commit invalid")
    _sha(value["expected_source_set_sha256"], "journal source set")
    _need(type(value["expected_gpu_uuid"]) is str
          and _GPU_UUID.fullmatch(value["expected_gpu_uuid"]),
          "journal GPU UUID invalid")
    import native_storage_authority as authority
    try:
        authority.validate_storage_admission_pin(
            value["storage_admission_pin"], fixture=fixture)
    except authority.StorageAuthorityError as exc:
        raise ControlError("journal storage admission pin invalid") from exc
    _utc(value["authorized_utc"], "journal authorization")
    for key in JOURNAL_KEYS[-4:]:
        _finite(value[key], key)
    _need(value["launcher_entry_wall_origin"] <= value["bootstrap_entry_wall_observed"]
          and value["launcher_entry_cpu_origin"] <= value["bootstrap_entry_cpu_observed"],
          "journal clock order differs")
    return value


def _validate_pins(pins):
    expected = ("permission_sha256", "attempt_id", "expected_commit",
                "expected_source_set_sha256", "expected_gpu_uuid",
                "attempt_directory")
    _keys(pins, expected, "launcher pins")
    for key in expected:
        _need(type(pins[key]) is str, "launcher pin must be a string")
    _need(pins["attempt_id"] == DEVELOPMENT_ATTEMPT_ID, "invalid pinned attempt")
    _need(_COMMIT.fullmatch(pins["expected_commit"]) is not None, "invalid pinned commit")
    _sha(pins["expected_source_set_sha256"], "pinned source set")
    return pins


def _verify_big_parent(path):
    absolute = Path(os.path.abspath(os.fspath(path)))
    _need(absolute.is_absolute(), "external parent must be absolute")
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        info = os.lstat(current)
        _need(not stat.S_ISLNK(info.st_mode), "external parent path contains symlink")
    info = os.stat(absolute)
    _need(stat.S_ISDIR(info.st_mode), "external parent is not a directory")
    try:
        absolute.relative_to(BIG_VOLUME["target"])
    except ValueError as exc:
        raise ControlError("external parent is outside verified big volume") from exc
    try:
        result = subprocess.run(["findmnt", "--noheadings", "--target", str(absolute),
            "--output", "TARGET,SOURCE,UUID,MAJ:MIN"], capture_output=True, text=True,
            check=True, timeout=5)
    except (OSError, subprocess.SubprocessError) as exc:
        raise ControlError("external parent mount probe failed") from exc
    lines = [line.split() for line in result.stdout.splitlines() if line.strip()]
    _need(len(lines) == 1 and len(lines[0]) == 4, "external parent mount row invalid")
    target, source, uuid, device = lines[0]
    _need((target, source, uuid) == (BIG_VOLUME["target"], BIG_VOLUME["source"],
                                    BIG_VOLUME["uuid"]),
          "external parent mount identity differs")
    try:
        major, minor = (int(value) for value in device.split(":", 1))
    except (TypeError, ValueError) as exc:
        raise ControlError("external parent mount device invalid") from exc
    _need((os.major(info.st_dev), os.minor(info.st_dev)) == (major, minor),
          "external parent stat/mount device differs")
    return absolute


def _write_all(fd, raw):
    view = memoryview(raw)
    while view:
        count = os.write(fd, view)
        if count <= 0:
            raise OSError("short write")
        view = view[count:]


def _write_new(dirfd, name, raw, maximum):
    _need(type(raw) is bytes and 0 < len(raw) <= maximum, "external record byte cap")
    fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                 0o600, dir_fd=dirfd)
    try:
        _write_all(fd, raw)
        os.fsync(fd)
        info = os.fstat(fd)
        _need(stat.S_ISREG(info.st_mode) and info.st_nlink == 1
              and info.st_size == len(raw), "external record identity changed")
    finally:
        os.close(fd)
    os.fsync(dirfd)
    return {"path":name, "size_bytes":len(raw),
            "sha256":hashlib.sha256(raw).hexdigest()}


def _partial_root(value):
    _keys(value, PARTIAL_ROOT_KEYS, "partial root")
    _need(type(value["path"]) is str and Path(value["path"]).is_absolute()
          and os.path.normpath(value["path"]) == value["path"],
          "partial root path invalid")
    _exact_nonnegative_int(value["device"], "partial root device")
    _exact_nonnegative_int(value["inode"], "partial root inode")
    _need(value["header_status"] in
          ("absent", "unobserved", "present_unverified", "verified_complete"),
          "partial root header status invalid")
    if value["header_status"] == "verified_complete":
        _sha(value["header_sha256"], "partial root header")
    else:
        _need(value["header_sha256"] is None,
              "unverified partial root cannot claim a header hash")
    return value


def _full_root(value):
    expected = ("path", "device", "inode", "header_sha256")
    _keys(value, expected, "root binding")
    _need(type(value["path"]) is str and Path(value["path"]).is_absolute()
          and os.path.normpath(value["path"]) == value["path"],
          "root binding path invalid")
    _exact_nonnegative_int(value["device"], "root binding device")
    _exact_nonnegative_int(value["inode"], "root binding inode")
    _sha(value["header_sha256"], "root binding header")
    return value


def _resource_snapshot(value):
    _keys(value, RESOURCE_KEYS, "failure resource snapshot")
    _need(type(value["stage"]) is str and value["stage"].isascii()
          and 0 < len(value["stage"]) <= 48, "failure resource stage invalid")
    for key in ("wall_seconds", "cpu_seconds"):
        _finite(value[key], "failure " + key)
    for key in ("peak_rss_bytes", "peak_cuda_allocated_bytes",
                "peak_cuda_reserved_bytes"):
        _exact_nonnegative_int(value[key], "failure " + key)
    _need(value["peak_cuda_allocated_bytes"] <= value["peak_cuda_reserved_bytes"],
          "failure CUDA allocation exceeds reservation")
    return value


def _failure_record(attempt_id, stage, reason, root_binding, partial_root,
                    resource_snapshot=None):
    _need(not (root_binding is not None and partial_root is not None),
          "launcher failure has conflicting root evidence")
    if root_binding is not None:
        _full_root(root_binding)
    if partial_root is not None:
        _partial_root(partial_root)
    if resource_snapshot is not None:
        _resource_snapshot(resource_snapshot)
    return {"schema":FAILURE_SCHEMA, "status":"failed", "attempt_id":attempt_id,
            "stage":stage, "reason":reason, "retry_allowed":False,
            "root_binding":root_binding, "partial_root":partial_root,
            "resource_snapshot":resource_snapshot}


def _external_failure(dirfd, attempt_id, stage, reason, root_binding, partial_root=None,
                      resource_snapshot=None):
    record = _failure_record(attempt_id, stage, reason, root_binding, partial_root,
                             resource_snapshot)
    raw = _json_bytes(record)
    try:
        _write_new(dirfd, EXTERNAL_FAILURE_NAME, raw, EXTERNAL_FAILURE_MAX)
    except (OSError, ControlError):
        pass


def _root_binding(store):
    device, inode = store.root_identity
    return {"path":str(store.root), "device":device, "inode":inode,
            "header_sha256":store.header_sha256}


def _validated_failure(value, attempt_id):
    if type(value) is dict and tuple(value) == FAILURE_KEYS_V1:
        _need(value["schema"] == FAILURE_SCHEMA_V1, "launcher failure schema differs")
        value = dict(value, partial_root=None, resource_snapshot=None)
    else:
        _keys(value, FAILURE_KEYS, "launcher failure")
        _need(value["schema"] == FAILURE_SCHEMA, "launcher failure schema differs")
    _need(value["status"] == "failed" and value["attempt_id"] == attempt_id
          and value["retry_allowed"] is False
          and value["stage"] in FAILURE_REASONS
          and value["reason"] == FAILURE_REASONS[value["stage"]],
          "launcher failure record differs")
    _need(not (value["root_binding"] is not None
               and value["partial_root"] is not None),
          "launcher failure root evidence conflicts")
    if value["root_binding"] is not None:
        _full_root(value["root_binding"])
    if value["partial_root"] is not None:
        _partial_root(value["partial_root"])
    if value["resource_snapshot"] is not None:
        _resource_snapshot(value["resource_snapshot"])
    return value


def _prepare_native_runtime(runtime, permission, phase, initialize_runtime, *,
                            entry_wall_origin, entry_cpu_origin, entry_process_id):
    """Trusted setup after durable consumption, never a substitute for permission.

    Check same-process clocks/RSS before the only CUDA-initializing callback and
    immediately validate the initialized device. No data or root is created here.
    The enclosing bootstrap/transition retains failure evidence on an exception.
    """
    _need(callable(initialize_runtime), "explicit native initializer required")
    _need(type(entry_process_id) is int and entry_process_id == os.getpid(),
          "runtime origins belong to a different process")
    _finite(entry_wall_origin, "runtime wall origin")
    _finite(entry_cpu_origin, "runtime CPU origin")
    wall, cpu = time.monotonic(), time.process_time()
    _need(entry_wall_origin <= wall and entry_cpu_origin <= cpu,
          "runtime origin follows current observation")
    _need(wall - entry_wall_origin <= runtime.PHASE_CAPS[phase]["wall_seconds"]
          and _native_rss() <= runtime.RSS_CAP, "pre-initialization resource cap")
    _need(not runtime.torch.cuda.is_initialized(),
          "native runtime was initialized before consumed setup")
    permission_snapshot = _json_bytes(permission)
    setup_permission = _json_loads(permission_snapshot, EXTERNAL_PHASE_FILE_MAX)
    _need(initialize_runtime(setup_permission) is None,
          "native initializer must return no payload or authority")
    _need(_json_bytes(permission) == permission_snapshot,
          "runtime initializer changed authoritative permission")
    identity = runtime._native_cuda_identity()
    _need(identity["stable_identity"] == permission["expected_gpu_uuid"],
          "initialized GPU differs from permission")
    guard = runtime.RuntimeGuard(
        phase, entry_wall_origin=entry_wall_origin, entry_cpu_origin=entry_cpu_origin,
        entry_process_id=entry_process_id)
    guard.check("runtime.initialized.after.consumption")
    return guard


def bootstrap_development(permission_bytes, *, pins, entry_wall_origin,
                          entry_cpu_origin, entry_process_id, collect_metadata,
                          create_store, wall_clock=None, cpu_clock=None,
                          rss_probe=None, cuda_probe=None, utc_now=None,
                          evidence_kind="native_producer_attestation",
                          initialize_runtime=None):
    """Consume one permission and create phase zero; no retry or native CLI.

    Supplied entry origins and PID cover same-process launcher work before this
    API.  The two clocks are observed before permission parsing, I/O or imports.
    Callbacks are trusted controller components and must not perform work before
    their applicable GO.
    """
    _need(type(evidence_kind) is str and evidence_kind in EVIDENCE_KINDS,
          "invalid evidence kind")
    fixture = evidence_kind == "synthetic_contract_fixture"
    if not fixture:
        _need(all(item is None for item in
                  (wall_clock, cpu_clock, rss_probe, cuda_probe)),
              "native bootstrap forbids resource probe overrides")
        _need(not _torch_is_imported(),
              "native bootstrap requires Torch absent before permission")
        _need(callable(initialize_runtime), "explicit native initializer required")
    else:
        _need(initialize_runtime is None, "fixture bootstrap forbids native initialization")
    wall_fn = time.monotonic if wall_clock is None else wall_clock
    cpu_fn = time.process_time if cpu_clock is None else cpu_clock
    bootstrap_wall = wall_fn()
    bootstrap_cpu = cpu_fn()
    rss_fn = _native_rss if rss_probe is None else rss_probe
    cuda_fn = (lambda:(0, 0)) if cuda_probe is None else cuda_probe
    _need(all(callable(item) for item in (wall_fn, cpu_fn, rss_fn, cuda_fn)),
          "bootstrap resource probes must be callable")
    origins = {"wall_origin":entry_wall_origin, "cpu_origin":entry_cpu_origin}
    checks = _BootstrapChecks(
        origins, entry_process_id, bootstrap_wall, bootstrap_cpu,
        wall_clock=wall_fn, cpu_clock=cpu_fn, rss_probe=rss_fn,
        cuda_probe=cuda_fn, fixture=fixture)
    _need(callable(collect_metadata) and callable(create_store) and callable(utc_now),
          "trusted bootstrap callbacks required")
    _need(EXTERNAL_TOTAL_MAX == 64 << 10
          < SHARED_FAILURE_RESERVE, "external evidence bounds exceed shared reserve")
    pins = _validate_pins(pins)
    permission = _permission(permission_bytes, pins, fixture=fixture)
    checks.check("permission", enforce=False)

    attempt = Path(pins["attempt_directory"])
    _need(str(attempt) == pins["attempt_directory"] and attempt.is_absolute()
          and os.path.normpath(str(attempt)) == str(attempt)
          and attempt.name == pins["attempt_id"], "attempt directory/pin differs")
    if evidence_kind == "native_producer_attestation":
        _need(str(attempt) == DEVELOPMENT_ATTEMPT_PATH,
              "native attempt path differs from fixed prospective singleton")
    parent = attempt.parent
    _verify_big_parent(parent)
    _need(os.statvfs(parent).f_bavail * os.statvfs(parent).f_frsize >= MIN_FREE_BYTES,
          "external journal free-space floor")
    checks.check("volume", enforce=False)
    parentfd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    dirfd = None
    store = None
    consumed = False
    stage = "journal_creation"
    storage = runtime = None
    try:
        try:
            os.mkdir(attempt.name, 0o700, dir_fd=parentfd)
        except FileExistsError as exc:
            raise ControlError("attempt evidence already exists; retry refused") from exc
        consumed = True
        os.fsync(parentfd)
        dirfd = os.open(attempt.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                        dir_fd=parentfd)
        journal = {"schema":JOURNAL_SCHEMA, "status":"consumed",
            "attempt_id":permission["attempt_id"], "phase":"development",
            "scope":SCOPE, "permission_sha256":pins["permission_sha256"],
            "expected_commit":pins["expected_commit"],
            "expected_source_set_sha256":pins["expected_source_set_sha256"],
            "expected_gpu_uuid":pins["expected_gpu_uuid"],
            "storage_admission_pin":dict(permission["storage_admission_pin"]),
            "authorized_utc":permission["authorized_utc"], "retry_allowed":False,
            "launcher_entry_wall_origin":entry_wall_origin,
            "launcher_entry_cpu_origin":entry_cpu_origin,
            "bootstrap_entry_wall_observed":bootstrap_wall,
            "bootstrap_entry_cpu_observed":bootstrap_cpu}
        journal_raw = _json_bytes(journal)
        pin = _write_new(dirfd, EXTERNAL_JOURNAL_NAME, journal_raw, EXTERNAL_JOURNAL_MAX)
        pin["path"] = str(attempt / EXTERNAL_JOURNAL_NAME)
        checks.check("journal")

        stage = "native_import"
        storage, _, codec, policy, runtime, environment = _modules()
        _need(type(evidence_kind) is str and evidence_kind in policy.KINDS,
              "invalid evidence kind")
        checks.check("native_import", runtime.torch)
        if not fixture:
            stage = "runtime_initialization"
            _prepare_native_runtime(
                runtime, permission, "development", initialize_runtime,
                entry_wall_origin=entry_wall_origin, entry_cpu_origin=entry_cpu_origin,
                entry_process_id=entry_process_id)
            checks.check("runtime_initialization", runtime.torch)
        stage = "metadata_collection"
        sources, native_environment = collect_metadata(permission)
        stage = "metadata_validation"
        environment.validate_sources(sources, profile=environment.SCIENTIFIC)
        environment.validate_environment(native_environment, profile=environment.SCIENTIFIC)
        _need(sources["repository_revision"] == pins["expected_commit"]
              and codec.tree_digest(sources) == pins["expected_source_set_sha256"],
              "collected source binding differs from permission")
        _need(sources["repository_root_realpath"]
              == native_environment["repository_root_realpath"]
              and native_environment["runtime_role"] == "native_source"
              and native_environment["cuda"]["devices"][0]["uuid"]
              == pins["expected_gpu_uuid"], "collected native environment differs")
        source_raw = codec.json_bytes(sources)
        environment_raw = codec.json_bytes(native_environment)
        checks.check("metadata", runtime.torch)

        stage = "go_formation"
        created_utc = utc_now()
        go = {"schema":"i7_native_phase_record_v1", "profile":policy.PROFILE,
            "evidence_kind":evidence_kind, "execution_enabled":False, "sequence":0,
            "event":"decision", "operation":"development_go", "phase":"development",
            "created_utc":created_utc, "root_binding":None,
            "sources_sha256":codec.tree_digest(sources),
            "environment_sha256":codec.tree_digest(native_environment),
            "previous_record_sha256":None, "artifacts":[], "resources":None,
            "error":None}
        policy.validate_record(go, evidence_kind=evidence_kind)
        go_raw = codec.json_bytes(go)
        checks.check("go_formation", runtime.torch)

        stage = "root_creation"
        store = create_store()
        required_profile = (storage.SCIENTIFIC if evidence_kind == "native_producer_attestation"
                            else storage.MLP_FIXTURE)
        _need(store.profile == required_profile, "evidence kind/store profile mismatch")
        binding = _root_binding(store)
        checks.check("root", runtime.torch)
        stage = "phase_zero_write"
        store.write_bytes(phase_name(0), go_raw)
        checks.check("phase_zero_copy", runtime.torch)
        stage = "metadata_copy"
        store.write_bytes(ROOT_JOURNAL_NAME, journal_raw)
        checks.check("journal_copy", runtime.torch)
        store.write_bytes(SOURCE_NAME, source_raw)
        checks.check("source_copy", runtime.torch)
        store.write_bytes(ENVIRONMENT_NAME, environment_raw)
        checks.check("environment_copy", runtime.torch)
        stage = "return_check"
        checks.check("return", runtime.torch)
        return {"store":store, "root_binding":binding, "journal_pin":pin,
                "go_record":go, "go_bytes":go_raw,
                "clock_origins":{"wall_origin":entry_wall_origin,
                                 "cpu_origin":entry_cpu_origin},
                "bootstrap_resources":[dict(row) for row in checks.snapshots],
                "clock_scope":"same-process cumulative cooperative bootstrap checks",
                "limitations":LIMITATIONS}
    except BaseException as exc:
        interrupted = isinstance(exc, (KeyboardInterrupt, SystemExit))
        binding = None
        partial_root = None
        if store is not None:
            try:
                binding = _root_binding(store)
            except BaseException:
                binding = None
            try:
                store.close()
            except BaseException:
                pass
        elif (storage is not None
              and isinstance(exc, storage.StoreInitializationError)):
            partial_root = exc.partial_root
        if interrupted:
            raise
        if consumed and dirfd is not None:
            failure_stage = ("resource_check"
                             if isinstance(exc, _BootstrapResourceError) else stage)
            reason = FAILURE_REASONS.get(failure_stage, "journal_creation_failed")
            _external_failure(dirfd, pins.get("attempt_id", "unavailable"), failure_stage,
                              reason, binding, partial_root,
                              checks.snapshots[-1] if checks.snapshots else None)
        if isinstance(exc, ControlError):
            raise
        raise ControlError("bootstrap failed at closed stage: " + stage) from None
    finally:
        if dirfd is not None:
            os.close(dirfd)
        os.close(parentfd)


def _read_pinned(path, size, digest, maximum):
    _need(type(path) is str and Path(path).is_absolute()
          and os.path.normpath(path) == path and path != "/",
          "pinned path must be canonical absolute")
    if size is not None:
        _need(type(size) is int and 0 < size <= maximum, "pinned size invalid")
    if digest is not None:
        _sha(digest, "pinned file")
    parts = path.split("/")[1:]
    dirfd = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    fd = None
    try:
        for part in parts[:-1]:
            _need(part not in ("", ".", ".."), "pinned path component invalid")
            nextfd = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                             dir_fd=dirfd)
            os.close(dirfd)
            dirfd = nextfd
        _need(parts[-1] not in ("", ".", ".."), "pinned filename invalid")
        fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                     dir_fd=dirfd)
        before = os.fstat(fd)
        _need(stat.S_ISREG(before.st_mode) and before.st_nlink == 1
              and 0 < before.st_size <= maximum
              and (size is None or before.st_size == size),
              "pinned file identity differs")
        raw = b""
        while len(raw) <= before.st_size:
            part = os.read(fd, before.st_size + 1 - len(raw))
            if not part:
                break
            raw += part
        after = os.fstat(fd)
        path_after = os.stat(parts[-1], dir_fd=dirfd, follow_symlinks=False)
        _need((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns,
               before.st_ctime_ns) == (after.st_dev, after.st_ino, after.st_size,
                                       after.st_mtime_ns, after.st_ctime_ns),
              "pinned file changed during read")
        _need((after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns,
               after.st_ctime_ns, after.st_mode, after.st_nlink)
              == (path_after.st_dev, path_after.st_ino, path_after.st_size,
                  path_after.st_mtime_ns, path_after.st_ctime_ns,
                  path_after.st_mode, path_after.st_nlink),
              "pinned file path changed during read")
    finally:
        if fd is not None:
            os.close(fd)
        os.close(dirfd)
    _need(len(raw) == before.st_size
          and (digest is None or hashlib.sha256(raw).hexdigest() == digest),
          "pinned file bytes differ")
    return raw


def record_boundary_failure(journal_pin, root_binding, *, expected_evidence_kind):
    """Retain one terminal post-close failure without writing or reopening the root."""
    _need(type(expected_evidence_kind) is str
          and expected_evidence_kind in EVIDENCE_KINDS,
          "invalid evidence kind")
    _keys(journal_pin, PIN_KEYS, "journal pin")
    _full_root(root_binding)
    journal_path = journal_pin["path"]
    if expected_evidence_kind == "native_producer_attestation":
        _need(journal_path == DEVELOPMENT_ATTEMPT_PATH + "/" + EXTERNAL_JOURNAL_NAME,
              "native journal path differs from fixed prospective singleton")
    external_dir = Path(journal_path).parent
    _verify_big_parent(external_dir)
    journal_raw = _read_pinned(journal_path, journal_pin["size_bytes"],
                               journal_pin["sha256"], EXTERNAL_JOURNAL_MAX)
    journal = _json_loads(journal_raw, EXTERNAL_JOURNAL_MAX)
    fixture = expected_evidence_kind == "synthetic_contract_fixture"
    _validate_journal(journal, fixture=fixture)
    _need(Path(journal_path).name == EXTERNAL_JOURNAL_NAME
          and external_dir.name == journal["attempt_id"],
          "journal path/attempt differs")
    names = {entry.name for entry in os.scandir(external_dir)}
    _need(EXTERNAL_FAILURE_NAME not in names
          and EXTERNAL_JOURNAL_NAME in names
          and names <= ({EXTERNAL_JOURNAL_NAME} | EXTERNAL_PHASE_NAMES),
          "boundary failure target already contains invalid attempt evidence")
    import phase_transition as transition
    transition.validate_external_for_failure(
        external_dir, names, journal_pin, root_binding, fixture=fixture)
    dirfd = os.open(external_dir, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        record = _failure_record(journal["attempt_id"], "boundary_close",
                                 FAILURE_REASONS["boundary_close"], root_binding, None)
        try:
            pin = _write_new(dirfd, EXTERNAL_FAILURE_NAME, _json_bytes(record),
                             EXTERNAL_FAILURE_MAX)
        except FileExistsError as exc:
            raise ControlError("boundary failure already retained; overwrite refused") from exc
        except OSError as exc:
            raise ControlError("boundary failure retention failed") from exc
        pin["path"] = str(external_dir / EXTERNAL_FAILURE_NAME)
        return pin
    finally:
        os.close(dirfd)


def _inspect_partial_root(root, evidence):
    _partial_root(evidence)
    root_path = os.fspath(root)
    _need(type(root_path) is str and root_path == evidence["path"],
          "partial root path differs")
    _verify_big_parent(root_path)
    path_info = os.lstat(root_path)
    _need(stat.S_ISDIR(path_info.st_mode) and not stat.S_ISLNK(path_info.st_mode),
          "partial root path invalid")
    dirfd = os.open(root_path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    lockfd = None
    try:
        opened = os.fstat(dirfd)
        _need((opened.st_dev, opened.st_ino)
              == (evidence["device"], evidence["inode"]),
              "partial root identity differs")
        with os.scandir(dirfd) as entries:
            scanned = {entry.name:entry.stat(follow_symlinks=False) for entry in entries}
        _need(set(scanned) <= {"store.lock", "store-header.json"},
              "partial root membership differs")
        for info in scanned.values():
            _need(stat.S_ISREG(info.st_mode) and info.st_nlink == 1,
                  "partial root entry invalid")
        if "store.lock" in scanned:
            _need(scanned["store.lock"].st_size == 0, "partial root lock invalid")
            lockfd = os.open("store.lock", os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                             dir_fd=dirfd)
            fcntl.flock(lockfd, fcntl.LOCK_SH | fcntl.LOCK_NB)
        header_info = scanned.get("store-header.json")
        status = evidence["header_status"]
        if status == "absent":
            _need(header_info is None, "partial root header presence differs")
        elif status == "present_unverified":
            _need(header_info is not None and header_info.st_size <= 4096,
                  "partial root header presence differs")
        elif status == "verified_complete":
            _need(header_info is not None and header_info.st_size <= 4096,
                  "partial root header missing")
            raw = _read_pinned(str(Path(root_path) / "store-header.json"),
                               header_info.st_size,
                               evidence["header_sha256"], 4096)
            _need(bool(raw), "partial root header empty")
        final_path = os.lstat(root_path)
        _need((final_path.st_dev, final_path.st_ino)
              == (opened.st_dev, opened.st_ino),
              "partial root changed during inspection")
        return [opened.st_dev, opened.st_ino], evidence["header_sha256"]
    finally:
        if lockfd is not None:
            try:
                fcntl.flock(lockfd, fcntl.LOCK_UN)
            finally:
                os.close(lockfd)
        os.close(dirfd)


def _report(status, reason, *, phase_records=0, validation=None, store=None,
            metadata_complete=False, artifacts=False, root_identity=None,
            header_sha256=None, journal_pin=None):
    boundary = (status == "sealed_boundary" and store is not None
                and store["terminal"] is False)
    return {"schema":INSPECTION_SCHEMA, "status":status, "reason":reason,
        "phase_records":phase_records,
        "completed_operations":None if validation is None else validation["completed_operations"],
        "last_event":None if validation is None else validation.get("last_event"),
        "pending_operation":None if validation is None else validation["pending_operation"],
        "store_terminal":None if store is None else store["terminal"],
        "store_failure_count":None if store is None else len(store["failures"]),
        "metadata_complete":metadata_complete,
        "actual_artifacts_verified":artifacts, "root_identity":root_identity,
        "header_sha256":header_sha256, "journal_pin":journal_pin,
        "eligible_for_writable_reopen":boundary, "can_resume_incomplete":False,
        "execution_authorized":False, "scientific_execution_certified":False}


def _signature(info):
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns,
            info.st_ctime_ns, info.st_mode, info.st_nlink)


def _positive_snapshot_stable(root_path, dirfd, lockfd, initial_root, scanned, storage):
    opened = os.fstat(dirfd)
    path_now = os.lstat(root_path)
    _need(_signature(opened) == _signature(initial_root) == _signature(path_now),
          "root identity changed during inspection")
    _need(_signature(os.fstat(lockfd)) == _signature(scanned["store.lock"]),
          "store lock changed during inspection")
    final, _, _ = storage._scan_regular(dirfd)
    _need({name:_signature(info) for name, info in final.items()}
          == {name:_signature(info) for name, info in scanned.items()},
          "inventory changed during inspection")


def inspect_native_attempt(root, journal_pin, *, expected_store_profile,
                           expected_evidence_kind):
    """Inspect one quiescent attempt through a separate nonblocking shared lock."""
    storage, _, codec, policy, _, environment = _modules()
    if (type(expected_evidence_kind) is not str
            or expected_evidence_kind not in EVIDENCE_KINDS):
        return _report("uninspectable", "expected_evidence_kind_invalid")
    required_profile = (storage.SCIENTIFIC
                        if expected_evidence_kind == "native_producer_attestation"
                        else storage.MLP_FIXTURE)
    if expected_store_profile != required_profile:
        return _report("uninspectable", "evidence_kind_store_profile_mismatch")
    dirfd = None
    try:
        _keys(journal_pin, PIN_KEYS, "journal pin")
        if expected_evidence_kind == "native_producer_attestation":
            _need(journal_pin["path"]
                  == DEVELOPMENT_ATTEMPT_PATH + "/" + EXTERNAL_JOURNAL_NAME,
                  "native journal path differs from fixed prospective singleton")
        _verify_big_parent(Path(journal_pin["path"]).parent)
        journal_raw = _read_pinned(journal_pin["path"], journal_pin["size_bytes"],
                                   journal_pin["sha256"], EXTERNAL_JOURNAL_MAX)
        journal = _json_loads(journal_raw, EXTERNAL_JOURNAL_MAX)
        fixture = expected_evidence_kind == "synthetic_contract_fixture"
        _validate_journal(journal, fixture=fixture)
        _need(Path(journal_pin["path"]).name == EXTERNAL_JOURNAL_NAME
              and Path(journal_pin["path"]).parent.name == journal["attempt_id"],
              "journal path/attempt differs")
        external_dir = Path(journal_pin["path"]).parent
        external_names = {entry.name for entry in os.scandir(external_dir)}
        _need(EXTERNAL_JOURNAL_NAME in external_names
              and external_names <= ({EXTERNAL_JOURNAL_NAME, EXTERNAL_FAILURE_NAME}
                                     | EXTERNAL_PHASE_NAMES),
              "external attempt evidence membership differs")
        external_failure = None
        if EXTERNAL_FAILURE_NAME in external_names:
            failure_raw = _read_pinned(str(external_dir / EXTERNAL_FAILURE_NAME),
                                       None, None, EXTERNAL_FAILURE_MAX)
            external_failure = _json_loads(failure_raw, EXTERNAL_FAILURE_MAX)
            external_failure = _validated_failure(external_failure,
                                                  journal["attempt_id"])
        if root is None:
            if external_failure is None:
                return _report("pre_root_incomplete", "consumed_without_root_or_failure",
                               journal_pin=journal_pin)
            if (external_failure["root_binding"] is not None
                    or external_failure["partial_root"] is not None):
                return _report("uninspectable", "absent_root_conflicts_with_failure_binding",
                               journal_pin=journal_pin)
            return _report("pre_root_failed", "launcher_failure_precedence",
                           journal_pin=journal_pin)
        if (external_failure is not None
                and external_failure["partial_root"] is not None):
            try:
                identity, header_sha = _inspect_partial_root(
                    root, external_failure["partial_root"])
            except (OSError, ControlError):
                return _report("uninspectable", "launcher_partial_root_mismatch",
                               journal_pin=journal_pin)
            return _report("launcher_failed_with_partial_root",
                           "launcher_failure_precedence",
                           root_identity=identity, header_sha256=header_sha,
                           journal_pin=journal_pin)
        if (external_failure is not None
                and external_failure["root_binding"] is None):
            return _report("uninspectable", "failure_binding_conflicts_with_supplied_root",
                           journal_pin=journal_pin)
        root_path = os.fspath(root)
        _need(type(root_path) is str and os.path.isabs(root_path), "root path invalid")
        root_info = os.lstat(root_path)
        _need(stat.S_ISDIR(root_info.st_mode) and not stat.S_ISLNK(root_info.st_mode),
              "root is not a non-symlink directory")
        _verify_big_parent(root_path)
        dirfd = os.open(root_path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        opened = os.fstat(dirfd)
        _need((root_info.st_dev, root_info.st_ino) == (opened.st_dev, opened.st_ino),
              "root changed during open")
    except (OSError, ControlError, ValueError, TypeError):
        if dirfd is not None:
            os.close(dirfd)
        return _report("uninspectable", "root_or_journal_invalid")
    lockfd = None
    try:
        try:
            lockfd = os.open("store.lock", os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                             dir_fd=dirfd)
            lock_info = os.fstat(lockfd)
            _need(stat.S_ISREG(lock_info.st_mode) and lock_info.st_nlink == 1
                  and lock_info.st_size == 0, "store lock invalid")
            fcntl.flock(lockfd, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            return _report("uninspectable", "writer_live")
        except OSError:
            return _report("uninspectable", "lock_invalid")
        try:
            store, scanned = storage._inspect_dirfd(dirfd)
            _need(_signature(lock_info) == _signature(scanned["store.lock"]),
                  "held lock differs from inspected lock")
        except (OSError, storage.StoreError):
            return _report("uninspectable", "store_snapshot_invalid")
        identity = [os.fstat(dirfd).st_dev, os.fstat(dirfd).st_ino]
        header_raw, _ = storage._read_regular(
            dirfd, "store-header.json", maximum=storage._RECEIPT_MAX,
            expected=scanned["store-header.json"])
        header_sha = hashlib.sha256(header_raw).hexdigest()
        if external_failure is not None and external_failure["root_binding"] is not None:
            failure_root = external_failure["root_binding"]
            if ((failure_root["path"], failure_root["device"], failure_root["inode"],
                 failure_root["header_sha256"])
                    != (root_path, identity[0], identity[1], header_sha)):
                return _report("uninspectable", "launcher_failure_root_binding_mismatch",
                               store=store, root_identity=identity,
                               header_sha256=header_sha, journal_pin=journal_pin)
        if store["header"]["profile"] != expected_store_profile:
            return _report("uninspectable", "store_profile_mismatch", store=store,
                           root_identity=identity, header_sha256=header_sha,
                           journal_pin=journal_pin)

        receipts = {row["name"]:row for row in store["receipts"]}
        phase_indices = []
        for name in receipts:
            match = _PHASE.fullmatch(name)
            if match:
                phase_indices.append(int(match.group(1)))
        phase_indices.sort()
        if phase_indices != list(range(len(phase_indices))) or len(phase_indices) > MAX_PHASE_RECORDS:
            return _report("uninspectable", "phase_membership_invalid", store=store,
                           root_identity=identity, header_sha256=header_sha,
                           journal_pin=journal_pin)
        records = []
        try:
            for index in phase_indices:
                name = phase_name(index)
                raw, _ = storage._read_regular(dirfd, name, maximum=policy.MAX_RECORD_BYTES,
                                                expected=scanned[name])
                records.append(codec.json_loads(raw, max_bytes=policy.MAX_RECORD_BYTES))
            if not records and store["terminal"]:
                return _report("store_failed_without_policy_record", "store_failure_precedence",
                               store=store, metadata_complete=False, artifacts=False,
                               root_identity=identity, header_sha256=header_sha,
                               journal_pin=journal_pin)
            if not records and external_failure is not None:
                external_status = ("pre_root_failed" if external_failure["root_binding"] is None
                                   else "launcher_failed_with_root")
                return _report(external_status, "launcher_failure_precedence",
                               store=store, metadata_complete=False, artifacts=False,
                               root_identity=identity, header_sha256=header_sha,
                               journal_pin=journal_pin)
            _need(records, "missing phase prefix")
            validation = policy.validate_transcript(
                records, evidence_kind=expected_evidence_kind, require_complete=False)
        except (OSError, KeyError, ControlError, ValueError, TypeError):
            return _report("uninspectable", "phase_prefix_invalid", store=store,
                           root_identity=identity, header_sha256=header_sha,
                           journal_pin=journal_pin)
        validation = dict(validation)
        validation["last_event"] = records[-1]["event"]

        actual_names = set()
        actual_refs_ok = True
        for record in records:
            for ref in record["artifacts"]:
                actual_names.add(ref["name"])
                receipt = receipts.get(ref["name"])
                receipt_file = ref["receipt_name"]
                if receipt is None or receipt_file not in scanned:
                    actual_refs_ok = False
                    continue
                receipt_raw, _ = storage._read_regular(
                    dirfd, receipt_file, maximum=storage._RECEIPT_MAX,
                    expected=scanned[receipt_file])
                if not (receipt["size"] == ref["size_bytes"]
                        and receipt["sha256"] == ref["sha256"]
                        and receipt["encoding"] == ref["encoding"]
                        and len(receipt_raw) == ref["receipt_size_bytes"]
                        and hashlib.sha256(receipt_raw).hexdigest() == ref["receipt_sha256"]):
                    actual_refs_ok = False

        metadata_names = (ROOT_JOURNAL_NAME, SOURCE_NAME, ENVIRONMENT_NAME)
        metadata_complete = all(name in receipts for name in metadata_names)
        metadata_ok = False
        if metadata_complete:
            try:
                copied_journal, _ = storage._read_regular(
                    dirfd, ROOT_JOURNAL_NAME, maximum=EXTERNAL_JOURNAL_MAX,
                    expected=scanned[ROOT_JOURNAL_NAME])
                source_raw, _ = storage._read_regular(
                    dirfd, SOURCE_NAME, maximum=environment.METADATA_CAP,
                    expected=scanned[SOURCE_NAME])
                environment_raw, _ = storage._read_regular(
                    dirfd, ENVIRONMENT_NAME, maximum=environment.METADATA_CAP,
                    expected=scanned[ENVIRONMENT_NAME])
                sources = codec.json_loads(source_raw, max_bytes=environment.METADATA_CAP)
                native_environment = codec.json_loads(
                    environment_raw, max_bytes=environment.METADATA_CAP)
                environment.validate_sources(sources, profile=environment.SCIENTIFIC)
                environment.validate_environment(native_environment, profile=environment.SCIENTIFIC)
                metadata_ok = (codec.json_bytes(sources) == source_raw
                    and codec.json_bytes(native_environment) == environment_raw
                    and copied_journal == journal_raw
                    and codec.tree_digest(sources) == records[0]["sources_sha256"]
                    and codec.tree_digest(native_environment) == records[0]["environment_sha256"]
                    and codec.tree_digest(sources) == journal["expected_source_set_sha256"]
                    and sources["repository_revision"] == journal["expected_commit"]
                    and native_environment["cuda"]["devices"][0]["uuid"]
                        == journal["expected_gpu_uuid"])
            except (OSError, KeyError, ValueError, TypeError):
                metadata_ok = False

        transition_result = {"status":"valid", "root_metadata_names":set()}
        phase_external_present = bool(external_names & EXTERNAL_PHASE_NAMES)
        if metadata_ok and (len(records) > PRE_TRANSITION_RECORDS
                or phase_external_present or any(
                name.startswith("native-") and name.endswith(("-permission.json",
                                                               "-environment.json"))
                for name in receipts)):
            try:
                import phase_transition as transition
                transition_result = transition.inspect_transition_evidence(
                    external_dir=external_dir, external_names_found=external_names,
                    journal_pin=journal_pin,
                    root_binding={"path":root_path, "device":identity[0],
                                  "inode":identity[1], "header_sha256":header_sha},
                    records=records, receipts=receipts, scanned=scanned, dirfd=dirfd,
                    sources=sources, storage=storage, codec=codec,
                    environment_schema=environment, fixture=fixture)
            except (OSError, KeyError, ValueError, TypeError, RuntimeError, ControlError):
                transition_result = {"status":"invalid", "root_metadata_names":set()}

        bindings_ok = all(record["root_binding"] is None or
            (record["root_binding"]["device"] == identity[0]
             and record["root_binding"]["inode"] == identity[1]
             and record["root_binding"]["header_sha256"] == header_sha
             and record["root_binding"]["path"] == root_path)
            for record in records)
        actual_refs_ok = actual_refs_ok and bindings_ok
        pending_outputs = set()
        if validation["interrupted_work"] and validation["pending_operation"] is not None:
            pending = next((row for row in policy.schedule()
                            if row["operation"] == validation["pending_operation"]), None)
            if pending is None:
                return _report("uninspectable", "pending_operation_invalid", store=store,
                               root_identity=identity, header_sha256=header_sha,
                               journal_pin=journal_pin)
            pending_outputs.update(pending["artifacts"])
        expected_receipts = (set(phase_name(i) for i in range(len(records)))
                             | actual_names | pending_outputs | set(metadata_names)
                             | set(transition_result["root_metadata_names"]))
        unknown_complete = set(receipts) - expected_receipts
        if store["terminal"]:
            terminal_status = ("store_terminal_with_policy_failure"
                if validation["status"] == "declared_failed" else
                "store_terminal_after_complete"
                if validation["status"] == "declared_complete" else
                "store_failed_without_policy_record")
            return _report(terminal_status, "store_failure_precedence",
                           phase_records=len(records), validation=validation, store=store,
                           metadata_complete=metadata_ok, artifacts=actual_refs_ok,
                           root_identity=identity, header_sha256=header_sha,
                           journal_pin=journal_pin)
        if external_failure is not None:
            external_status = ("pre_root_failed" if external_failure["root_binding"] is None
                               else "launcher_failed_with_root")
            return _report(external_status, "launcher_failure_precedence",
                           phase_records=len(records), validation=validation, store=store,
                           metadata_complete=metadata_ok, artifacts=actual_refs_ok,
                           root_identity=identity, header_sha256=header_sha,
                           journal_pin=journal_pin)
        if transition_result["status"] == "invalid":
            return _report("uninspectable", "transition_evidence_invalid",
                           phase_records=len(records), validation=validation, store=store,
                           metadata_complete=metadata_ok, artifacts=False,
                           root_identity=identity, header_sha256=header_sha,
                           journal_pin=journal_pin)
        if transition_result["status"] == "transition_incomplete":
            return _report("transition_incomplete", "consumed_transition_nonresumable",
                           phase_records=len(records), validation=validation, store=store,
                           metadata_complete=metadata_ok, artifacts=False,
                           root_identity=identity, header_sha256=header_sha,
                           journal_pin=journal_pin)
        if unknown_complete:
            return _report("uninspectable", "unknown_complete_payload", phase_records=len(records),
                           validation=validation, store=store, metadata_complete=metadata_complete,
                           artifacts=False, root_identity=identity, header_sha256=header_sha,
                           journal_pin=journal_pin)
        if not metadata_ok or not actual_refs_ok:
            return _report("uninspectable", "binding_or_artifact_mismatch",
                           phase_records=len(records), validation=validation, store=store,
                           metadata_complete=metadata_ok, artifacts=actual_refs_ok,
                           root_identity=identity, header_sha256=header_sha,
                           journal_pin=journal_pin)
        if validation["status"] == "declared_failed":
            status, reason = "declared_failed", "retained_policy_failure"
        elif validation["status"] == "declared_complete":
            status, reason = "complete", "complete_verified_attempt"
        elif records[-1]["event"] == "boundary":
            status, reason = "sealed_boundary", "verified_boundary_requires_next_go"
        else:
            status, reason = "interrupted_incomplete", "incomplete_prefix_nonresumable"
        if status in ("sealed_boundary", "complete"):
            try:
                _positive_snapshot_stable(root_path, dirfd, lockfd, root_info,
                                          scanned, storage)
            except (OSError, ControlError, storage.StoreError):
                return _report("uninspectable", "final_snapshot_changed", store=store,
                               root_identity=identity, header_sha256=header_sha,
                               journal_pin=journal_pin)
        return _report(status, reason, phase_records=len(records), validation=validation,
                       store=store, metadata_complete=True, artifacts=True,
                       root_identity=identity, header_sha256=header_sha,
                       journal_pin=journal_pin)
    except (OSError, KeyError, ValueError, TypeError, ControlError):
        return _report("uninspectable", "inspection_invariant_failed")
    finally:
        if lockfd is not None:
            try:
                fcntl.flock(lockfd, fcntl.LOCK_UN)
            finally:
                os.close(lockfd)
        os.close(dirfd)


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    _need(type(args) is list and not args, "native control has no launch CLI")
    print('{"status":"inert"}')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
