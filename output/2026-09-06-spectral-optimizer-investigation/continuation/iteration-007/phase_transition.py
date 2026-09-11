"""Inert later-phase permission and boundary-only writer acquisition.

Import and the default CLI are stdlib-only and perform no filesystem work.
Explicit APIs authenticate externally reviewed bytes before importing any
Torch-bearing I7 module.  They never append a policy record or authorize a GO.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from datetime import datetime
from pathlib import Path
import re
import stat
import sys


class TransitionError(RuntimeError):
    def __init__(self, message, *, failure_status=None):
        super().__init__(message)
        self.failure_status = failure_status


PERMISSION_SCHEMA = "i7_native_phase_permission_v2"
TRANSITION_SCHEMA = "i7_native_phase_transition_v1"
ATTEMPT_ID = "i7-native-development-attempt-001"
DEVELOPMENT_ATTEMPT_PATH = "/tmp/spectral-experiment-artifacts/i7-native-development-attempt-001"
PHASES = ("primary", "sensitivity", "audit")
SCOPES = {
    "primary":"frozen_current32_primary_three_bundle_and_four_plan_preparation",
    "sensitivity":"frozen_current32_sensitivity_b71901",
    "audit":"frozen_independent_cpu_audit_16_groups",
}
EVIDENCE_KINDS = ("synthetic_contract_fixture", "native_producer_attestation")
PREVIOUS = {
    "primary":("development", 10, 9, "development_complete"),
    "sensitivity":("primary", 44, 43, "primary_complete"),
    "audit":("sensitivity", 56, 55, "sensitivity_complete"),
}
PERMISSION_MAX = 8 << 10
TRANSITION_MAX = 8 << 10
PIN_FIELDS = ("path", "size_bytes", "sha256")
ROOT_FIELDS = ("path", "device", "inode", "header_sha256")
REF_FIELDS = ("name", "status", "encoding", "size_bytes", "sha256",
              "receipt_name", "receipt_size_bytes", "receipt_sha256")
PRIOR_FIELDS = ("phase", "phase_records", "root_binding", "journal_pin",
                "boundary_ref", "inspection")
PERMISSION_FIELDS = ("schema", "attempt_id", "phase", "scope", "authorized_utc",
    "decision", "retry_allowed", "expected_commit", "expected_source_set_sha256",
    "expected_environment_sha256", "expected_environment_role", "expected_gpu_uuid",
    "storage_admission_pin", "root_binding", "journal_pin", "previous_boundary_ref")
PERMISSION_WRITE_FIELDS = tuple(
    field for field in PERMISSION_FIELDS if field != "storage_admission_pin")
TRANSITION_FIELDS = ("schema", "status", "attempt_id", "phase", "scope",
                     "permission_pin", "root_binding", "journal_pin",
                     "previous_boundary_ref", "retry_allowed")
ACQUIRED_FIELDS = ("store", "guard", "phase", "permission", "permission_pin",
                   "transition_pin", "records", "evidence_kind", "sources",
                   "environment", "source_environment", "root_binding",
                   "journal_pin", "previous_boundary_ref", "execution_authorized",
                   "scientific_execution_certified")
_SHA = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)
_COMMIT = re.compile(r"[0-9a-f]{40}\Z", re.ASCII)
_UTC = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z\Z",
                  re.ASCII)
_UUID = re.compile(r"GPU-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\Z",
                   re.ASCII)


def permission_name(phase):
    _need(phase in PHASES and type(phase) is str, "invalid transition phase")
    return f"native-{phase}-permission.json"


def transition_name(phase):
    _need(phase in PHASES and type(phase) is str, "invalid transition phase")
    return f"native-{phase}-transition.json"


def environment_name(phase):
    _need(phase in PHASES and type(phase) is str, "invalid transition phase")
    return f"native-{phase}-environment.json"


def external_names():
    return frozenset(name for phase in PHASES
                     for name in (permission_name(phase), transition_name(phase)))


def _need(condition, message):
    if not condition:
        raise TransitionError(message)


def _keys(value, fields, label):
    _need(type(value) is dict and tuple(value) == fields
          and all(type(key) is str for key in value), label + ": exact ordered fields")


def _sha(value, label):
    _need(type(value) is str and _SHA.fullmatch(value) is not None,
          label + ": invalid SHA-256")


def _json_tree(value, active=None):
    active = set() if active is None else active
    if value is None or type(value) in (bool, int, str):
        if type(value) is str:
            _need(not any(0xD800 <= ord(char) <= 0xDFFF for char in value),
                  "surrogate JSON string")
        return
    if type(value) is float:
        _need(math.isfinite(value), "nonfinite JSON float")
        return
    _need(type(value) in (list, dict) and id(value) not in active,
          "unsupported or cyclic JSON value")
    active.add(id(value))
    rows = value.items() if type(value) is dict else enumerate(value)
    for key, item in rows:
        if type(value) is dict:
            _need(type(key) is str, "JSON key must be string")
        _json_tree(item, active)
    active.remove(id(value))


def _json_bytes(value):
    _json_tree(value)
    return (json.dumps(value, ensure_ascii=True, allow_nan=False, sort_keys=False,
                       separators=(",", ":")) + "\n").encode("ascii")


def _json_loads(raw, maximum):
    _need(type(raw) is bytes and 0 < len(raw) <= maximum, "bounded JSON required")
    try:
        text = raw.decode("ascii")
    except UnicodeDecodeError as exc:
        raise TransitionError("permission JSON is not ASCII") from exc
    def pairs(rows):
        result = {}
        for key, value in rows:
            _need(key not in result, "duplicate JSON key")
            result[key] = value
        return result
    def constant(_):
        raise TransitionError("nonfinite JSON constant")
    try:
        value = json.loads(text, object_pairs_hook=pairs, parse_constant=constant)
    except (json.JSONDecodeError, RecursionError, ValueError) as exc:
        if isinstance(exc, TransitionError):
            raise
        raise TransitionError("malformed permission JSON") from exc
    _need(_json_bytes(value) == raw, "noncanonical permission JSON")
    return value


def _read_pinned(pin, maximum):
    _keys(pin, PIN_FIELDS, "file pin")
    path, size, digest = pin["path"], pin["size_bytes"], pin["sha256"]
    _need(type(path) is str and path.startswith("/") and path != "/"
          and os.path.normpath(path) == path, "pin path must be canonical absolute")
    _need(type(size) is int and type(size) is not bool and 0 < size <= maximum,
          "pin size invalid")
    _sha(digest, "pin")
    parts = path.split("/")[1:]
    dirfd = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    fd = None
    try:
        for part in parts[:-1]:
            _need(part not in ("", ".", ".."), "pin component invalid")
            nextfd = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                             dir_fd=dirfd)
            os.close(dirfd)
            dirfd = nextfd
        fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                     dir_fd=dirfd)
        before = os.fstat(fd)
        _need(stat.S_ISREG(before.st_mode) and before.st_nlink == 1
              and before.st_size == size, "pinned file identity differs")
        raw = b""
        while len(raw) < size:
            part = os.read(fd, size - len(raw))
            if not part:
                break
            raw += part
        after = os.fstat(fd)
        path_after = os.stat(parts[-1], dir_fd=dirfd, follow_symlinks=False)
        signature = lambda info:(info.st_dev, info.st_ino, info.st_size,
                                  info.st_mtime_ns, info.st_ctime_ns,
                                  info.st_mode, info.st_nlink)
        _need(signature(before) == signature(after) == signature(path_after),
              "pinned file changed during read")
    finally:
        if fd is not None:
            os.close(fd)
        os.close(dirfd)
    _need(len(raw) == size and hashlib.sha256(raw).hexdigest() == digest,
          "pinned file bytes differ")
    return raw


def _pin(value, label):
    _keys(value, PIN_FIELDS, label)
    _need(type(value["path"]) is str and value["path"].startswith("/")
          and os.path.normpath(value["path"]) == value["path"], label + ": path")
    _need(type(value["size_bytes"]) is int and type(value["size_bytes"]) is not bool
          and value["size_bytes"] > 0, label + ": size")
    _sha(value["sha256"], label)


def _root(value):
    _keys(value, ROOT_FIELDS, "root binding")
    _need(type(value["path"]) is str and value["path"].startswith("/tmp/spectral-experiment-artifacts/")
          and os.path.normpath(value["path"]) == value["path"], "root path invalid")
    for key in ("device", "inode"):
        _need(type(value[key]) is int and type(value[key]) is not bool and value[key] >= 0,
              "root identity invalid")
    _need(value["inode"] > 0, "root inode invalid")
    _sha(value["header_sha256"], "root header")


def _reference(value, expected_name):
    _keys(value, REF_FIELDS, "boundary reference")
    _need(value["name"] == expected_name and value["status"] == "complete"
          and value["encoding"] == "bytes", "boundary reference identity differs")
    for key in ("size_bytes", "receipt_size_bytes"):
        _need(type(value[key]) is int and type(value[key]) is not bool and value[key] > 0,
              "boundary reference size invalid")
    for key in ("sha256", "receipt_sha256"):
        _sha(value[key], "boundary reference " + key)
    expected_receipt = "receipt-" + hashlib.sha256(expected_name.encode("ascii")).hexdigest() + ".json"
    _need(value["receipt_name"] == expected_receipt, "boundary receipt name differs")


def _permission_value(value, pin, phase, *, fixture):
    _keys(value, PERMISSION_FIELDS, "phase permission")
    previous_phase, count, index, _ = PREVIOUS[phase]
    expected_boundary = f"native-phase-{index:03d}.json"
    _need(value["schema"] == PERMISSION_SCHEMA and value["attempt_id"] == ATTEMPT_ID
          and value["phase"] == phase and value["scope"] == SCOPES[phase]
          and value["decision"] == "go" and value["retry_allowed"] is False,
          "phase permission scope differs")
    _need(type(value["authorized_utc"]) is str
          and _UTC.fullmatch(value["authorized_utc"]) is not None,
          "permission UTC invalid")
    try:
        parsed_utc = datetime.strptime(value["authorized_utc"], "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as exc:
        raise TransitionError("permission UTC invalid") from exc
    _need(parsed_utc.strftime("%Y-%m-%dT%H:%M:%SZ") == value["authorized_utc"],
          "permission UTC invalid")
    _need(type(value["expected_commit"]) is str
          and _COMMIT.fullmatch(value["expected_commit"]) is not None,
          "permission commit invalid")
    _sha(value["expected_source_set_sha256"], "permission source set")
    _sha(value["expected_environment_sha256"], "permission environment")
    expected_role = "cpu_audit" if phase == "audit" else "native_source"
    _need(value["expected_environment_role"] == expected_role,
          "permission environment role differs")
    if phase == "audit":
        _need(value["expected_gpu_uuid"] is None, "audit permission must omit GPU UUID")
    else:
        _need(type(value["expected_gpu_uuid"]) is str
              and _UUID.fullmatch(value["expected_gpu_uuid"]) is not None,
              "permission GPU UUID invalid")
    import native_storage_authority as authority
    try:
        authority.validate_storage_admission_pin(
            value["storage_admission_pin"], fixture=fixture)
    except authority.StorageAuthorityError as exc:
        raise TransitionError("permission storage admission pin invalid") from exc
    _root(value["root_binding"])
    _pin(value["journal_pin"], "journal pin")
    _reference(value["previous_boundary_ref"], expected_boundary)
    if pin is not None:
        _pin(pin, "permission pin")
        _need(Path(pin["path"]).name == permission_name(phase)
              and Path(pin["path"]).parent
                  == Path(value["journal_pin"]["path"]).parent,
              "permission path differs from attempt journal")
    return value


def _permission(raw, pin, phase, *, fixture=False):
    _need(type(fixture) is bool, "permission mode invalid")
    value = _json_loads(raw, PERMISSION_MAX)
    return _permission_value(value, pin, phase, fixture=fixture)


def _marker(value, phase):
    _keys(value, TRANSITION_FIELDS, "transition marker")
    _need(value["schema"] == TRANSITION_SCHEMA and value["status"] == "consumed"
          and value["attempt_id"] == ATTEMPT_ID and value["phase"] == phase
          and value["scope"] == SCOPES[phase] and value["retry_allowed"] is False,
          "transition marker scope differs")
    _pin(value["permission_pin"], "transition permission pin")
    _root(value["root_binding"])
    _pin(value["journal_pin"], "transition journal pin")
    _reference(value["previous_boundary_ref"],
               f"native-phase-{PREVIOUS[phase][2]:03d}.json")
    return value


def _fixture_mode(evidence_kind):
    _need(type(evidence_kind) is str and evidence_kind in EVIDENCE_KINDS,
          "invalid evidence kind")
    return evidence_kind == "synthetic_contract_fixture"


def _journal_value(journal_pin, *, fixture):
    import native_control as control
    raw = _read_pinned(journal_pin, control.EXTERNAL_JOURNAL_MAX)
    value = control._json_loads(raw, control.EXTERNAL_JOURNAL_MAX)
    value = control._validate_journal(value, fixture=fixture)
    _need(Path(journal_pin["path"]).name == control.EXTERNAL_JOURNAL_NAME
          and Path(journal_pin["path"]).parent.name == value["attempt_id"],
          "journal path/attempt differs")
    return value


def _authenticate_prior(value, phase, prior_boundary, evidence_kind):
    """Authenticate the actual predecessor and journal without importing Torch."""
    fixture = _fixture_mode(evidence_kind)
    _need(type(phase) is str and phase in PHASES, "invalid transition phase")
    _keys(prior_boundary, PRIOR_FIELDS, "prior boundary")
    previous_phase, count, index, operation = PREVIOUS[phase]
    _need(prior_boundary["phase"] == previous_phase
          and prior_boundary["phase_records"] == count,
          "prior boundary phase/count differs")
    _need(value["root_binding"] == prior_boundary["root_binding"]
          and value["journal_pin"] == prior_boundary["journal_pin"]
          and value["previous_boundary_ref"] == prior_boundary["boundary_ref"],
          "permission differs from supplied prior boundary")
    _root(prior_boundary["root_binding"])
    _pin(prior_boundary["journal_pin"], "prior journal pin")
    _reference(prior_boundary["boundary_ref"], f"native-phase-{index:03d}.json")
    inspection = prior_boundary["inspection"]
    _need(type(inspection) is dict and inspection.get("status") == "sealed_boundary"
          and inspection.get("phase_records") == count
          and inspection.get("last_event") == "boundary"
          and inspection.get("root_identity")
              == [value["root_binding"]["device"], value["root_binding"]["inode"]]
          and inspection.get("header_sha256") == value["root_binding"]["header_sha256"]
          and inspection.get("journal_pin") == value["journal_pin"]
          and inspection.get("eligible_for_writable_reopen") is True,
          "prior inspection is not an eligible sealed boundary")
    header_pin = {"path":value["root_binding"]["path"] + "/store-header.json",
                  "size_bytes":0, "sha256":value["root_binding"]["header_sha256"]}
    header_path = Path(header_pin["path"])
    info = os.lstat(header_path)
    _need(stat.S_ISREG(info.st_mode) and info.st_nlink == 1 and info.st_size > 0,
          "actual root header invalid")
    root_info = os.lstat(value["root_binding"]["path"])
    _need(stat.S_ISDIR(root_info.st_mode) and not stat.S_ISLNK(root_info.st_mode)
          and (root_info.st_dev, root_info.st_ino)
              == (value["root_binding"]["device"], value["root_binding"]["inode"]),
          "actual root identity differs")
    header_pin["size_bytes"] = info.st_size
    _read_pinned(header_pin, 4096)
    journal = _journal_value(value["journal_pin"], fixture=fixture)
    _need(journal["attempt_id"] == value["attempt_id"]
          and journal["expected_commit"] == value["expected_commit"]
          and journal["expected_source_set_sha256"]
              == value["expected_source_set_sha256"]
          and journal["storage_admission_pin"] == value["storage_admission_pin"],
          "permission differs from authenticated journal")
    boundary_raw = _read_pinned({"path":value["root_binding"]["path"] + "/"
        + value["previous_boundary_ref"]["name"],
        "size_bytes":value["previous_boundary_ref"]["size_bytes"],
        "sha256":value["previous_boundary_ref"]["sha256"]}, 128 << 10)
    boundary = _json_loads(boundary_raw, 128 << 10)
    _need(type(boundary) is dict and boundary.get("sequence") == index
          and boundary.get("event") == "boundary" and boundary.get("operation") == operation
          and boundary.get("phase") == previous_phase
          and boundary.get("sources_sha256") == value["expected_source_set_sha256"],
          "actual prior boundary content differs")
    receipt_raw = _read_pinned({"path":value["root_binding"]["path"] + "/"
        + value["previous_boundary_ref"]["receipt_name"],
        "size_bytes":value["previous_boundary_ref"]["receipt_size_bytes"],
        "sha256":value["previous_boundary_ref"]["receipt_sha256"]}, 4096)
    _need(bool(receipt_raw), "actual boundary receipt empty")
    return journal


def _authenticate_actual(permission_pin, phase, prior_boundary, evidence_kind):
    fixture = _fixture_mode(evidence_kind)
    _need(type(phase) is str and phase in PHASES, "invalid transition phase")
    raw = _read_pinned(permission_pin, PERMISSION_MAX)
    value = _permission(raw, permission_pin, phase, fixture=fixture)
    if not fixture:
        _need(Path(permission_pin["path"]).parent == Path(DEVELOPMENT_ATTEMPT_PATH),
              "native phase permission path differs from fixed attempt")
    _authenticate_prior(value, phase, prior_boundary, evidence_kind)
    return raw, value


def authenticate_permission_for_write(
        value, *, prior_boundary, attempt_parent=DEVELOPMENT_ATTEMPT_PATH,
        fixture=False):
    """Build and authenticate bounded permission bytes before exclusive creation.

    `value` deliberately omits the storage-admission pin.  That pin is derived
    only from the authenticated development journal.  `fixture=True` selects
    the exact `synthetic_contract_fixture` evidence literal; otherwise the
    helper selects `native_producer_attestation` and the fixed attempt path.
    """
    _need(type(fixture) is bool, "permission write mode invalid")
    _keys(value, PERMISSION_WRITE_FIELDS, "permission write input")
    phase = value["phase"]
    _need(type(phase) is str and phase in PHASES, "invalid transition phase")
    _keys(prior_boundary, PRIOR_FIELDS, "prior boundary")
    _pin(prior_boundary["journal_pin"], "prior journal pin")
    parent = Path(os.fspath(attempt_parent))
    _need(parent.is_absolute() and str(parent) == os.fspath(attempt_parent)
          and os.path.normpath(str(parent)) == str(parent),
          "permission attempt parent invalid")
    if not fixture:
        _need(str(parent) == DEVELOPMENT_ATTEMPT_PATH,
              "native permission attempt path differs")
    _need(parent == Path(prior_boundary["journal_pin"]["path"]).parent,
          "permission attempt differs from prior journal")
    evidence_kind = ("synthetic_contract_fixture" if fixture
                     else "native_producer_attestation")
    journal = _journal_value(prior_boundary["journal_pin"], fixture=fixture)
    permission = {}
    for field in PERMISSION_FIELDS:
        permission[field] = (dict(journal["storage_admission_pin"])
                             if field == "storage_admission_pin" else value[field])
    permission = _permission_value(permission, None, phase, fixture=fixture)
    import native_storage_authority as authority
    try:
        raw = authority.encode_bounded(permission, maximum=PERMISSION_MAX)
    except authority.StorageAuthorityError as exc:
        raise TransitionError("permission exceeds bounded JSON domain") from exc
    prospective_pin = _pin_for(parent / permission_name(phase), raw)
    permission = _permission(raw, prospective_pin, phase, fixture=fixture)
    _authenticate_prior(permission, phase, prior_boundary, evidence_kind)

    info = os.lstat(parent)
    _need(stat.S_ISDIR(info.st_mode) and not stat.S_ISLNK(info.st_mode),
          "permission attempt directory invalid")
    names = set()
    with os.scandir(parent) as entries:
        for entry in entries:
            names.add(entry.name)
            _need(len(names) <= 7, "permission prewrite namespace exceeds cardinality")
    prior_phases = PHASES[:PHASES.index(phase)]
    expected = {Path(prior_boundary["journal_pin"]["path"]).name}
    expected.update(name for prior in prior_phases
                    for name in (permission_name(prior), transition_name(prior)))
    _need(names == expected, "permission prewrite namespace differs")
    rows = _external_rows(parent, names, prior_boundary["journal_pin"],
                          prior_boundary["root_binding"], fixture=fixture)
    _need(set(rows) == set(prior_phases)
          and all(rows[item]["marker"] is not None for item in prior_phases),
          "permission predecessor transitions differ")
    return raw, permission


def _pin_for(path, raw):
    return {"path":str(path), "size_bytes":len(raw),
            "sha256":hashlib.sha256(raw).hexdigest()}


def _write_marker(control, permission_pin, permission):
    attempt = Path(permission_pin["path"]).parent
    value = {"schema":TRANSITION_SCHEMA, "status":"consumed",
        "attempt_id":ATTEMPT_ID, "phase":permission["phase"],
        "scope":permission["scope"], "permission_pin":dict(permission_pin),
        "root_binding":dict(permission["root_binding"]),
        "journal_pin":dict(permission["journal_pin"]),
        "previous_boundary_ref":dict(permission["previous_boundary_ref"]),
        "retry_allowed":False}
    raw = _json_bytes(value)
    dirfd = os.open(attempt, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        pin = control._write_new(dirfd, transition_name(permission["phase"]), raw,
                                 TRANSITION_MAX)
    except FileExistsError as exc:
        raise TransitionError("transition already consumed; retry refused") from exc
    finally:
        os.close(dirfd)
    pin["path"] = str(attempt / transition_name(permission["phase"]))
    return pin, value


def _existing_pin(path, maximum):
    info = os.lstat(path)
    _need(stat.S_ISREG(info.st_mode) and info.st_nlink == 1
          and 0 < info.st_size <= maximum, "external phase file identity invalid")
    provisional = {"path":str(path), "size_bytes":info.st_size, "sha256":"0" * 64}
    # Read once without trusting a caller digest, then bind and reread exactly.
    parts = str(path).split("/")[1:]
    dirfd = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    fd = None
    try:
        for part in parts[:-1]:
            nextfd = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                             dir_fd=dirfd)
            os.close(dirfd)
            dirfd = nextfd
        fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                     dir_fd=dirfd)
        raw = b""
        while len(raw) < info.st_size:
            chunk = os.read(fd, info.st_size - len(raw))
            if not chunk:
                break
            raw += chunk
        _need(len(raw) == info.st_size, "external phase file short read")
    finally:
        if fd is not None:
            os.close(fd)
        os.close(dirfd)
    provisional["sha256"] = hashlib.sha256(raw).hexdigest()
    return provisional, _read_pinned(provisional, maximum)


def _external_rows(external_dir, names, journal_pin, root_binding, *, fixture=False):
    _need(type(fixture) is bool, "external evidence mode invalid")
    journal = _journal_value(journal_pin, fixture=fixture)
    total = 0
    for name in names:
        path = Path(external_dir) / name
        info = os.lstat(path)
        _need(stat.S_ISREG(info.st_mode) and info.st_nlink == 1,
              "external phase evidence identity invalid")
        total += info.st_size
    _need(total <= 64 << 10, "external attempt evidence exceeds shared bound")
    rows = {}
    seen_gap = False
    for phase in PHASES:
        permission_file, marker_file = permission_name(phase), transition_name(phase)
        has_permission, has_marker = permission_file in names, marker_file in names
        _need(not has_marker or has_permission, "transition marker lacks permission")
        if seen_gap:
            _need(not has_permission and not has_marker, "future phase evidence appeared early")
            continue
        if not has_permission:
            seen_gap = True
            continue
        permission_pin, raw = _existing_pin(Path(external_dir) / permission_file,
                                            PERMISSION_MAX)
        permission = _permission(raw, permission_pin, phase, fixture=fixture)
        _need(permission["journal_pin"] == journal_pin
              and permission["root_binding"] == root_binding
              and permission["expected_commit"] == journal["expected_commit"]
              and permission["expected_source_set_sha256"]
                  == journal["expected_source_set_sha256"]
              and permission["storage_admission_pin"]
                  == journal["storage_admission_pin"],
              "phase permission root/journal/admission differs")
        marker = marker_pin = None
        if has_marker:
            marker_pin, marker_raw = _existing_pin(
                Path(external_dir) / marker_file, TRANSITION_MAX)
            marker = _marker(_json_loads(marker_raw, TRANSITION_MAX), phase)
            _need(marker["permission_pin"] == permission_pin
                  and marker["root_binding"] == root_binding
                  and marker["journal_pin"] == journal_pin
                  and marker["previous_boundary_ref"]
                      == permission["previous_boundary_ref"],
                  "transition marker differs from permission")
        else:
            seen_gap = True
        rows[phase] = {"permission":permission, "permission_pin":permission_pin,
                       "permission_raw":raw, "marker":marker,
                       "marker_pin":marker_pin}
    return rows


def validate_external_for_failure(external_dir, names, journal_pin, root_binding,
                                  *, fixture=False):
    """Torch-free validation used before appending the external failure file."""
    allowed = {"native-launch-journal.json", "native-launcher-failure.json"} | set(external_names())
    _need(set(names) <= allowed, "external phase evidence membership differs")
    return _external_rows(external_dir, set(names), journal_pin, root_binding,
                          fixture=fixture)


def inspect_transition_evidence(*, external_dir, external_names_found, journal_pin,
                                root_binding, records, receipts, scanned, dirfd,
                                sources, storage, codec, environment_schema,
                                fixture=False):
    """Validate phase metadata without adding operations to the fixed transcript."""
    rows = _external_rows(external_dir, set(external_names_found),
                          journal_pin, root_binding, fixture=fixture)
    count = len(records)
    if count < PREVIOUS["primary"][1]:
        allowed_phase = -1
    elif count <= PREVIOUS["sensitivity"][1]:
        allowed_phase = 0 if count < PREVIOUS["sensitivity"][1] else 1
    elif count <= PREVIOUS["audit"][1]:
        allowed_phase = 1 if count < PREVIOUS["audit"][1] else 2
    else:
        allowed_phase = 2
    for index, phase in enumerate(PHASES):
        _need(index <= allowed_phase or phase not in rows,
              "future phase permission appeared early")
    required = set()
    if count > PREVIOUS["primary"][1]:
        required.add("primary")
    if count > PREVIOUS["sensitivity"][1]:
        required.add("sensitivity")
    if count > PREVIOUS["audit"][1]:
        required.add("audit")
    _need(all(phase in rows and rows[phase]["marker"] is not None for phase in required),
          "advanced transcript lacks consumed transition")
    root_names = set()
    environments = {}
    incomplete = None
    for phase, row in rows.items():
        permission = row["permission"]
        marker = row["marker"]
        if marker is None:
            continue
        permission_file, environment_file = permission_name(phase), environment_name(phase)
        root_names.update((permission_file, environment_file))
        permission_receipt = receipts.get(permission_file)
        environment_receipt = receipts.get(environment_file)
        both = permission_receipt is not None and environment_receipt is not None
        if not both:
            _need(phase not in required, "advanced transcript lacks transition root copies")
            incomplete = phase
            continue
        permission_raw, _ = storage._read_regular(
            dirfd, permission_file, maximum=PERMISSION_MAX, expected=scanned[permission_file])
        _need(permission_raw == row["permission_raw"], "root permission copy differs")
        environment_raw, _ = storage._read_regular(
            dirfd, environment_file, maximum=environment_schema.METADATA_CAP,
            expected=scanned[environment_file])
        environment = codec.json_loads(
            environment_raw, max_bytes=environment_schema.METADATA_CAP)
        environment_schema.validate_environment(
            environment, profile=environment_schema.SCIENTIFIC)
        _need(codec.json_bytes(environment) == environment_raw
              and codec.tree_digest(environment)
                  == permission["expected_environment_sha256"]
              and environment["runtime_role"] == permission["expected_environment_role"],
              "retained phase environment differs")
        _need(permission["expected_source_set_sha256"] == codec.tree_digest(sources)
              and permission["expected_commit"] == sources["repository_revision"],
              "phase permission source differs")
        if phase == "audit":
            _need(permission["expected_gpu_uuid"] is None
                  and environment["cuda"]["initialized"] is False
                  and environment["cuda"]["devices"] == [],
                  "retained audit environment differs")
        else:
            _need(environment["cuda"]["devices"][0]["uuid"]
                  == permission["expected_gpu_uuid"],
                  "retained phase GPU differs")
        environments[phase] = environment_raw
        boundary_count = PREVIOUS[phase][1]
        if count == boundary_count:
            incomplete = phase
    if "sensitivity" in environments:
        _need(environments.get("primary") == environments["sensitivity"],
              "retained primary/sensitivity environments differ")
    for record in records:
        phase = record["phase"]
        if phase in PHASES:
            _need(phase in rows and rows[phase]["marker"] is not None,
                  "phase record lacks consumed transition")
            _need(record["environment_sha256"]
                  == rows[phase]["permission"]["expected_environment_sha256"],
                  "phase record environment differs from retained transition")
    for phase, row in rows.items():
        actual = _actual_reference(
            f"native-phase-{PREVIOUS[phase][2]:03d}.json", receipts, scanned,
            dirfd, storage)
        _need(actual == row["permission"]["previous_boundary_ref"],
              "permission predecessor boundary differs from retained bytes")
    # A prospective next permission without a marker is allowed only at its exact boundary.
    for phase, row in rows.items():
        if row["marker"] is None:
            _need(count == PREVIOUS[phase][1],
                  "unconsumed permission is not immediately prospective")
    return {"status":"transition_incomplete" if incomplete is not None else "valid",
            "incomplete_phase":incomplete, "root_metadata_names":root_names,
            "transition_metadata_verified":incomplete is None}


def _actual_reference(name, receipts, scanned, dirfd, storage):
    receipt = receipts.get(name)
    _need(type(receipt) is dict and name in scanned,
          "boundary payload/receipt missing")
    receipt_name = "receipt-" + hashlib.sha256(name.encode("ascii")).hexdigest() + ".json"
    _need(receipt_name in scanned, "boundary receipt missing")
    receipt_raw, _ = storage._read_regular(
        dirfd, receipt_name, maximum=4096, expected=scanned[receipt_name])
    return {"name":name, "status":"complete", "encoding":receipt["encoding"],
        "size_bytes":receipt["size"], "sha256":receipt["sha256"],
        "receipt_name":receipt_name, "receipt_size_bytes":len(receipt_raw),
        "receipt_sha256":hashlib.sha256(receipt_raw).hexdigest()}


def _load_records(store, storage, control, codec, policy, count, evidence_kind,
                  *, permission=None, include_current_metadata=False,
                  sources=None, environment_schema=None):
    report, scanned = storage._inspect_dirfd(store._dirfd)
    _need(report["terminal"] is False, "prior store became terminal")
    receipts = {row["name"]:row for row in report["receipts"]}
    phase_names = sorted(name for name in receipts if name.startswith("native-phase-"))
    _need(phase_names == [control.phase_name(index) for index in range(count)],
          "held prior phase prefix differs")
    records = []
    for index in range(count):
        name = control.phase_name(index)
        meta = store._indexed.get(name)
        _need(meta is not None, "prior policy record missing")
        raw, _ = storage._read_regular(store._dirfd, name,
                                       maximum=policy.MAX_RECORD_BYTES,
                                       expected=scanned[name])
        _need(len(raw) == meta["size"] and hashlib.sha256(raw).hexdigest() == meta["sha256"],
              "prior policy record bytes differ")
        records.append(codec.json_loads(raw, max_bytes=policy.MAX_RECORD_BYTES))
    policy.validate_transcript(records, evidence_kind=evidence_kind, require_complete=False)
    for record in records:
        for reference in record["artifacts"]:
            _need(_actual_reference(reference["name"], receipts, scanned,
                                    store._dirfd, storage) == reference,
                  "held artifact/receipt differs from policy reference")
    if permission is not None:
        _need(sources is not None and environment_schema is not None,
              "held metadata validation inputs required")
        journal_raw, _ = storage._read_regular(
            store._dirfd, control.ROOT_JOURNAL_NAME,
            maximum=control.EXTERNAL_JOURNAL_MAX,
            expected=scanned[control.ROOT_JOURNAL_NAME])
        source_raw, _ = storage._read_regular(
            store._dirfd, control.SOURCE_NAME,
            maximum=environment_schema.METADATA_CAP,
            expected=scanned[control.SOURCE_NAME])
        base_environment_raw, _ = storage._read_regular(
            store._dirfd, control.ENVIRONMENT_NAME,
            maximum=environment_schema.METADATA_CAP,
            expected=scanned[control.ENVIRONMENT_NAME])
        retained_sources = codec.json_loads(
            source_raw, max_bytes=environment_schema.METADATA_CAP)
        retained_base_environment = codec.json_loads(
            base_environment_raw, max_bytes=environment_schema.METADATA_CAP)
        environment_schema.validate_sources(
            retained_sources, profile=environment_schema.SCIENTIFIC)
        environment_schema.validate_environment(
            retained_base_environment, profile=environment_schema.SCIENTIFIC)
        _need(journal_raw == _read_pinned(permission["journal_pin"], 8 << 10)
              and source_raw == codec.json_bytes(sources)
              and codec.json_bytes(retained_sources) == source_raw
              and codec.json_bytes(retained_base_environment) == base_environment_raw
              and codec.tree_digest(retained_sources) == records[0]["sources_sha256"]
              and codec.tree_digest(retained_base_environment)
                  == records[0]["environment_sha256"],
              "held base metadata differs")

        attempt = Path(permission["journal_pin"]["path"]).parent
        external = {entry.name for entry in os.scandir(attempt)}
        _need("native-launch-journal.json" in external
              and "native-launcher-failure.json" not in external
              and external <= ({"native-launch-journal.json"} | set(external_names())),
              "held external attempt membership differs")
        rows = _external_rows(attempt, external, permission["journal_pin"],
                              permission["root_binding"],
                              fixture=(evidence_kind == "synthetic_contract_fixture"))
        current_index = PHASES.index(permission["phase"])
        _need(set(rows) == set(PHASES[:current_index + 1]),
              "held external phase order differs")
        retained_environment_bytes = {}
        for prior_phase in PHASES[:current_index + 1]:
            _need(prior_phase in rows and rows[prior_phase]["marker"] is not None,
                  "held phase lacks consumed external transition")
            row = rows[prior_phase]
            if prior_phase == permission["phase"]:
                _need(row["permission"] == permission,
                      "held current permission differs")
            _need(_actual_reference(
                    row["permission"]["previous_boundary_ref"]["name"], receipts,
                    scanned, store._dirfd, storage)
                  == row["permission"]["previous_boundary_ref"],
                  "held phase predecessor differs")
            if prior_phase == permission["phase"] and not include_current_metadata:
                continue
            root_permission_raw, _ = storage._read_regular(
                store._dirfd, permission_name(prior_phase), maximum=PERMISSION_MAX,
                expected=scanned[permission_name(prior_phase)])
            retained_environment_raw, _ = storage._read_regular(
                store._dirfd, environment_name(prior_phase),
                maximum=environment_schema.METADATA_CAP,
                expected=scanned[environment_name(prior_phase)])
            retained_environment = codec.json_loads(
                retained_environment_raw, max_bytes=environment_schema.METADATA_CAP)
            environment_schema.validate_environment(
                retained_environment, profile=environment_schema.SCIENTIFIC)
            expected = row["permission"]
            _need(root_permission_raw == row["permission_raw"]
                  and codec.json_bytes(retained_environment) == retained_environment_raw
                  and codec.tree_digest(retained_environment)
                      == expected["expected_environment_sha256"]
                  and retained_environment["runtime_role"]
                      == expected["expected_environment_role"],
                  "held phase metadata differs")
            if prior_phase == "audit":
                _need(expected["expected_gpu_uuid"] is None
                      and retained_environment["cuda"]["initialized"] is False
                      and retained_environment["cuda"]["devices"] == [],
                      "held audit environment differs")
            else:
                _need(retained_environment["cuda"]["devices"][0]["uuid"]
                      == expected["expected_gpu_uuid"],
                      "held phase GPU differs")
            retained_environment_bytes[prior_phase] = retained_environment_raw
        if "sensitivity" in retained_environment_bytes:
            _need(retained_environment_bytes.get("primary")
                  == retained_environment_bytes["sensitivity"],
                  "held primary/sensitivity environment differs")
        for record in records:
            if record["phase"] in PHASES:
                _need(record["environment_sha256"]
                      == rows[record["phase"]]["permission"]
                          ["expected_environment_sha256"],
                      "held policy environment differs from phase permission")
    expected_payloads = {control.ROOT_JOURNAL_NAME, control.SOURCE_NAME,
                         control.ENVIRONMENT_NAME}
    expected_payloads.update(control.phase_name(index) for index in range(count))
    for record in records:
        expected_payloads.update(ref["name"] for ref in record["artifacts"])
    phase = None if permission is None else permission["phase"]
    for prior in PHASES:
        if phase is not None and PHASES.index(prior) >= PHASES.index(phase):
            break
        expected_payloads.update((permission_name(prior), environment_name(prior)))
    if include_current_metadata:
        _need(phase is not None, "current transition metadata lacks phase")
        expected_payloads.update((permission_name(phase), environment_name(phase)))
    expected_files = {"store.lock", "store-header.json"} | expected_payloads
    expected_files.update("receipt-" + hashlib.sha256(name.encode("ascii")).hexdigest()
                          + ".json" for name in expected_payloads)
    _need(set(scanned) == expected_files, "held prior store membership differs")
    if permission is not None:
        actual = _actual_reference(permission["previous_boundary_ref"]["name"],
                                   receipts, scanned, store._dirfd, storage)
        _need(actual == permission["previous_boundary_ref"],
              "held predecessor boundary differs from permission")
    return records


def acquire_phase(permission_pin, *, phase, prior_boundary, entry_wall_origin,
                  entry_cpu_origin, entry_process_id, collect_metadata,
                  evidence_kind="native_producer_attestation", fixture_guard=None,
                  initialize_runtime=None):
    """Consume one phase transition and acquire only its sealed-boundary writer."""
    _need(type(evidence_kind) is str
          and evidence_kind in ("native_producer_attestation", "synthetic_contract_fixture"),
          "invalid evidence kind")
    raw, permission = _authenticate_actual(
        permission_pin, phase, prior_boundary, evidence_kind)
    if evidence_kind == "native_producer_attestation" and phase != "audit":
        _need(callable(initialize_runtime), "explicit native initializer required")
    else:
        _need(initialize_runtime is None, "audit/fixture forbids native initialization")

    # Permission is fully authenticated before these lazy imports, which may
    # transitively import Torch.  Inspection is intentionally before consumption.
    import artifact_store as storage
    import identity_codec as codec
    import native_control as control
    import native_phase_policy as policy
    import runtime_guard as runtime
    import source_environment_schema as environment_schema

    _need(type(evidence_kind) is str and evidence_kind in policy.KINDS,
          "invalid evidence kind")
    profile = (storage.SCIENTIFIC if evidence_kind == "native_producer_attestation"
               else storage.MLP_FIXTURE)
    inspected = control.inspect_native_attempt(
        permission["root_binding"]["path"], permission["journal_pin"],
        expected_store_profile=profile, expected_evidence_kind=evidence_kind)
    _need(inspected.get("status") == "sealed_boundary"
          and inspected.get("phase_records") == PREVIOUS[phase][1]
          and inspected.get("eligible_for_writable_reopen") is True,
          "actual prior boundary is not reopen-eligible")
    transition_pin = marker = None
    store = None
    failure_status = None
    marker_path = Path(permission_pin["path"]).parent / transition_name(phase)
    try:
        transition_pin, marker = _write_marker(control, permission_pin, permission)
        failure_status = "transition_marker_retained"
        if evidence_kind == "native_producer_attestation":
            _need(fixture_guard is None, "native transition forbids fixture guard")
            if phase == "audit":
                guard = runtime.RuntimeGuard(
                    phase, entry_wall_origin=entry_wall_origin,
                    entry_cpu_origin=entry_cpu_origin, entry_process_id=entry_process_id)
            else:
                guard = control._prepare_native_runtime(
                    runtime, permission, phase, initialize_runtime,
                    entry_wall_origin=entry_wall_origin, entry_cpu_origin=entry_cpu_origin,
                    entry_process_id=entry_process_id)
        else:
            _need(type(fixture_guard) is runtime.RuntimeGuard
                  and fixture_guard.profile == runtime.FIXTURE
                  and fixture_guard.phase == phase
                  and fixture_guard.entry_origins
                      == {"wall_origin":entry_wall_origin,
                          "cpu_origin":entry_cpu_origin},
                  "synthetic transition requires exact fixture guard")
            guard = fixture_guard
        guard.check("transition.permission.consumed")
        _need(callable(collect_metadata), "fresh metadata collector required")
        sources, fresh_environment = collect_metadata(permission)
        environment_schema.validate_sources(sources, profile=environment_schema.SCIENTIFIC)
        environment_schema.validate_environment(
            fresh_environment, profile=environment_schema.SCIENTIFIC)
        _need(sources["repository_revision"] == permission["expected_commit"]
              and codec.tree_digest(sources) == permission["expected_source_set_sha256"],
              "fresh source binding differs from permission")
        _need(fresh_environment["runtime_role"]
              == permission["expected_environment_role"]
              and codec.tree_digest(fresh_environment)
                  == permission["expected_environment_sha256"],
              "fresh phase environment differs from permission")
        if phase == "audit":
            _need(fresh_environment["cuda"]["initialized"] is False
                  and fresh_environment["cuda"]["devices"] == [],
                  "audit environment is not CUDA-uninitialized")
        else:
            _need(fresh_environment["cuda"]["devices"][0]["uuid"]
                  == permission["expected_gpu_uuid"],
                  "phase environment GPU differs from permission")
        guard.check("transition.metadata.verified")
        root = permission["root_binding"]
        native_context = ({"runtime_sources":sources, "runtime_environment":fresh_environment}
                          if profile == storage.SCIENTIFIC else {})
        store = storage.ArtifactStore.reopen(
            root["path"], expected_profile=profile,
            expected_header_sha256=root["header_sha256"],
            expected_root_identity=(root["device"], root["inode"]), **native_context)
        guard.check("transition.writer.acquired")
        records = _load_records(store, storage, control, codec, policy,
                                PREVIOUS[phase][1], evidence_kind,
                                permission=permission, sources=sources,
                                environment_schema=environment_schema)
        guard.check("transition.prior.verified")
        environment_raw = codec.json_bytes(fresh_environment)
        store.write_bytes(permission_name(phase), raw)
        guard.check("transition.permission.copied")
        store.write_bytes(environment_name(phase), environment_raw)
        guard.check("transition.environment.copied")
        records = _load_records(store, storage, control, codec, policy,
                                PREVIOUS[phase][1], evidence_kind,
                                permission=permission, include_current_metadata=True,
                                sources=sources, environment_schema=environment_schema)
        original_environment = fresh_environment
        if phase == "audit":
            meta = store._indexed.get(environment_name("primary"))
            _need(meta is not None, "audit requires retained primary source environment")
            source_raw, _ = storage._read_regular(
                store._dirfd, environment_name("primary"), maximum=environment_schema.METADATA_CAP)
            original_environment = codec.json_loads(
                source_raw, max_bytes=environment_schema.METADATA_CAP)
            environment_schema.validate_environment(
                original_environment, profile=environment_schema.SCIENTIFIC)
            _need(original_environment["runtime_role"] == "native_source",
                  "audit source environment role differs")
        if phase == "sensitivity":
            meta = store._indexed.get(environment_name("primary"))
            _need(meta is not None, "sensitivity requires retained primary environment")
            primary_raw, _ = storage._read_regular(
                store._dirfd, environment_name("primary"), maximum=environment_schema.METADATA_CAP)
            _need(primary_raw == environment_raw,
                  "sensitivity environment differs from primary")
        guard.check("transition.return")
        return {"store":store, "guard":guard, "phase":phase,
            "permission":permission, "permission_pin":dict(permission_pin),
            "transition_pin":transition_pin, "records":records,
            "evidence_kind":evidence_kind, "sources":sources,
            "environment":fresh_environment, "source_environment":original_environment,
            "root_binding":dict(permission["root_binding"]),
            "journal_pin":dict(permission["journal_pin"]),
            "previous_boundary_ref":dict(permission["previous_boundary_ref"]),
            "execution_authorized":False, "scientific_execution_certified":False}
    except BaseException as exc:
        interrupted = isinstance(exc, (KeyboardInterrupt, SystemExit))
        if transition_pin is None and marker_path.exists():
            failure_status = "transition_marker_unverified_or_partial"
        if store is not None:
            try:
                if not store._terminal:
                    store._fail("phase_transition_failed", "phase_transition_failed")
                retained = (store.failure_metadata_status == "retained"
                            and any(name.startswith("failure-")
                                    for name in store._indexed))
                failure_status = ("transition_marker_and_store_failure_retained"
                    if retained else
                    "transition_marker_retained_store_failure_unavailable")
            except BaseException:
                failure_status = "transition_marker_retained_store_failure_unavailable"
            try:
                store.close()
            except BaseException:
                pass
        if interrupted:
            raise
        if isinstance(exc, TransitionError):
            raise TransitionError(str(exc), failure_status=failure_status) from None
        raise TransitionError("phase transition failed", failure_status=failure_status) from None


def validate_acquired_phase(result):
    """Revalidate a trusted in-memory acquisition while its writer lock is held."""
    _keys(result, ACQUIRED_FIELDS, "acquired phase")
    phase = result["phase"]
    _need(phase in PHASES and result["execution_authorized"] is False
          and result["scientific_execution_certified"] is False,
          "acquired phase authority differs")
    permission_pin = result["permission_pin"]
    raw = _read_pinned(permission_pin, PERMISSION_MAX)
    permission = _permission(
        raw, permission_pin, phase,
        fixture=(result["evidence_kind"] == "synthetic_contract_fixture"))
    _need(permission == result["permission"]
          and result["root_binding"] == permission["root_binding"]
          and result["journal_pin"] == permission["journal_pin"]
          and result["previous_boundary_ref"] == permission["previous_boundary_ref"],
          "acquired phase bindings differ")
    marker_raw = _read_pinned(result["transition_pin"], TRANSITION_MAX)
    marker = _marker(_json_loads(marker_raw, TRANSITION_MAX), phase)
    _need(result["transition_pin"]["path"]
              == str(Path(permission["journal_pin"]["path"]).parent / transition_name(phase))
          and marker["permission_pin"] == permission_pin
          and marker["root_binding"] == permission["root_binding"]
          and marker["journal_pin"] == permission["journal_pin"]
          and marker["previous_boundary_ref"] == permission["previous_boundary_ref"],
          "acquired transition marker differs")

    import artifact_store as storage
    import identity_codec as codec
    import native_control as control
    import native_phase_policy as policy
    import source_environment_schema as environment_schema
    store = result["store"]
    _need(type(store) is storage.ArtifactStore and not store._closed
          and not store._terminal
          and str(store.root) == permission["root_binding"]["path"]
          and store.root_identity
              == (permission["root_binding"]["device"], permission["root_binding"]["inode"])
          and store.header_sha256 == permission["root_binding"]["header_sha256"],
          "acquired writer/root differs")
    records = _load_records(store, storage, control, codec, policy,
                            PREVIOUS[phase][1], result["evidence_kind"],
                            permission=permission, include_current_metadata=True,
                            sources=result["sources"],
                            environment_schema=environment_schema)
    _need(records == result["records"], "acquired policy prefix differs")
    stored_permission, _ = storage._read_regular(
        store._dirfd, permission_name(phase), maximum=PERMISSION_MAX)
    stored_environment, _ = storage._read_regular(
        store._dirfd, environment_name(phase), maximum=1 << 20)
    _need(stored_permission == raw
          and stored_environment == codec.json_bytes(result["environment"])
          and codec.tree_digest(result["environment"])
              == permission["expected_environment_sha256"]
          and codec.tree_digest(result["sources"])
              == permission["expected_source_set_sha256"],
          "acquired source/environment bytes differ")
    return result


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    _need(type(args) is list and not args, "phase transition has no launch CLI")
    print('{"status":"inert"}')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
