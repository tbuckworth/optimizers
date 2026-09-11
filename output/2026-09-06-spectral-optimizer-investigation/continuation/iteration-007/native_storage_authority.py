"""Bounded I7 review authority, not proof of historical execution.

Import/default CLI are inert and Torch-free. The externally reviewed exact
permission channel is the trust anchor. Hashes bind bytes; a caller who can
fabricate both evidence and that trusted permission is outside this model.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat

EVIDENCE_MAX = 64 << 10
PERMISSION_MAX = 8 << 10
MIN_FREE_BYTES = 1 << 30
MEASUREMENT_SCHEMA = "i7_native_storage_measurement_v3"
MEASUREMENT_PROTOCOL = "i7_complete_structural_recipe_v2"
MEASUREMENT_ROLE = "nonsemantic_cpu_serializer_crosscheck"
ADMISSION_SCHEMA = "i7_native_storage_admission_v1"
MEASUREMENT_FIELDS = (
    "schema", "status", "evidence_role", "attempt_id", "repository_revision",
    "source_set_sha256", "diagnostic_environment", "measurement_protocol", "component_rows",
    "serializer_runtime", "resource_limits", "observed_resources", "supervision", "failure",
    "execution_authorized", "scientific_execution_certified")
COMPONENT_FIELDS = ("component", "encoding", "payload_count",
    "analytic_pickle_bytes_upper", "analytic_body_bytes_upper", "status", "observation")
OBSERVATION_FIELDS = ("pickle_bytes", "body_bytes", "storage_count", "raw_storage_bytes",
    "serialized_sha256", "restricted_cpu_roundtrip", "exact_tree_roundtrip")
SERIALIZER_FIELDS = ("zip_runtime", "pickle_runtime_checked", "compiled_binary_provenance_attested")
RESOURCE_LIMITS = {"wall_ns": 120_000_000_000, "rss_bytes": 2 << 30, "buffer_bytes": 64 << 20}
RESOURCE_FIELDS = ("elapsed_wall_ns", "elapsed_cpu_ns", "peak_rss_bytes", "peak_buffer_bytes",
    "runner_pid", "process_scope", "cuda_visible_devices", "cuda_initialized_before",
    "cuda_initialized_after", "thread_settings", "rng_preserved")
THREAD_SETTINGS = {key: "1" for key in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")}
PIN_FIELDS = ("path", "size_bytes", "sha256")
ADMISSION_FIELDS = ("schema", "status", "attempt_id", "expected_commit",
    "expected_source_set_sha256", "root_budget_bytes", "shared_failure_reserve_bytes",
    "supervision_total_bytes", "layout_evidence_pin", "decision", "retry_allowed",
    "execution_authorized", "scientific_execution_certified")
_SHA = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)
_COMMIT = re.compile(r"[0-9a-f]{40}\Z", re.ASCII)


class StorageAuthorityError(ValueError):
    pass


def _need(condition, message):
    if not condition:
        raise StorageAuthorityError(message)


def _keys(value, expected, label):
    _need(type(value) is dict and tuple(value) == tuple(expected)
          and all(type(k) is str for k in value), label + ": exact ordered fields required")


def _integer(value, label, *, minimum=0, maximum=(1 << 64) - 1):
    _need(type(value) is int and minimum <= value <= maximum, label + ": invalid integer")


def _digest(value, pattern, label):
    _need(type(value) is str and pattern.fullmatch(value) is not None, label + ": invalid digest")


def encode_bounded(value, *, maximum=EVIDENCE_MAX):
    """Bound traversal/strings before encoding; no unbounded dumps allocation."""
    _integer(maximum, "encoding limit", minimum=1, maximum=EVIDENCE_MAX)
    remaining = [maximum, maximum]
    active = set()
    def visit(item, depth):
        remaining[0] -= 1
        _need(depth <= 32 and remaining[0] >= 0, "JSON traversal limit")
        if type(item) is str:
            remaining[1] -= len(item)
            _need(remaining[1] >= 0 and not any(0xD800 <= ord(c) <= 0xDFFF for c in item),
                  "JSON string limit or surrogate")
        elif item is None or type(item) is bool:
            pass
        elif type(item) is int:
            _integer(item, "JSON integer", minimum=-(1 << 64))
        elif type(item) is float:
            _need(math.isfinite(item), "nonfinite JSON number")
        else:
            _need(type(item) in (dict, list) and id(item) not in active, "invalid/cyclic JSON")
            _need(len(item) <= maximum, "JSON container limit")
            active.add(id(item))
            try:
                if type(item) is dict:
                    for key, child in item.items():
                        _need(type(key) is str, "JSON key must be string")
                        visit(key, depth + 1)
                        visit(child, depth + 1)
                else:
                    for child in item:
                        visit(child, depth + 1)
            finally:
                active.remove(id(item))
    visit(value, 0)
    chunks, size = [], 1
    for chunk in json.JSONEncoder(ensure_ascii=True, allow_nan=False,
                                   separators=(",", ":")).iterencode(value):
        encoded = chunk.encode("ascii")
        size += len(encoded)
        _need(size <= maximum, "canonical JSON exceeds byte ceiling")
        chunks.append(encoded)
    return b"".join(chunks) + b"\n"


def decode_bounded(raw, *, maximum=EVIDENCE_MAX):
    _need(type(raw) is bytes and 0 < len(raw) <= maximum, "bounded JSON bytes required")
    def pairs(rows):
        result = {}
        for key, value in rows:
            _need(key not in result, "duplicate JSON key")
            result[key] = value
        return result
    def constant(_):
        raise StorageAuthorityError("nonfinite JSON constant")
    try:
        value = json.loads(raw.decode("ascii"), object_pairs_hook=pairs, parse_constant=constant)
        _need(encode_bounded(value, maximum=maximum) == raw, "noncanonical JSON")
        return value
    except (ValueError, UnicodeError, RecursionError) as exc:
        if isinstance(exc, StorageAuthorityError):
            raise
        raise StorageAuthorityError("malformed bounded JSON") from exc


def _pin(pin, *, path=None):
    _keys(pin, PIN_FIELDS, "pin")
    p = pin["path"]
    _need(type(p) is str and p.isascii() and "\x00" not in p and 0 < len(p) <= 4096 and p.startswith("/")
          and os.path.normpath(p) == p and not p.startswith("//"), "noncanonical pin path")
    if path is not None:
        _need(p == path, "fixed evidence path differs")
    _integer(pin["size_bytes"], "pin size", minimum=1, maximum=EVIDENCE_MAX)
    _digest(pin["sha256"], _SHA, "pin")
    return dict(pin)


def validate_storage_admission_pin(pin, *, fixture=False):
    import process_supervision as supervision
    _need(type(fixture) is bool, "fixture flag must be boolean")
    return _pin(pin, path=None if fixture else supervision.STORAGE_ADMISSION_PATH)


def validate_measurement(value, *, expected_commit, expected_source_set_sha256,
                         require_complete=True, recompute=False):
    return _validate_measurement(value, expected_commit=expected_commit,
        expected_source_set_sha256=expected_source_set_sha256,
        require_complete=require_complete, recompute=recompute, candidate=False)


def validate_measurement_candidate(value, *, expected_commit, expected_source_set_sha256):
    """Worker output only, NOT finalized/admissible evidence; always recompute."""
    return _validate_measurement(value, expected_commit=expected_commit,
        expected_source_set_sha256=expected_source_set_sha256,
        require_complete=False, recompute=True, candidate=True)


def _validate_measurement(value, *, expected_commit, expected_source_set_sha256,
                          require_complete, recompute, candidate):
    """Validate supplied facts; recompute=True adds current analytic ceilings.

    Neither mode attests execution. No Torch import occurs in this decoder.
    A complete report must describe the full structural recipe, not a mere
    tensor inventory or scientifically valid envelope/native layout observation.
    """
    import native_tensor_inventory as inventory
    import process_supervision as supervision
    import zip_storage_bound as zip_bound
    _need(type(require_complete) is bool and type(recompute) is bool, "invalid decoder mode")
    encode_bounded(value)
    _digest(expected_commit, _COMMIT, "expected commit")
    _digest(expected_source_set_sha256, _SHA, "expected sources")
    _keys(value, MEASUREMENT_FIELDS, "measurement")
    _need(value["schema"] == MEASUREMENT_SCHEMA and value["status"] in ("complete", "failed")
          and value["evidence_role"] == MEASUREMENT_ROLE
          and value["attempt_id"] == supervision.ATTEMPT_ID
          and value["repository_revision"] == expected_commit
          and value["source_set_sha256"] == expected_source_set_sha256
          and value["measurement_protocol"] == MEASUREMENT_PROTOCOL
          and value["execution_authorized"] is False
          and value["scientific_execution_certified"] is False, "measurement binding differs")
    complete = value["status"] == "complete"
    _need(not require_complete or complete, "incomplete measurement is not admission evidence")
    environment = value["diagnostic_environment"]
    if environment is None:
        _need(not complete and type(value["failure"]) is dict
              and value["failure"].get("stage") in ("source", "runtime"),
              "complete/post-runtime measurement requires diagnostic environment")
    else:
        import source_environment_schema as environment_schema
        encode_bounded(environment, maximum=8192)
        try:
            environment_schema.validate_environment(environment, profile=environment_schema.SCIENTIFIC)
        except environment_schema.SourceEnvironmentError as exc:
            raise StorageAuthorityError("diagnostic environment schema rejected") from exc
        _need(environment["runtime_role"] == "storage_crosscheck_cpu",
              "measurement requires diagnostic CPU environment role")
    components = value["component_rows"]
    _need(type(components) is list and len(components) == len(inventory.COMPONENT_COUNTS) == 11,
          "exact eleven component rows required")
    bounds = layouts = None
    if recompute:
        import native_storage_topology_bound as topology
        bounds = topology.compute()["components"]
        layouts = inventory.component_layouts()
    ended = False
    failed_components = []
    for row, (name, count) in zip(components, inventory.COMPONENT_COUNTS.items()):
        _keys(row, COMPONENT_FIELDS, "component")
        is_json = name == "capture_comparison_pilot"
        _need(row["component"] == name and row["encoding"] ==
              ("bytes" if is_json else "torch_weights_only"), "component order/encoding differs")
        _integer(row["payload_count"], "payload count", minimum=1)
        _need(row["payload_count"] == count, "component count differs")
        _integer(row["analytic_body_bytes_upper"], "analytic body", minimum=1, maximum=64 << 20)
        if is_json:
            _need(row["analytic_pickle_bytes_upper"] is None, "JSON has no pickle")
        else:
            _integer(row["analytic_pickle_bytes_upper"], "analytic pickle", minimum=1, maximum=64 << 20)
        if recompute:
            expected = bounds[name]
            _need(row["analytic_body_bytes_upper"] == expected["body_bytes_upper"]
                  and row["analytic_pickle_bytes_upper"] == expected.get("pickle_bytes_upper"),
                  "claimed analytic ceiling differs from current calculation")
        status = row["status"]
        _need(status in ("measured", "failed", "not_run"), "component status differs")
        _need(environment is not None or status == "not_run",
              "missing diagnostic environment requires all components not_run")
        observation = row["observation"]
        if status != "measured":
            _need(not complete and observation is None, "unmeasured row in complete report")
            if status == "failed":
                _need(not ended, "multiple/out-of-order failed components")
                failed_components.append(name)
            ended = True
            continue
        _need(not ended, "measured row after stopped component")
        _keys(observation, OBSERVATION_FIELDS, "observation")
        _integer(observation["body_bytes"], "observed body", minimum=1,
                 maximum=row["analytic_body_bytes_upper"])
        _integer(observation["storage_count"], "observed storages")
        _integer(observation["raw_storage_bytes"], "observed raw bytes")
        _digest(observation["serialized_sha256"], _SHA, "serialized body")
        _need(observation["restricted_cpu_roundtrip"] is True
              and observation["exact_tree_roundtrip"] is True, "roundtrip not verified")
        if is_json:
            _need(observation["pickle_bytes"] is None and observation["storage_count"] == 0
                  and observation["raw_storage_bytes"] == 0, "JSON observation has tensor storage")
        else:
            _integer(observation["pickle_bytes"], "observed pickle", minimum=1,
                     maximum=row["analytic_pickle_bytes_upper"])
            _need(observation["pickle_bytes"] + observation["raw_storage_bytes"]
                  <= observation["body_bytes"], "archive cannot contain reported data")
        if recompute:
            _need(observation["storage_count"] == len(layouts[name])
                  and observation["raw_storage_bytes"] == sum(r["nbytes"] for r in layouts[name]),
                  "maximal structural recipe inventory differs")
    serializer = value["serializer_runtime"]
    _keys(serializer, SERIALIZER_FIELDS, "serializer")
    if serializer["zip_runtime"] is not None:
        try:
            zip_bound.validate_runtime_record(serializer["zip_runtime"])
        except zip_bound.ZipBoundError as exc:
            raise StorageAuthorityError("invalid ZIP runtime record") from exc
        _need(serializer["zip_runtime"]["require_cuda_uninitialized"] is True
              and serializer["zip_runtime"]["cuda_initialized"] is False,
              "measurement ZIP runtime was not CPU-only")
    _need(type(serializer["pickle_runtime_checked"]) is bool
          and serializer["compiled_binary_provenance_attested"] is False,
          "invalid serializer flags")
    _need(not complete or (serializer["zip_runtime"] is not None
                          and serializer["pickle_runtime_checked"] is True), "missing runtime checks")
    limits, resources = value["resource_limits"], value["observed_resources"]
    _keys(limits, RESOURCE_LIMITS, "resource limits")
    for key, limit in RESOURCE_LIMITS.items():
        _integer(limits[key], key, minimum=1)
        _need(limits[key] == limit, "resource limit differs")
    _keys(resources, RESOURCE_FIELDS, "observed resources")
    for key in RESOURCE_FIELDS[:5]:
        _integer(resources[key], key, minimum=1 if key == "runner_pid" else 0)
    _need(resources["process_scope"] == "runner_self" and
          type(resources["cuda_visible_devices"]) is str and len(resources["cuda_visible_devices"]) <= 256,
          "resource observation scope differs")
    for key in ("cuda_initialized_before", "cuda_initialized_after", "rng_preserved"):
        _need(type(resources[key]) is bool, "resource flags must be booleans")
    _keys(resources["thread_settings"], THREAD_SETTINGS, "thread settings")
    _need(all(type(v) is str and len(v) <= 32 for v in resources["thread_settings"].values()),
          "invalid thread settings")
    if complete:
        maximum_observed_body = max(row["observation"]["body_bytes"] for row in components)
        _need(resources["elapsed_wall_ns"] <= limits["wall_ns"]
              and 0 < resources["peak_rss_bytes"] <= limits["rss_bytes"]
              and maximum_observed_body <= resources["peak_buffer_bytes"] <= limits["buffer_bytes"]
              and resources["cuda_visible_devices"] == ""
              and resources["cuda_initialized_before"] is False
              and resources["cuda_initialized_after"] is False
              and resources["thread_settings"] == THREAD_SETTINGS
              and resources["rng_preserved"] is True, "measurement resource conditions failed")
        _need(value["failure"] is None, "complete report cannot have failure")
    else:
        failure = value["failure"]
        _keys(failure, ("stage", "component", "reason"), "failure")
        _need(failure["stage"] in ("source", "runtime", "recipe", "serialize", "roundtrip", "resource", "report")
              and failure["reason"] in ("contract_failure", "resource_limit"), "failure category differs")
        _need(failure["component"] is None or (type(failure["component"]) is str
              and failure["component"] in inventory.COMPONENT_COUNTS),
              "failure component differs")
        _need(failed_components == ([] if failure["component"] is None else [failure["component"]]),
              "failure does not match component rows")
    if candidate:
        _need(value["supervision"] is None, "candidate cannot claim observed supervision")
    else:
        import native_storage_measurement_service as service
        unsigned = dict(value)
        unsigned["supervision"] = None
        service.validate_record(value["supervision"], unsigned)
    return value


def validate_structural_admission(raw, pin, *, expected_commit, expected_source_set_sha256):
    """Torch-free pinned byte/schema checks, not runtime/storage attestation."""
    import process_supervision as supervision
    validate_storage_admission_pin(pin)
    _need(type(raw) is bytes and len(raw) == pin["size_bytes"]
          and hashlib.sha256(raw).hexdigest() == pin["sha256"], "admission bytes differ from pin")
    value = decode_bounded(raw)
    _keys(value, ADMISSION_FIELDS, "admission")
    _digest(expected_commit, _COMMIT, "expected commit")
    _digest(expected_source_set_sha256, _SHA, "expected sources")
    expected = make_admission(value["layout_evidence_pin"], expected_commit=expected_commit,
                              expected_source_set_sha256=expected_source_set_sha256)
    _need(value == expected, "admission review binding/decision differs")
    for key in ("root_budget_bytes", "shared_failure_reserve_bytes", "supervision_total_bytes"):
        _integer(value[key], key, minimum=1)
    for key in ("retry_allowed", "execution_authorized", "scientific_execution_certified"):
        _need(value[key] is False, "admission flag differs")
    mraw = supervision.load_pinned_bytes(**value["layout_evidence_pin"], maximum=EVIDENCE_MAX)
    validate_measurement(decode_bounded(mraw), expected_commit=expected_commit,
                         expected_source_set_sha256=expected_source_set_sha256)
    return value


def make_admission(layout_evidence_pin, *, expected_commit, expected_source_set_sha256):
    """Construct a review candidate only; does not read/write or approve evidence."""
    import process_supervision as supervision
    _digest(expected_commit, _COMMIT, "expected commit")
    _digest(expected_source_set_sha256, _SHA, "expected sources")
    pin = _pin(layout_evidence_pin, path=supervision.LAYOUT_EVIDENCE_PATH)
    return dict(schema=ADMISSION_SCHEMA, status="admitted", attempt_id=supervision.ATTEMPT_ID,
        expected_commit=expected_commit, expected_source_set_sha256=expected_source_set_sha256,
        root_budget_bytes=1 << 30, shared_failure_reserve_bytes=1 << 20,
        supervision_total_bytes=96 << 10, layout_evidence_pin=pin,
        decision="storage_fit_under_reviewed_bound", retry_allowed=False,
        execution_authorized=False, scientific_execution_certified=False)


def validate_runtime_admission(pin, *, sources, environment):
    """Consumed worker boundary. Caller supplies freshly collected metadata.

    The source-bound final_storage_accounting adapter delegates here, but no
    admission is possible without the separately reviewed complete M/A chain.
    Permission/request/journal equality must already be authenticated by the
    controller. This rereads evidence and recomputes the current analytic and
    installed serializer conditions; it does not certify M's historical run.
    """
    import identity_codec as codec
    import native_payload_guard as guard
    import native_write_ledger as ledger
    import pickle_storage_bound as pickle_bound
    import process_supervision as supervision
    import zip_storage_bound as zip_bound
    guard.validate_runtime_metadata(sources, environment)
    commit, digest = sources["repository_revision"], codec.tree_digest(sources)
    validate_storage_admission_pin(pin)
    raw = supervision.load_pinned_bytes(**pin, maximum=EVIDENCE_MAX)
    admission = validate_structural_admission(raw, pin, expected_commit=commit,
                                            expected_source_set_sha256=digest)
    mraw = supervision.load_pinned_bytes(**admission["layout_evidence_pin"], maximum=EVIDENCE_MAX)
    measurement = validate_measurement(decode_bounded(mraw), expected_commit=commit,
                                      expected_source_set_sha256=digest, recompute=True)
    _need(measurement["diagnostic_environment"]["repository_root_realpath"]
          == sources["repository_root_realpath"], "measurement/source repository roots differ")
    calculation = ledger.compute()
    _need(calculation["conditional_arithmetic_fits"] is True
          and calculation["logical_bytes"]["complete_success_plus_failure_reserve_upper"]
              <= admission["root_budget_bytes"], "current complete write ledger does not fit")
    pickle_bound.assert_pinned_pickle_runtime()
    zip_bound.validate_runtime_save_configuration(
        require_cuda_uninitialized=environment["runtime_role"] == "cpu_audit")


def _open_parent(path):
    _need(type(path) is str and path.startswith("/") and os.path.normpath(path) == path
          and not path.startswith("//"), "canonical absolute write path required")
    parts = path.split("/")[1:]
    _need(bool(parts) and all(parts), "invalid write path")
    fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for part in parts[:-1]:
            next_fd = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = next_fd
        return fd, parts[-1]
    except BaseException:
        os.close(fd)
        raise


def _write_exclusive(path, raw, *, maximum):
    """Single fixed slot. Preserve any partial/zero file on all failures.

    Internal primitive, not an arbitrary authorized writer. Public entrypoints
    resolve/validate exact targets and schemas before calling it.
    """
    import process_supervision as supervision
    _need(type(raw) is bytes and 0 < len(raw) <= maximum <= EVIDENCE_MAX, "write exceeds slot")
    parent, name = _open_parent(path)
    fd = None
    try:
        space = os.fstatvfs(parent)
        _need(space.f_bavail * space.f_frsize >= MIN_FREE_BYTES + len(raw), "free-space floor")
        fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                     0o600, dir_fd=parent)
        info = os.fstat(fd)
        _need(stat.S_ISREG(info.st_mode) and info.st_nlink == 1, "new slot is not private regular file")
        offset = 0
        while offset < len(raw):
            written = os.write(fd, memoryview(raw)[offset:])
            _need(type(written) is int and 0 < written <= len(raw) - offset, "short/invalid slot write")
            offset += written
        os.fsync(fd)
        os.fsync(parent)
        final = os.fstat(fd)
        current = os.stat(name, dir_fd=parent, follow_symlinks=False)
        _need(final.st_size == len(raw) and final.st_nlink == 1
              and (final.st_dev, final.st_ino, final.st_mode) == (info.st_dev, info.st_ino, info.st_mode)
              and (current.st_dev, current.st_ino, current.st_size, current.st_mode)
              == (final.st_dev, final.st_ino, final.st_size, final.st_mode), "written slot identity changed")
        pin = dict(path=path, size_bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
        _need(supervision.load_pinned_bytes(**pin, maximum=maximum) == raw, "written slot bytes differ")
        return pin
    finally:
        # A failed create may be an already-consumed slot. Never unlink it.
        try:
            os.fsync(parent)
        finally:
            if fd is not None:
                os.close(fd)
            os.close(parent)


def _require_absent(path):
    """Any entry, including a symlink or zero/partial file, consumes the slot."""
    parent, name = _open_parent(path)
    try:
        try:
            os.stat(name, dir_fd=parent, follow_symlinks=False)
        except FileNotFoundError:
            return
        raise StorageAuthorityError("fixed slot already consumed")
    finally:
        os.close(parent)


def write_measurement(value, *, expected_commit, expected_source_set_sha256):
    """Write only M's precharged slot, complete or structured failure; never A."""
    import process_supervision as supervision
    # Reviewed serial workflow: M must precede A. This is not a lock against
    # hostile same-user concurrent review/write processes (outside trust model).
    _require_absent(supervision.STORAGE_ADMISSION_PATH)
    _require_absent(supervision.LAYOUT_EVIDENCE_PATH)
    validate_measurement(value, expected_commit=expected_commit,
                         expected_source_set_sha256=expected_source_set_sha256,
                         require_complete=False, recompute=True)
    return _write_exclusive(supervision.LAYOUT_EVIDENCE_PATH, encode_bounded(value), maximum=EVIDENCE_MAX)


def write_reviewed_admission(value):
    """Explicit reviewer action. Runner must never call this creator."""
    import native_write_ledger as ledger
    import process_supervision as supervision
    _require_absent(supervision.STORAGE_ADMISSION_PATH)
    raw = encode_bounded(value)
    _keys(value, ADMISSION_FIELDS, "admission candidate")
    candidate_pin = dict(path=supervision.STORAGE_ADMISSION_PATH, size_bytes=len(raw),
                         sha256=hashlib.sha256(raw).hexdigest())
    validate_structural_admission(raw, candidate_pin, expected_commit=value["expected_commit"],
                                 expected_source_set_sha256=value["expected_source_set_sha256"])
    mraw = supervision.load_pinned_bytes(**value["layout_evidence_pin"], maximum=EVIDENCE_MAX)
    validate_measurement(decode_bounded(mraw), expected_commit=value["expected_commit"],
                         expected_source_set_sha256=value["expected_source_set_sha256"], recompute=True)
    _need(ledger.compute()["conditional_arithmetic_fits"] is True, "complete write ledger does not fit")
    return _write_exclusive(supervision.STORAGE_ADMISSION_PATH, raw, maximum=EVIDENCE_MAX)


def write_phase_permission(value, *, prior_boundary):
    """Derive A from the sealed journal and write only the fixed later slot.

    Production-only entry; no fixture path override. It does not acquire a
    phase or create a transition/GO. Partial or post-write-invalid files remain.
    """
    import phase_transition as transition
    import native_control as control
    import process_supervision as supervision
    raw, permission = transition.authenticate_permission_for_write(
        value, prior_boundary=prior_boundary, attempt_parent=control.DEVELOPMENT_ATTEMPT_PATH,
        fixture=False)
    _need(encode_bounded(permission, maximum=PERMISSION_MAX) == raw,
          "permission canonical bytes differ")
    admission_pin = permission["storage_admission_pin"]
    admission_raw = supervision.load_pinned_bytes(**admission_pin, maximum=EVIDENCE_MAX)
    validate_structural_admission(admission_raw, admission_pin,
        expected_commit=permission["expected_commit"],
        expected_source_set_sha256=permission["expected_source_set_sha256"])
    path = str(control.DEVELOPMENT_ATTEMPT_PATH) + "/" + transition.permission_name(permission["phase"])
    pin = _write_exclusive(path, raw, maximum=PERMISSION_MAX)
    evidence_kind = "native_producer_attestation"
    actual_raw, actual = transition._authenticate_actual(
        pin, permission["phase"], prior_boundary, evidence_kind)
    _need(actual_raw == raw and actual == permission, "post-write permission authentication differs")
    return pin


def main(argv=None):
    print("native_storage_authority: inert; no measurement, write or execution action")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
