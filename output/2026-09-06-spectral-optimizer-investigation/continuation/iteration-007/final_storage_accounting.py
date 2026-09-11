"""Pure final-file accounting for the prospective I7 native attempt.

Import and default CLI execution are stdlib-only and perform no filesystem I/O.
Explicit calls lazily inspect the fixed policy/schema modules.  This module does
not measure native output and cannot turn the existing CPU component projection
into a native storage certificate.
"""
from __future__ import annotations

import re
import sys


class FinalStorageAccountingError(ValueError):
    pass


class StorageAdmissionError(RuntimeError):
    pass


SCHEMA = "i7_final_storage_accounting_v1"
STORE_BUDGET_BYTES = 1 << 30
SHARED_FAILURE_RESERVE_BYTES = 1 << 20
PHASE_RECORD_COUNT = 90
PHASE_RECORD_BODY_CEILING_BYTES = 128 << 10
SCIENTIFIC_PAYLOAD_COUNT = 82
ROOT_METADATA_COUNT = 9
ROOT_BODY_COUNT = 181
ROOT_REGULAR_FILE_COUNT = 364
SOURCE_METADATA_BODY_CEILING_BYTES = 1 << 20
CONTROL_BODY_CEILING_BYTES = 8 << 10
EXISTING_INSPECTION_LOGICAL_BYTES = 1270
SUPERVISION_TOTAL_CEILING_BYTES = 96 << 10
STORAGE_EVIDENCE_BODY_CEILING_BYTES = 64 << 10
BASE_SOURCE_MEMBER_COUNT = 55

NATIVE_DELTA_EVIDENCE_KEYS = ("kind", "sha256")
NATIVE_DELTA_EVIDENCE_KIND = "referenced_native_payload_delta_upper_bound_v1"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)

ROOT_METADATA_BODY_CEILINGS = (
    ("native-launch-journal.json", CONTROL_BODY_CEILING_BYTES),
    ("native-sources.json", SOURCE_METADATA_BODY_CEILING_BYTES),
    ("native-development-environment.json", SOURCE_METADATA_BODY_CEILING_BYTES),
    ("native-primary-permission.json", CONTROL_BODY_CEILING_BYTES),
    ("native-primary-environment.json", SOURCE_METADATA_BODY_CEILING_BYTES),
    ("native-sensitivity-permission.json", CONTROL_BODY_CEILING_BYTES),
    ("native-sensitivity-environment.json", SOURCE_METADATA_BODY_CEILING_BYTES),
    ("native-audit-permission.json", CONTROL_BODY_CEILING_BYTES),
    ("native-audit-environment.json", SOURCE_METADATA_BODY_CEILING_BYTES),
)

DEVELOPMENT_EXTERNAL_SUCCESS_CEILINGS = (
    ("native-launch-journal.json", 8 << 10),
    ("native-primary-permission.json", 8 << 10),
    ("native-primary-transition.json", 8 << 10),
    ("native-sensitivity-permission.json", 8 << 10),
    ("native-sensitivity-transition.json", 8 << 10),
    ("native-audit-permission.json", 8 << 10),
    ("native-audit-transition.json", 8 << 10),
)

SUPERVISION_CEILINGS = tuple(
    (f"native-{phase}-process-{kind}.json", ceiling)
    for phase in ("development", "primary", "sensitivity", "audit")
    for kind, ceiling in (("request", 8 << 10), ("exit", 16 << 10))
)

EXISTING_INSPECTION_FILES = (
    ("native-layout-inspection-001-marker.json", 676),
    ("native-layout-inspection-001-failure.json", 594),
)

# The failed first serializer check permanently retained this zero-byte file.
# It is membership evidence, but contributes zero logical bytes and is not a
# prerequisite for the separately named replacement check.
EXISTING_STORAGE_ATTEMPT_FILES = (
    ("native-max-layout-measurement.json", 0),
)

STORAGE_EVIDENCE_FILES = (
    ("native-storage-admission.json", STORAGE_EVIDENCE_BODY_CEILING_BYTES),
    ("native-max-layout-measurement-attempt-002.json", STORAGE_EVIDENCE_BODY_CEILING_BYTES),
)

REQUIRED_SOURCE_ADDITIONS = (
    "phase_worker.py",
    "process_supervision.py",
    "final_storage_accounting.py",
    "native_layout_inspection.py",
    "pickle_storage_bound.py",
    "zip_storage_bound.py",
    "native_tensor_inventory.py",
    "comparison_storage_bound.py",
    "primitive_storage_bound.py",
    "core_primitive_bound.py",
    "branch_primitive_bound.py",
    "audit_primitive_bound.py",
    "native_storage_topology_bound.py",
    "native_payload_guard.py",
    "native_write_ledger.py",
    "root_storage_accounting.py",
    "native_storage_authority.py",
    "native_storage_recipe_common.py",
    "native_storage_recipe_core.py",
    "native_storage_recipe_branch_audit.py",
    "native_storage_crosscheck.py",
    "native_storage_measurement_service.py",
    "native_storage_measurement_controller.py",
    "native_storage_measurement_slot.py",
    "native_storage_measurement_worker.py",
)


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise FinalStorageAccountingError(message)


def validate_runtime_admission(pin, *, sources, environment) -> None:
    """Fail closed unless the strict reviewed evidence/runtime chain validates.

    This source-bound adapter must be frozen before M is measured. Connecting
    it is not a GO: the external M/A and separately reviewed phase permission
    are still required. No supplied success flag is accepted as a substitute.
    """
    if not (type(pin) is dict and tuple(pin) == ("path", "size_bytes", "sha256")
            and type(pin["path"]) is str and bool(pin["path"])
            and type(pin["size_bytes"]) is int and 0 < pin["size_bytes"] <= 64 << 10
            and type(pin["sha256"]) is str and _SHA256.fullmatch(pin["sha256"]) is not None
            and type(sources) is dict and type(environment) is dict):
        raise StorageAdmissionError("invalid storage admission input")
    try:
        import native_storage_authority as authority
        result = authority.validate_runtime_admission(pin, sources=sources, environment=environment)
        if result is not None:
            raise StorageAdmissionError("unexpected authority return")
    except Exception:
        # Fixed category only; unexpected runtime/I/O failures cannot grant GO.
        # BaseException (including user interruption) deliberately propagates.
        raise StorageAdmissionError("reviewed storage admission rejected") from None


def _modules():
    """Explicit accounting boundary; may import the CPU Torch storage module."""
    import native_control as control
    import native_phase_policy as policy
    import phase_transition as transition
    import process_supervision as supervision
    import root_storage_accounting as prior
    import source_environment_schema as source_schema
    return control, policy, transition, supervision, prior, source_schema


def _sum_ceilings(rows: tuple[tuple[str, int], ...]) -> int:
    _need(all(type(row) is tuple and len(row) == 2 and type(row[0]) is str
              and type(row[1]) is int and row[1] > 0 for row in rows),
          "malformed file ceiling definition")
    _need(len({name for name, _ in rows}) == len(rows),
          "duplicate file ceiling definition")
    return sum(size for _, size in rows)


def source_membership_status() -> dict:
    """Report declared schema membership; this does not collect/freeze sources."""
    _, _, _, _, _, source_schema = _modules()
    rows = source_schema.SCIENTIFIC_ROLES
    _need(type(rows) is tuple and all(type(row) is tuple and len(row) == 2
                                     and all(type(item) is str for item in row)
                                     for row in rows),
          "scientific source membership schema differs")
    paths = tuple(path for _, path in rows)
    _need(len(paths) == len(set(paths)), "scientific source membership has duplicates")
    prefix = source_schema.I7
    required = tuple(prefix + name for name in REQUIRED_SOURCE_ADDITIONS)
    missing = tuple(name for name, path in zip(REQUIRED_SOURCE_ADDITIONS, required)
                    if path not in paths)
    _need(len(rows) >= BASE_SOURCE_MEMBER_COUNT,
          "scientific source membership shrank below the verified baseline")
    return {
        "current_member_count": len(rows),
        "verified_baseline_member_count": BASE_SOURCE_MEMBER_COUNT,
        "required_new_members": REQUIRED_SOURCE_ADDITIONS,
        "missing_required_new_members": missing,
        "declared_schema_complete": not missing,
        "collected_clean_manifest_verified": False,
    }


def expected_membership() -> dict:
    """Derive and cross-check the complete successful retained-file membership."""
    control, policy, transition, supervision, prior, source_schema = _modules()
    scheduled = prior.scheduled_payloads()
    _need(len(scheduled) == SCIENTIFIC_PAYLOAD_COUNT
          and len({name for name, _ in scheduled}) == SCIENTIFIC_PAYLOAD_COUNT,
          "scientific payload membership differs")
    _need(len(policy.schedule()) == 49, "scientific operation membership differs")
    _need(policy.MAX_RECORD_BYTES == PHASE_RECORD_BODY_CEILING_BYTES,
          "phase-record body ceiling differs")
    phase_names = tuple(control.phase_name(index) for index in range(PHASE_RECORD_COUNT))
    _need(control.MAX_PHASE_RECORDS == PHASE_RECORD_COUNT
          and policy.MAX_RECORDS >= PHASE_RECORD_COUNT,
          "phase-record count differs")
    _need(phase_names == tuple(f"native-phase-{index:03d}.json"
                               for index in range(PHASE_RECORD_COUNT)),
          "native phase filename convention differs")

    expected_metadata = (
        (control.ROOT_JOURNAL_NAME, control.EXTERNAL_JOURNAL_MAX),
        (control.SOURCE_NAME, source_schema.METADATA_CAP),
        (control.ENVIRONMENT_NAME, source_schema.METADATA_CAP),
        *((transition.permission_name(phase), transition.PERMISSION_MAX)
          for phase in transition.PHASES),
        *((transition.environment_name(phase), source_schema.METADATA_CAP)
          for phase in transition.PHASES),
    )
    _need(set(expected_metadata) == set(ROOT_METADATA_BODY_CEILINGS)
          and len(expected_metadata) == ROOT_METADATA_COUNT,
          "root metadata names or ceilings differ")
    _need(control.EXTERNAL_PHASE_NAMES
          == frozenset(name for name, _ in DEVELOPMENT_EXTERNAL_SUCCESS_CEILINGS[1:]),
          "external transition membership differs")
    _need(control.EXTERNAL_TOTAL_MAX == 64 << 10
          and control.EXTERNAL_FAILURE_MAX == 8 << 10
          and control.SHARED_FAILURE_RESERVE == SHARED_FAILURE_RESERVE_BYTES,
          "external failure/storage limits differ")
    actual_supervision = tuple(
        (supervision.request_name(phase), supervision.REQUEST_MAX)
        for phase in supervision.PHASES)
    actual_supervision += tuple(
        (supervision.exit_name(phase), supervision.EXIT_MAX)
        for phase in supervision.PHASES)
    _need(supervision.PHASES == ("development", "primary", "sensitivity", "audit")
          and set(actual_supervision) == set(SUPERVISION_CEILINGS)
          and supervision.SUPERVISION_TOTAL_MAX == SUPERVISION_TOTAL_CEILING_BYTES
          and supervision.SHARED_FAILURE_RESERVE == SHARED_FAILURE_RESERVE_BYTES
          and supervision.ROOT_BUDGET == STORE_BUDGET_BYTES,
          "actual supervision storage contract differs")
    _need(_sum_ceilings(SUPERVISION_CEILINGS) == SUPERVISION_TOTAL_CEILING_BYTES,
          "supervision ceiling differs")
    _need(sum(size for _, size in EXISTING_INSPECTION_FILES)
          == EXISTING_INSPECTION_LOGICAL_BYTES,
          "consumed inspection size evidence differs")
    _need(EXISTING_STORAGE_ATTEMPT_FILES
          == (("native-max-layout-measurement.json", 0),),
          "consumed storage-attempt evidence differs")
    _need(_sum_ceilings(STORAGE_EVIDENCE_FILES) == 128 << 10,
          "storage evidence ceilings differ")
    _need(supervision.STORAGE_ADMISSION_MAX == STORAGE_EVIDENCE_BODY_CEILING_BYTES
          and supervision.STORAGE_ADMISSION_PATH.endswith("/native-storage-admission.json")
          and supervision.LAYOUT_EVIDENCE_PATH.endswith(
              "/native-max-layout-measurement-attempt-002.json"),
          "actual storage evidence names or ceilings differ")

    root_bodies = (tuple(name for name, _ in scheduled) + phase_names
                   + tuple(name for name, _ in ROOT_METADATA_BODY_CEILINGS))
    _need(len(root_bodies) == ROOT_BODY_COUNT
          and len(set(root_bodies)) == ROOT_BODY_COUNT,
          "root payload membership is not closed")
    receipts = tuple(prior.receipt_name(name) for name in root_bodies)
    root_files = ("store-header.json", "store.lock", *root_bodies, *receipts)
    _need(len(root_files) == ROOT_REGULAR_FILE_COUNT
          and len(set(root_files)) == ROOT_REGULAR_FILE_COUNT,
          "root regular-file membership is not closed")
    return {
        "scientific_payloads": tuple(name for name, _ in scheduled),
        "scientific_payload_components": tuple(key for _, key in scheduled),
        "phase_records": phase_names,
        "root_metadata": tuple(name for name, _ in ROOT_METADATA_BODY_CEILINGS),
        "root_receipts": receipts,
        "root_regular_files": root_files,
        "development_external_success": tuple(
            name for name, _ in DEVELOPMENT_EXTERNAL_SUCCESS_CEILINGS),
        "supervision": tuple(name for name, _ in SUPERVISION_CEILINGS),
        "existing_consumed_inspection": tuple(name for name, _ in EXISTING_INSPECTION_FILES),
        "existing_consumed_storage_attempt": tuple(
            name for name, _ in EXISTING_STORAGE_ATTEMPT_FILES),
        "storage_evidence_prerequisites": tuple(name for name, _ in STORAGE_EVIDENCE_FILES),
    }


def _component_sizes(value: object, prior) -> dict[str, int]:
    _need(type(value) is dict and set(value) == set(prior.COMPONENT_KEYS)
          and len(value) == len(prior.COMPONENT_KEYS),
          "component size membership differs")
    _need(all(type(key) is str and type(size) is int and size > 0
              for key, size in value.items()),
          "component sizes must be positive exact integers")
    return {key: value[key] for key in prior.COMPONENT_KEYS}


def _delta_evidence(value: object) -> dict | None:
    if value is None:
        return None
    _need(type(value) is dict and tuple(value) == NATIVE_DELTA_EVIDENCE_KEYS,
          "native delta evidence requires exact ordered fields")
    _need(value["kind"] == NATIVE_DELTA_EVIDENCE_KIND,
          "native delta evidence kind differs")
    _need(type(value["sha256"]) is str and _SHA256.fullmatch(value["sha256"]) is not None,
          "native delta evidence SHA-256 invalid")
    return dict(value)


def project(component_sizes: object, *, native_payload_delta_ceiling: object = None,
            native_payload_delta_evidence: object = None) -> dict:
    """Return the complete conditional logical-byte ledger.

    ``native_payload_delta_ceiling`` is the aggregate increase of all 82 native
    payload bodies over the supplied CPU component projection.  A number without
    an evidence reference is rejected, but this pure function does not
    authenticate the referenced bytes.  Omitting both leaves the launch decision
    explicitly unresolved.
    """
    control, policy, transition, supervision, prior, source_schema = _modules()
    membership = expected_membership()
    sizes = _component_sizes(component_sizes, prior)
    if native_payload_delta_ceiling is None:
        _need(native_payload_delta_evidence is None,
              "native delta evidence supplied without a ceiling")
        delta = evidence = None
    else:
        _need(type(native_payload_delta_ceiling) is int
              and native_payload_delta_ceiling >= 0,
              "native payload delta ceiling must be a nonnegative exact integer")
        delta = native_payload_delta_ceiling
        evidence = _delta_evidence(native_payload_delta_evidence)
        _need(evidence is not None, "native payload delta ceiling lacks an evidence reference")

    scheduled = tuple(zip(membership["scientific_payloads"],
                          membership["scientific_payload_components"]))
    component_body_bytes = sum(sizes[key] for _, key in scheduled)
    # Use the exact names/encodings but the ten-digit whole-store ceiling for
    # receipts.  Native body growth therefore cannot cross an uncharged digit
    # boundary in the receipt's decimal size field.
    component_receipt_ceiling = sum(
        len(prior.encode_receipt(name, STORE_BUDGET_BYTES, "0" * 64,
                                 "bytes" if name.endswith(".json")
                                 else "torch_weights_only"))
        for name, _ in scheduled)
    phase_body_ceiling = PHASE_RECORD_COUNT * PHASE_RECORD_BODY_CEILING_BYTES
    phase_receipt_ceiling = sum(
        len(prior.encode_receipt(name, PHASE_RECORD_BODY_CEILING_BYTES,
                                 "0" * 64, "bytes"))
        for name in membership["phase_records"])
    metadata_body_ceiling = _sum_ceilings(ROOT_METADATA_BODY_CEILINGS)
    metadata_receipt_ceiling = sum(
        len(prior.encode_receipt(name, ceiling, "0" * 64, "bytes"))
        for name, ceiling in ROOT_METADATA_BODY_CEILINGS)
    header_bytes = len(prior.encode_store_header())
    root_regular_before_delta = (component_body_bytes + component_receipt_ceiling
                                 + phase_body_ceiling + phase_receipt_ceiling
                                 + metadata_body_ceiling + metadata_receipt_ceiling
                                 + header_bytes)
    development_external_success = _sum_ceilings(DEVELOPMENT_EXTERNAL_SUCCESS_CEILINGS)
    supervision = _sum_ceilings(SUPERVISION_CEILINGS)
    existing = _sum_ceilings(EXISTING_INSPECTION_FILES)
    admission = _sum_ceilings(STORAGE_EVIDENCE_FILES)
    outside_root_normal = development_external_success + supervision + existing + admission
    conditional_before_delta = (root_regular_before_delta + outside_root_normal
                                + SHARED_FAILURE_RESERVE_BYTES)
    residual_before_delta = STORE_BUDGET_BYTES - conditional_before_delta
    sources = source_membership_status()
    blockers = []
    if delta is None:
        blockers.append("native_payload_delta_ceiling_unresolved")
    if evidence is not None:
        blockers.append("native_payload_delta_evidence_not_authenticated")
    if not sources["declared_schema_complete"]:
        blockers.append("declared_execution_source_schema_incomplete")
    blockers.append("collected_clean_source_manifest_not_verified")
    bounded_total = None if delta is None else conditional_before_delta + delta
    arithmetic_fit = None if bounded_total is None else bounded_total <= STORE_BUDGET_BYTES
    if arithmetic_fit is None:
        arithmetic_status = "unresolved_native_delta"
    elif arithmetic_fit:
        arithmetic_status = "conditional_fit_under_caller_referenced_delta"
    else:
        arithmetic_status = "conditional_exceeds_under_caller_referenced_delta"
    decision = "unresolved_requires_authenticated_bound_and_frozen_manifest"
    return {
        "schema": SCHEMA,
        "evidence_role": "conditional_complete_membership_accounting",
        "execution_enabled": False,
        "membership": {
            "scientific_payload_bodies": SCIENTIFIC_PAYLOAD_COUNT,
            "phase_record_bodies": PHASE_RECORD_COUNT,
            "root_metadata_bodies": ROOT_METADATA_COUNT,
            "root_body_files": ROOT_BODY_COUNT,
            "root_receipt_files": ROOT_BODY_COUNT,
            "root_header_files": 1,
            "root_empty_lock_files": 1,
            "root_regular_files": ROOT_REGULAR_FILE_COUNT,
            "development_external_success_files": len(DEVELOPMENT_EXTERNAL_SUCCESS_CEILINGS),
            "supervision_files": len(SUPERVISION_CEILINGS),
            "existing_consumed_inspection_files": len(EXISTING_INSPECTION_FILES),
            "storage_evidence_prerequisite_files": len(STORAGE_EVIDENCE_FILES),
        },
        "logical_bytes": {
            "measured_cpu_component_projection": component_body_bytes,
            "scheduled_payload_receipt_ceiling": component_receipt_ceiling,
            "phase_record_body_ceiling": phase_body_ceiling,
            "phase_record_receipt_ceiling": phase_receipt_ceiling,
            "root_metadata_body_ceiling": metadata_body_ceiling,
            "root_metadata_receipt_ceiling": metadata_receipt_ceiling,
            "store_header": header_bytes,
            "empty_store_lock": 0,
            "root_regular_before_native_delta": root_regular_before_delta,
            "development_external_success_ceiling": development_external_success,
            "supervision_ceiling": supervision,
            "existing_consumed_inspection_measured": existing,
            "storage_evidence_prerequisite_ceiling": admission,
            "outside_root_normal_ceiling": outside_root_normal,
            "shared_failure_reserve": SHARED_FAILURE_RESERVE_BYTES,
            "conditional_total_before_native_delta": conditional_before_delta,
            "residual_before_native_delta": residual_before_delta,
            "native_payload_delta_ceiling": delta,
            "bounded_total_with_native_delta": bounded_total,
            "store_budget": STORE_BUDGET_BYTES,
        },
        "native_payload_delta_evidence": evidence,
        "source_membership": sources,
        "arithmetic_fit_under_supplied_delta": arithmetic_fit,
        "conditional_arithmetic_status": arithmetic_status,
        "launch_storage_decision": decision,
        "launch_storage_blockers": tuple(blockers),
        "logical_cap_scope": "retained_regular_file_logical_bytes",
        "filesystem_allocation_status": "separate_not_bounded_by_logical_projection",
        "transient_memory_status": "separate_runtime_resource",
        "temporary_file_status": "direct_final_create_source_reviewed_not_machine_proved_by_accounting_api",
        "scientific_execution_certified": False,
    }


def main(argv: object = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    _need(type(args) is list and not args, "this inert helper has no command-line action")
    print("final_storage_accounting: inert; no filesystem or execution action")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
