"""Pure closed accounting for a prospective I7 ArtifactStore root.

Import and default CLI execution neither import Torch nor open a filesystem
path.  Explicit accounting calls lazily load the actual CPU policy/store.  The
90 phase filenames are a storage-fixture convention, not an adopted adapter.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import re
import sys


class AccountingError(ValueError):
    pass


SCHEMA = "i7_root_storage_projection_v1"
SCIENTIFIC_PROFILE = "scientific_mnist_current32_v1"
FIXTURE_PROFILES = ("fixture_tiny_cpu_v1", "fixture_tiny_mlp_cpu_v1")
STORE_BUDGET_BYTES = 1 << 30
FAILURE_RESERVE_BYTES = 1 << 20
MIN_FREE_BYTES = 1 << 30
PHASE_RECORD_BODY_CEILING_BYTES = 128 << 10
PHASE_RECORD_COUNT = 90
PAYLOAD_COUNT = 82
REGULAR_FILE_COUNT = 346

COMPONENT_KEYS = (
    "anchor_pilot",
    "anchor_long",
    "source_witness",
    "branch_results",
    "independent_audit_all_failure",
    "source_completion_pilot_on",
    "source_completion_pilot_off",
    "capture_comparison_pilot",
    "source_completion_long_on",
    "plan_pilot",
    "plan_long",
)
COMPONENT_COUNTS = (2, 16, 18, 18, 16, 1, 1, 1, 4, 1, 4)

_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}\Z", re.ASCII)
_SHA = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)
def _need(condition: bool, message: str) -> None:
    if not condition:
        raise AccountingError(message)


PHASE_PAYLOADS = tuple(f"storage-phase-{sequence:03d}.json"
                       for sequence in range(PHASE_RECORD_COUNT))


def _policy_and_storage():
    """Lazy boundary: explicit accounting may import the CPU policy and Torch store."""
    import artifact_store as storage
    import native_phase_policy as policy
    return policy, storage


def scheduled_payloads() -> tuple[tuple[str, str], ...]:
    """Classify the names returned by the actual fixed policy into its 11 size keys."""
    policy, _ = _policy_and_storage()
    classified: dict[str, str] = {}

    def add(name: str, key: str) -> None:
        _need(name not in classified, "component definition contains a duplicate payload")
        classified[name] = key

    add(policy.plan_name(71990), "plan_pilot")
    for bundle in (*policy.PRIMARY, 71901):
        add(policy.plan_name(bundle), "plan_long")
    for update in policy.PILOT_ANCHORS:
        add(policy.artifact_name("pilot", 71990, update, "anchor"), "anchor_pilot")
        add(policy.artifact_name("pilot", 71990, update, "source-witness"), "source_witness")
        add(policy.artifact_name("pilot", 71990, update, "branch-results"), "branch_results")
    add(policy.source_name("pilot", 71990), "source_completion_pilot_on")
    add(policy.source_name("pilot", 71990, "capture_off"), "source_completion_pilot_off")
    add("i7-native-pilot-b71990-capture-comparison.json", "capture_comparison_pilot")
    for role, bundles in (("primary", policy.PRIMARY), ("sensitivity", (71901,))):
        for bundle in bundles:
            add(policy.source_name(role, bundle), "source_completion_long_on")
            for update in policy.ANCHORS:
                add(policy.artifact_name(role, bundle, update, "anchor"), "anchor_long")
                add(policy.artifact_name(role, bundle, update, "source-witness"), "source_witness")
                add(policy.artifact_name(role, bundle, update, "branch-results"), "branch_results")
                add(policy.artifact_name(role, bundle, update, "independent-audit"),
                    "independent_audit_all_failure")

    program = policy.schedule()
    _need(type(program) is list and len(program) == 49, "actual policy operation count differs")
    _need(all(type(operation) is dict
              and tuple(operation) == ("phase", "operation", "kind", "artifacts")
              and type(operation["phase"]) is str
              and type(operation["operation"]) is str
              and type(operation["kind"]) is str
              and type(operation["artifacts"]) is list
              and all(type(name) is str for name in operation["artifacts"])
              for operation in program), "actual policy operation schema differs")
    kinds = Counter(operation["kind"] for operation in program)
    _need(kinds == Counter({"work":41, "decision":4, "boundary":4}),
          "actual policy operation kinds differ")
    events = sum(2 if operation["kind"] == "work" else 1 for operation in program)
    _need(events == PHASE_RECORD_COUNT and policy.MAX_RECORDS >= events,
          "actual policy phase-record count or bound differs")
    actual = tuple(name for operation in program for name in operation["artifacts"])
    _need(len(actual) == len(set(actual)) and set(actual) == set(classified),
          "actual policy payload membership differs from component classification")
    return tuple((name, classified[name]) for name in actual)


def _validate_name(name: object) -> str:
    _need(type(name) is str and _NAME.fullmatch(name) is not None,
          "invalid flat ASCII artifact name")
    _need(name not in ("store-header.json", "store.lock")
          and not name.startswith(("receipt-", "failure-")),
          "reserved artifact name")
    return name


def receipt_name(name: object) -> str:
    _, storage = _policy_and_storage()
    try:
        value = storage._validate_name(name)
    except storage.StoreError as exc:
        raise AccountingError("invalid artifact name for receipt") from exc
    return "receipt-" + hashlib.sha256(value.encode("ascii")).hexdigest() + ".json"


def encode_receipt(name: object, size: object, sha256: object, encoding: object) -> bytes:
    _, storage = _policy_and_storage()
    try:
        value = storage._validate_name(name)
    except storage.StoreError as exc:
        raise AccountingError("invalid artifact name for receipt") from exc
    _need(type(size) is int and size > 0, "receipt size must be a positive exact integer")
    _need(type(sha256) is str and _SHA.fullmatch(sha256) is not None,
          "receipt SHA-256 must be lowercase hexadecimal")
    _need(type(encoding) is str and encoding in ("bytes", "torch_weights_only"),
          "invalid receipt encoding")
    return storage._json_bytes({"schema":"i7_artifact_receipt_v1", "name":value,
                                "size":size, "sha256":sha256, "status":"complete",
                                "encoding":encoding})


def encode_store_header(*, profile: object = None, budget_bytes: object = None,
                        failure_reserve_bytes: object = None,
                        min_filesystem_free_bytes: object = None) -> bytes:
    _, storage = _policy_and_storage()
    profile = storage.SCIENTIFIC if profile is None else profile
    budget_bytes = storage.DEFAULT_BUDGET if budget_bytes is None else budget_bytes
    failure_reserve_bytes = (storage.DEFAULT_FAILURE_RESERVE if failure_reserve_bytes is None
                             else failure_reserve_bytes)
    min_filesystem_free_bytes = (storage.DEFAULT_MIN_FREE if min_filesystem_free_bytes is None
                                 else min_filesystem_free_bytes)
    _need(type(profile) is str and profile in (storage.SCIENTIFIC, *storage.FIXTURES),
          "unknown store profile")
    values = (budget_bytes, failure_reserve_bytes, min_filesystem_free_bytes)
    _need(all(type(value) is int and value >= 0 for value in values),
          "store limits must be nonnegative exact integers")
    _need(failure_reserve_bytes >= storage._FAILURE_MAX
          and budget_bytes > failure_reserve_bytes + storage._RECEIPT_MAX,
          "inconsistent store limits")
    if profile == storage.SCIENTIFIC:
        _need(values == (storage.DEFAULT_BUDGET, storage.DEFAULT_FAILURE_RESERVE,
                         storage.DEFAULT_MIN_FREE),
              "scientific store limits differ from defaults")
    return storage._json_bytes({"schema":"i7_artifact_store_v1", "profile":profile,
                                "budget_bytes":budget_bytes,
                                "failure_reserve_bytes":failure_reserve_bytes,
                                "min_filesystem_free_bytes":min_filesystem_free_bytes,
                                "cap_semantics":"sum_regular_file_logical_bytes"})


def _verify_definition(payloads: tuple[tuple[str, str], ...]) -> None:
    _need(len(payloads) == PAYLOAD_COUNT, "scheduled payload count differs")
    _need(all(type(row) is tuple and len(row) == 2 for row in payloads),
          "malformed scheduled payload row")
    names = tuple(row[0] for row in payloads)
    keys = tuple(row[1] for row in payloads)
    _need(len(set(names)) == PAYLOAD_COUNT and all(_validate_name(name) == name for name in names),
          "duplicate or invalid scheduled payload")
    _need(set(keys) == set(COMPONENT_KEYS), "unclassified scheduled payload")
    _need(tuple(Counter(keys)[key] for key in COMPONENT_KEYS) == COMPONENT_COUNTS,
          "component membership counts differ")
    _need(len(PHASE_PAYLOADS) == PHASE_RECORD_COUNT
          and len(set(PHASE_PAYLOADS)) == PHASE_RECORD_COUNT,
          "phase fixture membership differs")
    _need(all(name == f"storage-phase-{index:03d}.json"
              for index, name in enumerate(PHASE_PAYLOADS)),
          "phase fixture naming differs")


def expected_regular_files() -> tuple[str, ...]:
    """Return the closed conditional 346-file baseline, including metadata."""
    scheduled = scheduled_payloads()
    _verify_definition(scheduled)
    payload_names = tuple(name for name, _ in scheduled) + PHASE_PAYLOADS
    result = ("store-header.json", "store.lock", *payload_names,
              *(receipt_name(name) for name in payload_names))
    _need(len(result) == REGULAR_FILE_COUNT and len(set(result)) == REGULAR_FILE_COUNT,
          "closed regular-file membership is inconsistent")
    return result


def validate_regular_files(files: object) -> bool:
    """Reject duplicate, missing, unknown, non-string or unclassified filenames."""
    _need(type(files) in (list, tuple), "file inventory must be an exact list or tuple")
    _need(all(type(name) is str for name in files), "file inventory contains a non-string")
    _need(len(files) == len(set(files)), "file inventory contains a duplicate")
    expected = expected_regular_files()
    _need(len(files) == len(expected) and set(files) == set(expected),
          "file inventory has missing or unknown membership")
    return True


def _component_sizes(value: object) -> tuple[int, ...]:
    _need(type(value) is dict and all(type(key) is str for key in value),
          "component sizes must be an exact mapping")
    _need(set(value) == set(COMPONENT_KEYS) and len(value) == len(COMPONENT_KEYS),
          "component size membership differs")
    result = tuple(value[key] for key in COMPONENT_KEYS)
    _need(all(type(size) is int and size > 0 for size in result),
          "component sizes must be positive exact integers")
    return result


def _encoding(name: str) -> str:
    return "bytes" if name.endswith(".json") else "torch_weights_only"


def project(component_sizes: object) -> dict:
    """Compute a conditional logical-byte ledger; this is not a fit certificate."""
    policy, storage = _policy_and_storage()
    scheduled = scheduled_payloads()
    _verify_definition(scheduled)
    _need(policy.MAX_RECORD_BYTES == PHASE_RECORD_BODY_CEILING_BYTES,
          "actual phase-record ceiling differs from accounting contract")
    _need((storage.DEFAULT_BUDGET, storage.DEFAULT_FAILURE_RESERVE, storage.DEFAULT_MIN_FREE)
          == (STORE_BUDGET_BYTES, FAILURE_RESERVE_BYTES, MIN_FREE_BYTES),
          "actual shared store limits differ from accounting contract")
    sizes = _component_sizes(component_sizes)
    by_component = dict(zip(COMPONENT_KEYS, sizes))
    component_rows = []
    for key, count, size in zip(COMPONENT_KEYS, COMPONENT_COUNTS, sizes):
        component_rows.append({"component":key, "payload_count":count,
                               "serialized_bytes_each":size,
                               "payload_bytes":count * size})

    payload_body_bytes = sum(by_component[key] for _, key in scheduled)
    payload_receipt_bytes = sum(len(encode_receipt(name, by_component[key], "0" * 64,
                                                       _encoding(name)))
                                for name, key in scheduled)
    phase_body_bytes = PHASE_RECORD_COUNT * PHASE_RECORD_BODY_CEILING_BYTES
    phase_receipt_bytes = sum(len(encode_receipt(name, PHASE_RECORD_BODY_CEILING_BYTES,
                                                  "0" * 64, "bytes"))
                              for name in PHASE_PAYLOADS)
    header_bytes = len(encode_store_header())
    regular = (payload_body_bytes + payload_receipt_bytes + phase_body_bytes
               + phase_receipt_bytes + header_bytes)
    normal_ceiling = STORE_BUDGET_BYTES - FAILURE_RESERVE_BYTES
    return {
        "schema": SCHEMA,
        "evidence_role": "conditional_storage_accounting",
        "execution_enabled": False,
        "component_rows": component_rows,
        "membership": {
            "scheduled_payloads": PAYLOAD_COUNT,
            "fixture_phase_records": PHASE_RECORD_COUNT,
            "payload_and_phase_receipts": PAYLOAD_COUNT + PHASE_RECORD_COUNT,
            "header_files": 1,
            "empty_lock_files": 1,
            "regular_files": REGULAR_FILE_COUNT,
        },
        "logical_bytes": {
            "scheduled_payload_bodies": payload_body_bytes,
            "scheduled_payload_receipts": payload_receipt_bytes,
            "fixture_phase_record_body_ceilings": phase_body_bytes,
            "fixture_phase_record_receipts": phase_receipt_bytes,
            "store_header": header_bytes,
            "empty_store_lock": 0,
            "projected_regular_files": regular,
            "shared_failure_reserve": FAILURE_RESERVE_BYTES,
            "projected_regular_files_plus_reserve": regular + FAILURE_RESERVE_BYTES,
            "store_budget": STORE_BUDGET_BYTES,
            "normal_write_ceiling": normal_ceiling,
            "normal_write_headroom": normal_ceiling - regular,
        },
        "normal_write_ceiling_fit": regular <= normal_ceiling,
        "conditional_on": "adapter_retains_all_90_fixture_named_phase_records",
        "native_payload_size_delta": "unresolved",
        "native_adapter_additional_files": "unresolved",
        "phase_filename_status": "fixture_only_not_scheduled_by_native_policy",
        "scientific_execution_certified": False,
    }


def main(argv: object = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    _need(type(args) is list and not args, "this inert helper has no command-line action")
    print("root_storage_accounting: inert; no filesystem or execution action")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
