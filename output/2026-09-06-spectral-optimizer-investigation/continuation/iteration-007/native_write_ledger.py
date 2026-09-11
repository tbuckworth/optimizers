"""Closed logical-byte reservations for prospective scientific-root writes.

Import and default CLI execution are stdlib-only and perform no filesystem I/O.
Explicit calls lazily recompute the reviewed I7 topology and fixed membership.
This module neither writes an ArtifactStore nor grants execution authority.
"""
from __future__ import annotations

from functools import lru_cache
import re
import sys


class NativeWriteLedgerError(ValueError):
    pass


SCHEMA = "i7_native_write_ledger_v1"
STORE_BUDGET_BYTES = 1 << 30
SHARED_FAILURE_RESERVE_BYTES = 1 << 20
OUTSIDE_ROOT_NORMAL_BYTES = 287_990
EXTERNAL_FAILURE_BYTES = 8 << 10
ROOT_FAILURE_FILE_BYTES = 16 << 10
NORMAL_ROOT_CEILING_BYTES = (
    STORE_BUDGET_BYTES
    - OUTSIDE_ROOT_NORMAL_BYTES
    - SHARED_FAILURE_RESERVE_BYTES
)
FAILURE_ROOT_CEILING_BYTES = (
    STORE_BUDGET_BYTES
    - OUTSIDE_ROOT_NORMAL_BYTES
    - EXTERNAL_FAILURE_BYTES
)
EXPECTED_SCIENTIFIC_BODY_BYTES = 1_049_771_346
EXPECTED_SUCCESSFUL_ROOT_BYTES = 1_066_884_301
EXPECTED_COMPLETE_BYTES = 1_068_220_867
EXPECTED_NORMAL_ROOT_RESIDUAL_BYTES = 5_520_957
EXPECTED_HEADER_BYTES = 222
EXPECTED_ROOT_BODY_COUNT = 181
EXPECTED_ROOT_REGULAR_FILE_COUNT = 364
FAILURE_NAME_PATTERN = r"failure-[0-9]{6}\.json"
MAX_FAILURE_SEQUENCE = 999_999

_FAILURE_NAME = re.compile(FAILURE_NAME_PATTERN, re.ASCII)


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise NativeWriteLedgerError(message)


def _integer(value: object, label: str, *, minimum: int = 0) -> int:
    _need(type(value) is int and value >= minimum,
          label + " must be an exact bounded integer")
    return value


@lru_cache(maxsize=1)
def _definition() -> dict:
    """Build one private definition; no returned object exposes its mappings."""
    import artifact_store as storage
    import final_storage_accounting as final
    import native_control as control
    import native_storage_topology_bound as topology_bound
    import root_storage_accounting as prior

    topology = topology_bound.compute()
    membership = final.expected_membership()
    scheduled = tuple(zip(membership["scientific_payloads"],
                          membership["scientific_payload_components"]))
    _need(len(scheduled) == 82 and len({name for name, _ in scheduled}) == 82,
          "scientific payload membership differs")
    components = topology["components"]
    specs = []

    def add(name: str, encoding: str, body: int, receipt_basis: int,
            component: str) -> None:
        _need(type(name) is str and type(encoding) is str
              and encoding in ("bytes", "torch_weights_only"),
              "malformed body identity")
        _integer(body, "body ceiling", minimum=1)
        _integer(receipt_basis, "receipt size basis", minimum=1)
        receipt_name = prior.receipt_name(name)
        receipt = len(prior.encode_receipt(
            name, receipt_basis, "0" * 64, encoding))
        specs.append((name, encoding, body, receipt_name, receipt,
                      receipt_basis, component))

    for name, component in scheduled:
        _need(component in components, "scheduled component lacks topology")
        row = components[component]
        add(name, row["encoding"], row["body_bytes_upper"],
            STORE_BUDGET_BYTES, component)

    for name in membership["phase_records"]:
        add(name, "bytes", final.PHASE_RECORD_BODY_CEILING_BYTES,
            final.PHASE_RECORD_BODY_CEILING_BYTES, "phase_record")

    metadata = dict(final.ROOT_METADATA_BODY_CEILINGS)
    _need(tuple(metadata) == membership["root_metadata"],
          "root metadata ordering or membership differs")
    for name, ceiling in metadata.items():
        add(name, "bytes", ceiling, ceiling, "root_metadata")

    _need(len(specs) == EXPECTED_ROOT_BODY_COUNT
          and len({row[0] for row in specs}) == EXPECTED_ROOT_BODY_COUNT,
          "root body membership differs")
    body_names = tuple(row[0] for row in specs)
    receipt_names = tuple(row[3] for row in specs)
    _need(tuple(membership["root_receipts"]) == receipt_names,
          "root receipt membership differs")

    header_bytes = len(prior.encode_store_header())
    _need(header_bytes == EXPECTED_HEADER_BYTES,
          "scientific store header length differs")
    success_names = ("store-header.json", "store.lock",
                     *body_names, *receipt_names)
    _need(tuple(membership["root_regular_files"]) == success_names
          and len(success_names) == EXPECTED_ROOT_REGULAR_FILE_COUNT
          and len(set(success_names)) == EXPECTED_ROOT_REGULAR_FILE_COUNT,
          "successful root namespace differs")

    scientific_bodies = sum(
        components[key]["payload_count"] * components[key]["body_bytes_upper"]
        for key in components)
    phase_bodies = len(membership["phase_records"]) * final.PHASE_RECORD_BODY_CEILING_BYTES
    metadata_bodies = sum(metadata.values())
    receipts = sum(row[4] for row in specs)
    successful_root = scientific_bodies + phase_bodies + metadata_bodies + receipts + header_bytes

    outside_groups = (
        ("development_external_success", tuple(final.DEVELOPMENT_EXTERNAL_SUCCESS_CEILINGS)),
        ("supervision", tuple(final.SUPERVISION_CEILINGS)),
        ("existing_consumed_inspection", tuple(final.EXISTING_INSPECTION_FILES)),
        ("storage_evidence_prerequisites", tuple(final.STORAGE_EVIDENCE_FILES)),
    )
    outside = sum(size for _, rows in outside_groups for _, size in rows)
    _need(outside == OUTSIDE_ROOT_NORMAL_BYTES,
          "outside-root normal ceiling differs")
    _need(control.EXTERNAL_FAILURE_MAX == EXTERNAL_FAILURE_BYTES
          and control.EXTERNAL_TOTAL_MAX
              == sum(size for _, size in final.DEVELOPMENT_EXTERNAL_SUCCESS_CEILINGS)
                 + EXTERNAL_FAILURE_BYTES,
          "outside-root failure split differs")
    _need(storage._FAILURE_MAX == ROOT_FAILURE_FILE_BYTES
          and storage.DEFAULT_FAILURE_RESERVE == SHARED_FAILURE_RESERVE_BYTES
          and storage.DEFAULT_BUDGET == STORE_BUDGET_BYTES,
          "ArtifactStore failure or budget contract differs")
    _need(scientific_bodies == EXPECTED_SCIENTIFIC_BODY_BYTES,
          "scientific body ceiling differs")
    _need(successful_root == EXPECTED_SUCCESSFUL_ROOT_BYTES,
          "successful root ceiling differs")
    complete = successful_root + outside + SHARED_FAILURE_RESERVE_BYTES
    _need(complete == EXPECTED_COMPLETE_BYTES
          and topology["complete_logical_bytes_upper"] == complete,
          "complete topology/ledger total differs")
    _need(NORMAL_ROOT_CEILING_BYTES - successful_root
              == EXPECTED_NORMAL_ROOT_RESIDUAL_BYTES,
          "normal-root residual differs")

    return {
        "specs": tuple(specs),
        "body_names": body_names,
        "receipt_names": receipt_names,
        "success_names": success_names,
        "outside_groups": outside_groups,
        "scientific_bodies": scientific_bodies,
        "phase_bodies": phase_bodies,
        "metadata_bodies": metadata_bodies,
        "receipt_bytes": receipts,
        "header_bytes": header_bytes,
        "successful_root": successful_root,
        "complete": complete,
    }


def compute() -> dict:
    """Return the closed prospective ledger; this is not write authority."""
    definition = _definition()
    body_limits = {
        name: {
            "encoding": encoding,
            "body_bytes_upper": body,
            "receipt_name": receipt_name,
            "receipt_bytes_upper": receipt,
            "receipt_size_basis": receipt_basis,
            "component": component,
        }
        for (name, encoding, body, receipt_name, receipt,
             receipt_basis, component) in definition["specs"]
    }
    outside = {
        group: tuple({"name": name, "bytes_upper": size}
                     for name, size in rows)
        for group, rows in definition["outside_groups"]
    }
    return {
        "schema": SCHEMA,
        "evidence_role": "conditional_write_reservation_arithmetic",
        "execution_enabled": False,
        "body_limits": body_limits,
        "membership": {
            "body_names": definition["body_names"],
            "receipt_names": definition["receipt_names"],
            "static_success_names": definition["success_names"],
            "static_success_count": len(definition["success_names"]),
            "failure_name_pattern": FAILURE_NAME_PATTERN,
            "maximum_failure_sequence": MAX_FAILURE_SEQUENCE,
        },
        "outside_root": outside,
        "logical_bytes": {
            "scientific_payload_bodies": definition["scientific_bodies"],
            "phase_record_bodies": definition["phase_bodies"],
            "root_metadata_bodies": definition["metadata_bodies"],
            "root_receipts": definition["receipt_bytes"],
            "store_header": definition["header_bytes"],
            "empty_store_lock": 0,
            "successful_root_upper": definition["successful_root"],
            "outside_root_normal_upper": OUTSIDE_ROOT_NORMAL_BYTES,
            "shared_failure_reserve": SHARED_FAILURE_RESERVE_BYTES,
            "external_failure_upper": EXTERNAL_FAILURE_BYTES,
            "root_failure_file_upper": ROOT_FAILURE_FILE_BYTES,
            "normal_root_ceiling": NORMAL_ROOT_CEILING_BYTES,
            "failure_root_ceiling": FAILURE_ROOT_CEILING_BYTES,
            "normal_root_residual": (NORMAL_ROOT_CEILING_BYTES
                                     - definition["successful_root"]),
            "complete_success_plus_failure_reserve_upper": definition["complete"],
            "global_budget": STORE_BUDGET_BYTES,
        },
        "conditional_arithmetic_fits": definition["complete"] <= STORE_BUDGET_BYTES,
        "outside_writers_authenticated": False,
        "shared_store_integration_complete": False,
        "execution_authorized": False,
        "scientific_execution_certified": False,
    }


def _body_spec(name: object, encoding: object) -> tuple:
    _need(type(name) is str and type(encoding) is str,
          "write name and encoding must be exact strings")
    specs = {row[0]: row for row in _definition()["specs"]}
    _need(name in specs, "unknown or reserved scientific root body name")
    row = specs[name]
    _need(encoding == row[1], "scientific root body encoding differs")
    return row


def expected_receipt_bytes(name: object, encoding: object,
                           body_bytes: object) -> int:
    """Exact current receipt length from its charged decimal-size basis."""
    body_bytes = _integer(body_bytes, "body bytes", minimum=1)
    row = _body_spec(name, encoding)
    _need(body_bytes <= row[2], "body exceeds per-name ceiling")
    receipt_upper, basis = row[4], row[5]
    result = receipt_upper - len(str(basis)) + len(str(body_bytes))
    _need(0 < result <= receipt_upper, "receipt length arithmetic differs")
    return result


def initialization_reservation(name: object, size_bytes: object,
                               root_logical_before: object) -> dict:
    """Admit one fixed scientific-root initialization file write."""
    _need(type(name) is str, "initialization name must be an exact string")
    size_bytes = _integer(size_bytes, "initialization bytes")
    before = _integer(root_logical_before, "root logical bytes before initialization")
    expected = {"store.lock": 0,
                "store-header.json": _definition()["header_bytes"]}
    _need(name in expected and size_bytes == expected[name],
          "unknown or size-changing scientific initialization write")
    _need(before == 0, "scientific initialization root is not empty")
    post = before + size_bytes
    _need(post <= NORMAL_ROOT_CEILING_BYTES,
          "initialization exceeds normal root ceiling")
    return {"name": name, "bytes": size_bytes,
            "post_root_logical_upper": post,
            "root_ceiling": NORMAL_ROOT_CEILING_BYTES}


def normal_write_budget(name: object, encoding: object,
                        root_logical_before: object) -> dict:
    """Return the smaller per-name/global body capacity before serialization."""
    before = _integer(root_logical_before, "root logical bytes before write")
    _need(before <= NORMAL_ROOT_CEILING_BYTES,
          "existing root exceeds normal root ceiling")
    row = _body_spec(name, encoding)
    available = NORMAL_ROOT_CEILING_BYTES - before - row[4]
    maximum = min(row[2], available)
    _need(maximum >= 1, "no normal root body capacity remains")
    return {
        "name": row[0], "encoding": row[1],
        "body_bytes_upper": row[2], "receipt_name": row[3],
        "receipt_bytes_upper": row[4],
        "max_payload_bytes_now": maximum,
        "component": row[6],
        "root_ceiling": NORMAL_ROOT_CEILING_BYTES,
    }


def admit_normal_write(name: object, encoding: object, body_bytes: object,
                       receipt_bytes: object,
                       root_logical_before: object) -> dict:
    """Reserve a body and its bounded receipt before either immutable create."""
    body_bytes = _integer(body_bytes, "body bytes", minimum=1)
    receipt_bytes = _integer(receipt_bytes, "receipt bytes", minimum=1)
    budget = normal_write_budget(name, encoding, root_logical_before)
    expected = expected_receipt_bytes(name, encoding, body_bytes)
    _need(body_bytes <= budget["max_payload_bytes_now"],
          "body exceeds per-name or global normal-write capacity")
    _need(receipt_bytes == expected
          and receipt_bytes <= budget["receipt_bytes_upper"],
          "actual receipt length differs from bounded encoder")
    # Reserve the ceiling, not the possibly shorter actual decimal-size row.
    increment = body_bytes + budget["receipt_bytes_upper"]
    post = root_logical_before + increment
    _need(post <= NORMAL_ROOT_CEILING_BYTES,
          "normal write exceeds reserved root ceiling")
    return dict(budget, body_bytes=body_bytes,
                actual_receipt_bytes=receipt_bytes,
                reserved_increment_bytes=increment,
                post_root_logical_upper=post)


def admit_failure_write(name: object, failure_bytes: object,
                        root_logical_before: object) -> dict:
    """Admit the next store-generated terminal record using a fresh root scan."""
    _need(type(name) is str and _FAILURE_NAME.fullmatch(name) is not None,
          "failure name must have the six-digit store form")
    sequence = int(name[len("failure-"):-len(".json")])
    _need(1 <= sequence <= MAX_FAILURE_SEQUENCE,
          "failure name is outside the positive bounded store sequence")
    failure_bytes = _integer(failure_bytes, "failure bytes", minimum=1)
    _need(failure_bytes <= ROOT_FAILURE_FILE_BYTES,
          "failure record exceeds per-file ceiling")
    before = _integer(root_logical_before, "freshly scanned root logical bytes")
    post = before + failure_bytes
    _need(post <= FAILURE_ROOT_CEILING_BYTES,
          "failure write exceeds terminal root ceiling")
    return {"name": name, "sequence": sequence,
            "failure_bytes": failure_bytes,
            "post_root_logical_upper": post,
            "root_ceiling": FAILURE_ROOT_CEILING_BYTES,
            "external_failure_reserved_bytes": EXTERNAL_FAILURE_BYTES}


def allowed_root_names() -> dict:
    """Return the finite success set and bounded store-failure name domain."""
    return {
        "static_success_names": _definition()["success_names"],
        "failure_name_pattern": FAILURE_NAME_PATTERN,
        "maximum_failure_sequence": MAX_FAILURE_SEQUENCE,
    }


def main(argv: object = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    _need(type(args) is list and not args,
          "this inert helper has no command-line action")
    print("native_write_ledger: inert; no write or execution admission")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
