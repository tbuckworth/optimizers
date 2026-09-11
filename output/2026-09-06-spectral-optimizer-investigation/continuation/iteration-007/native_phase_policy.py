"""Pure native schedule/record validation. No filesystem, plans, execution or GO authority.

Synthetic contract fixtures may use scientific membership but never become
scientific results. A valid transcript is only a producer attestation; actual
files, semantic envelopes, resources and reviewed launch decisions need adapters.
"""
from __future__ import annotations

import copy
import math
import re

import identity_codec as codec
import artifact_store as storage
import runtime_guard as runtime

PROFILE = storage.SCIENTIFIC
KINDS = ("synthetic_contract_fixture", "native_producer_attestation")
REF_KEYS = ("name", "status", "encoding", "size_bytes", "sha256", "receipt_name",
            "receipt_size_bytes", "receipt_sha256")
ROOT_KEYS = ("path", "device", "inode", "header_sha256")
RESOURCE_KEYS = ("phase_wall_seconds", "phase_cpu_seconds", "peak_rss_bytes",
                 "peak_cuda_allocated_bytes", "peak_cuda_reserved_bytes",
                 "root_logical_bytes", "root_allocated_bytes")
RECORD_KEYS = ("schema", "profile", "evidence_kind", "execution_enabled", "sequence",
               "event", "operation", "phase", "created_utc", "root_binding",
               "sources_sha256", "environment_sha256", "previous_record_sha256",
               "artifacts", "resources", "error")
SHA = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)
MAX_RECORD_BYTES = 128 << 10
MAX_RECORDS = 96
PRIMARY = (71001, 71002, 71003)
ANCHORS = (101, 500, 1000, 2000)
PILOT_ANCHORS = (101, 200)


class PolicyError(ValueError):
    pass


def _need(condition, message):
    if not condition:
        raise PolicyError(message)


def _keys(value, expected, label):
    _need(type(value) is dict and tuple(value) == expected
          and all(type(key) is str for key in value), label + ": exact ordered keys")


def _hash(value, label):
    _need(type(value) is str and SHA.fullmatch(value) is not None, label + ": invalid SHA-256")


def _int(value, low, high, label):
    _need(type(value) is int and low <= value <= high, label + ": invalid integer")


def identity(role, bundle, update):
    """Pure fixed identity metadata, never seeded plan generation."""
    value = dict(run_id=codec._RUN_ID, iteration=7, execution_role=role,
        evidence_role="development" if role == "pilot" else role, bundle=bundle,
        anchor_update=update, source_policy="current32", condition="noise_0.9",
        steps_total=220 if role == "pilot" else 2000, anchor_phase="pre_forward_pre_observe",
        anchor_completed_updates=update-1, anchor_completed_observations=update-1,
        rng_namespace_prefix=[20260906, bundle], stream_roles=dict(codec.STREAM_ROLES))
    codec.validate_identity(value, profile=PROFILE)
    return value


def artifact_name(role, bundle, update, kind):
    return codec.artifact_id(identity(role, bundle, update), profile=PROFILE, kind=kind) + ".pt"


def plan_name(bundle):
    _need(type(bundle) is int and bundle in (*PRIMARY, 71901, 71990), "unknown plan bundle")
    return f"i7-native-b{bundle}-plan.pt"


def source_name(role, bundle, mode="capture_on"):
    identity(role, bundle, 101)
    _need(mode == "capture_on" or role == "pilot" and mode == "capture_off", "invalid source capture mode")
    return f"i7-native-{role}-b{bundle}-source-{mode}.pt"


def _source_outputs(role, bundle, anchors):
    result = []
    for update in anchors:
        result.extend(artifact_name(role, bundle, update, kind) for kind in ("anchor", "source-witness"))
    result.append(source_name(role, bundle))
    return result


def schedule():
    """One fixed attempt: 41 work operations + four decisions + four boundaries."""
    result = []
    def add(phase, name, kind, artifacts=()):
        result.append(dict(phase=phase, operation=name, kind=kind, artifacts=list(artifacts)))
    add("development", "development_go", "decision")
    add("development", "development.plan", "work", [plan_name(71990)])
    add("development", "development.source_pair", "work",
        [*_source_outputs("pilot", 71990, PILOT_ANCHORS), source_name("pilot", 71990, "capture_off"),
         "i7-native-pilot-b71990-capture-comparison.json"])
    for update in PILOT_ANCHORS:
        add("development", f"development.branch.u{update}", "work",
            [artifact_name("pilot", 71990, update, "branch-results")])
    add("development", "development_complete", "boundary")
    # Preparation is before primary GO but charged to primary, not an unbudgeted phase.
    add("primary", "scientific_plans", "work", [plan_name(bundle) for bundle in (*PRIMARY, 71901)])
    add("primary", "primary_go", "decision")
    for bundle in PRIMARY:
        add("primary", f"primary.source.b{bundle}", "work", _source_outputs("primary", bundle, ANCHORS))
    for bundle in PRIMARY:
        for update in ANCHORS:
            add("primary", f"primary.branch.b{bundle}.u{update}", "work",
                [artifact_name("primary", bundle, update, "branch-results")])
    add("primary", "primary_complete", "boundary")
    add("sensitivity", "sensitivity_go", "decision")
    add("sensitivity", "sensitivity.source.b71901", "work", _source_outputs("sensitivity", 71901, ANCHORS))
    for update in ANCHORS:
        add("sensitivity", f"sensitivity.branch.b71901.u{update}", "work",
            [artifact_name("sensitivity", 71901, update, "branch-results")])
    add("sensitivity", "sensitivity_complete", "boundary")
    add("audit", "audit_go", "decision")
    for role, bundles in (("primary", PRIMARY), ("sensitivity", (71901,))):
        for bundle in bundles:
            for update in ANCHORS:
                add("audit", f"audit.{role}.b{bundle}.u{update}", "work",
                    [artifact_name(role, bundle, update, "independent-audit")])
    add("audit", "audit_complete", "boundary")
    return result


def validate_reference(value, *, name, encoding):
    """Receipt-shape/hash consistency only; no bytes have been read here."""
    import hashlib
    _keys(value, REF_KEYS, "artifact reference")
    _need(value["name"] == name and type(value["name"]) is str, "reference membership/name")
    storage._validate_name(name)
    _need(type(value["status"]) is str and value["status"] == "complete"
          and type(value["encoding"]) is str and value["encoding"] == encoding, "reference status/encoding")
    _int(value["size_bytes"], 1, storage.DEFAULT_BUDGET, "artifact size")
    _int(value["receipt_size_bytes"], 1, storage._RECEIPT_MAX, "receipt size")
    _hash(value["sha256"], "artifact hash")
    _hash(value["receipt_sha256"], "receipt hash")
    receipt_name = "receipt-" + hashlib.sha256(name.encode("ascii")).hexdigest() + ".json"
    _need(type(value["receipt_name"]) is str and value["receipt_name"] == receipt_name, "receipt name")
    raw = storage._json_bytes(dict(schema="i7_artifact_receipt_v1", name=name,
        size=value["size_bytes"], sha256=value["sha256"], status="complete", encoding=encoding))
    _need(value["receipt_size_bytes"] == len(raw)
          and value["receipt_sha256"] == hashlib.sha256(raw).hexdigest(), "receipt declaration differs")


def _root(value):
    _keys(value, ROOT_KEYS, "root")
    path = value["path"]
    _need(type(path) is str and path.isascii() and len(path) <= 512
          and path.startswith("/tmp/spectral-experiment-artifacts/")
          and all(part not in ("", ".", "..") for part in path.split("/")[1:]), "root path representation")
    _int(value["device"], 0, (1 << 63)-1, "root device")
    _int(value["inode"], 1, (1 << 63)-1, "root inode")
    _hash(value["header_sha256"], "root header")


def _resources(value, phase, *, failed):
    _keys(value, RESOURCE_KEYS, "resources")
    for key in RESOURCE_KEYS[:2]:
        _need(type(value[key]) is float and math.isfinite(value[key]) and value[key] >= 0., key + ": invalid seconds")
    for key in RESOURCE_KEYS[2:]:
        _int(value[key], 0, (1 << 63)-1, key)
    _need(value["peak_cuda_allocated_bytes"] <= value["peak_cuda_reserved_bytes"], "CUDA reservation below allocation")
    if not failed:
        _need(value["phase_wall_seconds"] <= runtime.PHASE_CAPS[phase]["wall_seconds"], "declared wall cap exceeded")
        cap = runtime.PHASE_CAPS[phase]["cpu_seconds"]
        _need(cap is None or value["phase_cpu_seconds"] <= cap, "declared CPU cap exceeded")
        _need(value["peak_rss_bytes"] <= runtime.RSS_CAP
              and value["peak_cuda_allocated_bytes"] <= runtime.CUDA_ALLOCATED_CAP,
              "declared memory cap exceeded")
        _need(value["root_logical_bytes"] <= storage.DEFAULT_BUDGET - storage.DEFAULT_FAILURE_RESERVE,
              "declared normal root cap/failure reserve exceeded")
        if phase == "audit":
            _need(value["peak_cuda_allocated_bytes"] == value["peak_cuda_reserved_bytes"] == 0,
                  "CPU audit declares CUDA memory")


def validate_record(record, *, evidence_kind):
    _keys(record, RECORD_KEYS, "phase record")
    _need(type(evidence_kind) is str and evidence_kind in KINDS, "expected evidence kind")
    _need(record["schema"] == "i7_native_phase_record_v1" and type(record["schema"]) is str
          and record["profile"] == PROFILE and type(record["profile"]) is str
          and record["evidence_kind"] == evidence_kind and type(record["evidence_kind"]) is str
          and record["execution_enabled"] is False, "record profile/evidence/authority")
    _int(record["sequence"], 0, MAX_RECORDS-1, "record sequence")
    for key in ("event", "operation", "phase"):
        _need(type(record[key]) is str, key + ": string required")
    _need(record["event"] in ("decision", "started", "sealed", "boundary", "failed"), "unknown event")
    operation = next((row for row in schedule() if row["operation"] == record["operation"]), None)
    _need(operation is not None and record["phase"] == operation["phase"], "operation membership/phase")
    events = ("started", "sealed") if operation["kind"] == "work" else (operation["kind"],)
    _need(record["event"] == "failed" or record["event"] in events, "event incompatible with operation kind")
    codec.validate_created_utc(record["created_utc"])
    _hash(record["sources_sha256"], "sources")
    _hash(record["environment_sha256"], "environment")
    if record["previous_record_sha256"] is not None:
        _hash(record["previous_record_sha256"], "previous record")
    if record["operation"] == "development_go":
        _need(record["root_binding"] is None and record["resources"] is None,
              "development GO must precede root creation")
    else:
        _root(record["root_binding"])
        _resources(record["resources"], record["phase"], failed=record["event"] == "failed")
    _need(type(record["artifacts"]) is list, "artifact list")
    expected = operation["artifacts"] if record["event"] == "sealed" else []
    _need(len(record["artifacts"]) == len(expected), "artifact count")
    for ref, name in zip(record["artifacts"], expected):
        validate_reference(ref, name=name, encoding="bytes" if name.endswith(".json") else "torch_weights_only")
    if record["event"] == "failed":
        _keys(record["error"], ("category", "retained_failure_ref"), "failure")
        _need(type(record["error"]["category"]) is str and record["error"]["category"] in
              ("resource", "invariant", "io", "interrupted", "other"), "failure category")
        # None explicitly means failure metadata could not be retained. It is
        # never silently interpreted as a successful normal artifact write.
        ref = record["error"]["retained_failure_ref"]
        if ref is not None:
            _keys(ref, ("name", "size_bytes", "sha256"), "failure file reference")
            _need(type(ref["name"]) is str and re.fullmatch(r"failure-[0-9]{6}\.json", ref["name"]), "failure name")
            _int(ref["size_bytes"], 1, storage._FAILURE_MAX, "failure size")
            _hash(ref["sha256"], "failure hash")
    else:
        _need(record["error"] is None, "successful record carries failure")
    raw = codec.json_bytes(record)
    _need(len(raw) <= MAX_RECORD_BYTES, "record byte bound")
    return record


def validate_transcript(records, *, evidence_kind, require_complete=True):
    """Validate only declared order/bindings/caps; never authenticate or execute."""
    _need(type(records) is list and 0 < len(records) <= MAX_RECORDS, "bounded nonempty record list required")
    _need(type(require_complete) is bool, "require_complete exact boolean")
    program = schedule()
    cursor, awaiting_seal, terminal = 0, False, False
    previous_digest, root, sources = None, None, None
    environments, latest_resources, seen_artifacts = {}, {}, set()
    cumulative_bytes = cumulative_allocated = 0
    retained_payload_receipt_bytes = 0
    for index, record in enumerate(records):
        validate_record(record, evidence_kind=evidence_kind)
        _need(not terminal and cursor < len(program), "record follows terminal/completed transcript")
        row = program[cursor]
        _need(record["sequence"] == index and record["operation"] == row["operation"], "transcript sequence/order")
        _need(record["previous_record_sha256"] == previous_digest, "previous record hash differs")
        if sources is None:
            sources = record["sources_sha256"]
        _need(record["sources_sha256"] == sources, "sources changed across phases")
        phase = record["phase"]
        environments.setdefault(phase, record["environment_sha256"])
        _need(record["environment_sha256"] == environments[phase], "environment changed inside phase")
        if phase == "sensitivity":
            _need(record["environment_sha256"] == environments.get("primary"),
                  "sensitivity environment differs from primary")
        prior_root_bytes = cumulative_bytes
        if record["root_binding"] is not None:
            if root is None:
                root = copy.deepcopy(record["root_binding"])
            _need(codec.tree_digest(record["root_binding"]) == codec.tree_digest(root), "root identity changed")
            resources = record["resources"]
            _need(resources["root_logical_bytes"] >= cumulative_bytes
                  and resources["root_allocated_bytes"] >= cumulative_allocated,
                  "declared retained root usage decreased")
            cumulative_bytes, cumulative_allocated = resources["root_logical_bytes"], resources["root_allocated_bytes"]
            prior = latest_resources.get(phase)
            if prior is not None:
                _need(all(resources[key] >= prior[key] for key in RESOURCE_KEYS), "declared phase clock/peak reset")
            latest_resources[phase] = copy.deepcopy(resources)
        event = record["event"]
        if event == "failed":
            failure_ref = record["error"]["retained_failure_ref"]
            if failure_ref is not None:
                _need(record["resources"] is not None,
                      "pre-root failure cannot declare retained root file")
                retained_payload_receipt_bytes += failure_ref["size_bytes"]
                _need(record["resources"]["root_logical_bytes"] >=
                      max(retained_payload_receipt_bytes, prior_root_bytes + failure_ref["size_bytes"]),
                      "declared root omits retained failure bytes")
            terminal = True
        elif row["kind"] == "work":
            _need(event == ("sealed" if awaiting_seal else "started"), "work must start before sealing")
            if awaiting_seal:
                for ref in record["artifacts"]:
                    _need(ref["name"] not in seen_artifacts, "duplicate output artifact")
                    seen_artifacts.add(ref["name"])
                    retained_payload_receipt_bytes += ref["size_bytes"] + ref["receipt_size_bytes"]
                _need(record["resources"]["root_logical_bytes"] >= retained_payload_receipt_bytes,
                      "declared root smaller than retained payload/receipt lower bound")
                cursor += 1
            awaiting_seal = not awaiting_seal
        else:
            _need(not awaiting_seal and event == row["kind"], "decision/boundary event differs")
            cursor += 1
        import hashlib
        previous_digest = hashlib.sha256(codec.json_bytes(record)).hexdigest()
    complete = cursor == len(program) and not terminal and not awaiting_seal
    _need(not require_complete or complete, "transcript is not complete")
    return dict(schema="i7_native_policy_validation_v1", evidence_kind=evidence_kind,
        status="declared_complete" if complete else "declared_failed" if terminal else "declared_incomplete",
        records=len(records), completed_operations=cursor, artifact_count=len(seen_artifacts),
        phase_resources=latest_resources, final_record_sha256=previous_digest,
        pending_operation=None if complete or terminal else program[cursor]["operation"],
        interrupted_work=awaiting_seal, can_execute=False, can_resume=False,
        verified_file_bytes=False, verified_resource_observations=False,
        verified_execution_history=False, scientific_execution_certified=False)
