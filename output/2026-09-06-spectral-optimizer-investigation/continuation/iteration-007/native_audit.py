#!/usr/bin/env python3
"""Explicit sealed-input CPU audit adapter for iteration 007.

Import and default execution are inert.  ``run_audit`` admits only the frozen
16 scientific anchors or the one registered MLP fixture identity, independently
loads the owning plan and actual stored anchor/witness/branch records, and then
delegates numerical work and create-only sealing to ``audit_and_seal``.
"""
from __future__ import annotations


class NativeAuditError(RuntimeError):
    """The CPU audit adapter failed its fixed input or execution contract."""


RESULT_KEYS = ("artifact", "receipt")
SCIENTIFIC_MEMBERSHIP = tuple(
    (role, bundle, update)
    for role, bundles in (("primary", (71001, 71002, 71003)),
                          ("sensitivity", (71901,)))
    for bundle in bundles for update in (101, 500, 1000, 2000))


def _need(condition, message):
    if not condition:
        raise NativeAuditError(message)


def _check(guard, stage):
    _need(callable(guard), "cooperative guard(stage) is required")
    _need(type(stage) is str and stage.isascii() and 0 < len(stage) <= 96,
          "guard stage must be bounded ASCII")
    guard(stage)


def _admit_identity(identity, profile):
    import artifact_store as storage
    import identity_codec as codec
    import source_history

    codec.validate_identity(identity, profile=profile)
    track = source_history.trajectory(identity, profile)
    _need(identity["anchor_update"] in track["anchor_updates"],
          "identity is not a registered source anchor")
    if profile == storage.SCIENTIFIC:
        member = (identity["execution_role"], identity["bundle"],
                  identity["anchor_update"])
        _need(member in SCIENTIFIC_MEMBERSHIP,
              "scientific audit identity is outside the frozen 16 anchors")
    else:
        _need(profile == storage.MLP_FIXTURE,
              "unsupported audit profile")
    return track


def _verified_context(*, store, identity, profile, plan, plan_reference,
                      plan_artifact, images_bytes, labels_bytes, expected_files,
                      sources, environment, auditor_environment, created_utc, guard):
    import os
    _need(os.environ.get("CUDA_VISIBLE_DEVICES") == "",
          "CPU audit requires CUDA hidden before Torch import")
    import torch
    import anchor_envelope as envelopes
    import artifact_store as storage
    import identity_codec as codec
    import plan_bindings as plans
    import source_environment as provenance
    import source_history
    import verified_plan_load as verified

    _check(guard, "native_audit.entry")
    _need(not torch.cuda.is_initialized(), "CPU audit requires uninitialized CUDA")
    _need(type(store) is storage.ArtifactStore and not store._closed and
          not store._terminal and store.profile == profile,
          "exact writable profile-matched ArtifactStore required")
    _need(profile in (storage.SCIENTIFIC, storage.MLP_FIXTURE),
          "unsupported audit profile")
    codec.validate_created_utc(created_utc)
    track = _admit_identity(identity, profile)
    _check(guard, "native_audit.context.pre_provenance")
    provenance.validate_sources(sources, profile=profile)
    provenance.validate_environment(environment, profile=profile)
    provenance.validate_environment(auditor_environment, profile=profile)
    source_role = "native_source" if profile == storage.SCIENTIFIC else "fixture_cpu"
    audit_role = "cpu_audit" if profile == storage.SCIENTIFIC else "fixture_cpu"
    _need(environment["runtime_role"] == source_role and
          auditor_environment["runtime_role"] == audit_role and
          sources["repository_root_realpath"] == environment["repository_root_realpath"],
          "original source environment differs from audit context")
    _need(auditor_environment["repository_root_realpath"] ==
          sources["repository_root_realpath"],
          "auditor environment repository root differs")
    _check(guard, "native_audit.context.post_provenance")
    _check(guard, "native_audit.context.pre_fresh_auditor_environment")
    fresh_auditor = provenance.collect_runtime_environment(
        sources["repository_root_realpath"], profile=profile, runtime_role=audit_role)
    _check(guard, "native_audit.context.post_fresh_auditor_environment")
    _need(envelopes.same_exact(fresh_auditor, auditor_environment),
          "fresh CPU-audit environment differs from phase binding")
    _check(guard, "native_audit.context.pre_plan_validation")
    plans.validate_plan(plan, identity=identity, profile=profile)
    source_history._validate_plan_ref(plan_reference, profile, track)
    _check(guard, "native_audit.context.pre_plan_load")
    loaded = verified.load_verified_plan_from_store(
        store, plan_reference["name"], identity=identity, profile=profile,
        expected_sha256=plan_reference["sha256"])
    _check(guard, "native_audit.context.post_plan_load")
    expected_artifact = {
        "sha256": loaded["artifact"]["sha256"],
        "size_bytes": loaded["artifact"]["size_bytes"],
    }
    _check(guard, "native_audit.context.pre_plan_comparison")
    _need(envelopes.same_exact(loaded["artifact"], plan_reference),
          "plan reference differs from independently verified receipt bytes")
    _need(type(plan_artifact) is dict and
          tuple(plan_artifact) == ("sha256", "size_bytes") and
          envelopes.same_exact(plan_artifact, expected_artifact),
          "plan artifact differs from independently verified plan bytes")
    _need(envelopes.same_exact(plan, loaded["plan"]),
          "supplied plan differs from independently verified plan bytes")
    _check(guard, "native_audit.context.post_plan_comparison")
    return {
        "identity": identity, "profile": profile, "plan": loaded["plan"],
        "plan_artifact": expected_artifact, "images_bytes": images_bytes,
        "labels_bytes": labels_bytes, "expected_files": expected_files,
        "sources": sources, "environment": environment,
    }, auditor_environment


def _receipt_from_row(name, row):
    import hashlib
    import source_capture

    _need(type(row) is dict and
          tuple(row) == ("encoding", "name", "schema", "sha256", "size", "status"),
          "stored receipt row has unexpected schema")
    receipt = {key: row[key] for key in source_capture.RECEIPT_KEYS[:-1]}
    receipt["receipt_name"] = (
        "receipt-" + hashlib.sha256(name.encode("ascii")).hexdigest() + ".json")
    return receipt


def _load_transaction(store, context, guard):
    import artifact_store as storage
    import branch_execution as execution
    import identity_codec as codec

    _check(guard, "native_audit.transaction.pre_inventory")
    report, scanned = storage._inspect_dirfd(store._dirfd)
    _need(report["header"]["profile"] == context["profile"] and
          not report["terminal"], "transaction store is terminal or has wrong profile")
    _check(guard, "native_audit.transaction.post_inventory")
    result = {}
    for kind, key, schema in (
            ("anchor", "anchor", "i7_anchor"),
            ("source-witness", "witness", "i7_source_step_witness"),
            ("branch-results", "branch", "i7_branch_results")):
        name = codec.artifact_id(
            context["identity"], profile=context["profile"], kind=kind) + ".pt"
        rows = [row for row in report["receipts"] if row["name"] == name]
        _need(len(rows) == 1, "missing or ambiguous sealed " + kind)
        receipt = _receipt_from_row(name, rows[0])
        _need(name in scanned and receipt["receipt_name"] in scanned,
              "sealed transaction file or receipt is absent")
        _check(guard, "native_audit.transaction." + key + ".preload")
        value = storage.ArtifactStore.load_tensor_tree(
            store.root, name, expected_size=receipt["size"],
            expected_sha256=receipt["sha256"])
        value = execution._load_pinned(store, value, receipt, schema_name=schema)
        _check(guard, "native_audit.transaction." + key + ".postload")
        _need(value["identity"] == context["identity"] and
              value["artifact_id"] + ".pt" == name,
              "loaded transaction identity differs")
        result[key], result[key + "_receipt"] = value, receipt
    return result


def _restore_caller_rng(caller_rng, guard):
    import anchor_envelope as envelopes
    import state_core as state

    _check(guard, "native_audit.caller_rng.pre_verify")
    _need(envelopes.same_exact(state._raw_rng_state(), caller_rng),
          "auditor returned with changed caller RNG")
    _check(guard, "native_audit.caller_rng.pre_restore")
    state._set_rng_state(caller_rng)
    _need(envelopes.same_exact(state._raw_rng_state(), caller_rng),
          "audit adapter failed to restore caller RNG")
    _check(guard, "native_audit.caller_rng.post_restore")


def run_audit(*, store, identity, profile, plan, plan_reference, plan_artifact,
              images_bytes, labels_bytes, expected_files, sources, environment,
              auditor_environment, created_utc, guard):
    """Audit and seal one actual stored branch result in a CPU-only process."""
    import os
    _need(os.environ.get("CUDA_VISIBLE_DEVICES") == "",
          "CPU audit requires CUDA hidden before Torch import")
    import torch
    _need(not torch.cuda.is_initialized(), "CPU audit requires uninitialized CUDA")
    import anchor_envelope as envelopes
    import artifact_store as storage
    import audit_envelope as audit
    import state_core as state

    valid_store = type(store) is storage.ArtifactStore
    caller_rng = None
    try:
        _check(guard, "native_audit.pre_caller_rng_capture")
        caller_rng = state._raw_rng_state()
        _check(guard, "native_audit.post_caller_rng_capture")
        context, expected_auditor_environment = _verified_context(
            store=store, identity=identity, profile=profile, plan=plan,
            plan_reference=plan_reference, plan_artifact=plan_artifact,
            images_bytes=images_bytes, labels_bytes=labels_bytes,
            expected_files=expected_files, sources=sources,
            environment=environment, auditor_environment=auditor_environment,
            created_utc=created_utc, guard=guard)
        transaction = _load_transaction(store, context, guard)
        _check(guard, "native_audit.execute.pre")
        produced = audit.audit_and_seal(
            store=store, plan_root=store.root, plan_name=plan_reference["name"],
            created_utc=created_utc, guard=guard, **transaction, **context)
        _check(guard, "native_audit.execute.post")
        _need(type(produced) is dict and tuple(produced) == RESULT_KEYS,
              "audit producer returned unexpected fields")
        artifact, receipt = produced["artifact"], produced["receipt"]
        _check(guard, "native_audit.result.pre_binding")
        envelopes.validate_common(
            artifact, schema_name="i7_anchor_numerical_audit",
            kind="independent-audit", identity=identity, profile=profile)
        _need(receipt["name"] == artifact["artifact_id"] + ".pt",
              "audit artifact and actual receipt differ")
        _need(envelopes.same_exact(
            artifact["payload"]["exact_validation"]["auditor_environment"],
            expected_auditor_environment),
            "retained auditor environment differs from phase binding")
        failed = artifact["payload"]["overall_status"] != "pass"
        _need(store._terminal is failed,
              "audit result/store terminal status differs")
        _check(guard, "native_audit.result.post_binding")
        _restore_caller_rng(caller_rng, guard)
        return {"artifact": artifact, "receipt": receipt}
    except BaseException as exc:
        if caller_rng is not None:
            try:
                state._set_rng_state(caller_rng)
            except BaseException:
                pass
        if valid_store and not store._closed and not store._terminal:
            store._fail("native_audit_failed", type(exc).__name__)
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        if isinstance(exc, NativeAuditError):
            raise
        raise NativeAuditError("native audit failed: " + type(exc).__name__) from exc


if __name__ == "__main__":
    print("native_audit: inert; no audit or scientific action")
