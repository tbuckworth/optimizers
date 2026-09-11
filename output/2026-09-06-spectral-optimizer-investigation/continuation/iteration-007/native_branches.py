#!/usr/bin/env python3
"""Sealed-input branch adapter for iteration 007; import/default are inert.

Explicit ``run_branches`` calls bind one registered anchor identity to an
independently verified plan and the actual anchor/witness files already sealed
in the supplied store.  Numerical construction, validation and the create-only
branch-result write remain owned by ``branch_execution.execute_branches``.
This module grants no GO authority and performs no source, audit or data load.
"""
from __future__ import annotations


class NativeBranchesError(RuntimeError):
    """The sealed branch adapter failed its fixed input or execution contract."""


RESULT_KEYS = ("artifact", "receipt")


def _need(condition, message):
    if not condition:
        raise NativeBranchesError(message)


def _check(guard, stage):
    _need(callable(guard), "cooperative guard(stage) is required")
    _need(type(stage) is str and stage.isascii(), "guard stage must be ASCII")
    guard(stage)


def _verified_context(*, store, identity, profile, native_device, plan,
                      plan_reference, plan_artifact, images_bytes, labels_bytes,
                      expected_files, sources, environment, created_utc, guard):
    import anchor_envelope as envelopes
    import artifact_store as storage
    import identity_codec as codec
    import plan_bindings as plans
    import runtime_guard as runtime
    import source_capture
    import source_history
    import torch
    import verified_plan_load as verified

    _check(guard, "native_branches.entry")
    _need(type(store) is storage.ArtifactStore and not store._closed and
          not store._terminal and store.profile == profile,
          "exact writable profile-matched ArtifactStore required")
    _need(type(profile) is str and profile in
          (storage.SCIENTIFIC, storage.MLP_FIXTURE),
          "unsupported branch profile")
    codec.validate_identity(identity, profile=profile)
    codec.validate_created_utc(created_utc)
    track = source_history.trajectory(identity, profile)
    _need(identity["anchor_update"] in track["anchor_updates"],
          "identity is not a registered source anchor")
    if profile == storage.MLP_FIXTURE:
        expected_device = "cpu"
    else:
        _need(environment["cuda"]["current_device"] == runtime.CUDA_DEVICE_INDEX,
              "environment CUDA index differs from fixed runtime device")
        expected_device = "cuda:" + str(runtime.CUDA_DEVICE_INDEX)
    _need(type(native_device) is str and native_device == expected_device and
          str(torch.device(native_device)) == native_device,
          "native device must be the fixed canonical device")

    _check(guard, "native_branches.context.pre_plan_validation")
    plans.validate_plan(plan, identity=identity, profile=profile)
    source_history._validate_plan_ref(plan_reference, profile, track)
    _check(guard, "native_branches.context.pre_verified_plan_load")
    loaded = verified.load_verified_plan_from_store(
        store, plan_reference["name"], identity=identity, profile=profile,
        expected_sha256=plan_reference["sha256"])
    _check(guard, "native_branches.context.post_verified_plan_load")
    _check(guard, "native_branches.context.pre_plan_comparison")
    expected_artifact = {
        "sha256": loaded["artifact"]["sha256"],
        "size_bytes": loaded["artifact"]["size_bytes"],
    }
    _need(envelopes.same_exact(loaded["artifact"], plan_reference),
          "plan reference differs from independently verified receipt bytes")
    _need(type(plan_artifact) is dict and
          tuple(plan_artifact) == ("sha256", "size_bytes") and
          envelopes.same_exact(plan_artifact, expected_artifact),
          "plan artifact differs from independently verified plan bytes")
    _need(envelopes.same_exact(plan, loaded["plan"]),
          "supplied plan differs from independently verified plan bytes")
    _check(guard, "native_branches.context.post_plan_comparison")
    _check(guard, "native_branches.context.pre_provenance")
    source_capture._verify_live_context(
        sources, environment, profile, include_sources=True)
    _check(guard, "native_branches.context.post_provenance")
    return {
        "identity": identity, "profile": profile, "plan": loaded["plan"],
        "plan_artifact": expected_artifact, "images_bytes": images_bytes,
        "labels_bytes": labels_bytes, "expected_files": expected_files,
        "sources": sources, "environment": environment,
    }


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
    import identity_codec as codec

    _check(guard, "native_branches.transaction.pre_inventory")
    report, scanned = storage._inspect_dirfd(store._dirfd)
    _need(report["header"]["profile"] == context["profile"] and
          not report["terminal"], "transaction store is terminal or has wrong profile")
    _check(guard, "native_branches.transaction.post_inventory")
    result = {}
    for kind, key, schema in (
            ("anchor", "anchor", "i7_anchor"),
            ("source-witness", "witness", "i7_source_step_witness")):
        name = codec.artifact_id(
            context["identity"], profile=context["profile"], kind=kind) + ".pt"
        rows = [row for row in report["receipts"] if row["name"] == name]
        _need(len(rows) == 1, "missing or ambiguous sealed " + kind)
        receipt = _receipt_from_row(name, rows[0])
        _need(name in scanned and receipt["receipt_name"] in scanned,
              "sealed transaction file or receipt is absent")
        _check(guard, "native_branches.transaction." + kind + ".preload")
        value = storage.ArtifactStore.load_tensor_tree(
            store.root, name, expected_size=receipt["size"],
            expected_sha256=receipt["sha256"])
        _check(guard, "native_branches.transaction." + kind + ".postload")
        _need(type(value) is dict and value.get("schema_name") == schema and
              value.get("artifact_id") + ".pt" == name,
              "loaded transaction identity differs")
        # execute_branches will reread these values through its held-descriptor
        # _load_pinned path before using them; they are never caller artifacts.
        result[key] = value
        result[key + "_receipt"] = receipt
    return result


def _restore_caller_rng(caller_rng, guard):
    import anchor_envelope as envelopes
    import state_core as state

    _check(guard, "native_branches.caller_rng.pre_verify")
    _need(envelopes.same_exact(state._raw_rng_state(), caller_rng),
          "branch producer returned with changed caller RNG")
    _check(guard, "native_branches.caller_rng.pre_restore")
    state._set_rng_state(caller_rng)
    _need(envelopes.same_exact(state._raw_rng_state(), caller_rng),
          "branch adapter failed to restore caller RNG")
    _check(guard, "native_branches.caller_rng.post_restore")


def run_branches(*, store, identity, profile, native_device, plan,
                 plan_reference, plan_artifact, images_bytes, labels_bytes,
                 expected_files, sources, environment, created_utc, guard):
    """Execute and seal one branch-result artifact from actual stored inputs."""
    import artifact_store as storage
    import branch_execution as execution
    import state_core as state

    valid_store = type(store) is storage.ArtifactStore
    caller_rng = None
    try:
        _check(guard, "native_branches.pre_caller_rng_capture")
        caller_rng = state._raw_rng_state()
        _check(guard, "native_branches.post_caller_rng_capture")
        context = _verified_context(
            store=store, identity=identity, profile=profile,
            native_device=native_device, plan=plan,
            plan_reference=plan_reference, plan_artifact=plan_artifact,
            images_bytes=images_bytes, labels_bytes=labels_bytes,
            expected_files=expected_files, sources=sources,
            environment=environment, created_utc=created_utc, guard=guard)
        transaction = _load_transaction(store, context, guard)
        _check(guard, "native_branches.execute.pre")
        produced = execution.execute_branches(
            store=store, created_utc=created_utc, guard=guard,
            **transaction, **context)
        _check(guard, "native_branches.execute.post")
        _need(type(produced) is dict and tuple(produced) == execution.RETURN_KEYS,
              "branch producer returned unexpected fields")
        _check(guard, "native_branches.result.pre_binding")
        _need(produced["receipt"]["name"] ==
              produced["artifact"]["artifact_id"] + ".pt",
              "branch artifact and actual receipt differ")
        _check(guard, "native_branches.result.post_binding")
        _restore_caller_rng(caller_rng, guard)
        return {"artifact": produced["artifact"], "receipt": produced["receipt"]}
    except BaseException as exc:
        if caller_rng is not None:
            try:
                state._set_rng_state(caller_rng)
            except BaseException:
                pass
        if valid_store and not store._closed and not store._terminal:
            store._fail("native_branches_failed", type(exc).__name__)
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        if isinstance(exc, NativeBranchesError):
            raise
        raise NativeBranchesError(
            "native branches failed: " + type(exc).__name__) from exc


if __name__ == "__main__":
    print("native_branches: inert; no source, branch, audit or scientific action")
