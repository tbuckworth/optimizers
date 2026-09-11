"""Seal one anchor, then retain its uninterrupted actual live source step.

This is the bounded anchor/source-witness transaction only.  It does not replay
branches, run a scientific phase controller, or establish acquisition history.
"""
from __future__ import annotations

import hashlib
import math

import torch
import torch.nn.functional as F

import anchor_envelope as envelopes
import artifact_store as storage
import data_probe_bindings as data
import identity_codec as codec
import source_environment as provenance
import state_core as state


class SourceCaptureError(RuntimeError):
    pass


ROOT_KEYS = envelopes.ROOT_KEYS
PAYLOAD_KEYS = (
    "anchor_ref", "source_step", "capture_proof", "before_parameters_binding",
    "raw_gradient", "delivered_current_gradient", "parameters_after",
    "optimizer_after", "observer_after", "live_loss", "state_hashes",
)
REFERENCE_KEYS = (
    "artifact_id", "schema_name", "name", "size_bytes", "sha256", "status",
    "encoding", "receipt_name", "receipt_size_bytes", "receipt_sha256",
)
STEP_KEYS = (
    "anchor_update", "completed_updates_before", "completed_observations_before",
    "completed_updates_after", "completed_observations_after", "source_state_origin",
    "anchor_deserializations_before_live_step", "observation_ingests_in_live_step",
    "optimizer_steps_in_live_step",
)
PROOF_KEYS = (
    "proof_id", "ordered_events", "anchor_deserializations_before_live_step",
    "core_sha256_before", "core_sha256_after", "core_direct_typed_equal",
    "same_live_object_bindings", "anchor_sealed_before_live_step",
    "live_step_complete_at_payload_assembly",
)
BEFORE_BINDING_KEYS = (
    "anchor_artifact_id", "anchor_model_parameters_tree_sha256",
    "parameter_order_tree_sha256",
)
LOSS_KEYS = ("probe_key", "count", "value_native_float32", "sample_identity_sha256")
STATE_HASH_KEYS = (
    "parameters_before", "parameters_after", "raw_gradient",
    "delivered_current_gradient", "optimizer_after", "observer_after",
    "rng_before", "rng_after",
)
RECEIPT_KEYS = ("schema", "name", "size", "sha256", "status", "encoding", "receipt_name")
EVENTS = [
    "anchor_core_captured_and_validated",
    "anchor_envelope_sealed",
    "same_live_objects_recaptured",
    "complete_core_directly_equal",
    "actual_live_forward_and_backward",
    "actual_live_observer_ingest",
    "actual_live_optimizer_step",
    "source_witness_payload_assembled",
]
PROFILES = (storage.SCIENTIFIC, storage.MLP_FIXTURE)


def _require(condition, message):
    if not condition:
        raise SourceCaptureError(message)


def _guard(guard, stage):
    if guard is not None:
        guard(stage)


def _keys(value, expected, where):
    _require(type(value) is dict and tuple(value) == tuple(expected)
             and all(type(key) is str for key in value),
             where + ": exact ordered keys required")


def _receipt_reference(anchor, receipt, *, receipt_bytes=None):
    _keys(receipt, RECEIPT_KEYS, "anchor receipt")
    expected_name = anchor["artifact_id"] + ".pt"
    _require(all(type(receipt[key]) is str for key in
                 ("schema", "name", "sha256", "status", "encoding", "receipt_name"))
             and receipt["schema"] == "i7_artifact_receipt_v1"
             and receipt["name"] == expected_name
             and type(receipt["size"]) is int
             and receipt["size"] > 0
             and type(receipt["sha256"]) is str and len(receipt["sha256"]) == 64
             and all(character in "0123456789abcdef" for character in receipt["sha256"])
             and receipt["status"] == "complete"
             and receipt["encoding"] == "torch_weights_only"
             and type(receipt["receipt_name"]) is str,
             "anchor receipt is not a complete tensor-tree receipt")
    expected_receipt = "receipt-" + hashlib.sha256(expected_name.encode("ascii")).hexdigest() + ".json"
    _require(receipt["receipt_name"] == expected_receipt, "anchor receipt name differs")
    base = {key: receipt[key] for key in RECEIPT_KEYS[:-1]}
    encoded = storage._json_bytes(base)
    if receipt_bytes is not None:
        _require(type(receipt_bytes) is bytes and receipt_bytes == encoded,
                 "anchor receipt bytes differ from canonical complete receipt")
    return {
        "artifact_id": anchor["artifact_id"], "schema_name": "i7_anchor",
        "name": expected_name, "size_bytes": receipt["size"], "sha256": receipt["sha256"],
        "status": "complete", "encoding": "torch_weights_only",
        "receipt_name": receipt["receipt_name"], "receipt_size_bytes": len(encoded),
        "receipt_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _sealed_anchor_reference(store, anchor, receipt):
    """Inspect the held writer directory and receipt bytes without decoding anchor."""
    report, scanned = storage._inspect_dirfd(store._dirfd)
    _require(report["header"]["profile"] == anchor["profile"] and not report["terminal"],
             "anchor store is terminal or has the wrong profile")
    rows = [row for row in report["receipts"] if row["name"] == receipt["name"]]
    _require(len(rows) == 1, "sealed anchor has no unique complete receipt")
    _require(rows[0] == {key: receipt[key] for key in RECEIPT_KEYS[:-1]},
             "held-descriptor receipt differs from writer receipt")
    receipt_name = receipt["receipt_name"]
    _require(receipt_name in scanned, "sealed anchor receipt file missing")
    receipt_bytes, _ = storage._read_regular(store._dirfd, receipt_name,
        maximum=storage._RECEIPT_MAX, expected=scanned[receipt_name])
    return _receipt_reference(anchor, receipt, receipt_bytes=receipt_bytes)


def _same_live_bindings(model, optimizer, observer, parameter_refs):
    current = list(model.parameters())
    groups = optimizer.param_groups
    return (len(current) == len(parameter_refs)
            and all(left is right for left, right in zip(current, parameter_refs))
            and len(groups) == 1 and len(groups[0]["params"]) == len(parameter_refs)
            and all(left is right for left, right in zip(groups[0]["params"], parameter_refs))
            and observer.model is model and observer.base_optimizer is optimizer
            and type(observer.param_list) is list and len(observer.param_list) == len(parameter_refs)
            and all(left is right for left, right in zip(observer.param_list, parameter_refs)))


def _verify_live_context(sources, environment, profile, *, include_sources):
    """Recollect current process/repository facts; expected snapshots are not authority."""
    provenance.validate_sources(sources, profile=profile)
    provenance.validate_environment(environment, profile=profile)
    root = sources["repository_root_realpath"]
    _require(type(root) is str and environment["repository_root_realpath"] == root
             and type(environment["runtime_role"]) is str,
             "source/environment live-collection context differs")
    if include_sources:
        actual_sources = provenance.collect_verified_sources(root, profile=profile)
        _require(envelopes.same_exact(actual_sources, sources),
                 "current source snapshot differs from expected")
    actual_environment = provenance.collect_runtime_environment(
        root, profile=profile, runtime_role=environment["runtime_role"])
    _require(envelopes.same_exact(actual_environment, environment),
             "current runtime environment differs from expected")


def _gradient_record(model, *, profile, where):
    values = []
    for name, parameter in model.named_parameters():
        gradient = parameter.grad
        _require(type(gradient) is torch.Tensor and gradient.shape == parameter.shape
                 and gradient.dtype == parameter.dtype and gradient.device == parameter.device
                 and gradient.layout == torch.strided and not gradient.is_sparse
                 and bool(torch.isfinite(gradient).all()), where + ": invalid gradient for " + name)
        values.append(gradient.detach().reshape(-1))
    result = state._native_tensor(torch.cat(values))
    state._validate_native(result, where, (state.PROFILE[profile]["n_params"],),
                           "torch.float32", state.PROFILE[profile]["device_kind"])
    return result


def _sample_hash(inputs, labels):
    input_bytes = inputs.detach().cpu().contiguous().numpy().astype("<f4", copy=False).tobytes(order="C")
    label_bytes = labels.detach().cpu().contiguous().numpy().astype("<i8", copy=False).tobytes(order="C")
    return hashlib.sha256(input_bytes + label_bytes).hexdigest()


def _endpoint_core(anchor, payload):
    identity = anchor["identity"]
    model = dict(anchor["payload"]["model"])
    model["parameters"] = payload["parameters_after"]
    return {
        "schema_name": "i7_state_core", "schema_version": 1, "profile": anchor["profile"],
        "phase": identity["anchor_phase"], "anchor_update": identity["anchor_update"] + 1,
        "state_completed_updates": identity["anchor_update"], "model": model,
        "optimizer": payload["optimizer_after"], "observer": payload["observer_after"],
        "rng": anchor["payload"]["rng"],
    }


def _validate_witness_impl(value, *, anchor, anchor_receipt, identity, profile, plan,
                           plan_artifact, images_bytes, labels_bytes, expected_files,
                           sources, environment):
    envelopes.validate_anchor(anchor, identity=identity, profile=profile, plan=plan,
        plan_artifact=plan_artifact, images_bytes=images_bytes, labels_bytes=labels_bytes,
        expected_files=expected_files, sources=sources, environment=environment)
    envelopes.validate_common(value, schema_name="i7_source_step_witness", kind="source-witness",
                              identity=identity, profile=profile)
    _keys(value["payload"], PAYLOAD_KEYS, "source witness payload")
    payload, update = value["payload"], identity["anchor_update"]
    expected_ref = _receipt_reference(anchor, anchor_receipt)
    _keys(payload["anchor_ref"], REFERENCE_KEYS, "anchor reference")
    _require(envelopes.same_exact(payload["anchor_ref"], expected_ref), "anchor reference differs")

    _keys(payload["source_step"], STEP_KEYS, "source step")
    expected_step = {
        "anchor_update": update, "completed_updates_before": update - 1,
        "completed_observations_before": update - 1, "completed_updates_after": update,
        "completed_observations_after": update,
        "source_state_origin": "uninterrupted_live_objects_v1",
        "anchor_deserializations_before_live_step": 0,
        "observation_ingests_in_live_step": 1, "optimizer_steps_in_live_step": 1,
    }
    _require(envelopes.same_exact(payload["source_step"], expected_step), "source-step values differ")

    anchor_core = envelopes.core_from_anchor(anchor)
    state.validate_core(anchor_core)
    anchor_core_hash = codec.tree_digest(anchor_core)
    _keys(payload["capture_proof"], PROOF_KEYS, "capture proof")
    expected_proof = {
        "proof_id": "capture_neutrality_v1", "ordered_events": list(EVENTS),
        "anchor_deserializations_before_live_step": 0,
        "core_sha256_before": anchor_core_hash, "core_sha256_after": anchor_core_hash,
        "core_direct_typed_equal": True, "same_live_object_bindings": True,
        "anchor_sealed_before_live_step": True,
        "live_step_complete_at_payload_assembly": True,
    }
    _require(envelopes.same_exact(payload["capture_proof"], expected_proof), "capture proof differs")

    _keys(payload["before_parameters_binding"], BEFORE_BINDING_KEYS, "before binding")
    before_parameters = anchor["payload"]["model"]["parameters"]
    parameter_order = anchor["payload"]["model"]["parameter_order"]
    expected_before = {
        "anchor_artifact_id": anchor["artifact_id"],
        "anchor_model_parameters_tree_sha256": codec.tree_digest(before_parameters),
        "parameter_order_tree_sha256": codec.tree_digest(parameter_order),
    }
    _require(envelopes.same_exact(payload["before_parameters_binding"], expected_before),
             "before-parameter binding differs")

    spec = state.PROFILE[profile]
    native_device = before_parameters[0]["native_device"]
    for key in ("raw_gradient", "delivered_current_gradient"):
        state._validate_native(payload[key], key, (spec["n_params"],), "torch.float32",
                               spec["device_kind"])
        _require(payload[key]["native_device"] == native_device, key + " device differs from anchor")

    endpoint = _endpoint_core(anchor, payload)
    state.validate_core(endpoint)
    for before, after in zip(before_parameters, payload["parameters_after"]):
        _require(after["native_device"] == before["native_device"],
                 "post-step parameter device differs from anchor")

    materialized = data.materialize(images_bytes, labels_bytes, plan=plan, identity=identity,
                                    profile=profile, expected_files=expected_files)
    batch = materialized["probes"]["batch_noisy"]
    _keys(payload["live_loss"], LOSS_KEYS, "live loss")
    expected_count = 64 if profile == storage.SCIENTIFIC else 4
    loss = payload["live_loss"]
    _require(loss["probe_key"] == "batch_noisy" and type(loss["count"]) is int
             and loss["count"] == expected_count and type(loss["value_native_float32"]) is float
             and math.isfinite(loss["value_native_float32"])
             and float(torch.tensor(loss["value_native_float32"], dtype=torch.float32).item())
                 == loss["value_native_float32"]
             and loss["sample_identity_sha256"] == _sample_hash(batch["inputs"], batch["labels"]),
             "live-loss metadata differs")

    _keys(payload["state_hashes"], STATE_HASH_KEYS, "state hashes")
    rng_hash = codec.tree_digest(anchor["payload"]["rng"])
    expected_hashes = {
        "parameters_before": codec.tree_digest(before_parameters),
        "parameters_after": codec.tree_digest(payload["parameters_after"]),
        "raw_gradient": codec.tensor_digest(payload["raw_gradient"]["value"]),
        "delivered_current_gradient": codec.tensor_digest(payload["delivered_current_gradient"]["value"]),
        "optimizer_after": codec.tree_digest(payload["optimizer_after"]),
        "observer_after": codec.tree_digest(payload["observer_after"]),
        "rng_before": rng_hash, "rng_after": rng_hash,
    }
    _require(envelopes.same_exact(payload["state_hashes"], expected_hashes), "state hashes differ")
    codec.tree_digest({"anchor": anchor, "witness": value})  # Cross-envelope ownership gate.
    return value


def validate_source_witness(value, *, anchor, anchor_receipt, identity, profile, plan,
                            plan_artifact, images_bytes, labels_bytes, expected_files,
                            sources, environment):
    """Strict value validator; flags and hashes do not prove live event history."""
    try:
        return _validate_witness_impl(value, anchor=anchor, anchor_receipt=anchor_receipt,
            identity=identity, profile=profile, plan=plan, plan_artifact=plan_artifact,
            images_bytes=images_bytes, labels_bytes=labels_bytes, expected_files=expected_files,
            sources=sources, environment=environment)
    except SourceCaptureError:
        raise
    except Exception as exc:
        raise SourceCaptureError("source witness validation failed: " + type(exc).__name__) from None


def _make_witness(*, anchor, anchor_ref, anchor_receipt, before_core, after_core,
                  raw_gradient, delivered_gradient, endpoint_core, loss_value,
                  sample_identity, identity, profile, created_utc, context):
    update = identity["anchor_update"]
    before_parameters = anchor["payload"]["model"]["parameters"]
    parameters_after = state.clone_tree(endpoint_core["model"]["parameters"])
    optimizer_after = state.clone_tree(endpoint_core["optimizer"])
    observer_after = state.clone_tree(endpoint_core["observer"])
    core_hash_before, core_hash_after = codec.tree_digest(before_core), codec.tree_digest(after_core)
    rng_hash = codec.tree_digest(before_core["rng"])
    payload = {
        "anchor_ref": anchor_ref,
        "source_step": {
            "anchor_update": update, "completed_updates_before": update - 1,
            "completed_observations_before": update - 1, "completed_updates_after": update,
            "completed_observations_after": update,
            "source_state_origin": "uninterrupted_live_objects_v1",
            "anchor_deserializations_before_live_step": 0,
            "observation_ingests_in_live_step": 1, "optimizer_steps_in_live_step": 1,
        },
        "capture_proof": {
            "proof_id": "capture_neutrality_v1", "ordered_events": list(EVENTS),
            "anchor_deserializations_before_live_step": 0,
            "core_sha256_before": core_hash_before, "core_sha256_after": core_hash_after,
            "core_direct_typed_equal": True, "same_live_object_bindings": True,
            "anchor_sealed_before_live_step": True,
            "live_step_complete_at_payload_assembly": True,
        },
        "before_parameters_binding": {
            "anchor_artifact_id": anchor["artifact_id"],
            "anchor_model_parameters_tree_sha256": codec.tree_digest(before_parameters),
            "parameter_order_tree_sha256": codec.tree_digest(anchor["payload"]["model"]["parameter_order"]),
        },
        "raw_gradient": raw_gradient, "delivered_current_gradient": delivered_gradient,
        "parameters_after": parameters_after, "optimizer_after": optimizer_after,
        "observer_after": observer_after,
        "live_loss": {"probe_key": "batch_noisy", "count": int(context["inputs"].shape[0]),
                      "value_native_float32": loss_value,
                      "sample_identity_sha256": sample_identity},
        "state_hashes": {
            "parameters_before": codec.tree_digest(before_parameters),
            "parameters_after": codec.tree_digest(parameters_after),
            "raw_gradient": codec.tensor_digest(raw_gradient["value"]),
            "delivered_current_gradient": codec.tensor_digest(delivered_gradient["value"]),
            "optimizer_after": codec.tree_digest(optimizer_after),
            "observer_after": codec.tree_digest(observer_after),
            "rng_before": rng_hash, "rng_after": rng_hash,
        },
    }
    witness = envelopes.wrap(payload, schema_name="i7_source_step_witness", kind="source-witness",
                             identity=identity, profile=profile, created_utc=created_utc)
    validate_source_witness(witness, anchor=anchor, anchor_receipt=anchor_receipt,
                            identity=identity, profile=profile, **context["validation"])
    return witness


def capture_anchor_then_live_witness(model, optimizer, observer, *, store, identity,
                                     profile, created_utc, plan, plan_artifact,
                                     images_bytes, labels_bytes, expected_files,
                                     sources, environment, guard=None):
    """Seal anchor, prove neutrality, and seal the live step with optional guard checks."""
    valid_store = type(store) is storage.ArtifactStore
    try:
        _require(guard is None or callable(guard), "guard must be callable or None")
        _guard(guard, "source_capture.entry")
        _require(valid_store, "exact ArtifactStore required")
        store._ensure_writable()
        _require(type(profile) is str and profile in PROFILES and store.profile == profile,
                 "store/profile mismatch")
        codec.validate_identity(identity, profile=profile)
        codec.validate_created_utc(created_utc)
        _guard(guard, "source_capture.anchor.before_initial_context")
        _verify_live_context(sources, environment, profile, include_sources=True)
        _guard(guard, "source_capture.anchor.after_initial_context")
        parameter_refs = tuple(model.parameters())
        _require(_same_live_bindings(model, optimizer, observer, parameter_refs),
                 "initial live object bindings differ")

        _guard(guard, "source_capture.anchor.before_core_capture")
        before_core = state.capture_core(model, optimizer, observer, profile=profile,
                                         completed_updates=identity["anchor_update"] - 1)
        _guard(guard, "source_capture.anchor.after_core_capture")
        _guard(guard, "source_capture.anchor.before_assembly")
        anchor = envelopes.make_anchor(before_core, identity=identity, profile=profile,
            created_utc=created_utc, plan=plan, plan_artifact=plan_artifact,
            images_bytes=images_bytes, labels_bytes=labels_bytes, expected_files=expected_files,
            sources=sources, environment=environment)
        _guard(guard, "source_capture.anchor.after_assembly")
        _guard(guard, "source_capture.anchor.before_validation")
        envelopes.validate_anchor(anchor, identity=identity, profile=profile, plan=plan,
            plan_artifact=plan_artifact, images_bytes=images_bytes, labels_bytes=labels_bytes,
            expected_files=expected_files, sources=sources, environment=environment)
        _guard(guard, "source_capture.anchor.after_validation")
        anchor_name = anchor["artifact_id"] + ".pt"
        _guard(guard, "source_capture.anchor.artifact.prewrite")
        anchor_receipt = store.write_tensor_tree(anchor_name, anchor)
        _guard(guard, "source_capture.anchor.artifact.postwrite")
        _guard(guard, "source_capture.anchor.before_receipt_read")
        anchor_ref = _sealed_anchor_reference(store, anchor, anchor_receipt)
        _guard(guard, "source_capture.anchor.after_receipt_read")
        _guard(guard, "source_capture.anchor.before_postseal_context")
        _verify_live_context(sources, environment, profile, include_sources=True)
        _guard(guard, "source_capture.anchor.after_postseal_context")

        _guard(guard, "source_capture.anchor.before_neutrality_capture")
        after_core = state.capture_core(model, optimizer, observer, profile=profile,
                                        completed_updates=identity["anchor_update"] - 1)
        _guard(guard, "source_capture.anchor.after_neutrality_capture")
        _require(_same_live_bindings(model, optimizer, observer, parameter_refs),
                 "same live object bindings changed across anchor seal")
        _require(envelopes.same_exact(before_core, after_core),
                 "complete live core changed across anchor seal")

        materialized = data.materialize(images_bytes, labels_bytes, plan=plan, identity=identity,
                                        profile=profile, expected_files=expected_files)
        batch = materialized["probes"]["batch_noisy"]
        native_device = parameter_refs[0].device
        inputs = batch["inputs"].detach().clone().to(device=native_device, dtype=torch.float32)
        labels = batch["labels"].detach().clone().to(device=native_device, dtype=torch.int64)
        _guard(guard, "source_capture.source.before_forward")
        logits = model(inputs)
        _guard(guard, "source_capture.source.after_forward")
        loss = F.cross_entropy(logits, labels, reduction="mean")
        _require(type(loss) is torch.Tensor and loss.shape == torch.Size([])
                 and loss.dtype == torch.float32 and loss.device == native_device
                 and bool(torch.isfinite(loss)), "actual source loss is not finite native float32")
        loss_value = float(loss.detach().cpu().item())
        _guard(guard, "source_capture.source.before_backward")
        loss.backward()
        _guard(guard, "source_capture.source.after_backward")
        raw_gradient = _gradient_record(model, profile=profile, where="actual raw gradient")
        _guard(guard, "source_capture.source.before_observer")
        observer.filter_grad()
        _guard(guard, "source_capture.source.after_observer")
        delivered_gradient = _gradient_record(model, profile=profile, where="actual delivered gradient")
        _guard(guard, "source_capture.source.before_optimizer_step")
        optimizer.step()
        _guard(guard, "source_capture.source.after_optimizer_step")
        optimizer.zero_grad(set_to_none=True)
        endpoint_core = state.capture_core(model, optimizer, observer, profile=profile,
                                           completed_updates=identity["anchor_update"])
        _require(envelopes.same_exact(endpoint_core["rng"], before_core["rng"]),
                 "deterministic source step changed RNG state")
        _require(_same_live_bindings(model, optimizer, observer, parameter_refs),
                 "live object bindings changed during actual source step")
        _verify_live_context(sources, environment, profile, include_sources=False)

        context = {"inputs": batch["inputs"],
                   "validation": dict(plan=plan, plan_artifact=plan_artifact,
                                      images_bytes=images_bytes, labels_bytes=labels_bytes,
                                      expected_files=expected_files, sources=sources,
                                      environment=environment)}
        witness = _make_witness(anchor=anchor, anchor_ref=anchor_ref,
            anchor_receipt=anchor_receipt, before_core=before_core, after_core=after_core,
            raw_gradient=raw_gradient, delivered_gradient=delivered_gradient,
            endpoint_core=endpoint_core, loss_value=loss_value,
            sample_identity=_sample_hash(batch["inputs"], batch["labels"]), identity=identity,
            profile=profile, created_utc=created_utc, context=context)
        codec.tree_digest({"anchor": anchor, "witness": witness})
        witness_name = witness["artifact_id"] + ".pt"
        _guard(guard, "source_capture.witness.artifact.prewrite")
        witness_receipt = store.write_tensor_tree(witness_name, witness)
        _guard(guard, "source_capture.witness.artifact.postwrite")
        _require(not store._terminal, "store became terminal after witness seal")
        return {"anchor": anchor, "anchor_receipt": anchor_receipt,
                "witness": witness, "witness_receipt": witness_receipt}
    except BaseException as exc:
        if valid_store and not store._closed and not store._terminal:
            store._fail("source_capture_failed", type(exc).__name__)
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        if isinstance(exc, SourceCaptureError):
            raise
        raise SourceCaptureError("source capture failed: " + type(exc).__name__) from None
