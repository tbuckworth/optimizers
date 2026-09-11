"""Strict anchor envelopes bound to supplied plan/data/source/runtime evidence.

Pure consistency validation, not verified file loading or scientific launch GO.
The caller obtains the expected inputs through their separate verification APIs.
"""
from __future__ import annotations

import copy

import torch

import artifact_store as storage
import data_probe_bindings as data
import identity_codec as codec
import plan_bindings as plans
import source_environment as provenance
import state_core as state


class EnvelopeError(ValueError):
    pass


ROOT_KEYS = ("schema_name", "schema_version", "profile", "artifact_id", "created_utc",
             "identity", "payload", "payload_tensor_bytes")
ANCHOR_KEYS = ("model", "optimizer", "observer", "rng", "bindings")
BINDING_KEYS = ("plan", "probes", "data", "sources", "environment")
PROFILES = (storage.SCIENTIFIC, storage.MLP_FIXTURE)


def _require(condition, message):
    if not condition:
        raise EnvelopeError(message)


def _keys(value, expected, where):
    _require(type(value) is dict and tuple(value) == tuple(expected)
             and all(type(key) is str for key in value), where + ": exact ordered keys required")


def same_exact(left, right):
    """Direct typed/ordered comparison with byte-exact tensors and signed zeros."""
    if type(left) is not type(right):
        return False
    if type(left) is dict:
        return (len(left) == len(right) and all(same_exact(lk, rk) and same_exact(lv, rv)
                for (lk, lv), (rk, rv) in zip(left.items(), right.items())))
    if type(left) in (list, tuple):
        return len(left) == len(right) and all(same_exact(a, b) for a, b in zip(left, right))
    if type(left) is torch.Tensor:
        return (left.dtype == right.dtype and left.shape == right.shape
                and left.device == right.device
                and left.numpy().tobytes(order="C") == right.numpy().tobytes(order="C"))
    if type(left) is float:
        return left.hex() == right.hex()
    return left == right


def tensor_bytes(value):
    """Count every tensor leaf; callers first validate a finite non-aliased tree."""
    if type(value) is torch.Tensor:
        return value.numel() * value.element_size()
    if type(value) is dict:
        return sum(tensor_bytes(item) for item in value.values())
    if type(value) in (list, tuple):
        return sum(tensor_bytes(item) for item in value)
    return 0


def wrap(payload, *, schema_name, kind, identity, profile, created_utc):
    """Owned common envelope; artifact-specific validation remains mandatory."""
    _require(type(profile) is str and profile in PROFILES, "unsupported envelope profile")
    codec.validate_identity(identity, profile=profile)
    codec.validate_created_utc(created_utc)
    codec.tree_digest(payload)  # Whole-tree type/ownership/finite/Unicode gate.
    result = dict(schema_name=schema_name, schema_version=1, profile=profile,
                  artifact_id=codec.artifact_id(identity, profile=profile, kind=kind),
                  created_utc=created_utc, identity=copy.deepcopy(identity),
                  payload=state.clone_tree(payload), payload_tensor_bytes=tensor_bytes(payload))
    validate_common(result, schema_name=schema_name, kind=kind, identity=identity, profile=profile)
    return result


def validate_common(value, *, schema_name, kind, identity, profile):
    _require(type(profile) is str and profile in PROFILES, "unsupported envelope profile")
    _keys(value, ROOT_KEYS, "envelope")
    codec.validate_identity(identity, profile=profile)
    codec.validate_identity(value["identity"], profile=profile)
    codec.validate_created_utc(value["created_utc"])
    _require(same_exact(value["identity"], identity), "unexpected envelope identity")
    for key, expected in (("schema_name", schema_name), ("schema_version", 1),
                          ("profile", profile),
                          ("artifact_id", codec.artifact_id(identity, profile=profile, kind=kind))):
        _require(same_exact(value[key], expected), "invalid envelope " + key)
    codec.tree_digest(value)
    count = value["payload_tensor_bytes"]
    _require(type(count) is int and count == tensor_bytes(value["payload"]),
             "payload tensor byte count differs")
    return value


def core_from_anchor(value):
    """Projection only: call validate_anchor before using a consumer's envelope."""
    identity, payload = value["identity"], value["payload"]
    return dict(schema_name="i7_state_core", schema_version=1, profile=value["profile"],
                phase=identity["anchor_phase"], anchor_update=identity["anchor_update"],
                state_completed_updates=identity["anchor_completed_updates"],
                **{key: payload[key] for key in ANCHOR_KEYS[:-1]})


def _expected_bindings(*, identity, profile, plan, plan_artifact, images_bytes,
                       labels_bytes, expected_files, sources, environment):
    provenance.validate_sources(sources, profile=profile)
    provenance.validate_environment(environment, profile=profile)
    _require(sources["repository_root_realpath"] == environment["repository_root_realpath"],
             "source and environment repository roots differ")
    materialized = data.materialize(images_bytes, labels_bytes, plan=plan, identity=identity,
                                   profile=profile, expected_files=expected_files)
    return dict(plan=plans.make_plan_binding(plan, identity=identity, profile=profile,
                                            artifact=plan_artifact),
                probes=materialized["bindings"]["probes"], data=materialized["bindings"]["data"],
                sources=copy.deepcopy(sources), environment=copy.deepcopy(environment))


def make_anchor(core, *, identity, profile, created_utc, plan, plan_artifact,
                images_bytes, labels_bytes, expected_files, sources, environment):
    try:
        _require(type(profile) is str and profile in PROFILES, "unsupported anchor profile")
        codec.validate_identity(identity, profile=profile)
        state.validate_core(core)
        codec.tree_digest(core)  # Reject non-owned/lazy/grad-bearing inputs before any clone.
        _require(core["profile"] == profile and core["anchor_update"] == identity["anchor_update"]
                 and core["state_completed_updates"] == identity["anchor_completed_updates"],
                 "core profile/counters differ from anchor identity")
        provenance.validate_state_environment(core, environment, profile=profile)
        bindings = _expected_bindings(identity=identity, profile=profile, plan=plan,
            plan_artifact=plan_artifact, images_bytes=images_bytes, labels_bytes=labels_bytes,
            expected_files=expected_files, sources=sources, environment=environment)
        payload = {key: state.clone_tree(core[key]) for key in ANCHOR_KEYS[:-1]}
        payload["bindings"] = bindings
        return wrap(payload, schema_name="i7_anchor", kind="anchor", identity=identity,
                    profile=profile, created_utc=created_utc)
    except EnvelopeError:
        raise
    except Exception as exc:
        raise EnvelopeError("anchor construction failed: " + type(exc).__name__) from None


def validate_anchor(value, *, identity, profile, plan, plan_artifact, images_bytes,
                    labels_bytes, expected_files, sources, environment):
    try:
        validate_common(value, schema_name="i7_anchor", kind="anchor", identity=identity, profile=profile)
        payload = value["payload"]
        _keys(payload, ANCHOR_KEYS, "anchor payload")
        _keys(payload["bindings"], BINDING_KEYS, "anchor bindings")
        core = core_from_anchor(value)
        state.validate_core(core)
        expected = _expected_bindings(identity=identity, profile=profile, plan=plan,
            plan_artifact=plan_artifact, images_bytes=images_bytes, labels_bytes=labels_bytes,
            expected_files=expected_files, sources=sources, environment=environment)
        _require(same_exact(payload["bindings"], expected), "anchor binding values differ")
        provenance.validate_state_environment(core, environment, profile=profile)
        return value
    except EnvelopeError:
        raise
    except Exception as exc:
        raise EnvelopeError("anchor validation failed: " + type(exc).__name__) from None
