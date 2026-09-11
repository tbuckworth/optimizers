#!/usr/bin/env python3
"""Source-history records with bounded captured-state data for iteration 007.

The constructors consume supplied state streams. They own no training, data,
file or plan workflow and grant no scientific execution authority. Iterating
caller-provided streams can execute arbitrary caller code: this is not a
side-effect sandbox. Captured tensor/primitive state omits live hooks/callbacks.
Both
pilot arms retain a state hash after every update; ``capture_on`` additionally
seals the declared anchors and source witnesses.  Thus ``capture_off`` is an
instrumented baseline, not a completely uninstrumented execution.

Public API: ``trajectory``, ``initial_fingerprint``,
``validate_initial_fingerprint``, ``compare_initial``, ``step_fingerprint``,
``compare_step``, ``make_source_completion``, ``validate_source_completion``,
``make_capture_comparison``, ``validate_capture_comparison`` and
``validate_capture_bundle``.
"""
from __future__ import annotations

import copy
import hashlib
import re
import struct
from typing import Any

import torch

import artifact_store as storage
import identity_codec as codec
import state_core as state


class SourceHistoryError(ValueError):
    """A source-history value violates the frozen bounded contract."""


MLP_FIXTURE = state.MLP_FIXTURE_PROFILE
SCIENTIFIC = state.SCIENTIFIC_PROFILE
PROFILES = (SCIENTIFIC, MLP_FIXTURE)
CAPTURE_MODES = ("capture_on", "capture_off")
TRAJECTORY_KEYS = (
    "run_id", "iteration", "execution_role", "evidence_role", "bundle",
    "source_policy", "condition", "steps_total", "anchor_updates",
)
FINGERPRINT_KEYS = (
    "completed_updates", "completed_observations", "next_anchor_update",
    "core_sha256", "model_sha256", "optimizer_sha256", "moments_sha256",
    "observer_sha256", "rng_sha256",
)
INITIAL_ROOT_KEYS = (
    "schema_name", "schema_version", "profile", "phase",
    "state_completed_updates", "model", "optimizer", "observer", "rng",
)
INITIAL_OPTIMIZER_KEYS = (
    "class_name", "state_completed_updates", "parameter_order", "param_groups",
    "state_kind", "state",
)
INITIAL_FINGERPRINT_KEYS = (
    "state_sha256", "model_sha256", "optimizer_sha256", "observer_sha256",
    "rng_sha256",
)
COMPARISON_KEYS = (
    "completed_updates", "on", "off", "model_direct_typed_equal",
    "optimizer_direct_typed_equal", "moments_direct_typed_equal",
    "observer_direct_typed_equal", "rng_direct_typed_equal",
    "core_direct_typed_equal",
)
INITIAL_COMPARISON_KEYS = (
    "on", "off", "model_direct_typed_equal", "optimizer_direct_typed_equal",
    "observer_direct_typed_equal", "rng_direct_typed_equal",
    "state_direct_typed_equal",
)
PLAN_REF_KEYS = (
    "name", "status", "encoding", "size_bytes", "sha256", "receipt_name",
    "receipt_size_bytes", "receipt_sha256",
)
ARTIFACT_REF_KEYS = (
    "artifact_id", "schema_name", "name", "size_bytes", "sha256", "status",
    "encoding", "receipt_name", "receipt_size_bytes", "receipt_sha256",
)
ANCHOR_WITNESS_KEYS = ("anchor_update", "anchor_ref", "witness_ref")
PROVENANCE_KEYS = ("sources_sha256", "environment_sha256")
INSTRUMENTATION_KEYS = (
    "both_modes_common", "capture_on_additional", "capture_off_additional",
    "baseline_kind",
)
INSTRUMENTATION = {
    "both_modes_common": "state_core_hash_after_every_completed_update",
    "capture_on_additional": "seal_declared_anchor_and_source_witness_artifacts",
    "capture_off_additional": "none",
    "baseline_kind": "instrumented_state_hash_baseline_not_uninstrumented",
}
EVIDENCE_SCOPE_KEYS = (
    "trace_origin", "retained_full_state_scope", "validator_recomputed_byte_evidence",
    "discarded_historical_state_bytes_recomputed", "external_receipt_bytes_verified",
    "live_history_reexecuted",
)
SOURCE_EVIDENCE_SCOPE = {
    "trace_origin": "derived_at_construction_from_supplied_validated_full_state_cores",
    "retained_full_state_scope": "final_update_only",
    "validator_recomputed_byte_evidence": "retained_final_state_core_only",
    "discarded_historical_state_bytes_recomputed": False,
    "external_receipt_bytes_verified": False,
    "live_history_reexecuted": False,
}
COMPARISON_EVIDENCE_SCOPE = {
    "trace_origin": "derived_at_construction_from_supplied_validated_full_state_cores",
    "retained_full_state_scope": "none",
    "validator_recomputed_byte_evidence": "none_digest_row_self_consistency_only",
    "discarded_historical_state_bytes_recomputed": False,
    "external_receipt_bytes_verified": False,
    "live_history_reexecuted": False,
}
SOURCE_ROOT_KEYS = (
    "schema_name", "schema_version", "profile", "artifact_id", "artifact_name",
    "trajectory", "capture_mode", "instrumentation", "plan_ref", "provenance",
    "anchor_witness_refs", "trace", "final_state_core", "evidence_scope",
    "scientific_execution_certified",
)
SUMMARY_KEYS = (
    "steps_compared", "model_equal_steps", "optimizer_equal_steps",
    "moments_equal_steps", "observer_equal_steps", "rng_equal_steps",
    "core_equal_steps",
)
CAPTURE_ROOT_KEYS = (
    "schema_name", "schema_version", "profile", "artifact_id", "artifact_name",
    "trajectory", "instrumentation", "initial_comparison", "step_trace",
    "summary", "evidence_scope", "scientific_execution_certified",
)
_SHA = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)
CORE_TENSOR_BYTE_CAP = 32 << 20
CORE_METADATA_BYTE_CAP = 1 << 20
CORE_NODE_CAP = 10000
CORE_DEPTH_CAP = 16
CORE_STRING_CAP = 4096


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SourceHistoryError(message)


def _keys(value: Any, expected: tuple[str, ...], where: str) -> None:
    _require(type(value) is dict and tuple(value.keys()) == expected and
             all(type(key) is str for key in value),
             where + ": exact ordered keys required")


def _sha(value: Any, where: str) -> str:
    _require(type(value) is str and _SHA.fullmatch(value) is not None,
             where + ": lowercase SHA-256 required")
    return value


def _clone(value: Any) -> Any:
    try:
        return state.clone_tree(value)
    except (TypeError, ValueError) as exc:
        raise SourceHistoryError("unsupported tensor/primitive tree") from exc


def _same_exact(left: Any, right: Any) -> bool:
    """Ordered typed equality, including floating-point sign-zero bytes."""
    if type(left) is not type(right):
        return False
    if type(left) is torch.Tensor:
        if left.dtype != right.dtype or left.shape != right.shape:
            return False
        return (left.detach().cpu().contiguous().numpy().tobytes(order="C") ==
                right.detach().cpu().contiguous().numpy().tobytes(order="C"))
    if type(left) is dict:
        return (tuple(left.keys()) == tuple(right.keys()) and
                all(_same_exact(left[key], right[key]) for key in left))
    if type(left) in (list, tuple):
        return (len(left) == len(right) and
                all(_same_exact(a, b) for a, b in zip(left, right)))
    if type(left) is float:
        return struct.pack(">d", left) == struct.pack(">d", right)
    return left == right


def _bound_full_state(value: Any) -> None:
    """Reject oversized in-memory payloads before validation/hash byte copies.

    These are representation ceilings, not native RNG-layout certification or
    live-capture/process memory guards. File decoding needs its own admission.
    """
    stack = [(value, 0)]
    nodes = tensor_bytes = metadata_bytes = 0
    while stack:
        item, depth = stack.pop()
        nodes += 1
        _require(nodes <= CORE_NODE_CAP and depth <= CORE_DEPTH_CAP,
                 "full state node/depth bound exceeded")
        kind = type(item)
        if kind is torch.Tensor:
            _require(item.device.type == "cpu" and item.layout == torch.strided
                     and item.is_contiguous() and item.storage_offset() == 0
                     and item.untyped_storage().nbytes() == item.numel() * item.element_size(),
                     "full state tensor must have exact contiguous CPU storage")
            tensor_bytes += item.numel() * item.element_size()
            _require(tensor_bytes <= CORE_TENSOR_BYTE_CAP, "full state tensor byte bound exceeded")
        elif kind is str:
            _require(len(item) <= CORE_STRING_CAP, "full state string bound exceeded")
            metadata_bytes += len(item.encode("utf-8"))
        elif kind is dict:
            _require(len(item) <= CORE_NODE_CAP, "full state dictionary bound exceeded")
            stack.extend((part, depth+1) for pair in item.items() for part in pair)
        elif kind in (list, tuple):
            _require(len(item) <= CORE_NODE_CAP, "full state sequence bound exceeded")
            stack.extend((part, depth+1) for part in item)
        elif kind is int:
            _require(item.bit_length() <= 64, "full state integer bound exceeded")
            metadata_bytes += 16
        elif kind in (float, bool) or item is None:
            metadata_bytes += 16
        else:
            raise SourceHistoryError("full state unsupported built-in type")
        _require(nodes + len(stack) <= CORE_NODE_CAP, "full state pending node bound exceeded")
        _require(metadata_bytes <= CORE_METADATA_BYTE_CAP,
                 "full state metadata byte bound exceeded")


def trajectory(identity: Any, profile: str) -> dict[str, Any]:
    """Derive one exact trajectory from a registered anchor identity."""
    try:
        codec.validate_identity(identity, profile=profile)
    except (codec.CodecError, KeyError, TypeError) as exc:
        raise SourceHistoryError("invalid trajectory identity") from exc
    _require(profile in PROFILES, "source history supports scientific and MLP fixture only")
    if profile == MLP_FIXTURE:
        anchors = [5]
    elif identity["execution_role"] == "pilot":
        anchors = [101, 200]
    else:
        anchors = [101, 500, 1000, 2000]
    return {
        "run_id": identity["run_id"], "iteration": identity["iteration"],
        "execution_role": identity["execution_role"],
        "evidence_role": identity["evidence_role"], "bundle": identity["bundle"],
        "source_policy": identity["source_policy"], "condition": identity["condition"],
        "steps_total": identity["steps_total"], "anchor_updates": anchors,
    }


def _validate_trajectory(value: Any, profile: str) -> dict[str, Any]:
    _keys(value, TRAJECTORY_KEYS, "trajectory")
    anchor = value["anchor_updates"][0] if type(value["anchor_updates"]) is list and value["anchor_updates"] else -1
    identity = {
        "run_id": value["run_id"], "iteration": value["iteration"],
        "execution_role": value["execution_role"], "evidence_role": value["evidence_role"],
        "bundle": value["bundle"], "anchor_update": anchor,
        "source_policy": value["source_policy"], "condition": value["condition"],
        "steps_total": value["steps_total"], "anchor_phase": "pre_forward_pre_observe",
        "anchor_completed_updates": anchor - 1,
        "anchor_completed_observations": anchor - 1,
        "rng_namespace_prefix": [20260906, value["bundle"]],
        "stream_roles": dict(codec.STREAM_ROLES),
    }
    expected = trajectory(identity, profile)
    _require(_same_exact(value, expected), "trajectory membership differs")
    return value


def step_fingerprint(core: Any) -> dict[str, Any]:
    """Derive a compact row directly from one validated complete state core."""
    try:
        _bound_full_state(core)
        state.validate_core(core)
        completed = core["state_completed_updates"]
        return {
            "completed_updates": completed,
            "completed_observations": core["observer"]["state_completed_observations"],
            "next_anchor_update": core["anchor_update"],
            "core_sha256": codec.tree_digest(core),
            "model_sha256": codec.tree_digest(core["model"]),
            "optimizer_sha256": codec.tree_digest(core["optimizer"]),
            "moments_sha256": codec.tree_digest(core["optimizer"]["state"]),
            "observer_sha256": codec.tree_digest(core["observer"]),
            "rng_sha256": codec.tree_digest(core["rng"]),
        }
    except (ValueError, codec.CodecError, KeyError, TypeError) as exc:
        raise SourceHistoryError("invalid full state core") from exc


def _validate_fingerprint(value: Any, *, expected_update: int | None = None) -> None:
    _keys(value, FINGERPRINT_KEYS, "step fingerprint")
    update = value["completed_updates"]
    _require(type(update) is int and update >= 1 and
             type(value["completed_observations"]) is int and
             value["completed_observations"] == update and
             type(value["next_anchor_update"]) is int and
             value["next_anchor_update"] == update + 1,
             "step fingerprint counters differ")
    if expected_update is not None:
        _require(update == expected_update, "missing, duplicate, or reordered source update")
    for key in FINGERPRINT_KEYS[3:]:
        _sha(value[key], "step fingerprint." + key)


def compare_step(on_core: Any, off_core: Any) -> dict[str, Any]:
    """Compare same-step full cores directly; callers cannot supply pass flags."""
    try:
        _bound_full_state(on_core)
        _bound_full_state(off_core)
        state.validate_core(on_core)
        state.validate_core(off_core)
    except (ValueError, KeyError, TypeError) as exc:
        raise SourceHistoryError("invalid full state core in comparison") from exc
    _require(on_core["profile"] == off_core["profile"], "step profiles differ")
    _require(on_core["state_completed_updates"] == off_core["state_completed_updates"],
             "step counters differ")
    # Byte comparison deliberately precedes hashing.  Hashes are compact
    # retained identities; they are not a substitute for construction-time
    # typed equality (notably for signed zero).
    equal = {
        "model": _same_exact(on_core["model"], off_core["model"]),
        "optimizer": _same_exact(on_core["optimizer"], off_core["optimizer"]),
        "moments": _same_exact(on_core["optimizer"]["state"],
                               off_core["optimizer"]["state"]),
        "observer": _same_exact(on_core["observer"], off_core["observer"]),
        "rng": _same_exact(on_core["rng"], off_core["rng"]),
        "core": _same_exact(on_core, off_core),
    }
    on = step_fingerprint(on_core)
    off = step_fingerprint(off_core)
    return {
        "completed_updates": on["completed_updates"], "on": on, "off": off,
        "model_direct_typed_equal": equal["model"],
        "optimizer_direct_typed_equal": equal["optimizer"],
        "moments_direct_typed_equal": equal["moments"],
        "observer_direct_typed_equal": equal["observer"],
        "rng_direct_typed_equal": equal["rng"],
        "core_direct_typed_equal": equal["core"],
    }


def _validate_comparison(value: Any, *, expected_update: int) -> None:
    _keys(value, COMPARISON_KEYS, "step comparison")
    _require(type(value["completed_updates"]) is int and
             value["completed_updates"] == expected_update,
             "comparison updates missing, duplicated, or reordered")
    _validate_fingerprint(value["on"], expected_update=expected_update)
    _validate_fingerprint(value["off"], expected_update=expected_update)
    pairs = (
        ("model_direct_typed_equal", "model_sha256"),
        ("optimizer_direct_typed_equal", "optimizer_sha256"),
        ("moments_direct_typed_equal", "moments_sha256"),
        ("observer_direct_typed_equal", "observer_sha256"),
        ("rng_direct_typed_equal", "rng_sha256"),
        ("core_direct_typed_equal", "core_sha256"),
    )
    for flag, digest in pairs:
        _require(type(value[flag]) is bool and
                 value[flag] == (value["on"][digest] == value["off"][digest]),
                 flag + " inconsistent with retained byte digest evidence")


def _capture_initial_optimizer(optimizer: Any, model: torch.nn.Module,
                               profile: str) -> dict[str, Any]:
    parameters = list(model.parameters())
    _require(type(optimizer).__name__ == "AdamW", "initial optimizer must be AdamW")
    _require(len(optimizer.param_groups) == 1 and
             len(optimizer.param_groups[0]["params"]) == len(parameters) and
             all(a is b for a, b in zip(optimizer.param_groups[0]["params"], parameters)),
             "initial optimizer parameter binding/order differs")
    raw = optimizer.state_dict()
    _require(type(raw) is dict and tuple(raw.keys()) == ("state", "param_groups") and
             type(raw["state"]) is dict and not raw["state"] and
             type(raw["param_groups"]) is list and len(raw["param_groups"]) == 1,
             "initial optimizer state must be exactly empty")
    group = raw["param_groups"][0]
    _require(set(group) == set(state.GROUP_KEYS[:-1] + ("params",)),
             "initial AdamW group keys differ")
    encoded_group = {key: copy.deepcopy(group[key]) for key in state.GROUP_KEYS
                     if key != "param_indices"}
    encoded_group["param_indices"] = list(group["params"])
    expected_group = {**state.EXPECTED_GROUP,
                      "param_indices": list(range(len(parameters)))}
    _require(_same_exact(encoded_group, expected_group), "initial AdamW options differ")
    return {
        "class_name": "torch.optim.AdamW", "state_completed_updates": 0,
        "parameter_order": [name for name, _ in state.PROFILE[profile]["parameters"]],
        "param_groups": [encoded_group], "state_kind": "empty_before_first_step",
        "state": [],
    }


def initial_fingerprint(model: torch.nn.Module, optimizer: Any, observer: Any,
                        *, profile: str) -> dict[str, Any]:
    """Capture registered constructor-state fields at update counter zero.

    Despite the compact public name, this is a transient full snapshot so that
    ``compare_initial`` can perform direct typed/byte equality.  The comparison
    artifact retains only its hashes and equality booleans. Live hooks and
    callbacks are not captured or excluded; native entry must establish their
    absence and validate the live optimizer fields before invoking this API.
    """
    _require(type(profile) is str and profile in PROFILES,
             "unsupported initial-state profile")
    before_rng = state._raw_rng_state()
    try:
        model_value = state._capture_model(model, profile)
        optimizer_value = _capture_initial_optimizer(optimizer, model, profile)
        observer_value = state._capture_observer(observer, model, optimizer, profile, 0)
        rng_value = state._capture_rng()
        value = {
            "schema_name": "i7_initial_state", "schema_version": 1,
            "profile": profile, "phase": "before_first_source_update",
            "state_completed_updates": 0, "model": model_value,
            "optimizer": optimizer_value, "observer": observer_value, "rng": rng_value,
        }
        validate_initial_fingerprint(value)
        state._assert_no_tensor_aliases(value)
        _require(_same_exact(before_rng, state._raw_rng_state()),
                 "initial-state capture changed caller RNG")
        return value
    except SourceHistoryError:
        raise
    except (ValueError, KeyError, TypeError, codec.CodecError) as exc:
        raise SourceHistoryError("invalid live initial state") from exc


def validate_initial_fingerprint(value: Any) -> dict[str, Any]:
    """Validate the dedicated counter-zero snapshot (not an i7_state_core)."""
    try:
        _bound_full_state(value)
        _keys(value, INITIAL_ROOT_KEYS, "initial state")
        _require(type(value["schema_name"]) is str and
                 value["schema_name"] == "i7_initial_state" and
                 type(value["schema_version"]) is int and value["schema_version"] == 1 and
                 type(value["profile"]) is str and value["profile"] in PROFILES and
                 type(value["phase"]) is str and
                 value["phase"] == "before_first_source_update" and
                 type(value["state_completed_updates"]) is int and
                 value["state_completed_updates"] == 0,
                 "initial-state root differs")
        profile = value["profile"]
        state._validate_model(value["model"], profile)
        native_devices = [row["native_device"] for row in value["model"]["parameters"]]
        _require(len(set(native_devices)) == 1, "initial model spans devices")
        optimizer = value["optimizer"]
        _keys(optimizer, INITIAL_OPTIMIZER_KEYS, "initial optimizer")
        names = [name for name, _ in state.PROFILE[profile]["parameters"]]
        _require(type(optimizer["class_name"]) is str and
                 optimizer["class_name"] == "torch.optim.AdamW" and
                 type(optimizer["state_completed_updates"]) is int and
                 optimizer["state_completed_updates"] == 0 and
                 type(optimizer["parameter_order"]) is list and
                 all(type(name) is str for name in optimizer["parameter_order"]) and
                 optimizer["parameter_order"] == names and
                 type(optimizer["param_groups"]) is list and
                 len(optimizer["param_groups"]) == 1 and
                 type(optimizer["state_kind"]) is str and
                 optimizer["state_kind"] == "empty_before_first_step" and
                 type(optimizer["state"]) is list and optimizer["state"] == [],
                 "initial optimizer root differs")
        state._exact_keys(optimizer["param_groups"][0], state.GROUP_KEYS,
                          "initial optimizer group")
        _require(_same_exact(optimizer["param_groups"][0],
                             {**state.EXPECTED_GROUP,
                              "param_indices": list(range(len(names)))}),
                 "initial optimizer group differs")
        state._validate_observer(value["observer"], profile, 0, native_devices[0])
        _require(_same_exact(value["observer"]["state"], {
            "V": None, "S": None, "proj_k": None, "step_count": 0,
            "grad_mean": None, "stabilization_count": 0,
            "max_orthogonality_error": 0.0,
        }), "initial observer must be the exact empty constructor state")
        state._validate_rng(value["rng"], profile)
        state._assert_no_tensor_aliases(value)
        # Enforce exact built-in primitive/container types throughout fields
        # delegated to state_core validators (str/list subclasses are invalid).
        codec.tree_digest(value)
        return value
    except SourceHistoryError:
        raise
    except (ValueError, KeyError, TypeError) as exc:
        raise SourceHistoryError("initial-state validation failed") from exc


def _initial_digest(value: Any) -> dict[str, Any]:
    validate_initial_fingerprint(value)
    return {
        "state_sha256": codec.tree_digest(value),
        "model_sha256": codec.tree_digest(value["model"]),
        "optimizer_sha256": codec.tree_digest(value["optimizer"]),
        "observer_sha256": codec.tree_digest(value["observer"]),
        "rng_sha256": codec.tree_digest(value["rng"]),
    }


def compare_initial(on_initial: Any, off_initial: Any) -> dict[str, Any]:
    """Derive initial equality directly from two full counter-zero snapshots."""
    validate_initial_fingerprint(on_initial)
    validate_initial_fingerprint(off_initial)
    _require(on_initial["profile"] == off_initial["profile"], "initial profiles differ")
    equal = {
        "model": _same_exact(on_initial["model"], off_initial["model"]),
        "optimizer": _same_exact(on_initial["optimizer"], off_initial["optimizer"]),
        "observer": _same_exact(on_initial["observer"], off_initial["observer"]),
        "rng": _same_exact(on_initial["rng"], off_initial["rng"]),
        "state": _same_exact(on_initial, off_initial),
    }
    on, off = _initial_digest(on_initial), _initial_digest(off_initial)
    return {
        "on": on, "off": off,
        "model_direct_typed_equal": equal["model"],
        "optimizer_direct_typed_equal": equal["optimizer"],
        "observer_direct_typed_equal": equal["observer"],
        "rng_direct_typed_equal": equal["rng"],
        "state_direct_typed_equal": equal["state"],
    }


def _validate_initial_comparison(value: Any) -> None:
    _keys(value, INITIAL_COMPARISON_KEYS, "initial comparison")
    for side in ("on", "off"):
        _keys(value[side], INITIAL_FINGERPRINT_KEYS, "initial comparison." + side)
        for key in INITIAL_FINGERPRINT_KEYS:
            _sha(value[side][key], "initial comparison." + side + "." + key)
    pairs = (
        ("model_direct_typed_equal", "model_sha256"),
        ("optimizer_direct_typed_equal", "optimizer_sha256"),
        ("observer_direct_typed_equal", "observer_sha256"),
        ("rng_direct_typed_equal", "rng_sha256"),
        ("state_direct_typed_equal", "state_sha256"),
    )
    for flag, digest in pairs:
        _require(type(value[flag]) is bool and
                 value[flag] == (value["on"][digest] == value["off"][digest]),
                 flag + " inconsistent with retained initial digest evidence")


def _prefix(profile: str) -> str:
    return "i7-native" if profile == SCIENTIFIC else "i7-fixture"


def _source_names(profile: str, track: dict[str, Any], mode: str) -> tuple[str, str]:
    stem = (f"{_prefix(profile)}-{track['execution_role']}-b{track['bundle']}"
            f"-source-{mode}")
    return stem, stem + ".pt"


def _comparison_names(profile: str, track: dict[str, Any]) -> tuple[str, str]:
    stem = f"{_prefix(profile)}-{track['execution_role']}-b{track['bundle']}-capture-comparison"
    return stem, stem + ".json"


def _validate_receipt_fields(value: Any, *, expected_name: str) -> None:
    _require(type(value["name"]) is str and value["name"] == expected_name and
             type(value["status"]) is str and value["status"] == "complete" and
             type(value["encoding"]) is str and
             value["encoding"] == "torch_weights_only" and
             type(value["size_bytes"]) is int and
             0 < value["size_bytes"] <= storage.DEFAULT_BUDGET,
             "artifact reference metadata differs")
    _sha(value["sha256"], "artifact reference sha256")
    expected_receipt = "receipt-" + hashlib.sha256(expected_name.encode("ascii")).hexdigest() + ".json"
    receipt_base = {
        "schema": "i7_artifact_receipt_v1", "name": expected_name,
        "size": value["size_bytes"], "sha256": value["sha256"],
        "status": "complete", "encoding": "torch_weights_only",
    }
    receipt_bytes = storage._json_bytes(receipt_base)
    _require(type(value["receipt_name"]) is str and
             value["receipt_name"] == expected_receipt and
             type(value["receipt_size_bytes"]) is int and
             0 < value["receipt_size_bytes"] <= storage._RECEIPT_MAX and
             value["receipt_size_bytes"] == len(receipt_bytes) and
             type(value["receipt_sha256"]) is str and
             value["receipt_sha256"] == hashlib.sha256(receipt_bytes).hexdigest(),
             "artifact receipt reference differs")


def _validate_plan_ref(value: Any, profile: str, track: dict[str, Any]) -> None:
    _keys(value, PLAN_REF_KEYS, "plan reference")
    # The registered MLP integration fixture already seals this exact name; do
    # not invent a fixture analogue of the native plan naming policy.
    expected = (f"{_prefix(profile)}-b{track['bundle']}-plan.pt"
                if profile == SCIENTIFIC else "frozen-plan.pt")
    _validate_receipt_fields(value, expected_name=expected)


def _identity_for(track: dict[str, Any], update: int) -> dict[str, Any]:
    return {
        "run_id": track["run_id"], "iteration": track["iteration"],
        "execution_role": track["execution_role"], "evidence_role": track["evidence_role"],
        "bundle": track["bundle"], "anchor_update": update,
        "source_policy": track["source_policy"], "condition": track["condition"],
        "steps_total": track["steps_total"], "anchor_phase": "pre_forward_pre_observe",
        "anchor_completed_updates": update - 1,
        "anchor_completed_observations": update - 1,
        "rng_namespace_prefix": [20260906, track["bundle"]],
        "stream_roles": dict(codec.STREAM_ROLES),
    }


def _validate_artifact_ref(value: Any, *, identity: dict[str, Any], profile: str,
                           kind: str) -> None:
    _keys(value, ARTIFACT_REF_KEYS, kind + " reference")
    artifact_id = codec.artifact_id(identity, profile=profile, kind=kind)
    schema_name = "i7_anchor" if kind == "anchor" else "i7_source_step_witness"
    _require(type(value["artifact_id"]) is str and value["artifact_id"] == artifact_id and
             type(value["schema_name"]) is str and value["schema_name"] == schema_name,
             kind + " reference identity differs")
    _validate_receipt_fields(value, expected_name=artifact_id + ".pt")


def _validate_anchor_witness_refs(value: Any, *, profile: str,
                                  track: dict[str, Any], capture_mode: str) -> None:
    _require(type(value) is list, "anchor_witness_refs must be a list")
    expected = track["anchor_updates"] if capture_mode == "capture_on" else []
    _require(len(value) == len(expected), "capture mode has wrong anchor/witness reference count")
    for row, update in zip(value, expected):
        _keys(row, ANCHOR_WITNESS_KEYS, "pair")
        _require(type(row["anchor_update"]) is int and row["anchor_update"] == update,
                 "anchor/witness references missing, duplicated, or reordered")
        identity = _identity_for(track, update)
        _validate_artifact_ref(row["anchor_ref"], identity=identity, profile=profile, kind="anchor")
        _validate_artifact_ref(row["witness_ref"], identity=identity, profile=profile,
                               kind="source-witness")


def _validate_instrumentation(value: Any) -> None:
    _keys(value, INSTRUMENTATION_KEYS, "instrumentation")
    _require(_same_exact(value, INSTRUMENTATION), "instrumentation semantics differ")


def _validate_evidence_scope(value: Any, expected: dict[str, Any]) -> None:
    _keys(value, EVIDENCE_SCOPE_KEYS, "evidence scope")
    _require(_same_exact(value, expected), "evidence scope differs")


def make_source_completion(step_cores: Any, *, identity: Any, profile: str,
                           capture_mode: str, plan_ref: Any, anchor_refs: Any,
                           witness_refs: Any, sources_sha256: str,
                           environment_sha256: str) -> dict[str, Any]:
    """Consume exactly one full core per source update and make a compact record."""
    track = trajectory(identity, profile)
    _require(capture_mode in CAPTURE_MODES, "unknown capture mode")
    _require(capture_mode == "capture_on" or track["execution_role"] == "pilot" or
             profile == MLP_FIXTURE,
             "capture_off is reserved for the pilot comparison")
    _require(type(anchor_refs) is list and type(witness_refs) is list and
             len(anchor_refs) == len(witness_refs), "anchor/witness inputs must be paired lists")
    expected_reference_count = len(track["anchor_updates"]) if capture_mode == "capture_on" else 0
    _require(len(anchor_refs) == expected_reference_count,
             "capture mode requires the exact anchor/witness reference count")
    pairs = [{"anchor_update": update, "anchor_ref": _clone(anchor),
              "witness_ref": _clone(witness)}
             for update, anchor, witness in zip(track["anchor_updates"], anchor_refs, witness_refs)]
    iterator = iter(step_cores)
    trace = []
    final_core = None
    for update in range(1, track["steps_total"] + 1):
        try:
            core = next(iterator)
        except StopIteration as exc:
            raise SourceHistoryError("source core stream ended before steps_total") from exc
        _require(type(core) is dict and core.get("profile") == profile, "source core profile differs")
        row = step_fingerprint(core)
        _require(row["completed_updates"] == update,
                 "source cores contain a missing, duplicate, or reordered update")
        trace.append(row)
        if update == track["steps_total"]:
            final_core = _clone(core)
    try:
        next(iterator)
    except StopIteration:
        pass
    else:
        raise SourceHistoryError("source core stream exceeds steps_total")
    stem, name = _source_names(profile, track, capture_mode)
    value = {
        "schema_name": "i7_source_completion", "schema_version": 1,
        "profile": profile, "artifact_id": stem, "artifact_name": name,
        "trajectory": track, "capture_mode": capture_mode,
        "instrumentation": dict(INSTRUMENTATION), "plan_ref": _clone(plan_ref),
        "provenance": {"sources_sha256": sources_sha256,
                       "environment_sha256": environment_sha256},
        "anchor_witness_refs": pairs, "trace": trace,
        "final_state_core": final_core, "evidence_scope": dict(SOURCE_EVIDENCE_SCOPE),
        "scientific_execution_certified": False,
    }
    return validate_source_completion(value)


def validate_source_completion(value: Any) -> dict[str, Any]:
    """Pure validation; prior full cores and external receipt bytes are unavailable."""
    try:
        _keys(value, SOURCE_ROOT_KEYS, "source completion")
        _require(type(value["schema_name"]) is str and
                 value["schema_name"] == "i7_source_completion" and
                 type(value["schema_version"]) is int and value["schema_version"] == 1 and
                 type(value["profile"]) is str and value["profile"] in PROFILES,
                 "source completion root differs")
        profile = value["profile"]
        track = _validate_trajectory(value["trajectory"], profile)
        mode = value["capture_mode"]
        _require(type(mode) is str and mode in CAPTURE_MODES, "capture mode invalid")
        _require(mode == "capture_on" or track["execution_role"] == "pilot" or
                 profile == MLP_FIXTURE, "capture_off role invalid")
        stem, name = _source_names(profile, track, mode)
        _require(type(value["artifact_id"]) is str and value["artifact_id"] == stem and
                 type(value["artifact_name"]) is str and value["artifact_name"] == name,
                 "source completion artifact name differs")
        _validate_instrumentation(value["instrumentation"])
        _validate_plan_ref(value["plan_ref"], profile, track)
        _keys(value["provenance"], PROVENANCE_KEYS, "provenance")
        _sha(value["provenance"]["sources_sha256"], "sources_sha256")
        _sha(value["provenance"]["environment_sha256"], "environment_sha256")
        _validate_anchor_witness_refs(value["anchor_witness_refs"], profile=profile,
                                      track=track, capture_mode=mode)
        trace = value["trace"]
        _require(type(trace) is list and len(trace) == track["steps_total"],
                 "source trace must contain every update exactly once")
        for update, row in enumerate(trace, 1):
            _validate_fingerprint(row, expected_update=update)
        final_core = value["final_state_core"]
        _require(type(final_core) is dict and final_core.get("profile") == profile,
                 "final core profile differs")
        final_row = step_fingerprint(final_core)
        _require(final_row["completed_updates"] == track["steps_total"] and
                 _same_exact(final_row, trace[-1]), "retained final core differs from final trace row")
        _validate_evidence_scope(value["evidence_scope"], SOURCE_EVIDENCE_SCOPE)
        _require(value["scientific_execution_certified"] is False,
                 "source completion cannot certify scientific execution")
        return value
    except SourceHistoryError:
        raise
    except (ValueError, KeyError, TypeError, codec.CodecError) as exc:
        raise SourceHistoryError("source completion validation failed") from exc


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "steps_compared": len(rows),
        "model_equal_steps": sum(row["model_direct_typed_equal"] for row in rows),
        "optimizer_equal_steps": sum(row["optimizer_direct_typed_equal"] for row in rows),
        "moments_equal_steps": sum(row["moments_direct_typed_equal"] for row in rows),
        "observer_equal_steps": sum(row["observer_direct_typed_equal"] for row in rows),
        "rng_equal_steps": sum(row["rng_direct_typed_equal"] for row in rows),
        "core_equal_steps": sum(row["core_direct_typed_equal"] for row in rows),
    }


def make_capture_comparison(step_rows: Any, *, initial_states: Any, identity: Any,
                            profile: str) -> dict[str, Any]:
    """Build paired capture-on/off evidence from actual full states only."""
    track = trajectory(identity, profile)
    _require(track["execution_role"] == "pilot" or profile == MLP_FIXTURE,
             "capture comparison is pilot-only")
    _require(type(initial_states) in (list, tuple) and len(initial_states) == 2,
             "two actual full initial snapshots are required")
    initial = compare_initial(initial_states[0], initial_states[1])
    _require(initial_states[0]["profile"] == profile, "initial comparison profile differs")
    _require(initial["state_direct_typed_equal"] is True,
             "capture arms must enter update one from the same full initial state")
    iterator = iter(step_rows)
    rows = []
    for update in range(1, track["steps_total"] + 1):
        try:
            pair = next(iterator)
        except StopIteration as exc:
            raise SourceHistoryError("comparison stream ended before steps_total") from exc
        _require(type(pair) in (list, tuple) and len(pair) == 2,
                 "comparison row must contain actual on/off full cores")
        _require(type(pair[0]) is dict and pair[0].get("profile") == profile,
                 "comparison on-core profile differs")
        row = compare_step(pair[0], pair[1])
        _require(row["completed_updates"] == update,
                 "comparison cores contain a missing, duplicate, or reordered update")
        rows.append(row)
    try:
        next(iterator)
    except StopIteration:
        pass
    else:
        raise SourceHistoryError("comparison stream exceeds steps_total")
    stem, name = _comparison_names(profile, track)
    value = {
        "schema_name": "i7_capture_comparison", "schema_version": 1,
        "profile": profile, "artifact_id": stem, "artifact_name": name,
        "trajectory": track, "instrumentation": dict(INSTRUMENTATION),
        "initial_comparison": initial, "step_trace": rows, "summary": _summary(rows),
        "evidence_scope": dict(COMPARISON_EVIDENCE_SCOPE),
        "scientific_execution_certified": False,
    }
    return validate_capture_comparison(value)


def validate_capture_comparison(value: Any) -> dict[str, Any]:
    """Pure trace validation, not recomputation from discarded historical cores."""
    try:
        _keys(value, CAPTURE_ROOT_KEYS, "capture comparison")
        _require(type(value["schema_name"]) is str and
                 value["schema_name"] == "i7_capture_comparison" and
                 type(value["schema_version"]) is int and value["schema_version"] == 1 and
                 type(value["profile"]) is str and value["profile"] in PROFILES,
                 "capture comparison root differs")
        profile = value["profile"]
        track = _validate_trajectory(value["trajectory"], profile)
        _require(track["execution_role"] == "pilot" or profile == MLP_FIXTURE,
                 "capture comparison role invalid")
        stem, name = _comparison_names(profile, track)
        _require(type(value["artifact_id"]) is str and value["artifact_id"] == stem and
                 type(value["artifact_name"]) is str and value["artifact_name"] == name,
                 "capture comparison artifact name differs")
        _validate_instrumentation(value["instrumentation"])
        _validate_initial_comparison(value["initial_comparison"])
        _require(value["initial_comparison"]["state_direct_typed_equal"] is True,
                 "capture arms do not share the same initial-state binding")
        rows = value["step_trace"]
        _require(type(rows) is list and len(rows) == track["steps_total"],
                 "capture trace must contain every paired update exactly once")
        for update, row in enumerate(rows, 1):
            _validate_comparison(row, expected_update=update)
        _keys(value["summary"], SUMMARY_KEYS, "comparison summary")
        _require(_same_exact(value["summary"], _summary(rows)),
                 "comparison summary not derived from trace")
        _validate_evidence_scope(value["evidence_scope"], COMPARISON_EVIDENCE_SCOPE)
        _require(value["scientific_execution_certified"] is False,
                 "capture comparison cannot certify scientific execution")
        return value
    except SourceHistoryError:
        raise
    except (ValueError, KeyError, TypeError, codec.CodecError) as exc:
        raise SourceHistoryError("capture comparison validation failed") from exc


def validate_capture_bundle(on_completion: Any, off_completion: Any,
                            comparison: Any) -> tuple[dict[str, Any],
                                                      dict[str, Any],
                                                      dict[str, Any]]:
    """Link the two completions and compact comparison as one pure evidence set.

    This verifies in-memory record consistency only.  It reads no artifact
    bytes, proves no live history, and grants no execution or GO authority.
    Every historical fingerprint is linked exactly, while equality of the two
    retained final full cores is recomputed directly from their typed bytes.
    """
    on = validate_source_completion(on_completion)
    off = validate_source_completion(off_completion)
    paired = validate_capture_comparison(comparison)
    _require(on["capture_mode"] == "capture_on" and
             off["capture_mode"] == "capture_off",
             "capture bundle requires ordered on/off source completions")
    _require(type(on["profile"]) is str and on["profile"] == off["profile"] and
             on["profile"] == paired["profile"],
             "capture bundle profiles differ")
    _require(_same_exact(on["trajectory"], off["trajectory"]) and
             _same_exact(on["trajectory"], paired["trajectory"]),
             "capture bundle trajectories differ")
    _require(_same_exact(on["plan_ref"], off["plan_ref"]),
             "capture bundle plan references differ")
    _require(_same_exact(on["provenance"], off["provenance"]),
             "capture bundle source/environment digests differ")
    _require(len(on["trace"]) == len(off["trace"]) == len(paired["step_trace"]),
             "capture bundle trace lengths differ")
    for update, (on_row, off_row, comparison_row) in enumerate(
            zip(on["trace"], off["trace"], paired["step_trace"]), 1):
        _require(comparison_row["completed_updates"] == update and
                 _same_exact(on_row, comparison_row["on"]) and
                 _same_exact(off_row, comparison_row["off"]),
                 "capture bundle trace linkage differs at update " + str(update))
    final_comparison = compare_step(on["final_state_core"], off["final_state_core"])
    _require(_same_exact(final_comparison, paired["step_trace"][-1]),
             "capture bundle final full-core equality differs")
    return on, off, paired
