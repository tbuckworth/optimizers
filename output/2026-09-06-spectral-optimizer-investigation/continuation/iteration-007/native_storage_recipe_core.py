"""Explicit symbolic storage recipes for nine fixed I7 component classes.

The returned trees contain :class:`TensorSlot` markers, never tensors.  They
mirror complete ordered container layouts for a serializer diagnostic, but are
deliberately unregistered and nonsemantic.  No producer, validator, plan,
dataset, training, serialization, RNG, CUDA, or filesystem operation occurs.
"""
from __future__ import annotations

import hashlib
from typing import Any

from native_storage_recipe_common import PROFILE, SLOT, copy_primitive


RUN_ID = "2026-09-06-spectral-optimizer-investigation"
NATIVE_PREFIX = "i7-recipe"
PARAMETERS = (("0.weight", (64, 784)), ("0.bias", (64,)),
              ("2.weight", (10, 64)), ("2.bias", (10,)))
STREAM_ROLES = {0: "permutation", 1: "replacement_uniforms", 2: "replacement_digits",
                3: "initialization_seed", 4: "training_batches",
                5: "training_probe", 6: "unused"}
GROUP = {"lr": .001, "betas": (.9, .999), "eps": 1e-8,
         "weight_decay": .01, "amsgrad": False, "maximize": False,
         "foreach": False, "capturable": False, "differentiable": False,
         "fused": False, "decoupled_weight_decay": True,
         "param_indices": [0, 1, 2, 3]}
OBSERVER_CONFIG = {
    "rank": 32, "decay": .99, "warmup": 100, "filter_strength": 1.,
    "energy_threshold": None, "adaptive": "none", "normalize": "none",
    "weighting": "hard", "alpha": 1., "soft_residual": True,
    "stable_update": True, "relative_eig_tol": 1e-8,
    "absolute_eig_floor": 0., "stabilize_every": 100, "n_params": 50_890,
}
EVENTS = [
    "anchor_core_captured_and_validated", "anchor_envelope_sealed",
    "same_live_objects_recaptured", "complete_core_directly_equal",
    "actual_live_forward_and_backward", "actual_live_observer_ingest",
    "actual_live_optimizer_step", "source_witness_payload_assembled",
]
INSTRUMENTATION = {
    "both_modes_common": "state_core_hash_after_every_completed_update",
    "capture_on_additional": "seal_declared_anchor_and_source_witness_artifacts",
    "capture_off_additional": "none",
    "baseline_kind": "instrumented_state_hash_baseline_not_uninstrumented",
}
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
COMPONENTS = (
    "anchor_pilot", "anchor_long", "source_witness",
    "source_completion_pilot_on", "source_completion_pilot_off",
    "source_completion_long_on", "plan_pilot", "plan_long",
    "capture_comparison_pilot",
)
KINDS = ("anchor", "source-witness", "branch-results", "independent-audit")
SCHEMAS = {"anchor": "i7_anchor", "source-witness": "i7_source_step_witness",
           "branch-results": "i7_branch_results",
           "independent-audit": "i7_anchor_numerical_audit"}
PAYLOAD_TENSOR_BYTES = {"anchor": 7_378_608, "source-witness": 7_735_552,
                        "branch-results": 32_977_072, "independent-audit": 0}


class StorageRecipeCoreError(ValueError):
    """A requested symbolic component is outside the fixed structural recipe."""


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise StorageRecipeCoreError(message)


def _sha(label: str) -> str:
    return hashlib.sha256(("i7-storage-recipe:" + label).encode("ascii")).hexdigest()


def _identity(role: str, bundle: int, update: int) -> dict[str, Any]:
    steps = 220 if role == "pilot" else 2_000
    allowed = ((role == "pilot" and bundle == 71990 and update in (101, 200)) or
               (role == "primary" and bundle in (71001, 71002, 71003)
                and update in (101, 500, 1000, 2000)) or
               (role == "sensitivity" and bundle == 71901
                and update in (101, 500, 1000, 2000)))
    _need(type(role) is str and type(bundle) is int and type(update) is int and allowed,
          "identity is outside the fixed schedule")
    return {
        "run_id": RUN_ID, "iteration": 7, "execution_role": role,
        "evidence_role": "development" if role == "pilot" else role,
        "bundle": bundle, "anchor_update": update, "source_policy": "current32",
        "condition": "noise_0.9", "steps_total": steps,
        "anchor_phase": "pre_forward_pre_observe",
        "anchor_completed_updates": update - 1,
        "anchor_completed_observations": update - 1,
        "rng_namespace_prefix": [20260906, bundle],
        "stream_roles": dict(STREAM_ROLES),
    }


def _artifact_id(role: str, bundle: int, update: int, kind: str) -> str:
    _identity(role, bundle, update)
    _need(kind in KINDS, "unknown recipe artifact kind")
    return f"{PROFILE}--{role}--b{bundle}--u{update}--{kind}"


def native_tensor_metadata(shape, dtype="torch.float32", device="cuda:0") -> dict:
    """Return an ordered native-metadata mirror with one symbolic tensor slot."""
    _need(type(shape) is tuple and all(type(item) is int and item >= 0 for item in shape),
          "tensor shape must be an exact nonnegative tuple")
    _need(type(dtype) is str and type(device) is str, "tensor dtype/device must be strings")
    return {"value": SLOT, "shape": list(shape),
            "native_dtype": dtype, "native_device": device}


def parameter_rows() -> list[dict]:
    return [{"index": index, "name": name, "shape": list(shape),
             "requires_grad": False, "native_dtype": "torch.float32",
             "native_device": "cuda:0", "value": SLOT}
            for index, (name, shape) in enumerate(PARAMETERS)]


def _model() -> dict:
    return {
        "architecture": "Sequential(Linear(784,64),ReLU(),Linear(64,10))",
        "parameter_order": [name for name, _ in PARAMETERS],
        "parameters": parameter_rows(), "buffer_order": [], "buffers": [],
        "module_order": ["", "0", "1", "2"],
        "training_modes": [False] * 4, "gradient_null_mask": [False] * 4,
        "gradients": [None] * 4,
    }


def optimizer(*, completed_updates: int = 2_001) -> dict:
    _need(type(completed_updates) is int and 0 <= completed_updates <= 2_001,
          "optimizer counter outside structural domain")
    states = []
    for index, (name, shape) in enumerate(PARAMETERS):
        states.append({"parameter_index": index, "parameter_name": name, "step": SLOT,
                       "exp_avg": native_tensor_metadata(shape),
                       "exp_avg_sq": native_tensor_metadata(shape)})
    return {"class_name": "torch.optim.AdamW",
            "state_completed_updates": completed_updates,
            "parameter_order": [name for name, _ in PARAMETERS],
            "param_groups": [copy_primitive(GROUP)], "state": states}


def observer(*, completed_observations: int = 2_001) -> dict:
    _need(type(completed_observations) is int and 0 <= completed_observations <= 2_001,
          "observer counter outside structural domain")
    return {
        "class_name": "SpectralGradientFilter",
        "state_completed_observations": completed_observations,
        "excluded_live_aliases": ["model", "base_optimizer", "param_list"],
        "config": copy_primitive(OBSERVER_CONFIG),
        "state": {
            "V": native_tensor_metadata((50_890, 32)),
            "S": native_tensor_metadata((32,), "torch.float64", "cpu"),
            "proj_k": None, "step_count": completed_observations,
            "grad_mean": native_tensor_metadata((50_890,)),
            "stabilization_count": 4_000, "max_orthogonality_error": 0.0,
        },
    }


def _rng() -> dict:
    python = {"version": 3, "internal": [0xFFFFFFFF] * 624 + [624],
              "gauss_next": 0.0}
    numpy = {"algorithm": "MT19937", "keys": SLOT, "position": 624,
             "has_gauss": 1, "cached_gaussian": 0.0}
    cuda = [{"device_index": 0, "name": "x" * 8_192,
             "uuid": "x" * 40, "state": SLOT}]
    witness = {"spec": "four_draws_v1", "python": [0.0] * 4,
               "numpy": [0.0] * 4, "torch_cpu": SLOT,
               "torch_cuda": [SLOT]}
    return {"python": python, "numpy": numpy, "torch_cpu": SLOT,
            "torch_cuda": cuda, "continuation_witness": witness}


def _state_core() -> dict:
    # The analytic core theorem bounds these fields independently over
    # 0..2001.  A structural maximum is intentionally not a valid live core.
    counter = 2_001
    return {
        "schema_name": "i7_state_core", "schema_version": 1,
        "profile": PROFILE, "phase": "pre_forward_pre_observe",
        "anchor_update": counter, "state_completed_updates": counter,
        "model": _model(), "optimizer": optimizer(completed_updates=counter),
        "observer": observer(completed_observations=counter), "rng": _rng(),
    }


def envelope(payload, *, role, bundle, update, kind) -> dict:
    """Wrap a copied payload in the exact ordered common envelope shape."""
    _need(kind in KINDS, "unknown recipe envelope kind")
    identifier = _artifact_id(role, bundle, update, kind)
    return {"schema_name": SCHEMAS[kind], "schema_version": 1,
            "profile": PROFILE, "artifact_id": identifier,
            "created_utc": "9999-12-31T23:59:59Z",
            "identity": _identity(role, bundle, update),
            "payload": copy_primitive(payload),
            "payload_tensor_bytes": PAYLOAD_TENSOR_BYTES[kind]}


def _array_specs(steps: int) -> dict[str, tuple[tuple[int, ...], str]]:
    _need(type(steps) is int and steps in (220, 2_000), "unknown plan length")
    return {
        "permutation": ((60_000,), "torch.int64"),
        "train_indices": ((5_000,), "torch.int64"),
        "validation_indices": ((5_000,), "torch.int64"),
        "auxiliary_indices": ((5_000,), "torch.int64"),
        "replacement_uniforms": ((5_000,), "torch.float64"),
        "replacement_digits": ((5_000,), "torch.int64"),
        "training_batches": ((steps, 64), "torch.int64"),
        "training_probe_indices": ((256,), "torch.int64"),
    }


def _array_reference(shape, dtype, role, label) -> dict:
    return {"sha256": _sha(label), "dtype": dtype, "shape": list(shape),
            "semantic_role": role}


def _plan_binding(steps: int) -> dict:
    return {
        "artifact": {"sha256": _sha("plan-artifact"), "size_bytes": 4 << 20},
        "arrays": {name: _array_reference(shape, dtype, name, "plan-array-" + name)
                   for name, (shape, dtype) in _array_specs(steps).items()},
        "initialization_seed": {"value": 0xFFFFFFFF,
                                "sha256": _sha("initialization-seed")},
        "next_batch_row": steps - 1, "next_batch_indices": SLOT,
    }


def _data_binding() -> dict:
    arrays = {}
    for name in ("train_inputs", "train_clean_labels", "train_noisy_labels",
                 "auxiliary_inputs", "auxiliary_clean_labels"):
        inputs = "inputs" in name
        arrays[name] = _array_reference((5_000, 784) if inputs else (5_000,),
                                        "torch.float32" if inputs else "torch.int64",
                                        name, "data-array-" + name)
    return {
        "idx_files": {
            "training_images": {"sha256": _sha("training-images"),
                                "size_bytes": 47_040_016},
            "training_labels": {"sha256": _sha("training-labels"),
                                "size_bytes": 60_008},
        },
        "preprocessing": "mnist_training_idx_uint8_to_torch_float32_div255_v1",
        "arrays": arrays, "replacement_count": 5_000,
        "incorrect_label_count": 5_000,
    }


def _probe_binding() -> dict:
    def inputs(count, role):
        return _array_reference((count, 784), "torch.float32", role, role)
    def labels(count, role):
        return _array_reference((count,), "torch.int64", role, role)
    chunks = [{"chunk_index": index, "start": 500 * index, "end": 500 * (index + 1),
               "inputs": inputs(500, "auxiliary_chunk_inputs"),
               "labels": labels(500, "auxiliary_chunk_labels")}
              for index in range(10)]
    return {
        "next_batch": {"row": 1_999,
                       "coordinate_space": "train_subset_position_v1",
                       "inputs": inputs(64, "batch_noisy_inputs"),
                       "noisy_labels": labels(64, "batch_noisy_labels")},
        "training_probe_indices": SLOT,
        "training_probe_coordinate_space": "train_subset_position_v1",
        "training_probe_inputs": inputs(256, "training_probe_inputs"),
        "training_probe_clean_labels": labels(256, "training_probe_clean_labels"),
        "training_probe_noisy_labels": labels(256, "training_probe_noisy_labels"),
        "auxiliary_indices": SLOT,
        "auxiliary_coordinate_space": "original_training_idx_row_v1",
        "auxiliary_inputs": inputs(5_000, "auxiliary_inputs"),
        "auxiliary_clean_labels": labels(5_000, "auxiliary_clean_labels"),
        "auxiliary_chunks": chunks, "stream6": "unused",
    }


def _anchor(*, role: str, bundle: int, update: int,
            sources: dict, environment: dict) -> dict:
    payload = {
        "model": _model(), "optimizer": optimizer(), "observer": observer(), "rng": _rng(),
        "bindings": {"plan": _plan_binding(220 if role == "pilot" else 2_000),
                     "probes": _probe_binding(), "data": _data_binding(),
                     "sources": copy_primitive(sources),
                     "environment": copy_primitive(environment)},
    }
    return envelope(payload, role=role, bundle=bundle, update=update, kind="anchor")


def _reference(*, role: str, bundle: int, update: int, kind: str) -> dict:
    identifier = _artifact_id(role, bundle, update, kind)
    return {"artifact_id": identifier, "schema_name": SCHEMAS[kind],
            "name": identifier + ".pt", "size_bytes": 1 << 30,
            "sha256": _sha(identifier), "status": "complete",
            "encoding": "torch_weights_only", "receipt_name": "r" * 77,
            "receipt_size_bytes": 4_096, "receipt_sha256": _sha("receipt-" + identifier)}


def _witness(*, role: str, bundle: int, update: int) -> dict:
    source_step = {"anchor_update": update, "completed_updates_before": update - 1,
                   "completed_observations_before": update - 1,
                   "completed_updates_after": update,
                   "completed_observations_after": update,
                   "source_state_origin": "uninterrupted_live_objects_v1",
                   "anchor_deserializations_before_live_step": 0,
                   "observation_ingests_in_live_step": 1,
                   "optimizer_steps_in_live_step": 1}
    payload = {
        "anchor_ref": _reference(role=role, bundle=bundle, update=update, kind="anchor"),
        "source_step": source_step,
        "capture_proof": {
            "proof_id": "capture_neutrality_v1", "ordered_events": list(EVENTS),
            "anchor_deserializations_before_live_step": 0,
            "core_sha256_before": _sha("witness-core-before"),
            "core_sha256_after": _sha("witness-core-after"),
            "core_direct_typed_equal": False, "same_live_object_bindings": False,
            "anchor_sealed_before_live_step": False,
            "live_step_complete_at_payload_assembly": False,
        },
        "before_parameters_binding": {
            "anchor_artifact_id": _artifact_id(role, bundle, update, "anchor"),
            "anchor_model_parameters_tree_sha256": _sha("anchor-model-parameters"),
            "parameter_order_tree_sha256": _sha("parameter-order"),
        },
        "raw_gradient": native_tensor_metadata((50_890,)),
        "delivered_current_gradient": native_tensor_metadata((50_890,)),
        "parameters_after": parameter_rows(),
        "optimizer_after": optimizer(), "observer_after": observer(),
        "live_loss": {"probe_key": "batch_noisy", "count": 64,
                      "value_native_float32": 0.0,
                      "sample_identity_sha256": _sha("witness-live-loss")},
        "state_hashes": {name: _sha("witness-state-" + name) for name in (
            "parameters_before", "parameters_after", "raw_gradient",
            "delivered_current_gradient", "optimizer_after", "observer_after",
            "rng_before", "rng_after")},
    }
    return envelope(payload, role=role, bundle=bundle, update=update,
                    kind="source-witness")


def _plan_reference(bundle: int) -> dict:
    name = f"{NATIVE_PREFIX}-b{bundle}-plan.pt"
    return {"name": name, "status": "complete", "encoding": "torch_weights_only",
            "size_bytes": 4 << 20, "sha256": _sha(name),
            "receipt_name": "r" * 77, "receipt_size_bytes": 4_096,
            "receipt_sha256": _sha("receipt-" + name)}


def _source_completion(*, role: str, bundle: int, mode: str) -> dict:
    identity = _identity(role, bundle, 101)
    _need(mode == "capture_on" or role == "pilot" and mode == "capture_off",
          "invalid recipe capture mode")
    steps = identity["steps_total"]
    updates = [101, 200] if role == "pilot" else [101, 500, 1000, 2000]
    trajectory = {key: identity[key] for key in (
        "run_id", "iteration", "execution_role", "evidence_role", "bundle",
        "source_policy", "condition", "steps_total")}
    trajectory["anchor_updates"] = updates
    stem = f"{NATIVE_PREFIX}-{role}-b{bundle}-source-{mode}"
    pairs = [] if mode == "capture_off" else [
        {"anchor_update": update,
         "anchor_ref": _reference(role=role, bundle=bundle, update=update, kind="anchor"),
         "witness_ref": _reference(role=role, bundle=bundle, update=update,
                                    kind="source-witness")}
        for update in updates]
    trace = []
    for update in range(1, steps + 1):
        trace.append({
            "completed_updates": 2_001, "completed_observations": 2_001,
            "next_anchor_update": 2_001,
            **{name: _sha(f"{stem}-u{update}-{name}") for name in (
                "core_sha256", "model_sha256", "optimizer_sha256", "moments_sha256",
                "observer_sha256", "rng_sha256")},
        })
    return {
        "schema_name": "i7_source_completion", "schema_version": 1,
        "profile": PROFILE, "artifact_id": stem, "artifact_name": stem + ".pt",
        "trajectory": trajectory, "capture_mode": mode,
        "instrumentation": copy_primitive(INSTRUMENTATION),
        "plan_ref": _plan_reference(bundle),
        "provenance": {"sources_sha256": _sha(stem + "-sources"),
                       "environment_sha256": _sha(stem + "-environment")},
        "anchor_witness_refs": pairs, "trace": trace,
        "final_state_core": _state_core(),
        "evidence_scope": copy_primitive(SOURCE_EVIDENCE_SCOPE),
        "scientific_execution_certified": False,
    }


def _plan(bundle: int) -> dict:
    _need(bundle in (71990, 71001, 71002, 71003, 71901), "unknown plan bundle")
    steps = 220 if bundle == 71990 else 2_000
    return {"schema": "i7_frozen_plan_v1", "profile": PROFILE,
            "bundle": bundle, "steps_total": steps,
            "stream_roles": dict(STREAM_ROLES),
            **{name: SLOT for name in _array_specs(steps)},
            "initialization_seed": 0xFFFFFFFF}


def _fingerprint(label: str, update: int) -> dict:
    return {"completed_updates": update, "completed_observations": update,
            "next_anchor_update": update + 1,
            **{name: _sha(label + "-" + name) for name in (
                "core_sha256", "model_sha256", "optimizer_sha256", "moments_sha256",
                "observer_sha256", "rng_sha256")}}


def _initial_fingerprint(label: str) -> dict:
    return {name: _sha(label + "-" + name) for name in (
        "state_sha256", "model_sha256", "optimizer_sha256", "observer_sha256",
        "rng_sha256")}


def _comparison() -> dict:
    role, bundle, steps = "pilot", 71990, 220
    trajectory = {"run_id": RUN_ID, "iteration": 7, "execution_role": role,
                  "evidence_role": "development", "bundle": bundle,
                  "source_policy": "current32", "condition": "noise_0.9",
                  "steps_total": steps, "anchor_updates": [101, 200]}
    stem = f"{NATIVE_PREFIX}-{role}-b{bundle}-capture-comparison"
    initial = {"on": _initial_fingerprint("initial-on"),
               "off": _initial_fingerprint("initial-off"),
               "model_direct_typed_equal": False,
               "optimizer_direct_typed_equal": False,
               "observer_direct_typed_equal": False,
               "rng_direct_typed_equal": False,
               "state_direct_typed_equal": False}
    rows = []
    for update in range(1, steps + 1):
        rows.append({"completed_updates": update,
                     "on": _fingerprint(f"comparison-u{update}-on", update),
                     "off": _fingerprint(f"comparison-u{update}-off", update),
                     "model_direct_typed_equal": False,
                     "optimizer_direct_typed_equal": False,
                     "moments_direct_typed_equal": False,
                     "observer_direct_typed_equal": False,
                     "rng_direct_typed_equal": False,
                     "core_direct_typed_equal": False})
    return {
        "schema_name": "i7_capture_comparison", "schema_version": 1,
        "profile": PROFILE, "artifact_id": stem, "artifact_name": stem + ".json",
        "trajectory": trajectory, "instrumentation": copy_primitive(INSTRUMENTATION),
        "initial_comparison": initial, "step_trace": rows,
        "summary": {name: steps for name in (
            "steps_compared", "model_equal_steps", "optimizer_equal_steps",
            "moments_equal_steps", "observer_equal_steps", "rng_equal_steps",
            "core_equal_steps")},
        "evidence_scope": copy_primitive(COMPARISON_EVIDENCE_SCOPE),
        "scientific_execution_certified": False,
    }


def build_template(component, *, sources, environment):
    """Build one complete, unregistered symbolic component recipe.

    ``sources`` and ``environment`` must already have been collected and
    validated by the caller.  This builder copies but does not authenticate
    them; only anchor envelopes retain the full trees in the scientific schema.
    """
    _need(type(component) is str and component in COMPONENTS, "unknown core recipe component")
    _need(type(sources) is dict and type(environment) is dict,
          "source/environment templates must be dictionaries")
    if component == "anchor_pilot":
        return _anchor(role="pilot", bundle=71990, update=200,
                       sources=sources, environment=environment)
    if component == "anchor_long":
        return _anchor(role="sensitivity", bundle=71901, update=2000,
                       sources=sources, environment=environment)
    if component == "source_witness":
        return _witness(role="sensitivity", bundle=71901, update=2000)
    if component == "source_completion_pilot_on":
        return _source_completion(role="pilot", bundle=71990, mode="capture_on")
    if component == "source_completion_pilot_off":
        return _source_completion(role="pilot", bundle=71990, mode="capture_off")
    if component == "source_completion_long_on":
        return _source_completion(role="sensitivity", bundle=71901, mode="capture_on")
    if component == "plan_pilot":
        return _plan(71990)
    if component == "plan_long":
        return _plan(71901)
    return _comparison()


def main(argv=None) -> int:
    if argv not in (None, []):
        raise SystemExit("native_storage_recipe_core has no execution mode")
    print("I7 symbolic core storage recipes only; no tensors, serialization, data or execution.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
