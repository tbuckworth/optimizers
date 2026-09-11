"""Prospective primitive costs for the fixed native core/anchor/witness/history/plan.

Arithmetic only: no Torch, file reads, plan generation, data, RNG or runtime GO.
Every tensor leaf contributes zero here; reduction and storage bytes are separate.
See core-primitive-bound.md for source/domain and actual-admission obligations.
"""
from __future__ import annotations

from primitive_storage_bound import (
    BOOL, FLOAT, NULL, integer, json_subtree_bytes, literal, pdict, plist,
    repeat_list, text,
)

PROFILE = "scientific_mnist_current32_v1"
RUN_ID = "2026-09-06-spectral-optimizer-investigation"
PARAMETERS = (("0.weight", (64, 784)), ("0.bias", (64,)),
              ("2.weight", (10, 64)), ("2.bias", (10,)))
STREAM_ROLES = {0: "permutation", 1: "replacement_uniforms", 2: "replacement_digits",
                3: "initialization_seed", 4: "training_batches", 5: "training_probe", 6: "unused"}
GROUP = dict(lr=.001, betas=(.9, .999), eps=1e-8, weight_decay=.01,
             amsgrad=False, maximize=False, foreach=False, capturable=False,
             differentiable=False, fused=False, decoupled_weight_decay=True,
             param_indices=[0, 1, 2, 3])
OBSERVER_CONFIG = dict(rank=32, decay=.99, warmup=100, filter_strength=1.,
    energy_threshold=None, adaptive="none", normalize="none", weighting="hard",
    alpha=1., soft_residual=True, stable_update=True, relative_eig_tol=1e-8,
    absolute_eig_floor=0., stabilize_every=100, n_params=50890)
EVENTS = ["anchor_core_captured_and_validated", "anchor_envelope_sealed",
    "same_live_objects_recaptured", "complete_core_directly_equal",
    "actual_live_forward_and_backward", "actual_live_observer_ingest",
    "actual_live_optimizer_step", "source_witness_payload_assembled"]
INSTRUMENTATION = dict(both_modes_common="state_core_hash_after_every_completed_update",
    capture_on_additional="seal_declared_anchor_and_source_witness_artifacts",
    capture_off_additional="none",
    baseline_kind="instrumented_state_hash_baseline_not_uninstrumented")
SOURCE_EVIDENCE_SCOPE = dict(
    trace_origin="derived_at_construction_from_supplied_validated_full_state_cores",
    retained_full_state_scope="final_update_only",
    validator_recomputed_byte_evidence="retained_final_state_core_only",
    discarded_historical_state_bytes_recomputed=False,
    external_receipt_bytes_verified=False, live_history_reexecuted=False)
SHA = text(64)
TENSOR = 0
COUNTER = integer(0, 2001)
SOURCE_JSON_CAP, ENVIRONMENT_JSON_CAP = 32768, 8192


def fixed_identities():
    """18 immutable schedule tuples; not a scientific study plan."""
    return (("pilot", 71990, 101), ("pilot", 71990, 200), *(
        (role, bundle, update)
        for role, bundles in (("primary", (71001, 71002, 71003)), ("sensitivity", (71901,)))
        for bundle in bundles for update in (101, 500, 1000, 2000)))


def identity(role, bundle, update):
    if (type(role) is not str or type(bundle) is not int or type(update) is not int
            or (role, bundle, update) not in fixed_identities()):
        raise ValueError("identity outside the fixed native schedule")
    return dict(run_id=RUN_ID, iteration=7, execution_role=role,
        evidence_role="development" if role == "pilot" else role, bundle=bundle,
        anchor_update=update, source_policy="current32", condition="noise_0.9",
        steps_total=220 if role == "pilot" else 2000, anchor_phase="pre_forward_pre_observe",
        anchor_completed_updates=update-1, anchor_completed_observations=update-1,
        rng_namespace_prefix=[20260906, bundle], stream_roles=dict(STREAM_ROLES))


def artifact_id(role, bundle, update, kind):
    identity(role, bundle, update)
    if type(kind) is not str or kind not in (
            "anchor", "source-witness", "branch-results", "independent-audit"):
        raise ValueError("unknown native artifact kind")
    return f"{PROFILE}--{role}--b{bundle}--u{update}--{kind}"


def envelope_cost(payload_cost, *, role, bundle, update, kind):
    """Whole primitive tree cost; excludes root PROTO/STOP and all tensors."""
    schemas = {"anchor": "i7_anchor", "source-witness": "i7_source_step_witness",
               "branch-results": "i7_branch_results", "independent-audit": "i7_anchor_numerical_audit"}
    identifier = artifact_id(role, bundle, update, kind)
    return pdict(dict(schema_name=literal(schemas[kind]), schema_version=literal(1),
        profile=literal(PROFILE), artifact_id=literal(identifier), created_utc=text(20),
        identity=literal(identity(role, bundle, update)), payload=payload_cost,
        payload_tensor_bytes=integer(0, 64 << 20)))


def native_tensor_metadata(shape, dtype="torch.float32", device="cuda:0"):
    return pdict(dict(value=TENSOR, shape=literal(list(shape)),
                      native_dtype=literal(dtype), native_device=literal(device)))


def parameter_rows_cost():
    return plist([pdict(dict(index=literal(index), name=literal(name),
        shape=literal(list(shape)), requires_grad=BOOL, native_dtype=literal("torch.float32"),
        native_device=literal("cuda:0"), value=TENSOR))
        for index, (name, shape) in enumerate(PARAMETERS)])


def model_cost():
    return pdict(dict(architecture=literal("Sequential(Linear(784,64),ReLU(),Linear(64,10))"),
        parameter_order=literal([name for name, _ in PARAMETERS]), parameters=parameter_rows_cost(),
        buffer_order=literal([]), buffers=literal([]), module_order=literal(["", "0", "1", "2"]),
        training_modes=repeat_list(4, BOOL), gradient_null_mask=repeat_list(4, BOOL),
        gradients=repeat_list(4, NULL)))


def optimizer_cost():
    states = [pdict(dict(parameter_index=literal(index), parameter_name=literal(name),
        step=TENSOR, exp_avg=native_tensor_metadata(shape), exp_avg_sq=native_tensor_metadata(shape)))
        for index, (name, shape) in enumerate(PARAMETERS)]
    return pdict(dict(class_name=literal("torch.optim.AdamW"), state_completed_updates=COUNTER,
        parameter_order=literal([name for name, _ in PARAMETERS]),
        param_groups=plist([literal(GROUP)]), state=plist(states)))


def observer_cost():
    # A k<=32 shape has no wider integer encoding than k=32. Both None
    # and present states are admitted; taking each maximum is conservative.
    observer_state = pdict(dict(
        V=max(NULL, native_tensor_metadata((50890, 32))),
        S=max(NULL, native_tensor_metadata((32,), "torch.float64", "cpu")),
        proj_k=NULL, step_count=COUNTER,
        grad_mean=max(NULL, native_tensor_metadata((50890,))),
        stabilization_count=integer(0, 4000), max_orthogonality_error=FLOAT))
    return pdict(dict(class_name=literal("SpectralGradientFilter"),
        state_completed_observations=COUNTER,
        excluded_live_aliases=literal(["model", "base_optimizer", "param_list"]),
        config=literal(OBSERVER_CONFIG), state=observer_state))


def rng_cost():
    # The one native device's name is also present in admitted <=8KiB ASCII
    # JSON environment metadata; UTF8(name)<=escaped JSON bytes<=8192.
    py = pdict(dict(version=literal(3), internal=plist([
        *([integer(0, 0xFFFFFFFF)] * 624), integer(0, 624)]), gauss_next=max(NULL, FLOAT)))
    numpy = pdict(dict(algorithm=literal("MT19937"), keys=TENSOR,
        position=integer(0, 624), has_gauss=integer(0, 1), cached_gaussian=FLOAT))
    cuda = plist([pdict(dict(device_index=literal(0), name=text(ENVIRONMENT_JSON_CAP),
                             uuid=text(40), state=TENSOR))])
    witness = pdict(dict(spec=literal("four_draws_v1"), python=repeat_list(4, FLOAT),
        numpy=repeat_list(4, FLOAT), torch_cpu=TENSOR, torch_cuda=plist([TENSOR])))
    return pdict(dict(python=py, numpy=numpy, torch_cpu=TENSOR, torch_cuda=cuda,
                      continuation_witness=witness))


def core_cost():
    return pdict(dict(schema_name=literal("i7_state_core"), schema_version=literal(1),
        profile=literal(PROFILE), phase=literal("pre_forward_pre_observe"),
        anchor_update=COUNTER, state_completed_updates=COUNTER, model=model_cost(),
        optimizer=optimizer_cost(), observer=observer_cost(), rng=rng_cost()))


def _array_specs(steps):
    if type(steps) is not int or steps not in (220, 2000):
        raise ValueError("steps outside fixed pilot/long profile")
    return {"permutation": ((60000,), "torch.int64"),
        "train_indices": ((5000,), "torch.int64"),
        "validation_indices": ((5000,), "torch.int64"),
        "auxiliary_indices": ((5000,), "torch.int64"),
        "replacement_uniforms": ((5000,), "torch.float64"),
        "replacement_digits": ((5000,), "torch.int64"),
        "training_batches": ((steps, 64), "torch.int64"),
        "training_probe_indices": ((256,), "torch.int64")}


def _array_reference(shape, dtype, role):
    return pdict(dict(sha256=SHA, dtype=literal(dtype), shape=literal(list(shape)),
                      semantic_role=literal(role)))


def plan_binding_cost(steps):
    return pdict(dict(artifact=pdict(dict(sha256=SHA, size_bytes=integer(1, 4 << 20))),
        arrays=pdict({name: _array_reference(shape, dtype, name)
                      for name, (shape, dtype) in _array_specs(steps).items()}),
        initialization_seed=pdict(dict(value=integer(0, 2**32-1), sha256=SHA)),
        next_batch_row=integer(0, steps-1), next_batch_indices=TENSOR))


def data_binding_cost():
    arrays = {name: _array_reference((5000, 784) if "inputs" in name else (5000,),
              "torch.float32" if "inputs" in name else "torch.int64", name)
        for name in ("train_inputs", "train_clean_labels", "train_noisy_labels",
                     "auxiliary_inputs", "auxiliary_clean_labels")}
    return pdict(dict(idx_files=pdict({name: pdict(dict(sha256=SHA, size_bytes=literal(size)))
        for name, size in (("training_images", 47040016), ("training_labels", 60008))}),
        preprocessing=literal("mnist_training_idx_uint8_to_torch_float32_div255_v1"),
        arrays=pdict(arrays), replacement_count=integer(0, 5000),
        incorrect_label_count=integer(0, 5000)))


def probe_binding_cost():
    def inputs(n, role):
        return _array_reference((n, 784), "torch.float32", role)
    def labels(n, role):
        return _array_reference((n,), "torch.int64", role)
    chunks = [pdict(dict(chunk_index=literal(i), start=literal(500*i), end=literal(500*(i+1)),
        inputs=inputs(500, "auxiliary_chunk_inputs"), labels=labels(500, "auxiliary_chunk_labels")))
        for i in range(10)]
    return pdict(dict(next_batch=pdict(dict(row=integer(0, 1999),
        coordinate_space=literal("train_subset_position_v1"),
        inputs=inputs(64, "batch_noisy_inputs"), noisy_labels=labels(64, "batch_noisy_labels"))),
        training_probe_indices=TENSOR, training_probe_coordinate_space=literal("train_subset_position_v1"),
        training_probe_inputs=inputs(256, "training_probe_inputs"),
        training_probe_clean_labels=labels(256, "training_probe_clean_labels"),
        training_probe_noisy_labels=labels(256, "training_probe_noisy_labels"), auxiliary_indices=TENSOR,
        auxiliary_coordinate_space=literal("original_training_idx_row_v1"),
        auxiliary_inputs=inputs(5000, "auxiliary_inputs"),
        auxiliary_clean_labels=labels(5000, "auxiliary_clean_labels"),
        auxiliary_chunks=plist(chunks), stream6=literal("unused")))


def anchor_cost(*, role, bundle, update):
    value = identity(role, bundle, update)
    binding = pdict(dict(plan=plan_binding_cost(value["steps_total"]), probes=probe_binding_cost(),
        data=data_binding_cost(), sources=json_subtree_bytes(SOURCE_JSON_CAP),
        environment=json_subtree_bytes(ENVIRONMENT_JSON_CAP)))
    payload = pdict(dict(model=model_cost(), optimizer=optimizer_cost(), observer=observer_cost(),
                          rng=rng_cost(), bindings=binding))
    return envelope_cost(payload, role=role, bundle=bundle, update=update, kind="anchor")


def artifact_reference_cost(*, role, bundle, update, kind):
    identifier = artifact_id(role, bundle, update, kind)
    schemas = {"anchor": "i7_anchor", "source-witness": "i7_source_step_witness"}
    if kind not in schemas:
        raise ValueError("core/history reference kind must be anchor or witness")
    return pdict(dict(artifact_id=literal(identifier), schema_name=literal(schemas[kind]),
        name=literal(identifier + ".pt"), size_bytes=integer(1, 1 << 30), sha256=SHA,
        status=literal("complete"), encoding=literal("torch_weights_only"), receipt_name=text(77),
        receipt_size_bytes=integer(1, 4096), receipt_sha256=SHA))


def witness_cost(*, role, bundle, update):
    identity(role, bundle, update)
    source_step = dict(anchor_update=update, completed_updates_before=update-1,
        completed_observations_before=update-1, completed_updates_after=update,
        completed_observations_after=update, source_state_origin="uninterrupted_live_objects_v1",
        anchor_deserializations_before_live_step=0, observation_ingests_in_live_step=1,
        optimizer_steps_in_live_step=1)
    proof = pdict(dict(proof_id=literal("capture_neutrality_v1"), ordered_events=literal(EVENTS),
        anchor_deserializations_before_live_step=literal(0), core_sha256_before=SHA, core_sha256_after=SHA,
        core_direct_typed_equal=BOOL, same_live_object_bindings=BOOL,
        anchor_sealed_before_live_step=BOOL, live_step_complete_at_payload_assembly=BOOL))
    before = pdict(dict(anchor_artifact_id=literal(artifact_id(role, bundle, update, "anchor")),
        anchor_model_parameters_tree_sha256=SHA, parameter_order_tree_sha256=SHA))
    payload = pdict(dict(anchor_ref=artifact_reference_cost(role=role, bundle=bundle, update=update, kind="anchor"),
        source_step=literal(source_step), capture_proof=proof, before_parameters_binding=before,
        raw_gradient=native_tensor_metadata((50890,)), delivered_current_gradient=native_tensor_metadata((50890,)),
        parameters_after=parameter_rows_cost(), optimizer_after=optimizer_cost(), observer_after=observer_cost(),
        live_loss=pdict(dict(probe_key=literal("batch_noisy"), count=literal(64),
                            value_native_float32=FLOAT, sample_identity_sha256=SHA)),
        state_hashes=pdict({name: SHA for name in ("parameters_before", "parameters_after", "raw_gradient",
            "delivered_current_gradient", "optimizer_after", "observer_after", "rng_before", "rng_after")})))
    return envelope_cost(payload, role=role, bundle=bundle, update=update, kind="source-witness")


def plan_reference_cost(bundle):
    if type(bundle) is not int or bundle not in (71990, 71001, 71002, 71003, 71901):
        raise ValueError("unknown plan bundle")
    return pdict(dict(name=literal(f"i7-native-b{bundle}-plan.pt"), status=literal("complete"),
        encoding=literal("torch_weights_only"), size_bytes=integer(1, 4 << 20), sha256=SHA,
        receipt_name=text(77), receipt_size_bytes=integer(1, 4096), receipt_sha256=SHA))


def source_completion_cost(*, role, bundle, mode="capture_on"):
    item = identity(role, bundle, 101)
    if type(mode) is not str or not (mode == "capture_on" or role == "pilot" and mode == "capture_off"):
        raise ValueError("invalid source capture mode")
    steps = item["steps_total"]
    updates = [101, 200] if role == "pilot" else [101, 500, 1000, 2000]
    trajectory = {key: item[key] for key in ("run_id", "iteration", "execution_role",
        "evidence_role", "bundle", "source_policy", "condition", "steps_total")}
    trajectory["anchor_updates"] = updates
    stem = f"i7-native-{role}-b{bundle}-source-{mode}"
    pairs = [] if mode == "capture_off" else [pdict(dict(anchor_update=literal(update),
        anchor_ref=artifact_reference_cost(role=role, bundle=bundle, update=update, kind="anchor"),
        witness_ref=artifact_reference_cost(role=role, bundle=bundle, update=update, kind="source-witness")))
        for update in updates]
    # Every historical row is compact: 3 counters and 6 typed hashes, not cores.
    row = pdict({**{key: COUNTER for key in ("completed_updates", "completed_observations", "next_anchor_update")},
                 **{key: SHA for key in ("core_sha256", "model_sha256", "optimizer_sha256", "moments_sha256",
                                         "observer_sha256", "rng_sha256")}})
    return pdict(dict(schema_name=literal("i7_source_completion"), schema_version=literal(1),
        profile=literal(PROFILE), artifact_id=literal(stem), artifact_name=literal(stem + ".pt"),
        trajectory=literal(trajectory), capture_mode=literal(mode), instrumentation=literal(INSTRUMENTATION),
        plan_ref=plan_reference_cost(bundle), provenance=pdict(dict(sources_sha256=SHA, environment_sha256=SHA)),
        anchor_witness_refs=plist(pairs), trace=repeat_list(steps, row), final_state_core=core_cost(),
        evidence_scope=literal(SOURCE_EVIDENCE_SCOPE), scientific_execution_certified=BOOL))


def plan_cost(bundle):
    plan_reference_cost(bundle)  # exact membership, no file or plan construction
    steps = 220 if bundle == 71990 else 2000
    arrays = {name: TENSOR for name in _array_specs(steps)}
    return pdict(dict(schema=literal("i7_frozen_plan_v1"), profile=literal(PROFILE),
        bundle=literal(bundle), steps_total=literal(steps), stream_roles=literal(STREAM_ROLES),
        **arrays, initialization_seed=integer(0, 2**32-1)))


def core_component_bounds():
    """Max across each component's exact schedule. No root PROTO/STOP bytes."""
    ids = fixed_identities()
    return dict(anchor_pilot=max(anchor_cost(role=r, bundle=b, update=u) for r,b,u in ids if r=="pilot"),
        anchor_long=max(anchor_cost(role=r, bundle=b, update=u) for r,b,u in ids if r!="pilot"),
        source_witness=max(witness_cost(role=r, bundle=b, update=u) for r,b,u in ids),
        source_completion_pilot_on=source_completion_cost(role="pilot", bundle=71990),
        source_completion_pilot_off=source_completion_cost(role="pilot", bundle=71990, mode="capture_off"),
        source_completion_long_on=max(source_completion_cost(role=r, bundle=b) for r,b,u in ids if r!="pilot"),
        plan_pilot=plan_cost(71990), plan_long=max(plan_cost(b) for b in (71001,71002,71003,71901)))


if __name__ == "__main__":
    print("I7 core primitive arithmetic only; no runtime validation or execution admission.")
