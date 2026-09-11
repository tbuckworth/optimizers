#!/usr/bin/env python3
"""Fixed source-loop integration for iteration 007; import/default are inert.

Explicit calls consume an already validated plan and supplied pinned IDX bytes.
They do not load inputs, generate plans, grant GO authority, execute branches,
or perform a numerical audit.  Source states are reduced incrementally through
``source_streaming`` and final records are sealed create-only in the supplied
``ArtifactStore``.
"""
from __future__ import annotations


class NativeSourceError(RuntimeError):
    """The fixed source loop failed or violated its runtime invariants."""


SOURCE_RESULT_KEYS = (
    "completion", "completion_receipt", "invariant_pass", "adverse_reason",
)
PAIR_RESULT_KEYS = (
    "capture_on", "capture_on_receipt", "capture_off",
    "capture_off_receipt", "comparison", "comparison_receipt",
    "invariant_pass", "adverse_reason",
)
DIRECT_INITIAL_FLAGS = (
    "model_direct_typed_equal", "optimizer_direct_typed_equal",
    "observer_direct_typed_equal", "rng_direct_typed_equal",
    "state_direct_typed_equal",
)
DIRECT_STEP_FLAGS = (
    "model_direct_typed_equal", "optimizer_direct_typed_equal",
    "moments_direct_typed_equal", "observer_direct_typed_equal",
    "rng_direct_typed_equal", "core_direct_typed_equal",
)


def _need(condition, message):
    if not condition:
        raise NativeSourceError(message)


def _check(guard, stage):
    _need(callable(guard), "cooperative guard(stage) is required")
    _need(type(stage) is str and stage.isascii(), "guard stage must be ASCII")
    guard(stage)


def _context(*, store, identity, profile, native_device, plan, plan_reference,
             plan_artifact, images_bytes, labels_bytes, expected_files,
             sources, environment, created_utc, guard):
    import anchor_envelope as envelopes
    import artifact_store as storage
    import branch_execution
    import identity_codec as codec
    import plan_bindings as plans
    import runtime_guard as runtime
    import source_capture
    import source_history
    import torch
    import verified_plan_load as verified

    _check(guard, "native_source.entry")
    _need(type(store) is storage.ArtifactStore and not store._closed and
          not store._terminal and store.profile == profile,
          "exact writable profile-matched ArtifactStore required")
    _need(type(profile) is str and profile in
          (storage.SCIENTIFIC, storage.MLP_FIXTURE),
          "unsupported source profile")
    codec.validate_identity(identity, profile=profile)
    codec.validate_created_utc(created_utc)
    _check(guard, "native_source.context.pre_plan_validation")
    plans.validate_plan(plan, identity=identity, profile=profile)
    track = source_history.trajectory(identity, profile)
    source_history._validate_plan_ref(plan_reference, profile, track)
    _check(guard, "native_source.context.pre_verified_plan_load")
    loaded = verified.load_verified_plan_from_store(
        store, plan_reference["name"], identity=identity, profile=profile,
        expected_sha256=plan_reference["sha256"])
    _check(guard, "native_source.context.post_verified_plan_load")
    _check(guard, "native_source.context.pre_plan_comparison")
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
    _check(guard, "native_source.context.post_plan_comparison")
    if profile == storage.MLP_FIXTURE:
        expected_device = "cpu"
    else:
        _need(environment["cuda"]["current_device"] == runtime.CUDA_DEVICE_INDEX,
              "environment CUDA index differs from fixed runtime device")
        expected_device = "cuda:" + str(runtime.CUDA_DEVICE_INDEX)
    _need(type(native_device) is str and native_device == expected_device and
          str(torch.device(native_device)) == native_device,
          "native device must be canonical text")
    factories = branch_execution._fixed_factories(profile, native_device)
    _check(guard, "native_source.context.pre_provenance")
    source_capture._verify_live_context(
        sources, environment, profile, include_sources=True)
    _check(guard, "native_source.context.post_provenance")
    return {
        "store": store, "identity": identity, "profile": profile,
        "native_device": native_device, "plan": loaded["plan"],
        "plan_reference": loaded["artifact"],
        "plan_artifact": expected_artifact,
        "images_bytes": images_bytes, "labels_bytes": labels_bytes,
        "expected_files": expected_files, "sources": sources,
        "environment": environment, "created_utc": created_utc,
        "guard": guard, "track": track, "factories": factories,
        "sources_sha256": codec.tree_digest(sources),
        "environment_sha256": codec.tree_digest(environment),
    }


def _initialize(context, arm):
    import random
    import torch
    import source_history
    import state_core as state

    guard, plan = context["guard"], context["plan"]
    _check(guard, "native_source.initialize." + arm + ".pre_seed")
    seed = plan["initialization_seed"]
    random.seed(seed)
    torch.manual_seed(seed)
    if context["profile"] == "scientific_mnist_current32_v1":
        torch.cuda.manual_seed_all(seed)
    _check(guard, "native_source.initialize." + arm + ".post_seed")
    model_factory, optimizer_factory, observer_factory = context["factories"]
    model = model_factory()
    optimizer = optimizer_factory(model)
    observer = observer_factory(model, optimizer)
    optimizer.zero_grad(set_to_none=True)
    _check(guard, "native_source.initialize." + arm + ".pre_initial_snapshot")
    initial = source_history.initial_fingerprint(
        model, optimizer, observer, profile=context["profile"])
    _check(guard, "native_source.initialize." + arm + ".post_initial_snapshot")
    rng = state._raw_rng_state()
    _need(source_history._same_exact(rng, {
        key: initial["rng"][key] for key in state.RNG_KEYS[:-1]
    }), "initial snapshot RNG differs from live arm RNG")
    _check(guard, "native_source.initialize." + arm + ".complete")
    return {"model": model, "optimizer": optimizer, "observer": observer,
            "initial": initial, "rng": rng}


def _identity(context, update):
    import source_history
    return source_history._identity_for(context["track"], update)


def _materialize(context):
    import data_probe_bindings as data
    _check(context["guard"], "native_source.materialize.pre")
    value = data.materialize(
        context["images_bytes"], context["labels_bytes"],
        plan=context["plan"], identity=context["identity"],
        profile=context["profile"], expected_files=context["expected_files"])
    _check(context["guard"], "native_source.materialize.post")
    return value


def _normal_step(context, arm, materialized, update):
    import torch
    import torch.nn.functional as functional
    import source_history
    import state_core as state

    guard = context["guard"]
    model, optimizer, observer = arm["model"], arm["optimizer"], arm["observer"]
    prefix = "native_source.step." + str(update) + "." + arm["name"]
    _check(guard, prefix + ".pre_rng_restore")
    state._set_rng_state(arm["rng"])
    before_rng = state._raw_rng_state()
    _check(guard, prefix + ".post_rng_restore")
    _check(guard, prefix + ".pre_batch")
    indices = context["plan"]["training_batches"][update - 1]
    inputs = materialized["datasets"]["train_inputs"].index_select(0, indices)
    labels = materialized["datasets"]["train_noisy_labels"].index_select(0, indices)
    device = torch.device(context["native_device"])
    inputs = inputs.detach().clone().to(device=device, dtype=torch.float32)
    labels = labels.detach().clone().to(device=device, dtype=torch.int64)
    optimizer.zero_grad(set_to_none=True)
    _check(guard, prefix + ".pre_forward")
    loss = functional.cross_entropy(model(inputs), labels, reduction="mean")
    _need(type(loss) is torch.Tensor and loss.shape == torch.Size([]) and
          loss.dtype == torch.float32 and loss.device == device and
          bool(torch.isfinite(loss)), "actual source loss is not finite native float32")
    _check(guard, prefix + ".pre_backward")
    loss.backward()
    _check(guard, prefix + ".pre_observer")
    observer.filter_grad()
    _check(guard, prefix + ".pre_optimizer")
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    _check(guard, prefix + ".pre_core")
    core = state.capture_core(
        model, optimizer, observer, profile=context["profile"],
        completed_updates=update)
    after_rng = state._raw_rng_state()
    _check(guard, prefix + ".post_core")
    _need(source_history._same_exact(before_rng, after_rng),
          "source update consumed an arm RNG stream")
    arm["rng"] = before_rng
    _check(guard, prefix + ".complete")
    return core


def _captured_step(context, arm, update):
    import source_capture
    import source_history
    import source_streaming
    import state_core as state

    guard = context["guard"]
    _check(guard, "native_source.step." + str(update) + ".capture_on.pre_rng_restore")
    state._set_rng_state(arm["rng"])
    before_rng = state._raw_rng_state()
    _check(guard, "native_source.step." + str(update) + ".capture_on.post_rng_restore")
    ident = _identity(context, update)
    _check(guard, "native_source.step." + str(update) + ".capture_on.pre")
    transaction = source_capture.capture_anchor_then_live_witness(
        arm["model"], arm["optimizer"], arm["observer"],
        store=context["store"], identity=ident, profile=context["profile"],
        created_utc=context["created_utc"], plan=context["plan"],
        plan_artifact=context["plan_artifact"],
        images_bytes=context["images_bytes"], labels_bytes=context["labels_bytes"],
        expected_files=context["expected_files"], sources=context["sources"],
        environment=context["environment"], guard=guard)
    _check(guard, "native_source.step." + str(update) + ".capture_on.post")
    core = state.capture_core(
        arm["model"], arm["optimizer"], arm["observer"],
        profile=context["profile"], completed_updates=update)
    after_rng = state._raw_rng_state()
    _check(guard, "native_source.step." + str(update) + ".capture_on.post_core")
    _need(source_history._same_exact(before_rng, after_rng),
          "captured source update consumed an arm RNG stream")
    arm["rng"] = before_rng
    anchor_ref = source_streaming.artifact_reference(
        transaction["anchor"], transaction["anchor_receipt"], identity=ident,
        profile=context["profile"], kind="anchor")
    witness_ref = source_streaming.artifact_reference(
        transaction["witness"], transaction["witness_receipt"], identity=ident,
        profile=context["profile"], kind="source-witness")
    # Do not retain full anchor/witness envelopes in source-loop state.
    del transaction
    return core, anchor_ref, witness_ref


def _completion_write(context, completion, label):
    store, guard = context["store"], context["guard"]
    _check(guard, "native_source." + label + ".prewrite")
    receipt = store.write_tensor_tree(completion["artifact_name"], completion)
    _check(guard, "native_source." + label + ".postwrite")
    _need(not store._terminal, "store became terminal after source completion")
    return receipt


def _restore_caller_rng(caller_rng, context):
    import source_history
    import state_core as state
    _check(context["guard"], "native_source.caller_rng.pre_restore")
    state._set_rng_state(caller_rng)
    _need(source_history._same_exact(state._raw_rng_state(), caller_rng),
          "source runner failed to restore caller RNG")
    _check(context["guard"], "native_source.caller_rng_restored")


def run_source(*, store, identity, profile, native_device, capture_mode, plan,
               plan_reference, plan_artifact, images_bytes, labels_bytes,
               expected_files, sources, environment, created_utc, guard):
    """Run and seal one fixed source trajectory, without branch execution."""
    import artifact_store as storage
    import source_capture
    import source_streaming
    import state_core as state

    valid_store = type(store) is storage.ArtifactStore
    caller_rng = None
    context = None
    try:
        _check(guard, "native_source.pre_caller_rng_capture")
        caller_rng = state._raw_rng_state()
        _check(guard, "native_source.post_caller_rng_capture")
        context = _context(
            store=store, identity=identity, profile=profile,
            native_device=native_device, plan=plan,
            plan_reference=plan_reference, plan_artifact=plan_artifact,
            images_bytes=images_bytes, labels_bytes=labels_bytes,
            expected_files=expected_files, sources=sources,
            environment=environment, created_utc=created_utc, guard=guard)
        _need(capture_mode in ("capture_on", "capture_off"),
              "unknown capture mode")
        arm = _initialize(context, capture_mode)
        arm["name"] = capture_mode
        del arm["initial"]
        materialized = _materialize(context)
        _check(guard, "native_source.pre_stream_initialize")
        stream = source_streaming.SourceCompletionStream(
            identity=identity, profile=profile, capture_mode=capture_mode,
            plan_ref=plan_reference, sources_sha256=context["sources_sha256"],
            environment_sha256=context["environment_sha256"])
        _check(guard, "native_source.post_stream_initialize")
        for update in range(1, context["track"]["steps_total"] + 1):
            _check(guard, "native_source.update." + str(update) + ".begin")
            if capture_mode == "capture_on" and update in context["track"]["anchor_updates"]:
                core, anchor_ref, witness_ref = _captured_step(context, arm, update)
                _check(guard, "native_source.update." + str(update) + ".pre_record")
                stream.record(core, anchor_ref=anchor_ref, witness_ref=witness_ref)
            else:
                core = _normal_step(context, arm, materialized, update)
                _check(guard, "native_source.update." + str(update) + ".pre_record")
                stream.record(core)
            _check(guard, "native_source.update." + str(update) + ".post_record")
            _check(guard, "native_source.update." + str(update) + ".complete")
        _check(guard, "native_source.completion.pre_finish")
        completion = stream.finish()
        _check(guard, "native_source.completion.post_finish")
        _check(guard, "native_source.completion.pre_provenance")
        source_capture._verify_live_context(
            sources, environment, profile, include_sources=True)
        _check(guard, "native_source.completion.post_provenance")
        receipt = _completion_write(context, completion, "completion")
        _restore_caller_rng(caller_rng, context)
        return {
            "completion": completion, "completion_receipt": receipt,
            "invariant_pass": True, "adverse_reason": None,
        }
    except BaseException as exc:
        if caller_rng is not None:
            try:
                state._set_rng_state(caller_rng)
            except BaseException:
                pass
        if valid_store and not store._closed and not store._terminal:
            store._fail("native_source_failed", type(exc).__name__)
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        if isinstance(exc, NativeSourceError):
            raise
        raise NativeSourceError("native source failed: " + type(exc).__name__) from exc


def _pilot_invariant(comparison):
    initial = comparison["initial_comparison"]
    return (all(initial[key] is True for key in DIRECT_INITIAL_FLAGS) and
            all(row[key] is True for row in comparison["step_trace"]
                for key in DIRECT_STEP_FLAGS))


def run_pilot_pair(*, store, identity, profile, native_device, plan,
                   plan_reference, plan_artifact, images_bytes, labels_bytes,
                   expected_files, sources, environment, created_utc, guard):
    """Run one paired capture pilot and seal adverse-inclusive evidence.

    A false ``invariant_pass`` is a valid adverse result.  The caller must stop
    before pilot branches or a later GO while retaining the three sealed records.
    """
    import artifact_store as storage
    import identity_codec as codec
    import source_capture
    import source_streaming
    import state_core as state

    valid_store = type(store) is storage.ArtifactStore
    caller_rng = None
    context = None
    try:
        _check(guard, "native_source.pair.pre_caller_rng_capture")
        caller_rng = state._raw_rng_state()
        _check(guard, "native_source.pair.post_caller_rng_capture")
        context = _context(
            store=store, identity=identity, profile=profile,
            native_device=native_device, plan=plan,
            plan_reference=plan_reference, plan_artifact=plan_artifact,
            images_bytes=images_bytes, labels_bytes=labels_bytes,
            expected_files=expected_files, sources=sources,
            environment=environment, created_utc=created_utc, guard=guard)
        _need(context["track"]["execution_role"] == "pilot" or
              profile == storage.MLP_FIXTURE, "paired source is pilot-only")
        on = _initialize(context, "capture_on")
        off = _initialize(context, "capture_off")
        on["name"], off["name"] = "capture_on", "capture_off"
        _check(guard, "native_source.pair.pre_stream_initialize")
        stream = source_streaming.PilotPairStream(
            on_initial=on["initial"], off_initial=off["initial"],
            identity=identity, profile=profile, plan_ref=plan_reference,
            sources_sha256=context["sources_sha256"],
            environment_sha256=context["environment_sha256"])
        _check(guard, "native_source.pair.post_stream_initialize")
        del on["initial"], off["initial"]
        materialized = _materialize(context)
        for update in range(1, context["track"]["steps_total"] + 1):
            _check(guard, "native_source.pair." + str(update) + ".begin")
            if update in context["track"]["anchor_updates"]:
                on_core, anchor_ref, witness_ref = _captured_step(context, on, update)
            else:
                on_core = _normal_step(context, on, materialized, update)
                anchor_ref = witness_ref = None
            off_core = _normal_step(context, off, materialized, update)
            _check(guard, "native_source.pair." + str(update) + ".pre_record")
            stream.record(on_core, off_core, anchor_ref=anchor_ref,
                          witness_ref=witness_ref)
            _check(guard, "native_source.pair." + str(update) + ".post_record")
            _check(guard, "native_source.pair." + str(update) + ".complete")
        _check(guard, "native_source.pair.pre_finish")
        bundle = stream.finish()
        _check(guard, "native_source.pair.post_finish")
        _check(guard, "native_source.pair.pre_provenance")
        source_capture._verify_live_context(
            sources, environment, profile, include_sources=True)
        _check(guard, "native_source.pair.post_provenance")
        on_receipt = _completion_write(context, bundle["capture_on"], "capture_on")
        off_receipt = _completion_write(context, bundle["capture_off"], "capture_off")
        _check(guard, "native_source.comparison.preencode")
        encoded = codec.json_bytes(bundle["comparison"])
        _check(guard, "native_source.comparison.prewrite")
        comparison_receipt = store.write_bytes(
            bundle["comparison"]["artifact_name"], encoded)
        _check(guard, "native_source.comparison.postwrite")
        _need(not store._terminal, "store became terminal after comparison")
        invariant = _pilot_invariant(bundle["comparison"])
        _restore_caller_rng(caller_rng, context)
        return {
            "capture_on": bundle["capture_on"],
            "capture_on_receipt": on_receipt,
            "capture_off": bundle["capture_off"],
            "capture_off_receipt": off_receipt,
            "comparison": bundle["comparison"],
            "comparison_receipt": comparison_receipt,
            "invariant_pass": invariant,
            "adverse_reason": None if invariant else "capture_path_inequality",
        }
    except BaseException as exc:
        if caller_rng is not None:
            try:
                state._set_rng_state(caller_rng)
            except BaseException:
                pass
        if valid_store and not store._closed and not store._terminal:
            store._fail("native_source_pair_failed", type(exc).__name__)
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        if isinstance(exc, NativeSourceError):
            raise
        raise NativeSourceError("native source pair failed: " + type(exc).__name__) from exc


if __name__ == "__main__":
    print("native_source: inert; no source execution, branch execution or scientific action")
