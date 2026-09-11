#!/usr/bin/env python3
"""Actual six-branch producer for the strict I7 anchor envelope.

This module restores only fixed profile objects, executes native AdamW steps in
both orders, assembles measurements, and seals one branch-results artifact.  It
is not a phase controller or an independent numerical auditor.
"""
from __future__ import annotations

import hashlib
import importlib.util
import io
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

import anchor_envelope as envelope
import artifact_envelopes as artifacts
import artifact_store as storage
import data_probe_bindings as data
import identity_codec as codec
import measurement_assembly as assembly
import response_math as response
import source_capture as source
import state_core as state
import verified_plan_load as verified


class BranchExecutionError(RuntimeError):
    pass


BRANCHES = response.BRANCHES
RETURN_KEYS = ("artifact", "receipt", "candidates", "measurements")
BRANCH_KEYS = (
    "status", "reason", "delivered_gradient", "assigned_gradient_null_mask",
    "parameters_after", "parameters_after_flat", "optimizer_after", "measurement",
)
_HERE = Path(__file__).resolve().parent
_FILTER_SPEC = importlib.util.spec_from_file_location(
    "i7_branch_spectral_filter", _HERE.parents[3] / "spectral_filter.py")
if _FILTER_SPEC is None or _FILTER_SPEC.loader is None:  # pragma: no cover
    raise ImportError("cannot load bound spectral_filter.py")
_FILTER = importlib.util.module_from_spec(_FILTER_SPEC)
_FILTER_SPEC.loader.exec_module(_FILTER)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise BranchExecutionError(message)


def _same(left: Any, right: Any, message: str) -> None:
    _require(envelope.same_exact(left, right), message)


def _guard(guard, stage: str) -> None:
    if guard is not None:
        guard(stage)


def _fixed_factories(profile: str, native_device: str):
    spec = state.PROFILE[profile]
    device = torch.device(native_device)
    _require(device.type == spec["device_kind"], "anchor native device/profile mismatch")

    def model_factory():
        if profile == storage.SCIENTIFIC:
            model = torch.nn.Sequential(torch.nn.Linear(784, 64), torch.nn.ReLU(),
                                        torch.nn.Linear(64, 10))
        elif profile == storage.MLP_FIXTURE:
            model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.ReLU(),
                                        torch.nn.Linear(4, 2))
        else:  # The branch envelope deliberately excludes the one-layer fixture.
            raise BranchExecutionError("unsupported branch execution profile")
        return model.to(device=device, dtype=torch.float32)

    def optimizer_factory(model):
        return torch.optim.AdamW(model.parameters(), lr=0.001, betas=(0.9, 0.999),
                                 eps=1e-8, weight_decay=0.01, foreach=False, fused=False)

    def observer_factory(model, optimizer):
        return _FILTER.SpectralGradientFilter(
            model, optimizer, rank=spec["rank"], decay=0.99, warmup=spec["warmup"],
            filter_strength=1.0, energy_threshold=None, adaptive="none", normalize="none",
            weighting="hard", alpha=1.0, soft_residual=True, stable_update=True,
            relative_eig_tol=1e-8, absolute_eig_floor=0.0, stabilize_every=100)

    return model_factory, optimizer_factory, observer_factory


def _receipt_bytes(receipt: dict[str, Any]) -> bytes:
    keys = ("schema", "name", "size", "sha256", "status", "encoding")
    return storage._json_bytes({key: receipt[key] for key in keys})


def _load_pinned(store: storage.ArtifactStore, supplied: Any, receipt: Any,
                 *, schema_name: str) -> dict[str, Any]:
    _require(type(receipt) is dict and tuple(receipt) == source.RECEIPT_KEYS,
             schema_name + " receipt keys/order differ")
    _require(all(type(receipt[key]) is str for key in
                 ("schema", "name", "sha256", "status", "encoding", "receipt_name"))
             and type(receipt["size"]) is int and receipt["size"] > 0
             and receipt["schema"] == "i7_artifact_receipt_v1"
             and receipt["status"] == "complete" and receipt["encoding"] == "torch_weights_only",
             schema_name + " receipt values invalid")
    _require(type(supplied) is dict and supplied.get("schema_name") == schema_name
             and type(supplied.get("artifact_id")) is str
             and receipt["name"] == supplied["artifact_id"] + ".pt",
             schema_name + " artifact/receipt binding differs")
    expected_receipt_name = "receipt-" + hashlib.sha256(receipt["name"].encode("ascii")).hexdigest() + ".json"
    _require(receipt["receipt_name"] == expected_receipt_name,
             schema_name + " receipt filename differs")
    report, scanned = storage._inspect_dirfd(store._dirfd)
    _require(not report["terminal"] and report["header"]["profile"] == store.profile,
             "input store is terminal or has wrong profile")
    base = {key: receipt[key] for key in source.RECEIPT_KEYS[:-1]}
    _require(sum(row == base for row in report["receipts"]) == 1,
             schema_name + " has no unique indexed receipt")
    expected_receipt_bytes = _receipt_bytes(receipt)
    actual_receipt, _ = storage._read_regular(
        store._dirfd, receipt["receipt_name"], maximum=storage._RECEIPT_MAX,
        expected=scanned[receipt["receipt_name"]])
    _require(actual_receipt == expected_receipt_bytes, schema_name + " receipt bytes differ")
    raw, _ = storage._read_regular(store._dirfd, receipt["name"], maximum=receipt["size"],
                                   expected=scanned[receipt["name"]])
    _require(len(raw) == receipt["size"] and hashlib.sha256(raw).hexdigest() == receipt["sha256"],
             schema_name + " pinned bytes differ")
    try:
        verified._check_safe_globals()
        loaded = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
        storage._validate_tree(loaded)
    except Exception as exc:
        raise BranchExecutionError(schema_name + " restricted decode failed: " + type(exc).__name__) from None
    _same(loaded, supplied, schema_name + " supplied value differs from pinned file")
    return loaded


def _all_tensors(value: Any) -> list[torch.Tensor]:
    found: list[torch.Tensor] = []
    if type(value) is torch.Tensor:
        found.append(value)
    elif type(value) is dict:
        for child in value.values():
            found.extend(_all_tensors(child))
    elif type(value) in (list, tuple):
        for child in value:
            found.extend(_all_tensors(child))
    return found


def _live_state_tensors(live: dict[str, Any], *, include_gradients: bool = False) -> list[torch.Tensor]:
    model, optimizer, observer = live["model"], live["optimizer"], live["observer"]
    result = list(model.parameters())
    if include_gradients:
        result.extend(parameter.grad for parameter in model.parameters() if parameter.grad is not None)
    for parameter in model.parameters():
        row = optimizer.state[parameter]
        result.extend((row["step"], row["exp_avg"], row["exp_avg_sq"]))
    result.extend(value for value in (observer.V, observer.S, observer.grad_mean)
                  if type(value) is torch.Tensor)
    return result


def _span(tensor: torch.Tensor, where: str, *, compact: bool) -> tuple[str, int, int]:
    _require(type(tensor) in (torch.Tensor, torch.nn.Parameter) and tensor.layout == torch.strided
             and tensor.is_contiguous(), where + ": tensor is not contiguous live storage")
    size = tensor.numel() * tensor.element_size()
    if compact:
        # A live operator may return a full-storage view (notably V/S after
        # truncation).  Full range, zero offset, and exact storage size are the
        # relevant span properties; saved records are independently cloned.
        _require(tensor.storage_offset() == 0 and tensor.untyped_storage().nbytes() == size,
                 where + ": tensor does not occupy one compact storage span")
    else:
        _require(tensor.untyped_storage().nbytes() >= size,
                 where + ": live view exceeds its storage")
    device = str(tensor.device)
    return device, tensor.data_ptr(), size


def _assert_disjoint(groups) -> None:
    spans: list[tuple[str, int, int, str]] = []
    for group in groups:
        label, tensors = group[:2]
        compact = True if len(group) == 2 else group[2]
        for index, tensor in enumerate(tensors):
            device, start, size = _span(tensor, f"{label}[{index}]", compact=compact)
            if size == 0:
                continue
            end = start + size
            for other_device, other_start, other_end, other_label in spans:
                _require(device != other_device or end <= other_start or other_end <= start,
                         f"live tensor storage overlap: {label} and {other_label}")
            spans.append((device, start, end, label))


def _flat_live_gradient(model: torch.nn.Module, where: str) -> torch.Tensor:
    values = []
    for name, parameter in model.named_parameters():
        gradient = parameter.grad
        _require(type(gradient) is torch.Tensor and gradient.shape == parameter.shape
                 and gradient.dtype == torch.float32 and gradient.device == parameter.device
                 and gradient.layout == torch.strided and not gradient.is_sparse
                 and bool(torch.isfinite(gradient).all()), where + ": invalid " + name)
        values.append(gradient.detach().reshape(-1))
    result = torch.cat(values).contiguous().clone()
    _require(result.numel() > 0 and result._base is None, where + ": non-owning flat gradient")
    return result


def _parameter_map(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: parameter.detach().contiguous().clone() for name, parameter in model.named_parameters()}


def _basis_record(value: torch.Tensor | None) -> dict[str, Any] | None:
    return None if value is None else state._native_tensor(value)


def _candidate_projection(candidate_state: dict[str, Any]) -> dict[str, Any]:
    return {key: state.clone_tree(candidate_state[key]) for key in
            ("g", "c", "l", "previous_basis", "post_ingest_basis", "nc", "nl",
             "observer_after", "leverage", "zero_cases")}


def _zero_cases(candidates: dict[str, Any]) -> dict[str, Any]:
    branches = candidates["branches"]
    return {
        "current_zero": candidates["leverage"]["current_norm"] == 0.0,
        "lagged_zero": candidates["leverage"]["lagged_norm"] == 0.0,
        "branch_defined_mask": {key: branches[key]["status"] == "defined" for key in BRANCHES},
        "branch_reasons": {key: branches[key]["reason"] for key in BRANCHES},
    }


def _assigned_gradient(model: torch.nn.Module, flat: torch.Tensor) -> tuple[dict[str, Any], list[bool], tuple[torch.Tensor, ...]]:
    offset = 0
    retained = []
    for parameter in model.parameters():
        count = parameter.numel()
        assigned = flat[offset:offset + count].reshape(parameter.shape).contiguous().clone()
        _require(assigned.dtype == parameter.dtype and assigned.device == parameter.device,
                 "assigned gradient native metadata differs")
        _require(assigned.storage_offset() == 0 and assigned._base is None
                 and assigned.untyped_storage().nbytes() == assigned.numel() * assigned.element_size(),
                 "assigned gradient is not an owning compact tensor")
        parameter.grad = assigned
        retained.append(assigned)
        offset += count
    _require(offset == flat.numel(), "assigned gradient component count differs")
    mask = [parameter.grad is None for parameter in model.parameters()]
    _require(mask == [False] * len(mask), "defined branch did not assign every gradient")
    actual = _flat_live_gradient(model, "assigned branch gradient")
    _same(state._native_tensor(actual), state._native_tensor(flat), "assigned gradient differs")
    return state._native_tensor(actual), mask, tuple(retained)


def _undefined_row(candidate: dict[str, Any]) -> dict[str, Any]:
    return {"status": candidate["status"], "reason": candidate["reason"],
            "delivered_gradient": None, "assigned_gradient_null_mask": None,
            "parameters_after": None, "parameters_after_flat": None,
            "optimizer_after": None, "measurement": None}


def _execute_one(live: dict[str, Any] | None, candidate: dict[str, Any], *, core: dict[str, Any],
                 profile: str, completed_after: int, disjoint_groups, assigned_groups):
    if candidate["status"] == "undefined":
        _require(live is None and candidate["gradient"] is None, "undefined branch was instantiated")
        row = _undefined_row(candidate)
        return {"serialized": row, "endpoint_native": None, "proof": state.clone_tree(row),
                "checked_live_tensor_count": 0}
    _require(candidate["status"] == "defined" and live is not None
             and type(candidate["gradient"]) is torch.Tensor, "defined branch missing live clone")
    start = state.capture_core(live["model"], live["optimizer"], live["observer"],
                               profile=profile, completed_updates=completed_after - 1)
    _same(start, core, "branch start differs from anchor")
    assigned, assigned_mask, retained = _assigned_gradient(live["model"], candidate["gradient"])
    assigned_groups.append(retained)
    _assert_disjoint(disjoint_groups + [("assigned", list(group)) for group in assigned_groups])
    live["optimizer"].step()
    live["optimizer"].zero_grad(set_to_none=True)
    post_mask = [parameter.grad is None for parameter in live["model"].parameters()]
    _require(post_mask == [True] * len(post_mask), "post-step gradients are not all null")
    model_after = state._capture_model(live["model"], profile)
    expected_model_after = state.clone_tree(core["model"])
    expected_model_after["parameters"] = state.clone_tree(model_after["parameters"])
    _same(model_after, expected_model_after, "branch model metadata/modes changed")
    optimizer_after = state._capture_optimizer(live["optimizer"], live["model"], profile,
                                               completed_after)
    observer_after = state._capture_observer(live["observer"], live["model"], live["optimizer"],
                                             profile, completed_after - 1)
    _same(observer_after, core["observer"], "branch-private observer changed")
    rng_after = state._capture_rng()
    _same(rng_after, core["rng"], "branch changed RNG")
    parameters = model_after["parameters"]
    flat = torch.cat([entry["value"].reshape(-1) for entry in parameters]).contiguous().clone()
    delivered = assigned
    row = {"status": "defined", "reason": None, "delivered_gradient": delivered,
           "assigned_gradient_null_mask": assigned_mask, "parameters_after": parameters,
           "parameters_after_flat": flat, "optimizer_after": optimizer_after,
           "measurement": None}
    proof = {**state.clone_tree(row), "observer_unchanged": observer_after,
             "post_step_null_mask": post_mask, "rng": rng_after}
    return {"serialized": row, "endpoint_native": _parameter_map(live["model"]),
            "proof": proof, "checked_live_tensor_count": len(_live_state_tensors(live))}


def _validate_helper_proof(metadata: dict[str, Any], *, candidate_projection: dict[str, Any],
                           canonical: dict[str, Any], reverse: dict[str, Any], core: dict[str, Any],
                           witness: dict[str, Any], replay_loss: dict[str, Any], counts: dict[str, dict[str, int]]) -> None:
    # The envelope helper serializes attestations; these checks bind those
    # attestations to the live comparisons just completed by this producer.
    cp = metadata["candidate_proof"]
    digest = codec.tree_digest(candidate_projection)
    _require(cp["candidate_core_sha256"] == cp["candidate_core_after_canonical_sha256"]
             == cp["candidate_core_after_reverse_sha256"] == digest
             and cp["candidate_core_directly_unchanged"] is True,
             "candidate proof helper differs from actual projection")
    clone = metadata["clone_independence_proof"]
    _require(clone["anchor_core_sha256"] == clone["anchor_core_after_sha256"] == codec.tree_digest(core)
             and clone["anchor_directly_unchanged"] is True,
             "anchor proof helper differs from actual anchor")
    for order in ("canonical", "reverse"):
        for branch in BRANCHES:
            row = clone["per_order"][order][branch]
            expected = counts[order][branch]
            _require(row["checked_live_tensor_count"] == expected,
                     "clone proof live tensor count differs")
    proof = metadata["branch_execution_proof"]
    _require(proof["canonical_order"] == list(BRANCHES)
             and proof["verification_order"] == list(reversed(BRANCHES))
             and proof["all_defined_order_invariant"] is True,
             "branch-order proof helper differs")
    _same(canonical, reverse, "canonical/reverse actual results differ")
    current = metadata["current_replay_proof"]
    _require(current["overall_status"] == "exact", "current replay helper not exact")
    _same(replay_loss, witness["payload"]["live_loss"], "actual replay loss differs from witness")


def _execute_impl(*, store, anchor, anchor_receipt, witness, witness_receipt,
                  created_utc, caller_rng, guard, **context):
    profile, identity = context["profile"], context["identity"]
    _guard(guard, "branch_execution.anchor.preread")
    loaded_anchor = _load_pinned(store, anchor, anchor_receipt, schema_name="i7_anchor")
    _guard(guard, "branch_execution.anchor.postread")
    _guard(guard, "branch_execution.witness.preread")
    loaded_witness = _load_pinned(store, witness, witness_receipt,
                                  schema_name="i7_source_step_witness")
    _guard(guard, "branch_execution.witness.postread")
    envelope.validate_anchor(loaded_anchor, **context)
    source.validate_source_witness(loaded_witness, anchor=loaded_anchor,
                                   anchor_receipt=anchor_receipt, **context)
    codec.tree_digest({"anchor": loaded_anchor, "witness": loaded_witness})
    core = envelope.core_from_anchor(loaded_anchor)
    state.validate_core(core)
    native_device = core["model"]["parameters"][0]["native_device"]
    factories = _fixed_factories(profile, native_device)
    materialized = data.materialize(context["images_bytes"], context["labels_bytes"],
                                    plan=context["plan"], identity=identity, profile=profile,
                                    expected_files=context["expected_files"])
    batch = materialized["probes"]["batch_noisy"]

    _guard(guard, "branch_execution.candidate.before_creation")
    candidate_live = state.restore_core(core, *factories)
    cm, co, cf = candidate_live["model"], candidate_live["optimizer"], candidate_live["observer"]
    before_native = _parameter_map(cm)
    previous_basis = _basis_record(cf.V)
    previous_live = None if cf.V is None else cf.V.detach().contiguous().clone()
    device = next(cm.parameters()).device
    inputs = batch["inputs"].detach().to(device=device, dtype=torch.float32).contiguous().clone()
    labels = batch["labels"].detach().to(device=device, dtype=torch.int64).contiguous().clone()
    loss = F.cross_entropy(cm(inputs), labels, reduction="mean")
    _require(type(loss) is torch.Tensor and loss.shape == torch.Size([])
             and loss.dtype == torch.float32 and loss.device == device and bool(torch.isfinite(loss)),
             "candidate native loss invalid")
    replay_loss = {"probe_key": "batch_noisy", "count": int(inputs.shape[0]),
                   "value_native_float32": float(loss.detach().cpu().item()),
                   "sample_identity_sha256": source._sample_hash(batch["inputs"], batch["labels"])}
    _same(replay_loss, loaded_witness["payload"]["live_loss"], "candidate loss differs from witness")
    loss.backward()
    raw_native = _flat_live_gradient(cm, "candidate raw gradient")
    lagged_native = (raw_native.detach().clone() if previous_live is None else
                     (previous_live @ (previous_live.T @ raw_native)).contiguous().clone())
    cf.filter_grad()
    current_native = _flat_live_gradient(cm, "candidate current gradient")
    observer_after = state._capture_observer(cf, cm, co, profile, identity["anchor_update"])
    post_basis = _basis_record(cf.V)
    candidate_rng = state._capture_rng()
    _same(candidate_rng, core["rng"], "candidate computation changed RNG")
    raw_record, current_record = state._native_tensor(raw_native), state._native_tensor(current_native)
    _same(raw_record, loaded_witness["payload"]["raw_gradient"], "candidate raw gradient differs")
    _same(current_record, loaded_witness["payload"]["delivered_current_gradient"],
          "candidate current gradient differs")
    _same(observer_after, loaded_witness["payload"]["observer_after"],
          "candidate observer differs from source witness")
    _same(previous_basis, core["observer"]["state"]["V"],
          "candidate previous basis differs from anchor observer")
    _same(post_basis, observer_after["state"]["V"],
          "candidate post-ingest basis differs from shared observer")
    constructed_native = response.construct(raw_native, current_native, lagged_native)
    expected_candidates = state.clone_tree(constructed_native)
    candidate_state = {
        "g": raw_record, "c": current_record, "l": state._native_tensor(lagged_native),
        "previous_basis": previous_basis, "post_ingest_basis": post_basis,
        "nc": constructed_native["leverage"]["current_norm"],
        "nl": constructed_native["leverage"]["lagged_norm"],
        "observer_after": observer_after, "measurement_before": None,
        "leverage": state.clone_tree(expected_candidates["leverage"]),
        "zero_cases": _zero_cases(expected_candidates),
    }
    frozen_candidate = _candidate_projection(candidate_state)
    frozen_anchor = state.clone_tree(core)
    _guard(guard, "branch_execution.candidate.after_creation")

    order_names = {"canonical": tuple(BRANCHES), "reverse": tuple(reversed(BRANCHES))}
    live_by_order: dict[str, dict[str, Any]] = {}
    groups = [("anchor", _all_tensors(loaded_anchor)), ("witness", _all_tensors(loaded_witness)),
              ("candidate", _live_state_tensors(candidate_live)),
              ("candidate-gradients",
               [parameter.grad for parameter in cm.parameters() if parameter.grad is not None], False),
              ("candidate-native-values", _all_tensors(
                  (raw_native, current_native, lagged_native, previous_live, constructed_native))),
              ("candidate-values", _all_tensors((candidate_state, expected_candidates)))]
    for order_name, order in order_names.items():
        live_by_order[order_name] = {}
        for branch in order:
            if constructed_native["branches"][branch]["status"] == "undefined":
                live_by_order[order_name][branch] = None
                continue
            live = state.restore_core(core, *factories)
            live_by_order[order_name][branch] = live
            groups.append((order_name + ":" + branch, _live_state_tensors(live)))
    _assert_disjoint(groups)

    outputs: dict[str, dict[str, Any]] = {}
    assigned_groups: list[tuple[torch.Tensor, ...]] = []
    for order_name, order in order_names.items():
        found = {}
        for branch in order:
            _guard(guard, f"branch_execution.branch.{order_name}.{branch}.before")
            found[branch] = _execute_one(
                live_by_order[order_name][branch], constructed_native["branches"][branch],
                core=core, profile=profile, completed_after=identity["anchor_update"],
                disjoint_groups=groups, assigned_groups=assigned_groups)
            _guard(guard, f"branch_execution.branch.{order_name}.{branch}.after")
        outputs[order_name] = {branch: found[branch] for branch in BRANCHES}
        _same(_candidate_projection(candidate_state), frozen_candidate,
              "candidate changed during " + order_name + " execution")
    canonical_proof = {key: outputs["canonical"][key]["proof"] for key in BRANCHES}
    reverse_proof = {key: outputs["reverse"][key]["proof"] for key in BRANCHES}
    _same(canonical_proof, reverse_proof, "branch endpoints differ by execution order")
    _same(core, frozen_anchor, "anchor core changed during branch execution")

    canonical_rows = {key: outputs["canonical"][key]["serialized"] for key in BRANCHES}
    current = canonical_rows["current"]
    _require(current["status"] == "defined", "current branch unexpectedly undefined")
    _same(current["delivered_gradient"], loaded_witness["payload"]["delivered_current_gradient"],
          "current delivered gradient differs from witness")
    _same(current["parameters_after"], loaded_witness["payload"]["parameters_after"],
          "current parameters differ from witness")
    _same(current["optimizer_after"], loaded_witness["payload"]["optimizer_after"],
          "current optimizer differs from witness")

    endpoints = {key: outputs["canonical"][key]["endpoint_native"] for key in BRANCHES}
    branch_artifact_id = codec.artifact_id(identity, profile=profile, kind="branch-results")
    measurements = assembly.assemble(before_native, endpoints, constructed_native,
                                     state.clone_tree(materialized["probes"]),
                                     artifact_id=branch_artifact_id, profile=profile,
                                     native_device=native_device, guard=guard)
    candidate_state["measurement_before"] = state.clone_tree(measurements["measurement_before"])
    for branch in BRANCHES:
        canonical_rows[branch]["measurement"] = state.clone_tree(measurements["branches"][branch])
    _same(state._capture_rng(), core["rng"], "measurement assembly changed RNG")
    _same(_candidate_projection(candidate_state), frozen_candidate,
          "measurement assembly changed candidate state")
    _same(core, frozen_anchor, "measurement assembly changed anchor")
    _require(measurements["measurement_before"]["probes"]["batch_noisy"]["native32"]["before_ce"]
             == replay_loss["value_native_float32"], "assembled native before loss differs from live replay")

    metadata = artifacts.expected_audit_metadata(
        candidate_state, canonical_rows, anchor=loaded_anchor, anchor_receipt=anchor_receipt,
        witness=loaded_witness, witness_receipt=witness_receipt)
    counts = {order: {branch: outputs[order][branch]["checked_live_tensor_count"]
                      for branch in BRANCHES} for order in order_names}
    _validate_helper_proof(metadata, candidate_projection=frozen_candidate,
                           canonical=canonical_proof, reverse=reverse_proof, core=core,
                           witness=loaded_witness, replay_loss=replay_loss, counts=counts)
    artifact = artifacts.make_branch_results(
        candidate_state, canonical_rows, state.clone_tree(measurements["comparisons"]), metadata,
        created_utc=created_utc, anchor=loaded_anchor, anchor_receipt=anchor_receipt,
        witness=loaded_witness, witness_receipt=witness_receipt,
        expected_candidates=expected_candidates, expected_measurements=measurements, **context)
    artifacts.validate_branch_results(
        artifact, anchor=loaded_anchor, anchor_receipt=anchor_receipt,
        witness=loaded_witness, witness_receipt=witness_receipt,
        expected_candidates=expected_candidates, expected_measurements=measurements, **context)
    codec.tree_digest({"anchor": loaded_anchor, "witness": loaded_witness, "branch": artifact})
    source._verify_live_context(context["sources"], context["environment"], profile,
                                include_sources=True)
    name = artifact["artifact_id"] + ".pt"
    _guard(guard, "branch_execution.artifact.prewrite")
    receipt = store.write_tensor_tree(name, artifact)
    _guard(guard, "branch_execution.artifact.postwrite")
    _require(not store._terminal, "store became terminal after branch-results seal")
    source._verify_live_context(context["sources"], context["environment"], profile,
                                include_sources=True)
    _same(state._capture_rng(), core["rng"],
          "branch serialization or final runtime check changed anchor RNG")
    state._set_rng_state(caller_rng)
    _same(state._raw_rng_state(), caller_rng, "branch producer failed to restore caller RNG")
    return {"artifact": artifact, "receipt": receipt,
            "candidates": state.clone_tree(expected_candidates),
            "measurements": state.clone_tree(measurements)}


def execute_branches(*, store, anchor, anchor_receipt, witness, witness_receipt,
                     created_utc, guard=None, **anchor_context):
    """Execute and seal six branches, invoking the optional cooperative guard."""
    valid_store = type(store) is storage.ArtifactStore
    caller_rng = None
    try:
        _require(guard is None or callable(guard), "guard must be callable or None")
        _guard(guard, "branch_execution.entry")
        _require(valid_store, "exact ArtifactStore required")
        _require(not store._closed and not store._terminal, "store is closed or terminal")
        _require(type(anchor_context) is dict and set(anchor_context) == set(artifacts.CONTEXT_KEYS),
                 "exact anchor context required")
        profile = anchor_context["profile"]
        _require(type(profile) is str and profile in (storage.SCIENTIFIC, storage.MLP_FIXTURE)
                 and store.profile == profile, "store/profile mismatch")
        codec.validate_identity(anchor_context["identity"], profile=profile)
        codec.validate_created_utc(created_utc)
        source._verify_live_context(anchor_context["sources"], anchor_context["environment"],
                                    profile, include_sources=True)
        caller_rng = state._raw_rng_state()
        return _execute_impl(store=store, anchor=anchor, anchor_receipt=anchor_receipt,
                             witness=witness, witness_receipt=witness_receipt,
                             created_utc=created_utc, caller_rng=caller_rng, guard=guard,
                             **anchor_context)
    except BaseException as exc:
        if caller_rng is not None:
            try:
                state._set_rng_state(caller_rng)
            except BaseException:
                pass
        if valid_store and not store._closed and not store._terminal:
            store._fail("branch_execution_failed", type(exc).__name__)
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        if isinstance(exc, BranchExecutionError):
            raise
        raise BranchExecutionError("branch execution failed: " + type(exc).__name__) from None
