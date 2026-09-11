"""Bounded current-layout storage specimen; never scientific evidence or a GO gate.

The explicit measurement path first runs the registered tiny MLP producers and
semantic validators.  It then makes separate, unregistered storage-only trees
by replacing only exhaustively classified tensor shapes.  Lifted values and
all-failure diagnostics are synthetic and are deliberately never passed to a
scientific validator.  Import and the default CLI perform no fixture work.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import struct
import time
import zipfile

import torch

import artifact_store as storage


SPECIMEN_PROFILE = "i7_current_layout_storage_only_v1"
P, RANK = 50_890, 32
PARAMETERS = ((0, "0.weight", (4, 3), (64, 784)),
              (1, "0.bias", (4,), (64,)),
              (2, "2.weight", (2, 4), (10, 64)),
              (3, "2.bias", (2,), (10,)))
PARAMETER_BY_INDEX = {row[0]: row for row in PARAMETERS}
PARAMETER_BY_NAME = {row[1]: row for row in PARAMETERS}
PLAN_SHAPES = {
    "permutation": (60_000,), "train_indices": (5_000,),
    "validation_indices": (5_000,), "auxiliary_indices": (5_000,),
    "replacement_uniforms": (5_000,), "replacement_digits": (5_000,),
    "training_probe_indices": (256,),
}
KINDS = ("anchor", "source_witness", "branch_results", "independent_audit",
         "final_core", "plan_arrays", "source_completion", "capture_comparison")
GROUP_FIXED_TENSOR_BYTES = 48_083_616
FINAL_CORE_FIXED_TENSOR_BYTES = 7_328_432
FULL_PLAN_TENSOR_BYTES = 1_706_048
PILOT_PLAN_TENSOR_BYTES = 794_688
KNOWN_FIXED_BEFORE_RNG = (18 * GROUP_FIXED_TENSOR_BYTES
                          + 6 * FINAL_CORE_FIXED_TENSOR_BYTES
                          + 4 * FULL_PLAN_TENSOR_BYTES + PILOT_PLAN_TENSOR_BYTES)
MEMBERSHIP = {
    "all_defined_anchor_groups": 18, "independent_audits": 16,
    "long_capture_on_source_completions": 4,
    "pilot_capture_on_source_completions": 1,
    "pilot_capture_off_source_completions": 1,
    "pilot_capture_comparisons": 1, "long_plans": 4, "pilot_plans": 1,
}
_HEX64 = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)
BRANCHES = ("raw", "current", "lagged", "restored", "reciprocal", "zero")
PROBES = ("batch_noisy", "train_probe_noisy", "train_probe_clean", "auxiliary_clean")


def _cpu_only() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or torch.cuda.is_initialized():
        raise ValueError("storage fixture requires explicitly hidden, uninitialized CUDA")


def _path(path) -> str:
    return "root" + "".join("." + str(part) for part in path)


def tensor_leaves(tree, path=()):
    if type(tree) is torch.Tensor:
        yield _path(path), tree
    elif type(tree) is dict:
        for key, value in tree.items():
            yield from tensor_leaves(value, path + (key,))
    elif type(tree) in (list, tuple):
        for index, value in enumerate(tree):
            yield from tensor_leaves(value, path + (index,))


def inventory(tree):
    storage._validate_tree(tree)
    return [{"path": path, "shape": list(tensor.shape), "dtype": str(tensor.dtype),
             "tensor_bytes": tensor.numel() * tensor.element_size()}
            for path, tensor in tensor_leaves(tree)]


def _same_exact(left, right):
    """Insertion-ordered, type-exact equality, including float and tensor bytes."""
    if type(left) is not type(right):
        return False
    if type(left) is torch.Tensor:
        return (left.dtype == right.dtype and left.shape == right.shape and
                torch.equal(left.contiguous().reshape(-1).view(torch.uint8),
                            right.contiguous().reshape(-1).view(torch.uint8)))
    if type(left) is dict:
        return (tuple(left.keys()) == tuple(right.keys()) and
                all(_same_exact(left[key], right[key]) for key in left))
    if type(left) in (list, tuple):
        return len(left) == len(right) and all(_same_exact(a, b) for a, b in zip(left, right))
    if type(left) is float:
        return struct.pack(">d", left) == struct.pack(">d", right)
    return left == right


def _allowlisted_tensor_paths(kind):
    """Exact current paths and dtypes; known-looking tensors elsewhere are rejected."""
    paths = {}
    def add(path, dtype):
        if path in paths:
            raise AssertionError("duplicate tensor path declaration")
        paths[path] = dtype
    def core(prefix):
        for index, _, _, _ in PARAMETERS:
            add(prefix + ("model", "parameters", index, "value"), torch.float32)
            add(prefix + ("optimizer", "state", index, "step"), torch.float32)
            for moment in ("exp_avg", "exp_avg_sq"):
                add(prefix + ("optimizer", "state", index, moment, "value"), torch.float32)
        add(prefix + ("observer", "state", "V", "value"), torch.float32)
        add(prefix + ("observer", "state", "S", "value"), torch.float64)
        add(prefix + ("observer", "state", "grad_mean", "value"), torch.float32)
        add(prefix + ("rng", "numpy", "keys"), torch.uint32)
        add(prefix + ("rng", "torch_cpu"), torch.uint8)
        add(prefix + ("rng", "continuation_witness", "torch_cpu"), torch.float64)
    if kind == "anchor":
        core(("payload",))
        add(("payload", "bindings", "plan", "next_batch_indices"), torch.int64)
        add(("payload", "bindings", "probes", "training_probe_indices"), torch.int64)
        add(("payload", "bindings", "probes", "auxiliary_indices"), torch.int64)
    elif kind == "source_witness":
        for owner in ("raw_gradient", "delivered_current_gradient"):
            add(("payload", owner, "value"), torch.float32)
        for index, _, _, _ in PARAMETERS:
            add(("payload", "parameters_after", index, "value"), torch.float32)
            add(("payload", "optimizer_after", "state", index, "step"), torch.float32)
            for moment in ("exp_avg", "exp_avg_sq"):
                add(("payload", "optimizer_after", "state", index, moment, "value"), torch.float32)
        for owner, dtype in (("V", torch.float32), ("S", torch.float64),
                             ("grad_mean", torch.float32)):
            add(("payload", "observer_after", "state", owner, "value"), dtype)
    elif kind == "branch_results":
        for owner in ("g", "c", "l"):
            add(("payload", "candidate_state", owner, "value"), torch.float32)
        for owner in ("previous_basis", "post_ingest_basis"):
            add(("payload", "candidate_state", owner, "value"), torch.float32)
        for owner, dtype in (("V", torch.float32), ("S", torch.float64),
                             ("grad_mean", torch.float32)):
            add(("payload", "candidate_state", "observer_after", "state", owner, "value"), dtype)
        for probe in PROBES:
            add(("payload", "candidate_state", "measurement_before", "probes", probe,
                 "cpu64", "q", "value"), torch.float64)
        for branch in BRANCHES:
            prefix = ("payload", "branches", branch)
            add(prefix + ("delivered_gradient", "value"), torch.float32)
            add(prefix + ("parameters_after_flat",), torch.float32)
            for index, _, _, _ in PARAMETERS:
                add(prefix + ("parameters_after", index, "value"), torch.float32)
                add(prefix + ("optimizer_after", "state", index, "step"), torch.float32)
                for moment in ("exp_avg", "exp_avg_sq"):
                    add(prefix + ("optimizer_after", "state", index, moment, "value"), torch.float32)
            for owner in ("delta", "delta_data"):
                add(prefix + ("measurement", "displacement", owner, "value"), torch.float64)
    elif kind == "final_core":
        core(())
    elif kind == "source_completion":
        core(("final_state_core",))
    elif kind == "plan_arrays":
        for name in (*PLAN_SHAPES, "training_batches"):
            add((name,), torch.float64 if name == "replacement_uniforms" else torch.int64)
    elif kind == "independent_audit":
        pass
    else:
        raise ValueError("unsupported tensor-tree kind")
    return paths


def _parameter_ancestor(ancestors, *, optimizer: bool):
    keys = ("parameter_index", "parameter_name") if optimizer else ("index", "name")
    for item in reversed(ancestors):
        if type(item) is dict and all(key in item for key in keys):
            index, name = item[keys[0]], item[keys[1]]
            if type(index) is not int or type(name) is not str or index not in PARAMETER_BY_INDEX:
                raise ValueError("unknown parameter index/name in tensor path")
            row = PARAMETER_BY_INDEX[index]
            if name != row[1]:
                raise ValueError("parameter index/name mapping differs")
            return row
    raise ValueError("parameter tensor lacks explicit index/name mapping")


def _classified_shape(kind, path, tensor, ancestors, plan_steps):
    """Return (target shape, classification), failing on every unknown leaf."""
    old = tuple(tensor.shape)
    allowed = _allowlisted_tensor_paths(kind)
    if path not in allowed:
        raise ValueError("unknown tensor path: " + _path(path))
    if tensor.dtype != allowed[path]:
        raise ValueError("tensor dtype changed at " + _path(path))
    if "rng" in path:
        suffixes = (("rng", "numpy", "keys"), ("rng", "torch_cpu"),
                    ("rng", "continuation_witness", "torch_cpu"))
        if not any(tuple(path[-len(suffix):]) == suffix for suffix in suffixes):
            raise ValueError("unknown RNG tensor path: " + _path(path))
        if path[-2:] == ("numpy", "keys") and (old, tensor.dtype) != ((624,), torch.uint32):
            raise ValueError("observed NumPy RNG layout changed")
        if path[-2:] == ("continuation_witness", "torch_cpu") and (old, tensor.dtype) != ((4,), torch.float64):
            raise ValueError("observed CPU continuation witness layout changed")
        if path[-1] == "torch_cpu" and path[-2] == "rng" and not (tensor.dtype == torch.uint8 and tensor.ndim == 1):
            raise ValueError("observed Torch CPU RNG layout changed")
        return old, "observed_cpu_rng_unscaled"

    parent = ancestors[-1] if ancestors else None
    if path[-1:] == ("value",) and type(parent) is dict and "index" in parent and "name" in parent:
        row = _parameter_ancestor(ancestors, optimizer=False)
        if old != row[2]:
            raise ValueError("fixture parameter shape changed at " + _path(path))
        return row[3], "parameter:" + row[1]

    if path[-1:] == ("value",) and len(path) >= 2 and path[-2] in ("exp_avg", "exp_avg_sq"):
        row = _parameter_ancestor(ancestors, optimizer=True)
        if old != row[2]:
            raise ValueError("fixture optimizer moment shape changed at " + _path(path))
        return row[3], "optimizer_moment:" + row[1]

    if path[-1:] == ("step",):
        _parameter_ancestor(ancestors, optimizer=True)
        if old != () or tensor.dtype != torch.float32:
            raise ValueError("optimizer step layout changed")
        return (), "optimizer_step_scalar"

    owner = path[-2] if len(path) >= 2 and path[-1] == "value" else None
    if owner in ("V", "previous_basis", "post_ingest_basis"):
        if old != (26, 2):
            raise ValueError("fixture basis shape changed at " + _path(path))
        return (P, RANK), "observer_basis"
    if owner == "S":
        if old != (2,):
            raise ValueError("fixture spectrum shape changed at " + _path(path))
        return (RANK,), "observer_spectrum"
    if owner in ("grad_mean", "raw_gradient", "delivered_current_gradient", "g", "c", "l",
                 "delivered_gradient", "q", "delta", "delta_data"):
        if old != (26,):
            raise ValueError("fixture flat-vector shape changed at " + _path(path))
        return (P,), "flat_vector:" + owner
    if path[-1:] == ("parameters_after_flat",):
        if old != (26,):
            raise ValueError("fixture flat endpoint shape changed")
        return (P,), "flat_parameter_endpoint"

    index_targets = {
        ("bindings", "plan", "next_batch_indices"): (64,),
        ("bindings", "probes", "training_probe_indices"): (256,),
        ("bindings", "probes", "auxiliary_indices"): (5_000,),
    }
    for suffix, target in index_targets.items():
        if tuple(path[-len(suffix):]) == suffix:
            expected = {suffixes: shapes for suffixes, shapes in (
                (("bindings", "plan", "next_batch_indices"), (4,)),
                (("bindings", "probes", "training_probe_indices"), (4,)),
                (("bindings", "probes", "auxiliary_indices"), (10,)),
            )}[suffix]
            if old != expected or tensor.dtype != torch.int64:
                raise ValueError("fixture bound-index layout changed")
            return target, "bound_index:" + suffix[-1]

    if kind == "plan_arrays" and len(path) == 1 and path[0] in (*PLAN_SHAPES, "training_batches"):
        expected = {"permutation": (30,), "train_indices": (10,), "validation_indices": (10,),
                    "auxiliary_indices": (10,), "replacement_uniforms": (10,),
                    "replacement_digits": (10,), "training_batches": (8, 4),
                    "training_probe_indices": (4,)}[path[0]]
        if old != expected:
            raise ValueError("fixture plan-array shape changed at " + _path(path))
        target = (plan_steps, 64) if path[0] == "training_batches" else PLAN_SHAPES[path[0]]
        return target, "plan_array:" + path[0]
    raise ValueError("unknown tensor path: " + _path(path))


def _bitset_dimension(path, record):
    if tuple(record.keys()) != ("encoding", "dimension", "count", "bits_hex"):
        return None
    if record["encoding"] != "flat_bitset_hex_lsb0_v1":
        raise ValueError("unknown bitset encoding at " + _path(path))
    allowed = _allowed_audit_bitsets()
    if path not in allowed:
        raise ValueError("unknown audit bitset path: " + _path(path))
    fixture_dimension, lifted_dimension = allowed[path]
    if record["dimension"] != fixture_dimension:
        raise ValueError("fixture audit bitset dimension changed at " + _path(path))
    return lifted_dimension


def _allowed_audit_bitsets():
    """Exact 92 current all-defined-audit diagnostic paths and dimensions."""
    paths = {}
    def add(path, fixture_dimension, lifted_dimension):
        if path in paths:
            raise AssertionError("duplicate audit bitset path declaration")
        paths[path] = (fixture_dimension, lifted_dimension)
    prefix = ("payload", "candidate_operator_audit", "projections", "result")
    for candidate in ("current", "lagged"):
        for field in ("failing_flat_indices", "identity_mismatch_flat_indices"):
            add(prefix + (candidate, field), 26, P)
    prefix = ("payload", "branch_audits", "result")
    for branch in BRANCHES:
        for _, name, fixture_shape, lifted_shape in PARAMETERS:
            fixture_dimension = 1
            lifted_dimension = 1
            for size in fixture_shape:
                fixture_dimension *= size
            for size in lifted_shape:
                lifted_dimension *= size
            for check in ("theta", "moment", "variance"):
                add(prefix + (branch, "parameters", name, "checks", check,
                              "failing_flat_indices"), fixture_dimension, lifted_dimension)
    prefix = ("payload", "measurement_audits", "result", "measurement_audits")
    for probe in PROBES:
        add(prefix + ("before." + probe + ".q_values", "failing_flat_indices"), 26, P)
    for branch in BRANCHES:
        for field in ("delta_values", "delta_data_values"):
            add(prefix + ("branches." + branch + ".displacement." + field,
                          "failing_flat_indices"), 26, P)
    if len(paths) != 92:
        raise AssertionError("audit bitset allowlist count differs")
    return paths


def _binding_shape(path, value, plan_steps):
    if len(path) >= 2 and path[-2] == "arrays" and path[-1] in (*PLAN_SHAPES, "training_batches"):
        if type(value) is dict and "shape" in value:
            return list((plan_steps, 64) if path[-1] == "training_batches" else PLAN_SHAPES[path[-1]])
    return None


def lift_tree(kind, tree, *, plan_steps=2_000, pattern="zero"):
    """Make one owned storage-only copy, exhaustively classifying every tensor."""
    _cpu_only()
    if kind not in KINDS or kind == "capture_comparison":
        raise ValueError("unsupported tensor-tree kind")
    if type(plan_steps) is not int or type(plan_steps) is bool or plan_steps not in (220, 2_000):
        raise ValueError("plan_steps must be 220 or 2000")
    if pattern not in ("zero", "ones") or type(pattern) is not str:
        raise ValueError("pattern must be zero or ones")
    manifest = []
    bitsets = []
    seen_tensor_paths = set()
    seen_bitset_paths = set()

    def visit(value, path=(), ancestors=()):
        if type(value) is torch.Tensor:
            target, classification = _classified_shape(kind, path, value, ancestors, plan_steps)
            seen_tensor_paths.add(path)
            if classification == "observed_cpu_rng_unscaled":
                result = value.detach().cpu().contiguous().clone()
                value_policy = "actual_fixture_rng_bytes_cloned"
            else:
                result = torch.zeros(target, dtype=value.dtype, device="cpu")
                if pattern == "ones":
                    result.fill_(1)
                value_policy = "synthetic_" + pattern
            manifest.append({"path": _path(path), "fixture_shape": list(value.shape),
                             "lifted_shape": list(target), "dtype": str(value.dtype),
                             "classification": classification, "value_policy": value_policy})
            return result
        if type(value) is dict:
            dimension = _bitset_dimension(path, value) if "encoding" in value else None
            if dimension is not None:
                from audit_diagnostics import pack_indices
                result = pack_indices(list(range(dimension)), dimension)
                bitsets.append({"path": _path(path), "fixture_dimension": value["dimension"],
                                "lifted_dimension": dimension, "count": result["count"],
                                "bits_hex_chars": len(result["bits_hex"])})
                seen_bitset_paths.add(path)
                return result
            result = {key: visit(item, path + (key,), ancestors + (value,))
                      for key, item in value.items()}
            shape = _binding_shape(path, value, plan_steps)
            if shape is not None:
                result["shape"] = shape
            if type(value.get("value")) is torch.Tensor and "shape" in value:
                if type(result["shape"]) is not list:
                    raise ValueError("tensor shape metadata must remain a list")
                result["shape"] = list(result["value"].shape)
            return result
        if type(value) is list:
            return [visit(item, path + (index,), ancestors + (value,)) for index, item in enumerate(value)]
        if type(value) is tuple:
            return tuple(visit(item, path + (index,), ancestors + (value,)) for index, item in enumerate(value))
        if value is None or type(value) in (bool, int, float, str):
            return value
        raise ValueError("unsupported fixture leaf at " + _path(path))

    lifted = visit(tree)
    if type(lifted) is dict and "profile" in lifted:
        # This unregistered value is the on-tree, fail-closed label.  Current
        # schema fields/order remain present, but registered validators reject it.
        lifted["profile"] = SPECIMEN_PROFILE
    if kind == "plan_arrays":
        lifted["steps_total"] = plan_steps
    storage._validate_tree(lifted)
    expected_tensor_paths = set(_allowlisted_tensor_paths(kind))
    if seen_tensor_paths != expected_tensor_paths:
        raise ValueError("required tensor path set differs from exact current layout")
    expected_bitset_paths = set(_allowed_audit_bitsets()) if kind == "independent_audit" else set()
    if seen_bitset_paths != expected_bitset_paths:
        raise ValueError("required audit bitset path set differs from exact current layout")
    return {"tree": lifted, "tensor_manifest": manifest, "bitset_manifest": bitsets,
            "specimen_profile": SPECIMEN_PROFILE, "scientific_envelope": False,
            "semantic_validator_applied": False}


def _unique_digest(label, path):
    return hashlib.sha256((label + "|" + _path(path)).encode("utf-8")).hexdigest()


def _unique_digest_tree(value, label, path=()):
    if type(value) is dict:
        return {key: _unique_digest_tree(item, label, path + (key,)) for key, item in value.items()}
    if type(value) is list:
        return [_unique_digest_tree(item, label, path + (index,)) for index, item in enumerate(value)]
    if type(value) is tuple:
        return tuple(_unique_digest_tree(item, label, path + (index,)) for index, item in enumerate(value))
    if type(value) is str and _HEX64.fullmatch(value):
        return _unique_digest(label, path)
    return value


def lift_source_completion(tree, *, steps, capture_mode, long_run=False, pattern="zero"):
    """Lift one actual tiny completion to a nonsemantic current record layout."""
    expected = 2_000 if long_run else 220
    if steps != expected or capture_mode not in ("capture_on", "capture_off"):
        raise ValueError("invalid source layout membership")
    base = lift_tree("source_completion", tree, plan_steps=steps, pattern=pattern)
    value = base["tree"]
    value["capture_mode"] = capture_mode
    value["trajectory"]["steps_total"] = steps
    value["trajectory"]["anchor_updates"] = [101, 500, 1000, 2000] if long_run else [101, 200]
    template = value["trace"][-1]
    value["trace"] = []
    for update in range(1, steps + 1):
        row = _unique_digest_tree(template, "source-trace-" + str(update))
        row["completed_updates"] = update
        row["completed_observations"] = update
        row["next_anchor_update"] = update + 1
        value["trace"].append(row)
    refs = value["anchor_witness_refs"]
    if capture_mode == "capture_off":
        value["anchor_witness_refs"] = []
    else:
        template_ref = refs[0]
        value["anchor_witness_refs"] = []
        for update in value["trajectory"]["anchor_updates"]:
            row = _unique_digest_tree(template_ref, "source-ref-" + str(update))
            row["anchor_update"] = update
            value["anchor_witness_refs"].append(row)
    core = value["final_state_core"]
    core["anchor_update"] = steps + 1
    core["state_completed_updates"] = steps
    core["optimizer"]["state_completed_updates"] = steps
    core["observer"]["state_completed_observations"] = steps
    core["observer"]["state"]["step_count"] = steps
    value = _unique_digest_tree(value, "source-completion")
    storage._validate_tree(value)
    base["tree"] = value
    base["layout"] = "long_capture_on" if long_run else "pilot_" + capture_mode
    base["synthetic_unique_trace_digests"] = True
    return base


def lift_capture_comparison(tree, *, steps=220):
    """Expand exact compact comparison fields; no historical states are invented."""
    _cpu_only()
    if steps != 220:
        raise ValueError("only the registered pilot comparison length is represented")
    value = _unique_digest_tree(tree, "comparison-base")
    value["profile"] = SPECIMEN_PROFILE
    value["trajectory"]["steps_total"] = steps
    value["trajectory"]["anchor_updates"] = [101, 200]
    template = value["step_trace"][-1]
    rows = []
    for update in range(1, steps + 1):
        row = _unique_digest_tree(template, "comparison-row-" + str(update))
        row["completed_updates"] = update
        for side in ("on", "off"):
            row[side]["completed_updates"] = update
            row[side]["completed_observations"] = update
            row[side]["next_anchor_update"] = update + 1
        rows.append(row)
    value["step_trace"] = rows
    for key in value["summary"]:
        value["summary"][key] = steps
    storage._validate_tree(value)
    return {"tree": value, "tensor_manifest": [], "bitset_manifest": [],
            "specimen_profile": SPECIMEN_PROFILE, "scientific_envelope": False,
            "semantic_validator_applied": False, "layout": "pilot_comparison",
            "synthetic_unique_trace_digests": True}


def actual_tiny_records():
    """Run actual registered tiny producers/validators and return owned records."""
    _cpu_only()
    import tempfile
    import torch.nn.functional as F
    import anchor_envelope as anchor_schema
    import artifact_envelopes as branch_schema
    import audit_envelope as audit_schema
    import branch_execution as execution
    import data_probe_bindings as data
    import source_capture as source
    import source_history as history
    import state_core as state
    from envelope_fixture import fixture_context, warm_live
    from test_source_history import (artifact_ref, identity as history_identity, plan_ref,
                                     run_history)

    saved_rng = state._raw_rng_state()
    try:
        handle = {}
        with tempfile.TemporaryDirectory(prefix="i7-full-envelope-storage-") as directory:
            with storage.ArtifactStore(directory, profile=storage.MLP_FIXTURE,
                                       min_filesystem_free_bytes=0) as store:
                with fixture_context(plan_handle=handle, store=store) as context:
                    model, optimizer, observer = warm_live(context)
                    transaction = source.capture_anchor_then_live_witness(
                        model, optimizer, observer, store=store,
                        created_utc="2026-09-06T21:30:00Z", **context)
                    branch = execution.execute_branches(
                        store=store, created_utc="2026-09-06T21:30:01Z",
                        **transaction, **context)
                    audited = audit_schema.audit_and_seal(
                        store=store, plan_root=handle["root"], plan_name=handle["name"],
                        created_utc="2026-09-06T21:30:02Z", **transaction,
                        branch=branch["artifact"], branch_receipt=branch["receipt"], **context)

                    anchor_schema.validate_anchor(transaction["anchor"], **context)
                    source.validate_source_witness(transaction["witness"],
                        anchor=transaction["anchor"], anchor_receipt=transaction["anchor_receipt"],
                        **context)
                    branch_schema.validate_branch_results(branch["artifact"], **transaction,
                        **context, expected_candidates=branch["candidates"],
                        expected_measurements=branch["measurements"])
                    audit_inputs = dict(**transaction, branch=branch["artifact"],
                        branch_receipt=branch["receipt"], plan_reference=handle["reference"],
                        auditor_environment=audited["artifact"]["payload"]["exact_validation"]["auditor_environment"],
                        **context)
                    audit_schema.validate_audit(audited["artifact"], **audit_inputs)

                    materialized = data.materialize(context["images_bytes"], context["labels_bytes"],
                        plan=context["plan"], identity=context["identity"], profile=context["profile"],
                        expected_files=context["expected_files"])
                    for indices in context["plan"]["training_batches"][5:]:
                        optimizer.zero_grad(set_to_none=True)
                        F.cross_entropy(model(materialized["datasets"]["train_inputs"][indices]),
                            materialized["datasets"]["train_noisy_labels"][indices]).backward()
                        observer.filter_grad(); optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    final_core = state.capture_core(model, optimizer, observer,
                        profile=context["profile"], completed_updates=8)
                    state.validate_core(final_core)
                    plan = context["plan"]

        on_initial, on_cores = run_history()
        off_initial, off_cores = run_history()
        identity = history_identity()
        common = dict(identity=identity, profile=storage.MLP_FIXTURE, plan_ref=plan_ref(),
                      sources_sha256="5" * 64, environment_sha256="6" * 64)
        on = history.make_source_completion(on_cores, capture_mode="capture_on",
            anchor_refs=[artifact_ref("anchor")], witness_refs=[artifact_ref("source-witness")], **common)
        off = history.make_source_completion(off_cores, capture_mode="capture_off",
            anchor_refs=[], witness_refs=[], **common)
        comparison = history.make_capture_comparison(zip(on_cores, off_cores),
            initial_states=(on_initial, off_initial), identity=identity, profile=storage.MLP_FIXTURE)
        history.validate_capture_bundle(on, off, comparison)
        return {
            "anchor": transaction["anchor"], "source_witness": transaction["witness"],
            "branch_results": branch["artifact"], "independent_audit": audited["artifact"],
            "final_core": final_core, "plan_arrays": plan,
            "source_completion_on": on, "source_completion_off": off,
            "capture_comparison": comparison,
            "semantic_validator_valid": {"anchor": True, "source_witness": True,
                "branch_results": True, "independent_audit": True,
                "source_completion_on": True, "source_completion_off": True,
                "capture_comparison": True},
        }
    finally:
        state._set_rng_state(saved_rng)


def serialized_measurement(tree, *, buffer_limit=64 << 20):
    """Bounded uncompressed ZIP save, store validation and restricted CPU reload."""
    _cpu_only()
    if type(buffer_limit) is not int or type(buffer_limit) is bool or not 0 < buffer_limit <= 64 << 20:
        raise ValueError("buffer_limit must be an exact positive int at most 64 MiB")
    leaves = inventory(tree)
    logical = sum(row["tensor_bytes"] for row in leaves)
    with storage._BoundedBuffer(buffer_limit) as buffer:
        torch.save(tree, buffer, _use_new_zipfile_serialization=True)
        if buffer.exceeded:
            raise storage.StoreError("serialization exceeded bounded buffer")
        view = buffer.getbuffer()
        try:
            size, digest = len(view), hashlib.sha256(view).hexdigest()
        finally:
            view.release()
        buffer.seek(0)
        with zipfile.ZipFile(buffer, allowZip64=True) as archive:
            entries = archive.infolist()
            if any(row.compress_type != zipfile.ZIP_STORED or row.compress_size != row.file_size
                   for row in entries):
                raise ValueError("serialization is not wholly uncompressed ZIP storage")
            storages = [row for row in entries if "/data/" in row.filename]
            if len(storages) != len(leaves) or sum(row.file_size for row in storages) != logical:
                raise ValueError("ZIP storage entries differ from exact owned tensor inventory")
        buffer.seek(0)
        loaded = torch.load(buffer, weights_only=True, map_location="cpu")
        loaded_inventory = inventory(loaded)
        if loaded_inventory != leaves:
            raise ValueError("restricted CPU roundtrip inventory differs")
        if not _same_exact(tree, loaded):
            raise ValueError("restricted CPU roundtrip full tree differs")
        return {"tensor_bytes": logical, "tensor_count": len(leaves),
                "serialized_bytes": size, "metadata_and_zip_overhead_bytes": size - logical,
                "serialized_sha256": digest, "zip_entries": len(entries),
                "all_zip_entries_uncompressed": True, "zip64_capable_writer": True,
                "restricted_weights_only_cpu_roundtrip": True,
                "exact_full_tree_roundtrip": True,
                "exact_owned_tensor_inventory": leaves, "buffer_limit_bytes": buffer_limit}


def json_measurement(tree, *, buffer_limit=64 << 20):
    """Measure the comparison's actual strict compact-JSON representation."""
    _cpu_only()
    if type(buffer_limit) is not int or type(buffer_limit) is bool or not 0 < buffer_limit <= 64 << 20:
        raise ValueError("buffer_limit must be an exact positive int at most 64 MiB")
    import identity_codec as codec
    raw = codec.json_bytes(tree)
    if len(raw) > buffer_limit:
        raise storage.StoreError("JSON serialization exceeded bounded buffer")
    with storage._BoundedBuffer(buffer_limit) as buffer:
        buffer.write(raw)
        encoded = buffer.getvalue()
    loaded = codec.json_loads(encoded, max_bytes=buffer_limit)
    if not _same_exact(tree, loaded):
        raise ValueError("strict JSON roundtrip full tree differs")
    return {"tensor_bytes": 0, "tensor_count": 0, "serialized_bytes": len(encoded),
            "metadata_and_encoding_overhead_bytes": len(encoded),
            "serialized_sha256": hashlib.sha256(encoded).hexdigest(),
            "encoding": "strict_compact_ascii_json_v1", "trailing_newline": True,
            "exact_full_tree_roundtrip": True, "exact_owned_tensor_inventory": [],
            "buffer_limit_bytes": buffer_limit}


def measure(pattern="zero"):
    """Measure one specimen per component kind; never create the repeated study root."""
    _cpu_only()
    started = time.monotonic()
    before = torch.get_rng_state().clone()
    actual = actual_tiny_records()
    specs = {
        "anchor_pilot": lift_tree("anchor", actual["anchor"], plan_steps=220, pattern=pattern),
        "anchor_long": lift_tree("anchor", actual["anchor"], plan_steps=2_000, pattern=pattern),
        "source_witness": lift_tree("source_witness", actual["source_witness"], pattern=pattern),
        "branch_results": lift_tree("branch_results", actual["branch_results"], pattern=pattern),
        "independent_audit_all_failure": lift_tree("independent_audit", actual["independent_audit"], pattern=pattern),
        "source_completion_pilot_on": lift_source_completion(actual["source_completion_on"],
            steps=220, capture_mode="capture_on", pattern=pattern),
        "source_completion_pilot_off": lift_source_completion(actual["source_completion_off"],
            steps=220, capture_mode="capture_off", pattern=pattern),
        "capture_comparison_pilot": lift_capture_comparison(actual["capture_comparison"]),
        "source_completion_long_on": lift_source_completion(actual["source_completion_on"],
            steps=2_000, capture_mode="capture_on", long_run=True, pattern=pattern),
        "plan_pilot": lift_tree("plan_arrays", actual["plan_arrays"], plan_steps=220, pattern=pattern),
        "plan_long": lift_tree("plan_arrays", actual["plan_arrays"], plan_steps=2_000, pattern=pattern),
    }
    measured = {}
    for name, specimen in specs.items():
        row = (json_measurement(specimen["tree"]) if name == "capture_comparison_pilot"
               else serialized_measurement(specimen["tree"]))
        row["tensor_manifest"] = specimen["tensor_manifest"]
        row["bitset_manifest"] = specimen["bitset_manifest"]
        measured[name] = row

    group_tensor = (measured["anchor_long"]["tensor_bytes"]
                    + measured["source_witness"]["tensor_bytes"]
                    + measured["branch_results"]["tensor_bytes"])
    observed_per_core = {row["path"]: row["tensor_bytes"] for row in
        measured["anchor_long"]["exact_owned_tensor_inventory"] if ".rng." in row["path"]}
    observed_rng_per_core = sum(observed_per_core.values())
    if group_tensor - observed_rng_per_core != GROUP_FIXED_TENSOR_BYTES:
        raise ValueError("current group fixed-tensor ledger cross-check differs")
    source_core_tensor = measured["source_completion_long_on"]["tensor_bytes"]
    if source_core_tensor - observed_rng_per_core != FINAL_CORE_FIXED_TENSOR_BYTES:
        raise ValueError("current final-core fixed-tensor ledger cross-check differs")
    if measured["plan_long"]["tensor_bytes"] != FULL_PLAN_TENSOR_BYTES or \
       measured["plan_pilot"]["tensor_bytes"] != PILOT_PLAN_TENSOR_BYTES:
        raise ValueError("current plan-array tensor ledger cross-check differs")
    bitsets = measured["independent_audit_all_failure"]["bitset_manifest"]
    if len(bitsets) != 92 or sum(row["bits_hex_chars"] for row in bitsets) != 483_512:
        raise ValueError("complete audit bitset ledger differs")

    multipliers = {"anchor_pilot": 2, "anchor_long": 16, "source_witness": 18,
        "branch_results": 18, "independent_audit_all_failure": 16,
        "source_completion_pilot_on": 1, "source_completion_pilot_off": 1,
        "capture_comparison_pilot": 1, "source_completion_long_on": 4,
        "plan_pilot": 1, "plan_long": 4}
    aggregate_serialized = sum(measured[name]["serialized_bytes"] * count
                               for name, count in multipliers.items())
    aggregate_observed_tensor = sum(measured[name]["tensor_bytes"] * count
                                    for name, count in multipliers.items())
    # The anchor paths are present here; source-final-core paths have the same
    # registered fixture layout and are independently checked in their inventory.
    non_torch_cpu_per_core = 2_496 + 32
    known_cuda_witness_lower_bound = 24 * 16
    if not torch.equal(before, torch.get_rng_state()) or torch.cuda.is_initialized():
        raise ValueError("measurement changed caller Torch RNG or initialized CUDA")
    return {
        "schema": "i7_full_envelope_storage_measurement_v1",
        "evidence_role": "dataset_free_cpu_engineering", "specimen_profile": SPECIMEN_PROFILE,
        "pattern": pattern, "original_tiny_semantic_validator_valid": actual["semantic_validator_valid"],
        "lifted_semantic_validator_valid": False, "components": measured,
        "synthetic_value_policy": {"non_rng_tensors": "synthetic_" + pattern,
            "cpu_rng_tensors": "actual fixture bytes cloned into fresh exact-owned storage",
            "cpu_rng_interpretation": "observed fixture shape/state only; not native retained state"},
        "membership": dict(MEMBERSHIP), "serialized_multipliers": multipliers,
        "known_fixed_tensor_bytes_before_rng": KNOWN_FIXED_BEFORE_RNG,
        "known_tensor_lower_bound_including_fixed_rng_without_native_states":
            KNOWN_FIXED_BEFORE_RNG + 24 * (2_496 + 32 + 16),
        "aggregate_observed_fixture_tensor_bytes": aggregate_observed_tensor,
        "aggregate_measured_component_serialized_bytes": aggregate_serialized,
        "audit_all_failure_bits": {"records_per_audit": 92,
            "ascii_hex_chars_per_audit": 483_512, "audits": 16,
            "ascii_hex_chars_all_audits": 7_736_192,
            "synthetic_not_numerically_recomputed": True},
        "rng_scope": {"full_cores": 24, "observed_cpu_per_core": observed_per_core,
            "observed_cpu_tensor_bytes_per_core": observed_rng_per_core,
            "known_numpy_and_cpu_continuation_bytes_all_cores": 24 * non_torch_cpu_per_core,
            "known_cuda_continuation_witness_lower_bound_bytes_per_core": 16,
            "known_cuda_continuation_witness_lower_bound_bytes_all_cores": known_cuda_witness_lower_bound,
            "native_torch_cpu_state_layout": "unresolved",
            "native_cuda_rng_state_layout": "unresolved", "cuda_values_instantiated": False},
        "cap_bytes": 1 << 30, "allowance_fit_claim": None,
        "complete_attempt_fit_certificate": False, "scientific_launch_approved": False,
        "exclusions": ["native Torch CPU and CUDA RNG state layouts",
            "native adapter and remaining envelope fields", "actual scientific scalar/string values",
            "writer temporaries and simultaneous in-memory copies", "receipts, controller files and failures",
            "filesystem allocation, process peaks and all-phase resource history"],
        "in_memory_only_except_automatically_cleaned_tiny_fixture": True,
        "cuda_initialized": False, "rng_preserved": True,
        "wall_seconds": time.monotonic() - started,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measure", action="store_true",
                        help="explicitly run the bounded CPU-only fixture and storage lifts")
    parser.add_argument("--pattern", choices=("zero", "ones"), default="zero")
    args = parser.parse_args(argv)
    if not args.measure:
        parser.print_help()
        return 0
    print(json.dumps(measure(args.pattern), sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
