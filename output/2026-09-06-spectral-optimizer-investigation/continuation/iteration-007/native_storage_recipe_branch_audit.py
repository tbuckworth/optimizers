"""Explicit symbolic recipes for the maximum branch and audit components.

The branch recipe is one complete all-defined structural mirror.  The audit
recipe is the deliberately synthetic union used by the independent primitive
bound: it is a byte-cost witness, not a value an auditor can return.  Neither
builder imports Torch or any producer/validator, allocates tensors, reads files,
or grants execution authority.
"""
from __future__ import annotations

import hashlib
import json
import math

from native_storage_recipe_common import (
    PROFILE, SLOT, RecipeError, copy_primitive,
)
from native_storage_recipe_core import (
    envelope, native_tensor_metadata, observer, optimizer, parameter_rows,
)


COMPONENTS = ("branch_results", "independent_audit_all_failure")
BRANCHES = ("raw", "current", "lagged", "restored", "reciprocal", "zero")
PROBES = ("batch_noisy", "train_probe_noisy", "train_probe_clean", "auxiliary_clean")
PARAMETERS = (("0.weight", 50_176), ("0.bias", 64),
              ("2.weight", 640), ("2.bias", 10))
COEFFICIENTS = {
    "ordering": {"lagged": 1, "current": -1},
    "direction_at_current_norm": {"restored": 1, "current": -1},
    "direction_at_lagged_norm": {"lagged": 1, "reciprocal": -1},
    "norm_at_current_direction": {"current": 1, "reciprocal": -1},
    "norm_at_lagged_direction": {"restored": 1, "lagged": -1},
    "interaction": {"restored": 1, "lagged": -1, "current": -1, "reciprocal": 1},
    "current_minus_raw": {"current": 1, "raw": -1},
    "lagged_minus_raw": {"lagged": 1, "raw": -1},
    "restored_minus_raw": {"restored": 1, "raw": -1},
    "reciprocal_minus_raw": {"reciprocal": 1, "raw": -1},
    "raw_minus_zero": {"raw": 1, "zero": -1},
    "current_minus_zero": {"current": 1, "zero": -1},
    "lagged_minus_zero": {"lagged": 1, "zero": -1},
    "restored_minus_zero": {"restored": 1, "zero": -1},
    "reciprocal_minus_zero": {"reciprocal": 1, "zero": -1},
}
PAIR_KEYS = tuple(f"{left}__{right}" for index, left in enumerate(BRANCHES)
                  for right in BRANCHES[index + 1:])
UNDEFINED_REASON = {
    "restored": "positive_current_norm_zero_lagged_direction",
    "reciprocal": "positive_lagged_norm_zero_current_direction",
}
CHECK_KEYS = ("saved_structure", "projections", "delivery", "adamw", "measurements")
ERROR_CODES = (
    "AuditEnvelopeError", "EnvelopeError", "ArtifactEnvelopeError", "StateError",
    "ValueError", "TypeError", "KeyError", "IndexError", "RuntimeError",
    "OverflowError", "FloatingPointError", "AuditDiagnosticError",
    "ResourceGuardAbort", "OtherError",
)
EXCLUSIONS = [
    "native32_forward_reexecution", "independent_observer_history",
    "independent_capture_and_control_flow_history", "scientific_phase_order",
    "aggregate_resource_certification", "independent_resource_event_history",
    "hostile_same_user_provenance",
]
MEASUREMENT_EXCLUSIONS = [
    "external_provenance", "Adam_operator_audits", "live_source_replay",
    "native32_forward_reexecution",
]
P = 50_890
BRANCH_RAW_BYTES = 32_977_072
SOURCE_JSON_MAX = 32 << 10
ENVIRONMENT_JSON_MAX = 8 << 10
PARTIAL_VALIDATION_UTF8 = 4 << 20


class StorageRecipeBranchAuditError(RecipeError):
    """The symbolic branch/audit request is outside the closed recipe."""


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise StorageRecipeBranchAuditError(message)


def _sha(label: str) -> str:
    return hashlib.sha256(("i7-storage-recipe:" + label).encode("ascii")).hexdigest()


def _bounded_json_copy(value, maximum: int, label: str):
    """Copy a strict JSON tree and count compact ASCII encoding incrementally."""
    _need(type(value) is dict, label + " must be an exact dictionary")
    copied = copy_primitive(value)
    stack = [copied]
    while stack:
        item = stack.pop()
        kind = type(item)
        if kind is dict:
            _need(all(type(key) is str for key in item), label + " keys must be strings")
            for key in reversed(tuple(item)):
                stack.append(item[key])
        elif kind is list:
            stack.extend(reversed(item))
        else:
            _need(item is None or kind in (bool, int, float, str),
                  label + " contains a non-JSON value")
            _need(kind is not float or math.isfinite(item), label + " contains nonfinite float")
    encoder = json.JSONEncoder(ensure_ascii=True, allow_nan=False,
                               separators=(",", ":"), sort_keys=False)
    total = 1  # identity_codec.json_bytes appends one newline.
    try:
        for chunk in encoder.iterencode(copied):
            total += len(chunk.encode("ascii"))
            _need(total <= maximum, label + " compact JSON exceeds cap")
    except (TypeError, ValueError, UnicodeError, RecursionError) as error:
        raise StorageRecipeBranchAuditError(label + " JSON encoding failed") from error
    return copied, total


def _reference(role: str, bundle: int, update: int, kind: str, schema: str) -> dict:
    identifier = f"{PROFILE}--{role}--b{bundle}--u{update}--{kind}"
    return {
        "artifact_id": identifier, "schema_name": schema,
        "name": identifier + ".pt", "size_bytes": 1 << 30,
        "sha256": _sha(identifier), "status": "complete",
        "encoding": "torch_weights_only",
        "receipt_name": "receipt-" + "0" * 64 + ".json",
        "receipt_size_bytes": 4_096, "receipt_sha256": _sha("receipt-" + identifier),
    }


def _bounded_branch_reference(schema: str) -> dict:
    """Realize the branch theorem's admitted 128-byte identifier maximum."""
    identifier = "a" * 128
    return {"artifact_id": identifier, "schema_name": schema,
            "name": identifier + ".pt", "size_bytes": 1 << 30,
            "sha256": _sha("bounded-branch-reference-" + schema), "status": "complete",
            "encoding": "torch_weights_only",
            "receipt_name": "receipt-" + "0" * 64 + ".json",
            "receipt_size_bytes": 4_096,
            "receipt_sha256": _sha("bounded-branch-receipt-" + schema)}


# ---------------------------------------------------------------------------
# Branch-result mirror.  Private variant parameters retain every domain shape;
# build_template deliberately chooses the independently calculated maximum.

def _vector_record() -> dict:
    return {"value": SLOT, "shape": [P], "dtype": "float64", "device": "cpu",
            "sha256": _sha("vector")}


def _concordance() -> dict:
    return {"abs_discrepancy": 0.0, "descriptive_ceiling": 0.0,
            "status": "within_scale"}


def _geometry(mask_names, defined: bool, mask=None) -> dict:
    mask = ({name: defined for name in mask_names} if mask is None
            else {name: mask[name] for name in mask_names})
    if not defined:
        return {"defined": False, "defined_mask": mask,
                "reason": "domain_undefined_required_branch",
                "distance": None, "cosine": None, "cosine_reason": None,
                "left_norm": None, "right_norm": None}
    # The zero-norm alternative is the larger legal defined geometry.
    return {"defined": True, "defined_mask": mask, "reason": None,
            "distance": 0.0, "cosine": None, "cosine_reason": "zero_norm",
            "left_norm": 0.0, "right_norm": 0.0}


def _before_chunk(index: int) -> dict:
    return {"chunk_index": index, "start": index * 500, "end": (index + 1) * 500,
            "count": 500, "input_sha256": _sha(f"before-input-{index}"),
            "label_sha256": _sha(f"before-label-{index}"),
            "cpu64_before_ce": 0.0, "native32_before_ce": 0.0,
            "concordance": _concordance()}


def _before_probe(probe: str, sample_count: int) -> dict:
    chunks = [_before_chunk(index) for index in range(10)] if probe == "auxiliary_clean" else []
    return {"sample_count": sample_count, "sample_identity_sha256": _sha("sample-" + probe),
            "cpu64": {"before_ce": 0.0, "q": _vector_record(), "q_norm": 0.0},
            "native32": {"before_ce": 0.0}, "concordance": _concordance(),
            "auxiliary_chunks": chunks}


def _measurement_before() -> dict:
    return {"probe_order": list(PROBES),
            "probes": {probe: _before_probe(probe, count)
                       for probe, count in zip(PROBES, (64, 256, 256, 5_000))}}


def _branch_chunk(branch: str, index: int) -> dict:
    return {"chunk_index": index, "start": index * 500, "end": (index + 1) * 500,
            "count": 500, "input_sha256": _sha(f"{branch}-input-{index}"),
            "label_sha256": _sha(f"{branch}-label-{index}"),
            "cpu64_after_ce": 0.0, "cpu64_Y": 0.0,
            "native32_after_ce": 0.0, "native32_Y": 0.0,
            "concordance": {"after": _concordance(), "Y": _concordance()}}


def _branch_probe(branch: str, probe: str) -> dict:
    chunks = ([_branch_chunk(branch, index) for index in range(10)]
              if probe == "auxiliary_clean" else [])
    return {
        "before_binding": {
            "measurement_before_artifact_id": "m" * 128, "probe_key": probe,
            "before_ce_cpu64_sha256": _sha("before-ce-" + probe),
            "q_sha256": _sha("q-" + probe),
        },
        "cpu64": {"after_ce": 0.0, "Y": 0.0, "D": 0.0, "Ddata": 0.0, "R": 0.0},
        "native32": {"after_ce": 0.0, "Y": 0.0},
        "concordance": {"after": _concordance(), "Y": _concordance()},
        "auxiliary_chunks": chunks,
    }


def _branch_measurement(branch: str) -> dict:
    return {
        "displacement": {
            "delta": _vector_record(), "delta_data": _vector_record(),
            "delta_norm": 0.0, "delta_squared_norm": 0.0,
            "delta_data_norm": 0.0, "delta_data_squared_norm": 0.0,
            "delta_delta_data": _geometry((branch,), True),
        },
        "probes": {probe: _branch_probe(branch, probe) for probe in PROBES},
    }


def _defined_branch(branch: str, update: int) -> dict:
    return {"status": "defined", "reason": None,
            "delivered_gradient": native_tensor_metadata((P,)),
            "assigned_gradient_null_mask": [False] * 4,
            "parameters_after": parameter_rows(), "parameters_after_flat": SLOT,
            "optimizer_after": optimizer(completed_updates=update),
            "measurement": _branch_measurement(branch)}


def _undefined_branch(branch: str) -> dict:
    return {"status": "undefined", "reason": UNDEFINED_REASON[branch],
            "delivered_gradient": None, "assigned_gradient_null_mask": None,
            "parameters_after": None, "parameters_after_flat": None,
            "optimizer_after": None, "measurement": None}


def _leverage(domain: str) -> dict:
    _need(domain in ("both_positive", "both_zero", "restored_undefined",
                     "reciprocal_undefined"), "unknown leverage domain")
    values = {
        "both_positive": (0.0, 0.0, None, 0.0, 0.0, None),
        "both_zero": (None, None, "zero_lagged_norm", None, None, "zero_norm"),
        "restored_undefined": (0.0, None, "zero_lagged_norm", None, None, "zero_norm"),
        "reciprocal_undefined": (0.0, 0.0, None, None, None, "zero_norm"),
    }[domain]
    separation, ratio, ratio_reason, direction, cosine, direction_reason = values
    return {"current_norm": 0.0, "lagged_norm": 0.0,
            "signed_norm_difference": 0.0, "relative_norm_separation": separation,
            "current_lagged_norm_ratio": ratio, "ratio_reason": ratio_reason,
            "unit_direction_distance": direction, "unit_direction_cosine": cosine,
            "direction_reason": direction_reason, "norm_leverage": False,
            "direction_leverage": False}


def _candidate(undefined_branch: str | None) -> dict:
    leverage_domain = "both_zero" if undefined_branch is None else undefined_branch + "_undefined"
    defined = {name: name != undefined_branch for name in BRANCHES}
    reasons = {name: UNDEFINED_REASON[name] if name == undefined_branch else None
               for name in BRANCHES}
    return {
        "g": native_tensor_metadata((P,)), "c": native_tensor_metadata((P,)),
        "l": native_tensor_metadata((P,)),
        "previous_basis": native_tensor_metadata((P, 32)),
        "post_ingest_basis": native_tensor_metadata((P, 32)),
        "nc": 0.0, "nl": 0.0, "observer_after": observer(completed_observations=2_000),
        "measurement_before": _measurement_before(), "leverage": _leverage(leverage_domain),
        "zero_cases": {"current_zero": undefined_branch in (None, "reciprocal"),
                       "lagged_zero": undefined_branch in (None, "restored"),
                       "branch_defined_mask": defined, "branch_reasons": reasons},
    }


def _pair(pair_key: str, undefined_branch: str | None) -> dict:
    names = tuple(pair_key.split("__"))
    mask = {name: name != undefined_branch for name in names}
    defined = all(mask.values())
    if not defined:
        return {"defined": False, "defined_mask": mask,
                "reason": "domain_undefined_required_branch",
                "delivered_gradient": _geometry(names, False, mask),
                "delta": _geometry(names, False, mask),
                "delta_data": _geometry(names, False, mask),
                "full_data_distance_discrepancy": None,
                "full_data_rounding_ceiling": None, "rounding_status": "domain_undefined"}
    return {"defined": True, "defined_mask": {name: True for name in names}, "reason": None,
            "delivered_gradient": _geometry(names, True), "delta": _geometry(names, True),
            "delta_data": _geometry(names, True), "full_data_distance_discrepancy": 0.0,
            "full_data_rounding_ceiling": 0.0, "rounding_status": "pass"}


def _scalar_summary(coefficients: dict[str, int]) -> dict:
    return {"component_values": {name: 0.0 for name in coefficients},
            "direct_value": 0.0, "component_sum_value": 0.0,
            "identity_abs_discrepancy": 0.0, "identity_rounding_ceiling": 0.0,
            "identity_status": "pass"}


def _contrast_chunk(index: int, coefficients: dict[str, int]) -> dict:
    scalar = _scalar_summary(coefficients)
    return {"chunk_index": index, "start": index * 500, "end": (index + 1) * 500,
            "count": 500, "input_sha256": _sha(f"contrast-input-{index}"),
            "label_sha256": _sha(f"contrast-label-{index}"),
            "cpu64": {"Y": copy_primitive(scalar)},
            "native32": {"Y": copy_primitive(scalar)},
            "concordance": {"Y": _concordance()}}


def _contrast_probe(probe: str, coefficients: dict[str, int]) -> dict:
    return {"cpu64": {name: _scalar_summary(coefficients)
                      for name in ("Y", "D", "Ddata", "R")},
            "native32": {"Y": _scalar_summary(coefficients)},
            "concordance": {"Y": _concordance()},
            "auxiliary_chunks": ([_contrast_chunk(index, coefficients) for index in range(10)]
                                 if probe == "auxiliary_clean" else [])}


def _requirements(name: str) -> list[str]:
    if name.startswith("direction_"):
        return ["direction"]
    if name.startswith("norm_"):
        return ["norm"]
    if name == "interaction":
        return ["direction", "norm"]
    return []


def _contrast(name: str, coefficients: dict[str, int], undefined_branch: str | None) -> dict:
    requirements = _requirements(name)
    mask = {branch: branch != undefined_branch for branch in BRANCHES}
    defined = undefined_branch is None or undefined_branch not in coefficients
    def factor(kind: str) -> str:
        if kind not in requirements:
            return "not_required"
        if not defined or undefined_branch is not None and kind == "direction":
            return "unavailable"
        if undefined_branch is not None:
            return "qualified"
        return "unavailable"
    base = {"coefficients": dict(coefficients),
            "branch_defined_mask": mask,
            "defined": defined,
            "reason": None if defined else "domain_undefined_required_branch",
            "factor_requirements": requirements,
            "factor_leverage": {kind: factor(kind) for kind in ("direction", "norm")}}
    if not defined:
        return {**base, "vector": None, "probes": None}
    summary = {"representation": "derived_from_bound_branch_vectors",
               "canonical_sha256": _sha("contrast-" + name), "norm": 0.0,
               "squared_norm": 0.0, "component_count": P}
    vector = {"delta": copy_primitive(summary), "delta_data": copy_primitive(summary),
              "full_data_agreement": {"difference_norm": 0.0,
                                      "rounding_ceiling": 0.0, "status": "pass"}}
    return {**base, "vector": vector,
            "probes": {probe: _contrast_probe(probe, coefficients) for probe in PROBES}}


def _clone_row(defined: bool) -> dict:
    return {"executed": defined, "start_core_sha256": _sha("clone") if defined else None,
            "start_directly_equal": defined, "storage_disjoint": defined,
            "checked_live_tensor_count": 19 if defined else 0}


def _execution_row(defined: bool, update: int) -> dict:
    keys = ("parameters_sha256", "optimizer_sha256", "assigned_gradient_sha256",
            "gradient_null_mask_sha256", "observer_unchanged_sha256",
            "post_step_null_mask_sha256", "rng_sha256")
    if not defined:
        return {**{key: None for key in keys}, "optimizer_updates_after": None}
    return {**{key: _sha("execution-" + key) for key in keys},
            "optimizer_updates_after": update}


def _branch_audit_metadata(undefined_branch: str | None, role: str,
                           bundle: int, update: int) -> dict:
    defined = {name: name != undefined_branch for name in BRANCHES}
    clone_rows = {name: _clone_row(defined[name]) for name in BRANCHES}
    execution_rows = {name: _execution_row(defined[name], update) for name in BRANCHES}
    match = {"witness_sha256": _sha("witness"), "replay_sha256": _sha("replay"),
             "directly_equal": True}
    checks = {name: copy_primitive(match) for name in
              ("raw_gradient", "current_gradient", "delivered_gradient",
               "parameters_after", "optimizer_after", "observer_after")}
    checks["rng"] = {"anchor_rng_sha256": _sha("anchor-rng"),
                     "witness_rng_sha256": _sha("witness-rng"),
                     "replay_rng_sha256": _sha("replay-rng"),
                     "replay_directly_equals_anchor": True}
    checks["live_loss"] = copy_primitive(match)
    count = sum(defined.values())
    return {
        "producer_bindings": {
            "anchor_ref": _bounded_branch_reference("i7_anchor"),
            "source_witness_ref": _bounded_branch_reference("i7_source_step_witness"),
            "sources_tree_sha256": _sha("sources-tree"),
            "environment_tree_sha256": _sha("environment-tree"),
            "formula_ids": ["i7_response_math_v1", "i7_assembly_roundoff_v1"],
            "tolerance_ids": ["native_exact_v1", "delivery_relative_1e-6_v1",
                              "i7_numerical_contract_v1"],
        },
        "candidate_proof": {"observer_updates_before": update - 1, "observer_ingests": 1,
                            "observer_updates_after": update,
                            "candidate_core_sha256": _sha("candidate"),
                            "candidate_core_after_canonical_sha256": _sha("candidate-canonical"),
                            "candidate_core_after_reverse_sha256": _sha("candidate-reverse"),
                            "candidate_core_directly_unchanged": True},
        "clone_independence_proof": {
            "anchor_core_sha256": _sha("anchor-core"),
            "per_order": {"canonical": copy_primitive(clone_rows),
                          "reverse": copy_primitive(clone_rows)},
            "anchor_core_after_sha256": _sha("anchor-core-after"),
            "anchor_directly_unchanged": True,
        },
        "branch_execution_proof": {
            "canonical_order": list(BRANCHES), "verification_order": list(reversed(BRANCHES)),
            "observer_ingests": 0,
            "per_order": {"canonical": copy_primitive(execution_rows),
                          "reverse": copy_primitive(execution_rows)},
            "per_branch_directly_equal": {name: True for name in BRANCHES},
            "all_defined_order_invariant": True,
        },
        "current_replay_proof": {
            "source_witness_artifact_id": "w" * 128, "checks": checks,
            "overall_status": "exact",
        },
        "completion": {"defined_branch_count": count,
                       "undefined_branch_count": len(BRANCHES) - count,
                       "required_branch_keys_present": True,
                       "required_comparison_keys_present": True,
                       "exact_replay_complete": True,
                       "producer_artifact_status": "complete"},
    }


def _branch_payload(variant: str = "all_defined") -> dict:
    _need(variant in ("all_defined", "restored_undefined", "reciprocal_undefined"),
          "unknown branch recipe variant")
    undefined = None if variant == "all_defined" else variant.removesuffix("_undefined")
    defined = {name: name != undefined for name in BRANCHES}
    branches = {name: (_defined_branch(name, 2_000) if defined[name]
                       else _undefined_branch(name)) for name in BRANCHES}
    pairs = {key: _pair(key, undefined) for key in PAIR_KEYS}
    contrasts = {name: _contrast(name, coefficients, undefined)
                 for name, coefficients in COEFFICIENTS.items()}
    return {"candidate_state": _candidate(undefined), "branches": branches,
            "comparisons": {"branch_pairs": pairs, "contrasts": contrasts},
            "audit_metadata": _branch_audit_metadata(undefined, "sensitivity", 71901, 2_000)}


# ---------------------------------------------------------------------------
# Independent-audit adverse union.  Records below intentionally realize each
# local cost maximum, even where the frozen theorem combines exclusive states.

def _index_record(dimension: int) -> dict:
    full, remainder = divmod(dimension, 8)
    bits = "ff" * full + (f"{(1 << remainder) - 1:02x}" if remainder else "")
    return {"encoding": "flat_bitset_hex_lsb0_v1", "dimension": dimension,
            "count": dimension, "bits_hex": bits}


def _screen(dimension: int) -> dict:
    # ratio and reason together are the conservative, intentionally exclusive union.
    return {"passed": False, "max_absolute_error": 0.0, "max_envelope": 0.0,
            "max_error_envelope_ratio": 0.0,
            "ratio_null_reason": "zero_envelope_mismatch",
            "failing_flat_indices": _index_record(dimension),
            "error_l2": 0.0, "envelope_l2": 0.0}


def _projection_result() -> dict:
    return {"formula_id": "i7_projection_screen_v1", **_screen(P),
            "operator": "basis_outer_product", "rank": 32,
            "identity_bytes_equal": None,
            "identity_mismatch_flat_indices": _index_record(P)}


def _metadata_record(value, scalar: bool) -> dict:
    return {"producer_value": value, "auditor_value": value,
            "fixed_ceiling": 0.0 if scalar else None,
            "achieved_discrepancy": 0.0 if scalar else None,
            "status": "fatal_validation"}


def _delivery_metadata(domain: str) -> dict:
    _need(domain in ("both_positive", "both_zero", "restored_undefined",
                     "reciprocal_undefined"), "unknown audit delivery domain")
    nc_zero = domain in ("both_zero", "reciprocal_undefined")
    nl_zero = domain in ("both_zero", "restored_undefined")
    largest_zero, direction_undefined = nc_zero and nl_zero, nc_zero or nl_zero
    values = {
        "nc": (0.0, True), "nl": (0.0, True),
        "leverage.current_norm": (0.0, True), "leverage.lagged_norm": (0.0, True),
        "leverage.signed_norm_difference": (0.0, True),
        "leverage.relative_norm_separation": (None if largest_zero else 0.0, not largest_zero),
        "leverage.current_lagged_norm_ratio": (None if nl_zero else 0.0, not nl_zero),
        "leverage.ratio_reason": ("zero_lagged_norm" if nl_zero else None, False),
        "leverage.unit_direction_distance": (None if direction_undefined else 0.0,
                                             not direction_undefined),
        "leverage.unit_direction_cosine": (None if direction_undefined else 0.0,
                                           not direction_undefined),
        "leverage.direction_reason": ("zero_norm" if direction_undefined else None, False),
        "leverage.norm_leverage": (False, False),
        "leverage.direction_leverage": (False, False),
    }
    return {key: _metadata_record(value, scalar) for key, (value, scalar) in values.items()}


def _domain_mask(domain: str) -> dict[str, bool]:
    _need(domain in ("both_positive", "both_zero", "restored_undefined",
                     "reciprocal_undefined"), "unknown audit domain")
    undefined = ("restored" if domain == "restored_undefined" else
                 "reciprocal" if domain == "reciprocal_undefined" else None)
    return {branch: branch != undefined for branch in BRANCHES}


def _delivery_branch(branch: str, defined: bool) -> dict:
    if not defined:
        return {"status": "domain_undefined", "reason": UNDEFINED_REASON[branch],
                "target_norm": None, "actual_norm": None, "norm_error": None,
                "direction_error": None, "exact_unscaled": None}
    return {"status": "fatal_validation", "reason": None, "target_norm": 0.0,
            "actual_norm": 0.0, "norm_error": 0.0,
            "direction_error": None if branch == "zero" else 0.0,
            "exact_unscaled": False if branch in ("raw", "current", "lagged") else None}


def _adam_parameter(dimension: int) -> dict:
    return {"formula_id": "i7_adamw_screen_v1", "passed": False,
            "checks": {name: _screen(dimension) for name in ("theta", "moment", "variance")},
            "native_displacement_norm": 0.0,
            "endpoint_envelope_to_displacement": 0.0,
            # Paired non-null with ratio only in the synthetic union.
            "displacement_ratio_null_reason": "zero_native_displacement"}


def _adam_stage(mask: dict[str, bool]) -> dict:
    result = {}
    for branch in BRANCHES:
        if not mask[branch]:
            result[branch] = {"status": "domain_undefined",
                              "reason": UNDEFINED_REASON[branch], "parameters": None}
        else:
            result[branch] = {"status": "fatal_validation", "reason": None,
                              "parameters": {name: _adam_parameter(size)
                                             for name, size in PARAMETERS}}
    return result


def _scalar_record(sign: bool = False) -> dict:
    row = {"producer_value": 0.0, "auditor_value": 0.0, "fixed_ceiling": 0.0,
           "achieved_discrepancy": 0.0, "formula_id": "i7_assembly_roundoff_v1",
           "audit_status": "fatal_validation"}
    if sign:
        row.update({"producer_sign": -1, "auditor_sign": -1,
                    "resolution_status": "unresolved_numerical",
                    "resolution_reason": "insufficient_fixed_margin"})
    return row


def _vector_audit() -> dict:
    return {"max_absolute_error": 0.0, "error_l2": 0.0,
            "max_component_ceiling": 0.0, "max_error_ceiling_ratio": 0.0,
            "failing_flat_indices": _index_record(P), "auditor_sha256": _sha("auditor"),
            "audit_status": "fatal_validation"}


def _failure(path: str, reason: str) -> dict:
    return {"path": path, "reason": reason}


def _measurement_report(mask: dict[str, bool], artifact_id: str) -> tuple[dict, dict, dict]:
    measurements, comparisons, failures = {}, {}, []
    native_paths, unresolved_paths = [], []

    def scalar(path: str, *, comparison: bool = False, sign: bool = False) -> None:
        (comparisons if comparison else measurements)[path] = _scalar_record(sign)
        failures.append(_failure(path, "numerical ceiling exceeded"))

    def vector(path: str) -> None:
        measurements[path] = _vector_audit()
        failures.append(_failure(path, "vector component ceiling exceeded"))

    def concordance(path: str) -> None:
        scalar(path + ".abs_discrepancy")
        scalar(path + ".descriptive_ceiling")
        native_paths.append(path)

    def geometry(path: str) -> None:
        for field in ("left_norm", "right_norm", "distance"):
            scalar(path + "." + field)
        scalar(path + ".cosine", comparison=True)
        unresolved_paths.append(path + ".cosine")

    def scalar_binding(path: str) -> None:
        for suffix in ("direct_route", "component_route", "identity_discrepancy",
                       "identity_ceiling"):
            scalar(path + "." + suffix)
        failures.append(_failure(path, "direct/component identity ceiling exceeded"))

    for probe in PROBES:
        path = "before." + probe
        vector(path + ".q_values"); scalar(path + ".q_norm"); scalar(path + ".CE")
        concordance(path + ".concordance")
        if probe == "auxiliary_clean":
            for chunk in range(10):
                cp = path + f".chunk{chunk}"
                scalar(cp + ".CE"); concordance(cp + ".concordance")

    for branch in BRANCHES:
        if not mask[branch]:
            continue
        path = "branches." + branch
        dp = path + ".displacement"
        vector(dp + ".delta_values"); vector(dp + ".delta_data_values")
        for field in ("delta_norm", "delta_squared_norm", "delta_data_norm",
                      "delta_data_squared_norm"):
            scalar(dp + "." + field)
        geometry(dp + ".delta_delta_data")
        for probe in PROBES:
            pp = path + "." + probe
            scalar(pp + ".after_ce")
            for field in ("Y", "D", "Ddata", "R"):
                scalar(pp + "." + field); scalar(pp + ".stored_identity." + field)
            scalar(pp + ".native_Y")
            concordance(pp + ".concordance.after"); concordance(pp + ".concordance.Y")
            if probe == "auxiliary_clean":
                for chunk in range(10):
                    cp = pp + f".chunks{chunk}"
                    for field in ("after", "Y", "stored_Y", "native_Y"):
                        scalar(cp + "." + field)
                    concordance(cp + ".concordance.after"); concordance(cp + ".concordance.Y")

    for pair in PAIR_KEYS:
        left, right = pair.split("__")
        if not (mask[left] and mask[right]):
            continue
        path = "pairs." + pair
        for field in ("delivered_gradient", "delta", "delta_data"):
            geometry(path + "." + field)
        scalar(path + ".discrepancy_binding"); scalar(path + ".discrepancy_independent")
        scalar(path + ".ceiling")
        failures.append(_failure(path, "full/data pair ceiling exceeded"))

    defined_contrasts = []
    for name, coefficients in COEFFICIENTS.items():
        if not all(mask[branch] for branch in coefficients):
            continue
        defined_contrasts.append(name)
        path = "contrasts." + name
        for field in ("delta", "delta_data"):
            scalar(path + "." + field + ".norm"); scalar(path + "." + field + ".energy")
        scalar(path + ".agreement.discrepancy"); scalar(path + ".agreement.ceiling")
        failures.append(_failure(path, "full/data vector ceiling exceeded"))
        for probe in PROBES:
            pp = path + "." + probe
            for field in ("Y", "nativeY", "D", "Ddata", "R"):
                scalar_binding(pp + "." + field)
            concordance(pp + ".concordance.Y")
            scalar(pp + ".Y.independent", comparison=True, sign=True)
            for field in ("D", "Ddata", "R"):
                scalar(pp + "." + field + ".independent", comparison=True, sign=True)
            if probe == "auxiliary_clean":
                for chunk in range(10):
                    cp = pp + f".chunk{chunk}"
                    scalar_binding(cp + ".Y"); scalar_binding(cp + ".nativeY")
                    concordance(cp + ".concordance.Y")
                    scalar(cp + ".Y.independent", comparison=True, sign=True)

    failures.append(_failure("validation", "v" * PARTIAL_VALIDATION_UTF8))
    defined_count = sum(mask.values())
    counts = {"before_probes": 4, "branch_probes": 4 * defined_count,
              "defined_branches": defined_count, "pairs": 15, "contrasts": 15,
              "defined_contrasts": len(defined_contrasts), "auxiliary_before_chunks": 10,
              "auxiliary_branch_chunks": 10 * defined_count,
              "auxiliary_contrast_chunks": 10 * len(defined_contrasts)}
    undefined = [branch for branch in BRANCHES if not mask[branch]]
    weak = [name for name in defined_contrasts if name == "interaction"
            or name.startswith("direction_") or name.startswith("norm_")]
    report = {"profile": PROFILE, "scope": "raw_value_measurements_only",
              "overall_status": "fatal_validation", "audit_complete": False,
              "completion_counts": counts, "fatal_failures": failures,
              "measurement_audits": measurements,
              "domain_undefined_branches": undefined,
              "weak_factor_contrasts": weak,
              "native_discordant_paths": native_paths,
              "unresolved_geometry_paths": unresolved_paths,
              "binding_artifact_id": artifact_id,
              "scope_exclusions": list(MEASUREMENT_EXCLUSIONS)}
    topology = {"measurement_paths": len(measurements),
                "comparison_paths": len(comparisons),
                "failure_rows": len(failures), "native_paths": len(native_paths),
                "unresolved_paths": len(unresolved_paths)}
    return report, comparisons, topology


def _stage_row(result, mode: str = "completed_failure") -> dict:
    _need(mode in ("completed_pass", "completed_failure", "not_run", "error",
                   "resource_abort", "partial_failure"), "unknown audit stage mode")
    if mode in ("completed_pass", "completed_failure", "partial_failure"):
        return {"status": "pass" if mode == "completed_pass" else "fatal_validation",
                "completed": mode != "partial_failure", "error_code": None,
                "result": copy_primitive(result)}
    code = ("ResourceGuardAbort" if mode == "resource_abort" else
            "ArtifactEnvelopeError" if mode == "error" else None)
    return {"status": "not_run" if mode == "not_run" else "fatal_validation",
            "completed": False, "error_code": code, "result": None}


def _completion(mask: dict[str, bool]) -> dict:
    defined = sum(mask.values())
    contrasts = sum(all(mask[branch] for branch in coefficients)
                    for coefficients in COEFFICIENTS.values())
    counts = {"before_probes": 4, "branch_probes": 4 * defined,
              "defined_branches": defined, "pairs": 15, "contrasts": 15,
              "defined_contrasts": contrasts, "auxiliary_before_chunks": 10,
              "auxiliary_branch_chunks": 10 * defined,
              "auxiliary_contrast_chunks": 10 * contrasts}
    return {"required_stages": list(CHECK_KEYS), "completed_stages": list(CHECK_KEYS),
            "checks_complete": False, "overall_status": "fatal_validation",
            "expected_projection_checks": 2, "projection_checks": 2,
            "expected_delivery_rows": 6, "delivery_rows": 6,
            "expected_adam_parameter_checks": 4 * defined,
            "adam_parameter_checks": 4 * defined, "adam_tensor_screens": 12 * defined,
            "expected_measurement_counts": copy_primitive(counts),
            "measurement_counts": copy_primitive(counts),
            "scientific_execution_certified": False}


def _plan_reference(bundle: int) -> dict:
    name = f"i7-recipe-b{bundle}-plan.pt"
    return {"name": name, "status": "complete", "encoding": "torch_weights_only",
            "size_bytes": 4 << 20, "sha256": _sha(name),
            "receipt_name": "receipt-" + "0" * 64 + ".json",
            "receipt_size_bytes": 4_096, "receipt_sha256": _sha("receipt-" + name)}


def _audit_payload(environment: dict, domain: str = "both_positive") -> tuple[dict, dict]:
    # The global delivery maximum is restored-undefined even when the maximal
    # report topology is all-defined.  This mismatch is intentional and named.
    mask = _domain_mask(domain)
    role, bundle, update = "sensitivity", 71901, 2_000
    branch_id = f"{PROFILE}--{role}--b{bundle}--u{update}--branch-results"
    measurement, comparisons, topology = _measurement_report(mask, branch_id)
    delivery_mask = _domain_mask("restored_undefined")
    delivery = {"candidate_metadata": _delivery_metadata("restored_undefined"),
                "branches": {branch: _delivery_branch(branch, delivery_mask[branch])
                             for branch in BRANCHES}}
    saved = {"kind": "saved_value_consistency_not_history"}
    payload = {
        "input_bindings": {
            "anchor_ref": _reference(role, bundle, update, "anchor", "i7_anchor"),
            "source_witness_ref": _reference(role, bundle, update, "source-witness",
                                             "i7_source_step_witness"),
            "branch_results_ref": _reference(role, bundle, update, "branch-results",
                                             "i7_branch_results"),
            "plan_ref": _plan_reference(bundle),
            "sources_tree_sha256": _sha("audit-sources"),
            "plan_probe_data_tree_sha256": _sha("audit-plan-probe-data"),
        },
        "exact_validation": {
            "auditor_environment": copy_primitive(environment),
            "saved_structure": _stage_row(saved), "completion": _completion(mask),
            "scope_exclusions": list(EXCLUSIONS),
        },
        "candidate_operator_audit": {
            "scope": "represented_operator_only",
            "projections": _stage_row({"current": _projection_result(),
                                       "lagged": _projection_result()}),
            "delivery": _stage_row(delivery),
        },
        "branch_audits": _stage_row(_adam_stage(mask)),
        "measurement_audits": _stage_row(measurement, "partial_failure"),
        "comparison_audits": comparisons,
        "overall_status": "fatal_validation",
        "fatal_failures": [{"stage": "saved_structure", "completed": False,
                            "error_code": "ArtifactEnvelopeError"}],
    }
    return payload, topology


def _wrap(payload: dict, component: str) -> dict:
    return envelope(payload, role="sensitivity", bundle=71901, update=2_000,
                    kind="branch-results" if component == "branch_results"
                    else "independent-audit")


def build_template(component, *, sources, environment) -> dict:
    """Build one full symbolic maximum recipe without scientific execution.

    The supplied trees are validated and bounded.  Only ``auditor_environment``
    is retained by either original schema; source trees are represented there
    solely by digest fields and are therefore not added to these templates.
    """
    _need(type(component) is str and component in COMPONENTS,
          "unknown branch/audit recipe component")
    _sources, _source_bytes = _bounded_json_copy(sources, SOURCE_JSON_MAX, "sources")
    environment_copy, _environment_bytes = _bounded_json_copy(
        environment, ENVIRONMENT_JSON_MAX, "environment")
    payload = (_branch_payload("all_defined") if component == "branch_results"
               else _audit_payload(environment_copy, "both_positive")[0])
    return _wrap(payload, component)


def main(argv=None) -> int:
    if argv not in (None, []):
        raise SystemExit("native_storage_recipe_branch_audit has no execution mode")
    print("I7 symbolic branch/audit storage recipes only; no tensors or execution.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
