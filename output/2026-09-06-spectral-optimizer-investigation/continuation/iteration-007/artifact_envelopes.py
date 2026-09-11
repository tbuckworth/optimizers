#!/usr/bin/env python3
"""Strict saved-value envelope for the iteration-007 six-branch result.

Public API:

``make_branch_results(...)`` constructs one owned CPU envelope after validating
the caller's original trees. ``validate_branch_results(...)`` validates an
existing envelope against independently supplied candidate and measurement
snapshots. ``expected_audit_metadata(...)`` derives the frozen producer
attestation after the producer has already performed the live checks; it is not
proof that those checks ran. These functions execute no model or optimizer
operation and perform no filesystem I/O or CUDA initialization. Supplied IDX
bytes are validated and materialized by the existing anchor validators.
"""
from __future__ import annotations

import hashlib
import math
import struct
from typing import Any

import torch

import anchor_envelope as envelopes
import artifact_store as storage
import identity_codec as codec
import response_math as response
import source_capture as source
import state_core as state


class ArtifactEnvelopeError(ValueError):
    """A branch-result value or one of its expected bindings is malformed."""


BRANCHES = tuple(response.BRANCHES)
PAYLOAD_KEYS = ("candidate_state", "branches", "comparisons", "audit_metadata")
CANDIDATE_KEYS = (
    "g", "c", "l", "previous_basis", "post_ingest_basis", "nc", "nl",
    "observer_after", "measurement_before", "leverage", "zero_cases",
)
ZERO_KEYS = ("current_zero", "lagged_zero", "branch_defined_mask", "branch_reasons")
BRANCH_KEYS = (
    "status", "reason", "delivered_gradient", "assigned_gradient_null_mask",
    "parameters_after", "parameters_after_flat", "optimizer_after", "measurement",
)
AUDIT_KEYS = (
    "producer_bindings", "candidate_proof", "clone_independence_proof",
    "branch_execution_proof", "current_replay_proof", "completion",
)
PRODUCER_KEYS = (
    "anchor_ref", "source_witness_ref", "sources_tree_sha256",
    "environment_tree_sha256", "formula_ids", "tolerance_ids",
)
CANDIDATE_PROOF_KEYS = (
    "observer_updates_before", "observer_ingests", "observer_updates_after",
    "candidate_core_sha256", "candidate_core_after_canonical_sha256",
    "candidate_core_after_reverse_sha256", "candidate_core_directly_unchanged",
)
CLONE_PROOF_KEYS = (
    "anchor_core_sha256", "per_order", "anchor_core_after_sha256",
    "anchor_directly_unchanged",
)
CLONE_ROW_KEYS = (
    "executed", "start_core_sha256", "start_directly_equal", "storage_disjoint",
    "checked_live_tensor_count",
)
EXECUTION_PROOF_KEYS = (
    "canonical_order", "verification_order", "observer_ingests", "per_order",
    "per_branch_directly_equal", "all_defined_order_invariant",
)
EXECUTION_ROW_KEYS = (
    "parameters_sha256", "optimizer_sha256", "assigned_gradient_sha256",
    "gradient_null_mask_sha256", "observer_unchanged_sha256",
    "post_step_null_mask_sha256", "rng_sha256", "optimizer_updates_after",
)
REPLAY_PROOF_KEYS = ("source_witness_artifact_id", "checks", "overall_status")
REPLAY_CHECKS = (
    "raw_gradient", "current_gradient", "delivered_gradient", "parameters_after",
    "optimizer_after", "observer_after", "rng", "live_loss",
)
MATCH_KEYS = ("witness_sha256", "replay_sha256", "directly_equal")
RNG_MATCH_KEYS = (
    "anchor_rng_sha256", "witness_rng_sha256", "replay_rng_sha256",
    "replay_directly_equals_anchor",
)
COMPLETION_KEYS = (
    "defined_branch_count", "undefined_branch_count", "required_branch_keys_present",
    "required_comparison_keys_present", "exact_replay_complete",
    "producer_artifact_status",
)
RECEIPT_KEYS = source.RECEIPT_KEYS
REFERENCE_KEYS = source.REFERENCE_KEYS
EXPECTED_CANDIDATE_KEYS = ("branches", "leverage", "delivered_pairs")
EXPECTED_CANDIDATE_ROW_KEYS = (
    "status", "reason", "gradient", "target_norm", "actual_norm", "norm_error",
    "direction_error",
)
EXPECTED_MEASUREMENT_KEYS = ("measurement_before", "branches", "comparisons")
PROBES = ("batch_noisy", "train_probe_noisy", "train_probe_clean", "auxiliary_clean")
LEVERAGE_KEYS = (
    "current_norm", "lagged_norm", "signed_norm_difference",
    "relative_norm_separation", "current_lagged_norm_ratio", "ratio_reason",
    "unit_direction_distance", "unit_direction_cosine", "direction_reason",
    "norm_leverage", "direction_leverage",
)
PAIR_KEYS = tuple(
    f"{left}__{right}" for index, left in enumerate(BRANCHES)
    for right in BRANCHES[index + 1:]
)
FORMULA_IDS = ["i7_response_math_v1", "i7_assembly_roundoff_v1"]
TOLERANCE_IDS = [
    "native_exact_v1", "delivery_relative_1e-6_v1", "i7_numerical_contract_v1",
]
CONTEXT_KEYS = (
    "identity", "profile", "plan", "plan_artifact", "images_bytes", "labels_bytes",
    "expected_files", "sources", "environment",
)
PROFILES = (storage.SCIENTIFIC, storage.MLP_FIXTURE)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ArtifactEnvelopeError(message)


def _keys(value: Any, expected: tuple[Any, ...], where: str) -> None:
    _require(type(value) is dict and len(value) == len(expected),
             where + ": exact ordered keys required")
    actual = tuple(value.keys())
    _require(all(type(got) is type(want) and got == want
                 for got, want in zip(actual, expected)),
             where + ": exact ordered keys required")


def _context(context: dict[str, Any]) -> tuple[dict[str, Any], str]:
    _require(type(context) is dict and set(context) == set(CONTEXT_KEYS),
             "exact anchor-envelope context required")
    identity, profile = context["identity"], context["profile"]
    _require(type(profile) is str and profile in PROFILES, "unsupported branch profile")
    codec.validate_identity(identity, profile=profile)
    return identity, profile


def _digest(value: Any) -> str:
    return codec.tree_digest(value)


def _receipt_reference(artifact: dict[str, Any], receipt: dict[str, Any]) -> dict[str, Any]:
    _keys(receipt, RECEIPT_KEYS, "artifact receipt")
    expected_name = artifact["artifact_id"] + ".pt"
    _require(type(receipt["schema"]) is str and
             receipt["schema"] == "i7_artifact_receipt_v1" and
             type(receipt["name"]) is str and receipt["name"] == expected_name and
             type(receipt["size"]) is int and receipt["size"] > 0 and
             type(receipt["sha256"]) is str and len(receipt["sha256"]) == 64 and
             all(character in "0123456789abcdef" for character in receipt["sha256"]) and
             type(receipt["status"]) is str and receipt["status"] == "complete" and
             type(receipt["encoding"]) is str and receipt["encoding"] == "torch_weights_only" and
             type(receipt["receipt_name"]) is str,
             "artifact receipt is not canonical and complete")
    receipt_name = "receipt-" + hashlib.sha256(expected_name.encode("ascii")).hexdigest() + ".json"
    _require(receipt["receipt_name"] == receipt_name, "artifact receipt name differs")
    encoded = storage._json_bytes({key: receipt[key] for key in RECEIPT_KEYS[:-1]})
    return {
        "artifact_id": artifact["artifact_id"], "schema_name": artifact["schema_name"],
        "name": expected_name, "size_bytes": receipt["size"], "sha256": receipt["sha256"],
        "status": "complete", "encoding": "torch_weights_only",
        "receipt_name": receipt_name, "receipt_size_bytes": len(encoded),
        "receipt_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _native(record: Any, *, profile: str, where: str,
            shape: tuple[int, ...] | None = None, dtype: str = "torch.float32") -> None:
    spec = state.PROFILE[profile]
    try:
        state._validate_native(record, where, shape, dtype, spec["device_kind"])
    except Exception as exc:
        raise ArtifactEnvelopeError(where + " invalid: " + type(exc).__name__) from None


def _same_tensor_value(record: dict[str, Any], tensor: torch.Tensor, where: str) -> None:
    _require(type(tensor) is torch.Tensor and tensor.device.type == "cpu" and
             tensor.dtype == torch.float32 and tensor.ndim == 1 and
             not tensor.requires_grad, where + ": expected candidate tensor invalid")
    _require(envelopes.same_exact(record["value"], tensor), where + ": tensor bytes differ")


def _finite_float(value: Any, where: str, *, nonnegative: bool = False) -> float:
    _require(type(value) is float and math.isfinite(value) and
             (not nonnegative or value >= 0.0), where + ": invalid finite float")
    return value


def _validate_expected_candidates(value: Any, profile: str) -> None:
    _keys(value, EXPECTED_CANDIDATE_KEYS, "expected candidates")
    _keys(value["branches"], BRANCHES, "expected candidate branches")
    n = state.PROFILE[profile]["n_params"]
    for branch, row in value["branches"].items():
        _keys(row, EXPECTED_CANDIDATE_ROW_KEYS, "expected candidate " + branch)
        _require(type(row["status"]) is str and row["status"] in ("defined", "undefined"),
                 "expected candidate status invalid")
        if row["status"] == "undefined":
            expected_reason = ("positive_current_norm_zero_lagged_direction" if branch == "restored"
                               else "positive_lagged_norm_zero_current_direction" if branch == "reciprocal"
                               else None)
            _require(expected_reason is not None and row["reason"] == expected_reason and
                     all(row[key] is None for key in
                         ("gradient", "actual_norm", "norm_error", "direction_error")),
                     "expected candidate undefined row invalid")
            _finite_float(row["target_norm"], "undefined target norm", nonnegative=True)
        else:
            _require(row["reason"] is None and type(row["gradient"]) is torch.Tensor and
                     row["gradient"].shape == (n,) and row["gradient"].dtype == torch.float32 and
                     row["gradient"].device.type == "cpu" and not row["gradient"].requires_grad,
                     "expected candidate defined row invalid")
            for key in ("target_norm", "actual_norm", "norm_error"):
                _finite_float(row[key], "expected candidate " + key, nonnegative=True)
            _require(row["direction_error"] is None or
                     (type(row["direction_error"]) is float and
                      math.isfinite(row["direction_error"]) and row["direction_error"] >= 0.0),
                     "expected direction error invalid")
    _keys(value["leverage"], LEVERAGE_KEYS, "expected leverage")
    leverage = value["leverage"]
    for key in ("current_norm", "lagged_norm"):
        _finite_float(leverage[key], "expected leverage " + key, nonnegative=True)
    _finite_float(leverage["signed_norm_difference"], "expected signed norm difference")
    for key in ("relative_norm_separation", "current_lagged_norm_ratio",
                "unit_direction_distance"):
        _require(leverage[key] is None or
                 (type(leverage[key]) is float and math.isfinite(leverage[key]) and
                  leverage[key] >= 0.0), "expected leverage " + key + " invalid")
    _require(leverage["unit_direction_cosine"] is None or
             (type(leverage["unit_direction_cosine"]) is float and
              math.isfinite(leverage["unit_direction_cosine"])),
             "expected leverage cosine invalid")
    _require(leverage["ratio_reason"] in (None, "zero_lagged_norm") and
             leverage["direction_reason"] in (None, "zero_norm") and
             type(leverage["norm_leverage"]) is bool and
             type(leverage["direction_leverage"]) is bool,
             "expected leverage reason/flag invalid")
    nc, nl = leverage["current_norm"], leverage["lagged_norm"]
    largest = max(nc, nl)
    expected_separation = None if largest == 0.0 else abs(nc - nl) / largest
    expected_ratio = None if nl == 0.0 else nc / nl
    _require(envelopes.same_exact(leverage["signed_norm_difference"], nc - nl) and
             envelopes.same_exact(leverage["relative_norm_separation"], expected_separation) and
             envelopes.same_exact(leverage["current_lagged_norm_ratio"], expected_ratio) and
             leverage["ratio_reason"] == ("zero_lagged_norm" if nl == 0.0 else None) and
             leverage["direction_reason"] ==
                ("zero_norm" if nc == 0.0 or nl == 0.0 else None) and
             leverage["norm_leverage"] is
                (expected_separation is not None and expected_separation > 2 * response.TOL) and
             leverage["direction_leverage"] is
                (leverage["unit_direction_distance"] is not None and
                 leverage["unit_direction_distance"] > 2 * response.TOL),
             "expected leverage scalars/reasons/flags are inconsistent")
    _require((leverage["unit_direction_distance"] is None) == (nc == 0.0 or nl == 0.0) and
             (leverage["unit_direction_cosine"] is None) == (nc == 0.0 or nl == 0.0),
             "expected direction geometry nullability differs from declared norms")
    expected_targets = {"current": nc, "lagged": nl, "restored": nc,
                        "reciprocal": nl, "zero": 0.0}
    for branch, target in expected_targets.items():
        _require(envelopes.same_exact(value["branches"][branch]["target_norm"], target),
                 "expected candidate target norm differs for " + branch)
    for branch, row in value["branches"].items():
        if row["status"] == "undefined":
            _require(row["target_norm"] > 0.0,
                     "undefined candidate requires a positive target norm")
            continue
        target, actual, error = row["target_norm"], row["actual_norm"], row["norm_error"]
        if target == 0.0:
            _require(actual == 0.0 and error == 0.0 and row["direction_error"] is None and
                     bool(torch.count_nonzero(row["gradient"]).item() == 0),
                     "zero-target candidate scalar/vector consistency differs")
        else:
            derived_error = abs(actual - target) / target
            _require(actual > 0.0 and envelopes.same_exact(error, derived_error) and
                     error <= response.TOL and type(row["direction_error"]) is float and
                     row["direction_error"] <= response.TOL and
                     bool(torch.count_nonzero(row["gradient"]).item() > 0),
                     "positive-target candidate scalar consistency differs")
        if branch in ("raw", "current", "lagged"):
            _require(envelopes.same_exact(actual, target) and error == 0.0 and
                     row["direction_error"] == (None if target == 0.0 else 0.0),
                     "unscaled candidate scalar consistency differs")
    _keys(value["delivered_pairs"], PAIR_KEYS, "expected delivered pairs")
    for key, pair in value["delivered_pairs"].items():
        _keys(pair, ("distance", "cosine", "cosine_reason"), "expected pair " + key)
        _require(pair["distance"] is None or
                 (type(pair["distance"]) is float and math.isfinite(pair["distance"]) and
                  pair["distance"] >= 0.0), "expected pair distance invalid")
        _require(pair["cosine"] is None or
                 (type(pair["cosine"]) is float and math.isfinite(pair["cosine"])),
                 "expected pair cosine invalid")
        _require(pair["cosine_reason"] in (None, "zero_norm", "undefined_branch"),
                 "expected pair reason invalid")


def _expected_domain(expected: dict[str, Any]) -> tuple[dict[str, bool], dict[str, Any]]:
    mask = {branch: expected["branches"][branch]["status"] == "defined" for branch in BRANCHES}
    reasons = {branch: expected["branches"][branch]["reason"] for branch in BRANCHES}
    nc, nl = expected["leverage"]["current_norm"], expected["leverage"]["lagged_norm"]
    derived = {name: True for name in BRANCHES}
    derived["restored"] = not (nc > 0.0 and nl == 0.0)
    derived["reciprocal"] = not (nl > 0.0 and nc == 0.0)
    derived_reasons = {name: None for name in BRANCHES}
    if not derived["restored"]:
        derived_reasons["restored"] = "positive_current_norm_zero_lagged_direction"
    if not derived["reciprocal"]:
        derived_reasons["reciprocal"] = "positive_lagged_norm_zero_current_direction"
    _require(mask == derived and reasons == derived_reasons,
             "expected candidate domain differs from retained norm rules")
    return mask, reasons


def _candidate_projection(candidate: dict[str, Any]) -> dict[str, Any]:
    return {key: candidate[key] for key in CANDIDATE_KEYS if key != "measurement_before"}


def _parameter_flat(parameters: list[dict[str, Any]]) -> torch.Tensor:
    return torch.cat([entry["value"].reshape(-1) for entry in parameters]).contiguous()


def _sha_f64(value: torch.Tensor) -> str:
    array = value.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.astype("<f8", copy=False).tobytes(order="C")).hexdigest()


def _sha_scalar(value: float) -> str:
    return hashlib.sha256(struct.pack("<d", value)).hexdigest()


def _validate_vector_record(record: Any, expected: torch.Tensor, where: str) -> None:
    _keys(record, ("value", "shape", "dtype", "device", "sha256"), where)
    tensor = record["value"]
    _require(type(tensor) is torch.Tensor and tensor.device.type == "cpu" and
             tensor.dtype == torch.float64 and tensor.shape == expected.shape and
             not tensor.requires_grad and bool(torch.isfinite(tensor).all()),
             where + ": value invalid")
    _require(record["shape"] == [expected.numel()] and
             type(record["dtype"]) is str and record["dtype"] == "float64" and
             type(record["device"]) is str and record["device"] == "cpu" and
             type(record["sha256"]) is str and record["sha256"] == _sha_f64(tensor) and
             envelopes.same_exact(tensor, expected), where + ": metadata or bytes differ")


def _validate_before_measurement(value: Any, *, profile: str) -> None:
    _keys(value, ("probe_order", "probes"), "measurement_before")
    _require(value["probe_order"] == list(PROBES), "before probe order differs")
    _keys(value["probes"], PROBES, "before probes")
    counts = (64, 256, 256, 5000) if profile == storage.SCIENTIFIC else (4, 4, 4, 10)
    n = state.PROFILE[profile]["n_params"]
    for probe, count in zip(PROBES, counts):
        row = value["probes"][probe]
        _keys(row, ("sample_count", "sample_identity_sha256", "cpu64", "native32",
                    "concordance", "auxiliary_chunks"), "before probe " + probe)
        _require(type(row["sample_count"]) is int and row["sample_count"] == count and
                 type(row["sample_identity_sha256"]) is str and
                 len(row["sample_identity_sha256"]) == 64 and
                 all(ch in "0123456789abcdef" for ch in row["sample_identity_sha256"]),
                 "before probe identity/count invalid")
        _keys(row["cpu64"], ("before_ce", "q", "q_norm"), "before CPU64 " + probe)
        q = row["cpu64"]["q"]
        _keys(q, ("value", "shape", "dtype", "device", "sha256"), "before q " + probe)
        _validate_vector_record(q, q["value"], "before q " + probe)
        _require(q["value"].shape == (n,), "before q shape differs")
        _finite_float(row["cpu64"]["before_ce"], "before CPU64 CE")
        _finite_float(row["cpu64"]["q_norm"], "before q norm", nonnegative=True)
        _require(envelopes.same_exact(row["cpu64"]["q_norm"],
                                      float(torch.linalg.vector_norm(q["value"]).item())),
                 "before q norm differs from retained q")
        _keys(row["native32"], ("before_ce",), "before native32 " + probe)
        _finite_float(row["native32"]["before_ce"], "before native32 CE")
        _require(type(row["auxiliary_chunks"]) is list and
                 len(row["auxiliary_chunks"]) == (10 if probe == "auxiliary_clean" else 0),
                 "before auxiliary chunk count differs")


def _validate_measurement_structure(measurements: dict[str, Any], *, branches: dict[str, Any],
                                    anchor: dict[str, Any], profile: str,
                                    artifact_id: str) -> None:
    _validate_before_measurement(measurements["measurement_before"], profile=profile)
    defined = {name: branches[name]["status"] == "defined" for name in BRANCHES}
    before_parameters = anchor["payload"]["model"]["parameters"]
    before_flat = _parameter_flat(before_parameters).to(dtype=torch.float64)
    before_probes = measurements["measurement_before"]["probes"]
    for branch in BRANCHES:
        measurement = measurements["branches"][branch]
        if not defined[branch]:
            _require(measurement is None, "undefined branch measurement must be null")
            continue
        _keys(measurement, ("displacement", "probes"), "measurement branch " + branch)
        displacement = measurement["displacement"]
        _keys(displacement, ("delta", "delta_data", "delta_norm", "delta_squared_norm",
                             "delta_data_norm", "delta_data_squared_norm", "delta_delta_data"),
              "displacement " + branch)
        after_flat = _parameter_flat(branches[branch]["parameters_after"]).to(dtype=torch.float64)
        delta = (after_flat - before_flat).contiguous()
        delta_data = (delta + (before_flat * (0.001 * 0.01)).contiguous()).contiguous()
        _validate_vector_record(displacement["delta"], delta, "delta " + branch)
        _validate_vector_record(displacement["delta_data"], delta_data, "delta_data " + branch)
        expected_scalars = {
            "delta_norm": float(torch.linalg.vector_norm(delta).item()),
            "delta_squared_norm": float(torch.dot(delta, delta).item()),
            "delta_data_norm": float(torch.linalg.vector_norm(delta_data).item()),
            "delta_data_squared_norm": float(torch.dot(delta_data, delta_data).item()),
        }
        for key, expected in expected_scalars.items():
            _require(type(displacement[key]) is float and
                     envelopes.same_exact(displacement[key], expected),
                     "displacement scalar differs for " + branch + "." + key)
        _keys(measurement["probes"], PROBES, "branch probes " + branch)
        for probe in PROBES:
            probe_row = measurement["probes"][probe]
            _keys(probe_row, ("before_binding", "cpu64", "native32", "concordance",
                              "auxiliary_chunks"), "branch probe " + branch + "." + probe)
            binding = probe_row["before_binding"]
            _keys(binding, ("measurement_before_artifact_id", "probe_key",
                            "before_ce_cpu64_sha256", "q_sha256"),
                  "before binding " + branch + "." + probe)
            before = before_probes[probe]
            expected_binding = {
                "measurement_before_artifact_id": artifact_id, "probe_key": probe,
                "before_ce_cpu64_sha256": _sha_scalar(before["cpu64"]["before_ce"]),
                "q_sha256": before["cpu64"]["q"]["sha256"],
            }
            _require(envelopes.same_exact(binding, expected_binding),
                     "branch before binding differs for " + branch + "." + probe)

    comparisons = measurements["comparisons"]
    _keys(comparisons, ("branch_pairs", "contrasts"), "comparisons")
    _keys(comparisons["branch_pairs"], PAIR_KEYS, "branch pairs")
    for key, row in comparisons["branch_pairs"].items():
        left, right = key.split("__")
        mask = {left: defined[left], right: defined[right]}
        _keys(row, ("defined", "defined_mask", "reason", "delivered_gradient", "delta",
                    "delta_data", "full_data_distance_discrepancy",
                    "full_data_rounding_ceiling", "rounding_status"), "branch pair " + key)
        pair_defined = all(mask.values())
        _require(type(row["defined"]) is bool and row["defined"] is pair_defined and
                 envelopes.same_exact(row["defined_mask"], mask) and
                 row["reason"] == (None if pair_defined else "domain_undefined_required_branch"),
                 "branch pair domain propagation differs for " + key)
        for geometry_key in ("delivered_gradient", "delta", "delta_data"):
            geometry = row[geometry_key]
            _require(type(geometry) is dict and geometry.get("defined") is pair_defined and
                     envelopes.same_exact(geometry.get("defined_mask"), mask),
                     "branch pair geometry mask differs for " + key)

    contrast_keys = tuple(response.COEFFICIENTS.keys())
    _keys(comparisons["contrasts"], contrast_keys, "contrasts")
    full_mask = {name: defined[name] for name in BRANCHES}
    for name, coefficients in response.COEFFICIENTS.items():
        row = comparisons["contrasts"][name]
        _keys(row, ("coefficients", "branch_defined_mask", "defined", "reason",
                    "factor_requirements", "factor_leverage", "vector", "probes"),
              "contrast " + name)
        contrast_defined = all(defined[branch] for branch in coefficients)
        _require(envelopes.same_exact(row["coefficients"], coefficients) and
                 envelopes.same_exact(row["branch_defined_mask"], full_mask) and
                 type(row["defined"]) is bool and row["defined"] is contrast_defined and
                 row["reason"] == (None if contrast_defined
                                    else "domain_undefined_required_branch"),
                 "contrast domain/coefficient propagation differs for " + name)
        if contrast_defined:
            _require(type(row["vector"]) is dict and type(row["probes"]) is dict,
                     "defined contrast result missing for " + name)
            _keys(row["probes"], PROBES, "contrast probes " + name)
        else:
            _require(row["vector"] is None and row["probes"] is None,
                     "undefined contrast retains results for " + name)


def _validate_candidate(candidate: Any, *, anchor: dict[str, Any], witness: dict[str, Any],
                        expected: dict[str, Any], measurements: dict[str, Any],
                        profile: str) -> None:
    _keys(candidate, CANDIDATE_KEYS, "candidate_state")
    n = state.PROFILE[profile]["n_params"]
    native_device = anchor["payload"]["model"]["parameters"][0]["native_device"]
    for key in ("g", "c", "l"):
        _native(candidate[key], profile=profile, where="candidate_state." + key, shape=(n,))
        _require(candidate[key]["native_device"] == native_device,
                 "candidate gradient device differs from anchor")
    _same_tensor_value(candidate["g"], expected["branches"]["raw"]["gradient"], "candidate g")
    _same_tensor_value(candidate["c"], expected["branches"]["current"]["gradient"], "candidate c")
    _same_tensor_value(candidate["l"], expected["branches"]["lagged"]["gradient"], "candidate l")
    _require(envelopes.same_exact(candidate["g"], witness["payload"]["raw_gradient"]),
             "candidate raw gradient differs from witness")
    _require(envelopes.same_exact(candidate["c"], witness["payload"]["delivered_current_gradient"]),
             "candidate current gradient differs from witness")

    previous_v = anchor["payload"]["observer"]["state"]["V"]
    post_v = witness["payload"]["observer_after"]["state"]["V"]
    for key, expected_basis in (("previous_basis", previous_v), ("post_ingest_basis", post_v)):
        if expected_basis is None:
            _require(candidate[key] is None, key + " must retain identity fallback")
        else:
            _native(candidate[key], profile=profile, where="candidate_state." + key,
                    dtype="torch.float32")
            _require(envelopes.same_exact(candidate[key], expected_basis), key + " differs")

    nc = _finite_float(candidate["nc"], "candidate nc", nonnegative=True)
    nl = _finite_float(candidate["nl"], "candidate nl", nonnegative=True)
    _require(envelopes.same_exact(nc, expected["leverage"]["current_norm"]) and
             envelopes.same_exact(nl, expected["leverage"]["lagged_norm"]),
             "candidate norms differ from expected native construction")
    _require((nc == 0.0) == bool(torch.count_nonzero(candidate["c"]["value"]).item() == 0) and
             (nl == 0.0) == bool(torch.count_nonzero(candidate["l"]["value"]).item() == 0),
             "declared zero norm differs from retained vector")
    try:
        state._validate_observer(candidate["observer_after"], profile,
                                 anchor["identity"]["anchor_update"], native_device)
    except Exception as exc:
        raise ArtifactEnvelopeError("candidate observer invalid: " + type(exc).__name__) from None
    _require(envelopes.same_exact(candidate["observer_after"], witness["payload"]["observer_after"]),
             "candidate observer differs from witness")
    _require(envelopes.same_exact(candidate["measurement_before"], measurements["measurement_before"]),
             "candidate before measurement differs")
    _require(envelopes.same_exact(candidate["leverage"], expected["leverage"]),
             "candidate leverage differs")
    mask, reasons = _expected_domain(expected)
    _keys(candidate["zero_cases"], ZERO_KEYS, "candidate zero_cases")
    expected_zero = {
        "current_zero": nc == 0.0, "lagged_zero": nl == 0.0,
        "branch_defined_mask": mask, "branch_reasons": reasons,
    }
    _require(envelopes.same_exact(candidate["zero_cases"], expected_zero),
             "candidate zero/domain metadata differs")


def _validate_parameter_entries(parameters: Any, *, anchor: dict[str, Any], profile: str,
                                where: str) -> None:
    expected = anchor["payload"]["model"]["parameters"]
    _require(type(parameters) is list and len(parameters) == len(expected),
             where + ": parameter list invalid")
    for index, (entry, before) in enumerate(zip(parameters, expected)):
        _keys(entry, state.PARAMETER_ENTRY_KEYS, f"{where}[{index}]")
        for key in state.PARAMETER_ENTRY_KEYS[:-1]:
            _require(envelopes.same_exact(entry[key], before[key]),
                     f"{where}[{index}].{key} differs from anchor")
        tensor = entry["value"]
        _require(type(tensor) is torch.Tensor and tensor.device.type == "cpu" and
                 tensor.dtype == torch.float32 and tuple(tensor.shape) == tuple(entry["shape"]) and
                 not tensor.requires_grad and bool(torch.isfinite(tensor).all()),
                 f"{where}[{index}].value invalid")
    model = state.clone_tree(anchor["payload"]["model"])
    model["parameters"] = state.clone_tree(parameters)
    try:
        state._validate_model(model, profile)
    except Exception as exc:
        raise ArtifactEnvelopeError(where + " invalid: " + type(exc).__name__) from None


def _validate_branches(branches: Any, *, anchor: dict[str, Any], witness: dict[str, Any],
                       expected: dict[str, Any], measurements: dict[str, Any],
                       profile: str) -> None:
    _keys(branches, BRANCHES, "branches")
    n_params = len(state.PROFILE[profile]["parameters"])
    n = state.PROFILE[profile]["n_params"]
    update = anchor["identity"]["anchor_update"]
    native_device = anchor["payload"]["model"]["parameters"][0]["native_device"]
    for branch, row in branches.items():
        _keys(row, BRANCH_KEYS, "branch " + branch)
        expected_row = expected["branches"][branch]
        defined = expected_row["status"] == "defined"
        _require(type(row["status"]) is str and row["status"] == expected_row["status"] and
                 (row["reason"] is None or type(row["reason"]) is str) and
                 row["reason"] == expected_row["reason"], "branch status/reason differs")
        if not defined:
            _require(all(row[key] is None for key in BRANCH_KEYS[2:]),
                     "undefined branch retains result fields")
            _require(measurements["branches"][branch] is None,
                     "undefined branch has expected measurement")
            continue
        _native(row["delivered_gradient"], profile=profile,
                where="branch " + branch + " delivered gradient", shape=(n,))
        _require(row["delivered_gradient"]["native_device"] == native_device,
                 "branch delivered device differs")
        _same_tensor_value(row["delivered_gradient"], expected_row["gradient"],
                           "branch " + branch + " delivered gradient")
        _require(type(row["assigned_gradient_null_mask"]) is list and
                 len(row["assigned_gradient_null_mask"]) == n_params and
                 all(type(item) is bool and item is False
                     for item in row["assigned_gradient_null_mask"]),
                 "assigned gradient null mask must be all false")
        _validate_parameter_entries(row["parameters_after"], anchor=anchor, profile=profile,
                                    where="branch " + branch + " parameters_after")
        flat = row["parameters_after_flat"]
        _require(type(flat) is torch.Tensor and flat.device.type == "cpu" and
                 flat.dtype == torch.float32 and flat.shape == (n,) and
                 not flat.requires_grad and bool(torch.isfinite(flat).all()),
                 "branch flat parameters invalid")
        _require(envelopes.same_exact(flat, _parameter_flat(row["parameters_after"])),
                 "branch flat parameters differ from entries")
        try:
            state._validate_optimizer(row["optimizer_after"], profile, update, native_device)
        except Exception as exc:
            raise ArtifactEnvelopeError("branch optimizer invalid: " + type(exc).__name__) from None
        _require(envelopes.same_exact(row["measurement"], measurements["branches"][branch]),
                 "branch measurement differs")
    current = branches["current"]
    _require(current["status"] == "defined", "current branch must be defined")
    _require(envelopes.same_exact(current["delivered_gradient"],
                                  witness["payload"]["delivered_current_gradient"]) and
             envelopes.same_exact(current["parameters_after"], witness["payload"]["parameters_after"]) and
             envelopes.same_exact(current["optimizer_after"], witness["payload"]["optimizer_after"]),
             "current branch differs from source witness")


def _live_tensor_count(anchor: dict[str, Any]) -> int:
    model_count = len(anchor["payload"]["model"]["parameters"])
    optimizer_count = 3 * len(anchor["payload"]["optimizer"]["state"])
    observer_state = anchor["payload"]["observer"]["state"]
    observer_count = sum(observer_state[key] is not None for key in ("V", "S", "grad_mean"))
    return model_count + optimizer_count + observer_count


def _match(witness_value: Any, replay_value: Any, where: str) -> dict[str, Any]:
    _require(envelopes.same_exact(witness_value, replay_value), where + " replay differs")
    return {"witness_sha256": _digest(witness_value), "replay_sha256": _digest(replay_value),
            "directly_equal": True}


def expected_audit_metadata(candidate_state: dict[str, Any], branches: dict[str, Any], *,
                            anchor: dict[str, Any], anchor_receipt: dict[str, Any],
                            witness: dict[str, Any], witness_receipt: dict[str, Any]) -> dict[str, Any]:
    """Derive the exact attestation after, never instead of, producer live checks."""
    try:
        codec.tree_digest({"candidate_state": candidate_state, "branches": branches,
                           "anchor": anchor, "witness": witness})
        _keys(candidate_state, CANDIDATE_KEYS, "candidate_state")
        _keys(branches, BRANCHES, "branches")
        anchor_ref = _receipt_reference(anchor, anchor_receipt)
        witness_ref = _receipt_reference(witness, witness_receipt)
        update = anchor["identity"]["anchor_update"]
        anchor_core_hash = _digest(envelopes.core_from_anchor(anchor))
        candidate_hash = _digest(_candidate_projection(candidate_state))
        count = _live_tensor_count(anchor)
        for name in BRANCHES:
            _keys(branches[name], BRANCH_KEYS, "branch " + name)
            _require(type(branches[name]["status"]) is str and
                     branches[name]["status"] in ("defined", "undefined"),
                     "branch status invalid")
        defined = {name: branches[name]["status"] == "defined" for name in BRANCHES}
        per_clone: dict[str, Any] = {}
        execution: dict[str, Any] = {}
        observer_hash = _digest(anchor["payload"]["observer"])
        rng_hash = _digest(anchor["payload"]["rng"])
        post_mask_hash = _digest([True] * len(state.PROFILE[anchor["profile"]]["parameters"]))
        for name in BRANCHES:
            if defined[name]:
                row = branches[name]
                per_clone[name] = {"executed": True, "start_core_sha256": anchor_core_hash,
                    "start_directly_equal": True, "storage_disjoint": True,
                    "checked_live_tensor_count": count}
                execution[name] = {
                    "parameters_sha256": _digest(row["parameters_after"]),
                    "optimizer_sha256": _digest(row["optimizer_after"]),
                    "assigned_gradient_sha256": _digest(row["delivered_gradient"]),
                    "gradient_null_mask_sha256": _digest(row["assigned_gradient_null_mask"]),
                    "observer_unchanged_sha256": observer_hash,
                    "post_step_null_mask_sha256": post_mask_hash,
                    "rng_sha256": rng_hash, "optimizer_updates_after": update,
                }
            else:
                per_clone[name] = {"executed": False, "start_core_sha256": None,
                    "start_directly_equal": False, "storage_disjoint": False,
                    "checked_live_tensor_count": 0}
                execution[name] = {key: None for key in EXECUTION_ROW_KEYS}

        witness_payload = witness["payload"]
        # The producer separately compares its actual replay forward to this
        # witnessed record. The measurement assembler's sum/divide CE path is
        # intentionally not substituted for that native mean-loss execution.
        live_loss = witness_payload["live_loss"]
        checks = {
            "raw_gradient": _match(witness_payload["raw_gradient"], candidate_state["g"],
                                   "raw gradient"),
            "current_gradient": _match(witness_payload["delivered_current_gradient"],
                                       candidate_state["c"], "current gradient"),
            "delivered_gradient": _match(witness_payload["delivered_current_gradient"],
                                         branches["current"]["delivered_gradient"],
                                         "current delivered gradient"),
            "parameters_after": _match(witness_payload["parameters_after"],
                                       branches["current"]["parameters_after"],
                                       "current parameters"),
            "optimizer_after": _match(witness_payload["optimizer_after"],
                                      branches["current"]["optimizer_after"],
                                      "current optimizer"),
            "observer_after": _match(witness_payload["observer_after"],
                                     candidate_state["observer_after"], "shared observer"),
            "rng": {"anchor_rng_sha256": rng_hash,
                    "witness_rng_sha256": witness_payload["state_hashes"]["rng_after"],
                    "replay_rng_sha256": rng_hash,
                    "replay_directly_equals_anchor": True},
            "live_loss": _match(witness_payload["live_loss"], live_loss, "live loss"),
        }
        _require(checks["rng"]["witness_rng_sha256"] == rng_hash,
                 "witness RNG hash differs from anchor")
        return {
            "producer_bindings": {
                "anchor_ref": anchor_ref, "source_witness_ref": witness_ref,
                "sources_tree_sha256": _digest(anchor["payload"]["bindings"]["sources"]),
                "environment_tree_sha256": _digest(anchor["payload"]["bindings"]["environment"]),
                "formula_ids": list(FORMULA_IDS), "tolerance_ids": list(TOLERANCE_IDS),
            },
            "candidate_proof": {
                "observer_updates_before": update - 1, "observer_ingests": 1,
                "observer_updates_after": update, "candidate_core_sha256": candidate_hash,
                "candidate_core_after_canonical_sha256": candidate_hash,
                "candidate_core_after_reverse_sha256": candidate_hash,
                "candidate_core_directly_unchanged": True,
            },
            "clone_independence_proof": {
                "anchor_core_sha256": anchor_core_hash,
                "per_order": {"canonical": state.clone_tree(per_clone),
                              "reverse": state.clone_tree(per_clone)},
                "anchor_core_after_sha256": anchor_core_hash,
                "anchor_directly_unchanged": True,
            },
            "branch_execution_proof": {
                "canonical_order": list(BRANCHES), "verification_order": list(reversed(BRANCHES)),
                "observer_ingests": 0,
                "per_order": {"canonical": state.clone_tree(execution),
                              "reverse": state.clone_tree(execution)},
                "per_branch_directly_equal": {name: True for name in BRANCHES},
                "all_defined_order_invariant": True,
            },
            "current_replay_proof": {
                "source_witness_artifact_id": witness["artifact_id"],
                "checks": checks, "overall_status": "exact",
            },
            "completion": {
                "defined_branch_count": sum(defined.values()),
                "undefined_branch_count": len(BRANCHES) - sum(defined.values()),
                "required_branch_keys_present": True,
                "required_comparison_keys_present": True,
                "exact_replay_complete": True, "producer_artifact_status": "complete",
            },
        }
    except ArtifactEnvelopeError:
        raise
    except Exception as exc:
        raise ArtifactEnvelopeError("audit metadata derivation failed: " + type(exc).__name__) from None


def _validate_audit(value: Any, *, expected: dict[str, Any]) -> None:
    _keys(value, AUDIT_KEYS, "audit_metadata")
    _keys(value["producer_bindings"], PRODUCER_KEYS, "producer_bindings")
    _keys(value["candidate_proof"], CANDIDATE_PROOF_KEYS, "candidate_proof")
    _keys(value["clone_independence_proof"], CLONE_PROOF_KEYS, "clone proof")
    _keys(value["clone_independence_proof"]["per_order"], ("canonical", "reverse"),
          "clone per_order")
    for order in ("canonical", "reverse"):
        _keys(value["clone_independence_proof"]["per_order"][order], BRANCHES,
              "clone order " + order)
        for row in value["clone_independence_proof"]["per_order"][order].values():
            _keys(row, CLONE_ROW_KEYS, "clone proof row")
    execution = value["branch_execution_proof"]
    _keys(execution, EXECUTION_PROOF_KEYS, "branch execution proof")
    _keys(execution["per_order"], ("canonical", "reverse"), "execution per_order")
    for order in ("canonical", "reverse"):
        _keys(execution["per_order"][order], BRANCHES, "execution order " + order)
        for row in execution["per_order"][order].values():
            _keys(row, EXECUTION_ROW_KEYS, "execution proof row")
    _keys(execution["per_branch_directly_equal"], BRANCHES, "per-branch equality")
    replay = value["current_replay_proof"]
    _keys(replay, REPLAY_PROOF_KEYS, "current replay proof")
    _keys(replay["checks"], REPLAY_CHECKS, "current replay checks")
    for name, row in replay["checks"].items():
        _keys(row, RNG_MATCH_KEYS if name == "rng" else MATCH_KEYS, "replay check " + name)
    _keys(value["completion"], COMPLETION_KEYS, "completion")
    _require(envelopes.same_exact(value, expected), "audit metadata differs from derived proof")


def _validate_impl(value: Any, *, anchor: dict[str, Any], anchor_receipt: dict[str, Any],
                   witness: dict[str, Any], witness_receipt: dict[str, Any],
                   expected_candidates: dict[str, Any], expected_measurements: dict[str, Any],
                   context: dict[str, Any]) -> dict[str, Any]:
    identity, profile = _context(context)
    envelopes.validate_anchor(anchor, **context)
    source.validate_source_witness(witness, anchor=anchor, anchor_receipt=anchor_receipt, **context)
    _receipt_reference(anchor, anchor_receipt)
    _receipt_reference(witness, witness_receipt)
    envelopes.validate_common(value, schema_name="i7_branch_results", kind="branch-results",
                              identity=identity, profile=profile)
    _keys(value["payload"], PAYLOAD_KEYS, "branch-results payload")
    _keys(expected_measurements, EXPECTED_MEASUREMENT_KEYS, "expected measurements")
    _keys(expected_measurements["branches"], BRANCHES, "expected measurement branches")
    _keys(expected_measurements["comparisons"], ("branch_pairs", "contrasts"),
          "expected comparisons")
    _validate_expected_candidates(expected_candidates, profile)
    payload = value["payload"]
    _validate_candidate(payload["candidate_state"], anchor=anchor, witness=witness,
                        expected=expected_candidates, measurements=expected_measurements,
                        profile=profile)
    _validate_branches(payload["branches"], anchor=anchor, witness=witness,
                       expected=expected_candidates, measurements=expected_measurements,
                       profile=profile)
    _require(envelopes.same_exact(payload["comparisons"], expected_measurements["comparisons"]),
             "comparisons differ from expected measurements")
    assembled = {"measurement_before": payload["candidate_state"]["measurement_before"],
                 "branches": {name: payload["branches"][name]["measurement"]
                              for name in BRANCHES},
                 "comparisons": payload["comparisons"]}
    _validate_measurement_structure(assembled, branches=payload["branches"], anchor=anchor,
                                    profile=profile, artifact_id=value["artifact_id"])
    expected_audit = expected_audit_metadata(payload["candidate_state"], payload["branches"],
        anchor=anchor, anchor_receipt=anchor_receipt, witness=witness,
        witness_receipt=witness_receipt)
    _validate_audit(payload["audit_metadata"], expected=expected_audit)
    codec.tree_digest({"anchor": anchor, "witness": witness, "branch_results": value,
                       "expected_candidates": expected_candidates,
                       "expected_measurements": expected_measurements})
    return value


def validate_branch_results(value: Any, *, anchor: dict[str, Any],
                            anchor_receipt: dict[str, Any], witness: dict[str, Any],
                            witness_receipt: dict[str, Any],
                            expected_candidates: dict[str, Any],
                            expected_measurements: dict[str, Any], **anchor_context: Any) -> dict[str, Any]:
    """Validate one CPU saved result against independently supplied expectations."""
    try:
        return _validate_impl(value, anchor=anchor, anchor_receipt=anchor_receipt,
            witness=witness, witness_receipt=witness_receipt,
            expected_candidates=expected_candidates, expected_measurements=expected_measurements,
            context=anchor_context)
    except ArtifactEnvelopeError:
        raise
    except Exception as exc:
        raise ArtifactEnvelopeError("branch-result validation failed: " + type(exc).__name__) from None


def make_branch_results(candidate_state: dict[str, Any], branches: dict[str, Any],
                        comparisons: dict[str, Any], audit_metadata: dict[str, Any], *,
                        created_utc: str, anchor: dict[str, Any],
                        anchor_receipt: dict[str, Any], witness: dict[str, Any],
                        witness_receipt: dict[str, Any],
                        expected_candidates: dict[str, Any],
                        expected_measurements: dict[str, Any], **anchor_context: Any) -> dict[str, Any]:
    """Construct an owned branch-results envelope; original aliases fail first."""
    try:
        identity, profile = _context(anchor_context)
        codec.validate_created_utc(created_utc)
        # This precedes every clone and spans trusted expectations as well as payload inputs.
        codec.tree_digest({"candidate_state": candidate_state, "branches": branches,
                           "comparisons": comparisons, "audit_metadata": audit_metadata,
                           "anchor": anchor, "witness": witness,
                           "expected_candidates": expected_candidates,
                           "expected_measurements": expected_measurements})
        payload = {"candidate_state": candidate_state, "branches": branches,
                   "comparisons": comparisons, "audit_metadata": audit_metadata}
        result = envelopes.wrap(payload, schema_name="i7_branch_results", kind="branch-results",
                                identity=identity, profile=profile, created_utc=created_utc)
        return _validate_impl(result, anchor=anchor, anchor_receipt=anchor_receipt,
            witness=witness, witness_receipt=witness_receipt,
            expected_candidates=expected_candidates, expected_measurements=expected_measurements,
            context=anchor_context)
    except ArtifactEnvelopeError:
        raise
    except Exception as exc:
        raise ArtifactEnvelopeError("branch-result construction failed: " + type(exc).__name__) from None
