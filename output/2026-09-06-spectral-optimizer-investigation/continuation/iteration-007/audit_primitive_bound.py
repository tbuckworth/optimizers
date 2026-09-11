"""Prospective protocol-2 primitive bound for the 16 independent audits.

This is integer arithmetic over the frozen scientific constructors.  It does
not import Torch/NumPy, inspect an artifact, run an auditor, or admit a write.
Every cost expands repeated values and therefore takes no memo/alias saving.
"""
from __future__ import annotations

import primitive_storage_bound as primitive


PROFILE = "scientific_mnist_current32_v1"
UPDATES = (101, 500, 1000, 2000)
MEMBERS = tuple(
    [("primary", bundle, update) for bundle in (71001, 71002, 71003) for update in UPDATES]
    + [("sensitivity", 71901, update) for update in UPDATES]
)
BRANCHES = ("raw", "current", "lagged", "restored", "reciprocal", "zero")
PROBES = ("batch_noisy", "train_probe_noisy", "train_probe_clean", "auxiliary_clean")
PARAMETERS = (("0.weight", 50_176), ("0.bias", 64), ("2.weight", 640), ("2.bias", 10))
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
CHECK_KEYS = ("saved_structure", "projections", "delivery", "adamw", "measurements")
ERROR_CODES = (
    "AuditEnvelopeError", "EnvelopeError", "ArtifactEnvelopeError", "StateError",
    "ValueError", "TypeError", "KeyError", "IndexError", "RuntimeError",
    "OverflowError", "FloatingPointError", "AuditDiagnosticError",
    "ResourceGuardAbort", "OtherError",
)
EXCLUSIONS = (
    "native32_forward_reexecution", "independent_observer_history",
    "independent_capture_and_control_flow_history", "scientific_phase_order",
    "aggregate_resource_certification", "independent_resource_event_history",
    "hostile_same_user_provenance",
)
MEASUREMENT_EXCLUSIONS = (
    "external_provenance", "Adam_operator_audits", "live_source_replay",
    "native32_forward_reexecution",
)
P = 50_890
AUDITOR_ENVIRONMENT_JSON_CAP = 8 << 10
PARTIAL_VALIDATION_UTF8_CAP = 4 << 20
ARTIFACT_SIZE_CAP = 1 << 30
PLAN_SIZE_CAP = 4 << 20
RECEIPT_SIZE_CAP = 4096
HASH_BYTES = 64
RECEIPT_NAME_BYTES = len("receipt-") + HASH_BYTES + len(".json")

# Exact files from which this theorem was derived at base f55c4aee937d96f9d07e8da3ce2d29b13d5a27a5.
SOURCE_SHA256 = {
    "audit_envelope.py": "0bf084cbb42c13defdb0daa3b6114ccb2fadd7a2897f7fcc5c1fd988b29172b6",
    "audit_diagnostics.py": "ce9ef09a47f44f074ec7eb5b56659681931b36ebfa617be344c961f15797c2db",
    "independent_numerics.py": "27031928fb17ceacb3fc5dc4c3413d0233e5139d177d14fd96fb296134b8ca80",
    "measurement_audit.py": "100dd76c60115e927fea4083bdd637972aba47168e0a090640f68ca6d55995ba",
    "measurement_assembly.py": "0b8c305e60a97eb5a0483588be3d0962b8afeed512c13ac495e9ed487d49fba8",
    "response_math.py": "7645ca1b65d29c4278ccd1795e817ec6e8438ed8e6fa418e1d02e3d1d5b2276b",
    "artifact_envelopes.py": "dfe3558f5930acfcf24f2458c7186ab62f283076854b8743959e8ef7925bb441",
    "anchor_envelope.py": "d2290e07044d5ef26cef3b72e5427b00216848680f36b709fa547d944081fc17",
    "identity_codec.py": "e47c5de57e3bccb47fd96cc5183e35f614577a16867f7caf04819e562df15f5d",
    "state_core.py": "a727ce516d6d3e283bdbb75162a8b1701198d6d9cffd40a5a70a005c83a43c7e",
    "branch_execution.py": "9f09bd255ce1edde5203fbaace7c80858bebea3e28ad54d56d69688e27196c1d",
    "native_phase_policy.py": "17209724d32faf2cfa960f8e07422af270f512e44b1d6c377ec7066a87873de0",
    "verified_plan_load.py": "cc04fba0a97736611c01e7986e0ee9eb5d824c6b05ecd85a045070d14a9f736e",
    "source_capture.py": "b4fd71331040c82e1d28574e1df34f5da2eb78e9de9fd890877ff65ce2866254",
    "native_audit.py": "69cba7a5160101b426aadd216cc635fcf379dba2972268c8afb6b686e6980c11",
    "scientific_controller.py": "7b80b3dbd3b2cf8a5f6289618643e96243506703ba79f2f93ebeb457eee6463a",
    "native_controller.py": "752892c0f5554aac64b002bc704dcbf3c3b550f739bcc44f0af6d0d096e6475f",
}
DEPENDENCY_SHA256 = {
    "primitive_storage_bound.py": "6414824202152ed6cae710007e0e55564657ef9f406ede09ede3c5ac84a50709",
    "pickle_storage_bound.py": "df3843821ceb7eeeae69b1a36d81ba231d1d1b920286c753f257f4d4d9dbddf0",
}


def _maximum(*costs: int) -> int:
    if not costs or any(type(cost) is not int or cost < 0 for cost in costs):
        raise ValueError("finite alternatives require nonnegative integer costs")
    return max(costs)


def _enum(*values) -> int:
    return _maximum(*(primitive.literal(value) for value in values))


def _artifact_id(role: str, bundle: int, update: int, kind: str) -> str:
    return f"{PROFILE}--{role}--b{bundle}--u{update}--{kind}"


def _member(role: str, bundle: int, update: int) -> None:
    if type(role) is not str or type(bundle) is not int or type(update) is not int:
        raise ValueError("audit member fields require exact types")
    if (role, bundle, update) not in MEMBERS:
        raise ValueError("not one of the 16 scientific independent audits")


def _index_record(dimension: int) -> int:
    return primitive.pdict({
        "encoding": primitive.literal("flat_bitset_hex_lsb0_v1"),
        "dimension": primitive.integer(dimension, dimension),
        "count": primitive.integer(0, dimension),
        "bits_hex": primitive.text(2 * ((dimension + 7) // 8)),
    })


def _screen_fields(dimension: int) -> dict[str, int]:
    return {
        "passed": primitive.BOOL,
        "max_absolute_error": primitive.FLOAT,
        "max_envelope": primitive.FLOAT,
        "max_error_envelope_ratio": _maximum(primitive.FLOAT, primitive.NULL),
        "ratio_null_reason": _maximum(primitive.NULL, primitive.literal("zero_envelope_mismatch")),
        "failing_flat_indices": _index_record(dimension),
        "error_l2": primitive.FLOAT,
        "envelope_l2": primitive.FLOAT,
    }


def _projection_result() -> int:
    fields = {"formula_id": primitive.literal("i7_projection_screen_v1"), **_screen_fields(P)}
    fields.update({
        "operator": _enum("identity", "basis_outer_product"),
        "rank": _maximum(primitive.NULL, primitive.integer(1, 32)),
        "identity_bytes_equal": _maximum(primitive.NULL, primitive.BOOL),
        "identity_mismatch_flat_indices": _index_record(P),
    })
    return primitive.pdict(fields)


def _projection_stage_result() -> int:
    row = _projection_result()
    return primitive.pdict({"current": row, "lagged": row})


def _metadata_record(value_cost: int, scalar: bool) -> int:
    return primitive.pdict({
        "producer_value": value_cost,
        "auditor_value": value_cost,
        "fixed_ceiling": primitive.FLOAT if scalar else primitive.NULL,
        "achieved_discrepancy": primitive.FLOAT if scalar else primitive.NULL,
        "status": _enum("pass", "fatal_validation"),
    })


def _delivery_metadata(nc_zero: bool, nl_zero: bool) -> int:
    largest_zero = nc_zero and nl_zero
    direction_undefined = nc_zero or nl_zero
    values = {
        "nc": (primitive.FLOAT, True),
        "nl": (primitive.FLOAT, True),
        "leverage.current_norm": (primitive.FLOAT, True),
        "leverage.lagged_norm": (primitive.FLOAT, True),
        "leverage.signed_norm_difference": (primitive.FLOAT, True),
        "leverage.relative_norm_separation": ((primitive.NULL if largest_zero else primitive.FLOAT), not largest_zero),
        "leverage.current_lagged_norm_ratio": ((primitive.NULL if nl_zero else primitive.FLOAT), not nl_zero),
        "leverage.ratio_reason": ((primitive.literal("zero_lagged_norm") if nl_zero else primitive.NULL), False),
        "leverage.unit_direction_distance": ((primitive.NULL if direction_undefined else primitive.FLOAT), not direction_undefined),
        "leverage.unit_direction_cosine": ((primitive.NULL if direction_undefined else primitive.FLOAT), not direction_undefined),
        "leverage.direction_reason": ((primitive.literal("zero_norm") if direction_undefined else primitive.NULL), False),
        "leverage.norm_leverage": (primitive.BOOL, False),
        "leverage.direction_leverage": (primitive.BOOL, False),
    }
    return primitive.pdict({key: _metadata_record(cost, scalar) for key, (cost, scalar) in values.items()})


def _domain_mask(nc_zero: bool, nl_zero: bool) -> dict[str, bool]:
    return {
        branch: not (branch == "restored" and not nc_zero and nl_zero
                     or branch == "reciprocal" and not nl_zero and nc_zero)
        for branch in BRANCHES
    }


def _delivery_branch(branch: str, defined: bool) -> int:
    if not defined:
        return primitive.pdict({
            "status": primitive.literal("domain_undefined"),
            "reason": _enum("positive_current_norm_zero_lagged_direction",
                            "positive_lagged_norm_zero_current_direction"),
            "target_norm": primitive.NULL, "actual_norm": primitive.NULL,
            "norm_error": primitive.NULL, "direction_error": primitive.NULL,
            "exact_unscaled": primitive.NULL,
        })
    exact = primitive.BOOL if branch in ("raw", "current", "lagged") else primitive.NULL
    direction = primitive.NULL if branch == "zero" else _maximum(primitive.FLOAT, primitive.NULL)
    return primitive.pdict({
        "status": _enum("pass", "fatal_validation"), "reason": primitive.NULL,
        "target_norm": primitive.FLOAT, "actual_norm": primitive.FLOAT,
        "norm_error": primitive.FLOAT, "direction_error": direction,
        "exact_unscaled": exact,
    })


def _delivery_stage_result() -> int:
    alternatives = []
    for nc_zero, nl_zero in ((False, False), (True, True), (False, True), (True, False)):
        mask = _domain_mask(nc_zero, nl_zero)
        alternatives.append(primitive.pdict({
            "candidate_metadata": _delivery_metadata(nc_zero, nl_zero),
            "branches": primitive.pdict({branch: _delivery_branch(branch, mask[branch])
                                          for branch in BRANCHES}),
        }))
    return _maximum(*alternatives)


def _adam_screen(dimension: int) -> int:
    return primitive.pdict(_screen_fields(dimension))


def _adam_parameter(dimension: int) -> int:
    screens = {name: _adam_screen(dimension) for name in ("theta", "moment", "variance")}
    return primitive.pdict({
        "formula_id": primitive.literal("i7_adamw_screen_v1"),
        "passed": primitive.BOOL,
        "checks": primitive.pdict(screens),
        "native_displacement_norm": primitive.FLOAT,
        "endpoint_envelope_to_displacement": _maximum(primitive.FLOAT, primitive.NULL),
        "displacement_ratio_null_reason": _maximum(
            primitive.NULL, primitive.literal("zero_native_displacement")),
    })


def _adam_branch(branch: str, defined: bool) -> int:
    if not defined:
        return primitive.pdict({
            "status": primitive.literal("domain_undefined"),
            "reason": _enum("positive_current_norm_zero_lagged_direction",
                            "positive_lagged_norm_zero_current_direction"),
            "parameters": primitive.NULL,
        })
    return primitive.pdict({
        "status": _enum("pass", "fatal_validation"), "reason": primitive.NULL,
        "parameters": primitive.pdict({name: _adam_parameter(size) for name, size in PARAMETERS}),
    })


def _adam_stage_result(mask: dict[str, bool]) -> int:
    return primitive.pdict({branch: _adam_branch(branch, mask[branch]) for branch in BRANCHES})


def _scalar_record(*, sign: bool = False) -> int:
    fields = {
        "producer_value": primitive.FLOAT, "auditor_value": primitive.FLOAT,
        "fixed_ceiling": primitive.FLOAT, "achieved_discrepancy": primitive.FLOAT,
        "formula_id": primitive.literal("i7_assembly_roundoff_v1"),
        "audit_status": _enum("pass", "fatal_validation"),
    }
    if sign:
        fields.update({
            "producer_sign": primitive.integer(-1, 1),
            "auditor_sign": primitive.integer(-1, 1),
            "resolution_status": _enum("resolved_positive", "resolved_negative", "unresolved_numerical"),
            "resolution_reason": _maximum(
                primitive.NULL, primitive.literal("sign_disagreement"),
                primitive.literal("insufficient_fixed_margin")),
        })
    return primitive.pdict(fields)


def _unresolved_geometry_record() -> int:
    return primitive.pdict({
        "producer_value": primitive.FLOAT, "auditor_value": primitive.FLOAT,
        "audit_status": primitive.literal("unresolved_numerical"),
        "reason": primitive.literal("denominator_interval_contains_zero"),
    })


def _vector_record(dimension: int) -> int:
    return primitive.pdict({
        "max_absolute_error": primitive.FLOAT, "error_l2": primitive.FLOAT,
        "max_component_ceiling": primitive.FLOAT,
        "max_error_ceiling_ratio": _maximum(primitive.FLOAT, primitive.NULL),
        "failing_flat_indices": _index_record(dimension),
        "auditor_sha256": primitive.text(HASH_BYTES),
        "audit_status": _enum("pass", "fatal_validation"),
    })


def _failure(path: str, reason_cost: int | str) -> int:
    reason = reason_cost if type(reason_cost) is int else primitive.literal(reason_cost)
    return primitive.pdict({"path": primitive.literal(path), "reason": reason})


def _measurement_parts(mask: dict[str, bool], artifact_id: str,
                       failure_mode: str) -> tuple[int, int, dict[str, int]]:
    if failure_mode not in ("none", "fixed", "partial"):
        raise ValueError("unknown measurement failure mode")
    measurements: dict[str, int] = {}
    comparisons: dict[str, int] = {}
    failures: list[int] = []
    native_paths: list[str] = []
    unresolved_paths: list[str] = []

    def scalar(path: str, *, comparison: bool = False, sign: bool = False) -> None:
        (comparisons if comparison else measurements)[path] = _scalar_record(sign=sign)
        failures.append(_failure(path, "numerical ceiling exceeded"))

    def vector(path: str) -> None:
        measurements[path] = _vector_record(P)
        failures.append(_failure(path, "vector component ceiling exceeded"))

    def concordance(path: str) -> None:
        scalar(path + ".abs_discrepancy")
        scalar(path + ".descriptive_ceiling")
        native_paths.append(path)

    def geometry(path: str) -> None:
        for field in ("left_norm", "right_norm", "distance"):
            scalar(path + "." + field)
        cosine = path + ".cosine"
        comparisons[cosine] = _maximum(_scalar_record(), _unresolved_geometry_record())
        failures.append(_failure(cosine, "numerical ceiling exceeded"))
        unresolved_paths.append(cosine)

    def scalar_binding(path: str) -> None:
        for suffix in ("direct_route", "component_route", "identity_discrepancy", "identity_ceiling"):
            scalar(path + "." + suffix)
        failures.append(_failure(path, "direct/component identity ceiling exceeded"))

    for probe in PROBES:
        path = "before." + probe
        vector(path + ".q_values")
        scalar(path + ".q_norm")
        scalar(path + ".CE")
        concordance(path + ".concordance")
        if probe == "auxiliary_clean":
            for chunk in range(10):
                cp = path + f".chunk{chunk}"
                scalar(cp + ".CE")
                concordance(cp + ".concordance")

    for branch in BRANCHES:
        if not mask[branch]:
            continue
        path = "branches." + branch
        dp = path + ".displacement"
        vector(dp + ".delta_values")
        vector(dp + ".delta_data_values")
        for field in ("delta_norm", "delta_squared_norm", "delta_data_norm", "delta_data_squared_norm"):
            scalar(dp + "." + field)
        geometry(dp + ".delta_delta_data")
        for probe in PROBES:
            pp = path + "." + probe
            scalar(pp + ".after_ce")
            for field in ("Y", "D", "Ddata", "R"):
                scalar(pp + "." + field)
                scalar(pp + ".stored_identity." + field)
            scalar(pp + ".native_Y")
            concordance(pp + ".concordance.after")
            concordance(pp + ".concordance.Y")
            if probe == "auxiliary_clean":
                for chunk in range(10):
                    cp = pp + f".chunks{chunk}"
                    scalar(cp + ".after")
                    scalar(cp + ".Y")
                    scalar(cp + ".stored_Y")
                    scalar(cp + ".native_Y")
                    concordance(cp + ".concordance.after")
                    concordance(cp + ".concordance.Y")

    pairs = [(left, right) for index, left in enumerate(BRANCHES)
             for right in BRANCHES[index + 1:]]
    for left, right in pairs:
        if not (mask[left] and mask[right]):
            continue
        path = f"pairs.{left}__{right}"
        for field in ("delivered_gradient", "delta", "delta_data"):
            geometry(path + "." + field)
        scalar(path + ".discrepancy_binding")
        scalar(path + ".discrepancy_independent")
        scalar(path + ".ceiling")
        failures.append(_failure(path, "full/data pair ceiling exceeded"))

    defined_contrasts = []
    for name, coefficients in COEFFICIENTS.items():
        if not all(mask[branch] for branch in coefficients):
            continue
        defined_contrasts.append(name)
        path = "contrasts." + name
        for field in ("delta", "delta_data"):
            scalar(path + "." + field + ".norm")
            scalar(path + "." + field + ".energy")
        scalar(path + ".agreement.discrepancy")
        scalar(path + ".agreement.ceiling")
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
                    scalar_binding(cp + ".Y")
                    scalar_binding(cp + ".nativeY")
                    concordance(cp + ".concordance.Y")
                    scalar(cp + ".Y.independent", comparison=True, sign=True)

    fixed_failure_count = len(failures)
    selected_failures = [] if failure_mode == "none" else list(failures)
    if failure_mode == "partial":
        selected_failures.append(
            _failure("validation", primitive.text(PARTIAL_VALIDATION_UTF8_CAP)))
    defined_count = sum(mask.values())
    count_fields = {
        "before_probes": primitive.integer(0, 4),
        "branch_probes": primitive.integer(0, 4 * defined_count),
        "defined_branches": primitive.integer(0, defined_count),
        "pairs": primitive.integer(0, 15),
        "contrasts": primitive.integer(0, 15),
        "defined_contrasts": primitive.integer(0, len(defined_contrasts)),
        "auxiliary_before_chunks": primitive.integer(0, 10),
        "auxiliary_branch_chunks": primitive.integer(0, 10 * defined_count),
        "auxiliary_contrast_chunks": primitive.integer(0, 10 * len(defined_contrasts)),
    }
    undefined = [branch for branch in BRANCHES if not mask[branch]]
    weak = [name for name in defined_contrasts if name == "interaction"
            or name.startswith("direction_") or name.startswith("norm_")]
    report = primitive.pdict({
        "profile": primitive.literal(PROFILE),
        "scope": primitive.literal("raw_value_measurements_only"),
        "overall_status": _enum("pass", "fatal_validation"),
        "audit_complete": primitive.BOOL,
        "completion_counts": primitive.pdict(count_fields),
        "fatal_failures": primitive.plist(selected_failures),
        "measurement_audits": primitive.pdict(measurements),
        # comparison_audits is popped into the outer audit payload.
        "domain_undefined_branches": primitive.literal(undefined),
        "weak_factor_contrasts": primitive.literal(weak),
        "native_discordant_paths": primitive.plist([primitive.literal(path) for path in native_paths]),
        "unresolved_geometry_paths": primitive.plist([primitive.literal(path) for path in unresolved_paths]),
        "binding_artifact_id": _maximum(primitive.NULL, primitive.literal(artifact_id)),
        "scope_exclusions": primitive.literal(list(MEASUREMENT_EXCLUSIONS)),
    })
    counts = {
        "measurement_paths": len(measurements),
        "comparison_paths": len(comparisons),
        "fixed_failure_sites": fixed_failure_count,
        "failure_rows_with_validation": fixed_failure_count + 1,
        "native_discordance_paths": len(native_paths),
        "unresolved_geometry_paths": len(unresolved_paths),
        "defined_branches": defined_count,
        "defined_contrasts": len(defined_contrasts),
        "measurement_bitsets": 4 + 2 * defined_count,
    }
    return report, primitive.pdict(comparisons), counts


def _stage_row(result_cost: int, *, partial: bool = False) -> int:
    no_result = primitive.pdict({
        "status": _enum("not_run", "fatal_validation"),
        "completed": primitive.BOOL,
        "error_code": _maximum(primitive.NULL, *[primitive.literal(code) for code in ERROR_CODES]),
        "result": primitive.NULL,
    })
    completed = primitive.pdict({
        "status": _enum("pass", "fatal_validation"), "completed": primitive.BOOL,
        "error_code": primitive.NULL, "result": result_cost,
    })
    # The same field costs cover completed true and retained-partial false.
    return _maximum(no_result, completed) if partial else _maximum(no_result, completed)


def _expected_counts(mask: dict[str, bool]) -> int:
    defined = sum(mask.values())
    contrasts = sum(all(mask[branch] for branch in coefficients) for coefficients in COEFFICIENTS.values())
    return primitive.pdict({
        "before_probes": primitive.integer(4, 4),
        "branch_probes": primitive.integer(4 * defined, 4 * defined),
        "defined_branches": primitive.integer(defined, defined),
        "pairs": primitive.integer(15, 15),
        "contrasts": primitive.integer(15, 15),
        "defined_contrasts": primitive.integer(contrasts, contrasts),
        "auxiliary_before_chunks": primitive.integer(10, 10),
        "auxiliary_branch_chunks": primitive.integer(10 * defined, 10 * defined),
        "auxiliary_contrast_chunks": primitive.integer(10 * contrasts, 10 * contrasts),
    })


def _partial_counts(mask: dict[str, bool]) -> int:
    defined = sum(mask.values())
    contrasts = sum(all(mask[branch] for branch in coefficients) for coefficients in COEFFICIENTS.values())
    return primitive.pdict({
        "before_probes": primitive.integer(0, 4),
        "branch_probes": primitive.integer(0, 4 * defined),
        "defined_branches": primitive.integer(0, defined),
        "pairs": primitive.integer(0, 15),
        "contrasts": primitive.integer(0, 15),
        "defined_contrasts": primitive.integer(0, contrasts),
        "auxiliary_before_chunks": primitive.integer(0, 10),
        "auxiliary_branch_chunks": primitive.integer(0, 10 * defined),
        "auxiliary_contrast_chunks": primitive.integer(0, 10 * contrasts),
    })


def _completion(mask: dict[str, bool]) -> int:
    prefixes = [primitive.literal(list(CHECK_KEYS[:count])) for count in range(len(CHECK_KEYS) + 1)]
    expected = _expected_counts(mask)
    measured = _partial_counts(mask)
    return primitive.pdict({
        "required_stages": primitive.literal(list(CHECK_KEYS)),
        "completed_stages": _maximum(*prefixes),
        "checks_complete": primitive.BOOL,
        "overall_status": _enum("pass", "fatal_validation"),
        "expected_projection_checks": primitive.integer(2, 2),
        "projection_checks": primitive.integer(0, 2),
        "expected_delivery_rows": primitive.integer(6, 6),
        "delivery_rows": primitive.integer(0, 6),
        "expected_adam_parameter_checks": _maximum(primitive.NULL, primitive.integer(20, 24)),
        "adam_parameter_checks": primitive.integer(0, 24),
        "adam_tensor_screens": primitive.integer(0, 72),
        "expected_measurement_counts": _maximum(primitive.NULL, expected),
        "measurement_counts": _maximum(primitive.NULL, measured),
        "scientific_execution_certified": primitive.BOOL,
    })


def _reference(artifact_id: str, schema_name: str) -> int:
    return primitive.pdict({
        "artifact_id": primitive.literal(artifact_id),
        "schema_name": primitive.literal(schema_name),
        "name": primitive.literal(artifact_id + ".pt"),
        "size_bytes": primitive.integer(1, ARTIFACT_SIZE_CAP),
        "sha256": primitive.text(HASH_BYTES),
        "status": primitive.literal("complete"),
        "encoding": primitive.literal("torch_weights_only"),
        "receipt_name": primitive.text(RECEIPT_NAME_BYTES),
        "receipt_size_bytes": primitive.integer(1, RECEIPT_SIZE_CAP),
        "receipt_sha256": primitive.text(HASH_BYTES),
    })


def _plan_reference(bundle: int) -> int:
    name = f"i7-native-b{bundle}-plan.pt"
    return primitive.pdict({
        "name": primitive.literal(name), "status": primitive.literal("complete"),
        "encoding": primitive.literal("torch_weights_only"),
        "size_bytes": primitive.integer(1, PLAN_SIZE_CAP),
        "sha256": primitive.text(HASH_BYTES),
        "receipt_name": primitive.text(RECEIPT_NAME_BYTES),
        "receipt_size_bytes": primitive.integer(1, RECEIPT_SIZE_CAP),
        "receipt_sha256": primitive.text(HASH_BYTES),
    })


def _identity(role: str, bundle: int, update: int) -> int:
    evidence_role = "primary" if role == "primary" else "sensitivity"
    return primitive.pdict({
        "run_id": primitive.literal("2026-09-06-spectral-optimizer-investigation"),
        "iteration": primitive.integer(7, 7),
        "execution_role": primitive.literal(role),
        "evidence_role": primitive.literal(evidence_role),
        "bundle": primitive.integer(bundle, bundle),
        "anchor_update": primitive.integer(update, update),
        "source_policy": primitive.literal("current32"),
        "condition": primitive.literal("noise_0.9"),
        "steps_total": primitive.integer(2000, 2000),
        "anchor_phase": primitive.literal("pre_forward_pre_observe"),
        "anchor_completed_updates": primitive.integer(update - 1, update - 1),
        "anchor_completed_observations": primitive.integer(update - 1, update - 1),
        "rng_namespace_prefix": primitive.plist([
            primitive.integer(20260906, 20260906), primitive.integer(bundle, bundle)]),
        "stream_roles": primitive.pdict({
            0: primitive.literal("permutation"),
            1: primitive.literal("replacement_uniforms"),
            2: primitive.literal("replacement_digits"),
            3: primitive.literal("initialization_seed"),
            4: primitive.literal("training_batches"),
            5: primitive.literal("training_probe"),
            6: primitive.literal("unused"),
        }),
    })


def _top_failure() -> int:
    return primitive.pdict({
        "stage": _maximum(*(primitive.literal(stage) for stage in CHECK_KEYS)),
        "completed": primitive.BOOL,
        "error_code": _maximum(primitive.NULL, *[primitive.literal(code) for code in ERROR_CODES]),
    })


def _payload_for_state(role: str, bundle: int, update: int, environment_cost: int,
                       nc_zero: bool, nl_zero: bool, failure_mode: str) -> tuple[int, dict[str, int]]:
    mask = _domain_mask(nc_zero, nl_zero)
    branch_id = _artifact_id(role, bundle, update, "branch-results")
    measurement, comparisons, counts = _measurement_parts(mask, branch_id, failure_mode)
    saved = primitive.pdict({"kind": primitive.literal("saved_value_consistency_not_history")})
    payload = primitive.pdict({
        "input_bindings": primitive.pdict({
            "anchor_ref": _reference(_artifact_id(role, bundle, update, "anchor"), "i7_anchor"),
            "source_witness_ref": _reference(_artifact_id(role, bundle, update, "source-witness"), "i7_source_step_witness"),
            "branch_results_ref": _reference(branch_id, "i7_branch_results"),
            "plan_ref": _plan_reference(bundle),
            "sources_tree_sha256": primitive.text(HASH_BYTES),
            "plan_probe_data_tree_sha256": primitive.text(HASH_BYTES),
        }),
        "exact_validation": primitive.pdict({
            "auditor_environment": environment_cost,
            "saved_structure": _stage_row(saved),
            "completion": _completion(mask),
            "scope_exclusions": primitive.literal(list(EXCLUSIONS)),
        }),
        "candidate_operator_audit": primitive.pdict({
            "scope": primitive.literal("represented_operator_only"),
            "projections": _stage_row(_projection_stage_result()),
            "delivery": _stage_row(_delivery_stage_result()),
        }),
        "branch_audits": _stage_row(_adam_stage_result(mask)),
        "measurement_audits": _stage_row(measurement, partial=True),
        "comparison_audits": _maximum(primitive.NULL, comparisons),
        "overall_status": (_enum("pass", "fatal_validation") if failure_mode != "none"
                           else primitive.literal("pass")),
        "fatal_failures": (primitive.literal([]) if failure_mode == "none"
                           else primitive.plist([_top_failure()])),
    })
    counts = dict(counts)
    counts["projection_bitsets"] = 4
    counts["adam_bitsets"] = 12 * sum(mask.values())
    counts["total_bitsets"] = counts["projection_bitsets"] + counts["adam_bitsets"] + counts["measurement_bitsets"]
    return payload, counts


def audit_payload_subtree_cost(*, role: str, bundle: int, update: int,
                               auditor_environment_json_bytes: int = AUDITOR_ENVIRONMENT_JSON_CAP) -> int:
    """Maximum primitive/container cost of one admitted scientific audit payload.

    The environment argument is the actual separately admitted compact ASCII
    canonical-JSON byte ceiling and must not exceed 8192.
    """
    _member(role, bundle, update)
    if (type(auditor_environment_json_bytes) is not int
            or not 1 <= auditor_environment_json_bytes <= AUDITOR_ENVIRONMENT_JSON_CAP):
        raise ValueError("auditor environment canonical JSON must be in [1,8192] bytes")
    environment = primitive.json_subtree_bytes(auditor_environment_json_bytes)
    costs = [_payload_for_state(role, bundle, update, environment, nc_zero, nl_zero, "partial")[0]
             for nc_zero, nl_zero in ((False, False), (True, True), (False, True), (True, False))]
    return _maximum(*costs)


def _successful_payload_cost(role: str, bundle: int, update: int,
                             environment_cost: int) -> int:
    return _maximum(*[
        _payload_for_state(role, bundle, update, environment_cost, nc_zero, nl_zero, "none")[0]
        for nc_zero, nl_zero in ((False, False), (True, True), (False, True), (True, False))
    ])


def _wrapped_pickle_bytes(role: str, bundle: int, update: int, payload: int) -> int:
    root = primitive.pdict({
        "schema_name": primitive.literal("i7_anchor_numerical_audit"),
        "schema_version": primitive.integer(1, 1),
        "profile": primitive.literal(PROFILE),
        "artifact_id": primitive.literal(_artifact_id(role, bundle, update, "independent-audit")),
        "created_utc": primitive.text(20),
        "identity": _identity(role, bundle, update),
        "payload": payload,
        "payload_tensor_bytes": primitive.integer(0, 0),
    })
    return 3 + root


def audit_artifact_pickle_bytes(*, role: str, bundle: int, update: int,
                                auditor_environment_json_bytes: int = AUDITOR_ENVIRONMENT_JSON_CAP) -> int:
    """Maximum data.pkl bytes for one complete tensor-free audit envelope."""
    payload = audit_payload_subtree_cost(
        role=role, bundle=bundle, update=update,
        auditor_environment_json_bytes=auditor_environment_json_bytes)
    return _wrapped_pickle_bytes(role, bundle, update, payload)


def audit_topology_counts(*, nc_zero: bool = False, nl_zero: bool = False) -> dict[str, int]:
    """Expose source-derived counts for review; this performs arithmetic only."""
    if type(nc_zero) is not bool or type(nl_zero) is not bool:
        raise ValueError("zero flags must be exact booleans")
    _, counts = _payload_for_state(
        "primary", 71001, 101,
        primitive.json_subtree_bytes(AUDITOR_ENVIRONMENT_JSON_CAP), nc_zero, nl_zero, "partial")
    return counts


def scientific_audits_pickle_bytes(
        auditor_environment_json_bytes: int = AUDITOR_ENVIRONMENT_JSON_CAP) -> dict[str, object]:
    """Return exact member costs and the selected uniform conservative total."""
    if (type(auditor_environment_json_bytes) is not int
            or not 1 <= auditor_environment_json_bytes <= AUDITOR_ENVIRONMENT_JSON_CAP):
        raise ValueError("auditor environment canonical JSON must be in [1,8192] bytes")
    environment = primitive.json_subtree_bytes(auditor_environment_json_bytes)
    rows = []
    for role, bundle, update in MEMBERS:
        successful = _wrapped_pickle_bytes(
            role, bundle, update, _successful_payload_cost(role, bundle, update, environment))
        terminal = audit_artifact_pickle_bytes(
            role=role, bundle=bundle, update=update,
            auditor_environment_json_bytes=auditor_environment_json_bytes)
        rows.append({"role": role, "bundle": bundle, "update": update,
                     "successful_pickle_bytes": successful,
                     "terminal_pickle_bytes": terminal})
    all_pass = sum(row["successful_pickle_bytes"] for row in rows)
    prefix = 0
    terminal_scenarios = []
    for index, row in enumerate(rows):
        terminal_scenarios.append(prefix + row["terminal_pickle_bytes"])
        prefix += row["successful_pickle_bytes"]
    terminal_max = max(terminal_scenarios)
    independent_sum = sum(row["terminal_pickle_bytes"] for row in rows)
    uniform_sum = len(rows) * max(row["terminal_pickle_bytes"] for row in rows)
    return {
        "scheduled_payload_count": len(rows),
        "per_artifact": tuple(rows),
        "all_pass_pickle_bytes": all_pass,
        "terminal_prefix_pickle_bytes": terminal_max,
        "independent_per_artifact_max_sum": independent_sum,
        "uniform_independent_max_sum": uniform_sum,
        # Selected assembly term: it does not rely on the controller fail-stop proof.
        "pickle_bytes": uniform_sum,
    }


if __name__ == "__main__":
    print("I7 audit primitive arithmetic only; no audit, artifact, tensor, or admission work.")
