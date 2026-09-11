"""I7 saved-input numerical audit: fixed recomputation, bounded retained failures.

No native model/optimizer execution. Numerical references use only NumPy modules;
Torch is used for existing CPU artifact validation and conversion. Producer
attestations are checked for consistency, never promoted to historical proof.
"""
import copy
import hashlib
import math

import numpy as np
import torch

import anchor_envelope as base
import artifact_envelopes as branch_schema
import artifact_store as storage
import audit_diagnostics as diagnostics
import branch_execution as pinned
import data_probe_bindings as data
import identity_codec as codec
import independent_numerics as numerical
import measurement_audit as measurements
import source_capture as source
import source_environment as provenance
import state_core as state
import verified_plan_load as verified

PAYLOAD_KEYS = ("input_bindings", "exact_validation", "candidate_operator_audit",
                "branch_audits", "measurement_audits", "comparison_audits",
                "overall_status", "fatal_failures")
CHECK_KEYS = ("saved_structure", "projections", "delivery", "adamw", "measurements")
EXCLUSIONS = ["native32_forward_reexecution", "independent_observer_history",
              "independent_capture_and_control_flow_history", "scientific_phase_order",
              "aggregate_resource_certification", "independent_resource_event_history",
              "hostile_same_user_provenance"]
MAX_NODES, MAX_DEPTH, MAX_STRING_BYTES, MAX_JSON_BYTES = 200000, 32, 4*2**20, 8*2**20
ERROR_CODES = ("AuditEnvelopeError", "EnvelopeError", "ArtifactEnvelopeError", "StateError",
               "ValueError", "TypeError", "KeyError", "IndexError", "RuntimeError",
               "OverflowError", "FloatingPointError", "AuditDiagnosticError", "ResourceGuardAbort", "OtherError")


class AuditEnvelopeError(ValueError):
    pass


class _GuardAbort(Exception):
    pass


def _tick(guard, label):
    if guard is not None:
        try:
            guard(label)
        except Exception as error:
            raise _GuardAbort(type(error).__name__) from error


def _need(ok, code):
    if not ok:
        raise AuditEnvelopeError(code)


def _keys(value, keys, code):
    _need(type(value) is dict and len(value) == len(keys) and
          all(type(a) is type(b) and a == b for a, b in zip(value, keys)), code)


def _equal(a, b, code):
    _need(base.same_exact(a, b), code)


def _numpy(value):
    if type(value) is torch.Tensor:
        return value.numpy().copy()
    if type(value) is dict:
        return {k: _numpy(v) for k, v in value.items()}
    if type(value) is list:
        return [_numpy(v) for v in value]
    if type(value) is tuple:
        return tuple(_numpy(v) for v in value)
    return value


def _pack(value, dimension):
    if type(value) is dict:
        return {key: diagnostics.pack_indices(item, dimension)
                if key in ("failing_flat_indices", "identity_mismatch_flat_indices")
                else _pack(item, dimension) for key, item in value.items()}
    if type(value) is list:
        return [_pack(item, dimension) for item in value]
    return value


def _bounded(value):
    diagnostics.primitive_tree(value, max_nodes=MAX_NODES, max_depth=MAX_DEPTH,
                               max_string_bytes=MAX_STRING_BYTES)
    _need(len(codec.json_bytes(value)) <= MAX_JSON_BYTES, "audit_json_budget_exceeded")
    return value


def _result(value, passed):
    return dict(status="pass" if passed else "fatal_validation", completed=True,
                error_code=None, result=value)


def _failed(error):
    # Raw files remain the evidence. Never persist arbitrary exception text.
    return dict(status="fatal_validation", completed=False,
                error_code=type(error).__name__ if type(error).__name__ in ERROR_CODES else "OtherError", result=None)


def _structural(anchor, anchor_receipt, witness, witness_receipt, branch, context):
    profile, identity = context["profile"], context["identity"]
    base.validate_anchor(anchor, **context)
    source.validate_source_witness(witness, anchor=anchor, anchor_receipt=anchor_receipt, **context)
    base.validate_common(branch, schema_name="i7_branch_results", kind="branch-results",
                         identity=identity, profile=profile)
    codec.tree_digest(dict(anchor=anchor, witness=witness, branch=branch))
    payload = branch["payload"]
    _keys(payload, branch_schema.PAYLOAD_KEYS, "branch_payload")
    c, rows = payload["candidate_state"], payload["branches"]
    _keys(c, branch_schema.CANDIDATE_KEYS, "candidate_keys")
    _keys(rows, measurements.BRANCHES, "branch_order")
    core = base.core_from_anchor(anchor)
    p = state.PROFILE[profile]["n_params"]
    device = core["model"]["parameters"][0]["native_device"]
    for key in ("g", "c", "l"):
        branch_schema._native(c[key], profile=profile, where=key, shape=(p,))
        _need(c[key]["native_device"] == device, "candidate_device")
    for key, expected in (("g", witness["payload"]["raw_gradient"]),
                          ("c", witness["payload"]["delivered_current_gradient"]),
                          ("previous_basis", core["observer"]["state"]["V"]),
                          ("post_ingest_basis", witness["payload"]["observer_after"]["state"]["V"]),
                          ("observer_after", witness["payload"]["observer_after"])):
        _equal(c[key], expected, "candidate_witness_"+key)
    state._validate_observer(c["observer_after"], profile, identity["anchor_update"], device)
    zero = {key: not bool(torch.count_nonzero(c[key]["value"])) for key in ("c", "l")}
    for norm, key in (("nc", "c"), ("nl", "l")):
        _need(type(c[norm]) is float and math.isfinite(c[norm]) and c[norm] >= 0 and
              (c[norm] == 0) == zero[key], "candidate_norm_domain")
    defined = {key: True for key in measurements.BRANCHES}
    defined["restored"] = not (not zero["c"] and zero["l"])
    defined["reciprocal"] = not (not zero["l"] and zero["c"])
    reasons = {key: None for key in measurements.BRANCHES}
    if not defined["restored"]:
        reasons["restored"] = "positive_current_norm_zero_lagged_direction"
    if not defined["reciprocal"]:
        reasons["reciprocal"] = "positive_lagged_norm_zero_current_direction"
    _equal(c["zero_cases"], dict(current_zero=zero["c"], lagged_zero=zero["l"],
                               branch_defined_mask=defined, branch_reasons=reasons), "zero_cases")
    for key, row in rows.items():
        _keys(row, branch_schema.BRANCH_KEYS, "branch_row")
        _equal(row["status"], "defined" if defined[key] else "undefined", "branch_domain")
        _equal(row["reason"], reasons[key], "branch_reason")
        if not defined[key]:
            _need(all(row[k] is None for k in branch_schema.BRANCH_KEYS[2:]), "undefined_payload")
            continue
        branch_schema._native(row["delivered_gradient"], profile=profile, where="delivered", shape=(p,))
        _need(row["delivered_gradient"]["native_device"] == device, "delivery_device")
        _equal(row["assigned_gradient_null_mask"], [False]*4, "assigned_mask")
        model = copy.deepcopy(core["model"])
        model["parameters"] = row["parameters_after"]
        state._validate_model(model, profile)
        _need(all(entry["native_device"] == device for entry in row["parameters_after"]), "endpoint_device")
        _equal(row["parameters_after_flat"], torch.cat([x["value"].reshape(-1)
               for x in row["parameters_after"]]).contiguous(), "flat_endpoint")
        state._validate_optimizer(row["optimizer_after"], profile, identity["anchor_update"], device)
        _equal(row["optimizer_after"]["param_groups"], core["optimizer"]["param_groups"], "Adam_options")
    for key in ("delivered_gradient", "parameters_after", "optimizer_after"):
        wkey = "delivered_current_gradient" if key == "delivered_gradient" else key
        _equal(rows["current"][key], witness["payload"][wkey], "current_saved_replay_"+key)
    expected = branch_schema.expected_audit_metadata(c, rows, anchor=anchor,
        anchor_receipt=anchor_receipt, witness=witness, witness_receipt=witness_receipt)
    branch_schema._validate_audit(payload["audit_metadata"], expected=expected)
    return core, c, rows, defined


def _delivery(c, rows):
    vectors = {key: _numpy(c[key]["value"]) for key in ("g", "c", "l")}
    norms = {key: float(np.linalg.norm(value.astype(np.float64))) for key, value in vectors.items()}
    nc, nl = norms["c"], norms["l"]
    largest = max(nc, nl)
    distance = None if not nc or not nl else float(np.linalg.norm(
        vectors["c"].astype(np.float64)/nc-vectors["l"].astype(np.float64)/nl))
    cosine = None if distance is None else float((vectors["c"].astype(np.float64)/nc) @
                                               (vectors["l"].astype(np.float64)/nl))
    separation = None if not largest else abs(nc-nl)/largest
    expected = dict(current_norm=nc, lagged_norm=nl, signed_norm_difference=nc-nl,
        relative_norm_separation=separation, current_lagged_norm_ratio=None if not nl else nc/nl,
        ratio_reason="zero_lagged_norm" if not nl else None, unit_direction_distance=distance,
        unit_direction_cosine=cosine, direction_reason="zero_norm" if distance is None else None,
        norm_leverage=separation is not None and separation > 2e-6,
        direction_leverage=distance is not None and distance > 2e-6)
    _keys(c["leverage"], tuple(expected), "candidate_leverage_keys")
    metadata = {}
    # CPU64 norm/derived-scalar comparison; these roundoff ceilings do not
    # replace or enlarge the separate native delivery norm/direction gates.
    for name, observed, reference in [("nc",c["nc"],nc),("nl",c["nl"],nl)]+[
            ("leverage."+key,c["leverage"][key],value) for key,value in expected.items()]:
        if type(reference) is float:
            _need(type(observed) is float and math.isfinite(observed), "candidate_metadata_scalar")
            scale = max(abs(observed), abs(reference))
            if name == "leverage.signed_norm_difference":
                scale = max(scale, nc, nl)
            elif name in ("leverage.relative_norm_separation", "leverage.unit_direction_distance",
                          "leverage.unit_direction_cosine"):
                scale = max(scale, 1.)
            ceiling = 8*numerical.gamma(2*vectors["g"].size+4, numerical.U64)*scale+8*vectors["g"].size*numerical.H64
            error = abs(observed-reference)
            passed = error <= ceiling
        else:
            ceiling, error = None, None
            passed = type(observed) is type(reference) and observed == reference
        metadata[name] = dict(producer_value=observed, auditor_value=reference, fixed_ceiling=ceiling,
                              achieved_discrepancy=error, status="pass" if passed else "fatal_validation")
    reports = {}
    for key in measurements.BRANCHES:
        row = rows[key]
        if row["status"] == "undefined":
            reports[key] = dict(status="domain_undefined", reason=row["reason"],
                                target_norm=None, actual_norm=None, norm_error=None,
                                direction_error=None, exact_unscaled=None)
            continue
        direction_key = {"raw":"g", "current":"c", "lagged":"l", "restored":"l",
                         "reciprocal":"c", "zero":"g"}[key]
        target = norms[{"restored":"c", "reciprocal":"l"}.get(key, direction_key)] if key != "zero" else 0.
        direction = vectors[direction_key].astype(np.float64)
        delivered = _numpy(row["delivered_gradient"]["value"])
        actual = float(np.linalg.norm(delivered.astype(np.float64)))
        exact = delivered.tobytes() == vectors[direction_key].tobytes() if key in ("raw", "current", "lagged") else None
        norm_error = abs(actual-target)/target if target else actual
        direction_error = (float(np.linalg.norm(delivered.astype(np.float64)/actual -
                           direction/norms[direction_key])) if target and actual else None)
        passed = (actual == 0 if target == 0 else actual > 0 and norm_error <= 1e-6 and
                  direction_error is not None and direction_error <= 1e-6)
        reports[key] = dict(status="pass" if passed and exact is not False else "fatal_validation",
                            reason=None, target_norm=target, actual_norm=actual,
                            norm_error=norm_error, direction_error=direction_error, exact_unscaled=exact)
    return dict(candidate_metadata=metadata, branches=reports)


def _expected_counts(defined):
    contrasts = sum(all(defined[key] for key in coefficients) for coefficients in measurements.COEFFICIENTS.values())
    count = sum(defined.values())
    return dict(before_probes=4, branch_probes=4*count, defined_branches=count, pairs=15,
                contrasts=15, defined_contrasts=contrasts, auxiliary_before_chunks=10,
                auxiliary_branch_chunks=10*count, auxiliary_contrast_chunks=10*contrasts)


def _adam(core, rows, update):
    results = {}
    for branch, row in rows.items():
        if row["status"] == "undefined":
            results[branch] = dict(status="domain_undefined", reason=row["reason"], parameters=None)
            continue
        entries, offset = {}, 0
        group = core["optimizer"]["param_groups"][0]
        for index, (before, after) in enumerate(zip(core["model"]["parameters"], row["parameters_after"])):
            theta, endpoint = _numpy(before["value"]), _numpy(after["value"])
            old, new = core["optimizer"]["state"][index], row["optimizer_after"]["state"][index]
            grad = _numpy(row["delivered_gradient"]["value"])[offset:offset+theta.size].reshape(theta.shape)
            result = numerical.audit_adamw(theta, grad, _numpy(old["exp_avg"]["value"]),
                _numpy(old["exp_avg_sq"]["value"]), update,
                dict(theta=endpoint, moment=_numpy(new["exp_avg"]["value"]),
                     variance=_numpy(new["exp_avg_sq"]["value"]), next_step=int(new["step"])),
                options={key: group[key] for key in numerical.ADAMW_OPTIONS})
            entries[before["name"]] = _pack(dict(formula_id="i7_adamw_screen_v1", **result), theta.size)
            offset += theta.size
        results[branch] = dict(status="pass" if all(row["passed"] for row in entries.values()) else "fatal_validation",
                               reason=None, parameters=entries)
    return results


def _checks(anchor, anchor_receipt, witness, witness_receipt, branch, context, guard):
    checks = {key: dict(status="not_run", completed=False, error_code=None, result=None) for key in CHECK_KEYS}
    try:
        core, c, rows, defined = _structural(anchor, anchor_receipt, witness, witness_receipt, branch, context)
        checks["saved_structure"] = _result(dict(kind="saved_value_consistency_not_history"), True)
    except (ValueError, TypeError, KeyError, IndexError, RuntimeError) as error:
        checks["saved_structure"] = _failed(error)
        return checks
    p = state.PROFILE[context["profile"]]["n_params"]
    for stage in CHECK_KEYS[1:]:
        try:
            _tick(guard, "independent_audit."+stage)
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                if stage == "projections":
                    result = {key: _pack(dict(formula_id="i7_projection_screen_v1", **numerical.audit_projection(_numpy(c["g"]["value"]),
                        None if c[basis] is None else _numpy(c[basis]["value"]), _numpy(c[grad]["value"]))), p)
                        for key, basis, grad in (("current", "post_ingest_basis", "c"), ("lagged", "previous_basis", "l"))}
                    passed = all(row["passed"] for row in result.values())
                elif stage == "delivery":
                    result = _delivery(c, rows)
                    passed = all(row["status"] != "fatal_validation" for section in result.values() for row in section.values())
                elif stage == "adamw":
                    result = _adam(core, rows, context["identity"]["anchor_update"])
                    passed = all(row["status"] != "fatal_validation" for row in result.values())
                else:
                    materialized = data.materialize(context["images_bytes"], context["labels_bytes"],
                        plan=context["plan"], identity=context["identity"], profile=context["profile"],
                        expected_files=context["expected_files"])
                    before = {entry["name"]: _numpy(entry["value"]) for entry in core["model"]["parameters"]}
                    endpoints = {key: None if not defined[key] else {entry["name"]: _numpy(entry["value"])
                                 for entry in row["parameters_after"]} for key, row in rows.items()}
                    gradients = {key: None if not defined[key] else _numpy(row["delivered_gradient"]["value"])
                                 for key, row in rows.items()}
                    measured = dict(measurement_before=_numpy(c["measurement_before"]),
                        branches={key:_numpy(row["measurement"]) for key,row in rows.items()},
                        comparisons=_numpy(branch["payload"]["comparisons"]))
                    # Let resource exceptions escape unchanged, never become audit nulls.
                    result = measurements.audit(before, endpoints, gradients, _numpy(materialized["probes"]),
                                                 measured, profile=context["profile"],
                                                 guard=lambda label: _tick(guard, label))
                    result = _pack(result, p)
                    expected_contrasts = {f"contrasts.{name}.{probe}.{kind}.independent"
                        for name,coefficients in measurements.COEFFICIENTS.items() if all(defined[key] for key in coefficients)
                        for probe in measurements.PROBES for kind in ("Y", "D", "Ddata", "R")}
                    expected_contrasts.update(f"contrasts.{name}.auxiliary_clean.chunk{chunk}.Y.independent"
                        for name,coefficients in measurements.COEFFICIENTS.items() if all(defined[key] for key in coefficients) for chunk in range(10))
                    passed = (result["overall_status"] == "pass" and result["audit_complete"] and
                        result["completion_counts"] == _expected_counts(defined) and
                        {key for key in result["comparison_audits"] if key.endswith(".independent")} == expected_contrasts)
                checks[stage] = _result(_bounded(result), passed)
                if stage == "measurements":
                    # The numerical auditor can return a retained partial report
                    # after malformed input. Returning is not completing.
                    checks[stage]["completed"] = result["audit_complete"]
        except _GuardAbort:
            checks[stage] = dict(status="fatal_validation", completed=False,
                                error_code="ResourceGuardAbort", result=None)
            break
        except (ValueError, TypeError, KeyError, IndexError, RuntimeError, ArithmeticError) as error:
            checks[stage] = _failed(error)
        if checks[stage]["status"] == "fatal_validation":
            break
    return checks


def _bindings(anchor, anchor_receipt, witness, witness_receipt, branch, branch_receipt, plan_reference, context):
    _keys(plan_reference, ("name", "status", "encoding", "size_bytes", "sha256", "receipt_name",
                           "receipt_size_bytes", "receipt_sha256"), "plan_reference_keys")
    storage._validate_name(plan_reference["name"])
    _need(plan_reference["status"] == "complete" and plan_reference["encoding"] == "torch_weights_only", "plan_reference_status")
    _equal({k: plan_reference[k] for k in ("sha256", "size_bytes")}, context["plan_artifact"], "plan_reference_bytes")
    receipt_name = "receipt-"+hashlib.sha256(plan_reference["name"].encode("ascii")).hexdigest()+".json"
    receipt_bytes = storage._json_bytes(dict(schema="i7_artifact_receipt_v1", name=plan_reference["name"],
        size=plan_reference["size_bytes"], sha256=plan_reference["sha256"], status="complete", encoding="torch_weights_only"))
    _equal(plan_reference["receipt_name"], receipt_name, "plan_receipt_name")
    _equal(plan_reference["receipt_size_bytes"], len(receipt_bytes), "plan_receipt_size")
    _equal(plan_reference["receipt_sha256"], hashlib.sha256(receipt_bytes).hexdigest(), "plan_receipt_hash")
    return dict(anchor_ref=branch_schema._receipt_reference(anchor, anchor_receipt),
                source_witness_ref=branch_schema._receipt_reference(witness, witness_receipt),
                branch_results_ref=branch_schema._receipt_reference(branch, branch_receipt),
                plan_ref=copy.deepcopy(plan_reference),
                sources_tree_sha256=codec.tree_digest(anchor["payload"]["bindings"]["sources"]),
                plan_probe_data_tree_sha256=codec.tree_digest({key: anchor["payload"]["bindings"][key]
                    for key in ("plan", "probes", "data")}))


def build_audit(*, anchor, anchor_receipt, witness, witness_receipt, branch,
                branch_receipt, plan_reference, auditor_environment, created_utc, guard=None, **context):
    """Recompute an owned report from raw values. Pure receipt bindings are assertions.

    The file adapter below establishes the receipt/value association. This pure
    API does not itself authenticate files or inspect runtime/source bytes.
    """
    _need(set(context) == set(branch_schema.CONTEXT_KEYS), "audit_context_keys")
    _need(guard is None or callable(guard), "audit_guard")
    profile = context["profile"]
    _need(profile in (storage.SCIENTIFIC, storage.MLP_FIXTURE), "audit_profile")
    codec.validate_identity(context["identity"], profile=profile)
    provenance.validate_environment(auditor_environment, profile=profile)
    _need(auditor_environment["runtime_role"] == ("cpu_audit" if profile == storage.SCIENTIFIC else "fixture_cpu"), "audit_runtime_role")
    _equal(auditor_environment["repository_root_realpath"], context["sources"]["repository_root_realpath"], "audit_runtime_root")
    checks = _checks(anchor, anchor_receipt, witness, witness_receipt, branch, context, guard)
    completed = all(row["completed"] for row in checks.values())
    passed = completed and all(row["status"] == "pass" for row in checks.values())
    adam = checks["adamw"]["result"]
    measurement = checks["measurements"]["result"]
    defined = (None if not checks["saved_structure"]["completed"] else
               {key:row["status"] == "defined" for key,row in branch["payload"]["branches"].items()})
    completion = dict(required_stages=list(CHECK_KEYS), completed_stages=[k for k,v in checks.items() if v["completed"]],
        checks_complete=completed, overall_status="pass" if passed else "fatal_validation",
        expected_projection_checks=2, projection_checks=0 if checks["projections"]["result"] is None else len(checks["projections"]["result"]),
        expected_delivery_rows=6, delivery_rows=0 if checks["delivery"]["result"] is None else len(checks["delivery"]["result"]["branches"]),
        expected_adam_parameter_checks=None if defined is None else 4*sum(defined.values()),
        adam_parameter_checks=0 if adam is None else sum(len(row["parameters"]) for row in adam.values() if row["parameters"] is not None),
        adam_tensor_screens=0 if adam is None else sum(len(parameter["checks"]) for row in adam.values() if row["parameters"] is not None for parameter in row["parameters"].values()),
        expected_measurement_counts=None if defined is None else _expected_counts(defined),
        measurement_counts=None if measurement is None else copy.deepcopy(measurement["completion_counts"]),
        scientific_execution_certified=False)
    comparison_result = None if measurement is None else measurement.pop("comparison_audits")
    payload = dict(input_bindings=_bindings(anchor, anchor_receipt, witness, witness_receipt,
                    branch, branch_receipt, plan_reference, context),
        exact_validation=dict(auditor_environment=copy.deepcopy(auditor_environment),
                              saved_structure=checks["saved_structure"], completion=completion,
                              scope_exclusions=list(EXCLUSIONS)),
        candidate_operator_audit=dict(scope="represented_operator_only", projections=checks["projections"], delivery=checks["delivery"]),
        branch_audits=checks["adamw"], measurement_audits=checks["measurements"],
        comparison_audits=comparison_result,
        overall_status=completion["overall_status"],
        fatal_failures=[dict(stage=k, completed=v["completed"], error_code=v["error_code"])
                        for k,v in checks.items() if v["status"] == "fatal_validation"])
    _bounded(payload)
    return base.wrap(payload, schema_name="i7_anchor_numerical_audit", kind="independent-audit",
                     identity=context["identity"], profile=profile, created_utc=created_utc)


def validate_audit(value, *, guard=None, **inputs):
    """Recompute raw checks, or the completed prefix of a resource-aborted audit.

    An abort's stage/history is a retained external declaration, not something
    numerically reproducible. It can never yield a passing/complete certificate.
    Its prior completed checks and fixed incomplete/downstream shape are checked
    without requiring the original transient resource condition to recur.
    """
    base.validate_common(value, schema_name="i7_anchor_numerical_audit", kind="independent-audit",
                         identity=inputs["identity"], profile=inputs["profile"])
    _keys(value["payload"], PAYLOAD_KEYS, "audit_payload_keys")
    _bounded(value["payload"])
    _need(guard is None or callable(guard), "audit_validation_guard")
    payload = value["payload"]
    rows = dict(saved_structure=payload["exact_validation"]["saved_structure"],
        projections=payload["candidate_operator_audit"]["projections"],
        delivery=payload["candidate_operator_audit"]["delivery"],
        adamw=payload["branch_audits"], measurements=payload["measurement_audits"])
    aborted = []
    for stage, row in rows.items():
        _keys(row, ("status", "completed", "error_code", "result"), "audit_stage_keys")
        if row["error_code"] == "ResourceGuardAbort":
            aborted.append(stage)
    _need(len(aborted) <= 1 and "saved_structure" not in aborted, "audit_abort_membership")
    abort_at = None if not aborted else "independent_audit."+aborted[0]
    def verification_guard(label):
        if guard is not None:
            guard(label)
        if label == abort_at:
            raise _GuardAbort("verify_declared_incomplete_prefix")
    expected = build_audit(created_utc=value["created_utc"], guard=verification_guard, **inputs)
    _equal(value, expected, "audit_differs_from_raw_recomputation")
    return value


def audit_and_seal(*, store, anchor, anchor_receipt, witness, witness_receipt,
                   branch, branch_receipt, plan_root, plan_name, created_utc, guard=None, **context):
    """Pinned-file CPU audit; failed reports are sealed before terminal status."""
    valid_store = type(store) is storage.ArtifactStore
    rng = None
    try:
        _need(valid_store and not store._closed and not store._terminal, "audit_store_unavailable")
        _need(store.profile == context["profile"], "audit_store_profile")
        root = context["sources"]["repository_root_realpath"]
        profile = context["profile"]
        actual_sources = provenance.collect_verified_sources(root, profile=profile)
        _equal(actual_sources, context["sources"], "audit_sources_changed")
        role = "cpu_audit" if profile == storage.SCIENTIFIC else "fixture_cpu"
        environment = provenance.collect_runtime_environment(root, profile=profile, runtime_role=role)
        rng = state._raw_rng_state()
        # An exact same-root handle keeps the existing exclusive lock. A second
        # open/shared flock would conflict with this process's own writer lock.
        if str(plan_root) == str(store.root):
            plan_loaded = verified.load_verified_plan_from_store(store, plan_name,
                identity=context["identity"], profile=profile,
                expected_sha256=context["plan_artifact"]["sha256"])
        else:
            plan_loaded = verified.load_verified_plan(plan_root, plan_name, identity=context["identity"],
                profile=profile, expected_sha256=context["plan_artifact"]["sha256"])
        _equal(plan_loaded["plan"], context["plan"], "audit_plan_value_differs")
        _equal(plan_loaded["binding"], anchor["payload"]["bindings"]["plan"], "audit_plan_binding_differs")
        loaded = [pinned._load_pinned(store, obj, receipt, schema_name=schema) for obj,receipt,schema in (
            (anchor,anchor_receipt,"i7_anchor"), (witness,witness_receipt,"i7_source_step_witness"),
            (branch,branch_receipt,"i7_branch_results"))]
        artifact = build_audit(anchor=loaded[0], anchor_receipt=anchor_receipt,
            witness=loaded[1], witness_receipt=witness_receipt, branch=loaded[2], branch_receipt=branch_receipt,
            plan_reference=plan_loaded["artifact"], auditor_environment=environment,
            created_utc=created_utc, guard=guard, **context)
        _equal(state._raw_rng_state(), rng, "audit_consumed_rng")
        _equal(provenance.collect_verified_sources(root, profile=profile), actual_sources, "audit_sources_changed")
        _equal(provenance.collect_runtime_environment(root, profile=profile, runtime_role=role), environment, "audit_runtime_changed")
        receipt = store.write_tensor_tree(artifact["artifact_id"]+".pt", artifact)
        _need(not store._terminal, "audit_store_became_terminal")
        reloaded = pinned._load_pinned(store, artifact, receipt, schema_name="i7_anchor_numerical_audit")
        _equal(reloaded, artifact, "audit_sealed_value_differs")
        _equal(provenance.collect_verified_sources(root, profile=profile), actual_sources, "audit_seal_sources_changed")
        _equal(provenance.collect_runtime_environment(root, profile=profile, runtime_role=role), environment, "audit_seal_runtime_changed")
        _equal(state._raw_rng_state(), rng, "audit_seal_consumed_rng")
        if artifact["payload"]["overall_status"] != "pass":
            store._fail("independent_audit_failed", "sealed_numerical_audit")
        return dict(artifact=artifact, receipt=receipt)
    except BaseException as error:
        if rng is not None:
            state._set_rng_state(rng)
        if valid_store and not store._closed and not store._terminal:
            store._fail("independent_audit_failed", type(error).__name__)
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            raise
        raise AuditEnvelopeError("independent_audit_failed:"+type(error).__name__) from None
