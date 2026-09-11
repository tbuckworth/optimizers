#!/usr/bin/env python3
"""Pure, fixed-membership collection of validated I7 scientific results.

``collect_validated_pairs`` accepts the 16 branch/audit pairs only after the
caller has authenticated their files and run the full branch and audit
validators.  This module rechecks envelope identity, complete reference
declarations, audit-to-branch binding, and the fields it consumes.  It does not
read files or repeat the numerical audit, and therefore cannot replace that
mandatory caller precondition.

No inferential statistics or benefit decision is produced.  Primary summaries
are the frozen complete-three-bundle descriptive summaries; sensitivity cells
have the same shape but are deliberately not pooled.
"""
from __future__ import annotations

import copy
import math
from typing import Any

import anchor_envelope as envelopes
import artifact_envelopes as branch_schema
import artifact_store as storage
import audit_envelope as audit_schema
import identity_codec as codec
import measurement_assembly as measurement
import native_phase_policy as policy
import response_math as response


class CollectionError(ValueError):
    """A fixed result member or its validated evidence is unusable."""


PRIMARY = tuple(response.PRIMARY)
SENSITIVITY = (71901,)
UPDATES = (101, 500, 1000, 2000)
PROBES = tuple(measurement.PROBES)
QUANTITIES = ("Y", "D", "Ddata", "R")
CONTRASTS = tuple(response.COEFFICIENTS)
BRANCHES = tuple(response.BRANCHES)
PROFILE = storage.SCIENTIFIC
PAIR_KEYS = ("branch", "branch_ref", "audit", "audit_ref")
ROW_KEYS = ("identity", "branch_ref", "audit_ref", "leverage", "branch_domain",
            "contrasts", "audit_diagnostics", "provenance")
RESOLUTION_KEYS = ("producer_value", "auditor_value", "fixed_ceiling",
                   "achieved_discrepancy", "formula_id", "audit_status",
                   "producer_sign", "auditor_sign", "resolution_status",
                   "resolution_reason")
CONCORDANCE_KEYS = ("abs_discrepancy", "descriptive_ceiling", "status")


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise CollectionError(message)


def _keys(value: Any, expected: tuple[Any, ...], label: str) -> None:
    _need(type(value) is dict and tuple(value) == expected, label + ": exact ordered keys")


def _finite(value: Any, label: str) -> float:
    _need(type(value) is float and math.isfinite(value), label + ": finite float required")
    return value


def _sha(value: Any, label: str) -> str:
    _need(type(value) is str and len(value) == 64 and
          all(character in "0123456789abcdef" for character in value),
          label + ": invalid SHA-256")
    return value


def _path_list(value: Any, label: str, *, contrast_names: bool = False) -> list[str]:
    _need(type(value) is list and len(value) <= 4096 and
          all(type(item) is str and item.isascii() and 0 < len(item) <= 512 for item in value) and
          len(value) == len(set(value)), label + ": invalid path list")
    if contrast_names:
        _need(all(item in CONTRASTS for item in value), label + ": unknown contrast")
    return copy.deepcopy(value)


def _leverage(value: Any) -> dict[str, Any]:
    _keys(value, branch_schema.LEVERAGE_KEYS, "named-state leverage")
    for key in ("current_norm", "lagged_norm"):
        _need(_finite(value[key], "leverage " + key) >= 0.0, "negative leverage norm")
    _finite(value["signed_norm_difference"], "signed norm difference")
    for key in ("relative_norm_separation", "current_lagged_norm_ratio",
                "unit_direction_distance"):
        _need(value[key] is None or
              (type(value[key]) is float and math.isfinite(value[key]) and value[key] >= 0.0),
              "invalid leverage " + key)
    _need(value["unit_direction_cosine"] is None or
          (type(value["unit_direction_cosine"]) is float and
           math.isfinite(value["unit_direction_cosine"])), "invalid leverage cosine")
    _need(value["ratio_reason"] in (None, "zero_lagged_norm") and
          value["direction_reason"] in (None, "zero_norm") and
          type(value["norm_leverage"]) is bool and type(value["direction_leverage"]) is bool,
          "invalid leverage reason/flag")
    return copy.deepcopy(value)


def _same(left: Any, right: Any, label: str) -> None:
    _need(envelopes.same_exact(left, right), label)


def _expected_membership() -> tuple[tuple[str, int, int], ...]:
    return tuple(("primary", bundle, update) for bundle in PRIMARY for update in UPDATES) + tuple(
        ("sensitivity", bundle, update) for bundle in SENSITIVITY for update in UPDATES)


def _bound_branch_reference(branch: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    name = branch["artifact_id"] + ".pt"
    try:
        policy.validate_reference(reference, name=name, encoding="torch_weights_only")
    except Exception as exc:
        raise CollectionError("invalid branch artifact reference: " + type(exc).__name__) from None
    return {
        "artifact_id": branch["artifact_id"], "schema_name": "i7_branch_results",
        "name": reference["name"], "size_bytes": reference["size_bytes"],
        "sha256": reference["sha256"], "status": reference["status"],
        "encoding": reference["encoding"], "receipt_name": reference["receipt_name"],
        "receipt_size_bytes": reference["receipt_size_bytes"],
        "receipt_sha256": reference["receipt_sha256"],
    }


def _complete_audit(payload: dict[str, Any]) -> dict[str, Any]:
    _keys(payload, audit_schema.PAYLOAD_KEYS, "audit payload")
    completion = payload["exact_validation"]["completion"]
    _need(type(completion) is dict and completion.get("checks_complete") is True and
          completion.get("overall_status") == "pass", "audit completion is not passing")
    _need(payload["overall_status"] == "pass" and payload["fatal_failures"] == [],
          "failed audit refuses result collection")
    stages = (
        payload["exact_validation"]["saved_structure"],
        payload["candidate_operator_audit"]["projections"],
        payload["candidate_operator_audit"]["delivery"],
        payload["branch_audits"], payload["measurement_audits"],
    )
    for row in stages:
        _need(type(row) is dict and row.get("status") == "pass" and
              row.get("completed") is True and row.get("error_code") is None and
              row.get("result") is not None, "failed or incomplete audit stage")
    measured = payload["measurement_audits"]["result"]
    _need(measured.get("overall_status") == "pass" and measured.get("audit_complete") is True and
          measured.get("fatal_failures") == [], "measurement audit is not complete and passing")
    _need(type(payload["comparison_audits"]) is dict, "comparison audits missing")
    return measured


def _domain(branches: dict[str, Any]) -> dict[str, dict[str, Any]]:
    _keys(branches, BRANCHES, "branch membership")
    result = {}
    for name, row in branches.items():
        _need(type(row) is dict and row.get("status") in ("defined", "undefined"),
              "invalid branch domain row")
        defined = row["status"] == "defined"
        reason = row.get("reason")
        _need((defined and reason is None) or
              (not defined and type(reason) is str and bool(reason)),
              "branch domain reason differs")
        result[name] = {"defined": defined, "reason": reason}
    return result


def _audit_resolution(audits: dict[str, Any], path: str, value: float) -> dict[str, Any]:
    _need(path in audits and type(audits[path]) is dict, "missing auditor comparison: " + path)
    row = audits[path]
    _keys(row, RESOLUTION_KEYS, "auditor comparison")
    for key in ("producer_value", "auditor_value", "fixed_ceiling", "achieved_discrepancy"):
        _finite(row[key], "auditor comparison " + key)
    _need(row["fixed_ceiling"] >= 0.0 and row["achieved_discrepancy"] >= 0.0 and
          row["formula_id"] == "i7_assembly_roundoff_v1",
          "invalid auditor arithmetic metadata: " + path)
    _same(row["achieved_discrepancy"], abs(row["producer_value"] - row["auditor_value"]),
          "auditor discrepancy arithmetic differs")
    signs = tuple(int(item > 0) - int(item < 0)
                  for item in (row["producer_value"], row["auditor_value"]))
    resolved = (signs[0] == signs[1] and signs[0] != 0 and
                min(abs(row["producer_value"]), abs(row["auditor_value"])) >
                row["fixed_ceiling"])
    expected_status = (("resolved_positive" if signs[0] > 0 else "resolved_negative")
                       if resolved else "unresolved_numerical")
    expected_reason = (None if resolved else "sign_disagreement" if signs[0] != signs[1]
                       else "insufficient_fixed_margin")
    consistent = (row["producer_sign"] == signs[0] and row["auditor_sign"] == signs[1] and
                  row["resolution_status"] == expected_status and
                  row["resolution_reason"] == expected_reason)
    _need(type(row["producer_sign"]) is int and type(row["auditor_sign"]) is int and
          row.get("audit_status") == "pass" and
          row["achieved_discrepancy"] <= row["fixed_ceiling"] and consistent,
          "invalid auditor resolution: " + path)
    _same(row.get("producer_value"), value, "auditor value is not bound to producer result")
    return copy.deepcopy(row)


def _concordance(value: Any) -> dict[str, Any]:
    _keys(value, CONCORDANCE_KEYS, "native concordance")
    discrepancy = _finite(value["abs_discrepancy"], "native discrepancy")
    ceiling = _finite(value["descriptive_ceiling"], "native ceiling")
    _need(discrepancy >= 0.0 and ceiling >= 0.0 and
          value["status"] == ("within_scale" if discrepancy <= ceiling else "discordant"),
          "native concordance invalid")
    return copy.deepcopy(value)


def _extract_contrasts(branch: dict[str, Any], audit: dict[str, Any],
                       domain: dict[str, dict[str, Any]]) -> dict[str, Any]:
    comparisons = branch["payload"]["comparisons"]
    _keys(comparisons, ("branch_pairs", "contrasts"), "branch comparisons")
    rows = comparisons["contrasts"]
    _keys(rows, CONTRASTS, "contrast membership")
    audits = audit["payload"]["comparison_audits"]
    result: dict[str, Any] = {}
    for name, coefficients in response.COEFFICIENTS.items():
        row = rows[name]
        _same(row.get("coefficients"), coefficients, "contrast coefficients differ")
        _need(type(row.get("defined")) is bool, "contrast defined flag invalid")
        defined = row["defined"]
        expected_defined = all(domain[branch_name]["defined"] for branch_name in coefficients)
        _need(defined == expected_defined,
              "contrast domain flag differs from required branches")
        expected_mask = {branch_name: domain[branch_name]["defined"] for branch_name in BRANCHES}
        _same(row.get("branch_defined_mask"), expected_mask,
              "contrast branch-defined mask differs")
        reason = row.get("reason")
        _need((defined and reason is None) or
              (not defined and reason == "domain_undefined_required_branch"),
              "contrast domain reason differs")
        factor = row.get("factor_leverage")
        _keys(factor, ("direction", "norm"), "factor leverage")
        _need(all(value in ("qualified", "weak", "unavailable", "not_required")
                  for value in factor.values()), "factor leverage value invalid")
        if not defined:
            _need(row.get("probes") is None, "undefined contrast retained probe results")
            result[name] = {"defined": False, "reason": reason,
                            "factor_leverage": copy.deepcopy(factor), "probes": None}
            continue
        probes = row.get("probes")
        _keys(probes, PROBES, "contrast probes")
        probe_result = {}
        for probe, probe_row in probes.items():
            cpu = probe_row["cpu64"]
            _keys(cpu, QUANTITIES, "contrast quantities")
            values = {kind: _finite(cpu[kind]["direct_value"], "contrast value")
                      for kind in QUANTITIES}
            resolutions = {}
            for kind, value in values.items():
                path = f"contrasts.{name}.{probe}.{kind}.independent"
                resolutions[kind] = _audit_resolution(audits, path, value)
            concordance = _concordance(probe_row["concordance"]["Y"])
            chunks = []
            source_chunks = probe_row["auxiliary_chunks"]
            _need(type(source_chunks) is list and len(source_chunks) ==
                  (10 if probe == "auxiliary_clean" else 0), "auxiliary chunk membership")
            for index, chunk in enumerate(source_chunks):
                _need(chunk.get("chunk_index") == index, "auxiliary chunk order")
                audit_path = f"contrasts.{name}.auxiliary_clean.chunk{index}.Y.independent"
                chunk_value = _finite(chunk["cpu64"]["Y"]["direct_value"], "chunk Y")
                chunks.append({
                    "chunk_index": index, "start": chunk["start"], "end": chunk["end"],
                    "count": chunk["count"], "input_sha256": chunk["input_sha256"],
                    "label_sha256": chunk["label_sha256"], "Y": chunk_value,
                    "native_concordance": _concordance(chunk["concordance"]["Y"]),
                    "auditor_resolution": _audit_resolution(audits, audit_path, chunk_value),
                    "branch_path": ["payload", "comparisons", "contrasts", name, "probes",
                                    "auxiliary_clean", "auxiliary_chunks", index],
                    "audit_path": ["payload", "comparison_audits", audit_path],
                })
            probe_result[probe] = {
                "values": values, "auditor_resolution": resolutions,
                "native_concordance": concordance,
                "auxiliary_chunks": chunks,
            }
        result[name] = {"defined": True, "reason": None,
                        "factor_leverage": copy.deepcopy(factor), "probes": probe_result}
    return result


def _extract_pair(pair: dict[str, Any], expected: tuple[str, int, int]) -> dict[str, Any]:
    _keys(pair, PAIR_KEYS, "result pair")
    branch, audit = pair["branch"], pair["audit"]
    role, bundle, update = expected
    identity = branch.get("identity") if type(branch) is dict else None
    _need(type(identity) is dict and identity.get("execution_role") == role and
          identity.get("evidence_role") == role and identity.get("bundle") == bundle and
          identity.get("anchor_update") == update, "unexpected scientific anchor membership")
    try:
        codec.validate_identity(identity, profile=PROFILE)
        envelopes.validate_common(branch, schema_name="i7_branch_results",
                                  kind="branch-results", identity=identity, profile=PROFILE)
        envelopes.validate_common(audit, schema_name="i7_anchor_numerical_audit",
                                  kind="independent-audit", identity=identity, profile=PROFILE)
    except Exception as exc:
        raise CollectionError("invalid common result envelope: " + type(exc).__name__) from None
    _keys(branch["payload"], branch_schema.PAYLOAD_KEYS, "branch payload")
    branch_ref = pair["branch_ref"]
    audit_ref = pair["audit_ref"]
    expected_bound = _bound_branch_reference(branch, branch_ref)
    try:
        policy.validate_reference(audit_ref, name=audit["artifact_id"] + ".pt",
                                  encoding="torch_weights_only")
    except Exception as exc:
        raise CollectionError("invalid audit artifact reference: " + type(exc).__name__) from None
    measured = _complete_audit(audit["payload"])
    bindings = audit["payload"]["input_bindings"]
    _same(bindings.get("branch_results_ref"), expected_bound,
          "audit is not bound to supplied branch artifact reference")
    _same(audit["identity"], branch["identity"], "audit/branch identity differs")
    producer = branch["payload"]["audit_metadata"]["producer_bindings"]
    source_hash = producer["sources_tree_sha256"]
    environment_hash = producer["environment_tree_sha256"]
    _sha(source_hash, "producer sources")
    _sha(environment_hash, "producer environment")
    _same(bindings.get("sources_tree_sha256"), source_hash,
          "producer/auditor source binding differs")
    provenance = {
        "sources_tree_sha256": source_hash,
        "environment_tree_sha256": environment_hash,
        "auditor_environment_sha256": codec.tree_digest(
            audit["payload"]["exact_validation"]["auditor_environment"]),
        "plan_probe_data_tree_sha256": _sha(bindings["plan_probe_data_tree_sha256"],
                                             "plan/probe/data binding"),
    }
    domain = _domain(branch["payload"]["branches"])
    diagnostics = {
        "native_discordant_paths": _path_list(measured["native_discordant_paths"],
                                               "native discordance paths"),
        "unresolved_geometry_paths": _path_list(measured["unresolved_geometry_paths"],
                                                 "unresolved geometry paths"),
        "weak_factor_contrasts": _path_list(measured["weak_factor_contrasts"],
                                             "weak factor contrasts", contrast_names=True),
        "measurement_audit_path": ["payload", "measurement_audits", "result"],
        "comparison_audit_path": ["payload", "comparison_audits"],
    }
    return {
        "identity": copy.deepcopy(identity), "branch_ref": copy.deepcopy(branch_ref),
        "audit_ref": copy.deepcopy(audit_ref),
        "leverage": _leverage(branch["payload"]["candidate_state"]["leverage"]),
        "branch_domain": domain,
        "contrasts": _extract_contrasts(branch, audit, domain),
        "audit_diagnostics": diagnostics, "provenance": provenance,
    }


def _validate_row(row: dict[str, Any], expected: tuple[str, int, int]) -> None:
    """Validate the primitive extraction boundary; used by analytic fixtures."""
    _keys(row, ROW_KEYS, "collected anchor row")
    role, bundle, update = expected
    identity = row["identity"]
    _need(type(identity) is dict and identity.get("execution_role") == role and
          identity.get("bundle") == bundle and identity.get("anchor_update") == update,
          "collected anchor identity differs")
    _keys(row["branch_domain"], BRANCHES, "collected branch domain")
    _keys(row["contrasts"], CONTRASTS, "collected contrasts")
    for name, domain in row["branch_domain"].items():
        _keys(domain, ("defined", "reason"), "collected branch domain row")
        valid_domain = ((domain["defined"] and domain["reason"] is None) or
                        (not domain["defined"] and type(domain["reason"]) is str and
                         bool(domain["reason"])))
        _need(type(domain["defined"]) is bool and valid_domain,
              "collected branch domain row differs")
    for name, coefficients in response.COEFFICIENTS.items():
        item = row["contrasts"][name]
        _need(type(item) is dict and type(item.get("defined")) is bool and
              item["defined"] == all(row["branch_domain"][branch]["defined"]
                                     for branch in coefficients),
              "collected contrast domain differs from required branches")
        _need((item["defined"] and item.get("reason") is None) or
              (not item["defined"] and item.get("reason") ==
               "domain_undefined_required_branch"),
              "collected contrast domain reason differs")
    _need(type(row["leverage"]) is dict and tuple(row["leverage"]) == branch_schema.LEVERAGE_KEYS,
          "collected leverage membership")
    _need(type(row["provenance"]) is dict and tuple(row["provenance"]) ==
          ("sources_tree_sha256", "environment_tree_sha256", "auditor_environment_sha256",
           "plan_probe_data_tree_sha256"), "collected provenance membership")


def _cell(rows: list[dict[str, Any]], contrast: str, probe: str,
          quantity: str, *, primary: bool) -> dict[str, Any]:
    members = []
    values = {}
    for row in rows:
        item = row["contrasts"][contrast]
        value = None if not item["defined"] else item["probes"][probe]["values"][quantity]
        bundle = row["identity"]["bundle"]
        values[bundle] = value
        branch_path = ["payload", "comparisons", "contrasts", contrast, "probes", probe,
                       "cpu64", quantity, "direct_value"]
        audit_key = f"contrasts.{contrast}.{probe}.{quantity}.independent"
        audit_path = ["payload", "comparison_audits", audit_key]
        members.append({
            "bundle": bundle, "value": value, "defined": value is not None,
            "domain_reason": item["reason"], "branch_ref": copy.deepcopy(row["branch_ref"]),
            "branch_path": branch_path, "audit_ref": copy.deepcopy(row["audit_ref"]),
            "audit_path": audit_path if value is not None else None,
        })
    if primary:
        summary = response.primary_summary(values)
    else:
        _need(tuple(values) == SENSITIVITY, "sensitivity membership differs")
        value = values[SENSITIVITY[0]]
        summary = {"bundles": list(SENSITIVITY), "values": [value],
                   "defined_mask": [value is not None], "mean": None, "min": None,
                   "max": None, "reason": "sensitivity_not_pooled"}
    return {**summary, "members": members}


def _collect_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Assemble already-extracted primitive rows; public for bounded fixtures only."""
    membership = _expected_membership()
    _need(type(rows) is list and len(rows) == len(membership),
          "missing or extra scientific anchors")
    for row, expected in zip(rows, membership):
        _validate_row(row, expected)
    source_hashes = {(row["provenance"]["sources_tree_sha256"],
                      row["provenance"]["environment_tree_sha256"],
                      row["provenance"]["auditor_environment_sha256"]) for row in rows}
    _need(len(source_hashes) == 1, "heterogeneous source/runtime provenance cannot be pooled")
    primary_table, sensitivity_table = {}, {}
    for update in UPDATES:
        key = str(update)
        selected = [row for row in rows if row["identity"]["execution_role"] == "primary"
                    and row["identity"]["anchor_update"] == update]
        sensitivity = [row for row in rows if row["identity"]["execution_role"] == "sensitivity"
                       and row["identity"]["anchor_update"] == update]
        primary_table[key] = {
            contrast: {probe: {quantity: _cell(selected, contrast, probe, quantity, primary=True)
                               for quantity in QUANTITIES} for probe in PROBES}
            for contrast in CONTRASTS}
        sensitivity_table[key] = {
            contrast: {probe: {quantity: _cell(sensitivity, contrast, probe, quantity, primary=False)
                               for quantity in QUANTITIES} for probe in PROBES}
            for contrast in CONTRASTS}
    return {
        "schema_name": "i7_result_collection", "schema_version": 1, "profile": PROFILE,
        "membership": {"primary_bundles": list(PRIMARY),
                       "sensitivity_bundles": list(SENSITIVITY),
                       "anchor_updates": list(UPDATES), "contrasts": list(CONTRASTS),
                       "probes": list(PROBES), "quantities": list(QUANTITIES)},
        "anchors": copy.deepcopy(rows), "primary": primary_table,
        "sensitivity": sensitivity_table,
        "interpretation_scope": {
            "effect_claims": False, "inferential_statistics": False,
            "available_case_summaries": False, "sensitivity_pooled": False,
            "accuracy_measured": False,
        },
    }


def collect_validated_pairs(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    """Collect exactly 12 primary and four sensitivity validated artifact pairs.

    The list order is primary bundle-major then sensitivity, with ascending
    frozen anchor updates.  Missing/reordered anchors and any incomplete audit
    reject the whole collection before descriptive summaries are returned.
    """
    membership = _expected_membership()
    _need(type(pairs) is list and len(pairs) == len(membership),
          "missing or extra scientific artifact pairs")
    rows = [_extract_pair(pair, expected) for pair, expected in zip(pairs, membership)]
    return _collect_rows(rows)
