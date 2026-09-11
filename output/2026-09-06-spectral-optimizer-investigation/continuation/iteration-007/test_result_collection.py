"""Prospective primitive fixtures for the fixed I7 result collector."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import os
from pathlib import Path
import unittest
from unittest import mock

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TMPDIR"] = "/tmp/spectral-experiment-artifacts"

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("i7_result_collection", HERE / "result_collection.py")
collector = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(collector)


def reference(name, digit):
    digest = digit * 64
    size = 100 + int(digit)
    raw = collector.storage._json_bytes({
        "schema": "i7_artifact_receipt_v1", "name": name, "size": size,
        "sha256": digest, "status": "complete", "encoding": "torch_weights_only",
    })
    return {
        "name": name, "status": "complete", "encoding": "torch_weights_only",
        "size_bytes": size, "sha256": digest,
        "receipt_name": "receipt-" + hashlib.sha256(name.encode("ascii")).hexdigest() + ".json",
        "receipt_size_bytes": len(raw), "receipt_sha256": hashlib.sha256(raw).hexdigest(),
    }


def leverage(weak=False):
    return {
        "current_norm": 1.0, "lagged_norm": 1.0,
        "signed_norm_difference": 0.0, "relative_norm_separation": 0.0,
        "current_lagged_norm_ratio": 1.0, "ratio_reason": None,
        "unit_direction_distance": 0.0 if weak else 1.0,
        "unit_direction_cosine": 1.0 if weak else 0.5, "direction_reason": None,
        "norm_leverage": False, "direction_leverage": not weak,
    }


def resolution(value):
    return {
        "producer_value": value, "auditor_value": value, "fixed_ceiling": 0.0,
        "achieved_discrepancy": 0.0, "formula_id": "i7_assembly_roundoff_v1",
        "audit_status": "pass", "producer_sign": int(value > 0) - int(value < 0),
        "auditor_sign": int(value > 0) - int(value < 0),
        "resolution_status": ("resolved_positive" if value > 0 else
                              "resolved_negative" if value < 0 else "unresolved_numerical"),
        "resolution_reason": None if value else "insufficient_fixed_margin",
    }


def primitive_row(role, bundle, update, *, weak=False, undefined=()):
    identity = collector.policy.identity(role, bundle, update)
    branch_id = collector.codec.artifact_id(identity, profile=collector.PROFILE,
                                             kind="branch-results")
    audit_id = collector.codec.artifact_id(identity, profile=collector.PROFILE,
                                            kind="independent-audit")
    branch_ref = reference(branch_id + ".pt", "1")
    audit_ref = reference(audit_id + ".pt", "2")
    domains = {name: {"defined": True, "reason": None} for name in collector.BRANCHES}
    contrasts = {}
    for contrast_index, name in enumerate(collector.CONTRASTS):
        is_defined = name not in undefined
        factors = {
            "direction": "weak" if weak and name.startswith("direction_") else "not_required",
            "norm": "weak" if weak and name.startswith("norm_") else "not_required",
        }
        if name == "interaction":
            factors = {"direction": "weak" if weak else "qualified",
                       "norm": "weak" if weak else "qualified"}
        probes = None
        if is_defined:
            probes = {}
            for probe_index, probe in enumerate(collector.PROBES):
                values = {kind: float((bundle - 71000) * 1000 + update + contrast_index * 10 +
                                      probe_index + kind_index / 10)
                          for kind_index, kind in enumerate(collector.QUANTITIES)}
                probes[probe] = {
                    "values": values,
                    "auditor_resolution": {kind: resolution(value)
                                           for kind, value in values.items()},
                    "native_concordance": {"abs_discrepancy": 0.0,
                                           "descriptive_ceiling": 1e-5,
                                           "status": "within_scale"},
                    "auxiliary_chunks": ([{"chunk_index": index} for index in range(10)]
                                         if probe == "auxiliary_clean" else []),
                }
        contrasts[name] = {"defined": is_defined,
                           "reason": None if is_defined else "domain_undefined_required_branch",
                           "factor_leverage": factors, "probes": probes}
    diagnostics = {
        "native_discordant_paths": [], "unresolved_geometry_paths": [],
        "weak_factor_contrasts": (["interaction"] if weak else []),
        "measurement_audit_path": ["payload", "measurement_audits", "result"],
        "comparison_audit_path": ["payload", "comparison_audits"],
    }
    return {
        "identity": identity, "branch_ref": branch_ref, "audit_ref": audit_ref,
        "leverage": leverage(weak), "branch_domain": domains, "contrasts": contrasts,
        "audit_diagnostics": diagnostics,
        "provenance": {"sources_tree_sha256": "3" * 64,
                       "environment_tree_sha256": "4" * 64,
                       "auditor_environment_sha256": "5" * 64,
                       "plan_probe_data_tree_sha256": "6" * 64},
    }


def all_rows():
    return [primitive_row(role, bundle, update, weak=(bundle == 71002 and update == 500))
            for role, bundle, update in collector._expected_membership()]


def envelope_pair(row):
    """Small schema double: primitive metadata only, never a scientific specimen."""
    identity = row["identity"]
    comparisons, comparison_audits = {}, {}
    for name, item in row["contrasts"].items():
        if not item["defined"]:
            comparisons[name] = {"coefficients": dict(collector.response.COEFFICIENTS[name]),
                "branch_defined_mask": {branch: True for branch in collector.BRANCHES},
                "defined": False, "reason": "domain_undefined_required_branch",
                "factor_leverage": copy.deepcopy(item["factor_leverage"]), "probes": None}
            continue
        probes = {}
        for probe, pitem in item["probes"].items():
            cpu = {kind: {"direct_value": value} for kind, value in pitem["values"].items()}
            chunks = []
            if probe == "auxiliary_clean":
                for index in range(10):
                    value = pitem["values"]["Y"] + index / 1000
                    chunks.append({"chunk_index": index, "start": index, "end": index + 1,
                        "count": 1, "input_sha256": "7" * 64, "label_sha256": "8" * 64,
                        "cpu64": {"Y": {"direct_value": value}},
                        "concordance": {"Y": {"abs_discrepancy": 0.0,
                            "descriptive_ceiling": 1e-5, "status": "within_scale"}}})
                    path = f"contrasts.{name}.auxiliary_clean.chunk{index}.Y.independent"
                    comparison_audits[path] = resolution(value)
            probes[probe] = {"cpu64": cpu, "concordance": {"Y": copy.deepcopy(
                pitem["native_concordance"])}, "auxiliary_chunks": chunks}
            for kind, value in pitem["values"].items():
                path = f"contrasts.{name}.{probe}.{kind}.independent"
                comparison_audits[path] = resolution(value)
        comparisons[name] = {"coefficients": dict(collector.response.COEFFICIENTS[name]),
            "branch_defined_mask": {branch: True for branch in collector.BRANCHES},
            "defined": True, "reason": None,
            "factor_leverage": copy.deepcopy(item["factor_leverage"]), "probes": probes}
    branch_id = collector.codec.artifact_id(identity, profile=collector.PROFILE,
                                             kind="branch-results")
    audit_id = collector.codec.artifact_id(identity, profile=collector.PROFILE,
                                            kind="independent-audit")
    branches = {name: {"status": "defined", "reason": None}
                for name in collector.BRANCHES}
    branch = {"identity": identity, "artifact_id": branch_id,
              "payload": {"candidate_state": {"leverage": copy.deepcopy(row["leverage"])},
                  "branches": branches,
                  "comparisons": {"branch_pairs": {}, "contrasts": comparisons},
                  "audit_metadata": {"producer_bindings": {
                      "sources_tree_sha256": "3" * 64,
                      "environment_tree_sha256": "4" * 64}}}}
    bound = collector._bound_branch_reference(branch, row["branch_ref"])
    stage = {"status": "pass", "completed": True, "error_code": None, "result": {}}
    measured = {"overall_status": "pass", "audit_complete": True, "fatal_failures": [],
                "native_discordant_paths": [], "unresolved_geometry_paths": [],
                "weak_factor_contrasts": row["audit_diagnostics"]["weak_factor_contrasts"]}
    audit_payload = {
        "input_bindings": {"branch_results_ref": bound, "sources_tree_sha256": "3" * 64,
                           "plan_probe_data_tree_sha256": "6" * 64},
        "exact_validation": {"auditor_environment": {"role": "fixture-double"},
                             "saved_structure": copy.deepcopy(stage),
                             "completion": {"checks_complete": True, "overall_status": "pass"}},
        "candidate_operator_audit": {"projections": copy.deepcopy(stage),
                                     "delivery": copy.deepcopy(stage)},
        "branch_audits": copy.deepcopy(stage),
        "measurement_audits": {**copy.deepcopy(stage), "result": measured},
        "comparison_audits": comparison_audits, "overall_status": "pass",
        "fatal_failures": [],
    }
    audit = {"identity": copy.deepcopy(identity), "artifact_id": audit_id,
             "payload": audit_payload}
    return {"branch": branch, "branch_ref": copy.deepcopy(row["branch_ref"]),
            "audit": audit, "audit_ref": copy.deepcopy(row["audit_ref"])}


class ResultCollectionTests(unittest.TestCase):
    def test_complete_primary_math_sensitivity_separation_and_weak_leverage(self):
        result = collector._collect_rows(all_rows())
        cell = result["primary"]["500"]["ordering"]["batch_noisy"]["Y"]
        expected = [float((bundle - 71000) * 1000 + 500) for bundle in collector.PRIMARY]
        self.assertEqual(cell["values"], expected)
        self.assertEqual(cell["mean"], sum(expected) / 3)
        self.assertEqual(cell["min"], min(expected))
        self.assertEqual(cell["max"], max(expected))
        sensitive = result["sensitivity"]["500"]["ordering"]["batch_noisy"]["Y"]
        self.assertEqual(sensitive["bundles"], [71901])
        self.assertIsNone(sensitive["mean"])
        self.assertEqual(sensitive["reason"], "sensitivity_not_pooled")
        weak = next(row for row in result["anchors"] if row["identity"]["bundle"] == 71002
                    and row["identity"]["anchor_update"] == 500)
        self.assertEqual(weak["leverage"]["unit_direction_distance"], 0.0)

    def test_incomplete_predeclared_mask_preserves_negative_and_resolution(self):
        rows = all_rows()
        target = next(row for row in rows if row["identity"]["bundle"] == 71002 and
                      row["identity"]["anchor_update"] == 101)
        target["branch_domain"]["restored"] = {
            "defined": False, "reason": "positive_current_norm_zero_lagged_direction"}
        for name, coefficients in collector.response.COEFFICIENTS.items():
            if "restored" in coefficients:
                target["contrasts"][name].update(
                    defined=False, reason="domain_undefined_required_branch", probes=None)
        first = rows[0]["contrasts"]["ordering"]["probes"]["batch_noisy"]
        first["values"]["Y"] = -2.0
        first["auditor_resolution"]["Y"] = resolution(-2.0)
        result = collector._collect_rows(rows)
        masked = result["primary"]["101"]["interaction"]["batch_noisy"]["Y"]
        self.assertEqual(masked["defined_mask"], [True, False, True])
        self.assertIsNone(masked["mean"])
        self.assertEqual(masked["reason"], "incomplete_predeclared_primary_mask")
        self.assertEqual(result["primary"]["101"]["ordering"]["batch_noisy"]["Y"]
                         ["values"][0], -2.0)
        self.assertEqual(first["auditor_resolution"]["Y"]["resolution_status"],
                         "resolved_negative")

    def test_missing_or_reordered_anchor_is_invalid_not_a_mask(self):
        with self.assertRaisesRegex(collector.CollectionError, "missing or extra"):
            collector._collect_rows(all_rows()[:-1])
        rows = all_rows()
        rows[0], rows[1] = rows[1], rows[0]
        with self.assertRaisesRegex(collector.CollectionError, "identity differs"):
            collector._collect_rows(rows)

    def test_public_pair_wiring_membership_and_exact_ref_binding(self):
        rows = all_rows()
        probe = rows[0]["contrasts"]["ordering"]["probes"]["batch_noisy"]
        probe["values"]["Y"] = -2.0
        probe["auditor_resolution"]["Y"] = resolution(-2.0)
        probe["values"]["D"] = 0.0
        probe["auditor_resolution"]["D"] = resolution(0.0)
        pairs = [envelope_pair(row) for row in rows]
        with mock.patch.object(collector.envelopes, "validate_common", return_value=None):
            result = collector.collect_validated_pairs(pairs)
        self.assertEqual(len(result["anchors"]), 16)
        self.assertEqual(result["anchors"][0]["branch_ref"], rows[0]["branch_ref"])
        self.assertEqual(result["anchors"][-1]["identity"]["execution_role"], "sensitivity")
        extracted = result["anchors"][0]["contrasts"]["ordering"]["probes"]["batch_noisy"]
        self.assertEqual(extracted["auditor_resolution"]["Y"]["resolution_status"],
                         "resolved_negative")
        self.assertEqual(extracted["auditor_resolution"]["D"]["resolution_status"],
                         "unresolved_numerical")
        member = result["primary"]["101"]["ordering"]["batch_noisy"]["Y"]["members"][0]
        self.assertEqual(member["audit_path"], ["payload", "comparison_audits",
            "contrasts.ordering.batch_noisy.Y.independent"])

        changed = copy.deepcopy(pairs)
        changed[0]["audit"]["payload"]["input_bindings"]["branch_results_ref"]["sha256"] = "9" * 64
        with mock.patch.object(collector.envelopes, "validate_common", return_value=None), \
                self.assertRaisesRegex(collector.CollectionError, "not bound"):
            collector.collect_validated_pairs(changed)

    def test_failed_audit_refuses_collection_before_assembly(self):
        pairs = [envelope_pair(row) for row in all_rows()]
        pairs[0]["audit"]["payload"]["overall_status"] = "fatal_validation"
        pairs[0]["audit"]["payload"]["fatal_failures"] = [{"stage": "measurements"}]
        with mock.patch.object(collector.envelopes, "validate_common", return_value=None), \
                mock.patch.object(collector, "_collect_rows") as assemble, \
                self.assertRaisesRegex(collector.CollectionError, "failed audit"):
            collector.collect_validated_pairs(pairs)
        assemble.assert_not_called()

    def test_heterogeneous_provenance_refuses_pooling(self):
        rows = all_rows()
        rows[-1]["provenance"]["environment_tree_sha256"] = "9" * 64
        with self.assertRaisesRegex(collector.CollectionError, "heterogeneous"):
            collector._collect_rows(rows)

    def test_auditor_resolution_exact_three_statuses_and_consistency(self):
        rows = {}
        for key, value in (("positive", 2.0), ("negative", -3.0), ("unresolved", 0.0)):
            rows[key] = resolution(value)
            self.assertEqual(collector._audit_resolution(rows, key, value), rows[key])
        bad = copy.deepcopy(rows)
        bad["positive"]["producer_sign"] = -1
        with self.assertRaisesRegex(collector.CollectionError, "invalid auditor resolution"):
            collector._audit_resolution(bad, "positive", 2.0)
        bad = copy.deepcopy(rows)
        bad["positive"]["fixed_ceiling"] = 3.0
        with self.assertRaisesRegex(collector.CollectionError, "invalid auditor resolution"):
            collector._audit_resolution(bad, "positive", 2.0)
        bad = copy.deepcopy(rows)
        bad["positive"]["producer_value"] = bad["positive"]["auditor_value"] = -2.0
        with self.assertRaisesRegex(collector.CollectionError, "invalid auditor resolution"):
            collector._audit_resolution(bad, "positive", -2.0)
        bad = copy.deepcopy(rows)
        bad["positive"]["extra"] = "not allowed"
        with self.assertRaisesRegex(collector.CollectionError, "exact ordered keys"):
            collector._audit_resolution(bad, "positive", 2.0)
        concordance = {"abs_discrepancy": 0.0, "descriptive_ceiling": 1e-5,
                       "status": "within_scale"}
        self.assertEqual(collector._concordance(concordance), concordance)
        wrong_status = copy.deepcopy(concordance)
        wrong_status["status"] = "discordant"
        with self.assertRaisesRegex(collector.CollectionError, "native concordance invalid"):
            collector._concordance(wrong_status)
        concordance["extra"] = []
        with self.assertRaisesRegex(collector.CollectionError, "exact ordered keys"):
            collector._concordance(concordance)

    def test_contrast_domain_is_recomputed_from_branch_membership(self):
        rows = all_rows()
        rows[0]["branch_domain"]["lagged"] = {
            "defined": False, "reason": "positive_current_norm_zero_lagged_direction"}
        with self.assertRaisesRegex(collector.CollectionError, "required branches"):
            collector._collect_rows(rows)


if __name__ == "__main__":
    unittest.main()
