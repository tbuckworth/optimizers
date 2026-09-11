"""Actual synthetic sealed-input audit, retained adverse evidence and tampering."""
import copy
import os
import tempfile
import unittest
from unittest.mock import patch

if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
    raise RuntimeError("explicitly hide CUDA for audit fixtures")

import torch

import anchor_envelope as base
import artifact_envelopes as branches
import artifact_store as storage
import audit_diagnostics as diagnostics
import audit_envelope as audit
import branch_execution as execution
import source_capture as source
import state_core as state
import response_math as response
from envelope_fixture import fixture_context, warm_live


class AuditEnvelopeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def pipeline(self, check, mutation=None, domain_null=False):
        handle = {}
        with fixture_context(plan_handle=handle) as context, tempfile.TemporaryDirectory(prefix="i7-audit-envelope-") as directory:
            model, optimizer, observer = warm_live(context)
            if domain_null:
                with torch.no_grad():
                    for parameter in model.parameters():
                        parameter.zero_()
                    model[2].bias.copy_(torch.tensor([.4, -.4]))
                observer.V = torch.eye(26)[:, :2].clone()
                observer.S = torch.zeros(2, dtype=torch.float64)
                observer.grad_mean = torch.zeros(26)
                observer.max_orthogonality_error = 0.0
            with storage.ArtifactStore(directory, profile=context["profile"], min_filesystem_free_bytes=0) as store:
                transaction = source.capture_anchor_then_live_witness(model, optimizer, observer,
                    store=store, created_utc="2026-09-06T21:30:00Z", **context)
                original_write = store.write_tensor_tree
                def inject(name, value):
                    if mutation is not None and value.get("schema_name") == "i7_branch_results":
                        # Deliberately create correctly hash-bound adverse raw
                        # bytes after producer validation; this is a test fault.
                        mutation(value, transaction)
                    return original_write(name, value)
                with patch.object(store, "write_tensor_tree", side_effect=inject):
                    result = execution.execute_branches(store=store, created_utc="2026-09-06T21:30:01Z",
                                                        **transaction, **context)
                inputs = dict(**transaction, branch=result["artifact"], branch_receipt=result["receipt"], **context)
                before = state.capture_core(model, optimizer, observer, profile=context["profile"], completed_updates=5)
                check(store, inputs, handle)
                after = state.capture_core(model, optimizer, observer, profile=context["profile"], completed_updates=5)
                self.assertTrue(base.same_exact(before, after))
                self.assertFalse(torch.cuda.is_initialized())

    def seal(self, store, inputs, handle, **kwargs):
        return audit.audit_and_seal(store=store, plan_root=handle["root"], plan_name=handle["name"],
            created_utc="2026-09-06T21:30:02Z", **inputs, **kwargs)

    def test_tiny_current_norm_cannot_borrow_large_lagged_absolute_ceiling(self):
        g, c, l = torch.ones(26), torch.full((26,), 1e-20), torch.ones(26)
        produced = response.construct(g, c, l)
        candidate = dict(g={"value":g}, c={"value":c}, l={"value":l},
                         nc=response.norm(c), nl=response.norm(l), leverage=produced["leverage"])
        rows = {name:dict(status=row["status"], reason=row["reason"],
                         delivered_gradient={"value":row["gradient"]})
                for name,row in produced["branches"].items()}
        self.assertTrue(all(row["status"] == "pass" for row in audit._delivery(candidate, rows)["candidate_metadata"].values()))
        candidate["nc"] *= 2
        candidate["leverage"]["current_lagged_norm_ratio"] *= 2
        checked = audit._delivery(candidate, rows)["candidate_metadata"]
        self.assertEqual(checked["nc"]["status"], "fatal_validation")
        self.assertEqual(checked["leverage.current_lagged_norm_ratio"]["status"], "fatal_validation")

    def test_pinned_raw_pipeline_and_full_recomputation(self):
        def check(store, inputs, handle):
            result = self.seal(store, inputs, handle)
            artifact = result["artifact"]
            payload = artifact["payload"]
            self.assertEqual(payload["overall_status"], "pass", payload["fatal_failures"])
            self.assertEqual(artifact["schema_name"], "i7_anchor_numerical_audit")
            self.assertEqual(tuple(payload), audit.PAYLOAD_KEYS)
            counts = payload["exact_validation"]["completion"]
            self.assertTrue(counts["checks_complete"])
            self.assertEqual(counts["adam_parameter_checks"], 24)
            self.assertEqual(counts["expected_adam_parameter_checks"], 24)
            self.assertEqual(counts["adam_tensor_screens"], 72)
            self.assertEqual(counts["projection_checks"], 2)
            self.assertEqual(counts["delivery_rows"], 6)
            self.assertEqual(counts["measurement_counts"], counts["expected_measurement_counts"])
            self.assertEqual(len([key for key in payload["comparison_audits"] if key.endswith(".independent")]), 390)
            self.assertFalse(counts["scientific_execution_certified"])
            receipt = result["receipt"]
            loaded = storage.ArtifactStore.load_tensor_tree(store.root, receipt["name"],
                expected_size=receipt["size"], expected_sha256=receipt["sha256"])
            self.assertTrue(base.same_exact(loaded, artifact))
            audit_inputs = dict(**inputs, plan_reference=handle["reference"],
                                auditor_environment=payload["exact_validation"]["auditor_environment"])
            audit.validate_audit(loaded, **audit_inputs)
            corrupted = copy.deepcopy(loaded)
            corrupted["payload"]["exact_validation"]["completion"]["adam_parameter_checks"] = 23
            with self.assertRaises(audit.AuditEnvelopeError):
                audit.validate_audit(corrupted, **audit_inputs)
            self.assertFalse(store._terminal)
        self.pipeline(check)

    def test_actual_null_keeps_five_branches_and_unaffected_checks(self):
        def check(store, inputs, handle):
            payload = self.seal(store, inputs, handle)["artifact"]["payload"]
            self.assertEqual(payload["overall_status"], "pass", payload["fatal_failures"])
            self.assertEqual(payload["exact_validation"]["completion"]["adam_parameter_checks"], 20)
            restored = payload["branch_audits"]["result"]["restored"]
            self.assertEqual(restored["status"], "domain_undefined")
            self.assertEqual(restored["reason"], "positive_current_norm_zero_lagged_direction")
            self.assertEqual(len([key for key in payload["comparison_audits"] if key.endswith(".independent")]), 260)
        self.pipeline(check, domain_null=True)

    def test_adverse_saved_loss_is_sealed_and_terminals_without_retry(self):
        def mutate(value, _):
            value["payload"]["candidate_state"]["measurement_before"]["probes"]["batch_noisy"]["cpu64"]["before_ce"] += 1.0
        def check(store, inputs, handle):
            result = self.seal(store, inputs, handle)
            payload = result["artifact"]["payload"]
            self.assertEqual(payload["overall_status"], "fatal_validation")
            self.assertTrue(store._terminal)
            self.assertTrue(payload["measurement_audits"]["result"]["fatal_failures"])
            self.assertEqual(payload["measurement_audits"]["completed"],
                             payload["measurement_audits"]["result"]["audit_complete"])
            report = storage.ArtifactStore.inspect(store.root)
            self.assertTrue(any(row["name"] == result["receipt"]["name"] for row in report["receipts"]))
            with patch.object(audit.measurements, "audit", side_effect=AssertionError("retry recomputed")):
                with self.assertRaises(audit.AuditEnvelopeError):
                    self.seal(store, inputs, handle)
        self.pipeline(check, mutation=mutate)

    def test_adverse_saved_adam_moment_retains_coordinate_evidence(self):
        def mutate(value, transaction):
            payload = value["payload"]
            payload["branches"]["raw"]["optimizer_after"]["state"][0]["exp_avg"]["value"].reshape(-1)[0] += 1.0
            payload["audit_metadata"] = branches.expected_audit_metadata(payload["candidate_state"], payload["branches"], **transaction)
        def check(store, inputs, handle):
            result = self.seal(store, inputs, handle)
            payload = result["artifact"]["payload"]
            self.assertEqual(payload["exact_validation"]["saved_structure"]["status"], "pass")
            self.assertEqual(payload["branch_audits"]["status"], "fatal_validation")
            self.assertTrue(payload["branch_audits"]["completed"])
            self.assertEqual(payload["measurement_audits"]["status"], "not_run")
            record = payload["branch_audits"]["result"]["raw"]["parameters"]["0.weight"]["checks"]["moment"]
            self.assertEqual(diagnostics.unpack_indices(record["failing_flat_indices"]), [0])
            self.assertTrue(store._terminal)
        self.pipeline(check, mutation=mutate)

    def test_wrong_plan_handle_fails_before_numerical_audit(self):
        def check(store, inputs, handle):
            with patch.object(audit.measurements, "audit", side_effect=AssertionError("unexpected audit")):
                with self.assertRaises(audit.AuditEnvelopeError):
                    audit.audit_and_seal(store=store, plan_root=handle["root"], plan_name="absent.pt",
                        created_utc="2026-09-06T21:30:02Z", **inputs)
            self.assertTrue(store._terminal)
        self.pipeline(check)

    def test_guard_inside_measurement_stops_without_domain_null(self):
        def check(store, inputs, handle):
            labels = []
            def guard(label):
                labels.append(label)
                if label == "measurement_audit.begin":
                    raise RuntimeError("fixture resource guard")
            result = self.seal(store, inputs, handle, guard=guard)
            payload = result["artifact"]["payload"]
            self.assertIn("measurement_audit.begin", labels)
            self.assertEqual(payload["measurement_audits"]["error_code"], "ResourceGuardAbort")
            self.assertFalse(payload["measurement_audits"]["completed"])
            self.assertEqual(payload["branch_audits"]["status"], "pass")
            self.assertFalse(payload["exact_validation"]["completion"]["checks_complete"])
            # The old transient guard is not replayed. Only completed prefix
            # numerics and the declared incomplete suffix are revalidated.
            audit.validate_audit(result["artifact"], **inputs, plan_reference=handle["reference"],
                                 auditor_environment=payload["exact_validation"]["auditor_environment"])
            corrupted = copy.deepcopy(result["artifact"])
            corrupted["payload"]["branch_audits"]["result"]["raw"]["parameters"]["0.weight"]["checks"]["theta"]["max_absolute_error"] += 1.0
            with self.assertRaises(audit.AuditEnvelopeError):
                audit.validate_audit(corrupted, **inputs, plan_reference=handle["reference"],
                                     auditor_environment=payload["exact_validation"]["auditor_environment"])
            self.assertTrue(store._terminal)
        self.pipeline(check)

    def test_norm_leverage_unknown_keys_and_masks_cannot_pass(self):
        def check(store, inputs, handle):
            mutations = [
                lambda p: p["candidate_state"].__setitem__("nc", p["candidate_state"]["nc"]*2),
                lambda p: p["candidate_state"]["leverage"].__setitem__("norm_leverage", not p["candidate_state"]["leverage"]["norm_leverage"]),
                lambda p: p["candidate_state"]["leverage"].__setitem__("unknown", 0.0),
                lambda p: p["branches"]["raw"].__setitem__("assigned_gradient_null_mask", [True]*4),
                lambda p: p["branches"]["raw"]["optimizer_after"]["state"][0].__setitem__("parameter_name", "0.bias"),
            ]
            for index, mutate in enumerate(mutations):
                corrupted = copy.deepcopy(inputs["branch"])
                mutate(corrupted["payload"])
                # Consistently rehash the producer attestations so the test
                # exercises actual audit checks, not merely a stale hash.
                if index < 3:
                    p = corrupted["payload"]
                    p["audit_metadata"] = branches.expected_audit_metadata(p["candidate_state"], p["branches"],
                        anchor=inputs["anchor"], anchor_receipt=inputs["anchor_receipt"],
                        witness=inputs["witness"], witness_receipt=inputs["witness_receipt"])
                with self.subTest(case=index):
                    artifact = audit.build_audit(**dict(inputs, branch=corrupted),
                        plan_reference=handle["reference"], auditor_environment=inputs["environment"],
                        created_utc="2026-09-06T21:30:02Z")
                    self.assertEqual(artifact["payload"]["overall_status"], "fatal_validation")
                    self.assertTrue(artifact["payload"]["fatal_failures"])
                    if index < 3:
                        self.assertEqual(artifact["payload"]["exact_validation"]["saved_structure"]["status"], "pass")
                        self.assertEqual(artifact["payload"]["candidate_operator_audit"]["delivery"]["status"], "fatal_validation")
        self.pipeline(check)

    def test_post_seal_rng_fault_retains_audit_and_restores_caller(self):
        def check(store, inputs, handle):
            original = store.write_tensor_tree
            def faulty_write(name, value):
                receipt = original(name, value)
                if value.get("schema_name") == "i7_anchor_numerical_audit":
                    torch.rand(1)
                return receipt
            with patch.object(store, "write_tensor_tree", side_effect=faulty_write):
                with self.assertRaises(audit.AuditEnvelopeError):
                    self.seal(store, inputs, handle)
            self.assertTrue(store._terminal)
            report = storage.ArtifactStore.inspect(store.root)
            self.assertEqual(sum(row["name"].endswith("--independent-audit.pt") for row in report["receipts"]), 1)
        self.pipeline(check)

    def test_correct_plan_file_with_wrong_supplied_plan_fails_before_audit(self):
        def check(store, inputs, handle):
            altered = copy.deepcopy(inputs["plan"])
            altered["initialization_seed"] += 1
            with patch.object(audit.measurements, "audit", side_effect=AssertionError("unexpected audit")):
                with self.assertRaises(audit.AuditEnvelopeError):
                    self.seal(store, dict(inputs, plan=altered), handle)
            self.assertTrue(store._terminal)
        self.pipeline(check)


if __name__ == "__main__":
    unittest.main()
