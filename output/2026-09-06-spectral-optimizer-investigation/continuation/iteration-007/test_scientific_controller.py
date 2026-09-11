"""Later-phase integration on primitive CPU fixtures, never scientific results."""
from __future__ import annotations

from contextlib import contextmanager
import copy
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
    raise RuntimeError("explicitly hidden CUDA required")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import torch
import artifact_store as storage
import identity_codec as codec
import native_control as control
import native_controller as development
import native_phase_policy as policy
import phase_transition as transition
import scientific_controller as subject
import test_native_controller as development_fixture
import test_phase_transition as fixture


class ScientificControllerTests(fixture.PhaseFixture, unittest.TestCase):
    def setUp(self):
        self.started = time.monotonic()
        torch.set_num_threads(1)
        self.checkpoint()

    def tearDown(self):
        self.checkpoint()

    def build(self, acquired, work=None, **overrides):
        arguments = dict(entry_process_id=os.getpid(), data_directory="unused-no-idx-read",
            expected_files={}, utc_now=lambda:fixture.UTC,
            fixture_work=development_fixture.NativeControllerTests.write_contract if work is None else work)
        arguments.update(overrides)
        return subject.ScientificController(acquired, **arguments)

    @contextmanager
    def acquisition(self, target="primary"):
        with self.outer() as outer:
            prior, sources, native_environment = self.development_boundary(outer)
            for phase in transition.PHASES:
                environment = self.audit_environment(native_environment) if phase == "audit" else native_environment
                acquired = self.acquire(prior, phase, sources, environment)
                try:
                    if phase == target:
                        yield acquired
                        return
                    runner = self.build(acquired)
                    runner.run_phase()
                    prior = subject.prior_boundary(phase, runner.close_phase())
                finally:
                    acquired["store"].close()
            self.fail("unknown test phase")

    def inspect_closed(self, acquired):
        acquired["store"].close()
        return control.inspect_native_attempt(acquired["store"].root, acquired["journal_pin"],
            expected_store_profile=storage.MLP_FIXTURE, expected_evidence_kind=fixture.KIND)

    def test_import_default_and_unknown_cli_are_inert(self):
        with tempfile.TemporaryDirectory(dir=fixture.BIG_TMP) as directory:
            module = str(HERE / "scientific_controller.py")
            code = ("import importlib.util,sys,os;sys.path.insert(0,os.path.dirname(sys.argv[1]));"
                    "s=importlib.util.spec_from_file_location('x',sys.argv[1]);"
                    "m=importlib.util.module_from_spec(s);s.loader.exec_module(m);print('torch' in sys.modules)")
            result = subprocess.run([sys.executable, "-c", code, module], cwd=directory,
                capture_output=True, text=True, check=True, timeout=10)
            self.assertEqual(result.stdout, "False\n")
            result = subprocess.run([sys.executable, module], cwd=directory,
                capture_output=True, text=True, check=True, timeout=10)
            self.assertEqual(result.stdout, '{"status":"inert"}\n')
            result = subprocess.run([sys.executable, module, "--native"], cwd=directory,
                capture_output=True, text=True, timeout=10)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(os.listdir(directory), [])

    def test_actual_controllers_close_all_fixed_phases_without_automatic_next(self):
        initial = development_fixture.NativeControllerTests()
        initial.setUp()
        try:
            with initial.instance() as (dev, boot, development_calls):
                dev.run_development()
                prior = subject.prior_boundary("development", dev.close_development())
                calls = []
                for phase, count in (("primary", 44), ("sensitivity", 56), ("audit", 90)):
                    environment = self.audit_environment(dev.environment) if phase == "audit" else dev.environment
                    acquired = self.acquire(prior, phase, dev.sources, environment)
                    def operation(row, held, guard):
                        calls.append(row["operation"])
                        self.checkpoint()
                        return initial.write_contract(row, held, guard)
                    try:
                        runner = self.build(acquired, operation)
                        guard = runner.guard
                        result = runner.run_phase()
                        self.assertIs(runner.guard, guard)
                        self.assertEqual(result["status"], phase + "_complete")
                        self.assertEqual(result["phase_records"], count)
                        self.assertFalse(result["automatic_next_phase_enabled"])
                        self.assertFalse(result["scientific_execution_certified"])
                        with self.assertRaisesRegex(development.ControllerError, "no replay"):
                            runner.run_phase()
                        handoff = runner.close_phase()
                        self.assertTrue(runner.store._closed)
                        self.assertFalse(handoff["process_exit_verified"])
                        self.assertFalse(handoff["execution_authorized"])
                        self.assertEqual(handoff["inspection"]["phase_records"], count)
                        prior = subject.prior_boundary(phase, handoff)
                        self.assertEqual(handoff["inspection"]["eligible_for_writable_reopen"], phase != "audit")
                        if phase == "audit":
                            self.assertEqual(handoff["inspection"]["status"], "complete")
                            self.assertIsNone(result["next_operation"])
                            final_records = runner._records
                    finally:
                        acquired["store"].close()
                expected = [row["operation"] for row in policy.schedule()
                            if row["kind"] == "work" and row["phase"] != "development"]
                self.assertEqual(calls, expected)
                self.assertEqual(calls[:4], ["scientific_plans", *[f"primary.source.b{b}" for b in policy.PRIMARY]])
                self.assertEqual(len(final_records), 90)
                policy.validate_transcript(final_records, evidence_kind=fixture.KIND, require_complete=True)
                self.assertEqual(len([row for row in final_records if row["event"] == "sealed"]), 41)
                self.assertEqual(len([name for name in calls if name.startswith("audit.")]), 16)
        finally:
            initial.tearDown()

    def test_constructor_reauthenticates_exact_acquisition_and_marker(self):
        for mode in ("extra_key", "changed_marker"):
            with self.subTest(mode=mode), self.acquisition() as acquired:
                bad = dict(acquired)
                if mode == "extra_key":
                    bad["claimed_permission"] = True
                else:
                    pin = copy.deepcopy(bad["transition_pin"])
                    path = Path(pin["path"])
                    value = codec.json_loads(path.read_bytes(), max_bytes=transition.TRANSITION_MAX)
                    value["previous_boundary_ref"]["sha256"] = "0" * 64
                    raw = codec.json_bytes(value)
                    path.write_bytes(raw)  # Deliberate mutation of this disposable fixture only.
                    pin.update(size_bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
                    bad["transition_pin"] = pin
                with self.assertRaises(transition.TransitionError):
                    self.build(bad)
                self.assertTrue(acquired["store"]._terminal)
                self.assertEqual(len(storage.ArtifactStore.inspect(acquired["store"].root)["failures"]), 1)

    def test_entry_pid_and_fixture_callback_are_mandatory(self):
        for overrides in ({"entry_process_id":os.getpid() + 1}, {"fixture_work":None}):
            with self.subTest(overrides=overrides), self.acquisition() as acquired:
                with self.assertRaises(development.ControllerError):
                    self.build(acquired, **overrides)
                self.assertTrue(acquired["store"]._terminal)
                self.assertEqual(len(acquired["records"]), 10)

    def test_preparation_failure_keeps_started_and_never_reaches_primary_go(self):
        with self.acquisition() as acquired:
            calls = []
            def stop(row, store, guard):
                calls.append(row["operation"])
                store.write_tensor_tree(row["artifacts"][0], {"synthetic_partial_plan":True})
                raise RuntimeError("fixture preparation stop")
            runner = self.build(acquired, stop)
            with self.assertRaisesRegex(RuntimeError, "preparation stop"):
                runner.run_phase()
            self.assertEqual(calls, ["scientific_plans"])
            self.assertEqual((runner._records[-1]["event"], runner._records[-1]["operation"]),
                             ("started", "scientific_plans"))
            self.assertFalse(any(row["operation"] == "primary_go" for row in runner._records))
            observed = self.inspect_closed(acquired)
            self.assertFalse(observed["eligible_for_writable_reopen"])
            self.assertFalse(observed["can_resume_incomplete"])

    def test_primary_guard_is_not_reset_between_preparation_and_go(self):
        with self.acquisition() as acquired:
            guard = acquired["guard"]
            calls = []
            def work(row, store, checkpoint):
                calls.append(row["operation"])
                return development_fixture.NativeControllerTests.write_contract(row, store, checkpoint)
            runner = self.build(acquired, work)
            original = guard.check
            def stop(stage):
                if stage == "scientific_controller.context.pre" and runner._cursor == 7:
                    raise RuntimeError("same guard stopped before GO")
                return original(stage)
            with mock.patch.object(guard, "check", stop):
                with self.assertRaisesRegex(RuntimeError, "same guard"):
                    runner.run_phase()
            self.assertIs(runner.guard, guard)
            self.assertEqual(calls, ["scientific_plans"])
            self.assertEqual(len(runner._records), 12)
            self.assertEqual(runner._records[-1]["event"], "sealed")
            self.assertTrue(runner.store._terminal)

    def test_audit_terminal_report_is_retained_without_false_sealed_record(self):
        with self.acquisition("audit") as acquired:
            def adverse(row, store, guard):
                development_fixture.NativeControllerTests.write_contract(row, store, guard)
                store._fail("fixture_fatal_audit", "fixture_fatal_audit")
                return {"invariant_pass":False}
            runner = self.build(acquired, adverse)
            with self.assertRaises(storage.StoreError):
                runner.run_phase()
            self.assertEqual(runner._records[-1]["event"], "started")
            self.assertEqual(runner._records[-1]["operation"], "audit.primary.b71001.u101")
            report = storage.ArtifactStore.inspect(runner.store.root)
            self.assertEqual(len(report["failures"]), 1)
            audit_name = policy.artifact_name("primary", 71001, 101, "independent-audit")
            self.assertIn(audit_name, report["files"])
            self.assertFalse(any(row["event"] == "sealed" and row["phase"] == "audit"
                                 for row in runner._records))
            observed = self.inspect_closed(acquired)
            self.assertEqual(observed["status"], "store_failed_without_policy_record")
            self.assertFalse(observed["can_resume_incomplete"])

    def test_post_close_failure_retains_external_terminal_with_phase_metadata(self):
        with self.acquisition() as acquired:
            runner = self.build(acquired)
            runner.run_phase()
            files = set(runner.store._indexed)
            original = runner.guard.check
            def stop(stage):
                if stage == "controller.handoff.post_close":
                    raise RuntimeError("fixture after scientific close")
                return original(stage)
            with mock.patch.object(runner.guard, "check", stop):
                with mock.patch.object(storage.ArtifactStore, "reopen", side_effect=AssertionError("no reopen")):
                    with self.assertRaisesRegex(RuntimeError, "after scientific close"):
                        runner.close_phase()
            self.assertTrue(runner.store._closed)
            self.assertEqual(runner.handoff_failure_metadata_status, "retained")
            self.assertEqual(set(storage.ArtifactStore.inspect(runner.store.root)["files"]), files)
            observed = self.inspect_closed(acquired)
            self.assertEqual(observed["status"], "launcher_failed_with_root")
            self.assertFalse(observed["eligible_for_writable_reopen"])

    def test_native_adapter_context_calls_without_scientific_data_or_draws(self):
        import native_inputs
        import native_source
        import native_branches
        import native_audit
        import verified_plan_load
        with self.acquisition() as acquired:
            runner = self.build(acquired)
            runner.run_phase()  # Only primitive fixture payloads, including plan filenames.
            identity = policy.identity("primary", 71001, 101)
            _, plan_ref = runner._read(policy.plan_name(71001))
            plan = object()
            data = dict(images_bytes=b"synthetic-only", labels_bytes=b"synthetic-only", expected_files={})
            runner.kind = "native_producer_attestation"
            with mock.patch.object(verified_plan_load, "load_verified_plan_from_store",
                                   return_value={"plan":plan, "artifact":plan_ref}) as load:
                with mock.patch.object(native_inputs, "read_training_idx", return_value=data) as read:
                    native_context = runner._context(identity)
                    runner.phase = "audit"
                    runner.environment = self.audit_environment(runner.source_environment)
                    audit_context = runner._context(identity)
            self.assertEqual(load.call_count, 2)
            read.assert_called_once()
            self.assertIs(native_context["plan"], plan)
            self.assertEqual(native_context["native_device"], "cuda:0")
            self.assertNotIn("native_device", audit_context)
            self.assertEqual(audit_context["environment"]["runtime_role"], "native_source")
            self.assertEqual(audit_context["auditor_environment"]["runtime_role"], "cpu_audit")
            source_result = {"completion":{}, "completion_receipt":{}, "invariant_pass":False, "adverse_reason":"fixture"}
            for operation, module, method, context, result in (
                ("primary.source.b71001", native_source, "run_source", native_context, source_result),
                ("primary.branch.b71001.u101", native_branches, "run_branches", native_context,
                 {"artifact":{}, "receipt":{}}),
                ("audit.primary.b71001.u101", native_audit, "run_audit", audit_context,
                 {"artifact":{"payload":{"overall_status":"fatal_validation"}}, "receipt":{}})):
                row = next(row for row in policy.schedule() if row["operation"] == operation)
                with mock.patch.object(runner, "_context", return_value=context):
                    with mock.patch.object(module, method, autospec=True, return_value=result) as called:
                        produced = runner._produce(row)
                        expected = dict(context, capture_mode="capture_on") if method == "run_source" else context
                        called.assert_called_once_with(**expected)
                        self.assertEqual(produced["invariant_pass"], method == "run_branches")
            runner.kind = fixture.KIND

    def test_scientific_completion_binding_requires_all_four_actual_anchor_refs(self):
        with self.acquisition() as acquired:
            runner = self.build(acquired)
            runner.run_phase()
            refs = next(row["artifacts"] for row in runner._records
                        if row["operation"] == "primary.source.b71001" and row["event"] == "sealed")
            _, plan_ref = runner._read(policy.plan_name(71001))
            indexed = {ref["name"]:ref for ref in refs}
            value = dict(plan_ref=plan_ref, anchor_witness_refs=[dict(anchor_update=update,
                anchor_ref=indexed[policy.artifact_name("primary", 71001, update, "anchor")],
                witness_ref=indexed[policy.artifact_name("primary", 71001, update, "source-witness")])
                for update in policy.ANCHORS])
            runner._bind_completion_refs(value, refs, plan_ref, role="primary", bundle=71001, anchors=policy.ANCHORS)
            bad = copy.deepcopy(value)
            bad["anchor_witness_refs"][-1]["witness_ref"]["receipt_sha256"] = "0" * 64
            with self.assertRaisesRegex(development.ControllerError, "actual saved output"):
                runner._bind_completion_refs(bad, refs, plan_ref, role="primary", bundle=71001, anchors=policy.ANCHORS)

    def test_wrong_phase_methods_and_pilot_membership_are_refused(self):
        with self.acquisition() as acquired:
            runner = self.build(acquired)
            with self.assertRaises(development.ControllerError):
                runner.run_development()
            with self.assertRaises(development.ControllerError):
                runner.prepare_primary()
            with self.assertRaises(development.ControllerError):
                runner._member("audit.pilot.b71990.u101")
            with self.assertRaisesRegex(development.ControllerError, "untouched completed"):
                runner.close_phase()
            self.assertEqual(len(runner._records), 10)
            self.assertFalse(runner.store._terminal)


if __name__ == "__main__":
    unittest.main()
