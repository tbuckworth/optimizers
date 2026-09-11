"""Fixed-schedule controller contracts only; no study plans, MNIST or CUDA."""
from __future__ import annotations

from contextlib import contextmanager
import copy
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
    raise RuntimeError("explicitly hidden CUDA required")

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import artifact_store as storage
import identity_codec as codec
import native_control as control
import native_controller as controller
import native_phase_policy as policy
import runtime_guard as runtime
import test_native_control as bootstrap_fixture


class NativeControllerTests(unittest.TestCase):
    def setUp(self):
        self.fixture = bootstrap_fixture.NativeControlTests()
        self.fixture.setUp()

    def tearDown(self):
        self.fixture.tearDown()

    @staticmethod
    def phase_guard(phase):
        clock = {"wall":2.0, "cpu":2.0}
        def read(key):
            clock[key] += .001
            return clock[key]
        return runtime.RuntimeGuard(phase, profile=runtime.FIXTURE,
            entry_wall_origin=1.0, entry_cpu_origin=1.0, entry_process_id=os.getpid(),
            wall_clock=lambda:read("wall"), cpu_clock=lambda:read("cpu"))

    @staticmethod
    def write_contract(row, store, guard):
        for name in row["artifacts"]:
            guard("contract.write.pre")
            value = {"synthetic_contract_fixture":True, "operation":row["operation"], "name":name}
            if name.endswith(".json"):
                store.write_bytes(name, codec.json_bytes(value))
            else:
                store.write_tensor_tree(name, value)
            guard("contract.write.post")
        return {"invariant_pass":True}

    @contextmanager
    def instance(self, work=None):
        with self.fixture.outer() as outer:
            boot, _ = self.fixture.bootstrap(outer)
            store = boot["store"]
            calls = []
            def operation(row, held, guard):
                calls.append(row["operation"])
                self.fixture.guard()
                return (self.write_contract if work is None else work)(row, held, guard)
            try:
                runner = controller.NativeController(boot, entry_process_id=os.getpid(),
                    data_directory="unused-no-idx-read", expected_files={},
                    utc_now=lambda:bootstrap_fixture.UTC, fixture_work=operation,
                    fixture_guard=self.phase_guard("development"))
                yield runner, boot, calls
            finally:
                store.close()

    @staticmethod
    def inspect_closed(runner, boot):
        runner.store.close()
        return control.inspect_native_attempt(runner.store.root, boot["journal_pin"],
            expected_store_profile=storage.MLP_FIXTURE,
            expected_evidence_kind=bootstrap_fixture.KIND)

    def test_import_default_and_unknown_cli_inert(self):
        with tempfile.TemporaryDirectory(dir=bootstrap_fixture.BIG_TMP) as directory:
            module = str(HERE / "native_controller.py")
            code = ("import importlib.util,sys; s=importlib.util.spec_from_file_location('x',sys.argv[1]);"
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

    def test_development_order_actual_refs_and_boundary(self):
        with self.instance() as (runner, boot, calls):
            result = runner.run_development()
            self.assertEqual(calls, ["development.plan", "development.source_pair",
                "development.branch.u101", "development.branch.u200"])
            self.assertEqual(result["status"], "development_complete")
            self.assertEqual(result["phase_records"], 10)
            self.assertEqual(result["next_operation"], "scientific_plans")
            self.assertFalse(result["scientific_execution_certified"])
            records = copy.deepcopy(runner._records)
            self.assertEqual(len(records[4]["artifacts"]), 7)
            for record in records:
                for ref in record["artifacts"]:
                    self.assertEqual(runner._read(ref["name"])[1], ref)
            with self.assertRaisesRegex(controller.ControllerError, "no replay"):
                runner.run_development()
            observed = self.inspect_closed(runner, boot)
            self.assertEqual(observed["status"], "sealed_boundary")
            self.assertTrue(observed["eligible_for_writable_reopen"])

    def test_synthetic_preparation_uses_retained_primary_guard_then_stops(self):
        with self.instance() as (runner, boot, calls):
            runner.run_development()
            guard = self.phase_guard("primary")
            result = runner.prepare_primary(fixture_guard=guard)
            self.assertIs(runner.guard, guard)
            self.assertEqual(result["status"], "primary_go_required")
            self.assertEqual(calls[-1], "scientific_plans")
            self.assertEqual(result["phase_records"], 12)
            self.assertEqual(len(runner._records[-1]["artifacts"]), 4)
            before = result["resources"]["last_record"]["wall_seconds"]
            self.assertGreater(runner.summary()["resources"]["last_record"]["wall_seconds"], before)
            self.assertFalse(result["later_phase_execution_enabled"])
            with self.assertRaisesRegex(controller.ControllerError, "no replay"):
                runner.prepare_primary(fixture_guard=self.phase_guard("primary"))
            self.assertEqual(calls.count("scientific_plans"), 1)

    def test_valid_unequal_pair_seals_all_evidence_then_stops_before_branches(self):
        def adverse(row, store, guard):
            self.write_contract(row, store, guard)
            return {"invariant_pass":row["operation"] != "development.source_pair"}
        with self.instance(adverse) as (runner, boot, calls):
            with self.assertRaisesRegex(controller.ControllerError, "pilot invariant failed"):
                runner.run_development()
            self.assertEqual(calls, ["development.plan", "development.source_pair"])
            self.assertEqual((runner._records[-1]["operation"], runner._records[-1]["event"]),
                             ("development.source_pair", "sealed"))
            self.assertEqual(len(runner._records[-1]["artifacts"]), 7)
            report = storage.ArtifactStore.inspect(runner.store.root)
            self.assertTrue(report["terminal"])
            self.assertEqual(len(report["failures"]), 1)
            observed = self.inspect_closed(runner, boot)
            self.assertFalse(observed["eligible_for_writable_reopen"])
            self.assertEqual(observed["status"], "store_failed_without_policy_record")

    def test_partial_producer_failure_keeps_started_and_one_terminal_record(self):
        def interrupted(row, store, guard):
            if row["operation"] == "development.source_pair":
                store.write_tensor_tree(row["artifacts"][0], {"synthetic_contract_fixture":True})
                raise RuntimeError("injected interruption")
            return self.write_contract(row, store, guard)
        with self.instance(interrupted) as (runner, boot, calls):
            with self.assertRaisesRegex(RuntimeError, "injected"):
                runner.run_development()
            self.assertEqual(runner._records[-1]["event"], "started")
            self.assertEqual(len(storage.ArtifactStore.inspect(runner.store.root)["failures"]), 1)
            observed = self.inspect_closed(runner, boot)
            self.assertEqual(observed["status"], "store_failed_without_policy_record")
            self.assertFalse(observed["can_resume_incomplete"])

    def test_post_boundary_guard_failure_overrides_boundary(self):
        with self.instance() as (runner, boot, calls):
            original = runner.guard.check
            def check(stage):
                if stage == "controller.record.post_write" and len(runner._records) == 10:
                    raise RuntimeError("injected post-boundary guard failure")
                return original(stage)
            with mock.patch.object(runner.guard, "check", check):
                with self.assertRaisesRegex(RuntimeError, "post-boundary"):
                    runner.run_development()
            self.assertEqual(runner._records[-1]["event"], "boundary")
            observed = self.inspect_closed(runner, boot)
            self.assertEqual(observed["status"], "store_failed_without_policy_record")
            self.assertFalse(observed["eligible_for_writable_reopen"])

    def test_missing_extra_or_nonboolean_producer_output_refused(self):
        for mode in ("missing", "extra", "nonboolean"):
            with self.subTest(mode=mode):
                def bad(row, store, guard):
                    if mode != "missing":
                        self.write_contract(row, store, guard)
                    if mode == "extra":
                        store.write_bytes("unregistered.json", b"{}")
                    return {"invariant_pass":1 if mode == "nonboolean" else True}
                with self.instance(bad) as (runner, boot, calls):
                    with self.assertRaises(controller.ControllerError):
                        runner.run_development()
                    self.assertEqual(calls, ["development.plan"])
                    self.assertEqual(runner._records[-1]["event"], "started")
                    self.assertTrue(runner.store._terminal)

    def test_entry_pid_and_native_preparation_are_closed(self):
        with self.instance() as (runner, boot, calls):
            original = runner.kind
            runner.kind = "native_producer_attestation"
            with self.assertRaisesRegex(controller.ControllerError, "native preparation disabled"):
                runner.prepare_primary()
            runner.kind = original
            self.assertEqual(calls, [])
            with mock.patch.object(controller.os, "getpid", return_value=os.getpid() + 1):
                with self.assertRaisesRegex(controller.ControllerError, "cannot cross processes"):
                    runner.run_development()
            self.assertTrue(runner.store._terminal)

    def test_completion_nested_refs_bind_actual_files_not_just_valid_shapes(self):
        with self.instance() as (runner, boot, calls):
            runner.run_development()
            refs = runner._records[4]["artifacts"]
            _, plan_ref = runner._read(policy.plan_name(71990))
            indexed = {ref["name"]:ref for ref in refs}
            pairs = []
            for update in policy.PILOT_ANCHORS:
                pairs.append(dict(anchor_update=update,
                    anchor_ref=copy.deepcopy(indexed[policy.artifact_name("pilot", 71990, update, "anchor")]),
                    witness_ref=copy.deepcopy(indexed[policy.artifact_name("pilot", 71990, update, "source-witness")])))
            # Deliberately only the binding subset, not a scientific completion.
            value = dict(plan_ref=plan_ref, anchor_witness_refs=pairs)
            runner._bind_completion_refs(value, refs, plan_ref)
            bad = copy.deepcopy(value)
            bad["anchor_witness_refs"][0]["anchor_ref"]["sha256"] = "0" * 64
            with self.assertRaisesRegex(controller.ControllerError, "actual saved output"):
                runner._bind_completion_refs(bad, refs, plan_ref)
            bad = copy.deepcopy(value)
            bad["anchor_witness_refs"].reverse()
            with self.assertRaisesRegex(controller.ControllerError, "anchor order"):
                runner._bind_completion_refs(bad, refs, plan_ref)
            bad = copy.deepcopy(value)
            bad["plan_ref"]["sha256"] = "0" * 64
            with self.assertRaisesRegex(controller.ControllerError, "saved plan"):
                runner._bind_completion_refs(bad, refs, plan_ref)

    def test_native_context_api_wiring_without_any_scientific_draw_or_read(self):
        import native_inputs
        import native_source
        import native_branches
        import verified_plan_load
        with self.instance() as (runner, boot, calls):
            # Double only the loaders/producers. No scientific arrays, training
            # data or CUDA exist here; this checks exact call fields and ordering.
            runner.run_development()
            identity = policy.identity("pilot", 71990, 101)
            _, plan_ref = runner._read(policy.plan_name(71990))
            plan = object()
            loaded = {"plan":plan, "artifact":plan_ref}
            data = dict(images_bytes=b"synthetic-only", labels_bytes=b"synthetic-only", expected_files={})
            runner.kind = "native_producer_attestation"
            with mock.patch.object(verified_plan_load, "load_verified_plan_from_store", return_value=loaded) as load:
                with mock.patch.object(native_inputs, "read_training_idx", return_value=data) as read:
                    context = runner._context(identity)
            self.assertIs(context["plan"], plan)
            self.assertEqual(tuple(context["plan_artifact"]), ("sha256", "size_bytes"))
            self.assertEqual(context["plan_reference"], plan_ref)
            self.assertEqual(context["native_device"], "cuda:0")
            load.assert_called_once()
            read.assert_called_once()
            with mock.patch.object(runner, "_context", return_value=context):
                with mock.patch.object(native_source, "run_pilot_pair", autospec=True,
                                       return_value={"invariant_pass":False}) as pair:
                    result = runner._produce(policy.schedule()[2])
                    self.assertIs(result["invariant_pass"], False)
                    pair.assert_called_once_with(**context)
                with mock.patch.object(native_branches, "run_branches", autospec=True,
                                       return_value={"artifact":{}, "receipt":{}}) as branch:
                    result = runner._produce(policy.schedule()[3])
                    self.assertTrue(result["invariant_pass"])
                    branch.assert_called_once_with(**context)
            runner.kind = bootstrap_fixture.KIND

    def test_close_handoff_inspects_after_release_without_reopening(self):
        with self.instance() as (runner, boot, calls):
            with self.assertRaisesRegex(controller.ControllerError, "completed development"):
                runner.close_development()
            runner.run_development()
            before = set(runner.store._indexed)
            with mock.patch.object(storage.ArtifactStore, "reopen",
                                   side_effect=AssertionError("handoff must never reopen")) as reopen:
                handoff = runner.close_development()
                reopen.assert_not_called()
            self.assertTrue(runner.store._closed)
            self.assertEqual(handoff["status"], "closed_inspected_boundary")
            self.assertEqual(handoff["inspection"]["status"], "sealed_boundary")
            self.assertEqual(handoff["root_binding"], boot["root_binding"])
            self.assertEqual(handoff["journal_pin"], boot["journal_pin"])
            self.assertEqual(handoff["boundary_ref"]["name"], control.phase_name(9))
            self.assertEqual(handoff["resources"]["last_record"]["stage"], "controller.handoff.final")
            actual = storage.ArtifactStore.inspect(runner.store.root)
            self.assertEqual(handoff["root_accounting_before_close"]["root_logical_bytes"], actual["logical_bytes"])
            self.assertEqual(handoff["root_accounting_before_close"]["root_allocated_bytes"], actual["allocated_bytes"])
            self.assertGreaterEqual(handoff["resources"]["last_record"]["wall_seconds"],
                                    handoff["root_accounting_before_close"]["phase_wall_seconds"])
            self.assertFalse(handoff["execution_authorized"])
            self.assertFalse(handoff["automatic_reopen_enabled"])
            self.assertFalse(handoff["process_exit_verified"])
            self.assertEqual(set(storage.ArtifactStore.inspect(runner.store.root)["files"]), before)
            with self.assertRaisesRegex(controller.ControllerError, "untouched"):
                runner.close_development()

    def test_post_close_failure_retains_external_terminal_without_store_write(self):
        with self.instance() as (runner, boot, calls):
            runner.run_development()
            before = set(runner.store._indexed)
            check = runner.guard.check
            def fail(stage):
                if stage == "controller.handoff.post_close":
                    raise RuntimeError("injected after close")
                return check(stage)
            with mock.patch.object(runner.guard, "check", fail):
                with mock.patch.object(storage.ArtifactStore, "reopen",
                                       side_effect=AssertionError("no failure reopen")):
                    with self.assertRaisesRegex(RuntimeError, "injected after close"):
                        runner.close_development()
            self.assertTrue(runner.store._closed)
            self.assertEqual(runner.handoff_failure_metadata_status, "retained")
            self.assertEqual(set(storage.ArtifactStore.inspect(runner.store.root)["files"]), before)
            observed = control.inspect_native_attempt(runner.store.root, boot["journal_pin"],
                expected_store_profile=storage.MLP_FIXTURE,
                expected_evidence_kind=bootstrap_fixture.KIND)
            self.assertEqual(observed["status"], "launcher_failed_with_root")
            self.assertFalse(observed["eligible_for_writable_reopen"])

    def test_failed_read_only_inspection_and_failed_retention_return_no_handoff(self):
        for unavailable in (False, True):
            with self.subTest(unavailable=unavailable):
                with self.instance() as (runner, boot, calls):
                    runner.run_development()
                    with mock.patch.object(control, "inspect_native_attempt",
                                           return_value={"schema":control.INSPECTION_SCHEMA,"status":"writer_live"}):
                        if unavailable:
                            with mock.patch.object(control, "record_boundary_failure", side_effect=OSError("injected")):
                                with self.assertRaisesRegex(controller.ControllerError, "actual inspection"):
                                    runner.close_development()
                        else:
                            with self.assertRaisesRegex(controller.ControllerError, "actual inspection"):
                                runner.close_development()
                    self.assertTrue(runner.store._closed)
                    self.assertTrue(runner._terminal)
                    self.assertEqual(runner.handoff_failure_metadata_status,
                        "unavailable_or_preexisting" if unavailable else "retained")

    def test_handoff_rejects_contradictory_nested_inspection_authority(self):
        for field, value in (("schema", "unknown"), ("can_resume_incomplete", True),
                             ("execution_authorized", True), ("scientific_execution_certified", True)):
            with self.subTest(field=field):
                with self.instance() as (runner, boot, calls):
                    runner.run_development()
                    inspect = control.inspect_native_attempt
                    def contradict(*args, **kwargs):
                        result = inspect(*args, **kwargs)
                        result[field] = value
                        return result
                    with mock.patch.object(control, "inspect_native_attempt", contradict):
                        with self.assertRaisesRegex(controller.ControllerError, "actual inspection"):
                            runner.close_development()
                    self.assertTrue(runner.store._closed)
                    self.assertTrue(runner._terminal)
                    self.assertEqual(runner.handoff_failure_metadata_status, "retained")

    def test_root_replacement_after_inspection_is_not_successful_handoff(self):
        with self.instance() as (runner, boot, calls):
            runner.run_development()
            inspect = control.inspect_native_attempt
            def replace(*args, **kwargs):
                result = inspect(*args, **kwargs)
                root = runner.store.root
                root.rename(root.with_name(root.name + "-moved-fixture"))
                root.mkdir()
                return result
            with mock.patch.object(control, "inspect_native_attempt", replace):
                with self.assertRaises((OSError, control.ControlError)):
                    runner.close_development()
            self.assertTrue(runner.store._closed)
            self.assertTrue(runner._terminal)
            self.assertEqual(runner.handoff_failure_metadata_status, "retained")


if __name__ == "__main__":
    unittest.main()
