"""Dataset-free phase ordering, immutable boundary recovery and actual single-root smoke."""
from contextlib import contextmanager
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
    raise RuntimeError("explicitly hide CUDA for phase fixtures")

import torch

import artifact_store as storage
import identity_codec as codec
import phase_controller as phase
import scientific_runner as runner
import state_core as state
from envelope_fixture import fixture_context

UTC = "2026-09-06T22:30:00Z"


@contextmanager
def prepared(**limits):
    with tempfile.TemporaryDirectory(prefix="i7-phase-test-") as directory:
        with storage.ArtifactStore(directory, profile=storage.MLP_FIXTURE,
                                   min_filesystem_free_bytes=0, **limits) as store:
            with fixture_context(store=store) as context:
                controller = phase.PhaseController(store, context=context, plan_name="frozen-plan.pt", created_utc=UTC)
                yield controller, context


class PhaseControllerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_real_single_root_source_branches_audit_and_three_reopens(self):
        with tempfile.TemporaryDirectory(prefix="i7-single-root-smoke-") as directory:
            result = runner.run_fixture(directory)
            roots = list(Path(directory).iterdir())
            self.assertEqual(len(roots), 1)
            self.assertEqual(result["primary"]["event"], "primary_complete")
            self.assertEqual(result["final"]["event"], "complete")
            self.assertEqual(result["primary"]["root"], result["final"]["root"])
            self.assertGreater(result["final"]["logical_bytes"], result["primary"]["logical_bytes"])
            report = storage.ArtifactStore.inspect(roots[0])
            self.assertFalse(report["terminal"])
            self.assertEqual(result["final"]["logical_bytes"], report["logical_bytes"])
            self.assertEqual(result["final"]["allocated_bytes"], report["allocated_bytes"])
            self.assertEqual(len(report["receipts"]), 15)  # plan + 5 outputs + 9 events
            self.assertFalse(result["final"]["scientific_execution_certified"])
            events = [codec.json_loads((roots[0] / f"phase-{i:03d}.json").read_bytes(),
                                      max_bytes=phase.EVENT_MAX) for i in range(9)]
            self.assertEqual([row["event"] for row in events], list(phase.EVENTS))
            self.assertTrue(all(row["scientific_execution_certified"] is False for row in events))
            self.assertEqual([row["name"] for row in events[3]["outputs"]][-1], phase.SOURCE_COMPLETE)
            self.assertEqual(events[3]["runtime"]["phase"], events[5]["runtime"]["phase"])
            self.assertGreater(events[5]["runtime"]["check_count"], events[3]["runtime"]["check_count"])
            audit_ref, = events[-1]["outputs"]
            audit = storage.ArtifactStore.load_tensor_tree(roots[0], audit_ref["name"],
                expected_size=audit_ref["size_bytes"], expected_sha256=audit_ref["sha256"])
            counts = audit["payload"]["exact_validation"]["completion"]
            self.assertEqual(counts["adam_parameter_checks"], 24)
            self.assertEqual(counts["adam_tensor_screens"], 72)
            self.assertEqual(len([key for key in audit["payload"]["comparison_audits"] if key.endswith(".independent")]), 390)
            self.assertFalse(torch.cuda.is_initialized())

    def test_ready_reopen_preserves_full_plan_receipt_and_root(self):
        with prepared() as (controller, context):
            pin = controller.boundary_pin()
            initial = controller.summary()
            controller.store.close()
            reopened = phase.PhaseController.reopen(pin, context=context)
            try:
                self.assertEqual(reopened.summary(), initial)
                self.assertEqual(tuple(reopened.bindings["plan"]),
                    ("name", "status", "encoding", "size_bytes", "sha256", "receipt_name",
                     "receipt_size_bytes", "receipt_sha256"))
                self.assertEqual(reopened.store.budget, storage.DEFAULT_BUDGET)
            finally:
                reopened.store.close()

    def test_branches_before_source_refuses_before_callback(self):
        with prepared() as (controller, _):
            controller.go("primary", decision="dataset_free_fixture_only", created_utc=UTC)
            calls = []
            with self.assertRaises(phase.PhaseError):
                controller.execute("branches", lambda *_: calls.append("ran"), created_utc=UTC)
            self.assertEqual(calls, [])
            self.assertTrue(controller.store._terminal)

    def test_fake_success_is_not_source_completion(self):
        with prepared() as (controller, _):
            controller.go("primary", decision="dataset_free_fixture_only", created_utc=UTC)
            with self.assertRaises(phase.PhaseError):
                controller.execute("source", lambda *_: {"pass": True}, created_utc=UTC)
            self.assertEqual(controller.last["event"], "source_started")
            self.assertTrue(controller.store._terminal)
            report = storage.ArtifactStore.inspect(controller.store.root)
            self.assertTrue(report["failures"])

    def test_stale_ready_pin_and_go_reuse_are_rejected(self):
        with prepared() as (controller, context):
            pin = controller.boundary_pin()
            controller.go("primary", decision="dataset_free_fixture_only", created_utc=UTC)
            with self.assertRaises(phase.PhaseError):
                controller.boundary_pin()
            controller.store.close()
            with self.assertRaises(phase.PhaseError):
                phase.PhaseController.reopen(pin, context=context)
        with prepared() as (controller, _):
            controller.go("primary", decision="dataset_free_fixture_only", created_utc=UTC)
            with self.assertRaises(phase.PhaseError):
                controller.go("primary", decision="dataset_free_fixture_only", created_utc=UTC)

    def test_wrong_profile_cannot_enable_science(self):
        with prepared() as (controller, context):
            with self.assertRaises(phase.PhaseError):
                controller.go("primary", decision="scientific_GO", created_utc=UTC)
        with tempfile.TemporaryDirectory(prefix="i7-linear-reject-") as directory:
            with storage.ArtifactStore(directory, profile=storage.FIXTURE, min_filesystem_free_bytes=0) as store:
                with self.assertRaises(phase.PhaseError):
                    phase.PhaseController(store, context={}, plan_name="missing", created_utc=UTC)
        blueprint = phase.scientific_schedule()
        self.assertFalse(blueprint["execution_enabled"])
        self.assertEqual(blueprint["primary"]["bundles"], [71001, 71002, 71003])
        self.assertEqual(blueprint["sensitivity"]["bundles"], [71901])
        self.assertFalse(blueprint["sensitivity"]["pooled"])
        order = blueprint["order"]
        self.assertLess(order.index("all_scientific_plans_frozen"), order.index("primary_go"))
        self.assertLess(order.index("all_primary_sources_complete"), order.index("all_primary_branches_complete"))
        self.assertLess(order.index("all_primary_branches_complete"), order.index("sensitivity_go"))
        self.assertLess(order.index("sensitivity_branches_complete"), order.index("audit_go"))

    def test_unreported_indexed_file_blocks_boundary_reopen(self):
        with prepared() as (controller, context):
            pin = controller.boundary_pin()
            controller.store.write_bytes("unexpected-extra.bin", b"retained evidence")
            controller.store.close()
            with self.assertRaises(phase.PhaseError):
                phase.PhaseController.reopen(pin, context=context)
            self.assertTrue((Path(pin["root"]["path"]) / "unexpected-extra.bin").exists())

    def test_context_plan_and_source_digest_changes_rejected(self):
        for field in ("plan", "sources"):
            with self.subTest(field=field), prepared() as (controller, context):
                pin = controller.boundary_pin()
                altered = copy.deepcopy(context)
                if field == "plan":
                    altered["plan"]["initialization_seed"] += 1
                else:
                    altered["sources"]["files"][0]["sha256"] = "0" * 64
                controller.store.close()
                with self.assertRaises((phase.PhaseError, ValueError, RuntimeError)):
                    phase.PhaseController.reopen(pin, context=altered)

    def test_abrupt_exit_after_started_or_raw_write_never_restarts(self):
        for partial in (False, True):
            with self.subTest(partial=partial), tempfile.TemporaryDirectory(prefix="i7-crash-parent-") as directory:
                # A fresh process owns its clocks. Forking an existing guarded
                # process correctly fails its process_time regression gate.
                script = """
import json,os,sys,tempfile
tempfile.tempdir = sys.argv[1]
from test_phase_controller import prepared,UTC
with prepared() as (controller, context):
    print(json.dumps(controller.boundary_pin()),flush=True)
    controller.go('primary',decision='dataset_free_fixture_only',created_utc=UTC)
    def die(store, guard):
        if sys.argv[2] == 'True':
            store.write_bytes('partial-evidence.bin',b'retained')
        os._exit(19)
    controller.execute('source',die,created_utc=UTC)
"""
                result = subprocess.run([sys.executable, "-c", script, directory, str(partial)],
                    cwd=Path(__file__).parent, capture_output=True, text=True, timeout=30)
                self.assertEqual(result.returncode, 19, result.stderr)
                pin = json.loads(result.stdout)
                root = pin["root"]
                store = storage.ArtifactStore.reopen(root["path"], expected_profile=phase.FIXTURE,
                    expected_header_sha256=root["header_sha256"],
                    expected_root_identity=(root["device"], root["inode"]))
                try:
                    from test_plan_bindings import fixture
                    verifier = phase.PhaseController.__new__(phase.PhaseController)
                    verifier.store, verifier.root, verifier.identity = store, root, fixture()[0]
                    # Same exact chain verification used by public reopen;
                    # no reconstruction of a dead process or source is attempted.
                    with self.assertRaises(phase.PhaseError):
                        verifier._verify_chain(pin["event"])
                finally:
                    store.close()
                self.assertTrue((Path(root["path"]) / "phase-002.json").exists())
                self.assertEqual((Path(root["path"]) / "partial-evidence.bin").exists(), partial)

    def test_actual_source_finishes_eight_updates_before_branches(self):
        with prepared() as (controller, context):
            labels = []
            controller.go("primary", decision="dataset_free_fixture_only", created_utc=UTC)
            def body(store, guard):
                def observe(label):
                    labels.append(label)
                    guard(label)
                return runner.run_source_fixture(store, context, observe)
            controller.execute("source", body, created_utc=UTC)
            self.assertEqual([label for label in labels if label.endswith(".end") and label.startswith("source.step.")],
                             [f"source.step.{i}.end" for i in range(1, 9)])
            self.assertLess(labels.index("source.step.8.end"), labels.index("source.completion.pre_seal"))
            self.assertEqual(controller.last["event"], "source_complete")
            with self.assertRaises(phase.PhaseError):
                controller.boundary_pin()
            refs = controller.last["outputs"]
            endpoint_ref = refs[-1]
            endpoint = phase._validate_output(controller.store, endpoint_ref, context["identity"])
            self.assertEqual(endpoint["optimizer"]["state_completed_updates"], 8)
            self.assertEqual(endpoint["observer"]["state_completed_observations"], 8)

    def test_phase_metadata_obeys_same_cap_and_retains_failure(self):
        # Enough room for plan+ready but not an arbitrarily large extra manifest.
        with prepared(budget_bytes=70000, failure_reserve_bytes=16384) as (controller, _):
            before = controller.summary()["logical_bytes"]
            controller.store.write_bytes("fill.bin", b"x" * (70000 - 16384 - before - 1000))
            with self.assertRaises(storage.StoreError):
                controller.store.write_bytes("phase-budget-probe.json", b"x" * 1000)
            report = storage.ArtifactStore.inspect(controller.store.root)
            self.assertTrue(report["terminal"])
            self.assertLessEqual(report["logical_bytes"], 70000)
            self.assertTrue((controller.store.root / "fill.bin").exists())

    def test_rehashed_boolean_inventory_size_is_not_an_integer(self):
        with prepared() as (controller, context):
            pin = controller.boundary_pin()
            root = controller.store.root
            controller.store.close()
            event_path = root / pin["event"]["name"]
            event = codec.json_loads(event_path.read_bytes(), max_bytes=phase.EVENT_MAX)
            lock, = [row for row in event["inventory_before"] if row["name"] == "store.lock"]
            self.assertEqual(lock["size_bytes"], 0)
            lock["size_bytes"] = False
            raw = codec.json_bytes(event)
            event_path.write_bytes(raw)
            ref = phase._ref(pin["event"]["name"], raw)
            receipt_path = root / phase._receipt_name(ref["name"])
            receipt = codec.json_loads(receipt_path.read_bytes(), max_bytes=4096)
            receipt.update(size=len(raw), sha256=ref["sha256"])
            receipt_path.write_bytes(storage._json_bytes(receipt))
            pin["event"] = ref  # Rehash both files so stale hashes cannot explain rejection.
            with self.assertRaises(phase.PhaseError):
                phase.PhaseController.reopen(pin, context=context)

    def test_valid_core_at_seven_or_nine_is_not_source_completion(self):
        with prepared() as (controller, context):
            controller.go("primary", decision="dataset_free_fixture_only", created_utc=UTC)
            controller.execute("source", lambda store, guard: runner.run_source_fixture(store, context, guard), created_utc=UTC)
            actual = phase._validate_output(controller.store, controller.last["outputs"][-1], context["identity"])
            for count in (7, 9):
                changed = copy.deepcopy(actual)
                changed["state_completed_updates"], changed["anchor_update"] = count, count + 1
                changed["optimizer"]["state_completed_updates"] = count
                for row in changed["optimizer"]["state"]:
                    row["step"].fill_(count)
                changed["observer"]["state_completed_observations"] = count
                changed["observer"]["state"]["step_count"] = count
                state.validate_core(changed)  # Valid core, wrong phase completion count.
                with self.subTest(count=count), tempfile.TemporaryDirectory(prefix="i7-endpoint-count-") as directory:
                    with storage.ArtifactStore(directory, profile=phase.FIXTURE, min_filesystem_free_bytes=0) as store:
                        receipt = store.write_tensor_tree(phase.SOURCE_COMPLETE, changed)
                        with self.assertRaises(phase.PhaseError):
                            phase._validate_output(store, runner.receipt_ref(receipt), context["identity"])

    def test_default_entrypoints_and_import_are_inert(self):
        directory = Path(__file__).parent
        for file in ("scientific_runner.py", "audit_runner.py"):
            result = subprocess.run([sys.executable, str(directory / file)], capture_output=True,
                                    text=True, timeout=10, check=True)
            self.assertIn("No ", result.stdout)
        result = subprocess.run([sys.executable, "-c",
            "import scientific_runner,audit_runner,sys; assert 'torch' not in sys.modules"],
            cwd=directory, capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
