#!/usr/bin/env python3
"""Actual tiny CPU pipeline checks for the sealed-input native audit adapter."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock


if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
    raise RuntimeError("explicitly hide CUDA for native-audit fixtures")
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS"):
    if os.environ.get(key) != "1":
        raise RuntimeError("single numerical threads required for native-audit fixtures")

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))

import anchor_envelope as envelopes
import artifact_store as storage
import audit_envelope as audit
import branch_execution as execution
from envelope_fixture import fixture_context
import native_audit as subject
import native_branches
import native_phase_policy as policy
import native_source
import runtime_guard as runtime
import state_core as state


BIG_TMP = Path("/tmp/spectral-experiment-artifacts")
CREATED = "2026-09-07T16:30:00Z"


class NativeAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        if torch.cuda.is_initialized():
            raise RuntimeError("native-audit tests must not initialize CUDA")

    def setUp(self):
        self.started = time.monotonic()
        self.root = None
        self.checkpoint()

    def checkpoint(self, stage="fixture.local"):
        self.assertIs(type(stage), str)
        self.assertTrue(stage.isascii())
        self.assertLessEqual(time.monotonic() - self.started, 120.0)
        self.assertLessEqual(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                             2 << 30)
        self.assertFalse(torch.cuda.is_initialized())
        if self.root is not None:
            logical = sum(path.lstat().st_size for path in self.root.rglob("*")
                          if path.is_file())
            self.assertLessEqual(logical, (2 << 20) - (1 << 20))

    def store_context(self):
        runtime.verify_big_volume(BIG_TMP)
        self.assertGreaterEqual(shutil.disk_usage(BIG_TMP).free, 1 << 30)
        temporary = tempfile.TemporaryDirectory(prefix="i7-native-audit-test-", dir=BIG_TMP)
        self.addCleanup(temporary.cleanup)
        store = storage.ArtifactStore(
            temporary.name, profile=storage.MLP_FIXTURE,
            budget_bytes=2 << 20, failure_reserve_bytes=1 << 20,
            min_filesystem_free_bytes=1 << 30)
        self.addCleanup(store.close)
        self.root = store.root
        return store

    def kwargs(self, store, context, handle, guard=None):
        return {
            "store": store, "identity": context["identity"],
            "profile": context["profile"], "plan": context["plan"],
            "plan_reference": handle["reference"],
            "plan_artifact": context["plan_artifact"],
            "images_bytes": context["images_bytes"],
            "labels_bytes": context["labels_bytes"],
            "expected_files": context["expected_files"], "sources": context["sources"],
            "environment": context["environment"],
            "auditor_environment": context["environment"], "created_utc": CREATED,
            "guard": self.checkpoint if guard is None else guard,
        }

    def seal_source_branch(self, store, context, handle, *, mutate_branch=False):
        common = self.kwargs(store, context, handle)
        common.pop("auditor_environment")
        native_source.run_source(
            capture_mode="capture_on", native_device="cpu", **common)
        branch_kwargs = dict(common, native_device="cpu")
        if not mutate_branch:
            return native_branches.run_branches(**branch_kwargs)
        original = store.write_tensor_tree

        def inject(name, value):
            if type(value) is dict and value.get("schema_name") == "i7_branch_results":
                value["payload"]["candidate_state"]["measurement_before"]["probes"] \
                    ["batch_noisy"]["cpu64"]["before_ce"] += 1.0
            return original(name, value)

        with mock.patch.object(store, "write_tensor_tree", side_effect=inject):
            return native_branches.run_branches(**branch_kwargs)

    def test_actual_source_branch_to_complete_cpu_audit(self):
        store, handle = self.store_context(), {}
        caller_rng = state._raw_rng_state()
        with fixture_context(plan_handle=handle, store=store) as context:
            self.seal_source_branch(store, context, handle)
            result = subject.run_audit(**self.kwargs(store, context, handle))
            self.assertEqual(tuple(result), subject.RESULT_KEYS)
            artifact, receipt = result["artifact"], result["receipt"]
            envelopes.validate_common(
                artifact, schema_name="i7_anchor_numerical_audit",
                kind="independent-audit", identity=context["identity"],
                profile=context["profile"])
            self.assertEqual(artifact["payload"]["overall_status"], "pass")
            self.assertEqual(artifact["payload"]["exact_validation"]
                             ["auditor_environment"]["runtime_role"], "fixture_cpu")
            self.assertEqual(receipt["name"], artifact["artifact_id"] + ".pt")
            raw = (self.root / receipt["name"]).read_bytes()
            self.assertEqual((len(raw), hashlib.sha256(raw).hexdigest()),
                             (receipt["size"], receipt["sha256"]))
            report, _ = storage._inspect_dirfd(store._dirfd)
            self.assertFalse(report["terminal"])
            self.assertEqual(sum(row["name"] == receipt["name"]
                                 for row in report["receipts"]), 1)
            self.assertTrue(envelopes.same_exact(state._raw_rng_state(), caller_rng))
            self.checkpoint()

    def test_valid_failed_audit_is_returned_and_terminal_without_retry(self):
        store, handle = self.store_context(), {}
        caller_rng = state._raw_rng_state()
        with fixture_context(plan_handle=handle, store=store) as context:
            self.seal_source_branch(store, context, handle, mutate_branch=True)
            result = subject.run_audit(**self.kwargs(store, context, handle))
            self.assertEqual(result["artifact"]["payload"]["overall_status"],
                             "fatal_validation")
            self.assertTrue(result["artifact"]["payload"]["fatal_failures"])
            self.assertTrue(store._terminal)
            report = storage.ArtifactStore.inspect(store.root)
            self.assertIn(result["receipt"]["name"],
                          {row["name"] for row in report["receipts"]})
            self.assertEqual(len(report["failures"]), 1)
            self.assertTrue(envelopes.same_exact(state._raw_rng_state(), caller_rng))
            with mock.patch.object(audit, "audit_and_seal",
                                   side_effect=AssertionError("retry forbidden")) as retried:
                with self.assertRaisesRegex(subject.NativeAuditError,
                                            "writable profile-matched"):
                    subject.run_audit(**self.kwargs(store, context, handle))
            retried.assert_not_called()

    def test_guard_failure_after_seal_retains_audit_and_terminalizes(self):
        store, handle = self.store_context(), {}
        caller_rng = state._raw_rng_state()
        with fixture_context(plan_handle=handle, store=store) as context:
            self.seal_source_branch(store, context, handle)

            def stop_after_seal(stage):
                self.checkpoint(stage)
                if stage == "native_audit.execute.post":
                    raise RuntimeError("fixture_postseal_guard_stop")

            with self.assertRaisesRegex(subject.NativeAuditError,
                                        "native audit failed: RuntimeError"):
                subject.run_audit(**self.kwargs(store, context, handle, stop_after_seal))
            report, _ = storage._inspect_dirfd(store._dirfd)
            self.assertTrue(report["terminal"])
            self.assertEqual(report["failures"][0]["code"], "native_audit_failed")
            self.assertEqual(sum(row["name"].endswith("--independent-audit.pt")
                                 for row in report["receipts"]), 1)
            self.assertTrue(envelopes.same_exact(state._raw_rng_state(), caller_rng))

    def test_postseal_auditor_environment_mutation_fails_terminal(self):
        store, handle = self.store_context(), {}
        with fixture_context(plan_handle=handle, store=store) as context:
            self.seal_source_branch(store, context, handle)
            original = audit.audit_and_seal

            def mutate_returned_environment(**kwargs):
                result = original(**kwargs)
                result["artifact"]["payload"]["exact_validation"] \
                    ["auditor_environment"]["runtime_role"] = "native_source"
                return result

            with mock.patch.object(audit, "audit_and_seal",
                                   side_effect=mutate_returned_environment):
                with self.assertRaisesRegex(subject.NativeAuditError,
                                            "retained auditor environment"):
                    subject.run_audit(**self.kwargs(store, context, handle))
            report, _ = storage._inspect_dirfd(store._dirfd)
            self.assertTrue(report["terminal"])
            self.assertEqual(report["failures"][0]["code"], "native_audit_failed")
            self.assertEqual(sum(row["name"].endswith("--independent-audit.pt")
                                 for row in report["receipts"]), 1)

    def test_mutated_plan_rejects_before_transaction_or_audit(self):
        store, handle = self.store_context(), {}
        with fixture_context(plan_handle=handle, store=store) as context:
            self.seal_source_branch(store, context, handle)
            arguments = self.kwargs(store, context, handle)
            changed = state.clone_tree(context["plan"])
            changed["training_batches"][0, 0] = (
                int(changed["training_batches"][0, 0]) + 1) % 10
            arguments["plan"] = changed
            with mock.patch.object(audit, "audit_and_seal",
                                   side_effect=AssertionError("audit forbidden")) as called:
                with self.assertRaisesRegex(subject.NativeAuditError,
                                            "supplied plan differs"):
                    subject.run_audit(**arguments)
            called.assert_not_called()
            report, _ = storage._inspect_dirfd(store._dirfd)
            self.assertTrue(report["terminal"])
            self.assertFalse(any(row["name"].endswith("--independent-audit.pt")
                                 for row in report["receipts"]))

    def test_scientific_autospec_and_fixed_identity_admission(self):
        identity = policy.identity("primary", 71001, 101)
        self.assertEqual(subject._admit_identity(identity, storage.SCIENTIFIC)
                         ["anchor_updates"], [101, 500, 1000, 2000])
        with self.assertRaisesRegex(subject.NativeAuditError, "frozen 16"):
            subject._admit_identity(policy.identity("pilot", 71990, 101),
                                    storage.SCIENTIFIC)
        wired = mock.create_autospec(subject.run_audit)
        arguments = dict(store=object(), identity=identity, profile=storage.SCIENTIFIC,
            plan=object(), plan_reference=object(), plan_artifact=object(),
            images_bytes=object(), labels_bytes=object(), expected_files=object(),
            sources=object(), environment=object(), created_utc=CREATED,
            auditor_environment=object(), guard=lambda stage: None)
        wired(**arguments)
        with self.assertRaises(TypeError):
            wired(native_device="cuda:0", **arguments)

    def test_import_and_default_execution_are_inert(self):
        store = self.store_context()
        code = ("import sys;sys.path.insert(0,sys.argv[1]);import native_audit;"
                "print('torch' in sys.modules, 'audit_envelope' in sys.modules)")
        result = subprocess.run(
            [sys.executable, "-c", code, str(HERE)], cwd=store.root,
            capture_output=True, text=True, check=True, timeout=10)
        self.assertEqual(result.stdout, "False False\n")
        result = subprocess.run(
            [sys.executable, str(HERE / "native_audit.py")], cwd=store.root,
            capture_output=True, text=True, check=True, timeout=10)
        self.assertEqual(result.stdout,
                         "native_audit: inert; no audit or scientific action\n")
        self.assertEqual({path.name for path in store.root.iterdir()},
                         {"store.lock", "store-header.json"})


if __name__ == "__main__":
    unittest.main()
