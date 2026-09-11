#!/usr/bin/env python3
"""Actual sealed-source CPU checks for the fixed native branch adapter."""
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
    raise RuntimeError("explicitly hide CUDA for native-branch fixtures")
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS"):
    if os.environ.get(key) != "1":
        raise RuntimeError("single numerical threads required for native-branch fixtures")

import torch


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))

import anchor_envelope as envelopes
import artifact_store as storage
import branch_execution as execution
from envelope_fixture import fixture_context
import identity_codec as codec
import native_branches as subject
import native_source
import runtime_guard as runtime
import state_core as state


BIG_TMP = Path("/tmp/spectral-experiment-artifacts")
CREATED = "2026-09-07T12:34:56Z"


class NativeBranchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        if torch.cuda.is_initialized():
            raise RuntimeError("native-branch tests must not initialize CUDA")

    def setUp(self):
        self.started = time.monotonic()
        self.root = None
        self.checkpoint()

    def checkpoint(self, stage="fixture.local"):
        self.assertIs(type(stage), str)
        self.assertTrue(stage.isascii())
        self.assertLessEqual(time.monotonic() - self.started, 120.)
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
        temporary = tempfile.TemporaryDirectory(
            prefix="i7-native-branch-test-", dir=BIG_TMP)
        self.addCleanup(temporary.cleanup)
        store = storage.ArtifactStore(
            temporary.name, profile=storage.MLP_FIXTURE,
            budget_bytes=2 << 20, failure_reserve_bytes=1 << 20,
            min_filesystem_free_bytes=1 << 30)
        self.addCleanup(store.close)
        self.root = store.root
        return store

    def kwargs(self, store, context, handle, guard=None):
        return dict(
            store=store, identity=context["identity"], profile=context["profile"],
            native_device="cpu", plan=context["plan"],
            plan_reference=handle["reference"],
            plan_artifact=context["plan_artifact"],
            images_bytes=context["images_bytes"], labels_bytes=context["labels_bytes"],
            expected_files=context["expected_files"], sources=context["sources"],
            environment=context["environment"], created_utc=CREATED,
            guard=self.checkpoint if guard is None else guard)

    def seal_source(self, store, context, handle):
        return native_source.run_source(
            capture_mode="capture_on", **self.kwargs(store, context, handle))

    def test_actual_sealed_source_to_branch_result(self):
        store, handle = self.store_context(), {}
        caller_rng = state._raw_rng_state()
        with fixture_context(plan_handle=handle, store=store) as context:
            source_result = self.seal_source(store, context, handle)
            anchor_ref = source_result["completion"]["anchor_witness_refs"][0]
            result = subject.run_branches(**self.kwargs(store, context, handle))
            self.assertEqual(tuple(result), subject.RESULT_KEYS)
            envelopes.validate_common(
                result["artifact"], schema_name="i7_branch_results",
                kind="branch-results", identity=context["identity"],
                profile=context["profile"])
            self.assertEqual(result["receipt"]["name"],
                             result["artifact"]["artifact_id"] + ".pt")
            body = (self.root / result["receipt"]["name"]).read_bytes()
            self.assertEqual((len(body), hashlib.sha256(body).hexdigest()),
                             (result["receipt"]["size"], result["receipt"]["sha256"]))
            # The producer bindings point back to the exact source transaction.
            bindings = result["artifact"]["payload"]["audit_metadata"]["producer_bindings"]
            self.assertTrue(envelopes.same_exact(bindings["anchor_ref"],
                                                 anchor_ref["anchor_ref"]))
            self.assertTrue(envelopes.same_exact(bindings["source_witness_ref"],
                                                 anchor_ref["witness_ref"]))
            report, _ = storage._inspect_dirfd(store._dirfd)
            self.assertFalse(report["terminal"])
            self.assertEqual(sum(row["name"] == result["receipt"]["name"]
                                 for row in report["receipts"]), 1)
            self.assertFalse(any("audit" in row["name"] for row in report["receipts"]))
            self.assertTrue(envelopes.same_exact(state._raw_rng_state(), caller_rng))
            self.checkpoint()

    def test_unsealed_valid_plan_mutation_fails_before_branch_execution(self):
        store, handle = self.store_context(), {}
        with fixture_context(plan_handle=handle, store=store) as context:
            self.seal_source(store, context, handle)
            altered = state.clone_tree(context["plan"])
            original = int(altered["training_batches"][0, 0].item())
            altered["training_batches"][0, 0] = (original + 1) % 10
            arguments = self.kwargs(store, context, handle)
            arguments["plan"] = altered
            with mock.patch.object(
                    execution, "execute_branches",
                    side_effect=AssertionError("branch execution forbidden")) as execute:
                with self.assertRaisesRegex(
                        subject.NativeBranchesError,
                        "supplied plan differs from independently verified plan bytes"):
                    subject.run_branches(**arguments)
            execute.assert_not_called()
            report, _ = storage._inspect_dirfd(store._dirfd)
            self.assertTrue(report["terminal"])
            self.assertEqual(len(report["failures"]), 1)
            self.assertEqual(report["failures"][0]["code"], "native_branches_failed")
            self.assertFalse(any("branch-results" in row["name"]
                                 for row in report["receipts"]))

    def test_guard_failure_after_seal_keeps_branch_and_terminalizes(self):
        store, handle = self.store_context(), {}
        caller_rng = state._raw_rng_state()
        with fixture_context(plan_handle=handle, store=store) as context:
            self.seal_source(store, context, handle)

            def fail_after_seal(stage):
                self.checkpoint(stage)
                if stage == "native_branches.execute.post":
                    raise RuntimeError("fixture_postseal_guard_stop")

            expected_name = codec.artifact_id(
                context["identity"], profile=context["profile"],
                kind="branch-results") + ".pt"
            with self.assertRaisesRegex(
                    subject.NativeBranchesError,
                    "native branches failed: RuntimeError"):
                subject.run_branches(
                    **self.kwargs(store, context, handle, fail_after_seal))
            report, _ = storage._inspect_dirfd(store._dirfd)
            self.assertTrue(report["terminal"])
            self.assertEqual(len(report["failures"]), 1)
            self.assertEqual(report["failures"][0]["code"], "native_branches_failed")
            self.assertIn(expected_name, {row["name"] for row in report["receipts"]})
            self.assertTrue(envelopes.same_exact(state._raw_rng_state(), caller_rng))

    def test_import_and_default_cli_are_inert(self):
        store = self.store_context()
        code = ("import sys;sys.path.insert(0,sys.argv[1]);import native_branches;"
                "print('torch' in sys.modules, 'branch_execution' in sys.modules)")
        result = subprocess.run(
            [sys.executable, "-c", code, str(HERE)], cwd=store.root,
            capture_output=True, text=True, check=True, timeout=10)
        self.assertEqual(result.stdout, "False False\n")
        result = subprocess.run(
            [sys.executable, str(HERE / "native_branches.py")], cwd=store.root,
            capture_output=True, text=True, check=True, timeout=10)
        self.assertEqual(
            result.stdout,
            "native_branches: inert; no source, branch, audit or scientific action\n")
        self.assertEqual({path.name for path in store.root.iterdir()},
                         {"store.lock", "store-header.json"})


if __name__ == "__main__":
    unittest.main()
