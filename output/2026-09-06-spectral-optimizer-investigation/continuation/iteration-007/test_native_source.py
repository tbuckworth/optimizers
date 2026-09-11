#!/usr/bin/env python3
"""Bounded actual-writer CPU checks for the fixed source-loop adapter."""
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
    raise RuntimeError("explicitly hide CUDA for native-source fixtures")
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS"):
    if os.environ.get(key) != "1":
        raise RuntimeError("single numerical threads required for native-source fixtures")

import torch


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))

import anchor_envelope as envelopes
import artifact_store as storage
from envelope_fixture import fixture_context
import identity_codec as codec
import native_source as subject
import runtime_guard as runtime
import source_history as history
import state_core as state


BIG_TMP = Path("/tmp/spectral-experiment-artifacts")
CREATED = "2026-09-07T12:34:56Z"


class NativeSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        if torch.cuda.is_initialized():
            raise RuntimeError("native-source tests must not initialize CUDA")

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
            prefix="i7-native-source-test-", dir=BIG_TMP)
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

    def assert_actual_reference(self, reference):
        body = (self.root / reference["name"]).read_bytes()
        receipt = (self.root / reference["receipt_name"]).read_bytes()
        self.assertEqual((len(body), hashlib.sha256(body).hexdigest()),
                         (reference["size_bytes"], reference["sha256"]))
        self.assertEqual((len(receipt), hashlib.sha256(receipt).hexdigest()),
                         (reference["receipt_size_bytes"],
                          reference["receipt_sha256"]))

    def test_single_source_seals_actual_anchor_witness_and_bounded_completion(self):
        store, handle = self.store_context(), {}
        caller_rng = state._raw_rng_state()
        with fixture_context(plan_handle=handle, store=store) as context:
            result = subject.run_source(
                capture_mode="capture_on", **self.kwargs(store, context, handle))
            self.assertEqual(tuple(result), subject.SOURCE_RESULT_KEYS)
            self.assertIs(result["invariant_pass"], True)
            self.assertIsNone(result["adverse_reason"])
            history.validate_source_completion(result["completion"])
            self.assertEqual(result["completion"]["final_state_core"]
                             ["state_completed_updates"], 8)
            self.assertEqual(len(result["completion"]["anchor_witness_refs"]), 1)
            pair = result["completion"]["anchor_witness_refs"][0]
            self.assert_actual_reference(pair["anchor_ref"])
            self.assert_actual_reference(pair["witness_ref"])
            self.assertEqual(result["completion_receipt"]["name"],
                             result["completion"]["artifact_name"])
            report, _ = storage._inspect_dirfd(store._dirfd)
            self.assertFalse(report["terminal"])
            self.assertFalse(any("branch" in row["name"] for row in report["receipts"]))
            self.assertTrue(envelopes.same_exact(state._raw_rng_state(), caller_rng))
            self.checkpoint()

    def test_pilot_pair_writes_three_linked_records_from_one_pass(self):
        store, handle = self.store_context(), {}
        caller_rng = state._raw_rng_state()
        with fixture_context(plan_handle=handle, store=store) as context:
            result = subject.run_pilot_pair(**self.kwargs(store, context, handle))
            self.assertEqual(tuple(result), subject.PAIR_RESULT_KEYS)
            self.assertIs(result["invariant_pass"], True)
            self.assertIsNone(result["adverse_reason"])
            history.validate_capture_bundle(
                result["capture_on"], result["capture_off"], result["comparison"])
            self.assertEqual(result["comparison"]["summary"]["core_equal_steps"], 8)
            comparison_raw = (self.root / result["comparison"]["artifact_name"]).read_bytes()
            self.assertEqual(codec.json_loads(comparison_raw,
                             max_bytes=storage.DEFAULT_BUDGET), result["comparison"])
            names = [result[key]["name"] for key in
                     ("capture_on_receipt", "capture_off_receipt",
                      "comparison_receipt")]
            self.assertEqual(names, [result["capture_on"]["artifact_name"],
                                     result["capture_off"]["artifact_name"],
                                     result["comparison"]["artifact_name"]])
            report, _ = storage._inspect_dirfd(store._dirfd)
            self.assertFalse(report["terminal"])
            self.assertFalse(any("branch" in row["name"] for row in report["receipts"]))
            self.assertTrue(envelopes.same_exact(state._raw_rng_state(), caller_rng))
            self.checkpoint()

    def test_valid_adverse_pair_is_retained_and_reported_as_failed_invariant(self):
        store, handle = self.store_context(), {}
        with fixture_context(plan_handle=handle, store=store) as context:
            original = subject._normal_step

            def one_adverse_core(source_context, arm, materialized, update):
                core = original(source_context, arm, materialized, update)
                if arm["name"] == "capture_off" and update == 3:
                    core = state.clone_tree(core)
                    core["model"]["parameters"][0]["value"].view(-1)[0] += .25
                return core

            with mock.patch.object(subject, "_normal_step",
                                   side_effect=one_adverse_core):
                result = subject.run_pilot_pair(**self.kwargs(store, context, handle))
            self.assertIs(result["invariant_pass"], False)
            self.assertEqual(result["adverse_reason"], "capture_path_inequality")
            self.assertEqual(result["comparison"]["summary"]["core_equal_steps"], 7)
            self.assertFalse(result["comparison"]["step_trace"][2]
                             ["core_direct_typed_equal"])
            history.validate_capture_bundle(
                result["capture_on"], result["capture_off"], result["comparison"])
            report, _ = storage._inspect_dirfd(store._dirfd)
            self.assertFalse(report["terminal"])
            for key in ("capture_on_receipt", "capture_off_receipt",
                        "comparison_receipt"):
                self.assertIn(result[key]["name"],
                              {row["name"] for row in report["receipts"]})

    def test_non_adverse_failure_restores_rng_and_terminalizes_once(self):
        store, handle = self.store_context(), {}
        caller_rng = state._raw_rng_state()
        with fixture_context(plan_handle=handle, store=store) as context:
            def fail_at_second_update(stage):
                self.checkpoint(stage)
                if stage == "native_source.update.2.begin":
                    raise RuntimeError("fixture_guard_stop")

            with self.assertRaisesRegex(subject.NativeSourceError,
                                        "native source failed: RuntimeError"):
                subject.run_source(
                    capture_mode="capture_on",
                    **self.kwargs(store, context, handle, fail_at_second_update))
            report, _ = storage._inspect_dirfd(store._dirfd)
            self.assertTrue(report["terminal"])
            self.assertEqual(len(report["failures"]), 1)
            self.assertEqual(report["failures"][0]["code"], "native_source_failed")
            self.assertTrue(envelopes.same_exact(state._raw_rng_state(), caller_rng))

    def test_valid_but_unsealed_plan_mutation_fails_before_initialization(self):
        store, handle = self.store_context(), {}
        with fixture_context(plan_handle=handle, store=store) as context:
            altered = state.clone_tree(context["plan"])
            original = int(altered["training_batches"][0, 0].item())
            altered["training_batches"][0, 0] = (original + 1) % 10
            arguments = self.kwargs(store, context, handle)
            arguments["plan"] = altered
            with (mock.patch.object(subject, "_initialize",
                                    side_effect=AssertionError("initialization forbidden")) as initialize,
                  mock.patch.object(subject, "_materialize",
                                    side_effect=AssertionError("materialization forbidden")) as materialize,
                  mock.patch.object(subject, "_normal_step",
                                    side_effect=AssertionError("source update forbidden")) as normal,
                  mock.patch.object(subject, "_captured_step",
                                    side_effect=AssertionError("source capture forbidden")) as captured):
                with self.assertRaisesRegex(
                        subject.NativeSourceError,
                        "supplied plan differs from independently verified plan bytes"):
                    subject.run_source(capture_mode="capture_on", **arguments)
            initialize.assert_not_called()
            materialize.assert_not_called()
            normal.assert_not_called()
            captured.assert_not_called()
            report, _ = storage._inspect_dirfd(store._dirfd)
            self.assertTrue(report["terminal"])
            self.assertEqual(len(report["failures"]), 1)
            self.assertEqual(report["failures"][0]["code"], "native_source_failed")

    def test_import_and_default_cli_are_inert(self):
        store = self.store_context()
        code = ("import sys;sys.path.insert(0,sys.argv[1]);import native_source;"
                "print('torch' in sys.modules, 'source_capture' in sys.modules)")
        result = subprocess.run(
            [sys.executable, "-c", code, str(HERE)], cwd=store.root,
            capture_output=True, text=True, check=True, timeout=10)
        self.assertEqual(result.stdout, "False False\n")
        result = subprocess.run(
            [sys.executable, str(HERE / "native_source.py")], cwd=store.root,
            capture_output=True, text=True, check=True, timeout=10)
        self.assertEqual(
            result.stdout,
            "native_source: inert; no source execution, branch execution or scientific action\n")
        self.assertEqual({path.name for path in store.root.iterdir()},
                         {"store.lock", "store-header.json"})


if __name__ == "__main__":
    unittest.main()
