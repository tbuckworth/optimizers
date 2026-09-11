"""CPU-only cooperative-guard tests for the two actual artifact producers."""
import os
import tempfile
import unittest
from unittest import mock

if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
    raise RuntimeError("explicitly hide CUDA for producer-guard fixtures")

import torch

import artifact_store as storage
import branch_execution as execution
import identity_codec as codec
import source_capture as source
from envelope_fixture import fixture_context, warm_live


SOURCE_CREATED = "2026-09-06T21:00:00Z"
BRANCH_CREATED = "2026-09-06T21:00:01Z"


class ProducerGuardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def _capture(self, context, store, *, guard=None):
        model, optimizer, observer = warm_live(context)
        transaction = source.capture_anchor_then_live_witness(
            model, optimizer, observer, store=store, created_utc=SOURCE_CREATED,
            guard=guard, **context)
        return transaction, model, optimizer, observer

    def test_source_guard_before_step_retains_anchor_and_prevents_optimizer_work(self):
        with fixture_context() as context, tempfile.TemporaryDirectory(
                prefix="i7-source-guard-step-") as parent:
            model, optimizer, observer = warm_live(context)
            stages = []

            def guard(stage):
                stages.append(stage)
                if stage == "source_capture.source.before_optimizer_step":
                    raise RuntimeError("fixture resource limit")

            with storage.ArtifactStore(parent, profile=storage.MLP_FIXTURE,
                                       min_filesystem_free_bytes=0) as store:
                with mock.patch.object(optimizer, "step", wraps=optimizer.step) as step:
                    with self.assertRaises(source.SourceCaptureError):
                        source.capture_anchor_then_live_witness(
                            model, optimizer, observer, store=store,
                            created_utc=SOURCE_CREATED, guard=guard, **context)
                step.assert_not_called()
                report = storage.ArtifactStore.inspect(store.root)
                anchor_name = codec.artifact_id(
                    context["identity"], profile=context["profile"], kind="anchor") + ".pt"
                self.assertTrue(report["terminal"])
                self.assertEqual([row["name"] for row in report["receipts"]], [anchor_name])
                self.assertIn(anchor_name, report["files"])
                self.assertEqual(stages[-1], "source_capture.source.before_optimizer_step")
                self.assertIn("source_capture.source.before_forward", stages)
                self.assertIn("source_capture.source.after_observer", stages)
                self.assertNotIn("source_capture.source.after_optimizer_step", stages)
            self.assertFalse(torch.cuda.is_initialized())

    def test_source_postwrite_failure_retains_both_raw_artifacts_and_terminal(self):
        with fixture_context() as context, tempfile.TemporaryDirectory(
                prefix="i7-source-guard-postwrite-") as parent:
            stages = []

            def guard(stage):
                stages.append(stage)
                if stage == "source_capture.witness.artifact.postwrite":
                    raise RuntimeError("fixture post-seal limit")

            with storage.ArtifactStore(parent, profile=storage.MLP_FIXTURE,
                                       min_filesystem_free_bytes=0) as store:
                with self.assertRaises(source.SourceCaptureError):
                    self._capture(context, store, guard=guard)
                report = storage.ArtifactStore.inspect(store.root)
                expected = {
                    codec.artifact_id(context["identity"], profile=context["profile"],
                                      kind="anchor") + ".pt",
                    codec.artifact_id(context["identity"], profile=context["profile"],
                                      kind="source-witness") + ".pt",
                }
                self.assertTrue(report["terminal"])
                self.assertEqual({row["name"] for row in report["receipts"]}, expected)
                self.assertTrue(expected.issubset(report["files"]))
                self.assertEqual(stages[-1], "source_capture.witness.artifact.postwrite")
            self.assertFalse(torch.cuda.is_initialized())

    def test_branch_guard_stops_before_first_actual_branch(self):
        with fixture_context() as context, tempfile.TemporaryDirectory(
                prefix="i7-branch-guard-before-") as parent:
            with storage.ArtifactStore(parent, profile=storage.MLP_FIXTURE,
                                       min_filesystem_free_bytes=0) as store:
                transaction, _, _, _ = self._capture(context, store)
                stages = []

                def guard(stage):
                    stages.append(stage)
                    if stage == "branch_execution.branch.canonical.raw.before":
                        raise RuntimeError("fixture branch limit")

                with mock.patch.object(execution, "_execute_one",
                                       wraps=execution._execute_one) as execute_one:
                    with self.assertRaises(execution.BranchExecutionError):
                        execution.execute_branches(
                            store=store, created_utc=BRANCH_CREATED,
                            guard=guard, **transaction, **context)
                execute_one.assert_not_called()
                report = storage.ArtifactStore.inspect(store.root)
                branch_name = codec.artifact_id(
                    context["identity"], profile=context["profile"], kind="branch-results") + ".pt"
                self.assertTrue(report["terminal"])
                self.assertNotIn(branch_name, report["files"])
                self.assertEqual(len(report["receipts"]), 2)
                self.assertIn("branch_execution.candidate.before_creation", stages)
                self.assertIn("branch_execution.candidate.after_creation", stages)
                self.assertEqual(stages[-1], "branch_execution.branch.canonical.raw.before")
            self.assertFalse(torch.cuda.is_initialized())

    def test_branch_guard_covers_both_orders_and_propagates_loss_chunks(self):
        with fixture_context() as context, tempfile.TemporaryDirectory(
                prefix="i7-branch-guard-labels-") as parent:
            with storage.ArtifactStore(parent, profile=storage.MLP_FIXTURE,
                                       min_filesystem_free_bytes=0) as store:
                transaction, _, _, _ = self._capture(context, store)
                stages = []
                outcome = execution.execute_branches(
                    store=store, created_utc=BRANCH_CREATED,
                    guard=stages.append, **transaction, **context)
                self.assertEqual(tuple(outcome), execution.RETURN_KEYS)
                for order in ("canonical", "reverse"):
                    for branch in execution.BRANCHES:
                        before = f"branch_execution.branch.{order}.{branch}.before"
                        after = f"branch_execution.branch.{order}.{branch}.after"
                        self.assertLess(stages.index(before), stages.index(after))
                self.assertIn("before:assembly:validation", stages)
                self.assertIn("after:assembly:validation", stages)
                self.assertIn("loss.before_chunk.0.1", stages)
                self.assertIn("loss.after_chunk.0.1", stages)
                self.assertIn("loss.before_chunk.0.4", stages)
                self.assertIn("loss.after_chunk.0.4", stages)
                self.assertLess(stages.index("branch_execution.artifact.prewrite"),
                                stages.index("branch_execution.artifact.postwrite"))
                self.assertFalse(storage.ArtifactStore.inspect(store.root)["terminal"])
            self.assertFalse(torch.cuda.is_initialized())

    def test_branch_postwrite_failure_retains_complete_result_and_terminal(self):
        with fixture_context() as context, tempfile.TemporaryDirectory(
                prefix="i7-branch-guard-postwrite-") as parent:
            with storage.ArtifactStore(parent, profile=storage.MLP_FIXTURE,
                                       min_filesystem_free_bytes=0) as store:
                transaction, _, _, _ = self._capture(context, store)

                def guard(stage):
                    if stage == "branch_execution.artifact.postwrite":
                        raise RuntimeError("fixture post-seal limit")

                with self.assertRaises(execution.BranchExecutionError):
                    execution.execute_branches(
                        store=store, created_utc=BRANCH_CREATED,
                        guard=guard, **transaction, **context)
                report = storage.ArtifactStore.inspect(store.root)
                branch_name = codec.artifact_id(
                    context["identity"], profile=context["profile"], kind="branch-results") + ".pt"
                self.assertTrue(report["terminal"])
                self.assertIn(branch_name, report["files"])
                self.assertEqual(sum(row["name"] == branch_name for row in report["receipts"]), 1)
            self.assertFalse(torch.cuda.is_initialized())


if __name__ == "__main__":
    unittest.main()
