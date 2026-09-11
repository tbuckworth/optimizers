"""Focused CPU-only tests for the actual branch execution transaction."""
import copy
import os
import tempfile
import unittest
from unittest import mock

if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
    raise RuntimeError("explicitly hide CUDA for branch-execution fixtures")

import torch

import artifact_store as storage
import branch_execution as execution
import identity_codec as codec
import source_capture as source
import state_core as state
from envelope_fixture import fixture_context, warm_live


class BranchExecutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def _source(self, context, store):
        model, optimizer, observer = warm_live(context)
        transaction = source.capture_anchor_then_live_witness(
            model, optimizer, observer, store=store,
            created_utc="2026-09-06T20:00:00Z", **context)
        return transaction

    def test_zero_branch_assigns_real_zeros_and_executes_adamw(self):
        with fixture_context() as context, tempfile.TemporaryDirectory(
                prefix="i7-branch-execution-") as parent:
            with storage.ArtifactStore(parent, profile=storage.MLP_FIXTURE,
                                       min_filesystem_free_bytes=0) as store:
                transaction = self._source(context, store)
                outcome = execution.execute_branches(
                    store=store, created_utc="2026-09-06T20:00:01Z",
                    **transaction, **context)
                self.assertEqual(tuple(outcome), execution.RETURN_KEYS)
                zero = outcome["artifact"]["payload"]["branches"]["zero"]
                self.assertEqual(zero["status"], "defined")
                self.assertEqual(zero["assigned_gradient_null_mask"], [False] * 4)
                self.assertEqual(torch.count_nonzero(zero["delivered_gradient"]["value"]).item(), 0)
                self.assertEqual(zero["optimizer_after"]["state_completed_updates"], 5)
                self.assertTrue(all(float(row["step"].item()) == 5.0
                                    for row in zero["optimizer_after"]["state"]))
                before = transaction["anchor"]["payload"]["model"]["parameters"]
                self.assertTrue(any(not torch.equal(old["value"], new["value"])
                                    for old, new in zip(before, zero["parameters_after"])))
                self.assertFalse(torch.cuda.is_initialized())

    def test_terminal_store_rejected_before_any_model_forward(self):
        with fixture_context() as context, tempfile.TemporaryDirectory(
                prefix="i7-branch-terminal-") as parent:
            with storage.ArtifactStore(parent, profile=storage.MLP_FIXTURE,
                                       min_filesystem_free_bytes=0) as store:
                transaction = self._source(context, store)
                self.assertEqual(store._fail("fixture_terminal", "before branch"), "retained")
                with mock.patch.object(torch.nn.Sequential, "forward",
                                       side_effect=AssertionError("forward must not run")) as forward:
                    with self.assertRaisesRegex(execution.BranchExecutionError,
                                                "store is closed or terminal"):
                        execution.execute_branches(
                            store=store, created_utc="2026-09-06T20:00:01Z",
                            **transaction, **context)
                forward.assert_not_called()
                self.assertEqual(len(storage.ArtifactStore.inspect(store.root)["failures"]), 1)
                self.assertFalse(torch.cuda.is_initialized())

    def test_pinned_input_mismatch_terminals_before_forward_and_cannot_retry(self):
        with fixture_context() as context, tempfile.TemporaryDirectory(
                prefix="i7-branch-input-failure-") as parent:
            with storage.ArtifactStore(parent, profile=storage.MLP_FIXTURE,
                                       min_filesystem_free_bytes=0) as store:
                transaction = self._source(context, store)
                bad = dict(transaction)
                bad["witness"] = copy.deepcopy(transaction["witness"])
                bad["witness"]["payload"]["live_loss"]["value_native_float32"] += 0.25
                with mock.patch.object(torch.nn.Sequential, "forward",
                                       side_effect=AssertionError("forward must not run")) as forward:
                    with self.assertRaisesRegex(execution.BranchExecutionError,
                                                "supplied value differs"):
                        execution.execute_branches(store=store,
                            created_utc="2026-09-06T20:00:01Z", **bad, **context)
                    with self.assertRaisesRegex(execution.BranchExecutionError,
                                                "store is closed or terminal"):
                        execution.execute_branches(store=store,
                            created_utc="2026-09-06T20:00:01Z", **transaction, **context)
                forward.assert_not_called()
                report = storage.ArtifactStore.inspect(store.root)
                self.assertTrue(report["terminal"])
                self.assertEqual(len(report["failures"]), 1)
                result_name = codec.artifact_id(context["identity"], profile=context["profile"],
                                                kind="branch-results") + ".pt"
                self.assertNotIn(result_name, report["files"])
                self.assertFalse(torch.cuda.is_initialized())

    def test_rng_use_after_completed_branch_write_is_detected_and_retained(self):
        with fixture_context() as context, tempfile.TemporaryDirectory(
                prefix="i7-branch-post-write-rng-") as parent:
            with storage.ArtifactStore(parent, profile=storage.MLP_FIXTURE,
                                       min_filesystem_free_bytes=0) as store:
                transaction = self._source(context, store)
                caller_rng = state._raw_rng_state()
                result_name = codec.artifact_id(context["identity"], profile=context["profile"],
                                                kind="branch-results") + ".pt"
                actual_write = store.write_tensor_tree

                def consuming_write(name, tree):
                    receipt = actual_write(name, tree)
                    if name == result_name:
                        torch.rand(1)  # Inject use only after the completed result write.
                    return receipt

                with mock.patch.object(store, "write_tensor_tree", side_effect=consuming_write):
                    with self.assertRaisesRegex(execution.BranchExecutionError,
                                                "changed anchor RNG"):
                        execution.execute_branches(store=store,
                            created_utc="2026-09-06T20:00:01Z", **transaction, **context)
                self.assertTrue(state._tree_equal(state._raw_rng_state(), caller_rng))
                report = storage.ArtifactStore.inspect(store.root)
                self.assertTrue(report["terminal"])
                self.assertIn(result_name, report["files"])
                self.assertEqual(sum(row["name"] == result_name for row in report["receipts"]), 1)
                with mock.patch.object(torch.nn.Sequential, "forward",
                                       side_effect=AssertionError("retry forward must not run")) as forward:
                    with self.assertRaisesRegex(execution.BranchExecutionError,
                                                "store is closed or terminal"):
                        execution.execute_branches(store=store,
                            created_utc="2026-09-06T20:00:01Z", **transaction, **context)
                forward.assert_not_called()
                self.assertFalse(torch.cuda.is_initialized())


if __name__ == "__main__":
    unittest.main()
