"""Tiny actual-live transaction tests; never scientific data or launch evidence."""
import copy
import hashlib
import os
import tempfile
import unittest
from unittest import mock

if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
    raise RuntimeError("explicitly hide CUDA for source-capture fixtures")

import torch
import torch.nn.functional as F

import anchor_envelope as envelopes
import artifact_store as storage
import data_probe_bindings as data
from envelope_fixture import fixture_context, warm_live, model_factory, optimizer_factory, observer_factory
import source_capture as subject
import state_core as state


CREATED = "2026-09-06T12:34:56Z"


def capture(store, context, model, optimizer, observer):
    return subject.capture_anchor_then_live_witness(
        model, optimizer, observer, store=store, created_utc=CREATED, **context)


def validate(result, context, *, witness=None, anchor=None, receipt=None):
    return subject.validate_source_witness(
        result["witness"] if witness is None else witness,
        anchor=result["anchor"] if anchor is None else anchor,
        anchor_receipt=result["anchor_receipt"] if receipt is None else receipt,
        **context)


def reference_gradient(model):
    value = torch.cat([parameter.grad.detach().reshape(-1)
                       for parameter in model.parameters()])
    return {"value": value.cpu().contiguous().clone(), "shape": list(value.shape),
            "native_dtype": str(value.dtype), "native_device": str(value.device)}


class SourceCaptureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_anchor_is_sealed_before_forward_and_actual_values_match_independent_replay(self):
        with fixture_context() as context, tempfile.TemporaryDirectory(prefix="i7-source-store-") as parent:
            model, optimizer, observer = warm_live(context)
            anchor_id = subject.codec.artifact_id(context["identity"], profile=context["profile"], kind="anchor")
            anchor_name = anchor_id + ".pt"
            receipt_name = "receipt-" + hashlib.sha256(anchor_name.encode("ascii")).hexdigest() + ".json"
            forwards = []
            with storage.ArtifactStore(parent, profile=context["profile"],
                                       min_filesystem_free_bytes=0) as store:
                def before_forward(_module, _inputs):
                    self.assertIn(anchor_name, store._indexed)
                    self.assertIn(receipt_name, store._indexed)
                    report, _ = storage._inspect_dirfd(store._dirfd)
                    self.assertFalse(report["terminal"])
                    self.assertEqual([row["name"] for row in report["receipts"]], [anchor_name])
                    forwards.append("sealed")

                hook = model.register_forward_pre_hook(before_forward)
                try:
                    with mock.patch.object(subject.state, "restore_core",
                                           side_effect=AssertionError("restore before witness forbidden")) as restore:
                        result = capture(store, context, model, optimizer, observer)
                    restore.assert_not_called()
                finally:
                    hook.remove()
                self.assertEqual(forwards, ["sealed"])
                self.assertIs(validate(result, context), result["witness"])
                self.assertEqual(result["witness"]["payload"]["capture_proof"]["ordered_events"],
                                 subject.EVENTS)
                self.assertNotEqual(
                    result["witness"]["payload"]["raw_gradient"]["value"].untyped_storage().data_ptr(),
                    result["witness"]["payload"]["delivered_current_gradient"]["value"].untyped_storage().data_ptr())
                report, _ = storage._inspect_dirfd(store._dirfd)
                self.assertFalse(report["terminal"])
                self.assertEqual({row["name"] for row in report["receipts"]},
                                 {anchor_name, result["witness"]["artifact_id"] + ".pt"})

                # This reconstruction starts only after the witness receipt exists.
                anchor_core = envelopes.core_from_anchor(result["anchor"])
                restored = state.restore_core(anchor_core, model_factory, optimizer_factory, observer_factory)
                replay_model, replay_optimizer = restored["model"], restored["optimizer"]
                replay_observer = restored["observer"]
                materialized = data.materialize(context["images_bytes"], context["labels_bytes"],
                    plan=context["plan"], identity=context["identity"], profile=context["profile"],
                    expected_files=context["expected_files"])
                batch = materialized["probes"]["batch_noisy"]
                loss = F.cross_entropy(replay_model(batch["inputs"],), batch["labels"], reduction="mean")
                loss_value = float(loss.detach().item())
                loss.backward()
                raw = reference_gradient(replay_model)
                replay_observer.filter_grad()
                current = reference_gradient(replay_model)
                replay_optimizer.step()
                replay_optimizer.zero_grad(set_to_none=True)
                endpoint = state.capture_core(replay_model, replay_optimizer, replay_observer,
                                              profile=context["profile"], completed_updates=5)
                witness = result["witness"]["payload"]
                self.assertEqual(loss_value.hex(), witness["live_loss"]["value_native_float32"].hex())
                self.assertTrue(envelopes.same_exact(raw, witness["raw_gradient"]))
                self.assertTrue(envelopes.same_exact(current, witness["delivered_current_gradient"]))
                self.assertTrue(envelopes.same_exact(endpoint["model"]["parameters"],
                                                     witness["parameters_after"]))
                self.assertTrue(envelopes.same_exact(endpoint["optimizer"], witness["optimizer_after"]))
                self.assertTrue(envelopes.same_exact(endpoint["observer"], witness["observer_after"]))
                sample_bytes = (batch["inputs"].numpy().astype("<f4", copy=False).tobytes(order="C")
                                + batch["labels"].numpy().astype("<i8", copy=False).tobytes(order="C"))
                self.assertEqual(witness["live_loss"]["sample_identity_sha256"],
                                 hashlib.sha256(sample_bytes).hexdigest())
            self.assertFalse(torch.cuda.is_initialized())

    def test_strict_witness_tampering_types_order_masks_and_aliases_fail(self):
        with fixture_context() as context, tempfile.TemporaryDirectory(prefix="i7-source-tamper-") as parent:
            model, optimizer, observer = warm_live(context)
            with storage.ArtifactStore(parent, profile=context["profile"],
                                       min_filesystem_free_bytes=0) as store:
                result = capture(store, context, model, optimizer, observer)

                mutations = []
                mutations.append(lambda value: value["payload"]["capture_proof"].update(
                    ordered_events=list(reversed(subject.EVENTS))))
                mutations.append(lambda value: value["payload"]["source_step"].update(
                    anchor_deserializations_before_live_step=False))
                mutations.append(lambda value: value["payload"]["live_loss"].update(
                    sample_identity_sha256="0" * 64))
                mutations.append(lambda value: value["payload"]["parameters_after"][0].update(index=True))
                mutations.append(lambda value: value["payload"]["optimizer_after"].update(
                    state_completed_updates=True))
                mutations.append(lambda value: value.update(payload_tensor_bytes=True))
                for mutate in mutations:
                    bad = state.clone_tree(result["witness"])
                    mutate(bad)
                    with self.assertRaises(subject.SourceCaptureError):
                        validate(result, context, witness=bad)

                aliased = state.clone_tree(result["witness"])
                aliased["payload"]["delivered_current_gradient"]["value"] = \
                    aliased["payload"]["raw_gradient"]["value"]
                with self.assertRaises(subject.SourceCaptureError):
                    validate(result, context, witness=aliased)

                bad_anchor = state.clone_tree(result["anchor"])
                bad_anchor["payload"]["model"]["gradient_null_mask"][0] = False
                with self.assertRaises(subject.SourceCaptureError):
                    validate(result, context, anchor=bad_anchor)

                class TextSubclass(str):
                    pass
                bad_receipt = dict(result["anchor_receipt"])
                bad_receipt["status"] = TextSubclass("complete")
                with self.assertRaises(subject.SourceCaptureError):
                    validate(result, context, receipt=bad_receipt)

    def test_unexpected_gradient_fails_before_capture_step_and_second_call_is_inert(self):
        with fixture_context() as context, tempfile.TemporaryDirectory(prefix="i7-source-prefail-") as parent:
            model, optimizer, observer = warm_live(context)
            first = next(model.parameters())
            first.grad = torch.ones_like(first)
            unexpected_gradient = first.grad
            forwards = []
            hook = model.register_forward_pre_hook(lambda *_: forwards.append("forward"))
            with storage.ArtifactStore(parent, profile=context["profile"],
                                       min_filesystem_free_bytes=0) as store:
                original = subject.state.capture_core
                with mock.patch.object(subject.state, "capture_core", wraps=original) as captures:
                    with self.assertRaises(subject.SourceCaptureError):
                        capture(store, context, model, optimizer, observer)
                    calls = captures.call_count
                    rng = state._raw_rng_state()
                    with self.assertRaises(subject.SourceCaptureError):
                        capture(store, context, model, optimizer, observer)
                    self.assertEqual(captures.call_count, calls)
                    self.assertTrue(envelopes.same_exact(rng, state._raw_rng_state()))
                report, _ = storage._inspect_dirfd(store._dirfd)
                self.assertTrue(report["terminal"])
                self.assertEqual(report["receipts"], [])
                self.assertEqual(len(report["failures"]), 1)
                self.assertIs(first.grad, unexpected_gradient)
                self.assertEqual(forwards, [])
            hook.remove()

    def test_neutrality_failure_retains_sealed_anchor_and_never_forwards(self):
        with fixture_context() as context, tempfile.TemporaryDirectory(prefix="i7-source-neutrality-") as parent:
            model, optimizer, observer = warm_live(context)
            forwards = []
            hook = model.register_forward_pre_hook(lambda *_: forwards.append("forward"))
            with storage.ArtifactStore(parent, profile=context["profile"],
                                       min_filesystem_free_bytes=0) as store:
                original, calls = subject.state.capture_core, []

                def unequal_second(*args, **kwargs):
                    core = original(*args, **kwargs)
                    calls.append(1)
                    if len(calls) == 2:
                        core["model"]["training_modes"][0] = not core["model"]["training_modes"][0]
                    return core

                with mock.patch.object(subject.state, "capture_core", side_effect=unequal_second):
                    with self.assertRaises(subject.SourceCaptureError):
                        capture(store, context, model, optimizer, observer)
                    count = len(calls)
                    with self.assertRaises(subject.SourceCaptureError):
                        capture(store, context, model, optimizer, observer)
                    self.assertEqual(len(calls), count)
                report, _ = storage._inspect_dirfd(store._dirfd)
                self.assertTrue(report["terminal"])
                self.assertEqual(len(report["receipts"]), 1)
                self.assertEqual(report["receipts"][0]["name"],
                                 subject.codec.artifact_id(context["identity"],
                                     profile=context["profile"], kind="anchor") + ".pt")
                self.assertEqual(len(report["failures"]), 1)
                self.assertEqual(forwards, [])
            hook.remove()

    def test_stale_runtime_snapshot_after_anchor_write_fails_before_forward(self):
        with fixture_context() as context, tempfile.TemporaryDirectory(prefix="i7-source-runtime-") as parent:
            model, optimizer, observer = warm_live(context)
            forwards = []
            hook = model.register_forward_pre_hook(lambda *_: forwards.append("forward"))
            with storage.ArtifactStore(parent, profile=context["profile"],
                                       min_filesystem_free_bytes=0) as store:
                original_write = store.write_tensor_tree
                writes = []

                def write_then_change_default_dtype(name, tree):
                    receipt = original_write(name, tree)
                    writes.append(name)
                    torch.set_default_dtype(torch.float64)
                    return receipt

                original_dtype = torch.get_default_dtype()
                try:
                    with mock.patch.object(store, "write_tensor_tree",
                                           side_effect=write_then_change_default_dtype):
                        with self.assertRaises(subject.SourceCaptureError):
                            capture(store, context, model, optimizer, observer)
                finally:
                    torch.set_default_dtype(original_dtype)
                report, _ = storage._inspect_dirfd(store._dirfd)
                self.assertTrue(report["terminal"])
                self.assertEqual(len(report["receipts"]), 1)
                self.assertEqual(writes, [subject.codec.artifact_id(context["identity"],
                    profile=context["profile"], kind="anchor") + ".pt"])
                self.assertEqual(forwards, [])
            hook.remove()


if __name__ == "__main__":
    unittest.main()
