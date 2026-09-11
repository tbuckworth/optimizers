"""Synthetic, positive-path anchor binding tests; no scientific provenance claims.

Real temporary Git collection, CPU environment collection, restricted verified
plan loading, synthetic IDX materialization, and four planned live updates are
provided by envelope_fixture. No source/environment validator is mocked.
"""
import copy
import os
import tempfile
import unittest
from unittest import mock

if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
    raise RuntimeError("explicitly hide CUDA for anchor-envelope fixtures")

import torch

import anchor_envelope as subject
import artifact_store as storage
import source_environment as provenance
import state_core as state
from envelope_fixture import fixture_context, warm_live


CREATED = "2026-09-06T12:34:56Z"
ROOT = ("schema_name", "schema_version", "profile", "artifact_id", "created_utc",
        "identity", "payload", "payload_tensor_bytes")
PAYLOAD = ("model", "optimizer", "observer", "rng", "bindings")
BINDINGS = ("plan", "probes", "data", "sources", "environment")


def tensors(value):
    if type(value) is torch.Tensor:
        yield value
    elif type(value) is dict:
        for child in value.values():
            yield from tensors(child)
    elif type(value) in (tuple, list):
        for child in value:
            yield from tensors(child)


def assign(value, path, replacement):
    for key in path[:-1]:
        value = value[key]
    value[path[-1]] = replacement


class AnchorEnvelopeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.addClassCleanup(torch.set_num_threads, old_threads)

    def setUp(self):
        self.context = self.enterContext(fixture_context())
        self.live = warm_live(self.context)
        self.core = state.capture_core(*self.live, profile=storage.MLP_FIXTURE,
                                      completed_updates=4)
        self.anchor = subject.make_anchor(self.core, created_utc=CREATED, **self.context)

    def exact(self, left, right):
        """Independent typed comparison, including raw tensor and float zero bits."""
        self.assertIs(type(left), type(right))
        if type(left) is torch.Tensor:
            self.assertEqual((left.dtype, left.shape, left.device),
                             (right.dtype, right.shape, right.device))
            self.assertEqual(left.numpy().tobytes(), right.numpy().tobytes())
        elif type(left) is dict:
            self.assertEqual(len(left), len(right))
            for (lk, lv), (rk, rv) in zip(left.items(), right.items()):
                self.exact(lk, rk)
                self.exact(lv, rv)
        elif type(left) in (list, tuple):
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self.exact(a, b)
        elif type(left) is float:
            self.assertEqual(left.hex(), right.hex())
        else:
            self.assertEqual(left, right)

    def reject_value(self, path, replacement):
        value = copy.deepcopy(self.anchor)
        assign(value, path, replacement)
        with self.assertRaises(subject.EnvelopeError):
            subject.validate_anchor(value, **self.context)

    def test_positive_exact_schema_counters_and_collected_bindings(self):
        self.assertIs(subject.validate_anchor(self.anchor, **self.context), self.anchor)
        self.assertEqual(tuple(self.anchor), ROOT)
        self.assertEqual(tuple(self.anchor["payload"]), PAYLOAD)
        bindings = self.anchor["payload"]["bindings"]
        self.assertEqual(tuple(bindings), BINDINGS)
        self.assertEqual(self.anchor["schema_name"], "i7_anchor")
        self.assertIs(type(self.anchor["schema_version"]), int)
        self.assertEqual(self.anchor["schema_version"], 1)
        self.assertEqual(self.anchor["artifact_id"],
                         "fixture_tiny_mlp_cpu_v1--primary--b0--u5--anchor")
        self.assertEqual(self.anchor["created_utc"], CREATED)
        self.exact(subject.core_from_anchor(self.anchor), self.core)
        self.assertEqual(self.anchor["identity"]["anchor_completed_updates"], 4)
        self.assertEqual(self.anchor["payload"]["observer"]["state"]["step_count"], 4)
        for row in self.anchor["payload"]["optimizer"]["state"]:
            self.assertEqual(row["step"].item(), 4.0)
        self.assertEqual([row["path"] for row in bindings["sources"]["files"]],
                         ["fixture_source.py", "fixture_contract.md"])
        self.assertEqual(bindings["sources"]["worktree_status"], "clean_including_untracked_v1")
        for key in ("sources", "environment"):
            self.exact(bindings[key], self.context[key])
        provenance.validate_sources(bindings["sources"], profile=storage.MLP_FIXTURE)
        provenance.validate_environment(bindings["environment"], profile=storage.MLP_FIXTURE)
        provenance.validate_state_environment(self.core, bindings["environment"],
                                               profile=storage.MLP_FIXTURE)
        self.assertFalse(torch.cuda.is_initialized())

    def test_exact_root_payload_binding_membership_order_and_dict_types(self):
        class OtherDict(dict):
            pass
        for path in ((), ("payload",), ("payload", "bindings")):
            original = self.anchor
            for key in path:
                original = original[key]
            for replacement in ({**original, "unexpected": None},
                                dict(list(original.items())[1:]),
                                dict(reversed(list(original.items()))), OtherDict(original)):
                value = copy.deepcopy(self.anchor)
                if path:
                    assign(value, path, replacement)
                else:
                    value = replacement
                with self.subTest(path=path, keys=tuple(replacement)), \
                     self.assertRaises(subject.EnvelopeError):
                    subject.validate_anchor(value, **self.context)

    def test_universal_scalar_types_identity_and_profile_cannot_be_coerced(self):
        for path, wrong in ((("schema_name",), "i7_state_core"),
                            (("schema_version",), True), (("schema_version",), 1.0),
                            (("profile",), storage.FIXTURE), (("profile",), storage.SCIENTIFIC),
                            (("artifact_id",), "fixture_tiny_mlp_cpu_v1--primary--b0--u5--source-witness"),
                            (("created_utc",), "2026-09-06T12:34:56+00:00"),
                            (("identity", "bundle"), False),
                            (("identity", "anchor_update"), 6),
                            (("identity", "anchor_completed_observations"), 3),
                            (("identity", "stream_roles"), {**self.context["identity"]["stream_roles"], 7: "extra"})):
            with self.subTest(path=path, wrong=wrong):
                self.reject_value(path, wrong)
        for profile in (storage.FIXTURE, storage.SCIENTIFIC, None, True):
            context = {**self.context, "profile": profile}
            with self.subTest(profile=profile), self.assertRaises(subject.EnvelopeError):
                subject.make_anchor(self.core, created_utc=CREATED, **context)

    def test_core_and_nested_observer_optimizer_counters_are_bound(self):
        for path, wrong in ((("anchor_update",), 6), (("state_completed_updates",), 3),
                            (("optimizer", "state_completed_updates"), 3),
                            (("observer", "state_completed_observations"), 3),
                            (("observer", "state", "step_count"), 3)):
            core = copy.deepcopy(self.core)
            assign(core, path, wrong)
            with self.subTest(path=path), self.assertRaises(subject.EnvelopeError):
                subject.make_anchor(core, created_utc=CREATED, **self.context)
        self.reject_value(("payload", "optimizer", "state", 0, "step"), torch.tensor(3.0))
        self.reject_value(("payload", "observer", "state", "step_count"), 3)

    def test_payload_bytes_are_actual_tensor_bytes_not_file_size_or_bool(self):
        expected = sum(t.numel() * t.element_size() for t in tensors(self.anchor["payload"]))
        self.assertIs(type(self.anchor["payload_tensor_bytes"]), int)
        self.assertEqual(self.anchor["payload_tensor_bytes"], expected)
        for wrong in (expected + 1, expected - 1, float(expected), True, -1):
            with self.subTest(wrong=wrong):
                self.reject_value(("payload_tensor_bytes",), wrong)

    def test_clones_all_tensor_outputs_and_nested_metadata(self):
        inputs = list(tensors(self.core)) + list(tensors(self.context))
        output = list(tensors(self.anchor))
        input_pointers = {t.untyped_storage().data_ptr() for t in inputs}
        output_pointers = [t.untyped_storage().data_ptr() for t in output]
        self.assertEqual(len(output_pointers), len(set(output_pointers)))
        self.assertFalse(set(output_pointers) & input_pointers)
        original = copy.deepcopy(self.anchor)
        self.core["model"]["parameters"][0]["value"].add_(1.0)
        self.context["plan"]["training_batches"][4, 0] = 0
        self.context["sources"]["files"][0]["sha256"] = "0" * 64
        self.exact(self.anchor, original)

    def test_aliases_and_nonfinite_or_unsupported_payload_tensors_rejected(self):
        for location in ("core", "anchor"):
            value = copy.deepcopy(self.core if location == "core" else self.anchor)
            payload = value if location == "core" else value["payload"]
            payload["optimizer"]["state"][0]["exp_avg"]["value"] = \
                payload["model"]["parameters"][0]["value"]
            with self.subTest(location=location), self.assertRaises(subject.EnvelopeError):
                if location == "core":
                    subject.make_anchor(value, created_utc=CREATED, **self.context)
                else:
                    subject.validate_anchor(value, **self.context)
        parameter = self.anchor["payload"]["model"]["parameters"][0]["value"]
        for bad in (torch.full_like(parameter, float("nan")), parameter.double(),
                    parameter.clone().requires_grad_(), parameter.T, object()):
            self.reject_value(("payload", "model", "parameters", 0, "value"), bad)

    def test_constructor_rejects_unsupported_core_before_cloning(self):
        for variant in ("requires_grad", "oversized_storage_view"):
            core = copy.deepcopy(self.core)
            record = core["model"]["parameters"][0]
            if variant == "requires_grad":
                record["value"].requires_grad_()
            else:
                original = record["value"]
                backing = torch.empty(original.numel() + 1, dtype=original.dtype)
                record["value"] = backing[:-1].reshape(original.shape)
                record["value"].copy_(original)
                self.assertTrue(record["value"].is_contiguous())
                self.assertGreater(record["value"].untyped_storage().nbytes(),
                                   record["value"].numel() * record["value"].element_size())
            # Component checks alone permit these representations. The envelope
            # must reject them before an owned clone silently repairs the input.
            state.validate_core(core)
            before = (record["value"].requires_grad, record["value"].untyped_storage().data_ptr(),
                      record["value"].untyped_storage().nbytes())
            with self.subTest(variant=variant), self.assertRaises(subject.EnvelopeError):
                subject.make_anchor(core, created_utc=CREATED, **self.context)
            self.assertEqual(before, (record["value"].requires_grad,
                record["value"].untyped_storage().data_ptr(), record["value"].untyped_storage().nbytes()))

    def test_plan_indices_and_file_identity_changes_rejected(self):
        self.reject_value(("payload", "bindings", "plan", "next_batch_indices"),
                          torch.tensor([0, 9, 1, 0], dtype=torch.int64))
        for field in ("training_batches", "training_probe_indices"):
            context = copy.deepcopy(self.context)
            index = (4, 0) if field == "training_batches" else (0,)
            context["plan"][field][index] = 0
            with self.subTest(field=field), self.assertRaises(subject.EnvelopeError):
                subject.validate_anchor(self.anchor, **context)
        context = copy.deepcopy(self.context)
        context["plan_artifact"]["sha256"] = "0" * 64
        with self.assertRaises(subject.EnvelopeError):
            subject.validate_anchor(self.anchor, **context)

    def test_idx_byte_and_expected_hash_changes_rejected(self):
        for key in ("images_bytes", "labels_bytes"):
            context = copy.deepcopy(self.context)
            raw = context[key]
            context[key] = raw[:-1] + bytes([raw[-1] ^ 1])
            with self.subTest(key=key), self.assertRaises(subject.EnvelopeError):
                subject.validate_anchor(self.anchor, **context)
        context = copy.deepcopy(self.context)
        context["expected_files"]["training_images"]["sha256"] = "0" * 64
        with self.assertRaises(subject.EnvelopeError):
            subject.make_anchor(self.core, created_utc=CREATED, **context)
        self.reject_value(("payload", "bindings", "data", "replacement_count"), 0)

    def test_source_hash_and_runtime_values_require_direct_expected_equality(self):
        self.reject_value(("payload", "bindings", "sources", "files", 0, "sha256"), "0" * 64)
        self.reject_value(("payload", "bindings", "environment", "python", "version"), "different")
        for field in ("sources", "environment"):
            context = copy.deepcopy(self.context)
            if field == "sources":
                context[field]["files"][0]["sha256"] = "0" * 64
            else:
                context[field]["operating_system"]["release"] += "-different"
            with self.subTest(field=field), self.assertRaises(subject.EnvelopeError):
                subject.validate_anchor(self.anchor, **context)

    def test_repository_root_and_rng_layout_cross_bindings(self):
        for path, replacement in ((("repository_root_realpath",), "/different-fixture-root"),
                                 (("rng_layout", "python", "internal_length"), 624),
                                 (("rng_layout", "numpy", "keys_length"), 623),
                                 (("rng_layout", "torch_cpu", "state_length"), 1),
                                 (("cuda", "initialized"), True)):
            context = copy.deepcopy(self.context)
            assign(context["environment"], path, replacement)
            with self.subTest(path=path), self.assertRaises(subject.EnvelopeError):
                subject.make_anchor(self.core, created_utc=CREATED, **context)
        self.reject_value(("payload", "model", "parameters", 0, "native_device"), "cpu:0")
        self.reject_value(("payload", "rng", "torch_cuda"), [{"device_index": 0}])

    def test_signed_zero_is_preserved_and_direct_equality_is_typed(self):
        positive, negative = torch.tensor([0.0]), torch.tensor([-0.0])
        self.assertTrue(torch.equal(positive, negative))
        for left, right in ((positive, negative), (0.0, -0.0), (1, True),
                            ({"a": 1, "b": 2}, {"b": 2, "a": 1}), ([1], (1,))):
            self.assertFalse(subject.same_exact(left, right))
        core = copy.deepcopy(self.core)
        core["model"]["parameters"][0]["value"][0, 0] = -0.0
        anchor = subject.make_anchor(core, created_utc=CREATED, **self.context)
        self.exact(subject.core_from_anchor(anchor), core)
        self.assertTrue(torch.signbit(anchor["payload"]["model"]["parameters"][0]["value"][0, 0]))
        context = copy.deepcopy(self.context)
        self.assertEqual(context["plan"]["replacement_uniforms"][0].item(), 0.0)
        context["plan"]["replacement_uniforms"][0] = -0.0
        with self.assertRaises(subject.EnvelopeError):
            subject.validate_anchor(self.anchor, **context)

    def test_success_and_failure_leave_inputs_live_objects_and_rng_unchanged(self):
        before_core, before_context = copy.deepcopy(self.core), copy.deepcopy(self.context)
        rng = state._raw_rng_state()
        live_ids = tuple(id(value) for value in self.live)
        for _ in range(2):
            made = subject.make_anchor(self.core, created_utc=CREATED, **self.context)
            subject.validate_anchor(made, **self.context)
        with self.assertRaises(subject.EnvelopeError):
            subject.make_anchor(self.core, created_utc="bad", **self.context)
        self.exact(self.core, before_core)
        self.exact(self.context, before_context)
        self.exact(rng, state._raw_rng_state())
        self.assertEqual(live_ids, tuple(id(value) for value in self.live))
        self.exact(self.core, state.capture_core(*self.live, profile=storage.MLP_FIXTURE,
                                                completed_updates=4))
        self.assertFalse(torch.cuda.is_initialized())

    def test_real_restricted_store_roundtrip_validates_owned_exact_bytes(self):
        with tempfile.TemporaryDirectory(prefix="i7-anchor-roundtrip-") as directory:
            with storage.ArtifactStore(directory, profile=storage.MLP_FIXTURE,
                                       min_filesystem_free_bytes=0) as store:
                receipt = store.write_tensor_tree(self.anchor["artifact_id"] + ".pt", self.anchor)
            with mock.patch.object(storage.torch, "load", wraps=torch.load) as decoder:
                recovered = storage.ArtifactStore.load_tensor_tree(store.root, receipt["name"],
                    expected_size=receipt["size"], expected_sha256=receipt["sha256"])
                self.assertEqual(decoder.call_count, 1)
                self.assertEqual(decoder.call_args.kwargs, {"map_location": "cpu", "weights_only": True})
            self.exact(recovered, self.anchor)
            self.assertIs(subject.validate_anchor(recovered, **self.context), recovered)
            self.assertNotEqual(receipt["size"], recovered["payload_tensor_bytes"])
            original_addresses = {t.untyped_storage().data_ptr() for t in tensors(self.anchor)}
            self.assertFalse(original_addresses & {t.untyped_storage().data_ptr() for t in tensors(recovered)})
            self.assertFalse(storage.ArtifactStore.inspect(store.root)["terminal"])
            with mock.patch.object(storage.torch, "load") as decoder:
                with self.assertRaises(storage.StoreError):
                    storage.ArtifactStore.load_tensor_tree(store.root, receipt["name"],
                        expected_size=receipt["size"], expected_sha256="0" * 64)
                decoder.assert_not_called()


if __name__ == "__main__":
    unittest.main()
