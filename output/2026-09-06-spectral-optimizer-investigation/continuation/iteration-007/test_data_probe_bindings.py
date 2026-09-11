"""Synthetic IDX bytes only; deterministic MLP-v2 fixture, no scientific plan."""

import copy
import hashlib
import os
import random
import struct
import unittest
from unittest import mock

if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
    raise RuntimeError("explicitly hide CUDA for data/probe fixtures")
if os.environ.get("OPENBLAS_NUM_THREADS") != "1" or os.environ.get("OMP_NUM_THREADS") != "1":
    raise RuntimeError("set OPENBLAS_NUM_THREADS=1 and OMP_NUM_THREADS=1")

import numpy as np
import torch

import data_probe_bindings as subject
from test_plan_bindings import fixture


def receipts(images, labels):
    return {key: {"sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)}
            for key, data in (("training_images", images), ("training_labels", labels))}


def synthetic_case():
    identity, plan, _ = fixture()
    pixels = ((np.arange(90, dtype=np.uint16)*7) % 256).astype(np.uint8).reshape(30, 3)
    pixels[17] = [0, 1, 255]
    targets = (np.arange(30) % 2).astype(np.uint8)
    images = struct.pack(">IIII", 2051, 30, 1, 3)+pixels.tobytes()
    labels = struct.pack(">II", 2049, 30)+targets.tobytes()
    # Five selected rows, with three incorrect replacements and two unchanged.
    plan["replacement_uniforms"] = torch.tensor([0., .5, .9, np.nextafter(.9, 0.),
        np.nextafter(.9, 1.), .999, .1, .95, .89, .9], dtype=torch.float64)
    clean = torch.tensor(targets[plan["train_indices"].numpy()], dtype=torch.int64)
    digits = 1-clean
    digits[1], digits[6] = clean[1], clean[6]
    plan["replacement_digits"] = digits
    kw = dict(plan=plan, identity=identity, profile=subject.MLP_FIXTURE, expected_files=receipts(images, labels))
    return images, labels, kw, pixels, targets


def tensors(value):
    if type(value) is torch.Tensor:
        yield value
    elif type(value) is dict:
        for item in value.values():
            yield from tensors(item)
    elif type(value) is list:
        for item in value:
            yield from tensors(item)


class DataProbeBindingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.images, self.labels, self.kw, self.pixels, self.targets = synthetic_case()

    def make(self):
        return subject.materialize(self.images, self.labels, **self.kw)

    def validate(self, value):
        return subject.validate_materialization(value, self.images, self.labels, **self.kw)

    def test_exact_reordering_normalization_and_dataset_references(self):
        result = self.make()
        self.assertIs(self.validate(result), result)
        self.assertEqual(tuple(result), ("datasets", "probes", "bindings"))
        self.assertEqual(tuple(result["datasets"]), subject.DATASETS)
        plan, data = self.kw["plan"], result["datasets"]
        for key, indices in (("train_inputs", plan["train_indices"]), ("auxiliary_inputs", plan["auxiliary_indices"])):
            expected = torch.tensor(self.pixels[indices.numpy()], dtype=torch.float32)/255
            self.assertEqual(data[key].numpy().tobytes(), expected.numpy().tobytes())
            reference = result["bindings"]["data"]["arrays"][key]
            self.assertEqual(reference, dict(sha256=hashlib.sha256(expected.numpy().astype("<f4").tobytes()).hexdigest(),
                dtype="torch.float32", shape=[10, 3], semantic_role=key))
        self.assertEqual(data["train_inputs"][0].tolist(), [0., float(np.float32(1/255)), 1.])
        self.assertEqual(data["train_clean_labels"].tolist(), self.targets[plan["train_indices"].numpy()].tolist())
        self.assertEqual(result["bindings"]["data"]["preprocessing"], "fixture_idx_uint8_to_torch_float32_div255_v1")

    def test_exact_point_nine_boundary_and_selected_unchanged_labels(self):
        result = self.make()
        clean, noisy = (result["datasets"][name] for name in ("train_clean_labels", "train_noisy_labels"))
        self.assertEqual(result["bindings"]["data"]["replacement_count"], 5)
        self.assertEqual(result["bindings"]["data"]["incorrect_label_count"], 3)
        self.assertEqual(torch.nonzero(clean != noisy).flatten().tolist(), [0, 3, 8])
        for index in (1, 2, 4, 5, 6, 7, 9):
            self.assertEqual(clean[index].item(), noisy[index].item())
        self.assertEqual(self.kw["plan"]["replacement_uniforms"][2].item(), .9)

    def test_local_batch_probe_order_multiplicity_and_global_auxiliary_indices(self):
        result, plan = self.make(), self.kw["plan"]
        data, probes, binding = result["datasets"], result["probes"], result["bindings"]["probes"]
        self.assertEqual(tuple(probes), subject.PROBES)
        self.assertEqual(plan["training_batches"][4].tolist(), [9, 9, 1, 0])
        self.assertEqual(probes["batch_noisy"]["inputs"].numpy().tobytes(), data["train_inputs"][[9, 9, 1, 0]].numpy().tobytes())
        self.assertEqual(binding["next_batch"]["row"], 4)
        self.assertEqual(binding["training_probe_indices"].tolist(), [9, 0, 9, 3])
        self.assertEqual(probes["train_probe_clean"]["inputs"].numpy().tobytes(), data["train_inputs"][[9, 0, 9, 3]].numpy().tobytes())
        self.assertEqual(probes["train_probe_noisy"]["inputs"].numpy().tobytes(), probes["train_probe_clean"]["inputs"].numpy().tobytes())
        self.assertFalse(torch.equal(probes["train_probe_noisy"]["labels"], probes["train_probe_clean"]["labels"]))
        self.assertEqual(binding["auxiliary_indices"].tolist(), [10, 3, 27, 14, 8, 22, 16, 25, 18, 20])
        self.assertEqual(binding["auxiliary_coordinate_space"], "original_training_idx_row_v1")
        self.assertEqual(binding["stream6"], "unused")

    def test_ten_chunk_byte_references_and_no_retained_chunk_inputs(self):
        result = self.make()
        chunks, auxiliary = result["bindings"]["probes"]["auxiliary_chunks"], result["probes"]["auxiliary_clean"]
        self.assertEqual(len(chunks), 10)
        for j, row in enumerate(chunks):
            self.assertEqual(tuple(row), ("chunk_index", "start", "end", "inputs", "labels"))
            self.assertEqual((row["chunk_index"], row["start"], row["end"]), (j, j, j+1))
            self.assertEqual(row["inputs"]["sha256"], hashlib.sha256(auxiliary["inputs"][j:j+1].numpy().astype("<f4").tobytes()).hexdigest())
            self.assertEqual(row["labels"]["sha256"], hashlib.sha256(auxiliary["labels"][j:j+1].numpy().astype("<i8").tobytes()).hexdigest())
        self.assertEqual(len(list(tensors(result["bindings"]))), 2)

    def test_all_runtime_leaves_owned_isolated_from_plan_and_each_other(self):
        result = self.make()
        leaves = list(tensors(result))
        self.assertEqual(len(leaves), 15)
        addresses = [leaf.data_ptr() for leaf in leaves]
        self.assertEqual(len(set(addresses)), 15)
        self.assertTrue(all(leaf.device.type == "cpu" and leaf.is_contiguous() and leaf._base is None
                            and leaf.storage_offset() == 0 and not leaf.requires_grad for leaf in leaves))
        self.assertFalse(set(addresses) & {x.data_ptr() for x in tensors(self.kw["plan"])})
        original = result["datasets"]["train_inputs"].clone()
        result["probes"]["train_probe_clean"]["inputs"].zero_()
        self.assertTrue(torch.equal(original, result["datasets"]["train_inputs"]))
        result["bindings"]["data"]["idx_files"]["training_images"]["size_bytes"] += 1
        self.assertEqual(self.kw["expected_files"]["training_images"]["size_bytes"], len(self.images))

    def test_wrong_file_identity_rejected_before_parsing(self):
        for field, value in (("sha256", "0"*64), ("size_bytes", len(self.images)+1)):
            kw = copy.deepcopy(self.kw)
            kw["expected_files"]["training_images"][field] = value
            with mock.patch.object(subject, "_parse_idx", side_effect=AssertionError("must not parse")) as parser:
                with self.assertRaises(subject.DataBindingError):
                    subject.materialize(self.images, self.labels, **kw)
                parser.assert_not_called()
        with self.assertRaises(subject.DataBindingError):
            subject.materialize(bytearray(self.images), self.labels, **self.kw)

    def test_bad_idx_headers_counts_shapes_lengths_labels(self):
        image_cases = [b"short", struct.pack(">IIII", 2052, 30, 1, 3)+self.images[16:],
            struct.pack(">IIII", 2051, 10000, 1, 3)+self.images[16:],
            struct.pack(">IIII", 2051, 30, 3, 1)+self.images[16:], self.images[:-1], self.images+b"x"]
        label_cases = [b"short", struct.pack(">II", 2048, 30)+self.labels[8:],
            struct.pack(">II", 2049, 29)+self.labels[8:], self.labels[:-1], self.labels+b"x",
            self.labels[:8]+b"\x02"+self.labels[9:]]
        for images, labels in [(x, self.labels) for x in image_cases]+[(self.images, x) for x in label_cases]:
            kw = dict(self.kw, expected_files=receipts(images, labels))
            with self.subTest(lengths=(len(images), len(labels))), self.assertRaises(subject.DataBindingError):
                subject.materialize(images, labels, **kw)

    def test_invalid_receipt_shapes_types_and_order(self):
        mutations = [lambda e: e["training_images"].update(size_bytes=True),
                     lambda e: e["training_images"].update(sha256="A"*64),
                     lambda e: e["training_labels"].update(extra=1),
                     lambda e: e.update(extra={})]
        for mutate in mutations:
            kw = copy.deepcopy(self.kw)
            mutate(kw["expected_files"])
            with self.assertRaises(subject.DataBindingError):
                subject.materialize(self.images, self.labels, **kw)
        kw = dict(self.kw, expected_files=dict(reversed(list(self.kw["expected_files"].items()))))
        with self.assertRaises(subject.DataBindingError):
            subject.materialize(self.images, self.labels, **kw)

    def test_old_mlp_linear_and_scientific_profile_leakage_rejected(self):
        old = copy.deepcopy(self.kw)
        old["plan"]["schema"] = "i7_frozen_plan_v1"
        with mock.patch.object(subject, "_check_files", side_effect=AssertionError("must validate plan first")):
            with self.assertRaises(subject.DataBindingError):
                subject.materialize(self.images, self.labels, **old)
        for profile in ("fixture_tiny_cpu_v1", subject.SCIENTIFIC):
            with self.assertRaises(subject.DataBindingError):
                subject.materialize(self.images, self.labels, **dict(self.kw, profile=profile))

    def test_mutated_values_bindings_semantics_and_signed_zero_fail(self):
        mutations = [lambda x: x["datasets"]["train_inputs"].__setitem__((0, 1), .5),
            lambda x: x["probes"]["batch_noisy"]["labels"].__setitem__(0, 1-x["probes"]["batch_noisy"]["labels"][0]),
            lambda x: x["bindings"]["probes"].update(training_probe_coordinate_space="original_training_idx_row_v1"),
            lambda x: x["bindings"]["data"].update(replacement_count=6),
            lambda x: x["bindings"]["probes"]["auxiliary_chunks"][1].update(chunk_index=True),
            lambda x: x["bindings"]["data"]["arrays"]["train_inputs"].update(shape=(10, 3)),
            lambda x: x["bindings"]["probes"]["auxiliary_indices"].__setitem__(0, 0),
            lambda x: x.update(extra=None)]
        for mutate in mutations:
            value = self.make()
            mutate(value)
            with self.assertRaises(subject.DataBindingError):
                self.validate(value)
        value = self.make()
        value["datasets"]["train_inputs"][0, 0] = -0.
        self.assertTrue(torch.equal(value["datasets"]["train_inputs"], self.make()["datasets"]["train_inputs"]))
        with self.assertRaisesRegex(subject.DataBindingError, "exact tensor bytes differ"):
            self.validate(value)

    def test_aliases_views_nonfinite_and_tensor_metadata_fail(self):
        mutations = [lambda x: x["probes"]["train_probe_clean"].update(inputs=x["probes"]["train_probe_noisy"]["inputs"]),
            lambda x: x["bindings"]["probes"].update(training_probe_indices=self.kw["plan"]["training_probe_indices"]),
            lambda x: x["datasets"].update(train_inputs=x["datasets"]["train_inputs"].view(10, 3)),
            lambda x: x["datasets"].update(train_inputs=x["datasets"]["train_inputs"].double()),
            lambda x: x["datasets"]["train_inputs"].__setitem__((0, 0), float("nan"))]
        for mutate in mutations:
            value = self.make()
            mutate(value)
            with self.assertRaises(subject.DataBindingError):
                self.validate(value)

    def test_validation_does_not_trust_tree_digest(self):
        value = self.make()
        with mock.patch.object(subject.codec, "tree_digest", return_value="same-any-tree"):
            self.assertIs(self.validate(value), value)
            value["bindings"]["probes"]["next_batch"]["row"] = 3
            with self.assertRaises(subject.DataBindingError):
                self.validate(value)

    def test_no_rng_or_input_mutation(self):
        plan_before = subject.codec.tree_digest(self.kw["plan"])
        files_before = copy.deepcopy(self.kw["expected_files"])
        py_state, np_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
        result = self.make()
        self.validate(result)
        self.assertEqual(plan_before, subject.codec.tree_digest(self.kw["plan"]))
        self.assertEqual(files_before, self.kw["expected_files"])
        self.assertEqual(py_state, random.getstate())
        current = np.random.get_state()
        self.assertEqual(np_state[0], current[0])
        np.testing.assert_array_equal(np_state[1], current[1])
        self.assertEqual(np_state[2:], current[2:])
        self.assertTrue(torch.equal(torch_state, torch.get_rng_state()))
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == "__main__":
    unittest.main(verbosity=2)
