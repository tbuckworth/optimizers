"""Dataset-free, tiny-profile consistency/binding tests; no scientific plan."""
import copy
import hashlib
import os
import struct
import tempfile
import unittest
import warnings
from unittest import mock

if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
    raise RuntimeError("explicitly hide CUDA for plan-binding fixtures")

import numpy as np
import random
import torch

import artifact_store as storage
import plan_bindings as subject


def fixture(profile=storage.MLP_FIXTURE):
    roles = dict(enumerate(("permutation", "replacement_uniforms", "replacement_digits",
                            "initialization_seed", "training_batches", "training_probe", "unused")))
    identity = {
        "run_id": "2026-09-06-spectral-optimizer-investigation", "iteration": 7,
        "execution_role": "primary", "evidence_role": "primary", "bundle": 0,
        "anchor_update": 5, "source_policy": "fixture_current2", "condition": "fixture_only",
        "steps_total": 8, "anchor_phase": "pre_forward_pre_observe",
        "anchor_completed_updates": 4, "anchor_completed_observations": 4,
        "rng_namespace_prefix": [20260906, 0], "stream_roles": roles,
    }
    # Deliberately non-sorted global IDs distinguish subset-local batch rows.
    if profile == storage.MLP_FIXTURE:
        permutation = torch.tensor([
            17, 2, 29, 11, 5, 23, 0, 19, 7, 13,
            28, 4, 21, 9, 15, 1, 26, 12, 6, 24,
            10, 3, 27, 14, 8, 22, 16, 25, 18, 20,
        ], dtype=torch.int64)
        count, schema = 10, "i7_frozen_plan_v2"
        uniforms = torch.tensor([0., .1, .5, .899, .9, .91, .2, .8, .3, .999], dtype=torch.float64)
        digits = torch.tensor([0, 1, 1, 0, 1, 0, 1, 0, 0, 1])
        batches = torch.tensor([
            [0, 1, 1, 9], [9, 0, 2, 2], [3, 4, 5, 6], [7, 8, 9, 0],
            [9, 9, 1, 0], [2, 3, 2, 8], [4, 5, 6, 7], [8, 8, 0, 1],
        ])
        probe = torch.tensor([9, 0, 9, 3])
    else:
        permutation = torch.tensor([9, 4, 7, 2, 0, 11, 5, 8, 10, 3, 6, 1], dtype=torch.int64)
        count, schema = 4, "i7_frozen_plan_v1"
        uniforms = torch.tensor([0., .5, .9, .999], dtype=torch.float64)
        digits = torch.tensor([0, 1, 1, 0])
        batches = torch.tensor([[0, 1], [1, 0], [2, 2], [0, 3],
                                [3, 3], [3, 2], [1, 1], [2, 3]])
        probe = torch.tensor([3, 0, 3])
    plan = {
        "schema": schema, "profile": profile, "bundle": 0, "steps_total": 8,
        "stream_roles": dict(roles), "permutation": permutation,
        "train_indices": permutation[:count].clone(),
        "validation_indices": permutation[count:2 * count].clone(),
        "auxiliary_indices": permutation[2 * count:3 * count].clone(),
        "replacement_uniforms": uniforms, "replacement_digits": digits,
        "initialization_seed": 4294967295, "training_batches": batches,
        "training_probe_indices": probe,
    }
    artifact = {"sha256": hashlib.sha256(b"fixture-receipt-only").hexdigest(), "size_bytes": 123}
    return identity, plan, artifact


class PlanBindingTests(unittest.TestCase):
    def setUp(self):
        self.identity, self.plan, self.artifact = fixture()
        self.kw = dict(identity=self.identity, profile=storage.MLP_FIXTURE, artifact=self.artifact)

    def test_valid_both_fixture_profiles_and_independent_digest(self):
        for profile in storage.FIXTURES:
            identity, plan, artifact = fixture(profile)
            kw = dict(identity=identity, profile=profile, artifact=artifact)
            result = subject.make_plan_binding(plan, **kw)
            self.assertIs(subject.validate_plan_binding(result, plan=plan, **kw), result)
            expected_probe = [9, 0, 9, 3] if profile == storage.MLP_FIXTURE else [3, 0, 3]
            self.assertEqual(result["arrays"]["training_probe_indices"]["sha256"],
                             hashlib.sha256(struct.pack("<" + "q" * len(expected_probe),
                                                        *expected_probe)).hexdigest())
            self.assertEqual(result["next_batch_indices"].tolist(),
                             [9, 9, 1, 0] if profile == storage.MLP_FIXTURE else [3, 3])
            self.assertNotEqual(result["next_batch_indices"].data_ptr(), plan["training_batches"].data_ptr())
            self.assertEqual(plan["auxiliary_indices"].tolist(),
                             [10, 3, 27, 14, 8, 22, 16, 25, 18, 20]
                             if profile == storage.MLP_FIXTURE else [10, 3, 6, 1])

    def test_schema_v2_is_mlp_only_and_old_mlp_schema_is_rejected(self):
        old_mlp = copy.deepcopy(self.plan)
        old_mlp["schema"] = "i7_frozen_plan_v1"
        with self.assertRaises(subject.BindingError):
            subject.make_plan_binding(old_mlp, **self.kw)
        identity, linear, artifact = fixture(storage.FIXTURE)
        linear["schema"] = "i7_frozen_plan_v2"
        with self.assertRaises(subject.BindingError):
            subject.make_plan_binding(linear, identity=identity, profile=storage.FIXTURE,
                                      artifact=artifact)

    def test_fixture_rejected_when_scientific_expected(self):
        with self.assertRaises(subject.BindingError):
            subject.validate_plan(self.plan, identity=self.identity, profile=storage.SCIENTIFIC)

    def test_nested_tensor_rejected_as_bounded_binding_error(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            value = torch.nested.nested_tensor([torch.tensor([1.]), torch.tensor([2., 3.])])
        self.plan["replacement_uniforms"] = value
        with self.assertRaises(subject.BindingError) as caught:
            subject.make_plan_binding(self.plan, **self.kw)
        self.assertLess(len(str(caught.exception)), 100)

    def test_plan_metadata_is_exact_order_and_type(self):
        mutations = [lambda p: p.update(bundle=False), lambda p: p.update(steps_total=8.),
                     lambda p: p.update(initialization_seed=True),
                     lambda p: p.update(initialization_seed=2**32),
                     lambda p: p.update(initialization_seed=-1),
                     lambda p: p.update(extra=None),
                     lambda p: p.update(stream_roles={str(k): v for k, v in p["stream_roles"].items()}),
                     lambda p: p.update(stream_roles=dict(reversed(list(p["stream_roles"].items()))))]
        for mutate in mutations:
            plan = copy.deepcopy(self.plan)
            mutate(plan)
            with self.subTest(plan=tuple(plan)), self.assertRaises(subject.BindingError):
                subject.make_plan_binding(plan, **self.kw)
        with self.assertRaises(subject.BindingError):
            subject.make_plan_binding(dict(reversed(list(self.plan.items()))), **self.kw)

    def test_partition_permutation_dtype_shape_and_bounds(self):
        cases = [
            ("permutation", torch.zeros(30, dtype=torch.int64)),
            ("train_indices", torch.tensor([0, 1, 2, 3])),
            ("validation_indices", torch.tensor([0, 29, 8, 5])),
            ("auxiliary_indices", torch.tensor([1, 3, 6, 10])),
            ("replacement_uniforms", torch.tensor([0., .5, .9, .1, .2, .3, .4, .5, .6, 1.],
                                                   dtype=torch.float64)),
            ("replacement_uniforms", torch.tensor([0., .5, .9, .1, .2, .3, .4, .5, .6, float("nan")],
                                                   dtype=torch.float64)),
            ("replacement_uniforms", torch._neg_view(torch.zeros(10, dtype=torch.float64)).detach()),
            ("replacement_digits", torch.tensor([0, 1, 1, 0, 1, 0, 1, 0, 2, 1])),
            ("training_probe_indices", torch.tensor([3, -1, 3, 0])),
            ("training_probe_indices", torch.tensor([10, 4, 9, 0])),
            ("training_probe_indices", torch.tensor([3., 0., 3., 1.])),
            ("training_probe_indices", torch.tensor([[3, 0, 3, 1]])),
            ("training_batches", torch.zeros((8, 3), dtype=torch.int64)),
        ]
        for field, value in cases:
            plan = copy.deepcopy(self.plan)
            plan[field] = value
            with self.subTest(field=field), self.assertRaises(subject.BindingError):
                subject.make_plan_binding(plan, **self.kw)

    def test_no_alias_view_or_silent_repair(self):
        for field, value in (("train_indices", self.plan["permutation"][:10]),
                             ("auxiliary_indices", self.plan["train_indices"]),
                             ("training_probe_indices", torch.arange(12)[::3])):
            plan = dict(self.plan)
            plan[field] = value
            with self.subTest(field=field), self.assertRaises(subject.BindingError):
                subject.make_plan_binding(plan, **self.kw)

    def test_binding_mutations_rejected(self):
        binding = subject.make_plan_binding(self.plan, **self.kw)
        mutations = [
            lambda b: b.update(next_batch_row=3),
            lambda b: b.update(next_batch_row=True),
            lambda b: b.update(next_batch_indices=torch.tensor([3, 2])),
            lambda b: b["artifact"].update(size_bytes=124),
            lambda b: b["artifact"].update(sha256="A" * 64),
            lambda b: b["initialization_seed"].update(value=17),
            lambda b: b["arrays"]["train_indices"].update(shape=[True]),
            lambda b: b["arrays"]["training_batches"].update(semantic_role="training_probe_indices"),
            lambda b: b["arrays"]["training_batches"].update(dtype="int64"),
            lambda b: b["arrays"].update(extra=None),
        ]
        for mutate in mutations:
            value = copy.deepcopy(binding)
            mutate(value)
            with self.subTest(mutation=mutate), self.assertRaises(subject.BindingError):
                subject.validate_plan_binding(value, plan=self.plan, **self.kw)
        with self.assertRaises(subject.BindingError):
            subject.validate_plan_binding(dict(reversed(list(binding.items()))), plan=self.plan, **self.kw)

    def test_stale_binding_detects_changes_outside_next_batch(self):
        binding = subject.make_plan_binding(self.plan, **self.kw)
        for field, index, value in (("training_batches", (0, 0), 1),
                                    ("training_probe_indices", 0, 2),
                                    ("replacement_uniforms", 0, .1)):
            plan = copy.deepcopy(self.plan)
            plan[field][index] = value
            with self.subTest(field=field), self.assertRaises(subject.BindingError):
                subject.validate_plan_binding(binding, plan=plan, **self.kw)

    def test_structural_validation_does_not_depend_on_tree_hash_equality(self):
        binding = subject.make_plan_binding(self.plan, **self.kw)
        with mock.patch.object(subject.codec, "tree_digest",
                               return_value=binding["initialization_seed"]["sha256"]):
            subject.validate_plan_binding(binding, plan=self.plan, **self.kw)
            binding["next_batch_row"] = 3
            with self.assertRaises(subject.BindingError):
                subject.validate_plan_binding(binding, plan=self.plan, **self.kw)
        plan = copy.deepcopy(self.plan)
        plan["stream_roles"][6] = "unexpected"
        with mock.patch.object(subject.codec, "tree_digest", return_value="same-for-any-input"):
            with self.assertRaises(subject.BindingError):
                subject.make_plan_binding(plan, **self.kw)

    def test_invalid_external_receipt_rejected(self):
        for receipt in ({"sha256": "0" * 64, "size_bytes": True},
                        {"sha256": "0" * 64, "size_bytes": 0},
                        {"sha256": "0" * 63, "size_bytes": 1},
                        {"size_bytes": 1, "sha256": "0" * 64}):
            with self.assertRaises(subject.BindingError):
                subject.make_plan_binding(self.plan, identity=self.identity,
                                          profile=storage.MLP_FIXTURE, artifact=receipt)

    def test_restricted_store_roundtrip(self):
        with tempfile.TemporaryDirectory(prefix="i7-plan-fixture-") as directory:
            with storage.ArtifactStore(directory, profile=storage.MLP_FIXTURE,
                                       min_filesystem_free_bytes=0) as store:
                receipt = store.write_tensor_tree("fixture-plan.pt", self.plan)
                artifact = {"sha256": receipt["sha256"], "size_bytes": receipt["size"]}
                loaded = store.load_tensor_tree(store.root, "fixture-plan.pt",
                    expected_size=receipt["size"], expected_sha256=receipt["sha256"])
                binding = subject.make_plan_binding(loaded, identity=self.identity,
                                                    profile=storage.MLP_FIXTURE, artifact=artifact)
                bound_receipt = store.write_tensor_tree("fixture-binding.pt", binding)
                result = store.load_tensor_tree(store.root, "fixture-binding.pt",
                    expected_size=bound_receipt["size"], expected_sha256=bound_receipt["sha256"])
                subject.validate_plan_binding(result, plan=loaded, identity=self.identity,
                                              profile=storage.MLP_FIXTURE, artifact=artifact)

    def test_does_not_consume_rng_or_mutate_inputs(self):
        before = copy.deepcopy(self.plan)
        py_rng, np_rng, torch_rng = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
        binding = subject.make_plan_binding(self.plan, **self.kw)
        subject.validate_plan_binding(binding, plan=self.plan, **self.kw)
        self.assertEqual(subject.codec.tree_digest(before), subject.codec.tree_digest(self.plan))
        self.assertEqual(py_rng, random.getstate())
        current_np = np.random.get_state()
        self.assertEqual(np_rng[0], current_np[0])
        np.testing.assert_array_equal(np_rng[1], current_np[1])
        self.assertEqual(np_rng[2:], current_np[2:])
        self.assertTrue(torch.equal(torch_rng, torch.get_rng_state()))
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == "__main__":
    unittest.main()
