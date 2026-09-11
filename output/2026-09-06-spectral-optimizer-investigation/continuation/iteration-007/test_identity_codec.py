#!/usr/bin/env python3
"""Dataset-free CPU tests for the strict identity/hash/JSON codec."""
from __future__ import annotations

import hashlib
import os
import random
import struct
import unittest
import warnings

if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
    raise RuntimeError("run identity codec tests with CUDA_VISIBLE_DEVICES=''")

import numpy as np
import torch

import identity_codec as c


def scientific_identity(role="primary"):
    membership = {
        "primary": ("primary", 71001, 101, 2000),
        "sensitivity": ("sensitivity", 71901, 500, 2000),
        "pilot": ("development", 71990, 200, 220),
    }
    evidence, bundle, update, steps = membership[role]
    return {
        "run_id": "2026-09-06-spectral-optimizer-investigation",
        "iteration": 7,
        "execution_role": role,
        "evidence_role": evidence,
        "bundle": bundle,
        "anchor_update": update,
        "source_policy": "current32",
        "condition": "noise_0.9",
        "steps_total": steps,
        "anchor_phase": "pre_forward_pre_observe",
        "anchor_completed_updates": update - 1,
        "anchor_completed_observations": update - 1,
        "rng_namespace_prefix": [20260906, bundle],
        "stream_roles": dict(c.STREAM_ROLES),
    }


def fixture_identity():
    value = scientific_identity()
    value.update(bundle=0, anchor_update=5, source_policy="fixture_current2",
                 condition="fixture_only", steps_total=8,
                 anchor_completed_updates=4, anchor_completed_observations=4,
                 rng_namespace_prefix=[20260906, 0])
    return value


def clone_tree(value):
    if type(value) is dict:
        return {key: clone_tree(item) for key, item in value.items()}
    if type(value) is list:
        return [clone_tree(item) for item in value]
    if type(value) is tuple:
        return tuple(clone_tree(item) for item in value)
    if type(value) is torch.Tensor:
        return value.detach().clone()
    return value


class IdentityCodecTests(unittest.TestCase):
    def test_nested_tensor_is_a_bounded_codec_error(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            value = torch.nested.nested_tensor([torch.tensor([1.]), torch.tensor([2., 3.])])
        for call in (c.tensor_digest, c.tree_digest):
            with self.subTest(call=call.__name__), self.assertRaises(c.CodecError) as caught:
                call(value)
            self.assertLess(len(str(caught.exception)), 100)

    def test_scientific_membership_artifact_ids_and_exact_types(self):
        expected_ids = {
            "primary": "scientific_mnist_current32_v1--primary--b71001--u101--anchor",
            "sensitivity": "scientific_mnist_current32_v1--sensitivity--b71901--u500--anchor",
            "pilot": "scientific_mnist_current32_v1--pilot--b71990--u200--anchor",
        }
        for role in expected_ids:
            identity = scientific_identity(role)
            self.assertIs(c.validate_identity(identity, profile=c.SCIENTIFIC_PROFILE), identity)
            self.assertEqual(c.artifact_id(identity, profile=c.SCIENTIFIC_PROFILE, kind="anchor"),
                             expected_ids[role])
        identity = scientific_identity()
        self.assertTrue(c.artifact_id(identity, profile=c.SCIENTIFIC_PROFILE,
                                      kind="source-witness").endswith("--source-witness"))
        self.assertTrue(c.artifact_id(identity, profile=c.SCIENTIFIC_PROFILE,
                                      kind="branch-results").endswith("--branch-results"))

        for key in ("iteration", "bundle", "anchor_update", "steps_total",
                    "anchor_completed_updates", "anchor_completed_observations"):
            bad = clone_tree(identity)
            bad[key] = True
            with self.assertRaises(c.CodecError, msg=key):
                c.validate_identity(bad, profile=c.SCIENTIFIC_PROFILE)
        bad = clone_tree(identity)
        bad["rng_namespace_prefix"][0] = True
        with self.assertRaises(c.CodecError):
            c.validate_identity(bad, profile=c.SCIENTIFIC_PROFILE)
        bad = clone_tree(identity)
        bad["stream_roles"] = {True: "permutation", **dict(tuple(c.STREAM_ROLES.items())[1:])}
        with self.assertRaises(c.CodecError):
            c.validate_identity(bad, profile=c.SCIENTIFIC_PROFILE)

    def test_identity_fails_closed_on_membership_order_profiles_and_unknowns(self):
        valid = scientific_identity()
        mutations = []
        for key, value in (("bundle", 71901), ("anchor_update", 102),
                           ("evidence_role", "development"), ("steps_total", 220),
                           ("anchor_completed_updates", 101), ("source_policy", "raw")):
            bad = clone_tree(valid)
            bad[key] = value
            mutations.append(bad)
        mutations.append(dict(reversed(tuple(valid.items()))))
        missing = clone_tree(valid)
        del missing["condition"]
        mutations.append(missing)
        extra = clone_tree(valid)
        extra["extra"] = None
        mutations.append(extra)
        class Text(str):
            pass
        subclass_key = {Text(key) if index == 0 else key: clone_tree(value)
                        for index, (key, value) in enumerate(valid.items())}
        mutations.append(subclass_key)
        reordered_streams = clone_tree(valid)
        reordered_streams["stream_roles"] = dict(reversed(tuple(reordered_streams["stream_roles"].items())))
        mutations.append(reordered_streams)
        for bad in mutations:
            with self.assertRaises(c.CodecError):
                c.validate_identity(bad, profile=c.SCIENTIFIC_PROFILE)
        with self.assertRaises(c.CodecError):
            c.validate_identity(valid, profile="unknown")
        with self.assertRaises(c.CodecError):
            c.artifact_id(valid, profile=c.SCIENTIFIC_PROFILE, kind="measurements")

        fixture = fixture_identity()
        for profile in c.FIXTURE_PROFILES:
            self.assertIs(c.validate_identity(fixture, profile=profile), fixture)
        with self.assertRaises(c.CodecError):
            c.validate_identity(fixture, profile=c.SCIENTIFIC_PROFILE)
        for profile in c.FIXTURE_PROFILES:
            with self.assertRaises(c.CodecError):
                c.validate_identity(valid, profile=profile)

    def test_timestamp_is_exact_second_resolution_valid_gregorian_utc(self):
        for value in ("2024-02-29T00:00:00Z", "2026-09-06T23:59:59Z", "0001-01-01T00:00:00Z"):
            self.assertIs(c.validate_created_utc(value), value)
        for value in ("2023-02-29T00:00:00Z", "2026-13-01T00:00:00Z",
                      "2026-09-06T24:00:00Z", "2026-09-06T12:00:60Z",
                      "2026-09-06T12:00:00.0Z", "2026-09-06T12:00:00+00:00",
                      " 2026-09-06T12:00:00Z", 20260906, True):
            with self.assertRaises(c.CodecError, msg=repr(value)):
                c.validate_created_utc(value)

    def test_tensor_digest_known_answers_signed_zero_and_rejections(self):
        value = torch.tensor([0.0, -0.0, 1.0], dtype=torch.float32)
        self.assertEqual(c.tensor_digest(value),
                         "681ad5eadcfd6a5895ca2bc1c69166a4841517ecc59a7369dedf12eaef6c84ee")
        self.assertEqual(c.tensor_digest(torch.empty(0, dtype=torch.uint8)), hashlib.sha256(b"").hexdigest())
        positive = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)
        self.assertNotEqual(c.tensor_digest(value), c.tensor_digest(positive))
        uint = torch.tensor([0, 0x01020304], dtype=torch.uint32)
        self.assertEqual(c.tensor_digest(uint), hashlib.sha256(struct.pack("<II", 0, 0x01020304)).hexdigest())
        for dtype in (torch.uint8, torch.uint32, torch.int64, torch.float32, torch.float64, torch.bool):
            sample = torch.zeros(3, dtype=dtype)
            self.assertRegex(c.tensor_digest(sample), r"^[0-9a-f]{64}$")
        base = torch.zeros(8)
        malformed = [base[:2], base[:2].detach(), torch.zeros(2, 2).T,
                     torch.zeros(2, dtype=torch.float16), torch.zeros(2, dtype=torch.int32),
                     torch.tensor([float("inf")]), torch.tensor([float("nan")]),
                     torch.ones(2, requires_grad=True), np.zeros(2), [0, 1]]
        for bad in malformed:
            with self.assertRaises(c.CodecError, msg=repr(type(bad))):
                c.tensor_digest(bad)
        lazy_negative = torch._neg_view(torch.tensor([1.0], dtype=torch.float32)).detach()
        self.assertTrue(lazy_negative.is_neg())
        self.assertIsNone(lazy_negative._base)
        with self.assertRaisesRegex(c.CodecError, "lazy negative/conjugate"):
            c.tensor_digest(lazy_negative)
        with self.assertRaisesRegex(c.CodecError, "lazy negative/conjugate"):
            c.tree_digest({"lazy": lazy_negative})

    def test_typed_tree_known_answer_order_type_and_whole_tree_ownership(self):
        tensor = torch.tensor([0.0, -0.0, 1.0], dtype=torch.float32)
        tree = {"alpha": 7, 7: (None, True, -0.0), "tensor": tensor}
        self.assertEqual(c.tree_digest(tree),
                         "9c73922394a32c8487c7923b81d8f4ed9bb946e366b362e43483438e8e996a19")
        self.assertNotEqual(c.tree_digest(tree), c.tree_digest(dict(reversed(tuple(tree.items())))))
        self.assertNotEqual(c.tree_digest({7: "x"}), c.tree_digest({"7": "x"}))
        self.assertNotEqual(c.tree_digest([1, 2]), c.tree_digest((1, 2)))
        self.assertNotEqual(c.tree_digest(-0.0), c.tree_digest(0.0))
        shared = torch.arange(4, dtype=torch.int64)
        with self.assertRaisesRegex(c.CodecError, "shared tensor storage"):
            c.tree_digest({"first": shared, "second": shared.detach()})
        cyclic = []
        cyclic.append(cyclic)
        with self.assertRaises(c.CodecError):
            c.tree_digest(cyclic)
        for bad in ({"bad": torch.zeros(2, dtype=torch.float16)}, {("tuple",): 1}, {"set": {1}}):
            with self.assertRaises(c.CodecError):
                c.tree_digest(bad)

    def test_unicode_scalars_are_canonical_and_surrogate_code_points_rejected(self):
        scalar = "\U0001f600"
        surrogate_pair = "\ud83d\ude00"
        self.assertEqual(len(scalar), 1)
        self.assertEqual(len(surrogate_pair), 2)
        self.assertRegex(c.tree_digest(scalar), r"^[0-9a-f]{64}$")
        self.assertEqual(c.json_bytes(scalar), b'"\\ud83d\\ude00"\n')
        for value in (surrogate_pair, "\ud83d", "\ude00"):
            with self.assertRaisesRegex(c.CodecError, "surrogate"):
                c.tree_digest(value)
            with self.assertRaisesRegex(c.CodecError, "surrogate"):
                c.json_bytes(value)
        for mapping in ({surrogate_pair: 1}, {"\ud83d": 1}):
            with self.assertRaisesRegex(c.CodecError, "surrogate"):
                c.tree_digest(mapping)
            with self.assertRaisesRegex(c.CodecError, "surrogate"):
                c.json_bytes(mapping)
        self.assertEqual(c.json_loads(b'"\\ud83d\\ude00"', max_bytes=14), scalar)
        for encoded in (b'"\\ud83d"', b'"\\ude00"', b'{"\\ud83d":1}'):
            with self.assertRaisesRegex(c.CodecError, "surrogate"):
                c.json_loads(encoded, max_bytes=len(encoded))

    def test_strict_json_known_bytes_roundtrip_and_decode_rejections(self):
        value = {"z": 1, "a": [True, None, -0.0, "£"]}
        encoded = b'{"z":1,"a":[true,null,-0.0,"\\u00a3"]}\n'
        self.assertEqual(c.json_bytes(value), encoded)
        decoded = c.json_loads(encoded, max_bytes=len(encoded))
        self.assertEqual(tuple(decoded), ("z", "a"))
        self.assertEqual(decoded, value)
        self.assertEqual(tuple(c.json_loads(b' { "b" : 1, "a" : "\\u0061" } ', max_bytes=64)),
                         ("b", "a"))

        rejected = (b'{"a":1,"a":2}', b'{"outer":{"x":1,"x":2}}', b'NaN', b'Infinity',
                    b'-Infinity', b'1e999', b'{"a":1} {"b":2}', b'', b'\xc2\xa3')
        for data in rejected:
            with self.assertRaises(c.CodecError, msg=repr(data)):
                c.json_loads(data, max_bytes=max(1, len(data)))
        with self.assertRaises(c.CodecError):
            c.json_loads(b'{}', max_bytes=1)
        for limit in (True, 0, -1, 1.5):
            with self.assertRaises(c.CodecError):
                c.json_loads(b'{}', max_bytes=limit)
        with self.assertRaises(c.CodecError):
            c.json_loads(bytearray(b'{}'), max_bytes=2)

    def test_json_encode_rejects_nonexact_nonfinite_key_and_cycles(self):
        cycle = []
        cycle.append(cycle)

        class Integer(int):
            pass

        malformed = ((1, 2), {1: "integer key"}, float("nan"), float("inf"),
                     {"nested": cycle}, Integer(3), torch.tensor(1), np.int64(1))
        for value in malformed:
            with self.assertRaises(c.CodecError, msg=repr(type(value))):
                c.json_bytes(value)

        deep = None
        for _ in range(1200):
            deep = [deep]
        with self.assertRaises(c.CodecError):
            c.json_bytes(deep)
        with self.assertRaises(c.CodecError):
            c.tree_digest(deep)
        huge = 10 ** 5000
        with self.assertRaises(c.CodecError):
            c.json_bytes(huge)
        with self.assertRaises(c.CodecError):
            c.tree_digest(huge)

    def test_codec_does_not_consume_rng_or_initialize_cuda(self):
        python_before = random.getstate()
        numpy_before = np.random.get_state()
        torch_before = torch.get_rng_state().clone()
        identity = scientific_identity()
        c.validate_identity(identity, profile=c.SCIENTIFIC_PROFILE)
        c.artifact_id(identity, profile=c.SCIENTIFIC_PROFILE, kind="anchor")
        c.validate_created_utc("2026-09-06T12:34:56Z")
        c.tensor_digest(torch.tensor([1, 2], dtype=torch.int64))
        c.tree_digest({"x": [1, -0.0]})
        c.json_loads(c.json_bytes({"x": [1, -0.0]}), max_bytes=128)
        self.assertEqual(random.getstate(), python_before)
        after = np.random.get_state()
        self.assertEqual(after[0], numpy_before[0])
        self.assertTrue(np.array_equal(after[1], numpy_before[1]))
        self.assertEqual(after[2:], numpy_before[2:])
        self.assertTrue(torch.equal(torch.get_rng_state(), torch_before))
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == "__main__":
    unittest.main(verbosity=2)
