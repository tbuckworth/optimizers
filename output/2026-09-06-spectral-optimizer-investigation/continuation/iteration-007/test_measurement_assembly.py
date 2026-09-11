#!/usr/bin/env python3
"""Dataset-free CPU fixtures for measurement_assembly.py."""
from __future__ import annotations

import hashlib
import importlib.util
import math
import os
from pathlib import Path
import random
import unittest
from unittest import mock

if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
    raise RuntimeError("Run assembly fixtures with CUDA_VISIBLE_DEVICES=''")

import numpy as np
import torch


HERE = Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


a = load("i7_test_assembly", HERE / "measurement_assembly.py")
r = load("i7_test_assembly_response", HERE / "response_math.py")


def clone_parameters(parameters):
    return {key: value.detach().clone() for key, value in parameters.items()}


def clone_tree(value):
    if type(value) is dict:
        return {key: clone_tree(item) for key, item in value.items()}
    if type(value) is list:
        return [clone_tree(item) for item in value]
    if type(value) is torch.Tensor:
        return value.detach().clone()
    return value


def fixture_parameters():
    values = torch.linspace(-0.31, 0.37, 26, dtype=torch.float32)
    shapes = ((4, 3), (4,), (2, 4), (2,))
    result, offset = {}, 0
    for key, shape in zip(a.PARAMETERS, shapes):
        size = math.prod(shape)
        result[key] = values[offset:offset + size].reshape(shape).clone()
        offset += size
    return result


def fixture_probes():
    base = torch.tensor([[.1, .4, .9], [.7, .2, .5], [.9, .8, .1], [.3, .6, .2]],
                        dtype=torch.float32)
    labels = torch.tensor([0, 1, 1, 0], dtype=torch.int64)
    auxiliary = torch.cat([base, torch.flip(base, [0]), base[:2]], dim=0).contiguous()
    auxiliary_labels = torch.tensor([0, 1, 1, 0, 0, 1, 1, 0, 0, 1], dtype=torch.int64)
    return {
        "batch_noisy": {"inputs": base.clone(), "labels": labels.clone()},
        "train_probe_noisy": {"inputs": base.clone(), "labels": (1 - labels).clone()},
        "train_probe_clean": {"inputs": base.clone(), "labels": labels.clone()},
        "auxiliary_clean": {"inputs": auxiliary.clone(), "labels": auxiliary_labels.clone()},
    }


def fixture_candidates(*, undefined=False, weak=False):
    raw = torch.linspace(-.4, .6, 26, dtype=torch.float32)
    current = torch.linspace(.2, -.5, 26, dtype=torch.float32)
    if undefined:
        lagged = torch.zeros(26, dtype=torch.float32)
    elif weak:
        lagged = current.clone()
    else:
        lagged = torch.roll(current, 3) * torch.tensor(.73, dtype=torch.float32)
    return r.construct(raw, current, lagged)


def fixture_endpoints(before, candidates):
    result = {}
    for branch in a.BRANCHES:
        gradient = candidates["branches"][branch]["gradient"]
        if gradient is None:
            result[branch] = None
            continue
        values, offset = {}, 0
        for key, parameter in before.items():
            piece = gradient[offset:offset + parameter.numel()].reshape(parameter.shape)
            values[key] = (parameter.double() - .003 * piece.double()).float().contiguous().clone()
            offset += parameter.numel()
        result[branch] = values
    return result


def run_assembly(*, undefined=False, weak=False, guard=None):
    before = fixture_parameters()
    candidates = fixture_candidates(undefined=undefined, weak=weak)
    endpoints = fixture_endpoints(before, candidates)
    probes = fixture_probes()
    result = a.assemble(before, endpoints, candidates, probes,
                        artifact_id="fixture-anchor-measurements", profile=a.FIXTURE_PROFILE,
                        guard=guard)
    return result, before, endpoints, candidates, probes


class MeasurementAssemblyTests(unittest.TestCase):
    def test_complete_schema_identities_hashes_guards_and_ownership(self):
        python_state, numpy_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
        labels = []
        result, before, endpoints, candidates, probes = run_assembly(guard=labels.append)

        self.assertEqual(tuple(result), ("measurement_before", "branches", "comparisons"))
        self.assertEqual(result["measurement_before"]["probe_order"], list(a.PROBES))
        self.assertEqual(tuple(result["measurement_before"]["probes"]), a.PROBES)
        self.assertEqual(tuple(result["branches"]), a.BRANCHES)
        self.assertEqual(tuple(result["comparisons"]), ("branch_pairs", "contrasts"))
        self.assertEqual(tuple(result["comparisons"]["branch_pairs"]), a.PAIR_KEYS)
        self.assertEqual(tuple(result["comparisons"]["contrasts"]), tuple(a.COEFFICIENTS))

        for probe, count in zip(a.PROBES, (4, 4, 4, 10)):
            row = result["measurement_before"]["probes"][probe]
            self.assertEqual(row["sample_count"], count)
            self.assertEqual(row["cpu64"]["q"]["shape"], [26])
            self.assertEqual(row["cpu64"]["q"]["dtype"], "float64")
            raw = (probes[probe]["inputs"].numpy().astype("<f4", copy=False).tobytes() +
                   probes[probe]["labels"].numpy().astype("<i8", copy=False).tobytes())
            self.assertEqual(row["sample_identity_sha256"], hashlib.sha256(raw).hexdigest())
            self.assertEqual(len(row["auxiliary_chunks"]), 10 if probe == "auxiliary_clean" else 0)
        self.assertEqual([(x["start"], x["end"], x["count"]) for x in
                          result["measurement_before"]["probes"]["auxiliary_clean"]
                          ["auxiliary_chunks"]], [(i, i + 1, 1) for i in range(10)])

        for branch in a.BRANCHES:
            row = result["branches"][branch]
            self.assertEqual(tuple(row), ("displacement", "probes"))
            self.assertEqual(tuple(row["probes"]), a.PROBES)
            delta = row["displacement"]["delta"]["value"]
            data = row["displacement"]["delta_data"]["value"]
            before_flat = torch.cat([x.double().reshape(-1) for x in before.values()])
            after_flat = torch.cat([x.double().reshape(-1) for x in endpoints[branch].values()])
            self.assertTrue(torch.equal(delta, after_flat - before_flat))
            self.assertTrue(torch.equal(data, delta + (.001 * .01) * before_flat))
            for probe in a.PROBES:
                measurement = row["probes"][probe]
                self.assertEqual(measurement["before_binding"]["measurement_before_artifact_id"],
                                 "fixture-anchor-measurements")
                self.assertEqual(measurement["before_binding"]["probe_key"], probe)

        for pair in result["comparisons"]["branch_pairs"].values():
            self.assertTrue(pair["defined"])
            self.assertEqual(pair["rounding_status"], "pass")
        for contrast in result["comparisons"]["contrasts"].values():
            self.assertTrue(contrast["defined"])
            self.assertEqual(contrast["vector"]["full_data_agreement"]["status"], "pass")
            for probe in a.PROBES:
                scalars = contrast["probes"][probe]
                for record in scalars["cpu64"].values():
                    self.assertEqual(record["identity_status"], "pass")
                    self.assertEqual(tuple(record["component_values"]), tuple(contrast["coefficients"]))
                self.assertEqual(scalars["native32"]["Y"]["identity_status"], "pass")
                self.assertEqual(len(scalars["auxiliary_chunks"]),
                                 10 if probe == "auxiliary_clean" else 0)

        interaction = result["comparisons"]["contrasts"]["interaction"]
        self.assertEqual(interaction["factor_requirements"], ["direction", "norm"])
        self.assertEqual(interaction["factor_leverage"],
                         {"direction": "qualified", "norm": "qualified"})
        y = interaction["probes"]["batch_noisy"]["cpu64"]["Y"]["direct_value"]
        first = (result["comparisons"]["contrasts"]["norm_at_lagged_direction"]
                 ["probes"]["batch_noisy"]["cpu64"]["Y"]["direct_value"] -
                 result["comparisons"]["contrasts"]["norm_at_current_direction"]
                 ["probes"]["batch_noisy"]["cpu64"]["Y"]["direct_value"])
        second = (result["comparisons"]["contrasts"]["direction_at_current_norm"]
                  ["probes"]["batch_noisy"]["cpu64"]["Y"]["direct_value"] -
                  result["comparisons"]["contrasts"]["direction_at_lagged_norm"]
                  ["probes"]["batch_noisy"]["cpu64"]["Y"]["direct_value"])
        self.assertEqual(y, first)
        self.assertEqual(y, second)

        self.assertEqual(sum(x.startswith("loss.before_chunk.") for x in labels), 182)
        self.assertEqual(sum(x.startswith("loss.after_chunk.") for x in labels), 182)
        self.assertIn("before:branch:raw:displacement", labels)
        self.assertIn("after:contrast:interaction", labels)
        self.assertIn("before:assembly:validation", labels)
        self.assertIn("after:assembly:validation", labels)
        self.assertIn("before:assembly:before_flat_copy", labels)
        self.assertIn("after:assembly:before_flat_copy", labels)
        self.assertEqual(random.getstate(), python_state)
        self.assertTrue(np.array_equal(np.random.get_state()[1], numpy_state[1]))
        self.assertTrue(torch.equal(torch.get_rng_state(), torch_state))
        self.assertFalse(torch.cuda.is_initialized())

        original = before["0.weight"].clone()
        saved_data = result["branches"]["raw"]["displacement"]["delta_data"]["value"].clone()
        result["branches"]["raw"]["displacement"]["delta"]["value"].add_(10)
        result["measurement_before"]["probes"]["batch_noisy"]["cpu64"]["q"]["value"].zero_()
        self.assertTrue(torch.equal(result["branches"]["raw"]["displacement"]
                                    ["delta_data"]["value"], saved_data))
        self.assertTrue(torch.equal(before["0.weight"], original))
        self.assertTrue(torch.equal(endpoints["raw"]["0.weight"],
                                    fixture_endpoints(before, candidates)["raw"]["0.weight"]))

    def test_domain_null_propagates_without_available_case_contrast(self):
        result, _, endpoints, candidates, _ = run_assembly(undefined=True)
        self.assertIsNone(endpoints["restored"])
        self.assertEqual(candidates["branches"]["restored"]["reason"],
                         "positive_current_norm_zero_lagged_direction")
        self.assertIsNone(result["branches"]["restored"])
        affected = result["comparisons"]["contrasts"]["direction_at_current_norm"]
        self.assertFalse(affected["defined"])
        self.assertEqual(affected["reason"], "domain_undefined_required_branch")
        self.assertIsNone(affected["vector"])
        self.assertIsNone(affected["probes"])
        unaffected = result["comparisons"]["contrasts"]["current_minus_raw"]
        self.assertTrue(unaffected["defined"])
        self.assertIsNotNone(unaffected["probes"])
        pair = result["comparisons"]["branch_pairs"]["raw__restored"]
        self.assertFalse(pair["defined"])
        self.assertEqual(pair["rounding_status"], "domain_undefined")
        self.assertEqual(pair["defined_mask"], {"raw": True, "restored": False})

    def test_weak_leverage_retains_all_numeric_measurements(self):
        result, *_ = run_assembly(weak=True)
        interaction = result["comparisons"]["contrasts"]["interaction"]
        self.assertTrue(interaction["defined"])
        self.assertEqual(interaction["factor_leverage"], {"direction": "weak", "norm": "weak"})
        self.assertIsNotNone(interaction["vector"])
        self.assertIsNotNone(interaction["probes"])
        ordering = result["comparisons"]["contrasts"]["ordering"]
        self.assertEqual(ordering["factor_requirements"], [])
        self.assertEqual(ordering["factor_leverage"],
                         {"direction": "not_required", "norm": "not_required"})

    def test_strict_rejections_and_guard_exception(self):
        before = fixture_parameters()
        candidates = fixture_candidates()
        endpoints = fixture_endpoints(before, candidates)
        probes = fixture_probes()

        bad_endpoints = dict(reversed(tuple(endpoints.items())))
        with self.assertRaisesRegex(ValueError, "endpoints keys/order"):
            a.assemble(before, bad_endpoints, candidates, probes, artifact_id="x",
                       profile=a.FIXTURE_PROFILE)
        bad_probes = dict(probes)
        bad_probes["extra"] = bad_probes["batch_noisy"]
        with self.assertRaisesRegex(ValueError, "probes keys/order"):
            a.assemble(before, endpoints, candidates, bad_probes, artifact_id="x",
                       profile=a.FIXTURE_PROFILE)
        bad_null = dict(endpoints)
        bad_null["raw"] = None
        with self.assertRaisesRegex(ValueError, "endpoint/candidate domain"):
            a.assemble(before, bad_null, candidates, probes, artifact_id="x",
                       profile=a.FIXTURE_PROFILE)
        with self.assertRaisesRegex(ValueError, "profile/native device-kind"):
            a.assemble(before, endpoints, candidates, probes, artifact_id="x",
                       profile=a.SCIENTIFIC_PROFILE)
        bad_id = "bad/id"
        with self.assertRaisesRegex(ValueError, "canonical ASCII"):
            a.assemble(before, endpoints, candidates, probes, artifact_id=bad_id,
                       profile=a.FIXTURE_PROFILE)

        forged = clone_tree(candidates)
        forged["leverage"]["direction_leverage"] = not forged["leverage"]["direction_leverage"]
        with self.assertRaisesRegex(ValueError, "not bound"):
            a.assemble(before, endpoints, forged, probes, artifact_id="x",
                       profile=a.FIXTURE_PROFILE)
        forged = clone_tree(candidates)
        forged["branches"]["raw"]["gradient"][0] = torch.nextafter(
            forged["branches"]["raw"]["gradient"][0], torch.tensor(float("inf")))
        with self.assertRaisesRegex(ValueError, "not bound"):
            a.assemble(before, endpoints, forged, probes, artifact_id="x",
                       profile=a.FIXTURE_PROFILE)
        forged = clone_tree(candidates)
        forged["delivered_pairs"]["raw__current"]["cosine_reason"] = "zero_norm"
        with self.assertRaisesRegex(ValueError, "not bound"):
            a.assemble(before, endpoints, forged, probes, artifact_id="x",
                       profile=a.FIXTURE_PROFILE)

        corrupted_inputs = clone_tree(probes)
        corrupted_inputs["train_probe_noisy"]["inputs"][0, 0] += 1
        with self.assertRaisesRegex(ValueError, "byte-identical ordered inputs"):
            a.assemble(before, endpoints, candidates, corrupted_inputs, artifact_id="x",
                       profile=a.FIXTURE_PROFILE)
        bad_labels = clone_tree(probes)
        bad_labels["batch_noisy"]["labels"][0] = 2
        with self.assertRaisesRegex(ValueError, "outside class range"):
            a.assemble(before, endpoints, candidates, bad_labels, artifact_id="x",
                       profile=a.FIXTURE_PROFILE)

        class StopGuard(RuntimeError):
            pass

        def stopping_guard(label):
            if label.startswith("loss.before_chunk."):
                raise StopGuard(label)

        with self.assertRaises(StopGuard):
            a.assemble(before, endpoints, candidates, probes, artifact_id="x",
                       profile=a.FIXTURE_PROFILE, guard=stopping_guard)
        self.assertFalse(torch.cuda.is_initialized())

    def test_fatal_status_is_an_exception_with_retained_record(self):
        record = {"identity_status": "fatal_validation", "identity_abs_discrepancy": 1.0,
                  "identity_rounding_ceiling": 0.0}
        with self.assertRaises(a.AssemblyFailure) as caught:
            a._require_pass("contrasts.synthetic.cpu64.Y", record, "identity_status")
        self.assertEqual(caught.exception.location, "contrasts.synthetic.cpu64.Y")
        self.assertEqual(caught.exception.record, record)
        record["identity_abs_discrepancy"] = 9.0
        self.assertEqual(caught.exception.record["identity_abs_discrepancy"], 1.0)

        before = fixture_parameters()
        candidates = fixture_candidates()
        endpoints = fixture_endpoints(before, candidates)
        probes = fixture_probes()
        original_linear, calls = a._linear, {"count": 0}

        def corrupt_second_route(values, coefficients):
            calls["count"] += 1
            value = original_linear(values, coefficients)
            return value + 1.0 if calls["count"] == 2 else value

        with mock.patch.object(a, "_linear", side_effect=corrupt_second_route):
            with self.assertRaises(a.AssemblyFailure) as assembled:
                a.assemble(before, endpoints, candidates, probes, artifact_id="fatal-fixture",
                           profile=a.FIXTURE_PROFILE)
        self.assertEqual(assembled.exception.location,
                         "contrasts.ordering.probes.batch_noisy.cpu64.Y")
        self.assertEqual(assembled.exception.record["identity_status"], "fatal_validation")

    def test_no_grad_and_autocast_are_preserved(self):
        before = fixture_parameters()
        candidates = fixture_candidates()
        endpoints = fixture_endpoints(before, candidates)
        probes = fixture_probes()
        with torch.no_grad(), torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            self.assertFalse(torch.is_grad_enabled())
            self.assertTrue(torch.is_autocast_enabled("cpu"))
            result = a.assemble(before, endpoints, candidates, probes, artifact_id="modes",
                                profile=a.FIXTURE_PROFILE)
            self.assertFalse(torch.is_grad_enabled())
            self.assertTrue(torch.is_autocast_enabled("cpu"))
        self.assertTrue(math.isfinite(result["measurement_before"]["probes"]
                                      ["batch_noisy"]["cpu64"]["before_ce"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
