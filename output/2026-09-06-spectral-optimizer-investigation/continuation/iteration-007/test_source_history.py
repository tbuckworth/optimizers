#!/usr/bin/env python3
"""Dataset-free CPU checks for bounded source-history records."""
import copy
import hashlib
import os
from pathlib import Path
import random
import sys
import unittest
from unittest import mock

import numpy as np
import torch
import torch.nn.functional as F


os.environ["CUDA_VISIBLE_DEVICES"] = ""
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))

import artifact_store as storage
import identity_codec as codec
import source_history as history
import state_core as state
from spectral_filter import SpectralGradientFilter


class StringSubclass(str):
    pass


X = torch.tensor([[.2, -.4, .7], [1., .5, -.25], [-.3, .9, .1], [.6, -.2, .4]])
Y = torch.tensor([0, 1, 1, 0])


def identity(*, role="primary", bundle=0, update=5, profile=state.MLP_FIXTURE_PROFILE):
    fixture = profile == state.MLP_FIXTURE_PROFILE
    evidence = "primary" if role == "primary" else ("sensitivity" if role == "sensitivity" else "development")
    steps = 8 if fixture else (220 if role == "pilot" else 2000)
    return {
        "run_id": "2026-09-06-spectral-optimizer-investigation", "iteration": 7,
        "execution_role": role, "evidence_role": evidence, "bundle": bundle,
        "anchor_update": update,
        "source_policy": "fixture_current2" if fixture else "current32",
        "condition": "fixture_only" if fixture else "noise_0.9",
        "steps_total": steps, "anchor_phase": "pre_forward_pre_observe",
        "anchor_completed_updates": update - 1,
        "anchor_completed_observations": update - 1,
        "rng_namespace_prefix": [20260906, bundle],
        "stream_roles": dict(codec.STREAM_ROLES),
    }


def model_factory():
    return torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.ReLU(),
                               torch.nn.Linear(4, 2))


def optimizer_factory(model):
    return torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01,
                             betas=(.9, .999), eps=1e-8, foreach=False,
                             fused=False)


def observer_factory(model, optimizer):
    return SpectralGradientFilter(model, optimizer, rank=2, decay=.99,
                                  warmup=2, stable_update=True)


def run_history(seed=7317):
    random.seed(seed)
    np.random.seed(seed + 1)
    torch.manual_seed(seed + 2)
    model = model_factory()
    optimizer = optimizer_factory(model)
    observer = observer_factory(model, optimizer)
    initial = history.initial_fingerprint(model, optimizer, observer,
                                          profile=state.MLP_FIXTURE_PROFILE)
    cores = []
    for completed in range(1, 9):
        optimizer.zero_grad(set_to_none=True)
        F.cross_entropy(model(X), Y).backward()
        observer.filter_grad()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        cores.append(state.capture_core(model, optimizer, observer,
            profile=state.MLP_FIXTURE_PROFILE, completed_updates=completed))
    return initial, cores


def receipt_fields(name, *, digest="1" * 64, size=123):
    base = {"schema": "i7_artifact_receipt_v1", "name": name, "size": size,
            "sha256": digest, "status": "complete", "encoding": "torch_weights_only"}
    data = storage._json_bytes(base)
    return {
        "name": name, "status": "complete", "encoding": "torch_weights_only",
        "size_bytes": size, "sha256": digest,
        "receipt_name": "receipt-" + hashlib.sha256(name.encode("ascii")).hexdigest() + ".json",
        "receipt_size_bytes": len(data), "receipt_sha256": hashlib.sha256(data).hexdigest(),
    }


def plan_ref():
    return receipt_fields("frozen-plan.pt", digest="2" * 64, size=456)


def artifact_ref(kind):
    ident = identity()
    artifact_id = codec.artifact_id(ident, profile=state.MLP_FIXTURE_PROFILE, kind=kind)
    result = {
        "artifact_id": artifact_id,
        "schema_name": "i7_anchor" if kind == "anchor" else "i7_source_step_witness",
    }
    result.update(receipt_fields(artifact_id + ".pt", digest=("3" if kind == "anchor" else "4") * 64))
    # receipt_fields starts with name, whereas the source-reference contract has
    # identity fields first and a different exact order.
    return {key: result[key] for key in history.ARTIFACT_REF_KEYS}


class SourceHistoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
            raise RuntimeError("source-history tests require hidden CUDA")
        if torch.cuda.is_initialized():
            raise RuntimeError("source-history tests must not initialize CUDA")

    def setUp(self):
        self.on_initial, self.on = run_history()
        self.off_initial, self.off = run_history()

    def source_completion(self, cores, mode, *, plan=None, sources="5" * 64,
                          environment="6" * 64):
        enabled = mode == "capture_on"
        return history.make_source_completion(
            cores, identity=identity(), profile=state.MLP_FIXTURE_PROFILE,
            capture_mode=mode, plan_ref=plan_ref() if plan is None else plan,
            anchor_refs=[artifact_ref("anchor")] if enabled else [],
            witness_refs=[artifact_ref("source-witness")] if enabled else [],
            sources_sha256=sources, environment_sha256=environment)

    def capture_comparison(self, off=None):
        return history.make_capture_comparison(
            zip(self.on, self.off if off is None else off),
            initial_states=(self.on_initial, self.off_initial),
            identity=identity(), profile=state.MLP_FIXTURE_PROFILE)

    def test_registered_trajectory_membership_only(self):
        fixture = history.trajectory(identity(), state.MLP_FIXTURE_PROFILE)
        self.assertEqual((fixture["steps_total"], fixture["anchor_updates"]), (8, [5]))
        cases = (
            ("pilot", 71990, 101, 220, [101, 200]),
            ("primary", 71001, 500, 2000, [101, 500, 1000, 2000]),
            ("primary", 71002, 1000, 2000, [101, 500, 1000, 2000]),
            ("primary", 71003, 2000, 2000, [101, 500, 1000, 2000]),
            ("sensitivity", 71901, 101, 2000, [101, 500, 1000, 2000]),
        )
        for role, bundle, update, steps, anchors in cases:
            got = history.trajectory(identity(role=role, bundle=bundle, update=update,
                                     profile=state.SCIENTIFIC_PROFILE),
                                     state.SCIENTIFIC_PROFILE)
            self.assertEqual((got["steps_total"], got["anchor_updates"]), (steps, anchors))
        with self.assertRaises(history.SourceHistoryError):
            history.trajectory(identity(role="pilot", bundle=71990, update=5,
                               profile=state.SCIENTIFIC_PROFILE), state.SCIENTIFIC_PROFILE)
        with self.assertRaises(history.SourceHistoryError):
            history.trajectory(identity(), state.FIXTURE_PROFILE)

    def test_initial_state_is_actual_empty_constructor_state_and_rng_neutral(self):
        self.assertEqual(self.on_initial["state_completed_updates"], 0)
        self.assertEqual(self.on_initial["optimizer"]["state"], [])
        self.assertEqual(self.on_initial["observer"]["state"]["step_count"], 0)
        self.assertIsNone(self.on_initial["observer"]["state"]["grad_mean"])
        comparison = history.compare_initial(self.on_initial, self.off_initial)
        self.assertTrue(comparison["state_direct_typed_equal"])
        corrupt = state.clone_tree(self.on_initial)
        corrupt["optimizer"]["state_kind"] = "forged"
        with self.assertRaises(history.SourceHistoryError):
            history.validate_initial_fingerprint(corrupt)

    def test_fixed_declarations_require_exact_builtin_types(self):
        corrupt = state.clone_tree(self.on_initial)
        corrupt["schema_name"] = StringSubclass(corrupt["schema_name"])
        with self.assertRaises(history.SourceHistoryError):
            history.validate_initial_fingerprint(corrupt)

        corrupt = state.clone_tree(self.on_initial)
        corrupt["optimizer"]["parameter_order"] = tuple(
            corrupt["optimizer"]["parameter_order"])
        with self.assertRaises(history.SourceHistoryError):
            history.validate_initial_fingerprint(corrupt)
        corrupt = state.clone_tree(self.on_initial)
        corrupt["optimizer"]["class_name"] = StringSubclass("torch.optim.AdamW")
        with self.assertRaises(history.SourceHistoryError):
            history.validate_initial_fingerprint(corrupt)

    def test_initial_observer_rejects_historical_state_at_counter_zero(self):
        mutations = (dict(stabilization_count=1), dict(max_orthogonality_error=.1),
                     dict(max_orthogonality_error=-0.0),
                     {key: self.on[-1]["observer"]["state"][key] for key in ("V", "S")})
        for mutation in mutations:
            corrupt = state.clone_tree(self.on_initial)
            corrupt["observer"]["state"].update(state.clone_tree(mutation))
            with self.subTest(fields=list(mutation)), self.assertRaisesRegex(
                    history.SourceHistoryError, "empty constructor state"):
                history.validate_initial_fingerprint(corrupt)

    def test_step_comparison_checks_typed_bytes_before_retaining_hashes(self):
        row = history.compare_step(self.on[0], self.off[0])
        self.assertTrue(row["core_direct_typed_equal"])
        with self.assertRaises(TypeError):
            history.compare_step(self.on[0], self.off[0], expected_equal=True)

        on_zero = state.clone_tree(self.on[0])
        off_zero = state.clone_tree(self.off[0])
        on_zero["model"]["parameters"][0]["value"].view(-1)[0] = 0.0
        off_zero["model"]["parameters"][0]["value"].view(-1)[0] = -0.0
        signed = history.compare_step(on_zero, off_zero)
        self.assertFalse(signed["model_direct_typed_equal"])
        self.assertNotEqual(signed["on"]["model_sha256"], signed["off"]["model_sha256"])

    def test_mutated_moment_observer_and_rng_are_component_local(self):
        mutations = (
            ("optimizer", lambda core: core["optimizer"]["state"][0]["exp_avg"]["value"].view(-1).__setitem__(0, .25)),
            ("observer", lambda core: core["observer"]["state"]["grad_mean"]["value"].view(-1).__setitem__(0, .25)),
            ("rng", lambda core: core["rng"]["torch_cpu"].__setitem__(0, int(core["rng"]["torch_cpu"][0]) ^ 1)),
        )
        for component, mutate in mutations:
            changed = state.clone_tree(self.off[0])
            mutate(changed)
            row = history.compare_step(self.on[0], changed)
            self.assertFalse(row[component + "_direct_typed_equal"])
            if component == "optimizer":
                self.assertFalse(row["moments_direct_typed_equal"])
            self.assertFalse(row["core_direct_typed_equal"])

    def test_source_completion_streams_exact_updates_and_retains_one_full_core(self):
        value = self.source_completion(iter(self.on), "capture_on")
        self.assertEqual(len(value["trace"]), 8)
        self.assertEqual(value["final_state_core"]["state_completed_updates"], 8)
        self.assertEqual(value["scientific_execution_certified"], False)
        self.assertEqual(value["evidence_scope"]["external_receipt_bytes_verified"], False)
        self.assertEqual(value["instrumentation"]["baseline_kind"],
                         "instrumented_state_hash_baseline_not_uninstrumented")

        def count_cores(item):
            if type(item) is dict:
                return (item.get("schema_name") == "i7_state_core") + sum(count_cores(v) for v in item.values())
            if type(item) in (list, tuple):
                return sum(count_cores(v) for v in item)
            return 0
        self.assertEqual(count_cores(value), 1)

    def test_capture_off_has_common_hash_instrumentation_but_no_sealed_anchors(self):
        value = self.source_completion(self.off, "capture_off")
        self.assertEqual(value["anchor_witness_refs"], [])
        self.assertEqual(value["instrumentation"]["both_modes_common"],
                         "state_core_hash_after_every_completed_update")

    def test_completion_rejects_missing_reordered_counter_profile_and_refs(self):
        common = dict(identity=identity(), profile=state.MLP_FIXTURE_PROFILE,
                      capture_mode="capture_on", plan_ref=plan_ref(),
                      anchor_refs=[artifact_ref("anchor")],
                      witness_refs=[artifact_ref("source-witness")],
                      sources_sha256="5" * 64, environment_sha256="6" * 64)
        with self.assertRaises(history.SourceHistoryError):
            history.make_source_completion(self.on[:-1], **common)
        reordered = list(self.on)
        reordered[1] = reordered[0]
        with self.assertRaises(history.SourceHistoryError):
            history.make_source_completion(reordered, **common)
        bad_counter = state.clone_tree(self.on[0])
        bad_counter["state_completed_updates"] = 2
        with self.assertRaises(history.SourceHistoryError):
            history.make_source_completion([bad_counter, *self.on[1:]], **common)
        bad_profile = state.clone_tree(self.on[0])
        bad_profile["profile"] = state.FIXTURE_PROFILE
        with self.assertRaises(history.SourceHistoryError):
            history.make_source_completion([bad_profile, *self.on[1:]], **common)
        bad_plan = plan_ref()
        bad_plan["name"] = "wrong.pt"
        with self.assertRaises(history.SourceHistoryError):
            history.make_source_completion(self.on, **{**common, "plan_ref": bad_plan})
        with self.assertRaises(history.SourceHistoryError):
            history.make_source_completion(self.on, **{**common, "anchor_refs": []})
        with self.assertRaises(history.SourceHistoryError):
            history.make_source_completion(
                self.on, **{**common,
                            "anchor_refs": [artifact_ref("anchor"), artifact_ref("anchor")],
                            "witness_refs": [artifact_ref("source-witness"),
                                             artifact_ref("source-witness")]})

    def test_capture_comparison_is_bounded_and_has_no_full_core_copies(self):
        value = self.capture_comparison()
        self.assertEqual(len(value["step_trace"]), 8)
        self.assertEqual(value["summary"]["core_equal_steps"], 8)
        self.assertTrue(value["initial_comparison"]["state_direct_typed_equal"])
        self.assertNotIn("final_state_core", value)
        codec.json_loads(codec.json_bytes(value), max_bytes=1_000_000)

        forged = copy.deepcopy(value)
        forged["step_trace"][0]["core_direct_typed_equal"] = False
        with self.assertRaises(history.SourceHistoryError):
            history.validate_capture_comparison(forged)
        missing = copy.deepcopy(value)
        missing["step_trace"].pop()
        with self.assertRaises(history.SourceHistoryError):
            history.validate_capture_comparison(missing)
        counter = copy.deepcopy(value)
        counter["step_trace"][1]["completed_updates"] = 1
        with self.assertRaises(history.SourceHistoryError):
            history.validate_capture_comparison(counter)
        unequal_initial = state.clone_tree(self.off_initial)
        unequal_initial["model"]["parameters"][0]["value"].view(-1)[0] += .1
        with self.assertRaises(history.SourceHistoryError):
            history.make_capture_comparison(
                zip(self.on, self.off),
                initial_states=(self.on_initial, unequal_initial),
                identity=identity(), profile=state.MLP_FIXTURE_PROFILE)

    def test_comparison_detects_each_component_across_trajectory(self):
        off = [state.clone_tree(core) for core in self.off]
        off[1]["optimizer"]["state"][0]["exp_avg"]["value"].view(-1)[0] += .1
        off[2]["observer"]["state"]["grad_mean"]["value"].view(-1)[0] += .1
        off[3]["rng"]["torch_cpu"][0] ^= 1
        value = history.make_capture_comparison(
            zip(self.on, off), initial_states=(self.on_initial, self.off_initial),
            identity=identity(), profile=state.MLP_FIXTURE_PROFILE)
        self.assertEqual(value["summary"]["optimizer_equal_steps"], 7)
        self.assertEqual(value["summary"]["moments_equal_steps"], 7)
        self.assertEqual(value["summary"]["observer_equal_steps"], 7)
        self.assertEqual(value["summary"]["rng_equal_steps"], 7)
        self.assertEqual(value["summary"]["core_equal_steps"], 5)

    def test_canonical_fixed_dicts_and_receipt_bounds_reject_coercions(self):
        source = self.source_completion(self.on, "capture_on")
        false_as_zero = copy.deepcopy(source)
        false_as_zero["evidence_scope"]["external_receipt_bytes_verified"] = 0
        with self.assertRaises(history.SourceHistoryError):
            history.validate_source_completion(false_as_zero)
        subclass = copy.deepcopy(source)
        subclass["artifact_name"] = StringSubclass(subclass["artifact_name"])
        with self.assertRaises(history.SourceHistoryError):
            history.validate_source_completion(subclass)
        instrumentation = copy.deepcopy(source)
        instrumentation["instrumentation"]["baseline_kind"] = StringSubclass(
            instrumentation["instrumentation"]["baseline_kind"])
        with self.assertRaises(history.SourceHistoryError):
            history.validate_source_completion(instrumentation)

        oversized = receipt_fields("frozen-plan.pt", digest="2" * 64,
                                   size=storage.DEFAULT_BUDGET + 1)
        with self.assertRaises(history.SourceHistoryError):
            self.source_completion(self.on, "capture_on", plan=oversized)
        bad_ref = artifact_ref("anchor")
        bad_ref["status"] = StringSubclass("complete")
        with self.assertRaises(history.SourceHistoryError):
            history.make_source_completion(
                self.on, identity=identity(), profile=state.MLP_FIXTURE_PROFILE,
                capture_mode="capture_on", plan_ref=plan_ref(), anchor_refs=[bad_ref],
                witness_refs=[artifact_ref("source-witness")],
                sources_sha256="5" * 64, environment_sha256="6" * 64)

    def test_capture_bundle_links_every_record_and_recomputes_final_pair(self):
        on = self.source_completion(self.on, "capture_on")
        off = self.source_completion(self.off, "capture_off")
        comparison = self.capture_comparison()
        linked = history.validate_capture_bundle(on, off, comparison)
        self.assertIs(linked[0], on)
        self.assertIs(linked[1], off)
        self.assertIs(linked[2], comparison)

        trace_mismatch = copy.deepcopy(off)
        trace_mismatch["trace"][0]["model_sha256"] = "a" * 64
        history.validate_source_completion(trace_mismatch)
        with self.assertRaises(history.SourceHistoryError):
            history.validate_capture_bundle(on, trace_mismatch, comparison)

        alternate_plan = receipt_fields("frozen-plan.pt", digest="9" * 64, size=456)
        off_plan_mismatch = self.source_completion(self.off, "capture_off",
                                                   plan=alternate_plan)
        with self.assertRaises(history.SourceHistoryError):
            history.validate_capture_bundle(on, off_plan_mismatch, comparison)

        off_provenance_mismatch = self.source_completion(
            self.off, "capture_off", sources="7" * 64)
        with self.assertRaises(history.SourceHistoryError):
            history.validate_capture_bundle(on, off_provenance_mismatch, comparison)

        swapped = history.validate_capture_bundle
        with self.assertRaises(history.SourceHistoryError):
            swapped(off, on, comparison)

    def test_core_bounds_precede_validators_and_hash_copies(self):
        with mock.patch.object(history, "CORE_TENSOR_BYTE_CAP", 1), \
                mock.patch.object(state, "validate_core") as validator:
            with self.assertRaises(history.SourceHistoryError):
                history.step_fingerprint(self.on[0])
            validator.assert_not_called()
        too_long = state.clone_tree(self.on[0])
        too_long["schema_name"] = "x" * (history.CORE_STRING_CAP + 1)
        with mock.patch.object(state, "validate_core") as validator:
            with self.assertRaises(history.SourceHistoryError):
                history.step_fingerprint(too_long)
            validator.assert_not_called()
        cyclic = []
        cyclic.append(cyclic)
        with self.assertRaisesRegex(history.SourceHistoryError, "node/depth bound"):
            history._bound_full_state(cyclic)
        with self.assertRaisesRegex(history.SourceHistoryError, "pending node bound"):
            history._bound_full_state([[None] * history.CORE_NODE_CAP])
        with self.assertRaisesRegex(history.SourceHistoryError, "exact contiguous CPU storage"):
            history._bound_full_state(torch.zeros(100)[:1])

    def test_record_constructor_is_not_a_callback_or_iterator_sandbox(self):
        effects = []
        def supplied_stream():
            effects.append("caller iterator executed")
            yield from self.on
        value = self.source_completion(supplied_stream(), "capture_on")
        self.assertEqual(effects, ["caller iterator executed"])
        self.assertIs(value["scientific_execution_certified"], False)

        # Identical tensor snapshots do not establish that live hooks matched.
        torch.manual_seed(987)
        model = model_factory()
        optimizer = optimizer_factory(model)
        observer = observer_factory(model, optimizer)
        before = history.initial_fingerprint(model, optimizer, observer,
                                             profile=history.MLP_FIXTURE)
        handle = model.register_forward_hook(lambda module, args, output: output + 1.)
        try:
            after = history.initial_fingerprint(model, optimizer, observer,
                                                profile=history.MLP_FIXTURE)
            self.assertTrue(history.compare_initial(before, after)["state_direct_typed_equal"])
        finally:
            handle.remove()


if __name__ == "__main__":
    unittest.main()
