#!/usr/bin/env python3
"""Dataset-free CPU checks for one-pass source-history assembly."""
import copy
import hashlib
import os
from pathlib import Path
import random
import sys
import unittest

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
import torch
import torch.nn.functional as F


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))

import artifact_store as storage
import identity_codec as codec
import source_history as history
import source_streaming as streaming
import state_core as state
from spectral_filter import SpectralGradientFilter


X = torch.tensor([[.2, -.4, .7], [1., .5, -.25],
                  [-.3, .9, .1], [.6, -.2, .4]])
Y = torch.tensor([0, 1, 1, 0])


def identity(update=5):
    return {
        "run_id": "2026-09-06-spectral-optimizer-investigation",
        "iteration": 7, "execution_role": "primary",
        "evidence_role": "primary", "bundle": 0,
        "anchor_update": update, "source_policy": "fixture_current2",
        "condition": "fixture_only", "steps_total": 8,
        "anchor_phase": "pre_forward_pre_observe",
        "anchor_completed_updates": update - 1,
        "anchor_completed_observations": update - 1,
        "rng_namespace_prefix": [20260906, 0],
        "stream_roles": dict(codec.STREAM_ROLES),
    }


def model_factory():
    return torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.ReLU(),
                               torch.nn.Linear(4, 2))


def live_objects(seed=7317):
    random.seed(seed)
    np.random.seed(seed + 1)
    torch.manual_seed(seed + 2)
    model = model_factory()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=.001, weight_decay=.01,
        betas=(.9, .999), eps=1e-8, foreach=False, fused=False)
    observer = SpectralGradientFilter(
        model, optimizer, rank=2, decay=.99, warmup=2, stable_update=True)
    initial = history.initial_fingerprint(
        model, optimizer, observer, profile=state.MLP_FIXTURE_PROFILE)
    return model, optimizer, observer, initial


def live_step(objects, completed):
    model, optimizer, observer, _ = objects
    optimizer.zero_grad(set_to_none=True)
    F.cross_entropy(model(X), Y).backward()
    observer.filter_grad()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return state.capture_core(
        model, optimizer, observer, profile=state.MLP_FIXTURE_PROFILE,
        completed_updates=completed)


def receipt_fields(name, digest, size=123):
    base = {
        "schema": "i7_artifact_receipt_v1", "name": name, "size": size,
        "sha256": digest, "status": "complete",
        "encoding": "torch_weights_only",
    }
    encoded = storage._json_bytes(base)
    return {
        "name": name, "status": "complete", "encoding": "torch_weights_only",
        "size_bytes": size, "sha256": digest,
        "receipt_name": "receipt-" + hashlib.sha256(name.encode("ascii")).hexdigest() + ".json",
        "receipt_size_bytes": len(encoded),
        "receipt_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def plan_ref():
    return receipt_fields("frozen-plan.pt", "2" * 64, size=456)


def artifact_ref(kind):
    ident = identity()
    artifact_id = codec.artifact_id(
        ident, profile=state.MLP_FIXTURE_PROFILE, kind=kind)
    value = {
        "artifact_id": artifact_id,
        "schema_name": ("i7_anchor" if kind == "anchor"
                        else "i7_source_step_witness"),
    }
    value.update(receipt_fields(
        artifact_id + ".pt", ("3" if kind == "anchor" else "4") * 64))
    return {key: value[key] for key in history.ARTIFACT_REF_KEYS}


def stream_context():
    return dict(
        identity=identity(), profile=state.MLP_FIXTURE_PROFILE,
        plan_ref=plan_ref(), sources_sha256="5" * 64,
        environment_sha256="6" * 64)


def count_cores(item):
    if type(item) is dict:
        return ((item.get("schema_name") == "i7_state_core") +
                sum(count_cores(value) for value in item.values()))
    if type(item) in (list, tuple):
        return sum(count_cores(value) for value in item)
    return 0


class SourceStreamingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
            raise RuntimeError("streaming tests require hidden CUDA")
        if torch.cuda.is_initialized():
            raise RuntimeError("streaming tests must not initialize CUDA")

    def test_single_actual_stream_reduces_each_update_and_retains_only_final(self):
        stream = streaming.SourceCompletionStream(
            capture_mode="capture_on", **stream_context())
        objects = live_objects()
        for update in range(1, 9):
            core = live_step(objects, update)
            kwargs = ({"anchor_ref": artifact_ref("anchor"),
                       "witness_ref": artifact_ref("source-witness")}
                      if update == 5 else {})
            row = stream.record(core, **kwargs)
            self.assertEqual(row["completed_updates"], update)
            self.assertEqual(stream.retained_full_core_count,
                             1 if update == 8 else 0)
            if update < 8:
                core["model"]["parameters"][0]["value"].zero_()
        value = stream.finish()
        self.assertEqual(stream.retained_full_core_count, 0)
        self.assertEqual(count_cores(value), 1)
        self.assertEqual(len(value["trace"]), 8)
        self.assertEqual(value["anchor_witness_refs"][0]["anchor_update"], 5)
        history.validate_source_completion(value)
        with self.assertRaisesRegex(streaming.SourceStreamingError, "terminal"):
            stream.finish()

    def test_one_pass_pilot_pair_assembles_three_linked_artifacts(self):
        on = live_objects()
        off = live_objects()
        stream = streaming.PilotPairStream(
            on_initial=on[3], off_initial=off[3], **stream_context())
        for update in range(1, 9):
            on_core = live_step(on, update)
            off_core = live_step(off, update)
            kwargs = ({"anchor_ref": artifact_ref("anchor"),
                       "witness_ref": artifact_ref("source-witness")}
                      if update == 5 else {})
            row = stream.record(on_core, off_core, **kwargs)
            self.assertTrue(row["core_direct_typed_equal"])
            self.assertEqual(stream.retained_full_core_count,
                             2 if update == 8 else 0)
        value = stream.finish()
        self.assertEqual(tuple(value), streaming.PAIR_RESULT_KEYS)
        history.validate_capture_bundle(
            value["capture_on"], value["capture_off"], value["comparison"])
        self.assertEqual(count_cores(value["capture_on"]), 1)
        self.assertEqual(count_cores(value["capture_off"]), 1)
        self.assertEqual(count_cores(value["comparison"]), 0)
        self.assertEqual(value["comparison"]["summary"]["core_equal_steps"], 8)

    def test_valid_adverse_step_is_retained_instead_of_aborting(self):
        on = live_objects()
        off = live_objects()
        stream = streaming.PilotPairStream(
            on_initial=on[3], off_initial=off[3], **stream_context())
        for update in range(1, 9):
            on_core = live_step(on, update)
            off_core = live_step(off, update)
            if update == 3:
                off_core = state.clone_tree(off_core)
                off_core["model"]["parameters"][0]["value"].view(-1)[0] += .25
            kwargs = ({"anchor_ref": artifact_ref("anchor"),
                       "witness_ref": artifact_ref("source-witness")}
                      if update == 5 else {})
            stream.record(on_core, off_core, **kwargs)
        value = stream.finish()
        adverse = value["comparison"]["step_trace"][2]
        self.assertFalse(adverse["model_direct_typed_equal"])
        self.assertFalse(adverse["core_direct_typed_equal"])
        self.assertEqual(value["comparison"]["summary"]["core_equal_steps"], 7)

    def test_initial_and_update_failures_are_fail_closed(self):
        on = live_objects()
        off = live_objects()
        changed = copy.deepcopy(off[3])
        changed["model"]["parameters"][0]["value"].view(-1)[0] += .25
        with self.assertRaisesRegex(streaming.SourceStreamingError,
                                    "share the exact initial state"):
            streaming.PilotPairStream(
                on_initial=on[3], off_initial=changed, **stream_context())

        stream = streaming.SourceCompletionStream(
            capture_mode="capture_on", **stream_context())
        first = live_step(on, 1)
        stream.record(first)
        with self.assertRaisesRegex(streaming.SourceStreamingError,
                                    "missing, duplicated, or reordered"):
            stream.record(first)
        with self.assertRaisesRegex(streaming.SourceStreamingError, "terminal"):
            stream.record(live_step(on, 2))
        with self.assertRaisesRegex(streaming.SourceStreamingError, "terminal"):
            stream.finish()

    def test_missing_or_misplaced_refs_and_premature_finish_are_terminal(self):
        objects = live_objects()
        stream = streaming.SourceCompletionStream(
            capture_mode="capture_on", **stream_context())
        for update in range(1, 5):
            stream.record(live_step(objects, update))
        with self.assertRaisesRegex(streaming.SourceStreamingError,
                                    "requires its persisted reference pair"):
            stream.record(live_step(objects, 5))
        with self.assertRaisesRegex(streaming.SourceStreamingError, "terminal"):
            stream.finish()

        off = streaming.SourceCompletionStream(
            capture_mode="capture_off", **stream_context())
        with self.assertRaisesRegex(streaming.SourceStreamingError,
                                    "outside a capture-on anchor"):
            off.record(live_step(live_objects(), 1),
                       anchor_ref=artifact_ref("anchor"),
                       witness_ref=artifact_ref("source-witness"))

        premature = streaming.SourceCompletionStream(
            capture_mode="capture_on", **stream_context())
        with self.assertRaisesRegex(streaming.SourceStreamingError,
                                    "ended before steps_total"):
            premature.finish()
        with self.assertRaisesRegex(streaming.SourceStreamingError, "terminal"):
            premature.record(live_step(live_objects(), 1))

    def test_imported_module_has_no_execution_or_writer_entrypoint(self):
        self.assertFalse(hasattr(streaming, "main"))
        self.assertFalse(hasattr(streaming, "run"))
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == "__main__":
    unittest.main()
