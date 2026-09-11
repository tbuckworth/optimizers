"""Bounded CPU tests for the I13 acquisition runner's pure helpers."""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import run_mean_branches as runner


def _probe_descriptor(horizon: int) -> dict:
    return {
        "horizon": horizon,
        "record": {
            "horizon": horizon,
            "tensor_artifact": {
                "name": f"alignment-h{horizon}.pt",
                "bytes": 1,
                "sha256": "a" * 64,
            },
        },
    }


class RunnerTests(unittest.TestCase):
    def test_expected_key_sets_are_exact(self):
        expected_new = {(seed, step, objective, policy)
                        for seed in (100, 101, 102)
                        for step in (1500, 2000)
                        for objective in ("fixed", "soft", "redraw")
                        for policy in ("mean32", "leak01_32")}
        expected_reference = {(seed, step, objective, policy)
                              for seed in (100, 101, 102)
                              for step in (1500, 2000)
                              for objective in ("fixed", "soft", "redraw")
                              for policy in ("raw", "current32")}
        self.assertEqual(runner.expected_keys(), expected_new)
        self.assertEqual(runner.expected_keys(runner.REFERENCES), expected_reference)
        self.assertEqual(len(expected_new), 36)
        self.assertEqual(len(expected_reference), 36)

    def test_all_six_probe_plans_are_exact_and_caller_neutral(self):
        for seed in runner.SEEDS:
            anchors = {}
            for step in runner.ANCHORS:
                base = np.arange(1024, dtype=np.int64).reshape(1024) % 5000
                anchors[str(step)] = {
                    "loss_train": base.copy(),
                    "loss_aux": np.roll(base, 1).copy(),
                    "utility_aux": np.roll(base, 2).copy(),
                }
            saved = {"seed": seed, "anchors": anchors}
            before = copy.deepcopy(saved)
            for step in runner.ANCHORS:
                plan = runner.probe_plan(saved, seed, step)
                self.assertEqual(tuple(plan), ("loss_train", "loss_aux", "utility_aux"))
                for value in plan.values():
                    self.assertEqual(value.shape, (1024,))
                    self.assertIn(value.dtype.kind, "iu")
                    self.assertTrue(((value >= 0) & (value < 5000)).all())
            for step in runner.ANCHORS:
                for name in ("loss_train", "loss_aux", "utility_aux"):
                    self.assertTrue(np.array_equal(saved["anchors"][str(step)][name],
                                                   before["anchors"][str(step)][name]))

        malformed = {"seed": 100, "anchors": {"1500": {
            "loss_train": np.zeros(1023, dtype=np.int64),
            "loss_aux": np.zeros(1024, dtype=np.int64),
            "utility_aux": np.zeros(1024, dtype=np.int64)}}}
        with self.assertRaises(AssertionError):
            runner.probe_plan(malformed, 100, 1500)
        malformed["anchors"]["1500"]["loss_train"] = np.full(1024, 5000,
                                                                 dtype=np.int64)
        with self.assertRaises(AssertionError):
            runner.probe_plan(malformed, 100, 1500)

    def test_save_probe_is_exclusive_and_binds_tensor_payload(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "artifacts"
            root.mkdir()
            run = runner.Run(root, 60)
            # This unit test exercises serialization only; production resource
            # checks are covered by the frozen parent runner.
            run.check = lambda: None
            tensors = {"gradient": torch.arange(7, dtype=torch.float32)}
            record = {"schema": "synthetic_alignment_v1", "status": "complete"}
            descriptor = runner.save_probe(run, "probe", {"horizon": 0}, record, tensors)
            artifact = descriptor["tensor_artifact"]
            target = root / artifact["name"]
            self.assertEqual(artifact["sha256"], hashlib.sha256(target.read_bytes()).hexdigest())
            self.assertEqual(descriptor["tensor_digest"], runner.i9.tree_digest(tensors))
            payload = torch.load(target, weights_only=True, map_location="cpu")
            self.assertEqual(payload["tensor_digest"], descriptor["tensor_digest"])
            self.assertTrue(runner.i9.equal_tree(payload["tensors"], tensors))
            self.assertEqual(payload["record"], record)
            self.assertEqual(payload["horizon"], 0)
            with self.assertRaises(FileExistsError):
                runner.save_probe(run, "probe", {"horizon": 0}, record, tensors)

    def test_validate_branch_probes_complete_partial_and_missing_first_step(self):
        complete = {
            "status": "complete", "completed_steps": 500,
            "probes": [_probe_descriptor(horizon) for horizon in runner.HORIZONS],
            "first_step_raw_gradient_digest": "1" * 64,
            "first_step_observer_digest": "2" * 64,
        }
        self.assertEqual(runner.validate_branch_probes(complete), ("1" * 64, "2" * 64))

        partial = {
            "status": "numerical_failure", "completed_steps": 73,
            "last_evaluated_horizon": 50,
            "numerical_failure": {"phase": "evaluation"},
            "probes": [_probe_descriptor(horizon) for horizon in (0, 1, 10, 50)],
            "first_step_raw_gradient_digest": "3" * 64,
            "first_step_observer_digest": "4" * 64,
        }
        self.assertEqual(runner.validate_branch_probes(partial), ("3" * 64, "4" * 64))

        probe_failure = copy.deepcopy(partial)
        probe_failure["numerical_failure"] = {"phase": "probe"}
        probe_failure["probes"] = [_probe_descriptor(horizon) for horizon in (0, 1, 10)]
        self.assertEqual(runner.validate_branch_probes(probe_failure),
                         ("3" * 64, "4" * 64))

        pre_step_failure = {
            "status": "numerical_failure", "completed_steps": 0,
            "last_evaluated_horizon": 0, "probes": [_probe_descriptor(0)],
            "numerical_failure": {"phase": "step"},
            "first_step_raw_gradient_digest": None,
            "first_step_observer_digest": None,
        }
        self.assertEqual(runner.validate_branch_probes(pre_step_failure), (None, None))

        broken = copy.deepcopy(partial)
        broken["probes"][-1]["record"]["horizon"] = 49
        with self.assertRaises(AssertionError):
            runner.validate_branch_probes(broken)


if __name__ == "__main__":
    unittest.main()
