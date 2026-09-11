"""Synthetic CPU actions/state/NPZ fixtures only; no real data or CUDA work."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments import measure_grokking_function_response as collector


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([1., -2., .5]))


def saved_before():
    model = Toy()
    optimizer = torch.optim.AdamW(model.parameters(), **collector.ADAMW, foreach=False)
    model.weight.grad = torch.tensor([.3, -.2, .1])
    optimizer.step()  # tiny setup state, not a saved-data or scientific replay
    return {**collector.parameter_state(model, optimizer),
            "torch_cpu_rng_state": torch.get_rng_state().clone(), "torch_cuda_rng_states": []}


def first_fixture():
    return {"raw_gradient": torch.tensor([3., 4., 0.]), "actions": {
        "Q": torch.tensor([[1.], [0.], [0.]], dtype=torch.float64),
        "actions": {collector.POLICY: torch.tensor([1.2, 1.6, 0.]),
                    "norm_matched": torch.tensor([2., 0., 0.])}}}


def npz_fixture(directory):
    p = 3
    pairs = np.column_stack((np.repeat(np.arange(p), p), np.tile(np.arange(p), p)))
    structure = {"pairs": pairs, "sums": pairs.sum(axis=1) % p,
                 "train_ids": np.arange(4), "test_ids": np.arange(4, 9),
                 "probe_fit_indices": np.array([0, 1]), "probe_eval_indices": np.array([2, 3, 4]),
                 "null_permutations": np.array([[4, 3, 2, 1, 0]])}
    contract = {"structural_array_sha256": {key: collector._array_hash(key, value)
                                           for key, value in structure.items()}}
    paths = []
    for name, offset in (("before", 0.), ("raw", .1)):
        path = directory / f"{name}.npz"
        arrays = {**structure, "logits": np.full((9, 3), offset, dtype=np.float32),
                  "final_hidden": np.zeros((9, 2), dtype=np.float32),
                  "pre_attention": np.zeros((9, 4), dtype=np.float32)}
        np.savez_compressed(path, **arrays)
        paths.append(path)
    return paths, contract, structure


class TestFunctionResponseCollector(unittest.TestCase):
    def test_saved_action_projection_norm_and_collinearity(self):
        first = first_fixture()
        original = collector.tree_hash(first)
        actions, report = collector.actions_from_saved(first, "cpu")
        self.assertEqual(set(actions), {"raw", "trunc", "projected", "zero"})
        self.assertTrue(torch.equal(actions["trunc"], torch.tensor([1.2, 0., 0.])))
        self.assertAlmostEqual(report["rho"], .6)
        self.assertEqual(report["norms"]["zero"], 0.)
        self.assertLessEqual(report["trunc_rho_projected_relative_error"], report["relative_tolerance"])
        self.assertEqual(collector.tree_hash(first), original)

    def test_degenerate_nonfinite_or_malformed_saved_actions_fail(self):
        for key in ("raw_zero", "gradient_zero", "nonfinite", "wrong_norm", "nonorthogonal", "wrong_dtype"):
            first = first_fixture()
            if key == "raw_zero": first["actions"]["actions"][collector.POLICY].zero_()
            if key == "gradient_zero": first["raw_gradient"].zero_()
            if key == "nonfinite": first["actions"]["Q"][0, 0] = float("nan")
            if key == "wrong_norm": first["actions"]["actions"]["norm_matched"] *= 2
            if key == "nonorthogonal": first["actions"]["Q"] *= 2
            if key == "wrong_dtype": first["actions"]["actions"][collector.POLICY] = first["actions"]["actions"][collector.POLICY].double()
            with self.subTest(case=key), self.assertRaises(ValueError):
                collector.actions_from_saved(first, "cpu")

    def test_each_new_action_restores_same_adam_and_takes_one_step(self):
        before = saved_before()
        before_hash = collector.tree_hash(before)
        actions, _ = collector.actions_from_saved(first_fixture(), "cpu")
        origins, endings = [], []
        for name in collector.NEW_ACTIONS:
            model, after, identity = collector.apply_once(before, actions[name], name, "cpu", Toy)
            self.assertEqual(float(after["optimizer_state"]["state"][0]["step"]), 2.)
            self.assertFalse(after["optimizer_state"]["param_groups"][0]["foreach"])
            self.assertEqual(after["optimizer_state"]["param_groups"], before["optimizer_state"]["param_groups"])
            self.assertTrue(torch.equal(model.weight.grad, actions[name]))
            self.assertEqual(collector.tree_hash(before), before_hash)
            self.assertTrue(torch.equal(torch.get_rng_state(), before["torch_cpu_rng_state"]))
            origins.append(identity["before_state_sha256"])
            endings.append(identity["after_state_sha256"])
        self.assertEqual(len(set(origins)), 1)
        self.assertEqual(len(set(endings)), 3)

    def test_explicit_zero_still_moves_parameters_and_updates_moments(self):
        before = saved_before()
        model, after, identity = collector.apply_once(before, torch.zeros(3), "zero", "cpu", Toy)
        self.assertIsNotNone(model.weight.grad)
        self.assertEqual(torch.count_nonzero(model.weight.grad), 0)
        self.assertFalse(torch.equal(after["model_state"]["weight"], before["model_state"]["weight"]))
        self.assertFalse(torch.equal(after["optimizer_state"]["state"][0]["exp_avg"],
                                     before["optimizer_state"]["state"][0]["exp_avg"]))
        for key, value in identity["adam_arithmetic"].items():
            if key != "relative_tolerance":
                self.assertLessEqual(value, identity["adam_arithmetic"]["relative_tolerance"])

    def test_raw_and_before_replay_forbidden_before_model_creation(self):
        for name in ("raw", "before", "native"):
            with self.assertRaisesRegex(ValueError, "replay is forbidden"):
                collector.apply_once({}, torch.zeros(3), name, "cpu", lambda: self.fail("model constructed"))

    def test_wrong_parameter_identity_or_action_size_fails(self):
        before = saved_before()
        malformed = copy.deepcopy(before)
        malformed["parameter_identity"][0]["name"] = "wrong"
        with self.assertRaisesRegex(ValueError, "restoration"):
            collector.apply_once(malformed, torch.zeros(3), "zero", "cpu", Toy)
        with self.assertRaisesRegex(ValueError, "parameter order"):
            collector.apply_once(before, torch.zeros(2), "zero", "cpu", Toy)

    def test_adam_arithmetic_rejects_mutated_moment(self):
        before = saved_before()
        _, after, _ = collector.apply_once(before, torch.zeros(3), "zero", "cpu", Toy)
        after["optimizer_state"]["state"][0]["exp_avg"][0] += .01
        with self.assertRaisesRegex(ValueError, "recurrence"):
            collector.adam_checks(before, after, torch.zeros(3))

    def test_archived_logits_are_loaded_with_identical_structural_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            paths, contract, expected = npz_fixture(Path(directory))
            logits, structure = collector.archived_logits(*paths, contract, p=3)
            self.assertEqual(set(logits), {"before", "raw"})
            self.assertTrue(np.array_equal(logits["before"], np.zeros((9, 3), dtype=np.float32)))
            for key in expected:
                self.assertTrue(np.array_equal(structure[key], expected[key]))
            contract["structural_array_sha256"]["pairs"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "structural hash"):
                collector.archived_logits(*paths, contract, p=3)

    def test_writer_exclusivity_and_small_synthetic_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parent = root / "spectral-grokking-function-response-20260909.fixture"
            parent.mkdir()
            with patch.object(collector, "STORAGE", root), patch.object(collector, "MAX_BYTES", 4096), \
                    patch.object(collector, "FREE_RESERVE", 1):
                output = collector.Output(parent / "diagnostic-001")
                receipt = output.write("fixture.json", {"fixture": True}, "json")
                self.assertEqual(collector.file_hash(Path(receipt["path"])), receipt["sha256"])
                with self.assertRaisesRegex(ValueError, "new"):
                    output.write("fixture.json", {}, "json")
                with self.assertRaisesRegex(RuntimeError, "budget"):
                    output.write("too-large.json", {"data": "x" * 5000}, "json")
                with self.assertRaisesRegex(ValueError, "new"):
                    collector.Output(parent / "diagnostic-001")


if __name__ == "__main__":
    unittest.main()
