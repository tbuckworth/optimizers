"""Synthetic CPU tests; no experiment outcome/data reads."""
import hashlib
import math
import os
from pathlib import Path
import struct
import tempfile
import unittest

import verify_checkpoints_cpu as v


class CheckpointVerifierTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
            raise RuntimeError("Tests require CUDA hidden")
        v.torch.set_num_threads(1)

    def test_manual_forward_and_ce(self):
        state = {key: v.torch.zeros(shape) for key, shape in v.SHAPES.items()}
        observed = v.evaluate(state, v.torch.ones(513, 784), v.torch.zeros(513, dtype=v.torch.long))
        self.assertEqual(observed["correct"], 513)
        self.assertAlmostEqual(observed["cross_entropy"], math.log(10), places=6)
        state["2.bias"][3] = 4
        observed = v.evaluate(state, v.torch.zeros(2, 784), v.torch.tensor([3, 2]))
        self.assertEqual(observed["accuracy"], .5)
        self.assertAlmostEqual(observed["cross_entropy"], math.log(math.exp(4) + 9) - 2, places=6)

    def test_state_hash_and_schema(self):
        state = {key: v.torch.zeros(shape) for key, shape in v.SHAPES.items()}
        expected = hashlib.sha256(b"dict")
        for key in sorted(state):
            expected.update(repr(key).encode() + b"\0tensor")
            expected.update(str(("torch.float32", v.SHAPES[key])).encode())
            expected.update(bytes(4 * state[key].numel()))
        self.assertEqual(v.state_hash(state), expected.hexdigest())
        state["2.bias"][0] = float("nan")
        with self.assertRaises(AssertionError):
            v.state_hash(state)

    def test_strict_ties_and_missing_grid(self):
        run = {"validation_trajectory": [{"step": step, "count": 5000, "accuracy": .5, "cross_entropy": 1.}
                                          for step in range(0, 2001, 100)],
               "checkpoint_steps": {"final": 2000, "warmup100": 100, "min_val_ce": 0, "max_val_accuracy": 0}}
        self.assertEqual(len(v.selectors(run)), 21)
        run["checkpoint_steps"]["max_val_accuracy"] = 100
        with self.assertRaises(AssertionError):
            v.selectors(run)

    def test_tolerances_are_not_exact_equality_or_unbounded(self):
        reference = {"accuracy": .5, "count": 10000, "cross_entropy": 1.}
        observed = {"accuracy": .5001, "correct": 5001, "count": 10000, "cross_entropy": 1.00001}
        self.assertTrue(v.compare(observed, reference)["passed_tolerance"])
        observed["correct"] = 5002
        self.assertFalse(v.compare(observed, reference)["passed_tolerance"])
        observed.update(correct=5000, cross_entropy=1.0001)
        self.assertFalse(v.compare(observed, reference)["passed_tolerance"])

    def test_plan_namespaces_and_disjointness(self):
        plan = v.independent_plan(60006, True)
        self.assertEqual(plan["initialization_seed"], 2693870106)
        indices = v.np.concatenate([plan[name] for name in ("train_indices", "validation_indices", "auxiliary_indices")])
        self.assertEqual(len(v.np.unique(indices)), 15000)
        self.assertFalse(v.np.array_equal(plan["train_indices"], v.independent_plan(60006)["train_indices"]))
        self.assertEqual(plan["training_batches"].shape, (2000, 64))

    def test_idx_bytes_and_strict_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture"
            path.write_bytes(struct.pack(">II", 2049, 2) + bytes([1, 9]))
            self.assertEqual(v.idx(path, False, 2).tolist(), [1, 9])
            with self.assertRaises(AssertionError):
                v.idx(path, False, 3)
            path.write_text('{"a": 1, "a": 2}')
            with self.assertRaises(AssertionError):
                v.read_json(path)
            path.write_text('{"a": NaN}')
            with self.assertRaises(ValueError):
                v.read_json(path)


if __name__ == "__main__":
    unittest.main()
