import copy
import hashlib
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments import grokking_confirmation as gc


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 2)

    def forward(self, inputs):
        return self.linear(inputs)


def tiny_optimizer_and_filter(model, arm):
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-2, weight_decay=0.1, betas=(0.9, 0.98)
    )
    tracker = None
    if arm != "adamw":
        tracker = gc.SpectralGradientFilter(
            model, optimizer, rank=2, decay=0.9, warmup=0,
            stable_update=arm == "stable", stabilize_every=3,
            relative_eig_tol=1e-12,
        )
    return optimizer, tracker


def direct_step(model, optimizer, tracker, inputs, targets):
    optimizer.zero_grad(set_to_none=True)
    loss = F.cross_entropy(model(inputs), targets)
    loss.backward()
    if tracker is not None:
        tracker.filter_grad()
    optimizer.step()
    return float(loss.detach())


def assert_tree_equal(testcase, left, right):
    testcase.assertEqual(type(left), type(right))
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        testcase.assertEqual(left.keys(), right.keys())
        for key in left:
            assert_tree_equal(testcase, left[key], right[key])
    elif isinstance(left, (list, tuple)):
        testcase.assertEqual(len(left), len(right))
        for lvalue, rvalue in zip(left, right):
            assert_tree_equal(testcase, lvalue, rvalue)
    else:
        testcase.assertEqual(left, right)


class GrokkingConfirmationTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(13)
        self.inputs = torch.tensor([
            [0.2, -0.1, 0.7], [-0.3, 0.5, 0.1],
            [0.8, 0.4, -0.2], [0.0, -0.6, 0.9],
        ])
        self.targets = torch.tensor([0, 1, 1, 0])

    def test_stable_and_legacy_arms_are_pinned(self):
        self.assertEqual(gc.SEEDS, (100, 101, 102, 103, 104))
        self.assertEqual(
            gc.CHECKPOINT_STEPS,
            (0, 100, 500, 1000, 1500, 2000, 2500, 3000, 4000, 6000),
        )
        for arm, expected in (("legacy", False), ("stable", True)):
            model = TinyModel()
            _, tracker = gc.make_optimizer(model, arm)
            self.assertEqual(tracker.rank, 200)
            self.assertEqual(tracker.decay, 0.99)
            self.assertEqual(tracker.warmup, 100)
            self.assertIs(tracker.stable_update, expected)
        model = TinyModel()
        _, tracker = gc.make_optimizer(model, "adamw")
        self.assertIsNone(tracker)

    def test_filter_state_roundtrip_and_next_step_equivalence(self):
        for arm in ("adamw", "legacy", "stable"):
            with self.subTest(arm=arm), tempfile.TemporaryDirectory() as temporary:
                torch.manual_seed(21)
                model = TinyModel()
                optimizer, tracker = tiny_optimizer_and_filter(model, arm)
                for _ in range(5):
                    direct_step(model, optimizer, tracker, self.inputs, self.targets)

                config = {"fixture": "tiny", "arm": arm}
                sources = {"fixture_source": "v1"}
                split = {"sha256": "a" * 64}
                parameters = gc.parameter_identity(model)
                saved = gc.checkpoint_state(
                    model, optimizer, tracker, step=5, rows=[{
                        "step": 0, "train": {"accuracy": 0.5},
                        "test": {"accuracy": 0.5},
                    }], timing={"training_seconds": 0.1}, config=config,
                    sources=sources, split=split, parameters=parameters,
                    device="cpu",
                )
                path = Path(temporary) / "state.pt"
                receipt = gc.save_checkpoint(path, saved)
                loaded = gc.load_checkpoint(path, receipt["sha256"])
                loaded_before_restore = copy.deepcopy(loaded)

                expected_loss = direct_step(
                    model, optimizer, tracker, self.inputs, self.targets
                )
                expected_model = copy.deepcopy(model.state_dict())
                expected_optimizer = copy.deepcopy(optimizer.state_dict())
                expected_filter = gc.filter_state(tracker)

                restored = TinyModel()
                restored_optimizer, restored_tracker = tiny_optimizer_and_filter(
                    restored, arm
                )
                gc.restore_state(
                    loaded, restored, restored_optimizer, restored_tracker,
                    config=config, sources=sources, split=split,
                    parameters=gc.parameter_identity(restored), device="cpu",
                )
                actual_loss = direct_step(
                    restored, restored_optimizer, restored_tracker,
                    self.inputs, self.targets,
                )

                self.assertEqual(actual_loss, expected_loss)
                assert_tree_equal(self, restored.state_dict(), expected_model)
                assert_tree_equal(
                    self, restored_optimizer.state_dict(), expected_optimizer
                )
                assert_tree_equal(
                    self, gc.filter_state(restored_tracker), expected_filter
                )
                assert_tree_equal(self, loaded, loaded_before_restore)

    def test_safe_checkpoint_serialization_and_hash_binding(self):
        torch.manual_seed(31)
        model = TinyModel()
        optimizer, tracker = tiny_optimizer_and_filter(model, "stable")
        direct_step(model, optimizer, tracker, self.inputs, self.targets)
        config = {"fixture": "tiny"}
        sources = {"fixture_source": "v1"}
        split = {"sha256": "a" * 64}
        parameters = gc.parameter_identity(model)
        rows = [{
            "step": 0, "phase": "initial",
            "train": {"loss": 1.0, "accuracy": 0.5, "count": 4},
            "test": {"loss": 1.0, "accuracy": 0.5, "count": 4},
        }]
        payload = gc.checkpoint_state(
            model, optimizer, tracker, step=1, rows=rows,
            timing={"training_seconds": 0.1, "evaluation_seconds": 0.2},
            config=config, sources=sources, split=split,
            parameters=parameters, device="cpu",
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "checkpoint.pt"
            receipt = gc.save_checkpoint(path, payload)
            loaded = gc.load_checkpoint(path, receipt["sha256"])
            self.assertEqual(loaded["schema"], gc.CHECKPOINT_SCHEMA)
            assert_tree_equal(self, loaded["filter_state"], payload["filter_state"])
            with self.assertRaisesRegex(gc.ConfirmationError, "hash mismatch"):
                gc.load_checkpoint(path, "0" * 64)
            with self.assertRaises(FileExistsError):
                gc.save_checkpoint(path, payload)

    def test_split_identity_is_order_sensitive(self):
        tensors = (
            torch.tensor([[0, 1], [1, 0]]), torch.tensor([1, 0]),
            torch.tensor([[2, 1]]), torch.tensor([0]),
        )
        first = gc.tensor_set_identity(tensors)
        changed = gc.tensor_set_identity((tensors[0].flip(0), *tensors[1:]))
        self.assertNotEqual(first["sha256"], changed["sha256"])
        self.assertEqual(first["train_count"], 2)
        self.assertEqual(first["test_count"], 1)

    def test_threshold_is_explicitly_interval_censored(self):
        def row(step, accuracy):
            return {"step": step, "test": {"accuracy": accuracy}}

        observed = gc.threshold_summary([
            row(0, 0.01), row(50, 0.80), row(100, 0.91)
        ])
        self.assertEqual(observed["status"], "interval_censored")
        self.assertEqual(observed["lower_step_exclusive"], 50)
        self.assertEqual(observed["upper_step_inclusive"], 100)
        self.assertEqual(observed["sustained_attainment_step"], 100)
        censored = gc.threshold_summary([row(0, 0.01), row(50, 0.8)])
        self.assertEqual(censored["status"], "right_censored")
        self.assertEqual(censored["lower_step_exclusive"], 50)
        self.assertIsNone(censored["upper_step_inclusive"])

    def test_exclusive_json_never_replaces(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "record.json"
            gc.write_json(path, {"value": 1})
            before = hashlib.sha256(path.read_bytes()).hexdigest()
            with self.assertRaises(FileExistsError):
                gc.write_json(path, {"value": 2})
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), before)


if __name__ == "__main__":
    unittest.main()
