"""Small synthetic CPU checks for the I9 neural core; no dataset or run."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import random
import sys
import unittest

import numpy as np
import torch
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import neural_core as core


def fixture():
    model = core.make_model(17, "cpu", input_dim=4, width=5, classes=3)
    optimizer = core.make_optimizer(model)
    tracker = core.make_tracker(model, optimizer)
    x = torch.arange(48 * 4, dtype=torch.float32).reshape(48, 4).remainder(17).div(17)
    clean = torch.arange(48, dtype=torch.long).remainder(3)
    noisy = (clean + (torch.arange(48) % 2)).remainder(3)
    ax = x.flip(0).clone()
    ay = clean.flip(0).clone()
    data = {"x": x, "clean": clean, "noisy": noisy,
            "vx": x.clone(), "vy": clean.clone(), "ax": ax, "ay": ay}
    pairs = torch.arange(32 * 2 * 64, dtype=torch.long).reshape(32, 2, 64).remainder(48)
    plans = {"pairs": pairs, "update": torch.arange(64).remainder(48),
             "loss_train": torch.arange(1024).remainder(48),
             "loss_aux": torch.arange(17, 1041).remainder(48),
             "utility_aux": torch.arange(31, 1055).remainder(48)}
    return model, optimizer, tracker, data, plans


class NeuralCoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        source = ROOT / "spectral_filter.py"
        if hashlib.sha256(source.read_bytes()).hexdigest() != core.FILTER_SHA256:
            raise AssertionError("canonical spectral_filter.py hash differs")

    def test_model_initialization_is_deterministic_cpu_first_and_rng_neutral(self):
        random.seed(9)
        np.random.seed(10)
        torch.manual_seed(11)
        before = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone())
        left = core.make_model(123, "cpu", input_dim=4, width=5, classes=3)
        right = core.make_model(123, "cpu", input_dim=4, width=5, classes=3)
        self.assertTrue(torch.equal(core.flat_params(left), core.flat_params(right)))
        self.assertEqual(before[0], random.getstate())
        self.assertTrue(np.array_equal(before[1][1], np.random.get_state()[1]))
        self.assertTrue(torch.equal(before[2], torch.get_rng_state()))

    def test_snapshot_weights_only_roundtrip_and_same_next_step(self):
        model, optimizer, tracker, data, _ = fixture()
        core.train_step(model, optimizer, tracker, data["x"][:8], data["noisy"][:8], "raw")
        frozen = core.snapshot(model, optimizer, tracker)
        buffer = io.BytesIO()
        torch.save(frozen, buffer)
        buffer.seek(0)
        loaded = torch.load(buffer, map_location="cpu", weights_only=True)
        self.assertTrue(core.equal_tree(frozen, loaded))
        self.assertEqual(len(core.tree_digest(loaded)), 64)

        core.train_step(model, optimizer, tracker, data["x"][8:16], data["noisy"][8:16], "raw")
        expected = core.snapshot(model, optimizer, tracker)
        clone, clone_optimizer, clone_tracker = core.restore(loaded, "cpu")
        core.train_step(clone, clone_optimizer, clone_tracker,
                        data["x"][8:16], data["noisy"][8:16], "raw")
        self.assertTrue(core.equal_tree(expected, core.snapshot(clone, clone_optimizer, clone_tracker)))

    def test_soft_q_gradient_is_exact_convex_objective_gradient(self):
        model, _, _, data, _ = fixture()
        x, labels = data["x"][:9], data["clean"][:9]
        hard = core.gradient(model, x, labels)
        uniform = core.gradient(model, x, torch.full((9, 3), 1 / 3, dtype=torch.float64))
        q = 0.1 * F.one_hot(labels, num_classes=3).to(torch.float64) + 0.9 / 3
        soft = core.gradient(model, x, q)
        self.assertTrue(torch.allclose(soft, 0.1 * hard + 0.9 * uniform,
                                       atol=2e-7, rtol=2e-6))

    def test_raw_source_observes_native_then_restores_exact_training_gradient(self):
        model, optimizer, tracker, data, _ = fixture()
        n = core.flat_params(model).numel()
        tracker.grad_mean = torch.linspace(-0.2, 0.3, n)
        tracker.V = torch.eye(n)[:, :2].contiguous()
        tracker.S = torch.ones(2, dtype=torch.float64)
        tracker.step_count = 100
        frozen = core.snapshot(model, optimizer, tracker)
        plain, plain_optimizer, _ = core.restore(frozen, "cpu")
        plain_optimizer.zero_grad(set_to_none=True)
        loss = core._objective_loss(plain, data["x"][:8], data["noisy"][:8])
        loss.backward()
        plain_optimizer.step()

        actual, actual_optimizer, actual_tracker = core.restore(frozen, "cpu")
        row = core.train_step(actual, actual_optimizer, actual_tracker,
                              data["x"][:8], data["noisy"][:8], "raw")
        self.assertEqual(row["observer_step"], 101)
        self.assertTrue(row["filtering_active"])
        plain_state, actual_state = plain.state_dict(), actual.state_dict()
        self.assertEqual(tuple(plain_state), tuple(actual_state))
        self.assertTrue(all(torch.equal(plain_state[name], actual_state[name])
                            for name in plain_state))
        self.assertTrue(core.equal_tree(core._cpu_clone(plain_optimizer.state_dict()),
                                        core._cpu_clone(actual_optimizer.state_dict())))
        self.assertEqual(actual_tracker.step_count, 101)

    def test_rescale_is_reciprocal_and_zero_or_extreme_is_explicitly_undefined(self):
        vector = torch.tensor([3.0, 4.0])
        scaled, record = core.rescale_to_norm(vector, 10.0)
        self.assertIsNone(record["reason"])
        self.assertAlmostEqual(core._norm(scaled), 10.0, places=6)
        restored, reverse = core.rescale_to_norm(scaled, 5.0)
        self.assertIsNone(reverse["reason"])
        self.assertTrue(torch.allclose(restored, vector))
        for value, target, reason in ((torch.zeros(2), 0.0, "zero_source_norm"),
                                      (torch.tensor([1e-20]), 1.0, "scale_limit")):
            result, row = core.rescale_to_norm(value, target)
            self.assertIsNone(result)
            self.assertEqual(row["reason"], reason)
            self.assertIsNone(row["achieved_norm"])

    def test_pair_innovation_identity_before_and_after_projection(self):
        first = torch.tensor([1.0, -2.0, 0.5])
        second = torch.tensor([-0.5, 3.0, 1.0])
        mean = torch.tensor([0.2, -0.3, 0.1])
        basis = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
        row = core._pair_geometry(first, second, mean, basis)
        self.assertAlmostEqual(row["innovation"], row["fresh"] + row["surprise"], places=12)
        projected = row["after_projection"]
        self.assertAlmostEqual(projected["innovation"],
                               projected["fresh"] + projected["surprise"], places=12)
        negative = core._pair_geometry(torch.tensor([1.0]), torch.tensor([-1.0]),
                                       torch.tensor([0.0]), None)
        self.assertLess(negative["surprise"], 0)

    def test_probe_is_neutral_complete_deterministic_and_auditable(self):
        model, optimizer, tracker, data, plans = fixture()
        n = core.flat_params(model).numel()
        tracker.grad_mean = torch.zeros(n)
        tracker.V = torch.eye(n)[:, :2].contiguous()
        tracker.S = torch.ones(2, dtype=torch.float64)
        tracker.step_count = 100
        before = core.snapshot(model, optimizer, tracker)
        record, payload = core.probe_anchor(model, optimizer, tracker, data, plans)
        self.assertEqual(record["status"], "complete")
        self.assertEqual(len(record["pair_rows"]), 32)
        self.assertEqual(len(record["utility_order"]), 7)
        self.assertNotIn("old_V", payload)
        self.assertEqual(record["tensor_payload_sha256"], core.tree_digest(payload))
        self.assertTrue(core.equal_tree(before, core.snapshot(model, optimizer, tracker)))
        json.dumps(record, allow_nan=False)

        summaries = {row["name"]: row for row in record["interventions"]}
        raw_norm = summaries["raw_gradient"]["data_norm"]
        native_norm = summaries["native_gradient"]["data_norm"]
        self.assertAlmostEqual(summaries["raw_data_delta_to_native_norm"]["data_norm"],
                               native_norm, delta=max(1e-10, native_norm * 5e-5))
        self.assertAlmostEqual(summaries["native_data_delta_to_raw_norm"]["data_norm"],
                               raw_norm, delta=max(1e-10, raw_norm * 5e-5))


if __name__ == "__main__":
    unittest.main()
