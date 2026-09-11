"""Synthetic CPU checks for the deterministic I14 data plan."""
from __future__ import annotations

import copy
import json
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

import data_plan as plan


def clone_plan(value):
    return {
        key: (item.copy() if type(item) is np.ndarray else copy.deepcopy(item))
        for key, item in value.items()
    }


class DataPlanTests(unittest.TestCase):
    def test_constants_and_exact_global_phase_partition(self):
        self.assertEqual(plan.CALIBRATION_SEEDS, (190, 191))
        self.assertEqual(plan.CONFIRMATION_SEEDS, (200, 201, 202))
        self.assertEqual(plan.HORIZONS, (0, 100, 250, 500, 1000, 1500, 2000))
        self.assertEqual((plan.STEPS, plan.BATCH), (2000, 64))
        expected = np.random.default_rng(
            np.random.SeedSequence([20260907, 14, 0, 0])
        ).permutation(60_000)
        self.assertTrue(np.array_equal(plan._pool("calibration"), expected[:15_000]))
        self.assertTrue(np.array_equal(plan._pool("confirmation"), expected[15_000:]))
        self.assertEqual(np.intersect1d(
            plan._pool("calibration"), plan._pool("confirmation")
        ).size, 0)

    def test_exact_streams_determinism_and_registered_membership(self):
        for phase, seeds in (("calibration", plan.CALIBRATION_SEEDS),
                             ("confirmation", plan.CONFIRMATION_SEEDS)):
            pool = set(plan._pool(phase).tolist())
            serializations = []
            for seed in seeds:
                actual = plan.make_plan(seed, phase)
                plan.validate_plan(actual)
                self.assertEqual(tuple(actual), plan.PLAN_KEYS)
                expected_order = np.random.default_rng(
                    np.random.SeedSequence([20260907, 14, seed, 0])
                ).permutation(plan._pool(phase))
                self.assertTrue(np.array_equal(
                    actual["train_indices"], expected_order[:5000]))
                self.assertEqual(actual["initialization_seed"], int(
                    np.random.default_rng(np.random.SeedSequence(
                        [20260907, 14, seed, 1]
                    )).integers(0, 2**31, dtype=np.int64)))
                self.assertTrue(np.array_equal(actual["replacement_mask"],
                    np.random.default_rng(np.random.SeedSequence(
                        [20260907, 14, seed, 2]
                    )).random(5000) < .9))
                self.assertTrue(np.array_equal(actual["replacement_digits"],
                    np.random.default_rng(np.random.SeedSequence(
                        [20260907, 14, seed, 3]
                    )).integers(0, 10, 5000, dtype=np.int64)))
                self.assertTrue(np.array_equal(actual["training_batches"],
                    np.random.default_rng(np.random.SeedSequence(
                        [20260907, 14, seed, 4]
                    )).integers(0, 5000, (2000, 64), dtype=np.int64)))
                self.assertTrue(set(np.concatenate((
                    actual["train_indices"], actual["validation_indices"],
                    actual["auxiliary_indices"],
                )).tolist()).issubset(pool))
                serializations.append(json.dumps(
                    plan.json_tree(actual), separators=(",", ":")))
                self.assertEqual(serializations[-1], json.dumps(
                    plan.json_tree(plan.make_plan(seed, phase)),
                    separators=(",", ":")))
            self.assertEqual(len(set(serializations)), len(seeds))

    def test_all_five_plans_json_roundtrip_with_exact_dtypes(self):
        for phase, seeds in (("calibration", plan.CALIBRATION_SEEDS),
                             ("confirmation", plan.CONFIRMATION_SEEDS)):
            for seed in seeds:
                original = plan.make_plan(seed, phase)
                encoded = json.dumps(plan.json_tree(original),
                                     ensure_ascii=True, allow_nan=False,
                                     separators=(",", ":"))
                restored = plan.plan_from_json(json.loads(encoded))
                plan.validate_plan(restored)
                self.assertEqual(tuple(restored), plan.PLAN_KEYS)
                for name, dtype in plan.ARRAY_DTYPES.items():
                    self.assertEqual(str(restored[name].dtype), dtype)
                    self.assertTrue(np.array_equal(restored[name], original[name]))
                self.assertEqual(restored["auxiliary_indices"].shape, (5000,))

    def test_validate_rejects_structure_membership_bounds_and_mutation(self):
        original = plan.make_plan(190, "calibration")
        cases = []

        changed = clone_plan(original)
        changed["train_indices"][0] = changed["train_indices"][1]
        cases.append(changed)
        changed = clone_plan(original)
        changed["train_indices"][0] = plan._pool("confirmation")[0]
        cases.append(changed)
        changed = clone_plan(original)
        changed["training_batches"][0, 0] = 5000
        cases.append(changed)
        changed = clone_plan(original)
        changed["replacement_digits"][0] = 10
        cases.append(changed)
        changed = clone_plan(original)
        changed["replacement_mask"][0] = ~changed["replacement_mask"][0]
        cases.append(changed)
        changed = clone_plan(original)
        changed["train_indices"] = changed["train_indices"].astype(np.int32)
        cases.append(changed)
        changed = clone_plan(original)
        changed["initialization_seed"] += 1
        cases.append(changed)
        changed = dict(reversed(tuple(clone_plan(original).items())))
        cases.append(changed)

        for changed in cases:
            with self.subTest(case=len(cases)):
                with self.assertRaises(plan.DataPlanError):
                    plan.validate_plan(changed)
        with self.assertRaises(plan.DataPlanError):
            plan.make_plan(200, "calibration")
        with self.assertRaises(plan.DataPlanError):
            plan.make_plan(190, "confirmation")
        with self.assertRaises(plan.DataPlanError):
            plan.make_plan(True, "calibration")

    def test_plan_from_json_rejects_coercions_and_json_tree_nonfinite(self):
        payload = plan.json_tree(plan.make_plan(200, "confirmation"))
        bad = copy.deepcopy(payload)
        bad["seed"] = True
        with self.assertRaises(plan.DataPlanError):
            plan.plan_from_json(bad)
        bad = copy.deepcopy(payload)
        bad["training_batches"][0][0] = 1.0
        with self.assertRaises(plan.DataPlanError):
            plan.plan_from_json(bad)
        bad = copy.deepcopy(payload)
        bad["replacement_mask"][0] = 1
        with self.assertRaises(plan.DataPlanError):
            plan.plan_from_json(bad)
        bad = copy.deepcopy(payload)
        bad["training_batches"][0][0] = 2**100
        with self.assertRaises(plan.DataPlanError):
            plan.plan_from_json(bad)
        with self.assertRaises(plan.DataPlanError):
            plan.json_tree(float("nan"))
        with self.assertRaises(plan.DataPlanError):
            plan.json_tree(torch.tensor(1))

    def test_synthetic_data_materialization_is_exact_and_cpu_generic(self):
        p = plan.make_plan(200, "confirmation")
        x = torch.zeros((60_000, 784), dtype=torch.uint8)
        x[:, 0] = torch.arange(60_000, dtype=torch.int64).remainder(256).to(torch.uint8)
        y = torch.arange(60_000, dtype=torch.int64).remainder(10)
        data, stats = plan.data_for_plan(x, y, p, torch.device("cpu"))
        self.assertEqual(tuple(data), plan.DATA_KEYS)
        self.assertEqual(tuple(data["x"].shape), (5000, 784))
        self.assertEqual(data["x"].dtype, torch.float32)
        self.assertEqual(data["clean"].dtype, torch.int64)
        ti = torch.from_numpy(p["train_indices"])
        self.assertTrue(torch.equal(data["clean"], y[ti]))
        self.assertTrue(torch.equal(
            data["x"][:, 0], x[ti, 0].float().div(255)))
        expected_noisy = torch.where(
            torch.from_numpy(p["replacement_mask"]),
            torch.from_numpy(p["replacement_digits"]), y[ti])
        self.assertTrue(torch.equal(data["noisy"], expected_noisy))
        self.assertEqual(stats, {
            "replaced_count": int(p["replacement_mask"].sum()),
            "incorrect_count": int((expected_noisy != y[ti]).sum()),
            "train_count": 5000,
        })
        self.assertEqual(data["vx"].device.type, "cpu")
        self.assertEqual(data["ax"].device.type, "cpu")

    def test_synthetic_train_only_idx_reader(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / "train-images-idx3-ubyte"
            with images.open("wb") as handle:
                handle.write(struct.pack(">IIII", 2051, 60_000, 28, 28))
                handle.truncate(16 + 60_000 * 784)
            labels = root / "train-labels-idx1-ubyte"
            with labels.open("wb") as handle:
                handle.write(struct.pack(">II", 2049, 60_000))
                handle.truncate(8 + 60_000)
            x, y = plan.read_training(root)
            self.assertEqual((x.dtype, tuple(x.shape)),
                             (torch.uint8, (60_000, 784)))
            self.assertEqual((y.dtype, tuple(y.shape)),
                             (torch.int64, (60_000,)))
            self.assertEqual((int(x.sum()), int(y.sum())), (0, 0))
            self.assertFalse((root / "t10k-images-idx3-ubyte").exists())


if __name__ == "__main__":
    unittest.main()
