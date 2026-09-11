"""Tiny synthetic NumPy fixtures; no real IDX, checkpoint, GPU or acquisition."""

from dataclasses import FrozenInstanceError
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

import numpy as np

from experiments import spectral_strong_augmentation_data as data


def mini_dimensions(**overrides):
    values = dict(source_count=40, train_count=24, validation_count=8,
                  reporting_count=8, epochs=3, batch_size=5)
    return data.Dimensions(**(values | overrides))


def scalar_translation(images, shifts):
    result = np.zeros_like(images)
    for index, (dy, dx) in enumerate(shifts):
        for row in range(28):
            for column in range(28):
                new_row, new_column = row + int(dy), column + int(dx)
                if 0 <= new_row < 28 and 0 <= new_column < 28:
                    result[index, 28 * new_row + new_column] = images[index, 28 * row + column]
    return result


class StrongPlanTests(unittest.TestCase):
    def setUp(self):
        self.dimensions = mini_dimensions()
        self.labels = np.arange(40, dtype=np.int64) % 10

    def test_import_inert_and_production_constants(self):
        code = ("import numpy as np\nfrom unittest.mock import patch\n"
                "with patch.object(np.random, 'Generator', side_effect=AssertionError('RNG on import')):\n"
                "    import experiments.spectral_strong_augmentation_data\n")
        result = subprocess.run([sys.executable, "-c", code], cwd=Path(__file__).resolve().parents[1],
                                capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(data.SEEDS, (202609171, 202609172, 202609173))
        self.assertEqual((data.SOURCE_SIZE, data.TRAIN_SIZE, data.VALIDATION_SIZE,
                          data.REPORTING_SIZE, data.EPOCHS, data.BATCH), (60000, 50000, 5000, 5000, 72, 64))
        self.assertEqual((data.BATCHES_PER_EPOCH, data.STEPS), (782, 56304))
        self.assertEqual(data.Dimensions().updates, 56304)
        boundaries = np.r_[np.arange(0, data.TRAIN_SIZE, data.BATCH, dtype=np.int64), data.TRAIN_SIZE]
        self.assertEqual((len(boundaries), boundaries[-2], np.diff(boundaries)[-1]), (783, 49984, 16))

    def test_all_four_streams_exactly_reconstructed_in_specified_order(self):
        for seed in data.SEEDS:
            d = self.dimensions
            plan = data.make_plan(self.labels, seed, d)
            self.assertEqual(tuple(plan), data.PLAN_KEYS)
            rngs = [np.random.Generator(np.random.PCG64(np.random.SeedSequence([stream, seed])))
                    for stream in range(4)]
            order = rngs[0].permutation(40)
            np.testing.assert_array_equal(plan["train_ids"], order[:24])
            np.testing.assert_array_equal(plan["validation_ids"], order[24:32])
            np.testing.assert_array_equal(plan["reporting_ids"], order[32:])
            selected = rngs[1].random(24) < .9
            replacements = rngs[1].integers(0, 10, size=24, dtype=np.int64)
            np.testing.assert_array_equal(plan["corruption_mask"], selected)
            np.testing.assert_array_equal(plan["replacement_labels"], replacements)
            np.testing.assert_array_equal(plan["assigned_labels"], np.where(selected, replacements, self.labels[order[:24]]))
            np.testing.assert_array_equal(plan["changed_mask"], plan["assigned_labels"] != plan["train_labels"])
            expected_order = np.stack([rngs[2].permutation(24).astype(np.int32) for _ in range(3)])
            np.testing.assert_array_equal(plan["occurrences"], expected_order)
            np.testing.assert_array_equal(plan["shifts"], rngs[3].integers(-2, 3, size=(3, 24, 2), dtype=np.int8))
            np.testing.assert_array_equal(plan["batch_boundaries"], [0, 5, 10, 15, 20, 24])

    def test_selected_is_not_actually_wrong_and_replacements_cover_unselected(self):
        # Fixed fake stream guarantees all three relevant cases without relying
        # on the chance of one tiny seed drawing a coincidentally unchanged label.
        normal_rng = data._rng
        baseline = data.make_plan(self.labels, data.SEEDS[0], self.dimensions)
        true = baseline["train_labels"]
        replacements = (true + 1) % 10
        replacements[0] = true[0]
        probabilities = np.zeros(24)
        probabilities[1] = .95

        class CorruptionRNG:
            def random(self, count):
                self.called_random = count
                return probabilities.copy()

            def integers(self, low, high, *, size, dtype):
                if self.called_random != 24 or (low, high, size, dtype) != (0, 10, 24, np.int64):
                    raise AssertionError("wrong corruption draw order or count")
                return replacements.copy()

        corruption = CorruptionRNG()
        with patch.object(data, "_rng", side_effect=lambda seed, stream: corruption if stream == 1 else normal_rng(seed, stream)):
            plan = data.make_plan(self.labels, data.SEEDS[0], self.dimensions)
        self.assertTrue(plan["corruption_mask"][0])
        self.assertFalse(plan["changed_mask"][0])
        self.assertFalse(plan["corruption_mask"][1])
        self.assertEqual(plan["assigned_labels"][1], true[1])
        self.assertNotEqual(plan["replacement_labels"][1], true[1])
        self.assertTrue(plan["changed_mask"][2:].all())

    def test_disjoint_roles_epoch_coverage_fixed_labels_and_array_dtypes(self):
        plan = data.make_plan(self.labels, data.SEEDS[0], self.dimensions)
        roles = [plan[name + "_ids"] for name in ("train", "validation", "reporting")]
        np.testing.assert_array_equal(np.sort(np.concatenate(roles)), np.arange(40))
        for name in ("train", "validation", "reporting"):
            np.testing.assert_array_equal(plan[name + "_labels"], self.labels[plan[name + "_ids"]])
        for epoch in plan["occurrences"]:
            np.testing.assert_array_equal(np.sort(epoch), np.arange(24))
            recovered = np.empty(24, dtype=np.int64)
            recovered[epoch] = plan["assigned_labels"][epoch]
            np.testing.assert_array_equal(recovered, plan["assigned_labels"])
        for name, value in plan.items():
            expected = np.bool_ if name.endswith("mask") else np.int32 if name == "occurrences" else np.int8 if name == "shifts" else np.int64
            self.assertEqual(value.dtype, np.dtype(expected))
            self.assertTrue(value.flags.c_contiguous and value.flags.writeable)
        self.assertEqual(plan["shifts"].shape, (3, 24, 2))
        self.assertTrue(np.all((plan["shifts"] >= -2) & (plan["shifts"] <= 2)))

    def test_pairing_determinism_independent_storage_and_global_rng_preserved(self):
        self.labels.setflags(write=False)
        before = np.random.get_state()
        first = data.make_plan(self.labels, data.SEEDS[0], self.dimensions)
        second = data.make_plan(self.labels, np.int64(data.SEEDS[0]), self.dimensions)
        after = np.random.get_state()
        self.assertEqual(before[0], after[0])
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])
        for key in first:
            np.testing.assert_array_equal(first[key], second[key])
            self.assertFalse(np.shares_memory(first[key], second[key]))
        first["assigned_labels"].fill(99)
        self.assertTrue(np.all(second["assigned_labels"] < 10))

    def test_batch_layout_and_extra_epochs_do_not_perturb_prior_streams(self):
        first = data.make_plan(self.labels, data.SEEDS[0], self.dimensions)
        changed = data.make_plan(self.labels, data.SEEDS[0], mini_dimensions(batch_size=7, epochs=4))
        for key in first:
            if key == "batch_boundaries":
                continue
            actual = changed[key][:3] if key in ("occurrences", "shifts") else changed[key]
            np.testing.assert_array_equal(first[key], actual)
        np.testing.assert_array_equal(changed["batch_boundaries"], [0, 7, 14, 21, 24])

    def test_dimensions_and_invalid_inputs(self):
        for overrides in ({"epochs": 0}, {"batch_size": True}, {"source_count": 41}, {"train_count": -1}):
            with self.assertRaises(ValueError):
                mini_dimensions(**overrides)
        with self.assertRaises(FrozenInstanceError):
            self.dimensions.epochs = 4
        for labels in (self.labels[:-1], self.labels.reshape(4, 10), self.labels.astype(float),
                       self.labels.astype(bool), self.labels.tolist(), np.full(40, 10), np.full(40, -1)):
            with self.assertRaises((TypeError, ValueError)):
                data.make_plan(labels, data.SEEDS[0], self.dimensions)
        for seed in (True, -1, 2**63, 1.5, None):
            with self.assertRaises((TypeError, ValueError)):
                data.make_plan(self.labels, seed, self.dimensions)


class StrongTransformTests(unittest.TestCase):
    def test_all_25_dy_dx_translations_against_scalar_pixel_oracle(self):
        shifts = np.array([(dy, dx) for dy in range(-2, 3) for dx in range(-2, 3)], dtype=np.int8)
        images = np.tile(np.arange(784, dtype=np.float32) / np.float32(783), (25, 1))
        before, before_shifts = images.copy(), shifts.copy()
        with patch.object(data.ordinary, "translate", wraps=data.ordinary.translate) as reused:
            actual = data.translate(images, shifts)
        self.assertEqual(reused.call_count, 1)
        np.testing.assert_array_equal(reused.call_args.args[1], shifts[:, ::-1])
        np.testing.assert_array_equal(actual, scalar_translation(images, shifts))
        np.testing.assert_array_equal(images, before)
        np.testing.assert_array_equal(shifts, before_shifts)

    def test_asymmetric_impulse_and_standardized_black_padding(self):
        images = np.zeros((1, 784), dtype=np.float32)
        images[0, 28 * 10 + 11] = 1
        translated = data.translate(images, np.array([[-1, 2]], dtype=np.int8))
        self.assertEqual(translated[0, 28 * 9 + 13], 1)
        self.assertEqual(np.count_nonzero(translated), 1)
        normalized = data.normalize(translated)
        black = (np.float32(0) - np.float32(.1307)) / np.float32(.3081)
        white = (np.float32(1) - np.float32(.1307)) / np.float32(.3081)
        self.assertEqual(normalized[0, 0], black)
        self.assertNotEqual(normalized[0, 0], 0)
        self.assertEqual(normalized[0, 28 * 9 + 13], white)
        with self.assertRaises(ValueError):
            data.translate(data.normalize(images), np.array([[-1, 2]], dtype=np.int8))

    def test_normalization_exact_fp32_copy_and_invalids(self):
        images = np.linspace(0, 1, 2 * 1568, dtype=np.float32).reshape(2, 1568)[:, ::2]
        images.setflags(write=False)
        before = images.copy()
        result = data.normalize(images)
        np.testing.assert_array_equal(result, (images - np.float32(.1307)) / np.float32(.3081))
        np.testing.assert_array_equal(images, before)
        self.assertEqual(result.dtype, np.float32)
        self.assertTrue(result.flags.c_contiguous)
        self.assertFalse(np.shares_memory(result, images))
        self.assertEqual(data.normalize(np.empty((0, 784), dtype=np.float32)).shape, (0, 784))
        for bad in (images.astype(float), images[:, :-1], images.tolist(),
                    np.full_like(images, -1), np.full_like(images, np.nan), np.full_like(images, 1.1)):
            with self.assertRaises((TypeError, ValueError)):
                data.normalize(bad)
        for shifts in ([0, 0], np.array([0, 0]), np.zeros((2, 3)), np.full((2, 2), 3), np.zeros((2, 2), dtype=bool)):
            with self.assertRaises((TypeError, ValueError)):
                data.translate(before, shifts)


if __name__ == "__main__":
    unittest.main()
