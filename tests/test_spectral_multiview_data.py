"""Fabricated NumPy plans only: no dataset, model, GPU or scientific run."""

import unittest
from unittest.mock import patch

import numpy as np

from experiments import spectral_general_augmentation_data as ordinary
from experiments import spectral_multiview_data as data


class MultiviewPlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.labels = np.repeat(np.arange(10, dtype=np.uint8), 1100)

    def test_fixed_constants_and_eight_array_schema(self):
        self.assertEqual(data.SEEDS, (202609161, 202609162, 202609163))
        self.assertEqual(data.POLICIES, ("raw1", "native1", "observer4", "raw4"))
        self.assertEqual((data.STEPS, data.BATCH, data.TRAIN_PER_CLASS,
                          data.EVAL_PER_CLASS, data.VIEWS), (4000, 64, 500, 500, 4))
        shapes = {"train_ids": (5000,), "eval_ids": (5000,),
                  "occurrences": (4000, 64), "shifts": (4000, 64, 2),
                  "extra_shifts": (4000, 3, 64, 2), "eval_shifts": (5000, 2),
                  "train_labels": (5000,), "eval_labels": (5000,)}
        for seed in data.SEEDS:
            plan = data.make_plan(self.labels, seed)
            self.assertEqual(set(plan), set(shapes))
            for key, shape in shapes.items():
                value = plan[key]
                self.assertEqual(value.shape, shape)
                self.assertEqual(value.dtype, np.dtype("int8" if "shifts" in key else "int64"))
                self.assertTrue(value.flags.c_contiguous and value.flags.writeable)
                if "shifts" in key:
                    self.assertTrue(np.all((value >= -2) & (value <= 2)))
            self.assertEqual(np.intersect1d(plan["train_ids"], plan["eval_ids"]).size, 0)
            for split in ("train", "eval"):
                np.testing.assert_array_equal(plan[split + "_labels"], np.repeat(np.arange(10), 500))
                np.testing.assert_array_equal(plan[split + "_labels"], self.labels[plan[split + "_ids"]])
                self.assertEqual(np.unique(plan[split + "_ids"]).size, 5000)
            self.assertTrue(np.all((plan["occurrences"] >= 0) & (plan["occurrences"] < 5000)))

    def test_first_three_streams_exactly_reuse_ordinary_plan(self):
        for seed in data.SEEDS:
            expected = ordinary.make_plan(self.labels, seed)
            with patch.object(ordinary, "make_plan", wraps=ordinary.make_plan) as reused:
                actual = data.make_plan(self.labels, seed)
            reused.assert_called_once_with(self.labels, seed)
            for key in expected:
                np.testing.assert_array_equal(actual[key], expected[key])

    def test_extra_and_evaluation_streams_exact_independent_reconstruction(self):
        for seed in data.SEEDS:
            plan = data.make_plan(self.labels, seed)
            for stream, key, shape in ((3, "extra_shifts", (4000, 3, 64, 2)),
                                       (4, "eval_shifts", (5000, 2))):
                rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([stream, seed])))
                np.testing.assert_array_equal(plan[key], rng.integers(-2, 3, shape, dtype=np.int8))
            self.assertFalse(np.array_equal(plan["shifts"], plan["extra_shifts"][:, 0]))

    def test_determinism_pairing_fresh_storage_and_seed_difference(self):
        first = data.make_plan(self.labels, data.SEEDS[0])
        second = data.make_plan(self.labels, np.int64(data.SEEDS[0]))
        other = data.make_plan(self.labels, data.SEEDS[1])
        for key in first:
            np.testing.assert_array_equal(first[key], second[key])
            self.assertFalse(np.shares_memory(first[key], second[key]))
            if not key.endswith("labels"):
                self.assertFalse(np.array_equal(first[key], other[key]))
        first["extra_shifts"].fill(0)
        self.assertTrue(np.any(second["extra_shifts"] != 0))

    def test_split_draw_count_cannot_perturb_other_streams(self):
        first = data.make_plan(self.labels, data.SEEDS[0])
        second = data.make_plan(np.repeat(np.arange(10, dtype=np.int64), 1200), data.SEEDS[0])
        self.assertFalse(np.array_equal(first["train_ids"], second["train_ids"]))
        for key in ("occurrences", "shifts", "extra_shifts", "eval_shifts"):
            np.testing.assert_array_equal(first[key], second[key])

    def test_no_input_or_global_rng_mutation_and_no_label_alias(self):
        labels = self.labels.copy()
        labels.setflags(write=False)
        before = np.random.get_state()
        plan = data.make_plan(labels, data.SEEDS[0])
        after = np.random.get_state()
        self.assertEqual(before[0], after[0])
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])
        np.testing.assert_array_equal(labels, self.labels)
        self.assertFalse(np.shares_memory(labels, plan["train_labels"]))
        self.assertFalse(np.shares_memory(plan["train_labels"], plan["eval_labels"]))

    def test_invalid_inputs_delegate_to_accepted_validation(self):
        for labels in (self.labels.tolist(), self.labels.astype(float),
                       self.labels.astype(bool), self.labels.reshape(10, 1100),
                       np.array([], dtype=np.int64), np.append(self.labels, 10)):
            with self.assertRaises((TypeError, ValueError)):
                data.make_plan(labels, data.SEEDS[0])
        for seed in (-1, True, np.bool_(True), 1.5, None):
            with self.assertRaises((TypeError, ValueError)):
                data.make_plan(self.labels, seed)

    def test_translation_is_the_frozen_helper_not_a_copy(self):
        self.assertIs(data.translate, ordinary.translate)
        images = np.zeros((1, 784), dtype=np.float32)
        images[0, 28 * 10 + 10] = 1
        actual = data.translate(images, np.array([[2, -1]], dtype=np.int8))
        self.assertEqual(actual[0, 28 * 9 + 12], 1)
        self.assertEqual(np.count_nonzero(actual), 1)


if __name__ == "__main__":
    unittest.main()
