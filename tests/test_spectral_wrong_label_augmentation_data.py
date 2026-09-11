"""Synthetic label arrays only; no dataset, model, or scientific acquisition."""

import unittest

import numpy as np

from experiments import spectral_general_augmentation_data as ordinary
from experiments import spectral_wrong_label_augmentation_data as data


class WrongLabelPlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.labels = np.repeat(np.arange(10, dtype=np.int64), 1100)

    def test_exact_constants_and_unchanged_transform(self):
        self.assertEqual(data.SEEDS, (202609151, 202609152, 202609153))
        self.assertEqual((data.STEPS, data.BATCH, data.TRAIN_PER_CLASS, data.EVAL_PER_CLASS,
                          data.WRONG_PER_CLASS), (4000, 64, 500, 500, 400))
        self.assertIs(data.translate, ordinary.translate)
        self.assertEqual(ordinary.SEEDS, (202609141, 202609142, 202609143))

    def test_exactly_400_guaranteed_wrong_labels_per_true_class(self):
        for seed in data.SEEDS:
            with self.subTest(seed=seed):
                plan = data.make_plan(self.labels, seed)
                self.assertEqual(set(plan), {"train_ids", "eval_ids", "occurrences", "shifts",
                                            "train_labels", "eval_labels", "assigned_labels", "corruption_mask"})
                true, assigned, mask = (plan[key] for key in ("train_labels", "assigned_labels", "corruption_mask"))
                self.assertEqual(mask.dtype, np.dtype("bool"))
                self.assertEqual(mask.shape, (5000,))
                self.assertEqual(mask.sum(), 4000)
                np.testing.assert_array_equal(mask, assigned != true)
                np.testing.assert_array_equal(assigned[~mask], true[~mask])
                np.testing.assert_array_equal(np.bincount(true[mask], minlength=10), np.full(10, 400))
                np.testing.assert_array_equal(np.bincount(true[~mask], minlength=10), np.full(10, 100))
                np.testing.assert_array_equal(plan["train_labels"], self.labels[plan["train_ids"]])
                np.testing.assert_array_equal(plan["eval_labels"], self.labels[plan["eval_ids"]])
                for key in ("train_labels", "eval_labels", "assigned_labels"):
                    self.assertEqual((plan[key].shape, plan[key].dtype), ((5000,), np.dtype("int64")))
                    self.assertTrue(np.all((plan[key] >= 0) & (plan[key] <= 9)))
                self.assertEqual(np.intersect1d(plan["train_ids"], plan["eval_ids"]).size, 0)

    def test_exact_stream3_selection_and_stream4_offsets(self):
        seed = data.SEEDS[0]
        plan = data.make_plan(self.labels, seed)
        selection = np.random.Generator(np.random.PCG64(np.random.SeedSequence([3, seed])))
        expected_mask = np.zeros(5000, dtype=bool)
        for digit in range(10):
            local = np.flatnonzero(plan["train_labels"] == digit)
            selection.shuffle(local)
            expected_mask[local[:400]] = True
        offsets = np.random.Generator(np.random.PCG64(np.random.SeedSequence([4, seed]))).integers(
            1, 10, size=5000, dtype=np.int64)
        expected_labels = plan["train_labels"].copy()
        expected_labels[expected_mask] = (expected_labels[expected_mask] + offsets[expected_mask]) % 10
        np.testing.assert_array_equal(plan["corruption_mask"], expected_mask)
        np.testing.assert_array_equal(plan["assigned_labels"], expected_labels)

    def test_corruption_does_not_change_split_occurrences_or_shifts(self):
        for seed in data.SEEDS:
            base, noisy = ordinary.make_plan(self.labels, seed), data.make_plan(self.labels, seed)
            for key in base:
                np.testing.assert_array_equal(noisy[key], base[key])

    def test_repeatability_pairing_and_fresh_seed_differences(self):
        for seed in data.SEEDS:
            p, q = data.make_plan(self.labels, seed), data.make_plan(self.labels, seed)
            for key in p:
                np.testing.assert_array_equal(p[key], q[key])
                self.assertFalse(np.shares_memory(p[key], q[key]))
            # The target is indexed by local/source example identity. Duplicates
            # retain it even when different occurrence shifts are later applied.
            batch = np.array([0, 0, 500, 500, 0], dtype=np.int64)
            targets = p["assigned_labels"][batch]
            self.assertEqual(targets[0], targets[1])
            self.assertEqual(targets[0], targets[-1])
            self.assertEqual(targets[2], targets[3])
        p, q = (data.make_plan(self.labels, seed) for seed in data.SEEDS[:2])
        self.assertFalse(np.array_equal(p["corruption_mask"], q["corruption_mask"]))
        self.assertFalse(np.array_equal(p["assigned_labels"], q["assigned_labels"]))

    def test_no_input_or_global_rng_mutation_and_invalids(self):
        labels = self.labels.copy()
        labels.setflags(write=False)
        before = np.random.get_state()
        data.make_plan(labels, np.int64(data.SEEDS[0]))
        after = np.random.get_state()
        np.testing.assert_array_equal(labels, self.labels)
        self.assertEqual(before[0], after[0])
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])
        for invalid in (-1, True, 1.25, None):
            with self.assertRaises((TypeError, ValueError)):
                data.make_plan(labels, invalid)
        for invalid in (labels.astype(float), labels.astype(bool), labels[:-101]):
            with self.assertRaises((TypeError, ValueError)):
                data.make_plan(invalid, data.SEEDS[0])


if __name__ == "__main__":
    unittest.main()
