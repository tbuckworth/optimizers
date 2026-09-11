"""Synthetic NumPy fixtures only: no dataset, model, GPU, or training."""

import unittest

import numpy as np

from experiments.spectral_general_augmentation_data import (
    BATCH, EVAL_PER_CLASS, SEEDS, STEPS, TRAIN_PER_CLASS, make_plan, translate,
)


def pixel_loop_oracle(images, shifts):
    """Independent scalar source-to-destination mapping, without slice logic."""
    result = np.zeros((len(images), 784), dtype=np.float32)
    for i in range(len(images)):
        dx, dy = map(int, shifts[i])
        for row in range(28):
            for col in range(28):
                moved_row, moved_col = row + dy, col + dx
                if 0 <= moved_row < 28 and 0 <= moved_col < 28:
                    result[i, 28 * moved_row + moved_col] = images[i, 28 * row + col]
    return result


class PlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.labels = np.repeat(np.arange(10, dtype=np.int64), 1100)

    def test_constants_shapes_and_balanced_disjoint_splits(self):
        self.assertEqual(SEEDS, (202609141, 202609142, 202609143))
        self.assertEqual((STEPS, BATCH, TRAIN_PER_CLASS, EVAL_PER_CLASS),
                         (4000, 64, 500, 500))
        for seed in SEEDS:
            with self.subTest(seed=seed):
                plan = make_plan(self.labels, seed)
                self.assertEqual(set(plan), {"train_ids", "eval_ids", "occurrences", "shifts"})
                for name in ("train_ids", "eval_ids"):
                    ids = plan[name]
                    self.assertEqual((ids.shape, ids.dtype), ((5000,), np.dtype("int64")))
                    self.assertEqual(np.unique(ids).size, 5000)
                    self.assertTrue(np.all((ids >= 0) & (ids < len(self.labels))))
                    np.testing.assert_array_equal(self.labels[ids], np.repeat(np.arange(10), 500))
                self.assertEqual(np.intersect1d(plan["train_ids"], plan["eval_ids"]).size, 0)
                self.assertEqual(plan["occurrences"].shape, (4000, 64))
                self.assertEqual(plan["occurrences"].dtype, np.dtype("int64"))
                self.assertTrue(np.all((plan["occurrences"] >= 0) & (plan["occurrences"] < 5000)))
                self.assertEqual(plan["shifts"].shape, (4000, 64, 2))
                self.assertEqual(plan["shifts"].dtype, np.dtype("int8"))
                self.assertTrue(np.all((plan["shifts"] >= -2) & (plan["shifts"] <= 2)))
                for array in plan.values():
                    self.assertTrue(array.flags.c_contiguous)

    def test_exact_named_stream_contract(self):
        seed = SEEDS[0]
        plan = make_plan(self.labels, seed)
        rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([0, seed])))
        expected_train, expected_eval = [], []
        for digit in range(10):
            ids = np.flatnonzero(self.labels == digit)
            rng.shuffle(ids)
            expected_train.extend(ids[:500])
            expected_eval.extend(ids[500:1000])
        np.testing.assert_array_equal(plan["train_ids"], expected_train)
        np.testing.assert_array_equal(plan["eval_ids"], expected_eval)
        occurrence_rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([1, seed])))
        shift_rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([2, seed])))
        np.testing.assert_array_equal(plan["occurrences"], occurrence_rng.integers(
            0, 5000, size=(4000, 64), dtype=np.int64))
        np.testing.assert_array_equal(plan["shifts"], shift_rng.integers(
            -2, 3, size=(4000, 64, 2), dtype=np.int8))

    def test_determinism_pairing_and_independent_result_storage(self):
        for seed in SEEDS:
            p, q = make_plan(self.labels, seed), make_plan(self.labels, seed)
            for key in p:
                np.testing.assert_array_equal(p[key], q[key])
                self.assertFalse(np.shares_memory(p[key], q[key]))
            # A consumer's accidental mutation must not alter another plan.
            original = q["shifts"].copy()
            p["shifts"].fill(0)
            np.testing.assert_array_equal(q["shifts"], original)
        first, second = (make_plan(self.labels, seed) for seed in SEEDS[:2])
        for key in first:
            self.assertFalse(np.array_equal(first[key], second[key]))

    def test_split_consumption_does_not_change_other_streams(self):
        p = make_plan(self.labels, SEEDS[0])
        more_labels = np.repeat(np.arange(10, dtype=np.uint8), 1200)
        q = make_plan(more_labels, SEEDS[0])
        self.assertFalse(np.array_equal(p["train_ids"], q["train_ids"]))
        np.testing.assert_array_equal(p["occurrences"], q["occurrences"])
        np.testing.assert_array_equal(p["shifts"], q["shifts"])

    def test_inputs_and_global_rng_untouched(self):
        labels = self.labels.copy()
        before = labels.copy()
        labels.setflags(write=False)
        state = np.random.get_state()
        make_plan(labels, np.int64(SEEDS[0]))
        after = np.random.get_state()
        np.testing.assert_array_equal(labels, before)
        self.assertEqual(state[0], after[0])
        np.testing.assert_array_equal(state[1], after[1])
        self.assertEqual(state[2:], after[2:])

    def test_invalid_labels_and_seeds(self):
        for labels in (self.labels.tolist(), self.labels.astype(float),
                       self.labels.astype(bool), self.labels.reshape(10, 1100),
                       np.array([], dtype=np.int64), self.labels[:-101],
                       np.append(self.labels, -1), np.append(self.labels, 10)):
            with self.subTest(kind=type(labels), shape=getattr(labels, "shape", None)):
                with self.assertRaises((TypeError, ValueError)):
                    make_plan(labels, SEEDS[0])
        for seed in (-1, True, np.bool_(False), 1.5, "1", None):
            with self.subTest(seed=seed):
                with self.assertRaises((TypeError, ValueError)):
                    make_plan(self.labels, seed)


class TranslationTests(unittest.TestCase):
    def test_all_25_shifts_match_independent_pixel_loop(self):
        shifts = np.array([(dx, dy) for dx in range(-2, 3)
                           for dy in range(-2, 3)], dtype=np.int8)
        image = np.arange(784, dtype=np.float32) / np.float32(783)
        images = np.tile(image, (25, 1))
        before_images, before_shifts = images.copy(), shifts.copy()
        result = translate(images, shifts)
        np.testing.assert_array_equal(result, pixel_loop_oracle(images, shifts))
        np.testing.assert_array_equal(images, before_images)
        np.testing.assert_array_equal(shifts, before_shifts)
        self.assertFalse(np.shares_memory(result, images))
        self.assertEqual(result.dtype, np.dtype("float32"))
        self.assertTrue(result.flags.c_contiguous)
        ones = translate(np.ones_like(images), shifts)
        for row, (dx, dy) in enumerate(shifts):
            self.assertEqual(np.count_nonzero(ones[row]), (28 - abs(int(dx))) * (28 - abs(int(dy))))

    def test_axis_signs_and_no_wrap_at_edges(self):
        images = np.zeros((3, 784), dtype=np.float32)
        images[0, 28 * 10 + 11] = 1
        images[1, 0] = 1
        images[2, 783] = 1
        result = translate(images, np.array([[2, -1], [-1, 0], [0, 1]], dtype=np.int64))
        self.assertEqual(result[0, 28 * 9 + 13], 1)
        self.assertEqual(np.count_nonzero(result[0]), 1)
        self.assertEqual(np.count_nonzero(result[1:]), 0)

    def test_strided_readonly_and_zero_shift_copy(self):
        images = np.linspace(0, 1, 3 * 1568, dtype=np.float32).reshape(3, 1568)[:, ::2]
        shifts = np.array([[0, 99, 0, 99], [1, 99, -2, 99], [-2, 99, 2, 99]])[:, ::2]
        before_images, before_shifts = images.copy(), shifts.copy()
        self.assertFalse(images.flags.c_contiguous)
        self.assertFalse(shifts.flags.c_contiguous)
        images.setflags(write=False)
        shifts.setflags(write=False)
        result = translate(images, shifts)
        np.testing.assert_array_equal(result, pixel_loop_oracle(images, shifts))
        np.testing.assert_array_equal(result[0], images[0])
        self.assertTrue(result.flags.c_contiguous)
        self.assertFalse(np.shares_memory(result, images))
        np.testing.assert_array_equal(images, before_images)
        np.testing.assert_array_equal(shifts, before_shifts)
        unsigned = np.array([[0, 0]], dtype=np.uint64)
        np.testing.assert_array_equal(translate(images[:1], unsigned), images[:1])

    def test_empty_batch(self):
        result = translate(np.empty((0, 784), dtype=np.float32), np.empty((0, 2), dtype=np.int8))
        self.assertEqual((result.shape, result.dtype), ((0, 784), np.dtype("float32")))
        self.assertTrue(result.flags.c_contiguous)

    def test_invalid_images_and_shifts(self):
        images = np.zeros((1, 784), dtype=np.float32)
        shifts = np.zeros((1, 2), dtype=np.int8)
        for bad in (images.tolist(), images.astype(np.float64), images.astype(np.int64),
                    images.reshape(1, 28, 28), images[:, :-1],
                    np.full_like(images, np.nan), np.full_like(images, np.inf),
                    np.full_like(images, -.01), np.full_like(images, 1.01)):
            with self.subTest(kind=type(bad), shape=getattr(bad, "shape", None)):
                with self.assertRaises((TypeError, ValueError)):
                    translate(bad, shifts)
        for bad in (shifts.tolist(), shifts.astype(float), shifts.astype(bool),
                    shifts.ravel(), np.zeros((2, 2), dtype=np.int8),
                    np.array([[0, 3]]), np.array([[-3, 0]]),
                    np.array([[2**64 - 1, 0]], dtype=np.uint64)):
            with self.subTest(kind=type(bad), shape=getattr(bad, "shape", None)):
                with self.assertRaises((TypeError, ValueError)):
                    translate(images, bad)


if __name__ == "__main__":
    unittest.main()
