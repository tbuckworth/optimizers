"""Fabricated CPU/NumPy fixtures only; no model, dataset, or scientific run."""
import unittest
import numpy as np

from experiments.spectral_augmentation_masks import (
    MaskPlan, make_mask_plan, rectangles_for_mode, apply_masks,
    coverage_summary, exact_random_geometry, EXPERIMENT_SEEDS,
)


class MaskTests(unittest.TestCase):
    def test_global_rng_untouched_and_uncued_examples_masked(self):
        state = np.random.get_state()
        make_mask_plan(202609131, (2, 3))
        after = np.random.get_state()
        self.assertEqual(state[0], after[0])
        np.testing.assert_array_equal(state[1], after[1])
        self.assertEqual(state[2:], after[2:])
        p = MaskPlan(np.ones((1, 2), dtype=bool),
                     np.zeros((1, 2, 2), dtype=np.int16))
        uncued = np.full((2, 784), .5, dtype=np.float32)
        result = apply_masks(uncued, rectangles_for_mode(p, "targeted")[0])
        self.assertTrue(np.all((result == 0).sum(axis=1) == 64))
        self.assertTrue(np.all(uncued == .5))

    def test_repeatability_pairing_and_streams(self):
        for seed in EXPERIMENT_SEEDS:
            p = make_mask_plan(seed, (3, 8))
            q = make_mask_plan(seed, (3, 8))
            np.testing.assert_array_equal(p.gates, q.gates)
            np.testing.assert_array_equal(p.centers, q.centers)
            expected_gates = np.random.Generator(np.random.PCG64(
                np.random.SeedSequence([seed, 6]))).random((3, 8)) < .5
            expected_centers = np.random.Generator(np.random.PCG64(
                np.random.SeedSequence([seed, 7]))).integers(
                    0, 28, size=(3, 8, 2), dtype=np.int16)
            np.testing.assert_array_equal(p.gates, expected_gates)
            np.testing.assert_array_equal(p.centers, expected_centers)
            for mode in ("random", "targeted", "opposite"):
                rect = rectangles_for_mode(p, mode)
                np.testing.assert_array_equal(rect[..., 0] != -1, p.gates)
                self.assertEqual(rect.dtype, np.int16)
                self.assertFalse(rect.flags.writeable)
            self.assertFalse(p.gates.flags.writeable)
            self.assertFalse(p.centers.flags.writeable)
        self.assertFalse(np.array_equal(make_mask_plan(EXPERIMENT_SEEDS[0], (3, 8)).centers,
                                        make_mask_plan(EXPERIMENT_SEEDS[1], (3, 8)).centers))

    def test_after_cue_duplicates_no_mutation_and_noop(self):
        p = MaskPlan(np.array([[True, False, True]]),
                     np.array([[[0, 0], [0, 0], [27, 27]]], dtype=np.int16))
        x = np.full((3, 784), .25, dtype=np.float32)
        x.reshape(3, 28, 28)[:, :3, :3] = 1  # cue inserted first
        before = x.copy()
        none = apply_masks(x, rectangles_for_mode(p, "none")[0])
        np.testing.assert_array_equal(none, x)
        self.assertFalse(np.shares_memory(none, x))
        y = apply_masks(x, rectangles_for_mode(p, "targeted")[0]).reshape(3, 28, 28)
        self.assertTrue(np.all(y[0, :8, :8] == 0))
        np.testing.assert_array_equal(y[1].ravel(), x[1])
        self.assertEqual(y.dtype, x.dtype)
        z = apply_masks(x, rectangles_for_mode(p, "opposite")[0]).reshape(3, 28, 28)
        self.assertTrue(np.all(z[:, :3, :3] == 1))
        self.assertTrue(np.all(z[0, 20:, 20:] == 0))
        random = apply_masks(x, rectangles_for_mode(p, "random")[0])
        self.assertFalse(np.array_equal(random[0], random[2]))
        np.testing.assert_array_equal(x, before)
        for mode, any_count, full_count, area in (
                ("none", 0, 0, 0), ("targeted", 2, 2, 128), ("opposite", 0, 0, 128)):
            s = coverage_summary(rectangles_for_mode(p, mode))
            self.assertEqual((s["occurrences"], s["any_cue_coverage"],
                              s["full_cue_coverage"], s["erased_area_total"]),
                             (3, any_count, full_count, area))

    def test_exact_geometry_and_boundaries(self):
        geometry = exact_random_geometry()
        r, s = geometry["rectangles"], geometry["summary"]
        np.testing.assert_array_equal(r[0], [0, 4, 0, 4])
        np.testing.assert_array_equal(r[-1], [23, 28, 23, 28])
        np.testing.assert_array_equal(r[4 * 28 + 4], [0, 8, 0, 8])
        self.assertEqual((s["occurrences"], s["active"], s["any_cue_coverage"],
                          s["full_cue_coverage"], s["erased_area_total"]),
                         (784, 784, 49, 25, 43264))
        self.assertEqual(s["erased_area_counts"][64], 441)
        self.assertEqual(s["erased_area_counts"][16], 1)
        self.assertEqual(s["erased_area_counts"].sum(), 784)
        self.assertFalse(s["erased_area_counts"].flags.writeable)
        x = np.ones((784, 784), dtype=np.float64)
        y = apply_masks(x, r)
        self.assertEqual(int((y == 0).sum()), 43264)
        cue_erased = (y.reshape(784, 28, 28)[:, :3, :3] == 0).sum(axis=(1, 2))
        self.assertEqual(int((cue_erased > 0).sum()), 49)
        self.assertEqual(int((cue_erased == 9).sum()), 25)

    def test_invalids(self):
        for seed in (-1, True, 1.5, np.nan):
            with self.assertRaises((TypeError, ValueError)):
                make_mask_plan(seed, (1, 2))
        for shape in ((0, 2), (1,), [1, 2], (True, 2), (1.5, 2)):
            with self.assertRaises((TypeError, ValueError)):
                make_mask_plan(1, shape)
        p = make_mask_plan(1, (1, 2))
        with self.assertRaises(ValueError):
            rectangles_for_mode(p, "unknown")
        for bad in (MaskPlan(p.gates.astype(int), p.centers),
                    MaskPlan(p.gates, p.centers.astype(float)),
                    MaskPlan(p.gates, np.full((1, 2, 2), 28, dtype=np.int16)),
                    MaskPlan(p.gates.ravel(), p.centers)):
            with self.assertRaises((TypeError, ValueError)):
                rectangles_for_mode(bad, "none")
        good = np.array([[0, 8, 0, 8]], dtype=np.int16)
        for x in (np.ones((1, 784), dtype=int), np.ones((1, 28, 28)),
                  np.full((1, 784), np.nan), np.full((1, 784), np.inf),
                  np.full((1, 784), -0.1), np.full((1, 784), 1.1)):
            with self.assertRaises((TypeError, ValueError)):
                apply_masks(x, good)
        for r in (good.astype(float), good.astype(bool), good.ravel(),
                  np.array([[-1, 8, 0, 8]]), np.array([[0, 29, 0, 8]]),
                  np.array([[8, 0, 0, 8]]), np.array([[0, 8, 8, 8]])):
            with self.assertRaises((TypeError, ValueError)):
                apply_masks(np.ones((1, 784)), r)
        with self.assertRaises(ValueError):
            apply_masks(np.ones((2, 784)), good)


if __name__ == "__main__":
    unittest.main()
