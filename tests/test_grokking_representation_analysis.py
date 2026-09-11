"""Small arithmetic/roster fixtures; no scientific NPZ files are opened."""

import copy
import math
import unittest

import numpy as np

from experiments import analyze_grokking_representations as analysis


def panel_record(arm, fit_scores, eval_scores=None):
    if eval_scores is None:
        eval_scores = [-value for value in fit_scores]
    rows = [{"frequency": index + 1, "fit_r2": fit_scores[index],
             "eval_r2": eval_scores[index]} for index in range(56)]
    return {"seed": 100, "arm": arm, "step": 6000,
            "probes": {"final_hidden": {"observed": {"per_frequency": rows}}}}


class RepresentationAnalysisTest(unittest.TestCase):
    def test_fixed_panel_uses_fit_only_and_breaks_ties_by_frequency(self):
        scores = [0.0] * 56
        for frequency, value in ((7, .9), (4, .9), (12, .8), (2, .7), (9, .6), (1, .5)):
            scores[frequency - 1] = value
        records = [panel_record(arm, scores) for arm in analysis.ARMS]
        panel = analysis.choose_fixed_panel(records)
        self.assertEqual(panel["frequencies"], [4, 7, 12, 2, 9])

        changed = copy.deepcopy(records)
        for record in changed:
            for row in record["probes"]["final_hidden"]["observed"]["per_frequency"]:
                row["eval_r2"] = 1000.0 if row["frequency"] == 56 else -1000.0
        self.assertEqual(analysis.choose_fixed_panel(changed)["frequencies"],
                         panel["frequencies"])

    def test_pool_is_ratio_of_sums_and_margin_excludes_true_class(self):
        metrics = [
            {"count": 2, "numerator": 1.0, "denominator": 4.0,
             "raw_rms": 3.0},
            {"count": 3, "numerator": 3.0, "denominator": 6.0,
             "raw_rms": 2.0},
        ]
        pooled = analysis.pool_ratio(metrics, p=5)
        self.assertEqual(pooled["count"], 5)
        self.assertEqual(pooled["numerator"], 4.0)
        self.assertEqual(pooled["denominator"], 10.0)
        self.assertEqual(pooled["value"], .4)
        self.assertNotEqual(pooled["value"], (.25 + .5) / 2)
        self.assertAlmostEqual(pooled["centered_rms"], math.sqrt(10 / 50))
        self.assertAlmostEqual(pooled["raw_rms"], math.sqrt((36 * 5 + 24 * 5) / 50))

        logits = np.asarray([[3., 1., 0.], [0., 2., 1.], [4., 5., 7.]])
        margins = analysis.correct_class_summary(
            logits, np.asarray([0, 1, 2]), np.asarray([0, 1, 2]))
        self.assertEqual(margins["count"], 3)
        self.assertAlmostEqual(margins["mean"], 5 / 3)

    def test_zero_energy_pool_is_undefined(self):
        pooled = analysis.pool_ratio([
            {"count": 4, "numerator": 0.0, "denominator": 0.0, "raw_rms": 8.0}
        ], p=7)
        self.assertFalse(pooled["energy_defined"])
        self.assertIsNone(pooled["value"])
        self.assertEqual(pooled["centered_rms"], 0.0)

    def test_exact_disjoint_roster(self):
        calibration = analysis.expected_roster("calibration")
        remaining = analysis.expected_roster("remaining")
        analysis.validate_combined_roster(calibration, remaining)
        self.assertEqual(len(calibration), 6)
        self.assertEqual(len(remaining), 144)
        self.assertFalse(set(calibration) & set(remaining))
        self.assertEqual(len(set(calibration + remaining)), 150)

        with self.assertRaisesRegex(ValueError, r"6\+144"):
            analysis.validate_combined_roster(calibration[:-1], remaining)
        overlapping = remaining.copy()
        overlapping[0] = calibration[0]
        with self.assertRaisesRegex(ValueError, r"6\+144"):
            analysis.validate_combined_roster(calibration, overlapping)


if __name__ == "__main__":
    unittest.main()
