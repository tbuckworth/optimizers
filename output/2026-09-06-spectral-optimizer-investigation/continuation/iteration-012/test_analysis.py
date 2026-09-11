"""Pure scalar missingness checks for the I12 postprocessor."""
import unittest

import analyse_moments as analysis


class FactorialMissingnessTests(unittest.TestCase):
    def test_each_contrast_depends_only_on_its_declared_arms(self):
        base = {"inherited": 1.0, "zero_m": 2.0, "zero_v": 4.0,
                "zero_mv": 8.0, "fresh_adam": 16.0}
        complete = analysis.factorial_values(base)
        self.assertEqual(complete["delta_v_minus_m"], 2.0)

        no_fresh = {**base, "fresh_adam": None}
        values = analysis.factorial_values(no_fresh)
        self.assertEqual(values["delta_v_minus_m"], complete["delta_v_minus_m"])
        self.assertEqual(values["m_removal_marginal"], complete["m_removal_marginal"])
        self.assertEqual(values["v_removal_marginal"], complete["v_removal_marginal"])
        self.assertEqual(values["m_by_v_interaction"], complete["m_by_v_interaction"])
        self.assertIsNone(values["counter_effect_empty_moments"])

        no_inherited = {**base, "inherited": None}
        values = analysis.factorial_values(no_inherited)
        self.assertEqual(values["delta_v_minus_m"], complete["delta_v_minus_m"])
        self.assertIsNone(values["m_removal_marginal"])
        self.assertIsNone(values["v_removal_marginal"])
        self.assertIsNone(values["m_by_v_interaction"])
        self.assertEqual(values["counter_effect_empty_moments"],
                         complete["counter_effect_empty_moments"])


if __name__ == "__main__":
    unittest.main()
