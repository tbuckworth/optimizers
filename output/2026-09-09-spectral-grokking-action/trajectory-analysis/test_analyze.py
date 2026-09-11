"""Small synthetic JSON fixtures; no experiment input or numerical libraries."""

import copy
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('trajectory_analyze', Path(__file__).with_name('analyze.py'))
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def row(policy='norm_matched'):
    norms = {'orthogonal': 2.0}
    diagnostic = {'input_norm': 4.0, 'action_norms': norms, 'pairwise_cosines': {}}
    if policy == 'norm_matched':
        norms.update(native=6.0, norm_matched=6.0)
        diagnostic.update(norm_match_scale=3.0, norm_match_degenerate=False,
            norm_match_denominator_clamped=False, norm_match_exact=True,
            norm_match_relative_tolerance=analysis.RELATIVE_TOLERANCE,
            norm_match_absolute_mismatch=0.0, norm_match_relative_mismatch=0.0,
            pairwise_cosines={'native_orthogonal': 0.8, 'native_norm_matched': 0.8, 'orthogonal_norm_matched': 1.0})
    return {'step': 1502, 'P': 227313, 'k': 2, 'numerical_rank': 2,
            'basis_column_count': 2, 'full_column_rank': True,
            'singular_values': [2.0, 1.0], 'sigma_max': 2.0,
            'tolerance': 227313 * analysis.sys.float_info.epsilon * 2,
            'svd_truncation_relative_frobenius_residual': 0.0,
            'elapsed_seconds': 0.001, 'coefficients': {k: [1.0, 1.0] for k in norms},
            'diagnostics': diagnostic}


class TrajectoryFixtures(unittest.TestCase):
    def test_both_policy_availability(self):
        orth = analysis.extract(row('orthogonal'), 'orthogonal', 1502)
        matched = analysis.extract(row(), 'norm_matched', 1502)
        self.assertNotIn('c', orth)
        self.assertEqual(matched['c'], 3)
        self.assertEqual(orth['delivered_over_raw'], 0.5)
        self.assertEqual(matched['delivered_over_raw'], 1.5)

    def test_bad_step_rank_and_c_rejected(self):
        for key, value in [('step', 1503), ('numerical_rank', 1)]:
            data = row()
            data[key] = value
            with self.assertRaises(ValueError):
                analysis.extract(data, 'norm_matched', 1502)
        data = row()
        data['diagnostics']['norm_match_scale'] = 2.99
        with self.assertRaises(ValueError):
            analysis.extract(data, 'norm_matched', 1502)

    def test_missing_field_is_not_imputed(self):
        data = row()
        del data['diagnostics']['pairwise_cosines']['native_orthogonal']
        with self.assertRaises(ValueError):
            analysis.extract(data, 'norm_matched', 1502)

    def test_no_undefined_subset_average(self):
        value = analysis.stats([1.0, None, 3.0])
        self.assertEqual(value['defined_count'], 2)
        self.assertEqual(value['undefined_count'], 1)
        self.assertIsNone(value['mean'])
        self.assertIsNone(value['minimum'])
        self.assertIsNone(analysis.ratio(0.0, 0.0))

    def test_fixed_window_requires_every_step(self):
        initial = analysis.extract(row('orthogonal'), 'orthogonal', 1502)
        following = copy.deepcopy(initial)
        following['step'] = 1503
        following['k'] = 4
        result = analysis.summarize([initial, following], 'orthogonal', 1502, 1503)
        self.assertEqual(result['metrics']['k']['mean'], 3)
        self.assertEqual(result['metrics']['k']['median'], 3)
        with self.assertRaises(ValueError):
            analysis.summarize([initial], 'orthogonal', 1502, 1503)

    def test_nonfinite_is_rejected(self):
        data = row()
        data['diagnostics']['input_norm'] = float('nan')
        with self.assertRaises(ValueError):
            analysis.extract(data, 'norm_matched', 1502)


if __name__ == '__main__':
    unittest.main()
