"""Fabricated-only tests. Never read scientific JSON or NPZ files."""

import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import analyze


def fixture():
    rows = []
    for row_id in analyze.ALL_IDS:
        group, i, style = row_id.split('-')
        prefix = f'Fabricated {group} {i}'
        rows.append({'id': row_id, 'group': group, 'pair': f'{group}-{i}', 'style': style,
                     'split': 'fit' if int(i) < 4 else 'heldout',
                     'prefix': prefix if style == 'plain' else 'A brief factual note:\n'+prefix})
    selection = {'feature': 'activation_11', 'fit_ids': list(analyze.FIT_IDS)}
    topic = np.repeat([-3., -1., 1., 3.], 8)
    content = np.tile(np.repeat([-3., -1., 1., 3.], 2), 4)
    framing = np.tile([-2., 2.], 16)
    return rows, selection, np.stack([topic, content, framing, topic+content+framing], axis=1)


class TestDecomposition(unittest.TestCase):
    def test_known_components(self):
        rows, selection, z = fixture()
        result = analyze.decompose(rows, selection, z)
        expected = [(5, 5, 0, 0), (5, 0, 5, 0), (4, 0, 0, 4), (14, 5, 5, 4)]
        for axis, values in zip(result['axes'], expected):
            self.assertEqual(tuple(axis[k] for k in ['V', 'B', 'Q', 'F']), values)
            self.assertEqual(axis['reconstruction_residual_V_minus_B_Q_F'], 0)
            self.assertEqual(axis['framing_identity_residual_F_minus_mean_squared_shift_over_4'], 0)
        self.assertEqual(len(result['rows']), 32)
        self.assertEqual(len(result['topic_means']), 4)
        self.assertEqual(len(result['pairs']), 16)
        self.assertEqual(result['pairs'][0]['note_minus_plain64'], [0, 0, 4, 4])
        self.assertEqual(result['axes'][2]['mean_note_minus_plain'], 4)
        self.assertEqual(result['axes'][2]['rms_note_minus_plain'], 4)

    def test_content_dependent_framing(self):
        rows, selection, z = fixture()
        z[:] = np.tile([-2., 2., 2., -2.], 8)[:, None]
        for axis in analyze.decompose(rows, selection, z)['axes']:
            self.assertEqual([axis[k] for k in ['V', 'B', 'Q', 'F']], [4, 0, 0, 4])
            self.assertEqual(axis['mean_note_minus_plain'], 0)
            self.assertEqual(axis['rms_note_minus_plain'], 4)

    def test_zero_variance(self):
        rows, selection, z = fixture()
        for value in [0., 7.]:
            result = analyze.decompose(rows, selection, np.full_like(z, value))
            for axis in result['axes']:
                self.assertEqual([axis[k] for k in ['V', 'B', 'Q', 'F']], [0]*4)
                self.assertEqual(axis['fractions'], {'B': None, 'Q': None, 'F': None})

    def test_scaling_sign_and_translation(self):
        rows, selection, z = fixture()
        original = analyze.decompose(rows, selection, z)
        for scale in [3., -3.]:
            changed = analyze.decompose(rows, selection, scale*z+7)
            for a, b in zip(original['axes'], changed['axes']):
                for key in ['V', 'B', 'Q', 'F']:
                    self.assertEqual(b[key], 9*a[key])
                self.assertEqual(a['fractions'], b['fractions'])
                self.assertEqual(b['mean_note_minus_plain'], scale*a['mean_note_minus_plain'])
                self.assertEqual(b['rms_note_minus_plain'], 3*a['rms_note_minus_plain'])

    def test_bad_inputs(self):
        rows, selection, z = fixture()
        for key, value in [('group', 'wrong'), ('pair', 'wrong'), ('style', 'wrong'), ('split', 'heldout'), ('prefix', '')]:
            bad = copy.deepcopy(rows); bad[0][key] = value
            with self.assertRaises(ValueError): analyze.decompose(bad, selection, z)
        bad = copy.deepcopy(rows); bad[0], bad[1] = bad[1], bad[0]
        with self.assertRaises(ValueError): analyze.decompose(bad, selection, z)
        bad_selection = copy.deepcopy(selection); bad_selection['fit_ids'].reverse()
        with self.assertRaises(ValueError): analyze.decompose(rows, bad_selection, z)
        for bad_z in [z[:-1], z.astype(np.float32), z.tolist(), np.full_like(z, np.nan), np.full_like(z, np.inf)]:
            with self.assertRaises(ValueError): analyze.decompose(rows, selection, bad_z)
        bad = copy.deepcopy(rows); bad[1]['prefix'] = 'Different content'
        with self.assertRaises(ValueError): analyze.decompose(bad, selection, z)

    def test_heldout_text_excluded(self):
        rows, selection, z = fixture()
        first = analyze.decompose(rows, selection, z)
        for row in rows:
            if row['split'] == 'heldout': row['prefix'] = 'arbitrary heldout text'
        self.assertEqual(first, analyze.decompose(rows, selection, z))
        self.assertTrue(all(row['split'] == 'fit' for row in first['rows']))

    def test_guards_before_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            for name in ['attempt.json', 'analysis']:
                target = directory/name
                target.symlink_to(directory/'missing')
                with patch.object(analyze, 'load_inputs', side_effect=AssertionError('scientific read')) as loader:
                    with self.assertRaises(FileExistsError): analyze.run(directory)
                    loader.assert_not_called()
                target.unlink()
            (directory/'analysis').mkdir()
            with patch.object(analyze, 'load_inputs') as loader:
                with self.assertRaises(FileExistsError): analyze.run(directory)
                loader.assert_not_called()

    def test_failed_attempt_consumed(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            with patch.object(analyze, 'load_inputs', side_effect=ValueError('fabricated load failure')):
                with self.assertRaises(ValueError): analyze.run(directory)
            self.assertTrue((directory/'attempt.json').is_file())
            with patch.object(analyze, 'load_inputs') as loader:
                with self.assertRaises(FileExistsError): analyze.run(directory)
                loader.assert_not_called()


if __name__ == '__main__':
    unittest.main()
