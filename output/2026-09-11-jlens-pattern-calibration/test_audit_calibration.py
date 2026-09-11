"""Fabricated-only tests of the independent checker; no scientific files read."""
import unittest
import numpy as np
from audit_calibration import audit_moments, audit_topk, audit_references


def moments():
    h = np.array([[1, 2, 0], [-1, -2, 0], [0, 1, 3], [0, -1, -3]], dtype=np.float32)
    u = np.eye(3, dtype=np.float32)[:, :2]
    gaps = np.array([[2, 4], [0, 2]], dtype=np.float64)
    cross = np.array([[4, 8], [8, 20], [0, 12]], dtype=np.float64)
    energy = np.array([4, 20], dtype=np.float64)
    patterns = cross/energy
    unit = (patterns/np.linalg.norm(patterns, axis=0)).astype(np.float32).T
    saved = {'patterns64': patterns, 'cross64': cross, 'score_energy64': energy,
             'calibration_gaps64': gaps, 'max_single_pair_energy_share64': np.array([1., .8]),
             'direction_energy_fraction64': energy/60,
             'unit_score_identity64': np.ones(2),
             'signed_inputs32': np.stack([unit[0], -unit[0], unit[1], -unit[1]])}
    return h, u, [(0, 1), (2, 3)], saved


def ranks():
    logits = np.array([[9, 8, 8, 8, -1]], dtype=np.float32)
    ids = np.array([[0, 3, 1]], dtype=np.int64)
    values = np.array([[9, 8, 8]], dtype=np.float32)
    rows = [{'token_ids': [0, 3, 1], 'scores': [9., 8., 8.],
             'tokens': [' word', '\ufffd', 'fragment'], 'cutoff_logit': 8.,
             'strictly_above_cutoff_count': 1, 'cutoff_tie_count': 3,
             'selected_at_cutoff_count': 2}]
    return logits, ids, values, rows


def references():
    old = {'schema': 'jlens_fresh_references_v1', 'axes': []}
    rows = []
    for i in range(1, 5):
        old['axes'].append({'axis': f'PC{i}',
            'A': {p: [f'old-{i}-{p}-{j}' for j in range(12)] for p in ('positive', 'negative')},
            'C': {p: f'Example {i} {p}' for p in ('positive', 'negative')}})
        rows.extend({'tokens': [f'new-{i}-{p}-{j}' for j in range(12)]} for p in ('positive', 'negative'))
    public = {'schema': 'jlens_pattern_calibration_references_v1', 'arms': {}}
    for arm in ('U', 'P'):
        public['arms'][arm] = {'axes': [
            {'axis': a['axis'], **{p+'_reference': {'example_prefix': a['C'][p],
                'direction_tokens': list(a['A'][p] if arm == 'U' else rows[2*i+j]['tokens'])}
                for j, p in enumerate(('positive', 'negative'))}}
            for i, a in enumerate(old['axes'])]}
    return old, public, rows


class CalibrationAuditTests(unittest.TestCase):
    def test_known_moments(self):
        self.assertEqual(audit_moments(*moments())['pairs'], 2)

    def test_cross_corruption(self):
        args = moments(); args[-1]['cross64'][0, 0] += .1
        with self.assertRaisesRegex(ValueError, 'cross64 arithmetic'):
            audit_moments(*args)

    def test_sign_corruption(self):
        args = moments(); args[-1]['signed_inputs32'][0] *= -1
        with self.assertRaisesRegex(ValueError, 'signed normalized'):
            audit_moments(*args)

    def test_repeated_row(self):
        args = list(moments()); args[2] = [(0, 1), (1, 3)]
        with self.assertRaisesRegex(ValueError, 'each row'):
            audit_moments(*args)

    def test_zero_axis_fails_whole_fit(self):
        args = moments(); args[0][:, 1] = 0
        with self.assertRaisesRegex(ValueError, 'valid full fit'):
            audit_moments(*args)

    def test_any_cutoff_tie_order(self):
        args = ranks(); audit_topk(*args)
        args[1][0] = [0, 2, 3]; args[3][0]['token_ids'] = [0, 2, 3]
        audit_topk(*args)

    def test_non_winner_rejected(self):
        args = ranks(); args[1][0, 0] = 4; args[2][0, 0] = -1
        with self.assertRaisesRegex(ValueError, 'sorted saved'):
            audit_topk(*args)

    def test_duplicate_id(self):
        args = ranks(); args[1][0, 2] = 3
        with self.assertRaisesRegex(ValueError, 'unique valid'):
            audit_topk(*args)

    def test_bad_tie_count(self):
        args = ranks(); args[3][0]['cutoff_tie_count'] = 2
        with self.assertRaisesRegex(ValueError, 'cutoff accounting'):
            audit_topk(*args)

    def test_nonfinite(self):
        args = ranks(); args[0][0, -1] = np.nan
        with self.assertRaisesRegex(ValueError, 'vocabulary logits'):
            audit_topk(*args)

    def test_identical_reference_contract(self):
        audit_references(*references())

    def test_control_tokens_cannot_change(self):
        args = references(); args[1]['arms']['U']['axes'][0]['positive_reference']['direction_tokens'][0] = 'new gloss'
        with self.assertRaisesRegex(ValueError, 'unchanged examples/control'):
            audit_references(*args)

    def test_treatment_example_cannot_change(self):
        args = references(); args[1]['arms']['P']['axes'][1]['negative_reference']['example_prefix'] = 'replacement'
        with self.assertRaisesRegex(ValueError, 'unchanged examples/control'):
            audit_references(*args)

    def test_public_field_cannot_reveal_method(self):
        args = references(); args[1]['arms']['P']['axes'][0]['better_method'] = True
        with self.assertRaisesRegex(ValueError, 'same axis/order/schema'):
            audit_references(*args)


if __name__ == '__main__':
    unittest.main()
