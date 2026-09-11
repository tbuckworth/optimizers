"""Tiny fabricated arrays only; no scientific producer import or model work."""
import copy
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import torch

PATH = Path(__file__).resolve().parents[1] / 'scripts/audit_spectral_batch_composition.py'
SPEC = importlib.util.spec_from_file_location('independent_batch_composition_audit', PATH)
a = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(a)
torch.set_num_threads(1)


def fake_snapshot(step, theta, first, second, gradient=None):
    tensor = lambda x: torch.from_numpy(np.asarray(x, np.float32).copy())
    tracker = {'rank': 32, 'decay': .99, 'warmup': 100, 'stable_update': True,
        'stabilize_every': 100, 'relative_eig_tol': 1e-8, 'absolute_eig_floor': 0.,
        'weighting': 'hard', 'normalize': 'none', 'adaptive': 'none', 'proj_k': None,
        'filter_strength': 1., 'energy_threshold': None, 'n_params': 3,
        'step_count': step, 'grad_mean': tensor([.1, .1, .1]),
        'V': tensor([[1.], [0.], [0.]]), 'S': torch.tensor([1.], dtype=torch.float64)}
    group = {'params': [77], 'lr': .001, 'weight_decay': .01, 'betas': (.9, .999), 'eps': 1e-8,
        'foreach': False, 'fused': False, 'amsgrad': False, 'maximize': False,
        'capturable': False, 'differentiable': False}
    return {'schema': 'i9_neural_snapshot_v1', 'model_spec': {'input_dim': 784, 'width': 64, 'classes': 10},
        'model_state': {'toy': tensor(theta)}, 'gradients': [None if gradient is None else tensor(gradient)],
        'model_modes': [('', True)], 'optimizer': {'param_groups': [group],
            'state': {77: {'step': torch.tensor(float(step)), 'exp_avg': tensor(first), 'exp_avg_sq': tensor(second)}}},
        'tracker': tracker, 'rng': {'toy': torch.tensor([1, 2, 3], dtype=torch.uint8)}}


def fake_diagnostic(clean=False, update=101):
    """Three coordinates, recorded by pure algebra; no model or optimizer object."""
    old = np.array([1., -2., .5], np.float32)
    m0, v0 = np.array([.1, -.2, .05], np.float32), np.array([.01, .04, .0025], np.float32)
    raw, delivered = np.array([.2, .3, .1], np.float32), np.array([.2, 0., 0.], np.float32)
    warm = fake_snapshot(100, old, m0, v0)
    before = fake_snapshot(update - 1, old, m0, v0, delivered)
    before['tracker']['step_count'] = update
    m1 = (.9 * m0.astype(np.float64) + .1 * delivered).astype(np.float32)
    v1 = (.999 * v0.astype(np.float64) + .001 * delivered.astype(np.float64)**2).astype(np.float32)
    adaptive = -.001 * (m1.astype(np.float64) / (1 - .9**update)) / (np.sqrt(v1.astype(np.float64) / (1 - .999**update)) + 1e-8)
    new = (old.astype(np.float64) * (1 - .001 * .01) + adaptive).astype(np.float32)
    after = fake_snapshot(update, new, m1, v1, delivered)
    true = np.concatenate([np.full(6 if d != 8 else 50, d, np.int64) for d in range(10)])
    assigned = true.copy()
    if not clean:
        assigned[np.where(true != 8)[0]] = (true[true != 8] + 1) % 8
    plan = {'train_true': true, 'train_ids': np.arange(10, 10 + len(true), dtype=np.int64)}
    inputs = np.resize(np.arange(256, dtype=np.uint8), (len(true), 784)).astype(np.float32) / np.float32(255)
    positions = a.probe_positions(plan, assigned, a.SEEDS[0])
    v = before['tracker']['V'].numpy()
    q, basis = a.h.span_basis(v)
    vectors, movement = a.h.movements(old, new, q)
    groups, reports, means = {}, {}, {}
    for i, (name, ids) in enumerate(positions.items()):
        if not len(ids):
            groups[name], means[name] = None, None
            reports[name] = {'status': 'absent', 'reason': 'no_actually_changed_examples', 'count': 0}
            continue
        gradient = np.array([1. + i, .2 * i, -1.], np.float32)
        targets = assigned[ids] if name == 'wrong_assigned' else true[ids]
        pre = np.zeros((len(ids), 10), np.float32)
        post = pre.copy()
        post[:, 0] = .01
        groups[name] = {key: torch.from_numpy(value.copy()) for key, value in {
            'training_positions': ids, 'source_ids': plan['train_ids'][ids], 'inputs': inputs[ids],
            'targets': targets, 'true_targets': true[ids], 'assigned_targets': assigned[ids],
            'pre_logits': pre, 'post_logits': post, 'mean_gradient': gradient}.items()}
        pre_stats, post_stats = a.h.classification(pre, targets), a.h.classification(post, targets)
        reports[name] = {'status': 'defined', 'count': len(ids),
            'true_label_counts': np.bincount(true[ids], minlength=10).tolist(),
            'geometry': a.mean_geometry(gradient, v, q), 'pre': pre_stats, 'post': post_stats,
            'finite_ce_improvement': pre_stats['ce'] - post_stats['ce'],
            'signed_first_order_utilities': {key: -float(gradient.astype(np.float64) @ delta) for key, delta in vectors.items()}}
        means[name] = gradient
    differences = {'rare_minus_majority': means['rare8'] - means['majority'],
                   'wrong_assigned_minus_corrected': None if clean else means['wrong_assigned'] - means['wrong_corrected']}
    anchor = {'block': (update - 101) // 50, 'position': (update - 101) % 50,
              'update': update, 'label': 'high', 'rare_count': 32, 'equal_counts_in_block': False}
    identity = {'schema': a.SCHEMA, 'seed': a.SEEDS[0], 'cell': 'clean' if clean else 'diffuse',
                'schedule': 'grouped', 'policy': 'native32', 'update': update, 'anchor': anchor,
                'warmup_state_sha256': a.h.tree_hash(warm)}
    payload = {**identity, 'parameter_order': ['toy'], 'before': before, 'after': after,
        'probe_parameter_state': 'pre_update_after_one_training_observer_update',
        'raw_training_gradient': torch.from_numpy(raw), 'applied_training_gradient': torch.from_numpy(delivered),
        'groups': groups, 'mean_differences': {key: None if value is None else torch.from_numpy(value.copy()) for key, value in differences.items()}}
    report = {**identity, 'before_sha256': a.h.tree_hash(before), 'after_sha256': a.h.tree_hash(after),
        'basis': basis, 'movement': movement, 'groups': reports,
        'mean_difference_geometry': {key: None if value is None else a.mean_geometry(value, v, q) for key, value in differences.items()},
        'difference_arithmetic_dtype': 'float32', 'probe_gradient_definition': 'autograd.grad of FP32 group mean cross_entropy',
        'probe_side_effect_checks': 'PASS', 'per_example_gradients_or_coherence': False}
    return payload, report, plan, assigned, inputs, warm, anchor, (a.SEEDS[0], identity['cell'], 'grouped')


class SamplerTests(unittest.TestCase):
    def test_exact_occurrence_multisets_and_shared_within_group_order(self):
        labels = np.array([0, 8, 3, 8, 5, 6], np.int64)
        original = np.resize(np.array([0, 0, 1, 2, 3, 4, 5], np.int64), (15, 4))
        result = a.rebuild_schedules(original, labels, 121, block_size=5)
        again = a.rebuild_schedules(original, labels, 121, block_size=5)
        for key in result:
            np.testing.assert_array_equal(result[key], again[key])
        for block in range(3):
            original_draws = original[block * 5:(block + 1) * 5].reshape(-1)
            ordered = original_draws[result['occurrence_permutations'][block]]
            for schedule in a.SCHEDULES:
                actual = result[schedule + '_batches'][block * 5:(block + 1) * 5]
                np.testing.assert_array_equal(np.sort(actual.reshape(-1)), np.sort(original_draws))
                rare, common = [], []
                for position, row in enumerate(actual):
                    unshuffled = row[np.argsort(result['within_batch_permutations'][block, position])]
                    count = result[schedule + '_rare_counts'][block * 5 + position]
                    rare.extend(unshuffled[:count])
                    common.extend(unshuffled[count:])
                    self.assertEqual(int(np.count_nonzero(labels[row] == 8)), count)
                np.testing.assert_array_equal(rare, ordered[labels[ordered] == 8])
                np.testing.assert_array_equal(common, ordered[labels[ordered] != 8])
            inter = result['interleaved_rare_counts'][block * 5:(block + 1) * 5]
            group = result['grouped_rare_counts'][block * 5:(block + 1) * 5]
            self.assertLessEqual(int(inter.max() - inter.min()), 1)
            self.assertLessEqual(int(np.count_nonzero((group > 0) & (group < 4))), 1)
            self.assertEqual(int(inter.sum()), int(group.sum()))

    def test_all_rare_all_common_and_one_rare_extremes(self):
        for rare_count in (0, 1, 19, 20):
            with self.subTest(rare_count=rare_count):
                labels = np.array([8] * rare_count + [3] * (20 - rare_count), np.int64)
                original = np.arange(20, dtype=np.int64).reshape(5, 4)
                result = a.rebuild_schedules(original, labels, 71, block_size=5)
                for schedule in a.SCHEDULES:
                    self.assertEqual(int(result[schedule + '_rare_counts'].sum()), rare_count)
                    anchors = a.selected_anchors(result[schedule + '_rare_counts'], blocks=(0,), block_size=5)
                    self.assertEqual(len({r['update'] for r in anchors}), 2)
                    self.assertTrue(all(r['equal_counts_in_block'] == (rare_count in (0, 20)) for r in anchors))

    def test_anchor_tie_order_and_absolute_update_index(self):
        counts = np.array([0, 2, 2, 0, 1, 3, 3, 3, 3, 3], np.int64)
        result = a.selected_anchors(counts, blocks=(0, 1), block_size=5)
        self.assertEqual([(r['update'], r['label']) for r in result], [(101, 'low'), (102, 'high'), (106, 'high'), (107, 'low')])
        self.assertEqual([r['equal_counts_in_block'] for r in result], [False, False, True, True])

    def test_invalid_sampler_inputs_rejected(self):
        labels = np.array([0, 8], np.int64)
        for original in (np.ones((6, 4), np.int64), np.full((5, 4), 2, np.int64), np.zeros((5, 4), np.float32)):
            with self.subTest(shape=original.shape), self.assertRaises(a.h.AuditError):
                a.rebuild_schedules(original, labels, 1, block_size=5)


class MeanProbeTests(unittest.TestCase):
    def test_randomized_stratified_membership_and_actual_wrong_mask(self):
        true = np.concatenate([np.full(6 if d != 8 else 50, d, np.int64) for d in range(10)])
        plan = {'train_true': true}
        assigned = true.copy()
        selected = np.where(true != 8)[0][::2]
        assigned[selected] = (assigned[selected] + 1) % 8
        result = a.probe_positions(plan, assigned, 131)
        self.assertEqual(len(result['majority']), 54)
        self.assertEqual(len(result['rare8']), 32)
        self.assertEqual(np.bincount(true[result['majority']], minlength=10).tolist(), [6] * 8 + [0, 6])
        self.assertTrue((true[result['rare8']] == 8).all())
        self.assertTrue((assigned[result['wrong_assigned']] != true[result['wrong_assigned']]).all())
        np.testing.assert_array_equal(result['wrong_assigned'], result['wrong_corrected'])
        empty = a.probe_positions(plan, true.copy(), 131)
        self.assertEqual(len(empty['wrong_assigned']), 0)
        np.testing.assert_array_equal(empty['majority'], result['majority'])
        np.testing.assert_array_equal(empty['rare8'], result['rare8'])

    def test_native_and_numerical_span_are_distinct(self):
        v = np.array([[2.], [0.]], np.float32)
        g = np.array([1., 1.], np.float32)
        result = a.mean_geometry(g, v)
        self.assertEqual(result['squared_norm'], 2.)
        self.assertEqual(result['native_squared_norm'], 16.)
        self.assertEqual(result['native_retention'], 8.)
        self.assertEqual(result['numerical_span_retention'], .5)
        self.assertNotIn('coherence', result)

    def test_identity_fallback_zero_and_mean_vector_shape(self):
        result = a.mean_geometry(np.array([1., -1.], np.float32), None)
        self.assertEqual(result['native_retention'], 1.)
        self.assertIsNone(result['numerical_span_retention'])
        self.assertTrue(result['identity_fallback'])
        zeros = a.mean_geometry(np.zeros(2, np.float32), None)
        self.assertIsNone(zeros['native_retention'])
        self.assertTrue(zeros['zero_denominator'])
        with self.assertRaises(a.h.AuditError):
            a.mean_geometry(np.zeros((3, 2), np.float32), None)


class SummaryTests(unittest.TestCase):
    def test_full_56_metric_summary_nulls_and_interaction(self):
        labels = np.arange(10, dtype=np.int64)
        rows = []
        for seed in a.SEEDS:
            for cell in a.CELLS:
                assigned = labels.copy()
                if cell == 'diffuse':
                    assigned[1:4] = 0
                for schedule in a.SCHEDULES:
                    for policy in a.POLICIES:
                        margin = 1. + a.SEEDS.index(seed) * .1
                        if schedule == 'grouped':
                            margin += .1 if policy == 'raw' else .3
                        logits = np.eye(10, dtype=np.float32) * np.float32(margin)
                        row = a.evaluation(2000, logits, logits, labels, assigned, labels)
                        warm = a.evaluation(100, np.zeros_like(logits), np.zeros_like(logits), labels, assigned, labels)
                        rows.append({'seed': seed, 'cell': cell, 'schedule': schedule, 'policy': policy,
                                     'endpoint': row, 'warmup': warm})
        result = a.aggregate(rows)
        self.assertEqual(len(a.endpoint(rows[0]['endpoint'])), 56)
        self.assertEqual(len(result['per_group']), 12)
        self.assertEqual(len(result['policy_contrasts']), 8)
        self.assertEqual(len(result['schedule_contrasts']), 6)
        self.assertEqual(len(result['schedule_policy_interactions']), 4)
        self.assertIsNone(result['change_from_warmup']['clean/grouped/native32']['train_wrong_ce'])
        self.assertEqual(result['schedule_policy_interactions']['clean/native32_minus_raw']['rare_ce']['negative_count'], 3)
        with self.assertRaises(a.h.AuditError):
            a.aggregate(rows[:-1])
        with self.assertRaises(a.h.AuditError):
            a.difference(None, .5)


class RecordedDiagnosticTests(unittest.TestCase):
    def test_complete_mean_schema_counters_and_nonendpoint_anchor(self):
        for clean in (False, True):
            for update in (101, 1987):
                with self.subTest(clean=clean, update=update), patch.dict(a.h.MODEL_SHAPES, {'toy': (3,)}, clear=True):
                    values = fake_diagnostic(clean, update)
                    result = a.verify_diagnostic(*values, a.h.Checks())
                    self.assertEqual(result['update'], update)
                    cleared = {**values[0]['after'], 'gradients': [None]}
                    self.assertEqual(result['after_cleared_sha256'], a.h.tree_hash(cleared))

    def test_wrong_schema_membership_anchor_and_difference_rejected(self):
        def change_target(p, r):
            p['groups']['majority']['targets'][0] = 8
        def change_shape(p, r):
            p['groups']['rare8']['mean_gradient'] = torch.zeros((32, 3))
        def change_difference(p, r):
            p['mean_differences']['rare_minus_majority'][0] += 1
        def change_anchor(p, r):
            p['anchor'] = {**p['anchor'], 'rare_count': 31}
        def change_counter(p, r):
            p['before']['tracker']['step_count'] = 100
            r['before_sha256'] = a.h.tree_hash(p['before'])
        with patch.dict(a.h.MODEL_SHAPES, {'toy': (3,)}, clear=True):
            pristine = fake_diagnostic()
            for mutation in (change_target, change_shape, change_difference, change_anchor, change_counter):
                values = copy.deepcopy(pristine)
                mutation(values[0], values[1])
                with self.subTest(mutation=mutation.__name__), self.assertRaises(a.h.AuditError):
                    a.verify_diagnostic(*values, a.h.Checks())

    def test_fixed_receipt_roster(self):
        names = a.fixed_artifact_names()
        self.assertEqual(len(names), 179)
        self.assertEqual(len(names) + 2 * 72, 323)
        self.assertEqual(sum(name.startswith('curve-') for name in names), 36)
        self.assertNotIn('complete.json', names)
        self.assertFalse(any(name.startswith('diagnostic-') for name in names))


if __name__ == '__main__':
    unittest.main()
