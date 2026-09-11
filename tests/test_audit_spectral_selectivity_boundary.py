"""Tiny fabricated inputs only. Never imports the scientific producer."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
import torch

PATH = Path(__file__).resolve().parents[1] / 'scripts/audit_spectral_selectivity_boundary.py'
SPEC = importlib.util.spec_from_file_location('independent_selectivity_audit', PATH)
a = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(a)
torch.set_num_threads(1)


def save_json(path, value):
    path.write_text(json.dumps(value, allow_nan=False))


def receipt(path):
    return {'path': path.name, 'size_bytes': path.stat().st_size, 'sha256': a.sha256(path)}


def tiny_plan(seed=123):
    return a.rebuild_plan(np.repeat(np.arange(10), 20), seed, majority_count=6, rare_count=2,
                          heldout_count=3, poison_count=8, warmup_steps=2, branch_steps=3, batch=4)


def metric_row(step=2000, clean=False, shift=0.):
    true = np.arange(10, dtype=np.int64)
    assigned = true.copy()
    if not clean:
        assigned[1:3] = 0
    logits = np.eye(10, dtype=np.float32) * np.float32(1 + shift)
    patched = logits.copy()
    patched[1:3, 0] += 2
    return a.metrics(step, logits, logits, patched, true, assigned, true)


def fake_snapshot(step, theta, first, second, gradients=None):
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
        'model_state': {'toy': tensor(theta)}, 'gradients': [None if gradients is None else tensor(gradients)],
        'model_modes': [('', True)], 'optimizer': {'param_groups': [group],
            'state': {77: {'step': torch.tensor(float(step)), 'exp_avg': tensor(first), 'exp_avg_sq': tensor(second)}}},
        'tracker': tracker, 'rng': {'toy': torch.tensor([1, 2, 3], dtype=torch.uint8)}}


def fake_diagnostic(clean=False, update=101):
    """Recorded three-coordinate tensors, assembled by algebra; no model/Adam object."""
    old = np.array([1., -2., .5], np.float32)
    m0, v0 = np.array([.1, -.2, .05], np.float32), np.array([.01, .04, .0025], np.float32)
    raw, delivered = np.array([.2, .3, .1], np.float32), np.array([.2, 0., 0.], np.float32)
    warm = fake_snapshot(100, old, m0, v0)
    before = fake_snapshot(update - 1, old, m0, v0, delivered)
    before['tracker']['step_count'] = update
    m1 = (.9 * m0.astype(np.float64) + .1 * delivered).astype(np.float32)
    v1 = (.999 * v0.astype(np.float64) + .001 * delivered.astype(np.float64)**2).astype(np.float32)
    change = -.001 * (m1.astype(np.float64) / (1 - .9**update)) / (np.sqrt(v1.astype(np.float64) / (1 - .999**update)) + 1e-8)
    new = (old.astype(np.float64) * (1 - .001 * .01) + change).astype(np.float32)
    after = fake_snapshot(update, new, m1, v1, delivered)
    true = np.array([3, 3, 8, 8, 1, 2], np.int64)
    assigned = true.copy()
    if not clean:
        assigned[-2:] = 0
    plan = {'train_true': true, 'train_ids': np.arange(10, 16, dtype=np.int64)}
    clean_x = np.zeros((6, 784), np.float32)
    cell_x = a.patched_images(clean_x, np.array([False] * 4 + [not clean] * 2))
    definitions = {'common3': np.array([0, 1]), 'rare8': np.array([2, 3]),
                   'wrong_assigned': np.array([], np.int64) if clean else np.array([4, 5]),
                   'wrong_corrected': np.array([], np.int64) if clean else np.array([4, 5])}
    v = before['tracker']['V'].numpy()
    q, basis = a.span_basis(v)
    vectors, movement = a.movements(old, new, q)
    groups, summaries = {}, {}
    for i, (name, selected) in enumerate(definitions.items()):
        if not len(selected):
            groups[name] = None
            summaries[name] = {'status': 'absent', 'reason': 'no_actually_changed_examples', 'count': 0}
            continue
        gradients = np.array([[1 + i, 0, 1], [0, 2 + i, 1]], np.float32)
        targets = assigned[selected] if name == 'wrong_assigned' else true[selected]
        pre_logits, post_logits = np.zeros((2, 10), np.float32), np.zeros((2, 10), np.float32)
        post_logits[:, 0] = .01
        groups[name] = {k: torch.from_numpy(x.copy()) for k, x in {
            'training_positions': selected.astype(np.int64), 'source_ids': plan['train_ids'][selected],
            'inputs': (cell_x if name.startswith('wrong_') else clean_x)[selected],
            'targets': targets, 'true_targets': true[selected], 'pre_logits': pre_logits,
            'post_logits': post_logits, 'per_example_gradients': gradients}.items()}
        pre, post = a.classification(pre_logits, targets), a.classification(post_logits, targets)
        summaries[name] = {'status': 'defined', 'true_label_counts': np.bincount(true[selected], minlength=10).tolist(),
            'geometry': a.geometry(gradients, v), 'pre': pre, 'post': post,
            'finite_ce_improvement': pre['ce'] - post['ce'],
            'signed_first_order_utilities': {k: -float(gradients.astype(np.float64).mean(0) @ d) for k, d in vectors.items()}}
    identity = {'schema': a.SCHEMA, 'seed': a.SEEDS[0], 'cell': 'clean' if clean else 'shared',
                'policy': 'native32', 'update': update, 'warmup_state_sha256': a.tree_hash(warm)}
    payload = {**identity, 'parameter_order': ['toy'], 'before': before, 'after': after,
        'probe_parameter_state': 'pre_update_after_one_training_observer_update',
        'raw_training_gradient': torch.from_numpy(raw), 'applied_training_gradient': torch.from_numpy(delivered),
        'groups': groups}
    report = {**identity, 'before_sha256': a.tree_hash(before), 'after_sha256': a.tree_hash(after),
        'basis': basis, 'movement': movement, 'groups': summaries,
        'wrong_minus_true_gradient_geometry': None if clean else a.geometry(
            groups['wrong_assigned']['per_example_gradients'].numpy() - groups['wrong_corrected']['per_example_gradients'].numpy(), v),
        'residual_arithmetic_dtype': 'float32', 'probe_side_effect_checks': 'PASS'}
    return payload, report, plan, clean_x, cell_x, assigned, warm


class IOTests(unittest.TestCase):
    def test_scalar_tensor_digest_preserves_rank_zero(self):
        x = torch.tensor(2., dtype=torch.float32)
        descriptor = ['tensor', 'torch.float32', [], hashlib.sha256(np.float32(2).tobytes()).hexdigest()]
        encoded = json.dumps(descriptor, separators=(',', ':')).encode('ascii')
        expected = hashlib.sha256(b'i9_neural_tree_v1\n' + encoded).hexdigest()
        self.assertEqual(a.tree_hash(x), expected)
        self.assertNotEqual(a.tree_hash(x), a.tree_hash(x.reshape(1)))

    def test_receipt_rejects_wrong_hash_size_optional_fields_and_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / 'data.json'
            save_json(target, {'small': 1})
            valid = receipt(target)
            self.assertEqual(a.verify_receipt(root, valid), target)
            for bad in ({**valid, 'sha256': '0' * 64}, {**valid, 'size_bytes': 0},
                        {**valid, 'extra': 1}, {**valid, 'path': '../data.json'},
                        {**valid, 'path': str(target)}):
                with self.subTest(bad=bad), self.assertRaises(a.AuditError):
                    a.verify_receipt(root, bad)
            link = root / 'link.json'
            link.symlink_to(target)
            with self.assertRaises(a.AuditError):
                a.verify_receipt(root, {**valid, 'path': 'link.json'})

    def test_safe_npz_and_json_rejections(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            p = root / 'safe.npz'
            np.savez(p, x=np.arange(6, dtype=np.float32).reshape(2, 3))
            np.testing.assert_array_equal(a.load_npz(p)['x'], np.arange(6).reshape(2, 3))
            np.savez(root / 'object.npz', x=np.array([object()], dtype=object))
            with self.assertRaises(ValueError):
                a.load_npz(root / 'object.npz')
            with zipfile.ZipFile(root / 'escape.npz', 'w') as z:
                z.writestr('../x.npy', b'not loaded')
            with self.assertRaises(a.AuditError):
                a.load_npz(root / 'escape.npz')
            for i, text in enumerate(('{"x":1,"x":2}', '{"x":NaN}')):
                (root / f'bad{i}.json').write_text(text)
                with self.assertRaises(a.AuditError):
                    a.read_json(root / f'bad{i}.json')

    def test_restricted_cpu_tensor_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / 'fabricated.pt'
            value = {'scalar': torch.tensor(4.), 'a': torch.arange(5), 'tuple': (1, None),
                'rng': {'python': (3, (1, 2, 3), None),
                        'numpy': {'algorithm': 'MT19937', 'state': torch.tensor([1, 2, 3], dtype=torch.uint32),
                                  'position': 2, 'has_gauss': 0, 'cached_gaussian': 0.},
                        'torch_cpu': torch.tensor([1, 2], dtype=torch.uint8),
                        'torch_cuda': [torch.tensor([3, 4], dtype=torch.uint8)]}}
            torch.save(value, p)
            decoded = a.cpu_tensor_file(p)
            self.assertEqual(a.tree_hash(decoded), a.tree_hash(value))
            self.assertEqual(str(decoded['a'].device), 'cpu')

    def test_boolean_equality_is_distinct_from_integer_equality(self):
        checks = a.Checks()
        checks.equal({'present': True, 'absent': False, 'n': 1}, {'present': True, 'absent': False, 'n': 1}, 'exact')
        for observed, expected in ((1, True), (True, 1), (0, False), (False, 0)):
            with self.subTest(observed=observed, expected=expected), self.assertRaises(a.AuditError):
                checks.equal(observed, expected, 'typed exact')

    def test_exact_artifact_roster(self):
        names = a.expected_artifacts()
        self.assertEqual(len(names), 245)
        self.assertEqual(sum(n.startswith('diagnostic-') for n in names), 72)
        self.assertNotIn('complete.json', names)
        self.assertNotIn('failed.json', names)


class PlanAndMetricTests(unittest.TestCase):
    def test_seeded_plan_pairing_rare_exclusion_and_sham_quotas(self):
        plan, allocations = tiny_plan()
        repeat, again = tiny_plan()
        for key in plan:
            np.testing.assert_array_equal(plan[key], repeat[key])
        self.assertEqual(allocations, again)
        true = plan['train_true']
        self.assertEqual(np.bincount(true, minlength=10).tolist(), [6] * 8 + [2, 6])
        self.assertEqual(len(np.intersect1d(plan['train_ids'], plan['heldout_ids'])), 0)
        self.assertFalse((true[plan['warmup_batches']] == 8).any())
        self.assertFalse(plan['diffuse_selected'][true == 8].any())
        self.assertTrue(np.all(plan['diffuse_replacement_labels'][true != 8] != 8))
        for cell in a.CELLS:
            target, mask, counts = a.cell_data(plan, cell)
            np.testing.assert_array_equal(target[true == 8], true[true == 8])
            self.assertFalse(mask[true == 8].any())
            self.assertEqual(np.asarray(counts['contingency_counts']).sum(), len(true))
        for d in (1, 2, 3, 4, 5, 6, 7, 9):
            k = int(plan['poison_mask'][true == d].sum())
            self.assertEqual(int(plan['sham_patch_mask'][true == d].sum()), k)
            for r in (r for r in allocations if r['true_digit'] == d):
                self.assertLess(abs(r['rounding_deviation']), 1.)
        other, _ = tiny_plan(124)
        self.assertFalse(np.array_equal(plan['train_ids'], other['train_ids']))

    def test_exact_cpu_normalization_and_patch(self):
        pixels = np.resize(np.arange(256, dtype=np.uint8), (2, 784))
        x = pixels.astype(np.float32) / np.float32(255)
        y = a.patched_images(x, np.array([True, False]))
        self.assertEqual(y.dtype, np.float32)
        np.testing.assert_array_equal(y[1], x[1])
        self.assertTrue((y[0].reshape(28, 28)[:3, :3] == 1).all())
        mask = np.ones((28, 28), bool)
        mask[:3, :3] = False
        np.testing.assert_array_equal(y[0].reshape(28, 28)[mask], x[0].reshape(28, 28)[mask])
        self.assertNotEqual(a.array_hash(x), a.array_hash(y))

    def test_logsumexp_accuracy_empty_and_prediction_independent_cue_masks(self):
        labels = np.arange(10, dtype=np.int64)
        logits = np.zeros((10, 10), np.float32)
        out = a.classification(logits, labels)
        self.assertEqual(out['correct'], 1)
        self.assertAlmostEqual(out['ce'], np.log(10))
        shifted = logits + np.float32(1000)
        self.assertAlmostEqual(a.classification(shifted, labels)['ce'], out['ce'])
        empty = a.classification(logits, labels, labels < 0)
        self.assertEqual(empty, {'count': 0, 'correct': 0, 'ce_sum': 0., 'accuracy': None, 'ce': None})
        row = metric_row()
        self.assertEqual(row['cue']['majority_nonzero']['count'], 8)
        self.assertEqual(row['cue']['all_nonzero']['count'], 9)
        self.assertEqual(row['cue']['rare']['count'], 1)
        self.assertEqual(row['cue']['majority_nonzero']['patch_excess'], .25)

    def test_group_macro_not_micro_under_imbalance(self):
        y = np.array([0] * 10 + list(range(1, 10)), np.int64)
        logits = np.zeros((len(y), 10), np.float32)
        stats = a.grouped(logits, y)
        self.assertAlmostEqual(stats['balanced_total']['accuracy'], .1)
        self.assertAlmostEqual(stats['majority_macro']['accuracy'], 1 / 9)
        self.assertAlmostEqual(stats['micro']['accuracy'], 10 / 19)

    def test_full_aggregate_schema_and_all_seed_contrasts(self):
        rows = []
        for seed in a.SEEDS:
            for cell in a.CELLS:
                for policy in a.POLICIES:
                    shift = .1 * a.SEEDS.index(seed) + .2 * a.POLICIES.index(policy)
                    rows.append({'seed': seed, 'cell': cell, 'policy': policy,
                                 'endpoint': metric_row(clean=cell == 'clean', shift=shift),
                                 'warmup': metric_row(step=100, clean=cell == 'clean')})
        result = a.aggregate(rows)
        self.assertEqual(len(a.endpoint(rows[0]['endpoint'])), 86)
        self.assertEqual(set(result['change_from_warmup']['clean/raw']), set(a.endpoint(rows[0]['endpoint'])))
        self.assertIsNone(result['policy_contrasts']['clean/native32_minus_raw']['train_wrong_ce'])
        self.assertIsNone(result['change_from_warmup']['clean/raw']['train_wrong_accuracy'])
        self.assertEqual(result['policy_contrasts']['shared/native32_minus_raw']['rare_accuracy']['values'], [0., 0., 0.])
        self.assertEqual(result['policy_contrasts']['shared/native32_minus_raw']['rare_ce']['negative_count'], 3)
        self.assertEqual(result['cue_interactions']['native32_minus_raw/majority_nonzero_patch_excess']['mean'], 0.)
        for altered in (rows[:-1], rows[:-1] + [rows[0]]):
            with self.assertRaises(a.AuditError):
                a.aggregate(altered)

    def test_sample_sd_se_and_null_semantics(self):
        result = a.three_seed([-1., 0., 1.])
        self.assertEqual(result['sample_sd'], 1.)
        self.assertAlmostEqual(result['sample_se'], 1 / np.sqrt(3))
        self.assertEqual([result[k] for k in ('positive_count', 'negative_count', 'zero_count')], [1, 1, 1])
        self.assertIsNone(a.three_seed([None] * 3))
        with self.assertRaises(a.AuditError):
            a.three_seed([1., None, 2.])


class DiagnosticTests(unittest.TestCase):
    def test_nonorthogonal_native_action_not_ideal_projector_and_zero_rules(self):
        g = np.array([[1, 0], [0, 2]], np.float32)
        v = np.array([[2], [0]], np.float32)
        geo = a.geometry(g, v)
        self.assertEqual(geo['individual_native_retention'], [16., 0.])
        self.assertAlmostEqual(geo['energy_weighted_native_retention'], 16 / 5)
        self.assertAlmostEqual(geo['coherence'], .5)
        self.assertAlmostEqual(geo['native_mean_retention'], 16 / 5)
        identity = a.geometry(g, None)
        self.assertEqual(identity['individual_native_retention'], [1., 1.])
        zeros = a.geometry(np.zeros((2, 2), np.float32), v)
        self.assertIsNone(zeros['coherence'])
        self.assertEqual(zeros['individual_native_retention'], [None, None])
        opposite = a.geometry(np.array([[1, 0], [-1, 0]], np.float32), v)
        self.assertEqual(opposite['coherence'], 0.)
        self.assertIsNone(opposite['native_mean_retention'])

    def test_span_rank_and_rounded_decay_components(self):
        v = np.array([[1, 2], [0, 0], [0, 0]], np.float32)
        q, record = a.span_basis(v)
        self.assertEqual(record['rank'], 1)
        old, new = np.array([1., -2., .5], np.float32), np.array([1.1, -1.8, .6], np.float32)
        vectors, summary = a.movements(old, new, q)
        np.testing.assert_array_equal(vectors['decay'], (old * np.float32(.99999)).astype(np.float64) - old.astype(np.float64))
        np.testing.assert_allclose(vectors['total'], vectors['decay'] + vectors['adaptive'], rtol=0, atol=1e-16)
        self.assertGreater(summary['total']['outside_span_fraction'], 0.)

    def test_adam_identity_and_wrong_moment_or_parameter_rejected(self):
        payload, *_ = fake_diagnostic()
        b, c = payload['before'], payload['after']
        old, m0, v0 = a.snapshot_arrays(b, 100, True, {'toy': (3,)}, gradients_present=True)
        cleared = {**c, 'gradients': [None]}
        new, m1, v1 = a.snapshot_arrays(cleared, 101, True, {'toy': (3,)})
        g = payload['applied_training_gradient'].numpy()
        a.verify_adam(old, new, m0, v0, m1, v1, g, 101, a.Checks())
        for position in (1, 4, 5):
            args = [old.copy(), new.copy(), m0.copy(), v0.copy(), m1.copy(), v1.copy(), g.copy()]
            args[position][0] += .01
            with self.assertRaises(a.AuditError):
                a.verify_adam(*args, 101, a.Checks())

    def test_full_saved_diagnostic_positive_and_mutations(self):
        def verify(parts):
            payload, report, plan, clean_x, cell_x, assigned, warm = parts
            with patch.object(a, 'MODEL_SHAPES', {'toy': (3,)}):
                return a.verify_diagnostic(payload, report, plan, clean_x, cell_x, assigned,
                    (a.SEEDS[0], payload['cell'], payload['update']), warm, a.Checks())
        for clean, update in ((False, 101), (True, 101), (False, 2000)):
            parts = fake_diagnostic(clean, update)
            verified = verify(parts)
            self.assertEqual(verified['update'], update)
            cleared = {**parts[0]['after'], 'gradients': [None]}
            self.assertEqual(verified['after_without_gradients_sha256'], a.tree_hash(cleared))
        base = fake_diagnostic()
        for mode in ('wrong_ids', 'rare_inputs', 'wrong_null', 'bad_geometry', 'probe_timing', 'wrong_parameters'):
            parts = copy.deepcopy(base)
            payload, report = parts[:2]
            if mode == 'wrong_ids':
                payload['groups']['wrong_assigned']['source_ids'][0] += 1
            elif mode == 'rare_inputs':
                payload['groups']['rare8']['inputs'][0, 0] = 1
            elif mode == 'wrong_null':
                payload['groups']['wrong_assigned'] = None
            elif mode == 'bad_geometry':
                report['groups']['common3']['geometry']['coherence'] += .01
            elif mode == 'probe_timing':
                payload['probe_parameter_state'] = 'after_update'
            else:
                payload['parameter_order'] = ['wrong']
            with self.subTest(mode=mode), self.assertRaises(a.AuditError):
                verify(parts)

    def test_adam_parameter_order_counter_and_dtype_admission(self):
        state = fake_snapshot(100, [1, 2, 3], [.1] * 3, [.2] * 3)
        a.snapshot_arrays(state, 100, True, {'toy': (3,)})
        for key in ('counter', 'param_id', 'dtype'):
            bad = copy.deepcopy(state)
            if key == 'counter':
                bad['optimizer']['state'][77]['step'] = torch.tensor(99.)
            elif key == 'param_id':
                bad['optimizer']['param_groups'][0]['params'] = [78]
            else:
                bad['model_state']['toy'] = bad['model_state']['toy'].double()
            with self.subTest(key=key), self.assertRaises(a.AuditError):
                a.snapshot_arrays(bad, 100, True, {'toy': (3,)})

    def test_norm_history_and_zero_gradient_semantics(self):
        rows = []
        for step in range(101, 2001):
            raw, target = (0., 0.) if step == 101 else (2., 1.)
            rows.append({'step': step, 'training_batch_ce': 1., 'raw_norm': raw, 'native_norm': target,
                'applied_norm': target, 'observer_step': step, 'basis_rank': 1,
                'native_identity_fallback': False, 'norm_law': {'raw_norm': raw, 'target_norm': target,
                    'delivered_norm': target, 'scale': .5 if raw else None, 'relative_postcast_mismatch': 0.,
                    'relative_tolerance': 10 * a.EPS32, 'zero_raw': not bool(raw), 'zero_target': not bool(target)}})
        a.verify_actions(rows, 'norm_raw', a.Checks())
        rows[1]['applied_norm'] += .01
        with self.assertRaises(a.AuditError):
            a.verify_actions(rows, 'norm_raw', a.Checks())


if __name__ == '__main__':
    unittest.main()
