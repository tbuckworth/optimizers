"""Fabricated saved arrays and states only; no producer/model/observer replay."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('independent_observer_audit', ROOT/'scripts/audit_spectral_observer_pathway.py')
a = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(a)
torch.set_num_threads(1)


def observer(step=100):
    return {'rank': 32, 'decay': .99, 'warmup': 100, 'stable_update': True,
        'stabilize_every': 100, 'relative_eig_tol': 1e-8, 'absolute_eig_floor': 0.,
        'weighting': 'hard', 'normalize': 'none', 'adaptive': 'none', 'proj_k': None,
        'filter_strength': 1., 'energy_threshold': None, 'n_params': 3, 'step_count': step,
        'grad_mean': torch.tensor([.1, .1, .1]), 'V': torch.tensor([[1.], [0.], [0.]]),
        'S': torch.tensor([1.], dtype=torch.float64)}


def snapshot(step, theta, first, second, gradient=None, observer_step=None):
    tensor = lambda x: torch.from_numpy(np.asarray(x, np.float32).copy())
    group = {'params': [77], 'lr': .001, 'weight_decay': .01, 'betas': (.9, .999), 'eps': 1e-8,
        'foreach': False, 'fused': False, 'amsgrad': False, 'maximize': False,
        'capturable': False, 'differentiable': False}
    return {'schema': 'i9_neural_snapshot_v1', 'model_spec': {'input_dim': 784, 'width': 64, 'classes': 10},
        'model_state': {'toy': tensor(theta)}, 'gradients': [None if gradient is None else tensor(gradient)],
        'model_modes': [('', True)], 'optimizer': {'param_groups': [group],
            'state': {77: {'step': torch.tensor(float(step)), 'exp_avg': tensor(first), 'exp_avg_sq': tensor(second)}}},
        'tracker': None if observer_step is None else observer(observer_step),
        'rng': {'toy': torch.tensor([1, 2, 3], dtype=torch.uint8)}}


def fake_step(zero=False):
    old = np.array([1., -2., .5], np.float32)
    m0, v0 = np.array([.1, -.2, .05], np.float32), np.array([.01, .04, .0025], np.float32)
    g = np.zeros(3, np.float32) if zero else np.array([.2, 0., 0.], np.float32)
    parent = snapshot(100, old, m0, v0, observer_step=100)
    m1 = (.9*m0.astype(np.float64)+.1*g).astype(np.float32)
    v1 = (.999*v0.astype(np.float64)+.001*g.astype(np.float64)**2).astype(np.float32)
    adaptive = -.001*(m1.astype(np.float64)/(1-.9**101))/(np.sqrt(v1.astype(np.float64)/(1-.999**101))+1e-8)
    new = (old.astype(np.float64)*(1-.001*.01)+adaptive).astype(np.float32)
    observed = None if zero else observer(151)
    envelope = lambda state, step: {'schema': a.SCHEMA, 'model_adam': state,
        'observer': copy.deepcopy(observed), 'clocks': {'adam': step, 'observer': None if zero else 151}}
    before = envelope(snapshot(100, old, m0, v0, g), 100)
    after = envelope(snapshot(101, new, m1, v1, g), 101)
    return before, after, parent, g


def fake_oracles():
    return {'majority': np.array([1., 1., 0.], np.float32), 'rare8': np.array([-1., 0., 1.], np.float32)}


def fake_history():
    before, _, parent, _ = fake_step()
    data = {}
    report = {'observer_only_updates': 50, 'self_inclusion_updates': 1,
              'model_adam_preservation': 'PASS', 'nonaliasing': 'PASS'}
    common = np.array([.2, .3, .1], np.float32)
    for clock, phase, action_key in ((150, 'pre_inclusion', 'pre_action'), (151, 'post_inclusion', 'post_action')):
        env = copy.deepcopy(before)
        env['model_adam']['gradients'] = [None]
        env['observer'] = observer(clock)
        env['clocks']['observer'] = clock
        if clock == 151:
            env['observer']['V'] = torch.tensor([[0.], [1.], [0.]])
            env['observer']['grad_mean'] = torch.from_numpy((.99 * np.array([.1, .1, .1], np.float32).astype(np.float64) + .01 * common.astype(np.float64)).astype(np.float32))
        data['at'+str(clock)] = env
        v = env['observer']['V'].numpy()
        q, basis = a.h.span_basis(v)
        data[action_key] = torch.from_numpy(a.h.action(common, v))
        report['at'+str(clock)+'_sha256'] = a.h.tree_hash(env)
        report[phase] = {'basis': basis, 'geometry': {name: a.b.mean_geometry(g, v, q)
                            for name, g in {'block': common, **fake_oracles()}.items()}}
    report['self_inclusion_action_comparison'] = a.vector_pair(data['pre_action'], data['post_action'])
    return data, report, parent, common, fake_oracles()


def fake_readout(zero=False):
    before, after, parent, g = fake_step(zero)
    inherited = copy.deepcopy(before)
    inherited['model_adam']['gradients'] = [None]
    data = {'inherited': inherited, 'before': before, 'after': after,
            'delivered_gradient': torch.from_numpy(g), 'parameter_order': ['toy']}
    obs = before['observer']
    q, basis = (None, None) if obs is None else a.h.span_basis(obs['V'].numpy())
    old = before['model_adam']['model_state']['toy'].numpy()
    new = after['model_adam']['model_state']['toy'].numpy()
    vectors, movement = a.h.movements(old, new, q)
    report = {name+'_sha256': a.h.tree_hash(data[name]) for name in ('inherited', 'before', 'after')}
    report.update({'basis': basis, 'movement': movement, 'parent_sha256': a.h.tree_hash(parent),
        'oracle_signed_utilities': {name: {part: -float(grad.astype(np.float64) @ delta) for part, delta in vectors.items()}
                                   for name, grad in fake_oracles().items()},
        'gradient_semantics': 'explicit assigned gradient retained in before and after; inherited .grad saved separately',
        'physical_optimizer_steps': 1, 'new_observer_updates': 0, 'preservation': 'PASS', 'nonaliasing': 'PASS'})
    return data, report, parent, g, obs, fake_oracles()


class ArithmeticTests(unittest.TestCase):
    def test_equal_means_and_exact_symmetric_float32_input(self):
        x = np.resize(np.array([[1., .5, -1.], [-1., 1.5, 1.]], np.float32), (50, 3))
        y = np.tile(np.array([0., 1., 0.], np.float32), (50, 1))
        means, common, record = a.stream_means(x, y)
        np.testing.assert_array_equal(means[0], means[1])
        np.testing.assert_array_equal(common, np.array([0., 1., 0.], np.float32))
        self.assertEqual(record['difference_l2'], 0.)
        self.assertEqual(record['epsilon_multiplier'], 128)

    def test_mean_bound_is_not_outcome_dependent(self):
        x = np.ones((50, 3), np.float32)
        a.stream_means(x, x + np.float32(1e-6))
        for bad in (x+.01, np.full((50, 3), np.nan, np.float32), x[:49]):
            with self.subTest(shape=bad.shape), self.assertRaises(a.h.AuditError):
                a.stream_means(x, bad)
        _, common, record = a.stream_means(np.zeros_like(x), np.zeros_like(x))
        self.assertEqual(record['bound'], 1e-10)
        self.assertFalse(common.any())

    def test_separate_clocks_accepted_and_not_coerced(self):
        values = fake_step()
        digest = a.h.tree_hash(values[:3]), a.h.array_hash(values[3])
        result = a.verify_step(*values, 151, a.h.Checks(), {'toy': (3,)})
        self.assertLess(result['parameter_max_error_over_bound'], 1.)
        self.assertEqual((a.h.tree_hash(values[:3]), a.h.array_hash(values[3])), digest)
        history = {'schema': a.SCHEMA, 'model_adam': copy.deepcopy(values[0]['model_adam']),
                   'observer': observer(150), 'clocks': {'adam': 100, 'observer': 150}}
        a.envelope_arrays(history, 100, 150, True, {'toy': (3,)})
        for target in (100, 151):
            with self.assertRaises(a.h.AuditError):
                a.envelope_arrays(history, 100, target, True, {'toy': (3,)})

    def test_wrong_adam_counter_and_parent_mutation_rejected(self):
        before, after, parent, g = fake_step()
        before['model_adam']['optimizer']['state'][77]['step'] = torch.tensor(150.)
        with self.assertRaises(a.h.AuditError):
            a.verify_step(before, after, parent, g, 151, a.h.Checks(), {'toy': (3,)})
        before, after, parent, g = fake_step()
        before['model_adam']['model_state']['toy'][0] += .1
        with self.assertRaises(a.h.AuditError):
            a.verify_step(before, after, parent, g, 151, a.h.Checks(), {'toy': (3,)})

    def test_zero_is_real_gradient_and_can_move(self):
        before, after, parent, g = fake_step(True)
        a.verify_step(before, after, parent, g, None, a.h.Checks(), {'toy': (3,)})
        self.assertFalse(g.any())
        self.assertFalse(torch.equal(before['model_adam']['model_state']['toy'], after['model_adam']['model_state']['toy']))
        before['model_adam']['gradients'] = [None]
        with self.assertRaises(a.h.AuditError):
            a.verify_step(before, after, parent, g, None, a.h.Checks(), {'toy': (3,)})

    def test_native_projection_and_identity_fallback(self):
        g = np.array([.2, .3, .1], np.float32)
        obs = observer(151)
        a.native_action(g, obs, a.h.Checks(), np.array([.2, 0., 0.], np.float32))
        with self.assertRaises(a.h.AuditError):
            a.native_action(g, obs, a.h.Checks(), g)
        obs['V'] = obs['S'] = None
        np.testing.assert_array_equal(a.native_action(g, obs, a.h.Checks()), g.astype(np.float64))

    def test_usefulness_signs_and_undefined_wrong_group(self):
        labels = np.arange(10, dtype=np.int64)
        logits = np.eye(10, dtype=np.float32)
        before = a.b.evaluation(100, np.zeros_like(logits), np.zeros_like(logits), labels, labels, labels)
        after = a.b.evaluation(101, logits, logits, labels, labels, labels)
        result = a.absolute_utilities(before, after)
        self.assertGreater(result['rare_ce'], 0.)
        self.assertGreater(result['rare_accuracy'], 0.)
        self.assertIsNone(result['train_wrong_ce'])

    def test_complete_history_schema_and_state_preservation(self):
        values = fake_history()
        result = a.verify_history(*values, a.h.Checks(), {'toy': (3,)})
        self.assertEqual(set(result), {'pre_inclusion', 'post_inclusion', 'final_mean_recurrence'})
        values[0]['at151']['model_adam']['model_state']['toy'][0] += .1
        values[1]['at151_sha256'] = a.h.tree_hash(values[0]['at151'])
        with self.assertRaises(a.h.AuditError):
            a.verify_history(*values, a.h.Checks(), {'toy': (3,)})

    def test_wrong_final_common_mean_rejected_without_observer_replay(self):
        values = fake_history()
        values[0]['at151']['observer']['grad_mean'][0] += .001
        values[1]['at151_sha256'] = a.h.tree_hash(values[0]['at151'])
        with self.assertRaises(a.h.AuditError):
            a.verify_history(*values, a.h.Checks(), {'toy': (3,)})

    def test_complete_readout_schema_and_oracle_utility(self):
        for zero in (False, True):
            with self.subTest(zero=zero):
                values = fake_readout(zero)
                result = a.verify_readout(*values, a.h.Checks(), {'toy': (3,)})
                self.assertEqual(set(result), {'adam', 'basis', 'movement', 'oracle_signed_utilities'})
                values[1]['oracle_signed_utilities']['rare8']['total'] += .01
                with self.assertRaises(a.h.AuditError):
                    a.verify_readout(*values, a.h.Checks(), {'toy': (3,)})

    def test_full_all_seed_summary_and_control_deterioration(self):
        cases = []
        for seed in a.SEEDS:
            for cell in a.CELLS:
                cases.append({'seed': seed, 'cell': cell, 'actions': {
                    name: {'improvements': {'rare_ce': value, 'majority_macro_ce': -value,
                        'train_wrong_ce': None if cell == 'clean' else value}}
                    for name, value in zip(a.ACTIONS, (-.2, -.1, 0., .1))}})
        result = a.aggregate(cases)
        self.assertEqual(len(result['absolute_improvements']), 8)
        self.assertEqual(len(result['improvement_contrasts']), 12)
        primary = result['improvement_contrasts']['clean/native_grouped_minus_native_interleaved']['rare_ce']
        self.assertEqual(primary['positive_count'], 3)
        self.assertEqual(result['absolute_improvements']['clean/native_grouped']['rare_ce']['negative_count'], 3)
        self.assertIsNone(result['absolute_improvements']['clean/zero']['train_wrong_ce'])
        with self.assertRaises(a.h.AuditError):
            a.aggregate(cases[:-1])

    def test_exact_physical_and_artifact_rosters(self):
        self.assertEqual(len(a.physical_ids()), 21)
        self.assertEqual(sum(name.endswith('-zero') for name in a.physical_ids()), 3)
        self.assertEqual(len(a.artifact_names()), 122)
        self.assertEqual(sum(name.startswith('stream-') for name in a.artifact_names()), 12)
        self.assertNotIn('complete.json', a.artifact_names())

    def test_new_and_ancestor_io_are_distinct_and_cross_bound(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            new, prior = root/'new', root/'prior'
            new.mkdir()
            prior.mkdir()
            for path, label in ((new, 'new'), (prior, 'ancestor')):
                (path/'small.json').write_text(json.dumps({'label': label}))
            receipt = lambda path: {'path': 'small.json', 'size_bytes': (path/'small.json').stat().st_size,
                                    'sha256': a.h.sha256(path/'small.json')}
            bundle = object.__new__(a.Bundle)
            bundle.root, bundle.index = new, {'small.json': receipt(new)}
            ancestor = object.__new__(a.PriorInputs)
            ancestor.index, ancestor.used, ancestor.checks = {'small.json': receipt(prior)}, {}, a.h.Checks()
            with patch.object(a, 'PRIOR', prior):
                self.assertEqual(bundle.json('small.json'), {'label': 'new'})
                self.assertEqual(ancestor.json('small.json'), {'label': 'ancestor'})
                self.assertEqual(ancestor.used, ancestor.index)
                with self.assertRaises(a.h.AuditError):
                    bundle.bound(ancestor.index['small.json'], 'small.json')


if __name__ == '__main__':
    unittest.main()
