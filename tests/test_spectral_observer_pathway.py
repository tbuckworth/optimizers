"""Fabricated CPU-only observer-pathway tests. Never read scientific parents."""
import copy
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments import spectral_observer_pathway as o


class ArithmeticTests(unittest.TestCase):
    def test_symmetric_common_input_numpy_fp64_mean_bytes_and_exact_bound(self):
        left = torch.arange(50*7, dtype=torch.float32).reshape(50, 7)/256
        right = left.flip(0)
        star, means, report = o.common_input(left, right)
        self.assertTrue(report['passed'])
        expected_means = [x.numpy().astype(np.float64).mean(axis=0, dtype=np.float64) for x in (left, right)]
        expected_star = ((expected_means[0]+expected_means[1])/np.float64(2)).astype(np.float32)
        self.assertEqual(star.numpy().tobytes(), expected_star.tobytes())
        self.assertEqual(star.dtype, torch.float32)
        for actual, expected in zip(means, expected_means):
            self.assertEqual(actual.dtype, torch.float64)
            self.assertEqual(actual.numpy().tobytes(), expected.tobytes())
        expected_scale = float((left.double().norm(dim=1).sum()+right.double().norm(dim=1).sum())/100)
        self.assertEqual(report['average_batch_gradient_norm'], expected_scale)
        self.assertEqual(report['bound'], 128*torch.finfo(torch.float32).eps*expected_scale+1e-10)
        self.assertEqual(report['mean_difference_norm'], 0.)
        self.assertEqual(report['g_star_sha256'], o.core.tree_digest(star))

    def test_mean_failure_is_reported_without_relaxation_or_hiding_streams(self):
        left, right = torch.zeros(50, 3), torch.ones(50, 3)
        star, means, report = o.common_input(left, right)
        self.assertFalse(report['passed'])
        self.assertGreater(report['mean_difference_norm'], report['bound'])
        self.assertTrue(torch.equal(star, torch.full((3,), .5)))
        with self.assertRaises(RuntimeError):
            o.common_input(left[:49], right[:49])
        right[0, 0] = float('nan')
        with self.assertRaises(RuntimeError):
            o.common_input(left, right)

    def test_alias_detection_catches_views_and_accepts_deep_clones(self):
        value = torch.arange(8)
        with self.assertRaisesRegex(RuntimeError, 'aliases'):
            o.require_nonaliasing({'x': value}, {'different_view': value[2:]})
        o.require_nonaliasing({'x': value}, {'x': value.clone()}, None)
        self.assertEqual(len(list(o.tensor_leaves({'x': [value], 'a': None}))), 1)

    def test_zero_pair_cosine_is_undefined(self):
        record = o.vector_pair(torch.zeros(3), torch.ones(3))
        self.assertIsNone(record['cosine'])
        self.assertTrue(record['zero_norm'])
        self.assertAlmostEqual(record['difference_norm'], math.sqrt(3))
        self.assertAlmostEqual(o.vector_pair(torch.ones(3), -torch.ones(3))['cosine'], -1.)


class FrozenParentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.hidden = patch.object(torch.cuda, 'is_available', return_value=False)
        cls.hidden.start()
        cls.x = torch.arange(24, dtype=torch.float32).reshape(6, 4)/24
        cls.y = torch.tensor([0, 1, 2, 3, 8, 9])
        model = o.core.make_model(411, 'cpu', input_dim=4, width=3)
        optimizer = o.core.make_optimizer(model)
        tracker = o.core.make_tracker(model, optimizer)
        for step in range(1, 101):
            index = [(step+i) % 6 for i in range(3)]
            o.batch.training_update(model, optimizer, tracker, cls.x[index], cls.y[index], 'raw', step,
                                    int((cls.y[index] == 8).sum()))
        cls.parent = o.core.snapshot(model, optimizer, tracker)

    @classmethod
    def tearDownClass(cls):
        cls.hidden.stop()

    def oracles(self):
        model, optimizer, tracker = o.copied_parent(self.parent, 'cpu')
        gradients = {}
        for name, ids in [('majority', [0, 1, 2]), ('rare8', [4])]:
            gradient, _ = o.protected_gradient(model, optimizer, tracker, self.x[ids], self.y[ids], self.parent)
            gradients[name] = gradient
        return gradients

    def history(self):
        gradients = self.oracles()
        base = (gradients['majority']+gradients['rare8'])/2
        streams = torch.stack([base+(t % 2*2-1)*gradients['rare8']/4 for t in range(50)])
        star, _, report = o.common_input(streams, streams.flip(0))
        self.assertTrue(report['passed'])
        data, report = o.observer_history(self.parent, streams, star, gradients, 'cpu')
        return data, report, star, gradients

    def test_exact_parent_copy_and_writable_sibling_independence(self):
        a = o.copied_parent(self.parent, 'cpu')
        b = o.copied_parent(self.parent, 'cpu')
        o.require_nonaliasing(self.parent, o.live_tensors(*a), o.live_tensors(*b))
        parent_hash = o.core.tree_digest(self.parent)
        other_hash = o.core.tree_digest(o.core.snapshot(*b))
        with torch.no_grad():
            next(a[0].parameters()).add_(1)
        self.assertEqual(parent_hash, o.core.tree_digest(self.parent))
        self.assertEqual(other_hash, o.core.tree_digest(o.core.snapshot(*b)))

    def test_gradient_preserves_inherited_grads_modes_rng_and_all_clocks(self):
        model, optimizer, observer = o.copied_parent(self.parent, 'cpu')
        model.eval()
        o.core.set_grad(model, torch.ones_like(o.core.flat_params(model)))
        before = o.envelope(model, optimizer, observer)
        gradient, report = o.protected_gradient(model, optimizer, observer, self.x, self.y, self.parent)
        self.assertTrue(o.core.equal_tree(before, o.envelope(model, optimizer, observer)))
        self.assertEqual(report['before_sha256'], report['after_sha256'])
        self.assertEqual(report['preservation'], 'PASS')
        self.assertEqual(gradient.dtype, torch.float32)
        self.assertEqual(before['clocks'], {'adam': 100, 'observer': 100})

    def test_observer_only_150_then_self_inclusion_151_without_adam_updates(self):
        data, report, star, gradients = self.history()
        self.assertEqual(data['at150']['clocks'], {'adam': 100, 'observer': 150})
        self.assertEqual(data['at151']['clocks'], {'adam': 100, 'observer': 151})
        self.assertTrue(o.core.equal_tree(data['at150']['model_adam'], data['at151']['model_adam']))
        self.assertIsNone(data['at150']['model_adam']['tracker'])
        self.assertEqual(report['observer_only_updates'], 50)
        self.assertEqual(report['self_inclusion_updates'], 1)
        self.assertEqual(report['at150_sha256'], o.core.tree_digest(data['at150']))
        self.assertEqual(report['at151_sha256'], o.core.tree_digest(data['at151']))
        for phase, state in [('pre', data['at150']), ('post', data['at151'])]:
            v = state['observer']['V']
            expected = star if v is None else v @ (v.T @ star)
            torch.testing.assert_close(data[phase+'_action'], expected, rtol=0, atol=0)
        self.assertEqual(set(report['pre_inclusion']['geometry']), {'block', 'majority', 'rare8'})
        o.require_nonaliasing(data['at150'], data['at151'], self.parent)

    def test_native_readout_adam101_observer151_and_explicit_action_assignment(self):
        history, _, star, oracles = self.history()
        data, logits, report = o.readout(self.parent, history['post_action'], history['at151']['observer'],
                                         oracles, self.x, self.x, 'cpu')
        self.assertEqual(data['inherited']['clocks'], {'adam': 100, 'observer': 151})
        self.assertEqual(data['before']['clocks'], {'adam': 100, 'observer': 151})
        self.assertEqual(data['after']['clocks'], {'adam': 101, 'observer': 151})
        self.assertTrue(all(value is None for value in data['inherited']['model_adam']['gradients']))
        before_gradient = torch.cat([g.reshape(-1) for g in data['before']['model_adam']['gradients']])
        self.assertTrue(torch.equal(before_gradient, history['post_action']))
        self.assertTrue(o.core.equal_tree(data['before']['observer'], data['after']['observer']))
        self.assertEqual(report['physical_optimizer_steps'], 1)
        self.assertEqual(report['new_observer_updates'], 0)
        np.testing.assert_array_equal(logits['train'], logits['heldout'])
        for name, gradient in oracles.items():
            old = torch.cat([v.reshape(-1) for v in data['before']['model_adam']['model_state'].values()]).double()
            new = torch.cat([v.reshape(-1) for v in data['after']['model_adam']['model_state'].values()]).double()
            self.assertEqual(report['oracle_signed_utilities'][name]['total'], -float(gradient.double() @ (new-old)))
        json.dumps(report, allow_nan=False)

    def test_shared_zero_reference_is_real_adam_step_not_none(self):
        oracles = self.oracles()
        zero = torch.zeros_like(oracles['rare8'])
        data, logits, report = o.readout(self.parent, zero, None, oracles, self.x, self.x, 'cpu')
        self.assertEqual(data['before']['clocks'], {'adam': 100, 'observer': None})
        self.assertEqual(data['after']['clocks'], {'adam': 101, 'observer': None})
        self.assertFalse(o.core.equal_tree(data['before']['model_adam']['model_state'], data['after']['model_adam']['model_state']))
        for gradient in data['after']['model_adam']['gradients']:
            self.assertTrue(torch.equal(gradient, torch.zeros_like(gradient)))
        for key, state in data['before']['model_adam']['optimizer']['state'].items():
            after = data['after']['model_adam']['optimizer']['state'][key]
            torch.testing.assert_close(after['exp_avg'], .9*state['exp_avg'], rtol=2e-7, atol=1e-10)
            torch.testing.assert_close(after['exp_avg_sq'], .999*state['exp_avg_sq'], rtol=2e-7, atol=1e-10)
        self.assertGreater(report['movement']['total']['norm'], 0)


class ReadoutSummaryTests(unittest.TestCase):
    def test_improvement_orientation_and_undefined_training_fit(self):
        labels = np.arange(10)
        logits = np.eye(10, dtype=np.float32)
        before = o.batch.evaluation_row(100, logits, logits, labels, labels, labels)
        after = copy.deepcopy(before)
        after['heldout']['rare']['ce'] -= .2
        after['heldout']['rare']['accuracy'] -= .1
        improvement = o.action_improvement(before, after)
        self.assertAlmostEqual(improvement['rare_ce'], .2)
        self.assertAlmostEqual(improvement['rare_accuracy'], -.1)
        self.assertIsNone(improvement['train_wrong_ce'])

    def test_all_seed_utilities_and_primary_contrast_are_not_raw_loss_changes(self):
        cases = []
        for index, seed in enumerate(o.SEEDS):
            for cell in o.CELLS:
                actions = {}
                for action in o.ACTIONS:
                    rare = .1+index*.01 if action == 'native_grouped' else -.2
                    actions[action] = {'improvements': {'rare_ce': rare, 'majority_macro_ce': -.01,
                                                         'train_wrong_ce': None if cell == 'clean' else .05}}
                cases.append({'seed': seed, 'cell': cell, 'actions': actions})
        summary = o.summarize_cases(cases)
        result = summary['improvement_contrasts']['clean/native_grouped_minus_native_interleaved']['rare_ce']
        np.testing.assert_allclose(result['values'], [.3, .31, .32], atol=1e-15)
        self.assertAlmostEqual(result['mean'], .31)
        self.assertAlmostEqual(result['sample_sd'], .01)
        self.assertAlmostEqual(result['sample_se'], .01/math.sqrt(3))
        self.assertEqual(result['positive_count'], 3)
        self.assertEqual(summary['physical_readouts'], 21)
        self.assertEqual(summary['logical_case_readouts'], 24)
        self.assertIsNone(summary['absolute_improvements']['clean/zero']['train_wrong_ce'])
        with self.assertRaises(RuntimeError):
            o.summarize_cases(cases[:-1])


class StorageAndAdmissionTests(unittest.TestCase):
    def test_new_bounds_and_conservative_inventory(self):
        inventory = o.byte_inventory()
        self.assertLess(inventory['total_upper_bytes']+o.FOOTER_RESERVE, 1024**3)
        self.assertEqual(inventory['component_upper_bytes']['600_stream_gradients'], 600*50890*4)
        self.assertEqual(o.COOPERATIVE_SECONDS, 480)
        self.assertEqual(o.MAX_BYTES, 1024**3)
        self.assertEqual(o.accepted.MAX_BYTES, 3*1024**3)
        effective = {'memory.max': str(16*1024**3), 'memory.swap.max': '0', 'cpu.max': '100000 100000'}
        service = {'Type': 'exec', 'RuntimeMaxUSec': '10min', 'Restart': 'no', 'KillMode': 'control-group'}
        o.validate_bounds(effective, service)
        with self.assertRaises(RuntimeError):
            o.validate_bounds(effective, {**service, 'RuntimeMaxUSec': '30min'})

    def test_new_run_streamed_tensor_npz_json_cap_and_exclusive_receipts(self):
        with tempfile.TemporaryDirectory(prefix='observer-pathway-fixture-') as directory:
            run = o.Run(directory, 'cpu')
            with patch.object(run, 'check', return_value=None):
                a = run.save('test.pt', {'x': torch.arange(4)}, 'tensor')
                b = run.save('test.npz', {'x': np.arange(4)}, 'npz')
                c = run.save('test.json', {'success': True})
                self.assertEqual(run.used, sum(x['size_bytes'] for x in (a, b, c)))
                self.assertTrue(torch.equal(torch.load(Path(directory)/'test.pt', weights_only=True)['x'], torch.arange(4)))
                with np.load(Path(directory)/'test.npz', allow_pickle=False) as saved:
                    np.testing.assert_array_equal(saved['x'], np.arange(4))
                with self.assertRaises(FileExistsError):
                    run.save('test.json', {})
                with patch.object(o, 'MAX_BYTES', run.used+o.FOOTER_RESERVE+1):
                    with self.assertRaises(RuntimeError):
                        run.save('partial.json', {'too_large': True})
                self.assertTrue((Path(directory)/'partial.json').exists())

    def test_main_without_execute_never_admits_or_reads_scientific_data(self):
        with patch('sys.argv', ['fixture', '--output-dir', '/nonexistent/acquisition-001']), \
                patch.object(o, 'configure', side_effect=AssertionError('GPU configuration attempted')):
            with self.assertRaisesRegex(RuntimeError, 'explicit --execute'):
                o.main()


if __name__ == '__main__':
    unittest.main()
