"""Fabricated CPU fixtures only; no MNIST, saved scientific state, or CUDA use."""
import copy
import io
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments import spectral_selectivity_boundary as s


def toy_plan(seed=917):
    return s.make_plan(np.repeat(np.arange(10), 12), seed, majority_train=6,
                       rare_train=2, heldout_per_class=3, poison_count=8,
                       warmup_steps=3, branch_steps=6, batch_size=4)


def toy_logits(labels, correct=True):
    labels = np.asarray(labels)
    logits = np.full((len(labels), 10), -1., dtype=np.float32)
    logits[np.arange(len(labels)), labels if correct else (labels+1) % 10] = 2.
    return logits


class BoundaryDataTests(unittest.TestCase):
    def test_plan_is_paired_disjoint_and_stream_local(self):
        rng_before = np.random.get_state()
        a, quotas = toy_plan()
        b, quotas_b = toy_plan()
        for name in a:
            np.testing.assert_array_equal(a[name], b[name])
        self.assertEqual(quotas, quotas_b)
        after = np.random.get_state()
        self.assertEqual(rng_before[0], after[0])
        np.testing.assert_array_equal(rng_before[1], after[1])
        self.assertEqual(rng_before[2:], after[2:])
        self.assertEqual(len(a['train_ids']), 56)
        self.assertEqual(len(a['heldout_ids']), 30)
        self.assertEqual(np.intersect1d(a['train_ids'], a['heldout_ids']).size, 0)
        self.assertFalse((a['train_true'][a['warmup_batches']] == 8).any())
        self.assertTrue((a['continuation_batches'] >= 0).all())
        self.assertTrue((a['continuation_batches'] < 56).all())
        other, _ = toy_plan(918)
        self.assertFalse(np.array_equal(a['train_ids'], other['train_ids']))

    def test_no_rare_corruption_or_patch_and_fixed_target_pair(self):
        plan, _ = toy_plan()
        rare = plan['train_true'] == 8
        self.assertFalse(plan['diffuse_selected'][rare].any())
        np.testing.assert_array_equal(plan['diffuse_replacement_labels'][rare], -1)
        self.assertFalse(np.isin(plan['diffuse_targets'][~rare], [8]).any())
        defined = {cell: s.cell_definition(plan, cell) for cell in s.CELLS}
        for targets, mask, counts in defined.values():
            np.testing.assert_array_equal(targets[rare], 8)
            self.assertFalse(mask[rare].any())
            self.assertEqual(np.asarray(counts['contingency_counts']).sum(), 56)
        np.testing.assert_array_equal(defined['shared'][0], defined['sham'][0])
        self.assertEqual(defined['shared'][2]['actually_changed_count'], 8)
        self.assertEqual(defined['shared'][2]['patch_poison_overlap'], 8)
        self.assertEqual(defined['sham'][2]['patch_count'], 8)
        self.assertEqual(defined['clean'][2]['actually_changed_count'], 0)

    def test_sham_exact_true_class_counts_and_conditional_rounding(self):
        plan, allocations = toy_plan()
        labels, poison, sham = plan['train_true'], plan['poison_mask'], plan['sham_patch_mask']
        for digit in (1, 2, 3, 4, 5, 6, 7, 9):
            group = labels == digit
            k = int((group & poison).sum())
            self.assertEqual(int((group & sham).sum()), k)
            rows = [row for row in allocations if row['true_digit'] == digit]
            expected = s.largest_remainder(k, [int((group & ~poison).sum()), int((group & poison).sum())])
            self.assertEqual([row['sham_patch_count'] for row in rows], expected)
            for row in rows:
                realized = int((sham & group & (poison == row['poison'])).sum())
                self.assertEqual(realized, row['sham_patch_count'])
                self.assertLessEqual(abs(row['rounding_deviation']), 1.)
        self.assertEqual(s.largest_remainder(1, [1, 1]), [1, 0])
        self.assertEqual(s.largest_remainder(0, [3, 2]), [0, 0])
        with self.assertRaises(RuntimeError):
            s.largest_remainder(1, [-1, 2])

    def test_patch_clone_and_exact_pixel_region(self):
        images = torch.arange(4*784, dtype=torch.float32).reshape(4, 784)/4096
        original = images.clone()
        result = s.patch_images(images, np.array([True, False, False, True]))
        self.assertTrue(torch.equal(images, original))
        self.assertNotEqual(images.data_ptr(), result.data_ptr())
        expected = original.reshape(4, 28, 28).clone()
        expected[[0, 3], :3, :3] = 1.
        self.assertTrue(torch.equal(result, expected.reshape(4, 784)))
        all_patched = s.patch_images(images)
        self.assertTrue(torch.equal(all_patched.reshape(4, 28, 28)[:, :3, :3], torch.ones(4, 3, 3)))
        self.assertTrue(torch.equal(images, original))

    def test_cpu_numpy_normalization_and_hash_for_every_byte(self):
        images = (np.arange(3*784) % 256).astype(np.uint8).reshape(3, 784)
        before = images.copy()
        ids = np.array([2, 0], dtype=np.int64)
        result = s.normalized_inputs(images, ids, 'cpu')
        expected = np.ascontiguousarray(images[ids].astype(np.float32)/np.float32(255))
        np.testing.assert_array_equal(result.numpy().view(np.uint32), expected.view(np.uint32))
        np.testing.assert_array_equal(images, before)
        self.assertEqual(s.array_digest(result.numpy()), s.array_digest(expected))
        self.assertNotEqual(s.array_digest(expected), s.array_digest(expected.astype(np.float64)))
        self.assertNotEqual(s.array_digest(expected), s.array_digest(expected.reshape(784, 2)))

    def test_probe_definitions_are_fixed_and_use_actual_wrong_inputs(self):
        plan, _ = toy_plan()
        x = torch.zeros(56, 784)
        assigned, mask, _ = s.cell_definition(plan, 'shared')
        patched = s.patch_images(x, mask)
        groups = s.probe_inputs(plan, assigned, patched, x)
        wrong = np.flatnonzero(assigned != plan['train_true'])[:32]
        np.testing.assert_array_equal(groups['wrong_assigned']['training_positions'], wrong)
        self.assertTrue(torch.equal(groups['wrong_assigned']['x'], groups['wrong_corrected']['x']))
        np.testing.assert_array_equal(groups['wrong_assigned']['targets'], 0)
        np.testing.assert_array_equal(groups['wrong_corrected']['targets'], plan['train_true'][wrong])
        self.assertTrue(torch.equal(groups['rare8']['x'], torch.zeros(2, 784)))
        clean = s.probe_inputs(plan, plan['train_true'], x, x)
        self.assertIsNone(clean['wrong_assigned'])
        self.assertIsNone(clean['wrong_corrected'])


class BoundaryMetricTests(unittest.TestCase):
    def test_finite_ce_and_empty_wrong_semantics(self):
        labels = np.arange(10)
        logits = toy_logits(labels)
        row = s.classification_stats(logits, labels)
        expected = math.log(math.exp(2)+9*math.exp(-1))-2
        self.assertEqual(row['correct'], 10)
        self.assertAlmostEqual(row['ce'], expected, places=14)
        self.assertAlmostEqual(row['ce_sum'], 10*expected, places=13)
        empty = s.classification_stats(logits, labels, np.zeros(10, dtype=bool))
        self.assertEqual(empty, {'count': 0, 'correct': 0, 'ce_sum': 0., 'accuracy': None, 'ce': None})
        logits[0, 0] = np.nan
        with self.assertRaises(RuntimeError):
            s.classification_stats(logits, labels)

    def test_rare_majority_macro_does_not_use_frequency_weight(self):
        labels = np.concatenate([np.arange(10), np.repeat(0, 10)])
        logits = toy_logits(labels)
        logits[labels == 8] = toy_logits(labels[labels == 8], False)
        metrics = s.group_stats(logits, labels)
        self.assertEqual(metrics['rare']['accuracy'], 0.)
        self.assertEqual(metrics['majority_macro']['accuracy'], 1.)
        self.assertEqual(metrics['balanced_total']['accuracy'], .9)
        self.assertEqual(metrics['micro']['accuracy'], .95)

    def test_cue_excess_uses_matched_fixed_nonzero_populations(self):
        labels = np.arange(10)
        clean = toy_logits(labels)
        patched = clean.copy()
        clean[1] = toy_logits(np.array([0]))[0]
        patched[[1, 2, 8]] = toy_logits(np.array([0, 0, 0]))
        cue = s.cue_stats(clean, patched, labels)
        self.assertEqual(cue['majority_nonzero']['count'], 8)
        self.assertEqual(cue['majority_nonzero']['patch_excess'], 1/8)
        self.assertEqual(cue['rare']['count'], 1)
        self.assertEqual(cue['rare']['patch_excess'], 1.)
        self.assertEqual(cue['all_nonzero']['count'], 9)
        self.assertAlmostEqual(cue['all_nonzero']['patch_excess'], 2/9)

    def test_paired_summary_fixed_endpoint_interaction_and_signs(self):
        labels = np.arange(10)
        logits = toy_logits(labels)
        endpoint = s.evaluation_row(2000, logits, logits, logits, labels, labels, labels)
        rows = []
        for index, seed in enumerate(s.SEEDS):
            for cell in s.CELLS:
                for policy in s.POLICIES:
                    value = copy.deepcopy(endpoint)
                    shared = cell == 'shared'
                    effect = (index+1)*.01 if policy == 'native32' else 0.
                    value['heldout_unpatched']['rare']['accuracy'] = .5+effect
                    value['cue']['majority_nonzero']['patch_excess'] = (.2+effect) if shared else .05
                    value['cue']['majority_nonzero']['patched_target0_rate'] = (.4+effect) if shared else .1
                    warm = copy.deepcopy(value)
                    warm['step'] = 100
                    warm['heldout_unpatched']['rare']['accuracy'] -= .1
                    rows.append({'seed': seed, 'cell': cell, 'policy': policy, 'endpoint': value, 'warmup': warm})
        summary = s.summarize_results(rows)
        contrast = summary['policy_contrasts']['clean/native32_minus_raw']['rare_accuracy']
        np.testing.assert_allclose(contrast['values'], [.01, .02, .03], atol=1e-15)
        self.assertAlmostEqual(contrast['mean'], .02)
        self.assertAlmostEqual(contrast['sample_sd'], .01)
        self.assertAlmostEqual(contrast['sample_se'], .01/math.sqrt(3))
        self.assertEqual(contrast['positive_count'], 3)
        interaction = summary['cue_interactions']['native32_minus_raw/majority_nonzero_patch_excess']
        np.testing.assert_allclose(interaction['values'], [.01, .02, .03], atol=1e-15)
        self.assertIsNone(summary['per_group']['clean/raw']['train_wrong_accuracy'])
        self.assertEqual(summary['selection'], 'fixed_step_2000_no_checkpoint_selection')
        for key in ('patched_rare_accuracy', 'patched_majority_macro_ce', 'patched_balanced_total_accuracy',
                    'patched_class3_ce', 'unpatched_class8_accuracy', 'train_true_class0_ce',
                    'train_true_rare_accuracy', 'train_assigned_ce', 'train_wrong_ce'):
            self.assertIn(key, summary['per_group']['clean/raw'])
            self.assertIn(key, summary['policy_contrasts']['clean/native32_minus_raw'])
            self.assertIn(key, summary['change_from_warmup']['clean/raw'])
        with self.assertRaises(RuntimeError):
            s.summarize_results(rows[:-1])
        with self.assertRaises(RuntimeError):
            s.summarize_results(rows[:-1]+[rows[0]])
        with self.assertRaises(RuntimeError):
            s.sample_summary([None, 1., 2.])


class BoundaryActionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.no_cuda = patch.object(torch.cuda, 'is_available', return_value=False)
        cls.no_cuda.start()
        cls.x = torch.arange(24, dtype=torch.float32).reshape(6, 4)/24
        cls.y = torch.tensor([0, 1, 2, 3, 8, 9])
        model = s.core.make_model(51, 'cpu', input_dim=4, width=3, classes=10)
        optimizer = s.core.make_optimizer(model)
        tracker = s.core.make_tracker(model, optimizer)
        # A tiny fabricated 55-parameter state; never scientific MNIST or an archived parent.
        for step in range(1, 101):
            batch = [(step+i) % 6 for i in range(3)]
            s.training_update(model, optimizer, tracker, cls.x[batch], cls.y[batch], 'raw', step)
        cls.warmup = s.core.snapshot(model, optimizer, tracker)

    @classmethod
    def tearDownClass(cls):
        cls.no_cuda.stop()

    def fresh(self):
        return s.core.restore(self.warmup, 'cpu')

    def definitions(self, wrong=True):
        result = {}
        for name, positions, targets in [('common3', [0, 1], [3, 3]), ('rare8', [2, 3], [8, 8]),
                                         ('wrong_assigned', [4, 5], [0, 0]), ('wrong_corrected', [4, 5], [2, 4])]:
            result[name] = None if not wrong and name.startswith('wrong_') else {
                'training_positions': np.array(positions), 'source_ids': np.array(positions)+1000,
                'x': self.x[positions].clone(), 'targets': np.array(targets),
                'true_targets': np.array([2, 4] if name.startswith('wrong_') else targets)}
        return result

    def test_initial_dict_hash_and_exact_fork_restore(self):
        model, optimizer, tracker = self.fresh()
        restored = s.core.snapshot(model, optimizer, tracker)
        self.assertTrue(s.core.equal_tree(restored, self.warmup))
        self.assertEqual(s.core.tree_digest(restored), s.core.tree_digest(self.warmup))
        self.assertEqual(len(s.core.tree_digest(dict(model.state_dict()))), 64)
        with self.assertRaises(s.core.NeuralCoreError):
            s.core.tree_digest(model.state_dict())
        raw = s.core.snapshot(model, optimizer, None)
        self.assertIsNone(raw['tracker'])
        for name in ('model_state', 'gradients', 'optimizer', 'rng'):
            self.assertTrue(s.core.equal_tree(restored[name], raw[name]))

    def test_norm_direction_fp64_scale_cast_and_zero_semantics(self):
        raw = torch.tensor([1.125, -3.5, .0625])
        native = torch.tensor([.2, .3, -.1])
        actual, info = s.norm_direction(raw, native)
        expected = (raw.double()*(native.double().norm()/raw.double().norm())).float()
        self.assertTrue(torch.equal(actual, expected))
        self.assertLessEqual(info['relative_postcast_mismatch'], 10*torch.finfo(torch.float32).eps)
        zero, info = s.norm_direction(raw, torch.zeros_like(raw))
        self.assertTrue(torch.equal(zero, torch.zeros_like(raw)))
        self.assertEqual(info['scale'], 0.)
        zero, info = s.norm_direction(torch.zeros_like(raw), torch.zeros_like(raw))
        self.assertTrue(info['zero_raw'])
        self.assertIsNone(info['scale'])
        with self.assertRaises(RuntimeError):
            s.norm_direction(torch.zeros_like(raw), native)
        with self.assertRaises(RuntimeError):
            s.norm_direction(torch.tensor([float('nan')]), torch.ones(1))

    def test_native_startup_identity_fallback(self):
        model = s.core.make_model(4, 'cpu', input_dim=4, width=3)
        optimizer = s.core.make_optimizer(model)
        tracker = s.core.make_tracker(model, optimizer)
        raw = torch.ones_like(s.core.flat_params(model))
        self.assertIsNone(tracker.V)
        native = tracker._project_gradient(raw)
        self.assertTrue(torch.equal(native, raw))
        delivery, info = s.norm_direction(raw, native)
        self.assertTrue(torch.equal(delivery, raw))
        self.assertEqual(info['scale'], 1.)
        geometry = s.gradient_geometry(torch.stack([raw, raw]), tracker)
        self.assertEqual(geometry['coherence'], 1.)
        self.assertEqual(geometry['native_mean_retention'], 1.)
        self.assertTrue(geometry['basis_identity_fallback'])
        zeros = s.gradient_geometry(torch.zeros(2, len(raw)), tracker)
        self.assertIsNone(zeros['coherence'])
        self.assertEqual(zeros['individual_native_retention'], [None, None])

    def test_raw_observe_action_adam_order_and_pairing(self):
        first = {}
        for policy in s.POLICIES:
            model, optimizer, tracker = self.fresh()
            if policy == 'raw':
                tracker = None
            events = []
            original_step = optimizer.step
            def step(*args, **kwargs):
                events.append('adam')
                return original_step(*args, **kwargs)
            with patch.object(optimizer, 'step', side_effect=step):
                if tracker is not None:
                    original_observe = tracker._update_svd
                    def observation(raw):
                        events.append('observe')
                        return original_observe(raw)
                    # Do not snapshot while Mock is an instance attribute: core correctly rejects it.
                    with patch.object(tracker, '_update_svd', side_effect=observation):
                        with patch.object(s.core, '_tracker_state', side_effect=lambda t: {'step_count': t.step_count}):
                            row, _ = s.training_update(model, optimizer, tracker, self.x, self.y, policy, 101)
                else:
                    row, _ = s.training_update(model, optimizer, tracker, self.x, self.y, policy, 101)
            first[policy] = row
            self.assertEqual(events, ['adam'] if tracker is None else ['observe', 'adam'])
            self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
            if policy == 'native32':
                self.assertEqual(row['native_norm'], row['applied_norm'])
            if policy == 'norm_raw':
                self.assertLessEqual(row['norm_law']['relative_postcast_mismatch'], row['norm_law']['relative_tolerance'])
        self.assertEqual(len({row['raw_gradient_sha256'] for row in first.values()}), 1)
        self.assertEqual(first['native32']['post_observe_tracker_sha256'], first['norm_raw']['post_observe_tracker_sha256'])

    def test_probe_gradients_match_single_example_and_preserve_all_state(self):
        model, optimizer, tracker = self.fresh()
        s.core.set_grad(model, torch.arange(len(s.core.flat_params(model)), dtype=torch.float32)/100)
        before = s.core.snapshot(model, optimizer, tracker)
        _, groups = s.collect_pre_probes(model, optimizer, tracker, self.definitions())
        self.assertTrue(s.core.equal_tree(before, s.core.snapshot(model, optimizer, tracker)))
        group = groups['common3']
        for index in range(2):
            logits = model(group['inputs'][index:index+1])
            loss = torch.nn.functional.cross_entropy(logits, group['targets'][index:index+1])
            gradients = torch.autograd.grad(loss, tuple(model.parameters()))
            flat = torch.cat([gradient.reshape(-1) for gradient in gradients])
            torch.testing.assert_close(group['per_example_gradients'][index], flat, rtol=2e-6, atol=1e-7)
        self.assertTrue(s.core.equal_tree(before, s.core.snapshot(model, optimizer, tracker)))

    def test_native_diagnostic_saves_actual_update_and_finite_probes(self):
        model, optimizer, tracker = self.fresh()
        row, diagnostic = s.training_update(model, optimizer, tracker, self.x, self.y, 'native32', 101,
            self.definitions(), {'seed': 17, 'cell': 'shared', 'policy': 'native32', 'warmup_state_sha256': 'fixture'})
        tensor, report = diagnostic
        self.assertEqual(report['probe_side_effect_checks'], 'PASS')
        self.assertEqual(report['update'], 101)
        self.assertEqual(tensor['before']['tracker']['step_count'], 101)
        self.assertTrue(s.core.equal_tree(tensor['before']['tracker'], tensor['after']['tracker']))
        self.assertTrue(torch.equal(tensor['applied_training_gradient'], torch.cat([
            gradient.reshape(-1) for gradient in tensor['before']['gradients']])))
        self.assertEqual(report['groups']['wrong_assigned']['true_label_counts'], [0, 0, 1, 0, 1, 0, 0, 0, 0, 0])
        self.assertIsNotNone(report['wrong_minus_true_gradient_geometry'])
        for group in tensor['groups'].values():
            self.assertTrue(bool(torch.isfinite(group['pre_logits']).all()))
            self.assertTrue(bool(torch.isfinite(group['post_logits']).all()))
        q, _ = s.numerical_basis(tracker)
        vectors, movement = s.movement_summary(tensor['before'], tensor['after'], q)
        torch.testing.assert_close(vectors['total'], vectors['decay']+vectors['adaptive'], rtol=0, atol=0)
        self.assertEqual(movement, report['movement'])
        self.assertEqual(row['applied_norm'], float(tensor['applied_training_gradient'].double().norm()))
        json.dumps(report, allow_nan=False)
        model, optimizer, tracker = self.fresh()
        _, diagnostic = s.training_update(model, optimizer, tracker, self.x, self.y, 'native32', 101,
            self.definitions(False), {'seed': 17, 'cell': 'clean', 'policy': 'native32'})
        self.assertEqual(diagnostic[1]['groups']['wrong_assigned']['status'], 'absent')
        self.assertIsNone(diagnostic[0]['groups']['wrong_assigned'])
        self.assertIsNone(diagnostic[1]['wrong_minus_true_gradient_geometry'])

    def test_explicit_zero_updates_moments_and_counter(self):
        model, optimizer, tracker = self.fresh()
        before = s.core.snapshot(model, optimizer, tracker)
        s.core.set_grad(model, torch.zeros_like(s.core.flat_params(model)))
        optimizer.step()
        after = s.core.snapshot(model, optimizer, tracker)
        self.assertFalse(s.core.equal_tree(before['model_state'], after['model_state']))
        for key, state in before['optimizer']['state'].items():
            updated = after['optimizer']['state'][key]
            self.assertEqual(float(updated['step']), float(state['step'])+1)
            torch.testing.assert_close(updated['exp_avg'], state['exp_avg']*.9, rtol=2e-7, atol=1e-10)
            torch.testing.assert_close(updated['exp_avg_sq'], state['exp_avg_sq']*.999, rtol=2e-7, atol=1e-10)


class BoundaryStorageGuardTests(unittest.TestCase):
    def test_conservative_inventory_is_under_3gib_with_failure_room(self):
        inventory = s.byte_inventory()
        self.assertEqual(inventory['total_upper_bytes'], sum(inventory['component_upper_bytes'].values()))
        self.assertLess(inventory['total_upper_bytes']+s.RESERVE_BYTES, 3*1024**3)
        self.assertEqual(inventory['component_upper_bytes']['4032_per_example_fp32_gradients'], 4032*50890*4)
        self.assertEqual(inventory['component_upper_bytes']['36_full_logit_histories'], 36*21*3*5000*10*4)

    def test_stream_cap_counts_nbytes_not_first_dimension(self):
        handle = io.BytesIO()
        writer = s.CappedWriter(handle, 16)
        writer.write(np.zeros((2, 2), dtype=np.float32))
        self.assertEqual(writer.allowance, 0)
        self.assertEqual(len(handle.getvalue()), 16)
        with self.assertRaises(RuntimeError):
            writer.write(b'x')

    def test_exclusive_npz_tensor_json_receipts_and_cap(self):
        with tempfile.TemporaryDirectory(prefix='selectivity-fixture-') as temporary:
            run = s.Run(temporary, 'cpu')
            # This only isolates writer tests from the host's unrelated free-space state.
            with patch.object(run, 'check', return_value=None):
                values = {'x': np.arange(6, dtype=np.float32).reshape(2, 3)}
                npz = run.save('fixture.npz', values, 'npz')
                with np.load(Path(temporary)/npz['path'], allow_pickle=False) as loaded:
                    np.testing.assert_array_equal(loaded['x'], values['x'])
                tensor = run.save('fixture.pt', {'x': torch.arange(3)}, 'tensor')
                loaded = torch.load(Path(temporary)/tensor['path'], weights_only=True, map_location='cpu')
                self.assertTrue(torch.equal(loaded['x'], torch.arange(3)))
                saved = run.save('fixture.json', {'finite': 1.})
                self.assertEqual(saved['sha256'], s.digest(Path(temporary)/saved['path']))
                self.assertEqual(run.used, sum(item['size_bytes'] for item in run.receipts))
                with self.assertRaises(FileExistsError):
                    run.save('fixture.json', {})
                with self.assertRaises(RuntimeError):
                    run.save('../escape.json', {})
                with patch.object(s, 'MAX_BYTES', run.used+s.RESERVE_BYTES+2):
                    with self.assertRaises(RuntimeError):
                        run.save('too-large.json', {'long': 'x'*100})
                self.assertTrue((Path(temporary)/'too-large.json').exists())

    def test_resource_guard_rejects_changed_limits(self):
        effective = {'memory.max': str(16*1024**3), 'memory.swap.max': '0', 'cpu.max': '100000 100000'}
        service = {'Type': 'exec', 'RuntimeMaxUSec': '30min', 'Restart': 'no', 'KillMode': 'control-group'}
        s.validate_bounds(effective, service)
        for key, value in [('memory.max', 'max'), ('memory.swap.max', '1'), ('cpu.max', '200000 100000')]:
            with self.assertRaises(RuntimeError):
                s.validate_bounds({**effective, key: value}, service)
        with self.assertRaises(RuntimeError):
            s.validate_bounds(effective, {**service, 'Restart': 'on-failure'})

    def test_no_execute_does_not_read_data_or_launch(self):
        with patch('sys.argv', ['fixture', '--output-dir', '/nonexistent/acquisition-001']), \
                patch.object(s, 'read_training', side_effect=AssertionError('scientific data read')), \
                patch.object(s, 'configure', side_effect=AssertionError('GPU configured')):
            with self.assertRaisesRegex(RuntimeError, 'explicit --execute'):
                s.main()


if __name__ == '__main__':
    unittest.main()
