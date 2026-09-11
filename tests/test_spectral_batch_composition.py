"""Fabricated CPU-only fixtures, never MNIST or any scientific parent state."""
import copy
import json
import math
import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments import spectral_batch_composition as b


def toy_plan(seed=421):
    return b.accepted.make_plan(np.repeat(np.arange(10), 64), seed, majority_train=12,
        rare_train=40, heldout_per_class=10, poison_count=20,
        warmup_steps=3, branch_steps=150, batch_size=8)[0]


class ScheduleTests(unittest.TestCase):
    def test_extreme_rare_counts_and_one_partial_group(self):
        positions = np.array([2, 0, 3, 1], dtype=np.int64)
        for total in (0, 1, 3, 4, 5, 15, 16):
            values = b.rare_count_vectors(total, positions, batch_size=4)
            for counts in values.values():
                self.assertEqual(int(counts.sum()), total)
                self.assertTrue(((counts >= 0) & (counts <= 4)).all())
            self.assertLessEqual(np.ptp(values['interleaved']), 1)
            self.assertLessEqual(int(((values['grouped'] > 0) & (values['grouped'] < 4)).sum()), 1)
        np.testing.assert_array_equal(b.rare_count_vectors(5, positions, 4)['grouped'], [1, 0, 4, 0])
        with self.assertRaises(RuntimeError):
            b.rare_count_vectors(17, positions, 4)
        with self.assertRaises(RuntimeError):
            b.rare_count_vectors(1, np.array([0, 0, 1, 2]), 4)

    def test_real_shape_all_rare_and_no_rare_preserve_repeated_occurrences(self):
        original = (np.arange(3200) % 31).reshape(50, 64).astype(np.int64)
        for label, expected in [(1, 0), (8, 64)]:
            schedules, counts, _ = b.schedule_block(original, np.full(31, label), 9, 0)
            for name in b.SCHEDULES:
                np.testing.assert_array_equal(np.sort(schedules[name].reshape(-1)), np.sort(original.reshape(-1)))
                np.testing.assert_array_equal(counts[name], expected)
            np.testing.assert_array_equal(schedules['grouped'], schedules['interleaved'])

    def test_exact_named_stream_draw_order_and_common_slot_permutation(self):
        original = (np.arange(40) % 13).reshape(5, 8).astype(np.int64)
        labels = np.array([8, 1, 2, 8, 3, 4, 8, 5, 6, 7, 8, 9, 0])
        schedules, counts, permutations = b.schedule_block(original, labels, 715, 2)
        order = np.random.Generator(np.random.PCG64(np.random.SeedSequence([715, 6, 2]))).permutation(40)
        positions = np.random.Generator(np.random.PCG64(np.random.SeedSequence([715, 7, 2]))).permutation(5)
        np.testing.assert_array_equal(permutations['occurrence'], order)
        np.testing.assert_array_equal(permutations['positions'], positions)
        source = original.reshape(-1)[order]
        rare, common = source[labels[source] == 8], source[labels[source] != 8]
        for name in b.SCHEDULES:
            r = c = 0
            for t, nr in enumerate(counts[name]):
                slot = np.random.Generator(np.random.PCG64(np.random.SeedSequence([715, 8, 2, t]))).permutation(8)
                np.testing.assert_array_equal(permutations['slots'][t], slot)
                expected = np.r_[rare[r:r+nr], common[c:c+8-nr]][slot]
                np.testing.assert_array_equal(schedules[name][t], expected)
                r, c = r+nr, c+8-nr
            self.assertEqual(r, len(rare))
            self.assertEqual(c, len(common))

    def test_whole_schedule_block_multisets_and_rng_locality(self):
        plan = toy_plan()
        before = np.random.get_state()
        schedules, checks = b.make_schedules(plan, 421)
        repeated, repeated_checks = b.make_schedules(plan, 421)
        for key in schedules:
            np.testing.assert_array_equal(schedules[key], repeated[key])
        self.assertEqual(checks, repeated_checks)
        self.assertEqual(len(checks), 3)
        after = np.random.get_state()
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])
        for row in checks:
            self.assertEqual(row['original_sorted_multiset_sha256'], row['interleaved_sorted_multiset_sha256'])
            self.assertEqual(row['original_sorted_multiset_sha256'], row['grouped_sorted_multiset_sha256'])
        for name in b.SCHEDULES:
            self.assertEqual(schedules[name+'_batches'].shape, (150, 8))
            np.testing.assert_array_equal((plan['train_true'][schedules[name+'_batches']] == 8).sum(1),
                                          schedules[name+'_rare_counts'])
        broken = {**plan, 'continuation_batches': plan['continuation_batches'][:-1]}
        with self.assertRaises(RuntimeError):
            b.make_schedules(broken, 421)

    def test_anchor_ties_first_extremum_excluding_high_and_chronology(self):
        counts = np.array([3, 1, 3, 1, 0, 0, 0, 0, 2, 3, 2, 1])
        events = b.select_anchors(counts, block_size=4, blocks=(0, 1, 2))
        self.assertEqual([row['update'] for row in events], [101, 102, 105, 106, 110, 112])
        self.assertEqual([row['label'] for row in events], ['high', 'low', 'high', 'low', 'high', 'low'])
        self.assertTrue(events[2]['equal_counts_in_block'])
        self.assertTrue(events[3]['equal_counts_in_block'])
        self.assertEqual(len({row['update'] for row in events}), 6)
        reversed_events = b.select_anchors(np.array([0, 3, 1, 2]), block_size=4, blocks=(0,))
        self.assertEqual([row['label'] for row in reversed_events], ['low', 'high'])

    def test_fixed_theta_mean_invariant_and_two_group_covariance(self):
        original = (np.arange(80) % 11).reshape(10, 8).astype(np.int64)
        labels = np.array([8, 8, 0, 1, 2, 3, 4, 5, 6, 7, 9])
        schedules, counts, _ = b.schedule_block(original, labels, 414, 0)
        common, rare = np.array([1., 2., -3.]), np.array([4., -1., 0.])
        means = {}
        for name in b.SCHEDULES:
            q = counts[name]/8
            gradients = common+q[:, None]*(rare-common)
            means[name] = gradients.mean(0)
            centered = gradients-gradients.mean(0)
            actual = centered.T @ centered/len(centered)
            expected = np.var(q)*np.outer(rare-common, rare-common)
            np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-14)
            fixed_losses = np.array([float(x*x+x) for x in range(11)])
            self.assertAlmostEqual(fixed_losses[schedules[name]].mean(), fixed_losses[original].mean(), places=13)
        np.testing.assert_allclose(means['interleaved'], means['grouped'], rtol=1e-14, atol=1e-14)


class ProbeMembershipTests(unittest.TestCase):
    def test_named_probe_streams_stratification_and_true_targets(self):
        plan = toy_plan()
        members = b.make_probe_plan(plan, 421)
        labels = plan['train_true']
        self.assertEqual(len(members['majority']), 54)
        self.assertEqual(len(members['rare8']), 32)
        self.assertEqual(len(np.unique(members['majority'])), 54)
        np.testing.assert_array_equal(np.bincount(labels[members['majority']], minlength=10), [6, 6, 6, 6, 6, 6, 6, 6, 0, 6])
        np.testing.assert_array_equal(labels[members['rare8']], 8)
        changed = np.flatnonzero(plan['diffuse_targets'] != labels)
        expected = np.random.Generator(np.random.PCG64(np.random.SeedSequence([421, 9, 2]))).permutation(changed)[:32]
        np.testing.assert_array_equal(members['wrong'], expected)
        x = torch.arange(len(labels)*4, dtype=torch.float32).reshape(len(labels), 4)
        groups = b.probe_definitions(plan, members, x, 'diffuse')
        for name in ('majority', 'rare8'):
            np.testing.assert_array_equal(groups[name]['targets'], groups[name]['true_targets'])
        self.assertTrue((groups['majority']['assigned_targets'] != groups['majority']['true_targets']).any())
        self.assertTrue(torch.equal(groups['wrong_assigned']['x'], groups['wrong_corrected']['x']))
        self.assertTrue((groups['wrong_assigned']['targets'] != groups['wrong_corrected']['targets']).all())
        clean = b.probe_definitions(plan, members, x, 'clean')
        self.assertIsNone(clean['wrong_assigned'])
        self.assertIsNone(clean['wrong_corrected'])

    def test_no_changed_examples_are_absent_not_zero(self):
        plan = toy_plan()
        plan['diffuse_targets'] = plan['train_true'].copy()
        probes = b.make_probe_plan(plan, 421)
        self.assertEqual(len(probes['wrong']), 0)
        groups = b.probe_definitions(plan, probes, torch.zeros(len(plan['train_true']), 4), 'diffuse')
        self.assertIsNone(groups['wrong_assigned'])
        self.assertIsNone(groups['wrong_corrected'])


class MeanProbeTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        hidden = patch.object(torch.cuda, 'is_available', return_value=False)
        hidden.start()
        self.addCleanup(hidden.stop)
        self.model = b.core.make_model(421, 'cpu', input_dim=4, width=3)
        self.optimizer = b.core.make_optimizer(self.model)
        self.tracker = b.core.make_tracker(self.model, self.optimizer)
        self.x = torch.arange(24, dtype=torch.float32).reshape(6, 4)/24
        self.labels = np.array([0, 1, 2, 3, 8, 9], dtype=np.int64)

    def test_mean_probe_matches_direct_mean_ce_and_preserves_state(self):
        b.core.set_grad(self.model, torch.ones_like(b.core.flat_params(self.model)))
        before = b.core.snapshot(self.model, self.optimizer, self.tracker)
        logits, gradient = b.mean_probe(self.model, self.x, self.labels)
        after = b.core.snapshot(self.model, self.optimizer, self.tracker)
        self.assertTrue(b.core.equal_tree(before, after))
        expected_logits = self.model(self.x)
        loss = torch.nn.functional.cross_entropy(expected_logits, torch.from_numpy(self.labels))
        parts = torch.autograd.grad(loss, tuple(self.model.parameters()))
        expected = torch.cat([p.reshape(-1) for p in parts])
        self.assertTrue(torch.equal(logits, expected_logits))
        self.assertTrue(torch.equal(gradient, expected))
        self.assertTrue(b.core.equal_tree(before, b.core.snapshot(self.model, self.optimizer, self.tracker)))

    def test_mean_geometry_keeps_fallback_and_zero_distinct(self):
        gradient = torch.ones_like(b.core.flat_params(self.model))
        row = b.mean_geometry(gradient, self.tracker)
        self.assertEqual(row['native_retention'], 1.)
        self.assertIsNone(row['numerical_span_retention'])
        self.assertTrue(row['identity_fallback'])
        row = b.mean_geometry(torch.zeros_like(gradient), self.tracker)
        self.assertIsNone(row['native_retention'])
        self.assertTrue(row['zero_denominator'])

    def test_native_action_retention_not_replaced_by_orthogonal_span(self):
        p = len(b.core.flat_params(self.model))
        v = torch.zeros(p, 2)
        v[0, 0], v[1, 1] = 2., 1.
        self.tracker.V = v
        gradient = torch.zeros(p)
        gradient[:2] = torch.tensor([1., 2.])
        geometry = b.mean_geometry(gradient, self.tracker)
        self.assertEqual(geometry['native_retention'], 4.)
        self.assertAlmostEqual(geometry['numerical_span_retention'], 1.)
        self.assertFalse(geometry['identity_fallback'])

    def test_nonfinite_mean_probe_rejected(self):
        self.x[0, 0] = float('nan')
        with self.assertRaises(RuntimeError):
            b.mean_probe(self.model, self.x, self.labels)


class TinyUpdateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.hidden = patch.object(torch.cuda, 'is_available', return_value=False)
        cls.hidden.start()
        model = b.core.make_model(87, 'cpu', input_dim=4, width=3)
        optimizer = b.core.make_optimizer(model)
        tracker = b.core.make_tracker(model, optimizer)
        cls.x = torch.arange(24, dtype=torch.float32).reshape(6, 4)/24
        cls.y = torch.tensor([0, 1, 2, 3, 8, 9])
        for step in range(1, 101):
            batch = [(step+i) % 6 for i in range(3)]
            b.training_update(model, optimizer, tracker, cls.x[batch], cls.y[batch], 'raw', step,
                              int((cls.y[batch] == 8).sum()))
        cls.state = b.core.snapshot(model, optimizer, tracker)

    @classmethod
    def tearDownClass(cls):
        cls.hidden.stop()

    def definitions(self, wrong=True):
        groups = {}
        for name, positions, targets in [('majority', [0, 1], [2, 3]), ('rare8', [2, 3], [8, 8]),
                                         ('wrong_assigned', [4, 5], [0, 0]), ('wrong_corrected', [4, 5], [2, 4])]:
            truth = [2, 4] if name.startswith('wrong') else targets
            groups[name] = None if not wrong and name.startswith('wrong') else {
                'training_positions': np.array(positions), 'source_ids': np.array(positions)+200,
                'x': self.x[positions], 'targets': np.array(targets), 'true_targets': np.array(truth),
                'assigned_targets': np.array([0, 0] if name.startswith('wrong') else targets)}
        return groups

    def test_fork_first_raw_and_complete_observer_identities(self):
        rows = {}
        for policy in b.POLICIES:
            model, optimizer, tracker = b.core.restore(self.state, 'cpu')
            self.assertTrue(b.core.equal_tree(self.state, b.core.snapshot(model, optimizer, tracker)))
            if policy == 'raw':
                tracker = None
            row, diagnostic = b.training_update(model, optimizer, tracker, self.x, self.y, policy, 101, 1)
            rows[policy] = row
            self.assertIsNone(diagnostic)
            self.assertTrue(all(p.grad is None for p in model.parameters()))
        self.assertEqual(len({r['raw_gradient_sha256'] for r in rows.values()}), 1)
        self.assertEqual(rows['native32']['post_observe_tracker_sha256'], rows['norm_raw']['post_observe_tracker_sha256'])
        self.assertEqual(rows['native32']['native_norm'], rows['native32']['applied_norm'])
        control = rows['norm_raw']['norm_law']
        self.assertLessEqual(control['relative_postcast_mismatch'], control['relative_tolerance'])

    def test_mean_diagnostic_snapshots_action_differences_and_adam_step(self):
        model, optimizer, tracker = b.core.restore(self.state, 'cpu')
        anchor = {'block': 0, 'position': 0, 'label': 'high', 'update': 101, 'rare_count': 1, 'equal_counts_in_block': False}
        identity = {'seed': 11, 'cell': 'diffuse', 'schedule': 'grouped', 'policy': 'native32',
                    'warmup_state_sha256': b.core.tree_digest(self.state)}
        _, diagnostic = b.training_update(model, optimizer, tracker, self.x, self.y, 'native32', 101, 1,
                                          self.definitions(), anchor, identity)
        payload, report = diagnostic
        self.assertEqual(payload['schedule'], 'grouped')
        self.assertEqual(report['anchor'], anchor)
        self.assertEqual(report['probe_side_effect_checks'], 'PASS')
        self.assertFalse(report['per_example_gradients_or_coherence'])
        self.assertTrue(b.core.equal_tree(payload['before']['tracker'], payload['after']['tracker']))
        for group in payload['groups'].values():
            self.assertEqual(group['mean_gradient'].ndim, 1)
            self.assertNotIn('per_example_gradients', group)
            self.assertTrue(bool(torch.isfinite(group['post_logits']).all()))
        expected = payload['groups']['rare8']['mean_gradient']-payload['groups']['majority']['mean_gradient']
        self.assertTrue(torch.equal(payload['mean_differences']['rare_minus_majority'], expected))
        self.assertEqual(expected.dtype, torch.float32)
        self.assertTrue(torch.equal(payload['applied_training_gradient'], torch.cat([
            value.reshape(-1) for value in payload['before']['gradients']])))
        for key in payload['before']['optimizer']['state']:
            before = payload['before']['optimizer']['state'][key]['step']
            after = payload['after']['optimizer']['state'][key]['step']
            self.assertEqual(float(after-before), 1.)
        q, _ = b.accepted.numerical_basis(tracker)
        vectors, movement = b.accepted.movement_summary(payload['before'], payload['after'], q)
        self.assertEqual(movement, report['movement'])
        torch.testing.assert_close(vectors['total'], vectors['decay']+vectors['adaptive'], atol=0, rtol=0)
        for name, group in payload['groups'].items():
            for component, delta in vectors.items():
                self.assertEqual(report['groups'][name]['signed_first_order_utilities'][component],
                                 -float(group['mean_gradient'].double() @ delta))
        json.dumps(report, allow_nan=False)

    def test_clean_absent_wrong_diagnostic_and_mismatched_anchor_rejected(self):
        model, optimizer, tracker = b.core.restore(self.state, 'cpu')
        anchor = {'block': 0, 'position': 0, 'label': 'high', 'update': 101, 'rare_count': 1, 'equal_counts_in_block': False}
        _, diagnostic = b.training_update(model, optimizer, tracker, self.x, self.y, 'native32', 101, 1,
            self.definitions(False), anchor, {'seed': 11, 'cell': 'clean', 'schedule': 'interleaved', 'policy': 'native32'})
        payload, report = diagnostic
        self.assertIsNone(payload['groups']['wrong_assigned'])
        self.assertIsNone(payload['mean_differences']['wrong_assigned_minus_corrected'])
        self.assertIsNone(report['mean_difference_geometry']['wrong_assigned_minus_corrected'])
        self.assertEqual(report['groups']['wrong_assigned']['status'], 'absent')
        model, optimizer, tracker = b.core.restore(self.state, 'cpu')
        with self.assertRaisesRegex(RuntimeError, 'anchor identity'):
            b.training_update(model, optimizer, tracker, self.x, self.y, 'native32', 101, 2,
                              self.definitions(), anchor, {})


class SummaryAndGuardTests(unittest.TestCase):
    def test_paired_means_schedule_policy_interaction_and_nulls(self):
        truth = np.arange(10)
        logits = np.eye(10, dtype=np.float32)
        base = b.evaluation_row(2000, logits, logits, truth, truth, truth)
        rows = []
        for index, seed in enumerate(b.SEEDS):
            for cell in b.CELLS:
                for schedule in b.SCHEDULES:
                    for policy in b.POLICIES:
                        row = copy.deepcopy(base)
                        effect = (.01*(index+1) if policy == 'native32' else .002) if schedule == 'grouped' else 0.
                        row['heldout']['rare']['accuracy'] = .2+effect
                        row['heldout']['rare']['ce'] = 2.-effect
                        warmup = copy.deepcopy(row)
                        warmup['step'] = 100
                        warmup['heldout']['rare']['accuracy'] = 0.
                        rows.append({'seed': seed, 'cell': cell, 'schedule': schedule, 'policy': policy,
                                     'endpoint': row, 'warmup': warmup})
        summary = b.summarize_results(rows)
        effect = summary['schedule_policy_interactions']['clean/native32_minus_raw']['rare_accuracy']
        np.testing.assert_allclose(effect['values'], [.008, .018, .028], atol=1e-15)
        self.assertAlmostEqual(effect['mean'], .018)
        self.assertAlmostEqual(effect['sample_sd'], .01)
        self.assertAlmostEqual(effect['sample_se'], .01/math.sqrt(3))
        self.assertEqual(effect['positive_count'], 3)
        ce = summary['schedule_policy_interactions']['clean/native32_minus_raw']['rare_ce']
        np.testing.assert_allclose(ce['values'], [-.008, -.018, -.028], atol=1e-15)
        self.assertEqual(ce['negative_count'], 3)
        self.assertIsNone(summary['per_group']['clean/grouped/raw']['train_wrong_accuracy'])
        self.assertIn('heldout_class8_ce', summary['policy_contrasts']['diffuse/grouped/native32_minus_norm_raw'])
        self.assertEqual(summary['selection'], 'fixed_step_2000_no_checkpoint_selection')
        with self.assertRaises(RuntimeError):
            b.summarize_results(rows[:-1])
        with self.assertRaises(RuntimeError):
            b.summarize_results(rows[:-1]+[rows[0]])

    def test_inventory_and_immutable_inherited_bounds(self):
        inventory = b.byte_inventory()
        self.assertEqual(inventory['total_upper_bytes'], sum(inventory['component_upper_bytes'].values()))
        self.assertLess(inventory['total_upper_bytes']+b.RESERVE_BYTES, 3*1024**3)
        self.assertEqual((b.MAX_BYTES, b.RESERVE_BYTES, b.DEADLINE_SECONDS),
                         (b.accepted.MAX_BYTES, b.accepted.RESERVE_BYTES, b.accepted.DEADLINE_SECONDS))
        self.assertEqual(b.digest(b.ROOT/'experiments/spectral_selectivity_boundary.py'), b.ACCEPTED_SHA)

    def test_explicit_execution_guard_precedes_any_scientific_action(self):
        with patch('sys.argv', ['fixture', '--output-dir', '/nonexistent/acquisition-001']), \
                patch.object(b, 'configure', side_effect=AssertionError('GPU configuration called')), \
                patch.object(b, 'acquire', side_effect=AssertionError('scientific acquisition called')):
            with self.assertRaisesRegex(RuntimeError, 'explicit --execute'):
                b.main()


if __name__ == '__main__':
    unittest.main()
