"""Fabricated CPU-only producer integration; no source data/model replay."""
import contextlib
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import zipfile

import numpy as np
import torch

from experiments import spectral_component_utility as runner


def receipt(name):
    return {'path': name, 'size_bytes': 12, 'sha256': '0' * 64}


def metadata_fixture():
    inventory = {'schema': 'spectral_component_utility_parent_inventory_v1',
                 'archive': '/fabricated/archive', 'source_results_sha256': '1' * 64,
                 'source_audit_sha256': '2' * 64, 'parents': [],
                 'source_pins': {'fabricated.py': '3' * 64},
                 'data_pins': dict(runner.base.DATA_PINS), 'plan_receipts': []}
    old = {'schema': 'spectral_strong_augmentation_v1', 'status': 'complete',
           'source_pins': inventory['source_pins'], 'data_pins': inventory['data_pins'],
           'branches': [], 'receipts': []}
    audit = {'schema': 'spectral_strong_augmentation_audit_v1', 'status': 'PASS',
             'input_results_sha256': inventory['source_results_sha256'], 'input_dir': inventory['archive'],
             'source_pins': inventory['source_pins'], 'data_pins': inventory['data_pins']}
    for seed in runner.SEEDS:
        for aug in runner.AUGMENTATIONS:
            parent = {'seed': seed, 'augmentation': aug,
                      'plan_npz': f'plan-s{seed}.npz', 'plan_json': f'plan-s{seed}.json'}
            branch = {'seed': seed, 'augmentation': aug, 'policy': 'native200', 'metrics': []}
            for stage, step in zip(('warmup', 'final'), runner.STEPS):
                parent[stage] = receipt(f'{stage}-s{seed}-{aug}.pt')
                parent[stage + '_readout'] = receipt(f'logits-s{seed}-{aug}-h{step:05d}.npz')
                branch[stage + '_receipt'] = parent[stage]
                branch['metrics'].append({'step': step, 'readout_receipt': parent[stage + '_readout']})
                old['receipts'].extend((parent[stage], parent[stage + '_readout']))
            inventory['parents'].append(parent)
            old['branches'].append(branch)
        inventory['plan_receipts'].extend(receipt(f'plan-s{seed}.{ext}') for ext in ('npz', 'json'))
    old['receipts'].extend(inventory['plan_receipts'])
    return inventory, old, audit


def endpoint_fixture(row):
    baseline = {'objectives': dict.fromkeys(runner.KEYS, 10.)}
    endpoints = []
    scale = 1 + runner.SEEDS.index(row['seed']) + 3 * runner.STEPS.index(row['step'])
    scale *= 1 if row['augmentation'] == 'none' else -1
    for batch in range(2):
        for policy in ('raw', 'native', 'decay'):
            for fraction, name in runner.FRACTIONS:
                finite = scale * (batch + 1) * {'raw': .2, 'native': .3, 'decay': .01}[policy] * fraction
                endpoints.append({'batch': batch, 'policy': policy, 'fraction': fraction, 'fraction_id': name,
                       'metrics': {'objectives': dict.fromkeys(runner.KEYS, 10. - finite)},
                       'effects': {k: {'linear': finite / 2, 'linear_decay': .001, 'linear_data': finite / 2 - .001}
                                   for k in runner.KEYS}})
    contrasts = runner.finish_effects(baseline, endpoints)
    return {**row, 'endpoints': endpoints, 'contrasts': contrasts}, baseline


class ProducerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_roster_and_exact_budget(self):
        rows = runner.roster()
        self.assertEqual(len(rows), 12)
        self.assertEqual(len({r['parent_id'] for r in rows}), 12)
        self.assertEqual(rows[0]['step'], 100)
        self.assertEqual(rows[-1]['augmentation'], 'translate')
        inv = runner.byte_inventory()
        self.assertEqual(inv['total_upper_bytes'], 6413406496)
        self.assertEqual(inv['total_upper_bytes'], sum(inv['component_upper_bytes'].values()))
        self.assertLess(inv['total_upper_bytes'] + inv['failure_reserve_bytes'], inv['cap_bytes'])
        self.assertEqual(inv['expected_receipted_artifacts'], 376)

    def test_default_cli_inert(self):
        with mock.patch.object(runner, 'verify_manifest', side_effect=AssertionError('must be inert')), \
             mock.patch.object(runner, 'acquire', side_effect=AssertionError('no model work')), \
             contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(runner.main([]), 0)
        self.assertEqual(json.loads(out.getvalue())['status'], 'inert')

    def test_execute_requires_all_arguments_before_reads(self):
        with mock.patch.object(runner, 'verify_manifest', side_effect=AssertionError('premature read')):
            with self.assertRaisesRegex(ValueError, 'requires output'):
                runner.main(['--execute'])

    def test_pinned_read_hash_size_type_caps(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / 'source'
            payload = b'fabricated numeric input'
            p.write_bytes(payload)
            sha = hashlib.sha256(payload).hexdigest()
            self.assertEqual(runner.read_pinned(p, sha, expected_size=len(payload)), payload)
            for kwargs in ({'expected_size': 1}, {'cap': 1}):
                with self.assertRaises(ValueError):
                    runner.read_pinned(p, sha, **kwargs)
            with self.assertRaises(ValueError):
                runner.read_pinned(p, '0' * 64)
            link = Path(directory) / 'link'
            link.symlink_to(p)
            with self.assertRaises(OSError):
                runner.read_pinned(link, sha)
            fifo = Path(directory) / 'fifo'
            os.mkfifo(fifo)
            with self.assertRaisesRegex(ValueError, 'type/byte cap'):
                runner.read_pinned(fifo, sha)

    def test_receipt_rejects_path_escape(self):
        with self.assertRaisesRegex(ValueError, 'direct input'):
            runner.receipt_bytes('/fabricated', receipt('../escape.npz'))

    def test_numeric_npz_and_empty_basis(self):
        buf = io.BytesIO()
        np.savez(buf, a=np.arange(12, dtype=np.float32), b=np.empty((5, 0), np.float32))
        values = runner.numeric_npz(buf.getvalue())
        np.testing.assert_array_equal(values['a'], np.arange(12, dtype=np.float32))
        self.assertEqual(values['b'].shape, (5, 0))
        with self.assertRaisesRegex(ValueError, 'byte cap'):
            runner.numeric_npz(buf.getvalue(), cap=2)

    def test_numeric_npz_rejects_objects_nonfinite_and_inflation(self):
        for array in (np.array([object()], dtype=object), np.array([float('nan')])):
            buf = io.BytesIO()
            np.savez(buf, value=array)
            with self.assertRaises(ValueError):
                runner.numeric_npz(buf.getvalue())
        header = io.BytesIO()
        np.lib.format.write_array_header_1_0(header, {'descr': '<f8', 'fortran_order': False,
                                                    'shape': (1_000_000_000,)})
        archive = io.BytesIO()
        with zipfile.ZipFile(archive, 'w') as z:
            z.writestr('bad.npy', header.getvalue())
        with self.assertRaisesRegex(ValueError, 'declared shape'):
            runner.numeric_npz(archive.getvalue())

    def test_metadata_chain_and_mutations(self):
        inventory, old, audit = metadata_fixture()
        def load(path, *args, **kwargs):
            return json.dumps(old if Path(path).name == 'results.json' else audit).encode()
        with mock.patch.object(runner, 'read_pinned', side_effect=load):
            self.assertIs(runner.verify_input_metadata(inventory), inventory)
            audit['status'] = 'FAIL'
            with self.assertRaisesRegex(ValueError, 'audit/results'):
                runner.verify_input_metadata(inventory)
            audit['status'] = 'PASS'
            altered = copy.deepcopy(inventory)
            altered['parents'][0]['warmup']['sha256'] = 'f' * 64
            with self.assertRaisesRegex(ValueError, 'checkpoint binding'):
                runner.verify_input_metadata(altered)
            altered = copy.deepcopy(inventory)
            altered['parents'].reverse()
            with self.assertRaisesRegex(ValueError, 'roster'):
                runner.verify_input_metadata(altered)
            old['receipts'].append(old['receipts'][0])
            with self.assertRaisesRegex(ValueError, 'duplicate'):
                runner.verify_input_metadata(inventory)

    def test_required_sources_includes_all_new_code_tests_and_old_pins(self):
        sources = runner.required_sources({'source_pins': {'old.py': '0' * 64}})
        self.assertEqual(len(sources), 19)
        for suffix in ('', '_actions', '_objectives', '_restore', '_panels', '_guard', '_audit'):
            self.assertIn(f'experiments/spectral_component_utility{suffix}.py', sources)
            self.assertIn(f'tests/test_spectral_component_utility{suffix}.py', sources)

    def test_materialization_does_not_change_original_model(self):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(13)
            parent = torch.nn.Linear(3, 2)
        before = torch.cat([p.detach().reshape(-1) for p in parent.parameters()])
        private = copy.deepcopy(parent)
        point = before + .01
        runner.install_point(private, point)
        self.assertTrue(torch.equal(before, torch.cat([p.detach().reshape(-1) for p in parent.parameters()])))
        self.assertTrue(torch.equal(point, torch.cat([p.detach().reshape(-1) for p in private.parameters()])))

    def test_grid_transform_order_and_original_index(self):
        images = np.zeros((600, 784), np.uint8)
        images[:, 14 * 28 + 14] = 255
        panels = {'evaluation_ids': np.arange(256), 'reporting_ids': np.arange(128),
                  'baseline_ids': np.arange(500), 'action_ids': np.arange(128).reshape(2, 64),
                  'action_shifts': np.tile(np.array([1, -2], np.int8), (2, 64, 1)),
                  'view_shifts': np.array([(dy, dx) for dy in range(-2, 3) for dx in range(-2, 3)], np.int8)}
        result = runner.make_inputs(images, panels, 'cpu')
        grid = result['I'].reshape(256, 25, 784)
        self.assertTrue(torch.equal(grid[0, 12], result['R_original'][0]))
        self.assertEqual(int(grid[0, 0].argmax()), 12 * 28 + 12)
        self.assertEqual(int(grid[0, 24].argmax()), 16 * 28 + 16)
        self.assertEqual(int(result['actions'][0][0].argmax()), 15 * 28 + 12)
        self.assertAlmostEqual(float(grid[0, 0, 0]), float(-np.float32(.1307) / np.float32(.3081)), places=7)

    def test_baseline_gradients_preserve_saved_grad_and_release_graph(self):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(97)
            model = torch.nn.Linear(2, 10)
            inputs = {'I': torch.randn(6400, 2), 'R': torch.randn(3200, 2),
                      'R_original': torch.randn(128, 2)}
        for p in model.parameters():
            p.grad = torch.ones_like(p) * .17
        before = [p.grad.clone() for p in model.parameters()]
        panels = {'evaluation_true': np.arange(256, dtype=np.int64) % 10,
                  'evaluation_assigned': (np.arange(256, dtype=np.int64) + 2) % 10,
                  'reporting_true': np.arange(128, dtype=np.int64) % 10}
        check = mock.Mock()
        logits, gradients, metrics = runner.baseline_gradients(model, inputs, panels, check)
        self.assertEqual(logits['I'].shape, (256, 25, 10))
        self.assertEqual(set(gradients), set(runner.KEYS))
        self.assertTrue(all(g.grad_fn is None and not g.requires_grad for g in gradients.values()))
        self.assertTrue(all(torch.equal(p.grad, v) for p, v in zip(model.parameters(), before)))
        self.assertEqual(metrics['metadata']['I_wrong_examples'], 256)
        self.assertEqual(check.call_count, 21)
        from experiments import spectral_component_utility_audit as audit
        audit.Checks().tree(metrics, audit.metrics(logits, panels), 'producer/auditor metric integration')

    def test_actual_fabricated_torch_actions_pass_independent_numpy_audit(self):
        from experiments import spectral_component_utility_audit as audit
        for step in (100, 56304):
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(951)
                model = torch.nn.Sequential(torch.nn.Linear(784, 256), torch.nn.ReLU(),
                           torch.nn.Linear(256, 128), torch.nn.ReLU(), torch.nn.Linear(128, 10))
            model._strong_augmentation_spec = dict(runner.core.MODEL_SPEC)
            optimizer = runner.core.make_optimizer(model)
            tracker = runner.core.make_tracker(model, optimizer)
            for parameter in model.parameters():
                optimizer.state[parameter] = {'step': torch.tensor(float(step)),
                     'exp_avg': torch.full_like(parameter, .001),
                     'exp_avg_sq': torch.full_like(parameter, .01)}
                parameter.grad = torch.full_like(parameter, .003)
            tracker.step_count = step
            tracker.grad_mean = torch.linspace(-.001, .001, runner.P)
            tracker.V = torch.zeros(runner.P, 2)
            tracker.V[0, 0], tracker.V[1, 1] = 1., 1.
            tracker.S = torch.tensor([.03, .02], dtype=torch.float64)
            raw = torch.linspace(-.02, .02, runner.P)
            with mock.patch.object(torch.cuda, '_lazy_init', side_effect=AssertionError('CPU fixture only')):
                records = runner.action_helper.paired_actions((model, optimizer, tracker), raw)
            for record in records:
                arrays = {k: v.numpy() for k, v in record.items() if isinstance(v, torch.Tensor)}
                rank = 0 if record['post_basis'] is None else record['post_basis'].shape[1]
                if rank == 0:
                    arrays['post_basis'] = np.empty((runner.P, 0), np.float32)
                    arrays['post_singular_values'] = np.empty((0,), np.float64)
                metadata = {k: record[k] for k in ('policy', 'observer_steps_before', 'observer_steps_after')}
                metadata['basis_rank'] = rank
                audit.check_action(arrays, metadata, arrays['theta_before'], step, audit.Checks())
                for fraction, _ in runner.FRACTIONS:
                    path = runner.objective.path_accounting(record['theta_before'], record['theta_after'],
                                                            record['decay_endpoint'], fraction)
                    audit.check_path({k: v.numpy() for k, v in path.items() if isinstance(v, torch.Tensor)},
                                     arrays['theta_before'], arrays['theta_after'], arrays['decay_endpoint'],
                                     fraction, audit.Checks())

    def test_effects_decay_and_summary_all_seeds_cells(self):
        parents = [endpoint_fixture(row)[0] for row in runner.roster()]
        for parent in parents:
            for row in parent['endpoints']:
                effect = row['effects']['H_O']
                self.assertAlmostEqual(effect['finite'], effect['finite_decay'] + effect['finite_data'])
                self.assertAlmostEqual(effect['residual'], effect['finite'] - effect['linear'])
                if row['policy'] == 'decay':
                    self.assertEqual(effect['finite_data'], 0.)
        summary = runner.summarize(parents)
        self.assertEqual(len(summary['cells']), 8)
        self.assertEqual(summary['primary']['augmentation'], 'translate')
        self.assertEqual(summary['primary']['step'], 56304)
        self.assertEqual(summary['primary']['fraction'], 1.)
        for index, seed_row in enumerate(summary['primary']['seed_rows']):
            self.assertEqual(set(seed_row['objectives']), {'H_O', 'C'})
            expected = -(4 + index) * 1.5 * .1
            self.assertAlmostEqual(seed_row['objectives']['H_O']['contrast_finite'], expected)
            self.assertAlmostEqual(seed_row['objectives']['H_O']['contrast_linear'], expected / 2)
        with self.assertRaisesRegex(ValueError, 'roster'):
            runner.summarize(parents[:-1])

    def test_mocked_parent_flow_and_baseline_failure_before_actions(self):
        # Explicitly fabricated two-input linear model; every source/restore
        # read is mocked. This is not a production checkpoint or GPU replay.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(351)
            model = torch.nn.Linear(2, 10)
            inputs = {'I': torch.randn(6400, 2), 'R': torch.randn(3200, 2),
                      'R_original': torch.randn(128, 2), 'baseline': torch.randn(500, 2),
                      'actions': [torch.randn(64, 2), torch.randn(64, 2)]}
        with torch.no_grad():
            for p in model.parameters():
                p.zero_()
                p.grad = torch.ones_like(p) * .125
        before = [p.grad.clone() for p in model.parameters()]
        panels = {'evaluation_true': np.arange(256, dtype=np.int64) % 10,
                  'evaluation_assigned': (np.arange(256, dtype=np.int64) + 1) % 10,
                  'reporting_true': np.arange(128, dtype=np.int64) % 10,
                  'action_assigned': np.arange(128, dtype=np.int64).reshape(2, 64) % 10,
                  'baseline_ids': np.arange(500, dtype=np.int64)}
        source = {'warmup': receipt('warmup.pt'), 'warmup_readout': receipt('readout.npz')}
        saved = {'rng': 'fabricated-RNG-placeholder'}
        original_readout = {'step': np.array([100], np.int64), 'train': np.zeros(1, np.float32),
                            'validation': np.zeros(1, np.float32), 'reporting': np.zeros((5000, 10), np.float32)}
        artifacts = {}
        fake_run = mock.Mock(device='cpu')
        def save(name, value, kind='json'):
            artifacts[name] = copy.deepcopy(value)
            return receipt(name)
        fake_run.save.side_effect = save
        def pair(parent, gradient):
            theta = runner.core.flat_params(parent[0]).cpu()
            return [{'policy': policy, 'theta_before': theta.clone(),
                     'theta_after': theta + (.001 if policy == 'raw' else .002),
                     'decay_endpoint': theta.clone(), 'gradient_raw': gradient.clone(),
                     'gradient_delivered': gradient.clone(), 'm_before': theta.clone(),
                     'v_before': theta.clone(), 'm_after': theta.clone(), 'v_after': theta.clone(),
                     'adam_steps_before': torch.full((6,), 100, dtype=torch.int64),
                     'adam_steps_after': torch.full((6,), 101, dtype=torch.int64),
                     'observer_steps_before': 100, 'observer_steps_after': 100 + (policy == 'native'),
                     'post_basis': None, 'post_singular_values': None}
                    for policy in ('raw', 'native')]
        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.object(runner, 'P', 30))
            stack.enter_context(mock.patch.object(runner.restore_helper, 'load_snapshot', return_value=saved))
            stack.enter_context(mock.patch.object(runner.restore_helper, 'restore', return_value=(model, None, None)))
            stack.enter_context(mock.patch.object(runner.restore_helper, 'snapshot_with_rng', return_value=saved))
            stack.enter_context(mock.patch.object(runner, 'receipt_bytes', return_value=b'fabricated'))
            stack.enter_context(mock.patch.object(runner, 'numeric_npz', return_value=original_readout))
            paired = stack.enter_context(mock.patch.object(runner.action_helper, 'paired_actions', side_effect=pair))
            result = runner.acquire_parent(fake_run, runner.roster()[0], source,
                                          {'archive': '/fabricated/archive'}, panels, inputs)
            self.assertEqual(len(artifacts), 31)
            self.assertEqual(paired.call_count, 2)
            self.assertEqual(len(result['endpoints']), 12)
            self.assertEqual(len(result['actions']), 4)
            self.assertEqual(len(result['contrasts']), 4)
            self.assertEqual(result['parent_digest_before'], result['parent_digest_after'])
            self.assertTrue(all(torch.equal(p.grad, v) for p, v in zip(model.parameters(), before)))
            self.assertEqual(artifacts['action-s202609171-none-h00100-b0-raw.npz']['post_basis'].shape, (30, 0))
            # Mismatch is saved, then fails before objective/action work.
            artifacts.clear()
            paired.reset_mock()
            original_readout['reporting'][0, 0] = 1.
            with self.assertRaisesRegex(ValueError, 'prediction mismatch'):
                runner.acquire_parent(fake_run, runner.roster()[0], source,
                                      {'archive': '/fabricated/archive'}, panels, inputs)
            paired.assert_not_called()
            self.assertEqual(list(artifacts), ['baseline-check-s202609171-none-h00100.npz'])

    def test_bad_metric_identity_rejected(self):
        metrics = {'objectives': {'L': 1., 'S': .3, 'F': .2, 'C': .5, 'S_true': 3., 'S_uniform': 0.}}
        runner.check_identity(metrics)
        metrics['objectives']['C'] = .4
        with self.assertRaisesRegex(ValueError, 'S\\+F\\+C'):
            runner.check_identity(metrics)


if __name__ == '__main__':
    unittest.main()
