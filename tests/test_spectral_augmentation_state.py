"""Fabricated CPU fixtures only: no datasets, checkpoints, or model execution."""
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from experiments import spectral_augmentation_state as runner


class StateRunnerFixtures(unittest.TestCase):
    def source(self):
        return {'train_ids': np.arange(5000, dtype=np.int64),
                'eval_ids': np.arange(5000, 10000, dtype=np.int64)}

    def test_plan_exact_streams_and_pairing(self):
        seed = runner.SEEDS[0]
        source = self.source()
        labels = np.arange(10000, dtype=np.int64) % 10
        old = labels.copy()
        a, b = (runner.diagnostic_plan(seed, source, labels) for _ in range(2))
        self.assertEqual(set(a), {'train_positions', 'eval_positions', 'train_ids', 'eval_ids',
                                 'train_labels', 'eval_labels', 'train_shifts', 'eval_shifts'})
        for key in a:
            np.testing.assert_array_equal(a[key], b[key])
        rng = lambda stream: np.random.Generator(np.random.PCG64(np.random.SeedSequence([stream, seed])))
        np.testing.assert_array_equal(a['train_positions'], rng(10).permutation(5000)[:64])
        np.testing.assert_array_equal(a['eval_positions'], rng(12).permutation(5000)[:256])
        np.testing.assert_array_equal(a['train_shifts'], rng(11).integers(-2, 3, (4,64,2), dtype=np.int8))
        np.testing.assert_array_equal(a['eval_shifts'], rng(13).integers(-2, 3, (256,2), dtype=np.int8))
        self.assertEqual(len(np.unique(a['train_ids'])), 64)
        self.assertEqual(len(np.unique(a['eval_ids'])), 256)
        self.assertFalse(np.intersect1d(a['train_ids'], a['eval_ids']).size)
        np.testing.assert_array_equal(labels, old)
        self.assertFalse(np.shares_memory(a['train_ids'], source['train_ids']))

    def test_invalid_plans(self):
        labels = np.arange(10000, dtype=np.int64) % 10
        for seed in (True, -1, 0, 202609141.):
            with self.assertRaises(RuntimeError):
                runner.diagnostic_plan(seed, self.source(), labels)
        source = self.source()
        source['eval_ids'][0] = source['train_ids'][0]
        with self.assertRaises(RuntimeError):
            runner.diagnostic_plan(runner.SEEDS[0], source, labels)
        with self.assertRaises(RuntimeError):
            runner.diagnostic_plan(runner.SEEDS[0], self.source(), labels.astype(np.float32))

    def test_restricted_load_hash_before_deserialization(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'fake.pt'
            path.write_bytes(b'fabricated bytes, not a checkpoint')
            receipt = {'path': path.name, 'size_bytes': path.stat().st_size,
                       'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
            sentinel = object()
            with patch.object(runner.torch, 'load', return_value=sentinel) as load, patch.object(runner, 'validate_snapshot') as validate:
                self.assertIs(runner.load_parent(temp, receipt), sentinel)
                self.assertEqual(load.call_args.kwargs, {'weights_only': True, 'map_location': 'cpu'})
                validate.assert_called_once_with(sentinel)
            receipt['sha256'] = '0'*64
            with patch.object(runner.torch, 'load') as load:
                with self.assertRaises(RuntimeError):
                    runner.load_parent(temp, receipt)
                load.assert_not_called()
            receipt['path'] = '../fake.pt'
            with self.assertRaises(RuntimeError):
                runner.receipt_bytes(temp, receipt)

    def test_receipt_symlink_and_cap(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'fake'
            path.write_bytes(b'x')
            receipt = {'path': 'fake', 'size_bytes': 1, 'sha256': hashlib.sha256(b'x').hexdigest()}
            with self.assertRaises(RuntimeError):
                runner.receipt_bytes(temp, receipt, cap=0)
            (Path(temp)/'link').symlink_to(path)
            with self.assertRaises(RuntimeError):
                runner.receipt_bytes(temp, {**receipt, 'path': 'link'})

    def test_no_execute_no_io(self):
        with patch.object(runner, 'configure') as configure, patch.object(runner, 'source_pins') as pins:
            with self.assertRaisesRegex(RuntimeError, 'explicit --execute'):
                runner.main(['--output-dir', '/not/used/acquisition-001'])
            configure.assert_not_called()
            pins.assert_not_called()

    def test_inventory_and_guards(self):
        inventory = runner.byte_inventory()
        self.assertLess(inventory['per_parent_upper_bytes'], 128*1024**2)
        self.assertLess(inventory['total_upper_bytes'], 1024**3)
        effective = {'memory.max': str(16*1024**3), 'memory.swap.max': '0', 'cpu.max': '100000 100000'}
        service = {'Type': 'exec', 'RuntimeMaxUSec': '20min', 'Restart': 'no', 'KillMode': 'control-group'}
        runner.base.validate_bounds(effective, service)
        with self.assertRaises(RuntimeError):
            runner.base.validate_bounds(effective, {**service, 'Restart': 'on-failure'})
        with self.assertRaises(RuntimeError):
            runner.base.validate_bounds({**effective, 'memory.swap.max': '1'}, service)
        self.assertEqual(runner.UNIT, 'spectral-augmentation-state-001.service')

    def test_snapshot_bad_schema(self):
        for value in (None, {}, {'schema': 'arbitrary'}):
            with self.assertRaises(RuntimeError):
                runner.validate_snapshot(value)

    def test_source_status_binding(self):
        registry = {'input_dir': '/fabricated', 'source_results_sha256': 'abc', 'receipts': []}
        results = {'schema': runner.base.SCHEMA, 'status': 'complete', 'source_pins': {}, 'receipts': []}
        audit = {'status': 'PASS', 'input_dir': '/fabricated', 'input_results_sha256': 'abc'}
        attempt = {'schema': runner.base.SCHEMA, 'output_dir': '/fabricated', 'source_pins': {}}
        runner.validate_source_metadata(registry, results, audit, attempt)
        with self.assertRaises(RuntimeError):
            runner.validate_source_metadata(registry, results, {**audit, 'status': 'FAIL'}, attempt)
        with self.assertRaises(RuntimeError):
            runner.validate_source_metadata(registry, {**results, 'status': 'failed'}, audit, attempt)
        with self.assertRaises(RuntimeError):
            runner.validate_source_metadata(registry, results, audit, {**attempt, 'output_dir': '/elsewhere'})

    def test_fabricated_snapshot_validation(self):
        torch = runner.torch
        shapes = {'0.weight': (64,784), '0.bias': (64,), '2.weight': (10,64), '2.bias': (10,)}
        tracker = {'step_count': 100, 'warmup': 100, 'rank': 32, 'stable_update': True,
                   'decay': .99, 'filter_strength': 1., 'energy_threshold': None,
                   'adaptive': 'none', 'normalize': 'none', 'weighting': 'hard',
                   'alpha': 1., 'soft_residual': True, 'relative_eig_tol': 1e-8,
                   'absolute_eig_floor': 0., 'stabilize_every': 100,
                   'V': torch.zeros(runner.P,32), 'S': torch.zeros(32),
                   'grad_mean': torch.zeros(runner.P)}
        group = {'lr': .001, 'weight_decay': .01, 'betas': (.9,.999), 'eps': 1e-8,
                 'amsgrad': False, 'maximize': False, 'foreach': False, 'fused': False,
                 'params': list(range(4))}
        value = {'schema': 'i9_neural_snapshot_v1',
                 'model_spec': {'input_dim': 784, 'width': 64, 'classes': 10},
                 'model_state': {k: torch.zeros(s) for k,s in shapes.items()},
                 'gradients': [None]*4, 'model_modes': [(k,True) for k in ('','0','1','2')],
                 'optimizer': {'state': {i: {'step': torch.tensor(100.), 'exp_avg': torch.zeros(s),
                                            'exp_avg_sq': torch.zeros(s)} for i,s in enumerate(shapes.values())},
                               'param_groups': [group]},
                 'tracker': tracker, 'rng': {'python': (), 'numpy': {},
                                            'torch_cpu': torch.zeros(8,dtype=torch.uint8), 'torch_cuda': []}}
        runner.validate_snapshot(value)
        value['optimizer']['state'][0]['step'] = torch.tensor(101.)
        with self.assertRaisesRegex(RuntimeError, '100'):
            runner.validate_snapshot(value)
        value['optimizer']['state'][0]['step'] = torch.tensor(100.)
        value['model_state']['0.bias'][0] = float('nan')
        with self.assertRaises(RuntimeError):
            runner.validate_snapshot(value)


if __name__ == '__main__':
    unittest.main()
