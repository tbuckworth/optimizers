"""Fabricated arrays and mocked updates only: no model, IDX, or acquisition."""
import io
import json
import unittest
from pathlib import Path
from unittest.mock import patch, Mock

import numpy as np
import torch

from experiments import spectral_augmentation_boundary as b


class ContractTests(unittest.TestCase):
    def test_roster_and_inventory(self):
        roster = b.branch_roster()
        self.assertEqual(len(roster), 72)
        self.assertEqual(len({tuple(r.values()) for r in roster}), 72)
        self.assertEqual(roster[0], {'seed': 202609131, 'cell': 'clean',
                                    'augmentation': 'none', 'policy': 'raw'})
        for seed in b.SEEDS:
            rows = [r for r in roster if r['seed'] == seed]
            self.assertEqual(len(rows), 24)
            for i in range(0, 24, 2):
                left, right = rows[i:i+2]
                self.assertEqual({left['policy'], right['policy']}, {'raw', 'native32'})
                self.assertEqual((left['cell'], left['augmentation']),
                                 (right['cell'], right['augmentation']))
        inventory = b.byte_inventory()
        self.assertEqual(inventory['total_upper_bytes'], sum(inventory['component_upper_bytes'].values()))
        self.assertLess(inventory['total_upper_bytes'] + b.RESERVE_BYTES, 2 * 1024**3)
        self.assertEqual(inventory['component_upper_bytes']['137100_action_rows_1024_bytes_each'],
                         (72 * 1900 + 3 * 100) * 1024)

    def test_inert_cli_refuses_without_execution_and_no_overrides(self):
        with patch.object(b, 'acquire', side_effect=AssertionError('acquisition forbidden')), \
             patch.object(b, 'configure', side_effect=AssertionError('GPU forbidden')), \
             patch.object(b.base, 'read_training', side_effect=AssertionError('IDX forbidden')):
            with self.assertRaisesRegex(RuntimeError, 'without explicit'):
                b.main(['--output-dir', '/not/a/real/path'])
            for flag in ('--seed', '--steps', '--batch', '--resume'):
                with self.assertRaises(SystemExit), patch('sys.stderr', new=io.StringIO()):
                    b.parse_args(['--output-dir', '/unused', flag, '1'])

    def test_cell_cue_then_occurrence_erasure(self):
        # Same fabricated example repeated in the batch; one occurrence erased.
        clean = torch.full((2, 784), .25, dtype=torch.float32)
        cell_cpu = b.base.patch_images(clean, np.array([True, False])).numpy()
        before = cell_cpu.copy()
        batch = np.array([0, 0, 1], dtype=np.int64)
        rectangles = np.array([[0, 8, 0, 8], [-1, -1, -1, -1], [20, 28, 20, 28]], dtype=np.int16)
        result = b.augmented_batch(cell_cpu, batch, rectangles, 'cpu')
        self.assertEqual(result.dtype, torch.float32)
        pixels = result.numpy().reshape(3, 28, 28)
        self.assertTrue(np.all(pixels[0, :8, :8] == 0))
        self.assertTrue(np.all(pixels[1, :3, :3] == 1))
        self.assertTrue(np.all(pixels[2, 20:, 20:] == 0))
        np.testing.assert_array_equal(cell_cpu, before)
        self.assertFalse(np.shares_memory(result.numpy(), cell_cpu))
        for invalid in (batch.astype(float), np.array([-1]), np.array([2])):
            with self.assertRaises(RuntimeError):
                b.augmented_batch(cell_cpu, invalid, rectangles, 'cpu')
        with self.assertRaises(RuntimeError):
            b.augmented_batch(cell_cpu.astype(np.float64), batch, rectangles, 'cpu')

    def test_mocked_update_reuses_base_contract_without_diagnostics(self):
        cpu = np.ones((2, 784), dtype=np.float32)
        assigned = np.array([0, 8], dtype=np.int64)
        batch = np.array([1, 0], dtype=np.int64)
        rect = np.array([[0, 8, 0, 8], [-1, -1, -1, -1]], dtype=np.int16)
        marker, check = {'step': 101, 'raw_gradient_sha256': 'fabricated'}, Mock()
        with patch.object(b.base, 'training_update', return_value=(marker, None)) as update:
            result = b.continuation_update('model', 'optimizer', 'tracker', cpu, assigned,
                batch, rect, 'native32', 101, 'cpu', check)
        self.assertIs(result, marker)
        args, kwargs = update.call_args
        self.assertEqual(args[:3], ('model', 'optimizer', 'tracker'))
        np.testing.assert_array_equal(args[4].numpy(), [8, 0])
        self.assertEqual(args[5:], ('native32', 101))
        self.assertEqual(kwargs, {'definitions': None, 'identity': None, 'check': check})
        self.assertEqual(int((args[3][0] == 0).sum()), 64)
        with patch.object(b.base, 'training_update', return_value=(marker, ('forbidden',))):
            with self.assertRaisesRegex(RuntimeError, 'diagnostics forbidden'):
                b.continuation_update(None, None, None, cpu, assigned, batch, rect,
                                      'raw', 101, 'cpu', check)

    def test_occurrence_denominators_and_empty_cued_group(self):
        batches = np.array([[0, 0, 1], [1, 2, 0]], dtype=np.int64)
        gates = np.array([[True, False, True], [False, True, True]])
        plan = b.masks.MaskPlan(gates, np.zeros((2, 3, 2), dtype=np.int16))
        rect = b.masks.rectangles_for_mode(plan, 'targeted')
        result = b.occurrence_coverage(rect, gates, batches,
                                      np.array([True, False, False]), np.array([1, 8, 0]))
        self.assertEqual(result['all']['denominator'], 6)
        self.assertEqual(result['actually_cued']['denominator'], 3)
        self.assertEqual(result['actually_cued']['full_cue_coverage'], 2)
        self.assertEqual(result['rare']['denominator'], 2)
        self.assertEqual(result['rare']['active'], 1)
        empty = b.occurrence_coverage(rect, gates, batches,
                                     np.zeros(3, dtype=bool), np.array([1, 8, 0]))
        self.assertEqual(empty['actually_cued']['denominator'], 0)
        self.assertEqual(sum(empty['actually_cued']['erased_area_counts']), 0)
        json.dumps(empty, allow_nan=False)

    def test_full_snapshot_restore_check_precedes_tracker_removal(self):
        snapshot = {'fabricated': 1}
        with patch.object(b.core, 'restore', return_value=('m', 'o', 't')) as restore, \
             patch.object(b.core, 'snapshot', return_value=snapshot), \
             patch.object(b.core, 'tree_digest', return_value='same'):
            self.assertEqual(b.verified_restore(snapshot, 'same', 'cpu'), ('m', 'o', 't'))
            restore.assert_called_once_with(snapshot, 'cpu')
            with self.assertRaisesRegex(RuntimeError, 'fork identity'):
                b.verified_restore(snapshot, 'different', 'cpu')


class ResourceTests(unittest.TestCase):
    def test_capped_memory_writer_and_partial_write(self):
        handle = io.BytesIO()
        check = Mock()
        writer = b.CappedWriter(handle, 4, check)
        self.assertEqual(writer.write(b'abcd'), 4)
        self.assertEqual(writer.allowance, 0)
        with self.assertRaisesRegex(RuntimeError, 'byte cap'):
            writer.write(b'e')
        self.assertEqual(handle.getvalue(), b'abcd')
        short = Mock()
        short.write.return_value = 1
        with self.assertRaisesRegex(RuntimeError, 'short'):
            b.CappedWriter(short, 4).write(b'abc')

    def test_numpy_archive_through_capped_writer_in_memory(self):
        handle = io.BytesIO()
        arrays = {'steps': np.array([0, 100]), 'train': np.zeros((2, 3, 10), dtype=np.float32)}
        np.savez(b.CappedWriter(handle, 10000), **arrays)
        handle.seek(0)
        with np.load(handle, allow_pickle=False) as saved:
            for key, expected in arrays.items():
                np.testing.assert_array_equal(saved[key], expected)

    def test_gpu_and_cgroup_admission_fixtures(self):
        self.assertEqual(len(b.validate_gpu_clients('2101, 260\n8861, 27\n')), 2)
        self.assertEqual(b.validate_gpu_clients(''), [])
        for text in ('9999, 1', '2101, 513', '8861, -1', '2101, N/A', 'not csv'):
            with self.assertRaises((RuntimeError, ValueError)):
                b.validate_gpu_clients(text)
        effective = {'memory.max': str(16 * 1024**3), 'memory.swap.max': '0', 'cpu.max': '100000 100000'}
        service = {'Type': 'exec', 'RuntimeMaxUSec': '30min', 'Restart': 'no', 'KillMode': 'control-group'}
        b.base.validate_bounds(effective, service)
        for key, value in (('memory.max', 'max'), ('memory.swap.max', '1'), ('cpu.max', '200000 100000')):
            with self.assertRaises(RuntimeError):
                b.base.validate_bounds({**effective, key: value}, service)
        with self.assertRaises(RuntimeError):
            b.base.validate_bounds(effective, {**service, 'Restart': 'always'})

    def test_run_cooperative_deadline_without_files_or_gpu(self):
        run = b.Run(Path('/unused'), device='cpu')
        with patch.object(b.time, 'monotonic', return_value=run.started + b.DEADLINE_SECONDS):
            with self.assertRaisesRegex(RuntimeError, 'deadline'):
                run.check()


if __name__ == '__main__':
    unittest.main()
