#!/usr/bin/env python3
"""I20 admission/serialization fixtures, never scientific parent acquisition."""
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np

import run_response as run


class RunnerTests(unittest.TestCase):
    def test_roster(self):
        cells = run.expected_cells()
        self.assertEqual(len(cells), 192)
        self.assertEqual(len(set(cells)), 192)
        self.assertEqual(cells[0], (19000, 0, 0))
        self.assertEqual(cells[-1], (19031, 2, 1))
        self.assertEqual({cell[0] for cell in cells}, set(range(19000, 19032)))

    def test_json_exclusive_nonfinite_duplicate(self):
        with tempfile.TemporaryDirectory(prefix='i20-json-fixture-') as directory:
            path = Path(directory) / 'a.json'
            run.json_once(path, {'ok': True})
            self.assertEqual(run.read_json(path), {'ok': True})
            with self.assertRaises(FileExistsError):
                run.json_once(path, {})
            bad = Path(directory) / 'bad.json'
            with self.assertRaises(ValueError):
                run.json_once(bad, {'bad': float('nan')})
            self.assertFalse(bad.exists())
            bad.write_text('{"a":1,"a":2}')
            with self.assertRaisesRegex(RuntimeError, 'Duplicate'):
                run.read_json(bad)
            bad.write_text('{"a":NaN}')
            with self.assertRaises(ValueError):
                run.read_json(bad)

    def test_regular_rejects_symlink(self):
        with tempfile.TemporaryDirectory(prefix='i20-link-fixture-') as directory:
            source, link = Path(directory) / 'source', Path(directory) / 'link'
            source.write_bytes(b'fixture')
            link.symlink_to(source)
            run.regular(source)
            with self.assertRaises(RuntimeError):
                run.regular(link)

    def test_configuration(self):
        valid = {'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '1',
                 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
        with patch.dict(os.environ, valid, clear=True):
            run.configure()
        for name in valid:
            with patch.dict(os.environ, dict(valid, **{name: '2'}), clear=True):
                with self.assertRaises(RuntimeError):
                    run.configure()

    def test_root_wrong_path_never_admitted(self):
        with tempfile.TemporaryDirectory(prefix='i20-root-fixture-') as directory:
            with self.assertRaisesRegex(RuntimeError, 'Unexpected root'):
                run.preflight_root(Path(directory))

    def test_source_freeze_rejects_changed_current(self):
        with tempfile.TemporaryDirectory(prefix='i20-source-fixture-') as directory:
            root = Path(directory)
            source = root / 'core.py'
            source.write_bytes(b'original')
            with patch.object(run, 'HERE', root), patch.object(run, 'REPO', root), \
                 patch.object(run, 'SOURCE_NAMES', ('core.py',)), \
                 patch.object(run.subprocess, 'check_output', return_value=b'original'):
                rows = run.source_manifest('a' * 40)
                self.assertEqual(rows[0]['sha256'], hashlib.sha256(b'original').hexdigest())
                source.write_bytes(b'changed')
                with self.assertRaisesRegex(RuntimeError, 'Changed source'):
                    run.source_manifest('a' * 40)
                with self.assertRaises(RuntimeError):
                    run.source_manifest('short')

    def test_inventory_exact_and_hash_bound(self):
        with tempfile.TemporaryDirectory(prefix='i20-inventory-fixture-') as directory:
            root = Path(directory)
            streams = root / 'streams'
            streams.mkdir()
            rows = []
            for seed, pi, ri in ((19000, 0, 0), (19000, 0, 1)):
                identity = run.cell_id(seed, pi, ri)
                row = {'id': identity}
                for key, extension in (('array', '.npz'), ('metadata', '.json')):
                    path = streams / (identity + extension)
                    path.write_bytes(b'fixture')
                    row[key] = {'path': str(path.relative_to(root)), 'size': path.stat().st_size,
                                'sha256': run.sha(path)}
                rows.append(row)
            with patch.object(run, 'expected_cells', return_value=[(19000, 0, 0), (19000, 0, 1)]):
                run.inventory_files(root, rows)
                with self.assertRaises(RuntimeError):
                    run.inventory_files(root, rows[::-1])
                extra = streams / 'extra'
                extra.write_bytes(b'x')
                with self.assertRaisesRegex(RuntimeError, 'Extra/missing'):
                    run.inventory_files(root, rows)
                extra.unlink()
                (root / rows[0]['array']['path']).write_bytes(b'changed')
                with self.assertRaisesRegex(RuntimeError, 'hash/size'):
                    run.inventory_files(root, rows)

    def test_numeric_archive_exact_members_and_exclusive(self):
        import response_core as core
        shapes = core.expected_shapes(run.HORIZON)
        arrays = {name: np.zeros(shape, dtype=bool if name == 'direction_present' else np.float64)
                  for name, shape in shapes.items()}
        self.assertEqual(sum(a.nbytes for a in arrays.values()), run.HORIZON * 1062)
        self.assertLess(192 * (run.HORIZON * 1062 + 4096), run.ARRAY_LIMIT)
        with tempfile.TemporaryDirectory(prefix='i20-archive-fixture-') as directory:
            root = Path(directory)
            (root / 'streams').mkdir()
            row = run.write_stream(root, '19000-p0-r0', arrays, {'fixture': True}, core)
            with zipfile.ZipFile(root / row['array']['path']) as archive:
                self.assertEqual(archive.namelist(), [name + '.npy' for name in core.ARRAY_ORDER])
                self.assertTrue(all(i.compress_type == zipfile.ZIP_STORED for i in archive.infolist()))
            with np.load(root / row['array']['path'], allow_pickle=False) as archive:
                self.assertEqual(archive.files, list(core.ARRAY_ORDER))
                self.assertNotIn('allow_pickle', archive.files)
            with self.assertRaisesRegex(RuntimeError, 'Consumed'):
                run.write_stream(root, '19000-p0-r0', arrays, {}, core)
            mutated = dict(arrays)
            mutated['weighted_gap'] = np.zeros(shapes['weighted_gap'], dtype=object)
            with self.assertRaisesRegex(RuntimeError, 'numeric array'):
                run.write_stream(root, '19000-p0-r1', mutated, {}, core)
            self.assertFalse((root / 'streams/19000-p0-r1.npz').exists())
            mutated = dict(arrays)
            mutated['weighted_gap'] = np.full(shapes['weighted_gap'], np.nan)
            with self.assertRaises(RuntimeError):
                run.write_stream(root, '19000-p0-r1', mutated, {}, core)

    def test_failed_attempt_retains_prefix_and_completion(self):
        with tempfile.TemporaryDirectory(prefix='i20-failure-fixture-') as directory:
            root = Path(directory)
            with patch.object(run, 'configure'), patch.object(run, 'source_manifest', return_value=[]), \
                 patch.object(run, 'preflight_root'), patch.object(run, 'verify_parent', return_value=({}, [{}] * 192)), \
                 patch.object(run, 'load_parent', side_effect=RuntimeError('fixture input failure')), \
                 patch('sys.stdout', new_callable=io.StringIO):
                self.assertEqual(run.acquire(root, 'a' * 40), 1)
            completion = run.read_json(root / 'completion.json')
            self.assertEqual(completion['status'], 'failed')
            self.assertEqual(completion['completed_streams'], 0)
            self.assertEqual(completion['failure']['message'], 'fixture input failure')
            self.assertEqual(completion['attempt_sha256'], run.sha(root / 'attempt.json'))
            with patch.object(run, 'configure'), patch.object(run, 'source_manifest', return_value=[]), \
                 patch.object(run, 'preflight_root'), patch.object(run, 'verify_parent', return_value=({}, [{}] * 192)):
                with self.assertRaises(FileExistsError):
                    run.acquire(root, 'a' * 40)

    def test_parent_closures_all_controls_and_corruption(self):
        import response_core as core
        parent_names = ['native_cp', 'native_cp_star', 'oracle_useful',
                        'oracle_useful_star', 'oracle_nuisance', 'ema_q0p9']
        arrays = {'output': np.zeros((3, 36, 2), dtype=np.float64)}
        inputs = {'output': np.zeros((3, 6, 2), dtype=np.float64)}
        run.check_parent_closures(arrays, inputs, parent_names, core.POLICIES, core.ESTIMATORS)
        names = ['native/r0p9/cp', 'native/rstar/cp', 'oracle_useful/r0p9/cp',
                 'oracle_useful/rstar/cp', 'oracle_nuisance/r0p9/cp']
        names += [name + '/r0p9/rec9' for name in core.ESTIMATORS]
        for name in names:
            changed = {'output': arrays['output'].copy()}
            changed['output'][1, core.POLICIES.index(name), 0] = .01
            with self.assertRaisesRegex(RuntimeError, 'closure differs'):
                run.check_parent_closures(changed, inputs, parent_names, core.POLICIES, core.ESTIMATORS)

    def test_cooperative_expiry_records_failure_without_loading(self):
        with tempfile.TemporaryDirectory(prefix='i20-expiry-fixture-') as directory:
            root = Path(directory)
            with patch.object(run, 'configure'), patch.object(run, 'source_manifest', return_value=[]), \
                 patch.object(run, 'preflight_root'), patch.object(run, 'verify_parent', return_value=({}, [{}] * 192)), \
                 patch.object(run, 'load_parent') as loader, \
                 patch.object(run.time, 'monotonic', side_effect=[0.0, 541.0, 542.0]), \
                 patch('sys.stdout', new_callable=io.StringIO):
                self.assertEqual(run.acquire(root, 'a' * 40), 1)
                loader.assert_not_called()
            result = run.read_json(root / 'completion.json')
            self.assertEqual(result['status'], 'failed')
            self.assertEqual(result['completed_streams'], 0)
            self.assertIn('Cooperative', result['failure']['message'])

    def test_parent_archive_rejects_extra_member_before_numeric_load(self):
        with tempfile.TemporaryDirectory(prefix='i20-parent-fixture-') as directory:
            root = Path(directory)
            (root / 'streams').mkdir()
            array_path = root / 'streams/19000-p0-r0.npz'
            meta_path = root / 'streams/19000-p0-r0.json'
            arrays = {name: np.zeros((2, 2), dtype=np.float64) for name in run.INPUT_NAMES}
            metadata = {'id': '19000-p0-r0', 'horizon': 2, 'core': {
                'array_order': list(arrays), 'array_shapes': {name: [2, 2] for name in arrays},
                'array_dtypes': {name: 'float64' for name in arrays}}}
            with array_path.open('wb') as handle:
                np.savez(handle, **arrays)
            run.json_once(meta_path, metadata)
            row = {'id': metadata['id'], 'array': {'path': str(array_path.relative_to(root)), 'sha256': run.sha(array_path)},
                   'metadata': {'path': str(meta_path.relative_to(root)), 'sha256': run.sha(meta_path)}}
            with patch.object(run, 'PARENT', root), patch.object(run, 'HORIZON', 2):
                loaded, _ = run.load_parent(row)
                self.assertEqual(list(loaded), list(run.INPUT_NAMES))
                with array_path.open('wb') as handle:
                    np.savez(handle, **arrays, unwanted=np.zeros(1))
                row['array']['sha256'] = run.sha(array_path)
                with self.assertRaisesRegex(RuntimeError, 'ZIP membership'):
                    run.load_parent(row)


if __name__ == '__main__':
    unittest.main()
