"""Fabricated saved-array checker tests; never reads measured artifacts."""
import copy
import json
import importlib.util
import os
from pathlib import Path
import unittest
from unittest.mock import patch, Mock

import numpy as np
import test_forward as tf

SPEC = importlib.util.spec_from_file_location('natural_addon_check_fixture', Path(__file__).with_name('check.py'))
c = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(c)
f = tf.f


def fabricated():
    rows, pairs = tf.data()
    u = np.eye(1024, 4, dtype=np.float32); mean = np.arange(1024, dtype=np.float64)/32
    h = np.zeros((32, 1, 1024), dtype=np.float32)
    h[::2, :, :4] = 1
    scores = h[:, :, :4].astype(np.float64)-mean[:4]
    gaps = scores[::2]-scores[1::2]
    roles = ['prefix_end']
    exports = {'scores': {'schema': 'jlens_natural_addon_scores_v1', 'locations': {
        name: [{'axis': f'PC{a+1}', 'values': {r['id']: float(scores[i, j, a]) for i, r in enumerate(rows)}}
               for a in range(4)] for j, name in enumerate(roles)}},
        'gaps': {'locations': roles, 'axes': ['PC1', 'PC2', 'PC3', 'PC4'], 'orientation': 'left minus right',
                 'pairs': [{**pair, 'gaps64': gaps[i].tolist()} for i, pair in enumerate(pairs)]}}
    return [h, u, mean, scores, gaps, np.array(roles), rows, pairs, exports]


class PureTests(unittest.TestCase):
    def test_helper_pin_before_import(self):
        with patch.object(c, 'sha', return_value='0'*64), \
             patch.object(c.importlib.util, 'spec_from_file_location', side_effect=AssertionError('import before pin')) as loader:
            with self.assertRaises(ValueError): c.load_helper()
            loader.assert_not_called()

    def test_gate_pin_before_import(self):
        with patch.object(c, 'sha', return_value='0'*64), \
             patch.object(c.importlib.util, 'spec_from_file_location', side_effect=AssertionError('gate import before pin')) as loader:
            with self.assertRaises(ValueError): c.reader_gate('p', '1'*64, 'r', '2'*64, '3'*40)
            loader.assert_not_called()

    def test_gate_public_binding_and_exact_call(self):
        gate = Mock()
        gate.FROZEN = {'dataset': '4'*64}
        inputs = {'dataset': {'sha256': '4'*64}, 'preflight': {'sha256': '5'*64},
                  'tokens': {'sha256': '6'*64}}
        gate.verify_response_lock.return_value = {'manifest': {'inputs': inputs}}
        spec = Mock()
        with patch.object(c, 'JUDGING_SHA', 'a'*64), patch.object(c, 'sha', return_value='a'*64), \
             patch.object(c.importlib.util, 'spec_from_file_location', return_value=spec), \
             patch.object(c.importlib.util, 'module_from_spec', return_value=gate):
            value = c.reader_gate('/fabricated/p', '1'*64, '/fabricated/r', '2'*64, '3'*40)
            self.assertEqual('5'*64, value['preflight_sha256'])
            gate.verify_response_lock.assert_called_once_with(packets='/fabricated/p', manifest_sha256='1'*64,
                responses='/fabricated/r', lock_sha256='2'*64, lock_commit='3'*40)
            inputs['dataset']['sha256'] = '0'*64
            with self.assertRaises(ValueError): c.reader_gate('/fabricated/p', '1'*64, '/fabricated/r', '2'*64, '3'*40)

    def test_all_positive_and_exact_zero(self):
        args = fabricated(); result = c.corroborate(*args)
        self.assertEqual(16, result['summaries']['prefix_end'][3]['positive'])
        self.assertEqual(128, result['scalar_checks']); self.assertEqual(64, result['gap_checks'])
        args[0][0, 0, 3] = 0; args[3][0, 0, 3] = -args[2][3]; args[4][0, 0, 3] = 0
        args[8]['scores']['locations']['prefix_end'][3]['values'][f.IDS[0]] = float(args[3][0, 0, 3])
        args[8]['gaps']['pairs'][0]['gaps64'] = args[4][0].tolist()
        result = c.corroborate(*args)
        self.assertEqual(15, result['summaries']['prefix_end'][3]['positive'])
        self.assertEqual(1, result['summaries']['prefix_end'][3]['zero'])

    def test_bad_arrays_and_json_joins(self):
        for kind in ['shape', 'dtype', 'nan', 'score', 'gap', 'roles', 'axis', 'json', 'pair']:
            args = fabricated()
            if kind == 'shape': args[0] = args[0][:16]
            if kind == 'dtype': args[0] = args[0].astype(np.float64)
            if kind == 'nan': args[0][0, 0, 0] = np.nan
            if kind == 'score': args[3][0, 0, 0] += 0.1
            if kind == 'gap': args[4][0, 0, 0] += 0.1
            if kind == 'roles': args[5] = np.array(['sentence_end'])
            if kind == 'axis': args[8]['scores']['locations']['prefix_end'][0]['axis'] = 'PC4'
            if kind == 'json': args[8]['scores']['locations']['prefix_end'][0]['values'][f.IDS[0]] = True
            if kind == 'pair': args[8]['gaps']['pairs'][0]['left'] = f.IDS[2]
            with self.assertRaises(ValueError): c.corroborate(*args)


class StageTests(unittest.TestCase):
    def setUp(self):
        fixture = tf.StageTests(); fixture.setUp(); self.addCleanup(fixture.doCleanups)
        self.root, self.fixture = fixture.root, fixture
        ctx = patch.dict(os.environ, {'JLENS_NATURAL_ADDON_CHECK_RELEASE': '1'}); ctx.start(); self.addCleanup(ctx.stop)
        ctx = patch.object(c, 'FORWARD_SHA', f.sha(Path(f.__file__))); ctx.start(); self.addCleanup(ctx.stop)
        # Entirely fabricated amended-producer file; no actual wrapper source read.
        fake_producer = self.root/'forward_runtime_amendment.py'; fake_producer.write_text('# fixture')
        ctx = patch.object(c, 'OUT', self.root); ctx.start(); self.addCleanup(ctx.stop)
        ctx = patch.object(c, 'RECOVERY_SHA', f.sha(fake_producer)); ctx.start(); self.addCleanup(ctx.stop)
        ctx = patch.object(c, 'load_helper', return_value=f); ctx.start(); self.addCleanup(ctx.stop)
        ctx = patch.object(c, 'reader_gate'); self.gate = ctx.start(); self.addCleanup(ctx.stop)
        self.lock_args = dict(packets=self.root/'packets', manifest_sha='1'*64,
                              responses=self.root/'responses', lock_sha='2'*64, lock_commit='3'*40)

    def prepare(self):
        args = fabricated(); h, u, mean, scores, gaps, locations, rows, pairs, exports = args
        (self.root/'preparation').mkdir(); archive = self.root/'preparation/directions.npz'
        np.savez(archive, u32=u, source_mean64=mean)
        for ctx in [patch.object(f, 'OLD', self.root), patch.dict(f.PINS, {archive: f.sha(archive)})]:
            ctx.start(); self.addCleanup(ctx.stop)
        with patch.object(f, 'runtime', return_value={'python': c.PREFLIGHT_PYTHON, 'packages': {'fixture': '1'}}):
            digest = self.fixture.preflight()
        expected = f.pins(self.fixture.ds, self.fixture.ps)
        records, preflight = f.frozen_tokens(self.root, digest, expected, rows)
        self.gate.return_value = {'preflight_sha256': digest, 'tokens_sha256': preflight['outputs'][0]['sha256']}
        target = f.claim(self.root, c.MEASUREMENT_DIR)
        with (target/'features.npz').open('xb') as handle:
            np.savez(handle, activation_11=h, scores64=scores, gaps64=gaps, locations=locations)
        f.write_json(target/'inputs.json', {'schema': 'jlens_natural_addon_inputs_v1', 'records': records})
        for name in ['scores', 'gaps']: f.write_json(target/f'{name}.json', exports[name])
        f.finish(target, expected, [target/n for n in ['features.npz', 'inputs.json', 'scores.json', 'gaps.json']],
            {'runtime': {**preflight['runtime'], 'python': c.MEASUREMENT_PYTHON},
             'preflight_runtime': preflight['runtime'],
             'runtime_transition': {'preflight_python': c.PREFLIGHT_PYTHON, 'new_python': c.MEASUREMENT_PYTHON,
                 'old_preflight_source_sha256': c.FORWARD_SHA, 'package_versions': c.DISTRO, 'binary_sha256s': c.BINARIES},
             'base_producer_sha256': c.FORWARD_SHA, 'runtime_amendment_sha256': c.AMENDMENT_SHA,
             'failed_admission_sha256s': c.FAILED, 'preflight_receipt_sha256': digest,
             'forward_count': 32, 'capture_locations': f.LOCATIONS, 'parameters_unchanged': True,
             'reference_decodes': 0, 'pca_refit': False, 'tokenizer_loaded': False,
             'model_weight_sha256': f.WEIGHT_SHA, 'adapter_revision': f.UPSTREAM_REV})
        receipt = f.read_json(target/'receipt.json')
        receipt['schema'] = 'jlens_natural_addon_forwards_receipt_v1'
        receipt['source_sha256'] = c.RECOVERY_SHA
        (target/'receipt.json').write_text(json.dumps(receipt))
        return f.sha(target/'receipt.json')

    def test_missing_lock_blocks_even_producer_import(self):
        with patch.object(c, 'reader_gate', side_effect=ValueError('lock absent')), \
             patch.object(c, 'load_helper', side_effect=AssertionError('premature producer')) as helper, \
             patch.object(np, 'load', side_effect=AssertionError('premature arrays')) as arrays:
            with self.assertRaises(ValueError): c.run('0'*64, root=self.root, **self.lock_args)
            helper.assert_not_called(); arrays.assert_not_called()
        self.assertTrue((self.root/'checked/failure.json').exists())

    def test_complete_check_and_repeat_refusal(self):
        digest = self.prepare()
        with patch.object(f, 'load_model', side_effect=AssertionError('model called')):
            c.run(digest, root=self.root, **self.lock_args)
        result = f.read_json(self.root/'checked/results.json')
        self.assertEqual(16, result['summaries']['prefix_end'][3]['positive'])
        receipt = f.read_json(self.root/'checked/receipt.json')
        self.assertFalse(receipt['model_loaded']); self.assertFalse(receipt['tokenizer_loaded'])
        self.assertEqual(receipt['outputs'][0]['sha256'], f.sha(self.root/'checked/results.json'))
        with patch.object(c, 'load_helper', side_effect=AssertionError('guard late')):
            with self.assertRaises(FileExistsError): c.run(digest, root=self.root, **self.lock_args)

    def test_tamper_blocks_before_arrays(self):
        digest = self.prepare()
        with (self.root/c.MEASUREMENT_DIR/'inputs.json').open('ab') as handle: handle.write(b' ')
        with patch.object(np, 'load', side_effect=AssertionError('arrays before pins')) as load:
            with self.assertRaises(ValueError): c.run(digest, root=self.root, **self.lock_args)
            load.assert_not_called()
        self.assertTrue((self.root/'checked/failure.json').exists())

    def test_different_judged_tokens_block_arrays(self):
        digest = self.prepare(); self.gate.return_value['tokens_sha256'] = '0'*64
        with patch.object(np, 'load', side_effect=AssertionError('arrays before token binding')) as load:
            with self.assertRaises(ValueError): c.run(digest, root=self.root, **self.lock_args)
            load.assert_not_called()

    def test_unapproved_runtime_transition_blocks_arrays(self):
        self.prepare()
        p = self.root/c.MEASUREMENT_DIR/'receipt.json'
        receipt = f.read_json(p)
        receipt['runtime_transition']['binary_sha256s'] = {}
        p.write_text(json.dumps(receipt))
        with patch.object(np, 'load', side_effect=AssertionError('arrays before runtime binding')) as load:
            with self.assertRaisesRegex(ValueError, 'runtime transition'):
                c.run(f.sha(p), root=self.root, **self.lock_args)
            load.assert_not_called()

    def test_bad_pin_and_dangling(self):
        with self.assertRaises(ValueError): c.run('bad', root=self.root, **self.lock_args)
        with self.assertRaises(FileExistsError): c.run('bad', root=self.root, **self.lock_args)
        second = self.root/'another'; second.mkdir(); (second/'checked').symlink_to(second/'absent')
        with self.assertRaises(FileExistsError): c.run('0'*64, root=second, **self.lock_args)


if __name__ == '__main__': unittest.main()
