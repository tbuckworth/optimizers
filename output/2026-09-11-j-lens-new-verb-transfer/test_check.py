"""Fabricated saved-array checker tests; never reads measured artifacts."""
import copy
import importlib.util
import os
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import test_forward as tf

SPEC = importlib.util.spec_from_file_location('new_verb_check_fixture', Path(__file__).with_name('check.py'))
c = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(c)
f = tf.f


def fabricated():
    rows, pairs = tf.data()
    u = np.eye(1024, 4, dtype=np.float32); mean = np.arange(1024, dtype=np.float64)/32
    h = np.zeros((32, 2, 1024), dtype=np.float32)
    h[::2, :, :4] = 1
    scores = h[:, :, :4].astype(np.float64)-mean[:4]
    gaps = scores[::2]-scores[1::2]
    roles = ['verb', 'sentence_end']
    exports = {'scores': {'schema': 'jlens_new_verb_transfer_scores_v1', 'locations': {
        name: [{'axis': f'PC{a+1}', 'values': {r['id']: float(scores[i, j, a]) for i, r in enumerate(rows)}}
               for a in range(4)] for j, name in enumerate(roles)}},
        'gaps': {'locations': roles, 'axes': ['PC1', 'PC2', 'PC3', 'PC4'], 'orientation': 'observation minus provision',
                 'pairs': [{**pair, 'gaps64': gaps[i].tolist()} for i, pair in enumerate(pairs)]}}
    return [h, u, mean, scores, gaps, np.array(roles), rows, pairs, exports]


class PureTests(unittest.TestCase):
    def test_helper_pin_before_import(self):
        with patch.object(c, 'sha', return_value='0'*64), \
             patch.object(c.importlib.util, 'spec_from_file_location', side_effect=AssertionError('import before pin')) as loader:
            with self.assertRaises(ValueError): c.load_helper()
            loader.assert_not_called()

    def test_all_positive_and_exact_zero(self):
        args = fabricated(); result = c.corroborate(*args)
        self.assertTrue(result['primary_all_16_verb_PC4_positive'])
        self.assertEqual(256, result['scalar_checks']); self.assertEqual(128, result['gap_checks'])
        self.assertEqual(8, result['summaries']['verb'][3]['both_templates_positive'])
        args[0][0, 0, 3] = 0; args[3][0, 0, 3] = -args[2][3]; args[4][0, 0, 3] = 0
        args[8]['scores']['locations']['verb'][3]['values'][f.IDS[0]] = float(args[3][0, 0, 3])
        args[8]['gaps']['pairs'][0]['gaps64'] = args[4][0].tolist()
        result = c.corroborate(*args)
        self.assertFalse(result['primary_all_16_verb_PC4_positive'])
        self.assertEqual(1, result['summaries']['verb'][3]['zero'])
        self.assertEqual(7, result['summaries']['verb'][3]['both_templates_positive'])
        self.assertEqual(['nv01'], result['summaries']['verb'][3]['template_sign_changes'])

    def test_bad_arrays_and_json_joins(self):
        for kind in ['shape', 'dtype', 'nan', 'score', 'gap', 'roles', 'axis', 'json', 'pair']:
            args = fabricated()
            if kind == 'shape': args[0] = args[0][:16]
            if kind == 'dtype': args[0] = args[0].astype(np.float64)
            if kind == 'nan': args[0][0, 0, 0] = np.nan
            if kind == 'score': args[3][0, 0, 0] += 0.1
            if kind == 'gap': args[4][0, 0, 0] += 0.1
            if kind == 'roles': args[5] = args[5][::-1]
            if kind == 'axis': args[8]['scores']['locations']['verb'][0]['axis'] = 'PC4'
            if kind == 'json': args[8]['scores']['locations']['verb'][0]['values'][f.IDS[0]] = True
            if kind == 'pair': args[8]['gaps']['pairs'][0]['observation_id'] = f.IDS[2]
            with self.assertRaises(ValueError): c.corroborate(*args)


class StageTests(unittest.TestCase):
    def setUp(self):
        fixture = tf.StageTests(); fixture.setUp(); self.addCleanup(fixture.doCleanups)
        self.root, self.fixture = fixture.root, fixture
        ctx = patch.dict(os.environ, {'JLENS_NEW_VERB_TRANSFER_CHECK_RELEASE': '1'}); ctx.start(); self.addCleanup(ctx.stop)
        ctx = patch.object(c, 'load_helper', return_value=f); ctx.start(); self.addCleanup(ctx.stop)

    def prepare(self):
        args = fabricated(); h, u, mean, scores, gaps, locations, rows, pairs, exports = args
        (self.root/'preparation').mkdir(); archive = self.root/'preparation/directions.npz'
        np.savez(archive, u32=u, source_mean64=mean)
        for ctx in [patch.object(f, 'OLD', self.root), patch.dict(f.PINS, {archive: f.sha(archive)})]:
            ctx.start(); self.addCleanup(ctx.stop)
        digest = self.fixture.preflight(); expected = f.pins(self.fixture.ds, self.fixture.ps)
        records, preflight = f.frozen_tokens(self.root, digest, expected, rows)
        target = f.claim(self.root, 'forwards')
        with (target/'features.npz').open('xb') as handle:
            np.savez(handle, activation_11=h, scores64=scores, gaps64=gaps, locations=locations)
        f.write_json(target/'inputs.json', {'schema': 'jlens_new_verb_transfer_inputs_v1', 'records': records})
        for name in ['scores', 'gaps']: f.write_json(target/f'{name}.json', exports[name])
        f.finish(target, expected, [target/n for n in ['features.npz', 'inputs.json', 'scores.json', 'gaps.json']],
            {'runtime': preflight['runtime'], 'preflight_receipt_sha256': digest,
             'forward_count': 32, 'capture_locations': f.LOCATIONS, 'parameters_unchanged': True,
             'reference_decodes': 0, 'pca_refit': False, 'tokenizer_loaded': False,
             'model_weight_sha256': f.WEIGHT_SHA, 'adapter_revision': f.UPSTREAM_REV})
        return f.sha(target/'receipt.json')

    def test_complete_check_and_repeat_refusal(self):
        digest = self.prepare()
        with patch.object(f, 'load_model', side_effect=AssertionError('model called')):
            c.run(digest, self.root)
        result = f.read_json(self.root/'checked/results.json')
        self.assertTrue(result['primary_all_16_verb_PC4_positive'])
        receipt = f.read_json(self.root/'checked/receipt.json')
        self.assertFalse(receipt['model_loaded']); self.assertFalse(receipt['tokenizer_loaded'])
        self.assertEqual(receipt['outputs'][0]['sha256'], f.sha(self.root/'checked/results.json'))
        with patch.object(c, 'load_helper', side_effect=AssertionError('guard late')):
            with self.assertRaises(FileExistsError): c.run(digest, self.root)

    def test_tamper_blocks_before_arrays(self):
        digest = self.prepare()
        with (self.root/'forwards/inputs.json').open('ab') as handle: handle.write(b' ')
        with patch.object(np, 'load', side_effect=AssertionError('arrays before pins')) as load:
            with self.assertRaises(ValueError): c.run(digest, self.root)
            load.assert_not_called()
        self.assertTrue((self.root/'checked/failure.json').exists())

    def test_bad_pin_and_dangling(self):
        with self.assertRaises(ValueError): c.run('bad', self.root)
        with self.assertRaises(FileExistsError): c.run('bad', self.root)
        second = self.root/'another'; second.mkdir(); (second/'checked').symlink_to(second/'absent')
        with self.assertRaises(FileExistsError): c.run('0'*64, second)


if __name__ == '__main__': unittest.main()
