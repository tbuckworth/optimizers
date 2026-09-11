"""Fabricated CPU-only tests. No real tokenizer, model, corpus or scientific archive access."""

import copy
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

SPEC = importlib.util.spec_from_file_location('independent_forward_fixture', Path(__file__).with_name('forward.py'))
f = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(f)


def data():
    rows = [{'id': ident, 'topic': ident.split('-')[0], 'prefix': ' '.join([ident]+['fabricated']*15)} for ident in f.IDS]
    pairs = [{'id': f'P{i+1:02}', 'left': f.IDS[2*i], 'right': f.IDS[2*i+1]} for i in range(12)]
    return rows, pairs


class Tokenizer:
    is_fast, bos_token_id = True, None
    def __call__(self, text, **kwargs):
        assert kwargs == {'add_special_tokens': False, 'truncation': False, 'return_attention_mask': True}
        ids = list(range(1, len(text.split())+1))
        return {'input_ids': ids, 'attention_mask': [1]*len(ids)}


class Layer(torch.nn.Module):
    def forward(self, x): return x+1


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.layers = torch.nn.ModuleList([Layer() for _ in range(24)])
        self.calls = 0
        self.eval()

    def forward(self, ids):
        self.calls += 1
        x = ids.float().unsqueeze(-1).expand(-1, -1, f.WIDTH)
        for layer in self.layers: x = layer(x)
        self.last_output = x
        return x


class PureTests(unittest.TestCase):
    def test_import_inert(self):
        with patch.object(Path, 'read_bytes', side_effect=AssertionError('import I/O')):
            spec = importlib.util.spec_from_file_location('inert_forward', Path(f.__file__))
            module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)

    def test_exact_prefix_and_pair_rosters(self):
        rows, pairs = data(); f.validate_rows(rows, pairs)
        for change in ['order', 'length', 'spacing', 'pair']:
            rr, pp = copy.deepcopy((rows, pairs))
            if change == 'order': rr.reverse()
            if change == 'length': rr[0]['prefix'] += ' extra'
            if change == 'spacing': rr[0]['prefix'] += ' '
            if change == 'pair': pp[0]['right'], pp[1]['right'] = pp[1]['right'], pp[0]['right']
            with self.assertRaises(ValueError): f.validate_rows(rr, pp)

    def test_preflight_no_cutting_and_capture_postblock_last(self):
        rows, pairs = data()
        records = f.tokenize(rows, Tokenizer())
        self.assertEqual(24, len(records))
        self.assertTrue(all(r['input_ids'] == list(range(1, 17)) and r['attention_mask'] == [1]*16 and r['captured_position'] == 15 for r in records))
        model = Model(); before = f.parameter_state(model)
        h = f.capture(model, records, 'cpu')
        self.assertTrue(np.array_equal(h, np.full((24, f.WIDTH), 28, dtype=np.float32)))
        self.assertEqual(24, model.calls)
        self.assertTrue(torch.all(model.last_output[0, -1] == 40))
        self.assertEqual(before, f.parameter_state(model))
        self.assertFalse(model.layers[11]._forward_hooks)
        with torch.no_grad(): model.weight.add_(1)
        self.assertNotEqual(before, f.parameter_state(model))

    def test_bad_token_counts_ids_and_masks(self):
        row = data()[0][:1]
        for ids, mask in [([], []), (list(range(97)), [1]*97), ([True], [1]), ([f.VOCAB], [1]), ([1], [0]), ([1], [True])]:
            with self.assertRaises(ValueError):
                f.tokenize(row, lambda *a, **k: {'input_ids': ids, 'attention_mask': mask})

    def test_canonical_score_formula_and_bad_arrays(self):
        rows, pairs = data()
        u = np.eye(f.WIDTH, 4, dtype=np.float32)
        mean = np.arange(f.WIDTH, dtype=np.float64)/16
        h = np.arange(24*f.WIDTH, dtype=np.float32).reshape(24, f.WIDTH)/32
        scores, gaps, export = f.score_export(h, u, mean, rows, pairs)
        self.assertTrue(np.array_equal(scores, h[:, :4].astype(np.float64)-mean[:4]))
        self.assertTrue(np.array_equal(gaps, scores[::2]-scores[1::2]))
        self.assertEqual('jlens_independent_content_scores_v1', export['schema'])
        self.assertEqual(f.IDS, list(export['scores'][0]['values']))
        for bad in [u[:-1], u.astype(np.float64), np.zeros_like(u), np.full_like(u, np.nan)]:
            with self.assertRaises(ValueError): f.validate_directions(bad, mean)
        with self.assertRaises(ValueError): f.score_export(np.full_like(h, np.inf), u, mean, rows, pairs)


class StageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='jlens-fake-forward-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root/'excerpts-resumed').mkdir()
        self.rows, self.pairs = data()
        for name, value in [('dataset', self.rows), ('pairs', self.pairs)]: f.write_json(self.root/'excerpts-resumed'/f'{name}.json', value)
        self.dataset_sha, self.pairs_sha = [f.sha(self.root/'excerpts-resumed'/f'{name}.json') for name in ['dataset', 'pairs']]
        for context in [patch.object(f, 'MAIN', self.root), patch.dict(f.PINS, {}, clear=True),
                        patch.object(f, 'DATASET_SHA', self.dataset_sha), patch.object(f, 'PAIRS_SHA', self.pairs_sha),
                        patch.object(f, 'runtime', return_value={'python': sys.version, 'packages': {'fixture': '1'}}),
                        patch.dict(os.environ, {'JLENS_INDEPENDENT_PREFLIGHT_RELEASE': '1', 'JLENS_INDEPENDENT_GPU_RELEASE': '1'})]:
            context.start(); self.addCleanup(context.stop)

    def preflight(self):
        with patch('transformers.AutoTokenizer.from_pretrained', return_value=Tokenizer()):
            f.preflight(self.dataset_sha, self.pairs_sha, self.root)
        return f.sha(self.root/'token-preflight/receipt.json')

    def test_complete_fabricated_preflight_and_receipt_binding(self):
        digest = self.preflight()
        expected = f.pins(self.dataset_sha, self.pairs_sha)
        records, receipt = f.frozen_tokens(self.root, digest, expected, self.rows)
        self.assertEqual(24, len(records))
        self.assertFalse(receipt['model_loaded'])
        self.assertFalse(receipt['scientific_array_loaded'])
        with patch.object(f, 'verify', side_effect=AssertionError('read before guard')):
            with self.assertRaises(FileExistsError): f.preflight(self.dataset_sha, self.pairs_sha, self.root)
        with (self.root/'token-preflight/tokens.json').open('ab') as handle: handle.write(b' ')
        with self.assertRaises(ValueError): f.frozen_tokens(self.root, digest, expected, self.rows)

    def test_failed_pin_and_dangling_guard_before_model(self):
        with self.assertRaises(ValueError): f.preflight('0'*64, self.pairs_sha, self.root)
        self.assertTrue((self.root/'token-preflight/failure.json').exists())
        with self.assertRaises(FileExistsError): f.preflight(self.dataset_sha, self.pairs_sha, self.root)
        (self.root/'forwards').symlink_to(self.root/'missing')
        with patch.object(f, 'verify', side_effect=AssertionError('input read')):
            with self.assertRaises(FileExistsError): f.forwards(self.dataset_sha, self.pairs_sha, '0'*64, self.root)

    def test_missing_preflight_blocks_model_and_archive(self):
        with patch.object(f, 'load_model', side_effect=AssertionError('model access')) as model, \
             patch.object(np, 'load', side_effect=AssertionError('archive access')) as archive:
            with self.assertRaises(FileNotFoundError): f.forwards(self.dataset_sha, self.pairs_sha, '0'*64, self.root)
            model.assert_not_called(); archive.assert_not_called()

    def test_complete_fake_forward_stage_without_tokenizer(self):
        (self.root/'preparation').mkdir()
        archive = self.root/'preparation/directions.npz'
        np.savez(archive, u32=np.eye(f.WIDTH, 4, dtype=np.float32), source_mean64=np.zeros(f.WIDTH, dtype=np.float64))
        model = Model()
        original_capture = f.capture
        with patch.dict(f.PINS, {archive: f.sha(archive)}), patch.object(f, 'OLD', self.root):
            digest = self.preflight()
            with patch.object(f, 'load_model', return_value=(model, model)), \
                 patch.object(f, 'capture', side_effect=lambda m, r, d: original_capture(m, r, 'cpu')), \
                 patch('transformers.AutoTokenizer.from_pretrained', side_effect=AssertionError('tokenizer in forward')), \
                 patch.object(torch.cuda, 'get_device_name', return_value='fabricated CPU'), \
                 patch.object(torch.cuda, 'max_memory_allocated', return_value=0), \
                 patch.object(torch.cuda, 'max_memory_reserved', return_value=0):
                f.forwards(self.dataset_sha, self.pairs_sha, digest, self.root)
            receipt = f.read_json(self.root/'forwards/receipt.json')
            self.assertEqual(24, receipt['forward_count'])
            self.assertEqual(24, model.calls)
            self.assertEqual(0, receipt['reference_decodes'])
            self.assertTrue(receipt['parameters_unchanged'])
            self.assertFalse(receipt['tokenizer_loaded'])
            inputs = f.read_json(self.root/'forwards/inputs.json')
            self.assertEqual(f.IDS, [r['id'] for r in inputs['records']])
            scores = f.read_json(self.root/'forwards/scores.json')
            self.assertEqual('jlens_independent_content_scores_v1', scores['schema'])
            self.assertTrue(all(v == 28.0 for a in scores['scores'] for v in a['values'].values()))
            for output in receipt['outputs']:
                self.assertEqual(output['sha256'], f.sha(self.root/'forwards'/output['path']))
            with patch.object(f, 'verify', side_effect=AssertionError('repeat read')):
                with self.assertRaises(FileExistsError): f.forwards(self.dataset_sha, self.pairs_sha, digest, self.root)

    def test_strict_json(self):
        for i, raw in enumerate(['{"a":1,"a":2}', '{"x":NaN}', '{"x":1e999}']):
            path = self.root/f'invalid{i}.json'; path.write_text(raw)
            with self.assertRaises(ValueError): f.read_json(path)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
