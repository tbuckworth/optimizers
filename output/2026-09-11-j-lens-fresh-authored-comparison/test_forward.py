"""Fabricated CPU-only fixtures; no real corpus, tokenizer, model or scientific NPZ."""
import copy
import importlib.util
import os
from pathlib import Path
import re
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

SPEC = importlib.util.spec_from_file_location('fresh_authored_fixture', Path(__file__).with_name('forward.py'))
f = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(f)


def data():
    rows = []
    for ident in f.IDS:
        pair, template, pole = ident.split('-')
        verb = 'observed' if pole == 'O' else 'provided'
        text = (f'The actor{pair} {verb} the item{pole}.' if template == 'active'
                else f'The item{pole} was {verb} by the actor{pair}.')
        start = text.index(verb)
        rows.append(dict(id=ident, pair_id=pair, template=template, pole=pole,
                         text=text, verb=verb, verb_span=[start, start+len(verb)]))
    pairs = [dict(id=f'{p}-{t}', content_pair=p, template=t,
                  observation_id=f'{p}-{t}-O', provision_id=f'{p}-{t}-P') for p, t in f.CELLS]
    return rows, pairs


class Tokenizer:
    is_fast, bos_token_id = True, None
    def __call__(self, text, **kwargs):
        assert kwargs == dict(add_special_tokens=False, padding=False, truncation=False,
                              return_attention_mask=True, return_offsets_mapping=True)
        offsets = [list(m.span()) for m in re.finditer(r'\S+', text)]
        return dict(input_ids=list(range(1, len(offsets)+1)), attention_mask=[1]*len(offsets), offset_mapping=offsets)


def encoded(row):
    return Tokenizer()(row['text'], add_special_tokens=False, padding=False, truncation=False,
                       return_attention_mask=True, return_offsets_mapping=True)


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
        assert not torch.is_grad_enabled()
        self.calls += 1
        x = ids.float().unsqueeze(-1).expand(-1, -1, f.WIDTH)
        for layer in self.layers: x = layer(x)
        self.last_output = x
        return x


class PureTests(unittest.TestCase):
    def test_import_inert(self):
        with patch.object(Path, 'read_bytes', side_effect=AssertionError('import I/O')):
            spec = importlib.util.spec_from_file_location('inert', Path(f.__file__))
            module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)

    def test_roster_metadata_span_and_pair_guards(self):
        rows, pairs = data(); f.validate_rows(rows, pairs)
        for kind in ['order', 'words', 'span', 'pole', 'pair', 'wrapper', 'old16']:
            rr, pp = copy.deepcopy((rows, pairs))
            if kind == 'order': rr.reverse()
            if kind == 'words': rr[0]['text'] += ' word'
            if kind == 'span': rr[0]['verb_span'][0] += 1
            if kind == 'pole': rr[0]['pole'] = 'P'
            if kind == 'pair': pp[0]['provision_id'] = pp[1]['provision_id']
            if kind == 'wrapper': rr = {'rows': rr}
            if kind == 'old16': rr = rr[:16]
            with self.assertRaises(ValueError): f.validate_rows(rr, pp)

    def test_split_verb_and_leading_space_allowed(self):
        row = data()[0][0]; e = encoded(row); a, b = row['verb_span']
        e['offset_mapping'][2:3] = [[a-1, a+3], [a+3, b]]
        e['input_ids'] = list(range(1, 7)); e['attention_mask'] = [1]*6
        record = f.token_record(row, e)
        self.assertEqual({'verb': 3}, record['captured_positions'])
        self.assertEqual(row['verb'][3:], record['selected_substrings']['verb'])

    def test_bad_offsets_and_ids(self):
        row = data()[0][0]
        for kind in ['past', 'gap', 'period', 'empty', 'bool', 'mask', 'overlong']:
            e = encoded(row)
            if kind == 'past': e['offset_mapping'][2][1] += 1
            if kind == 'gap': e['offset_mapping'][2][0] += 1
            if kind == 'period': e['offset_mapping'][-1][1] -= 1
            if kind == 'empty': e['offset_mapping'][-1] = [0, 0]
            if kind == 'bool': e['input_ids'][0] = True
            if kind == 'mask': e['attention_mask'][0] = 0
            if kind == 'overlong': e['input_ids'] = [1]*97; e['attention_mask'] = [1]*97
            with self.assertRaises(ValueError): f.token_record(row, e)

    def test_single_postblock_position_one_forward_no_mutation(self):
        rows, _ = data(); records = f.tokenize(rows, Tokenizer())
        model = Model(); before = f.parameter_state(model)
        h = f.capture(model, records, 'cpu')
        expected = np.array([[[r['input_ids'][r['captured_positions'][name]]+12]*f.WIDTH
                              for name in f.LOCATIONS] for r in records], dtype=np.float32)
        self.assertEqual((32, 1, f.WIDTH), h.shape)
        self.assertEqual(np.float32, h.dtype)
        self.assertTrue(np.array_equal(h, expected)); self.assertEqual(32, model.calls)
        self.assertTrue(torch.all(model.last_output[0, -1] == 31))
        self.assertEqual(before, f.parameter_state(model)); self.assertFalse(model.layers[11]._forward_hooks)

    def test_scores_gap_order_and_export(self):
        rows, pairs = data(); u = np.eye(f.WIDTH, 4, dtype=np.float32)
        mean = np.arange(f.WIDTH, dtype=np.float64)/16
        h = np.arange(32*1*f.WIDTH, dtype=np.float32).reshape(32, 1, f.WIDTH)/32
        scores, gaps, export = f.score_export(h, u, mean, rows, pairs)
        self.assertTrue(np.array_equal(scores, h[:, :, :4].astype(np.float64)-mean[:4]))
        self.assertTrue(np.array_equal(gaps, scores[::2]-scores[1::2]))
        self.assertEqual((16, 1, 4), gaps.shape); self.assertEqual(np.float64, scores.dtype)
        self.assertEqual(f.LOCATIONS, list(export['locations']))
        self.assertEqual(f.IDS, list(export['locations']['verb'][0]['values']))
        self.assertEqual('jlens_fresh_authored_scores_v1', export['schema'])
        for bad in [h[:, 0], h.astype(np.float64), np.full_like(h, np.nan)]:
            with self.assertRaises(ValueError): f.score_export(bad, u, mean, rows, pairs)
        for bad in [u[:-1], u.astype(np.float64), np.zeros_like(u)]:
            with self.assertRaises(ValueError): f.validate_directions(bad, mean)


class StageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='jlens-fake-endpoint-'); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.rows, self.pairs = data()
        (self.root/'inputs').mkdir()
        f.write_json(self.root/'inputs/dataset.json', {'rows': self.rows}); f.write_json(self.root/'inputs/pairs.json', {'pairs': self.pairs})
        self.ds, self.ps = f.sha(self.root/'inputs/dataset.json'), f.sha(self.root/'inputs/pairs.json')
        for ctx in [patch.object(f, 'MAIN', self.root), patch.dict(f.PINS, {}, clear=True),
                    patch.object(f, 'DATASET_SHA', self.ds), patch.object(f, 'PAIRS_SHA', self.ps),
                    patch.object(f, 'runtime', return_value={'python': sys.version, 'packages': {'fixture': '1'}}),
                    patch.dict(os.environ, {'JLENS_FRESH_AUTHORED_PREFLIGHT_RELEASE': '1', 'JLENS_FRESH_AUTHORED_GPU_RELEASE': '1'})]:
            ctx.start(); self.addCleanup(ctx.stop)

    def preflight(self):
        with patch('transformers.AutoTokenizer.from_pretrained', return_value=Tokenizer()): f.preflight(self.ds, self.ps, self.root)
        return f.sha(self.root/'token-preflight/receipt.json')

    def test_preflight_freeze_tamper_and_repeat(self):
        digest = self.preflight(); expected = f.pins(self.ds, self.ps)
        records, receipt = f.frozen_tokens(self.root, digest, expected, self.rows)
        self.assertEqual(32, len(records)); self.assertFalse(receipt['model_loaded'])
        self.assertFalse(receipt['scientific_array_loaded'])
        with patch.object(f, 'verify', side_effect=AssertionError('guard late')):
            with self.assertRaises(FileExistsError): f.preflight(self.ds, self.ps, self.root)
        with (self.root/'token-preflight/tokens.json').open('ab') as h: h.write(b' ')
        with self.assertRaises(ValueError): f.frozen_tokens(self.root, digest, expected, self.rows)

    def test_wrapper(self):
        with patch.object(f, 'read_json', side_effect=[self.rows, {'pairs': self.pairs}]):
            with self.assertRaises(ValueError): f.load_panel()

    def test_failure_and_dangling_exclusive_guard(self):
        with self.assertRaises(ValueError): f.preflight('0'*64, self.ps, self.root)
        self.assertTrue((self.root/'token-preflight/failure.json').exists())
        with self.assertRaises(FileExistsError): f.preflight(self.ds, self.ps, self.root)
        (self.root/'forwards').symlink_to(self.root/'absent')
        with patch.object(f, 'verify', side_effect=AssertionError('late guard')):
            with self.assertRaises(FileExistsError): f.forwards(self.ds, self.ps, '0'*64, self.root)

    def test_missing_preflight_blocks_archive_and_model(self):
        with patch.object(np, 'load', side_effect=AssertionError('scientific I/O')) as ar, patch.object(f, 'load_model') as mo:
            with self.assertRaises(FileNotFoundError): f.forwards(self.ds, self.ps, '0'*64, self.root)
            ar.assert_not_called(); mo.assert_not_called()

    def test_late_mapping_failure_stops_whole_panel(self):
        calls = []
        class LateBad(Tokenizer):
            def __call__(self, text, **kwargs):
                calls.append(text); value = super().__call__(text, **kwargs)
                if len(calls) == 32: value['offset_mapping'][-1][1] -= 1
                return value
        with patch('transformers.AutoTokenizer.from_pretrained', return_value=LateBad()):
            with self.assertRaises(ValueError): f.preflight(self.ds, self.ps, self.root)
        self.assertEqual([r['text'] for r in self.rows], calls)
        self.assertTrue((self.root/'token-preflight/failure.json').exists())
        self.assertFalse((self.root/'token-preflight/tokens.json').exists())
        self.assertFalse((self.root/'token-preflight/receipt.json').exists())

    def test_fake_end_to_end_no_forward_tokenizer(self):
        (self.root/'preparation').mkdir(); archive = self.root/'preparation/directions.npz'
        np.savez(archive, u32=np.eye(f.WIDTH, 4, dtype=np.float32), source_mean64=np.zeros(f.WIDTH, dtype=np.float64))
        model = Model(); original = f.capture
        with patch.dict(f.PINS, {archive: f.sha(archive)}), patch.object(f, 'OLD', self.root):
            digest = self.preflight()
            with patch.object(f, 'load_model', return_value=(model, model)), \
                 patch.object(f, 'capture', side_effect=lambda m, r, d: original(m, r, 'cpu')), \
                 patch('transformers.AutoTokenizer.from_pretrained', side_effect=AssertionError('forward tokenizer')), \
                 patch.object(torch.cuda, 'get_device_name', return_value='fake CPU'), \
                 patch.object(torch.cuda, 'max_memory_allocated', return_value=0), patch.object(torch.cuda, 'max_memory_reserved', return_value=0):
                f.forwards(self.ds, self.ps, digest, self.root)
            receipt = f.read_json(self.root/'forwards/receipt.json')
            self.assertEqual(32, model.calls); self.assertEqual(f.LOCATIONS, receipt['capture_locations'])
            self.assertEqual(0, receipt['reference_decodes']); self.assertTrue(receipt['parameters_unchanged'])
            gaps = f.read_json(self.root/'forwards/gaps.json')
            self.assertEqual(f.LOCATIONS, gaps['locations']); self.assertEqual(16, len(gaps['pairs']))
            with np.load(self.root/'forwards/features.npz', allow_pickle=False) as saved:
                self.assertEqual(f.LOCATIONS, saved['locations'].tolist())
                self.assertEqual((32, 1, f.WIDTH), saved['activation_11'].shape)
                self.assertEqual((32, 1, 4), saved['scores64'].shape)
                self.assertEqual((16, 1, 4), saved['gaps64'].shape)
            for output in receipt['outputs']: self.assertEqual(output['sha256'], f.sha(self.root/'forwards'/output['path']))
            with self.assertRaises(FileExistsError): f.forwards(self.ds, self.ps, digest, self.root)

    def test_strict_json(self):
        for i, raw in enumerate(['{"a":1,"a":2}', '{"x":NaN}', '{"x":1e999}']):
            path = self.root/f'bad{i}.json'; path.write_text(raw)
            with self.assertRaises(ValueError): f.read_json(path)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
