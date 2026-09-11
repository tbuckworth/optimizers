"""Fabricated CPU fixtures only; no cached model/lens/tokenizer/feature access."""
import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

spec = importlib.util.spec_from_file_location('aggregate_decoder_test', Path(__file__).with_name('decode.py'))
d = importlib.util.module_from_spec(spec); spec.loader.exec_module(d)
torch.set_num_threads(1)


def fixture():
    raw = np.array([[1., i/110, 2., -1.] for i in range(110)], dtype=np.float64)
    records, inputs = [], []
    for name, row in zip(d.NAMES, raw, strict=True):
        family = name.split('/')[0]; sign = -1 if family == 'gradient' else 1
        norm = float(np.linalg.norm(row))
        records.append({'name': name, 'family': family, 'sign': sign, 'norm': norm, 'defined': True})
        inputs.append(sign*row/norm)
    return np.array(inputs, dtype=np.float32), raw, records


class Tokenizer:
    def __init__(self): self.calls = []
    def decode(self, ids):
        self.calls.append(ids)
        return ' \n中'+str(ids[0])
    def __call__(self, *args, **kwargs): raise AssertionError('tokenization forbidden')


class Model:
    def __init__(self, zero=False):
        torch.manual_seed(19)
        self.weight = torch.zeros((17, 4), dtype=torch.bfloat16) if zero else torch.randn((17, 4)).to(torch.bfloat16)
        self.calls = 0
    def unembed(self, z):
        self.calls += 1
        z = z.to(torch.bfloat16)
        return self.weight@(z/torch.sqrt(z.square().mean()+1e-5))
    def forward(self, *args, **kwargs): raise AssertionError('model forward forbidden')


class Tests(unittest.TestCase):
    def test_complete_110_roster(self):
        inputs, raw, records = fixture(); d.validate_inputs(inputs, raw, records, 4)
        self.assertEqual(len(set(d.NAMES)), 110)
        self.assertEqual(sum('/query_pooled/' in name for name in d.NAMES), 6)
        self.assertEqual(d.NAMES[51:55], ['gradient/reference0', 'gradient/reference1', 'gradient/reference_pooled', 'gradient/fit_mean'])

    def test_wrong_roster_precision_sign_and_zero(self):
        inputs, raw, records = fixture()
        for mutate in (lambda r: r.reverse(), lambda r: r[0].update(sign=1),
                       lambda r: r[0].update(defined=False), lambda r: r[0].update(norm=8)):
            bad = copy.deepcopy(records); mutate(bad)
            with self.assertRaises(ValueError): d.validate_inputs(inputs, raw, bad, 4)
        with self.assertRaises(ValueError): d.validate_inputs(inputs.astype(np.float64), raw, records, 4)
        bad_inputs = inputs.copy(); bad_inputs[0, 0] += .01
        with self.assertRaises(ValueError): d.validate_inputs(bad_inputs, raw, records, 4)
        bad_inputs[0, 0] = np.nan
        with self.assertRaises(ValueError): d.validate_inputs(bad_inputs, raw, records, 4)
        raw[0] = 0; inputs[0] = 0; records[0].update(norm=0., defined=False)
        d.validate_inputs(inputs, raw, records, 4)

    def test_transport_topk_full_logits_strings_and_fidelity(self):
        inputs, _, records = fixture(); model, tokenizer = Model(), Tokenizer()
        j = torch.tensor([[1., 2., 0., 0.], [0., 3., 0., 0.], [0., 0., 2., 0.], [0., 0., 0., 1.]])
        arrays, readouts = d.decode_all(model, tokenizer, j, inputs, records, 'cpu', width=4, vocab=17)
        self.assertEqual(model.calls, 110); self.assertEqual(len(tokenizer.calls), 1320)
        np.testing.assert_array_equal(arrays['inputs32'], inputs)
        np.testing.assert_allclose(arrays['transported32'], inputs@j.numpy().T, rtol=1e-6)
        self.assertFalse(np.allclose(arrays['transported32'], inputs@j.numpy()))
        self.assertEqual(arrays['logits32'].shape, (110, 17))
        self.assertEqual(arrays['top_ids64'].dtype, np.int64)
        for i, row in enumerate(readouts):
            self.assertEqual(len(set(row['token_ids'])), 12)
            self.assertEqual(row['tokens'], [' \n中'+str(t) for t in row['token_ids']])
            np.testing.assert_array_equal(arrays['logits32'][i, row['token_ids']], arrays['top_logits32'][i])
        d.fidelity(arrays, readouts)
        for i in (51, 52, 53, 106, 107, 108):
            name = readouts[i]['name'].split('/')[1]
            same = readouts[i]['reference_fidelity'][name]
            self.assertAlmostEqual(same['full_logit_cosine'], 1.)
            self.assertEqual(same['top12_overlap_count'], 12)

    def test_undefined_not_decoded_and_ties_retained(self):
        inputs, _, records = fixture(); inputs[0] = 0; records[0].update(defined=False, norm=0.)
        model, tokenizer = Model(zero=True), Tokenizer()
        arrays, rows = d.decode_all(model, tokenizer, torch.eye(4), inputs, records, 'cpu', width=4, vocab=17)
        self.assertEqual(model.calls, 109); self.assertEqual(len(tokenizer.calls), 109*12)
        self.assertEqual(rows[0]['tokens'], []); self.assertIsNone(rows[0]['ties'])
        self.assertTrue((arrays['top_ids64'][0] == -1).all())
        self.assertFalse(arrays['defined'][0])
        self.assertEqual(rows[1]['ties']['cutoff_tie_count'], 17)
        self.assertEqual(rows[1]['ties']['selected_at_cutoff_count'], 12)
        d.fidelity(arrays, rows)
        self.assertIsNone(rows[0]['reference_fidelity']['reference0']['top12_overlap_count'])
        self.assertIsNone(rows[1]['reference_fidelity']['reference0']['full_logit_cosine'])

    def test_nonfinite_jacobian_rejected(self):
        inputs, _, records = fixture(); j = torch.eye(4); j[0, 0] = float('nan')
        model = Model()
        with self.assertRaises(ValueError): d.decode_all(model, Tokenizer(), j, inputs, records, 'cpu', width=4, vocab=17)
        self.assertEqual(model.calls, 0)

    def test_release_and_existing_stage_guard(self):
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp)/'output'
            args = (Path(temp), 'a'*64, 'b'*64, 'c'*64, 'd'*64, target)
            with mock.patch.dict(d.os.environ, {}, clear=True), mock.patch.object(d, 'helper') as helper:
                with self.assertRaisesRegex(ValueError, 'release'): d.run(*args)
                helper.assert_not_called(); self.assertFalse(target.exists())
            # Reuse only the immutable helper's filesystem claim, no actual stage or input load.
            a = d.helper(); target.mkdir()
            args = (*args[:4], a.sha(Path(d.__file__)), target)
            with mock.patch.dict(d.os.environ, {'JLENS_AGGREGATE_DECODE_RELEASE': '1'}), \
                 mock.patch.object(d, 'decode_all') as decode:
                with self.assertRaises(FileExistsError): d.run(*args)
                decode.assert_not_called()


if __name__ == '__main__': unittest.main()
