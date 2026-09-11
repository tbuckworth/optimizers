"""Fabricated CPU-only fixtures. Never load corpus, tokenizer or cached weights."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock

import numpy as np
import torch

spec = importlib.util.spec_from_file_location('aggregate_acquire_fixture', Path(__file__).with_name('acquire.py'))
a = importlib.util.module_from_spec(spec); spec.loader.exec_module(a)
torch.set_num_threads(1)


class Block(torch.nn.Module):
    def __init__(self, tuple_output=False):
        super().__init__(); self.tuple_output = tuple_output

    def forward(self, x):
        return (x, 'retained') if self.tuple_output else x


class Tiny(torch.nn.Module):
    def __init__(self, tuple_output=False):
        super().__init__()
        torch.manual_seed(71)
        self.embedding = torch.nn.Embedding(17, 4)
        self.layers = torch.nn.ModuleList([Block() for _ in range(11)]+[Block(tuple_output)])
        self._lm_head = torch.nn.Linear(4, 17, bias=False)
        self.calls = 0
        self.eval(); self.requires_grad_(False)

    def forward(self, ids):
        self.calls += 1
        h = self.embedding(ids)
        for layer in self.layers:
            h = layer(h)
            if isinstance(h, tuple):
                assert h[1] == 'retained'
                h = h[0]
        return types.SimpleNamespace(last_hidden_state=h.cumsum(dim=1)/40)


def panel():
    rows = []
    for role, batch, articles in [('fit', None, 256)]+[('query', i, 4) for i in range(16)]+[('reference', i, 64) for i in range(2)]:
        for article in range(articles):
            for span in range(4):
                index = len(rows)
                rows.append({'id': f'row{index}', 'article_id': f'{role}-{batch}-{article}',
                    'role': role, 'batch': batch, 'input_ids': [index]+list(range(1, 40)),
                    'attention_mask': [1]*40})
    return {'schema': 'jlens_aggregate_panel_v1', 'rows': rows}


class Tests(unittest.TestCase):
    def test_exact_capture_and_loss_indexing(self):
        ids = torch.tensor([[i % 17 for i in range(40)]])
        for tuples in (False, True):
            model = Tiny(tuples); before = a.parameter_state(model)
            h, g, losses, mean_loss = a.capture_one(model, ids, width=4)
            leaf = model.embedding(ids).detach().requires_grad_(True)
            logits = model._lm_head(leaf.cumsum(dim=1)/40)
            manual = torch.stack([-logits[0, t-1].log_softmax(0)[ids[0, t]] for t in range(32, 40)])
            expected, = torch.autograd.grad(manual.mean(), leaf)
            np.testing.assert_allclose(losses, manual.detach().numpy(), rtol=1e-6)
            np.testing.assert_allclose(mean_loss, manual.detach().mean().numpy(), rtol=1e-6)
            np.testing.assert_allclose(h, leaf[0, 31].detach().numpy())
            np.testing.assert_allclose(g, expected[0, 31].numpy(), rtol=1e-5, atol=1e-8)
            self.assertGreater(float(np.linalg.norm(g)), 0)
            self.assertEqual(model.calls, 1)
            self.assertEqual(before, a.parameter_state(model))
            self.assertFalse(model.layers[11]._forward_hooks)
            self.assertEqual(h.dtype, np.float32); self.assertEqual(g.dtype, np.float32)

    def test_grad_disabled_and_bad_shapes(self):
        model = Tiny(); ids = torch.zeros((1, 40), dtype=torch.long)
        with torch.no_grad(), self.assertRaisesRegex(ValueError, 'grad mode'):
            a.capture_one(model, ids, width=4)
        with self.assertRaises(ValueError): a.capture_one(model, ids[:, :39], width=4)
        with self.assertRaises(ValueError): a.capture_one(model, ids.float(), width=4)
        self.assertEqual(model.calls, 0)

    def test_hook_removed_on_failure(self):
        model = Tiny()
        with torch.no_grad(): model.embedding.weight[0, 0] = float('nan')
        with self.assertRaisesRegex(ValueError, 'nonfinite'):
            a.capture_one(model, torch.zeros((1, 40), dtype=torch.long), width=4)
        self.assertFalse(model.layers[11]._forward_hooks)

    def test_parameter_mutations_detectable(self):
        model = Tiny(); before = a.parameter_state(model)
        with torch.no_grad(): model.embedding.weight.add_(1)
        self.assertNotEqual(before, a.parameter_state(model))
        model.embedding.weight.requires_grad_(True)
        with self.assertRaises(ValueError): a.parameter_state(model)

    def test_panel_valid_and_bad_rosters(self):
        good = panel(); self.assertEqual(len(a.validate_panel(good)), 1792)
        mutations = [lambda p: p['rows'].pop(),
            lambda p: p['rows'][0].update(role='heldout'),
            lambda p: p['rows'][1024].update(article_id=p['rows'][0]['article_id']),
            lambda p: p['rows'][1024].update(batch=16),
            lambda p: p['rows'][0].update(id=p['rows'][1]['id']),
            lambda p: p['rows'][0].update(input_ids=p['rows'][1]['input_ids']),
            lambda p: p['rows'][0]['input_ids'].__setitem__(0, True),
            lambda p: p['rows'][0]['attention_mask'].__setitem__(0, 0),
            lambda p: p['rows'][0].update(article_id='unexpected fifth article')]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                bad = copy.deepcopy(good); mutation(bad)
                with self.assertRaises(ValueError): a.validate_panel(bad)

    def test_stage_claim_existing_and_dangling(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp)/'new'; a.claim(target, 'a'*64)
            original = (target/'attempt.json').read_bytes()
            with self.assertRaises(FileExistsError): a.claim(target, 'b'*64)
            self.assertEqual(original, (target/'attempt.json').read_bytes())
            dangling = Path(tmp)/'dangling'; dangling.symlink_to(Path(tmp)/'missing')
            with self.assertRaises(FileExistsError): a.claim(dangling, 'a'*64)

    def test_guard_before_reads_and_failed_stage_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); panel_path = root/'panel.json'; protocol = root/'protocol.md'
            args = (panel_path, 'a'*64, a.PROTOCOL, a.PROTOCOL_SHA, a.sha(Path(a.__file__)), root/'stage')
            with mock.patch.dict(a.os.environ, {}, clear=True), mock.patch.object(a, 'load_model') as load:
                with self.assertRaisesRegex(ValueError, 'release'): a.run(*args)
                self.assertFalse(args[-1].exists()); load.assert_not_called()
            with mock.patch.dict(a.os.environ, {'JLENS_AGGREGATE_ACQUIRE_RELEASE': '1'}), \
                 mock.patch.object(a, 'verify', side_effect=ValueError('fabricated input failure')), \
                 mock.patch.object(a, 'load_model') as load:
                with self.assertRaisesRegex(ValueError, 'fabricated'): a.run(*args)
                self.assertEqual(json.loads((args[-1]/'failure.json').read_text())['completed_rows'], 0)
                with self.assertRaises(FileExistsError): a.run(*args)
                load.assert_not_called()

    def test_strict_json_and_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'data.json'
            for bad in ('{"x":1,"x":2}', '{"x":NaN}', '{"x":1e999}'):
                path.write_text(bad)
                with self.assertRaises(ValueError): a.read_json(path)
            path.write_text('{"x":1}'); self.assertEqual(a.read_json(path), {'x': 1})
            a.verify({path: a.sha(path)})
            with self.assertRaises(ValueError): a.verify({path: 'a'*64})


if __name__ == '__main__': unittest.main()
