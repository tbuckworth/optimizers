"""CPU fabricated arrays/tiny modules only; no real tokens, weights or scientific archives."""
import copy
import importlib.util
import json
from pathlib import Path
import re
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch
import numpy as np
import torch

SPEC = importlib.util.spec_from_file_location('pattern_calibrate', Path(__file__).with_name('calibrate.py'))
m = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(m)
P = m.import_pinned(m.OUT/'preflight.py', m.PREFLIGHT_SOURCE_SHA)
F = m.import_pinned(P.BASE, P.BASE_SHA)
A = m.import_pinned(P.HOST_HELPER, P.HOST_HELPER_SHA)
PAT = m.import_pinned(m.MAIN/'patterns.py', m.PATTERNS_SHA)
sys.path.insert(0, str(F.UPSTREAM))
from jlens.hf import HFLensModel
from jlens.lens import JacobianLens
torch.set_num_threads(1)


def panel():
    rows, pairs, records, candidates = [], [], [], []
    for i in range(86):
        candidates.append({'candidate_id': f'K{i+1:03}', 'topic': 'programming',
                           'role': 'evaluation' if i % 3 == 2 else 'calibration'})
    chosen = [p for p in candidates if p['role'] == 'calibration'][:32]
    for i, candidate in enumerate(chosen, 1):
        ident = f'C{i:02}'
        pairs.append({'id': ident, 'left': ident+'-L', 'right': ident+'-R', 'topic': 'programming',
                      'candidate_id': candidate['candidate_id'], 'role': 'calibration'})
        for side in ('L', 'R'):
            row = {'id': ident+'-'+side, 'topic': 'programming', 'prefix': ' '.join([ident+side]+['word']*15)}
            offsets = [(x.start(), x.end()) for x in re.finditer(r'\S+', row['prefix'])]
            value = {'input_ids': list(range(16*len(rows), 16*len(rows)+16)), 'attention_mask': [1]*16,
                     'offset_mapping': offsets}
            rows.append(row); records.append(F.token_record(row, value))
    manifest = {'schema': 'jlens_pattern_calibration_candidate_manifest_v1', 'protocol_sha256': P.PROTOCOL_SHA,
                'seed': '20260921', 'candidates': candidates}
    return rows, pairs, records, manifest


def references():
    return {'schema': 'jlens_fresh_references_v1', 'axes': [{'axis': a,
        'A': {s: [f'old {a} {s} {i}' for i in range(12)] for s in ('positive', 'negative')},
        'B': {'positive': ['unchanged unused list'], 'negative': ['also unchanged']},
        'C': {'positive': 'exact original\npositive prefix', 'negative': 'exact original negative prefix'}} for a in m.AXES]}


class DecodeOnly:
    def __init__(self): self.ids = []
    def __call__(self, *args, **kwargs): raise AssertionError('no retokenization')
    def decode(self, ids):
        self.ids.extend(ids)
        return ['', ' café', '\n', '中文', 'same'][ids[0] % 5]


class RecordingNorm(torch.nn.Module):
    def __init__(self): super().__init__(); self.seen = []
    def forward(self, x): self.seen.append(x.dtype); return x


class TinyText(torch.nn.Module):
    def __init__(self, width):
        super().__init__(); self.layers = torch.nn.ModuleList([torch.nn.Identity() for _ in range(12)])
        self.norm = RecordingNorm(); self.embed_tokens = torch.nn.Embedding(1, width).to(torch.bfloat16)
        self.width, self.calls = width, []
    def forward(self, input_ids, use_cache):
        assert use_cache is False
        self.calls.append(input_ids.detach().clone())
        h = (input_ids.float()[..., None]+torch.arange(self.width)[None, None, :]).to(torch.bfloat16)
        original = h.clone()
        for layer in self.layers: h = layer(h)
        assert torch.equal(h, original)
        return h


class TinyHF(torch.nn.Module):
    def __init__(self, width=16, vocab=32):
        super().__init__(); self.model = TinyText(width)
        self.lm_head = torch.nn.Linear(width, vocab, bias=False).to(torch.bfloat16)
        with torch.no_grad(): self.lm_head.weight.zero_()  # All ties exercise actual top-k handling.
        self.config = types.SimpleNamespace(get_text_config=lambda: types.SimpleNamespace(
            num_hidden_layers=12, hidden_size=width, final_logit_softcapping=None))


class Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.root = Path(self.tmp.name)
    def tearDown(self): self.tmp.cleanup()

    def test_import_inert(self):
        with patch.object(Path, 'open', side_effect=AssertionError('import file read')), \
             patch.object(Path, 'read_bytes', side_effect=AssertionError('import bytes')):
            module = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(module)

    def test_read_set_excludes_evaluation_and_pins_actual_implementation(self):
        pins = m.input_pins(P, F)
        self.assertFalse(any('evaluation-' in p.name for p in pins))
        self.assertNotIn(m.OUT/'excerpts/attribution.json', pins)
        self.assertEqual(pins[m.QWEN_SOURCE], m.QWEN_SOURCE_SHA)
        self.assertIn(m.DIRECTIONS, pins); self.assertIn(m.ORIGINAL, pins)
        with self.assertRaisesRegex(ValueError, 'evaluation file'):
            m.verify({self.root/'evaluation-tokens.json': 'unused'})

    def calibration_files(self):
        rows, pairs, records, manifest = panel()
        for name in ('token-preflight', 'excerpts', 'selection'): (self.root/name).mkdir()
        def write(path, value): path.write_text(json.dumps(value))
        write(self.root/'excerpts/calibration-dataset.json', rows)
        write(self.root/'excerpts/calibration-pairs.json', pairs)
        write(self.root/'selection/manifest.json', manifest)
        tokens = self.root/'token-preflight/calibration-tokens.json'
        write(tokens, {'schema': 'jlens_pattern_calibration_tokens_v1', 'role': 'calibration', 'records': records})
        receipt = {'schema': 'jlens_pattern_calibration_token_preflight_receipt_v1', 'status': 'complete',
            'source_sha256': m.PREFLIGHT_SOURCE_SHA, 'base_helper_sha256': P.BASE_SHA, 'host_helper_sha256': P.HOST_HELPER_SHA,
            'input_pins': {str(k): v for k, v in P.frozen_pins(F).items()}, 'count': 96,
            'role_counts': {'calibration': 64, 'evaluation': 32}, 'token_limit': 96, 'capture_locations': ['prefix_end'],
            'model_loaded': False, 'scientific_array_loaded': False, 'reference_decodes': 0,
            'outputs': [{'path': 'calibration-tokens.json', 'sha256': m.CALIBRATION_TOKENS_SHA, 'size_bytes': tokens.stat().st_size},
                        {'path': 'evaluation-tokens.json', 'sha256': 'metadata only, never opened'}]}
        write(self.root/'token-preflight/receipt.json', receipt)
        return rows, pairs, records, receipt

    def test_calibration_loader_never_opens_evaluation_even_when_missing(self):
        rows, pairs, records, receipt = self.calibration_files()
        seen = []; original_open = Path.open
        def guarded(path, *args, **kwargs):
            self.assertNotIn('evaluation-', path.name); seen.append(str(path))
            return original_open(path, *args, **kwargs)
        with patch.multiple(m, OUT=self.root, MAIN=self.root), patch.object(Path, 'open', guarded):
            self.assertEqual(m.load_calibration(P, F), (rows, pairs, records, receipt))
        self.assertTrue(seen)
        self.assertFalse((self.root/'token-preflight/evaluation-tokens.json').exists())

    def test_loader_rejects_wrong_role_record_position_source_and_old_roster(self):
        rows, pairs, records, receipt = self.calibration_files()
        tokenfile = self.root/'token-preflight/calibration-tokens.json'
        original = json.loads(tokenfile.read_text())
        for change in ('role', 'record', 'position', 'count'):
            bad = copy.deepcopy(original)
            if change == 'role': bad['role'] = 'evaluation'
            elif change == 'record': bad['records'][0]['id'] = 'T01-L'
            elif change == 'position': bad['records'][0]['captured_positions']['prefix_end'] = 0
            else: bad['records'] = bad['records'][:32]
            tokenfile.write_text(json.dumps(bad))
            r = copy.deepcopy(receipt); r['outputs'][0]['size_bytes'] = tokenfile.stat().st_size
            (self.root/'token-preflight/receipt.json').write_text(json.dumps(r))
            with patch.multiple(m, OUT=self.root, MAIN=self.root):
                with self.assertRaises(ValueError, msg=change): m.load_calibration(P, F)
        receipt['source_sha256'] = 'wrong'; (self.root/'token-preflight/receipt.json').write_text(json.dumps(receipt))
        with patch.multiple(m, OUT=self.root, MAIN=self.root):
            with self.assertRaisesRegex(ValueError, 'source/schema'): m.load_calibration(P, F)

    def calculate(self, target=None, model_change=False):
        rows, pairs, records, _ = panel(); refs = references()
        raw = (json.dumps(refs, ensure_ascii=False, indent=3)+'\n').encode()
        hf = TinyHF(); model = HFLensModel(hf, None, force_bos=False)
        lens = JacobianLens({11: torch.eye(16)}, n_prompts=3, d_model=16)
        tok = DecodeOnly(); u = np.eye(16, 4, dtype=np.float32); mean = np.zeros(16, dtype=np.float64)
        if model_change:
            old_decode = tok.decode
            def mutate(ids):
                with torch.no_grad(): hf.lm_head.weight.add_(1)
                return old_decode(ids)
            tok.decode = mutate
        with patch.multiple(m, WIDTH=16, VOCAB=32, ORIGINAL_SHA=__import__('hashlib').sha256(raw).hexdigest()), \
             patch.object(F, 'WIDTH', 16):
            result = m.calculate(target or self.root, F, PAT, hf, model, records, pairs, u, mean, refs, raw,
                                 lambda: (lens, tok), 'cpu')
        return result, hf, model, lens, tok, raw, records

    def test_whole_fake_capture_patterns_decodes_ties_exact_reference_bytes(self):
        topk = torch.Tensor.topk; calls = []
        def once(tensor, k, *args, **kwargs): calls.append(k); return topk(tensor, k, *args, **kwargs)
        with patch.object(torch.Tensor, 'topk', once):
            result, hf, model, lens, tok, raw, records = self.calculate()
        self.assertEqual(calls, [12]*8); self.assertEqual(len(tok.ids), 96)
        self.assertEqual(len(hf.model.calls), 64)
        self.assertEqual(hf.model.norm.seen, [torch.bfloat16]*8)
        self.assertEqual((self.root/'original-interpretations.json').read_bytes(), raw)
        with np.load(self.root/'features.npz', allow_pickle=False) as archive:
            h, u = archive['activation_11'], archive['u32']
            self.assertEqual(h.shape, (64, 1, 16)); self.assertEqual(h.dtype, np.float32)
        with np.load(self.root/'patterns.npz', allow_pickle=False) as archive:
            delta = h[::2, 0].astype(np.float64)-h[1::2, 0].astype(np.float64)
            dz = delta @ u.astype(np.float64)
            np.testing.assert_allclose(archive['cross64'], delta.T @ dz, rtol=1e-14, atol=1e-14)
            np.testing.assert_allclose(archive['score_energy64'], np.sum(dz**2, axis=0), rtol=1e-14, atol=1e-14)
            np.testing.assert_array_equal(archive['calibration_gaps64'], dz)
            signed = archive['signed_inputs32']
        with np.load(self.root/'decoder.npz', allow_pickle=False) as archive:
            np.testing.assert_array_equal(archive['inputs32'], signed)
            self.assertEqual(archive['logits32'].shape, (8, 32)); self.assertEqual(archive['top_ids64'].dtype, np.int64)
            self.assertTrue((archive['logits32'] == 0).all())
        readouts = json.loads((self.root/'readouts.json').read_text())['readouts']
        self.assertEqual([r['name'] for r in readouts], m.NAMES)
        self.assertTrue(all((r['cutoff_tie_count'], r['strictly_above_cutoff_count'], r['selected_at_cutoff_count']) == (32, 0, 12) for r in readouts))
        refs = json.loads((self.root/'references.json').read_text())
        self.assertEqual(refs['schema'], 'jlens_pattern_calibration_references_v1')
        for i, axis in enumerate(references()['axes']):
            for j, pole in enumerate(('positive', 'negative')):
                old = refs['arms']['U']['axes'][i][pole+'_reference']; new = refs['arms']['P']['axes'][i][pole+'_reference']
                self.assertEqual(old['example_prefix'], new['example_prefix'])
                self.assertEqual(old['example_prefix'], axis['C'][pole])
                self.assertEqual(old['direction_tokens'], axis['A'][pole])
                self.assertEqual(new['direction_tokens'], readouts[2*i+j]['tokens'])

    def test_zero_energy_preserves_states_and_never_loads_decoder(self):
        rows, pairs, records, _ = panel(); hf = TinyHF(); model = HFLensModel(hf, None, force_bos=False)
        factory = Mock(side_effect=AssertionError('decoder must not be loaded'))
        with patch.object(m, 'WIDTH', 16), patch.object(F, 'capture', return_value=np.zeros((64, 1, 16), dtype=np.float32)):
            with self.assertRaisesRegex(ValueError, 'zero score energy'):
                m.calculate(self.root, F, PAT, hf, model, records, pairs, np.eye(16, 4, dtype=np.float32),
                            np.zeros(16), references(), b'not used', factory, 'cpu')
        self.assertTrue((self.root/'features.npz').exists()); self.assertFalse((self.root/'patterns.npz').exists())
        factory.assert_not_called()

    def test_mutation_and_nonfinite_or_invalid_lens_rejected(self):
        with self.assertRaisesRegex(ValueError, 'mutation'): self.calculate(model_change=True)
        with patch.object(m, 'WIDTH', 16):
            lens = JacobianLens({11: torch.eye(16)}, n_prompts=3, d_model=16)
            lens.jacobians[11][0, 0] = float('nan')
            with self.assertRaisesRegex(ValueError, 'invalid FP32'): m.lens_state(lens)
            lens = JacobianLens({11: torch.eye(16)}, n_prompts=3, d_model=16)
            before = m.lens_state(lens); lens.jacobians[11].add_(1)
            self.assertNotEqual(before, m.lens_state(lens))

    def test_nonfinite_logits_and_bad_signed_inputs_fail_before_topk(self):
        hf = TinyHF(); model = HFLensModel(hf, None, force_bos=False)
        lens = JacobianLens({11: torch.eye(16)}, n_prompts=3, d_model=16)
        signed = np.repeat(np.eye(16, dtype=np.float32)[:4], 2, axis=0); signed[1::2] *= -1
        with patch.multiple(m, WIDTH=16, VOCAB=32):
            bad = signed.copy(); bad[1] = bad[0]
            with self.assertRaisesRegex(ValueError, 'signed unit'): m.decode(model, DecodeOnly(), lens, bad, 'cpu')
            with torch.no_grad(): hf.lm_head.weight.fill_(float('nan'))
            with self.assertRaisesRegex(ValueError, 'finite vocabulary'): m.decode(model, DecodeOnly(), lens, signed, 'cpu')

    def test_safe_direction_load_is_pinned_closed_and_immutable(self):
        path = self.root/'fabricated.npz'
        with path.open('wb') as handle: np.savez(handle, u32=np.eye(16, 4, dtype=np.float32), source_mean64=np.zeros(16))
        with patch.multiple(m, DIRECTIONS=path, DIRECTIONS_SHA=m.sha(path)), patch.object(F, 'WIDTH', 16):
            u, mean = m.load_directions(F)
            self.assertFalse(u.flags.writeable); self.assertFalse(mean.flags.writeable)
        with patch.multiple(m, DIRECTIONS=path, DIRECTIONS_SHA='wrong'):
            with self.assertRaisesRegex(ValueError, 'pin before'): m.load_directions(F)

    def test_release_source_and_consumed_stage_guards(self):
        envelope = {'JLENS_PATTERN_CALIBRATION_GPU_RELEASE': '1', 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
                    'MKL_NUM_THREADS': '1', 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'HF_HUB_DISABLE_IMPLICIT_TOKEN': '1'}
        with patch.dict(m.os.environ, {'JLENS_PATTERN_CALIBRATION_GPU_RELEASE': ''}):
            with self.assertRaisesRegex(ValueError, 'admission'): m.run('wrong')
        with patch.dict(m.os.environ, envelope), patch.object(m, 'OUT', self.root):
            with self.assertRaisesRegex(ValueError, 'reviewed calibration source'): m.run('0'*64)
            target = self.root/'calibration'; target.symlink_to(self.root/'absent')
            with patch.object(m, 'import_pinned', side_effect=AssertionError('must guard first')):
                with self.assertRaises(FileExistsError): m.run(m.sha(Path(m.__file__)))

    def test_full_wrapper_fake_success_receipt_and_once_only_failure(self):
        rows, pairs, records, _ = panel(); refs = references(); raw = json.dumps(refs).encode()
        hf = TinyHF(); model = HFLensModel(hf, None, force_bos=False)
        lens = JacobianLens({11: torch.eye(16)}, n_prompts=3, d_model=16)
        env = {'python': A.NEW_PYTHON, 'interpreter': '/usr/bin/python3', 'packages': A.NUMERICAL}
        host = {'python': A.NEW_PYTHON, 'interpreter': '/usr/bin/python3', 'resolved_interpreter': '/usr/bin/python3.12',
                'package_versions': A.PACKAGES, 'binary_sha256s': A.BINARIES}
        receipt = {'runtime': env, 'host': host}
        envelope = {'JLENS_PATTERN_CALIBRATION_GPU_RELEASE': '1', 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
                    'MKL_NUM_THREADS': '1', 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'HF_HUB_DISABLE_IMPLICIT_TOKEN': '1'}
        modules = {n: types.SimpleNamespace(__version__=v) for n, v in A.NUMERICAL.items() if n not in ('numpy', 'torch')}
        calculate = m.calculate
        def cpu_calculation(*args):
            self.assertEqual(args[-1], 'cuda')  # Production device contract, replaced only in this fixture.
            return calculate(*args[:-1], 'cpu')
        with patch.dict(m.os.environ, envelope), patch.dict(sys.modules, modules), \
             patch.multiple(m, OUT=self.root, WIDTH=16, VOCAB=32, ORIGINAL_SHA=__import__('hashlib').sha256(raw).hexdigest()), \
             patch.object(m, 'import_pinned', side_effect=[P, F, A, PAT]), \
             patch.object(m, 'input_pins', return_value={}), \
             patch.object(m, 'load_calibration', return_value=(rows, pairs, records, receipt)), \
             patch.object(m, 'original_references', return_value=(raw, refs)), \
             patch.object(m, 'load_directions', return_value=(np.eye(16, 4, dtype=np.float32), np.zeros(16))), \
             patch.object(m, 'load_decoder', return_value=(lens, DecodeOnly())), \
             patch.object(m, 'calculate', side_effect=cpu_calculation), \
             patch.multiple(F, WIDTH=16, runtime=lambda: env, load_model=lambda: (hf, model)), \
             patch.object(A, 'host_identity', return_value=host), \
             patch.object(torch.cuda, 'get_device_name', return_value='fabricated CPU-only'), \
             patch.object(torch.cuda, 'max_memory_allocated', return_value=0), \
             patch.object(torch.cuda, 'max_memory_reserved', return_value=0):
            m.run(m.sha(Path(m.__file__)))
            with self.assertRaises(FileExistsError): m.run(m.sha(Path(m.__file__)))
        done = json.loads((self.root/'calibration/receipt.json').read_text())
        self.assertEqual(done['schema'], 'jlens_pattern_calibration_calibration_receipt_v1')
        self.assertEqual((done['forward_count'], done['reference_decodes'], done['evaluation_rows_loaded']), (64, 8, 0))
        self.assertEqual(len(done['outputs']), 7)
        for artifact in done['outputs']:
            path = self.root/'calibration'/artifact['path']
            self.assertEqual(m.sha(path), artifact['sha256']); self.assertEqual(path.stat().st_size, artifact['size_bytes'])
        with tempfile.TemporaryDirectory() as directory, patch.dict(m.os.environ, envelope), \
             patch.object(m, 'OUT', Path(directory)), patch.object(m, 'import_pinned', side_effect=[P, F, A]), \
             patch.object(m, 'input_pins', side_effect=ValueError('fabricated pin failure')):
            with self.assertRaisesRegex(ValueError, 'fabricated pin failure'): m.run(m.sha(Path(m.__file__)))
            target = Path(directory)/'calibration'
            self.assertEqual({p.name for p in target.iterdir()}, {'attempt.json', 'failure.json'})
            self.assertEqual(json.loads((target/'failure.json').read_text())['phase'], 'inputs')


if __name__ == '__main__':
    unittest.main()
