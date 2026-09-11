"""Fabricated CPU-only evaluation fixtures; no real input/model/archive/stage."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import numpy as np
import torch

SPEC = importlib.util.spec_from_file_location('pattern_evaluate', Path(__file__).with_name('evaluate.py'))
m = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(m)
C = m.import_helper()
P = C.import_pinned(C.OUT/'preflight.py', C.PREFLIGHT_SOURCE_SHA)
F = C.import_pinned(P.BASE, P.BASE_SHA)
A = C.import_pinned(P.HOST_HELPER, P.HOST_HELPER_SHA)
torch.set_num_threads(1)
ENVELOPE = {k: '1' for k in ('JLENS_PATTERN_EVALUATION_GPU_RELEASE', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
    'MKL_NUM_THREADS', 'HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_HUB_DISABLE_IMPLICIT_TOKEN')}


def panel():
    candidates = [{'candidate_id': f'K{i+1:03}', 'topic': 'programming',
        'role': 'evaluation' if i % 3 == 2 else 'calibration'} for i in range(86)]
    rows, pairs, records = [], [], []
    for i, candidate in enumerate([c for c in candidates if c['role'] == 'evaluation'][:16], 1):
        ident = f'T{i:02}'
        pairs.append({'id': ident, 'left': ident+'-L', 'right': ident+'-R', 'topic': 'programming',
            'candidate_id': candidate['candidate_id'], 'role': 'evaluation'})
        for side in ('L', 'R'):
            row = {'id': ident+'-'+side, 'topic': 'programming', 'prefix': ' '.join([ident+side]+['word']*15)}
            encoded = {'input_ids': list(range(16*len(rows), 16*len(rows)+16)), 'attention_mask': [1]*16,
                'offset_mapping': [[v.start(), v.end()] for v in re.finditer(r'\S+', row['prefix'])]}
            rows.append(row); records.append(F.token_record(row, encoded))
    manifest = {'schema': 'jlens_pattern_calibration_candidate_manifest_v1', 'protocol_sha256': P.PROTOCOL_SHA,
                'seed': '20260921', 'candidates': candidates}
    return rows, pairs, records, manifest


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.layers = torch.nn.ModuleList([torch.nn.Identity() for _ in range(12)])
        self.weight = torch.nn.Parameter(torch.ones(4), requires_grad=False)
        self.eval(); self.calls = []
    def forward(self, ids):
        assert not torch.is_grad_enabled()
        self.calls.append(ids.clone())
        h = (ids.float()[..., None]+torch.arange(4)[None, None, :]).to(torch.bfloat16)
        original = h.clone()
        for layer in self.layers: h = layer(h)
        assert torch.equal(original, h)
        return h


class Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.root = Path(self.tmp.name)
    def tearDown(self): self.tmp.cleanup()
    def put(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value), encoding='utf-8')

    def test_import_inert(self):
        with patch.object(Path, 'read_bytes', side_effect=AssertionError('import read')), \
             patch.object(Path, 'open', side_effect=AssertionError('import open')):
            module = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(module)

    def test_input_set_has_no_calibration_arrays_tokens_or_lens(self):
        pins = m.input_pins(C, P, F)
        self.assertIn(C.DIRECTIONS, pins)
        self.assertIn(m.OUT/'calibration/references.json', pins)
        self.assertIn(C.QWEN_SOURCE, pins)
        for name in ('calibration-tokens.json', 'calibration-dataset.json', 'features.npz', 'patterns.npz', 'decoder.npz'):
            self.assertFalse(any(p.name == name for p in pins))
            with self.assertRaisesRegex(ValueError, 'forbidden'):
                m.verify(C, {self.root/name: 'not-read'})
        self.assertFalse(any(p.suffix == '.pt' for p in pins))

    def test_roster_roles_order_words_and_manifest(self):
        rows, pairs, records, manifest = panel()
        m.validate_rows(rows, pairs, manifest, P, F)
        for change in ('row_order', 'old_id', 'pair_side', 'role', 'topic', 'wordcount', 'candidate_role'):
            r, p, man = copy.deepcopy((rows, pairs, manifest))
            if change == 'row_order': r[0], r[1] = r[1], r[0]
            elif change == 'old_id': r[0]['id'] = 'C01-L'
            elif change == 'pair_side': p[0]['left'] = p[0]['right']
            elif change == 'role': p[0]['role'] = 'calibration'
            elif change == 'topic': r[0]['topic'] = 'cooking'
            elif change == 'wordcount': r[0]['prefix'] += ' extra'
            else: man['candidates'][2]['role'] = 'calibration'
            with self.assertRaises(ValueError, msg=change): m.validate_rows(r, p, man, P, F)

    def files(self):
        rows, pairs, records, manifest = panel()
        self.put(self.root/'excerpts/evaluation-dataset.json', rows)
        self.put(self.root/'excerpts/evaluation-pairs.json', pairs)
        self.put(self.root/'selection/manifest.json', manifest)
        tokenpath = self.root/'token-preflight/evaluation-tokens.json'
        self.put(tokenpath, {'schema': 'jlens_pattern_calibration_tokens_v1', 'role': 'evaluation', 'records': records})
        preflight = {'schema': 'jlens_pattern_calibration_token_preflight_receipt_v1', 'status': 'complete',
            'source_sha256': C.PREFLIGHT_SOURCE_SHA, 'base_helper_sha256': P.BASE_SHA, 'host_helper_sha256': P.HOST_HELPER_SHA,
            'input_pins': {str(k): v for k, v in P.frozen_pins(F).items()}, 'count': 96,
            'role_counts': {'calibration': 64, 'evaluation': 32}, 'token_limit': 96, 'capture_locations': m.LOCATIONS,
            'model_loaded': False, 'scientific_array_loaded': False, 'reference_decodes': 0,
            'outputs': [{'path': 'calibration-tokens.json'}, {'path': 'evaluation-tokens.json',
                'sha256': m.EVALUATION_TOKENS_SHA, 'size_bytes': tokenpath.stat().st_size}]}
        self.put(self.root/'token-preflight/receipt.json', preflight)
        collection = {'schema': 'jlens_pattern_calibration_excerpts_v1', 'status': 'complete',
            'source_sha256': P.COLLECTOR_SHA, 'protocol_sha256': P.PROTOCOL_SHA, 'manifest_sha256': P.MANIFEST_SHA,
            'selected_count': 96, 'accepted_pairs': {'calibration': 32, 'evaluation': 16},
            'outputs': [{'path': n, 'sha256': P.COLLECTION_OUTPUTS[n], 'size_bytes': (self.root/'excerpts'/n).stat().st_size}
                        for n in ('evaluation-dataset.json', 'evaluation-pairs.json')]}
        self.put(self.root/'excerpts/receipt.json', collection)
        return rows, pairs, records, preflight

    def test_loader_without_any_calibration_files_and_position_rejection(self):
        expected = self.files(); opened = []; original = Path.open
        def guard(path, *args, **kwargs):
            self.assertNotIn('calibration-', path.name); opened.append(path)
            return original(path, *args, **kwargs)
        with patch.multiple(m, OUT=self.root, MAIN=self.root), patch.object(Path, 'open', guard):
            self.assertEqual(m.load_evaluation(C, P, F), expected)
        self.assertTrue(opened)
        tokenpath = self.root/'token-preflight/evaluation-tokens.json'
        value = json.loads(tokenpath.read_text()); value['records'][0]['captured_positions']['prefix_end'] = 0
        self.put(tokenpath, value)
        preflight = expected[-1]; preflight['outputs'][1]['size_bytes'] = tokenpath.stat().st_size
        self.put(self.root/'token-preflight/receipt.json', preflight)
        with patch.multiple(m, OUT=self.root, MAIN=self.root):
            with self.assertRaisesRegex(ValueError, 'position mismatch'): m.load_evaluation(C, P, F)

    def test_reference_commit_binds_three_exact_blobs_before_arrays(self):
        blobs = {n: ('fabricated '+n).encode() for n in ('receipt.json', 'references.json', 'original-interpretations.json')}
        pins = {n: hashlib.sha256(v).hexdigest() for n, v in blobs.items()}
        path = self.root/'calibration'; path.mkdir()
        for n, raw in blobs.items(): (path/n).write_bytes(raw)
        receipt = {'schema': 'jlens_pattern_calibration_calibration_receipt_v1', 'status': 'complete',
            'source_sha256': m.HELPER_SHA, 'preflight_receipt_sha256': C.PREFLIGHT_RECEIPT_SHA,
            'forward_count': 64, 'calibration_pairs': 32, 'reference_decodes': 8, 'topk_calls': 8,
            'evaluation_rows_loaded': 0, 'evaluation_forwards': 0, 'prefix_tokenizations': 0,
            'parameters_unchanged': True, 'lens_parameters_unchanged': True, 'pca_refit': False,
            'outputs': [{'path': n, 'sha256': pins.get(n), 'size_bytes': len(blobs.get(n, b''))}
                for n in ('references.json', 'original-interpretations.json', 'features.npz', 'patterns.npz',
                          'decoder.npz', 'inputs.json', 'readouts.json')]}
        with patch.multiple(m, ROOT=self.root, OUT=self.root, CALIBRATION_RECEIPT_SHA=pins['receipt.json'],
                            REFERENCE_SHA=pins['references.json']), patch.object(C, 'ORIGINAL_SHA', pins['original-interpretations.json']), \
             patch.object(m.subprocess, 'run') as ancestry, \
             patch.object(m.subprocess, 'check_output', side_effect=lambda args, **kw: blobs[Path(args[2]).name]) as show, \
             patch.object(F, 'read_json', return_value=receipt):
            self.assertEqual(m.committed_references(C, F), receipt)
            self.assertEqual(show.call_count, 3); self.assertTrue(ancestry.call_args.kwargs['check'])
            (path/'references.json').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'commit/file'): m.committed_references(C, F)
        with patch.object(m.subprocess, 'run', side_effect=subprocess.CalledProcessError(1, 'git')), \
             patch.object(F, 'read_json', side_effect=AssertionError('must check commit first')):
            with self.assertRaises(subprocess.CalledProcessError): m.committed_references(C, F)

    def test_actual_pure_capture_on_tiny_cpu_model_and_scalar_export(self):
        rows, pairs, records, _ = panel(); hf = Tiny()
        u, mean = np.eye(4, dtype=np.float32), np.array([.125, -.375, .2, .75], dtype=np.float64)
        with patch.object(F, 'WIDTH', 4): m.calculate(C, F, self.root, hf, hf, rows, pairs, records, u, mean, 'cpu')
        self.assertEqual(len(hf.calls), 32)
        for call, record in zip(hf.calls, records): self.assertEqual(call[0].tolist(), record['input_ids'])
        self.assertFalse(hf.layers[11]._forward_hooks)
        with np.load(self.root/'features.npz', allow_pickle=False) as archive:
            h, s, g = archive['activation_11'], archive['scores64'], archive['gaps64']
            self.assertEqual(h.shape, (32, 1, 4)); self.assertEqual(h.dtype, np.float32)
            np.testing.assert_array_equal(s, (h.astype(np.float64)-mean) @ u.astype(np.float64))
            np.testing.assert_array_equal(g, s[::2]-s[1::2])
        exported = json.loads((self.root/'scores.json').read_text())
        self.assertEqual(exported['schema'], 'jlens_pattern_calibration_evaluation_scores_v1')
        for a, row in enumerate(exported['locations']['prefix_end']):
            self.assertEqual(list(row['values']), m.IDS)
            np.testing.assert_array_equal(list(row['values'].values()), s[:, 0, a])

    def test_exact_saved_score_gap_not_delta_projection_and_invalid_arrays(self):
        rows, pairs, _, _ = panel(); h = np.zeros((32, 1, 4), dtype=np.float32); h[::2] = 1
        u = np.eye(4, dtype=np.float32); mean = np.full(4, 2.**54)
        with patch.object(F, 'WIDTH', 4):
            s, g, _ = F.score_export(h, u, mean, rows, pairs)
            np.testing.assert_array_equal(g, s[::2]-s[1::2])
            self.assertFalse(np.array_equal(g, (h[::2].astype(np.float64)-h[1::2]) @ u.astype(np.float64)))
            for bad in (h[:, 0], h.astype(np.float64), np.full_like(h, np.nan)):
                with self.assertRaises(ValueError): F.score_export(bad, u, mean, rows, pairs)

    def test_model_mutation_or_training_fails_before_export(self):
        rows, pairs, records, _ = panel(); hf = Tiny()
        def mutate(*args):
            hf.weight.add_(1); return np.zeros((32, 1, 4), dtype=np.float32)
        with patch.object(F, 'capture', side_effect=mutate):
            with self.assertRaisesRegex(ValueError, 'mutation'):
                m.calculate(C, F, self.root, hf, hf, rows, pairs, records, np.eye(4, dtype=np.float32), np.zeros(4), 'cpu')
        self.assertFalse(list(self.root.iterdir()))
        hf.train()
        with self.assertRaisesRegex(ValueError, 'eval'): F.parameter_state(hf)

    def test_release_source_consumed_guard_and_helper_failure_attempt(self):
        with patch.dict(m.os.environ, {'JLENS_PATTERN_EVALUATION_GPU_RELEASE': ''}):
            with self.assertRaisesRegex(ValueError, 'admission'): m.run('bad')
        with patch.dict(m.os.environ, ENVELOPE), patch.object(m, 'OUT', self.root):
            with self.assertRaisesRegex(ValueError, 'reviewed'): m.run('0'*64)
            with patch.object(m, 'import_helper', side_effect=ValueError('fabricated helper mismatch')):
                with self.assertRaises(ValueError): m.run(C.sha(Path(m.__file__)))
            target = self.root/'evaluation'
            self.assertEqual({p.name for p in target.iterdir()}, {'attempt.json', 'failure.json'})
            self.assertNotIn('fabricated', (target/'failure.json').read_text())
            with self.assertRaises(FileExistsError): m.run(C.sha(Path(m.__file__)))
        with tempfile.TemporaryDirectory() as d, patch.object(m, 'OUT', Path(d)), patch.dict(m.os.environ, ENVELOPE):
            (Path(d)/'evaluation').symlink_to(Path(d)/'absent')
            with patch.object(m, 'import_helper', side_effect=AssertionError('must not import')):
                with self.assertRaises(FileExistsError): m.run(C.sha(Path(m.__file__)))

    def test_full_fake_wrapper_receipt_contains_no_key(self):
        rows, pairs, records, _ = panel(); hf = Tiny()
        env = {'python': A.NEW_PYTHON, 'interpreter': '/usr/bin/python3', 'packages': A.NUMERICAL}
        host = {'python': A.NEW_PYTHON, 'interpreter': '/usr/bin/python3', 'resolved_interpreter': '/usr/bin/python3.12',
                'package_versions': A.PACKAGES, 'binary_sha256s': A.BINARIES}
        prior = {'runtime': env, 'host': host}
        versions = {n: types.SimpleNamespace(__version__=v) for n, v in A.NUMERICAL.items() if n not in ('numpy', 'torch')}
        calculate = m.calculate
        def cpu(*args):
            self.assertEqual(args[-1], 'cuda'); return calculate(*args[:-1], 'cpu')
        with patch.dict(m.os.environ, ENVELOPE), patch.dict(sys.modules, versions), patch.object(m, 'OUT', self.root), \
             patch.object(m, 'import_helper', return_value=C), patch.object(C, 'import_pinned', side_effect=[P, F, A]), \
             patch.object(m, 'input_pins', return_value={}), patch.object(m, 'committed_references', return_value=prior), \
             patch.object(m, 'load_evaluation', return_value=(rows, pairs, records, prior)), \
             patch.object(C, 'load_directions', return_value=(np.eye(4, dtype=np.float32), np.zeros(4))), \
             patch.object(m, 'calculate', side_effect=cpu), patch.multiple(F, WIDTH=4, runtime=lambda: env, load_model=lambda: (hf, hf)), \
             patch.object(A, 'host_identity', return_value=host), patch.object(torch.cuda, 'get_device_name', return_value='fabricated CPU'), \
             patch.object(torch.cuda, 'max_memory_allocated', return_value=0), patch.object(torch.cuda, 'max_memory_reserved', return_value=0):
            m.run(C.sha(Path(m.__file__)))
        receipt = json.loads((self.root/'evaluation/receipt.json').read_text())
        self.assertEqual(receipt['forward_count'], 32); self.assertEqual(receipt['evaluation_pairs'], 16)
        self.assertEqual(receipt['reference_commit'], m.REFERENCE_COMMIT)
        self.assertEqual(len(receipt['outputs']), 4)
        self.assertFalse(receipt['lens_loaded']); self.assertFalse(receipt['pattern_fit'])
        self.assertFalse({'scores', 'gaps', 'positive_count', 'accuracy'} & set(receipt))
        for row in receipt['outputs']:
            path = self.root/'evaluation'/row['path']
            self.assertEqual(C.sha(path), row['sha256']); self.assertEqual(path.stat().st_size, row['size_bytes'])


if __name__ == '__main__': unittest.main()
