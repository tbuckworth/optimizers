"""Synthetic panel/tokenizer/runtime fixtures only. No real tokenizer or stage inputs."""
import copy
import importlib.util
import json
from pathlib import Path
import re
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location('pattern_preflight', Path(__file__).with_name('preflight.py'))
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)
F = m.import_pinned(m.BASE, m.BASE_SHA)
A = m.import_pinned(m.HOST_HELPER, m.HOST_HELPER_SHA)


def panel():
    datasets, pairsets = {r: [] for r in m.ROLE_COUNTS}, {r: [] for r in m.ROLE_COUNTS}
    candidates, articles, dispositions = [], [], []
    for i in range(86):
        role = 'evaluation' if i % 3 == 2 else 'calibration'
        candidate = {'candidate_id': f'K{i+1:03}', 'role': role, 'topic': 'programming',
                     'left_pageid': 2*i+1, 'right_pageid': 2*i+2}
        candidates.append(candidate)
        if len(pairsets[role]) == m.ROLE_COUNTS[role]//2:
            dispositions.append({**candidate, 'status': 'capacity_skip_no_requests'})
            continue
        ident = ('C' if role == 'calibration' else 'T')+f'{len(pairsets[role])+1:02}'
        pairsets[role].append({'id': ident, 'left': ident+'-L', 'right': ident+'-R',
            'topic': 'programming', 'candidate_id': candidate['candidate_id'], 'role': role})
        dispositions.append({**candidate, 'status': 'accepted', 'accepted_pair_id': ident})
        for j, side in enumerate(('L', 'R')):
            row = {'id': ident+'-'+side, 'topic': 'programming', 'prefix': ' '.join([ident+side]+['word']*15)}
            datasets[role].append(row)
            articles.append({**row, 'pageid': 2*i+1+j, 'role': role, 'pair_id': ident,
                             'candidate_id': candidate['candidate_id']})
    manifest = {'schema': 'jlens_pattern_calibration_candidate_manifest_v1',
        'protocol_sha256': m.PROTOCOL_SHA, 'seed': '20260921', 'candidates': candidates}
    return datasets, pairsets, manifest, articles, dispositions


class FakeTokenizer:
    is_fast, bos_token_id = True, None
    def __init__(self): self.calls = []
    def __call__(self, text, **kwargs):
        assert kwargs == {'add_special_tokens': False, 'padding': False, 'truncation': False,
                          'return_attention_mask': True, 'return_offsets_mapping': True}
        self.calls.append(text)
        offsets = [(x.start(), x.end()) for x in re.finditer(r'\S+', text)]
        return {'input_ids': list(range(len(offsets))), 'attention_mask': [1]*len(offsets),
                'offset_mapping': offsets}


def environment():
    env = {'python': A.NEW_PYTHON, 'interpreter': '/usr/bin/python3', 'packages': dict(A.NUMERICAL)}
    host = {'python': A.NEW_PYTHON, 'interpreter': '/usr/bin/python3', 'resolved_interpreter': '/usr/bin/python3.12',
            'package_versions': dict(A.PACKAGES), 'binary_sha256s': dict(A.BINARIES)}
    return env, host


class Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.root = Path(self.tmp.name)
    def tearDown(self): self.tmp.cleanup()

    def test_import_inert(self):
        with patch.object(Path, 'read_bytes', side_effect=AssertionError('import read')), \
             patch.object(Path, 'open', side_effect=AssertionError('import open')):
            module = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(module)

    def test_exact_role_panel_and_all_joins(self):
        m.validate_panel(*panel())
        for change in ('roworder', 'old32', 'role', 'crossjoin', 'topic', 'prefix', 'sharedpage', 'wrongpage', 'candidate', 'disposition'):
            p = panel(); data, pairs, manifest, articles, decisions = p
            if change == 'roworder': data['calibration'].reverse()
            elif change == 'old32': data['calibration'] = data['calibration'][:32]
            elif change == 'role': pairs['calibration'][0]['role'] = 'evaluation'
            elif change == 'crossjoin': pairs['evaluation'][0]['left'] = 'C01-L'
            elif change == 'topic': data['calibration'][0]['topic'] = 'cooking'
            elif change == 'prefix': data['evaluation'][0]['prefix'] = data['calibration'][0]['prefix']
            elif change == 'sharedpage': articles[2]['pageid'] = articles[0]['pageid']
            elif change == 'wrongpage': articles[0]['pageid'] = 999
            elif change == 'candidate': manifest['candidates'][0]['role'] = 'evaluation'
            else: decisions[0]['status'] = 'rejected'
            with self.assertRaises((ValueError, KeyError), msg=change): m.validate_panel(*p)

    def test_exact_word_count_whitespace_and_field_guards(self):
        for text in ('short', ' word '*16, ' '.join(['x']*17)):
            p = panel(); p[0]['calibration'][0]['prefix'] = text
            with self.assertRaises(ValueError): m.validate_panel(*p)
        p = panel(); p[0]['calibration'][0]['new_field'] = 1
        with self.assertRaises(ValueError): m.validate_panel(*p)

    def test_pure_tokenize_whole96_and_last_subtoken(self):
        tokenizer = FakeTokenizer(); data = panel()[0]
        records = {role: F.tokenize(rows, tokenizer) for role, rows in data.items()}
        self.assertEqual(len(tokenizer.calls), 96)
        self.assertEqual(list(map(len, records.values())), [64, 32])
        for role in data:
            for row, record in zip(data[role], records[role], strict=True):
                self.assertEqual(record, F.token_record(row, record))
                self.assertEqual(record['captured_positions'], {'prefix_end': 15})
                self.assertEqual(record['selected_substrings'], {'prefix_end': 'word'})

    def test_token_limits_masks_offsets_and_unicode_overlap(self):
        row = panel()[0]['calibration'][0]; encoded = FakeTokenizer()(row['prefix'],
            add_special_tokens=False, padding=False, truncation=False, return_attention_mask=True, return_offsets_mapping=True)
        for change in ('empty', '97', 'mask', 'boolid', 'zerooffset', 'incomplete', 'gap', 'reverse'):
            value = copy.deepcopy(encoded)
            if change == 'empty': value['input_ids'] = []
            elif change == '97': value['input_ids'] = [1]*97
            elif change == 'mask': value['attention_mask'][0] = 0
            elif change == 'boolid': value['input_ids'][0] = True
            elif change == 'zerooffset': value['offset_mapping'][0] = (0, 0)
            elif change == 'incomplete': value['offset_mapping'][-1] = (len(row['prefix'])-2, len(row['prefix'])-1)
            elif change == 'gap': value['offset_mapping'][0] = (0, 1)
            else: value['offset_mapping'].reverse()
            with self.assertRaises(ValueError, msg=change): F.token_record(row, value)
        value = {'input_ids': list(range(96)), 'attention_mask': [1]*96,
                 'offset_mapping': [(0, len(row['prefix']))]*96}
        self.assertEqual(F.token_record(row, value)['captured_positions']['prefix_end'], 95)
        unicode_row = {'id': 'synthetic', 'topic': 'programming', 'prefix': 'é word'}
        value = {'input_ids': [1, 2, 3], 'attention_mask': [1, 1, 1], 'offset_mapping': [(0, 1), (0, 1), (2, 6)]}
        self.assertEqual(F.token_record(unicode_row, value)['selected_substrings']['prefix_end'], 'word')

    def test_current_runtime_only_and_exact_binary_packages(self):
        env, host = environment(); m.validate_runtime(env, host, A)
        for change in ('oldpython', 'binary', 'distro', 'numerical', 'interpreter'):
            e, h = environment()
            if change == 'oldpython': e['python'] = A.OLD_PYTHON
            elif change == 'binary': h['binary_sha256s']['/usr/bin/python3.12'] = 'wrong'
            elif change == 'distro': h['package_versions']['libc6'] = 'wrong'
            elif change == 'numerical': e['packages']['numpy'] = 'wrong'
            else: h['resolved_interpreter'] = '/another/python'
            with self.assertRaises(ValueError, msg=change): m.validate_runtime(e, h, A)

    def test_cache_only_scope_and_missing_collection_pins(self):
        pins = m.frozen_pins(F)
        self.assertFalse(any(str(p).endswith(('.npz', '.safetensors', '.pt')) for p in pins))
        self.assertEqual({p.name for p in pins if p.parent == F.SNAPSHOT}, set(m.CACHE_NAMES))
        with patch.object(m, 'COLLECTION_SHA', None):
            with self.assertRaisesRegex(ValueError, 'audited collection pins'): m.frozen_pins(F)
        path = self.root/'pin'; path.write_text('fabricated')
        m.verify({path: m.sha(path)})
        with self.assertRaisesRegex(ValueError, 'input pin mismatch'): m.verify({path: 'wrong'})

    def test_local_tokenizer_factory_explicit_arguments(self):
        calls = []
        def factory(*args, **kwargs): calls.append((args, kwargs)); return FakeTokenizer()
        stub = types.SimpleNamespace(AutoTokenizer=types.SimpleNamespace(from_pretrained=factory))
        with patch.dict(sys.modules, {'transformers': stub}): m.load_tokenizer(F)
        self.assertEqual(calls, [((F.MODEL,), {'revision': F.REV, 'cache_dir': str(F.HUB),
            'local_files_only': True, 'trust_remote_code': False, 'use_fast': True, 'token': False})])

    def test_collection_receipt_binding_with_fabricated_files(self):
        datasets, pairsets, manifest, articles, dispositions = panel()
        excerpts = self.root/'excerpts'; excerpts.mkdir()
        main = self.root/'main'; (main/'selection').mkdir(parents=True)
        (main/'selection/manifest.json').write_text(json.dumps(manifest))
        values = {**{r+'-dataset.json': datasets[r] for r in m.ROLE_COUNTS},
                  **{r+'-pairs.json': pairsets[r] for r in m.ROLE_COUNTS},
                  'attribution.json': {'schema': 'jlens_pattern_calibration_attribution_v1', 'articles': articles},
                  'candidate-dispositions.json': dispositions}
        outputs = []
        for name, value in values.items():
            path = excerpts/name; path.write_text(json.dumps(value))
            outputs.append({'path': name, 'sha256': m.sha(path), 'size_bytes': path.stat().st_size})
        receipt = {'schema': 'jlens_pattern_calibration_excerpts_v1', 'status': 'complete',
            'source_sha256': m.COLLECTOR_SHA, 'protocol_sha256': m.PROTOCOL_SHA, 'manifest_sha256': m.MANIFEST_SHA,
            'selected_count': 96, 'accepted_pairs': {'calibration': 32, 'evaluation': 16},
            'requested_pairs': 48, 'article_requests': 96, 'candidate_dispositions': 86,
            'tokenizer_or_model_called': False, 'outputs': outputs}
        (excerpts/'receipt.json').write_text(json.dumps(receipt))
        with patch.multiple(m, OUT=self.root, MAIN=main,
                            COLLECTION_OUTPUTS={r['path']: r['sha256'] for r in outputs}):
            self.assertEqual(m.load_panel(F), (datasets, pairsets))
            receipt['outputs'][0]['sha256'] = 'wrong'
            (excerpts/'receipt.json').write_text(json.dumps(receipt))
            with self.assertRaisesRegex(ValueError, 'output pin'): m.load_panel(F)

    def test_admission_and_reviewed_source_guards_precede_attempt(self):
        with patch.dict(m.os.environ, {'JLENS_PATTERN_CALIBRATION_PREFLIGHT_RELEASE': ''}):
            with self.assertRaisesRegex(ValueError, 'admission'): m.run('wrong', self.root)
        with patch.dict(m.os.environ, {'JLENS_PATTERN_CALIBRATION_PREFLIGHT_RELEASE': '1',
            'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}):
            with self.assertRaisesRegex(ValueError, 'reviewed preflight source'): m.run('0'*64, self.root)
        self.assertFalse((self.root/'token-preflight').exists())

    def synthetic_run(self, root, tokenizer=None, drift=False):
        data, pairs, *_ = panel(); env, host = environment(); tokenizer = tokenizer or FakeTokenizer()
        fake_f = types.SimpleNamespace(runtime=lambda: env, tokenize=F.tokenize, token_record=F.token_record)
        fake_a = types.SimpleNamespace(**{name: getattr(A, name) for name in ('NEW_PYTHON', 'NUMERICAL', 'PACKAGES', 'BINARIES')},
                                      host_identity=lambda: host)
        modules = {n: types.SimpleNamespace(__version__=v) for n, v in A.NUMERICAL.items()}
        envelope = {'JLENS_PATTERN_CALIBRATION_PREFLIGHT_RELEASE': '1', 'CUDA_VISIBLE_DEVICES': '',
                    'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
        with patch.dict(m.os.environ, envelope), patch.dict(sys.modules, modules), \
             patch.object(m, 'import_pinned', side_effect=[fake_f, fake_a]), \
             patch.object(m, 'frozen_pins', return_value={}), \
             patch.object(m, 'load_panel', return_value=(data, pairs)), \
             patch.object(m, 'load_tokenizer', return_value=tokenizer), \
             patch.object(m, 'verify', side_effect=[None, ValueError('synthetic drift')] if drift else None):
            m.run(m.sha(Path(m.__file__)), root)
        return tokenizer

    def test_exclusive_complete_stage_and_separate_role_records(self):
        tokenizer = self.synthetic_run(self.root)
        self.assertEqual(len(tokenizer.calls), 96)
        target = self.root/'token-preflight'
        receipt = json.loads((target/'receipt.json').read_text())
        self.assertEqual(receipt['role_counts'], {'calibration': 64, 'evaluation': 32})
        self.assertEqual(receipt['count'], 96)
        self.assertFalse(receipt['model_loaded']); self.assertFalse(receipt['scientific_array_loaded'])
        for role, count in m.ROLE_COUNTS.items():
            value = json.loads((target/(role+'-tokens.json')).read_text())
            self.assertEqual(value['role'], role); self.assertEqual(len(value['records']), count)
        with self.assertRaises(FileExistsError): self.synthetic_run(self.root)

    def test_whole_panel_failure_no_partial_success_and_dangling_guard(self):
        class BadLast(FakeTokenizer):
            def __call__(self, text, **kwargs):
                value = super().__call__(text, **kwargs)
                if len(self.calls) == 96: value['input_ids'] = [1]*97
                return value
        with self.assertRaises(ValueError): self.synthetic_run(self.root, BadLast())
        target = self.root/'token-preflight'
        self.assertEqual({p.name for p in target.iterdir()}, {'attempt.json', 'failure.json'})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root/'token-preflight').symlink_to(root/'missing')
            with self.assertRaises(FileExistsError): self.synthetic_run(root)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, 'synthetic drift'): self.synthetic_run(root, drift=True)
            self.assertFalse((root/'token-preflight/receipt.json').exists())


if __name__ == '__main__':
    unittest.main()
