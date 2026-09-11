"""Fabricated JSON/history fixtures only. Never load actual packets or keys."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location('pattern_responses', Path(__file__).with_name('responses.py'))
m = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(m)
P = m.helper()


def fixture():
    refs = {'schema': 'jlens_pattern_calibration_references_v1', 'arms': {a: {'axes': [
        {'axis': x, **{s+'_reference': {'example_prefix': x+s, 'direction_tokens': ['', '\n', '中文']+[a+s]*9}
                       for s in ('positive', 'negative')}} for x in m.AXES]} for a in m.ARMS}}
    rows, pairs = [], []
    for i, pair in enumerate(m.PAIRS, 1):
        topic = m.TOPICS[i % 4]
        pairs.append({'id': pair, 'left': pair+'-L', 'right': pair+'-R', 'topic': topic, 'role': 'evaluation',
                      'candidate_id': f'K{3*i:03}'})
        rows.extend({'id': pair+'-'+s, 'topic': topic, 'prefix': ' '.join([pair+s]+['word']*15)} for s in ('L', 'R'))
    public, mapping = P.make_packets(refs, rows, pairs)
    scores = {'schema': 'jlens_pattern_calibration_evaluation_scores_v1', 'locations': {'prefix_end': [
        {'axis': a, 'values': {r['id']: float(i % 2 == 0) for i, r in enumerate(rows)}} for a in m.AXES]}}
    choices = {r: {} for r in m.RATERS}
    for v in mapping['rows']:
        truth = 'FIRST' if v['first_id'].endswith('-L') else 'SECOND'
        choices[v['rater']][v['item_id']] = truth
    return refs, rows, pairs, public, mapping, scores, choices


class Tests(unittest.TestCase):
    def setUp(self): self.tmp = tempfile.TemporaryDirectory(); self.root = Path(self.tmp.name)
    def tearDown(self): self.tmp.cleanup()
    def put(self, name, value):
        path = self.root/name; path.parent.mkdir(parents=True, exist_ok=True)
        raw = P.encode(value); path.write_bytes(raw)
        return {'path': str(path), 'sha256': m.sha(raw), 'size_bytes': len(raw)}
    def metadata_fixture(self):
        refs, rows, pairs, public, mapping, scores, choices = fixture()
        pins = {n: self.put('inputs/'+n+'.json', v) for n, v in [('references', refs), ('dataset', rows), ('pairs', pairs),
            ('protocol', {}), ('preflight', {}), ('calibration_receipt', {}), ('random_source', {})]}
        outputs = []
        for r in m.RATERS:
            item = self.put('packets/'+r+'.json', public[r]); outputs.append({**item, 'path': r+'.json'})
        private = self.put('packets/private-map.json', mapping); outputs.append({**private, 'path': 'private-map.json'})
        manifest = {'schema': 'jlens_pattern_calibration_packets_v1', 'seed': 20260922, 'source_sha256': m.PACKET_SHA,
            'python': sys.version, 'reference_commit': m.REFERENCE_COMMIT,
            'inputs': {n: {'path': v['path'], 'sha256': v['sha256']} for n, v in pins.items()},
            'outputs': outputs, 'score_key_or_array_files_opened': 0, 'completed_utc': 'fabricated'}
        manifest_pin = self.put('packets/manifest.json', manifest)
        evals = {'source': {'path': str(self.root/'evaluation/evaluate.py'), 'size_bytes': 1, 'sha256': m.EVALUATOR_SHA},
            **{n: {'path': str(self.root/'evaluation'/f'{n}.json'), 'size_bytes': 1, 'sha256': '0'*64} for n in ('receipt', 'scores')}}
        release = {'schema': 'jlens_pattern_calibration_judging_release_v1', 'reference_commit': m.REFERENCE_COMMIT,
                   'packet_commit': '1'*40, 'packet_manifest': manifest_pin, 'evaluation': evals}
        release_pin = self.put('release.json', release)
        raw_paths = {r: self.put('raw/'+r+'.json', {'schema': 'jlens_pattern_calibration_responses_v1',
            'packet_id': public[r]['packet_id'], 'responses': [{'item_id': k, 'choice': v} for k, v in choices[r].items()]}) for r in m.RATERS}
        self.fakepins = {n: (Path(v['path']), v['sha256']) for n, v in pins.items()}
        self.kw = {'release_path': release_pin['path'], 'release_sha': release_pin['sha256'], 'release_commit': '2'*40}
        self.raw_paths, self.choices, self.release_value = raw_paths, choices, release
        return raw_paths
    def sealed(self):
        self.metadata_fixture()
        with patch.object(m, 'helper', return_value=P), patch.object(P, 'PINS', self.fakepins), \
             patch.object(m, 'committed'), patch.object(m, 'ancestor'), \
             patch.dict(m.os.environ, {'JLENS_PATTERN_SEAL_RELEASE': '1'}):
            pin = m.seal(**self.kw, raw_commit='3'*40, raw_paths=self.raw_paths, target=self.root/'responses')
        return {**self.kw, 'responses': self.root/'responses', 'lock_sha': pin['sha256'], 'lock_commit': '4'*40}

    def test_import_inert(self):
        with patch.object(Path, 'open', side_effect=AssertionError('no import I/O')), \
             patch.object(Path, 'read_bytes', side_effect=AssertionError('no import read')):
            value = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(value)

    def test_perfect_ties_controls_and_counts(self):
        *_, mapping, scores, choices = fixture()
        result = m.grade_values(P, mapping, choices, scores)
        self.assertEqual([v['credit'] for v in result['arms'].values()], [128, 128])
        self.assertFalse(result['primary']['pilot_criterion_met'])
        self.assertEqual(len(result['items']), 256); self.assertEqual(len(result['gains_harm']), 128)
        self.assertEqual(result['agreement']['arms']['P']['agree'], 64)
        for a in scores['locations']['prefix_end']: a['values'] = {ident: 0. for ident in m.IDS}
        result = m.grade_values(P, mapping, choices, scores)
        self.assertEqual(result['arms']['P']['credit'], 64)
        self.assertEqual(result['arms']['P']['exact_ties'], 128)
        self.assertFalse(result['primary']['pilot_criterion_met'])

    def test_primary_needs_both_cohorts_and_realized_control(self):
        *_, mapping, scores, choices = fixture()
        for row in mapping['rows']:
            if row['rater'] == 'rater2': choices[row['rater']][row['item_id']] = 'SECOND' if choices[row['rater']][row['item_id']] == 'FIRST' else 'FIRST'
        result = m.grade_values(P, mapping, choices, scores)
        self.assertEqual(result['primary']['P_minus_U_by_cohort_out_of_64'], {'1': 64., '2': 0.})
        self.assertFalse(result['primary']['pilot_criterion_met'])
        for row in mapping['rows']:
            if row['rater'] == 'rater4': choices[row['rater']][row['item_id']] = 'SECOND' if choices[row['rater']][row['item_id']] == 'FIRST' else 'FIRST'
        self.assertTrue(m.grade_values(P, mapping, choices, scores)['primary']['pilot_criterion_met'])
        for r in ('rater1', 'rater3'): choices[r] = {k: 'FIRST' for k in choices[r]}
        self.assertFalse(m.grade_values(P, mapping, choices, scores)['primary']['pilot_criterion_met'])

    def test_agreement_uses_canonical_choice_not_first_second(self):
        *_, mapping, scores, choices = fixture()
        for r in m.RATERS: choices[r] = {k: 'FIRST' for k in choices[r]}
        result = m.grade_values(P, mapping, choices, scores)
        self.assertEqual(result['agreement']['arms']['P']['agree'], 0)
        self.assertEqual(result['arms']['P']['credit'], 64)

    def test_adverse_cells_and_invalid_scores_rosters(self):
        *_, mapping, scores, choices = fixture()
        for row in mapping['rows']:
            if row['arm'] == 'P': choices[row['rater']][row['item_id']] = 'SECOND' if choices[row['rater']][row['item_id']] == 'FIRST' else 'FIRST'
        result = m.grade_values(P, mapping, choices, scores)
        self.assertTrue(all(v['outcome'] == 'harm' for v in result['gains_harm']))
        self.assertEqual(result['primary']['P_minus_U_credit_out_of_128'], -128)
        for value in (True, float('nan'), float('inf')):
            bad = copy.deepcopy(scores); bad['locations']['prefix_end'][0]['values']['T01-L'] = value
            with self.assertRaises(ValueError): m.grade_values(P, mapping, choices, bad)
        bad = copy.deepcopy(mapping); bad['rows'][0] = bad['rows'][1]
        with self.assertRaises(ValueError): m.grade_values(P, bad, choices, scores)

    def test_seal_and_public_gate_never_open_private_map_or_key(self):
        kwargs = self.sealed(); seen = []; real_read = m.read
        def guarded(pin):
            self.assertNotIn('private-map', pin['path']); self.assertNotIn('/evaluation/', pin['path'])
            seen.append(pin['path']); return real_read(pin)
        with patch.object(m, 'helper', return_value=P), patch.object(P, 'PINS', self.fakepins), \
             patch.object(m, 'read', side_effect=guarded), patch.object(m, 'committed') as commits, patch.object(m, 'ancestor'):
            checked = m.verify_response_lock(**kwargs)
        self.assertEqual(checked['choices'], self.choices)
        self.assertIn(5, [len(v.args[0]) for v in commits.call_args_list])
        self.assertIn(4, [len(v.args[0]) for v in commits.call_args_list])
        self.assertFalse((self.root/'evaluation').exists())
        for r in m.RATERS:
            self.assertEqual((self.root/'responses'/f'{r}.json').read_bytes(), Path(self.raw_paths[r]['path']).read_bytes())

    def test_altered_raw_copy_lock_or_missing_commit_blocks_before_key(self):
        kwargs = self.sealed()
        with patch.object(m, 'helper', return_value=P), patch.object(P, 'PINS', self.fakepins), \
             patch.object(m, 'committed', side_effect=ValueError('not committed')), patch.object(m, 'ancestor'), \
             patch.object(m, 'reconstruct', side_effect=AssertionError('must gate first')), \
             patch.dict(m.os.environ, {'JLENS_PATTERN_GRADE_RELEASE': '1'}):
            with self.assertRaisesRegex(ValueError, 'not committed'): m.grade(**kwargs, target=self.root/'grade')
        self.assertEqual({p.name for p in (self.root/'grade').iterdir()}, {'attempt.json', 'failure.json'})
        Path(self.raw_paths['rater1']['path']).write_bytes(b'changed')
        with patch.object(m, 'helper', return_value=P), patch.object(P, 'PINS', self.fakepins), \
             patch.object(m, 'committed'), patch.object(m, 'ancestor'):
            with self.assertRaises(ValueError): m.verify_response_lock(**kwargs)

    def test_bad_response_and_consumed_stage(self):
        self.metadata_fixture()
        raw = P.decode(Path(self.raw_paths['rater1']['path']).read_bytes()); raw['responses'].pop()
        self.raw_paths['rater1'] = self.put('raw/rater1.json', raw)
        with patch.object(m, 'helper', return_value=P), patch.object(P, 'PINS', self.fakepins), \
             patch.object(m, 'committed'), patch.object(m, 'ancestor'), patch.dict(m.os.environ, {'JLENS_PATTERN_SEAL_RELEASE': '1'}):
            with self.assertRaises(ValueError): m.seal(**self.kw, raw_commit='3'*40, raw_paths=self.raw_paths, target=self.root/'responses')
            self.assertEqual({p.name for p in (self.root/'responses').iterdir()}, {'attempt.json', 'failure.json'})
            with self.assertRaises(FileExistsError): m.seal(**self.kw, raw_commit='3'*40, raw_paths=self.raw_paths, target=self.root/'responses')
        with patch.dict(m.os.environ, {'JLENS_PATTERN_SEAL_RELEASE': ''}):
            with self.assertRaises(ValueError): m.Stage(self.root/'other', 'seal', P)

    def test_commit_blob_mismatch_and_history(self):
        path = self.root/'file'; path.write_bytes(b'exact')
        def fakegit(root, *args):
            if args[:1] == ('rev-parse',): return str(self.root).encode()
            if args[:2] == ('cat-file', '-t'): return b'commit\n'
            if args[:2] == ('cat-file', '-s'): return b'5\n'
            if args[:2] == ('cat-file', 'blob'): return b'wrong'
            return b''
        with patch.object(m, 'git', side_effect=fakegit):
            with self.assertRaisesRegex(ValueError, 'blob differs'): m.committed({path: b'exact'}, '1'*40)
        with patch.object(m, 'git', side_effect=subprocess.CalledProcessError(1, 'git')):
            with self.assertRaises(subprocess.CalledProcessError): m.ancestor(self.root, '1'*40, '2'*40)

    def test_strict_json_bounded_read_and_symlink(self):
        for raw in (b'{"a":1,"a":2}', b'{"v":NaN}', b'{"v":1e999}'):
            with self.assertRaises(ValueError): P.decode(raw)
        pin = self.put('one', {}); link = self.root/'link'; link.symlink_to(self.root/'one')
        with self.assertRaises(OSError): m.read({**pin, 'path': str(link)})
        with self.assertRaises(ValueError): m.read({**pin, 'size_bytes': m.CAP+1})

    def evaluation_receipt(self, source_pin, score_pin):
        return {'schema': 'jlens_pattern_calibration_evaluation_receipt_v1', 'status': 'complete',
            'source_sha256': source_pin['sha256'], 'reference_commit': m.REFERENCE_COMMIT,
            'references_sha256': self.fakepins['references'][1], 'calibration_receipt_sha256': self.fakepins['calibration_receipt'][1],
            'preflight_receipt_sha256': self.fakepins['preflight'][1], 'forward_count': 32, 'evaluation_pairs': 16,
            'capture_locations': ['prefix_end'], 'parameters_unchanged': True, 'calibration_rows_loaded': 0,
            'reference_decodes': 0, 'prefix_tokenizations': 0, 'lens_loaded': False, 'tokenizer_loaded': False,
            'pca_refit': False, 'pattern_fit': False, 'runtime': {'python': m.PYTHON, 'packages':
                {'numpy': '1.26.4', 'torch': '2.11.0+cu128', 'transformers': '5.5.0', 'huggingface_hub': '1.8.0'}},
            'input_pins': {str(path): pin for path, pin in self.fakepins.values()},
            'outputs': [{**score_pin, 'path': n} for n in ('features.npz', 'inputs.json', 'scores.json', 'gaps.json')]}

    def test_full_fake_seal_lock_reconstruct_receipt_then_grade(self):
        self.metadata_fixture(); *_, scores, _ = fixture()
        source = self.put('evaluation/evaluate.py', {'fake': 'source'})
        score = self.put('evaluation/scores.json', scores)
        receipt = self.put('evaluation/receipt.json', self.evaluation_receipt(source, score))
        self.release_value['evaluation'] = {'source': source, 'receipt': receipt, 'scores': score}
        released = self.put('release.json', self.release_value); self.kw['release_sha'] = released['sha256']
        gate = m.verify_response_lock; passed = []
        def checked_gate(**kwargs):
            value = gate(**kwargs); passed.append(True); return value
        original = m.read
        def guarded(pin):
            if pin['path'] == score['path']: self.assertTrue(passed, 'key opened before committed public lock')
            return original(pin)
        with patch.object(m, 'helper', return_value=P), patch.object(P, 'PINS', self.fakepins), \
             patch.object(m, 'EVALUATOR_SHA', source['sha256']), patch.object(m, 'committed'), patch.object(m, 'ancestor'), \
             patch.dict(m.os.environ, {'JLENS_PATTERN_SEAL_RELEASE': '1', 'JLENS_PATTERN_GRADE_RELEASE': '1'}):
            lock = m.seal(**self.kw, raw_commit='3'*40, raw_paths=self.raw_paths, target=self.root/'responses')
            with patch.object(m, 'verify_response_lock', side_effect=checked_gate), patch.object(m, 'read', side_effect=guarded):
                pin = m.grade(**self.kw, responses=self.root/'responses', lock_sha=lock['sha256'],
                              lock_commit='4'*40, target=self.root/'graded')
        result = P.decode((self.root/'graded/grades.json').read_bytes())
        self.assertEqual(result['arms']['P']['credit'], 128); self.assertEqual(len(result['items']), 256)
        self.assertEqual(pin['sha256'], m.sha((self.root/'graded/grades.json').read_bytes()))
        self.assertEqual(result['provenance']['raw_commit'], '3'*40)

    def test_receipt_wrong_scope_or_score_pin_rejected_without_key(self):
        self.metadata_fixture()
        source = self.put('evaluation/evaluate.py', {'fake': 'source'})
        score = {'path': str(self.root/'evaluation/scores.json'), 'size_bytes': 1, 'sha256': '0'*64}
        original = self.evaluation_receipt(source, score)
        for kind in ('count', 'source', 'decode', 'input', 'score', 'runtime'):
            value = copy.deepcopy(original)
            if kind == 'count': value['forward_count'] = 64
            elif kind == 'source': value['source_sha256'] = '1'*64
            elif kind == 'decode': value['reference_decodes'] = 8
            elif kind == 'input': value['input_pins'] = {}
            elif kind == 'score': value['outputs'][2]['sha256'] = '1'*64
            else: value['runtime']['python'] = 'old build'
            receipt = self.put('evaluation/receipt.json', value)
            with patch.object(P, 'PINS', self.fakepins), patch.object(m, 'EVALUATOR_SHA', source['sha256']):
                with self.assertRaises(ValueError, msg=kind):
                    m.forward_scope(P, {'evaluation': {'source': source, 'receipt': receipt, 'scores': score}})
        self.assertFalse(Path(score['path']).exists())


if __name__ == '__main__': unittest.main()
