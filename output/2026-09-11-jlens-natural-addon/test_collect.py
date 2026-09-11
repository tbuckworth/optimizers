"""Synthetic collector tests: no real corpus acquisition, tokenizer or model."""
import copy
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request

SPEC = importlib.util.spec_from_file_location('addon_collect', Path(__file__).with_name('collect.py'))
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)
H = m.helper()


def inventory():
    return {'schema': 'jlens_natural_addon_inventory_v1', 'excluded_pageids': [2], 'groups': [
        {'topic': topic, 'category': 'Category:' + topic, 'candidates': [
            {'pageid': i + 1 + 20 * j, 'ns': 0, 'title': f'Fabricated {j} {i}'} for i in range(20)]}
        for j, topic in enumerate(m.TOPICS)]}


def response(pageid, prefix=None, **extra):
    page = {'pageid': pageid, 'ns': 0, 'title': f'Fabricated {pageid}',
            'fullurl': f'https://en.wikipedia.org/wiki/Fabricated_{pageid}',
            'revisions': [{'revid': pageid + 100, 'timestamp': '2000-01-01T00:00:00Z'}],
            'extract': prefix or ' '.join([f'word{pageid}'] + ['example'] * 20), **extra}
    return {'query': {'pages': [page]}}


class CollectorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
    def tearDown(self):
        self.tmp.cleanup()

    def stage(self, name='excerpts'):
        return H.Stage(self.root, name)

    def transport(self, changes=None):
        calls = []
        def run(params, cap):
            calls.append(params['pageids'])
            require_params = {'action': 'query', 'format': 'json', 'formatversion': 2,
                              'maxlag': 5, 'explaintext': 1, 'exintro': 1, 'exchars': 1200,
                              'rvprop': 'ids|timestamp', 'rvlimit': 1, 'inprop': 'url'}
            for key, value in require_params.items():
                self.assertEqual(params[key], value)
            value = response(params['pageids'])
            if changes:
                value = changes(params['pageids'], value)
            return 200, H.encode(value), 'identity'
        return run, calls

    def test_import_inert(self):
        with patch.object(Path, 'read_bytes', side_effect=AssertionError('import read')):
            module = importlib.util.module_from_spec(SPEC)
            SPEC.loader.exec_module(module)

    def test_hash_order_and_pairing_independent_reconstruction(self):
        inv = inventory()
        actual = m.candidate_pairs(inv)
        expected = []
        for group in inv['groups']:
            rows = [r for r in group['candidates'] if r['pageid'] != 2]
            rows.sort(key=lambda r: (m.sha(('20260919|candidate|' + group['topic'] + '|' + str(r['pageid'])).encode()), r['pageid']))
            for left, right in zip(rows[::2], rows[1::2]):
                expected.append((group['topic'], left['pageid'], right['pageid']))
        expected.sort(key=lambda r: (m.sha(('20260919|pair|' + '|'.join(map(str, r))).encode()), m.TOPICS.index(r[0]), r[1], r[2]))
        self.assertEqual([(p['topic'], p['left']['pageid'], p['right']['pageid']) for p in actual['pairs']], expected)
        self.assertEqual(actual['available_count'], 79)
        self.assertEqual(len(actual['odd_tails']), 1)
        self.assertEqual([p['id'] for p in actual['pairs']], [f'K{i:03}' for i in range(1, 40)])
        reversed_inv = copy.deepcopy(inv)
        for group in reversed_inv['groups']:
            group['candidates'].reverse()
        self.assertEqual(m.candidate_pairs(reversed_inv), actual)

    def test_first_ownership_before_exclusion_and_odd_tail(self):
        inv = inventory()
        inv['groups'][1]['candidates'].append(copy.deepcopy(inv['groups'][0]['candidates'][1]))
        manifest = m.candidate_pairs(inv)
        self.assertEqual(manifest['available_count'], 79)
        self.assertEqual(manifest['duplicate_ownership'], [{'pageid': 2, 'owner': 'astronomy', 'other_topic': 'cooking'}])
        ids = [p[s]['pageid'] for p in manifest['pairs'] for s in ('left', 'right')]
        self.assertNotIn(2, ids)
        self.assertEqual(len(ids), len(set(ids)))
        self.assertNotIn(manifest['odd_tails'][0]['candidate']['pageid'], ids)

    def test_bad_inventory(self):
        for changed in ('order', 'bool', 'exclusions', 'outside'):
            inv = inventory()
            if changed == 'order': inv['groups'].reverse()
            elif changed == 'bool': inv['groups'][0]['candidates'][0]['pageid'] = True
            elif changed == 'exclusions': inv['excluded_pageids'] = [2, 2]
            else: inv['excluded_pageids'] = [10000]
            with self.assertRaises(ValueError): m.candidate_pairs(inv)

    def test_whole_panel_and_exact_unicode_words(self):
        manifest = m.candidate_pairs(inventory())
        prefix = '\u2003'.join(['café'] + ['word'] * 15) + '\nextra ignored text'
        first = manifest['pairs'][0]['left']['pageid']
        run, calls = self.transport(lambda pid, value: response(pid, prefix) if pid == first else value)
        stage = self.stage()
        result = m.collect_pairs(H, stage, manifest, set(), run)
        self.assertEqual(result['selected_count'], 32)
        self.assertEqual(result['article_requests'], 32)
        rows = H.decode((stage.path / 'dataset.json').read_bytes())
        pairs = H.decode((stage.path / 'pairs.json').read_bytes())
        self.assertEqual(rows[0]['prefix'], ' '.join(prefix.split()[:16]))
        self.assertEqual([r['id'] for r in rows], [f'N{i:02}-{s}' for i in range(1, 17) for s in ('L', 'R')])
        self.assertEqual(pairs[-1]['id'], 'N16')
        self.assertEqual(len(calls), len(set(calls)))

    def test_invalid_left_still_requests_right_and_never_repairs(self):
        manifest = m.candidate_pairs(inventory())
        bad = manifest['pairs'][0]['left']['pageid']
        run, calls = self.transport(lambda pid, value: response(pid, redirect=True) if pid == bad else value)
        stage = self.stage()
        result = m.collect_pairs(H, stage, manifest, set(), run, target=1)
        self.assertEqual(calls, [p[s]['pageid'] for p in manifest['pairs'][:2] for s in ('left', 'right')])
        self.assertEqual(result['visited_pairs'], 2)
        self.assertEqual(H.decode((stage.path / 'pairs.json').read_bytes())[0]['candidate_id'], 'K002')

    def test_rejected_pair_prefixes_are_not_reserved(self):
        manifest = m.candidate_pairs(inventory())
        first, second = manifest['pairs'][:2]
        same = ' '.join(['shared'] * 16)
        changes = {first['left']['pageid']: response(first['left']['pageid'], same),
                   first['right']['pageid']: response(first['right']['pageid'], redirect=True),
                   second['left']['pageid']: response(second['left']['pageid'], same)}
        run, calls = self.transport(lambda pid, value: changes.get(pid, value))
        stage = self.stage()
        result = m.collect_pairs(H, stage, manifest, set(), run, target=1)
        self.assertEqual(result['article_requests'], 4)
        self.assertEqual(H.decode((stage.path / 'dataset.json').read_bytes())[0]['prefix'], same)

    def test_within_pair_old_and_accepted_duplicates(self):
        manifest = m.candidate_pairs(inventory())
        same = ' '.join(['duplicate'] * 16)
        for case in ('within', 'old', 'accepted'):
            stage = self.stage(case)
            changes = {}
            pair = manifest['pairs'][0 if case != 'accepted' else 1]
            changes[pair['left']['pageid']] = response(pair['left']['pageid'], same)
            if case == 'within': changes[pair['right']['pageid']] = response(pair['right']['pageid'], same)
            if case == 'accepted':
                ident = manifest['pairs'][0]['left']['pageid']
                changes[ident] = response(ident, same)
            run, calls = self.transport(lambda pid, value: changes.get(pid, value))
            result = m.collect_pairs(H, stage, manifest, {same} if case == 'old' else set(), run,
                                     target=2 if case == 'accepted' else 1)
            self.assertEqual(result['visited_pairs'], 3 if case == 'accepted' else 2)

    def test_integrity_failure_stops_before_right(self):
        for i, marker in enumerate(({}, {'redirect': True}, {'ns': 1}, {'pageprops': {'disambiguation': ''}})):
            run, calls = self.transport(lambda pid, value: response(9999, **marker))
            with self.assertRaisesRegex(ValueError, 'different page ID'):
                m.collect_pairs(H, self.stage(f'wrong{i}'), m.candidate_pairs(inventory()), set(), run)
            self.assertEqual(len(calls), 1)

    def test_pair_cap_and_no_partial_dataset(self):
        run, calls = self.transport(lambda pid, value: response(pid, extract='too short'))
        stage = self.stage()
        with self.assertRaisesRegex(ValueError, 'fixed cap'):
            m.collect_pairs(H, stage, m.candidate_pairs(inventory()), set(), run, target=1, max_pairs=3)
        self.assertEqual(len(calls), 6)
        self.assertFalse((stage.path / 'dataset.json').exists())

    def test_no_redirect_handler(self):
        request = Request(m.API)
        with self.assertRaises(HTTPError) as caught:
            m.NoRedirect().redirect_request(request, io.BytesIO(b'redirect'), 302, 'Found', {}, 'https://example.org')
        self.assertEqual(caught.exception.code, 302)

    def test_strict_json_and_http_failures_preserved(self):
        cases = [(200, b'{"x":1,"x":2}', 'identity'), (200, b'[NaN]', 'identity'),
                 (200, b'[1e999]', 'identity'), (200, b'{"warnings":{"x":1}}', 'identity'),
                 (200, b'{"continue":{}}', 'identity'), (302, b'moved', 'identity'),
                 (200, b'{}', 'gzip')]
        for i, case in enumerate(cases):
            stage = self.stage('bad' + str(i))
            with self.assertRaises(ValueError):
                m.request(H, stage, 'first', {}, lambda params, cap: case)
            self.assertTrue((stage.path / 'first.response.json').exists())
            self.assertTrue((stage.path / 'first.request.json').exists())

    def test_response_cap_and_output_reserve(self):
        stage = self.stage()
        with patch.object(H, 'MAX_RESPONSE', 16):
            with self.assertRaisesRegex(ValueError, 'response byte cap'):
                m.request(H, stage, 'over', {}, lambda params, cap: (200, b'x' * (cap + 1), 'identity'))
        metadata = H.decode((stage.path / 'over.request.json').read_bytes())
        self.assertTrue(metadata['body_truncated_at_cap'])
        self.assertEqual((stage.path / 'over.response.json').stat().st_size, 16)
        with patch.object(H, 'MAX_OUTPUT', stage.used + H.RESERVE):
            with self.assertRaisesRegex(ValueError, 'output byte cap'):
                stage.write('normal.json', {'x': 1})
            stage.fail(ValueError('synthetic reserved failure'))
        self.assertTrue((stage.path / 'failure.json').exists())

    def fixture_inputs(self):
        old = self.root / 'old'
        (old / 'catalogue').mkdir(parents=True)
        (old / 'excerpts-resumed').mkdir()
        helper_raw = (m.OLD / 'collect_text.py').read_bytes()
        (old / 'collect_text.py').write_bytes(helper_raw)
        inv = inventory()
        catalogue = {'groups': inv['groups']}
        catraw = H.encode(catalogue)
        catpath = old / 'catalogue/catalogue.json'
        catpath.write_bytes(catraw)
        data = H.encode([{'prefix': f'old {i}'} for i in range(24)])
        olddata = old / 'excerpts-resumed/dataset.json'
        olddata.write_bytes(data)
        inv['catalogue'] = {'path': str(catpath), 'sha256': m.sha(catraw)}
        inv['sources'] = [m.snapshot(H, p)[1] for p in (catpath, olddata)]
        invpath = self.root / 'inventory.json'
        invraw = H.encode(inv)
        invpath.write_bytes(invraw)
        protocol = b'Fabricated protocol.'
        (self.root / 'protocol.md').write_bytes(protocol)
        return old, invpath, m.sha(invraw), m.sha(catraw), m.sha(data), m.sha(protocol)

    def test_full_selection_excerpts_binding_and_exclusive_stages(self):
        old, invpath, invsha, catsha, datasha, protosha = self.fixture_inputs()
        with patch.multiple(m, OLD=old, CATALOGUE_SHA=catsha, OLD_DATA_SHA=datasha, PROTOCOL_SHA=protosha):
            m.selection(invpath, invsha, self.root)
            receipt = self.root / 'selection/receipt.json'
            run, calls = self.transport()
            m.excerpts(invpath, invsha, m.sha(receipt.read_bytes()), self.root, run)
            self.assertEqual(len(calls), 32)
            done = H.decode((self.root / 'excerpts/receipt.json').read_bytes())
            self.assertEqual(done['schema'], 'jlens_natural_addon_excerpts_v1')
            self.assertTrue(any(r['path'] == str(Path(m.__file__)) for r in done['inputs']))
            with self.assertRaises(FileExistsError): m.selection(invpath, invsha, self.root)
            with self.assertRaises(FileExistsError): m.excerpts(invpath, invsha, 'bad', self.root, run)
            self.assertEqual(len(calls), 32)

    def test_source_inventory_selection_drift(self):
        old, invpath, invsha, catsha, datasha, protosha = self.fixture_inputs()
        with patch.multiple(m, OLD=old, CATALOGUE_SHA=catsha, OLD_DATA_SHA=datasha, PROTOCOL_SHA=protosha):
            m.selection(invpath, invsha, self.root)
            receiptsha = m.sha((self.root / 'selection/receipt.json').read_bytes())
            (self.root / 'selection/manifest.json').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                m.excerpts(invpath, invsha, receiptsha, self.root,
                           lambda *_: self.fail('network must not be called'))
            self.assertTrue((self.root / 'excerpts/failure.json').exists())
        target = self.root / 'drift.json'
        target.write_bytes(b'first')
        record = m.snapshot(H, target)[1]
        target.write_bytes(b'changed')
        with self.assertRaises(ValueError): m.recheck(H, [record])

    def test_transport_exception_metadata(self):
        stage = self.stage()
        def fail(*args): raise TimeoutError('fabricated timeout')
        with self.assertRaises(TimeoutError): m.request(H, stage, 'first', {}, fail)
        self.assertEqual(H.decode((stage.path / 'first.request.json').read_bytes())['transport_error_type'], 'TimeoutError')

    def test_revision_only_continuation_preserved_not_followed(self):
        manifest = m.candidate_pairs(inventory())
        token = {'rvcontinue': '20000101000000|1234', 'continue': '||extracts|info|pageprops'}
        run, calls = self.transport(lambda pid, value: dict(value, **{'continue': token}))
        stage = self.stage()
        m.collect_pairs(H, stage, manifest, set(), run, target=1)
        self.assertEqual(len(calls), 2)
        raw = H.decode((stage.path / 'K001-left.response.json').read_bytes())
        self.assertEqual(raw['continue'], token)
        for i, bad in enumerate(({}, {'excontinue': 1, 'continue': '-||'},
                                  {'rvcontinue': '', 'continue': 'x'},
                                  {'rvcontinue': 'x', 'continue': 'x', 'other': 'y'},
                                  {'rvcontinue': 2, 'continue': 'x'})):
            with self.assertRaisesRegex(ValueError, 'non-revision'):
                m.request(H, self.stage(f'cont{i}'), 'first', {},
                          lambda *_: (200, H.encode({'continue': bad}), 'identity'))


if __name__ == '__main__':
    unittest.main()
