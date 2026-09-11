"""Fabricated-only collector tests: no Wikipedia, model, tokenizer or corpus access."""

import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location('collector_fixture', Path(__file__).with_name('collect_text.py'))
c = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(c)


def article(pageid):
    return {'query': {'pages': [{'pageid': pageid, 'ns': 0, 'title': f'Fabricated {pageid}',
        'fullurl': f'https://en.wikipedia.org/wiki/Fabricated_{pageid}',
        'revisions': [{'revid': pageid+10000, 'timestamp': '2026-01-01T00:00:00Z'}],
        'extract': '\t  '+f'Unique{pageid} '+'\n'.join(f'word{i}' for i in range(1, 25))}]}}


class FakeAPI:
    def __init__(self):
        self.calls = []
        self.members = {topic: [{'pageid': 100*j+i, 'ns': 0, 'title': f'Page {100*j+i}'} for i in range(1, 26)]
                        for j, (topic, _) in enumerate(c.ROOTS, 1)}
        self.overrides = {}
        self.catalogue_extra = {}

    def __call__(self, params, cap):
        self.calls.append(copy.deepcopy(params))
        assert params['maxlag'] == 5 and params['formatversion'] == 2
        if 'list' in params:
            topic = next(t for t, category in c.ROOTS if category == params['cmtitle'])
            value = {'query': {'categorymembers': self.members[topic]}, **self.catalogue_extra}
        else:
            assert params['exchars'] == 1200 and params['explaintext'] == params['exintro'] == 1
            assert 'redirects' not in params and 'exsentences' not in params
            value = self.overrides.get(params['pageids'], article(params['pageids']))
        return 200, c.encode(value), 'identity'


class CollectorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='jlens-fake-collector-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root/'protocol.md').write_bytes(b'Fabricated protocol only.\n')
        self.pin = patch.object(c, 'PROTOCOL_SHA', c.sha((self.root/'protocol.md').read_bytes()))
        self.pin.start(); self.addCleanup(self.pin.stop)
        self.no_network = patch.object(c, 'urlopen', side_effect=AssertionError('network forbidden in fixture'))
        self.no_network.start(); self.addCleanup(self.no_network.stop)
        self.api = FakeAPI()

    def ordered(self, topic):
        return c.ordered_members({'query': {'categorymembers': self.api.members[topic]}}, topic)

    def test_import_inert(self):
        with patch.object(Path, 'read_bytes', side_effect=AssertionError('import read')):
            spec = importlib.util.spec_from_file_location('second_collector', Path(c.__file__))
            mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)

    def test_complete_order_exact_prefix_roster_and_provenance(self):
        c.catalogue(self.root, self.api)
        self.assertEqual(4, len(self.api.calls))
        c.excerpts(self.root, self.api)
        self.assertEqual(28, len(self.api.calls))
        rows = c.decode((self.root/'excerpts/dataset.json').read_bytes())
        self.assertEqual([f'{t}-wiki-{i}' for t, _ in c.ROOTS for i in range(6)], [r['id'] for r in rows])
        expected_pages = [p['pageid'] for topic, _ in c.ROOTS for p in self.ordered(topic)[:6]]
        self.assertEqual(expected_pages, [p['pageids'] for p in self.api.calls[4:]])
        for row, pageid in zip(rows, expected_pages):
            self.assertEqual(' '.join(article(pageid)['query']['pages'][0]['extract'].split()[:16]), row['prefix'])
            self.assertEqual(16, len(row['prefix'].split()))
        pairs = c.decode((self.root/'excerpts/pairs.json').read_bytes())
        self.assertEqual([{'id': f'P{i+1:02}', 'left': rows[2*i]['id'], 'right': rows[2*i+1]['id']} for i in range(12)], pairs)
        attrs = c.decode((self.root/'excerpts/attribution.json').read_bytes())['articles']
        self.assertEqual(24, len(attrs))
        for row in attrs:
            self.assertIn('action=history', row['history_url'])
            self.assertEqual('https://creativecommons.org/licenses/by-sa/4.0/', row['license_url'])
        for folder in ['catalogue', 'excerpts']:
            receipt = c.decode((self.root/folder/'receipt.json').read_bytes())
            for item in receipt['outputs']:
                raw = (self.root/folder/item['path']).read_bytes()
                self.assertEqual(item['sha256'], c.sha(raw))
        request = c.decode((self.root/'excerpts/astronomy-01.request.json').read_bytes())
        self.assertEqual(c.USER_AGENT, request['user_agent'])
        self.assertFalse(request['body_truncated_at_cap'])
        self.assertEqual(c.encode(article(expected_pages[0])), (self.root/'excerpts/astronomy-01.response.json').read_bytes())

    def test_eligibility_all_technical_skips_and_no_semantic_filter(self):
        for marker in ['missing', 'invalid', 'redirect']:
            value = article(1); value['query']['pages'][0][marker] = True
            self.assertEqual(marker, c.eligible(value, 1, set(), set())[1])
        mutations = [('ns', 1, 'non-main namespace'), ('pageprops', {'disambiguation': ''}, 'disambiguation'),
                     ('title', '', 'invalid title'), ('fullurl', 'https://other.invalid/x', 'invalid article URL'),
                     ('revisions', [], 'invalid revision metadata'), ('extract', 'too short', 'short/missing extract')]
        for key, val, reason in mutations:
            value = article(1); value['query']['pages'][0][key] = val
            self.assertEqual(reason, c.eligible(value, 1, set(), set())[1])
        value = article(1); value['query']['pages'][0]['extract'] = ' odd!? '*300
        selected, reason = c.eligible(value, 1, set(), set())
        self.assertIsNone(reason)  # >1200 chars and odd content are not exclusions.
        self.assertEqual(' '.join(['odd!?']*16), selected['prefix'])

    def test_global_dedup_first_root_and_prefix_skips(self):
        # Same full catalogue for every root: root order, not an article topic claim, owns overlaps.
        common = list(self.api.members['astronomy'])
        self.api.members = {t: list(common) for t, _ in c.ROOTS}
        c.catalogue(self.root, self.api)
        # A duplicate prefix on the second astronomy candidate is retained as a skip.
        ordered = self.ordered('astronomy')
        first, second = [p['pageid'] for p in ordered[:2]]
        duplicate = article(second)
        duplicate['query']['pages'][0]['extract'] = article(first)['query']['pages'][0]['extract']
        self.api.overrides[second] = duplicate
        # Build expected acceptance from fake input metadata only, using the same prescribed root order.
        chosen, prefixes, counts = set(), set(), []
        for topic, _ in c.ROOTS:
            count = 0
            for candidate in self.ordered(topic)[:20]:
                pageid = candidate['pageid']
                selected, _ = c.eligible(self.api.overrides.get(pageid, article(pageid)), pageid, chosen, prefixes)
                if selected:
                    chosen.add(pageid); prefixes.add(selected['prefix']); count += 1
                    if count == 6: break
            counts.append(count)
        if counts == [6]*4:
            c.excerpts(self.root, self.api)
            attrs = c.decode((self.root/'excerpts/attribution.json').read_bytes())['articles']
            self.assertEqual(24, len({a['pageid'] for a in attrs}))
        else:
            with self.assertRaises(ValueError): c.excerpts(self.root, self.api)
        decisions = [c.decode(p.read_bytes()) for p in (self.root/'excerpts').glob('*.decision.json')]
        self.assertIn('previously selected prefix', [d['skip_reason'] for d in decisions])
        self.assertIn('previously selected page ID', [d['skip_reason'] for d in decisions])

    def test_twenty_candidate_limit_and_failed_stage_no_repeat(self):
        for p in self.api.members['astronomy']:
            value = article(p['pageid']); value['query']['pages'][0]['extract'] = ''
            self.api.overrides[p['pageid']] = value
        c.catalogue(self.root, self.api)
        with self.assertRaises(ValueError): c.excerpts(self.root, self.api)
        self.assertEqual(24, len(self.api.calls))
        self.assertTrue((self.root/'excerpts/failure.json').exists())
        before = len(self.api.calls)
        with self.assertRaises(FileExistsError): c.excerpts(self.root, self.api)
        self.assertEqual(before, len(self.api.calls))

    def test_catalogue_continuation_short_and_api_failures(self):
        self.api.catalogue_extra = {'continue': {'cmcontinue': 'fabricated'}}
        with self.assertRaises(ValueError): c.catalogue(self.root, self.api)
        self.assertEqual(1, len(self.api.calls))
        self.assertTrue((self.root/'catalogue/astronomy.response.json').exists())
        with self.assertRaises(FileExistsError): c.catalogue(self.root, self.api)
        for value in [{'query': {'categorymembers': []}}, {'query': {'categorymembers': self.api.members['astronomy'][:5]}}]:
            with self.assertRaises(ValueError): c.ordered_members(value, 'astronomy')
        for raw in [b'{"error":{"code":"maxlag"}}', b'{"warnings":{"x":"review"}}', b'{"a":1,"a":2}', b'{"n":NaN}', b'{"n":1e999}']:
            with tempfile.TemporaryDirectory() as tmp:
                stage = c.Stage(Path(tmp), 'stage')
                with self.assertRaises(ValueError): stage.request('x', {}, lambda p, cap: (200, raw, 'identity'))

    def test_http_transport_caps_and_guards(self):
        for status, body in [(429, b'Rate limited'), (200, b'x'*(c.MAX_RESPONSE+1))]:
            with tempfile.TemporaryDirectory() as tmp:
                stage = c.Stage(Path(tmp), 'stage')
                with self.assertRaises(ValueError): stage.request('x', {}, lambda p, cap: (status, body, 'identity'))
                self.assertTrue((stage.path/'x.response.json').exists())
                self.assertLessEqual((stage.path/'x.response.json').stat().st_size, c.MAX_RESPONSE)
        stage = c.Stage(self.root, 'guard')
        stage.used = c.MAX_OUTPUT-c.RESERVE
        with self.assertRaises(ValueError): stage.write('too-large.json', {})
        with patch.object(c, 'protocol', side_effect=AssertionError('read before guard')):
            (self.root/'catalogue').symlink_to(self.root/'missing')
            with self.assertRaises(FileExistsError): c.catalogue(self.root, self.api)
        self.assertEqual([], self.api.calls)
        with self.assertRaises(ValueError): c.decode(b'{"value":Infinity}')

    def test_catalogue_tamper_stops_before_candidate_request(self):
        c.catalogue(self.root, self.api)
        (self.root/'catalogue/catalogue.json').write_bytes(b'{}')
        with self.assertRaises(ValueError): c.excerpts(self.root, self.api)
        self.assertEqual(4, len(self.api.calls))

    def test_http_wrapper_single_get_identity_and_transport_failure(self):
        class Response:
            status, headers = 200, {}
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def read(self, n):
                self.read_count = n
                return b'{}'
        response = Response()
        with patch.object(c, 'urlopen', return_value=response) as opened:
            self.assertEqual((200, b'{}', 'identity'), c.fetch({'maxlag': 5}, 100))
            opened.assert_called_once()
            request = opened.call_args.args[0]
            self.assertEqual('GET', request.get_method())
            self.assertEqual(c.USER_AGENT, request.get_header('User-agent'))
            self.assertEqual('identity', request.get_header('Accept-encoding'))
            self.assertEqual(20, opened.call_args.kwargs['timeout'])
            self.assertEqual(101, response.read_count)
        stage = c.Stage(self.root, 'transport-failure')
        def broken(params, cap): raise TimeoutError('fabricated timeout')
        with self.assertRaises(TimeoutError): stage.request('x', {}, broken)
        record = c.decode((stage.path/'x.request.json').read_bytes())
        self.assertEqual('TimeoutError', record['transport_error_type'])
        self.assertFalse((stage.path/'x.response.json').exists())


if __name__ == '__main__':
    unittest.main()
