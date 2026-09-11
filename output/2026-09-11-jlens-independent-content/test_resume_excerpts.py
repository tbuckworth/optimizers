"""Fabricated-only tests of saved-response continuation; no network or real corpus."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('resume_fixture', HERE/'resume_excerpts.py')
r = importlib.util.module_from_spec(spec); spec.loader.exec_module(r)
spec = importlib.util.spec_from_file_location('collector_tests_fixture', HERE/'test_collect_text.py')
t = importlib.util.module_from_spec(spec); spec.loader.exec_module(t)
c = r.c


class ResumeTests(unittest.TestCase):
    def test_revision_continuation_only(self):
        value = {'continue': {'rvcontinue': '123|456', 'continue': '||extracts|info|pageprops'},
                 'query': {'pages': []}}
        self.assertEqual(value, r.validate_response(200, c.encode(value), 'identity'))
        for token in [{'cmcontinue': '123', 'continue': '-||'}, {'continue': 'x'},
                      {'rvcontinue': 'x', 'continue': 'y', 'excontinue': 1}]:
            with self.assertRaises(ValueError): r.validate_response(200, c.encode({'continue': token}), 'identity')
        for status, raw in [(429, b'{}'), (200, b'{"warnings":{"x":1}}'), (200, b'{"a":1,"a":2}')]:
            with self.assertRaises(ValueError): r.validate_response(status, raw, 'identity')

    def test_saved_response_reused_and_exact_panel(self):
        with tempfile.TemporaryDirectory(prefix='jlens-resume-fixture-') as tmp:
            root = Path(tmp)
            (root/'protocol.md').write_bytes(b'Fabricated protocol only.\n')
            api = t.FakeAPI()
            with patch.object(c, 'PROTOCOL_SHA', c.sha((root/'protocol.md').read_bytes())), \
                 patch.object(c, 'urlopen', side_effect=AssertionError('network forbidden')):
                c.catalogue(root, api)
                catalogue_sha = c.sha((root/'catalogue/receipt.json').read_bytes())
                groups, _, _ = c.load_catalogue(root)
                first = groups[0]['candidates'][0]['pageid']
                value = t.article(first)
                value['continue'] = {'rvcontinue': '123|456', 'continue': '||extracts|info|pageprops'}
                api.overrides[first] = value
                with self.assertRaisesRegex(ValueError, 'unexpected continuation'): c.excerpts(root, api)
                before = {p.name: p.read_bytes() for p in (root/'excerpts').iterdir()}
                pins = {name: c.sha(raw) for name, raw in before.items()}
                calls_before = len(api.calls)
                with patch.object(r, 'CATALOGUE_SHA', catalogue_sha), patch.object(r, 'SAVED', pins):
                    r.resume(root, api)
                    with self.assertRaises(FileExistsError): r.resume(root, api)
                self.assertEqual(23, len(api.calls)-calls_before)
                self.assertNotIn(first, [p['pageids'] for p in api.calls[calls_before:]])
                self.assertEqual(before, {p.name: p.read_bytes() for p in (root/'excerpts').iterdir()})
                rows = c.decode((root/'excerpts-resumed/dataset.json').read_bytes())
                self.assertEqual(24, len(rows))
                self.assertEqual(' '.join(t.article(first)['query']['pages'][0]['extract'].split()[:16]), rows[0]['prefix'])
                receipt = c.decode((root/'excerpts-resumed/receipt.json').read_bytes())
                self.assertEqual((24, 23, 1), (receipt['candidate_requests'], receipt['new_network_requests'], receipt['reused_response_count']))
                for output in receipt['outputs']:
                    self.assertEqual(output['sha256'], c.sha((root/'excerpts-resumed'/output['path']).read_bytes()))

    def test_saved_request_binding_before_network(self):
        with tempfile.TemporaryDirectory() as tmp:
            stage = c.Stage(Path(tmp), 'stage')
            with self.assertRaises(ValueError):
                r.request(stage, 'x', r.params_for(1), lambda *args: self.fail('network called'),
                          ({'params': r.params_for(2)}, b'{}'))


if __name__ == '__main__':
    unittest.main()
