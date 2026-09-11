"""Fabricated metadata only; never reads real corpus or request records."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location('inventory', Path(__file__).with_name('inventory.py'))
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


def fixture(root, mutate=None):
    values = {}
    def encoded(value):
        return value if type(value) is bytes else json.dumps(value).encode()
    def pin(name):
        data = encoded(values[name])
        return {'path': Path(name).name, 'sha256': m.sha(data), 'size_bytes': len(data)}
    def request(response, pageid=None, topic=None):
        params = {'pageids': pageid, 'prop': 'extracts|revisions|info|pageprops'} if pageid else {
            'list': 'categorymembers', 'cmtitle': m.CATEGORIES[m.TOPICS.index(topic)],
            'cmnamespace': 0, 'cmlimit': 500}
        return {'endpoint': 'https://en.wikipedia.org/w/api.php', 'params': params,
                'status': 200, 'body_truncated_at_cap': False,
                'started_utc': '2000-01-01T00:00:00Z', 'completed_utc': '2000-01-01T00:00:01Z',
                'response': pin(response)}
    groups = []
    for n, (topic, category) in enumerate(zip(m.TOPICS, m.CATEGORIES)):
        members = [{'pageid': 11 + n * 10 + i, 'ns': 0, 'title': f'Fabricated {n} {i}'} for i in range(3)]
        response = f'catalogue/{topic}.response.json'
        values[response] = {'batchcomplete': True, 'query': {'categorymembers': members}}
        values[f'catalogue/{topic}.request.json'] = request(response, topic=topic)
        groups.append({'topic': topic, 'category': category, 'response': pin(response),
                       'candidates': sorted(members, key=lambda r: (
                           m.sha(f"20260914|{topic}|{r['pageid']}".encode()), r['pageid']))})
    values['catalogue/catalogue.json'] = {'schema': 'jlens_independent_catalogue_data_v1', 'groups': groups}
    original = 'excerpts/astronomy-01'
    resumed = 'excerpts-resumed/astronomy-01'
    values[original + '.response.json'] = b'{"fabricated_article_bytes":true}'
    values[original + '.request.json'] = request(original + '.response.json', pageid=11)
    values[resumed + '.response.json'] = values[original + '.response.json']
    values[resumed + '.request.json'] = dict(values[original + '.request.json'],
        new_network_request=False, reused_from='../excerpts/astronomy-01.response.json')
    values[resumed + '.decision.json'] = {'topic': 'astronomy', 'candidate_rank': 1, 'pageid': 11,
        'selected': True, 'skip_reason': None, 'response': pin(resumed + '.response.json')}
    skipped = 'excerpts-resumed/astronomy-02'
    values[skipped + '.response.json'] = b'{"fabricated_redirect":true}'
    values[skipped + '.request.json'] = request(skipped + '.response.json', pageid=12)
    values[skipped + '.decision.json'] = {'topic': 'astronomy', 'candidate_rank': 2, 'pageid': 12,
        'selected': False, 'skip_reason': 'redirect', 'response': pin(skipped + '.response.json')}
    values['excerpts-resumed/attribution.json'] = {'articles': [{'id': 'fake-0', 'pageid': 11,
        'prefix': 'Entirely fabricated text.', 'url': 'https://en.wikipedia.org/wiki/Fabricated'}]}
    values['excerpts-resumed/dataset.json'] = [{'id': 'fake-0', 'prefix': 'Entirely fabricated text.'}]
    values['excerpts-resumed/continuation.json'] = {'preserved_attempt': {
        Path(name).name: pin(name)['sha256'] for name in values if name.startswith('excerpts/')}}
    values['attribution-review.md'] = b'Fabricated attribution record only.'
    if mutate:
        mutate(values)
    for directory in ('catalogue', 'excerpts-resumed'):
        schema = 'jlens_independent_catalogue_v1' if directory == 'catalogue' else 'jlens_independent_excerpts_v1'
        values[f'{directory}/receipt.json'] = {'schema': schema, 'status': 'complete', 'outputs': [
            pin(name) for name in sorted(values) if Path(name).parent == Path(directory)]}
    base = root / m.STUDY
    for name, value in values.items():
        target = base / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(encoded(value))
    pins = {name: pin(name)['sha256'] for name in m.PINS}
    return pins


class InventoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
    def tearDown(self):
        self.temp.cleanup()

    def test_complete_metadata_excludes_skips_not_catalogue_and_counts_copy_once(self):
        result = m.inspect(self.root, fixture(self.root))
        self.assertEqual(result['excluded_pageids'], [11, 12])
        self.assertEqual(result['article_request_record_count'], 3)
        self.assertEqual(result['distinct_article_network_events'], 2)
        self.assertEqual(result['technical_skipped_pageids'], [12])
        self.assertEqual(result['available_unique_count'], 10)
        self.assertFalse(result['candidate_pairing_performed'])
        self.assertEqual(sum('same_network_event_as' in r for r in result['request_records']), 1)
        self.assertTrue(all(Path(s['path']).is_absolute() for s in result['sources']))

    def test_ownership_deduplicates_before_exclusion_in_original_group_order(self):
        groups = [{'topic': 'a', 'candidates': [{'pageid': 1}, {'pageid': 2}]},
                  {'topic': 'b', 'candidates': [{'pageid': 1}, {'pageid': 3}]}]
        owners, counts = m.ownership(groups, {1})
        self.assertEqual(owners, [{'pageid': 1, 'topic': 'a'}, {'pageid': 2, 'topic': 'a'},
                                  {'pageid': 3, 'topic': 'b'}])
        self.assertEqual([c['available_owned'] for c in counts], [1, 1])

    def test_unexpected_scoped_request_fails(self):
        pins = fixture(self.root)
        extra = self.root / 'output/2099-j-lens-other/extra.request.json'
        extra.parent.mkdir()
        extra.write_text('{}')
        with self.assertRaisesRegex(ValueError, 'scoped request'):
            m.inspect(self.root, pins)

    def test_pin_and_final_recheck(self):
        pins = fixture(self.root)
        target = self.root / m.STUDY / 'attribution-review.md'
        target.write_text('drift')
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            m.inspect(self.root, pins)
        snap = m.Snapshot(self.root)
        snap.read(target.relative_to(self.root), m.sha(b'drift'))
        target.write_text('later drift')
        with self.assertRaisesRegex(ValueError, 'after read'):
            snap.finish()

    def test_copy_cannot_count_as_new_request(self):
        def mutate(v):
            v['excerpts-resumed/astronomy-01.request.json']['new_network_request'] = True
        with self.assertRaisesRegex(ValueError, 'unique original'):
            m.inspect(self.root, fixture(self.root, mutate))

    def test_wrong_decision_and_dataset_join(self):
        for name, field, value in [('excerpts-resumed/astronomy-02.decision.json', 'pageid', 13),
                                   ('excerpts-resumed/dataset.json', None, None)]:
            with self.subTest(name=name):
                def mutate(v):
                    if field:
                        v[name][field] = value
                    else:
                        v[name][0]['prefix'] = 'Wrong text.'
                with self.assertRaises(ValueError):
                    m.inspect(self.root, fixture(self.root, mutate))

    def test_catalogue_continuation_and_raw_mismatch(self):
        for edit in ('continue', 'order'):
            def mutate(v):
                if edit == 'continue':
                    v['catalogue/astronomy.response.json']['continue'] = {'cmcontinue': 'x'}
                else:
                    v['catalogue/catalogue.json']['groups'][0]['candidates'].reverse()
            with self.assertRaises(ValueError):
                m.inspect(self.root, fixture(self.root, mutate))

    def test_strict_json(self):
        for data in (b'{"x":1,"x":2}', b'[NaN]', b'[Infinity]', b'[1e999]'):
            with self.assertRaises(ValueError):
                m.decode(data)

    def test_exclusive_output_guard_before_any_input_read(self):
        output = self.root / 'inventory.json'
        output.write_text('keep')
        with patch.object(m, 'inspect', side_effect=AssertionError('must not read')):
            with self.assertRaisesRegex(ValueError, 'already exists'):
                m.build(self.root, output)
        output.unlink()
        output.symlink_to(self.root / 'absent')
        with self.assertRaisesRegex(ValueError, 'already exists'):
            m.build(self.root, output)

    def test_failed_build_preserves_empty_attempt_and_success_is_exclusive(self):
        output = self.root / 'inventory.json'
        with patch.object(m, 'inspect', side_effect=ValueError('fabricated failure')):
            with self.assertRaises(ValueError):
                m.build(self.root, output)
        self.assertEqual(output.read_bytes(), b'')
        success = self.root / 'success.json'
        digest = m.build(self.root, success, fixture(self.root))
        self.assertEqual(digest, m.sha(success.read_bytes()))

    def test_import_inert(self):
        with patch.object(Path, 'read_bytes', side_effect=AssertionError('import read')):
            module = importlib.util.module_from_spec(SPEC)
            SPEC.loader.exec_module(module)


if __name__ == '__main__':
    unittest.main()
