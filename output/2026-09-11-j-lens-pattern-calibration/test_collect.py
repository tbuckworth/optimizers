"""Fabricated fixtures only: never access the real manifest, corpus or transport."""
import copy
import importlib.util
import io
from pathlib import Path
import signal
import tempfile
import unittest
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError
from urllib.request import Request

SPEC = importlib.util.spec_from_file_location('pattern_collect', Path(__file__).with_name('collect.py'))
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)
H = m.helper()  # Reads only the reviewed helper Python source, never a prior stage.


def manifest(n=86):
    return {'schema': 'jlens_pattern_calibration_candidate_manifest_v1',
        'seed': '20260921', 'protocol_sha256': m.PROTOCOL_SHA, 'inventory_sha256': m.INVENTORY_SHA,
        'source_sha256': m.PLANNER_SHA, 'targets': {'calibration_pairs': 32, 'evaluation_pairs': 16},
        'maximum_requested_pairs': 80, 'maximum_article_requests': 160,
        'candidate_role_counts': {'calibration': 58, 'evaluation': 28},
        'eligibility_or_measurement_performed': False,
        'candidates': [{'candidate_id': f'K{i+1:03}', 'topic': 'programming',
            'left_pageid': 2*i+1, 'right_pageid': 2*i+2,
            'role': 'evaluation' if i % 3 == 2 else 'calibration'} for i in range(n)],
        'odd_tails': [{'topic': 'programming', 'pageid': 173}]}


def response(ident, text=None, **extra):
    return {'query': {'pages': [{'pageid': ident, 'ns': 0, 'title': f'Fabricated {ident}',
        'fullurl': f'https://en.wikipedia.org/wiki/Fabricated_{ident}',
        'revisions': [{'revid': 1000+ident, 'timestamp': '2000-01-01T00:00:00Z'}],
        'extract': text or ' '.join([f'word{ident}']+['example']*20), **extra}]}}


class Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
    def tearDown(self):
        self.tmp.cleanup()
    def stage(self):
        return m.Stage(self.root)
    def read(self, stage, name):
        return H.decode((stage.path/name).read_bytes())
    def transport(self, changes=None):
        calls = []
        def run(params, cap):
            calls.append(params['pageids'])
            self.assertEqual({k: params[k] for k in ('action', 'formatversion', 'maxlag', 'exchars', 'rvlimit')},
                             {'action': 'query', 'formatversion': 2, 'maxlag': 5, 'exchars': 1200, 'rvlimit': 1})
            value = response(params['pageids'])
            if changes: value = changes(params['pageids'], value)
            return 200, H.encode(value), 'identity'
        return run, calls

    def test_import_inert(self):
        with patch.object(Path, 'read_bytes', side_effect=AssertionError('import read')), \
             patch.object(Path, 'open', side_effect=AssertionError('import open')):
            module = importlib.util.module_from_spec(SPEC)
            SPEC.loader.exec_module(module)

    def test_fixed_manifest_roles_ids_and_exclusions(self):
        inv = {'groups': [{'topic': t, 'remaining_pageids': list(range(1, 174)) if t == 'programming' else []}
                          for t in m.TOPICS], 'excluded_pageids': [999]}
        m.validate_manifest(manifest(), inv)
        for change in ('role', 'ID', 'duplicate', 'tail', 'target', 'seed', 'old', 'owner'):
            v, x = manifest(), copy.deepcopy(inv)
            if change == 'role': v['candidates'][0]['role'] = 'evaluation'
            elif change == 'ID': v['candidates'][0]['candidate_id'] = 'K002'
            elif change == 'duplicate': v['candidates'][1]['left_pageid'] = 1
            elif change == 'tail': v['odd_tails'][0]['pageid'] = 1
            elif change == 'target': v['targets']['calibration_pairs'] = 31
            elif change == 'seed': v['seed'] = 'other'
            elif change == 'old': x['excluded_pageids'].append(1)
            else: v['candidates'][0]['topic'] = 'astronomy'
            with self.assertRaises(ValueError, msg=change): m.validate_manifest(v, x)

    def test_all_roles_roster_96_and_every_candidate_disposition(self):
        stage = self.stage(); run, calls = self.transport()
        stats = m.collect(H, stage, manifest(), set(), run)
        self.assertEqual(stats['accepted_pairs'], {'calibration': 32, 'evaluation': 16})
        self.assertEqual(stats['selected_count'], 96)
        self.assertEqual(len(calls), 96)
        self.assertEqual(len(calls), len(set(calls)))
        for role, letter, count in [('calibration', 'C', 32), ('evaluation', 'T', 16)]:
            data = self.read(stage, role+'-dataset.json')
            self.assertEqual([r['id'] for r in data], [f'{letter}{i:02}-{s}' for i in range(1, count+1) for s in ('L', 'R')])
            self.assertTrue(all(len(r['prefix'].split()) == 16 for r in data))
            pairs = self.read(stage, role+'-pairs.json')
            self.assertTrue(all(p['role'] == role for p in pairs))
        dispositions = self.read(stage, 'candidate-dispositions.json')
        self.assertEqual(len(dispositions), 86)
        self.assertEqual(sum(d['status'] == 'capacity_skip_no_requests' for d in dispositions), 38)
        stage.audit()

    def test_rejection_never_reassigns_role_and_requests_both(self):
        stage = self.stage()
        run, calls = self.transport(lambda pid, v: response(pid, redirect=True) if pid == 1 else v)
        m.collect(H, stage, manifest(), set(), run, targets={'calibration': 1, 'evaluation': 1})
        self.assertEqual(calls, [1, 2, 3, 4, 5, 6])
        self.assertEqual(self.read(stage, 'calibration-pairs.json')[0]['candidate_id'], 'K002')
        self.assertEqual(self.read(stage, 'evaluation-pairs.json')[0]['candidate_id'], 'K003')
        self.assertEqual(self.read(stage, 'K001.decision.json')['status'], 'rejected')

    def test_rejected_prefix_not_reserved_and_exact_unicode_words(self):
        stage = self.stage()
        text = '\u2003'.join(['café']+['word']*15)+'\nnot selected'
        def changes(pid, value):
            if pid in (1, 3): return response(pid, text)
            return response(pid, redirect=True) if pid == 2 else value
        run, _ = self.transport(changes)
        m.collect(H, stage, manifest(), set(), run, targets={'calibration': 1, 'evaluation': 1})
        self.assertEqual(self.read(stage, 'calibration-dataset.json')[0]['prefix'], ' '.join(text.split()[:16]))

    def test_duplicate_old_within_and_cross_role(self):
        for case in ('old', 'within', 'cross'):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as folder:
                stage = m.Stage(Path(folder)); same = ' '.join(['duplicate']*16)
                ids = {1, 2} if case == 'within' else {1, 5} if case == 'cross' else {1}
                run, calls = self.transport(lambda pid, v: response(pid, same) if pid in ids else v)
                m.collect(H, stage, manifest(), {same} if case == 'old' else set(), run,
                          targets={'calibration': 1, 'evaluation': 1})
                name = 'K003' if case == 'cross' else 'K001'
                self.assertEqual(self.read(stage, name+'.decision.json')['status'], 'rejected')
                self.assertEqual(len(calls), 6)

    def test_wrong_ID_integrity_before_technical_skip_stops_right(self):
        for extra in ({}, {'redirect': True}, {'missing': True}, {'ns': 1}, {'pageprops': {'disambiguation': ''}}):
            with tempfile.TemporaryDirectory() as folder:
                stage = m.Stage(Path(folder))
                run, calls = self.transport(lambda pid, v: response(999, **extra))
                with self.assertRaisesRegex(ValueError, 'different returned'):
                    m.collect(H, stage, manifest(), set(), run)
                self.assertEqual(calls, [1])

    def test_technical_short_namespace_revision_title_and_pageprops(self):
        for extra in ({'extract': 'short'}, {'ns': 1}, {'revisions': []}, {'title': ''}, {'pageprops': {'disambiguation': ''}}):
            with tempfile.TemporaryDirectory() as folder:
                stage = m.Stage(Path(folder))
                run, calls = self.transport(lambda pid, v: response(pid, **extra) if pid == 1 else v)
                m.collect(H, stage, manifest(), set(), run, targets={'calibration': 1, 'evaluation': 1})
                self.assertEqual(calls[:2], [1, 2])
                self.assertEqual(self.read(stage, 'K001.decision.json')['status'], 'rejected')

    def test_pair_request_and_time_caps_no_partial_dataset(self):
        stage = self.stage(); run, calls = self.transport(lambda pid, v: response(pid, extract='short'))
        with self.assertRaisesRegex(ValueError, 'pair cap'):
            m.collect(H, stage, manifest(), set(), run, max_pairs=3)
        self.assertEqual(len(calls), 6)
        self.assertFalse((stage.path/'calibration-dataset.json').exists())
        stage.requests = 160
        with self.assertRaisesRegex(ValueError, 'request cap'): m.request(H, stage, 'cap', 1, run)
        stage.requests = 0; stage.started -= 3601
        with self.assertRaisesRegex(ValueError, 'time cap'): m.request(H, stage, 'time', 1, run)
        self.assertEqual(len(calls), 6)

    def test_exhausted_role_never_promoted_from_other(self):
        stage = self.stage(); run, _ = self.transport(lambda pid, v: response(pid, extract='short') if pid in (5, 6) else v)
        with self.assertRaisesRegex(ValueError, 'manifest exhausted'):
            m.collect(H, stage, manifest(3), set(), run, targets={'calibration': 1, 'evaluation': 1})
        self.assertFalse((stage.path/'evaluation-dataset.json').exists())

    def test_strict_response_failures_and_revision_continuation(self):
        cases = [b'{"x":1,"x":2}', b'[NaN]', b'[1e999]', b'{"warnings":{}}',
                 b'{"error":{}}', b'{"query":{"redirects":[]}}',
                 b'{"continue":{"rvcontinue":"x","continue":"x","extra":1}}']
        for raw in cases:
            with tempfile.TemporaryDirectory() as folder:
                stage = m.Stage(Path(folder))
                with self.assertRaises(ValueError): m.request(H, stage, 'bad', 1, lambda *_: (200, raw, 'identity'))
                self.assertEqual((stage.path/'bad.response.json').read_bytes(), raw)
                self.assertTrue((stage.path/'bad.request.json').is_file())
        stage = self.stage()
        value = response(1); value['continue'] = {'rvcontinue': 'old|123', 'continue': '||extracts'}
        calls = []
        def transport(*args): calls.append(1); return 200, H.encode(value), 'identity'
        self.assertEqual(m.request(H, stage, 'good', 1, transport)[0], value)
        self.assertEqual(calls, [1])

    def test_http_encoding_errors_and_transport_no_retry(self):
        for status, encoding in ((302, 'identity'), (500, 'identity'), (200, 'gzip')):
            with tempfile.TemporaryDirectory() as folder:
                stage = m.Stage(Path(folder))
                with self.assertRaises(ValueError):
                    m.request(H, stage, 'bad', 1, lambda *_: (status, b'{}', encoding))
        stage = self.stage(); calls = []
        def timeout(*args): calls.append(1); raise TimeoutError('fabricated')
        with self.assertRaises(TimeoutError): m.request(H, stage, 'timeout', 1, timeout)
        self.assertEqual(calls, [1])
        self.assertEqual(self.read(stage, 'timeout.request.json')['transport_error_type'], 'TimeoutError')

    def test_cap_retained_prefix_and_failure_reserve_accounting(self):
        stage = self.stage()
        with patch.object(m, 'MAX_RESPONSE', 16):
            with self.assertRaisesRegex(ValueError, 'response cap'):
                m.request(H, stage, 'big', 1, lambda p, cap: (200, b'x'*(cap+1), 'identity'))
        self.assertEqual((stage.path/'big.response.json').read_bytes(), b'x'*16)
        self.assertTrue(self.read(stage, 'big.request.json')['body_truncated_at_cap'])
        with patch.object(m, 'MAX_OUTPUT', stage.used+m.RESERVE):
            with self.assertRaisesRegex(ValueError, 'output byte cap'): stage.write('normal.json', {})
            stage.fail(ValueError('fabricated'))
            self.assertLessEqual(sum(p.stat().st_size for p in stage.path.iterdir()), m.MAX_OUTPUT)
        stage.audit()
        receipt = self.read(stage, 'failure.json')
        self.assertEqual(receipt['bytes_before_failure'], sum(r['size_bytes'] for r in receipt['artifacts_before_failure']))

    def test_no_redirect_and_mocked_20_second_deadline_restoration(self):
        with self.assertRaises(HTTPError):
            m.NoRedirect().redirect_request(Request(m.API), io.BytesIO(b'redirect'), 302, 'Moved', {}, 'https://example.org')
        prior = object(); handlers = []
        def install(signum, handler): handlers.append(handler)
        def alarm_open(*args, **kwargs): handlers[0](signal.SIGALRM, None)
        opener = MagicMock(); opener.open.side_effect = alarm_open
        with patch.object(m.signal, 'getsignal', return_value=prior), \
             patch.object(m.signal, 'getitimer', return_value=(0.0, 0.0)), \
             patch.object(m.signal, 'signal', side_effect=install), \
             patch.object(m.signal, 'setitimer') as timer, patch.object(m, 'build_opener', return_value=opener):
            with self.assertRaisesRegex(TimeoutError, '20-second'): m.fetch({}, 100)
            self.assertEqual(timer.call_args_list[0].args, (signal.ITIMER_REAL, 20))
            self.assertEqual(timer.call_args_list[-1].args, (signal.ITIMER_REAL, 0.0, 0.0))
            self.assertIs(handlers[-1], prior)
            self.assertEqual(opener.open.call_count, 1)
        with patch.object(m.signal, 'getitimer', return_value=(5.0, 1.0)), \
             patch.object(m.signal, 'setitimer') as timer, patch.object(m, 'build_opener') as build:
            with self.assertRaisesRegex(ValueError, 'existing process alarm'): m.fetch({}, 100)
            timer.assert_not_called(); build.assert_not_called()

    def test_successful_transport_restores_alarm_and_retains_exact_bytes(self):
        response_mock = MagicMock()
        response_mock.__enter__.return_value = response_mock
        response_mock.status = 200; response_mock.read.return_value = b'raw response'
        response_mock.headers.get.return_value = 'identity'
        opener = MagicMock(); opener.open.return_value = response_mock
        with patch.object(m.signal, 'getsignal', return_value=signal.SIG_DFL), \
             patch.object(m.signal, 'getitimer', return_value=(0.0, 0.0)), \
             patch.object(m.signal, 'signal') as install, patch.object(m.signal, 'setitimer') as timer, \
             patch.object(m, 'build_opener', return_value=opener):
            self.assertEqual(m.fetch({'pageids': 1}, 123), (200, b'raw response', 'identity'))
            response_mock.read.assert_called_once_with(124)
            self.assertEqual(opener.open.call_args.kwargs, {'timeout': 20})
            self.assertEqual(install.call_args.args, (signal.SIGALRM, signal.SIG_DFL))
            self.assertEqual(timer.call_args.args, (signal.ITIMER_REAL, 0.0, 0.0))

    def test_old_prefix_source_joins_are_reconstructed(self):
        data_path = self.root/'synthetic-dataset.json'
        rows = [{'id': str(i), 'prefix': f'fabricated prefix {i}'} for i in range(296)]
        data = H.encode(rows)
        records = [{'source_index': 0, 'row_index': i, 'id': r['id'], 'prefix': r['prefix']} for i, r in enumerate(rows)]
        old = {'schema': 'jlens_pattern_calibration_old_prefix_inventory_v1',
            'sources': [{'path': str(data_path), 'sha256': m.sha(data), 'size_bytes': len(data),
                         'container': 'list', 'text_field': 'prefix', 'row_count': 296}],
            'records': records, 'unique_prefixes': sorted(r['prefix'] for r in rows),
            'row_count': 296, 'unique_prefix_count': 296}
        inventory = {'groups': [{'topic': t, 'remaining_pageids': list(range(1, 174)) if t == 'programming' else []}
                               for t in m.TOPICS], 'excluded_pageids': [999]}
        receipt = {'status': 'METADATA_SELECTION_COMPLETE', 'network_calls': 0,
            'manifest_sha256': m.MANIFEST_SHA, 'inventory_sha256': m.INVENTORY_SHA,
            'protocol_sha256': m.PROTOCOL_SHA, 'source_sha256': m.PLANNER_SHA}
        values = {str(m.STUDY/'selection/manifest.json'): H.encode(manifest()),
            str(m.STUDY/'selection/receipt.json'): H.encode(receipt),
            str(m.OUT/'inventory.json'): H.encode(inventory),
            str(m.OUT/'old-prefixes.json'): H.encode(old), str(data_path): data}
        def fake_snapshot(path, expected=None):
            raw = values.get(str(path), b'fabricated source')
            return raw, {'path': str(path), 'sha256': m.sha(raw), 'size_bytes': len(raw)}
        with patch.object(m, 'snapshot', side_effect=fake_snapshot):
            self.assertEqual(m.load_inputs(H)[1], {r['prefix'] for r in rows})
            old['records'][0]['prefix'] = 'invented replacement'
            values[str(m.OUT/'old-prefixes.json')] = H.encode(old)
            with self.assertRaisesRegex(ValueError, 'exact old prefix'): m.load_inputs(H)
        data_path.write_bytes(data)
        m.snapshot(data_path, m.sha(data))
        with self.assertRaisesRegex(ValueError, 'hash mismatch'): m.snapshot(data_path, 'wrong')

    def test_failure_receipt_includes_partial_untracked_file(self):
        stage = self.stage()
        (stage.path/'partial.json').write_bytes(b'{')
        stage.fail(OSError('fabricated interrupted write'))
        failure = self.read(stage, 'failure.json')
        self.assertIn({'path': 'partial.json', 'size_bytes': 1, 'sha256': m.sha(b'{')}, failure['artifacts_before_failure'])

    def test_complete_guard_and_drift_failure_without_real_inputs(self):
        run, calls = self.transport()
        source = Path(m.__file__); pin = m.snapshot(source)[1]
        with patch.object(m, 'helper', return_value=H), \
             patch.object(m, 'load_inputs', return_value=(manifest(), set(), [pin])):
            m.run(self.root, run)
            with self.assertRaises(FileExistsError): m.run(self.root, run)
        self.assertEqual(len(calls), 96)
        receipt = H.decode((self.root/'excerpts/receipt.json').read_bytes())
        self.assertEqual(receipt['selected_count'], 96)
        self.assertEqual(receipt['bytes_before_receipt'], sum(r['size_bytes'] for r in receipt['outputs']))
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder); (target/'excerpts').symlink_to(target/'missing')
            with patch.object(m, 'helper', side_effect=AssertionError('must guard first')):
                with self.assertRaises(FileExistsError): m.run(target, run)
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(m, 'helper', return_value=H), patch.object(m, 'load_inputs', side_effect=ValueError('pin mismatch')):
                with self.assertRaisesRegex(ValueError, 'pin mismatch'): m.run(Path(folder), run)
            self.assertTrue((Path(folder)/'excerpts/failure.json').is_file())
        self.assertEqual(len(calls), 96)


if __name__ == '__main__':
    unittest.main()
