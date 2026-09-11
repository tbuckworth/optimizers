"""Fabricated-only recovery tests. No real tokenizer, model, corpus or archive."""

import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

SOURCE = Path(__file__).with_name('forward_runtime_amendment.py')
spec = importlib.util.spec_from_file_location('runtime_amendment_fixture', SOURCE)
w = importlib.util.module_from_spec(spec)
spec.loader.exec_module(w)


def runtimes():
    old = {'python': w.OLD_PYTHON, 'interpreter': '/usr/bin/python3', 'packages': dict(w.NUMERICAL)}
    new = {**old, 'python': w.NEW_PYTHON}
    host = {'python': w.NEW_PYTHON, 'interpreter': '/usr/bin/python3',
            'resolved_interpreter': '/usr/bin/python3.12', 'package_versions': dict(w.PACKAGES),
            'binary_sha256s': dict(w.BINARIES)}
    return old, new, host


class PureTests(unittest.TestCase):
    def test_import_inert(self):
        with (mock.patch.object(Path, 'read_bytes', side_effect=AssertionError('read')),
              mock.patch.object(w.subprocess, 'check_output', side_effect=AssertionError('process'))):
            spec = importlib.util.spec_from_file_location('inert_recovery_again', SOURCE)
            spec.loader.exec_module(importlib.util.module_from_spec(spec))

    def test_exact_transition_and_each_invalid_field(self):
        old, new, host = runtimes()
        result = w.validate_transition(old, new, host)
        self.assertEqual(result['old_preflight_source_sha256'], w.BASE_SHA)
        self.assertEqual(result['preflight_python'], w.OLD_PYTHON)
        self.assertEqual(result['new_python'], w.NEW_PYTHON)
        mutations = [(0, 'python', w.NEW_PYTHON), (1, 'python', w.OLD_PYTHON),
                     (2, 'python', '3.12.3'), (0, 'interpreter', '/wrong'),
                     (2, 'resolved_interpreter', '/wrong'), (1, 'packages', {}),
                     (2, 'package_versions', {}), (2, 'binary_sha256s', {})]
        for i, key, value in mutations:
            args = copy.deepcopy([old, new, host]); args[i][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): w.validate_transition(*args)

    def test_host_identity_queries_only_pinned_packages_and_binaries(self):
        output = ''.join(f'{p}\t{v}\n' for p, v in w.PACKAGES.items())
        with (mock.patch.object(w.subprocess, 'check_output', return_value=output) as command,
              mock.patch.object(w, 'sha', side_effect=lambda p: w.BINARIES[str(p)]) as hashes):
            host = w.host_identity()
        self.assertEqual(host['package_versions'], w.PACKAGES)
        self.assertEqual(host['binary_sha256s'], w.BINARIES)
        self.assertEqual(command.call_args.args[0][-len(w.PACKAGES):], [p+':amd64' for p in w.PACKAGES])
        self.assertEqual(hashes.call_count, 2)
        with mock.patch.object(w.subprocess, 'check_output', return_value='libc6\tx\nlibc6\tx\n'):
            with self.assertRaises(ValueError): w.host_identity()

    def test_pinned_source_bytes_not_old_entrypoint_or_cache(self):
        with tempfile.TemporaryDirectory(prefix='jlens-recovery-helper-fixture-') as directory:
            root = Path(directory); source = root/'forward.py'
            source.write_text('from pathlib import Path\nOUT = Path(__file__).parent\nVALUE = 17\n')
            with (mock.patch.object(w, 'OUT', root), mock.patch.object(w, 'BASE', source),
                  mock.patch.object(w, 'BASE_SHA', w.sha(source))):
                self.assertEqual(w.load_base().VALUE, 17)
                source.write_text('raise AssertionError("must not execute changed source")\n')
                with self.assertRaisesRegex(ValueError, 'producer changed'): w.load_base()

    def test_metadata_failure_files_immutable_and_extra_outputs_rejected(self):
        with tempfile.TemporaryDirectory(prefix='jlens-recovery-metadata-fixture-') as directory:
            root = Path(directory); (root/'forwards').mkdir()
            files = [root/'forwards'/name for name in ('attempt.json', 'failure.json')]
            for path in files: path.write_text('{}\n')
            pins = {path: w.sha(path) for path in files}
            with mock.patch.object(w, 'OUT', root):
                w.verify_metadata(pins)
                (root/'forwards'/'unexpected.json').write_text('{}\n')
                with self.assertRaisesRegex(ValueError, 'unexpected output'): w.verify_metadata(pins)
                files[0].write_text('[]\n')
                with self.assertRaisesRegex(ValueError, 'metadata pin'): w.verify_metadata(pins)


class StageTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix='jlens-recovery-stage-fixture-')
        self.addCleanup(temp.cleanup); self.root = Path(temp.name)
        self.old, self.new, self.host = runtimes()
        self.rows = [{'id': f'N{i:02}-{side}', 'topic': 'fabricated', 'prefix': 'fixture'}
                     for i in range(1, 17) for side in ('L', 'R')]
        self.pairs = [{'id': f'N{i:02}', 'left': f'N{i:02}-L', 'right': f'N{i:02}-R'} for i in range(1, 17)]
        self.records = [{**r, 'input_ids': [1, 2], 'attention_mask': [1, 1],
                         'captured_positions': {'prefix_end': 1}} for r in self.rows]
        self.preflight = {'runtime': self.old}
        self.expected = {self.root/'fabricated-input.json': 'a'*64}
        self.h = np.zeros((32, 1, 1024), dtype=np.float32)
        self.u = np.eye(1024, 4, dtype=np.float32)
        self.mean = np.zeros(1024, dtype=np.float64)
        self.events = []
        def capture(*args):
            self.events.append('capture'); return self.h
        def export(h, u, mean, rows, pairs):
            scores = (h.astype(np.float64)-mean) @ u.astype(np.float64)
            return scores, scores[::2]-scores[1::2], {'schema': 'fabricated'}
        self.base = SimpleNamespace(OUT=self.root, OLD=self.root/'old-fixture', LOCATIONS=['prefix_end'],
            WEIGHT_SHA='b'*64, UPSTREAM_REV='c'*40, pins=mock.Mock(return_value=self.expected),
            verify=mock.Mock(), load_panel=mock.Mock(return_value=(self.rows, self.pairs)),
            frozen_tokens=mock.Mock(return_value=(self.records, self.preflight)),
            runtime=mock.Mock(return_value=self.new), validate_directions=mock.Mock(),
            load_model=mock.Mock(return_value=('fake-hf', 'fake-model')),
            parameter_state=mock.Mock(return_value=[('fake-param', 0)]),
            capture=mock.Mock(side_effect=capture), score_export=mock.Mock(side_effect=export),
            preflight=mock.Mock(side_effect=AssertionError('old preflight entrypoint')),
            forwards=mock.Mock(side_effect=AssertionError('old forward entrypoint')))
        archive = mock.MagicMock()
        archive.__enter__.return_value = {'u32': self.u, 'source_mean64': self.mean}
        cuda = SimpleNamespace(get_device_name=lambda _: 'fabricated', max_memory_allocated=lambda: 0,
                               max_memory_reserved=lambda: 0)
        self.patches = [mock.patch.object(w, 'OUT', self.root),
            mock.patch.object(w, 'verify_metadata'), mock.patch.object(w, 'load_base', return_value=self.base),
            mock.patch.object(w, 'host_identity', side_effect=lambda: copy.deepcopy(self.host)),
            mock.patch.object(np, 'load', return_value=archive),
            mock.patch.dict(w.os.environ, {'JLENS_NATURAL_ADDON_RUNTIME_AMENDMENT_RELEASE': '1'}),
            mock.patch.dict(sys.modules, {'torch': SimpleNamespace(__version__=w.NUMERICAL['torch'], cuda=cuda),
                'transformers': SimpleNamespace(__version__=w.NUMERICAL['transformers']),
                'huggingface_hub': SimpleNamespace(__version__=w.NUMERICAL['huggingface_hub'])}),
            mock.patch.object(np, 'savez_compressed', wraps=np.savez_compressed)]
        self.started = [p.start() for p in self.patches]
        for p in reversed(self.patches): self.addCleanup(p.stop)

    def run_stage(self):
        w.run('dataset-fixture-sha', 'pairs-fixture-sha', w.PREFLIGHT_SHA)

    def test_success_exact_old_token_location_provenance_and_shapes(self):
        original = copy.deepcopy(self.preflight)
        self.run_stage()
        receipt = json.loads((self.root/w.STAGE/'receipt.json').read_text())
        self.assertEqual(receipt['schema'], 'jlens_natural_addon_forwards_receipt_v1')
        self.assertEqual(receipt['source_sha256'], w.sha(SOURCE))
        self.assertEqual(receipt['base_producer_sha256'], w.BASE_SHA)
        self.assertEqual(receipt['runtime_amendment_sha256'], w.AMENDMENT_SHA)
        self.assertEqual(receipt['failed_admission_sha256s'], w.FAILED)
        self.assertEqual(receipt['preflight_runtime'], original['runtime'])
        self.assertEqual(receipt['runtime'], self.new)
        self.assertEqual(receipt['runtime_transition'], w.validate_transition(self.old, self.new, self.host))
        self.assertEqual(receipt['input_pins'], {str(p): s for p, s in self.expected.items()})
        self.assertEqual(receipt['forward_count'], 32)
        self.assertEqual(receipt['capture_locations'], ['prefix_end'])
        self.assertFalse(receipt['tokenizer_loaded'])
        self.assertEqual(receipt['reference_decodes'], 0)
        self.assertEqual(self.preflight, original)
        self.assertEqual(self.base.frozen_tokens.call_count, 2)
        for call in self.base.frozen_tokens.call_args_list:
            self.assertEqual(call.args, (self.root, w.PREFLIGHT_SHA, self.expected, self.rows))
        self.base.capture.assert_called_once_with('fake-model', self.records, 'cuda')
        self.assertEqual(self.events, ['capture'])
        self.base.preflight.assert_not_called(); self.base.forwards.assert_not_called()
        self.started[4].assert_called_once_with(self.base.OLD/'preparation/directions.npz', allow_pickle=False)
        for output in receipt['outputs']:
            self.assertEqual(output['sha256'], w.sha(self.root/w.STAGE/output['path']))
        self.assertEqual(len(receipt['outputs']), 4)
        saved = self.started[-1].call_args.kwargs
        self.assertEqual(saved['activation_11'].shape, (32, 1, 1024))
        self.assertEqual(saved['scores64'].shape, (32, 1, 4))
        self.assertEqual(saved['gaps64'].shape, (16, 1, 4))
        self.assertEqual(saved['locations'].tolist(), ['prefix_end'])

    def test_existing_stage_guard_precedes_source_and_metadata_reads(self):
        (self.root/w.STAGE).mkdir()
        with mock.patch.object(w, 'sha', side_effect=AssertionError('read before guard')):
            with self.assertRaises(FileExistsError): self.run_stage()
        self.base.load_model.assert_not_called()

    def test_no_admission_creates_nothing(self):
        with mock.patch.dict(w.os.environ, {'JLENS_NATURAL_ADDON_RUNTIME_AMENDMENT_RELEASE': ''}):
            with self.assertRaises(ValueError): self.run_stage()
        self.assertFalse((self.root/w.STAGE).exists())

    def test_bad_build_binary_or_versions_stop_before_array_or_model(self):
        for n, (field, value) in enumerate((('python', w.OLD_PYTHON), ('binary_sha256s', {}), ('package_versions', {}))):
            root = self.root/f'case{n}'; root.mkdir()
            host = copy.deepcopy(self.host); host[field] = value
            with mock.patch.object(w, 'OUT', root), mock.patch.object(w, 'host_identity', return_value=host):
                with self.assertRaises(ValueError): self.run_stage()
            self.assertTrue((root/w.STAGE/'failure.json').exists())
            self.assertFalse((root/w.STAGE/'receipt.json').exists())
        self.started[4].assert_not_called(); self.base.load_model.assert_not_called()

    def test_failure_pin_and_original_preflight_pin_fail_closed(self):
        with self.assertRaisesRegex(ValueError, 'original preflight'):
            w.run('a', 'b', '0'*64)
        self.started[4].assert_not_called(); self.base.load_model.assert_not_called()
        with self.assertRaises(FileExistsError): self.run_stage()

    def test_runtime_changes_after_capture_preserve_failure_not_success(self):
        bad = copy.deepcopy(self.host); bad['binary_sha256s'] = {}
        with mock.patch.object(w, 'host_identity', side_effect=[self.host, self.host, bad]):
            with self.assertRaisesRegex(ValueError, 'binary hashes'): self.run_stage()
        self.base.capture.assert_called_once()
        self.assertTrue((self.root/w.STAGE/'failure.json').exists())
        self.assertFalse((self.root/w.STAGE/'receipt.json').exists())


if __name__ == '__main__':
    unittest.main()
