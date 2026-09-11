"""Tiny invented-vector fixtures only; no scientific archive or Torch import."""
import copy
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

SOURCE = Path(__file__).resolve().parents[1]/'scripts/analyze_spectral_observer_signal.py'
spec = importlib.util.spec_from_file_location('observer_signal_fixtures', SOURCE)
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)


def invented_inputs():
    """Six scalar joins around three-coordinate invented inputs, not a model."""
    cases, readouts, histories = {}, {}, {}
    for seed_number, seed in enumerate(s.SEEDS):
        for cell in s.CELLS:
            rows = {}
            for action_number, action in enumerate(s.ACTIONS):
                ident = f's{seed}-zero' if action == 'zero' else f's{seed}-{cell}-{action}'
                readouts[ident] = {'physical_id': ident,
                    'oracle_signed_utilities': {q: {'total': seed_number+action_number-.5,
                        'adaptive': seed_number+action_number-.6, 'decay': .1} for q in s.PROBES},
                    'movement': {'total': {'norm': .25}, 'adaptive': {'norm': .2}, 'decay': {'norm': .05}}}
                record = {'physical_id': ident, 'parent_sha256': 'a'*64,
                          'state': {'path': ident+'.pt'}, 'logits': {'path': ident+'.npz'},
                          # Deliberately wrong producer-copy J: must not be joined.
                          'oracle_signed_utilities': {'rare8': {'total': 999999.}}}
                value = seed_number*.1-action_number*.01
                improvements = {'rare_ce': value, 'majority_macro_ce': -value/2,
                    'train_true_rare_ce': value+.001, 'train_true_majority_macro_ce': -value/2+.001,
                    'rare_accuracy': 0., 'majority_macro_accuracy': .002,
                    'train_true_rare_accuracy': 0., 'train_true_majority_macro_accuracy': -.001,
                    'train_assigned_ce': .03, 'train_wrong_ce': None if cell == 'clean' else .04}
                rows[action] = {'readout': record, 'improvements': improvements,
                                'shared_across_label_cells': action == 'zero'}
            cases[seed, cell] = {'seed': seed, 'cell': cell, 'actions': rows,
                                'mean_admission': {'g_star_sha256': s.vector_hash32(np.array([1, 2, -1], np.float32))}}
            for schedule in s.SCHEDULES:
                histories[seed, cell, schedule] = {'post_inclusion': {'geometry':
                    {q: {'native_retention': .8, 'numerical_span_retention': .8} for q in s.PROBES}}}
    return cases, readouts, histories


class AlignmentTests(unittest.TestCase):
    def test_signed_products_norms_and_undefined_zero_cosine(self):
        row = s.alignment(np.array([1., 2.]), np.array([-3., 1.]))
        self.assertEqual(row['dot'], -1.)
        self.assertEqual(row['sign_status'], 'negative')
        self.assertAlmostEqual(row['cosine'], -1/math.sqrt(50))
        zero = s.alignment(np.ones(2), np.zeros(2))
        self.assertEqual(zero['dot'], 0.)
        self.assertIsNone(zero['cosine'])
        self.assertTrue(zero['zero_norm'])
        self.assertEqual(zero['sign_status'], 'roundoff_unresolved')

    def test_compensated_reference_and_unresolved_cancellation(self):
        row = s.alignment(np.ones(3), np.array([1e16, 1., -1e16]))
        self.assertEqual(row['dot'], 1.)
        self.assertEqual(row['absolute_product_sum'], 2e16)
        self.assertEqual(row['raw_sign'], 'positive')
        self.assertEqual(row['sign_status'], 'roundoff_unresolved')
        self.assertEqual(row['operational_bound'], 128*s.EPS64*2e16+1e-12)

    def test_difference_is_cast_to_fp64_before_subtraction(self):
        left = np.array([33554432.], np.float32)
        right = np.array([1.], np.float32)
        result = s.difference_alignment(np.ones(1), left, right)
        self.assertEqual(result['dot'], 33554431.)
        self.assertEqual(result['paired_identity']['discrepancy'], 0.)
        # Stronger cancellation example: direct result 1 versus parent-dot difference 2.
        row = s.difference_alignment(np.ones(2), np.array([1e16, 2.]), np.array([1e16, 1.]))
        self.assertEqual(row['dot'], 1.)
        self.assertEqual(row['paired_identity']['paired_dot_difference'], 2.)
        self.assertEqual(row['paired_identity']['discrepancy'], 1.)
        self.assertGreater(row['paired_identity']['operational_bound'], 1.)
        self.assertEqual(row['sign_status'], 'roundoff_unresolved')

    def test_nonfinite_and_reduction_failure_preserve_evidence(self):
        with self.assertRaises(s.AnalysisError):
            s.alignment(np.array([float('nan')]), np.ones(1))
        with patch.object(s.np, 'dot', return_value=7.):
            with self.assertRaises(s.AnalysisError) as caught:
                s.alignment(np.ones(2), np.zeros(2))
        self.assertEqual(caught.exception.evidence['numpy_dot'], 7.)
        self.assertEqual(caught.exception.evidence['dot'], 0.)
        with self.assertRaises(s.AnalysisError):
            s.alignment(np.ones(3), np.ones(2))

    def test_three_seed_summary_and_nulls(self):
        row = s.sample_summary([-1., 0., 1.])
        self.assertEqual(row['mean'], 0.)
        self.assertEqual(row['sample_sd'], 1.)
        self.assertAlmostEqual(row['sample_se'], 1/math.sqrt(3))
        self.assertEqual([row[x] for x in ('positive_count', 'negative_count', 'zero_count')], [1, 1, 1])
        self.assertIsNone(s.sample_summary([None]*3)['mean'])
        partial = s.sample_summary([1., None, 2.])
        self.assertIsNone(partial['positive_count'])
        self.assertEqual(partial['present_count'], 2)
        with self.assertRaises(s.AnalysisError):
            s.sample_summary([1., 2.])


class RosterAndJoinTests(unittest.TestCase):
    def case(self, seed=s.SEEDS[0], cell='clean'):
        cases, readouts, histories = invented_inputs()
        probes = {'majority': np.array([1., 0., 1.]), 'rare8': np.array([0., 2., 1.])}
        common = np.array([1., 2., -1.])
        native = {'interleaved': np.array([1., 1., 0.]), 'grouped': np.array([0., 2., 0.])}
        return s.analyze_case(seed, cell, probes, common, native, cases[seed, cell], readouts, histories)

    def test_control_identities_and_independent_J_not_producer_copy(self):
        case = self.case()
        for probe in s.PROBES:
            row = case['probes'][probe]
            self.assertEqual(row['actions']['raw']['F']['dot'], row['B']['dot'])
            self.assertEqual(row['actions']['raw']['K']['dot'], 0.)
            self.assertEqual(row['actions']['zero']['F']['dot'], 0.)
            self.assertEqual(row['actions']['zero']['K']['dot'], -row['B']['dot'])
            self.assertEqual(row['actions']['native_interleaved']['accepted_join']['J']['total'], -.5)
            self.assertEqual(row['actions']['zero']['accepted_join']['E_over_zero'], 0.)
        rare = case['probes']['rare8']['actions']['native_grouped']['accepted_join']
        common = case['probes']['majority']['actions']['native_grouped']['accepted_join']
        self.assertEqual(rare['U_heldout_ce'], -.01)
        self.assertEqual(common['U_heldout_ce'], .005)
        self.assertIsNone(rare['U_train_wrong_ce'])

    def test_all_six_pairs_both_probes_and_orientation(self):
        case = self.case()
        self.assertEqual(len(case['contrasts']), 12)
        self.assertEqual(sum(x['primary'] for x in case['contrasts']), 2)
        row = next(x for x in case['contrasts'] if x['primary'] and x['probe'] == 'rare8')
        self.assertEqual(row['D_filter']['dot'], 2.)
        self.assertEqual(row['D_Adam']['total'], 1.)
        self.assertEqual(row['D_U_heldout_ce'], -.01)

    def test_complete_summary_shape_counts_and_seed_order(self):
        cases = [self.case(seed, cell) for seed in s.SEEDS for cell in s.CELLS]
        labels = [{'seed': seed, 'probe': probe, 'alignment': s.alignment(np.ones(1), np.array([float(i)]))}
                  for i, seed in enumerate(s.SEEDS) for probe in s.PROBES]
        summary = s.summarize(cases, labels)
        self.assertEqual([len(summary[k]) for k in ('input', 'actions', 'contrasts', 'label_intervention')], [4, 16, 24, 2])
        self.assertEqual(summary['actions']['clean/rare8/native_grouped']['J']['total']['values'], [.5, 1.5, 2.5])
        self.assertEqual(summary['actions']['clean/rare8/zero']['F']['cosine']['present_count'], 0)
        self.assertEqual(summary['actions']['clean/rare8/zero']['F']['sign_status_counts']['roundoff_unresolved'], 3)
        with self.assertRaises(s.AnalysisError):
            s.summarize(cases[:-1], labels)
        with self.assertRaises(s.AnalysisError):
            s.summarize(cases+[cases[0]], labels)


class AdmissionTests(unittest.TestCase):
    def test_exact_selected_inventory_is_21_nonstream_archives(self):
        names = s.selected_names()
        self.assertEqual(len(names), 21)
        self.assertEqual(len(set(names)), 21)
        self.assertEqual(sum(n.startswith('oracles-') for n in names), 3)
        self.assertEqual(sum(n.startswith('common-action-') for n in names), 6)
        self.assertEqual(sum(n.startswith('observer-') for n in names), 12)
        self.assertFalse(any('stream-' in n or 'readout-' in n for n in names))
        self.assertEqual(s.ARCHIVE_BYTES, 189384513)

    def test_regular_hash_receipts_duplicates_and_escape(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root/'invented.json'
            path.write_text('{}')
            receipt = {'path': path.name, 'size_bytes': 2, 'sha256': s.sha256(path)}
            self.assertEqual(s.verify_file(root, receipt), path)
            self.assertEqual(s.receipt_index([receipt])[path.name], receipt)
            for bad in ({**receipt, 'sha256': '0'*64}, {**receipt, 'size_bytes': 3}):
                with self.assertRaises(s.AnalysisError):
                    s.verify_file(root, bad)
            with self.assertRaises(s.AnalysisError):
                s.receipt_index([receipt, receipt])
            with self.assertRaises(s.AnalysisError):
                s.receipt_index([{**receipt, 'path': '../invented.json'}])
            link = root/'link.json'
            link.symlink_to(path)
            with self.assertRaises(s.AnalysisError):
                s.verify_file(root, {**receipt, 'path': link.name})

    def test_restricted_loader_schema_with_stub_torch_and_incidental_envelopes(self):
        class Tensor:
            def __init__(self, array):
                self.array, self.device, self.dtype = array, SimpleNamespace(type='cpu'), 'float32'
                self.shape = array.shape
            def detach(self):
                return self
            def numpy(self):
                return self.array
        tensor = Tensor(np.array([1., -2., 3.], np.float32))
        stub = SimpleNamespace(__version__='2.11.0+cu128', Tensor=Tensor, float32='float32',
                               set_num_threads=Mock(), load=Mock())
        base = {'schema': s.PARENT_SCHEMA, 'seed': s.SEEDS[0], 'parent_sha256': 'a'*64}
        payloads = [
            ({**base, 'groups': {q: {'mean_gradient': tensor, 'inputs': object(), 'targets': object()}
                                for q in s.PROBES}, 'use': 'oracle diagnostic only; never delivered or observed'}, 'oracles', None, None),
            ({**base, 'cell': 'clean', 'g_star': tensor, 'mean_interleaved': object(), 'mean_grouped': object()}, 'common', 'clean', None),
            ({**base, 'cell': 'clean', 'schedule': 'grouped', 'g_star_sha256': 'b'*64, 'post_action': tensor,
              'pre_action': object(), 'at150': {'model_adam': object(), 'observer': object()},
              'at151': {'model_adam': object(), 'observer': object()}}, 'observer', 'clean', 'grouped')]
        with patch.dict(sys.modules, {'torch': stub}), patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': ''}), patch.object(s, 'PARAMETERS', 3):
            for payload, kind, cell, schedule in payloads:
                stub.load.return_value = payload
                arrays, meta = s.restricted_vectors(Path('fabricated-only.pt'), kind, s.SEEDS[0], cell, schedule)
                self.assertTrue(all(np.array_equal(x, tensor.array) for x in arrays.values()))
                stub.load.assert_called_with(Path('fabricated-only.pt'), map_location='cpu', weights_only=True, mmap=True)
            stub.load.return_value = {**payloads[-1][0], 'seed': s.SEEDS[1]}
            with self.assertRaises(s.AnalysisError):
                s.restricted_vectors(Path('fabricated-only.pt'), 'observer', s.SEEDS[0], 'clean', 'grouped')

    def test_exclusive_capped_JSON_and_reserved_failure_receipt(self):
        with tempfile.TemporaryDirectory() as temp:
            run = s.Run(Path(temp), s.time.monotonic())
            with patch.object(run, 'check'):
                receipt = run.save('tiny.json', {'made_up': 1})
                self.assertEqual(receipt['sha256'], s.sha256(Path(temp)/'tiny.json'))
                with self.assertRaises(FileExistsError):
                    run.save('tiny.json', {})
                with patch.object(s, 'MAX_BYTES', 512), patch.object(s, 'FOOTER_RESERVE', 128):
                    with self.assertRaises(s.AnalysisError):
                        run.save('too-big.json', {'text': 'x'*400})
                    failure = run.save('failed.json', {'status': 'failed', 'no_retry': True}, footer=True)
                    self.assertEqual(failure['path'], 'failed.json')
                self.assertTrue((Path(temp)/'tiny.json').exists())

    def test_no_execute_does_not_read_or_import_scientific_loader(self):
        with patch.object(s, 'resource_guard') as guard, patch.object(s, 'Inputs') as inputs:
            with self.assertRaisesRegex(s.AnalysisError, 'explicit --execute'):
                s.main(['--output-dir', '/nonexistent/analysis-001', '--expected-sources-json', '/nonexistent/map.json'])
            guard.assert_not_called()
            inputs.assert_not_called()
        command = [sys.executable, '-c', 'import runpy,sys; runpy.run_path('+repr(str(SOURCE))+'); assert "torch" not in sys.modules']
        subprocess.run(command, check=True, capture_output=True)


if __name__ == '__main__':
    unittest.main()
