"""Fabricated NumPy arrays only; no scientific producer, data, model, or GPU."""
import copy
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np

from experiments import spectral_augmentation_audit as a


D = a.Dimensions(common_train=2, rare_train=1, heldout_per_class=2, poison_count=2, batch=2)


def tiny_plan():
    true = np.repeat(np.arange(10, dtype=np.int64), [1 if d == 8 else 2 for d in range(10)])
    poison = true == 1
    return {'train_ids': np.arange(D.train, dtype=np.int64),
        'heldout_ids': np.arange(D.train, D.train + D.heldout, dtype=np.int64),
        'train_true': true, 'heldout_true': np.repeat(np.arange(10, dtype=np.int64), 2),
        'warmup_batches': np.zeros((100, 2), np.int64),
        'continuation_batches': np.zeros((1900, 2), np.int64),
        'diffuse_replacement_labels': np.where(true == 8, -1, 0),
        'diffuse_targets': true.copy(), 'diffuse_selected': np.zeros(D.train, bool),
        'poison_mask': poison, 'shared_sham_targets': np.where(poison, 0, true),
        'sham_patch_mask': poison.copy()}


def tiny_masks():
    gates = np.zeros((1900, 2), bool)
    gates[:, 0] = True
    out = {'gates': gates, 'centers': np.full((1900, 2, 2), 14, np.int16)}
    for mode, bounds in [('none', None), ('random', (10, 18, 10, 18)),
                         ('targeted', (0, 8, 0, 8)), ('opposite', (20, 28, 20, 28))]:
        rect = np.full((1900, 2, 4), -1, np.int16)
        if bounds is not None:
            rect[gates] = bounds
        out[mode + '_rectangles'] = rect
    return out


def actions(policy, warmup=False):
    result = []
    for step in (range(1, 101) if warmup else range(101, 2001)):
        observing = warmup or policy == 'native32'
        row = {'step': step, 'observer_step': step if observing else None,
            'basis_rank': 1 if observing else None, 'norm_law': None,
            'training_batch_ce': 1., 'raw_norm': 1., 'native_norm': 1., 'applied_norm': 1.}
        if step == 101:
            row['raw_gradient_sha256'] = 'a' * 64
            if observing:
                row['post_observe_tracker_sha256'] = 'b' * 64
        result.append(row)
    return result


def fabricated_acquisition(root):
    """Complete logical roster, tiny fabricated populations; never calls CLI."""
    receipts = {}
    def save(name, value):
        if name.endswith('.npz'):
            stream = io.BytesIO()
            np.savez(stream, **value)
            payload = stream.getvalue()
        elif name.endswith('.pt'):
            payload = b'opaque fixture; not a tensor archive'
        else:
            payload = json.dumps(value, allow_nan=False, separators=(',', ':')).encode()
        (root / name).write_bytes(payload)
        receipt = {'path': name, 'size_bytes': len(payload), 'sha256': hashlib.sha256(payload).hexdigest()}
        receipts[name] = receipt
        return receipt
    pins = {'fabricated_source.py': 'c' * 64}
    save('manifest.json', {'schema': a.SCHEMA, 'seeds': list(a.SEEDS), 'cells': list(a.CELLS),
        'policies': list(a.POLICIES), 'augmentations': list(a.MODES), 'warmup': 100, 'steps': 2000,
        'batch_size': 2, 'eval_steps': list(a.EVAL), 'rank': 32, 'cloud_spend_usd': 0,
        'diagnostics': [], 'branch_roster': a.expected_roster(), 'source_pins': pins,
        'data_pins': a.DATA_PINS, 'numpy': np.__version__})
    rows = []
    for seed in a.SEEDS:
        plan, masks = tiny_plan(), tiny_masks()
        pr = save(f'plan-s{seed}.npz', plan)
        mr = save(f'masks-s{seed}.npz', masks)
        ph, mh = {k: a.array_hash(v) for k, v in plan.items()}, {k: a.array_hash(v) for k, v in masks.items()}
        metadata = save(f'plan-s{seed}.json', {'seed': seed, 'array_hashes': ph,
            'mask_array_hashes': mh, 'sham_allocation': a.validate_plan(plan, D), 'numpy_version': np.__version__})
        initial, warmup = save(f'initial-s{seed}.pt', None), save(f'warmup-s{seed}.pt', None)
        save(f'warmup-actions-s{seed}.json', actions('raw', True))
        train = np.zeros((21, D.train, 10), np.float32)
        held = np.zeros((21, D.heldout, 10), np.float32)
        save(f'common-logits-s{seed}.npz', {'steps': np.array([0, 100], np.int64),
            'heldout_unpatched': held[:2], 'heldout_patched': held[:2],
            **{cell + '_train': train[:2] for cell in a.CELLS}})
        for identity in (r for r in a.expected_roster() if r['seed'] == seed):
            stem = a.identifier(identity)
            cell, mode, policy = identity['cell'], identity['augmentation'], identity['policy']
            assigned, patch_mask, counts = a.cell_values(plan, cell)
            curve = [a.evaluation_row(step, train[i], held[i], held[i], plan['train_true'],
                assigned, plan['heldout_true']) for i, step in enumerate(a.EVAL)]
            action = actions(policy)
            record = {**identity, 'schema': a.SCHEMA, 'plan': pr, 'mask_plan': mr, 'plan_metadata': metadata,
                'plan_array_hashes': ph, 'mask_array_hashes': mh, 'initial_state': initial,
                'warmup_state': warmup, 'initial_state_sha256': '1' * 64, 'warmup_state_sha256': '2' * 64,
                'restored_state_sha256': '2' * 64, 'full_state_fork_check': 'PASS',
                'final_state_sha256': '3' * 64, 'cpu_cell_inputs_sha256': '4' * 64,
                'assigned_targets_sha256': a.array_hash(assigned), 'cell_patch_mask_sha256': a.array_hash(patch_mask),
                'cell_counts': counts, 'coverage': a.coverage(masks[mode + '_rectangles'], masks['gates'],
                    plan['continuation_batches'], patch_mask, plan['train_true']),
                'final_state': save(f'final-{stem}.pt', None),
                'logits': save(f'logits-{stem}.npz', {'steps': np.array(a.EVAL, np.int64), 'train': train,
                    'heldout_unpatched': held, 'heldout_patched': held}),
                'actions': save(f'actions-{stem}.json', action),
                'first_action_binding': a.validate_actions(action, policy), 'diagnostics': [],
                'curve': curve, 'warmup_plus_branch_seconds': 1.}
            receipt = save(f'curve-{stem}.json', record)
            rows.append({**identity, 'curve': receipt, 'endpoint': curve[-1], 'warmup': curve[1],
                         'warmup_plus_branch_seconds': 1.})
        save(f'pairing-s{seed}.json', {'status': 'PASS', 'full_state_forks': 24,
             'first_raw_gradient_sha256': {c + '/' + m: 'a' * 64 for c in a.CELLS for m in a.MODES}})
    result = save('results.json', {'schema': a.SCHEMA, 'rows': rows})
    save('complete.json', {'schema': a.SCHEMA, 'status': 'complete', 'completed_trajectories': 72,
        'completed_diagnostics': 0, 'results': result, 'source_pins': pins, 'data_pins': a.DATA_PINS,
        'artifact_bytes_before_completion': sum(r['size_bytes'] for r in receipts.values()),
        'receipts': list(receipts.values())})


class MetricTests(unittest.TestCase):
    def test_stable_ce_ties_counts_and_empty_group(self):
        y = np.arange(10, dtype=np.int64)
        logits = np.full((10, 10), 1000., np.float32)
        stats = a.classification(logits, y)
        self.assertEqual(stats['correct'], 1)
        self.assertAlmostEqual(stats['ce'], np.log(10), places=13)
        logits[:, 0] = -1000.
        self.assertTrue(np.isfinite(a.classification(logits, y)['ce']))
        self.assertEqual(a.classification(logits, y, y < 0),
                         {'count': 0, 'correct': 0, 'ce_sum': 0., 'accuracy': None, 'ce': None})

    def test_macro_differs_from_micro_and_cue_membership_is_true_label(self):
        y = np.array([0] * 10 + list(range(1, 10)), np.int64)
        z = np.zeros((len(y), 10), np.float32)
        stats = a.grouped(z, y)
        self.assertAlmostEqual(stats['balanced_total']['accuracy'], .1)
        self.assertAlmostEqual(stats['majority_macro']['accuracy'], 1 / 9)
        self.assertAlmostEqual(stats['micro']['accuracy'], 10 / 19)
        patched = z.copy()
        z[:, 1] = 1
        row = a.evaluation_row(2000, z, z, patched, y, y, y)
        self.assertEqual(row['cue']['majority_nonzero']['count'], 8)
        self.assertEqual(row['cue']['majority_nonzero']['patch_excess'], 1.)
        self.assertIsNone(row['train_actually_changed']['ce'])

    def test_reject_invalid_labels_nonfinite_and_wrong_dtype(self):
        for z in (np.zeros((10, 10), np.float64), np.full((10, 10), np.nan, np.float32)):
            with self.assertRaises(a.AuditError):
                a.classification(z, np.arange(10))
        with self.assertRaises(a.AuditError):
            a.classification(np.zeros((10, 10), np.float32), np.arange(1, 11))

    def test_scalar_comparison_rejects_type_and_metric_tampering(self):
        c = a.Checks()
        c.equal(1. + 1e-11, 1.)
        for left, right in [(True, 1), (1, True), (1.01, 1.), (None, 0.)]:
            with self.assertRaises(a.AuditError):
                c.equal(left, right)


class ArtifactTests(unittest.TestCase):
    def test_json_and_npz_reject_unsafe_encodings(self):
        for value in (b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":1e999}'):
            with self.assertRaises(a.AuditError):
                a.parse_json(value)
        buf = io.BytesIO()
        np.savez(buf, bad=np.array([object()], dtype=object))
        with self.assertRaises(a.AuditError):
            a.parse_npz(buf.getvalue())
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, 'w') as archive:
            archive.writestr('../bad.npy', b'bad')
        with self.assertRaises(a.AuditError):
            a.parse_npz(buf.getvalue())

    def test_once_hash_and_confinement(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'value.json').write_bytes(b'{}')
            r = a.Reader(root)
            good = {'path': 'value.json', 'size_bytes': 2, 'sha256': hashlib.sha256(b'{}').hexdigest()}
            r.receipt(good)
            self.assertEqual(r.json('value.json'), {})
            self.assertEqual(r.json('value.json'), {})  # cached, not a second file read
            with self.assertRaises(a.AuditError):
                r.payload('value.json')
            for name in ('../value.json', str(root / 'value.json')):
                with self.assertRaises(a.AuditError):
                    r.receipt({**good, 'path': name})
            (root / 'link.json').symlink_to(root / 'value.json')
            with self.assertRaises(a.AuditError):
                r.receipt({**good, 'path': 'link.json'})
            r = a.Reader(root)
            r.receipt({**good, 'sha256': 'f' * 64})
            with self.assertRaises(a.AuditError):
                r.json('value.json')

    def test_plan_and_masks_detect_changed_rare_label_or_geometry(self):
        plan, masks = tiny_plan(), tiny_masks()
        a.validate_plan(plan, D)
        a.validate_masks(masks, D)
        plan['shared_sham_targets'][plan['train_true'] == 8] = 0
        with self.assertRaises(a.AuditError):
            a.validate_plan(plan, D)
        masks['targeted_rectangles'][0, 0, 1] = 7
        with self.assertRaises(a.AuditError):
            a.validate_masks(masks, D)

    def test_actions_detect_step_and_policy_mismatches(self):
        values = actions('native32')
        a.validate_actions(values, 'native32')
        values[4]['step'] += 1
        with self.assertRaises(a.AuditError):
            a.validate_actions(values, 'native32')
        with self.assertRaises(a.AuditError):
            a.validate_actions(actions('raw'), 'native32')

    def test_import_and_missing_execute_are_inert(self):
        program = "import sys; from experiments import spectral_augmentation_audit; assert 'torch' not in sys.modules"
        subprocess.run([sys.executable, '-c', program], check=True)
        with patch.object(a, 'audit_saved', side_effect=AssertionError('must not read')):
            with self.assertRaises(a.AuditError):
                a.main(['--source-dir', '/does/not/exist', '--output-dir', '/does/not/exist/out'])

    def test_complete_fabricated_roster_and_opaque_checkpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fabricated_acquisition(root)
            report, summary = a.audit_saved(root, dimensions=D)
            self.assertEqual(report['status'], 'PASS')
            self.assertEqual(report['logical_evaluation_records'], 1512)
            self.assertEqual(report['verified_artifacts'], 315)
            self.assertEqual(summary['cue_association']['I_Q']['random']['values'], [0., 0., 0.])
            self.assertEqual(set(p.name for p in root.iterdir()), a.expected_files())
            (root / 'extra.json').write_text('{}')
            with self.assertRaises(a.AuditError):
                a.audit_saved(root, dimensions=D)


if __name__ == '__main__':
    unittest.main()
