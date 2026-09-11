"""Independent-auditor fixtures: fabricated data/artifacts only, no torch."""
import io
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from experiments import spectral_wrong_label_augmentation_audit as a


class PureTests(unittest.TestCase):
    def setUp(self):
        self.dim = a.Dimensions(per_class=5, wrong_per_class=4, updates=200, batch=3, eval_steps=(0, 100, 200))
        self.labels = np.repeat(np.arange(10, dtype=np.int64), 12)

    def test_exact_corruption_and_fixed_repeated_views(self):
        seed = a.SEEDS[0]
        plan = a.regenerate_plan(self.labels, seed, self.dim)
        a.validate_plan(plan, self.labels, seed, self.dim)
        mask, true, assigned = (plan[k] for k in ('corruption_mask', 'train_labels', 'assigned_labels'))
        self.assertEqual(mask.sum(), 40)
        np.testing.assert_array_equal(np.bincount(true[mask], minlength=10), 4)
        np.testing.assert_array_equal(assigned != true, mask)
        offsets = np.random.Generator(np.random.PCG64(np.random.SeedSequence([4, seed]))).integers(
            1, 10, size=50, dtype=np.int64)
        np.testing.assert_array_equal(assigned[mask], (true[mask] + offsets[mask]) % 10)
        occurrence_labels = assigned[plan['occurrences']]
        repeated = plan['occurrences'] == plan['occurrences'][0, 0]
        self.assertTrue(np.all(occurrence_labels[repeated] == occurrence_labels[0, 0]))
        broken = {**plan, 'assigned_labels': assigned.copy()}
        broken['assigned_labels'][np.flatnonzero(mask)[0]] = true[np.flatnonzero(mask)[0]]
        with self.assertRaises(a.AuditError):
            a.validate_plan(broken, self.labels, seed, self.dim)
        with self.assertRaises(a.AuditError):
            a.validate_plan({**plan, 'corruption_mask': mask.astype(np.int8)}, self.labels, seed, self.dim)

    def test_wrong_target_fit_is_not_true_competence(self):
        plan = a.regenerate_plan(self.labels, a.SEEDS[0], self.dim)
        train = np.eye(10, dtype=np.float32)[plan['assigned_labels']] * 2
        heldout = np.eye(10, dtype=np.float32)[plan['eval_labels']] * 2
        row = a.evaluation_row(200, train, heldout, plan)
        self.assertEqual(row['train_assigned']['accuracy'], 1.)
        self.assertEqual(row['wrong_target']['count'], 40)
        self.assertEqual(row['wrong_target']['accuracy'], 1.)
        self.assertEqual(row['wrong_true']['accuracy'], 0.)
        self.assertEqual(row['train_clean']['accuracy'], .2)
        self.assertEqual(row['heldout']['accuracy'], 1.)
        self.assertAlmostEqual(row['heldout']['ce'], math.log(1 + 9 * math.exp(-2)), places=14)
        self.assertAlmostEqual(row['wrong_true']['ce'] - row['wrong_target']['ce'], 2., places=14)

    def test_summary_warmup_and_benefit_signs(self):
        branches = []
        for seed in a.SEEDS:
            for policy in a.POLICIES:
                for aug in a.AUGMENTATIONS:
                    benefit = (0.2 if policy == 'native32' else .1) if aug == 'translate' else 0
                    warmup = {key: {'accuracy': .1, 'ce': 2.} for key in a.MEASUREMENTS}
                    endpoint = {key: {'accuracy': .5 + benefit, 'ce': 1. - benefit} for key in a.MEASUREMENTS}
                    branches.append({'seed': seed, 'policy': policy, 'augmentation': aug,
                        'metrics': [{'step': 100, **warmup}, {'step': 4000, **endpoint}]})
        summary = a.summarize(branches)
        self.assertAlmostEqual(summary['interaction']['accuracy']['mean'], .1)
        self.assertAlmostEqual(summary['interaction']['ce']['mean'], .1)
        self.assertAlmostEqual(summary['change_from_warmup']['wrong_target']['raw/none']['ce']['mean'], -1.)
        self.assertEqual(set(summary['endpoint_by_measurement']), set(a.MEASUREMENTS))
        self.assertEqual(summary['warmup_by_measurement']['heldout']['raw/none']['accuracy']['values'], [.1] * 3)
        with self.assertRaises(a.AuditError):
            a.summarize(branches[:-1])


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.inputs, self.sources = self.root / 'input', self.root / 'source'
        self.inputs.mkdir()
        self.sources.mkdir()
        (self.sources / 'fixture.py').write_text('# fabricated source only\n')
        self.labels = np.repeat(np.arange(10, dtype=np.int64), 12)
        self.dim = a.Dimensions(per_class=5, wrong_per_class=4, updates=200, batch=3, eval_steps=(0, 100, 200))
        self.receipts, self.branches, self.roster = [], [], []
        for seed in a.SEEDS:
            plan = a.regenerate_plan(self.labels, seed, self.dim)
            self.save(f'plan-s{seed}.npz', arrays=plan)
            self.save(f'initial-s{seed}.pt', payload=b'opaque, never deserialized')
            for policy in a.POLICIES:
                for aug in a.AUGMENTATIONS:
                    name = f's{seed}-{policy}-{aug}'
                    row = {'seed': seed, 'policy': policy, 'augmentation': aug}
                    self.roster.append(row)
                    tr, ev = np.zeros((3, 50, 10), dtype=np.float32), np.zeros((3, 50, 10), dtype=np.float32)
                    for i, strength in ((1, .2 if aug == 'translate' else .1), (2, .4 if policy == 'raw' else .3)):
                        tr[i] = np.eye(10, dtype=np.float32)[plan['assigned_labels']] * strength
                        ev[i] = np.eye(10, dtype=np.float32)[plan['eval_labels']] * strength
                    logits = self.save(f'logits-{name}.npz', arrays={
                        'steps': np.array([0, 100, 200], dtype=np.int64), 'train': tr, 'heldout': ev})
                    stream = self.save(f'stream-{name}.npz', arrays={'loss': np.zeros(200, dtype=np.float64)})
                    warmup = self.save(f'warmup-{name}.pt', payload=b'opaque warmup')
                    final = self.save(f'final-{name}.pt', payload=b'opaque final')
                    self.branches.append({**row, 'name': name, 'initial_model_sha256': '0' * 64,
                        'warmup_learning_sha256': ('1' if aug == 'none' else '2') * 64,
                        'logits_receipt': logits, 'stream_receipt': stream, 'warmup_receipt': warmup, 'final_receipt': final,
                        'training_seconds': 1., 'augmentation_seconds': 0., 'evaluation_seconds': .1, 'wall_seconds': 1.2,
                        'metrics': [a.evaluation_row(step, tr[i], ev[i], plan) for i, step in enumerate((0, 100, 200))]})
        self.result = {'schema': a.SCHEMA, 'status': 'complete', 'source_pins': {
            'fixture.py': a.common.sha((self.sources / 'fixture.py').read_bytes())},
            'data_pins': dict(a.common.DATA_PINS), 'eval_steps': [0, 100, 200],
            'branches': self.branches, 'roster': self.roster, 'receipts': self.receipts}
        self.write_result()

    def save(self, name, *, arrays=None, payload=None):
        if arrays is not None:
            handle = io.BytesIO()
            np.savez(handle, **arrays)
            payload = handle.getvalue()
        (self.inputs / name).write_bytes(payload)
        receipt = {'path': name, 'size_bytes': len(payload), 'sha256': a.common.sha(payload)}
        self.receipts.append(receipt)
        return receipt

    def write_result(self):
        (self.inputs / 'results.json').write_text(json.dumps(self.result))

    def audit(self):
        return a.audit_saved(self.inputs, self.labels, dimensions=self.dim, source_root=self.sources)

    def test_full_small_bundle_and_zero_loss(self):
        report = self.audit()
        self.assertEqual(report['status'], 'PASS')
        self.assertEqual(report['logical_evaluation_records'], 36)
        self.assertEqual(report['verified_artifacts'], 54)
        self.assertEqual(report['corruption']['count_per_seed'], 40)
        self.assertEqual(report['max_absolute_scalar_error'], 0.)
        self.assertEqual(report['input_bytes_read_once'], sum(p.stat().st_size for p in self.inputs.iterdir()))

    def test_wrong_subset_metric_mismatch(self):
        self.branches[0]['metrics'][-1]['wrong_target']['count'] = 50
        self.write_result()
        with self.assertRaisesRegex(a.AuditError, 'integer metric differs'):
            self.audit()

    def test_corruption_modified_even_with_correct_receipt(self):
        receipt = self.receipts[0]
        path = self.inputs / receipt['path']
        with np.load(path, allow_pickle=False) as archive:
            values = {k: archive[k] for k in archive.files}
        i = np.flatnonzero(values['corruption_mask'])[0]
        values['assigned_labels'][i] = values['train_labels'][i]
        handle = io.BytesIO()
        np.savez(handle, **values)
        payload = handle.getvalue()
        path.write_bytes(payload)
        receipt.update(size_bytes=len(payload), sha256=a.common.sha(payload))
        self.write_result()
        with self.assertRaisesRegex(a.AuditError, 'independent plan differs'):
            self.audit()

    def test_invalid_pairing(self):
        self.branches[2]['warmup_learning_sha256'] = 'f' * 64
        self.write_result()
        with self.assertRaisesRegex(a.AuditError, 'warmup learning pairing'):
            self.audit()

    def test_hash_inventory_path_and_nonfinite(self):
        stray = self.inputs / 'stray.json'
        stray.write_text('{}')
        with self.assertRaisesRegex(a.AuditError, 'inventory'):
            self.audit()
        stray.unlink()
        self.branches[0]['training_seconds'] = float('nan')
        self.write_result()
        with self.assertRaisesRegex(a.AuditError, 'nonfinite JSON'):
            self.audit()

    def test_wrong_checkpoint_receipt(self):
        self.branches[0]['final_receipt'] = self.branches[1]['final_receipt']
        self.write_result()
        with self.assertRaisesRegex(a.AuditError, 'branch receipt'):
            self.audit()


if __name__ == '__main__':
    unittest.main()
