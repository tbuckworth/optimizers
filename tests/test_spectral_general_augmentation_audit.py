"""Small fabricated bundles only; no training, image IDX, torch or GPU use."""
import copy
import hashlib
import io
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from experiments import spectral_general_augmentation_audit as a


class MetricsTests(unittest.TestCase):
    def test_uniform_ties_extremes_and_float64_loss(self):
        logits = np.zeros((3, 10), dtype=np.float32)
        labels = np.array([0, 1, 9], dtype=np.int64)
        result = a.classification(logits, labels)
        self.assertEqual(result['correct'], 1)
        self.assertAlmostEqual(result['ce'], math.log(10), places=14)
        logits[1] = -10000
        logits[1, 1] = 10000
        logits[2] = 10000
        logits[2, 9] = -10000
        result = a.classification(logits, labels)
        self.assertEqual(result['correct'], 2)
        self.assertAlmostEqual(result['ce_sum'], math.log(10) + 20000 + math.log(9), places=9)
        for invalid in (logits.astype(np.float64), np.full((3, 10), np.nan, dtype=np.float32), logits[:, :9]):
            with self.assertRaises(a.AuditError):
                a.classification(invalid, labels)
        with self.assertRaises(a.AuditError):
            a.classification(logits, labels.astype(np.int8))
        with self.assertRaises(a.AuditError):
            a.compare_metrics({**result, 'count': True}, result)
        with self.assertRaises(a.AuditError):
            a.compare_metrics({**result, 'accuracy': result['accuracy'] + 1e-5}, result)

    def test_contrast_signs_keep_all_seed_values(self):
        branches = []
        for index, seed in enumerate(a.SEEDS):
            for policy in a.POLICIES:
                for augmentation in a.AUGMENTATIONS:
                    translated = augmentation == 'translate'
                    native = policy == 'native32'
                    accuracy = .5 + index * .01 + translated * (.2 if native else .1)
                    ce = 1. + index * .02 - translated * (.3 if native else .1)
                    branches.append({'seed': seed, 'policy': policy, 'augmentation': augmentation,
                        'metrics': [{'step': 4000, 'heldout': {'accuracy': accuracy, 'ce': ce}}]})
        summary = a.summarize(branches)
        np.testing.assert_allclose(summary['augmentation_benefits']['raw']['accuracy']['values'], .1)
        np.testing.assert_allclose(summary['augmentation_benefits']['native32']['ce']['values'], .3)
        self.assertAlmostEqual(summary['interaction']['accuracy']['mean'], .1)
        self.assertAlmostEqual(summary['interaction']['ce']['mean'], .2)
        self.assertEqual(len(summary['groups']), 4)
        with self.assertRaises(a.AuditError):
            a.summarize(branches[:-1])

    def test_independent_plan_and_wrong_entropy_order(self):
        dimensions = a.Dimensions(per_class=2, updates=200, batch=3, eval_steps=(0, 100, 200))
        raw = np.repeat(np.arange(10, dtype=np.int64), 6)
        plan = a.regenerate_plan(raw, a.SEEDS[0], dimensions)
        a.validate_plan(plan, raw, a.SEEDS[0], dimensions)
        rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([2, a.SEEDS[0]])))
        np.testing.assert_array_equal(plan['shifts'], rng.integers(-2, 3, size=(200, 3, 2), dtype=np.int8))
        broken = {**plan, 'shifts': plan['shifts'].astype(np.int64)}
        with self.assertRaises(a.AuditError):
            a.validate_plan(broken, raw, a.SEEDS[0], dimensions)
        broken = {**plan, 'train_ids': plan['train_ids'][::-1].copy()}
        with self.assertRaises(a.AuditError):
            a.validate_plan(broken, raw, a.SEEDS[0], dimensions)

    def test_strict_json(self):
        for payload in ('{"x":1,"x":2}', '{"x":NaN}', '{"x":1e999}'):
            with self.assertRaises(a.AuditError):
                a.strict_json(payload)


class BundleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.inputs = self.root / 'input'
        self.inputs.mkdir()
        self.sources = self.root / 'source'
        self.sources.mkdir()
        (self.sources / 'fixture.py').write_text('# fabricated source, not a model\n')
        self.dimensions = a.Dimensions(per_class=2, updates=200, batch=3, eval_steps=(0, 100, 200))
        self.labels = np.repeat(np.arange(10, dtype=np.int64), 6)
        self.receipts = []
        self.branches = []
        self.roster = []
        for seed in a.SEEDS:
            plan = a.regenerate_plan(self.labels, seed, self.dimensions)
            self.save(f'plan-s{seed}.npz', arrays=plan)
            self.save(f'initial-s{seed}.pt', payload=b'opaque fabricated checkpoint, never deserialized')
            for policy in a.POLICIES:
                for augmentation in a.AUGMENTATIONS:
                    name = f's{seed}-{policy}-{augmentation}'
                    self.roster.append({'seed': seed, 'policy': policy, 'augmentation': augmentation})
                    # Explicit small logits. Both policies agree through warmup.
                    z = np.zeros((3, 20, 10), dtype=np.float32)
                    z[1, :, 0] = .1 if augmentation == 'none' else .2
                    z[2, :, 0] = .3 if policy == 'raw' else .4
                    logits = self.save(f'logits-{name}.npz', arrays={
                        'steps': np.array([0, 100, 200], dtype=np.int64), 'train': z, 'heldout': z.copy()})
                    stream = self.save(f'stream-{name}.npz', arrays={'loss': np.ones(200, dtype=np.float64)})
                    warmup = self.save(f'warmup-{name}.pt', payload=b'opaque warmup')
                    final = self.save(f'final-{name}.pt', payload=b'opaque final')
                    metrics = [{'step': step, 'train': a.classification(z[i], plan['train_labels']),
                                'heldout': a.classification(z[i], plan['eval_labels'])}
                               for i, step in enumerate((0, 100, 200))]
                    self.branches.append({'name': name, 'seed': seed, 'policy': policy, 'augmentation': augmentation,
                        'initial_model_sha256': '0' * 64, 'warmup_learning_sha256': ('1' if augmentation == 'none' else '2') * 64,
                        'metrics': metrics, 'logits_receipt': logits, 'stream_receipt': stream,
                        'warmup_receipt': warmup, 'final_receipt': final, 'training_seconds': 1.,
                        'augmentation_seconds': 0., 'evaluation_seconds': .1, 'wall_seconds': 1.1})
        self.results = {'schema': a.SCHEMA, 'status': 'complete', 'roster': self.roster,
            'eval_steps': [0, 100, 200], 'branches': self.branches, 'receipts': self.receipts,
            'source_pins': {'fixture.py': hashlib.sha256((self.sources / 'fixture.py').read_bytes()).hexdigest()},
            'data_pins': dict(a.DATA_PINS)}
        self.write_results()

    def save(self, name, *, payload=None, arrays=None):
        if arrays is not None:
            handle = io.BytesIO()
            np.savez(handle, **arrays)
            payload = handle.getvalue()
        (self.inputs / name).write_bytes(payload)
        result = {'path': name, 'size_bytes': len(payload), 'sha256': hashlib.sha256(payload).hexdigest()}
        self.receipts.append(result)
        return result

    def write_results(self):
        (self.inputs / 'results.json').write_text(json.dumps(self.results))

    def run_audit(self):
        return a.audit_saved(self.inputs, self.labels, dimensions=self.dimensions, source_root=self.sources)

    def test_complete_small_saved_bundle(self):
        result = self.run_audit()
        self.assertEqual(result['status'], 'PASS')
        self.assertEqual(result['trajectories'], 12)
        self.assertEqual(result['logical_evaluation_records'], 36)
        self.assertEqual(result['verified_artifacts'], len(self.receipts))
        self.assertEqual(result['max_absolute_scalar_error'], 0.)
        self.assertEqual(result['input_bytes_read_once'], sum(p.stat().st_size for p in self.inputs.iterdir()))
        self.assertEqual(len(result['branches']), 12)

    def test_bad_inventory_hash_and_source(self):
        stray = self.inputs / 'unexpected.json'
        stray.write_text('{}')
        with self.assertRaisesRegex(a.AuditError, 'inventory'):
            self.run_audit()
        stray.unlink()
        opaque = self.inputs / f'initial-s{a.SEEDS[0]}.pt'
        original = opaque.read_bytes()
        opaque.write_bytes(bytes([original[0] ^ 1]) + original[1:])
        with self.assertRaisesRegex(a.AuditError, 'hash/size'):
            self.run_audit()
        opaque.write_bytes(original)
        (self.sources / 'fixture.py').write_text('# changed\n')
        with self.assertRaisesRegex(a.AuditError, 'source bytes changed'):
            self.run_audit()

    def test_invalid_pairing_and_metric(self):
        self.branches[2]['warmup_learning_sha256'] = 'f' * 64
        self.write_results()
        with self.assertRaisesRegex(a.AuditError, 'warmup learning pairing'):
            self.run_audit()
        self.branches[2]['warmup_learning_sha256'] = '1' * 64
        self.branches[0]['metrics'][-1]['heldout']['ce'] += 1e-4
        self.write_results()
        with self.assertRaisesRegex(a.AuditError, 'metric differs'):
            self.run_audit()

    def test_nonfinite_stream_and_duplicate_receipt(self):
        self.results['receipts'] = self.receipts + [self.receipts[0]]
        self.write_results()
        with self.assertRaisesRegex(a.AuditError, 'duplicate'):
            self.run_audit()
        self.results['receipts'] = self.receipts
        branch = self.branches[0]
        name = branch['stream_receipt']['path']
        handle = io.BytesIO()
        np.savez(handle, loss=np.full(200, np.nan, dtype=np.float64))
        payload = handle.getvalue()
        (self.inputs / name).write_bytes(payload)
        branch['stream_receipt'].update(size_bytes=len(payload), sha256=a.sha(payload))
        self.write_results()
        with self.assertRaisesRegex(a.AuditError, 'nonfinite array'):
            self.run_audit()

    def test_initial_logit_mismatch_even_with_updated_receipt_and_metrics(self):
        branch = self.branches[2]
        path = self.inputs / branch['logits_receipt']['path']
        with np.load(path, allow_pickle=False) as archive:
            values = {k: archive[k] for k in archive.files}
        values['train'][0, 0, 0] = 1.
        handle = io.BytesIO()
        np.savez(handle, **values)
        payload = handle.getvalue()
        path.write_bytes(payload)
        branch['logits_receipt'].update(size_bytes=len(payload), sha256=a.sha(payload))
        self.write_results()
        with self.assertRaisesRegex(a.AuditError, 'initial logits differ'):
            self.run_audit()

    def test_zero_loss_valid_negative_loss_invalid(self):
        branch = self.branches[0]
        name = branch['stream_receipt']['path']
        for loss in (0., -0.01):
            handle = io.BytesIO()
            np.savez(handle, loss=np.full(200, loss, dtype=np.float64))
            payload = handle.getvalue()
            (self.inputs / name).write_bytes(payload)
            branch['stream_receipt'].update(size_bytes=len(payload), sha256=a.sha(payload))
            self.write_results()
            if loss == 0:
                self.assertEqual(self.run_audit()['status'], 'PASS')
            else:
                with self.assertRaisesRegex(a.AuditError, 'nonnegative'):
                    self.run_audit()

    def test_wrong_checkpoint_binding_and_input_cap(self):
        self.branches[0]['final_receipt'] = self.branches[1]['final_receipt']
        self.write_results()
        with self.assertRaisesRegex(a.AuditError, 'checkpoint receipt path'):
            self.run_audit()
        too_large = [{'path': 'oversized.pt', 'size_bytes': 1024**3 + 1, 'sha256': '0' * 64}]
        with self.assertRaisesRegex(a.AuditError, 'input byte cap'):
            a.validate_inventory(self.inputs, too_large)

    def test_path_escape_and_missing_required_receipt(self):
        broken = copy.deepcopy(self.receipts)
        broken[0]['path'] = '../outside'
        with self.assertRaisesRegex(a.AuditError, 'direct-child'):
            a.validate_inventory(self.inputs, broken)
        del self.branches[0]['final_receipt']
        self.write_results()
        with self.assertRaisesRegex(a.AuditError, 'required branch receipt'):
            self.run_audit()


if __name__ == '__main__':
    unittest.main()
