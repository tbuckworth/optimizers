"""Tiny fabricated CPU arrays only. No existing states/data/GPU are accessed."""
import copy
import math
import unittest

import numpy as np

from experiments import spectral_augmentation_state_audit as audit


def zero_fixture():
    d = audit.Dimensions(p=4, b=2, n=3, k=1, sizes=(1, 1, 1, 1))
    a = {k: np.zeros(shape, dtype=dtype) for k, (dtype, shape) in audit.specs(d).items()}
    a['param_sizes'][:] = d.sizes
    a['adam_steps'][:] = 100
    a['basis_before'][0, 0] = 1
    a['basis_before_rank'][()] = 1
    a['basis_after'][:, 0, 0] = 1
    a['basis_after_ranks'][:] = 1
    a['valid'][:, [0, 1, 4]] = True
    a['scales'][:, :2] = 1
    return d, a, np.array([0, 1, 2], dtype=np.int64)


class StateAuditTests(unittest.TestCase):
    def test_zero_actions_and_null_controls(self):
        d, a, labels = zero_fixture()
        out = audit.audit_parent(a, labels, d)
        self.assertAlmostEqual(out['baseline']['original']['ce'], math.log(10))
        self.assertIsNone(out['geometry']['within']['operator_retention'])
        self.assertEqual(len(out['readouts']), 100)
        for row in out['summary']:
            self.assertEqual(row['actions']['raw']['ce_improvement'], 0)
            self.assertIsNone(row['contrasts']['native_to_raw-minus-raw']['ce_improvement']['mean'])
            self.assertEqual(row['contrasts']['native-minus-raw']['ce_improvement']['mean'], 0)

    def test_bad_invalid_filler_rejected(self):
        d, a, labels = zero_fixture()
        a['after_logits'][1, 2, 0, 0, 0, 0] = 1
        with self.assertRaisesRegex(audit.common.AuditError, 'invalid zero fill'):
            audit.audit_parent(a, labels, d)

    def test_bad_rank_and_nonfinite_rejected(self):
        d, a, labels = zero_fixture()
        a['basis_before_rank'][()] = 2
        with self.assertRaisesRegex(audit.common.AuditError, 'basis rank'):
            audit.audit_parent(a, labels, d)
        a['basis_before_rank'][()] = 1
        a['theta'][0] = np.nan
        with self.assertRaisesRegex(audit.common.AuditError, 'nonfinite'):
            audit.audit_parent(a, labels, d)

    def test_operator_not_coordinate_energy_and_identity(self):
        g = np.zeros((5, 2, 3), np.float32)
        g[1:, 0, 0], g[1:, 1, 0] = 1, -1
        g[1:, :, 1] = np.array([1, -1, 1, -1])[:, None]
        q = np.array([[2.], [0.], [0.]])
        out = audit.geometry(g, q)
        self.assertEqual(out['total']['trace'], 2)
        self.assertEqual(out['between']['trace'], 1)
        self.assertEqual(out['within']['trace'], 1)
        self.assertEqual(out['between']['operator_trace'], 16)
        self.assertEqual(out['within']['operator_trace'], 0)
        self.assertEqual(out['identity_residuals']['operator_trace'], 0)

    def test_adam_carried_zero_gradient_analytic(self):
        delta = audit.adam_reference(np.zeros(4), np.ones(4), np.full(4, 4),
                                    np.zeros(4), np.full(4, 100), np.ones(4, dtype=int))
        exact = -.001*(.9/(1-.9**101))/(math.sqrt(3.996/(1-.999**101))+1e-8)
        np.testing.assert_allclose(delta, exact, atol=1e-15, rtol=0)
        self.assertTrue(np.all(delta < 0))

    def test_delta_relative_tolerance_not_parameter_scale(self):
        with self.assertRaises(audit.common.AuditError):
            audit.close_vector(np.full(4, .0001), np.full(4, .00001), 'step')

    def test_empty_native_basis_is_canonical_identity_fallback(self):
        g = np.array([1, -2, 3], np.float32)
        np.testing.assert_array_equal(audit.native_projection(np.zeros((3, 0)), g), g)

    def test_four_view_summary_requires_every_draw(self):
        d, a, labels = zero_fixture()
        rows = audit.audit_parent(a, labels, d)['readouts']
        for r in rows:
            if r['action'] == 'raw' and r['view'] == 2:
                r['ce_improvement'] = None
                r['available'] = False
        out = audit.summarize_parent(rows)
        translated = next(r for r in out if r['input_mode'] == 'translate')
        self.assertIsNone(translated['actions']['raw']['ce_improvement'])
        contrast = translated['contrasts']['native-minus-raw']['ce_improvement']
        self.assertEqual(contrast['available_draws'], 3)
        self.assertIsNone(contrast['mean'])
        self.assertEqual(contrast['draws'], [0, None, 0, 0])

    def test_plans_reconstruct_from_fabricated_labels(self):
        labels = np.repeat(np.arange(10, dtype=np.int64), 1000)
        seed = audit.SEEDS[0]
        a = audit.diagnostic_plan(labels, seed)
        self.assertFalse(np.intersect1d(a['train_ids'], a['eval_ids']).size)
        self.assertEqual(len(np.unique(a['train_ids'])), 64)
        source = audit.common.regenerate_plan(labels, seed)
        rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([10, seed])))
        np.testing.assert_array_equal(a['train_ids'], source['train_ids'][rng.permutation(5000)[:64]])
        for key, value in a.items():
            np.testing.assert_array_equal(value, audit.diagnostic_plan(labels, seed)[key])

    def test_full_synthetic_producer_auditor_integration(self):
        import torch
        from experiments import spectral_augmentation_state_core as helper
        class TinyTracker:
            def __init__(self, model, optimizer):
                self.model, self.base_optimizer = model, optimizer
                self.V, self.proj_k, self.step_count = torch.eye(62, 32), None, 100

            def filter_grad(self):
                if self.step_count != 100:
                    raise AssertionError('observer copy reused')
                self.step_count += 1
                params = list(self.model.parameters())
                g = torch.cat([p.grad.flatten() for p in params])
                if g[0] < 0:
                    self.V = torch.roll(self.V, 17, dims=0)
                projected = self.V @ (self.V.T @ g)
                offset = 0
                for p in params:
                    p.grad = projected[offset:offset+p.numel()].reshape_as(p).clone()
                    offset += p.numel()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(37)
            model = torch.nn.Sequential(torch.nn.Linear(2, 4), torch.nn.ReLU(), torch.nn.Linear(4, 10))
        opt = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01,
                                betas=(.9, .999), eps=1e-8, foreach=False, fused=False)
        for p in model.parameters():
            opt.state[p] = {'step': torch.tensor(100.), 'exp_avg': torch.full_like(p, .02),
                            'exp_avg_sq': torch.full_like(p, .1)}
        tracker = TinyTracker(model, opt)
        tracker.V = torch.eye(62, 32)
        views = torch.arange(30, dtype=torch.float32).reshape(3, 5, 2)/30
        labels = torch.tensor([0, 1, 2])
        gradients = helper.collect_gradients(model, views, labels)
        proposals = helper.propose_actions(model, opt, tracker, gradients['batch'])
        objective_views = torch.arange(16, dtype=torch.float32).reshape(2, 4, 2)/16
        eval_labels = torch.tensor([0, 1, 2, 3])
        measured = helper.measure_actions(model, proposals, objective_views, eval_labels)
        d = audit.Dimensions(p=62, b=3, n=4, k=32, sizes=(8, 4, 40, 10))
        a = {key: value for key, value in {**proposals, **measured}.items() if key in audit.specs(d)}
        a.update(per_example_grads=gradients['per_example'].transpose(1, 0, 2).copy(), batch_grads=gradients['batch'])
        out = audit.audit_parent(a, eval_labels.numpy(), d)
        self.assertEqual(len(out['readouts']), 100)
        for row in out['readouts']:
            if row['available']:
                i, j, s, o = row['view'], helper.CANDIDATES.index(row['action']), helper.STEP_SCALES.index(row['fraction']), ('original', 'translate').index(row['objective'])
                self.assertAlmostEqual(row['ce'], measured['ce'][i, j, s, o], places=12)
                self.assertAlmostEqual(row['linear_data'], measured['data_derivative_utility'][i, j, s, o], places=12)
        corrupted = copy.deepcopy(a)
        corrupted['after_parameters'][0, 0, 0, 0] += .001
        with self.assertRaisesRegex(audit.common.AuditError, 'carried Adam'):
            audit.audit_parent(corrupted, eval_labels.numpy(), d)


if __name__ == '__main__':
    unittest.main()
