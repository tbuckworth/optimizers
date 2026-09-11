"""Dataset-free CPU fixtures; imports no producer or prior experiment module.

CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 python3 -m unittest test_numerical_contract -v
"""
import json
import math
import os
import unittest

if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
    raise RuntimeError("Run these fixtures with CUDA_VISIBLE_DEVICES='' explicitly")
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(name, "1")
import numpy as np
import torch

U32, U64, H32, H64, K, P = 2.**-24, 2.**-53, 2.**-126, 2.**-1022, 32., 50890
SUMMARY = {"adam_coordinates": 0, "adam_max_ratios": dict(moment=0., variance=0., endpoint=0.),
           "ce_max_abs_error": 0., "ce_max_ratio": 0., "gradient_max_abs_error": 0.,
           "gradient_max_ratio": 0., "gradient_max_l2_error": 0.}


def gamma(n, u):
    return n*u/(1-n*u)


def tl(a, b):
    return 1e-10 + 1e-11*np.maximum(1., np.maximum(np.abs(a), np.abs(b)))


def tq(a, b):
    return 1e-12 + 1e-9*np.maximum(np.abs(a), np.abs(b))


def rsum(terms):
    terms = np.asarray(terms, dtype=np.float64)
    return 2*gamma(2*len(terms)+4, U64)*np.abs(terms).sum() + 4*len(terms)*H64


def adam_reference(theta, a, m, v, t):
    """NumPy implementation of the algebra and unchanged K=32 envelope."""
    theta, a, m, v = [np.asarray(x, dtype=np.float64) for x in (theta, a, m, v)]
    M, V = .9*m+.1*a, .999*v+.001*a*a
    b, s, q = math.sqrt(1-.999**t), .001/(1-.9**t), 1-.001*.01
    D = np.sqrt(V)/b + 1e-8
    em, ev = K*U32*(np.abs(m)+.1*np.abs(a))+K*H32, K*U32*V+K*H32
    lo, hi = np.sqrt(np.maximum(V-ev, 0))/b+1e-8, np.sqrt(V+ev)/b+1e-8
    ed = np.maximum(D-lo, hi-D)+K*U32*hi+K*H32
    lower = np.maximum(D-ed, 1e-8/2)
    eq = em/lower+np.abs(M)*ed/(D*lower)
    ep = abs(s)*eq+K*U32*(np.abs(q*theta)+abs(s)*(np.abs(M)/D+eq))+K*H32
    return (M, V, q*theta-s*M/D), (em, ev, ep)


def numpy_mlp(parameters, inputs, labels, chunk):
    """Independent explicit softmax-CE and chain-rule gradients, no autograd."""
    w1, b1, w2, b2 = [np.asarray(x, dtype=np.float64) for x in parameters]
    inputs = np.asarray(inputs, dtype=np.float64)
    totals = [np.zeros_like(x) for x in (w1, b1, w2, b2)]
    loss = 0.
    for start in range(0, len(inputs), chunk):
        x, y = inputs[start:start+chunk], labels[start:start+chunk]
        z = x@w1.T+b1
        hidden = np.maximum(z, 0.)
        logits = hidden@w2.T+b2
        shifted = logits-logits.max(axis=1, keepdims=True)
        exponentials = np.exp(shifted)
        denominators = exponentials.sum(axis=1, keepdims=True)
        loss += float((np.log(denominators[:, 0])-shifted[np.arange(len(y)), y]).sum())
        dlogits = exponentials/denominators
        dlogits[np.arange(len(y)), y] -= 1.
        dz = (dlogits@w2)*(z > 0.)  # Derivative at exactly zero is zero.
        gradients = (dz.T@x, dz.sum(axis=0), dlogits.T@hidden, dlogits.sum(axis=0))
        for total, gradient in zip(totals, gradients):
            total += gradient
    return loss/len(inputs), np.concatenate([x.ravel() for x in totals])/len(inputs)


def torch_mlp(parameters, inputs, labels, chunk):
    tensors = [torch.tensor(x, dtype=torch.float64, requires_grad=True) for x in parameters]
    totals = [torch.zeros_like(x) for x in tensors]
    loss = 0.
    for start in range(0, len(inputs), chunk):
        x = torch.tensor(inputs[start:start+chunk], dtype=torch.float64)
        y = torch.tensor(labels[start:start+chunk], dtype=torch.int64)
        w1, b1, w2, b2 = tensors
        logits = torch.relu(x@w1.T+b1)@w2.T+b2
        value = torch.nn.functional.cross_entropy(logits, y, reduction="sum")
        loss += float(value.detach())
        for total, gradient in zip(totals, torch.autograd.grad(value, tensors)):
            total.add_(gradient)
    return loss/len(inputs), torch.cat([x.reshape(-1) for x in totals]).numpy()/len(inputs)


class NumericalContractFixtures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        if torch.cuda.is_initialized():
            raise RuntimeError("Fixture unexpectedly has a CUDA context")

    @classmethod
    def tearDownClass(cls):
        SUMMARY["cuda_initialized"] = torch.cuda.is_initialized()
        print("NUMERICAL_FIXTURES " + json.dumps(SUMMARY, sort_keys=True, allow_nan=False))

    def compare_mlp(self, parameters, x, y, chunk):
        self.assertEqual(x.dtype, np.float32)  # Promote given bytes, never renormalize.
        native, qnative = torch_mlp(parameters, x, y, chunk)
        reference, qreference = numpy_mlp(parameters, x, y, chunk)
        errors = np.abs(qnative-qreference)
        self.assertLessEqual(abs(native-reference), float(tl(native, reference)))
        self.assertTrue(np.all(errors <= tq(qnative, qreference)))
        SUMMARY["ce_max_abs_error"] = max(SUMMARY["ce_max_abs_error"], abs(native-reference))
        SUMMARY["ce_max_ratio"] = max(SUMMARY["ce_max_ratio"], float(abs(native-reference)/tl(native, reference)))
        SUMMARY["gradient_max_abs_error"] = max(SUMMARY["gradient_max_abs_error"], float(errors.max()))
        SUMMARY["gradient_max_ratio"] = max(SUMMARY["gradient_max_ratio"], float((errors/tq(qnative, qreference)).max()))
        SUMMARY["gradient_max_l2_error"] = max(SUMMARY["gradient_max_l2_error"], float(np.linalg.norm(errors)))
        return native, qnative

    def test_adam_128_coordinates_independent_numpy_envelopes(self):
        permutation = np.random.default_rng(710007).permutation(8)
        for t in (101, 500, 1000, 2000):
            for scale in (1e-20, 1e-6, 1., 1e6):
                with self.subTest(t=t, scale=scale):
                    theta = np.array([0., .1, -.2, 1., -.3, 0., .01, -.01], np.float32)[permutation]
                    a = np.array([1., -1., .001, 0., -.027, 1e-10, 0., .02], np.float32)[permutation]*np.float32(scale)
                    m = np.array([.1, -.1, 0., .05, .003, 0., 0., -.01], np.float32)[permutation]*np.float32(scale)
                    v = np.array([.02, .02, 1e-6, .003, .001, 0., 0., .001], np.float32)[permutation]*np.float32(scale*scale)
                    parameter = torch.nn.Parameter(torch.from_numpy(theta.copy()))
                    optimizer = torch.optim.AdamW([parameter], lr=.001, betas=(.9, .999), eps=1e-8,
                                                 weight_decay=.01, foreach=False, fused=False)
                    optimizer.state[parameter] = {"step": torch.tensor(float(t-1)),
                        "exp_avg": torch.from_numpy(m.copy()), "exp_avg_sq": torch.from_numpy(v.copy())}
                    parameter.grad = torch.from_numpy(a.copy())
                    expected, envelopes = adam_reference(theta, a, m, v, t)
                    optimizer.step()
                    state = optimizer.state[parameter]
                    actual = [state["exp_avg"].numpy().copy(), state["exp_avg_sq"].numpy().copy(),
                              parameter.detach().numpy().copy()]
                    self.assertEqual(int(state["step"]), t)
                    for name, observed, wanted, envelope in zip(("moment", "variance", "endpoint"), actual, expected, envelopes):
                        ratio = np.abs(observed.astype(np.float64)-wanted)/envelope
                        self.assertTrue(np.isfinite(ratio).all())
                        self.assertTrue(np.all(ratio <= 1.), (name, ratio))
                        SUMMARY["adam_max_ratios"][name] = max(SUMMARY["adam_max_ratios"][name], float(ratio.max()))
                    self.assertTrue(np.all(actual[1] >= 0))
                    SUMMARY["adam_coordinates"] += len(theta)
        self.assertEqual(SUMMARY["adam_coordinates"], 128)

    def test_adam_envelope_rejects_material_endpoint_perturbation(self):
        expected, envelopes = adam_reference([.1], [.2], [.03], [.04], 101)
        deliberately_wrong = expected[2]+4*envelopes[2]
        self.assertTrue(np.all(np.abs(deliberately_wrong-expected[2]) > envelopes[2]))

    def test_known_log_two_and_exact_relu_zero_gradient(self):
        params = [np.zeros((2, 2), np.float32), np.zeros(2, np.float32),
                  np.zeros((2, 2), np.float32), np.zeros(2, np.float32)]
        loss, gradient = self.compare_mlp(params, np.ones((4, 2), np.float32), np.zeros(4, np.int64), 2)
        self.assertAlmostEqual(loss, math.log(2), places=15)
        expected = np.zeros(12)
        expected[-2:] = [-.5, .5]
        np.testing.assert_array_equal(gradient, expected)

    def test_duplicate_inputs_and_two_500_example_chunks(self):
        rng = np.random.default_rng(710008)
        params = [rng.normal(0, .3, shape).astype(np.float32) for shape in ((4, 3), (4,), (2, 4), (2,))]
        base = rng.uniform(0, 1, (7, 3)).astype(np.float32)
        labels = np.array([0, 1, 1, 0, 1, 0, 1], np.int64)
        indices = rng.integers(0, 7, size=1000)
        x, y = base[indices], labels[indices]
        loss, gradient = self.compare_mlp(params, x, y, 500)
        one_loss, one_gradient = self.compare_mlp(params, x, y, 1000)
        self.assertLessEqual(abs(loss-one_loss), float(tl(loss, one_loss)))
        self.assertTrue(np.all(np.abs(gradient-one_gradient) <= tq(gradient, one_gradient)))
        deduplicated_loss, _ = numpy_mlp(params, base, labels, 7)
        self.assertGreater(abs(loss-deduplicated_loss), 1e-6)

    def test_relu_zero_and_near_zero_native_input_values(self):
        params = [np.array([[1.], [-1.], [0.]], np.float32), np.array([0., 0., 2.**-40], np.float32),
                  np.array([[.5, -.25, .75], [-.5, .25, -.75]], np.float32), np.zeros(2, np.float32)]
        x = np.array([[0.], [2.**-20], [-2.**-20], [0.]], np.float32)
        self.compare_mlp(params, x, np.array([0, 1, 1, 0]), 2)

    def test_stable_logsumexp_with_extreme_finite_logits(self):
        params = [np.array([[1.]], np.float32), np.zeros(1, np.float32),
                  np.zeros((2, 1), np.float32), np.array([700., -700.], np.float32)]
        loss, _ = self.compare_mlp(params, np.ones((2, 1), np.float32), np.array([0, 1]), 2)
        self.assertEqual(loss, 700.)

    def test_tl_tq_reject_outside_ceiling_errors(self):
        self.assertGreater(abs(1.-(1.+4*tl(1., 1.))), tl(1., 1.+4*tl(1., 1.)))
        self.assertGreater(abs(.1-(.1+4*tq(.1, .1))), tq(.1, .1+4*tq(.1, .1)))

    def test_dot_error_propagation_with_shared_gradient(self):
        rng = np.random.default_rng(710009)
        q = rng.normal(size=P)*1e-3
        q_other = q+.25*tq(q, q)*rng.choice([-1., 1.], size=P)
        delta = rng.normal(size=P)*1e-4
        first = float(torch.dot(torch.from_numpy(q), torch.from_numpy(delta)))
        second = math.fsum(float(a)*float(b) for a, b in zip(q_other, delta))
        rounding = 2*gamma(2*P+4, U64)*max(np.abs(q*delta).sum(), np.abs(q_other*delta).sum())+4*P*H64
        bound = float(np.sum(tq(q, q_other)*np.abs(delta)))+rounding
        self.assertLessEqual(abs(first-second), bound)
        SUMMARY["dot_error"] = abs(first-second)
        SUMMARY["dot_error_bound"] = bound

    def test_shared_baseline_pair_and_nonzero_interaction(self):
        # Ordering: current, lagged, restored, reciprocal; exact dyadic CE values.
        after = np.array([1.75, 1.875, 1.625, 1.6875])
        baseline, interaction = 2., np.array([-1., -1., 1., 1.])
        y = after-baseline
        self.assertEqual(float(interaction@after), -.3125)
        self.assertEqual(float(interaction@y), -.3125)
        self.assertEqual((after[2]-after[1])-(after[0]-after[3]), -.3125)
        self.assertEqual((after[2]-after[0])-(after[1]-after[3]), -.3125)
        other = after+tl(after, after)*np.array([.25, -.5, .75, -.25])
        other_b = baseline+.8*float(tl(baseline, baseline))
        for coefficients in (np.array([-1., 1., 0., 0.]), interaction):
            selected = coefficients != 0
            ceiling = float(np.sum(np.abs(coefficients)*tl(after, other)))
            rounding = max(rsum(coefficients[selected]*after[selected]), rsum(coefficients[selected]*other[selected]))
            discrepancy = abs(float(coefficients@after-coefficients@other))
            self.assertLessEqual(discrepancy, ceiling+rounding)
            achieved = float(np.sum(np.abs(coefficients)*np.abs(after-other)))+rounding
            self.assertLessEqual(discrepancy, achieved)
            naive = ceiling+float(np.abs(coefficients).sum())*float(tl(baseline, other_b))
            self.assertGreater(naive, ceiling)  # The common baseline must not enter TC/TI.
        for a, other_a in zip(after, other):
            bound = float(tl(a, other_a)+tl(baseline, other_b))+max(rsum([a, -baseline]), rsum([other_a, -other_b]))
            self.assertLessEqual(abs((a-baseline)-(other_a-other_b)), bound)

    def test_direct_after_contrast_avoids_large_baseline_cancellation(self):
        # Deliberately extreme scalar arithmetic fixture, not plausible study CE.
        after = np.array([2., 2.+1e-8, 2.+4e-8, 2.+2e-8])
        y = after-1e16
        self.assertGreater(after[1]-after[0], 0.)
        self.assertEqual(y[1]-y[0], 0.)
        self.assertGreater(after[2]-after[1]-after[0]+after[3], 0.)
        self.assertEqual(y[2]-y[1]-y[0]+y[3], 0.)


if __name__ == "__main__":
    unittest.main(verbosity=2)
