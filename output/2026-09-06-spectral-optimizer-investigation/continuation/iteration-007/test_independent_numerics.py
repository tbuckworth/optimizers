"""Dataset-free CPU checks; expected formulae never use a producer module."""

import json
import math
import os
import unittest

if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
    raise RuntimeError("Set CUDA_VISIBLE_DEVICES='' explicitly")
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(key, "1")

import numpy as np
import torch

import independent_numerics as audit


class IndependentNumericsTests(unittest.TestCase):
    maxima = dict(projection_error=0., projection_ratio=0., ce_error=0., gradient_error=0.,
                  adam_moment_ratio=0., adam_variance_ratio=0., adam_theta_ratio=0.)

    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        if torch.cuda.is_initialized():
            raise RuntimeError("CUDA context already initialized")

    @classmethod
    def tearDownClass(cls):
        print("INDEPENDENT_NUMERICS " + json.dumps(dict(cls.maxima, cuda_initialized=torch.cuda.is_initialized()),
                                                 sort_keys=True, allow_nan=False))

    def projection_check(self, raw, basis):
        B, g = torch.from_numpy(basis.copy()), torch.from_numpy(raw.copy())
        native = (B@(B.T@g)).numpy().copy()
        result = audit.audit_projection(raw, basis, native)
        self.assertTrue(result["passed"], result)
        self.maxima["projection_error"] = max(self.maxima["projection_error"], result["max_absolute_error"])
        self.maxima["projection_ratio"] = max(self.maxima["projection_ratio"], result["max_error_envelope_ratio"])
        return native, result

    def test_projection_rank_one_near_cancellation(self):
        rng = np.random.default_rng(710071)
        B = rng.normal(size=(129, 1))
        B /= np.linalg.norm(B)
        raw = rng.normal(size=129)
        raw -= B[:, 0]*(B[:, 0]@raw)
        B, raw = B.astype(np.float32), raw.astype(np.float32)
        native, result = self.projection_check(raw, B)
        reference = audit.projection_reference(raw, B)
        self.assertLess(np.linalg.norm(reference["candidate"]), 1e-6*np.linalg.norm(raw))
        self.assertLessEqual(result["max_absolute_error"], result["max_envelope"])
        self.assertGreater(result["max_absolute_error"], 0.)
        changed = native.copy()
        changed[0] = np.float32(reference["candidate"][0]+4*reference["envelope"][0])
        self.assertFalse(audit.audit_projection(raw, B, changed)["passed"])

    def test_projection_rank_32_small_and_study_parameter_count(self):
        rng = np.random.default_rng(710072)
        for p in (64, 50890):
            with self.subTest(p=p):
                B = np.linalg.qr(rng.normal(size=(p, 32)), mode="reduced")[0].astype(np.float32)
                raw = rng.normal(size=p).astype(np.float32)
                self.projection_check(raw, B)

    def test_projection_exact_algebra_and_fixed_envelope(self):
        raw, basis = np.array([2., -4.], np.float32), np.array([[1.], [-.5]], np.float32)
        reference = audit.projection_reference(raw, basis)
        np.testing.assert_array_equal(reference["candidate"], [4., -2.])
        u = 2.**-24
        ez = (4*u/(1-4*u))*4.
        efirst = ez + (2*u/(1-2*u))*(4.+ez)
        floor = 32*4*2.**-126
        np.testing.assert_allclose(reference["envelope"], [efirst+floor, .5*efirst+floor], rtol=1e-15, atol=0)
        result = audit.audit_projection(raw, basis, np.array([4., -2.], np.float32))
        self.assertEqual(result["identity_mismatch_flat_indices"], [])

    def test_projection_subnormal_and_zero_inputs(self):
        tiny = np.nextafter(np.float32(0), np.float32(1))
        B = np.array([[.5], [-.5], [.5], [-.5]], np.float32)
        for raw in (np.array([tiny, -tiny, tiny, -tiny]), np.zeros(4, np.float32)):
            self.projection_check(raw, B)
        self.assertGreater(float(tiny), 0.)

    def test_missing_basis_identity_preserves_signed_zero_and_ownership(self):
        raw = np.array([0., -0., 1., -1.], np.float32)
        result = audit.projection_reference(raw, None)
        self.assertTrue(result["candidate"].flags.owndata)
        self.assertTrue(np.signbit(result["candidate"][1]))
        self.assertTrue(np.all(result["envelope"] == 0))
        self.assertTrue(audit.audit_projection(raw, None, raw.copy())["passed"])
        bad = raw.copy()
        bad[1] = 0.
        failure = audit.audit_projection(raw, None, bad)
        self.assertFalse(failure["passed"])
        self.assertEqual(failure["identity_mismatch_flat_indices"], [1])
        result["candidate"][:] = 5
        self.assertEqual(raw[2], 1.)

    def test_projection_rejects_bad_types_shapes_and_nonfinite(self):
        raw, B = np.ones(4, np.float32), np.ones((4, 1), np.float32)
        for badraw, badbasis in ((raw.astype(np.float64), B), (raw, B.astype(np.float64)),
                                  (raw, np.ones((3, 1), np.float32)), (raw, np.empty((4, 0), np.float32)),
                                  (raw, np.full((4, 1), np.nan, np.float32))):
            with self.assertRaises(ValueError):
                audit.projection_reference(badraw, badbasis)
        with self.assertRaises(ValueError):
            audit.projection_reference(np.ones(40, np.float32), np.ones((40, 33), np.float32))
        with self.assertRaises(ValueError):
            audit.audit_projection(raw, B, np.full(4, np.inf, np.float32))

    def test_adam_saved_native_moments_all_128_coordinates(self):
        order = np.random.default_rng(710007).permutation(8)
        count = 0
        for t in (101, 500, 1000, 2000):
            for scale in (1e-20, 1e-6, 1., 1e6):
                theta = np.array([0., .1, -.2, 1., -.3, 0., .01, -.01], np.float32)[order]
                a = np.array([1., -1., .001, 0., -.027, 1e-10, 0., .02], np.float32)[order]*np.float32(scale)
                m = np.array([.1, -.1, 0., .05, .003, 0., 0., -.01], np.float32)[order]*np.float32(scale)
                v = np.array([.02, .02, 1e-6, .003, .001, 0., 0., .001], np.float32)[order]*np.float32(scale*scale)
                parameter = torch.nn.Parameter(torch.from_numpy(theta.copy()))
                constructor = dict(audit.ADAMW_OPTIONS)
                self.assertIs(constructor.pop("decoupled_weight_decay"), True)
                optimizer = torch.optim.AdamW([parameter], **constructor)
                self.assertIs(optimizer.param_groups[0]["decoupled_weight_decay"], True)
                optimizer.state[parameter] = dict(step=torch.tensor(float(t-1)), exp_avg=torch.from_numpy(m.copy()),
                                                  exp_avg_sq=torch.from_numpy(v.copy()))
                parameter.grad = torch.from_numpy(a.copy())
                optimizer.step()
                state = optimizer.state[parameter]
                saved = dict(theta=parameter.detach().numpy().copy(), moment=state["exp_avg"].numpy().copy(),
                             variance=state["exp_avg_sq"].numpy().copy(), next_step=int(state["step"]))
                result = audit.audit_adamw(theta, a, m, v, t, saved, options=dict(audit.ADAMW_OPTIONS))
                self.assertTrue(result["passed"], result)
                for name in ("moment", "variance", "theta"):
                    self.maxima["adam_"+name+"_ratio"] = max(self.maxima["adam_"+name+"_ratio"],
                                                          result["checks"][name]["max_error_envelope_ratio"])
                count += len(theta)
        self.assertEqual(count, 128)

    def test_adam_explicit_reference_formula_and_owned_arrays(self):
        theta, a, m, v = [np.array(x, np.float32) for x in ([.25, -.5], [.2, -.1], [.03, .04], [.05, .06])]
        result = audit.adamw_reference(theta, a, m, v, 500, options=dict(audit.ADAMW_OPTIONS))
        mm = np.array([.9*float(old)+.1*float(g) for old, g in zip(m, a)])
        vv = np.array([.999*float(old)+.001*float(g)**2 for old, g in zip(v, a)])
        expected = np.array([float(p)*(1-.001*.01)-(.001/(1-.9**500))*first/
                             (math.sqrt(second)/math.sqrt(1-.999**500)+1e-8)
                             for p, first, second in zip(theta, mm, vv)])
        np.testing.assert_array_equal(result["moment"], mm)
        np.testing.assert_allclose(result["variance"], vv, rtol=1e-15, atol=0)
        np.testing.assert_allclose(result["theta"], expected, rtol=1e-15, atol=0)
        for value in result.values():
            self.assertTrue(value.flags.owndata)
        result["theta"][:] = 0
        self.assertEqual(theta[0], .25)

    def test_adam_invalid_moment_counter_options_and_corrupted_endpoint(self):
        z = np.zeros(2, np.float32)
        for m, v, t, options in ((z, -np.ones(2, np.float32), 101, dict(audit.ADAMW_OPTIONS)),
                                (np.full(2, np.nan, np.float32), z, 101, dict(audit.ADAMW_OPTIONS)),
                                (z, z, True, dict(audit.ADAMW_OPTIONS)),
                                (z, z, 101, dict(audit.ADAMW_OPTIONS, foreach=None)),
                                (z, z, 101, dict(audit.ADAMW_OPTIONS, decoupled_weight_decay=False))):
            with self.assertRaises(ValueError):
                audit.adamw_reference(z, z, m, v, t, options=options)
        saved = dict(theta=np.ones(2, np.float32), moment=z.copy(), variance=z.copy(), next_step=101)
        self.assertFalse(audit.audit_adamw(z, z, z, z, 101, saved, options=dict(audit.ADAMW_OPTIONS))["passed"])
        saved["next_step"] = 100
        with self.assertRaises(ValueError):
            audit.audit_adamw(z, z, z, z, 101, saved, options=dict(audit.ADAMW_OPTIONS))

    def torch_loss(self, parameters, x, y, chunk=500, dtype=torch.float64):
        params = [torch.tensor(a, dtype=dtype, requires_grad=True) for a in parameters]
        total, grads = 0., [torch.zeros_like(a) for a in params]
        for start in range(0, len(x), chunk):
            xx, yy = torch.tensor(x[start:start+chunk], dtype=dtype), torch.tensor(y[start:start+chunk])
            w1, b1, w2, b2 = params
            loss = torch.nn.functional.cross_entropy(torch.relu(xx@w1.T+b1)@w2.T+b2, yy, reduction="sum")
            total += float(loss.detach())
            for destination, source in zip(grads, torch.autograd.grad(loss, params)):
                destination.add_(source)
        return total/len(x), torch.cat([a.ravel() for a in grads]).numpy()/len(x)

    def test_mlp_duplicate_500_chunks_native_promotion_and_gradient(self):
        rng = np.random.default_rng(710073)
        params = [rng.normal(0, .3, shape).astype(np.float32) for shape in ((4, 3), (4,), (2, 4), (2,))]
        base, indices = rng.uniform(0, 1, (7, 3)).astype(np.float32), rng.integers(0, 7, 1003)
        labels = np.array([0, 1, 1, 0, 1, 0, 1], np.int64)
        x, y = base[indices], labels[indices]
        result = audit.mlp_ce_gradient(params, x, y)
        loss, gradient = self.torch_loss(params, x, y)
        self.assertEqual(result["chunk_counts"], [500, 500, 3])
        self.assertLessEqual(abs(loss-result["mean_ce"]), audit.loss_tolerance(loss, result["mean_ce"]))
        errors = np.abs(gradient-result["gradient"])
        self.assertTrue(np.all(errors <= audit.gradient_tolerance(gradient, result["gradient"])))
        self.maxima["ce_error"] = float(abs(loss-result["mean_ce"]))
        self.maxima["gradient_error"] = float(errors.max())
        self.assertNotAlmostEqual(result["mean_ce"], audit.mlp_ce_gradient(params, base, labels)["mean_ce"], places=6)
        self.assertTrue(result["gradient"].flags.owndata)
        result["parameter_gradients"][0][:] = 123
        self.assertFalse(np.all(result["gradient"][:12] == 123))
        native, _ = self.torch_loss(params, x, y, dtype=torch.float32)
        self.assertFalse(audit.native_ce_concordance(native, result["mean_ce"])["discordant"])

    def test_known_log_two_relu_zero_and_near_zero(self):
        params = [np.zeros((2, 1), np.float32), np.zeros(2, np.float32),
                  np.zeros((2, 2), np.float32), np.zeros(2, np.float32)]
        result = audit.mlp_ce_gradient(params, np.ones((4, 1), np.float32), np.zeros(4, np.int64), 2)
        self.assertEqual(result["mean_ce"], math.log(2))
        expected = np.zeros(10)
        expected[-2:] = [-.5, .5]
        np.testing.assert_array_equal(result["gradient"], expected)
        params[0][:, 0], params[1][:] = [1, -1], [0, 2.**-40]
        params[2][:] = [[.5, -.25], [-.5, .25]]
        x, y = np.array([[0.], [2.**-20], [-2.**-20], [0.]], np.float32), np.array([0, 1, 1, 0], np.int64)
        result = audit.mlp_ce_gradient(params, x, y, 2)
        loss, gradient = self.torch_loss(params, x, y, 2)
        self.assertLessEqual(abs(loss-result["mean_ce"]), audit.loss_tolerance(loss, result["mean_ce"]))
        self.assertTrue(np.all(np.abs(gradient-result["gradient"]) <= audit.gradient_tolerance(gradient, result["gradient"])))

    def test_mlp_invalid_inputs(self):
        params = [np.zeros((1, 1), np.float32), np.zeros(1, np.float32),
                  np.zeros((2, 1), np.float32), np.zeros(2, np.float32)]
        x, y = np.ones((2, 1), np.float32), np.array([0, 1], np.int64)
        for xx, yy, chunk in ((x.astype(np.float64), y, 500), (x, y.astype(np.int32), 500),
                              (x, np.array([0, 2], np.int64), 500), (x, y, 0),
                              (np.full_like(x, np.nan), y, 500)):
            with self.assertRaises(ValueError):
                audit.mlp_ce_gradient(params, xx, yy, chunk)

    def test_native_concordance_finite_disagreement_is_not_failure(self):
        self.assertTrue(audit.native_ce_concordance(1.01, 1.)["discordant"])
        self.assertFalse(audit.native_ce_concordance(1.+1e-6, 1.)["discordant"])
        for value in (float("nan"), float("inf"), -float("inf")):
            with self.assertRaises(ValueError):
                audit.native_ce_concordance(value, 1.)

    def test_frozen_scalar_constants_and_conservative_signs(self):
        self.assertEqual(audit.loss_tolerance(1., 2.), 1e-10+2e-11)
        np.testing.assert_array_equal(audit.gradient_tolerance([0., -2.], [0., 1.]), [1e-12, 1e-12+2e-9])
        terms = np.array([1., -2., 4.])
        g = (10*2.**-53)/(1-10*2.**-53)
        self.assertEqual(audit.sum_roundoff(terms), 2*g*7+12*2.**-1022)
        self.assertEqual(audit.dot_roundoff(terms, terms), 2*g*21+12*2.**-1022)
        self.assertEqual(audit.numerical_sign(-2., -3., 1.)["status"], "resolved_negative")
        for a, b, ceiling in ((1., 2., 1.), (0., 0., 0.), (2., -2., 0.), (1e-12, 1e-12, 1e-10)):
            self.assertEqual(audit.numerical_sign(a, b, ceiling)["status"], "unresolved")
        with self.assertRaises(ValueError):
            audit.numerical_sign(None, 1., 0.)
        for unsupported in (10**400, float("inf"), float("nan"), True):
            with self.assertRaises(ValueError):
                audit.loss_tolerance(unsupported, 1.)
            with self.assertRaises(ValueError):
                audit.native_ce_concordance(unsupported, 1.)

    def test_shared_baseline_factorial_error_propagation(self):
        after = np.array([1.75, 1.875, 1.625, 1.6875])  # current,lagged,restored,reciprocal
        coefficients = np.array([-1., -1., 1., 1.])
        self.assertEqual(float(coefficients@after), -.3125)
        baseline = 2.
        self.assertEqual(float(coefficients@(after-baseline)), -.3125)
        other = after+np.array([1e-11, -2e-11, 3e-11, -1e-11])
        bound = sum(abs(k)*audit.loss_tolerance(a, b) for k, a, b in zip(coefficients, after, other))
        bound += max(audit.sum_roundoff(coefficients*after), audit.sum_roundoff(coefficients*other))
        self.assertLessEqual(abs(float(coefficients@after-coefficients@other)), bound)
        self.assertEqual(audit.numerical_sign(float(coefficients@after), float(coefficients@other), bound)["status"],
                         "resolved_negative")


if __name__ == "__main__":
    unittest.main(verbosity=2)
