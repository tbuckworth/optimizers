"""Synthetic CPU tests only: no datasets, model fitting or CUDA initialization.

Run with CUDA_VISIBLE_DEVICES='' python3 -m unittest test_policy -v.
"""
import copy
import json
import os
import unittest
from unittest import mock

import policy_math as p

torch, nn = p.torch, p.torch.nn


class RotatingTracker:
    """Adversarial observer: overwrites a same-sized stored basis in place."""

    def __init__(self, previous, following, step=100):
        self.rank, self.warmup, self.step_count = p.WIDTH, p.WARMUP, step
        self.V = None if previous is None else previous.clone()
        self.following = following
        self.observed = []

    def _update_svd(self, raw):
        self.observed.append(raw.detach().clone())
        if self.following is None:
            self.V = None
        elif self.V is not None and self.V.shape == self.following.shape:
            self.V.copy_(self.following)
        else:
            self.V = self.following.clone()


def tracker_and_model(dimension=40):
    model = nn.Linear(dimension, 1, bias=False)
    optimizer = p.h.make_optimizer(model)
    return p.h.make_tracker(model, optimizer, p.WIDTH), model, optimizer


class PolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
            raise RuntimeError("Synthetic tests require CUDA_VISIBLE_DEVICES=''")
        p.h.configure()
        if torch.cuda.is_initialized():
            raise RuntimeError("CPU tests must not have a CUDA context")

    def setUp(self):
        self.e1 = torch.tensor([[1.], [0.]])
        self.e2 = torch.tensor([[0.], [1.]])

    def test_six_policy_formulae_actual_grad_and_nonaliasing(self):
        raw = torch.tensor([3., 4.])
        expected = {
            "adamw": [3., 4.], "current32": [0., 4.], "lagged32": [3., 0.],
            "lagged32_current_norm": [4., 0.],
            "scalar_current32": [2.4, 3.2], "scalar_lagged32": [1.8, 2.4],
        }
        for arm in p.ARMS:
            with self.subTest(arm=arm):
                tracker = None if arm == "adamw" else RotatingTracker(self.e1, self.e2)
                applied, current, lagged, meta = p.transform_gradient(arm, tracker, raw)
                self.assertTrue(torch.allclose(applied, torch.tensor(expected[arm]), atol=0, rtol=1e-6))
                self.assertNotEqual(raw.data_ptr(), applied.data_ptr())
                model = nn.Linear(1, 1)
                p.h.set_grad(model, applied)
                row = p.delivery_metrics(raw, current, lagged, p.h.flat_grad(model), meta)
                self.assertEqual(set(row), set(p.METRIC_KEYS) | {"null_reasons"})
                self.assertEqual(set(row["null_reasons"]), {k for k in p.METRIC_KEYS if row[k] is None})
                json.dumps({"policy": meta, "metrics": row}, allow_nan=False)
                p.h.set_grad(model, -applied)
                with self.assertRaisesRegex(AssertionError, "direction error"):
                    p.delivery_metrics(raw, current, lagged, p.h.flat_grad(model), meta)
                if tracker is not None:
                    self.assertEqual(tracker.step_count, 101)
                    self.assertEqual(len(tracker.observed), 1)
                    self.assertTrue(torch.equal(tracker.observed[0], raw))
                    self.assertTrue(torch.equal(current, torch.tensor([0., 4.])))
                    self.assertTrue(torch.equal(lagged, torch.tensor([3., 0.])))
                    tracker.V.zero_()
                    self.assertTrue(torch.equal(lagged, torch.tensor([3., 0.])))
                    self.assertTrue(torch.equal(current, torch.tensor([0., 4.])))
        self.assertTrue(torch.equal(raw, torch.tensor([3., 4.])))

    def test_scalar_control_is_not_projected_into_candidate_span(self):
        raw = torch.tensor([3., 4.])
        applied, current, lagged, meta = p.transform_gradient(
            "scalar_current32", RotatingTracker(self.e1, self.e1), raw)
        self.assertEqual(meta["delivery_operator"], "scalar_identity")
        self.assertGreater(float(applied[1]), 0.)
        self.assertEqual(float(current[1]), 0.)
        row = p.delivery_metrics(raw, current, lagged, applied, meta)
        self.assertAlmostEqual(row["raw_applied_cosine"], 1., places=12)
        self.assertAlmostEqual(row["applied_norm"], row["current_norm"], places=6)
        self.assertNotIn("leakage_squared_fraction", row)

    def test_step100_then101_uses_real_stored_basis_not_warmup_return(self):
        tracker = RotatingTracker(self.e1, self.e2, step=99)
        raw = torch.tensor([3., 4.])
        applied, current, lagged, meta = p.transform_gradient("lagged32", tracker, raw)
        self.assertTrue(torch.equal(applied, raw))
        self.assertIsNone(current)
        self.assertIsNone(lagged)
        self.assertFalse(meta["policy_active"])
        self.assertEqual(meta["active_policy"], "identity_warmup")
        self.assertEqual(meta["current_basis_rank"], 1)
        tracker.following = self.e1
        applied, current, lagged, meta = p.transform_gradient("lagged32", tracker, raw)
        self.assertTrue(torch.equal(applied, torch.tensor([0., 4.])))
        self.assertTrue(torch.equal(current, torch.tensor([3., 0.])))
        self.assertEqual(meta["observing_step"], 101)
        self.assertEqual(len(tracker.observed), 2)

    def test_canonical_current_equivalence_growth_warmup_and_repairs(self):
        actual, actual_model, _ = tracker_and_model()
        helper, _, _ = tracker_and_model()
        canonical, canonical_model, _ = tracker_and_model()
        stream = torch.randn(220, 40, generator=torch.Generator().manual_seed(6001))
        checkpoints = set()
        for step, raw in enumerate(stream, 1):
            applied, current, lagged, meta = p.transform_gradient("current32", actual, raw)
            reference, _ = p.h.observe_and_project(helper, raw)
            p.h.set_grad(canonical_model, raw.clone())
            canonical.filter_grad()
            self.assertTrue(torch.equal(applied, reference), f"helper mismatch step {step}")
            self.assertTrue(torch.equal(applied, p.h.flat_grad(canonical_model)), f"canonical mismatch step {step}")
            self.assertTrue(p.h.equal_tree(p.h.tracker_state(actual), p.h.tracker_state(helper)))
            self.assertTrue(p.h.equal_tree(p.h.tracker_state(actual), p.h.tracker_state(canonical)))
            self.assertEqual(actual.step_count, step)
            p.h.set_grad(actual_model, applied)
            p.delivery_metrics(raw, current, lagged, p.h.flat_grad(actual_model), meta)
            if step in (100, 101, 199, 200, 201):
                checkpoints.add(step)
                self.assertEqual(meta["policy_active"], step > 100)
        self.assertEqual(checkpoints, {100, 101, 199, 200, 201})
        self.assertGreaterEqual(actual.stabilization_count, 2)
        self.assertEqual(actual.V.shape[1], 32)

    def test_common_raw_warmup_parameters_optimizer_and_observers(self):
        stream = torch.randn(100, 40, generator=torch.Generator().manual_seed(6002))
        results = {}
        for arm in p.ARMS:
            tracker, model, optimizer = tracker_and_model()
            model.weight.data.fill_(.2)
            if arm == "adamw":
                tracker = None
            params, observers = [], []
            for raw in stream:
                applied, current, lagged, meta = p.transform_gradient(arm, tracker, raw)
                self.assertTrue(torch.equal(applied, raw))
                self.assertIsNone(current)
                self.assertIsNone(lagged)
                p.h.set_grad(model, applied)
                row = p.delivery_metrics(raw, current, lagged, p.h.flat_grad(model), meta)
                self.assertIsNone(row["current_norm"])
                optimizer.step()
                params.append(p.h.flat_params(model).clone())
                if tracker is not None:
                    observers.append(p.h.tracker_state(tracker))
            results[arm] = (params, copy.deepcopy(optimizer.state_dict()), observers)
        baseline = results["adamw"]
        for arm in p.ARMS[1:]:
            self.assertTrue(p.h.equal_tree(results[arm][0], baseline[0]))
            self.assertTrue(p.h.equal_tree(results[arm][1], baseline[1]))
            self.assertTrue(p.h.equal_tree(results[arm][2], results["current32"][2]))

    def test_delayed_initialization_missing_bases_are_identity(self):
        raw = torch.tensor([3., 4.])
        for old, new in ((None, None), (None, self.e1), (self.e1, None)):
            with self.subTest(previous=old is None, current=new is None):
                applied, current, lagged, meta = p.transform_gradient(
                    "lagged32", RotatingTracker(old, new), raw)
                self.assertEqual(meta["current_basis_missing"], new is None)
                self.assertEqual(meta["lagged_basis_missing"], old is None)
                self.assertTrue(torch.equal(current, p.h.project(raw, new)))
                self.assertTrue(torch.equal(lagged, p.h.project(raw, old)))
                p.delivery_metrics(raw, current, lagged, applied, meta)

    def test_canonical_allzero_stream_remains_identity_fallback(self):
        tracker, _, _ = tracker_and_model(2)
        for _ in range(101):
            applied, current, lagged, meta = p.transform_gradient("scalar_current32", tracker, torch.zeros(2))
        self.assertIsNone(tracker.V)
        self.assertTrue(meta["policy_active"])
        self.assertEqual(meta["current_candidate_operator"], "identity_missing_basis")
        row = p.delivery_metrics(torch.zeros(2), current, lagged, applied, meta)
        self.assertEqual(meta["scale"], 0.)
        self.assertIsNone(row["current_energy_retention"])
        self.assertEqual(row["null_reasons"]["current_energy_retention"], "zero_raw_norm")
        self.assertIsNone(row["direction_relative_error"])

    def test_zero_targets_and_undefined_restored_direction(self):
        for arm in p.ARMS[1:]:
            applied, current, lagged, meta = p.transform_gradient(
                arm, RotatingTracker(self.e1, self.e2), torch.zeros(2))
            self.assertTrue(torch.equal(applied, torch.zeros(2)))
            p.delivery_metrics(torch.zeros(2), current, lagged, applied, meta)
        raw = torch.tensor([1., 0.])
        applied, current, lagged, meta = p.transform_gradient(
            "lagged32_current_norm", RotatingTracker(self.e1, self.e2), raw)
        self.assertTrue(torch.equal(applied, torch.zeros(2)))
        self.assertEqual(meta["scale"], 0.)
        with self.assertRaisesRegex(AssertionError, "Positive target with zero direction"):
            p.transform_gradient("lagged32_current_norm", RotatingTracker(self.e2, self.e1), raw)

    def test_zero_gradient_still_allows_adamw_momentum_and_decay(self):
        model = nn.Linear(1, 1, bias=False)
        optimizer = p.h.make_optimizer(model)
        p.h.set_grad(model, torch.ones(1))
        optimizer.step()
        before = p.h.flat_params(model).clone()
        applied, current, lagged, meta = p.transform_gradient("adamw", None, torch.zeros(1))
        p.h.set_grad(model, applied)
        p.delivery_metrics(torch.zeros(1), current, lagged, p.h.flat_grad(model), meta)
        optimizer.step()
        self.assertFalse(torch.equal(before, p.h.flat_params(model)))
        self.assertEqual(int(optimizer.state[model.weight]["step"]), 2)

    def test_nonfinite_raw_and_bases_rejected_before_zero_shortcuts(self):
        for invalid in (float("nan"), float("inf"), -float("inf")):
            tracker = RotatingTracker(self.e1, self.e2)
            with self.assertRaisesRegex(AssertionError, "raw: nonfinite"):
                p.transform_gradient("current32", tracker, torch.tensor([invalid, 0.]))
            self.assertEqual(tracker.step_count, 100)
            bad = torch.tensor([[invalid], [0.]])
            for old, new in ((bad, self.e1), (self.e1, bad)):
                with self.assertRaisesRegex(AssertionError, "basis: nonfinite"):
                    p.transform_gradient("scalar_current32", RotatingTracker(old, new), torch.zeros(2))

    def test_scalar_approximate_contraction_and_uncapped_restoration(self):
        raw = torch.tensor([1., 0.])
        for factor, passes in ((1.002, True), (1.006, False)):
            basis = self.e1 * factor ** .5
            if passes:
                _, _, _, meta = p.transform_gradient("scalar_current32", RotatingTracker(self.e1, basis), raw)
                self.assertGreater(meta["scale"], 1.)
                self.assertAlmostEqual(meta["scale"], factor, places=6)
            else:
                with self.assertRaisesRegex(AssertionError, "contraction"):
                    p.transform_gradient("scalar_current32", RotatingTracker(self.e1, basis), raw)
        applied, _, _, meta = p.transform_gradient(
            "lagged32_current_norm", RotatingTracker(self.e2, self.e1), torch.tensor([4., 1.]))
        self.assertEqual(meta["scale"], 4.)
        self.assertTrue(torch.equal(applied, torch.tensor([0., 4.])))

    def test_huge_finite_scale_passes_without_premature_float32_overflow(self):
        raw = torch.tensor([1., 0.])
        previous = torch.tensor([[1e-40], [1.]])
        applied, current, lagged, meta = p.transform_gradient(
            "lagged32_current_norm", RotatingTracker(previous, self.e1), raw)
        self.assertGreater(meta["scale"], torch.finfo(torch.float32).max)
        self.assertGreater(p.h.norm(lagged), 0.)
        self.assertTrue(torch.equal(applied, torch.tensor([0., 1.])))
        row = p.delivery_metrics(raw, current, lagged, applied, meta)
        self.assertEqual(row["norm_matching_relative_error"], 0.)

    def test_subnormal_representable_target_passes(self):
        tiny = torch.nextafter(torch.tensor(0.), torch.tensor(1.))
        raw = torch.tensor([1., 0.])
        with mock.patch.object(p.h, "project", return_value=torch.stack((tiny, tiny * 0))):
            applied, current, lagged, meta = p.transform_gradient(
                "scalar_current32", RotatingTracker(self.e1, self.e1), raw)
        self.assertTrue(torch.equal(applied, torch.stack((tiny, tiny * 0))))
        p.delivery_metrics(raw, current, lagged, applied, meta)

    def test_subnormal_underflow_and_quantization_fail_without_absolute_floor(self):
        tiny = torch.nextafter(torch.tensor(0.), torch.tensor(1.))
        for dimension in (2, 9):
            raw = torch.ones(dimension)
            candidate = torch.zeros(dimension)
            candidate[0] = tiny
            basis = torch.eye(dimension)[:, :1]
            with mock.patch.object(p.h, "project", return_value=candidate):
                with self.assertRaisesRegex(AssertionError, "underflow|relative norm error"):
                    p.transform_gradient("scalar_current32", RotatingTracker(basis, basis), raw)

    def test_nonfinite_candidates_rejected_even_when_raw_zero(self):
        for bad in (torch.tensor([float("nan"), 0.]), torch.tensor([float("inf"), 0.])):
            with mock.patch.object(p.h, "project", return_value=bad):
                with self.assertRaisesRegex(AssertionError, "current: nonfinite"):
                    p.transform_gradient("scalar_current32", RotatingTracker(self.e1, self.e1), torch.zeros(2))

    def test_actual_delivery_norm_and_metadata_tampering_fail(self):
        raw = torch.tensor([3., 4.])
        applied, current, lagged, meta = p.transform_gradient(
            "scalar_current32", RotatingTracker(self.e1, self.e2), raw)
        for wrong in (applied * 2, torch.zeros(2), torch.tensor([float("nan"), 0.])):
            with self.assertRaises(AssertionError):
                p.delivery_metrics(raw, current, lagged, wrong, meta)
        for key, wrong in (("scale", 1.), ("target_norm", 1.), ("scale_direction", "lagged"),
                           ("observing_step", 100), ("delivery_operator", "native_current")):
            changed = dict(meta, **{key: wrong})
            with self.assertRaises(AssertionError):
                p.delivery_metrics(raw, current, lagged, applied, changed)

    def test_measurements_are_stateless_and_json_safe(self):
        tracker = RotatingTracker(self.e1, self.e2)
        raw = torch.tensor([3., 4.])
        applied, current, lagged, meta = p.transform_gradient("lagged32", tracker, raw)
        tensors = [raw, applied, current, lagged, tracker.V]
        saved, saved_meta = [x.clone() for x in tensors], copy.deepcopy(meta)
        rng = torch.get_rng_state().clone()
        row = p.delivery_metrics(raw, current, lagged, applied, meta)
        for original, before in zip(tensors, saved):
            self.assertTrue(torch.equal(original, before))
        self.assertEqual(meta, saved_meta)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        json.dumps(row, allow_nan=False)
        self.assertFalse(torch.cuda.is_initialized())

    def test_invalid_contract_and_once_only_count_are_rejected(self):
        raw = torch.ones(2)
        with self.assertRaises(AssertionError):
            p.transform_gradient("adamw", RotatingTracker(self.e1, self.e1), raw)
        with self.assertRaises(AssertionError):
            p.transform_gradient("current32", None, raw)
        with self.assertRaises(AssertionError):
            p.transform_gradient("unknown", None, raw)
        with self.assertRaises(AssertionError):
            p.transform_gradient("adamw", None, raw.double())
        tracker = RotatingTracker(self.e1, self.e1)
        original = tracker._update_svd

        def broken_update(value):
            tracker.step_count += 1
            original(value)

        tracker._update_svd = broken_update
        with self.assertRaisesRegex(AssertionError, "exactly once"):
            p.transform_gradient("current32", tracker, raw)


if __name__ == "__main__":
    unittest.main(verbosity=2)
