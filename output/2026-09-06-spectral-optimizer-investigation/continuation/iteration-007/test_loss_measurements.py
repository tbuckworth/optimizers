#!/usr/bin/env python3
"""Dataset-free CPU tests for iteration-007 functional loss measurements."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import importlib.util
import math
from pathlib import Path
import random
import unittest

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("iteration007_loss_measurements_tested",
                                              HERE / "loss_measurements.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def parameters(w1, b1, w2, b2):
    return {name: torch.tensor(value, dtype=torch.float32).contiguous().clone()
            for name, value in zip(m.PARAMETER_ORDER, (w1, b1, w2, b2))}


def tiny_zero():
    return parameters([[0., 0.], [0., 0.]], [0., 0.],
                      [[0., 0.], [0., 0.]], [0., 0.])


def bytes_of(value):
    return value.detach().cpu().contiguous().numpy().tobytes()


class LossMeasurementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
            raise RuntimeError("loss tests require CUDA_VISIBLE_DEVICES=''")
        if torch.cuda.is_initialized():
            raise RuntimeError("loss tests must not initialize CUDA")

    def test_known_log2_loss_uneven_chunks_and_gradient(self):
        params = tiny_zero()
        x = torch.tensor([[1., 2.], [3., 4.], [5., 6.]], dtype=torch.float32)
        labels = torch.tensor([0, 1, 1], dtype=torch.int64)
        result = m.evaluate(params, x, labels, chunk_size=2)
        self.assertEqual(tuple(result), m.RESULT_KEYS)
        self.assertEqual([(c["start"], c["end"], c["count"]) for c in result["chunks"]],
                         [(0, 2, 2), (2, 3, 1)])
        self.assertAlmostEqual(result["chunks"][0]["sum_cross_entropy"], 2 * math.log(2), places=14)
        self.assertAlmostEqual(result["chunks"][1]["sum_cross_entropy"], math.log(2), places=14)
        self.assertAlmostEqual(result["mean_cross_entropy"], math.log(2), places=14)
        expected = torch.tensor([0.] * 10 + [1 / 6, -1 / 6], dtype=torch.float64)
        self.assertTrue(torch.allclose(result["mean_gradient"], expected, atol=1e-15, rtol=0))
        self.assertEqual(result["mean_gradient"].dtype, torch.float64)
        self.assertIsNone(result["mean_gradient"]._base)

    def test_relu_exact_zero_and_near_positive_derivatives(self):
        params = parameters([[1.]], [0.], [[1.], [-1.]], [0., 0.])
        labels = torch.tensor([0], dtype=torch.int64)
        at_zero = m.evaluate(params, torch.tensor([[0.]], dtype=torch.float32), labels, chunk_size=1)
        self.assertTrue(torch.equal(at_zero["mean_gradient"][:4], torch.zeros(4, dtype=torch.float64)))
        near = m.evaluate(params, torch.tensor([[2 ** -20]], dtype=torch.float32), labels, chunk_size=1)
        self.assertNotEqual(float(near["mean_gradient"][0]), 0.)
        self.assertNotEqual(float(near["mean_gradient"][1]), 0.)
        self.assertLess(near["mean_cross_entropy"], at_zero["mean_cross_entropy"])

    def test_duplicate_examples_keep_order_and_multiplicity(self):
        params = parameters([[1., -.5]], [.25], [[1.], [-1.]], [.1, -.2])
        base_x = torch.tensor([[1., 2.], [-2., 1.]], dtype=torch.float32)
        base_y = torch.tensor([0, 1], dtype=torch.int64)
        dup_x = torch.stack((base_x[0], base_x[0], base_x[1])).contiguous()
        dup_y = torch.tensor([0, 0, 1], dtype=torch.int64)
        base = m.evaluate(params, base_x, base_y, chunk_size=1)
        duplicate = m.evaluate(params, dup_x, dup_y, chunk_size=2)
        p64 = [value.double().requires_grad_(True) for value in params.values()]
        logits = F.linear(F.relu(F.linear(dup_x.double(), p64[0], p64[1])), p64[2], p64[3])
        loss = F.cross_entropy(logits, dup_y, weight=None, reduction="sum", label_smoothing=0.0)
        grad = torch.cat([g.reshape(-1) for g in torch.autograd.grad(loss, p64)]) / 3
        self.assertAlmostEqual(duplicate["mean_cross_entropy"], float((loss / 3).detach()), places=14)
        self.assertTrue(torch.allclose(duplicate["mean_gradient"], grad, atol=1e-15, rtol=0))
        self.assertNotEqual(duplicate["mean_cross_entropy"], base["mean_cross_entropy"])

    def test_no_gradient_mode_has_exact_schema(self):
        result = m.evaluate(tiny_zero(), torch.zeros(4, 2), torch.tensor([0, 1, 0, 1]),
                            chunk_size=99, with_gradient=False)
        self.assertIsNone(result["mean_gradient"])
        self.assertEqual(result["count"], 4)
        self.assertEqual(result["chunks"], [{"start": 0, "end": 4, "count": 4,
                                             "sum_cross_entropy": 4 * math.log(2),
                                             "mean_cross_entropy": math.log(2)}])

    def test_outputs_are_owned_and_inputs_unmodified(self):
        params = parameters([[.5, -.25], [1., .75]], [.1, -.2],
                            [[1., -1.], [.25, .5]], [0., .3])
        x = torch.tensor([[1., 2.], [3., -1.]], dtype=torch.float32)
        labels = torch.tensor([0, 1], dtype=torch.int64)
        before_p = {k: bytes_of(v) for k, v in params.items()}
        before_x, before_y = bytes_of(x), bytes_of(labels)
        first = m.evaluate(params, x, labels, chunk_size=1)
        expected = first["mean_gradient"].clone()
        first["mean_gradient"].add_(100)
        second = m.evaluate(params, x, labels, chunk_size=1)
        self.assertTrue(torch.equal(second["mean_gradient"], expected))
        self.assertEqual({k: bytes_of(v) for k, v in params.items()}, before_p)
        self.assertEqual(bytes_of(x), before_x)
        self.assertEqual(bytes_of(labels), before_y)
        self.assertTrue(all(value.grad is None for value in params.values()))

    def test_rng_and_global_settings_are_unchanged(self):
        random.seed(11); np.random.seed(12); torch.manual_seed(13)
        py, nr, tr = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
        grad_enabled = torch.is_grad_enabled()
        default_dtype = torch.get_default_dtype()
        m.evaluate(tiny_zero(), torch.zeros(2, 2), torch.tensor([0, 1]), chunk_size=1)
        self.assertEqual(random.getstate(), py)
        now = np.random.get_state()
        self.assertEqual(now[0], nr[0]); self.assertTrue(np.array_equal(now[1], nr[1]))
        self.assertEqual(now[2:], nr[2:])
        self.assertTrue(torch.equal(torch.get_rng_state(), tr))
        self.assertEqual(torch.is_grad_enabled(), grad_enabled)
        self.assertEqual(torch.get_default_dtype(), default_dtype)
        self.assertFalse(torch.cuda.is_initialized())

    def test_cpu32_and_cpu64_paths_are_separate_diagnostics(self):
        params = parameters([[.2, -.3], [.4, .1]], [.01, -.02],
                            [[.5, -.6], [-.2, .7]], [.03, -.04])
        x = torch.tensor([[.5, -.25], [2., 1.], [-1., .75]], dtype=torch.float32)
        labels = torch.tensor([0, 1, 1], dtype=torch.int64)
        r64 = m.evaluate(params, x, labels, chunk_size=2, precision="cpu64")
        r32 = m.evaluate(params, x, labels, chunk_size=2, precision="native32")
        self.assertEqual(r64["mean_gradient"].dtype, torch.float64)
        self.assertEqual(r32["mean_gradient"].dtype, torch.float32)
        self.assertEqual([(c["start"], c["end"]) for c in r64["chunks"]],
                         [(c["start"], c["end"]) for c in r32["chunks"]])
        self.assertLess(abs(r64["mean_cross_entropy"] - r32["mean_cross_entropy"]), 1e-6)
        self.assertLess(float((r64["mean_gradient"] - r32["mean_gradient"].double()).abs().max()), 1e-6)

    def test_no_grad_and_cpu_autocast_are_scoped_and_preserved(self):
        params = parameters([[.2, -.3], [.4, .1]], [.01, -.02],
                            [[.5, -.6], [-.2, .7]], [.03, -.04])
        x = torch.tensor([[.5, -.25], [2., 1.]], dtype=torch.float32)
        labels = torch.tensor([0, 1], dtype=torch.int64)
        self.assertTrue(torch.is_grad_enabled())
        self.assertFalse(torch.is_autocast_enabled("cpu"))
        with torch.no_grad(), torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            self.assertFalse(torch.is_grad_enabled())
            self.assertTrue(torch.is_autocast_enabled("cpu"))
            result = m.evaluate(params, x, labels, chunk_size=1,
                                with_gradient=True, precision="native32")
            self.assertEqual(result["mean_gradient"].dtype, torch.float32)
            self.assertFalse(torch.is_grad_enabled())
            self.assertTrue(torch.is_autocast_enabled("cpu"))
        self.assertTrue(torch.is_grad_enabled())
        self.assertFalse(torch.is_autocast_enabled("cpu"))

    def test_inference_mode_is_rejected_before_evaluation(self):
        with torch.inference_mode():
            with self.assertRaisesRegex(ValueError, "inference_mode"):
                m.evaluate(tiny_zero(), torch.zeros(1, 2), torch.tensor([0]), chunk_size=1)

    def test_scientific_shape_validation_is_distinct_from_tiny_mode(self):
        params = {"0.weight": torch.zeros(64, 784), "0.bias": torch.zeros(64),
                  "2.weight": torch.zeros(10, 64), "2.bias": torch.zeros(10)}
        result = m.evaluate(params, torch.zeros(2, 784), torch.tensor([0, 9]), chunk_size=1,
                            with_gradient=False)
        self.assertEqual(result["shape_mode"], "scientific_784_64_10")
        self.assertAlmostEqual(result["mean_cross_entropy"], math.log(10), places=14)
        tiny = m.evaluate(tiny_zero(), torch.zeros(1, 2), torch.tensor([0]), chunk_size=1,
                          with_gradient=False)
        self.assertEqual(tiny["shape_mode"], "generic_tiny_mlp")

    def test_rejects_malformed_parameters(self):
        good = tiny_zero()
        cases = []
        cases.append({k: good[k] for k in reversed(m.PARAMETER_ORDER)})
        bad = {k: v.clone() for k, v in good.items()}; bad["0.weight"][0, 0] = float("nan"); cases.append(bad)
        bad = {k: v.clone() for k, v in good.items()}; bad["0.bias"] = bad["0.bias"].double(); cases.append(bad)
        bad = {k: v.clone() for k, v in good.items()}; bad["2.weight"] = torch.zeros(2, 3); cases.append(bad)
        bad = {k: v.clone() for k, v in good.items()}; bad["0.weight"] = torch.zeros(4).view(2, 2); cases.append(bad)
        bad = {k: v.clone() for k, v in good.items()}; bad["2.bias"] = bad["0.bias"]; cases.append(bad)
        large = {"0.weight": torch.zeros(17, 2), "0.bias": torch.zeros(17),
                 "2.weight": torch.zeros(2, 17), "2.bias": torch.zeros(2)}; cases.append(large)
        x, y = torch.zeros(2, 2), torch.tensor([0, 1])
        for index, bad in enumerate(cases):
            with self.subTest(index=index), self.assertRaises(ValueError):
                m.evaluate(bad, x, y, chunk_size=1)

    def test_rejects_detached_views_with_oversized_storage(self):
        params = tiny_zero()
        oversized_parameter = torch.zeros(3, 2)[:2].detach()
        self.assertIsNone(oversized_parameter._base)
        self.assertTrue(oversized_parameter.is_contiguous())
        self.assertGreater(oversized_parameter.untyped_storage().nbytes(),
                           oversized_parameter.numel() * oversized_parameter.element_size())
        params["0.weight"] = oversized_parameter
        with self.assertRaises(ValueError):
            m.evaluate(params, torch.zeros(2, 2), torch.tensor([0, 1]), chunk_size=1)
        oversized_input = torch.zeros(3, 2)[:2].detach()
        with self.assertRaises(ValueError):
            m.evaluate(tiny_zero(), oversized_input, torch.tensor([0, 1]), chunk_size=1)
        oversized_labels = torch.zeros(3, dtype=torch.int64)[:2].detach()
        with self.assertRaises(ValueError):
            m.evaluate(tiny_zero(), torch.zeros(2, 2), oversized_labels, chunk_size=1)

    def test_rejects_malformed_inputs_labels_and_options(self):
        params = tiny_zero()
        x, y = torch.zeros(2, 2), torch.tensor([0, 1])
        cases = [
            (x.double(), y, 1, True, "cpu64"),
            (torch.tensor([[float("inf"), 0.], [0., 0.]]), y, 1, True, "cpu64"),
            (torch.zeros(2, 3), y, 1, True, "cpu64"),
            (torch.empty(0, 2), torch.empty(0, dtype=torch.int64), 1, True, "cpu64"),
            (x, y.to(torch.int32), 1, True, "cpu64"),
            (x, y[:, None], 1, True, "cpu64"),
            (x, torch.tensor([0, 2]), 1, True, "cpu64"),
            (x, y, 0, True, "cpu64"),
            (x, y, True, True, "cpu64"),
            (x, y, 1, 1, "cpu64"),
            (x, y, 1, True, "float64"),
        ]
        for index, (bx, by, chunk, gradient, precision) in enumerate(cases):
            with self.subTest(index=index), self.assertRaises(ValueError):
                m.evaluate(params, bx, by, chunk_size=chunk,
                           with_gradient=gradient, precision=precision)


if __name__ == "__main__":
    unittest.main()
