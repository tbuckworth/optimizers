"""Parent-owned resource-callback integration, dataset-free CPU only."""
import importlib.util
import os
from pathlib import Path
import unittest
from unittest import mock

if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
    raise RuntimeError("Explicitly hide CUDA for CPU fixtures")

import torch
import numpy as np

spec = importlib.util.spec_from_file_location("i7_guarded_loss", Path(__file__).with_name("loss_measurements.py"))
loss = importlib.util.module_from_spec(spec)
spec.loader.exec_module(loss)
spec = importlib.util.spec_from_file_location("i7_guarded_independent", Path(__file__).with_name("independent_numerics.py"))
independent = importlib.util.module_from_spec(spec)
spec.loader.exec_module(independent)


class GuardedLossTests(unittest.TestCase):
    def setUp(self):
        self.params = {"0.weight": torch.ones(4, 3), "0.bias": torch.zeros(4),
                       "2.weight": torch.zeros(2, 4), "2.bias": torch.zeros(2)}
        self.x = torch.ones(5, 3)
        self.y = torch.zeros(5, dtype=torch.int64)

    def evaluate(self, **kwargs):
        return loss.evaluate(self.params, self.x, self.y, chunk_size=2, **kwargs)

    def test_guard_preserves_values_and_brackets_every_chunk(self):
        events = []
        rng = torch.get_rng_state().clone()
        baseline = self.evaluate()
        observed = self.evaluate(guard=events.append)
        self.assertEqual(events, ["loss.before_validation", "loss.after_validation",
            "loss.before_copy", "loss.after_copy", "loss.before_chunk.0.2",
            "loss.after_chunk.0.2", "loss.before_chunk.2.4", "loss.after_chunk.2.4",
            "loss.before_chunk.4.5", "loss.after_chunk.4.5",
            "loss.before_reduction", "loss.after_reduction"])
        self.assertTrue(torch.equal(observed.pop("mean_gradient"), baseline.pop("mean_gradient")))
        self.assertEqual(observed, baseline)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertFalse(torch.cuda.is_initialized())

    def test_failure_stops_later_chunks_and_restores_grad_mode(self):
        events = []
        def guard(label):
            events.append(label)
            if label == "loss.after_chunk.0.2":
                raise RuntimeError("fixture resource limit")
        with torch.no_grad(), mock.patch.object(loss, "_functional_logits", wraps=loss._functional_logits) as forward:
            with self.assertRaisesRegex(RuntimeError, "fixture resource limit"):
                self.evaluate(guard=guard)
            self.assertFalse(torch.is_grad_enabled())
            self.assertEqual(forward.call_count, 1)
        self.assertEqual(events[-1], "loss.after_chunk.0.2")

    def test_invalid_guard_is_rejected_before_computation(self):
        with mock.patch.object(loss, "_functional_logits") as forward:
            with self.assertRaisesRegex(ValueError, "guard"):
                self.evaluate(guard=42)
            forward.assert_not_called()

    def test_independent_callback_preserves_reductions_and_stops_on_failure(self):
        args = ([value.numpy() for value in self.params.values()], self.x.numpy(), self.y.numpy())
        baseline = independent.mlp_ce_gradient(*args, chunk_size=2)
        events = []
        observed = independent.mlp_ce_gradient(*args, chunk_size=2, guard=events.append)
        self.assertEqual(baseline["mean_ce"], observed["mean_ce"])
        self.assertTrue(np.array_equal(baseline["gradient"], observed["gradient"]))
        self.assertEqual(events, ["independent_loss.before_validation_copy",
            "independent_loss.after_validation_copy", "independent_loss.before_chunk.0.2",
            "independent_loss.after_chunk.0.2", "independent_loss.before_chunk.2.4",
            "independent_loss.after_chunk.2.4", "independent_loss.before_chunk.4.5",
            "independent_loss.after_chunk.4.5", "independent_loss.before_reduction",
            "independent_loss.after_reduction"])
        for error_class in (RuntimeError, ValueError, FloatingPointError):
            for stop_label in ("independent_loss.before_chunk.0.2", "independent_loss.after_chunk.0.2"):
                with self.subTest(error_class=error_class, stop_label=stop_label):
                    events.clear()
                    failure = error_class("fixture resource limit")
                    def fail(label):
                        events.append(label)
                        if label == stop_label:
                            raise failure
                    with self.assertRaises(error_class) as caught:
                        independent.mlp_ce_gradient(*args, chunk_size=2, guard=fail)
                    self.assertIs(caught.exception, failure)
                    self.assertEqual(events[-1], stop_label)
        with self.assertRaisesRegex(ValueError, "guard"):
            independent.mlp_ce_gradient(*args, guard=False)


if __name__ == "__main__":
    unittest.main()
