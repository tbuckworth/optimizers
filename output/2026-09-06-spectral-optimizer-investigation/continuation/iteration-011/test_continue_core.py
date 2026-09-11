"""Synthetic CPU tests for exact I11 split continuation; no MNIST or GPU."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from pathlib import Path
import sys
import unittest

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
import continue_core as core


class TestContinuation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        generator = torch.Generator().manual_seed(11011)
        x = torch.rand((128, 6), generator=generator)
        clean = torch.randint(0, 3, (128,), generator=generator)
        noisy = torch.randint(0, 3, (128,), generator=generator)
        cls.data = {
            "x": x, "clean": clean, "noisy": noisy,
            "vx": x.flip(0).clone(), "vy": clean.flip(0).clone(),
            "ax": x.roll(1, 0).clone(), "ay": clean.roll(1, 0).clone(),
        }
        model = core.i9.make_model(11011, "cpu", input_dim=6, width=5, classes=3)
        optimizer = core.i9.make_optimizer(model)
        tracker = core.i9.make_tracker(model, optimizer)
        for step in range(100):
            start = step % 64
            indices = torch.arange(start, start + 64)
            core.i9.train_step(model, optimizer, tracker, x[indices], noisy[indices], "current32")
        cls.state = core.i9.snapshot(model, optimizer, tracker)
        cls.original_basis = core.i9._basis(tracker).detach().cpu().clone()
        cls.plan = np.random.default_rng(11011).integers(0, 128, (12, 64))

    def test_split_matches_uninterrupted_for_all_policies(self):
        input_digest = core.i9.tree_digest(self.state)
        basis_digest = core.i9.tree_digest(self.original_basis)
        baselines = []
        for policy in core.i10.POLICIES:
            first_curve, first_steps, middle = core.continue_fixed(
                self.state, self.data, self.plan[:5], policy, self.original_basis,
                steps=5, eval_horizons=(0, 5))
            second_curve, second_steps, split_final = core.continue_fixed(
                middle, self.data, self.plan[5:], policy, self.original_basis,
                steps=7, eval_horizons=(0, 7))
            full_curve, full_steps, full_final = core.continue_fixed(
                self.state, self.data, self.plan, policy, self.original_basis,
                steps=12, eval_horizons=(0, 5, 12))

            self.assertEqual(core.i9.tree_digest(split_final), core.i9.tree_digest(full_final))
            self.assertTrue(core.i9.equal_tree(split_final, full_final))
            self.assertEqual([row["horizon"] for row in first_steps], list(range(1, 6)))
            self.assertEqual([row["horizon"] for row in second_steps], list(range(1, 8)))
            self.assertEqual([row["horizon"] for row in full_steps], list(range(1, 13)))
            self.assertEqual([row["horizon"] for row in full_curve], [0, 5, 12])
            self.assertEqual({key: value for key, value in first_curve[-1].items()
                              if key != "horizon"},
                             {key: value for key, value in second_curve[0].items()
                              if key != "horizon"})
            baselines.append(full_curve[0])

            self.assertEqual(core._optimizer_counter(full_final),
                             core._optimizer_counter(self.state) + 12)
            expected_observer = (self.state["tracker"]["step_count"] if policy == "frozen32"
                                 else self.state["tracker"]["step_count"] + 12)
            self.assertEqual(full_final["tracker"]["step_count"], expected_observer)
            if policy == "frozen32":
                self.assertTrue(core.i9.equal_tree(full_final["tracker"],
                                                   self.state["tracker"]))

        self.assertTrue(all(row == baselines[0] for row in baselines[1:]))
        self.assertEqual(core.i9.tree_digest(self.state), input_digest)
        self.assertEqual(core.i9.tree_digest(self.original_basis), basis_digest)

    def test_evaluation_is_complete_state_neutral(self):
        model, optimizer, tracker = core.i9.restore(self.state, "cpu")
        before = core.i9.snapshot(model, optimizer, tracker)
        result = core.i10.evaluate(model, self.data, chunk_size=31)
        after = core.i9.snapshot(model, optimizer, tracker)
        self.assertEqual(result["schema"], "i10_branch_evaluation_v1")
        self.assertTrue(core.i9.equal_tree(before, after))
        self.assertEqual(core.i9.tree_digest(before), core.i9.tree_digest(after))

    def test_check_callback_and_frozen_reference_are_explicit(self):
        calls = []
        _, diagnostics, final = core.continue_fixed(
            self.state, self.data, self.plan[:2], "frozen32", self.original_basis,
            steps=2, eval_horizons=(0, 1, 2), check=lambda: calls.append(len(calls)))
        # Before h0, before each update, and before each nonzero evaluation.
        self.assertEqual(len(calls), 5)
        self.assertTrue(all(row["gradient_filter_applied"] for row in diagnostics))
        self.assertTrue(core.i9.equal_tree(final["tracker"], self.state["tracker"]))
        wrong = self.original_basis.roll(1, 0)
        with self.assertRaises((core.ContinuationError, core.i10.BranchCoreError)):
            core.continue_fixed(self.state, self.data, self.plan[:1], "frozen32", wrong,
                                steps=1, eval_horizons=(0, 1))

    def test_initial_evaluation_mismatch_rejects_before_update_loop(self):
        model, optimizer, tracker = core.i9.restore(self.state, "cpu")
        expected = core.i10.evaluate(model, self.data)
        expected["auxiliary"]["clean_ce"] += 1.0
        calls = []
        input_digest = core.i9.tree_digest(self.state)
        with self.assertRaisesRegex(core.ContinuationError, "bound I10 endpoint"):
            core.continue_fixed(
                self.state, self.data, self.plan[:1], "current32", self.original_basis,
                steps=1, eval_horizons=(0, 1), check=lambda: calls.append(len(calls)),
                expected_initial_evaluation=expected)
        # The only call is before the neutral h0 evaluation; the per-update check is unreached.
        self.assertEqual(len(calls), 1)
        self.assertEqual(core.i9.tree_digest(self.state), input_digest)


if __name__ == "__main__":
    unittest.main()
