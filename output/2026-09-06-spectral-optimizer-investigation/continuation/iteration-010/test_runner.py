"""I10 pure CPU plan/target/writer checks; no MNIST or GPU execution."""
import io
import json
import unittest
from pathlib import Path
import sys
import numpy as np
import torch
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_branches as run


class RunnerTests(unittest.TestCase):
    def test_local_plans_are_deterministic_complete_and_separate(self):
        for seed in run.SEEDS:
            for step in run.ANCHORS:
                a, b = run.branch_plan(seed, step), run.branch_plan(seed, step)
                self.assertEqual(json.dumps(run.parent_run.json_tree(a)),
                                 json.dumps(run.parent_run.json_tree(b)))
                for key in ("batches", "redraw_mask", "redraw_digits"):
                    self.assertEqual(a[key].shape, (500, 64))
                self.assertTrue(((a["batches"] >= 0) & (a["batches"] < 5000)).all())
                self.assertTrue(((a["redraw_digits"] >= 0) & (a["redraw_digits"] < 10)).all())
                self.assertEqual(a["redraw_mask"].dtype, np.bool_)
        self.assertFalse(np.array_equal(run.branch_plan(100, 100)["batches"],
                                        run.branch_plan(100, 500)["batches"]))

    def test_targets_use_fixed_soft_or_per_occurrence_redraw_exactly(self):
        data = {"clean": torch.tensor([1, 2, 3]), "noisy": torch.tensor([4, 5, 6])}
        indices = torch.tensor([0, 0, 1])
        plan = {"redraw_mask": np.array([[True, False, True]]),
                "redraw_digits": np.array([[9, 7, 0]])}
        fixed = run.target_for(data, indices, plan, 0, "fixed")
        soft = run.target_for(data, indices, plan, 0, "soft")
        redraw = run.target_for(data, indices, plan, 0, "redraw")
        self.assertTrue(torch.equal(fixed, torch.tensor([4, 4, 5])))
        self.assertTrue(torch.equal(redraw, torch.tensor([9, 1, 0])))
        self.assertEqual(soft.shape, (3, 10))
        self.assertTrue(torch.allclose(soft.sum(1), torch.ones(3, dtype=torch.double)))
        self.assertAlmostEqual(soft[0, 1].item(), .19)
        self.assertAlmostEqual(soft[0, 2].item(), .09)

    def test_alias_design_counts_and_writer_cap(self):
        unique = sum(1 if step == 100 else 2 for step in run.ANCHORS) * 3 * 9
        self.assertEqual(unique, 189)
        self.assertEqual(len(run.SEEDS) * len(run.ANCHORS) * 2 * 9, 216)
        self.assertEqual(run.ARTIFACT_CAP, 2 * 1024**3)
        self.assertEqual(run.parent_run.MAX_ARTIFACT_BYTES, 512 * 1024**2)
        handle = io.BytesIO()
        writer = run.parent_run.BudgetWriter(handle, 3)
        writer.write(b"ab")
        with self.assertRaises(RuntimeError):
            writer.write(b"cd")
        self.assertEqual(handle.getvalue(), b"ab")


if __name__ == "__main__":
    unittest.main()
