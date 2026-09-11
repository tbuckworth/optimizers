"""Pure CPU harness checks: no MNIST, training, GPU or external service calls."""
import io
import json
import unittest

import numpy as np
import torch

import run_neural_mechanism as runner


class RunnerChecks(unittest.TestCase):
    def test_prospective_plans_are_paired_finite_and_disjoint(self):
        for seed in runner.SEEDS:
            a = runner.make_plan(seed)
            b = runner.make_plan(seed)
            self.assertEqual(json.dumps(runner.json_tree(a)), json.dumps(runner.json_tree(b)))
            groups = [set(a[k]) for k in ["train_indices", "validation_indices", "auxiliary_indices"]]
            self.assertTrue(all(len(g) == 5000 for g in groups))
            self.assertEqual(len(set.union(*groups)), 15000)
            self.assertEqual(a["training_batches"].shape, (2000, 64))
            self.assertEqual(set(a["anchors"]), {str(t) for t in runner.ANCHORS})
            for plan in a["anchors"].values():
                self.assertEqual(plan["pairs"].shape, (32, 2, 64))
                self.assertEqual(plan["update"].shape, (64,))
                for value in plan.values():
                    self.assertTrue(((value >= 0) & (value < 5000)).all())
            self.assertEqual(a["replacement_mask"].dtype, np.bool_)

    def test_budget_writer_cannot_write_over_limit(self):
        handle = io.BytesIO()
        writer = runner.BudgetWriter(handle, 5)
        self.assertEqual(writer.write(b"123"), 3)
        with self.assertRaisesRegex(RuntimeError, "budget"):
            writer.write(b"456")
        self.assertEqual(handle.getvalue(), b"123")

    def test_tensor_serialization_works_through_bounded_writer(self):
        handle = io.BytesIO()
        payload = {"tensor": torch.arange(8), "position": 100, "state": [None, .99]}
        torch.save(payload, runner.BudgetWriter(handle, 65536))
        restored = torch.load(io.BytesIO(handle.getvalue()), weights_only=True)
        self.assertTrue(torch.equal(restored["tensor"], payload["tensor"]))
        self.assertEqual(restored["position"], 100)


if __name__ == "__main__":
    unittest.main()
