"""Small CPU-only I12 membership, plan and failure-sealing checks."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location("i12_runner_under_test", HERE / "run_moment_branches.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class RunnerTests(unittest.TestCase):
    def test_membership_is_six_parents_and_24_existing_baselines(self):
        self.assertEqual(len(runner.expected_baseline_keys()), 24)
        self.assertEqual(len(runner.expected_keys()), 96)
        self.assertEqual(len({(s, t) for s, t, _, _, _ in runner.expected_keys()}), 6)
        self.assertEqual(runner.HORIZONS, (0, 1, 10, 50, 100, 250, 500))
        self.assertNotIn("inherited", runner.ARMS)
        self.assertNotIn("fixed", runner.OBJECTIVES)
        self.assertNotIn("frozen32", runner.POLICIES)

    def test_saved_plan_types_ranges_and_no_mutation(self):
        raw = runner.parent_run.json_tree(runner.previous.branch_plan(100, 1500))
        before = json.dumps(raw)
        plan = runner.validate_plan(raw, 100, 1500)
        self.assertEqual(json.dumps(raw), before)
        self.assertTrue(np.array_equal(plan["batches"], raw["batches"]))
        self.assertTrue(np.array_equal(plan["redraw_mask"], raw["redraw_mask"]))
        self.assertTrue(np.array_equal(plan["redraw_digits"], raw["redraw_digits"]))
        raw["batches"][0][0] = 0.5
        with self.assertRaises(AssertionError):
            runner.validate_plan(raw, 100, 1500)

    def test_shared_cap_and_exclusive_writes(self):
        self.assertEqual(runner.ARTIFACT_CAP, 2 * 1024**3)
        with tempfile.TemporaryDirectory(prefix="i12-tiny-writer-") as temp:
            root = Path(temp)
            smoke, full = root / "smoke", root / "full"
            smoke.mkdir()
            full.mkdir()
            (smoke / "sibling").write_bytes(b"a" * 32)
            run = runner.Run(full, 10)
            with patch.object(run, "check"), patch.object(runner, "ARTIFACT_CAP", 1024**2 + 128):
                self.assertEqual(run.used(), 32)
                run.save("tiny.json", {"ok": True})
                with self.assertRaises(FileExistsError):
                    run.save("tiny.json", {})
                with self.assertRaises(RuntimeError):
                    run.save("oversized.json", {"large": "x" * 128})
                with self.assertRaises(AssertionError):
                    run.save("../escape.json", {})
                self.assertFalse((full / "oversized.json").exists())

    def test_numerical_failure_seals_both_states_without_fake_endpoint(self):
        with tempfile.TemporaryDirectory(prefix="i12-tiny-failure-") as temp:
            root = Path(temp) / "full"
            root.mkdir()
            run = runner.Run(root, 10)
            failure = {"status": "numerical_failure", "completed_steps": 2,
                "attempted_step": 3, "curve": [{"horizon": 0}, {"horizon": 1}],
                "steps": [{"horizon": 1}, {"horizon": 2}],
                "terminal_state": {"tensor": torch.tensor(float("nan"))},
                "last_evaluated_state": {"tensor": torch.tensor(1.0)},
                "last_evaluated_horizon": 1,
                "first_step_applied_gradient_digest": "a" * 64,
                "numerical_failure": {"reason": "synthetic nonfinite", "stage": "step"}}
            with patch.object(run, "check"):
                record = runner.seal_branch(run, {"id": "synthetic"}, failure)
            self.assertEqual(record["status"], "numerical_failure")
            saved = json.loads((root / record["branch_artifact"]["name"]).read_text())
            self.assertEqual(saved["last_evaluated_horizon"], 1)
            self.assertEqual(saved["attempted_step"], 3)
            self.assertNotIn(500, [r["horizon"] for r in saved["curve"]])
            self.assertTrue((root / saved["last_evaluated_artifact"]["name"]).exists())
            self.assertTrue((root / saved["terminal_artifact"]["name"]).exists())

    def test_short_branch_cannot_be_sealed_as_complete(self):
        with tempfile.TemporaryDirectory(prefix="i12-tiny-incomplete-") as temp:
            run = runner.Run(Path(temp), 10)
            with self.assertRaises(AssertionError):
                runner.seal_branch(run, {"id": "synthetic"}, {
                    "status": "complete", "completed_steps": 2, "attempted_step": 2})
            self.assertEqual(list(Path(temp).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
