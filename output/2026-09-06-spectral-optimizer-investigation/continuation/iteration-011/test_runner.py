"""Small CPU-only I11 plan/membership/resource checks; no dataset or GPU."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location("i11_runner_under_test", HERE / "run_continuation.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class RunnerTests(unittest.TestCase):
    def test_plans_are_fixed_shared_inputs_not_global_rng(self):
        before = np.random.get_state()
        first = runner.continuation_plan(100, 1500)
        again = runner.continuation_plan(100, 1500)
        second = runner.continuation_plan(100, 2000)
        self.assertEqual(first["batches"].shape, (1500, 64))
        self.assertTrue(np.array_equal(first["batches"], again["batches"]))
        self.assertFalse(np.array_equal(first["batches"], second["batches"]))
        self.assertTrue(((first["batches"] >= 0) & (first["batches"] < 5000)).all())
        after = np.random.get_state()
        self.assertEqual(before[0], after[0])
        self.assertTrue(np.array_equal(before[1], after[1]))
        self.assertEqual(before[2:], after[2:])

    def test_physical_membership_and_logical_alias_count(self):
        keys = runner.expected_keys()
        self.assertEqual(len(keys), 63)
        self.assertEqual(sum(2 if step == 100 else 1 for _, _, step, _ in keys), 72)
        self.assertEqual({p for _, _, _, p in keys}, {"raw", "current32", "frozen32"})
        self.assertEqual(runner.HORIZONS, [0, 100, 500, 1000, 1500])

    def test_new_cap_counts_smoke_sibling_and_refuses_overwrite(self):
        self.assertEqual(runner.ARTIFACT_CAP, 1024**3)
        self.assertEqual(runner.previous.ARTIFACT_CAP, 2 * 1024**3)
        with tempfile.TemporaryDirectory(prefix="i11-tiny-writer-") as temp:
            root = Path(temp)
            smoke, full = root / "smoke", root / "full"
            smoke.mkdir()
            full.mkdir()
            (smoke / "existing").write_bytes(b"a" * 32)
            run = runner.Run(full, 10)
            with patch.object(run, "check"), patch.object(runner, "ARTIFACT_CAP", 1024**2 + 128):
                self.assertEqual(run.used(), 32)
                record = run.save("ok.json", {"tiny": True})
                self.assertEqual(run.used(), 32 + record["bytes"])
                with self.assertRaises(FileExistsError):
                    run.save("ok.json", {"tiny": True})
                with self.assertRaises(RuntimeError):
                    run.save("too-big.json", {"text": "x" * 128})
                self.assertFalse((full / "too-big.json").exists())


if __name__ == "__main__":
    unittest.main()
