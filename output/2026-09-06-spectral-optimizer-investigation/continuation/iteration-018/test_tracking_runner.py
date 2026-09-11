"""I18 runner synthetic tests: no native/RNG/scientific acquisition invocation."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

spec = importlib.util.spec_from_file_location("i18_runner_test", Path(__file__).with_name("run_tracking.py"))
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class RunnerTests(unittest.TestCase):
    def test_membership(self):
        members = runner.expected_cells()
        self.assertEqual(len(members), 192)
        self.assertEqual(len(set(members)), 192)
        self.assertEqual(members[0], (18000, 0, 0))
        self.assertEqual(members[-1], (18031, 2, 1))
        self.assertEqual(runner.HORIZON * len(members), 768000)

    def test_numeric_archive_and_replay_rejection(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "streams").mkdir()
            arrays = {"g": np.arange(8, dtype=np.float64).reshape(4, 2),
                      "basis_present": np.array([False, True, True, True])}
            row = runner.write_stream(root, "9-d0-r0", arrays, {"fixture": True})
            with np.load(root / row["array"]["path"], allow_pickle=False) as loaded:
                np.testing.assert_array_equal(loaded["g"], arrays["g"])
            self.assertEqual(runner.sha(root / row["array"]["path"]), row["array"]["sha256"])
            with self.assertRaisesRegex(RuntimeError, "Consumed"):
                runner.write_stream(root, "9-d0-r0", arrays, {})

    def test_bad_arrays_leave_no_output(self):
        for array in (np.array([float("nan")]), np.array([object()], dtype=object)):
            with tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                (root / "streams").mkdir()
                with self.assertRaisesRegex(RuntimeError, "Nonfinite"):
                    runner.write_stream(root, "9-d0-r0", {"g": array}, {})
                self.assertEqual(list((root / "streams").iterdir()), [])

    def test_bad_metadata_leaves_no_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "streams").mkdir()
            with self.assertRaises(ValueError):
                runner.write_stream(root, "9-d0-r0", {"g": np.zeros(1)}, {"bad": float("nan")})
            self.assertEqual(list((root / "streams").iterdir()), [])

    def test_json_exclusive(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "record.json"
            runner.json_once(path, {"x": 1})
            with self.assertRaises(FileExistsError):
                runner.json_once(path, {"x": 2})
            self.assertEqual(json.loads(path.read_text()), {"x": 1})

    def test_root_rejects_unapproved_location(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(RuntimeError, "Unexpected root"):
                runner.preflight_root(Path(temp))

    def test_source_requires_full_pin(self):
        with self.assertRaisesRegex(RuntimeError, "Full frozen"):
            runner.source_manifest("HEAD")


if __name__ == "__main__":
    unittest.main()
