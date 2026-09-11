"""Mocked runner and tiny NumPy/logit fixtures; never launch acquisition."""

import hashlib
import io
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from experiments import spectral_wrong_label_augmentation as b


class WrongLabelRunnerTests(unittest.TestCase):
    def test_roster_schedule_namespace_and_inherited_bounds(self):
        expected = []
        for si, seed in enumerate((202609151, 202609152, 202609153)):
            for ai, augmentation in enumerate(("none", "translate")):
                policies = ("raw", "native32") if (si + ai) % 2 == 0 else ("native32", "raw")
                expected.extend({"seed": seed, "policy": policy, "augmentation": augmentation} for policy in policies)
        self.assertEqual(b.branch_roster(), expected)
        self.assertEqual(len({tuple(row.values()) for row in expected}), 12)
        self.assertEqual(b.EVAL_STEPS, (0, 100) + tuple(range(200, 4001, 200)))
        self.assertEqual((len(b.EVAL_STEPS), b.STEPS, b.BATCH), (22, 4000, 64))
        self.assertEqual(b.UNIT, "spectral-wrong-label-augmentation-001.service")
        self.assertNotEqual(b.UNIT, b.ordinary.UNIT)
        self.assertNotEqual(b.DOCS, b.ordinary.DOCS)
        self.assertEqual((b.MAX_BYTES, b.RESERVE_BYTES, b.DEADLINE_SECONDS), (1024**3, 1024**2, 900))
        self.assertIs(b.Run, b.ordinary.Run)
        self.assertIs(b.CappedWriter, b.ordinary.CappedWriter)
        self.assertIs(b.validate_bounds, b.ordinary.validate_bounds)
        self.assertIs(b.validate_gpu_clients, b.ordinary.validate_gpu_clients)
        inventory = b.byte_inventory()
        self.assertLess(inventory["total_upper_bytes"] + b.RESERVE_BYTES, b.MAX_BYTES)

    def test_cli_without_execute_is_inert_and_no_science_overrides(self):
        forbidden = AssertionError("science forbidden in fixtures")
        with patch.object(b, "acquire", side_effect=forbidden), \
             patch.object(b, "configure", side_effect=forbidden), \
             patch.object(b.ordinary, "read_training", side_effect=forbidden), \
             patch.object(Path, "open", side_effect=forbidden), \
             patch.object(Path, "mkdir", side_effect=forbidden):
            with self.assertRaisesRegex(RuntimeError, "without explicit --execute"):
                b.main(["--output-dir", "/unused"])
        for flags in (("--seed", "1"), ("--steps", "1"), ("--noise", "0.1"),
                      ("--fraction", "0.1"), ("--device", "cpu"), ("--resume", "/unused"),
                      ("--policy", "raw"), ("--exec",)):
            with self.subTest(flags=flags), patch("sys.stderr", new=io.StringIO()):
                with self.assertRaises(SystemExit):
                    b.parse_args(["--output-dir", "/unused", *flags])

    def test_source_pin_contract_and_immutable_ordinary_dependencies(self):
        for name, expected in b.FROZEN_ORDINARY.items():
            self.assertEqual(hashlib.sha256((b.ROOT / name).read_bytes()).hexdigest(), expected)
        def fixture_digest(path):
            name = str(Path(path).resolve().relative_to(b.ROOT))
            if name in b.FROZEN_ORDINARY:
                return b.FROZEN_ORDINARY[name]
            if name == "spectral_filter.py":
                return b.core.FILTER_SHA256
            if Path(path).resolve() == Path(b.core.__file__).resolve():
                return b.ordinary.CORE_SHA
            return hashlib.sha256(name.encode()).hexdigest()
        with patch.object(b, "digest", side_effect=fixture_digest):
            pins = b.source_pins()
        for name in ("experiments/spectral_wrong_label_augmentation.py",
                     "experiments/spectral_wrong_label_augmentation_data.py",
                     "experiments/spectral_wrong_label_augmentation_audit.py",
                     "experiments/spectral_general_augmentation_audit.py",
                     "tests/test_spectral_wrong_label_augmentation.py",
                     "tests/test_spectral_wrong_label_augmentation_data.py",
                     "tests/test_spectral_wrong_label_augmentation_audit.py"):
            self.assertIn(name, pins)
        for document in ("protocol.md", "implementation-check.md", "visual-review.md"):
            self.assertIn(str((b.DOCS / document).relative_to(b.ROOT)), pins)
        with patch.object(b, "digest", side_effect=lambda path: "wrong" if
                          Path(path).resolve() == Path(b.ordinary.__file__).resolve() else fixture_digest(path)):
            with self.assertRaisesRegex(RuntimeError, "frozen ordinary dependency changed"):
                b.source_pins()

    def test_five_readout_denominators_and_wrong_vs_true_labels(self):
        train_logits = np.zeros((3, 10), dtype=np.float32)
        train_logits[0, 7] = train_logits[1, 1] = train_logits[2, 2] = 2
        heldout_logits = np.zeros((2, 10), dtype=np.float32)
        heldout_logits[0, 3] = heldout_logits[1, 4] = 2
        plan = {"train_labels": np.array([0, 1, 2]), "assigned_labels": np.array([7, 8, 2]),
                "eval_labels": np.array([3, 4]), "corruption_mask": np.array([True, True, False])}
        result = b.evaluation_row(100, train_logits, heldout_logits, plan)
        self.assertEqual(set(result), {"step", "train_clean", "train_assigned", "wrong_true", "wrong_target", "heldout"})
        self.assertEqual(result["step"], 100)
        expected = {"train_clean": (3, 2), "train_assigned": (3, 2),
                    "wrong_true": (2, 1), "wrong_target": (2, 1), "heldout": (2, 2)}
        for key, (count, correct) in expected.items():
            self.assertEqual((result[key]["count"], result[key]["correct"]), (count, correct))
        good_ce = math.log(math.exp(2) + 9) - 2
        self.assertAlmostEqual(result["heldout"]["ce"], good_ce, places=12)
        self.assertAlmostEqual(result["wrong_target"]["ce"], good_ce + 1, places=12)
        np.testing.assert_array_equal(plan["assigned_labels"], [7, 8, 2])
        for invalid in (np.array([1, 1, 0]), np.array([True]), np.zeros(3, dtype=bool)):
            with self.assertRaisesRegex(RuntimeError, "corruption mask schema"):
                b.evaluation_row(100, train_logits, heldout_logits, {**plan, "corruption_mask": invalid})

    def test_fixed_assigned_targets_through_repeated_differently_shifted_views(self):
        images = np.zeros((2, 784), dtype=np.float32)
        images[:, 28 * 10 + 10] = 1
        assigned = np.array([7, 4], dtype=np.int64)
        batch = np.array([0, 0, 1], dtype=np.int64)
        shifts = np.array([[1, 0], [-1, 0], [0, 2]], dtype=np.int8)
        before = images.copy()
        for mode in ("none", "translate"):
            with patch.object(b.ordinary, "training_update", return_value=(2.5, None)) as update:
                loss, diagnostic, seconds = b.update_occurrences(
                    "model", "optimizer", None, images, assigned, batch, shifts, "raw", mode, "cpu")
            args = update.call_args.args
            self.assertEqual(args[:3], ("model", "optimizer", None))
            self.assertEqual(args[-1], "raw")
            np.testing.assert_array_equal(args[4].numpy(), [7, 7, 4])
            expected = images[batch] if mode == "none" else b.data.translate(images[batch], shifts)
            np.testing.assert_array_equal(args[3].numpy(), expected)
            self.assertEqual((loss, diagnostic), (2.5, None))
            self.assertGreaterEqual(seconds, 0)
            if mode == "none":
                self.assertEqual(seconds, 0)
        np.testing.assert_array_equal(images, before)
        np.testing.assert_array_equal(assigned, [7, 4])

    def test_update_rejects_invalid_occurrences_before_training(self):
        images, assigned = np.zeros((2, 784), dtype=np.float32), np.array([7, 4], dtype=np.int64)
        with patch.object(b.ordinary, "training_update", side_effect=AssertionError("training forbidden")):
            for batch in (np.array([-1], dtype=np.int64), np.array([2], dtype=np.int64), np.array([0.])):
                with self.assertRaisesRegex(RuntimeError, "batch schema"):
                    b.update_occurrences(None, None, None, images, assigned, batch, None, "raw", "none", "cpu")
            with self.assertRaisesRegex(RuntimeError, "assigned label schema"):
                b.update_occurrences(None, None, None, images, assigned.astype(float), np.array([0]),
                                     None, "raw", "none", "cpu")
            with self.assertRaisesRegex(RuntimeError, "unknown augmentation"):
                b.update_occurrences(None, None, None, images, assigned, np.array([0]),
                                     None, "raw", "unknown", "cpu")

    def test_new_unit_rejection_without_old_configure_or_gpu(self):
        environ = {key: "1" for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")}
        environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        with patch.dict(b.os.environ, environ), \
             patch.object(Path, "read_text", return_value="0::/user.slice/" + b.ordinary.UNIT + "\n"), \
             patch.object(b.ordinary, "configure", side_effect=AssertionError("old unit configure forbidden")), \
             patch.object(b.torch.cuda, "is_available", side_effect=AssertionError("GPU forbidden")):
            with self.assertRaisesRegex(RuntimeError, "unexpected acquisition unit"):
                b.configure()

    def test_inherited_writer_npz_roundtrip_and_exclusive_child(self):
        with tempfile.TemporaryDirectory() as directory:
            run = b.Run(directory, device="cpu")
            arrays = {"assigned_labels": np.array([7, 4], dtype=np.int64),
                      "corruption_mask": np.array([True, True])}
            with patch.object(run, "check"):
                receipt = run.save("fixture.npz", arrays, "npz")
                with self.assertRaises(FileExistsError):
                    run.save("fixture.npz", arrays, "npz")
                with self.assertRaisesRegex(RuntimeError, "direct child"):
                    run.save("../escape.json", {})
            path = Path(directory) / "fixture.npz"
            with np.load(path, allow_pickle=False) as saved:
                for key, value in arrays.items():
                    np.testing.assert_array_equal(saved[key], value)
            self.assertEqual(receipt["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())


if __name__ == "__main__":
    unittest.main()
