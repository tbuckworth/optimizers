"""Fabricated/mocked runner contracts only; no IDX, acquisition, or real GPU."""

from collections import OrderedDict
import hashlib
import io
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from experiments import spectral_general_augmentation as b


def expected_source_paths():
    """Independent inventory of the files required by the acquisition freeze."""
    return [
        b.ROOT / "spectral_filter.py", Path(b.core.__file__), Path(b.__file__),
        Path(b.data.__file__), b.DOCS / "protocol.md", b.DOCS / "implementation-check.md",
        b.DOCS / "visual-review.md", b.ROOT / "experiments/spectral_general_augmentation_audit.py",
        b.ROOT / "experiments/spectral_general_augmentation_preview.py",
        b.ROOT / "tests/test_spectral_general_augmentation.py",
        b.ROOT / "tests/test_spectral_general_augmentation_data.py",
        b.ROOT / "tests/test_spectral_general_augmentation_audit.py",
        b.DOCS / "preview.json", b.DOCS / "preview-1.png",
        b.DOCS / "preview-2.png", b.DOCS / "preview-3.png",
    ]


def fixture_source_digest(path):
    """Do not require unfinished companion files for the contract-only test."""
    if Path(path) == b.ROOT / "spectral_filter.py":
        return b.core.FILTER_SHA256
    if Path(path) == Path(b.core.__file__):
        return b.CORE_SHA
    return hashlib.sha256(str(path).encode()).hexdigest()


class FixedContractTests(unittest.TestCase):
    def test_roster_all_12_paired_branches_and_alternating_order(self):
        roster = b.branch_roster()
        self.assertEqual(b.SEEDS, (202609141, 202609142, 202609143))
        self.assertEqual((b.STEPS, b.BATCH), (4000, 64))
        self.assertEqual(b.POLICIES, ("raw", "native32"))
        self.assertEqual(b.AUGMENTATIONS, ("none", "translate"))
        expected = []
        for si, seed in enumerate(b.SEEDS):
            for ai, augmentation in enumerate(("none", "translate")):
                policies = ("raw", "native32") if (si + ai) % 2 == 0 else ("native32", "raw")
                expected.extend({"seed": seed, "policy": policy, "augmentation": augmentation}
                                for policy in policies)
        self.assertEqual(roster, expected)
        self.assertEqual(len(roster), 12)
        self.assertEqual(len({(r["seed"], r["policy"], r["augmentation"]) for r in roster}), 12)

    def test_evaluation_schedule_and_byte_inventory(self):
        self.assertEqual(b.EVAL_STEPS, (0, 100) + tuple(range(200, 4001, 200)))
        self.assertEqual(len(b.EVAL_STEPS), 22)
        self.assertEqual(len(b.branch_roster()) * len(b.EVAL_STEPS), 264)
        inventory = b.byte_inventory()
        self.assertEqual(inventory["component_upper_bytes"], {
            "27_full_states_upper": 27 * (37 * 50890 * 4 + 128 * 1024),
            "12_logit_histories": 12 * 22 * 2 * 5000 * 10 * 4,
            "plans_streams_metadata_receipts_allowance": 64 * 1024**2,
        })
        self.assertEqual(inventory["total_upper_bytes"], sum(inventory["component_upper_bytes"].values()))
        self.assertEqual((inventory["cap_bytes"], b.RESERVE_BYTES), (1024**3, 1024**2))
        self.assertLess(inventory["total_upper_bytes"] + b.RESERVE_BYTES, b.MAX_BYTES)

    def test_cli_without_execute_is_inert(self):
        forbidden = AssertionError("scientific access is forbidden in fixtures")
        with patch.object(b, "acquire", side_effect=forbidden) as acquire, \
             patch.object(b, "configure", side_effect=forbidden) as configure, \
             patch.object(b, "read_training", side_effect=forbidden) as read, \
             patch.object(b, "source_pins", side_effect=forbidden) as pins, \
             patch.object(Path, "open", side_effect=forbidden), \
             patch.object(Path, "mkdir", side_effect=forbidden):
            with self.assertRaisesRegex(RuntimeError, "without explicit --execute"):
                b.main(["--output-dir", "/not/a/real/acquisition"])
            for method in (acquire, configure, read, pins):
                method.assert_not_called()

    def test_cli_has_no_scientific_overrides_or_abbreviation(self):
        args = b.parse_args(["--execute", "--output-dir", "/unused"])
        self.assertEqual(vars(args), {"execute": True, "output_dir": Path("/unused")})
        for extra in (("--seed", "1"), ("--steps", "1"), ("--batch", "1"),
                      ("--rank", "1"), ("--device", "cpu"), ("--resume", "/unused"),
                      ("--policy", "raw"), ("--augmentation", "none"), ("--exec",),
                      ("--legacy-update",)):
            with self.subTest(extra=extra), patch("sys.stderr", new=io.StringIO()):
                with self.assertRaises(SystemExit) as caught:
                    b.parse_args(["--output-dir", "/unused", *extra])
                self.assertEqual(caught.exception.code, 2)
        with patch("sys.stdout", new=io.StringIO()):
            with self.assertRaises(SystemExit) as caught:
                b.parse_args(["--help"])
            self.assertEqual(caught.exception.code, 0)

    def test_source_pin_inventory_without_requiring_unfinished_scaffold(self):
        with patch.object(b, "digest", side_effect=fixture_source_digest) as digest:
            pins = b.source_pins()
        expected = {str(path.relative_to(b.ROOT)) for path in expected_source_paths()}
        self.assertEqual(set(pins), expected)
        self.assertEqual(digest.call_count, len(expected))
        self.assertEqual(pins["spectral_filter.py"], b.core.FILTER_SHA256)

    def test_source_pins_reject_changed_filter_and_core(self):
        for target, message in ((b.ROOT / "spectral_filter.py", "canonical filter changed"),
                                (Path(b.core.__file__), "accepted I9 core changed")):
            def changed(path):
                return "wrong" if Path(path) == target else fixture_source_digest(path)
            with self.subTest(target=target), patch.object(b, "digest", side_effect=changed):
                with self.assertRaisesRegex(RuntimeError, message):
                    b.source_pins()

    def test_actual_source_pins_once_companion_files_exist(self):
        paths = expected_source_paths()
        missing = [str(path.relative_to(b.ROOT)) for path in paths if not path.is_file()]
        if missing:
            self.skipTest("unfinished scaffold: " + ", ".join(missing))
        pins = b.source_pins()
        self.assertEqual(set(pins), {str(path.relative_to(b.ROOT)) for path in paths})
        for path in paths:
            self.assertEqual(pins[str(path.relative_to(b.ROOT))], hashlib.sha256(path.read_bytes()).hexdigest())


class WriterTests(unittest.TestCase):
    def test_capped_writer_counts_bytes_checks_and_flushes(self):
        handle, check = io.BytesIO(), Mock()
        writer = b.CappedWriter(handle, 4, check)
        self.assertEqual(writer.write(memoryview(b"abcd")), 4)
        self.assertEqual(writer.allowance, 0)
        with self.assertRaisesRegex(RuntimeError, "byte cap"):
            writer.write(b"e")
        self.assertEqual(handle.getvalue(), b"abcd")
        self.assertEqual(check.call_count, 2)
        writer.flush()
        other = Mock()
        b.CappedWriter(other, 1).flush()
        other.flush.assert_called_once_with()

    def test_short_write_and_guard_failure(self):
        short = Mock()
        short.write.return_value = 1
        with self.assertRaisesRegex(RuntimeError, "short write"):
            b.CappedWriter(short, 4).write(b"abc")
        handle = io.BytesIO()
        with self.assertRaisesRegex(RuntimeError, "fixture guard"):
            b.CappedWriter(handle, 4, Mock(side_effect=RuntimeError("fixture guard"))).write(b"a")
        self.assertEqual(handle.getvalue(), b"")

    def test_numpy_archive_uses_capped_writer_without_files(self):
        handle = io.BytesIO()
        arrays = {"steps": np.array([0, 100], dtype=np.int64),
                  "heldout": np.zeros((2, 3, 10), dtype=np.float32)}
        np.savez(b.CappedWriter(handle, 10000), **arrays)
        handle.seek(0)
        with np.load(handle, allow_pickle=False) as saved:
            self.assertEqual(set(saved.files), set(arrays))
            for key, expected in arrays.items():
                np.testing.assert_array_equal(saved[key], expected)

    def test_actual_numpy_archive_save_load_and_receipt(self):
        arrays = {"steps": np.array([0, 100], dtype=np.int64),
                  "train": np.arange(60, dtype=np.float32).reshape(2, 3, 10)}
        with tempfile.TemporaryDirectory() as directory:
            run = b.Run(directory, device="cpu")
            with patch.object(run, "check"):
                receipt = run.save("fixture.npz", arrays, "npz")
            path = Path(directory) / "fixture.npz"
            with np.load(path, allow_pickle=False) as saved:
                self.assertEqual(set(saved.files), set(arrays))
                for key, expected in arrays.items():
                    np.testing.assert_array_equal(saved[key], expected)
            self.assertEqual(receipt["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertEqual(receipt["size_bytes"], path.stat().st_size)
            self.assertEqual(run.used, receipt["size_bytes"])

    def test_run_receipts_exclusive_writes_and_direct_children(self):
        with tempfile.TemporaryDirectory() as directory:
            run = b.Run(directory, device="cpu")
            with patch.object(run, "check"):
                receipt = run.save("fixture.json", {"count": 3})
                target = Path(directory) / "fixture.json"
                payload = target.read_bytes()
                self.assertEqual(receipt, {"path": "fixture.json", "size_bytes": len(payload),
                                           "sha256": hashlib.sha256(payload).hexdigest()})
                self.assertEqual(run.used, len(payload))
                self.assertEqual(run.receipts, [receipt])
                with self.assertRaises(FileExistsError):
                    run.save("fixture.json", {"count": 99})
                self.assertEqual(target.read_bytes(), payload)
                for name in ("", ".", "..", "../escaped.json", "nested/item.json", "/escaped.json"):
                    with self.subTest(name=name):
                        with self.assertRaisesRegex(RuntimeError, "direct child"):
                            run.save(name, {})
                with self.assertRaisesRegex(RuntimeError, "unknown artifact"):
                    run.save("wrong-kind", {}, "unknown")
                self.assertEqual({p.name for p in Path(directory).iterdir()}, {"fixture.json"})

    def test_run_cumulative_budget_without_allocating_large_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            run = b.Run(directory, device="cpu")
            # Simulate accounted prior bytes; never patch the scientific cap or
            # allocate its GiB-sized value. A JSON empty object needs three bytes.
            run.used = b.MAX_BYTES - b.RESERVE_BYTES - 2
            with patch.object(run, "check"):
                with self.assertRaisesRegex(RuntimeError, "byte cap"):
                    run.save("over-cap.json", {})
            self.assertEqual((Path(directory) / "over-cap.json").stat().st_size, 0)
            self.assertEqual(run.receipts, [])
            self.assertEqual(run.used, 0)  # Actual partial-file accounting after failure.

    def test_run_partial_failure_is_accounted_but_not_receipted(self):
        def partial_failure(value, writer):
            writer.write(b"partial fixture")
            raise ValueError("fixture serialization failure")
        with tempfile.TemporaryDirectory() as directory:
            run = b.Run(directory, device="cpu")
            with patch.object(run, "check"), patch.object(b.torch, "save", side_effect=partial_failure):
                first = run.save("first.json", {"ok": True})
                with self.assertRaisesRegex(ValueError, "fixture serialization"):
                    run.save("partial.pt", object(), "tensor")
            self.assertEqual((Path(directory) / "partial.pt").read_bytes(), b"partial fixture")
            self.assertEqual(run.used, first["size_bytes"] + len(b"partial fixture"))
            self.assertEqual(run.receipts, [first])

    def test_run_exclusive_open_does_not_follow_existing_symlink(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            outside = root / "outside.json"
            outside.write_bytes(b"untouched fixture")
            child = root / "artifacts"
            child.mkdir()
            (child / "link.json").symlink_to(outside)
            run = b.Run(child, device="cpu")
            with patch.object(run, "check"):
                with self.assertRaises(FileExistsError):
                    run.save("link.json", {})
            self.assertEqual(outside.read_bytes(), b"untouched fixture")
            self.assertEqual(run.receipts, [])


class ResourceTests(unittest.TestCase):
    def test_gpu_client_admission_and_rejection(self):
        self.assertEqual(b.validate_gpu_clients("2101, 512\n8861, 128\n"),
                         [{"pid": 2101, "memory_mib": 512}, {"pid": 8861, "memory_mib": 128}])
        self.assertEqual(b.validate_gpu_clients("\n"), [])
        for value in ("9999, 1", "2101, 513", "8861, 129", "2101, -1", "2101, N/A", "bad csv"):
            with self.subTest(value=value):
                with self.assertRaises((RuntimeError, ValueError)):
                    b.validate_gpu_clients(value)

    def test_exact_service_and_cgroup_bounds(self):
        effective = {"memory.max": str(16 * 1024**3), "memory.swap.max": "0", "cpu.max": "100000 100000"}
        service = {"Type": "exec", "RuntimeMaxUSec": "20min", "Restart": "no", "KillMode": "control-group"}
        b.validate_bounds(effective, service)
        for key, value in (("memory.max", "max"), ("memory.max", str(8 * 1024**3)),
                           ("memory.swap.max", "1"), ("cpu.max", "200000 100000"),
                           ("cpu.max", "max 100000"), ("cpu.max", "0 0")):
            with self.subTest(key=key, value=value):
                with self.assertRaises(RuntimeError):
                    b.validate_bounds({**effective, key: value}, service)
        for key, value in (("Type", "simple"), ("RuntimeMaxUSec", "30min"),
                           ("Restart", "always"), ("KillMode", "process")):
            with self.subTest(key=key, value=value):
                with self.assertRaises(RuntimeError):
                    b.validate_bounds(effective, {**service, key: value})

    def test_cooperative_deadline_rejects_before_resource_access(self):
        run = b.Run("/unused", device="cpu")
        with patch.object(b.time, "monotonic", return_value=run.started + b.DEADLINE_SECONDS), \
             patch.object(b.shutil, "disk_usage", side_effect=AssertionError("resource access forbidden")):
            with self.assertRaisesRegex(RuntimeError, "deadline"):
                run.check()

    def test_resource_boundaries_with_no_real_gpu_or_resource_query(self):
        cases = (("cpu", 1024**3 + 1, 16 * 1024**3, 0, None),
                 ("cpu", 1024**3, 0, 0, "free disk"),
                 ("cpu", 1024**3 + 1, 16 * 1024**3 + 1024, 0, "host memory"),
                 ("cuda", 1024**3 + 1, 0, 8 * 1024**3, None),
                 ("cuda", 1024**3 + 1, 0, 8 * 1024**3 + 1, "GPU memory"))
        for device, free, rss_bytes, gpu_bytes, error in cases:
            with self.subTest(device=device, error=error):
                run = b.Run("/unused", device=device)
                with patch.object(b.time, "monotonic", return_value=run.started + 1), \
                     patch.object(b.shutil, "disk_usage", return_value=SimpleNamespace(free=free)), \
                     patch.object(b.resource, "getrusage", return_value=SimpleNamespace(ru_maxrss=rss_bytes // 1024)), \
                     patch.object(b.torch.cuda, "max_memory_allocated", return_value=gpu_bytes) as gpu:
                    if error:
                        with self.assertRaisesRegex(RuntimeError, error):
                            run.check()
                    else:
                        run.check()
                    if device == "cpu":
                        gpu.assert_not_called()


class ComputationContractTests(unittest.TestCase):
    def test_learning_digest_converts_ordered_model_state_to_plain_mapping(self):
        state = OrderedDict((("weight", "fabricated weight"), ("bias", "fabricated bias")))
        optimizer_state = {"state": {}, "param_groups": []}
        model, optimizer = Mock(), Mock()
        model.state_dict.return_value = state
        optimizer.state_dict.return_value = optimizer_state

        def exact_mapping_digest(value):
            self.assertIs(type(value), dict)
            self.assertIs(type(value["model"]), dict)
            self.assertEqual(value, {"model": dict(state), "optimizer": optimizer_state})
            return "fixture digest"

        with patch.object(b.core, "tree_digest", side_effect=exact_mapping_digest):
            self.assertEqual(b.learning_digest(model, optimizer), "fixture digest")
        self.assertIs(type(state), OrderedDict)
        self.assertEqual(list(state), ["weight", "bias"])
        model.state_dict.assert_called_once_with()
        optimizer.state_dict.assert_called_once_with()

    def test_mocked_backward_filter_adam_order_and_raw_no_observer(self):
        for policy in ("raw", "native32"):
            with self.subTest(policy=policy):
                events, diagnostic = [], {"fixture": True}
                model = Mock(side_effect=lambda x: events.append("forward") or "logits")
                optimizer = Mock()
                optimizer.zero_grad.side_effect = lambda **kw: events.append("zero")
                optimizer.step.side_effect = lambda: events.append("Adam")
                loss = Mock()
                loss.backward.side_effect = lambda: events.append("backward")
                loss.detach.return_value = 2.25
                tracker = Mock() if policy == "native32" else None
                if tracker is not None:
                    tracker.filter_grad.side_effect = lambda: events.append("filter") or diagnostic
                with patch.object(b.F, "cross_entropy", side_effect=lambda *a: events.append("loss") or loss), \
                     patch.object(b.torch, "isfinite", return_value=True), \
                     patch.object(b.core, "make_tracker", side_effect=AssertionError("new observer forbidden")):
                    result = b.training_update(model, optimizer, tracker, "inputs", "labels", policy)
                self.assertEqual(events, ["zero", "forward", "loss", "backward"] +
                                 (["filter"] if tracker is not None else []) + ["Adam"])
                self.assertEqual(result, (2.25, diagnostic if tracker is not None else None))
                optimizer.zero_grad.assert_called_once_with(set_to_none=True)
                model.assert_called_once_with("inputs")
                if tracker is not None:
                    tracker.filter_grad.assert_called_once_with()

    def test_update_rejects_policy_tracker_mismatch_and_nonfinite_loss(self):
        for policy, tracker in (("raw", Mock()), ("native32", None), ("unknown", None)):
            optimizer = Mock()
            with self.assertRaisesRegex(RuntimeError, "tracker/policy mismatch"):
                b.training_update(Mock(), optimizer, tracker, None, None, policy)
            optimizer.zero_grad.assert_not_called()
            optimizer.step.assert_not_called()
        optimizer, tracker, loss = Mock(), Mock(), Mock()
        with patch.object(b.F, "cross_entropy", return_value=loss), \
             patch.object(b.torch, "isfinite", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "nonfinite training loss"):
                b.training_update(Mock(), optimizer, tracker, None, None, "native32")
        loss.backward.assert_not_called()
        tracker.filter_grad.assert_not_called()
        optimizer.step.assert_not_called()

    def test_classification_stats_matches_scalar_fixture_and_shift_invariance(self):
        logits = np.zeros((3, 10), dtype=np.float32)
        logits[1, 2], logits[2, 4] = 2, -2
        labels = np.array([3, 2, 4], dtype=np.int64)
        before = logits.copy()
        expected_sum = math.log(10) + math.log(math.exp(2) + 9) - 2 + math.log(math.exp(-2) + 9) + 2
        for values in (logits, logits + np.float32(1000)):
            stats = b.classification_stats(values, labels)
            self.assertEqual((stats["count"], stats["correct"]), (3, 1))
            self.assertEqual(stats["accuracy"], 1 / 3)
            self.assertAlmostEqual(stats["ce_sum"], expected_sum, places=12)
            self.assertAlmostEqual(stats["ce"], expected_sum / 3, places=12)
            json.dumps(stats, allow_nan=False)
        np.testing.assert_array_equal(logits, before)
        np.testing.assert_array_equal(labels, [3, 2, 4])

    def test_classification_rejects_nonfinite_and_wrong_shape(self):
        labels = np.array([0, 1], dtype=np.int64)
        for logits in (np.zeros((2, 9), dtype=np.float32), np.zeros((1, 10), dtype=np.float32),
                       np.full((2, 10), np.nan, dtype=np.float32), np.full((2, 10), np.inf, dtype=np.float32)):
            with self.assertRaisesRegex(RuntimeError, "finite logits required"):
                b.classification_stats(logits, labels)


if __name__ == "__main__":
    unittest.main()
