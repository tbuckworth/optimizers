"""Dataset-free CPU fixtures for the cooperative I7 runtime guard."""
import importlib.util
import math
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import time
import unittest
from unittest import mock

if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
    raise RuntimeError("Run runtime-guard fixtures with CUDA_VISIBLE_DEVICES='' explicitly")

import torch

spec = importlib.util.spec_from_file_location("i7_runtime_guard", Path(__file__).with_name("runtime_guard.py"))
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)


class Sequence:
    def __init__(self, *values):
        self.values = list(values)
        self.calls = 0

    def __call__(self):
        self.calls += 1
        if not self.values:
            raise AssertionError("probe called after terminal failure")
        return self.values.pop(0)


def limits(wall=10.0, cpu=5.0, rss=100, cuda=50):
    return {"wall_seconds":wall, "cpu_seconds":cpu,
            "rss_bytes":rss, "cuda_allocated_bytes":cuda}


def cuda(allocated=0, reserved=0, device=None):
    return {"allocated_bytes":allocated, "reserved_bytes":reserved, "device":device}


def fixture_guard(phase="audit", *, wall=(0, 1), cpu=(0, 1), rss=lambda:0,
                  gpu=lambda:cuda(), caps=None, origins=None):
    origin_kwargs = {} if origins is None else {
        "entry_wall_origin":origins[0], "entry_cpu_origin":origins[1],
        "entry_process_id":os.getpid()}
    return r.RuntimeGuard(phase, profile=r.FIXTURE, wall_clock=Sequence(*wall),
                          cpu_clock=Sequence(*cpu), rss_probe=rss, cuda_probe=gpu,
                          limits=limits() if caps is None else caps, **origin_kwargs)


class RuntimeTests(unittest.TestCase):
    def test_fixture_scope_is_cpu_only_single_threaded_and_on_verified_tmp(self):
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                     "NUMEXPR_NUM_THREADS"):
            self.assertEqual(os.environ.get(name), "1")
        self.assertEqual(torch.get_num_threads(), 1)
        self.assertEqual(Path(os.environ["TMPDIR"]).resolve(), Path("/tmp/spectral-experiment-artifacts"))
        self.assertFalse(torch.cuda.is_initialized())

    def test_exact_boundaries_pass_and_actual_overrun_is_terminal(self):
        wall, cpu = Sequence(0, 10, 10.25), Sequence(0, 5, 5)
        guard = r.RuntimeGuard("audit", profile=r.FIXTURE, wall_clock=wall, cpu_clock=cpu,
                               rss_probe=lambda:100, cuda_probe=lambda:cuda(50, 80),
                               limits=limits())
        passed = guard.check("exact-boundary")
        self.assertEqual((passed["wall_seconds"], passed["cpu_seconds"], passed["status"]),
                         (10, 5, "pass"))
        with self.assertRaises(r.GuardTerminal) as caught:
            guard("after-blocking-operation")
        self.assertEqual(caught.exception.record["wall_seconds"], 10.25)
        self.assertEqual(caught.exception.record["violations"], ["wall_seconds"])
        calls = (wall.calls, cpu.calls)
        with self.assertRaises(r.GuardTerminal):
            guard.check("no-retry")
        self.assertEqual((wall.calls, cpu.calls), calls)
        copy = guard.failure_info()
        copy["stage"] = "mutated"
        self.assertEqual(guard.failure_info()["stage"], "after-blocking-operation")

    def test_entry_origins_include_preconstruction_cost_and_never_reset(self):
        wall, cpu = Sequence(5, 6, 9), Sequence(7, 8, 10)
        guard = r.RuntimeGuard(
            "primary", profile=r.FIXTURE, wall_clock=wall, cpu_clock=cpu,
            rss_probe=lambda:0, cuda_probe=lambda:cuda(),
            limits=limits(wall=20, cpu=None),
            entry_wall_origin=2, entry_cpu_origin=3,
            entry_process_id=os.getpid())
        self.assertEqual((wall.calls, cpu.calls), (1, 1))
        origins = guard.entry_origins
        origins["wall_origin"] = 999
        self.assertEqual(guard.entry_origins, {"wall_origin":2.0, "cpu_origin":3.0})
        prepared = guard.check("primary-plan-preparation")
        gone = guard.check("primary-go")
        self.assertEqual((prepared["wall_seconds"], prepared["cpu_seconds"]), (4, 5))
        self.assertEqual((gone["wall_seconds"], gone["cpu_seconds"]), (7, 7))
        self.assertEqual((wall.calls, cpu.calls), (3, 3))
        self.assertEqual(guard.entry_origins, {"wall_origin":2.0, "cpu_origin":3.0})
        self.assertNotIn("origin", repr(guard.summary()).lower())

    def test_entry_origin_can_expire_cap_at_first_check(self):
        guard = fixture_guard(
            "primary", wall=(9, 10.01), cpu=(4, 4.5), origins=(0, 0),
            caps=limits(wall=10, cpu=None))
        with self.assertRaises(r.GuardTerminal) as caught:
            guard.check("first-check")
        self.assertEqual(caught.exception.record["wall_seconds"], 10.01)
        self.assertEqual(caught.exception.record["violations"], ["wall_seconds"])

    def test_entry_origins_reject_missing_malformed_and_future_values(self):
        base = {"profile":r.FIXTURE, "wall_clock":Sequence(5),
                "cpu_clock":Sequence(7), "rss_probe":lambda:0,
                "cuda_probe":lambda:cuda(), "limits":limits()}
        for kwargs in ({"entry_wall_origin":0}, {"entry_cpu_origin":0},
                       {"entry_process_id":os.getpid()}):
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(r.GuardError, "together"):
                r.RuntimeGuard("audit", **base, **kwargs)
        for wall_origin, cpu_origin in (
                (True, 0), (math.inf, 0), (-1, 0), (0, "0")):
            with self.subTest(origins=(wall_origin, cpu_origin)), self.assertRaises(r.GuardError):
                r.RuntimeGuard("audit", **base, entry_wall_origin=wall_origin,
                               entry_cpu_origin=cpu_origin, entry_process_id=os.getpid())
        for process_id in (True, 0, -1, os.getpid() + 1):
            with self.subTest(process_id=process_id), self.assertRaises(r.GuardError):
                r.RuntimeGuard(
                    "audit", **base, entry_wall_origin=0, entry_cpu_origin=0,
                    entry_process_id=process_id)
        for wall_origin, cpu_origin in ((5.01, 7), (5, 7.01)):
            wall, cpu = Sequence(5), Sequence(7)
            with self.subTest(origins=(wall_origin, cpu_origin)), \
                    self.assertRaisesRegex(r.GuardError, "later"):
                r.RuntimeGuard(
                    "audit", profile=r.FIXTURE, wall_clock=wall, cpu_clock=cpu,
                    rss_probe=lambda:0, cuda_probe=lambda:cuda(), limits=limits(),
                    entry_wall_origin=wall_origin, entry_cpu_origin=cpu_origin,
                    entry_process_id=os.getpid())
            self.assertEqual((wall.calls, cpu.calls), (1, 1))

    def test_guard_rejects_check_after_process_identity_changes(self):
        wall, cpu = Sequence(5, 6), Sequence(7, 8)
        guard = r.RuntimeGuard(
            "audit", profile=r.FIXTURE, wall_clock=wall, cpu_clock=cpu,
            rss_probe=lambda:0, cuda_probe=lambda:cuda(), limits=limits(),
            entry_wall_origin=2, entry_cpu_origin=3,
            entry_process_id=os.getpid())
        with mock.patch.object(r.os, "getpid", return_value=os.getpid() + 1), \
                self.assertRaises(r.GuardTerminal) as caught:
            guard.check("after-fork")
        self.assertEqual(caught.exception.record["probe_error"], "process_identity_changed")
        self.assertEqual((wall.calls, cpu.calls), (1, 1))

    def test_entry_origin_does_not_mask_postconstructor_clock_regression(self):
        guard = fixture_guard(wall=(5, 4), cpu=(7, 8), origins=(2, 3))
        with self.assertRaises(r.GuardTerminal) as caught:
            guard.check("regression")
        self.assertEqual(caught.exception.record["wall_seconds"], 2)
        self.assertEqual(caught.exception.record["cpu_seconds"], 5)
        self.assertEqual(caught.exception.record["probe_error"], "clock_regression")

    def test_wall_cpu_rss_and_cuda_caps_are_independent(self):
        cases = [
            ("primary", (0, 10.01), (0, 999), lambda:0, lambda:cuda(), limits(wall=10, cpu=None), "wall_seconds"),
            ("audit", (0, 1), (0, 5.01), lambda:0, lambda:cuda(), limits(), "cpu_seconds"),
            ("audit", (0, 1), (0, 1), lambda:101, lambda:cuda(), limits(), "peak_rss_bytes"),
            ("audit", (0, 1), (0, 1), lambda:0, lambda:cuda(51, 100), limits(), "peak_cuda_allocated_bytes"),
        ]
        for phase, wall, cpu_values, rss, gpu, caps, expected in cases:
            with self.subTest(expected=expected):
                guard = fixture_guard(phase, wall=wall, cpu=cpu_values, rss=rss, gpu=gpu, caps=caps)
                with self.assertRaises(r.GuardTerminal) as caught:
                    guard.check("cap")
                self.assertIn(expected, caught.exception.record["violations"])
                self.assertIsNotNone(caught.exception.record["wall_seconds"])
                self.assertIsNotNone(caught.exception.record["cpu_seconds"])

    def test_time_regression_and_probe_failures_retain_bounded_state(self):
        guard = fixture_guard(wall=(5, 4), cpu=(2, 2))
        with self.assertRaises(r.GuardTerminal) as caught:
            guard.check("regression")
        self.assertEqual(caught.exception.record["probe_error"], "clock_regression")
        self.assertEqual(caught.exception.record["wall_seconds"], -1)

        def broken_rss():
            raise OSError("x" * 1000)

        failed = fixture_guard(rss=broken_rss)
        with self.assertRaises(r.GuardTerminal):
            failed.check("resource-probe")
        info = failed.failure_info()
        self.assertLessEqual(len(info["probe_error"]), 256)
        self.assertEqual(info["stage"], "resource-probe")
        self.assertEqual(info["wall_seconds"], 1)

    def test_invalid_stage_is_bounded_terminal_and_inputs_fail_closed(self):
        wall, cpu = Sequence(0), Sequence(0)
        guard = fixture_guard(wall=wall.values, cpu=cpu.values)
        with self.assertRaises(r.GuardTerminal) as caught:
            guard.check("\N{SNOWMAN}" * 1000)
        self.assertEqual(caught.exception.record["stage"], "<invalid-stage>")
        self.assertEqual(caught.exception.record["violations"], ["invalid_stage"])
        self.assertLessEqual(len(repr(caught.exception.record)), 1024)
        with self.assertRaises(r.GuardTerminal):
            guard.check("later")
        with self.assertRaises(r.GuardError):
            fixture_guard(caps=limits(wall=10 ** 10000))
        for name in ("wall_clock", "cpu_clock", "rss_probe", "cuda_probe"):
            kwargs = {"wall_clock":lambda:0, "cpu_clock":lambda:0,
                      "rss_probe":lambda:0, "cuda_probe":lambda:cuda(),
                      "limits":limits()}
            kwargs[name] = 1
            with self.subTest(probe=name), self.assertRaisesRegex(r.GuardError, "callable"):
                r.RuntimeGuard("audit", profile=r.FIXTURE, **kwargs)

    def test_peak_tracking_reservation_and_bad_probe(self):
        readings = iter((cuda(10, 30), cuda(20, 80)))
        guard = fixture_guard(wall=(0, 1, 2), cpu=(0, 1, 2), gpu=lambda:next(readings))
        guard.check("one")
        row = guard.check("two")
        self.assertEqual((row["peak_cuda_allocated_bytes"], row["peak_cuda_reserved_bytes"]), (20, 80))
        self.assertEqual(guard.summary()["cpu_scope"],
                         "process_time for this process; child processes excluded")

        bad = fixture_guard(gpu=lambda:{"allocated_bytes":2, "reserved_bytes":1, "device":None})
        with self.assertRaises(r.GuardTerminal) as caught:
            bad.check("bad-cuda")
        self.assertIn("exceeds reservation", caught.exception.record["probe_error"])

    def test_fixed_phase_caps_and_scientific_override_rejection(self):
        self.assertEqual(r.PHASE_CAPS, {
            "development":{"wall_seconds":180.0, "cpu_seconds":None},
            "primary":{"wall_seconds":600.0, "cpu_seconds":None},
            "sensitivity":{"wall_seconds":240.0, "cpu_seconds":None},
            "audit":{"wall_seconds":600.0, "cpu_seconds":600.0},
        })
        with self.assertRaises(TypeError):
            r.PHASE_CAPS["audit"]["wall_seconds"] = 999
        guard = fixture_guard()
        copied_limits = guard.limits
        copied_limits["wall_seconds"] = 999
        self.assertEqual(guard.limits["wall_seconds"], 10)
        with self.assertRaisesRegex(r.GuardError, "forbids"):
            r.RuntimeGuard("primary", profile=r.SCIENTIFIC, wall_clock=lambda:0)
        with self.assertRaisesRegex(r.GuardError, "forbids"):
            r.verify_big_volume("/missing", profile=r.SCIENTIFIC, runner=lambda *a, **k:None)

    def test_fixture_and_scientific_audit_are_cuda_inert(self):
        self.assertFalse(torch.cuda.is_initialized())
        fixture_guard().check("fixture")
        self.assertFalse(torch.cuda.is_initialized())
        plain_audit = r.RuntimeGuard("audit", profile=r.SCIENTIFIC)
        self.assertIsNone(plain_audit.entry_origins)
        entry_wall, entry_cpu, entry_pid = time.monotonic(), time.process_time(), os.getpid()
        audit = r.RuntimeGuard("audit", profile=r.SCIENTIFIC,
                               entry_wall_origin=entry_wall, entry_cpu_origin=entry_cpu,
                               entry_process_id=entry_pid)
        self.assertEqual(audit.entry_origins,
                         {"wall_origin":entry_wall, "cpu_origin":entry_cpu})
        row = audit.check("cpu-audit")
        self.assertEqual((row["peak_cuda_allocated_bytes"], row["peak_cuda_reserved_bytes"]), (0, 0))
        self.assertIsNone(row["device"])
        self.assertEqual(row["runtime"]["torch_version"], "2.11.0+cu128")
        self.assertFalse(row["cuda_initialized"])
        self.assertFalse(audit.summary()["cuda_initialized"])
        self.assertFalse(torch.cuda.is_initialized())
        with self.assertRaisesRegex(r.GuardError, "already be initialized"):
            r.RuntimeGuard("primary", profile=r.SCIENTIFIC)
        self.assertFalse(torch.cuda.is_initialized())

    def test_mock_native_identity_requires_complete_canonical_identifier(self):
        def identity(uuid):
            props = SimpleNamespace(name=r.CUDA_DEVICE_NAME, total_memory=24 * r.GIB,
                                    uuid=uuid)
            with mock.patch.object(r.torch.cuda, "is_initialized", return_value=True), \
                    mock.patch.object(r.torch.cuda, "current_device", return_value=0), \
                    mock.patch.object(r.torch.cuda, "get_device_properties", return_value=props):
                return r._native_cuda_identity()
        for value in ("   ", None, "GPU-fixture"):
            with self.subTest(value=value), self.assertRaisesRegex(r.GuardError, "UUID"):
                identity(value)
        bare = "994B97CD-D768-1F03-78FB-6A70E3CC1E0C"
        observed = identity(bare)
        self.assertTrue(observed["stable_identity_verified"])
        self.assertEqual(observed["stable_identity"], "GPU-" + bare.lower())
        self.assertFalse(torch.cuda.is_initialized())


class MountTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = self.root / "study-parent"
        self.path.mkdir()
        info = os.stat(self.path)
        self.device = (os.major(info.st_dev), os.minor(info.st_dev))
        self.expected = {"target":str(self.root), "source":"/dev/fixture",
                         "uuid":"fixture-uuid"}

    def runner(self, *, target=None, source=None, uuid=None, device=None):
        target = self.expected["target"] if target is None else target
        source = self.expected["source"] if source is None else source
        uuid = self.expected["uuid"] if uuid is None else uuid
        device = self.device if device is None else device

        def run(*args, **kwargs):
            self.assertEqual(kwargs, {"capture_output":True, "text":True,
                                      "check":True, "timeout":5})
            return SimpleNamespace(stdout=f"{target} {source} {uuid} {device[0]}:{device[1]}\n")
        return run

    def verify(self, **kwargs):
        return r.verify_big_volume(self.path, profile=r.FIXTURE,
                                   runner=kwargs.pop("runner", self.runner()),
                                   expected=kwargs.pop("expected", self.expected), **kwargs)

    def test_verified_mount_and_device_metadata(self):
        row = self.verify()
        self.assertEqual((row["target"], row["source"], row["uuid"]),
                         (str(self.root), "/dev/fixture", "fixture-uuid"))
        self.assertEqual((row["major"], row["minor"]), self.device)
        self.assertTrue(row["read_only_verification"])

    def test_wrong_mount_root_source_uuid_and_device_rejected(self):
        wrongs = [
            self.runner(target="/"), self.runner(source="/dev/wrong"),
            self.runner(uuid="wrong"), self.runner(device=(self.device[0], self.device[1] + 1)),
        ]
        for runner in wrongs:
            with self.subTest(output=runner):
                with self.assertRaises(r.GuardError):
                    self.verify(runner=runner)
        outside = dict(self.expected, target=str(self.root / "different"))
        with self.assertRaisesRegex(r.GuardError, "outside"):
            self.verify(expected=outside)

    def test_symlink_and_invalid_stat_metadata_rejected(self):
        link = self.root / "link"
        link.symlink_to(self.path, target_is_directory=True)
        with self.assertRaisesRegex(r.GuardError, "symlink"):
            r.verify_big_volume(link, profile=r.FIXTURE, runner=self.runner(), expected=self.expected)
        with self.assertRaisesRegex(r.GuardError, "device metadata"):
            self.verify(stat_probe=lambda path:SimpleNamespace(st_dev=True))

    def test_timeout_and_noncanonical_device_metadata_rejected(self):
        def timed_out(*args, **kwargs):
            raise __import__("subprocess").TimeoutExpired("findmnt", 5)
        with self.assertRaisesRegex(r.GuardError, "timed out"):
            self.verify(runner=timed_out)
        for st_dev in (-1, 1.5, "1", True):
            with self.subTest(st_dev=st_dev), self.assertRaisesRegex(r.GuardError, "device metadata"):
                self.verify(stat_probe=lambda path, value=st_dev:SimpleNamespace(st_dev=value))
        for device in (("01", "0"), ("+1", "0"), ("-1", "0"), ("\N{ARABIC-INDIC DIGIT ONE}", "0")):
            with self.subTest(device=device), self.assertRaisesRegex(r.GuardError, "device number"):
                self.verify(runner=self.runner(device=device))


if __name__ == "__main__":
    unittest.main()
