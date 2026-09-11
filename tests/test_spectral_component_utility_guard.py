"""Fabricated CPU/IO tests only: every configure observation is mocked."""
from contextlib import contextmanager, ExitStack
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from experiments import spectral_component_utility_guard as guard


class WriterTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.directory = Path(self.stack.enter_context(tempfile.TemporaryDirectory())).resolve()
        self.clock = 1000.0
        self.stack.enter_context(mock.patch.object(guard.time, "monotonic", side_effect=lambda: self.clock))
        self.disk = self.stack.enter_context(mock.patch.object(
            guard.shutil, "disk_usage", return_value=SimpleNamespace(free=64 * 1024**3)))
        self.rss = self.stack.enter_context(mock.patch.object(
            guard.resource, "getrusage", return_value=SimpleNamespace(ru_maxrss=1024)))
        self.gpu = self.stack.enter_context(mock.patch.object(
            guard.torch.cuda, "max_memory_allocated", side_effect=AssertionError("no GPU queries")))
        self.run = self.stack.enter_context(guard.Run(self.directory, device="cpu"))

    def assert_bytes_accounted(self):
        self.assertEqual(self.run.used, sum(p.stat().st_size for p in self.directory.iterdir()))
        self.assertLessEqual(self.run.used, guard.MAX_BYTES)

    def test_json_receipt_hash_copy_and_final_receipt_exclusion(self):
        receipt = self.run.save("first.json", {"finite": 1.5, "text": "λ", "list": [None, True, 2]})
        payload = (self.directory / "first.json").read_bytes()
        self.assertEqual(receipt, {"path": "first.json", "size_bytes": len(payload),
                                   "sha256": hashlib.sha256(payload).hexdigest()})
        receipt["path"] = "corrupted"
        copy = self.run.receipts
        copy[0]["sha256"] = "corrupted"
        self.assertEqual(self.run.receipts[0]["path"], "first.json")
        self.assertNotEqual(self.run.receipts[0]["sha256"], "corrupted")
        final = {"receipts": self.run.receipts, "status": "complete"}
        self.run.save("results.json", final)
        self.assertEqual([r["path"] for r in json.loads((self.directory / "results.json").read_text())["receipts"]],
                         ["first.json"])
        self.assertEqual(len(self.run.receipts), 2)
        self.assert_bytes_accounted()
        self.gpu.assert_not_called()

    def test_npz_numeric_roundtrip_including_empty_noncontiguous_and_bool(self):
        arrays = {"matrix": np.arange(24, dtype=np.float32).reshape(4, 6)[:, ::2],
                  "empty": np.empty((7, 0), dtype=np.float64),
                  "ids": np.arange(5, dtype=np.int64), "mask": np.array([True, False])}
        receipt = self.run.save("numeric.npz", arrays, kind="npz")
        with np.load(self.directory / "numeric.npz", allow_pickle=False) as saved:
            self.assertEqual(set(saved.files), set(arrays))
            for name, expected in arrays.items():
                np.testing.assert_array_equal(saved[name], expected)
                self.assertEqual(saved[name].dtype, expected.dtype)
        self.assertGreater(receipt["size_bytes"], sum(a.nbytes for a in arrays.values()))
        self.assert_bytes_accounted()

    def test_invalid_json_before_any_write_and_no_resume(self):
        cycle = []
        cycle.append(cycle)
        bad = [float("nan"), float("inf"), {1: "not a string"}, {"v": np.int64(1)},
               (1, 2), cycle, 2**300]
        for i, value in enumerate(bad):
            with self.subTest(i=i), tempfile.TemporaryDirectory() as directory:
                with guard.Run(Path(directory).resolve(), device="cpu") as run:
                    with self.assertRaises(guard.GuardError):
                        run.save("bad.json", value)
                    self.assertEqual(list(Path(directory).iterdir()), [])
                    with self.assertRaisesRegex(guard.GuardError, "cannot resume"):
                        run.save("next.json", {})

    def test_invalid_npz_before_write(self):
        invalid = [{"x": np.array([object()], dtype=object)}, {"x": np.array(["text"])},
                   {"x": np.array([complex(1, 2)])}, {"x": np.array([float("nan")])},
                   {"x": np.zeros(1, dtype=[("field", "f4")])}, {"x": [1, 2]},
                   {"../path": np.zeros(1)}, {"file": np.zeros(1)}, {},
                   {"x": np.zeros(1, dtype=np.dtype("f4", metadata={"arbitrary": "not archived"}))},
                   {"x": np.zeros(1).view(np.ma.MaskedArray)}]
        for i, value in enumerate(invalid):
            with self.subTest(i=i), tempfile.TemporaryDirectory() as directory:
                with guard.Run(Path(directory).resolve(), device="cpu") as run:
                    with self.assertRaises(guard.GuardError):
                        run.save("bad.npz", value, kind="npz")
                    self.assertEqual(list(Path(directory).iterdir()), [])

    def test_direct_child_names_and_kind(self):
        for name, kind in [("../x.json", "json"), ("/x.json", "json"), ("a/b.json", "json"),
                           ("x.npz", "json"), ("failure.json", "json"),
                           ("x.pt", "tensor"), ("a\n.json", "json")]:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                with guard.Run(Path(directory).resolve(), device="cpu") as run:
                    with self.assertRaises(guard.GuardError):
                        run.save(name, {}, kind)
                    self.assertEqual(list(Path(directory).iterdir()), [])

    def test_constructor_requires_empty_real_absolute_directory(self):
        with self.assertRaises(guard.GuardError):
            guard.Run("relative", device="cpu")
        self.run.save("existing.json", {})
        with self.assertRaises(guard.GuardError):
            guard.Run(self.directory, device="cpu")
        link = self.directory / "link"
        link.symlink_to(self.directory, target_is_directory=True)
        with self.assertRaises(guard.GuardError):
            guard.Run(link, device="cpu")
        with self.assertRaises(guard.GuardError):
            guard.Run(self.directory, device="cuda")

    def test_no_overwrite(self):
        first = self.run.save("original.json", {"first": True})
        payload = (self.directory / "original.json").read_bytes()
        with self.assertRaises(FileExistsError):
            self.run.save("original.json", {"first": False})
        self.assertEqual((self.directory / "original.json").read_bytes(), payload)
        self.assertEqual(self.run.receipts, [first])
        self.assert_bytes_accounted()

    def test_external_symlink_or_file_cannot_be_written_through(self):
        with tempfile.TemporaryDirectory() as outside:
            target = Path(outside) / "target.json"
            target.write_bytes(b"preserve")
            (self.directory / "linked.json").symlink_to(target)
            with self.assertRaisesRegex(guard.GuardError, "nonregular"):
                self.run.save("linked.json", {})
            self.assertEqual(target.read_bytes(), b"preserve")

    def test_external_regular_file_consumes_budget_and_blocks_science(self):
        (self.directory / "external.json").write_bytes(b"123456789")
        with self.assertRaisesRegex(guard.GuardError, "inventory changed"):
            self.run.save("attempt.json", {})
        self.assertEqual(self.run.used, 9)
        self.run.write_failure(RuntimeError("external file"))
        self.assert_bytes_accounted()

    def test_json_per_file_cap_preserves_partial_and_footer(self):
        with mock.patch.object(guard, "JSON_MAX_BYTES", 50):
            with self.assertRaisesRegex(guard.GuardError, "byte cap"):
                self.run.save("oversized.json", {"many": ["abc"] * 100})
        partial = self.directory / "oversized.json"
        self.assertTrue(partial.exists())
        self.assertTrue(0 < partial.stat().st_size <= 50)
        self.assertEqual(self.run.receipts, [])
        self.run.write_failure(RuntimeError("file too big"))
        self.assertTrue(partial.exists())
        self.assert_bytes_accounted()

    def test_npz_per_file_cap_includes_zip_overhead(self):
        with mock.patch.object(guard, "NPZ_MAX_BYTES", 200):
            with self.assertRaisesRegex(guard.GuardError, "byte cap"):
                self.run.save("oversized.npz", {"x": np.zeros(30, dtype=np.float32)}, "npz")
            self.assertTrue((self.directory / "oversized.npz").exists())
            self.assertLessEqual((self.directory / "oversized.npz").stat().st_size, 200)
        self.assert_bytes_accounted()

    def test_whole_archive_cap_and_reserved_exclusive_failure_footer(self):
        with mock.patch.object(guard, "MAX_BYTES", 8192), mock.patch.object(guard, "RESERVE_BYTES", 4096):
            self.run.save("first.json", {"text": "a" * 3000})
            with self.assertRaisesRegex(guard.GuardError, "byte cap"):
                self.run.save("second.json", {"text": "b" * 3000})
            self.assertLessEqual(self.run.used, 4096)
            self.clock += 1300
            receipt = self.run.write_failure(guard.GuardError("expired"))
            self.assertLessEqual(receipt["size_bytes"], 4096)
            self.assert_bytes_accounted()
            payload = (self.directory / "failure.json").read_bytes()
            with self.assertRaisesRegex(guard.GuardError, "already attempted"):
                self.run.write_failure(RuntimeError("again"))
            self.assertEqual(payload, (self.directory / "failure.json").read_bytes())
            with self.assertRaisesRegex(guard.GuardError, "cannot resume"):
                self.run.save("new.json", {})

    def test_deadline_is_checked_even_inside_resource_rate_limit(self):
        self.run.check()
        self.clock = self.run.started + guard.DEADLINE_SECONDS
        with self.assertRaisesRegex(guard.GuardError, "deadline"):
            self.run.save("expired.json", {})
        self.assertEqual(list(self.directory.iterdir()), [])
        self.run.write_failure(TimeoutError("deadline"))

    def test_host_disk_and_mocked_gpu_caps(self):
        self.rss.return_value.ru_maxrss = guard.HOST_MAX_BYTES // 1024 + 1
        with self.assertRaisesRegex(guard.GuardError, "host"):
            self.run.check()
        self.rss.return_value.ru_maxrss = 1024
        with self.assertRaisesRegex(guard.GuardError, "cannot resume"):
            self.run.check()
        self.disk.return_value.free = guard.DISK_MIN_FREE_BYTES - 1
        with tempfile.TemporaryDirectory() as directory:
            with guard.Run(Path(directory).resolve(), device="cpu") as run:
                with self.assertRaisesRegex(guard.GuardError, "free-disk"):
                    run.check()
        self.disk.return_value.free = guard.DISK_MIN_FREE_BYTES
        self.gpu.side_effect = None
        self.gpu.return_value = guard.GPU_MAX_BYTES + 1
        with tempfile.TemporaryDirectory() as directory:
            with guard.Run(Path(directory).resolve(), device="cuda:0") as run:
                with self.assertRaisesRegex(guard.GuardError, "allocator"):
                    run.check()
        self.gpu.assert_called_once_with(device="cuda:0")

    def test_closed_or_replaced_directory_rejected(self):
        moved = self.directory.with_name(self.directory.name + "-moved")
        self.directory.rename(moved)
        try:
            self.directory.mkdir()
            with self.assertRaisesRegex(guard.GuardError, "identity changed"):
                self.run.check()
        finally:
            self.directory.rmdir()
            moved.rename(self.directory)
        self.run.close()
        with tempfile.TemporaryDirectory() as directory:
            run = guard.Run(Path(directory).resolve(), device="cpu")
            run.close()
            with self.assertRaisesRegex(guard.GuardError, "closed"):
                run.check()

    def test_short_write_accounting_and_cap_before_write(self):
        handle = mock.Mock()
        handle.write.return_value = 2
        accounted = []
        writer = guard._BudgetWriter(handle, 5, lambda: None, accounted.append)
        with self.assertRaisesRegex(guard.GuardError, "short"):
            writer.write(b"abcd")
        self.assertEqual(accounted, [2])
        self.assertEqual(writer.allowance, 3)
        with self.assertRaisesRegex(guard.GuardError, "byte cap"):
            writer.write(b"abcd")
        self.assertEqual(handle.write.call_count, 1)

    def test_failure_footer_preserves_serialization_time_and_bounds_message(self):
        def tick(_value):
            self.clock += 0.1
        with mock.patch.object(guard.os, "fsync", side_effect=tick):
            self.run.save("timed.json", {})
        self.assertAlmostEqual(self.run.serialization_seconds, 0.1)
        self.run.write_failure(RuntimeError("z" * 10000))
        footer = json.loads((self.directory / "failure.json").read_text())
        self.assertEqual(len(footer["message"]), 4096)
        self.assertAlmostEqual(footer["serialization_seconds"], 0.1)
        self.assertEqual(footer["status"], "failed")


class ConfigureTests(unittest.TestCase):
    def test_import_is_inert(self):
        spec = importlib.util.spec_from_file_location("_inert_guard_fixture", guard.__file__)
        module = importlib.util.module_from_spec(spec)
        with ExitStack() as stack:
            forbidden = []
            for owner, name in [(guard.subprocess, "check_output"), (Path, "read_text"),
                                (guard.os, "open"), (guard.torch, "set_num_threads"),
                                (guard.torch, "set_num_interop_threads"),
                                (guard.torch, "use_deterministic_algorithms"),
                                (guard.torch.cuda, "is_available"), (guard.torch.cuda, "get_device_name"),
                                (guard.torch.cuda, "mem_get_info"), (guard.torch.cuda, "max_memory_allocated")]:
                forbidden.append(stack.enter_context(mock.patch.object(owner, name, side_effect=AssertionError("inert import"))))
            spec.loader.exec_module(module)
            for method in forbidden:
                method.assert_not_called()

    @contextmanager
    def fabricated(self, *, change=None, clients="2101, 512\n8861, 128\n34567, 12\n"):
        pid, invocation = 34567, "a" * 32
        group = "/user.slice/user-1000.slice/user@1000.service/app.slice/" + guard.UNIT
        effective = {"memory.max": str(guard.HOST_MAX_BYTES), "memory.swap.max": "0",
                     "cpu.max": "100000 100000"}
        service = {"Type": "exec", "RuntimeMaxUSec": "30min", "Restart": "no",
                   "KillMode": "control-group", "MainPID": str(pid), "InvocationID": invocation,
                   "ActiveState": "active", "SubState": "running", "ControlGroup": group}
        env = {**guard.THREAD_ENV, "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "INVOCATION_ID": invocation}
        fake = SimpleNamespace(
            __version__="2.11.0+cu128", set_num_threads=mock.Mock(), set_num_interop_threads=mock.Mock(),
            use_deterministic_algorithms=mock.Mock(), get_num_threads=mock.Mock(return_value=1),
            get_num_interop_threads=mock.Mock(return_value=1),
            are_deterministic_algorithms_enabled=mock.Mock(return_value=True),
            backends=SimpleNamespace(cudnn=SimpleNamespace(benchmark=True, allow_tf32=True),
                                     cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=True))),
            cuda=SimpleNamespace(is_available=mock.Mock(return_value=True),
                                 get_device_name=mock.Mock(return_value="NVIDIA GeForce RTX 3090"),
                                 mem_get_info=mock.Mock(return_value=(12 * 1024**3, 24 * 1024**3)),
                                 memory=SimpleNamespace(set_per_process_memory_fraction=mock.Mock())))
        cfg = {"group": group, "cgroup_lines": "0::" + group + "\n", "effective": effective,
               "service": service, "env": env, "torch": fake, "numpy_version": "1.26.4",
               "clients": clients}
        if change:
            change(cfg)

        def read_text(path, *args, **kwargs):
            if str(path) == "/proc/self/cgroup":
                return cfg["cgroup_lines"]
            if path.parent == Path("/sys/fs/cgroup") / cfg["group"].lstrip("/"):
                return cfg["effective"][path.name] + "\n"
            raise AssertionError("unexpected filesystem read: " + str(path))

        def output(command, **kwargs):
            self.assertEqual(kwargs, {"text": True, "timeout": 10})
            if command[0] == "systemctl":
                self.assertEqual(command[:4], ["systemctl", "--user", "show", guard.UNIT])
                self.assertEqual(set(command[4:]), {"--property=" + key for key in guard.SERVICE_KEYS})
                return "\n".join(key + "=" + value for key, value in cfg["service"].items())
            self.assertEqual(command, ["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory",
                                       "--format=csv,noheader,nounits", "--id=0"])
            return cfg["clients"]

        with mock.patch.object(guard, "torch", fake), mock.patch.object(guard.np, "__version__", cfg["numpy_version"]), \
                mock.patch.dict(guard.os.environ, cfg["env"], clear=True), \
                mock.patch.object(guard.os, "getpid", return_value=pid), \
                mock.patch.object(Path, "read_text", read_text), \
                mock.patch.object(guard.subprocess, "check_output", side_effect=output) as subprocess_mock:
            yield cfg, subprocess_mock

    def test_exact_receipt_and_allocator_configuration(self):
        with self.fabricated() as (cfg, called):
            receipt = guard.configure()
            self.assertEqual(set(receipt), {"unit", "pid", "invocation_id", "cgroup", "effective", "service",
                                           "gpu_clients", "preexisting_gpu_clients", "device", "device_name",
                                           "gpu_free_bytes_at_configure", "gpu_total_bytes",
                                           "gpu_allocator_limit_bytes", "gpu_allocator_fraction", "torch_version",
                                           "numpy_version", "deterministic"})
            self.assertEqual(receipt["pid"], 34567)
            self.assertEqual(receipt["service"], cfg["service"])
            self.assertEqual(receipt["effective"], cfg["effective"])
            self.assertEqual(receipt["preexisting_gpu_clients"], [{"pid": 2101, "memory_mib": 512},
                                                                 {"pid": 8861, "memory_mib": 128}])
            self.assertEqual(receipt["gpu_allocator_limit_bytes"], 4 * 1024**3)
            self.assertEqual(receipt["gpu_allocator_fraction"], 1 / 6)
            cfg["torch"].cuda.memory.set_per_process_memory_fraction.assert_called_once_with(1 / 6, device="cuda:0")
            self.assertEqual(receipt["deterministic"], {"thread_env": guard.THREAD_ENV,
                             "cublas_workspace": ":4096:8", "intraop_threads": 1, "interop_threads": 1,
                             "deterministic_algorithms": True, "cudnn_benchmark": False,
                             "matmul_allow_tf32": False, "cudnn_allow_tf32": False})
            self.assertEqual(called.call_count, 2)

    def test_cgroup_caps_and_membership_fail_closed(self):
        changes = [lambda c: c["effective"].update({"memory.max": "max"}),
                   lambda c: c["effective"].update({"memory.swap.max": "1"}),
                   lambda c: c["effective"].update({"cpu.max": "200000 100000"}),
                   lambda c: c.update(cgroup_lines="0::/another.service\n"),
                   lambda c: c.update(cgroup_lines="0::" + c["group"] + "\n1:cpu:/elsewhere\n"),
                   lambda c: c.update(cgroup_lines="0::/../" + guard.UNIT + "\n")]
        for i, change in enumerate(changes):
            with self.subTest(i=i), self.fabricated(change=change) as (cfg, _):
                with self.assertRaises(guard.GuardError):
                    guard.configure()
                cfg["torch"].cuda.is_available.assert_not_called()
                cfg["torch"].set_num_threads.assert_not_called()

    def test_exact_service_properties_current_pid_and_invocation(self):
        changes = {"Type": "simple", "RuntimeMaxUSec": "31min", "Restart": "on-failure",
                   "KillMode": "process", "MainPID": "999", "InvocationID": "b" * 32,
                   "ActiveState": "inactive", "SubState": "dead", "ControlGroup": "/elsewhere"}
        for key, value in changes.items():
            with self.subTest(key=key), self.fabricated(change=lambda c: c["service"].update({key: value})) as (cfg, _):
                with self.assertRaisesRegex(guard.GuardError, "service identity"):
                    guard.configure()
                cfg["torch"].cuda.is_available.assert_not_called()

    def test_unknown_duplicate_and_oversize_gpu_clients(self):
        for clients in ["999, 1", "2101, 513", "8861, 129", "34567, 4097", "2101, 1\n2101, 2",
                        "2101, N/A", "2101, -1", "2101, 12, extra"]:
            with self.subTest(clients=clients), self.fabricated(clients=clients) as (cfg, _):
                with self.assertRaises(guard.GuardError):
                    guard.configure()
                cfg["torch"].set_num_threads.assert_not_called()

    def test_no_gpu_clients_is_valid_no_displacement_or_creation(self):
        with self.fabricated(clients=""):
            receipt = guard.configure()
            self.assertEqual(receipt["gpu_clients"], [])
            self.assertEqual(receipt["preexisting_gpu_clients"], [])

    def test_version_environment_and_target_fail_before_observations(self):
        changes = [lambda c: setattr(c["torch"], "__version__", "2.10.0"),
                   lambda c: c.update(numpy_version="2.0.0"),
                   lambda c: c["env"].update({"OMP_NUM_THREADS": "2"}),
                   lambda c: c["env"].update({"CUBLAS_WORKSPACE_CONFIG": ":16:8"}),
                   lambda c: c["env"].update({"INVOCATION_ID": "not-an-id"})]
        for i, change in enumerate(changes):
            with self.subTest(i=i), self.fabricated(change=change) as (_, called):
                with self.assertRaises(guard.GuardError):
                    guard.configure()
                called.assert_not_called()
        with self.fabricated() as (_, called):
            with self.assertRaises(guard.GuardError):
                guard.configure("cpu")
            called.assert_not_called()

    def test_wrong_device_insufficient_free_and_unavailable_fail(self):
        changes = [lambda c: setattr(c["torch"].cuda.is_available, "return_value", False),
                   lambda c: setattr(c["torch"].cuda.get_device_name, "return_value", "NVIDIA A100"),
                   lambda c: setattr(c["torch"].cuda.mem_get_info, "return_value", (7 * 1024**3, 24 * 1024**3)),
                   lambda c: setattr(c["torch"].cuda.mem_get_info, "return_value", (24 * 1024**3, 12 * 1024**3))]
        for i, change in enumerate(changes):
            with self.subTest(i=i), self.fabricated(change=change) as (cfg, _):
                with self.assertRaises(guard.GuardError):
                    guard.configure()
                cfg["torch"].set_num_threads.assert_not_called()

    def test_deterministic_readback_failure_and_service_query_failure_propagate(self):
        with self.fabricated(change=lambda c: setattr(c["torch"].get_num_threads, "return_value", 2)):
            with self.assertRaisesRegex(guard.GuardError, "readback"):
                guard.configure()
        with self.fabricated(), mock.patch.object(guard.subprocess, "check_output", side_effect=OSError("unavailable")):
            with self.assertRaises(OSError):
                guard.configure()

    def test_missing_duplicate_unknown_or_malformed_service_properties_fail(self):
        for suffix, replace in [("Type=exec", False), ("Unexpected=yes", False),
                                ("no-equals", False), ("Type=exec", True)]:
            with self.subTest(suffix=suffix, replace=replace), self.fabricated() as (_, called):
                original = called.side_effect

                def malformed(command, **kwargs):
                    result = original(command, **kwargs)
                    return suffix if replace else result + "\n" + suffix

                called.side_effect = malformed
                with self.assertRaises(guard.GuardError):
                    guard.configure()


if __name__ == "__main__":
    unittest.main()
