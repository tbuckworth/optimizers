"""Tiny service-control fixtures plus one explicitly opt-in user-unit test.

The integration test never names, starts, stops, or resets the fixed production
unit.  It does not create M, A, a frozen checkout, or a scientific artifact.
"""
from __future__ import annotations

import copy
import os
from pathlib import Path
import re
import secrets
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import native_control as control
import native_storage_authority as authority
import native_storage_measurement_service as service
from test_native_storage_authority import synthetic_measurement


BIG_TMP = "/tmp/spectral-experiment-artifacts"
_TEST_UNIT = re.compile(r"i7-storage-unit-[0-9a-f]{16}\.service\Z", re.ASCII)
_ENVIRONMENT = service.process_environment()
_HARNESS_ENVIRONMENT = {key: _ENVIRONMENT[key] for key in (
    "CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "TMPDIR")}


def _write_cgroup_fixture(root, cgroup, *, pid):
    directory = Path(str(root) + cgroup)
    directory.mkdir(parents=True)
    rows = {
        "cgroup.type": "domain\n", "cgroup.stat": "nr_descendants 0\n",
        "memory.max": str(service.MEMORY_MAX) + "\n", "memory.swap.max": "0\n",
        "pids.max": "128\n", "memory.peak": "4096\n",
        "memory.events": "".join(f"{key} 0\n" for key in service.EVENT_FIELDS),
        "cgroup.procs": str(pid) + "\n", "memory.zswap.max": "0\n",
        "memory.oom.group": "1\n", "pids.events": "max 0\n",
    }
    for name, value in rows.items():
        (directory / name).write_text(value, encoding="ascii")
    return directory


def _run(command, *, check=True, timeout=5):
    result = subprocess.run(command, stdin=subprocess.DEVNULL,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            env=service.process_environment(), timeout=timeout, check=False)
    if len(result.stdout) > 8192:
        raise AssertionError("system command output exceeded fixture cap")
    if check and result.returncode != 0:
        raise AssertionError("bounded system command failed")
    return result


def _require_test_unit(unit):
    if _TEST_UNIT.fullmatch(unit) is None or unit == service.UNIT:
        raise AssertionError("refusing non-fixture unit")


def _proc_bytes(pid, name, maximum=8192):
    with open(f"/proc/{pid}/{name}", "rb") as handle:
        raw = handle.read(maximum + 1)
    if len(raw) > maximum:
        raise AssertionError("proc fixture observation exceeded cap")
    return raw


def _proc_environment(pid):
    values = {}
    for item in _proc_bytes(pid, "environ").split(b"\0"):
        if not item:
            continue
        key, separator, value = item.partition(b"=")
        if not separator or key in values:
            raise AssertionError("malformed process environment")
        values[key.decode("ascii")] = value.decode("ascii")
    return values


def _proc_identity(pid):
    raw = _proc_bytes(pid, "stat").decode("ascii")
    close = raw.rfind(")")
    if close < 0:
        raise AssertionError("malformed process stat")
    fields = raw[close + 2:].split()
    if len(fields) < 20:
        raise AssertionError("short process stat")
    return {"parent": int(fields[1]), "process_group": int(fields[2]),
            "session": int(fields[3]), "start_ticks": int(fields[19])}


def _held_member_pids(context):
    try:
        fd = os.open("cgroup.procs", os.O_RDONLY | os.O_NOFOLLOW, dir_fd=context.fd)
    except FileNotFoundError:
        return []
    try:
        raw = os.read(fd, 8193)
    finally:
        os.close(fd)
    if len(raw) > 8192:
        raise AssertionError("cgroup process observation exceeded cap")
    return sorted({int(value) for value in raw.split()})


class MeasurementServicePureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        control._verify_big_parent(Path(BIG_TMP))

    def test_live_properties_require_every_fixed_control_and_identity(self):
        unit = "i7-storage-unit-0123456789abcdef.service"
        cgroup = "/user.slice/user-1000.slice/user@1000.service/app.slice/" + unit
        value = dict(service.PROPERTIES)
        value.update(Id=unit, ControlGroup=cgroup, MainPID="123", ActiveState="active",
                     SubState="running", InvocationID="a" * 32,
                     ActiveEnterTimestampMonotonic="1")
        self.assertEqual(service.validate_live_properties(
            value, controller_pid=123, cgroup=cgroup, unit=unit), service.PROPERTIES)
        for key, replacement in (("MemoryMax", "1"), ("LimitCORE", "infinity"),
                                 ("StandardOutput", "journal"), ("KillMode", "process"),
                                 ("MainPID", "124"), ("ActiveState", "failed")):
            changed = dict(value)
            changed[key] = replacement
            with self.subTest(key=key), self.assertRaises(service.MeasurementServiceError):
                service.validate_live_properties(
                    changed, controller_pid=123, cgroup=cgroup, unit=unit)

    def test_exact_environment_rejects_before_any_service_probe(self):
        changed = service.process_environment()
        changed["UNREVIEWED"] = "value"
        with mock.patch.dict(os.environ, changed, clear=True), \
             mock.patch.object(service, "assert_service_member",
                               side_effect=AssertionError("must not probe")) as member:
            with self.assertRaisesRegex(service.MeasurementServiceError,
                                        "exact_process_environment"):
                service.assert_controller()
        member.assert_not_called()

    def test_explicit_dependency_directory_is_canonical_owned_and_package_complete(self):
        self.assertEqual(service.validate_dependency_directory(),
                         service.DEPENDENCY_DIRECTORY)
        for changed in (service.DEPENDENCY_DIRECTORY + "/.",
                        str(Path(service.DEPENDENCY_DIRECTORY).parent), None):
            with self.subTest(changed=changed), self.assertRaisesRegex(
                    service.MeasurementServiceError, "dependency_directory"):
                service.validate_dependency_directory(changed)

    def test_snapshot_reads_bounded_exact_kernel_controls(self):
        with tempfile.TemporaryDirectory(prefix="i7-service-cgroup-fixture-",
                                         dir=BIG_TMP) as temporary:
            cgroup = "/user.slice/fixture.service"
            directory = _write_cgroup_fixture(temporary, cgroup, pid=os.getpid())
            fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            try:
                with mock.patch.object(service, "CGROUP_ROOT", temporary):
                    result = service.snapshot(cgroup, dir_fd=fd)
            finally:
                os.close(fd)
            self.assertEqual(result["member_pids"], [os.getpid()])
            self.assertEqual(result["memory_max_bytes"], service.MEMORY_MAX)
            self.assertEqual(result["memory_events"],
                             {key: 0 for key in service.EVENT_FIELDS})
            self.assertGreater(result["sampled_rss_bytes"], 0)

    def test_service_context_rejects_path_replacement(self):
        with tempfile.TemporaryDirectory(prefix="i7-service-identity-fixture-",
                                         dir=BIG_TMP) as temporary:
            cgroup = "/user.slice/fixture.service"
            directory = _write_cgroup_fixture(temporary, cgroup, pid=os.getpid())
            with mock.patch.object(service, "CGROUP_ROOT", temporary):
                context = service.ServiceContext(cgroup)
                try:
                    context.snapshot()
                    displaced = directory.with_name("displaced.service")
                    directory.rename(displaced)
                    _write_cgroup_fixture(temporary, cgroup, pid=os.getpid())
                    with self.assertRaisesRegex(service.MeasurementServiceError,
                                                "cgroup_replaced"):
                        context.snapshot()
                finally:
                    context.close()

    def test_supervision_record_is_bound_to_candidate_and_controls(self):
        complete = synthetic_measurement()
        record = complete["supervision"]
        candidate = copy.deepcopy(complete)
        candidate["supervision"] = None
        self.assertIs(service.validate_record(record, candidate), record)
        for key, replacement in (("MemoryMax", "1"), ("StandardError", "journal")):
            changed = copy.deepcopy(record)
            changed["applied_properties"][key] = replacement
            with self.subTest(key=key), self.assertRaises(authority.StorageAuthorityError):
                service.validate_record(changed, candidate)
        changed = copy.deepcopy(record)
        changed["candidate_sha256"] = "0" * 64
        with self.assertRaises(authority.StorageAuthorityError):
            service.validate_record(changed, candidate)


class MeasurementServiceLifecycleTests(unittest.TestCase):
    @unittest.skipUnless(os.environ.get("I7_TEST_TRANSIENT_SERVICE") == "1",
                         "set I7_TEST_TRANSIENT_SERVICE=1 for one harmless user service")
    def test_unique_transient_service_kills_separate_session_child_with_main(self):
        control._verify_big_parent(Path(BIG_TMP))
        for key, expected in _HARNESS_ENVIRONMENT.items():
            self.assertEqual(os.environ.get(key), expected)
        unit = "i7-storage-unit-" + secrets.token_hex(8) + ".service"
        _require_test_unit(unit)
        print("unique service fixture: " + unit, flush=True)
        loaded = _run(["/usr/bin/systemctl", "--user", "show", unit,
                       "--property=LoadState", "--value"])
        self.assertEqual(loaded.stdout.strip(), b"not-found")

        token = secrets.token_hex(16)
        child_code = "import time;time.sleep(15)"
        controller_code = (
            "import subprocess,sys,time;"
            "subprocess.Popen(['/usr/bin/python3','-I','-B','-c',sys.argv[1],sys.argv[2]],"
            "start_new_session=True,stdin=subprocess.DEVNULL,"
            "stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL);"
            "time.sleep(15)")
        command = ["/usr/bin/systemd-run", "--user", "--quiet", "--collect",
                   "--unit=" + unit]
        command.extend("--property=" + item for item in service.RUN_PROPERTIES)
        command.extend(["--", "/usr/bin/env", "-i",
                        *(key + "=" + value for key, value in _ENVIRONMENT.items()),
                        "/usr/bin/python3", "-I", "-B", "-c",
                        controller_code, child_code, token])
        launch_attempted = False
        context = None
        try:
            launch_attempted = True
            _run(command)
            deadline = time.monotonic() + 5
            live = None
            while time.monotonic() < deadline:
                try:
                    _require_test_unit(unit)
                    self.assertNotIn("torch", sys.modules)
                    self.assertNotIn("numpy", sys.modules)
                    candidate = service._show(unit)
                    self.assertNotIn("torch", sys.modules)
                    self.assertNotIn("numpy", sys.modules)
                    if candidate.get("ActiveState") == "active" and candidate.get("MainPID") != "0":
                        live = candidate
                        break
                except (AssertionError, subprocess.SubprocessError):
                    pass
                time.sleep(.02)
            self.assertIsNotNone(live)
            main_pid = int(live["MainPID"])
            cgroup = live["ControlGroup"]
            self.assertEqual(cgroup,
                "/user.slice/user-1000.slice/user@1000.service/app.slice/" + unit)
            self.assertEqual(service.validate_live_properties(
                live, controller_pid=main_pid, cgroup=cgroup, unit=unit), service.PROPERTIES)
            self.assertIn(token.encode("ascii"), _proc_bytes(main_pid, "cmdline"))
            context = service.ServiceContext(cgroup)
            deadline = time.monotonic() + 5
            before = None
            while time.monotonic() < deadline:
                candidate = context.snapshot()
                if len(candidate["member_pids"]) == 2:
                    before = candidate
                    break
                time.sleep(.02)
            self.assertIsNotNone(before)
            self.assertIn(main_pid, before["member_pids"])
            child_pid = next(pid for pid in before["member_pids"] if pid != main_pid)
            print(f"fixture live main={main_pid} child={child_pid} "
                  f"charged_peak={before['memory_peak_bytes']} "
                  f"sampled_rss_sum={before['sampled_rss_bytes']}", flush=True)
            self.assertIn(token.encode("ascii"), _proc_bytes(child_pid, "cmdline"))
            child_identity = _proc_identity(child_pid)
            self.assertEqual(child_identity["parent"], main_pid)
            self.assertEqual((child_identity["process_group"], child_identity["session"]),
                             (child_pid, child_pid))
            for pid in (main_pid, child_pid):
                environment = _proc_environment(pid)
                self.assertEqual(environment, _ENVIRONMENT)

            # Exact verified fixture unit, main process only. KillMode then makes
            # systemd remove the separate-session child from the whole cgroup.
            _run(["/usr/bin/systemctl", "--user", "kill", "--kill-whom=main",
                  "--signal=SIGKILL", unit])
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline and _held_member_pids(context):
                time.sleep(.02)
            self.assertEqual(_held_member_pids(context), [])
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline and (
                    Path(f"/proc/{main_pid}").exists()
                    or Path(f"/proc/{child_pid}").exists()):
                time.sleep(.02)
            self.assertFalse(Path(f"/proc/{main_pid}").exists())
            self.assertFalse(Path(f"/proc/{child_pid}").exists())
        finally:
            if launch_attempted:
                # Never stop/reset a fixed or unverified name.
                _require_test_unit(unit)
                _run(["/usr/bin/systemctl", "--user", "stop", unit],
                     check=False, timeout=5)
            if context is not None:
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline and _held_member_pids(context):
                    time.sleep(.02)
                remaining = _held_member_pids(context)
                context.close()
                if remaining:
                    raise AssertionError("owned fixture cgroup retained processes after cleanup")


if __name__ == "__main__":
    unittest.main()
