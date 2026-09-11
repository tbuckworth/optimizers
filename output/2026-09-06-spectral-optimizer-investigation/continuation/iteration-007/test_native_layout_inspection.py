"""CPU-only tests for the inert one-shot native-layout inspector.

Run with:
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
  python3 -m unittest test_native_layout_inspection -v
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest import mock


if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
    raise RuntimeError("native-layout-inspection tests require CUDA_VISIBLE_DEVICES=''")

import native_layout_inspection as subject


def _success_report():
    commit = "a" * 40
    uuid = "GPU-994b97cd-d768-1f03-78fb-6a70e3cc1e0c"
    source_files = [
        {"path": path, "role": role, "git_mode": "100644", "size_bytes": 1,
         "sha256": "b" * 64, "git_blob_oid": "c" * 40}
        for role, path in subject.environment_schema.SCIENTIFIC_ROLES
    ]
    environment = {
        "schema": "i7_environment_binding_v1",
        "profile": subject.environment_schema.SCIENTIFIC,
        "runtime_role": "native_source", "repository_root_realpath": subject.REPO_ROOT,
        "python": {"implementation": "CPython", "version": "3.12.3",
                   "version_info": [3, 12, 3], "cache_tag": "cpython-312",
                   "executable_realpath": "/usr/bin/python3"},
        "libraries": {"numpy_version": "1.26.4", "torch_version": "2.11.0+cu128",
                      "torch_git_revision":
                          "70d99e998b4955e0049d13a98d77ae1b14db1f45",
                      "torch_cuda_build": "12.8", "cudnn_version": 91900},
        "operating_system": {"sys_platform": "linux", "system": "Linux",
                             "release": "fixture", "machine": "x86_64",
                             "libc_name": "glibc", "libc_version": "fixture"},
        "safe_environment": {"CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                             "CUDA_VISIBLE_DEVICES": uuid, "MKL_NUM_THREADS": "1",
                             "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                             "PYTHONHASHSEED": "0"},
        "torch_settings": dict(subject.EXPECTED_TORCH_SETTINGS),
        "cuda": {"initialized": True, "visible_device_count": 1, "current_device": 0,
                 "driver_version": "999.1",
                 "devices": [{"index": 0, "name": subject.EXPECTED_GPU_NAME,
                              "uuid": uuid, "total_memory_bytes": 1,
                              "capability_major": 8, "capability_minor": 6}]},
        "rng_layout": {
            "python": {"state_version": 3, "internal_length": 625},
            "numpy": {"algorithm": "MT19937", "keys_dtype": "uint32",
                      "keys_length": 624},
            "torch_cpu": {"dtype": "torch.uint8", "state_length": 5056},
            "torch_cuda": [{"index": 0, "name": subject.EXPECTED_GPU_NAME,
                            "uuid": uuid, "dtype": "torch.uint8", "state_length": 16}],
        },
    }
    return {
        "schema": subject.SCHEMA, "status": "worker_pass", "expected_commit": commit,
        "expected_gpu_uuid": uuid,
        "source_binding": {
            "schema": "i7_source_binding_v1", "profile": "scientific_mnist_current32_v1",
            "repository_revision": commit, "git_object_format": "sha1",
            "repository_root_realpath": subject.REPO_ROOT,
            "worktree_status": "clean_including_untracked_v1",
            "files": source_files,
        },
        "native_environment_binding": environment,
        "helper_binding": {
            "path": subject.RELATIVE_PATH, "role": "engineering_native_layout_inspector",
            "repository_revision": commit, "git_mode": "100644", "size_bytes": 1,
            "sha256": "d" * 64, "git_blob_oid": "e" * 40,
        },
        "resources": {
            "limits": {"wall_seconds": subject.DEADLINE_SECONDS,
                       "peak_rss_bytes": subject.RSS_CAP,
                       "torch_peak_allocated_bytes": subject.CUDA_ALLOCATED_CAP},
            "wall_seconds": 1.0, "cpu_seconds": 0.5, "peak_rss_bytes": 1024,
            "torch_peak_allocated_bytes": 0, "torch_peak_reserved_bytes": 0,
            "own_nvml_gpu_bytes": None, "own_nvml_observable": False,
            **subject.WORKER_RESOURCE_SCOPES,
            "caps_are_cooperative_post_call_observations": True,
        },
        "side_effects": {"cuda_context_initialized": True,
                         "cuda_initialization_is_not_read_only": True,
                         "random_draws_performed": False,
                         "rng_state_restoration_performed": False},
        "certificates": {"scientific_execution": False, "pilot": False,
                         "full_fit": False, "capture_neutrality": False,
                         "continuation_restore": False},
    }


class InertCliTests(unittest.TestCase):
    def test_import_help_and_default_do_not_import_torch(self):
        module_dir = str(Path(subject.__file__).parent)
        code = (
            "import sys; import native_layout_inspection as s; "
            "assert 'torch' not in sys.modules; assert s.main([]) == 0; "
            "assert 'torch' not in sys.modules"
        )
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
        result = subprocess.run([sys.executable, "-c", code], cwd=module_dir, env=env,
                                capture_output=True, text=True, timeout=5, check=True)
        self.assertEqual(result.stdout, '{"status":"inert"}\n')
        help_result = subprocess.run([sys.executable, subject.__file__, "--help"], env=env,
                                     capture_output=True, text=True, timeout=5, check=True)
        self.assertNotIn("torch", help_result.stdout.lower())

class ValidationTests(unittest.TestCase):
    def test_bad_expected_values_are_rejected_without_preflight(self):
        with self.assertRaisesRegex(subject.InspectionError, "expected commit"):
            subject.run_controller("F" * 40,
                                   "GPU-994b97cd-d768-1f03-78fb-6a70e3cc1e0c")
        with self.assertRaisesRegex(subject.InspectionError, "GPU UUID"):
            subject.run_controller("a" * 40, "not-a-gpu")
        with self.assertRaisesRegex(subject.InspectionError, "GPU UUID"):
            subject.run_controller("a" * 40,
                                   "GPU-994B97CD-d768-1f03-78fb-6a70e3cc1e0c")
        with self.assertRaisesRegex(subject.InspectionError, "expected clean commit mismatch"):
            subject._bind_helper(subject.REPO_ROOT, "0" * 40,
                                 deadline=time.monotonic() + 3)

    def test_bad_gpu_preflight_and_unknown_compute_process_are_rejected(self):
        expected = "GPU-994b97cd-d768-1f03-78fb-6a70e3cc1e0c"
        bad_gpu = b"0, GPU-other, NVIDIA GeForce RTX 3090, 23596, 999.1\n"
        with mock.patch.object(subject, "_run_command", return_value=bad_gpu):
            with self.assertRaisesRegex(subject.InspectionError, "UUID/name"):
                subject._gpu_preflight(expected, deadline=time.monotonic() + 1)

        gpu = (f"0, {expected}, NVIDIA GeForce RTX 3090, 23596, 999.1\n").encode()
        process = (f"{expected}, 321, /usr/bin/python3, 200\n").encode()
        with mock.patch.object(subject, "_run_command", side_effect=[gpu, process]), \
             mock.patch.object(subject, "_proc_identity",
                               return_value=("/usr/bin/python3", "python3", os.getuid())):
            with self.assertRaisesRegex(subject.InspectionError, "research or unknown"):
                subject._gpu_preflight(expected, deadline=time.monotonic() + 1)

    def test_known_desktop_process_needs_exact_exe_comm_and_uid(self):
        expected = "GPU-994b97cd-d768-1f03-78fb-6a70e3cc1e0c"
        gpu = (f"0, {expected}, NVIDIA GeForce RTX 3090, 23596, 999.1\n").encode()
        exe = "/app/opt/stremio/stremio"
        process = (f"{expected}, 321, stremio, 200\n").encode()
        with mock.patch.object(subject, "_run_command", side_effect=[gpu, process]), \
             mock.patch.object(subject, "_proc_identity",
                               return_value=(exe, "stremio", os.getuid())):
            row = subject._gpu_preflight(expected, deadline=time.monotonic() + 1)
        self.assertEqual(row["tolerated_processes"], [
            {"kind": "known_unrelated_desktop_application", "pid": 321,
             "used_gpu_memory_mib": 200}
        ])
        self.assertFalse(row["driver_version_pinned"])
        self.assertFalse(row["occupancy_exclusive"])
        unavailable = subject._parse_compute_rows(
            (f"{expected}, 321, stremio, N/A\n").encode())
        self.assertIsNone(unavailable[0][3])


class OutputOwnershipTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="i7-native-inspection-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = str(Path(self.temporary.name) / "singleton")

    def _new_root(self):
        with mock.patch.object(subject, "ATTEMPT_ROOT", self.root):
            return subject._create_attempt_root(self.root)

    def test_fixed_singleton_collision_and_exclusive_marker(self):
        rootfd, identity = self._new_root()
        self.addCleanup(os.close, rootfd)
        with mock.patch.object(subject, "ATTEMPT_ROOT", self.root):
            with self.assertRaisesRegex(subject.InspectionError, "retry refused"):
                subject._create_attempt_root(self.root)
        marker = {"schema": subject.SCHEMA, "status": "started"}
        subject._write_new_json(rootfd, self.root, identity, subject.MARKER_NAME,
                                marker, cap=1024, normal=True)
        with self.assertRaises(FileExistsError):
            subject._write_new_json(rootfd, self.root, identity, subject.MARKER_NAME,
                                    marker, cap=1024, normal=True)

    def test_failure_is_preserved_and_never_overwritten(self):
        rootfd, identity = self._new_root()
        self.addCleanup(os.close, rootfd)
        marker = {"schema": subject.SCHEMA, "status": "started"}
        first = {"schema": subject.SCHEMA, "status": "failed", "failure": "first"}
        second = {"schema": subject.SCHEMA, "status": "failed", "failure": "second"}
        subject._write_new_json(rootfd, self.root, identity, subject.MARKER_NAME,
                                marker, cap=1024, normal=True)
        subject._write_new_json(rootfd, self.root, identity, subject.FAILURE_NAME,
                                first, cap=1024, normal=False)
        before = (Path(self.root) / subject.FAILURE_NAME).read_bytes()
        with self.assertRaises(FileExistsError):
            subject._write_new_json(rootfd, self.root, identity, subject.FAILURE_NAME,
                                    second, cap=1024, normal=False)
        self.assertEqual((Path(self.root) / subject.FAILURE_NAME).read_bytes(), before)
        self.assertFalse((Path(self.root) / subject.RESULT_NAME).exists())

        controller_root = str(Path(self.temporary.name) / "controller-singleton")
        helper = {"path": subject.RELATIVE_PATH, "sha256": "1" * 64}
        expected_uuid = "GPU-994b97cd-d768-1f03-78fb-6a70e3cc1e0c"
        with mock.patch.object(subject, "ATTEMPT_ROOT", controller_root), \
             mock.patch.object(subject, "sys",
                               SimpleNamespace(modules={}, platform=sys.platform)), \
             mock.patch.object(subject, "_bind_helper", return_value=helper), \
             mock.patch.object(subject, "_verify_attempt_parent", return_value=({}, ())), \
             mock.patch.object(subject, "_gpu_preflight",
                               side_effect=subject.InspectionError("bad preflight")), \
             mock.patch("builtins.print"):
            self.assertEqual(subject.run_controller("a" * 40, expected_uuid), 1)
        saved = json.loads((Path(controller_root) / subject.FAILURE_NAME).read_bytes())
        self.assertEqual(saved["failure"], {
            "origin": "controller", "classification": "controller_guard_failure",
            "reason": "controller_guard_failure", "stage": "gpu_preflight",
            "inspector_cuda_initialization_state": "not_attempted",
            "worker_report_complete": False,
        })
        self.assertNotIn(b"bad preflight",
                         (Path(controller_root) / subject.FAILURE_NAME).read_bytes())
        self.assertFalse((Path(controller_root) / subject.RESULT_NAME).exists())

    def test_nonzero_worker_failure_and_protocol_failures_publish_exact_diagnostics(self):
        expected_uuid = "GPU-994b97cd-d768-1f03-78fb-6a70e3cc1e0c"
        helper = {"path": subject.RELATIVE_PATH, "sha256": "1" * 64}
        preflight = {"driver_version_observed": "999.1"}

        def run_case(name, returncode, output):
            root = str(Path(self.temporary.name) / name)
            with mock.patch.object(subject, "ATTEMPT_ROOT", root), \
                 mock.patch.object(subject, "sys",
                                   SimpleNamespace(modules={}, platform=sys.platform)), \
                 mock.patch.object(subject, "_bind_helper", return_value=helper), \
                 mock.patch.object(subject, "_verify_attempt_parent", return_value=({}, ())), \
                 mock.patch.object(subject, "_gpu_preflight", return_value=dict(preflight)), \
                 mock.patch.object(subject, "_worker_process",
                                   return_value=(returncode, output, 321)), \
                 mock.patch.object(subject, "_assert_pid_absent_from_gpu", return_value=True), \
                 mock.patch("builtins.print"):
                self.assertEqual(subject.run_controller("a" * 40, expected_uuid), 1)
            return json.loads((Path(root) / subject.FAILURE_NAME).read_bytes())

        worker_failure = {
            "schema": subject.SCHEMA, "status": "worker_failed",
            "failure_stage": "cuda_initialization",
            "failure_class": "source_environment_error",
            "failure_reason": "invalid_cuda_uuid",
            "inspector_cuda_initialization_state": "attempted_outcome_unknown",
            "report_complete": True,
        }
        saved = run_case("structured", 1, subject._canonical_json(
            worker_failure, cap=subject.FAILURE_RESERVE))
        self.assertEqual(saved["failure"], {
            "origin": "worker", "classification": "source_environment_error",
            "reason": "invalid_cuda_uuid", "stage": "cuda_initialization",
            "inspector_cuda_initialization_state": "attempted_outcome_unknown",
            "worker_report_complete": True,
        })

        malformed = run_case(
            "malformed", 1,
            b'{"schema":"i7_native_layout_inspection_v2","status":"worker_failed",'
            b'"status":"worker_failed"}\n')
        self.assertEqual(malformed["failure"]["classification"],
                         "malformed_worker_report")
        self.assertEqual(
            malformed["failure"]["inspector_cuda_initialization_state"], "unknown")

        absent = run_case("absent", -signal.SIGKILL, b"")
        self.assertEqual(absent["failure"]["classification"],
                         "worker_terminated_without_report")
        self.assertEqual(absent["failure"]["stage"], "worker_report_decode")
        self.assertFalse(absent["failure"]["worker_report_complete"])

        disagreement = run_case("disagreement", 0, subject._canonical_json(
            worker_failure, cap=subject.FAILURE_RESERVE))
        self.assertEqual(disagreement["failure"]["classification"],
                         "worker_exit_status_disagreement")

    def test_strict_worker_json_and_actionable_reason_mapping(self):
        duplicate = (b'{"schema":"i7_native_layout_inspection_v2",'
                     b'"schema":"i7_native_layout_inspection_v2",'
                     b'"status":"worker_failed"}\n')
        nonfinite = (b'{"schema":"i7_native_layout_inspection_v2",'
                     b'"status":"worker_failed","failure_stage":"cuda_initialization",'
                     b'"failure_class":"source_environment_error",'
                     b'"failure_reason":"invalid_cuda_uuid",'
                     b'"inspector_cuda_initialization_state":"attempted_outcome_unknown",'
                     b'"report_complete":NaN}\n')
        overflow = nonfinite.replace(b"NaN", b"1e999")
        legacy_v1 = nonfinite.replace(
            b"i7_native_layout_inspection_v2", b"i7_native_layout_inspection_v1"
        ).replace(b"NaN", b"true")
        for payload in (duplicate, nonfinite, overflow, legacy_v1):
            with self.subTest(payload=payload[:30]):
                with self.assertRaises(subject.InspectionError):
                    subject._decode_worker_output(payload)

        class SourceEnvironmentError(ValueError):
            pass

        fake_source = SimpleNamespace(SourceEnvironmentError=SourceEnvironmentError)
        progress = {"failure_stage": "environment_collection",
                    "inspector_cuda_initialization_state": "initialized"}
        with mock.patch.dict(subject.sys.modules, {"source_environment": fake_source}):
            invalid = subject._worker_failure_report(
                progress, SourceEnvironmentError("CUDA device has no valid stable UUID"))
            driver = subject._worker_failure_report(
                progress, SourceEnvironmentError(
                    "CUDA UUID has no unique driver-version row"))
            generic = subject._worker_failure_report(
                progress, SourceEnvironmentError("secret arbitrary detail"))
        self.assertEqual(invalid["failure_reason"], "invalid_cuda_uuid")
        self.assertEqual(driver["failure_reason"], "driver_uuid_unmatched")
        self.assertEqual(generic["failure_reason"], "source_environment_failure")
        self.assertNotIn(b"secret", subject._canonical_json(generic, cap=1024))

        success = _success_report()
        self.assertEqual(subject._decode_worker_output(subject._canonical_json(
            success, cap=subject.COMBINED_CHILD_OUTPUT_CAP)), success)
        mutations = []
        for path, value in (
                (("side_effects", "cuda_context_initialized"), 1),
                (("certificates", "scientific_execution"), 0),
                (("resources", "wall_seconds"), subject.DEADLINE_SECONDS + 0.1),
                (("resources", "peak_rss_bytes"), subject.RSS_CAP + 1),
                (("resources", "torch_peak_allocated_bytes"),
                 subject.CUDA_ALLOCATED_CAP + 1),
                (("resources", "own_nvml_observable"), True)):
            changed = copy.deepcopy(success)
            changed[path[0]][path[1]] = value
            mutations.append((path, changed))
        bad_source = copy.deepcopy(success)
        bad_source["source_binding"]["files"][0]["size_bytes"] = True
        mutations.append((("source_binding", "files", "size_bytes"), bad_source))
        bad_environment = copy.deepcopy(success)
        bad_environment["native_environment_binding"]["cuda"]["devices"][0]["uuid"] = (
            "GPU-994B97CD-d768-1f03-78fb-6a70e3cc1e0c")
        mutations.append((("native_environment_binding", "cuda", "uuid"), bad_environment))
        bad_setting = copy.deepcopy(success)
        bad_setting["native_environment_binding"]["torch_settings"][
            "deterministic_algorithms"] = 1
        mutations.append((
            ("native_environment_binding", "torch_settings", "deterministic_algorithms"),
            bad_setting))
        bad_wall_limit = copy.deepcopy(success)
        bad_wall_limit["resources"]["limits"]["wall_seconds"] = 45
        mutations.append((("resources", "limits", "wall_seconds_type"), bad_wall_limit))
        bad_byte_limit = copy.deepcopy(success)
        bad_byte_limit["resources"]["limits"]["peak_rss_bytes"] = float(subject.RSS_CAP)
        mutations.append((("resources", "limits", "peak_rss_bytes_type"), bad_byte_limit))
        bad_scope = copy.deepcopy(success)
        bad_scope["resources"]["rss_scope"] = "arbitrary scope"
        mutations.append((("resources", "rss_scope"), bad_scope))
        for path, changed in mutations:
            with self.subTest(field=".".join(path)):
                with self.assertRaises(subject.InspectionError):
                    subject._decode_worker_output(subject._canonical_json(
                        changed, cap=subject.COMBINED_CHILD_OUTPUT_CAP))

        for error in (TypeError("malformed nested type"), RecursionError("too deep")):
            with self.subTest(error=type(error).__name__), \
                 mock.patch.object(subject, "_decode_worker_output", side_effect=error):
                with self.assertRaisesRegex(subject.WorkerProtocolFailure,
                                            "worker protocol failure"):
                    subject._interpret_worker_exit(1, b"nonempty")

    def test_nested_collector_key_order_survives_write_and_reload(self):
        rootfd, identity = self._new_root()
        self.addCleanup(os.close, rootfd)
        source = {"schema": "source", "profile": "scientific",
                  "repository_revision": "a" * 40,
                  "files": [{"path": "x", "role": "producer", "size": 1}]}
        environment = {"schema": "environment", "profile": "scientific",
                       "runtime_role": "native_source",
                       "torch_settings": {"num_threads": 1, "grad_enabled": True}}
        value = {"schema": subject.SCHEMA, "source_binding": source,
                 "native_environment_binding": environment}
        subject._write_new_json(rootfd, self.root, identity, subject.RESULT_NAME,
                                value, cap=4096, normal=True)
        loaded = json.loads((Path(self.root) / subject.RESULT_NAME).read_bytes())
        self.assertEqual(tuple(loaded["source_binding"]), tuple(source))
        self.assertEqual(tuple(loaded["source_binding"]["files"][0]),
                         tuple(source["files"][0]))
        self.assertEqual(tuple(loaded["native_environment_binding"]), tuple(environment))
        self.assertEqual(tuple(loaded["native_environment_binding"]["torch_settings"]),
                         tuple(environment["torch_settings"]))


class BoundedProcessTests(unittest.TestCase):
    def _process(self, code):
        return subprocess.Popen([sys.executable, "-c", code], stdin=subprocess.DEVNULL,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                start_new_session=True)

    def test_combined_output_cap_terminates_and_reaps_own_group(self):
        process = self._process("import os; os.write(1, b'x' * 10000)")
        with self.assertRaisesRegex(subject.InspectionError, "output exceeded"):
            subject._capture_process(process, deadline=time.monotonic() + 2, cap=256,
                                     grace=0.1)
        self.assertIsNotNone(process.returncode)

    def test_timeout_escalates_to_kill_and_reaps_own_group(self):
        process = self._process(
            "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(30)"
        )
        started = time.monotonic()
        with self.assertRaisesRegex(subject.InspectionError, "deadline exceeded"):
            subject._capture_process(process, deadline=time.monotonic() + 0.15, cap=256,
                                     grace=0.1)
        self.assertEqual(process.returncode, -9)
        self.assertLess(time.monotonic() - started, 2)

    def test_exited_leader_cannot_leave_child_and_grandchild_group(self):
        child_code = (
            "import signal,subprocess,sys,time; "
            "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
            "subprocess.Popen([sys.executable,'-c',"
            "'import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(30)']); "
            "time.sleep(30)"
        )
        leader_code = (
            "import subprocess,sys; "
            f"subprocess.Popen([sys.executable,'-c',{child_code!r}])"
        )
        process = self._process(leader_code)
        with self.assertRaisesRegex(subject.InspectionError,
                                    "deadline exceeded|owned descendants remained"):
            subject._capture_process(process, deadline=time.monotonic() + 0.25, cap=256,
                                     grace=0.1)
        self.assertIsNotNone(process.returncode)
        self.assertEqual(subject._live_group_members(process.pid), ())


class ExplicitSettingsTests(unittest.TestCase):
    def test_all_torch_settings_are_explicit_and_cpu_configuration_is_cuda_inert(self):
        module_dir = str(Path(subject.__file__).parent)
        code = (
            "import native_layout_inspection as s, torch; "
            "assert not torch.cuda.is_initialized(); "
            "assert s._configure_and_assert_torch(torch) == s.EXPECTED_TORCH_SETTINGS; "
            "assert not torch.cuda.is_initialized()"
        )
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
        result = subprocess.run([sys.executable, "-c", code], cwd=module_dir, env=env,
                                capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
