"""Tiny parser/subprocess contracts; no production slots, service or specimens."""
import copy
import os
from pathlib import Path
import subprocess
import sys
import time
import unittest
from unittest import mock

import native_storage_authority as authority
import native_storage_crosscheck as cross
import native_storage_measurement_controller as controller
import native_storage_measurement_service as service
from test_native_storage_authority import C, S, synthetic_measurement


def validate(value):
    return authority.validate_measurement(value, expected_commit=C,
        expected_source_set_sha256=S, require_complete=False)


class MeasurementSchemaTests(unittest.TestCase):
    def test_final_and_candidate_are_disjoint(self):
        value = synthetic_measurement()
        validate(value)
        with self.assertRaises(authority.StorageAuthorityError):
            authority.validate_measurement_candidate(value, expected_commit=C, expected_source_set_sha256=S)
        value["supervision"] = None
        authority.validate_measurement_candidate(value, expected_commit=C, expected_source_set_sha256=S)
        with self.assertRaises(authority.StorageAuthorityError):
            validate(value)

    def test_supervision_rejects_unobserved_controls_identity_and_cleanup(self):
        for key, replacement in (("worker_exit_code", 1), ("worker_pid", 5),
                ("worker_process_group", 5), ("worker_group_gone", False),
                ("max_poll_wait_ns", 1), ("sampled_aggregate_peak_rss_bytes", service.MEMORY_MAX + 1),
                ("stderr_bytes", 1), ("elapsed_wall_ns", service.WALL_NS + 1),
                ("candidate_size_bytes", 1), ("candidate_sha256", "e" * 64),
                ("invocation_id", "bad"), ("service_active_enter_ns", True)):
            value = synthetic_measurement()
            value["supervision"][key] = replacement
            with self.subTest(key=key), self.assertRaises(authority.StorageAuthorityError):
                validate(value)

    def test_candidate_mutation_does_not_rebind_itself(self):
        value = synthetic_measurement()
        value["diagnostic_environment"]["operating_system"]["release"] += "-changed"
        with self.assertRaisesRegex(authority.StorageAuthorityError, "candidate bytes differ"):
            validate(value)

    def test_cgroup_charge_events_and_descendants_fail_closed(self):
        for key, replacement in (("memory_peak_bytes", service.MEMORY_MAX + 1),
                ("memory_max_bytes", service.MEMORY_MAX - 1), ("memory_oom_group", 0),
                ("pids_max_events", 1), ("nr_descendants", 1),
                ("member_pids", [2, 3]), ("inode", 3), ("memory_zswap_max_bytes", 1)):
            value = synthetic_measurement()
            value["supervision"]["after"][key] = replacement
            with self.subTest(key=key), self.assertRaises(authority.StorageAuthorityError):
                validate(value)
        value = synthetic_measurement()
        value["supervision"]["after"]["memory_events"]["oom"] = 1
        with self.assertRaises(authority.StorageAuthorityError):
            validate(value)

    def test_candidate_and_supervision_keep_whole_record_headroom(self):
        value = synthetic_measurement()
        signed = authority.encode_bounded(value)
        self.assertLessEqual(len(signed), 64 << 10)
        self.assertLessEqual(len(authority.encode_bounded(value["supervision"])), 8 << 10)
        value["supervision"] = None
        self.assertLessEqual(len(authority.encode_bounded(value)), 56 << 10)

    def test_stage_and_peak_callbacks_observe_tiny_json(self):
        peaks, stages = [], []
        result = cross._json_roundtrip({"profile": "diagnostic_mnist_current32_v1", "tiny": [1, None]},
            body_ceiling=1024, on_buffer_peak=peaks.append, on_stage=stages.append)
        self.assertEqual(stages, ["serialize", "roundtrip"])
        self.assertEqual(peaks[0], 0)
        self.assertEqual(max(peaks), result["body_bytes"])
        def stop(_):
            raise RuntimeError("fixed test observer stop")
        with self.assertRaises(RuntimeError):
            cross._json_roundtrip({"profile": "diagnostic_mnist_current32_v1"},
                body_ceiling=1024, on_buffer_peak=stop)


class MeasurementControllerTests(unittest.TestCase):
    def test_os_preflight_helper_is_torch_free_and_uses_exact_environment(self):
        code = ("import sys,os,native_storage_measurement_service as s;"
                "s.verify_big_tmp();"
                "assert s._run_command(['/usr/bin/true'])==b'';"
                "assert 'torch' not in sys.modules and 'numpy' not in sys.modules;"
                "assert s._run_command(['/usr/bin/python3','-I','-B','-c',"
                "'import os;assert os.environ.get(\\\"GIT_CONFIG_GLOBAL\\\")==\\\"/dev/null\\\"'])==b''")
        run = subprocess.run([sys.executable, "-B", "-c", code],
            cwd=Path(__file__).resolve().parent, capture_output=True, timeout=10)
        self.assertEqual(run.returncode, 0, run.stderr.decode())

    def test_signal_failure_preserves_original_error_and_closes_descriptors(self):
        import io
        import process_supervision as supervision
        process = mock.Mock(pid=99999999, returncode=None, stdout=io.BytesIO(), stderr=io.BytesIO())
        def wait(**kwargs):
            process.returncode = -9
            return -9
        process.wait.side_effect = wait
        with mock.patch.object(os, "pidfd_open", return_value=99999998), \
             mock.patch.object(os, "close") as close, \
             mock.patch.object(supervision, "_proc_identity", return_value=None), \
             mock.patch.object(controller.signal, "pidfd_send_signal", side_effect=ProcessLookupError):
            with self.assertRaisesRegex(controller.MeasurementControllerError, "worker_ownership"):
                controller._capture(process, None, deadline_ns=time.monotonic_ns()+1000000000)
            close.assert_called_once_with(99999998)
        self.assertTrue(process.stdout.closed and process.stderr.closed)
        process.wait.assert_called_once()

    def test_import_and_default_cli_are_inert_torch_free(self):
        code = ("import sys,native_storage_measurement_controller as c;"
                "assert c.main([])==0;assert 'torch' not in sys.modules;"
                "assert 'numpy' not in sys.modules")
        run = subprocess.run([sys.executable, "-B", "-c", code],
            cwd=Path(__file__).resolve().parent, capture_output=True, timeout=10)
        self.assertEqual(run.returncode, 0, run.stderr.decode())

    def test_direct_unfrozen_execution_rejects_before_slot(self):
        import native_storage_measurement_slot as slot
        with mock.patch.object(slot, "reserve_measurement", side_effect=AssertionError("must not reserve")):
            with self.assertRaisesRegex(controller.MeasurementControllerError, "fixed_frozen_root"):
                controller.run_one_shot(expected_commit=C, expected_source_set_sha256=S)

    def test_worker_command_is_fixed_isolated_cpu_entry(self):
        import process_supervision as supervision
        command = controller._worker_command(expected_commit=C, expected_source_set_sha256=S,
                                             controller_pid=123)
        self.assertEqual(command[:4], [supervision.PYTHON_EXECUTABLE, "-I", "-B", "-c"])
        self.assertEqual(command[-5:], [C, S, supervision.REPOSITORY_ROOT,
                                        "123", service.CGROUP])
        self.assertEqual(command[6], service.DEPENDENCY_DIRECTORY)
        self.assertIn("sys.path.append(dependency_directory)", command[4])
        self.assertNotIn("phase_worker", command[4])
        self.assertIn("build_measurement_candidate", command[4])

    def test_exact_isolated_interpreter_loads_production_dependency_closure(self):
        import process_supervision as supervision
        code = ("import sys;sys.path.insert(0,sys.argv[1]);"
                "import native_storage_measurement_service as s;"
                "path=s.validate_dependency_directory(sys.argv[2]);"
                "assert sys.flags.isolated==1 and path not in sys.path;"
                "sys.path.append(path);"
                "import native_storage_measurement_worker as w;"
                "dependencies=w._production_dependencies();"
                "assert tuple(dependencies)==w._DEPENDENCY_KEYS;"
                "assert 'numpy' in sys.modules and 'torch' in sys.modules")
        run = subprocess.run(
            [supervision.PYTHON_EXECUTABLE, "-I", "-B", "-c", code,
             str(Path(__file__).resolve().parent), service.DEPENDENCY_DIRECTORY],
            cwd="/", env=service.process_environment(), stdin=subprocess.DEVNULL,
            capture_output=True, timeout=30)
        self.assertEqual(run.returncode, 0, run.stderr.decode(errors="replace"))

    def capture(self, code, *, seconds=3, resource_failure=False):
        class FakeContext:
            # Explicit substitute for observation tests. This is not OS evidence.
            def snapshot(self):
                return dict(sampled_rss_bytes=service.MEMORY_MAX + 1 if resource_failure else 1,
                    memory_peak_bytes=1, memory_events={key: 0 for key in service.EVENT_FIELDS},
                    pids_max_events=0)
        process = subprocess.Popen([sys.executable, "-I", "-B", "-c", code],
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            start_new_session=True, close_fds=True)
        self.last_process = process
        return controller._capture(process, FakeContext(),
                                   deadline_ns=time.monotonic_ns()+int(seconds*1e9))

    def test_bounded_child_exit_and_separate_stderr_count(self):
        outcome = self.capture("import os;os.write(1,b'candidate');os.write(2,b'note')")
        self.assertEqual(outcome["raw"], b"candidate")
        self.assertEqual(outcome["stderr"], b"note")
        self.assertEqual(outcome["stderr_bytes"], 4)
        self.assertEqual(outcome["worker_exit_code"], 0)
        self.assertTrue(outcome["worker_group_gone"])
        self.assertIsNotNone(self.last_process.returncode)

    def test_worker_diagnostic_maps_only_closed_codes_to_service_status(self):
        prefix = controller._WORKER_DIAGNOSTIC_PREFIX
        for code, status in controller._WORKER_DIAGNOSTIC_EXITS.items():
            outcome = {"stderr": prefix + code.encode("ascii") + b"\n"}
            self.assertEqual(controller._worker_diagnostic(outcome), code)
            self.assertEqual(controller.MeasurementWorkerFailure(code).exit_status, status)
        for raw in (b"", b"traceback", prefix + b"unknown\n", prefix + b"runtime_unobserved"):
            self.assertEqual(controller._worker_diagnostic({"stderr": raw}),
                             "worker_unobserved")

    def test_pipe_overflow_timeout_and_resource_stop_kill_owned_child(self):
        cases = (("import os;os.write(1,b'x'*60000)", 3, False, "candidate_output_limit"),
                 ("import os;os.write(2,b'x'*5000)", 3, False, "stderr_output_limit"),
                 ("import time;time.sleep(10)", .1, False, "worker_wall_limit"),
                 ("import time;time.sleep(10)", 3, True, "worker_resource_limit"))
        for code, seconds, bad, reason in cases:
            with self.subTest(reason=reason):
                with self.assertRaisesRegex(controller.MeasurementControllerError, reason):
                    self.capture(code, seconds=seconds, resource_failure=bad)
                self.assertIsNotNone(self.last_process.returncode)
                self.assertFalse(Path(f"/proc/{self.last_process.pid}").exists())

    def test_descendant_holding_pipe_cannot_outlive_timeout(self):
        # A tiny child/grandchild group; outer service integration covers setsid
        # descendants and hard controller death, which a process group cannot.
        code = ("import os,time;"
                "pid=os.fork();"
                "time.sleep(10) if pid==0 else None")
        with self.assertRaisesRegex(controller.MeasurementControllerError, "worker_wall_limit"):
            self.capture(code, seconds=.2)
        import process_supervision as supervision
        deadline = time.monotonic() + 1
        while supervision._live_group_members(self.last_process.pid) and time.monotonic() < deadline:
            time.sleep(.01)
        self.assertFalse(supervision._live_group_members(self.last_process.pid))

    def test_attach_changes_only_none_supervision_and_binds_raw(self):
        candidate = synthetic_measurement()
        candidate["supervision"] = None
        snap = synthetic_measurement()["supervision"]["before"]
        snap["member_pids"] = [os.getpid()]
        outcome = dict(raw=authority.encode_bounded(candidate), stderr_bytes=0,
            worker_pid=1, worker_start_ticks=1, worker_process_group=1, worker_exit_code=0,
            sampled_aggregate_peak_rss_bytes=4096, worker_group_gone=True)
        final = controller._attach_observation(candidate, outcome, properties=dict(service.PROPERTIES),
            before=copy.deepcopy(snap), after=copy.deepcopy(snap),
            invocation_id="d"*32, service_origin_ns=time.monotonic_ns()-1000000)
        validate(final)
        final["supervision"] = None
        self.assertEqual(final, candidate)
        outcome["raw"] += b" "
        with self.assertRaisesRegex(controller.MeasurementControllerError, "candidate_bytes_changed"):
            controller._attach_observation(candidate, outcome, properties=dict(service.PROPERTIES),
                before=snap, after=snap, invocation_id="d"*32, service_origin_ns=1)


if __name__ == "__main__":
    unittest.main()
