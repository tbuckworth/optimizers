"""Fresh CPU-only worker-entry fixtures; no native request, plans or IDX read."""
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import unittest
from unittest import mock

if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
    raise RuntimeError("worker tests require hidden CUDA")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import phase_worker as worker


class PhaseWorkerTests(unittest.TestCase):
    def env(self):
        return dict(PATH="/usr/bin:/bin", LANG="C.UTF-8", LC_ALL="C.UTF-8", TZ="UTC",
            TMPDIR="/tmp/spectral-experiment-artifacts", PYTHONDONTWRITEBYTECODE="1", PYTHONHASHSEED="0",
            CUDA_VISIBLE_DEVICES="", CUBLAS_WORKSPACE_CONFIG=":4096:8",
            OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
            NUMEXPR_NUM_THREADS="1")

    def test_import_and_default_cli_are_inert_and_torch_free(self):
        code = ("import sys;sys.path.insert(0,sys.argv[1]);import phase_worker;"
                "print('torch' in sys.modules)")
        result = subprocess.run([sys.executable, "-B", "-c", code, str(HERE)],
            env=self.env(), cwd=HERE, capture_output=True, text=True, timeout=20, check=True)
        self.assertEqual(result.stdout, "False\n")
        result = subprocess.run([sys.executable, "-B", str(HERE / "phase_worker.py")],
            env=self.env(), cwd=HERE, capture_output=True, text=True, timeout=20, check=True)
        self.assertEqual(json.loads(result.stdout), {"status":"inert"})
        self.assertEqual(result.stderr, "")

    def test_fresh_cpu_attestation_has_child_clocks_and_no_cuda(self):
        started = time.monotonic()
        with subprocess.Popen([sys.executable, "-B", str(HERE / "phase_worker.py"),
                "--fixture-audit-attestation"], env=self.env(), cwd=HERE,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as child:
            stdout, stderr = child.communicate(timeout=120)
            self.assertEqual(child.returncode, 0, stderr[:1000])
            value = json.loads(stdout)
            self.assertEqual(value["schema"], "i7_worker_cpu_fixture_v1")
            self.assertTrue(value["fixture_only"])
            self.assertFalse(value["execution_authorized"])
            self.assertFalse(value["scientific_execution_certified"])
            resources = value["resources"]
            self.assertEqual(resources["worker_entry_process_id"], child.pid)
            self.assertNotEqual(resources["worker_entry_process_id"], os.getpid())
            self.assertGreaterEqual(resources["worker_entry_wall_origin"], started)
            self.assertLess(resources["worker_entry_cpu_origin"], 1.0)
            self.assertGreater(resources["cpu_seconds"], 0.0)
            self.assertFalse(resources["cuda_initialized"])
            self.assertEqual(resources["peak_cuda_allocated_bytes"], 0)
            self.assertLess(resources["peak_rss_bytes"], 2 << 30)

    def test_forked_origin_is_rejected(self):
        with mock.patch.object(worker.os, "getpid", return_value=worker._ENTRY_PID + 1):
            with self.assertRaisesRegex(worker.WorkerError, "forked"):
                worker._origins()

    def test_worker_deadline_and_audit_cuda_checks(self):
        resources = dict(wall_seconds=1.0, cpu_seconds=1.0, peak_rss_bytes=100,
                         peak_cuda_allocated_bytes=0, cuda_initialized=False)
        request = dict(phase="audit", deadline_monotonic=time.monotonic()+10,
            limits=dict(wall_seconds=600.0, cpu_seconds=600.0,
                        peak_rss_bytes=12 << 30, peak_cuda_allocated_bytes=8 << 30))
        with mock.patch.object(worker, "_resources", return_value=resources):
            self.assertEqual(worker._check_resources(request), resources)
            resources["cuda_initialized"] = True
            with self.assertRaisesRegex(worker.WorkerError, "initialized CUDA"):
                worker._check_resources(request)
            resources["cuda_initialized"] = False
            request["deadline_monotonic"] = time.monotonic()-1
            with self.assertRaisesRegex(worker.WorkerError, "resource limit"):
                worker._check_resources(request)

    def test_native_rejection_does_not_disclose_exception_text(self):
        with mock.patch.object(worker, "_native_main", side_effect=RuntimeError("private detail")):
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                code = worker.main(["--run-native-phase", "--request-path", "/missing",
                    "--request-size", "1", "--request-sha256", "0" * 64,
                    "--supervisor-pid", str(os.getpid())])
            self.assertEqual(code, 2)
            self.assertEqual(stderr.getvalue(), "worker request rejected\n")

    @staticmethod
    def request(phase):
        return {"phase":phase, "attempt_id":"i7-native-development-attempt-001",
            "worker_binding":{"data_directory":"/fixture-no-read", "expected_files":{}},
            "development_permission_ascii":"{}", "development_pins":{},
            "phase_permission_pin":{}, "prior_boundary":{}}

    @staticmethod
    def handoff():
        return {"status":"closed_inspected_boundary", "execution_authorized":False,
            "automatic_reopen_enabled":False, "scientific_execution_certified":False,
            "root_binding":{}, "journal_pin":{}, "boundary_ref":{},
            "inspection":{"phase_records":10}, "other_closed_handoff_field":None}

    def test_exact_one_phase_dispatch_and_audit_never_gets_initializer(self):
        import native_control
        import phase_transition
        import native_controller
        import scientific_controller
        for phase in ("development", "primary", "sensitivity", "audit"):
            with self.subTest(phase=phase), contextlib.ExitStack() as stack:
                store = mock.Mock(_closed=False)
                opened = {"store":store}
                boot = stack.enter_context(mock.patch.object(native_control,
                    "bootstrap_development", autospec=True, return_value=opened))
                acquire = stack.enter_context(mock.patch.object(phase_transition,
                    "acquire_phase", autospec=True, return_value=opened))
                native = stack.enter_context(mock.patch.object(native_controller,
                    "NativeController", autospec=True))
                scientific = stack.enter_context(mock.patch.object(scientific_controller,
                    "ScientificController", autospec=True))
                native.return_value.close_development.return_value = self.handoff()
                scientific.return_value.close_phase.return_value = self.handoff()
                result = worker._execute_request(self.request(phase), {})
                self.assertEqual(result, self.handoff())
                store.close.assert_called_once()
                if phase == "development":
                    boot.assert_called_once()
                    acquire.assert_not_called()
                    scientific.assert_not_called()
                    native.return_value.run_development.assert_called_once_with()
                    native.return_value.prepare_primary.assert_not_called()
                else:
                    boot.assert_not_called()
                    native.assert_not_called()
                    acquire.assert_called_once()
                    self.assertEqual(acquire.call_args.kwargs["phase"], phase)
                    self.assertEqual(acquire.call_args.kwargs["entry_process_id"], os.getpid())
                    initializer = acquire.call_args.kwargs["initialize_runtime"]
                    self.assertEqual(initializer is None, phase == "audit")
                    scientific.return_value.run_phase.assert_called_once_with()
                    scientific.return_value.prepare_primary.assert_not_called()

    def test_development_writer_receives_only_successfully_collected_metadata(self):
        import artifact_store
        import native_control
        import native_controller
        sources, environment = {"unit": "sources"}, {"unit": "environment"}
        store = mock.Mock(_closed=False)
        def bootstrap(*args, **kwargs):
            with self.assertRaisesRegex(worker.WorkerError, "completed metadata/admission"):
                kwargs["create_store"]()
            self.assertEqual(kwargs["collect_metadata"]({}), (sources, environment))
            return {"store": kwargs["create_store"]()}
        with mock.patch.object(worker, "_collect_metadata", return_value=(sources, environment)), \
             mock.patch.object(native_control, "bootstrap_development", side_effect=bootstrap), \
             mock.patch.object(artifact_store, "ArtifactStore", return_value=store) as create, \
             mock.patch.object(native_controller, "NativeController", autospec=True) as controller:
            controller.return_value.close_development.return_value = self.handoff()
            self.assertEqual(worker._execute_request(self.request("development"), {}), self.handoff())
            create.assert_called_once_with("/tmp/spectral-experiment-artifacts", profile=artifact_store.SCIENTIFIC,
                runtime_sources=sources, runtime_environment=environment)
            store.close.assert_called_once()

    def test_report_normalizes_handoff_and_never_grants_authority(self):
        with mock.patch.object(worker, "_execute_request", return_value=self.handoff()) as execute, \
             mock.patch.object(worker, "_check_resources", return_value={}):
            report = worker._execute_and_report(self.request("development"), "0" * 64)
        execute.assert_called_once()
        self.assertEqual(tuple(report["handoff"]),
            ("phase", "phase_records", "root_binding", "journal_pin", "boundary_ref", "inspection"))
        self.assertEqual(report["status"], "boundary_closed")
        self.assertFalse(report["execution_authorized"])
        self.assertFalse(report["scientific_execution_certified"])

    def test_structured_failure_survives_resource_probe_failure_and_has_no_retry(self):
        with mock.patch.object(worker, "_check_resources", side_effect=worker.WorkerResourceError()), \
             mock.patch.object(worker, "_resources", side_effect=RuntimeError("private")), \
             mock.patch.object(worker, "_execute_request") as execute:
            report = worker._execute_and_report(self.request("development"), "0" * 64)
        execute.assert_not_called()
        self.assertIsNone(report["resources"])
        self.assertIsNone(report["handoff"])
        self.assertEqual(report["failure"], {"stage":"request_validation",
            "reason":"resource_limit", "retention":"consumed_request_no_retry"})

    def test_admission_failure_preserves_stage_without_controller_or_retry(self):
        def fail(request, progress):
            progress["stage"] = "storage_admission"
            raise RuntimeError("unavailable private proof")
        with mock.patch.object(worker, "_check_resources", return_value={}), \
             mock.patch.object(worker, "_resources", return_value={}), \
             mock.patch.object(worker, "_execute_request", side_effect=fail) as execute:
            report = worker._execute_and_report(self.request("development"), "0" * 64)
        execute.assert_called_once()
        self.assertEqual(report["failure"], {"stage":"storage_admission",
            "reason":"worker_stage_failed", "retention":"consumed_request_no_retry"})
        self.assertIsNone(report["handoff"])


if __name__ == "__main__":
    unittest.main()
