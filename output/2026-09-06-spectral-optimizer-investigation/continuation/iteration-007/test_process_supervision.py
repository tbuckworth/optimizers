"""CPU-only tests for the inert I7 process supervisor."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

import process_supervision as subject


BIG_TMP = Path("/tmp/spectral-experiment-artifacts")
UUID = "GPU-994b97cd-d768-1f03-78fb-6a70e3cc1e0c"


class ProcessSupervisionTests(unittest.TestCase):
    def setUp(self):
        self.entry = time.monotonic()
        import native_control
        native_control._verify_big_parent(BIG_TMP)
        self.assertGreaterEqual(shutil.disk_usage(BIG_TMP).free, 1 << 30)
        self.temp = tempfile.TemporaryDirectory(
            prefix="i7-process-supervision-fixture-", dir=BIG_TMP)
        self.outer = Path(self.temp.name)
        self.namespace = self.outer / "i7-native-development-attempt-001-supervision"
        for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS"):
            self.assertEqual(os.environ.get(key), "1")
        self.assertEqual(os.environ.get("CUDA_VISIBLE_DEVICES"), "")

    def tearDown(self):
        logical = sum(row.stat(follow_symlinks=False).st_size
                      for row in self.outer.rglob("*") if row.is_file())
        self.assertLessEqual(logical, 2 << 20)
        self.assertLessEqual(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                             2 << 30)
        self.assertLessEqual(time.monotonic() - self.entry, 120.0)
        self.temp.cleanup()
        self.assertFalse(self.outer.exists())

    def request(self, *, phase="development", prior=None, wall=2.0,
                output=4096):
        origin = time.monotonic()
        limits = subject.fixed_limits(phase)
        limits["wall_seconds"] = wall
        limits["termination_grace_seconds"] = 0.2
        limits["kill_disappear_seconds"] = 0.3
        limits["worker_output_bytes"] = output
        worker = {
            "repository_root":str(self.outer),
            "worker_path":str(Path(__file__).resolve()),
            "worker_relative_path":"test_process_supervision.py",
            "python_executable":subject.PYTHON_EXECUTABLE,
            "repository_revision":"a" * 40,
            "git_mode":"100644", "size_bytes":1, "sha256":"b" * 64,
            "git_blob_oid":"c" * 40, "data_directory":str(self.outer),
            "expected_files":{
                "training_images":{"sha256":"d" * 64, "size_bytes":1},
                "training_labels":{"sha256":"e" * 64, "size_bytes":1}}}
        value = {
            "schema":subject.REQUEST_SCHEMA, "status":"authorized_once",
            "attempt_id":subject.ATTEMPT_ID, "phase":phase,
            "scope":subject.SCOPES[phase], "evidence_kind":"synthetic_contract_fixture",
            "retry_allowed":False, "created_utc":"2026-09-07T12:00:00Z",
            "supervisor_pid":os.getpid(), "supervisor_wall_origin":origin,
            "deadline_monotonic":origin + wall, "expected_commit":"a" * 40,
            "expected_source_set_sha256":"f" * 64,
            "expected_environment_sha256":None if phase == "development" else "1" * 64,
            "expected_gpu_uuid":None if phase == "audit" else UUID,
            "authority_kind":"fixture", "development_permission_ascii":None,
            "development_pins":None, "phase_permission_pin":None,
            "prior_boundary":prior, "storage_admission_pin":{
                "path":str(self.outer / "fixture-storage-admission.json"),
                "size_bytes":2, "sha256":hashlib.sha256(b"{}\n").hexdigest()},
            "worker_binding":worker, "limits":limits}
        return value, subject.encode_request(value, fixture=True)

    @staticmethod
    def handoff(phase="development"):
        return {"phase":phase, "phase_records":subject.PHASE_RECORDS[phase],
                "root_binding":{}, "journal_pin":{}, "boundary_ref":{},
                "inspection":{}}

    def report_program(self, request, *, failure=False, exit_code=0):
        digest = hashlib.sha256(subject.json_bytes(
            request, maximum=subject.REQUEST_MAX)).hexdigest()
        status = "worker_failed" if failure else "boundary_closed"
        handoff = None if failure else self.handoff(request["phase"])
        failure_row = ({"stage":"controller", "reason":"worker_stage_failed",
                        "retention":"consumed_request_no_retry"}
                       if failure else None)
        code = (
            "import json,os,time,resource\n"
            "resources=None if " + repr(failure) + " else {"
            "'worker_entry_wall_origin':time.monotonic(),"
            "'worker_entry_cpu_origin':time.process_time(),"
            "'worker_entry_process_id':os.getpid(),'wall_seconds':0.0,"
            "'cpu_seconds':0.0,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,"
            "'peak_cuda_allocated_bytes':0,'peak_cuda_reserved_bytes':0,"
            "'cuda_initialized':False}\n"
            "row={'schema':" + repr(subject.WORKER_REPORT_SCHEMA)
            + ",'status':" + repr(status)
            + ",'attempt_id':" + repr(subject.ATTEMPT_ID)
            + ",'phase':" + repr(request["phase"])
            + ",'request_sha256':" + repr(digest)
            + ",'worker_pid':os.getpid(),'handoff':" + repr(handoff)
            + ",'failure':" + repr(failure_row)
            + ",'resources':resources,'execution_authorized':False,"
              "'scientific_execution_certified':False}\n"
            "print(json.dumps(row,separators=(',',':')))\n"
            f"raise SystemExit({exit_code})\n")
        return [subject.PYTHON_EXECUTABLE, "-I", "-c", code]

    def supervise(self, request, raw, argv, verifier=lambda _request, _report: True):
        return subject._supervise_fixture(raw,
            expected_request_sha256=hashlib.sha256(raw).hexdigest(),
            namespace=str(self.namespace), argv=argv, verify_handoff=verifier)

    def test_import_and_default_cli_are_torch_free_and_inert(self):
        module = Path(subject.__file__).resolve()
        code = ("import importlib.util,sys;"
                f"s=importlib.util.spec_from_file_location('ps',{str(module)!r});"
                "m=importlib.util.module_from_spec(s);s.loader.exec_module(m);"
                "assert 'torch' not in sys.modules")
        run = subprocess.run([subject.PYTHON_EXECUTABLE, "-c", code],
                             env={"PATH":"/usr/bin:/bin"}, capture_output=True,
                             timeout=10)
        self.assertEqual(run.returncode, 0)
        run = subprocess.run([subject.PYTHON_EXECUTABLE, str(module)],
                             env={"PATH":"/usr/bin:/bin"}, capture_output=True,
                             timeout=10)
        self.assertEqual(run.stdout, b'{"status":"inert"}\n')
        self.assertFalse(self.namespace.exists())

    def test_request_codec_is_exact_bounded_and_sanitized(self):
        request, raw = self.request()
        self.assertLessEqual(len(raw), subject.REQUEST_MAX)
        decoded = subject.validate_request(raw,
            expected_sha256=hashlib.sha256(raw).hexdigest(), fixture=True,
            expected_supervisor_pid=os.getpid(), authenticate=False)
        environment = subject.sanitized_environment(decoded)
        self.assertNotIn("HOME", environment)
        self.assertNotIn("PYTHONPATH", environment)
        self.assertEqual(environment["CUDA_VISIBLE_DEVICES"], UUID)
        legacy = dict(request, schema="i7_native_process_request_v1")
        legacy_raw = subject.json_bytes(legacy, maximum=subject.REQUEST_MAX)
        with self.assertRaisesRegex(subject.SupervisionError, "scope/authority"):
            subject.validate_request(
                legacy_raw, expected_sha256=hashlib.sha256(legacy_raw).hexdigest(),
                fixture=True, authenticate=False)
        changed = dict(request)
        changed["expected_gpu_uuid"] = UUID.upper()
        bad = subject.json_bytes(changed, maximum=subject.REQUEST_MAX)
        with self.assertRaisesRegex(subject.SupervisionError, "GPU"):
            subject.validate_request(bad, expected_sha256=hashlib.sha256(bad).hexdigest(),
                                     fixture=True, authenticate=False)
        duplicate = raw[:-2] + b',"schema":"x"}\n'
        with self.assertRaisesRegex(subject.SupervisionError, "duplicate"):
            subject.json_loads(duplicate, maximum=subject.REQUEST_MAX)
        with self.assertRaisesRegex(subject.SupervisionError, "nonfinite"):
            subject.json_loads(b'{"x":NaN}\n', maximum=100)
        for phase, previous in zip(subject.PHASES[1:], subject.PHASES[:-1]):
            shaped, shaped_raw = self.request(
                phase=phase, prior=self.handoff(previous))
            self.assertLessEqual(len(shaped_raw), subject.REQUEST_MAX)
            report = {"schema":subject.WORKER_REPORT_SCHEMA,
                "status":"boundary_closed", "attempt_id":subject.ATTEMPT_ID,
                "phase":phase,
                "request_sha256":hashlib.sha256(shaped_raw).hexdigest(),
                "worker_pid":os.getpid(), "handoff":self.handoff(phase),
                "failure":None, "resources":{
                    "worker_entry_wall_origin":time.monotonic(),
                    "worker_entry_cpu_origin":time.process_time(),
                    "worker_entry_process_id":os.getpid(), "wall_seconds":0.0,
                    "cpu_seconds":0.0, "peak_rss_bytes":1,
                    "peak_cuda_allocated_bytes":0, "peak_cuda_reserved_bytes":0,
                    "cuda_initialized":False}, "execution_authorized":False,
                "scientific_execution_certified":False}
            self.assertLessEqual(len(subject.encode_worker_report(report, shaped)),
                                 subject.WORKER_OUTPUT_MAX)

    def test_prospective_native_request_shapes_fit_eight_kib(self):
        import native_control as control
        admission = {"path":subject.STORAGE_ADMISSION_PATH,
            "size_bytes":subject.STORAGE_ADMISSION_MAX, "sha256":"f" * 64}
        development_permission = {"schema":control.PERMISSION_SCHEMA,
            "attempt_id":subject.ATTEMPT_ID, "phase":"development",
            "scope":control.SCOPE, "expected_commit":"f" * 40,
            "expected_source_set_sha256":"f" * 64,
            "expected_gpu_uuid":"GPU-ffffffff-ffff-ffff-ffff-ffffffffffff",
            "storage_admission_pin":admission,
            "authorized_utc":"9999-12-31T23:59:59Z", "decision":"go",
            "retry_allowed":False}
        permission_raw = control._json_bytes(development_permission)
        development_pins = {
            "permission_sha256":hashlib.sha256(permission_raw).hexdigest(),
            "attempt_id":subject.ATTEMPT_ID, "expected_commit":"f" * 40,
            "expected_source_set_sha256":"f" * 64,
            "expected_gpu_uuid":"GPU-ffffffff-ffff-ffff-ffff-ffffffffffff",
            "attempt_directory":control.DEVELOPMENT_ATTEMPT_PATH}
        worker = {"repository_root":subject.REPOSITORY_ROOT,
            "worker_path":subject.WORKER_PATH,
            "worker_relative_path":subject.WORKER_RELATIVE_PATH,
            "python_executable":subject.PYTHON_EXECUTABLE,
            "repository_revision":"f" * 40, "git_mode":"100644",
            "size_bytes":(1 << 64) - 1, "sha256":"f" * 64,
            "git_blob_oid":"f" * 40, "data_directory":subject.DATA_DIRECTORY,
            "expected_files":subject.EXPECTED_FILES}
        root = {"path":"/tmp/spectral-experiment-artifacts/i7-artifacts-" + "a" * 8,
            "device":(1 << 64) - 1, "inode":(1 << 64) - 1,
            "header_sha256":"f" * 64}
        journal = {"path":control.DEVELOPMENT_ATTEMPT_PATH +
                           "/native-launch-journal.json",
            "size_bytes":control.EXTERNAL_JOURNAL_MAX, "sha256":"f" * 64}
        origin = 1.7976931348623155e+308
        sizes = {}
        requests = {}
        for phase in subject.PHASES:
            phase_index = subject.PHASES.index(phase)
            prior = None
            if phase_index:
                previous = subject.PHASES[phase_index - 1]
                sequence = {"primary":9, "sensitivity":43, "audit":55}[phase]
                name = f"native-phase-{sequence:03d}.json"
                reference = {"name":name, "status":"complete", "encoding":"bytes",
                    "size_bytes":128 << 10, "sha256":"f" * 64,
                    "receipt_name":"receipt-" + hashlib.sha256(
                        name.encode("ascii")).hexdigest() + ".json",
                    "receipt_size_bytes":4096, "receipt_sha256":"f" * 64}
                inspection = {"schema":control.INSPECTION_SCHEMA,
                    "status":"sealed_boundary",
                    "reason":"verified_boundary_requires_next_go",
                    "phase_records":subject.PHASE_RECORDS[previous],
                    "completed_operations":92, "last_event":"boundary",
                    "pending_operation":None, "store_terminal":False,
                    "store_failure_count":0, "metadata_complete":True,
                    "actual_artifacts_verified":True,
                    "root_identity":[(1 << 64) - 1, (1 << 64) - 1],
                    "header_sha256":"f" * 64, "journal_pin":journal,
                    "eligible_for_writable_reopen":True,
                    "can_resume_incomplete":False, "execution_authorized":False,
                    "scientific_execution_certified":False}
                prior = {"phase":previous,
                    "phase_records":subject.PHASE_RECORDS[previous],
                    "root_binding":root, "journal_pin":journal,
                    "boundary_ref":reference, "inspection":inspection}
            permission_pin = None if phase == "development" else {
                "path":control.DEVELOPMENT_ATTEMPT_PATH + "/" +
                       f"native-{phase}-permission.json",
                "size_bytes":8192, "sha256":"f" * 64}
            request = {"schema":subject.REQUEST_SCHEMA, "status":"authorized_once",
                "attempt_id":subject.ATTEMPT_ID, "phase":phase,
                "scope":subject.SCOPES[phase],
                "evidence_kind":"native_producer_attestation",
                "retry_allowed":False, "created_utc":"9999-12-31T23:59:59Z",
                "supervisor_pid":(1 << 31) - 1,
                "supervisor_wall_origin":origin, "deadline_monotonic":origin,
                "expected_commit":"f" * 40,
                "expected_source_set_sha256":"f" * 64,
                "expected_environment_sha256":(
                    None if phase == "development" else "f" * 64),
                "expected_gpu_uuid":(
                    None if phase == "audit" else
                    "GPU-ffffffff-ffff-ffff-ffff-ffffffffffff"),
                "authority_kind":("development_launcher" if phase == "development"
                                  else "phase_permission_pin"),
                "development_permission_ascii":(
                    permission_raw.decode("ascii") if phase == "development" else None),
                "development_pins":(
                    development_pins if phase == "development" else None),
                "phase_permission_pin":permission_pin, "prior_boundary":prior,
                "storage_admission_pin":admission, "worker_binding":worker,
                "limits":subject.fixed_limits(phase)}
            encoded = subject.encode_request(request, fixture=False)
            sizes[phase] = len(encoded)
            requests[phase] = (request, encoded)
        self.assertEqual(sizes, {"development":3670, "primary":4338,
                                 "sensitivity":4314, "audit":4272})
        self.assertLessEqual(max(sizes.values()), subject.REQUEST_MAX)

        import phase_transition as transition
        for phase, (request, _encoded) in requests.items():
            changed = dict(request)
            changed["storage_admission_pin"] = dict(
                admission, sha256="0" * 64)
            changed_raw = subject.json_bytes(changed, maximum=subject.REQUEST_MAX)
            authenticated = {"expected_commit":request["expected_commit"],
                "expected_source_set_sha256":request["expected_source_set_sha256"],
                "expected_environment_sha256":request["expected_environment_sha256"],
                "expected_gpu_uuid":request["expected_gpu_uuid"],
                "storage_admission_pin":admission}
            with mock.patch.object(
                    subject, "load_pinned_bytes",
                    side_effect=AssertionError("admission read must follow equality")) as load:
                if phase == "development":
                    with self.assertRaisesRegex(subject.SupervisionError,
                                                "permission differs from request"):
                        subject.validate_request(
                            changed_raw,
                            expected_sha256=hashlib.sha256(changed_raw).hexdigest(),
                            expected_phase=phase, fixture=False, authenticate=True)
                else:
                    with mock.patch.object(
                            transition, "_authenticate_actual",
                            return_value=(b"permission", authenticated)):
                        with self.assertRaisesRegex(subject.SupervisionError,
                                                    "permission differs from request"):
                            subject.validate_request(
                                changed_raw,
                                expected_sha256=hashlib.sha256(changed_raw).hexdigest(),
                                expected_phase=phase, fixture=False, authenticate=True)
                load.assert_not_called()

        development = requests["development"][0]
        extra = dict(development)
        extra["development_pins"] = dict(
            development["development_pins"], unexpected="forbidden")
        extra_raw = subject.json_bytes(extra, maximum=subject.REQUEST_MAX)
        with self.assertRaisesRegex(control.ControlError, "exact ordered keys"):
            subject.validate_request(
                extra_raw, expected_sha256=hashlib.sha256(extra_raw).hexdigest(),
                expected_phase="development", fixture=False, authenticate=True)
        wrong = dict(development)
        wrong["development_pins"] = dict(
            development["development_pins"],
            attempt_directory="/tmp/spectral-experiment-artifacts/wrong-attempt")
        wrong_raw = subject.json_bytes(wrong, maximum=subject.REQUEST_MAX)
        with self.assertRaisesRegex(subject.SupervisionError, "attempt directory"):
            subject.validate_request(
                wrong_raw, expected_sha256=hashlib.sha256(wrong_raw).hexdigest(),
                expected_phase="development", fixture=False, authenticate=True)

    def test_success_observes_true_exit_and_persists_closed_record(self):
        request, raw = self.request()
        record = self.supervise(request, raw, self.report_program(request))
        self.assertEqual(record["exit_classification"], "boundary_verified")
        self.assertEqual(record["observed_returncode"], 0)
        self.assertTrue(record["group_disappearance_observed"])
        exit_raw = (self.namespace / subject.exit_name("development")).read_bytes()
        self.assertLessEqual(len(exit_raw), subject.EXIT_MAX)
        self.assertEqual(subject.validate_exit_record(exit_raw, request), record)

    def test_structured_nonzero_failure_is_preserved_without_raw_text(self):
        request, raw = self.request()
        argv = self.report_program(request, failure=True, exit_code=1)
        record = self.supervise(request, raw, argv)
        self.assertEqual(record["exit_classification"], "structured_worker_failure")
        self.assertEqual(record["worker_report"]["failure"], {
            "stage":"controller", "reason":"worker_stage_failed",
            "retention":"consumed_request_no_retry"})
        persisted = (self.namespace / subject.exit_name("development")).read_text()
        self.assertNotIn("Traceback", persisted)

    def test_timeout_kills_owned_leader_and_grandchild_group(self):
        request, raw = self.request(wall=0.25)
        code = ("import signal,subprocess,time,sys;"
                "signal.signal(signal.SIGTERM,signal.SIG_IGN);"
                "subprocess.Popen([sys.executable,'-c',"
                "'import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(30)']);"
                "time.sleep(30)")
        record = self.supervise(request, raw,
                                [subject.PYTHON_EXECUTABLE, "-I", "-c", code])
        self.assertEqual(record["exit_classification"], "timeout")
        self.assertTrue(record["term_sent"])
        self.assertTrue(record["kill_sent"])
        self.assertTrue(record["group_disappearance_observed"])
        self.assertEqual(subject._live_group_members(record["process_group_id"]), [])

    def test_output_limit_is_bounded_and_raw_output_is_not_persisted(self):
        request, raw = self.request(output=256)
        code = "import os,time;os.write(1,b'Z'*4096);time.sleep(30)"
        record = self.supervise(request, raw,
                                [subject.PYTHON_EXECUTABLE, "-I", "-c", code])
        self.assertEqual(record["exit_classification"], "output_limit")
        self.assertEqual(record["captured_output_bytes"], 257)
        persisted = (self.namespace / subject.exit_name("development")).read_bytes()
        self.assertNotIn(b"ZZZZ", persisted)

    def test_launch_failure_publishes_exit_and_collision_never_retries(self):
        request, raw = self.request()
        record = self.supervise(request, raw,
            [subject.PYTHON_EXECUTABLE, str(self.outer / "absent.py")])
        self.assertEqual(record["exit_classification"], "nonzero_without_valid_failure")
        self.assertTrue((self.namespace / subject.request_name("development")).exists())
        self.assertTrue((self.namespace / subject.exit_name("development")).exists())
        with self.assertRaisesRegex(subject.SupervisionError, "already consumed"):
            self.supervise(request, raw, self.report_program(request))

    def test_signaled_and_zero_malformed_exits_are_distinct(self):
        request, raw = self.request()
        record = self.supervise(request, raw, [subject.PYTHON_EXECUTABLE, "-I", "-c",
            "import os,signal;os.kill(os.getpid(),signal.SIGKILL)"])
        self.assertEqual(record["exit_classification"], "signaled")
        self.assertLess(record["observed_returncode"], 0)
        self.temp.cleanup()
        self.temp = tempfile.TemporaryDirectory(
            prefix="i7-process-supervision-fixture-", dir=BIG_TMP)
        self.outer = Path(self.temp.name)
        self.namespace = self.outer / "i7-native-development-attempt-001-supervision"
        request, raw = self.request()
        record = self.supervise(request, raw,
            [subject.PYTHON_EXECUTABLE, "-I", "-c", "print('malformed')"])
        self.assertEqual(record["exit_classification"], "malformed_report")
        self.assertEqual(record["observed_returncode"], 0)

    def test_sequential_phase_authenticates_prior_request_exit_and_handoff(self):
        development, raw = self.request()
        first = self.supervise(development, raw, self.report_program(development))
        prior = first["worker_report"]["handoff"]
        primary, primary_raw = self.request(phase="primary", prior=prior)
        changed = dict(primary)
        changed["storage_admission_pin"] = dict(
            primary["storage_admission_pin"], sha256="0" * 64)
        changed_raw = subject.encode_request(changed, fixture=True)
        with self.assertRaisesRegex(subject.SupervisionError, "source/admission"):
            self.supervise(changed, changed_raw, self.report_program(changed))
        second = self.supervise(primary, primary_raw, self.report_program(primary))
        self.assertEqual(second["exit_classification"], "boundary_verified")
        primary_exit = self.namespace / subject.exit_name("primary")
        changed = json.loads(primary_exit.read_text())
        changed["verified_handoff"] = False
        primary_exit.write_bytes(subject.json_bytes(changed, maximum=subject.EXIT_MAX))
        sensitivity, sensitivity_raw = self.request(
            phase="sensitivity", prior=second["worker_report"]["handoff"])
        with self.assertRaisesRegex(subject.SupervisionError, "successful exit"):
            self.supervise(sensitivity, sensitivity_raw,
                           self.report_program(sensitivity))

    def test_unverified_ownership_kills_only_pidfd_leader_and_is_terminal(self):
        request, raw = self.request()
        argv = [subject.PYTHON_EXECUTABLE, "-I", "-c", "import time;time.sleep(30)"]
        with mock.patch.object(subject, "_proc_identity", return_value=None):
            record = self.supervise(request, raw, argv)
        self.assertEqual(record["exit_classification"], "ownership_unverified")
        self.assertTrue(record["kill_sent"])
        self.assertIsNone(record["process_group_id"])
        self.assertFalse(record["group_disappearance_observed"])

    def test_sanitized_system_python_imports_torch_cpu_only(self):
        request, _ = self.request(phase="audit", prior=self.handoff("sensitivity"))
        environment = subject.sanitized_environment(request)
        code = ("import os,torch;assert 'HOME' not in os.environ;"
                "assert torch.get_num_threads()==1;"
                "assert not torch.cuda.is_initialized();"
                "assert os.environ['CUDA_VISIBLE_DEVICES']==''")
        run = subprocess.run([subject.PYTHON_EXECUTABLE, "-c", code],
                             env=environment, capture_output=True, timeout=30)
        self.assertEqual(run.returncode, 0, run.stderr[:200])


if __name__ == "__main__":
    unittest.main()
