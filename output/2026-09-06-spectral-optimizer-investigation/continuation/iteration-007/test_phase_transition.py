"""Tiny CPU fixtures for create-once native phase transitions."""
from __future__ import annotations

from contextlib import contextmanager
import copy
import hashlib
import os
from pathlib import Path
import resource
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock


if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
    raise RuntimeError("phase-transition tests require hidden CUDA")
for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "NUMEXPR_NUM_THREADS"):
    if os.environ.get(_key) != "1":
        raise RuntimeError("phase-transition tests require single numerical threads")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import torch
import artifact_store as storage
import identity_codec as codec
import native_control as control
import native_phase_policy as policy
import phase_transition as transition
import runtime_guard as runtime
import source_environment_schema as environment_schema
from test_native_layout_inspection import _success_report


UUID = "GPU-994b97cd-d768-1f03-78fb-6a70e3cc1e0c"
COMMIT = "a" * 40
UTC = "2026-09-07T18:00:00Z"
KIND = "synthetic_contract_fixture"
BIG_TMP = Path("/tmp/spectral-experiment-artifacts")
BUDGET = 2 << 20
RESERVE = 1 << 20
MIN_FREE = 1 << 30


class PhaseFixture:
    """Reusable primitive-only transition fixture; never execution authority."""

    def checkpoint(self):
        self.assertLessEqual(time.monotonic() - self.started, 120.0)
        self.assertLessEqual(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                             2 << 30)
        self.assertEqual(torch.get_num_threads(), 1)
        self.assertFalse(torch.cuda.is_initialized())

    @contextmanager
    def outer(self):
        self.checkpoint()
        runtime.verify_big_volume(BIG_TMP)
        self.assertGreaterEqual(shutil.disk_usage(BIG_TMP).free, MIN_FREE)
        with tempfile.TemporaryDirectory(prefix="i7-phase-transition-test-",
                                         dir=BIG_TMP) as temporary:
            path = Path(temporary)
            yield path
            logical = sum(item.lstat().st_size for item in path.rglob("*")
                          if item.is_file())
            self.assertLessEqual(logical, BUDGET)
        self.assertFalse(path.exists())
        self.checkpoint()

    @staticmethod
    def metadata():
        report = _success_report()
        return report["source_binding"], report["native_environment_binding"]

    @staticmethod
    def audit_environment(native_environment):
        value = copy.deepcopy(native_environment)
        value["runtime_role"] = "cpu_audit"
        value["cuda"] = {"initialized":False, "visible_device_count":None,
                         "current_device":None, "driver_version":None, "devices":[]}
        value["rng_layout"]["torch_cuda"] = []
        environment_schema.validate_environment(value,
                                                profile=environment_schema.SCIENTIFIC)
        return value

    @staticmethod
    def _root(store):
        return {"path":str(store.root), "device":store.root_identity[0],
                "inode":store.root_identity[1], "header_sha256":store.header_sha256}

    @staticmethod
    def _resources(store):
        report = storage.ArtifactStore.inspect(store.root)
        return {"phase_wall_seconds":0.0, "phase_cpu_seconds":0.0,
            "peak_rss_bytes":1024, "peak_cuda_allocated_bytes":0,
            "peak_cuda_reserved_bytes":0, "root_logical_bytes":report["logical_bytes"],
            "root_allocated_bytes":report["allocated_bytes"]}

    def _artifact(self, store, name):
        body = {"schema":"i7_phase_transition_storage_fixture_v1",
                "storage_only":True, "nonsemantic":True,
                "execution_enabled":False, "artifact_name":name}
        receipt = (store.write_bytes(name, codec.json_bytes(body)) if name.endswith(".json")
                   else store.write_tensor_tree(name, body))
        receipt_raw = (store.root / receipt["receipt_name"]).read_bytes()
        return {"name":name, "status":"complete", "encoding":receipt["encoding"],
            "size_bytes":receipt["size"], "sha256":receipt["sha256"],
            "receipt_name":receipt["receipt_name"],
            "receipt_size_bytes":len(receipt_raw),
            "receipt_sha256":hashlib.sha256(receipt_raw).hexdigest()}

    def _append(self, store, records, operation, event, artifacts, sources, environment):
        previous = hashlib.sha256(codec.json_bytes(records[-1])).hexdigest()
        record = {"schema":"i7_native_phase_record_v1", "profile":policy.PROFILE,
            "evidence_kind":KIND, "execution_enabled":False, "sequence":len(records),
            "event":event, "operation":operation["operation"], "phase":operation["phase"],
            "created_utc":UTC, "root_binding":self._root(store),
            "sources_sha256":codec.tree_digest(sources),
            "environment_sha256":codec.tree_digest(environment),
            "previous_record_sha256":previous, "artifacts":artifacts,
            "resources":self._resources(store), "error":None}
        policy.validate_record(record, evidence_kind=KIND)
        store.write_bytes(control.phase_name(len(records)), codec.json_bytes(record))
        records.append(record)

    @staticmethod
    def existing_reference(root, name):
        raw = (Path(root) / name).read_bytes()
        receipt_name = "receipt-" + hashlib.sha256(name.encode("ascii")).hexdigest() + ".json"
        receipt_raw = (Path(root) / receipt_name).read_bytes()
        receipt = codec.json_loads(receipt_raw, max_bytes=storage._RECEIPT_MAX)
        return {"name":name, "status":"complete", "encoding":receipt["encoding"],
            "size_bytes":len(raw), "sha256":hashlib.sha256(raw).hexdigest(),
            "receipt_name":receipt_name, "receipt_size_bytes":len(receipt_raw),
            "receipt_sha256":hashlib.sha256(receipt_raw).hexdigest()}

    def _advance(self, store, records, sources, environment, stop_phase):
        cursor = policy.validate_transcript(
            records, evidence_kind=KIND, require_complete=False)["completed_operations"]
        for operation in policy.schedule()[cursor:]:
            self.assertEqual(operation["phase"], stop_phase)
            if operation["kind"] == "work":
                self._append(store, records, operation, "started", [], sources, environment)
                refs = [self._artifact(store, name) for name in operation["artifacts"]]
                self._append(store, records, operation, "sealed", refs, sources, environment)
            else:
                self._append(store, records, operation, operation["kind"], [],
                             sources, environment)
            if operation["kind"] == "boundary":
                break
        return records

    def development_boundary(self, outer):
        sources, native_environment = self.metadata()
        attempt = Path(outer) / transition.ATTEMPT_ID
        launcher_permission = {"schema":control.PERMISSION_SCHEMA,
            "attempt_id":transition.ATTEMPT_ID, "phase":"development",
            "scope":control.SCOPE, "expected_commit":COMMIT,
            "expected_source_set_sha256":codec.tree_digest(sources),
            "expected_gpu_uuid":UUID, "storage_admission_pin":{
                "path":str(Path(outer) / "fixture-storage-admission.json"),
                "size_bytes":3, "sha256":hashlib.sha256(b"{}\n").hexdigest()},
            "authorized_utc":UTC,
            "decision":"go", "retry_allowed":False}
        permission_raw = codec.json_bytes(launcher_permission)
        pins = {"permission_sha256":hashlib.sha256(permission_raw).hexdigest(),
            "attempt_id":transition.ATTEMPT_ID, "expected_commit":COMMIT,
            "expected_source_set_sha256":codec.tree_digest(sources),
            "expected_gpu_uuid":UUID, "attempt_directory":str(attempt)}
        result = control.bootstrap_development(
            permission_raw, pins=pins, entry_wall_origin=1.0, entry_cpu_origin=1.0,
            entry_process_id=os.getpid(),
            collect_metadata=lambda _:(sources, native_environment),
            create_store=lambda:storage.ArtifactStore(
                outer, profile=storage.MLP_FIXTURE, budget_bytes=BUDGET,
                failure_reserve_bytes=RESERVE, min_filesystem_free_bytes=MIN_FREE),
            wall_clock=lambda:2.0, cpu_clock=lambda:2.0, utc_now=lambda:UTC,
            evidence_kind=KIND)
        store = result["store"]
        records = self._advance(store, [result["go_record"]], sources,
                                native_environment, "development")
        root_binding = self._root(store)
        root = store.root
        store.close()
        inspection = control.inspect_native_attempt(
            root, result["journal_pin"], expected_store_profile=storage.MLP_FIXTURE,
            expected_evidence_kind=KIND)
        self.assertEqual(inspection["status"], "sealed_boundary")
        return {"phase":"development", "phase_records":10,
            "root_binding":root_binding, "journal_pin":result["journal_pin"],
            "boundary_ref":self.existing_reference(root, control.phase_name(9)),
            "inspection":inspection}, sources, native_environment

    def permission_input(self, prior, phase, sources, environment):
        expected_gpu = None if phase == "audit" else UUID
        return {"schema":transition.PERMISSION_SCHEMA,
            "attempt_id":transition.ATTEMPT_ID, "phase":phase,
            "scope":transition.SCOPES[phase], "authorized_utc":UTC,
            "decision":"go", "retry_allowed":False,
            "expected_commit":COMMIT,
            "expected_source_set_sha256":codec.tree_digest(sources),
            "expected_environment_sha256":codec.tree_digest(environment),
            "expected_environment_role":"cpu_audit" if phase == "audit" else "native_source",
            "expected_gpu_uuid":expected_gpu,
            "root_binding":copy.deepcopy(prior["root_binding"]),
            "journal_pin":copy.deepcopy(prior["journal_pin"]),
            "previous_boundary_ref":copy.deepcopy(prior["boundary_ref"])}

    def write_permission(self, prior, phase, sources, environment):
        value = self.permission_input(prior, phase, sources, environment)
        raw, complete = transition.authenticate_permission_for_write(
            value, prior_boundary=prior,
            attempt_parent=Path(prior["journal_pin"]["path"]).parent,
            fixture=True)
        path = Path(prior["journal_pin"]["path"]).parent / transition.permission_name(phase)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        try:
            os.write(fd, raw)
            os.fsync(fd)
        finally:
            os.close(fd)
        pin = {"path":str(path), "size_bytes":len(raw),
               "sha256":hashlib.sha256(raw).hexdigest()}
        post_raw, post = transition._authenticate_actual(pin, phase, prior, KIND)
        self.assertEqual((post_raw, post), (raw, complete))
        return pin

    def acquire(self, prior, phase, sources, environment, **kwargs):
        pin = self.write_permission(prior, phase, sources, environment)
        guard = runtime.RuntimeGuard(
            phase, profile=runtime.FIXTURE, wall_clock=lambda:2.0,
            cpu_clock=lambda:2.0, rss_probe=lambda:1024,
            cuda_probe=lambda:{"allocated_bytes":0, "reserved_bytes":0, "device":None},
            entry_wall_origin=1.0,
            entry_cpu_origin=1.0, entry_process_id=os.getpid())
        return transition.acquire_phase(
            pin, phase=phase, prior_boundary=prior, entry_wall_origin=1.0,
            entry_cpu_origin=1.0, entry_process_id=os.getpid(),
            collect_metadata=lambda _:(sources, environment), evidence_kind=KIND,
            fixture_guard=guard, **kwargs)

    def close_boundary(self, acquired):
        phase = acquired["phase"]
        records = list(acquired["records"])
        self._advance(acquired["store"], records, acquired["sources"],
                      acquired["environment"], phase)
        root_binding = self._root(acquired["store"])
        root = acquired["store"].root
        acquired["store"].close()
        inspection = control.inspect_native_attempt(
            root, acquired["journal_pin"], expected_store_profile=storage.MLP_FIXTURE,
            expected_evidence_kind=KIND)
        self.assertEqual(inspection["status"],
                         "complete" if phase == "audit" else "sealed_boundary")
        expected_count = {"primary":44, "sensitivity":56, "audit":90}[phase]
        return {"phase":phase, "phase_records":expected_count,
            "root_binding":root_binding, "journal_pin":acquired["journal_pin"],
            "boundary_ref":self.existing_reference(root, control.phase_name(expected_count - 1)),
            "inspection":inspection}


class PhaseTransitionTests(PhaseFixture, unittest.TestCase):
    def setUp(self):
        self.started = time.monotonic()
        self.checkpoint()

    def tearDown(self):
        self.checkpoint()

    def test_prospective_later_permission_fits_eight_kib(self):
        import process_supervision as supervision
        admission = {"path":supervision.STORAGE_ADMISSION_PATH,
            "size_bytes":supervision.STORAGE_ADMISSION_MAX, "sha256":"f" * 64}
        root = {"path":"/tmp/spectral-experiment-artifacts/i7-artifacts-" + "a" * 8,
            "device":(1 << 64) - 1, "inode":(1 << 64) - 1,
            "header_sha256":"f" * 64}
        journal = {"path":transition.DEVELOPMENT_ATTEMPT_PATH +
                           "/native-launch-journal.json",
            "size_bytes":control.EXTERNAL_JOURNAL_MAX, "sha256":"f" * 64}
        name = "native-phase-009.json"
        reference = {"name":name, "status":"complete", "encoding":"bytes",
            "size_bytes":128 << 10, "sha256":"f" * 64,
            "receipt_name":"receipt-" + hashlib.sha256(
                name.encode("ascii")).hexdigest() + ".json",
            "receipt_size_bytes":4096, "receipt_sha256":"f" * 64}
        permission = {"schema":transition.PERMISSION_SCHEMA,
            "attempt_id":transition.ATTEMPT_ID, "phase":"primary",
            "scope":transition.SCOPES["primary"],
            "authorized_utc":"9999-12-31T23:59:59Z", "decision":"go",
            "retry_allowed":False, "expected_commit":"f" * 40,
            "expected_source_set_sha256":"f" * 64,
            "expected_environment_sha256":"f" * 64,
            "expected_environment_role":"native_source",
            "expected_gpu_uuid":"GPU-ffffffff-ffff-ffff-ffff-ffffffffffff",
            "storage_admission_pin":admission, "root_binding":root,
            "journal_pin":journal, "previous_boundary_ref":reference}
        raw = transition._json_bytes(permission)
        pin = {"path":transition.DEVELOPMENT_ATTEMPT_PATH +
                      "/native-primary-permission.json",
            "size_bytes":len(raw), "sha256":hashlib.sha256(raw).hexdigest()}
        self.assertEqual(transition._permission(
            raw, pin, "primary", fixture=False), permission)
        self.assertEqual(len(raw), 1734)
        self.assertLessEqual(len(raw), transition.PERMISSION_MAX)

    def test_import_and_default_cli_are_torch_free_and_inert(self):
        with tempfile.TemporaryDirectory() as cwd:
            code = ("import importlib.util,sys;"
                "s=importlib.util.spec_from_file_location('isolated',sys.argv[1]);"
                "m=importlib.util.module_from_spec(s);s.loader.exec_module(m);"
                "print('torch' in sys.modules)")
            result = subprocess.run([sys.executable, "-c", code,
                str(HERE / "phase_transition.py")], cwd=cwd, check=True,
                capture_output=True, text=True, timeout=10)
            self.assertEqual(result.stdout, "False\n")
            result = subprocess.run([sys.executable, str(HERE / "phase_transition.py")],
                cwd=cwd, check=True, capture_output=True, text=True, timeout=10)
            self.assertEqual(result.stdout, '{"status":"inert"}\n')
            self.assertEqual(os.listdir(cwd), [])

    def test_primary_acquisition_consumes_once_and_validates_held_writer(self):
        with self.outer() as outer:
            prior, sources, native = self.development_boundary(outer)
            acquired = self.acquire(prior, "primary", sources, native)
            try:
                self.assertIs(transition.validate_acquired_phase(acquired), acquired)
                self.assertEqual((len(acquired["records"]), acquired["phase"]), (10, "primary"))
                self.assertFalse(acquired["execution_authorized"])
            finally:
                acquired["store"].close()
            report = control.inspect_native_attempt(
                prior["root_binding"]["path"], prior["journal_pin"],
                expected_store_profile=storage.MLP_FIXTURE, expected_evidence_kind=KIND)
            self.assertEqual(report["status"], "transition_incomplete")
            self.assertFalse(report["actual_artifacts_verified"])
            with self.assertRaisesRegex(transition.TransitionError, "reopen-eligible"):
                transition.acquire_phase(
                    acquired["permission_pin"], phase="primary", prior_boundary=prior,
                    entry_wall_origin=1.0, entry_cpu_origin=1.0,
                    entry_process_id=os.getpid(), collect_metadata=lambda _:(sources, native),
                    evidence_kind=KIND)
            control.record_boundary_failure(
                prior["journal_pin"], prior["root_binding"],
                expected_evidence_kind=KIND)
            failed = control.inspect_native_attempt(
                prior["root_binding"]["path"], prior["journal_pin"],
                expected_store_profile=storage.MLP_FIXTURE, expected_evidence_kind=KIND)
            self.assertEqual((failed["status"], failed["reason"]),
                             ("launcher_failed_with_root", "launcher_failure_precedence"))

    def test_all_three_phase_acquisitions_bind_environments_and_boundaries(self):
        with self.outer() as outer:
            prior, sources, native = self.development_boundary(outer)
            primary = self.acquire(prior, "primary", sources, native)
            prior = self.close_boundary(primary)
            sensitivity = self.acquire(prior, "sensitivity", sources, native)
            self.assertEqual(codec.json_bytes(sensitivity["environment"]),
                             codec.json_bytes(primary["environment"]))
            prior = self.close_boundary(sensitivity)
            audit_environment = self.audit_environment(native)
            audit = self.acquire(prior, "audit", sources, audit_environment)
            self.assertEqual(audit["environment"]["runtime_role"], "cpu_audit")
            self.assertEqual(audit["source_environment"]["runtime_role"], "native_source")
            prior = self.close_boundary(audit)
            self.assertEqual(prior["phase_records"], 90)
            self.assertEqual(prior["inspection"]["status"], "complete")

    def test_marker_only_and_permission_only_are_nonresumable(self):
        with self.outer() as outer:
            prior, sources, native = self.development_boundary(outer)
            pin = self.write_permission(prior, "primary", sources, native)
            guard = runtime.RuntimeGuard(
                "primary", profile=runtime.FIXTURE, wall_clock=lambda:2.0,
                cpu_clock=lambda:2.0, rss_probe=lambda:1024,
                cuda_probe=lambda:{"allocated_bytes":0, "reserved_bytes":0, "device":None},
                entry_wall_origin=1.0,
                entry_cpu_origin=1.0, entry_process_id=os.getpid())
            with self.assertRaises(transition.TransitionError) as caught:
                transition.acquire_phase(
                    pin, phase="primary", prior_boundary=prior,
                    entry_wall_origin=1.0, entry_cpu_origin=1.0,
                    entry_process_id=os.getpid(),
                    collect_metadata=lambda _:(_ for _ in ()).throw(RuntimeError("secret")),
                    evidence_kind=KIND, fixture_guard=guard)
            self.assertEqual(caught.exception.failure_status, "transition_marker_retained")
            report = control.inspect_native_attempt(
                prior["root_binding"]["path"], prior["journal_pin"],
                expected_store_profile=storage.MLP_FIXTURE, expected_evidence_kind=KIND)
            self.assertEqual(report["status"], "transition_incomplete")

            store = storage.ArtifactStore.reopen(
                prior["root_binding"]["path"], expected_profile=storage.MLP_FIXTURE,
                expected_header_sha256=prior["root_binding"]["header_sha256"],
                expected_root_identity=(prior["root_binding"]["device"],
                                        prior["root_binding"]["inode"]))
            store.write_bytes(transition.permission_name("primary"), Path(pin["path"]).read_bytes())
            store.close()
            report = control.inspect_native_attempt(
                prior["root_binding"]["path"], prior["journal_pin"],
                expected_store_profile=storage.MLP_FIXTURE, expected_evidence_kind=KIND)
            self.assertEqual(report["status"], "transition_incomplete")
            self.assertFalse(report["eligible_for_writable_reopen"])

    def test_partial_marker_and_unretained_store_failure_are_reported_exactly(self):
        with self.outer() as outer:
            prior, sources, native = self.development_boundary(outer)
            pin = self.write_permission(prior, "primary", sources, native)
            guard = runtime.RuntimeGuard(
                "primary", profile=runtime.FIXTURE, wall_clock=lambda:2.0,
                cpu_clock=lambda:2.0, rss_probe=lambda:1024,
                cuda_probe=lambda:{"allocated_bytes":0, "reserved_bytes":0, "device":None},
                entry_wall_origin=1.0,
                entry_cpu_origin=1.0, entry_process_id=os.getpid())
            def partial(dirfd, name, raw, maximum):
                fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                             0o600, dir_fd=dirfd)
                os.write(fd, b"{")
                os.close(fd)
                raise OSError("fixture partial marker")
            with mock.patch.object(control, "_write_new", side_effect=partial), \
                    self.assertRaises(transition.TransitionError) as caught:
                transition.acquire_phase(
                    pin, phase="primary", prior_boundary=prior,
                    entry_wall_origin=1.0, entry_cpu_origin=1.0,
                    entry_process_id=os.getpid(), collect_metadata=lambda _:(sources, native),
                    evidence_kind=KIND, fixture_guard=guard)
            self.assertEqual(caught.exception.failure_status,
                             "transition_marker_unverified_or_partial")
            report = control.inspect_native_attempt(
                prior["root_binding"]["path"], prior["journal_pin"],
                expected_store_profile=storage.MLP_FIXTURE, expected_evidence_kind=KIND)
            self.assertEqual((report["status"], report["reason"]),
                             ("uninspectable", "transition_evidence_invalid"))

        with self.outer() as outer:
            prior, sources, native = self.development_boundary(outer)
            pin = self.write_permission(prior, "primary", sources, native)
            held = storage.ArtifactStore.reopen(
                prior["root_binding"]["path"], expected_profile=storage.MLP_FIXTURE,
                expected_header_sha256=prior["root_binding"]["header_sha256"],
                expected_root_identity=(prior["root_binding"]["device"],
                                        prior["root_binding"]["inode"]))
            guard = runtime.RuntimeGuard(
                "primary", profile=runtime.FIXTURE, wall_clock=lambda:2.0,
                cpu_clock=lambda:2.0, rss_probe=lambda:1024,
                cuda_probe=lambda:{"allocated_bytes":0, "reserved_bytes":0, "device":None},
                entry_wall_origin=1.0, entry_cpu_origin=1.0,
                entry_process_id=os.getpid())
            def unavailable(_name, _raw):
                held._terminal = True
                held.failure_metadata_status = "not_retained:fixture"
                raise storage.StoreError("fixture unretained write failure")
            held.write_bytes = unavailable
            with mock.patch.object(control, "inspect_native_attempt",
                                   return_value=prior["inspection"]), \
                    mock.patch.object(storage.ArtifactStore, "reopen",
                                      return_value=held), \
                    self.assertRaises(transition.TransitionError) as caught:
                transition.acquire_phase(
                    pin, phase="primary", prior_boundary=prior,
                    entry_wall_origin=1.0, entry_cpu_origin=1.0,
                    entry_process_id=os.getpid(), collect_metadata=lambda _:(sources, native),
                    evidence_kind=KIND, fixture_guard=guard)
            self.assertEqual(caught.exception.failure_status,
                             "transition_marker_retained_store_failure_unavailable")
            self.assertTrue(held._closed)

    def test_permission_calendar_and_actual_predecessor_are_strict(self):
        with self.outer() as outer:
            prior, sources, native = self.development_boundary(outer)
            pin = self.write_permission(prior, "primary", sources, native)
            value = codec.json_loads(Path(pin["path"]).read_bytes(),
                                     max_bytes=transition.PERMISSION_MAX)
            value["authorized_utc"] = "2026-02-31T18:00:00Z"
            bad_raw = codec.json_bytes(value)
            bad_pin = dict(pin, size_bytes=len(bad_raw),
                           sha256=hashlib.sha256(bad_raw).hexdigest())
            Path(pin["path"]).write_bytes(bad_raw)
            calls = []
            with self.assertRaisesRegex(transition.TransitionError, "UTC"):
                transition.acquire_phase(
                    bad_pin, phase="primary", prior_boundary=prior,
                    entry_wall_origin=1.0, entry_cpu_origin=1.0,
                    entry_process_id=os.getpid(),
                    collect_metadata=lambda _:calls.append(1), evidence_kind=KIND)
            self.assertEqual(calls, [])
            self.assertFalse((Path(pin["path"]).parent /
                              transition.transition_name("primary")).exists())

        with self.outer() as outer:
            prior, sources, native = self.development_boundary(outer)
            pin = self.write_permission(prior, "primary", sources, native)
            permission = codec.json_loads(Path(pin["path"]).read_bytes(),
                                          max_bytes=transition.PERMISSION_MAX)
            permission["previous_boundary_ref"]["sha256"] = "0" * 64
            permission_raw = codec.json_bytes(permission)
            Path(pin["path"]).write_bytes(permission_raw)
            pin = {"path":pin["path"], "size_bytes":len(permission_raw),
                   "sha256":hashlib.sha256(permission_raw).hexdigest()}
            marker = {"schema":transition.TRANSITION_SCHEMA, "status":"consumed",
                "attempt_id":transition.ATTEMPT_ID, "phase":"primary",
                "scope":transition.SCOPES["primary"], "permission_pin":pin,
                "root_binding":permission["root_binding"],
                "journal_pin":permission["journal_pin"],
                "previous_boundary_ref":permission["previous_boundary_ref"],
                "retry_allowed":False}
            marker_path = Path(pin["path"]).parent / transition.transition_name("primary")
            marker_path.write_bytes(codec.json_bytes(marker))
            store = storage.ArtifactStore.reopen(
                prior["root_binding"]["path"], expected_profile=storage.MLP_FIXTURE,
                expected_header_sha256=prior["root_binding"]["header_sha256"],
                expected_root_identity=(prior["root_binding"]["device"],
                                        prior["root_binding"]["inode"]))
            store.write_bytes(transition.permission_name("primary"), permission_raw)
            store.write_bytes(transition.environment_name("primary"), codec.json_bytes(native))
            store.close()
            report = control.inspect_native_attempt(
                prior["root_binding"]["path"], prior["journal_pin"],
                expected_store_profile=storage.MLP_FIXTURE, expected_evidence_kind=KIND)
            self.assertEqual((report["status"], report["reason"]),
                             ("uninspectable", "transition_evidence_invalid"))

    def test_storage_admission_chain_rejects_legacy_substitution_and_unknown_names(self):
        with self.outer() as outer:
            prior, sources, native = self.development_boundary(outer)
            pin = self.write_permission(prior, "primary", sources, native)
            raw = Path(pin["path"]).read_bytes()
            permission = codec.json_loads(raw, max_bytes=transition.PERMISSION_MAX)
            journal = control._validate_journal(control._json_loads(
                Path(prior["journal_pin"]["path"]).read_bytes(),
                control.EXTERNAL_JOURNAL_MAX), fixture=True)
            self.assertEqual(permission["storage_admission_pin"],
                             journal["storage_admission_pin"])
            self.assertLessEqual(len(raw), transition.PERMISSION_MAX)

            legacy = copy.deepcopy(permission)
            legacy["schema"] = "i7_native_phase_permission_v1"
            del legacy["storage_admission_pin"]
            legacy_raw = codec.json_bytes(legacy)
            with self.assertRaisesRegex(transition.TransitionError,
                                        "exact ordered fields"):
                transition._permission(legacy_raw, pin, "primary", fixture=True)

            changed = copy.deepcopy(permission)
            changed["storage_admission_pin"]["sha256"] = "0" * 64
            changed_raw = codec.json_bytes(changed)
            Path(pin["path"]).write_bytes(changed_raw)
            changed_pin = {"path":pin["path"], "size_bytes":len(changed_raw),
                "sha256":hashlib.sha256(changed_raw).hexdigest()}
            with self.assertRaisesRegex(transition.TransitionError,
                                        "authenticated journal"):
                transition._authenticate_actual(
                    changed_pin, "primary", prior, KIND)

        with self.outer() as outer:
            prior, sources, native = self.development_boundary(outer)
            attempt = Path(prior["journal_pin"]["path"]).parent
            for index in range(7):
                (attempt / f"unknown-permission-sidecar-{index}").write_bytes(b"x")
            with self.assertRaisesRegex(transition.TransitionError,
                                        "exceeds cardinality"):
                self.write_permission(prior, "primary", sources, native)

        with self.outer() as outer:
            prior, sources, native = self.development_boundary(outer)
            candidate = self.permission_input(prior, "primary", sources, native)
            candidate["root_binding"]["path"] = "/tmp/spectral-experiment-artifacts/" + "x" * 9000
            with self.assertRaisesRegex(transition.TransitionError,
                                        "bounded JSON domain"):
                transition.authenticate_permission_for_write(
                    candidate, prior_boundary=prior,
                    attempt_parent=Path(prior["journal_pin"]["path"]).parent,
                    fixture=True)

    def test_stale_phase_record_environment_is_structured_uninspectable(self):
        with self.outer() as outer:
            prior, sources, native = self.development_boundary(outer)
            acquired = self.acquire(prior, "primary", sources, native)
            stale = copy.deepcopy(native)
            stale["safe_environment"]["PYTHONHASHSEED"] = "fixture-stale"
            operation = policy.schedule()[6]
            self.assertEqual(operation["operation"], "scientific_plans")
            self._append(acquired["store"], acquired["records"], operation,
                         "started", [], sources, stale)
            acquired["store"].close()
            report = control.inspect_native_attempt(
                prior["root_binding"]["path"], prior["journal_pin"],
                expected_store_profile=storage.MLP_FIXTURE, expected_evidence_kind=KIND)
            self.assertEqual((report["status"], report["reason"]),
                             ("uninspectable", "transition_evidence_invalid"))
            self.assertFalse(report["eligible_for_writable_reopen"])

    def test_held_snapshot_rejects_consistent_payload_replacement_before_copy(self):
        with self.outer() as outer:
            prior, sources, native = self.development_boundary(outer)
            pin = self.write_permission(prior, "primary", sources, native)
            original_inspect = control.inspect_native_attempt
            def inspect_then_replace(*args, **kwargs):
                report = original_inspect(*args, **kwargs)
                writer = storage.ArtifactStore.reopen(
                    prior["root_binding"]["path"], expected_profile=storage.MLP_FIXTURE,
                    expected_header_sha256=prior["root_binding"]["header_sha256"],
                    expected_root_identity=(prior["root_binding"]["device"],
                                            prior["root_binding"]["inode"]))
                name = policy.plan_name(71990)
                receipt_name = "receipt-" + hashlib.sha256(name.encode("ascii")).hexdigest() + ".json"
                os.unlink(writer.root / name)
                os.unlink(writer.root / receipt_name)
                payload = b"consistent replacement"
                receipt = storage._json_bytes({"schema":"i7_artifact_receipt_v1",
                    "name":name, "size":len(payload),
                    "sha256":hashlib.sha256(payload).hexdigest(), "status":"complete",
                    "encoding":"torch_weights_only"})
                for target, body in ((name, payload), (receipt_name, receipt)):
                    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                                 0o600, dir_fd=writer._dirfd)
                    try:
                        os.write(fd, body)
                        os.fsync(fd)
                    finally:
                        os.close(fd)
                writer.close()
                return report
            guard = runtime.RuntimeGuard(
                "primary", profile=runtime.FIXTURE, wall_clock=lambda:2.0,
                cpu_clock=lambda:2.0, rss_probe=lambda:1024,
                cuda_probe=lambda:{"allocated_bytes":0, "reserved_bytes":0, "device":None},
                entry_wall_origin=1.0, entry_cpu_origin=1.0,
                entry_process_id=os.getpid())
            with mock.patch.object(control, "inspect_native_attempt",
                                   side_effect=inspect_then_replace), \
                    self.assertRaises(transition.TransitionError) as caught:
                transition.acquire_phase(
                    pin, phase="primary", prior_boundary=prior,
                    entry_wall_origin=1.0, entry_cpu_origin=1.0,
                    entry_process_id=os.getpid(), collect_metadata=lambda _:(sources, native),
                    evidence_kind=KIND, fixture_guard=guard)
            self.assertEqual(caught.exception.failure_status,
                             "transition_marker_and_store_failure_retained")
            self.assertFalse((Path(prior["root_binding"]["path"]) /
                              transition.permission_name("primary")).exists())

        with self.outer() as outer:
            prior, sources, native = self.development_boundary(outer)
            pin = self.write_permission(prior, "primary", sources, native)
            original_inspect = control.inspect_native_attempt
            def inspect_then_add_future(*args, **kwargs):
                report = original_inspect(*args, **kwargs)
                future_ref = copy.deepcopy(prior["boundary_ref"])
                future_ref["name"] = control.phase_name(43)
                future_ref["receipt_name"] = ("receipt-" + hashlib.sha256(
                    future_ref["name"].encode("ascii")).hexdigest() + ".json")
                future = {"schema":transition.PERMISSION_SCHEMA,
                    "attempt_id":transition.ATTEMPT_ID, "phase":"sensitivity",
                    "scope":transition.SCOPES["sensitivity"], "authorized_utc":UTC,
                    "decision":"go", "retry_allowed":False, "expected_commit":COMMIT,
                    "expected_source_set_sha256":codec.tree_digest(sources),
                    "expected_environment_sha256":codec.tree_digest(native),
                    "expected_environment_role":"native_source", "expected_gpu_uuid":UUID,
                    "storage_admission_pin":codec.json_loads(
                        Path(pin["path"]).read_bytes(),
                        max_bytes=transition.PERMISSION_MAX)["storage_admission_pin"],
                    "root_binding":prior["root_binding"], "journal_pin":prior["journal_pin"],
                    "previous_boundary_ref":future_ref}
                path = Path(pin["path"]).parent / transition.permission_name("sensitivity")
                path.write_bytes(codec.json_bytes(future))
                return report
            guard = runtime.RuntimeGuard(
                "primary", profile=runtime.FIXTURE, wall_clock=lambda:2.0,
                cpu_clock=lambda:2.0, rss_probe=lambda:1024,
                cuda_probe=lambda:{"allocated_bytes":0, "reserved_bytes":0, "device":None},
                entry_wall_origin=1.0, entry_cpu_origin=1.0,
                entry_process_id=os.getpid())
            with mock.patch.object(control, "inspect_native_attempt",
                                   side_effect=inspect_then_add_future), \
                    self.assertRaises(transition.TransitionError):
                transition.acquire_phase(
                    pin, phase="primary", prior_boundary=prior,
                    entry_wall_origin=1.0, entry_cpu_origin=1.0,
                    entry_process_id=os.getpid(), collect_metadata=lambda _:(sources, native),
                    evidence_kind=KIND, fixture_guard=guard)
            self.assertFalse((Path(prior["root_binding"]["path"]) /
                              transition.permission_name("primary")).exists())

        with self.outer() as outer:
            prior, sources, native = self.development_boundary(outer)
            pin = self.write_permission(prior, "primary", sources, native)
            original_inspect = control.inspect_native_attempt
            def inspect_then_fail(*args, **kwargs):
                report = original_inspect(*args, **kwargs)
                control.record_boundary_failure(
                    prior["journal_pin"], prior["root_binding"],
                    expected_evidence_kind=KIND)
                return report
            guard = runtime.RuntimeGuard(
                "primary", profile=runtime.FIXTURE, wall_clock=lambda:2.0,
                cpu_clock=lambda:2.0, rss_probe=lambda:1024,
                cuda_probe=lambda:{"allocated_bytes":0, "reserved_bytes":0, "device":None},
                entry_wall_origin=1.0, entry_cpu_origin=1.0,
                entry_process_id=os.getpid())
            with mock.patch.object(control, "inspect_native_attempt",
                                   side_effect=inspect_then_fail), \
                    self.assertRaises(transition.TransitionError) as caught:
                transition.acquire_phase(
                    pin, phase="primary", prior_boundary=prior,
                    entry_wall_origin=1.0, entry_cpu_origin=1.0,
                    entry_process_id=os.getpid(), collect_metadata=lambda _:(sources, native),
                    evidence_kind=KIND, fixture_guard=guard)
            self.assertEqual(caught.exception.failure_status,
                             "transition_marker_and_store_failure_retained")
            self.assertFalse((Path(prior["root_binding"]["path"]) /
                              transition.permission_name("primary")).exists())

    def test_actual_reference_has_no_small_payload_read_cap(self):
        with self.outer() as outer:
            store = storage.ArtifactStore(
                outer, profile=storage.MLP_FIXTURE, budget_bytes=BUDGET,
                failure_reserve_bytes=RESERVE, min_filesystem_free_bytes=MIN_FREE)
            name = "bounded-large-storage-fixture.bin"
            payload = b"storage-only-nonsemantic:" + b"x" * (140 << 10)
            receipt = store.write_bytes(name, payload)
            report, scanned = storage._inspect_dirfd(store._dirfd)
            receipts = {row["name"]:row for row in report["receipts"]}
            reference = transition._actual_reference(
                name, receipts, scanned, store._dirfd, storage)
            self.assertEqual((reference["size_bytes"], reference["sha256"]),
                             (len(payload), hashlib.sha256(payload).hexdigest()))
            self.assertGreater(reference["size_bytes"], 128 << 10)
            store.close()


if __name__ == "__main__":
    unittest.main()
