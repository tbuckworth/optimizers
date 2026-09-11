"""CPU-only native-control fixtures; no study plans, data, source loop or CUDA."""
from __future__ import annotations

from contextlib import contextmanager, ExitStack
import copy
import fcntl
import hashlib
import json
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
    raise RuntimeError("native-control tests require CUDA_VISIBLE_DEVICES='' ")

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import torch
import artifact_store as storage
import identity_codec as codec
import native_control as control
import native_phase_policy as policy
import runtime_guard as runtime
from test_native_layout_inspection import _success_report


UUID = "GPU-994b97cd-d768-1f03-78fb-6a70e3cc1e0c"
COMMIT = "a" * 40
ATTEMPT = "i7-native-development-attempt-001"
UTC = "2026-09-07T12:00:00Z"
KIND = "synthetic_contract_fixture"
BIG_TMP = Path("/tmp/spectral-experiment-artifacts")
BUDGET = 2 << 20
RESERVE = 1 << 20
MIN_FREE = 1 << 30


class NativeControlTests(unittest.TestCase):
    def setUp(self):
        self.started = time.monotonic()
        self.guard()

    def tearDown(self):
        self.guard()

    def guard(self):
        self.assertLessEqual(time.monotonic() - self.started, 120.0)
        self.assertLessEqual(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                             2 << 30)
        for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS"):
            self.assertEqual(os.environ.get(key), "1")
        self.assertEqual(torch.get_num_threads(), 1)
        self.assertFalse(torch.cuda.is_initialized())

    @contextmanager
    def outer(self):
        self.guard()
        runtime.verify_big_volume(BIG_TMP)
        self.assertGreaterEqual(shutil.disk_usage(BIG_TMP).free, MIN_FREE)
        with tempfile.TemporaryDirectory(prefix="i7-native-control-test-", dir=BIG_TMP) as outer:
            path = Path(outer)
            yield outer
            logical = 0
            for current, directories, files in os.walk(path):
                del directories
                for name in files:
                    info = os.stat(Path(current) / name, follow_symlinks=False)
                    if stat.S_ISREG(info.st_mode):
                        logical += info.st_size
            self.assertLessEqual(logical, BUDGET)
            for header in path.rglob("store-header.json"):
                report = storage.ArtifactStore.inspect(header.parent)
                failure_bytes = sum(row["size"] for name, row in report["files"].items()
                                    if name.startswith("failure-"))
                self.assertLessEqual(report["logical_bytes"] - failure_bytes,
                                     BUDGET - RESERVE)
        self.assertFalse(path.exists())
        self.guard()

    @staticmethod
    def metadata():
        report = _success_report()
        return report["source_binding"], report["native_environment_binding"]

    def permission(self, attempt_directory):
        sources, _ = self.metadata()
        source_digest = codec.tree_digest(sources)
        value = {"schema":control.PERMISSION_SCHEMA, "attempt_id":ATTEMPT,
            "phase":"development", "scope":control.SCOPE,
            "expected_commit":COMMIT, "expected_source_set_sha256":source_digest,
            "expected_gpu_uuid":UUID, "storage_admission_pin":{
                "path":str(Path(attempt_directory).parent /
                           "fixture-storage-admission.json"),
                "size_bytes":3, "sha256":hashlib.sha256(b"{}\n").hexdigest()},
            "authorized_utc":UTC, "decision":"go",
            "retry_allowed":False}
        raw = codec.json_bytes(value)
        pins = {"permission_sha256":hashlib.sha256(raw).hexdigest(),
            "attempt_id":ATTEMPT, "expected_commit":COMMIT,
            "expected_source_set_sha256":source_digest, "expected_gpu_uuid":UUID,
            "attempt_directory":str(attempt_directory)}
        return raw, pins

    @staticmethod
    def store_factory(parent, events=None):
        def create():
            if events is not None:
                events.append("root")
            return storage.ArtifactStore(parent, profile=storage.MLP_FIXTURE,
                budget_bytes=BUDGET, failure_reserve_bytes=RESERVE,
                min_filesystem_free_bytes=MIN_FREE)
        return create

    def bootstrap(self, outer, *, collect=None, events=None):
        self.guard()
        attempt = Path(outer) / ATTEMPT
        raw, pins = self.permission(attempt)
        if collect is None:
            def collect(_permission):
                if events is not None:
                    events.append("metadata")
                return self.metadata()
        result = control.bootstrap_development(
            raw, pins=pins, entry_wall_origin=1.0, entry_cpu_origin=1.0,
            entry_process_id=os.getpid(),
            collect_metadata=collect, create_store=self.store_factory(outer, events),
            wall_clock=(lambda: (events.append("wall"), 2.0)[1]) if events is not None
                else (lambda:2.0),
            cpu_clock=(lambda: (events.append("cpu"), 2.0)[1]) if events is not None
                else (lambda:2.0),
            utc_now=lambda: UTC,
            evidence_kind=KIND)
        permission = codec.json_loads(raw, max_bytes=control.EXTERNAL_JOURNAL_MAX)
        journal_raw = Path(result["journal_pin"]["path"]).read_bytes()
        journal = control._validate_journal(
            control._json_loads(journal_raw, control.EXTERNAL_JOURNAL_MAX),
            fixture=True)
        self.assertEqual(journal["storage_admission_pin"],
                         permission["storage_admission_pin"])
        self.assertEqual((result["store"].root / control.ROOT_JOURNAL_NAME).read_bytes(),
                         journal_raw)
        self.guard()
        return result, pins

    def test_import_and_default_cli_are_torch_free_and_inert(self):
        module = HERE / "native_control.py"
        with tempfile.TemporaryDirectory() as cwd:
            code = ("import importlib.util,sys;"
                "s=importlib.util.spec_from_file_location('native_control_isolated',sys.argv[1]);"
                "m=importlib.util.module_from_spec(s);s.loader.exec_module(m);"
                "print('torch' in sys.modules)")
            result = subprocess.run([sys.executable, "-c", code, str(module)], cwd=cwd,
                check=True, capture_output=True, text=True, timeout=10)
            self.assertEqual(result.stdout, "False\n")
            result = subprocess.run([sys.executable, str(module)], cwd=cwd,
                check=True, capture_output=True, text=True, timeout=10)
            self.assertEqual(result.stdout, '{"status":"inert"}\n')
            self.assertEqual(os.listdir(cwd), [])

    def test_prospective_permission_and_journal_fit_eight_kib(self):
        import process_supervision as supervision
        admission = {"path":supervision.STORAGE_ADMISSION_PATH,
            "size_bytes":supervision.STORAGE_ADMISSION_MAX, "sha256":"f" * 64}
        permission = {"schema":control.PERMISSION_SCHEMA,
            "attempt_id":control.DEVELOPMENT_ATTEMPT_ID, "phase":"development",
            "scope":control.SCOPE, "expected_commit":"f" * 40,
            "expected_source_set_sha256":"f" * 64,
            "expected_gpu_uuid":"GPU-ffffffff-ffff-ffff-ffff-ffffffffffff",
            "storage_admission_pin":admission,
            "authorized_utc":"9999-12-31T23:59:59Z", "decision":"go",
            "retry_allowed":False}
        journal = {"schema":control.JOURNAL_SCHEMA, "status":"consumed",
            "attempt_id":control.DEVELOPMENT_ATTEMPT_ID, "phase":"development",
            "scope":control.SCOPE, "permission_sha256":"f" * 64,
            "expected_commit":"f" * 40,
            "expected_source_set_sha256":"f" * 64,
            "expected_gpu_uuid":"GPU-ffffffff-ffff-ffff-ffff-ffffffffffff",
            "storage_admission_pin":admission,
            "authorized_utc":"9999-12-31T23:59:59Z", "retry_allowed":False,
            "launcher_entry_wall_origin":sys.float_info.max,
            "launcher_entry_cpu_origin":sys.float_info.max,
            "bootstrap_entry_wall_observed":sys.float_info.max,
            "bootstrap_entry_cpu_observed":sys.float_info.max}
        pin_raw = json.dumps(admission, ensure_ascii=True, allow_nan=False,
                             separators=(",", ":")).encode("ascii")
        member = b',"storage_admission_pin":' + pin_raw
        self.assertEqual((len(pin_raw), len(member), len(member) + 14),
                         (277, 302, 316))
        self.assertEqual((len(control._json_bytes(permission)),
                          len(control._json_bytes(journal))), (767, 1071))
        self.assertLessEqual(1071, control.EXTERNAL_JOURNAL_MAX)

    def test_permission_rejection_precedes_attempt_and_callbacks(self):
        with self.outer() as outer:
            attempt = Path(outer) / ATTEMPT
            raw, pins = self.permission(attempt)
            calls = []
            bad = dict(pins)
            bad["permission_sha256"] = "0" * 64
            with self.assertRaisesRegex(control.ControlError, "permission byte hash"):
                control.bootstrap_development(raw, pins=bad, entry_wall_origin=1.0,
                    entry_cpu_origin=1.0, entry_process_id=os.getpid(),
                    collect_metadata=lambda _: calls.append("metadata"),
                    create_store=lambda: calls.append("root"), wall_clock=lambda: 2.0,
                    cpu_clock=lambda: 2.0, utc_now=lambda: UTC,
                    evidence_kind=KIND)
            self.assertEqual(calls, [])
            self.assertFalse(attempt.exists())

            legacy = codec.json_loads(raw, max_bytes=control.EXTERNAL_JOURNAL_MAX)
            legacy["schema"] = "i7_native_launcher_permission_v1"
            del legacy["storage_admission_pin"]
            legacy_raw = codec.json_bytes(legacy)
            legacy_pins = dict(
                pins, permission_sha256=hashlib.sha256(legacy_raw).hexdigest())
            with self.assertRaisesRegex(control.ControlError, "exact ordered keys"):
                control.bootstrap_development(
                    legacy_raw, pins=legacy_pins, entry_wall_origin=1.0,
                    entry_cpu_origin=1.0, entry_process_id=os.getpid(),
                    collect_metadata=lambda _:calls.append("metadata"),
                    create_store=lambda:calls.append("root"), wall_clock=lambda:2.0,
                    cpu_clock=lambda:2.0, utc_now=lambda:UTC, evidence_kind=KIND)
            self.assertEqual(calls, [])
            self.assertFalse(attempt.exists())

            other_attempt = Path(outer) / "i7-native-development-attempt-002"
            permission = codec.json_loads(raw, max_bytes=control.EXTERNAL_JOURNAL_MAX)
            permission["attempt_id"] = "i7-native-development-attempt-002"
            other_raw = codec.json_bytes(permission)
            other_pins = dict(pins, attempt_id="i7-native-development-attempt-002",
                attempt_directory=str(other_attempt),
                permission_sha256=hashlib.sha256(other_raw).hexdigest())
            with self.assertRaisesRegex(control.ControlError, "pinned attempt"):
                control.bootstrap_development(other_raw, pins=other_pins,
                    entry_wall_origin=1.0, entry_cpu_origin=1.0,
                    entry_process_id=os.getpid(),
                    collect_metadata=lambda _: calls.append("metadata"),
                    create_store=lambda: calls.append("root"), wall_clock=lambda:2.0,
                    cpu_clock=lambda:2.0, utc_now=lambda:UTC, evidence_kind=KIND)
            self.assertEqual(calls, [])
            self.assertFalse(other_attempt.exists())

    def test_bootstrap_resource_and_process_gates_are_preimport_and_terminal(self):
        with self.outer() as outer:
            attempt = Path(outer) / ATTEMPT
            raw, pins = self.permission(attempt)
            common = {"pins":pins, "entry_wall_origin":1.0,
                "entry_cpu_origin":1.0, "entry_process_id":os.getpid(),
                "collect_metadata":lambda _:self.metadata(),
                "create_store":self.store_factory(outer), "cpu_clock":lambda:2.0,
                "utc_now":lambda:UTC, "evidence_kind":KIND}
            wall = iter((2.0, 182.0, 182.0, 182.0))
            with mock.patch.object(control, "_modules") as forbidden_import, \
                    self.assertRaisesRegex(control.ControlError, "wall cap"):
                control.bootstrap_development(raw, wall_clock=lambda:next(wall), **common)
            forbidden_import.assert_not_called()
            self.assertTrue((attempt / control.EXTERNAL_JOURNAL_NAME).is_file())
            failed = codec.json_loads(
                (attempt / control.EXTERNAL_FAILURE_NAME).read_bytes(),
                max_bytes=control.EXTERNAL_FAILURE_MAX)
            self.assertEqual((failed["stage"], failed["reason"]),
                ("resource_check", "bootstrap_resource_check_failed"))
            self.assertEqual(failed["resource_snapshot"]["stage"], "journal")
            self.assertEqual(failed["resource_snapshot"]["wall_seconds"], 181.0)

        with self.outer() as outer:
            attempt = Path(outer) / ATTEMPT
            raw, pins = self.permission(attempt)
            with mock.patch.object(control, "_modules") as forbidden_import, \
                    self.assertRaisesRegex(control.ControlError, "RSS cap"):
                control.bootstrap_development(
                    raw, pins=pins, entry_wall_origin=1.0, entry_cpu_origin=1.0,
                    entry_process_id=os.getpid(), collect_metadata=lambda _:self.metadata(),
                    create_store=self.store_factory(outer), wall_clock=lambda:2.0,
                    cpu_clock=lambda:2.0,
                    rss_probe=lambda:control.BOOTSTRAP_RSS_BYTES + 1,
                    utc_now=lambda:UTC, evidence_kind=KIND)
            forbidden_import.assert_not_called()
            self.assertTrue((attempt / control.EXTERNAL_FAILURE_NAME).is_file())

        with self.outer() as outer:
            attempt = Path(outer) / ATTEMPT
            raw, pins = self.permission(attempt)
            with self.assertRaisesRegex(control.ControlError, "different process"):
                control.bootstrap_development(raw, pins=pins, entry_wall_origin=1.0,
                    entry_cpu_origin=1.0, entry_process_id=os.getpid() + 1,
                    collect_metadata=lambda _:self.metadata(),
                    create_store=self.store_factory(outer), wall_clock=lambda:2.0,
                    cpu_clock=lambda:2.0, rss_probe=lambda:0,
                    utc_now=lambda:UTC, evidence_kind=KIND)
            self.assertFalse(attempt.exists())

        with self.outer() as outer:
            attempt = Path(outer) / ATTEMPT
            raw, pins = self.permission(attempt)
            cuda_rows = iter(((0, 0), (0, 0), (0, 0),
                              (control.BOOTSTRAP_CUDA_ALLOCATED_BYTES + 1,
                               control.BOOTSTRAP_CUDA_ALLOCATED_BYTES + 1)))
            with self.assertRaisesRegex(control.ControlError, "CUDA allocation cap"):
                control.bootstrap_development(raw, pins=pins, entry_wall_origin=1.0,
                    entry_cpu_origin=1.0, entry_process_id=os.getpid(),
                    collect_metadata=lambda _:self.metadata(),
                    create_store=self.store_factory(outer), wall_clock=lambda:2.0,
                    cpu_clock=lambda:2.0, rss_probe=lambda:0,
                    cuda_probe=lambda:next(cuda_rows), utc_now=lambda:UTC,
                    evidence_kind=KIND)
            failure = codec.json_loads(
                (attempt / control.EXTERNAL_FAILURE_NAME).read_bytes(),
                max_bytes=control.EXTERNAL_FAILURE_MAX)
            self.assertEqual((failure["stage"], failure["reason"]),
                ("resource_check", "bootstrap_resource_check_failed"))
            self.assertEqual(failure["resource_snapshot"]["stage"], "native_import")
            self.assertIsNone(failure["root_binding"])
            self.assertIsNone(failure["partial_root"])

    def test_native_bootstrap_rejects_overrides_and_preimport_violation(self):
        with self.outer() as outer:
            attempt = Path(outer) / ATTEMPT
            raw, pins = self.permission(attempt)
            calls = []
            origin_wall, origin_cpu = time.monotonic(), time.process_time()
            with self.assertRaisesRegex(control.ControlError, "forbids"):
                control.bootstrap_development(raw, pins=pins,
                    entry_wall_origin=origin_wall, entry_cpu_origin=origin_cpu,
                    entry_process_id=os.getpid(), collect_metadata=lambda _:calls.append(1),
                    create_store=lambda:calls.append(2), wall_clock=lambda:origin_wall,
                    utc_now=lambda:UTC)
            self.assertEqual(calls, [])
            self.assertFalse(attempt.exists())
            with self.assertRaisesRegex(control.ControlError, "Torch absent"):
                control.bootstrap_development(raw, pins=pins,
                    entry_wall_origin=origin_wall, entry_cpu_origin=origin_cpu,
                    entry_process_id=os.getpid(), collect_metadata=lambda _:calls.append(1),
                    create_store=lambda:calls.append(2), utc_now=lambda:UTC)
            self.assertEqual(calls, [])
            self.assertFalse(attempt.exists())

    def test_bootstrap_order_exact_go_first_and_pinned_journal(self):
        events = []
        original_validate = policy.validate_record

        def validate(*args, **kwargs):
            events.append("go_validated")
            return original_validate(*args, **kwargs)

        with self.outer() as outer, ExitStack() as stack:
            policy.validate_record = validate
            stack.callback(setattr, policy, "validate_record", original_validate)
            result, _ = self.bootstrap(outer, events=events)
            store = stack.enter_context(result["store"])
            self.assertEqual(events[:2], ["wall", "cpu"])
            self.assertEqual([item for item in events if item not in ("wall", "cpu")],
                             ["metadata", "go_validated", "root"])
            self.assertEqual([row["stage"] for row in result["bootstrap_resources"]],
                ["permission", "volume", "journal", "native_import", "metadata",
                 "go_formation", "root", "phase_zero_copy", "journal_copy",
                 "source_copy", "environment_copy", "return"])
            self.assertTrue(all(tuple(row) == control.RESOURCE_KEYS
                                for row in result["bootstrap_resources"]))
            self.assertEqual((store.root / control.phase_name(0)).read_bytes(),
                             result["go_bytes"])
            self.assertEqual(result["go_record"]["created_utc"], UTC)
            self.assertIsNone(result["go_record"]["root_binding"])
            pin = result["journal_pin"]
            raw = Path(pin["path"]).read_bytes()
            self.assertEqual((len(raw), hashlib.sha256(raw).hexdigest()),
                             (pin["size_bytes"], pin["sha256"]))
            journal = codec.json_loads(raw, max_bytes=control.EXTERNAL_JOURNAL_MAX)
            self.assertEqual((journal["launcher_entry_wall_origin"],
                              journal["bootstrap_entry_wall_observed"]), (1.0, 2.0))
            self.assertEqual(control.cumulative_clocks(result["clock_origins"],
                wall_clock=lambda:3.5, cpu_clock=lambda:2.5),
                {"wall_seconds":2.5, "cpu_seconds":1.5})
            report = storage.ArtifactStore.inspect(store.root)
            complete_names = {row["name"] for row in report["receipts"]}
            self.assertEqual(complete_names, {control.phase_name(0),
                control.ROOT_JOURNAL_NAME, control.SOURCE_NAME, control.ENVIRONMENT_NAME})
            self.assertFalse(torch.cuda.is_initialized())

    def test_pre_root_failure_consumes_attempt_and_second_call_is_refused(self):
        with self.outer() as outer:
            attempt = Path(outer) / ATTEMPT
            raw, pins = self.permission(attempt)

            def fail(_):
                raise RuntimeError("arbitrary fixture detail must not persist")

            kwargs = dict(pins=pins, entry_wall_origin=1.0, entry_cpu_origin=1.0,
                entry_process_id=os.getpid(),
                collect_metadata=fail, create_store=self.store_factory(outer),
                wall_clock=lambda:2.0, cpu_clock=lambda:2.0, utc_now=lambda:UTC,
                evidence_kind=KIND)
            with self.assertRaisesRegex(control.ControlError, "metadata_collection"):
                control.bootstrap_development(raw, **kwargs)
            self.assertTrue((attempt / control.EXTERNAL_JOURNAL_NAME).is_file())
            failure_raw = (attempt / control.EXTERNAL_FAILURE_NAME).read_bytes()
            self.assertNotIn(b"arbitrary fixture", failure_raw)
            failure = codec.json_loads(failure_raw, max_bytes=control.EXTERNAL_FAILURE_MAX)
            resource_snapshot = failure["resource_snapshot"]
            self.assertEqual(failure, {"schema":control.FAILURE_SCHEMA, "status":"failed",
                "attempt_id":ATTEMPT, "stage":"metadata_collection",
                "reason":"metadata_collection_failed", "retry_allowed":False,
                "root_binding":None, "partial_root":None,
                "resource_snapshot":resource_snapshot})
            self.assertEqual(resource_snapshot["stage"], "native_import")
            journal_raw = (attempt / control.EXTERNAL_JOURNAL_NAME).read_bytes()
            pin = {"path":str(attempt / control.EXTERNAL_JOURNAL_NAME),
                "size_bytes":len(journal_raw),
                "sha256":hashlib.sha256(journal_raw).hexdigest()}
            inspected = control.inspect_native_attempt(None, pin,
                expected_store_profile=storage.MLP_FIXTURE, expected_evidence_kind=KIND)
            self.assertEqual((inspected["status"], inspected["reason"]),
                             ("pre_root_failed", "launcher_failure_precedence"))
            self.assertIs(inspected["can_resume_incomplete"], False)
            legacy = {key:value for key, value in failure.items()
                      if key not in ("partial_root", "resource_snapshot")}
            legacy["schema"] = control.FAILURE_SCHEMA_V1
            (attempt / control.EXTERNAL_FAILURE_NAME).write_bytes(codec.json_bytes(legacy))
            legacy_report = control.inspect_native_attempt(None, pin,
                expected_store_profile=storage.MLP_FIXTURE, expected_evidence_kind=KIND)
            self.assertEqual(legacy_report["status"], "pre_root_failed")
            unrelated = self.store_factory(outer)()
            unrelated_root = unrelated.root
            unrelated.close()
            conflict = control.inspect_native_attempt(unrelated_root, pin,
                expected_store_profile=storage.MLP_FIXTURE, expected_evidence_kind=KIND)
            self.assertEqual((conflict["status"], conflict["reason"]),
                ("uninspectable", "failure_binding_conflicts_with_supplied_root"))
            with self.assertRaisesRegex(control.ControlError, "retry refused"):
                control.bootstrap_development(raw, **kwargs)
            self.assertEqual({path.name for path in attempt.iterdir()},
                             {control.EXTERNAL_FAILURE_NAME, control.EXTERNAL_JOURNAL_NAME})

    def test_launcher_failure_with_partial_root_is_terminal_on_inspection(self):
        with self.outer() as outer:
            attempt = Path(outer) / ATTEMPT
            raw, pins = self.permission(attempt)
            holder = {}

            def create():
                store = self.store_factory(outer)()
                holder["store"] = store
                original = store.write_bytes
                def write(name, body):
                    if name == control.ENVIRONMENT_NAME:
                        raise RuntimeError("fixture metadata-copy interruption")
                    return original(name, body)
                store.write_bytes = write
                return store

            with self.assertRaisesRegex(control.ControlError, "metadata_copy"):
                control.bootstrap_development(raw, pins=pins, entry_wall_origin=1.0,
                    entry_cpu_origin=1.0, entry_process_id=os.getpid(),
                    collect_metadata=lambda _:self.metadata(),
                    create_store=create, wall_clock=lambda:2.0, cpu_clock=lambda:2.0,
                    utc_now=lambda:UTC, evidence_kind=KIND)
            root = holder["store"].root
            journal_raw = (attempt / control.EXTERNAL_JOURNAL_NAME).read_bytes()
            pin = {"path":str(attempt / control.EXTERNAL_JOURNAL_NAME),
                "size_bytes":len(journal_raw),
                "sha256":hashlib.sha256(journal_raw).hexdigest()}
            report = control.inspect_native_attempt(root, pin,
                expected_store_profile=storage.MLP_FIXTURE, expected_evidence_kind=KIND)
            self.assertEqual((report["status"], report["reason"]),
                             ("launcher_failed_with_root", "launcher_failure_precedence"))
            self.assertIs(report["eligible_for_writable_reopen"], False)
            failure_path = attempt / control.EXTERNAL_FAILURE_NAME
            failure = codec.json_loads(failure_path.read_bytes(),
                                       max_bytes=control.EXTERNAL_FAILURE_MAX)
            failure["root_binding"]["inode"] += 1
            failure_path.write_bytes(codec.json_bytes(failure))
            mismatch = control.inspect_native_attempt(root, pin,
                expected_store_profile=storage.MLP_FIXTURE, expected_evidence_kind=KIND)
            self.assertEqual((mismatch["status"], mismatch["reason"]),
                             ("uninspectable", "launcher_failure_root_binding_mismatch"))

    def test_structured_initialization_partial_root_is_terminal_and_nonresumable(self):
        with self.outer() as outer:
            attempt = Path(outer) / ATTEMPT
            raw, pins = self.permission(attempt)
            free = os.statvfs(outer).f_bavail * os.statvfs(outer).f_frsize

            def fail_initialization():
                return storage.ArtifactStore(
                    outer, profile=storage.MLP_FIXTURE, budget_bytes=BUDGET,
                    failure_reserve_bytes=RESERVE,
                    min_filesystem_free_bytes=free + BUDGET)

            with self.assertRaisesRegex(control.ControlError, "root_creation"):
                control.bootstrap_development(
                    raw, pins=pins, entry_wall_origin=1.0, entry_cpu_origin=1.0,
                    entry_process_id=os.getpid(), collect_metadata=lambda _:self.metadata(),
                    create_store=fail_initialization, wall_clock=lambda:2.0,
                    cpu_clock=lambda:2.0, utc_now=lambda:UTC, evidence_kind=KIND)
            failure = codec.json_loads(
                (attempt / control.EXTERNAL_FAILURE_NAME).read_bytes(),
                max_bytes=control.EXTERNAL_FAILURE_MAX)
            self.assertIsNone(failure["root_binding"])
            partial = failure["partial_root"]
            self.assertEqual(partial["header_status"], "absent")
            root = Path(partial["path"])
            self.assertTrue(root.is_dir())
            journal_raw = (attempt / control.EXTERNAL_JOURNAL_NAME).read_bytes()
            pin = {"path":str(attempt / control.EXTERNAL_JOURNAL_NAME),
                "size_bytes":len(journal_raw),
                "sha256":hashlib.sha256(journal_raw).hexdigest()}
            report = control.inspect_native_attempt(
                root, pin, expected_store_profile=storage.MLP_FIXTURE,
                expected_evidence_kind=KIND)
            self.assertEqual((report["status"], report["reason"]),
                ("launcher_failed_with_partial_root", "launcher_failure_precedence"))
            self.assertEqual(report["root_identity"],
                             [partial["device"], partial["inode"]])
            self.assertIs(report["eligible_for_writable_reopen"], False)
            self.assertIs(report["can_resume_incomplete"], False)
            with self.assertRaisesRegex(control.ControlError, "retry refused"):
                control.bootstrap_development(
                    raw, pins=pins, entry_wall_origin=1.0, entry_cpu_origin=1.0,
                    entry_process_id=os.getpid(), collect_metadata=lambda _:self.metadata(),
                    create_store=fail_initialization, wall_clock=lambda:2.0,
                    cpu_clock=lambda:2.0, utc_now=lambda:UTC, evidence_kind=KIND)

        with self.outer() as outer:
            attempt = Path(outer) / ATTEMPT
            raw, pins = self.permission(attempt)
            real_fsync = storage.os.fsync

            def fail_after_complete_header():
                calls = 0
                def fsync(fd):
                    nonlocal calls
                    calls += 1
                    if calls == 3:
                        raise OSError("fixture directory fsync")
                    return real_fsync(fd)
                with mock.patch.object(storage.os, "fsync", side_effect=fsync):
                    return storage.ArtifactStore(
                        outer, profile=storage.MLP_FIXTURE, budget_bytes=BUDGET,
                        failure_reserve_bytes=RESERVE,
                        min_filesystem_free_bytes=MIN_FREE)

            with self.assertRaisesRegex(control.ControlError, "root_creation"):
                control.bootstrap_development(
                    raw, pins=pins, entry_wall_origin=1.0, entry_cpu_origin=1.0,
                    entry_process_id=os.getpid(), collect_metadata=lambda _:self.metadata(),
                    create_store=fail_after_complete_header, wall_clock=lambda:2.0,
                    cpu_clock=lambda:2.0, utc_now=lambda:UTC, evidence_kind=KIND)
            failure = codec.json_loads(
                (attempt / control.EXTERNAL_FAILURE_NAME).read_bytes(),
                max_bytes=control.EXTERNAL_FAILURE_MAX)
            partial = failure["partial_root"]
            self.assertEqual(partial["header_status"], "verified_complete")
            journal_raw = (attempt / control.EXTERNAL_JOURNAL_NAME).read_bytes()
            pin = {"path":str(attempt / control.EXTERNAL_JOURNAL_NAME),
                "size_bytes":len(journal_raw),
                "sha256":hashlib.sha256(journal_raw).hexdigest()}
            report = control.inspect_native_attempt(
                partial["path"], pin, expected_store_profile=storage.MLP_FIXTURE,
                expected_evidence_kind=KIND)
            self.assertEqual(report["status"], "launcher_failed_with_partial_root")
            self.assertEqual(report["header_sha256"], partial["header_sha256"])

    def test_caught_interrupt_closes_returned_store_before_reraise(self):
        with self.outer() as outer:
            attempt = Path(outer) / ATTEMPT
            raw, pins = self.permission(attempt)
            holder = {}

            def create():
                store = self.store_factory(outer)()
                holder["store"] = store
                def interrupt(_name, _raw):
                    raise KeyboardInterrupt()
                store.write_bytes = interrupt
                return store

            with self.assertRaises(KeyboardInterrupt):
                control.bootstrap_development(raw, pins=pins, entry_wall_origin=1.0,
                    entry_cpu_origin=1.0, entry_process_id=os.getpid(),
                    collect_metadata=lambda _:self.metadata(),
                    create_store=create, wall_clock=lambda:2.0, cpu_clock=lambda:2.0,
                    utc_now=lambda:UTC, evidence_kind=KIND)
            fd = os.open(holder["store"].root / "store.lock",
                         os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            finally:
                os.close(fd)

    def test_boundary_failure_is_create_only_and_terminal_even_if_root_replaced(self):
        with self.outer() as outer:
            result, _ = self.bootstrap(outer)
            store = result["store"]
            binding, root = result["root_binding"], store.root
            store.close()
            failure_pin = control.record_boundary_failure(
                result["journal_pin"], binding, expected_evidence_kind=KIND)
            raw = Path(failure_pin["path"]).read_bytes()
            self.assertEqual((len(raw), hashlib.sha256(raw).hexdigest()),
                             (failure_pin["size_bytes"], failure_pin["sha256"]))
            failure = codec.json_loads(raw, max_bytes=control.EXTERNAL_FAILURE_MAX)
            self.assertEqual((failure["stage"], failure["reason"]),
                             ("boundary_close", "boundary_close_failed"))
            self.assertEqual(failure["root_binding"], binding)
            self.assertIsNone(failure["partial_root"])
            self.assertIsNone(failure["resource_snapshot"])
            with self.assertRaisesRegex(control.ControlError, "attempt evidence"):
                control.record_boundary_failure(
                    result["journal_pin"], binding, expected_evidence_kind=KIND)
            self.assertEqual(Path(failure_pin["path"]).read_bytes(), raw)
            report = control.inspect_native_attempt(
                root, result["journal_pin"], expected_store_profile=storage.MLP_FIXTURE,
                expected_evidence_kind=KIND)
            self.assertEqual((report["status"], report["reason"]),
                             ("launcher_failed_with_root", "launcher_failure_precedence"))
            self.assertIs(report["eligible_for_writable_reopen"], False)

        with self.outer() as outer:
            result, _ = self.bootstrap(outer)
            store, binding = result["store"], result["root_binding"]
            root = store.root
            store.close()
            moved = root.with_name(root.name + "-moved")
            root.rename(moved)
            root.mkdir()
            control.record_boundary_failure(
                result["journal_pin"], binding, expected_evidence_kind=KIND)
            report = control.inspect_native_attempt(
                root, result["journal_pin"], expected_store_profile=storage.MLP_FIXTURE,
                expected_evidence_kind=KIND)
            self.assertEqual(report["status"], "uninspectable")
            self.assertIs(report["eligible_for_writable_reopen"], False)

    def _resource(self, store):
        view = storage.ArtifactStore.inspect(store.root)
        return {"phase_wall_seconds":0.0, "phase_cpu_seconds":0.0,
            "peak_rss_bytes":1024, "peak_cuda_allocated_bytes":0,
            "peak_cuda_reserved_bytes":0, "root_logical_bytes":view["logical_bytes"],
            "root_allocated_bytes":view["allocated_bytes"]}

    def _reference(self, store, name):
        self.guard()
        marker = {"schema":"i7_native_control_storage_fixture_v1",
            "storage_only":True, "nonsemantic":True, "execution_enabled":False,
            "artifact_name":name}
        receipt = (store.write_bytes(name, codec.json_bytes(marker)) if name.endswith(".json")
                   else store.write_tensor_tree(name, marker))
        raw = (store.root / receipt["receipt_name"]).read_bytes()
        result = {"name":name, "status":"complete", "encoding":receipt["encoding"],
            "size_bytes":receipt["size"], "sha256":receipt["sha256"],
            "receipt_name":receipt["receipt_name"], "receipt_size_bytes":len(raw),
            "receipt_sha256":hashlib.sha256(raw).hexdigest()}
        self.guard()
        return result

    def _append(self, store, records, operation, event, artifacts, sources, environment,
                error=None):
        self.guard()
        sequence = len(records)
        previous = hashlib.sha256(codec.json_bytes(records[-1])).hexdigest()
        record = {"schema":"i7_native_phase_record_v1", "profile":policy.PROFILE,
            "evidence_kind":KIND, "execution_enabled":False, "sequence":sequence,
            "event":event, "operation":operation["operation"], "phase":operation["phase"],
            "created_utc":UTC, "root_binding":{"path":str(store.root),
                "device":store.root_identity[0], "inode":store.root_identity[1],
                "header_sha256":store.header_sha256},
            "sources_sha256":codec.tree_digest(sources),
            "environment_sha256":codec.tree_digest(environment),
            "previous_record_sha256":previous, "artifacts":artifacts,
            "resources":self._resource(store), "error":error}
        policy.validate_record(record, evidence_kind=KIND)
        store.write_bytes(control.phase_name(sequence), codec.json_bytes(record))
        records.append(record)
        self.guard()

    def _extend(self, result, *, stop_at_boundary=False, complete=False):
        store, records = result["store"], [result["go_record"]]
        sources, environment = self.metadata()
        for operation in policy.schedule()[1:]:
            if operation["kind"] == "work":
                self._append(store, records, operation, "started", [], sources, environment)
                refs = [self._reference(store, name) for name in operation["artifacts"]]
                self._append(store, records, operation, "sealed", refs, sources, environment)
            else:
                self._append(store, records, operation, operation["kind"], [], sources, environment)
            if stop_at_boundary and operation["operation"] == "development_complete":
                break
        if complete:
            self.assertEqual(len(records), 90)
        return records

    def test_reader_refuses_live_writer_then_reports_incomplete_nonresumable(self):
        with self.outer() as outer:
            result, _ = self.bootstrap(outer)
            store = result["store"]
            try:
                live = control.inspect_native_attempt(store.root, result["journal_pin"],
                    expected_store_profile=storage.MLP_FIXTURE,
                    expected_evidence_kind=KIND)
                self.assertEqual((live["status"], live["reason"]),
                                 ("uninspectable", "writer_live"))
            finally:
                store.close()
            report = control.inspect_native_attempt(store.root, result["journal_pin"],
                expected_store_profile=storage.MLP_FIXTURE, expected_evidence_kind=KIND)
            self.assertEqual(report["status"], "interrupted_incomplete")
            self.assertEqual(report["phase_records"], 1)
            self.assertIs(report["can_resume_incomplete"], False)
            self.assertIs(report["eligible_for_writable_reopen"], False)
            self.assertTrue(report["metadata_complete"])
            alternate_native = control.inspect_native_attempt(store.root,
                result["journal_pin"], expected_store_profile=storage.SCIENTIFIC,
                expected_evidence_kind="native_producer_attestation")
            self.assertEqual((alternate_native["status"], alternate_native["reason"]),
                             ("uninspectable", "root_or_journal_invalid"))

    def test_store_failure_precedes_missing_policy_failure(self):
        with self.outer() as outer:
            result, _ = self.bootstrap(outer)
            store = result["store"]
            with self.assertRaises(storage.StoreError):
                store.write_bytes(control.phase_name(0), result["go_bytes"])
            store.close()
            report = control.inspect_native_attempt(store.root, result["journal_pin"],
                expected_store_profile=storage.MLP_FIXTURE, expected_evidence_kind=KIND)
            self.assertEqual(report["status"], "store_failed_without_policy_record")
            self.assertEqual(report["store_failure_count"], 1)
            self.assertIs(report["eligible_for_writable_reopen"], False)

    def test_native_attestation_rejects_fixture_store_after_consuming_permission(self):
        with self.outer() as outer:
            attempt = Path(outer) / ATTEMPT
            raw, pins = self.permission(attempt)
            permission = codec.json_loads(
                raw, max_bytes=control.EXTERNAL_JOURNAL_MAX)
            permission["storage_admission_pin"]["path"] = (
                ""
                "output/2026-09-06-spectral-optimizer-investigation/continuation/"
                "iteration-007/native-storage-admission.json")
            raw = codec.json_bytes(permission)
            pins["permission_sha256"] = hashlib.sha256(raw).hexdigest()
            entry_wall, entry_cpu = time.monotonic(), time.process_time()
            with mock.patch.object(control, "DEVELOPMENT_ATTEMPT_PATH", str(attempt)), \
                 mock.patch.object(control, "_torch_is_imported", return_value=False), \
                 mock.patch.object(control, "_prepare_native_runtime"), \
                 self.assertRaisesRegex(control.ControlError, "store profile"):
                    control.bootstrap_development(raw, pins=pins,
                        entry_wall_origin=entry_wall, entry_cpu_origin=entry_cpu,
                        entry_process_id=os.getpid(),
                        collect_metadata=lambda _:self.metadata(),
                        create_store=self.store_factory(outer), utc_now=lambda:UTC,
                        initialize_runtime=lambda _:None)
            failure = codec.json_loads(
                (attempt / control.EXTERNAL_FAILURE_NAME).read_bytes(),
                max_bytes=control.EXTERNAL_FAILURE_MAX)
            self.assertEqual(failure["stage"], "root_creation")
            self.assertIsNotNone(failure["root_binding"])

    def test_pending_operation_outputs_allowed_but_future_outputs_rejected(self):
        with self.outer() as outer:
            result, _ = self.bootstrap(outer)
            store = result["store"]
            sources, environment = self.metadata()
            records = [result["go_record"]]
            pending = policy.schedule()[1]
            self._append(store, records, pending, "started", [], sources, environment)
            self._reference(store, pending["artifacts"][0])
            root, pin = store.root, result["journal_pin"]
            store.close()
            report = control.inspect_native_attempt(root, pin,
                expected_store_profile=storage.MLP_FIXTURE, expected_evidence_kind=KIND)
            self.assertEqual(report["status"], "interrupted_incomplete")
            self.assertIs(report["eligible_for_writable_reopen"], False)

        with self.outer() as outer:
            result, _ = self.bootstrap(outer)
            store = result["store"]
            sources, environment = self.metadata()
            records = [result["go_record"]]
            pending = policy.schedule()[1]
            self._append(store, records, pending, "started", [], sources, environment)
            future = policy.schedule()[2]["artifacts"][0]
            self._reference(store, future)
            root, pin = store.root, result["journal_pin"]
            store.close()
            report = control.inspect_native_attempt(root, pin,
                expected_store_profile=storage.MLP_FIXTURE, expected_evidence_kind=KIND)
            self.assertEqual((report["status"], report["reason"]),
                             ("uninspectable", "unknown_complete_payload"))

    def test_fifo_journal_substitution_is_nonblocking_and_uninspectable(self):
        with self.outer() as outer:
            result, _ = self.bootstrap(outer)
            store, pin = result["store"], result["journal_pin"]
            root = store.root
            store.close()
            journal = Path(pin["path"])
            journal.unlink()
            os.mkfifo(journal, 0o600)
            report = control.inspect_native_attempt(root, pin,
                expected_store_profile=storage.MLP_FIXTURE, expected_evidence_kind=KIND)
            self.assertEqual((report["status"], report["reason"]),
                             ("uninspectable", "root_or_journal_invalid"))

    def test_declared_failure_boundary_and_complete_classification(self):
        with self.outer() as outer:
            result, _ = self.bootstrap(outer)
            store = result["store"]
            sources, environment = self.metadata()
            operation = policy.schedule()[1]
            self._append(store, [result["go_record"]], operation, "failed", [],
                         sources, environment,
                         error={"category":"other", "retained_failure_ref":None})
            root, pin = store.root, result["journal_pin"]
            store.close()
            report = control.inspect_native_attempt(root, pin,
                expected_store_profile=storage.MLP_FIXTURE, expected_evidence_kind=KIND)
            self.assertEqual(report["status"], "declared_failed")
            self.assertIs(report["can_resume_incomplete"], False)

        with self.outer() as outer:
            result, _ = self.bootstrap(outer)
            records = self._extend(result, stop_at_boundary=True)
            root, pin = result["store"].root, result["journal_pin"]
            result["store"].close()
            report = control.inspect_native_attempt(root, pin,
                expected_store_profile=storage.MLP_FIXTURE, expected_evidence_kind=KIND)
            self.assertEqual((records[-1]["event"], report["status"]),
                             ("boundary", "sealed_boundary"))
            self.assertIs(report["eligible_for_writable_reopen"], True)

        with self.outer() as outer:
            result, _ = self.bootstrap(outer)
            self._extend(result, stop_at_boundary=True)
            store, pin = result["store"], result["journal_pin"]
            with self.assertRaises(storage.StoreError):
                store.write_bytes(control.phase_name(0), result["go_bytes"])
            root = store.root
            store.close()
            report = control.inspect_native_attempt(root, pin,
                expected_store_profile=storage.MLP_FIXTURE, expected_evidence_kind=KIND)
            self.assertEqual(report["status"], "store_failed_without_policy_record")
            self.assertIs(report["eligible_for_writable_reopen"], False)

        with self.outer() as outer:
            result, _ = self.bootstrap(outer)
            store = result["store"]
            sources, environment = self.metadata()
            operation = policy.schedule()[1]
            records = [result["go_record"]]
            self._append(store, records, operation, "failed", [], sources, environment,
                         error={"category":"other", "retained_failure_ref":None})
            with self.assertRaises(storage.StoreError):
                store.write_bytes(control.phase_name(0), result["go_bytes"])
            root, pin = store.root, result["journal_pin"]
            store.close()
            report = control.inspect_native_attempt(root, pin,
                expected_store_profile=storage.MLP_FIXTURE, expected_evidence_kind=KIND)
            self.assertEqual(report["status"], "store_terminal_with_policy_failure")
            self.assertIs(report["eligible_for_writable_reopen"], False)

        with self.outer() as outer:
            result, _ = self.bootstrap(outer)
            records = self._extend(result, complete=True)
            root, pin = result["store"].root, result["journal_pin"]
            root_identity = result["store"].root_identity
            header_sha = result["store"].header_sha256
            result["store"].close()
            report = control.inspect_native_attempt(root, pin,
                expected_store_profile=storage.MLP_FIXTURE, expected_evidence_kind=KIND)
            self.assertEqual((len(records), report["status"]), (90, "uninspectable"))
            self.assertEqual(report["reason"], "transition_evidence_invalid")
            self.assertIs(report["eligible_for_writable_reopen"], False)
            self.assertIs(report["actual_artifacts_verified"], False)
            reopened = storage.ArtifactStore.reopen(root,
                expected_profile=storage.MLP_FIXTURE,
                expected_header_sha256=header_sha,
                expected_root_identity=root_identity)
            with self.assertRaises(storage.StoreError):
                reopened.write_bytes(control.phase_name(0), result["go_bytes"])
            reopened.close()
            terminal = control.inspect_native_attempt(root, pin,
                expected_store_profile=storage.MLP_FIXTURE, expected_evidence_kind=KIND)
            self.assertEqual(terminal["status"], "store_terminal_after_complete")
            self.assertIs(terminal["eligible_for_writable_reopen"], False)
            self.assertFalse(torch.cuda.is_initialized())


if __name__ == "__main__":
    unittest.main()
