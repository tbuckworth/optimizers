"""CPU-only accounting and disposable metadata-store fixtures; CUDA stays hidden."""
from __future__ import annotations

import copy
from contextlib import ExitStack
import hashlib
import importlib.util
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


if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
    raise RuntimeError("explicitly hide CUDA for root-storage accounting tests")

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import torch
import artifact_store as storage
import identity_codec as codec
import native_phase_policy as policy
import root_storage_accounting as accounting
import runtime_guard as runtime


BIG_TMP = Path("/tmp/spectral-experiment-artifacts")
FIXTURE_BUDGET = 2 << 20
FIXTURE_RESERVE = 1 << 20
FIXTURE_MIN_FREE = 1 << 30
FIXTURE_SECONDS = 120.0
FIXTURE_RSS = 2 << 30
UTC = "2026-09-07T12:00:00Z"
KIND = "synthetic_contract_fixture"


class RootStorageAccountingTests(unittest.TestCase):
    @staticmethod
    def final_component_sizes():
        path = HERE / "full-storage-final-measurement.json"
        value = json.loads(path.read_bytes())
        measurements = value["measurements"]
        if len(measurements) != 2:
            raise AssertionError("final component measurement membership changed")
        rows = []
        for pattern, measurement in zip(("zero", "ones"), measurements):
            if measurement["pattern"] != pattern:
                raise AssertionError("final component measurement order changed")
            rows.append({key: measurement["components"][key]["serialized_bytes"]
                         for key in accounting.COMPONENT_KEYS})
        if rows[0] != rows[1]:
            raise AssertionError("final zero/ones serialized component sizes differ")
        return rows[0]

    def verified_outer(self):
        runtime.verify_big_volume(BIG_TMP)
        self.assertEqual(BIG_TMP.resolve(), BIG_TMP)
        self.assertGreaterEqual(shutil.disk_usage(BIG_TMP).free, FIXTURE_MIN_FREE)
        return tempfile.TemporaryDirectory(prefix="i7-root-accounting-test-", dir=BIG_TMP)

    def test_import_and_default_cli_are_torch_free_and_inert(self):
        module = HERE / "root_storage_accounting.py"
        with tempfile.TemporaryDirectory() as cwd:
            code = (
                "import importlib.util,sys;"
                "s=importlib.util.spec_from_file_location('isolated_accounting',sys.argv[1]);"
                "m=importlib.util.module_from_spec(s);s.loader.exec_module(m);"
                "print('torch' in sys.modules)"
            )
            result = subprocess.run([sys.executable, "-c", code, str(module)], cwd=cwd,
                                    check=True, capture_output=True, text=True, timeout=10)
            self.assertEqual(result.stdout, "False\n")
            self.assertEqual(os.listdir(cwd), [])
            result = subprocess.run([sys.executable, str(module)], cwd=cwd, check=True,
                                    capture_output=True, text=True, timeout=10)
            self.assertEqual(result.stdout,
                             "root_storage_accounting: inert; no filesystem or execution action\n")
            self.assertEqual(os.listdir(cwd), [])

    def test_fixture_numerical_environment_is_explicitly_single_threaded(self):
        for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS"):
            self.assertEqual(os.environ.get(key), "1")
        self.assertEqual(torch.get_num_threads(), 1)
        self.assertFalse(torch.cuda.is_initialized())

    def test_actual_policy_membership_and_closed_inventory(self):
        payloads = accounting.scheduled_payloads()
        actual = tuple(name for operation in policy.schedule()
                       for name in operation["artifacts"])
        self.assertEqual(tuple(name for name, _ in payloads), actual)
        self.assertEqual(len(payloads), 82)
        self.assertEqual(tuple(sum(key == wanted for _, key in payloads)
                               for wanted in accounting.COMPONENT_KEYS),
                         accounting.COMPONENT_COUNTS)
        files = accounting.expected_regular_files()
        self.assertEqual((len(files), len(set(files))), (346, 346))
        self.assertTrue(accounting.validate_regular_files(list(reversed(files))))
        for bad in (list(files)[:-1], [*files, "unknown.bin"], [*files[:-1], files[0]],
                    set(files), [*files[:-1], False]):
            with self.subTest(kind=type(bad).__name__, length=len(bad)), \
                    self.assertRaises(accounting.AccountingError):
                accounting.validate_regular_files(bad)
        self.assertFalse(torch.cuda.is_initialized())

    def test_policy_kind_change_cannot_preserve_accounting_membership(self):
        original = policy.schedule

        def altered():
            rows = copy.deepcopy(original())
            next(row for row in rows if row["kind"] == "decision")["kind"] = "work"
            return rows

        policy.schedule = altered
        try:
            with self.assertRaisesRegex(accounting.AccountingError, "operation kinds"):
                accounting.scheduled_payloads()
        finally:
            policy.schedule = original

    def test_actual_metadata_encoders_and_exact_type_rejection(self):
        name = "storage-phase-000.json"
        digest = "a" * 64
        expected_receipt = storage._json_bytes({"schema":"i7_artifact_receipt_v1",
            "name":name, "size":123, "sha256":digest, "status":"complete",
            "encoding":"bytes"})
        self.assertEqual(accounting.encode_receipt(name, 123, digest, "bytes"),
                         expected_receipt)
        self.assertEqual(accounting.receipt_name(name),
                         "receipt-" + hashlib.sha256(name.encode("ascii")).hexdigest() + ".json")
        expected_header = storage._json_bytes({"schema":"i7_artifact_store_v1",
            "profile":storage.SCIENTIFIC, "budget_bytes":storage.DEFAULT_BUDGET,
            "failure_reserve_bytes":storage.DEFAULT_FAILURE_RESERVE,
            "min_filesystem_free_bytes":storage.DEFAULT_MIN_FREE,
            "cap_semantics":"sum_regular_file_logical_bytes"})
        self.assertEqual(accounting.encode_store_header(), expected_header)
        for args in ((name, True, digest, "bytes"), (name, 1, "A" * 64, "bytes"),
                     (name, 1, digest, "unknown"), ("../bad", 1, digest, "bytes")):
            with self.subTest(args=args), self.assertRaises(accounting.AccountingError):
                accounting.encode_receipt(*args)
        with self.assertRaises(accounting.AccountingError):
            accounting.encode_store_header(budget_bytes=True)

    def test_projection_uses_final_committed_components_and_labels_limits(self):
        sizes = self.final_component_sizes()
        report = accounting.project(sizes)
        logical = report["logical_bytes"]
        self.assertEqual(logical["scheduled_payload_bodies"], 945341202)
        self.assertEqual(logical["projected_regular_files"], 957176530)
        self.assertEqual(logical["projected_regular_files_plus_reserve"], 958225106)
        self.assertEqual(logical["normal_write_headroom"], 115516718)
        self.assertEqual(report["membership"]["regular_files"], 346)
        self.assertIs(report["normal_write_ceiling_fit"], True)
        self.assertEqual(report["native_payload_size_delta"], "unresolved")
        self.assertEqual(report["native_adapter_additional_files"], "unresolved")
        self.assertEqual(report["phase_filename_status"],
                         "fixture_only_not_scheduled_by_native_policy")
        self.assertIs(report["execution_enabled"], False)
        self.assertIs(report["scientific_execution_certified"], False)
        for mutation in (lambda row: row.pop("anchor_pilot"),
                         lambda row: row.__setitem__("unknown", 1),
                         lambda row: row.__setitem__("anchor_pilot", True),
                         lambda row: row.__setitem__("anchor_pilot", 1.0),
                         lambda row: row.__setitem__("anchor_pilot", 0)):
            altered = dict(sizes)
            mutation(altered)
            with self.assertRaises(accounting.AccountingError):
                accounting.project(altered)

    def _guard(self, started):
        self.assertLessEqual(time.monotonic() - started, FIXTURE_SECONDS)
        self.assertLessEqual(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                             FIXTURE_RSS)
        self.assertFalse(torch.cuda.is_initialized())

    @staticmethod
    def _root_binding(store):
        device, inode = store.root_identity
        return {"path":str(store.root), "device":device, "inode":inode,
                "header_sha256":store.header_sha256}

    def _resources(self, store, *, wall_seconds=0.0, cpu_seconds=0.0):
        report = storage.ArtifactStore.inspect(store.root)
        return ({"phase_wall_seconds":float(wall_seconds),
                 "phase_cpu_seconds":float(cpu_seconds),
                 "peak_rss_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                 "peak_cuda_allocated_bytes":0, "peak_cuda_reserved_bytes":0,
                 "root_logical_bytes":report["logical_bytes"],
                 "root_allocated_bytes":report["allocated_bytes"]}, report)

    def _write_payload(self, store, name):
        marker = {"schema":"i7_storage_only_fixture_payload_v1",
                  "storage_only":True, "nonsemantic":True,
                  "execution_enabled":False, "evidence_kind":KIND,
                  "artifact_name":name}
        if name.endswith(".json"):
            receipt = store.write_bytes(name, codec.json_bytes(marker))
        else:
            receipt = store.write_tensor_tree(name, marker)
        payload_raw = (store.root / name).read_bytes()
        self.assertEqual(len(payload_raw), receipt["size"])
        self.assertEqual(hashlib.sha256(payload_raw).hexdigest(), receipt["sha256"])
        if name.endswith(".json"):
            self.assertEqual(codec.json_loads(payload_raw, max_bytes=65536), marker)
        else:
            self.assertEqual(storage.ArtifactStore.load_tensor_tree(
                store.root, name, expected_size=receipt["size"],
                expected_sha256=receipt["sha256"]), marker)
        receipt_raw = (store.root / receipt["receipt_name"]).read_bytes()
        self.assertEqual(receipt_raw, accounting.encode_receipt(
            name, receipt["size"], receipt["sha256"], receipt["encoding"]))
        decoded = codec.json_loads(receipt_raw, max_bytes=storage._RECEIPT_MAX)
        self.assertEqual(decoded, {"encoding":receipt["encoding"], "name":name,
            "schema":"i7_artifact_receipt_v1", "sha256":receipt["sha256"],
            "size":receipt["size"], "status":"complete"})
        return {"name":name, "status":"complete", "encoding":receipt["encoding"],
                "size_bytes":receipt["size"], "sha256":receipt["sha256"],
                "receipt_name":receipt["receipt_name"],
                "receipt_size_bytes":len(receipt_raw),
                "receipt_sha256":hashlib.sha256(receipt_raw).hexdigest()}

    def test_tiny_actual_writer_transcript_inventory_and_pinned_reopen(self):
        started = time.monotonic()
        torch_rng = torch.random.get_rng_state().clone()
        with self.verified_outer() as outer, ExitStack() as fixture_stack:
            outer_path = Path(outer)
            store = fixture_stack.enter_context(storage.ArtifactStore(
                outer, profile=storage.MLP_FIXTURE, budget_bytes=FIXTURE_BUDGET,
                failure_reserve_bytes=FIXTURE_RESERVE,
                min_filesystem_free_bytes=FIXTURE_MIN_FREE))
            root, root_identity, header_sha = store.root, store.root_identity, store.header_sha256
            records, declared_prewrite, previous = [], [], None
            phase_wall_started, phase_cpu_started = {}, {}

            def write_record(operation, event, references):
                nonlocal previous
                sequence = len(records)
                first = operation["operation"] == "development_go"
                phase = operation["phase"]
                phase_wall_started.setdefault(phase, time.monotonic())
                phase_cpu_started.setdefault(phase, time.process_time())
                resources, before = (None, storage.ArtifactStore.inspect(root)) if first \
                    else self._resources(
                        store,
                        wall_seconds=time.monotonic() - phase_wall_started[phase],
                        cpu_seconds=time.process_time() - phase_cpu_started[phase])
                binding = None if first else self._root_binding(store)
                record = {"schema":"i7_native_phase_record_v1",
                    "profile":policy.PROFILE, "evidence_kind":KIND,
                    "execution_enabled":False, "sequence":sequence, "event":event,
                    "operation":operation["operation"], "phase":operation["phase"],
                    "created_utc":UTC, "root_binding":binding,
                    "sources_sha256":"a" * 64, "environment_sha256":"b" * 64,
                    "previous_record_sha256":previous, "artifacts":references,
                    "resources":resources, "error":None}
                raw = codec.json_bytes(record)
                policy.validate_record(record, evidence_kind=KIND)
                store.write_bytes(accounting.PHASE_PAYLOADS[sequence], raw)
                records.append(record)
                declared_prewrite.append(before)
                previous = hashlib.sha256(raw).hexdigest()
                self._guard(started)

            for operation in policy.schedule():
                if operation["kind"] == "work":
                    write_record(operation, "started", [])
                    references = [self._write_payload(store, name)
                                  for name in operation["artifacts"]]
                    write_record(operation, "sealed", references)
                else:
                    write_record(operation, operation["kind"], [])

            self.assertEqual(len(records), 90)
            self.assertEqual(records[0]["resources"], None)
            for record, before in zip(records[1:], declared_prewrite[1:]):
                self.assertEqual(record["resources"]["root_logical_bytes"],
                                 before["logical_bytes"])
                self.assertEqual(record["resources"]["root_allocated_bytes"],
                                 before["allocated_bytes"])
            report = storage.ArtifactStore.inspect(root)
            self.assertEqual((len(report["files"]), len(report["receipts"]), report["failures"]),
                             (346, 172, []))
            self.assertTrue(accounting.validate_regular_files(list(report["files"])))
            self.assertEqual(report["logical_bytes"],
                             sum(row["size"] for row in report["files"].values()))
            self.assertEqual(report["allocated_bytes"],
                             sum(row["allocated_bytes"] for row in report["files"].values()))
            self.assertLessEqual(report["logical_bytes"], FIXTURE_BUDGET - FIXTURE_RESERVE)
            self.assertGreater(report["logical_bytes"],
                               records[-1]["resources"]["root_logical_bytes"])
            self.assertGreaterEqual(report["allocated_bytes"],
                                    records[-1]["resources"]["root_allocated_bytes"])
            self.assertEqual((root / "store-header.json").read_bytes(),
                accounting.encode_store_header(profile=storage.MLP_FIXTURE,
                    budget_bytes=FIXTURE_BUDGET,
                    failure_reserve_bytes=FIXTURE_RESERVE,
                    min_filesystem_free_bytes=FIXTURE_MIN_FREE))
            for receipt in report["receipts"]:
                raw = (root / accounting.receipt_name(receipt["name"])).read_bytes()
                self.assertEqual(raw, accounting.encode_receipt(
                    receipt["name"], receipt["size"], receipt["sha256"], receipt["encoding"]))
            loaded = [codec.json_loads((root / name).read_bytes(),
                                       max_bytes=policy.MAX_RECORD_BYTES)
                      for name in accounting.PHASE_PAYLOADS]
            self.assertEqual(loaded, records)
            validated = policy.validate_transcript(loaded, evidence_kind=KIND)
            self.assertEqual((validated["status"], validated["records"],
                              validated["artifact_count"]),
                             ("declared_complete", 90, 82))
            self.assertTrue(torch.equal(torch_rng, torch.random.get_rng_state()))
            self._guard(started)
            store.close()
            reopened = fixture_stack.enter_context(storage.ArtifactStore.reopen(
                root, expected_profile=storage.MLP_FIXTURE,
                expected_header_sha256=header_sha,
                expected_root_identity=root_identity))
            try:
                reopened_report = storage.ArtifactStore.inspect(root)
                self.assertEqual(reopened_report["logical_bytes"], report["logical_bytes"])
                self.assertEqual(reopened_report["allocated_bytes"], report["allocated_bytes"])
            finally:
                reopened.close()
            with self.assertRaises(accounting.AccountingError):
                accounting.validate_regular_files([*report["files"], "extra.bin"])
            self._guard(started)
            fixture_metrics = {"schema":"i7_root_storage_fixture_metrics_v1",
                "fixture":"closed_success", "execution_enabled":False,
                "regular_files":len(report["files"]), "receipts":len(report["receipts"]),
                "logical_bytes":report["logical_bytes"],
                "allocated_bytes":report["allocated_bytes"],
                "final_minus_last_prewrite_logical_bytes":report["logical_bytes"]
                    - records[-1]["resources"]["root_logical_bytes"],
                "final_minus_last_prewrite_allocated_bytes":report["allocated_bytes"]
                    - records[-1]["resources"]["root_allocated_bytes"],
                "temporary_roots_cleaned":True}
        self.assertFalse(outer_path.exists())
        print(json.dumps(fixture_metrics, sort_keys=True, separators=(",", ":")))

    def test_duplicate_write_is_terminal_and_chained_failure_cannot_append(self):
        started = time.monotonic()
        cpu_started = time.process_time()
        with self.verified_outer() as outer, ExitStack() as fixture_stack:
            outer_path = Path(outer)
            store = fixture_stack.enter_context(storage.ArtifactStore(
                outer, profile=storage.MLP_FIXTURE, budget_bytes=FIXTURE_BUDGET,
                failure_reserve_bytes=FIXTURE_RESERVE,
                min_filesystem_free_bytes=FIXTURE_MIN_FREE))
            root, root_identity, header_sha = store.root, store.root_identity, store.header_sha256
            first = {"schema":"i7_native_phase_record_v1", "profile":policy.PROFILE,
                "evidence_kind":KIND, "execution_enabled":False, "sequence":0,
                "event":"decision", "operation":"development_go", "phase":"development",
                "created_utc":UTC, "root_binding":None, "sources_sha256":"a" * 64,
                "environment_sha256":"b" * 64, "previous_record_sha256":None,
                "artifacts":[], "resources":None, "error":None}
            first_raw = codec.json_bytes(first)
            store.write_bytes(accounting.PHASE_PAYLOADS[0], first_raw)
            with self.assertRaisesRegex(storage.StoreError, "immutable target already exists"):
                store.write_bytes(accounting.PHASE_PAYLOADS[0], first_raw)
            report = storage.ArtifactStore.inspect(root)
            self.assertTrue(report["terminal"])
            self.assertEqual(len(report["failures"]), 1)
            self.assertEqual(report["failures"][0], {
                "schema":"i7_store_failure_v1", "code":"write_failed",
                "detail":"immutable target already exists", "files":[], "terminal":True})
            self.assertEqual((root / accounting.PHASE_PAYLOADS[0]).read_bytes(), first_raw)
            failure_name = next(name for name in report["files"] if name.startswith("failure-"))
            failure_raw = (root / failure_name).read_bytes()
            resources, _ = self._resources(
                store, wall_seconds=time.monotonic() - started,
                cpu_seconds=time.process_time() - cpu_started)
            failed = {"schema":"i7_native_phase_record_v1", "profile":policy.PROFILE,
                "evidence_kind":KIND, "execution_enabled":False, "sequence":1,
                "event":"failed", "operation":"development.plan", "phase":"development",
                "created_utc":UTC, "root_binding":self._root_binding(store),
                "sources_sha256":"a" * 64, "environment_sha256":"b" * 64,
                "previous_record_sha256":hashlib.sha256(first_raw).hexdigest(),
                "artifacts":[], "resources":resources,
                "error":{"category":"io", "retained_failure_ref":{"name":failure_name,
                    "size_bytes":len(failure_raw),
                    "sha256":hashlib.sha256(failure_raw).hexdigest()}}}
            self.assertEqual(policy.validate_transcript(
                [first, failed], evidence_kind=KIND, require_complete=False)["status"],
                "declared_failed")
            failed_raw = codec.json_bytes(failed)
            for name, body in (("post-terminal.json", b"x"),
                               (accounting.PHASE_PAYLOADS[1], failed_raw)):
                with self.subTest(name=name), self.assertRaisesRegex(storage.StoreError, "terminal"):
                    store.write_bytes(name, body)
                self.assertFalse((root / name).exists())
            store.close()
            with self.assertRaisesRegex(storage.StoreError, "reopen failed|terminal"):
                storage.ArtifactStore.reopen(
                    root, expected_profile=storage.MLP_FIXTURE,
                    expected_header_sha256=header_sha,
                    expected_root_identity=root_identity)
            final = storage.ArtifactStore.inspect(root)
            self.assertEqual(final["logical_bytes"], report["logical_bytes"])
            self.assertEqual(final["allocated_bytes"], report["allocated_bytes"])
            self._guard(started)
            fixture_metrics = {"schema":"i7_root_storage_fixture_metrics_v1",
                "fixture":"duplicate_terminal", "execution_enabled":False,
                "regular_files":len(final["files"]), "failures":len(final["failures"]),
                "logical_bytes":final["logical_bytes"],
                "allocated_bytes":final["allocated_bytes"],
                "post_terminal_write_refused":True,
                "chained_failure_append_refused":True,
                "writable_reopen_refused":True, "temporary_roots_cleaned":True}
        self.assertFalse(outer_path.exists())
        print(json.dumps(fixture_metrics, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    unittest.main()
