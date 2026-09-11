"""Tiny CPU/API fixtures, not native artifacts, measurements or admission evidence.

Synthetic metadata exercises shape/cap validation only. Temporary stores hold
at most tiny test bytes; no data, plan, scientific producer or CUDA init is used.
"""
import copy
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
    raise RuntimeError("native store contract tests require hidden CUDA")

import torch
import artifact_store as storage
import identity_codec as codec
import native_payload_guard as payload_guard
import native_write_ledger as ledger
import pickle_storage_bound as pickle_bound
import zip_storage_bound as zip_bound
from test_native_layout_inspection import _success_report


class NativeStoreEnforcementTests(unittest.TestCase):
    def setUp(self):
        self.assertFalse(torch.cuda.is_initialized())
        self.assertEqual(os.environ.get("TMPDIR"), "/tmp/spectral-experiment-artifacts")
        self.outer = tempfile.TemporaryDirectory(prefix="i7-write-ledger-unit-",
                                                  dir=os.environ["TMPDIR"])
        self.addCleanup(self.outer.cleanup)
        synthetic = _success_report()
        self.sources = synthetic["source_binding"]
        self.environment = synthetic["native_environment_binding"]

    def tearDown(self):
        self.assertFalse(torch.cuda.is_initialized())

    def store(self):
        store = storage.ArtifactStore(self.outer.name, profile=storage.SCIENTIFIC,
            runtime_sources=self.sources, runtime_environment=self.environment)
        self.addCleanup(store.close)
        return store

    def test_scientific_missing_metadata_rejected_before_root_creation(self):
        with mock.patch.object(storage.tempfile, "mkdtemp") as create:
            with self.assertRaisesRegex(storage.StoreError, "runtime metadata"):
                storage.ArtifactStore(self.outer.name)
            create.assert_not_called()

    def test_fixture_context_is_rejected_and_ordinary_fixture_remains_unchanged(self):
        with self.assertRaisesRegex(storage.StoreError, "scientific-profile only"):
            storage.ArtifactStore(self.outer.name, profile=storage.FIXTURE,
                                 runtime_sources=self.sources)
        with storage.ArtifactStore(self.outer.name, profile=storage.FIXTURE,
                budget_bytes=65536, failure_reserve_bytes=16384,
                min_filesystem_free_bytes=0) as store:
            self.assertEqual(store.write_bytes("arbitrary.bin", b"x")["size"], 1)

    def test_context_is_private_canonical_copy_and_header_uses_normal_ledger(self):
        with mock.patch.object(ledger, "initialization_reservation",
                               wraps=ledger.initialization_reservation) as reservation:
            store = self.store()
        self.assertEqual([call.args for call in reservation.call_args_list],
                         [("store.lock", 0, 0), ("store-header.json", 222, 0)])
        original = codec.json_bytes(store._runtime_sources)
        self.sources["files"][0]["size_bytes"] = 99
        self.assertEqual(codec.json_bytes(store._runtime_sources), original)
        with mock.patch.object(ledger, "NORMAL_ROOT_CEILING_BYTES", 222):
            with self.assertRaisesRegex(storage.StoreError, "budget"):
                store._precheck(1, reserve_failure=False)

    def test_unknown_name_and_wrong_encoding_stop_before_payload_creation(self):
        store = self.store()
        with self.assertRaises(storage.StoreError):
            store.write_bytes("unknown.bin", b"x")
        self.assertFalse((store.root / "unknown.bin").exists())
        self.assertEqual(store.failure_metadata_status, "retained")
        other = self.store()
        with self.assertRaises(storage.StoreError):
            other.write_bytes("i7-native-b71990-plan.pt", b"x")
        self.assertFalse((other.root / "i7-native-b71990-plan.pt").exists())

    def test_metadata_bytes_must_match_private_binding(self):
        store = self.store()
        changed = copy.deepcopy(self.environment)
        changed["operating_system"]["release"] = "changed-fixture"
        with self.assertRaisesRegex(storage.StoreError, "differs from writer binding"):
            store.write_bytes("native-development-environment.json", codec.json_bytes(changed))
        self.assertFalse((store.root / "native-development-environment.json").exists())

    def test_pair_reserves_full_receipt_ceiling_before_direct_final_writes(self):
        store = self.store()
        name = "native-phase-000.json"
        allowance = ledger.normal_write_budget(name, "bytes", 222)
        with mock.patch.object(store, "_precheck", wraps=store._precheck) as check:
            receipt = store.write_bytes(name, b"{}\n")
        self.assertEqual(check.call_args_list[0].args,
                         (3 + allowance["receipt_bytes_upper"],))
        self.assertEqual(receipt["size"], 3)
        self.assertEqual(len(list(store.root.iterdir())), 4)

    def test_partial_body_and_repeated_failures_remain_bounded_and_retained(self):
        store = self.store()
        original = store._write_all
        def partial(fd, raw):
            store._write_all = original
            os.write(fd, raw[:1])
            raise OSError("unit partial")
        store._write_all = partial
        with self.assertRaises(storage.StoreError):
            store.write_bytes("native-phase-000.json", b"{}\n")
        self.assertEqual((store.root / "native-phase-000.json").read_bytes(), b"{")
        self.assertEqual(store._fail("unit_second", "repeat"), "retained")
        self.assertTrue((store.root / "failure-000002.json").is_file())
        logical = sum(item.stat().st_size for item in store.root.iterdir())
        with mock.patch.object(ledger, "FAILURE_ROOT_CEILING_BYTES", logical):
            self.assertTrue(store._fail("unit_third", "no room").startswith("not_retained:"))
        self.assertFalse((store.root / "failure-000003.json").exists())

    def test_comparison_requires_canonical_decode_before_guard_or_write(self):
        store = self.store()
        name = "i7-native-pilot-b71990-capture-comparison.json"
        with mock.patch.object(payload_guard, "admit_json_payload") as guard:
            with self.assertRaises(storage.StoreError):
                store.write_bytes(name, b'{"x":1,"x":2}\n')
            guard.assert_not_called()
        self.assertFalse((store.root / name).exists())
        other = self.store()
        # Mock only the semantic/component domain: this tests central dispatch,
        # not a valid comparison, measured size or scientific pass.
        with mock.patch.object(payload_guard, "admit_json_payload") as guard:
            other.write_bytes(name, b"{}\n")
            self.assertEqual(guard.call_args.args, (name, {}))
            self.assertIs(guard.call_args.kwargs["sources"], other._runtime_sources)

    def test_aggregate_failure_pool_counts_partial_files_without_large_allocation(self):
        # Stat-only fixtures, not a megabyte of disk output. 63 full records and
        # one half record exactly occupy the root share of the failure reserve.
        scanned = {f"failure-{index:06d}.json": SimpleNamespace(st_size=16384)
                   for index in range(1, 64)}
        scanned["failure-000064.json"] = SimpleNamespace(st_size=8191)
        storage._native_failure_reserve(scanned, 1)
        with self.assertRaisesRegex(storage.StoreError, "aggregate root failure"):
            storage._native_failure_reserve(scanned, 2)
        storage._native_failure_reserve({"failure-000001.json": SimpleNamespace(st_size=0)}, 1)
        for name, size in (("failure-000000.json", 0), ("failure-bad.json", 0),
                           ("failure-000001.json", 16385), ("failure-000001.json", -1)):
            with self.subTest(name=name, size=size), self.assertRaises(storage.StoreError):
                storage._native_failure_reserve({name: SimpleNamespace(st_size=size)}, 1)
        store = self.store()
        with mock.patch.object(storage, "_native_failure_reserve",
                side_effect=storage.StoreError("unit aggregate cap")) as reserve, \
             mock.patch.object(store, "_create_file") as create:
            self.assertTrue(store._fail("unit", "reserve rejected").startswith("not_retained:"))
            reserve.assert_called_once()
            create.assert_not_called()

    def test_actual_tree_rejection_occurs_before_serializer(self):
        store = self.store()
        with mock.patch.object(torch, "save") as save:
            with self.assertRaises(storage.StoreError):
                store.write_tensor_tree("i7-native-b71990-plan.pt", {})
            save.assert_not_called()
        self.assertFalse((store.root / "i7-native-b71990-plan.pt").exists())

    def test_scientific_path_preserves_finite_tensor_contract(self):
        store = self.store()
        # Four bytes, not a scientific tensor layout. Isolate the preserved
        # numeric guard from the separate component topology validator.
        tree = {"unit": torch.tensor([float("nan")], dtype=torch.float32)}
        with mock.patch.object(payload_guard, "admit_torch_payload"), \
             mock.patch.object(torch, "save") as save:
            with self.assertRaisesRegex(storage.StoreError, "nonfinite tensor"):
                store.write_tensor_tree("i7-native-b71990-plan.pt", tree)
            save.assert_not_called()

    def test_serializer_uses_explicit_pins_after_guard_and_bounded_buffer(self):
        store = self.store()
        events = []
        def guard(*args, **kwargs):
            events.append("tree")
            return {"zip_bytes_upper": 8}
        def save(tree, buffer, **kwargs):
            events.append("save")
            self.assertEqual(kwargs, zip_bound.required_torch_save_kwargs())
            self.assertEqual(buffer.limit, 8)
            buffer.write(b"unit")
        with mock.patch.object(payload_guard, "admit_torch_payload",
                side_effect=guard), \
             mock.patch.object(pickle_bound, "assert_pinned_pickle_runtime",
                side_effect=lambda: events.append("pickle")), \
             mock.patch.object(zip_bound, "validate_runtime_save_configuration",
                side_effect=lambda **kw: events.append(("zip", kw))), \
             mock.patch.object(torch, "save", side_effect=save):
            store.write_tensor_tree("i7-native-b71990-plan.pt", {})
        self.assertEqual(events, ["tree", "pickle",
            ("zip", {"require_cuda_uninitialized": False}), "save"])

    def test_serializer_overrun_leaves_no_body_or_receipt(self):
        store = self.store()
        allowance = ledger.normal_write_budget("i7-native-b71990-plan.pt", "torch_weights_only", 222)
        allowance["max_payload_bytes_now"] = 3
        with mock.patch.object(ledger, "normal_write_budget", return_value=allowance), \
             mock.patch.object(payload_guard, "admit_torch_payload", return_value={"zip_bytes_upper": 8}), \
             mock.patch.object(pickle_bound, "assert_pinned_pickle_runtime"), \
             mock.patch.object(zip_bound, "validate_runtime_save_configuration"), \
             mock.patch.object(torch, "save", side_effect=lambda t, b, **kw: b.write(b"four")):
            with self.assertRaisesRegex(storage.StoreError, "bounded serialization limit"):
                store.write_tensor_tree("i7-native-b71990-plan.pt", {})
        self.assertFalse((store.root / "i7-native-b71990-plan.pt").exists())
        self.assertFalse((store.root / allowance["receipt_name"]).exists())

    def test_per_tree_zip_prediction_also_caps_serialization(self):
        store = self.store()
        with mock.patch.object(payload_guard, "admit_torch_payload", return_value={"zip_bytes_upper": 3}), \
             mock.patch.object(pickle_bound, "assert_pinned_pickle_runtime"), \
             mock.patch.object(zip_bound, "validate_runtime_save_configuration"), \
             mock.patch.object(torch, "save", side_effect=lambda t, b, **kw: b.write(b"four")):
            with self.assertRaisesRegex(storage.StoreError, "bounded serialization limit"):
                store.write_tensor_tree("i7-native-b71990-plan.pt", {})
        self.assertFalse((store.root / "i7-native-b71990-plan.pt").exists())

    def test_reopen_restores_context_and_rejects_nonledger_retained_name(self):
        store = self.store()
        store.write_bytes("native-phase-000.json", b"{}\n")
        pins = dict(expected_profile=storage.SCIENTIFIC,
            expected_header_sha256=store.header_sha256, expected_root_identity=store.root_identity,
            runtime_sources=self.sources, runtime_environment=self.environment)
        store.close()
        reopened = storage.ArtifactStore.reopen(store.root, **pins)
        self.addCleanup(reopened.close)
        self.assertEqual(reopened._runtime_sources, self.sources)
        # An intentional unit-only bypass builds a correctly receipted unknown
        # file; production SCIENTIFIC writes cannot take this fixture branch.
        with mock.patch.object(reopened, "profile", storage.FIXTURE):
            reopened.write_bytes("rogue.bin", b"x")
        reopened.close()
        with self.assertRaises(storage.StoreError):
            storage.ArtifactStore.reopen(store.root, **pins)


if __name__ == "__main__":
    unittest.main()
