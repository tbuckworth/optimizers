"""Tiny CPU tests for held-plan loading and pinned same-root store reopen."""
from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
    raise RuntimeError("explicitly hide CUDA for store-I/O fixtures")

import artifact_store as storage
import identity_codec as codec
from test_plan_bindings import fixture
import verified_plan_load as verified


class PhaseStoreIOTests(unittest.TestCase):
    def setUp(self):
        self.outer = tempfile.TemporaryDirectory(prefix="i7-phase-store-io-")
        self.addCleanup(self.outer.cleanup)

    def new_store(self, **overrides):
        options = {
            "profile": storage.MLP_FIXTURE,
            "budget_bytes": 64 << 10,
            "failure_reserve_bytes": 16 << 10,
            "min_filesystem_free_bytes": 0,
        }
        options.update(overrides)
        store = storage.ArtifactStore(self.outer.name, **options)
        self.addCleanup(store.close)
        return store

    @staticmethod
    def pins(store):
        return store.root_identity, store.header_sha256

    def reopen(self, root, root_identity, header_sha256, **overrides):
        options = {
            "expected_profile": storage.MLP_FIXTURE,
            "expected_header_sha256": header_sha256,
            "expected_root_identity": root_identity,
        }
        options.update(overrides)
        reopened = storage.ArtifactStore.reopen(root, **options)
        self.addCleanup(reopened.close)
        return reopened

    def test_held_store_plan_load_reuses_exclusive_lock_descriptor(self):
        identity, plan, _ = fixture(storage.MLP_FIXTURE)
        store = self.new_store()
        receipt = store.write_tensor_tree("plan.pt", plan)

        with mock.patch.object(verified.fcntl, "flock",
                               side_effect=AssertionError("must not open or relock store.lock")) as flock:
            loaded = verified.load_verified_plan_from_store(
                store, "plan.pt", identity=identity, profile=storage.MLP_FIXTURE,
                expected_sha256=receipt["sha256"])

        flock.assert_not_called()
        self.assertEqual(tuple(loaded),
                         ("store", "artifact", "plan", "binding", "plan_content_sha256"))
        self.assertEqual(loaded["artifact"]["sha256"], receipt["sha256"])
        self.assertEqual(loaded["plan_content_sha256"], codec.tree_digest(plan))
        self.assertFalse(store._closed)

    def test_competing_reopen_is_rejected_and_original_writer_remains_usable(self):
        store = self.new_store()
        root_identity, header_sha256 = self.pins(store)
        with self.assertRaises(storage.StoreError):
            self.reopen(store.root, root_identity, header_sha256)
        receipt = store.write_bytes("still-owned.bin", b"owner retained lock")
        self.assertEqual(receipt["status"], "complete")

    def test_close_reopen_preserves_inventory_accounting_limits_and_writes(self):
        store = self.new_store()
        first = store.write_bytes("first.bin", b"a" * 4096)
        root, (root_identity, header_sha256) = store.root, self.pins(store)
        before = storage.ArtifactStore.inspect(root)
        store.close()

        reopened = self.reopen(root, root_identity, header_sha256)
        self.assertEqual(reopened.root_identity, root_identity)
        self.assertEqual(reopened.header_sha256, header_sha256)
        self.assertEqual(reopened.budget, 64 << 10)
        self.assertEqual(reopened.failure_reserve, 16 << 10)
        self.assertEqual(reopened._indexed[first["name"]]["sha256"], first["sha256"])
        self.assertEqual(reopened._inventory(reopened._indexed)[0], before["logical_bytes"])
        reopened.write_bytes("second.bin", b"b" * 4096)
        self.assertGreater(storage.ArtifactStore.inspect(root)["logical_bytes"],
                           before["logical_bytes"])
        with self.assertRaises(storage.StoreError):
            reopened.write_bytes("over-cap.bin", b"x" * (40 << 10))

    def test_terminal_store_refuses_reopen(self):
        store = self.new_store()
        root_identity, header_sha256 = self.pins(store)
        with self.assertRaises(storage.StoreError):
            store.write_bytes("invalid.bin", bytearray(b"not exact bytes"))
        root = store.root
        store.close()
        with self.assertRaises(storage.StoreError):
            self.reopen(root, root_identity, header_sha256)

    def test_bad_root_header_and_profile_pins_are_rejected(self):
        store = self.new_store()
        root, (root_identity, header_sha256) = store.root, self.pins(store)
        store.close()
        cases = (
            {"expected_root_identity": (root_identity[0], root_identity[1] + 1)},
            {"expected_header_sha256": "0" * 64},
            {"expected_profile": storage.FIXTURE},
        )
        for overrides in cases:
            with self.subTest(overrides=overrides), self.assertRaises(storage.StoreError):
                self.reopen(root, root_identity, header_sha256, **overrides)

    def test_replaced_root_and_symlinked_ancestor_are_rejected(self):
        store = self.new_store()
        root, (root_identity, header_sha256) = store.root, self.pins(store)
        store.close()
        moved = root.with_name(root.name + "-moved")
        os.rename(root, moved)
        root.mkdir()
        with self.assertRaises(storage.StoreError):
            self.reopen(root, root_identity, header_sha256)

        parent_link = Path(self.outer.name) / "parent-link"
        parent_link.symlink_to(Path(self.outer.name), target_is_directory=True)
        aliased = parent_link / moved.name
        with self.assertRaises(storage.StoreError):
            self.reopen(aliased, root_identity, header_sha256)

    def test_indexed_file_replacement_during_reopen_is_rejected(self):
        store = self.new_store()
        store.write_bytes("payload.bin", b"payload")
        root, (root_identity, header_sha256) = store.root, self.pins(store)
        store.close()
        original = storage._inspect_dirfd

        def replace_after_inspection(dirfd):
            report, scanned = original(dirfd)
            replacement = root / "replacement.tmp"
            replacement.write_bytes((root / "payload.bin").read_bytes())
            os.replace(replacement, root / "payload.bin")
            return report, scanned

        with mock.patch.object(storage, "_inspect_dirfd",
                               side_effect=replace_after_inspection) as inspection:
            with self.assertRaises(storage.StoreError):
                self.reopen(root, root_identity, header_sha256)
        inspection.assert_called_once()

    def test_unindexed_file_and_hardlink_refuse_reopen(self):
        for corruption in ("unknown", "hardlink"):
            store = self.new_store()
            receipt = store.write_bytes("payload.bin", b"payload")
            root, (root_identity, header_sha256) = store.root, self.pins(store)
            store.close()
            if corruption == "unknown":
                (root / "unknown.bin").write_bytes(b"not indexed")
            else:
                os.link(root / receipt["name"], root / "payload-hardlink.bin")
            with self.subTest(corruption=corruption), self.assertRaises(storage.StoreError):
                self.reopen(root, root_identity, header_sha256)


if __name__ == "__main__":
    unittest.main()
