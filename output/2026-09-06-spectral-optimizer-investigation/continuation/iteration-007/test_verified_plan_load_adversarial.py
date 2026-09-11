"""Adversarial tiny-MLP tests for descriptor-held verified plan loading.

These fixtures exercise local corruption and cooperative-writer races only. They
do not certify the scientific profile, hostile pickle safety, or authentication.
"""
from __future__ import annotations

import copy
import fcntl
import hashlib
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
    raise RuntimeError("explicitly hide CUDA for verified-load fixtures")

import torch

import artifact_store as storage
import identity_codec as codec
from test_plan_bindings import fixture
import verified_plan_load as subject


class VerifiedPlanLoadAdversarialTests(unittest.TestCase):
    NAME = "fixture-plan.pt"

    def setUp(self):
        self.outer = tempfile.TemporaryDirectory(prefix="i7-verified-load-")
        self.addCleanup(self.outer.cleanup)
        self.identity, self.plan, _ = fixture(storage.MLP_FIXTURE)

    def _new_store(self, *, close=True):
        store = storage.ArtifactStore(
            self.outer.name,
            profile=storage.MLP_FIXTURE,
            budget_bytes=64 << 10,
            failure_reserve_bytes=16 << 10,
            min_filesystem_free_bytes=0,
        )
        try:
            receipt = store.write_tensor_tree(self.NAME, self.plan)
        except BaseException:
            store.close()
            raise
        if close:
            store.close()
        else:
            self.addCleanup(store.close)
        return store, receipt

    def _load(self, root, expected_sha256):
        return subject.load_verified_plan(
            str(root), self.NAME, identity=self.identity,
            profile=storage.MLP_FIXTURE, expected_sha256=expected_sha256,
        )

    def _assert_rejected_before_decode(self, root, expected_sha256):
        with mock.patch.object(subject.torch, "load",
                               side_effect=AssertionError("decoder must not run")) as decoder:
            with self.assertRaises(subject.VerifiedPlanLoadError):
                self._load(root, expected_sha256)
        decoder.assert_not_called()

    @staticmethod
    def _receipt_path(root):
        suffix = hashlib.sha256(VerifiedPlanLoadAdversarialTests.NAME.encode("ascii")).hexdigest()
        return Path(root) / f"receipt-{suffix}.json"

    def test_baseline_valid_load_reaches_decode_and_binds_exact_bytes(self):
        store, receipt = self._new_store()
        original = subject.torch.load
        with mock.patch.object(subject.torch, "load", wraps=original) as decoder:
            loaded = self._load(store.root, receipt["sha256"])
        decoder.assert_called_once()
        self.assertEqual(loaded["artifact"]["sha256"], receipt["sha256"])
        self.assertEqual(loaded["artifact"]["size_bytes"], receipt["size"])
        self.assertEqual(loaded["plan_content_sha256"], codec.tree_digest(loaded["plan"]))
        self.assertEqual(loaded["binding"]["artifact"], {
            "sha256": receipt["sha256"], "size_bytes": receipt["size"],
        })

    def test_duplicate_metadata_key_is_rejected_before_decode(self):
        store, receipt = self._new_store()
        receipt_path = self._receipt_path(store.root)
        duplicate = (
            b'{"encoding":"torch_weights_only","encoding":"torch_weights_only",'
            b'"name":"fixture-plan.pt","schema":"i7_artifact_receipt_v1",'
            b'"sha256":"' + receipt["sha256"].encode("ascii") +
            b'","size":' + str(receipt["size"]).encode("ascii") +
            b',"status":"complete"}\n'
        )
        receipt_path.write_bytes(duplicate)

        self._assert_rejected_before_decode(store.root, receipt["sha256"])

    def test_root_and_inventory_symlinks_are_rejected(self):
        store, receipt = self._new_store()
        root_link = Path(self.outer.name) / "store-root-link"
        root_link.symlink_to(store.root, target_is_directory=True)
        with self.subTest(case="root"):
            self._assert_rejected_before_decode(root_link, receipt["sha256"])

        (store.root / "unindexed-link").symlink_to(self.NAME)
        with self.subTest(case="inventory"):
            self._assert_rejected_before_decode(store.root, receipt["sha256"])

    def test_symlinked_ancestor_component_is_rejected(self):
        store, receipt = self._new_store()
        parent_link = Path(self.outer.name) / "parent-link"
        parent_link.symlink_to(Path(self.outer.name), target_is_directory=True)
        aliased_root = parent_link / Path(store.root).name

        self._assert_rejected_before_decode(aliased_root, receipt["sha256"])

    def test_hardlinked_payload_is_rejected(self):
        store, receipt = self._new_store()
        os.link(store.root / self.NAME, store.root / "hardlink-plan.pt")
        self.assertEqual((store.root / self.NAME).stat().st_nlink, 2)

        self._assert_rejected_before_decode(store.root, receipt["sha256"])

    def test_unindexed_regular_file_is_rejected(self):
        store, receipt = self._new_store()
        (store.root / "unindexed.bin").write_bytes(b"not in a receipt")

        self._assert_rejected_before_decode(store.root, receipt["sha256"])

    def test_open_writer_lock_is_rejected_without_waiting(self):
        store, receipt = self._new_store(close=False)
        probe = os.open(store.root / "store.lock", os.O_RDONLY | os.O_NOFOLLOW)
        self.addCleanup(os.close, probe)
        with self.assertRaises(BlockingIOError):
            fcntl.flock(probe, fcntl.LOCK_SH | fcntl.LOCK_NB)

        self._assert_rejected_before_decode(store.root, receipt["sha256"])

    def test_late_inventory_addition_is_rejected(self):
        store, receipt = self._new_store()
        original = subject.plans.validate_plan_binding

        def add_after_binding(*args, **kwargs):
            result = original(*args, **kwargs)
            (store.root / "late-unindexed.bin").write_bytes(b"race")
            return result

        with mock.patch.object(subject.plans, "validate_plan_binding",
                               side_effect=add_after_binding) as callback:
            with self.assertRaises(subject.VerifiedPlanLoadError):
                self._load(store.root, receipt["sha256"])
        callback.assert_called_once()
        self.assertEqual((store.root / "late-unindexed.bin").read_bytes(), b"race")

    def test_target_replacement_between_inspection_and_read_is_rejected(self):
        store, receipt = self._new_store()
        original = subject.storage._inspect_dirfd
        original_inode = (store.root / self.NAME).stat().st_ino

        def replace_after_inspection(dirfd):
            report, scanned = original(dirfd)
            replacement = store.root / "replacement.tmp"
            replacement.write_bytes((store.root / self.NAME).read_bytes())
            os.replace(replacement, store.root / self.NAME)
            return report, scanned

        with mock.patch.object(subject.storage, "_inspect_dirfd",
                               side_effect=replace_after_inspection) as callback:
            self._assert_rejected_before_decode(store.root, receipt["sha256"])
        callback.assert_called_once()
        self.assertNotEqual((store.root / self.NAME).stat().st_ino, original_inode)

    def test_root_path_replacement_during_load_is_rejected(self):
        store, receipt = self._new_store()
        root = Path(store.root)
        moved = root.with_name(root.name + "-original")
        replacement = root.with_name(root.name + "-replacement")
        replacement.mkdir()
        original = subject.plans.validate_plan_binding

        def replace_root_after_binding(*args, **kwargs):
            result = original(*args, **kwargs)
            os.rename(root, moved)
            os.rename(replacement, root)
            return result

        with mock.patch.object(subject.plans, "validate_plan_binding",
                               side_effect=replace_root_after_binding) as callback:
            with self.assertRaises(subject.VerifiedPlanLoadError):
                self._load(root, receipt["sha256"])
        callback.assert_called_once()
        self.assertTrue(moved.is_dir())
        self.assertTrue(root.is_dir())

    def test_self_consistent_plan_and_receipt_rewrite_fails_external_digest(self):
        store, original_receipt = self._new_store()
        replacement_plan = copy.deepcopy(self.plan)
        replacement_plan["initialization_seed"] -= 1
        payload = io.BytesIO()
        torch.save(replacement_plan, payload)
        replacement_bytes = payload.getvalue()
        replacement_sha = hashlib.sha256(replacement_bytes).hexdigest()
        self.assertNotEqual(replacement_sha, original_receipt["sha256"])

        (store.root / self.NAME).write_bytes(replacement_bytes)
        replacement_receipt = {
            "encoding": "torch_weights_only",
            "name": self.NAME,
            "schema": "i7_artifact_receipt_v1",
            "sha256": replacement_sha,
            "size": len(replacement_bytes),
            "status": "complete",
        }
        self._receipt_path(store.root).write_bytes(codec.json_bytes(replacement_receipt))
        inspected = storage.ArtifactStore.inspect(store.root)
        self.assertEqual(inspected["receipts"], [replacement_receipt])

        self._assert_rejected_before_decode(store.root, original_receipt["sha256"])

    def test_adversarial_failures_do_not_initialize_cuda(self):
        store, receipt = self._new_store()
        (store.root / "unindexed.bin").write_bytes(b"x")
        self._assert_rejected_before_decode(store.root, receipt["sha256"])
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == "__main__":
    unittest.main()
