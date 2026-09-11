"""Tiny CPU/filesystem fixtures only; no scientific data or GPU work."""
import fcntl
import importlib.util
import os
from pathlib import Path
import random
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

spec = importlib.util.spec_from_file_location("i7_store", Path(__file__).with_name("artifact_store.py"))
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.outer = tempfile.TemporaryDirectory()
        self.addCleanup(self.outer.cleanup)

    def make_store(self, budget=65536):
        store = s.ArtifactStore(self.outer.name, profile=s.FIXTURE, budget_bytes=budget,
                                failure_reserve_bytes=16384, min_filesystem_free_bytes=0)
        self.addCleanup(store.close)
        return store

    def test_success_receipt_inventory_and_lock(self):
        store = self.make_store()
        receipt = store.write_bytes("raw.bin", b"abc")
        self.assertEqual((receipt["size"], receipt["sha256"], receipt["status"]),
                         (3, s._sha(b"abc"), "complete"))
        other = os.open(store.root / "store.lock", os.O_RDWR | os.O_NOFOLLOW)
        self.addCleanup(os.close, other)
        with self.assertRaises(BlockingIOError):
            fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
        view = s.ArtifactStore.inspect(store.root)
        self.assertFalse(view["terminal"])
        self.assertEqual(view["logical_bytes"], sum(x["size"] for x in view["files"].values()))
        self.assertEqual(view["allocated_bytes"], sum(x["allocated_bytes"] for x in view["files"].values()))

    def test_reserve_prevents_overshoot_and_terminal_blocks_retry(self):
        store = self.make_store()
        with self.assertRaisesRegex(s.StoreError, "budget"):
            store.write_bytes("large.bin", b"x" * store.budget)
        self.assertFalse((store.root / "large.bin").exists())
        with self.assertRaisesRegex(s.StoreError, "terminal"):
            store.write_bytes("retry.bin", b"x")
        view = s.ArtifactStore.inspect(store.root)
        self.assertTrue(view["terminal"])
        self.assertLessEqual(view["logical_bytes"], store.budget)

        full = self.make_store()
        used = sum(entry.stat().st_size for entry in os.scandir(full.root))
        (full.root / "external-full.bin").write_bytes(b"z" * (full.budget - used))
        with self.assertRaisesRegex(s.StoreError, "failure_metadata=not_retained"):
            full.write_bytes("blocked.bin", b"x")
        actual = sum(entry.stat().st_size for entry in os.scandir(full.root))
        self.assertEqual(actual, full.budget)
        self.assertTrue(full.failure_metadata_status.startswith("not_retained:"))

    def test_partial_io_is_retained_and_indexed(self):
        store = self.make_store()
        original = store._write_all

        def fail_once(fd, payload):
            store._write_all = original
            os.write(fd, payload[:3])
            raise OSError("fixture partial write")

        store._write_all = fail_once
        with self.assertRaises(s.StoreError):
            store.write_bytes("partial.bin", b"abcdef")
        self.assertEqual((store.root / "partial.bin").read_bytes(), b"abc")
        view = s.ArtifactStore.inspect(store.root)
        rows = [row for failure in view["failures"] for row in failure["files"]]
        self.assertEqual([(x["name"], x["size"]) for x in rows], [("partial.bin", 3)])

        receipt_store = self.make_store()
        original = receipt_store._write_all
        calls = 0

        def fail_receipt(fd, payload):
            nonlocal calls
            calls += 1
            if calls == 2:
                receipt_store._write_all = original
                os.write(fd, payload[:3])
                raise OSError("fixture partial receipt")
            original(fd, payload)

        receipt_store._write_all = fail_receipt
        with self.assertRaises(s.StoreError):
            receipt_store.write_bytes("orphan.bin", b"complete")
        self.assertTrue(s.ArtifactStore.inspect(receipt_store.root)["terminal"])

    def test_overwrite_and_raced_target_are_unchanged(self):
        store = self.make_store()
        store.write_bytes("fixed.bin", b"first")
        with self.assertRaises(s.StoreError):
            store.write_bytes("fixed.bin", b"second")
        self.assertEqual((store.root / "fixed.bin").read_bytes(), b"first")

        raced = self.make_store()
        (raced.root / "race.bin").write_bytes(b"outsider")
        with self.assertRaises(s.StoreError):
            raced.write_bytes("race.bin", b"inside")
        self.assertEqual((raced.root / "race.bin").read_bytes(), b"outsider")

        injected = self.make_store()
        original = injected._precheck

        def race_after_precheck(*args, **kwargs):
            original(*args, **kwargs)
            (injected.root / "late.bin").write_bytes(b"late outsider")
            injected._precheck = original

        injected._precheck = race_after_precheck
        with self.assertRaises(s.StoreError):
            injected.write_bytes("late.bin", b"writer")
        self.assertEqual((injected.root / "late.bin").read_bytes(), b"late outsider")

        postcheck = self.make_store()
        original = postcheck._precheck
        calls = 0

        def fail_postcheck(prospective, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise s.StoreError("fixture post-write free-space change")
            return original(prospective, **kwargs)

        postcheck._precheck = fail_postcheck
        with self.assertRaises(s.StoreError):
            postcheck.write_bytes("committed.bin", b"retained")
        self.assertEqual((postcheck.root / "committed.bin").read_bytes(), b"retained")
        self.assertTrue(s.ArtifactStore.inspect(postcheck.root)["terminal"])

    def test_path_symlink_nonregular_and_hardlink_rejected(self):
        path_store = self.make_store()
        with self.assertRaises(s.StoreError):
            path_store.write_bytes("../escape", b"x")
        self.assertFalse(Path(self.outer.name, "escape").exists())

        symlink_store = self.make_store()
        os.symlink("missing", symlink_store.root / "bad-link")
        with self.assertRaisesRegex(s.StoreError, "symlink"):
            symlink_store.write_bytes("safe.bin", b"x")

        directory_store = self.make_store()
        os.mkdir(directory_store.root / "bad-dir")
        with self.assertRaisesRegex(s.StoreError, "nonregular"):
            directory_store.write_bytes("safe.bin", b"x")

        hard_store = self.make_store()
        external = Path(self.outer.name, "external")
        external.write_bytes(b"x")
        os.link(external, hard_store.root / "bad-hardlink")
        with self.assertRaisesRegex(s.StoreError, "hardlinked"):
            hard_store.write_bytes("safe.bin", b"x")

    def test_tensor_tree_roundtrip_and_mismatch_precedes_decode(self):
        store = self.make_store()
        tree = {"x": torch.arange(4, dtype=torch.float32), "meta":[1, "ok", None]}
        receipt = store.write_tensor_tree("tree.pt", tree)
        loaded = s.ArtifactStore.load_tensor_tree(store.root, "tree.pt",
                                                   expected_size=receipt["size"],
                                                   expected_sha256=receipt["sha256"])
        self.assertTrue(torch.equal(loaded["x"], tree["x"]))
        with self.assertRaisesRegex(s.StoreError, "exact integer"):
            s.ArtifactStore.load_tensor_tree(store.root, "tree.pt", expected_size=True,
                                             expected_sha256=receipt["sha256"])
        with self.assertRaisesRegex(s.StoreError, "SHA-256"):
            s.ArtifactStore.load_tensor_tree(store.root, "tree.pt", expected_size=receipt["size"],
                                             expected_sha256="bad")
        with self.assertRaisesRegex(s.StoreError, "budget ceiling"):
            s.ArtifactStore.load_tensor_tree(store.root, "tree.pt", expected_size=store.budget + 1,
                                             expected_sha256=receipt["sha256"])
        with open(store.root / "tree.pt", "r+b") as handle:
            handle.write(b"X")
        with self.assertRaisesRegex(s.StoreError, "hash mismatch"):
            s.ArtifactStore.load_tensor_tree(store.root, "tree.pt",
                                             expected_size=receipt["size"],
                                             expected_sha256=receipt["sha256"])

        second = self.make_store()
        receipt2 = second.write_tensor_tree("size.pt", {"x":torch.tensor([1.])})
        with open(second.root / "size.pt", "ab") as handle:
            handle.write(b"X")
        with self.assertRaisesRegex(s.StoreError, "size"):
            s.ArtifactStore.load_tensor_tree(second.root, "size.pt",
                                             expected_size=receipt2["size"],
                                             expected_sha256=receipt2["sha256"])

        growing = self.make_store()
        receipt3 = growing.write_tensor_tree("grow.pt", {"x":torch.arange(8.)})
        target_inode = os.stat(growing.root / "grow.pt").st_ino
        real_read, real_load, changed, decoded = s.os.read, s.torch.load, False, False

        def append_during_read(fd, count):
            nonlocal changed
            data = real_read(fd, count)
            if not changed and os.fstat(fd).st_ino == target_inode:
                changed = True
                with open(growing.root / "grow.pt", "ab") as handle:
                    handle.write(b"X")
            return data

        def forbidden_decode(*args, **kwargs):
            nonlocal decoded
            decoded = True
            raise AssertionError("decoder called")

        s.os.read, s.torch.load = append_during_read, forbidden_decode
        try:
            with self.assertRaisesRegex(s.StoreError, "changed during"):
                s.ArtifactStore.load_tensor_tree(growing.root, "grow.pt",
                                                 expected_size=receipt3["size"],
                                                 expected_sha256=receipt3["sha256"])
        finally:
            s.os.read, s.torch.load = real_read, real_load
        self.assertFalse(decoded)

        linked = self.make_store()
        receipt4 = linked.write_tensor_tree("linked.pt", {"x":torch.arange(8.)})
        target = linked.root / "linked.pt"
        target_inode = os.stat(target).st_ino
        external_link = Path(self.outer.name, "late-hardlink")
        real_read, real_load, changed, decoded = s.os.read, s.torch.load, False, False

        def link_during_read(fd, count):
            nonlocal changed
            data = real_read(fd, count)
            if not changed and os.fstat(fd).st_ino == target_inode:
                changed = True
                os.link(target, external_link)
            return data

        s.os.read, s.torch.load = link_during_read, forbidden_decode
        try:
            with self.assertRaisesRegex(s.StoreError, "changed during"):
                s.ArtifactStore.load_tensor_tree(linked.root, "linked.pt",
                                                 expected_size=receipt4["size"],
                                                 expected_sha256=receipt4["sha256"])
        finally:
            s.os.read, s.torch.load = real_read, real_load
        self.assertFalse(decoded)

    def test_constructor_failure_closes_descriptors_and_preserves_root(self):
        free = os.statvfs(self.outer.name).f_bavail * os.statvfs(self.outer.name).f_frsize
        before = len(os.listdir("/proc/self/fd"))
        with self.assertRaises(s.StoreInitializationError) as caught:
            s.ArtifactStore(self.outer.name, profile=s.FIXTURE, budget_bytes=65536,
                            failure_reserve_bytes=16384,
                            min_filesystem_free_bytes=free + 1048576)
        self.assertEqual(len(os.listdir("/proc/self/fd")), before)
        roots = list(Path(self.outer.name).glob("i7-artifacts-*"))
        self.assertEqual(len(roots), 1)
        identity = os.stat(roots[0])
        self.assertEqual(caught.exception.partial_root,
            {"path":str(roots[0]), "device":identity.st_dev, "inode":identity.st_ino,
             "header_status":"absent", "header_sha256":None})
        copied = caught.exception.partial_root
        copied["inode"] += 1
        self.assertEqual(caught.exception.partial_root["inode"], identity.st_ino)
        self.assertTrue((roots[0] / "store.lock").exists())
        fd = os.open(roots[0] / "store.lock", os.O_RDWR | os.O_NOFOLLOW)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            os.close(fd)

    def test_constructor_partial_header_states_and_interrupt_cleanup(self):
        original_write = s.ArtifactStore._write_all
        calls = 0

        def partial_header(store, fd, payload):
            nonlocal calls
            calls += 1
            if calls == 2:
                os.write(fd, payload[:3])
                raise OSError("fixture partial header")
            return original_write(store, fd, payload)

        with mock.patch.object(s.ArtifactStore, "_write_all", partial_header), \
                self.assertRaises(s.StoreInitializationError) as caught:
            s.ArtifactStore(self.outer.name, profile=s.FIXTURE, budget_bytes=65536,
                            failure_reserve_bytes=16384, min_filesystem_free_bytes=0)
        self.assertEqual(caught.exception.partial_root["header_status"],
                         "present_unverified")
        self.assertIsNone(caught.exception.partial_root["header_sha256"])

        real_fsync = s.os.fsync
        fsync_calls = 0
        def fail_directory_fsync(fd):
            nonlocal fsync_calls
            fsync_calls += 1
            if fsync_calls == 3:
                raise OSError("fixture post-header fsync")
            return real_fsync(fd)
        with mock.patch.object(s.os, "fsync", side_effect=fail_directory_fsync), \
                self.assertRaises(s.StoreInitializationError) as complete:
            s.ArtifactStore(self.outer.name, profile=s.FIXTURE, budget_bytes=65536,
                            failure_reserve_bytes=16384, min_filesystem_free_bytes=0)
        self.assertEqual(complete.exception.partial_root["header_status"],
                         "verified_complete")
        self.assertRegex(complete.exception.partial_root["header_sha256"],
                         r"\A[0-9a-f]{64}\Z")

        real_stat = s.os.stat
        def header_stat_failure(path, *args, **kwargs):
            if path == "store-header.json":
                raise PermissionError("fixture header observation")
            return real_stat(path, *args, **kwargs)
        free = os.statvfs(self.outer.name).f_bavail * os.statvfs(self.outer.name).f_frsize
        with mock.patch.object(s.os, "stat", side_effect=header_stat_failure), \
                self.assertRaises(s.StoreInitializationError) as unobserved:
            s.ArtifactStore(self.outer.name, profile=s.FIXTURE, budget_bytes=65536,
                            failure_reserve_bytes=16384,
                            min_filesystem_free_bytes=free + 1048576)
        self.assertEqual(unobserved.exception.partial_root["header_status"], "unobserved")
        self.assertIsNotNone(unobserved.exception.partial_root["inode"])

        before = len(os.listdir("/proc/self/fd"))
        with mock.patch.object(s.ArtifactStore, "_precheck",
                               side_effect=KeyboardInterrupt), \
                self.assertRaises(KeyboardInterrupt):
            s.ArtifactStore(self.outer.name, profile=s.FIXTURE, budget_bytes=65536,
                            failure_reserve_bytes=16384, min_filesystem_free_bytes=0)
        self.assertEqual(len(os.listdir("/proc/self/fd")), before)
        newest = max(Path(self.outer.name).glob("i7-artifacts-*"),
                     key=lambda path:path.stat().st_ctime_ns)
        lockfd = os.open(newest / "store.lock", os.O_RDONLY | os.O_NOFOLLOW)
        try:
            fcntl.flock(lockfd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            os.close(lockfd)

    def test_inspection_rejects_malformed_header_limits(self):
        store = self.make_store()
        header_path = store.root / "store-header.json"
        import json
        header = json.loads(header_path.read_bytes())
        header["budget_bytes"] = True
        header_path.write_bytes(s._json_bytes(header))
        with self.assertRaisesRegex(s.StoreError, "limits"):
            s.ArtifactStore.inspect(store.root)

    def test_bounded_serialization_and_invalid_tree_are_terminal(self):
        view_store = self.make_store()
        with self.assertRaisesRegex(s.StoreError, "compact"):
            view_store.write_tensor_tree("view.pt", {"x":torch.arange(4.).view(2, 2)})
        self.assertTrue(s.ArtifactStore.inspect(view_store.root)["terminal"])
        store = self.make_store()
        with self.assertRaisesRegex(s.StoreError, "bounded serialization"):
            store.write_tensor_tree("large.pt", {"x":torch.zeros(20000)})
        self.assertFalse((store.root / "large.pt").exists())
        self.assertTrue(s.ArtifactStore.inspect(store.root)["terminal"])

        shared_store = self.make_store()
        tensor = torch.ones(2)
        with self.assertRaisesRegex(s.StoreError, "shared"):
            shared_store.write_tensor_tree("shared.pt", {"a":tensor, "b":tensor})
        self.assertTrue(s.ArtifactStore.inspect(shared_store.root)["terminal"])

    def test_fixture_limits_only_and_rng_cuda_inert(self):
        with self.assertRaisesRegex(s.StoreError, "fixture-profile"):
            s.ArtifactStore(self.outer.name, profile=s.SCIENTIFIC, budget_bytes=65536,
                            failure_reserve_bytes=16384, min_filesystem_free_bytes=0)
        rng = torch.random.get_rng_state().clone()
        python_rng = random.getstate()
        numpy_rng = np.random.get_state()
        cuda_before = torch.cuda.is_initialized()
        store = self.make_store()
        store.write_tensor_tree("inert.pt", {"x":torch.tensor([1., 2.])})
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        self.assertEqual(python_rng, random.getstate())
        numpy_after = np.random.get_state()
        self.assertEqual(numpy_rng[0], numpy_after[0])
        self.assertTrue(np.array_equal(numpy_rng[1], numpy_after[1]))
        self.assertEqual(numpy_rng[2:], numpy_after[2:])
        self.assertEqual(cuda_before, torch.cuda.is_initialized())


if __name__ == "__main__":
    unittest.main()
