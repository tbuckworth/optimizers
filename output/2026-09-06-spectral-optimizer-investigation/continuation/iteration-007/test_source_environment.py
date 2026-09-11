"""Dataset-free CPU and temporary-Git tests for source_environment."""
from __future__ import annotations

import copy
import hashlib
import os
from pathlib import Path
import random
import subprocess
import tempfile
import time
import unittest
from unittest import mock

if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
    raise RuntimeError("source/environment tests require CUDA_VISIBLE_DEVICES=''")

import numpy as np
import torch

import artifact_store as storage
import source_environment as subject
from test_state_core import fixture as state_fixture


def _git(root, *arguments):
    return subprocess.run(["git", *arguments], cwd=root, check=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout


class TemporaryGitMixin:
    def setUp(self):
        self.outer = tempfile.TemporaryDirectory(prefix="i7-source-environment-")
        self.addCleanup(self.outer.cleanup)

    def repository(self):
        root = Path(tempfile.mkdtemp(prefix="repo-", dir=self.outer.name))
        _git(root, "init", "-q")
        (root / "fixture_source.py").write_bytes(b"VALUE = 17\n")
        (root / "fixture_contract.md").write_bytes(b"# Fixed fixture contract\n")
        _git(root, "add", "fixture_source.py", "fixture_contract.md")
        _git(root, "-c", "user.name=I7 Fixture", "-c", "user.email=i7@example.invalid",
             "commit", "-q", "-m", "fixture")
        return root


class SourceCollectionTests(TemporaryGitMixin, unittest.TestCase):
    def test_clean_repository_binds_exact_committed_bytes(self):
        root = self.repository()
        result = subject.collect_verified_sources(root, profile=storage.MLP_FIXTURE)
        self.assertIs(subject.validate_sources(result, profile=storage.MLP_FIXTURE), result)
        self.assertEqual(result["repository_root_realpath"], str(root))
        self.assertEqual([(row["role"], row["path"]) for row in result["files"]],
                         list(subject.FIXTURE_ROLES))
        for row in result["files"]:
            data = (root / row["path"]).read_bytes()
            self.assertEqual(row["size_bytes"], len(data))
            self.assertEqual(row["sha256"], hashlib.sha256(data).hexdigest())
            self.assertEqual(_git(root, "cat-file", "blob", row["git_blob_oid"]), data)

    def test_untracked_and_ordinary_dirty_worktrees_are_rejected(self):
        root = self.repository()
        (root / "untracked.txt").write_text("not allowed")
        with self.assertRaises(subject.SourceEnvironmentError):
            subject.collect_verified_sources(root, profile=storage.FIXTURE)

        other = self.repository()
        (other / "fixture_contract.md").write_text("changed")
        with self.assertRaises(subject.SourceEnvironmentError):
            subject.collect_verified_sources(other, profile=storage.FIXTURE)

    def test_clean_status_cannot_hide_changed_source_bytes(self):
        root = self.repository()
        _git(root, "update-index", "--assume-unchanged", "fixture_source.py")
        (root / "fixture_source.py").write_bytes(b"VALUE = 18\n")
        self.assertEqual(_git(root, "status", "--porcelain=v1", "-z", "--untracked-files=all"), b"")
        with self.assertRaisesRegex(subject.SourceEnvironmentError, "committed blob"):
            subject.collect_verified_sources(root, profile=storage.MLP_FIXTURE)

    def test_symlinked_source_and_root_ancestor_are_rejected(self):
        root = self.repository()
        _git(root, "update-index", "--assume-unchanged", "fixture_source.py")
        target = Path(self.outer.name) / "outside.py"
        target.write_bytes(b"VALUE = 17\n")
        (root / "fixture_source.py").unlink()
        (root / "fixture_source.py").symlink_to(target)
        with self.assertRaises(subject.SourceEnvironmentError):
            subject.collect_verified_sources(root, profile=storage.FIXTURE)

        clean = self.repository()
        parent_link = Path(self.outer.name) / "parent-link"
        parent_link.symlink_to(clean.parent, target_is_directory=True)
        with self.assertRaises(subject.SourceEnvironmentError):
            subject.collect_verified_sources(parent_link / clean.name,
                                              profile=storage.FIXTURE)

    def test_hardlinked_source_is_rejected_even_while_status_is_clean(self):
        root = self.repository()
        outside = Path(self.outer.name) / "outside-hardlink.py"
        os.link(root / "fixture_source.py", outside)
        self.assertEqual(_git(root, "status", "--porcelain=v1", "-z", "--untracked-files=all"), b"")
        self.assertEqual((root / "fixture_source.py").stat().st_nlink, 2)
        with self.assertRaises(subject.SourceEnvironmentError):
            subject.collect_verified_sources(root, profile=storage.MLP_FIXTURE)

    def test_fifo_source_is_opened_nonblocking_then_rejected(self):
        root = self.repository()
        _git(root, "update-index", "--assume-unchanged", "fixture_source.py")
        (root / "fixture_source.py").unlink()
        os.mkfifo(root / "fixture_source.py")
        real_open = subject.os.open

        def checked_open(path, flags, *args, **kwargs):
            if path == "fixture_source.py":
                self.assertTrue(flags & os.O_NONBLOCK)
            return real_open(path, flags, *args, **kwargs)

        started = time.monotonic()
        with mock.patch.object(subject.os, "open", side_effect=checked_open) as opened:
            with self.assertRaises(subject.SourceEnvironmentError):
                subject.collect_verified_sources(root, profile=storage.FIXTURE)
        self.assertTrue(any(call.args[0] == "fixture_source.py" for call in opened.call_args_list))
        self.assertLess(time.monotonic() - started, subject.COMMAND_TIMEOUT)

    def test_unrelated_ancestor_sibling_creation_does_not_change_identity(self):
        root = self.repository()
        original = subject._check_root_snapshot
        sibling = Path(self.outer.name) / "unrelated-sibling"

        def create_sibling(rows):
            sibling.mkdir()
            return original(rows)

        with mock.patch.object(subject, "_check_root_snapshot", side_effect=create_sibling) as callback:
            result = subject.collect_verified_sources(root, profile=storage.FIXTURE)
        callback.assert_called_once()
        self.assertTrue(sibling.is_dir())
        self.assertIs(subject.validate_sources(result, profile=storage.FIXTURE), result)

    def test_nested_read_signature_binds_intermediate_directories(self):
        root = Path(tempfile.mkdtemp(prefix="nested-", dir=self.outer.name))
        leaf = root / "one" / "two" / "source.py"
        leaf.parent.mkdir(parents=True)
        leaf.write_bytes(b"VALUE = 1\n")
        rootfd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        self.addCleanup(os.close, rootfd)
        data_before, signature_before = subject._read_source(rootfd, "one/two/source.py")

        moved = root / "old-one"
        os.rename(root / "one", moved)
        (root / "one" / "two").mkdir(parents=True)
        os.rename(moved / "two" / "source.py", leaf)
        data_after, signature_after = subject._read_source(rootfd, "one/two/source.py")

        self.assertEqual(data_before, data_after)
        # Moving the same inode can advance ctime (and coarse clock resolution
        # can hide that advance). It is not a portable invariant of this fixture.
        # Compare the unchanged fields, then isolate the directory contribution
        # even if the leaf's ctime is deliberately ignored in this assertion.
        self.assertEqual(signature_before[1][:-1], signature_after[1][:-1],
                         "fixture should preserve leaf identity/content metadata except ctime")
        self.assertNotEqual(signature_before[0], signature_after[0])
        self.assertNotEqual((signature_before[0], signature_before[1][:-1]),
                            (signature_after[0], signature_after[1][:-1]))
        self.assertNotEqual(signature_before, signature_after)

    def test_file_and_root_races_execute_and_are_rejected(self):
        root = self.repository()
        original_read = subject._read_source
        calls = 0

        def race_file(rootfd, path):
            nonlocal calls
            result = original_read(rootfd, path)
            calls += 1
            if calls == len(subject.FIXTURE_ROLES):
                (root / "fixture_source.py").write_bytes(b"VALUE = 19\n")
            return result

        with mock.patch.object(subject, "_read_source", side_effect=race_file) as callback:
            with self.assertRaises(subject.SourceEnvironmentError):
                subject.collect_verified_sources(root, profile=storage.FIXTURE)
        self.assertEqual(callback.call_count, len(subject.FIXTURE_ROLES))

        clean = self.repository()
        moved = clean.with_name(clean.name + "-moved")
        replacement = clean.with_name(clean.name + "-replacement")
        replacement.mkdir()
        original_check = subject._check_root_snapshot

        def race_root(rows):
            os.rename(clean, moved)
            os.rename(replacement, clean)
            return original_check(rows)

        with mock.patch.object(subject, "_check_root_snapshot", side_effect=race_root) as callback:
            with self.assertRaises(subject.SourceEnvironmentError):
                subject.collect_verified_sources(clean, profile=storage.FIXTURE)
        callback.assert_called_once()
        self.assertTrue(clean.is_dir())
        self.assertTrue(moved.is_dir())

    def test_source_schema_is_exact_and_profile_specific(self):
        value = subject.collect_verified_sources(self.repository(), profile=storage.FIXTURE)
        class StringSubclass(str):
            pass

        cases = []
        wrong_order = dict(reversed(list(value.items())))
        cases.append(wrong_order)
        extra = copy.deepcopy(value); extra["extra"] = None; cases.append(extra)
        wrong_role = copy.deepcopy(value); wrong_role["files"][0]["role"] = "contract"; cases.append(wrong_role)
        bool_size = copy.deepcopy(value); bool_size["files"][0]["size_bytes"] = True; cases.append(bool_size)
        huge = copy.deepcopy(value); huge["files"][0]["size_bytes"] = subject.SOURCE_FILE_CAP + 1; cases.append(huge)
        for field in ("schema", "profile", "worktree_status"):
            changed = copy.deepcopy(value); changed[field] = StringSubclass(changed[field]); cases.append(changed)
        for field in ("role", "git_mode"):
            changed = copy.deepcopy(value); changed["files"][0][field] = StringSubclass(
                changed["files"][0][field]); cases.append(changed)
        for index, case in enumerate(cases):
            with self.subTest(index=index), self.assertRaises(subject.SourceEnvironmentError):
                subject.validate_sources(case, profile=storage.FIXTURE)
        with self.assertRaises(subject.SourceEnvironmentError):
            subject.validate_sources(value, profile=storage.SCIENTIFIC)
        bad_root = copy.deepcopy(value)
        bad_root["repository_root_realpath"] = "//" + bad_root["repository_root_realpath"].lstrip("/")
        with self.assertRaises(subject.SourceEnvironmentError):
            subject.validate_sources(bad_root, profile=storage.FIXTURE)
        with self.assertRaises(subject.SourceEnvironmentError):
            subject.collect_verified_sources("//" + str(self.repository()).lstrip("/"),
                                              profile=storage.FIXTURE)


class RuntimeEnvironmentTests(TemporaryGitMixin, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if torch.cuda.is_initialized():
            raise RuntimeError("CPU environment tests started with initialized CUDA")

    def collect(self, profile=storage.MLP_FIXTURE):
        return subject.collect_runtime_environment(
            self.repository(), profile=profile, runtime_role="fixture_cpu")

    def test_cpu_capture_is_rng_neutral_and_calls_no_cuda_probe(self):
        root = self.repository()
        random.seed(101); np.random.seed(202); torch.manual_seed(303)
        py_before = random.getstate()
        np_before = np.random.get_state()
        torch_before = torch.get_rng_state().clone()
        forbidden = ("is_available", "device_count", "current_device",
                     "get_device_properties", "get_rng_state_all")
        patches = [mock.patch.object(torch.cuda, name,
                    side_effect=AssertionError(f"CPU capture called torch.cuda.{name}"))
                   for name in forbidden]
        mocks = [patch.start() for patch in patches]
        try:
            result = subject.collect_runtime_environment(
                root, profile=storage.MLP_FIXTURE, runtime_role="fixture_cpu")
        finally:
            for patch in reversed(patches):
                patch.stop()
        self.assertIs(subject.validate_environment(result, profile=storage.MLP_FIXTURE), result)
        self.assertTrue(all(item.call_count == 0 for item in mocks))
        self.assertEqual(py_before, random.getstate())
        current_np = np.random.get_state()
        self.assertEqual(np_before[0], current_np[0])
        np.testing.assert_array_equal(np_before[1], current_np[1])
        self.assertEqual(np_before[2:], current_np[2:])
        self.assertTrue(torch.equal(torch_before, torch.get_rng_state()))
        self.assertFalse(torch.cuda.is_initialized())
        self.assertEqual(result["cuda"], {"initialized": False, "visible_device_count": None,
                         "current_device": None, "driver_version": None, "devices": []})

    def test_environment_schema_roles_and_scalar_types_are_strict(self):
        value = self.collect()
        class StringSubclass(str):
            pass

        mutations = [
            lambda row: row.update(runtime_role="cpu_audit"),
            lambda row: row.update(extra=None),
            lambda row: row["torch_settings"].update(num_threads=True),
            lambda row: row["cuda"].update(initialized=0),
            lambda row: row["rng_layout"]["python"].update(internal_length=624),
            lambda row: row["rng_layout"]["numpy"].update(keys_dtype="torch.uint32"),
            lambda row: row["rng_layout"]["torch_cpu"].update(state_length=0),
            lambda row: row.update(schema=StringSubclass(row["schema"])),
            lambda row: row.update(profile=StringSubclass(row["profile"])),
        ]
        for mutate in mutations:
            changed = copy.deepcopy(value)
            mutate(changed)
            with self.subTest(mutate=mutate), self.assertRaises(subject.SourceEnvironmentError):
                subject.validate_environment(changed, profile=storage.MLP_FIXTURE)
        with self.assertRaises(subject.SourceEnvironmentError):
            subject.validate_environment(value, profile=storage.SCIENTIFIC)
        double_root = copy.deepcopy(value)
        double_root["repository_root_realpath"] = "//" + value["repository_root_realpath"].lstrip("/")
        with self.assertRaises(subject.SourceEnvironmentError):
            subject.validate_environment(double_root, profile=storage.MLP_FIXTURE)

    def test_saved_state_environment_cross_binding_is_exact_and_pure(self):
        _, _, _, core = state_fixture()
        environment = self.collect(profile=storage.FIXTURE)
        before = torch.get_rng_state().clone()
        self.assertIsNone(subject.validate_state_environment(
            core, environment, profile=storage.FIXTURE))
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertFalse(torch.cuda.is_initialized())

        bad_length = copy.deepcopy(environment)
        bad_length["rng_layout"]["torch_cpu"]["state_length"] += 1
        with self.assertRaises(subject.SourceEnvironmentError):
            subject.validate_state_environment(core, bad_length, profile=storage.FIXTURE)
        bad_device = copy.deepcopy(core)
        bad_device["model"]["parameters"][0]["native_device"] = "cpu:0"
        with self.assertRaises(subject.SourceEnvironmentError):
            subject.validate_state_environment(bad_device, environment, profile=storage.FIXTURE)
        with self.assertRaises(subject.SourceEnvironmentError):
            subject.validate_state_environment({}, environment, profile=storage.FIXTURE)


if __name__ == "__main__":
    unittest.main()
