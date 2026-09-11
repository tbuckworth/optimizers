"""Dataset-free input-adapter fixtures; scientific RNG calls fail before drawing."""
from __future__ import annotations

import copy
import fcntl
import hashlib
import os
from pathlib import Path
import random
import resource
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
    raise RuntimeError("explicitly hide CUDA for native input fixtures")
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    if os.environ.get(key) != "1":
        raise RuntimeError("single numerical threads required for native input fixtures")

import numpy as np
import torch
import artifact_store as storage
import data_probe_bindings as data
import identity_codec as codec
import native_inputs as subject
import native_phase_policy as policy
import plan_bindings as plans
import runtime_guard as runtime
import verified_plan_load
from test_data_probe_bindings import synthetic_case
from test_plan_bindings import fixture

HERE = Path(__file__).resolve().parent
BIG_TMP = Path("/tmp/spectral-experiment-artifacts")


class NativeInputTests(unittest.TestCase):
    def setUp(self):
        self.started = time.monotonic()
        self.identity, _, _ = fixture()
        self.images, self.labels, self.kw, _, _ = synthetic_case()
        self.pins = copy.deepcopy(self.kw["expected_files"])
        self.directory = None
        self.checkpoint()

    def checkpoint(self, stage="fixture.local"):
        self.assertIs(type(stage), str)
        self.assertTrue(stage.isascii())
        self.assertLessEqual(time.monotonic()-self.started, 120.)
        self.assertLessEqual(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024, 2 << 30)
        self.assertFalse(torch.cuda.is_initialized())
        if self.directory is not None:
            logical = sum(path.lstat().st_size for path in self.directory.rglob("*") if path.is_file())
            self.assertLessEqual(logical, (2 << 20)-(1 << 20))

    def temporary(self):
        runtime.verify_big_volume(BIG_TMP)
        self.assertGreaterEqual(shutil.disk_usage(BIG_TMP).free, 1 << 30)
        temporary = tempfile.TemporaryDirectory(prefix="i7-native-input-test-", dir=BIG_TMP)
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        return self.directory

    def files(self):
        directory = self.temporary()
        for name, body in zip(subject.FILE_NAMES, (self.images, self.labels)):
            (directory/name).write_bytes(body)
        self.checkpoint()
        return directory

    def read(self, directory, **updates):
        kwargs = dict(directory=directory, expected_files=self.pins,
                      profile=subject.MLP_FIXTURE, checkpoint=self.checkpoint)
        kwargs.update(updates)
        return subject.read_training_idx(**kwargs)

    def generate(self, **updates):
        kwargs = dict(identity=self.identity, profile=subject.MLP_FIXTURE, checkpoint=self.checkpoint)
        kwargs.update(updates)
        return subject.generate_plan(**kwargs)

    def test_import_and_default_cli_inert_without_numerical_imports(self):
        directory = self.temporary()
        code = ("import sys;sys.path.insert(0,sys.argv[1]);import native_inputs;"
                "print('torch' in sys.modules, 'numpy' in sys.modules)")
        result = subprocess.run([sys.executable, "-c", code, str(HERE)], cwd=directory,
                                capture_output=True, text=True, check=True, timeout=10)
        self.assertEqual(result.stdout, "False False\n")
        result = subprocess.run([sys.executable, str(HERE/"native_inputs.py")], cwd=directory,
                                capture_output=True, text=True, check=True, timeout=10)
        self.assertEqual(result.stdout, "native_inputs: inert; no plan generation, data read or execution action\n")
        self.assertEqual(list(directory.iterdir()), [])

    def test_mlp_generation_exact_local_streams_and_global_rng_neutrality(self):
        before = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone())
        plan = self.generate()
        def rng(stream):
            return np.random.Generator(np.random.PCG64(np.random.SeedSequence([20260906, 0, stream])))
        expected = dict(permutation=rng(0).permutation(30), replacement_uniforms=rng(1).random(10),
            replacement_digits=rng(2).integers(0, 2, size=10),
            training_batches=rng(4).integers(0, 10, size=(8, 4)),
            training_probe_indices=rng(5).integers(0, 10, size=4))
        for name, array in expected.items():
            np.testing.assert_array_equal(plan[name].numpy(), array)
        self.assertEqual(plan["initialization_seed"], int(rng(3).integers(0, 2**32, dtype=np.uint32)))
        self.assertEqual(tuple(plan), plans.PLAN_KEYS)
        self.assertEqual(codec.tree_digest(plan), codec.tree_digest(self.generate()))
        pointers = []
        for name in plans.ARRAY_KEYS:
            tensor = plan[name]
            self.assertIsNone(tensor._base)
            self.assertEqual(tensor.untyped_storage().nbytes(), tensor.numel()*tensor.element_size())
            pointers.append(tensor.untyped_storage().data_ptr())
        self.assertEqual(len(pointers), len(set(pointers)))
        for index, name in enumerate(("train_indices", "validation_indices", "auxiliary_indices")):
            self.assertTrue(torch.equal(plan[name], plan["permutation"][index*10:(index+1)*10]))
        self.assertEqual(random.getstate(), before[0])
        after = np.random.get_state()
        self.assertEqual(before[1][0], after[0])
        np.testing.assert_array_equal(before[1][1], after[1])
        self.assertEqual(before[1][2:], after[2:])
        self.assertTrue(torch.equal(before[2], torch.get_rng_state()))

    def test_scientific_call_shapes_all_members_without_any_draw(self):
        class NoDraw(Exception):
            pass
        calls = []
        class FailBeforeDraw:
            def __getattr__(self, method):
                def fail(*args, **kwargs):
                    calls.append((method, args, kwargs))
                    raise NoDraw
                return fail
        members = [("primary", bundle, 101) for bundle in (71001, 71002, 71003)]
        members += [("sensitivity", 71901, 101), ("pilot", 71990, 101)]
        for role, bundle, update in members:
            identity = policy.identity(role, bundle, update)
            expected = [
                ("permutation", "permutation", (60000,), {}),
                ("replacement_uniforms", "random", (5000,), {}),
                ("replacement_digits", "integers", (0, 10), {"size": 5000}),
                ("initialization_seed", "integers", (0, 2**32), {"dtype": np.uint32}),
                ("training_batches", "integers", (0, 5000), {"size": (identity["steps_total"], 64)}),
                ("training_probe_indices", "integers", (0, 5000), {"size": 256}),
            ]
            for stream, (field, method, args, kwargs) in enumerate(expected):
                with mock.patch.object(subject, "_rng", return_value=FailBeforeDraw()) as factory:
                    with self.assertRaises(NoDraw):
                        subject._draw_component(field, identity=identity, profile=subject.SCIENTIFIC)
                    factory.assert_called_once_with(bundle, stream)
                    self.assertEqual(calls[-1], (method, args, kwargs))
        self.assertEqual(len(calls), 30)
        # Public orchestration also stops at stream0, before producing any plan.
        with mock.patch.object(subject, "_rng", return_value=FailBeforeDraw()) as factory:
            with self.assertRaises(NoDraw):
                self.generate(identity=policy.identity("pilot", 71990, 101), profile=subject.SCIENTIFIC)
            factory.assert_called_once_with(71990, 0)

    def test_invalid_identity_and_guard_failure_never_draw(self):
        invalid = copy.deepcopy(self.identity)
        invalid["bundle"] = 1
        with mock.patch.object(subject, "_rng") as factory:
            with self.assertRaises(codec.CodecError):
                self.generate(identity=invalid)
            with self.assertRaises(subject.InputError):
                self.generate(profile=storage.FIXTURE)
            with self.assertRaisesRegex(RuntimeError, "guard_stop"):
                self.generate(checkpoint=mock.Mock(side_effect=RuntimeError("guard_stop")))
            factory.assert_not_called()

    def test_existing_runtime_guard_signature_and_closed_stages(self):
        guard = runtime.RuntimeGuard("primary", profile=storage.MLP_FIXTURE)
        stages = []
        def checkpoint(stage):
            stages.append(stage)
            guard.check(stage)
            self.checkpoint(stage)
        self.generate(checkpoint=checkpoint)
        fields = ("permutation", "replacement_uniforms", "replacement_digits",
                  "initialization_seed", "training_batches", "training_probe_indices")
        self.assertEqual(stages, ["adapter.plan.import.pre", *[
            "adapter.plan."+field+"."+edge for field in fields for edge in ("pre", "post")],
            "adapter.plan.final"])
        directory = self.files()
        stages.clear()
        self.read(directory, checkpoint=checkpoint)
        self.assertEqual(stages, ["adapter.idx.open.pre", "adapter.idx.open.training_images",
            "adapter.idx.open.training_labels", "adapter.idx.read.training_images",
            "adapter.idx.verified.training_images", "adapter.idx.read.training_labels",
            "adapter.idx.verified.training_labels", "adapter.idx.final"])

    def test_generate_store_verified_load_materialize_integration(self):
        directory = self.files()
        buffers = self.read(directory)
        plan = self.generate()
        with storage.ArtifactStore(directory, profile=storage.MLP_FIXTURE, budget_bytes=2 << 20,
                failure_reserve_bytes=1 << 20, min_filesystem_free_bytes=1 << 30) as store:
            receipt = store.write_tensor_tree("fixture-plan.pt", plan)
        self.checkpoint()
        loaded = verified_plan_load.load_verified_plan(store.root, "fixture-plan.pt", identity=self.identity,
            profile=storage.MLP_FIXTURE, expected_sha256=receipt["sha256"])
        result = data.materialize(buffers["images_bytes"], buffers["labels_bytes"], plan=loaded["plan"],
            identity=self.identity, profile=storage.MLP_FIXTURE, expected_files=buffers["expected_files"])
        data.validate_materialization(result, buffers["images_bytes"], buffers["labels_bytes"],
            plan=loaded["plan"], identity=self.identity, profile=storage.MLP_FIXTURE,
            expected_files=buffers["expected_files"])
        self.assertEqual(tuple(result), ("datasets", "probes", "bindings"))
        self.assertEqual(result["datasets"]["train_inputs"].shape, (10, 3))
        self.assertEqual(result["probes"]["auxiliary_clean"]["inputs"].shape, (10, 3))
        self.checkpoint()

    def test_loader_exact_bytes_owned_pins_only_two_training_opens(self):
        directory = self.files()
        (directory/"t10k-images-idx3-ubyte").write_bytes(b"must not open")
        with mock.patch.object(subject.os, "open", wraps=os.open) as opener:
            result = self.read(directory)
        actual = [call.args[0] for call in opener.call_args_list if not call.args[1] & os.O_DIRECTORY]
        self.assertEqual(actual, list(subject.FILE_NAMES))
        self.assertEqual(result, dict(images_bytes=self.images, labels_bytes=self.labels, expected_files=self.pins))
        result["expected_files"]["training_images"]["sha256"] = "0"*64
        self.assertNotEqual(result["expected_files"], self.pins)

    def test_loader_rejects_pin_schema_before_any_open(self):
        bad = []
        for field, value in (("sha256", "A"*64), ("size_bytes", True), ("size_bytes", 47040017)):
            pins = copy.deepcopy(self.pins)
            pins["training_images"][field] = value
            bad.append(pins)
        bad += [dict(reversed(list(self.pins.items()))), {**self.pins, "test": {}}, {}]
        with mock.patch.object(subject.os, "open") as opener:
            for pins in bad:
                with self.assertRaises(subject.InputError):
                    self.read("/nonexistent", expected_files=pins)
            opener.assert_not_called()

    def test_loader_rejects_hash_length_header_and_swapped_inputs(self):
        directory = self.files()
        image_path = directory/subject.FILE_NAMES[0]
        with self.assertRaises(subject.InputError):
            wrong = copy.deepcopy(self.pins)
            wrong["training_images"]["sha256"] = "0"*64
            self.read(directory, expected_files=wrong)
        for body in (self.images[:-1], self.images+b"x", self.labels):
            image_path.write_bytes(body)
            with self.assertRaises(subject.InputError):
                self.read(directory)
        body = b"\0"*16+self.images[16:]
        image_path.write_bytes(body)
        wrong = copy.deepcopy(self.pins)
        wrong["training_images"]["sha256"] = hashlib.sha256(body).hexdigest()
        with self.assertRaisesRegex(subject.InputError, "header"):
            self.read(directory, expected_files=wrong)

    def test_loader_refuses_symlinks_hardlinks_fifo_and_noncanonical_directory(self):
        directory = self.files()
        image_path = directory/subject.FILE_NAMES[0]
        saved = directory/"saved-images"
        image_path.rename(saved)
        for kind in ("symlink", "hardlink", "fifo"):
            if kind == "symlink":
                image_path.symlink_to(saved)
            elif kind == "hardlink":
                os.link(saved, image_path)
            else:
                os.mkfifo(image_path)
            try:
                with self.assertRaises(subject.InputError):
                    self.read(directory)
            finally:
                image_path.unlink()
        saved.rename(image_path)
        alias = directory/"alias"
        alias.symlink_to(directory, target_is_directory=True)
        for path in (alias, str(directory)+"/", str(directory)+"/../"+directory.name, "relative"):
            with self.assertRaises(subject.InputError):
                self.read(path)

    def test_both_file_locks_held_before_first_read_and_contention_refused(self):
        directory = self.files()
        locked = []
        original_flock, original_read = fcntl.flock, os.read
        def lock(fd, operation):
            self.assertEqual(operation, fcntl.LOCK_SH | fcntl.LOCK_NB)
            original_flock(fd, operation)
            locked.append(fd)
        def read(fd, size):
            self.assertEqual(len(locked), 2)
            self.assertLessEqual(size, subject._CHUNK)
            return original_read(fd, size)
        with mock.patch.object(subject.fcntl, "flock", side_effect=lock), \
             mock.patch.object(subject.os, "read", side_effect=read):
            self.read(directory)
        with (directory/subject.FILE_NAMES[1]).open("rb") as writer:
            fcntl.flock(writer, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with mock.patch.object(subject.os, "read") as reader:
                with self.assertRaises(subject.InputError):
                    self.read(directory)
                reader.assert_not_called()

    def test_loader_detects_replacement_after_hash_and_propagates_guard(self):
        directory = self.files()
        changed = False
        original_read = os.read
        def read(fd, size):
            nonlocal changed
            body = original_read(fd, size)
            if not changed and body == self.images:
                image = directory/subject.FILE_NAMES[0]
                image.rename(directory/"old-images")
                image.write_bytes(self.images)
                changed = True
            return body
        with mock.patch.object(subject.os, "read", side_effect=read):
            with self.assertRaisesRegex(subject.InputError, "changed"):
                self.read(directory)
        with mock.patch.object(subject.os, "open") as opener:
            with self.assertRaisesRegex(RuntimeError, "guard_stop"):
                self.read(directory, checkpoint=mock.Mock(side_effect=RuntimeError("guard_stop")))
            opener.assert_not_called()


if __name__ == "__main__":
    unittest.main()
