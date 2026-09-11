"""Tiny CPU store→receipt→exact bytes→plan→binding integration; no scientific plans."""
import io
import os
from pathlib import Path
import random
import tempfile
import unittest
from unittest import mock
import zipfile

if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
    raise RuntimeError("explicitly hide CUDA for verified-plan fixtures")

import numpy as np
import torch
import artifact_store as storage
import identity_codec as codec
import verified_plan_load as subject
from test_plan_bindings import fixture


class VerifiedPlanLoadTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="i7-verified-plan-")
        self.addCleanup(self.directory.cleanup)

    def store(self, profile=storage.MLP_FIXTURE):
        return storage.ArtifactStore(self.directory.name, profile=profile, min_filesystem_free_bytes=0)

    def test_both_fixture_plans_exact_verified_return_and_no_rng(self):
        py, np_before, cpu = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
        for profile in storage.FIXTURES:
            identity, plan, _ = fixture(profile)
            with self.store(profile) as store:
                receipt = store.write_tensor_tree("frozen-plan.pt", plan)
            with mock.patch.object(subject.torch, "load", wraps=torch.load) as decoder:
                result = subject.load_verified_plan(store.root, "frozen-plan.pt", identity=identity,
                                                    profile=profile, expected_sha256=receipt["sha256"])
            self.assertEqual(decoder.call_count, 1)
            self.assertEqual(decoder.call_args.kwargs, {"weights_only": True, "map_location": "cpu"})
            self.assertEqual(tuple(result), ("store", "artifact", "plan", "binding", "plan_content_sha256"))
            self.assertEqual(result["artifact"]["sha256"], receipt["sha256"])
            self.assertEqual(result["artifact"]["name"], "frozen-plan.pt")
            self.assertEqual(result["artifact"]["status"], "complete")
            self.assertEqual(result["artifact"]["receipt_name"], receipt["receipt_name"])
            self.assertEqual(result["plan_content_sha256"], codec.tree_digest(plan))
            self.assertEqual(codec.tree_digest(result["plan"]), codec.tree_digest(plan))
            self.assertTrue(torch.equal(result["binding"]["next_batch_indices"], plan["training_batches"][4]))
            self.assertNotEqual(result["binding"]["next_batch_indices"].untyped_storage().data_ptr(),
                                result["plan"]["training_batches"].untyped_storage().data_ptr())
        self.assertEqual(random.getstate(), py)
        now = np.random.get_state()
        self.assertEqual(np_before[0], now[0])
        np.testing.assert_array_equal(np_before[1], now[1])
        self.assertEqual(np_before[2:], now[2:])
        self.assertTrue(torch.equal(cpu, torch.get_rng_state()))
        self.assertFalse(torch.cuda.is_initialized())

    def test_ordinary_adamw_lazy_imports_do_not_reject_verified_plan(self):
        with torch.random.fork_rng(devices=[]):
            model = torch.nn.Linear(3, 2)
            torch.optim.AdamW(model.parameters())
        safe_before = list(torch.serialization.get_safe_globals())
        identity, plan, _ = fixture()
        with self.store() as store:
            receipt = store.write_tensor_tree("plan.pt", plan)
        loaded = subject.load_verified_plan(store.root, "plan.pt", identity=identity,
            profile=storage.MLP_FIXTURE, expected_sha256=receipt["sha256"])
        self.assertEqual(loaded["plan_content_sha256"], codec.tree_digest(plan))
        self.assertEqual(set(torch.serialization.get_safe_globals()), set(safe_before))
        self.assertFalse(torch.cuda.is_initialized())

    def test_header_profile_and_bytes_encoding_rejected_before_decoder(self):
        identity, plan, _ = fixture(storage.FIXTURE)
        for wrong_profile in (True, False):
            with self.store(storage.MLP_FIXTURE if wrong_profile else storage.FIXTURE) as store:
                if wrong_profile:
                    receipt = store.write_tensor_tree("plan.pt", plan)
                else:
                    buffer = io.BytesIO()
                    torch.save(plan, buffer)
                    receipt = store.write_bytes("plan.pt", buffer.getvalue())
            with mock.patch.object(subject.torch, "load") as decoder:
                with self.assertRaises(subject.VerifiedPlanLoadError):
                    subject.load_verified_plan(store.root, "plan.pt", identity=identity,
                        profile=storage.FIXTURE, expected_sha256=receipt["sha256"])
                decoder.assert_not_called()

    def test_structurally_invalid_plan_rejected_after_restricted_load(self):
        identity, plan, _ = fixture()
        plan["training_probe_indices"][0] = 10
        with self.store() as store:
            receipt = store.write_tensor_tree("plan.pt", plan)
        with mock.patch.object(subject.torch, "load", wraps=torch.load) as decoder:
            with self.assertRaises(subject.VerifiedPlanLoadError):
                subject.load_verified_plan(store.root, "plan.pt", identity=identity,
                    profile=storage.MLP_FIXTURE, expected_sha256=receipt["sha256"])
            self.assertEqual(decoder.call_count, 1)

    def test_custom_safe_globals_rejected_without_modifying_them(self):
        identity, plan, _ = fixture()
        with self.store() as store:
            receipt = store.write_tensor_tree("plan.pt", plan)
        with mock.patch.object(subject.torch.serialization, "get_safe_globals", return_value=[object]), \
             mock.patch.object(subject.torch, "load") as decoder:
            with self.assertRaisesRegex(subject.VerifiedPlanLoadError, "custom safe globals"):
                subject.load_verified_plan(store.root, "plan.pt", identity=identity,
                    profile=storage.MLP_FIXTURE, expected_sha256=receipt["sha256"])
            decoder.assert_not_called()

    def test_preflight_storage_counts_sizes_and_compression(self):
        identity, plan, _ = fixture()
        buffer = io.BytesIO()
        torch.save(plan, buffer)
        data = buffer.getvalue()
        subject._preflight(data, profile=storage.MLP_FIXTURE, steps=identity["steps_total"])
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            rows = [(entry.filename, archive.read(entry.filename)) for entry in archive.infolist()]
        for mode in ("compressed", "wrong_size", "wrong_storage", "duplicate", "path_escape"):
            output = io.BytesIO()
            with zipfile.ZipFile(output, "w") as archive:
                for name, part in rows:
                    if mode == "wrong_size" and name.endswith("/data/0"):
                        part += b"x"
                    if mode == "wrong_storage" and name.endswith("/data/0"):
                        name = name[:-1] + "8"
                    if mode == "path_escape" and name.endswith("/data/0"):
                        name = "../data/0"
                    archive.writestr(name, part, compress_type=zipfile.ZIP_DEFLATED if mode == "compressed"
                                     else zipfile.ZIP_STORED)
                if mode == "duplicate":
                    # Duplicate names are intentional adversarial fixture data.
                    import warnings
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", UserWarning)
                        archive.writestr(rows[0][0], rows[0][1])
            with self.subTest(mode=mode), self.assertRaises(subject.VerifiedPlanLoadError):
                subject._preflight(output.getvalue(), profile=storage.MLP_FIXTURE,
                                   steps=identity["steps_total"])

    def test_absolute_path_name_digest_and_size_bounds(self):
        identity, plan, _ = fixture()
        with self.store() as store:
            receipt = store.write_tensor_tree("plan.pt", plan)
        for root, name, digest in (("relative", "plan.pt", receipt["sha256"]),
                                   (str(store.root) + "/", "plan.pt", receipt["sha256"]),
                                   (store.root, "../plan.pt", receipt["sha256"]),
                                   (store.root, "plan.pt", "A" * 64),
                                   (store.root, "plan.pt", "0" * 64)):
            with self.subTest(root=root, name=name), mock.patch.object(subject.torch, "load") as decoder:
                with self.assertRaises(subject.VerifiedPlanLoadError):
                    subject.load_verified_plan(root, name, identity=identity, profile=storage.MLP_FIXTURE,
                                               expected_sha256=digest)
                decoder.assert_not_called()
        with self.store() as oversized:
            receipt = oversized.write_bytes("plan.pt", b"x" * (subject.FIXTURE_FILE_MAX + 1))
        with mock.patch.object(subject.torch, "load") as decoder:
            with self.assertRaises(subject.VerifiedPlanLoadError):
                subject.load_verified_plan(oversized.root, "plan.pt", identity=identity,
                    profile=storage.MLP_FIXTURE, expected_sha256=receipt["sha256"])
            decoder.assert_not_called()

    def test_held_dirfd_inspection_matches_public_api(self):
        _, plan, _ = fixture()
        with self.store() as store:
            store.write_tensor_tree("plan.pt", plan)
        fd = os.open(store.root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            report, snapshot = storage._inspect_dirfd(fd)
            self.assertEqual(report, storage.ArtifactStore.inspect(store.root))
            self.assertEqual(set(snapshot), set(report["files"]))
            os.fstat(fd)  # Helper must leave ownership with its caller.
        finally:
            os.close(fd)


if __name__ == "__main__":
    unittest.main()
