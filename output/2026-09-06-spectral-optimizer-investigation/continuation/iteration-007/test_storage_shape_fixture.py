"""CPU-only full-width storage specimen tests; never a scientific pilot."""
import os
import contextlib
import io
import unittest
from unittest import mock

if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
    raise RuntimeError("explicitly hide CUDA for sizing fixtures")

import torch
import storage_shape_fixture as subject


class StorageShapeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_exact_fixed_tensor_ledger_and_no_aliases(self):
        before = torch.get_rng_state().clone()
        trees = subject.specimens()
        inventory = {key: subject.inventory(tree) for key, tree in trees.items()}
        self.assertEqual({key: sum(x["tensor_bytes"] for x in rows) for key, rows in inventory.items()},
                         subject.TENSOR_BYTES)
        self.assertEqual(sum(subject.CORE_FIXED_BYTES.values()), 48041056)
        self.assertEqual(sum(subject.TENSOR_BYTES.values()), 48083616)
        indices = trees["anchor"]["payload"]["bindings"]
        self.assertEqual(indices["plan"]["next_batch_indices"].shape, (64,))
        self.assertEqual(indices["probes"]["training_probe_indices"].shape, (256,))
        self.assertEqual(indices["probes"]["auxiliary_indices"].shape, (5000,))
        self.assertTrue(all(value.dtype == torch.int64 for _, value in subject.tensor_leaves(indices)))
        leaves = list(subject.tensor_leaves(trees))
        self.assertEqual(len({value.untyped_storage().data_ptr() for _, value in leaves}), len(leaves))
        self.assertEqual(sum(value.shape == (50890, 32) for _, value in leaves), 5)
        self.assertEqual(sum(value.shape == (50890,) and value.dtype == torch.float64 for _, value in leaves), 16)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertFalse(torch.cuda.is_initialized())

    def test_uncompressed_roundtrip_and_pattern_independent_size(self):
        zero = subject.measure("zero")
        ones = subject.measure("ones")
        self.assertEqual(zero["specimen_serialized_bytes"], ones["specimen_serialized_bytes"])
        self.assertFalse(zero["scientific_launch_approved"])
        self.assertEqual(zero["complete_attempt_feasibility"], "unresolved")
        self.assertGreater(zero["specimen_serialized_bytes"], zero["measured_tensor_bytes"])
        for name, row in zero["artifacts"].items():
            self.assertTrue(row["all_zip_entries_uncompressed"])
            self.assertTrue(row["restricted_tensor_roundtrip"])
            self.assertNotEqual(row["serialized_sha256"], ones["artifacts"][name]["serialized_sha256"])

    def test_oversize_is_terminal_and_does_not_decode(self):
        tree = dict(profile=subject.PROFILE, scientific_envelope=False, value=torch.zeros(32))
        with mock.patch.object(subject.torch, "load") as decoder:
            with self.assertRaises((subject.storage.StoreError, RuntimeError)):
                subject.serialized_measurement(tree, buffer_limit=64)
            decoder.assert_not_called()

    def test_reject_alias_view_bad_profile_and_limit(self):
        value = torch.zeros(32)
        tree = dict(profile=subject.PROFILE, scientific_envelope=False, value=value, duplicate=value)
        with self.assertRaisesRegex(subject.storage.StoreError, "shared"):
            subject.serialized_measurement(tree)
        del tree["duplicate"]
        tree["value"] = value[:2]
        with self.assertRaises(subject.storage.StoreError):
            subject.serialized_measurement(tree)
        tree["value"] = value
        for limit in (True, -1, 0, 65 << 20, 1.5):
            with self.assertRaises(ValueError):
                subject.serialized_measurement(tree, buffer_limit=limit)
        tree["profile"] = "scientific_mnist_current32_v1"
        with self.assertRaisesRegex(ValueError, "non-scientific"):
            subject.serialized_measurement(tree)
        with self.assertRaises(ValueError):
            subject.specimens("random")

    def test_default_cli_is_inert_and_cpu_boundary_is_enforced(self):
        with mock.patch("sys.argv", ["storage_shape_fixture.py"]), contextlib.redirect_stdout(io.StringIO()):
            with mock.patch.object(subject, "measure") as measurement:
                subject.main()
                measurement.assert_not_called()
        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"}):
            with self.assertRaisesRegex(ValueError, "hide CUDA"):
                subject.specimens()
        with mock.patch.object(torch.cuda, "is_initialized", return_value=True):
            with self.assertRaisesRegex(ValueError, "uninitialized"):
                subject.specimens()


if __name__ == "__main__":
    unittest.main()
