"""CPU-only parsing, inertness and paired-sizing regression tests."""
import contextlib
import hashlib
import io
import json
import os
import tempfile
import unittest
from unittest import mock

import native_metadata_storage as subject


class NativeMetadataStorageTests(unittest.TestCase):
    def observation(self):
        return {
            "schema": "i7_native_layout_inspection_v2", "status": "pass",
            "scope": "engineering_native_layout_only",
            "worker_gpu_process_absent_after_exit": True,
            "certificates": {"scientific_execution": False, "pilot": False,
                "full_fit": False, "capture_neutrality": False, "continuation_restore": False},
            "source_binding": {"z": 1, "a": 2},
        }

    def test_order_and_sha_bound_load(self):
        value = self.observation()
        raw = json.dumps(value, separators=(",", ":")).encode() + b"\n"
        with tempfile.NamedTemporaryFile() as handle:
            handle.write(raw); handle.flush()
            loaded, size = subject.load_inspection(handle.name, hashlib.sha256(raw).hexdigest())
            self.assertEqual(size, len(raw))
            self.assertEqual(tuple(loaded["source_binding"]), ("z", "a"))
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                subject.load_inspection(handle.name, "0" * 64)

    def test_reject_duplicate_nonfinite_failed_and_symlink(self):
        failed = self.observation(); failed["status"] = "failed"
        old_version = self.observation(); old_version["schema"] = "i7_native_layout_inspection_v1"
        wrong_bool = self.observation(); wrong_bool["certificates"]["scientific_execution"] = 0
        candidates = [b'{"x":1,"x":2}\n', b'{"x":NaN}\n',
                      b'[]\n',
                      json.dumps(wrong_bool, separators=(",", ":")).encode() + b"\n",
                      json.dumps(old_version, separators=(",", ":")).encode() + b"\n",
                      json.dumps(failed, separators=(",", ":")).encode() + b"\n"]
        for raw in candidates:
            with self.subTest(raw=raw), tempfile.NamedTemporaryFile() as handle:
                handle.write(raw); handle.flush()
                with self.assertRaises(ValueError):
                    subject.load_inspection(handle.name, hashlib.sha256(raw).hexdigest())
        with tempfile.TemporaryDirectory() as directory:
            link = directory + "/link"
            os.symlink("missing", link)
            with self.assertRaises(OSError):
                subject.load_inspection(link, "0" * 64)

    def test_default_inert_and_measure_requires_hidden_cuda(self):
        with mock.patch.object(subject, "measure") as measure:
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(subject.main([]), 0)
            measure.assert_not_called()
        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"}):
            with self.assertRaisesRegex(ValueError, "hidden CUDA"):
                subject.measure("not-read", "not-read")

    def test_paired_measurement_changes_only_metadata(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        import torch
        import artifact_store as storage
        import full_envelope_storage_fixture as fixture
        import source_environment as provenance
        torch.set_num_threads(1)
        value = self.observation()
        value.update(expected_commit="a" * 40, expected_gpu_uuid="test-uuid")
        sources = {"repository_revision": "a" * 40,
                   "repository_root_realpath": "/test/root", "files": ["synthetic"]}
        environment = {"repository_root_realpath": "/test/root", "runtime_role": "native_source",
            "cuda": {"devices": [{"uuid": "test-uuid"}]},
            "rng_layout": {"torch_cpu": {"state_length": 5056},
                           "torch_cuda": [{"state_length": 16}]}}
        value.update(source_binding=sources, native_environment_binding=environment)
        raw = json.dumps(value, separators=(",", ":")).encode() + b"\n"
        tree = {"profile": fixture.SPECIMEN_PROFILE, "payload": {
            "bindings": {"sources": {}, "environment": {}},
            "rng": {"torch_cpu": torch.zeros(5056, dtype=torch.uint8)}}}
        with tempfile.NamedTemporaryFile() as handle:
            handle.write(raw); handle.flush()
            with mock.patch.object(provenance, "validate_sources") as validate_sources, \
                 mock.patch.object(provenance, "validate_environment") as validate_environment, \
                 mock.patch.object(fixture, "actual_tiny_records", return_value={"anchor": tree}), \
                 mock.patch.object(fixture, "lift_tree", return_value={"tree": tree}):
                result = subject.measure(handle.name, hashlib.sha256(raw).hexdigest())
            validate_sources.assert_called_once_with(sources, profile=storage.SCIENTIFIC)
            validate_environment.assert_called_once_with(environment, profile=storage.SCIENTIFIC)
        self.assertEqual(tree["payload"]["bindings"], {"sources": {}, "environment": {}})
        delta = result["components"]["pilot"]["serialized_delta_bytes"]
        self.assertGreater(delta, 0)
        self.assertEqual(result["aggregate_paired_anchor_metadata_delta_bytes"], 18 * delta)
        self.assertEqual(result["rng_tensor_arithmetic"]["native_state_tensor_delta_all_cores"], 384)
        self.assertEqual(result["rng_tensor_arithmetic"]["schema_cuda_continuation_witness_bytes_all_cores"], 384)
        self.assertFalse(result["complete_attempt_fit_certificate"])
        self.assertFalse(result["cuda_initialized"])


if __name__ == "__main__":
    unittest.main()
