"""Synthetic CPU-only representation tests; no device UUID is observed here."""
import copy
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
    raise RuntimeError("CUDA identity tests require hidden CUDA")

import torch
import artifact_store as storage
import cuda_identity as subject
import native_layout_inspection as inspection
import runtime_guard as guard
import source_environment as environment
import source_environment_schema as schema
import state_core as state
from test_source_environment import TemporaryGitMixin
from test_state_core import fixture as state_fixture

BARE = "994b97cd-d768-1f03-78fb-6a70e3cc1e0c"
CANONICAL = "GPU-" + BARE


class RepresentationTests(unittest.TestCase):
    def test_text_bytes_and_canonical_predicate(self):
        for text in (BARE, CANONICAL, BARE.upper(), "gPu-" + BARE.upper()):
            for value in (text, text.encode("ascii")):
                with self.subTest(value=value):
                    self.assertEqual(subject.canonical_cuda_uuid(value), CANONICAL)
        for value in (BARE, CANONICAL.upper(), CANONICAL.encode(), None):
            self.assertFalse(subject.is_canonical_cuda_uuid(value))
        self.assertTrue(subject.is_canonical_cuda_uuid(CANONICAL))

    def test_malformed_and_arbitrary_objects_rejected_without_conversion(self):
        class StringSubclass(str):
            def __str__(self):
                raise AssertionError("must not convert unknown subclass")
        class BytesSubclass(bytes):
            pass
        class Duck:
            def __str__(self):
                raise AssertionError("must not convert arbitrary object")
        bad = [None, 1, True, Duck(), StringSubclass(CANONICAL),
               BytesSubclass(CANONICAL.encode()), bytearray(CANONICAL.encode()),
               memoryview(CANONICAL.encode()), "", "GPU-fixture", " " + BARE,
               BARE + "\n", BARE + "\0", "{" + BARE + "}", BARE.replace("-", ""),
               "MIG-" + BARE, "GPU--" + BARE, "X" + BARE[1:],
               b"\xff" * 36, "\N{SNOWMAN}" * 36]
        for value in bad:
            with self.subTest(kind=type(value).__name__), self.assertRaises(ValueError):
                subject.canonical_cuda_uuid(value)
        self.assertFalse(torch.cuda.is_initialized())

    def test_mock_exact_native_type_and_three_consumer_agreement(self):
        class NativeUUIDDouble:
            def __str__(self):
                return BARE
        class NativeSubclass(NativeUUIDDouble):
            pass
        properties = SimpleNamespace(uuid=NativeUUIDDouble(), name=guard.CUDA_DEVICE_NAME,
                                     total_memory=24 << 30)
        with self.assertRaises(ValueError):
            subject.canonical_cuda_uuid(properties.uuid)
        with mock.patch.object(torch._C, "_CUuuid", NativeUUIDDouble):
            with self.assertRaises(ValueError):
                subject.canonical_cuda_uuid(NativeSubclass())
            self.assertEqual(environment._uuid(properties), CANONICAL)
            with mock.patch.object(torch.cuda, "get_device_properties", return_value=properties):
                self.assertEqual(state._cuda_identity(0), (guard.CUDA_DEVICE_NAME, CANONICAL))
                with mock.patch.object(torch.cuda, "is_initialized", return_value=True), \
                     mock.patch.object(torch.cuda, "current_device", return_value=0):
                    self.assertEqual(guard._native_cuda_identity()["stable_identity"], CANONICAL)
        self.assertFalse(torch.cuda.is_initialized())

    def test_import_and_text_conversion_are_torch_free(self):
        code = ("import sys,cuda_identity as c; assert 'torch' not in sys.modules; "
                f"assert c.canonical_cuda_uuid({BARE!r}) == {CANONICAL!r}; "
                "assert 'torch' not in sys.modules")
        subprocess.run([sys.executable, "-c", code], cwd=Path(subject.__file__).parent,
                       env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
                       check=True, capture_output=True, timeout=5)

    def test_driver_rows_use_same_normalization_and_unique_match(self):
        for uuid in (BARE, CANONICAL, CANONICAL.upper()):
            with mock.patch.object(environment, "_run", return_value=(uuid + ", 595.84\n").encode()):
                self.assertEqual(environment._driver_version("/synthetic", CANONICAL), "595.84")
        bad = [b"", b"malformed\n", b"GPU-fixture, 595.84\n",
               ("GPU-" + "0" * 8 + BARE[8:] + ", 595.84\n").encode(),
               ((BARE + ", 595.84\n") + (CANONICAL + ", 595.84\n")).encode()]
        for output in bad:
            with mock.patch.object(environment, "_run", return_value=output), \
                 self.assertRaises(environment.SourceEnvironmentError):
                environment._driver_version("/synthetic", CANONICAL)

    def test_schema_extraction_is_shared_and_torch_free(self):
        self.assertIs(environment.validate_sources, schema.validate_sources)
        self.assertIs(environment.validate_environment, schema.validate_environment)
        self.assertIs(environment.SourceEnvironmentError, schema.SourceEnvironmentError)
        self.assertEqual(schema.PROFILES,
                         (storage.SCIENTIFIC, storage.FIXTURE, storage.MLP_FIXTURE))
        self.assertEqual(len(schema.SCIENTIFIC_ROLES), 80)
        self.assertIn(("source_binding", schema.I7 + "source_environment_schema.py"),
                      schema.SCIENTIFIC_ROLES)
        for role, name in (("phase_controller", "native_control.py"),
                           ("producer", "native_inputs.py"),
                           ("producer", "source_streaming.py"),
                           ("producer", "native_source.py"),
                           ("producer", "native_branches.py"),
                           ("phase_controller", "native_controller.py"),
                           ("phase_controller", "phase_transition.py"),
                           ("phase_controller", "scientific_controller.py"),
                           ("independent_auditor", "native_audit.py"),
                           ("producer", "result_collection.py")):
            self.assertIn((role, schema.I7 + name), schema.SCIENTIFIC_ROLES)
        code = ("import sys,source_environment_schema as s; "
                "assert 'torch' not in sys.modules and 'numpy' not in sys.modules; "
                "assert len(s.SCIENTIFIC_ROLES) == 80; "
                "assert s.is_canonical_cuda_uuid(" + repr(CANONICAL) + "); "
                "assert 'torch' not in sys.modules and 'numpy' not in sys.modules")
        subprocess.run([sys.executable, "-c", code], cwd=Path(subject.__file__).parent,
                       env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
                       check=True, capture_output=True, timeout=5)


class StoredIdentityTests(TemporaryGitMixin, unittest.TestCase):
    def test_synthetic_native_environment_requires_already_canonical_ids(self):
        value = environment.collect_runtime_environment(
            self.repository(), profile=storage.MLP_FIXTURE, runtime_role="fixture_cpu")
        value.update(profile=storage.SCIENTIFIC, runtime_role="native_source")
        value["safe_environment"] = dict(inspection.EXPECTED_SAFE_ENVIRONMENT,
                                        CUDA_VISIBLE_DEVICES=CANONICAL)
        value["torch_settings"] = dict(inspection.EXPECTED_TORCH_SETTINGS)
        value["cuda"] = {"initialized": True, "visible_device_count": 1,
            "current_device": 0, "driver_version": "synthetic",
            "devices": [{"index": 0, "name": guard.CUDA_DEVICE_NAME, "uuid": CANONICAL,
                "total_memory_bytes": 24 << 30, "capability_major": 8, "capability_minor": 6}]}
        value["rng_layout"]["torch_cuda"] = [{"index": 0, "name": guard.CUDA_DEVICE_NAME,
            "uuid": CANONICAL, "dtype": "torch.uint8", "state_length": 16}]
        environment.validate_environment(value, profile=storage.SCIENTIFIC)
        for uuid in (BARE, CANONICAL.upper(), "GPU-fixture"):
            for path in ("device", "rng"):
                bad = copy.deepcopy(value)
                row = (bad["cuda"]["devices"][0] if path == "device"
                       else bad["rng_layout"]["torch_cuda"][0])
                row["uuid"] = uuid
                with self.assertRaisesRegex(environment.SourceEnvironmentError, "noncanonical"):
                    environment.validate_environment(bad, profile=storage.SCIENTIFIC)
        self.assertFalse(torch.cuda.is_initialized())

    def test_saved_rng_component_rejects_noncanonical_uuid(self):
        _, _, _, core = state_fixture()
        rng = state.clone_tree(core["rng"])
        rng["torch_cuda"] = [{"device_index": 0, "name": guard.CUDA_DEVICE_NAME,
            "uuid": CANONICAL, "state": torch.zeros(16, dtype=torch.uint8)}]
        rng["continuation_witness"]["torch_cuda"] = [torch.zeros(4, dtype=torch.float32)]
        state._validate_rng(rng, state.SCIENTIFIC_PROFILE)
        for uuid in (BARE, CANONICAL.upper(), "GPU-fixture"):
            bad = state.clone_tree(rng)
            bad["torch_cuda"][0]["uuid"] = uuid
            with self.assertRaisesRegex(ValueError, "CUDA identity invalid"):
                state._validate_rng(bad, state.SCIENTIFIC_PROFILE)
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == "__main__":
    unittest.main()
