"""CPU-only tests for the inert Torch ZIP accounting helper."""

from __future__ import annotations

import copy
import functools
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import unittest
from unittest import mock

import zip_storage_bound as subject


BIG_TMP = Path("/tmp/spectral-experiment-artifacts")
MOUNT_DEVICE = os.makedev(8, 16)
THREAD_ENV = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")


class ZipStorageBoundTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        info = BIG_TMP.stat()
        if not BIG_TMP.is_dir() or info.st_dev != MOUNT_DEVICE:
            raise RuntimeError("test temporary parent is not the verified big volume")
        volume = os.statvfs(BIG_TMP)
        if volume.f_bavail * volume.f_frsize < 1 << 30:
            raise RuntimeError("test temporary parent has less than 1 GiB free")
        for name in THREAD_ENV:
            if os.environ.get(name) != "1":
                raise RuntimeError("numerical thread environment is not frozen")
        cls._wall_origin = time.monotonic()
        cls._cpu_origin = time.process_time()

    @classmethod
    def tearDownClass(cls) -> None:
        if time.monotonic() - cls._wall_origin > 120.0:
            raise RuntimeError("ZIP tests exceeded the wall cap")
        if time.process_time() - cls._cpu_origin > 120.0:
            raise RuntimeError("ZIP tests exceeded the CPU cap")
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 > 2 << 30:
            raise RuntimeError("ZIP tests exceeded the RSS cap")

    def test_import_and_default_cli_are_torch_free_and_inert(self) -> None:
        directory = Path(subject.__file__).resolve().parent
        script = "import sys; sys.path.insert(0, sys.argv[1]); import zip_storage_bound; assert 'torch' not in sys.modules"
        environment = {
            "PATH": "/usr/bin:/bin",
            "PYTHONDONTWRITEBYTECODE": "1",
            "CUDA_VISIBLE_DEVICES": "",
            **{name: "1" for name in THREAD_ENV},
        }
        imported = subprocess.run(
            ["/usr/bin/python3.12", "-c", script, os.fspath(directory)],
            env=environment,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        self.assertEqual((imported.returncode, imported.stdout, imported.stderr), (0, "", ""))
        default = subprocess.run(
            ["/usr/bin/python3.12", os.fspath(Path(subject.__file__).resolve())],
            env=environment,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        self.assertEqual((default.returncode, default.stdout, default.stderr), (0, "", ""))

    def test_closed_six_record_arithmetic(self) -> None:
        value = subject.torch_save_zip_ceiling(10, ())
        self.assertEqual(value["record_count"], 6)
        self.assertEqual(value["record_payload_bytes_upper"], 61)
        self.assertEqual(value["local_header_bytes"], 180)
        self.assertEqual(value["local_filename_bytes"], 127)
        self.assertEqual(value["fb_extra_header_bytes"], 24)
        self.assertEqual(value["alignment_padding_bytes"], 272)
        self.assertEqual(value["data_descriptor_bytes_upper"], 96)
        self.assertEqual(value["local_and_data_bytes_upper"], 760)
        self.assertEqual(value["central_directory_bytes"], 403)
        self.assertEqual(value["zip64_end_bytes"], 98)
        self.assertEqual(value["archive_bytes_upper"], 1261)

    def test_storage_order_and_decimal_filename_crossing_are_counted(self) -> None:
        eleven = subject.torch_save_zip_ceiling(100, (20,) * 11)
        twelve = subject.torch_save_zip_ceiling(100, (20,) * 12)
        self.assertEqual(eleven["record_count"], 17)
        self.assertEqual(twelve["record_count"], 18)
        # data/11 has one more digit than data/9; both local and central names
        # are counted, while alignment is recalculated rather than guessed.
        self.assertEqual(twelve["local_filename_bytes"] - eleven["local_filename_bytes"], 15)
        self.assertEqual(twelve["central_directory_bytes"] - eleven["central_directory_bytes"], 61)
        retained_core = subject.torch_save_zip_ceiling(100, (16, 4))
        self.assertEqual(retained_core["storage_nbytes_upper"], [16, 4])

    def test_bound_is_componentwise_monotone(self) -> None:
        for pickle_size in range(1, 80):
            smaller = subject.torch_save_zip_ceiling(pickle_size, (0, 4, 16))["archive_bytes_upper"]
            larger = subject.torch_save_zip_ceiling(pickle_size + 1, (0, 4, 16))["archive_bytes_upper"]
            self.assertLessEqual(smaller, larger)
        baseline = subject.torch_save_zip_ceiling(100, (3, 7, 11))["archive_bytes_upper"]
        for index in range(3):
            values = [3, 7, 11]
            values[index] += 1
            self.assertLessEqual(
                baseline,
                subject.torch_save_zip_ceiling(100, tuple(values))["archive_bytes_upper"],
            )

    def test_zip64_threshold_branches_are_explicit_but_main_domain_rejects(self) -> None:
        threshold = subject.ZIP32_LIMIT
        self.assertEqual(subject.zip64_extra_bytes(threshold - 1, threshold - 1), 0)
        self.assertEqual(subject.zip64_extra_bytes(threshold, threshold - 1), 20)
        self.assertEqual(subject.zip64_extra_bytes(threshold - 1, threshold), 12)
        self.assertEqual(subject.zip64_extra_bytes(threshold, threshold), 28)
        with self.assertRaisesRegex(subject.ZipBoundError, "record size leaves"):
            subject.torch_save_zip_ceiling(1, (subject.BUFFER_LIMIT_BYTES,))
        with self.assertRaisesRegex(subject.ZipBoundError, "64-MiB domain"):
            subject.torch_save_zip_ceiling(subject.BUFFER_LIMIT_BYTES - 1, ())

    def test_small_buffer_and_malformed_types_fail_closed(self) -> None:
        bound = subject.torch_save_zip_ceiling(10, ())
        with self.assertRaisesRegex(subject.ZipBoundError, "buffer limit"):
            subject.torch_save_zip_ceiling(10, (), buffer_limit_bytes=1260)
        for pickle_size in (True, 0, -1, 1.0):
            with self.assertRaises(subject.ZipBoundError):
                subject.torch_save_zip_ceiling(pickle_size, ())
        with self.assertRaisesRegex(subject.ZipBoundError, "must be a tuple"):
            subject.torch_save_zip_ceiling(1, [4])
        with self.assertRaisesRegex(subject.ZipBoundError, "record count"):
            subject.torch_save_zip_ceiling(1, (object(),) * 65529)
        for storages in ((True,), (-1,), (1.0,)):
            with self.assertRaises(subject.ZipBoundError):
                subject.torch_save_zip_ceiling(1, storages)
        malformed = copy.deepcopy(bound)
        malformed["record_count"] = True
        with self.assertRaises(subject.ZipBoundError):
            subject.validate_bound(malformed)
        reordered = {key: bound[key] for key in reversed(bound)}
        with self.assertRaises(subject.ZipBoundError):
            subject.validate_bound(reordered)

    def test_runtime_admission_is_cpu_only_and_disclaims_binary_attestation(self) -> None:
        self.assertEqual(os.environ.get("CUDA_VISIBLE_DEVICES"), "")
        result = subject.validate_runtime_save_configuration()
        import torch

        self.assertEqual(torch.get_num_threads(), 1)
        self.assertFalse(torch.cuda.is_initialized())
        self.assertTrue(result["package_registry_verified"])
        self.assertTrue(result["source_revision_and_configuration_admitted"])
        self.assertFalse(result["compiled_binary_provenance_attested"])
        self.assertTrue(result["save_call_defaults_verified"])
        self.assertTrue(result["require_cuda_uninitialized"])
        self.assertFalse(result["cuda_initialized"])
        self.assertTrue(result["cuda_initialization_unchanged"])
        self.assertEqual(subject.required_torch_save_kwargs(), {
            "pickle_protocol": 2,
            "_use_new_zipfile_serialization": True,
            "_disable_byteorder_record": False,
        })

    def test_mutable_runtime_registry_and_flags_are_rejected(self) -> None:
        import torch
        import torch.serialization as serialization
        from torch.utils.serialization import config

        with mock.patch.object(serialization, "_package_registry", list(serialization._package_registry) + [(1, lambda _: "cpu", lambda *_: None)]):
            with self.assertRaisesRegex(subject.ZipBoundError, "package registry"):
                subject.validate_runtime_save_configuration()
        with mock.patch.object(serialization, "location_tag", return_value="changed"):
            with self.assertRaisesRegex(subject.ZipBoundError, "location tag"):
                subject.validate_runtime_save_configuration()
        altered = list(serialization._package_registry)
        priority, tagger, deserializer = altered[1]
        altered[1] = (
            priority,
            functools.partial(tagger.func, *tagger.args, unexpected=True),
            deserializer,
        )
        with mock.patch.object(serialization, "_package_registry", altered):
            with self.assertRaisesRegex(subject.ZipBoundError, "package registry"):
                subject.validate_runtime_save_configuration()
        altered = list(serialization._package_registry)
        priority, tagger, deserializer = altered[0]
        altered[0] = (float(priority), tagger, deserializer)
        with mock.patch.object(serialization, "_package_registry", altered):
            with self.assertRaisesRegex(subject.ZipBoundError, "priority"):
                subject.validate_runtime_save_configuration()
        with mock.patch.object(config.save, "compute_crc32", False):
            with self.assertRaisesRegex(subject.ZipBoundError, "CRC"):
                subject.validate_runtime_save_configuration()
        runtime = subject.validate_runtime_save_configuration()
        malformed = dict(runtime)
        malformed["compiled_binary_provenance_attested"] = 0
        with self.assertRaises(subject.ZipBoundError):
            subject.validate_runtime_record(malformed)
        with mock.patch.object(torch.cuda, "is_initialized", side_effect=(True, True)):
            initialized = subject.validate_runtime_save_configuration(
                require_cuda_uninitialized=False
            )
        self.assertTrue(initialized["cuda_initialized"])
        self.assertFalse(initialized["require_cuda_uninitialized"])
        with mock.patch.object(torch.cuda, "is_initialized", side_effect=(False, True)):
            with self.assertRaisesRegex(subject.ZipBoundError, "state changed"):
                subject.validate_runtime_save_configuration(
                    require_cuda_uninitialized=False
                )
        with self.assertRaisesRegex(subject.ZipBoundError, "must be a boolean"):
            subject.validate_runtime_save_configuration(require_cuda_uninitialized=0)


if __name__ == "__main__":
    unittest.main()
