#!/usr/bin/env python3
"""Small CPU-only tests for the branch payload primitive arithmetic."""
from __future__ import annotations

import copy
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import unittest
from unittest import mock

import branch_primitive_bound as bound


I7 = Path(__file__).resolve().parent
REPO = I7.parents[3]
BIG_TMP = Path("/tmp/spectral-experiment-artifacts")
THREAD_ENV = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")


class BranchPrimitiveBoundTests(unittest.TestCase):
    def setUp(self) -> None:
        self.started = time.monotonic()
        self.rss_started = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        self.assertEqual(os.environ.get("CUDA_VISIBLE_DEVICES"), "")
        self.assertTrue(all(os.environ.get(name) == "1" for name in THREAD_ENV))
        self.assertEqual(Path(os.environ["TMPDIR"]).resolve(), BIG_TMP.resolve())
        self.assertTrue(BIG_TMP.is_dir())

    def tearDown(self) -> None:
        self.assertLess(time.monotonic() - self.started, 120.0)
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        self.assertLess(max(self.rss_started, rss), 2 << 30)

    def test_closed_variant_counts_and_golden_costs(self) -> None:
        result = bound.compute_branch_payload_bound()
        rows = result["variants"]
        self.assertEqual(tuple(rows), bound.VARIANTS)
        self.assertEqual(
            rows["all_defined"],
            {"undefined_branch": None, "defined_branch_count": 6,
             "undefined_branch_count": 0, "branch_pair_count": 15,
             "defined_branch_pair_count": 15, "undefined_branch_pair_count": 0,
             "contrast_count": 15, "defined_contrast_count": 15,
             "undefined_contrast_count": 0, "auxiliary_chunk_row_count": 220,
             "tensor_count": 132, "payload_primitive_pickle_bytes_upper": 423_786},
        )
        self.assertEqual(rows["restored_undefined"]["payload_primitive_pickle_bytes_upper"],
                         313_184)
        self.assertEqual(rows["reciprocal_undefined"]["payload_primitive_pickle_bytes_upper"],
                         312_845)

    def test_undefined_variants_retain_all_pair_and_contrast_rows(self) -> None:
        result = bound.compute_branch_payload_bound()
        for name in ("restored_undefined", "reciprocal_undefined"):
            row = result["variants"][name]
            self.assertEqual(row["branch_pair_count"], 15)
            self.assertEqual(row["defined_branch_pair_count"], 10)
            self.assertEqual(row["undefined_branch_pair_count"], 5)
            self.assertEqual(row["contrast_count"], 15)
            self.assertEqual(row["defined_contrast_count"], 10)
            self.assertEqual(row["undefined_contrast_count"], 5)
            self.assertEqual(row["auxiliary_chunk_row_count"], 160)
            self.assertEqual(row["tensor_count"], 112)

    def test_eighteen_payload_aggregation_is_payload_only(self) -> None:
        result = bound.compute_branch_payload_bound()
        maxima = result["maxima"]
        self.assertEqual(result["payload_count"], 18)
        self.assertEqual(maxima["payload_primitive_pickle_bytes_upper"], 423_786)
        self.assertEqual(maxima["all_payloads_primitive_pickle_bytes_upper"], 7_628_148)
        self.assertEqual(maxima["tensor_count_per_payload_upper"], 132)
        self.assertEqual(maxima["tensor_count_all_payloads_upper"], 2_376)
        self.assertEqual(result["exclusions"], {
            "common_envelope_and_identity": True, "tensor_pickle_descriptors": True,
            "raw_storage_bytes": True, "zip_container_bytes": True,
            "terminal_failure_metadata": True,
        })

    def test_admission_assumptions_and_authority_flags_are_closed(self) -> None:
        result = bound.compute_branch_payload_bound()
        self.assertEqual(result["assumptions"], {
            "profile": "scientific_mnist_current32_v1", "native_device": "cuda:0",
            "anchor_update_min": 101, "anchor_update_max": 2_000,
            "observer_rank_max": 32, "stabilization_count_max": 4_000,
            "artifact_id_utf8_bytes_max": 128, "artifact_name_utf8_bytes_max": 131,
            "artifact_payload_bytes_max": 1 << 30, "receipt_bytes_max": 4_096,
            "tensor_leaf_primitive_cost": 0, "alias_savings_assumed": False,
        })
        self.assertIs(result["storage_fit_proven"], False)
        self.assertIs(result["execution_authorized"], False)
        self.assertIs(result["scientific_execution_certified"], False)

    def test_decoder_rejects_wrong_exact_type_and_nested_change(self) -> None:
        result = bound.compute_branch_payload_bound()
        wrong_type = copy.deepcopy(result)
        wrong_type["payload_count"] = True
        with self.assertRaisesRegex(ValueError, "closed calculation|exact leaf types"):
            bound.validate_branch_payload_bound(wrong_type)
        changed = copy.deepcopy(result)
        changed["variants"]["restored_undefined"]["undefined_contrast_count"] = 4
        with self.assertRaisesRegex(ValueError, "closed calculation"):
            bound.validate_branch_payload_bound(changed)

    def test_exact_source_manifest_and_drift_rejection(self) -> None:
        self.assertEqual(bound.validate_source_bindings(REPO), bound.SOURCE_SHA256)
        first = next(iter(bound.SOURCE_SHA256))
        with mock.patch.dict(bound.SOURCE_SHA256, {first: "0" * 64}, clear=False):
            with self.assertRaisesRegex(ValueError, "source bytes differ"):
                bound.validate_source_bindings(REPO)

    def test_import_and_default_cli_are_torch_free_and_inert(self) -> None:
        env = {key: value for key, value in os.environ.items()
               if key not in ("PYTHONPATH", "PYTHONHOME")}
        check = subprocess.run(
            [sys.executable, "-c",
             "import sys,branch_primitive_bound as b; print('torch' in sys.modules, b.PAYLOAD_COUNT)"],
            cwd=I7, env=env, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, timeout=10,
        )
        self.assertEqual(check.stdout, "False 18\n")
        cli = subprocess.run(
            [sys.executable, str(I7 / "branch_primitive_bound.py")], cwd=I7, env=env,
            check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=10,
        )
        self.assertEqual(cli.stdout,
                         "I7 branch primitive arithmetic only; no payload, data, model, CUDA or execution.\n")
        self.assertLess(len(cli.stdout.encode("utf-8")), 256)


if __name__ == "__main__":
    unittest.main()
