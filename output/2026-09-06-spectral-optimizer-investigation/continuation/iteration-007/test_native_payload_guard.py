#!/usr/bin/env python3
"""Bounded CPU-only tests for the actual-payload storage preflight."""
from __future__ import annotations

from collections import Counter
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import unittest
from unittest import mock

if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
    raise RuntimeError("payload-guard tests require CUDA_VISIBLE_DEVICES='' ")

import torch

import native_payload_guard as guard


I7 = Path(__file__).resolve().parent
BIG_TMP = Path("/tmp/spectral-experiment-artifacts")
THREAD_ENV = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")


class NativePayloadGuardTests(unittest.TestCase):
    def setUp(self) -> None:
        self.started = time.monotonic()
        self.assertTrue(all(os.environ.get(name) == "1" for name in THREAD_ENV))
        self.assertEqual(Path(os.environ["TMPDIR"]).resolve(), BIG_TMP.resolve())
        self.assertTrue(BIG_TMP.is_dir())
        self.assertEqual(torch.get_num_threads(), 1)
        self.assertFalse(torch.cuda.is_initialized())

    def tearDown(self) -> None:
        self.assertFalse(torch.cuda.is_initialized())
        self.assertLess(time.monotonic() - self.started, 120.0)
        self.assertLess(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024, 2 << 30)

    def test_fixed_schedule_has_exact_component_membership(self) -> None:
        rows = guard._schedule()
        self.assertEqual(len(rows), 82)
        self.assertEqual(len({row.name for row in rows}), 82)
        self.assertEqual(Counter(row.component for row in rows), Counter({
            "anchor_pilot": 2, "anchor_long": 16, "source_witness": 18,
            "branch_results": 18, "independent_audit_all_failure": 16,
            "source_completion_pilot_on": 1, "source_completion_pilot_off": 1,
            "capture_comparison_pilot": 1, "source_completion_long_on": 4,
            "plan_pilot": 1, "plan_long": 4,
        }))
        self.assertEqual(guard.scheduled_component("i7-native-b71990-plan.pt"), "plan_pilot")
        with self.assertRaisesRegex(guard.PayloadAdmissionError, "fixed82-member"):
            guard.scheduled_component("i7-native-b71991-plan.pt")

    def test_reviewed_analytic_ceilings_are_recomputed_once_and_immutable(self) -> None:
        guard._fixed_admission.cache_clear()
        first = guard._fixed_admission()
        second = guard._fixed_admission()
        self.assertIs(first, second)
        self.assertEqual(len(first), 82)
        by_name = {name: (ceiling, tensors) for name, ceiling, tensors in first}
        self.assertEqual(by_name["i7-native-b71990-plan.pt"], (799_641, 8))
        branch = next(row for row in guard._schedule() if row.component == "branch_results")
        self.assertEqual(by_name[branch.name], (33_452_716, 132))

    def test_tensor_paths_require_exact_shape_dtype_and_canonical_stride(self) -> None:
        import native_tensor_inventory as inventory

        row = {"path": ("x",), "dtype": "float32", "shape": (1, 2), "nbytes": 8}
        with mock.patch.object(inventory, "component_layouts",
                               return_value={"plan_pilot": (row,)}):
            valid = {"x": torch.zeros((1, 2), dtype=torch.float32)}
            found, expected = guard._validate_tensor_layout(valid, "plan_pilot")
            self.assertEqual(tuple(found), (("x",),))
            self.assertEqual(expected, (row,))

            noncanonical = {"x": torch.empty_strided((1, 2), (9, 1), dtype=torch.float32)}
            self.assertTrue(noncanonical["x"].is_contiguous())
            with self.assertRaisesRegex(guard.PayloadAdmissionError, "canonical stride"):
                guard._validate_tensor_layout(noncanonical, "plan_pilot")

            wrong = {"x": torch.zeros((1, 2), dtype=torch.float64)}
            with self.assertRaisesRegex(guard.PayloadAdmissionError, "dtype differs"):
                guard._validate_tensor_layout(wrong, "plan_pilot")

    def _patched_admit(self, tree, *, ceiling=100_000, tensor_count=1):
        tensor = next(value for value in tree.values() if type(value) is torch.Tensor)
        found = {(key,): value for key, value in tree.items() if type(value) is torch.Tensor}
        expected = tuple({"path": path, "dtype": "float32",
                          "shape": tuple(value.shape), "nbytes": value.numel() * 4}
                         for path, value in found.items())
        patches = (
            mock.patch.object(guard, "_source_environment", return_value=("a" * 64, "b" * 64)),
            mock.patch.object(guard, "_metadata_binding"),
            mock.patch.object(guard, "_counter_bindings"),
            mock.patch.object(guard, "_validate_tensor_layout", return_value=(found, expected)),
            mock.patch.object(guard, "_rng_metadata"),
            mock.patch.object(guard, "_ceiling", return_value=(ceiling, tensor_count)),
        )
        return tensor, patches

    def test_actual_protocol2_and_zip_plumbing_returns_bounded_integer_summary(self) -> None:
        tree = {"x": torch.arange(4, dtype=torch.float32)}
        _, patches = self._patched_admit(tree)
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
            result = guard.admit_torch_payload(
                "i7-native-b71990-plan.pt", tree, sources={}, environment={})
        self.assertEqual(tuple(result), guard.SUMMARY_FIELDS)
        self.assertEqual(result["component"], "plan_pilot")
        self.assertEqual(result["storage_count"], 1)
        self.assertEqual(result["raw_storage_bytes"], 16)
        self.assertGreater(result["pickle_bytes_upper"], 0)
        self.assertGreater(result["zip_bytes_upper"], result["pickle_bytes_upper"])
        self.assertIs(result["component_bound_satisfied"], True)
        self.assertEqual(result["semantic_validation_external"],
                         "required_immediately_before_guard_not_attested")
        self.assertIs(result["storage_fit_proven"], False)
        self.assertIs(result["execution_authorized"], False)

    def test_repeated_tensor_identity_and_component_overflow_fail_closed(self) -> None:
        shared = torch.arange(4, dtype=torch.float32)
        tree = {"a": shared, "b": shared}
        _, patches = self._patched_admit(tree, tensor_count=2)
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
            with self.assertRaisesRegex(guard.PayloadAdmissionError, "protocol2 tree"):
                guard.admit_torch_payload(
                    "i7-native-b71990-plan.pt", tree, sources={}, environment={})

        one = {"x": torch.arange(4, dtype=torch.float32)}
        _, patches = self._patched_admit(one, ceiling=100)
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
            with self.assertRaisesRegex(guard.PayloadAdmissionError, "component ceiling"):
                guard.admit_torch_payload(
                    "i7-native-b71990-plan.pt", one, sources={}, environment={})

    def test_branch_domains_and_counters_are_exactly_bounded(self) -> None:
        spec = next(row for row in guard._schedule() if row.kind == "branch-results")
        rows = {name: {"status": "defined", "reason": None,
                       "optimizer_after": {"state_completed_updates": spec.update}}
                for name in ("raw", "current", "lagged", "restored", "reciprocal", "zero")}
        tree = {"payload": {"candidate_state": {"observer_after": {
            "state_completed_observations": spec.update,
            "state": {"step_count": spec.update, "stabilization_count": 2 * spec.update},
        }}, "branches": rows}}
        guard._counter_bindings(tree, spec)
        rows["restored"].update(status="undefined",
                                reason="positive_current_norm_zero_lagged_direction")
        guard._counter_bindings(tree, spec)
        rows["restored"]["reason"] = "arbitrary"
        with self.assertRaisesRegex(guard.PayloadAdmissionError, "domain reason"):
            guard._counter_bindings(tree, spec)
        rows["restored"]["reason"] = "positive_current_norm_zero_lagged_direction"
        tree["payload"]["candidate_state"]["observer_after"]["state"][
            "stabilization_count"] = 2 * spec.update + 1
        with self.assertRaisesRegex(guard.PayloadAdmissionError, "stabilization"):
            guard._counter_bindings(tree, spec)

    def test_anchor_counter_binding_uses_actual_anchor_payload_shape(self) -> None:
        spec = next(row for row in guard._schedule() if row.kind == "anchor")
        completed = spec.update - 1
        tree = {"payload": {
            "model": {},
            "optimizer": {"state_completed_updates": completed},
            "observer": {
                "state_completed_observations": completed,
                "state": {"step_count": completed, "stabilization_count": 2 * completed},
            },
            "rng": {},
            "bindings": {},
        }}
        guard._counter_bindings(tree, spec)
        tree["payload"]["optimizer"]["state_completed_updates"] = completed + 1
        with self.assertRaisesRegex(guard.PayloadAdmissionError, "anchor.optimizer differs"):
            guard._counter_bindings(tree, spec)

    def test_metadata_caps_and_comparison_json_encoder_is_not_store_metadata_encoder(self) -> None:
        import artifact_store as storage
        import identity_codec as codec

        value = {"z": [1, True, None, 2.5, "ascii"], "a": 1}
        self.assertNotEqual(codec.json_bytes(value), storage._json_bytes(value))
        guard._bounded_primitive(value, max_nodes=20, max_depth=4, max_string_bytes=8)
        with self.assertRaisesRegex(guard.PayloadAdmissionError, "string exceeds cap"):
            guard._bounded_primitive("x" * 8_193, max_nodes=2, max_depth=1,
                                     max_string_bytes=8_192)
        with self.assertRaisesRegex(guard.PayloadAdmissionError, "source/environment schema"):
            guard.validate_runtime_metadata({}, {})

    def test_source_and_environment_roots_are_cross_bound(self) -> None:
        sources = {"repository_root_realpath": "/fixed/source"}
        environment = {"repository_root_realpath": "/different/source",
                       "runtime_role": "native_source"}
        with mock.patch("source_environment_schema.validate_sources"), \
             mock.patch("source_environment_schema.validate_environment"):
            with self.assertRaisesRegex(guard.PayloadAdmissionError, "repository roots differ"):
                guard._source_environment(sources, environment, audit=False)

    def test_json_path_is_bounded_before_schema_and_encoder_calls(self) -> None:
        value = {"profile": guard.PROFILE,
                 "artifact_name": "i7-native-pilot-b71990-capture-comparison.json"}
        with mock.patch.object(guard, "_source_environment", return_value=("a" * 64, "b" * 64)), \
             mock.patch("source_history.validate_capture_comparison", return_value=value) as validate, \
             mock.patch("identity_codec.json_bytes", return_value=b"{}\n") as identity, \
             mock.patch.object(guard, "_ceiling", return_value=(308_940, 0)):
            result = guard.admit_json_payload(value["artifact_name"], value,
                                              sources={}, environment={})
        validate.assert_called_once_with(value)
        identity.assert_called_once_with(value)
        self.assertEqual(result["json_bytes"], 3)
        self.assertIsNone(result["zip_bytes_upper"])

    def test_import_and_default_cli_are_torch_free_and_inert(self) -> None:
        env = {key: value for key, value in os.environ.items()
               if key not in ("PYTHONPATH", "PYTHONHOME")}
        check = subprocess.run(
            [sys.executable, "-c",
             "import sys,native_payload_guard as g; print('torch' in sys.modules, g.SCHEMA)"],
            cwd=I7, env=env, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, timeout=10)
        self.assertEqual(check.stdout, "False i7_actual_payload_storage_preflight_v1\n")
        cli = subprocess.run(
            [sys.executable, str(I7 / "native_payload_guard.py")], cwd=I7, env=env,
            check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=10)
        self.assertEqual(cli.stdout,
                         "I7 actual-payload storage guard is library-only; no payload, write, CUDA or execution.\n")
        self.assertLess(len(cli.stdout.encode()), 256)


if __name__ == "__main__":
    unittest.main()
