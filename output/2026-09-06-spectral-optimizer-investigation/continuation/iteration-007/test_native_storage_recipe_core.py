#!/usr/bin/env python3
"""Bounded symbolic tests for the nine core/history storage recipes."""
from __future__ import annotations

import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import unittest
import json

if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
    raise RuntimeError("recipe-core tests require hidden CUDA")

import native_storage_recipe_common as common
import native_storage_recipe_core as recipe


I7 = Path(__file__).resolve().parent
BIG_TMP = Path("/tmp/spectral-experiment-artifacts")
THREADS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")


class NativeStorageRecipeCoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.started = time.monotonic()
        cls.rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        if not BIG_TMP.is_dir() or Path(os.environ["TMPDIR"]).resolve() != BIG_TMP.resolve():
            raise RuntimeError("tests require verified big/tmp")
        if any(os.environ.get(key) != "1" for key in THREADS):
            raise RuntimeError("tests require four single-thread settings")
        if "torch" in sys.modules:
            raise RuntimeError("symbolic recipe tests must start without Torch")

    @classmethod
    def tearDownClass(cls) -> None:
        assert "torch" not in sys.modules
        assert time.monotonic() - cls.started < 120.0
        assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 < 2 << 30

    @staticmethod
    def metadata():
        return ({"schema": "source", "files": [{"path": "a", "sha256": "1" * 64}]},
                {"schema": "environment", "runtime_role": "storage_crosscheck_cpu"})

    def test_component_membership_and_public_helpers(self) -> None:
        self.assertEqual(recipe.COMPONENTS, (
            "anchor_pilot", "anchor_long", "source_witness",
            "source_completion_pilot_on", "source_completion_pilot_off",
            "source_completion_long_on", "plan_pilot", "plan_long",
            "capture_comparison_pilot"))
        self.assertEqual(tuple(recipe.native_tensor_metadata((2, 3))),
                         ("value", "shape", "native_dtype", "native_device"))
        self.assertIs(recipe.native_tensor_metadata((2,))["value"], common.SLOT)
        self.assertEqual(len(recipe.parameter_rows()), 4)
        self.assertEqual(tuple(recipe.optimizer()),
                         ("class_name", "state_completed_updates", "parameter_order",
                          "param_groups", "state"))
        self.assertEqual(tuple(recipe.observer()),
                         ("class_name", "state_completed_observations",
                          "excluded_live_aliases", "config", "state"))
        for kind, expected in recipe.PAYLOAD_TENSOR_BYTES.items():
            with self.subTest(kind=kind):
                wrapped = recipe.envelope({}, role=("pilot" if kind != "independent-audit"
                                                     else "primary"),
                                          bundle=(71990 if kind != "independent-audit" else 71001),
                                          update=101, kind=kind)
                self.assertEqual(wrapped["payload_tensor_bytes"], expected)

    def test_all_nine_slot_memberships_match_maximum_inventory(self) -> None:
        import native_tensor_inventory as inventory
        sources, environment = self.metadata()
        layouts = inventory.component_layouts()
        for component in recipe.COMPONENTS:
            with self.subTest(component=component):
                tree = recipe.build_template(component, sources=sources,
                                             environment=environment)
                actual = common.slot_paths(tree)
                expected = tuple(row["path"] for row in layouts[component])
                self.assertEqual(len(actual), len(expected))
                self.assertEqual(set(actual), set(expected))
                validated = common.validate_template(component, tree)
                self.assertEqual(validated["slot_count"], len(expected))
                self.assertIs(validated["scientific_artifact"], False)
                self.assertIs(validated["execution_authorized"], False)
                if component.startswith("anchor_"):
                    self.assertEqual(tree["payload_tensor_bytes"], 7_378_608)
                elif component == "source_witness":
                    self.assertEqual(tree["payload_tensor_bytes"], 7_735_552)

    def test_primitive_costs_are_bounded_by_independent_calculators(self) -> None:
        import core_primitive_bound as bounds
        sources, environment = self.metadata()
        expected = bounds.core_component_bounds()
        for component in recipe.COMPONENTS[:-1]:
            with self.subTest(component=component):
                tree = recipe.build_template(component, sources=sources,
                                             environment=environment)
                self.assertLessEqual(common.primitive_cost(tree), expected[component])

    def test_anchor_embeds_distinct_owned_metadata_copies(self) -> None:
        sources, environment = self.metadata()
        first = recipe.build_template("anchor_pilot", sources=sources,
                                      environment=environment)
        second = recipe.build_template("anchor_pilot", sources=sources,
                                       environment=environment)
        left = first["payload"]["bindings"]
        right = second["payload"]["bindings"]
        self.assertEqual(left["sources"], sources)
        self.assertEqual(left["environment"], environment)
        self.assertIsNot(left["sources"], sources)
        self.assertIsNot(left["environment"], environment)
        self.assertIsNot(left["sources"], right["sources"])
        self.assertIsNot(left["environment"], right["environment"])

    def test_profiles_and_name_prefixes_are_unregistered_diagnostic_values(self) -> None:
        sources, environment = self.metadata()
        self.assertEqual(len(common.PROFILE), len("scientific_mnist_current32_v1"))
        for component in recipe.COMPONENTS:
            tree = recipe.build_template(component, sources=sources,
                                         environment=environment)
            self.assertEqual(tree["profile"], common.PROFILE)
            if "artifact_id" in tree:
                if tree["artifact_id"].startswith(common.PROFILE):
                    self.assertEqual(len(common.PROFILE), 29)
                else:
                    self.assertTrue(tree["artifact_id"].startswith(recipe.NATIVE_PREFIX + "-"))
        self.assertEqual(len(recipe.NATIVE_PREFIX), len("i7-native"))

    def test_source_completion_counts_modes_and_full_final_core(self) -> None:
        sources, environment = self.metadata()
        on = recipe.build_template("source_completion_pilot_on", sources=sources,
                                   environment=environment)
        off = recipe.build_template("source_completion_pilot_off", sources=sources,
                                    environment=environment)
        long = recipe.build_template("source_completion_long_on", sources=sources,
                                     environment=environment)
        self.assertEqual((len(on["trace"]), len(off["trace"]), len(long["trace"])),
                         (220, 220, 2_000))
        self.assertEqual((len(on["anchor_witness_refs"]), len(off["anchor_witness_refs"]),
                          len(long["anchor_witness_refs"])), (2, 0, 4))
        self.assertEqual(long["trace"][-1]["completed_updates"], 2_001)
        self.assertEqual(long["final_state_core"]["state_completed_updates"], 2_001)
        self.assertIs(long["final_state_core"]["rng"]["torch_cuda"][0]["state"], common.SLOT)

    def test_comparison_has_exact_order_and_220_complete_rows(self) -> None:
        sources, environment = self.metadata()
        tree = recipe.build_template("capture_comparison_pilot", sources=sources,
                                     environment=environment)
        self.assertEqual(tuple(tree), (
            "schema_name", "schema_version", "profile", "artifact_id", "artifact_name",
            "trajectory", "instrumentation", "initial_comparison", "step_trace",
            "summary", "evidence_scope", "scientific_execution_certified"))
        self.assertEqual(len(tree["step_trace"]), 220)
        self.assertEqual(tree["step_trace"][0]["completed_updates"], 1)
        self.assertEqual(tree["step_trace"][-1]["completed_updates"], 220)
        self.assertEqual(set(common.slot_paths(tree)), set())
        encoded = (json.dumps(tree, ensure_ascii=True, allow_nan=False,
                              separators=(",", ":")) + "\n").encode("ascii")
        import native_storage_topology_bound as topology
        self.assertLessEqual(len(encoded), topology.JSON_COMPARISON_BYTES)

    def test_bad_component_and_bad_helper_domains_fail_closed(self) -> None:
        sources, environment = self.metadata()
        with self.assertRaisesRegex(recipe.StorageRecipeCoreError, "unknown core"):
            recipe.build_template("branch_results", sources=sources, environment=environment)
        with self.assertRaisesRegex(recipe.StorageRecipeCoreError, "shape"):
            recipe.native_tensor_metadata([1])
        with self.assertRaisesRegex(recipe.StorageRecipeCoreError, "counter"):
            recipe.optimizer(completed_updates=True)

    def test_import_default_cli_and_build_remain_torch_free(self) -> None:
        env = {key: value for key, value in os.environ.items()
               if key not in ("PYTHONPATH", "PYTHONHOME")}
        code = ("import sys,native_storage_recipe_core as r;"
                "s={'x':[1]};e={'y':[2]};r.build_template('plan_pilot',sources=s,environment=e);"
                "print('torch' in sys.modules,r.PROFILE)")
        check = subprocess.run([sys.executable, "-c", code], cwd=I7, env=env,
                               check=True, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, timeout=10)
        self.assertEqual(check.stdout, "False diagnostic_mnist_current32_v1\n")
        cli = subprocess.run([sys.executable, str(I7 / "native_storage_recipe_core.py")],
                             cwd=I7, env=env, check=True, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, timeout=10)
        self.assertEqual(cli.stdout,
                         "I7 symbolic core storage recipes only; no tensors, serialization, data or execution.\n")


if __name__ == "__main__":
    unittest.main()
