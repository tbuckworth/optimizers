"""Symbolic branch/audit recipe checks; no materialization or producer imports."""
from __future__ import annotations

import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import audit_primitive_bound as audit_bound
import branch_primitive_bound as branch_bound
import native_storage_recipe_branch_audit as recipe
import native_storage_recipe_common as common
import native_tensor_inventory as inventory


BIG_TMP = Path("/tmp/spectral-experiment-artifacts")
THREAD_ENV = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "NUMEXPR_NUM_THREADS")


def sample_inputs():
    sources = {"schema": "i7_source_binding_v1", "files": [
        {"path": "alpha.py", "size": 123, "sha256": "0" * 64},
        {"path": "beta.py", "size": 456, "sha256": "1" * 64},
    ], "repository_root_realpath": "/diagnostic/repository"}
    environment = {
        "schema": "i7_environment_binding_v1", "profile": common.PROFILE,
        "runtime_role": "storage_crosscheck_cpu",
        "repository_root_realpath": "/diagnostic/repository",
        "python": {"implementation": "CPython", "version": [3, 12, 3]},
        "cuda": {"initialized": False, "devices": []},
        "safe_environment": {"CUDA_VISIBLE_DEVICES": ""},
    }
    return sources, environment


def walk(value):
    yield value
    if type(value) is dict:
        for child in value.values():
            yield from walk(child)
    elif type(value) in (list, tuple):
        for child in value:
            yield from walk(child)


def tree_stats(value):
    nodes = strings = maximum_depth = 0
    stack = [(value, 0, False)]
    while stack:
        item, depth, key_node = stack.pop()
        nodes += 1
        maximum_depth = max(maximum_depth, depth)
        if type(item) is str:
            strings += len(item.encode("utf-8", "surrogatepass"))
        if type(item) is dict:
            for key in reversed(tuple(item)):
                stack.append((item[key], depth + 1, False))
                stack.append((key, depth + 1, True))
        elif type(item) in (list, tuple):
            stack.extend((child, depth + 1, False) for child in reversed(item))
    return nodes, maximum_depth, strings


class BranchAuditRecipeTests(unittest.TestCase):
    def setUp(self):
        self.started = time.monotonic()
        self.assertEqual(os.environ.get("CUDA_VISIBLE_DEVICES"), "")
        self.assertTrue(all(os.environ.get(name) == "1" for name in THREAD_ENV))
        self.assertEqual(Path(os.environ["TMPDIR"]).resolve(), BIG_TMP.resolve())

    def tearDown(self):
        self.assertLess(time.monotonic() - self.started, 120.0)
        self.assertLess(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024, 2 << 30)

    def test_branch_full_mirror_has_exact_inventory_and_calculator_cost(self):
        sources, environment = sample_inputs()
        tree = recipe.build_template("branch_results", sources=sources,
                                     environment=environment)
        self.assertEqual(tuple(tree), ("schema_name", "schema_version", "profile", "artifact_id",
                                      "created_utc", "identity", "payload",
                                      "payload_tensor_bytes"))
        self.assertEqual(tree["profile"], common.PROFILE)
        self.assertEqual(tree["payload_tensor_bytes"], recipe.BRANCH_RAW_BYTES)
        self.assertEqual(common.primitive_cost(tree["payload"]), 423_786)
        paths = common.slot_paths(tree)
        rows = inventory.component_layouts()["branch_results"]
        self.assertEqual(len(paths), 132)
        self.assertEqual(set(paths), {row["path"] for row in rows})
        self.assertEqual(sum(row["nbytes"] for row in rows), recipe.BRANCH_RAW_BYTES)
        report = common.validate_template("branch_results", tree)
        self.assertEqual(report["slot_count"], 132)
        self.assertEqual(report["raw_storage_bytes"], recipe.BRANCH_RAW_BYTES)

    def test_all_branch_domain_helpers_match_each_independent_golden_cost(self):
        expected = branch_bound.compute_branch_payload_bound()["variants"]
        for variant in ("all_defined", "restored_undefined", "reciprocal_undefined"):
            payload = recipe._branch_payload(variant)
            self.assertEqual(common.primitive_cost(payload),
                             expected[variant]["payload_primitive_pickle_bytes_upper"])
        all_defined = recipe._branch_payload("all_defined")
        zeros = all_defined["candidate_state"]["zero_cases"]
        self.assertIs(zeros["current_zero"], True)
        self.assertIs(zeros["lagged_zero"], True)
        restored = recipe._branch_payload("restored_undefined")
        self.assertEqual(restored["branches"]["restored"]["status"], "undefined")
        self.assertEqual(restored["comparisons"]["branch_pairs"]["raw__restored"]
                         ["defined_mask"], {"raw": True, "restored": False})

    def test_audit_adverse_union_has_every_count_bitset_and_no_slot(self):
        sources, environment = sample_inputs()
        tree = recipe.build_template("independent_audit_all_failure", sources=sources,
                                     environment=environment)
        self.assertEqual(tree["payload_tensor_bytes"], 0)
        self.assertEqual(common.slot_paths(tree), ())
        payload = tree["payload"]
        report = payload["measurement_audits"]["result"]
        self.assertEqual(len(report["measurement_audits"]), 4_010)
        self.assertEqual(len(payload["comparison_audits"]), 441)
        self.assertEqual(len(report["fatal_failures"]), 5_082)
        self.assertEqual(len(report["native_discordant_paths"]), 392)
        self.assertEqual(len(report["unresolved_geometry_paths"]), 51)
        bitsets = [item for item in walk(payload)
                   if type(item) is dict and tuple(item) ==
                   ("encoding", "dimension", "count", "bits_hex")]
        self.assertEqual(len(bitsets), 92)
        self.assertEqual({row["dimension"] for row in bitsets}, {P for P in (50_890, 50_176, 64, 640, 10)})
        for row in bitsets:
            self.assertEqual(len(row["bits_hex"]), 2 * ((row["dimension"] + 7) // 8))
            self.assertEqual(row["count"], row["dimension"])
        validated = common.validate_template("independent_audit_all_failure", tree)
        self.assertEqual(validated["slot_count"], 0)
        self.assertEqual(validated["raw_storage_bytes"], 0)
        nodes, depth, strings = tree_stats(tree)
        self.assertLess(nodes, common.MAX_NODES)
        self.assertLessEqual(depth, common.MAX_DEPTH)
        self.assertLess(strings, common.MAX_TOTAL_STRING_BYTES)
        self.assertGreater(strings, recipe.PARTIAL_VALIDATION_UTF8)

    def test_audit_concrete_cost_is_dominated_only_by_environment_lemma_slack(self):
        sources, environment = sample_inputs()
        environment_copy, json_bytes = recipe._bounded_json_copy(
            environment, recipe.ENVIRONMENT_JSON_MAX, "environment")
        payload, topology = recipe._audit_payload(environment_copy, "both_positive")
        bound = audit_bound.audit_payload_subtree_cost(
            role="sensitivity", bundle=71901, update=2_000,
            auditor_environment_json_bytes=json_bytes)
        actual = common.primitive_cost(payload)
        self.assertLessEqual(actual, bound)
        self.assertEqual(bound - actual,
                         5 * json_bytes - common.primitive_cost(environment_copy))
        self.assertEqual(topology, {"measurement_paths": 4_010,
                                   "comparison_paths": 441, "failure_rows": 5_082,
                                   "native_paths": 392, "unresolved_paths": 51})

    def test_audit_domain_and_outer_stage_alternatives_remain_explicit(self):
        _, environment = sample_inputs()
        environment, _ = recipe._bounded_json_copy(environment, 8192, "environment")
        expected = {
            "both_positive": (4_010, 441, 5_082),
            "both_zero": (4_010, 441, 5_082),
            "restored_undefined": (2_835, 295, 3_551),
            "reciprocal_undefined": (2_835, 295, 3_551),
        }
        for domain, counts in expected.items():
            _, topology = recipe._audit_payload(environment, domain)
            self.assertEqual((topology["measurement_paths"], topology["comparison_paths"],
                              topology["failure_rows"]), counts)
        marker = {"kind": "saved_value_consistency_not_history"}
        self.assertEqual(recipe._stage_row(marker, "not_run"),
                         {"status": "not_run", "completed": False,
                          "error_code": None, "result": None})
        self.assertEqual(recipe._stage_row(marker, "resource_abort")["error_code"],
                         "ResourceGuardAbort")
        self.assertEqual(recipe._stage_row(marker, "error")["error_code"],
                         "ArtifactEnvelopeError")
        self.assertTrue(recipe._stage_row(marker, "completed_pass")["completed"])

    def test_synthetic_union_is_named_and_not_misrepresented_as_reachable(self):
        sources, environment = sample_inputs()
        payload = recipe.build_template("independent_audit_all_failure", sources=sources,
                                        environment=environment)["payload"]
        screen = payload["candidate_operator_audit"]["projections"]["result"]["current"]
        self.assertIsInstance(screen["max_error_envelope_ratio"], float)
        self.assertEqual(screen["ratio_null_reason"], "zero_envelope_mismatch")
        self.assertEqual(payload["exact_validation"]["completion"]["completed_stages"],
                         list(recipe.CHECK_KEYS))
        self.assertEqual(payload["fatal_failures"], [
            {"stage": "saved_structure", "completed": False,
             "error_code": "ArtifactEnvelopeError"}])
        self.assertEqual(payload["candidate_operator_audit"]["delivery"]["result"]
                         ["branches"]["restored"]["status"], "domain_undefined")
        self.assertEqual(len(payload["measurement_audits"]["result"]
                             ["domain_undefined_branches"]), 0)

    def test_supplied_environment_is_owned_full_tree_and_caps_fail_closed(self):
        sources, environment = sample_inputs()
        tree = recipe.build_template("independent_audit_all_failure", sources=sources,
                                     environment=environment)
        retained = tree["payload"]["exact_validation"]["auditor_environment"]
        self.assertEqual(retained, environment)
        self.assertIsNot(retained, environment)
        self.assertIsNot(retained["python"], environment["python"])
        environment["python"]["implementation"] = "changed"
        self.assertEqual(retained["python"]["implementation"], "CPython")
        for bad_sources, bad_environment in (
                ({"x": "s" * (recipe.SOURCE_JSON_MAX + 1)}, sample_inputs()[1]),
                (sample_inputs()[0], {"x": "e" * recipe.ENVIRONMENT_JSON_MAX}),
                ({"x": (1, 2)}, sample_inputs()[1]),
                (sample_inputs()[0], {1: "bad key"})):
            with self.assertRaises(recipe.StorageRecipeBranchAuditError):
                recipe.build_template("branch_results", sources=bad_sources,
                                      environment=bad_environment)

    def test_import_default_and_both_builds_are_torch_free(self):
        code = (
            f"import sys;sys.path.insert(0,{str(HERE)!r});"
            "import native_storage_recipe_branch_audit as r;"
            "s={'x':1};e={'runtime_role':'storage_crosscheck_cpu'};"
            "r.build_template('branch_results',sources=s,environment=e);"
            "r.build_template('independent_audit_all_failure',sources=s,environment=e);"
            "assert 'torch' not in sys.modules;assert r.main([])==0"
        )
        run = subprocess.run([sys.executable, "-I", "-B", "-c", code],
                             cwd=HERE, capture_output=True, timeout=30)
        self.assertEqual(run.returncode, 0, run.stderr.decode())
        self.assertEqual(run.stdout.decode(),
                         "I7 symbolic branch/audit storage recipes only; no tensors or execution.\n")


if __name__ == "__main__":
    unittest.main()
