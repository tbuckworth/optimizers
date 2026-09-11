import contextlib
import io
from pathlib import Path
import subprocess
import sys
import unittest


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import final_storage_accounting as accounting


COMPONENT_SIZES = {
    "anchor_pilot": 7_404_312,
    "anchor_long": 7_404_312,
    "source_witness": 7_747_940,
    "branch_results": 33_169_975,
    "independent_audit_all_failure": 1_214_765,
    "source_completion_pilot_on": 7_458_337,
    "source_completion_pilot_off": 7_457_057,
    "capture_comparison_pilot": 307_621,
    "source_completion_long_on": 8_311_649,
    "plan_pilot": 797_965,
    "plan_long": 1_709_325,
}


class FinalStorageAccountingTests(unittest.TestCase):
    def test_import_and_default_cli_are_inert_and_torch_free(self):
        code = (
            "import sys; import final_storage_accounting as x; "
            "assert 'torch' not in sys.modules; assert x.main([]) == 0; "
            "assert 'torch' not in sys.modules"
        )
        completed = subprocess.run(
            [sys.executable, "-I", "-c",
             f"sys_path={str(HERE)!r}; import sys; sys.path.insert(0,sys_path); {code}"],
            check=False, capture_output=True, text=True, timeout=20)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout,
                         "final_storage_accounting: inert; no filesystem or execution action\n")

    def test_exact_complete_membership(self):
        membership = accounting.expected_membership()
        self.assertEqual(len(membership["scientific_payloads"]), 82)
        self.assertEqual(len(membership["phase_records"]), 90)
        self.assertEqual(membership["phase_records"][0], "native-phase-000.json")
        self.assertEqual(membership["phase_records"][-1], "native-phase-089.json")
        self.assertEqual(len(membership["root_metadata"]), 9)
        self.assertEqual(len(membership["root_receipts"]), 181)
        self.assertEqual(len(membership["root_regular_files"]), 364)
        self.assertEqual(len(set(membership["root_regular_files"])), 364)
        self.assertEqual(len(membership["development_external_success"]), 7)
        self.assertEqual(len(membership["supervision"]), 8)
        self.assertEqual(len(membership["existing_consumed_inspection"]), 2)
        self.assertEqual(membership["existing_consumed_storage_attempt"],
                         ("native-max-layout-measurement.json",))
        self.assertEqual(membership["storage_evidence_prerequisites"],
                         ("native-storage-admission.json",
                          "native-max-layout-measurement-attempt-002.json"))

    def test_unresolved_projection_has_exact_conditional_arithmetic(self):
        result = accounting.project(COMPONENT_SIZES)
        logical = result["logical_bytes"]
        self.assertEqual(logical["measured_cpu_component_projection"], 945_341_202)
        self.assertEqual(logical["scheduled_payload_receipt_ceiling"], 21_126)
        self.assertEqual(logical["phase_record_body_ceiling"], 11_796_480)
        self.assertEqual(logical["phase_record_receipt_ceiling"], 17_640)
        self.assertEqual(logical["root_metadata_body_ceiling"], 5_275_648)
        self.assertEqual(logical["root_metadata_receipt_ceiling"], 1_839)
        self.assertEqual(logical["store_header"], 222)
        self.assertEqual(logical["root_regular_before_native_delta"], 962_454_157)
        self.assertEqual(logical["storage_evidence_prerequisite_ceiling"], 131_072)
        self.assertEqual(logical["outside_root_normal_ceiling"], 287_990)
        self.assertEqual(logical["conditional_total_before_native_delta"], 963_790_723)
        self.assertEqual(logical["residual_before_native_delta"], 109_951_101)
        self.assertIsNone(logical["native_payload_delta_ceiling"])
        self.assertIsNone(logical["bounded_total_with_native_delta"])
        self.assertIsNone(result["arithmetic_fit_under_supplied_delta"])
        self.assertEqual(result["launch_storage_decision"],
                         "unresolved_requires_authenticated_bound_and_frozen_manifest")
        self.assertIn("native_payload_delta_ceiling_unresolved",
                      result["launch_storage_blockers"])

    def test_referenced_delta_produces_only_conditional_arithmetic(self):
        evidence = {"kind": accounting.NATIVE_DELTA_EVIDENCE_KIND,
                    "sha256": "a" * 64}
        result = accounting.project(
            COMPONENT_SIZES, native_payload_delta_ceiling=109_951_101,
            native_payload_delta_evidence=evidence)
        self.assertEqual(result["logical_bytes"]["bounded_total_with_native_delta"],
                         1 << 30)
        self.assertTrue(result["arithmetic_fit_under_supplied_delta"])
        self.assertEqual(result["conditional_arithmetic_status"],
                         "conditional_fit_under_caller_referenced_delta")
        self.assertEqual(result["launch_storage_decision"],
                         "unresolved_requires_authenticated_bound_and_frozen_manifest")
        self.assertIn("native_payload_delta_evidence_not_authenticated",
                      result["launch_storage_blockers"])
        self.assertIn("collected_clean_source_manifest_not_verified",
                      result["launch_storage_blockers"])
        over = accounting.project(
            COMPONENT_SIZES, native_payload_delta_ceiling=109_951_102,
            native_payload_delta_evidence=evidence)
        self.assertFalse(over["arithmetic_fit_under_supplied_delta"])
        self.assertEqual(over["conditional_arithmetic_status"],
                         "conditional_exceeds_under_caller_referenced_delta")
        self.assertEqual(over["launch_storage_decision"],
                         "unresolved_requires_authenticated_bound_and_frozen_manifest")

    def test_strict_delta_and_component_inputs(self):
        with self.assertRaises(accounting.FinalStorageAccountingError):
            accounting.project(COMPONENT_SIZES,
                               native_payload_delta_ceiling=True,
                               native_payload_delta_evidence={
                                   "kind": accounting.NATIVE_DELTA_EVIDENCE_KIND,
                                   "sha256": "a" * 64})
        with self.assertRaises(accounting.FinalStorageAccountingError):
            accounting.project(COMPONENT_SIZES, native_payload_delta_ceiling=1)
        with self.assertRaises(accounting.FinalStorageAccountingError):
            accounting.project(COMPONENT_SIZES, native_payload_delta_evidence={
                "kind": accounting.NATIVE_DELTA_EVIDENCE_KIND,
                "sha256": "a" * 64})
        with self.assertRaises(accounting.FinalStorageAccountingError):
            accounting.project(COMPONENT_SIZES, native_payload_delta_ceiling=1,
                               native_payload_delta_evidence={
                                   "kind": "unreviewed",
                                   "sha256": "a" * 64})
        missing = dict(COMPONENT_SIZES)
        missing.pop("plan_long")
        with self.assertRaises(accounting.FinalStorageAccountingError):
            accounting.project(missing)

    def test_source_growth_is_explicit(self):
        status = accounting.source_membership_status()
        self.assertGreaterEqual(status["current_member_count"], 55)
        self.assertEqual(status["verified_baseline_member_count"], 55)
        self.assertEqual(status["declared_schema_complete"],
                         not status["missing_required_new_members"])
        self.assertFalse(status["collected_clean_manifest_verified"])
        self.assertEqual(set(status["required_new_members"]),
                         {"phase_worker.py", "process_supervision.py",
                          "final_storage_accounting.py", "native_layout_inspection.py",
                          "pickle_storage_bound.py", "zip_storage_bound.py",
                          "native_tensor_inventory.py", "comparison_storage_bound.py",
                          "primitive_storage_bound.py", "core_primitive_bound.py",
                          "branch_primitive_bound.py", "audit_primitive_bound.py",
                          "native_storage_topology_bound.py", "native_payload_guard.py",
                          "native_write_ledger.py", "root_storage_accounting.py",
                          "native_storage_authority.py", "native_storage_recipe_common.py",
                          "native_storage_recipe_core.py", "native_storage_recipe_branch_audit.py",
             "native_storage_crosscheck.py", "native_storage_measurement_service.py",
             "native_storage_measurement_controller.py", "native_storage_measurement_slot.py",
             "native_storage_measurement_worker.py"})

    def test_declared_scientific_sources_cover_local_execution_imports(self):
        import ast
        import source_environment_schema as schema
        here = Path(__file__).resolve().parent
        repository = here.parents[3]
        declared = {path for _, path in schema.SCIENTIFIC_ROLES}
        missing = []
        class Imports(ast.NodeVisitor):
            def __init__(self, source):
                self.source = source
            def visit_FunctionDef(self, node):
                # Explicit CPU fixture entry only; not reachable from the
                # fixed native worker/controller path. Keep this exception
                # narrow: other future unbound local imports must fail.
                if self.source == schema.I7 + "scientific_runner.py" and node.name == "run_fixture":
                    return
                self.generic_visit(node)
            def check(self, name):
                candidate = here / (name.split(".")[0] + ".py")
                if candidate.is_file():
                    relative = candidate.relative_to(repository).as_posix()
                    if relative not in declared:
                        missing.append((self.source, relative))
            def visit_Import(self, node):
                for alias in node.names:
                    self.check(alias.name)
            def visit_ImportFrom(self, node):
                if node.module:
                    self.check(node.module)
        for path in sorted(declared):
            if path.endswith(".py"):
                Imports(path).visit(ast.parse((repository / path).read_text()))
        self.assertEqual(missing, [])

    def test_runtime_admission_fails_closed_without_valid_evidence(self):
        with self.assertRaisesRegex(accounting.StorageAdmissionError,
                                    "reviewed storage admission rejected"):
            accounting.validate_runtime_admission(
                {"path":"/fixed/native-storage-admission.json", "size_bytes":1,
                 "sha256":"a" * 64}, sources={}, environment={})
        with self.assertRaisesRegex(accounting.StorageAdmissionError,
                                    "invalid storage admission input"):
            accounting.validate_runtime_admission(
                {"path":"/fixed/native-storage-admission.json", "size_bytes":True,
                 "sha256":"a" * 64}, sources={}, environment={})


if __name__ == "__main__":
    unittest.main()
