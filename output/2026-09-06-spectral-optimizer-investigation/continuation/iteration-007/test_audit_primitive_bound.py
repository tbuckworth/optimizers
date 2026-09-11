"""Tiny arithmetic/source-binding checks; no audit or experiment producer runs."""
import hashlib
from pathlib import Path
import stat
import subprocess
import sys
import unittest

import audit_primitive_bound as audit


HERE = Path(__file__).resolve().parent


class AuditPrimitiveBoundTests(unittest.TestCase):
    def test_exact_sixteen_member_schedule(self):
        expected = tuple(
            [("primary", bundle, update) for bundle in (71001, 71002, 71003)
             for update in (101, 500, 1000, 2000)]
            + [("sensitivity", 71901, update) for update in (101, 500, 1000, 2000)]
        )
        self.assertEqual(audit.MEMBERS, expected)

    def test_all_defined_topology_counts_every_bitset(self):
        expected = {
            "measurement_paths": 4010,
            "comparison_paths": 441,
            "fixed_failure_sites": 5081,
            "failure_rows_with_validation": 5082,
            "native_discordance_paths": 392,
            "unresolved_geometry_paths": 51,
            "defined_branches": 6,
            "defined_contrasts": 15,
            "measurement_bitsets": 16,
            "projection_bitsets": 4,
            "adam_bitsets": 72,
            "total_bitsets": 92,
        }
        self.assertEqual(audit.audit_topology_counts(), expected)
        self.assertEqual(audit.audit_topology_counts(nc_zero=True, nl_zero=True), expected)

    def test_each_permitted_undefined_domain_is_enumerated(self):
        expected = {
            "measurement_paths": 2835,
            "comparison_paths": 295,
            "fixed_failure_sites": 3550,
            "failure_rows_with_validation": 3551,
            "native_discordance_paths": 294,
            "unresolved_geometry_paths": 35,
            "defined_branches": 5,
            "defined_contrasts": 10,
            "measurement_bitsets": 14,
            "projection_bitsets": 4,
            "adam_bitsets": 60,
            "total_bitsets": 78,
        }
        self.assertEqual(audit.audit_topology_counts(nc_zero=False, nl_zero=True), expected)
        self.assertEqual(audit.audit_topology_counts(nc_zero=True, nl_zero=False), expected)

    def test_bitset_widths_cover_all_parameter_and_flat_screens(self):
        dimensions = [audit.P, *(dimension for _, dimension in audit.PARAMETERS)]
        self.assertEqual(
            {dimension: 2 * ((dimension + 7) // 8) for dimension in dimensions},
            {50_890: 12_724, 50_176: 12_544, 64: 16, 640: 160, 10: 4},
        )
        self.assertEqual(4 + 4 + 12 + 6 * 4 * 3, 92)

    def test_frozen_single_payload_and_envelope_costs(self):
        self.assertEqual(audit.audit_payload_subtree_cost(
            role="primary", bundle=71001, update=101), 7_049_379)
        self.assertEqual(audit.audit_artifact_pickle_bytes(
            role="primary", bundle=71001, update=101), 7_050_475)
        self.assertEqual(audit.audit_artifact_pickle_bytes(
            role="sensitivity", bundle=71901, update=2000), 7_050_526)

    def test_study_bound_reports_both_failstop_and_selected_uniform_sum(self):
        result = audit.scientific_audits_pickle_bytes()
        self.assertEqual(result["scheduled_payload_count"], 16)
        self.assertEqual(len(result["per_artifact"]), 16)
        self.assertEqual(result["all_pass_pickle_bytes"], 33_587_508)
        self.assertEqual(result["terminal_prefix_pickle_bytes"], 38_538_780)
        self.assertEqual(result["independent_per_artifact_max_sum"], 112_807_860)
        self.assertEqual(result["uniform_independent_max_sum"], 112_808_416)
        self.assertEqual(result["pickle_bytes"], 112_808_416)
        self.assertLess(result["terminal_prefix_pickle_bytes"], result["pickle_bytes"])

    def test_environment_term_is_conditional_linear_and_bounded(self):
        small = audit.audit_artifact_pickle_bytes(
            role="primary", bundle=71001, update=101,
            auditor_environment_json_bytes=8191)
        large = audit.audit_artifact_pickle_bytes(
            role="primary", bundle=71001, update=101,
            auditor_environment_json_bytes=8192)
        self.assertEqual(large - small, 5)
        for value in (0, 8193, True):
            with self.assertRaises(ValueError):
                audit.audit_artifact_pickle_bytes(
                    role="primary", bundle=71001, update=101,
                    auditor_environment_json_bytes=value)

    def test_only_exact_scientific_audit_members_are_accepted(self):
        invalid = (("pilot", 71990, 101), ("primary", 71001, 200),
                   ("sensitivity", 71001, 101), ("primary", True, 101))
        for role, bundle, update in invalid:
            with self.assertRaises(ValueError):
                audit.audit_payload_subtree_cost(role=role, bundle=bundle, update=update)

    def test_derived_sources_remain_exact(self):
        for name, expected in {**audit.SOURCE_SHA256, **audit.DEPENDENCY_SHA256}.items():
            path = HERE / name
            info = path.stat(follow_symlinks=False)
            self.assertTrue(stat.S_ISREG(info.st_mode), name)
            self.assertLessEqual(info.st_size, 1 << 20, name)
            digest = hashlib.sha256()
            total = 0
            with path.open("rb") as handle:
                while chunk := handle.read(65536):
                    total += len(chunk)
                    self.assertLessEqual(total, 1 << 20, name)
                    digest.update(chunk)
            self.assertEqual(total, info.st_size, name)
            self.assertEqual(digest.hexdigest(), expected, name)

    def test_import_and_default_cli_are_inert_and_torch_free(self):
        for code in (
                "import sys; import audit_primitive_bound; assert 'torch' not in sys.modules; assert 'numpy' not in sys.modules",
                "import runpy,sys; runpy.run_module('audit_primitive_bound',run_name='__main__'); assert 'torch' not in sys.modules; assert 'numpy' not in sys.modules",
        ):
            result = subprocess.run([sys.executable, "-B", "-c", code], cwd=HERE,
                                    capture_output=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr.decode())


if __name__ == "__main__":
    unittest.main()
