"""Focused pure-ledger checks; no store, payload, plan, data, or producer."""
from pathlib import Path
import subprocess
import sys
import unittest


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import native_write_ledger as ledger


class NativeWriteLedgerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = ledger.compute()

    def test_import_and_default_cli_are_inert_and_torch_free(self):
        code = (
            "import sys; import native_write_ledger as x; "
            "assert 'torch' not in sys.modules; assert 'numpy' not in sys.modules; "
            "assert x.main([]) == 0; assert 'torch' not in sys.modules"
        )
        completed = subprocess.run(
            [sys.executable, "-I", "-B", "-c",
             f"import sys; sys.path.insert(0,{str(HERE)!r}); {code}"],
            check=False, capture_output=True, text=True, timeout=20)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(
            completed.stdout,
            "native_write_ledger: inert; no write or execution admission\n")

    def test_exact_global_inequalities_and_no_authority(self):
        logical = self.result["logical_bytes"]
        self.assertEqual(ledger.NORMAL_ROOT_CEILING_BYTES, 1_072_405_258)
        self.assertEqual(ledger.FAILURE_ROOT_CEILING_BYTES, 1_073_445_642)
        self.assertEqual(logical["successful_root_upper"], 1_066_884_301)
        self.assertEqual(logical["outside_root_normal_upper"], 287_990)
        self.assertEqual(logical["complete_success_plus_failure_reserve_upper"],
                         1_068_220_867)
        self.assertEqual(logical["normal_root_residual"], 5_520_957)
        self.assertTrue(self.result["conditional_arithmetic_fits"])
        for key in ("outside_writers_authenticated", "shared_store_integration_complete",
                    "execution_authorized", "scientific_execution_certified"):
            self.assertIs(self.result[key], False)

    def test_closed_body_receipt_and_static_success_membership(self):
        limits = self.result["body_limits"]
        membership = self.result["membership"]
        self.assertEqual(len(limits), 181)
        self.assertEqual(len(membership["body_names"]), 181)
        self.assertEqual(len(membership["receipt_names"]), 181)
        self.assertEqual(len(membership["static_success_names"]), 364)
        self.assertEqual(len(set(membership["static_success_names"])), 364)
        self.assertEqual(sum(row["receipt_bytes_upper"] for row in limits.values()),
                         40_605)
        self.assertEqual(sum(row["body_bytes_upper"] for row in limits.values()),
                         1_066_843_474)
        self.assertEqual(limits["native-phase-000.json"]["body_bytes_upper"],
                         128 << 10)
        self.assertEqual(limits["native-phase-089.json"]["encoding"], "bytes")
        self.assertEqual(limits["native-sources.json"]["body_bytes_upper"],
                         1 << 20)
        self.assertEqual(limits["native-primary-permission.json"]["body_bytes_upper"],
                         8 << 10)
        comparisons = [row for row in limits.values()
                       if row["component"] == "capture_comparison_pilot"]
        self.assertEqual(len(comparisons), 1)
        self.assertEqual(comparisons[0]["encoding"], "bytes")
        self.assertEqual(comparisons[0]["body_bytes_upper"], 308_940)
        torch_rows = [row for row in limits.values()
                      if row["component"] not in ("capture_comparison_pilot",
                                                  "phase_record", "root_metadata")]
        self.assertTrue(torch_rows)
        self.assertTrue(all(row["encoding"] == "torch_weights_only"
                            for row in torch_rows))

    def test_outside_namespace_and_shared_failure_split(self):
        outside = self.result["outside_root"]
        self.assertEqual(sum(row["bytes_upper"] for rows in outside.values()
                             for row in rows), 287_990)
        self.assertEqual(len(outside["development_external_success"]), 7)
        self.assertEqual(len(outside["supervision"]), 8)
        self.assertEqual(len(outside["existing_consumed_inspection"]), 2)
        self.assertEqual(len(outside["storage_evidence_prerequisites"]), 2)
        self.assertEqual(ledger.FAILURE_ROOT_CEILING_BYTES
                         - ledger.NORMAL_ROOT_CEILING_BYTES,
                         (1 << 20) - (8 << 10))

    def test_initialization_is_exact_and_unknown_names_fail(self):
        lock = ledger.initialization_reservation("store.lock", 0, 0)
        header = ledger.initialization_reservation("store-header.json", 222, 0)
        self.assertEqual(lock["post_root_logical_upper"], 0)
        self.assertEqual(header["post_root_logical_upper"], 222)
        for args in (("store.lock", 1, 0),
                     ("store-header.json", 221, 0),
                     ("store-header.json", 222, 1),
                     ("other", 0, 0),
                     ("store.lock", False, 0)):
            with self.subTest(args=args), self.assertRaises(ledger.NativeWriteLedgerError):
                ledger.initialization_reservation(*args)

    def test_receipt_formula_and_normal_pair_reservation(self):
        name = "native-phase-001.json"
        encoding = "bytes"
        spec = self.result["body_limits"][name]
        for body in (1, 9, 10, 99, 100, spec["body_bytes_upper"]):
            receipt = ledger.expected_receipt_bytes(name, encoding, body)
            self.assertLessEqual(receipt, spec["receipt_bytes_upper"])
            before = ledger.NORMAL_ROOT_CEILING_BYTES - body - spec["receipt_bytes_upper"]
            admitted = ledger.admit_normal_write(
                name, encoding, body, receipt, before)
            self.assertEqual(admitted["post_root_logical_upper"],
                             ledger.NORMAL_ROOT_CEILING_BYTES)
            self.assertEqual(admitted["reserved_increment_bytes"],
                             body + spec["receipt_bytes_upper"])

    def test_normal_budget_rejects_wrong_domain_and_overflow(self):
        name = "native-phase-002.json"
        spec = self.result["body_limits"][name]
        budget = ledger.normal_write_budget(name, "bytes", 222)
        self.assertEqual(budget["body_bytes_upper"], 128 << 10)
        self.assertEqual(budget["max_payload_bytes_now"], 128 << 10)
        self.assertEqual(budget["component"], "phase_record")
        bad_calls = (
            ("unknown.json", "bytes", 222),
            (name, "torch_weights_only", 222),
            (name, "bytes", True),
            (name, "bytes", ledger.NORMAL_ROOT_CEILING_BYTES),
        )
        for args in bad_calls:
            with self.subTest(args=args), self.assertRaises(ledger.NativeWriteLedgerError):
                ledger.normal_write_budget(*args)
        body = spec["body_bytes_upper"]
        receipt = ledger.expected_receipt_bytes(name, "bytes", body)
        with self.assertRaises(ledger.NativeWriteLedgerError):
            ledger.admit_normal_write(name, "bytes", body + 1, receipt, 222)
        with self.assertRaises(ledger.NativeWriteLedgerError):
            ledger.admit_normal_write(name, "bytes", body, receipt + 1, 222)

    def test_repeated_failure_sequence_is_preserved_and_bounded(self):
        first = ledger.admit_failure_write(
            "failure-000001.json", 16_384,
            ledger.FAILURE_ROOT_CEILING_BYTES - 16_384)
        self.assertEqual(first["post_root_logical_upper"],
                         ledger.FAILURE_ROOT_CEILING_BYTES)
        second = ledger.admit_failure_write("failure-000002.json", 1, 100)
        self.assertEqual(second["sequence"], 2)
        last = ledger.admit_failure_write("failure-999999.json", 1, 100)
        self.assertEqual(last["sequence"], 999_999)
        for args in (
            ("failure-000000.json", 1, 0),
            ("failure-1000000.json", 1, 0),
            ("failure-000001.json", 16_385, 0),
            ("failure-000001.json", 1,
             ledger.FAILURE_ROOT_CEILING_BYTES),
            ("failure-000001.json", True, 0),
        ):
            with self.subTest(args=args), self.assertRaises(ledger.NativeWriteLedgerError):
                ledger.admit_failure_write(*args)

    def test_allowed_names_separate_static_set_from_failure_pattern(self):
        names = ledger.allowed_root_names()
        self.assertEqual(len(names["static_success_names"]), 364)
        self.assertNotIn("failure-000001.json", names["static_success_names"])
        self.assertEqual(names["failure_name_pattern"],
                         r"failure-[0-9]{6}\.json")
        self.assertEqual(names["maximum_failure_sequence"], 999_999)


if __name__ == "__main__":
    unittest.main()
