"""CPU-only tests for lossless, bounded independent-audit diagnostics."""

import math
import unittest

import numpy as np

import audit_diagnostics as diagnostics


class IndexCodecTests(unittest.TestCase):
    def test_fixture_bits_are_lsb_zero_and_fixed_width(self):
        record = diagnostics.pack_indices([0, 1, 7, 8, 25], 26)
        self.assertEqual(record, {
            "encoding": "flat_bitset_hex_lsb0_v1",
            "dimension": 26,
            "count": 5,
            "bits_hex": "83010002",
        })
        self.assertEqual(diagnostics.unpack_indices(record), [0, 1, 7, 8, 25])
        empty = diagnostics.pack_indices([], 26)
        self.assertEqual(empty["bits_hex"], "00000000")
        self.assertEqual(diagnostics.unpack_indices(empty), [])

    def test_scientific_all_failure_round_trip_retains_every_coordinate(self):
        indices = list(range(50_890))
        record = diagnostics.pack_indices(indices, 50_890)
        self.assertEqual(record["count"], 50_890)
        self.assertEqual(len(record["bits_hex"]), 2 * math.ceil(50_890 / 8))
        self.assertEqual(record["bits_hex"][-2:], "03")
        self.assertEqual(diagnostics.unpack_indices(record), indices)

    def test_pack_rejects_noncanonical_or_out_of_domain_inputs(self):
        bad_cases = (
            ((0, 1), 26),
            ([0, 0], 26),
            ([1, 0], 26),
            ([-1], 26),
            ([26], 26),
            ([True], 26),
            ([np.int64(1)], 26),
            ([1.0], 26),
            ([], True),
            ([], 0),
            ([], 50_891),
            ([], np.int64(26)),
        )
        for indices, dimension in bad_cases:
            with self.subTest(indices=indices, dimension=dimension):
                with self.assertRaises(diagnostics.AuditDiagnosticError):
                    diagnostics.pack_indices(indices, dimension)

    def test_unpack_rejects_malformed_records(self):
        valid = diagnostics.pack_indices([0, 25], 26)
        malformed = []
        malformed.append({"dimension": 26, "encoding": diagnostics.ENCODING,
                          "count": 2, "bits_hex": valid["bits_hex"]})
        malformed.append(dict(valid, extra=None))
        malformed.append(dict(valid, encoding="flat_bitset_hex_msb0_v1"))
        malformed.append(dict(valid, dimension=True))
        malformed.append(dict(valid, dimension=50_891))
        malformed.append(dict(valid, count=True))
        malformed.append(dict(valid, count=1))
        malformed.append(dict(valid, bits_hex="010000"))
        malformed.append(dict(valid, bits_hex="0100000G"))
        malformed.append(dict(valid, bits_hex="0100000A"))
        malformed.append(dict(valid, bits_hex="01000004"))
        malformed.append([])
        for record in malformed:
            with self.subTest(record=record):
                with self.assertRaises(diagnostics.AuditDiagnosticError):
                    diagnostics.unpack_indices(record)

    def test_unpack_rejects_string_subclass_key(self):
        class StringSubclass(str):
            pass

        record = diagnostics.pack_indices([0], 26)
        record = {StringSubclass(key) if key == "count" else key: value
                  for key, value in record.items()}
        self.assertEqual(tuple(record), diagnostics.INDEX_RECORD_KEYS)
        with self.assertRaises(diagnostics.AuditDiagnosticError):
            diagnostics.unpack_indices(record)


class PrimitiveTreeTests(unittest.TestCase):
    def test_accepts_exact_bounded_primitives_without_copy_or_coercion(self):
        value = {"status": "fatal_validation", "failed": True,
                 "indices": [0, 2], "error": 0.25, "reason": None}
        result = diagnostics.primitive_tree(
            value, max_nodes=14, max_depth=2, max_string_bytes=64)
        self.assertIs(result, value)
        self.assertIs(result["indices"], value["indices"])

    def test_enforces_each_budget_exactly(self):
        value = {"a": ["é"]}  # dict + key + list + UTF-8 value = four nodes, three bytes.
        self.assertIs(diagnostics.primitive_tree(
            value, max_nodes=4, max_depth=2, max_string_bytes=3), value)
        for kwargs in (
            dict(max_nodes=3, max_depth=2, max_string_bytes=3),
            dict(max_nodes=4, max_depth=1, max_string_bytes=3),
            dict(max_nodes=4, max_depth=2, max_string_bytes=2),
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(diagnostics.AuditDiagnosticError):
                    diagnostics.primitive_tree(value, **kwargs)

    def test_shared_alias_costs_each_occurrence(self):
        shared = []
        value = [shared, shared]
        self.assertIs(diagnostics.primitive_tree(
            value, max_nodes=3, max_depth=1, max_string_bytes=0), value)
        with self.assertRaises(diagnostics.AuditDiagnosticError):
            diagnostics.primitive_tree(
                value, max_nodes=2, max_depth=1, max_string_bytes=0)

    def test_rejects_unsupported_nonfinite_and_cyclic_values(self):
        cycle = []
        cycle.append(cycle)
        bad_values = (
            ("tuple", (1,)),
            ("set", {1}),
            ("numpy scalar", np.int64(1)),
            ("numpy array", np.array([1])),
            ("nonfinite", float("inf")),
            ("nonfinite", float("nan")),
            ("non-string key", {1: "x"}),
            ("cycle", cycle),
        )
        for label, value in bad_values:
            with self.subTest(label=label):
                with self.assertRaises(diagnostics.AuditDiagnosticError):
                    diagnostics.primitive_tree(
                        value, max_nodes=20, max_depth=10, max_string_bytes=100)

    def test_rejects_invalid_budget_scalars(self):
        for kwargs in (
            dict(max_nodes=True, max_depth=0, max_string_bytes=0),
            dict(max_nodes=0, max_depth=0, max_string_bytes=0),
            dict(max_nodes=1, max_depth=True, max_string_bytes=0),
            dict(max_nodes=1, max_depth=-1, max_string_bytes=0),
            dict(max_nodes=1, max_depth=0, max_string_bytes=np.int64(0)),
            dict(max_nodes=1, max_depth=0, max_string_bytes=-1),
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(diagnostics.AuditDiagnosticError):
                    diagnostics.primitive_tree(None, **kwargs)


if __name__ == "__main__":
    unittest.main()
