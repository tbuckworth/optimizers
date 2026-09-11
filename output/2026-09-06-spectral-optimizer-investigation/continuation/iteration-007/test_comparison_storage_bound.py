"""Small schema/encoder checks; no state stream or experimental artifact."""
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

import comparison_storage_bound as bound


class ComparisonStorageBoundTests(unittest.TestCase):
    def test_json_composition_formula(self):
        import identity_codec as codec
        for value in ({}, {"a": False}, {"one": [1, "a"], "two": True},
                      {"escaped\nkey": "x", "unicode": "\u2603"}):
            lengths = {key: bound._constant_bytes(item) for key, item in value.items()}
            self.assertEqual(bound._object_bytes(lengths), bound._constant_bytes(value))
            self.assertEqual(bound._constant_bytes(value) + 1, len(codec.json_bytes(value)))

    def test_fixed_schema_ceiling_and_negative_authority(self):
        result = bound.pilot_comparison_json_ceiling()
        self.assertEqual(result["json_bytes_upper"], 308_940)
        self.assertEqual(result["trace_row_count"], 220)
        self.assertLess(result["json_bytes_upper"], 320 << 10)
        self.assertGreater(result["json_bytes_upper"], 307_621)
        self.assertEqual(result["trailing_newline_bytes"], 1)
        for key in ("native_measurement", "whole_study_fit_proven", "execution_authorized",
                    "scientific_execution_certified"):
            self.assertIs(result[key], False)

    def test_schema_drift_rejected(self):
        import source_history as history
        for field in ("FINGERPRINT_KEYS", "INITIAL_FINGERPRINT_KEYS", "COMPARISON_KEYS",
                      "INITIAL_COMPARISON_KEYS", "SUMMARY_KEYS", "CAPTURE_ROOT_KEYS"):
            with patch.object(history, field, getattr(history, field) + ("extra",)):
                with self.assertRaises(ValueError):
                    bound.pilot_comparison_json_ceiling()

    def test_import_is_inert(self):
        result = subprocess.run([sys.executable, "-B", "-c",
            "import sys; import comparison_storage_bound; assert 'torch' not in sys.modules"],
            cwd=Path(__file__).parent, capture_output=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr.decode())


if __name__ == "__main__":
    unittest.main()
