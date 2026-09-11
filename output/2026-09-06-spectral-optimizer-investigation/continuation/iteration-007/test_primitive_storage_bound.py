"""Tiny pure serializer arithmetic checks, not study/fixture execution."""
import json
import math
from pathlib import Path
import pickle
import subprocess
import sys
import unittest

import primitive_storage_bound as bound


class PrimitiveStorageBoundTests(unittest.TestCase):
    def test_literal_bound_dominates_small_primitive_pickles(self):
        values = [None, True, False, -0.0, 1.5, "", "a", "\u2603", "\U0001f600", "\ud800",
                  [], {}, (), (1,), (1, 2, 3), (1, 2, 3, 4),
                  {0: "zero", "list": [True, None, -0.0], "tuple": ("x", "y")},
                  *range(-260, 261), -(1 << 64), 1 << 2048]
        for value in values:
            self.assertLessEqual(len(pickle.dumps(value, protocol=2)), 3 + bound.literal(value))

    def test_json_five_times_bound_covers_primitive_extremes(self):
        strings = ["", "abc", "\x00\n\"\\", "\u2603", "\U0001f600", "\ud800"]
        scalars = [None, True, False, 0., -0., 5e-324, -sys.float_info.max,
                   sys.float_info.max, 1e20, -1e-5, -1, 255, 256, 65535,
                   -(1 << 31), (1 << 32)-1, 1 << 2048, -(1 << 2048), *strings]
        values = [*scalars, [], {}, (), [*scalars], tuple(scalars),
                  {str(index): value for index, value in enumerate(scalars)},
                  {-1: [], 100: "", "many": [None] * 1001}]
        for value in values:
            encoded = json.dumps(value, ensure_ascii=True, allow_nan=False, separators=(",", ":"))
            self.assertLessEqual(bound.literal(value), bound.json_subtree_bytes(len(encoded)))

    def test_batch_boundaries_are_charged_without_alias_savings(self):
        for count in (0, 1, 2, 999, 1000, 1001, 1999, 2000, 2001):
            values = [None] * count
            self.assertEqual(bound.literal(values), bound.repeat_list(count, bound.NULL))
            self.assertLessEqual(len(pickle.dumps(values, 2)), 3 + bound.literal(values))
            mapping = {str(index): None for index in range(count)}
            self.assertLessEqual(len(pickle.dumps(mapping, 2)), 3 + bound.literal(mapping))
            for value in (values, mapping):
                compact = json.dumps(value, separators=(",", ":"))
                self.assertLessEqual(bound.literal(value), 5 * len(compact))

    def test_aliases_never_increase_no_alias_cost(self):
        shared = ["small", (1, 2)]
        value = [shared, shared, shared[0], shared[0], (), ()]
        self.assertLessEqual(len(pickle.dumps(value, 2)), 3 + bound.literal(value))

    def test_integer_interval_endpoints_dominate(self):
        for low, high in ((-300, 300), (0, 260), (65000, 66000),
                          ((1 << 31)-2, (1 << 31)+2), (-(1 << 31)-2, -(1 << 31)+2)):
            ceiling = bound.integer(low, high)
            self.assertTrue(all(bound.literal(value) <= ceiling for value in range(low, high+1)))

    def test_malformed_inputs_rejected(self):
        for function, arguments in ((bound.text, (-1,)), (bound.text, (True,)),
                (bound.text, (1 << 32,)), (bound.integer, (2, 1)),
                (bound.integer, (False, 1)), (bound.repeat_list, (-1, 1)),
                (bound.repeat_list, (1, -1)), (bound.plist, ([False],)),
                (bound.pdict, ({True: 1},)), (bound.json_subtree_bytes, (0,)),
                (bound.literal, (math.inf,)), (bound.literal, (object(),))):
            with self.assertRaises(ValueError):
                function(*arguments)

    def test_import_and_arithmetic_are_torch_free(self):
        code = "import sys; import primitive_storage_bound as p; p.literal({'x':[1]}); assert 'torch' not in sys.modules"
        result = subprocess.run([sys.executable, "-B", "-c", code], cwd=Path(__file__).parent,
                                capture_output=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr.decode())


if __name__ == "__main__":
    unittest.main()
