"""Symbolic/tiny contracts only; never materialize a full-width component."""
import math
from pathlib import Path
import subprocess
import sys
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import native_storage_recipe_common as common
import primitive_storage_bound as primitive


class RecipeCommonTests(unittest.TestCase):
    def test_import_cli_and_symbolic_helpers_are_torch_free(self):
        code = (f"import sys;sys.path.insert(0,{str(HERE)!r});"
                "import native_storage_recipe_common as c;"
                "v={'profile':c.PROFILE,'a':[c.SLOT,3]};"
                "assert c.slot_paths(v)==(('a',0),);c.primitive_cost(v);"
                "c.copy_primitive(v);assert c.main([])==0;assert 'torch' not in sys.modules")
        run = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, timeout=20)
        self.assertEqual(run.returncode, 0, run.stderr.decode())

    def test_copy_has_owned_containers_and_same_exact_shapes(self):
        shared = {"key": [1, 2.0, None, True, common.SLOT]}
        tree = {"a": shared, "b": (shared,), 3: "value"}
        result = common.copy_primitive(tree)
        self.assertEqual(tuple(result), ("a", "b", 3))
        self.assertIs(type(result["b"]), tuple)
        self.assertIsNot(result["a"], shared)
        self.assertIsNot(result["a"], result["b"][0])
        self.assertIs(result["a"]["key"][-1], common.SLOT)

    def test_symbolic_primitive_cost_is_zero_only_for_slots(self):
        value = {"a": [common.SLOT, 0, False], 2: (None, "x")}
        expected = primitive.pdict({"a": primitive.plist([0, primitive.literal(0), primitive.BOOL]),
                                    2: primitive.ptuple([primitive.NULL, primitive.literal("x")])})
        self.assertEqual(common.primitive_cost(value), expected)
        self.assertEqual(common.slot_paths(value), (("a", 0),))

    def test_cycles_custom_values_nonfinite_large_ints_and_keys_rejected(self):
        cycle = []
        cycle.append(cycle)
        for value in (cycle, object(), math.inf, 1 << 65, {True: 1}, {1.0: 1}, '\ud800'):
            with self.subTest(value=type(value)), self.assertRaises(common.RecipeError):
                common.copy_primitive(value)

    def test_bounded_depth_nodes_and_strings(self):
        nested = 0
        for _ in range(common.MAX_DEPTH + 1):
            nested = [nested]
        for value in (nested, [0] * (common.MAX_NODES + 1), "a" * (common.MAX_STRING_BYTES + 1)):
            with self.assertRaises(common.RecipeError):
                common.primitive_cost(value)

    def test_profile_and_unknown_or_missing_inventory_rejected_without_torch(self):
        with self.assertRaises(common.RecipeError):
            common.validate_template("made_up", {"profile": common.PROFILE})
        with self.assertRaises(common.RecipeError):
            common.validate_template("anchor_pilot", {"profile": "scientific_mnist_current32_v1"})
        with self.assertRaises(common.RecipeError):
            common.validate_template("anchor_pilot", {"profile": common.PROFILE})

    def test_tiny_only_materialization_distinct_storages_and_exact_stride(self):
        import torch
        import native_tensor_inventory as inventory
        tree = {'profile': common.PROFILE, 'a': common.SLOT, 'b': common.SLOT}
        rows = [dict(path=(key,), shape=(2, 3), dtype='float32', nbytes=24)
                for key in ('a', 'b')]
        # Tiny substituted inventory ONLY; no real component materialization.
        with mock.patch.object(common, 'validate_template', return_value={}), \
             mock.patch.object(inventory, 'component_layouts', return_value={'tiny_unit': rows}):
            value = common.materialize('tiny_unit', tree)
            self.assertNotEqual(value['a'].untyped_storage()._cdata,
                                value['b'].untyped_storage()._cdata)
            self.assertEqual(tuple(value['a'].stride()), (3, 1))
            self.assertTrue(torch.equal(value['a'], torch.zeros((2, 3))))
            with mock.patch.object(torch.Tensor, 'stride', return_value=(1, 2)):
                with self.assertRaisesRegex(common.RecipeError, 'storage differs'):
                    common.materialize('tiny_unit', tree)
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == "__main__":
    unittest.main()
