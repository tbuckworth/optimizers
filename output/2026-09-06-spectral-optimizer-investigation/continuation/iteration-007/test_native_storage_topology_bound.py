"""Arithmetic-only complete-topology checks; no serialized specimen or study."""
from itertools import permutations
import subprocess
import sys
import unittest

import native_storage_topology_bound as bound
import native_tensor_inventory as inventory
import pickle_storage_bound as pickle_bound
import zip_storage_bound as zip_bound


class NativeTopologyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = bound.compute()

    def test_complete_sums_and_explicit_nonadmission(self):
        result = self.result
        rows = result["components"]
        self.assertEqual(tuple(rows), tuple(inventory.COMPONENT_COUNTS))
        self.assertEqual(sum(row["payload_count"] for row in rows.values()), 82)
        self.assertEqual(sum(row["payload_count"] * row.get("storage_count", 0) for row in rows.values()), 3424)
        self.assertEqual(sum(row["payload_count"] * row.get("record_count", 0) for row in rows.values()), 3910)
        self.assertEqual(sum(row["aggregate_body_bytes_upper"] for row in rows.values()), 1049771346)
        self.assertEqual(result["complete_logical_bytes_upper"], 1068220867)
        self.assertEqual(result["conditional_residual_bytes"], 5520957)
        self.assertTrue(result["conditional_arithmetic_fits"])
        for key in ("actual_runtime_domains_validated", "source_manifest_authenticated", "native_layout_observed",
                    "storage_fit_proven", "execution_authorized", "scientific_execution_certified"):
            self.assertIs(result[key], False)

    def test_every_proto_stop_and_tensor_count_charged_once(self):
        for key, row in self.result["components"].items():
            if row["encoding"] == "bytes":
                self.assertEqual(key, "capture_comparison_pilot")
                continue
            self.assertEqual(row["pickle_bytes_upper"], 3 + row["primitive_subtree_bytes_upper"]
                             + row["tensor_reduction_bytes_upper"])
            self.assertEqual(row["record_count"], row["storage_count"] + 6)
            self.assertEqual(row["archive_bytes_upper"], row["raw_tensor_bytes"] + row["pickle_bytes_upper"]
                             + 51 + row["zip_overhead_bytes_upper"])
            self.assertLess(row["archive_bytes_upper"], 64 << 20)
        audit = self.result["components"]["independent_audit_all_failure"]
        self.assertEqual(audit["storage_count"], 0)
        self.assertEqual(audit["tensor_reduction_bytes_upper"], 0)
        self.assertEqual(audit["pickle_bytes_upper"], 7050526)
        self.assertEqual(audit["aggregate_body_bytes_upper"], 16*audit["body_bytes_upper"])

    def test_order_independent_zip_dominates_all_small_permutations(self):
        # Size tuples only; no Tensor, archive, synthetic study or serialization.
        for pickle_size in (1, 63, 64, 65, 999, 1000):
            for sizes in ((), (0,), (1, 63, 64, 1025)):
                ceiling = bound.order_independent_zip_ceiling(pickle_size, sizes)["archive_bytes_upper"]
                for permuted in permutations(sizes):
                    actual_order = zip_bound.torch_save_zip_ceiling(pickle_size, permuted)
                    self.assertLessEqual(actual_order["archive_bytes_upper"], ceiling)

    def test_zip_index_boundaries_and_omitted_storages(self):
        for count in (0, 1, 9, 10, 11, 99, 100, 101, 132):
            sizes = (65,) * count
            ceiling = bound.order_independent_zip_ceiling(512, sizes)["archive_bytes_upper"]
            for lesser in (0, count//2, count):
                exact_order = zip_bound.torch_save_zip_ceiling(512, (64,) * lesser)
                self.assertLessEqual(exact_order["archive_bytes_upper"], ceiling)

    def test_width_reduction_dominance(self):
        for width in range(1, 33):
            for shape, dtype, itemsize in (((50890, width), "torch.float32", 4),
                                          ((width,), "torch.float64", 8)):
                maximum = (50890, 32) if len(shape)==2 else (32,)
                def cost(size):
                    return pickle_bound.protocol2_tensor_bytes(dtype_name=dtype,
                        storage_nbytes=bound.prod(size)*itemsize, shape=size,
                        stride=bound._contiguous_stride(size), tensor_count_cap=132)
                self.assertLessEqual(cost(shape), cost(maximum))

    def test_large_cost_does_not_become_permission(self):
        primitive = {key: row["primitive_subtree_bytes_upper"]
                     for key,row in self.result["components"].items() if row["encoding"] != "bytes"}
        primitive["independent_audit_all_failure"] += 1 << 20
        result = bound.combine(primitive)
        self.assertFalse(result["conditional_arithmetic_fits"])
        self.assertFalse(result["execution_authorized"])

    def test_reject_bad_arithmetic_arguments(self):
        for value in (True, 0, -1, 1.5):
            with self.assertRaises(ValueError):
                bound.order_independent_zip_ceiling(value, ())
        for values in ([1], (True,), (-1,), (1.5,)):
            with self.assertRaises(ValueError):
                bound.order_independent_zip_ceiling(1, values)
        with self.assertRaises(ValueError):
            bound.order_independent_zip_ceiling(64 << 20, ())
        for values in ({}, {"bad": 1}, []):
            with self.assertRaises(ValueError):
                bound.combine(values)

    def test_import_and_default_cli_are_torch_free(self):
        code = "import sys; import native_storage_topology_bound; assert 'torch' not in sys.modules; assert 'numpy' not in sys.modules"
        result = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        result = subprocess.run([sys.executable, "-B", "-m", "native_storage_topology_bound"],
                                capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("no payload, runtime or execution admission", result.stdout)


if __name__ == "__main__":
    unittest.main()
