"""Descriptor arithmetic tests; no tensors, fixtures, data or plans are generated."""
import json
from pathlib import Path
import subprocess
import sys
import unittest

import native_tensor_inventory as inventory


class NativeTensorInventoryTests(unittest.TestCase):
    def test_arithmetic_and_negative_authority(self):
        result = inventory.summary()
        self.assertEqual(result["payload_count"], 82)
        self.assertEqual((result["torch_payload_count"], result["json_payload_count"]), (81, 1))
        self.assertEqual(result["aggregate_tensor_count"], 3424)
        self.assertEqual(result["aggregate_raw_tensor_bytes"], 917_277_344)
        self.assertEqual(result["fixture_absent_cuda_storage_count"], 48)
        self.assertEqual(result["fixture_absent_cuda_raw_bytes"], 768)
        for key in ("native_layout_observed", "serialized_size_bounded", "storage_fit_proven",
                    "execution_authorized", "scientific_execution_certified"):
            self.assertIs(result[key], False)

    def test_component_maxima(self):
        result = inventory.summary()["components"]
        expected = {
            "anchor_pilot": (27, 7_378_608), "anchor_long": (27, 7_378_608),
            "source_witness": (21, 7_735_552), "branch_results": (132, 32_977_072),
            "independent_audit_all_failure": (0, 0),
            "source_completion_pilot_on": (24, 7_336_048),
            "source_completion_pilot_off": (24, 7_336_048),
            "capture_comparison_pilot": (0, 0),
            "source_completion_long_on": (24, 7_336_048),
            "plan_pilot": (8, 794_688), "plan_long": (8, 1_706_048),
        }
        self.assertEqual({key: (row["tensor_count"], row["raw_tensor_bytes"])
                          for key, row in result.items()}, expected)

    def test_two_distinct_native_rng_leaves_per_core(self):
        layouts = inventory.component_layouts()
        for key in ("anchor_pilot", "anchor_long", "source_completion_pilot_on",
                    "source_completion_pilot_off", "source_completion_long_on"):
            cuda_rows = [row for row in layouts[key] if "torch_cuda" in row["path"]]
            self.assertEqual(len(cuda_rows), 2)
            self.assertEqual({(row["dtype"], row["shape"], row["nbytes"])
                              for row in cuda_rows}, {("uint8", (16,), 16), ("float32", (4,), 16)})
            self.assertEqual(sum(row["dtype"] == "uint32" for row in layouts[key]), 1)

    def test_descriptor_ownership(self):
        first, second = inventory.component_layouts(), inventory.component_layouts()
        first["anchor_pilot"][0]["nbytes"] = -1
        self.assertGreater(second["anchor_pilot"][0]["nbytes"], 0)
        self.assertGreater(first["anchor_long"][0]["nbytes"], 0)

    def test_prior_recorded_cpu_bytes_plus_both_cuda_tensors(self):
        # Read already completed evidence only. This neither reruns the specimen
        # nor promotes the CPU observation into a native measurement.
        path = Path(__file__).with_name("full-storage-final-measurement.json")
        document = json.loads(path.read_bytes())
        observations = document["measurements"]
        for observed in observations:
            components = observed["components"]
            for key, rows in inventory.component_layouts().items():
                native = [row for row in rows if "torch_cuda" in row["path"]]
                self.assertEqual(len(rows), components[key]["tensor_count"] + len(native))
                self.assertEqual(sum(row["nbytes"] for row in rows),
                                 components[key]["tensor_bytes"] + sum(row["nbytes"] for row in native))

    def test_import_remains_torch_free(self):
        code = "import sys; import native_tensor_inventory as n; n.summary(); assert 'torch' not in sys.modules"
        result = subprocess.run([sys.executable, "-B", "-c", code],
                                cwd=Path(__file__).parent, capture_output=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr.decode())

    def test_exact_paths_dtypes_and_shapes_against_current_fixture_declarations(self):
        # Import declarations only; do not run fixture_context, tiny producers,
        # lift_tree, plan generation or any serialization measurement.
        import full_envelope_storage_fixture as fixture
        kinds = {"anchor_pilot": "anchor", "anchor_long": "anchor",
                 "source_witness": "source_witness", "branch_results": "branch_results",
                 "independent_audit_all_failure": "independent_audit",
                 "source_completion_pilot_on": "source_completion",
                 "source_completion_pilot_off": "source_completion",
                 "source_completion_long_on": "source_completion",
                 "plan_pilot": "plan_arrays", "plan_long": "plan_arrays"}
        for key, rows in inventory.component_layouts().items():
            if key == "capture_comparison_pilot":
                self.assertEqual(rows, ())
                continue
            cpu_rows = [row for row in rows if "torch_cuda" not in row["path"]]
            declared = fixture._allowlisted_tensor_paths(kinds[key])
            self.assertEqual({row["path"]: "torch." + row["dtype"] for row in cpu_rows},
                             {path: str(dtype) for path, dtype in declared.items()})
        self.assertEqual(tuple((index, name, shape) for index, name, _, shape in fixture.PARAMETERS),
                         inventory.PARAMETERS)
        self.assertEqual((fixture.P, fixture.RANK), (inventory.P, inventory.RANK))
        for key in ("plan_pilot", "plan_long"):
            for row in inventory.component_layouts()[key]:
                if row["path"] != ("training_batches",):
                    self.assertEqual(row["shape"], fixture.PLAN_SHAPES[row["path"][0]])


if __name__ == "__main__":
    unittest.main()
