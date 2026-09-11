"""Primary-calculator cross-checks: CPU imports, no data/plan/producer execution."""
import unittest

import comparison_storage_bound as comparison
import final_storage_accounting as accounting
import native_storage_topology_bound as bound
import native_tensor_inventory as inventory


class NativeTopologyAccountingTests(unittest.TestCase):
    def test_comparison_constant_matches_primary_analytic_encoder(self):
        import torch
        self.assertFalse(torch.cuda.is_initialized())
        result = comparison.pilot_comparison_json_ceiling()
        self.assertEqual(result["json_bytes_upper"], bound.JSON_COMPARISON_BYTES)
        self.assertFalse(torch.cuda.is_initialized())

    def test_nonbody_constant_matches_complete_ledger_not_root_only(self):
        import torch
        self.assertFalse(torch.cuda.is_initialized())
        # Unit arithmetic costs, not a measured or synthetic science artifact.
        result = accounting.project({key: 1 for key in inventory.COMPONENT_COUNTS})
        amounts = result["logical_bytes"]
        self.assertEqual(amounts["measured_cpu_component_projection"], 82)
        self.assertEqual(amounts["conditional_total_before_native_delta"] - 82,
                         bound.NONBODY_COMPLETE_LEDGER_BYTES)
        self.assertEqual(amounts["outside_root_normal_ceiling"], 287990)
        self.assertEqual(amounts["shared_failure_reserve"], 1 << 20)
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == "__main__":
    unittest.main()
