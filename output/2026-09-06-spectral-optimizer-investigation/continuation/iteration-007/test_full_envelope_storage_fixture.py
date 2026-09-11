"""Focused CPU-only tests for the current-envelope storage-only lift."""
import contextlib
import copy
import io
import os
import unittest
from unittest import mock

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch

import audit_diagnostics as diagnostics
import full_envelope_storage_fixture as subject
import identity_codec as codec


class FullEnvelopeStorageFixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.actual = subject.actual_tiny_records()

    def test_actual_current_envelopes_pass_real_tiny_validators_and_old_count(self):
        self.assertEqual(self.actual["semantic_validator_valid"], {
            "anchor": True, "source_witness": True, "branch_results": True,
            "independent_audit": True, "source_completion_on": True,
            "source_completion_off": True, "capture_comparison": True})
        count = sum(len(list(subject.tensor_leaves(self.actual[key]))) for key in
                    ("anchor", "source_witness", "branch_results"))
        self.assertEqual(count, 178)
        lifted = subject.lift_tree("anchor", self.actual["anchor"])["tree"]
        self.assertEqual(tuple(lifted.keys()), tuple(self.actual["anchor"].keys()))
        self.assertEqual(tuple(lifted["payload"].keys()),
                         tuple(self.actual["anchor"]["payload"].keys()))
        self.assertEqual(lifted["profile"], subject.SPECIMEN_PROFILE)
        self.assertNotIn(lifted["profile"], subject.storage.FIXTURES + (subject.storage.SCIENTIFIC,))
        self.assertFalse(torch.cuda.is_initialized())

    def test_every_current_tensor_path_is_explicitly_accounted(self):
        cases = (("anchor", "anchor"), ("source_witness", "source_witness"),
                 ("branch_results", "branch_results"), ("final_core", "final_core"),
                 ("plan_arrays", "plan_arrays"))
        for kind, key in cases:
            with self.subTest(kind=kind):
                lifted = subject.lift_tree(kind, self.actual[key], plan_steps=2_000)
                source_paths = [path for path, _ in subject.tensor_leaves(self.actual[key])]
                self.assertEqual([row["path"] for row in lifted["tensor_manifest"]], source_paths)
                self.assertEqual(len(subject.inventory(lifted["tree"])), len(source_paths))
        anchor = subject.lift_tree("anchor", self.actual["anchor"])["tree"]
        self.assertEqual(anchor["payload"]["model"]["parameters"][0]["shape"], [64, 784])
        self.assertEqual(anchor["payload"]["observer"]["state"]["V"]["shape"], [50_890, 32])
        self.assertEqual(anchor["payload"]["bindings"]["plan"]["arrays"]["training_batches"]["shape"],
                         [2_000, 64])

    def test_unknown_tensor_and_changed_known_shape_fail_closed(self):
        malformed = {**self.actual["independent_audit"], "unknown": torch.zeros(1)}
        with self.assertRaisesRegex(ValueError, "unknown tensor path"):
            subject.lift_tree("independent_audit", malformed)
        moved_known_shape = copy.deepcopy(self.actual["independent_audit"])
        moved_known_shape["unknown"] = {"g": {"value": torch.zeros(26)}}
        with self.assertRaisesRegex(ValueError, "unknown tensor path"):
            subject.lift_tree("independent_audit", moved_known_shape)
        malformed = copy.deepcopy(self.actual["anchor"])
        malformed["payload"]["model"]["parameters"][0]["value"] = torch.zeros(13)
        with self.assertRaisesRegex(ValueError, "fixture parameter shape changed"):
            subject.lift_tree("anchor", malformed)
        missing = copy.deepcopy(self.actual["anchor"])
        missing["payload"]["observer"]["state"]["grad_mean"]["value"] = None
        with self.assertRaisesRegex(ValueError, "required tensor path set differs"):
            subject.lift_tree("anchor", missing)

    def test_known_looking_audit_bitset_moved_to_unknown_path_fails(self):
        malformed = copy.deepcopy(self.actual["independent_audit"])
        rows = malformed["payload"]["measurement_audits"]["result"]["measurement_audits"]
        record = rows["before.batch_noisy.q_values"].pop("failing_flat_indices")
        rows["unknown.batch_noisy.q_values"] = {"failing_flat_indices": record}
        with self.assertRaisesRegex(ValueError, "unknown audit bitset path"):
            subject.lift_tree("independent_audit", malformed)

    def test_every_lifted_tensor_has_fresh_exact_owned_storage(self):
        left = subject.lift_tree("anchor", self.actual["anchor"])["tree"]
        right = subject.lift_tree("anchor", self.actual["anchor"])["tree"]
        left_leaves, right_leaves = list(subject.tensor_leaves(left)), list(subject.tensor_leaves(right))
        left_ptrs = [tensor.untyped_storage().data_ptr() for _, tensor in left_leaves]
        right_ptrs = [tensor.untyped_storage().data_ptr() for _, tensor in right_leaves]
        self.assertEqual(len(left_ptrs), len(set(left_ptrs)))
        self.assertTrue(set(left_ptrs).isdisjoint(right_ptrs))
        for _, tensor in left_leaves + right_leaves:
            self.assertEqual(tensor.untyped_storage().nbytes(), tensor.numel() * tensor.element_size())
        original = dict(subject.tensor_leaves(self.actual["anchor"]))
        lifted = dict(left_leaves)
        rng_rows = [row for row in subject.lift_tree("anchor", self.actual["anchor"])["tensor_manifest"]
                    if row["classification"] == "observed_cpu_rng_unscaled"]
        self.assertEqual(len(rng_rows), 3)
        self.assertTrue(all(row["value_policy"] == "actual_fixture_rng_bytes_cloned"
                            for row in rng_rows))
        for row in rng_rows:
            self.assertTrue(torch.equal(original[row["path"]], lifted[row["path"]]))
            self.assertNotEqual(original[row["path"]].untyped_storage().data_ptr(),
                                lifted[row["path"]].untyped_storage().data_ptr())

    def test_all_failure_audit_bitsets_are_full_width_and_not_truncated(self):
        lifted = subject.lift_tree("independent_audit", self.actual["independent_audit"])
        rows = lifted["bitset_manifest"]
        self.assertEqual(len(rows), 92)
        self.assertEqual(sum(row["bits_hex_chars"] for row in rows), 483_512)
        self.assertTrue(all(row["count"] == row["lifted_dimension"] for row in rows))
        tree = lifted["tree"]
        first = tree["payload"]["candidate_operator_audit"]["projections"]["result"]["current"]["failing_flat_indices"]
        unpacked = diagnostics.unpack_indices(first)
        self.assertEqual((len(unpacked), unpacked[0], unpacked[-1]), (50_890, 0, 50_889))

    def test_source_trace_plan_and_fixed_membership_ledgers(self):
        pilot = subject.lift_source_completion(self.actual["source_completion_on"],
            steps=220, capture_mode="capture_on")
        long = subject.lift_source_completion(self.actual["source_completion_on"],
            steps=2_000, capture_mode="capture_on", long_run=True)
        comparison = subject.lift_capture_comparison(self.actual["capture_comparison"])
        self.assertEqual((len(pilot["tree"]["trace"]), len(long["tree"]["trace"]),
                          len(comparison["tree"]["step_trace"])), (220, 2_000, 220))
        self.assertEqual((len(pilot["tree"]["anchor_witness_refs"]),
                          len(long["tree"]["anchor_witness_refs"])), (2, 4))
        hashes = [row["core_sha256"] for row in long["tree"]["trace"]]
        self.assertEqual(len(hashes), len(set(hashes)))
        pilot_plan = subject.lift_tree("plan_arrays", self.actual["plan_arrays"], plan_steps=220)["tree"]
        long_plan = subject.lift_tree("plan_arrays", self.actual["plan_arrays"], plan_steps=2_000)["tree"]
        self.assertEqual(pilot_plan["training_batches"].shape, (220, 64))
        self.assertEqual(long_plan["training_batches"].shape, (2_000, 64))
        self.assertEqual(subject.KNOWN_FIXED_BEFORE_RNG, 917_094_560)
        self.assertEqual(subject.MEMBERSHIP["long_capture_on_source_completions"], 4)
        self.assertEqual(subject.MEMBERSHIP["pilot_capture_on_source_completions"]
                         + subject.MEMBERSHIP["pilot_capture_off_source_completions"], 2)
        self.assertEqual(subject.MEMBERSHIP["long_plans"] + subject.MEMBERSHIP["pilot_plans"], 5)
        json_row = subject.json_measurement(comparison["tree"])
        self.assertEqual(json_row["encoding"], "strict_compact_ascii_json_v1")
        self.assertEqual(json_row["serialized_bytes"], len(codec.json_bytes(comparison["tree"])))
        self.assertTrue(json_row["exact_full_tree_roundtrip"])

    def test_bounded_uncompressed_roundtrip_and_inert_default(self):
        audit = subject.lift_tree("independent_audit", self.actual["independent_audit"])["tree"]
        row = subject.serialized_measurement(audit)
        self.assertTrue(row["all_zip_entries_uncompressed"])
        self.assertTrue(row["restricted_weights_only_cpu_roundtrip"])
        self.assertLess(row["serialized_bytes"], row["buffer_limit_bytes"])
        altered = copy.deepcopy(audit)
        altered["schema_name"] = "changed-after-load"
        with mock.patch.object(subject.torch, "load", return_value=altered):
            with self.assertRaisesRegex(ValueError, "full tree differs"):
                subject.serialized_measurement(audit)
        anchor = subject.lift_tree("anchor", self.actual["anchor"])["tree"]
        with mock.patch.object(subject.torch, "load") as decoder:
            with self.assertRaises((subject.storage.StoreError, RuntimeError)):
                subject.serialized_measurement(anchor, buffer_limit=1_024)
            decoder.assert_not_called()
        with mock.patch.object(subject, "measure") as measure:
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(subject.main([]), 0)
            measure.assert_not_called()
        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"}):
            with self.assertRaisesRegex(ValueError, "hidden"):
                subject.lift_tree("anchor", self.actual["anchor"])


if __name__ == "__main__":
    unittest.main()
