"""Tiny synthetic schema/receipt checks only; no scientific states or inference."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from experiments import measure_grokking_raw_direction_states as measurement
from experiments.grokking_confirmation import FILTER, canonical_hash, file_hash


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    return {"path": str(path.resolve()), "sha256": file_hash(path),
            "size_bytes": path.stat().st_size}


def build_batch(parent, pins):
    environment = {"device": "cuda", "synthetic": True}
    common = {"schema": "grokking_raw_direction_batch_v1", "seeds": list(measurement.SEEDS),
              "policy": measurement.POLICY, "source_sha256": pins, "source_commit": "a" * 40,
              "first_seed_is_resource_only_admission": True, "paid_spend_usd": 0,
              "batch_seconds_limit": measurement.BATCH_SECONDS,
              "seed_seconds_limit": measurement.SEED_SECONDS}
    manifest = write_json(parent / "batch-manifest.json", common)
    records, references, accepted, seed100 = {}, {}, [], None
    for seed in measurement.SEEDS:
        directory = parent / f"seed{seed}"
        directory.mkdir()
        checkpoint = {"path": f"/synthetic/{seed}/checkpoint-step-001500.pt",
                      "sha256": f"{seed:064x}", "size_bytes": 1}
        metrics_hash = f"{seed + 1:064x}"
        record = {"checkpoints": [checkpoint], "config": {"seed": seed, "arm": "legacy",
                  "filter": {**FILTER, "stable_update": False}},
                  "split_identity": {"seed": seed}, "source_identity": {"synthetic": True},
                  "parameter_identity": [{"name": "weight", "shape": [2, 3],
                                          "dtype": "torch.float32"}]}
        records[seed, "legacy"] = (record, metrics_hash)
        references[seed] = {"synthetic": True, "fork_scientific_state_sha256": f"{seed + 2:064x}"}
        seed_common = {"schema": measurement.ACQUISITION_SCHEMA, "seed": seed,
                       "policy": measurement.POLICY, "source_sha256": pins,
                       "source_commit": f"{seed:040x}", "environment": environment,
                       "parent_checkpoint": checkpoint, "parent_metrics_sha256": metrics_hash,
                       "fork_scientific_state_sha256": references[seed]["fork_scientific_state_sha256"],
                       "archived_reference": references[seed], "seed100_admission": seed100,
                       "capture_steps": list(measurement.CAPTURE_STEPS),
                       "no_reference_policy_execution": True}
        artifacts = [write_json(directory / "manifest.json", seed_common),
                     write_json(directory / "first-step-tensors.pt", {"synthetic": True}),
                     write_json(directory / "first-step-summary.json", {"synthetic": True})]
        checkpoints = []
        for step in measurement.CAPTURE_STEPS:
            receipt = write_json(directory / f"{measurement.POLICY}-step-{step:06d}.pt",
                                 {"synthetic": True})
            receipt.update({"seed": seed, "policy": measurement.POLICY, "step": step})
            checkpoints.append(receipt)
            artifacts.append(receipt)
            artifacts.append(write_json(directory / f"{measurement.POLICY}-through-{step:06d}.json", {
                "schema": measurement.ACQUISITION_SCHEMA, "seed": seed,
                "policy": measurement.POLICY, "step": step, "source_sha256": pins,
                "checkpoint": receipt, "action_history": [{"step": value}
                                                           for value in range(1501, step + 1)]}))
        complete = {**seed_common, "status": "complete", "elapsed_seconds": 1.,
                    "accepted_roster": [[measurement.POLICY, step] for step in measurement.CAPTURE_STEPS],
                    "completed_updates": 1000, "history_steps": list(range(1501, 2501)),
                    "checkpoints": checkpoints, "artifact_receipts": artifacts}
        completion_receipt = write_json(directory / "complete.json", complete)
        if seed == 100:
            seed100 = completion_receipt
        accepted.append({"seed": seed, **completion_receipt})
    write_json(parent / "batch-complete.json", {
        **common, "status": "complete", "elapsed_seconds": 5., "batch_manifest": manifest,
        "environment": environment, "accepted": accepted})
    return records, references


def envelope_fixture():
    seed, step = 100, 1501
    config = {"seed": seed, "arm": "legacy", "filter": {**FILTER, "stable_update": False}}
    sources, split = {"synthetic": True}, {"sha256": "synthetic"}
    layout = [{"name": "weight", "shape": [2, 3], "dtype": "torch.float32"}]
    filter_state = {name: None for name in measurement.FILTER_MUTABLE}
    filter_state.update({"step_count": step, "stabilization_count": 0,
                         "max_orthogonality_error": 0.})
    inner = {"schema": measurement.CHECKPOINT_SCHEMA, "step": step,
             "model_state": {"weight": torch.zeros(2, 3)}, "optimizer_state": {},
             "filter_state": filter_state,
             "filter_configuration": {**config["filter"], "n_params": 6},
             "torch_cpu_rng_state": torch.zeros(1, dtype=torch.uint8),
             "torch_cuda_rng_states": [torch.zeros(1, dtype=torch.uint8)],
             "device_type": "cuda", "cuda_device_count": 1,
             "config": config, "config_sha256": canonical_hash(config),
             "source_identity": sources, "source_identity_sha256": canonical_hash(sources),
             "split_identity": split, "parameter_identity": layout,
             "evaluation_rows": [], "timing": {}}
    parent, first = {"synthetic": "parent"}, {"synthetic": "first"}
    completion = {"source_sha256": {"synthetic": "source"}, "parent_checkpoint": parent,
                  "environment": {"device": "cuda"}}
    item = {"seed": seed, "policy": measurement.POLICY, "step": step,
            "first_step_tensors": first, "completion": completion,
            "record": {"config": config, "source_identity": sources,
                       "split_identity": split, "parameter_identity": layout}}
    envelope = {"schema": measurement.OUTER_SCHEMA, "seed": seed, "policy": measurement.POLICY,
                "step": step, "parent_checkpoint": parent, "first_step_tensors": first,
                "source_sha256": completion["source_sha256"], "native_compatible_state": inner,
                "note": "synthetic"}
    return envelope, item, split


class RawDirectionMeasurementTests(unittest.TestCase):
    def test_fixed_new_roster_and_frozen_helpers(self):
        roster = measurement.expected_roster()
        self.assertEqual(len(roster), 15)
        self.assertEqual(len(set(roster)), 15)
        self.assertEqual({policy for _, policy, _ in roster}, {"raw_norm_matched"})
        self.assertEqual(roster[0], (100, "raw_norm_matched", 1501))
        self.assertEqual(roster[-1], (104, "raw_norm_matched", 2500))
        from experiments import measure_grokking_action_states as original
        for name in ("Output", "extract_activations", "evaluate_fourier_probes", "analyze_state",
                     "_structural_arrays", "load_prior_recipe_contracts", "validate_environment_contract"):
            self.assertIs(getattr(measurement, name), getattr(original, name))
        self.assertEqual(measurement.FIXED_PANEL, {"frequencies": [9, 33, 32, 49, 11]})
        self.assertEqual(measurement.RECIPE["ridge"], .001)
        self.assertEqual(measurement.RECIPE["top_k"], 5)
        self.assertEqual(measurement.RECIPE["n_nulls"], 20)

    def test_batch_admission_binds_all_fifteen_new_states(self):
        pins = {"synthetic": "source"}
        with tempfile.TemporaryDirectory(dir="/tmp/spectral-experiment-artifacts") as temporary:
            parent = Path(temporary)
            records, references = build_batch(parent, pins)
            with patch.object(measurement, "archived_parent", side_effect=lambda seed, *_: references[seed]), \
                    patch.object(measurement, "load_checkpoint", side_effect=AssertionError("No tensor loading")):
                admitted = measurement.verify_raw_batch(parent, records, pins)
                self.assertEqual([(item["seed"], item["policy"], item["step"])
                                  for item in admitted["states"]], measurement.expected_roster())
                self.assertEqual(len(admitted["input_receipts"]), 52)
                with self.assertRaisesRegex(ValueError, "source"):
                    measurement.verify_raw_batch(parent, records, {"synthetic": "changed"})
                history = parent / "seed103/raw_norm_matched-through-002500.json"
                history.write_bytes(history.read_bytes() + b" ")
                with self.assertRaisesRegex(ValueError, "receipt"):
                    measurement.verify_raw_batch(parent, records, pins)

    def test_missing_batch_and_failure_markers_reject_without_loading(self):
        pins = {"synthetic": "source"}
        with tempfile.TemporaryDirectory(dir="/tmp/spectral-experiment-artifacts") as temporary:
            parent = Path(temporary)
            with patch.object(measurement, "load_checkpoint", side_effect=AssertionError("No loading")):
                with self.assertRaises(ValueError):
                    measurement.verify_raw_batch(parent, {}, pins)
            records, references = build_batch(parent, pins)
            write_json(parent / "batch-failure.json", {"synthetic": True})
            with patch.object(measurement, "archived_parent", side_effect=lambda seed, *_: references[seed]):
                with self.assertRaisesRegex(ValueError, "failure"):
                    measurement.verify_raw_batch(parent, records, pins)

    def test_outer_new_policy_and_inner_legacy_are_separate(self):
        envelope, item, split = envelope_fixture()
        self.assertIs(measurement.validate_envelope(envelope, item, split),
                      envelope["native_compatible_state"])
        for field, value in (("policy", "norm_matched"), ("schema", "grokking_action_checkpoint_v1"),
                             ("first_step_tensors", {"changed": True})):
            changed = copy.deepcopy(envelope)
            changed[field] = value
            with self.assertRaisesRegex(ValueError, "Outer"):
                measurement.validate_envelope(changed, item, split)
        changed = copy.deepcopy(envelope)
        changed["native_compatible_state"]["config"]["arm"] = measurement.POLICY
        with self.assertRaisesRegex(ValueError, "Inherited"):
            measurement.validate_envelope(changed, item, split)

    def test_inner_filter_device_finiteness_and_evaluation_contracts(self):
        envelope, item, split = envelope_fixture()
        for mutate in (
            lambda inner: inner["filter_configuration"].update({"stable_update": True}),
            lambda inner: inner.update({"device_type": "cpu"}),
            lambda inner: inner["filter_state"].update({"step_count": 1500}),
            lambda inner: inner["model_state"]["weight"].fill_(float("nan")),
            lambda inner: inner.update({"evaluation_rows": [{"step": 1501}]}),
        ):
            changed = copy.deepcopy(envelope)
            mutate(changed["native_compatible_state"])
            with self.assertRaises(ValueError):
                measurement.validate_envelope(changed, item, split)
        envelope["step"] = item["step"] = 2000
        inner = envelope["native_compatible_state"]
        inner["step"] = inner["filter_state"]["step_count"] = 2000
        inner["evaluation_rows"] = [{"step": step} for step in range(1550, 2001, 50)]
        measurement.validate_envelope(envelope, item, split)

    def test_original_grid_tolerances_and_no_first_step_replay(self):
        measured = {name: {"loss": 1., "accuracy": .25} for name in ("train", "test")}
        grid = [{"step": 2000, **copy.deepcopy(measured)}]
        measurement.check_training_grid(measured, grid, 2000)
        measurement.check_training_grid(measured, [], 1501)
        grid[0]["test"]["loss"] += .0002
        with self.assertRaisesRegex(ValueError, "tolerance"):
            measurement.check_training_grid(measured, grid, 2000)
        grid = [{"step": 2000, **copy.deepcopy(measured)}]
        grid[0]["test"]["accuracy"] += .000002
        with self.assertRaisesRegex(ValueError, "tolerance"):
            measurement.check_training_grid(measured, grid, 2000)
        with self.assertRaises(ValueError):
            measurement.check_training_grid(measured, [], 2000)

    def test_old_source_map_is_bound_to_accepted_receipt(self):
        pins = {"synthetic": "old-source"}
        with tempfile.TemporaryDirectory(dir="/tmp/spectral-experiment-artifacts") as temporary:
            parent = Path(temporary)
            manifest = write_json(parent / "manifest.json", {
                "schema": "grokking_action_measurement_v1", "measurement_source_sha256": pins,
                "recipe": measurement.RECIPE, "fixed_frequency_panel": measurement.FIXED_PANEL})
            complete = write_json(parent / "complete.json", {
                "schema": "grokking_action_measurement_complete_v1", "status": "complete",
                "state_count": 35, "manifest": manifest, "measurement_source_sha256": pins})
            with patch.object(measurement, "OLD_MEASUREMENT", parent), \
                    patch.object(measurement, "OLD_MEASUREMENT_COMPLETE_SHA256", complete["sha256"]), \
                    patch.object(measurement, "original_measurement_sources", return_value=pins), \
                    patch.object(measurement, "load_checkpoint", side_effect=AssertionError("No loading")):
                record, receipts = measurement.verify_old_measurement_sources()
                self.assertEqual(record["measurement_source_sha256"], pins)
                self.assertEqual(len(receipts), 2)
            with patch.object(measurement, "OLD_MEASUREMENT", parent), \
                    patch.object(measurement, "OLD_MEASUREMENT_COMPLETE_SHA256", complete["sha256"]), \
                    patch.object(measurement, "original_measurement_sources", return_value={"changed": "source"}):
                with self.assertRaisesRegex(ValueError, "source"):
                    measurement.verify_old_measurement_sources()

    def test_old_policy_rejected_before_extraction(self):
        with patch.object(measurement, "load_checkpoint", side_effect=AssertionError("No loading")), \
                patch.object(measurement, "extract_activations", side_effect=AssertionError("No inference")):
            for policy in ("native", "norm_matched", "orthogonal"):
                with self.assertRaisesRegex(ValueError, "unregistered"):
                    measurement.measure_one({"seed": 100, "policy": policy, "step": 1501},
                                            None, None, None, None, None)


if __name__ == "__main__":
    unittest.main()
