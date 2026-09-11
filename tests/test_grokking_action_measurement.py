"""Synthetic contracts for action-state measurement; no scientific states."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments import measure_grokking_action_states as measurement
from experiments.grokking_confirmation import FILTER_MUTABLE, canonical_hash, file_hash


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def receipt(path: Path, **extra):
    return {"path": str(path.resolve()), "sha256": file_hash(path),
            "size_bytes": path.stat().st_size, **extra}


def build_batch(parent: Path, pins: dict[str, str]):
    source_commit = "a" * 40
    records = {}
    write_json(parent / "batch-manifest.json", {
        "schema": "grokking_action_batch_v1", "seeds": list(measurement.SEEDS),
        "source_commit": source_commit,
        "first_seed_is_resource_only_admission": True, "paid_spend_usd": 0})
    batch_accepted = []
    expected = [[policy, step] for policy in measurement.POLICIES
                for step in ((1501,) if policy == "native" else measurement.CAPTURE_STEPS)]
    for seed in measurement.SEEDS:
        seed_dir = parent / f"seed{seed}"
        seed_dir.mkdir()
        parent_checkpoint = {"path": f"/synthetic/seed{seed}/checkpoint-step-001500.pt",
                             "sha256": f"{seed:064x}", "size_bytes": 17}
        record_hash = f"{seed + 10:064x}"
        records[seed, "legacy"] = ({
            "checkpoints": [parent_checkpoint], "config": {"seed": seed, "arm": "legacy"},
            "source_identity": {"fixture": True}, "split_identity": {"seed": seed},
            "parameter_identity": [], "evaluation_rows": []}, record_hash)
        seed_commit = f"{seed:040x}"
        common = {
            "schema": measurement.ACTION_SCHEMA, "seed": seed,
            "parent_checkpoint": parent_checkpoint, "parent_metrics_sha256": record_hash,
            "source_sha256": pins, "source_commit": seed_commit,
            "environment": {"fixture": "same"},
            "fork_scientific_state_sha256": f"{seed + 20:064x}",
            "capture_steps": list(measurement.CAPTURE_STEPS),
            "policies": list(measurement.POLICIES),
            "native_policy_is_one_step_diagnostic_only": True,
        }
        write_json(seed_dir / "manifest.json", common)
        shared_path = seed_dir / "shared-first-action.pt"
        shared_path.write_bytes(b"synthetic shared action")
        diagnostic_path = seed_dir / "first-step-adam-diagnostics.pt"
        diagnostic_path.write_bytes(b"synthetic Adam diagnostics")
        checkpoints = []
        for policy, step in expected:
            path = seed_dir / f"{policy}-step-{step:06d}.pt"
            path.write_bytes(f"synthetic {seed} {policy} {step}".encode())
            checkpoints.append(receipt(path, policy=policy, seed=seed, step=step))
        complete = {**common, "status": "complete", "accepted_roster": expected,
                    "checkpoints": checkpoints,
                    "shared_first_action": receipt(shared_path),
                    "first_step_adam_diagnostics": receipt(diagnostic_path)}
        write_json(seed_dir / "complete.json", complete)
        batch_accepted.append({"seed": seed, "path": str((seed_dir / "complete.json").resolve()),
                               "sha256": file_hash(seed_dir / "complete.json")})
    write_json(parent / "batch-complete.json", {
        "status": "complete", "seeds": list(measurement.SEEDS),
        "accepted": batch_accepted, "paid_spend_usd": 0})
    return records


class ActionMeasurementTest(unittest.TestCase):
    def test_exact_35_state_roster(self):
        roster = measurement.expected_roster()
        self.assertEqual(len(roster), 35)
        self.assertEqual(len(set(roster)), 35)
        self.assertEqual(roster[0], (100, "native", 1501))
        self.assertEqual(roster[-1], (104, "norm_matched", 2500))
        self.assertEqual(sum(policy == "native" for _, policy, _ in roster), 5)

    def test_completed_batch_receipts_and_sources_are_binding(self):
        pins = {"action.py": "1" * 64}
        with tempfile.TemporaryDirectory(
                dir="/tmp/spectral-experiment-artifacts", prefix="action-measure-fixture-") as directory:
            parent = Path(directory)
            records = build_batch(parent, pins)
            admitted = measurement.verify_action_batch(parent, records, pins)
            self.assertEqual(len(admitted["states"]), 35)
            self.assertEqual(admitted["source_sha256"], pins)
            self.assertEqual(admitted["source_commits"]["batch_start"], "a" * 40)
            self.assertEqual(len(set(admitted["source_commits"]["seeds"].values())), 5)
            with self.assertRaisesRegex(ValueError, "source"):
                measurement.verify_action_batch(parent, records, {"action.py": "2" * 64})

            damaged = Path(admitted["states"][7]["checkpoint"]["path"])
            damaged.write_bytes(damaged.read_bytes() + b"corruption")
            with self.assertRaisesRegex(ValueError, "size|hash"):
                measurement.verify_action_batch(parent, records, pins)

    def test_batch_roster_rejection(self):
        pins = {"action.py": "3" * 64}
        with tempfile.TemporaryDirectory(
                dir="/tmp/spectral-experiment-artifacts", prefix="action-roster-fixture-") as directory:
            parent = Path(directory)
            records = build_batch(parent, pins)
            complete_path = parent / "batch-complete.json"
            complete = json.loads(complete_path.read_text())
            complete["accepted"][1]["seed"] = 100
            write_json(complete_path, complete)
            with self.assertRaisesRegex(ValueError, "seed"):
                measurement.verify_action_batch(parent, records, pins)

    def test_envelope_separates_outer_policy_from_inherited_legacy(self):
        seed, policy, step = 100, "orthogonal", 1501
        config = {"seed": seed, "arm": "legacy"}
        source = {"fixture": True}
        split = {"sha256": "split"}
        parameter_layout = [{"name": "weight", "shape": [2], "dtype": "torch.float32"}]
        filter_state = {name: None for name in FILTER_MUTABLE}
        filter_state.update({"step_count": step, "stabilization_count": 0,
                             "max_orthogonality_error": 0.0})
        inner = {
            "schema": measurement.CHECKPOINT_SCHEMA, "step": step,
            "model_state": {}, "optimizer_state": {}, "filter_state": filter_state,
            "filter_configuration": {"stable_update": False},
            "torch_cpu_rng_state": None, "torch_cuda_rng_states": [],
            "device_type": "cpu", "cuda_device_count": 0,
            "config": config, "config_sha256": canonical_hash(config),
            "source_identity": source, "source_identity_sha256": canonical_hash(source),
            "split_identity": split, "parameter_identity": parameter_layout,
            "evaluation_rows": [], "timing": {},
        }
        parent = {"path": "/synthetic/parent.pt", "sha256": "4" * 64}
        shared = {"path": "/synthetic/shared.pt", "sha256": "5" * 64}
        completion = {"source_sha256": {"action": "6" * 64},
                      "parent_checkpoint": parent, "shared_first_action": shared}
        record = {"config": config, "source_identity": source,
                  "split_identity": split, "parameter_identity": parameter_layout}
        item = {"seed": seed, "policy": policy, "step": step,
                "completion": completion, "record": record}
        envelope = {"schema": measurement.ACTION_CHECKPOINT_SCHEMA,
                    "policy": policy, "seed": seed, "step": step,
                    "parent_checkpoint": parent, "source_sha256": completion["source_sha256"],
                    "shared_first_action": shared, "native_compatible_state": inner,
                    "note": "synthetic intervention-safe envelope"}
        self.assertIs(measurement.validate_envelope(envelope, item, split), inner)
        changed = copy.deepcopy(envelope)
        changed["policy"] = "native"
        with self.assertRaisesRegex(ValueError, "outer"):
            measurement.validate_envelope(changed, item, split)
        changed = copy.deepcopy(envelope)
        changed["native_compatible_state"]["split_identity"] = {"sha256": "wrong"}
        with self.assertRaisesRegex(ValueError, "inherited"):
            measurement.validate_envelope(changed, item, split)

    def test_accepted_original_source_pins_are_binding(self):
        acquisition = {"extract.py": "1" * 64, "model.py": "2" * 64}
        analysis = {"analyze.py": "3" * 64, "symmetry.py": "4" * 64}
        scalars = [copy.deepcopy(acquisition) for _ in measurement.SEEDS]
        measurement.validate_original_source_bindings(
            acquisition, scalars, analysis, copy.deepcopy(acquisition),
            copy.deepcopy(analysis))
        changed = copy.deepcopy(scalars)
        changed[3]["extract.py"] = "5" * 64
        with self.assertRaisesRegex(ValueError, "scalars"):
            measurement.validate_original_source_bindings(
                acquisition, changed, analysis, acquisition, analysis)
        changed_acquisition = copy.deepcopy(acquisition)
        changed_acquisition["model.py"] = "7" * 64
        with self.assertRaisesRegex(ValueError, "acquisition sources"):
            measurement.validate_original_source_bindings(
                acquisition, scalars, analysis, changed_acquisition, analysis)
        changed_current = copy.deepcopy(analysis)
        changed_current["symmetry.py"] = "6" * 64
        with self.assertRaisesRegex(ValueError, "analyzer"):
            measurement.validate_original_source_bindings(
                acquisition, scalars, analysis, acquisition, changed_current)

    def test_cuda_environment_contract_is_exact(self):
        prior = {"python": "3.12.3", "torch": "2.11.0+cu128", "numpy": "1.26.4",
                 "device": "cuda", "gpu": "NVIDIA GeForce RTX 3090"}
        flags = {"deterministic_algorithms": False}
        batch = {"python": "3.12.3", "platform": "Linux-fixture",
                 "torch": "2.11.0+cu128", "device": "cuda",
                 "cuda_runtime": "12.8", "cudnn": 91000,
                 "cuda_device_name": "NVIDIA GeForce RTX 3090",
                 "cuda_capability": [8, 6], "torch_cpu_threads": 1,
                 "current_backend_flags": flags}
        measurement.validate_environment_contract(
            prior, copy.deepcopy(prior), batch, copy.deepcopy(prior), copy.deepcopy(batch))
        changed = copy.deepcopy(prior)
        changed["numpy"] = "2.0.0"
        with self.assertRaisesRegex(ValueError, "prior environment"):
            measurement.validate_environment_contract(
                prior, prior, batch, changed, batch)
        changed_batch = copy.deepcopy(batch)
        changed_batch["cuda_device_name"] = "different GPU"
        with self.assertRaisesRegex(ValueError, "action-batch"):
            measurement.validate_environment_contract(
                prior, prior, batch, prior, changed_batch)

    def test_output_is_exclusive_and_bounded(self):
        with tempfile.TemporaryDirectory(
                dir="/tmp/spectral-experiment-artifacts", prefix="action-output-fixture-") as directory:
            target = Path(directory) / "new-output"
            with patch.object(measurement.shutil, "disk_usage") as usage:
                usage.return_value.free = measurement.OUTPUT_LIMIT + measurement.OUTPUT_RESERVE + 1
                output = measurement.Output(target)
                output.write_bytes("manifest.json", b"fixture")
                with self.assertRaises(FileExistsError):
                    output.write_bytes("manifest.json", b"overwrite")
                output.bytes = measurement.OUTPUT_LIMIT
                with self.assertRaises(RuntimeError):
                    output.write_bytes("extra", b"x")


if __name__ == "__main__":
    unittest.main()
