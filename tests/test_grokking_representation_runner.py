"""Small structural fixtures for the observation runner; no scientific states."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from experiments import measure_grokking_representations as runner


def calibration_result(arm, step):
    score = .8 if step == 6000 else .01
    return {"seed": 100, "arm": arm, "step": step, "probes": {
        "final_hidden": {"observed": {"selected_eval_mean_r2": score},
                         "null": {"selected_eval_mean_r2_max": .02}},
        "pre_attention": {"observed": {"selected_eval_mean_r2": .01}}}}


class RepresentationRunnerTest(unittest.TestCase):
    def test_fixed_disjoint_rosters(self):
        first, rest = runner.roster("calibration"), runner.roster("remaining")
        self.assertEqual(len(first), 6)
        self.assertEqual(len(rest), 144)
        self.assertFalse(set(first) & set(rest))
        self.assertEqual(len(set(first + rest)), 150)
        with self.assertRaises(ValueError):
            runner.roster("all")

    def test_output_guard(self):
        for path in (Path("/tmp/not-authorized"), Path("/tmp/spectral-experiment-artifacts"),
                     Path("/tmp/spectral-experiment-artifacts/../../elsewhere")):
            with self.assertRaises(ValueError):
                runner.validate_output_location(path)
        with tempfile.TemporaryDirectory(dir="/tmp/spectral-experiment-artifacts", prefix="grokking-fixture-") as directory:
            target = Path(directory) / "new-output"
            with patch.object(runner.shutil, "disk_usage") as usage:
                usage.return_value.free = 20 * 1024**3
                self.assertEqual(runner.validate_output_location(target), target.resolve())
                usage.return_value.free = 1024
                with self.assertRaises(ValueError):
                    runner.validate_output_location(target)
            with self.assertRaises(ValueError):
                runner.validate_output_location(Path(directory))

    def test_gate_requires_each_arm_and_margin(self):
        records = [calibration_result(arm, step) for _, arm, step in runner.roster("calibration")]
        self.assertTrue(runner.calibration_gate(records)["admitted"])
        for feature, subfield, replacement in (
                ("final_hidden", "observed", .49),
                ("pre_attention", "observed", .7),
                ("final_hidden", "null", .7)):
            changed = copy.deepcopy(records)
            final = next(r for r in changed if r["arm"] == "stable" and r["step"] == 6000)
            key = "selected_eval_mean_r2_max" if subfield == "null" else "selected_eval_mean_r2"
            final["probes"][feature][subfield][key] = replacement
            self.assertFalse(runner.calibration_gate(changed)["admitted"])
        changed = copy.deepcopy(records)
        next(r for r in changed if r["arm"] == "legacy" and r["step"] == 0)["probes"]["final_hidden"]["observed"]["selected_eval_mean_r2"] = .7
        self.assertFalse(runner.calibration_gate(changed)["admitted"])
        with self.assertRaises(ValueError):
            runner.calibration_gate(records[:-1])

    def test_calibration_receipts_and_sources_are_binding(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            pins, environment = {"fixture": "source-pin"}, {"device": "cpu"}
            runner.write_json(path / "manifest.json", {"stage": "calibration", "source_sha256": pins,
                              "recipe": runner.RECIPE, "environment": environment})
            records, receipts = [], []
            for _, arm, step in runner.roster("calibration"):
                result = calibration_result(arm, step)
                raw = path / f"{arm}-{step}.raw-fixture"
                runner.atomic_exclusive(raw, lambda handle: handle.write(b"small non-scientific receipt fixture"))
                result["raw_activations"] = {"path": str(raw), "sha256": runner.file_hash(raw)}
                target = path / f"{arm}-{step}.json"
                runner.write_json(target, result)
                records.append(result)
                receipts.append({"seed": 100, "arm": arm, "step": step,
                                 "path": str(target), "sha256": runner.file_hash(target)})
            runner.write_json(path / "complete.json", {"stage": "calibration", "status": "complete",
                              "source_sha256": pins, "accepted_results": receipts,
                              "calibration_gate": runner.calibration_gate(records)})
            self.assertEqual(runner.verify_calibration(path, pins, environment)["path"], str(path))
            with self.assertRaises(ValueError):
                runner.verify_calibration(path, {"fixture": "different-source"}, environment)
            with self.assertRaises(ValueError):
                runner.verify_calibration(path, pins, {"device": "cuda"})
            with self.assertRaises(FileExistsError):
                runner.write_json(path / "complete.json", {"overwrite": True})

    def test_checkpoint_metadata_and_prefix(self):
        data = (torch.tensor([[0, 0], [1, 1]]), torch.tensor([0, 2]),
                torch.tensor([[0, 1], [1, 0]]), torch.tensor([1, 1]))
        source = {"files": {key: {"sha256": runner.file_hash(runner.REPO / path)} for key, path in (
            ("model", "experiments/grokking_model.py"),
            ("harness", "experiments/grokking_confirmation.py"),
            ("filter", "spectral_filter.py"))}}
        config = {"seed": 100, "arm": "adamw"}
        record = {"config": config, "source_identity": source,
                  "split_identity": runner.tensor_set_identity(data), "parameter_identity": [],
                  "evaluation_rows": [{"step": 0}]}
        state = {"schema": runner.CHECKPOINT_SCHEMA, "step": 0, **record,
                 "config_sha256": runner.canonical_hash(config),
                 "source_identity_sha256": runner.canonical_hash(source)}
        runner.validate_checkpoint(state, record, 100, "adamw", 0, data)
        for key, value in (("source_identity_sha256", "wrong"), ("step", 50),
                           ("evaluation_rows", [{"step": 0}, {"step": 0}])):
            changed = copy.deepcopy(state)
            changed[key] = value
            with self.assertRaises(ValueError):
                runner.validate_checkpoint(changed, record, 100, "adamw", 0, data)


if __name__ == "__main__":
    unittest.main()
