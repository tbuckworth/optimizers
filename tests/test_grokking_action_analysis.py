import copy
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest

from experiments import analyze_grokking_action_results as action


def _receipt(path: Path) -> dict:
    raw = path.read_bytes()
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(raw).hexdigest(),
            "size_bytes": len(raw)}


def _write_json(path: Path, value: dict) -> dict:
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
    return _receipt(path)


def _row(seed: int, policy: str, step: int, value: float = 0.0) -> dict:
    probe = {
        "selected_frequencies": [1, 2, 3, 4, 5],
        "selected_eval_mean_r2": value,
        "null_max_eval_mean_r2": value + 0.01,
        "fixed_panel_frequencies": list(action.FIXED_PANEL),
        "fixed_panel_eval_mean_r2": value + 0.02,
    }
    return {
        "seed": seed, "arm": policy, "policy": policy, "step": step,
        "behavior": {
            "train": {"loss": value + 1, "accuracy": value + 2, "count": 10,
                      "correct_class_margin_mean": value + 3},
            "test": {"loss": value + 4, "accuracy": value + 5, "count": 20,
                     "correct_class_margin_mean": value + 6},
        },
        "probes": {"final_hidden": copy.deepcopy(probe),
                   "pre_attention": copy.deepcopy(probe)},
        "symmetry": {
            "centered_logit_rms": {"all": value + 7, "train": value + 8,
                                   "test": value + 9},
            "heldout_shift_pooled": {
                "correct": {"value": value + 10},
                "wrong_shift": {"value": value + 11}},
            "exchange": {"correct": {"value": value + 12}},
            "training_membership_pooled": {"excess": value + 13},
        },
    }


def _endpoint_rows():
    new, native = [], []
    for seed in action.SEEDS:
        for step in action.ENDPOINT_STEPS:
            native.append(_row(seed, "native", step, 0.0))
            new.append(_row(seed, "orthogonal", step, float(seed - 99)))
            new.append(_row(seed, "norm_matched", step, float(2 * (seed - 99))))
    return new, native


def _case_paired_contrast_roster_arithmetic_units_and_signs():
    new, native = _endpoint_rows()
    contrasts = action.paired_contrasts(new, native)
    assert [(row["step"], row["contrast"]) for row in contrasts] == [
        (step, name) for step in action.ENDPOINT_STEPS
        for name, _, _ in action.CONTRASTS]
    metric = contrasts[0]["metrics"]["heldout_cross_entropy"]
    assert [row["difference"] for row in metric["paired"]] == [1, 2, 3, 4, 5]
    assert metric["mean_difference"] == 3
    assert metric["sample_sd"] == math.sqrt(2.5)
    assert metric["sample_se"] == math.sqrt(0.5)
    assert metric["positive_count"] == 5
    assert metric["negative_count"] == metric["zero_count"] == 0
    assert metric["unit"] == "nats_per_example"
    assert metric["favorable_direction"] == "lower"


def _case_primary_missing_or_nonfinite_fails_closed():
    new, native = _endpoint_rows()
    new[0]["behavior"]["test"]["loss"] = None
    with unittest.TestCase().assertRaisesRegex(
            ValueError, "missing/non-numeric required metric"):
        action.paired_contrasts(new, native)
    new, native = _endpoint_rows()
    new[0]["probes"]["final_hidden"]["selected_eval_mean_r2"] = float("inf")
    with unittest.TestCase().assertRaisesRegex(ValueError, "nonfinite required metric"):
        action.paired_contrasts(new, native)


def _case_undefined_energy_ratio_is_preserved_without_subset_aggregate():
    new, native = _endpoint_rows()
    new[0]["symmetry"]["heldout_shift_pooled"]["correct"]["value"] = None
    contrasts = action.paired_contrasts(new, native)
    metric = contrasts[0]["metrics"]["heldout_correct_shift_defect"]
    assert metric["paired"][0]["difference"] is None
    assert metric["defined_count"] == 4
    assert metric["complete_five_seed_aggregate"] is False
    assert metric["mean_difference"] is None
    assert metric["sample_sd"] is None and metric["sample_se"] is None
    assert metric["positive_count"] is None


def _case_duplicate_or_missing_endpoint_is_rejected():
    new, native = _endpoint_rows()
    with unittest.TestCase().assertRaisesRegex(ValueError, "duplicate or lacks"):
        action.paired_contrasts(new[:-1], native)
    with unittest.TestCase().assertRaisesRegex(ValueError, "duplicate or lacks"):
        action.paired_contrasts(new + [new[0]], native)


def _prior_summary_fixture(path: Path) -> tuple[Path, str]:
    rows, raw_receipts = [], []
    for seed, arm, step in action.expected_prior_roster():
        row = _row(seed, arm, step, seed + step / 10000)
        row.pop("policy")
        sha = hashlib.sha256(f"{seed}-{arm}-{step}".encode()).hexdigest()
        row["source"] = {"source_npz_sha256": sha}
        rows.append(row)
        raw_receipts.append({"seed": seed, "arm": arm, "step": step,
                             "path": f"/archived/{seed}-{arm}-{step}.npz",
                             "sha256": sha, "size_bytes": 1})
    summary = {
        "schema": "grokking_representation_analysis_summary_v1",
        "fixed_frequency_panel": {"frequencies": list(action.FIXED_PANEL)},
        "rows": rows,
        "input_receipts": {"calibration": {"raw_receipts": raw_receipts[:6]},
                           "remaining": {"raw_receipts": raw_receipts[6:]}},
        "analysis_source_sha256": {"old": "pin"},
    }
    receipt = _write_json(path, summary)
    return path, receipt["sha256"]


def _case_prior_summary_exact_roster_and_raw_references(tmp_path):
    path, sha = _prior_summary_fixture(tmp_path / "summary.json")
    prior = action.load_prior_summary(path, sha)
    assert len(prior["rows"]) == 15
    assert [(row["seed"], row["step"]) for row in prior["rows"]] == [
        (seed, step) for seed in action.SEEDS for step in action.REFERENCE_STEPS]
    assert all(row["reference_origin"] == "accepted_archived_legacy"
               and row["raw_receipt"]["sha256"] == row["source"]["source_npz_sha256"]
               for row in prior["rows"])
    broken = json.loads(path.read_text())
    broken["rows"].pop()
    broken_path = tmp_path / "broken.json"
    broken_receipt = _write_json(broken_path, broken)
    with unittest.TestCase().assertRaisesRegex(ValueError, "roster"):
        action.load_prior_summary(broken_path, broken_receipt["sha256"])


def _measurement_fixture(path: Path) -> Path:
    batch = path.parent / "action-batch"
    batch.mkdir()
    batch_manifest = _write_json(batch / "batch-manifest.json", {"synthetic": True})
    batch_complete = _write_json(batch / "batch-complete.json", {"status": "complete"})
    for directory in (path, path / "raw", path / "scalars", path / "states"):
        directory.mkdir()
    source_path = "experiments/analyze_grokking_action_results.py"
    source_map = {source_path: action.file_hash(action.REPO / source_path)}
    accepted = []
    for seed, policy, step in action.expected_new_roster():
        stem = f"seed{seed}-{policy}-step{step:06d}"
        raw_path = path / "raw" / f"{stem}.npz"
        raw_path.write_bytes(f"opaque-{stem}".encode())
        raw = _receipt(raw_path)
        checkpoint = {"path": f"/checkpoint/{stem}.pt",
                      "sha256": hashlib.sha256(stem.encode()).hexdigest(),
                      "size_bytes": 1}
        provenance = {"outer_seed": seed, "outer_policy": policy, "outer_step": step,
                      "action_checkpoint": checkpoint}
        scalar = _row(seed, policy, step, 0.0)
        scalar.update({"schema": "grokking_action_measurement_state_v1",
                       "source_sha256": source_map, "raw_activations": raw,
                       "checkpoint_provenance": provenance})
        scalar_receipt = _write_json(path / "scalars" / f"{stem}.json", scalar)
        state = _row(seed, policy, step, 0.0)
        state.update({"checkpoint_provenance": provenance,
                      "source": {"stage": "action",
                                 "source_scalar_sha256": scalar_receipt["sha256"],
                                 "source_npz_sha256": raw["sha256"],
                                 "action_checkpoint_sha256": checkpoint["sha256"]},
                      "full_symmetry": {"preserved": True}})
        state_receipt = _write_json(path / "states" / f"{stem}.json", state)
        accepted.append({"seed": seed, "policy": policy, "step": step,
                         "checkpoint": checkpoint, "raw": raw, "scalar": scalar_receipt,
                         "analyzed_state": state_receipt})
    prior = {"path": str(action.PRIOR_SUMMARY.resolve()),
             "sha256": action.PRIOR_SUMMARY_SHA256, "size_bytes": 1}
    manifest = {
        "schema": "grokking_action_measurement_v1",
        "roster": action.expected_new_roster(),
        "fixed_frequency_panel": {"frequencies": list(action.FIXED_PANEL)},
        "measurement_source_sha256": source_map,
        "action_source_sha256": source_map,
        "prior_recipe": {"summary": prior},
        "action_batch": {"path": str(batch.resolve()), "manifest": batch_manifest,
                         "completion": batch_complete},
    }
    manifest_receipt = _write_json(path / "manifest.json", manifest)
    complete = {
        "schema": "grokking_action_measurement_complete_v1", "status": "complete",
        "state_count": 35, "accepted_results": accepted, "manifest": manifest_receipt,
        "measurement_source_sha256": source_map, "action_source_sha256": source_map,
        "prior_summary": prior, "input_batch_completion": batch_complete,
    }
    _write_json(path / "complete.json", complete)
    return path


def _case_measurement_admission_preserves_35_and_checks_raw_receipts(tmp_path):
    path = _measurement_fixture(tmp_path / "measurement")
    admitted = action.verify_measurement(path)
    assert len(admitted["rows"]) == 35
    assert len(admitted["scalar_copies"]) == len(admitted["state_copies"]) == 35
    assert len(admitted["receipts"]) == 105
    raw = Path(admitted["rows"][0]["raw_receipt"]["path"])
    original = raw.read_bytes()
    raw.write_bytes(bytes([original[0] ^ 1]) + original[1:])
    with unittest.TestCase().assertRaisesRegex(ValueError, "hash mismatch"):
        action.verify_measurement(path)


def _case_source_map_rejects_current_source_corruption_shape():
    relative = "experiments/analyze_grokking_action_results.py"
    with unittest.TestCase().assertRaisesRegex(ValueError, "source hash mismatch"):
        action._verify_source_map({relative: "0" * 64})
    assert action._verify_source_map(
        {relative: action.file_hash(action.REPO / relative)})[relative]


class TestGrokkingActionAnalysis(unittest.TestCase):
    def test_paired_contrast_roster_arithmetic_units_and_signs(self):
        _case_paired_contrast_roster_arithmetic_units_and_signs()

    def test_primary_missing_or_nonfinite_fails_closed(self):
        _case_primary_missing_or_nonfinite_fails_closed()

    def test_undefined_energy_ratio_is_preserved_without_subset_aggregate(self):
        _case_undefined_energy_ratio_is_preserved_without_subset_aggregate()

    def test_duplicate_or_missing_endpoint_is_rejected(self):
        _case_duplicate_or_missing_endpoint_is_rejected()

    def test_prior_summary_exact_roster_and_raw_references(self):
        with tempfile.TemporaryDirectory() as directory:
            _case_prior_summary_exact_roster_and_raw_references(Path(directory))

    def test_measurement_admission_preserves_35_and_checks_raw_receipts(self):
        with tempfile.TemporaryDirectory() as directory:
            _case_measurement_admission_preserves_35_and_checks_raw_receipts(
                Path(directory))

    def test_source_map_rejects_current_source_corruption_shape(self):
        _case_source_map_rejects_current_source_corruption_shape()


if __name__ == "__main__":
    unittest.main()
