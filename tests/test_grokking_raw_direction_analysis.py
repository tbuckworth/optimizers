"""Tiny JSON/opaque-byte fixtures only: no checkpoint, NPZ or model loading."""
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from experiments import analyze_grokking_raw_direction_results as analysis


def receipt(path):
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "size_bytes": path.stat().st_size}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
    return receipt(path)


def opaque(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"synthetic opaque bytes: deliberately not a checkpoint or NPZ")
    return receipt(path)


def row(seed, policy, step, value=0.):
    probe = {"selected_frequencies": [1, 2, 3, 4, 5], "selected_eval_mean_r2": value,
             "null_max_eval_mean_r2": value + .01,
             "fixed_panel_frequencies": analysis.FIXED_PANEL,
             "fixed_panel_eval_mean_r2": value + .02}
    return {"seed": seed, "policy": policy, "arm": policy, "step": step,
            "behavior": {split: {"loss": value + 2, "accuracy": .5,
                                  "count": 10, "correct_class_margin_mean": value + 1}
                         for split in ("train", "test")},
            "probes": {feature: copy.deepcopy(probe)
                       for feature in ("final_hidden", "pre_attention")},
            "symmetry": {"centered_logit_rms": {"test": value + 1},
                         "heldout_shift_pooled": {"correct": {"value": value + 2},
                                                   "wrong_shift": {"value": value + 3}},
                         "exchange": {"correct": {"value": value + 4}},
                         "training_membership_pooled": {"excess": value + 5}}}


def groups():
    return ([row(seed, policy, step, float(seed - 99))
             for seed, policy, step in analysis.expected_roster()],
            [row(seed, policy, step) for seed, policy, step in analysis.old_roster()],
            [row(seed, "native", step) for seed in analysis.SEEDS for step in (1500, 2000, 2500)])


def fixture(root):
    """Full receipt graph, retaining the real differing scalar/analyzed schemas."""
    new_rows, old_rows, native_rows = groups()
    old_path, batch_path, measured = root / "old", root / "batch", root / "measurement"
    old_sources = {"experiments/analyze_grokking_action_results.py":
                   analysis.file_hash(analysis.REPO / "experiments/analyze_grokking_action_results.py")}
    prior = {"path": str(root / "prior-summary.json"),
             "sha256": analysis.PRIOR_SUMMARY_SHA256, "size_bytes": 1}
    parent_receipts, old_seeds, old_seed_entries = {}, {}, []
    for seed in analysis.SEEDS:
        parent_receipts[seed] = opaque(root / f"parent{seed}.pt")
        complete = {"status": "complete", "seed": seed, "source_sha256": old_sources,
                    "fork_scientific_state_sha256": hashlib.sha256(str(seed).encode()).hexdigest(),
                    "parent_checkpoint": parent_receipts[seed], "parent_metrics_sha256": "1" * 64}
        seed_receipt = {"seed": seed, **write_json(old_path / f"seed{seed}" / "complete.json", complete)}
        # Historical accepted entries have no size_bytes. Preserve this exact
        # schema in both the old batch and new raw archived_reference fields.
        seed_receipt.pop("size_bytes")
        old_seed_entries.append(seed_receipt)
        old_seeds[seed] = {"complete": complete, "completion": seed_receipt}
    old_batch = write_json(old_path / "batch-complete.json",
                           {"status": "complete", "accepted": old_seed_entries})
    recipe_contract = {str(seed): {"probe_split_sha256": "2" * 64,
                                    "structural_array_sha256": {"train_ids": "3" * 64}}
                       for seed in analysis.SEEDS}
    prior_recipe = {"summary": prior, "seeds": recipe_contract}
    for item in old_rows:
        stem = f"{item['seed']}-{item['policy']}-{item['step']}"
        item["raw_receipt"] = opaque(old_path / f"{stem}.npz")
        item["source_scalar_receipt"] = opaque(old_path / f"{stem}-scalar.json")
        item["source_state_receipt"] = opaque(old_path / f"{stem}-state.json")
        checkpoint = opaque(old_path / f"{stem}.pt")
        item["checkpoint_provenance"] = {
            "outer_seed": item["seed"], "outer_policy": item["policy"], "outer_step": item["step"],
            "action_checkpoint": checkpoint, "parent_legacy_checkpoint": parent_receipts[item["seed"]],
            "parent_metrics_sha256": "1" * 64}
        item["source"] = {"source_npz_sha256": item["raw_receipt"]["sha256"],
                          "source_scalar_sha256": item["source_scalar_receipt"]["sha256"],
                          "action_checkpoint_sha256": checkpoint["sha256"]}
    for item in native_rows:
        item.update(arm="legacy", reference_origin="accepted_archived_legacy")
        item["raw_receipt"] = opaque(old_path / f"native-{item['seed']}-{item['step']}.npz")
        item["source"] = {"source_npz_sha256": item["raw_receipt"]["sha256"]}
    old_manifest = {"schema": "grokking_action_measurement_v1", "prior_recipe": prior_recipe,
                    "measurement_source_sha256": old_sources, "action_source_sha256": old_sources,
                    "fixed_frequency_panel": {"frequencies": analysis.FIXED_PANEL},
                    "recipe": {"ridge": .001, "top_k": 5, "n_nulls": 20},
                    "action_batch": {"path": str(old_path), "completion": old_batch}}
    old_manifest_receipt = write_json(old_path / "manifest.json", old_manifest)
    old_complete = {"schema": "grokking_action_measurement_complete_v1", "status": "complete",
                    "state_count": 35, "manifest": old_manifest_receipt, "prior_summary": prior,
                    "measurement_source_sha256": old_sources, "action_source_sha256": old_sources,
                    "input_batch_completion": old_batch}
    old_complete_receipt = write_json(old_path / "complete.json", old_complete)
    summary = {"schema": "grokking_action_analysis_summary_v1", "new_state_count": 35,
               "new_state_rows": old_rows, "archived_native_reference_rows": native_rows,
               "analysis_source_sha256": old_sources,
               "input_receipts": {"archived_summary": prior,
                                  "measurement_manifest": old_manifest_receipt,
                                  "measurement_completion": old_complete_receipt}}
    archived_summary = write_json(old_path / "summary.json", summary)
    pins = {**old_sources, **{name: analysis.file_hash(analysis.REPO / name)
                             for name in analysis.ACQUISITION_PATHS}}
    measurement_pins = {**pins, **{name: analysis.file_hash(analysis.REPO / name)
                                 for name in analysis.MEASUREMENT_PATHS}}
    accepted_seeds, checkpoint_map, seed_metadata = [], {}, {}
    admission = None
    for seed in analysis.SEEDS:
        directory = batch_path / f"seed{seed}"
        metadata = {"schema": "grokking_raw_direction_acquisition_v1", "seed": seed,
                    "policy": analysis.POLICY, "source_sha256": pins, "source_commit": "a" * 40,
                    "environment": {"fixture": True}, "seed100_admission": admission,
                    "capture_steps": list(analysis.CAPTURE_STEPS), "no_reference_policy_execution": True,
                    "parent_checkpoint": parent_receipts[seed], "parent_metrics_sha256": "1" * 64,
                    "fork_scientific_state_sha256": old_seeds[seed]["complete"]["fork_scientific_state_sha256"],
                    "archived_reference": {"batch_completion_sha256": old_batch["sha256"],
                        "seed_completion": old_seeds[seed]["completion"],
                        "fork_scientific_state_sha256": old_seeds[seed]["complete"]["fork_scientific_state_sha256"]}}
        manifest_receipt = write_json(directory / "manifest.json", metadata)
        artifacts = [manifest_receipt, opaque(directory / "first-step-tensors.pt"),
                     write_json(directory / "first-step-summary.json", {"fixture": True})]
        checkpoints = []
        for step in analysis.CAPTURE_STEPS:
            checkpoint = {**opaque(directory / f"{analysis.POLICY}-step-{step:06d}.pt"),
                          "seed": seed, "policy": analysis.POLICY, "step": step}
            checkpoints.append(checkpoint)
            artifacts.extend((checkpoint, write_json(
                directory / f"{analysis.POLICY}-through-{step:06d}.json", {"fixture": True})))
            checkpoint_map[seed, step] = analysis._identity_receipt(checkpoint)
        complete = {**metadata, "status": "complete", "completed_updates": 1000,
                    "accepted_roster": [[analysis.POLICY, step] for step in analysis.CAPTURE_STEPS],
                    "history_steps": list(range(1501, 2501)), "elapsed_seconds": 1.,
                    "artifact_receipts": artifacts, "checkpoints": checkpoints}
        completion_receipt = write_json(directory / "complete.json", complete)
        accepted_seeds.append({"seed": seed, **completion_receipt})
        if seed == 100:
            admission = completion_receipt
        seed_metadata[seed] = (manifest_receipt, completion_receipt)
    batch_metadata = {"schema": "grokking_raw_direction_batch_v1", "seeds": list(analysis.SEEDS),
                      "policy": analysis.POLICY, "source_sha256": pins, "source_commit": "a" * 40,
                      "first_seed_is_resource_only_admission": True, "paid_spend_usd": 0,
                      "batch_seconds_limit": 43200, "seed_seconds_limit": 10800}
    batch_manifest = write_json(batch_path / "batch-manifest.json", batch_metadata)
    batch_complete = write_json(batch_path / "batch-complete.json", {
        **batch_metadata, "status": "complete", "environment": {"fixture": True},
        "batch_manifest": batch_manifest, "elapsed_seconds": 5., "accepted": accepted_seeds})
    accepted_results = []
    for state in new_rows:
        seed, step = state["seed"], state["step"]
        stem = f"seed{seed}-{analysis.POLICY}-step{step:06d}"
        raw = opaque(measured / "raw" / f"{stem}.npz")
        checkpoint = checkpoint_map[seed, step]
        provenance = {"outer_seed": seed, "outer_policy": analysis.POLICY, "outer_step": step,
                      "raw_direction_checkpoint": checkpoint,
                      "raw_direction_seed_manifest": seed_metadata[seed][0],
                      "raw_direction_seed_completion": seed_metadata[seed][1],
                      "acquisition_source_sha256": pins, "parent_legacy_checkpoint": parent_receipts[seed],
                      "parent_metrics_sha256": "1" * 64}
        scalar = {"schema": "grokking_raw_direction_measurement_state_v1", "seed": seed,
                  "arm": analysis.POLICY, "policy": analysis.POLICY, "step": step,
                  "source_sha256": measurement_pins, "raw_activations": raw,
                  "checkpoint_provenance": provenance, "prior_recipe_contract": recipe_contract[str(seed)],
                  "probe_split_sha256": "2" * 64, "behavior": {}, "probes": {}}
        for split in ("train", "test"):
            scalar["behavior"][split] = {key: state["behavior"][split][key]
                                         for key in ("loss", "accuracy", "count")}
        for feature in ("final_hidden", "pre_attention"):
            probe = state["probes"][feature]
            scalar["probes"][feature] = {"observed": {
                "selected_frequencies": probe["selected_frequencies"],
                "selected_eval_mean_r2": probe["selected_eval_mean_r2"],
                "per_frequency": [{"frequency": i, "fit_r2": .1, "eval_r2": .2} for i in range(1, 57)]},
                "null": {"selected_eval_mean_r2_max": probe["null_max_eval_mean_r2"]}}
        scalar_receipt = write_json(measured / "scalars" / f"{stem}.json", scalar)
        state.update(schema="grokking_raw_direction_analyzed_state_v1",
                     checkpoint_provenance=provenance, prior_recipe_contract=recipe_contract[str(seed)],
                     full_symmetry={"fixture": True}, source={
                         "stage": "raw_direction", "source_npz_sha256": raw["sha256"],
                         "source_scalar_sha256": scalar_receipt["sha256"],
                         "raw_direction_checkpoint_sha256": checkpoint["sha256"]})
        state_receipt = write_json(measured / "states" / f"{stem}.json", state)
        accepted_results.append({"seed": seed, "policy": analysis.POLICY, "step": step,
                                 "checkpoint": checkpoint, "raw": raw, "scalar": scalar_receipt,
                                 "analyzed_state": state_receipt})
    manifest = {"schema": "grokking_raw_direction_measurement_v1", "roster": analysis.expected_roster(),
                "acquisition_source_sha256": pins, "measurement_source_sha256": measurement_pins,
                "prior_recipe": prior_recipe, "recipe": old_manifest["recipe"],
                "fixed_frequency_panel": {"frequencies": analysis.FIXED_PANEL},
                "accepted_old_measurement_sources": {"completion": old_complete_receipt,
                    "manifest": old_manifest_receipt, "measurement_source_sha256": old_sources},
                "raw_batch": {"path": str(batch_path), "manifest": batch_manifest,
                              "completion": batch_complete}}
    manifest_receipt = write_json(measured / "manifest.json", manifest)
    write_json(measured / "complete.json", {
        "schema": "grokking_raw_direction_measurement_complete_v1", "status": "complete",
        "state_count": 15, "accepted_results": accepted_results, "manifest": manifest_receipt,
        "prior_summary": prior, "input_batch_completion": batch_complete,
        "acquisition_source_sha256": pins, "measurement_source_sha256": measurement_pins})
    return {"measurement": measured, "archived_summary": archived_summary,
            "old_measurement_sha": old_complete_receipt["sha256"], "old_batch_sha": old_batch["sha256"]}


class TestRawDirectionAnalysis(unittest.TestCase):
    def test_fixed_roster_math_units_favorable_counts(self):
        contrasts = analysis.paired_contrasts(*groups())
        self.assertEqual(len(contrasts), 6)
        self.assertEqual([(r["step"], r["contrast"]) for r in contrasts],
                         [(step, name) for step in (2000, 2500) for name, _, _ in analysis.CONTRASTS])
        metric = contrasts[0]["metrics"]["heldout_cross_entropy"]
        self.assertEqual([r["difference"] for r in metric["paired"]], [1, 2, 3, 4, 5])
        self.assertEqual(metric["mean_difference"], 3.)
        self.assertEqual(metric["sample_sd"], math.sqrt(2.5))
        self.assertEqual(metric["sample_se"], math.sqrt(.5))
        self.assertEqual(metric["favorable_count"], 0)
        self.assertEqual(metric["unit"], "nats_per_example")
        self.assertEqual(contrasts[0]["metrics"]["heldout_accuracy"]["unit"], "fraction")
        self.assertEqual(contrasts[0]["metrics"]["heldout_correct_margin_mean"]["favorable_count"], 5)
        self.assertIsNone(contrasts[0]["metrics"]["final_hidden_null_max_heldout_r2"]["favorable_count"])

    def test_all_five_signs_and_no_equivalence_inference(self):
        new, old, native = groups()
        for item in new:
            item["behavior"]["test"]["loss"] = float(item["seed"] - 100)
        metric = analysis.paired_contrasts(new, old, native)[0]["metrics"]["heldout_cross_entropy"]
        self.assertEqual([r["difference"] for r in metric["paired"]], [-2., -1., 0., 1., 2.])
        self.assertEqual((metric["negative_count"], metric["zero_count"], metric["positive_count"]), (2, 1, 2))
        self.assertEqual(metric["favorable_count"], 2)
        self.assertEqual(metric["mean_difference"], 0.)
        self.assertNotIn("equivalent", metric)

    def test_exact_order_duplicates_missing_and_extra_fail(self):
        for mutate in (lambda rows: rows[:-1], lambda rows: rows + [rows[0]],
                       lambda rows: list(reversed(rows))):
            with self.assertRaisesRegex(ValueError, "roster"):
                new, old, native = groups()
                analysis.paired_contrasts(mutate(new), old, native)

    def test_nonfinite_primary_and_1501_admission_fail(self):
        for bad in (None, True, float("nan"), float("inf")):
            new, old, native = groups()
            new[0]["behavior"]["test"]["loss"] = bad
            with self.assertRaises(ValueError):
                analysis.paired_contrasts(new, old, native)

    def test_undefined_secondary_no_subset_aggregate(self):
        new, old, native = groups()
        new[1]["symmetry"]["heldout_shift_pooled"]["correct"]["value"] = None
        result = analysis.paired_contrasts(new, old, native)[0]["metrics"]["heldout_correct_shift_defect"]
        self.assertEqual(result["defined_count"], 4)
        self.assertIsNone(result["paired"][0]["difference"])
        for key in ("mean_difference", "sample_sd", "sample_se", "positive_count", "favorable_count"):
            self.assertIsNone(result[key])

    def _with_fixture(self, operation):
        with tempfile.TemporaryDirectory() as directory:
            value = fixture(Path(directory))
            with patch.object(analysis, "OLD_MEASUREMENT_SHA256", value["old_measurement_sha"]), \
                    patch.object(analysis, "OLD_BATCH_SHA256", value["old_batch_sha"]):
                prior = analysis.load_archived_summary(Path(value["archived_summary"]["path"]),
                                                       value["archived_summary"]["sha256"])
                operation(value, prior)

    def test_real_transformed_schema_and_opaque_receipts_admitted(self):
        def operation(value, prior):
            self.assertTrue(all(set(item["completion"]) == {"path", "seed", "sha256"}
                                for item in prior["old_seeds"].values()))
            old_seed_paths = {item["completion"]["path"] for item in prior["old_seeds"].values()}
            self.assertEqual(sum(item["path"] in old_seed_paths and "size_bytes" in item
                                 for item in prior["receipts"]), 5)
            admitted = analysis.verify_measurement(value["measurement"], prior)
            self.assertEqual(len(admitted["rows"]), 15)
            self.assertEqual(len(admitted["scalar_copies"]), 15)
            self.assertEqual(len(admitted["state_copies"]), 15)
            self.assertNotIn("full_symmetry", admitted["rows"][0])
            self.assertIn("full_symmetry", json.loads(admitted["state_copies"][0][1]))
            self.assertEqual(len(analysis.paired_contrasts(admitted["rows"], prior["rows"], prior["native_rows"])), 6)
        self._with_fixture(operation)

    def test_archived_seed_receipt_optional_size_and_original_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory) / "seed100"
            wanted = {"fixture": True}
            full = write_json(parent / "complete.json", wanted)
            historical = {"path": full["path"], "seed": 100, "sha256": full["sha256"]}
            original = dict(historical)
            observed = []
            self.assertEqual(analysis._archived_seed_receipt_json(historical, parent, 100, observed), wanted)
            self.assertEqual(historical, original)
            self.assertEqual(observed, [full])
            self.assertEqual(analysis._archived_seed_receipt_json(
                {**historical, "size_bytes": full["size_bytes"]}, parent, 100, []), wanted)

    def test_archived_seed_receipt_wrong_hash_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory) / "seed100"
            full = write_json(parent / "complete.json", {"fixture": True})
            historical = {"path": full["path"], "seed": 100, "sha256": "0" * 64}
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                analysis._archived_seed_receipt_json(historical, parent, 100, [])

    def test_archived_seed_receipt_wrong_parent_or_name_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for path in (root / "seed101" / "complete.json", root / "seed100" / "other.json"):
                full = write_json(path, {"fixture": True})
                historical = {"path": full["path"], "seed": 100, "sha256": full["sha256"]}
                with self.assertRaisesRegex(ValueError, "path/type/parent"):
                    analysis._archived_seed_receipt_json(historical, root / "seed100", 100, [])

    def test_archived_seed_receipt_wrong_optional_size_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory) / "seed100"
            full = write_json(parent / "complete.json", {"fixture": True})
            historical = {"path": full["path"], "seed": 100, "sha256": full["sha256"]}
            for bad in (full["size_bytes"] + 1, -1, True, None, "1"):
                with self.assertRaisesRegex(ValueError, "size"):
                    analysis._archived_seed_receipt_json({**historical, "size_bytes": bad}, parent, 100, [])

    def test_archived_seed_receipt_symlink_or_hardlink_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = write_json(root / "original.json", {"fixture": True})
            for kind in ("symlink", "hardlink"):
                parent = root / kind / "seed100"
                parent.mkdir(parents=True)
                path = parent / "complete.json"
                if kind == "symlink":
                    path.symlink_to(original["path"])
                else:
                    os.link(original["path"], path)
                historical = {"path": str(path), "seed": 100, "sha256": original["sha256"]}
                with self.assertRaisesRegex(ValueError, "path/type/parent"):
                    analysis._archived_seed_receipt_json(historical, parent, 100, [])

    def test_new_receipt_still_requires_size(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory) / "seed100"
            full = write_json(parent / "complete.json", {"fixture": True})
            no_size = {"path": full["path"], "seed": 100, "sha256": full["sha256"]}
            with self.assertRaisesRegex(ValueError, "malformed receipt"):
                analysis._receipt_json(no_size, parent, "complete.json", [])

    def test_mutated_opaque_raw_hash_fails(self):
        def operation(value, prior):
            path = next((value["measurement"] / "raw").iterdir())
            original = path.read_bytes()
            path.write_bytes(b"X" + original[1:])
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                analysis.verify_measurement(value["measurement"], prior)
        self._with_fixture(operation)

    def test_changed_analyzed_shared_metric_fails_even_with_updated_receipts(self):
        def operation(value, prior):
            complete_path = value["measurement"] / "complete.json"
            complete = json.loads(complete_path.read_text())
            accepted = complete["accepted_results"][0]
            path = Path(accepted["analyzed_state"]["path"])
            state = json.loads(path.read_text())
            state["behavior"]["test"]["loss"] += .125
            accepted["analyzed_state"] = write_json(path, state)
            write_json(complete_path, complete)
            with self.assertRaisesRegex(ValueError, "behavioral measurements"):
                analysis.verify_measurement(value["measurement"], prior)
        self._with_fixture(operation)

    def test_checkpoint_receipt_swap_and_failure_marker_fail(self):
        def operation(value, prior):
            complete_path = value["measurement"] / "complete.json"
            complete = json.loads(complete_path.read_text())
            complete["accepted_results"][0]["checkpoint"] = complete["accepted_results"][1]["checkpoint"]
            write_json(complete_path, complete)
            with self.assertRaisesRegex(ValueError, "checkpoint not bound"):
                analysis.verify_measurement(value["measurement"], prior)
            write_json(value["measurement"] / "failure.json", {"status": "failed"})
            with self.assertRaisesRegex(ValueError, "failure marker"):
                analysis.verify_measurement(value["measurement"], prior)
        self._with_fixture(operation)

    def test_incomplete_source_map_fails(self):
        def operation(value, prior):
            manifest_path = value["measurement"] / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["measurement_source_sha256"].pop(analysis.MEASUREMENT_PATHS[-1])
            complete_path = value["measurement"] / "complete.json"
            complete = json.loads(complete_path.read_text())
            complete["measurement_source_sha256"] = manifest["measurement_source_sha256"]
            complete["manifest"] = write_json(manifest_path, manifest)
            write_json(complete_path, complete)
            with self.assertRaisesRegex(ValueError, "full recipe/acquisition union"):
                analysis.verify_measurement(value["measurement"], prior)
        self._with_fixture(operation)

    def test_import_and_help_do_not_import_tensor_or_array_libraries(self):
        command = [sys.executable, "-c", "import sys; from experiments import analyze_grokking_raw_direction_results; "
                   "assert not {'torch', 'numpy'} & set(sys.modules)"]
        subprocess.run(command, cwd=analysis.REPO, check=True, timeout=10)


if __name__ == "__main__":
    unittest.main()
