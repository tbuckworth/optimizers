"""Synthetic stdlib-only tests for the independent I16 report corroborator."""
from __future__ import annotations

import copy
import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import corroborate_report as report


def _raw(value) -> bytes:
    return json.dumps(value, separators=(",", ":"), allow_nan=False).encode() + b"\n"


def _record(name, value=None, raw=None):
    payload = raw if raw is not None else _raw(value)
    return {"name": name, "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest()}


def _write_archive(directory: Path, schema: str, objects: dict, **extra):
    directory.mkdir()
    rows = []
    for original in sorted(objects):
        raw = _raw(objects[original])
        packed = gzip.compress(raw, compresslevel=9, mtime=0)
        target = directory / (original + ".gz")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(packed)
        rows.append({"original": original, "original_bytes": len(raw),
            "original_sha256": hashlib.sha256(raw).hexdigest(),
            "archive": original + ".gz", "archive_bytes": len(packed),
            "archive_sha256": hashlib.sha256(packed).hexdigest()})
    manifest = {"schema": schema, "status": "complete", **extra,
        "json_count": len(rows),
        "original_bytes": sum(row["original_bytes"] for row in rows),
        "archive_bytes": sum(row["archive_bytes"] for row in rows), "files": rows}
    (directory / "collection.json").write_bytes(_raw(manifest))
    return report.sha256_path(directory / "collection.json")


def _point(seed, target, policy, horizon):
    p = report.POLICIES.index(policy)
    offset = (seed - 199) * .013 + (target == "fixed") * .07 + p * .019
    # Distinct optima exercise both selectors and the k tie order.
    ce_h = report.HORIZONS[(p + seed) % len(report.HORIZONS)]
    acc_h = report.HORIZONS[(p + seed + 2) % len(report.HORIZONS)]

    def basic(ce, accuracy):
        return {"clean_ce": ce, "clean_accuracy": accuracy,
            "mean_max_probability": .6 + offset / 20,
            "mean_true_label_probability": .5 + offset / 30}

    train = basic(1.2 + offset + horizon / 1e6, .6 + offset / 20)
    train.update({"soft_ce": train["clean_ce"] + .2,
                  "fixed_ce": train["clean_ce"] + .3,
                  "fixed_minus_soft_ce": .1,
                  "fixed_accuracy": train["clean_accuracy"] - .1})
    validation = basic(1 + offset + abs(horizon - ce_h) / 1e5,
                       .7 + offset / 20 - abs(horizon - acc_h) / 1e5)
    auxiliary = basic(1.1 + offset + horizon / 1e6, .68 + offset / 20)
    return {"horizon": horizon, "train": train,
            "validation": validation, "auxiliary": auxiliary}


def _steps(seed, policy, scalar=False):
    base = 1 + (seed - 200) * .1 + report.POLICIES.index(policy) * .05
    rows = []
    for step, multiplier in ((101, 1.0), (1001, 2.0)):
        displacement = {
            "data_norm": 2 * base * multiplier,
            "total_norm": 3 * base * multiplier,
            "raw_gradient_dot_data_delta": -base * multiplier,
            "post_mean_dot_data_delta": (-.5 * base * multiplier
                                           if policy != "k1" else None),
            "applied_gradient_dot_data_delta": (-.7 * base * multiplier
                                                  if policy != "k1" else None),
            "data_current_basis_leakage": {"reason": None,
                "outside_squared_norm": base * multiplier,
                "squared_norm": (2 * base * multiplier) ** 2},
            "total_current_basis_leakage": {"reason": None,
                "outside_squared_norm": 2 * base * multiplier,
                "squared_norm": (3 * base * multiplier) ** 2}}
        row = {"step": step, "displacement": displacement}
        if scalar:
            row["components"] = {}
            for index, name in enumerate(("raw_gradient", "post_mean",
                    "native_projection_diagnostic", "applied_delivery",
                    "old_momentum_buffer", "scaled_old_momentum_buffer",
                    "new_momentum_buffer"), start=1):
                energy = base * multiplier * index
                row["components"][name] = {"squared_energy": energy,
                    "current_basis_leakage": {"reason": None,
                        "outside_squared_norm": energy / 4,
                        "squared_norm": energy}}
        rows.append(row)
    return rows


class CorroboratorTests(unittest.TestCase):
    def fixture(self, root: Path):
        i14_source, i15_source, i16_source = ("/synthetic/i14", "/synthetic/i15",
                                               "/synthetic/i16")
        i14_objects, i14_artifacts, i14_refs = {}, [], []
        for seed in report.SEEDS:
            for target in report.TARGETS:
                for old_policy, policy in (("raw", "k1"), ("current32", "k1")):
                    identity = f"s{seed}-sgdm-{target}-{old_policy}"
                    name = "curve-" + identity + ".json"
                    curve = {"schema": "i14_trajectory_v1", "id": identity,
                        "seed": seed, "target": target, "policy": old_policy,
                        "base": "sgdm", "lr": .03, "status": "complete",
                        "curve": [{"horizon": 0}] + [_point(seed, target, policy, h)
                                                           for h in report.HORIZONS],
                        "steps": _steps(seed, "k1")}
                    i14_objects["confirmation/" + name] = curve
                    record = _record(name, curve)
                    i14_artifacts.append(record)
                    i14_refs.append({"seed": seed, "target": target,
                        "policy": old_policy, "curve_artifact": record,
                        "checkpoint_records": [], "warmup_state_digest": "1" * 64})
        i14_completion = {"schema": "i14_completion_v1", "status": "complete",
                          "phase": "confirmation", "artifacts": i14_artifacts}
        i14_objects["confirmation/completion.json"] = i14_completion
        i14_completion_sha = _record("completion.json", i14_completion)["sha256"]
        i14_dir = root / "i14"
        i14_sha = _write_archive(i14_dir, "i14_scalar_collection_v1", i14_objects,
            source_directory=i14_source,
            source_completion_sha256={"confirmation": i14_completion_sha},
            runtime_directory_supplement_sha256="2" * 64)

        i15_objects, i15_artifacts, spectral_refs = {}, [], []
        for seed in report.SEEDS:
            for target in report.TARGETS:
                identity = f"s{seed}-sgdm-{target}-{report.SPECTRAL}"
                checkpoints, projected = [], []
                for horizon in report.HORIZONS[1:]:
                    kind = "full_state" if horizon == 2000 else "model_state"
                    state_record = {"name": f"{kind}-{identity}-h{horizon}.pt",
                                    "bytes": 1, "sha256": "3" * 64}
                    i15_artifacts.append(state_record)
                    checkpoints.append({"horizon": horizon,
                                        "relative_horizon": horizon - 100,
                                        kind: state_record})
                    projected.append({"horizon": horizon, kind: state_record})
                branch = {"schema": "i15_history_branch_v1", "id": identity,
                    "seed": seed, "target": target,
                    "policy": report.SPECTRAL, "base": "sgdm", "lr": .03,
                    "status": "complete", "failure": None,
                    "completed_updates": 1900,
                    "parent_state_digest": "4" * 64,
                    "parent_evaluation_digest": "5" * 64,
                    "curve": [_point(seed, target, report.SPECTRAL, h)
                              for h in report.HORIZONS],
                    "steps": _steps(seed, report.SPECTRAL),
                    "checkpoints": checkpoints}
                name = "branch-" + identity + ".json"
                i15_objects["confirmation/" + name] = branch
                branch_record = _record(name, branch)
                i15_artifacts.append(branch_record)
                spectral_refs.append({"seed": seed, "target": target,
                    "curve_artifact": branch_record, "checkpoint_records": projected,
                    "parent_state_digest": "4" * 64,
                    "parent_evaluation_digest": "5" * 64})
        i15_completion = {"schema": "i15_completion_v1", "status": "complete",
                          "phase": "confirmation", "artifacts": i15_artifacts}
        i15_objects["confirmation/completion.json"] = i15_completion
        i15_completion_sha = _record("completion.json", i15_completion)["sha256"]
        i15_dir = root / "i15"
        i15_audit_sha = "6" * 64
        i15_sha = _write_archive(i15_dir, "i15_scalar_collection_v1", i15_objects,
            source_root=i15_source, analysis_audit={"status": "pass",
                                                    "sha256": i15_audit_sha},
            phase_completion_sha256={"confirmation": i15_completion_sha})

        raw_refs = []
        for seed in report.SEEDS:
            for target in report.TARGETS:
                name = f"curve-s{seed}-sgdm-{target}-raw.json"
                raw_refs.append({"seed": seed, "target": target,
                    "curve": i14_objects["confirmation/" + name]})
        i14_binding = {"schema": "i15_i14_inputs_v1", "source_root": i14_source,
            "runtime_supplement_sha256": "2" * 64,
            "confirmation_completion_sha256": i14_completion_sha,
            "references": i14_refs}
        binding = {"schema": "i16_parent_and_reference_binding_v1",
            "i14": i14_binding, "i15": {"artifact_root": i15_source,
                "audit_sha256": i15_audit_sha,
                "summary_sha256": "a" * 64,
                "collection_sha256": i15_sha,
                "confirmation_completion_sha256": i15_completion_sha},
            "raw_references": raw_refs, "spectral_references": spectral_refs,
            "source_replayed": False}

        i16_objects, i16_artifacts, entries = {
            "confirmation/parent-inputs.json": binding}, [], []
        i16_artifacts.append(_record("parent-inputs.json", binding))
        for seed in report.SEEDS:
            for target in report.TARGETS:
                for label in report.NEW_LABELS:
                    identity = f"s{seed}-sgdm-{target}-{label}"
                    branch = {"schema": "i16_scalar_branch_v1", "id": identity,
                        "seed": seed, "target": target,
                        "k": report.K_BY_LABEL[label], "k_label": label,
                        "status": "complete", "completed_updates": 1900,
                        "parent_state_digest": "7" * 64,
                        "parent_evaluation_digest": "8" * 64,
                        "first_step_digests": {"raw_gradient": "9" * 64},
                        "scalar_update_seconds": 2.0,
                        "curve": [_point(seed, target, label, h)
                                  for h in report.HORIZONS],
                        "steps": _steps(seed, label, scalar=True), "failure": None}
                    name = "branch-" + identity + ".json"
                    i16_objects["confirmation/" + name] = branch
                    artifact = _record(name, branch)
                    i16_artifacts.append(artifact)
                    entries.append({key: branch[key] for key in (
                        "id", "seed", "target", "k", "k_label", "status",
                        "completed_updates", "parent_state_digest",
                        "parent_evaluation_digest", "first_step_digests",
                        "scalar_update_seconds")} | {"artifact": artifact})
        confirmation_index = {"entries": entries}
        i16_objects["confirmation/branches.json"] = confirmation_index
        i16_artifacts.append(_record("branches.json", confirmation_index))
        confirmation_manifest = {"smoke_based_seconds_forecast":
                                 .6 / 60 * 34200 * 1.5 + 60}
        i16_objects["confirmation/manifest.json"] = confirmation_manifest
        i16_artifacts.append(_record("manifest.json", confirmation_manifest))
        confirmation_completion = {"schema": "i16_completion_v1",
            "status": "complete", "phase": "confirmation", "artifacts": i16_artifacts,
            "scalar_update_seconds": 36.0, "completed_training_updates": 34200}
        i16_objects["confirmation/completion.json"] = confirmation_completion
        confirmation_sha = _record("completion.json", confirmation_completion)["sha256"]

        smoke_entries = [{"completed_updates": 10, "scalar_update_seconds": .1}
                         for _ in range(6)]
        smoke_index = {"entries": smoke_entries}
        smoke_manifest = {"phase": "smoke"}
        smoke_artifacts = [_record("branches.json", smoke_index),
                           _record("manifest.json", smoke_manifest)]
        smoke_completion = {"schema": "i16_completion_v1", "status": "complete",
            "phase": "smoke", "artifacts": smoke_artifacts,
            "scalar_update_seconds": .6, "completed_training_updates": 260}
        i16_objects.update({"smoke/branches.json": smoke_index,
                            "smoke/manifest.json": smoke_manifest,
                            "smoke/completion.json": smoke_completion})
        smoke_sha = _record("completion.json", smoke_completion)["sha256"]

        analysis = root / "analysis-001"
        analysis.mkdir()
        phase_shas = {"smoke": smoke_sha, "confirmation": confirmation_sha}
        attempt_shas = {"smoke": "b" * 64, "confirmation": "c" * 64}
        runtime_inventory = {"declared_name": "runtime", "entries": []}
        audit = {"schema": "i16_scalar_analysis_audit_v1", "status": "pass",
                 "artifact_root": i16_source,
                 "phase_completion_sha256": phase_shas,
                 "attempt_sha256": attempt_shas,
                 "runtime_inventory": runtime_inventory}
        audit_path = analysis / "audit.json"
        audit_path.write_bytes(_raw(audit))
        audit_sha = report.sha256_path(audit_path)
        i16_dir = root / "i16"
        i16_sha = _write_archive(i16_dir, "i16_scalar_collection_v1", i16_objects,
            source_root=i16_source,
            analysis_audit={"status": "pass", "sha256": audit_sha},
            phase_completion_sha256=phase_shas, attempt_sha256=attempt_shas,
            runtime_inventory=runtime_inventory)

        archives = (report.Archive(i16_dir, i16_sha, "i16_scalar_collection_v1"),
                    report.Archive(i15_dir, i15_sha, "i15_scalar_collection_v1"),
                    report.Archive(i14_dir, i14_sha, "i14_scalar_collection_v1"))
        branches = report.load_branches(*archives)
        choices, primary, per_k = report.selection_results(branches)
        paths, components = report.geometry_results(branches)
        summary = {"schema": "i16_scalar_analysis_summary_v1",
            "artifact_root": i16_source, "audit_status": "pass",
            "primary_validation_selected_spectral_minus_scalar": primary,
            "validation_selected_choices": choices,
            "validation_selected_spectral_minus_each_k": per_k,
            "fixed_k_endpoint_spectral_minus_scalar": report.endpoint_effects(branches),
            "all_policy_trajectories": report.trajectory_rows(branches),
            "scheduled_metric_equal_seed_aggregates":
                report.scheduled_aggregates(branches),
            "path_and_displacement_geometry": paths,
            "component_energy_and_action_complement": components}
        summary_path = analysis / "summary.json"
        summary_path.write_bytes(_raw(summary))
        return {"i16": i16_dir, "i16_sha": i16_sha,
            "i15": i15_dir, "i15_sha": i15_sha,
            "i14": i14_dir, "i14_sha": i14_sha,
            "summary": summary_path, "summary_sha": report.sha256_path(summary_path),
            "audit": audit_path, "audit_sha": audit_sha,
            "output": analysis / "report-audit.json"}

    def run_fixture(self, f):
        return report.corroborate(f["i16"], f["i16_sha"], f["i14"], f["i14_sha"],
            f["i15"], f["i15_sha"], f["summary"], f["summary_sha"],
            f["audit"], f["audit_sha"], f["output"])

    def test_complete_fixture_corroborates_all_registered_sections(self):
        with tempfile.TemporaryDirectory() as directory:
            f = self.fixture(Path(directory))
            result = self.run_fixture(f)
            self.assertEqual(result["status"], "pass")
            self.assertEqual(result["complete_trajectories"], 30)
            self.assertEqual(result["retained_scheduled_points"], 180)
            self.assertEqual(result["maximum_absolute_numeric_difference"], 0)
            self.assertEqual({name: row["rows"] for name, row in
                              result["sections"].items()}, {
                "primary_validation_selected_spectral_minus_scalar": 8,
                "validation_selected_choices": 12,
                "validation_selected_spectral_minus_each_k": 32,
                "fixed_k_endpoint_spectral_minus_scalar": 16,
                "all_policy_trajectories": 30,
                "scheduled_metric_equal_seed_aggregates": 960,
                "path_and_displacement_geometry": 20,
                "component_energy_and_action_complement": 84})
            self.assertEqual(result["timing_checks"]["status"], "pass")

    def test_joint_selection_ties_use_earliest_horizon_then_smaller_k(self):
        branches = {}
        for seed in report.SEEDS:
            for target in report.TARGETS:
                for label in report.POLICIES:
                    curve = [_point(seed, target, label, h) for h in report.HORIZONS]
                    for point in curve:
                        point["validation"]["clean_ce"] = 1.0
                        point["validation"]["clean_accuracy"] = .5
                    branches[seed, target, label] = {"status": "complete",
                        "curve": curve, "steps": [], "failure": None}
        choices, _, _ = report.selection_results(branches)
        self.assertTrue(all(row["scalar"]["k"] == 0.0
                            and row["scalar"]["horizon"] == 100
                            and row["spectral"]["horizon"] == 100 for row in choices))

    def test_missing_scalar_member_makes_joint_effect_unavailable(self):
        branches = {}
        for seed in report.SEEDS:
            for target in report.TARGETS:
                for label in report.POLICIES:
                    branches[seed, target, label] = {"status": "complete",
                        "curve": [_point(seed, target, label, h)
                                  for h in report.HORIZONS], "steps": [], "failure": None}
        branches[201, "fixed", "k0p5"]["status"] = "numerical_failure"
        _, primaries, _ = report.selection_results(branches)
        affected = [row for row in primaries if row["target"] == "fixed"]
        self.assertTrue(all(row["effect"]["per_seed"]["201"] is None
                            and row["effect"]["mean"] is None
                            and not row["effect"]["available"] for row in affected))

    def test_passing_audit_and_archive_hashes_are_mandatory_before_output(self):
        with tempfile.TemporaryDirectory() as directory:
            f = self.fixture(Path(directory))
            audit = json.loads(f["audit"].read_text())
            audit["status"] = "fail"
            f["audit"].write_bytes(_raw(audit))
            f["audit_sha"] = report.sha256_path(f["audit"])
            with self.assertRaisesRegex(ValueError, "audit is not passing"):
                self.run_fixture(f)
            self.assertFalse(f["output"].exists())

        with tempfile.TemporaryDirectory() as directory:
            f = self.fixture(Path(directory))
            with self.assertRaisesRegex(ValueError, "manifest differs"):
                report.corroborate(f["i16"], "0" * 64, f["i14"], f["i14_sha"],
                    f["i15"], f["i15_sha"], f["summary"], f["summary_sha"],
                    f["audit"], f["audit_sha"], f["output"])
            self.assertFalse(f["output"].exists())

    def test_summary_difference_writes_failing_exclusive_report(self):
        with tempfile.TemporaryDirectory() as directory:
            f = self.fixture(Path(directory))
            value = json.loads(f["summary"].read_text())
            value["primary_validation_selected_spectral_minus_scalar"][0][
                "effect"]["mean"] += .25
            f["summary"].write_bytes(_raw(value))
            f["summary_sha"] = report.sha256_path(f["summary"])
            result = self.run_fixture(f)
            self.assertEqual(result["status"], "fail")
            self.assertTrue(result["errors"])
            with self.assertRaises(FileExistsError):
                self.run_fixture(f)


if __name__ == "__main__":
    unittest.main()
