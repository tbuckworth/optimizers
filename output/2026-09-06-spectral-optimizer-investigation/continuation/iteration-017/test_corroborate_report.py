"""Synthetic standard-library tests for the I17 report corroborator."""
from __future__ import annotations

import copy
import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import corroborate_report as report


def _raw(value):
    return json.dumps(value, separators=(",", ":"), allow_nan=False).encode() + b"\n"


def _record(name, value):
    raw = _raw(value)
    return {"name": name, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def _write_archive(path, schema, objects, **extra):
    path.mkdir()
    rows = []
    for original, value in sorted(objects.items()):
        raw = _raw(value)
        packed = gzip.compress(raw, compresslevel=9, mtime=0)
        member = path / (original + ".gz")
        member.parent.mkdir(parents=True, exist_ok=True)
        member.write_bytes(packed)
        rows.append({"original": original, "original_bytes": len(raw),
                     "original_sha256": hashlib.sha256(raw).hexdigest(),
                     "archive": original + ".gz", "archive_bytes": len(packed),
                     "archive_sha256": hashlib.sha256(packed).hexdigest()})
    manifest = {"schema": schema, "status": "complete", **extra,
                "json_count": len(rows),
                "original_bytes": sum(row["original_bytes"] for row in rows),
                "archive_bytes": sum(row["archive_bytes"] for row in rows),
                "files": rows}
    (path / "collection.json").write_bytes(_raw(manifest))
    return report.sha256_path(path / "collection.json")


def _point(seed, target, policy, horizon):
    offset = (seed - 200) * .01 + report.POLICIES.index(policy) * .03 \
        + (target == "fixed") * .1
    optimum = report.HORIZONS[(seed + report.POLICIES.index(policy)) % 6]

    def basic(ce, acc):
        return {"clean_ce": ce, "clean_accuracy": acc,
                "mean_max_probability": .6 + offset / 10,
                "mean_true_label_probability": .5 + offset / 10}

    train = basic(1.2 + offset + horizon / 1e5, .6 + offset)
    train.update({"soft_ce": train["clean_ce"] + .2,
                  "fixed_ce": train["clean_ce"] + .3,
                  "fixed_minus_soft_ce": .1,
                  "fixed_accuracy": train["clean_accuracy"] - .1})
    return {"horizon": horizon, "train": train,
            "validation": basic(1 + offset + abs(horizon - optimum) / 1e5,
                                .7 + offset - abs(horizon - optimum) / 1e5),
            "auxiliary": basic(1.1 + offset + horizon / 1e6,
                               .68 + offset - horizon / 1e7)}


def _steps(seed, policy):
    scale = 1 + (seed - 200) * .1 + report.POLICIES.index(policy) * .05
    rows = []
    for step in range(101, 2001):
        norm = scale * (1 + step / 10000)
        leakage = {"reason": None, "outside_squared_norm": norm * norm / 4,
                   "squared_norm": norm * norm}
        displacement = {
            "data_norm": norm, "total_norm": 2 * norm,
            "data_current_basis_leakage": leakage,
            "total_current_basis_leakage": {**leakage,
                "outside_squared_norm": norm * norm / 2,
                "squared_norm": 4 * norm * norm},
            "raw_gradient_dot_data_delta": -norm,
            "post_mean_dot_data_delta": -.5 * norm}
        if policy != "scalar_k0":
            displacement["ideal_data_norm"] = .9 * norm
            displacement["ideal_data_current_basis_leakage"] = {**leakage,
                "outside_squared_norm": norm * norm / 8,
                "squared_norm": .81 * norm * norm}
        rows.append({"step": step, "displacement": displacement})
    return rows


def _branch(seed, target, policy, schema):
    return {"schema": schema, "id": f"s{seed}-sgdm-{target}-{policy}",
            "seed": seed, "target": target,
            "policy": policy if schema.startswith("i17") else None,
            "k": 0.0 if schema.startswith("i16") else None,
            "k_label": "k0" if schema.startswith("i16") else None,
            "status": "complete", "completed_updates": 1900,
            "parent_state_digest": "1" * 64,
            "parent_evaluation_digest": "2" * 64,
            "first_step_digests": {"raw_gradient": "3" * 64},
            "update_seconds": 1.0,
            "curve": [_point(seed, target, policy, h) for h in report.HORIZONS],
            "steps": _steps(seed, policy), "checkpoints": [], "failure": None}


class CorroboratorTests(unittest.TestCase):
    def fixture(self, root):
        source16, source17 = "/synthetic/i16", "/synthetic/i17"
        i16_objects, i16_artifacts, references = {}, [], []
        for seed in report.SEEDS:
            for target in report.TARGETS:
                branch = _branch(seed, target, "scalar_k0", "i16_scalar_branch_v1")
                branch["policy"] = None
                name = f"branch-s{seed}-sgdm-{target}-k0.json"
                record = _record(name, branch)
                i16_objects["confirmation/" + name] = branch
                i16_artifacts.append(record)
                references.append({"seed": seed, "target": target,
                    "curve_artifact": record, "checkpoint_records": [],
                    "parent_state_digest": "1" * 64,
                    "parent_evaluation_digest": "2" * 64})
        i16_index = {"entries": []}
        i16_objects["confirmation/branches.json"] = i16_index
        i16_artifacts.append(_record("branches.json", i16_index))
        sources16 = {"runner.py": "5" * 64}
        smoke16 = {"schema": "i16_completion_v1", "status": "complete",
                   "phase": "smoke", "frozen_commit": "a" * 40,
                   "source_hashes": sources16, "artifacts": []}
        confirm16 = {"schema": "i16_completion_v1", "status": "complete",
                     "phase": "confirmation", "frozen_commit": "a" * 40,
                     "source_hashes": sources16, "artifacts": i16_artifacts}
        i16_objects.update({"smoke/completion.json": smoke16,
                            "confirmation/completion.json": confirm16,
                            "attempt-smoke.json": {"schema": "i16_attempt_v1",
                                "phase": "smoke", "frozen_commit": "a" * 40,
                                "source_hashes": sources16, "restart": "forbidden"},
                            "attempt-confirmation.json": {"schema": "i16_attempt_v1",
                                "phase": "confirmation", "frozen_commit": "a" * 40,
                                "source_hashes": sources16, "restart": "forbidden"}})
        phase16 = {"smoke": _record("completion.json", smoke16)["sha256"],
                   "confirmation": _record("completion.json", confirm16)["sha256"]}
        attempts16 = {name: _record("attempt.json", i16_objects["attempt-" + name + ".json"])["sha256"]
                      for name in ("smoke", "confirmation")}
        i16_dir = root / "i16"
        i16_sha = _write_archive(i16_dir, "i16_scalar_collection_v1", i16_objects,
                                 source_root=source16, phase_completion_sha256=phase16,
                                 attempt_sha256=attempts16,
                                 source_provenance={phase: {"frozen_commit": "a" * 40,
                                     "source_hashes": sources16}
                                     for phase in ("smoke", "confirmation")})

        binding = {"schema": "i17_parent_and_reference_binding_v1",
            "i16": {"artifact_root": source16,
                    "confirmation_sha256": phase16["confirmation"],
                    "pinned_sha256": {"raw-results-001/collection.json": i16_sha}},
            "k0_references": references, "source_replayed": False}
        i17_objects = {"confirmation/parent-inputs.json": binding}
        i17_artifacts = [_record("parent-inputs.json", binding)]
        entries = []
        direct = {}
        for seed in report.SEEDS:
            for target in report.TARGETS:
                for policy in report.NEW_POLICIES:
                    branch = _branch(seed, target, policy, "i17_gain_branch_v1")
                    name = "branch-" + branch["id"] + ".json"
                    record = _record(name, branch)
                    i17_objects["confirmation/" + name] = branch
                    i17_artifacts.append(record)
                    fields = ("id", "seed", "target", "policy", "status",
                              "completed_updates", "parent_state_digest",
                              "parent_evaluation_digest", "first_step_digests",
                              "update_seconds")
                    entries.append({key: branch[key] for key in fields} | {"artifact": record})
                    direct[seed, target, policy] = branch
        branch_index = {"entries": entries}
        i17_objects["confirmation/branches.json"] = branch_index
        i17_artifacts.append(_record("branches.json", branch_index))
        sources17 = {"gain.py": "6" * 64}
        smoke17 = {"schema": "i17_completion_v1", "status": "complete",
                   "phase": "smoke", "frozen_commit": "b" * 40,
                   "source_hashes": sources17, "artifacts": []}
        confirm17 = {"schema": "i17_completion_v1", "status": "complete",
                     "phase": "confirmation", "frozen_commit": "b" * 40,
                     "source_hashes": sources17, "artifacts": i17_artifacts}
        i17_objects.update({"smoke/completion.json": smoke17,
                            "confirmation/completion.json": confirm17,
                            "attempt-smoke.json": {"schema": "i17_attempt_v1",
                                "phase": "smoke", "frozen_commit": "b" * 40,
                                "source_hashes": sources17, "restart": "forbidden"},
                            "attempt-confirmation.json": {"schema": "i17_attempt_v1",
                                "phase": "confirmation", "frozen_commit": "b" * 40,
                                "source_hashes": sources17, "restart": "forbidden"}})
        phase17 = {"smoke": _record("completion.json", smoke17)["sha256"],
                   "confirmation": _record("completion.json", confirm17)["sha256"]}
        attempts17 = {name: _record("attempt.json", i17_objects["attempt-" + name + ".json"])["sha256"]
                      for name in ("smoke", "confirmation")}
        runtime = {"declared_name": "runtime", "entries": [],
                   "manifest_sha256": "4" * 64, "regular_file_bytes": 0,
                   "logical_bytes_only": True}
        analysis = root / "analysis-001"
        analysis.mkdir()
        audit = {"schema": "i17_gain_analysis_audit_v1", "status": "pass",
                 "errors": [], "artifact_root": source17,
                 "phase_completion_sha256": phase17,
                 "attempt_sha256": attempts17, "runtime_inventory": runtime}
        audit_path = analysis / "audit.json"
        audit_path.write_bytes(_raw(audit))
        audit_sha = report.sha256_path(audit_path)
        i17_dir = root / "i17"
        i17_sha = _write_archive(i17_dir, "i17_gain_collection_v1", i17_objects,
                                 source_root=source17,
                                 analysis_audit={"path": str(audit_path), "status": "pass",
                                     "schema": "i17_gain_analysis_audit_v1", "sha256": audit_sha},
                                 phase_completion_sha256=phase17,
                                 attempt_sha256=attempts17, runtime_inventory=runtime,
                                 source_provenance={phase: {"frozen_commit": "b" * 40,
                                     "source_hashes": sources17}
                                     for phase in ("smoke", "confirmation")})

        archives = (report.Archive(i17_dir, i17_sha, "i17_gain_collection_v1"),
                    report.Archive(i16_dir, i16_sha, "i16_scalar_collection_v1"))
        branches = report.load_branches(*archives)
        choices, primary, per_k = report.selection_results(branches)
        summary = {"schema": "i17_gain_analysis_summary_v1", "audit_status": "pass",
                   "artifact_root": source17,
                   "primary_validation_selected_spectral_minus_scalar": primary,
                   "validation_selected_choices": choices,
                   "validation_selected_spectral_minus_each_k": per_k,
                   "fixed_k_endpoint_spectral_minus_scalar": report.endpoint_effects(branches),
                   "all_policy_trajectories": report.trajectory_rows(branches),
                   "scheduled_metric_equal_seed_aggregates": report.scheduled_aggregates(branches),
                   "path_and_displacement_geometry": report.geometry_results(branches)}
        summary_path = analysis / "summary.json"
        summary_path.write_bytes(_raw(summary))
        return {"i17": i17_dir, "i17_sha": i17_sha, "i16": i16_dir,
                "i16_sha": i16_sha, "summary": summary_path,
                "summary_sha": report.sha256_path(summary_path), "audit": audit_path,
                "audit_sha": audit_sha, "output": analysis / "report-audit.json"}

    def run_fixture(self, fixture):
        return report.corroborate(fixture["i17"], fixture["i17_sha"],
            fixture["i16"], fixture["i16_sha"], fixture["summary"],
            fixture["summary_sha"], fixture["audit"], fixture["audit_sha"],
            fixture["output"])

    def test_complete_fixture_recomputes_every_registered_section(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.fixture(Path(directory))
            result = self.run_fixture(fixture)
            self.assertEqual(result["status"], "pass")
            self.assertEqual(result["complete_trajectories"], 30)
            self.assertEqual(result["retained_scheduled_points"], 180)
            self.assertEqual(result["maximum_absolute_numeric_difference"], 0)
            self.assertEqual({name: value["rows"] for name, value in result["sections"].items()},
                {"primary_validation_selected_spectral_minus_scalar": 8,
                 "validation_selected_choices": 12,
                 "validation_selected_spectral_minus_each_k": 32,
                 "fixed_k_endpoint_spectral_minus_scalar": 16,
                 "all_policy_trajectories": 30,
                 "scheduled_metric_equal_seed_aggregates": 960,
                 "path_and_displacement_geometry": 20})

    def test_joint_ties_are_earliest_horizon_then_lower_k(self):
        branches = {}
        for seed in report.SEEDS:
            for target in report.TARGETS:
                for policy in report.POLICIES:
                    curve = [_point(seed, target, policy, horizon) for horizon in report.HORIZONS]
                    for point in curve:
                        point["validation"]["clean_ce"] = 1.0
                        point["validation"]["clean_accuracy"] = .5
                    branches[seed, target, policy] = {"status": "complete", "curve": curve,
                                                       "steps": [], "failure": None}
        choices, _, _ = report.selection_results(branches)
        self.assertTrue(all(row["scalar"]["k"] == 0.0
                            and row["scalar"]["horizon"] == 100
                            and row["spectral"]["horizon"] == 100 for row in choices))

    def test_failed_scalar_forbids_survivor_primary_mean(self):
        branches = {}
        for seed in report.SEEDS:
            for target in report.TARGETS:
                for policy in report.POLICIES:
                    branches[seed, target, policy] = {"status": "complete",
                        "curve": [_point(seed, target, policy, h) for h in report.HORIZONS],
                        "steps": [], "failure": None}
        branches[201, "fixed", "scalar_k0p5"]["status"] = "numerical_failure"
        _, primary, _ = report.selection_results(branches)
        affected = [row for row in primary if row["target"] == "fixed"]
        self.assertTrue(all(row["effect"]["per_seed"]["201"] is None
                            and row["effect"]["mean"] is None
                            and not row["effect"]["available"] for row in affected))

    def test_hash_bound_audit_precedes_output_and_output_is_exclusive(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.fixture(Path(directory))
            with self.assertRaisesRegex(ValueError, "audit.*SHA256"):
                report.corroborate(fixture["i17"], fixture["i17_sha"], fixture["i16"],
                    fixture["i16_sha"], fixture["summary"], fixture["summary_sha"],
                    fixture["audit"], "0" * 64, fixture["output"])
            self.assertFalse(fixture["output"].exists())
            self.assertEqual(self.run_fixture(fixture)["status"], "pass")
            with self.assertRaises(FileExistsError):
                self.run_fixture(fixture)

    def test_changed_summary_produces_a_retained_fail_report(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.fixture(Path(directory))
            summary = json.loads(fixture["summary"].read_text())
            summary["fixed_k_endpoint_spectral_minus_scalar"][0]["effect"]["mean"] += .2
            fixture["summary"].write_bytes(_raw(summary))
            fixture["summary_sha"] = report.sha256_path(fixture["summary"])
            result = self.run_fixture(fixture)
            self.assertEqual(result["status"], "fail")
            self.assertTrue(result["errors"])


if __name__ == "__main__":
    unittest.main()
