"""Synthetic stdlib-only tests for the independent I15 corroborator."""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import corroborate_report as corroborator


def _raw(value) -> bytes:
    return json.dumps(value, separators=(",", ":"), allow_nan=False).encode() + b"\n"


def _record(name: str, value) -> dict:
    raw = _raw(value)
    return {"name": name, "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest()}


def _point(seed: int, target: str, policy: str, horizon: int) -> dict:
    policy_index = corroborator.POLICIES.index(policy)
    offset = ((seed - 199) * .013 + (target == "fixed") * .07
              + policy_index * .019 + horizon / 1_000_000)
    preferred_ce = corroborator.HORIZONS[(policy_index + seed) % 6]
    preferred_accuracy = corroborator.HORIZONS[(policy_index + seed + 2) % 6]
    validation_ce = 1 + offset + abs(horizon - preferred_ce) / 100_000
    validation_accuracy = .7 + offset / 20 - abs(
        horizon - preferred_accuracy) / 100_000

    def evaluation(ce, accuracy):
        return {"clean_ce": ce, "soft_ce": ce + .2,
                "clean_accuracy": accuracy,
                "mean_max_probability": .6 + offset / 30,
                "mean_true_label_probability": .5 + offset / 40}

    train = evaluation(1.2 + offset, .65 + offset / 20)
    train.update({"fixed_ce": 1.4 + offset,
                  "fixed_accuracy": .4 + offset / 20,
                  "fixed_minus_soft_ce": (1.4 + offset) - train["soft_ce"]})
    return {"horizon": horizon, "train": train,
            "validation": evaluation(validation_ce, validation_accuracy),
            "auxiliary": evaluation(1.1 + offset, .68 + offset / 20)}


def _steps(seed: int, policy: str) -> list[dict]:
    base = 1 + (seed - 200) * .1 + corroborator.POLICIES.index(policy) * .05
    rows = []
    for step, multiplier in ((101, 1.), (1001, 2.)):
        rows.append({"step": step, "displacement": {
            "data_current_basis_leakage": {
                "reason": None, "outside_squared_norm": base * multiplier,
                "squared_norm": 4 * base * multiplier},
            "total_current_basis_leakage": {
                "reason": None, "outside_squared_norm": 2 * base * multiplier,
                "squared_norm": 5 * base * multiplier}}})
    return rows


def _write_archive(directory: Path, schema: str, objects: dict[str, object], **extra):
    directory.mkdir()
    rows = []
    for original in sorted(objects):
        raw = _raw(objects[original])
        archive = gzip.compress(raw, compresslevel=9, mtime=0)
        target = directory / (original + ".gz")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(archive)
        rows.append({"original": original, "original_bytes": len(raw),
                     "original_sha256": hashlib.sha256(raw).hexdigest(),
                     "archive": original + ".gz", "archive_bytes": len(archive),
                     "archive_sha256": hashlib.sha256(archive).hexdigest()})
    manifest = {"schema": schema, "status": "complete", **extra,
                "json_count": len(rows),
                "original_bytes": sum(row["original_bytes"] for row in rows),
                "archive_bytes": sum(row["archive_bytes"] for row in rows),
                "files": rows}
    (directory / "collection.json").write_bytes(_raw(manifest))
    return corroborator.sha256_path(directory / "collection.json")


class CorroboratorTests(unittest.TestCase):
    def fixture(self, root: Path, failure=False):
        i14_objects = {}
        references = []
        i14_source = "/synthetic/i14-root"
        i14_completion_sha = "4" * 64
        supplement_sha = "5" * 64
        for seed in corroborator.SEEDS:
            for target in corroborator.TARGETS:
                for old_policy, policy in (("raw", "raw"),
                                           ("current32", "current_native")):
                    identity = f"s{seed}-sgdm-lr0.03-{target}-{old_policy}"
                    name = "curve-" + identity + ".json"
                    curve = {"schema": "i14_trajectory_v1", "id": identity,
                             "seed": seed, "target": target, "policy": old_policy,
                             "base": "sgdm", "lr": .03, "status": "complete",
                             "completed_steps": 2000,
                             "curve": [_point(seed, target, policy, horizon)
                                       for horizon in corroborator.HORIZONS],
                             "steps": _steps(seed, policy)}
                    i14_objects["confirmation/" + name] = curve
                    references.append({"seed": seed, "target": target,
                        "policy": old_policy, "curve_artifact": _record(name, curve),
                        "checkpoint_records": [], "warmup_state_digest": "6" * 64})
        i14_objects["confirmation/completion.json"] = {"status": "complete"}
        i14_dir = root / "i14-archive"
        i14_sha = _write_archive(i14_dir, "i14_scalar_collection_v1", i14_objects,
            source_directory=i14_source,
            source_completion_sha256={"confirmation": i14_completion_sha},
            runtime_directory_supplement_sha256=supplement_sha)

        i15_objects = {}
        binding = {"schema": "i15_i14_inputs_v1", "source_root": i14_source,
                   "confirmation_completion_sha256": i14_completion_sha,
                   "runtime_supplement_sha256": supplement_sha,
                   "references": references}
        i15_objects["confirmation/parent-inputs.json"] = binding
        entries = []
        failed_key = (201, "fixed", "mean_projected_history")
        for seed in corroborator.SEEDS:
            for target in corroborator.TARGETS:
                for policy in corroborator.NEW_POLICIES:
                    key = seed, target, policy
                    failed = failure and key == failed_key
                    identity = f"s{seed}-sgdm-{target}-{policy}"
                    name = "branch-" + identity + ".json"
                    curve = [_point(seed, target, policy, horizon)
                             for horizon in (corroborator.HORIZONS[:2]
                                             if failed else corroborator.HORIZONS)]
                    branch = {"schema": "i15_history_branch_v1", "id": identity,
                        "seed": seed, "target": target, "policy": policy,
                        "status": "numerical_failure" if failed else "complete",
                        "completed_updates": 250 if failed else 1900,
                        "parent_state_digest": "7" * 64,
                        "parent_evaluation_digest": "8" * 64,
                        "first_step_digests": {"raw_gradient": "9" * 64},
                        "curve": curve, "steps": _steps(seed, policy),
                        "failure": ({"attempted_step": 351} if failed else None)}
                    i15_objects["confirmation/" + name] = branch
                    entries.append({key: branch[key] for key in (
                        "id", "seed", "target", "policy", "status",
                        "completed_updates", "parent_state_digest",
                        "parent_evaluation_digest", "first_step_digests")}
                        | {"artifact": _record(name, branch)})
        i15_objects["confirmation/branches.json"] = {"entries": entries}
        for phase in ("smoke", "confirmation"):
            i15_objects[f"{phase}/completion.json"] = {
                "schema": "i15_completion_v1", "status": "complete", "phase": phase}
            i15_objects[f"attempt-{phase}.json"] = {
                "schema": "i15_attempt_v1", "phase": phase}
        i15_dir = root / "i15-archive"
        i15_root = "/synthetic/i15-root"
        i15_sha = _write_archive(i15_dir, "i15_scalar_collection_v1", i15_objects,
            source_root=i15_root, analysis_audit={"status": "pass"})

        i15_loaded = corroborator.Archive(i15_dir, i15_sha,
                                           "i15_scalar_collection_v1")
        i14_loaded = corroborator.Archive(i14_dir, i14_sha,
                                           "i14_scalar_collection_v1")
        branches = corroborator.load_branches(i15_loaded, i14_loaded)
        outcomes, selected = corroborator.selected_results(branches)
        summary = {"schema": "i15_history_analysis_summary_v1",
                   "artifact_root": i15_root, "audit_status": "pass",
                   "primary_endpoint_effects": corroborator.primary_effects(branches),
                   "validation_selected_primary_effects": selected,
                   "validation_selected_auxiliary_outcomes": outcomes,
                   "all_policy_trajectories": corroborator.trajectory_rows(branches),
                   "scheduled_metric_equal_seed_aggregates":
                       corroborator.scheduled_aggregates(branches),
                   "energy_weighted_current_action_complement":
                       corroborator.energy_ratios(branches)}
        analysis = root / "analysis-001"
        analysis.mkdir()
        summary_path = analysis / "summary.json"
        summary_path.write_bytes(_raw(summary))
        return {"i15": i15_dir, "i15_sha": i15_sha,
                "i14": i14_dir, "i14_sha": i14_sha,
                "summary": summary_path,
                "summary_sha": corroborator.sha256_path(summary_path),
                "output": analysis / "report-audit.json"}

    def run_fixture(self, fixture):
        return corroborator.corroborate(
            fixture["i15"], fixture["i15_sha"], fixture["i14"],
            fixture["i14_sha"], fixture["summary"], fixture["summary_sha"],
            fixture["output"])

    def test_complete_fixture_corroborates_every_registered_section(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.fixture(Path(directory))
            report = self.run_fixture(fixture)
            self.assertEqual(report["status"], "pass")
            self.assertEqual(report["complete_trajectories"], 30)
            self.assertEqual(report["retained_scheduled_points"], 180)
            self.assertEqual(report["maximum_absolute_numeric_difference"], 0)
            self.assertEqual({name: row["rows"] for name, row in
                              report["sections"].items()}, {
                "primary_endpoint_effects": 12,
                "validation_selected_primary_effects": 24,
                "validation_selected_auxiliary_outcomes": 60,
                "all_policy_trajectories": 30,
                "scheduled_metric_equal_seed_aggregates": 960,
                "energy_weighted_current_action_complement": 40})
            self.assertEqual(json.loads(fixture["output"].read_text()), report)

    def test_numerical_failure_preserves_null_semantics_without_survivor_mean(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.fixture(Path(directory), failure=True)
            report = self.run_fixture(fixture)
            self.assertEqual(report["status"], "pass")
            self.assertEqual(report["complete_trajectories"], 29)
            summary = json.loads(fixture["summary"].read_text())
            affected = next(row for row in summary["primary_endpoint_effects"]
                            if row["family"] == "M_mean_under_projected_history"
                            and row["target"] == "fixed" and row["metric"] == "ce")
            self.assertIsNone(affected["effect"]["per_seed"]["201"])
            self.assertFalse(affected["effect"]["available"])
            self.assertIsNone(affected["effect"]["mean"])
            outcome = next(row for row in summary[
                "validation_selected_auxiliary_outcomes"]
                if row["seed"] == 201 and row["target"] == "fixed"
                and row["policy"] == "mean_projected_history")
            self.assertIsNone(outcome["selected_horizon"])
            self.assertIsNone(outcome["auxiliary"])

    def test_summary_difference_writes_failing_report(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.fixture(Path(directory))
            summary = json.loads(fixture["summary"].read_text())
            summary["primary_endpoint_effects"][0]["effect"]["mean"] += .5
            fixture["summary"].write_bytes(_raw(summary))
            fixture["summary_sha"] = corroborator.sha256_path(fixture["summary"])
            report = self.run_fixture(fixture)
            self.assertEqual(report["status"], "fail")
            self.assertEqual(report["sections"]["primary_endpoint_effects"]["status"],
                             "fail")
            self.assertTrue(report["errors"])

    def test_archive_or_summary_hash_failure_precedes_output(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.fixture(Path(directory))
            with self.assertRaisesRegex(ValueError, "summary differs"):
                corroborator.corroborate(
                    fixture["i15"], fixture["i15_sha"], fixture["i14"],
                    fixture["i14_sha"], fixture["summary"], "0" * 64,
                    fixture["output"])
            self.assertFalse(fixture["output"].exists())

            member = next(fixture["i14"].rglob("*.json.gz"))
            member.write_bytes(member.read_bytes() + b"tamper")
            with self.assertRaisesRegex(ValueError, "Gzip member differs"):
                self.run_fixture(fixture)
            self.assertFalse(fixture["output"].exists())

    def test_i15_to_i14_reference_binding_is_mandatory(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.fixture(Path(directory))
            manifest_path = fixture["i15"] / "collection.json"
            manifest = json.loads(manifest_path.read_text())
            row = next(row for row in manifest["files"]
                       if row["original"] == "confirmation/parent-inputs.json")
            archive = fixture["i15"] / row["archive"]
            binding = json.loads(gzip.decompress(archive.read_bytes()))
            binding["references"][0]["curve_artifact"]["sha256"] = "0" * 64
            raw, compressed = _raw(binding), gzip.compress(_raw(binding), mtime=0)
            archive.write_bytes(compressed)
            row.update(original_bytes=len(raw), original_sha256=hashlib.sha256(raw).hexdigest(),
                       archive_bytes=len(compressed),
                       archive_sha256=hashlib.sha256(compressed).hexdigest())
            manifest["original_bytes"] = sum(r["original_bytes"] for r in manifest["files"])
            manifest["archive_bytes"] = sum(r["archive_bytes"] for r in manifest["files"])
            manifest_path.write_bytes(_raw(manifest))
            fixture["i15_sha"] = corroborator.sha256_path(manifest_path)
            with self.assertRaisesRegex(ValueError, "reference record differs"):
                self.run_fixture(fixture)
            self.assertFalse(fixture["output"].exists())

    def test_output_is_exclusive_and_exactly_named(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.fixture(Path(directory))
            fixture["output"].write_text("occupied")
            with self.assertRaises(FileExistsError):
                self.run_fixture(fixture)
            fixture["output"].unlink()
            with self.assertRaisesRegex(ValueError, "absolute absent"):
                corroborator.corroborate(
                    fixture["i15"], fixture["i15_sha"], fixture["i14"],
                    fixture["i14_sha"], fixture["summary"], fixture["summary_sha"],
                    Path("report-audit.json"))


if __name__ == "__main__":
    unittest.main()
