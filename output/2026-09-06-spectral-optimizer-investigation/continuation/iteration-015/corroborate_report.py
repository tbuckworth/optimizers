#!/usr/bin/env python3
"""Independently corroborate I15 scalar summaries from lossless JSON archives."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import re


SEEDS = (200, 201, 202)
TARGETS = ("clean", "fixed")
NEW_POLICIES = ("current_projected_history", "mean_native",
                "mean_projected_history")
POLICIES = ("raw", "current_native") + NEW_POLICIES
HORIZONS = (100, 250, 500, 1000, 1500, 2000)
SELECTORS = ("minimum_validation_ce", "maximum_validation_accuracy")
METRICS = ("ce", "accuracy")
FAMILIES = ("H_history_under_current", "M_mean_under_projected_history",
            "S_mean_by_history_interaction")
WINDOWS = {"steps101_2000": (101, 2000),
           "steps1001_2000": (1001, 2000)}
SHA_RE = re.compile(r"[0-9a-f]{64}")
CHUNK = 1024 * 1024


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(CHUNK):
            digest.update(block)
    return digest.hexdigest()


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("JSON contains a duplicate key")
        result[key] = value
    return result


def _invalid_constant(value):
    raise ValueError("JSON contains a nonfinite constant: " + value)


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle, object_pairs_hook=_object,
                         parse_constant=_invalid_constant)


def _safe_relative(value) -> bool:
    return (type(value) is str and value not in ("", ".", "..")
            and not Path(value).is_absolute() and ".." not in Path(value).parts
            and Path(value).as_posix() == value)


def _regular(path: Path, label: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise ValueError(label + " must be a regular nonsymlink file")


def _archive_file_members(directory: Path) -> set[str]:
    members = set()

    def visit(parent: Path, prefix: Path) -> None:
        with os.scandir(parent) as iterator:
            children = list(iterator)
        for child in children:
            relative = prefix / child.name
            if child.is_symlink():
                raise ValueError("Archive contains a symlink: " + relative.as_posix())
            if child.is_dir(follow_symlinks=False):
                visit(Path(child.path), relative)
            elif child.is_file(follow_symlinks=False):
                members.add(relative.as_posix())
            else:
                raise ValueError("Archive contains a special file: " + relative.as_posix())

    visit(directory, Path())
    return members


class Archive:
    def __init__(self, directory, expected_sha: str, schema: str):
        supplied = Path(directory)
        if supplied.is_symlink():
            raise ValueError("Archive directory must be nonsymlink")
        self.directory = supplied.resolve(strict=True)
        if not self.directory.is_dir() or SHA_RE.fullmatch(expected_sha or "") is None:
            raise ValueError("Archive directory or manifest SHA256 is invalid")
        manifest_path = self.directory / "collection.json"
        _regular(manifest_path, "Collection manifest")
        if sha256_path(manifest_path) != expected_sha:
            raise ValueError("Collection manifest differs from its supplied SHA256")
        self.manifest = load_json(manifest_path)
        if type(self.manifest) is not dict or self.manifest.get("schema") != schema \
                or self.manifest.get("status") != "complete":
            raise ValueError("Collection manifest schema or status differs")
        rows = self.manifest.get("files")
        if type(rows) is not list or self.manifest.get("json_count") != len(rows):
            raise ValueError("Collection row count differs")
        self.rows = {}
        archive_names = set()
        for row in rows:
            keys = {"original", "original_bytes", "original_sha256", "archive",
                    "archive_bytes", "archive_sha256"}
            if type(row) is not dict or set(row) != keys:
                raise ValueError("Collection row topology differs")
            original, archive = row["original"], row["archive"]
            if not _safe_relative(original) or not _safe_relative(archive) \
                    or not archive.endswith(".gz"):
                raise ValueError("Collection row path is unsafe")
            if original in self.rows or archive in archive_names:
                raise ValueError("Collection contains duplicate paths")
            if type(row["original_bytes"]) is not int or isinstance(
                    row["original_bytes"], bool) or row["original_bytes"] < 0 \
                    or type(row["archive_bytes"]) is not int or isinstance(
                        row["archive_bytes"], bool) or row["archive_bytes"] < 0 \
                    or SHA_RE.fullmatch(row["original_sha256"] or "") is None \
                    or SHA_RE.fullmatch(row["archive_sha256"] or "") is None:
                raise ValueError("Collection row digest or size differs")
            self.rows[original] = row
            archive_names.add(archive)
        actual = _archive_file_members(self.directory)
        if actual != archive_names | {"collection.json"}:
            raise ValueError("Archive file membership differs from collection manifest")
        if self.manifest.get("original_bytes") != sum(
                row["original_bytes"] for row in rows) \
                or self.manifest.get("archive_bytes") != sum(
                    row["archive_bytes"] for row in rows):
            raise ValueError("Collection byte totals differ")
        self.objects = {}
        for original, row in self.rows.items():
            path = self.directory / row["archive"]
            _regular(path, "Gzip member")
            if path.stat().st_size != row["archive_bytes"] \
                    or sha256_path(path) != row["archive_sha256"]:
                raise ValueError("Gzip member differs: " + original)
            size, digest = self._roundtrip(path, row["original_bytes"])
            if size != row["original_bytes"] or digest != row["original_sha256"]:
                raise ValueError("Original JSON roundtrip differs: " + original)
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                self.objects[original] = json.load(
                    handle, object_pairs_hook=_object,
                    parse_constant=_invalid_constant)

    @staticmethod
    def _roundtrip(path: Path, expected_size: int) -> tuple[int, str]:
        size = 0
        digest = hashlib.sha256()
        with gzip.open(path, "rb") as handle:
            while block := handle.read(CHUNK):
                size += len(block)
                if size > expected_size:
                    raise ValueError("Gzip expands beyond its declared original size")
                digest.update(block)
        return size, digest.hexdigest()

    def require(self, name: str):
        if name not in self.objects:
            raise ValueError("Required archived JSON is absent: " + name)
        return self.objects[name]

    def matches_record(self, original: str, record) -> bool:
        row = self.rows.get(original)
        return (type(record) is dict and row is not None
                and record.get("name") == Path(original).name
                and record.get("bytes") == row["original_bytes"]
                and record.get("sha256") == row["original_sha256"])


def _curve_points(branch) -> list[dict]:
    curve = branch.get("curve")
    if type(curve) is not list:
        raise ValueError("Trajectory curve is absent")
    points = [row for row in curve if type(row) is dict
              and row.get("horizon") in HORIZONS]
    if branch.get("status") == "complete" \
            and [row.get("horizon") for row in points] != list(HORIZONS):
        raise ValueError("Complete trajectory lacks all six scheduled horizons")
    return points


def load_branches(i15: Archive, i14: Archive) -> dict[tuple[int, str, str], dict]:
    binding = i15.require("confirmation/parent-inputs.json")
    i14_root = i14.manifest.get("source_directory")
    if type(binding) is not dict or binding.get("schema") != "i15_i14_inputs_v1" \
            or binding.get("source_root") != i14_root \
            or i14.manifest.get("source_completion_sha256", {}).get("confirmation") \
            != binding.get("confirmation_completion_sha256") \
            or i14.manifest.get("runtime_directory_supplement_sha256") \
            != binding.get("runtime_supplement_sha256"):
        raise ValueError("I15-to-I14 archive binding differs")

    branches = {}
    references = binding.get("references")
    expected_refs = {(seed, target, policy) for seed in SEEDS for target in TARGETS
                     for policy in ("raw", "current32")}
    if type(references) is not list or len(references) != 12:
        raise ValueError("I14 reference roster differs")
    observed_refs = set()
    for reference in references:
        if type(reference) is not dict:
            raise ValueError("I14 reference is malformed")
        seed, target, old_policy = (reference.get("seed"), reference.get("target"),
                                    reference.get("policy"))
        key = (seed, target, old_policy)
        observed_refs.add(key)
        record = reference.get("curve_artifact")
        name = record.get("name") if type(record) is dict else ""
        original = "confirmation/" + name
        if not i14.matches_record(original, record):
            raise ValueError("I14 reference record differs from its archive")
        curve = i14.require(original)
        if curve.get("seed") != seed or curve.get("target") != target \
                or curve.get("policy") != old_policy or curve.get("base") != "sgdm" \
                or curve.get("lr") != .03 or curve.get("status") != "complete":
            raise ValueError("I14 reference curve identity differs")
        policy = "raw" if old_policy == "raw" else "current_native"
        branches[seed, target, policy] = {
            "seed": seed, "target": target, "policy": policy,
            "status": "complete", "curve": _curve_points(curve),
            "steps": [row for row in curve.get("steps", [])
                      if type(row) is dict and row.get("step", 0) >= 101],
            "failure": None}
    if observed_refs != expected_refs:
        raise ValueError("I14 reference membership differs")

    index = i15.require("confirmation/branches.json")
    entries = index.get("entries") if type(index) is dict else None
    expected_new = {(seed, target, policy) for seed in SEEDS for target in TARGETS
                    for policy in NEW_POLICIES}
    if type(entries) is not list or len(entries) != 18:
        raise ValueError("I15 new-branch roster differs")
    observed_new = set()
    for entry in entries:
        if type(entry) is not dict:
            raise ValueError("I15 branch-index entry is malformed")
        key = (entry.get("seed"), entry.get("target"), entry.get("policy"))
        observed_new.add(key)
        record = entry.get("artifact")
        name = record.get("name") if type(record) is dict else ""
        original = "confirmation/" + name
        if not i15.matches_record(original, record):
            raise ValueError("I15 branch record differs from its archive")
        branch = i15.require(original)
        if branch.get("schema") != "i15_history_branch_v1" \
                or (branch.get("seed"), branch.get("target"), branch.get("policy")) != key \
                or branch.get("status") not in ("complete", "numerical_failure"):
            raise ValueError("I15 branch identity or status differs")
        for field in ("id", "seed", "target", "policy", "status", "completed_updates",
                      "parent_state_digest", "parent_evaluation_digest",
                      "first_step_digests"):
            if branch.get(field) != entry.get(field):
                raise ValueError("I15 branch index identity differs")
        branches[key] = {"seed": key[0], "target": key[1], "policy": key[2],
                         "status": branch["status"], "curve": _curve_points(branch),
                         "steps": branch.get("steps", []),
                         "failure": branch.get("failure")}
    if observed_new != expected_new:
        raise ValueError("I15 new-branch membership differs")
    expected_all = {(seed, target, policy) for seed in SEEDS for target in TARGETS
                    for policy in POLICIES}
    if set(branches) != expected_all:
        raise ValueError("Five-policy branch membership differs")
    return branches


def _point(branch, horizon):
    if branch is None or branch.get("status") != "complete":
        return None
    return next((row for row in branch["curve"] if row["horizon"] == horizon), None)


def _utility(point, metric):
    return (-point["auxiliary"]["clean_ce"] if metric == "ce"
            else point["auxiliary"]["clean_accuracy"])


def _family(points, metric, family):
    required = {
        "H_history_under_current": ("current_native", "current_projected_history"),
        "M_mean_under_projected_history": ("mean_projected_history",
                                             "current_projected_history"),
        "S_mean_by_history_interaction": ("current_native",
                                           "current_projected_history", "mean_native",
                                           "mean_projected_history")}
    if any(points.get(policy) is None for policy in required[family]):
        return None
    utility = {policy: _utility(point, metric) for policy, point in points.items()
               if point is not None}
    if family == "H_history_under_current":
        return utility["current_native"] - utility["current_projected_history"]
    if family == "M_mean_under_projected_history":
        return utility["mean_projected_history"] - utility["current_projected_history"]
    return ((utility["mean_projected_history"]
             - utility["current_projected_history"])
            - (utility["mean_native"] - utility["current_native"]))


def _effect(per_seed):
    ordered = [per_seed[str(seed)] for seed in SEEDS]
    available = all(value is not None for value in ordered)
    return {"per_seed": per_seed, "all_three_seed_values": ordered,
            "available": available,
            "mean": sum(ordered) / len(SEEDS) if available else None,
            "no_survivor_averaging": True}


def primary_effects(branches):
    rows = []
    for family in FAMILIES:
        for target in TARGETS:
            for metric in METRICS:
                values = {}
                for seed in SEEDS:
                    points = {policy: _point(branches[seed, target, policy], 2000)
                              for policy in POLICIES}
                    values[str(seed)] = _family(points, metric, family)
                rows.append({"family": family, "target": target, "metric": metric,
                             "horizon": 2000, "effect": _effect(values)})
    return rows


def _selection(curve):
    candidates = [row for row in curve if row.get("horizon") in HORIZONS]
    if not candidates:
        raise ValueError("Complete branch has no selectable horizon")
    by_ce = min(candidates,
                key=lambda row: (row["validation"]["clean_ce"], row["horizon"]))
    by_accuracy = min(candidates,
        key=lambda row: (-row["validation"]["clean_accuracy"], row["horizon"]))
    return {"minimum_validation_ce": by_ce["horizon"],
            "maximum_validation_accuracy": by_accuracy["horizon"]}


def selected_results(branches):
    outcomes = []
    chosen = {}
    for seed in SEEDS:
        for target in TARGETS:
            for policy in POLICIES:
                branch = branches[seed, target, policy]
                for selector in SELECTORS:
                    point = None
                    if branch["status"] == "complete":
                        point = _point(branch, _selection(branch["curve"])[selector])
                    chosen[seed, target, policy, selector] = point
                    outcomes.append({"seed": seed, "target": target, "policy": policy,
                        "selector": selector,
                        "selected_horizon": None if point is None else point["horizon"],
                        "auxiliary": None if point is None else {
                            "clean_ce": point["auxiliary"]["clean_ce"],
                            "clean_accuracy": point["auxiliary"]["clean_accuracy"]}})
    effects = []
    for family in FAMILIES:
        for target in TARGETS:
            for selector in SELECTORS:
                for metric in METRICS:
                    values = {}
                    for seed in SEEDS:
                        points = {policy: chosen[seed, target, policy, selector]
                                  for policy in POLICIES}
                        values[str(seed)] = _family(points, metric, family)
                    effects.append({"family": family, "target": target,
                                    "selector": selector,
                                    "auxiliary_metric": metric,
                                    "effect": _effect(values)})
    return outcomes, effects


def trajectory_rows(branches):
    rows = []
    for key in sorted(branches):
        branch = branches[key]
        h100 = next((row for row in branch["curve"] if row["horizon"] == 100), None)
        curve = []
        for point in branch["curve"]:
            values = {
                "train_clean_ce": point["train"]["clean_ce"],
                "train_soft_ce": point["train"]["soft_ce"],
                "train_clean_accuracy": point["train"]["clean_accuracy"],
                "auxiliary_clean_accuracy": point["auxiliary"]["clean_accuracy"],
                "auxiliary_clean_ce": point["auxiliary"]["clean_ce"],
                "auxiliary_confidence": point["auxiliary"]["mean_max_probability"],
                "auxiliary_true_label_probability":
                    point["auxiliary"]["mean_true_label_probability"],
                "validation_clean_ce": point["validation"]["clean_ce"],
                "validation_clean_accuracy": point["validation"]["clean_accuracy"],
                "validation_confidence": point["validation"]["mean_max_probability"],
                "validation_true_label_probability":
                    point["validation"]["mean_true_label_probability"],
                "train_fixed_ce": point["train"]["fixed_ce"],
                "train_fixed_minus_soft_ce": point["train"]["fixed_minus_soft_ce"],
                "train_fixed_accuracy": point["train"]["fixed_accuracy"],
                "train_confidence": point["train"]["mean_max_probability"],
                "train_true_label_probability":
                    point["train"]["mean_true_label_probability"]}
            if h100 is not None:
                progress = {}
                for split in ("train", "validation", "auxiliary"):
                    for metric in ("clean_ce", "clean_accuracy"):
                        progress[f"{split}_{metric}_change"] = (
                            point[split][metric] - h100[split][metric])
                for metric in ("soft_ce", "fixed_ce", "fixed_accuracy"):
                    progress[f"train_{metric}_change"] = (
                        point["train"][metric] - h100["train"][metric])
                values["absolute_progress_from_h100"] = progress
            curve.append({"horizon": point["horizon"], "values": values})
        rows.append({"seed": key[0], "target": key[1], "policy": key[2],
                     "status": branch["status"], "failure": branch.get("failure"),
                     "curve": curve})
    return rows


SCHEDULED_METRICS = {
    "train_clean_ce": ("train", "clean_ce"),
    "train_soft_ce": ("train", "soft_ce"),
    "train_fixed_ce": ("train", "fixed_ce"),
    "train_fixed_minus_soft_ce": ("train", "fixed_minus_soft_ce"),
    "train_clean_accuracy": ("train", "clean_accuracy"),
    "train_fixed_accuracy": ("train", "fixed_accuracy"),
    "train_confidence": ("train", "mean_max_probability"),
    "train_true_label_probability": ("train", "mean_true_label_probability"),
    "validation_clean_ce": ("validation", "clean_ce"),
    "validation_clean_accuracy": ("validation", "clean_accuracy"),
    "validation_confidence": ("validation", "mean_max_probability"),
    "validation_true_label_probability": ("validation", "mean_true_label_probability"),
    "auxiliary_clean_ce": ("auxiliary", "clean_ce"),
    "auxiliary_clean_accuracy": ("auxiliary", "clean_accuracy"),
    "auxiliary_confidence": ("auxiliary", "mean_max_probability"),
    "auxiliary_true_label_probability": ("auxiliary", "mean_true_label_probability")}


def scheduled_aggregates(branches):
    rows = []
    for target in TARGETS:
        for policy in POLICIES:
            for horizon in HORIZONS:
                for metric, (split, field) in SCHEDULED_METRICS.items():
                    values, changes = {}, {}
                    for seed in SEEDS:
                        branch = branches[seed, target, policy]
                        point, start = _point(branch, horizon), _point(branch, 100)
                        values[str(seed)] = None if point is None else point[split][field]
                        changes[str(seed)] = (None if point is None or start is None else
                                              point[split][field] - start[split][field])
                    available = all(value is not None for value in values.values())
                    rows.append({"target": target, "policy": policy,
                        "horizon": horizon, "metric": metric, "per_seed": values,
                        "equal_seed_mean": sum(values.values()) / 3 if available else None,
                        "per_seed_change_from_h100": changes,
                        "equal_seed_mean_change_from_h100":
                            sum(changes.values()) / 3 if available else None,
                        "available": available, "no_survivor_averaging": True})
    return rows


def energy_ratios(branches):
    rows = []
    for target in TARGETS:
        for policy in POLICIES:
            for window, (low, high) in WINDOWS.items():
                for displacement in ("data", "total"):
                    values = {}
                    for seed in SEEDS:
                        branch = branches[seed, target, policy]
                        pairs = []
                        if branch["status"] == "complete":
                            field = displacement + "_current_basis_leakage"
                            for step in branch["steps"]:
                                if low <= step.get("step", -1) <= high:
                                    leakage = step.get("displacement", {}).get(field)
                                    if type(leakage) is dict and leakage.get("reason") is None:
                                        pairs.append((leakage["outside_squared_norm"],
                                                      leakage["squared_norm"]))
                        denominator = sum(pair[1] for pair in pairs)
                        values[str(seed)] = (sum(pair[0] for pair in pairs) / denominator
                                             if pairs and denominator > 0 else None)
                    available = all(value is not None for value in values.values())
                    rows.append({"target": target, "policy": policy, "window": window,
                        "displacement": displacement,
                        "per_seed_branch_fractions": values,
                        "equal_seed_mean": sum(values.values()) / 3 if available else None,
                        "available": available,
                        "aggregation": "energy ratio within branch, then equal seeds"})
    return rows


class Comparison:
    def __init__(self):
        self.errors = []
        self.scalars = 0
        self.max_abs_difference = 0.0

    def compare(self, observed, expected, path="root"):
        if type(observed) is dict and type(expected) is dict:
            if set(observed) != set(expected):
                self.errors.append(path + ": mapping keys differ")
                return
            for key in expected:
                self.compare(observed[key], expected[key], path + "." + str(key))
            return
        if type(observed) is list and type(expected) is list:
            if len(observed) != len(expected):
                self.errors.append(path + ": list length differs")
                return
            for index, (left, right) in enumerate(zip(observed, expected)):
                self.compare(left, right, f"{path}[{index}]")
            return
        if (type(observed) in (int, float) and not isinstance(observed, bool)
                and type(expected) in (int, float) and not isinstance(expected, bool)):
            left, right = float(observed), float(expected)
            self.scalars += 1
            if not math.isfinite(left) or not math.isfinite(right):
                self.errors.append(path + ": nonfinite number")
                return
            difference = abs(left - right)
            self.max_abs_difference = max(self.max_abs_difference, difference)
            if not math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12):
                self.errors.append(path + f": numeric difference {left!r} vs {right!r}")
            return
        self.scalars += 1
        if type(observed) is not type(expected) or observed != expected:
            self.errors.append(path + f": value differs {observed!r} vs {expected!r}")


def corroborate(i15_archive, i15_collection_sha256, i14_archive,
                i14_collection_sha256, summary_path, summary_sha256, output):
    output = Path(output)
    if not output.is_absolute() or output.name != "report-audit.json" \
            or not output.parent.name.startswith("analysis-"):
        raise ValueError("Output must be an absolute absent analysis-*/report-audit.json")
    output.parent.resolve(strict=True)
    if output.exists() or output.is_symlink():
        raise FileExistsError("Report audit output already exists")
    summary_path = Path(summary_path)
    if summary_path.is_symlink():
        raise ValueError("Summary must be a regular nonsymlink file")
    summary_path = summary_path.resolve(strict=True)
    _regular(summary_path, "Pinned I15 summary")
    if SHA_RE.fullmatch(summary_sha256 or "") is None \
            or sha256_path(summary_path) != summary_sha256:
        raise ValueError("I15 summary differs from its supplied SHA256")
    summary = load_json(summary_path)

    i15 = Archive(i15_archive, i15_collection_sha256, "i15_scalar_collection_v1")
    i14 = Archive(i14_archive, i14_collection_sha256, "i14_scalar_collection_v1")
    i15_root = i15.manifest.get("source_root")
    if type(summary) is not dict \
            or summary.get("schema") != "i15_history_analysis_summary_v1" \
            or summary.get("audit_status") != "pass" \
            or summary.get("artifact_root") != i15_root \
            or i15.manifest.get("analysis_audit", {}).get("status") != "pass":
        raise ValueError("Pinned I15 summary/archive provenance differs")

    branches = load_branches(i15, i14)
    outcomes, selected = selected_results(branches)
    recomputed = {
        "primary_endpoint_effects": primary_effects(branches),
        "validation_selected_primary_effects": selected,
        "validation_selected_auxiliary_outcomes": outcomes,
        "all_policy_trajectories": trajectory_rows(branches),
        "scheduled_metric_equal_seed_aggregates": scheduled_aggregates(branches),
        "energy_weighted_current_action_complement": energy_ratios(branches)}
    expected_counts = {
        "primary_endpoint_effects": 12,
        "validation_selected_primary_effects": 24,
        "validation_selected_auxiliary_outcomes": 60,
        "all_policy_trajectories": 30,
        "scheduled_metric_equal_seed_aggregates": 960,
        "energy_weighted_current_action_complement": 40}
    comparison = Comparison()
    sections = {}
    for name, expected in recomputed.items():
        observed = summary.get(name)
        before = len(comparison.errors)
        if len(expected) != expected_counts[name]:
            raise AssertionError("Internal corroborator row count differs: " + name)
        comparison.compare(observed, expected, name)
        sections[name] = {"rows": len(expected),
                          "status": "pass" if len(comparison.errors) == before else "fail"}
    trajectory_points = sum(len(row["curve"]) for row in recomputed[
        "all_policy_trajectories"])
    complete_trajectories = sum(row["status"] == "complete" for row in recomputed[
        "all_policy_trajectories"])
    report = {"schema": "i15_report_corroboration_v1",
        "status": "pass" if not comparison.errors else "fail",
        "inputs": {
            "i15_collection": {"path": str(Path(i15_archive).resolve()),
                               "sha256": i15_collection_sha256},
            "i14_collection": {"path": str(Path(i14_archive).resolve()),
                               "sha256": i14_collection_sha256},
            "i15_summary": {"path": str(summary_path), "sha256": summary_sha256}},
        "scope": ("Independent standard-library recomputation from collected JSON only; "
                  "no tensor loading, forward pass, optimizer replay, or audit replay."),
        "sections": sections,
        "complete_trajectories": complete_trajectories,
        "retained_scheduled_points": trajectory_points,
        "expected_points_if_all_complete": 30 * 6,
        "numeric_and_discrete_scalars_compared": comparison.scalars,
        "maximum_absolute_numeric_difference": comparison.max_abs_difference,
        "no_survivor_averaging": True,
        "errors": comparison.errors}
    with output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=True, allow_nan=False)
        handle.write("\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--i15-archive", required=True, type=Path)
    parser.add_argument("--i15-collection-sha256", required=True)
    parser.add_argument("--i14-archive", required=True, type=Path)
    parser.add_argument("--i14-collection-sha256", required=True)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--summary-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = corroborate(args.i15_archive, args.i15_collection_sha256,
                         args.i14_archive, args.i14_collection_sha256,
                         args.summary, args.summary_sha256, args.output)
    print(json.dumps({key: value for key, value in report.items()
                      if key not in ("sections", "errors")}, allow_nan=False))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
