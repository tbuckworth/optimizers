#!/usr/bin/env python3
"""Independently corroborate registered I17 report arithmetic from JSON archives.

This standard-library-only program reads losslessly archived JSON. It does not
load tensors, import acquisition/analysis code, construct a model, or replay an
optimizer. The primary integrity audit remains authoritative for tensor and
scientific-source integrity.
"""
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
HORIZONS = (100, 250, 500, 1000, 1500, 2000)
SELECTORS = ("minimum_validation_ce", "maximum_validation_accuracy")
METRICS = ("ce", "accuracy")
K_BY_POLICY = {"scalar_k0": 0.0, "scalar_k0p5": 0.5,
               "scalar_k0p9": 0.9, "scalar_k1": 1.0}
SCALARS = tuple(K_BY_POLICY)
SPECTRAL = "spectral_mean_projected"
NEW_POLICIES = SCALARS[1:] + (SPECTRAL,)
POLICIES = SCALARS + (SPECTRAL,)
WINDOWS = {"steps101_2000": (101, 2000),
           "steps1001_2000": (1001, 2000)}
SHA_RE = re.compile(r"[0-9a-f]{64}")
CHUNK = 1024 * 1024
METRIC_PATHS = {
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
    "auxiliary_true_label_probability": ("auxiliary", "mean_true_label_probability"),
}


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


def _archive_members(directory: Path) -> set[str]:
    result = set()

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
                result.add(relative.as_posix())
            else:
                raise ValueError("Archive contains a special file: " + relative.as_posix())

    visit(directory, Path())
    return result


class Archive:
    """Verify a lossless JSON collection before decoding any member."""

    def __init__(self, directory, expected_sha: str, schema: str):
        supplied = Path(directory)
        if supplied.is_symlink():
            raise ValueError("Archive directory must be nonsymlink")
        self.directory = supplied.resolve(strict=True)
        if not self.directory.is_dir() or SHA_RE.fullmatch(expected_sha or "") is None:
            raise ValueError("Archive directory or collection SHA256 is invalid")
        manifest_path = self.directory / "collection.json"
        _regular(manifest_path, "Collection manifest")
        if sha256_path(manifest_path) != expected_sha:
            raise ValueError("Collection manifest differs from supplied SHA256")
        self.manifest_sha256 = expected_sha
        self.manifest = load_json(manifest_path)
        if type(self.manifest) is not dict or self.manifest.get("schema") != schema \
                or self.manifest.get("status") != "complete":
            raise ValueError("Collection schema/status differs")
        rows = self.manifest.get("files")
        if type(rows) is not list or self.manifest.get("json_count") != len(rows):
            raise ValueError("Collection row count differs")
        self.rows, archives = {}, set()
        for row in rows:
            expected = {"original", "original_bytes", "original_sha256",
                        "archive", "archive_bytes", "archive_sha256"}
            if type(row) is not dict or set(row) != expected:
                raise ValueError("Collection row topology differs")
            original, archive = row["original"], row["archive"]
            if not _safe_relative(original) or not _safe_relative(archive) \
                    or not archive.endswith(".gz") or original in self.rows \
                    or archive in archives:
                raise ValueError("Collection paths are unsafe or duplicated")
            for name in ("original_bytes", "archive_bytes"):
                if type(row[name]) is not int or isinstance(row[name], bool) or row[name] < 0:
                    raise ValueError("Collection byte count differs")
            if SHA_RE.fullmatch(row["original_sha256"] or "") is None \
                    or SHA_RE.fullmatch(row["archive_sha256"] or "") is None:
                raise ValueError("Collection digest differs")
            self.rows[original] = row
            archives.add(archive)
        if _archive_members(self.directory) != archives | {"collection.json"}:
            raise ValueError("Archive membership differs")
        if self.manifest.get("original_bytes") != sum(r["original_bytes"] for r in rows) \
                or self.manifest.get("archive_bytes") != sum(r["archive_bytes"] for r in rows):
            raise ValueError("Collection byte totals differ")
        self.objects = {}
        for original, row in self.rows.items():
            path = self.directory / row["archive"]
            _regular(path, "Gzip member")
            if path.stat().st_size != row["archive_bytes"] \
                    or sha256_path(path) != row["archive_sha256"]:
                raise ValueError("Gzip member differs: " + original)
            size, digest = 0, hashlib.sha256()
            with gzip.open(path, "rb") as handle:
                while block := handle.read(CHUNK):
                    size += len(block)
                    if size > row["original_bytes"]:
                        raise ValueError("Gzip expands beyond declared size")
                    digest.update(block)
            if size != row["original_bytes"] or digest.hexdigest() != row["original_sha256"]:
                raise ValueError("Original JSON roundtrip differs: " + original)

    def require(self, name: str):
        if name not in self.rows:
            raise ValueError("Required archived JSON is absent: " + name)
        if name not in self.objects:
            with gzip.open(self.directory / self.rows[name]["archive"], "rt",
                           encoding="utf-8") as handle:
                self.objects[name] = json.load(handle, object_pairs_hook=_object,
                                               parse_constant=_invalid_constant)
        return self.objects[name]

    def matches_record(self, original: str, record) -> bool:
        row = self.rows.get(original)
        return (type(record) is dict and row is not None
                and record.get("name") == Path(original).name
                and record.get("bytes") == row["original_bytes"]
                and record.get("sha256") == row["original_sha256"])


def _completion(archive: Archive, phase: str, schema: str) -> dict:
    value = archive.require(f"{phase}/completion.json")
    expected = archive.manifest.get("phase_completion_sha256", {}).get(phase)
    row = archive.rows.get(f"{phase}/completion.json")
    if type(value) is not dict or value.get("schema") != schema \
            or value.get("status") != "complete" or value.get("phase") != phase \
            or row is None or row["original_sha256"] != expected:
        raise ValueError("Phase completion binding differs: " + phase)
    return value


def _artifact_index(completion: dict) -> dict[str, dict]:
    rows = completion.get("artifacts")
    if type(rows) is not list:
        raise ValueError("Completion artifact list is absent")
    result = {row.get("name"): row for row in rows if type(row) is dict}
    if len(result) != len(rows) or None in result:
        raise ValueError("Completion artifact membership differs")
    return result


def _exact_archive_membership(archive: Archive, completion_schema: str,
                              attempt_schema: str) -> None:
    expected = set()
    for phase in ("smoke", "confirmation"):
        completion = _completion(archive, phase, completion_schema)
        index = _artifact_index(completion)
        for name, record in index.items():
            if name.endswith(".json"):
                original = f"{phase}/{name}"
                if not archive.matches_record(original, record):
                    raise ValueError("Archived artifact record differs: " + original)
                expected.add(original)
        expected.add(f"{phase}/completion.json")
        attempt = f"attempt-{phase}.json"
        if archive.rows.get(attempt, {}).get("original_sha256") \
                != archive.manifest.get("attempt_sha256", {}).get(phase):
            raise ValueError("Archived attempt binding differs: " + phase)
        attempt_value = archive.require(attempt)
        provenance = archive.manifest.get("source_provenance", {}).get(phase)
        if type(provenance) is not dict or type(attempt_value) is not dict \
                or attempt_value.get("schema") != attempt_schema \
                or attempt_value.get("phase") != phase \
                or attempt_value.get("restart") != "forbidden" \
                or attempt_value.get("frozen_commit") != completion.get("frozen_commit") \
                or attempt_value.get("source_hashes") != completion.get("source_hashes") \
                or provenance.get("frozen_commit") != completion.get("frozen_commit") \
                or provenance.get("source_hashes") != completion.get("source_hashes"):
            raise ValueError("Archived attempt/source provenance differs: " + phase)
        expected.add(attempt)
    if set(archive.rows) != expected:
        raise ValueError("Collected JSON membership differs from phase completions")


def _record(archive: Archive, index: dict[str, dict], record, label: str):
    name = record.get("name") if type(record) is dict else None
    if record != index.get(name) or not archive.matches_record("confirmation/" + str(name), record):
        raise ValueError(label + " artifact record differs")
    return archive.require("confirmation/" + name)


def _curve(branch) -> list[dict]:
    curve = branch.get("curve") if type(branch) is dict else None
    if type(curve) is not list:
        raise ValueError("Branch curve is absent")
    points = [row for row in curve if type(row) is dict and row.get("horizon") in HORIZONS]
    if branch.get("status") == "complete" and [row.get("horizon") for row in points] != list(HORIZONS):
        raise ValueError("Complete branch lacks scheduled curve")
    return points


def load_branches(i17: Archive, i16: Archive) -> dict[tuple[int, str, str], dict]:
    """Reconstruct the 30 logical curves and recorded steps from original JSON."""
    i17_completion = _completion(i17, "confirmation", "i17_completion_v1")
    i17_index = _artifact_index(i17_completion)
    binding = i17.require("confirmation/parent-inputs.json")
    if type(binding) is not dict or binding.get("schema") != "i17_parent_and_reference_binding_v1" \
            or binding.get("source_replayed") is not False:
        raise ValueError("I17 parent/reference binding differs")
    i16_binding = binding.get("i16")
    if type(i16_binding) is not dict or i16_binding.get("artifact_root") != i16.manifest.get("source_root") \
            or i16_binding.get("confirmation_sha256") \
                != i16.manifest.get("phase_completion_sha256", {}).get("confirmation") \
            or i16_binding.get("pinned_sha256", {}).get("raw-results-001/collection.json") \
                != i16.manifest_sha256:
        raise ValueError("I16 archive is not the directly bound reference collection")

    i16_completion = _completion(i16, "confirmation", "i16_completion_v1")
    i16_index = _artifact_index(i16_completion)
    references = binding.get("k0_references")
    expected_pairs = {(seed, target) for seed in SEEDS for target in TARGETS}
    if type(references) is not list or len(references) != 6 \
            or {(r.get("seed"), r.get("target")) for r in references if type(r) is dict} != expected_pairs:
        raise ValueError("I16 k0 reference roster differs")
    branches = {}
    for reference in references:
        key = reference["seed"], reference["target"]
        branch = _record(i16, i16_index, reference.get("curve_artifact"), "I16 k0")
        if branch.get("schema") != "i16_scalar_branch_v1" \
                or (branch.get("seed"), branch.get("target"), branch.get("k"), branch.get("k_label")) \
                != (key[0], key[1], 0.0, "k0") or branch.get("status") != "complete" \
                or branch.get("failure") is not None \
                or branch.get("parent_state_digest") != reference.get("parent_state_digest") \
                or branch.get("parent_evaluation_digest") != reference.get("parent_evaluation_digest"):
            raise ValueError("I16 k0 branch identity differs")
        checkpoints = branch.get("checkpoints")
        if reference.get("checkpoint_records") != checkpoints or type(checkpoints) is not list:
            raise ValueError("I16 k0 checkpoint projection differs")
        for checkpoint in checkpoints:
            kind = "full_state" if checkpoint.get("horizon") == 2000 else "model_state"
            record = checkpoint.get(kind)
            if record != i16_index.get(record.get("name") if type(record) is dict else None):
                raise ValueError("I16 k0 checkpoint binding differs")
        branches[key + ("scalar_k0",)] = {
            "status": "complete", "failure": None, "curve": _curve(branch),
            "steps": branch.get("steps", [])}

    envelope = i17.require("confirmation/branches.json")
    entries = envelope.get("entries") if type(envelope) is dict else None
    expected_new = {(seed, target, policy) for seed in SEEDS for target in TARGETS
                    for policy in NEW_POLICIES}
    if type(entries) is not list or len(entries) != 24 \
            or {(r.get("seed"), r.get("target"), r.get("policy"))
                for r in entries if type(r) is dict} != expected_new:
        raise ValueError("I17 branch index roster differs")
    for entry in entries:
        key = entry["seed"], entry["target"], entry["policy"]
        branch = _record(i17, i17_index, entry.get("artifact"), "I17 branch")
        fields = ("id", "seed", "target", "policy", "status", "completed_updates",
                  "parent_state_digest", "parent_evaluation_digest",
                  "first_step_digests", "update_seconds")
        if branch.get("schema") != "i17_gain_branch_v1" \
                or any(branch.get(name) != entry.get(name) for name in fields) \
                or branch.get("status") not in ("complete", "numerical_failure"):
            raise ValueError("I17 branch/index identity differs")
        branches[key] = {"status": branch["status"], "failure": branch.get("failure"),
                         "curve": _curve(branch), "steps": branch.get("steps", [])}
    expected_all = {(seed, target, policy) for seed in SEEDS for target in TARGETS
                    for policy in POLICIES}
    if set(branches) != expected_all:
        raise ValueError("Five-policy logical membership differs")
    return branches


def _point(branch, horizon):
    if branch is None or branch.get("status") != "complete":
        return None
    return next((row for row in branch["curve"] if row["horizon"] == horizon), None)


def _utility(point, metric):
    return -point["auxiliary"]["clean_ce"] if metric == "ce" else point["auxiliary"]["clean_accuracy"]


def _effect(values):
    ordered = [values[str(seed)] for seed in SEEDS]
    available = all(value is not None for value in ordered)
    return {"per_seed": values, "all_three_seed_values": ordered,
            "available": available, "mean": sum(ordered) / 3 if available else None,
            "no_survivor_averaging": True}


def _best_policy(branch, selector):
    if branch is None or branch.get("status") != "complete":
        return None
    field = "clean_ce" if selector == SELECTORS[0] else "clean_accuracy"
    sign = 1 if selector == SELECTORS[0] else -1
    return min(branch["curve"], key=lambda row: (sign * row["validation"][field], row["horizon"]))


def _best_scalar(branches, seed, target, selector):
    members = [branches.get((seed, target, policy)) for policy in SCALARS]
    if any(row is None or row.get("status") != "complete" for row in members):
        return None
    field = "clean_ce" if selector == SELECTORS[0] else "clean_accuracy"
    sign = 1 if selector == SELECTORS[0] else -1
    candidates = []
    for policy, branch in zip(SCALARS, members):
        for point in branch["curve"]:
            candidates.append(((sign * point["validation"][field], point["horizon"],
                                K_BY_POLICY[policy]), point, K_BY_POLICY[policy]))
    _, point, k = min(candidates, key=lambda row: row[0])
    return point, k


def selection_results(branches):
    choices, primary, per_k, selected = [], [], [], {}
    for seed in SEEDS:
        for target in TARGETS:
            for selector in SELECTORS:
                pair = _best_scalar(branches, seed, target, selector)
                scalar, k = pair if pair is not None else (None, None)
                spectral = _best_policy(branches.get((seed, target, SPECTRAL)), selector)
                selected[seed, target, selector] = scalar, spectral
                field = "clean_ce" if selector == SELECTORS[0] else "clean_accuracy"

                def choice(point, *, scalar_k=None):
                    if point is None:
                        return None
                    identity = {"k": scalar_k} if scalar_k is not None else {"policy": SPECTRAL}
                    return {**identity, "horizon": point["horizon"],
                            "validation_value": point["validation"][field],
                            "auxiliary_clean_ce": point["auxiliary"]["clean_ce"],
                            "auxiliary_clean_accuracy": point["auxiliary"]["clean_accuracy"]}

                choices.append({"seed": seed, "target": target, "selector": selector,
                                "scalar": choice(scalar, scalar_k=k),
                                "spectral": choice(spectral)})
    for target in TARGETS:
        for selector in SELECTORS:
            for metric in METRICS:
                values = {}
                for seed in SEEDS:
                    scalar, spectral = selected[seed, target, selector]
                    values[str(seed)] = (None if scalar is None or spectral is None else
                                         _utility(spectral, metric) - _utility(scalar, metric))
                primary.append({"target": target, "selector": selector,
                                "auxiliary_metric": metric, "effect": _effect(values)})
                for policy in SCALARS:
                    fixed = {}
                    for seed in SEEDS:
                        spectral = selected[seed, target, selector][1]
                        scalar = _best_policy(branches.get((seed, target, policy)), selector)
                        fixed[str(seed)] = (None if scalar is None or spectral is None else
                                            _utility(spectral, metric) - _utility(scalar, metric))
                    per_k.append({"k": K_BY_POLICY[policy], "k_label": policy,
                                  "target": target, "selector": selector,
                                  "auxiliary_metric": metric, "effect": _effect(fixed)})
    return choices, primary, per_k


def endpoint_effects(branches):
    rows = []
    for policy in SCALARS:
        for target in TARGETS:
            for metric in METRICS:
                values = {}
                for seed in SEEDS:
                    scalar = _point(branches.get((seed, target, policy)), 2000)
                    spectral = _point(branches.get((seed, target, SPECTRAL)), 2000)
                    values[str(seed)] = (None if scalar is None or spectral is None else
                                         _utility(spectral, metric) - _utility(scalar, metric))
                rows.append({"k": K_BY_POLICY[policy], "k_label": policy,
                             "target": target, "metric": metric, "horizon": 2000,
                             "effect": _effect(values)})
    return rows


def trajectory_rows(branches):
    rows = []
    for seed, target, policy in sorted(branches):
        branch = branches[seed, target, policy]
        start = next((point for point in branch["curve"] if point["horizon"] == 100), None)
        curve = []
        for point in branch["curve"]:
            values = {name: point[split][field]
                      for name, (split, field) in METRIC_PATHS.items()}
            values["absolute_progress_from_h100"] = (None if start is None else {
                name + "_change": point[split][field] - start[split][field]
                for name, (split, field) in METRIC_PATHS.items()})
            curve.append({"horizon": point["horizon"], "values": values})
        rows.append({"seed": seed, "target": target, "policy": policy,
                     "status": branch["status"], "failure": branch.get("failure"),
                     "curve": curve})
    return rows


def scheduled_aggregates(branches):
    rows = []
    for target in TARGETS:
        for policy in POLICIES:
            for horizon in HORIZONS:
                for metric, (split, field) in METRIC_PATHS.items():
                    values, changes = {}, {}
                    for seed in SEEDS:
                        point = _point(branches.get((seed, target, policy)), horizon)
                        start = _point(branches.get((seed, target, policy)), 100)
                        values[str(seed)] = None if point is None else point[split][field]
                        changes[str(seed)] = (None if point is None or start is None else
                                              point[split][field] - start[split][field])
                    available = all(value is not None for value in values.values())
                    rows.append({"target": target, "policy": policy, "horizon": horizon,
                                 "metric": metric, "per_seed": values,
                                 "equal_seed_mean": sum(values.values()) / 3 if available else None,
                                 "per_seed_change_from_h100": changes,
                                 "equal_seed_mean_change_from_h100":
                                     sum(changes.values()) / 3 if available else None,
                                 "available": available, "no_survivor_averaging": True})
    return rows


def _equal_seed_tree(values):
    if any(value is None for value in values):
        return None
    if type(values[0]) is dict:
        return {key: _equal_seed_tree([value[key] for value in values]) for key in values[0]}
    return sum(values) / len(SEEDS)


def geometry_results(branches):
    rows = []
    for target in TARGETS:
        for policy in POLICIES:
            for window, (low, high) in WINDOWS.items():
                per_seed = {}
                for seed in SEEDS:
                    branch = branches.get((seed, target, policy))
                    steps = ([row for row in branch["steps"] if low <= row.get("step", -1) <= high]
                             if branch is not None and branch.get("status") == "complete" else [])
                    if [row.get("step") for row in steps] != list(range(low, high + 1)):
                        per_seed[str(seed)] = None
                        continue
                    value = {}
                    displacements = ("data", "total") + (() if policy == "scalar_k0" else ("ideal_data",))
                    for name in displacements:
                        norms = [row["displacement"][name + "_norm"] for row in steps]
                        energy = sum(norm * norm for norm in norms)
                        leaks = [row["displacement"][name + "_current_basis_leakage"] for row in steps]
                        valid = all(leak.get("reason") in (None, "zero_displacement") for leak in leaks)
                        value[name] = {
                            "squared_energy_sum": energy,
                            "path_length_sum": sum(norms),
                            "current_action_complement_energy_fraction":
                                (sum(leak["outside_squared_norm"] for leak in leaks) / energy
                                 if energy > 0 and valid else None)}
                    for name in ("raw_gradient", "post_mean"):
                        dots = [row["displacement"][name + "_dot_data_delta"] for row in steps]
                        value["sum_" + name + "_dot_data_step"] = sum(dots)
                        value["mean_" + name + "_dot_data_step"] = sum(dots) / len(steps)
                    per_seed[str(seed)] = value
                rows.append({"target": target, "policy": policy,
                             "k": K_BY_POLICY.get(policy), "window": window,
                             "per_seed": per_seed,
                             "equal_seed_mean": _equal_seed_tree(list(per_seed.values())),
                             "available": all(value is not None for value in per_seed.values()),
                             "aggregation": "within branch, then equal seeds; exact complete window required"})
    return rows


class Comparison:
    def __init__(self):
        self.errors, self.scalars, self.maximum = [], 0, 0.0

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
            self.maximum = max(self.maximum, difference)
            if not math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12):
                self.errors.append(path + f": numeric difference {left!r} vs {right!r}")
            return
        self.scalars += 1
        if type(observed) is not type(expected) or observed != expected:
            self.errors.append(path + f": value differs {observed!r} vs {expected!r}")


def _pinned_json(path, digest, label):
    path = Path(path)
    if path.is_symlink():
        raise ValueError(label + " must be nonsymlink")
    resolved = path.resolve(strict=True)
    _regular(resolved, label)
    if SHA_RE.fullmatch(digest or "") is None or sha256_path(resolved) != digest:
        raise ValueError(label + " differs from supplied SHA256")
    return resolved, load_json(resolved)


def corroborate(i17_archive, i17_collection_sha256, i16_archive,
                i16_collection_sha256, summary_path, summary_sha256,
                audit_path, audit_sha256, output):
    output = Path(output)
    if not output.is_absolute() or output.name != "report-audit.json" \
            or output.parent.name != "analysis-001" or output.parent.is_symlink():
        raise ValueError("Output must be an absolute analysis-001/report-audit.json")
    parent = output.parent.resolve(strict=True)
    output = parent / output.name
    if output.exists() or output.is_symlink():
        raise FileExistsError("Report audit output already exists")
    summary_path, summary = _pinned_json(summary_path, summary_sha256, "I17 summary")
    audit_path, audit = _pinned_json(audit_path, audit_sha256, "I17 audit")
    if summary_path.parent.resolve() != parent or audit_path.parent.resolve() != parent:
        raise ValueError("Summary, audit, and report audit must share analysis-001")
    if type(audit) is not dict or audit.get("schema") != "i17_gain_analysis_audit_v1" \
            or audit.get("status") != "pass" or audit.get("errors") != []:
        raise ValueError("I17 primary audit is not passing")
    i17 = Archive(i17_archive, i17_collection_sha256, "i17_gain_collection_v1")
    i16 = Archive(i16_archive, i16_collection_sha256, "i16_scalar_collection_v1")
    _exact_archive_membership(i17, "i17_completion_v1", "i17_attempt_v1")
    _exact_archive_membership(i16, "i16_completion_v1", "i16_attempt_v1")
    root = i17.manifest.get("source_root")
    if audit.get("artifact_root") != root \
            or i17.manifest.get("analysis_audit", {}).get("sha256") != audit_sha256 \
            or i17.manifest.get("analysis_audit", {}).get("status") != "pass" \
            or i17.manifest.get("analysis_audit", {}).get("schema") \
                != "i17_gain_analysis_audit_v1" \
            or i17.manifest.get("analysis_audit", {}).get("path") != str(audit_path) \
            or i17.manifest.get("phase_completion_sha256") != audit.get("phase_completion_sha256") \
            or i17.manifest.get("attempt_sha256") != audit.get("attempt_sha256") \
            or i17.manifest.get("runtime_inventory") != audit.get("runtime_inventory") \
            or type(summary) is not dict or summary.get("schema") != "i17_gain_analysis_summary_v1" \
            or summary.get("audit_status") != "pass" or summary.get("artifact_root") != root:
        raise ValueError("I17 summary/audit/archive provenance differs")

    branches = load_branches(i17, i16)
    choices, primary, per_k = selection_results(branches)
    recomputed = {
        "primary_validation_selected_spectral_minus_scalar": primary,
        "validation_selected_choices": choices,
        "validation_selected_spectral_minus_each_k": per_k,
        "fixed_k_endpoint_spectral_minus_scalar": endpoint_effects(branches),
        "all_policy_trajectories": trajectory_rows(branches),
        "scheduled_metric_equal_seed_aggregates": scheduled_aggregates(branches),
        "path_and_displacement_geometry": geometry_results(branches),
    }
    expected_counts = {
        "primary_validation_selected_spectral_minus_scalar": 8,
        "validation_selected_choices": 12,
        "validation_selected_spectral_minus_each_k": 32,
        "fixed_k_endpoint_spectral_minus_scalar": 16,
        "all_policy_trajectories": 30,
        "scheduled_metric_equal_seed_aggregates": 960,
        "path_and_displacement_geometry": 20,
    }
    comparison, sections = Comparison(), {}
    for name, expected in recomputed.items():
        if len(expected) != expected_counts[name]:
            raise AssertionError("Internal corroborator row count differs: " + name)
        before = len(comparison.errors)
        comparison.compare(summary.get(name), expected, name)
        sections[name] = {"rows": len(expected),
                          "status": "pass" if len(comparison.errors) == before else "fail"}
    complete = sum(row["status"] == "complete" for row in recomputed["all_policy_trajectories"])
    points = sum(len(row["curve"]) for row in recomputed["all_policy_trajectories"])
    report = {
        "schema": "i17_gain_report_corroboration_v1",
        "status": "pass" if not comparison.errors else "fail",
        "inputs": {
            "i17_collection": {"path": str(Path(i17_archive).resolve()),
                               "sha256": i17_collection_sha256},
            "i16_collection": {"path": str(Path(i16_archive).resolve()),
                               "sha256": i16_collection_sha256},
            "i17_summary": {"path": str(summary_path), "sha256": summary_sha256},
            "i17_audit": {"path": str(audit_path), "sha256": audit_sha256}},
        "scope": ("Independent standard-library recomputation from lossless JSON only; "
                  "no tensors, model forward, optimizer replay, or primary integrity-audit replay."),
        "sections": sections, "complete_trajectories": complete,
        "retained_scheduled_points": points,
        "expected_points_if_all_complete": 30 * 6,
        "numeric_and_discrete_scalars_compared": comparison.scalars,
        "maximum_absolute_numeric_difference": comparison.maximum,
        "no_survivor_averaging": True, "errors": comparison.errors}
    with output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=True, allow_nan=False)
        handle.write("\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--i17-archive", required=True, type=Path)
    parser.add_argument("--i17-collection-sha256", required=True)
    parser.add_argument("--i16-archive", required=True, type=Path)
    parser.add_argument("--i16-collection-sha256", required=True)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--summary-sha256", required=True)
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument("--audit-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = corroborate(args.i17_archive, args.i17_collection_sha256,
        args.i16_archive, args.i16_collection_sha256,
        args.summary, args.summary_sha256, args.audit, args.audit_sha256, args.output)
    print(json.dumps({key: value for key, value in report.items()
                      if key not in ("sections", "errors")}, allow_nan=False))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
