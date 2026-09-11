#!/usr/bin/env python3
"""Independently corroborate I16 report scalars from lossless JSON archives."""
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
SCALAR_LABELS = ("k0", "k0p5", "k0p9", "k1")
NEW_LABELS = SCALAR_LABELS[:3]
SPECTRAL = "mean_projected_history"
POLICIES = SCALAR_LABELS + (SPECTRAL,)
K_BY_LABEL = {"k0": 0.0, "k0p5": 0.5, "k0p9": 0.9, "k1": 1.0}
HORIZONS = (100, 250, 500, 1000, 1500, 2000)
SELECTORS = ("minimum_validation_ce", "maximum_validation_accuracy")
METRICS = ("ce", "accuracy")
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


def _archive_members(directory: Path) -> set[str]:
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
    """Eagerly verify and decode one small, lossless scalar-JSON collection."""

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
        self.manifest_sha256 = expected_sha
        self.manifest = load_json(manifest_path)
        if type(self.manifest) is not dict or self.manifest.get("schema") != schema \
                or self.manifest.get("status") != "complete":
            raise ValueError("Collection manifest schema or status differs")
        rows = self.manifest.get("files")
        if type(rows) is not list or self.manifest.get("json_count") != len(rows):
            raise ValueError("Collection row count differs")
        self.rows, archive_names = {}, set()
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
            for field in ("original_bytes", "archive_bytes"):
                if type(row[field]) is not int or isinstance(row[field], bool) \
                        or row[field] < 0:
                    raise ValueError("Collection byte count differs")
            if SHA_RE.fullmatch(row["original_sha256"] or "") is None \
                    or SHA_RE.fullmatch(row["archive_sha256"] or "") is None:
                raise ValueError("Collection digest differs")
            self.rows[original] = row
            archive_names.add(archive)
        if _archive_members(self.directory) != archive_names | {"collection.json"}:
            raise ValueError("Archive membership differs from collection manifest")
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
            size, digest = 0, hashlib.sha256()
            with gzip.open(path, "rb") as handle:
                while block := handle.read(CHUNK):
                    size += len(block)
                    if size > row["original_bytes"]:
                        raise ValueError("Gzip expands beyond declared size")
                    digest.update(block)
            if size != row["original_bytes"] \
                    or digest.hexdigest() != row["original_sha256"]:
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


def _curve_points(branch) -> list[dict]:
    curve = branch.get("curve")
    if type(curve) is not list:
        raise ValueError("Trajectory curve is absent")
    points = [row for row in curve if type(row) is dict
              and row.get("horizon") in HORIZONS]
    if branch.get("status") == "complete" \
            and [row.get("horizon") for row in points] != list(HORIZONS):
        raise ValueError("Complete trajectory lacks all scheduled horizons")
    return points


def _completion(archive: Archive, phase: str, schema: str) -> dict:
    value = archive.require(f"{phase}/completion.json")
    expected_sha = archive.manifest.get("phase_completion_sha256", {}).get(phase)
    if expected_sha is None:  # The accepted I14 collector used this older field name.
        expected_sha = archive.manifest.get("source_completion_sha256", {}).get(phase)
    row = archive.rows.get(f"{phase}/completion.json")
    if type(value) is not dict or value.get("schema") != schema \
            or value.get("status") != "complete" or value.get("phase") != phase \
            or row is None or row["original_sha256"] != expected_sha:
        raise ValueError("Archived phase completion binding differs: " + phase)
    return value


def _artifact_index(completion: dict) -> dict[str, dict]:
    rows = completion.get("artifacts")
    if type(rows) is not list:
        raise ValueError("Completion artifact list is absent")
    result = {row.get("name"): row for row in rows if type(row) is dict}
    if len(result) != len(rows) or None in result:
        raise ValueError("Completion artifact membership differs")
    return result


def _reference_record(archive: Archive, completion_index: dict[str, dict], record,
                      label: str):
    name = record.get("name") if type(record) is dict else None
    if record != completion_index.get(name) \
            or not archive.matches_record("confirmation/" + str(name), record):
        raise ValueError(label + " record differs from archive")
    return archive.require("confirmation/" + name)


def load_branches(i16: Archive, i15: Archive, i14: Archive) \
        -> dict[tuple[int, str, str], dict]:
    """Reconstruct 30 logical trajectories without tensor or source replay."""
    binding = i16.require("confirmation/parent-inputs.json")
    if type(binding) is not dict or binding.get("schema") \
            != "i16_parent_and_reference_binding_v1" \
            or binding.get("source_replayed") is not False:
        raise ValueError("I16 parent/reference binding differs")
    i14_binding = binding.get("i14")
    if type(i14_binding) is not dict or i14_binding.get("schema") \
            != "i15_i14_inputs_v1" \
            or i14.manifest.get("source_directory") != i14_binding.get("source_root") \
            or i14.manifest.get("source_completion_sha256", {}).get("confirmation") \
            != i14_binding.get("confirmation_completion_sha256") \
            or i14.manifest.get("runtime_directory_supplement_sha256") \
            != i14_binding.get("runtime_supplement_sha256"):
        raise ValueError("I14 archive/reference binding differs")
    i15_binding = binding.get("i15")
    i15_completion = _completion(i15, "confirmation", "i15_completion_v1")
    i15_index = _artifact_index(i15_completion)
    if type(i15_binding) is not dict \
            or i15.manifest.get("source_root") != i15_binding.get("artifact_root") \
            or i15.manifest_sha256 != i15_binding.get("collection_sha256") \
            or i15.manifest.get("analysis_audit", {}).get("sha256") \
            != i15_binding.get("audit_sha256") \
            or i15.manifest.get("phase_completion_sha256", {}).get("confirmation") \
            != i15_binding.get("confirmation_completion_sha256"):
        raise ValueError("I15 archive/reference binding differs")

    i14_completion = _completion(i14, "confirmation", "i14_completion_v1")
    i14_index = _artifact_index(i14_completion)
    i14_references = i14_binding.get("references")
    expected_i14 = {(seed, target, policy) for seed in SEEDS for target in TARGETS
                    for policy in ("raw", "current32")}
    if type(i14_references) is not list or len(i14_references) != 12 \
            or {(row.get("seed"), row.get("target"), row.get("policy"))
                for row in i14_references if type(row) is dict} != expected_i14:
        raise ValueError("I14 reference membership differs")
    raw_direct = {(row.get("seed"), row.get("target")): row
                  for row in binding.get("raw_references", [])
                  if type(row) is dict}
    if set(raw_direct) != {(seed, target) for seed in SEEDS for target in TARGETS}:
        raise ValueError("I16 raw reference membership differs")

    branches = {}
    for reference in i14_references:
        if reference["policy"] != "raw":
            continue
        key = reference["seed"], reference["target"]
        record = reference.get("curve_artifact")
        curve = _reference_record(i14, i14_index, record, "I14 raw")
        direct = raw_direct[key]
        if direct.get("curve") != curve or curve.get("schema") != "i14_trajectory_v1" \
                or (curve.get("seed"), curve.get("target"), curve.get("policy")) \
                != (key[0], key[1], "raw") or curve.get("base") != "sgdm" \
                or curve.get("lr") != .03 or curve.get("status") != "complete":
            raise ValueError("I14 raw curve identity or embedded copy differs")
        branches[key + ("k1",)] = {"seed": key[0], "target": key[1],
            "policy": "k1", "status": "complete", "failure": None,
            "curve": _curve_points(curve),
            "steps": [row for row in curve.get("steps", [])
                      if type(row) is dict and row.get("step", 0) >= 101]}

    spectral_records = binding.get("spectral_references")
    if type(spectral_records) is not list or len(spectral_records) != 6:
        raise ValueError("I15 spectral reference membership differs")
    observed_spectral = set()
    for reference in spectral_records:
        key = reference.get("seed"), reference.get("target")
        observed_spectral.add(key)
        branch = _reference_record(i15, i15_index, reference.get("curve_artifact"),
                                   "I15 spectral")
        if branch.get("schema") != "i15_history_branch_v1" \
                or (branch.get("seed"), branch.get("target"), branch.get("policy")) \
                != (key[0], key[1], SPECTRAL) \
                or branch.get("status") != "complete" or branch.get("failure") is not None \
                or branch.get("parent_state_digest") \
                    != reference.get("parent_state_digest") \
                or branch.get("parent_evaluation_digest") \
                    != reference.get("parent_evaluation_digest"):
            raise ValueError("I15 spectral branch identity differs")
        expected_checkpoints = []
        for checkpoint in branch.get("checkpoints", []):
            horizon = checkpoint.get("horizon") if type(checkpoint) is dict else None
            kind = "full_state" if horizon == 2000 else "model_state"
            expected_checkpoints.append({"horizon": horizon, kind: checkpoint.get(kind)})
            record = checkpoint.get(kind) if type(checkpoint) is dict else None
            if record != i15_index.get(record.get("name") if type(record) is dict else None):
                raise ValueError("I15 spectral checkpoint completion binding differs")
        if reference.get("checkpoint_records") != expected_checkpoints:
            raise ValueError("I15 spectral checkpoint projection differs")
        branches[key + (SPECTRAL,)] = {"seed": key[0], "target": key[1],
            "policy": SPECTRAL, "status": "complete", "failure": None,
            "curve": _curve_points(branch), "steps": branch.get("steps", [])}
    if observed_spectral != {(seed, target) for seed in SEEDS for target in TARGETS}:
        raise ValueError("I15 spectral identities differ")

    i16_completion = _completion(i16, "confirmation", "i16_completion_v1")
    i16_index = _artifact_index(i16_completion)
    envelope = i16.require("confirmation/branches.json")
    entries = envelope.get("entries") if type(envelope) is dict else None
    expected_new = {(seed, target, label) for seed in SEEDS for target in TARGETS
                    for label in NEW_LABELS}
    if type(entries) is not list or len(entries) != 18:
        raise ValueError("I16 branch index membership differs")
    observed_new = set()
    for entry in entries:
        key = entry.get("seed"), entry.get("target"), entry.get("k_label")
        observed_new.add(key)
        branch = _reference_record(i16, i16_index, entry.get("artifact"), "I16 branch")
        identity = (branch.get("seed"), branch.get("target"), branch.get("k_label"))
        if branch.get("schema") != "i16_scalar_branch_v1" or identity != key \
                or branch.get("status") not in ("complete", "numerical_failure") \
                or any(branch.get(field) != entry.get(field) for field in (
                    "id", "seed", "target", "k", "k_label", "status",
                    "completed_updates", "parent_state_digest",
                    "parent_evaluation_digest", "first_step_digests",
                    "scalar_update_seconds")):
            raise ValueError("I16 branch identity or index binding differs")
        branches[key] = {"seed": key[0], "target": key[1], "policy": key[2],
            "status": branch["status"], "failure": branch.get("failure"),
            "curve": _curve_points(branch), "steps": branch.get("steps", [])}
    if observed_new != expected_new:
        raise ValueError("I16 new branch identities differ")
    expected_all = {(seed, target, policy) for seed in SEEDS for target in TARGETS
                    for policy in POLICIES}
    if set(branches) != expected_all:
        raise ValueError("Five-policy trajectory membership differs")
    return branches


def _point(branch, horizon):
    if branch is None or branch.get("status") != "complete":
        return None
    return next((row for row in branch["curve"] if row["horizon"] == horizon), None)


def _utility(point, metric):
    return (-point["auxiliary"]["clean_ce"] if metric == "ce"
            else point["auxiliary"]["clean_accuracy"])


def _effect(values):
    ordered = [values[str(seed)] for seed in SEEDS]
    available = all(value is not None for value in ordered)
    return {"per_seed": values, "all_three_seed_values": ordered,
            "available": available,
            "mean": sum(ordered) / 3 if available else None,
            "no_survivor_averaging": True}


def _selection(curve):
    return {
        SELECTORS[0]: min(curve, key=lambda row: (
            row["validation"]["clean_ce"], row["horizon"]))["horizon"],
        SELECTORS[1]: min(curve, key=lambda row: (
            -row["validation"]["clean_accuracy"], row["horizon"]))["horizon"],
    }


def _best_policy(branch, selector):
    if branch is None or branch.get("status") != "complete":
        return None
    return _point(branch, _selection(branch["curve"])[selector])


def _best_scalar(branches, seed, target, selector):
    members = [branches.get((seed, target, label)) for label in SCALAR_LABELS]
    if any(row is None or row.get("status") != "complete" for row in members):
        return None
    candidates = []
    for label, branch in zip(SCALAR_LABELS, members):
        for point in branch["curve"]:
            value = point["validation"][
                "clean_ce" if selector == SELECTORS[0] else "clean_accuracy"]
            order = ((value, point["horizon"], K_BY_LABEL[label])
                     if selector == SELECTORS[0]
                     else (-value, point["horizon"], K_BY_LABEL[label]))
            candidates.append((order, point, K_BY_LABEL[label]))
    _, point, k = min(candidates, key=lambda row: row[0])
    return point, k


def selection_results(branches):
    choices, primary, per_k = [], [], []
    selected = {}
    for seed in SEEDS:
        for target in TARGETS:
            for selector in SELECTORS:
                scalar = _best_scalar(branches, seed, target, selector)
                scalar_point, k = scalar if scalar is not None else (None, None)
                spectral = _best_policy(branches.get((seed, target, SPECTRAL)), selector)
                selected[seed, target, selector] = scalar_point, k, spectral
                choices.append({"seed": seed, "target": target, "selector": selector,
                    "scalar": None if scalar_point is None else {
                        "k": k, "horizon": scalar_point["horizon"],
                        "validation_value": scalar_point["validation"][
                            "clean_ce" if selector == SELECTORS[0] else "clean_accuracy"],
                        "auxiliary_clean_ce": scalar_point["auxiliary"]["clean_ce"],
                        "auxiliary_clean_accuracy": scalar_point["auxiliary"]["clean_accuracy"]},
                    "spectral": None if spectral is None else {
                        "policy": SPECTRAL, "horizon": spectral["horizon"],
                        "validation_value": spectral["validation"][
                            "clean_ce" if selector == SELECTORS[0] else "clean_accuracy"],
                        "auxiliary_clean_ce": spectral["auxiliary"]["clean_ce"],
                        "auxiliary_clean_accuracy": spectral["auxiliary"]["clean_accuracy"]}})
    for target in TARGETS:
        for selector in SELECTORS:
            for metric in METRICS:
                values = {}
                for seed in SEEDS:
                    scalar, _, spectral = selected[seed, target, selector]
                    values[str(seed)] = (None if scalar is None or spectral is None else
                                         _utility(spectral, metric) - _utility(scalar, metric))
                primary.append({"target": target, "selector": selector,
                    "auxiliary_metric": metric, "effect": _effect(values)})
                for label in SCALAR_LABELS:
                    fixed = {}
                    for seed in SEEDS:
                        spectral = selected[seed, target, selector][2]
                        point = _best_policy(branches.get((seed, target, label)), selector)
                        fixed[str(seed)] = (None if spectral is None or point is None else
                            _utility(spectral, metric) - _utility(point, metric))
                    per_k.append({"k": K_BY_LABEL[label], "k_label": label,
                        "target": target, "selector": selector,
                        "auxiliary_metric": metric, "effect": _effect(fixed)})
    return choices, primary, per_k


def endpoint_effects(branches):
    rows = []
    for label in SCALAR_LABELS:
        for target in TARGETS:
            for metric in METRICS:
                values = {}
                for seed in SEEDS:
                    scalar = _point(branches.get((seed, target, label)), 2000)
                    spectral = _point(branches.get((seed, target, SPECTRAL)), 2000)
                    values[str(seed)] = (None if scalar is None or spectral is None else
                                         _utility(spectral, metric) - _utility(scalar, metric))
                rows.append({"k": K_BY_LABEL[label], "k_label": label,
                    "target": target, "metric": metric, "horizon": 2000,
                    "effect": _effect(values)})
    return rows


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


def trajectory_rows(branches):
    rows = []
    for key in sorted(branches):
        branch = branches[key]
        start = next((point for point in branch["curve"]
                      if point["horizon"] == 100), None)
        curve = []
        for point in branch["curve"]:
            values = {name: point[split][field]
                      for name, (split, field) in METRIC_PATHS.items()}
            values["absolute_progress_from_h100"] = None if start is None else {
                name + "_change": point[split][field] - start[split][field]
                for name, (split, field) in METRIC_PATHS.items()}
            curve.append({"horizon": point["horizon"], "values": values})
        rows.append({"seed": key[0], "target": key[1], "policy": key[2],
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
                    rows.append({"target": target, "policy": policy,
                        "horizon": horizon, "metric": metric, "per_seed": values,
                        "equal_seed_mean": sum(values.values()) / 3 if available else None,
                        "per_seed_change_from_h100": changes,
                        "equal_seed_mean_change_from_h100":
                            sum(changes.values()) / 3 if available else None,
                        "available": available, "no_survivor_averaging": True})
    return rows


def _legacy_leak(step, displacement):
    value = step.get("displacement", {}).get(
        displacement + "_current_basis_leakage")
    return value if type(value) is dict else None


def geometry_results(branches):
    paths, components = [], []
    for target in TARGETS:
        for label in POLICIES:
            for window, (low, high) in WINDOWS.items():
                per_seed = {}
                for seed in SEEDS:
                    branch = branches.get((seed, target, label))
                    steps = ([row for row in branch["steps"]
                              if low <= row.get("step", -1) <= high]
                             if branch is not None and branch.get("status") == "complete"
                             else [])
                    if not steps:
                        per_seed[str(seed)] = None
                        continue
                    result = {}
                    for displacement in ("data", "total"):
                        norm_name = displacement + "_norm"
                        norms = [row.get("displacement", {}).get(norm_name) for row in steps]
                        energy = (sum(value * value for value in norms)
                                  if all(type(value) in (int, float)
                                         and math.isfinite(value) for value in norms) else None)
                        leaks = [_legacy_leak(row, displacement) for row in steps]
                        valid_leaks = (all(type(row) is dict and row.get("reason") in
                                           (None, "zero_displacement") for row in leaks))
                        result[displacement] = {
                            "squared_energy_sum": energy,
                            "path_length_sum": (sum(norms) if energy is not None else None),
                            "current_action_complement_energy_fraction":
                                (sum(row["outside_squared_norm"] for row in leaks) / energy
                                 if valid_leaks and energy is not None and energy > 0 else None)}
                    for prefix, field in (
                            ("raw_gradient_dot_data_step", "raw_gradient_dot_data_delta"),
                            ("post_mean_dot_data_step", "post_mean_dot_data_delta"),
                            ("applied_gradient_dot_data_step",
                             "applied_gradient_dot_data_delta")):
                        values = [row.get("displacement", {}).get(field) for row in steps]
                        valid = all(type(value) in (int, float) and math.isfinite(value)
                                    for value in values)
                        result["sum_" + prefix] = sum(values) if valid else None
                        result["mean_" + prefix] = sum(values) / len(values) if valid else None
                    per_seed[str(seed)] = result
                available = all(value is not None for value in per_seed.values())
                paths.append({"target": target, "policy": label,
                    "k": K_BY_LABEL.get(label),
                    "k_label": label if label in SCALAR_LABELS else None,
                    "window": window, "per_seed": per_seed,
                    "equal_seed_mean": (None if not available else {
                        name: ({field: (sum(value[name][field]
                                           for value in per_seed.values()) / 3
                                        if all(value[name][field] is not None
                                               for value in per_seed.values()) else None)
                                for field in per_seed[str(SEEDS[0])][name]}
                               if type(per_seed[str(SEEDS[0])][name]) is dict else
                               (sum(value[name] for value in per_seed.values()) / 3
                                if all(value[name] is not None
                                       for value in per_seed.values()) else None))
                        for name in per_seed[str(SEEDS[0])]}),
                    "available": available,
                    "aggregation": "within branch, then equal seeds"})
                if label not in NEW_LABELS:
                    continue
                for component in ("raw_gradient", "post_mean",
                        "native_projection_diagnostic", "applied_delivery",
                        "old_momentum_buffer", "scaled_old_momentum_buffer",
                        "new_momentum_buffer"):
                    values = {}
                    for seed in SEEDS:
                        branch = branches.get((seed, target, label))
                        steps = ([row for row in branch["steps"]
                                  if low <= row.get("step", -1) <= high]
                                 if branch is not None
                                 and branch.get("status") == "complete" else [])
                        energy = sum(row["components"][component]["squared_energy"]
                                     for row in steps)
                        leaks = [row["components"][component][
                            "current_basis_leakage"] for row in steps]
                        ratio = (sum(row["outside_squared_norm"] for row in leaks) / energy
                                 if steps and energy > 0 and all(row.get("reason") in
                                    (None, "zero_displacement") for row in leaks) else None)
                        values[str(seed)] = None if not steps else {
                            "mean_squared_energy": energy / len(steps),
                            "current_action_complement_energy_fraction": ratio}
                    okay = all(value is not None for value in values.values())
                    ratios = okay and all(value[
                        "current_action_complement_energy_fraction"] is not None
                        for value in values.values())
                    components.append({"target": target, "k": K_BY_LABEL[label],
                        "k_label": label, "window": window, "component": component,
                        "per_seed": values,
                        "equal_seed_mean_squared_energy": (sum(value[
                            "mean_squared_energy"] for value in values.values()) / 3
                            if okay else None),
                        "equal_seed_mean_of_branch_complement_fractions": (sum(value[
                            "current_action_complement_energy_fraction"]
                            for value in values.values()) / 3 if ratios else None),
                        "available": okay,
                        "aggregation": "within branch, then equal seeds"})
    return paths, components


class Comparison:
    def __init__(self):
        self.errors, self.scalars, self.max_abs_difference = [], 0, 0.0

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


def timing_checks(i16: Archive) -> dict:
    values = {}
    completions = {}
    for phase in ("smoke", "confirmation"):
        completion = _completion(i16, phase, "i16_completion_v1")
        index = i16.require(f"{phase}/branches.json")
        entries = index.get("entries") if type(index) is dict else None
        if type(entries) is not list:
            raise ValueError("Timing branch roster differs")
        seconds = sum(row["scalar_update_seconds"] for row in entries)
        updates = sum(row["completed_updates"] for row in entries) \
            + (200 if phase == "smoke" else 0)
        if not math.isclose(seconds, completion.get("scalar_update_seconds", math.nan),
                            rel_tol=1e-12, abs_tol=1e-12) \
                or updates != completion.get("completed_training_updates"):
            raise ValueError("Completion timing/update accounting differs")
        values[phase] = {"branch_scalar_update_seconds_sum": seconds,
                         "completed_training_updates": updates,
                         "elapsed_seconds": completion.get("elapsed_seconds")}
        completions[phase] = completion
    manifest = i16.require("confirmation/manifest.json")
    forecast = values["smoke"]["branch_scalar_update_seconds_sum"] / 60 \
        * 34200 * 1.5 + 60
    if not math.isclose(forecast, manifest.get("smoke_based_seconds_forecast", math.nan),
                        rel_tol=1e-12, abs_tol=1e-12) or forecast > 1800:
        raise ValueError("Confirmation timing admission differs")
    return {"status": "pass", "forecast_seconds": forecast,
            "forecast_limit_seconds": 1800, "phases": values}


def corroborate(i16_archive, i16_collection_sha256, i14_archive,
                i14_collection_sha256, i15_archive, i15_collection_sha256,
                summary_path, summary_sha256, audit_path, audit_sha256, output):
    output = Path(output)
    if not output.is_absolute() or output.name != "report-audit.json" \
            or not output.parent.name.startswith("analysis-"):
        raise ValueError("Output must be an absolute absent analysis-*/report-audit.json")
    output.parent.resolve(strict=True)
    if output.exists() or output.is_symlink():
        raise FileExistsError("Report audit output already exists")
    summary_path, audit_path = Path(summary_path), Path(audit_path)
    for path, digest, label in ((summary_path, summary_sha256, "summary"),
                                (audit_path, audit_sha256, "audit")):
        if path.is_symlink():
            raise ValueError(label + " must be regular and nonsymlink")
        _regular(path.resolve(strict=True), "Pinned I16 " + label)
        if SHA_RE.fullmatch(digest or "") is None or sha256_path(path) != digest:
            raise ValueError("I16 " + label + " differs from supplied SHA256")
    summary, audit = load_json(summary_path), load_json(audit_path)
    if type(audit) is not dict or audit.get("schema") != "i16_scalar_analysis_audit_v1" \
            or audit.get("status") != "pass":
        raise ValueError("I16 audit is not passing")

    i16 = Archive(i16_archive, i16_collection_sha256, "i16_scalar_collection_v1")
    i14 = Archive(i14_archive, i14_collection_sha256, "i14_scalar_collection_v1")
    i15 = Archive(i15_archive, i15_collection_sha256, "i15_scalar_collection_v1")
    root = i16.manifest.get("source_root")
    if audit.get("artifact_root") != root \
            or i16.manifest.get("analysis_audit", {}).get("sha256") != audit_sha256 \
            or i16.manifest.get("analysis_audit", {}).get("status") != "pass" \
            or i16.manifest.get("phase_completion_sha256") \
                != audit.get("phase_completion_sha256") \
            or i16.manifest.get("attempt_sha256") != audit.get("attempt_sha256") \
            or i16.manifest.get("runtime_inventory") != audit.get("runtime_inventory") \
            or type(summary) is not dict \
            or summary.get("schema") != "i16_scalar_analysis_summary_v1" \
            or summary.get("audit_status") != "pass" \
            or summary.get("artifact_root") != root:
        raise ValueError("I16 summary/audit/archive provenance differs")

    branches = load_branches(i16, i15, i14)
    choices, primary, per_k = selection_results(branches)
    paths, components = geometry_results(branches)
    recomputed = {
        "primary_validation_selected_spectral_minus_scalar": primary,
        "validation_selected_choices": choices,
        "validation_selected_spectral_minus_each_k": per_k,
        "fixed_k_endpoint_spectral_minus_scalar": endpoint_effects(branches),
        "all_policy_trajectories": trajectory_rows(branches),
        "scheduled_metric_equal_seed_aggregates": scheduled_aggregates(branches),
        "path_and_displacement_geometry": paths,
        "component_energy_and_action_complement": components,
    }
    expected_counts = {
        "primary_validation_selected_spectral_minus_scalar": 8,
        "validation_selected_choices": 12,
        "validation_selected_spectral_minus_each_k": 32,
        "fixed_k_endpoint_spectral_minus_scalar": 16,
        "all_policy_trajectories": 30,
        "scheduled_metric_equal_seed_aggregates": 960,
        "path_and_displacement_geometry": 20,
        "component_energy_and_action_complement": 84,
    }
    comparison, sections = Comparison(), {}
    for name, expected in recomputed.items():
        if len(expected) != expected_counts[name]:
            raise AssertionError("Internal corroborator row count differs: " + name)
        before = len(comparison.errors)
        comparison.compare(summary.get(name), expected, name)
        sections[name] = {"rows": len(expected),
                          "status": "pass" if len(comparison.errors) == before else "fail"}
    complete = sum(row["status"] == "complete" for row in recomputed[
        "all_policy_trajectories"])
    points = sum(len(row["curve"]) for row in recomputed["all_policy_trajectories"])
    report = {"schema": "i16_report_corroboration_v1",
        "status": "pass" if not comparison.errors else "fail",
        "inputs": {
            "i16_collection": {"path": str(Path(i16_archive).resolve()),
                               "sha256": i16_collection_sha256},
            "i14_collection": {"path": str(Path(i14_archive).resolve()),
                               "sha256": i14_collection_sha256},
            "i15_collection": {"path": str(Path(i15_archive).resolve()),
                               "sha256": i15_collection_sha256},
            "i16_summary": {"path": str(summary_path.resolve()),
                            "sha256": summary_sha256},
            "i16_audit": {"path": str(audit_path.resolve()),
                          "sha256": audit_sha256}},
        "scope": ("Independent standard-library recomputation from lossless JSON only; "
                  "no tensor loading, forward pass, optimizer replay, or semantic audit replay."),
        "sections": sections, "timing_checks": timing_checks(i16),
        "complete_trajectories": complete, "retained_scheduled_points": points,
        "expected_points_if_all_complete": 30 * 6,
        "numeric_and_discrete_scalars_compared": comparison.scalars,
        "maximum_absolute_numeric_difference": comparison.max_abs_difference,
        "no_survivor_averaging": True, "errors": comparison.errors}
    with output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=True, allow_nan=False)
        handle.write("\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--i16-archive", required=True, type=Path)
    parser.add_argument("--i16-collection-sha256", required=True)
    parser.add_argument("--i14-archive", required=True, type=Path)
    parser.add_argument("--i14-collection-sha256", required=True)
    parser.add_argument("--i15-archive", required=True, type=Path)
    parser.add_argument("--i15-collection-sha256", required=True)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--summary-sha256", required=True)
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument("--audit-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = corroborate(args.i16_archive, args.i16_collection_sha256,
        args.i14_archive, args.i14_collection_sha256,
        args.i15_archive, args.i15_collection_sha256,
        args.summary, args.summary_sha256, args.audit, args.audit_sha256, args.output)
    print(json.dumps({key: value for key, value in report.items()
                      if key not in ("sections", "errors")}, allow_nan=False))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
