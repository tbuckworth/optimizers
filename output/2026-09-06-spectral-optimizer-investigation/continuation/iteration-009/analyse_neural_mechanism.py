#!/usr/bin/env python3
"""Audit and summarize completed or partial I9 neural-mechanism artifacts.

This is a CPU-only postprocessor.  It does not import the acquisition module,
load MNIST, restore a model, or execute an optimizer step.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any

import numpy as np
import torch


SEEDS = (100, 101, 102)
SOURCES = ("raw", "current32")
ANCHORS = (100, 500, 1500, 2000)
UTILITY_NAMES = ("train_clean", "train_soft", "train_noisy", "aux_clean", "aux_soft")
COMPONENT_NAMES = (*UTILITY_NAMES, "soft_minus_clean", "fixed_minus_soft")
INTERVENTION_NAMES = (
    "raw_gradient", "native_gradient", "raw_gradient_to_native_norm",
    "native_gradient_to_raw_norm", "raw_data_delta_to_native_norm",
    "native_data_delta_to_raw_norm",
)
PROBE_RE = re.compile(r"^probe-s(\d+)-(raw|current32)-t(\d+)\.json$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle, parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON constant {value} in {path.name}")))


def digest_node(value: Any) -> Any:
    """Mirror neural_core.tree_digest without importing executable run code."""
    if type(value) is torch.Tensor:
        tensor = value.detach().cpu().contiguous()
        raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
        return ["tensor", str(tensor.dtype), list(tensor.shape), hashlib.sha256(raw).hexdigest()]
    if type(value) is dict:
        return ["dict", [[digest_node(k), digest_node(v)] for k, v in value.items()]]
    if type(value) is list:
        return ["list", [digest_node(v) for v in value]]
    if type(value) is tuple:
        return ["tuple", [digest_node(v) for v in value]]
    if value is None or type(value) in (bool, int, str):
        return [type(value).__name__, value]
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("non-finite float in tree digest")
        return ["float", value.hex()]
    raise TypeError(f"unsupported tree-digest type {type(value)!r}")


def tree_digest(value: Any) -> str:
    raw = json.dumps(digest_node(value), ensure_ascii=True, allow_nan=False,
                     separators=(",", ":")).encode("ascii")
    return hashlib.sha256(b"i9_neural_tree_v1\n" + raw).hexdigest()


def finite_float(value: Any, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} is non-finite")
    return result


def dot(left: torch.Tensor, right: torch.Tensor) -> float:
    return finite_float(torch.dot(left.to(torch.float64), right.to(torch.float64)).item(), "dot")


def norm(value: torch.Tensor) -> float:
    return finite_float(torch.linalg.vector_norm(value.to(torch.float64)).item(), "norm")


def basis_from_tracker(tracker: dict[str, Any]) -> torch.Tensor | None:
    value = tracker.get("V")
    if value is None:
        return None
    if type(value) is not torch.Tensor or value.ndim != 2:
        raise ValueError("anchor tracker V is not a matrix")
    proj_k = tracker.get("proj_k")
    k = value.shape[1] if proj_k is None else min(int(proj_k), value.shape[1])
    return value[:, :k].detach().cpu()


def projection_geometry(value: torch.Tensor, basis: torch.Tensor | None) -> dict[str, Any]:
    squared = dot(value, value)
    if basis is None:
        return {"squared_norm": squared, "projected_squared_norm": None,
                "fraction": None, "reason": "basis_unavailable"}
    projected = basis @ (basis.T @ value)
    projected_squared = dot(projected, projected)
    return {"squared_norm": squared, "projected_squared_norm": projected_squared,
            "fraction": None if squared == 0 else projected_squared / squared,
            "reason": "zero_vector" if squared == 0 else None}


def leakage(value: torch.Tensor, basis: torch.Tensor | None) -> dict[str, Any]:
    squared = dot(value, value)
    if basis is None:
        return {"fraction": None, "reason": "basis_unavailable"}
    if squared == 0:
        return {"fraction": None, "reason": "zero_displacement"}
    projected = basis @ (basis.T @ value)
    return {"fraction": dot(value - projected, value - projected) / squared, "reason": None}


class Audit:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.checks = 0
        self.max_abs = {"pair_identity": 0.0, "norm": 0.0, "utility_dot": 0.0,
                        "gram": 0.0, "projection": 0.0, "leakage": 0.0}
        self.max_rel = dict(self.max_abs)

    def require(self, condition: bool, message: str) -> None:
        self.checks += 1
        if not condition:
            self.errors.append(message)

    def compare(self, actual: Any, expected: Any, kind: str, label: str,
                rtol: float = 5e-7, atol: float = 1e-9) -> None:
        self.checks += 1
        if actual is None or expected is None:
            if actual is not expected:
                self.errors.append(f"{label}: null mismatch ({actual!r} versus {expected!r})")
            return
        a, e = finite_float(actual, label), finite_float(expected, label)
        absolute = abs(a - e)
        relative = absolute / max(abs(a), abs(e), 1e-300)
        self.max_abs[kind] = max(self.max_abs[kind], absolute)
        self.max_rel[kind] = max(self.max_rel[kind], relative)
        if not math.isclose(a, e, rel_tol=rtol, abs_tol=atol):
            self.errors.append(f"{label}: {a:.17g} != recomputed {e:.17g} "
                               f"(abs {absolute:.3g}, rel {relative:.3g})")


def safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator == 0:
        return None
    return numerator / denominator


def verify_file_record(root: Path, record: dict[str, Any], audit: Audit, label: str) -> Path | None:
    try:
        name = record["name"]
        if type(name) is not str or Path(name).name != name:
            raise ValueError("artifact name is not a basename")
        path = root / name
        audit.require(path.is_file(), f"{label}: missing {name}")
        if not path.is_file():
            return None
        audit.require(path.stat().st_size == int(record["bytes"]),
                      f"{label}: byte count differs for {name}")
        audit.require(sha256(path) == record["sha256"], f"{label}: SHA256 differs for {name}")
        return path
    except Exception as exc:
        audit.errors.append(f"{label}: invalid file record: {exc}")
        return None


def aggregate_pairs(rows: list[dict[str, Any]], audit: Audit, label: str) -> dict[str, Any]:
    if len(rows) != 32:
        audit.errors.append(f"{label}: expected 32 pair rows, found {len(rows)}")
    result: dict[str, Any] = {"pair_count": len(rows)}
    for space, key in (("full", None), ("projected", "after_projection")):
        selected = [row if key is None else row.get(key) for row in rows]
        if any(row is None for row in selected):
            result[space] = None
            continue
        values: dict[str, list[float]] = {name: [] for name in ("fresh", "surprise", "innovation")}
        for index, row in enumerate(selected):
            for name in values:
                value = row.get(name)
                if value is not None:
                    values[name].append(finite_float(value, f"{label} pair {index} {space} {name}"))
            if row.get("surprise") is not None and row.get("innovation") is not None:
                audit.compare(row["innovation"], row["fresh"] + row["surprise"],
                              "pair_identity", f"{label} pair {index} {space} identity",
                              rtol=2e-6, atol=2e-7)
        means = {name: (float(np.mean(items)) if items else None) for name, items in values.items()}
        surprise = values["surprise"]
        result[space] = {
            "means": means,
            "fresh_over_innovation": safe_ratio(means["fresh"], means["innovation"]),
            "surprise_over_innovation": safe_ratio(means["surprise"], means["innovation"]),
            "surprise_negative_count": sum(item < 0 for item in surprise),
            "surprise_min": min(surprise) if surprise else None,
            "surprise_max": max(surprise) if surprise else None,
        }
    full, projected = result.get("full"), result.get("projected")
    result["retained_over_full"] = None if projected is None or full is None else {
        "fresh": safe_ratio(projected["means"]["fresh"], full["means"]["fresh"]),
        "surprise": safe_ratio(projected["means"]["surprise"], full["means"]["surprise"]),
        "innovation": safe_ratio(projected["means"]["innovation"], full["means"]["innovation"]),
    }
    result["raw_signed_pair_values_location"] = label
    return result


def compare_geometry(saved: dict[str, Any], computed: dict[str, Any], audit: Audit,
                     label: str, kind: str = "projection") -> None:
    audit.require(saved.get("reason") == computed.get("reason"), f"{label}: reason differs")
    for key in ("squared_norm", "projected_squared_norm", "fraction"):
        if key in computed:
            audit.compare(saved.get(key), computed.get(key), kind, f"{label} {key}",
                          rtol=5e-5 if key != "squared_norm" else 5e-7,
                          atol=5e-6 if key == "fraction" else 1e-8)


def analyse_probe(root: Path, probe_path: Path, audit: Audit) -> dict[str, Any] | None:
    match = PROBE_RE.match(probe_path.name)
    if match is None:
        audit.errors.append(f"unrecognized probe filename {probe_path.name}")
        return None
    seed, source, step = int(match.group(1)), match.group(2), int(match.group(3))
    label = probe_path.name
    try:
        wrapper = read_json(probe_path)
        measurement = wrapper["measurement"]
        audit.require((wrapper["seed"], wrapper["source"], wrapper["completed_step"])
                      == (seed, source, step), f"{label}: filename metadata mismatch")
        audit.require(measurement.get("schema") == "i9_neural_anchor_probe_v1" and
                      measurement.get("status") == "complete", f"{label}: bad probe schema/status")
        anchor_path = verify_file_record(root, wrapper["anchor_artifact"], audit, label + " anchor")
        tensor_path = verify_file_record(root, wrapper["tensor_artifact"], audit, label + " tensors")
        if anchor_path is None or tensor_path is None:
            return None
        anchor_binding = torch.load(anchor_path, map_location="cpu", weights_only=True)
        tensors = torch.load(tensor_path, map_location="cpu", weights_only=True)
        anchor = anchor_binding["state"]
        audit.require((anchor_binding["seed"], anchor_binding["source"],
                       anchor_binding["completed_step"]) == (seed, source, step),
                      f"{label}: anchor metadata mismatch")
        state_hash = tree_digest(anchor)
        for name, expected in (("binding", anchor_binding["complete_state_digest"]),
                               ("wrapper", wrapper["complete_state_digest"]),
                               ("measurement", measurement["anchor_snapshot_sha256"])):
            audit.require(state_hash == expected, f"{label}: {name} complete-state digest differs")
        audit.require(tree_digest(tensors) == measurement["tensor_payload_sha256"],
                      f"{label}: tensor payload tree digest differs")

        old_basis = basis_from_tracker(anchor["tracker"])
        current_basis = tensors["current_V"]
        if current_basis is not None:
            current_basis = current_basis.detach().cpu()
        utilities = {name: value.detach().cpu()
                     for name, value in tensors["mean_utility_gradients"].items()}
        audit.require(tuple(utilities) == UTILITY_NAMES, f"{label}: utility tensor keys differ")
        components = dict(utilities)
        components["soft_minus_clean"] = utilities["train_soft"] - utilities["train_clean"]
        components["fixed_minus_soft"] = utilities["train_noisy"] - utilities["train_soft"]

        designated = measurement["designated_update"]
        audit.compare(designated["raw_gradient_norm"], norm(tensors["g0"]), "norm",
                      f"{label} designated raw-gradient norm")
        audit.compare(designated["native_gradient_norm"], norm(tensors["pg"]), "norm",
                      f"{label} designated native-gradient norm")
        audit.require(designated["previous_basis_rank"] ==
                      (0 if old_basis is None else old_basis.shape[1]),
                      f"{label}: previous basis rank differs")
        audit.require(designated["current_basis_rank"] ==
                      (0 if current_basis is None else current_basis.shape[1]),
                      f"{label}: current basis rank differs")

        gram_saved = {(row["left"], row["right"]): row for row in measurement["signed_utility_gram"]}
        gram_summary = []
        for left_index, left in enumerate(COMPONENT_NAMES):
            for right in COMPONENT_NAMES[left_index:]:
                value = dot(components[left], components[right])
                saved = gram_saved.get((left, right))
                audit.require(saved is not None, f"{label}: missing Gram {left}/{right}")
                if saved is not None:
                    audit.compare(saved["value"], value, "gram", f"{label} Gram {left}/{right}")
                    audit.require(saved["sign"] == (1 if value > 0 else (-1 if value < 0 else 0)),
                                  f"{label}: Gram sign differs for {left}/{right}")
                gram_summary.append({"left": left, "right": right, "value": value})
        audit.require(len(gram_saved) == 28, f"{label}: Gram must contain 28 upper-triangle cells")

        retention_summary = {}
        for name in COMPONENT_NAMES:
            retention_summary[name] = {}
            for basis_name, basis in (("old_basis", old_basis), ("current_basis", current_basis)):
                computed = projection_geometry(components[name], basis)
                saved = measurement["utility_retention"][name][basis_name]
                compare_geometry(saved, computed, audit, f"{label} {name} {basis_name}")
                retention_summary[name][basis_name] = computed

        intervention_saved = {row["name"]: row for row in measurement["interventions"]}
        intervention_summary = {}
        for name in INTERVENTION_NAMES:
            saved = intervention_saved[name]
            payload = tensors["interventions"][name]
            total, data = payload["total_delta"], payload["data_delta"]
            if total is None or data is None:
                audit.require(saved["status"] == "undefined", f"{label} {name}: undefined mismatch")
                intervention_summary[name] = saved
                continue
            total, data = total.detach().cpu(), data.detach().cpu()
            audit.require(saved["status"] == "defined", f"{label} {name}: defined mismatch")
            audit.compare(saved["total_norm"], norm(total), "norm", f"{label} {name} total norm")
            audit.compare(saved["data_norm"], norm(data), "norm", f"{label} {name} data norm")
            predicted = {}
            for utility in UTILITY_NAMES:
                value = -dot(utilities[utility], data)
                row = saved["negative_gradient_dot_data_delta"][utility]
                audit.compare(row["value"], value, "utility_dot",
                              f"{label} {name} {utility} utility")
                audit.require(row["sign"] == (1 if value > 0 else (-1 if value < 0 else 0)),
                              f"{label} {name} {utility}: utility sign differs")
                predicted[utility] = value
            total_leak, data_leak = {}, {}
            for basis_name, basis in (("old_basis", old_basis), ("current_basis", current_basis)):
                for vector_name, vector, destination, saved_group in (
                        ("total", total, total_leak, saved["total_leakage"]),
                        ("data", data, data_leak, saved["data_leakage"])):
                    computed = leakage(vector, basis)
                    audit.require(saved_group[basis_name]["reason"] == computed["reason"],
                                  f"{label} {name} {vector_name} {basis_name}: leakage reason differs")
                    audit.compare(saved_group[basis_name]["fraction"], computed["fraction"],
                                  "leakage", f"{label} {name} {vector_name} {basis_name} leakage",
                                  rtol=5e-5, atol=5e-6)
                    destination[basis_name] = computed
            intervention_summary[name] = {
                "status": "defined", "reason": None, "gradient_scale": saved["gradient_scale"],
                "total_norm": norm(total), "data_norm": norm(data),
                "loss_change": saved["loss_change"],
                "negative_gradient_dot_data_delta": predicted,
                "total_leakage": total_leak, "data_leakage": data_leak,
            }
        audit.require(tuple(intervention_saved) == INTERVENTION_NAMES,
                      f"{label}: intervention keys/order differ")

        pair_summary = aggregate_pairs(measurement["pair_rows"], audit,
                                       f"{probe_path.name}:measurement.pair_rows")
        return {"seed": seed, "source": source, "anchor": step,
                "probe_artifact": probe_path.name, "pair_geometry": pair_summary,
                "designated_update": designated,
                "utility_components": retention_summary, "signed_utility_gram": gram_summary,
                "interventions": intervention_summary}
    except Exception as exc:
        audit.errors.append(f"{label}: analysis failed: {type(exc).__name__}: {exc}")
        return None


def mean_or_none(values: list[float | None]) -> float | None:
    finite = [value for value in values if value is not None]
    return None if not finite else float(np.mean(finite))


def primary_summary(probes: list[dict[str, Any]]) -> dict[str, Any]:
    contrast_specs = (
        ("native_direction_at_raw_data_norm_minus_raw",
         "native_data_delta_to_raw_norm", "raw_gradient"),
        ("native_actual_minus_raw_direction_at_native_data_norm",
         "native_gradient", "raw_data_delta_to_native_norm"),
    )
    seed_rows = []
    for seed in SEEDS:
        rows = {(row["anchor"]): row for row in probes
                if row["seed"] == seed and row["source"] == "current32"
                and row["anchor"] in (1500, 2000)}
        result: dict[str, Any] = {"seed": seed, "anchors_required": [1500, 2000],
                                  "anchors_present": sorted(rows), "contrasts": {}}
        for title, native_name, raw_name in contrast_specs:
            anchor_values = {}
            for anchor in (1500, 2000):
                row = rows.get(anchor)
                value = None
                if row is not None:
                    native = row["interventions"][native_name]
                    raw = row["interventions"][raw_name]
                    if native["status"] == raw["status"] == "defined":
                        value = (native["loss_change"]["aux_clean"]
                                 - raw["loss_change"]["aux_clean"])
                anchor_values[str(anchor)] = value
            complete = all(value is not None for value in anchor_values.values())
            result["contrasts"][title] = {
                "anchor_values": anchor_values,
                "seed_first_late_mean": (float(np.mean(list(anchor_values.values())))
                                          if complete else None),
                "status": "complete" if complete else "incomplete",
            }
        seed_rows.append(result)
    names = [spec[0] for spec in contrast_specs]
    means = {}
    for name in names:
        values = [row["contrasts"][name]["seed_first_late_mean"] for row in seed_rows]
        means[name] = {"all_three_seed_values": values,
                       "mean": float(np.mean(values)) if all(v is not None for v in values) else None,
                       "status": "complete" if all(v is not None for v in values) else "incomplete"}
    return {"definition": "negative favors native direction; finite auxiliary-clean loss change",
            "source": "current32", "late_anchors": [1500, 2000],
            "per_seed": seed_rows, "across_seeds": means}


def aggregate_covariance(probes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for source in SOURCES:
        for anchor in ANCHORS:
            rows = [row for row in probes if row["source"] == source and row["anchor"] == anchor]
            metrics: dict[str, list[float | None]] = {
                "full_fresh_mean": [], "full_surprise_mean": [], "full_innovation_mean": [],
                "projected_fresh_mean": [], "projected_surprise_mean": [],
                "projected_innovation_mean": [],
                "full_fresh_over_innovation": [], "full_surprise_over_innovation": [],
                "projected_fresh_over_innovation": [], "projected_surprise_over_innovation": [],
                "retained_fresh": [], "retained_surprise": [], "retained_innovation": [],
            }
            per_seed = []
            for row in sorted(rows, key=lambda item: item["seed"]):
                geometry = row["pair_geometry"]
                full, projected = geometry["full"], geometry["projected"]
                retained = geometry["retained_over_full"]
                values = {
                    "full_fresh_mean": None if full is None else full["means"]["fresh"],
                    "full_surprise_mean": None if full is None else full["means"]["surprise"],
                    "full_innovation_mean": None if full is None else full["means"]["innovation"],
                    "projected_fresh_mean": None if projected is None else projected["means"]["fresh"],
                    "projected_surprise_mean": None if projected is None else projected["means"]["surprise"],
                    "projected_innovation_mean": None if projected is None else projected["means"]["innovation"],
                    "full_fresh_over_innovation": None if full is None else full["fresh_over_innovation"],
                    "full_surprise_over_innovation": None if full is None else full["surprise_over_innovation"],
                    "projected_fresh_over_innovation": None if projected is None else projected["fresh_over_innovation"],
                    "projected_surprise_over_innovation": None if projected is None else projected["surprise_over_innovation"],
                    "retained_fresh": None if retained is None else retained["fresh"],
                    "retained_surprise": None if retained is None else retained["surprise"],
                    "retained_innovation": None if retained is None else retained["innovation"],
                }
                per_seed.append({"seed": row["seed"], **values})
                for name, value in values.items():
                    metrics[name].append(value)
            result.append({"source": source, "anchor": anchor,
                           "seeds_present": [row["seed"] for row in per_seed],
                           "per_seed": per_seed,
                           "seed_first_means": {name: mean_or_none(values)
                                                for name, values in metrics.items()}})
    return result


def load_curves(root: Path, completion_records: dict[str, dict[str, Any]],
                audit: Audit) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    curves, summaries = [], []
    for seed in SEEDS:
        for source in SOURCES:
            path = root / f"curve-s{seed}-{source}.json"
            if not path.is_file():
                continue
            if path.name in completion_records:
                verify_file_record(root, completion_records[path.name], audit, "completion curve")
            record = read_json(path)
            audit.require((record["seed"], record["source"]) == (seed, source),
                          f"{path.name}: curve metadata mismatch")
            rows = record["curve"]
            steps = [int(row["step"]) for row in rows]
            audit.require(steps == sorted(set(steps)), f"{path.name}: curve steps not unique/increasing")
            for row in rows:
                for key, value in row.items():
                    if key != "step":
                        finite_float(value, f"{path.name} {key}")
            curves.append(record)
            if rows:
                best = min(rows, key=lambda row: row["validation_ce"])
                summaries.append({"seed": seed, "source": source,
                                  "best_validation": dict(best), "endpoint": dict(rows[-1])})
    combined_path = root / "all-curves.json"
    if combined_path.is_file():
        combined = read_json(combined_path)
        audit.require(combined == curves, "all-curves.json differs from ordered individual curves")
    return curves, summaries


def write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.artifacts.resolve(strict=True)
    if not root.is_dir():
        raise SystemExit("--artifacts must be a directory")
    args.output.parent.resolve(strict=True)
    args.output.mkdir(exist_ok=False)
    torch.set_num_threads(1)
    audit = Audit()

    manifest = read_json(root / "manifest.json") if (root / "manifest.json").is_file() else None
    audit.require(manifest is not None, "manifest.json missing")
    completion_records: dict[str, dict[str, Any]] = {}
    completion = read_json(root / "completion.json") if (root / "completion.json").is_file() else None
    failure = read_json(root / "failure.json") if (root / "failure.json").is_file() else None
    audit.require(not (completion is not None and failure is not None),
                  "both completion.json and failure.json exist")
    if completion is not None:
        audit.require(completion.get("status") == "complete", "completion status is not complete")
        records = completion.get("artifacts", [])
        names = [record.get("name") for record in records]
        audit.require(len(names) == len(set(names)), "completion artifact names are duplicated")
        completion_records = {record["name"]: record for record in records}
        for record in records:
            verify_file_record(root, record, audit, "completion")
    else:
        audit.warnings.append("partial artifact directory: completion.json absent")
        if failure is not None:
            audit.warnings.append(f"acquisition failure retained: {failure.get('type')}: "
                                  f"{failure.get('message')}")

    probe_paths = sorted(root.glob("probe-s*-t*.json"))
    probes = [row for path in probe_paths if (row := analyse_probe(root, path, audit)) is not None]
    keys = [(row["seed"], row["source"], row["anchor"]) for row in probes]
    audit.require(len(keys) == len(set(keys)), "duplicate probe seed/source/anchor keys")
    expected = {(seed, source, anchor) for seed in SEEDS for source in SOURCES for anchor in ANCHORS}
    missing = sorted(expected - set(keys))
    unexpected = sorted(set(keys) - expected)
    if completion is not None:
        audit.require(not missing, f"complete run missing probe cells: {missing}")
    elif missing:
        audit.warnings.append(f"partial run missing {len(missing)} of 24 probe cells")
    audit.require(not unexpected, f"unexpected probe cells: {unexpected}")

    curves, curve_summaries = load_curves(root, completion_records, audit)
    if completion is not None:
        audit.require(len(curves) == 6, f"complete run has {len(curves)} rather than 6 curves")

    primary = primary_summary(probes)
    summary = {
        "schema": "i9_neural_analysis_summary_v1",
        "scientific_scope": [
            "Conditional frozen-state and one-step measurements; not unique mediation of long-run effects.",
            "Batch-pair fresh terms condition on the fixed corrupted finite dataset, not fresh label redraws.",
            "Negative primary contrasts favor native direction; mixed and negative outcomes are retained.",
        ],
        "artifact_directory": str(root),
        "acquisition_status": "complete" if completion is not None else "partial",
        "coverage": {"expected_probe_cells": 24, "analysed_probe_cells": len(probes),
                     "missing_probe_cells": missing, "unexpected_probe_cells": unexpected,
                     "curve_trajectories": len(curves)},
        "primary_aux_clean_contrasts": primary,
        "covariance_seed_first": aggregate_covariance(probes),
        "anchors": probes,
        "curves": curves,
        "curve_best_and_endpoint": curve_summaries,
    }
    audit_payload = {
        "schema": "i9_neural_analysis_audit_v1",
        "status": "pass" if not audit.errors else "fail",
        "checks": audit.checks,
        "errors": audit.errors,
        "warnings": audit.warnings,
        "maximum_absolute_discrepancy": audit.max_abs,
        "maximum_relative_discrepancy": audit.max_rel,
        "hash_scope": ("All completion-manifest artifacts plus every probe-referenced anchor and "
                       "tensor payload; tree digests independently recomputed for analysed probes."),
        "pair_audit_scope": ("Saved scalar identity innovation=fresh+surprise checked. Individual "
                             "pair gradients were not saved, so pair geometry cannot be tensor-recomputed."),
    }
    write_json(args.output / "summary.json", summary)
    write_json(args.output / "audit.json", audit_payload)
    print(json.dumps({"status": audit_payload["status"], "analysed_probes": len(probes),
                      "missing_probes": len(missing), "output": str(args.output)},
                     allow_nan=False))
    return 0 if not audit.errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
