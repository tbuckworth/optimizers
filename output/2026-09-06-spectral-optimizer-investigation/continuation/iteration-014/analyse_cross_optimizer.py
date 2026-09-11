#!/usr/bin/env python3
"""Independent CPU-only integrity audit and scalar analysis for I14."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess
from typing import Any

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
BASES = ("sgd", "sgdm", "adamw")
POLICIES = ("raw", "current32")
TARGETS = ("clean", "fixed")
CALIBRATION_SEEDS = (190, 191)
CONFIRMATION_SEEDS = (200, 201, 202)
RATES = {"sgd": (0.03, 0.1, 0.3), "sgdm": (0.003, 0.01, 0.03),
         "adamw": (0.0003, 0.001, 0.003)}
HORIZONS = (0, 100, 250, 500, 1000, 1500, 2000)
COMPETENCE_FLOOR = 0.85
DATA_ROOT = Path("data/MNIST/raw")
PLAN_KEYS = ("schema", "phase", "seed", "initialization_seed", "train_indices",
             "validation_indices", "auxiliary_indices", "replacement_mask",
             "replacement_digits", "training_batches")
_GLOBAL_ORDER = np.random.default_rng(
    np.random.SeedSequence([20260907, 14, 0, 0])).permutation(60000)


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle, parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"nonfinite JSON constant {value} in {path.name}")))


def digest_node(value: Any) -> Any:
    """Independent implementation of the unchanged I9 tree-digest format."""
    if type(value) is torch.Tensor:
        tensor = value.detach().cpu().contiguous()
        raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
        return ["tensor", str(tensor.dtype), list(tensor.shape), hashlib.sha256(raw).hexdigest()]
    if type(value) is dict:
        return ["dict", [[digest_node(key), digest_node(item)] for key, item in value.items()]]
    if type(value) is list:
        return ["list", [digest_node(item) for item in value]]
    if type(value) is tuple:
        return ["tuple", [digest_node(item) for item in value]]
    if value is None or type(value) in (bool, int, str):
        return [type(value).__name__, value]
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("nonfinite float in complete state")
        return ["float", value.hex()]
    raise TypeError(f"unsupported complete-state value {type(value)!r}")


def tree_digest(value: Any) -> str:
    payload = json.dumps(digest_node(value), ensure_ascii=True, allow_nan=False,
                         separators=(",", ":")).encode("ascii")
    return hashlib.sha256(b"i9_neural_tree_v1\n" + payload).hexdigest()


def failure_fingerprint(value: Any) -> str:
    """Independent implementation of the acquisition's nonfinite-safe digest."""
    def encode(item: Any) -> Any:
        if type(item) is torch.Tensor:
            tensor = item.detach().cpu().contiguous()
            raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
            return ["tensor", str(tensor.dtype), list(tensor.shape),
                    hashlib.sha256(raw).hexdigest()]
        if type(item) is dict:
            return ["dict", [[encode(key), encode(value)] for key, value in item.items()]]
        if type(item) in (list, tuple):
            return [type(item).__name__, [encode(value) for value in item]]
        if type(item) is float:
            return ["float", item.hex()]
        if item is None or type(item) in (bool, int, str):
            return [type(item).__name__, item]
        raise ValueError(f"unsupported startup-failure evidence {type(item)!r}")

    payload = json.dumps(encode(value), separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def finite(value: Any, label: str) -> float:
    if type(value) not in (int, float):
        raise ValueError(label + " is not a JSON scalar")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(label + " is nonfinite")
    return result


class Audit:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.checks = 0
        self.hash_files = 0
        self.hash_bytes = 0
        self.tree_digests = 0
        self.max_residual_error = 0.0
        self.max_leakage_error = 0.0

    def require(self, condition: bool, message: str) -> None:
        self.checks += 1
        if not condition:
            self.errors.append(message)


def verify_record(directory: Path, record: dict[str, Any], audit: Audit,
                  label: str) -> Path | None:
    try:
        name = record["name"]
        if type(name) is not str or Path(name).name != name:
            raise ValueError("artifact name is not a basename")
        path = directory / name
        audit.require(path.is_file(), f"{label}: missing {name}")
        if not path.is_file():
            return None
        size = path.stat().st_size
        audit.require(type(record.get("bytes")) is int and record["bytes"] == size,
                      f"{label}: byte count differs for {name}")
        audit.require(type(record.get("sha256")) is str
                      and record["sha256"] == sha256(path),
                      f"{label}: SHA256 differs for {name}")
        audit.hash_files += 1
        audit.hash_bytes += size
        return path
    except Exception as exc:
        audit.errors.append(f"{label}: malformed artifact record: {exc}")
        return None


def require_member(index: dict[str, dict[str, Any]], name: str,
                   audit: Audit, label: str) -> None:
    audit.require(name in index, f"{label}: {name} is absent from completion artifacts")


def corruption_counts(plan: dict[str, Any], labels: np.ndarray) -> dict[str, int]:
    indices = np.asarray(plan["train_indices"], dtype=np.int64)
    clean = labels[indices]
    mask = np.asarray(plan["replacement_mask"], dtype=np.bool_)
    digits = np.asarray(plan["replacement_digits"], dtype=np.uint8)
    noisy = np.where(mask, digits, clean)
    return {"replaced_count": int(mask.sum()),
            "incorrect_count": int((noisy != clean).sum()), "train_count": 5000}


def verify_sources(manifests: list[dict[str, Any]], audit: Audit) -> dict[str, Any]:
    first = manifests[0]
    commit, sources = first.get("frozen_commit"), first.get("source_hashes")
    audit.require(type(commit) is str and len(commit) == 40, "frozen commit is malformed")
    audit.require(type(sources) is dict and bool(sources), "source hash manifest is missing")
    for manifest in manifests[1:]:
        audit.require(manifest.get("frozen_commit") == commit
                      and manifest.get("source_hashes") == sources,
                      "phase frozen-source manifests differ")
    verified = []
    if type(commit) is str and type(sources) is dict:
        for relative, expected in sources.items():
            try:
                path = ROOT / relative
                audit.require(path.is_file() and sha256(path) == expected,
                              f"worktree source differs: {relative}")
                frozen = subprocess.check_output(
                    ["git", "show", commit + ":" + relative], cwd=ROOT)
                audit.require(hashlib.sha256(frozen).hexdigest() == expected,
                              f"commit source differs: {relative}")
                verified.append(relative)
            except Exception as exc:
                audit.errors.append(f"source verification failed for {relative}: {exc}")
    return {"frozen_commit": commit, "source_files_verified": verified}


def analysis_source_provenance(audit: Audit) -> dict[str, Any]:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                     text=True).strip()
    hashes = {}
    for path in (Path(__file__).resolve(), HERE / "test_analyse_cross_optimizer.py"):
        relative = str(path.relative_to(ROOT))
        current = sha256(path)
        frozen = subprocess.check_output(["git", "show", commit + ":" + relative], cwd=ROOT)
        audit.require(hashlib.sha256(frozen).hexdigest() == current,
                      f"analysis source is not frozen at HEAD: {relative}")
        hashes[relative] = current
    return {"git_commit": commit, "source_hashes": hashes}


def validate_plan(value: Any, phase: str, seed: int, audit: Audit,
                  label: str) -> dict[str, Any]:
    audit.require(type(value) is dict and tuple(value) == PLAN_KEYS,
                  f"{label}: plan topology differs")
    audit.require(value.get("schema") == "i14_cross_optimizer_data_plan_v1"
                  and value.get("phase") == phase and value.get("seed") == seed,
                  f"{label}: plan identity differs")
    sizes = {"train_indices": 5000, "validation_indices": 5000,
             "auxiliary_indices": 5000, "replacement_mask": 5000,
             "replacement_digits": 5000, "training_batches": 2000}
    for name, size in sizes.items():
        audit.require(type(value.get(name)) is list and len(value[name]) == size,
                      f"{label}: {name} shape differs")
    splits = [item for name in ("train_indices", "validation_indices", "auxiliary_indices")
              for item in value.get(name, [])]
    audit.require(len(splits) == 15000 and len(set(splits)) == 15000
                  and all(type(item) is int and 0 <= item < 60000 for item in splits),
                  f"{label}: split uniqueness/range differs")
    mask = value.get("replacement_mask", [])
    digits = value.get("replacement_digits", [])
    batches = value.get("training_batches", [])
    audit.require(all(type(item) is bool for item in mask), f"{label}: mask types differ")
    audit.require(all(type(item) is int and 0 <= item < 10 for item in digits),
                  f"{label}: replacement digit differs")
    audit.require(all(type(row) is list and len(row) == 64
                      and all(type(item) is int and 0 <= item < 5000 for item in row)
                      for row in batches), f"{label}: batch plan differs")
    pool = _GLOBAL_ORDER[:15000] if phase == "calibration" else _GLOBAL_ORDER[15000:]
    order = np.random.default_rng(
        np.random.SeedSequence([20260907, 14, seed, 0])).permutation(pool)
    expected = {
        "initialization_seed": int(np.random.default_rng(
            np.random.SeedSequence([20260907, 14, seed, 1])).integers(
                0, 2**31, dtype=np.int64)),
        "train_indices": order[:5000], "validation_indices": order[5000:10000],
        "auxiliary_indices": order[10000:15000],
        "replacement_mask": np.random.default_rng(
            np.random.SeedSequence([20260907, 14, seed, 2])).random(5000) < 0.9,
        "replacement_digits": np.random.default_rng(
            np.random.SeedSequence([20260907, 14, seed, 3])).integers(
                0, 10, 5000, dtype=np.int64),
        "training_batches": np.random.default_rng(
            np.random.SeedSequence([20260907, 14, seed, 4])).integers(
                0, 5000, (2000, 64), dtype=np.int64),
    }
    audit.require(value.get("initialization_seed") == expected["initialization_seed"],
                  f"{label}: initialization seed differs from deterministic plan")
    for name in expected:
        if name != "initialization_seed":
            audit.require(np.array_equal(np.asarray(value.get(name)), expected[name]),
                          f"{label}: {name} differs from deterministic plan")
    return {"phase": phase, "seed": seed, "indices": set(splits),
            "initialization_seed": value.get("initialization_seed")}


def validate_evaluation(row: dict[str, Any], calibration: bool, audit: Audit,
                        label: str) -> None:
    expected_groups = ("train", "validation") if calibration else (
        "train", "validation", "auxiliary")
    audit.require(tuple(key for key in row if key != "horizon") == expected_groups,
                  f"{label}: evaluation groups/order differ")
    expected = {
        "train": ("clean_ce", "soft_ce", "clean_accuracy", "mean_max_probability",
                  "mean_true_label_probability", "fixed_ce", "fixed_accuracy",
                  "fixed_minus_soft_ce"),
        "validation": ("clean_ce", "soft_ce", "clean_accuracy",
                       "mean_max_probability", "mean_true_label_probability"),
        "auxiliary": ("clean_ce", "soft_ce", "clean_accuracy",
                      "mean_max_probability", "mean_true_label_probability"),
    }
    for group in expected_groups:
        values = row.get(group, {})
        audit.require(tuple(values) == expected[group], f"{label}: {group} keys/order differ")
        for key in expected[group]:
            number = finite(values[key], f"{label} {group}/{key}")
            if "accuracy" in key or "probability" in key:
                audit.require(0 <= number <= 1, f"{label}: {group}/{key} outside [0,1]")
            elif key != "fixed_minus_soft_ce":
                audit.require(number >= 0, f"{label}: {group}/{key} is negative")
    residual = abs(row["train"]["fixed_minus_soft_ce"]
                   - (row["train"]["fixed_ce"] - row["train"]["soft_ce"]))
    audit.max_residual_error = max(audit.max_residual_error, residual)
    audit.require(residual <= 2e-12, f"{label}: fixed-soft residual identity differs")


def selection(curve: list[dict[str, Any]]) -> dict[str, int]:
    choices = [row for row in curve if row["horizon"] > 0]
    if not choices:
        raise ValueError("no nonzero selection horizon")
    by_ce = min(choices, key=lambda row: (row["validation"]["clean_ce"], row["horizon"]))
    by_accuracy = min(choices, key=lambda row: (-row["validation"]["clean_accuracy"],
                                                row["horizon"]))
    return {"minimum_validation_ce": by_ce["horizon"],
            "maximum_validation_accuracy": by_accuracy["horizon"]}


def choose_rates(rows: dict[tuple[int, str, float], dict[str, Any]]) -> dict[str, Any]:
    candidates, choices = [], {}
    for base in BASES:
        eligible: list[tuple[float, float]] = []
        for lr in RATES[base]:
            pair = [rows.get((seed, base, lr)) for seed in CALIBRATION_SEEDS]
            finals = [None if row is None else row["curve_by_horizon"].get(2000)
                      for row in pair]
            validation_rows = [None if final is None else final["validation"]
                               for final in finals]
            complete = all(row is not None and row["status"] == "complete"
                           and final is not None for row, final in zip(pair, finals))
            competent = complete and all(final["validation"]["clean_accuracy"]
                                         >= COMPETENCE_FLOOR for final in finals)
            mean_ce = (sum(final["validation"]["clean_ce"] for final in finals) / 2
                       if complete else None)
            candidates.append({"base": base, "lr": lr, "finite_complete": complete,
                "eligible": bool(competent),
                "validation_by_seed": validation_rows,
                "mean_final_validation_ce": mean_ce,
                "competence_floor_each_seed": COMPETENCE_FLOOR})
            if competent:
                eligible.append((mean_ce, lr))
        choices[base] = min(eligible)[1] if eligible else None
    return {"schema": "i14_calibration_selection_v1", "selected_rates": choices,
            "ready_for_confirmation": all(value is not None for value in choices.values()),
            "candidates": candidates, "confirmation_outcomes_used": False}


def leakage_energy(value: dict[str, Any], expected_squared: float, audit: Audit,
                   label: str) -> tuple[float, float] | None:
    squared = finite(value.get("squared_norm"), label + " squared norm")
    audit.require(math.isclose(squared, expected_squared, rel_tol=2e-6, abs_tol=1e-14),
                  f"{label}: step norm/energy identity differs")
    reason, fraction = value.get("reason"), value.get("fraction")
    if reason is not None:
        audit.require(fraction is None and reason in ("basis_unavailable", "zero_displacement"),
                      f"{label}: invalid missing leakage")
        if reason == "zero_displacement":
            audit.require(squared == 0 and value.get("outside_squared_norm") == 0,
                          f"{label}: zero leakage identity differs")
        return None
    outside = finite(value.get("outside_squared_norm"), label + " outside energy")
    fraction_value = finite(fraction, label + " fraction")
    error = abs(outside - squared * fraction_value)
    audit.max_leakage_error = max(audit.max_leakage_error, error)
    audit.require(math.isclose(outside, squared * fraction_value,
                               rel_tol=2e-12, abs_tol=1e-18),
                  f"{label}: leakage identity differs")
    audit.require(-1e-7 <= fraction_value <= 1.000001,
                  f"{label}: leakage outside numerical range")
    return outside, squared


def validate_steps(rows: list[dict[str, Any]], branch: dict[str, Any],
                   audit: Audit, label: str) -> dict[str, Any]:
    policy, base, lr = branch["policy"], branch["base"], branch["lr"]
    sums = {name: 0.0 for name in ("loss", "raw_gradient_norm", "applied_gradient_norm",
                                   "native_gradient_norm", "total_displacement_norm",
                                   "data_displacement_norm", "nominal_decay_norm",
                                   "raw_gradient_dot_data_displacement")}
    leakage = {kind: [0.0, 0.0] for kind in ("total", "data")}
    signed_dots: list[float] = []
    active_sums = {name: 0.0 for name in sums}
    active_leakage = {kind: [0.0, 0.0] for kind in ("total", "data")}
    active_dots: list[float] = []
    for index, row in enumerate(rows, start=1):
        prefix = f"{label} step {index}"
        audit.require(row.get("step") == index and row.get("schema") == "i14_optimizer_step_v1"
                      and row.get("base") == base and row.get("lr") == lr
                      and row.get("policy") == policy, f"{prefix}: identity differs")
        observer = row.get("observer", {})
        active = index > 100
        audit.require(observer == {"step_before": index - 1, "step_after": index,
                                   "filtering_active": active, "used": True},
                      f"{prefix}: observer counter/status differs")
        audit.require(row.get("gradient_filter_applied") is (policy == "current32" and active),
                      f"{prefix}: filter-applied flag differs")
        gradient, displacement = row["gradient"], row["displacement"]
        values = {
            "loss": finite(row["loss"], prefix + " loss"),
            "raw_gradient_norm": finite(gradient["raw_norm"], prefix + " raw norm"),
            "applied_gradient_norm": finite(gradient["applied_norm"], prefix + " applied norm"),
            "native_gradient_norm": finite(gradient["native_norm"], prefix + " native norm"),
            "total_displacement_norm": finite(displacement["total_norm"], prefix + " total norm"),
            "data_displacement_norm": finite(displacement["data_norm"], prefix + " data norm"),
            "nominal_decay_norm": finite(displacement["nominal_decay_norm"], prefix + " decay norm"),
            "raw_gradient_dot_data_displacement": finite(
                displacement["raw_gradient_dot_data_delta"], prefix + " raw/data dot"),
        }
        audit.require(all(value >= 0 for key, value in values.items()
                          if key != "raw_gradient_dot_data_displacement"),
                      f"{prefix}: norm/loss is negative")
        for key, value in values.items():
            sums[key] += value
        signed_dots.append(values["raw_gradient_dot_data_displacement"])
        if active:
            for key, value in values.items():
                active_sums[key] += value
            active_dots.append(values["raw_gradient_dot_data_displacement"])
        decay = row.get("decay", {})
        audit.require(decay.get("coefficient") == 0.01
                      and decay.get("factor") == 1 - lr * 0.01
                      and decay.get("manual_before_optimizer_step") is (base != "adamw"),
                      f"{prefix}: decay declaration differs")
        for kind, norm_key in (("total", "total_displacement_norm"),
                               ("data", "data_displacement_norm")):
            saved = leakage_energy(displacement[kind + "_current_basis_leakage"],
                                   values[norm_key] ** 2, audit, f"{prefix} {kind} leakage")
            if saved is not None:
                leakage[kind][0] += saved[0]
                leakage[kind][1] += saved[1]
                if active:
                    active_leakage[kind][0] += saved[0]
                    active_leakage[kind][1] += saved[1]
    count = len(rows)
    active_count = max(0, count - 100)
    active_energy = {
        kind: {"outside_energy": values[0], "total_energy": values[1],
               "fraction": None if values[1] == 0 else values[0] / values[1],
               "definition": "sum outside squared energy / sum total squared energy over steps101+"}
        for kind, values in active_leakage.items()}
    return {"step_count": count,
            "arithmetic_step_means": {key: (value / count if count else None)
                                      for key, value in sums.items()},
            "signed_raw_gradient_dot_data_displacement": {
                "mean": None if not signed_dots else sum(signed_dots) / len(signed_dots),
                "minimum": None if not signed_dots else min(signed_dots),
                "maximum": None if not signed_dots else max(signed_dots),
                "negative_count": sum(value < 0 for value in signed_dots),
                "zero_count": sum(value == 0 for value in signed_dots),
                "positive_count": sum(value > 0 for value in signed_dots),
            },
            "energy_weighted_current_basis_leakage": {
                kind: {"outside_energy": values[0], "total_energy": values[1],
                       "fraction": None if values[1] == 0 else values[0] / values[1],
                       "definition": "sum outside squared energy / sum total squared energy within trajectory"}
                for kind, values in leakage.items()},
            "filtering_active_window": {
                "first_step": 101, "last_step": count if active_count else None,
                "step_count": active_count,
                "arithmetic_step_means": {
                    key: (value / active_count if active_count else None)
                    for key, value in active_sums.items()},
                "signed_raw_gradient_dot_data_displacement": {
                    "mean": None if not active_dots else sum(active_dots) / len(active_dots),
                    "minimum": None if not active_dots else min(active_dots),
                    "maximum": None if not active_dots else max(active_dots),
                    "negative_count": sum(value < 0 for value in active_dots),
                    "zero_count": sum(value == 0 for value in active_dots),
                    "positive_count": sum(value > 0 for value in active_dots),
                },
                "energy_weighted_current_basis_leakage": active_energy,
                "scope": "observer filtering_active=True; raw policy still restores raw delivery",
            }}


def utility(row: dict[str, Any], split: str, metric: str) -> float:
    if metric == "ce":
        return -float(row[split]["clean_ce"])
    if metric == "accuracy":
        return float(row[split]["clean_accuracy"])
    raise ValueError(metric)


def complete_pair_effect(branches: dict[tuple[Any, ...], dict[str, Any]], seed: int,
                         base: str, target: str, horizon: int, split: str,
                         metric: str) -> float | None:
    raw, current = (branches.get((seed, base, target, policy)) for policy in POLICIES)
    if raw is None or current is None:
        return None
    left, right = raw["curve_by_horizon"].get(horizon), current["curve_by_horizon"].get(horizon)
    if left is None or right is None:
        return None
    return utility(right, split, metric) - utility(left, split, metric)


def seed_aggregate(values: dict[int, float | None]) -> dict[str, Any]:
    ordered = [values.get(seed) for seed in CONFIRMATION_SEEDS]
    available = all(value is not None for value in ordered)
    return {"per_seed": {str(seed): values.get(seed) for seed in CONFIRMATION_SEEDS},
            "all_three_seed_values": ordered, "available": available,
            "mean": sum(ordered) / 3 if available else None,
            "no_survivor_averaging": True}


def endpoint_effects(branches: dict[tuple[Any, ...], dict[str, Any]],
                     horizon: int) -> list[dict[str, Any]]:
    result = []
    for base in BASES:
        for target in TARGETS:
            for metric in ("ce", "accuracy"):
                values = {seed: complete_pair_effect(branches, seed, base, target, horizon,
                                                     "auxiliary", metric)
                          for seed in CONFIRMATION_SEEDS}
                result.append({"base": base, "target": target, "metric": metric,
                               "horizon": horizon, "effect": seed_aggregate(values)})
    return result


def base_interactions(primary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    indexed = {(row["base"], row["target"], row["metric"]): row for row in primary}
    result = []
    for target in TARGETS:
        for metric in ("ce", "accuracy"):
            for high, low in (("sgdm", "sgd"), ("adamw", "sgdm")):
                h, l = indexed[(high, target, metric)]["effect"], indexed[(low, target, metric)]["effect"]
                values = {seed: (None if h["per_seed"][str(seed)] is None
                                      or l["per_seed"][str(seed)] is None else
                                      h["per_seed"][str(seed)] - l["per_seed"][str(seed)])
                          for seed in CONFIRMATION_SEEDS}
                result.append({"higher_base": high, "lower_base": low, "target": target,
                               "metric": metric, "effect_difference": seed_aggregate(values),
                               "scope": "descriptive; rates and per-update decay factors differ"})
    return result


def selector_outcomes(branches: dict[tuple[Any, ...], dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for key, branch in sorted(branches.items()):
        selected = branch.get("independent_selection")
        for selector in ("minimum_validation_ce", "maximum_validation_accuracy"):
            horizon = None if selected is None else selected[selector]
            outcome = None
            if horizon is not None:
                point = branch["curve_by_horizon"][horizon]
                outcome = {"auxiliary_clean_ce": point["auxiliary"]["clean_ce"],
                           "auxiliary_clean_accuracy": point["auxiliary"]["clean_accuracy"]}
            rows.append({"seed": key[0], "base": key[1], "target": key[2],
                         "policy": key[3], "selector": selector,
                         "selected_horizon": horizon, "auxiliary_outcome": outcome})
    return rows


def selector_effects(branches: dict[tuple[Any, ...], dict[str, Any]]) -> list[dict[str, Any]]:
    """Paired effects when each arm uses its own predeclared validation selector."""
    result = []
    for base in BASES:
        for target in TARGETS:
            for selector in ("minimum_validation_ce", "maximum_validation_accuracy"):
                for metric in ("ce", "accuracy"):
                    values: dict[int, float | None] = {}
                    horizons: dict[str, Any] = {}
                    for seed in CONFIRMATION_SEEDS:
                        raw, current = (branches[(seed, base, target, policy)]
                                        for policy in POLICIES)
                        if raw["status"] != "complete" or current["status"] != "complete":
                            values[seed] = None
                            horizons[str(seed)] = None
                            continue
                        raw_h = raw["independent_selection"][selector]
                        current_h = current["independent_selection"][selector]
                        values[seed] = (utility(current["curve_by_horizon"][current_h],
                                                "auxiliary", metric)
                                        - utility(raw["curve_by_horizon"][raw_h],
                                                  "auxiliary", metric))
                        horizons[str(seed)] = {"raw": raw_h, "current32": current_h}
                    result.append({"base": base, "target": target, "selector": selector,
                                   "auxiliary_metric": metric,
                                   "selected_horizons_by_seed": horizons,
                                   "effect": seed_aggregate(values)})
    return result


def progress_rows(branches: dict[tuple[Any, ...], dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for key, branch in sorted(branches.items()):
        for horizon in HORIZONS:
            point = branch["curve_by_horizon"].get(horizon)
            values = None
            if point is not None:
                values = {}
                for split in ("train", "validation", "auxiliary"):
                    h0_point = branch["curve_by_horizon"][0]
                    h100_point = branch["curve_by_horizon"].get(100)
                    now_ce, now_accuracy = (point[split]["clean_ce"],
                                            point[split]["clean_accuracy"])
                    values[f"{split}_clean_ce"] = now_ce
                    values[f"{split}_clean_ce_reduction_from_h0"] = (
                        h0_point[split]["clean_ce"] - now_ce)
                    values[f"{split}_clean_ce_reduction_from_h100"] = (
                        None if h100_point is None else
                        h100_point[split]["clean_ce"] - now_ce)
                    values[f"{split}_clean_accuracy"] = now_accuracy
                    values[f"{split}_clean_accuracy_gain_from_h0"] = (
                        now_accuracy - h0_point[split]["clean_accuracy"])
                    values[f"{split}_clean_accuracy_gain_from_h100"] = (
                        None if h100_point is None else
                        now_accuracy - h100_point[split]["clean_accuracy"])
                train = point["train"]
                values.update(train_fixed_ce=train["fixed_ce"],
                              train_soft_ce=train["soft_ce"],
                              train_fixed_minus_soft_ce=train["fixed_minus_soft_ce"],
                              train_fixed_accuracy=train["fixed_accuracy"],
                              train_confidence=train["mean_max_probability"])
            rows.append({"seed": key[0], "base": key[1], "target": key[2],
                         "policy": key[3], "horizon": horizon, "values": values})
    return rows


def trajectory_leakage_seed_means(branches: dict[tuple[Any, ...], dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for base in BASES:
        for target in TARGETS:
            for policy in POLICIES:
                for window in ("full_trajectory", "filtering_active_steps101_2000"):
                    per_seed = {}
                    complete = {}
                    for seed in CONFIRMATION_SEEDS:
                        branch = branches[(seed, base, target, policy)]
                        summary = branch.get("diagnostic_summary")
                        complete[str(seed)] = branch["status"] == "complete"
                        if summary is None:
                            per_seed[str(seed)] = None
                        elif window == "full_trajectory":
                            per_seed[str(seed)] = summary[
                                "energy_weighted_current_basis_leakage"]
                        else:
                            per_seed[str(seed)] = summary["filtering_active_window"][
                                "energy_weighted_current_basis_leakage"]
                    for kind in ("data", "total"):
                        fractions = [None if per_seed[str(seed)] is None else
                                     per_seed[str(seed)][kind]["fraction"]
                                     for seed in CONFIRMATION_SEEDS]
                        available = all(complete.values()) and all(
                            value is not None for value in fractions)
                        result.append({"base": base, "target": target, "policy": policy,
                            "window": window, "displacement": kind,
                            "per_seed_branch_fractions": {
                                str(seed): fractions[index]
                                for index, seed in enumerate(CONFIRMATION_SEEDS)},
                            "equal_seed_mean": (sum(fractions) / 3 if available else None),
                            "available": available, "complete_by_seed": complete,
                            "aggregation": "energy ratio within trajectory, then equal seeds"})
    return result


def step_diagnostic_seed_means(
        branches: dict[tuple[Any, ...], dict[str, Any]]) -> list[dict[str, Any]]:
    """Equal-seed summaries of within-trajectory arithmetic norm/dot means."""
    result = []
    metrics = ("raw_gradient_norm", "applied_gradient_norm", "native_gradient_norm",
               "total_displacement_norm", "data_displacement_norm", "nominal_decay_norm",
               "raw_gradient_dot_data_displacement")
    for base in BASES:
        for target in TARGETS:
            for policy in POLICIES:
                for window in ("full_trajectory", "filtering_active_steps101_2000"):
                    for metric in metrics:
                        values = {}
                        complete = {}
                        for seed in CONFIRMATION_SEEDS:
                            branch = branches[(seed, base, target, policy)]
                            complete[str(seed)] = branch["status"] == "complete"
                            diagnostic = branch.get("diagnostic_summary")
                            if diagnostic is None:
                                value = None
                            else:
                                source = (diagnostic["arithmetic_step_means"]
                                          if window == "full_trajectory" else
                                          diagnostic["filtering_active_window"][
                                              "arithmetic_step_means"])
                                value = source[metric]
                            values[str(seed)] = value
                        ordered = [values[str(seed)] for seed in CONFIRMATION_SEEDS]
                        available = all(complete.values()) and all(
                            value is not None for value in ordered)
                        result.append({"base": base, "target": target, "policy": policy,
                            "window": window, "metric": metric, "per_seed": values,
                            "equal_seed_mean": sum(ordered) / 3 if available else None,
                            "available": available, "complete_by_seed": complete,
                            "aggregation": "arithmetic mean within trajectory, then equal seeds"})
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    root = args.artifacts.resolve(strict=True)
    if not root.is_dir():
        raise SystemExit("--artifacts must be a directory")
    args.output.parent.resolve(strict=True)
    args.output.mkdir(exist_ok=False)
    torch.set_num_threads(1)
    audit = Audit()
    analysis_provenance = analysis_source_provenance(audit)

    phases = ("smoke", "calibration", "confirmation")
    manifests, completions, indices = {}, {}, {}
    artifact_indices: dict[str, dict[str, dict[str, Any]]] = {}
    for phase in phases:
        directory = root / phase
        manifests[phase] = read_json(directory / "manifest.json")
        completions[phase] = read_json(directory / "completion.json")
        manifest, completion = manifests[phase], completions[phase]
        audit.require(manifest.get("schema") == "i14_manifest_v1" and manifest.get("phase") == phase,
                      f"{phase}: manifest identity differs")
        audit.require(completion.get("schema") == "i14_completion_v1"
                      and completion.get("status") == "complete"
                      and completion.get("phase") == phase,
                      f"{phase}: completion identity/status differs")
        audit.require(completion.get("source_hashes") == manifest.get("source_hashes")
                      and completion.get("frozen_commit") == manifest.get("frozen_commit"),
                      f"{phase}: completion source binding differs")
        records = completion.get("artifacts", [])
        index = {record.get("name"): record for record in records}
        audit.require(len(index) == len(records), f"{phase}: duplicate artifact names")
        artifact_indices[phase] = index
        for record in records:
            verify_record(directory, record, audit, phase)
        actual_names = {path.name for path in directory.iterdir() if path.is_file()}
        audit.require(actual_names == set(index) | {"completion.json"},
                      f"{phase}: unlisted, partial, or missing phase files exist")
        audit.require(index.get("manifest.json") is not None,
                      f"{phase}: consumed manifest is absent from completion artifacts")
        if phase != "smoke":
            audit.require(index.get("trajectories.json") is not None,
                          f"{phase}: consumed trajectory index is absent from completion artifacts")
            indices[phase] = read_json(directory / "trajectories.json")
    source_audit = verify_sources(list(manifests.values()), audit)
    expected_root_names = set(phases) | {f"attempt-{phase}.json" for phase in phases}
    audit.require({path.name for path in root.iterdir()} == expected_root_names,
                  "artifact root contains an unlisted, partial, or missing entry")
    for phase in phases:
        attempt = read_json(root / f"attempt-{phase}.json")
        audit.require(attempt.get("phase") == phase and attempt.get("restart") == "forbidden"
                      and attempt.get("frozen_commit") == source_audit["frozen_commit"]
                      and attempt.get("source_hashes") == manifests[phase]["source_hashes"],
                      f"{phase}: attempt provenance differs")

    plan_info = []
    saved_plans: dict[tuple[str, int], dict[str, Any]] = {}
    for phase, seeds in (("calibration", CALIBRATION_SEEDS),
                         ("confirmation", CONFIRMATION_SEEDS)):
        for seed in seeds:
            # All plans originate in calibration. Confirmation plans are copied byte-for-byte.
            source_path = root / "calibration" / f"plan-{phase}-s{seed}.json"
            audit.require(source_path.name in artifact_indices["calibration"],
                          f"{source_path.name}: absent from calibration completion artifacts")
            plan = read_json(source_path)
            saved_plans[(phase, seed)] = plan
            info = validate_plan(plan, phase, seed, audit, source_path.name)
            plan_info.append(info)
            if phase == "confirmation":
                copied = root / "confirmation" / source_path.name
                audit.require(copied.name in artifact_indices["confirmation"],
                              f"{copied.name}: absent from confirmation completion artifacts")
                audit.require(copied.is_file() and sha256(copied) == sha256(source_path),
                              f"confirmation plan copy differs for seed {seed}")
    calibration_indices = set().union(*(row["indices"] for row in plan_info
                                        if row["phase"] == "calibration"))
    confirmation_indices = set().union(*(row["indices"] for row in plan_info
                                         if row["phase"] == "confirmation"))
    audit.require(calibration_indices.isdisjoint(confirmation_indices),
                  "calibration and confirmation plan indices overlap")

    # Bind the exact acquisition inputs and independently recover corruption counts.
    audit.require("dataset-inputs.json" in artifact_indices["calibration"],
                  "dataset-inputs.json is absent from calibration completion artifacts")
    dataset_inputs = read_json(root / "calibration" / "dataset-inputs.json")
    audit.require(tuple(dataset_inputs) == ("train-images-idx3-ubyte",
                                            "train-labels-idx1-ubyte"),
                  "dataset input hash topology differs")
    for name, expected in dataset_inputs.items():
        path = DATA_ROOT / name
        audit.require(path.is_file() and sha256(path) == expected,
                      f"current dataset input differs: {name}")
    label_bytes = (DATA_ROOT / "train-labels-idx1-ubyte").read_bytes()
    audit.require(len(label_bytes) == 60008
                  and struct.unpack(">II", label_bytes[:8]) == (2049, 60000),
                  "MNIST label IDX topology differs")
    labels = np.frombuffer(label_bytes, dtype=np.uint8, offset=8)
    corruption_audit = []
    for phase, seeds in (("calibration", CALIBRATION_SEEDS),
                         ("confirmation", CONFIRMATION_SEEDS)):
        for seed in seeds:
            name = f"corruption-s{seed}.json"
            audit.require(name in artifact_indices[phase],
                          f"{phase}/{name}: absent from completion artifacts")
            saved = read_json(root / phase / name)
            plan = saved_plans[(phase, seed)]
            expected = corruption_counts(plan, labels)
            audit.require(saved == expected, f"{phase}/{name}: corruption counts differ")
            corruption_audit.append({"phase": phase, "seed": seed, **expected})

    branches: dict[str, dict[tuple[Any, ...], dict[str, Any]]] = {
        "calibration": {}, "confirmation": {}}
    expected_keys = {
        "calibration": {(seed, base, lr) for seed in CALIBRATION_SEEDS
                        for base in BASES for lr in RATES[base]},
        "confirmation": {(seed, base, target, policy) for seed in CONFIRMATION_SEEDS
                         for base in BASES for target in TARGETS for policy in POLICIES},
    }
    trajectory_summaries = {"calibration": [], "confirmation": []}
    for phase in ("calibration", "confirmation"):
        directory, entries = root / phase, indices[phase].get("entries", [])
        audit.require(indices[phase].get("expected") == len(expected_keys[phase])
                      and len(entries) == len(expected_keys[phase]),
                      f"{phase}: trajectory entry count differs")
        for entry in entries:
            identity = entry.get("id", "unknown")
            try:
                record = entry["artifact"]
                audit.require(artifact_indices[phase].get(record["name"]) == record,
                              f"{identity}: trajectory artifact binding differs")
                trajectory = read_json(directory / record["name"])
                key = ((trajectory["seed"], trajectory["base"], trajectory["lr"])
                       if phase == "calibration" else
                       (trajectory["seed"], trajectory["base"], trajectory["target"],
                        trajectory["policy"]))
                audit.require(key not in branches[phase], f"{phase}: duplicate key {key}")
                audit.require(trajectory.get("schema") == "i14_trajectory_v1"
                              and trajectory.get("phase") == phase,
                              f"{identity}: trajectory schema/phase differs")
                for name in ("id", "phase", "seed", "base", "lr", "policy", "target",
                             "status", "completed_steps", "initial_model_digest",
                             "initial_full_state_digest", "warmup_full_state_digest",
                             "initial_evaluation_digest", "warmup_evaluation_digest",
                             "startup_failure_signature"):
                    audit.require(entry.get(name) == trajectory.get(name),
                                  f"{identity}: index/{name} binding differs")
                if phase == "calibration":
                    audit.require(trajectory["policy"] == "raw" and trajectory["target"] == "clean"
                                  and trajectory["lr"] in RATES[trajectory["base"]],
                                  f"{identity}: calibration identity differs")
                else:
                    audit.require(trajectory["policy"] in POLICIES
                                  and trajectory["target"] in TARGETS,
                                  f"{identity}: confirmation identity differs")
                status, completed = trajectory["status"], trajectory["completed_steps"]
                expected_id = (f"s{trajectory['seed']}-{trajectory['base']}-"
                               f"lr{trajectory['lr']:.8g}-{trajectory['target']}-"
                               f"{trajectory['policy']}")
                audit.require(status in ("complete", "numerical_failure")
                              and type(completed) is int and 0 <= completed <= 2000,
                              f"{identity}: status/completed steps differ")
                audit.require(identity == trajectory["id"] == expected_id
                              and trajectory.get("requested_steps") == 2000,
                              f"{identity}: ID/requested steps differ")
                curve = trajectory.get("curve", [])
                curve_horizons = [row.get("horizon") for row in curve]
                audit.require(curve_horizons == list(HORIZONS[:len(curve_horizons)])
                              and bool(curve_horizons) and curve_horizons[-1] <= completed,
                              f"{identity}: curve is not a scheduled completed prefix")
                curve_by_horizon = {}
                for point in curve:
                    validate_evaluation(point, phase == "calibration", audit,
                                        f"{identity} h{point.get('horizon')}")
                    curve_by_horizon[point["horizon"]] = point
                steps = trajectory.get("steps", [])
                audit.require(len(steps) == completed, f"{identity}: step count differs")
                diagnostic = validate_steps(steps, trajectory, audit, identity)
                if status == "complete":
                    audit.require(completed == 2000 and curve_horizons == list(HORIZONS)
                                  and trajectory.get("failure") is None,
                                  f"{identity}: complete trajectory is incomplete")
                    independent_selection = selection(curve)
                    audit.require(trajectory.get("selected_horizons") == independent_selection,
                                  f"{identity}: saved selector differs")
                else:
                    independent_selection = None
                    failure = trajectory.get("failure")
                    attempted = failure.get("attempted_step") if type(failure) is dict else None
                    attempted_consistent = (attempted == completed + 1 or
                        (attempted == completed and attempted in HORIZONS
                         and curve_horizons[-1] < attempted))
                    audit.require(type(failure) is dict and failure.get("type") == "NumericalFailure"
                                  and failure.get("completed_steps") == completed
                                  and failure.get("last_valid_horizon") == curve_horizons[-1]
                                  and attempted_consistent,
                                  f"{identity}: numerical failure evidence differs")
                    failed_record = failure.get("state_artifact", {})
                    audit.require(artifact_indices[phase].get(failed_record.get("name"))
                                  == failed_record,
                                  f"{identity}: failed-state artifact binding differs")
                    failed_state = torch.load(directory / failed_record["name"],
                                              weights_only=True, map_location="cpu")
                    if trajectory.get("warmup_full_state_digest") is None:
                        failure_without_artifact = {key: value for key, value in failure.items()
                                                    if key != "state_artifact"}
                        steps_without_policy = [
                            {key: value for key, value in row.items() if key != "policy"}
                            for row in steps]
                        independent_failure_signature = failure_fingerprint({
                            "state": failed_state, "failure": failure_without_artifact,
                            "curve": curve, "steps": steps_without_policy})
                        audit.require(trajectory.get("startup_failure_signature")
                                      == independent_failure_signature,
                                      f"{identity}: startup failure signature differs")
                    else:
                        audit.require(trajectory.get("startup_failure_signature") is None,
                                      f"{identity}: post-warmup failure has startup signature")
                    audit.require(trajectory.get("selected_horizons") is None,
                                  f"{identity}: failed trajectory has selected outcome")
                checkpoints = trajectory.get("checkpoints", [])
                checkpoint_horizons = [row.get("horizon") for row in checkpoints]
                audit.require(checkpoint_horizons == curve_horizons,
                              f"{identity}: checkpoint prefix differs")
                for checkpoint in checkpoints:
                    horizon = checkpoint["horizon"]
                    if horizon in (0, 100, 2000):
                        state_record = checkpoint.get("full_state")
                        audit.require(artifact_indices[phase].get(state_record.get("name")) == state_record,
                                      f"{identity} h{horizon}: state artifact binding differs")
                        state = torch.load(directory / state_record["name"], weights_only=True,
                                           map_location="cpu")
                        digest = tree_digest(state)
                        audit.tree_digests += 1
                        audit.require(digest == checkpoint.get("full_state_digest"),
                                      f"{identity} h{horizon}: full-state digest differs")
                        audit.require(state.get("schema") == "i14_optimizer_snapshot_v1"
                                      and state.get("base") == trajectory["base"]
                                      and state.get("lr") == trajectory["lr"]
                                      and state.get("tracker", {}).get("step_count") == horizon,
                                      f"{identity} h{horizon}: state identity/counter differs")
                        if horizon == 0:
                            audit.require(digest == trajectory["initial_full_state_digest"],
                                          f"{identity}: initial state digest differs")
                            audit.require(tree_digest(state["model_state"])
                                          == trajectory["initial_model_digest"],
                                          f"{identity}: initial model digest differs")
                        if horizon == 100:
                            audit.require(digest == trajectory["warmup_full_state_digest"],
                                          f"{identity}: warmup state digest differs")
                    else:
                        model_record = checkpoint.get("model_state")
                        audit.require(artifact_indices[phase].get(model_record.get("name")) == model_record,
                                      f"{identity} h{horizon}: model artifact binding differs")
                audit.require(tree_digest(curve[0]) == trajectory["initial_evaluation_digest"],
                              f"{identity}: initial evaluation digest differs")
                if 100 in curve_by_horizon:
                    audit.require(tree_digest(curve_by_horizon[100])
                                  == trajectory["warmup_evaluation_digest"],
                                  f"{identity}: warmup evaluation digest differs")
                row = dict(trajectory)
                row.update(curve_by_horizon=curve_by_horizon,
                           independent_selection=independent_selection,
                           diagnostic_summary=diagnostic)
                branches[phase][key] = row
                trajectory_summaries[phase].append({"id": identity, "key": list(key),
                    "status": status, "completed_steps": completed,
                    "failure": trajectory.get("failure"),
                    "curve": curve, "independent_selection": independent_selection,
                    "diagnostic_summary": diagnostic})
            except Exception as exc:
                audit.errors.append(f"{phase}/{identity}: analysis failed: {type(exc).__name__}: {exc}")
        audit.require(set(branches[phase]) == expected_keys[phase],
                      f"{phase}: trajectory membership differs")
        statuses = list(branches[phase].values())
        audit.require(completions[phase].get("trajectories") == len(expected_keys[phase])
                      and completions[phase].get("numerical_failures")
                      == sum(row["status"] == "numerical_failure" for row in statuses)
                      and completions[phase].get("completed_training_updates")
                      == sum(row["completed_steps"] for row in statuses),
                      f"{phase}: completion trajectory/update counts differ")
        audit.require(completions[phase].get("all_requested_endpoints_present")
                      is all(row["status"] == "complete" for row in statuses),
                      f"{phase}: completion endpoint-coverage flag differs")

    # Independently recover and bind the calibration decision.
    independent_choice = choose_rates(branches["calibration"])
    require_member(artifact_indices["calibration"], "selection.json", audit, "calibration")
    saved_choice = read_json(root / "calibration" / "selection.json")
    audit.require({key: saved_choice.get(key) for key in independent_choice} == independent_choice,
                  "saved calibration selection differs from independent selection")
    audit.require(saved_choice.get("dataset_inputs") == dataset_inputs,
                  "calibration selection dataset binding differs")
    require_member(artifact_indices["confirmation"], "calibration-binding.json", audit,
                   "confirmation")
    binding = read_json(root / "confirmation" / "calibration-binding.json")
    audit.require(binding.get("selection_sha256") == sha256(root / "calibration" / "selection.json")
                  and binding.get("completion_sha256") == sha256(root / "calibration" / "completion.json")
                  and binding.get("selection") == saved_choice,
                  "confirmation calibration binding differs")
    audit.require(independent_choice["ready_for_confirmation"],
                  "calibration did not qualify confirmation")

    # Exact common starts and untreated warmup for each confirmation pair.
    confirmation = branches["confirmation"]
    calibration = branches["calibration"]
    for seed in CALIBRATION_SEEDS:
        rows = [row for key, row in calibration.items() if key[0] == seed]
        audit.require(len({row["initial_model_digest"] for row in rows}) == 1
                      and len({row["initial_evaluation_digest"] for row in rows}) == 1,
                      f"calibration seed {seed}: initial weights/evaluation differ")
    pair_checked = 0
    pair_missing = 0
    for seed in CONFIRMATION_SEEDS:
        initial_models = {row["initial_model_digest"] for key, row in confirmation.items()
                          if key[0] == seed}
        initial_evaluations = {row["initial_evaluation_digest"] for key, row in confirmation.items()
                               if key[0] == seed}
        audit.require(len(initial_models) == 1 and len(initial_evaluations) == 1,
                      f"seed {seed}: initial model weights/evaluation differ")
        for base in BASES:
            expected_lr = independent_choice["selected_rates"][base]
            base_rows = [row for key, row in confirmation.items()
                         if key[0] == seed and key[1] == base]
            audit.require(len({row["initial_full_state_digest"] for row in base_rows}) == 1,
                          f"seed {seed}/{base}: initial complete states differ")
            for target in TARGETS:
                raw, current = (confirmation[(seed, base, target, policy)] for policy in POLICIES)
                audit.require(raw["lr"] == current["lr"] == expected_lr,
                              f"seed {seed}/{base}/{target}: selected rate differs")
                audit.require(raw["initial_full_state_digest"] == current["initial_full_state_digest"]
                              and raw["initial_evaluation_digest"] == current["initial_evaluation_digest"],
                              f"seed {seed}/{base}/{target}: initial pair differs")
                if raw["warmup_full_state_digest"] is not None \
                        and current["warmup_full_state_digest"] is not None:
                    audit.require(raw["warmup_full_state_digest"] == current["warmup_full_state_digest"]
                                  and raw["warmup_evaluation_digest"] == current["warmup_evaluation_digest"],
                                  f"seed {seed}/{base}/{target}: warmup pair differs")
                    audit.require(raw["curve_by_horizon"].get(100)
                                  == current["curve_by_horizon"].get(100),
                                  f"seed {seed}/{base}/{target}: warmup metrics differ")
                    pair_checked += 1
                else:
                    audit.require(raw["status"] == current["status"] == "numerical_failure"
                                  and raw["startup_failure_signature"] is not None
                                  and raw["startup_failure_signature"]
                                  == current["startup_failure_signature"],
                                  f"seed {seed}/{base}/{target}: startup failure differs")
                    pair_missing += 1

    independent_pair_checks = {"expected_pairs": 18,
        "available_pairs_checked_exact": pair_checked,
        "pairs_unavailable_due_to_numerical_failure": pair_missing,
        "all_pairs_reached_warmup": pair_missing == 0}
    audit.require(indices["confirmation"].get("initial_weights_shared") is True
                  and indices["confirmation"].get("warmup_pair_checks")
                  == independent_pair_checks,
                  "confirmation declared warmup-pair audit differs")

    primary = endpoint_effects(confirmation, 2000)
    summary = {
        "schema": "i14_cross_optimizer_analysis_summary_v1",
        "artifact_root": str(root),
        "scope": [
            "Calibration and confirmation are separate; calibration outcomes are not pooled into science effects.",
            "Positive endpoint CE/accuracy effects favor current32 over raw within the same base/target/seed.",
            "Cross-base interactions are descriptive because rates and per-update decay factors differ.",
            "A missing required pair makes that estimand unavailable; no survivor means are reported.",
            "Saved scalar evaluation records and per-step scalar identities were audited; no model/optimizer forward replay or per-step vector replay was performed.",
        ],
        "counts": {"calibration_trajectories": len(branches["calibration"]),
                   "confirmation_trajectories": len(confirmation),
                   "calibration_numerical_failures": sum(row["status"] != "complete"
                                                         for row in branches["calibration"].values()),
                   "confirmation_numerical_failures": sum(row["status"] != "complete"
                                                          for row in confirmation.values())},
        "input_provenance": {"dataset_inputs": dataset_inputs,
                             "corruption_counts": corruption_audit},
        "calibration": {"independent_selection": independent_choice,
                        "trajectories": trajectory_summaries["calibration"],
                        "not_pooled_with_confirmation": True},
        "effect_definition": {"ce": "auxiliary clean CE(raw)-CE(current32)",
                              "accuracy": "auxiliary clean accuracy(current32)-accuracy(raw)",
                              "positive_favors": "current32"},
        "primary_endpoint_effects": primary,
        "descriptive_base_interactions": base_interactions(primary),
        "all_horizon_effects": [row for horizon in HORIZONS
                                for row in endpoint_effects(confirmation, horizon)],
        "validation_selected_auxiliary_outcomes": selector_outcomes(confirmation),
        "validation_selected_auxiliary_effects": selector_effects(confirmation),
        "absolute_progress_and_training_curves": progress_rows(confirmation),
        "trajectory_step_diagnostics": trajectory_summaries["confirmation"],
        "energy_weighted_leakage": trajectory_leakage_seed_means(confirmation),
        "step_norm_and_signed_dot_seed_means": step_diagnostic_seed_means(confirmation),
    }
    audit_payload = {
        "schema": "i14_cross_optimizer_analysis_audit_v1",
        "status": "pass" if not audit.errors else "fail",
        "checks": audit.checks, "errors": audit.errors, "warnings": audit.warnings,
        "source_provenance": source_audit,
        "analysis_source_provenance": analysis_provenance,
        "completion_hash_files_verified": audit.hash_files,
        "hash_bytes_streamed": audit.hash_bytes,
        "complete_state_tree_digests_verified": audit.tree_digests,
        "maximum_training_residual_identity_error": audit.max_residual_error,
        "maximum_leakage_energy_identity_error": audit.max_leakage_error,
        "plan_checks": {"saved_plans": len(plan_info),
                        "within_plan_split_uniqueness": True,
                        "calibration_confirmation_indices_disjoint":
                            calibration_indices.isdisjoint(confirmation_indices)},
        "coverage": {"calibration": len(branches["calibration"]),
                     "confirmation": len(confirmation), "primary_estimands": len(primary)},
        "audit_boundaries": [
            "Artifact SHA256 and saved full-state tree digests were independently recomputed on CPU.",
            "Saved scalar evaluations were not rerun against MNIST or the model.",
            "Intermediate model checkpoints were file-hashed but not semantically replayed.",
            "Per-step tensors were not retained; leakage and displacement checks are scalar identities only.",
        ],
    }
    with (args.output / "summary.json").open("x", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, allow_nan=False)
        handle.write("\n")
    with (args.output / "audit.json").open("x", encoding="utf-8") as handle:
        json.dump(audit_payload, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"status": audit_payload["status"], "checks": audit.checks,
                      "errors": len(audit.errors), "hash_files": audit.hash_files,
                      "tree_digests": audit.tree_digests}, allow_nan=False))
    return 0 if not audit.errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
