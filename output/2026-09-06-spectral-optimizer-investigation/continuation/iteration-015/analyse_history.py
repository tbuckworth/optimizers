#!/usr/bin/env python3
"""Independent CPU audit and scalar analysis for the fixed I15 history study."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import subprocess
from typing import Any

import torch


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
I14 = HERE.parent / "iteration-014"
I14_ROOT = Path("/tmp/spectral-experiment-artifacts/spectral-i14-001.SJWZCj")
DATA_ROOT = Path("data/MNIST/raw")
I14_SUPPLEMENT_SHA = "13c2ea817a141cc5b5062fd8ac9e56611fa19dea5ef8bbdacfc56ad7cf717809"
ACQUISITION_COMMIT = "2bccbc6a883f1c4c11950965f9801d2c03d4350b"
PHASES = ("smoke", "confirmation")
SEEDS = (200, 201, 202)
TARGETS = ("clean", "fixed")
NEW_POLICIES = ("current_projected_history", "mean_native", "mean_projected_history")
POLICIES = ("raw", "current_native") + NEW_POLICIES
HORIZONS = (100, 250, 500, 1000, 1500, 2000)
SELECTORS = ("minimum_validation_ce", "maximum_validation_accuracy")
METRICS = ("ce", "accuracy")
PRIMARY_FAMILIES = ("H_history_under_current", "M_mean_under_projected_history",
                    "S_mean_by_history_interaction")
ARTIFACT_CAP = 3 * 1024**3
SHA_RE = __import__("re").compile(r"[0-9a-f]{64}")
ACQUISITION_SOURCES = (
    "spectral_filter.py",
    "output/2026-09-06-spectral-optimizer-investigation/continuation/iteration-009/neural_core.py",
    "output/2026-09-06-spectral-optimizer-investigation/continuation/iteration-014/optimizer_core.py",
    "output/2026-09-06-spectral-optimizer-investigation/continuation/iteration-014/data_plan.py",
    "output/2026-09-06-spectral-optimizer-investigation/continuation/iteration-014/run_cross_optimizer.py",
    "output/2026-09-06-spectral-optimizer-investigation/continuation/iteration-013/mean_core.py",
    "output/2026-09-06-spectral-optimizer-investigation/continuation/iteration-015/history_core.py",
    "output/2026-09-06-spectral-optimizer-investigation/continuation/iteration-015/test_history_core.py",
    "output/2026-09-06-spectral-optimizer-investigation/continuation/iteration-015/run_history_branches.py",
    "output/2026-09-06-spectral-optimizer-investigation/continuation/iteration-015/test_history_runner.py",
    "output/2026-09-06-spectral-optimizer-investigation/continuation/iteration-015/protocol.md",
    "output/2026-09-06-spectral-optimizer-investigation/continuation/iteration-015/predictions.md",
)


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024**2), b""):
            result.update(block)
    return result.hexdigest()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle, parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"nonfinite JSON constant {value} in {path.name}")))


def _digest_node(value: Any) -> Any:
    if type(value) is torch.Tensor:
        tensor = value.detach().cpu().contiguous()
        raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
        return ["tensor", str(tensor.dtype), list(tensor.shape),
                hashlib.sha256(raw).hexdigest()]
    if type(value) is dict:
        return ["dict", [[_digest_node(key), _digest_node(item)]
                         for key, item in value.items()]]
    if type(value) is list:
        return ["list", [_digest_node(item) for item in value]]
    if type(value) is tuple:
        return ["tuple", [_digest_node(item) for item in value]]
    if value is None or type(value) in (bool, int, str):
        return [type(value).__name__, value]
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("nonfinite float in complete state")
        return ["float", value.hex()]
    raise TypeError(f"unsupported complete-state value {type(value)!r}")


def tree_digest(value: Any) -> str:
    raw = json.dumps(_digest_node(value), ensure_ascii=True, allow_nan=False,
                     separators=(",", ":")).encode("ascii")
    return hashlib.sha256(b"i9_neural_tree_v1\n" + raw).hexdigest()


def finite(value: Any, label: str) -> float:
    if type(value) not in (int, float):
        raise ValueError(label + " is not a numeric JSON scalar")
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
        self.max_algebra_residual = 0.0
        self.max_leakage_identity_error = 0.0
        self.phase_completion_sha256: dict[str, str] = {}
        self.attempt_sha256: dict[str, str] = {}
        self.failure_state_tree_digests: dict[str, str] = {}
        self.artifact_root: str | None = None
        self.runtime_inventory: dict[str, Any] | None = None

    def require(self, condition: bool, message: str) -> None:
        self.checks += 1
        if not condition:
            self.errors.append(message)


def _safe_relative(name: Any) -> bool:
    return (type(name) is str and name not in ("", ".", "..")
            and not Path(name).is_absolute() and ".." not in Path(name).parts)


def verify_file(path: Path, expected_size: Any, expected_sha: Any,
                audit: Audit, label: str) -> bool:
    try:
        info = path.lstat()
        audit.require(stat.S_ISREG(info.st_mode) and not path.is_symlink(),
                      label + " is not a regular nonsymlink file")
        if not stat.S_ISREG(info.st_mode) or path.is_symlink():
            return False
        digest = sha256(path)
        audit.require(type(expected_size) is int and expected_size >= 0
                      and info.st_size == expected_size, label + " byte count differs")
        audit.require(type(expected_sha) is str and SHA_RE.fullmatch(expected_sha) is not None
                      and digest == expected_sha, label + " SHA256 differs")
        audit.hash_files += 1
        audit.hash_bytes += info.st_size
        return info.st_size == expected_size and digest == expected_sha
    except Exception as exc:
        audit.errors.append(f"{label}: file verification failed: {type(exc).__name__}: {exc}")
        return False


def verify_record(directory: Path, record: Any, audit: Audit,
                  label: str) -> Path | None:
    if type(record) is not dict or set(record) != {"name", "bytes", "sha256"} \
            or not _safe_relative(record.get("name")) \
            or Path(record["name"]).name != record["name"]:
        audit.errors.append(label + ": malformed artifact record")
        return None
    path = directory / record["name"]
    return path if verify_file(path, record["bytes"], record["sha256"], audit, label) else None


def phase_records(root: Path, phase: str, audit: Audit) -> tuple[dict[str, Any],
                                                                  dict[str, Any],
                                                                  dict[str, Any]]:
    directory = root / phase
    try:
        manifest = read_json(directory / "manifest.json")
        completion = read_json(directory / "completion.json")
        manifest_keys = ("schema", "phase", "frozen_commit", "source_hashes",
                         "torch_version", "numpy_version", "python_version", "gpu",
                         "policies", "horizons", "smoke_based_seconds_forecast",
                         "timing_safety_factor", "cloud_spend_usd", "authorized_budget_usd",
                         "shared_artifact_cap_bytes", "wall_limit_seconds",
                         "runtime_directory", "runtime_bytes_count_in_cap")
        completion_keys = ("schema", "status", "phase", "source_hashes", "frozen_commit",
                           "branches", "numerical_failures", "all_requested_endpoints_present",
                           "completed_training_updates", "history_update_seconds",
                           "elapsed_seconds", "peak_torch_gpu_bytes",
                           "shared_artifact_bytes_before_completion", "artifacts")
        audit.require(type(manifest) is dict and tuple(manifest) == manifest_keys
                      and manifest.get("schema") == "i15_manifest_v1"
                      and manifest.get("phase") == phase, phase + " manifest differs")
        audit.require(type(completion) is dict and tuple(completion) == completion_keys
                      and completion.get("schema") == "i15_completion_v1"
                      and completion.get("status") == "complete"
                      and completion.get("phase") == phase, phase + " completion differs")
        audit.require(manifest.get("frozen_commit") == completion.get("frozen_commit")
                      and manifest.get("source_hashes") == completion.get("source_hashes"),
                      phase + " source binding differs")
        audit.require(manifest.get("runtime_directory") == "runtime"
                      and manifest.get("runtime_bytes_count_in_cap") is True
                      and manifest.get("shared_artifact_cap_bytes") == ARTIFACT_CAP,
                      phase + " runtime/cap declaration differs")
        audit.require(manifest.get("policies") == list(NEW_POLICIES)
                      and tuple(manifest.get("horizons", ())) == HORIZONS
                      and manifest.get("timing_safety_factor") == 1.5
                      and manifest.get("cloud_spend_usd") == 0
                      and manifest.get("authorized_budget_usd") == 100
                      and manifest.get("wall_limit_seconds")
                      == (100 if phase == "smoke" else 1800),
                      phase + " fixed protocol manifest differs")
        expected_count = 6 if phase == "smoke" else 18
        audit.require(completion.get("branches") == expected_count,
                      phase + " branch count differs")
        records = completion.get("artifacts")
        audit.require(type(records) is list, phase + " artifact list missing")
        records = records if type(records) is list else []
        index = {row.get("name"): row for row in records if type(row) is dict}
        audit.require(len(index) == len(records), phase + " artifact names duplicate/malformed")
        actual = {path.name for path in directory.iterdir()}
        audit.require(actual == set(index) | {"completion.json"},
                      phase + " contains an unlisted, partial, or missing entry")
        for name, record in index.items():
            verify_record(directory, record, audit, f"{phase}/{name}")
        completion_path = directory / "completion.json"
        completion_info = completion_path.lstat()
        audit.require(stat.S_ISREG(completion_info.st_mode)
                      and not completion_path.is_symlink(),
                      phase + " completion is not a regular nonsymlink file")
        if stat.S_ISREG(completion_info.st_mode) and not completion_path.is_symlink():
            audit.phase_completion_sha256[phase] = sha256(completion_path)
            audit.hash_files += 1
            audit.hash_bytes += completion_info.st_size
        audit.require(type(completion.get("numerical_failures")) is int
                      and 0 <= completion["numerical_failures"] <= expected_count
                      and completion.get("all_requested_endpoints_present")
                      is (completion["numerical_failures"] == 0),
                      phase + " completion failure aggregate differs")
        for field in ("history_update_seconds", "elapsed_seconds"):
            try:
                audit.require(finite(completion.get(field), phase + " " + field) >= 0,
                              phase + " " + field + " is negative")
            except Exception as exc:
                audit.errors.append(str(exc))
        audit.require(type(completion.get("peak_torch_gpu_bytes")) is int
                      and completion["peak_torch_gpu_bytes"] >= 0
                      and type(completion.get("shared_artifact_bytes_before_completion")) is int
                      and 0 <= completion["shared_artifact_bytes_before_completion"]
                      <= ARTIFACT_CAP,
                      phase + " resource counters differ")
        return manifest, completion, index
    except Exception as exc:
        audit.errors.append(f"{phase}: phase loading failed: {type(exc).__name__}: {exc}")
        return {}, {}, {}


def runtime_inventory(root: Path, audit: Audit) -> dict[str, Any]:
    expected_root = set(PHASES) | {f"attempt-{phase}.json" for phase in PHASES} | {"runtime"}
    try:
        audit.require({path.name for path in root.iterdir()} == expected_root,
                      "artifact root membership differs")
        runtime = root / "runtime"
        info = runtime.lstat()
        audit.require(stat.S_ISDIR(info.st_mode) and not runtime.is_symlink(),
                      "runtime is not a real nonsymlink directory")
        entries = []
        for current, directories, files in os.walk(runtime, topdown=True, followlinks=False):
            current_path = Path(current)
            for name in sorted(directories):
                path = current_path / name
                row_info = path.lstat()
                audit.require(stat.S_ISDIR(row_info.st_mode) and not path.is_symlink(),
                              "runtime directory entry is not a real directory: " + str(path))
                entries.append({"path": str(path.relative_to(runtime)), "kind": "directory",
                                "bytes": 0, "sha256": None})
            for name in sorted(files):
                path = current_path / name
                row_info = path.lstat()
                audit.require(stat.S_ISREG(row_info.st_mode) and not path.is_symlink(),
                              "runtime file entry is not a regular nonsymlink file: " + str(path))
                if stat.S_ISREG(row_info.st_mode) and not path.is_symlink():
                    digest = sha256(path)
                    audit.hash_files += 1
                    audit.hash_bytes += row_info.st_size
                    entries.append({"path": str(path.relative_to(runtime)), "kind": "file",
                                    "bytes": row_info.st_size, "sha256": digest})
        entries.sort(key=lambda row: (row["path"], row["kind"]))
        canonical = json.dumps(entries, ensure_ascii=True, allow_nan=False,
                               separators=(",", ":")).encode("ascii")
        return {"declared_name": "runtime", "entries": entries,
                "manifest_sha256": hashlib.sha256(canonical).hexdigest(),
                "regular_file_bytes": sum(row["bytes"] for row in entries),
                "logical_bytes_only": True}
    except Exception as exc:
        audit.errors.append(f"runtime inventory failed: {type(exc).__name__}: {exc}")
        return {"declared_name": "runtime", "entries": None,
                "manifest_sha256": None, "regular_file_bytes": None,
                "logical_bytes_only": True}


def total_regular_bytes(root: Path, audit: Audit) -> int:
    total = 0
    try:
        for current, directories, files in os.walk(root, topdown=True, followlinks=False):
            base = Path(current)
            for name in directories:
                path = base / name
                info = path.lstat()
                audit.require(stat.S_ISDIR(info.st_mode) and not path.is_symlink(),
                              "artifact tree contains a symlink/non-directory: " + str(path))
            for name in files:
                path = base / name
                info = path.lstat()
                audit.require(stat.S_ISREG(info.st_mode) and not path.is_symlink(),
                              "artifact tree contains a symlink/non-file: " + str(path))
                if stat.S_ISREG(info.st_mode) and not path.is_symlink():
                    total += info.st_size
        audit.require(total <= ARTIFACT_CAP, "artifact root exceeds the shared 3GiB cap")
    except Exception as exc:
        audit.errors.append(f"artifact byte accounting failed: {type(exc).__name__}: {exc}")
    return total


def verify_sources(manifests: list[dict[str, Any]], audit: Audit) -> dict[str, Any]:
    commit = manifests[0].get("frozen_commit") if manifests else None
    sources = manifests[0].get("source_hashes") if manifests else None
    audit.require(commit == ACQUISITION_COMMIT, "acquisition commit differs from I15 freeze")
    audit.require(type(sources) is dict and tuple(sources) == ACQUISITION_SOURCES
                  and all(type(value) is str and SHA_RE.fullmatch(value) is not None
                          for value in sources.values()),
                  "acquisition source map differs from the fixed 12-file closure")
    for manifest in manifests[1:]:
        audit.require(manifest.get("frozen_commit") == commit
                      and manifest.get("source_hashes") == sources,
                      "acquisition phase sources differ")
    verified = []
    if type(commit) is str and type(sources) is dict:
        for relative, expected in sources.items():
            try:
                if not _safe_relative(relative):
                    raise ValueError("unsafe source path")
                path = REPO / relative
                audit.require(path.is_file() and sha256(path) == expected,
                              "worktree acquisition source differs: " + relative)
                frozen = subprocess.check_output(["git", "show", commit + ":" + relative],
                                                 cwd=REPO)
                audit.require(hashlib.sha256(frozen).hexdigest() == expected,
                              "committed acquisition source differs: " + relative)
                verified.append(relative)
            except Exception as exc:
                audit.errors.append(f"acquisition source failed {relative}: {type(exc).__name__}: {exc}")
    return {"frozen_commit": commit, "source_hashes": sources,
            "verified_files": verified}


def verify_analysis_sources(commit: str, audit: Audit) -> dict[str, Any]:
    audit.require(type(commit) is str and len(commit) == 40, "analysis commit malformed")
    hashes = {}
    for path in (Path(__file__).resolve(), HERE / "test_analyse_history.py"):
        relative = str(path.relative_to(REPO))
        current = sha256(path)
        try:
            frozen = subprocess.check_output(["git", "show", commit + ":" + relative], cwd=REPO)
            audit.require(hashlib.sha256(frozen).hexdigest() == current,
                          "analysis source is not frozen: " + relative)
        except Exception as exc:
            audit.errors.append(f"analysis source failed {relative}: {type(exc).__name__}: {exc}")
        hashes[relative] = current
    return {"frozen_commit": commit, "source_hashes": hashes}


def validate_evaluation(value: Any, audit: Audit, label: str) -> None:
    expected = {
        "train": ("clean_ce", "soft_ce", "clean_accuracy", "mean_max_probability",
                  "mean_true_label_probability", "fixed_ce", "fixed_accuracy",
                  "fixed_minus_soft_ce"),
        "validation": ("clean_ce", "soft_ce", "clean_accuracy", "mean_max_probability",
                       "mean_true_label_probability"),
        "auxiliary": ("clean_ce", "soft_ce", "clean_accuracy", "mean_max_probability",
                      "mean_true_label_probability"),
    }
    audit.require(type(value) is dict and tuple(value) == tuple(expected),
                  label + " evaluation split topology differs")
    if type(value) is not dict:
        return
    for split, keys in expected.items():
        row = value.get(split)
        audit.require(type(row) is dict and tuple(row) == keys,
                      f"{label} {split} metric topology differs")
        if type(row) is not dict:
            continue
        for name in keys:
            try:
                number = finite(row.get(name), f"{label} {split}.{name}")
                if name.endswith("_accuracy") or "probability" in name:
                    audit.require(0 <= number <= 1, f"{label} {split}.{name} outside [0,1]")
                elif name != "fixed_minus_soft_ce":
                    audit.require(number >= 0, f"{label} {split}.{name} is negative")
            except Exception as exc:
                audit.errors.append(str(exc))
        if split == "train" and all(type(row.get(name)) in (int, float)
                                    for name in ("fixed_ce", "soft_ce", "fixed_minus_soft_ce")):
            audit.require(math.isclose(row["fixed_minus_soft_ce"],
                                       row["fixed_ce"] - row["soft_ce"],
                                       rel_tol=2e-12, abs_tol=2e-12),
                          label + " fixed-minus-soft identity differs")


def validate_leakage(value: Any, expected_squared: float | None,
                     audit: Audit, label: str) -> tuple[float, float] | None:
    audit.require(type(value) is dict and tuple(value) == (
        "squared_norm", "outside_squared_norm", "fraction", "reason"),
        label + " leakage topology differs")
    if type(value) is not dict:
        return None
    try:
        squared = finite(value.get("squared_norm"), label + " squared")
        audit.require(squared >= 0, label + " squared energy is negative")
        if expected_squared is not None:
            error = abs(squared - expected_squared)
            audit.max_leakage_identity_error = max(audit.max_leakage_identity_error, error)
            audit.require(math.isclose(squared, expected_squared, rel_tol=2e-6, abs_tol=1e-14),
                          label + " norm/energy identity differs")
        reason = value.get("reason")
        if reason is not None:
            audit.require(reason in ("basis_unavailable", "zero_displacement")
                          and value.get("fraction") is None,
                          label + " missing-leakage reason differs")
            if reason == "zero_displacement":
                audit.require(squared == 0 and value.get("outside_squared_norm") == 0,
                              label + " zero leakage differs")
            return None
        outside = finite(value.get("outside_squared_norm"), label + " outside")
        fraction = finite(value.get("fraction"), label + " fraction")
        error = abs(outside - squared * fraction)
        audit.max_leakage_identity_error = max(audit.max_leakage_identity_error, error)
        # The native action is deliberately not assumed to be an exact
        # orthoprojector. Its x-Ax ratio can therefore exceed one.
        audit.require(outside >= 0 and fraction >= 0,
                      label + " leakage range differs")
        audit.require(math.isclose(outside, squared * fraction,
                                   rel_tol=2e-12, abs_tol=1e-18),
                      label + " leakage fraction identity differs")
        return outside, squared
    except Exception as exc:
        audit.errors.append(str(exc))
        return None


def validate_component(value: Any, audit: Audit, label: str) -> None:
    audit.require(type(value) is dict and tuple(value) == (
        "norm", "squared_energy", "current_basis_leakage"),
        label + " component topology differs")
    if type(value) is not dict:
        return
    try:
        norm = finite(value.get("norm"), label + " norm")
        energy = finite(value.get("squared_energy"), label + " energy")
        audit.require(norm >= 0 and energy >= 0
                      and math.isclose(energy, norm * norm, rel_tol=2e-6, abs_tol=1e-14),
                      label + " norm/energy differs")
        validate_leakage(value.get("current_basis_leakage"), energy, audit, label)
    except Exception as exc:
        audit.errors.append(str(exc))


def validate_step(row: Any, expected_step: int, policy: str,
                  audit: Audit, label: str) -> dict[str, float]:
    expected_keys = ("step", "relative_step", "schema", "base", "lr", "policy", "loss",
                     "observer", "gradient_filter_applied", "gradient", "components",
                     "displacement", "decay", "momentum_history", "algebra_residuals",
                     "digests")
    audit.require(type(row) is dict and tuple(row) == expected_keys,
                  label + " step topology differs")
    if type(row) is not dict:
        return {}
    audit.require(row.get("step") == expected_step
                  and row.get("relative_step") == expected_step - 100
                  and row.get("schema") == "i15_history_step_v1"
                  and row.get("base") == "sgdm" and row.get("lr") == .03
                  and row.get("policy") == policy, label + " step identity differs")
    observer = row.get("observer")
    audit.require(observer == {"step_before": expected_step - 1, "step_after": expected_step,
                               "filtering_active": True, "used": True,
                               "basis_rank": observer.get("basis_rank") if type(observer) is dict else None}
                  and type(observer.get("basis_rank")) is int
                  and 1 <= observer["basis_rank"] <= 32,
                  label + " observer identity differs")
    audit.require(row.get("gradient_filter_applied") is True,
                  label + " filter-applied flag differs")
    try:
        finite(row.get("loss"), label + " loss")
    except Exception as exc:
        audit.errors.append(str(exc))
    components = row.get("components")
    component_keys = ("raw_gradient", "native_projection", "post_mean",
                      "native_action_post_mean", "post_mean_complement",
                      "mean_delivery", "applied_delivery")
    audit.require(type(components) is dict and tuple(components) == component_keys,
                  label + " component membership differs")
    if type(components) is dict:
        for name in component_keys:
            validate_component(components.get(name), audit, label + " " + name)
    displacement = row.get("displacement")
    displacement_keys = ("total_norm", "data_norm", "nominal_decay_norm",
                         "total_current_basis_leakage", "data_current_basis_leakage",
                         "raw_gradient_dot_data_delta", "mean_complement_dot_data_delta",
                         "projected_old_history_dot_data_delta",
                         "removed_old_history_dot_data_delta")
    audit.require(type(displacement) is dict and tuple(displacement) == displacement_keys,
                  label + " displacement topology differs")
    signed = {}
    if type(displacement) is dict:
        norm_values: dict[str, float] = {}
        for name in ("total_norm", "data_norm", "nominal_decay_norm"):
            try:
                number = finite(displacement.get(name), label + " " + name)
                norm_values[name] = number
                audit.require(number >= 0, label + " " + name + " is negative")
            except Exception as exc:
                audit.errors.append(str(exc))
        validate_leakage(displacement.get("total_current_basis_leakage"),
                         (norm_values["total_norm"] ** 2
                          if "total_norm" in norm_values else None),
                         audit, label + " total displacement")
        validate_leakage(displacement.get("data_current_basis_leakage"),
                         (norm_values["data_norm"] ** 2
                          if "data_norm" in norm_values else None),
                         audit, label + " data displacement")
        for name in displacement_keys[5:]:
            try:
                signed[name] = finite(displacement.get(name), label + " " + name)
            except Exception as exc:
                audit.errors.append(str(exc))
    gradient = row.get("gradient")
    audit.require(type(gradient) is dict and tuple(gradient) == (
        "raw_norm", "applied_norm", "native_norm", "mean_complement_norm"),
        label + " gradient topology differs")
    if type(gradient) is dict and type(components) is dict:
        correspond = {"raw_norm": "raw_gradient", "applied_norm": "applied_delivery",
                      "native_norm": "native_projection",
                      "mean_complement_norm": "post_mean_complement"}
        for name, component in correspond.items():
            try:
                number = finite(gradient.get(name), label + " " + name)
                audit.require(number >= 0 and number == components[component]["norm"],
                              label + " gradient/component norm differs: " + name)
            except Exception as exc:
                audit.errors.append(str(exc))
    decay = row.get("decay")
    audit.require(type(decay) is dict and tuple(decay) == (
                      "coefficient", "factor", "manual_before_optimizer_step",
                      "manual_actual_norm", "manual_minus_nominal_norm")
                  and decay.get("coefficient") == .01
                  and decay.get("factor") == 1 - .03 * .01
                  and decay.get("manual_before_optimizer_step") is True,
                  label + " decay declaration differs")
    if type(decay) is dict:
        for name in ("manual_actual_norm", "manual_minus_nominal_norm"):
            try:
                number = finite(decay.get(name), label + " " + name)
                audit.require(number >= 0, label + " " + name + " is negative")
            except Exception as exc:
                audit.errors.append(str(exc))
    momentum = row.get("momentum_history")
    expected_history = policy.endswith("projected_history")
    momentum_keys = ("history_projected", "buffer_before", "native_action_old_buffer",
                     "removed_old_buffer", "selected_old_buffer", "buffer_after",
                     "expected_buffer_after_norm", "native_action_idempotence_defect",
                     "current_basis_orthogonality_error", "native_action_old_dot_removed",
                     "removed_old_dot_mean_complement",
                     "empirical_ideal_data_step_defect_norm")
    audit.require(type(momentum) is dict and tuple(momentum) == momentum_keys
                  and momentum.get("history_projected") is expected_history,
                  label + " momentum policy differs")
    if type(momentum) is dict:
        for name in ("buffer_before", "native_action_old_buffer", "removed_old_buffer",
                     "selected_old_buffer", "buffer_after"):
            validate_component(momentum.get(name), audit, label + " " + name)
        if expected_history:
            audit.require(momentum.get("selected_old_buffer")
                          == momentum.get("native_action_old_buffer"),
                          label + " projected selected-history scalar record differs")
        else:
            audit.require(momentum.get("selected_old_buffer") == momentum.get("buffer_before"),
                          label + " native selected-history scalar record differs")
        defect = momentum.get("native_action_idempotence_defect")
        audit.require(type(defect) is dict and tuple(defect) == (
            "norm", "homogeneous_roundoff_reference", "enforced")
            and defect.get("enforced") is False, label + " idempotence record differs")
        if type(defect) is dict:
            try:
                defect_norm = finite(defect.get("norm"), label + " action defect")
                defect_reference = finite(defect.get("homogeneous_roundoff_reference"),
                                          label + " action defect reference")
                audit.require(defect_norm >= 0 and defect_reference >= 0,
                              label + " action defect is negative")
            except Exception as exc:
                audit.errors.append(str(exc))
        for name in ("expected_buffer_after_norm", "current_basis_orthogonality_error",
                     "native_action_old_dot_removed", "removed_old_dot_mean_complement",
                     "empirical_ideal_data_step_defect_norm"):
            try:
                finite(momentum.get(name), label + " " + name)
            except Exception as exc:
                audit.errors.append(str(exc))
    residuals = row.get("algebra_residuals")
    expected_residuals = ("native_minus_actual_action_raw", "post_mean_recurrence",
                          "mean_delivery_definition", "mean_alternate_definition",
                          "momentum_buffer_recurrence")
    audit.require(type(residuals) is dict and tuple(residuals) == expected_residuals,
                  label + " algebra residual membership differs")
    if type(residuals) is dict:
        for name in expected_residuals:
            value = residuals.get(name)
            audit.require(type(value) is dict and tuple(value) == (
                "norm", "homogeneous_error_bound"), label + " residual topology differs")
            if type(value) is dict:
                try:
                    norm = finite(value.get("norm"), label + " residual norm")
                    bound = finite(value.get("homogeneous_error_bound"), label + " residual bound")
                    audit.max_algebra_residual = max(audit.max_algebra_residual, norm)
                    audit.require(0 <= norm <= bound and bound >= 0,
                                  label + " residual exceeds saved bound")
                except Exception as exc:
                    audit.errors.append(str(exc))
    digests = row.get("digests")
    if expected_step == 101:
        expected_digest_keys = ("raw_gradient", "post_observer", "applied_gradient",
                                "old_momentum_buffer", "native_action_old_buffer",
                                "new_momentum_buffer")
        audit.require(type(digests) is dict and tuple(digests) == expected_digest_keys
                      and all(type(value) is str and SHA_RE.fullmatch(value)
                              for value in digests.values()),
                      label + " first-step digest bundle differs")
    else:
        audit.require(digests is None, label + " unexpected repeated digest bundle")
    return signed


def _selection(curve: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "minimum_validation_ce": min(
            curve, key=lambda row: (row["validation"]["clean_ce"], row["horizon"]))["horizon"],
        "maximum_validation_accuracy": min(
            curve, key=lambda row: (-row["validation"]["clean_accuracy"], row["horizon"]))["horizon"],
    }


def validate_branch(branch: Any, phase: str, audit: Audit,
                    label: str) -> dict[str, Any] | None:
    if type(branch) is not dict:
        audit.errors.append(label + " branch is not an object")
        return None
    expected_updates = 10 if phase == "smoke" else 1900
    expected_horizons = (100, 110) if phase == "smoke" else HORIZONS
    expected_keys = ("schema", "id", "seed", "target", "policy", "base", "lr",
                     "parent_horizon", "parent_state_digest", "parent_evaluation_digest",
                     "status", "requested_updates", "completed_updates",
                     "last_completed_horizon", "curve", "history_update_seconds", "steps",
                     "checkpoints", "failure", "first_step_digests", "selected_horizons")
    audit.require(tuple(branch) == expected_keys, label + " branch topology differs")
    policy, seed, target = branch.get("policy"), branch.get("seed"), branch.get("target")
    audit.require(branch.get("schema") == "i15_history_branch_v1"
                  and branch.get("base") == "sgdm" and branch.get("lr") == .03
                  and policy in NEW_POLICIES and target in TARGETS
                  and type(seed) is int,
                  label + " branch identity differs")
    expected_seeds = (215,) if phase == "smoke" else SEEDS
    audit.require(seed in expected_seeds
                  and branch.get("id") == f"s{seed}-sgdm-{target}-{policy}"
                  and type(branch.get("parent_state_digest")) is str
                  and SHA_RE.fullmatch(branch["parent_state_digest"]) is not None
                  and type(branch.get("parent_evaluation_digest")) is str
                  and SHA_RE.fullmatch(branch["parent_evaluation_digest"]) is not None,
                  label + " branch ID/digest identity differs")
    status = branch.get("status")
    audit.require(status in ("complete", "numerical_failure"), label + " status differs")
    audit.require(branch.get("parent_horizon") == 100
                  and branch.get("requested_updates") == expected_updates,
                  label + " requested horizon differs")
    steps = branch.get("steps")
    audit.require(type(steps) is list and branch.get("completed_updates") == len(steps),
                  label + " completed-step count differs")
    steps = steps if type(steps) is list else []
    signed_steps = []
    for offset, row in enumerate(steps, start=101):
        signed_steps.append(validate_step(row, offset, policy, audit,
                                          f"{label} step {offset}"))
    curve = branch.get("curve")
    audit.require(type(curve) is list and bool(curve), label + " curve missing")
    curve = curve if type(curve) is list else []
    actual_horizons = [row.get("horizon") for row in curve if type(row) is dict]
    audit.require(actual_horizons
                  == list(expected_horizons[:len(actual_horizons)]) and bool(actual_horizons),
                  label + " curve is not a scheduled prefix")
    if status == "complete":
        audit.require(len(steps) == expected_updates and actual_horizons == list(expected_horizons)
                      and branch.get("completed_updates") == expected_updates
                      and branch.get("last_completed_horizon") == expected_horizons[-1]
                      and branch.get("failure") is None,
                      label + " complete coverage differs")
        audit.require(branch.get("selected_horizons") == _selection(curve),
                      label + " saved selector differs")
    else:
        failure = branch.get("failure")
        audit.require(type(failure) is dict and branch.get("selected_horizons") is None
                      and len(steps) <= expected_updates
                      and branch.get("last_completed_horizon") == 100 + len(steps),
                      label + " numerical failure topology differs")
        if type(failure) is dict:
            attempted, completed_step = (failure.get("attempted_step"),
                                         failure.get("completed_step"))
            valid_failure_position = (
                attempted == completed_step + 1
                or (attempted == completed_step
                    and completed_step in expected_horizons[1:]
                    and completed_step not in actual_horizons))
            audit.require(tuple(failure) == ("type", "message", "attempted_step",
                                             "completed_step", "last_valid_horizon",
                                             "state_artifact")
                          and failure.get("type") in ("NumericalFailure",)
                          and type(failure.get("message")) is str
                          and 0 < len(failure["message"]) <= 2000
                          and completed_step == 100 + len(steps)
                          and valid_failure_position
                          and failure.get("last_valid_horizon")
                          == (actual_horizons[-1] if actual_horizons else None),
                          label + " numerical failure record differs")
    for point in curve:
        if type(point) is dict:
            audit.require(tuple(point) == ("horizon", "train", "validation", "auxiliary"),
                          f"{label} h{point.get('horizon')} point topology differs")
            validate_evaluation({key: point.get(key) for key in ("train", "validation", "auxiliary")},
                                audit, f"{label} h{point.get('horizon')}")
    try:
        elapsed = finite(branch.get("history_update_seconds"), label + " update time")
        audit.require(elapsed >= 0, label + " update time is negative")
    except Exception as exc:
        audit.errors.append(str(exc))
    first = branch.get("first_step_digests")
    audit.require(first == (steps[0].get("digests") if steps else None),
                  label + " first-step digest copy differs")
    return {"seed": seed, "target": target, "policy": policy, "status": status,
            "curve": curve, "steps": steps, "signed_steps": signed_steps,
            "parent_state_digest": branch.get("parent_state_digest"),
            "parent_evaluation_digest": branch.get("parent_evaluation_digest"),
            "first_step_digests": first, "checkpoints": branch.get("checkpoints", []),
            "failure": branch.get("failure")}


def _point(branch: dict[str, Any] | None, horizon: int) -> dict[str, Any] | None:
    if branch is None or branch.get("status") != "complete":
        return None
    return next((row for row in branch["curve"] if row["horizon"] == horizon), None)


def _utility(point: dict[str, Any], metric: str) -> float:
    return (-point["auxiliary"]["clean_ce"] if metric == "ce"
            else point["auxiliary"]["clean_accuracy"])


def _family_value(branches: dict[tuple[int, str, str], dict[str, Any]], seed: int,
                  target: str, metric: str, family: str,
                  points: dict[str, dict[str, Any] | None]) -> float | None:
    required = {
        "H_history_under_current": ("current_native", "current_projected_history"),
        "M_mean_under_projected_history": (
            "mean_projected_history", "current_projected_history"),
        "S_mean_by_history_interaction": (
            "current_native", "current_projected_history", "mean_native",
            "mean_projected_history"),
    }
    if family not in required:
        raise ValueError(family)
    if any(points.get(name) is None for name in required[family]):
        return None
    u = {name: _utility(point, metric) for name, point in points.items() if point is not None}
    if family == "H_history_under_current":
        return u["current_native"] - u["current_projected_history"]
    if family == "M_mean_under_projected_history":
        return u["mean_projected_history"] - u["current_projected_history"]
    if family == "S_mean_by_history_interaction":
        return ((u["mean_projected_history"] - u["current_projected_history"])
                - (u["mean_native"] - u["current_native"]))
    raise AssertionError("unreachable primary family")


def _effect(values: dict[str, float | None]) -> dict[str, Any]:
    ordered = [values[str(seed)] for seed in SEEDS]
    available = all(value is not None for value in ordered)
    return {"per_seed": values, "all_three_seed_values": ordered,
            "available": available,
            "mean": sum(ordered) / 3 if available else None,
            "no_survivor_averaging": True}


def primary_effects(branches: dict[tuple[int, str, str], dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for family in PRIMARY_FAMILIES:
        for target in TARGETS:
            for metric in METRICS:
                values = {}
                for seed in SEEDS:
                    points = {policy: _point(branches.get((seed, target, policy)), 2000)
                              for policy in POLICIES}
                    values[str(seed)] = _family_value(branches, seed, target, metric,
                                                       family, points)
                result.append({"family": family, "target": target, "metric": metric,
                               "horizon": 2000, "effect": _effect(values)})
    return result


def complementary_effects(branches: dict[tuple[int, str, str], dict[str, Any]]) -> list[dict[str, Any]]:
    definitions = {
        "mean_under_native_history": ("mean_native", "current_native"),
        "native_history_under_mean_delivery": ("mean_native", "mean_projected_history"),
    }
    result = []
    for name, (positive, negative) in definitions.items():
        for target in TARGETS:
            for metric in METRICS:
                values = {}
                for seed in SEEDS:
                    left = _point(branches.get((seed, target, positive)), 2000)
                    right = _point(branches.get((seed, target, negative)), 2000)
                    values[str(seed)] = (None if left is None or right is None else
                                         _utility(left, metric) - _utility(right, metric))
                result.append({"contrast": name, "target": target, "metric": metric,
                               "horizon": 2000, "effect": _effect(values)})
    return result


def selector_results(branches: dict[tuple[int, str, str], dict[str, Any]]) \
        -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    outcomes = []
    chosen: dict[tuple[int, str, str, str], dict[str, Any] | None] = {}
    for seed in SEEDS:
        for target in TARGETS:
            for policy in POLICIES:
                branch = branches.get((seed, target, policy))
                for selector in SELECTORS:
                    point = None
                    if branch is not None and branch.get("status") == "complete":
                        horizon = _selection(branch["curve"])[selector]
                        point = _point(branch, horizon)
                    chosen[seed, target, policy, selector] = point
                    outcomes.append({"seed": seed, "target": target, "policy": policy,
                        "selector": selector,
                        "selected_horizon": None if point is None else point["horizon"],
                        "auxiliary": None if point is None else {
                            "clean_ce": point["auxiliary"]["clean_ce"],
                            "clean_accuracy": point["auxiliary"]["clean_accuracy"]}})
    effects = []
    for family in PRIMARY_FAMILIES:
        for target in TARGETS:
            for selector in SELECTORS:
                for metric in METRICS:
                    values = {}
                    for seed in SEEDS:
                        points = {policy: chosen[seed, target, policy, selector]
                                  for policy in POLICIES}
                        values[str(seed)] = _family_value(branches, seed, target, metric,
                                                           family, points)
                    effects.append({"family": family, "target": target,
                                    "selector": selector, "auxiliary_metric": metric,
                                    "effect": _effect(values)})
    return outcomes, effects


def trajectory_rows(branches: dict[tuple[int, str, str], dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for key in sorted(branches):
        branch = branches[key]
        h100 = next((row for row in branch["curve"] if row["horizon"] == 100), None)
        curve = []
        for point in branch["curve"]:
            values = {"train_clean_ce": point["train"]["clean_ce"],
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
                      "train_true_label_probability": point["train"]["mean_true_label_probability"]}
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
        result.append({"seed": key[0], "target": key[1], "policy": key[2],
                       "status": branch["status"], "failure": branch.get("failure"),
                       "curve": curve})
    return result


def scheduled_metric_aggregates(
        branches: dict[tuple[int, str, str], dict[str, Any]]) -> list[dict[str, Any]]:
    """Equal-seed scheduled cells and absolute h100 changes; no survivor means."""
    metrics = {
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
    rows = []
    for target in TARGETS:
        for policy in POLICIES:
            for horizon in HORIZONS:
                for metric, (split, field) in metrics.items():
                    values, changes = {}, {}
                    for seed in SEEDS:
                        point = _point(branches.get((seed, target, policy)), horizon)
                        start = _point(branches.get((seed, target, policy)), 100)
                        values[str(seed)] = (None if point is None else point[split][field])
                        changes[str(seed)] = (None if point is None or start is None else
                                              point[split][field] - start[split][field])
                    available = all(value is not None for value in values.values())
                    rows.append({"target": target, "policy": policy, "horizon": horizon,
                                 "metric": metric, "per_seed": values,
                                 "equal_seed_mean":
                                 (sum(values.values()) / 3 if available else None),
                                 "per_seed_change_from_h100": changes,
                                 "equal_seed_mean_change_from_h100":
                                 (sum(changes.values()) / 3 if available else None),
                                 "available": available,
                                 "no_survivor_averaging": True})
    return rows


def component_energy_results(
        branches: dict[tuple[int, str, str], dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate each new-arm component within branch, then equally over seeds."""
    locations = {
        **{name: ("components", name) for name in (
            "raw_gradient", "native_projection", "post_mean",
            "native_action_post_mean", "post_mean_complement", "mean_delivery",
            "applied_delivery")},
        **{"momentum_" + name: ("momentum_history", name) for name in (
            "buffer_before", "native_action_old_buffer", "removed_old_buffer",
            "selected_old_buffer", "buffer_after")},
    }
    windows = {"steps101_2000": (101, 2000), "steps1001_2000": (1001, 2000)}
    rows = []
    for target in TARGETS:
        for policy in NEW_POLICIES:
            for window, (low, high) in windows.items():
                for name, (container, field) in locations.items():
                    per_seed = {}
                    for seed in SEEDS:
                        branch = branches.get((seed, target, policy))
                        component_rows = ([row[container][field] for row in branch["steps"]
                                           if low <= row["step"] <= high]
                                          if branch is not None
                                          and branch.get("status") == "complete" else [])
                        total_energy = sum(row["squared_energy"] for row in component_rows)
                        leakage = [row["current_basis_leakage"] for row in component_rows]
                        ratio = None
                        if component_rows and total_energy > 0 \
                                and all(row.get("reason") in (None, "zero_displacement")
                                        for row in leakage):
                            ratio = sum(row["outside_squared_norm"] for row in leakage) / total_energy
                        per_seed[str(seed)] = (None if not component_rows else {
                            "mean_squared_energy": total_energy / len(component_rows),
                            "outside_energy_fraction": ratio})
                    available_energy = all(value is not None for value in per_seed.values())
                    available_ratio = available_energy and all(
                        value["outside_energy_fraction"] is not None
                        for value in per_seed.values())
                    rows.append({"target": target, "policy": policy, "window": window,
                                 "component": name, "per_seed": per_seed,
                                 "equal_seed_mean_squared_energy":
                                 (sum(value["mean_squared_energy"]
                                      for value in per_seed.values()) / 3
                                  if available_energy else None),
                                 "equal_seed_mean_of_branch_outside_energy_fractions":
                                 (sum(value["outside_energy_fraction"]
                                      for value in per_seed.values()) / 3
                                  if available_ratio else None),
                                 "energy_available": available_energy,
                                 "outside_fraction_available": available_ratio,
                                 "aggregation": "energy ratio within branch, then equal seeds"})
    return rows


def geometry_results(branches: dict[tuple[int, str, str], dict[str, Any]]) \
        -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    leakage_rows, signed_rows = [], []
    windows = {"steps101_2000": (101, 2000), "steps1001_2000": (1001, 2000)}
    signed_names = ("raw_gradient_dot_data_delta", "mean_complement_dot_data_delta",
                    "projected_old_history_dot_data_delta",
                    "removed_old_history_dot_data_delta")
    for target in TARGETS:
        for policy in POLICIES:
            for window, (low, high) in windows.items():
                for displacement in ("data", "total"):
                    per_seed = {}
                    for seed in SEEDS:
                        branch = branches.get((seed, target, policy))
                        fraction = None
                        if branch is not None and branch.get("status") == "complete":
                            pairs = []
                            name = displacement + "_current_basis_leakage"
                            for row in branch["steps"]:
                                step = row["step"]
                                if low <= step <= high:
                                    value = row["displacement"].get(name)
                                    if type(value) is dict and value.get("reason") is None:
                                        pairs.append((value["outside_squared_norm"],
                                                      value["squared_norm"]))
                            if pairs and sum(item[1] for item in pairs) > 0:
                                fraction = sum(item[0] for item in pairs) / sum(
                                    item[1] for item in pairs)
                        per_seed[str(seed)] = fraction
                    available = all(value is not None for value in per_seed.values())
                    leakage_rows.append({"target": target, "policy": policy,
                        "window": window, "displacement": displacement,
                        "per_seed_branch_fractions": per_seed,
                        "equal_seed_mean": (sum(per_seed.values()) / 3 if available else None),
                        "available": available,
                        "aggregation": "energy ratio within branch, then equal seeds"})
                for metric in signed_names:
                    per_seed = {}
                    for seed in SEEDS:
                        branch = branches.get((seed, target, policy))
                        values = []
                        if branch is not None and branch.get("status") == "complete":
                            for row in branch["steps"]:
                                if low <= row["step"] <= high:
                                    value = row["displacement"].get(metric)
                                    if type(value) in (int, float) and math.isfinite(value):
                                        values.append(float(value))
                        per_seed[str(seed)] = sum(values) / len(values) if values else None
                    available = all(value is not None for value in per_seed.values())
                    signed_rows.append({"target": target, "policy": policy,
                        "window": window, "metric": metric, "per_seed_branch_means": per_seed,
                        "equal_seed_mean": (sum(per_seed.values()) / 3 if available else None),
                        "available": available,
                        "aggregation": "arithmetic mean within branch, then equal seeds"})
    return leakage_rows, signed_rows


def numerical_action_results(branches: dict[tuple[int, str, str], dict[str, Any]]) \
        -> list[dict[str, Any]]:
    """Summarize implementation defects separately from substantive geometry."""
    result = []
    windows = {"steps101_2000": (101, 2000), "steps1001_2000": (1001, 2000)}
    extractors = {
        "current_basis_orthogonality_error": lambda row: row["momentum_history"][
            "current_basis_orthogonality_error"],
        "native_action_idempotence_defect_norm": lambda row: row["momentum_history"][
            "native_action_idempotence_defect"]["norm"],
        "native_action_idempotence_roundoff_reference": lambda row: row[
            "momentum_history"]["native_action_idempotence_defect"][
                "homogeneous_roundoff_reference"],
        "empirical_ideal_data_step_defect_norm": lambda row: row["momentum_history"][
            "empirical_ideal_data_step_defect_norm"],
        "manual_minus_nominal_decay_norm": lambda row: row["decay"][
            "manual_minus_nominal_norm"],
    }
    for target in TARGETS:
        for policy in NEW_POLICIES:
            for window, (low, high) in windows.items():
                for metric, getter in extractors.items():
                    per_seed = {}
                    for seed in SEEDS:
                        branch = branches.get((seed, target, policy))
                        values = ([float(getter(row)) for row in branch["steps"]
                                   if low <= row["step"] <= high]
                                  if branch is not None
                                  and branch.get("status") == "complete" else [])
                        per_seed[str(seed)] = ({"mean": sum(values) / len(values),
                                                "maximum": max(values)} if values else None)
                    available = all(value is not None for value in per_seed.values())
                    result.append({"target": target, "policy": policy, "window": window,
                                   "metric": metric, "per_seed": per_seed,
                                   "equal_seed_mean_of_branch_means":
                                   (sum(value["mean"] for value in per_seed.values()) / 3
                                    if available else None),
                                   "maximum_over_available_branches":
                                   (max(value["maximum"] for value in per_seed.values()
                                        if value is not None)
                                    if any(value is not None for value in per_seed.values())
                                    else None),
                                   "available": available})
    return result


def _load_state(path: Path, expected_digest: str | None, audit: Audit,
                label: str) -> str | None:
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
        digest = tree_digest(value)
        audit.tree_digests += 1
        if expected_digest is not None:
            audit.require(type(expected_digest) is str and SHA_RE.fullmatch(expected_digest)
                          is not None and digest == expected_digest,
                          label + " complete-state digest differs")
        return digest
    except Exception as exc:
        audit.errors.append(f"{label}: complete-state load/digest failed: {type(exc).__name__}: {exc}")
        return None


def load_i14_references(binding: Any, audit: Audit) \
        -> dict[tuple[int, str, str], dict[str, Any]]:
    result = {}
    try:
        binding_keys = ("schema", "source_root", "i14_commit",
                        "runtime_supplement_sha256", "confirmation_completion_sha256",
                        "source_records", "dataset_inputs", "corruption_counts", "parents",
                        "references", "source_replayed")
        audit.require(type(binding) is dict and tuple(binding) == binding_keys
                      and binding.get("schema") == "i15_i14_inputs_v1"
                      and binding.get("source_root") == str(I14_ROOT)
                      and binding.get("i14_commit")
                      == "d48f20a76f32d5b717dd797678f276a16ec57f22"
                      and binding.get("runtime_supplement_sha256") == I14_SUPPLEMENT_SHA
                      and binding.get("source_replayed") is False,
                      "I14 input binding identity differs")
        supplement_path = I14 / "analysis-001/runtime-directory-supplement.json"
        audit.require(sha256(supplement_path) == I14_SUPPLEMENT_SHA,
                      "accepted I14 supplement differs")
        supplement = read_json(supplement_path)
        completion_path = I14_ROOT / "confirmation/completion.json"
        audit.require(sha256(completion_path) == binding.get("confirmation_completion_sha256"),
                      "I14 completion binding differs")
        for name, key in (("audit.json", "original_audit_sha256"),
                          ("summary.json", "original_summary_sha256")):
            audit.require(sha256(I14 / "analysis-001" / name) == supplement[key],
                          "original I14 analysis evidence differs: " + name)
        records = binding.get("source_records")
        audit.require(type(records) is dict and bool(records), "I14 source record map missing")
        records = records if type(records) is dict else {}
        for name, record in records.items():
            audit.require(_safe_relative(name) and Path(name).name == name
                          and type(record) is dict and record.get("name") == name,
                          "I14 source record key differs: " + str(name))
            verify_record(I14_ROOT / "confirmation", record, audit, "I14/" + str(name))
        dataset_inputs = binding.get("dataset_inputs")
        audit.require(type(dataset_inputs) is dict and tuple(dataset_inputs) == (
            "train-images-idx3-ubyte", "train-labels-idx1-ubyte")
            and all(type(value) is str and SHA_RE.fullmatch(value) is not None
                    for value in dataset_inputs.values()),
            "I14 dataset-input binding differs")
        for name, expected in dataset_inputs.items() if type(dataset_inputs) is dict else []:
            verify_file(DATA_ROOT / name, (DATA_ROOT / name).stat().st_size, expected, audit,
                        "dataset input " + name)
        references = binding.get("references")
        audit.require(type(references) is list and len(references) == 12,
                      "I14 raw/current reference count differs")
        for reference in references if type(references) is list else []:
            audit.require(tuple(reference) == ("seed", "target", "policy", "curve_artifact",
                                                "checkpoint_records", "warmup_state_digest"),
                          "I14 reference topology differs")
            seed, target, old_policy = (reference.get("seed"), reference.get("target"),
                                        reference.get("policy"))
            audit.require(seed in SEEDS and target in TARGETS
                          and old_policy in ("raw", "current32"),
                          "I14 reference identity differs")
            record = reference.get("curve_artifact")
            path = verify_record(I14_ROOT / "confirmation", record, audit,
                                 f"I14 curve {seed}/{target}/{old_policy}")
            if path is None:
                continue
            curve = read_json(path)
            audit.require(curve.get("seed") == seed and curve.get("target") == target
                          and curve.get("policy") == old_policy and curve.get("base") == "sgdm"
                          and curve.get("lr") == .03 and curve.get("status") == "complete"
                          and curve.get("completed_steps") == 2000,
                          "I14 reference curve identity differs")
            audit.require(reference.get("checkpoint_records") == curve.get("checkpoints")
                          and reference.get("warmup_state_digest")
                          == curve.get("warmup_full_state_digest"),
                          "I14 reference checkpoint binding differs")
            checkpoints = curve.get("checkpoints")
            audit.require(type(checkpoints) is list
                          and [row.get("horizon") for row in checkpoints
                               if type(row) is dict] == [0, *HORIZONS],
                          "I14 reference checkpoint horizons differ")
            for checkpoint in curve.get("checkpoints", []):
                if type(checkpoint) is not dict:
                    audit.errors.append("I14 malformed checkpoint record")
                    continue
                kind = ("full_state" if checkpoint.get("horizon") in (0, 100, 2000)
                        else "model_state")
                expected_keys = (("horizon", "full_state", "full_state_digest")
                                 if kind == "full_state" else ("horizon", "model_state"))
                audit.require(tuple(checkpoint) == expected_keys,
                              f"I14 checkpoint topology differs {seed}/{target}/"
                              f"{old_policy}/h{checkpoint.get('horizon')}")
                record = checkpoint.get(kind)
                verify_record(I14_ROOT / "confirmation", record, audit,
                              f"I14 checkpoint {seed}/{target}/{old_policy}/h"
                              f"{checkpoint.get('horizon')}")
            points = [row for row in curve.get("curve", []) if row.get("horizon") in HORIZONS]
            audit.require([row.get("horizon") for row in points] == list(HORIZONS),
                          "I14 reference horizon coverage differs")
            for point in points:
                audit.require(type(point) is dict and tuple(point) == (
                    "horizon", "train", "validation", "auxiliary"),
                    f"I14 reference point topology differs {seed}/{target}/{old_policy}")
                validate_evaluation({key: point.get(key) for key in (
                    "train", "validation", "auxiliary")}, audit,
                    f"I14 reference {seed}/{target}/{old_policy}/h{point.get('horizon')}")
            result[seed, target, "raw" if old_policy == "raw" else "current_native"] = {
                "seed": seed, "target": target,
                "policy": "raw" if old_policy == "raw" else "current_native",
                "status": "complete", "curve": points,
                "steps": [row for row in curve.get("steps", []) if row.get("step", 0) >= 101],
                "failure": None, "source": "accepted_i14_reference"}
        parents = binding.get("parents")
        audit.require(type(parents) is list and len(parents) == 6,
                      "I14 used-parent count differs")
        for row in parents if type(parents) is list else []:
            audit.require(tuple(row) == ("seed", "target", "path", "artifact", "state_digest")
                          and row.get("seed") in SEEDS and row.get("target") in TARGETS,
                          "I14 parent topology differs")
            path = Path(row.get("path", ""))
            audit.require(path.parent == I14_ROOT / "confirmation"
                          and row.get("artifact") == records.get(path.name),
                          "I14 parent artifact binding differs")
            if path.parent == I14_ROOT / "confirmation" and path.name in records:
                _load_state(path, row.get("state_digest"), audit,
                            f"I14 parent {row.get('seed')}/{row.get('target')}")
    except Exception as exc:
        audit.errors.append(f"I14 reference loading failed: {type(exc).__name__}: {exc}")
    audit.require(set(result) == {(seed, target, policy) for seed in SEEDS
                                  for target in TARGETS
                                  for policy in ("raw", "current_native")},
                  "I14 reference membership differs")
    for seed in SEEDS:
        for target in TARGETS:
            raw = result.get((seed, target, "raw"))
            current = result.get((seed, target, "current_native"))
            if raw is not None and current is not None:
                audit.require(raw["curve"][0] == current["curve"][0],
                              f"I14 h100 raw/current seam differs for {seed}/{target}")
    return result


def _validate_new_checkpoints(branch: dict[str, Any], index: dict[str, Any], directory: Path,
                              audit: Audit, label: str) -> None:
    checkpoints = branch.get("checkpoints")
    if type(checkpoints) is not list:
        audit.errors.append(label + " checkpoint list missing")
        return
    expected_horizons = [row.get("horizon") for row in branch.get("curve", [])[1:]
                         if type(row) is dict]
    audit.require([row.get("horizon") for row in checkpoints if type(row) is dict]
                  == expected_horizons, label + " checkpoint/curve horizon coverage differs")
    for row in checkpoints:
        if type(row) is not dict:
            audit.errors.append(label + " malformed checkpoint")
            continue
        horizon = row.get("horizon")
        if horizon == branch.get("last_completed_horizon") and branch.get("status") == "complete":
            audit.require(tuple(row) == ("horizon", "relative_horizon", "full_state",
                                         "full_state_digest"),
                          label + " terminal checkpoint topology differs")
            record = row.get("full_state")
            audit.require(record == index.get(record.get("name") if type(record) is dict else None),
                          label + " terminal state membership differs")
            path = verify_record(directory, record, audit, label + " terminal state")
            if path is not None:
                _load_state(path, row.get("full_state_digest"), audit, label + " terminal state")
        else:
            audit.require(tuple(row) == ("horizon", "relative_horizon", "model_state"),
                          label + " intermediate checkpoint topology differs")
            record = row.get("model_state")
            audit.require(record == index.get(record.get("name") if type(record) is dict else None),
                          label + " intermediate model membership differs")
            verify_record(directory, record, audit, label + " intermediate model")
            # Model-only states have no saved semantic digest. Their completion-pinned
            # file hashes are checked, but they are not independently replayable states.
        audit.require(row.get("relative_horizon") == horizon - 100
                      if type(horizon) is int else False,
                      label + " relative checkpoint horizon differs")
    failure = branch.get("failure")
    if branch.get("status") == "numerical_failure" and type(failure) is dict:
        record = failure.get("state_artifact")
        audit.require(record == index.get(record.get("name") if type(record) is dict else None),
                      label + " failure-state membership differs")
        path = verify_record(directory, record, audit, label + " failure state")
        if path is not None:
            digest = _load_state(path, None, audit, label + " failure state")
            if digest is not None:
                audit.failure_state_tree_digests[label] = digest


def analyze(root: Path, analysis_commit: str, audit: Audit) -> dict[str, Any]:
    runtime = runtime_inventory(root, audit)
    audit.artifact_root = str(root)
    audit.runtime_inventory = runtime
    total_bytes = total_regular_bytes(root, audit)
    loaded = {phase: phase_records(root, phase, audit) for phase in PHASES}
    manifests = [loaded[phase][0] for phase in PHASES]
    acquisition = verify_sources(manifests, audit)
    analysis = verify_analysis_sources(analysis_commit, audit)
    for phase in PHASES:
        attempt_path = root / f"attempt-{phase}.json"
        info = attempt_path.lstat()
        audit.require(stat.S_ISREG(info.st_mode) and not attempt_path.is_symlink(),
                      phase + " attempt is not a regular nonsymlink file")
        if stat.S_ISREG(info.st_mode) and not attempt_path.is_symlink():
            audit.attempt_sha256[phase] = sha256(attempt_path)
            audit.hash_files += 1
            audit.hash_bytes += info.st_size
        attempt = read_json(attempt_path)
        audit.require(tuple(attempt) == ("schema", "phase", "frozen_commit", "source_hashes",
                                         "pid", "restart")
                      and attempt.get("schema") == "i15_attempt_v1"
                      and attempt.get("phase") == phase
                      and attempt.get("frozen_commit") == acquisition["frozen_commit"]
                      and attempt.get("source_hashes") == acquisition["source_hashes"]
                      and type(attempt.get("pid")) is int and attempt["pid"] > 0
                      and attempt.get("restart") == "forbidden",
                      phase + " attempt binding differs")
    confirmation_dir = root / "confirmation"
    confirmation_index = loaded["confirmation"][2]
    parent_record = confirmation_index.get("parent-inputs.json")
    parent_path = verify_record(confirmation_dir, parent_record, audit, "parent inputs")
    binding = read_json(parent_path) if parent_path is not None else {}
    branches = load_i14_references(binding, audit)
    parent_by_key = {(row.get("seed"), row.get("target")): row
                     for row in binding.get("parents", []) if type(row) is dict}
    seams_record = confirmation_index.get("parent-seams.json")
    seams_path = verify_record(confirmation_dir, seams_record, audit, "parent seams")
    seams = read_json(seams_path) if seams_path is not None else {}
    audit.require(type(seams) is dict and tuple(seams) == (
        "parents", "scientific_updates_before_admission")
        and seams.get("scientific_updates_before_admission") == 0
        and type(seams.get("parents")) is list and len(seams["parents"]) == 6,
        "parent-seam envelope differs")
    seam_by_key = {}
    for row in seams.get("parents", []) if type(seams) is dict else []:
        audit.require(type(row) is dict and tuple(row) == (
            "seed", "target", "state_digest", "evaluation_digest", "status")
            and row.get("seed") in SEEDS and row.get("target") in TARGETS
            and row.get("status") == "pass", "parent-seam row differs")
        seam_by_key[row.get("seed"), row.get("target")] = row
    audit.require(set(seam_by_key) == {(seed, target) for seed in SEEDS for target in TARGETS},
                  "parent-seam membership differs")
    plan_bindings = []
    for seed in SEEDS:
        plan_name, corruption_name = f"plan-s{seed}.json", f"corruption-s{seed}.json"
        plan_path = verify_record(confirmation_dir, confirmation_index.get(plan_name), audit,
                                  "I15 " + plan_name)
        corruption_path = verify_record(confirmation_dir, confirmation_index.get(corruption_name),
                                        audit, "I15 " + corruption_name)
        old_plan = binding.get("source_records", {}).get(f"plan-confirmation-s{seed}.json")
        if plan_path is not None and type(old_plan) is dict:
            old_path = verify_record(I14_ROOT / "confirmation", old_plan, audit,
                                     "I14 plan " + str(seed))
            if old_path is not None:
                audit.require(read_json(plan_path) == read_json(old_path),
                              f"seed {seed} plan content differs from bound I14 plan")
                plan_bindings.append({"seed": seed, "i15_sha256": sha256(plan_path),
                                      "i14_sha256": sha256(old_path),
                                      "exact_json_tree_equal":
                                      read_json(plan_path) == read_json(old_path)})
        if corruption_path is not None:
            corruption = read_json(corruption_path)
            audit.require(type(corruption) is dict and tuple(corruption) == (
                "replaced_count", "incorrect_count", "train_count")
                and all(type(corruption.get(name)) is int for name in corruption)
                and 0 <= corruption["incorrect_count"] <= corruption["replaced_count"]
                <= corruption["train_count"] == 5000
                and corruption == binding.get("corruption_counts", {}).get(str(seed)),
                          f"seed {seed} corruption counts differ")
    for phase in PHASES:
        directory, index = root / phase, loaded[phase][2]
        branches_record = index.get("branches.json")
        branches_path = verify_record(directory, branches_record, audit, phase + " branch index")
        if branches_path is None:
            continue
        branch_index = read_json(branches_path)
        expected_index_keys = (("entries", "first_step_pair_checks", "synthetic_warmup_updates",
                                "new_branch_updates") if phase == "smoke" else
                               ("entries", "first_step_pair_checks", "new_branches",
                                "reused_native_current", "reused_raw"))
        audit.require(type(branch_index) is dict and tuple(branch_index) == expected_index_keys,
                      phase + " branch-index topology differs")
        if phase == "smoke":
            audit.require(branch_index.get("synthetic_warmup_updates") == 200
                          and branch_index.get("new_branch_updates") == 60,
                          "smoke fixed update declarations differ")
        else:
            audit.require(branch_index.get("new_branches") == 18
                          and branch_index.get("reused_native_current") == 6
                          and branch_index.get("reused_raw") == 6,
                          "confirmation reuse declarations differ")
        entries = branch_index.get("entries")
        expected_members = ({(215, target, policy) for target in TARGETS for policy in NEW_POLICIES}
                            if phase == "smoke" else
                            {(seed, target, policy) for seed in SEEDS for target in TARGETS
                             for policy in NEW_POLICIES})
        audit.require(type(entries) is list and len(entries) == len(expected_members),
                      phase + " branch index count differs")
        observed = set()
        group_rows = []
        for entry in entries if type(entries) is list else []:
            audit.require(tuple(entry) == ("id", "seed", "target", "policy", "status",
                                           "completed_updates", "parent_state_digest",
                                           "parent_evaluation_digest", "first_step_digests",
                                           "history_update_seconds", "artifact"),
                          phase + " branch-index entry topology differs")
            key = (entry.get("seed"), entry.get("target"), entry.get("policy"))
            observed.add(key)
            artifact = entry.get("artifact")
            audit.require(type(artifact) is dict
                          and artifact == index.get(artifact.get("name")),
                          phase + " branch artifact membership differs")
            path = verify_record(directory, artifact, audit, phase + " branch " + str(key))
            if path is None:
                continue
            branch = read_json(path)
            audit.require(all(branch.get(name) == entry.get(name) for name in (
                "id", "seed", "target", "policy", "status", "completed_updates",
                "parent_state_digest", "parent_evaluation_digest", "first_step_digests")),
                phase + " branch index identity differs: " + str(key))
            validated = validate_branch(branch, phase, audit, phase + "/" + str(key))
            if validated is not None:
                group_rows.append(validated)
                if phase == "confirmation":
                    audit.require(key not in branches, "duplicate confirmation branch key")
                    branches[key] = validated
                _validate_new_checkpoints(branch, index, directory, audit,
                                          phase + "/" + str(key))
        audit.require(observed == expected_members, phase + " exact branch membership differs")
        completed_sum = sum(row.get("completed_updates", 0) for row in entries
                            if type(row) is dict and type(row.get("completed_updates")) is int)
        failures = sum(row.get("status") == "numerical_failure" for row in entries
                       if type(row) is dict)
        completion = loaded[phase][1]
        audit.require(completion.get("completed_training_updates")
                      == completed_sum + (200 if phase == "smoke" else 0),
                      phase + " completion update accounting differs")
        audit.require(completion.get("numerical_failures") == failures,
                      phase + " completion failure count differs")
        recomputed_pairs = []
        for seed, target in sorted({(row["seed"], row["target"]) for row in group_rows}):
            group = [row for row in group_rows
                     if (row["seed"], row["target"]) == (seed, target)]
            audit.require(len(group) == 3 and {row["policy"] for row in group}
                          == set(NEW_POLICIES), phase + " first-step group differs")
            common = ("parent_state_digest", "parent_evaluation_digest")
            for name in common:
                audit.require(len({row[name] for row in group}) == 1,
                              phase + " parent seam differs: " + name)
            available = [row for row in group if type(row["first_step_digests"]) is dict]
            for name in ("raw_gradient", "post_observer", "old_momentum_buffer",
                         "native_action_old_buffer"):
                audit.require(len({row["first_step_digests"][name] for row in available}) <= 1,
                              phase + " first-step common digest differs: " + name)
            means = [row for row in available if row["policy"].startswith("mean_")]
            audit.require(len({row["first_step_digests"]["applied_gradient"]
                               for row in means}) <= 1,
                          phase + " first-step mean delivery differs")
            recomputed_pairs.append({"seed": seed, "target": target,
                "available_first_steps": len(available),
                "status": "pass" if len(available) == 3 else "partial_numerical_evidence"})
            if phase == "confirmation":
                parent = parent_by_key.get((seed, target), {})
                seam = seam_by_key.get((seed, target), {})
                current = branches.get((seed, target, "current_native"))
                current_h100 = _point(current, 100)
                expected_evaluation_digest = (tree_digest(current_h100)
                                              if current_h100 is not None else None)
                for row in group:
                    audit.require(row["parent_state_digest"] == parent.get("state_digest")
                                  == seam.get("state_digest"),
                                  f"confirmation parent state does not bind I14 {seed}/{target}")
                    audit.require(row["parent_evaluation_digest"]
                                  == seam.get("evaluation_digest")
                                  == expected_evaluation_digest,
                                  f"confirmation parent evaluation does not bind I14 {seed}/{target}")
                    audit.require(row["curve"] and row["curve"][0] == current_h100,
                                  f"confirmation h100 curve seam differs {seed}/{target}")
        audit.require(branch_index.get("first_step_pair_checks") == recomputed_pairs,
                      phase + " saved first-step pair checks differ")
    expected_all = {(seed, target, policy) for seed in SEEDS for target in TARGETS
                    for policy in POLICIES}
    audit.require(set(branches) == expected_all, "complete five-policy analysis membership differs")
    selectors, selected_effects = selector_results(branches)
    leakage, signed = geometry_results(branches)
    return {
        "schema": "i15_history_analysis_summary_v1",
        "artifact_root": str(root),
        "scope": [
            "I14 raw/current-native curves are accepted hash-bound references and were not rerun.",
            "New complete states were CPU tree-digested; intermediate model-only states were hash-checked only.",
            "A missing branch contribution makes an estimand unavailable; no survivor averaging.",
            "Signed alignment and complement-energy summaries are descriptive, not mediation fractions.",
        ],
        "counts": {"new_smoke_branches": loaded["smoke"][1].get("branches"),
                   "new_confirmation_branches": loaded["confirmation"][1].get("branches"),
                   "confirmation_numerical_failures": loaded["confirmation"][1].get(
                       "numerical_failures")},
        "input_provenance": {"acquisition": acquisition, "analysis": analysis,
                             "i14_binding": binding,
                             "plan_bindings": plan_bindings,
                             "runtime_inventory": runtime,
                             "artifact_root_regular_file_bytes": total_bytes,
                             "artifact_cap_bytes": ARTIFACT_CAP},
        "primary_endpoint_effects": primary_effects(branches),
        "complementary_endpoint_effects": complementary_effects(branches),
        "all_policy_trajectories": trajectory_rows(branches),
        "scheduled_metric_equal_seed_aggregates": scheduled_metric_aggregates(branches),
        "validation_selected_auxiliary_outcomes": selectors,
        "validation_selected_primary_effects": selected_effects,
        "energy_weighted_current_action_complement": leakage,
        "component_energy_and_complement": component_energy_results(branches),
        "signed_step_component_means": signed,
        "numerical_action_diagnostics": numerical_action_results(branches),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--frozen-commit", required=True)
    args = parser.parse_args()
    audit = Audit()
    try:
        unresolved_root = args.root.absolute()
        root = args.root.resolve(strict=True)
        audit.require(not unresolved_root.is_symlink() and unresolved_root == root,
                      "artifact root must be the direct canonical nonsymlink path")
    except Exception as exc:
        raise SystemExit(f"invalid artifact root: {exc}")
    output = args.output
    output.parent.resolve(strict=True)
    if output.parent.resolve() != HERE or not output.name.startswith("analysis-"):
        raise SystemExit("output must be a new analysis-* directory in iteration-015")
    try:
        output.resolve().relative_to(root)
        raise SystemExit("analysis output must not be inside the artifact root")
    except ValueError:
        pass
    output.mkdir(exist_ok=False)
    try:
        summary = analyze(root, args.frozen_commit, audit)
    except BaseException as exc:
        audit.errors.append(f"analysis aborted: {type(exc).__name__}: {exc}")
        summary = {"schema": "i15_history_analysis_summary_v1", "artifact_root": str(root),
                   "status": "unavailable_due_to_analysis_error"}
    summary["audit_status"] = "pass" if not audit.errors else "fail"
    audit_payload = {
        "schema": "i15_history_analysis_audit_v1",
        "status": "pass" if not audit.errors else "fail",
        "artifact_root": audit.artifact_root,
        "phase_completion_sha256": audit.phase_completion_sha256,
        "attempt_sha256": audit.attempt_sha256,
        "runtime_inventory": audit.runtime_inventory,
        "checks": audit.checks, "errors": audit.errors, "warnings": audit.warnings,
        "hash_files_verified": audit.hash_files, "hash_bytes_streamed": audit.hash_bytes,
        "complete_state_tree_digests_verified": audit.tree_digests,
        "maximum_saved_algebra_residual": audit.max_algebra_residual,
        "maximum_leakage_identity_error": audit.max_leakage_identity_error,
        "failure_state_tree_digests": audit.failure_state_tree_digests,
    }
    for name, value in (("summary.json", summary), ("audit.json", audit_payload)):
        with (output / name).open("x", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, ensure_ascii=True, allow_nan=False)
            handle.write("\n")
    print(json.dumps({"status": audit_payload["status"], "checks": audit.checks,
                      "errors": len(audit.errors), "hash_files": audit.hash_files,
                      "tree_digests": audit.tree_digests}, allow_nan=False))
    return 0 if not audit.errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
