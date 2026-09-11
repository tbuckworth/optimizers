#!/usr/bin/env python3
"""Analyze the complete saved representation corpus without model inference."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Iterable

# Pin common CPU math backends before NumPy is imported in direct execution.
for _thread_variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                         "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_thread_variable] = "1"

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from experiments.grokking_confirmation import atomic_exclusive, file_hash  # noqa: E402
from experiments.grokking_representation import (  # noqa: E402
    make_probe_split, make_row_permutations,
)
from experiments.grokking_symmetry import (  # noqa: E402
    DEFAULT_DELTAS, symmetry_diagnostics,
)

ARMS = ("adamw", "legacy", "stable")
SEEDS = tuple(range(100, 105))
STEPS = (0, 100, 500, 1000, 1500, 2000, 2500, 3000, 4000, 6000)
RECIPE = {"ridge": .001, "top_k": 5, "n_nulls": 20, "p": 113,
          "split_seed_offset": 20260909, "null_seed_offset": 20261909,
          "batch_size": 1024, "threads": 1,
          "features": ["final_hidden", "pre_attention"],
          "max_output_bytes": 10 * 1024**3, "cooperative_seconds": 1740}
OUTPUT_LIMIT = 1024**3
OUTPUT_RESERVE = 1024**2
RAW_KEYS = {"logits", "final_hidden", "pre_attention", "pairs", "sums",
            "train_ids", "test_ids", "probe_fit_indices", "probe_eval_indices",
            "null_permutations"}


def expected_roster(stage: str) -> list[tuple[int, str, int]]:
    if stage not in ("calibration", "remaining"):
        raise ValueError("stage must be calibration or remaining")
    rows = []
    for seed in SEEDS:
        for arm in ARMS:
            for step in STEPS:
                calibration = seed == 100 and step in (0, 6000)
                if calibration == (stage == "calibration"):
                    rows.append((seed, arm, step))
    return rows


def validate_combined_roster(calibration: Iterable[tuple], remaining: Iterable[tuple]) -> None:
    first, rest = list(calibration), list(remaining)
    if (len(first) != 6 or len(rest) != 144 or len(set(first)) != 6
            or len(set(rest)) != 144 or set(first) & set(rest)
            or set(first) != set(expected_roster("calibration"))
            or set(rest) != set(expected_roster("remaining"))):
        raise ValueError("inputs do not form the exact disjoint 6+144 roster")


def _read_json(path: Path) -> tuple[dict, str, bytes]:
    if path.is_symlink() or not path.is_file() or path.stat().st_nlink != 1:
        raise ValueError(f"not a singly-linked regular JSON file: {path}")
    raw = path.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"JSON root is not an object: {path}")
    return value, file_hash(path), raw


def _validate_probe(probe: dict) -> None:
    observed, null = probe["observed"], probe["null"]
    frequency_rows = observed["per_frequency"]
    if ([row["frequency"] for row in frequency_rows] != list(range(1, 57))
            or len(observed["selected_frequencies"]) != 5
            or null["n_nulls"] != 20 or len(null["runs"]) != 20):
        raise ValueError("probe scalar schema differs from the fixed recipe")
    for run in null["runs"]:
        if ([row["frequency"] for row in run["per_frequency"]] != list(range(1, 57))
                or len(run["selected_frequencies"]) != 5):
            raise ValueError("null probe scalar schema differs from the fixed recipe")


def _verify_stage(path: Path, stage: str) -> dict[str, Any]:
    path = path.resolve()
    manifest, manifest_hash, _ = _read_json(path / "manifest.json")
    complete, complete_hash, _ = _read_json(path / "complete.json")
    roster = expected_roster(stage)
    manifest_roster = [tuple(row) for row in manifest.get("roster", [])]
    accepted = complete.get("accepted_results", [])
    accepted_roster = [(row.get("seed"), row.get("arm"), row.get("step"))
                       for row in accepted]
    if (manifest.get("stage") != stage or complete.get("stage") != stage
            or complete.get("status") != "complete"
            or manifest.get("recipe") != RECIPE
            or manifest.get("source_sha256") != complete.get("source_sha256")
            or manifest_roster != roster or accepted_roster != roster):
        raise ValueError(f"{stage} manifest/completion contract mismatch")
    records, result_receipts, raw_receipts = [], [], []
    for receipt in accepted:
        result_path = Path(receipt["path"])
        if result_path.resolve().parent != path:
            raise ValueError("result path escapes its acquisition directory")
        result, result_hash, result_bytes = _read_json(result_path)
        identity = (result.get("seed"), result.get("arm"), result.get("step"))
        if (identity != (receipt["seed"], receipt["arm"], receipt["step"])
                or result_hash != receipt["sha256"]
                or result.get("source_sha256") != manifest["source_sha256"]):
            raise ValueError("result receipt, identity, or source pin mismatch")
        for feature in RECIPE["features"]:
            _validate_probe(result["probes"][feature])
        raw = result["raw_activations"]
        raw_path = Path(raw["path"])
        if (raw_path.resolve().parent != path or raw_path.is_symlink()
                or not raw_path.is_file() or raw_path.stat().st_nlink != 1
                or raw_path.stat().st_size != raw["size_bytes"]
                or file_hash(raw_path) != raw["sha256"]):
            raise ValueError("raw activation receipt mismatch")
        records.append({"stage": stage, "result": result,
                        "result_path": result_path.resolve(),
                        "result_sha256": result_hash, "result_bytes": result_bytes,
                        "raw_path": raw_path.resolve(), "raw_sha256": raw["sha256"],
                        "raw_size_bytes": raw["size_bytes"]})
        result_receipts.append({"seed": identity[0], "arm": identity[1],
                                "step": identity[2], "path": str(result_path.resolve()),
                                "sha256": result_hash, "size_bytes": len(result_bytes)})
        raw_receipts.append({"seed": identity[0], "arm": identity[1],
                             "step": identity[2], "path": str(raw_path.resolve()),
                             "sha256": raw["sha256"], "size_bytes": raw["size_bytes"]})
    return {"stage": stage, "path": str(path), "manifest": manifest,
            "manifest_receipt": {"path": str(path / "manifest.json"),
                                 "sha256": manifest_hash},
            "completion": complete,
            "completion_receipt": {"path": str(path / "complete.json"),
                                   "sha256": complete_hash},
            "result_receipts": result_receipts, "raw_receipts": raw_receipts,
            "records": records}


def verify_inputs(calibration_dir: Path, remaining_dir: Path) -> dict[str, Any]:
    calibration = _verify_stage(calibration_dir, "calibration")
    remaining = _verify_stage(remaining_dir, "remaining")
    validate_combined_roster(
        [(r["result"]["seed"], r["result"]["arm"], r["result"]["step"])
         for r in calibration["records"]],
        [(r["result"]["seed"], r["result"]["arm"], r["result"]["step"])
         for r in remaining["records"]],
    )
    cm, rm = calibration["manifest"], remaining["manifest"]
    inherited = rm.get("calibration", {})
    if (cm["source_sha256"] != rm["source_sha256"] or cm["recipe"] != rm["recipe"]
            or cm.get("parent_summary_sha256") != rm.get("parent_summary_sha256")
            or cm.get("environment") != rm.get("environment")
            or inherited.get("completion_sha256")
               != calibration["completion_receipt"]["sha256"]
            or Path(inherited.get("path", "")).resolve()
               != Path(calibration["path"]).resolve()):
        raise ValueError("calibration and remaining source/recipe bindings differ")
    records = calibration["records"] + remaining["records"]
    records.sort(key=lambda r: (r["result"]["seed"], ARMS.index(r["result"]["arm"]),
                               r["result"]["step"]))
    splits = {}
    for record in records:
        result = record["result"]
        seed, split_hash = result["seed"], result["probe_split_sha256"]
        if seed in splits and splits[seed] != split_hash:
            raise ValueError("probe split changed across one seed's states")
        splits[seed] = split_hash
    return {"calibration": {k: v for k, v in calibration.items() if k != "records"},
            "remaining": {k: v for k, v in remaining.items() if k != "records"},
            "records": records}


def choose_fixed_panel(records: Iterable[dict]) -> dict[str, Any]:
    selected = []
    for item in records:
        result = item.get("result", item)
        if result["seed"] == 100 and result["step"] == 6000:
            selected.append(result)
    if len(selected) != 3 or {row["arm"] for row in selected} != set(ARMS):
        raise ValueError("fixed panel requires the three seed100 step6000 arms")
    means = []
    for frequency in range(1, 57):
        scores = [row["probes"]["final_hidden"]["observed"]
                  ["per_frequency"][frequency - 1] for row in selected]
        if any(score["frequency"] != frequency for score in scores):
            raise ValueError("frequency rows are not ordered 1..56")
        means.append(sum(score["fit_r2"] for score in scores) / 3)
    frequencies = sorted(range(1, 57), key=lambda k: (-means[k - 1], k))[:5]
    return {"selection_source": "mean fit R2 across seed100 step6000 final_hidden arms",
            "frequencies": frequencies,
            "mean_fit_r2_by_frequency": [{"frequency": k, "mean_fit_r2": means[k - 1]}
                                         for k in range(1, 57)]}


def pool_ratio(metrics: Iterable[dict], *, p: int = 113,
               energy_floor: float = 1e-12) -> dict[str, Any]:
    rows = list(metrics)
    count = sum(row["count"] for row in rows)
    numerator = float(sum(row["numerator"] for row in rows))
    denominator = float(sum(row["denominator"] for row in rows))
    raw_energy = sum(row["raw_rms"] ** 2 * (2 * row["count"] * p) for row in rows)
    centered_rms = math.sqrt(max(0.0, denominator) / (2 * count * p)) if count else 0.0
    raw_rms = math.sqrt(max(0.0, raw_energy) / (2 * count * p)) if count else 0.0
    defined = centered_rms > energy_floor and denominator > 0
    return {"count": count, "numerator": numerator, "denominator": denominator,
            "centered_rms": centered_rms, "raw_rms": raw_rms,
            "value": numerator / denominator if defined else None,
            "energy_defined": defined}


def correct_class_summary(logits: np.ndarray, sums: np.ndarray,
                          indices: np.ndarray) -> dict[str, Any]:
    ids = np.asarray(indices, dtype=np.int64)
    values = np.asarray(logits, dtype=np.float64)[ids]
    labels = np.asarray(sums, dtype=np.int64)[ids]
    if not len(ids) or values.ndim != 2 or values.shape[0] != len(labels):
        raise ValueError("margin inputs are empty or misaligned")
    correct = values[np.arange(len(ids)), labels]
    competitors = values.copy()
    competitors[np.arange(len(ids)), labels] = -np.inf
    margins = correct - competitors.max(axis=1)
    if not np.isfinite(margins).all():
        raise ValueError("correct-class margin is non-finite")
    return {"mean": float(margins.mean()), "count": len(ids)}


def _centered_rms(logits: np.ndarray, indices: np.ndarray) -> float:
    values = np.asarray(logits, dtype=np.float64)[indices]
    centered = values - values.mean(axis=1, keepdims=True)
    return float(np.sqrt(np.mean(centered * centered)))


def _probe_summary(probe: dict, fixed_frequencies: list[int]) -> dict[str, Any]:
    observed = probe["observed"]
    by_frequency = {row["frequency"]: row for row in observed["per_frequency"]}
    fixed = [{"frequency": frequency,
              "eval_r2": by_frequency[frequency]["eval_r2"]}
             for frequency in fixed_frequencies]
    return {"selected_frequencies": observed["selected_frequencies"],
            "selected_eval_mean_r2": observed["selected_eval_mean_r2"],
            "null_max_eval_mean_r2": probe["null"]["selected_eval_mean_r2_max"],
            "fixed_panel_frequencies": fixed_frequencies,
            "fixed_panel_eval_r2_by_frequency": fixed,
            "fixed_panel_eval_mean_r2": sum(row["eval_r2"] for row in fixed) / len(fixed),
            "per_frequency": [{"frequency": row["frequency"],
                               "fit_r2": row["fit_r2"], "eval_r2": row["eval_r2"]}
                              for row in observed["per_frequency"]]}


def _validate_arrays(archive, result: dict) -> dict[str, np.ndarray]:
    if set(archive.files) != RAW_KEYS:
        raise ValueError("raw NPZ member schema mismatch")
    arrays = {name: archive[name] for name in archive.files}
    p, n = 113, 113 * 113
    expected_pairs = np.column_stack((np.repeat(np.arange(p), p), np.tile(np.arange(p), p)))
    if (arrays["logits"].shape != (n, p)
            or arrays["final_hidden"].shape != (n, 128)
            or arrays["pre_attention"].shape != (n, 256)
            or not np.array_equal(arrays["pairs"], expected_pairs)
            or not np.array_equal(arrays["sums"], expected_pairs.sum(1) % p)
            or not all(np.isfinite(arrays[name]).all()
                       for name in ("logits", "final_hidden", "pre_attention"))):
        raise ValueError("raw activation shape/grid/finiteness mismatch")
    train, test = arrays["train_ids"], arrays["test_ids"]
    if (not np.issubdtype(train.dtype, np.integer)
            or not np.issubdtype(test.dtype, np.integer)
            or not np.array_equal(np.sort(np.concatenate((train, test))), np.arange(n))
            or len(np.intersect1d(train, test))):
        raise ValueError("raw train/test IDs do not partition the grid")
    seed = result["seed"]
    split = make_probe_split(torch.from_numpy(arrays["sums"][test]),
                             RECIPE["split_seed_offset"] + seed)
    permutations = make_row_permutations(len(test), RECIPE["null_seed_offset"] + seed,
                                         RECIPE["n_nulls"])
    if (not np.array_equal(arrays["probe_fit_indices"], split["fit_indices"])
            or not np.array_equal(arrays["probe_eval_indices"], split["eval_indices"])
            or not np.array_equal(arrays["null_permutations"],
                                  [row["permutation"] for row in permutations])
            or result["probe_split_sha256"] != split["sha256"]):
        raise ValueError("raw split/null recipe mismatch")
    return arrays


def analyze_state(record: dict, fixed_panel: dict) -> dict[str, Any]:
    result = record["result"]
    with np.load(record["raw_path"], allow_pickle=False) as archive:
        arrays = _validate_arrays(archive, result)
        logits = np.asarray(arrays["logits"], dtype=np.float64)
        sums = np.asarray(arrays["sums"], dtype=np.int64)
        train = np.asarray(arrays["train_ids"], dtype=np.int64)
        test = np.asarray(arrays["test_ids"], dtype=np.int64)
        symmetry = symmetry_diagnostics(logits, train, test, p=113)
        heldout_correct = [symmetry["heldout_symmetry"][axis][str(delta)]["correct"]
                           for axis in ("a", "b") for delta in DEFAULT_DELTAS]
        heldout_wrong = [symmetry["heldout_symmetry"][axis][str(delta)]["wrong_shift"]
                         for axis in ("a", "b") for delta in DEFAULT_DELTAS]
        centered = {"all": _centered_rms(logits, np.arange(len(logits))),
                    "train": _centered_rms(logits, train),
                    "test": _centered_rms(logits, test)}
        margins = {"train": correct_class_summary(logits, sums, train),
                   "test": correct_class_summary(logits, sums, test)}
    behavior = {name: {"loss": result["behavior"][name]["loss"],
                       "accuracy": result["behavior"][name]["accuracy"],
                       "count": result["behavior"][name]["count"],
                       "correct_class_margin_mean": margins[name]["mean"]}
                for name in ("train", "test")}
    frequencies = fixed_panel["frequencies"]
    return {"seed": result["seed"], "arm": result["arm"], "step": result["step"],
            "source": {"stage": record["stage"],
                       "source_scalar_sha256": record["result_sha256"],
                       "source_npz_sha256": record["raw_sha256"]},
            "behavior": behavior,
            "probes": {name: _probe_summary(result["probes"][name], frequencies)
                       for name in RECIPE["features"]},
            "symmetry": {"centered_logit_rms": centered,
                         "heldout_shift_pooled": {
                             "correct": pool_ratio(heldout_correct),
                             "wrong_shift": pool_ratio(heldout_wrong)},
                         "exchange": symmetry["exchange"],
                         "training_membership_pooled": symmetry["cleanup"]["pooled"]},
            "full_symmetry": symmetry}


AGGREGATE_PATHS = {
    "train_loss": ("behavior", "train", "loss"),
    "train_accuracy": ("behavior", "train", "accuracy"),
    "train_margin": ("behavior", "train", "correct_class_margin_mean"),
    "test_loss": ("behavior", "test", "loss"),
    "test_accuracy": ("behavior", "test", "accuracy"),
    "test_margin": ("behavior", "test", "correct_class_margin_mean"),
    "final_hidden_selected_r2": ("probes", "final_hidden", "selected_eval_mean_r2"),
    "final_hidden_fixed_r2": ("probes", "final_hidden", "fixed_panel_eval_mean_r2"),
    "final_hidden_null_max": ("probes", "final_hidden", "null_max_eval_mean_r2"),
    "pre_attention_selected_r2": ("probes", "pre_attention", "selected_eval_mean_r2"),
    "pre_attention_fixed_r2": ("probes", "pre_attention", "fixed_panel_eval_mean_r2"),
    "pre_attention_null_max": ("probes", "pre_attention", "null_max_eval_mean_r2"),
    "heldout_correct_defect": ("symmetry", "heldout_shift_pooled", "correct", "value"),
    "heldout_wrong_defect": ("symmetry", "heldout_shift_pooled", "wrong_shift", "value"),
    "exchange_defect": ("symmetry", "exchange", "correct", "value"),
    "training_membership_excess": ("symmetry", "training_membership_pooled", "excess"),
    "centered_logit_rms_all": ("symmetry", "centered_logit_rms", "all"),
    "centered_logit_rms_train": ("symmetry", "centered_logit_rms", "train"),
    "centered_logit_rms_test": ("symmetry", "centered_logit_rms", "test"),
}


def _nested(row: dict, path: tuple[str, ...]):
    value = row
    for key in path:
        value = value[key]
    return value


def aggregate_rows(rows: list[dict]) -> list[dict[str, Any]]:
    output = []
    for arm in ARMS:
        for step in STEPS:
            group = [row for row in rows if row["arm"] == arm and row["step"] == step]
            if len(group) != 5 or {row["seed"] for row in group} != set(SEEDS):
                raise ValueError("aggregate group is not the five fixed seeds")
            metrics = {}
            for name, path in AGGREGATE_PATHS.items():
                values = [_nested(row, path) for row in group]
                defined = np.asarray([value for value in values if value is not None],
                                     dtype=np.float64)
                metrics[name] = {"mean": float(defined.mean()) if len(defined) else None,
                                 "se": float(defined.std(ddof=1) / math.sqrt(len(defined)))
                                 if len(defined) > 1 else None,
                                 "defined_count": len(defined), "total_count": 5}
            output.append({"arm": arm, "step": step, "metrics": metrics})
    return output


def analysis_sources() -> dict[str, str]:
    paths = ("experiments/analyze_grokking_representations.py",
             "experiments/grokking_symmetry.py", "experiments/grokking_representation.py",
             "experiments/grokking_confirmation.py",
             "tests/test_grokking_representation_analysis.py",
             "tests/test_grokking_symmetry.py",
             "output/2026-09-09-spectral-grokking-mechanism/analysis-protocol.md")
    return {path: file_hash(REPO / path) for path in paths}


class Output:
    def __init__(self, path: Path):
        storage = Path("/tmp/spectral-experiment-artifacts").resolve()
        target = path.resolve()
        if (target == storage or not target.is_relative_to(storage) or target.exists()
                or not target.parent.is_dir()):
            raise ValueError("output must be an absent child of /tmp/spectral-experiment-artifacts")
        if shutil.disk_usage(target.parent).free < OUTPUT_LIMIT + OUTPUT_RESERVE:
            raise ValueError("insufficient free space for analysis output bound")
        target.mkdir(mode=0o700)
        (target / "raw").mkdir()
        (target / "states").mkdir()
        self.path, self.bytes = target, 0

    def write_bytes(self, relative: Path | str, raw: bytes) -> dict[str, Any]:
        if self.bytes + len(raw) + OUTPUT_RESERVE > OUTPUT_LIMIT:
            raise RuntimeError("analysis output budget exhausted")
        path = self.path / relative
        atomic_exclusive(path, lambda handle: handle.write(raw))
        self.bytes += path.stat().st_size
        return {"path": str(relative), "sha256": file_hash(path),
                "size_bytes": path.stat().st_size}

    def write_json(self, relative: Path | str, value: Any) -> dict[str, Any]:
        raw = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
        return self.write_bytes(relative, raw)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-dir", required=True, type=Path)
    parser.add_argument("--remaining-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    started = time.monotonic()
    pins = analysis_sources()
    output = Output(args.output_dir)
    accepted, source_copies = [], []
    manifest_receipt = output.write_json(
        "manifest.json", {"schema": "grokking_representation_analysis_v1",
                          "calibration_dir": str(args.calibration_dir.resolve()),
                          "remaining_dir": str(args.remaining_dir.resolve()),
                          "analysis_source_sha256": pins,
                          "output_limit_bytes": OUTPUT_LIMIT,
                          "cooperative_seconds": 1740, "threads": 1})
    try:
        inputs = verify_inputs(args.calibration_dir, args.remaining_dir)
        panel = choose_fixed_panel(inputs["records"])
        rows = []
        for record in inputs["records"]:
            if time.monotonic() - started > 1740:
                raise TimeoutError("cooperative analysis deadline exceeded")
            result = record["result"]
            stem = f"seed{result['seed']}-{result['arm']}-step{result['step']:06d}"
            copy_receipt = output.write_bytes(Path("raw") / (stem + ".json"),
                                              record["result_bytes"])
            if copy_receipt["sha256"] != record["result_sha256"]:
                raise RuntimeError("byte-identical scalar copy check failed")
            state = analyze_state(record, panel)
            state["source"]["scalar_copy_sha256"] = copy_receipt["sha256"]
            state_receipt = output.write_json(Path("states") / (stem + ".json"), state)
            state["state_json"] = state_receipt
            # Summary row excludes the duplicate full block-level symmetry tree.
            summary_row = {key: value for key, value in state.items()
                           if key != "full_symmetry"}
            rows.append(summary_row)
            source_copies.append({"seed": result["seed"], "arm": result["arm"],
                                  "step": result["step"], **copy_receipt})
            accepted.append({"seed": result["seed"], "arm": result["arm"],
                             "step": result["step"], **state_receipt})
            print(json.dumps({"analyzed": [result["seed"], result["arm"], result["step"]]}),
                  flush=True)
        rows.sort(key=lambda row: (row["seed"], ARMS.index(row["arm"]), row["step"]))
        summary = {"schema": "grokking_representation_analysis_summary_v1",
                   "fixed_frequency_panel": panel, "rows": rows,
                   "aggregate_rows": aggregate_rows(rows),
                   "input_receipts": {stage: {key: inputs[stage][key] for key in
                                      ("manifest_receipt", "completion_receipt",
                                       "result_receipts", "raw_receipts")}
                                      for stage in ("calibration", "remaining")},
                   "source_scalar_copies": source_copies,
                   "analysis_source_sha256": pins,
                   "interpretation": "descriptive saved-state analysis; seed is the replicate"}
        summary_receipt = output.write_json("summary.json", summary)
        if analysis_sources() != pins:
            raise RuntimeError("frozen analysis source changed during execution")
        complete = {"schema": "grokking_representation_analysis_complete_v1",
                    "status": "complete", "state_count": 150,
                    "state_receipts": accepted, "summary": summary_receipt,
                    "manifest": manifest_receipt,
                    "analysis_source_sha256": pins, "elapsed_seconds": time.monotonic() - started,
                    "output_bytes_before_completion": output.bytes}
        output.write_json("complete.json", complete)
    except BaseException as error:
        try:
            output.write_json("failure.json", {
                "schema": "grokking_representation_analysis_failure_v1",
                "status": "failed_preserved", "type": type(error).__name__,
                "message": str(error), "state_receipts": accepted,
                "source_scalar_copies": source_copies,
                "manifest": manifest_receipt,
                "analysis_source_sha256": pins,
                "elapsed_seconds": time.monotonic() - started,
                "output_bytes_before_failure": output.bytes})
        except BaseException:
            pass
        raise


if __name__ == "__main__":
    main()
