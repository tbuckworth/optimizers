#!/usr/bin/env python3
"""Independent saved-array audit; never loads checkpoints or runs a model."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path("/tmp/spectral-experiment-artifacts/spectral-grokking-representation-20260909.5FsNeX")
ARMS = ("adamw", "legacy", "stable")
SEEDS = tuple(range(100, 105))
STEPS = (0, 100, 500, 1000, 1500, 2000, 2500, 3000, 4000, 6000)
AUDIT_STEPS = (1500, 2500)  # fixed before inspecting intermediate outcomes
FEATURES = ("final_hidden", "pre_attention")
NULLS = (0, 19)  # fixed endpoints, not selected by their scores
ANALYSIS = ROOT / "analysis-001"


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def int64_hash(name: str, value: np.ndarray) -> str:
    array = np.ascontiguousarray(value, dtype=np.int64)
    digest = hashlib.sha256()
    digest.update(name.encode() + b"\0")
    digest.update(str(tuple(array.shape)).encode("ascii") + b"\0")
    digest.update(array.tobytes())
    return digest.hexdigest()


def permutations_hash(permutations: np.ndarray) -> str:
    digest = hashlib.sha256()
    for index, permutation in enumerate(permutations):
        digest.update(str(index).encode("ascii") + b"\0")
        digest.update(int64_hash("row_permutation", permutation).encode("ascii") + b"\0")
    return digest.hexdigest()


def fourier_targets(sums: np.ndarray, p: int = 113) -> np.ndarray:
    frequencies = np.arange(1, (p - 1) // 2 + 1, dtype=np.float64)
    angles = (2 * np.pi / p) * sums.astype(np.float64)[:, None] * frequencies[None, :]
    return np.stack((np.cos(angles), np.sin(angles)), axis=2)


def probe(features: np.ndarray, sums: np.ndarray, fit: np.ndarray,
          evaluation: np.ndarray, targets: np.ndarray | None = None) -> dict:
    values = np.asarray(features, dtype=np.float64)
    feature_mean = values[fit].mean(axis=0)
    centered_fit = values[fit] - feature_mean
    rms = float(np.sqrt(np.mean(centered_fit * centered_fit)))
    scale = 1.0 if rms == 0.0 else rms
    x_fit = centered_fit / scale
    x_eval = (values[evaluation] - feature_mean) / scale
    gram = x_fit.T @ x_fit / len(fit) + 0.001 * np.eye(values.shape[1])

    values_y = fourier_targets(sums) if targets is None else targets
    y_fit, y_eval = values_y[fit], values_y[evaluation]
    target_mean = y_fit.mean(axis=0)
    rhs = x_fit.T @ (y_fit - target_mean).reshape(len(fit), -1) / len(fit)
    coefficients = np.linalg.solve(gram, rhs)
    fit_prediction = (x_fit @ coefficients).reshape(y_fit.shape) + target_mean
    eval_prediction = (x_eval @ coefficients).reshape(y_eval.shape) + target_mean
    fit_r2 = 1 - np.sum((y_fit - fit_prediction) ** 2, axis=(0, 2)) / np.sum(
        (y_fit - target_mean) ** 2, axis=(0, 2))
    eval_r2 = 1 - np.sum((y_eval - eval_prediction) ** 2, axis=(0, 2)) / np.sum(
        (y_eval - target_mean) ** 2, axis=(0, 2))
    selected_zero = sorted(range(56), key=lambda index: (-float(fit_r2[index]), index + 1))[:5]
    return {
        "fit_r2": fit_r2, "eval_r2": eval_r2, "target_mean": target_mean,
        "feature_mean": feature_mean, "rms": scale,
        "selected": [index + 1 for index in selected_zero],
        "selected_eval": eval_r2[selected_zero],
        "selected_mean": float(eval_r2[selected_zero].mean()),
    }


def compare_probe(calculated: dict, stored: dict, maxima: dict) -> None:
    rows = stored["per_frequency"]
    assert [row["frequency"] for row in rows] == list(range(1, 57))
    maxima["r2"] = max(maxima["r2"], *(abs(calculated["fit_r2"][i] - rows[i]["fit_r2"])
                                              for i in range(56)),
                       *(abs(calculated["eval_r2"][i] - rows[i]["eval_r2"])
                         for i in range(56)))
    maxima["target_mean"] = max(
        maxima["target_mean"],
        *(abs(calculated["target_mean"][i, 0] - rows[i]["fit_target_mean_cos"])
          for i in range(56)),
        *(abs(calculated["target_mean"][i, 1] - rows[i]["fit_target_mean_sin"])
          for i in range(56)),
    )
    assert calculated["selected"] == stored["selected_frequencies"]
    if "selected_eval_r2" in stored:
        maxima["r2"] = max(
            maxima["r2"], abs(calculated["selected_mean"] - stored["selected_eval_mean_r2"]),
            *(abs(x - y) for x, y in zip(calculated["selected_eval"],
                                         stored["selected_eval_r2"])),
        )


def verify_stored_probe(probe_record: dict) -> None:
    observed = probe_record["observed"]
    rows = observed["per_frequency"]
    assert [row["frequency"] for row in rows] == list(range(1, 57))
    selected = sorted(range(56), key=lambda index: (-rows[index]["fit_r2"], index + 1))[:5]
    assert observed["selected_frequencies"] == [index + 1 for index in selected]
    assert observed["selected_eval_r2"] == [rows[index]["eval_r2"] for index in selected]
    assert abs(observed["selected_eval_mean_r2"] - np.mean(observed["selected_eval_r2"])) < 1e-15
    null_means = []
    for expected_index, null in enumerate(probe_record["null"]["runs"]):
        assert null["null_index"] == expected_index
        rows = null["per_frequency"]
        assert [row["frequency"] for row in rows] == list(range(1, 57))
        selected = sorted(range(56), key=lambda index: (-rows[index]["fit_r2"], index + 1))[:5]
        assert null["selected_frequencies"] == [index + 1 for index in selected]
        selected_mean = float(np.mean([rows[index]["eval_r2"] for index in selected]))
        assert abs(null["selected_eval_mean_r2"] - selected_mean) < 1e-15
        null_means.append(selected_mean)
    summary = probe_record["null"]
    assert abs(summary["selected_eval_mean_r2_mean"] - np.mean(null_means)) < 1e-15
    assert abs(summary["selected_eval_mean_r2_std_population"] - np.std(null_means)) < 1e-15
    assert abs(summary["selected_eval_mean_r2_max"] - np.max(null_means)) < 1e-15


def heldout_shift(logits: np.ndarray, test: np.ndarray, p: int = 113) -> dict:
    """Independently pool correct-shift held-out-to-held-out edges."""
    values = np.asarray(logits, dtype=np.float64)
    centered = values - values.mean(axis=1, keepdims=True)
    heldout = np.zeros(p * p, dtype=bool)
    heldout[test] = True
    count, numerator, denominator, raw_energy = 0, 0.0, 0.0, 0.0
    for axis in (0, 1):
        for delta in (1, 2, 4, 8, 16, 32):
            a, b = test // p, test % p
            destination = (((a + delta) % p) * p + b if axis == 0
                           else a * p + (b + delta) % p)
            mask = heldout[destination]
            source_ids, destination_ids = test[mask], destination[mask]
            source, target = centered[source_ids], centered[destination_ids]
            residual = target - np.roll(source, shift=delta, axis=1)
            numerator += float(np.sum(residual * residual))
            denominator += float(np.sum(source * source) + np.sum(target * target))
            raw_source, raw_target = values[source_ids], values[destination_ids]
            raw_energy += float(np.sum(raw_source * raw_source) + np.sum(raw_target * raw_target))
            count += len(source_ids)
    return {"count": count, "numerator": numerator, "denominator": denominator,
            "centered_rms": math.sqrt(denominator / (2 * count * p)),
            "raw_rms": math.sqrt(raw_energy / (2 * count * p)),
            "value": numerator / denominator, "energy_defined": True}


def nested(row: dict, path: tuple[str, ...]):
    value = row
    for key in path:
        value = value[key]
    return value


def main() -> None:
    stages = {name: ROOT / name for name in ("calibration", "remaining")}
    manifests = {name: json.loads((path / "manifest.json").read_text())
                 for name, path in stages.items()}
    completions = {name: json.loads((path / "complete.json").read_text())
                   for name, path in stages.items()}
    assert manifests["calibration"]["source_sha256"] == manifests["remaining"]["source_sha256"]
    assert manifests["calibration"]["recipe"] == manifests["remaining"]["recipe"]
    assert manifests["calibration"]["environment"] == manifests["remaining"]["environment"]
    assert completions["calibration"]["source_sha256"] == completions["remaining"]["source_sha256"]
    assert manifests["remaining"]["calibration"]["completion_sha256"] == file_hash(
        stages["calibration"] / "complete.json")

    expected = {(seed, arm, step) for seed in SEEDS for arm in ARMS for step in STEPS}
    records: dict[tuple[int, str, int], tuple[dict, Path]] = {}
    for stage, directory in stages.items():
        manifest, completion = manifests[stage], completions[stage]
        assert completion["status"] == "complete" and completion["stage"] == stage
        assert completion["source_sha256"] == manifest["source_sha256"]
        receipts = completion["accepted_results"]
        assert len(receipts) == (6 if stage == "calibration" else 144)
        counted_bytes = (directory / "manifest.json").stat().st_size
        for receipt in receipts:
            result_path = Path(receipt["path"])
            assert result_path.parent == directory and file_hash(result_path) == receipt["sha256"]
            record = json.loads(result_path.read_text())
            key = (record["seed"], record["arm"], record["step"])
            assert key == (receipt["seed"], receipt["arm"], receipt["step"])
            assert key not in records and record["source_sha256"] == manifest["source_sha256"]
            raw = record["raw_activations"]
            raw_path = Path(raw["path"])
            assert raw_path.parent == directory and raw_path.stat().st_size == raw["size_bytes"]
            assert file_hash(raw_path) == raw["sha256"]
            counted_bytes += result_path.stat().st_size + raw_path.stat().st_size
            for feature in FEATURES:
                verify_stored_probe(record["probes"][feature])
            records[key] = record, raw_path
        assert counted_bytes == completion["output_bytes"]
    assert set(records) == expected
    assert set(map(tuple, manifests["calibration"]["roster"])) | set(
        map(tuple, manifests["remaining"]["roster"])) == expected
    assert not (set(map(tuple, manifests["calibration"]["roster"])) & set(
        map(tuple, manifests["remaining"]["roster"])))

    maxima = {"r2": 0.0, "target_mean": 0.0, "transform": 0.0,
              "behavior": 0.0, "margin": 0.0}
    audited_rows = []
    expected_fields = {"logits", "final_hidden", "pre_attention", "pairs", "sums",
                       "train_ids", "test_ids", "probe_fit_indices", "probe_eval_indices",
                       "null_permutations"}
    for seed in SEEDS:
        common = None
        for arm in ARMS:
            for step in AUDIT_STEPS:
                record, raw_path = records[seed, arm, step]
                with np.load(raw_path, allow_pickle=False) as archive:
                    assert set(archive.files) == expected_fields
                    arrays = {name: archive[name] for name in archive.files}
                assert all(np.isfinite(arrays[name]).all()
                           for name in ("logits", "final_hidden", "pre_attention"))
                pairs, sums = arrays["pairs"], arrays["sums"]
                expected_pairs = np.stack((np.repeat(np.arange(113), 113),
                                           np.tile(np.arange(113), 113)), axis=1)
                assert np.array_equal(pairs, expected_pairs)
                assert np.array_equal(sums, expected_pairs.sum(axis=1) % 113)
                train, test = arrays["train_ids"], arrays["test_ids"]
                fit, evaluation = arrays["probe_fit_indices"], arrays["probe_eval_indices"]
                permutations = arrays["null_permutations"]
                assert len(train) == 3830 and len(test) == 8939
                assert len(np.intersect1d(train, test)) == 0
                assert np.array_equal(np.sort(np.r_[train, test]), np.arange(12769))
                assert len(np.intersect1d(fit, evaluation)) == 0
                assert np.array_equal(np.sort(np.r_[fit, evaluation]), np.arange(8939))
                assert not np.isin(test[fit], train).any() and not np.isin(test[evaluation], train).any()
                labels = sums[test]
                for label in range(113):
                    count = int(np.sum(labels == label))
                    assert np.sum(labels[fit] == label) == count // 2
                    assert np.sum(labels[evaluation] == label) == count - count // 2
                assert permutations.shape == (20, 8939)
                for null_index, permutation in enumerate(permutations):
                    assert np.array_equal(np.sort(permutation), np.arange(8939))
                    assert not np.array_equal(permutation, np.arange(8939))
                    expected_hash = int64_hash("row_permutation", permutation)
                    for feature in FEATURES:
                        null = record["probes"][feature]["null"]["runs"][null_index]
                        assert null["permutation_sha256"] == expected_hash
                assert all(record["probes"][feature]["null"]["permutations_sha256"] ==
                           permutations_hash(permutations) for feature in FEATURES)
                invariant = tuple(hashlib.sha256(np.ascontiguousarray(arrays[name]).tobytes()).hexdigest()
                                  for name in ("pairs", "sums", "train_ids", "test_ids",
                                               "probe_fit_indices", "probe_eval_indices",
                                               "null_permutations"))
                if common is None:
                    common = invariant
                else:
                    assert invariant == common

                logits = arrays["logits"].astype(np.float64)
                behavior_and_margins = {}
                for name, indices in (("train", train), ("test", test)):
                    values = logits[indices]
                    targets = sums[indices].astype(np.int64)
                    maximum = values.max(axis=1)
                    loss = float(np.mean(maximum + np.log(np.exp(values - maximum[:, None]).sum(axis=1))
                                         - values[np.arange(len(indices)), targets]))
                    accuracy = float(np.mean(values.argmax(axis=1) == targets))
                    maxima["behavior"] = max(maxima["behavior"],
                                             abs(loss - record["behavior"][name]["loss"]),
                                             abs(accuracy - record["behavior"][name]["accuracy"]))
                    competitors = values.copy()
                    competitors[np.arange(len(indices)), targets] = -np.inf
                    margin = float(np.mean(values[np.arange(len(indices)), targets]
                                           - competitors.max(axis=1)))
                    behavior_and_margins[name] = {"loss": loss, "accuracy": accuracy,
                                                   "margin": margin}

                heldout = heldout_shift(logits, test)

                probe_scores = {}
                for feature in FEATURES:
                    calculated = probe(arrays[feature][test], labels, fit, evaluation)
                    stored = record["probes"][feature]
                    compare_probe(calculated, stored["observed"], maxima)
                    transform = stored["fit_transform"]
                    maxima["transform"] = max(
                        maxima["transform"],
                        float(np.max(np.abs(calculated["feature_mean"]
                                            - np.asarray(transform["feature_mean"])))),
                        abs(calculated["rms"] - transform["feature_rms_scalar"]),
                    )
                    for null_index in NULLS:
                        null_targets = fourier_targets(labels[permutations[null_index]])
                        calculated_null = probe(arrays[feature][test], labels, fit, evaluation,
                                                targets=null_targets)
                        stored_null = stored["null"]["runs"][null_index]
                        compare_probe(calculated_null, stored_null, maxima)
                        maxima["r2"] = max(
                            maxima["r2"],
                            abs(calculated_null["selected_mean"]
                                - stored_null["selected_eval_mean_r2"]),
                        )
                    probe_scores[feature] = {
                        "selected": calculated["selected"],
                        "selected_eval_mean_r2": calculated["selected_mean"],
                    }
                audited_rows.append({"seed": seed, "arm": arm, "step": step,
                                     "behavior": behavior_and_margins,
                                     "probes": probe_scores,
                                     "heldout_shift": heldout})

    # Verify the completed saved-array collector against all source scalar JSONs.
    analysis_manifest = json.loads((ANALYSIS / "manifest.json").read_text())
    analysis_complete = json.loads((ANALYSIS / "complete.json").read_text())
    summary_path = ANALYSIS / analysis_complete["summary"]["path"]
    assert file_hash(summary_path) == analysis_complete["summary"]["sha256"]
    assert summary_path.stat().st_size == analysis_complete["summary"]["size_bytes"]
    assert file_hash(ANALYSIS / analysis_complete["manifest"]["path"]) == \
        analysis_complete["manifest"]["sha256"]
    assert analysis_complete["status"] == "complete" and analysis_complete["state_count"] == 150
    assert analysis_complete["analysis_source_sha256"] == analysis_manifest["analysis_source_sha256"]
    assert sum(path.stat().st_size for path in ANALYSIS.rglob("*")
               if path.is_file() and path.name != "complete.json") == \
        analysis_complete["output_bytes_before_completion"]
    for relative, expected_hash in analysis_manifest["analysis_source_sha256"].items():
        assert file_hash(Path(relative)) == expected_hash
    summary = json.loads(summary_path.read_text())
    assert summary["analysis_source_sha256"] == analysis_manifest["analysis_source_sha256"]
    assert len(summary["rows"]) == len(summary["source_scalar_copies"]) == 150
    assert len(summary["aggregate_rows"]) == 30
    for stage, directory in stages.items():
        input_receipts = summary["input_receipts"][stage]
        for name in ("manifest_receipt", "completion_receipt"):
            receipt = input_receipts[name]
            assert file_hash(Path(receipt["path"])) == receipt["sha256"]
        result_receipts = {(row["seed"], row["arm"], row["step"]): row
                           for row in input_receipts["result_receipts"]}
        raw_receipts = {(row["seed"], row["arm"], row["step"]): row
                        for row in input_receipts["raw_receipts"]}
        stage_expected = set(map(tuple, manifests[stage]["roster"]))
        assert set(result_receipts) == set(raw_receipts) == stage_expected
        for key in stage_expected:
            source, raw_path = records[key]
            source_path = raw_path.with_suffix(".json")
            assert result_receipts[key]["sha256"] == file_hash(source_path)
            assert result_receipts[key]["size_bytes"] == source_path.stat().st_size
            assert raw_receipts[key]["sha256"] == file_hash(raw_path)
            assert raw_receipts[key]["size_bytes"] == raw_path.stat().st_size

    # The fixed panel is selected only from the three frozen calibration fits.
    fit_means = []
    for frequency in range(1, 57):
        fit_means.append(np.mean([
            records[100, arm, 6000][0]["probes"]["final_hidden"]["observed"]
            ["per_frequency"][frequency - 1]["fit_r2"] for arm in ARMS]))
    fixed_panel = sorted(range(1, 57), key=lambda k: (-fit_means[k - 1], k))[:5]
    stored_panel = summary["fixed_frequency_panel"]
    assert fixed_panel == stored_panel["frequencies"]
    maxima["fixed_panel_fit_mean"] = max(
        abs(fit_means[i] - stored_panel["mean_fit_r2_by_frequency"][i]["mean_fit_r2"])
        for i in range(56))

    summary_rows = {(row["seed"], row["arm"], row["step"]): row for row in summary["rows"]}
    assert set(summary_rows) == expected
    state_receipts = {(row["seed"], row["arm"], row["step"]): row
                      for row in analysis_complete["state_receipts"]}
    scalar_copies = {(row["seed"], row["arm"], row["step"]): row
                     for row in summary["source_scalar_copies"]}
    assert set(state_receipts) == set(scalar_copies) == expected
    for key, row in summary_rows.items():
        source, raw_path = records[key]
        source_json_path = raw_path.with_suffix(".json")
        source_hash = file_hash(source_json_path)
        assert row["source"]["source_scalar_sha256"] == source_hash
        assert row["source"]["source_npz_sha256"] == file_hash(raw_path)
        copy = scalar_copies[key]
        copy_path = ANALYSIS / copy["path"]
        assert file_hash(copy_path) == copy["sha256"] == source_hash
        assert copy_path.read_bytes() == source_json_path.read_bytes()
        receipt = state_receipts[key]
        state_path = ANALYSIS / receipt["path"]
        assert file_hash(state_path) == receipt["sha256"] == row["state_json"]["sha256"]
        assert state_path.stat().st_size == receipt["size_bytes"] == row["state_json"]["size_bytes"]
        state = json.loads(state_path.read_text())
        assert {name: value for name, value in state.items() if name != "full_symmetry"} == \
            {name: value for name, value in row.items() if name != "state_json"}
        for name in ("train", "test"):
            assert row["behavior"][name]["loss"] == source["behavior"][name]["loss"]
            assert row["behavior"][name]["accuracy"] == source["behavior"][name]["accuracy"]
            assert row["behavior"][name]["count"] == source["behavior"][name]["count"]
        for feature in FEATURES:
            probe_source = source["probes"][feature]
            probe_summary = row["probes"][feature]
            assert probe_summary["selected_frequencies"] == probe_source["observed"]["selected_frequencies"]
            assert probe_summary["selected_eval_mean_r2"] == probe_source["observed"]["selected_eval_mean_r2"]
            assert probe_summary["null_max_eval_mean_r2"] == probe_source["null"]["selected_eval_mean_r2_max"]
            expected_frequency_rows = [{"frequency": item["frequency"], "fit_r2": item["fit_r2"],
                                        "eval_r2": item["eval_r2"]}
                                       for item in probe_source["observed"]["per_frequency"]]
            assert probe_summary["per_frequency"] == expected_frequency_rows
            assert probe_summary["fixed_panel_frequencies"] == fixed_panel
            fixed_rows = [{"frequency": frequency,
                           "eval_r2": probe_source["observed"]["per_frequency"]
                           [frequency - 1]["eval_r2"]} for frequency in fixed_panel]
            assert probe_summary["fixed_panel_eval_r2_by_frequency"] == fixed_rows
            assert abs(probe_summary["fixed_panel_eval_mean_r2"]
                       - np.mean([item["eval_r2"] for item in fixed_rows])) < 1e-15

    # Recompute every aggregate from the 150 summary rows.
    aggregate_paths = {
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
    aggregate_rows = {(row["arm"], row["step"]): row for row in summary["aggregate_rows"]}
    assert set(aggregate_rows) == {(arm, step) for arm in ARMS for step in STEPS}
    maxima["aggregate"] = 0.0
    for (arm, step), aggregate in aggregate_rows.items():
        group = [summary_rows[seed, arm, step] for seed in SEEDS]
        for name, path in aggregate_paths.items():
            values = np.asarray([nested(row, path) for row in group
                                 if nested(row, path) is not None], dtype=np.float64)
            stored = aggregate["metrics"][name]
            assert stored["defined_count"] == len(values) and stored["total_count"] == 5
            if len(values):
                maxima["aggregate"] = max(maxima["aggregate"],
                    abs(float(values.mean()) - stored["mean"]))
            if len(values) > 1:
                maxima["aggregate"] = max(maxima["aggregate"],
                    abs(float(values.std(ddof=1) / math.sqrt(len(values))) - stored["se"]))

    # Compare independent margins and simple correct-shift pooling on the fixed 30-state subset.
    subset_rows = {(row["seed"], row["arm"], row["step"]): row for row in audited_rows}
    maxima["margin"] = maxima["heldout_shift_value_or_rms"] = 0.0
    maxima["heldout_shift_raw_sum"] = 0.0
    for key, audited in subset_rows.items():
        row = summary_rows[key]
        for name in ("train", "test"):
            maxima["margin"] = max(maxima["margin"],
                abs(audited["behavior"][name]["margin"]
                    - row["behavior"][name]["correct_class_margin_mean"]))
        for field, value in audited["heldout_shift"].items():
            stored = row["symmetry"]["heldout_shift_pooled"]["correct"][field]
            if isinstance(value, float):
                bucket = ("heldout_shift_raw_sum" if field in ("numerator", "denominator")
                          else "heldout_shift_value_or_rms")
                maxima[bucket] = max(maxima[bucket], abs(value - stored))
            else:
                assert value == stored

    print(json.dumps({
        "status": "pass",
        "full_roster_count": len(records),
        "arithmetic_subset": {"seeds": list(SEEDS), "arms": list(ARMS),
                              "steps": list(AUDIT_STEPS), "states": len(audited_rows),
                              "features": list(FEATURES), "null_indices": list(NULLS)},
        "max_abs_differences": maxima,
        "analysis_summary": {"rows": len(summary_rows), "aggregate_rows": len(aggregate_rows),
                             "fixed_panel": fixed_panel,
                             "independent_margin_and_shift_states": len(subset_rows)},
    }, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
