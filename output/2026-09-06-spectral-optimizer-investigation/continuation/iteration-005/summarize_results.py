"""Independent, strict descriptive analysis; no Torch, datasets or replay imports."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
SEEDS = (3, 4, 5)
STEPS = (200, 500, 1000, 2000)
WIDTHS = ("width32", "width128")
PROBES = ("clean", "noisy", "corruption_residual", "auxiliary_clean")
PRIMARY = "span_energy_fraction"
OBSERVER_METRICS = (
    PRIMARY, "span_projector_distance", "relative_covariance_error",
    "represented_operator_energy_fraction", "trace_P_C", "covariance_estimator_trace",
    *(f"native_{probe}_retention" for probe in PROBES),
    "native_clean_minus_corruption_retention", "native_clean_corruption_cosine",
    "native_projected_clean_corruption_cosine", "native_current_gradient_retention",
    "native_previous_current_gradient_retention", "self_inclusion_retention_increment",
)
REFERENCE_METRICS = (
    "optimal_rank32_energy", "covariance_trace", "covariance_squared_frobenius",
    *(f"{probe}_retention" for probe in PROBES), "clean_minus_corruption_retention",
)
LOCAL_SOURCES = ("replay_harness.py", "reference_math.py", "artifact_store.py", "protocol.md",
                 "design-intent.md", "best-practices-check.md", "reference-identities.md",
                 "check_reference_identities.py", "reference-identity-checks.json", "analysis-intent.md",
                 "result-schema.md", "implementation-decision.md", "test_harness.py",
                 "analysis-plan.md", "summarize_results.py", "test_summary.py")


def expected_source_paths():
    paths = [*(HERE / name for name in LOCAL_SOURCES),
             HERE.parent / "iteration-004" / "norm_control_harness.py",
             HERE.parent / "iteration-003" / "neural_harness.py", REPO / "spectral_filter.py"]
    return {str(path.relative_to(REPO)) for path in paths}


def validate_source_map(mapping):
    require(isinstance(mapping, dict) and set(mapping) == expected_source_paths(),
            "Missing/unexpected source bindings")
    require(all(isinstance(value, str) and len(value) == 64
                and all(c in "0123456789abcdef" for c in value) for value in mapping.values()),
            "Malformed source SHA256")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite_tree(value):
    if isinstance(value, float):
        require(math.isfinite(value), "Non-finite JSON number")
    elif isinstance(value, dict):
        for item in value.values():
            finite_tree(item)
    elif isinstance(value, list):
        for item in value:
            finite_tree(item)


def number(value):
    return type(value) in (int, float) and math.isfinite(value)


def same_number(actual, expected):
    return actual is None if expected is None else number(actual) and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-14)


def derived_difference(block, target, left, right):
    a, b = block["metrics"][left], block["metrics"][right]
    expected = None if a is None or b is None else a - b
    require(same_number(block["metrics"][target], expected), f"Inconsistent derived metric: {target}")


def metric_row(row, expected):
    require(set(row["metrics"]) == set(expected), "Unexpected metric schema")
    nulls = {name for name, value in row["metrics"].items() if value is None}
    require(set(row["null_reasons"]) == nulls, "Null-reason mask mismatch")
    for name, value in row["metrics"].items():
        require(value is None or number(value), f"Invalid metric {name}")
    require(all(isinstance(reason, str) and reason for reason in row["null_reasons"].values()),
            "Missing null reason")
    require(isinstance(row["energies"], dict) and isinstance(row["diagnostics"], dict),
            "Missing raw energy/diagnostic evidence")


def statistics_for(values):
    require(bool(values), "Cannot summarize empty values")
    return {"mean": statistics.mean(values), "median": statistics.median(values),
            "minimum": min(values), "maximum": max(values),
            "sample_sd": statistics.stdev(values) if len(values) > 1 else None}


def group(items, secondary=True):
    valid = [item for item in items if item["value"] is not None]
    unavailable = [item for item in items if item["value"] is None]
    result = {"individual_values": items, "valid_seed_mask": [v["seed"] for v in valid],
              "unavailable": unavailable, "valid_count": len(valid), "required_count": len(items),
              "complete_case_statistics": statistics_for([v["value"] for v in valid]) if not unavailable else None}
    if secondary:
        result["secondary_available_case_statistics"] = statistics_for([v["value"] for v in valid]) if valid else None
    return result


def item(seed, value, reason=None):
    return {"seed": seed, "value": value, "reason": reason if value is None else None}


def measured(row, source, name):
    block = row["reference"] if source == "reference" else row["observers"][source]
    return item(row["seed"], block["metrics"][name], block["null_reasons"].get(name))


def paired(row, name):
    narrow, wide = (measured(row, width, name) for width in WIDTHS)
    value = None if narrow["value"] is None or wide["value"] is None else wide["value"] - narrow["value"]
    reasons = {width: rec["reason"] for width, rec in zip(WIDTHS, (narrow, wide)) if rec["value"] is None}
    result = item(row["seed"], value, reasons)
    result["width32_value"], result["width128_value"] = narrow["value"], wide["value"]
    return result


def four_snapshot_mean(seed, entries):
    missing = [{"step": step, "reason": entry["reason"]} for step, entry in zip(STEPS, entries)
               if entry["value"] is None]
    result = item(seed, None if missing else statistics.mean(entry["value"] for entry in entries), missing)
    result["valid_step_mask"] = [step for step, entry in zip(STEPS, entries) if entry["value"] is not None]
    result["scheduled_step_values"] = [{"step": step, **entry} for step, entry in zip(STEPS, entries)]
    return result


def validate_execution(execution):
    finite_tree(execution)
    require(execution["schema_version"] == 1 and execution["mode"] == "full"
            and execution["status"] == "complete" and execution["all_gates_passed"] is True,
            "Not a completed gated confirmation")
    require(execution["official_test_opened"] is False
            and execution["new_accuracy_or_checkpoint_selection_computed"] is False, "Unexpected test/selection work")
    require(execution["completed_replay_seeds"] == list(SEEDS), "Missing/duplicate replay seeds")
    expected = [(seed, step) for seed in SEEDS for step in STEPS]
    require([(r["seed"], r["step"]) for r in execution["completed_snapshots"]] == expected,
            "Incomplete/unexpected snapshot schedule")


def validate(execution, rows, replays):
    validate_execution(execution)
    finite_tree([rows, replays])
    expected = [(seed, step) for seed in SEEDS for step in STEPS]
    require(sorted((r["seed"], r["step"]) for r in rows) == expected, "Missing/duplicate raw snapshots")
    require(sorted(r["seed"] for r in replays) == list(SEEDS), "Missing/duplicate replay records")
    for replay in replays:
        require(replay["schema_version"] == 1 and replay["steps"] == 2000 and replay["instrumented"] is True,
                "Invalid replay shape")
        require(replay["all_historical_gates_passed"] is True and replay["all_state_gates_passed"] is True,
                "Failed reproduction/state gate")
        require(replay["historical_scalar_comparison_steps"] == 2000 and replay["snapshot_count"] == 4
                and replay["probe_state_check_count"] == 4, "Incomplete reproduction/probe checks")
        require(replay["observer_means_bitwise_equal_steps"] == 2000 and replay["delivered_raw_equal_steps"] == 2000
                and replay["historical_warmup_hash_steps"] == 100 and replay["no_observer_resets"] is True,
                "Incomplete common-stream invariants")
        for name in ("trajectory_parameter_sha256", "raw_gradient_sha256", "innovation_sha256"):
            require(len(replay[name]) == 2000, "Incomplete new hash stream")
            require(all(isinstance(s, str) and len(s) == 64 and all(c in "0123456789abcdef" for c in s)
                        for s in replay[name]), "Malformed SHA256 stream")
        require(len(replay["warmup_trajectory_hashes"]) == 100, "Incomplete warmup hash checks")
        for step, warm in enumerate(replay["warmup_trajectory_hashes"], 1):
            require(set(warm) == {"step", "parameters", "raw_gradient", "applied_gradient"} and warm["step"] == step,
                    "Malformed warmup hash record")
            require(all(isinstance(warm[key], str) and len(warm[key]) == 64
                        and all(c in "0123456789abcdef" for c in warm[key])
                        for key in ("parameters", "raw_gradient", "applied_gradient")), "Malformed warmup SHA")
            require(warm["parameters"] == replay["trajectory_parameter_sha256"][step - 1]
                    and warm["raw_gradient"] == warm["applied_gradient"] == replay["raw_gradient_sha256"][step - 1],
                    "Warmup/current hash streams disagree")
        anchors = replay["checkpoint_comparisons"]
        require(len(anchors) == 4 and {r["name"] for r in anchors} == {"final", "min_val_ce", "max_val_accuracy", "warmup100"}
                and all(r["bitwise_equal"] is True for r in anchors), "Failed/missing historical checkpoint anchor")
        for anchor in anchors:
            require(type(anchor["step"]) is int and 0 <= anchor["step"] <= 2000 and anchor["step"] % 100 == 0,
                    "Invalid historical checkpoint step")
            require(anchor["name"] not in {"final", "warmup100"}
                    or anchor["step"] == {"final": 2000, "warmup100": 100}[anchor["name"]], "Wrong fixed checkpoint step")
    for row in rows:
        require(row["schema_version"] == 1 and row["all_numerical_gates_passed"] is True,
                "Failed snapshot numerical gate")
        require(row["state_timing"] == "pre_adam_after_current_gradient_observation", "Wrong snapshot phase")
        require(set(row["observers"]) == set(WIDTHS), "Unexpected observer widths")
        metric_row(row["reference"], REFERENCE_METRICS)
        ref = row["reference"]
        rank = ref["diagnostics"]["positive_rank"]
        gap = ref["diagnostics"]["boundary_relative_gap"]
        optimum = ref["metrics"]["optimal_rank32_energy"]
        require(type(rank) is int and rank >= 0 and number(optimum) and optimum >= 0,
                "Invalid reference rank/energy")
        require(gap is None or number(gap), "Invalid reference boundary gap")
        require(rank < 32 or gap is not None, "Missing full-rank reference boundary gap")
        projector_valid = rank >= 32 and gap is not None and gap > 1e-6
        require(ref["diagnostics"]["reference_projector_valid"] is projector_valid,
                "Reference validity flag contradicts rank/gap")
        derived_difference(ref, "clean_minus_corruption_retention", "clean_retention", "corruption_residual_retention")
        for width, size in zip(WIDTHS, (32, 128)):
            block = row["observers"][width]
            require(block["estimation_width"] == size and type(block["actual_rank"]) is int
                    and 0 <= block["actual_rank"] <= size, "Invalid observer rank/width")
            metric_row(block, OBSERVER_METRICS)
            mandatory = {"reference_squared_frobenius", "estimator_squared_frobenius", "covariance_inner_product",
                         "covariance_squared_error_raw", "span_captured_energy", "represented_operator_output_energy", "probes", "joint"}
            require(mandatory <= set(block["energies"]), "Missing mandatory covariance/probe energies")
            span_valid = block["actual_rank"] >= 32 and rank >= 32 and optimum > 0
            require((block["metrics"][PRIMARY] is not None) == span_valid,
                    "Primary validity contradicts rank/energy")
            distance_valid = block["actual_rank"] >= 32 and projector_valid
            require((block["metrics"]["span_projector_distance"] is not None) == distance_valid,
                    "Projector-distance validity contradicts rank/gap")
            for name in (PRIMARY, "span_projector_distance"):
                value = block["metrics"][name]
                require(value is None or -1e-6 <= value <= 1 + 1e-6, "Out-of-range span diagnostic")
            energy = block["energies"]
            require(same_number(energy["reference_squared_frobenius"], ref["metrics"]["covariance_squared_frobenius"]),
                    "Observers do not share the reference covariance norm")
            captured = energy["span_captured_energy"]
            require(captured is None or number(captured) and captured >= 0, "Invalid span-captured energy")
            require(not span_valid or number(captured) and same_number(block["metrics"][PRIMARY], captured / optimum),
                    "Primary disagrees with raw span energy")
            out_energy = energy["represented_operator_output_energy"]
            require(number(out_energy) and out_energy >= 0, "Invalid represented output energy")
            require(same_number(block["metrics"]["represented_operator_energy_fraction"],
                                out_energy / optimum if optimum else None), "Operator ratio disagrees with energy")
            for target, left, right in (("native_clean_minus_corruption_retention", "native_clean_retention", "native_corruption_residual_retention"),
                                        ("self_inclusion_retention_increment", "native_current_gradient_retention", "native_previous_current_gradient_retention")):
                derived_difference(block, target, left, right)
        for source in (*WIDTHS, "reference"):
            block = row["reference"] if source == "reference" else row["observers"][source]
            energies = block["energies"] if source == "reference" else block["energies"]["probes"]
            for probe in PROBES:
                inp, out = (energies[probe][key] for key in ("input_squared_norm", "output_squared_norm"))
                require(number(inp) and inp >= 0 and (out is None or number(out) and out >= 0), "Invalid probe energies")
                require(same_number(inp, ref["energies"][probe]["input_squared_norm"]),
                        "Observers do not share the same probe input energy")
                require(source == "reference" or out is not None, "Native output energy cannot be unavailable")
                require(inp != 0 or out in (None, 0), "Zero input has nonzero projected energy")
                if source == "reference":
                    require((out is not None) == projector_valid, "Reference probe contradicts projector validity")
                name = ("" if source == "reference" else "native_") + probe + "_retention"
                value = block["metrics"][name]
                expected_ratio = None if inp == 0 or out is None else out / inp
                require(value is None if expected_ratio is None else value is not None
                        and math.isclose(value, expected_ratio, rel_tol=1e-12, abs_tol=1e-14), "Retention disagrees with raw energy")


def build_summary(execution, rows, replays):
    validate(execution, rows, replays)
    indexed = {(row["seed"], row["step"]): row for row in rows}
    fixed = {}
    for step in STEPS:
        source_rows = {}
        for source in (*WIDTHS, "reference"):
            names = REFERENCE_METRICS if source == "reference" else OBSERVER_METRICS
            source_rows[source] = {name: group([measured(indexed[seed, step], source, name) for seed in SEEDS])
                                   for name in names}
        source_rows["width128_minus_width32"] = {name: group([paired(indexed[seed, step], name) for seed in SEEDS])
                                                 for name in OBSERVER_METRICS}
        fixed[str(step)] = source_rows
    four = {}
    for source in (*WIDTHS, "reference", "width128_minus_width32"):
        names = REFERENCE_METRICS if source == "reference" else OBSERVER_METRICS
        four[source] = {}
        for name in names:
            values = []
            for seed in SEEDS:
                entries = [paired(indexed[seed, step], name) if source == "width128_minus_width32"
                           else measured(indexed[seed, step], source, name) for step in STEPS]
                values.append(four_snapshot_mean(seed, entries))
            four[source][name] = group(values, secondary=False)
    weighted = {}
    for source in (*WIDTHS, "reference"):
        weighted[source] = {}
        for probe in PROBES:
            entries = []
            for seed in SEEDS:
                e = []
                for step in STEPS:
                    block = indexed[seed, step]["reference"] if source == "reference" else indexed[seed, step]["observers"][source]
                    e.append((block["energies"] if source == "reference" else block["energies"]["probes"])[probe])
                inp = sum(rec["input_squared_norm"] for rec in e)
                missing = [step for step, rec in zip(STEPS, e) if rec["output_squared_norm"] is None]
                out = None if missing else sum(rec["output_squared_norm"] for rec in e)
                value = None if out is None or inp == 0 else out / inp
                entry = item(seed, value, {"missing_output_steps": missing, "zero_total_input_energy": inp == 0})
                entry.update(summed_input_squared_norm=inp, summed_output_squared_norm=out)
                entries.append(entry)
            weighted[source][probe] = group(entries, secondary=False)
    result = {"schema_version": 1, "execution_status": "complete", "seeds": list(SEEDS), "snapshots": list(STEPS),
              "primary": {"metric": PRIMARY, "step": 2000, "direction": "width128_minus_width32",
                          **group([paired(indexed[seed, 2000], PRIMARY) for seed in SEEDS], secondary=False)},
              "fixed_step_secondary": fixed, "complete_four_snapshot_means": four,
              "secondary_energy_weighted_four_snapshot_ratios": weighted,
              "raw_snapshot_records": [indexed[key] for key in sorted(indexed)],
              "interpretation": "Three reused streams; dependent prefixes. Descriptive only. No new learning-policy or test outcome."}
    finite_tree(result)
    return result


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reject_duplicates(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path):
    return json.loads(path.read_text(), object_pairs_hook=reject_duplicates)


def summarize_directory(directory):
    execution_path = directory / "execution.json"
    execution = read_json(execution_path)
    # Refuse incomplete/pilot work before reading any outcome records.
    validate_execution(execution)
    snapshot_paths = sorted(directory.glob("snapshot-seed*-step*.json"))
    replay_paths = sorted(directory.glob("replay-seed*.json"))
    validate_source_map(execution["source_sha256"])
    for relative, expected in execution["source_sha256"].items():
        target = (REPO / relative).resolve()
        require(target.is_relative_to(REPO.resolve()), "Source path escapes repository")
        require(digest(target) == expected, f"Source changed after execution: {relative}")
    result = build_summary(execution, [read_json(p) for p in snapshot_paths], [read_json(p) for p in replay_paths])
    result["provenance"] = {"execution": execution,
                            "summary_source_sha256": digest(Path(__file__)),
                            "input_sha256": {str(path.resolve()): digest(path) for path in [execution_path, *snapshot_paths, *replay_paths]}}
    return result


def write_new(path, result):
    payload = json.dumps(result, indent=2, allow_nan=False) + "\n"
    with path.open("x") as handle:
        handle.write(payload)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=HERE / "results")
    parser.add_argument("--output", type=Path, default=HERE / "summary.json")
    args = parser.parse_args()
    require(not args.output.exists(), "Refusing to overwrite summary")
    result = summarize_directory(args.results)
    write_new(args.output, result)
    print(json.dumps({"output": str(args.output), "sha256": digest(args.output), "primary": result["primary"]}, allow_nan=False))


if __name__ == "__main__":
    main()
