#!/usr/bin/env python3
"""Independent raw/provenance/summary audit; no experiment/summary imports.

Refuses outcome reads unless full execution is complete and the parent summary
exists. CPU tensor loading is used only for saved-state/hash checks, never model
evaluation or training. No reference eigensystem is recomputed here.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
RESULTS = HERE / "results"
SEEDS, STEPS = (3, 4, 5), (200, 500, 1000, 2000)
SOURCES = ("width32", "width128", "reference")
PROBES = ("clean", "noisy", "corruption_residual", "auxiliary_clean")
FREEZE = "de1ba26d43a6df80162ee9530d8b64c9ac08acf3"
LAUNCH = "bc29741aa9e15eacddd148e99a640c7d338040d9"
EXECUTION = "fa384618a2a3d8177960be419a144214aa2b21fc"
OBS = ("span_energy_fraction", "span_projector_distance", "relative_covariance_error",
       "represented_operator_energy_fraction", "trace_P_C", "covariance_estimator_trace",
       *(f"native_{p}_retention" for p in PROBES), "native_clean_minus_corruption_retention",
       "native_clean_corruption_cosine", "native_projected_clean_corruption_cosine",
       "native_current_gradient_retention", "native_previous_current_gradient_retention",
       "self_inclusion_retention_increment")
REF = ("optimal_rank32_energy", "covariance_trace", "covariance_squared_frobenius",
       *(f"{p}_retention" for p in PROBES), "clean_minus_corruption_retention")


def require(value, message):
    if not value:
        raise AssertionError(message)


def read(path):
    def unique(pairs):
        result = {}
        for k, v in pairs:
            require(k not in result, "duplicate JSON key")
            result[k] = v
        return result
    return json.loads(Path(path).read_text(), object_pairs_hook=unique)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(part)
    return digest.hexdigest()


def tensor_hash(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def close(a, b):
    if a is None or b is None:
        return a is b
    return math.isfinite(a) and math.isfinite(b) and math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-14)


def compare_tree(a, b, path="root"):
    if isinstance(b, dict):
        require(isinstance(a, dict) and set(a) == set(b), "dictionary schema " + path)
        for k in b:
            compare_tree(a[k], b[k], path + "." + k)
    elif isinstance(b, list):
        require(isinstance(a, list) and len(a) == len(b), "list length " + path)
        for i, (x, y) in enumerate(zip(a, b)):
            compare_tree(x, y, path + f"[{i}]")
    elif type(b) is float:
        require(type(a) in (float, int) and close(a, b), "number " + path)
    else:
        require(a == b, "value " + path)


def stat(values):
    x = np.asarray(values, dtype=np.float64)
    return {"mean": float(x.mean()), "median": float(np.median(x)),
            "minimum": float(x.min()), "maximum": float(x.max()),
            "sample_sd": float(x.std(ddof=1)) if len(x) > 1 else None}


def finite_tree(value):
    if isinstance(value, dict):
        for v in value.values(): finite_tree(v)
    elif isinstance(value, list):
        for v in value: finite_tree(v)
    elif isinstance(value, float):
        require(math.isfinite(value), "nonfinite JSON scalar")


def main():
    marker = read(RESULTS / "execution.json")
    require(marker.get("mode") == "full" and marker.get("status") == "complete"
            and marker.get("all_gates_passed") is True and (HERE / "summary.json").exists(),
            "OUTCOME GATE CLOSED")
    target = HERE / "raw-summary-audit.json"
    require(not target.exists(), "Refuse audit overwrite")
    started = datetime.now(timezone.utc).isoformat()
    clock = time.perf_counter()
    torch.set_num_threads(1)
    require(not torch.cuda.is_initialized(), "CPU-only audit")
    summary = read(HERE / "summary.json")
    pilot = read(HERE / "pilot" / "execution.json")
    require(marker["repository_revision"] == EXECUTION, "execution revision")
    git = lambda *args: subprocess.check_output(["git", "-C", str(ROOT), *args])
    commits = {}
    for earlier, later in ((FREEZE, LAUNCH), (LAUNCH, EXECUTION)):
        subprocess.run(["git", "-C", str(ROOT), "merge-base", "--is-ancestor", earlier, later], check=True)
    for revision in (FREEZE, LAUNCH, EXECUTION):
        stamp = git("show", "-s", "--format=%cI", revision).decode().strip()
        require(datetime.fromisoformat(stamp) <= datetime.fromisoformat(marker["started_utc"]), "pre-execution commit chronology")
        commits[revision] = stamp
    require(marker["started_utc"] <= marker["all_replay_completed_utc"] <= marker["completed_utc"], "phase chronology")
    require(marker["completed_replay_seeds"] == list(SEEDS), "three replay completions")
    identities = [(s, t) for s in SEEDS for t in STEPS]
    require([(r["seed"], r["step"]) for r in marker["completed_snapshots"]] == identities, "twelve ordered snapshot completions")
    require(marker["official_test_opened"] is False and marker["new_accuracy_or_checkpoint_selection_computed"] is False, "no evaluation flags")
    require(marker["source_sha256"] == pilot["source_sha256"] and len(marker["source_sha256"]) == 19, "pilot/source map")
    for relative, digest in marker["source_sha256"].items():
        require(sha(ROOT / relative) == digest, "current source " + relative)
        for revision in (FREEZE, LAUNCH, EXECUTION):
            require(hashlib.sha256(git("show", revision + ":" + relative)).hexdigest() == digest, "committed source " + relative)
    require(marker["historical_bindings"] == pilot["historical_bindings"], "historical/pilot bindings")
    verified = {}
    def artifact(item):
        path = str(Path(item["path"]).resolve())
        if path not in verified:
            verified[path] = {"size_bytes": Path(path).stat().st_size, "sha256": sha(path)}
        require(verified[path]["size_bytes"] == item["size_bytes"] and verified[path]["sha256"] == item["sha256"], "artifact " + path)
    require(len(marker["historical_bindings"]["artifacts"]) == 13, "thirteen history bindings")
    for item in marker["historical_bindings"]["artifacts"]: artifact(item)
    paths = sorted(RESULTS.glob("snapshot-seed*-step*.json"))
    replay_paths = sorted(RESULTS.glob("replay-seed*.json"))
    require(len(paths) == 12 and len(replay_paths) == 3, "record count")
    rows = {(r["seed"], r["step"]): r for r in map(read, paths)}
    require(sorted(rows) == identities, "unique snapshot identities")
    replays = {r["seed"]: r for r in map(read, replay_paths)}
    require(sorted(replays) == list(SEEDS), "unique replay seeds")
    finite_tree([marker, summary, list(rows.values()), list(replays.values())])
    stream_rows, anchors_checked, state_checks, raw_norm_error = 0, [], [], 0.
    for seed, replay in replays.items():
        require(replay["schema_version"] == 1 and replay["steps"] == 2000 and replay["instrumented"] is True, "replay identity")
        for key in ("all_historical_gates_passed", "all_state_gates_passed", "no_observer_resets"):
            require(replay[key] is True, "replay gate " + key)
        for key, expected in (("historical_scalar_comparison_steps", 2000), ("snapshot_count", 4),
                              ("probe_state_check_count", 4), ("observer_means_bitwise_equal_steps", 2000),
                              ("delivered_raw_equal_steps", 2000), ("historical_warmup_hash_steps", 100)):
            require(replay[key] == expected, "counter " + key)
        for key in ("trajectory_parameter_sha256", "raw_gradient_sha256", "innovation_sha256", "ranks_by_step", "repair_counts_by_step"):
            require(len(replay[key]) == 2000, "complete sequence " + key)
        historical = read(HERE.parent / "iteration-004" / "results" / f"seed{seed}-adamw.json")
        require(replay["warmup_trajectory_hashes"] == historical["warmup_trajectory_hashes"] and len(replay["warmup_trajectory_hashes"]) == 100, "historical warmup")
        require(replay["warmup_core_sha256"] == historical["warmup_core_sha256"], "historical core hash")
        streams = {}
        require(len(replay["stream_artifacts"]) == 2 and len(replay["snapshot_artifacts"]) == 4, "bulk record schedule")
        for item in replay["stream_artifacts"]:
            artifact(item)
            require(item["completed_rows"] == 2000 and item["shape"] == [2000, 50890] and item["dtype"] == "float32", "stream header metadata")
            kind = "innovations" if item["path"].endswith("-innovations.npy") else "raw"
            array = np.load(item["path"], mmap_mode="r", allow_pickle=False)
            require(array.shape == (2000, 50890) and array.dtype == np.float32, "stream actual header")
            streams[kind] = array
            hashes = replay["innovation_sha256" if kind == "innovations" else "raw_gradient_sha256"]
            for i in range(2000):
                require(hashlib.sha256(array[i].tobytes()).hexdigest() == hashes[i], "individual stream row hash")
                if kind == "raw":
                    vector = array[i].astype(np.float64)
                    squared = float(np.dot(vector, vector))
                    old_norm = historical["steps_raw"][i]["raw_squared_norm"]
                    require(math.isclose(squared, old_norm, rel_tol=5e-12, abs_tol=1e-14), "historical raw norm CPU comparison")
                    raw_norm_error = max(raw_norm_error, abs(squared - old_norm))
                stream_rows += 1
        first = replay["first_accepted_innovation_step"]
        require(first is not None and not np.count_nonzero(streams["innovations"][:first - 1]), "initialization history")
        require(np.count_nonzero(streams["innovations"][first - 1]), "first accepted innovation")
        for i, warm in enumerate(replay["warmup_trajectory_hashes"]):
            require(warm["step"] == i + 1 and warm["parameters"] == replay["trajectory_parameter_sha256"][i]
                    and warm["raw_gradient"] == warm["applied_gradient"] == replay["raw_gradient_sha256"][i], "warmup/new stream link")
        states = torch.load(historical["checkpoint_path"], map_location="cpu", weights_only=True)
        require(set(states) == {"final", "min_val_ce", "max_val_accuracy", "warmup100"}, "four saved anchors")
        require(len(replay["checkpoint_comparisons"]) == 4, "four anchor records")
        for entry in replay["checkpoint_comparisons"]:
            name, step = entry["name"], entry["step"]
            require(entry["bitwise_equal"] is True and step == historical["checkpoint_steps"][name], "historical anchor identity")
            state = states[name]
            require(set(state) == {"0.weight", "0.bias", "2.weight", "2.bias"}, "checkpoint keys")
            vector = torch.cat([state[k].reshape(-1) for k in ("0.weight", "0.bias", "2.weight", "2.bias")])
            require(vector.numel() == 50890, "checkpoint dimension")
            if step > 0:
                require(tensor_hash(vector) == replay["trajectory_parameter_sha256"][step - 1], "checkpoint/poststep parameter hash")
            anchors_checked.append({"seed": seed, "name": name, "step": step, "parameter_hash_verified": step > 0})
        for item in replay["snapshot_artifacts"]:
            artifact(item)
            saved = torch.load(item["path"], map_location="cpu", weights_only=True)
            step = saved["step"]
            row = rows[seed, step]
            require(saved["seed"] == seed and saved["state_timing"] == row["state_timing"] == "pre_adam_after_current_gradient_observation", "snapshot timing")
            require(saved["parameter_hash"] == row["parameter_hash"] == replay["trajectory_parameter_sha256"][step - 2], "pre-Adam parameter hash")
            require(tensor_hash(saved["raw"]) == replay["raw_gradient_sha256"][step - 1], "snapshot current gradient")
            require(tensor_hash(saved["innovation"]) == replay["innovation_sha256"][step - 1], "snapshot innovation")
            require(torch.equal(saved["raw"] - saved["mean"], saved["innovation"]), "saved rounded centering")
            require(torch.equal(saved["mean"], saved["observers"]["32"]["grad_mean"]) and torch.equal(saved["mean"], saved["observers"]["128"]["grad_mean"]), "same observer means")
            for width in (32, 128):
                state = saved["observers"][str(width)]
                require(state["step_count"] == step and state["V"].shape[1] == row["observers"][f"width{width}"]["actual_rank"], "saved observer rank/time")
            state_checks.append({"seed": seed, "step": step, "phase_raw_mean_links_verified": True})
        del streams, states
    metric_checks = 0
    for (seed, step), row in rows.items():
        require(row["schema_version"] == 1 and row["all_numerical_gates_passed"] is True, "snapshot gate")
        require(set(row["observers"]) == {"width32", "width128"}, "snapshot widths")
        for item in row["artifacts"]: artifact(item)
        ref = row["reference"]
        positive, gap = ref["diagnostics"]["positive_rank"], ref["diagnostics"]["boundary_relative_gap"]
        valid = positive >= 32 and gap is not None and gap > 1e-6
        require(ref["diagnostics"]["reference_projector_valid"] is valid, "reference rank/gap flag")
        optimum = ref["metrics"]["optimal_rank32_energy"]
        for source in SOURCES:
            block = ref if source == "reference" else row["observers"][source]
            require(set(block["metrics"]) == set(REF if source == "reference" else OBS), "metric schema")
            require(set(block["null_reasons"]) == {k for k, v in block["metrics"].items() if v is None}, "exact null mask")
            energy = block["energies"] if source == "reference" else block["energies"]["probes"]
            for probe in PROBES:
                a, b = energy[probe]["input_squared_norm"], energy[probe]["output_squared_norm"]
                name = ("" if source == "reference" else "native_") + probe + "_retention"
                require(a >= 0 and (b is None or b >= 0), "nonnegative energies")
                require(close(a, ref["energies"][probe]["input_squared_norm"]), "common probe energy")
                require(close(block["metrics"][name], None if not a or b is None else b / a), "raw retention ratio")
                if source == "reference": require((b is not None) == valid, "reference projection availability")
                metric_checks += 1
            prefix = "" if source == "reference" else "native_"
            c, r = block["metrics"][prefix + "clean_retention"], block["metrics"][prefix + "corruption_residual_retention"]
            require(close(block["metrics"][prefix + "clean_minus_corruption_retention"], None if c is None or r is None else c - r), "selectivity arithmetic")
            if source != "reference":
                e, m = block["energies"], block["metrics"]
                require(block["estimation_width"] == int(source[5:]) and 0 <= block["actual_rank"] <= int(source[5:]), "rank cap")
                good = block["actual_rank"] >= 32 and positive >= 32 and optimum > 0
                require(close(m["span_energy_fraction"], e["span_captured_energy"] / optimum if good else None), "primary energy/rank arithmetic")
                require((m["span_projector_distance"] is not None) == (block["actual_rank"] >= 32 and valid), "distance null gate")
                require(close(m["represented_operator_energy_fraction"], e["represented_operator_output_energy"] / optimum if optimum else None), "operator energy ratio")
                raw_error = e["reference_squared_frobenius"] + e["estimator_squared_frobenius"] - 2 * e["covariance_inner_product"]
                require(close(raw_error, e["covariance_squared_error_raw"]), "covariance error arithmetic")
                require(close(m["relative_covariance_error"], math.sqrt(max(raw_error, 0.) / e["reference_squared_frobenius"]) if e["reference_squared_frobenius"] else None), "relative covariance error")
                a, b = m["native_current_gradient_retention"], m["native_previous_current_gradient_retention"]
                require(close(m["self_inclusion_retention_increment"], None if a is None or b is None else a - b), "self inclusion arithmetic")
            metric_checks += len(block["metrics"])
    # Independently derive every seed value and null mask before checking groups.
    def entry(seed, step, source, metric):
        row = rows[seed, step]
        if source == "width128_minus_width32":
            a, b = (entry(seed, step, w, metric) for w in ("width32", "width128"))
            value = None if a["value"] is None or b["value"] is None else b["value"] - a["value"]
            reason = {w: v["reason"] for w, v in zip(("width32", "width128"), (a, b)) if v["value"] is None}
            return {"seed": seed, "value": value, "reason": reason if value is None else None, "width32_value": a["value"], "width128_value": b["value"]}
        block = row["reference"] if source == "reference" else row["observers"][source]
        value = block["metrics"][metric]
        return {"seed": seed, "value": value, "reason": block["null_reasons"].get(metric) if value is None else None}
    audited_groups = []
    def group_check(actual, entries, label, secondary):
        good = [e for e in entries if e["value"] is not None]
        missing = [e for e in entries if e["value"] is None]
        expected = {"individual_values": entries, "valid_seed_mask": [e["seed"] for e in good], "unavailable": missing,
                    "valid_count": len(good), "required_count": len(entries),
                    "complete_case_statistics": stat([e["value"] for e in good]) if not missing else None}
        if secondary:
            expected["secondary_available_case_statistics"] = stat([e["value"] for e in good]) if good else None
        compare_tree(actual, expected, label)
        audited_groups.append({"label": label, **expected})
    require(summary["schema_version"] == 1 and summary["execution_status"] == "complete" and summary["seeds"] == list(SEEDS) and summary["snapshots"] == list(STEPS), "summary identity")
    primary = summary["primary"]
    require((primary["metric"], primary["step"], primary["direction"]) == ("span_energy_fraction", 2000, "width128_minus_width32"), "primary definition")
    group_check({k: v for k, v in primary.items() if k not in ("metric", "step", "direction")}, [entry(s, 2000, "width128_minus_width32", "span_energy_fraction") for s in SEEDS], "primary", False)
    require(set(summary["fixed_step_secondary"]) == {str(s) for s in STEPS}, "summary time schema")
    all_sources = (*SOURCES, "width128_minus_width32")
    for step in STEPS:
        require(set(summary["fixed_step_secondary"][str(step)]) == set(all_sources), "fixed source schema")
        for source in all_sources:
            names = REF if source == "reference" else OBS
            require(set(summary["fixed_step_secondary"][str(step)][source]) == set(names), "fixed metric schema")
            for name in names:
                group_check(summary["fixed_step_secondary"][str(step)][source][name], [entry(s, step, source, name) for s in SEEDS], f"fixed.{step}.{source}.{name}", True)
    require(set(summary["complete_four_snapshot_means"]) == set(all_sources), "four snapshot sources")
    for source in all_sources:
        names = REF if source == "reference" else OBS
        require(set(summary["complete_four_snapshot_means"][source]) == set(names), "four snapshot metrics")
        for name in names:
            entries = []
            for seed in SEEDS:
                values = [entry(seed, step, source, name) for step in STEPS]
                absent = [{"step": step, "reason": e["reason"]} for step, e in zip(STEPS, values) if e["value"] is None]
                value = None if absent else float(np.mean([e["value"] for e in values]))
                entries.append({"seed": seed, "value": value, "reason": absent if absent else None,
                                "valid_step_mask": [step for step, e in zip(STEPS, values) if e["value"] is not None],
                                "scheduled_step_values": [{"step": step, **e} for step, e in zip(STEPS, values)]})
            group_check(summary["complete_four_snapshot_means"][source][name], entries, f"four.{source}.{name}", False)
    require(set(summary["secondary_energy_weighted_four_snapshot_ratios"]) == set(SOURCES), "weighted sources")
    for source in SOURCES:
        require(set(summary["secondary_energy_weighted_four_snapshot_ratios"][source]) == set(PROBES), "weighted probes")
        for probe in PROBES:
            entries = []
            for seed in SEEDS:
                energies = []
                for step in STEPS:
                    block = rows[seed, step]["reference"] if source == "reference" else rows[seed, step]["observers"][source]
                    energies.append((block["energies"] if source == "reference" else block["energies"]["probes"])[probe])
                inp = sum(e["input_squared_norm"] for e in energies)
                absent = [step for step, e in zip(STEPS, energies) if e["output_squared_norm"] is None]
                out = None if absent else sum(e["output_squared_norm"] for e in energies)
                value = None if out is None or inp == 0 else out / inp
                entries.append({"seed": seed, "value": value, "reason": {"missing_output_steps": absent, "zero_total_input_energy": inp == 0} if value is None else None,
                                "summed_input_squared_norm": inp, "summed_output_squared_norm": out})
            group_check(summary["secondary_energy_weighted_four_snapshot_ratios"][source][probe], entries, f"weighted.{source}.{probe}", False)
    compare_tree(summary["raw_snapshot_records"], [rows[k] for k in identities], "embedded raw snapshots")
    provenance = summary["provenance"]
    compare_tree(provenance["execution"], marker, "summary execution copy")
    require(provenance["summary_source_sha256"] == sha(HERE / "summarize_results.py"), "summary source")
    input_hashes = {str(p.resolve()): sha(p) for p in [RESULTS / "execution.json", *paths, *replay_paths]}
    require(provenance["input_sha256"] == input_hashes, "summary input hashes")
    output = {"status": "passed", "started_utc": started, "completed_utc": datetime.now(timezone.utc).isoformat(),
              "elapsed_seconds": time.perf_counter() - clock, "audit_source_sha256": sha(__file__),
              "execution_sha256": sha(RESULTS / "execution.json"), "summary_sha256": sha(HERE / "summary.json"),
              "source_hashes_verified": marker["source_sha256"], "commit_chronology": commits,
              "historical_artifact_count": 13, "verified_artifacts": verified, "input_hashes": input_hashes,
              "raw_stream_rows_hashed": stream_rows, "historical_raw_norm_comparisons": 6000,
              "historical_raw_norm_max_abs_error": raw_norm_error, "historical_raw_norm_tolerance": {"rtol": 5e-12, "atol": 1e-14},
              "checkpoint_anchors": anchors_checked, "snapshot_state_links": state_checks,
              "raw_scalar_consistency_checks": metric_checks, "group_count": len(audited_groups), "groups": audited_groups,
              "primary": audited_groups[0], "scope": "Raw scalar reaggregation, stored-state/hash linkage and provenance. No eigensystem recomputation, model/dataset evaluation, optimizer training or GPU. New parameter arrays are not retained at every step, so historical displacement gates beyond saved anchors remain source/record-assertion evidence."}
    with target.open("x") as stream:
        json.dump(output, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": output["status"], "groups": len(audited_groups), "raw_rows_hashed": stream_rows,
                      "anchors": len(anchors_checked), "artifact_count": len(verified), "elapsed_seconds": output["elapsed_seconds"],
                      "primary": output["primary"], "output": str(target), "sha256": sha(target)}, indent=2))


if __name__ == "__main__":
    main()
