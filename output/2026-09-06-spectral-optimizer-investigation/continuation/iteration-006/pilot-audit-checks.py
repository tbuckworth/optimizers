#!/usr/bin/env python3
"""Independent runtime/isolation pilot audit; stdlib, no producer imports.

The author also wrote the full-study summarizer, which is NOT imported here.
Hash strings are compared; missing parameter/final-state tensors are not rebuilt.
"""
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import subprocess
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
FREEZE = "1221d70f1d95c8c8b60e701a08c9cedd3c3b5772"
ARMS = ("adamw", "current32", "lagged32", "lagged32_current_norm",
        "scalar_current32", "scalar_lagged32")
PREFIX = "output/2026-09-06-spectral-optimizer-investigation/continuation/iteration-006/"
SOURCES = {PREFIX + name for name in (
    "delivery_order_harness.py", "policy_math.py", "test_policy.py", "test_harness.py",
    "artifact_store.py", "protocol.md", "result-schema.md", "design-intent.md",
    "analysis-plan.md", "summarize_results.py", "test_summary.py", "best-practices-check.md",
    "challenge/decision.md")}
SOURCES.update({"spectral_filter.py", "output/2026-09-06-spectral-optimizer-investigation/continuation/iteration-003/neural_harness.py"})


def demand(condition, label):
    if not condition:
        raise AssertionError(label)


def digest(path):
    accumulator = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for piece in iter(lambda: stream.read(1024 * 1024), b""):
            accumulator.update(piece)
    return accumulator.hexdigest()


def json_read(path):
    def pairs(items):
        output = {}
        for key, value in items:
            demand(key not in output, "duplicate JSON key")
            output[key] = value
        return output
    def reject(value):
        raise AssertionError("nonfinite JSON constant " + value)
    data = json.loads(Path(path).read_text(), object_pairs_hook=pairs, parse_constant=reject)
    finite(data)
    return data


def finite(value):
    if isinstance(value, dict):
        for child in value.values():
            finite(child)
    elif isinstance(value, list):
        for child in value:
            finite(child)
    elif type(value) is float:
        demand(math.isfinite(value), "nonfinite numeric payload")


def hash_string(value):
    demand(type(value) is str and re.fullmatch("[0-9a-f]{64}", value), "invalid digest")


def resource(row):
    demand(0 <= row["elapsed_seconds"] <= 180, "pilot time cap")
    demand(type(row["peak_rss_bytes"]) is int and 0 <= row["peak_rss_bytes"] <= 12 * 1024**3, "RSS cap")
    demand(type(row["peak_gpu_allocated_bytes"]) is int and 0 <= row["peak_gpu_allocated_bytes"] <= 8 * 1024**3, "allocated GPU cap")
    demand(type(row["peak_gpu_reserved_bytes"]) is int and row["peak_gpu_reserved_bytes"] >= 0, "reservation")


def timestamp(text):
    value = datetime.fromisoformat(text)
    demand(value.tzinfo is not None, "naive timestamp")
    return value.astimezone(timezone.utc)


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT)


def main():
    began = time.perf_counter()
    started = datetime.now(timezone.utc).isoformat()
    output = HERE / "pilot-audit-checks.json"
    demand(not output.exists(), "refuse audit overwrite")
    execution_path = HERE / "pilot" / "execution.json"
    marker = json_read(execution_path)
    demand(marker["mode"] == "pilot" and marker["status"] in ("complete_passed", "failed"), "terminal gate closed")
    demand(marker["status"] == "complete_passed", "pilot failed; inspect preserved failure without retry")
    demand(marker["schema_version"] == 1 and marker["completed_traces"] == 12, "completion counts")
    demand(marker["completed_cells"] == [], "pilot contains completed scientific cells")
    demand(marker["official_test_opened"] is False and marker["validation_or_accuracy_computed"] is False, "outcome-access scope")
    demand(marker["all_gates_passed"] is True and marker["warmup_checks_passed"] is True, "pilot invariant flags")
    demand(not any(key in marker for key in ("test_data_artifacts", "test_evaluations", "runs", "checkpoints")), "unexpected scientific artifacts")
    demand(set(marker["source_sha256"]) == SOURCES, "exact scientific-source membership")
    revision = marker["repository_revision"]
    demand(re.fullmatch("[0-9a-f]{40}", revision), "invalid execution revision")
    source_records = {}
    for name, expected in marker["source_sha256"].items():
        hash_string(expected)
        demand(digest(ROOT / name) == expected, "current source mismatch " + name)
        for commit in (FREEZE, revision):
            demand(hashlib.sha256(git("show", f"{commit}:{name}")).hexdigest() == expected,
                   "committed source mismatch " + name)
        source_records[name] = expected
    subprocess.run(["git", "merge-base", "--is-ancestor", FREEZE, revision], cwd=ROOT, check=True)
    commits = {commit: git("show", "-s", "--format=%cI", commit).decode().strip() for commit in (FREEZE, revision)}
    for value in commits.values():
        demand(timestamp(value) <= timestamp(marker["started_utc"]), "source commit after execution")
    decision = PREFIX + "pilot-decision.md"
    demand(hashlib.sha256(git("show", f"{revision}:{decision}")).hexdigest() == digest(ROOT / decision), "pilot authority not committed at execution")
    changed = git("diff", "--name-only", FREEZE, revision).decode().splitlines()
    demand(not set(changed) & SOURCES, "scientific files changed between freeze and pilot launch")
    bulk = Path(marker["bulk_root"])
    demand(bulk.resolve().parent == Path("/tmp/spectral-experiment-artifacts").resolve(), "wrong bulk root")
    mount = json.loads(subprocess.check_output(["findmnt", "-J", "-T", str(bulk), "-o", "TARGET,SOURCE,UUID"], text=True))["filesystems"][0]
    demand(mount == marker["bulk_mount"] and mount["target"] == "/private-artifacts/storage"
           and mount["uuid"] == "00000000-0000-4000-8000-000000000000", "bulk mount binding")
    demand(len(marker["plans"]) == 1 and marker["plans"][0]["seed"] == 9880, "development plan identity")
    demand(len(marker["training_data_artifacts"]) == 2 and {Path(a["path"]).name for a in marker["training_data_artifacts"]}
           == {"train-images-idx3-ubyte", "train-labels-idx1-ubyte"}, "training data membership")
    artifacts = marker["training_data_artifacts"] + marker["plans"] + [marker["timing_and_invariants"]]
    bindings = {}
    for item in artifacts:
        path = Path(item["path"])
        demand(path.is_absolute() and path.is_file() and str(path) not in bindings, "artifact path/uniqueness")
        actual = {"size_bytes": path.stat().st_size, "sha256": digest(path)}
        demand(actual["size_bytes"] == item["size_bytes"] and actual["sha256"] == item["sha256"], "artifact byte binding")
        bindings[str(path)] = actual
    local_files = sorted(str(path.resolve()) for path in bulk.iterdir())
    expected_files = sorted([marker["plans"][0]["path"], marker["timing_and_invariants"]["path"]])
    demand(local_files == expected_files, "unbound or extra bulk artifacts")
    bulk_bytes = sum(Path(path).stat().st_size for path in local_files)
    demand(bulk_bytes == marker["artifact_total_bytes"] and bulk_bytes <= 1024**3, "artifact total/cap")
    demand(not (HERE / "pilot" / "failure.json").exists(), "unexpected failure record on successful attempt")
    reports = json_read(marker["timing_and_invariants"]["path"])
    demand(len(reports) == 6 and {row["arm"] for row in reports} == set(ARMS), "six distinct arm reports")
    traces, timings = {}, []
    counts = dict(trajectory_digest_entries=0, on_off_trajectory_pairs=0,
                  warmup_row_records=0, warmup_raw_applied_pairs=0,
                  warmup_parameter_trajectory_links=0, cross_arm_warmup_row_pairs=0,
                  observer_digest_entries=0, cross_arm_observer_history_pairs=0,
                  on_off_observer_history_pairs=0, mandatory_delivery_gates=0, optional_state_checks=0)
    allowed_trace = {"schema_version", "run_key", "seed", "replacement_probability", "arm", "steps", "instrumented",
                     "all_invariant_gates_passed", "delivery_gate_steps", "measurement_state_checks", "elapsed_seconds",
                     "step_elapsed_seconds", "estimation_rank_by_step", "repair_count_by_step", "trajectory_parameter_sha256",
                     "warmup_trajectory_hashes", "warmup_observer_hashes", "warmup_core_sha256", "warmup_observer_sha256",
                     "resources", "warmup_checks_passed"}
    for report in reports:
        arm = report["arm"]
        demand(set(report) == {"arm", "trajectory_bitwise_identical", "final_state_bitwise_identical", "warmup_checks_passed",
                               "uninstrumented", "instrumented"}, "pilot report schema")
        for flag in ("trajectory_bitwise_identical", "final_state_bitwise_identical", "warmup_checks_passed"):
            demand(report[flag] is True, "recorded invariant failure")
        for name, enabled in (("uninstrumented", False), ("instrumented", True)):
            row = report[name]
            demand(set(row) == allowed_trace, "unexpected outcome/other trace field")
            demand((row["seed"], row["replacement_probability"], row["arm"], row["steps"], row["instrumented"])
                   == (9880, .9, arm, 220, enabled), "trace identity")
            demand(row["run_key"] == f"seed9880-noise0.9-{arm}" and row["schema_version"] == 1, "trace key/schema")
            demand(row["all_invariant_gates_passed"] is True and row["warmup_checks_passed"] is True, "trace gate flag")
            demand(row["delivery_gate_steps"] == 220 and row["measurement_state_checks"] == (5 if enabled else 0), "trace gate count")
            counts["mandatory_delivery_gates"] += row["delivery_gate_steps"]
            counts["optional_state_checks"] += row["measurement_state_checks"]
            history, ranks, repairs, seconds = (row[key] for key in ("trajectory_parameter_sha256", "estimation_rank_by_step",
                                                                      "repair_count_by_step", "step_elapsed_seconds"))
            demand(len(history) == len(ranks) == len(repairs) == len(seconds) == 220, "full pilot histories")
            for value in history:
                hash_string(value)
                counts["trajectory_digest_entries"] += 1
            demand(all(type(value) is int and 0 <= value <= 32 for value in ranks), "stored rank domain")
            demand(all(type(value) is int and value >= 0 for value in repairs)
                   and repairs == sorted(repairs), "repair counter domain/monotonicity")
            demand(all(type(value) in (int, float) and math.isfinite(value) and value >= 0 for value in seconds), "finite timing")
            demand(sum(seconds) <= row["elapsed_seconds"] + 1e-6, "step timing exceeds full trace time")
            resource(row["resources"])
            warm = row["warmup_trajectory_hashes"]
            demand(len(warm) == 100, "warmup cardinality")
            for index, witness in enumerate(warm):
                demand(set(witness) == {"step", "parameters", "raw_gradient", "applied_gradient"}
                       and witness["step"] == index + 1, "warmup witness shape")
                for key in ("parameters", "raw_gradient", "applied_gradient"):
                    hash_string(witness[key])
                demand(witness["parameters"] == history[index], "warmup parameter link")
                demand(witness["raw_gradient"] == witness["applied_gradient"], "warmup raw/applied equality")
                for key in ("warmup_row_records", "warmup_raw_applied_pairs", "warmup_parameter_trajectory_links"):
                    counts[key] += 1
            hash_string(row["warmup_core_sha256"])
            observer = row["warmup_observer_hashes"]
            if arm == "adamw":
                demand(ranks == repairs == [0] * 220 and observer == [] and row["warmup_observer_sha256"] is None, "baseline fake observer")
            else:
                demand(len(observer) == 100 and observer[-1] == row["warmup_observer_sha256"], "observer warmup link")
                for value in observer:
                    hash_string(value)
                    counts["observer_digest_entries"] += 1
            traces[arm, enabled] = row
            timings.append({"arm": arm, "instrumented": enabled, "elapsed_seconds": row["elapsed_seconds"],
                            "summed_step_seconds": sum(seconds), "warmup100_seconds": sum(seconds[:100]),
                            "active101_220_mean_seconds": statistics.fmean(seconds[100:]),
                            "active101_220_median_seconds": statistics.median(seconds[100:]),
                            "rank_minimum": min(ranks), "rank_maximum": max(ranks), "rank_final": ranks[-1],
                            "repair_steps": [i + 1 for i, value in enumerate(repairs) if value > (repairs[i - 1] if i else 0)],
                            "repair_count_final": repairs[-1]})
        for key in ("trajectory_parameter_sha256", "warmup_trajectory_hashes", "warmup_core_sha256",
                    "warmup_observer_hashes", "warmup_observer_sha256", "estimation_rank_by_step", "repair_count_by_step"):
            demand(report["uninstrumented"][key] == report["instrumented"][key], "on/off mismatch " + key)
        counts["on_off_trajectory_pairs"] += 220
        counts["on_off_observer_history_pairs"] += len(report["instrumented"]["warmup_observer_hashes"])
    for enabled in (False, True):
        base, observer = traces["adamw", enabled], traces["current32", enabled]
        for arm in ARMS[1:]:
            row = traces[arm, enabled]
            demand(row["warmup_trajectory_hashes"] == base["warmup_trajectory_hashes"]
                   and row["warmup_core_sha256"] == base["warmup_core_sha256"], "cross-arm core/warmup mismatch")
            counts["cross_arm_warmup_row_pairs"] += 100
            if arm != "current32":
                demand(row["warmup_observer_hashes"] == observer["warmup_observer_hashes"]
                       and row["warmup_observer_sha256"] == observer["warmup_observer_sha256"], "cross-arm observer mismatch")
                counts["cross_arm_observer_history_pairs"] += 100
    resource(marker["resources"])
    environment = marker["environment"]
    demand(environment["cpu_threads"] == 1 and environment["deterministic_algorithms"] is True
           and environment["tf32"] is False and environment["cudnn_benchmark"] is False
           and environment["cublas_workspace"] == ":4096:8" and environment["foreach"] is False
           and environment["fused"] is False, "numerical settings")
    occupancy = marker["occupancy_before"]
    demand("RTX 3090" in occupancy["gpu"] and float(occupancy["gpu"].rsplit(",", 1)[1]) >= 8192
           and occupancy["available_ram_bytes"] >= 16 * 1024**3, "recorded launch resource gate")
    for line in occupancy["compute_processes"].splitlines():
        fields = [piece.strip() for piece in line.split(",")]
        demand(int(fields[0]) == occupancy["own_pid"] or Path(fields[1]).name in
               {"gnome-remote-desktop-daemon", "stremio", "gnome-shell", "Xorg"}, "recorded foreign compute job")
    log_path = HERE / "pilot-run.log"
    log = log_path.read_text()
    demand("PILOT_PROCESS_EXIT=0" in log and 'COMMAND_EXIT_CODE="0"' in log, "successful process-exit evidence")
    demand("Traceback" not in log and "--full" not in log and "--pilot --development-go" in log, "log launch/failure scope")
    progress = [json.loads(line) for line in log.splitlines() if line.startswith("{")]
    expected_progress = {(arm, enabled, step) for arm in ARMS for enabled in (False, True) for step in (100, 200)}
    demand(len(progress) == 24 and {(row["arm"], row["instrumented"], row["step"]) for row in progress} == expected_progress,
           "exact pilot progress log coverage")
    for row in progress:
        demand(set(row) == {"run_key", "seed", "replacement_probability", "arm", "step", "pilot", "instrumented", "elapsed_seconds"}
               and row["pilot"] is True and row["seed"] == 9880 and row["replacement_probability"] == .9,
               "unexpected outcome/scope in progress log")
        demand(0 < row["elapsed_seconds"] < traces[row["arm"], row["instrumented"]]["elapsed_seconds"], "progress timing bound")
    wall_elapsed = (timestamp(marker["completed_utc"]) - timestamp(marker["started_utc"])).total_seconds()
    demand(abs(wall_elapsed - marker["resources"]["elapsed_seconds"]) < .01, "wall/monotonic timing discrepancy")
    full_loop_seconds = 6 * sum(row["warmup100_seconds"] + 1900 * row["active101_220_mean_seconds"]
                                for row in timings if row["instrumented"])
    result = {"status": "runtime_invariance_evidence_passed", "started_utc": started,
              "completed_utc": datetime.now(timezone.utc).isoformat(), "audit_elapsed_seconds": time.perf_counter() - began,
              "audit_source_sha256": digest(Path(__file__)), "pilot_execution_sha256": digest(execution_path),
              "pilot_log_sha256": digest(log_path), "pilot_decision_sha256": digest(ROOT / decision),
              "scientific_source_sha256": source_records, "commit_chronology": commits,
              "freeze_to_execution_changed_paths": changed, "verified_artifacts": bindings,
              "bulk_files": local_files, "bulk_bytes": bulk_bytes, "counts": counts, "timings": timings,
              "pilot_utc_elapsed_seconds": wall_elapsed, "pilot_resources": marker["resources"],
              "process_elapsed_seconds": float(re.search(r"PILOT_PROCESS_ELAPSED_SECONDS=([0-9.]+)", log)[1]),
              "instrumented_36_run_loop_extrapolation_seconds": full_loop_seconds,
              "extrapolation_excludes": ["validation and training/test scoring", "full-run checkpoint and raw-JSON I/O",
                                         "later-trajectory changes in per-step cost", "source/artifact hashes and startup overhead"],
              "independence": "Author wrote full-study summarizer, not runner/pilot; no local module imported here.",
              "limits": ["No new-seed or fresh-context scientific re-execution; no learning verdict.",
                         "Digest sequences compared, not rehashed from absent per-step tensors.",
                         "Full final/core state equalities are source/live-assertion evidence; complete tensors were not retained.",
                         "Numerical gate counts are preserved execution evidence, not recomputation of omitted gradient metrics."]}
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(output), "sha256": digest(output), "status": result["status"],
                      "counts": counts, "full_loop_extrapolation_seconds": full_loop_seconds}))


if __name__ == "__main__":
    main()
