#!/usr/bin/env python3
"""Independent post-run audit; imports neither harness nor result summarizer.

Requested before iteration004 confirmation; prepared without inspecting its
outcomes. DO NOT execute before parent reports completion. The run gate also
requires complete12/48 metadata. CPU checkpoint
replay tolerances are fixed prospectively: absolute CE error <=5e-5 and accuracy
disagreement <=one example per evaluated set. Every observed error is retained.
No training or GPU calls occur. Stored scalar arithmetic can be reaggregated;
unsaved training-gradient vectors, optimizer moments and trajectories are not
replayed. New audit outputs are written once; source artifacts stay unchanged.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import struct
import subprocess
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
RESULTS = HERE / "results"
DATA = Path("data/MNIST/raw")
SEEDS = (3, 4, 5)
ARMS = ("adamw", "estimate32_project32", "estimate128_project32", "scalar32_norm")
CHECKPOINTS = ("final", "min_val_ce", "max_val_accuracy", "warmup100")
WINDOWS = {"all": (101, 2000), "early": (101, 500), "late": (1501, 2000)}
CONTRASTS = ((ARMS[1], ARMS[0]), (ARMS[1], ARMS[3]), (ARMS[2], ARMS[1]),
             (ARMS[2], ARMS[0]), (ARMS[2], ARMS[3]), (ARMS[3], ARMS[0]))
CE_REPLAY_ATOL = 5e-5
ACCURACY_REPLAY_EXAMPLES = 1
SCALAR_ATOL = 1e-11


def utc():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, description):
    if not condition:
        raise AssertionError(description)


def close(a, b):
    if a is None or b is None:
        return a is b
    return math.isfinite(a) and math.isfinite(b) and math.isclose(a, b, rel_tol=1e-11, abs_tol=SCALAR_ATOL)


def evaluation_valid(row, count):
    require(row["count"] == count, "complete evaluation count")
    require(math.isfinite(row["cross_entropy"]) and row["cross_entropy"] >= 0, "finite nonnegative CE")
    require(math.isfinite(row["accuracy"]) and 0 <= row["accuracy"] <= 1, "valid accuracy")
    require(abs(row["accuracy"] * count - round(row["accuracy"] * count)) < 1e-8, "accuracy represents integer correct count")


def recompute_selectors(run):
    trajectory = run["validation_trajectory"]
    require([r["step"] for r in trajectory] == list(range(0, 2001, 100)), "full validation grid")
    best_ce, best_accuracy = math.inf, -math.inf
    selected = {"final": 2000, "warmup100": 100}
    for row in trajectory:
        evaluation_valid(row, 5000)
        if row["cross_entropy"] < best_ce:
            best_ce, selected["min_val_ce"] = row["cross_entropy"], row["step"]
        if row["accuracy"] > best_accuracy:
            best_accuracy, selected["max_val_accuracy"] = row["accuracy"], row["step"]
    require(selected == run["checkpoint_steps"], "four independent fixed checkpoint choices and earliest strict ties")
    return {r["step"]: r for r in trajectory}


def validate_stored_geometry(run):
    require(run["steps"] == 2000 and [r["step"] for r in run["steps_raw"]] == list(range(1, 2001)), "complete ordered step rows")
    require(run["instrumented"] and run["all_invariant_gates_passed"] and run["warmup_checks_passed"], "recorded runtime gates")
    require(run["measurement_state_checks"] == 42, "all scheduled measurement-state checks")
    require(len(run["estimation_rank_by_step"]) == 2000, "full rank trajectory")
    require(len(run["step_elapsed_seconds"]) == 2000 and all(math.isfinite(t) and t >= 0 for t in run["step_elapsed_seconds"]), "finite complete timings")
    width = {ARMS[0]: 0, ARMS[1]: 32, ARMS[2]: 128, ARMS[3]: 32}[run["arm"]]
    maxima = {"norm_matching_absolute_error": 0., "scalar_collinearity_relative_error": 0.,
              "scalar_alpha_recomputed_error": 0., "update_cosine_recomputed_error": 0.}
    for row in run["steps_raw"]:
        step, rank = row["step"], row["estimation_rank"]
        require(0 <= rank <= width and rank == run["estimation_rank_by_step"][step - 1], "recorded estimation cap/trajectory")
        identity = run["arm"] == ARMS[0] or step <= 100 or rank == 0
        expected_operator = "identity" if identity else ("scalar_identity" if run["arm"] == ARMS[3] else "hard_projection")
        require(row["operator"] == expected_operator, "policy operator/warmup")
        require(row["candidate_rank"] == (50890 if identity else min(32, rank)), "candidate versus estimation rank")
        energies = [row[k + "_squared_norm"] for k in ("raw", "candidate", "applied")]
        require(all(math.isfinite(v) and v >= 0 for v in energies), "finite nonnegative gradient energies")
        g, c, a = [math.sqrt(v) for v in energies]
        require(close(abs(a - c), row["norm_matching_absolute_error"]), "gradient norm error arithmetic")
        require(close(1e-6 * max(g, c) + 1e-12, row["norm_matching_tolerance"]), "gradient norm tolerance arithmetic")
        maxima["norm_matching_absolute_error"] = max(maxima["norm_matching_absolute_error"], row["norm_matching_absolute_error"])
        cosine = row["raw_applied_cosine"]
        require((cosine is None) == (g * a == 0), "gradient cosine null mask")
        if cosine is not None:
            require(math.isfinite(cosine) and abs(cosine) <= 1 + 1e-6, "gradient cosine range")
        if identity:
            require(energies[0] == energies[1] == energies[2] and row["alpha"] == 1., "identity policy energies/alpha")
        elif run["arm"] == ARMS[3]:
            alpha = c / g if g else 0.
            require(close(alpha, row["alpha"]) and 0 <= alpha <= 1.005, "scalar alpha reconstructed from stored energies")
            maxima["scalar_alpha_recomputed_error"] = max(maxima["scalar_alpha_recomputed_error"], abs(alpha - row["alpha"]))
        else:
            require(row["alpha"] is None and energies[1] == energies[2], "hard policy has no scalar alpha")
        if run["arm"] == ARMS[3]:
            require(row["norm_matching_absolute_error"] <= row["norm_matching_tolerance"], "recorded scalar norm gate")
            collinearity = row["scalar_collinearity_relative_error"]
            require(math.isfinite(collinearity) and 0 <= collinearity <= 1e-6, "recorded scalar collinearity gate")
            require(cosine is None or abs(cosine - 1) <= 1e-6, "recorded scalar cosine gate")
            maxima["scalar_collinearity_relative_error"] = max(maxima["scalar_collinearity_relative_error"], collinearity)
        else:
            require(row["scalar_collinearity_relative_error"] is None, "non-scalar collinearity remains undefined")
        for kind in ("total", "decay_subtracted"):
            update = row[kind]
            require(not any(k in update for k in ("identity_closure_residual", "leakage_squared_fraction", "in_subspace_contribution")), "no invalid scalar orthoprojector metric")
            energy = update["squared_norm"]
            require(math.isfinite(energy) and energy >= 0, "finite update energy")
            d = math.sqrt(energy)
            for component, gradient_norm in (("raw", g), ("applied", a)):
                entry = update[component + "_gradient_dot_update"]
                value, tolerance = entry["value"], entry["tolerance"]
                require(math.isfinite(value) and close(tolerance, 1e-6 * gradient_norm * d + 1e-14), "direction tolerance rederived from energies")
                require(entry["sign"] == int(value > tolerance) - int(value < -tolerance), "direction sign rederived")
                denom = gradient_norm * d
                expected_cosine = value / denom if denom else None
                recorded_cosine = update[component + "_gradient_update_cosine"]
                require(close(expected_cosine, recorded_cosine), "update cosine rederived from dot and norms")
                if expected_cosine is not None:
                    require(abs(expected_cosine) <= 1 + 1e-6, "update cosine Cauchy bound")
                    maxima["update_cosine_recomputed_error"] = max(maxima["update_cosine_recomputed_error"], abs(expected_cosine - recorded_cosine))
    return maxima


def independent_summary(run, validation):
    metrics, counts = {}, {}
    def collect(name, values, steps):
        values_present = [float(v) for v in values if v is not None]
        require(all(math.isfinite(v) for v in values_present), "finite aggregation values")
        metrics[name] = float(np.mean(values_present)) if values_present else None
        counts[name] = {"finite": len(values_present), "null": len(values) - len(values_present),
                        "null_steps": [s for s, v in zip(steps, values) if v is None]}
    for name in CHECKPOINTS:
        evaluation_valid(run["test"][name], 10000)
        step = run["checkpoint_steps"][name]
        metrics[f"checkpoint.{name}.step"] = step
        for field in ("accuracy", "cross_entropy"):
            metrics[f"test.{name}.{field}"] = run["test"][name][field]
            metrics[f"validation.{name}.{field}"] = validation[step][field]
            if name != "warmup100":
                metrics[f"test.{name}.minus_warmup100.{field}"] = run["test"][name][field] - run["test"]["warmup100"][field]
    for name in ("final_training_clean", "final_training_noisy", "final_validation"):
        evaluation_valid(run[name], 5000)
        for field in ("accuracy", "cross_entropy"):
            metrics[f"learning.{name}.{field}"] = run[name][field]
    for window, (lo, hi) in WINDOWS.items():
        rows = [row for row in run["steps_raw"] if lo <= row["step"] <= hi]
        require(len(rows) == hi - lo + 1, "all scheduled steps in window denominator")
        steps = [row["step"] for row in rows]
        prefix = "gradient." + window
        for component in ("raw", "candidate", "applied"):
            energies = [row[component + "_squared_norm"] for row in rows]
            collect(prefix + "." + component + "_squared_norm", energies, steps)
            collect(prefix + "." + component + "_norm", [math.sqrt(v) for v in energies], steps)
        for component in ("candidate", "applied"):
            collect(prefix + "." + component + "_raw_norm_ratio", [math.sqrt(row[component + "_squared_norm"] / row["raw_squared_norm"]) if row["raw_squared_norm"] else None for row in rows], steps)
        for field in ("alpha", "raw_applied_cosine", "scalar_collinearity_relative_error", "norm_matching_absolute_error"):
            collect(prefix + "." + field, [row[field] for row in rows], steps)
        metrics[prefix + ".max_norm_matching_absolute_error"] = max(row["norm_matching_absolute_error"] for row in rows)
        collinear = [row["scalar_collinearity_relative_error"] for row in rows if row["scalar_collinearity_relative_error"] is not None]
        metrics[prefix + ".max_scalar_collinearity_relative_error"] = max(collinear) if collinear else None
        for kind in ("total", "decay_subtracted"):
            prefix_update = "update." + window + "." + kind
            updates = [row[kind] for row in rows]
            collect(prefix_update + ".squared_norm", [u["squared_norm"] for u in updates], steps)
            collect(prefix_update + ".norm", [math.sqrt(u["squared_norm"]) for u in updates], steps)
            for component in ("raw", "applied"):
                field = component + "_gradient_dot_update"
                collect(prefix_update + "." + field, [u[field]["value"] for u in updates], steps)
                collect(prefix_update + "." + component + "_gradient_ascent_frequency", [int(u[field]["value"] > u[field]["tolerance"]) for u in updates], steps)
                field = component + "_gradient_update_cosine"
                collect(prefix_update + "." + field, [u[field] for u in updates], steps)
    return {"seed": run["seed"], "arm": run["arm"], "replacement_probability": .9,
            "realized_incorrect_fraction": run["realized_incorrect_fraction"],
            "realized_replacement_fraction": run["realized_replacement_fraction"], "metrics": metrics, "counts": counts}


def independently_check_summary(derived, saved):
    saved_runs = {(r["seed"], r["arm"]): r for r in saved["seed_summaries"]}
    require(len(saved_runs) == len(saved["seed_summaries"]) == 12, "twelve unique seed summaries")
    index = {(r["seed"], r["arm"]): r for r in derived}
    largest = 0.
    for key, expected in index.items():
        observed = saved_runs[key]
        require(expected["counts"] == observed["counts"], "all null counts and masks")
        require(expected["metrics"].keys() == observed["metrics"].keys(), "all summary metrics covered")
        for name, value in expected["metrics"].items():
            require(close(value, observed["metrics"][name]), "seed summary metric " + name)
            if value is not None:
                largest = max(largest, abs(value - observed["metrics"][name]))
    expected_keys = {(a, b, m) for a, b in CONTRASTS for m in derived[0]["metrics"]}
    observed = {(r["treatment"], r["control"], r["metric"]): r for r in saved["paired_contrasts"]}
    require(set(observed) == expected_keys and len(observed) == len(saved["paired_contrasts"]), "complete unique paired contrast grid")
    for (treatment, control, name), row in observed.items():
        differences, unavailable = [], []
        for seed in SEEDS:
            left, right = index[seed, treatment], index[seed, control]
            a, b = left["metrics"][name], right["metrics"][name]
            if a is None or b is None or left["counts"].get(name) != right["counts"].get(name):
                unavailable.append(seed)
            else:
                differences.append({"seed": seed, "difference": a - b})
        require(row["unavailable_seeds"] == unavailable and len(row["paired_differences"]) == len(differences), "paired availability/null masks")
        for a, b in zip(differences, row["paired_differences"]):
            require(a["seed"] == b["seed"] and close(a["difference"], b["difference"]), "individual paired difference")
        values = [v["difference"] for v in differences]
        complete = len(values) == 3
        expected = {"mean": statistics.fmean(values), "median": statistics.median(values),
                    "minimum": min(values), "maximum": max(values), "sample_sd_descriptive": statistics.stdev(values)} if complete else dict.fromkeys(("mean", "median", "minimum", "maximum", "sample_sd_descriptive"))
        require(row["all_three_pairs_available"] == complete, "complete three-seed contrast gate")
        primary = name == "test.max_val_accuracy.accuracy" and (treatment, control) in CONTRASTS[:2]
        require(row["co_primary"] == primary, "two co-primary contrasts")
        for field, value in expected.items():
            require(close(value, row[field]), "paired aggregate " + field)
            if value is not None:
                largest = max(largest, abs(value - row[field]))
    return {"seed_metric_count_each": len(derived[0]["metrics"]), "paired_contrasts": len(observed),
            "largest_scalar_reaggregation_abs_error": largest, "all_metrics_counts_masks_and_contrasts_match": True}


def idx(path, images):
    raw = path.read_bytes()
    if images:
        magic, count, rows, cols = struct.unpack(">4I", raw[:16])
        require(magic == 2051 and rows == cols == 28 and len(raw) == 16 + count * 784, "image IDX schema")
        array = np.frombuffer(raw, dtype=np.uint8, offset=16).copy().reshape(count, 784)
        return torch.from_numpy(array).float() / 255
    magic, count = struct.unpack(">2I", raw[:8])
    require(magic == 2049 and len(raw) == 8 + count, "label IDX schema")
    return torch.from_numpy(np.frombuffer(raw, dtype=np.uint8, offset=8).copy()).long()


@torch.no_grad()
def evaluate_state(state, x, y):
    total_loss, correct = 0., 0
    for start in range(0, len(y), 512):
        hidden = torch.relu(x[start:start + 512] @ state["0.weight"].T + state["0.bias"])
        logits = hidden @ state["2.weight"].T + state["2.bias"]
        targets = y[start:start + 512]
        total_loss += float(torch.nn.functional.cross_entropy(logits, targets, reduction="sum"))
        correct += int((logits.argmax(1) == targets).sum())
    return {"cross_entropy": total_loss / len(y), "accuracy": correct / len(y), "correct": correct, "count": len(y)}


def independent_plan(seed):
    generator = lambda stream: np.random.default_rng(np.random.SeedSequence([20260906, 3, seed, stream]))
    permutation = generator(0).permutation(60000)
    return {"seed": seed, "initialization_seed": int(generator(3).integers(0, 2**32, dtype=np.uint32)),
            "train_indices": permutation[:5000], "validation_indices": permutation[5000:10000],
            "auxiliary_indices": permutation[10000:15000], "replacement_uniforms": generator(1).random(5000),
            "replacement_digits": generator(2).integers(0, 10, size=5000),
            "training_batches": generator(4).integers(0, 5000, size=(2000, 64)),
            "primary_probe_batches": generator(5).integers(0, 5000, size=(2000, 256)),
            "auxiliary_probe_batches": generator(6).integers(0, 5000, size=(2000, 256))}


def audit(execution, summary, report, expected_revision):
    revision = execution["repository_revision"]
    require(revision == expected_revision, "explicit expected freeze revision")
    git = lambda *args: subprocess.check_output(["git", "-C", str(ROOT), *args])
    commit_time = git("show", "-s", "--format=%cI", revision).decode().strip()
    require(datetime.fromisoformat(commit_time) <= datetime.fromisoformat(execution["started_utc"]), "source commit predates execution")
    chronology = [execution[k] for k in ("started_utc", "all_training_completed_utc", "test_first_loaded_utc", "completed_utc")]
    require(chronology == sorted(chronology), "all twelve training decisions before test access")
    report["execution_revision"] = revision
    report["source_checks"] = []
    for relative, digest in execution["source_sha256"].items():
        path = (ROOT / relative).resolve()
        require(path.is_relative_to(ROOT.resolve()), "source path remains inside repository")
        require(sha(path) == digest, "recorded/current source hash " + relative)
        require(hashlib.sha256(git("show", revision + ":" + relative)).hexdigest() == digest, "committed source bytes " + relative)
        report["source_checks"].append({"path": relative, "sha256": digest, "matches_commit_and_current": True})
    pilot = read(HERE / "pilot" / "execution.json")
    require(pilot["status"] == "complete_passed" and pilot["completed_traces"] == 8, "passing eight-trace pilot")
    require(pilot["source_sha256"] == execution["source_sha256"], "same source as passing pilot")
    require(pilot["training_data_artifacts"] == execution["training_data_artifacts"], "same training artifacts as pilot")
    artifacts = execution["training_data_artifacts"] + execution["test_data_artifacts"] + execution["plans"] + execution["checkpoints"]
    require(len(execution["plans"]) == 3 and len(execution["checkpoints"]) == 12, "all plans and checkpoint bundles")
    report["artifact_checks"] = []
    for item in artifacts:
        path = Path(item["path"]).resolve()
        require(path.is_relative_to(RESULTS.resolve()) or path.parent == DATA.resolve(), "bounded artifact path")
        require(path.stat().st_size == item["size_bytes"] and sha(path) == item["sha256"], "artifact bytes and size " + str(path))
        report["artifact_checks"].append({**item, "matches": True})
    require(summary["source_sha256"] == sha(HERE / "summarize_results.py"), "summary implementation provenance")
    for item in summary["inputs"]:
        require(sha(Path(item["path"])) == item["sha256"], "summary input provenance")
    train_x, train_y = idx(DATA / "train-images-idx3-ubyte", True), idx(DATA / "train-labels-idx1-ubyte", False)
    test_x, test_y = idx(DATA / "t10k-images-idx3-ubyte", True), idx(DATA / "t10k-labels-idx1-ubyte", False)
    report["raw_sources"], report["plan_checks"], report["warmup_checks"], report["checkpoint_replays"] = [], [], [], []
    derived, recorded = [], {}
    for seed in SEEDS:
        plan = independent_plan(seed)
        with np.load(RESULTS / f"plan-seed{seed}.npz", allow_pickle=False) as saved_plan:
            require(set(saved_plan.files) == set(plan), "plan schema")
            for name, value in plan.items():
                require(np.array_equal(value, saved_plan[name]), "independent seeded plan reconstruction " + name)
        report["plan_checks"].append({"seed": seed, "all_arrays_reconstructed_exactly": True})
        clean = train_y[plan["train_indices"]]
        replaced = torch.from_numpy(plan["replacement_uniforms"] < .9)
        noisy = torch.where(replaced, torch.from_numpy(plan["replacement_digits"]), clean)
        replacement_fraction = float(replaced.float().mean())
        incorrect_fraction = float((noisy != clean).float().mean())
        vx, vy = train_x[plan["validation_indices"]], train_y[plan["validation_indices"]]
        common_warmup_state = None
        for arm in ARMS:
            tag = f"seed{seed}-{arm}"
            report["active_context"] = {"tag": tag, "phase": "stored_scalar_reaggregation"}
            path = RESULTS / (tag + ".json")
            run = read(path)
            require((run["seed"], run["arm"], run["replacement_probability"]) == (seed, arm, .9), "run identity")
            require(run["realized_replacement_fraction"] == replacement_fraction and run["realized_incorrect_fraction"] == incorrect_fraction, "fixed corruption reconstruction")
            require(set(run["test"]) == set(CHECKPOINTS), "four test checkpoint choices")
            report["raw_sources"].append({"path": str(path), "sha256": sha(path)})
            validation = recompute_selectors(run)
            recorded[seed, arm] = run
            maxima = validate_stored_geometry(run)
            report.setdefault("geometry_checks", []).append({"tag": tag, "stored_arithmetic_and_gates_pass": True, "maxima": maxima})
            derived.append(independent_summary(run, validation))
            require([r["step"] for r in run["warmup_trajectory_hashes"]] == list(range(1, 101)), "complete warmup hashes")
            if arm != ARMS[0]:
                baseline = recorded[seed, ARMS[0]]
                require(run["warmup_trajectory_hashes"] == baseline["warmup_trajectory_hashes"], "all100 parameter/raw/applied-gradient warmup hashes match")
                require(run["warmup_core_sha256"] == baseline["warmup_core_sha256"], "recorded warmup core/moment fingerprint matches")
                require(run["test"]["warmup100"] == baseline["test"]["warmup100"], "common warmup test metrics")
            if arm == ARMS[3]:
                hard = recorded[seed, ARMS[1]]
                require(len(run["warmup_observer_hashes"]) == 100 and run["warmup_observer_hashes"] == hard["warmup_observer_hashes"], "all100 width32 observer warmup hashes match")
                require(run["warmup_observer_sha256"] == hard["warmup_observer_sha256"], "warmup observer final hash matches")
            cp = Path(run["checkpoint_path"]).resolve()
            require(cp == (RESULTS / (tag + "-checkpoints.pt")).resolve(), "checkpoint condition path")
            bundle = torch.load(cp, map_location="cpu", weights_only=True)
            require(set(bundle) == set(CHECKPOINTS), "saved four-checkpoint bundle")
            if common_warmup_state is None:
                common_warmup_state = bundle["warmup100"]
            else:
                require(bundle["warmup100"].keys() == common_warmup_state.keys() and all(torch.equal(v, common_warmup_state[k]) for k, v in bundle["warmup100"].items()), "saved warmup parameters bitwise equal across arms")
            for name in CHECKPOINTS:
                state = bundle[name]
                report["active_context"] = {"tag": tag, "checkpoint": name, "phase": "independent_cpu_forward_replay"}
                require(set(state) == {"0.weight", "0.bias", "2.weight", "2.bias"}, "MLP state schema")
                require(sum(v.numel() for v in state.values()) == 50890 and all(v.dtype == torch.float32 and bool(torch.isfinite(v).all()) for v in state.values()), "finite float32 checkpoint parameter count")
                for dataset, x, y, reference in (("test", test_x, test_y, run["test"][name]),
                                                  ("validation", vx, vy, validation[run["checkpoint_steps"][name]])):
                    observed = evaluate_state(state, x, y)
                    ce_error = abs(observed["cross_entropy"] - reference["cross_entropy"])
                    error_examples = abs(observed["correct"] - round(reference["accuracy"] * len(y)))
                    passed = ce_error <= CE_REPLAY_ATOL and error_examples <= ACCURACY_REPLAY_EXAMPLES
                    report["checkpoint_replays"].append({"tag": tag, "checkpoint": name, "dataset": dataset,
                                                         "observed": observed, "recorded": reference,
                                                         "ce_absolute_error": ce_error, "accuracy_error_examples": error_examples,
                                                         "accuracy_absolute_error": abs(observed["accuracy"] - reference["accuracy"]),
                                                         "passed_tolerance": passed})
        report["warmup_checks"].append({"seed": seed, "all100_stored_core_gradient_parameter_hashes_match": True,
                                        "width32_observer_hashes_match": True, "saved_step100_parameters_bitwise_equal": True,
                                        "optimizer_moments": "Recorded matching runtime fingerprints only; full moments were not saved for independent replay."})
    report["independent_seed_summaries"] = derived
    report["summary_check"] = independently_check_summary(derived, summary)
    require(len(report["checkpoint_replays"]) == 96, "48 test and48 validation checkpoint evaluations")
    report["replay_maxima"] = {"ce_absolute_error": max(r["ce_absolute_error"] for r in report["checkpoint_replays"]),
                               "accuracy_error_examples": max(r["accuracy_error_examples"] for r in report["checkpoint_replays"]),
                               "accuracy_absolute_error": max(r["accuracy_absolute_error"] for r in report["checkpoint_replays"])}
    report["status"] = "PASS" if all(r["passed_tolerance"] for r in report["checkpoint_replays"]) else "FAIL_REPLAY_TOLERANCE"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--expected-revision", required=True)
    args = parser.parse_args()
    if not args.run:
        parser.error("Audit execution requires parent completion notice and --run")
    execution = read(RESULTS / "execution.json")
    require((execution.get("mode"), execution.get("status"), execution.get("completed_runs"), execution.get("test_evaluations")) == ("confirmatory", "complete", 12, 48), "GATE CLOSED: do not read outcome/test inputs")
    require(execution["warmup_checks_passed"], "completed warmup gate")
    output_json, output_md = HERE / "audit-results.json", HERE / "audit-results.md"
    require(not output_json.exists() and not output_md.exists(), "refuse audit overwrite")
    report = {"status": "RUNNING", "started_utc": utc(), "audit_source_sha256": sha(Path(__file__)),
              "execution_sha256": sha(RESULTS / "execution.json"), "summary_sha256": sha(HERE / "summary.json"),
              "cpu_only": True, "training_or_new_gradient_replay": False,
              "replay_tolerances_fixed_before_confirmation": {"ce_absolute": CE_REPLAY_ATOL, "accuracy_examples_per_set": ACCURACY_REPLAY_EXAMPLES},
              "stored_scalar_reaggregation_tolerance": {"absolute": SCALAR_ATOL, "relative": 1e-11},
              "python": sys.version, "numpy": np.__version__, "torch": torch.__version__,
              "limits": ["Saved scalar arithmetic and summaries are independently reaggregated; full gradient vectors are not saved/replayed.",
                         "Stored moment fingerprints/runtime invariants are checked for equality, not independently reconstructed from training.",
                         "Checkpoint test and validation metrics are independently replayed on CPU; passing tolerances do not prove every discrepancy is numerical."]}
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    try:
        audit(execution, read(HERE / "summary.json"), report, args.expected_revision)
    except Exception as exc:
        report["status"], report["error"] = "FAIL_PRESERVED", repr(exc)
    report["completed_utc"] = utc()
    report["elapsed_seconds"] = (datetime.fromisoformat(report["completed_utc"]) - datetime.fromisoformat(report["started_utc"])).total_seconds()
    with output_json.open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    replays = report.get("checkpoint_replays", [])
    maximum = report.get("replay_maxima", {})
    lines = ["# Independent iteration-004 completed-result audit", "", f"Verdict: **{report['status']}**.", "",
             "The auditor imports neither the iteration-004 harness nor its summarizer. No training or GPU work occurred.", "",
             f"Checkpoint replays: {len(replays)}. Prespecified tolerance: CE absolute error at most 5e-5; accuracy at most one example per evaluated set.",
             f"Observed maximum CE error: {maximum.get('ce_absolute_error', 'not reached')}; accuracy discrepancy: {maximum.get('accuracy_error_examples', 'not reached')} examples.", "",
             "Audit scope includes source/data/plan/checkpoint provenance, schedules, both strict selectors, common warmup evidence, all seed metrics and paired contrasts; completed checks and any failure are recorded in the JSON.",
             "Stored scalar arithmetic is not a replay of unsaved gradients or optimizer moment trajectories. Saved checkpoint parameters and their evaluation metrics are independently replayed.", "",
             f"[Full audit record](audit-results.json), SHA256 `{sha(output_json)}`."]
    if "error" in report:
        lines += ["", "Preserved failure: " + report["error"]]
    with output_md.open("x") as handle:
        handle.write("\n".join(lines) + "\n")
    print(json.dumps({"status": report["status"], "output": str(output_json), "sha256": sha(output_json),
                      "elapsed_seconds": report["elapsed_seconds"], "checkpoint_replays": len(replays)}))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
