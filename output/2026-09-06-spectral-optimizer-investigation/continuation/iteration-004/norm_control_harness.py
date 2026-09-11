#!/usr/bin/env python3
"""Iteration004, gated separately for development and confirmation."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / "iteration-003" / "neural_harness.py"
spec = importlib.util.spec_from_file_location("iteration003_validated_helpers", PREVIOUS)
h = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h)
torch, np, F = h.torch, h.np, h.F
REPO = h.REPO
ARMS = ("adamw", "estimate32_project32", "estimate128_project32", "scalar32_norm")
WIDTHS = {"adamw": None, "estimate32_project32": 32, "estimate128_project32": 128, "scalar32_norm": 32}
SEEDS, STEPS, PILOT_SEED, PILOT_STEPS = (3, 4, 5), 2000, 9877, 220
CHECKPOINTS = ("final", "min_val_ce", "max_val_accuracy", "warmup100")
ANALYSIS_FILES = ("analysis-plan.md", "summarize_results.py", "test_summary.py")


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(v) for v in value)
    return h.clone_tree(value)


def tree_hash(value):
    digest = hashlib.sha256()
    def visit(node):
        if isinstance(node, torch.Tensor):
            digest.update(b"tensor")
            digest.update(str((str(node.dtype), tuple(node.shape))).encode())
            digest.update(node.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(node, dict):
            digest.update(b"dict")
            for key in sorted(node, key=repr):
                visit(key)
                visit(node[key])
        elif isinstance(node, (tuple, list)):
            digest.update(type(node).__name__.encode())
            for item in node:
                visit(item)
        else:
            digest.update(repr(node).encode())
            digest.update(b"\0")
    visit(value)
    return digest.hexdigest()


def source_paths():
    return [Path(__file__), HERE / "protocol.md", HERE / "design-intent.md", HERE / "test_harness.py",
            *(HERE / name for name in ANALYSIS_FILES), PREVIOUS, REPO / "spectral_filter.py"]


def source_hashes():
    return {str(path.relative_to(REPO)): h.file_hash(path) for path in source_paths()}


def committed_source_gate():
    for path in source_paths():
        committed = subprocess.check_output(["git", "show", f"HEAD:{path.relative_to(REPO).as_posix()}"], cwd=REPO)
        assert h.digest_bytes(committed) == h.file_hash(path), f"Uncommitted source: {path}"


def manifest(mode):
    return {
        "mode": mode, "status": "running", "started_utc": h.utc(),
        "repository_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "source_sha256": source_hashes(),
        "training_data_artifacts": [h.artifact(h.DATA / name) for name in h.TRAINING_FILES],
        "python": sys.version, "torch": torch.__version__, "numpy": np.__version__,
        "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
        "occupancy_before": h.occupancy(), "deterministic_algorithms": True,
        "precision": "float32 model/basis; float64 norm/dot reductions and small eigensolves",
        "torch_threads": torch.get_num_threads(), "tf32": False,
        "checkpoint_names": list(CHECKPOINTS), "replacement_probability": .9,
    }


def scalar_gradient(raw, candidate, active):
    assert bool(torch.isfinite(raw).all()) and bool(torch.isfinite(candidate).all()), "Non-finite scalar-policy inputs"
    if not active:
        return raw.clone(), 1.
    raw_norm = h.norm(raw)
    alpha = 0. if raw_norm == 0 else h.norm(candidate) / raw_norm
    assert math.isfinite(alpha) and 0 <= alpha <= 1.005, f"Invalid scalar alpha {alpha}"
    return raw * alpha, alpha


def transform_gradient(arm, tracker, raw):
    candidate, basis = h.observe_and_project(tracker, raw)
    if arm == "scalar32_norm":
        applied, alpha = scalar_gradient(raw, candidate, basis is not None)
        operator = "scalar_identity" if basis is not None else "identity"
    else:
        applied, alpha = candidate, (1. if basis is None else None)
        operator = "hard_projection" if basis is not None else "identity"
    return applied, candidate, {"operator": operator, "alpha": alpha,
                               "candidate_rank": len(raw) if basis is None else basis.shape[1],
                               "estimation_rank": 0 if tracker is None or tracker.V is None else tracker.V.shape[1]}


def delivered_gradient_metrics(arm, raw, candidate, delivered, policy):
    gn, cn, an = h.norm(raw), h.norm(candidate), h.norm(delivered)
    matching_error = abs(an - cn)
    matching_tolerance = 1e-6 * max(gn, cn) + 1e-12
    cosine = h.ratio(h.dot(raw, delivered), gn * an)
    collinearity = None
    if arm == "scalar32_norm":
        alpha = policy["alpha"]
        collinearity = h.norm(delivered - alpha * raw) / max(gn, an, 1e-30)
        assert matching_error <= matching_tolerance, "Delivered scalar norm does not match candidate"
        assert collinearity <= 1e-6, "Delivered scalar gradient changed direction"
        assert cosine is None or abs(cosine - 1) <= 1e-6, "Delivered scalar cosine differs from one"
    return {
        **policy, "raw_squared_norm": h.dot(raw, raw), "candidate_squared_norm": h.dot(candidate, candidate),
        "applied_squared_norm": h.dot(delivered, delivered), "norm_matching_absolute_error": matching_error,
        "norm_matching_tolerance": matching_tolerance, "raw_applied_cosine": cosine,
        "scalar_collinearity_relative_error": collinearity,
    }


def update_metrics(raw, applied, delta):
    # Deliberately no P*delta, in/out decomposition or orthoprojector identity.
    raw_dot, applied_dot = h.dot(raw, delta), h.dot(applied, delta)
    return {"squared_norm": h.dot(delta, delta),
            "raw_gradient_dot_update": h.sign_class(raw_dot, raw, delta),
            "applied_gradient_dot_update": h.sign_class(applied_dot, applied, delta),
            "raw_gradient_update_cosine": h.ratio(raw_dot, h.norm(raw) * h.norm(delta)),
            "applied_gradient_update_cosine": h.ratio(applied_dot, h.norm(applied) * h.norm(delta))}


def strict_selector_updates(best_ce, best_accuracy, evaluation):
    ce, accuracy = evaluation["cross_entropy"], evaluation["accuracy"]
    assert math.isfinite(ce) and math.isfinite(accuracy) and 0 <= accuracy <= 1
    return {"min_val_ce": ce < best_ce, "max_val_accuracy": accuracy > best_accuracy}


def append_validated_evaluation(validations, step, evaluation, best_ce, best_accuracy):
    # Reject non-finite values before they enter JSON-backed failure context.
    decisions = strict_selector_updates(best_ce, best_accuracy, evaluation)
    validations.append({"step": step, **evaluation})
    return decisions


def warmup_snapshot(model, optimizer, tracker):
    full = h.snapshot(model, optimizer, tracker)
    observer = full.pop("tracker")
    return {"core": cpu_tree(full), "observer": cpu_tree(observer)}


def verify_warmup_pair(reference_result, reference_state, result, state, compare_observer=False):
    assert reference_result["warmup_trajectory_hashes"] == result["warmup_trajectory_hashes"], "Warmup parameter/gradient trajectory differs"
    assert h.equal_tree(reference_state["core"], state["core"]), "Warmup optimizer/core state differs"
    if compare_observer:
        assert reference_result["warmup_observer_hashes"] == result["warmup_observer_hashes"], "Warmup width32 observer trajectory differs"
        assert h.equal_tree(reference_state["observer"], state["observer"]), "Warmup width32 observer state differs"


def train_arm(plan, data, arm, steps, instrumented, pilot, active_context, device="cuda"):
    active_context.update({"seed": plan["seed"], "arm": arm, "step": 0,
                           "pilot": pilot, "instrumented": instrumented, "phase": "model_initialization"})
    model = h.make_model(plan["initialization_seed"], device)
    optimizer = h.make_optimizer(model)
    tracker = h.make_tracker(model, optimizer, WIDTHS[arm])
    rows, validations, timing, ranks = [], [], [], []
    warmup_hashes, observer_hashes, trajectory_hashes = [], [], []
    active_context.update({"steps_raw": rows, "validation_trajectory": validations})
    batches = torch.as_tensor(plan["training_batches"], device=device)
    checkpoints, checkpoint_steps = {}, {}
    best_ce, best_accuracy = float("inf"), float("-inf")
    if not pilot:
        initial = h.evaluate(model, data["vx"], data["vy"])
        append_validated_evaluation(validations, 0, initial, best_ce, best_accuracy)
        best_ce, best_accuracy = initial["cross_entropy"], initial["accuracy"]
        for name in ("min_val_ce", "max_val_accuracy"):
            checkpoints[name], checkpoint_steps[name] = cpu_tree(model.state_dict()), 0
    warm_state = None
    measurement_checks = 0
    if str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    h.synchronize(device)
    started = time.perf_counter()
    for step in range(1, steps + 1):
        active_context.update({"step": step, "phase": "training_gradient"})
        h.synchronize(device)
        begin = time.perf_counter()
        batch = batches[step - 1]
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(data["x"][batch]), data["noisy"][batch])
        loss.backward()
        raw, before_theta = h.flat_grad(model), h.flat_params(model)
        active_context["phase"] = "covariance_and_policy"
        constructed, candidate, policy = transform_gradient(arm, tracker, raw)
        h.set_grad(model, constructed)
        # Inspect what AdamW will actually read, not only the constructed tensor.
        delivered = h.flat_grad(model)
        if arm == "scalar32_norm":
            # Mandatory delivery gates apply even when optional logging is off.
            delivered_gradient_metrics(arm, raw, candidate, delivered, policy)
        optimizer.step()
        after_theta = h.flat_params(model)
        assert bool(torch.isfinite(after_theta).all()), "Non-finite model parameters"
        ranks.append(policy["estimation_rank"])
        if tracker is not None and h.is_probe_step(step):
            assert tracker.orthogonality_error() <= 5e-3, "Nonorthogonal basis"
        if step <= h.WARMUP:
            warmup_hashes.append({"step": step, "parameters": h.tensor_hash(after_theta),
                                  "raw_gradient": h.tensor_hash(raw), "applied_gradient": h.tensor_hash(delivered)})
            if WIDTHS[arm] == 32:
                observer_hashes.append(tree_hash(h.tracker_state(tracker)))
        if step == h.WARMUP:
            warm_state = warmup_snapshot(model, optimizer, tracker)
            if not pilot:
                checkpoints["warmup100"], checkpoint_steps["warmup100"] = cpu_tree(model.state_dict()), step
        if instrumented:
            active_context["phase"] = "measurement"
            check = h.is_probe_step(step)
            state_before = h.snapshot(model, optimizer, tracker) if check else None
            delta = after_theta - before_theta
            measurement = {"step": step, **delivered_gradient_metrics(arm, raw, candidate, delivered, policy),
                           "total": update_metrics(raw, delivered, delta),
                           "decay_subtracted": update_metrics(raw, delivered, h.decay_subtracted_update(delta, before_theta))}
            if check:
                assert h.equal_tree(state_before, h.snapshot(model, optimizer, tracker)), "Measurement mutated training state"
                measurement_checks += 1
            if not pilot:
                rows.append(measurement)
        if pilot:
            trajectory_hashes.append(h.tensor_hash(after_theta))
        if not pilot and step % 100 == 0:
            active_context["phase"] = "validation_selection"
            evaluation = h.evaluate(model, data["vx"], data["vy"])
            decisions = append_validated_evaluation(validations, step, evaluation, best_ce, best_accuracy)
            if decisions["min_val_ce"]:
                best_ce = evaluation["cross_entropy"]
            if decisions["max_val_accuracy"]:
                best_accuracy = evaluation["accuracy"]
            for name, improve in decisions.items():
                if improve:
                    checkpoints[name], checkpoint_steps[name] = cpu_tree(model.state_dict()), step
        h.synchronize(device)
        timing.append(time.perf_counter() - begin)
        if step % 50 == 0:
            print(json.dumps({"seed": plan["seed"], "arm": arm, "step": step,
                              "instrumented": instrumented, "estimation_rank": ranks[-1],
                              "elapsed_seconds": time.perf_counter() - started}), flush=True)
    assert warm_state is not None
    if WIDTHS[arm] == 128:
        assert ranks[159] == 128, "Estimator did not reach128 by160"
    result = {
        "seed": plan["seed"], "arm": arm, "replacement_probability": .9, "steps": steps,
        "instrumented": instrumented, "elapsed_seconds": time.perf_counter() - started,
        "step_elapsed_seconds": timing, "estimation_rank_by_step": ranks,
        "steady_steps129_to220_mean_seconds": float(np.mean(timing[128:220])),
        "steady_steps129_to220_median_seconds": float(np.median(timing[128:220])),
        "scheduled_repair_step200_seconds": timing[199],
        "warmup_trajectory_hashes": warmup_hashes, "warmup_observer_hashes": observer_hashes,
        "warmup_core_sha256": tree_hash(warm_state["core"]),
        "warmup_observer_sha256": tree_hash(warm_state["observer"]),
        "trajectory_parameter_sha256": trajectory_hashes, "measurement_state_checks": measurement_checks,
        "all_invariant_gates_passed": True,
    }
    if not pilot:
        checkpoints["final"], checkpoint_steps["final"] = cpu_tree(model.state_dict()), steps
        result.update({"steps_raw": rows, "validation_trajectory": validations, "checkpoint_steps": checkpoint_steps,
                       "final_training_clean": h.evaluate(model, data["x"], data["clean"]),
                       "final_training_noisy": h.evaluate(model, data["x"], data["noisy"]),
                       "final_validation": validations[-1],
                       "realized_replacement_fraction": data["realized_replacement_fraction"],
                       "realized_incorrect_fraction": data["realized_incorrect_fraction"]})
        assert set(checkpoints) == set(CHECKPOINTS)
    final_state = cpu_tree(h.snapshot(model, optimizer, tracker))
    result["peak_gpu_allocated_bytes"] = torch.cuda.max_memory_allocated() if str(device).startswith("cuda") else 0
    result["peak_gpu_reserved_bytes"] = torch.cuda.max_memory_reserved() if str(device).startswith("cuda") else 0
    return result, checkpoints, final_state, warm_state


def run_pilot(output):
    output.mkdir(exist_ok=False)
    record = manifest("development_pilot")
    h.write_json(output / "execution.json", record)
    context = {"phase": "loading_training_data", "seed": PILOT_SEED}
    try:
        x, y = h.load_training()
        plan = h.make_plan(PILOT_SEED, PILOT_STEPS)
        record["plans"] = [h.save_plan(output / "development-plan.npz", plan)]
        data = h.dataset_for_plan(x, y, plan, .9, "cuda")
        reports, warmup_references = [], {}
        for arm in ARMS:
            runs, finals = {}, {}
            for mode in (False, True):
                result, _, final_state, warm_state = train_arm(plan, data, arm, PILOT_STEPS, mode, True, context)
                if arm == "adamw":
                    warmup_references[(mode, "adamw")] = (result, warm_state)
                else:
                    verify_warmup_pair(*warmup_references[(mode, "adamw")], result, warm_state)
                if arm == "estimate32_project32":
                    warmup_references[(mode, "hard32")] = (result, warm_state)
                if arm == "scalar32_norm":
                    verify_warmup_pair(*warmup_references[(mode, "hard32")], result, warm_state, compare_observer=True)
                runs[mode], finals[mode] = result, final_state
            assert runs[False]["trajectory_parameter_sha256"] == runs[True]["trajectory_parameter_sha256"], "Instrumentation changed trajectory"
            assert h.equal_tree(finals[False], finals[True]), "Instrumentation changed final state"
            reports.append({"arm": arm, "trajectory_bitwise_identical": True, "final_state_bitwise_identical": True,
                            "warmup_checks_passed": True, "uninstrumented": runs[False], "instrumented": runs[True]})
            h.write_json(output / "timing-and-invariants.json", reports)
        record.update({"status": "complete_passed", "completed_utc": h.utc(), "completed_traces": 8,
                       "warmup_checks_passed": True, "validation_or_accuracy_computed": False,
                       "official_test_loaded": False})
    except Exception as exc:
        h.preserve_failure(output, record, context, exc)
        raise
    h.write_json(output / "execution.json", record)


def full_launch_gate():
    committed_source_gate()
    pilot = json.loads((HERE / "pilot" / "execution.json").read_text())
    assert pilot["status"] == "complete_passed" and pilot["completed_traces"] == 8
    assert pilot["source_sha256"] == source_hashes(), "Source changed after passing pilot"
    for item in pilot["training_data_artifacts"]:
        assert h.file_hash(item["path"]) == item["sha256"], "Training data changed after pilot"


def run_full(output):
    full_launch_gate()
    output.mkdir(exist_ok=False)
    record = manifest("confirmatory")
    h.write_json(output / "execution.json", record)
    context = {"phase": "loading_training_data"}
    try:
        x, y = h.load_training()
        record["plans"], record["checkpoints"], all_results = [], [], []
        for seed in SEEDS:
            plan = h.make_plan(seed, STEPS)
            record["plans"].append(h.save_plan(output / f"plan-seed{seed}.npz", plan))
            data = h.dataset_for_plan(x, y, plan, .9, "cuda")
            references = {}
            for arm in ARMS:
                tag = f"seed{seed}-{arm}"
                context = {"phase": "model_initialization", "seed": seed, "arm": arm, "step": 0}
                result, checkpoints, _, warm_state = train_arm(plan, data, arm, STEPS, True, False, context)
                context["phase"] = "cross_arm_warmup_gate"
                if arm == "adamw":
                    references["adamw"] = (result, warm_state)
                else:
                    verify_warmup_pair(*references["adamw"], result, warm_state)
                if arm == "estimate32_project32":
                    references["hard32"] = (result, warm_state)
                if arm == "scalar32_norm":
                    verify_warmup_pair(*references["hard32"], result, warm_state, compare_observer=True)
                result["warmup_checks_passed"] = True
                checkpoint_path = output / f"{tag}-checkpoints.pt"
                torch.save(checkpoints, checkpoint_path)
                record["checkpoints"].append(h.artifact(checkpoint_path))
                result["checkpoint_path"] = str(checkpoint_path)
                h.write_json(output / f"{tag}.json", result)
                all_results.append((tag, result))
        assert len(all_results) == 12
        record["all_training_completed_utc"] = h.utc()
        context = {"phase": "test_loading_after_all_training"}
        tx, ty = h.load_test_after_training()
        tx, ty = tx.cuda(), ty.cuda()
        record["test_first_loaded_utc"] = h.utc()
        record["test_data_artifacts"] = [h.artifact(h.DATA / name) for name in ("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte")]
        for tag, result in all_results:
            model = h.make_model(0, "cuda")
            bundle = torch.load(result["checkpoint_path"], map_location="cuda", weights_only=True)
            result["test"] = {}
            for name in CHECKPOINTS:
                context = {"phase": "test_evaluation", "tag": tag, "checkpoint": name}
                model.load_state_dict(bundle[name])
                result["test"][name] = h.evaluate(model, tx, ty)
            h.write_json(output / f"{tag}.json", result)
        record.update({"status": "complete", "completed_utc": h.utc(), "completed_runs": 12,
                       "test_evaluations": 48, "warmup_checks_passed": True})
    except Exception as exc:
        h.preserve_failure(output, record, context, exc)
        raise
    h.write_json(output / "execution.json", record)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pilot", action="store_true")
    group.add_argument("--full", action="store_true")
    parser.add_argument("--development-go", action="store_true")
    parser.add_argument("--confirmatory-go", action="store_true")
    args = parser.parse_args()
    if args.pilot and not args.development_go:
        parser.error("Pilot requires separate parent approval and --development-go")
    if args.full and not args.confirmatory_go:
        parser.error("Full run requires separate parent approval and --confirmatory-go")
    h.configure()
    assert torch.cuda.is_available(), "Local GPU required; no paid or CPU-training fallback"
    run_pilot(HERE / "pilot") if args.pilot else run_full(HERE / "results")


if __name__ == "__main__":
    main()
