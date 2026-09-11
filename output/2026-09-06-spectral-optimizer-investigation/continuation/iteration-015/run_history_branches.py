#!/usr/bin/env python3
"""Exclusive I15 saved-state branches; completed I14 trajectories are references only."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(name, "1")

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
I14 = HERE.parent / "iteration-014"
sys.path.insert(0, str(I14))
sys.path.insert(0, str(HERE))
import history_core as core
import run_cross_optimizer as old

c14, plans, i9 = old.core, old.plans, old.core.i9
SOURCE_ROOT = Path("/tmp/spectral-experiment-artifacts/spectral-i14-001.SJWZCj")
I14_COMMIT = "d48f20a76f32d5b717dd797678f276a16ec57f22"
SUPPLEMENT_SHA = "13c2ea817a141cc5b5062fd8ac9e56611fa19dea5ef8bbdacfc56ad7cf717809"
SEEDS = (200, 201, 202)
TARGETS = ("clean", "fixed")
HORIZONS = (100, 250, 500, 1000, 1500, 2000)
LIMITS = {"smoke": 100, "confirmation": 1800}
TIMING_SAFETY_FACTOR = 1.5
NUMERICAL_FAILURES = (core.NumericalFailure, old.core.NumericalFailure)


def need(condition, message):
    if not condition:
        raise ValueError(message)


def source_manifest(commit):
    need(re.fullmatch(r"[0-9a-f]{40}", commit) is not None, "Full frozen commit required")
    old.source_manifest(I14_COMMIT)
    paths = [ROOT / "spectral_filter.py", HERE.parent / "iteration-009/neural_core.py",
             I14 / "optimizer_core.py", I14 / "data_plan.py", I14 / "run_cross_optimizer.py",
             HERE.parent / "iteration-013/mean_core.py", HERE / "history_core.py",
             HERE / "test_history_core.py", Path(__file__).resolve(),
             HERE / "test_history_runner.py", HERE / "protocol.md", HERE / "predictions.md"]
    hashes = {}
    for path in paths:
        name = str(path.relative_to(ROOT))
        current = old.digest(path)
        frozen = subprocess.check_output(["git", "show", commit + ":" + name], cwd=ROOT)
        need(hashlib.sha256(frozen).hexdigest() == current, "Frozen source differs: " + name)
        hashes[name] = current
    return hashes


def checked_artifact(directory, record):
    need(type(record) is dict and set(record) == {"name", "bytes", "sha256"},
         "Artifact record differs")
    name = record["name"]
    need(type(name) is str and name not in ("", ".", "..") and Path(name).name == name,
         "Artifact name must be a single filename")
    path = directory / name
    need(not path.is_symlink() and path.is_file(), "Artifact must be a regular nonsymlink file")
    need(path.stat().st_size == record["bytes"] and old.digest(path) == record["sha256"],
         "Artifact content differs: " + name)
    return path


def bound_inputs():
    """Read/hash only required existing inputs; no model loading or full audit replay."""
    supplement_path = I14 / "analysis-001/runtime-directory-supplement.json"
    need(old.digest(supplement_path) == SUPPLEMENT_SHA, "I14 acceptance supplement differs")
    supplement = json.loads(supplement_path.read_text())
    need(supplement["status"] == "accepted_with_empty_runtime_directory_exception"
         and supplement["original_audit_status"] == "fail"
         and supplement["artifact_root"] == str(SOURCE_ROOT), "I14 accepted provenance differs")
    for name, expected in (("audit.json", supplement["original_audit_sha256"]),
                           ("summary.json", supplement["original_summary_sha256"])):
        need(old.digest(I14 / "analysis-001" / name) == expected, "I14 original evidence differs")
    confirmation = SOURCE_ROOT / "confirmation"
    completion_path = confirmation / "completion.json"
    need(old.digest(completion_path) == supplement["bound_terminal_and_attempt_sha256"][
        "confirmation/completion.json"], "I14 confirmation completion differs")
    completed = json.loads(completion_path.read_text())
    need(completed["status"] == "complete" and completed["numerical_failures"] == 0
         and completed["frozen_commit"] == I14_COMMIT, "I14 completion state differs")
    declared = {row["name"]: row for row in completed["artifacts"]}
    need(len(declared) == len(completed["artifacts"]), "Duplicate I14 artifact names")
    records = {}

    def read_record(name):
        path = checked_artifact(confirmation, declared[name])
        records[name] = declared[name]
        return path

    index = json.loads(read_record("trajectories.json").read_text())
    entries = [row for row in index["entries"] if row["base"] == "sgdm"]
    expected = {(s, t, p) for s in SEEDS for t in TARGETS for p in ("raw", "current32")}
    need(len(entries) == 12 and {(r["seed"], r["target"], r["policy"]) for r in entries}
         == expected, "I14 SGDm reference membership differs")
    curves = {}
    parents = {}
    for entry in entries:
        need(entry["lr"] == .03 and entry["status"] == "complete"
             and entry["completed_steps"] == 2000, "I14 SGDm operating point differs")
        need(entry["artifact"] == declared[entry["artifact"]["name"]], "Curve manifest differs")
        curve = json.loads(read_record(entry["artifact"]["name"]).read_text())
        key = (entry["seed"], entry["target"], entry["policy"])
        need(all(curve[name] == entry[name] for name in (
            "id", "seed", "base", "lr", "target", "policy", "status", "completed_steps",
            "warmup_full_state_digest", "warmup_evaluation_digest")), "Curve identity differs")
        need([row["horizon"] for row in curve["curve"]] == list(plans.HORIZONS),
             "I14 reference horizon coverage differs")
        need([row["horizon"] for row in curve["checkpoints"]] == list(plans.HORIZONS),
             "I14 reference checkpoint coverage differs")
        for checkpoint in curve["checkpoints"]:
            kind = "full_state" if checkpoint["horizon"] in (0, 100, 2000) else "model_state"
            record = checkpoint[kind]
            need(record == declared[record["name"]], "Reference checkpoint membership differs")
        curves[key] = curve
        if entry["policy"] == "current32":
            anchor = next(row for row in curve["checkpoints"] if row["horizon"] == 100)
            need(anchor["full_state"] == declared[anchor["full_state"]["name"]],
                 "Parent artifact manifest differs")
            parent_path = read_record(anchor["full_state"]["name"])
            need(anchor["full_state_digest"] == curve["warmup_full_state_digest"],
                 "Parent semantic digest differs")
            parents[key[:2]] = {"path": str(parent_path), "artifact": anchor["full_state"],
                                "state_digest": anchor["full_state_digest"]}
    for seed in SEEDS:
        for target in TARGETS:
            raw, current = (curves[(seed, target, p)] for p in ("raw", "current32"))
            need(raw["warmup_full_state_digest"] == current["warmup_full_state_digest"]
                 and raw["curve"][:2] == current["curve"][:2], "I14 h100 pair differs")
    saved_plans, corruption_counts = {}, {}
    for seed in SEEDS:
        path = read_record(f"plan-confirmation-s{seed}.json")
        saved_plans[seed] = plans.plan_from_json(json.loads(path.read_text()))
        corruption_counts[seed] = json.loads(read_record(f"corruption-s{seed}.json").read_text())
    calibration_binding = json.loads(read_record("calibration-binding.json").read_text())
    dataset_hashes = calibration_binding["selection"]["dataset_inputs"]
    need(set(dataset_hashes) == {"train-images-idx3-ubyte", "train-labels-idx1-ubyte"},
         "Dataset input membership differs")
    need(all(old.digest(plans.DATA / name) == digest for name, digest in dataset_hashes.items()),
         "Training dataset differs")
    binding = {"schema": "i15_i14_inputs_v1", "source_root": str(SOURCE_ROOT),
               "i14_commit": I14_COMMIT, "runtime_supplement_sha256": SUPPLEMENT_SHA,
               "confirmation_completion_sha256": old.digest(completion_path),
               "source_records": records, "dataset_inputs": dataset_hashes,
               "corruption_counts": {str(seed): value for seed, value in corruption_counts.items()},
               "parents": [{"seed": s, "target": t, **row} for (s, t), row in parents.items()],
               "references": [{"seed": s, "target": t, "policy": p,
                   "curve_artifact": declared[f"curve-{row['id']}.json"],
                   "checkpoint_records": row["checkpoints"],
                   "warmup_state_digest": row["warmup_full_state_digest"]}
                   for (s, t, p), row in curves.items()],
               "source_replayed": False}
    return binding, parents, curves, saved_plans


def run_branch(run, state, data, plan, target, policy, expected_start, *,
               smoke=False, end=2000, horizons=HORIZONS):
    """Continue one complete h100 state; no raw/native baseline production branch."""
    need(policy in core.REAL_POLICIES and target in TARGETS, "Unregistered new arm")
    identity = f"s{plan['seed']}-sgdm-{target}-{policy}"
    if hasattr(run, "path"):
        need(not any(run.path.glob(f"*-{identity}*")), "Branch identity already has artifacts")
    need(type(state) is dict and state.get("base") == "sgdm" and state.get("lr") == .03
         and state.get("tracker", {}).get("step_count") == 100, "Parent is not I14 SGDm h100")
    core.validate_parent_snapshot(state)
    if not smoke:
        plans.validate_plan(plan)
        need(end == 2000 and tuple(horizons) == HORIZONS
             and state["model_spec"] == {"input_dim": 784, "width": 64, "classes": 10},
             "Real branch differs from fixed protocol")
    else:
        need(end == 110 and tuple(horizons) == (100, 110), "Synthetic smoke dimensions differ")
    need(type(data) is dict and tuple(data) == plans.DATA_KEYS, "Data layout differs")
    device = data["x"].device
    need(all(value.device == device for value in data.values()), "Data devices differ")
    need(type(plan["training_batches"]) is np.ndarray
         and plan["training_batches"].dtype == np.int64
         and plan["training_batches"].shape == (end, 64)
         and bool(((plan["training_batches"] >= 0)
                   & (plan["training_batches"] < len(data["x"]))).all()), "Batch plan differs")
    parent_digest = i9.tree_digest(state)
    model, optimizer, tracker = c14.restore(state, device)
    need(i9.tree_digest(c14.snapshot(model, optimizer, tracker, "sgdm", .03)) == parent_digest,
         "Complete-state restore seam differs")
    c14._finite_live(model, optimizer, tracker, "branch_parent")
    measured = {"horizon": 100, **old.evaluate(model, data)}
    need(measured == expected_start, "Parent evaluation seam differs")
    need(i9.tree_digest(c14.snapshot(model, optimizer, tracker, "sgdm", .03)) == parent_digest,
         "Parent evaluation changed complete state")
    curve = [measured]
    diagnostics, checkpoints = [], []
    batches = torch.as_tensor(plan["training_batches"], dtype=torch.long, device=device)
    targets = data["clean"] if target == "clean" else data["noisy"]
    failure = None
    completed = 100
    history_update_seconds = 0.0
    for step in range(101, end + 1):
        if step == 101 or step % 25 == 0:
            run.check()
        try:
            indices = batches[step - 1]
            update_started = time.monotonic()
            row = core.history_step(model, optimizer, tracker, data["x"][indices],
                                    targets[indices], policy, capture_digests=(step == 101))
            history_update_seconds += time.monotonic() - update_started
            diagnostics.append({"step": step, "relative_step": step - 100, **row})
            completed = step
            if step in horizons:
                measured = {"horizon": step, **old.evaluate(model, data)}
                curve.append(measured)
                checkpoint = {"horizon": step, "relative_horizon": step - 100}
                if step == end:
                    saved = c14.snapshot(model, optimizer, tracker, "sgdm", .03)
                    checkpoint.update(full_state=run.save(
                        f"state-{identity}-h{step}.pt", saved, tensor=True),
                        full_state_digest=i9.tree_digest(saved))
                else:
                    saved = {key: value.detach().cpu().clone()
                             for key, value in model.state_dict().items()}
                    checkpoint["model_state"] = run.save(
                        f"model-{identity}-h{step}.pt", saved, tensor=True)
                checkpoints.append(checkpoint)
        except NUMERICAL_FAILURES as exc:
            failure = {"type": type(exc).__name__, "message": str(exc),
                       "attempted_step": step, "completed_step": completed,
                       "last_valid_horizon": curve[-1]["horizon"]}
            failure["state_artifact"] = run.save(f"failed-state-{identity}.pt",
                c14.snapshot(model, optimizer, tracker, "sgdm", .03), tensor=True)
            break
    need(i9.tree_digest(state) == parent_digest, "Branch mutated its parent input")
    result = {"schema": "i15_history_branch_v1", "id": identity,
              "seed": plan["seed"], "target": target, "policy": policy,
              "base": "sgdm", "lr": .03, "parent_horizon": 100,
              "parent_state_digest": parent_digest,
              "parent_evaluation_digest": i9.tree_digest(expected_start),
              "status": "numerical_failure" if failure else "complete",
              "requested_updates": end - 100, "completed_updates": completed - 100,
              "last_completed_horizon": completed, "curve": curve,
              "history_update_seconds": history_update_seconds,
              "steps": diagnostics, "checkpoints": checkpoints, "failure": failure,
              "first_step_digests": diagnostics[0]["digests"] if diagnostics else None,
              "selected_horizons": old.selection(curve) if failure is None else None}
    artifact = run.save(f"branch-{identity}.json", result)
    print(json.dumps({"branch": identity, "status": result["status"],
                      "elapsed_seconds": time.monotonic() - run.started}), flush=True)
    return {key: result[key] for key in ("id", "seed", "target", "policy", "status",
        "completed_updates", "parent_state_digest", "parent_evaluation_digest",
        "first_step_digests", "history_update_seconds")} | {"artifact": artifact}


def validate_first_step_pairs(entries):
    """Boundary checks only; missing numerical arms never become outcome averages."""
    checks = []
    for seed, target in sorted({(row["seed"], row["target"]) for row in entries}):
        group = [row for row in entries if (row["seed"], row["target"]) == (seed, target)]
        need(len(group) == 3 and {row["policy"] for row in group} == set(core.REAL_POLICIES),
             "First-step policy membership differs")
        for key in ("parent_state_digest", "parent_evaluation_digest"):
            need(len({row[key] for row in group}) == 1, "Parent pair boundary differs: " + key)
        available = [row for row in group if row["first_step_digests"] is not None]
        need(all(row["status"] == "numerical_failure" for row in group
                 if row["first_step_digests"] is None), "Complete branch lacks first-step evidence")
        for key in ("raw_gradient", "post_observer", "old_momentum_buffer", "native_action_old_buffer"):
            need(len({row["first_step_digests"][key] for row in available}) <= 1,
                 "First-step common tensor differs: " + key)
        mean = [row for row in available if row["policy"].startswith("mean_")]
        need(len({row["first_step_digests"]["applied_gradient"] for row in mean}) <= 1,
             "First-step mean deliveries differ")
        checks.append({"seed": seed, "target": target, "available_first_steps": len(available),
                       "status": "pass" if len(available) == 3 else "partial_numerical_evidence"})
    return checks


def synthetic_smoke(run):
    generator = np.random.default_rng(2026090715)
    x = torch.as_tensor(generator.random((128, 784)), dtype=torch.float32, device="cuda")
    y = torch.as_tensor(generator.integers(0, 10, 128), dtype=torch.long, device="cuda")
    data = {"x": x[:64], "clean": y[:64], "noisy": (y[:64] + 1) % 10,
            "vx": x[64:96], "vy": y[64:96], "ax": x[96:], "ay": y[96:]}
    plan = {"seed": 215, "training_batches": generator.integers(0, 64, (110, 64), dtype=np.int64)}
    entries = []
    for target in TARGETS:
        old.seed_all(2026090715)
        model = i9.make_model(2026090715, "cuda")
        optimizer = c14.make_optimizer(model, "sgdm", .03)
        tracker = i9.make_tracker(model, optimizer)
        labels = data["clean"] if target == "clean" else data["noisy"]
        for step in range(1, 101):
            run.check()
            indices = torch.as_tensor(plan["training_batches"][step - 1], device="cuda")
            c14.train_step(model, optimizer, tracker, data["x"][indices], labels[indices],
                           "raw", "sgdm")
        state = c14.snapshot(model, optimizer, tracker, "sgdm", .03)
        run.save(f"synthetic-parent-{target}.pt", state, tensor=True)
        expected = {"horizon": 100, **old.evaluate(model, data)}
        del model, optimizer, tracker
        for policy in core.REAL_POLICIES:
            entries.append(run_branch(run, state, data, plan, target, policy, expected,
                                      smoke=True, end=110, horizons=(100, 110)))
    pairs = validate_first_step_pairs(entries)
    run.save("branches.json", {"entries": entries, "first_step_pair_checks": pairs,
                              "synthetic_warmup_updates": 200,
                              "new_branch_updates": 60})
    return entries, 200


def confirmation(run):
    binding, parents, references, saved_plans = bound_inputs()
    run.save("parent-inputs.json", binding)
    x, y = plans.read_training()
    data_cache, state_cache, seams = {}, {}, []
    # Admit every parent before the first scientific update. Per-arm restoration
    # below remains independent and restores the saved RNG again.
    for seed in SEEDS:
        data, corruption = plans.data_for_plan(x, y, saved_plans[seed], "cuda")
        need(corruption == binding["corruption_counts"][str(seed)],
             "Materialized fixed-corruption counts differ from I14")
        data_cache[seed] = data
        run.save(f"plan-s{seed}.json", saved_plans[seed])
        run.save(f"corruption-s{seed}.json", corruption)
        for target in TARGETS:
            run.check()
            parent = parents[(seed, target)]
            path = checked_artifact(SOURCE_ROOT / "confirmation", parent["artifact"])
            state = torch.load(path, map_location="cpu", weights_only=False)
            core.validate_parent_snapshot(state)
            need(i9.tree_digest(state) == parent["state_digest"], "Parent tensor digest differs")
            model, optimizer, tracker = c14.restore(state, "cuda")
            before = i9.tree_digest(c14.snapshot(model, optimizer, tracker, "sgdm", .03))
            need(before == parent["state_digest"], "Preflight complete-state seam differs")
            measured = {"horizon": 100, **old.evaluate(model, data)}
            expected = next(row for row in references[(seed, target, "current32")]["curve"]
                            if row["horizon"] == 100)
            need(measured == expected, "Preflight parent evaluation seam differs")
            need(i9.tree_digest(c14.snapshot(model, optimizer, tracker, "sgdm", .03)) == before,
                 "Preflight evaluation changed state")
            state_cache[(seed, target)] = state
            seams.append({"seed": seed, "target": target, "state_digest": before,
                          "evaluation_digest": i9.tree_digest(measured), "status": "pass"})
            del model, optimizer, tracker
    run.save("parent-seams.json", {"parents": seams, "scientific_updates_before_admission": 0})
    entries = []
    for seed in SEEDS:
        plan = saved_plans[seed]
        data = data_cache[seed]
        for target in TARGETS:
            state = state_cache[(seed, target)]
            expected = next(row for row in references[(seed, target, "current32")]["curve"]
                            if row["horizon"] == 100)
            for policy in core.REAL_POLICIES:
                entries.append(run_branch(run, state, data, plan, target, policy, expected))
            del state
        del data
    expected_members = {(s, t, p) for s in SEEDS for t in TARGETS for p in core.REAL_POLICIES}
    need(len(entries) == 18 and {(r["seed"], r["target"], r["policy"]) for r in entries}
         == expected_members, "I15 branch membership differs")
    need(bound_inputs()[0] == binding, "Bound parent/reference inputs changed during I15")
    pairs = validate_first_step_pairs(entries)
    run.save("branches.json", {"entries": entries, "first_step_pair_checks": pairs,
                              "new_branches": 18,
                              "reused_native_current": 6, "reused_raw": 6})
    return entries, 0


def main():
    if not __debug__:
        raise RuntimeError("Assertion checks must remain enabled")
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--phase", required=True, choices=tuple(LIMITS))
    parser.add_argument("--frozen-commit", required=True)
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    need(root.parent == Path("/tmp/spectral-experiment-artifacts") and root.name.startswith("spectral-i15-001."),
         "I15 requires its exclusive large-volume root")
    need(subprocess.check_output(["findmnt", "-n", "-o", "SOURCE", "-T", str(root)],
         text=True).strip() == "/dev/RECONFIGURE_FOR_LOCAL_STORAGE", "I15 root is not on the intended large volume")
    need(Path(os.environ.get("TMPDIR", "")).resolve() == root / "runtime"
         and (root / "runtime").is_dir() and not (root / "runtime").is_symlink(),
         "I15 requires its prospectively declared runtime directory")
    sources = source_manifest(args.frozen_commit)
    forecast = None
    if args.phase == "confirmation":
        smoke = old.verify_phase(root / "smoke", sources)
        need(smoke["numerical_failures"] == 0 and smoke["completed_training_updates"] == 260
             and smoke["branches"] == 6, "Synthetic smoke did not qualify")
        forecast = smoke["history_update_seconds"] / 60 * 34200 * TIMING_SAFETY_FACTOR + 60
        need(forecast < LIMITS["confirmation"], "Smoke projection exceeds confirmation limit")
    with (root / f"attempt-{args.phase}.json").open("xb") as handle:
        handle.write(old.json_bytes({"schema": "i15_attempt_v1", "phase": args.phase,
            "frozen_commit": args.frozen_commit, "source_hashes": sources,
            "pid": os.getpid(), "restart": "forbidden"}))
    run = old.Run(root, args.phase, LIMITS[args.phase])
    try:
        old.configure()
        run.save("manifest.json", {"schema": "i15_manifest_v1", "phase": args.phase,
            "frozen_commit": args.frozen_commit, "source_hashes": sources,
            "torch_version": str(torch.__version__), "numpy_version": np.__version__,
            "python_version": sys.version, "gpu": torch.cuda.get_device_name(),
            "policies": list(core.REAL_POLICIES), "horizons": HORIZONS,
            "smoke_based_seconds_forecast": forecast,
            "timing_safety_factor": TIMING_SAFETY_FACTOR,
            "cloud_spend_usd": 0, "authorized_budget_usd": 100,
            "shared_artifact_cap_bytes": old.ARTIFACT_CAP,
            "wall_limit_seconds": LIMITS[args.phase],
            "runtime_directory": "runtime", "runtime_bytes_count_in_cap": True})
        entries, warmup = synthetic_smoke(run) if args.phase == "smoke" else confirmation(run)
        need(source_manifest(args.frozen_commit) == sources, "Source changed during acquisition")
        failures = sum(row["status"] == "numerical_failure" for row in entries)
        run.terminal("completion.json", {"schema": "i15_completion_v1", "status": "complete",
            "phase": args.phase, "source_hashes": sources, "frozen_commit": args.frozen_commit,
            "branches": len(entries), "numerical_failures": failures,
            "all_requested_endpoints_present": failures == 0,
            "completed_training_updates": warmup + sum(row["completed_updates"] for row in entries),
            "history_update_seconds": sum(row["history_update_seconds"] for row in entries),
            "elapsed_seconds": time.monotonic() - run.started,
            "peak_torch_gpu_bytes": torch.cuda.max_memory_allocated(),
            "shared_artifact_bytes_before_completion": run.used(), "artifacts": run.artifacts})
    except BaseException as exc:
        run.terminal("failure.json", {"schema": "i15_phase_failure_v1", "status": "failed",
            "phase": args.phase, "exception_type": type(exc).__name__, "message": str(exc)[:2000],
            "source_hashes": sources, "elapsed_seconds": time.monotonic() - run.started,
            "artifacts": run.artifacts, "restart": "forbidden"})
        raise


if __name__ == "__main__":
    main()
