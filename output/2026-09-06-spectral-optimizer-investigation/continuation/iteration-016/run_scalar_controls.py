#!/usr/bin/env python3
"""Exclusive I16 scalar temporal controls from the six accepted I14 h100 parents."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
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
I15 = HERE.parent / "iteration-015"
sys.path.insert(0, str(I15))
sys.path.insert(0, str(HERE))
import run_history_branches as previous
import scalar_core as core


old, c14, plans, i9 = previous.old, previous.c14, previous.plans, previous.i9
SEEDS = (200, 201, 202)
TARGETS = ("clean", "fixed")
HORIZONS = (100, 250, 500, 1000, 1500, 2000)
LIMITS = {"smoke": 100, "confirmation": 1800}
TIMING_SAFETY_FACTOR = 1.5
CONFIRMATION_UPDATES = 34200
I15_ROOT = Path("/tmp/spectral-experiment-artifacts/spectral-i15-001.RTFjXq")
I15_ACQUISITION_COMMIT = "2bccbc6a883f1c4c11950965f9801d2c03d4350b"
I15_AUDIT_SHA = "53f9c65d7e189567d017bf5b8a4488629e27308d02d90cc1426b000f802837ac"
I15_SUMMARY_SHA = "71cb51a6933b52594fb97f29e322b3dd3b46489cc28f4eab442c209dfb15a754"
I15_COLLECTION_SHA = "b50bd33762ac58b77fc0755418934432e503f3a92343724f33cef6cfffa700fa"
I15_CONFIRMATION_SHA = "2e28e146ef84d934bc33d41f30dfa92cfbcbe7cd19a9b506aec83a4ef7635be6"
K_NAMES = {0.0: "k0", 0.5: "k0p5", 0.9: "k0p9"}
NUMERICAL_FAILURES = (core.NumericalFailure, previous.core.NumericalFailure,
                      old.core.NumericalFailure)


def need(condition, message):
    if not condition:
        raise ValueError(message)


def source_manifest(commit):
    need(re.fullmatch(r"[0-9a-f]{40}", commit or "") is not None,
         "Full frozen commit required")
    hashes = dict(previous.source_manifest(commit))
    paths = [HERE / "scalar_core.py", HERE / "test_scalar_core.py",
             Path(__file__).resolve(), HERE / "test_scalar_runner.py",
             HERE / "protocol.md", HERE / "predictions.md"]
    for path in paths:
        name = str(path.relative_to(ROOT))
        current = old.digest(path)
        frozen = subprocess.check_output(["git", "show", commit + ":" + name], cwd=ROOT)
        need(hashlib.sha256(frozen).hexdigest() == current,
             "Frozen source differs: " + name)
        hashes[name] = current
    return hashes


def _pinned_json(path, expected_sha, label):
    need(not path.is_symlink() and path.is_file(), label + " must be regular and nonsymlink")
    need(old.digest(path) == expected_sha, label + " differs from its frozen SHA256")
    return json.loads(path.read_text())


def _summary_values_match(values, expected):
    if type(values) is not dict or type(expected) is not dict:
        return False
    fields = {
        "train_clean_ce": expected["train"]["clean_ce"],
        "train_soft_ce": expected["train"]["soft_ce"],
        "train_clean_accuracy": expected["train"]["clean_accuracy"],
        "train_fixed_ce": expected["train"]["fixed_ce"],
        "train_fixed_minus_soft_ce": expected["train"]["fixed_minus_soft_ce"],
        "train_fixed_accuracy": expected["train"]["fixed_accuracy"],
        "train_confidence": expected["train"]["mean_max_probability"],
        "train_true_label_probability": expected["train"]["mean_true_label_probability"],
        "validation_clean_ce": expected["validation"]["clean_ce"],
        "validation_clean_accuracy": expected["validation"]["clean_accuracy"],
        "validation_confidence": expected["validation"]["mean_max_probability"],
        "validation_true_label_probability":
            expected["validation"]["mean_true_label_probability"],
        "auxiliary_clean_ce": expected["auxiliary"]["clean_ce"],
        "auxiliary_clean_accuracy": expected["auxiliary"]["clean_accuracy"],
        "auxiliary_confidence": expected["auxiliary"]["mean_max_probability"],
        "auxiliary_true_label_probability":
            expected["auxiliary"]["mean_true_label_probability"]}
    return all(values.get(name) == value for name, value in fields.items())


def _summary_h100_matches(reference, expected):
    curve = reference.get("curve") if type(reference) is dict else None
    return (type(curve) is list and len(curve) == len(HORIZONS)
            and [row.get("horizon") for row in curve] == list(HORIZONS)
            and _summary_values_match(curve[0].get("values"), expected))


def _summary_curve_matches(reference, branch):
    summary_curve = reference.get("curve") if type(reference) is dict else None
    branch_curve = branch.get("curve") if type(branch) is dict else None
    if type(summary_curve) is not list or type(branch_curve) is not list \
            or [row.get("horizon") for row in summary_curve] != list(HORIZONS) \
            or [row.get("horizon") for row in branch_curve] != list(HORIZONS):
        return False
    return all(summary_point.get("horizon") == horizon
               and branch_point.get("horizon") == horizon
               and _summary_values_match(summary_point.get("values"), branch_point)
               for horizon, summary_point, branch_point in zip(
                   HORIZONS, summary_curve, branch_curve))


def bound_inputs():
    """Bind accepted I14 parents/references and terminal I15 scalar evidence."""
    i14_binding, parents, i14_curves, saved_plans = previous.bound_inputs()
    audit_path = I15 / "analysis-001/audit.json"
    summary_path = I15 / "analysis-001/summary.json"
    collection_path = I15 / "raw-results-001/collection.json"
    audit = _pinned_json(audit_path, I15_AUDIT_SHA, "I15 audit")
    summary = _pinned_json(summary_path, I15_SUMMARY_SHA, "I15 summary")
    collection = _pinned_json(collection_path, I15_COLLECTION_SHA, "I15 collection")
    completion_path = I15_ROOT / "confirmation/completion.json"
    completion = _pinned_json(completion_path, I15_CONFIRMATION_SHA,
                              "I15 confirmation completion")
    need(audit.get("schema") == "i15_history_analysis_audit_v1"
         and audit.get("status") == "pass" and audit.get("artifact_root") == str(I15_ROOT)
         and audit.get("phase_completion_sha256") == {
             "smoke": audit["phase_completion_sha256"].get("smoke"),
             "confirmation": I15_CONFIRMATION_SHA}, "I15 audit binding differs")
    need(summary.get("schema") == "i15_history_analysis_summary_v1"
         and summary.get("audit_status") == "pass"
         and summary.get("artifact_root") == str(I15_ROOT)
         and summary.get("counts", {}).get("new_confirmation_branches") == 18
         and summary.get("counts", {}).get("confirmation_numerical_failures") == 0,
         "I15 summary binding differs")
    need(collection.get("schema") == "i15_scalar_collection_v1"
         and collection.get("status") == "complete"
         and collection.get("source_root") == str(I15_ROOT)
         and collection.get("analysis_audit", {}).get("sha256") == I15_AUDIT_SHA
         and collection.get("phase_completion_sha256", {}).get("confirmation")
             == I15_CONFIRMATION_SHA,
         "I15 collection binding differs")
    need(completion.get("schema") == "i15_completion_v1"
         and completion.get("status") == "complete"
         and completion.get("phase") == "confirmation"
         and completion.get("frozen_commit") == I15_ACQUISITION_COMMIT
         and completion.get("branches") == 18
         and completion.get("numerical_failures") == 0
         and completion.get("all_requested_endpoints_present") is True,
         "I15 confirmation state differs")

    raw_references = {}
    for seed in SEEDS:
        for target in TARGETS:
            raw = i14_curves[seed, target, "raw"]
            current = i14_curves[seed, target, "current32"]
            need(raw["status"] == current["status"] == "complete"
                 and raw["curve"][:2] == current["curve"][:2],
                 "I14 raw/current h100 reference differs")
            raw_references[seed, target] = raw
    trajectories = summary.get("all_policy_trajectories")
    need(type(trajectories) is list and len(trajectories) == 30,
         "I15 trajectory summary membership differs")
    spectral = {(row.get("seed"), row.get("target")): row for row in trajectories
                if type(row) is dict and row.get("policy") == "mean_projected_history"}
    need(set(spectral) == {(seed, target) for seed in SEEDS for target in TARGETS},
         "I15 spectral reference membership differs")
    for key, reference in spectral.items():
        expected = next(row for row in i14_curves[key + ("current32",)]["curve"]
                        if row["horizon"] == 100)
        need(reference.get("status") == "complete"
             and reference.get("failure") is None
             and _summary_h100_matches(reference, expected),
             "I15 spectral h100 reference differs")

    declared_rows = completion.get("artifacts")
    need(type(declared_rows) is list
         and all(type(row) is dict and type(row.get("name")) is str
                 for row in declared_rows), "I15 artifact manifest differs")
    declared = {row["name"]: row for row in declared_rows}
    need(len(declared) == len(declared_rows), "Duplicate I15 artifact names")
    confirmation_dir = I15_ROOT / "confirmation"
    index_record = declared.get("branches.json")
    index_path = previous.checked_artifact(confirmation_dir, index_record)
    branch_index = json.loads(index_path.read_text())
    indexed = branch_index.get("entries") if type(branch_index) is dict else None
    need(type(indexed) is list and len(indexed) == 18,
         "I15 branch index membership differs")
    new_members = {(seed, target, policy) for seed in SEEDS for target in TARGETS
                   for policy in previous.core.REAL_POLICIES}
    need({(row.get("seed"), row.get("target"), row.get("policy"))
          for row in indexed if type(row) is dict} == new_members,
         "I15 branch index identities differ")
    direct_spectral = {}
    spectral_records = []
    for entry in indexed:
        if entry["policy"] != "mean_projected_history":
            continue
        key = (entry["seed"], entry["target"])
        artifact = entry.get("artifact")
        need(artifact == declared.get(artifact.get("name")
                                      if type(artifact) is dict else None),
             "I15 spectral branch manifest differs")
        branch_path = previous.checked_artifact(confirmation_dir, artifact)
        branch = json.loads(branch_path.read_text())
        expected_start = next(row for row in i14_curves[key + ("current32",)]["curve"]
                              if row["horizon"] == 100)
        identity_fields = ("id", "seed", "target", "policy", "status",
                           "completed_updates", "parent_state_digest",
                           "parent_evaluation_digest")
        need(branch.get("schema") == "i15_history_branch_v1"
             and all(branch.get(name) == entry.get(name) for name in identity_fields)
             and branch.get("base") == "sgdm" and branch.get("lr") == .03
             and branch.get("status") == "complete"
             and branch.get("failure") is None
             and branch.get("completed_updates") == 1900
             and branch.get("parent_state_digest") == parents[key]["state_digest"]
             and branch.get("parent_evaluation_digest")
                 == i9.tree_digest(expected_start)
             and [row.get("horizon") for row in branch.get("curve", [])]
                 == list(HORIZONS), "I15 spectral branch content differs")
        checkpoints = branch.get("checkpoints")
        need(type(checkpoints) is list
             and [row.get("horizon") for row in checkpoints]
                 == list(HORIZONS[1:]), "I15 spectral checkpoints differ")
        checkpoint_records = []
        for checkpoint in checkpoints:
            kind = "full_state" if checkpoint["horizon"] == 2000 else "model_state"
            record = checkpoint.get(kind)
            need(record == declared.get(record.get("name")
                                         if type(record) is dict else None),
                 "I15 spectral checkpoint manifest differs")
            previous.checked_artifact(confirmation_dir, record)
            checkpoint_records.append({"horizon": checkpoint["horizon"],
                                       kind: record})
        need(_summary_curve_matches(spectral[key], branch),
             "I15 spectral summary differs from original branch")
        direct_spectral[key] = branch
        spectral_records.append({"seed": key[0], "target": key[1],
            "curve_artifact": artifact, "checkpoint_records": checkpoint_records,
            "parent_state_digest": branch["parent_state_digest"],
            "parent_evaluation_digest": branch["parent_evaluation_digest"]})
    need(set(direct_spectral) == {(seed, target) for seed in SEEDS
                                  for target in TARGETS},
         "I15 direct spectral reference membership differs")

    binding = {"schema": "i16_parent_and_reference_binding_v1",
        "i14": i14_binding,
        "i15": {"artifact_root": str(I15_ROOT),
            "acquisition_commit": I15_ACQUISITION_COMMIT,
            "audit_sha256": I15_AUDIT_SHA, "summary_sha256": I15_SUMMARY_SHA,
            "collection_sha256": I15_COLLECTION_SHA,
            "confirmation_completion_sha256": I15_CONFIRMATION_SHA},
        "raw_references": [{"seed": seed, "target": target,
                            "curve": raw_references[seed, target]}
                           for seed in SEEDS for target in TARGETS],
        "spectral_references": spectral_records,
        "source_replayed": False}
    return binding, parents, i14_curves, saved_plans, spectral


def _validate_data(data, state, smoke):
    need(type(data) is dict and tuple(data) == plans.DATA_KEYS,
         "Data layout differs")
    spec = state.get("model_spec", {})
    if not smoke:
        need(spec == {"input_dim": 784, "width": 64, "classes": 10},
             "Real model dimensions differ")
    need(type(spec) is dict and all(type(spec.get(key)) is int and spec[key] > 0
                                    for key in ("input_dim", "width", "classes")),
         "Model specification differs")
    device = data["x"].device
    need(all(type(value) is torch.Tensor and value.layout == torch.strided
             and value.device == device for value in data.values()),
         "Data tensors or devices differ")
    for xkey, ykey in (("x", "clean"), ("x", "noisy"),
                       ("vx", "vy"), ("ax", "ay")):
        x, y = data[xkey], data[ykey]
        need(x.dtype == torch.float32 and x.ndim == 2
             and x.shape[1] == spec["input_dim"] and len(x) > 0
             and y.dtype == torch.long and y.ndim == 1 and len(x) == len(y)
             and bool(torch.isfinite(x).all())
             and bool(((y >= 0) & (y < spec["classes"])).all()),
             "Data topology or values differ")
    return device


def run_branch(run, state, data, plan, target, k, expected_start, *,
               smoke=False, end=2000, horizons=HORIZONS):
    need(type(k) is float and k in core.REAL_K and target in TARGETS,
         "Unregistered real scalar arm")
    label = K_NAMES[k]
    identity = f"s{plan.get('seed')}-sgdm-{target}-{label}"
    if hasattr(run, "path"):
        need(not any(run.path.glob(f"*-{identity}*")),
             "Branch identity already has artifacts")
    need(type(state) is dict and state.get("base") == "sgdm" and state.get("lr") == .03
         and state.get("tracker", {}).get("step_count") == 100,
         "Parent is not I14 SGDm h100")
    core.validate_parent_snapshot(state)
    if smoke:
        need(end == 110 and tuple(horizons) == (100, 110),
             "Synthetic smoke dimensions differ")
    else:
        plans.validate_plan(plan)
        need(end == 2000 and tuple(horizons) == HORIZONS,
             "Real branch differs from fixed protocol")
    device = _validate_data(data, state, smoke)
    batches_raw = plan.get("training_batches")
    need(type(batches_raw) is np.ndarray and batches_raw.dtype == np.int64
         and batches_raw.shape == (end, 64)
         and bool(((batches_raw >= 0) & (batches_raw < len(data["x"]))).all()),
         "Batch plan differs")
    parent_digest = i9.tree_digest(state)
    model, optimizer, tracker = c14.restore(state, device)
    need(i9.tree_digest(c14.snapshot(model, optimizer, tracker, "sgdm", .03))
         == parent_digest, "Complete-state restore seam differs")
    c14._finite_live(model, optimizer, tracker, "scalar_branch_parent")
    measured = {"horizon": 100, **old.evaluate(model, data)}
    need(measured == expected_start, "Parent evaluation seam differs")
    need(i9.tree_digest(c14.snapshot(model, optimizer, tracker, "sgdm", .03))
         == parent_digest, "Parent evaluation changed complete state")

    curve = [measured]
    diagnostics, checkpoints = [], []
    batches = torch.as_tensor(batches_raw, dtype=torch.long, device=device)
    targets = data["clean"] if target == "clean" else data["noisy"]
    failure = None
    completed = 100
    scalar_update_seconds = 0.0
    for step in range(101, end + 1):
        if step == 101 or step % 25 == 0:
            run.check()
        try:
            indices = batches[step - 1]
            started = time.monotonic()
            row = core.scalar_step(model, optimizer, tracker, data["x"][indices],
                                   targets[indices], k,
                                   capture_digests=(step == 101))
            scalar_update_seconds += time.monotonic() - started
            data_norm = row["displacement"]["data_norm"]
            raw_dot = row["displacement"]["raw_gradient_dot_data_delta"]
            path = {"data_step_squared_energy": data_norm * data_norm,
                    "data_path_length_increment": data_norm,
                    "raw_gradient_dot_data_step": raw_dot}
            need(all(type(value) is float and math.isfinite(value)
                     for value in path.values()), "Scalar path statistic is nonfinite")
            diagnostics.append({"step": step, "relative_step": step - 100,
                                "path_statistics": path, **row})
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
                    saved = {name: value.detach().cpu().clone()
                             for name, value in model.state_dict().items()}
                    checkpoint["model_state"] = run.save(
                        f"model-{identity}-h{step}.pt", saved, tensor=True)
                checkpoints.append(checkpoint)
        except NUMERICAL_FAILURES as exc:
            failure = {"type": type(exc).__name__, "message": str(exc),
                       "attempted_step": step, "completed_step": completed,
                       "last_valid_horizon": curve[-1]["horizon"]}
            failure["state_artifact"] = run.save(
                f"failed-state-{identity}.pt",
                c14.snapshot(model, optimizer, tracker, "sgdm", .03), tensor=True)
            break
    need(i9.tree_digest(state) == parent_digest, "Branch mutated its parent input")
    result = {"schema": "i16_scalar_branch_v1", "id": identity,
        "seed": plan["seed"], "target": target, "k": k, "k_label": label,
        "base": "sgdm", "lr": .03, "parent_horizon": 100,
        "parent_state_digest": parent_digest,
        "parent_evaluation_digest": i9.tree_digest(expected_start),
        "status": "numerical_failure" if failure else "complete",
        "requested_updates": end - 100, "completed_updates": completed - 100,
        "last_completed_horizon": completed, "curve": curve,
        "scalar_update_seconds": scalar_update_seconds, "steps": diagnostics,
        "checkpoints": checkpoints, "failure": failure,
        "first_step_digests": diagnostics[0]["digests"] if diagnostics else None,
        "selected_horizons": old.selection(curve) if failure is None else None}
    artifact = run.save(f"branch-{identity}.json", result)
    print(json.dumps({"branch": identity, "status": result["status"],
                      "elapsed_seconds": time.monotonic() - run.started}), flush=True)
    keys = ("id", "seed", "target", "k", "k_label", "status", "completed_updates",
            "parent_state_digest", "parent_evaluation_digest", "first_step_digests",
            "scalar_update_seconds")
    return {key: result[key] for key in keys} | {"artifact": artifact}


def validate_first_step_pairs(entries):
    checks = []
    groups = {(row["seed"], row["target"]) for row in entries}
    for seed, target in sorted(groups):
        group = [row for row in entries
                 if (row["seed"], row["target"]) == (seed, target)]
        need(len(group) == 3 and {row["k"] for row in group} == set(core.REAL_K),
             "First-step scalar membership differs")
        for name in ("parent_state_digest", "parent_evaluation_digest"):
            need(len({row[name] for row in group}) == 1,
                 "Parent pair boundary differs: " + name)
        available = [row for row in group if row["first_step_digests"] is not None]
        need(all(row["status"] == "numerical_failure" for row in group
                 if row["first_step_digests"] is None),
             "Complete scalar branch lacks first-step evidence")
        for name in ("raw_gradient", "post_observer", "old_momentum_buffer"):
            need(len({row["first_step_digests"][name] for row in available}) <= 1,
                 "First-step common tensor differs: " + name)
        checks.append({"seed": seed, "target": target,
                       "available_first_steps": len(available),
                       "status": "pass" if len(available) == 3
                                 else "partial_numerical_evidence"})
    return checks


def synthetic_smoke(run):
    generator = np.random.default_rng(2026090716)
    x = torch.as_tensor(generator.random((128, 784)), dtype=torch.float32, device="cuda")
    y = torch.as_tensor(generator.integers(0, 10, 128), dtype=torch.long, device="cuda")
    data = {"x": x[:64], "clean": y[:64], "noisy": (y[:64] + 1) % 10,
            "vx": x[64:96], "vy": y[64:96], "ax": x[96:], "ay": y[96:]}
    plan = {"seed": 216,
            "training_batches": generator.integers(0, 64, (110, 64), dtype=np.int64)}
    entries = []
    for target in TARGETS:
        old.seed_all(2026090716)
        model = i9.make_model(2026090716, "cuda")
        optimizer = c14.make_optimizer(model, "sgdm", .03)
        tracker = i9.make_tracker(model, optimizer)
        labels = data["clean"] if target == "clean" else data["noisy"]
        for step in range(1, 101):
            run.check()
            indices = torch.as_tensor(plan["training_batches"][step - 1], device="cuda")
            c14.train_step(model, optimizer, tracker, data["x"][indices],
                           labels[indices], "raw", "sgdm")
        state = c14.snapshot(model, optimizer, tracker, "sgdm", .03)
        run.save(f"synthetic-parent-{target}.pt", state, tensor=True)
        expected = {"horizon": 100, **old.evaluate(model, data)}
        del model, optimizer, tracker
        for k in core.REAL_K:
            entries.append(run_branch(run, state, data, plan, target, k, expected,
                                      smoke=True, end=110, horizons=(100, 110)))
    pairs = validate_first_step_pairs(entries)
    run.save("branches.json", {"entries": entries, "first_step_pair_checks": pairs,
                              "synthetic_warmup_updates": 200,
                              "new_scalar_updates": 60,
                              "k1_parity_scope": "synthetic_unit_test_only"})
    return entries, 200


def confirmation(run):
    binding, parents, references, saved_plans, spectral = bound_inputs()
    run.save("parent-inputs.json", binding)
    x, y = plans.read_training()
    data_cache, state_cache, seams = {}, {}, []
    # All six complete-state/evaluation/RNG seams are admitted before any new step.
    for seed in SEEDS:
        data, corruption = plans.data_for_plan(x, y, saved_plans[seed], "cuda")
        need(corruption == binding["i14"]["corruption_counts"][str(seed)],
             "Materialized fixed-corruption counts differ from I14")
        data_cache[seed] = data
        run.save(f"plan-s{seed}.json", saved_plans[seed])
        run.save(f"corruption-s{seed}.json", corruption)
        for target in TARGETS:
            run.check()
            parent = parents[seed, target]
            path = previous.checked_artifact(previous.SOURCE_ROOT / "confirmation",
                                             parent["artifact"])
            state = torch.load(path, map_location="cpu", weights_only=False)
            core.validate_parent_snapshot(state)
            need(i9.tree_digest(state) == parent["state_digest"],
                 "Parent tensor digest differs")
            model, optimizer, tracker = c14.restore(state, "cuda")
            before = i9.tree_digest(c14.snapshot(model, optimizer, tracker, "sgdm", .03))
            rng_before = i9._rng_state()
            need(before == parent["state_digest"], "Preflight complete-state seam differs")
            measured = {"horizon": 100, **old.evaluate(model, data)}
            rng_after = i9._rng_state()
            raw = references[seed, target, "raw"]
            current = references[seed, target, "current32"]
            raw_start = next(row for row in raw["curve"] if row["horizon"] == 100)
            current_start = next(row for row in current["curve"] if row["horizon"] == 100)
            need(measured == raw_start == current_start,
                 "Preflight I14 raw/current evaluation seam differs")
            need(_summary_h100_matches(spectral[seed, target], measured),
                 "Preflight I15 spectral evaluation seam differs")
            need(i9.tree_digest(c14.snapshot(model, optimizer, tracker, "sgdm", .03))
                 == before and i9.equal_tree(rng_before, rng_after),
                 "Preflight evaluation changed complete state or RNG")
            state_cache[seed, target] = state
            seams.append({"seed": seed, "target": target, "state_digest": before,
                          "evaluation_digest": i9.tree_digest(measured),
                          "rng_neutral": True, "status": "pass"})
            del model, optimizer, tracker
    run.save("parent-seams.json", {"parents": seams,
                                  "parents_loaded": 6,
                                  "scientific_updates_before_admission": 0})

    entries = []
    for seed in SEEDS:
        for target in TARGETS:
            state = state_cache[seed, target]
            expected = next(row for row in references[seed, target, "current32"]["curve"]
                            if row["horizon"] == 100)
            for k in core.REAL_K:
                entries.append(run_branch(run, state, data_cache[seed], saved_plans[seed],
                                          target, k, expected))
    expected_members = {(seed, target, k) for seed in SEEDS for target in TARGETS
                        for k in core.REAL_K}
    need(len(entries) == 18
         and {(row["seed"], row["target"], row["k"]) for row in entries}
             == expected_members, "I16 branch membership differs")
    need(bound_inputs()[0] == binding, "Bound I14/I15 inputs changed during I16")
    pairs = validate_first_step_pairs(entries)
    run.save("branches.json", {"entries": entries, "first_step_pair_checks": pairs,
                              "new_scalar_branches": 18,
                              "reused_i14_raw_k1": 6,
                              "reused_i15_mean_projected": 6})
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
    need(root.parent == Path("/tmp/spectral-experiment-artifacts")
         and root.name.startswith("spectral-i16-001."),
         "I16 requires its exclusive large-volume root")
    need(subprocess.check_output(["findmnt", "-n", "-o", "SOURCE", "-T", str(root)],
                                 text=True).strip() == "/dev/RECONFIGURE_FOR_LOCAL_STORAGE",
         "I16 root is not on the intended large volume")
    need(Path(os.environ.get("TMPDIR", "")).resolve() == root / "runtime"
         and (root / "runtime").is_dir() and not (root / "runtime").is_symlink(),
         "I16 requires its prospectively declared runtime directory")
    sources = source_manifest(args.frozen_commit)
    forecast = None
    if args.phase == "confirmation":
        smoke = old.verify_phase(root / "smoke", sources)
        need(smoke["numerical_failures"] == 0
             and smoke["completed_training_updates"] == 260
             and smoke["branches"] == 6, "Synthetic smoke did not qualify")
        forecast = (smoke["scalar_update_seconds"] / 60 * CONFIRMATION_UPDATES
                    * TIMING_SAFETY_FACTOR + 60)
        need(forecast <= LIMITS["confirmation"],
             "Smoke projection exceeds confirmation limit")
    with (root / f"attempt-{args.phase}.json").open("xb") as handle:
        handle.write(old.json_bytes({"schema": "i16_attempt_v1", "phase": args.phase,
            "frozen_commit": args.frozen_commit, "source_hashes": sources,
            "pid": os.getpid(), "restart": "forbidden"}))
    run = old.Run(root, args.phase, LIMITS[args.phase])
    try:
        old.configure()
        run.save("manifest.json", {"schema": "i16_manifest_v1", "phase": args.phase,
            "frozen_commit": args.frozen_commit, "source_hashes": sources,
            "torch_version": str(torch.__version__), "numpy_version": np.__version__,
            "python_version": sys.version, "gpu": torch.cuda.get_device_name(),
            "real_k": list(core.REAL_K), "test_only_k": list(core.TEST_ONLY_K),
            "horizons": HORIZONS, "smoke_based_seconds_forecast": forecast,
            "timing_safety_factor": TIMING_SAFETY_FACTOR,
            "cloud_spend_usd": 0, "authorized_budget_usd": 100,
            "shared_artifact_cap_bytes": old.ARTIFACT_CAP,
            "wall_limit_seconds": LIMITS[args.phase], "runtime_directory": "runtime",
            "runtime_bytes_count_in_cap": True})
        entries, warmup = synthetic_smoke(run) if args.phase == "smoke" else confirmation(run)
        need(source_manifest(args.frozen_commit) == sources,
             "Source changed during acquisition")
        failures = sum(row["status"] == "numerical_failure" for row in entries)
        run.terminal("completion.json", {"schema": "i16_completion_v1",
            "status": "complete", "phase": args.phase, "source_hashes": sources,
            "frozen_commit": args.frozen_commit, "branches": len(entries),
            "numerical_failures": failures,
            "all_requested_endpoints_present": failures == 0,
            "completed_training_updates": warmup
                + sum(row["completed_updates"] for row in entries),
            "scalar_update_seconds": sum(row["scalar_update_seconds"] for row in entries),
            "elapsed_seconds": time.monotonic() - run.started,
            "peak_torch_gpu_bytes": torch.cuda.max_memory_allocated(),
            "shared_artifact_bytes_before_completion": run.used(),
            "artifacts": run.artifacts})
    except BaseException as exc:
        run.terminal("failure.json", {"schema": "i16_phase_failure_v1",
            "status": "failed", "phase": args.phase,
            "exception_type": type(exc).__name__, "message": str(exc)[:2000],
            "source_hashes": sources, "elapsed_seconds": time.monotonic() - run.started,
            "artifacts": run.artifacts, "restart": "forbidden"})
        raise


if __name__ == "__main__":
    main()
