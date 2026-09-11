#!/usr/bin/env python3
"""I17 exclusive gain-normalized continuations; never restart consumed runs."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
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
I16 = HERE.parent / "iteration-016"


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


previous = _module("_i17_frozen_i16_runner", I16 / "run_scalar_controls.py")
core = _module("_i17_gain_core", HERE / "gain_core.py")
old, c14, plans, i9 = previous.old, previous.c14, previous.plans, previous.i9
need = previous.need
SEEDS, TARGETS, HORIZONS = previous.SEEDS, previous.TARGETS, previous.HORIZONS
LIMITS = {"smoke": 100, "confirmation": 1800}
CONFIRMATION_UPDATES = 45600
TIMING_SAFETY_FACTOR = 1.5
I16_ROOT = Path("/tmp/spectral-experiment-artifacts/spectral-i16-001.NdCmZ1")
I16_COMMIT = "f8c99aa82d8de106148cde90e6814069d692d953"
PINS = {
    "analysis-001/audit.json": "ace6596d8dd79ca419d0193ca9239864d7ad62f26bcf2be2fe85627e88f1b473",
    "analysis-001/summary.json": "11da55533e66c371132f939ce7e4a3a4495efc0fc5f5b202164d496bb438e401",
    "analysis-001/report-audit.json": "87be955c4f6f01514486969d51414d58b6ac583c0fb68f21cfb74de53ef6e96f",
    "raw-results-001/collection.json": "cc112411fd23c43f0c87d876491fc8a91ecd8e018fd58488668a037420a0687a",
}
I16_COMPLETION_SHA = "eed6d1497f6a3672a6a59bd8962b5c9e4d33e8b97521ba2f499df0f7a7ec4cb4"
NUMERICAL_FAILURES = (core.NumericalFailure,) + previous.NUMERICAL_FAILURES


def source_manifest(commit):
    need(re.fullmatch(r"[0-9a-f]{40}", commit or "") is not None,
         "Full frozen commit required")
    hashes = dict(previous.source_manifest(commit))
    for name in ("gain_core.py", "test_gain_core.py", "run_gain_controls.py",
                 "test_gain_runner.py", "protocol.md", "predictions.md"):
        path = HERE / name
        relative = str(path.relative_to(ROOT))
        current = old.digest(path)
        frozen = subprocess.check_output(["git", "show", commit + ":" + relative], cwd=ROOT)
        need(hashlib.sha256(frozen).hexdigest() == current,
             "Frozen source differs: " + relative)
        hashes[relative] = current
    return hashes


def bound_inputs():
    """Hash and bind original scalar k0 curves; no tensor or forward replay."""
    inherited, parents, i14_curves, saved_plans, _ = previous.bound_inputs()
    pinned = {name: previous._pinned_json(I16 / name, digest, "I16 " + name)
              for name, digest in PINS.items()}
    audit = pinned["analysis-001/audit.json"]
    summary = pinned["analysis-001/summary.json"]
    report = pinned["analysis-001/report-audit.json"]
    collection = pinned["raw-results-001/collection.json"]
    completion = previous._pinned_json(I16_ROOT / "confirmation/completion.json",
                                      I16_COMPLETION_SHA, "I16 completion")
    need(audit.get("status") == "pass" and audit.get("errors") == []
         and audit.get("artifact_root") == str(I16_ROOT)
         and audit.get("phase_completion_sha256", {}).get("confirmation")
             == I16_COMPLETION_SHA, "I16 audit binding differs")
    need(summary.get("audit_status") == "pass"
         and summary.get("schema") == "i16_scalar_analysis_summary_v1"
         and summary.get("artifact_root") == str(I16_ROOT), "I16 summary binding differs")
    need(report.get("status") == "pass" and report.get("errors") == []
         and report.get("maximum_absolute_numeric_difference") == 0
         and report.get("inputs", {}).get("i16_summary", {}).get("sha256")
             == PINS["analysis-001/summary.json"], "I16 report binding differs")
    need(collection.get("status") == "complete"
         and collection.get("source_root") == str(I16_ROOT)
         and collection.get("analysis_audit", {}).get("sha256")
             == PINS["analysis-001/audit.json"]
         and collection.get("phase_completion_sha256", {}).get("confirmation")
             == I16_COMPLETION_SHA, "I16 collection binding differs")
    need(completion.get("schema") == "i16_completion_v1"
         and completion.get("status") == "complete"
         and completion.get("phase") == "confirmation"
         and completion.get("frozen_commit") == I16_COMMIT
         and completion.get("branches") == 18
         and completion.get("numerical_failures") == 0
         and completion.get("completed_training_updates") == 34200
         and completion.get("all_requested_endpoints_present") is True,
         "I16 completion membership differs")
    artifacts = completion.get("artifacts")
    need(type(artifacts) is list and all(type(row) is dict for row in artifacts),
         "I16 artifact list differs")
    declared = {row.get("name"): row for row in artifacts}
    need(len(declared) == len(artifacts), "Duplicate I16 artifact names")
    directory = I16_ROOT / "confirmation"
    index_path = previous.previous.checked_artifact(directory, declared.get("branches.json"))
    entries = json.loads(index_path.read_text()).get("entries")
    expected_members = {(seed, target, k) for seed in SEEDS for target in TARGETS
                        for k in (0.0, 0.5, 0.9)}
    need(type(entries) is list and len(entries) == 18
         and {(row.get("seed"), row.get("target"), row.get("k")) for row in entries}
             == expected_members, "I16 branch index membership differs")
    trajectories = summary.get("all_policy_trajectories")
    need(type(trajectories) is list and len(trajectories) == 30,
         "I16 trajectory summary membership differs")
    selected = [row for row in trajectories if row.get("policy") == "k0"]
    references = {(row.get("seed"), row.get("target")): row for row in selected}
    expected_keys = {(seed, target) for seed in SEEDS for target in TARGETS}
    need(len(selected) == 6 and set(references) == expected_keys, "I16 k0 membership differs")
    records = []
    for entry in entries:
        if entry["k"] != 0.0:
            continue
        key = (entry["seed"], entry["target"])
        artifact = entry.get("artifact")
        need(type(artifact) is dict and artifact == declared.get(artifact.get("name")),
             "I16 k0 branch artifact differs")
        path = previous.previous.checked_artifact(directory, artifact)
        branch = json.loads(path.read_text())
        expected_start = next(row for row in i14_curves[key + ("raw",)]["curve"]
                              if row["horizon"] == 100)
        identity_fields = ("id", "seed", "target", "k", "k_label", "status",
                           "completed_updates", "parent_state_digest", "parent_evaluation_digest")
        need(branch.get("schema") == "i16_scalar_branch_v1"
             and all(branch.get(field) == entry.get(field) for field in identity_fields)
             and branch.get("k") == 0.0 and branch.get("k_label") == "k0"
             and branch.get("base") == "sgdm" and branch.get("lr") == .03
             and branch.get("status") == "complete" and branch.get("failure") is None
             and branch.get("completed_updates") == 1900
             and branch.get("parent_state_digest") == parents[key]["state_digest"]
             and branch.get("parent_evaluation_digest") == i9.tree_digest(expected_start)
             and references[key].get("status") == "complete"
             and references[key].get("failure") is None
             and previous._summary_curve_matches(references[key], branch),
             "I16 k0 branch or summary content differs")
        checkpoints = branch.get("checkpoints")
        need(type(checkpoints) is list
             and [row.get("horizon") for row in checkpoints] == list(HORIZONS[1:]),
             "I16 k0 checkpoint horizons differ")
        for checkpoint in checkpoints:
            kind = "full_state" if checkpoint["horizon"] == 2000 else "model_state"
            record = checkpoint.get(kind)
            need(type(record) is dict and record == declared.get(record.get("name")),
                 "I16 k0 checkpoint manifest differs")
            previous.previous.checked_artifact(directory, record)
        records.append({"seed": key[0], "target": key[1], "curve_artifact": artifact,
                        "checkpoint_records": checkpoints,
                        "parent_state_digest": branch["parent_state_digest"],
                        "parent_evaluation_digest": branch["parent_evaluation_digest"]})
    need(len(records) == 6, "Missing I16 k0 direct references")
    binding = {"schema": "i17_parent_and_reference_binding_v1", "inherited_i16": inherited,
               "i16": {"artifact_root": str(I16_ROOT), "acquisition_commit": I16_COMMIT,
                       "pinned_sha256": PINS, "confirmation_sha256": I16_COMPLETION_SHA},
               "k0_references": records, "source_replayed": False}
    return binding, parents, i14_curves, saved_plans, references


def run_branch(run, state, data, plan, target, policy, expected_start, *,
               smoke=False, end=2000, horizons=HORIZONS):
    need(type(policy) is str and policy in core.REAL_POLICIES and target in TARGETS,
         "Unregistered real gain-normalized arm")
    identity = f"s{plan.get('seed')}-sgdm-{target}-{policy}"
    if hasattr(run, "path"):
        need(not any(run.path.glob(f"*-{identity}*")), "Branch identity already has artifacts")
    core.validate_parent_snapshot(state)
    if smoke:
        need(end == 110 and tuple(horizons) == (100, 110), "Synthetic dimensions differ")
    else:
        plans.validate_plan(plan)
        need(end == 2000 and tuple(horizons) == HORIZONS, "Real horizon differs")
    device = previous._validate_data(data, state, smoke)
    batches_raw = plan.get("training_batches")
    need(type(batches_raw) is np.ndarray and batches_raw.dtype == np.int64
         and batches_raw.shape == (end, 64)
         and bool(((batches_raw >= 0) & (batches_raw < len(data["x"]))).all()),
         "Batch plan differs")
    parent_digest = i9.tree_digest(state)
    model, optimizer, tracker = c14.restore(state, device)
    need(i9.tree_digest(c14.snapshot(model, optimizer, tracker, "sgdm", .03))
         == parent_digest, "Complete-state restore seam differs")
    measured = {"horizon": 100, **old.evaluate(model, data)}
    need(measured == expected_start, "Parent evaluation seam differs")
    need(i9.tree_digest(c14.snapshot(model, optimizer, tracker, "sgdm", .03))
         == parent_digest, "Parent evaluation changed complete state")
    curve, diagnostics, checkpoints = [measured], [], []
    batches = torch.as_tensor(batches_raw, dtype=torch.long, device=device)
    labels = data["clean"] if target == "clean" else data["noisy"]
    failure, completed, update_seconds = None, 100, 0.0
    for step in range(101, end + 1):
        if step == 101 or step % 25 == 0:
            run.check()
        try:
            indices = batches[step - 1]
            started = time.monotonic()
            row = core.gain_step(model, optimizer, tracker, data["x"][indices],
                                 labels[indices], policy, capture_digests=(step == 101))
            update_seconds += time.monotonic() - started
            norm = row["displacement"]["data_norm"]
            path = {"data_step_squared_energy": norm * norm,
                    "data_path_length_increment": norm,
                    "raw_gradient_dot_data_step": row["displacement"]["raw_gradient_dot_data_delta"]}
            need(all(type(value) is float and math.isfinite(value) for value in path.values()),
                 "Gain path statistic is nonfinite")
            diagnostics.append({"step": step, "relative_step": step - 100,
                                "path_statistics": path, **row})
            completed = step
            if step in horizons:
                curve.append({"horizon": step, **old.evaluate(model, data)})
                checkpoint = {"horizon": step, "relative_horizon": step - 100}
                if step == end:
                    saved = c14.snapshot(model, optimizer, tracker, "sgdm", .03)
                    checkpoint.update(full_state=run.save(f"state-{identity}-h{step}.pt", saved, tensor=True),
                                      full_state_digest=i9.tree_digest(saved))
                else:
                    saved = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
                    checkpoint["model_state"] = run.save(f"model-{identity}-h{step}.pt", saved, tensor=True)
                checkpoints.append(checkpoint)
        except NUMERICAL_FAILURES as exc:
            failure = {"type": type(exc).__name__, "message": str(exc), "attempted_step": step,
                       "completed_step": completed, "last_valid_horizon": curve[-1]["horizon"]}
            failure["state_artifact"] = run.save(f"failed-state-{identity}.pt",
                c14.snapshot(model, optimizer, tracker, "sgdm", .03), tensor=True)
            break
    need(i9.tree_digest(state) == parent_digest, "Branch mutated its parent input")
    result = {"schema": "i17_gain_branch_v1", "id": identity, "seed": plan["seed"],
              "target": target, "policy": policy, "base": "sgdm", "lr": .03,
              "parent_horizon": 100, "parent_state_digest": parent_digest,
              "parent_evaluation_digest": i9.tree_digest(expected_start),
              "status": "numerical_failure" if failure else "complete",
              "requested_updates": end - 100, "completed_updates": completed - 100,
              "last_completed_horizon": completed, "curve": curve, "steps": diagnostics,
              "update_seconds": update_seconds, "checkpoints": checkpoints, "failure": failure,
              "first_step_digests": diagnostics[0]["digests"] if diagnostics else None,
              "selected_horizons": old.selection(curve) if failure is None else None}
    artifact = run.save(f"branch-{identity}.json", result)
    print(json.dumps({"branch": identity, "status": result["status"],
                      "elapsed_seconds": time.monotonic() - run.started}), flush=True)
    fields = ("id", "seed", "target", "policy", "status", "completed_updates",
              "parent_state_digest", "parent_evaluation_digest", "first_step_digests", "update_seconds")
    return {name: result[name] for name in fields} | {"artifact": artifact}


def validate_first_step_pairs(entries):
    checks = []
    for seed, target in sorted({(row["seed"], row["target"]) for row in entries}):
        group = [row for row in entries if (row["seed"], row["target"]) == (seed, target)]
        need(len(group) == 4 and {row["policy"] for row in group} == set(core.REAL_POLICIES),
             "First-step policy membership differs")
        for name in ("parent_state_digest", "parent_evaluation_digest"):
            need(len({row[name] for row in group}) == 1, "Parent pair differs: " + name)
        available = [row for row in group if row["first_step_digests"] is not None]
        need(all(row["status"] == "numerical_failure" for row in group
                 if row["first_step_digests"] is None), "Complete branch lacks first-step evidence")
        for name in ("raw_gradient", "post_observer", "old_momentum_buffer"):
            need(len({row["first_step_digests"][name] for row in available}) <= 1,
                 "First-step common tensor differs: " + name)
        checks.append({"seed": seed, "target": target, "available_first_steps": len(available),
                       "status": "pass" if len(available) == 4 else "partial_numerical_evidence"})
    return checks


def smoke_forecast(entries):
    need(len(entries) == 8 and {(row["target"], row["policy"]) for row in entries}
         == {(target, policy) for target in TARGETS for policy in core.REAL_POLICIES},
         "Smoke membership differs")
    rates = {}
    for policy in core.REAL_POLICIES:
        group = [row for row in entries if row["policy"] == policy]
        need(all(row["seed"] == 217 and row["status"] == "complete"
                 and row["completed_updates"] == 10
                 and type(row["update_seconds"]) is float
                 and math.isfinite(row["update_seconds"]) and row["update_seconds"] > 0
                 for row in group), "Smoke policy did not qualify")
        rates[policy] = sum(row["update_seconds"] for row in group) / 20
    return {"policy_seconds_per_update": rates,
            "confirmation_seconds": TIMING_SAFETY_FACTOR * max(rates.values()) * CONFIRMATION_UPDATES + 60}


def verify_smoke(root, sources, commit):
    """Validate the one completed synthetic phase without re-executing it."""
    directory = root / "smoke"
    need(directory.is_dir() and not directory.is_symlink(), "Smoke directory differs")
    completion_path = directory / "completion.json"
    attempt_path = root / "attempt-smoke.json"
    need(all(path.is_file() and not path.is_symlink() for path in (completion_path, attempt_path)),
         "Smoke terminal/attempt must be regular files")
    attempt = json.loads(attempt_path.read_text())
    need(attempt.get("schema") == "i17_attempt_v1" and attempt.get("phase") == "smoke"
         and attempt.get("frozen_commit") == commit and attempt.get("source_hashes") == sources
         and attempt.get("restart") == "forbidden", "Smoke attempt binding differs")
    completion = json.loads(completion_path.read_text())
    need(completion.get("schema") == "i17_completion_v1" and completion.get("status") == "complete"
         and completion.get("phase") == "smoke" and completion.get("frozen_commit") == commit
         and completion.get("source_hashes") == sources and completion.get("branches") == 8
         and completion.get("numerical_failures") == 0
         and completion.get("all_requested_endpoints_present") is True
         and completion.get("completed_training_updates") == 280, "Smoke did not qualify")
    records = completion.get("artifacts")
    need(type(records) is list and all(type(row) is dict for row in records), "Smoke artifact list differs")
    declared = {row.get("name"): row for row in records}
    expected_names = {"manifest.json", "branches.json", "synthetic-parent-clean.pt", "synthetic-parent-fixed.pt"}
    for target in TARGETS:
        for policy in core.REAL_POLICIES:
            identity = f"s217-sgdm-{target}-{policy}"
            expected_names.update((f"branch-{identity}.json", f"state-{identity}-h110.pt"))
    need(len(records) == len(declared) and set(declared) == expected_names,
         "Smoke declared artifact membership differs")
    need({path.name for path in directory.iterdir()} == expected_names | {"completion.json"},
         "Smoke physical artifact membership differs")
    for record in records:
        previous.previous.checked_artifact(directory, record)
    manifest = json.loads((directory / "manifest.json").read_text())
    need(manifest.get("schema") == "i17_manifest_v1" and manifest.get("phase") == "smoke"
         and manifest.get("frozen_commit") == commit and manifest.get("source_hashes") == sources
         and manifest.get("real_policies") == list(core.REAL_POLICIES)
         and manifest.get("test_only_policies") == list(core.TEST_ONLY_POLICIES),
         "Smoke manifest binding differs")
    index = json.loads((directory / "branches.json").read_text())
    entries = index.get("entries")
    forecast = smoke_forecast(entries)
    pairs = validate_first_step_pairs(entries)
    need(all(row["status"] == "pass" for row in pairs)
         and index.get("first_step_pair_checks") == pairs
         and index.get("synthetic_warmup_updates") == 200
         and index.get("new_gain_updates") == 80, "Smoke pair/count evidence differs")
    for entry in entries:
        artifact = entry.get("artifact")
        need(type(artifact) is dict and artifact == declared.get(artifact.get("name")),
             "Smoke indexed branch artifact differs")
        branch = json.loads((directory / artifact["name"]).read_text())
        need(branch.get("schema") == "i17_gain_branch_v1"
             and all(branch.get(name) == value for name, value in entry.items() if name != "artifact")
             and branch.get("failure") is None
             and [row.get("horizon") for row in branch.get("curve", [])] == [100, 110]
             and len(branch.get("steps", [])) == 10, "Smoke branch content differs")
        state_name = f"state-{entry['id']}-h110.pt"
        checkpoints = branch.get("checkpoints")
        need(type(checkpoints) is list and len(checkpoints) == 1
             and checkpoints[0].get("horizon") == 110
             and checkpoints[0].get("relative_horizon") == 10
             and checkpoints[0].get("full_state") == declared.get(state_name)
             and re.fullmatch(r"[0-9a-f]{64}", checkpoints[0].get("full_state_digest", "")) is not None,
             "Smoke terminal checkpoint association differs")
    need(forecast["confirmation_seconds"] <= LIMITS["confirmation"], "Smoke forecast exceeds cap")
    return forecast


def synthetic_smoke(run):
    generator = np.random.default_rng(2026090817)
    x = torch.as_tensor(generator.random((128, 784)), dtype=torch.float32, device="cuda")
    y = torch.as_tensor(generator.integers(0, 10, 128), dtype=torch.long, device="cuda")
    data = {"x": x[:64], "clean": y[:64], "noisy": (y[:64] + 1) % 10,
            "vx": x[64:96], "vy": y[64:96], "ax": x[96:], "ay": y[96:]}
    plan = {"seed": 217, "training_batches": generator.integers(0, 64, (110, 64), dtype=np.int64)}
    entries = []
    for target in TARGETS:
        old.seed_all(2026090817)
        model = i9.make_model(2026090817, "cuda")
        optimizer = c14.make_optimizer(model, "sgdm", .03)
        tracker = i9.make_tracker(model, optimizer)
        labels = data["clean"] if target == "clean" else data["noisy"]
        for step in range(1, 101):
            run.check()
            indices = torch.as_tensor(plan["training_batches"][step - 1], device="cuda")
            c14.train_step(model, optimizer, tracker, data["x"][indices], labels[indices], "raw", "sgdm")
        state = c14.snapshot(model, optimizer, tracker, "sgdm", .03)
        run.save(f"synthetic-parent-{target}.pt", state, tensor=True)
        expected = {"horizon": 100, **old.evaluate(model, data)}
        del model, optimizer, tracker
        for policy in core.REAL_POLICIES:
            entries.append(run_branch(run, state, data, plan, target, policy, expected,
                                      smoke=True, end=110, horizons=(100, 110)))
    pairs = validate_first_step_pairs(entries)
    run.save("branches.json", {"entries": entries, "first_step_pair_checks": pairs,
                              "synthetic_warmup_updates": 200, "new_gain_updates": 80})
    return entries, 200


def confirmation(run):
    binding, parents, references, saved_plans, k0 = bound_inputs()
    run.save("parent-inputs.json", binding)
    x, y = plans.read_training()
    data_cache, state_cache, seams = {}, {}, []
    for seed in SEEDS:
        data, corruption = plans.data_for_plan(x, y, saved_plans[seed], "cuda")
        need(corruption == binding["inherited_i16"]["i14"]["corruption_counts"][str(seed)],
             "Fixed-corruption counts differ")
        data_cache[seed] = data
        run.save(f"plan-s{seed}.json", saved_plans[seed])
        run.save(f"corruption-s{seed}.json", corruption)
        for target in TARGETS:
            run.check()
            parent = parents[seed, target]
            path = previous.previous.checked_artifact(previous.previous.SOURCE_ROOT / "confirmation",
                                                      parent["artifact"])
            state = torch.load(path, map_location="cpu", weights_only=False)
            core.validate_parent_snapshot(state)
            need(i9.tree_digest(state) == parent["state_digest"], "Parent tensor digest differs")
            model, optimizer, tracker = c14.restore(state, "cuda")
            before = i9.tree_digest(c14.snapshot(model, optimizer, tracker, "sgdm", .03))
            rng_before = i9._rng_state()
            need(before == parent["state_digest"], "Preflight restore seam differs")
            measured = {"horizon": 100, **old.evaluate(model, data)}
            raw_start = next(row for row in references[seed, target, "raw"]["curve"] if row["horizon"] == 100)
            current_start = next(row for row in references[seed, target, "current32"]["curve"] if row["horizon"] == 100)
            need(measured == raw_start == current_start
                 and previous._summary_h100_matches(k0[seed, target], measured),
                 "Preflight inherited evaluation seam differs")
            need(i9.tree_digest(c14.snapshot(model, optimizer, tracker, "sgdm", .03)) == before
                 and i9.equal_tree(rng_before, i9._rng_state()), "Evaluation mutated complete state/RNG")
            state_cache[seed, target] = state
            seams.append({"seed": seed, "target": target, "state_digest": before,
                          "evaluation_digest": i9.tree_digest(measured), "rng_neutral": True, "status": "pass"})
            del model, optimizer, tracker
    run.save("parent-seams.json", {"parents": seams, "parents_loaded": 6,
                                  "scientific_updates_before_admission": 0})
    entries = []
    for seed in SEEDS:
        for target in TARGETS:
            expected = next(row for row in references[seed, target, "raw"]["curve"] if row["horizon"] == 100)
            for policy in core.REAL_POLICIES:
                entries.append(run_branch(run, state_cache[seed, target], data_cache[seed],
                                          saved_plans[seed], target, policy, expected))
    need(len(entries) == 24 and {(r["seed"], r["target"], r["policy"]) for r in entries}
         == {(s, t, p) for s in SEEDS for t in TARGETS for p in core.REAL_POLICIES},
         "I17 confirmation membership differs")
    need(bound_inputs()[0] == binding, "Bound old inputs changed during I17")
    run.save("branches.json", {"entries": entries, "first_step_pair_checks": validate_first_step_pairs(entries),
                              "new_gain_branches": 24, "reused_i16_k0": 6})
    return entries, 0


def main():
    if not __debug__:
        raise RuntimeError("Assertions must remain enabled")
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--phase", required=True, choices=tuple(LIMITS))
    parser.add_argument("--frozen-commit", required=True)
    args = parser.parse_args()
    need(not args.root.is_symlink(), "Root must not be a symlink")
    root = args.root.resolve(strict=True)
    need(root.parent == Path("/tmp/spectral-experiment-artifacts") and root.name.startswith("spectral-i17-001."),
         "I17 requires its exclusive large-volume root")
    need(subprocess.check_output(["findmnt", "-n", "-o", "SOURCE", "-T", str(root)], text=True).strip()
         == "/dev/RECONFIGURE_FOR_LOCAL_STORAGE", "I17 root is not on the intended volume")
    need(Path(os.environ.get("TMPDIR", "")).resolve() == root / "runtime"
         and (root / "runtime").is_dir() and not (root / "runtime").is_symlink(),
         "I17 requires its declared runtime directory")
    sources = source_manifest(args.frozen_commit)
    forecast = None
    if args.phase == "confirmation":
        forecast = verify_smoke(root, sources, args.frozen_commit)
    with (root / f"attempt-{args.phase}.json").open("xb") as handle:
        handle.write(old.json_bytes({"schema": "i17_attempt_v1", "phase": args.phase,
            "frozen_commit": args.frozen_commit, "source_hashes": sources,
            "pid": os.getpid(), "restart": "forbidden"}))
    run = old.Run(root, args.phase, LIMITS[args.phase])
    try:
        old.configure()
        run.save("manifest.json", {"schema": "i17_manifest_v1", "phase": args.phase,
            "frozen_commit": args.frozen_commit, "source_hashes": sources,
            "torch_version": str(torch.__version__), "numpy_version": np.__version__,
            "python_version": sys.version, "gpu": torch.cuda.get_device_name(),
            "real_policies": core.REAL_POLICIES, "test_only_policies": core.TEST_ONLY_POLICIES,
            "horizons": HORIZONS, "smoke_forecast": forecast,
            "timing_safety_factor": TIMING_SAFETY_FACTOR, "cloud_spend_usd": 0,
            "authorized_budget_usd": 100, "shared_artifact_cap_bytes": old.ARTIFACT_CAP,
            "wall_limit_seconds": LIMITS[args.phase], "runtime_directory": "runtime",
            "runtime_bytes_count_in_cap": True})
        entries, warmup = synthetic_smoke(run) if args.phase == "smoke" else confirmation(run)
        need(source_manifest(args.frozen_commit) == sources, "Source changed during acquisition")
        failures = sum(row["status"] == "numerical_failure" for row in entries)
        run.terminal("completion.json", {"schema": "i17_completion_v1", "status": "complete",
            "phase": args.phase, "source_hashes": sources, "frozen_commit": args.frozen_commit,
            "branches": len(entries), "numerical_failures": failures,
            "all_requested_endpoints_present": failures == 0,
            "completed_training_updates": warmup + sum(row["completed_updates"] for row in entries),
            "update_seconds": sum(row["update_seconds"] for row in entries),
            "elapsed_seconds": time.monotonic() - run.started,
            "peak_torch_gpu_bytes": torch.cuda.max_memory_allocated(),
            "shared_artifact_bytes_before_completion": run.used(), "artifacts": run.artifacts})
    except BaseException as exc:
        run.terminal("failure.json", {"schema": "i17_phase_failure_v1", "status": "failed",
            "phase": args.phase, "exception_type": type(exc).__name__, "message": str(exc)[:2000],
            "source_hashes": sources, "elapsed_seconds": time.monotonic() - run.started,
            "artifacts": run.artifacts, "restart": "forbidden"})
        raise


if __name__ == "__main__":
    main()
