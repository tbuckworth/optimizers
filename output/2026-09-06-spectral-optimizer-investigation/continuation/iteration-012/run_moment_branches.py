#!/usr/bin/env python3
"""I12 one-off moment/counter branches with immutable I10 baseline references."""
import argparse
import gzip
import json
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
I10 = HERE.parent / "iteration-010"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(I10))
sys.path.insert(0, str(HERE))
import numpy as np
import torch
import run_branches as previous
import moment_core

core = moment_core.i9
parent_run = previous.parent_run
digest = previous.digest
I10_ROOT = Path("/tmp/spectral-experiment-artifacts/spectral-i10-001.INWOm8/artifacts")
I9_ROOT = previous.PARENTS
SEEDS, ANCHORS = (100, 101, 102), (1500, 2000)
OBJECTIVES, POLICIES = ("soft", "redraw"), ("raw", "current32")
ARMS = ("zero_m", "zero_v", "zero_mv", "fresh_adam")
HORIZONS = (0, 1, 10, 50, 100, 250, 500)
ARTIFACT_CAP = 2 * 1024**3


class Run(previous.Run):
    def save(self, name, value, tensor=False):
        self.check()
        assert Path(name).name == name and name not in (".", "..")
        target = self.path / name
        remaining = ARTIFACT_CAP - self.used() - 1024**2
        if tensor:
            with target.open("xb") as handle:
                torch.save(value, parent_run.BudgetWriter(handle, remaining))
        else:
            payload = (json.dumps(parent_run.json_tree(value), indent=2, allow_nan=False) + "\n").encode()
            if len(payload) > remaining:
                raise RuntimeError("I12 shared artifact budget exhausted")
            with target.open("xb") as handle:
                handle.write(payload)
        record = {"name": name, "bytes": target.stat().st_size, "sha256": digest(target)}
        self.artifacts.append(record)
        self.check()
        return record


def expected_baseline_keys():
    return {(seed, step, objective, policy) for seed in SEEDS for step in ANCHORS
            for objective in OBJECTIVES for policy in POLICIES}


def expected_keys():
    return {(*key, arm) for key in expected_baseline_keys() for arm in ARMS}


def source_manifest():
    original = json.loads(gzip.decompress((I10 / "raw-results/manifest.json.gz").read_bytes()))
    old = original["source_hashes"]
    assert previous.source_manifest() == old
    for path, expected in old.items():
        assert digest(ROOT / path) == expected, path
    result = dict(old)
    for path in (Path(__file__).resolve(), HERE / "moment_core.py",
                 HERE / "test_moment_core.py", HERE / "test_runner.py", HERE / "protocol.md"):
        result[str(path.relative_to(ROOT))] = digest(path)
    return result


def check_record(root, record):
    assert Path(record["name"]).name == record["name"]
    path = root / record["name"]
    assert path.is_file() and path.stat().st_size == record["bytes"]
    assert digest(path) == record["sha256"], str(path)


def validate_plan(raw, seed, step):
    assert (raw["seed"], raw["parent_step"]) == (seed, step)
    plan = dict(raw)
    for name in ("batches", "redraw_mask", "redraw_digits"):
        plan[name] = np.asarray(raw[name])
        assert plan[name].shape == (500, 64)
    assert plan["batches"].dtype.kind in "iu"
    assert ((plan["batches"] >= 0) & (plan["batches"] < 5000)).all()
    assert plan["redraw_mask"].dtype == np.bool_
    assert plan["redraw_digits"].dtype.kind in "iu"
    assert ((plan["redraw_digits"] >= 0) & (plan["redraw_digits"] < 10)).all()
    return plan


def verify_inputs():
    collection = json.loads((I10 / "raw-results/collection.json").read_text())
    original = gzip.decompress((I10 / "raw-results/completion.json.gz").read_bytes())
    assert (I10_ROOT / "completion.json").read_bytes() == original
    assert digest(I10_ROOT / "completion.json") == collection["source_completion_sha256"]
    completion = json.loads(original)
    assert completion["status"] == "complete" and completion["mode"] == "full"
    index = {r["name"]: r for r in completion["artifacts"]}
    assert len(index) == len(completion["artifacts"])
    hashes = {"completion.json": digest(I10_ROOT / "completion.json")}
    for name in ("manifest.json", "branches.json"):
        check_record(I10_ROOT, index[name])
        hashes[name] = index[name]["sha256"]
    manifest = json.loads((I10_ROOT / "manifest.json").read_text())
    assert manifest["source_hashes"] == previous.source_manifest()
    old_index = json.loads((I10_ROOT / "branches.json").read_text())
    assert old_index["unique_branches"] == 189 and old_index["logical_cells"] == 216
    selected = [r for r in old_index["entries"] if r["physical_source"] == "current32"
                and r["parent_step"] in ANCHORS and r["objective"] in OBJECTIVES
                and r["policy"] in POLICIES]
    assert len(selected) == 24
    assert {(r["seed"], r["parent_step"], r["objective"], r["policy"])
            for r in selected} == expected_baseline_keys()
    baselines = {}
    for entry in selected:
        assert entry["logical_sources"] == ["current32"]
        assert len(entry["parent_artifacts"]) == 1 and entry["completed_horizon"] == 500
        for field in ("branch_artifact", "final_artifact"):
            record = entry[field]
            assert index[record["name"]] == record
            check_record(I10_ROOT, record)
            hashes[record["name"]] = record["sha256"]
        branch = json.loads((I10_ROOT / entry["branch_artifact"]["name"]).read_text())
        for key in ("id", "seed", "parent_step", "physical_source", "logical_sources",
                    "objective", "policy", "parent_artifacts", "parent_complete_digest",
                    "plan_file", "completed_horizon", "final_artifact"):
            assert branch[key] == entry[key], key
        assert [row["horizon"] for row in branch["curve"]] == list(HORIZONS)
        assert len(branch["steps"]) == 500
        key = (entry["seed"], entry["parent_step"], entry["objective"], entry["policy"])
        baselines[key] = {"entry": entry, "branch": branch}
    plans = {}
    for seed in SEEDS:
        for step in ANCHORS:
            name = f"plan-s{seed}-t{step}.json"
            check_record(I10_ROOT, index[name])
            hashes[name] = index[name]["sha256"]
            plans[(seed, step)] = validate_plan(json.loads((I10_ROOT / name).read_text()), seed, step)
    parent_index, parent_hashes, data_hashes = previous.verify_inputs()
    return baselines, plans, parent_index, {"i10_directory": str(I10_ROOT),
        "i10_hashes": hashes, "i9_directory": str(I9_ROOT), "i9_hashes": parent_hashes,
        "data_hashes": data_hashes, "baseline_count": 24, "parent_count": 6,
        "baseline_source_hashes": manifest["source_hashes"]}


def seal_branch(run, metadata, result):
    assert result["status"] in ("complete", "numerical_failure")
    metadata = {**metadata, "status": result["status"],
                "completed_steps": result["completed_steps"],
                "attempted_step": result["attempted_step"], "requested_horizon": 500}
    if result["status"] == "complete":
        assert result["completed_steps"] == 500
        assert len(result["steps"]) == 500
        assert [r["horizon"] for r in result["curve"]] == list(HORIZONS)
    if result["completed_steps"] > 0:
        value = result["first_step_applied_gradient_digest"]
        assert isinstance(value, str) and len(value) == 64
    terminal_digest, digest_reason = None, None
    try:
        terminal_digest = core.tree_digest(result["terminal_state"])
    except core.NeuralCoreError as exc:
        if result["status"] != "numerical_failure" or str(exc) != "nonfinite tree float":
            raise
        digest_reason = "Nonfinite Python scalar; terminal file hash binds exact saved bytes instead."
    terminal = run.save(f"terminal-{metadata['id']}.pt", {
        **metadata, "state": result["terminal_state"], "complete_state_digest": terminal_digest,
        "complete_state_digest_unavailable_reason": digest_reason}, tensor=True)
    last_good = None
    if result["status"] == "numerical_failure":
        state = result["last_evaluated_state"]
        last_good = run.save(f"last-evaluated-{metadata['id']}.pt", {
            **metadata, "last_evaluated_horizon": result["last_evaluated_horizon"],
            "state": state, "complete_state_digest": core.tree_digest(state)}, tensor=True)
    branch = run.save(f"branch-{metadata['id']}.json", {
        **metadata, "curve": result["curve"], "steps": result["steps"],
        "numerical_failure": result.get("numerical_failure"),
        "first_step_applied_gradient_digest": result.get("first_step_applied_gradient_digest"),
        "terminal_artifact": terminal, "terminal_complete_digest": terminal_digest,
        "terminal_digest_unavailable_reason": digest_reason,
        "last_evaluated_artifact": last_good,
        "last_evaluated_horizon": result.get("last_evaluated_horizon")})
    return {**metadata, "branch_artifact": branch, "terminal_artifact": terminal,
            "last_evaluated_artifact": last_good}


def smoke(run):
    gen = torch.Generator().manual_seed(12012)
    x = torch.rand((128, 784), generator=gen).cuda()
    y = torch.randint(0, 10, (128,), generator=gen).cuda()
    noisy = torch.randint(0, 10, (128,), generator=gen).cuda()
    data = {"x": x, "clean": y, "noisy": noisy, "vx": x, "vy": y,
            "ax": x.flip(0), "ay": y.flip(0)}
    model = core.make_model(12012, "cuda")
    opt = core.make_optimizer(model)
    tracker = core.make_tracker(model, opt)
    for _ in range(100):
        run.check()
        core.train_step(model, opt, tracker, x[:64], noisy[:64], "current32")
    state = core.snapshot(model, opt, tracker)
    record = run.save("synthetic-parent.pt", state, tensor=True)
    state = torch.load(run.path / record["name"], weights_only=True, map_location="cpu")
    plan = previous.branch_plan(12012, 100)
    plan["batches"] %= 128
    short_plan = {key: value[:10] if isinstance(value, np.ndarray) else value
                  for key, value in plan.items()}
    checks = []
    for objective in OBJECTIVES:
        for policy in POLICIES:
            old_curve, old_steps, old_final = previous.execute_branch(
                run, state, data, plan, objective, policy, 10, (0, 1, 10))
            inherited = moment_core.run_branch(state, data, short_plan, objective, policy,
                steps=10, horizons=(0, 1, 10), check=run.check)
            assert inherited["status"] == "complete"
            assert inherited["curve"] == old_curve and inherited["steps"] == old_steps
            assert core.equal_tree(inherited["terminal_state"], old_final)
            expected = {k: v for k, v in old_curve[0].items() if k != "horizon"}
            first_gradient = inherited["first_step_applied_gradient_digest"]
            for arm in ARMS:
                edited, mutation = moment_core.edit_moments(state, arm)
                result = moment_core.run_branch(edited, data, short_plan, objective, policy,
                    steps=10, horizons=(0, 1, 10), check=run.check,
                    expected_initial_evaluation=expected)
                assert result["status"] == "complete"
                assert result["first_step_applied_gradient_digest"] == first_gradient
                record = run.save(f"smoke-{objective}-{policy}-{arm}.pt", result["terminal_state"], tensor=True)
                restored = torch.load(run.path / record["name"], weights_only=True, map_location="cpu")
                assert core.equal_tree(restored, result["terminal_state"])
                checks.append({"objective": objective, "policy": policy, "arm": arm,
                               "state_roundtrip_exact": True, "initial_evaluation_exact": True,
                               "first_gradient_equal_inherited": True,
                               "mutation_audit": mutation})
    assert len(checks) == 16
    run.save("smoke-checks.json", {"status": "pass", "synthetic_only": True,
        "checks": checks, "four_inherited_paths_equal_i10": True,
        "source_hashes": source_manifest(), "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name()})
    return {"attempted_branches": 16, "numerical_failures": 0, "science_complete": None}


def full(run, smoke_dir):
    smoke_dir = smoke_dir.resolve(strict=True)
    assert smoke_dir.parent == run.path.parent.resolve()
    completion = json.loads((smoke_dir / "completion.json").read_text())
    assert completion["status"] == "complete" and completion["mode"] == "smoke"
    record = next(r for r in completion["artifacts"] if r["name"] == "smoke-checks.json")
    check_record(smoke_dir, record)
    checks = json.loads((smoke_dir / record["name"]).read_text())
    assert checks["status"] == "pass" and checks["synthetic_only"]
    assert checks["source_hashes"] == source_manifest()
    assert checks["torch"] == torch.__version__ and checks["gpu"] == torch.cuda.get_device_name()
    baselines, plans, parent_index, inputs = verify_inputs()
    run.save("parent-inputs.json", inputs)
    run.save("baseline-references.json", {"directory": str(I10_ROOT),
        "entries": [baselines[key]["entry"] for key in sorted(baselines)],
        "count": 24, "new_baseline_training_updates": 0})
    for (seed, step), plan in plans.items():
        name = f"plan-s{seed}-t{step}.json"
        saved = run.save(name, plan)
        assert saved["sha256"] == inputs["i10_hashes"][name], "copied I10 plan bytes differ"
    x, y = parent_run.read_training()
    entries, edited_entries = [], []
    first_gradients = {}
    for seed in SEEDS:
        data = previous.make_data(x, y, seed)
        for step in ANCHORS:
            binding, parent_record = previous.load_parent(seed, "current32", step, parent_index)
            for arm in ARMS:
                edited, mutation = moment_core.edit_moments(binding["state"], arm)
                edited_digest = core.tree_digest(edited)
                edited_record = run.save(f"edited-s{seed}-t{step}-{arm}.pt", {
                    "seed": seed, "parent_step": step, "arm": arm,
                    "parent_artifact": parent_record,
                    "parent_complete_digest": binding["complete_state_digest"],
                    "state": edited, "complete_state_digest": edited_digest,
                    "mutation_audit": mutation}, tensor=True)
                edited_entries.append({"seed": seed, "parent_step": step, "arm": arm,
                                       "artifact": edited_record, "complete_state_digest": edited_digest})
                for objective in OBJECTIVES:
                    for policy in POLICIES:
                        baseline = baselines[(seed, step, objective, policy)]
                        assert baseline["entry"]["parent_artifacts"] == [parent_record]
                        assert baseline["entry"]["parent_complete_digest"] == binding["complete_state_digest"]
                        expected = {k: v for k, v in baseline["branch"]["curve"][0].items() if k != "horizon"}
                        result = moment_core.run_branch(edited, data, plans[(seed, step)],
                            objective, policy, steps=500, horizons=HORIZONS, check=run.check,
                            expected_initial_evaluation=expected)
                        first_gradient = result["first_step_applied_gradient_digest"]
                        if first_gradient is not None:
                            gradient_key = (seed, step, objective, policy)
                            if gradient_key in first_gradients:
                                assert first_gradients[gradient_key] == first_gradient
                            else:
                                first_gradients[gradient_key] = first_gradient
                        assert core.tree_digest(edited) == edited_digest
                        metadata = {"id": f"s{seed}-current32-t{step}-{objective}-{policy}-{arm}",
                            "seed": seed, "parent_step": step, "source": "current32",
                            "objective": objective, "policy": policy, "arm": arm,
                            "parent_artifact": parent_record,
                            "parent_complete_digest": binding["complete_state_digest"],
                            "edited_parent_artifact": edited_record,
                            "edited_parent_complete_digest": edited_digest,
                            "baseline_branch_artifact": baseline["entry"]["branch_artifact"],
                            "baseline_final_artifact": baseline["entry"]["final_artifact"],
                            "plan_file": f"plan-s{seed}-t{step}.json", "initial_evaluation_exact": True}
                        entries.append(seal_branch(run, metadata, result))
                        print(json.dumps({"branch": metadata["id"], "status": result["status"],
                            "finished": len(entries), "expected": 96,
                            "elapsed_seconds": time.monotonic() - run.started}), flush=True)
                        del result
                        torch.cuda.empty_cache()
                assert core.tree_digest(binding["state"]) == binding["complete_state_digest"]
                del edited
            del binding
        del data
    assert len(entries) == 96 and len(edited_entries) == 24
    assert {(r["seed"], r["parent_step"], r["objective"], r["policy"], r["arm"])
            for r in entries} == expected_keys()
    for root, hashes in ((I10_ROOT, inputs["i10_hashes"]), (I9_ROOT, inputs["i9_hashes"]),
                         (parent_run.DATA, inputs["data_hashes"])):
        for name, expected in hashes.items():
            assert digest(root / name) == expected, "immutable input changed"
    failures = sum(r["status"] == "numerical_failure" for r in entries)
    result = {"attempted_branches": 96, "numerical_failures": failures,
              "science_complete": failures == 0}
    run.save("branches.json", {**result, "entries": entries, "edited_parents": edited_entries,
                               "baseline_count": 24, "parent_hashes_unchanged": True})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--smoke", action="store_true")
    modes.add_argument("--full", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke-dir", type=Path)
    args = parser.parse_args()
    parent = args.output.parent.resolve(strict=True)
    assert parent.is_relative_to(Path("/tmp/spectral-experiment-artifacts"))
    assert subprocess.check_output(["findmnt", "-n", "-o", "TARGET,SOURCE", "--target", str(parent)],
                                    text=True).split() == ["/private-artifacts/storage", "/dev/RECONFIGURE_FOR_LOCAL_STORAGE"]
    args.output.mkdir(exist_ok=False)
    run = Run(args.output, 100 if args.smoke else 1500)
    try:
        parent_run.configure()
        manifest = {"status": "running", "mode": "smoke" if args.smoke else "full",
            "started_unix": time.time(), "pid": os.getpid(),
            "systemd_invocation": os.environ.get("INVOCATION_ID"),
            "source_hashes": source_manifest(),
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "torch": torch.__version__, "gpu": torch.cuda.get_device_name(),
            "cloud_spend_usd": 0, "parent_directory": str(I9_ROOT),
            "baseline_directory": str(I10_ROOT), "planned_branches": 96, "baseline_count": 24,
            "edited_parent_count": 24, "arms": ARMS, "objectives": OBJECTIVES,
            "policies": POLICIES, "horizons": HORIZONS}
        run.save("manifest.json", manifest)
        if args.smoke:
            outcome = smoke(run)
        else:
            assert args.smoke_dir is not None
            outcome = full(run, args.smoke_dir)
        assert source_manifest() == manifest["source_hashes"]
        torch.cuda.synchronize()
        complete = {"status": "complete", "mode": manifest["mode"], **outcome,
            "finished_unix": time.time(), "wall_seconds": time.monotonic() - run.started,
            "shared_root_artifact_bytes_before_completion": run.used(),
            "peak_torch_gpu_bytes": torch.cuda.max_memory_allocated(), "artifacts": list(run.artifacts)}
        run.save("completion.json", complete)
        print(json.dumps({k: v for k, v in complete.items() if k != "artifacts"}), flush=True)
    except Exception as exc:
        with (run.path / "failure.json").open("x") as handle:
            json.dump({"status": "failed", "type": type(exc).__name__, "message": str(exc),
                       "finished_unix": time.time()}, handle, indent=2)
            handle.write("\n")
        raise


if __name__ == "__main__":
    main()
