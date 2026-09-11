#!/usr/bin/env python3
"""I10 one-off cloned-state branches; no source trajectory restart."""
import argparse
import hashlib
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
I9 = HERE.parent / "iteration-009"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(I9))
import numpy as np
import torch
import neural_core as core
import run_neural_mechanism as parent_run
import branch_core

PARENTS = Path("/tmp/spectral-experiment-artifacts/spectral-i9-001.2ykC0Z/artifacts")
SEEDS, ANCHORS = [100, 101, 102], [100, 500, 1500, 2000]
OBJECTIVES, POLICIES = ["fixed", "soft", "redraw"], ["raw", "current32", "frozen32"]
HORIZONS = [0, 1, 10, 50, 100, 250, 500]
ARTIFACT_CAP = 2 * 1024**3
digest = parent_run.digest


class Run(parent_run.Run):
    def used(self):
        # One shared I10 root includes both synthetic smoke and full artifacts.
        return sum(p.stat().st_size for p in self.path.parent.rglob("*") if p.is_file())

    def save(self, name, value, tensor=False):
        self.check()
        target = self.path / name
        remaining = ARTIFACT_CAP - self.used() - 1024**2
        if tensor:
            with target.open("xb") as handle:
                torch.save(value, parent_run.BudgetWriter(handle, remaining))
        else:
            payload = (json.dumps(parent_run.json_tree(value), indent=2, allow_nan=False) + "\n").encode()
            if len(payload) > remaining:
                raise RuntimeError("I10 shared artifact budget exhausted")
            with target.open("xb") as handle:
                handle.write(payload)
        record = {"name": name, "bytes": target.stat().st_size, "sha256": digest(target)}
        self.artifacts.append(record)
        self.check()
        return record


def source_manifest():
    old = json.loads((I9 / "artifacts/manifest.json").read_text())["source_hashes"]
    for path, expected in old.items():
        assert digest(ROOT / path) == expected, path
    result = dict(old)
    for path in [Path(__file__).resolve(), HERE / "branch_core.py",
                 HERE / "test_branch_core.py", HERE / "test_runner.py", HERE / "protocol.md"]:
        result[str(path.relative_to(ROOT))] = digest(path)
    return result


def branch_plan(seed, step):
    def rng(stream):
        return np.random.default_rng(np.random.SeedSequence([20260907, 10, seed, step, stream]))
    return {"seed": seed, "parent_step": step,
            "batches": rng(0).integers(0, 5000, (500, 64)),
            "redraw_mask": rng(1).random((500, 64)) < .9,
            "redraw_digits": rng(2).integers(0, 10, (500, 64))}


def verify_inputs():
    local_completion = I9 / "artifacts/completion.json"
    assert digest(PARENTS / "completion.json") == digest(local_completion)
    completion = json.loads(local_completion.read_text())
    assert completion["status"] == "complete"
    index = {row["name"]: row for row in completion["artifacts"]}
    names = ["manifest.json"]
    names += [f"plan-s{s}.json" for s in SEEDS]
    names += [f"anchor-s{s}-{source}-t{step}.pt" for s in SEEDS
              for source in ["raw", "current32"] for step in ANCHORS]
    input_hashes = {"completion.json": digest(PARENTS / "completion.json")}
    for name in names:
        record = index[name]
        path = PARENTS / name
        assert path.stat().st_size == record["bytes"]
        assert digest(path) == record["sha256"], path
        input_hashes[name] = record["sha256"]
    manifest = json.loads((PARENTS / "manifest.json").read_text())
    assert manifest["source_hashes"] == json.loads((I9 / "artifacts/manifest.json").read_text())["source_hashes"]
    for name, expected in manifest["data_hashes"].items():
        assert digest(parent_run.DATA / name) == expected
    return index, input_hashes, manifest["data_hashes"]


def load_parent(seed, source, step, index):
    name = f"anchor-s{seed}-{source}-t{step}.pt"
    binding = torch.load(PARENTS / name, weights_only=True, map_location="cpu")
    assert (binding["seed"], binding["source"], binding["completed_step"]) == (seed, source, step)
    assert binding["plan_file"] == f"plan-s{seed}.json"
    assert core.tree_digest(binding["state"]) == binding["complete_state_digest"]
    return binding, index[name]


def make_data(x, y, seed):
    saved = json.loads((PARENTS / f"plan-s{seed}.json").read_text())
    for key in ["train_indices", "validation_indices", "auxiliary_indices",
                "replacement_mask", "replacement_digits"]:
        saved[key] = np.asarray(saved[key])
    return parent_run.data_for_plan(x, y, saved)[0]


def target_for(data, indices, plan, position, objective):
    if objective == "fixed":
        return data["noisy"][indices]
    clean = data["clean"][indices]
    if objective == "soft":
        return core._soft(clean, 10)
    assert objective == "redraw"
    mask = torch.as_tensor(plan["redraw_mask"][position], device=indices.device)
    digits = torch.as_tensor(plan["redraw_digits"][position], device=indices.device)
    return torch.where(mask, digits, clean)


def execute_branch(run, state, data, plan, objective, policy, steps, horizons):
    model, opt, tracker = core.restore(state, "cuda")
    fingerprint = core.tree_digest(state)
    assert core.tree_digest(core.snapshot(model, opt, tracker)) == fingerprint
    frozen_basis = core._basis(tracker)
    assert frozen_basis is not None
    before = core.snapshot(model, opt, tracker)
    baseline = branch_core.evaluate(model, data)
    assert core.equal_tree(before, core.snapshot(model, opt, tracker))
    curve = [{"horizon": 0, **baseline}]
    diagnostics = []
    for h in range(1, steps + 1):
        run.check()
        indices = torch.as_tensor(plan["batches"][h - 1], device="cuda")
        target = target_for(data, indices, plan, h - 1, objective)
        row = branch_core.branch_step(model, opt, tracker, data["x"][indices], target,
                                      policy, frozen_basis)
        diagnostics.append({"horizon": h, **row})
        if h in horizons:
            before_eval = core.snapshot(model, opt, tracker)
            curve.append({"horizon": h, **branch_core.evaluate(model, data)})
            assert core.equal_tree(before_eval, core.snapshot(model, opt, tracker))
    final = core.snapshot(model, opt, tracker)
    if policy == "frozen32":
        assert core.equal_tree(state["tracker"], final["tracker"]), "frozen observer changed"
    assert core.tree_digest(state) == fingerprint, "in-memory parent changed"
    return curve, diagnostics, final


def smoke(run):
    gen = torch.Generator().manual_seed(19010)
    x = torch.rand((128, 784), generator=gen).cuda()
    y = torch.randint(0, 10, (128,), generator=gen).cuda()
    noisy = torch.randint(0, 10, (128,), generator=gen).cuda()
    data = {"x": x, "clean": y, "noisy": noisy, "vx": x, "vy": y,
            "ax": x.flip(0), "ay": y.flip(0)}
    model = core.make_model(19010, "cuda")
    opt = core.make_optimizer(model)
    tracker = core.make_tracker(model, opt)
    for step in range(100):
        core.train_step(model, opt, tracker, x[:64], noisy[:64], "current32")
    state = core.snapshot(model, opt, tracker)
    record = run.save("synthetic-parent.pt", state, tensor=True)
    state = torch.load(run.path / record["name"], weights_only=True, map_location="cpu")
    plan = branch_plan(19010, 100)
    plan["batches"] %= 128
    baseline = None
    for objective in OBJECTIVES:
        for policy in POLICIES:
            curve, _, final = execute_branch(run, state, data, plan, objective, policy, 10, [1, 10])
            if baseline is None:
                baseline = curve[0]
            assert curve[0] == baseline
            record = run.save(f"smoke-{objective}-{policy}.pt", final, tensor=True)
            loaded = torch.load(run.path / record["name"], weights_only=True, map_location="cpu")
            assert core.equal_tree(final, loaded)
    run.save("smoke-checks.json", {"status": "pass", "source_hashes": source_manifest(),
             "synthetic_only": True, "nine_arms": True, "initial_states_exact": True,
             "evaluations_neutral": True, "full_final_roundtrips": True,
             "torch": torch.__version__, "gpu": torch.cuda.get_device_name()})


def full(run, smoke_dir):
    smoke_dir = smoke_dir.resolve(strict=True)
    assert smoke_dir.parent == run.path.parent.resolve(), "smoke must share capped I10 root"
    smoke_completion = json.loads((smoke_dir / "completion.json").read_text())
    assert smoke_completion["status"] == "complete" and smoke_completion["mode"] == "smoke"
    smoke_record = next(row for row in smoke_completion["artifacts"]
                        if row["name"] == "smoke-checks.json")
    assert (smoke_dir / "smoke-checks.json").stat().st_size == smoke_record["bytes"]
    assert digest(smoke_dir / "smoke-checks.json") == smoke_record["sha256"]
    smoke_check = json.loads((smoke_dir / "smoke-checks.json").read_text())
    assert smoke_check["status"] == "pass" and smoke_check["synthetic_only"]
    assert smoke_check["source_hashes"] == source_manifest()
    assert smoke_check["torch"] == torch.__version__
    assert smoke_check["gpu"] == torch.cuda.get_device_name()
    index, input_hashes, data_hashes = verify_inputs()
    run.save("parent-inputs.json", {"directory": str(PARENTS), "hashes": input_hashes,
                                   "data_hashes": data_hashes})
    plans = {(s, t): branch_plan(s, t) for s in SEEDS for t in ANCHORS}
    for (s, t), plan in plans.items():
        run.save(f"plan-s{s}-t{t}.json", plan)
    x, y = parent_run.read_training()
    entries = []
    for seed in SEEDS:
        data = make_data(x, y, seed)
        for step in ANCHORS:
            for source in (["raw"] if step == 100 else ["raw", "current32"]):
                binding, parent_record = load_parent(seed, source, step, index)
                logical_sources, parent_records = [source], [parent_record]
                if step == 100:
                    twin, twin_record = load_parent(seed, "current32", step, index)
                    assert core.equal_tree(binding["state"], twin["state"])
                    assert twin["complete_state_digest"] == binding["complete_state_digest"]
                    logical_sources.append("current32")
                    parent_records.append(twin_record)
                    del twin
                baseline = None
                for objective in OBJECTIVES:
                    for policy in POLICIES:
                        identity = f"s{seed}-{source}-t{step}-{objective}-{policy}"
                        curve, diagnostics, final = execute_branch(
                            run, binding["state"], data, plans[(seed, step)],
                            objective, policy, 500, HORIZONS)
                        if baseline is None:
                            baseline = curve[0]
                        assert curve[0] == baseline, "branch baseline mismatch"
                        metadata = {"id": identity, "seed": seed, "parent_step": step,
                                    "physical_source": source, "logical_sources": logical_sources,
                                    "objective": objective, "policy": policy,
                                    "parent_artifacts": parent_records,
                                    "parent_complete_digest": binding["complete_state_digest"],
                                    "plan_file": f"plan-s{seed}-t{step}.json", "completed_horizon": 500}
                        final_digest = core.tree_digest(final)
                        snapshot_record = run.save(f"final-{identity}.pt",
                            {**metadata, "state": final, "complete_state_digest": final_digest},
                            tensor=True)
                        record = run.save(f"branch-{identity}.json",
                            {**metadata, "curve": curve, "steps": diagnostics,
                             "final_artifact": snapshot_record, "final_complete_digest": final_digest})
                        entries.append({**metadata, "branch_artifact": record,
                                        "final_artifact": snapshot_record})
                        print(json.dumps({"branch": identity, "finished": len(entries),
                                          "expected": 189, "elapsed_seconds": time.monotonic() - run.started}),
                              flush=True)
                        del final, diagnostics, curve
                        torch.cuda.empty_cache()
                del binding
        del data
    assert len(entries) == 189
    assert sum(len(row["logical_sources"]) for row in entries) == 216
    for name, expected in input_hashes.items():
        assert digest(PARENTS / name) == expected, "parent input changed"
    for name, expected in data_hashes.items():
        assert digest(parent_run.DATA / name) == expected, "training input changed"
    run.save("branches.json", {"unique_branches": 189, "logical_cells": 216, "entries": entries,
                               "parent_hashes_unchanged": True})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--full", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke-dir", type=Path)
    args = parser.parse_args()
    parent = args.output.parent.resolve(strict=True)
    assert parent.is_relative_to(Path("/tmp/spectral-experiment-artifacts"))
    mount = subprocess.check_output(["findmnt", "-n", "-o", "TARGET,SOURCE", "--target", str(parent)],
                                    text=True).split()
    assert mount == ["/private-artifacts/storage", "/dev/RECONFIGURE_FOR_LOCAL_STORAGE"]
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
                    "cloud_spend_usd": 0, "parent_directory": str(PARENTS),
                    "unique_branches": 189, "logical_cells": 216, "horizons": HORIZONS}
        run.save("manifest.json", manifest)
        if args.smoke:
            smoke(run)
        else:
            assert args.smoke_dir is not None
            full(run, args.smoke_dir)
        assert source_manifest() == manifest["source_hashes"]
        torch.cuda.synchronize()
        complete = {"status": "complete", "mode": manifest["mode"], "finished_unix": time.time(),
                    "wall_seconds": time.monotonic() - run.started,
                    "shared_root_artifact_bytes_before_completion": run.used(),
                    "peak_torch_gpu_bytes": torch.cuda.max_memory_allocated(),
                    "artifacts": list(run.artifacts)}
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
