#!/usr/bin/env python3
"""One-off I11 continuation of all completed I10 fixed-label lineages."""
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
I9 = HERE.parent / "iteration-009"
sys.path[:0] = [str(ROOT), str(I10), str(I9)]
import numpy as np
import torch
import run_branches as previous
import continue_core

core = previous.core
base = previous.parent_run
digest = base.digest
PARENTS = Path("/tmp/spectral-experiment-artifacts/spectral-i10-001.INWOm8/artifacts")
SEEDS, ANCHORS, POLICIES = previous.SEEDS, previous.ANCHORS, previous.POLICIES
STEPS = 1500
HORIZONS = [0, 100, 500, 1000, 1500]
ARTIFACT_CAP = 1024**3


class Run(base.Run):
    def used(self):
        return sum(p.stat().st_size for p in self.path.parent.rglob("*") if p.is_file())

    def save(self, name, value, tensor=False):
        self.check()
        target = self.path / name
        remaining = ARTIFACT_CAP - self.used() - 1024**2
        if tensor:
            with target.open("xb") as handle:
                torch.save(value, base.BudgetWriter(handle, remaining))
        else:
            raw = (json.dumps(base.json_tree(value), indent=2, allow_nan=False) + "\n").encode()
            if len(raw) > remaining:
                raise RuntimeError("I11 shared artifact cap exhausted")
            with target.open("xb") as handle:
                handle.write(raw)
        record = {"name": name, "bytes": target.stat().st_size, "sha256": digest(target)}
        self.artifacts.append(record)
        self.check()
        return record


def original_json(name):
    return json.loads(gzip.decompress((I10 / "raw-results" / (name + ".gz")).read_bytes()))


def source_manifest():
    sources = original_json("manifest.json")["source_hashes"]
    for path, expected in sources.items():
        assert digest(ROOT / path) == expected, path
    for name in ("run_continuation.py", "continue_core.py", "test_continue_core.py",
                 "test_runner.py", "protocol.md"):
        path = HERE / name
        sources[str(path.relative_to(ROOT))] = digest(path)
    return sources


def continuation_plan(seed, step):
    rng = np.random.default_rng(np.random.SeedSequence([20260907, 11, seed, step, 0]))
    return {"seed": seed, "i9_parent_step": step,
            "batches": rng.integers(0, 5000, (STEPS, 64))}


def expected_keys():
    return {(s, source, t, p) for s in SEEDS for t in ANCHORS
            for source in (["raw"] if t == 100 else ["raw", "current32"])
            for p in POLICIES}


def check_record(record):
    name = record["name"]
    assert Path(name).name == name
    path = PARENTS / name
    assert path.stat().st_size == record["bytes"], name
    assert digest(path) == record["sha256"], name


def verify_inputs():
    collection = json.loads((I10 / "raw-results/collection.json").read_text())
    assert digest(PARENTS / "completion.json") == collection["source_completion_sha256"]
    completion = json.loads((PARENTS / "completion.json").read_text())
    assert completion == original_json("completion.json")
    assert completion["status"] == "complete" and completion["mode"] == "full"
    index = {r["name"]: r for r in completion["artifacts"]}
    assert len(index) == len(completion["artifacts"])
    hashes = {"completion.json": digest(PARENTS / "completion.json")}
    for name in ("manifest.json", "branches.json"):
        check_record(index[name])
        hashes[name] = index[name]["sha256"]
    branches = json.loads((PARENTS / "branches.json").read_text())
    assert branches["unique_branches"] == 189 and branches["logical_cells"] == 216
    entries = [r for r in branches["entries"] if r["objective"] == "fixed"]
    assert len(entries) == 63
    keys = {(r["seed"], r["physical_source"], r["parent_step"], r["policy"]) for r in entries}
    assert keys == expected_keys()
    assert sum(len(r["logical_sources"]) for r in entries) == 72
    i9_index, i9_hashes, data_hashes = previous.verify_inputs()
    for entry in entries:
        assert entry["completed_horizon"] == 500
        expected_sources = ["raw", "current32"] if entry["parent_step"] == 100 else [entry["physical_source"]]
        assert entry["logical_sources"] == expected_sources
        expected_parents = [i9_index[f"anchor-s{entry['seed']}-{source}-t{entry['parent_step']}.pt"]
                            for source in expected_sources]
        assert entry["parent_artifacts"] == expected_parents
        for field in ("branch_artifact", "final_artifact"):
            record = entry[field]
            assert record == index[record["name"]]
            check_record(record)
            hashes[record["name"]] = record["sha256"]
    return entries, hashes, i9_index, i9_hashes, data_hashes


def load_entry(entry):
    branch = json.loads((PARENTS / entry["branch_artifact"]["name"]).read_text())
    final = torch.load(PARENTS / entry["final_artifact"]["name"], weights_only=True, map_location="cpu")
    for key, value in entry.items():
        if key not in ("branch_artifact", "final_artifact"):
            assert branch[key] == final[key] == value, key
    assert branch["final_artifact"] == entry["final_artifact"]
    assert core.tree_digest(final["state"]) == final["complete_state_digest"] == branch["final_complete_digest"]
    endpoint = next(r for r in branch["curve"] if r["horizon"] == 500)
    return final, endpoint


def smoke(run):
    gen = torch.Generator().manual_seed(19011)
    x = torch.rand((128, 784), generator=gen).cuda()
    clean = torch.randint(0, 10, (128,), generator=gen).cuda()
    noisy = torch.randint(0, 10, (128,), generator=gen).cuda()
    data = {"x": x, "clean": clean, "noisy": noisy, "vx": x, "vy": clean,
            "ax": x.flip(0), "ay": clean.flip(0)}
    model = core.make_model(19011, "cuda")
    optimizer = core.make_optimizer(model)
    tracker = core.make_tracker(model, optimizer)
    for _ in range(100):
        core.train_step(model, optimizer, tracker, x[:64], noisy[:64], "current32")
    state, frozen = core.snapshot(model, optimizer, tracker), core._basis(tracker)
    plan = continuation_plan(19011, 100)["batches"][:12] % 128
    checks = []
    for policy in POLICIES:
        direct_curve, _, direct = continue_core.continue_fixed(
            state, data, plan, policy, frozen, steps=12, eval_horizons=(0, 5, 12), check=run.check)
        _, _, middle = continue_core.continue_fixed(
            state, data, plan[:5], policy, frozen, steps=5, eval_horizons=(0, 5), check=run.check)
        rec = run.save(f"smoke-mid-{policy}.pt", middle, tensor=True)
        loaded = torch.load(run.path / rec["name"], weights_only=True, map_location="cpu")
        assert core.equal_tree(middle, loaded)
        tail_curve, _, final = continue_core.continue_fixed(
            loaded, data, plan[5:], policy, frozen, steps=7, eval_horizons=(0, 7), check=run.check)
        assert core.equal_tree(direct, final), policy
        assert {k: v for k, v in direct_curve[-1].items() if k != "horizon"} == {
            k: v for k, v in tail_curve[-1].items() if k != "horizon"}
        rec = run.save(f"smoke-final-{policy}.pt", final, tensor=True)
        assert core.equal_tree(final, torch.load(run.path / rec["name"], weights_only=True, map_location="cpu"))
        checks.append({"policy": policy, "split_full_state_equals_uninterrupted": True,
                       "final_digest": core.tree_digest(final)})
    run.save("smoke-checks.json", {"status": "pass", "synthetic_only": True, "checks": checks,
             "source_hashes": source_manifest(), "torch": torch.__version__,
             "gpu": torch.cuda.get_device_name()})


def verify_smoke(run, smoke_dir):
    smoke_dir = smoke_dir.resolve(strict=True)
    assert smoke_dir.parent == run.path.parent.resolve()
    completion = json.loads((smoke_dir / "completion.json").read_text())
    assert completion["status"] == "complete" and completion["mode"] == "smoke"
    record = next(r for r in completion["artifacts"] if r["name"] == "smoke-checks.json")
    assert (smoke_dir / record["name"]).stat().st_size == record["bytes"]
    assert digest(smoke_dir / record["name"]) == record["sha256"]
    checks = json.loads((smoke_dir / record["name"]).read_text())
    assert checks["status"] == "pass" and checks["synthetic_only"]
    assert checks["source_hashes"] == source_manifest()
    assert checks["torch"] == torch.__version__ and checks["gpu"] == torch.cuda.get_device_name()


def full(run, smoke_dir):
    verify_smoke(run, smoke_dir)
    entries, hashes, i9_index, i9_hashes, data_hashes = verify_inputs()
    run.save("parent-inputs.json", {"i10_directory": str(PARENTS), "i10_hashes": hashes,
             "i9_directory": str(previous.PARENTS), "i9_hashes": i9_hashes, "data_hashes": data_hashes})
    plans = {(s, t): continuation_plan(s, t) for s in SEEDS for t in ANCHORS}
    for (seed, step), plan in plans.items():
        run.save(f"plan-s{seed}-t{step}.json", plan)
    x, y = base.read_training()
    outputs = []
    for seed in SEEDS:
        data = previous.make_data(x, y, seed)
        for entry in [r for r in entries if r["seed"] == seed]:
            original, original_record = previous.load_parent(seed, entry["physical_source"], entry["parent_step"], i9_index)
            assert original["complete_state_digest"] == entry["parent_complete_digest"]
            model, optimizer, tracker = core.restore(original["state"], "cuda")
            frozen = core._basis(tracker)
            del model, optimizer, tracker
            parent, endpoint = load_entry(entry)
            curve, diagnostics, final = continue_core.continue_fixed(
                parent["state"], data, plans[(seed, entry["parent_step"])]["batches"],
                entry["policy"], frozen, steps=STEPS, eval_horizons=HORIZONS, check=run.check,
                expected_initial_evaluation={k: v for k, v in endpoint.items() if k != "horizon"})
            assert {k: v for k, v in curve[0].items() if k != "horizon"} == {
                k: v for k, v in endpoint.items() if k != "horizon"}, "I10/I11 seam differs"
            for row in curve:
                row["continuation_horizon"] = row["horizon"] + 500
            for row in diagnostics:
                row["continuation_horizon"] = row["horizon"] + 500
            metadata = {"id": entry["id"], "seed": seed, "i9_parent_step": entry["parent_step"],
                        "physical_source": entry["physical_source"], "logical_sources": entry["logical_sources"],
                        "objective": "fixed", "policy": entry["policy"],
                        "i10_branch_artifact": entry["branch_artifact"], "i10_final_artifact": entry["final_artifact"],
                        "i10_complete_digest": parent["complete_state_digest"],
                        "i9_parent_artifact": original_record, "i9_parent_complete_digest": original["complete_state_digest"],
                        "frozen_reference": "original I9-parent basis, unchanged across I10/I11",
                        "plan_file": f"plan-s{seed}-t{entry['parent_step']}.json",
                        "completed_horizon": STEPS, "completed_continuation_horizon": 2000,
                        "seam_evaluation_exact": True}
            final_digest = core.tree_digest(final)
            final_record = run.save("final-" + entry["id"] + ".pt",
                {**metadata, "state": final, "complete_state_digest": final_digest}, tensor=True)
            branch_record = run.save("branch-" + entry["id"] + ".json",
                {**metadata, "curve": curve, "steps": diagnostics,
                 "final_artifact": final_record, "final_complete_digest": final_digest})
            outputs.append({**metadata, "branch_artifact": branch_record, "final_artifact": final_record})
            print(json.dumps({"branch": entry["id"], "finished": len(outputs), "expected": 63,
                              "elapsed_seconds": time.monotonic() - run.started}), flush=True)
            del original, frozen, parent, curve, diagnostics, final
            torch.cuda.empty_cache()
        del data
    assert len(outputs) == 63 and sum(len(r["logical_sources"]) for r in outputs) == 72
    for name, expected in hashes.items():
        assert digest(PARENTS / name) == expected
    for name, expected in i9_hashes.items():
        assert digest(previous.PARENTS / name) == expected
    for name, expected in data_hashes.items():
        assert digest(base.DATA / name) == expected
    run.save("branches.json", {"unique_branches": 63, "logical_cells": 72,
                              "entries": outputs, "parent_hashes_unchanged": True})


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
        base.configure()
        manifest = {"status": "running", "mode": "smoke" if args.smoke else "full",
                    "started_unix": time.time(), "pid": os.getpid(),
                    "systemd_invocation": os.environ.get("INVOCATION_ID"),
                    "source_hashes": source_manifest(),
                    "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                    "torch": torch.__version__, "gpu": torch.cuda.get_device_name(), "cloud_spend_usd": 0,
                    "parent_directory": str(PARENTS), "unique_branches": 63, "logical_cells": 72,
                    "horizons": HORIZONS, "continuation_horizons": [h + 500 for h in HORIZONS]}
        run.save("manifest.json", manifest)
        if args.smoke:
            smoke(run)
        else:
            assert args.smoke_dir is not None
            full(run, args.smoke_dir)
        assert source_manifest() == manifest["source_hashes"]
        torch.cuda.synchronize()
        result = {"status": "complete", "mode": manifest["mode"], "finished_unix": time.time(),
                  "wall_seconds": time.monotonic() - run.started,
                  "shared_root_artifact_bytes_before_completion": run.used(),
                  "peak_torch_gpu_bytes": torch.cuda.max_memory_allocated(), "artifacts": list(run.artifacts)}
        run.save("completion.json", result)
        print(json.dumps({k: v for k, v in result.items() if k != "artifacts"}), flush=True)
    except Exception as exc:
        with (run.path / "failure.json").open("x") as handle:
            json.dump({"status": "failed", "type": type(exc).__name__, "message": str(exc),
                       "finished_unix": time.time()}, handle, indent=2)
            handle.write("\n")
        raise


if __name__ == "__main__":
    main()
