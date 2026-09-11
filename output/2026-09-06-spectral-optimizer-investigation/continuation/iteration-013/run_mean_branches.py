#!/usr/bin/env python3
"""One-off I13 mean-preserving and matched-leak saved-state experiment."""
import argparse
import gzip
import json
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(name, "1")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
I12 = HERE.parent / "iteration-012"
I10 = HERE.parent / "iteration-010"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(I12))
sys.path.insert(0, str(HERE))
import numpy as np
import torch
import run_moment_branches as previous
import mean_core
import mean_probes

i9, i10 = mean_core.i9, mean_core.i10
parent_run = previous.parent_run
Run, digest = previous.Run, previous.digest
I9_ROOT, I10_ROOT = previous.I9_ROOT, previous.I10_ROOT
SEEDS, ANCHORS = (100, 101, 102), (1500, 2000)
OBJECTIVES = ("fixed", "soft", "redraw")
POLICIES = ("mean32", "leak01_32")
REFERENCES = ("raw", "current32")
HORIZONS = (0, 1, 10, 50, 100, 250, 500)


def expected_keys(policies=POLICIES):
    return {(s, t, o, p) for s in SEEDS for t in ANCHORS
            for o in OBJECTIVES for p in policies}


def source_manifest():
    old = json.loads(gzip.decompress((I12 / "raw-results/manifest.json.gz").read_bytes()))
    result = previous.source_manifest()
    assert result == old["source_hashes"]
    for name in ("protocol.md", "run_mean_branches.py", "test_runner.py", "mean_core.py",
                 "test_mean_core.py", "mean_probes.py", "test_mean_probes.py"):
        path = HERE / name
        result[str(path.relative_to(ROOT))] = digest(path)
    return result


def probe_plan(saved, seed, step):
    assert saved["seed"] == seed
    anchor = saved["anchors"][str(step)]
    result = {}
    for name in ("loss_train", "loss_aux", "utility_aux"):
        array = np.asarray(anchor[name])
        assert array.shape == (1024,) and array.dtype.kind in "iu"
        assert ((array >= 0) & (array < 5000)).all()
        result[name] = array
    return result


def verify_inputs():
    raw = gzip.decompress((I10 / "raw-results/completion.json.gz").read_bytes())
    assert (I10_ROOT / "completion.json").read_bytes() == raw
    collection = json.loads((I10 / "raw-results/collection.json").read_text())
    assert digest(I10_ROOT / "completion.json") == collection["source_completion_sha256"]
    completion = json.loads(raw)
    assert completion["status"] == "complete" and completion["mode"] == "full"
    index = {r["name"]: r for r in completion["artifacts"]}
    assert len(index) == len(completion["artifacts"])
    hashes = {"completion.json": digest(I10_ROOT / "completion.json")}

    def bind(record):
        assert index[record["name"]] == record
        previous.check_record(I10_ROOT, record)
        hashes[record["name"]] = record["sha256"]

    for name in ("manifest.json", "branches.json"):
        bind(index[name])
    manifest = json.loads((I10_ROOT / "manifest.json").read_text())
    assert manifest["source_hashes"] == previous.previous.source_manifest()
    old_index = json.loads((I10_ROOT / "branches.json").read_text())
    assert (old_index["unique_branches"], old_index["logical_cells"]) == (189, 216)
    entries = [r for r in old_index["entries"] if r["physical_source"] == "current32"
               and r["parent_step"] in ANCHORS and r["objective"] in OBJECTIVES
               and r["policy"] in REFERENCES]
    assert len(entries) == 36
    baselines = {}
    for entry in entries:
        key = (entry["seed"], entry["parent_step"], entry["objective"], entry["policy"])
        assert key not in baselines and key in expected_keys(REFERENCES)
        assert entry["logical_sources"] == ["current32"] and len(entry["parent_artifacts"]) == 1
        assert entry["completed_horizon"] == 500
        bind(entry["branch_artifact"])
        bind(entry["final_artifact"])
        branch = json.loads((I10_ROOT / entry["branch_artifact"]["name"]).read_text())
        for field in ("id", "seed", "parent_step", "physical_source", "logical_sources",
                      "objective", "policy", "parent_artifacts", "parent_complete_digest",
                      "plan_file", "completed_horizon", "final_artifact"):
            assert branch[field] == entry[field]
        assert [r["horizon"] for r in branch["curve"]] == list(HORIZONS)
        assert len(branch["steps"]) == 500
        baselines[key] = {"entry": entry, "branch": branch}
    assert set(baselines) == expected_keys(REFERENCES)
    plans, probes = {}, {}
    parent_index, parent_hashes, data_hashes = previous.previous.verify_inputs()
    for seed in SEEDS:
        saved = json.loads((I9_ROOT / f"plan-s{seed}.json").read_text())
        for step in ANCHORS:
            name = f"plan-s{seed}-t{step}.json"
            bind(index[name])
            plans[(seed, step)] = previous.validate_plan(
                json.loads((I10_ROOT / name).read_text()), seed, step)
            probes[(seed, step)] = probe_plan(saved, seed, step)
    return baselines, plans, probes, parent_index, {
        "i10_directory": str(I10_ROOT), "i10_hashes": hashes,
        "i9_directory": str(I9_ROOT), "i9_hashes": parent_hashes,
        "data_hashes": data_hashes, "baseline_count": 36, "parent_count": 6,
        "baseline_source_hashes": manifest["source_hashes"]}


def save_probe(run, name, metadata, record, tensors):
    tensor_digest = i9.tree_digest(tensors)
    saved = run.save(name + ".pt", {**metadata, "record": record,
        "tensors": tensors, "tensor_digest": tensor_digest}, tensor=True)
    return {**metadata, "record": record, "tensor_digest": tensor_digest, "tensor_artifact": saved}


def validate_branch_probes(result):
    horizons = [p["horizon"] for p in result["probes"]]
    last = result["completed_steps"] if result["status"] == "complete" else result["last_evaluated_horizon"]
    expected = [h for h in HORIZONS if h <= last]
    if result["status"] == "numerical_failure" and result["numerical_failure"]["phase"] == "probe":
        expected = expected[:-1]
    assert horizons == expected
    for p in result["probes"]:
        assert p["record"]["horizon"] == p["horizon"]
        assert type(p["record"]["tensor_artifact"]["name"]) is str
    if result["status"] == "complete":
        assert horizons == list(HORIZONS)
    fingerprints = (result["first_step_raw_gradient_digest"], result["first_step_observer_digest"])
    assert all(v is None or (type(v) is str and len(v) == 64) for v in fingerprints)
    if result["completed_steps"] > 0:
        assert all(v is not None for v in fingerprints)
    return fingerprints


def smoke(run):
    gen = torch.Generator().manual_seed(13013)
    x = torch.rand((128, 784), generator=gen).cuda()
    y = torch.randint(0, 10, (128,), generator=gen).cuda()
    noisy = torch.randint(0, 10, (128,), generator=gen).cuda()
    data = {"x": x, "clean": y, "noisy": noisy, "vx": x, "vy": y,
            "ax": x.flip(0), "ay": y.flip(0)}
    model = i9.make_model(13013, "cuda")
    opt = i9.make_optimizer(model)
    tracker = i9.make_tracker(model, opt)
    for _ in range(100):
        run.check()
        i9.train_step(model, opt, tracker, x[:64], noisy[:64], "current32")
    state = i9.snapshot(model, opt, tracker)
    saved = run.save("synthetic-parent.pt", state, tensor=True)
    state = torch.load(run.path / saved["name"], weights_only=True, map_location="cpu")
    plan = previous.previous.branch_plan(13013, 100)
    plan["batches"] %= 128
    short = {k: v[:10] if isinstance(v, np.ndarray) else v for k, v in plan.items()}
    rng = np.random.default_rng(13013)
    pp = {k: rng.integers(0, 128, 1024) for k in ("loss_train", "loss_aux", "utility_aux")}
    checks = []
    for objective in OBJECTIVES:
        probe_record, tensors = mean_probes.paired_step_probe(state, data, plan, pp,
                                                             objective, check=run.check)
        save_probe(run, f"paired-{objective}", {"objective": objective}, probe_record, tensors)
        for policy in REFERENCES:
            curve, _, final = previous.previous.execute_branch(
                run, state, data, plan, objective, policy, 10, (0, 1, 10))
            reference = mean_core.run_branch(state, data, short, objective, policy,
                steps=10, horizons=(0, 1, 10), check=run.check)
            assert reference["status"] == "complete" and reference["curve"] == curve
            assert i9.equal_tree(reference["terminal_state"], final)
        expected = {k: v for k, v in curve[0].items() if k != "horizon"}
        first = None
        for policy in POLICIES:
            def probe(snapshot, horizon):
                record, tensors = mean_probes.alignment_probe(snapshot, data, pp, check=run.check)
                return save_probe(run, f"alignment-{objective}-{policy}-h{horizon}",
                    {"horizon": horizon}, record, tensors)
            result = mean_core.run_branch(state, data, short, objective, policy,
                steps=10, horizons=(0, 1, 10), check=run.check, probe=probe,
                expected_initial_evaluation=expected)
            assert result["status"] == "complete" and len(result["probes"]) == 3
            fingerprints = (result["first_step_raw_gradient_digest"], result["first_step_observer_digest"])
            if first is not None:
                assert first == fingerprints
            first = fingerprints
            record = run.save(f"smoke-{objective}-{policy}.pt", result["terminal_state"], tensor=True)
            restored = torch.load(run.path / record["name"], weights_only=True, map_location="cpu")
            assert i9.equal_tree(restored, result["terminal_state"])
            checks.append({"objective": objective, "policy": policy,
                "initial_evaluation_exact": True, "state_roundtrip_exact": True,
                "probe_count": len(result["probes"]), "first_fingerprints": fingerprints})
    run.save("smoke-checks.json", {"status": "pass", "synthetic_only": True,
        "checks": checks, "six_inherited_paths_equal_i10": True,
        "source_hashes": source_manifest(), "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name()})
    return {"attempted_branches": 6, "numerical_failures": 0, "science_complete": None}


def full(run, smoke_dir):
    smoke_dir = smoke_dir.resolve(strict=True)
    assert smoke_dir.parent == run.path.parent.resolve()
    completion = json.loads((smoke_dir / "completion.json").read_text())
    assert completion["status"] == "complete" and completion["mode"] == "smoke"
    record = next(r for r in completion["artifacts"] if r["name"] == "smoke-checks.json")
    previous.check_record(smoke_dir, record)
    checks = json.loads((smoke_dir / record["name"]).read_text())
    assert checks["status"] == "pass" and checks["synthetic_only"]
    assert checks["source_hashes"] == source_manifest()
    assert checks["torch"] == torch.__version__ and checks["gpu"] == torch.cuda.get_device_name()
    baselines, plans, probes, parent_index, inputs = verify_inputs()
    run.save("parent-inputs.json", inputs)
    run.save("baseline-references.json", {"directory": str(I10_ROOT),
        "entries": [baselines[k]["entry"] for k in sorted(baselines)],
        "count": 36, "new_baseline_training_updates": 0})
    for (seed, step), plan in plans.items():
        name = f"plan-s{seed}-t{step}.json"
        assert run.save(name, plan)["sha256"] == inputs["i10_hashes"][name]
        run.save(f"probe-plan-s{seed}-t{step}.json", {"seed": seed, "parent_step": step,
            "source_plan": f"plan-s{seed}.json", "source_sha256": inputs["i9_hashes"][f"plan-s{seed}.json"],
            "indices": probes[(seed, step)]})
    x, y = parent_run.read_training()
    entries, paired, baseline_alignments, common_alignments = [], [], [], []
    for seed in SEEDS:
        data = previous.previous.make_data(x, y, seed)
        for step in ANCHORS:
            binding, parent_record = previous.previous.load_parent(seed, "current32", step, parent_index)
            state, state_digest = binding["state"], binding["complete_state_digest"]
            pp, plan = probes[(seed, step)], plans[(seed, step)]
            model, opt, tracker = i9.restore(state, "cuda")
            preflight = i9.snapshot(model, opt, tracker)
            expected = i10.evaluate(model, data)
            assert i9.equal_tree(preflight, i9.snapshot(model, opt, tracker))
            assert i9.tree_digest(preflight) == state_digest
            del model, opt, tracker, preflight
            for objective in OBJECTIVES:
                for policy in REFERENCES:
                    baseline = baselines[(seed, step, objective, policy)]
                    assert baseline["entry"]["parent_artifacts"] == [parent_record]
                    assert baseline["entry"]["parent_complete_digest"] == state_digest
                    assert {k: v for k, v in baseline["branch"]["curve"][0].items() if k != "horizon"} == expected
                    final_record = baseline["entry"]["final_artifact"]
                    final = torch.load(I10_ROOT / final_record["name"], weights_only=True, map_location="cpu")
                    assert i9.tree_digest(final["state"]) == final["complete_state_digest"] == baseline["branch"]["final_complete_digest"]
                    rec, payload = mean_probes.alignment_probe(final["state"], data, pp, check=run.check)
                    baseline_alignments.append(save_probe(run, f"alignment-baseline-{baseline['entry']['id']}",
                        {"seed": seed, "parent_step": step, "objective": objective, "policy": policy,
                         "horizon": 500, "state_digest": final["complete_state_digest"],
                         "baseline_final_artifact": final_record}, rec, payload))
                    del final, payload
            rec, payload = mean_probes.alignment_probe(state, data, pp, check=run.check)
            common = save_probe(run, f"alignment-common-s{seed}-t{step}-h0",
                {"seed": seed, "parent_step": step, "horizon": 0, "state_digest": state_digest}, rec, payload)
            common_alignments.append(common)
            del payload
            for objective in OBJECTIVES:
                rec, payload = mean_probes.paired_step_probe(state, data, plan, pp, objective, check=run.check)
                paired_record = save_probe(run, f"paired-s{seed}-t{step}-{objective}",
                    {"seed": seed, "parent_step": step, "objective": objective, "state_digest": state_digest}, rec, payload)
                paired.append(paired_record)
                del payload
                first = None
                for policy in POLICIES:
                    identity = f"s{seed}-current32-t{step}-{objective}-{policy}"

                    def probe(snapshot, horizon):
                        if horizon == 0:
                            assert i9.tree_digest(snapshot) == state_digest
                            return common
                        rec, payload = mean_probes.alignment_probe(snapshot, data, pp, check=run.check)
                        return save_probe(run, f"alignment-{identity}-h{horizon}",
                            {"id": identity, "horizon": horizon, "state_digest": i9.tree_digest(snapshot)}, rec, payload)

                    result = mean_core.run_branch(state, data, plan, objective, policy,
                        steps=500, horizons=HORIZONS, check=run.check, probe=probe,
                        expected_initial_evaluation=expected)
                    fingerprints = validate_branch_probes(result)
                    if all(v is not None for v in fingerprints):
                        if first is not None:
                            assert first == fingerprints
                        first = fingerprints
                    assert i9.tree_digest(state) == state_digest
                    metadata = {"id": identity, "seed": seed, "parent_step": step,
                        "source": "current32", "objective": objective, "policy": policy,
                        "parent_artifact": parent_record, "parent_complete_digest": state_digest,
                        "plan_file": f"plan-s{seed}-t{step}.json", "initial_evaluation_exact": True,
                        "baseline_branches": {p: baselines[(seed, step, objective, p)]["entry"]["branch_artifact"] for p in REFERENCES},
                        "paired_step_probe": paired_record,
                        "alignment_probes": result["probes"],
                        "first_step_raw_gradient_digest": fingerprints[0],
                        "first_step_observer_digest": fingerprints[1]}
                    entries.append(previous.seal_branch(run, metadata, result))
                    print(json.dumps({"branch": identity, "status": result["status"], "finished": len(entries),
                        "expected": 36, "elapsed_seconds": time.monotonic() - run.started}), flush=True)
                    del result
                    torch.cuda.empty_cache()
            del binding, state
        del data
    assert len(entries) == 36 and len(paired) == 18
    assert len(common_alignments) == 6 and len(baseline_alignments) == 36
    assert {(r["seed"], r["parent_step"], r["objective"], r["policy"]) for r in entries} == expected_keys()
    for root, hashes in ((I10_ROOT, inputs["i10_hashes"]), (I9_ROOT, inputs["i9_hashes"]),
                         (parent_run.DATA, inputs["data_hashes"])):
        for name, expected_hash in hashes.items():
            assert digest(root / name) == expected_hash
    failures = sum(r["status"] == "numerical_failure" for r in entries)
    physical_alignments = {p["tensor_artifact"]["name"] for p in common_alignments + baseline_alignments}
    for entry in entries:
        physical_alignments.update(p["record"]["tensor_artifact"]["name"] for p in entry["alignment_probes"])
    if failures == 0:
        assert len(physical_alignments) == 258
    result = {"attempted_branches": 36, "numerical_failures": failures, "science_complete": failures == 0}
    run.save("branches.json", {**result, "entries": entries, "baseline_count": 36,
        "physical_alignment_records": len(physical_alignments),
        "paired_step_probes": paired, "common_h0_alignments": common_alignments,
        "baseline_h500_alignments": baseline_alignments, "parent_hashes_unchanged": True})
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
            "started_unix": time.time(), "pid": os.getpid(), "systemd_invocation": os.environ.get("INVOCATION_ID"),
            "source_hashes": source_manifest(),
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "torch": torch.__version__, "gpu": torch.cuda.get_device_name(), "cloud_spend_usd": 0,
            "planned_branches": 36, "baseline_count": 36, "parent_count": 6,
            "paired_step_groups": 18, "diagnostic_adam_calls": 54,
            "physical_alignment_records_if_complete": 258,
            "policies": POLICIES, "objectives": OBJECTIVES, "horizons": HORIZONS}
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
