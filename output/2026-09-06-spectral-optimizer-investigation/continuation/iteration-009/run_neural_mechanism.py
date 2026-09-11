#!/usr/bin/env python3
"""I9 bounded neural acquisition; exclusive artifacts, no test set or restarts."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import struct
import subprocess
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
import neural_core as core

DATA = Path("data/MNIST/raw")
STEPS, SEEDS, ANCHORS = 2000, [100, 101, 102], [100, 500, 1500, 2000]
SOURCES = ["raw", "current32"]
MAX_ARTIFACT_BYTES = 512 * 1024**2
CANONICAL_SHA = "9280c7d360aef4c775baf0d781a5afd56e016e9cfb93ef829c3ae4b8cc3df943"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def rng(seed, stream):
    return np.random.default_rng(np.random.SeedSequence([20260907, 9, seed, stream]))


def make_plan(seed):
    order = rng(seed, 0).permutation(60000)
    result = {
        "seed": seed, "initialization_seed": int(rng(seed, 1).integers(0, 2**31)),
        "train_indices": order[:5000], "validation_indices": order[5000:10000],
        "auxiliary_indices": order[10000:15000],
        "replacement_mask": rng(seed, 2).random(5000) < .9,
        "replacement_digits": rng(seed, 3).integers(0, 10, 5000),
        "training_batches": rng(seed, 4).integers(0, 5000, (STEPS, 64)),
        "anchors": {},
    }
    for step in ANCHORS:
        result["anchors"][str(step)] = {
            "pairs": rng(seed, step * 10 + 1).integers(0, 5000, (32, 2, 64)),
            "update": rng(seed, step * 10 + 2).integers(0, 5000, 64),
            "loss_train": rng(seed, step * 10 + 3).integers(0, 5000, 1024),
            "loss_aux": rng(seed, step * 10 + 4).integers(0, 5000, 1024),
            "utility_aux": rng(seed, step * 10 + 5).integers(0, 5000, 1024),
        }
    return result


def json_tree(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): json_tree(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_tree(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def read_training():
    # No path to official test files exists in this harness.
    raw = (DATA / "train-images-idx3-ubyte").read_bytes()
    magic, count, rows, cols = struct.unpack(">IIII", raw[:16])
    assert (magic, count, rows, cols) == (2051, 60000, 28, 28)
    assert len(raw) == 16 + count * 784
    x = torch.from_numpy(np.frombuffer(raw, np.uint8, offset=16).copy().reshape(count, 784))
    raw_y = (DATA / "train-labels-idx1-ubyte").read_bytes()
    assert struct.unpack(">II", raw_y[:8]) == (2049, 60000)
    assert len(raw_y) == 60008
    y = torch.from_numpy(np.frombuffer(raw_y, np.uint8, offset=8).copy()).long()
    assert bool(((y >= 0) & (y < 10)).all())
    return x, y


def data_for_plan(x, y, plan):
    ti, vi, ai = (torch.from_numpy(plan[k]) for k in
                  ["train_indices", "validation_indices", "auxiliary_indices"])
    clean = y[ti]
    noisy = torch.where(torch.from_numpy(plan["replacement_mask"]),
                        torch.from_numpy(plan["replacement_digits"]), clean)
    data = {"x": x[ti].float().div(255).cuda(), "clean": clean.cuda(), "noisy": noisy.cuda(),
            "vx": x[vi].float().div(255).cuda(), "vy": y[vi].cuda(),
            "ax": x[ai].float().div(255).cuda(), "ay": y[ai].cuda()}
    return data, {"replaced_count": int(plan["replacement_mask"].sum()),
                  "incorrect_count": int((noisy != clean).sum()), "train_count": 5000}


@torch.no_grad()
def evaluate(model, data):
    train_logits = model(data["x"]).double()
    val_logits = model(data["vx"]).double()
    return {
        "train_clean_ce": F.cross_entropy(train_logits, data["clean"]).item(),
        "train_fixed_ce": F.cross_entropy(train_logits, data["noisy"]).item(),
        "train_clean_accuracy": (train_logits.argmax(1) == data["clean"]).double().mean().item(),
        "train_fixed_accuracy": (train_logits.argmax(1) == data["noisy"]).double().mean().item(),
        "validation_ce": F.cross_entropy(val_logits, data["vy"]).item(),
        "validation_accuracy": (val_logits.argmax(1) == data["vy"]).double().mean().item(),
    }


class BudgetWriter:
    """Cap a new file while torch.save writes it, preserving a partial on failure."""
    def __init__(self, handle, remaining):
        self.handle, self.remaining = handle, remaining

    def write(self, value):
        if len(value) > self.remaining:
            raise RuntimeError("I9 artifact budget exhausted")
        count = self.handle.write(value)
        self.remaining -= count
        return count

    def flush(self):
        return self.handle.flush()


class Run:
    def __init__(self, path, limit_seconds):
        self.path = path
        self.started = time.monotonic()
        self.deadline = self.started + limit_seconds
        self.artifacts = []

    def check(self):
        if time.monotonic() >= self.deadline:
            raise RuntimeError("I9 cooperative total wall deadline exceeded")
        if torch.cuda.max_memory_allocated() > 4 * 1024**3:
            raise RuntimeError("I9 PyTorch GPU allocation budget exceeded")
        if shutil.disk_usage(self.path).free < 1024**3:
            raise RuntimeError("I9 output mount has less than1GiB free")

    def used(self):
        return sum(p.stat().st_size for p in self.path.rglob("*") if p.is_file())

    def save(self, name, value, tensor=False):
        self.check()
        target = self.path / name
        # Reserve1MiB for final/failure metadata; no overwriting existing outputs.
        allowance = MAX_ARTIFACT_BYTES - self.used() - 1024**2
        if tensor:
            with target.open("xb") as f:
                torch.save(value, BudgetWriter(f, allowance))
        else:
            payload = (json.dumps(json_tree(value), indent=2, allow_nan=False) + "\n").encode()
            if len(payload) > allowance:
                raise RuntimeError("I9 JSON artifact budget exceeded")
            with target.open("xb") as f:
                f.write(payload)
        record = {"name": name, "bytes": target.stat().st_size, "sha256": digest(target)}
        self.artifacts.append(record)
        self.check()
        return record


def configure():
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert torch.cuda.is_available()
    meminfo = {row.split(":", 1)[0]: int(row.split()[1]) * 1024
               for row in Path("/proc/meminfo").read_text().splitlines()}
    if meminfo["MemAvailable"] < 16 * 1024**3:
        raise RuntimeError("Less than16GiB available host RAM")
    occupancy = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
        "--format=csv,noheader"], text=True)
    for row in occupancy.splitlines():
        pid, name, _ = [field.strip() for field in row.split(",", 2)]
        if int(pid) != os.getpid() and name not in {
                "/usr/libexec/gnome-remote-desktop-daemon", "stremio"}:
            raise RuntimeError("Foreign GPU compute process at launch; leaving it untouched")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.init()
    torch.manual_seed(2026090709)
    torch.cuda.manual_seed_all(2026090709)
    np.random.seed(2026090709)
    random.seed(2026090709)
    free, total = torch.cuda.mem_get_info()
    if free < 8 * 1024**3:
        raise RuntimeError("Less than8GiB free GPU at launch")
    torch.cuda.set_per_process_memory_fraction(4 * 1024**3 / total)


def source_manifest():
    paths = [ROOT / "spectral_filter.py", HERE / "neural_core.py", Path(__file__).resolve(),
             HERE / "test_neural_core.py", HERE / "test_runner.py", HERE / "protocol.md"]
    assert digest(paths[0]) == CANONICAL_SHA
    return {str(p.relative_to(ROOT)): digest(p) for p in paths}


def smoke(run):
    """Synthetic target-device round-trip/neutrality, no MNIST acquisition."""
    gen = torch.Generator().manual_seed(19009)
    x = torch.rand((128, 784), generator=gen).cuda()
    clean = torch.randint(0, 10, (128,), generator=gen).cuda()
    noisy = torch.randint(0, 10, (128,), generator=gen).cuda()
    data = {"x": x, "clean": clean, "noisy": noisy, "vx": x, "vy": clean,
            "ax": x.flip(0), "ay": clean.flip(0)}
    model = core.make_model(19009, "cuda")
    opt = core.make_optimizer(model)
    observer = core.make_tracker(model, opt)
    for step in range(100):
        run.check()
        index = torch.arange(step % 64, step % 64 + 64, device="cuda")
        core.train_step(model, opt, observer, x[index], noisy[index], "current32")
    before = core.snapshot(model, opt, observer)
    run.save("smoke-anchor.pt", before, tensor=True)
    loaded = torch.load(run.path / "smoke-anchor.pt", weights_only=True, map_location="cpu")
    assert core.equal_tree(before, loaded), "smoke serialization mismatch"
    rgen = np.random.default_rng(19009)
    plans = {"pairs": rgen.integers(0, 128, (32, 2, 64)),
             "update": rgen.integers(0, 128, 64),
             "loss_train": rgen.integers(0, 128, 1024),
             "loss_aux": rgen.integers(0, 128, 1024),
             "utility_aux": rgen.integers(0, 128, 1024)}
    record, _ = core.probe_anchor(model, opt, observer, data, plans)
    assert core.equal_tree(before, core.snapshot(model, opt, observer)), "smoke probe mutation"
    restored = core.restore(loaded, "cuda")
    i = torch.as_tensor(plans["update"], device="cuda")
    core.train_step(model, opt, observer, x[i], noisy[i], "current32")
    core.train_step(*restored, x[i], noisy[i], "current32")
    assert core.equal_tree(core.snapshot(model, opt, observer), core.snapshot(*restored)), "smoke next-step mismatch"
    run.save("smoke-checks.json", {"status": "pass", "source_hashes": source_manifest(),
             "snapshot_roundtrip": True, "probe_neutral": True, "next_step_equal": True,
             "probe_record_keys": sorted(record), "synthetic_only": True,
             "torch": torch.__version__, "gpu": torch.cuda.get_device_name()})


def full(run, smoke_dir):
    smoke_check = json.loads((smoke_dir / "smoke-checks.json").read_text())
    assert smoke_check["status"] == "pass" and smoke_check["synthetic_only"]
    assert smoke_check["source_hashes"] == source_manifest(), "source changed after GPU smoke"
    assert smoke_check["torch"] == torch.__version__
    assert smoke_check["gpu"] == torch.cuda.get_device_name()
    plans = [make_plan(seed) for seed in SEEDS]
    for plan in plans:
        run.save(f"plan-s{plan['seed']}.json", plan)
    x, y = read_training()
    all_curves = []
    for plan in plans:
        data, corruption = data_for_plan(x, y, plan)
        run.save(f"corruption-s{plan['seed']}.json", corruption)
        warmup_digest = None
        for source in SOURCES:
            model = core.make_model(plan["initialization_seed"], "cuda")
            assert sum(p.numel() for p in model.parameters()) == 50890
            opt = core.make_optimizer(model)
            observer = core.make_tracker(model, opt)
            trajectory_id = f"s{plan['seed']}-{source}"
            curve = [{"step": 0, **evaluate(model, data)}]
            for step, indices in enumerate(plan["training_batches"], start=1):
                run.check()
                i = torch.as_tensor(indices, device="cuda")
                core.train_step(model, opt, observer, data["x"][i], data["noisy"][i], source)
                if step % 100 == 0:
                    curve.append({"step": step, **evaluate(model, data)})
                    print(json.dumps({"trajectory": trajectory_id, "step": step,
                                      "elapsed_seconds": time.monotonic() - run.started}), flush=True)
                if step in ANCHORS:
                    anchor_id = f"{trajectory_id}-t{step}"
                    anchor = core.snapshot(model, opt, observer)
                    fingerprint = core.tree_digest(anchor)
                    if step == 100:
                        if source == "raw":
                            warmup_digest = fingerprint
                        else:
                            assert fingerprint == warmup_digest, "raw/current warmup state mismatch"
                    binding = {"seed": plan["seed"], "source": source, "completed_step": step,
                               "state": anchor, "complete_state_digest": fingerprint,
                               "plan_file": f"plan-s{plan['seed']}.json"}
                    anchor_file = run.save(f"anchor-{anchor_id}.pt", binding, tensor=True)
                    # Actual serialized anchor, not an in-memory clone, must round-trip.
                    reloaded = torch.load(run.path / anchor_file["name"], weights_only=True, map_location="cpu")
                    assert core.equal_tree(anchor, reloaded["state"])
                    del reloaded
                    record, tensors = core.probe_anchor(model, opt, observer, data,
                                                        plan["anchors"][str(step)])
                    assert fingerprint == core.tree_digest(core.snapshot(model, opt, observer))
                    tensor_file = run.save(f"probe-tensors-{anchor_id}.pt", tensors, tensor=True)
                    run.save(f"probe-{anchor_id}.json", {
                        "seed": plan["seed"], "source": source, "completed_step": step,
                        "anchor_artifact": anchor_file, "tensor_artifact": tensor_file,
                        "complete_state_digest": fingerprint, "measurement": record})
                    del anchor, binding, tensors
                run.check()
            record = {"seed": plan["seed"], "source": source, "curve": curve}
            run.save(f"curve-{trajectory_id}.json", record)
            all_curves.append(record)
            del model, opt, observer
            torch.cuda.empty_cache()
        del data
    run.save("all-curves.json", all_curves)


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--full", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke-dir", type=Path)
    args = parser.parse_args()
    parent = args.output.parent.resolve(strict=True)
    if not parent.is_relative_to(Path("/tmp/spectral-experiment-artifacts")):
        raise SystemExit("I9 output must be on the verified large temporary volume")
    mount = subprocess.check_output(["findmnt", "-n", "-o", "TARGET,SOURCE", "--target", str(parent)], text=True).split()
    assert mount == ["/private-artifacts/storage", "/dev/RECONFIGURE_FOR_LOCAL_STORAGE"]
    args.output.mkdir(exist_ok=False)
    run = Run(args.output, 100 if args.smoke else 1500)
    try:
        configure()
        manifest = {"status": "running", "mode": "smoke" if args.smoke else "full",
                    "started_unix": time.time(), "pid": os.getpid(),
                    "systemd_invocation": os.environ.get("INVOCATION_ID"),
                    "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                    "source_hashes": source_manifest(), "torch": torch.__version__,
                    "numpy": np.__version__, "python": sys.version,
                    "gpu": torch.cuda.get_device_name(), "seeds": SEEDS, "steps": STEPS,
                    "anchors": ANCHORS, "sources": SOURCES, "cloud_spend_usd": 0}
        if args.full:
            assert args.smoke_dir is not None
            manifest["data_hashes"] = {name: digest(DATA / name) for name in
                ["train-images-idx3-ubyte", "train-labels-idx1-ubyte"]}
            manifest["smoke_directory"] = str(args.smoke_dir)
        run.save("manifest.json", manifest)
        if args.smoke:
            smoke(run)
        else:
            full(run, args.smoke_dir)
        assert source_manifest() == manifest["source_hashes"], "sources changed during execution"
        torch.cuda.synchronize()
        complete = {"status": "complete", "mode": manifest["mode"],
                    "finished_unix": time.time(), "wall_seconds": time.monotonic() - run.started,
                    "artifact_bytes_before_completion": run.used(),
                    "peak_torch_gpu_bytes": torch.cuda.max_memory_allocated(),
                    "artifacts": list(run.artifacts)}
        run.save("completion.json", complete)
        print(json.dumps({k: v for k, v in complete.items() if k != "artifacts"}), flush=True)
    except Exception as exc:
        # Preserve exact scientific failure type/message; no secrets are used by this worker.
        failure = {"status": "failed", "type": type(exc).__name__, "message": str(exc),
                   "finished_unix": time.time(), "artifact_bytes": run.used()}
        with (run.path / "failure.json").open("x") as f:
            json.dump(failure, f, indent=2, allow_nan=False)
            f.write("\n")
        raise


if __name__ == "__main__":
    main()
