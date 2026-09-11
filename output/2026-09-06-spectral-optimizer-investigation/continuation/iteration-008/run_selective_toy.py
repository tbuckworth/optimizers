#!/usr/bin/env python3
"""Prospective I8 fixed-corruption toy; exclusive outputs, CPU only."""
import argparse
import hashlib
import importlib.util
import itertools
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
CANONICAL = ROOT / "spectral_filter.py"
EXPECTED = "9280c7d360aef4c775baf0d781a5afd56e016e9cfb93ef829c3ae4b8cc3df943"
assert hashlib.sha256(CANONICAL.read_bytes()).hexdigest() == EXPECTED
spec = importlib.util.spec_from_file_location("i8_canonical_filter", CANONICAL)
canonical = importlib.util.module_from_spec(spec)
spec.loader.exec_module(canonical)

STEPS, WARMUP, LR = 2000, 100, 0.002
SEEDS = list(range(8))
ANGLES = [0, 30]
DESIGNS = {
    "useful": [[.25, 1], [1.75, 1], [.25, 1], [1.75, 1]],
    "nuisance": [[1, .25], [1, 1.75], [1, .25], [1, 1.75]],
    "both": [[.25, .25], [.25, 1.75], [1.75, .25], [1.75, 1.75]],
    "none": [[1, 1]] * 4,
}
ARMS = ["raw", "live", "frozen", "oracle"]
SNAPSHOTS = {0, 50, 100, 101, 150, 250, 500, 1000, 2000}


def directions(angle):
    a = math.radians(angle)
    return torch.tensor([[math.cos(a), -math.sin(a)],
                         [math.sin(a), math.cos(a)]], dtype=torch.float64)


def batch_order(design, seed, steps=STEPS):
    assert steps % 4 == 0
    rng = np.random.default_rng(seed)
    types = np.array(DESIGNS[design], dtype=np.float64)
    return np.concatenate([types[rng.permutation(4)] for _ in range(steps // 4)])


def gradient(theta, basis, h):
    return basis @ (h * (basis.T @ theta - 1.0))


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.theta = torch.nn.Parameter(torch.zeros(2, dtype=torch.float64))


@torch.no_grad()
def run_one(optimizer, design, angle, seed, arm, steps=STEPS, warmup=WARMUP):
    basis = directions(angle)
    u = basis[:, :1]
    model = Model()
    common = dict(lr=LR, weight_decay=0, foreach=False, fused=False)
    if optimizer == "sgd":
        opt = torch.optim.SGD(model.parameters(), momentum=0, **common)
    else:
        assert optimizer == "adam"
        opt = torch.optim.Adam(model.parameters(), betas=(.9, .99), eps=1e-8,
                               amsgrad=False, **common)
    filt = None
    if arm in {"live", "frozen"}:
        filt = canonical.SpectralGradientFilter(
            model, opt, rank=1, decay=.99, warmup=warmup,
            stable_update=True, weighting="hard", normalize="none", adaptive="none")
    batches = batch_order(design, seed, steps)
    batches_tensor = torch.from_numpy(batches)
    risk_trace = np.empty(steps + 1, dtype=np.float64)
    risk_trace[0] = .5
    snapshots = [{"step": 0, "clean_risk": .5, "train_risk": 1.0,
                  "theta_uv": [0.0, 0.0], "basis_useful_alignment": None}]
    frozen = None
    step_energy, outside_energy = 0.0, 0.0
    for t, h in enumerate(batches_tensor, start=1):
        opt.zero_grad(set_to_none=True)
        before = model.theta.detach().clone()
        raw = gradient(before, basis, h)
        model.theta.grad = raw.clone()
        applied_basis = None
        observed_basis = None
        if arm == "live":
            filt.filter_grad()
            observed_basis = filt.V
            if t > warmup:
                applied_basis = filt.V
        elif arm == "frozen":
            if t <= warmup:
                filt.filter_grad()
                observed_basis = filt.V
                if t == warmup:
                    assert filt.V is not None and filt.V.shape == (2, 1)
                    frozen = filt.V.detach().clone()
            else:
                applied_basis = observed_basis = frozen
                model.theta.grad = frozen @ (frozen.T @ raw)
        elif arm == "oracle":
            observed_basis = u
            if t > warmup:
                applied_basis = u
                model.theta.grad = u @ (u.T @ raw)
        else:
            assert arm == "raw"
        filtered = model.theta.grad
        opt.step()
        delta = model.theta - before
        uv = basis.T @ model.theta
        clean = .5 * ((uv[0] - 1).square() + uv[1].square()).item()
        if not math.isfinite(clean):
            raise RuntimeError(f"nonfinite risk in {optimizer}/{design}/{angle}/{seed}/{arm}/{t}")
        risk_trace[t] = clean
        if applied_basis is not None:
            outside = delta - applied_basis @ (applied_basis.T @ delta)
            outside_energy += outside.square().sum().item()
            step_energy += delta.square().sum().item()
        if t in SNAPSHOTS or t == steps:
            snapshots.append({
                "step": t, "clean_risk": clean,
                "train_risk": .5 * (uv - 1).square().sum().item(),
                "theta_uv": uv.tolist(),
                "basis_useful_alignment": None if observed_basis is None else
                    (observed_basis.T @ u).square().sum().item(),
                "raw_gradient_uv": (basis.T @ raw).tolist(),
                "filtered_gradient_uv": (basis.T @ filtered).tolist(),
                "actual_step_uv": (basis.T @ delta).tolist(),
            })
    record = {
        "id": f"{optimizer}-{design}-a{angle}-s{seed}-{arm}",
        "optimizer": optimizer, "design": design, "angle_degrees": angle,
        "seed": seed, "arm": arm,
        "batch_sha256": hashlib.sha256(batches.tobytes()).hexdigest(),
        "endpoint_clean_risk": float(risk_trace[-1]),
        "minimum_clean_risk": float(risk_trace.min()),
        "minimum_step": int(risk_trace.argmin()),
        "post_warmup_minimum_clean_risk": float(risk_trace[warmup + 1:].min()),
        "post_warmup_minimum_step": int(risk_trace[warmup + 1:].argmin() + warmup + 1),
        "out_of_applied_subspace_step_energy_fraction":
            outside_energy / step_energy if step_energy > 0 else None,
        "post_warmup_projected_step_energy": step_energy,
        "post_warmup_outside_step_energy": outside_energy,
        "snapshots": snapshots,
    }
    return record, risk_trace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    # mkdir is exclusive: an interrupted/completed run cannot silently restart.
    args.output.mkdir(exist_ok=False)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    started = time.time()
    manifest = {
        "status": "running", "started_unix": started,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"],
                                               cwd=ROOT, text=True).strip(),
        "source_hashes": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in [CANONICAL, Path(__file__).resolve(), HERE / "protocol.md"]},
        "python": platform.python_version(), "torch": torch.__version__,
        "numpy": np.__version__, "device": "cpu", "dtype": "float64",
        "pid": os.getpid(), "systemd_invocation": os.environ.get("INVOCATION_ID"),
        "steps": STEPS, "warmup": WARMUP, "lr": LR, "seeds": SEEDS,
        "angles": ANGLES, "designs": DESIGNS, "arms": ARMS,
        "planned_trajectories": 512, "cloud_spend_usd": 0,
    }
    with (args.output / "manifest.json").open("x") as f:
        json.dump(manifest, f, indent=2, allow_nan=False)
        f.write("\n")
    traces = {}
    with (args.output / "trajectories.jsonl").open("x") as f:
        for index, (optimizer, design, angle, seed, arm) in enumerate(itertools.product(
                ["sgd", "adam"], DESIGNS, ANGLES, SEEDS, ARMS), start=1):
            record, trace = run_one(optimizer, design, angle, seed, arm)
            traces[record["id"]] = trace
            f.write(json.dumps(record, allow_nan=False) + "\n")
            f.flush()
            if index % 32 == 0:
                print(json.dumps({"completed": index, "total": 512,
                                  "elapsed_seconds": time.time() - started}), flush=True)
    with (args.output / "clean_risk_curves.npz").open("xb") as f:
        np.savez_compressed(f, **traces)
    complete = {
        "status": "complete", "trajectories": len(traces),
        "finished_unix": time.time(), "wall_seconds": time.time() - started,
        "artifact_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in sorted(args.output.iterdir()) if p.is_file()},
    }
    with (args.output / "completion.json").open("x") as f:
        json.dump(complete, f, indent=2, allow_nan=False)
        f.write("\n")
    print(json.dumps(complete), flush=True)


if __name__ == "__main__":
    main()
