#!/usr/bin/env python3
"""Generate replay data for the train-vs-test memorization viz.

Trains the REAL WeightCovarianceFilterV2 (and an Adam baseline) on noisy
two-moons problems and records, at sampled epochs, the decision-boundary grid
(P(class 1)) for each optimizer plus train/test accuracy. Produces multiple
selectable environments (different noise models). Faithful by construction —
no JS reimplementation.
"""
import sys, os, json
import numpy as np
import torch
import torch.nn as nn
sys.path.insert(0, os.path.dirname(__file__))
from weight_cov_optimizer_v2 import WeightCovarianceFilterV2
from verify_moons_memorization import make_moons, flip_labels, acc

# --- config (the verified dramatic regime) ---
N_TRAIN, N_TEST = 240, 1000
FEAT_NOISE, NOISE_FRAC = 0.08, 0.40
H, EPOCHS, LR, BS = 256, 3000, 0.01, 64
RANK, DECAY, WARMUP = 24, 0.95, 50
SAMPLE_EVERY = 50
GRID = 56
OUT = os.path.join(os.path.dirname(__file__), "..", "results", "moons_viz_data.json")


class MLP(nn.Module):
    def __init__(s, h=H):
        super().__init__()
        s.net = nn.Sequential(nn.Linear(2, h), nn.ReLU(),
                              nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 2))

    def forward(s, x):
        return s.net(x)


def env_flip():
    """40% of training labels flipped; clean test."""
    Xtr, yclean = make_moons(N_TRAIN, FEAT_NOISE, seed=0)
    ytr, flipped = flip_labels(yclean, NOISE_FRAC, seed=1)
    noise = np.zeros(len(ytr), bool); noise[flipped] = True
    Xte, yte = make_moons(N_TEST, FEAT_NOISE, seed=2)
    return Xtr.astype(np.float32), ytr, noise, Xte.astype(np.float32), yte


def env_random():
    """Clean moons + random outlier points (uniform in-region, random labels)."""
    Xm, ym = make_moons(N_TRAIN, FEAT_NOISE, seed=0)
    n_out = 200  # denser than the flip count: isolated outliers need volume to bite test acc
    rng = np.random.RandomState(7)
    lo, hi = Xm.min(0) - 0.2, Xm.max(0) + 0.2
    Xo = rng.uniform(lo, hi, size=(n_out, 2))
    yo = rng.randint(0, 2, size=n_out)
    Xtr = np.concatenate([Xm, Xo], 0).astype(np.float32)
    ytr = np.concatenate([ym, yo]).astype(np.int64)
    noise = np.zeros(len(ytr), bool); noise[len(ym):] = True  # the random points
    Xte, yte = make_moons(N_TEST, FEAT_NOISE, seed=2)
    return Xtr, ytr, noise, Xte.astype(np.float32), yte


ENVS = [
    ("flip", "40% of train labels flipped", env_flip),
    ("random", "Random outlier points added (random labels)", env_random),
]


def build_grid(Xtr, Xte):
    allp = np.concatenate([Xtr, Xte], 0)
    lo, hi = allp.min(0) - 0.4, allp.max(0) + 0.4
    gx = np.linspace(lo[0], hi[0], GRID)
    gy = np.linspace(lo[1], hi[1], GRID)
    GX, GY = np.meshgrid(gx, gy)
    pts = torch.tensor(np.stack([GX.ravel(), GY.ravel()], 1).astype(np.float32))
    meta = {"n": GRID, "xmin": float(lo[0]), "xmax": float(hi[0]),
            "ymin": float(lo[1]), "ymax": float(hi[1])}
    return pts, meta


def run(mode, Xtr, ytr, Xte, yte, grid_pts):
    torch.manual_seed(0)
    model = MLP()
    base = torch.optim.Adam(model.parameters(), lr=LR)
    opt = (WeightCovarianceFilterV2(model, base, rank=RANK, decay=DECAY, warmup=WARMUP)
           if mode == "filter" else None)
    n = len(Xtr)

    def grid_prob():
        with torch.no_grad():
            p = torch.softmax(model(grid_pts), 1)[:, 1]
        return [round(float(v), 3) for v in p.tolist()]

    frames = []
    for ep in range(EPOCHS):
        perm = torch.randperm(n)
        for i in range(0, n, BS):
            idx = perm[i:i + BS]; xb, yb = Xtr[idx], ytr[idx]
            if mode == "filter":
                opt.step(xb, yb)
            else:
                base.zero_grad()
                loss = nn.functional.cross_entropy(model(xb), yb)
                loss.backward(); base.step()
        if ep % SAMPLE_EVERY == 0 or ep == EPOCHS - 1:
            frames.append({"epoch": ep,
                           "train_acc": round(acc(model, Xtr, ytr), 4),
                           "test_acc": round(acc(model, Xte, yte), 4),
                           "grid": grid_prob()})
    return frames


def main():
    out = {"environments": {}, "order": [k for k, _, _ in ENVS],
           "config": {"n_train": N_TRAIN, "n_test": N_TEST, "feat_noise": FEAT_NOISE,
                      "noise_frac": NOISE_FRAC, "hidden": H, "epochs": EPOCHS,
                      "lr": LR, "rank": RANK, "decay": DECAY, "warmup": WARMUP}}
    for key, label, fn in ENVS:
        print(f"=== env: {key} ===")
        Xtr_np, ytr_np, noise, Xte_np, yte_np = fn()
        Xtr, ytr = torch.tensor(Xtr_np), torch.tensor(ytr_np)
        Xte, yte = torch.tensor(Xte_np), torch.tensor(yte_np)
        grid_pts, gmeta = build_grid(Xtr_np, Xte_np)
        adam = run("adam", Xtr, ytr, Xte, yte, grid_pts)
        filt = run("filter", Xtr, ytr, Xte, yte, grid_pts)
        print(f"  adam   final train={adam[-1]['train_acc']:.3f} test={adam[-1]['test_acc']:.3f}")
        print(f"  filter final train={filt[-1]['train_acc']:.3f} test={filt[-1]['test_acc']:.3f}")
        out["environments"][key] = {
            "label": label, "grid": gmeta,
            "train_pts": [{"x": float(Xtr_np[i, 0]), "y": float(Xtr_np[i, 1]),
                           "label": int(ytr_np[i]), "noise": bool(noise[i])}
                          for i in range(len(ytr_np))],
            "test_pts": [{"x": float(Xte_np[i, 0]), "y": float(Xte_np[i, 1]),
                          "label": int(yte_np[i])} for i in range(len(yte_np))],
            "adam": adam, "filter": filt,
        }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f)
    print(f"saved {OUT} ({os.path.getsize(OUT)/1e6:.2f} MB)")


if __name__ == "__main__":
    main()
