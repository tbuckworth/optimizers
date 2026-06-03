#!/usr/bin/env python3
"""Generate replay data for the train-vs-test memorization viz.

Trains the REAL WeightCovarianceFilterV2 (and an Adam baseline) on a noisy
two-moons problem and records, at sampled epochs, the decision-boundary grid
(P(class 1)) for each optimizer plus train/test accuracy. Dumps a single JSON
that the HTML viz replays. Faithful by construction — no JS reimplementation.
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
FEAT_NOISE, LABEL_NOISE = 0.08, 0.40
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


def main():
    Xtr_np, ytr_clean = make_moons(N_TRAIN, FEAT_NOISE, seed=0)
    ytr_np, flipped = flip_labels(ytr_clean, LABEL_NOISE, seed=1)
    Xte_np, yte_np = make_moons(N_TEST, FEAT_NOISE, seed=2)
    Xtr, ytr = torch.tensor(Xtr_np), torch.tensor(ytr_np)
    Xte, yte = torch.tensor(Xte_np), torch.tensor(yte_np)

    # grid extent covering both sets, with margin
    allp = np.concatenate([Xtr_np, Xte_np], 0)
    lo = allp.min(0) - 0.4
    hi = allp.max(0) + 0.4
    gx = np.linspace(lo[0], hi[0], GRID)
    gy = np.linspace(lo[1], hi[1], GRID)
    GX, GY = np.meshgrid(gx, gy)
    grid_pts = torch.tensor(np.stack([GX.ravel(), GY.ravel()], 1).astype(np.float32))

    def grid_prob(model):
        with torch.no_grad():
            p = torch.softmax(model(grid_pts), 1)[:, 1]  # P(class 1)
        return [round(float(v), 3) for v in p.tolist()]

    def run(mode):
        torch.manual_seed(0)
        model = MLP()
        base = torch.optim.Adam(model.parameters(), lr=LR)
        opt = (WeightCovarianceFilterV2(model, base, rank=RANK, decay=DECAY, warmup=WARMUP)
               if mode == "filter" else None)
        n = len(Xtr)
        frames = []
        for ep in range(EPOCHS):
            perm = torch.randperm(n)
            for i in range(0, n, BS):
                idx = perm[i:i + BS]
                xb, yb = Xtr[idx], ytr[idx]
                if mode == "filter":
                    opt.step(xb, yb)
                else:
                    base.zero_grad()
                    loss = nn.functional.cross_entropy(model(xb), yb)
                    loss.backward()
                    base.step()
            if ep % SAMPLE_EVERY == 0 or ep == EPOCHS - 1:
                frames.append({
                    "epoch": ep,
                    "train_acc": round(acc(model, Xtr, ytr), 4),
                    "test_acc": round(acc(model, Xte, yte), 4),
                    "grid": grid_prob(model),
                })
                print(f"  {mode} ep{ep:4d} train={frames[-1]['train_acc']:.3f} "
                      f"test={frames[-1]['test_acc']:.3f}")
        return frames

    print("training adam..."); adam = run("adam")
    print("training filter..."); filt = run("filter")

    data = {
        "config": {"n_train": N_TRAIN, "n_test": N_TEST, "feat_noise": FEAT_NOISE,
                   "label_noise": LABEL_NOISE, "hidden": H, "epochs": EPOCHS,
                   "lr": LR, "rank": RANK, "decay": DECAY, "warmup": WARMUP},
        "grid": {"n": GRID, "xmin": float(lo[0]), "xmax": float(hi[0]),
                 "ymin": float(lo[1]), "ymax": float(hi[1])},
        "train_pts": [{"x": float(Xtr_np[i, 0]), "y": float(Xtr_np[i, 1]),
                       "label": int(ytr_np[i]), "clean": int(ytr_clean[i]),
                       "flipped": bool(i in set(flipped.tolist()))}
                      for i in range(N_TRAIN)],
        "test_pts": [{"x": float(Xte_np[i, 0]), "y": float(Xte_np[i, 1]),
                      "label": int(yte_np[i])} for i in range(N_TEST)],
        "adam": adam, "filter": filt,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(data, f)
    sz = os.path.getsize(OUT) / 1e6
    print(f"saved {OUT} ({sz:.2f} MB, {len(adam)} frames)")


if __name__ == "__main__":
    main()
