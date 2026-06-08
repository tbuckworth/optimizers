#!/usr/bin/env python3
"""Verify the filter resists label-noise memorization on a 2D toy (two-moons).

If the effect shows here (filter holds test acc / smooth boundary while Adam
memorizes train noise and test acc drops), we build the train-vs-test
mechanism viz on top of it.
"""
import sys, os, json, argparse
import numpy as np
import torch
import torch.nn as nn
sys.path.insert(0, os.path.dirname(__file__))
from weight_cov_optimizer_v2 import WeightCovarianceFilterV2


def make_moons(n, feat_noise, seed):
    rng = np.random.RandomState(seed)
    n_per = n // 2
    t = np.linspace(0, np.pi, n_per)
    # upper moon (class 0)
    x0 = np.stack([np.cos(t), np.sin(t)], 1)
    # lower moon (class 1), shifted
    x1 = np.stack([1 - np.cos(t), 1 - np.sin(t) - 0.5], 1)
    X = np.concatenate([x0, x1], 0)
    y = np.concatenate([np.zeros(n_per), np.ones(n_per)]).astype(np.int64)
    X += rng.randn(*X.shape) * feat_noise
    return X.astype(np.float32), y


def flip_labels(y, frac, seed):
    rng = np.random.RandomState(seed)
    y = y.copy()
    idx = rng.choice(len(y), int(frac * len(y)), replace=False)
    y[idx] = 1 - y[idx]
    return y, idx


class MLP(nn.Module):
    def __init__(self, h=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, h), nn.ReLU(),
            nn.Linear(h, h), nn.ReLU(),
            nn.Linear(h, 2))

    def forward(self, x):
        return self.net(x)


def acc(model, X, y):
    with torch.no_grad():
        pred = model(X).argmax(1)
        return (pred == y).float().mean().item()


def train(mode, Xtr, ytr, Xte, yte, epochs, bs, lr, rank, decay, warmup, seed):
    torch.manual_seed(seed)
    model = MLP()
    base = torch.optim.Adam(model.parameters(), lr=lr)
    if mode == "filter":
        opt = WeightCovarianceFilterV2(model, base, rank=rank, decay=decay, warmup=warmup)
    n = len(Xtr)
    hist = []
    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            xb, yb = Xtr[idx], ytr[idx]
            if mode == "filter":
                opt.step(xb, yb)
            else:
                base.zero_grad()
                loss = nn.functional.cross_entropy(model(xb), yb)
                loss.backward()
                base.step()
        hist.append((ep, acc(model, Xtr, ytr), acc(model, Xte, yte)))
    return model, hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--noise", type=float, default=0.30)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--decay", type=float, default=0.9)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--lr", type=float, default=0.01)
    args = ap.parse_args()

    Xtr_np, ytr_clean = make_moons(300, 0.18, seed=0)
    ytr_np, flipped = flip_labels(ytr_clean, args.noise, seed=1)
    Xte_np, yte_np = make_moons(1000, 0.18, seed=2)  # clean labels

    Xtr, ytr = torch.tensor(Xtr_np), torch.tensor(ytr_np)
    Xte, yte = torch.tensor(Xte_np), torch.tensor(yte_np)

    for mode in ["adam", "filter"]:
        _, hist = train(mode, Xtr, ytr, Xte, yte, args.epochs, 32, args.lr,
                        args.rank, args.decay, args.warmup, seed=0)
        tr_final = np.mean([h[1] for h in hist[-10:]])
        te_final = np.mean([h[2] for h in hist[-10:]])
        te_best = max(h[2] for h in hist)
        print(f"{mode:7s}  train_acc={tr_final:.3f}  test_acc(final)={te_final:.3f}  "
              f"test_acc(best)={te_best:.3f}  gap={tr_final - te_final:+.3f}")


if __name__ == "__main__":
    main()
