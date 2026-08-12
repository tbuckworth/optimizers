#!/usr/bin/env python3
"""Sparse parity grokking: does the filter accelerate it like on modular addition?

(n, k)-sparse parity (Barak et al. 2022): input is an n-bit vector; the label is
the parity (XOR) of k hidden bits. A classic grokking-adjacent task — the network
memorizes then suddenly generalizes once it finds the k relevant bits.

Compares AdamW vs filter+AdamW (WeightCovarianceFilterV2). Full-batch training.

Usage:
  python3 sparse_parity.py --mode ours  --seed 0
  python3 sparse_parity.py --mode adamw --seed 0
"""
import argparse, json, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from weight_cov_optimizer_v2 import WeightCovarianceFilterV2


class ParityMLP(nn.Module):
    def __init__(self, n, hidden=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 2))

    def forward(self, x):
        return self.net(x)


def make_data(n, k, train_size, test_size, seed):
    rng = np.random.RandomState(seed)
    idx = rng.choice(n, k, replace=False)           # the k relevant bits
    def sample(m):
        X = rng.randint(0, 2, (m, n)).astype(np.float32)
        y = (X[:, idx].sum(1) % 2).astype(np.int64)
        return torch.tensor(X * 2 - 1), torch.tensor(y)   # +/-1 encoding
    Xtr, ytr = sample(train_size)
    Xte, yte = sample(test_size)
    return Xtr, ytr, Xte, yte, idx


def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    Xtr, ytr, Xte, yte, idx = make_data(args.n, args.k, args.train_size, args.test_size, args.seed)
    Xtr, ytr, Xte, yte = Xtr.to(device), ytr.to(device), Xte.to(device), yte.to(device)
    model = ParityMLP(args.n, args.hidden).to(device)
    print(f"device={device} n={args.n} k={args.k} relevant_bits={idx.tolist()} "
          f"params={sum(p.numel() for p in model.parameters()):,} mode={args.mode}")

    base = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    if args.mode == "ours":
        opt = WeightCovarianceFilterV2(
            model, base, rank=args.rank, decay=args.decay,
            warmup=args.warmup, stable_update=not args.legacy_update)
    metrics = []; t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        if args.mode == "ours":
            opt.step(Xtr, ytr)
        else:
            base.zero_grad(); F.cross_entropy(model(Xtr), ytr).backward(); base.step()
        if epoch % args.log_every == 0:
            model.eval()
            with torch.no_grad():
                tr = (model(Xtr).argmax(1) == ytr).float().mean().item()
                te = (model(Xte).argmax(1) == yte).float().mean().item()
            metrics.append({"epoch": epoch, "train_acc": round(tr, 4), "test_acc": round(te, 4)})
            if epoch % (args.log_every * 20) == 0:
                print(f"  ep{epoch:6d} train={tr:.3f} test={te:.3f} [{time.time()-t0:.0f}s]")
    grok = next((m["epoch"] for m in metrics if m["test_acc"] >= 0.9), None)
    os.makedirs(args.save_dir, exist_ok=True)
    out = {"config": vars(args), "metrics": metrics, "grok_epoch": grok,
           "relevant_bits": idx.tolist(), "time_s": round(time.time() - t0, 1)}
    path = os.path.join(args.save_dir, f"{args.name}.json")
    json.dump(out, open(path, "w"))
    print(f"Done: grok@{grok}  final_test={metrics[-1]['test_acc']:.3f}  -> {path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["adamw", "ours"], required=True)
    p.add_argument("--n", type=int, default=40)
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--train_size", type=int, default=1000)
    p.add_argument("--test_size", type=int, default=2000)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=1e-2)
    p.add_argument("--epochs", type=int, default=20000)
    p.add_argument("--rank", type=int, default=200)
    p.add_argument("--decay", type=float, default=0.99)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--legacy_update", action="store_true")
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--name", type=str, required=True)
    p.add_argument("--save_dir", type=str, default="../results/sparse_parity")
    run(p.parse_args())
