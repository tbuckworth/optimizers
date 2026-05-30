#!/usr/bin/env python3
"""CIFAR-10 label-noise robustness: first non-MNIST test of the filter.

Small CNN (~1.1M params) on CIFAR-10 with symmetric label noise. Compares:
  adam    full Adam (reference)
  ours    filter+Adam from the start
  switch  Adam until train_acc>=THRESH, then enable the filter (the recipe:
          learn fast early, prevent late memorization)

Reports per-epoch clean-test accuracy and train accuracy (to show fitting-the-noise).

Usage:
  python3 run_cifar_noise.py --mode ours --noise 0.4 --seed 0 --name ours_n40_s0
"""
import argparse, json, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from weight_cov_optimizer_v2 import WeightCovarianceFilterV2


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(64 * 8 * 8, 128), nn.ReLU(), nn.Linear(128, 10))

    def forward(self, x):
        return self.head(self.features(x))


def get_data(noise, seed, data_dir):
    mean = (0.4914, 0.4822, 0.4465); std = (0.2470, 0.2435, 0.2616)
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    tr = datasets.CIFAR10(data_dir, train=True, download=True, transform=tf)
    te = datasets.CIFAR10(data_dir, train=False, download=True, transform=tf)
    Xtr = torch.stack([tr[i][0] for i in range(len(tr))])
    ytr = torch.tensor(tr.targets)
    Xte = torch.stack([te[i][0] for i in range(len(te))])
    yte = torch.tensor(te.targets)
    if noise > 0:
        rng = np.random.RandomState(seed)
        mask = rng.rand(len(ytr)) < noise
        ytr[torch.tensor(mask)] = torch.tensor(rng.randint(0, 10, mask.sum()))
    return (DataLoader(TensorDataset(Xtr, ytr), batch_size=128, shuffle=True),
            DataLoader(TensorDataset(Xte, yte), batch_size=512))


def evaluate(model, loader, device):
    model.eval(); c = t = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            c += (model(x).argmax(1) == y).sum().item(); t += len(y)
    return c / t


def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    train_loader, test_loader = get_data(args.noise, args.seed, args.data_dir)
    model = SmallCNN().to(device)
    print(f"device={device} noise={args.noise} params={sum(p.numel() for p in model.parameters()):,} mode={args.mode}")
    base = torch.optim.Adam(model.parameters(), lr=args.lr)

    filt = None
    if args.mode == "ours":
        filt = WeightCovarianceFilterV2(model, base, rank=args.rank, decay=args.decay, warmup=args.warmup)
    filtering = (args.mode == "ours")

    metrics = []; t0 = time.time(); switch_ep = -1
    for epoch in range(args.epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            if filtering and filt is not None:
                filt.step(x, y)
            else:
                base.zero_grad(); F.cross_entropy(model(x), y).backward(); base.step()
        tr = evaluate(model, train_loader, device); te = evaluate(model, test_loader, device)
        metrics.append({"epoch": epoch, "train_acc": round(tr, 4), "test_acc": round(te, 4)})
        # switch recipe: enable filter once train acc crosses threshold
        if args.mode == "switch" and filt is None and tr >= args.switch_at:
            filt = WeightCovarianceFilterV2(model, base, rank=args.rank, decay=args.decay, warmup=args.warmup)
            filtering = True; switch_ep = epoch
            print(f"  >>> switch ON at epoch {epoch} (train_acc={tr:.3f})")
        print(f"  ep{epoch:3d} train={tr:.4f} test={te:.4f} [{time.time()-t0:.0f}s]")

    best = max(m["test_acc"] for m in metrics)
    os.makedirs(args.save_dir, exist_ok=True)
    out = {"config": vars(args), "metrics": metrics, "best_test": best,
           "final_test": metrics[-1]["test_acc"], "switch_epoch": switch_ep, "time_s": round(time.time()-t0, 1)}
    json.dump(out, open(os.path.join(args.save_dir, f"{args.name}.json"), "w"))
    print(f"Done: best={best:.4f} final={metrics[-1]['test_acc']:.4f} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["adam", "ours", "switch"], required=True)
    p.add_argument("--noise", type=float, default=0.0)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--rank", type=int, default=200)
    p.add_argument("--decay", type=float, default=0.99)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--switch_at", type=float, default=0.6)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--name", type=str, required=True)
    p.add_argument("--save_dir", type=str, default="../results/cifar_noise")
    p.add_argument("--data_dir", type=str, default="./data")
    run(p.parse_args())
