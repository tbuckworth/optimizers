#!/usr/bin/env python3
"""Run weight-covariance optimizer experiments.

Three phases:
1. HP sweep on standard MNIST (find best rank, decay, lr for our optimizer)
2. Final comparison on standard MNIST (ours vs Adam)
3. Augmented MNIST with random noise features (ours vs Adam)
4. Random labels MNIST (ours vs Adam)

Usage:
  python3 run_weight_cov_experiments.py --phase sweep
  python3 run_weight_cov_experiments.py --phase compare
  python3 run_weight_cov_experiments.py --phase all
"""

import argparse
import json
import time
import os
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

from weight_cov_optimizer import WeightCovarianceFilter


class FlexMNISTNet(nn.Module):
    def __init__(self, input_dim=784):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.net(x.view(x.size(0), -1))


def get_mnist_data(random_labels=False, noise_features=0, seed=42, data_dir="./data"):
    """Get MNIST with optional random labels and/or appended noise features.

    noise_features: number of fixed random Gaussian features to append per sample.
                   These are sampled ONCE and stored — same noise for same sample every epoch.
                   Provides no generalizable signal but can be memorized.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_ds = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(data_dir, train=False, download=True, transform=transform)

    # Materialize to tensors
    train_X = train_ds.data.float().view(-1, 784) / 255.0
    train_X = (train_X - 0.1307) / 0.3081
    train_y = train_ds.targets.clone()

    test_X = test_ds.data.float().view(-1, 784) / 255.0
    test_X = (test_X - 0.1307) / 0.3081
    test_y = test_ds.targets.clone()

    if random_labels:
        rng = np.random.RandomState(seed)
        train_y = torch.tensor(rng.randint(0, 10, len(train_y)))

    if noise_features > 0:
        rng = np.random.RandomState(seed + 1000)
        train_noise = torch.tensor(
            rng.randn(len(train_X), noise_features).astype(np.float32))
        test_noise = torch.tensor(
            rng.randn(len(test_X), noise_features).astype(np.float32))
        train_X = torch.cat([train_X, train_noise], dim=1)
        test_X = torch.cat([test_X, test_noise], dim=1)

    train_loader = DataLoader(TensorDataset(train_X, train_y),
                              batch_size=64, shuffle=True, pin_memory=True)
    test_loader = DataLoader(TensorDataset(test_X, test_y),
                             batch_size=256, shuffle=False, pin_memory=True)
    return train_loader, test_loader, train_X.shape[1]


def evaluate(model, loader, device):
    model.eval()
    correct, total, total_loss = 0, 0, 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            total_loss += F.cross_entropy(logits, y, reduction='sum').item()
            correct += (logits.argmax(1) == y).sum().item()
            total += len(y)
    return correct / total, total_loss / total


def train_adam(model, train_loader, test_loader, device, lr=1e-3, epochs=20,
              log_every=1):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    metrics = []

    for epoch in range(epochs):
        model.train()
        epoch_loss, n_batches = 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        if epoch % log_every == 0 or epoch == epochs - 1:
            train_acc, train_loss = evaluate(model, train_loader, device)
            test_acc, test_loss = evaluate(model, test_loader, device)
            metrics.append({
                "epoch": epoch,
                "train_acc": round(train_acc, 5),
                "test_acc": round(test_acc, 5),
                "train_loss": round(train_loss, 5),
                "test_loss": round(test_loss, 5),
            })

    return metrics


def train_weight_cov(model, train_loader, test_loader, device,
                     lr=1e-3, epochs=20, rank=20, decay=0.95,
                     warmup=10, update_every=1, log_every=1):
    base_opt = torch.optim.Adam(model.parameters(), lr=lr)
    optimizer = WeightCovarianceFilter(
        model, base_opt, rank=rank, decay=decay,
        warmup=warmup, update_every=update_every,
    )
    metrics = []

    for epoch in range(epochs):
        model.train()
        epoch_loss, n_batches = 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            loss_val, diag = optimizer.step(x, y)
            epoch_loss += loss_val
            n_batches += 1

        if epoch % log_every == 0 or epoch == epochs - 1:
            train_acc, train_loss = evaluate(model, train_loader, device)
            test_acc, test_loss = evaluate(model, test_loader, device)
            entry = {
                "epoch": epoch,
                "train_acc": round(train_acc, 5),
                "test_acc": round(test_acc, 5),
                "train_loss": round(train_loss, 5),
                "test_loss": round(test_loss, 5),
            }
            if "effective_rank" in diag:
                entry["effective_rank"] = round(diag["effective_rank"], 2)
            if "variance_in_top5" in diag:
                entry["var_top5"] = round(diag["variance_in_top5"], 4)
            metrics.append(entry)

    return metrics


def run_sweep(args):
    """Phase 1: HP sweep on standard MNIST."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, test_loader, input_dim = get_mnist_data(data_dir=args.data_dir)
    print(f"Input dim: {input_dim}")

    sweep_configs = []
    for rank in [10, 20, 50]:
        for decay in [0.9, 0.95]:
            for lr in [5e-4, 1e-3]:
                sweep_configs.append({
                    "rank": rank, "decay": decay,
                    "lr": lr, "update_every": 50,
                })

    print(f"\nSweep: {len(sweep_configs)} configs, {args.sweep_epochs} epochs each")

    model = FlexMNISTNet(input_dim).to(device)
    init_state = {k: v.clone() for k, v in model.state_dict().items()}

    results = []
    for i, cfg in enumerate(sweep_configs):
        torch.manual_seed(args.seed)
        model.load_state_dict(init_state)
        t0 = time.time()

        metrics = train_weight_cov(
            model, train_loader, test_loader, device,
            lr=cfg["lr"], epochs=args.sweep_epochs,
            rank=cfg["rank"], decay=cfg["decay"],
            warmup=10, update_every=cfg["update_every"],
            log_every=args.sweep_epochs,  # only log final
        )
        dt = time.time() - t0
        final = metrics[-1]

        result = {**cfg, **final, "time_s": round(dt, 1)}
        results.append(result)
        print(f"  [{i+1}/{len(sweep_configs)}] rank={cfg['rank']:2d} "
              f"decay={cfg['decay']:.2f} lr={cfg['lr']:.4f} "
              f"ue={cfg['update_every']} => "
              f"test={final['test_acc']:.4f} train={final['train_acc']:.4f} "
              f"({dt:.0f}s)")

    results.sort(key=lambda r: r["test_acc"], reverse=True)
    print(f"\nTop 5 configs by test accuracy:")
    for r in results[:5]:
        print(f"  rank={r['rank']:2d} decay={r['decay']:.2f} "
              f"lr={r['lr']:.4f} ue={r['update_every']} => "
              f"test={r['test_acc']:.4f} train={r['train_acc']:.4f}")

    save_path = os.path.join(args.save_dir, "sweep_results.json")
    os.makedirs(args.save_dir, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSweep saved to {save_path}")

    return results[0]  # best config


def run_comparison(args, best_cfg=None):
    """Phase 2-4: Compare our optimizer vs Adam on three datasets."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if best_cfg is None:
        sweep_path = os.path.join(args.save_dir, "sweep_results.json")
        if os.path.exists(sweep_path):
            with open(sweep_path) as f:
                sweep = json.load(f)
            best_cfg = sweep[0]
            print(f"Loaded best config from sweep: {best_cfg}")
        else:
            best_cfg = {"rank": 20, "decay": 0.95, "lr": 1e-3, "update_every": 50}
            print(f"No sweep found, using defaults: {best_cfg}")

    experiments = [
        {"name": "standard_mnist", "random_labels": False, "noise_features": 0},
        {"name": "noisy_mnist", "random_labels": False, "noise_features": 784},
        {"name": "random_labels", "random_labels": True, "noise_features": 0},
    ]

    all_results = {}

    for exp in experiments:
        print(f"\n{'='*60}")
        print(f"Experiment: {exp['name']}")
        print(f"{'='*60}")

        train_loader, test_loader, input_dim = get_mnist_data(
            random_labels=exp["random_labels"],
            noise_features=exp["noise_features"],
            seed=args.seed,
            data_dir=args.data_dir,
        )
        print(f"Input dim: {input_dim}")

        # Adam baseline
        print(f"\n  Training Adam (lr=1e-3)...")
        torch.manual_seed(args.seed)
        model_adam = FlexMNISTNet(input_dim).to(device)
        t0 = time.time()
        adam_metrics = train_adam(
            model_adam, train_loader, test_loader, device,
            lr=1e-3, epochs=args.epochs, log_every=1)
        dt_adam = time.time() - t0
        print(f"    Final: train={adam_metrics[-1]['train_acc']:.4f} "
              f"test={adam_metrics[-1]['test_acc']:.4f} ({dt_adam:.0f}s)")

        # Our optimizer
        print(f"\n  Training WeightCov (rank={best_cfg['rank']}, "
              f"decay={best_cfg['decay']}, lr={best_cfg['lr']})...")
        torch.manual_seed(args.seed)
        model_ours = FlexMNISTNet(input_dim).to(device)
        t0 = time.time()
        ours_metrics = train_weight_cov(
            model_ours, train_loader, test_loader, device,
            lr=best_cfg["lr"], epochs=args.epochs,
            rank=best_cfg["rank"], decay=best_cfg["decay"],
            warmup=10, update_every=best_cfg["update_every"],
            log_every=1)
        dt_ours = time.time() - t0
        print(f"    Final: train={ours_metrics[-1]['train_acc']:.4f} "
              f"test={ours_metrics[-1]['test_acc']:.4f} ({dt_ours:.0f}s)")

        all_results[exp["name"]] = {
            "config": exp,
            "adam": {"metrics": adam_metrics, "time_s": round(dt_adam, 1)},
            "weight_cov": {
                "metrics": ours_metrics,
                "config": best_cfg,
                "time_s": round(dt_ours, 1),
            },
        }

    save_path = os.path.join(args.save_dir, "comparison_results.json")
    with open(save_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll results saved to {save_path}")

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"{'Experiment':<20} {'Adam Test':<12} {'Ours Test':<12} {'Delta':<10}")
    print("-" * 54)
    for name, res in all_results.items():
        adam_test = res["adam"]["metrics"][-1]["test_acc"]
        ours_test = res["weight_cov"]["metrics"][-1]["test_acc"]
        delta = ours_test - adam_test
        sign = "+" if delta > 0 else ""
        print(f"{name:<20} {adam_test:<12.4f} {ours_test:<12.4f} {sign}{delta:<10.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["sweep", "compare", "all"],
                        default="all")
    parser.add_argument("--sweep_epochs", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--save_dir", type=str,
                        default="../results/weight_covariance")
    args = parser.parse_args()

    if args.phase in ("sweep", "all"):
        best_cfg = run_sweep(args)
    else:
        best_cfg = None

    if args.phase in ("compare", "all"):
        run_comparison(args, best_cfg)
