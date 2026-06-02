#!/usr/bin/env python3
"""Run a single weight-covariance v2 optimizer training run.

Usage:
  python3 run_single_weight_cov_v2.py --mode ours --lr 1e-3 --rank 200 --decay 0.99
  python3 run_single_weight_cov_v2.py --mode adam --lr 1e-3
  python3 run_single_weight_cov_v2.py --mode ours --dataset random_labels --rank 200
"""

import argparse
import json
import time
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

from weight_cov_optimizer_v2 import WeightCovarianceFilterV2
from random_subspace_optimizer import RandomSubspaceFilter
from lora_mlp import LoRAMLP


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


def get_mnist_data(random_labels=False, noise_features=0, label_noise=0.0, seed=42, data_dir="./data"):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_ds = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(data_dir, train=False, download=True, transform=transform)

    train_X = train_ds.data.float().view(-1, 784) / 255.0
    train_X = (train_X - 0.1307) / 0.3081
    train_y = train_ds.targets.clone()
    test_X = test_ds.data.float().view(-1, 784) / 255.0
    test_X = (test_X - 0.1307) / 0.3081
    test_y = test_ds.targets.clone()

    if random_labels:
        rng = np.random.RandomState(seed)
        train_y = torch.tensor(rng.randint(0, 10, len(train_y)))
    elif label_noise > 0:
        rng = np.random.RandomState(seed)
        mask = torch.tensor(rng.random(len(train_y)) < label_noise)
        random_y = torch.tensor(rng.randint(0, 10, len(train_y)))
        train_y[mask] = random_y[mask]

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


def make_base_optimizer(name, params, lr):
    if name == "adam":
        return torch.optim.Adam(params, lr=lr)
    elif name == "sgd":
        return torch.optim.SGD(params, lr=lr)
    elif name == "sgdm":
        return torch.optim.SGD(params, lr=lr, momentum=0.9)
    raise ValueError(f"Unknown base optimizer: {name}")


def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    random_labels = (args.dataset == "random_labels")
    noise_features = 784 if args.dataset == "noisy_mnist" else 0
    train_loader, test_loader, input_dim = get_mnist_data(
        random_labels=random_labels, noise_features=noise_features,
        label_noise=args.label_noise, seed=args.seed, data_dir=args.data_dir)

    if args.mode == "lora":
        model = LoRAMLP(input_dim, r=args.lora_rank, alpha=args.lora_alpha).to(device)
    else:
        model = FlexMNISTNet(input_dim).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Dataset: {args.dataset}, Input: {input_dim}, Params: {n_params:,}, "
          f"Mode: {args.mode}, Device: {device}")

    # Evaluate before any training
    init_train_acc, init_train_loss = evaluate(model, train_loader, device)
    init_test_acc, init_test_loss = evaluate(model, test_loader, device)
    print(f"  Epoch  -1: train={init_train_acc:.4f} test={init_test_acc:.4f} "
          f"loss={init_train_loss:.4f} (0s)")

    metrics = [{
        "epoch": -1, "train_acc": round(init_train_acc, 5),
        "test_acc": round(init_test_acc, 5), "train_loss": round(init_train_loss, 5),
        "test_loss": round(init_test_loss, 5),
    }]
    t_start = time.time()

    if args.mode in ("adam", "baseline", "lora"):
        if args.mode == "adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        elif args.mode == "lora":
            optimizer = torch.optim.Adam(model.trainable_parameters(), lr=args.lr)
        else:
            optimizer = make_base_optimizer(args.base_optimizer,
                                           model.parameters(), args.lr)
        for epoch in range(args.epochs):
            model.train()
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                loss = F.cross_entropy(model(x), y)
                loss.backward()
                optimizer.step()

            train_acc, train_loss = evaluate(model, train_loader, device)
            test_acc, test_loss = evaluate(model, test_loader, device)
            metrics.append({
                "epoch": epoch, "train_acc": round(train_acc, 5),
                "test_acc": round(test_acc, 5), "train_loss": round(train_loss, 5),
                "test_loss": round(test_loss, 5),
            })
            print(f"  Epoch {epoch:3d}: train={train_acc:.4f} test={test_acc:.4f} "
                  f"loss={train_loss:.4f} ({time.time()-t_start:.0f}s)")

    elif args.mode in ("ours", "random_subspace"):
        base_opt = make_base_optimizer(args.base_optimizer,
                                       model.parameters(), args.lr)
        if args.mode == "ours":
            optimizer = WeightCovarianceFilterV2(
                model, base_opt, rank=args.rank, decay=args.decay,
                warmup=args.warmup, filter_strength=args.filter_strength,
                normalize=args.normalize)
        else:
            optimizer = RandomSubspaceFilter(
                model, base_opt, rank=args.rank, seed=args.seed)

        for epoch in range(args.epochs):
            model.train()
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.step(x, y)

            train_acc, train_loss = evaluate(model, train_loader, device)
            test_acc, test_loss = evaluate(model, test_loader, device)
            entry = {
                "epoch": epoch, "train_acc": round(train_acc, 5),
                "test_acc": round(test_acc, 5), "train_loss": round(train_loss, 5),
                "test_loss": round(test_loss, 5),
            }
            if hasattr(optimizer, 'S') and optimizer.S is not None:
                entry["effective_rank"] = round(optimizer._effective_rank(), 2)
            metrics.append(entry)
            print(f"  Epoch {epoch:3d}: train={train_acc:.4f} test={test_acc:.4f} "
                  f"loss={train_loss:.4f} ({time.time()-t_start:.0f}s)")

    dt = time.time() - t_start
    result = {
        "config": vars(args), "metrics": metrics, "time_s": round(dt, 1),
        "n_params": n_params, "input_dim": input_dim,
    }

    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, args.name + ".json")
    with open(save_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nDone: {dt:.0f}s. Saved to {save_path}")
    final = metrics[-1]
    print(f"Final: train={final['train_acc']:.4f} test={final['test_acc']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["adam", "ours", "baseline",
                        "random_subspace", "lora"], required=True)
    parser.add_argument("--base_optimizer", choices=["adam", "sgd", "sgdm"],
                        default="adam")
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--lora_alpha", type=float, default=32.0)
    parser.add_argument("--dataset", choices=["standard", "noisy_mnist", "random_labels"],
                        default="standard")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--rank", type=int, default=200)
    parser.add_argument("--decay", type=float, default=0.99)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--filter_strength", type=float, default=1.0)
    parser.add_argument("--normalize", choices=["none", "var", "degree"], default="none",
                        help="basis: none=covariance, var=correlation, degree=spectral/normalized-affinity")
    parser.add_argument("--label_noise", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="../results/weight_covariance_v2")
    parser.add_argument("--data_dir", type=str, default="./data")
    args = parser.parse_args()
    run(args)
