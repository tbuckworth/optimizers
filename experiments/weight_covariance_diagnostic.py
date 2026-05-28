#!/usr/bin/env python3
"""Diagnostic: does the weight-covariance eigenspectrum differ between
real and random label training?

For each weight in the network, its gradient varies across samples.
We compute the p×p weight-covariance matrix (via the N×N Gram trick)
at several training checkpoints, and compare eigenvalue spectra
for real vs random labels.

Hypothesis: real labels → few large eigenvalues (weight clusters learning
features). Random labels → flat spectrum (many small independent groups
from memorization).
"""

import argparse
import json
import time
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.func import vmap, grad_and_value, functional_call
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset


class MNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.net(x.view(x.size(0), -1))


def get_data(random_labels=False, seed=42, data_dir="./data"):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_ds = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(data_dir, train=False, download=True, transform=transform)

    if random_labels:
        rng = np.random.RandomState(seed)
        train_ds.targets = torch.tensor(rng.randint(0, 10, len(train_ds.targets)))

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True,
                              num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False,
                             num_workers=2, pin_memory=True)
    return train_ds, train_loader, test_loader


def compute_weight_covariance_spectrum(model, dataset, n_samples=1000, seed=42, device="cuda"):
    """Compute eigenvalues of the p×p weight-covariance matrix using the Gram trick.

    Each sample gives a gradient vector in R^p. We collect N such vectors (N×p matrix X),
    then eigendecompose X X^T / N (N×N) which has the same nonzero eigenvalues as
    X^T X / N (p×p).

    Returns eigenvalues (descending), plus the raw flattened gradient matrix for
    optional further analysis.
    """
    model.eval()
    params = {k: v for k, v in model.named_parameters()}
    buffers = {k: v for k, v in model.named_buffers()}
    param_keys = list(params.keys())
    loss_fn = nn.CrossEntropyLoss()

    def compute_loss(params_, buffers_, x, y):
        out = functional_call(model, (params_, buffers_), (x.unsqueeze(0),))
        return loss_fn(out, y.unsqueeze(0))

    grad_fn = vmap(
        grad_and_value(compute_loss),
        in_dims=(None, None, 0, 0),
    )

    rng = np.random.RandomState(seed)
    indices = rng.choice(len(dataset), n_samples, replace=False)

    images = torch.stack([dataset[i][0] for i in indices]).to(device)
    labels = torch.tensor([dataset[i][1] for i in indices]).to(device)

    chunk_size = 200
    all_grads = []
    for start in range(0, n_samples, chunk_size):
        end = min(start + chunk_size, n_samples)
        x_chunk = images[start:end]
        y_chunk = labels[start:end]

        per_sample_grads, _ = grad_fn(params, buffers, x_chunk, y_chunk)

        flat_list = []
        for key in param_keys:
            g = per_sample_grads[key]
            flat_list.append(g.reshape(end - start, -1))
        flat = torch.cat(flat_list, dim=1)
        all_grads.append(flat.detach())

    G = torch.cat(all_grads, dim=0)  # N × p

    G_centered = G - G.mean(dim=0, keepdim=True)

    gram = G_centered @ G_centered.T / n_samples  # N × N

    eigenvalues = torch.linalg.eigvalsh(gram).flip(0).clamp(min=0)

    effective_rank = compute_effective_rank(eigenvalues)

    total_var = eigenvalues.sum().item()
    cumvar = eigenvalues.cumsum(0) / total_var if total_var > 0 else eigenvalues

    return {
        "eigenvalues": eigenvalues.cpu().tolist(),
        "effective_rank": effective_rank,
        "total_variance": total_var,
        "cumulative_variance": cumvar.cpu().tolist(),
        "n_samples": n_samples,
        "n_params": G.shape[1],
    }


def compute_effective_rank(eigenvalues):
    """exp(entropy of normalized eigenvalues) — measures 'how many dimensions matter'."""
    eigs = eigenvalues[eigenvalues > 1e-12]
    if len(eigs) == 0:
        return 0.0
    p = eigs / eigs.sum()
    entropy = -(p * p.log()).sum().item()
    return float(np.exp(entropy))


def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += len(y)
    return correct / total


def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    results = {}

    for label_type in ["real", "random"]:
        random_labels = (label_type == "random")
        print(f"\n{'='*60}")
        print(f"Training with {label_type} labels")
        print(f"{'='*60}")

        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        train_ds, train_loader, test_loader = get_data(
            random_labels=random_labels, seed=args.seed)

        model = MNISTNet().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Params: {n_params:,}")

        checkpoints = sorted(set([0] + args.checkpoints))
        spectra = []
        train_accs = []
        test_accs = []

        for epoch in range(max(checkpoints) + 1):
            if epoch in checkpoints:
                print(f"\n  Checkpoint epoch {epoch}:")
                train_acc = evaluate(model, train_loader, device)
                test_acc = evaluate(model, test_loader, device)
                print(f"    Train acc: {train_acc:.4f}, Test acc: {test_acc:.4f}")

                t0 = time.time()
                spectrum = compute_weight_covariance_spectrum(
                    model, train_ds, n_samples=args.n_samples,
                    seed=args.seed, device=device)
                dt = time.time() - t0
                print(f"    Spectrum computed in {dt:.1f}s")
                print(f"    Effective rank: {spectrum['effective_rank']:.1f}")
                print(f"    Top 5 eigenvalues: {[f'{e:.4f}' for e in spectrum['eigenvalues'][:5]]}")

                var_5 = spectrum["cumulative_variance"][4] if len(spectrum["cumulative_variance"]) > 4 else 0
                var_20 = spectrum["cumulative_variance"][19] if len(spectrum["cumulative_variance"]) > 19 else 0
                print(f"    Variance in top-5: {var_5:.4f}, top-20: {var_20:.4f}")

                spectra.append({
                    "epoch": epoch,
                    "eigenvalues": spectrum["eigenvalues"],
                    "effective_rank": spectrum["effective_rank"],
                    "total_variance": spectrum["total_variance"],
                    "cumulative_variance": spectrum["cumulative_variance"],
                })
                train_accs.append({"epoch": epoch, "accuracy": train_acc})
                test_accs.append({"epoch": epoch, "accuracy": test_acc})

            model.train()
            epoch_loss = 0
            n_batches = 0
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                loss = F.cross_entropy(model(x), y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            if epoch % 5 == 0 or epoch in checkpoints:
                print(f"  Epoch {epoch}: loss={epoch_loss/n_batches:.4f}")

        results[label_type] = {
            "spectra": spectra,
            "train_accs": train_accs,
            "test_accs": test_accs,
            "n_params": n_params,
            "n_samples_for_spectrum": args.n_samples,
        }

    save_path = os.path.join(args.save_dir, "weight_covariance_diagnostic.json")
    os.makedirs(args.save_dir, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(results, f)
    print(f"\nResults saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_samples", type=int, default=1000,
                        help="Number of samples for spectrum computation")
    parser.add_argument("--checkpoints", type=int, nargs="+",
                        default=[0, 1, 2, 5, 10, 20, 50],
                        help="Epochs at which to compute spectrum")
    parser.add_argument("--save_dir", type=str, default="../results/weight_covariance")
    args = parser.parse_args()
    run(args)
