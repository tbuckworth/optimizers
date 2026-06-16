#!/usr/bin/env python3
"""Minimal, self-contained demo of the spectral gradient filter.

Trains a small MLP on MNIST with 90% of the training labels randomised, twice:
once with plain Adam, once with Adam wrapped by the spectral gradient filter.

Plain Adam first fits the real digits (test accuracy peaks early) but then
*memorises the noise* — its training accuracy on the corrupted labels keeps
climbing while its test accuracy collapses. The filter projects out those
inconsistent, noise-fitting gradient directions, so its training accuracy stays
flat (it refuses to memorise) and its test accuracy holds. By the end the filter
is far ahead on test. This is the repo's headline result (H1) in one short script.

Run:
    python3 example.py

Downloads MNIST to ./data on first run. Takes ~1 minute (uses a 10k subset);
runs on CUDA/MPS automatically if available, else CPU.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets

from spectral_filter import SpectralGradientFilter

# --- config (small train subset + high noise so the effect is fast & dramatic) ---
LABEL_NOISE = 0.9      # fraction of training labels randomised
N_TRAIN = 10000        # subset size — smaller set memorises faster, so the gap is clear
EPOCHS = 20
RANK = 50              # eigendirections the filter keeps
LR = 1e-3
SEED = 0


def get_data(label_noise):
    tr = datasets.MNIST("./data", train=True, download=True)
    te = datasets.MNIST("./data", train=False, download=True)
    Xtr = ((tr.data.float() / 255.0 - 0.1307) / 0.3081).view(-1, 784)[:N_TRAIN]
    Xte = ((te.data.float() / 255.0 - 0.1307) / 0.3081).view(-1, 784)
    ytr, yte = tr.targets.clone()[:N_TRAIN], te.targets.clone()
    rng = np.random.RandomState(SEED)
    mask = rng.random(N_TRAIN) < label_noise
    ytr[mask] = torch.tensor(rng.randint(0, 10, mask.sum()))
    train = DataLoader(TensorDataset(Xtr, ytr), batch_size=64, shuffle=True)
    test = DataLoader(TensorDataset(Xte, yte), batch_size=512)
    return train, test


def mlp():
    return nn.Sequential(
        nn.Linear(784, 256), nn.ReLU(),
        nn.Linear(256, 128), nn.ReLU(),
        nn.Linear(128, 10),
    )


@torch.no_grad()
def accuracy(model, loader, device):
    model.eval()
    correct = total = 0
    for x, y in loader:
        pred = model(x.to(device)).argmax(1).cpu()
        correct += (pred == y).sum().item()
        total += len(y)
    return correct / total


def train_run(use_filter, train, test, device):
    torch.manual_seed(SEED)
    model = mlp().to(device)
    base_opt = torch.optim.Adam(model.parameters(), lr=LR)
    filt = SpectralGradientFilter(model, base_opt, rank=RANK) if use_filter else None

    for epoch in range(EPOCHS):
        model.train()
        for x, y in train:
            x, y = x.to(device), y.to(device)
            base_opt.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            if filt is not None:
                filt.filter_grad()      # <-- the only added line vs. plain Adam
            base_opt.step()
        tr_acc = accuracy(model, train, device)
        te_acc = accuracy(model, test, device)
        print(f"  epoch {epoch:2d}: train(noisy labels)={tr_acc:.3f}  test(clean)={te_acc:.3f}")
    return accuracy(model, test, device)


def main():
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device={device}, {int(LABEL_NOISE*100)}% label noise, {N_TRAIN} train examples\n")
    train, test = get_data(LABEL_NOISE)

    print("Plain Adam (watch train climb as it memorises noise, test then collapse):")
    adam_test = train_run(False, train, test, device)
    print("\nAdam + spectral gradient filter (train stays flat; it refuses to memorise):")
    filt_test = train_run(True, train, test, device)

    print("\n" + "=" * 56)
    print(f"final test accuracy   plain Adam: {adam_test:.3f}")
    print(f"final test accuracy   + filter:   {filt_test:.3f}")
    print(f"the filter holds {filt_test - adam_test:+.3f} test accuracy over plain Adam "
          f"by\nrefusing to memorise the random labels.")


if __name__ == "__main__":
    main()
