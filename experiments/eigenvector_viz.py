#!/usr/bin/env python3
"""Mini example: what do the gradient-covariance eigenvectors look like?

We train a LINEAR softmax classifier on MNIST (W: 10x784) so that each parameter
direction lives directly in input (pixel) space and is therefore interpretable as
an image. We add 20% label noise. While training, we collect the batch-mean
gradient at every step, then form the FULL gradient covariance (exact, no
streaming needed at this scale) and eigendecompose it.

Thesis under test: the TOP eigenvectors are structured, class-discriminative
"feature" directions (the consensus signal), while the LOW-eigenvalue directions
are unstructured noise / memorization of specific (corrupted) examples.

Output: results/weight_covariance_v2/eigenvector_features.png
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 0
LABEL_NOISE = 0.2
EPOCHS = 8
BATCH = 128
WARMUP_STEPS = 20      # skip the initial transient before collecting gradients
DATA_DIR = "./data"
OUT = "../results/weight_covariance_v2/eigenvector_features.png"


def get_data():
    tf = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize((0.1307,), (0.3081,))])
    tr = datasets.MNIST(DATA_DIR, train=True, download=True, transform=tf)
    X = tr.data.float().view(-1, 784) / 255.0
    X = (X - 0.1307) / 0.3081
    y = tr.targets.clone()
    rng = np.random.RandomState(SEED)
    mask = torch.tensor(rng.random(len(y)) < LABEL_NOISE)
    rand_y = torch.tensor(rng.randint(0, 10, len(y)))
    y[mask] = rand_y[mask]
    return DataLoader(TensorDataset(X, y), batch_size=BATCH, shuffle=True)


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    loader = get_data()

    # Linear softmax classifier: logits = X @ W^T + b, W is (10, 784)
    W = torch.zeros(10, 784, requires_grad=True)
    b = torch.zeros(10, requires_grad=True)
    torch.nn.init.kaiming_uniform_(W, a=5 ** 0.5)
    opt = torch.optim.Adam([W, b], lr=1e-3)

    grads = []          # batch-mean gradients of W (flattened 7840) per step
    step = 0
    for epoch in range(EPOCHS):
        for x, yb in loader:
            opt.zero_grad()
            logits = x @ W.t() + b
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            step += 1
            if step > WARMUP_STEPS:
                grads.append(W.grad.detach().reshape(-1).clone())
            opt.step()

    G = torch.stack(grads).numpy()              # (n_steps, 7840)
    print(f"Collected {G.shape[0]} gradient snapshots of dim {G.shape[1]}")

    Gc = G - G.mean(0, keepdims=True)           # center
    cov = Gc.T @ Gc / Gc.shape[0]               # (7840, 7840) exact covariance
    evals, evecs = np.linalg.eigh(cov)          # ascending
    evals = evals[::-1]                         # descending
    evecs = evecs[:, ::-1]

    # effective rank (entropy of normalized eigenvalue spectrum)
    pos = evals[evals > 1e-12]
    p = pos / pos.sum()
    eff_rank = float(np.exp(-(p * np.log(p)).sum()))
    print(f"Top eigenvalues: {evals[:8]}")
    print(f"Effective rank: {eff_rank:.1f}")

    # Each eigenvector is 7840 = 10 classes x 784 pixels.
    # Per-pixel L2 norm across the 10 class-maps -> one 28x28 "where it acts" image.
    def pixel_map(vec):
        m = vec.reshape(10, 784)
        return np.linalg.norm(m, axis=0).reshape(28, 28)

    n_show = 8
    fig = plt.figure(figsize=(14, 7.5), facecolor="#0f1117")
    gs = fig.add_gridspec(3, n_show, height_ratios=[1.25, 1, 1], hspace=0.35, wspace=0.1)

    # Row 0 (spanning): eigenvalue spectrum
    ax = fig.add_subplot(gs[0, :])
    ax.semilogy(range(1, 61), evals[:60], "o-", color="#37c87a", ms=3, lw=1)
    ax.set_title(f"Gradient-covariance eigenvalue spectrum   (effective rank ≈ {eff_rank:.0f})",
                 color="#e6e9ef", fontsize=12)
    ax.set_xlabel("eigenvalue index", color="#9aa3b2")
    ax.set_ylabel("eigenvalue (log)", color="#9aa3b2")
    ax.tick_params(colors="#9aa3b2"); ax.set_facecolor("#181b24")
    for s in ax.spines.values(): s.set_color("#262b38")

    def strip(ax, img, title, color):
        ax.imshow(img, cmap="magma")
        ax.set_title(title, color=color, fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])

    # Row 1: top-8 eigenvectors (largest eigenvalues = consensus signal)
    for i in range(n_show):
        ax = fig.add_subplot(gs[1, i])
        strip(ax, pixel_map(evecs[:, i]), f"#{i+1}\nλ={evals[i]:.1e}", "#37c87a")
        if i == 0:
            ax.set_ylabel("TOP\n(signal)", color="#37c87a", fontsize=10, rotation=0,
                          labelpad=28, va="center")

    # Row 2: bottom-8 nonzero eigenvectors (smallest eigenvalues = noise/memorization)
    nz = int((evals > 1e-12).sum())
    for j in range(n_show):
        idx = nz - n_show + j
        ax = fig.add_subplot(gs[2, j])
        strip(ax, pixel_map(evecs[:, idx]), f"#{idx+1}\nλ={evals[idx]:.1e}", "#ff5d6c")
        if j == 0:
            ax.set_ylabel("BOTTOM\n(noise)", color="#ff5d6c", fontsize=10, rotation=0,
                          labelpad=28, va="center")

    fig.suptitle("What do the gradient-covariance eigenvectors look like?  "
                 "(linear MNIST, 20% label noise)", color="#e6e9ef", fontsize=14, y=0.99)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=130, bbox_inches="tight", facecolor="#0f1117")
    print(f"Saved {OUT}")

    # Also save the #1 eigenvector's 10 class-maps to show class-contrast structure
    fig2, axes = plt.subplots(1, 10, figsize=(16, 2), facecolor="#0f1117")
    m = evecs[:, 0].reshape(10, 784)
    vmax = np.abs(m).max()
    for c in range(10):
        axes[c].imshow(m[c].reshape(28, 28), cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        axes[c].set_title(f"class {c}", color="#e6e9ef", fontsize=9)
        axes[c].set_xticks([]); axes[c].set_yticks([])
    fig2.suptitle("Top eigenvector, broken out by class (red=+, blue=−): a class-contrast template",
                  color="#e6e9ef", fontsize=12)
    out2 = OUT.replace(".png", "_top_classmaps.png")
    fig2.savefig(out2, dpi=130, bbox_inches="tight", facecolor="#0f1117")
    print(f"Saved {out2}")


if __name__ == "__main__":
    main()
