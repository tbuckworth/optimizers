#!/usr/bin/env python3
"""Direction B — eigenvector → layer attribution.

Each gradient-covariance eigenvector lives in flattened-parameter space, so every
component maps back to a (layer, weight). We train the small MLP (784->256->128->10)
with the real WeightCovarianceFilterV2, then take the eigenvectors the filter
actually uses (optimizer.V) and ask: what fraction of each eigenvector's L2 mass
sits in each layer?

Hypothesis: the TOP (high-eigenvalue, high-agreement) eigenvectors are cross-layer
coherent function-space directions; the TAIL eigenvectors are layer-local noise.

Outputs:
  results/weight_covariance_v2/eigvec_layer_attribution_{clean,noisy}.png
  results/weight_covariance_v2/eigvec_layer_attribution.json
"""
import os, sys, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from weight_cov_optimizer_v2 import WeightCovarianceFilterV2

SEED = 0
EPOCHS = 6
BATCH = 128
RANK = 60          # track a decent basis so we can see the tail
DECAY = 0.99
WARMUP = 100
LR = 1e-3
DATA_DIR = "./data"
OUTDIR = "../results/weight_covariance_v2"


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 10))

    def forward(self, x):
        return self.net(x.view(x.size(0), -1))


# Parameter blocks in flatten order, grouped into the 3 weight layers
# (each layer = weight + bias). Names match nn.Sequential indices.
LAYER_GROUPS = [
    ("L1: in→256", ["net.0.weight", "net.0.bias"]),
    ("L2: 256→128", ["net.2.weight", "net.2.bias"]),
    ("L3: 128→10", ["net.4.weight", "net.4.bias"]),
]


def layer_index_ranges(model):
    """Return list of (label, start, end) slices into the flat param vector."""
    sizes = {n: p.numel() for n, p in model.named_parameters()}
    order = [n for n, _ in model.named_parameters()]
    offsets = {}
    off = 0
    for n in order:
        offsets[n] = (off, off + sizes[n])
        off += sizes[n]
    ranges = []
    for label, names in LAYER_GROUPS:
        # contiguous in flatten order; take min start, max end
        s = min(offsets[n][0] for n in names)
        e = max(offsets[n][1] for n in names)
        ranges.append((label, s, e))
    return ranges, off


def get_data(label_noise):
    tf = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize((0.1307,), (0.3081,))])
    tr = datasets.MNIST(DATA_DIR, train=True, download=True, transform=tf)
    X = tr.data.float().view(-1, 784) / 255.0
    X = (X - 0.1307) / 0.3081
    y = tr.targets.clone()
    if label_noise > 0:
        rng = np.random.RandomState(SEED)
        mask = torch.tensor(rng.random(len(y)) < label_noise)
        rand_y = torch.tensor(rng.randint(0, 10, len(y)))
        y[mask] = rand_y[mask]
    return DataLoader(TensorDataset(X, y), batch_size=BATCH, shuffle=True)


def train_and_attribute(label_noise):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(SEED); np.random.seed(SEED)
    model = MLP().to(device)
    base = torch.optim.Adam(model.parameters(), lr=LR)
    opt = WeightCovarianceFilterV2(model, base, rank=RANK, decay=DECAY, warmup=WARMUP)
    loader = get_data(label_noise)

    for ep in range(EPOCHS):
        for x, y in loader:
            opt.step(x.to(device), y.to(device))
        print(f"  noise={label_noise} epoch {ep} done")

    V = opt.V.detach().cpu().numpy()    # (p, k)
    S = opt.S.detach().cpu().numpy()    # (k,)
    ranges, p = layer_index_ranges(model)
    k = V.shape[1]

    # Per-eigenvector fraction of L2 mass in each layer
    mass = np.zeros((k, len(ranges)))
    for li, (_, s, e) in enumerate(ranges):
        mass[:, li] = (V[s:e, :] ** 2).sum(0)
    mass = mass / mass.sum(1, keepdims=True).clip(1e-30)   # rows sum to 1

    # Per-layer parameter share (uniform baseline = what a random vector would give)
    layer_param_frac = np.array([(e - s) / p for _, s, e in ranges])

    labels = [lab for lab, _, _ in ranges]
    eigvals = (S ** 2)
    # how cross-layer is each eigenvector? entropy of its layer distribution,
    # normalized so 1.0 = perfectly spread, 0 = all in one layer
    ent = -(mass * np.log(mass.clip(1e-30))).sum(1) / np.log(len(ranges))
    return dict(V_mass=mass.tolist(), eigvals=eigvals.tolist(), labels=labels,
                layer_param_frac=layer_param_frac.tolist(), spread=ent.tolist(),
                k=k, p=int(p))


def plot(res, label_noise, path):
    mass = np.array(res["V_mass"])
    k = mass.shape[0]
    labels = res["labels"]
    base = np.array(res["layer_param_frac"])
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 6),
                                  gridspec_kw={"width_ratios": [3, 1]})
    im = ax.imshow(mass.T, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("eigenvector index (0 = top / highest eigenvalue)")
    ax.set_title(f"Fraction of eigenvector L2 mass per layer (noise={label_noise})")
    fig.colorbar(im, ax=ax, fraction=0.025, label="mass fraction")
    # spread (cross-layer-ness) vs eigenvector index
    ax2.plot(res["spread"], range(k), "o-", ms=3)
    ax2.set_ylim(k - 0.5, -0.5)
    ax2.set_xlabel("layer spread\n(1=cross-layer, 0=local)")
    ax2.set_title("how cross-layer?")
    ax2.axvline(1.0, ls=":", c="gray")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f"  saved {path}")


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    out = {"config": dict(epochs=EPOCHS, rank=RANK, decay=DECAY, warmup=WARMUP,
                          lr=LR, seed=SEED)}
    for noise, tag in [(0.0, "clean"), (0.2, "noisy")]:
        print(f"=== training (noise={noise}) ===")
        res = train_and_attribute(noise)
        out[tag] = res
        plot(res, noise, os.path.join(OUTDIR, f"eigvec_layer_attribution_{tag}.png"))
        # quick text summary: top-5 vs bottom-5 mean spread
        sp = np.array(res["spread"])
        print(f"  [{tag}] mean spread top-5={sp[:5].mean():.3f} "
              f"bottom-5={sp[-5:].mean():.3f}  (param-share baseline spread="
              f"{-(np.array(res['layer_param_frac'])*np.log(np.array(res['layer_param_frac']))).sum()/np.log(3):.3f})")
    with open(os.path.join(OUTDIR, "eigvec_layer_attribution.json"), "w") as f:
        json.dump(out, f)
    print("done")


if __name__ == "__main__":
    main()
