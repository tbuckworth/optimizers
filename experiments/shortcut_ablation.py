#!/usr/bin/env python3
"""Targeted eigenvector ablation on a planted-shortcut MNIST task.

A controllable proxy for "find and remove the reward-hack direction" where we
KNOW the ground truth. We plant a class-indicating shortcut pixel into training
images. A model that "reward hacks" leans on the shortcut and fails when the
shortcut is misleading; a robust model ignores it.

Pipeline:
  1. Plant a per-class shortcut pixel (10 distinct top-edge positions) in TRAIN
     (pixel position encodes the TRUE label).
  2. Two test sets: CLEAN (no shortcut) and SPURIOUS (shortcut points to a RANDOM
     label -> actively misleading). Shortcut-reliant models tank on SPURIOUS.
  3. Train a linear softmax classifier, collecting gradients -> exact gradient
     covariance -> eigenbasis.
  4. Identify the shortcut direction via a CONTRASTIVE probe: mean of
     grad(image+shortcut) - grad(image_clean). Find which covariance eigenvector
     it aligns with. Validate against the KNOWN planted-pixel weights.
  5. Retrain from scratch under 3 conditions and compare CLEAN/SPURIOUS test:
       - baseline        (no ablation)
       - ablate_shortcut (project out the identified shortcut eigenvector each step)
       - ablate_random   (control: project out a random direction)
  6. Visualize the identified shortcut direction as an image (should light up the
     10 planted pixel positions).

Output: results/weight_covariance_v2/shortcut_ablation/{figure.png, metrics.json}
"""

import os, json
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 0
EPOCHS = 6
BATCH = 128
LR = 1e-3
WARMUP_STEPS = 20
DATA_DIR = "./data"
OUTDIR = "../results/weight_covariance_v2/shortcut_ablation"

# 10 distinct shortcut pixel positions in the top two rows (normally background)
POSITIONS = [(0, 1 + 3 * c) for c in range(10)]            # (row, col) per class
POS_IDX = [r * 28 + col for (r, col) in POSITIONS]          # flat pixel index per class
MEAN, STD = 0.1307, 0.3081


def _normalize(x01):
    return (x01 - MEAN) / STD


def load_raw():
    tr = datasets.MNIST(DATA_DIR, train=True, download=True)
    te = datasets.MNIST(DATA_DIR, train=False, download=True)
    Xtr = tr.data.float().view(-1, 784) / 255.0
    ytr = tr.targets.clone()
    Xte = te.data.float().view(-1, 784) / 255.0
    yte = te.targets.clone()
    return Xtr, ytr, Xte, yte


def plant(X01, ref_labels):
    """Set the shortcut pixel (white) at the position indexed by ref_labels. Returns a copy."""
    X = X01.clone()
    idx = torch.tensor([POS_IDX[int(l)] for l in ref_labels])
    X[torch.arange(len(X)), idx] = 1.0
    return X


def evaluate(W, b, X, y):
    with torch.no_grad():
        logits = X @ W.t() + b
        return (logits.argmax(1) == y).float().mean().item()


def make_model():
    g = torch.Generator().manual_seed(SEED)
    W = torch.zeros(10, 784)
    torch.nn.init.kaiming_uniform_(W, a=5 ** 0.5, generator=g)
    return W.clone().requires_grad_(True), torch.zeros(10, requires_grad=True)


def train(Xtr, ytr, evals_sets, ablate_dir=None, collect=False):
    """Train linear model. ablate_dir: (7840,) unit vector over W to project out of grad.
    evals_sets: dict name->(X,y). Returns (W,b, history, grads)."""
    torch.manual_seed(SEED); np.random.seed(SEED)
    W, b = make_model()
    opt = torch.optim.Adam([W, b], lr=LR)
    if ablate_dir is not None:
        ad = torch.as_tensor(ablate_dir, dtype=torch.float32)
        ad = ad / ad.norm()
    grads = []
    n = len(Xtr); step = 0
    for epoch in range(EPOCHS):
        perm = torch.randperm(n)
        for i in range(0, n, BATCH):
            bi = perm[i:i + BATCH]
            xb, yb = Xtr[bi], ytr[bi]
            opt.zero_grad()
            loss = F.cross_entropy(xb @ W.t() + b, yb)
            loss.backward()
            step += 1
            if collect and step > WARMUP_STEPS:
                grads.append(W.grad.detach().reshape(-1).clone())
            if ablate_dir is not None:
                g = W.grad.detach().reshape(-1)
                g -= (ad @ g) * ad            # project out the direction
                W.grad = g.reshape(10, 784)
            opt.step()
    hist = {name: evaluate(W, b, X, y) for name, (X, y) in evals_sets.items()}
    return W, b, hist, grads


def main():
    torch.manual_seed(SEED); np.random.seed(SEED)
    Xtr01, ytr, Xte01, yte = load_raw()
    rng = np.random.RandomState(SEED)

    # Datasets
    Xtr = _normalize(plant(Xtr01, ytr))                       # train: shortcut = true label
    spurious_ref = torch.tensor(rng.randint(0, 10, len(yte))) # misleading shortcut at test
    Xte_clean = _normalize(Xte01)
    Xte_spur = _normalize(plant(Xte01, spurious_ref))
    evsets = {"clean": (Xte_clean, yte), "spurious": (Xte_spur, yte)}

    # 1) Baseline training + gradient collection
    W0, b0, hist_base, grads = train(Xtr, ytr, evsets, ablate_dir=None, collect=True)
    G = torch.stack(grads).numpy()
    Gc = G - G.mean(0, keepdims=True)
    cov = Gc.T @ Gc / Gc.shape[0]
    evals_, evecs = np.linalg.eigh(cov)
    evals_ = evals_[::-1]; evecs = evecs[:, ::-1]

    # 2) Contrastive probe to isolate the shortcut direction
    probe_idx = torch.randperm(len(Xtr01))[:512]
    xp01, yp = Xtr01[probe_idx], ytr[probe_idx]
    Wp, bp = W0.detach().clone().requires_grad_(True), b0.detach().clone().requires_grad_(True)

    def probe_grad(Xnorm):
        Wp.grad = None
        loss = F.cross_entropy(Xnorm @ Wp.t() + bp, yp)
        loss.backward()
        return Wp.grad.detach().reshape(-1).clone()

    g_with = probe_grad(_normalize(plant(xp01, yp)))
    g_without = probe_grad(_normalize(xp01))
    g_shortcut = (g_with - g_without).numpy()
    g_shortcut /= np.linalg.norm(g_shortcut)

    # Known ground-truth shortcut subspace: weights touching the 10 planted pixels
    known = np.zeros((10, 784))
    for c in range(10):
        known[c, POS_IDX[c]] = 1.0           # the weight from class-c's pixel to class c
    known = known.reshape(-1); known /= np.linalg.norm(known)

    # Which eigenvector aligns with the shortcut?
    cos_evec = np.abs(evecs.T @ g_shortcut)               # (p,)
    top_eig = int(np.argmax(cos_evec))
    v_hack = evecs[:, top_eig]
    cos_probe_known = float(np.abs(known @ g_shortcut))
    cos_vhack_known = float(np.abs(known @ v_hack))
    cos_vhack_probe = float(np.abs(v_hack @ g_shortcut))

    # Use the identified eigenvector as the ablation direction
    rand_dir = rng.randn(7840); rand_dir /= np.linalg.norm(rand_dir)

    # 3) Retrain under conditions
    _, _, hist_ab, _ = train(Xtr, ytr, evsets, ablate_dir=v_hack)
    _, _, hist_rnd, _ = train(Xtr, ytr, evsets, ablate_dir=rand_dir)

    results = {
        "baseline": hist_base, "ablate_shortcut": hist_ab, "ablate_random": hist_rnd,
        "top_eig_index": top_eig, "top_eig_alignment": float(cos_evec[top_eig]),
        "cos(probe, known)": cos_probe_known,
        "cos(v_hack, known)": cos_vhack_known,
        "cos(v_hack, probe)": cos_vhack_probe,
        "top5_eig_alignments": [round(float(x), 3) for x in np.sort(cos_evec)[::-1][:5]],
        "positions": POSITIONS,
    }
    print(json.dumps(results, indent=2))

    # 4) Figure
    def pix(vec):
        return np.linalg.norm(vec.reshape(10, 784), axis=0).reshape(28, 28)

    fig = plt.figure(figsize=(13, 7), facecolor="#0f1117")
    gs = fig.add_gridspec(2, 3, hspace=0.32, wspace=0.25, height_ratios=[1, 1])

    # bar chart: clean vs spurious per condition
    ax = fig.add_subplot(gs[0, :])
    conds = ["baseline", "ablate_shortcut", "ablate_random"]
    hd = {"baseline": hist_base, "ablate_shortcut": hist_ab, "ablate_random": hist_rnd}
    x = np.arange(len(conds)); w = 0.36
    ax.bar(x - w/2, [hd[c]["clean"] for c in conds], w, label="clean test", color="#37c87a")
    ax.bar(x + w/2, [hd[c]["spurious"] for c in conds], w, label="spurious (misleading) test", color="#ff5d6c")
    ax.set_xticks(x); ax.set_xticklabels(conds, color="#e6e9ef")
    ax.set_ylim(0, 1); ax.set_ylabel("accuracy", color="#9aa3b2")
    ax.set_title("Does ablating the shortcut direction restore robustness to a misleading shortcut?",
                 color="#e6e9ef", fontsize=12)
    ax.tick_params(colors="#9aa3b2"); ax.set_facecolor("#181b24")
    for s in ax.spines.values(): s.set_color("#262b38")
    ax.legend(facecolor="#181b24", labelcolor="#e6e9ef", edgecolor="#262b38")
    for xi, c in enumerate(conds):
        ax.text(xi - w/2, hd[c]["clean"] + .02, f"{hd[c]['clean']*100:.0f}", ha="center", color="#37c87a", fontsize=9)
        ax.text(xi + w/2, hd[c]["spurious"] + .02, f"{hd[c]['spurious']*100:.0f}", ha="center", color="#ff5d6c", fontsize=9)

    for col, (img, title) in enumerate([
        (pix(g_shortcut), "contrastive probe\n(isolated shortcut)"),
        (pix(v_hack), f"identified eigenvector #{top_eig+1}\n(ablation target)"),
        (pix(known), "known planted pixels\n(ground truth)")]):
        axi = fig.add_subplot(gs[1, col])
        axi.imshow(img, cmap="magma"); axi.set_xticks([]); axi.set_yticks([])
        axi.set_title(title, color="#e6e9ef", fontsize=10)

    fig.suptitle("Targeted eigenvector ablation on planted-shortcut MNIST", color="#e6e9ef", fontsize=14, y=0.99)
    os.makedirs(OUTDIR, exist_ok=True)
    figpath = os.path.join(OUTDIR, "shortcut_ablation.png")
    fig.savefig(figpath, dpi=130, bbox_inches="tight", facecolor="#0f1117")
    with open(os.path.join(OUTDIR, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {figpath}")


if __name__ == "__main__":
    main()
