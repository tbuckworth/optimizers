#!/usr/bin/env python3
"""Targeted gradient-direction ablation to PREVENT a backdoor being learned.

A faithful, controllable proxy for "find and remove the reward-hack direction"
where we KNOW the ground truth. A fraction of training images get a TRIGGER
patch and their label flipped to a fixed target T. Normally the model learns the
backdoor (trigger -> T) on top of the clean task. The trigger is the ONLY thing
triggered examples share (their digit content is random), so the MEAN of their
gradients isolates the backdoor direction by consensus -- exactly our mechanism.

We ablate that direction from the gradient during training and ask: can we
prevent the backdoor (low attack success) while keeping clean accuracy?

Key fix vs first attempt: the backdoor direction must be estimated WHILE the
gradient still points toward learning it (at init / online), not at convergence
where grad~0. We track it ONLINE (EMA, re-estimated each epoch) to follow drift
-- which is the user's actual proposal.

Conditions:
  baseline       no ablation
  ablate_init    project out the trigger direction estimated at init (static)
  ablate_online  re-estimate the trigger direction each epoch (EMA), ablate it
  ablate_eig     project out the covariance eigenvector most aligned with it
  ablate_random  control

Output: results/weight_covariance_v2/backdoor_ablation/{figure.png, metrics.json}
"""

import os, json
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 0
EPOCHS = 6
BATCH = 128
LR = 1e-3
WARMUP_STEPS = 20
POISON_FRAC = 0.10
TARGET = 0
EMA_BETA = 0.5
DATA_DIR = "./data"
OUTDIR = "../results/weight_covariance_v2/backdoor_ablation"
MEAN, STD = 0.1307, 0.3081

TRIG_RC = [(r, c) for r in range(24, 27) for c in range(24, 27)]   # 3x3 corner trigger
TRIG_IDX = [r * 28 + c for (r, c) in TRIG_RC]


def _norm(x01):
    return (x01 - MEAN) / STD


def add_trigger(X01):
    X = X01.clone(); X[:, TRIG_IDX] = 1.0
    return X


def load_raw():
    tr = datasets.MNIST(DATA_DIR, train=True, download=True)
    te = datasets.MNIST(DATA_DIR, train=False, download=True)
    return (tr.data.float().view(-1, 784) / 255.0, tr.targets.clone(),
            te.data.float().view(-1, 784) / 255.0, te.targets.clone())


def make_model():
    g = torch.Generator().manual_seed(SEED)
    W = torch.zeros(10, 784)
    torch.nn.init.kaiming_uniform_(W, a=5 ** 0.5, generator=g)
    return W.clone().requires_grad_(True), torch.zeros(10, requires_grad=True)


def clean_acc(W, b, X, y):
    with torch.no_grad():
        return ((X @ W.t() + b).argmax(1) == y).float().mean().item()


def attack_success(W, b, Xte01, yte):
    mask = yte != TARGET
    Xt = _norm(add_trigger(Xte01[mask]))
    with torch.no_grad():
        return ((Xt @ W.t() + b).argmax(1) == TARGET).float().mean().item()


def backdoor_dir_at(W, b, Xtrig, ytrig):
    """Mean gradient over triggered (label=T) images at the given weights -> the
    direction that learning the backdoor pushes. Normalized, over W only."""
    Wd = W.detach().clone().requires_grad_(True)
    bd = b.detach().clone().requires_grad_(True)
    F.cross_entropy(Xtrig @ Wd.t() + bd, ytrig).backward()
    g = Wd.grad.detach().reshape(-1).numpy()
    nrm = np.linalg.norm(g)
    return g / nrm if nrm > 1e-12 else g


def train(Xtr, ytr, ablate_dir=None, online=None, collect=False):
    """online: dict(Xtrig,ytrig) -> re-estimate & EMA the ablation dir each epoch."""
    torch.manual_seed(SEED); np.random.seed(SEED)
    W, b = make_model()
    opt = torch.optim.Adam([W, b], lr=LR)
    ad = None
    if ablate_dir is not None:
        ad = torch.as_tensor(ablate_dir, dtype=torch.float32); ad = ad / ad.norm()
    ema = None
    grads = []; n = len(Xtr); step = 0
    for epoch in range(EPOCHS):
        if online is not None:
            d = backdoor_dir_at(W, b, online["X"], online["y"])
            ema = d if ema is None else EMA_BETA * ema + (1 - EMA_BETA) * d
            t = torch.as_tensor(ema, dtype=torch.float32); ad = t / t.norm()
        perm = torch.randperm(n)
        for i in range(0, n, BATCH):
            bi = perm[i:i + BATCH]
            opt.zero_grad()
            F.cross_entropy(Xtr[bi] @ W.t() + b, ytr[bi]).backward()
            step += 1
            if collect and step > WARMUP_STEPS:
                grads.append(W.grad.detach().reshape(-1).clone())
            if ad is not None:
                g = W.grad.detach().reshape(-1)
                g -= (ad @ g) * ad
                W.grad = g.reshape(10, 784)
            opt.step()
    return W, b, grads


def main():
    torch.manual_seed(SEED); np.random.seed(SEED)
    Xtr01, ytr, Xte01, yte = load_raw()
    rng = np.random.RandomState(SEED)

    n = len(Xtr01)
    poison = rng.rand(n) < POISON_FRAC
    pm = torch.tensor(poison)
    Xtr01p = Xtr01.clone(); ytr_p = ytr.clone()
    Xtr01p[pm] = add_trigger(Xtr01p[pm]); ytr_p[pm] = TARGET
    Xtr = _norm(Xtr01p)
    print(f"Poisoned {poison.sum()} / {n} ({poison.mean()*100:.1f}%) -> target {TARGET}")

    # Fixed triggered probe batch (all labeled TARGET)
    pidx = torch.tensor(np.where(poison)[0][:1024])
    Xtrig, ytrig = Xtr[pidx], ytr_p[pidx]

    # 1) Baseline + gradient collection for covariance
    W0, b0, grads = train(Xtr, ytr_p, collect=True)
    base = {"clean": clean_acc(W0, b0, _norm(Xte01), yte), "asr": attack_success(W0, b0, Xte01, yte)}

    G = torch.stack(grads).numpy(); Gc = G - G.mean(0, keepdims=True)
    cov = Gc.T @ Gc / Gc.shape[0]
    evals_, evecs = np.linalg.eigh(cov); evals_ = evals_[::-1]; evecs = evecs[:, ::-1]

    # 2) Backdoor direction estimated at INIT (strong, pure trigger signal)
    Wi, bi = make_model()
    g_bd = backdoor_dir_at(Wi, bi, Xtrig, ytrig)

    known = np.zeros((10, 784)); known[:, TRIG_IDX] = 1.0
    known = known.reshape(-1); known /= np.linalg.norm(known)
    trig_mass = float(np.linalg.norm(g_bd.reshape(10, 784)[:, TRIG_IDX]) ** 2)  # frac of energy on trigger cols

    cos_evec = np.abs(evecs.T @ g_bd); top_eig = int(np.argmax(cos_evec)); v_eig = evecs[:, top_eig]
    ident = {
        "cos(init_probe, known_trigger)": round(float(abs(known @ g_bd)), 3),
        "frac_probe_energy_on_trigger_cols": round(trig_mass, 3),
        "top_eig_index": top_eig,
        "top_eig_alignment": round(float(cos_evec[top_eig]), 3),
        "top5_eig_alignments": [round(float(x), 3) for x in np.sort(cos_evec)[::-1][:5]],
        "energy_in_top10_eig": round(float((cos_evec[:10] ** 2).sum()), 3),
        "cos(v_eig, known_trigger)": round(float(abs(known @ v_eig)), 3),
    }

    # 3) Conditions
    rand_dir = rng.randn(7840); rand_dir /= np.linalg.norm(rand_dir)
    res = {"baseline": base}
    for name, kw in [("ablate_init", dict(ablate_dir=g_bd)),
                     ("ablate_online", dict(online={"X": Xtrig, "y": ytrig})),
                     ("ablate_eig", dict(ablate_dir=v_eig)),
                     ("ablate_random", dict(ablate_dir=rand_dir))]:
        W, b, _ = train(Xtr, ytr_p, **kw)
        res[name] = {"clean": clean_acc(W, b, _norm(Xte01), yte), "asr": attack_success(W, b, Xte01, yte)}
    res.update(ident); res["poison_frac"] = float(poison.mean()); res["target"] = TARGET
    print(json.dumps(res, indent=2))

    # 4) Figure
    def pix(v):
        return np.linalg.norm(v.reshape(10, 784), axis=0).reshape(28, 28)

    fig = plt.figure(figsize=(13, 7), facecolor="#0f1117")
    gs = fig.add_gridspec(2, 3, hspace=0.34, wspace=0.25)
    ax = fig.add_subplot(gs[0, :])
    order = ["baseline", "ablate_init", "ablate_online", "ablate_eig", "ablate_random"]
    x = np.arange(len(order)); w = 0.38
    ax.bar(x - w/2, [res[c]["clean"] for c in order], w, label="clean accuracy", color="#37c87a")
    ax.bar(x + w/2, [res[c]["asr"] for c in order], w, label="attack success (backdoor)", color="#ff5d6c")
    ax.set_xticks(x); ax.set_xticklabels(order, color="#e6e9ef", fontsize=9); ax.set_ylim(0, 1.08)
    ax.set_ylabel("rate", color="#9aa3b2"); ax.tick_params(colors="#9aa3b2"); ax.set_facecolor("#181b24")
    ax.set_title("Ablating the backdoor direction: prevent the attack, keep clean accuracy?",
                 color="#e6e9ef", fontsize=12)
    for s in ax.spines.values(): s.set_color("#262b38")
    ax.legend(facecolor="#181b24", labelcolor="#e6e9ef", edgecolor="#262b38")
    for xi, c in enumerate(order):
        ax.text(xi - w/2, res[c]["clean"] + .02, f"{res[c]['clean']*100:.0f}", ha="center", color="#37c87a", fontsize=8)
        ax.text(xi + w/2, res[c]["asr"] + .02, f"{res[c]['asr']*100:.0f}", ha="center", color="#ff5d6c", fontsize=8)

    for col, (img, title) in enumerate([
        (pix(g_bd), f"init backdoor direction\n(trigger energy {trig_mass:.2f})"),
        (pix(v_eig), f"top eigenvector #{top_eig+1}\n(align {cos_evec[top_eig]:.2f})"),
        (pix(known), "known trigger pixels\n(ground truth)")]):
        axi = fig.add_subplot(gs[1, col]); axi.imshow(img, cmap="magma")
        axi.set_xticks([]); axi.set_yticks([]); axi.set_title(title, color="#e6e9ef", fontsize=10)

    fig.suptitle("Preventing a backdoor by ablating its gradient direction (linear MNIST)",
                 color="#e6e9ef", fontsize=14, y=0.99)
    os.makedirs(OUTDIR, exist_ok=True)
    figpath = os.path.join(OUTDIR, "backdoor_ablation.png")
    fig.savefig(figpath, dpi=130, bbox_inches="tight", facecolor="#0f1117")
    with open(os.path.join(OUTDIR, "metrics.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(f"Saved {figpath}")


if __name__ == "__main__":
    main()
