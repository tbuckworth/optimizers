#!/usr/bin/env python3
"""Backdoor-ablation on an MLP (the version Titus asked for), with a per-sample d.

Same setup as the linear backdoor_ablation.py, now on a 784-256-128-10 MLP (can actually
memorize, so "remove the hack without losing capability" is a real test). Conditions:

  baseline         no ablation
  ablate_init      d = mean gradient over triggered images at init (static), projected out
  ablate_online    d re-estimated each epoch (EMA), projected out  (Titus's drift-tracking)
  ablate_persample d = top-n eigenvectors of the UNCENTERED per-sample gradient covariance
                   of the triggered images (re-estimated each epoch). The trigger->T mapping
                   is the direction all triggered samples SHARE -> it dominates the uncentered
                   second moment. Projected out.  (the rank-B / per-sample d idea)
  ablate_random    project out a fixed random direction (control)

Metrics: clean test accuracy, and ASR = fraction of triggered non-target test images -> T.

Usage: python3 backdoor_ablation_mlp.py --persample_n 3
"""
import argparse, json, os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call, vmap, grad
from torchvision import datasets

SEED = 0; EPOCHS = 15; BATCH = 128; LR = 1e-3; WARMUP_STEPS = 50  # overridable via --warmup
POISON_FRAC = 0.10; TARGET = 0; EMA_BETA = 0.5
DATA_DIR = "./data"; MEAN, STD = 0.1307, 0.3081
TRIG_IDX = [r * 28 + c for r in range(24, 27) for c in range(24, 27)]


def _norm(x): return (x - MEAN) / STD
def add_trigger(X):
    X = X.clone(); X[:, TRIG_IDX] = 1.0; return X


def load_raw():
    tr = datasets.MNIST(DATA_DIR, train=True, download=True)
    te = datasets.MNIST(DATA_DIR, train=False, download=True)
    return (tr.data.float().view(-1, 784) / 255.0, tr.targets.clone(),
            te.data.float().view(-1, 784) / 255.0, te.targets.clone())


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(784, 256), nn.ReLU(),
                                 nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 10))
    def forward(self, x): return self.net(x)


def make_model(device):
    torch.manual_seed(SEED)
    return MLP().to(device)


def flat_grad(model):
    return torch.cat([p.grad.reshape(-1) for p in model.parameters()])

def set_flat_grad(model, flat):
    off = 0
    for p in model.parameters():
        n = p.numel(); p.grad = flat[off:off + n].reshape(p.shape); off += n


@torch.no_grad()
def clean_acc(model, X, y):
    return (model(X).argmax(1) == y).float().mean().item()

@torch.no_grad()
def attack_success(model, Xte01, yte, device):
    mask = (yte != TARGET).to(Xte01.device)
    Xt = _norm(add_trigger(Xte01[mask])).to(device)
    return (model(Xt).argmax(1) == TARGET).float().mean().item()


def mean_trigger_dir(model, Xtrig, ytrig):
    """Mean gradient over triggered images at current weights (flattened, all params)."""
    model.zero_grad()
    F.cross_entropy(model(Xtrig), ytrig).backward()
    g = flat_grad(model).detach().clone()
    model.zero_grad()
    return g / (g.norm() + 1e-12)


def persample_trigger_dirs(model, Xtrig, ytrig, n):
    """Top-n right singular vectors of the per-sample gradient matrix of triggered images
    (= top-n eigenvectors of their UNCENTERED second moment)."""
    params = {k: v.detach() for k, v in model.named_parameters()}
    buffers = {k: v.detach() for k, v in model.named_buffers()}
    names = list(params.keys())
    def loss_fn(ps, bs, x, t):
        return F.cross_entropy(functional_call(model, (ps, bs), (x.unsqueeze(0),)), t.unsqueeze(0))
    g = vmap(grad(loss_fn), in_dims=(None, None, 0, 0))(params, buffers, Xtrig, ytrig)
    G = torch.cat([g[k].reshape(Xtrig.shape[0], -1) for k in names], dim=1)  # (B, p)
    U, S, Vh = torch.linalg.svd(G, full_matrices=False)
    dirs = Vh[:n]                                   # (n, p), orthonormal rows
    energy = (S[:n] ** 2).sum() / (S ** 2).sum()
    return dirs.contiguous(), float(energy.item())


def train(Xtr, ytr, Xtrig, ytrig, mode, device, persample_n=3, rand_dir=None):
    model = make_model(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    n = len(Xtr); step = 0
    dirs = None  # (m, p) directions to project out
    for epoch in range(EPOCHS):
        # (re)estimate ablation directions
        if mode == "ablate_init" and dirs is None:
            dirs = mean_trigger_dir(make_model(device), Xtrig, ytrig).unsqueeze(0)
        elif mode == "ablate_online":
            d = mean_trigger_dir(model, Xtrig, ytrig).unsqueeze(0)
            dirs = d if dirs is None else (EMA_BETA * dirs + (1 - EMA_BETA) * d)
            dirs = dirs / (dirs.norm(dim=1, keepdim=True) + 1e-12)
        elif mode == "ablate_persample":
            d, _ = persample_trigger_dirs(model, Xtrig, ytrig, persample_n)
            dirs = d
        elif mode == "ablate_random" and dirs is None:
            dirs = (rand_dir / rand_dir.norm()).unsqueeze(0)

        perm = torch.randperm(n, device=device)
        for i in range(0, n, BATCH):
            bi = perm[i:i + BATCH]
            opt.zero_grad()
            F.cross_entropy(model(Xtr[bi]), ytr[bi]).backward()
            step += 1
            if dirs is not None and step > WARMUP_STEPS:
                g = flat_grad(model)
                g = g - dirs.T @ (dirs @ g)        # project out all rows of `dirs`
                set_flat_grad(model, g)
            opt.step()
    return model


def main():
    global WARMUP_STEPS
    ap = argparse.ArgumentParser()
    ap.add_argument("--persample_n", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=WARMUP_STEPS)
    ap.add_argument("--sweep_m", type=str, default="",
                    help="comma list of subspace sizes for ablate_persample, e.g. 1,3,10,30,100. "
                         "If set, runs ONLY the subspace-size sweep (+ baseline + mean-grad ref).")
    ap.add_argument("--save_dir", default="../results/weight_covariance_v2/backdoor_mlp")
    args = ap.parse_args()
    WARMUP_STEPS = args.warmup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(SEED); np.random.seed(SEED)

    Xtr01, ytr, Xte01, yte = load_raw()
    rng = np.random.RandomState(SEED)
    poison = rng.rand(len(Xtr01)) < POISON_FRAC
    pm = torch.tensor(poison)
    Xtr01p = Xtr01.clone(); ytr_p = ytr.clone()
    Xtr01p[pm] = add_trigger(Xtr01p[pm]); ytr_p[pm] = TARGET
    Xtr = _norm(Xtr01p).to(device); ytr_p = ytr_p.to(device)
    Xte = _norm(Xte01).to(device); yte = yte.to(device)
    pidx = torch.tensor(np.where(poison)[0][:512])
    Xtrig, ytrig = Xtr[pidx], ytr_p[pidx]
    print(f"poison={poison.sum()} ({poison.mean()*100:.1f}%) target={TARGET} device={device}")

    # report per-sample energy concentration of the trigger direction
    _, e1 = persample_trigger_dirs(make_model(device), Xtrig, ytrig, 1)
    _, e3 = persample_trigger_dirs(make_model(device), Xtrig, ytrig, 3)
    print(f"per-sample trigger energy: top1={e1:.3f} top3={e3:.3f}")

    rand_dir = torch.randn(sum(p.numel() for p in make_model(device).parameters()), device=device)
    res = {}

    if args.sweep_m:
        # Subspace-size sweep: does ablating MORE per-sample eigenvectors ever kill the backdoor?
        ms = [int(x) for x in args.sweep_m.split(",")]
        # references
        for mode in ["baseline", "ablate_online"]:
            m = train(Xtr, ytr_p, Xtrig, ytrig, mode, device, rand_dir=rand_dir)
            ca = clean_acc(m, Xte, yte); asr = attack_success(m, Xte01, yte, device)
            res[mode] = {"clean": round(ca, 4), "asr": round(asr, 4)}
            print(f"{mode:22s} clean={ca:.4f} asr={asr:.4f}")
        sweep = {}
        for mm in ms:
            _, em = persample_trigger_dirs(make_model(device), Xtrig, ytrig, mm)
            model = train(Xtr, ytr_p, Xtrig, ytrig, "ablate_persample", device, persample_n=mm,
                          rand_dir=rand_dir)
            ca = clean_acc(model, Xte, yte); asr = attack_success(model, Xte01, yte, device)
            sweep[str(mm)] = {"clean": round(ca, 4), "asr": round(asr, 4),
                              "subspace_energy": round(em, 4)}
            print(f"persample m={mm:<4d}        clean={ca:.4f} asr={asr:.4f} energy={em:.3f}")
        res["sweep_m"] = sweep
        res["warmup"] = WARMUP_STEPS
        os.makedirs(args.save_dir, exist_ok=True)
        out_path = os.path.join(args.save_dir, "subspace_sweep.json")
        json.dump(res, open(out_path, "w"), indent=2)
        print("saved", out_path)
        return

    for mode in ["baseline", "ablate_init", "ablate_online", "ablate_persample", "ablate_random"]:
        m = train(Xtr, ytr_p, Xtrig, ytrig, mode if mode != "baseline" else "baseline",
                  device, persample_n=args.persample_n, rand_dir=rand_dir)
        ca = clean_acc(m, Xte, yte); asr = attack_success(m, Xte01, yte, device)
        res[mode] = {"clean": round(ca, 4), "asr": round(asr, 4)}
        print(f"{mode:18s} clean={ca:.4f} asr={asr:.4f}")

    res["persample_n"] = args.persample_n
    res["persample_energy_top1"] = e1; res["persample_energy_top3"] = e3
    os.makedirs(args.save_dir, exist_ok=True)
    json.dump(res, open(os.path.join(args.save_dir, "metrics.json"), "w"), indent=2)
    print("saved", os.path.join(args.save_dir, "metrics.json"))


if __name__ == "__main__":
    main()
