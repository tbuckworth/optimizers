"""Sparse-parity ADAPTIVE-rank test on Modal (A10G).

Tests the effective-rank-targeting rule (Titus's broaden-then-narrow intuition, done
faithfully this time): each step set k = round(exp(entropy of the spectrum)) instead of
a fixed rank or an energy threshold. Also a 'gap' (log-spectrum elbow) variant. Fixed
decay=0.99, rank cap=200. AdamW baseline + fixed r10 (the best fixed) for reference.

Logs live_cols (= adaptively chosen kept rank) and eff_rank each checkpoint, so we can
see whether the kept rank actually follows the broaden-then-narrow curve.

Run:   python3 -m modal run experiments/modal_parity_adaptive.py
Saves: results/parity_adaptive/*.json
"""
import os
import modal

GPU = "A10G"
HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = "/Users/titus/pyg/optimizers/results/parity_adaptive"

IMAGE = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.1", "numpy>=1.24")
    .add_local_file(os.path.join(HERE, "sparse_parity.py"), "/root/sparse_parity.py")
    .add_local_file(os.path.join(HERE, "weight_cov_optimizer_v2.py"), "/root/weight_cov_optimizer_v2.py")
)
app = modal.App("parity-adaptive", image=IMAGE)


@app.function(gpu=GPU, timeout=1800)
def run_one(cfg):
    import sys, time
    sys.path.insert(0, "/root")
    import numpy as np, torch
    import torch.nn.functional as F
    from sparse_parity import ParityMLP, make_data
    from weight_cov_optimizer_v2 import WeightCovarianceFilterV2

    mode, seed = cfg["mode"], cfg["seed"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed); np.random.seed(seed)
    Xtr, ytr, Xte, yte, idx = make_data(cfg["n"], cfg["k"], cfg["train_size"], cfg["test_size"], seed)
    Xtr, ytr, Xte, yte = Xtr.to(dev), ytr.to(dev), Xte.to(dev), yte.to(dev)
    model = ParityMLP(cfg["n"], cfg["hidden"]).to(dev)
    base = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    opt = None
    if mode == "ours":
        opt = WeightCovarianceFilterV2(model, base, rank=cfg["rank"], decay=cfg["decay"],
                                       warmup=cfg["warmup"], adaptive=cfg.get("adaptive", "none"))

    epochs, le = cfg["epochs"], cfg["log_every"]
    metrics = []
    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        if mode == "ours":
            opt.step(Xtr, ytr)
        else:
            base.zero_grad(); F.cross_entropy(model(Xtr), ytr).backward(); base.step()
        if epoch % le == 0:
            model.eval()
            with torch.no_grad():
                tr = (model(Xtr).argmax(1) == ytr).float().mean().item()
                te = (model(Xte).argmax(1) == yte).float().mean().item()
            rec = {"epoch": epoch, "train_acc": round(tr, 4), "test_acc": round(te, 4)}
            if opt is not None and opt.S is not None:
                rec["proj_k"] = int(opt.proj_k) if opt.proj_k is not None else int(len(opt.S))
                rec["basis_rank"] = int(len(opt.S))         # full tracked basis (up to cap)
                rec["eff_rank"] = round(opt._effective_rank(), 2)
            metrics.append(rec)
    grok = next((m["epoch"] for m in metrics if m["test_acc"] >= 0.9), None)
    return {"config": cfg, "metrics": metrics, "grok_epoch": grok,
            "final_test": metrics[-1]["test_acc"], "device": dev, "time_s": round(time.time() - t0, 1)}


@app.local_entrypoint()
def main():
    import json
    common = dict(n=40, k=3, train_size=2000, test_size=2000, hidden=256,
                  lr=1e-3, wd=1.0, decay=0.99, warmup=100, epochs=6000, log_every=50, rank=200)
    configs = [dict(mode="adamw", seed=s, **common) for s in (0, 1, 2)]
    configs += [dict(mode="ours", adaptive="effrank", seed=s, **common) for s in (0, 1, 2)]
    configs += [dict(mode="ours", adaptive="gap", seed=s, **common) for s in (0, 1, 2)]

    os.makedirs(OUTDIR, exist_ok=True)
    done = []
    for res in run_one.map(configs):
        c = res["config"]
        tag = "adamw" if c["mode"] == "adamw" else f"ours_{c['adaptive']}"
        name = f"{tag}_s{c['seed']}"
        json.dump(res, open(os.path.join(OUTDIR, name + ".json"), "w"))
        krs = [m.get("proj_k") for m in res["metrics"] if m.get("proj_k")]
        krrng = f"{min(krs)}-{max(krs)}" if krs else "-"
        print(f"{name:18s} grok@{str(res['grok_epoch']):>5}  final={res['final_test']:.3f}  "
              f"proj_k={krrng}  ({res['device']}, {res['time_s']}s)")
        done.append(name)
    print(f"ALL SAVED ({len(done)} runs) -> {OUTDIR}")
