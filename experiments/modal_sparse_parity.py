"""Sparse-parity low-rank sweep on Modal (A10G).

Tests Titus's hypothesis (reply #3): with (n=40,k=3) sparse parity there are ~37 noise
dims and 3 signal dims, so a LOW-rank focus should be STRONGER (help grokking), not
weaker. Sweeps rank in {1,3,10,50,200} at FIXED decay=0.99 (low rank is below the
~100 eff-rank ceiling, so no decay change is needed/warranted), vs an AdamW baseline.

Also logs:
  - train AND test curves (B3)
  - effective rank + live column count each checkpoint
  - the full eigenvalue spectrum at several epochs (B2 — is there a drop-off at ~3?)

Run:   python3 -m modal run experiments/modal_sparse_parity.py
Saves: results/sparse_parity_ranksweep/*.json  (written locally by the entrypoint)
"""
import os
import modal

GPU = "A10G"
HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = "/Users/titus/pyg/optimizers/results/sparse_parity_ranksweep"

IMAGE = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.1", "numpy>=1.24")
    .add_local_file(os.path.join(HERE, "sparse_parity.py"), "/root/sparse_parity.py")
    .add_local_file(os.path.join(HERE, "weight_cov_optimizer_v2.py"), "/root/weight_cov_optimizer_v2.py")
)

app = modal.App("sparse-parity-ranksweep", image=IMAGE)


@app.function(gpu=GPU, timeout=1800)
def run_one(cfg):
    import sys, time
    sys.path.insert(0, "/root")
    import numpy as np, torch
    import torch.nn.functional as F
    from sparse_parity import ParityMLP, make_data
    from weight_cov_optimizer_v2 import WeightCovarianceFilterV2

    mode, rank, seed = cfg["mode"], cfg["rank"], cfg["seed"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed); np.random.seed(seed)
    Xtr, ytr, Xte, yte, idx = make_data(cfg["n"], cfg["k"], cfg["train_size"], cfg["test_size"], seed)
    Xtr, ytr, Xte, yte = Xtr.to(dev), ytr.to(dev), Xte.to(dev), yte.to(dev)
    model = ParityMLP(cfg["n"], cfg["hidden"]).to(dev)
    base = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    opt = (WeightCovarianceFilterV2(model, base, rank=rank, decay=cfg["decay"], warmup=cfg["warmup"])
           if mode == "ours" else None)

    epochs, le, wu = cfg["epochs"], cfg["log_every"], cfg["warmup"]
    spectrum_at = {wu + 2, 200, 500, 1000, 2000, 4000, epochs - 1}
    metrics, spectra = [], []
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
                rec["eff_rank"] = round(opt._effective_rank(), 2)
                rec["live_cols"] = int(len(opt.S))
            metrics.append(rec)
        if opt is not None and epoch in spectrum_at and opt.S is not None:
            ev = (opt.S ** 2).tolist()
            spectra.append({"epoch": epoch, "eigenvalues": [round(float(x), 8) for x in ev[:200]]})
    grok = next((m["epoch"] for m in metrics if m["test_acc"] >= 0.9), None)
    return {"config": cfg, "metrics": metrics, "spectra": spectra, "grok_epoch": grok,
            "relevant_bits": idx.tolist(), "final_test": metrics[-1]["test_acc"],
            "device": dev, "time_s": round(time.time() - t0, 1)}


@app.local_entrypoint()
def main():
    import json
    common = dict(n=40, k=3, train_size=2000, test_size=2000, hidden=256,
                  lr=1e-3, wd=1.0, decay=0.99, warmup=100, epochs=6000, log_every=50)
    configs = [dict(mode="adamw", rank=200, seed=s, **common) for s in (0, 1, 2)]
    for r in (1, 3, 10, 50, 200):
        for s in (0, 1, 2):
            configs.append(dict(mode="ours", rank=r, seed=s, **common))

    os.makedirs(OUTDIR, exist_ok=True)
    done = []
    for res in run_one.map(configs):
        c = res["config"]
        name = f"adamw_s{c['seed']}" if c["mode"] == "adamw" else f"ours_r{c['rank']}_s{c['seed']}"
        json.dump(res, open(os.path.join(OUTDIR, name + ".json"), "w"))
        print(f"{name:16s} grok@{str(res['grok_epoch']):>5}  final={res['final_test']:.3f}  ({res['device']}, {res['time_s']}s)")
        done.append(name)
    print(f"ALL SAVED ({len(done)} runs) -> {OUTDIR}")
