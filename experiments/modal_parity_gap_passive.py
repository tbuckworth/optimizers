"""Two parity follow-ups on Modal (A10G):

1. GAP FILL: rank {4,5,6,7,8} at fixed λ=0.99, 3 seeds — there's a big gap in the
   rank sweep between r3 (never groks) and r10 (~1250); do 4-8 grok, and faster?
   Saved to results/sparse_parity_ranksweep/ (same naming as the original sweep).

2. PASSIVE AdamW: track the gradient covariance along a PURE AdamW trajectory but
   DO NOT filter with it (filter_strength=0). Logs the effective rank (active
   gradient directions) over time — does the un-filtered baseline broaden-then-
   narrow on its own? Saved to results/parity_passive/.

Run:   python3 -m modal run experiments/modal_parity_gap_passive.py
"""
import os
import modal

GPU = "A10G"
HERE = os.path.dirname(os.path.abspath(__file__))
RANKDIR = "/Users/titus/pyg/optimizers/results/sparse_parity_ranksweep"
PASSDIR = "/Users/titus/pyg/optimizers/results/parity_passive"

IMAGE = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.1", "numpy>=1.24")
    .add_local_file(os.path.join(HERE, "sparse_parity.py"), "/root/sparse_parity.py")
    .add_local_file(os.path.join(HERE, "weight_cov_optimizer_v2.py"), "/root/weight_cov_optimizer_v2.py")
)
app = modal.App("parity-gap-passive", image=IMAGE)


@app.function(gpu=GPU, timeout=1800)
def run_one(cfg):
    import sys, time
    sys.path.insert(0, "/root")
    import numpy as np, torch
    import torch.nn.functional as F
    from sparse_parity import ParityMLP, make_data
    from weight_cov_optimizer_v2 import WeightCovarianceFilterV2

    seed = cfg["seed"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed); np.random.seed(seed)
    Xtr, ytr, Xte, yte, idx = make_data(cfg["n"], cfg["k"], cfg["train_size"], cfg["test_size"], seed)
    Xtr, ytr, Xte, yte = Xtr.to(dev), ytr.to(dev), Xte.to(dev), yte.to(dev)
    model = ParityMLP(cfg["n"], cfg["hidden"]).to(dev)
    base = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    # filter_strength=0 -> passive: covariance is tracked but the gradient is NOT changed
    opt = WeightCovarianceFilterV2(model, base, rank=cfg["rank"], decay=cfg["decay"],
                                   warmup=cfg["warmup"], filter_strength=cfg["filter_strength"])

    epochs, le = cfg["epochs"], cfg["log_every"]
    metrics = []
    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        opt.step(Xtr, ytr)
        if epoch % le == 0:
            model.eval()
            with torch.no_grad():
                tr = (model(Xtr).argmax(1) == ytr).float().mean().item()
                te = (model(Xte).argmax(1) == yte).float().mean().item()
            rec = {"epoch": epoch, "train_acc": round(tr, 4), "test_acc": round(te, 4)}
            if opt.S is not None:
                rec["eff_rank"] = round(opt._effective_rank(), 2)
                rec["live_cols"] = int(len(opt.S))
            metrics.append(rec)
    grok = next((m["epoch"] for m in metrics if m["test_acc"] >= 0.9), None)
    return {"config": cfg, "metrics": metrics, "grok_epoch": grok,
            "final_test": metrics[-1]["test_acc"], "device": dev, "time_s": round(time.time()-t0, 1)}


@app.local_entrypoint()
def main():
    import json
    common = dict(n=40, k=3, train_size=2000, test_size=2000, hidden=256,
                  lr=1e-3, wd=1.0, decay=0.99, warmup=100, epochs=6000, log_every=50)
    gap = [dict(rank=r, seed=s, filter_strength=1.0, **common)
           for r in (4, 5, 6, 7, 8) for s in (0, 1, 2)]
    passive = [dict(rank=200, seed=s, filter_strength=0.0, **common) for s in (0, 1, 2)]

    os.makedirs(RANKDIR, exist_ok=True); os.makedirs(PASSDIR, exist_ok=True)
    configs = gap + passive
    done = []
    for res in run_one.map(configs):
        c = res["config"]
        if c["filter_strength"] == 0.0:
            outdir, name = PASSDIR, f"adamw_passive_s{c['seed']}"
        else:
            outdir, name = RANKDIR, f"ours_r{c['rank']}_s{c['seed']}"
        json.dump(res, open(os.path.join(outdir, name + ".json"), "w"))
        print(f"{name:20s} grok@{str(res['grok_epoch']):>5}  final={res['final_test']:.3f}  ({res['time_s']}s)")
        done.append(name)
    print(f"ALL SAVED ({len(done)} runs)")
