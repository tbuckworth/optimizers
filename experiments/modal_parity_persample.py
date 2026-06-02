"""Parity: per-sample (rank-B, uncentered, project-top-k) filter vs batch-mean filter vs AdamW.

Tests Titus's question: does sourcing the top-k projection from the PER-SAMPLE gradient
covariance (across examples) — rather than the batch-mean gradient over steps — change
grokking? Plus an adaptive-rank arm (keep X% of eigenvalue energy) on both filters.

Run:   python3 -m modal run experiments/modal_parity_persample.py
Saves: results/parity_persample/*.json (written locally by the entrypoint)
"""
import os
import modal

GPU = "A10G"
HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = "/Users/titus/pyg/optimizers/results/parity_persample"

IMAGE = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.1", "numpy>=1.24")
    .add_local_file(os.path.join(HERE, "sparse_parity.py"), "/root/sparse_parity.py")
    .add_local_file(os.path.join(HERE, "weight_cov_optimizer_v2.py"), "/root/weight_cov_optimizer_v2.py")
    .add_local_file(os.path.join(HERE, "persample_cov_optimizer.py"), "/root/persample_cov_optimizer.py")
)
app = modal.App("parity-persample", image=IMAGE)


@app.function(gpu=GPU, timeout=2400)
def run_one(cfg):
    import sys, time
    sys.path.insert(0, "/root")
    import numpy as np, torch
    import torch.nn.functional as F
    from sparse_parity import ParityMLP, make_data
    from weight_cov_optimizer_v2 import WeightCovarianceFilterV2
    from persample_cov_optimizer import PerSampleCovarianceFilter

    mode, seed = cfg["mode"], cfg["seed"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed); np.random.seed(seed)
    Xtr, ytr, Xte, yte, idx = make_data(cfg["n"], cfg["k"], cfg["train_size"], cfg["test_size"], seed)
    Xtr, ytr, Xte, yte = Xtr.to(dev), ytr.to(dev), Xte.to(dev), yte.to(dev)
    model = ParityMLP(cfg["n"], cfg["hidden"]).to(dev)
    base = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])

    et = cfg.get("energy_threshold")
    filt = None
    if mode == "v2":
        filt = WeightCovarianceFilterV2(model, base, rank=cfg["rank"], decay=cfg["decay"],
                                        warmup=cfg["warmup"], energy_threshold=et)
    elif mode == "persample":
        filt = PerSampleCovarianceFilter(model, base, rank=cfg["rank"], decay=cfg["decay"],
                                         warmup=cfg["warmup"], cov_batch=cfg["cov_batch"],
                                         energy_threshold=et)

    epochs, le = cfg["epochs"], cfg["log_every"]
    metrics = []; t0 = time.time()
    for epoch in range(epochs):
        model.train()
        if mode == "adamw":
            base.zero_grad(); F.cross_entropy(model(Xtr), ytr).backward(); base.step()
        elif mode == "v2":
            filt.step(Xtr, ytr)
        else:
            filt.step(Xtr, ytr)
        if epoch % le == 0:
            model.eval()
            with torch.no_grad():
                tr = (model(Xtr).argmax(1) == ytr).float().mean().item()
                te = (model(Xte).argmax(1) == yte).float().mean().item()
            rec = {"epoch": epoch, "train_acc": round(tr, 4), "test_acc": round(te, 4)}
            if filt is not None:
                er = filt.effective_rank() if hasattr(filt, "effective_rank") else None
                if er is None and getattr(filt, "S", None) is not None:
                    er = filt._effective_rank()
                if er is not None:
                    rec["eff_rank"] = round(er, 2)
                    rec["live_cols"] = int(len(filt.S)) if filt.S is not None else 0
            metrics.append(rec)
    grok = next((m["epoch"] for m in metrics if m["test_acc"] >= 0.9), None)
    return {"config": cfg, "metrics": metrics, "grok_epoch": grok,
            "final_test": metrics[-1]["test_acc"], "device": dev, "time_s": round(time.time() - t0, 1)}


@app.local_entrypoint()
def main():
    import json
    common = dict(n=40, k=3, train_size=2000, test_size=2000, hidden=256,
                  lr=1e-3, wd=1.0, decay=0.99, warmup=100, epochs=6000, log_every=50,
                  cov_batch=256)
    configs = []
    for s in (0, 1, 2):
        configs.append(dict(mode="adamw", rank=0, **common, seed=s))
    # adaptive-rank on the batch-mean (V2) filter — Titus's energy-threshold idea
    for s in (0, 1, 2):
        configs.append(dict(mode="v2", rank=200, energy_threshold=0.99, **common, seed=s))
    # per-sample filter, rank sweep
    for r in (3, 10, 50):
        for s in (0, 1, 2):
            configs.append(dict(mode="persample", rank=r, **common, seed=s))
    # per-sample adaptive
    for s in (0, 1, 2):
        configs.append(dict(mode="persample", rank=200, energy_threshold=0.99, **common, seed=s))

    os.makedirs(OUTDIR, exist_ok=True)
    for res in run_one.map(configs):
        c = res["config"]
        tag = (f"adamw" if c["mode"] == "adamw"
               else f"{c['mode']}_e99" if c.get("energy_threshold")
               else f"{c['mode']}_r{c['rank']}")
        name = f"{tag}_s{c['seed']}"
        json.dump(res, open(os.path.join(OUTDIR, name + ".json"), "w"))
        print(f"{name:20s} grok@{str(res['grok_epoch']):>5}  final={res['final_test']:.3f}  ({res['time_s']}s)")
    print(f"ALL SAVED -> {OUTDIR}")
