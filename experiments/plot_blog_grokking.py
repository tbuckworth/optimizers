#!/usr/bin/env python3
"""Modular-addition grokking: filter accelerates the test-accuracy onset (5 seeds)."""
import json, glob, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = "/Users/titus/pyg/optimizers/results/grokking_v2_seeds"
SEEDS = [42, 43, 44, 45, 46]
def load(prefix):
    runs = []
    for s in SEEDS:
        p = os.path.join(D, f"{prefix}_s{s}", "metrics.json")
        if os.path.exists(p):
            runs.append(json.load(open(p))["metrics"])
    return runs
def mean_test(runs):
    ep = np.array([m["epoch"] for m in runs[0]])
    te = np.array([[m["test_acc"] for m in r] for r in runs])
    return ep, te.mean(0), te.std(0)
def grok_ep(runs, thr=0.9):
    eps = []
    for r in runs:
        g = next((m["epoch"] for m in r if m["test_acc"] >= thr), None)
        if g: eps.append(g)
    return np.mean(eps) if eps else None

BG="#0f1117"; PANEL="#181b24"; INK="#e6e9ef"; MUT="#9aa3b2"; GRID="#2a2f3a"
GREEN="#37c87a"; BLUE="#4aa3ff"
plt.rcParams.update({"text.color":INK,"axes.labelcolor":INK,"xtick.color":MUT,
                     "ytick.color":MUT,"axes.edgecolor":GRID,"font.size":10})
fig, ax = plt.subplots(figsize=(8.4, 5.0), facecolor=BG); ax.set_facecolor(PANEL)

adamw = load("adamw"); filt = load("filter_adamw")
for runs, c, lab in [(adamw, MUT, "AdamW (no filter)"), (filt, GREEN, "filter + AdamW")]:
    ep, mu, sd = mean_test(runs)
    ax.plot(ep, mu, color=c, lw=2.4, label=lab)
    ax.fill_between(ep, np.clip(mu-sd,0,1), np.clip(mu+sd,0,1), color=c, alpha=0.15)
ga, gf = grok_ep(adamw), grok_ep(filt)
ax.axhline(0.9, color=MUT, ls=":", lw=0.8)
if ga: ax.axvline(ga, color=MUT, ls="--", lw=1)
if gf: ax.axvline(gf, color=GREEN, ls="--", lw=1)
if ga and gf:
    ax.annotate("", xy=(gf, 0.5), xytext=(ga, 0.5),
                arrowprops=dict(arrowstyle="<->", color=BLUE, lw=1.6))
    ax.text((ga+gf)/2, 0.54, f"{(1-gf/ga)*100:.0f}% earlier", color=BLUE, ha="center", fontsize=10)
ax.set_xlabel("epoch"); ax.set_ylabel("test accuracy")
ax.set_ylim(0, 1.03); ax.set_xlim(0, 6000)
ax.set_title("Modular addition: the filter accelerates grokking\n(mean ± std over 5 seeds; grok = test ≥ 0.9)",
             color=INK, fontsize=12)
ax.legend(facecolor=PANEL, edgecolor=GRID, fontsize=10, labelcolor=INK, loc="center left")
ax.grid(True, color=GRID, lw=0.5, alpha=0.5)
fig.tight_layout()
out = "/Users/titus/pyg/optimizers/research/blog_grokking_modadd.png"
fig.savefig(out, dpi=130, facecolor=BG); print("saved", out, "| adamw grok", ga, "filter grok", gf)
