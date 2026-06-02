#!/usr/bin/env python3
"""Parity adaptive effective-rank: it groks, and proj_k follows broaden-then-narrow."""
import json, glob, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = "/Users/titus/pyg/optimizers/results/parity_adaptive"
def load(prefix, seeds=(0,1,2)):
    return [json.load(open(os.path.join(D, f"{prefix}_s{s}.json"))) for s in seeds]

def curve(runs, key):
    ep = [r["epoch"] for r in runs[0]["metrics"]]
    vals = np.array([[m[key] for m in r["metrics"]] for r in runs])
    return np.array(ep), vals.mean(0)

BG="#0f1117"; PANEL="#181b24"; INK="#e6e9ef"; MUT="#9aa3b2"; GRID="#2a2f3a"
GREEN="#37c87a"; BLUE="#4aa3ff"; AMBER="#ffb454"; PURP="#b07cff"
plt.rcParams.update({"text.color":INK,"axes.labelcolor":INK,"xtick.color":MUT,
                     "ytick.color":MUT,"axes.edgecolor":GRID,"font.size":10})
fig = plt.figure(figsize=(13, 4.8), facecolor=BG)
gs = fig.add_gridspec(1, 2, wspace=0.22, left=0.07, right=0.93, top=0.84, bottom=0.14)

adamw = load("adamw"); eff = load("ours_effrank"); gap = load("ours_gap")

# Panel 1: test accuracy curves
ax = fig.add_subplot(gs[0, 0]); ax.set_facecolor(PANEL)
for runs, c, lab in [(adamw, MUT, "AdamW (no filter)"), (eff, GREEN, "adaptive effrank"), (gap, AMBER, "adaptive gap")]:
    ep, te = curve(runs, "test_acc")
    ax.plot(ep, te, color=c, lw=2.2, label=lab)
ax.axhline(0.9, color=MUT, ls=":", lw=0.8); ax.text(5200, 0.91, "grok (0.9)", color=MUT, fontsize=8)
ax.set_xlabel("epoch"); ax.set_ylabel("test accuracy"); ax.set_ylim(0.45, 1.03)
ax.set_title("Adaptive effrank GROKS (unlike fixed-low-rank / energy)\n— slower than AdamW, but doesn't break it",
             color=INK, fontsize=11.5)
ax.legend(facecolor=PANEL, edgecolor=GRID, fontsize=9, labelcolor=INK, loc="center right")
ax.grid(True, color=GRID, lw=0.5, alpha=0.5)

# Panel 2: proj_k trajectory (broaden-then-narrow) + test on twin axis, seed 0
ax = fig.add_subplot(gs[0, 1]); ax.set_facecolor(PANEL)
r0 = eff[0]["metrics"]
ep = [m["epoch"] for m in r0]
pk = [m.get("proj_k") for m in r0]
te = [m["test_acc"] for m in r0]
ax.plot(ep, pk, color=PURP, lw=2.2, label="projection rank (proj_k)")
ax.set_xlabel("epoch"); ax.set_ylabel("projection rank  (= round(effective rank))", color=PURP)
ax.tick_params(axis="y", labelcolor=PURP)
grok = eff[0]["grok_epoch"]
if grok: ax.axvline(grok, color=GREEN, ls="--", lw=1.2); ax.text(grok+60, 2, f"grok @ {grok}", color=GREEN, fontsize=8.5)
ax2 = ax.twinx(); ax2.set_facecolor("none")
ax2.plot(ep, te, color=BLUE, lw=1.6, alpha=0.8, label="test acc")
ax2.set_ylabel("test accuracy", color=BLUE); ax2.tick_params(axis="y", labelcolor=BLUE); ax2.set_ylim(0.45, 1.03)
ax.set_title("Projection rank: broaden-then-narrow\ndips during memorization, peaks at the grok, narrows after",
             color=INK, fontsize=11.5)
ax.grid(True, color=GRID, lw=0.5, alpha=0.4)

fig.suptitle("Sparse parity: effective-rank-targeting (the faithful adaptive rule)",
             color=INK, fontsize=12.5, y=0.965)
out = "/Users/titus/pyg/optimizers/research/parity_adaptive.png"
fig.savefig(out, dpi=130, facecolor=BG)
print("saved", out)
