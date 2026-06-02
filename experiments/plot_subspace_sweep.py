#!/usr/bin/env python3
"""Backdoor subspace-size sweep: ablating m per-sample directions.
MLP (solid) vs linear (dashed); ASR (red) and clean accuracy (green), one axis."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MLP = json.load(open("/Users/titus/pyg/optimizers/results/weight_covariance_v2/backdoor_mlp/subspace_sweep.json"))
LIN = json.load(open("/Users/titus/pyg/optimizers/results/weight_covariance_v2/backdoor_ablation/subspace_sweep.json"))

def series(J):
    sw = J["sweep_m"]; ms = sorted(int(k) for k in sw)
    return ms, [sw[str(m)]["asr"] for m in ms], [sw[str(m)]["clean"] for m in ms]

BG="#0f1117"; PANEL="#181b24"; INK="#e6e9ef"; MUT="#9aa3b2"; GRID="#2a2f3a"
RED="#ff5d6c"; GREEN="#37c87a"
plt.rcParams.update({"text.color":INK,"axes.labelcolor":INK,"xtick.color":MUT,
                     "ytick.color":MUT,"axes.edgecolor":GRID,"font.size":10})
fig, ax = plt.subplots(figsize=(9,5.4), facecolor=BG); ax.set_facecolor(PANEL)

for J, ls, modtag in [(MLP, "-", "MLP"), (LIN, "--", "linear")]:
    ms, asr, clean = series(J)
    ax.plot(ms, asr, ls, color=RED, lw=2.3, marker="o", ms=6, label=f"ASR — {modtag}")
    ax.plot(ms, clean, ls, color=GREEN, lw=2.3, marker="s", ms=6, label=f"clean — {modtag}")

ax.set_xscale("log"); ax.set_xticks([1,3,10,30,100]); ax.set_xticklabels([1,3,10,30,100])
ax.set_ylim(0,1.06); ax.set_xlabel("ablated subspace size  m  (top-m per-sample gradient directions removed)")
ax.set_ylabel("rate")
ax.set_title("Ablating the backdoor: works on the linear model (dashed — ASR crashes\n"
             "at m=1, clean held), fails on the MLP (solid — ASR stays ~100% until clean collapses)",
             color=INK, fontsize=11.5)
ax.legend(facecolor=PANEL, edgecolor=GRID, fontsize=9.5, labelcolor=INK, loc="center right", ncol=2)
ax.grid(True, color=GRID, lw=0.5, alpha=0.5)
fig.tight_layout()
out="/Users/titus/pyg/optimizers/research/subspace_sweep.png"
fig.savefig(out, dpi=130, facecolor=BG); print("saved", out)
