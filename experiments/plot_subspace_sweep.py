#!/usr/bin/env python3
"""Subspace-size sweep on the MLP backdoor: does ablating MORE directions remove it?"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

J = json.load(open("/Users/titus/pyg/optimizers/results/weight_covariance_v2/backdoor_mlp/subspace_sweep.json"))
sw = J["sweep_m"]
ms = sorted(int(k) for k in sw)
asr = [sw[str(m)]["asr"] for m in ms]
clean = [sw[str(m)]["clean"] for m in ms]
energy = [sw[str(m)]["subspace_energy"] for m in ms]

BG="#0f1117"; PANEL="#181b24"; INK="#e6e9ef"; MUT="#9aa3b2"; GRID="#2a2f3a"
RED="#ff5d6c"; GREEN="#37c87a"; BLUE="#4aa3ff"
plt.rcParams.update({"text.color":INK,"axes.labelcolor":INK,"xtick.color":MUT,
                     "ytick.color":MUT,"axes.edgecolor":GRID,"font.size":10})

fig, ax = plt.subplots(figsize=(8.2, 5.0), facecolor=BG)
ax.set_facecolor(PANEL)
x = np.arange(len(ms))
ax.plot(x, asr, "o-", color=RED, lw=2.2, ms=8, label="ASR (backdoor strength)")
ax.plot(x, clean, "s-", color=GREEN, lw=2.2, ms=8, label="clean test accuracy")
ax.axhline(J["baseline"]["asr"], color=MUT, ls=":", lw=1)
ax.set_xticks(x); ax.set_xticklabels([f"m={m}\n({e:.0%} energy)" for m, e in zip(ms, energy)], fontsize=9)
ax.set_ylim(0.0, 1.06); ax.set_ylabel("rate")
ax.set_xlabel("ablated subspace size (# top per-sample gradient eigenvectors removed)")
ax.set_title("MLP backdoor: ablating MORE directions never removes it\n"
             "ASR stays ~100% until clean accuracy is already destroyed (m=100)",
             color=INK, fontsize=12)
ax.legend(facecolor=PANEL, edgecolor=GRID, fontsize=10, labelcolor=INK, loc="center left")
ax.grid(True, color=GRID, lw=0.5, alpha=0.5)
fig.tight_layout()
out = "/Users/titus/pyg/optimizers/research/subspace_sweep.png"
fig.savefig(out, dpi=130, facecolor=BG)
print("saved", out)
