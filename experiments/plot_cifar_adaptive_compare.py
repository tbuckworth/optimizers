#!/usr/bin/env python3
"""CIFAR: energy vs effective-rank adaptive rules both underfit (spectrum too top-heavy)."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

AD = "/Users/titus/pyg/optimizers/results/cifar_noise/adaptive"   # energy threshold
EF = "/Users/titus/pyg/optimizers/results/cifar_noise/effrank"    # effective rank
def L(p): return json.load(open(p))

noises = [0.0, 0.4, 0.8]; tags = {0.0:"0", 0.4:"40", 0.8:"80"}
adam = {0.0: 0.733, 0.4: 0.446, 0.8: 0.194}
best_fixed = {0.0: 0.742, 0.4: 0.648, 0.8: 0.441}

def effrank_final_kr(nz):
    d = L(os.path.join(EF, f"effrank_n{tags[nz]}.json"))
    krs = [r["kept_rank_mean"] for r in d["metrics"] if "kept_rank_mean" in r]
    return d["final_test"], float(np.mean(krs)) if krs else 0
def energy_final_kr(nz, e="99"):
    d = L(os.path.join(AD, f"adapt_e{e}_n{tags[nz]}.json"))
    krs = [r["kept_rank_mean"] for r in d["metrics"] if "kept_rank_mean" in r]
    return d["final_test"], float(np.mean(krs)) if krs else 0

BG="#0f1117"; PANEL="#181b24"; INK="#e6e9ef"; MUT="#9aa3b2"; GRID="#2a2f3a"
GREEN="#37c87a"; PURP="#b07cff"; AMBER="#ffb454"
plt.rcParams.update({"text.color":INK,"axes.labelcolor":INK,"xtick.color":MUT,
                     "ytick.color":MUT,"axes.edgecolor":GRID,"font.size":10})
fig = plt.figure(figsize=(13, 4.8), facecolor=BG)
gs = fig.add_gridspec(1, 2, wspace=0.24, left=0.07, right=0.98, top=0.83, bottom=0.13)
x = np.arange(len(noises))

ax = fig.add_subplot(gs[0, 0]); ax.set_facecolor(PANEL)
ax.plot(x, [adam[n] for n in noises], "o--", color=MUT, lw=1.8, ms=7, label="Adam (no filter)")
ax.plot(x, [best_fixed[n] for n in noises], "s-", color=GREEN, lw=2.4, ms=8, label="best FIXED rank")
ax.plot(x, [energy_final_kr(n)[0] for n in noises], "^-", color=AMBER, lw=2, ms=7, label="adaptive energy 99%")
ax.plot(x, [effrank_final_kr(n)[0] for n in noises], "D-", color=PURP, lw=2, ms=7, label="adaptive effrank")
ax.axhline(0.1, color=MUT, ls=":", lw=0.8); ax.text(1.7, 0.115, "chance", color=MUT, fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(["0%","40%","80%"]); ax.set_xlabel("label noise")
ax.set_ylim(0, 0.8); ax.set_ylabel("final test accuracy")
ax.set_title("CIFAR: both adaptive rules underfit\n(effrank no better than energy)", color=INK, fontsize=11.5)
ax.legend(facecolor=PANEL, edgecolor=GRID, fontsize=8.5, labelcolor=INK, loc="upper right")
ax.grid(True, color=GRID, lw=0.5, alpha=0.5)

ax = fig.add_subplot(gs[0, 1]); ax.set_facecolor(PANEL)
w = 0.35
ax.bar(x - w/2, [energy_final_kr(n)[1] for n in noises], w, color=AMBER, label="energy 99%")
ax.bar(x + w/2, [effrank_final_kr(n)[1] for n in noises], w, color=PURP, label="effrank")
ax.axhline(200, color=GREEN, ls="--", lw=1.4); ax.text(2.3, 210, "fixed r=200 (fits)", color=GREEN, fontsize=8, ha="right")
ax.set_yscale("log"); ax.set_ylim(0.8, 400)
ax.set_xticks(x); ax.set_xticklabels(["0%","40%","80%"]); ax.set_xlabel("label noise")
ax.set_ylabel("# directions kept (log)")
ax.set_title("Both land at ~10-13 directions\nspectrum is too top-heavy for any concentration rule", color=INK, fontsize=11.5)
ax.legend(facecolor=PANEL, edgecolor=GRID, fontsize=8.5, labelcolor=INK, loc="upper left")
ax.grid(True, axis="y", color=GRID, lw=0.5, alpha=0.4)

fig.suptitle("CIFAR adaptive rank: effective-rank doesn't rescue it (unlike parity) — high-dim task, low-dim gradient spectrum",
             color=INK, fontsize=12, y=0.965)
out = "/Users/titus/pyg/optimizers/research/cifar_adaptive_compare.png"
fig.savefig(out, dpi=130, facecolor=BG); print("saved", out)
