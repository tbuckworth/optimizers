#!/usr/bin/env python3
"""Adaptive energy-threshold rank on CIFAR noise vs fixed-rank references."""
import json, glob, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

AD = "/Users/titus/pyg/optimizers/results/cifar_noise/adaptive"
RS = "/Users/titus/pyg/optimizers/results/cifar_noise/ranksweep"
def L(p): return json.load(open(p))

# adaptive: final test + kept_rank per (energy, noise)
adapt = {}
for f in glob.glob(os.path.join(AD, "*.json")):
    d = L(f); c = d["config"]
    krs = [r["kept_rank_mean"] for r in d["metrics"] if "kept_rank_mean" in r]
    adapt[(round(c["energy_threshold"], 2), c["noise"])] = (d["final_test"], np.mean(krs) if krs else 0)

noises = [0.0, 0.4, 0.8]; ens = [0.90, 0.95, 0.99]
# fixed references
adam = {0.0: 0.733, 0.4: 0.446, 0.8: 0.194}
best_fixed = {0.0: 0.742, 0.4: 0.648, 0.8: 0.441}   # Pareto front across r200/r500/r1200

BG="#0f1117"; PANEL="#181b24"; INK="#e6e9ef"; MUT="#9aa3b2"; GRID="#2a2f3a"
RED="#ff5d6c"; GREEN="#37c87a"; BLUE="#4aa3ff"; AMBER="#ffb454"; PURP="#b07cff"
plt.rcParams.update({"text.color":INK,"axes.labelcolor":INK,"xtick.color":MUT,
                     "ytick.color":MUT,"axes.edgecolor":GRID,"font.size":10})
fig = plt.figure(figsize=(13, 4.8), facecolor=BG)
gs = fig.add_gridspec(1, 2, wspace=0.24, left=0.07, right=0.98, top=0.83, bottom=0.13)

# Panel 1: final test vs noise
ax = fig.add_subplot(gs[0, 0]); ax.set_facecolor(PANEL)
x = np.arange(len(noises))
ax.plot(x, [adam[n] for n in noises], "o--", color=MUT, lw=1.8, ms=7, label="Adam (no filter)")
ax.plot(x, [best_fixed[n] for n in noises], "s-", color=GREEN, lw=2.4, ms=8, label="best FIXED rank (Pareto)")
encol = {0.90: RED, 0.95: AMBER, 0.99: PURP}
for e in ens:
    ax.plot(x, [adapt[(e, n)][0] for n in noises], "^-", color=encol[e], lw=2, ms=7,
            label=f"adaptive {int(e*100)}% energy")
ax.axhline(0.1, color=MUT, ls=":", lw=0.8); ax.text(1.7, 0.115, "chance", color=MUT, fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(["0%", "40%", "80%"]); ax.set_xlabel("label noise")
ax.set_ylim(0, 0.8); ax.set_ylabel("final test accuracy")
ax.set_title("Adaptive energy-rule UNDERFITS everywhere\n(dominated by fixed rank at every noise level)",
             color=INK, fontsize=11.5)
ax.legend(facecolor=PANEL, edgecolor=GRID, fontsize=8.5, labelcolor=INK, loc="upper right")
ax.grid(True, color=GRID, lw=0.5, alpha=0.5)

# Panel 2: kept rank (log) — adaptive vs fixed
ax = fig.add_subplot(gs[0, 1]); ax.set_facecolor(PANEL)
w = 0.25
for i, e in enumerate(ens):
    ax.bar(x + (i-1)*w, [adapt[(e, n)][1] for n in noises], w, color=encol[e],
           label=f"adaptive {int(e*100)}%")
for r, ls in [(200, "--"), (1200, ":")]:
    ax.axhline(r, color=GREEN, ls=ls, lw=1.4)
    ax.text(2.3, r*1.05, f"fixed r={r}", color=GREEN, fontsize=8, ha="right")
ax.set_yscale("log"); ax.set_ylim(0.8, 2000)
ax.set_xticks(x); ax.set_xticklabels(["0%", "40%", "80%"]); ax.set_xlabel("label noise")
ax.set_ylabel("# directions kept (log scale)")
ax.set_title("Why: 99% of gradient ENERGY lives in ~10 directions\n"
             "fitting CIFAR needs ~200 — the rule keeps 20-200x too few",
             color=INK, fontsize=11.5)
ax.legend(facecolor=PANEL, edgecolor=GRID, fontsize=8.5, labelcolor=INK, loc="upper right")
ax.grid(True, axis="y", color=GRID, lw=0.5, alpha=0.4)

fig.suptitle("CIFAR adaptive energy-threshold rank: a clean negative — energy is the wrong currency",
             color=INK, fontsize=12.5, y=0.965)
out = "/Users/titus/pyg/optimizers/research/cifar_adaptive.png"
fig.savefig(out, dpi=130, facecolor=BG)
print("saved", out)
