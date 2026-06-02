#!/usr/bin/env python3
"""CIFAR effective-rank sweep figure: clean-fit recovery, the tradeoff, the ceiling."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = "/Users/titus/pyg/optimizers/results/cifar_noise/ranksweep"
def L(n): return json.load(open(os.path.join(D, n + ".json")))
def fin(n): return L(n)["metrics"][-1]
def curve(n):
    m = L(n)["metrics"]; return [r["epoch"] for r in m], [r["train_acc"] for r in m], [r["test_acc"] for r in m]

BG="#0f1117"; PANEL="#181b24"; INK="#e6e9ef"; MUT="#9aa3b2"; GRID="#2a2f3a"
GREEN="#37c87a"; RED="#ff5d6c"; BLUE="#4aa3ff"; AMBER="#ffb454"; PURPLE="#b18cff"
plt.rcParams.update({"text.color":INK,"axes.labelcolor":INK,"xtick.color":MUT,
                     "ytick.color":MUT,"axes.edgecolor":GRID,"font.size":10})

fig = plt.figure(figsize=(15, 4.7), facecolor=BG)
gs = fig.add_gridspec(1, 3, wspace=0.30, left=0.06, right=0.985, top=0.84, bottom=0.16)

# Panel 1: clean-data fit recovers as effective rank rises (0% noise)
ax = fig.add_subplot(gs[0, 0]); ax.set_facecolor(PANEL)
labels = ["r200\nd.99", "r500\nd.997", "r1200\nd.999"]
names0 = ["ours_d99_r200_n0", "ours_d997_r500_n0", "ours_d999_r1200_n0"]
tr = [fin(n)["train_acc"] for n in names0]; te = [fin(n)["test_acc"] for n in names0]
x = np.arange(3); w = 0.36
ax.bar(x - w/2, tr, w, color=PURPLE, label="train")
ax.bar(x + w/2, te, w, color=BLUE, label="test")
ax.axhline(fin("adam_n0")["test_acc"], color=GREEN, ls="--", lw=1.6, label="adam test (0.733)")
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
ax.set_ylim(0, 1.02); ax.set_ylabel("accuracy")
ax.set_title("0% noise: higher effective rank\nRECOVERS clean fit (→ adam)", color=INK, fontsize=11)
ax.legend(facecolor=PANEL, edgecolor=GRID, fontsize=8.5, labelcolor=INK, loc="lower right")
ax.grid(True, axis="y", color=GRID, lw=0.5, alpha=0.5)

# Panel 2: the tradeoff — test_final across noise for the 3 effective-rank settings
ax = fig.add_subplot(gs[0, 1]); ax.set_facecolor(PANEL)
noises = [0, 40, 80]
series = {
    "adam":            (GREEN, ["adam_n0","adam_n40","adam_n80"]),
    "r200/d.99 (most filtering)":  (RED, ["ours_d99_r200_n0","ours_d99_r200_n40","ours_d99_r200_n80"]),
    "r500/d.997":      (AMBER, ["ours_d997_r500_n0","ours_d997_r500_n40","ours_d997_r500_n80"]),
    "r1200/d.999 (least)": (BLUE, ["ours_d999_r1200_n0","ours_d999_r1200_n40","ours_d999_r1200_n80"]),
}
for lab,(col,ns) in series.items():
    ax.plot(noises, [fin(n)["test_acc"] for n in ns], "-o", color=col, lw=2, markersize=6, label=lab)
ax.set_xticks(noises); ax.set_xlabel("label noise (%)"); ax.set_ylabel("final test accuracy")
ax.set_title("The dial: high rank wins clean,\nlow rank wins noisy (no free lunch)", color=INK, fontsize=11)
ax.legend(facecolor=PANEL, edgecolor=GRID, fontsize=8, labelcolor=INK, loc="upper right")
ax.grid(True, color=GRID, lw=0.5, alpha=0.5)

# Panel 3: degeneracy control — train_final at 80% noise, rank varies @ fixed decay 0.99
ax = fig.add_subplot(gs[0, 2]); ax.set_facecolor(PANEL)
ctrl = ["ours_d99_r200_n80", "ours_d99_r1000_n80", "ours_d99_r4000_n80"]
clab = ["r200", "r1000", "r4000"]
trc = [fin(n)["train_acc"] for n in ctrl]; tec = [fin(n)["test_acc"] for n in ctrl]
x = np.arange(3)
ax.bar(x - w/2, trc, w, color=PURPLE, label="train (noise memorized)")
ax.bar(x + w/2, tec, w, color=BLUE, label="test")
ax.set_xticks(x); ax.set_xticklabels(clab, fontsize=10)
ax.set_ylim(0, 1.02); ax.set_ylabel("accuracy (final)")
ax.set_title("Ceiling control @ decay 0.99, 80% noise:\nr1000≈r4000 (both memorize); r200 differs", color=INK, fontsize=11)
ax.legend(facecolor=PANEL, edgecolor=GRID, fontsize=8.5, labelcolor=INK, loc="upper left")
ax.grid(True, axis="y", color=GRID, lw=0.5, alpha=0.5)
ax.annotate("below the ceiling\n→ still filtering", xy=(0, trc[0]), xytext=(0.1, 0.55),
            color=RED, fontsize=8.5, ha="center")

fig.suptitle("CIFAR-10 effective-rank sweep (small CNN): rank & decay feed one quantity — how many gradient directions are kept",
             color=INK, fontsize=12.5, y=0.97)
out = "/Users/titus/pyg/optimizers/research/cifar_ranksweep.png"
fig.savefig(out, dpi=130, facecolor=BG)
print("saved", out)
