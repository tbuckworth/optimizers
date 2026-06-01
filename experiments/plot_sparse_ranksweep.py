#!/usr/bin/env python3
"""Visualize the sparse-parity low-rank sweep: rank->grok, destabilization, spectrum."""
import json, os, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = "/Users/titus/pyg/optimizers/results/sparse_parity_ranksweep"
def load(n): return json.load(open(os.path.join(D, n + ".json")))

BG="#0f1117"; PANEL="#181b24"; INK="#e6e9ef"; MUT="#9aa3b2"
GRID="#2a2f3a"; GREEN="#37c87a"; RED="#ff5d6c"; BLUE="#4aa3ff"; AMBER="#ffb454"
plt.rcParams.update({"text.color":INK,"axes.labelcolor":INK,"xtick.color":MUT,
                     "ytick.color":MUT,"axes.edgecolor":GRID,"font.size":10})

fig = plt.figure(figsize=(15, 4.6), facecolor=BG)
gs = fig.add_gridspec(1, 3, wspace=0.28, left=0.06, right=0.98, top=0.86, bottom=0.14)

# ---- Panel 1: rank -> grok epoch (non-monotonic), AdamW baseline ----
ax = fig.add_subplot(gs[0, 0]); ax.set_facecolor(PANEL)
ranks = [1, 3, 10, 50, 200]
adamw = [load(f"adamw_s{s}")["grok_epoch"] for s in (0, 1, 2)]
ax.axhline(np.mean(adamw), color=GREEN, ls="--", lw=1.6, label=f"AdamW baseline (~{int(np.mean(adamw))})")
CAP = 6000
for r in ranks:
    gs_ = []
    for s in (0, 1, 2):
        g = load(f"ours_r{r}_s{s}")["grok_epoch"]
        gs_.append(g)
    xr = [r]*3
    grok_pts = [g for g in gs_ if g is not None]
    never = [CAP for g in gs_ if g is None]
    if grok_pts:
        ax.scatter([r]*len(grok_pts), grok_pts, s=70, color=BLUE, zorder=3, edgecolor="white", linewidth=0.5)
    if never:
        ax.scatter([r]*len(never), never, s=80, marker="x", color=RED, zorder=3, linewidth=2)
ax.set_xscale("log"); ax.set_xticks(ranks); ax.set_xticklabels(ranks)
ax.set_ylim(0, 6400)
ax.text(1.0, CAP+120, "never groks", color=RED, fontsize=9, ha="left")
ax.set_xlabel("filter rank (log)"); ax.set_ylabel("grok epoch (test≥0.9)")
ax.set_title("Rank → grokking speed: non-monotonic,\nnever beats AdamW", color=INK, fontsize=11)
ax.grid(True, color=GRID, lw=0.5, alpha=0.5)
ax.legend(loc="upper left", facecolor=PANEL, edgecolor=GRID, fontsize=8.5, labelcolor=INK)
ax.scatter([], [], s=70, color=BLUE, edgecolor="white", linewidth=0.5, label="groks")  # legend dummy

# ---- Panel 2: train/test curves — rank1 destabilizes vs rank10 groks vs AdamW ----
ax = fig.add_subplot(gs[0, 1]); ax.set_facecolor(PANEL)
def curve(name):
    m = load(name)["metrics"]
    return [r["epoch"] for r in m], [r["train_acc"] for r in m], [r["test_acc"] for r in m]
for name, col, lab in [("adamw_s0", GREEN, "AdamW"), ("ours_r1_s0", RED, "ours r1"),
                       ("ours_r10_s0", BLUE, "ours r10")]:
    e, tr, te = curve(name)
    ax.plot(e, tr, color=col, lw=1.5, alpha=0.55, ls="--")
    ax.plot(e, te, color=col, lw=2.0, label=lab)
ax.plot([], [], color=MUT, lw=2.0, label="— test")
ax.plot([], [], color=MUT, lw=1.5, ls="--", alpha=0.6, label="-- train")
ax.set_xlim(0, 6000); ax.set_ylim(0.4, 1.03)
ax.axhline(0.5, color=MUT, lw=0.7, ls=":", alpha=0.6)
ax.set_xlabel("epoch"); ax.set_ylabel("accuracy")
ax.set_title("rank-1 memorizes then DESTABILIZES;\nrank-10 groks (late); AdamW groks early", color=INK, fontsize=11)
ax.grid(True, color=GRID, lw=0.5, alpha=0.5)
ax.legend(loc="center right", facecolor=PANEL, edgecolor=GRID, fontsize=8.5, labelcolor=INK)

# ---- Panel 3: eigenvalue spectrum broadens during grok, collapses after ----
ax = fig.add_subplot(gs[0, 2]); ax.set_facecolor(PANEL)
d = load("ours_r200_s0")
active = []  # epoch -> #eigenvalues > 1% of top
for sp in d["spectra"]:
    ev = sp["eigenvalues"]
    if not ev: continue
    top = ev[0]
    active.append((sp["epoch"], sum(1 for x in ev if x > 0.01 * top)))
active.sort()
ep = [a[0] for a in active]; na = [a[1] for a in active]
ax.plot(ep, na, "-o", color=AMBER, lw=2, markersize=6)
ax.axvspan(1000, 2000, color=BLUE, alpha=0.12)
ax.text(1500, max(na)*0.92, "grokking\ntransition", color=BLUE, ha="center", fontsize=9)
ax.set_xlabel("epoch"); ax.set_ylabel("# active gradient directions  (eig > 1% of top)")
ax.set_title("Gradient subspace broadens through the\ntransition, then collapses (~6 dims)", color=INK, fontsize=11)
ax.grid(True, color=GRID, lw=0.5, alpha=0.5)

fig.suptitle("Sparse parity (n=40,k=3): filter rank sweep at fixed λ=0.99  —  top eigenvectors are memorization, not the 3-bit signal",
             color=INK, fontsize=12.5, y=0.985)
out = "/Users/titus/pyg/optimizers/research/sparse_parity_ranksweep.png"
fig.savefig(out, dpi=130, facecolor=BG)
print("saved", out)
