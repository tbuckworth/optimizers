#!/usr/bin/env python3
"""Two limitation results: per-sample filter on parity, and backdoor ablation on an MLP."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PP = "/Users/titus/pyg/optimizers/results/parity_persample"
BD = "/Users/titus/pyg/optimizers/results/weight_covariance_v2"
def L(p): return json.load(open(p))

BG="#0f1117"; PANEL="#181b24"; INK="#e6e9ef"; MUT="#9aa3b2"; GRID="#2a2f3a"
GREEN="#37c87a"; RED="#ff5d6c"; BLUE="#4aa3ff"; AMBER="#ffb454"
plt.rcParams.update({"text.color":INK,"axes.labelcolor":INK,"xtick.color":MUT,
                     "ytick.color":MUT,"axes.edgecolor":GRID,"font.size":10})

fig = plt.figure(figsize=(13, 4.7), facecolor=BG)
gs = fig.add_gridspec(1, 2, wspace=0.26, left=0.07, right=0.98, top=0.84, bottom=0.16)

# Panel 1: parity final test — adamw vs per-sample filter (all ranks)
ax = fig.add_subplot(gs[0, 0]); ax.set_facecolor(PANEL)
def meanfinal(prefix, seeds=(0,1,2)):
    return np.mean([L(os.path.join(PP, f"{prefix}_s{s}.json"))["final_test"] for s in seeds])
labels = ["AdamW", "per-samp\nr3", "per-samp\nr10", "per-samp\nr50", "per-samp\nadaptive"]
vals = [meanfinal("adamw"), meanfinal("persample_r3"), meanfinal("persample_r10"),
        meanfinal("persample_r50"), meanfinal("persample_e99")]
cols = [GREEN, RED, RED, RED, RED]
ax.bar(range(len(vals)), vals, color=cols)
ax.axhline(0.5, color=MUT, ls=":", lw=0.8)
ax.text(2, 0.52, "chance", color=MUT, fontsize=8, ha="center")
ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, fontsize=8.5)
ax.set_ylim(0, 1.05); ax.set_ylabel("final test accuracy")
ax.set_title("Parity: per-sample top-k projection\nNEVER groks at any rank (≈ chance)", color=INK, fontsize=11)
ax.grid(True, axis="y", color=GRID, lw=0.5, alpha=0.5)

# Panel 2: backdoor ASR — linear (ablatable) vs MLP (not)
ax = fig.add_subplot(gs[0, 1]); ax.set_facecolor(PANEL)
conds = ["baseline", "ablate_init", "ablate_online", "ablate_persample", "ablate_random"]
# linear numbers from summary §5b (ablate_eig stands in the linear-only slot; use init/online)
linear = {"baseline":0.999, "ablate_init":0.080, "ablate_online":0.167,
          "ablate_persample":0.098, "ablate_random":0.999}  # persample slot = linear eig route (0.098)
mlp = L(os.path.join(BD, "backdoor_mlp", "metrics.json"))
mlp_asr = [mlp[c]["asr"] for c in conds]
lin_asr = [linear[c] for c in conds]
x = np.arange(len(conds)); w = 0.38
ax.bar(x - w/2, lin_asr, w, color=AMBER, label="linear model")
ax.bar(x + w/2, mlp_asr, w, color=RED, label="MLP")
ax.set_xticks(x); ax.set_xticklabels([c.replace("ablate_", "abl_") for c in conds], fontsize=8, rotation=15)
ax.set_ylim(0, 1.08); ax.set_ylabel("attack success rate (ASR)")
ax.set_title("Backdoor ablation: works on linear,\nFAILS on MLP (every condition ~100%)", color=INK, fontsize=11)
ax.legend(facecolor=PANEL, edgecolor=GRID, fontsize=9, labelcolor=INK, loc="center left")
ax.grid(True, axis="y", color=GRID, lw=0.5, alpha=0.5)

fig.suptitle("Two limitation results: where 'keep/remove the top gradient directions' stops working",
             color=INK, fontsize=12.5, y=0.965)
out = "/Users/titus/pyg/optimizers/research/limitations.png"
fig.savefig(out, dpi=130, facecolor=BG)
print("saved", out)
