#!/usr/bin/env python3
"""CIFAR-10 test accuracy over training, by effective rank, at each noise level."""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = "/Users/titus/pyg/optimizers/results/cifar_noise/ranksweep"
def m(name):
    p = os.path.join(D, name)
    return json.load(open(p))["metrics"] if os.path.exists(p) else None
def xy(rec): return [r["epoch"] for r in rec], [r["test_acc"] for r in rec]

BG="#0f1117"; PANEL="#181b24"; INK="#e6e9ef"; MUT="#9aa3b2"; GRID="#2a2f3a"
COLS={"adam":MUT,"r200":"#ff5d6c","r500":"#ffb454","r1200":"#37c87a"}
plt.rcParams.update({"text.color":INK,"axes.labelcolor":INK,"xtick.color":MUT,
                     "ytick.color":MUT,"axes.edgecolor":GRID,"font.size":10})
fig = plt.figure(figsize=(14,4.6), facecolor=BG)
gs = fig.add_gridspec(1,3,wspace=0.16,left=0.05,right=0.99,top=0.83,bottom=0.14)

NOISE=[("0%","n0"),("40%","n40"),("80%","n80")]
series=[("adam","adam_{t}.json","Adam (no filter)"),
        ("r200","ours_d99_r200_{t}.json","filter r200/d.99 (most filtering)"),
        ("r500","ours_d997_r500_{t}.json","filter r500/d.997"),
        ("r1200","ours_d999_r1200_{t}.json","filter r1200/d.999 (least)")]
for col,(lab,tag) in enumerate(NOISE):
    ax=fig.add_subplot(gs[0,col]); ax.set_facecolor(PANEL)
    for key,pat,leg in series:
        rec=m(pat.format(t=tag))
        if rec is None: continue
        e,te=xy(rec); ax.plot(e,te,color=COLS[key],lw=2.0,label=leg)
    ax.set_ylim(0,0.8); ax.set_xlabel("epoch")
    if col==0: ax.set_ylabel("test accuracy (clean test set)")
    ax.set_title(f"{lab} label noise", color=INK, fontsize=12)
    ax.grid(True,color=GRID,lw=0.5,alpha=0.5)
    if col==0: ax.legend(facecolor=PANEL,edgecolor=GRID,fontsize=8,labelcolor=INK,loc="lower right")
fig.suptitle("CIFAR-10 test accuracy over training, by effective rank — small CNN, clean-task ceiling ~74%",
             color=INK, fontsize=12.5, y=0.96)
out="/Users/titus/pyg/optimizers/research/cifar_ranks_overtime.png"
fig.savefig(out,dpi=130,facecolor=BG); print("saved",out)
