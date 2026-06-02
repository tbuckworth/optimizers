#!/usr/bin/env python3
"""Fair A/B/C: covariance vs correlation vs spectral basis on MNIST @ 50% label noise."""
import json, glob, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = "/Users/titus/pyg/optimizers/results/mnist_normalize"
SEEDS = [0, 1, 2]
CONDS = [("adam", "Adam (no filter)"), ("ours_none", "covariance"),
         ("ours_var", "correlation"), ("ours_degree", "spectral (degree*)")]
def runs(prefix):
    out = []
    for s in SEEDS:
        p = os.path.join(D, f"{prefix}_s{s}.json")
        if os.path.exists(p): out.append(json.load(open(p)))
    return out
def curve(rs):
    ep = np.array([m["epoch"] for m in rs[0]["metrics"]])
    te = np.array([[m["test_acc"] for m in r["metrics"]] for r in rs])
    return ep, te.mean(0), te.std(0)

BG="#0f1117"; PANEL="#181b24"; INK="#e6e9ef"; MUT="#9aa3b2"; GRID="#2a2f3a"
COL={"adam":MUT,"ours_none":"#4aa3ff","ours_var":"#37c87a","ours_degree":"#ffb454"}
plt.rcParams.update({"text.color":INK,"axes.labelcolor":INK,"xtick.color":MUT,
                     "ytick.color":MUT,"axes.edgecolor":GRID,"font.size":10})
fig = plt.figure(figsize=(13,4.8), facecolor=BG)
gs = fig.add_gridspec(1,2,wspace=0.22,left=0.07,right=0.98,top=0.84,bottom=0.13)

# Panel 1: test curves
ax = fig.add_subplot(gs[0,0]); ax.set_facecolor(PANEL)
summary = {}
for key,lab in CONDS:
    rs = runs(key)
    if not rs: continue
    ep,mu,sd = curve(rs)
    ax.plot(ep,mu,color=COL[key],lw=2.2,label=lab)
    ax.fill_between(ep,mu-sd,mu+sd,color=COL[key],alpha=0.13)
    finals=[r["metrics"][-1]["test_acc"] for r in rs]
    bests=[max(m["test_acc"] for m in r["metrics"]) for r in rs]
    summary[key]=(np.mean(finals),np.std(finals),np.mean(bests))
ax.set_xlabel("epoch"); ax.set_ylabel("test accuracy")
ax.set_title("MNIST @ 50% label noise — basis A/B/C\n(mean ± std, 3 seeds; one knob = normalization)",
             color=INK, fontsize=11.5)
ax.legend(facecolor=PANEL,edgecolor=GRID,fontsize=9.5,labelcolor=INK,loc="lower right")
ax.grid(True,color=GRID,lw=0.5,alpha=0.5)

# Panel 2: final & best bars
ax = fig.add_subplot(gs[0,1]); ax.set_facecolor(PANEL)
keys=[k for k,_ in CONDS if k in summary]; labs=[l for k,l in CONDS if k in summary]
x=np.arange(len(keys)); w=0.38
ax.bar(x-w/2,[summary[k][0] for k in keys],w,yerr=[summary[k][1] for k in keys],
       color=[COL[k] for k in keys],label="final",capsize=3)
ax.bar(x+w/2,[summary[k][2] for k in keys],w,color=[COL[k] for k in keys],alpha=0.45,label="best")
for i,k in enumerate(keys):
    ax.text(i-w/2,summary[k][0]+0.012,f"{summary[k][0]*100:.1f}",ha="center",fontsize=8.5,color=INK)
ax.set_xticks(x); ax.set_xticklabels(labs,fontsize=8.5,rotation=12)
ax.set_ylabel("test accuracy"); ax.set_ylim(0,1.0)
ax.set_title("Final (solid) vs best (faded) test accuracy",color=INK,fontsize=11.5)
ax.legend(facecolor=PANEL,edgecolor=GRID,fontsize=9,labelcolor=INK,loc="lower right")
ax.grid(True,axis="y",color=GRID,lw=0.5,alpha=0.4)

fig.suptitle("Which basis to project onto? covariance vs correlation vs spectral (matched, MNIST 50% noise)",
             color=INK,fontsize=12.5,y=0.965)
out="/Users/titus/pyg/optimizers/research/normalize_test.png"
fig.savefig(out,dpi=130,facecolor=BG); print("saved",out)
for k,l in CONDS:
    if k in summary: print(f"  {l:24s} final={summary[k][0]*100:.1f}±{summary[k][1]*100:.1f}  best={summary[k][2]*100:.1f}")
