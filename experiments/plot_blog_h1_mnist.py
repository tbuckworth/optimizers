#!/usr/bin/env python3
"""H1 (the strong version): MNIST @ 90% label noise — filter holds while Adam/LoRA collapse."""
import json, glob, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = "/Users/titus/pyg/optimizers/results/weight_covariance_v2/noise90_long"
SEEDS = [42, 43, 44]
def runs(prefix):
    return [json.load(open(os.path.join(D, f"{prefix}_s{s}.json"))) for s in SEEDS
            if os.path.exists(os.path.join(D, f"{prefix}_s{s}.json"))]
def curve(prefix):
    rs = runs(prefix)
    ep = np.array([m["epoch"] for m in rs[0]["metrics"]])
    te = np.array([[m["test_acc"] for m in r["metrics"]] for r in rs])
    return ep, te.mean(0), te.std(0)
def finals(prefix):
    rs = runs(prefix); return np.mean([r["metrics"][-1]["test_acc"] for r in rs])

BG="#0f1117"; PANEL="#181b24"; INK="#e6e9ef"; MUT="#9aa3b2"; GRID="#2a2f3a"
GREEN="#37c87a"; RED="#ff5d6c"; BLUE="#4aa3ff"; AMBER="#ffb454"; PURP="#b07cff"
plt.rcParams.update({"text.color":INK,"axes.labelcolor":INK,"xtick.color":MUT,
                     "ytick.color":MUT,"axes.edgecolor":GRID,"font.size":10})
fig = plt.figure(figsize=(13,4.8), facecolor=BG)
gs = fig.add_gridspec(1,2,wspace=0.22,left=0.07,right=0.98,top=0.84,bottom=0.13)

# Panel A: test curves over time at 90% noise
ax = fig.add_subplot(gs[0,0]); ax.set_facecolor(PANEL)
for pref,c,lab in [("ours_r200_n90",GREEN,"filter (rank 200)"),
                   ("random_subspace_n90",PURP,"random subspace (control)"),
                   ("adam_n90",RED,"Adam"),("lora_n90",AMBER,"LoRA")]:
    ep,mu,sd = curve(pref)
    ax.plot(ep,mu,color=c,lw=2.3,label=lab); ax.fill_between(ep,mu-sd,mu+sd,color=c,alpha=0.13)
ax.set_xlabel("epoch"); ax.set_ylabel("test accuracy (clean test set)"); ax.set_ylim(0,1.0)
ax.set_title("MNIST @ 90% label noise — Adam & LoRA peak\nthen collapse; the filter holds", color=INK, fontsize=11.5)
ax.legend(facecolor=PANEL,edgecolor=GRID,fontsize=9,labelcolor=INK,loc="center right")
ax.grid(True,color=GRID,lw=0.5,alpha=0.5)

# Panel B: final test bars at 90% noise (rank matters; beats LoRA & random-subspace)
ax = fig.add_subplot(gs[0,1]); ax.set_facecolor(PANEL)
order=[("adam_n90","Adam",RED),("lora_n90","LoRA",AMBER),
       ("random_subspace_n90","random\nsubspace",PURP),
       ("ours_r50_n90","filter\nr50",BLUE),("ours_r100_n90","filter\nr100",BLUE),
       ("ours_r200_n90","filter\nr200",GREEN)]
vals=[finals(p) for p,_,_ in order]; cols=[c for _,_,c in order]; labs=[l for _,l,_ in order]
ax.bar(range(len(vals)),vals,color=cols)
for i,v in enumerate(vals): ax.text(i,v+0.012,f"{v*100:.0f}",ha="center",fontsize=9,color=INK)
ax.set_xticks(range(len(labs))); ax.set_xticklabels(labs,fontsize=8.5)
ax.set_ylim(0,1.0); ax.set_ylabel("final test accuracy")
ax.set_title("Final test @ 90% noise: learned subspace > random,\n> LoRA > Adam; higher rank holds more", color=INK, fontsize=11)
ax.grid(True,axis="y",color=GRID,lw=0.5,alpha=0.4)

fig.suptitle("H1 — extreme label noise (MNIST, 90%, 3 seeds): the filter resists memorization where Adam, LoRA, and random subspaces fail",
             color=INK,fontsize=11.5,y=0.965)
out="/Users/titus/pyg/optimizers/research/blog_h1_mnist.png"
fig.savefig(out,dpi=130,facecolor=BG); print("saved",out)
