#!/usr/bin/env python3
"""H1 time-series: Adam memorizes noise then test accuracy collapses; the filter holds."""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = "/Users/titus/pyg/optimizers/results/cifar_noise/ranksweep"
def m(name): return json.load(open(os.path.join(D, name)))["metrics"]
def xy(rec, key): return [r["epoch"] for r in rec], [r[key] for r in rec]

BG="#0f1117"; PANEL="#181b24"; INK="#e6e9ef"; MUT="#9aa3b2"; GRID="#2a2f3a"
GREEN="#37c87a"; RED="#ff5d6c"; BLUE="#4aa3ff"
plt.rcParams.update({"text.color":INK,"axes.labelcolor":INK,"xtick.color":MUT,
                     "ytick.color":MUT,"axes.edgecolor":GRID,"font.size":10})
fig = plt.figure(figsize=(13,4.8), facecolor=BG)
gs = fig.add_gridspec(1,2,wspace=0.18,left=0.07,right=0.98,top=0.84,bottom=0.13)

for col,(noise,tag) in enumerate([("40%","n40"),("80%","n80")]):
    ax = fig.add_subplot(gs[0,col]); ax.set_facecolor(PANEL)
    adam, ours = m(f"adam_{tag}.json"), m(f"ours_d99_r200_{tag}.json")
    # Adam: test (collapses) and train (climbs = memorizing noise)
    e,te = xy(adam,"test_acc"); ax.plot(e,te,color=RED,lw=2.3,label="Adam — test")
    e,tr = xy(adam,"train_acc"); ax.plot(e,tr,color=RED,lw=1.2,ls=":",alpha=0.8,label="Adam — train")
    e,te = xy(ours,"test_acc"); ax.plot(e,te,color=GREEN,lw=2.3,label="filter — test")
    e,tr = xy(ours,"train_acc"); ax.plot(e,tr,color=GREEN,lw=1.2,ls=":",alpha=0.8,label="filter — train")
    # mark Adam's peak then collapse
    pk = max(range(len(adam)), key=lambda i: adam[i]["test_acc"])
    ax.scatter([adam[pk]["epoch"]],[adam[pk]["test_acc"]],color=RED,zorder=5,s=28)
    ax.set_xlabel("epoch"); ax.set_ylim(0,1.0)
    if col==0: ax.set_ylabel("accuracy")
    ax.set_title(f"CIFAR-10, {noise} label noise", color=INK, fontsize=12)
    ax.legend(facecolor=PANEL,edgecolor=GRID,fontsize=8.5,labelcolor=INK,loc="upper left")
    ax.grid(True,color=GRID,lw=0.5,alpha=0.5)

fig.suptitle("H1 — Adam fits the noise (train↑) and its test accuracy collapses; the filter refuses to memorize and holds",
             color=INK,fontsize=12,y=0.965)
out="/Users/titus/pyg/optimizers/research/blog_h1_curves.png"
fig.savefig(out,dpi=130,facecolor=BG); print("saved",out)
