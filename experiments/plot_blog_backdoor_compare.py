#!/usr/bin/env python3
"""Backdoor ablation: linear model (works) vs MLP (fails) — same conditions, side by side."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LIN = json.load(open("/Users/titus/pyg/optimizers/results/weight_covariance_v2/backdoor_ablation/metrics.json"))
MLP = json.load(open("/Users/titus/pyg/optimizers/results/weight_covariance_v2/backdoor_mlp/metrics.json"))

# conditions that are the SAME method on both (mean-gradient ablation + controls)
CONDS = [("baseline","baseline"),("ablate_init","ablate\ninit"),
         ("ablate_online","ablate\nonline"),("ablate_random","ablate\nrandom")]
keys=[k for k,_ in CONDS]; labs=[l for _,l in CONDS]

BG="#0f1117"; PANEL="#181b24"; INK="#e6e9ef"; MUT="#9aa3b2"; GRID="#2a2f3a"
AMBER="#ffb454"; RED="#ff5d6c"; GREEN="#37c87a"
plt.rcParams.update({"text.color":INK,"axes.labelcolor":INK,"xtick.color":MUT,
                     "ytick.color":MUT,"axes.edgecolor":GRID,"font.size":10})
fig = plt.figure(figsize=(13,4.8), facecolor=BG)
gs = fig.add_gridspec(1,2,wspace=0.2,left=0.07,right=0.98,top=0.84,bottom=0.16)
x=np.arange(len(keys)); w=0.38

# Panel A: ASR (backdoor strength)
ax=fig.add_subplot(gs[0,0]); ax.set_facecolor(PANEL)
lin=[LIN[k]["asr"] for k in keys]; mlp=[MLP[k]["asr"] for k in keys]
ax.bar(x-w/2,lin,w,color=AMBER,label="linear model")
ax.bar(x+w/2,mlp,w,color=RED,label="MLP")
for i,v in enumerate(lin): ax.text(i-w/2,v+0.015,f"{v*100:.0f}",ha="center",fontsize=8.5,color=INK)
for i,v in enumerate(mlp): ax.text(i+w/2,v+0.015,f"{v*100:.0f}",ha="center",fontsize=8.5,color=INK)
ax.set_xticks(x); ax.set_xticklabels(labs,fontsize=8.5); ax.set_ylim(0,1.12)
ax.set_ylabel("attack success rate (ASR)")
ax.set_title("Backdoor strength: ablation kills it on the\nlinear model (8–17%), not on the MLP (~100%)",color=INK,fontsize=11)
ax.legend(facecolor=PANEL,edgecolor=GRID,fontsize=9,labelcolor=INK,loc="center left")
ax.grid(True,axis="y",color=GRID,lw=0.5,alpha=0.4)

# Panel B: clean accuracy (ablation is cheap on both -> MLP failure isn't a broken model)
ax=fig.add_subplot(gs[0,1]); ax.set_facecolor(PANEL)
lin=[LIN[k]["clean"] for k in keys]; mlp=[MLP[k]["clean"] for k in keys]
ax.bar(x-w/2,lin,w,color=AMBER,label="linear model")
ax.bar(x+w/2,mlp,w,color=RED,label="MLP")
ax.set_xticks(x); ax.set_xticklabels(labs,fontsize=8.5); ax.set_ylim(0,1.05)
ax.set_ylabel("clean test accuracy")
ax.set_title("Clean accuracy barely moves on either —\nso the MLP failure is redundancy, not damage",color=INK,fontsize=11)
ax.legend(facecolor=PANEL,edgecolor=GRID,fontsize=9,labelcolor=INK,loc="lower left")
ax.grid(True,axis="y",color=GRID,lw=0.5,alpha=0.4)

fig.suptitle("Targeted backdoor ablation: linear vs MLP (10% poison, 3×3 trigger → class 0)",
             color=INK,fontsize=12.5,y=0.965)
out="/Users/titus/pyg/optimizers/research/backdoor_compare.png"
fig.savefig(out,dpi=130,facecolor=BG); print("saved",out)
print("linear ASR:",[round(LIN[k]['asr'],3) for k in keys])
print("mlp ASR:",[round(MLP[k]['asr'],3) for k in keys])
