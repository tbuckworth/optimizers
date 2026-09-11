"""Plot audit-pinned scheduled means; no models or training replay."""
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
PINS = {
    "analysis-001/audit.json":"3c5a15655eb2882a5d228722556aedd4b3ef67223a286a909de9b78db8fe2d0b",
    "analysis-001/summary.json":"6e39eb9d7602ab159bcef060f710a8dac0a33f6b9d8aca4bfb5810591adfab53",
    "raw-results-001/collection.json":"0f9342d471d6b9dc955cce7eb1d9b9c7d9a75b93d09fb8be357cc6478ae3d400",
    "../iteration-016/analysis-001/summary.json":"11da55533e66c371132f939ce7e4a3a4495efc0fc5f5b202164d496bb438e401",
    "../iteration-016/analysis-001/report-audit.json":"87be955c4f6f01514486969d51414d58b6ac583c0fb68f21cfb74de53ef6e96f",
}
data = {}
for name,digest in PINS.items():
    raw = (HERE/name).read_bytes()
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError("Evidence changed: "+name)
    data[name] = json.loads(raw)
assert data["analysis-001/audit.json"]["status"] == "pass"
assert data["../iteration-016/analysis-001/report-audit.json"]["status"] == "pass"
new = data["analysis-001/summary.json"]["scheduled_metric_equal_seed_aggregates"]
old = data["../iteration-016/analysis-001/summary.json"]["scheduled_metric_equal_seed_aggregates"]
series = [
    (old,"mean_projected_history","Spectral: previous high gain","#bb6454","--"),
    (new,"spectral_mean_projected","Spectral: normalized gain","#315e98","-"),
    (new,"scalar_k0","Slow EMA: decay .99","#298267","-"),
    (new,"scalar_k1","Fast EMA: decay .9","#aa862a",":"),
    (old,"k1","Raw SGDm: previous high gain","#7c8188","--"),
]
fig,axes = plt.subplots(1,2,figsize=(10,5.4))
fig.patch.set_facecolor("white")
plotted = []
for ax,target,title in zip(axes,("fixed","clean"),("Fixed-corrupted training labels","Clean training labels")):
    for rows,policy,label,color,style in series:
        chosen = sorted((r for r in rows if r["target"]==target and r["policy"]==policy
                         and r["metric"]=="auxiliary_clean_accuracy"),key=lambda r:r["horizon"])
        assert [r["horizon"] for r in chosen] == [100,250,500,1000,1500,2000]
        for r in chosen:
            assert r["available"] and set(r["per_seed"]) == {"200","201","202"}
            assert abs(sum(r["per_seed"].values())/3-r["equal_seed_mean"])<1e-12
        ax.plot([r["horizon"] for r in chosen],[100*r["equal_seed_mean"] for r in chosen],
                color=color,ls=style,lw=2.1,marker="o",ms=3,label=label)
        plotted.append({"target":target,"policy":policy,"label":label,
                        "horizons":[r["horizon"] for r in chosen],
                        "mean_accuracy_percent":[100*r["equal_seed_mean"] for r in chosen]})
    ax.set(title=title,xlabel="Global training update",ylabel="Auxiliary clean accuracy (%)",
           xlim=(100,2000),ylim=(20,72) if target=="fixed" else (84,95))
    ax.set_xticks([100,500,1000,1500,2000])
    ax.grid(alpha=.16)
    ax.spines[["top","right"]].set_visible(False)
handles,labels = axes[0].get_legend_handles_labels()
fig.legend(handles,labels,loc="lower center",ncol=2,frameon=False,fontsize=9)
fig.suptitle("Gain normalization makes spectral far more competitive",fontsize=14)
fig.text(.5,.19,"Three-seed means; same saved parents and batch plans. Not a fresh-task or equivalence test.",ha="center",fontsize=9,color="#56677a")
fig.tight_layout(rect=(0,.23,1,.96))
out = HERE/"plots-001"
out.mkdir(exist_ok=False)
png = out/"gain-normalization.png"
fig.savefig(png,dpi=150,bbox_inches="tight",facecolor="white")
plt.close(fig)
manifest = {"input_sha256":PINS,"evidence_type":"audited three-seed neural outcomes; report corroboration pending",
            "series":plotted,"png_sha256":{png.name:hashlib.sha256(png.read_bytes()).hexdigest()}}
with (out/"manifest.json").open("x") as f:
    json.dump(manifest,f,indent=2)
    f.write("\n")
print(json.dumps({"png":str(png),"sha256":manifest["png_sha256"][png.name]}))
