"""Plot exact conditional tracking formulas; never reads neural outcomes."""
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

here = Path(__file__).resolve().parent
i17 = here.parents[1] / "continuation/iteration-017"
names = ("covariance-tracking-steelman.md", "check_covariance_tracking.py",
         "covariance-tracking-review.md", "tracking-risk-steelman.md")
pins = {n: hashlib.sha256((i17/n).read_bytes()).hexdigest() for n in names}
route = F(81)*F(3,100)**2 + F(1,19) + F(4,199)
bound = F(5,29)
assert route < bound
lags = [i/20 for i in range(1201)]
risk = [.03**2*l*l+5/(2*l+1) for l in lags]
fig, ax = plt.subplots(figsize=(8.8,4.6))
fig.patch.set_facecolor("#ffffff")
ax.plot(lags, risk, color="#3b638c", lw=2.6, label="One shared EMA decay: bias² + noise")
ax.axhline(float(route), color="#23836a", lw=2.3, label=f"Directional fast/slow: {float(route):.3f}")
ax.axhline(float(bound), color="#b88035", lw=1.6, ls="--", label=f"Proven lower bound for ANY shared decay: {float(bound):.3f}")
ax.set(xlim=(0,60), ylim=(.1,5.25), yscale="log", xlabel="Mean lag of a shared EMA, L = q / (1 − q)", ylabel="Asymptotic tracking MSE (log scale)", title="Different directions can need different averaging speeds")
ax.set_yticks([.15,.3,1,3,5], ["0.15","0.3","1","3","5"])
ax.grid(True, which="major", alpha=.16)
ax.spines[["top","right"]].set_visible(False)
ax.legend(loc="upper center", bbox_to_anchor=(.5,-.19), frameon=False, fontsize=9)
fig.tight_layout()
out = here / "plots"
out.mkdir(exist_ok=False)
png = out / "tracking-risk.png"
fig.savefig(png, dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig)
manifest = {"evidence_type":"own conditional theory; not measured neural results",
            "input_sha256":pins, "routed_risk":float(route), "uniform_lower_bound":float(bound),
            "plot_range_lag":[0,60], "proof_range_lag":"all L >= 0",
            "png_sha256":{"tracking-risk.png":hashlib.sha256(png.read_bytes()).hexdigest()}}
with (out/"manifest.json").open("x") as f:
    json.dump(manifest, f, indent=2)
    f.write("\n")
print(json.dumps(manifest))
