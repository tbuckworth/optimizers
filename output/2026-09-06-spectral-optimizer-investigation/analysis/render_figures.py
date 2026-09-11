#!/usr/bin/env python3
"""Render report figures directly from the audited metric artifacts."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DEST = HERE.parent / "paper" / "figures"
DEST.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({
    "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
    "savefig.dpi": 180, "figure.facecolor": "white", "axes.facecolor": "white",
})
COLORS = {"control": "#555B65", "treatment": "#087F8C"}

fig, ax = plt.subplots(figsize=(9, 4.8), layout="constrained")
for method, label, color in [
    ("adam", "Adam", COLORS["control"]),
    ("ours_r200", "Spectral rank 200", COLORS["treatment"]),
]:
    curves = []
    for seed in (42, 43, 44):
        path = ROOT / "results/weight_covariance_v2/noise90_long" / f"{method}_n90_s{seed}.json"
        rows = [r for r in json.loads(path.read_text())["metrics"] if r["epoch"] >= 0]
        epochs = np.array([r["epoch"] + 1 for r in rows])
        values = np.array([100 * r["test_acc"] for r in rows])
        curves.append(values)
        ax.plot(epochs, values, color=color, alpha=.25, lw=.9)
    ax.plot(epochs, np.mean(curves, axis=0), color=color, lw=2.4, label=label)
ax.set(xlabel="Training epoch", ylabel="Clean-test accuracy (%)",
       title="Late memorization is suppressed in the original noisy-MNIST runs",
       xlim=(1, 60), ylim=(20, 100))
ax.legend(frameon=False, loc="lower left")
ax.text(.99, .97, "Three seeds; thin lines are individual runs\n90% uniform relabeling ≈ 81% wrong labels\nHistorical legacy covariance implementation",
        transform=ax.transAxes, ha="right", va="top", fontsize=9)
fig.savefig(DEST / "mnist_noise_trajectories.png")
plt.close(fig)

data = json.loads((HERE / "external-evidence.json").read_text())
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharex=True, sharey=True, layout="constrained")
for ax, key, title in zip(
    axes, ("cifar_global200", "cifar_matrix512"),
    ("Global rank 200, epoch 195", "Per-matrix cap 512, epoch 278"),
):
    study = data[key]
    for arm, label in (("control", "AdamW"), ("treatment", "Spectral")):
        values = study["arms"][arm]
        ax.plot(values["radius_grid"], 100 * np.array(values["robust_accuracy"]),
                marker="o", ms=3, lw=2, label=label, color=COLORS[arm])
    ax.set(title=title, xlabel="L2 attack radius", xlim=(0, .25), ylim=(0, 100))
    ax.text(.96, .93, f"Clean gap: {study['clean_gap_pp']:.3f} pp\nAURAC difference: {study['aurac_delta_mean']:+.6f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=9)
axes[0].set_ylabel("Unconditional robust accuracy (%)")
axes[0].legend(frameon=False)
fig.suptitle("The later near-clean-matched recipe has worse robustness\nSeparate five-pair studies; layout, horizon and selection also differ", fontsize=12)
fig.savefig(DEST / "cifar_robustness_comparison.png")
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(10, 4.3), layout="constrained")
ax = axes[0]
for nuisance in (-2, -1, 1, 2):
    ax.scatter(1, nuisance, color=COLORS["control"], s=28)
ax.axvline(0, color=COLORS["treatment"], lw=3, label="Covariance rank-one space")
ax.quiver(0, 0, 1, 0, angles="xy", scale_units="xy", scale=1, color="#BA4B39", width=.012)
ax.annotate("Constant mean (1, 0)", (1, 0), (1.15, .15), fontsize=9)
ax.annotate("Raw gradients (1, ξ)", (1, 2), (.25, 2.4), fontsize=9)
ax.set(xlim=(-.7, 2.7), ylim=(-2.7, 2.8), xlabel="Useful coordinate",
       ylabel="Nuisance coordinate", title="Centering can exclude a useful mean", aspect="equal")
ax.legend(frameon=False, loc="lower right", fontsize=8)
ax = axes[1]
u = np.array([1., 2.]) / np.sqrt(5)
ax.plot([-1.25*u[0], 1.25*u[0]], [-1.25*u[1], 1.25*u[1]],
        color=COLORS["treatment"], lw=2, label="Projected-gradient space")
step = np.array([-.7, -.7])
proj = u * (u @ step)
ax.quiver(0, 0, *step, angles="xy", scale_units="xy", scale=1,
          color="#BA4B39", width=.012)
ax.plot([step[0], proj[0]], [step[1], proj[1]], "--", color="#BA4B39")
ax.annotate("First Adam update", step, (-1.15, -.4), fontsize=9)
ax.text(.03, .96, "31.62% of update norm lies outside\nthis rank-one gradient space",
        transform=ax.transAxes, va="top", fontsize=9)
ax.set(xlim=(-1.25, 1.05), ylim=(-1.3, 1.3), xlabel="Parameter 1",
       ylabel="Parameter 2", title="Projection before Adam is not update projection", aspect="equal")
ax.legend(frameon=False, loc="lower right", fontsize=8)
fig.savefig(DEST / "geometry_counterexamples.png")
plt.close(fig)
print(json.dumps({"figures": [str(p) for p in sorted(DEST.glob("*.png"))]}, indent=2))
