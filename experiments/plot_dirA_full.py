#!/usr/bin/env python3
"""Direction A (full) — filter lift across 6 base optimizers, uncrowded.

Left:  heatmap of filter lift (filtered - baseline test acc) over optimizer x noise.
Right: absolute test acc at 90% noise (the decisive regime), baseline vs filter.
Reads adam/sgd/sgdm from dirA_poc and rmsprop/lion/muon from dirA_ext.
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.join(os.path.dirname(__file__), "..", "results", "weight_covariance_v2")
DIRS = {"adam": "dirA_poc", "sgd": "dirA_poc", "sgdm": "dirA_poc",
        "rmsprop": "dirA_ext", "lion": "dirA_ext", "muon": "dirA_ext"}
# vanilla gradient-followers first, geometry-constrained last
ORDER = ["sgd", "sgdm", "adam", "rmsprop", "lion", "muon"]
LABELS = {"sgd": "SGD", "sgdm": "SGDm", "adam": "Adam", "rmsprop": "RMSprop",
          "lion": "Lion", "muon": "Muon (NS)"}
NOISES = ["0.0", "0.5", "0.9"]


def te(opt, kind, n):
    d = os.path.join(BASE, DIRS[opt])
    p = os.path.join(d, f"n{n}_{kind}_{opt}.json")
    return json.load(open(p))["metrics"][-1]["test_acc"]


def main():
    lift = np.array([[te(o, "filt", n) - te(o, "base", n) for n in NOISES] for o in ORDER])

    fig, (axh, axb) = plt.subplots(1, 2, figsize=(13, 5.5),
                                   gridspec_kw={"width_ratios": [1.1, 1]})

    # --- heatmap of lift ---
    vmax = np.abs(lift).max()
    im = axh.imshow(lift, cmap="RdBu", vmin=-vmax, vmax=vmax, aspect="auto")
    axh.set_xticks(range(len(NOISES))); axh.set_xticklabels([f"{n}" for n in NOISES])
    axh.set_yticks(range(len(ORDER))); axh.set_yticklabels([LABELS[o] for o in ORDER])
    axh.set_xlabel("label noise fraction")
    axh.set_title("Filter lift (filtered − baseline test acc)\nblue = filter helps, red = filter hurts")
    for i in range(len(ORDER)):
        for j in range(len(NOISES)):
            axh.text(j, i, f"{lift[i,j]:+.3f}", ha="center", va="center",
                     fontsize=9, color="black")
    axh.axhline(3.5, color="k", lw=1.5)  # divider: vanilla vs geometry-constrained
    axh.text(2.62, 1.5, "gradient\nfollowers", fontsize=7, color="gray", va="center")
    axh.text(2.62, 4.5, "geometry-\nconstrained", fontsize=7, color="gray", va="center")
    fig.colorbar(im, ax=axh, fraction=0.046, label="Δ test acc")

    # --- bars at 90% noise ---
    x = np.arange(len(ORDER)); w = 0.38
    base90 = [te(o, "base", "0.9") for o in ORDER]
    filt90 = [te(o, "filt", "0.9") for o in ORDER]
    axb.bar(x - w/2, base90, w, label="baseline", color="#bdc3c7")
    axb.bar(x + w/2, filt90, w, label="+ filter", color="#e84393")
    axb.set_xticks(x); axb.set_xticklabels([LABELS[o] for o in ORDER], rotation=30, ha="right")
    axb.set_ylabel("test accuracy")
    axb.set_title("At 90% label noise: baseline vs + filter")
    axb.axhline(0.1, ls=":", color="gray", lw=1)
    axb.text(len(ORDER)-0.5, 0.11, "chance", fontsize=7, color="gray", ha="right")
    axb.legend()
    axb.grid(axis="y", alpha=0.3)

    fig.suptitle("Direction A (full) — the filter helps vanilla gradient-followers but "
                 "hurts already-constrained optimizers (Lion, Muon)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    p = os.path.join(BASE, "dirA_full_6optimizers.png")
    fig.savefig(p, dpi=120); plt.close(fig); print("saved", p)


if __name__ == "__main__":
    main()
