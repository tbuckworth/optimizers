#!/usr/bin/env python3
"""Plot Direction A (base-optimizer ablation) and Direction C (soft alpha sweep)."""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.join(os.path.dirname(__file__), "..", "results", "weight_covariance_v2")


def final(path):
    return json.load(open(path))["metrics"][-1]["test_acc"]


def plot_A():
    d = os.path.join(BASE, "dirA_poc")
    noises = [0.0, 0.5, 0.9]
    opts = [("adam", "Adam", "#e67e22"), ("sgd", "SGD", "#2459c4"),
            ("sgdm", "SGDm", "#16a085")]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for key, label, c in opts:
        base = [final(os.path.join(d, f"n{n}_base_{key}.json")) for n in noises]
        filt = [final(os.path.join(d, f"n{n}_filt_{key}.json")) for n in noises]
        ax.plot(noises, base, "o--", color=c, alpha=0.55, label=f"{label} (no filter)")
        ax.plot(noises, filt, "o-", color=c, lw=2.4, label=f"Filter+{label}")
    ax.set_xlabel("label noise fraction")
    ax.set_ylabel("test accuracy (clean test set)")
    ax.set_title("Direction A — filter robustness survives the base optimizer\n"
                 "(solid = with filter, dashed = base optimizer alone)")
    ax.set_xticks(noises)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=3, loc="lower left")
    fig.tight_layout()
    p = os.path.join(BASE, "dirA_base_opt_ablation.png")
    fig.savefig(p, dpi=120); plt.close(fig); print("saved", p)


def plot_C():
    d = os.path.join(BASE, "dirC_poc")
    alphas = [-1, 0, 0.5, 1, 2, 4]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    colors = {"0.0": "#27ae60", "0.5": "#c0392b"}
    for n in ["0.0", "0.5"]:
        soft = [final(os.path.join(d, f"n{n}_soft_a{a}.json")) for a in alphas]
        hard = final(os.path.join(d, f"n{n}_hard.json"))
        ax.plot(alphas, soft, "o-", color=colors[n], lw=2.2,
                label=f"soft, noise={n}")
        ax.axhline(hard, ls=":", color=colors[n], alpha=0.8,
                   label=f"hard top-k, noise={n}")
    ax.axvline(0, color="gray", ls="--", alpha=0.5)
    ax.annotate("α=0\nno filter", (0, ax.get_ylim()[0]), fontsize=7,
                color="gray", ha="center", va="bottom")
    ax.set_xlabel("α  (0 = no filter, 1 = consensus, >1 = dominant dir, <0 = whiten)")
    ax.set_ylabel("test accuracy")
    ax.set_title("Direction C — soft eigenvalue^α weighting\n"
                 "consensus weighting (α≈1) beats both no-filter and hard top-k under noise")
    ax.set_xticks(alphas)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = os.path.join(BASE, "dirC_alpha_sweep.png")
    fig.savefig(p, dpi=120); plt.close(fig); print("saved", p)


if __name__ == "__main__":
    plot_A()
    plot_C()
