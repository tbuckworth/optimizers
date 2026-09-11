"""One-off explanatory charts from pinned I16 JSON; no training or tensor loads."""
import hashlib
import json
import math
from pathlib import Path
import shutil
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
I16 = HERE.parents[1] / "continuation/iteration-016"
PINS = {
    "analysis-001/summary.json": "11da55533e66c371132f939ce7e4a3a4495efc0fc5f5b202164d496bb438e401",
    "analysis-001/report-audit.json": "87be955c4f6f01514486969d51414d58b6ac583c0fb68f21cfb74de53ef6e96f",
}


def main():
    loaded = {}
    for name, digest in PINS.items():
        raw = (I16 / name).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == digest, name
        loaded[name] = json.loads(raw)
    summary = loaded["analysis-001/summary.json"]
    assert summary["audit_status"] == "pass"
    assert loaded["analysis-001/report-audit.json"]["status"] == "pass"
    horizons = [100, 250, 500, 1000, 1500, 2000]
    policies = {"k0": ("EMA-only", "#087f8c"),
                "mean_projected_history": ("Spectral candidate", "#8457a6"),
                "k1": ("Raw momentum", "#d88124")}
    all_rows = summary["all_policy_trajectories"]
    expected = {(t, p, s) for t in ("clean", "fixed")
                for p in ("k0", "k0p5", "k0p9", "k1", "mean_projected_history")
                for s in (200, 201, 202)}
    assert len(all_rows) == len(expected) == 30
    assert {(r["target"], r["policy"], r["seed"]) for r in all_rows} == expected
    assert all(r["status"] == "complete" and r["failure"] is None for r in all_rows)
    curves = {}
    for target in ("fixed", "clean"):
        curves[target] = {}
        for policy in policies:
            rows = sorted((r for r in all_rows if r["target"] == target and
                           r["policy"] == policy), key=lambda r: r["seed"])
            assert [r["seed"] for r in rows] == [200, 201, 202]
            assert all([c["horizon"] for c in r["curve"]] == horizons for r in rows)
            values = [[100 * c["values"]["auxiliary_clean_accuracy"]
                       for c in r["curve"]] for r in rows]
            assert all(math.isfinite(x) and 0 <= x <= 100 for row in values for x in row)
            curves[target][policy] = {"seed_accuracy_percent": values,
                                     "mean_accuracy_percent": [sum(v[i] for v in values) / 3
                                                               for i in range(6)]}
        starts = [curves[target][p]["seed_accuracy_percent"] for p in policies]
        assert all([v[0] for v in starts[0]] == [v[0] for v in a] for a in starts[1:])

    output = HERE / "plots"
    output.mkdir(exist_ok=False)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False})
    with tempfile.TemporaryDirectory(prefix="spectral-ema-email-") as directory:
        render = Path(directory)
        fig, ax = plt.subplots(figsize=(9.2, 3.7))
        fig.patch.set_facecolor("#ffffff")
        ages = list(range(401))
        ax.plot(ages, [100 * .01 * .99**j for j in ages], color="#087f8c", lw=2.5,
                label="Tested EMA: older gradients fade gradually")
        ax.step(ages, [1.0 if j < 100 else 0.0 for j in ages], where="post",
                color="#627386", lw=2, linestyle="--", label="Illustration: a 100-step simple average")
        ax.set(xlabel="Age of gradient (updates ago)", ylabel="Weight in average (%)",
               title="What is being averaged? Past gradients, with different weights")
        ax.set_ylim(-.03, 1.12)
        ax.grid(alpha=.18)
        ax.legend(frameon=False, fontsize=10)
        fig.tight_layout()
        fig.savefig(render / "weights.png", dpi=150, facecolor="white")
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(10.3, 4.5))
        fig.patch.set_facecolor("#ffffff")
        for ax, target, title in zip(axes, ("fixed", "clean"),
                                    ("Heavily corrupted training labels", "Clean training labels")):
            for policy, (label, color) in policies.items():
                item = curves[target][policy]
                for values in item["seed_accuracy_percent"]:
                    ax.plot(horizons, values, color=color, lw=.7, alpha=.25)
                ax.plot(horizons, item["mean_accuracy_percent"], color=color,
                        lw=2.5, marker="o", ms=3.5, label=label)
            ax.set(title=title, xlabel="Training update", ylabel="Clean held-out accuracy (%)")
            ax.set_xticks([100, 500, 1000, 1500, 2000])
            ax.grid(alpha=.18)
        axes[0].set_ylim(15, 72)
        axes[1].set_ylim(85.5, 95)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=10)
        fig.suptitle("Measured I16 results: averaging helps here, but is not universally best", fontsize=13)
        fig.tight_layout(rect=[0, .07, 1, .93])
        fig.savefig(render / "evidence.png", dpi=150, facecolor="white")
        plt.close(fig)
        for name in ("weights.png", "evidence.png"):
            shutil.copyfile(render / name, output / name)
    manifest = {
        "schema": "spectral_moving_average_explainer_v1", "input_sha256": PINS,
        "horizons": horizons, "curves": curves,
        "weights_plot": "Analytical kernel illustration, not measured training. EMA infinite-history weights; 100-step SMA not run.",
        "evidence_plot": "All three seeds as thin lines, arithmetic means thick. Fixed checkpoints, no selection. Distinct labeled y ranges. Only three named policies shown; I16 has five.",
        "ema_beta": .99, "mean_age_updates_stationary": 99,
        "weight_half_life_updates": math.log(.5) / math.log(.99),
        "iid_stationary_variance_ratio": 1 / 199,
        "png_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in sorted(output.glob("*.png"))},
    }
    with (output / "manifest.json").open("x") as handle:
        json.dump(manifest, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"status": "complete", "output": str(output),
                      "png_sha256": manifest["png_sha256"]}))


if __name__ == "__main__":
    main()
