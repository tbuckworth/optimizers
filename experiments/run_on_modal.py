#!/usr/bin/env python3
"""Run base optimizer ablation experiments on Modal.

Usage:
  python3 run_on_modal.py --phase 0          # LR sweep only
  python3 run_on_modal.py --phase 1 --sgd_lr 0.01 --sgdm_lr 0.01  # full ablation
  python3 run_on_modal.py --phase all --sgd_lr 0.01 --sgdm_lr 0.01  # both
"""

import argparse
import json
import os
import modal

app = modal.App("optimizer-ablation")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "torchvision", "numpy")
    .add_local_dir(
        os.path.dirname(os.path.abspath(__file__)),
        remote_path="/root/experiments",
        ignore=lambda path: not str(path).endswith(".py"),
    )
)


@app.function(image=image, gpu="T4", timeout=600)
def run_single(mode, base_optimizer, lr, epochs, label_noise, rank=200,
               decay=0.99, warmup=100, name="run"):
    import subprocess, json, tempfile
    save_dir = tempfile.mkdtemp()
    cmd = [
        "python3", "/root/experiments/run_single_weight_cov_v2.py",
        "--mode", mode,
        "--base_optimizer", base_optimizer,
        "--lr", str(lr),
        "--epochs", str(epochs),
        "--label_noise", str(label_noise),
        "--rank", str(rank),
        "--decay", str(decay),
        "--warmup", str(warmup),
        "--name", name,
        "--save_dir", save_dir,
        "--data_dir", "/tmp/data",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)
    if result.returncode != 0:
        print("STDERR:", result.stderr[-500:])
        return {"error": result.stderr, "name": name}

    result_path = os.path.join(save_dir, f"{name}.json")
    with open(result_path) as f:
        return json.load(f)


def phase0_lr_sweep():
    """Quick LR sweep for SGD variants: 5 epochs, clean MNIST."""
    print("=== Phase 0: LR Sweep ===")
    runs = []
    for lr in [0.005, 0.01, 0.05]:
        for base in ["sgd", "sgdm"]:
            name = f"lr_{base}_{lr}"
            runs.append((name, base, lr))

    handles = []
    for name, base, lr in runs:
        h = run_single.spawn(
            mode="baseline", base_optimizer=base, lr=lr,
            epochs=5, label_noise=0.0, name=name,
        )
        handles.append((name, h))

    results = {}
    for name, h in handles:
        result = h.get()
        results[name] = result
        if "error" not in result:
            final = result["metrics"][-1]
            print(f"  {name}: test={final['test_acc']:.4f} train={final['train_acc']:.4f}")
        else:
            print(f"  {name}: ERROR")

    return results


def phase1_ablation(sgd_lr, sgdm_lr):
    """Full ablation: 6 conditions x 3 noise levels."""
    adam_lr = 0.001
    print(f"\n=== Phase 1: Full Ablation ===")
    print(f"  SGD lr={sgd_lr}, SGDm lr={sgdm_lr}, Adam lr={adam_lr}")

    conditions = [
        ("baseline", "adam",  adam_lr,  "baseline_adam"),
        ("baseline", "sgd",   sgd_lr,  "baseline_sgd"),
        ("baseline", "sgdm",  sgdm_lr, "baseline_sgdm"),
        ("ours",     "adam",  adam_lr,  "filter_adam"),
        ("ours",     "sgd",   sgd_lr,  "filter_sgd"),
        ("ours",     "sgdm",  sgdm_lr, "filter_sgdm"),
    ]
    noises = [0.0, 0.2, 0.4]

    handles = []
    for noise in noises:
        for mode, base, lr, label in conditions:
            name = f"n{noise}_{label}"
            h = run_single.spawn(
                mode=mode, base_optimizer=base, lr=lr,
                epochs=20, label_noise=noise, name=name,
            )
            handles.append((name, h))

    results = {}
    for name, h in handles:
        result = h.get()
        results[name] = result
        if "error" not in result:
            final = result["metrics"][-1]
            print(f"  {name}: test={final['test_acc']:.4f}")
        else:
            print(f"  {name}: ERROR")

    # Print summary table
    print("\n=== SUMMARY ===")
    header = f"{'Noise':<8}"
    labels = [c[3] for c in conditions]
    for l in labels:
        header += f"{l:<14}"
    print(header)
    print("-" * (8 + 14 * len(labels)))

    for noise in noises:
        row = f"{noise:<8}"
        for label in labels:
            name = f"n{noise}_{label}"
            r = results.get(name, {})
            if "error" in r or "metrics" not in r:
                row += f"{'ERR':<14}"
            else:
                row += f"{r['metrics'][-1]['test_acc']:<14.4f}"
        print(row)

    return results


@app.local_entrypoint()
def main(
    phase: str = "0",
    sgd_lr: float = 0.01,
    sgdm_lr: float = 0.01,
):
    save_dir = os.path.join(os.path.dirname(__file__),
                            "..", "results", "weight_covariance_v2", "base_opt_ablation")
    os.makedirs(save_dir, exist_ok=True)

    all_results = {}

    if phase in ("0", "all"):
        lr_results = phase0_lr_sweep()
        lr_dir = os.path.join(save_dir, "lr_sweep")
        os.makedirs(lr_dir, exist_ok=True)
        for name, result in lr_results.items():
            with open(os.path.join(lr_dir, f"{name}.json"), "w") as f:
                json.dump(result, f, indent=2)
        all_results.update(lr_results)

        if phase == "0":
            print("\nPhase 0 done. Check LR sweep results, then run:")
            print(f"  python3 -m modal run run_on_modal.py --phase 1 --sgd-lr <best> --sgdm-lr <best>")
            return

    if phase in ("1", "all"):
        ablation_results = phase1_ablation(sgd_lr, sgdm_lr)
        for name, result in ablation_results.items():
            with open(os.path.join(save_dir, f"{name}.json"), "w") as f:
                json.dump(result, f, indent=2)
        all_results.update(ablation_results)

    print(f"\nAll results saved to {save_dir}")
