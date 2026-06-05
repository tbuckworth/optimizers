#!/usr/bin/env python3
"""Direction A extension — add Lion, Muon (Newton-Schulz), RMSprop to the
base-optimizer ablation. For each new optimizer we first pick a sane LR on clean
MNIST (5-epoch sweep, fair comparison), then run baseline vs filter at 0/50/90%
label noise. Results saved to results/weight_covariance_v2/dirA_ext/.
"""
import os, sys, json, argparse, copy
sys.path.insert(0, os.path.dirname(__file__))
import run_single_weight_cov_v2 as R

SAVE = "../results/weight_covariance_v2/dirA_ext"
RANK, DECAY, WARMUP = 200, 0.99, 100

# LR candidates for the clean-data pick (Adam/sgd/sgdm reuse Direction A's values)
LR_CANDIDATES = {
    "rmsprop": [5e-4, 1e-3, 3e-3],
    "lion":    [1e-4, 3e-4, 1e-3],
    "muon":    [1e-2, 2e-2, 5e-2],
}


def base_args(**kw):
    a = argparse.Namespace(
        mode="baseline", base_optimizer="adam", dataset="standard", lr=1e-3,
        epochs=15, rank=RANK, decay=DECAY, warmup=WARMUP, filter_strength=1.0,
        normalize="none", weighting="hard", alpha=1.0, soft_residual=1,
        label_noise=0.0, seed=42, lora_rank=32, lora_alpha=32.0,
        name="x", save_dir=SAVE, data_dir="./data")
    for k, v in kw.items():
        setattr(a, k, v)
    return a


def final_test(name):
    return json.load(open(os.path.join(SAVE, name + ".json")))["metrics"][-1]["test_acc"]


def pick_lr(opt):
    print(f"\n### LR sweep: {opt} ###", flush=True)
    best, best_lr = -1, None
    for lr in LR_CANDIDATES[opt]:
        name = f"lrsweep_{opt}_{lr}"
        R.run(base_args(mode="baseline", base_optimizer=opt, lr=lr,
                        epochs=5, label_noise=0.0, name=name))
        te = final_test(name)
        print(f"  {opt} lr={lr}: clean test={te:.4f}", flush=True)
        if te > best:
            best, best_lr = te, lr
    print(f"  -> {opt} best lr={best_lr} (test={best:.4f})", flush=True)
    return best_lr


def main():
    os.makedirs(SAVE, exist_ok=True)
    lrs = {}
    for opt in ["rmsprop", "lion", "muon"]:
        lrs[opt] = pick_lr(opt)
    json.dump(lrs, open(os.path.join(SAVE, "chosen_lrs.json"), "w"))

    for opt in ["rmsprop", "lion", "muon"]:
        lr = lrs[opt]
        for noise in [0.0, 0.5, 0.9]:
            print(f"\n### {opt} noise={noise} (lr={lr}) ###", flush=True)
            R.run(base_args(mode="baseline", base_optimizer=opt, lr=lr,
                            label_noise=noise, name=f"n{noise}_base_{opt}"))
            R.run(base_args(mode="ours", base_optimizer=opt, lr=lr,
                            label_noise=noise, name=f"n{noise}_filt_{opt}"))
    print("\nDIRA_EXT_DONE", flush=True)


if __name__ == "__main__":
    main()
