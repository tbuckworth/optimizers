#!/usr/bin/env python3
"""HP sweep for spectral consensus on noisy spirals.
Tries different learning rates, mp_factors, and soft_temps to find
settings where spectral saturates properly."""

import json
import time
import os
import itertools
import torch
import torch.nn as nn

from model import BigTinyNet
from datasets import get_noisy_spiral_loaders
from spectral_optimizer import SpectralConsensusFilter
from train import train_epoch_spectral, evaluate


def run_sweep_trial(config, train_loader, test_loader, ood_loader, device, epochs=500, seed=42):
    torch.manual_seed(seed)
    model = BigTinyNet().to(device)
    base_opt = torch.optim.SGD(model.parameters(), lr=config["lr"], momentum=config.get("momentum", 0.9))
    optimizer = SpectralConsensusFilter(
        model, base_opt,
        mp_factor=config.get("mp_factor", 2.0),
        soft=config.get("soft", False),
        soft_temp=config.get("soft_temp", 0.5),
    )

    best_test = 0
    best_ood = 0
    history = []

    for epoch in range(epochs):
        train_res = train_epoch_spectral(optimizer, train_loader, device)
        test_res = evaluate(model, test_loader, device)
        ood_res = evaluate(model, ood_loader, device)

        if test_res["acc"] > best_test:
            best_test = test_res["acc"]
        if ood_res["acc"] > best_ood:
            best_ood = ood_res["acc"]

        history.append({
            "epoch": epoch,
            "train_loss": train_res["train_loss"],
            "train_acc": train_res["train_acc"],
            "test_loss": test_res["loss"],
            "test_acc": test_res["acc"],
            "ood_loss": ood_res["loss"],
            "ood_acc": ood_res["acc"],
        })

        if epoch % 50 == 0 or epoch == epochs - 1:
            print(f"    ep {epoch}: train={train_res['train_acc']:.3f} "
                  f"test={test_res['acc']:.3f} ood={ood_res['acc']:.3f} "
                  f"loss={train_res['train_loss']:.4f}")

    final = history[-1]
    return {
        "config": config,
        "epochs": history,
        "final_train_acc": final["train_acc"],
        "final_test_acc": final["test_acc"],
        "final_ood_acc": final["ood_acc"],
        "best_test_acc": best_test,
        "best_ood_acc": best_ood,
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, test_loader, ood_loader = get_noisy_spiral_loaders(
        n_train=400, train_noise=0.8, batch_size=64, seed=42)

    # Also run standard SGD and Adam baselines for reference
    from train import train_epoch_standard
    baselines = {}
    for opt_name, opt_cls, opt_kwargs in [
        ("sgd", torch.optim.SGD, {"lr": 0.05, "momentum": 0.9}),
        ("adam", torch.optim.Adam, {"lr": 0.001}),
    ]:
        print(f"\n=== Baseline: {opt_name} ===")
        torch.manual_seed(42)
        model = BigTinyNet().to(device)
        opt = opt_cls(model.parameters(), **opt_kwargs)
        history = []
        for epoch in range(500):
            train_res = train_epoch_standard(model, opt, train_loader, device)
            test_res = evaluate(model, test_loader, device)
            ood_res = evaluate(model, ood_loader, device)
            history.append({
                "epoch": epoch, "train_acc": train_res["train_acc"],
                "test_acc": test_res["acc"], "ood_acc": ood_res["acc"],
                "train_loss": train_res["train_loss"],
                "test_loss": test_res["loss"], "ood_loss": ood_res["loss"],
            })
            if epoch % 100 == 0 or epoch == 499:
                print(f"  ep {epoch}: train={train_res['train_acc']:.3f} "
                      f"test={test_res['acc']:.3f} ood={ood_res['acc']:.3f}")
        baselines[opt_name] = {
            "config": {"optimizer": opt_name, **opt_kwargs},
            "epochs": history,
            "final_train_acc": history[-1]["train_acc"],
            "final_test_acc": history[-1]["test_acc"],
            "final_ood_acc": history[-1]["ood_acc"],
        }

    # Spectral sweep grid
    configs = []
    for lr in [0.005, 0.01, 0.02, 0.05]:
        for mp_factor in [1.5, 2.0, 3.0]:
            configs.append({"lr": lr, "mp_factor": mp_factor, "soft": False,
                            "name": f"hard_lr{lr}_mp{mp_factor}"})
    for lr in [0.005, 0.01, 0.02, 0.05]:
        for mp_factor in [1.5, 2.0, 3.0]:
            for soft_temp in [0.3, 0.5, 1.0]:
                configs.append({"lr": lr, "mp_factor": mp_factor, "soft": True,
                                "soft_temp": soft_temp,
                                "name": f"soft_lr{lr}_mp{mp_factor}_t{soft_temp}"})

    results = {"baselines": baselines, "sweep": {}}
    save_path = os.path.join(os.path.dirname(__file__), "..", "results",
                             "spectral_spiral_sweep.json")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    total = len(configs)
    for i, cfg in enumerate(configs):
        name = cfg.pop("name")
        print(f"\n=== [{i+1}/{total}] {name} ===")
        print(f"  Config: {cfg}")
        t0 = time.time()
        try:
            res = run_sweep_trial(cfg, train_loader, test_loader, ood_loader,
                                  device, epochs=500)
            res["total_time_s"] = time.time() - t0
            results["sweep"][name] = res
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback; traceback.print_exc()
            results["sweep"][name] = {"error": str(e), "config": cfg}

        with open(save_path, "w") as f:
            json.dump(results, f, indent=2)

    print(f"\n=== Done! Results: {save_path} ===")

    # Print top results
    print("\n--- Top 10 by test accuracy ---")
    valid = [(k, v) for k, v in results["sweep"].items() if "error" not in v]
    valid.sort(key=lambda x: x[1]["best_test_acc"], reverse=True)
    for name, res in valid[:10]:
        print(f"  {name:40s} best_test={res['best_test_acc']:.3f} "
              f"best_ood={res['best_ood_acc']:.3f} "
              f"final_train={res['final_train_acc']:.3f}")

    print("\n--- Top 10 by OOD accuracy ---")
    valid.sort(key=lambda x: x[1]["best_ood_acc"], reverse=True)
    for name, res in valid[:10]:
        print(f"  {name:40s} best_ood={res['best_ood_acc']:.3f} "
              f"best_test={res['best_test_acc']:.3f} "
              f"final_train={res['final_train_acc']:.3f}")


if __name__ == "__main__":
    main()
