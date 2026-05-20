#!/usr/bin/env python3
"""OOD generalization experiment: spectral consensus vs standard optimizers.

Two setups designed so memorization is easy and visible:
1. Small MNIST (500 train, 235K params = 470x overparameterized)
2. Noisy spirals (400 train, ~50K params, high noise)
"""

import json
import time
import os
import torch
import torch.nn as nn

from model import MNISTNet, BigTinyNet
from datasets import get_mnist_small_loaders, get_noisy_spiral_loaders
from custom_optimizers import Lion, Muon
from spectral_optimizer import SpectralConsensusFilter
from train import (train_epoch_standard, train_epoch_spectral,
                   evaluate)


OPTIMIZER_CONFIGS = {
    "sgd":           {"cls": torch.optim.SGD,  "kwargs": {"lr": 0.05, "momentum": 0.9}},
    "adam":          {"cls": torch.optim.Adam,  "kwargs": {"lr": 0.001}},
    "adamw":         {"cls": torch.optim.AdamW, "kwargs": {"lr": 0.001, "weight_decay": 0.01}},
    "spectral":      {"cls": None, "kwargs": {"lr": 0.05, "momentum": 0.9}},
    "spectral_soft": {"cls": None, "kwargs": {"lr": 0.05, "momentum": 0.9}},
}


def make_optimizer(name, model, device):
    cfg = OPTIMIZER_CONFIGS[name]
    if name == "spectral":
        base = torch.optim.SGD(model.parameters(), **cfg["kwargs"])
        return SpectralConsensusFilter(model, base, mp_factor=2.0)
    if name == "spectral_soft":
        base = torch.optim.SGD(model.parameters(), **cfg["kwargs"])
        return SpectralConsensusFilter(model, base, mp_factor=2.0, soft=True, soft_temp=0.5)
    return cfg["cls"](model.parameters(), **cfg["kwargs"])


def run_one(opt_name, dataset, train_loader, test_loader, ood_loader, model, device, epochs, seed=42):
    torch.manual_seed(seed)
    optimizer = make_optimizer(opt_name, model, device)
    is_spectral = isinstance(optimizer, SpectralConsensusFilter)
    results = {"epochs": [], "config": {"optimizer": opt_name, "dataset": dataset}}

    for epoch in range(epochs):
        t0 = time.time()
        if is_spectral:
            train_res = train_epoch_spectral(optimizer, train_loader, device)
        else:
            train_res = train_epoch_standard(model, optimizer, train_loader, device)

        test_res = evaluate(model, test_loader, device)
        ood_res = evaluate(model, ood_loader, device)

        epoch_data = {
            "epoch": epoch,
            "train_loss": train_res["train_loss"],
            "train_acc": train_res["train_acc"],
            "test_loss": test_res["loss"],
            "test_acc": test_res["acc"],
            "ood_loss": ood_res["loss"],
            "ood_acc": ood_res["acc"],
            "time_s": time.time() - t0,
        }
        if is_spectral and "diagnostics" in train_res:
            epoch_data["spectral_diagnostics"] = train_res["diagnostics"]

        results["epochs"].append(epoch_data)
        print(f"  [{opt_name}/{dataset}] ep {epoch}: "
              f"train={train_res['train_acc']:.3f} test={test_res['acc']:.3f} "
              f"ood={ood_res['acc']:.3f} ({time.time()-t0:.1f}s)")

    results["final_train_acc"] = results["epochs"][-1]["train_acc"]
    results["final_test_acc"] = results["epochs"][-1]["test_acc"]
    results["final_ood_acc"] = results["epochs"][-1]["ood_acc"]
    return results


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    all_opts = list(OPTIMIZER_CONFIGS.keys())
    all_results = {"device": str(device), "experiments": {}}
    save_path = os.path.join(os.path.dirname(__file__), "..", "results",
                             "ood_experiment_results.json")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # === Setup 1: Small MNIST ===
    print("\n========== SMALL MNIST (500 train) ==========")
    mnist_train, mnist_test, mnist_ood = get_mnist_small_loaders(
        n_train=500, batch_size=64, seed=42)
    for opt_name in all_opts:
        key = f"{opt_name}_mnist_small"
        print(f"\n--- {key} ---")
        model = MNISTNet().to(device)
        torch.manual_seed(42)
        t0 = time.time()
        try:
            res = run_one(opt_name, "mnist_small", mnist_train, mnist_test,
                          mnist_ood, model, device, epochs=100)
            res["total_time_s"] = time.time() - t0
            all_results["experiments"][key] = res
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback; traceback.print_exc()
            all_results["experiments"][key] = {"error": str(e)}

        with open(save_path, "w") as f:
            json.dump(all_results, f, indent=2)

    # === Setup 2: Noisy Spirals ===
    print("\n========== NOISY SPIRALS (400 train, noise=0.8) ==========")
    spiral_train, spiral_test, spiral_ood = get_noisy_spiral_loaders(
        n_train=400, train_noise=0.8, batch_size=64, seed=42)
    for opt_name in all_opts:
        key = f"{opt_name}_spiral_noisy"
        print(f"\n--- {key} ---")
        model = BigTinyNet().to(device)
        torch.manual_seed(42)
        t0 = time.time()
        try:
            res = run_one(opt_name, "spiral_noisy", spiral_train, spiral_test,
                          spiral_ood, model, device, epochs=200)
            res["total_time_s"] = time.time() - t0
            all_results["experiments"][key] = res
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback; traceback.print_exc()
            all_results["experiments"][key] = {"error": str(e)}

        with open(save_path, "w") as f:
            json.dump(all_results, f, indent=2)

    print(f"\n=== Done! Results: {save_path} ===")
    print("\n--- Summary ---")
    for key, res in sorted(all_results["experiments"].items()):
        if "error" in res:
            print(f"  {key}: FAILED")
        else:
            print(f"  {key}: train={res['final_train_acc']:.3f} "
                  f"test={res['final_test_acc']:.3f} "
                  f"ood={res['final_ood_acc']:.3f} "
                  f"time={res.get('total_time_s',0):.1f}s")


if __name__ == "__main__":
    main()
