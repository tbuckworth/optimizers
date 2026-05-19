#!/usr/bin/env python3
"""Run all spectral consensus optimizer experiments."""

import json
import time
import os
import sys
import copy
import torch
import torch.nn as nn

from model import MNISTNet, TinyNet
from datasets import get_mnist_loaders, get_spiral_loaders
from custom_optimizers import Lion, Muon
from spectral_optimizer import SpectralConsensusFilter
from train import (train_epoch_standard, train_epoch_spectral,
                   evaluate, compute_epoch_diagnostics)


OPTIMIZER_CONFIGS = {
    "sgd":   {"cls": torch.optim.SGD, "kwargs": {"lr": 0.05, "momentum": 0.9}},
    "adam":   {"cls": torch.optim.Adam, "kwargs": {"lr": 0.001}},
    "adamw":  {"cls": torch.optim.AdamW, "kwargs": {"lr": 0.001, "weight_decay": 0.01}},
    "lion":   {"cls": Lion, "kwargs": {"lr": 3e-4, "weight_decay": 0.01}},
    "muon":   {"cls": Muon, "kwargs": {"lr": 0.02, "momentum": 0.9}},
    "spectral": {"cls": None, "kwargs": {"lr": 0.05, "momentum": 0.9}},
}


def make_optimizer(name, model, device):
    cfg = OPTIMIZER_CONFIGS[name]
    if name == "spectral":
        base_opt = torch.optim.SGD(model.parameters(), **cfg["kwargs"])
        return SpectralConsensusFilter(model, base_opt, variance_threshold=0.8)
    return cfg["cls"](model.parameters(), **cfg["kwargs"])


def run_single_experiment(opt_name, dataset_name, random_labels, epochs, device, seed=42):
    torch.manual_seed(seed)
    batch_size = 64

    if dataset_name == "mnist":
        train_loader, test_loader = get_mnist_loaders(
            batch_size=batch_size, random_labels=random_labels, seed=seed)
        model = MNISTNet().to(device)
        ood_loader = None
    else:
        train_loader, test_loader, ood_loader = get_spiral_loaders(
            batch_size=batch_size, random_labels=random_labels, seed=seed)
        model = TinyNet().to(device)

    optimizer = make_optimizer(opt_name, model, device)
    is_spectral = isinstance(optimizer, SpectralConsensusFilter)
    loss_fn = nn.CrossEntropyLoss()
    results = {"epochs": [], "config": {"optimizer": opt_name, "dataset": dataset_name,
                                         "random_labels": random_labels}}

    for epoch in range(epochs):
        t0 = time.time()

        if is_spectral:
            train_res = train_epoch_spectral(optimizer, train_loader, device)
        else:
            train_res = train_epoch_standard(model, optimizer, train_loader, device)

        test_res = evaluate(model, test_loader, device)
        ood_res = evaluate(model, ood_loader, device) if ood_loader else None

        # Eigenvalue diagnostics for non-spectral optimizers (every 2 epochs to save time)
        epoch_diag = None
        if not is_spectral and epoch % 2 == 0:
            try:
                epoch_diag = compute_epoch_diagnostics(model, train_loader, device, max_batches=3)
            except Exception:
                epoch_diag = None

        epoch_data = {
            "epoch": epoch,
            "train_loss": train_res["train_loss"],
            "train_acc": train_res["train_acc"],
            "test_loss": test_res["loss"],
            "test_acc": test_res["acc"],
            "time_s": time.time() - t0,
        }
        if ood_res:
            epoch_data["ood_loss"] = ood_res["loss"]
            epoch_data["ood_acc"] = ood_res["acc"]
        if is_spectral and "diagnostics" in train_res:
            epoch_data["spectral_diagnostics"] = train_res["diagnostics"]
        if epoch_diag is not None:
            epoch_data["eigenvalue_spectrum"] = epoch_diag

        results["epochs"].append(epoch_data)

        label_str = "random" if random_labels else "real"
        print(f"  [{opt_name}/{dataset_name}/{label_str}] epoch {epoch}: "
              f"train_loss={train_res['train_loss']:.4f} train_acc={train_res['train_acc']:.4f} "
              f"test_acc={test_res['acc']:.4f} ({time.time()-t0:.1f}s)")

    results["final_test_acc"] = results["epochs"][-1]["test_acc"]
    results["final_train_acc"] = results["epochs"][-1]["train_acc"]
    return results


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    epochs_mnist = 10
    epochs_spiral = 30
    all_optimizers = ["sgd", "adam", "adamw", "lion", "muon", "spectral"]

    results = {"device": str(device), "experiments": {}}
    save_path = os.path.join(os.path.dirname(__file__), "..", "results",
                             "spectral_experiment_results.json")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    total = len(all_optimizers) * 2 * 2  # 6 opts x 2 datasets x 2 label modes
    done = 0

    for dataset_name, epochs in [("mnist", epochs_mnist), ("spiral", epochs_spiral)]:
        for random_labels in [False, True]:
            for opt_name in all_optimizers:
                label_str = "random" if random_labels else "real"
                key = f"{opt_name}_{dataset_name}_{label_str}"
                done += 1
                print(f"\n=== [{done}/{total}] {key} ===")

                t_start = time.time()
                try:
                    res = run_single_experiment(
                        opt_name, dataset_name, random_labels, epochs, device)
                    res["total_time_s"] = time.time() - t_start
                    results["experiments"][key] = res
                except Exception as e:
                    print(f"  FAILED: {e}")
                    import traceback
                    traceback.print_exc()
                    results["experiments"][key] = {"error": str(e)}

                # Save intermediate results
                with open(save_path, "w") as f:
                    json.dump(results, f, indent=2)
                print(f"  Saved intermediate results to {save_path}")

    print(f"\n=== Done! Results saved to {save_path} ===")

    # Print summary
    print("\n--- Summary ---")
    for key, res in sorted(results["experiments"].items()):
        if "error" in res:
            print(f"  {key}: FAILED ({res['error']})")
        else:
            print(f"  {key}: train_acc={res['final_train_acc']:.4f} "
                  f"test_acc={res['final_test_acc']:.4f} "
                  f"time={res.get('total_time_s', 0):.1f}s")


if __name__ == "__main__":
    main()
