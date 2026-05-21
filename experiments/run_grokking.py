#!/usr/bin/env python3
"""Run one grokking experiment with a specified optimizer.

Usage:
  python3 run_grokking.py --optimizer adamw --lr 1e-3 --wd 1.0 --epochs 100000
  python3 run_grokking.py --optimizer spectral_soft --lr 0.1 --mp_factor 1.5 --epochs 50000
"""

import argparse
import json
import time
import os
import torch
import torch.nn.functional as F

from grokking_model import GrokkingTransformer, get_modular_addition_data
from spectral_optimizer import SpectralConsensusFilter


def make_optimizer(name, model, args):
    """Returns (optimizer, is_spectral)."""
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=args.lr,
                                 weight_decay=args.wd, betas=(0.9, 0.98)), False
    elif name == "adam":
        return torch.optim.Adam(model.parameters(), lr=args.lr,
                                betas=(0.9, 0.98)), False
    elif name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=args.lr,
                               momentum=0.9, weight_decay=args.wd), False
    elif name.startswith("spectral"):
        is_soft = "soft" in name
        if "adamw" in name:
            base = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                     weight_decay=args.wd, betas=(0.9, 0.98))
        elif "adam" in name:
            base = torch.optim.Adam(model.parameters(), lr=args.lr,
                                    betas=(0.9, 0.98))
        else:
            base = torch.optim.SGD(model.parameters(), lr=args.lr,
                                   momentum=0.9)
        return SpectralConsensusFilter(
            model, base,
            mp_factor=args.mp_factor,
            soft=is_soft,
            soft_temp=args.soft_temp,
        ), True
    else:
        raise ValueError(f"Unknown optimizer: {name}")


def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    torch.manual_seed(args.seed)

    train_X, train_y, test_X, test_y = get_modular_addition_data(
        p=args.p, train_frac=args.train_frac, seed=args.seed)
    train_X, train_y = train_X.to(device), train_y.to(device)
    test_X, test_y = test_X.to(device), test_y.to(device)
    print(f"Train: {len(train_X)}, Test: {len(test_X)}, Params: ", end="")

    model = GrokkingTransformer(p=args.p).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{n_params:,}")

    optimizer, is_spectral = make_optimizer(args.optimizer, model, args)
    print(f"Optimizer: {args.optimizer}, spectral={is_spectral}")

    save_dir = os.path.join(args.save_dir, args.name or args.optimizer)
    os.makedirs(save_dir, exist_ok=True)

    metrics = []
    t_start = time.time()

    for epoch in range(args.epochs):
        model.train()
        if is_spectral:
            loss_val, diag = optimizer.step(train_X, train_y)
            with torch.no_grad():
                train_logits = model(train_X)
                train_acc = (train_logits.argmax(-1) == train_y).float().mean().item()
            train_loss = loss_val
        else:
            optimizer.zero_grad()
            train_logits = model(train_X)
            loss = F.cross_entropy(train_logits, train_y)
            loss.backward()
            optimizer.step()
            train_loss = loss.item()
            train_acc = (train_logits.detach().argmax(-1) == train_y).float().mean().item()

        if epoch % args.log_every == 0:
            model.eval()
            with torch.no_grad():
                test_logits = model(test_X)
                test_loss = F.cross_entropy(test_logits, test_y).item()
                test_acc = (test_logits.argmax(-1) == test_y).float().mean().item()

            entry = {
                "epoch": epoch,
                "train_loss": round(train_loss, 6),
                "train_acc": round(train_acc, 6),
                "test_loss": round(test_loss, 6),
                "test_acc": round(test_acc, 6),
                "time_s": round(time.time() - t_start, 1),
            }
            if is_spectral:
                entry["k"] = diag.get("k", 0)
                entry["consensus_ratio"] = round(diag.get("consensus_ratio", 0), 4)
            metrics.append(entry)

            if epoch % (args.log_every * 10) == 0:
                elapsed = time.time() - t_start
                print(f"  ep {epoch:6d}: train_loss={train_loss:.4f} "
                      f"train_acc={train_acc:.4f} test_acc={test_acc:.4f} "
                      f"[{elapsed:.0f}s]")

        if args.save_every > 0 and epoch > 0 and epoch % args.save_every == 0:
            torch.save(model.state_dict(),
                       os.path.join(save_dir, f"checkpoint_{epoch}.pt"))

        if epoch % (args.log_every * 100) == 0 and epoch > 0:
            with open(os.path.join(save_dir, "metrics.json"), "w") as f:
                json.dump({"config": vars(args), "metrics": metrics}, f)

    torch.save(model.state_dict(), os.path.join(save_dir, "checkpoint_final.pt"))
    with open(os.path.join(save_dir, "metrics.json"), "w") as f:
        json.dump({"config": vars(args), "metrics": metrics}, f)

    elapsed = time.time() - t_start
    print(f"\nDone: {args.optimizer} — {elapsed:.0f}s total")
    print(f"  Final: train_acc={train_acc:.4f} test_acc={test_acc:.4f}")
    print(f"  Results: {save_dir}/metrics.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimizer", required=True,
                        choices=["adamw", "adam", "sgd",
                                 "spectral_hard", "spectral_soft",
                                 "spectral_soft_adam", "spectral_hard_adam",
                                 "spectral_soft_adamw", "spectral_hard_adamw"])
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=100000)
    parser.add_argument("--mp_factor", type=float, default=2.0)
    parser.add_argument("--soft_temp", type=float, default=1.0)
    parser.add_argument("--p", type=int, default=113)
    parser.add_argument("--train_frac", type=float, default=0.3)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=5000)
    parser.add_argument("--save_dir", type=str, default="../results/grokking")
    parser.add_argument("--name", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run(args)
