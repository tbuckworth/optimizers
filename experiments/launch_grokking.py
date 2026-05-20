#!/usr/bin/env python3
"""Launch all grokking experiments. Modes: timing, sweep, full."""

import subprocess
import sys
import os
import time
import json
import argparse

RESULTS_DIR = "../results/grokking"


def build_cmd(cfg):
    parts = ["python3", "-u", "run_grokking.py"]
    for k, v in cfg.items():
        parts.append(f"--{k}")
        parts.append(str(v))
    return " ".join(parts)


def run_sequential(configs, label=""):
    """Run configs one at a time (safe for GPU memory)."""
    for i, cfg in enumerate(configs):
        name = cfg.get("name", cfg["optimizer"])
        print(f"\n[{i+1}/{len(configs)}] {name}")
        cmd = build_cmd(cfg)
        t0 = time.time()
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        elapsed = time.time() - t0
        if result.returncode != 0:
            print(f"  FAILED ({elapsed:.0f}s): {result.stderr[-200:]}")
        else:
            last_lines = result.stdout.strip().split("\n")[-3:]
            for line in last_lines:
                print(f"  {line}")
            print(f"  ({elapsed:.0f}s)")


def run_timing_test():
    """Quick 50-step test to measure per-step time."""
    print("=== Timing test (50 steps) ===")
    configs = []
    for opt in ["adamw", "spectral_soft", "spectral_hard"]:
        cfg = {"optimizer": opt, "lr": 1e-3, "epochs": 50,
               "log_every": 10, "save_every": 0, "name": f"timing_{opt}"}
        if "spectral" in opt:
            cfg["mp_factor"] = 2.0
            cfg["soft_temp"] = 1.0
        configs.append(cfg)
    run_sequential(configs)


def run_sweep():
    """Lean HP sweep: 8 key spectral configs at 2000 steps."""
    configs = [
        # Spectral soft with SGD base — vary LR and mp_factor
        {"optimizer": "spectral_soft", "lr": 0.1, "mp_factor": 1.5,
         "soft_temp": 1.0, "epochs": 2000, "log_every": 10,
         "save_every": 0, "name": "sweep_soft_lr0.1_mp1.5"},
        {"optimizer": "spectral_soft", "lr": 0.3, "mp_factor": 1.5,
         "soft_temp": 1.0, "epochs": 2000, "log_every": 10,
         "save_every": 0, "name": "sweep_soft_lr0.3_mp1.5"},
        {"optimizer": "spectral_soft", "lr": 1.0, "mp_factor": 1.5,
         "soft_temp": 1.0, "epochs": 2000, "log_every": 10,
         "save_every": 0, "name": "sweep_soft_lr1.0_mp1.5"},
        # Spectral hard with SGD base
        {"optimizer": "spectral_hard", "lr": 0.1, "mp_factor": 2.0,
         "epochs": 2000, "log_every": 10,
         "save_every": 0, "name": "sweep_hard_lr0.1_mp2.0"},
        {"optimizer": "spectral_hard", "lr": 0.3, "mp_factor": 2.0,
         "epochs": 2000, "log_every": 10,
         "save_every": 0, "name": "sweep_hard_lr0.3_mp2.0"},
        {"optimizer": "spectral_hard", "lr": 1.0, "mp_factor": 2.0,
         "epochs": 2000, "log_every": 10,
         "save_every": 0, "name": "sweep_hard_lr1.0_mp2.0"},
        # Spectral soft with Adam base
        {"optimizer": "spectral_soft_adam", "lr": 1e-3, "mp_factor": 1.5,
         "soft_temp": 1.0, "epochs": 2000, "log_every": 10,
         "save_every": 0, "name": "sweep_soft_adam_lr1e-3_mp1.5"},
        {"optimizer": "spectral_soft_adam", "lr": 3e-3, "mp_factor": 1.5,
         "soft_temp": 1.0, "epochs": 2000, "log_every": 10,
         "save_every": 0, "name": "sweep_soft_adam_lr3e-3_mp1.5"},
    ]
    print(f"=== HP Sweep ({len(configs)} configs, sequential) ===")
    run_sequential(configs)

    # Print summary
    print("\n=== Sweep Summary ===")
    for cfg in configs:
        name = cfg["name"]
        metrics_path = os.path.join(RESULTS_DIR, name, "metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                data = json.load(f)
            m = data["metrics"]
            if m:
                last = m[-1]
                print(f"  {name:40s} train_acc={last['train_acc']:.4f} "
                      f"test_acc={last['test_acc']:.4f}")


def run_full(spectral_lr=0.3, spectral_mp=1.5, adam_lr=1e-3, adam_mp=1.5):
    """Launch full experiment as nohup background processes (max 2 spectral)."""
    log_dir = os.path.join(RESULTS_DIR, "logs")
    os.makedirs(log_dir, exist_ok=True)

    # Standard optimizers: fast, can all run at once
    standard = [
        {"optimizer": "adamw", "lr": 1e-3, "wd": 1.0, "epochs": 100000,
         "name": "adamw_wd1"},
        {"optimizer": "adam", "lr": 1e-3, "wd": 0.0, "epochs": 100000,
         "name": "adam_nowd"},
        {"optimizer": "sgd", "lr": 1.0, "wd": 0.0, "epochs": 100000,
         "name": "sgd_nowd"},
        {"optimizer": "sgd", "lr": 1.0, "wd": 1.0, "epochs": 100000,
         "name": "sgd_wd1"},
    ]

    # Spectral: must run max 2 at a time (10.6 GB each)
    spectral = [
        {"optimizer": "spectral_soft", "lr": spectral_lr,
         "mp_factor": spectral_mp, "soft_temp": 1.0,
         "epochs": 50000, "name": "spectral_soft_sgd"},
        {"optimizer": "spectral_hard", "lr": spectral_lr,
         "mp_factor": spectral_mp,
         "epochs": 50000, "name": "spectral_hard_sgd"},
        {"optimizer": "spectral_soft_adam", "lr": adam_lr,
         "mp_factor": adam_mp, "soft_temp": 1.0,
         "epochs": 50000, "name": "spectral_soft_adam"},
    ]

    print("=== Launching standard optimizers (all at once) ===")
    for cfg in standard:
        name = cfg["name"]
        log_file = os.path.join(log_dir, f"{name}.log")
        cmd = build_cmd(cfg)
        print(f"  {name}")
        subprocess.Popen(f"nohup {cmd} > {log_file} 2>&1 &", shell=True)
        time.sleep(0.3)

    print("\n=== Launching spectral optimizers (sequential pairs via wrapper) ===")
    # Write a small shell script that runs spectral configs sequentially
    script_lines = ["#!/bin/bash", "cd ~/pyg/optimizers/experiments"]
    for cfg in spectral:
        name = cfg["name"]
        log_file = os.path.join(log_dir, f"{name}.log")
        cmd = build_cmd(cfg)
        script_lines.append(f'echo "Starting {name}..."')
        script_lines.append(f"{cmd} > {log_file} 2>&1")
        script_lines.append(f'echo "Finished {name}"')

    script_path = os.path.join(RESULTS_DIR, "run_spectral.sh")
    with open(script_path, "w") as f:
        f.write("\n".join(script_lines) + "\n")
    os.chmod(script_path, 0o755)

    subprocess.Popen(
        f"nohup bash {script_path} > {os.path.join(log_dir, 'spectral_all.log')} 2>&1 &",
        shell=True)
    print("  Spectral configs will run sequentially (to avoid OOM)")
    for cfg in spectral:
        print(f"    {cfg['name']} ({cfg['epochs']} epochs)")

    print(f"\nMonitor: tail -f {log_dir}/*.log")
    print("Standard runs: ~1.75 hrs each")
    print("Spectral runs: ~10 hrs each, ~30 hrs total (sequential)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["timing", "sweep", "full"])
    parser.add_argument("--spectral_lr", type=float, default=0.3)
    parser.add_argument("--spectral_mp", type=float, default=1.5)
    parser.add_argument("--adam_lr", type=float, default=1e-3)
    parser.add_argument("--adam_mp", type=float, default=1.5)
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    if args.mode == "timing":
        run_timing_test()
    elif args.mode == "sweep":
        run_sweep()
    elif args.mode == "full":
        run_full(args.spectral_lr, args.spectral_mp, args.adam_lr, args.adam_mp)
