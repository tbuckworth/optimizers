#!/usr/bin/env python3
"""One fixed five-seed raw-direction acquisition; no outcome-driven admission."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.grokking_raw_direction import (
    BATCH_SECONDS, BoundedWriter, POLICY, SEEDS, SEED_SECONDS, STORAGE,
    source_pins, verify_completion,
)

SCHEMA = "grokking_raw_direction_batch_v1"


def command_for(seed, parent):
    if seed not in SEEDS:
        raise ValueError("Seed outside fixed roster")
    command = [sys.executable, str(ROOT / "experiments/grokking_raw_direction.py"),
               "--seed", str(seed), "--output-dir", str(parent / f"seed{seed}")]
    if seed != 100:
        command.extend(["--admission-dir", str(parent / "seed100")])
    return command


def run(parent):
    parent = parent.resolve()
    if (parent.parent != STORAGE.resolve() or not parent.is_dir()
            or list(parent.iterdir())):
        raise ValueError("Need a fresh empty mktemp parent directly under large-volume tmp")
    started = time.monotonic()
    writer = BoundedWriter(parent)
    pins = source_pins()
    metadata = {"schema": SCHEMA, "seeds": list(SEEDS), "policy": POLICY,
                "source_sha256": pins,
                "source_commit": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "first_seed_is_resource_only_admission": True, "paid_spend_usd": 0,
                "batch_seconds_limit": BATCH_SECONDS, "seed_seconds_limit": SEED_SECONDS}
    manifest = writer.json(parent / "batch-manifest.json", metadata)
    accepted, environment, seed = [], None, None
    try:
        for seed in SEEDS:
            remaining = BATCH_SECONDS - (time.monotonic() - started)
            if remaining <= 120:
                raise TimeoutError("Twelve-hour batch deadline; no automatic retry")
            subprocess.run(command_for(seed, parent), check=True, cwd=ROOT,
                           timeout=min(SEED_SECONDS, remaining - 60))
            output = parent / f"seed{seed}"
            completion_path = output / "complete.json"
            if environment is None:
                environment = json.loads(completion_path.read_text())["environment"]
            verify_completion(output, seed, pins, environment)
            if source_pins() != pins:
                raise ValueError("Source changed during fixed batch")
            accepted.append({"seed": seed, **writer.receipt(completion_path)})
        writer.json(parent / "batch-complete.json", {
            **metadata, "status": "complete", "accepted": accepted,
            "batch_manifest": manifest,
            "environment": environment, "elapsed_seconds": time.monotonic() - started})
    except BaseException as error:
        failure = {**metadata, "status": "failed_preserved", "current_seed": seed,
                   "batch_manifest": manifest,
                   "accepted": accepted, "error_type": type(error).__name__,
                   "error": str(error), "elapsed_seconds": time.monotonic() - started}
        print(json.dumps(failure, allow_nan=False), flush=True)
        try:
            writer.json(parent / "batch-failure.json", failure)
        except Exception as receipt_error:
            print(f"Failure receipt unavailable; raw stdout preserves it: {receipt_error}", flush=True)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-parent", type=Path, required=True)
    run(parser.parse_args().output_parent)
