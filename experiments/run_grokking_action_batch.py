#!/usr/bin/env python3
"""Single fixed-roster acquisition with seed100 resource-only admission."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.grokking_confirmation import file_hash, write_json


def run(parent):
    parent = parent.resolve()
    storage = Path("/tmp/spectral-experiment-artifacts").resolve()
    if parent.parent != storage or not parent.is_dir() or list(parent.iterdir()):
        raise ValueError("Need a fresh empty mktemp parent directly under large-volume tmp")
    started = time.monotonic()
    roster = list(range(100, 105))
    write_json(parent / "batch-manifest.json", {
        "schema": "grokking_action_batch_v1", "seeds": roster,
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "first_seed_is_resource_only_admission": True, "paid_spend_usd": 0})
    accepted = []
    seed = None
    try:
        for seed in roster:
            output = parent / f"seed{seed}"
            command = [sys.executable, str(ROOT / "experiments/grokking_action_intervention.py"),
                       "--seed", str(seed), "--output-dir", str(output)]
            if seed != 100:
                command.extend(["--admission-dir", str(parent / "seed100")])
            subprocess.run(command, check=True, cwd=ROOT, timeout=3 * 3600)
            completion_path = output / "complete.json"
            completion = json.loads(completion_path.read_text())
            if completion["status"] != "complete" or completion["seed"] != seed:
                raise ValueError("Missing complete seed acquisition")
            accepted.append({"seed": seed, "path": str(completion_path),
                             "sha256": file_hash(completion_path)})
        write_json(parent / "batch-complete.json", {
            "status": "complete", "seeds": roster, "accepted": accepted,
            "elapsed_seconds": time.monotonic() - started, "paid_spend_usd": 0})
    except BaseException as error:
        write_json(parent / "batch-failure.json", {
            "status": "failed_preserved", "current_seed": seed, "accepted": accepted,
            "error_type": type(error).__name__, "error": str(error),
            "elapsed_seconds": time.monotonic() - started})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-parent", type=Path, required=True)
    run(parser.parse_args().output_parent)
