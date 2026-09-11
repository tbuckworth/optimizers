"""Bounded sequential execution of the registered grokking confirmation roster.

Use under a systemd service with an outer wall-time and memory ceiling. This
launcher never resumes or overwrites an existing acquisition directory.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
SOURCE_FILES = ("experiments/launch_grokking_confirmation.py",
                "experiments/grokking_confirmation.py",
                "experiments/grokking_model.py", "spectral_filter.py",
                "tests/test_grokking_confirmation.py",
                "output/2026-09-08-spectral-paper-planning/grokking-confirmation-protocol.md")
ARMS = ("adamw", "legacy", "stable")


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def source_hashes():
    return {name: hashlib.sha256((REPO / name).read_bytes()).hexdigest()
            for name in SOURCE_FILES}


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, choices=range(100, 105), required=True)
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("Repeated seed is not permitted.")
    output = args.output_dir.resolve()
    if not output.is_relative_to(Path("/tmp/spectral-experiment-artifacts")):
        parser.error("Scientific output must be below /tmp/spectral-experiment-artifacts.")
    pins = source_hashes()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    output.mkdir(parents=False, exist_ok=False)
    manifest = {"created_utc": utc(), "source_commit": commit, "source_sha256": pins,
                "seeds": args.seeds, "arms": ARMS, "steps": 6000,
                "per_arm_timeout_seconds": 3600,
                "protocol": "output/2026-09-08-spectral-paper-planning/grokking-confirmation-protocol.md",
                "parent_pid": os.getpid(), "restart_policy": "never automatically restart"}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    environment = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                   "OPENBLAS_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1"}
    accepted = []
    with (output / "events.jsonl").open("x") as events:
        for seed in args.seeds:
            pair_dir = output / f"seed{seed}"
            pair_dir.mkdir()
            for arm in ARMS:
                if (output / "STOP").exists():
                    raise SystemExit("STOP marker present; no further arm will start.")
                if source_hashes() != pins:
                    raise SystemExit("Source changed since registration of this batch.")
                command = [sys.executable, str(REPO / "experiments/grokking_confirmation.py"),
                           "--arm", arm, "--seed", str(seed), "--steps", "6000",
                           "--device", "cuda", "--output-dir", str(pair_dir / arm),
                           "--threads", "1", "--checkpoint-steps",
                           "0,100,500,1000,1500,2000,2500,3000,4000,6000"]
                start = {"event": "start", "utc": utc(), "seed": seed, "arm": arm,
                         "command": command}
                events.write(json.dumps(start) + "\n")
                events.flush()
                print(json.dumps(start), flush=True)
                timer = time.monotonic()
                try:
                    completed = subprocess.run(command, cwd=REPO, env=environment, timeout=3600,
                                               check=False)
                    code = completed.returncode
                except subprocess.TimeoutExpired:
                    code = 124
                end = {"event": "end", "utc": utc(), "seed": seed, "arm": arm,
                       "exit_code": code, "elapsed_seconds": time.monotonic() - timer}
                events.write(json.dumps(end) + "\n")
                events.flush()
                print(json.dumps(end), flush=True)
                if code:
                    raise SystemExit(f"Arm failed ({code}); preserve artifacts and diagnose.")
                if source_hashes() != pins:
                    raise SystemExit("Source changed during the arm; preserve output and diagnose.")
                result_path = pair_dir / arm / "metrics.json"
                if not result_path.is_file() or result_path.is_symlink():
                    raise SystemExit("Missing regular metrics.json after successful child exit.")
                result = json.loads(result_path.read_text())
                expected_steps = list(range(0, 6001, 50))
                if (result.get("status") != "complete" or result.get("completed_steps") != 6000
                        or [row["step"] for row in result.get("evaluation_rows", [])] != expected_steps
                        or result["config"]["seed"] != seed or result["config"]["arm"] != arm):
                    raise SystemExit("Completed result does not match the registered arm/grid.")
                checkpoints = result.get("checkpoints", [])
                expected_names = {f"checkpoint-step-{step:06d}.pt" for step in
                                  (0, 100, 500, 1000, 1500, 2000, 2500, 3000, 4000, 6000)}
                if len(checkpoints) != 10 or {Path(item["path"]).name for item in checkpoints} != expected_names:
                    raise SystemExit("Checkpoint roster mismatch.")
                for item in checkpoints:
                    path = Path(item["path"])
                    if (path.parent != pair_dir / arm or not path.is_file() or path.is_symlink()
                            or path.stat().st_size != item["size_bytes"]
                            or re.fullmatch(r"[0-9a-f]{64}", item["sha256"]) is None
                            or file_hash(path) != item["sha256"]):
                        raise SystemExit("Checkpoint receipt/file mismatch.")
                accepted.append({"seed": seed, "arm": arm, "metrics_path": str(result_path),
                                 "metrics_sha256": hashlib.sha256(result_path.read_bytes()).hexdigest(),
                                 "checkpoint_receipts": checkpoints})
    if source_hashes() != pins:
        raise SystemExit("Source changed before batch completion.")
    (output / "complete.json").write_text(json.dumps({"completed_utc": utc(),
        "seeds": args.seeds, "arms": ARMS, "source_sha256": pins,
        "accepted_results": accepted}, indent=2) + "\n")


if __name__ == "__main__":
    main()
