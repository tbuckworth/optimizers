"""One-shot resource/source guard for the reviewed 35-new-state measurement."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[2]
UNIT = "spectral-base-grokking-action-measurement-001.service"
TRAINING_UNIT = "spectral-base-grokking-action-001.service"
BATCH = Path("/tmp/spectral-experiment-artifacts/spectral-grokking-action-20260909.ghvEvD")
OUTPUT = BATCH / "measurement-001"
PINS = {
    "experiments/measure_grokking_action_states.py":
        "b84af98413d152a24006e32e599140e04c14926723fe7bbad85f3ef876cc459d",
    "output/2026-09-09-spectral-grokking-action/measurement-protocol.md":
        "56cc9bd7af6f3e909cd250aacc57889f32a72801bf403f76b7e7cf31de82ba49",
    "tests/test_grokking_action_measurement.py":
        "406718f8a41177329d5b65109a26846f95e25df71722affeb391a9e69d727410",
}


def service_properties(unit, keys):
    result = subprocess.check_output(
        ["systemctl", "--user", "show", unit,
         *["--property=" + key for key in keys]], text=True)
    return dict(line.split("=", 1) for line in result.splitlines())


def validate_bounds(cgroup, effective, service):
    if UNIT not in Path(cgroup).parts:
        raise RuntimeError("Not running in the intended measurement service")
    quota, period = effective["cpu.max"].split()
    if (effective["memory.max"] != str(16 * 1024**3)
            or effective["memory.swap.max"] != "0" or quota == "max"
            or int(quota) != int(period) or int(period) <= 0):
        raise RuntimeError("Effective memory/swap/CPU limits differ from protocol")
    if service != {"RuntimeMaxUSec": "20min", "Restart": "no",
                   "KillMode": "control-group", "Type": "exec"}:
        raise RuntimeError("Service lifetime/restart/kill contract differs")


def validate_training_terminal(training):
    if (training.get("MainPID") != "0"
            or training.get("ActiveState") != "inactive"
            or training.get("Result") != "success"):
        raise RuntimeError("Training service is not successfully terminal")


def main():
    if len(sys.argv) != 1:
        raise ValueError("This fixed one-shot guard takes no arguments")
    if OUTPUT.exists() or OUTPUT.is_symlink():
        raise FileExistsError("Measurement output already exists; do not retry")
    if not (BATCH / "batch-complete.json").is_file():
        raise RuntimeError("The full five-seed batch has not completed")
    training = service_properties(TRAINING_UNIT, ("MainPID", "ActiveState", "Result"))
    validate_training_terminal(training)
    for relative, digest in PINS.items():
        if hashlib.sha256((REPO / relative).read_bytes()).hexdigest() != digest:
            raise RuntimeError("Reviewed measurement source/protocol changed")
    lines = Path("/proc/self/cgroup").read_text().splitlines()
    cgroup = next(line.split(":", 2)[2] for line in lines if line.startswith("0::"))
    root = Path("/sys/fs/cgroup") / cgroup.lstrip("/")
    effective = {key: (root / key).read_text().strip()
                 for key in ("memory.max", "memory.swap.max", "cpu.max")}
    service = service_properties(UNIT, ("RuntimeMaxUSec", "Restart", "KillMode", "Type"))
    validate_bounds(cgroup, effective, service)
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS"):
        if os.environ.get(key) != "1":
            raise RuntimeError("CPU math thread environment differs from protocol")
    print(json.dumps({"resource_guard": "pass", "unit": UNIT,
                      "cgroup": cgroup, "effective": effective, "service": service,
                      "training": training, "reviewed_pins": PINS}), flush=True)
    os.execv(sys.executable, [sys.executable,
                             str(REPO / "experiments/measure_grokking_action_states.py"),
                             "--action-batch-dir", str(BATCH),
                             "--output-dir", str(OUTPUT), "--device", "cuda"])


if __name__ == "__main__":
    main()
