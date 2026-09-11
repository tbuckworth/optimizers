"""One-shot guard: measure only this completed batch's fifteen new states."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[2]
UNIT = "spectral-base-grokking-raw-direction-measurement-001.service"
TRAINING_UNIT = "spectral-base-grokking-raw-direction-001.service"
BATCH = Path("/tmp/spectral-experiment-artifacts/spectral-grokking-raw-direction-20260909.KzGtkl")
OUTPUT = BATCH / "measurement-001"


def service_properties(unit, keys):
    result = subprocess.check_output(
        ["systemctl", "--user", "show", unit,
         *["--property=" + key for key in keys]], text=True)
    return dict(line.split("=", 1) for line in result.splitlines())


def validate_bounds(cgroup, effective, service):
    if Path(cgroup).name != UNIT:
        raise RuntimeError("Not inside the intended measurement service")
    quota, period = effective["cpu.max"].split()
    if (effective["memory.max"] != str(16 * 1024**3)
            or effective["memory.swap.max"] != "0" or quota == "max"
            or int(quota) != int(period) or int(period) <= 0):
        raise RuntimeError("Effective memory/swap/CPU limits differ from protocol")
    if service != {"RuntimeMaxUSec": "20min", "Restart": "no",
                   "KillMode": "control-group", "Type": "exec"}:
        raise RuntimeError("Service lifetime/restart/kill/type contract differs")


def validate_training_terminal(training):
    if (training.get("MainPID") != "0" or training.get("ActiveState") != "inactive"
            or training.get("Result") != "success"):
        raise RuntimeError("Training is not successfully terminal")


def main():
    if len(sys.argv) != 1:
        raise ValueError("This fixed one-shot guard accepts no arguments")
    if OUTPUT.exists() or OUTPUT.is_symlink():
        raise FileExistsError("New measurement output exists; do not retry")
    if not (BATCH / "batch-complete.json").is_file():
        raise RuntimeError("Full raw-direction batch has not completed")
    training = service_properties(TRAINING_UNIT, ("MainPID", "ActiveState", "Result"))
    validate_training_terminal(training)
    if Path(sys.executable).resolve() != Path("/usr/bin/python3.12").resolve():
        raise RuntimeError("Unexpected measurement interpreter")
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        if os.environ.get(key) != "1":
            raise RuntimeError("CPU math environment is not single-threaded")
    cgroup = next(line.split(":", 2)[2] for line in Path("/proc/self/cgroup").read_text().splitlines()
                  if line.startswith("0::"))
    root = Path("/sys/fs/cgroup") / cgroup.lstrip("/")
    effective = {key: (root / key).read_text().strip()
                 for key in ("memory.max", "memory.swap.max", "cpu.max")}
    service = service_properties(UNIT, ("RuntimeMaxUSec", "Restart", "KillMode", "Type"))
    validate_bounds(cgroup, effective, service)
    if os.stat(BATCH).st_dev == os.stat("/").st_dev:
        raise RuntimeError("Measurement would use the root filesystem")
    sys.path.insert(0, str(REPO))
    from experiments.measure_grokking_raw_direction_states import measurement_sources
    pins = measurement_sources()
    for path in ("output/2026-09-09-spectral-raw-direction/guarded_measurement.py",
                 "tests/test_grokking_raw_direction_measurement_guard.py"):
        pins[path] = hashlib.sha256((REPO / path).read_bytes()).hexdigest()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    for path, expected in pins.items():
        committed = subprocess.check_output(["git", "show", f"{commit}:{path}"], cwd=REPO)
        if hashlib.sha256(committed).hexdigest() != expected:
            raise RuntimeError("Measurement scientific/guard sources differ from committed HEAD")
    print(json.dumps({"resource_guard": "pass", "unit": UNIT, "pid": os.getpid(),
                      "cgroup": cgroup, "effective": effective, "service": service,
                      "training": training, "source_commit": commit, "source_sha256": pins,
                      "batch": str(BATCH), "output": str(OUTPUT),
                      "interpreter": str(Path(sys.executable).resolve())}), flush=True)
    os.execv(sys.executable, [sys.executable,
                             str(REPO / "experiments/measure_grokking_raw_direction_states.py"),
                             "--raw-batch-dir", str(BATCH), "--output-dir", str(OUTPUT)])


if __name__ == "__main__":
    main()
