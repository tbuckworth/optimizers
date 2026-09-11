"""Fixed one-shot common-state diagnostic under verified service limits."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[2]
UNIT = "spectral-base-grokking-function-response-001.service"
PARENT = Path("/tmp/spectral-experiment-artifacts/spectral-grokking-function-response-20260909.PegeVa")
OUTPUT = PARENT / "diagnostic-001"
TERMINAL_UNITS = (
    "spectral-base-grokking-raw-direction-001.service",
    "spectral-base-grokking-raw-direction-measurement-001.service",
    "spectral-base-grokking-raw-direction-analysis-002.service",
)


def service_properties(unit, keys):
    result = subprocess.check_output(["systemctl", "--user", "show", unit,
        *["--property=" + key for key in keys]], text=True)
    return dict(line.split("=", 1) for line in result.splitlines())


def validate_bounds(cgroup, effective, service):
    quota, period = effective["cpu.max"].split()
    if (Path(cgroup).name != UNIT or effective["memory.max"] != str(16 * 1024**3)
            or effective["memory.swap.max"] != "0" or quota == "max"
            or int(quota) != int(period) or int(period) <= 0
            or service != {"RuntimeMaxUSec": "10min", "Restart": "no",
                           "KillMode": "control-group", "Type": "exec"}):
        raise RuntimeError("Effective diagnostic bounds differ from the protocol")


def validate_terminal(value):
    if value != {"MainPID": "0", "ActiveState": "inactive", "Result": "success"}:
        raise RuntimeError("Prior experiment is not successfully terminal")


def main():
    if len(sys.argv) != 1:
        raise ValueError("Fixed one-shot guard accepts no arguments")
    if OUTPUT.exists() or OUTPUT.is_symlink():
        raise FileExistsError("Diagnostic already attempted; do not retry")
    if (PARENT.is_symlink() or not PARENT.is_dir() or list(PARENT.iterdir())
            or os.stat(PARENT).st_dev == os.stat("/").st_dev):
        raise RuntimeError("Need the exclusive empty large-volume parent")
    if Path(sys.executable).resolve() != Path("/usr/bin/python3.12").resolve():
        raise RuntimeError("Unexpected interpreter")
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        if os.environ.get(key) != "1":
            raise RuntimeError("CPU math environment must be single-threaded")
    previous = {unit: service_properties(unit, ("MainPID", "ActiveState", "Result"))
                for unit in TERMINAL_UNITS}
    for record in previous.values():
        validate_terminal(record)
    cgroup = next(line.split(":", 2)[2] for line in Path("/proc/self/cgroup").read_text().splitlines()
                  if line.startswith("0::"))
    root = Path("/sys/fs/cgroup") / cgroup.lstrip("/")
    effective = {key: (root / key).read_text().strip()
                 for key in ("memory.max", "memory.swap.max", "cpu.max")}
    service = service_properties(UNIT, ("RuntimeMaxUSec", "Restart", "KillMode", "Type"))
    validate_bounds(cgroup, effective, service)
    sys.path.insert(0, str(REPO))
    from experiments.measure_grokking_function_response import source_pins
    pins = source_pins()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    for name, expected in pins.items():
        current = (REPO / name).read_bytes()
        committed = subprocess.check_output(["git", "show", f"{commit}:{name}"], cwd=REPO)
        if (hashlib.sha256(committed).hexdigest() != expected
                or hashlib.sha256(current).hexdigest() != expected):
            raise RuntimeError("Diagnostic source differs from committed HEAD: " + name)
    print(json.dumps({"resource_guard": "pass", "unit": UNIT, "pid": os.getpid(),
        "cgroup": cgroup, "effective": effective, "service": service, "previous": previous,
        "source_commit": commit, "source_sha256": pins, "output": str(OUTPUT),
        "interpreter": str(Path(sys.executable).resolve())}), flush=True)
    os.execv(sys.executable, [sys.executable,
        str(REPO / "experiments/measure_grokking_function_response.py"), "--output-dir", str(OUTPUT)])


if __name__ == "__main__":
    main()
