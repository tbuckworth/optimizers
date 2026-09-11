"""Check the effective one-shot service contract before new acquisition."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

UNIT = "spectral-base-grokking-raw-direction-001.service"
REPO = Path(__file__).resolve().parents[2]
STORAGE = Path("/tmp/spectral-experiment-artifacts")


def check_limits(actual, service):
    quota, period = actual["cpu.max"].split()
    if (actual["memory.max"] != str(16 * 1024**3)
            or actual["memory.swap.max"] != "0" or quota == "max"
            or not 0 < int(quota) / int(period) <= 4):
        raise RuntimeError("Effective memory/swap/CPU bounds differ from protocol")
    if service != {"RuntimeMaxUSec": "12h", "Restart": "no",
                   "KillMode": "control-group", "Type": "exec"}:
        raise RuntimeError("Service runtime/restart/kill/type contract differs")


def main():
    if len(sys.argv) != 2:
        raise ValueError("Provide one exclusive output parent")
    output = Path(sys.argv[1]).resolve()
    if (output.parent != STORAGE.resolve() or not output.is_dir()
            or list(output.iterdir())
            or os.stat(output).st_dev == os.stat("/").st_dev):
        raise ValueError("Require an empty exclusive output on the large volume")
    if Path(sys.executable).resolve() != Path("/usr/bin/python3.12").resolve():
        raise RuntimeError("Unexpected acquisition interpreter")
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS"):
        if os.environ.get(key) != "1":
            raise RuntimeError("CPU math environment is not single-threaded")
    lines = Path("/proc/self/cgroup").read_text().splitlines()
    cgroup = next(line.split(":", 2)[2] for line in lines if line.startswith("0::"))
    if Path(cgroup).name != UNIT:
        raise RuntimeError("Not inside the intended one-shot service")
    root = Path("/sys/fs/cgroup") / cgroup.lstrip("/")
    actual = {key: (root / key).read_text().strip()
              for key in ("memory.max", "memory.swap.max", "cpu.max")}
    service = {key: subprocess.check_output(
        ["systemctl", "--user", "show", UNIT, "--property=" + key, "--value"],
        text=True).strip() for key in ("RuntimeMaxUSec", "Restart", "KillMode", "Type")}
    check_limits(actual, service)
    sys.path.insert(0, str(REPO))
    from experiments.grokking_raw_direction import source_pins
    pins = source_pins()
    guard_path = str(Path(__file__).resolve().relative_to(REPO))
    pins[guard_path] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    for path, expected in pins.items():
        committed = subprocess.check_output(["git", "show", f"{commit}:{path}"], cwd=REPO)
        if hashlib.sha256(committed).hexdigest() != expected:
            raise RuntimeError("Scientific/guard sources are not the committed snapshot")
    print(json.dumps({"resource_guard": "pass", "unit": UNIT, "pid": os.getpid(),
                      "output": str(output), "cgroup": cgroup, "effective": actual,
                      "service": service, "source_commit": commit, "source_sha256": pins,
                      "interpreter": str(Path(sys.executable).resolve())}), flush=True)
    os.execv(sys.executable, [sys.executable, str(REPO / "experiments/run_grokking_raw_direction_batch.py"),
                             "--output-parent", str(output)])


if __name__ == "__main__":
    main()
