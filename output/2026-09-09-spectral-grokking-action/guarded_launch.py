"""Verify effective operational bounds before starting the frozen acquisition."""
import json
import os
from pathlib import Path
import subprocess
import sys

UNIT = "spectral-base-grokking-action-001.service"
REPO = Path("/private-artifacts/repositories/optimizers-launch-investigation").resolve()


def main():
    if len(sys.argv) != 2:
        raise ValueError("Provide the exclusive output parent")
    lines = Path("/proc/self/cgroup").read_text().splitlines()
    cgroup = next(line.split(":", 2)[2] for line in lines if line.startswith("0::"))
    if UNIT not in cgroup:
        raise RuntimeError("Not running inside the intended service")
    root = Path("/sys/fs/cgroup") / cgroup.lstrip("/")
    actual = {key: (root / key).read_text().strip()
              for key in ("memory.max", "memory.swap.max", "cpu.max")}
    quota, period = actual["cpu.max"].split()
    if (actual["memory.max"] != str(16 * 1024**3)
            or actual["memory.swap.max"] != "0" or quota == "max"
            or int(quota) / int(period) != 4):
        raise RuntimeError("Effective memory/swap/CPU limits differ from protocol")
    service = {}
    for key in ("RuntimeMaxUSec", "Restart", "KillMode"):
        service[key] = subprocess.check_output(
            ["systemctl", "--user", "show", UNIT, "--property=" + key, "--value"],
            text=True).strip()
    if service != {"RuntimeMaxUSec": "12h", "Restart": "no", "KillMode": "control-group"}:
        raise RuntimeError("Service runtime/restart/kill contract differs")
    print(json.dumps({"resource_guard": "pass", "unit": UNIT,
                      "cgroup": cgroup, "effective": actual, "service": service}), flush=True)
    os.execv(sys.executable, [sys.executable, str(REPO / "experiments/run_grokking_action_batch.py"),
                             "--output-parent", sys.argv[1]])


if __name__ == "__main__":
    main()
