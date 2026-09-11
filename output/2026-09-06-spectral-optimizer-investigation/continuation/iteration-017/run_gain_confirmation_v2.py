"""Resource-amended FIRST I17 confirmation; immutable scientific kernel/smoke."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
spec = importlib.util.spec_from_file_location("_i17_unchanged_gain_runner", HERE / "run_gain_controls.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
need = runner.need
ROOT = Path("/tmp/spectral-experiment-artifacts/spectral-i17-001.k4VKcx")
SMOKE_COMMIT = "14678e2a7a6fdc3aacc25a320516f1c07a037002"
SMOKE_SHA = "4985e7e1b5280f7e01f8ef4412518639e9aea7618ad9b068129a76dc1e55849d"
LIMIT = 2400
EXTRA_SOURCES = ("run_gain_confirmation_v2.py", "test_confirmation_resource.py", "resource-amendment.md")


def source_manifest(commit):
    original = runner.source_manifest(SMOKE_COMMIT)
    current = runner.source_manifest(commit)
    need(current == original and len(current) == 24, "Original scientific source closure changed")
    for name in EXTRA_SOURCES:
        path = HERE / name
        need(path.is_file() and not path.is_symlink(), "Amendment source is not regular")
        relative = str(path.relative_to(REPO))
        raw = path.read_bytes()
        frozen = subprocess.check_output(["git", "show", commit + ":" + relative], cwd=REPO)
        need(raw == frozen, "Amendment source differs from committed bytes: " + name)
        current[relative] = hashlib.sha256(raw).hexdigest()
    return current


def admit_smoke(root, original_sources):
    """Hash-bound reuse of a terminal synthetic phase, not another measurement."""
    need({p.name for p in root.iterdir()} == {"smoke", "runtime", "attempt-smoke.json"},
         "Root already attempted confirmation or has unexpected entries")
    for name in ("smoke", "runtime"):
        need((root / name).is_dir() and not (root / name).is_symlink(), "Root child is not a real directory")
    directory = root / "smoke"
    path = directory / "completion.json"
    need(path.is_file() and not path.is_symlink() and runner.old.digest(path) == SMOKE_SHA,
         "Original smoke completion differs from resource amendment")
    completion = json.loads(path.read_text())
    need(completion["schema"] == "i17_completion_v1" and completion["status"] == "complete"
         and completion["phase"] == "smoke" and completion["frozen_commit"] == SMOKE_COMMIT
         and completion["source_hashes"] == original_sources and completion["branches"] == 8
         and completion["numerical_failures"] == 0 and completion["completed_training_updates"] == 280
         and completion["all_requested_endpoints_present"] is True
         and 0 <= completion["elapsed_seconds"] <= 100, "Original smoke metadata differs")
    records = completion["artifacts"]
    index = {record["name"]: record for record in records}
    expected = {"manifest.json", "branches.json", "synthetic-parent-clean.pt", "synthetic-parent-fixed.pt"}
    for target in runner.TARGETS:
        for policy in runner.core.REAL_POLICIES:
            identity = f"s217-sgdm-{target}-{policy}"
            expected.update((f"branch-{identity}.json", f"state-{identity}-h110.pt"))
    need(len(records) == len(index) == 20 and set(index) == expected
         and {p.name for p in directory.iterdir()} == expected | {"completion.json"},
         "Original smoke artifact roster differs")
    for record in records:
        runner.previous.previous.checked_artifact(directory, record)
    path = root / "attempt-smoke.json"
    need(path.is_file() and not path.is_symlink(), "Original smoke attempt is not regular")
    attempt = json.loads(path.read_text())
    need(attempt == {"schema": "i17_attempt_v1", "phase": "smoke", "frozen_commit": SMOKE_COMMIT,
                    "source_hashes": original_sources, "pid": 2117929, "restart": "forbidden"},
         "Original smoke attempt differs")
    manifest = json.loads((directory / "manifest.json").read_text())
    need(manifest["source_hashes"] == original_sources and manifest["frozen_commit"] == SMOKE_COMMIT
         and manifest["wall_limit_seconds"] == 100 and manifest["smoke_forecast"] is None,
         "Original smoke manifest differs")
    branches = json.loads((directory / "branches.json").read_text())
    forecast = runner.smoke_forecast(branches["entries"])
    pairs = runner.validate_first_step_pairs(branches["entries"])
    need(branches["first_step_pair_checks"] == pairs and all(p["status"] == "pass" for p in pairs),
         "Original smoke pairs differ")
    need(1800 < forecast["confirmation_seconds"] <= LIMIT,
         "Forecast is outside the explicit resource-only amendment")
    return forecast


def main():
    need(__debug__, "Assertions must remain enabled")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--frozen-commit", required=True)
    args = parser.parse_args()
    need(not args.root.is_symlink() and args.root.absolute() == ROOT
         and args.root.resolve(strict=True) == ROOT, "Only the original exclusive I17 root is admitted")
    need(subprocess.check_output(["findmnt", "-n", "-o", "SOURCE", "-T", str(ROOT)], text=True).strip()
         == "/dev/RECONFIGURE_FOR_LOCAL_STORAGE", "I17 root is not on its declared volume")
    need(Path(os.environ.get("TMPDIR", "")).resolve() == ROOT / "runtime", "Declared runtime directory required")
    need(all(os.environ.get(name) == "1" for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")),
         "Numerical threads must remain one")
    sources = source_manifest(args.frozen_commit)
    original = dict(list(sources.items())[:24])
    forecast = admit_smoke(ROOT, original)
    with (ROOT / "attempt-confirmation.json").open("xb") as handle:
        handle.write(runner.old.json_bytes({"schema": "i17_attempt_v1", "phase": "confirmation",
            "frozen_commit": args.frozen_commit, "source_hashes": sources, "pid": os.getpid(), "restart": "forbidden"}))
    run = runner.old.Run(ROOT, "confirmation", LIMIT)
    try:
        runner.old.configure()
        run.save("manifest.json", {"schema": "i17_manifest_v1", "phase": "confirmation",
            "frozen_commit": args.frozen_commit, "source_hashes": sources,
            "torch_version": str(runner.torch.__version__), "numpy_version": runner.np.__version__,
            "python_version": sys.version, "gpu": runner.torch.cuda.get_device_name(),
            "real_policies": runner.core.REAL_POLICIES, "test_only_policies": runner.core.TEST_ONLY_POLICIES,
            "horizons": runner.HORIZONS, "smoke_forecast": forecast,
            "timing_safety_factor": 1.5, "cloud_spend_usd": 0, "authorized_budget_usd": 100,
            "shared_artifact_cap_bytes": runner.old.ARTIFACT_CAP, "wall_limit_seconds": LIMIT,
            "runtime_directory": "runtime", "runtime_bytes_count_in_cap": True})
        entries, warmup = runner.confirmation(run)
        need(warmup == 0 and source_manifest(args.frozen_commit) == sources, "Confirmation source/accounting changed")
        failures = sum(row["status"] == "numerical_failure" for row in entries)
        run.terminal("completion.json", {"schema": "i17_completion_v1", "status": "complete",
            "phase": "confirmation", "source_hashes": sources, "frozen_commit": args.frozen_commit,
            "branches": len(entries), "numerical_failures": failures,
            "all_requested_endpoints_present": failures == 0,
            "completed_training_updates": sum(row["completed_updates"] for row in entries),
            "update_seconds": sum(row["update_seconds"] for row in entries),
            "elapsed_seconds": time.monotonic() - run.started,
            "peak_torch_gpu_bytes": runner.torch.cuda.max_memory_allocated(),
            "shared_artifact_bytes_before_completion": run.used(), "artifacts": run.artifacts})
    except BaseException as exc:
        run.terminal("failure.json", {"schema": "i17_phase_failure_v1", "status": "failed",
            "phase": "confirmation", "exception_type": type(exc).__name__, "message": str(exc)[:2000],
            "source_hashes": sources, "elapsed_seconds": time.monotonic() - run.started,
            "artifacts": run.artifacts, "restart": "forbidden"})
        raise


if __name__ == "__main__":
    main()
