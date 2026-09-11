#!/usr/bin/env python3
"""Resolve only I14's empty-runtime-directory audit exception; no semantic replay."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
ROOT = Path("/tmp/spectral-experiment-artifacts/spectral-i14-001.SJWZCj")
PHASES = ("smoke", "calibration", "confirmation")
AUDIT_SHA = "1a19e0b5f8fd6c3d4259551df97558943d1d66bc1ff3d84ba678115d4fdde14b"
SUMMARY_SHA = "52394d1134e1fe3d69827ec95d64e6b8dd6bad0751e82ef68eb5fd1924ca342a"
BINDINGS = {
    "smoke/completion.json": "b763a9fcad9a2508efd99fbe0f20b9384be0e7aa6e06555d7157fd4ff930a331",
    "calibration/completion.json": "89525f06e028f047043a7d442f30a9344290275d4c46152f2604a03ef7a110a4",
    "confirmation/completion.json": "3bdb65bb2c724f3d4ca90a47593d40c67ef2c01121fbb928c74ae6e481569586",
    "attempt-smoke.json": "e90b485e193617fbe0206c6ac259e067d57b9378d070c4175d29d07aa32484c3",
    "attempt-calibration.json": "aa53230205d04176e7af74514976bb9e88e55ef548119e344525266bc578aa1f",
    "attempt-confirmation.json": "2217a1e9e5130979ca1bcc0b9607566ff1b0cb88c4dfbc5547baf071c9e7d656",
}


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024**2), b""):
            result.update(block)
    return result.hexdigest()


def runtime_identity(root):
    expected = set(PHASES) | {f"attempt-{phase}.json" for phase in PHASES}
    require({path.name for path in root.iterdir()} == expected | {"torchinductor_titus"},
            "Root entries differ from exact exception target")
    descriptor = os.open(root / "torchinductor_titus",
                         os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        info = os.fstat(descriptor)
        require(not os.listdir(descriptor), "Runtime directory is not empty")
        return {"device": info.st_dev, "inode": info.st_ino,
                "mode": info.st_mode, "mtime_ns": info.st_mtime_ns}
    finally:
        os.close(descriptor)


def verify_sources(commit, hashes):
    for name, expected in hashes.items():
        path = REPO / name
        require(digest(path) == expected, "Worktree source changed: " + name)
        raw = subprocess.check_output(["git", "show", commit + ":" + name], cwd=REPO)
        require(hashlib.sha256(raw).hexdigest() == expected, "Commit source differs: " + name)


def main():
    output = HERE / "analysis-001/runtime-directory-supplement.json"
    require(not output.exists(), "Supplement already exists; do not rerun")
    audit_path, summary_path = HERE / "analysis-001/audit.json", HERE / "analysis-001/summary.json"
    require(digest(audit_path) == AUDIT_SHA and digest(summary_path) == SUMMARY_SHA,
            "Original audit or summary changed")
    audit = json.loads(audit_path.read_text())
    require(audit.get("schema") == "i14_cross_optimizer_analysis_audit_v1"
            and audit.get("status") == "fail" and audit.get("errors") == [
                "artifact root contains an unlisted, partial, or missing entry"],
            "Exception is not the sole frozen root-layout failure")
    before = runtime_identity(ROOT)
    for name, expected in BINDINGS.items():
        require(digest(ROOT / name) == expected, "Bound terminal/attempt record changed")
    files, byte_count = 0, 0
    for phase in PHASES:
        directory = ROOT / phase
        completion = json.loads((directory / "completion.json").read_text())
        require(completion["phase"] == phase and completion["status"] == "complete",
                "Phase identity/status differs")
        records = completion["artifacts"]
        names = {row["name"] for row in records}
        require(len(names) == len(records) and {path.name for path in directory.iterdir()}
                == names | {"completion.json"}, "Phase file membership differs")
        for record in records:
            name = record["name"]
            require(Path(name).name == name, "Artifact name is not a basename")
            path = directory / name
            require(path.is_file() and not path.is_symlink()
                    and path.stat().st_size == record["bytes"]
                    and digest(path) == record["sha256"], "Artifact bytes changed: " + name)
            files += 1
            byte_count += record["bytes"]
        require(set(completion["source_hashes"]) == set(
            audit["source_provenance"]["source_files_verified"]), "Source set differs")
        verify_sources(completion["frozen_commit"], completion["source_hashes"])
        attempt = json.loads((ROOT / f"attempt-{phase}.json").read_text())
        require(attempt["phase"] == phase and attempt["restart"] == "forbidden"
                and attempt["frozen_commit"] == completion["frozen_commit"]
                and attempt["source_hashes"] == completion["source_hashes"],
                "Attempt provenance differs")
    analysis = audit["analysis_source_provenance"]
    verify_sources(analysis["git_commit"], analysis["source_hashes"])
    require(runtime_identity(ROOT) == before, "Runtime directory changed during verification")
    require(digest(audit_path) == AUDIT_SHA and digest(summary_path) == SUMMARY_SHA,
            "Original audit/summary changed during verification")
    for name, expected in BINDINGS.items():
        require(digest(ROOT / name) == expected, "Terminal/attempt changed during verification")
    result = {"schema": "i14_runtime_directory_supplement_v1",
        "status": "accepted_with_empty_runtime_directory_exception",
        "original_audit_status": "fail", "original_audit_sha256": AUDIT_SHA,
        "original_summary_sha256": SUMMARY_SHA, "artifact_root": str(ROOT),
        "bound_terminal_and_attempt_sha256": BINDINGS,
        "empty_directory": {"name": "torchinductor_titus", "before_and_after": before},
        "artifact_files_rehashed": files, "artifact_bytes_rehashed": byte_count,
        "supplement_source_sha256": digest(Path(__file__).resolve()),
        "scope": "Disposition of exactly one root-layout exception. Original audit/summary remain unchanged and failed status is not rewritten. All declared artifact hashes and source bindings rechecked; scalar/state semantic checks are inherited from the original audit, not rerun. Empty runtime directory contains no excluded evidence.",
        "terminal_hash_binding_note": "These six terminal/attempt hashes were recorded during diagnosis after the original audit; they are not claimed as hashes stored by that audit."}
    with output.open("x") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
