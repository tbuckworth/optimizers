#!/usr/bin/env python3
"""Losslessly archive every I14 JSON byte after all three phases complete."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import audit_runtime_directory as supplement_rules

PHASES = ("smoke", "calibration", "confirmation")
SOURCE_CAP = 512 * 1024**2
ARCHIVE_CAP = 128 * 1024**2
RUNTIME_DIRECTORY = "torchinductor_titus"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def validate_root(source):
    expected = set(PHASES) | {f"attempt-{phase}.json" for phase in PHASES}
    actual = {path.name for path in source.iterdir()}
    if actual == expected:
        return None
    cache = source / RUNTIME_DIRECTORY
    if actual != expected | {RUNTIME_DIRECTORY} or cache.is_symlink() \
            or not cache.is_dir() or list(cache.iterdir()):
        raise ValueError("Source root has missing, material, or unlisted entries")
    descriptor = os.open(cache, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        if os.listdir(descriptor):
            raise ValueError("Runtime directory became nonempty")
        info = os.fstat(descriptor)
        return {"name": RUNTIME_DIRECTORY, "empty": True, "symlink": False,
                "device": info.st_dev, "inode": info.st_ino, "mtime_ns": info.st_mtime_ns,
                "scope": "Empty runtime directory only; no source artifact excluded"}
    finally:
        os.close(descriptor)


def validate_supplement(source, runtime, path, expected_sha):
    if runtime is None:
        if path is not None or expected_sha is not None:
            raise ValueError("Ordinary root must not use an exception supplement")
        return None
    if path is None or expected_sha is None:
        raise ValueError("Exceptional root requires a hash-pinned supplement")
    path = Path(path)
    raw = path.read_bytes()
    acceptance = json.loads(raw)
    expected_directory = acceptance.get("empty_directory", {})
    if sha(raw) != expected_sha \
            or acceptance.get("schema") != "i14_runtime_directory_supplement_v1" \
            or acceptance.get("status") != "accepted_with_empty_runtime_directory_exception" \
            or acceptance.get("artifact_root") != str(source) \
            or acceptance.get("original_audit_status") != "fail" \
            or acceptance.get("original_audit_sha256") != supplement_rules.AUDIT_SHA \
            or acceptance.get("original_summary_sha256") != supplement_rules.SUMMARY_SHA \
            or acceptance.get("bound_terminal_and_attempt_sha256") != supplement_rules.BINDINGS \
            or acceptance.get("supplement_source_sha256") != sha(Path(supplement_rules.__file__).read_bytes()) \
            or expected_directory.get("name") != RUNTIME_DIRECTORY \
            or any(expected_directory.get("before_and_after", {}).get(key) != runtime[key]
                   for key in ("device", "inode", "mtime_ns")):
        raise ValueError("Supplement does not authenticate this exact exception")
    audit_raw, summary_raw = (path.parent / "audit.json").read_bytes(), (path.parent / "summary.json").read_bytes()
    audit = json.loads(audit_raw)
    if sha(audit_raw) != supplement_rules.AUDIT_SHA or sha(summary_raw) != supplement_rules.SUMMARY_SHA \
            or audit.get("status") != "fail" or audit.get("errors") != [
                "artifact root contains an unlisted, partial, or missing entry"]:
        raise ValueError("Original audit or summary does not match accepted exception")
    for name, expected in acceptance["bound_terminal_and_attempt_sha256"].items():
        if Path(name).is_absolute() or ".." in Path(name).parts \
                or sha((source / name).read_bytes()) != expected:
            raise ValueError("Supplement terminal/attempt binding differs")
    return expected_sha


def collect(source, output, supplement=None, supplement_sha256=None):
    source = Path(source).resolve(strict=True)
    output = Path(output)
    runtime_directory = validate_root(source)
    supplement_hash = validate_supplement(source, runtime_directory, supplement, supplement_sha256)
    paths, completion_hashes = [], {}
    for phase in PHASES:
        directory = source / phase
        completion_raw = (directory / "completion.json").read_bytes()
        completion = json.loads(completion_raw)
        if completion.get("status") != "complete" or completion.get("phase") != phase:
            raise ValueError("Every phase must have its exact terminal completion")
        records = completion["artifacts"]
        index = {record["name"]: record for record in records}
        if len(index) != len(records) or {path.name for path in directory.iterdir()} \
                != set(index) | {"completion.json"}:
            raise ValueError("Phase artifact membership differs")
        completion_hashes[phase] = sha(completion_raw)
        for path in sorted(directory.glob("*.json")):
            record = index.get(path.name)
            if record is not None:
                raw = path.read_bytes()
                if len(raw) != record["bytes"] or sha(raw) != record["sha256"]:
                    raise ValueError("JSON differs from terminal artifact manifest")
            paths.append(path)
        paths.append(source / f"attempt-{phase}.json")
    if sum(path.stat().st_size for path in paths) > SOURCE_CAP:
        raise ValueError("JSON source exceeds frozen collection ceiling")
    output.mkdir(exist_ok=False)
    rows, written = [], 0
    for path in paths:
        raw = path.read_bytes()
        relative = path.relative_to(source)
        archive = gzip.compress(raw, compresslevel=9, mtime=0)
        if gzip.decompress(archive) != raw:
            raise ValueError("Archive roundtrip differs")
        written += len(archive)
        if written > ARCHIVE_CAP:
            raise ValueError("Archive ceiling exhausted; partial output retained")
        target = output / (str(relative) + ".gz")
        target.parent.mkdir(exist_ok=True)
        with target.open("xb") as handle:
            handle.write(archive)
        rows.append({"original": str(relative), "original_bytes": len(raw),
            "original_sha256": sha(raw), "archive": str(target.relative_to(output)),
            "archive_bytes": len(archive), "archive_sha256": sha(archive)})
    if validate_root(source) != runtime_directory:
        raise ValueError("Runtime directory changed during collection")
    if validate_supplement(source, runtime_directory, supplement, supplement_sha256) != supplement_hash:
        raise ValueError("Supplement changed during collection")
    manifest = {"schema": "i14_scalar_collection_v1", "status": "complete",
        "source_directory": str(source), "source_completion_sha256": completion_hashes,
        "collector_sha256": sha(Path(__file__).read_bytes()),
        "empty_runtime_directory_exception": runtime_directory,
        "runtime_directory_supplement_sha256": supplement_hash,
        "scope": "Every JSON byte from all phases and attempts; bulk tensor files remain hash-bound on the large volume, not backed up.",
        "json_count": len(rows), "original_bytes": sum(row["original_bytes"] for row in rows),
        "archive_bytes": written, "files": rows}
    with (output / "collection.json").open("x") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--supplement", type=Path)
    parser.add_argument("--supplement-sha256")
    args = parser.parse_args()
    manifest = collect(args.artifacts, args.output, args.supplement, args.supplement_sha256)
    print(json.dumps({key: value for key, value in manifest.items() if key != "files"}))


if __name__ == "__main__":
    main()
