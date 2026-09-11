#!/usr/bin/env python3
"""Losslessly collect every I16 scalar JSON after a hash-pinned passing audit."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import shutil


PHASES = ("smoke", "confirmation")
AUDIT_SCHEMA = "i16_scalar_analysis_audit_v1"
COLLECTION_SCHEMA = "i16_scalar_collection_v1"
OUTPUT_CAP = 512 * 1024**2
CHUNK = 1024 * 1024


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(CHUNK):
            digest.update(block)
    return digest.hexdigest()


def _object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("JSON contains a duplicate object key")
        value[key] = item
    return value


def load_json(path: Path):
    def invalid_constant(value):
        raise ValueError("JSON contains a nonfinite number: " + value)

    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle, object_pairs_hook=_object,
                             parse_constant=invalid_constant)
    except UnicodeDecodeError as exc:
        raise ValueError("JSON is not valid UTF-8: " + path.name) from exc


def canonical_bytes(value) -> bytes:
    return json.dumps(value, ensure_ascii=True, allow_nan=False,
                      separators=(",", ":")).encode("ascii")


def _regular(path: Path, description: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise ValueError(description + " must be a regular nonsymlink file")


def runtime_inventory(runtime: Path) -> dict:
    if runtime.is_symlink() or not runtime.is_dir():
        raise ValueError("Declared runtime must be a nonsymlink directory")
    entries = []

    def visit(directory: Path, prefix: Path) -> None:
        with os.scandir(directory) as iterator:
            children = sorted(iterator, key=lambda item: item.name)
        for child in children:
            relative = prefix / child.name
            relative_text = relative.as_posix()
            if child.is_symlink():
                raise ValueError("Runtime inventory contains a symlink: " + relative_text)
            if child.is_dir(follow_symlinks=False):
                entries.append({"path": relative_text, "kind": "directory",
                                "bytes": 0, "sha256": None})
                visit(Path(child.path), relative)
            elif child.is_file(follow_symlinks=False):
                path = Path(child.path)
                entries.append({"path": relative_text, "kind": "file",
                                "bytes": path.stat().st_size,
                                "sha256": sha256_path(path)})
            else:
                raise ValueError("Runtime inventory contains a special file: " + relative_text)

    visit(runtime, Path())
    entries.sort(key=lambda row: (row["path"], row["kind"]))
    return {"declared_name": "runtime", "entries": entries,
            "manifest_sha256": hashlib.sha256(canonical_bytes(entries)).hexdigest(),
            "regular_file_bytes": sum(row["bytes"] for row in entries
                                      if row["kind"] == "file"),
            "logical_bytes_only": True}


def _artifact_record(record) -> tuple[str, int, str]:
    if type(record) is not dict or set(record) != {"name", "bytes", "sha256"}:
        raise ValueError("Artifact record has unexpected metadata")
    name, size, digest = record["name"], record["bytes"], record["sha256"]
    if type(name) is not str or name in ("", ".", "..") or Path(name).name != name:
        raise ValueError("Artifact name must be one basename")
    if type(size) is not int or isinstance(size, bool) or size < 0:
        raise ValueError("Artifact byte count is invalid")
    if type(digest) is not str or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ValueError("Artifact SHA256 is invalid")
    return name, size, digest


def _preflight(source: Path, audit_path: Path, audit_sha256: str):
    if source.is_symlink() or not source.is_dir():
        raise ValueError("Source must be a nonsymlink directory")
    expected_root = {"runtime", *PHASES,
                     *(f"attempt-{phase}.json" for phase in PHASES)}
    actual_root = {path.name for path in source.iterdir()}
    if actual_root != expected_root:
        raise ValueError("Source root membership differs from the I16 protocol")
    for phase in PHASES:
        path = source / phase
        if path.is_symlink() or not path.is_dir():
            raise ValueError("Scientific phase must be a nonsymlink directory")

    _regular(audit_path, "Analysis audit")
    if re.fullmatch(r"[0-9a-f]{64}", audit_sha256 or "") is None:
        raise ValueError("Audit SHA256 must be a lowercase 64-digit hex string")
    observed_audit_sha = sha256_path(audit_path)
    if observed_audit_sha != audit_sha256:
        raise ValueError("Analysis audit differs from its supplied SHA256")
    audit = load_json(audit_path)
    if type(audit) is not dict or audit.get("schema") != AUDIT_SCHEMA \
            or audit.get("status") != "pass" \
            or audit.get("artifact_root") != str(source):
        raise ValueError("Analysis audit does not pass for this exact artifact root")
    expected_mapping = set(PHASES)
    if type(audit.get("phase_completion_sha256")) is not dict \
            or set(audit["phase_completion_sha256"]) != expected_mapping \
            or type(audit.get("attempt_sha256")) is not dict \
            or set(audit["attempt_sha256"]) != expected_mapping:
        raise ValueError("Analysis audit phase bindings differ")

    inventory = runtime_inventory(source / "runtime")
    if audit.get("runtime_inventory") != inventory:
        raise ValueError("Runtime inventory differs from the passing audit")

    collect_paths = []
    completions = {}
    attempts = {}
    provenance = {}
    for phase in PHASES:
        phase_dir = source / phase
        completion_path = phase_dir / "completion.json"
        _regular(completion_path, "Phase completion")
        completion_sha = sha256_path(completion_path)
        if audit["phase_completion_sha256"][phase] != completion_sha:
            raise ValueError("Phase completion differs from the passing audit")
        completion = load_json(completion_path)
        failures = completion.get("numerical_failures")
        endpoints = completion.get("all_requested_endpoints_present")
        if type(completion) is not dict \
                or completion.get("schema") != "i16_completion_v1" \
                or completion.get("status") != "complete" \
                or completion.get("phase") != phase \
                or type(failures) is not int or isinstance(failures, bool) or failures < 0 \
                or type(endpoints) is not bool or endpoints != (failures == 0):
            raise ValueError("I16 phase completion or numerical-failure accounting differs")
        records = completion.get("artifacts")
        if type(records) is not list:
            raise ValueError("I16 completion lacks an artifact list")
        manifest = {}
        for record in records:
            name, size, digest = _artifact_record(record)
            if name in manifest:
                raise ValueError("I16 completion contains duplicate artifact names")
            manifest[name] = (size, digest)
        actual = {path.name for path in phase_dir.iterdir()}
        if actual != set(manifest) | {"completion.json"}:
            raise ValueError("Scientific phase artifact membership differs")
        for name, (size, digest) in manifest.items():
            path = phase_dir / name
            _regular(path, "Declared phase artifact")
            if path.stat().st_size != size:
                raise ValueError("Declared artifact byte count differs: " + name)
            if path.suffix == ".json":
                if sha256_path(path) != digest:
                    raise ValueError("Declared JSON artifact hash differs: " + name)
                load_json(path)
                collect_paths.append(path)
        collect_paths.append(completion_path)
        completions[phase] = completion_sha
        provenance[phase] = {"frozen_commit": completion.get("frozen_commit"),
                             "source_hashes": completion.get("source_hashes")}

        attempt_path = source / f"attempt-{phase}.json"
        _regular(attempt_path, "Attempt record")
        attempt_sha = sha256_path(attempt_path)
        if audit["attempt_sha256"][phase] != attempt_sha:
            raise ValueError("Attempt record differs from the passing audit")
        attempt = load_json(attempt_path)
        if type(attempt) is not dict \
                or attempt.get("schema") != "i16_attempt_v1" \
                or attempt.get("phase") != phase \
                or attempt.get("restart") != "forbidden" \
                or attempt.get("frozen_commit") != completion.get("frozen_commit") \
                or attempt.get("source_hashes") != completion.get("source_hashes"):
            raise ValueError("Attempt metadata differs from its completed phase")
        collect_paths.append(attempt_path)
        attempts[phase] = attempt_sha

    relative_names = [path.relative_to(source).as_posix() for path in collect_paths]
    if len(relative_names) != len(set(relative_names)):
        raise ValueError("Scalar JSON collection contains duplicate paths")
    collect_paths.sort(key=lambda path: path.relative_to(source).as_posix())
    return {"audit": audit, "audit_sha256": observed_audit_sha,
            "runtime_inventory": inventory, "paths": collect_paths,
            "completion_sha256": completions, "attempt_sha256": attempts,
            "provenance": provenance}


class _BoundedWriter:
    def __init__(self, handle, remaining: int):
        self.handle = handle
        self.remaining = remaining

    def write(self, raw):
        if len(raw) > self.remaining:
            if self.remaining:
                self.handle.write(raw[:self.remaining])
                self.remaining = 0
            raise RuntimeError("I16 scalar collection exceeds the 512 MiB output cap")
        written = self.handle.write(raw)
        self.remaining -= written
        return written

    def flush(self):
        return self.handle.flush()

    def tell(self):
        return self.handle.tell()


def _archive(source_path: Path, target: Path, remaining: int) -> tuple[int, str]:
    target.parent.mkdir(parents=True, exist_ok=True)
    with source_path.open("rb") as original, target.open("xb") as raw_target:
        bounded = _BoundedWriter(raw_target, remaining)
        with gzip.GzipFile(filename="", mode="wb", compresslevel=9, mtime=0,
                           fileobj=bounded) as compressed:
            shutil.copyfileobj(original, compressed, length=CHUNK)
    size = target.stat().st_size
    return size, sha256_path(target)


def _check_roundtrip(source: Path, archive: Path, expected_size: int,
                     expected_sha: str) -> None:
    digest = hashlib.sha256()
    size = 0
    with source.open("rb") as original, gzip.open(archive, "rb") as recovered:
        while True:
            left, right = original.read(CHUNK), recovered.read(CHUNK)
            if left != right:
                raise ValueError("Gzip roundtrip differs: " + source.name)
            if not left:
                break
            digest.update(left)
            size += len(left)
    if size != expected_size or digest.hexdigest() != expected_sha:
        raise ValueError("Gzip roundtrip digest differs: " + source.name)


def collect(root, output, audit, audit_sha256):
    supplied_root = Path(root)
    if supplied_root.is_symlink():
        raise ValueError("Source must be a nonsymlink directory")
    source = supplied_root.resolve(strict=True)
    output = Path(output)
    supplied_audit = Path(audit)
    if supplied_audit.is_symlink():
        raise ValueError("Analysis audit must be a regular nonsymlink file")
    audit_path = supplied_audit.resolve(strict=True)
    if not output.is_absolute():
        raise ValueError("Output must be an absolute exclusive directory")
    if output.exists() or output.is_symlink():
        raise FileExistsError("Output directory already exists")
    output_parent = output.parent.resolve(strict=True)
    output = output_parent / output.name
    if source == output_parent or source in output.parents or output in source.parents:
        raise ValueError("Output must be separate from the source artifact root")

    # Every source name, JSON byte, audit binding and runtime entry is checked
    # before the exclusive output directory is created.
    admitted = _preflight(source, audit_path, audit_sha256)
    output.mkdir(exist_ok=False)
    rows = []
    archive_bytes = 0
    for path in admitted["paths"]:
        relative = path.relative_to(source).as_posix()
        original_size = path.stat().st_size
        original_sha = sha256_path(path)
        target = output / (relative + ".gz")
        size, digest = _archive(path, target, OUTPUT_CAP - archive_bytes)
        archive_bytes += size
        _check_roundtrip(path, target, original_size, original_sha)
        rows.append({"original": relative, "original_bytes": original_size,
                     "original_sha256": original_sha,
                     "archive": target.relative_to(output).as_posix(),
                     "archive_bytes": size, "archive_sha256": digest})

    # Recheck the externally mutable evidence after copying. A changed source
    # leaves the exclusive output intact for inspection and forbids auto-retry.
    after = _preflight(source, audit_path, audit_sha256)
    if [{key: row[key] for key in ("original", "original_bytes", "original_sha256")}
            for row in rows] != [
                {"original": path.relative_to(source).as_posix(),
                 "original_bytes": path.stat().st_size,
                 "original_sha256": sha256_path(path)} for path in after["paths"]]:
        raise ValueError("Scalar source changed during collection")
    if admitted["runtime_inventory"] != after["runtime_inventory"]:
        raise ValueError("Runtime inventory changed during collection")

    manifest = {"schema": COLLECTION_SCHEMA, "status": "complete",
        "source_root": str(source),
        "analysis_audit": {"path": str(audit_path), "sha256": audit_sha256,
                           "schema": AUDIT_SCHEMA, "status": "pass"},
        "phase_completion_sha256": admitted["completion_sha256"],
        "attempt_sha256": admitted["attempt_sha256"],
        "runtime_inventory": admitted["runtime_inventory"],
        "source_provenance": admitted["provenance"],
        "collector_sha256": sha256_path(Path(__file__).resolve()),
        "scope": ("Every original I16 JSON artifact, both phase completions, and both root "
                  "attempt records; runtime entries and I14/I15 references remain hash-bound but "
                  "are not archived."),
        "json_count": len(rows),
        "original_bytes": sum(row["original_bytes"] for row in rows),
        "archive_bytes": archive_bytes, "output_cap_bytes": OUTPUT_CAP,
        "files": rows}
    manifest_raw = json.dumps(manifest, indent=2, ensure_ascii=True,
                              allow_nan=False).encode("ascii") + b"\n"
    if archive_bytes + len(manifest_raw) > OUTPUT_CAP:
        raise RuntimeError("I16 scalar collection exceeds the 512 MiB output cap")
    with (output / "collection.json").open("xb") as handle:
        handle.write(manifest_raw)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument("--audit-sha256", required=True)
    args = parser.parse_args()
    manifest = collect(args.root, args.output, args.audit, args.audit_sha256)
    print(json.dumps({key: value for key, value in manifest.items() if key != "files"},
                     allow_nan=False))


if __name__ == "__main__":
    main()
