#!/usr/bin/env python3
"""Losslessly archive completed raw JSON, preserving every original file."""
import gzip
import hashlib
import json
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent


def sha_stream(stream):
    digest = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def sha_file(path):
    with path.open("rb") as stream:
        return sha_stream(stream)


def main():
    source = HERE / "results"
    execution = json.loads((source / "execution.json").read_text())
    if (execution["mode"], execution["status"], execution["completed_runs"]) != ("confirmatory", "complete", 18):
        raise SystemExit("Only archive the complete 18-run experiment")
    names = ["execution.json"] + [
        f"seed{seed}-noise{noise:g}-{arm}.json"
        for seed in (0, 1, 2) for noise in (0., .9)
        for arm in ("adamw", "estimate32_project32", "estimate128_project32")]
    if not all((source / name).is_file() for name in names):
        raise SystemExit("A required raw result is missing")
    destination = HERE / "raw-results"
    destination.mkdir(exist_ok=False)
    records = []
    for name in names:
        original = source / name
        compressed = destination / (name + ".gz")
        before_hash = sha_file(original)
        with original.open("rb") as incoming, compressed.open("xb") as outgoing:
            with gzip.GzipFile(filename="", mode="wb", fileobj=outgoing, compresslevel=6, mtime=0) as archive:
                shutil.copyfileobj(incoming, archive)
        with gzip.open(compressed, "rb") as restored:
            restored_hash = sha_stream(restored)
        if before_hash != restored_hash or sha_file(original) != before_hash:
            raise RuntimeError("Archive roundtrip or original-file preservation failed")
        records.append({"original_path": f"results/{name}", "archive": compressed.name,
                        "original_bytes": original.stat().st_size, "archive_bytes": compressed.stat().st_size,
                        "original_sha256": before_hash, "archive_sha256": sha_file(compressed),
                        "roundtrip_verified": True})
    manifest = {"purpose": "Lossless portable raw evidence; originals remain local and unchanged.",
                "source_revision": execution["repository_revision"], "archive_script_sha256": sha_file(Path(__file__)),
                "files": records, "file_count": len(records),
                "original_bytes": sum(v["original_bytes"] for v in records),
                "archive_bytes": sum(v["archive_bytes"] for v in records)}
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({key: manifest[key] for key in ("file_count", "original_bytes", "archive_bytes")}))


if __name__ == "__main__":
    main()
