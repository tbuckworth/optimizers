#!/usr/bin/env python3
"""Losslessly archive completed raw JSON without replacing original evidence."""
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
    if (execution["mode"], execution["status"], execution["completed_runs"]) != ("confirmatory", "complete", 12):
        raise SystemExit("Only archive the completed twelve-run confirmation")
    names = ["execution.json"] + [f"seed{seed}-{arm}.json" for seed in (3, 4, 5)
                                for arm in ("adamw", "estimate32_project32", "estimate128_project32", "scalar32_norm")]
    if not all((source / name).is_file() for name in names):
        raise SystemExit("A required raw result is missing")
    destination = HERE / "raw-results"
    destination.mkdir(exist_ok=False)
    records = []
    for name in names:
        original, compressed = source / name, destination / (name + ".gz")
        before = sha_file(original)
        with original.open("rb") as incoming, compressed.open("xb") as outgoing:
            with gzip.GzipFile(filename="", mode="wb", fileobj=outgoing, compresslevel=6, mtime=0) as archive:
                shutil.copyfileobj(incoming, archive)
        with gzip.open(compressed, "rb") as restored:
            if sha_stream(restored) != before or sha_file(original) != before:
                raise RuntimeError("Archive roundtrip or original preservation failed")
        records.append({"original_path": f"results/{name}", "archive": compressed.name,
                        "original_bytes": original.stat().st_size, "archive_bytes": compressed.stat().st_size,
                        "original_sha256": before, "archive_sha256": sha_file(compressed), "roundtrip_verified": True})
    manifest = {"purpose": "Portable lossless raw evidence; original files remain unchanged locally.",
                "source_revision": execution["repository_revision"], "archive_script_sha256": sha_file(Path(__file__)),
                "files": records, "file_count": len(records),
                "original_bytes": sum(r["original_bytes"] for r in records),
                "archive_bytes": sum(r["archive_bytes"] for r in records)}
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({key: manifest[key] for key in ("file_count", "original_bytes", "archive_bytes")}))


if __name__ == "__main__":
    main()
