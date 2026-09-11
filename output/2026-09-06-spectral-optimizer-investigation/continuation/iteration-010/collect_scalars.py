#!/usr/bin/env python3
"""Losslessly archive I10 scalar evidence; retain bulk tensor files in place."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path


def sha(value):
    return hashlib.sha256(value).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.artifacts.resolve(strict=True)
    completion_raw = (source / "completion.json").read_bytes()
    completion = json.loads(completion_raw)
    assert completion["status"] == "complete" and completion["mode"] == "full"
    index = {r["name"]: r for r in completion["artifacts"]}
    paths = sorted(source.glob("*.json"))
    assert set(p.name for p in paths) == {
        name for name in index if name.endswith(".json")} | {"completion.json"}
    args.output.mkdir(exist_ok=False)
    rows = []
    for path in paths:
        raw = path.read_bytes()
        if path.name in index:
            assert len(raw) == index[path.name]["bytes"]
            assert sha(raw) == index[path.name]["sha256"]
        archive = gzip.compress(raw, compresslevel=9, mtime=0)
        assert gzip.decompress(archive) == raw
        name = path.name + ".gz"
        with (args.output / name).open("xb") as handle:
            handle.write(archive)
        rows.append({"original": path.name, "original_bytes": len(raw),
                     "original_sha256": sha(raw), "archive": name,
                     "archive_bytes": len(archive), "archive_sha256": sha(archive)})
    manifest = {"status": "complete", "source_directory": str(source),
                "source_completion_sha256": sha(completion_raw),
                "scope": "Every original JSON byte, including all step diagnostics; tensors remain hash-indexed on the desktop large volume, not backed up.",
                "json_count": len(rows), "original_bytes": sum(r["original_bytes"] for r in rows),
                "archive_bytes": sum(r["archive_bytes"] for r in rows), "files": rows}
    with (args.output / "collection.json").open("x") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    print(json.dumps({k: v for k, v in manifest.items() if k != "files"}))


if __name__ == "__main__":
    main()
