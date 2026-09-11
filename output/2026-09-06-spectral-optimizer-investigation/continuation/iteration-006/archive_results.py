#!/usr/bin/env python3
"""Lossless portable raw-JSON copies; originals and tensor files remain untouched."""
import gzip
import hashlib
import json
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent


def stream_sha(stream):
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(block)
    return digest.hexdigest()


def sha(path):
    with path.open("rb") as stream:
        return stream_sha(stream)


def main():
    destination = HERE / "raw-results"
    if destination.exists() or shutil.disk_usage(HERE).free < 1024 ** 3:
        raise RuntimeError("Exclusive destination and one-GiB workspace headroom required")
    records, manifests = [], []
    for group, relative, mode, count in (("primary", "results/execution.json", "full", 36),
                                         ("independent", "audit-reexecution/execution.json", "audit_reexecution", 6)):
        path = HERE / relative
        manifest = json.loads(path.read_bytes())
        if (manifest["mode"], manifest["status"], manifest["completed_runs"], manifest["test_evaluations"]) != (mode, "complete", count, 4 * count):
            raise RuntimeError("Execution is not terminal complete")
        if not manifest["all_gates_passed"]:
            raise RuntimeError("Execution gates failed")
        artifacts = manifest["training_runs"] + manifest["runs"]
        if len(artifacts) != 2 * count or len({item["path"] for item in artifacts}) != 2 * count:
            raise RuntimeError("Wrong raw artifact count or aliased files")
        for item in artifacts:
            source = Path(item["path"])
            if (source.parent.resolve() != Path(manifest["bulk_root"]).resolve()
                    or source.stat().st_size != item["size_bytes"] or sha(source) != item["sha256"]):
                raise RuntimeError("Raw input binding mismatch")
            records.append({"group": group, "original": item})
        manifests.append({"group": group, "path": relative, "sha256": sha(path),
                          "repository_revision": manifest["repository_revision"]})
    destination.mkdir(exist_ok=False)
    output = {"status": "INCOMPLETE", "archive_script_sha256": sha(Path(__file__)),
              "execution_manifests": manifests, "files": [],
              "scope": "All 84 original pre-test and final raw JSON files, byte-for-byte. Tensor checkpoints and plans remain hash-bound in local bulk storage; not included here."}
    try:
        for record in records:
            item = record["original"]
            source = Path(item["path"])
            target = destination / (record["group"] + "-" + source.name + ".gz")
            with source.open("rb") as incoming, target.open("xb") as outgoing:
                with gzip.GzipFile(filename="", fileobj=outgoing, mode="wb", mtime=0, compresslevel=6) as compressed:
                    shutil.copyfileobj(incoming, compressed)
            with gzip.open(target, "rb") as restored:
                if stream_sha(restored) != item["sha256"] or sha(source) != item["sha256"]:
                    raise RuntimeError("Lossless roundtrip or original preservation failed")
            output["files"].append({**record, "archive": target.name, "archive_bytes": target.stat().st_size,
                                     "archive_sha256": sha(target), "roundtrip_verified": True})
        output.update(status="PASS", file_count=len(output["files"]),
                      original_bytes=sum(row["original"]["size_bytes"] for row in output["files"]),
                      archive_bytes=sum(row["archive_bytes"] for row in output["files"]))
        if output["archive_bytes"] >= 256 * 1024 ** 2:
            raise RuntimeError("Archive exceeded conservative 256-MiB bound")
    except BaseException:
        output["status"] = "ERROR_PARTIAL_PRESERVED"
        raise
    finally:
        with (destination / "manifest.json").open("x") as stream:
            json.dump(output, stream, indent=2, allow_nan=False)
            stream.write("\n")
    print(json.dumps({key: output[key] for key in ("status", "file_count", "original_bytes", "archive_bytes")}))


if __name__ == "__main__":
    main()
