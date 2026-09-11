"""Synthetic collector integrity tests; no real artifact reads."""
import gzip
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import collect_scalars as collector


class CollectionTests(unittest.TestCase):
    def source(self, root):
        source = root / "source"
        source.mkdir()
        for phase in collector.PHASES:
            directory = source / phase
            directory.mkdir()
            raw = b'{"synthetic":true}\n'
            (directory / "metrics.json").write_bytes(raw)
            completion = {"status": "complete", "phase": phase, "artifacts": [
                {"name": "metrics.json", "bytes": len(raw), "sha256": collector.sha(raw)}]}
            (directory / "completion.json").write_text(json.dumps(completion))
            (source / f"attempt-{phase}.json").write_text(json.dumps({"phase": phase}))
        return source

    def test_roundtrip_all_phases_and_exclusive_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.source(root)
            output = root / "archive"
            result = collector.collect(source, output)
            self.assertEqual(result["json_count"], 9)
            for row in result["files"]:
                self.assertEqual(gzip.decompress((output / row["archive"]).read_bytes()),
                                 (source / row["original"]).read_bytes())
            with self.assertRaises(FileExistsError):
                collector.collect(source, output)

    def test_hash_mismatch_rejects_before_output_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.source(root)
            (source / "calibration/metrics.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "differs"):
                collector.collect(source, root / "archive")
            self.assertFalse((root / "archive").exists())

    def test_supplement_authentication_and_wrong_hash_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.source(root)
            (source / collector.RUNTIME_DIRECTORY).mkdir()
            runtime = collector.validate_root(source)
            audit_raw = json.dumps({"status": "fail", "errors": [
                "artifact root contains an unlisted, partial, or missing entry"]}).encode()
            summary_raw = b'{"synthetic":true}'
            (root / "audit.json").write_bytes(audit_raw)
            (root / "summary.json").write_bytes(summary_raw)
            audit_sha, summary_sha = collector.sha(audit_raw), collector.sha(summary_raw)
            payload = {"schema": "i14_runtime_directory_supplement_v1",
                "status": "accepted_with_empty_runtime_directory_exception",
                "artifact_root": str(source), "original_audit_status": "fail",
                "original_audit_sha256": audit_sha, "original_summary_sha256": summary_sha,
                "supplement_source_sha256": collector.sha(Path(collector.supplement_rules.__file__).read_bytes()),
                "empty_directory": {"name": collector.RUNTIME_DIRECTORY,
                    "before_and_after": {key: runtime[key] for key in ("device", "inode", "mtime_ns")}},
                "bound_terminal_and_attempt_sha256": {
                    "attempt-smoke.json": collector.sha((source / "attempt-smoke.json").read_bytes())}}
            path = root / "supplement.json"
            path.write_text(json.dumps(payload))
            pinned = collector.sha(path.read_bytes())
            with mock.patch.multiple(collector.supplement_rules, AUDIT_SHA=audit_sha, SUMMARY_SHA=summary_sha,
                                     BINDINGS=payload["bound_terminal_and_attempt_sha256"]):
                self.assertEqual(collector.validate_supplement(source, runtime, path, pinned), pinned)
                with self.assertRaisesRegex(ValueError, "authenticate"):
                    collector.collect(source, root / "bad", path, "wrong-sha")
                self.assertFalse((root / "bad").exists())

    def test_only_exact_empty_nonsymlink_runtime_directory_is_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.source(root)
            cache = source / collector.RUNTIME_DIRECTORY
            cache.mkdir()
            with self.assertRaisesRegex(ValueError, "requires"):
                collector.collect(source, root / "missing-acceptance")
            self.assertFalse((root / "missing-acceptance").exists())
            with mock.patch.object(collector, "validate_supplement", return_value="synthetic-pinned"):
                result = collector.collect(source, root / "archive", root / "synthetic-acceptance", "synthetic-pinned")
            self.assertTrue(result["empty_runtime_directory_exception"]["empty"])
            (cache / "unexpected.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "unlisted"):
                collector.collect(source, root / "rejected")
            self.assertFalse((root / "rejected").exists())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.source(root)
            (source / collector.RUNTIME_DIRECTORY).symlink_to(root, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "unlisted"):
                collector.collect(source, root / "rejected")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.source(root)
            (source / "unexpected.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "unlisted"):
                collector.collect(source, root / "rejected")


if __name__ == "__main__":
    unittest.main()
