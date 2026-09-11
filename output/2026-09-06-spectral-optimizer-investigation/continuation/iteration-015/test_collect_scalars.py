"""Synthetic CPU-only integrity tests for the I15 scalar collector."""
from __future__ import annotations

import gzip
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import collect_scalars as collector


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, allow_nan=False) + "\n")


class CollectorTests(unittest.TestCase):
    def fixture(self, root: Path):
        source = root / "source"
        source.mkdir()
        runtime = source / "runtime"
        (runtime / "nested" / "empty").mkdir(parents=True)
        (runtime / "nested" / "cache.bin").write_bytes(b"runtime-cache")
        completions, attempts = {}, {}
        frozen = "2" * 40
        source_hashes = {"runner.py": "3" * 64}
        for phase in collector.PHASES:
            phase_dir = source / phase
            phase_dir.mkdir()
            metrics = phase_dir / "metrics.json"
            _write_json(metrics, {"phase": phase, "values": [1, -2, 3.5]})
            state = phase_dir / "state.pt"
            state.write_bytes(b"synthetic-tensor-not-loaded")
            records = [{"name": path.name, "bytes": path.stat().st_size,
                        "sha256": collector.sha256_path(path)}
                       for path in (metrics, state)]
            completion = {"schema": "i15_completion_v1", "status": "complete",
                          "phase": phase, "frozen_commit": frozen,
                          "source_hashes": source_hashes, "numerical_failures": 0,
                          "all_requested_endpoints_present": True,
                          "artifacts": records}
            completion_path = phase_dir / "completion.json"
            _write_json(completion_path, completion)
            completions[phase] = collector.sha256_path(completion_path)
            attempt = {"schema": "i15_attempt_v1", "phase": phase,
                       "frozen_commit": frozen, "source_hashes": source_hashes,
                       "pid": 123, "restart": "forbidden"}
            attempt_path = source / f"attempt-{phase}.json"
            _write_json(attempt_path, attempt)
            attempts[phase] = collector.sha256_path(attempt_path)
        audit = {"schema": collector.AUDIT_SCHEMA, "status": "pass",
                 "artifact_root": str(source.resolve()),
                 "phase_completion_sha256": completions,
                 "attempt_sha256": attempts,
                 "runtime_inventory": collector.runtime_inventory(runtime),
                 "checks": ["synthetic"], "errors": [], "warnings": []}
        audit_path = root / "audit.json"
        _write_json(audit_path, audit)
        return source, audit_path, collector.sha256_path(audit_path)

    def test_regular_collection_is_complete_and_byte_exact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, audit, audit_sha = self.fixture(root)
            output = root / "collection"
            result = collector.collect(source, output, audit, audit_sha)
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["json_count"], 6)
            self.assertEqual(result["runtime_inventory"],
                             collector.runtime_inventory(source / "runtime"))
            self.assertEqual(result["runtime_inventory"]["entries"], [
                {"path": "nested", "kind": "directory", "bytes": 0,
                 "sha256": None},
                {"path": "nested/cache.bin", "kind": "file", "bytes": 13,
                 "sha256": collector.sha256_path(source / "runtime/nested/cache.bin")},
                {"path": "nested/empty", "kind": "directory", "bytes": 0,
                 "sha256": None},
            ])
            for row in result["files"]:
                recovered = gzip.open(output / row["archive"], "rb").read()
                self.assertEqual(recovered, (source / row["original"]).read_bytes())
                self.assertEqual(collector.sha256_path(output / row["archive"]),
                                 row["archive_sha256"])
            on_disk = json.loads((output / "collection.json").read_text())
            self.assertEqual(on_disk, result)
            self.assertEqual(on_disk["analysis_audit"]["sha256"], audit_sha)

    def test_json_hash_mismatch_rejects_before_output_exists(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, audit, audit_sha = self.fixture(root)
            (source / "smoke/metrics.json").write_text("{}\n")
            output = root / "collection"
            with self.assertRaisesRegex(ValueError, "byte count differs|hash differs"):
                collector.collect(source, output, audit, audit_sha)
            self.assertFalse(output.exists())

    def test_phase_level_failure_or_extra_membership_is_never_exempted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, audit, audit_sha = self.fixture(root)
            (source / "confirmation/failure.json").write_text("{}\n")
            output = root / "collection"
            with self.assertRaisesRegex(ValueError, "membership differs"):
                collector.collect(source, output, audit, audit_sha)
            self.assertFalse(output.exists())

    def test_passing_audit_may_bind_retained_typed_numerical_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, audit, audit_sha = self.fixture(root)
            completion = source / "confirmation/completion.json"
            value = json.loads(completion.read_text())
            branch = source / "confirmation/branch-numerical.json"
            _write_json(branch, {"status": "numerical_failure",
                                 "failure": {"attempted_step": 101},
                                 "steps": [], "curve": []})
            failed_state = source / "confirmation/failed-state.pt"
            failed_state.write_bytes(b"last-good-complete-state")
            value["artifacts"].extend([
                {"name": branch.name, "bytes": branch.stat().st_size,
                 "sha256": collector.sha256_path(branch)},
                {"name": failed_state.name, "bytes": failed_state.stat().st_size,
                 "sha256": collector.sha256_path(failed_state)},
            ])
            value["numerical_failures"] = 1
            value["all_requested_endpoints_present"] = False
            _write_json(completion, value)
            audit_value = json.loads(audit.read_text())
            audit_value["phase_completion_sha256"]["confirmation"] = (
                collector.sha256_path(completion))
            _write_json(audit, audit_value)
            output = root / "collection"
            result = collector.collect(source, output, audit,
                                       collector.sha256_path(audit))
            self.assertEqual(result["json_count"], 7)
            row = next(row for row in result["files"]
                       if row["original"] == "confirmation/branch-numerical.json")
            self.assertEqual(gzip.open(output / row["archive"], "rb").read(),
                             branch.read_bytes())

    def test_symlinks_are_rejected_before_output_exists(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, audit, audit_sha = self.fixture(root)
            target = source / "runtime/nested/cache.bin"
            target.unlink()
            target.symlink_to(root / "audit.json")
            output = root / "collection"
            with self.assertRaisesRegex(ValueError, "symlink"):
                collector.collect(source, output, audit, audit_sha)
            self.assertFalse(output.exists())

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, audit, audit_sha = self.fixture(root)
            target = source / "smoke/metrics.json"
            contents = target.read_bytes()
            elsewhere = root / "elsewhere.json"
            elsewhere.write_bytes(contents)
            target.unlink()
            target.symlink_to(elsewhere)
            output = root / "collection"
            with self.assertRaisesRegex(ValueError, "nonsymlink"):
                collector.collect(source, output, audit, audit_sha)
            self.assertFalse(output.exists())

    def test_existing_relative_or_source_nested_output_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, audit, audit_sha = self.fixture(root)
            existing = root / "existing"
            existing.mkdir()
            with self.assertRaises(FileExistsError):
                collector.collect(source, existing, audit, audit_sha)
            with self.assertRaisesRegex(ValueError, "absolute"):
                collector.collect(source, Path("relative-output"), audit, audit_sha)
            nested = source / "new-output"
            with self.assertRaisesRegex(ValueError, "separate"):
                collector.collect(source, nested, audit, audit_sha)
            self.assertFalse(nested.exists())

    def test_audit_hash_and_adversarial_metadata_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, audit, audit_sha = self.fixture(root)
            output = root / "collection"
            with self.assertRaisesRegex(ValueError, "supplied SHA256"):
                collector.collect(source, output, audit, "0" * 64)
            self.assertFalse(output.exists())

            value = json.loads(audit.read_text())
            value["artifact_root"] = str(root / "different")
            _write_json(audit, value)
            with self.assertRaisesRegex(ValueError, "exact artifact root"):
                collector.collect(source, output, audit,
                                  collector.sha256_path(audit))
            self.assertFalse(output.exists())

            value["artifact_root"] = str(source.resolve())
            value["runtime_inventory"]["entries"][0]["path"] = "forged"
            _write_json(audit, value)
            with self.assertRaisesRegex(ValueError, "Runtime inventory differs"):
                collector.collect(source, output, audit,
                                  collector.sha256_path(audit))
            self.assertFalse(output.exists())

    def test_duplicate_json_keys_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, audit, audit_sha = self.fixture(root)
            metrics = source / "smoke/metrics.json"
            metrics.write_text('{"x":1,"x":2}\n')
            completion = source / "smoke/completion.json"
            value = json.loads(completion.read_text())
            record = next(row for row in value["artifacts"]
                          if row["name"] == "metrics.json")
            record["bytes"] = metrics.stat().st_size
            record["sha256"] = collector.sha256_path(metrics)
            _write_json(completion, value)
            audit_value = json.loads(audit.read_text())
            audit_value["phase_completion_sha256"]["smoke"] = (
                collector.sha256_path(completion))
            _write_json(audit, audit_value)
            output = root / "collection"
            with self.assertRaisesRegex(ValueError, "duplicate object key"):
                collector.collect(source, output, audit,
                                  collector.sha256_path(audit))
            self.assertFalse(output.exists())

    def test_output_cap_failure_preserves_bounded_partial_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, audit, audit_sha = self.fixture(root)
            output = root / "collection"
            with mock.patch.object(collector, "OUTPUT_CAP", 32):
                with self.assertRaisesRegex(RuntimeError, "output cap"):
                    collector.collect(source, output, audit, audit_sha)
            self.assertTrue(output.is_dir())
            self.assertLessEqual(sum(path.stat().st_size for path in output.rglob("*")
                                     if path.is_file()), 32)
            self.assertFalse((output / "collection.json").exists())


if __name__ == "__main__":
    unittest.main()
