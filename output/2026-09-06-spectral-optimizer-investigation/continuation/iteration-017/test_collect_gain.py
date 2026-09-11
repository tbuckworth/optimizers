"""Synthetic integrity tests for the stdlib-only I17 gain collector."""
from __future__ import annotations

import gzip
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import collect_gain as collector


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, allow_nan=False) + "\n")


class GainCollectorTests(unittest.TestCase):
    def fixture(self, root: Path):
        source = root / "source"
        source.mkdir()
        runtime = source / "runtime"
        (runtime / "nested" / "empty").mkdir(parents=True)
        (runtime / "nested" / "cache.bin").write_bytes(b"runtime-cache")
        completions, attempts = {}, {}
        frozen = "2" * 40
        source_hashes = {"run_gain_controls.py": "3" * 64}
        for phase in collector.PHASES:
            phase_dir = source / phase
            phase_dir.mkdir()
            names = collector._expected_json_artifacts(phase)
            for name in sorted(names):
                _write_json(phase_dir / name, {"name": name, "phase": phase,
                                               "values": [1, -2, 3.5]})
            if phase == "smoke":
                tensor = phase_dir / "synthetic-parent-clean.pt"
            else:
                tensor = phase_dir / "model-s200-sgdm-clean-scalar_k0p5-h250.pt"
            tensor.write_bytes(b"synthetic-tensor-never-loaded")
            records = [{"name": path.name, "bytes": path.stat().st_size,
                        "sha256": collector.sha256_path(path)}
                       for path in sorted(phase_dir.iterdir())]
            completion = {"schema": collector.COMPLETION_SCHEMA,
                          "status": "complete", "phase": phase,
                          "frozen_commit": frozen, "source_hashes": source_hashes,
                          "branches": 8 if phase == "smoke" else 24,
                          "numerical_failures": 0,
                          "all_requested_endpoints_present": True,
                          "artifacts": records}
            completion_path = phase_dir / "completion.json"
            _write_json(completion_path, completion)
            completions[phase] = collector.sha256_path(completion_path)
            attempt = {"schema": collector.ATTEMPT_SCHEMA, "phase": phase,
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

    @staticmethod
    def rebind_completion(audit: Path, completion: Path) -> str:
        audit_value = json.loads(audit.read_text())
        phase = json.loads(completion.read_text())["phase"]
        audit_value["phase_completion_sha256"][phase] = collector.sha256_path(completion)
        _write_json(audit, audit_value)
        return collector.sha256_path(audit)

    def test_regular_collection_is_complete_and_byte_exact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, audit, audit_sha = self.fixture(root)
            output = root / "collection"
            result = collector.collect(source, output, audit, audit_sha)
            self.assertEqual(result["schema"], collector.COLLECTION_SCHEMA)
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["json_count"], 48)
            self.assertEqual(result["runtime_inventory"],
                             collector.runtime_inventory(source / "runtime"))
            originals = {row["original"] for row in result["files"]}
            self.assertEqual(sum(name.startswith("smoke/branch-") for name in originals), 8)
            self.assertEqual(sum(name.startswith("confirmation/branch-")
                                 for name in originals), 24)
            for row in result["files"]:
                recovered = gzip.open(output / row["archive"], "rb").read()
                self.assertEqual(recovered, (source / row["original"]).read_bytes())
                self.assertEqual(collector.sha256_path(output / row["archive"]),
                                 row["archive_sha256"])
            self.assertEqual(json.loads((output / "collection.json").read_text()), result)

    def test_json_hash_mismatch_rejects_before_output_exists(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, audit, audit_sha = self.fixture(root)
            branch = source / "smoke/branch-s217-sgdm-clean-scalar_k0p5.json"
            branch.write_text("{}\n")
            output = root / "collection"
            with self.assertRaisesRegex(ValueError, "byte count differs|hash differs"):
                collector.collect(source, output, audit, audit_sha)
            self.assertFalse(output.exists())

    def test_missing_or_extra_scientific_json_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, audit, _ = self.fixture(root)
            phase = source / "confirmation"
            missing = phase / "branch-s200-sgdm-clean-scalar_k0p5.json"
            completion = phase / "completion.json"
            value = json.loads(completion.read_text())
            value["artifacts"] = [row for row in value["artifacts"]
                                  if row["name"] != missing.name]
            missing.unlink()
            _write_json(completion, value)
            audit_sha = self.rebind_completion(audit, completion)
            output = root / "collection"
            with self.assertRaisesRegex(ValueError, "JSON membership differs"):
                collector.collect(source, output, audit, audit_sha)
            self.assertFalse(output.exists())

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, audit, _ = self.fixture(root)
            phase = source / "smoke"
            extra = phase / "posthoc.json"
            _write_json(extra, {})
            completion = phase / "completion.json"
            value = json.loads(completion.read_text())
            value["artifacts"].append({"name": extra.name, "bytes": extra.stat().st_size,
                                       "sha256": collector.sha256_path(extra)})
            _write_json(completion, value)
            audit_sha = self.rebind_completion(audit, completion)
            output = root / "collection"
            with self.assertRaisesRegex(ValueError, "JSON membership differs"):
                collector.collect(source, output, audit, audit_sha)
            self.assertFalse(output.exists())

    def test_passing_audit_may_bind_retained_typed_numerical_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, audit, _ = self.fixture(root)
            completion = source / "confirmation/completion.json"
            value = json.loads(completion.read_text())
            failed_state = source / "confirmation/failed-state-s200-sgdm-clean-scalar_k0p5.pt"
            failed_state.write_bytes(b"last-good-complete-state")
            value["artifacts"].append({"name": failed_state.name,
                                       "bytes": failed_state.stat().st_size,
                                       "sha256": collector.sha256_path(failed_state)})
            value["numerical_failures"] = 1
            value["all_requested_endpoints_present"] = False
            _write_json(completion, value)
            audit_sha = self.rebind_completion(audit, completion)
            result = collector.collect(source, root / "collection", audit, audit_sha)
            self.assertEqual(result["json_count"], 48)

    def test_undeclared_artifact_and_symlinks_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, audit, audit_sha = self.fixture(root)
            (source / "confirmation/undeclared.json").write_text("{}\n")
            output = root / "collection"
            with self.assertRaisesRegex(ValueError, "artifact membership differs"):
                collector.collect(source, output, audit, audit_sha)
            self.assertFalse(output.exists())

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
            target = source / "smoke/manifest.json"
            contents = target.read_bytes()
            elsewhere = root / "elsewhere.json"
            elsewhere.write_bytes(contents)
            target.unlink()
            target.symlink_to(elsewhere)
            output = root / "collection"
            with self.assertRaisesRegex(ValueError, "nonsymlink"):
                collector.collect(source, output, audit, audit_sha)
            self.assertFalse(output.exists())

    def test_existing_relative_nested_and_audit_mismatch_reject(self):
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
            output = root / "collection"
            with self.assertRaisesRegex(ValueError, "supplied SHA256"):
                collector.collect(source, output, audit, "0" * 64)
            self.assertFalse(output.exists())

    def test_adversarial_audit_runtime_and_duplicate_keys_reject(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, audit, _ = self.fixture(root)
            value = json.loads(audit.read_text())
            value["artifact_root"] = str(root / "different")
            _write_json(audit, value)
            output = root / "collection"
            with self.assertRaisesRegex(ValueError, "exact artifact root"):
                collector.collect(source, output, audit, collector.sha256_path(audit))
            self.assertFalse(output.exists())

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, audit, _ = self.fixture(root)
            value = json.loads(audit.read_text())
            value["runtime_inventory"]["entries"][0]["path"] = "forged"
            _write_json(audit, value)
            output = root / "collection"
            with self.assertRaisesRegex(ValueError, "Runtime inventory differs"):
                collector.collect(source, output, audit, collector.sha256_path(audit))
            self.assertFalse(output.exists())

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, audit, _ = self.fixture(root)
            branch = source / "smoke/branch-s217-sgdm-clean-scalar_k0p5.json"
            branch.write_text('{"x":1,"x":2}\n')
            completion = source / "smoke/completion.json"
            value = json.loads(completion.read_text())
            record = next(row for row in value["artifacts"] if row["name"] == branch.name)
            record["bytes"] = branch.stat().st_size
            record["sha256"] = collector.sha256_path(branch)
            _write_json(completion, value)
            audit_sha = self.rebind_completion(audit, completion)
            output = root / "collection"
            with self.assertRaisesRegex(ValueError, "duplicate object key"):
                collector.collect(source, output, audit, audit_sha)
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
