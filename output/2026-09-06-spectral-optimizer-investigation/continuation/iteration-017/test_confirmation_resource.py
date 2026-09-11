"""Synthetic tests for the resource-only wrapper; no GPU or scientific data."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("_resource_wrapper_test", HERE / "run_gain_confirmation_v2.py")
wrapper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wrapper)


class ResourceConfirmationTests(unittest.TestCase):
    def fixture(self, root, seconds=.3):
        (root / "runtime").mkdir()
        directory = root / "smoke"
        directory.mkdir()
        sources = {f"source-{i}": "b" * 64 for i in range(24)}
        records, entries = [], []

        def write(name, value):
            path = directory / name
            raw = json.dumps(value).encode() if name.endswith(".json") else b"synthetic-not-a-tensor"
            path.write_bytes(raw)
            record = {"name": name, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
            records.append(record)
            return record

        write("manifest.json", {"source_hashes": sources, "frozen_commit": wrapper.SMOKE_COMMIT,
              "wall_limit_seconds": 100, "smoke_forecast": None})
        for target in wrapper.runner.TARGETS:
            write(f"synthetic-parent-{target}.pt", {})
            for policy in wrapper.runner.core.REAL_POLICIES:
                identity = f"s217-sgdm-{target}-{policy}"
                artifact = write(f"branch-{identity}.json", {"synthetic": True})
                write(f"state-{identity}-h110.pt", {})
                entries.append({"seed": 217, "target": target, "policy": policy, "status": "complete",
                    "completed_updates": 10, "update_seconds": seconds,
                    "parent_state_digest": "c" * 64, "parent_evaluation_digest": "d" * 64,
                    "first_step_digests": {name: "e" * 64 for name in
                        ("raw_gradient", "post_observer", "old_momentum_buffer")}, "artifact": artifact})
        write("branches.json", {"entries": entries,
              "first_step_pair_checks": wrapper.runner.validate_first_step_pairs(entries)})
        completion = {"schema": "i17_completion_v1", "status": "complete", "phase": "smoke",
            "frozen_commit": wrapper.SMOKE_COMMIT, "source_hashes": sources, "branches": 8,
            "numerical_failures": 0, "completed_training_updates": 280,
            "all_requested_endpoints_present": True, "elapsed_seconds": 6.0, "artifacts": records}
        raw = json.dumps(completion).encode()
        (directory / "completion.json").write_bytes(raw)
        (root / "attempt-smoke.json").write_text(json.dumps({"schema": "i17_attempt_v1", "phase": "smoke",
            "frozen_commit": wrapper.SMOKE_COMMIT, "source_hashes": sources, "pid": 2117929, "restart": "forbidden"}))
        return sources, hashlib.sha256(raw).hexdigest()

    def test_fixed_formula_reuses_terminal_smoke_without_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources, digest = self.fixture(root)
            with mock.patch.object(wrapper, "SMOKE_SHA", digest), \
                 mock.patch.object(wrapper.runner, "confirmation", side_effect=AssertionError("training")), \
                 mock.patch.object(wrapper.runner, "synthetic_smoke", side_effect=AssertionError("smoke replay")), \
                 mock.patch.object(wrapper.runner.torch, "load", side_effect=AssertionError("tensor load")):
                value = wrapper.admit_smoke(root, sources)
            self.assertAlmostEqual(value["confirmation_seconds"], 2112.0)
            self.assertEqual({p.name for p in root.iterdir()}, {"runtime", "smoke", "attempt-smoke.json"})

    def test_original_and_amended_timing_boundaries_are_explicit(self):
        for seconds in (.1, .4):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                sources, digest = self.fixture(root, seconds)
                with mock.patch.object(wrapper, "SMOKE_SHA", digest), self.assertRaisesRegex(ValueError, "outside"):
                    wrapper.admit_smoke(root, sources)

    def test_repeated_confirmation_source_pin_and_artifact_tampering_fail(self):
        for mutation, expected in (("attempt", "already attempted"), ("pin", "completion differs"),
                                   ("artifact", "Artifact content differs"), ("source", "metadata differs")):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                sources, digest = self.fixture(root)
                if mutation == "attempt":
                    (root / "attempt-confirmation.json").write_text("{}")
                elif mutation == "pin":
                    digest = "0" * 64
                elif mutation == "artifact":
                    path = root / "smoke/synthetic-parent-clean.pt"
                    raw = path.read_bytes()
                    path.write_bytes(b"X" + raw[1:])
                elif mutation == "source":
                    sources = {**sources, "source-0": "f" * 64}
                with mock.patch.object(wrapper, "SMOKE_SHA", digest), self.assertRaisesRegex(ValueError, expected):
                    wrapper.admit_smoke(root, sources)

    def test_source_map_requires_identical_original_twenty_four_files(self):
        old = {f"source-{i}": "a" * 64 for i in range(24)}
        changed = {**old, "source-0": "b" * 64}
        with mock.patch.object(wrapper.runner, "source_manifest", side_effect=[old, changed]), \
             self.assertRaisesRegex(ValueError, "scientific source closure changed"):
            wrapper.source_manifest("a" * 40)

    def test_amendment_sources_are_exact_and_committed(self):
        original = {f"source-{i}": "a" * 64 for i in range(24)}

        def git_show(args, **kwargs):
            name = args[-1].split(":", 1)[1]
            return (wrapper.REPO / name).read_bytes()

        with mock.patch.object(wrapper.runner, "source_manifest", side_effect=lambda commit: dict(original)), \
             mock.patch.object(wrapper.subprocess, "check_output", side_effect=git_show):
            sources = wrapper.source_manifest("a" * 40)
        self.assertEqual(len(sources), 27)
        self.assertEqual(tuple(sources)[24:], tuple(str((HERE / name).relative_to(wrapper.REPO))
                                                  for name in wrapper.EXTRA_SOURCES))
        with mock.patch.object(wrapper.runner, "source_manifest", side_effect=lambda commit: dict(original)), \
             mock.patch.object(wrapper.subprocess, "check_output", return_value=b"changed"), \
             self.assertRaisesRegex(ValueError, "committed bytes"):
            wrapper.source_manifest("a" * 40)


if __name__ == "__main__":
    unittest.main()
