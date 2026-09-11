"""Synthetic CPU tests for audit_reexecution.py. No dataset or GPU access."""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("iteration006_audit_reexecution_tested", HERE / "audit_reexecution.py")
a = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(a)
torch, np = a.torch, a.np

PLAN_TREE_SHA256 = "93a91e80702d029359960b3bbc5ed607f147ac56e8fe70e39cc29fb77fbed0e7"


def completion_fixture(store, *, include_test=False):
    rows = [{"step": step} for step in range(1, 2001)]
    grid = [{"step": step, "accuracy": step / 2000, "cross_entropy": 3 - step / 2000, "count": 5000}
            for step in range(0, 2001, 100)]
    results, bindings = [], []
    for identity in a.expected_audit_cells():
        states = {
            "final": {"weight": torch.tensor([2000.])},
            "min_val_ce": {"weight": torch.tensor([2000.])},
            "max_val_accuracy": {"weight": torch.tensor([2000.])},
            "warmup100": {"weight": torch.tensor([100.])},
        }
        binding = store.checkpoints(identity["run_key"] + ".pt", states, **identity)
        noise = identity["replacement_probability"]
        result = {
            **identity, "schema_version": 1, "mode": "audit_reexecution", "audit_bundle": a.BUNDLE,
            "steps": 2000, "instrumented": True, "all_invariant_gates_passed": True,
            "warmup_checks_passed": True, "delivery_gate_steps": 2000, "measurement_state_checks": 22,
            "plan_sha256": "a" * 64, "steps_raw": rows, "validation_trajectory": grid,
            "trajectory_parameter_sha256": ["p"] * 2000,
            "estimation_rank_by_step": [32] * 2000, "repair_count_by_step": [0] * 2000,
            "step_elapsed_seconds": [0.] * 2000,
            "warmup_trajectory_hashes": [{"step": step} for step in range(1, 101)],
            "warmup_observer_hashes": ["o"] * 100,
            "checkpoint_steps": {"final": 2000, "min_val_ce": 2000,
                                 "max_val_accuracy": 2000, "warmup100": 100},
            "checkpoint_sha256": {name: a.producer.tree_hash(state) for name, state in states.items()},
            "checkpoint_path": binding["path"],
            "realized_replacement_fraction": 0. if noise == 0 else .9,
            "realized_incorrect_fraction": 0. if noise == 0 else .81,
        }
        if include_test:
            result["test"] = {name: {"accuracy": .5, "cross_entropy": 1., "count": 10000}
                              for name in a.CHECKPOINTS}
        results.append(result)
        bindings.append(binding)
    return results, bindings


class AuditReexecutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
            raise RuntimeError("Audit tests require CUDA_VISIBLE_DEVICES=''")
        torch.set_num_threads(1)
        if torch.cuda.is_initialized():
            raise RuntimeError("Audit tests must not initialize CUDA")

    def test_plan_is_exact_independent_seedsequence_recipe(self):
        with mock.patch.object(a.producer.h, "make_plan", side_effect=AssertionError("producer plan called")):
            first = a.make_audit_plan()
            second = a.make_audit_plan()
        a.validate_audit_plan(first)
        self.assertEqual(first["seed"], 60006)
        self.assertEqual(first["initialization_seed"], 2693870106)
        self.assertEqual(a.producer.tree_hash(first), PLAN_TREE_SHA256)
        self.assertTrue(all(np.array_equal(first[key], second[key]) for key in first if isinstance(first[key], np.ndarray)))
        self.assertEqual(len(np.unique(np.concatenate([
            first["train_indices"], first["validation_indices"], first["auxiliary_indices"]]))), 15000)

    def test_plan_validation_rejects_any_recipe_drift(self):
        plan = a.make_audit_plan(100)
        changed = {**plan, "training_batches": plan["training_batches"].copy()}
        changed["training_batches"][0, 0] ^= 1
        with self.assertRaisesRegex(AssertionError, "SeedSequence"):
            a.validate_audit_plan(changed, 100)
        with self.assertRaises(AssertionError):
            a.audit_rng(7)

    def test_six_cells_and_separate_source_namespaces(self):
        cells = a.expected_audit_cells()
        self.assertEqual(len(cells), 6)
        self.assertEqual({row["seed"] for row in cells}, {60006})
        self.assertEqual({row["arm"] for row in cells}, set(a.ARMS))
        self.assertEqual({row["replacement_probability"] for row in cells}, {0., .9})
        primary = {str(path.relative_to(a.REPO)) for path in a.producer.source_paths()}
        audit = set(a.audit_source_hashes())
        self.assertEqual(len(primary), 15)
        self.assertEqual(len(audit), 2)
        self.assertTrue(primary.isdisjoint(audit))
        self.assertNotIn("audit_reexecution.py", " ".join(primary))

    def test_audit_caps_are_tighter_without_changing_frozen_classes(self):
        guard = a.AuditGuard("cpu")
        self.assertEqual(guard.limit, 300)
        self.assertEqual(a.storage.Guard(False, "cpu").limit, 900)
        with tempfile.TemporaryDirectory() as tmp:
            store = a.AuditStore("test", synthetic_parent=tmp)
            with mock.patch.object(store, "bytes_used", return_value=a.ARTIFACT_CAP):
                with self.assertRaisesRegex(RuntimeError, "256-MiB"):
                    store.capacity(1)
        self.assertEqual(a.storage.GIB, 1024 ** 3)

    def test_complete_training_gate_accepts_only_all_six_pretest_cells(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = a.AuditStore("test", synthetic_parent=tmp)
            results, bindings = completion_fixture(store)
            a.complete_training_gate(results, bindings, "a" * 64)
            with self.assertRaises(AssertionError):
                a.complete_training_gate(results[:-1], bindings[:-1], "a" * 64)
            results[0]["test"] = {}
            with self.assertRaisesRegex(AssertionError, "early test"):
                a.complete_training_gate(results, bindings, "a" * 64)

    def test_global_test_loader_stays_behind_six_cell_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = a.AuditStore("test", synthetic_parent=tmp)
            results, bindings = completion_fixture(store)
            loader = mock.Mock(return_value=("images", "labels"))
            manifest = {"all_training_completed_utc": "2026-01-01T00:00:00+00:00"}
            with self.assertRaises(AssertionError):
                a.producer.load_test_with_gate(results[:-1], bindings[:-1], manifest, loader,
                                                a.expected_audit_cells())
            loader.assert_not_called()
            self.assertEqual(a.producer.load_test_with_gate(results, bindings, manifest, loader,
                                                             a.expected_audit_cells()), ("images", "labels"))
            loader.assert_called_once()

    def test_four_contrasts_have_fixed_orientation_and_no_pooling(self):
        rows = []
        accuracies = {
            (0., "current32"): .70, (0., "lagged32"): .68, (0., "lagged32_current_norm"): .72,
            (.9, "current32"): .40, (.9, "lagged32"): .45, (.9, "lagged32_current_norm"): .43,
        }
        for identity in a.expected_audit_cells():
            metric = {"accuracy": accuracies[(identity["replacement_probability"], identity["arm"])],
                      "cross_entropy": 1., "count": 10000}
            rows.append({**identity, "checkpoint_steps": {"max_val_accuracy": 500},
                         "test": {"max_val_accuracy": metric}})
        contrasts = a.primary_contrasts(rows)
        self.assertEqual(len(contrasts), 4)
        for actual, expected in zip(
                [row["accuracy_difference"] for row in contrasts], [-.02, .02, .05, .03]):
            self.assertAlmostEqual(actual, expected)
        self.assertTrue(all(row["audit_bundle"] == 60006 for row in contrasts))
        self.assertNotIn("mean", json.dumps(contrasts))

    def test_primary_completion_gate_checks_terminal_environment_and_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train, test = [], []
            for group, target in (("train", train), ("test", test)):
                for index in range(2):
                    path = root / f"{group}-{index}.bin"
                    path.write_bytes(f"{group}-{index}".encode())
                    target.append(a.storage.artifact(path))
            manifest = {
                "mode": "full", "status": "complete", "completed_runs": 36,
                "test_evaluations": 144, "all_gates_passed": True, "official_test_opened": True,
                "source_sha256": {"primary": "digest"}, "environment": {"synthetic": True},
                "training_data_artifacts": train, "test_data_artifacts": test,
            }
            path = root / "execution.json"
            path.write_text(json.dumps(manifest))
            with mock.patch.object(a, "PRIMARY_EXECUTION", path), \
                    mock.patch.object(a.storage, "verify_artifact", wraps=a.storage.verify_artifact) as verify:
                binding, returned_test = a.primary_completion_gate(
                    {"primary": "digest"}, {"synthetic": True}, train)
                self.assertEqual(returned_test, test)
                self.assertEqual([call.args[0] for call in verify.call_args_list], train)
                a.storage.verify_artifact(binding)
                manifest["status"] = "test"
                path.write_text(json.dumps(manifest))
                with self.assertRaisesRegex(AssertionError, "not terminal"):
                    a.primary_completion_gate({"primary": "digest"}, {"synthetic": True}, train)

    def test_source_stability_checks_both_maps(self):
        primary, audit = {"p": "1"}, {"a": "2"}
        with mock.patch.object(a.producer, "source_hashes", return_value=primary), \
                mock.patch.object(a, "audit_source_hashes", return_value=audit):
            a.source_stability_gate(primary, audit)
            with self.assertRaisesRegex(AssertionError, "Audit scientific"):
                a.source_stability_gate(primary, {"a": "changed"})

    def test_committed_gate_rejects_script_changed_after_import(self):
        primary = {str(path.relative_to(a.REPO)): "p" for path in a.producer.source_paths()}
        audit = a.audit_source_hashes()
        script_key = str(Path(a.__file__).resolve().relative_to(a.REPO))
        with mock.patch.object(a.producer, "committed_source_gate", return_value=primary), \
                mock.patch.object(a, "_committed_hashes", return_value={**audit, script_key: "0" * 64}):
            with self.assertRaisesRegex(AssertionError, "changed after import"):
                a.committed_source_gate()

    def test_cli_requires_both_separate_go_flags(self):
        for argv in (["audit"], ["audit", "--audit-go"], ["audit", "--audit-reexecution"]):
            with mock.patch.object(a.sys, "argv", argv), mock.patch.object(a, "launch") as launch:
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    a.main()
                launch.assert_not_called()
        with tempfile.TemporaryDirectory() as tmp:
            outer = Path(tmp) / "outer.log"
            with mock.patch.object(a.sys, "argv", ["audit", "--audit-reexecution", "--audit-go"]), \
                    mock.patch.object(a, "OUTER_LOG", outer), mock.patch.object(a, "launch") as launch:
                a.main()
                launch.assert_called_once_with()
                self.assertTrue(outer.is_file())

    def test_outer_log_captures_pre_store_failure_traceback(self):
        with tempfile.TemporaryDirectory() as tmp:
            outer = Path(tmp) / "outer.log"
            with mock.patch.object(a.sys, "argv", ["audit", "--audit-reexecution", "--audit-go"]), \
                    mock.patch.object(a, "OUTER_LOG", outer), \
                    mock.patch.object(a, "launch", side_effect=RuntimeError("pre-store-synthetic")), \
                    contextlib.redirect_stderr(io.StringIO()), self.assertRaisesRegex(RuntimeError, "pre-store"):
                a.main()
            content = outer.read_text()
            self.assertIn("Traceback", content)
            self.assertIn("pre-store-synthetic", content)


if __name__ == "__main__":
    unittest.main()
