"""Synthetic CPU-only I17 runner tests; no scientific input or replay."""
from __future__ import annotations

import copy
import importlib.util
import json
import math
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import numpy as np
import torch
import run_gain_controls as runner


def fixtures():
    generator = torch.Generator().manual_seed(17101)
    x = torch.rand((29, 4), generator=generator)
    labels = torch.randint(2, (29,), generator=generator)
    data = {"x": x[:16].clone(), "clean": labels[:16].clone(),
            "noisy": (1 - labels[:16]).clone(), "vx": x[16:22].clone(),
            "vy": labels[16:22].clone(), "ax": x[22:].clone(), "ay": labels[22:].clone()}
    plan = {"seed": 217, "training_batches": np.tile(np.arange(64, dtype=np.int64) % 16, (110, 1))}
    states, starts = {}, {}
    for target in runner.TARGETS:
        runner.old.seed_all(17102)
        model = runner.i9.make_model(17102, "cpu", input_dim=4, width=3, classes=2)
        optimizer = runner.c14.make_optimizer(model, "sgdm", .03)
        tracker = runner.i9.make_tracker(model, optimizer)
        y = data["clean"] if target == "clean" else data["noisy"]
        for step in range(100):
            indices = torch.as_tensor(plan["training_batches"][step])
            runner.c14.train_step(model, optimizer, tracker, data["x"][indices], y[indices], "raw", "sgdm")
        states[target] = runner.c14.snapshot(model, optimizer, tracker, "sgdm", .03)
        starts[target] = {"horizon": 100, **runner.old.evaluate(model, data)}
    return data, plan, states, starts


class NoWrite:
    def save(self, *args, **kwargs):
        raise AssertionError("invalid input reached write")

    def check(self):
        return None


class GainRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        if torch.get_num_interop_threads() != 1:
            torch.set_num_interop_threads(1)
        cls.data, cls.plan, cls.states, cls.starts = fixtures()

    def call_branch(self, run, policy="scalar_k0p5", target="clean", **kwargs):
        return runner.run_branch(run, self.states[target], self.data, self.plan,
                                 target, policy, self.starts[target], smoke=True,
                                 end=110, horizons=(100, 110), **kwargs)

    def test_all_eight_synthetic_branches_complete_and_pair(self):
        with tempfile.TemporaryDirectory(prefix="i17-unit-") as directory:
            run = runner.old.Run(directory, "cpu-test", 60)
            entries = []
            for target in runner.TARGETS:
                for policy in runner.core.REAL_POLICIES:
                    entry = self.call_branch(run, policy, target)
                    entries.append(entry)
                    branch = json.loads((run.path / entry["artifact"]["name"]).read_text())
                    self.assertEqual(branch["schema"], "i17_gain_branch_v1")
                    self.assertEqual(branch["status"], "complete")
                    self.assertEqual(branch["completed_updates"], 10)
                    self.assertEqual([point["horizon"] for point in branch["curve"]], [100, 110])
                    self.assertEqual(len(branch["steps"]), 10)
                    self.assertIsNone(branch["failure"])
                    self.assertEqual(branch["parent_state_digest"], runner.i9.tree_digest(self.states[target]))
                    state = torch.load(run.path / branch["checkpoints"][-1]["full_state"]["name"],
                                       weights_only=False, map_location="cpu")
                    self.assertEqual(runner.i9.tree_digest(state), branch["checkpoints"][-1]["full_state_digest"])
                    self.assertEqual(state["tracker"]["step_count"], 110)
            checks = runner.validate_first_step_pairs(entries)
            self.assertEqual(len(checks), 2)
            self.assertTrue(all(row["status"] == "pass" for row in checks))
            forecast = runner.smoke_forecast(entries)
            self.assertGreater(forecast["confirmation_seconds"], 60)

    def test_invalid_policy_and_target_reject_without_write(self):
        for policy in ("scalar_k0", "k0p5", 0.5, None, True, "spectral_other"):
            with self.subTest(policy=policy), self.assertRaises(ValueError):
                self.call_branch(NoWrite(), policy)
        with self.assertRaises(ValueError):
            runner.run_branch(NoWrite(), self.states["clean"], self.data, self.plan,
                              "other", "scalar_k1", self.starts["clean"], smoke=True,
                              end=110, horizons=(100, 110))

    def test_invalid_dimensions_batches_and_h100_reject_without_write(self):
        for end, horizons in ((111, (100, 111)), (110, (100, 105, 110)), (2000, runner.HORIZONS)):
            with self.assertRaises(ValueError):
                runner.run_branch(NoWrite(), self.states["clean"], self.data, self.plan,
                    "clean", "scalar_k1", self.starts["clean"], smoke=True, end=end, horizons=horizons)
        bad_plan = copy.deepcopy(self.plan)
        bad_plan["training_batches"][0, 0] = 10000
        with self.assertRaises(ValueError):
            runner.run_branch(NoWrite(), self.states["clean"], self.data, bad_plan,
                "clean", "scalar_k1", self.starts["clean"], smoke=True, end=110, horizons=(100, 110))
        bad = copy.deepcopy(self.states["clean"])
        bad["tracker"]["step_count"] = 99
        with self.assertRaises(ValueError):
            runner.run_branch(NoWrite(), bad, self.data, self.plan, "clean", "scalar_k1",
                              self.starts["clean"], smoke=True, end=110, horizons=(100, 110))

    def test_evaluation_seam_failure_before_new_step(self):
        expected = copy.deepcopy(self.starts["clean"])
        expected["validation"]["clean_ce"] += 1
        with mock.patch.object(runner.core, "gain_step", side_effect=AssertionError("new step")):
            with self.assertRaisesRegex(ValueError, "evaluation seam"):
                runner.run_branch(NoWrite(), self.states["clean"], self.data, self.plan,
                    "clean", "scalar_k1", expected, smoke=True, end=110, horizons=(100, 110))

    def test_typed_failure_saved_but_structural_error_propagates(self):
        with tempfile.TemporaryDirectory(prefix="i17-unit-") as directory:
            run = runner.old.Run(directory, "typed", 60)
            with mock.patch.object(runner.core, "gain_step", side_effect=runner.core.NumericalFailure("synthetic nonfinite")):
                entry = self.call_branch(run)
            branch = json.loads((run.path / entry["artifact"]["name"]).read_text())
            self.assertEqual(branch["status"], "numerical_failure")
            self.assertEqual(branch["completed_updates"], 0)
            self.assertEqual(branch["curve"], [self.starts["clean"]])
            self.assertIsNone(branch["selected_horizons"])
            self.assertIsNone(branch["first_step_digests"])
            self.assertTrue((run.path / branch["failure"]["state_artifact"]["name"]).is_file())
            run2 = runner.old.Run(directory, "structural", 60)
            with mock.patch.object(runner.core, "gain_step", side_effect=RuntimeError("synthetic CUDA error")):
                with self.assertRaisesRegex(RuntimeError, "synthetic CUDA"):
                    self.call_branch(run2)
            self.assertEqual(run2.artifacts, [])

    def test_duplicate_branch_rejected_without_replay(self):
        with tempfile.TemporaryDirectory(prefix="i17-unit-") as directory:
            run = runner.old.Run(directory, "duplicate", 60)
            self.call_branch(run)
            with mock.patch.object(runner.core, "gain_step", side_effect=AssertionError("replay")):
                with self.assertRaisesRegex(ValueError, "already has artifacts"):
                    self.call_branch(run)

    def test_pair_membership_and_digest_mismatch_rejected(self):
        rows = [{"seed": 217, "target": "clean", "policy": policy, "status": "complete",
                 "parent_state_digest": "a", "parent_evaluation_digest": "b",
                 "first_step_digests": {name: "c" for name in ("raw_gradient", "post_observer", "old_momentum_buffer")}}
                for policy in runner.core.REAL_POLICIES]
        self.assertEqual(runner.validate_first_step_pairs(rows)[0]["status"], "pass")
        with self.assertRaises(ValueError):
            runner.validate_first_step_pairs(rows[:-1])
        bad = copy.deepcopy(rows)
        bad[-1]["first_step_digests"]["raw_gradient"] = "different"
        with self.assertRaises(ValueError):
            runner.validate_first_step_pairs(bad)
        bad = copy.deepcopy(rows)
        bad[-1]["first_step_digests"] = None
        with self.assertRaises(ValueError):
            runner.validate_first_step_pairs(bad)
        bad[-1]["status"] = "numerical_failure"
        self.assertEqual(runner.validate_first_step_pairs(bad)[0]["status"], "partial_numerical_evidence")

    def test_forecast_uses_slowest_policy_not_pooled_average(self):
        rows = [{"seed": 217, "target": target, "policy": policy, "status": "complete",
                 "completed_updates": 10, "update_seconds": float(index + 1)}
                for target in runner.TARGETS for index, policy in enumerate(runner.core.REAL_POLICIES)]
        forecast = runner.smoke_forecast(rows)
        self.assertAlmostEqual(forecast["confirmation_seconds"], 1.5 * .4 * 45600 + 60)
        for key, value in (("update_seconds", math.nan), ("update_seconds", -1.0),
                           ("update_seconds", True), ("completed_updates", 9),
                           ("status", "numerical_failure"), ("seed", 216)):
            bad = copy.deepcopy(rows)
            bad[0][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                runner.smoke_forecast(bad)
        with self.assertRaises(ValueError):
            runner.smoke_forecast(rows[:-1])

    def synthetic_phase(self, directory, omit_checkpoint=False):
        root = Path(directory)
        commit, sources = "a" * 40, {"synthetic-source": "b" * 64}
        run = runner.old.Run(root, "smoke", 60)
        with (root / "attempt-smoke.json").open("x") as handle:
            json.dump({"schema": "i17_attempt_v1", "phase": "smoke", "frozen_commit": commit,
                       "source_hashes": sources, "restart": "forbidden"}, handle)
        run.save("manifest.json", {"schema": "i17_manifest_v1", "phase": "smoke",
                 "frozen_commit": commit, "source_hashes": sources,
                 "real_policies": runner.core.REAL_POLICIES, "test_only_policies": runner.core.TEST_ONLY_POLICIES})
        entries = []
        for target in runner.TARGETS:
            run.save(f"synthetic-parent-{target}.pt", {}, tensor=True)
            for policy in runner.core.REAL_POLICIES:
                identity = f"s217-sgdm-{target}-{policy}"
                state_record = run.save(f"state-{identity}-h110.pt", {}, tensor=True)
                entry = {"id": identity, "seed": 217, "target": target, "policy": policy,
                         "status": "complete", "completed_updates": 10, "update_seconds": .01,
                         "parent_state_digest": "c" * 64, "parent_evaluation_digest": "d" * 64,
                         "first_step_digests": {key: "e" * 64 for key in
                           ("raw_gradient", "post_observer", "old_momentum_buffer")}}
                branch = {**entry, "schema": "i17_gain_branch_v1", "failure": None,
                          "curve": [{"horizon": 100}, {"horizon": 110}], "steps": [{}] * 10,
                          "checkpoints": [{"horizon": 110, "relative_horizon": 10,
                              "full_state": state_record, "full_state_digest": "f" * 64}]}
                if omit_checkpoint:
                    branch.pop("checkpoints")
                entry["artifact"] = run.save(f"branch-{identity}.json", branch)
                entries.append(entry)
        run.save("branches.json", {"entries": entries,
                 "first_step_pair_checks": runner.validate_first_step_pairs(entries),
                 "synthetic_warmup_updates": 200, "new_gain_updates": 80})
        run.terminal("completion.json", {"schema": "i17_completion_v1", "status": "complete",
                     "phase": "smoke", "frozen_commit": commit, "source_hashes": sources,
                     "branches": 8, "numerical_failures": 0, "all_requested_endpoints_present": True,
                     "completed_training_updates": 280, "artifacts": run.artifacts})
        return root, sources, commit

    def test_smoke_admission_checks_actual_roster_and_corruption(self):
        with tempfile.TemporaryDirectory(prefix="i17-unit-") as directory:
            root, sources, commit = self.synthetic_phase(directory)
            self.assertLess(runner.verify_smoke(root, sources, commit)["confirmation_seconds"], 1800)
            with (root / "smoke/undeclared.json").open("x") as handle:
                json.dump({}, handle)
            with self.assertRaisesRegex(ValueError, "physical artifact"):
                runner.verify_smoke(root, sources, commit)
        with tempfile.TemporaryDirectory(prefix="i17-unit-") as directory:
            root, sources, commit = self.synthetic_phase(directory)
            with (root / "smoke/manifest.json").open("a") as handle:
                handle.write(" ")
            with self.assertRaisesRegex(ValueError, "Artifact content"):
                runner.verify_smoke(root, sources, commit)

    def test_smoke_admission_rejects_wrong_source_and_attempt(self):
        with tempfile.TemporaryDirectory(prefix="i17-unit-") as directory:
            root, sources, commit = self.synthetic_phase(directory)
            with self.assertRaisesRegex(ValueError, "attempt binding"):
                runner.verify_smoke(root, {"wrong": "source"}, commit)
            with self.assertRaisesRegex(ValueError, "attempt binding"):
                runner.verify_smoke(root, sources, "f" * 40)

    def test_smoke_checkpoint_association_is_required(self):
        with tempfile.TemporaryDirectory(prefix="i17-unit-") as directory:
            root, sources, commit = self.synthetic_phase(directory, omit_checkpoint=True)
            with self.assertRaisesRegex(ValueError, "checkpoint association"):
                runner.verify_smoke(root, sources, commit)


if __name__ == "__main__":
    unittest.main()
