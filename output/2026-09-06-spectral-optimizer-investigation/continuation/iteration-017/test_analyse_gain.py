"""Synthetic CPU-only tests; no scientific artifact or scientific replay is used."""
from __future__ import annotations

import copy
from contextlib import ExitStack
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import torch

HERE = Path(__file__).resolve().parent
_SPEC = importlib.util.spec_from_file_location("_i17_analysis_under_test", HERE / "analyse_gain.py")
analysis = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(analysis)


def evaluation(offset=0.0):
    fixed, soft = 1.2 + offset, 1.0 + offset
    return {"train": {"clean_ce": .8 + offset, "soft_ce": soft, "clean_accuracy": .6,
        "mean_max_probability": .7, "mean_true_label_probability": .5, "fixed_ce": fixed,
        "fixed_accuracy": .4, "fixed_minus_soft_ce": fixed - soft},
        "validation": {"clean_ce": .9 + offset, "soft_ce": 1.1 + offset, "clean_accuracy": .55,
            "mean_max_probability": .65, "mean_true_label_probability": .45},
        "auxiliary": {"clean_ce": .85 + offset, "soft_ce": 1.05 + offset, "clean_accuracy": .57,
            "mean_max_probability": .66, "mean_true_label_probability": .46}}


def leakage(norm, fraction=.25):
    return {"squared_norm": norm * norm, "outside_squared_norm": norm * norm * fraction,
            "fraction": fraction, "reason": None}


def component(norm=2.0, fraction=.25):
    return {"norm": norm, "squared_energy": norm * norm, "current_basis_leakage": leakage(norm, fraction)}


def step_record(step=101, policy="scalar_k0p5"):
    scalar, k = policy in analysis.K_BY_LABEL, analysis.K_BY_LABEL.get(policy)
    gain = 1 - .9 * k if scalar else None
    delivered = gain * 2 if scalar else 2.0
    data = .03 * delivered
    components = {name: component() for name in analysis.COMPONENTS}
    components["delivered_buffer"] = component(delivered)
    components["selected_old_momentum_buffer"] = component(k * 2 if scalar else 2.0)
    return {"step": step, "relative_step": step - 100,
        "path_statistics": {"data_step_squared_energy": data * data, "data_path_length_increment": data,
                            "raw_gradient_dot_data_step": -.01},
        "schema": "i17_gain_step_v1", "base": "sgdm", "lr": .03, "policy": policy,
        "family": "scalar" if scalar else "spectral", "k": k, "loss": 1.0,
        "observer": {"step_before": step - 1, "step_after": step, "filtering_active": True, "used": True, "basis_rank": 2},
        "gradient_filter_applied": not scalar, "native_projection_used_for_recurrence": not scalar,
        "normalized_delivery": {"operator": "(1-rho*k)*b_t" if scalar else "(I-rho*A_t)*b_t",
            "scalar_gain_correction": gain, "manual_parameter_update": True, "unnormalized_buffer_retained": True,
            "delivery_literal": "raw_gradient" if k == 1 else None,
            "history_literal": "old_buffer" if k == 1 else None if scalar else "native_action_old_buffer",
            "unnormalized_buffer_norm": 2.0, "delivered_buffer_norm": delivered},
        "gradient": {"raw_norm": 2.0, "native_projection_norm": 2.0, "post_mean_norm": 2.0, "recurrence_input_norm": 2.0},
        "components": components,
        "displacement": {"total_norm": data, "data_norm": data, "ideal_data_norm": data, "nominal_decay_norm": .1,
            "actual_minus_ideal_data_norm": 1e-8, "actual_minus_ideal_relative": 1e-8 / data,
            "total_current_basis_leakage": leakage(data), "data_current_basis_leakage": leakage(data),
            "ideal_data_current_basis_leakage": leakage(data), "raw_gradient_dot_data_delta": -.01,
            "post_mean_dot_data_delta": -.005, "delivered_buffer_dot_data_delta": -.003},
        "action_diagnostics": {"current_basis_orthogonality_error": .1,
            "new_buffer_action_idempotence_defect": {"norm": .2, "homogeneous_roundoff_reference": 1e-7, "enforced": False}},
        "decay": {"coefficient": .01, "factor": .9997, "manual_before_data_step": True,
                  "manual_actual_norm": .1, "manual_minus_nominal_norm": 1e-8},
        "algebra_residuals": {name: {"norm": 0.0, "homogeneous_error_bound": 1e-6} for name in analysis.RESIDUALS},
        "digests": ({name: hashlib.sha256(name.encode()).hexdigest() for name in analysis.DIGESTS} if step == 101 else None)}


def branch_record(policy="scalar_k0p5", target="clean"):
    curve = [{"horizon": horizon, **evaluation(-.01 * index)} for index, horizon in enumerate((100, 110))]
    steps = [step_record(step, policy) for step in range(101, 111)]
    return {"schema": "i17_gain_branch_v1", "id": f"s217-sgdm-{target}-{policy}", "seed": 217,
        "target": target, "policy": policy, "base": "sgdm", "lr": .03, "parent_horizon": 100,
        "parent_state_digest": "a" * 64, "parent_evaluation_digest": analysis.tree_digest(curve[0]),
        "status": "complete", "requested_updates": 10, "completed_updates": 10, "last_completed_horizon": 110,
        "curve": curve, "steps": steps, "update_seconds": .001,
        "checkpoints": [{"horizon": 110, "relative_horizon": 10,
            "full_state": {"name": "state.pt", "bytes": 1, "sha256": "b" * 64}, "full_state_digest": "c" * 64}],
        "failure": None, "first_step_digests": steps[0]["digests"], "selected_horizons": analysis.old._selection(curve)}


def scalar_fixture():
    branches = {}
    for seed in analysis.SEEDS:
        for target in analysis.TARGETS:
            for ordinal, policy in enumerate(analysis.POLICIES):
                curve = []
                for horizon in analysis.HORIZONS:
                    point = {"horizon": horizon, **evaluation()}
                    point["validation"]["clean_ce"] = 1.0
                    point["validation"]["clean_accuracy"] = .5
                    point["auxiliary"]["clean_ce"] = .9 - ordinal * .1
                    point["auxiliary"]["clean_accuracy"] = .4 + ordinal * .1
                    curve.append(point)
                branches[seed, target, policy] = {"seed": seed, "target": target, "policy": policy,
                    "status": "complete", "curve": curve, "steps": [], "failure": None}
    return branches


def record(path):
    return {"name": path.name, "bytes": path.stat().st_size, "sha256": analysis.sha256(path)}


def save_json(path, value):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, allow_nan=False)
    return record(path)


def synthetic_smoke(root):
    directory = root / "smoke"
    directory.mkdir()
    artifacts, entries = [], []
    state = {"vector": torch.arange(3), "scalar": torch.tensor(1.)}
    state_digest = analysis.tree_digest(state)
    manifest = {"schema": "i17_manifest_v1", "phase": "smoke", "frozen_commit": analysis.SMOKE_COMMIT,
        "source_hashes": {name: "b" * 64 for name in analysis.ACQUISITION_SOURCES},
        "torch_version": "synthetic", "numpy_version": "synthetic", "python_version": "synthetic", "gpu": "synthetic",
        "real_policies": list(analysis.NEW_POLICIES), "test_only_policies": ["scalar_k0"], "horizons": list(analysis.HORIZONS),
        "smoke_forecast": None, "timing_safety_factor": 1.5, "cloud_spend_usd": 0, "authorized_budget_usd": 100,
        "shared_artifact_cap_bytes": analysis.ARTIFACT_CAP, "wall_limit_seconds": 100, "runtime_directory": "runtime",
        "runtime_bytes_count_in_cap": True}
    artifacts.append(save_json(directory / "manifest.json", manifest))
    for target in analysis.TARGETS:
        path = directory / f"synthetic-parent-{target}.pt"
        torch.save(state, path)
        artifacts.append(record(path))
        for policy in analysis.NEW_POLICIES:
            branch = branch_record(policy, target)
            branch["parent_state_digest"] = state_digest
            path = directory / f"state-{branch['id']}-h110.pt"
            torch.save(state, path)
            terminal = record(path)
            artifacts.append(terminal)
            branch["checkpoints"][0]["full_state"] = terminal
            branch["checkpoints"][0]["full_state_digest"] = state_digest
            artifact = save_json(directory / f"branch-{branch['id']}.json", branch)
            artifacts.append(artifact)
            fields = ("id", "seed", "target", "policy", "status", "completed_updates", "parent_state_digest",
                      "parent_evaluation_digest", "first_step_digests", "update_seconds")
            entries.append({name: branch[name] for name in fields} | {"artifact": artifact})
    pairs = [{"seed": 217, "target": target, "available_first_steps": 4, "status": "pass"} for target in analysis.TARGETS]
    envelope = {"entries": entries, "first_step_pair_checks": pairs, "synthetic_warmup_updates": 200, "new_gain_updates": 80}
    artifacts.append(save_json(directory / "branches.json", envelope))
    completion = {"schema": "i17_completion_v1", "status": "complete", "phase": "smoke",
        "source_hashes": manifest["source_hashes"], "frozen_commit": manifest["frozen_commit"], "branches": 8,
        "numerical_failures": 0, "all_requested_endpoints_present": True, "completed_training_updates": 280,
        "update_seconds": sum(row["update_seconds"] for row in entries), "elapsed_seconds": .5,
        "peak_torch_gpu_bytes": 0, "shared_artifact_bytes_before_completion": 1000, "artifacts": artifacts}
    save_json(directory / "completion.json", completion)
    return manifest, completion, {row["name"]: row for row in artifacts}


class GainAnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            if torch.get_num_interop_threads() != 1:
                raise

    def test_every_new_policy_schema_and_nonideal_action_is_not_rejected(self):
        for policy in analysis.NEW_POLICIES:
            with self.subTest(policy=policy):
                value = branch_record(policy)
                value["steps"][0]["components"]["raw_gradient"] = component(2.0, 1.2)
                audit = analysis.Audit()
                self.assertIsNotNone(analysis.validate_branch(value, "smoke", audit, "synthetic"))
                self.assertEqual(audit.errors, [])

    def test_actual_tiny_cpu_core_records_match_independent_schema(self):
        # Test-only import: the analyzer itself imports no acquisition module.
        spec = importlib.util.spec_from_file_location("_i17_tiny_cpu_fixture", HERE / "test_gain_core.py")
        fixture = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fixture)
        maker = fixture.GainCoreTests()
        for policy in analysis.NEW_POLICIES:
            model, optimizer, tracker, x, target = maker.fixture()
            value = fixture.core.gain_step(model, optimizer, tracker, x, target, policy, capture_digests=True)
            norm = value["displacement"]["data_norm"]
            row = {"step": 101, "relative_step": 1,
                   "path_statistics": {"data_step_squared_energy": norm * norm,
                       "data_path_length_increment": norm,
                       "raw_gradient_dot_data_step": value["displacement"]["raw_gradient_dot_data_delta"]}, **value}
            audit = analysis.Audit()
            analysis.validate_step(row, 101, policy, audit, "tiny CPU " + policy)
            self.assertEqual(audit.errors, [])

    def test_schema_mutations_fail_closed(self):
        mutations = (
            lambda row: row["normalized_delivery"].update(unnormalized_buffer_retained=False),
            lambda row: row["normalized_delivery"].update(scalar_gain_correction=.9),
            lambda row: row["path_statistics"].update(data_step_squared_energy=.4),
            lambda row: row["algebra_residuals"]["delivered_buffer_definition"].update(norm=1.0),
            lambda row: row["action_diagnostics"]["new_buffer_action_idempotence_defect"].update(enforced=True),
            lambda row: row["displacement"].update(actual_minus_ideal_relative=.9),
            lambda row: row["gradient"].update(raw_norm=9.0),
            lambda row: row.update(loss=float("nan")),
            lambda row: row.update(k=True),
        )
        for index, mutation in enumerate(mutations):
            with self.subTest(index=index):
                value, audit = step_record(), analysis.Audit()
                mutation(value)
                analysis.validate_step(value, 101, "scalar_k0p5", audit, "mutated")
                self.assertTrue(audit.errors)

    def test_typed_training_and_evaluation_failure_are_retained(self):
        for evaluation_failure in (False, True):
            branch = branch_record()
            if not evaluation_failure:
                branch["steps"] = branch["steps"][:-1]
                branch["completed_updates"] = 9
                branch["last_completed_horizon"] = 109
            branch["curve"] = branch["curve"][:1]
            branch["checkpoints"] = []
            branch["status"] = "numerical_failure"
            branch["selected_horizons"] = None
            branch["failure"] = {"type": "NumericalFailure", "message": "synthetic nonfinite", "attempted_step": 110,
                "completed_step": branch["last_completed_horizon"], "last_valid_horizon": 100,
                "state_artifact": {"name": "failure.pt", "bytes": 0, "sha256": "f" * 64}}
            audit = analysis.Audit()
            analysis.validate_branch(branch, "smoke", audit, "failure")
            self.assertEqual(audit.errors, [])
            branch["failure"]["type"] = "RuntimeError"
            analysis.validate_branch(branch, "smoke", audit, "structural")
            self.assertTrue(audit.errors)

    def test_eight_primaries_ties_and_auxiliary_blind_selection(self):
        branches = scalar_fixture()
        choices, primary, per_k = analysis.selection_results(branches)
        self.assertEqual((len(choices), len(primary), len(per_k)), (12, 8, 32))
        self.assertTrue(all(row["scalar"]["k"] == 0 and row["scalar"]["horizon"] == 100 for row in choices))
        self.assertTrue(all(abs(row["effect"]["mean"] - .4) < 1e-12 for row in primary))
        # Earlier horizon beats lower k when the validation value ties.
        branches[200, "clean", "scalar_k0"]["curve"][0]["validation"]["clean_ce"] = 2.0
        point, k = analysis._best_scalar(branches, 200, "clean", analysis.SELECTORS[0])
        self.assertEqual((point["horizon"], k), (100, .5))
        # Auxiliary outcomes do not affect the choice.
        branches[200, "clean", "scalar_k0p5"]["curve"][0]["auxiliary"]["clean_ce"] = 999.0
        self.assertEqual(analysis._best_scalar(branches, 200, "clean", analysis.SELECTORS[0])[1], .5)

    def test_missing_member_does_not_select_or_average_survivors(self):
        branches = scalar_fixture()
        branches[200, "fixed", "scalar_k0p9"]["status"] = "numerical_failure"
        choices, primary, per_k = analysis.selection_results(branches)
        affected = next(row for row in choices if row["seed"] == 200 and row["target"] == "fixed")
        self.assertIsNone(affected["scalar"])
        self.assertIsNotNone(affected["spectral"])
        for row in primary:
            if row["target"] == "fixed":
                self.assertIsNone(row["effect"]["mean"])
                self.assertFalse(row["effect"]["available"])
        unaffected = next(row for row in per_k if row["target"] == "fixed" and row["k_label"] == "scalar_k0")
        self.assertTrue(unaffected["effect"]["available"])

    def test_trajectory_endpoint_and_progress_counts(self):
        branches = scalar_fixture()
        for branch in branches.values():
            branch["curve"][-1]["auxiliary"]["clean_ce"] -= .1
            branch["curve"][-1]["auxiliary"]["clean_accuracy"] += .05
        self.assertEqual(len(analysis.endpoint_effects(branches)), 16)
        self.assertEqual(len(analysis.trajectory_rows(branches)), 30)
        rows = analysis.scheduled_aggregates(branches)
        self.assertEqual(len(rows), 960)
        selected = next(row for row in rows if row["horizon"] == 2000 and row["metric"] == "auxiliary_clean_ce")
        self.assertAlmostEqual(selected["equal_seed_mean_change_from_h100"], -.1)
        branches[200, "clean", "scalar_k0"]["status"] = "numerical_failure"
        missing = next(row for row in analysis.scheduled_aggregates(branches)
                       if row["target"] == "clean" and row["policy"] == "scalar_k0")
        self.assertIsNone(missing["equal_seed_mean"])

    def test_geometry_requires_exact_window_and_includes_ideal_delivery(self):
        branches = {}
        for seed in analysis.SEEDS:
            branches[seed, "clean", "scalar_k0p5"] = {"status": "complete",
                "steps": [step_record(step) for step in range(1001, 2001)]}
        paths, components = analysis.geometry_results(branches)
        late = next(row for row in paths if row["target"] == "clean" and row["policy"] == "scalar_k0p5"
                    and row["window"] == "steps1001_2000")
        early = next(row for row in paths if row["target"] == "clean" and row["policy"] == "scalar_k0p5"
                     and row["window"] == "steps101_2000")
        self.assertTrue(late["available"])
        self.assertAlmostEqual(late["equal_seed_mean"]["ideal_data"]["path_length_sum"], 1000 * .033)
        self.assertFalse(early["available"])
        self.assertTrue(any(row["component"] == "delivered_buffer" and row["available"] for row in components))
        del branches[200, "clean", "scalar_k0p5"]["steps"][-1]
        late = next(row for row in analysis.geometry_results(branches)[0] if row["target"] == "clean"
                    and row["policy"] == "scalar_k0p5" and row["window"] == "steps1001_2000")
        self.assertIsNone(late["equal_seed_mean"])

    def test_maximum_policy_smoke_forecast_and_failure_accounting(self):
        entries = [{"seed": 217, "target": target, "policy": policy, "status": "complete", "completed_updates": 10,
                    "update_seconds": .25 + .01 * index} for target in analysis.TARGETS
                   for index, policy in enumerate(analysis.NEW_POLICIES)]
        rates = {policy: sum(row["update_seconds"] for row in entries if row["policy"] == policy) / 20
                 for policy in analysis.NEW_POLICIES}
        phases = {"smoke": ({"smoke_forecast": None}, {}, {}),
                  "confirmation": ({"smoke_forecast": {"policy_seconds_per_update": rates,
                                     "confirmation_seconds": 1.5 * max(rates.values()) * 45600 + 60}}, {}, {})}
        audit = analysis.Audit()
        amendment = analysis.validate_cross_phase_timing(phases, entries, audit)
        analysis.validate_phase_aggregate({"update_seconds": sum(row["update_seconds"] for row in entries),
            "completed_training_updates": 280, "numerical_failures": 0}, entries, 200, audit, "synthetic")
        self.assertEqual(audit.errors, [])
        self.assertFalse(amendment["original_gate_passed"])
        self.assertTrue(amendment["amended_gate_passed"])
        self.assertEqual(amendment["original_confirmation_limit_seconds"], 1800)
        self.assertEqual(amendment["amended_confirmation_limit_seconds"], 2400)
        phases["confirmation"][0]["smoke_forecast"]["confirmation_seconds"] -= 1
        analysis.validate_cross_phase_timing(phases, entries, audit)
        self.assertTrue(any("forecast" in error for error in audit.errors))
        for row in entries:
            row["update_seconds"] *= 2
        rates = {policy: sum(row["update_seconds"] for row in entries if row["policy"] == policy) / 20
                 for policy in analysis.NEW_POLICIES}
        phases["confirmation"][0]["smoke_forecast"] = {"policy_seconds_per_update": rates,
            "confirmation_seconds": 1.5 * max(rates.values()) * 45600 + 60}
        audit = analysis.Audit()
        amendment = analysis.validate_cross_phase_timing(phases, entries, audit)
        self.assertFalse(amendment["amended_gate_passed"])
        self.assertTrue(any("gate" in error for error in audit.errors))

    def test_phase_and_checkpoint_role_integration_on_synthetic_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            synthetic_smoke(root)
            audit = analysis.Audit()
            with mock.patch.object(analysis, "SMOKE_COMPLETION_SHA", analysis.sha256(root / "smoke/completion.json")):
                phase = analysis.phase_records(root, "smoke", audit)
            branches, entries = analysis._phase_branches(root, "smoke", phase, audit)
            self.assertEqual(audit.errors, [])
            self.assertEqual((len(branches), len(entries), audit.tree_digests), (8, 8, 10))
            extra = root / "smoke/unregistered.json"
            extra_record = save_json(extra, {"unexpected": True})
            phase[2][extra.name] = extra_record
            audit = analysis.Audit()
            analysis._phase_branches(root, "smoke", phase, audit)
            self.assertTrue(any("role membership" in error for error in audit.errors))

    def test_state_hash_digest_tampering_and_runtime_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            state = {"vector": torch.arange(2)}
            path = root / "state.pt"
            torch.save(state, path)
            artifact = record(path)
            branch = branch_record()
            branch["checkpoints"][0]["full_state"] = artifact
            branch["checkpoints"][0]["full_state_digest"] = "0" * 64
            audit = analysis.Audit()
            analysis.validate_checkpoints(branch, {path.name: artifact}, root, audit, "state")
            self.assertTrue(any("complete-state digest" in error for error in audit.errors))
            self.assertEqual(audit.tree_digests, 1)
            (root / "runtime").mkdir()
            os.symlink(root / "runtime", root / "runtime/link")
            audit = analysis.Audit()
            analysis.runtime_inventory(root, audit)
            self.assertTrue(any("not a real directory" in error for error in audit.errors))

    def test_json_duplicate_nonfinite_and_record_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for index, text in enumerate(('{"a":1,"a":2}', '{"a":NaN}')):
                path = root / f"{index}.json"
                path.write_text(text)
                with self.assertRaises(ValueError):
                    analysis.read_json(path)
            audit = analysis.Audit()
            result = analysis.verify_record(root, {"name": "../escape", "bytes": 0, "sha256": "f" * 64}, audit, "traversal")
            self.assertIsNone(result)
            self.assertTrue(audit.errors)

    def test_exact_source_closure_and_worktree_commit_hash_binding(self):
        self.assertEqual(len(analysis.ACQUISITION_SOURCES), 24)
        self.assertEqual(len(analysis.CONFIRMATION_SOURCES), 27)
        self.assertEqual(analysis.CONFIRMATION_SOURCES[:24], analysis.ACQUISITION_SOURCES)
        self.assertEqual([Path(name).name for name in analysis.CONFIRMATION_SOURCES[24:]],
                         ["run_gain_confirmation_v2.py", "test_confirmation_resource.py", "resource-amendment.md"])
        digest = hashlib.sha256(b"synthetic source").hexdigest()
        sources = {name: digest for name in analysis.ACQUISITION_SOURCES}
        amended = {name: digest for name in analysis.CONFIRMATION_SOURCES}
        manifests = [{"phase": "smoke", "frozen_commit": analysis.SMOKE_COMMIT, "source_hashes": sources},
                     {"phase": "confirmation", "frozen_commit": "a" * 40, "source_hashes": amended}]
        with mock.patch.object(analysis, "sha256", return_value=digest), \
             mock.patch.object(Path, "is_file", return_value=True), \
             mock.patch.object(Path, "is_symlink", return_value=False), \
             mock.patch.object(analysis.subprocess, "check_output", return_value=b"synthetic source") as git_show:
            audit = analysis.Audit()
            lineage = analysis.verify_sources(manifests, audit)
            self.assertEqual(audit.errors, [])
            self.assertTrue(lineage["shared_original_24_source_hashes_unchanged"])
            self.assertEqual(lineage["smoke"]["frozen_commit"], analysis.SMOKE_COMMIT)
            self.assertEqual(lineage["frozen_commit"], "a" * 40)
            self.assertEqual(len(git_show.call_args_list), 51)
            self.assertEqual(sum(call.args[0][-1].startswith(analysis.SMOKE_COMMIT + ":")
                                 for call in git_show.call_args_list), 24)
            # Equal prefix values alone cannot excuse reordered/extra sources.
            mutations = []
            wrong_commit = copy.deepcopy(manifests)
            wrong_commit[0]["frozen_commit"] = "b" * 40
            mutations.append((wrong_commit, "lineage commit"))
            missing = copy.deepcopy(manifests)
            missing[0]["source_hashes"].pop(next(iter(sources)))
            mutations.append((missing, "exact 24-file"))
            changed = copy.deepcopy(manifests)
            changed[1]["source_hashes"][next(iter(sources))] = "e" * 64
            mutations.append((changed, "identical ordered 24-file"))
            reordered = copy.deepcopy(manifests)
            pairs = list(reordered[1]["source_hashes"].items())
            pairs[-1], pairs[-2] = pairs[-2], pairs[-1]
            reordered[1]["source_hashes"] = dict(pairs)
            mutations.append((reordered, "exact 27-file"))
            extra = copy.deepcopy(manifests)
            extra[1]["source_hashes"]["unregistered.py"] = digest
            mutations.append((extra, "exact 27-file"))
            for changed, expected in mutations:
                audit = analysis.Audit()
                analysis.verify_sources(changed, audit)
                self.assertTrue(any(expected in error for error in audit.errors), audit.errors)
            git_show.return_value = b"wrong committed bytes"
            audit = analysis.Audit()
            analysis.verify_sources(manifests, audit)
            self.assertTrue(any("committed source differs" in error for error in audit.errors))

    def test_consumed_smoke_completion_pin_is_mandatory(self):
        self.assertEqual(analysis.SMOKE_COMMIT, "14678e2a7a6fdc3aacc25a320516f1c07a037002")
        self.assertEqual(analysis.SMOKE_COMPLETION_SHA,
                         "4985e7e1b5280f7e01f8ef4412518639e9aea7618ad9b068129a76dc1e55849d")
        self.assertEqual(str(analysis.ARTIFACT_ROOT), "/tmp/spectral-experiment-artifacts/spectral-i17-001.k4VKcx")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            synthetic_smoke(root)
            audit = analysis.Audit()
            analysis.phase_records(root, "smoke", audit)
            self.assertTrue(any("consumed smoke completion/commit pin" in error for error in audit.errors))

    def test_accepted_k0_raw_json_checkpoint_and_transitive_bindings(self):
        # This fixture substitutes only the already-tested historical semantic
        # validators. The I17 admission, SHA hashes, JSON/summary associations,
        # exact six-reference roster and no-terminal-load rule run normally.
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
            outer = Path(temporary)
            evidence, root = outer / "iteration-016", outer / "bulk"
            (evidence / "analysis-001").mkdir(parents=True)
            (evidence / "raw-results-001").mkdir()
            directory = root / "confirmation"
            directory.mkdir(parents=True)
            artifacts, entries, references, trajectories, parent_rows, inherited_refs = [], [], [], [], [], {}
            for seed in analysis.SEEDS:
                for target in analysis.TARGETS:
                    curve = [{"horizon": horizon, **evaluation(-index * .01)}
                             for index, horizon in enumerate(analysis.HORIZONS)]
                    parent_rows.append({"seed": seed, "target": target, "state_digest": "a" * 64})
                    inherited_refs[seed, target, "k1"] = {"status": "complete", "curve": curve}
                    checkpoints = []
                    for horizon in analysis.HORIZONS[1:]:
                        kind = "full_state" if horizon == 2000 else "model_state"
                        path = directory / f"state-s{seed}-{target}-h{horizon}.pt"
                        path.write_bytes(f"synthetic hash-only {seed}/{target}/{horizon}".encode())
                        artifact = record(path)
                        artifacts.append(artifact)
                        checkpoints.append({"horizon": horizon, "relative_horizon": horizon - 100, kind: artifact,
                                            **({"full_state_digest": "b" * 64} if horizon == 2000 else {})})
                    branch = {"schema": "i16_scalar_branch_v1", "id": f"s{seed}-sgdm-{target}-k0",
                        "seed": seed, "target": target, "k": 0.0, "k_label": "k0", "base": "sgdm", "lr": .03,
                        "status": "complete", "completed_updates": 1900, "parent_state_digest": "a" * 64,
                        "parent_evaluation_digest": analysis.tree_digest(curve[0]), "first_step_digests": None,
                        "scalar_update_seconds": .1, "curve": curve, "steps": [], "checkpoints": checkpoints, "failure": None}
                    artifact = save_json(directory / f"branch-s{seed}-{target}-k0.json", branch)
                    artifacts.append(artifact)
                    fields = ("id", "seed", "target", "k", "k_label", "status", "completed_updates", "parent_state_digest",
                              "parent_evaluation_digest", "first_step_digests", "scalar_update_seconds")
                    entries.append({name: branch[name] for name in fields} | {"artifact": artifact})
                    for k in ("k0p5", "k0p9"):
                        entries.append({"seed": seed, "target": target, "k_label": k})
                    references.append({"seed": seed, "target": target, "curve_artifact": artifact,
                        "checkpoint_records": checkpoints, "parent_state_digest": "a" * 64,
                        "parent_evaluation_digest": branch["parent_evaluation_digest"]})
                    trajectories.append({"seed": seed, "target": target, "policy": "k0", "status": "complete", "failure": None,
                        "curve": [{"horizon": point["horizon"], "values": {name: point[split][field]
                                  for name, (split, field) in analysis.METRIC_PATHS.items()}} for point in curve]})
            trajectories.extend({"policy": "unused", "ordinal": index} for index in range(24))
            inherited = {"schema": "synthetic_transitive_binding", "i14": {"parents": parent_rows}}
            artifacts.append(save_json(directory / "parent-inputs.json", inherited))
            artifacts.append(save_json(directory / "branches.json", {"entries": entries}))
            completion = {"schema": "i16_completion_v1", "status": "complete", "phase": "confirmation",
                "frozen_commit": analysis.I16_COMMIT, "branches": 18, "numerical_failures": 0,
                "completed_training_updates": 34200, "all_requested_endpoints_present": True, "artifacts": artifacts}
            completion_sha = save_json(directory / "completion.json", completion)["sha256"]
            audit_sha = save_json(evidence / "analysis-001/audit.json", {
                "schema": "i16_scalar_analysis_audit_v1", "status": "pass", "errors": [], "artifact_root": str(root),
                "phase_completion_sha256": {"confirmation": completion_sha}})["sha256"]
            summary_sha = save_json(evidence / "analysis-001/summary.json", {
                "schema": "i16_scalar_analysis_summary_v1", "audit_status": "pass", "artifact_root": str(root),
                "all_policy_trajectories": trajectories})["sha256"]
            report_sha = save_json(evidence / "analysis-001/report-audit.json", {
                "status": "pass", "errors": [], "maximum_absolute_numeric_difference": 0,
                "inputs": {"i16_summary": {"sha256": summary_sha}}})["sha256"]
            collection_sha = save_json(evidence / "raw-results-001/collection.json", {
                "status": "complete", "source_root": str(root), "analysis_audit": {"sha256": audit_sha},
                "phase_completion_sha256": {"confirmation": completion_sha}})["sha256"]
            pins = {"analysis-001/audit.json": audit_sha, "analysis-001/summary.json": summary_sha,
                    "analysis-001/report-audit.json": report_sha, "raw-results-001/collection.json": collection_sha}
            binding = {"schema": "i17_parent_and_reference_binding_v1", "inherited_i16": inherited,
                "i16": {"artifact_root": str(root), "acquisition_commit": analysis.I16_COMMIT,
                        "pinned_sha256": pins, "confirmation_sha256": completion_sha},
                "k0_references": references, "source_replayed": False}
            for name, value in (("I16", evidence), ("I16_ROOT", root), ("PINS", pins), ("I16_COMPLETION_SHA", completion_sha)):
                stack.enter_context(mock.patch.object(analysis, name, value))
            old_loader = stack.enter_context(mock.patch.object(analysis.old, "load_references", return_value=inherited_refs))
            stack.enter_context(mock.patch.object(analysis.old, "validate_branch", side_effect=lambda value, *args: dict(value)))
            stack.enter_context(mock.patch.object(torch, "load", side_effect=AssertionError("unused old terminal loaded")))
            audit = analysis.Audit()
            result = analysis.load_references(binding, audit)
            self.assertEqual(audit.errors, [])
            self.assertEqual(set(result), {(seed, target, "scalar_k0") for seed in analysis.SEEDS for target in analysis.TARGETS})
            self.assertEqual(old_loader.call_count, 1)
            self.assertEqual(audit.tree_digests, 0)
            altered = copy.deepcopy(binding)
            altered["k0_references"][0]["checkpoint_records"][0]["horizon"] = 500
            audit = analysis.Audit()
            analysis.load_references(altered, audit)
            self.assertTrue(any("direct branch binding" in error for error in audit.errors))
            altered = copy.deepcopy(binding)
            altered["inherited_i16"]["schema"] = "unaccepted"
            old_loader.reset_mock()
            audit = analysis.Audit()
            self.assertEqual(analysis.load_references(altered, audit), {})
            self.assertTrue(any("transitive" in error for error in audit.errors))
            old_loader.assert_not_called()

    def test_resource_limits_and_h100_digest_mismatch(self):
        audit = analysis.Audit()
        completion = {"update_seconds": 1.0, "elapsed_seconds": 100.0,
                      "peak_torch_gpu_bytes": 4 * 1024**3, "shared_artifact_bytes_before_completion": analysis.ARTIFACT_CAP}
        analysis.validate_phase_resources(completion, "smoke", audit)
        self.assertEqual(audit.errors, [])
        completion["elapsed_seconds"] += .01
        completion["peak_torch_gpu_bytes"] += 1
        analysis.validate_phase_resources(completion, "smoke", audit)
        self.assertTrue(any("wall limit" in error for error in audit.errors))
        self.assertTrue(any("resource" in error for error in audit.errors))
        completion["elapsed_seconds"] = 2400.0
        completion["peak_torch_gpu_bytes"] = 4 * 1024**3
        audit = analysis.Audit()
        analysis.validate_phase_resources(completion, "confirmation", audit)
        self.assertEqual(audit.errors, [])
        completion["elapsed_seconds"] += .01
        analysis.validate_phase_resources(completion, "confirmation", audit)
        self.assertTrue(any("wall limit" in error for error in audit.errors))
        branch = branch_record()
        branch["curve"][0]["auxiliary"]["clean_ce"] += .01
        audit = analysis.Audit()
        analysis.validate_branch(branch, "smoke", audit, "h100")
        self.assertTrue(any("h100 evaluation digest" in error for error in audit.errors))

    def test_import_is_artifact_free_and_does_not_import_acquisition(self):
        code = """import importlib.util, pathlib, sys, torch
from unittest import mock
path=pathlib.Path(sys.argv[1])
spec=importlib.util.spec_from_file_location('_independent_i17_probe', path)
value=importlib.util.module_from_spec(spec)
with mock.patch.object(pathlib.Path, 'open', side_effect=AssertionError('artifact read')), mock.patch.object(torch, 'load', side_effect=AssertionError('state load')):
    spec.loader.exec_module(value)
assert not any('run_gain_controls' in name or 'gain_core' in name for name in sys.modules)
print('import-safe')
"""
        result = subprocess.run([sys.executable, "-c", code, str(HERE / "analyse_gain.py")],
                                check=True, capture_output=True, text=True)
        self.assertEqual(result.stdout.strip(), "import-safe")


if __name__ == "__main__":
    unittest.main()
