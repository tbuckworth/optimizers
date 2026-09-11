"""Parent integration: real live tiny state -> measurements -> sealed bytes -> audit.

Dataset-free, CPU-only engineering fixture; not a scientific artifact envelope.
"""
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest

if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
    raise RuntimeError("Explicitly hide CUDA for CPU fixtures")

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


state = load("i7_pipeline_state", HERE / "state_core.py")
response = load("i7_pipeline_response", HERE / "response_math.py")
storage = load("i7_pipeline_store", HERE / "artifact_store.py")
spectral = load("i7_pipeline_filter", HERE.parents[3] / "spectral_filter.py")


def model_factory():
    return torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.ReLU(), torch.nn.Linear(4, 2))


def optimizer_factory(model):
    return torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01,
                             betas=(.9, .999), eps=1e-8, foreach=False, fused=False)


def observer_factory(model, optimizer):
    return spectral.SpectralGradientFilter(model, optimizer, rank=2, decay=.99,
                                           warmup=2, stable_update=True)


def parameter_values(model):
    return {name: value.detach().clone() for name, value in model.named_parameters()}


def flat(values):
    return torch.cat([value.detach().reshape(-1) for value in values]).clone()


def numpy_tree(value):
    if type(value) is torch.Tensor:
        return value.detach().numpy().copy()
    if type(value) is dict:
        return {key: numpy_tree(item) for key, item in value.items()}
    if type(value) in (list, tuple):
        return type(value)(numpy_tree(item) for item in value)
    return value


def check_equal(left, right, label):
    if not state._tree_equal(left, right):
        raise AssertionError(label)


def fixture(*, domain_null=False, plan=None, materialized=None):
    """Generate a new tiny live source; never access prior scientific artifacts."""
    if (plan is None) != (materialized is None):
        raise ValueError("plan and materialized data must be supplied together")
    torch.manual_seed(7317 if plan is None else plan["initialization_seed"])
    if materialized is None:
        x = torch.tensor([[.1, .4, .9], [.7, .2, .5], [.9, .8, .1], [.1, .4, .9]])
        y = torch.tensor([0, 1, 1, 0])
    else:
        x = materialized["probes"]["batch_noisy"]["inputs"].clone()
        y = materialized["probes"]["batch_noisy"]["labels"].clone()
    model = model_factory()
    optimizer = optimizer_factory(model)
    observer = observer_factory(model, optimizer)
    for row in range(4):
        optimizer.zero_grad(set_to_none=True)
        if plan is None:
            warm_x, warm_y = x, y
        else:
            local_indices = plan["training_batches"][row]
            warm_x = materialized["datasets"]["train_inputs"][local_indices]
            warm_y = materialized["datasets"]["train_noisy_labels"][local_indices]
        F.cross_entropy(model(warm_x), warm_y).backward()
        observer.filter_grad()
        optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    anchor = state.capture_core(model, optimizer, observer,
        profile=state.MLP_FIXTURE_PROFILE, completed_updates=4)
    anchor_copy = state.clone_tree(anchor)
    before = parameter_values(model)
    F.cross_entropy(model(x), y).backward()
    live_raw = flat([parameter.grad for parameter in model.parameters()])
    observer.filter_grad()
    live_current = flat([parameter.grad for parameter in model.parameters()])
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    witness = state.capture_core(model, optimizer, observer,
        profile=state.MLP_FIXTURE_PROFILE, completed_updates=5)
    check_equal(anchor["rng"], witness["rng"], "live source consumed RNG")

    restored = state.restore_core(anchor, model_factory, optimizer_factory, observer_factory)
    rm, ro, rf = restored["model"], restored["optimizer"], restored["observer"]
    previous_basis = rf.V.clone()
    F.cross_entropy(rm(x), y).backward()
    raw = flat([parameter.grad for parameter in rm.parameters()])
    lagged = previous_basis @ (previous_basis.T @ raw)
    rf.filter_grad()
    current = flat([parameter.grad for parameter in rm.parameters()])
    check_equal(raw, live_raw, "raw differs from actual live source")
    check_equal(current, live_current, "current differs from actual live source")
    check_equal(state._capture_observer(rf, rm, ro, state.MLP_FIXTURE_PROFILE, 5),
                witness["observer"], "observer differs from actual live source")
    rng_snapshots = {"after_candidate": state._capture_rng()}
    check_equal(rng_snapshots["after_candidate"], anchor["rng"], "candidate consumed RNG")
    # Explicit synthetic domain intervention, not a claimed native lagged operator.
    if domain_null:
        lagged = torch.zeros_like(lagged)
    candidates = response.construct(raw, current, lagged)
    endpoints, optimizer_endpoints, branch_observers = {}, {}, {}
    for name in response.BRANCHES:
        clone = state.restore_core(anchor, model_factory, optimizer_factory, observer_factory)
        bm, bo = clone["model"], clone["optimizer"]
        offset = 0
        vector = candidates["branches"][name]["gradient"]
        if vector is None:
            endpoints[name] = optimizer_endpoints[name] = None
            branch_observers[name] = None
            continue
        for parameter in bm.parameters():
            parameter.grad = vector[offset:offset + parameter.numel()].reshape(parameter.shape).clone()
            offset += parameter.numel()
        bo.step()
        bo.zero_grad(set_to_none=True)
        rng_snapshots[name] = state._capture_rng()
        check_equal(rng_snapshots[name], anchor["rng"], "branch consumed RNG")
        branch_observers[name] = state._capture_observer(clone["observer"], bm, bo,
            state.MLP_FIXTURE_PROFILE, 4)
        check_equal(branch_observers[name], anchor["observer"], "branch reobserved")
        endpoints[name] = parameter_values(bm)
        optimizer_endpoints[name] = state.clone_tree(bo.state_dict())
    check_equal(endpoints["current"], parameter_values(model), "current endpoint differs")
    check_equal(optimizer_endpoints["current"], optimizer.state_dict(), "current optimizer differs")
    check_equal(anchor, anchor_copy, "anchor mutated")
    indices = torch.tensor([0, 1, 2, 3, 3, 2, 1, 0, 0, 2])
    probes = state.clone_tree(materialized["probes"]) if materialized is not None else {
        "batch_noisy": {"inputs": x.clone(), "labels": y.clone()},
        "train_probe_noisy": {"inputs": x.clone(), "labels": (1-y).clone()},
        "train_probe_clean": {"inputs": x.clone(), "labels": y.clone()},
        "auxiliary_clean": {"inputs": x[indices].clone(), "labels": y[indices].clone()},
    }
    return {"anchor": anchor, "witness": witness, "before": before,
            "endpoints": endpoints, "optimizer_endpoints": optimizer_endpoints,
            "candidates": candidates, "probes": probes,
            "rng_snapshots": rng_snapshots, "branch_observers": branch_observers,
            "candidate_fixture_mode": "forced_zero_lagged_direction" if domain_null else "native_tiny_observer"}


class FullMeasurementPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def audit_loaded(self, payload, measurements, *, guard=None):
        audit_module = load("i7_pipeline_audit", HERE / "measurement_audit.py")
        candidates = {branch: row["gradient"] for branch, row in payload["candidates"]["branches"].items()}
        return audit_module.audit(numpy_tree(payload["before"]), numpy_tree(payload["endpoints"]),
            numpy_tree(candidates), numpy_tree(payload["probes"]), numpy_tree(measurements),
            profile=state.MLP_FIXTURE_PROFILE, guard=guard)

    def test_live_source_full_measurements_and_storage_roundtrip(self):
        assembly = load("i7_pipeline_assembly", HERE / "measurement_assembly.py")
        runtime = load("i7_pipeline_runtime", HERE / "runtime_guard.py")
        payload = fixture()
        original = state.clone_tree(payload)
        rng_before = state._capture_rng()
        runtime_guard = runtime.RuntimeGuard("audit", profile=runtime.FIXTURE)
        labels = []
        def guard(label):
            labels.append(label)
            return runtime_guard(label)
        with tempfile.TemporaryDirectory() as parent:
            with storage.ArtifactStore(parent, profile=storage.MLP_FIXTURE, budget_bytes=8 << 20,
                failure_reserve_bytes=1 << 20, min_filesystem_free_bytes=0) as store:
                guard("before.raw_write")
                receipt = store.write_tensor_tree("live-source-fixture.pt", payload)
                guard("after.raw_write")
                recovered = storage.ArtifactStore.load_tensor_tree(store.root, receipt["name"],
                    expected_size=receipt["size"], expected_sha256=receipt["sha256"])
                guard("after.raw_load")
                self.assertTrue(state._tree_equal(original, recovered))
                independent = load("i7_pipeline_independent", HERE / "independent_numerics.py")
                for branch in response.BRANCHES:
                    offset = 0
                    for index, (name, parameter) in enumerate(recovered["endpoints"][branch].items()):
                        old = recovered["anchor"]["optimizer"]["state"][index]
                        new = recovered["optimizer_endpoints"][branch]["state"][index]
                        gradient = recovered["candidates"]["branches"][branch]["gradient"]
                        delivered = gradient[offset:offset+parameter.numel()].reshape(parameter.shape)
                        self.assertEqual(int(new["step"]), 5)
                        options = recovered["optimizer_endpoints"][branch]["param_groups"][0]
                        checked = independent.audit_adamw(recovered["before"][name].numpy(),
                            delivered.numpy(), old["exp_avg"]["value"].numpy(),
                            old["exp_avg_sq"]["value"].numpy(), 5,
                            {"theta": parameter.numpy(), "moment": new["exp_avg"].numpy(),
                             "variance": new["exp_avg_sq"].numpy(), "next_step": int(new["step"])},
                            options={key: options[key] for key in independent.ADAMW_OPTIONS})
                        self.assertTrue(checked["passed"], checked)
                        offset += parameter.numel()
                self.assertFalse(state._tree_equal(recovered["endpoints"]["zero"], recovered["before"]))
                clean, noisy = recovered["probes"]["train_probe_clean"], recovered["probes"]["train_probe_noisy"]
                self.assertTrue(state._tree_equal(clean["inputs"], noisy["inputs"]))
                self.assertFalse(state._tree_equal(clean["labels"], noisy["labels"]))
                measurements = assembly.assemble(recovered["before"], recovered["endpoints"],
                    recovered["candidates"], recovered["probes"], artifact_id="fixture-measurements",
                    profile=assembly.FIXTURE_PROFILE, guard=guard)
                self.assertTrue(state._tree_equal(original, recovered))
                self.assertTrue(state._tree_equal(rng_before, state._capture_rng()))
                self.assertEqual(list(measurements["branches"]), list(response.BRANCHES))
                self.assertEqual(len(measurements["comparisons"]["branch_pairs"]), 15)
                self.assertEqual(list(measurements["comparisons"]["contrasts"]), list(response.COEFFICIENTS))
                for row in measurements["comparisons"]["contrasts"].values():
                    self.assertTrue(row["defined"])
                    for probe in row["probes"].values():
                        for key in ("Y", "D", "Ddata", "R"):
                            self.assertEqual(probe["cpu64"][key]["identity_status"], "pass")
                guard("before.measurement_write")
                measurement_receipt = store.write_tensor_tree("measurements.pt", measurements)
                guard("after.measurement_write")
                loaded_measurements = storage.ArtifactStore.load_tensor_tree(store.root,
                    measurement_receipt["name"], expected_size=measurement_receipt["size"],
                    expected_sha256=measurement_receipt["sha256"])
                guard("after.measurement_load")
                self.assertTrue(state._tree_equal(measurements, loaded_measurements))
                report = self.audit_loaded(recovered, loaded_measurements, guard=guard)
                self.assertEqual(report["overall_status"], "pass", report["fatal_failures"][:10])
                self.assertTrue(report["audit_complete"])
                self.assertEqual(report["scope"], "raw_value_measurements_only")
                self.assertEqual(report["domain_undefined_branches"], [])
                self.assertEqual(report["completion_counts"], {
                    "before_probes": 4, "branch_probes": 24, "defined_branches": 6,
                    "pairs": 15, "contrasts": 15, "defined_contrasts": 15,
                    "auxiliary_before_chunks": 10, "auxiliary_branch_chunks": 60,
                    "auxiliary_contrast_chunks": 150})
                for probe in assembly.PROBES:
                    self.assertEqual(report["measurement_audits"][f"before.{probe}.q_values"]["audit_status"], "pass")
                    for branch in response.BRANCHES:
                        self.assertEqual(report["measurement_audits"][f"branches.{branch}.{probe}.after_ce"]["audit_status"], "pass")
                required = {f"contrasts.{contrast}.{probe}.{kind}.independent"
                    for contrast in response.COEFFICIENTS for probe in assembly.PROBES
                    for kind in ("Y", "D", "Ddata", "R")}
                required.update(f"contrasts.{contrast}.auxiliary_clean.chunk{chunk}.Y.independent"
                    for contrast in response.COEFFICIENTS for chunk in range(10))
                found = {key for key in report["comparison_audits"] if key.endswith(".independent")}
                self.assertEqual(found, required)
                self.assertEqual(len(required), 390)
                self.assertTrue(all(report["comparison_audits"][key]["audit_status"] == "pass" for key in required))
                guard("before.audit_write")
                audit_receipt = store.write_tensor_tree("fixture-measurement-audit.pt", report)
                guard("after.audit_write")
                recovered_report = storage.ArtifactStore.load_tensor_tree(store.root, audit_receipt["name"],
                    expected_size=audit_receipt["size"], expected_sha256=audit_receipt["sha256"])
                guard("after.audit_load")
                self.assertEqual(report, recovered_report)
                inventory = storage.ArtifactStore.inspect(store.root)
                self.assertFalse(inventory["terminal"])
                self.assertEqual(len(inventory["receipts"]), 3)
                self.assertEqual(inventory["header"]["profile"], state.MLP_FIXTURE_PROFILE)
                self.assertEqual(recovered["anchor"]["profile"], state.MLP_FIXTURE_PROFILE)
                self.assertEqual(assembly.FIXTURE_PROFILE, state.MLP_FIXTURE_PROFILE)
                self.assertEqual(runtime_guard.summary()["profile"], state.MLP_FIXTURE_PROFILE)
                self.assertEqual(measurements["measurement_before"]["probes"]["auxiliary_clean"]["cpu64"]["q"]["shape"], [26])
                self.assertIn("before:assembly:validation", labels)
                self.assertIn("after:assembly:validation", labels)
                self.assertIn("after:contrast:interaction", labels)
                self.assertEqual(sum(label.startswith("loss.before_chunk.") for label in labels), 182)
                self.assertEqual(sum(label.startswith("loss.after_chunk.") for label in labels), 182)
                self.assertFalse(runtime_guard.summary()["terminal"])
                self.assertTrue(state._tree_equal(rng_before, state._capture_rng()))
        self.assertFalse(torch.cuda.is_initialized())

    def test_serialized_domain_null_and_corruption_are_distinct(self):
        assembly = load("i7_pipeline_domain_assembly", HERE / "measurement_assembly.py")
        payload = fixture(domain_null=True)
        self.assertEqual(payload["candidate_fixture_mode"], "forced_zero_lagged_direction")
        measurements = assembly.assemble(payload["before"], payload["endpoints"], payload["candidates"],
            payload["probes"], artifact_id="fixture-domain-measurements", profile=assembly.FIXTURE_PROFILE)
        payload["measurements"] = measurements
        with tempfile.TemporaryDirectory() as parent:
            with storage.ArtifactStore(parent, profile=storage.MLP_FIXTURE, budget_bytes=8 << 20,
                failure_reserve_bytes=1 << 20, min_filesystem_free_bytes=0) as store:
                receipt = store.write_tensor_tree("domain-fixture.pt", payload)
                recovered = storage.ArtifactStore.load_tensor_tree(store.root, receipt["name"],
                    expected_size=receipt["size"], expected_sha256=receipt["sha256"])
                self.assertIsNone(recovered["candidates"]["branches"]["restored"]["gradient"])
                self.assertIsNone(recovered["endpoints"]["restored"])
                self.assertIsNone(recovered["optimizer_endpoints"]["restored"])
                self.assertIsNone(recovered["measurements"]["branches"]["restored"])
                report = self.audit_loaded(recovered, recovered["measurements"])
                self.assertEqual(report["overall_status"], "pass", report["fatal_failures"][:10])
                self.assertTrue(report["audit_complete"])
                self.assertEqual(report["domain_undefined_branches"], ["restored"])
                self.assertEqual(report["completion_counts"], {
                    "before_probes": 4, "branch_probes": 20, "defined_branches": 5,
                    "pairs": 15, "contrasts": 15, "defined_contrasts": 10,
                    "auxiliary_before_chunks": 10, "auxiliary_branch_chunks": 50,
                    "auxiliary_contrast_chunks": 100})
                contrast = recovered["measurements"]["comparisons"]["contrasts"]["interaction"]
                self.assertFalse(contrast["defined"])
                self.assertEqual(contrast["reason"], "domain_undefined_required_branch")
                unaffected = "contrasts.current_minus_raw.auxiliary_clean.Y.independent"
                self.assertEqual(report["comparison_audits"][unaffected]["audit_status"], "pass")
                bad = state.clone_tree(recovered["measurements"])
                bad["branches"]["raw"]["probes"]["auxiliary_clean"]["cpu64"]["after_ce"] += .01
                bad_receipt = store.write_tensor_tree("deliberately-corrupted-measurements.pt", bad)
                loaded_bad = storage.ArtifactStore.load_tensor_tree(store.root, bad_receipt["name"],
                    expected_size=bad_receipt["size"], expected_sha256=bad_receipt["sha256"])
                bad_report = self.audit_loaded(recovered, loaded_bad)
                self.assertEqual(bad_report["overall_status"], "fatal_validation")
                self.assertTrue(bad_report["fatal_failures"])
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == "__main__":
    unittest.main()
