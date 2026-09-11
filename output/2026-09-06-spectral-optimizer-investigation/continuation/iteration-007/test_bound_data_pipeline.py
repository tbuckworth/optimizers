"""Synthetic IDX bytes and verified plan through live-state measurements/audit."""
import copy
import hashlib
import os
import struct
import tempfile
import unittest

if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
    raise RuntimeError("explicitly hide CUDA for bound-data fixtures")

import torch
import torch.nn.functional as F

import artifact_store as storage
import data_probe_bindings as data_binding
import identity_codec as codec
import independent_numerics as independent
import measurement_assembly as assembly
import measurement_audit as audit
import response_math as response
import verified_plan_load as verified
from test_plan_bindings import fixture as plan_fixture
import test_full_measurement_pipeline as live


def synthetic_idx():
    images = struct.pack(">IIII", 2051, 30, 1, 3) + bytes((17 * i + 31) % 256 for i in range(90))
    labels = struct.pack(">II", 2049, 30) + bytes(i % 2 for i in range(30))
    refs = {key: {"sha256": hashlib.sha256(value).hexdigest(), "size_bytes": len(value)}
            for key, value in (("training_images", images), ("training_labels", labels))}
    return images, labels, refs


class BoundDataPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_verified_plan_idx_and_actual_source_to_independent_audit(self):
        identity, plan, _ = plan_fixture()
        images, labels, refs = synthetic_idx()
        with tempfile.TemporaryDirectory(prefix="i7-bound-pipeline-") as directory:
            with storage.ArtifactStore(directory, profile=storage.MLP_FIXTURE,
                                       min_filesystem_free_bytes=0) as plan_store:
                receipt = plan_store.write_tensor_tree("frozen-plan.pt", plan)
            loaded = verified.load_verified_plan(plan_store.root, "frozen-plan.pt",
                identity=identity, profile=storage.MLP_FIXTURE, expected_sha256=receipt["sha256"])
            plan = loaded["plan"]
            materialized = data_binding.materialize(images, labels, plan=plan, identity=identity,
                profile=storage.MLP_FIXTURE, expected_files=refs)
            data_binding.validate_materialization(materialized, images, labels, plan=plan,
                identity=identity, profile=storage.MLP_FIXTURE, expected_files=refs)
            unchanged_data, unchanged_plan = codec.tree_digest(materialized), codec.tree_digest(plan)
            payload = live.fixture(plan=plan, materialized=materialized)
            # Independently rebuild all four pre-anchor rows, including the bound init seed.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(plan["initialization_seed"])
                model = live.model_factory()
                optimizer = live.optimizer_factory(model)
                observer = live.observer_factory(model, optimizer)
                for local in plan["training_batches"][:4]:
                    optimizer.zero_grad(set_to_none=True)
                    F.cross_entropy(model(materialized["datasets"]["train_inputs"][local]),
                                    materialized["datasets"]["train_noisy_labels"][local]).backward()
                    observer.filter_grad()
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                anchor = live.state.capture_core(model, optimizer, observer,
                    profile=storage.MLP_FIXTURE, completed_updates=4)
                for key in ("model", "optimizer", "observer", "rng"):
                    live.check_equal(anchor[key], payload["anchor"][key], "bound warm anchor " + key)
            payload["data_bindings"] = copy.deepcopy(materialized["bindings"])
            payload["plan_binding"] = copy.deepcopy(loaded["binding"])
            with storage.ArtifactStore(directory, profile=storage.MLP_FIXTURE,
                                       min_filesystem_free_bytes=0) as result_store:
                raw_receipt = result_store.write_tensor_tree("bound-raw.pt", payload)
                raw = storage.ArtifactStore.load_tensor_tree(result_store.root, raw_receipt["name"],
                    expected_size=raw_receipt["size"], expected_sha256=raw_receipt["sha256"])
                self.assertEqual(codec.tree_digest(raw["probes"]), codec.tree_digest(materialized["probes"]))
                before_rng = live.state._capture_rng()
                measurement = assembly.assemble(raw["before"], raw["endpoints"], raw["candidates"],
                    raw["probes"], artifact_id="bound-data-fixture", profile=storage.MLP_FIXTURE)
                measured = result_store.write_tensor_tree("bound-measurements.pt", measurement)
                recovered = storage.ArtifactStore.load_tensor_tree(result_store.root, measured["name"],
                    expected_size=measured["size"], expected_sha256=measured["sha256"])
                candidates = {branch: row["gradient"] for branch, row in raw["candidates"]["branches"].items()}
                checked = audit.audit(live.numpy_tree(raw["before"]), live.numpy_tree(raw["endpoints"]),
                    live.numpy_tree(candidates), live.numpy_tree(raw["probes"]), live.numpy_tree(recovered),
                    profile=storage.MLP_FIXTURE)
                self.assertEqual(checked["overall_status"], "pass", checked["fatal_failures"][:5])
                self.assertTrue(checked["audit_complete"])
                self.assertEqual(checked["completion_counts"], {
                    "before_probes": 4, "branch_probes": 24, "defined_branches": 6,
                    "pairs": 15, "contrasts": 15, "defined_contrasts": 15,
                    "auxiliary_before_chunks": 10, "auxiliary_branch_chunks": 60,
                    "auxiliary_contrast_chunks": 150})
                required = {f"contrasts.{contrast}.{probe}.{kind}.independent"
                            for contrast in response.COEFFICIENTS for probe in assembly.PROBES
                            for kind in ("Y", "D", "Ddata", "R")}
                required.update(f"contrasts.{contrast}.auxiliary_clean.chunk{chunk}.Y.independent"
                                for contrast in response.COEFFICIENTS for chunk in range(10))
                found = {key for key in checked["comparison_audits"] if key.endswith(".independent")}
                self.assertEqual(len(required), 390)
                self.assertEqual(found, required)
                self.assertTrue(all(checked["comparison_audits"][key]["audit_status"] == "pass"
                                    for key in required))
                # Actual native Adam endpoints and moment states are checked from raw values.
                adam_checked = 0
                for branch, endpoints in raw["endpoints"].items():
                    offset = 0
                    for index, (name, endpoint) in enumerate(endpoints.items()):
                        old = raw["anchor"]["optimizer"]["state"][index]
                        new = raw["optimizer_endpoints"][branch]["state"][index]
                        gradient = candidates[branch][offset:offset + endpoint.numel()].reshape(endpoint.shape)
                        group = raw["optimizer_endpoints"][branch]["param_groups"][0]
                        check = independent.audit_adamw(raw["before"][name].numpy(), gradient.numpy(),
                            old["exp_avg"]["value"].numpy(), old["exp_avg_sq"]["value"].numpy(), 5,
                            {"theta": endpoint.numpy(), "moment": new["exp_avg"].numpy(),
                             "variance": new["exp_avg_sq"].numpy(), "next_step": int(new["step"])},
                            options={key: group[key] for key in independent.ADAMW_OPTIONS})
                        self.assertTrue(check["passed"], check)
                        adam_checked += 1
                        offset += endpoint.numel()
                self.assertEqual(adam_checked, 24)
                result_store.write_tensor_tree("bound-audit.pt", checked)
                self.assertEqual(codec.tree_digest(before_rng), codec.tree_digest(live.state._capture_rng()))
            self.assertEqual(codec.tree_digest(materialized), unchanged_data)
            self.assertEqual(codec.tree_digest(plan), unchanged_plan)
            self.assertFalse(torch.cuda.is_initialized())


if __name__ == "__main__":
    unittest.main()
