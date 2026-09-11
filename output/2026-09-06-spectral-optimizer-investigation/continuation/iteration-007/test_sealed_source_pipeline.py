"""Actual sealed source evidence through six endpoints and independent CPU audit."""
import os
import tempfile
import unittest

if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
    raise RuntimeError("explicitly hide CUDA for sealed-source pipeline fixtures")

import torch
import torch.nn.functional as F

import anchor_envelope as envelope
import artifact_store as storage
import data_probe_bindings as data
import independent_numerics as independent
import measurement_assembly as assembly
import measurement_audit as audit
import response_math as response
import source_capture as source
import state_core as state
from envelope_fixture import fixture_context, warm_live
from test_full_measurement_pipeline import (model_factory, optimizer_factory,
    observer_factory, parameter_values, flat, numpy_tree)


class SealedSourcePipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_actual_sealed_source_to_measurements_and_independent_audit(self):
        with fixture_context() as context, tempfile.TemporaryDirectory(prefix="i7-sealed-pipeline-") as root:
            model, optimizer, observer = warm_live(context)
            with storage.ArtifactStore(root, profile=storage.MLP_FIXTURE,
                                       min_filesystem_free_bytes=0) as store:
                transaction = source.capture_anchor_then_live_witness(model, optimizer, observer,
                    store=store, created_utc="2026-09-06T20:00:00Z", **context)

                def reload(receipt):
                    return storage.ArtifactStore.load_tensor_tree(store.root, receipt["name"],
                        expected_size=receipt["size"], expected_sha256=receipt["sha256"])

                anchor = reload(transaction["anchor_receipt"])
                witness = reload(transaction["witness_receipt"])
                envelope.validate_anchor(anchor, **context)
                source.validate_source_witness(witness, anchor=anchor,
                    anchor_receipt=transaction["anchor_receipt"], **context)
                # Bind the loaded witness to the ORIGINAL live objects before
                # constructing any replay clone; replay agreement alone cannot
                # establish that the original trajectory actually reached it.
                live_endpoint = state.capture_core(model, optimizer, observer,
                    profile=context["profile"], completed_updates=5)
                self.assertTrue(envelope.same_exact(live_endpoint["model"]["parameters"],
                                                   witness["payload"]["parameters_after"]))
                self.assertTrue(envelope.same_exact(live_endpoint["optimizer"],
                                                   witness["payload"]["optimizer_after"]))
                self.assertTrue(envelope.same_exact(live_endpoint["observer"],
                                                   witness["payload"]["observer_after"]))
                self.assertTrue(envelope.same_exact(live_endpoint["rng"], anchor["payload"]["rng"]))
                original = state.clone_tree(anchor)
                core = envelope.core_from_anchor(anchor)
                materialized = data.materialize(context["images_bytes"], context["labels_bytes"],
                    plan=context["plan"], identity=context["identity"], profile=context["profile"],
                    expected_files=context["expected_files"])
                batch = materialized["probes"]["batch_noisy"]

                replay = state.restore_core(core, model_factory, optimizer_factory, observer_factory)
                rm, ro, rf = replay["model"], replay["optimizer"], replay["observer"]
                previous = None if rf.V is None else rf.V.clone()
                loss = F.cross_entropy(rm(batch["inputs"]), batch["labels"])
                self.assertEqual(float(loss.detach()), witness["payload"]["live_loss"]["value_native_float32"])
                loss.backward()
                raw_gradient = flat([parameter.grad for parameter in rm.parameters()])
                lagged = raw_gradient.clone() if previous is None else previous @ (previous.T @ raw_gradient)
                rf.filter_grad()
                current = flat([parameter.grad for parameter in rm.parameters()])
                for name, actual in (("raw_gradient", raw_gradient), ("delivered_current_gradient", current)):
                    self.assertTrue(envelope.same_exact(actual, witness["payload"][name]["value"]))
                actual_observer = state._capture_observer(rf, rm, ro, context["profile"], 5)
                self.assertTrue(envelope.same_exact(actual_observer, witness["payload"]["observer_after"]))
                self.assertTrue(envelope.same_exact(state._capture_rng(), core["rng"]))
                candidates = response.construct(raw_gradient, current, lagged)

                def execute(order):
                    found = {}
                    for branch in order:
                        cloned = state.restore_core(core, model_factory, optimizer_factory, observer_factory)
                        bm, bo = cloned["model"], cloned["optimizer"]
                        gradient = candidates["branches"][branch]["gradient"]
                        self.assertIsNotNone(gradient)  # This specific synthetic fixture is all-defined.
                        offset = 0
                        for parameter in bm.parameters():
                            parameter.grad = gradient[offset:offset+parameter.numel()].reshape(parameter.shape).clone()
                            offset += parameter.numel()
                        bo.step()
                        bo.zero_grad(set_to_none=True)
                        found[branch] = dict(parameters=parameter_values(bm),
                            optimizer=state._capture_optimizer(bo, bm, context["profile"], 5))
                        self.assertTrue(envelope.same_exact(state._capture_rng(), core["rng"]))
                        self.assertTrue(envelope.same_exact(state._capture_observer(cloned["observer"],
                            bm, bo, context["profile"], 4), core["observer"]))
                    return {branch: found[branch] for branch in response.BRANCHES}

                canonical = execute(response.BRANCHES)
                reverse = execute(reversed(response.BRANCHES))
                self.assertTrue(envelope.same_exact(canonical, reverse))
                live_parameters = {entry["name"]: entry["value"] for entry in witness["payload"]["parameters_after"]}
                self.assertTrue(envelope.same_exact(canonical["current"]["parameters"], live_parameters))
                self.assertTrue(envelope.same_exact(canonical["current"]["optimizer"],
                                                   witness["payload"]["optimizer_after"]))
                self.assertTrue(envelope.same_exact(anchor, original))
                before = {entry["name"]: entry["value"].clone() for entry in core["model"]["parameters"]}
                branch_raw = dict(before=before, endpoints={key: value["parameters"] for key, value in canonical.items()},
                                  optimizer_endpoints={key: value["optimizer"] for key, value in canonical.items()},
                                  candidates=candidates, probes=state.clone_tree(materialized["probes"]))
                raw_receipt = store.write_tensor_tree("sealed-source-branch-raw.pt", branch_raw)
                recovered = reload(raw_receipt)
                measured = assembly.assemble(recovered["before"], recovered["endpoints"], recovered["candidates"],
                    recovered["probes"], artifact_id="sealed-source-fixture", profile=context["profile"])
                measurement_receipt = store.write_tensor_tree("sealed-source-measurements.pt", measured)
                measurements = reload(measurement_receipt)
                gradients = {key: row["gradient"] for key, row in recovered["candidates"]["branches"].items()}
                checked = audit.audit(numpy_tree(recovered["before"]), numpy_tree(recovered["endpoints"]),
                    numpy_tree(gradients), numpy_tree(recovered["probes"]), numpy_tree(measurements),
                    profile=context["profile"])
                self.assertEqual(checked["overall_status"], "pass", checked["fatal_failures"][:5])
                self.assertTrue(checked["audit_complete"])
                self.assertEqual(checked["completion_counts"], dict(before_probes=4, branch_probes=24,
                    defined_branches=6, pairs=15, contrasts=15, defined_contrasts=15,
                    auxiliary_before_chunks=10, auxiliary_branch_chunks=60, auxiliary_contrast_chunks=150))
                required = {f"contrasts.{contrast}.{probe}.{kind}.independent"
                    for contrast in response.COEFFICIENTS for probe in assembly.PROBES
                    for kind in ("Y", "D", "Ddata", "R")}
                required.update(f"contrasts.{contrast}.auxiliary_clean.chunk{chunk}.Y.independent"
                    for contrast in response.COEFFICIENTS for chunk in range(10))
                self.assertEqual(len(required), 390)
                self.assertEqual({key for key in checked["comparison_audits"] if key.endswith(".independent")}, required)
                self.assertTrue(all(checked["comparison_audits"][key]["audit_status"] == "pass" for key in required))
                adam_checks = 0
                for branch, endpoints in recovered["endpoints"].items():
                    offset = 0
                    group = recovered["optimizer_endpoints"][branch]["param_groups"][0]
                    for index, (name, endpoint) in enumerate(endpoints.items()):
                        old, new = core["optimizer"]["state"][index], recovered["optimizer_endpoints"][branch]["state"][index]
                        grad = gradients[branch][offset:offset+endpoint.numel()].reshape(endpoint.shape)
                        result = independent.audit_adamw(recovered["before"][name].numpy(), grad.numpy(),
                            old["exp_avg"]["value"].numpy(), old["exp_avg_sq"]["value"].numpy(), 5,
                            dict(theta=endpoint.numpy(), moment=new["exp_avg"]["value"].numpy(),
                                 variance=new["exp_avg_sq"]["value"].numpy(), next_step=int(new["step"])),
                            options={key: group[key] for key in independent.ADAMW_OPTIONS})
                        self.assertTrue(result["passed"], result)
                        offset += endpoint.numel()
                        adam_checks += 1
                self.assertEqual(adam_checks, 24)
                store.write_tensor_tree("sealed-source-audit.pt", checked)
                self.assertTrue(envelope.same_exact(anchor, original))
                self.assertTrue(envelope.same_exact(state._capture_rng(), core["rng"]))
                self.assertFalse(torch.cuda.is_initialized())


if __name__ == "__main__":
    unittest.main()
