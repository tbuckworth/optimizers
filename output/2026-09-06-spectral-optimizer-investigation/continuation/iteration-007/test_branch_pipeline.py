"""Parent raw-file-to-independent-audit integration; synthetic CPU engineering."""
import os
import tempfile
import unittest

if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
    raise RuntimeError("explicitly hide CUDA for branch pipeline fixtures")

import torch

import anchor_envelope as envelope
import artifact_envelopes as envelopes
import artifact_store as storage
import branch_execution as execution
import data_probe_bindings as data
import independent_numerics as independent
import measurement_assembly as assembly
import measurement_audit as audit
import response_math as response
import source_capture as source
import state_core as state
from envelope_fixture import fixture_context, warm_live
from test_full_measurement_pipeline import numpy_tree


class BranchPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def run_pipeline(self, *, domain_null, different_caller_rng=False):
        with fixture_context() as context, tempfile.TemporaryDirectory(prefix="i7-branch-audit-") as root:
            model, optimizer, observer = warm_live(context)
            if domain_null:
                # Explicit engineered synthetic starting state, NOT a continuation
                # of the warm-up trajectory or scientific evidence. A valid
                # orthonormal prior basis is supported only on dead first-layer
                # weights. Actual next raw gradient has output-bias support only.
                # Zero old eigenvalues allow the observer to acquire that new
                # direction. No producer gradient/operator callback is replaced.
                with torch.no_grad():
                    for parameter in model.parameters():
                        parameter.zero_()
                    model[2].bias.copy_(torch.tensor([.4, -.4]))
                observer.V = torch.eye(26, dtype=torch.float32)[:, :2].clone()
                observer.S = torch.zeros(2, dtype=torch.float64)
                observer.grad_mean = torch.zeros(26, dtype=torch.float32)
                observer.max_orthogonality_error = 0.0
            with storage.ArtifactStore(root, profile=context["profile"],
                                       min_filesystem_free_bytes=0) as store:
                transaction = source.capture_anchor_then_live_witness(model, optimizer, observer,
                    store=store, created_utc="2026-09-06T21:00:00Z", **context)
                if different_caller_rng:
                    # Branch phase can run after later source trajectories. Its
                    # caller's RNG need not equal this earlier anchor's RNG.
                    torch.manual_seed(900001)
                caller_rng = state._capture_rng()
                live_original = state.capture_core(model, optimizer, observer,
                    profile=context["profile"], completed_updates=5)
                outcome = execution.execute_branches(store=store,
                    created_utc="2026-09-06T21:00:01Z", **transaction, **context)
                receipt = outcome["receipt"]
                loaded = storage.ArtifactStore.load_tensor_tree(store.root, receipt["name"],
                    expected_size=receipt["size"], expected_sha256=receipt["sha256"])
                self.assertTrue(envelope.same_exact(loaded, outcome["artifact"]))
                envelopes.validate_branch_results(loaded, **transaction, **context,
                    expected_candidates=outcome["candidates"],
                    expected_measurements=outcome["measurements"])
                core = envelope.core_from_anchor(transaction["anchor"])
                candidate, branches = loaded["payload"]["candidate_state"], loaded["payload"]["branches"]
                witness = transaction["witness"]["payload"]
                # The original live model must remain at its actual source
                # endpoint throughout all branch execution and measurements.
                live_after = state.capture_core(model, optimizer, observer,
                    profile=context["profile"], completed_updates=5)
                self.assertTrue(envelope.same_exact(live_original, live_after))
                self.assertTrue(envelope.same_exact(live_after["model"]["parameters"],
                                                   branches["current"]["parameters_after"]))
                self.assertTrue(envelope.same_exact(live_after["optimizer"],
                                                   branches["current"]["optimizer_after"]))
                self.assertTrue(envelope.same_exact(live_after["observer"], candidate["observer_after"]))
                self.assertTrue(envelope.same_exact(live_after["rng"], caller_rng))
                if different_caller_rng:
                    self.assertFalse(envelope.same_exact(caller_rng, core["rng"]))
                self.assertTrue(envelope.same_exact(candidate["g"], witness["raw_gradient"]))
                self.assertTrue(envelope.same_exact(candidate["c"], witness["delivered_current_gradient"]))

                before = {entry["name"]: entry["value"].clone() for entry in core["model"]["parameters"]}
                endpoints = {key: None if row["status"] == "undefined" else
                    {entry["name"]: entry["value"].clone() for entry in row["parameters_after"]}
                    for key, row in branches.items()}
                gradients = {key: None if row["status"] == "undefined" else
                             row["delivered_gradient"]["value"].clone() for key, row in branches.items()}
                measured = dict(measurement_before=state.clone_tree(candidate["measurement_before"]),
                    branches={key: state.clone_tree(row["measurement"]) for key, row in branches.items()},
                    comparisons=state.clone_tree(loaded["payload"]["comparisons"]))
                materialized = data.materialize(context["images_bytes"], context["labels_bytes"],
                    plan=context["plan"], identity=context["identity"], profile=context["profile"],
                    expected_files=context["expected_files"])
                checked = audit.audit(numpy_tree(before), numpy_tree(endpoints), numpy_tree(gradients),
                    numpy_tree(materialized["probes"]), numpy_tree(measured), profile=context["profile"])
                self.assertEqual(checked["overall_status"], "pass", checked["fatal_failures"][:5])
                self.assertTrue(checked["audit_complete"])
                defined = [key for key, row in branches.items() if row["status"] == "defined"]
                required_contrasts = [key for key, coefficients in response.COEFFICIENTS.items()
                                      if all(branch in defined for branch in coefficients)]
                required = {f"contrasts.{contrast}.{probe}.{kind}.independent"
                    for contrast in required_contrasts for probe in assembly.PROBES
                    for kind in ("Y", "D", "Ddata", "R")}
                required.update(f"contrasts.{contrast}.auxiliary_clean.chunk{chunk}.Y.independent"
                    for contrast in required_contrasts for chunk in range(10))
                self.assertEqual({key for key in checked["comparison_audits"] if key.endswith(".independent")}, required)
                self.assertTrue(all(checked["comparison_audits"][key]["audit_status"] == "pass" for key in required))
                self.assertEqual(checked["completion_counts"], dict(before_probes=4,
                    branch_probes=4*len(defined), defined_branches=len(defined), pairs=15, contrasts=15,
                    defined_contrasts=len(required_contrasts), auxiliary_before_chunks=10,
                    auxiliary_branch_chunks=10*len(defined), auxiliary_contrast_chunks=10*len(required_contrasts)))
                adam_checks = 0
                for branch in defined:
                    group = branches[branch]["optimizer_after"]["param_groups"][0]
                    offset = 0
                    for index, (name, endpoint) in enumerate(endpoints[branch].items()):
                        old = core["optimizer"]["state"][index]
                        new = branches[branch]["optimizer_after"]["state"][index]
                        grad = gradients[branch][offset:offset+endpoint.numel()].reshape(endpoint.shape)
                        result = independent.audit_adamw(before[name].numpy(), grad.numpy(),
                            old["exp_avg"]["value"].numpy(), old["exp_avg_sq"]["value"].numpy(), 5,
                            dict(theta=endpoint.numpy(), moment=new["exp_avg"]["value"].numpy(),
                                 variance=new["exp_avg_sq"]["value"].numpy(), next_step=int(new["step"])),
                            options={key: group[key] for key in independent.ADAMW_OPTIONS})
                        self.assertTrue(result["passed"], result)
                        offset += endpoint.numel()
                        adam_checks += 1
                self.assertEqual(adam_checks, 4*len(defined))
                self.assertFalse(envelope.same_exact(endpoints["zero"], before))
                if domain_null:
                    self.assertEqual(defined, ["raw", "current", "lagged", "reciprocal", "zero"])
                    self.assertGreater(candidate["nc"], 0.0)
                    self.assertEqual(candidate["nl"], 0.0)
                    self.assertEqual(branches["restored"]["reason"], "positive_current_norm_zero_lagged_direction")
                    self.assertTrue(all(branches["restored"][key] is None for key in
                                       tuple(branches["restored"])[2:]))
                else:
                    self.assertEqual(len(defined), 6)
                    self.assertEqual(len(required), 390)
                # This raw independent audit record is not yet the future
                # complete provenance/audit envelope or all-phase manifest.
                audit_receipt = store.write_tensor_tree("branch-pipeline-independent-audit.pt", checked)
                audit_loaded = storage.ArtifactStore.load_tensor_tree(store.root, audit_receipt["name"],
                    expected_size=audit_receipt["size"], expected_sha256=audit_receipt["sha256"])
                self.assertTrue(envelope.same_exact(audit_loaded, checked))
                self.assertTrue(envelope.same_exact(state._capture_rng(), caller_rng))
                self.assertFalse(torch.cuda.is_initialized())

    def test_actual_sealed_branches_to_independent_numerical_audit(self):
        self.run_pipeline(domain_null=False)

    def test_engineered_zero_lagged_direction_preserves_null_and_unaffected_contrasts(self):
        self.run_pipeline(domain_null=True)

    def test_branch_phase_restores_distinct_caller_rng_after_earlier_anchor_replay(self):
        self.run_pipeline(domain_null=False, different_caller_rng=True)


if __name__ == "__main__":
    unittest.main()
