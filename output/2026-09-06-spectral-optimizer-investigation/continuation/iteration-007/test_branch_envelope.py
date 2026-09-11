"""Adversarial saved-branch tests using actual synthetic source and branch steps.

No source/runtime, candidate, measurement, or envelope validation is mocked on
the positive paths. Synthetic observer edge states are not historical evidence.
"""
from contextlib import ExitStack, contextmanager
import copy
import os
import tempfile
import unittest
from unittest import mock

if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
    raise RuntimeError("explicitly hide CUDA for branch-envelope fixtures")

import torch

import artifact_envelopes as subject
import artifact_store as storage
import branch_execution as execution
from envelope_fixture import fixture_context, warm_live
import response_math as response
import source_capture as source
import state_core as state


CREATED = "2026-09-06T12:34:56Z"
BRANCHES = ("raw", "current", "lagged", "restored", "reciprocal", "zero")
ROOT_KEYS = ("schema_name", "schema_version", "profile", "artifact_id", "created_utc",
             "identity", "payload", "payload_tensor_bytes")
PAYLOAD_KEYS = ("candidate_state", "branches", "comparisons", "audit_metadata")
CANDIDATE_KEYS = ("g", "c", "l", "previous_basis", "post_ingest_basis", "nc", "nl",
                  "observer_after", "measurement_before", "leverage", "zero_cases")
BRANCH_KEYS = ("status", "reason", "delivered_gradient", "assigned_gradient_null_mask",
               "parameters_after", "parameters_after_flat", "optimizer_after", "measurement")


def tensor_leaves(value):
    if type(value) is torch.Tensor:
        yield value
    elif type(value) is dict:
        for child in value.values():
            yield from tensor_leaves(child)
    elif type(value) in (list, tuple):
        for child in value:
            yield from tensor_leaves(child)


def replace(value, path, new):
    for key in path[:-1]:
        value = value[key]
    value[path[-1]] = new


def exact(left, right):
    """Independent typed byte equality, not subject hashes or proof flags."""
    if type(left) is not type(right):
        return False
    if type(left) is torch.Tensor:
        return (left.dtype == right.dtype and left.shape == right.shape and left.device == right.device
                and left.detach().numpy().tobytes() == right.detach().numpy().tobytes())
    if type(left) is dict:
        return len(left) == len(right) and all(exact(ak, bk) and exact(av, bv)
            for (ak, av), (bk, bv) in zip(left.items(), right.items()))
    if type(left) in (list, tuple):
        return len(left) == len(right) and all(exact(a, b) for a, b in zip(left, right))
    return left.hex() == right.hex() if type(left) is float else left == right


@contextmanager
def completed_case(mode="ordinary"):
    with fixture_context() as context, tempfile.TemporaryDirectory(prefix="i7-branch-envelope-") as parent:
        model, optimizer, observer = warm_live(context)
        if mode == "zero_lagged":
            # A deliberately constructed but valid synthetic anchor: both old
            # basis directions lie in inactive hidden coordinates. The next raw
            # gradient lives only in output biases and the new innovation wins.
            with torch.no_grad():
                model[0].weight.zero_()
                model[0].bias.fill_(-1.0)
            observer.V = torch.zeros((26, 2), dtype=torch.float32)
            observer.V[0, 0], observer.V[1, 1] = 1.0, 1.0
            observer.S = torch.full((2,), 1e-30, dtype=torch.float64)
            observer.grad_mean = torch.zeros(26, dtype=torch.float32)
        elif mode == "missing_previous_basis":
            observer.V, observer.S = None, None
        elif mode != "ordinary":
            raise ValueError("unknown synthetic fixture mode")
        with storage.ArtifactStore(parent, profile=context["profile"],
                                   min_filesystem_free_bytes=0) as store:
            captured = source.capture_anchor_then_live_witness(
                model, optimizer, observer, store=store, created_utc=CREATED, **context)
            before = state.clone_tree(captured)
            before_rng = state._raw_rng_state()
            produced = execution.execute_branches(store=store, created_utc=CREATED,
                                                   **captured, **context)
            if not exact(before, captured) or not exact(before_rng, state._raw_rng_state()):
                raise AssertionError("actual branch execution mutated saved input or caller RNG")
            yield {"context": context, "source": captured, "produced": produced, "store": store}


class BranchEnvelopeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.addClassCleanup(torch.set_num_threads, old_threads)
        stack = ExitStack()
        cls.addClassCleanup(stack.close)
        cls.case = stack.enter_context(completed_case())

    def setUp(self):
        self.original = self.case["produced"]["artifact"]
        self.value = copy.deepcopy(self.original)

    def kwargs(self, case=None):
        case = self.case if case is None else case
        return {**case["context"], **case["source"],
                "expected_candidates": case["produced"]["candidates"],
                "expected_measurements": case["produced"]["measurements"]}

    def reject(self, path, new):
        value = copy.deepcopy(self.original)
        replace(value, path, new)
        with self.assertRaises(subject.ArtifactEnvelopeError):
            subject.validate_branch_results(value, **self.kwargs())

    def test_actual_source_and_branches_have_exact_ordered_schema(self):
        produced, value = self.case["produced"], self.original
        self.assertEqual(tuple(produced), ("artifact", "receipt", "candidates", "measurements"))
        self.assertIs(subject.validate_branch_results(value, **self.kwargs()), value)
        self.assertEqual(tuple(value), ROOT_KEYS)
        self.assertEqual(tuple(value["payload"]), PAYLOAD_KEYS)
        self.assertEqual(tuple(value["payload"]["candidate_state"]), CANDIDATE_KEYS)
        self.assertEqual(tuple(value["payload"]["branches"]), BRANCHES)
        self.assertEqual(value["schema_name"], "i7_branch_results")
        self.assertEqual(value["artifact_id"], "fixture_tiny_mlp_cpu_v1--primary--b0--u5--branch-results")
        for branch, row in value["payload"]["branches"].items():
            self.assertEqual(tuple(row), BRANCH_KEYS)
            self.assertEqual(row["status"], "defined", branch)
            self.assertIsNone(row["reason"])
            self.assertEqual(row["assigned_gradient_null_mask"], [False] * 4)
            flat = torch.cat([entry["value"].reshape(-1) for entry in row["parameters_after"]])
            self.assertTrue(exact(flat, row["parameters_after_flat"]))
            self.assertEqual(row["optimizer_after"]["state_completed_updates"], 5)
        self.assertEqual(tuple(value["payload"]["comparisons"]), ("branch_pairs", "contrasts"))
        self.assertEqual(len(value["payload"]["comparisons"]["branch_pairs"]), 15)
        self.assertEqual(tuple(value["payload"]["comparisons"]["contrasts"]), tuple(response.COEFFICIENTS))
        self.assertFalse(torch.cuda.is_initialized())

    def test_current_native_endpoints_and_gradients_bind_directly_to_source(self):
        payload, witness = self.original["payload"], self.case["source"]["witness"]["payload"]
        candidate, current = payload["candidate_state"], payload["branches"]["current"]
        for left, right in ((candidate["g"], witness["raw_gradient"]),
                            (candidate["c"], witness["delivered_current_gradient"]),
                            (current["delivered_gradient"], witness["delivered_current_gradient"]),
                            (current["parameters_after"], witness["parameters_after"]),
                            (current["optimizer_after"], witness["optimizer_after"]),
                            (candidate["observer_after"], witness["observer_after"])):
            self.assertTrue(exact(left, right))
        proof = payload["audit_metadata"]["current_replay_proof"]
        self.assertEqual(proof["overall_status"], "exact")
        self.assertEqual(tuple(proof["checks"]), ("raw_gradient", "current_gradient", "delivered_gradient",
            "parameters_after", "optimizer_after", "observer_after", "rng", "live_loss"))

    def test_restricted_store_roundtrip_and_payload_byte_count(self):
        receipt, store = self.case["produced"]["receipt"], self.case["store"]
        with mock.patch.object(storage.torch, "load", wraps=torch.load) as decoder:
            recovered = storage.ArtifactStore.load_tensor_tree(store.root, receipt["name"],
                expected_size=receipt["size"], expected_sha256=receipt["sha256"])
        self.assertEqual(decoder.call_count, 1)
        self.assertEqual(decoder.call_args.kwargs, {"map_location": "cpu", "weights_only": True})
        self.assertTrue(exact(recovered, self.original))
        self.assertIs(subject.validate_branch_results(recovered, **self.kwargs()), recovered)
        expected_bytes = sum(t.numel() * t.element_size() for t in tensor_leaves(recovered["payload"]))
        self.assertEqual(recovered["payload_tensor_bytes"], expected_bytes)
        self.assertNotEqual(receipt["size"], expected_bytes)
        self.assertFalse(storage.ArtifactStore.inspect(store.root)["terminal"])

    def test_every_envelope_and_branch_key_set_order_and_type_is_exact(self):
        class DictSubclass(dict):
            pass
        for path in ((), ("payload",), ("payload", "candidate_state"), ("payload", "branches"),
                     ("payload", "branches", "raw"), ("payload", "audit_metadata")):
            original = self.original
            for part in path:
                original = original[part]
            for new in ({**original, "extra": None}, dict(list(original.items())[1:]),
                        dict(reversed(list(original.items()))), DictSubclass(original)):
                value = copy.deepcopy(self.original)
                if path:
                    replace(value, path, new)
                else:
                    value = new
                with self.subTest(path=path, keys=tuple(new)), self.assertRaises(subject.ArtifactEnvelopeError):
                    subject.validate_branch_results(value, **self.kwargs())

    def test_profile_identity_utc_scalar_types_and_byte_budget_rejected(self):
        for path, new in ((("schema_version",), True), (("profile",), storage.FIXTURE),
                          (("profile",), storage.SCIENTIFIC), (("identity", "bundle"), False),
                          (("created_utc",), "2026-09-06T12:34:56+00:00"),
                          (("payload_tensor_bytes",), True),
                          (("payload_tensor_bytes",), self.original["payload_tensor_bytes"] + 1),
                          (("payload", "candidate_state", "nc"), 1),
                          (("payload", "candidate_state", "nl"), float("nan"))):
            with self.subTest(path=path):
                self.reject(path, new)

    def test_candidate_bases_gradients_observer_and_norms_are_bound(self):
        root = ("payload", "candidate_state")
        candidate = self.original["payload"]["candidate_state"]
        for field in ("g", "c", "l", "previous_basis", "post_ingest_basis"):
            tensor = candidate[field]["value"].clone()
            tensor.reshape(-1)[0] += 0.125
            with self.subTest(field=field):
                self.reject(root + (field, "value"), tensor)
        self.reject(root + ("previous_basis",), None)
        self.reject(root + ("post_ingest_basis",), None)
        self.reject(root + ("nc",), candidate["nc"] + 1.0)
        self.reject(root + ("observer_after", "state", "step_count"), 4)
        self.reject(root + ("observer_after", "state_completed_observations"), 4)

    def test_zero_domain_mask_and_assigned_masks_cannot_be_self_reported(self):
        base = ("payload", "candidate_state", "zero_cases")
        for path, new in ((base + ("current_zero",), True), (base + ("lagged_zero",), True),
                          (base + ("branch_defined_mask", "restored"), False),
                          (base + ("branch_reasons", "restored"), "positive_current_norm_zero_lagged_direction"),
                          (("payload", "branches", "raw", "assigned_gradient_null_mask"), [0] * 4),
                          (("payload", "branches", "raw", "assigned_gradient_null_mask"), [True] * 4),
                          (("payload", "branches", "restored", "status"), "undefined"),
                          (("payload", "branches", "zero", "reason"), "numerical_failure")):
            with self.subTest(path=path):
                self.reject(path, new)
        zero = self.original["payload"]["branches"]["zero"]["delivered_gradient"]["value"].clone()
        zero[0] = 1e-30
        self.reject(("payload", "branches", "zero", "delivered_gradient", "value"), zero)

    def test_flat_endpoint_metadata_moments_and_counter_corruptions_rejected(self):
        root = ("payload", "branches", "raw")
        row = self.original["payload"]["branches"]["raw"]
        flat = row["parameters_after_flat"].clone()
        flat[0] += 0.1
        self.reject(root + ("parameters_after_flat",), flat)
        self.reject(root + ("parameters_after", 0, "index"), True)
        self.reject(root + ("parameters_after", 0, "native_device"), "cpu:0")
        self.reject(root + ("optimizer_after", "state_completed_updates"), 4)
        self.reject(root + ("optimizer_after", "state", 0, "step"), torch.tensor(4.0))
        self.reject(root + ("optimizer_after", "param_groups", 0, "decoupled_weight_decay"), False)
        for field in ("exp_avg", "exp_avg_sq"):
            changed = row["optimizer_after"]["state"][0][field]["value"].clone()
            changed.reshape(-1)[0] += 0.125
            self.reject(root + ("optimizer_after", "state", 0, field, "value"), changed)

    def test_endpoint_displacement_binding_survives_consistently_rehashed_proof(self):
        bad = copy.deepcopy(self.original)
        payload = bad["payload"]
        row = payload["branches"]["raw"]
        row["parameters_after"][0]["value"].reshape(-1)[0] += 0.125
        row["parameters_after_flat"] = torch.cat(
            [entry["value"].reshape(-1) for entry in row["parameters_after"]]).clone()
        # Rehashing attacker-modified endpoints is not independent evidence.
        # The separately supplied measurements still bind the true displacement.
        payload["audit_metadata"] = subject.expected_audit_metadata(
            payload["candidate_state"], payload["branches"], **self.case["source"])
        with self.assertRaises(subject.ArtifactEnvelopeError):
            subject.validate_branch_results(bad, **self.kwargs())

    def test_proof_flags_hashes_orders_and_completion_counts_rejected(self):
        root = ("payload", "audit_metadata")
        for path, new in ((root + ("candidate_proof", "observer_ingests"), True),
                          (root + ("candidate_proof", "candidate_core_sha256"), "0" * 64),
                          (root + ("clone_independence_proof", "anchor_directly_unchanged"), False),
                          (root + ("clone_independence_proof", "per_order", "canonical", "raw", "storage_disjoint"), False),
                          (root + ("branch_execution_proof", "canonical_order"), list(reversed(BRANCHES))),
                          (root + ("branch_execution_proof", "observer_ingests"), 1),
                          (root + ("branch_execution_proof", "per_order", "reverse", "raw", "optimizer_sha256"), "0" * 64),
                          (root + ("current_replay_proof", "checks", "live_loss", "replay_sha256"), "0" * 64),
                          (root + ("current_replay_proof", "checks", "rng", "replay_directly_equals_anchor"), False),
                          (root + ("completion", "defined_branch_count"), 5),
                          (root + ("completion", "producer_artifact_status"), "independently_audited")):
            with self.subTest(path=path):
                self.reject(path, new)

    def test_receipt_references_and_external_expected_values_are_bound(self):
        for reference in ("anchor_ref", "source_witness_ref"):
            for field, new in (("sha256", "0" * 64), ("receipt_sha256", "0" * 64),
                               ("size_bytes", True), ("receipt_size_bytes", 1), ("encoding", "pickle")):
                self.reject(("payload", "audit_metadata", "producer_bindings", reference, field), new)
        for field in ("anchor_receipt", "witness_receipt"):
            kwargs = self.kwargs()
            kwargs[field] = copy.deepcopy(kwargs[field])
            kwargs[field]["sha256"] = "0" * 64
            with self.subTest(field=field), self.assertRaises(subject.ArtifactEnvelopeError):
                subject.validate_branch_results(self.value, **kwargs)

    def test_jointly_malformed_expected_comparison_membership_and_masks_rejected(self):
        paths = (("branch_pairs",), ("contrasts",))
        mutations = []
        for path in paths:
            original = self.original["payload"]["comparisons"][path[0]]
            mutations.extend((path, changed) for changed in (
                dict(list(original.items())[1:]), dict(reversed(list(original.items()))),
                {**original, "extra": copy.deepcopy(next(iter(original.values())))}))
        mutations.extend((
            (("branch_pairs", "raw__current", "defined_mask"), {"raw": False, "current": True}),
            (("branch_pairs", "raw__current", "defined"), False),
            (("contrasts", "ordering", "branch_defined_mask"), {key: 1 for key in BRANCHES}),
            (("contrasts", "ordering", "branch_defined_mask"),
             {key: key != "raw" for key in BRANCHES}),
            (("contrasts", "ordering", "coefficients"), {"lagged": 1.0, "current": -1.0}),
            (("contrasts", "ordering", "defined"), False),
        ))
        for path, changed in mutations:
            bad = copy.deepcopy(self.original)
            kwargs = self.kwargs()
            kwargs["expected_measurements"] = copy.deepcopy(kwargs["expected_measurements"])
            replace(bad["payload"]["comparisons"], path, copy.deepcopy(changed))
            replace(kwargs["expected_measurements"]["comparisons"], path, copy.deepcopy(changed))
            # Direct equality to a malformed expected snapshot cannot substitute
            # for the independently fixed 15-pair/contrast schema and masks.
            with self.subTest(path=path), self.assertRaises(subject.ArtifactEnvelopeError):
                subject.validate_branch_results(bad, **kwargs)
        for field in ("expected_candidates", "expected_measurements"):
            kwargs = self.kwargs()
            kwargs[field] = copy.deepcopy(kwargs[field])
            if field == "expected_candidates":
                kwargs[field]["branches"]["raw"]["gradient"][0] += 1.0
            else:
                kwargs[field]["measurement_before"]["probes"]["batch_noisy"]["cpu64"]["before_ce"] += 1.0
            with self.subTest(field=field), self.assertRaises(subject.ArtifactEnvelopeError):
                subject.validate_branch_results(self.value, **kwargs)

    def test_measurement_and_live_loss_bindings_cannot_silently_change(self):
        payload = self.original["payload"]
        before = payload["candidate_state"]["measurement_before"]["probes"]["batch_noisy"]
        self.reject(("payload", "candidate_state", "measurement_before", "probes", "batch_noisy",
                     "native32", "before_ce"), before["native32"]["before_ce"] + 0.125)
        probe = payload["branches"]["raw"]["measurement"]["probes"]["batch_noisy"]
        self.reject(("payload", "branches", "raw", "measurement", "probes", "batch_noisy",
                     "cpu64", "after_ce"), probe["cpu64"]["after_ce"] + 0.125)
        self.reject(("payload", "comparisons", "contrasts", "ordering", "coefficients"),
                    {"lagged": 1.0, "current": -1.0})
        kwargs = self.kwargs()
        kwargs["witness"] = copy.deepcopy(kwargs["witness"])
        loss = kwargs["witness"]["payload"]["live_loss"]
        previous = torch.tensor(loss["value_native_float32"], dtype=torch.float32)
        loss["value_native_float32"] = float(torch.nextafter(previous, torch.tensor(float("inf"))).item())
        with self.assertRaises(subject.ArtifactEnvelopeError):
            subject.validate_branch_results(self.value, **kwargs)

    def test_signed_zero_gradient_change_is_not_numeric_equality(self):
        before = self.original["payload"]["branches"]["zero"]["delivered_gradient"]["value"]
        changed = before.clone()
        changed[0] = -0.0
        self.assertTrue(torch.equal(before, changed))
        self.assertFalse(exact(before, changed))
        self.reject(("payload", "branches", "zero", "delivered_gradient", "value"), changed)

    def test_alias_grad_bearing_and_oversized_storage_fail_before_constructor_clone(self):
        for variant in ("alias", "requires_grad", "oversized_storage"):
            value = copy.deepcopy(self.original)
            candidate = value["payload"]["candidate_state"]
            if variant == "alias":
                candidate["c"]["value"] = candidate["g"]["value"]
            elif variant == "requires_grad":
                candidate["g"]["value"].requires_grad_()
            else:
                previous = candidate["g"]["value"]
                backing = torch.empty(previous.numel() + 1, dtype=previous.dtype)
                candidate["g"]["value"] = backing[:-1].copy_(previous)
            with self.subTest(variant=variant), self.assertRaises(subject.ArtifactEnvelopeError):
                subject.validate_branch_results(value, **self.kwargs())
            payload = value["payload"]
            with self.subTest(constructor=variant), self.assertRaises(subject.ArtifactEnvelopeError):
                subject.make_branch_results(*(payload[key] for key in PAYLOAD_KEYS),
                    created_utc=CREATED, **self.kwargs())

    def test_constructor_outputs_are_owned_and_success_failure_are_rng_neutral(self):
        before_inputs = copy.deepcopy({"value": self.value, "expected": self.kwargs()})
        before_rng = state._raw_rng_state()
        payload = self.value["payload"]
        made = subject.make_branch_results(*(payload[key] for key in PAYLOAD_KEYS),
            created_utc=CREATED, **self.kwargs())
        self.assertTrue(exact(made, self.value))
        self.assertTrue(exact(before_rng, state._raw_rng_state()))
        pointers = [t.untyped_storage().data_ptr() for t in tensor_leaves(made)]
        self.assertEqual(len(pointers), len(set(pointers)))
        inputs = {t.untyped_storage().data_ptr() for t in tensor_leaves({"value": self.value, "expected": self.kwargs()})}
        self.assertFalse(set(pointers) & inputs)
        with self.assertRaises(subject.ArtifactEnvelopeError):
            subject.make_branch_results(*(payload[key] for key in PAYLOAD_KEYS),
                created_utc="bad", **self.kwargs())
        self.assertTrue(exact(before_inputs, {"value": self.value, "expected": self.kwargs()}))
        self.assertTrue(exact(before_rng, state._raw_rng_state()))

    def test_actual_zero_lagged_domain_is_retained_without_available_case_repair(self):
        with completed_case("zero_lagged") as case:
            value = case["produced"]["artifact"]
            self.assertIs(subject.validate_branch_results(value, **self.kwargs(case)), value)
            candidate = value["payload"]["candidate_state"]
            self.assertGreater(candidate["nc"], 0.0)
            self.assertEqual(candidate["nl"], 0.0)
            row = value["payload"]["branches"]["restored"]
            self.assertEqual(row["status"], "undefined")
            self.assertEqual(row["reason"], "positive_current_norm_zero_lagged_direction")
            self.assertTrue(all(row[key] is None for key in BRANCH_KEYS[2:]))
            self.assertEqual(value["payload"]["audit_metadata"]["completion"]["defined_branch_count"], 5)
            kwargs = self.kwargs(case)
            kwargs["expected_candidates"] = copy.deepcopy(kwargs["expected_candidates"])
            kwargs["expected_candidates"]["branches"]["restored"]["target_norm"] = 0.0
            with self.assertRaises(subject.ArtifactEnvelopeError):
                subject.validate_branch_results(value, **kwargs)
            for path, new in ((("payload", "branches", "restored", "parameters_after"), []),
                              (("payload", "candidate_state", "zero_cases", "branch_defined_mask", "restored"), True)):
                bad = copy.deepcopy(value)
                replace(bad, path, new)
                with self.assertRaises(subject.ArtifactEnvelopeError):
                    subject.validate_branch_results(bad, **self.kwargs(case))

    def test_actual_missing_previous_basis_means_identity_not_null_domain(self):
        with completed_case("missing_previous_basis") as case:
            value = case["produced"]["artifact"]
            self.assertIs(subject.validate_branch_results(value, **self.kwargs(case)), value)
            candidate = value["payload"]["candidate_state"]
            self.assertIsNone(candidate["previous_basis"])
            self.assertTrue(exact(candidate["g"], candidate["l"]))
            self.assertEqual(value["payload"]["branches"]["lagged"]["status"], "defined")


if __name__ == "__main__":
    unittest.main()
