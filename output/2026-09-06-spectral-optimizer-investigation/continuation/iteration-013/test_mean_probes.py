"""Small synthetic CPU tests for I13 probes; no dataset or experiment execution."""
from __future__ import annotations

from pathlib import Path
import random
import sys
import unittest
from unittest import mock

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import mean_core
import mean_probes as probes


i9 = mean_core.i9


def fixture():
    generator = torch.Generator().manual_seed(13013)
    x = torch.rand((48, 4), generator=generator)
    clean = torch.arange(48, dtype=torch.long).remainder(3)
    noisy = (clean + 1 + torch.arange(48).remainder(2)).remainder(3)
    data = {"x": x, "clean": clean, "noisy": noisy,
            "vx": x.flip(0).clone(), "vy": clean.flip(0).clone(),
            "ax": x.roll(7, 0).clone(), "ay": clean.roll(7).clone()}
    model = i9.make_model(13013, "cpu", input_dim=4, width=5, classes=3)
    optimizer = i9.make_optimizer(model)
    tracker = i9.make_tracker(model, optimizer)
    for step in range(105):
        indices = torch.arange(step, step + 8).remainder(48)
        i9.train_step(model, optimizer, tracker, data["x"][indices],
                      data["noisy"][indices], "current32")
    state = i9.snapshot(model, optimizer, tracker)
    probe_plan = {
        "loss_train": torch.arange(1024).remainder(48),
        "loss_aux": torch.arange(17, 1041).remainder(48),
        "utility_aux": torch.arange(31, 1055).remainder(48),
    }
    rows = torch.arange(500 * 64, dtype=torch.long).reshape(500, 64)
    branch_plan = {
        "seed": 13013, "parent_step": 1500, "batches": rows.remainder(48),
        "redraw_mask": rows.remainder(4) != 0,
        "redraw_digits": (rows * 2 + 1).remainder(3),
    }
    return state, data, probe_plan, branch_plan


class MeanProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            if torch.get_num_interop_threads() != 1:
                raise

    def test_alignment_exact_vectors_and_neutrality(self):
        state, data, probe_plan, _ = fixture()
        state_before = i9._cpu_clone(state)
        data_before = {key: value.clone() for key, value in data.items()}
        random.seed(91)
        np.random.seed(92)
        torch.manual_seed(93)
        rng_before = i9._rng_state()
        calls = []

        record, tensors = probes.alignment_probe(
            state, data, probe_plan, check=lambda: calls.append(len(calls)))

        self.assertEqual(record["schema"], "i13_alignment_probe_v1")
        self.assertEqual(record["status"], "complete")
        self.assertGreater(len(calls), 1)
        self.assertTrue(all(value.device.type == "cpu" for value in (
            tensors["incumbent_mean"], tensors["mean_projection"],
            tensors["complement_mean"])))
        saved_tracker = state["tracker"]
        k = saved_tracker["V"].shape[1] if saved_tracker["proj_k"] is None \
            else min(saved_tracker["proj_k"], saved_tracker["V"].shape[1])
        basis, mean = saved_tracker["V"][:, :k], tensors["incumbent_mean"]
        self.assertTrue(torch.equal(tensors["mean_projection"], basis @ (basis.T @ mean)))
        self.assertTrue(torch.equal(tensors["complement_mean"],
                                    mean - tensors["mean_projection"]))
        gradients = tensors["gradients"]
        self.assertTrue(torch.equal(gradients["fixed_minus_soft"],
                                    gradients["train_fixed"] - gradients["train_soft"]))
        expected_dot = i9._dot(tensors["complement_mean"], gradients["aux_clean"])
        self.assertEqual(record["alignments"]["aux_clean"]["dot"], expected_dot)
        self.assertEqual(record["tensor_payload_sha256"], i9.tree_digest(tensors))
        self.assertEqual(record["tensor_digests"]["incumbent_basis"],
                         i9.tree_digest(basis))
        self.assertTrue(i9.equal_tree(state_before, state))
        self.assertTrue(i9.equal_tree(rng_before, i9._rng_state()))
        self.assertTrue(all(torch.equal(data_before[key], data[key]) for key in data))

    def test_paired_three_objectives_identities_norm_controls_and_neutrality(self):
        state, data, probe_plan, branch_plan = fixture()
        state_digest = i9.tree_digest(state)
        data_before = {key: value.clone() for key, value in data.items()}
        rng_before = i9._rng_state()
        first = torch.as_tensor(branch_plan["batches"])[0]
        basis = None
        for objective in probes.OBJECTIVES:
            with self.subTest(objective=objective):
                record, tensors = probes.paired_step_probe(
                    state, data, branch_plan, probe_plan, objective)
                self.assertEqual(record["schema"], "i13_paired_step_probe_v1")
                self.assertEqual(record["status"], "complete")
                self.assertEqual(record["objective"], objective)
                self.assertEqual(record["tensor_payload_sha256"], i9.tree_digest(tensors))
                self.assertTrue(all(value.device.type == "cpu" for value in (
                    tensors["raw_gradient"], tensors["post_mean"], tensors["post_basis"])))
                if basis is None:
                    basis = tensors["post_basis"].clone()
                if objective == "fixed":
                    expected_target = data["noisy"][first]
                elif objective == "soft":
                    expected_target = i9._soft(data["clean"][first], 3)
                else:
                    expected_target = torch.where(branch_plan["redraw_mask"][0],
                                                  branch_plan["redraw_digits"][0],
                                                  data["clean"][first])
                self.assertTrue(torch.equal(tensors["first_target"], expected_target.cpu()))
                projection = tensors["post_basis"] @ (
                    tensors["post_basis"].T @ tensors["raw_gradient"])
                self.assertTrue(torch.equal(tensors["native_projection"], projection))
                outside_raw = tensors["raw_gradient"] - tensors["native_projection"]
                mean_expected = tensors["native_projection"] + (
                    tensors["post_mean"]
                    - tensors["post_basis"] @ (tensors["post_basis"].T @ tensors["post_mean"]))
                leak_expected = tensors["native_projection"] + mean_core.LEAK * outside_raw
                self.assertTrue(torch.equal(tensors["delivered_gradients"]["mean32"], mean_expected))
                self.assertTrue(torch.equal(tensors["delivered_gradients"]["leak01_32"], leak_expected))
                self.assertTrue(torch.equal(tensors["delivered_gradients"]["current32"],
                                            tensors["native_projection"]))
                for name, row in record["interventions"].items():
                    self.assertEqual(row["status"], "defined", name)
                    for loss_name in probes.LOSS_NAMES:
                        self.assertEqual(row["loss_changes"][loss_name],
                                         row["losses"][loss_name]
                                         - record["baseline_losses"][loss_name])
                pairs = (("mean_to_current_data_norm", "current32"),
                         ("current_to_mean_data_norm", "mean32"),
                         ("leak_to_current_data_norm", "current32"),
                         ("current_to_leak_data_norm", "leak01_32"))
                for direct, target in pairs:
                    self.assertTrue(torch.isclose(
                        torch.linalg.vector_norm(tensors["interventions"][direct]["data_delta"].double()),
                        torch.linalg.vector_norm(tensors["interventions"][target]["data_delta"].double()),
                        rtol=5e-5, atol=1e-10))
        self.assertEqual(i9.tree_digest(state), state_digest)
        self.assertTrue(i9.equal_tree(rng_before, i9._rng_state()))
        self.assertTrue(all(torch.equal(data_before[key], data[key]) for key in data))

    def test_undefined_direct_controls_are_explicit(self):
        state, data, probe_plan, branch_plan = fixture()
        original = probes.i9.rescale_to_norm
        calls = 0

        def selective(value, target):
            nonlocal calls
            calls += 1
            if calls == 1:
                return None, {"scale": None, "source_norm": 0.0,
                              "target_norm": target, "achieved_norm": None,
                              "reason": "zero_source_norm"}
            if calls == 3:
                return None, {"scale": None, "source_norm": i9._norm(value),
                              "target_norm": target, "achieved_norm": None,
                              "reason": "scale_limit"}
            return original(value, target)

        with mock.patch.object(probes.i9, "rescale_to_norm", side_effect=selective):
            record, tensors = probes.paired_step_probe(
                state, data, branch_plan, probe_plan, "soft")
        self.assertEqual(record["interventions"]["mean_to_current_data_norm"]["status"],
                         "undefined")
        self.assertEqual(record["interventions"]["mean_to_current_data_norm"]["reason"],
                         "zero_source_norm")
        self.assertEqual(record["interventions"]["leak_to_current_data_norm"]["reason"],
                         "scale_limit")
        self.assertIsNone(tensors["interventions"]["mean_to_current_data_norm"]["data_delta"])
        self.assertIsNone(tensors["interventions"]["leak_to_current_data_norm"]["total_delta"])

    def test_resource_callback_propagates_and_restores_rng(self):
        state, data, probe_plan, branch_plan = fixture()
        state_digest = i9.tree_digest(state)
        rng_before = i9._rng_state()

        def stop():
            raise RuntimeError("resource stop")

        with self.assertRaisesRegex(RuntimeError, "resource stop"):
            probes.paired_step_probe(state, data, branch_plan, probe_plan,
                                     "redraw", check=stop)
        self.assertEqual(i9.tree_digest(state), state_digest)
        self.assertTrue(i9.equal_tree(rng_before, i9._rng_state()))

    def test_strict_plan_shapes_reject_before_probe(self):
        state, data, probe_plan, branch_plan = fixture()
        bad_probe = dict(probe_plan)
        bad_probe["loss_train"] = bad_probe["loss_train"][:-1]
        with self.assertRaisesRegex(Exception, "probe indices invalid"):
            probes.alignment_probe(state, data, bad_probe)
        bad_branch = dict(branch_plan)
        bad_branch["batches"] = bad_branch["batches"][:-1]
        with self.assertRaisesRegex(probes.MeanProbeError, "branch plan tensor topology"):
            probes.paired_step_probe(state, data, bad_branch, probe_plan, "fixed")


if __name__ == "__main__":
    unittest.main()
