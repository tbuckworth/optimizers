"""Small CPU algebra tests for the I13 postprocessor."""
from __future__ import annotations

from pathlib import Path
import sys
import unittest

import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import analyse_means as analysis


class AnalysisTests(unittest.TestCase):
    def test_expected_coverage_and_seed_first_missingness(self):
        self.assertEqual(len(analysis.expected_new_keys()), 36)
        self.assertEqual(len(analysis.expected_reference_keys()), 36)
        cells = [{"seed": seed, "parent_step": parent, "effect": float(seed + parent)}
                 for seed in analysis.SEEDS for parent in analysis.ANCHORS]
        result = analysis.aggregate(cells, "effect")
        self.assertTrue(result["available"])
        self.assertEqual(len(result["all_three_seed_values"]), 3)
        cells[-1]["effect"] = None
        missing = analysis.aggregate(cells, "effect")
        self.assertFalse(missing["available"])
        self.assertIsNone(missing["mean"])
        self.assertIsNone(missing["all_three_seed_values"][-1])
        self.assertEqual(analysis.expected_probe_horizons([0, 1, 10, 50],
                         "numerical_failure", {"phase": "evaluation"}), [0, 1, 10, 50])
        self.assertEqual(analysis.expected_probe_horizons([0, 1, 10, 50],
                         "numerical_failure", {"phase": "probe"}), [0, 1, 10])
        self.assertEqual(analysis.expected_probe_horizons(list(analysis.HORIZONS),
                         "complete", None), list(analysis.HORIZONS))
        same_applied = [
            {"completed": 1, "raw": "r", "observer": "o", "applied": "same"},
            {"completed": 1, "raw": "r", "observer": "o", "applied": "same"},
        ]
        check = analysis.fingerprint_check(same_applied)
        self.assertTrue(check["available"] and check["raw_equal"] and check["observer_equal"])
        self.assertFalse(check["applied_different"])

    def test_alignment_signed_arithmetic(self):
        mean = torch.tensor([2.0, -1.0, 3.0])
        projected = torch.tensor([2.0, 0.0, 0.0])
        complement = mean - projected
        gradients = {
            "train_fixed": torch.tensor([0.0, 2.0, -1.0]),
            "train_soft": torch.tensor([0.0, 1.0, 1.5]),
            "train_clean": torch.tensor([1.0, 0.0, -1.0]),
            "aux_clean": torch.tensor([0.0, 3.0, -2.0]),
        }
        gradients["fixed_minus_soft"] = gradients["train_fixed"] - gradients["train_soft"]
        alignments = {}
        left_norm = float(torch.linalg.vector_norm(complement.double()))
        for name, gradient in gradients.items():
            right_norm = float(torch.linalg.vector_norm(gradient.double()))
            dot = float(torch.dot(complement.double(), gradient.double()))
            alignments[name] = {"gradient_norm": right_norm, "dot": dot,
                                "cosine": dot / (left_norm * right_norm), "reason": None}
        tensors = {"incumbent_mean": mean, "mean_projection": projected,
                   "complement_mean": complement, "gradients": gradients}
        record = {
            "schema": "i13_alignment_probe_v1", "status": "complete",
            "state_sha256": "a" * 64, "observer_step": 9, "basis_rank": 1,
            "complement_mean_norm": left_norm, "alignments": alignments,
            "tensor_digests": {
                "incumbent_mean": analysis.a10.tree_digest(mean),
                "incumbent_basis": "b" * 64,
                "mean_projection": analysis.a10.tree_digest(projected),
                "complement_mean": analysis.a10.tree_digest(complement),
                "gradients": {name: analysis.a10.tree_digest(value)
                              for name, value in gradients.items()},
            },
        }
        audit = analysis.a10.Audit()
        result = analysis.alignment_arithmetic(record, tensors, audit, "synthetic")
        self.assertFalse(audit.errors)
        self.assertLess(result["alignments"]["train_fixed"]["dot"], 0)
        self.assertGreater(result["alignments"]["train_soft"]["dot"], 0)

    def test_paired_delivery_and_reciprocal_norm_arithmetic(self):
        raw = torch.tensor([1.0, 2.0, 3.0, 4.0])
        basis = torch.tensor([[1.0], [0.0], [0.0], [0.0]])
        native = basis @ (basis.T @ raw)
        pre = torch.tensor([0.0, 2.0, 0.0, 0.0])
        post = .99 * pre + .01 * raw
        mean = native + post - basis @ (basis.T @ post)
        leak = native + .01 * (raw - native)
        vectors = {
            "current32": torch.tensor([1.0, 0.0, 0.0, 0.0]),
            "mean32": torch.tensor([0.0, 2.0, 0.0, 0.0]),
            "leak01_32": torch.tensor([0.0, 0.0, 3.0, 0.0]),
            "mean_to_current_data_norm": torch.tensor([0.0, 1.0, 0.0, 0.0]),
            "current_to_mean_data_norm": torch.tensor([2.0, 0.0, 0.0, 0.0]),
            "leak_to_current_data_norm": torch.tensor([0.0, 0.0, 1.0, 0.0]),
            "current_to_leak_data_norm": torch.tensor([3.0, 0.0, 0.0, 0.0]),
        }
        interventions = {name: {"total_delta": value, "data_delta": value.clone()}
                         for name, value in vectors.items()}
        baseline = {name: 1.0 for name in
                    ("train_fixed", "train_soft", "train_clean", "aux_clean", "aux_soft")}
        saved_interventions = {}
        for name, value in vectors.items():
            norm = float(torch.linalg.vector_norm(value.double()))
            losses = {key: 1.0 + .01 * norm for key in baseline}
            saved_interventions[name] = {
                "status": "defined", "reason": None, "scale": None,
                "total_norm": norm, "data_norm": norm, "losses": losses,
                "loss_changes": {key: .01 * norm for key in baseline},
            }
        tensors = {"pre_mean": pre, "raw_gradient": raw, "native_projection": native,
                   "post_mean": post, "post_basis": basis,
                   "delivered_gradients": {"current32": native, "mean32": mean,
                                           "leak01_32": leak},
                   "interventions": interventions}
        record = {
            "schema": "i13_paired_step_probe_v1", "status": "complete",
            "objective": "soft", "baseline_losses": baseline,
            "interventions": saved_interventions,
            "tensor_digests": {
                "raw_gradient": analysis.a10.tree_digest(raw),
                "post_basis": analysis.a10.tree_digest(basis),
                "delivered_gradients": {name: analysis.a10.tree_digest(value)
                                        for name, value in tensors["delivered_gradients"].items()},
                "interventions": {name: analysis.a10.tree_digest(value)
                                  for name, value in interventions.items()},
            },
        }
        parent = {"model_state": {str(index): torch.zeros(1) for index in range(4)}}
        audit = analysis.a10.Audit()
        result = analysis.paired_arithmetic(record, tensors, parent, audit, "synthetic")
        self.assertFalse(audit.errors)
        self.assertLess(max(result["vector_identity_max_abs_errors"].values()), 1e-6)


if __name__ == "__main__":
    unittest.main()
