import copy
import json
import math
import unittest

import torch

from experiments.grokking_model import GrokkingTransformer
from experiments import grokking_representation as gr


def modular_grid(p):
    a = torch.arange(p).repeat_interleave(p)
    b = torch.arange(p).repeat(p)
    return torch.stack((a, b), dim=1), (a + b) % p


def fourier_features(sums, p, frequency):
    angle = (2 * math.pi * frequency / p) * sums.double()
    return torch.stack((angle.cos(), angle.sin()), dim=1)


def assert_finite_json(testcase, value):
    if isinstance(value, dict):
        for item in value.values():
            assert_finite_json(testcase, item)
    elif isinstance(value, list):
        for item in value:
            assert_finite_json(testcase, item)
    elif isinstance(value, float):
        testcase.assertTrue(math.isfinite(value))


class GrokkingRepresentationTest(unittest.TestCase):
    def test_split_and_row_permutation_contracts(self):
        _, sums = modular_grid(11)
        first = gr.make_probe_split(sums, seed=123)
        second = gr.make_probe_split(sums, seed=123)
        self.assertEqual(first, second)
        fit, evaluation = set(first["fit_indices"]), set(first["eval_indices"])
        self.assertFalse(fit & evaluation)
        self.assertEqual(fit | evaluation, set(range(len(sums))))
        for row in first["strata"]:
            self.assertEqual(row["fit"], row["total"] // 2)
            self.assertEqual(row["fit"] + row["eval"], row["total"])

        permutations = gr.make_row_permutations(len(sums), seed=456, n_nulls=3)
        self.assertEqual(permutations,
                         gr.make_row_permutations(len(sums), seed=456, n_nulls=3))
        for record in permutations:
            permutation = torch.tensor(record["permutation"])
            self.assertEqual(permutation.sort().values.tolist(), list(range(len(sums))))
        # A row shuffle breaks the within-class codebook identity: equal original
        # sums do not all map to one common permuted sum.
        permuted_sums = sums[torch.tensor(permutations[0]["permutation"])]
        self.assertTrue(any(permuted_sums[sums == label].unique().numel() > 1
                            for label in sums.unique()))

    def test_true_fourier_feature_beats_row_permutation_nulls(self):
        pairs, sums = modular_grid(11)
        split = gr.make_probe_split(sums, seed=10)
        features = torch.cat((fourier_features(sums, 11, 3),
                              pairs.double() / 11), dim=1)
        result = gr.evaluate_fourier_probes(
            features, sums, split["fit_indices"], split["eval_indices"],
            p=11, ridge=1e-3, top_k=2, null_seed=20, n_nulls=4,
        )
        self.assertEqual(len(result["observed"]["per_frequency"]), 5)
        self.assertIn(3, result["observed"]["selected_frequencies"])
        score3 = result["observed"]["per_frequency"][2]
        self.assertGreater(score3["fit_r2"], 0.999)
        self.assertGreater(score3["eval_r2"], 0.999)
        self.assertGreater(result["observed"]["selected_eval_mean_r2"],
                           result["null"]["selected_eval_mean_r2_max"] + 0.5)
        self.assertTrue(all(len(run["per_frequency"]) == 5
                            for run in result["null"]["runs"]))

    def test_memorized_independent_labels_do_not_generalize(self):
        _, true_sums = modular_grid(11)
        split = gr.make_probe_split(true_sums, seed=30)
        fit = torch.tensor(split["fit_indices"])
        evaluation = torch.tensor(split["eval_indices"])
        random_labels = true_sums[torch.randperm(len(true_sums),
                                                 generator=torch.Generator().manual_seed(31))]
        features = torch.zeros(len(true_sums), len(fit), dtype=torch.float64)
        features[fit, torch.arange(len(fit))] = 1.0
        result = gr.evaluate_fourier_probes(
            features, random_labels, fit, evaluation, p=11, ridge=1e-6,
            top_k=2, null_seed=32, n_nulls=20,
        )
        self.assertGreater(max(row["fit_r2"]
                               for row in result["observed"]["per_frequency"]), 0.99)
        self.assertLess(result["observed"]["selected_eval_mean_r2"], 0.2)
        # Descriptive construct check only: this fixed independent-label result
        # lies inside its finite row-permutation envelope, not a significance test.
        self.assertLessEqual(result["observed"]["selected_eval_mean_r2"],
                             result["null"]["selected_eval_mean_r2_max"] + 1e-9)

    def test_eval_labels_cannot_change_fit_ranking(self):
        pairs, sums = modular_grid(11)
        split = gr.make_probe_split(sums, seed=40)
        features = torch.cat((fourier_features(sums, 11, 2), pairs.double()), dim=1)
        permutations = gr.make_row_permutations(len(sums), seed=41, n_nulls=2)
        common = dict(fit_indices=split["fit_indices"],
                      eval_indices=split["eval_indices"], p=11, top_k=2,
                      null_seed=41, n_nulls=2, null_permutations=permutations)
        original = gr.evaluate_fourier_probes(features, sums, **common)
        changed = sums.clone()
        evaluation = torch.tensor(split["eval_indices"])
        changed[evaluation] = (changed[evaluation] + 1) % 11
        altered = gr.evaluate_fourier_probes(features, changed, **common)
        self.assertEqual(original["observed"]["selected_frequencies"],
                         altered["observed"]["selected_frequencies"])
        self.assertEqual([row["fit_r2"] for row in original["observed"]["per_frequency"]],
                         [row["fit_r2"] for row in altered["observed"]["per_frequency"]])
        self.assertEqual(original["fit_transform"], altered["fit_transform"])

        perturbed_features = features.clone()
        perturbed_features[evaluation] = perturbed_features[evaluation] * 13 + 101
        feature_altered = gr.evaluate_fourier_probes(perturbed_features, sums, **common)
        self.assertEqual(original["fit_transform"], feature_altered["fit_transform"])
        self.assertEqual(original["observed"]["selected_frequencies"],
                         feature_altered["observed"]["selected_frequencies"])
        self.assertEqual([row["fit_r2"] for row in original["observed"]["per_frequency"]],
                         [row["fit_r2"] for row in
                          feature_altered["observed"]["per_frequency"]])

        with self.assertRaisesRegex(ValueError, "disjoint and cover"):
            gr.evaluate_fourier_probes(
                features, sums, list(range(len(sums))), list(range(len(sums))),
                p=11, top_k=2, n_nulls=1,
            )

    def test_global_scale_offset_invariance_and_degenerate_features(self):
        pairs, sums = modular_grid(11)
        split = gr.make_probe_split(sums, seed=50)
        features = torch.cat((fourier_features(sums, 11, 4), pairs.double()), dim=1)
        kwargs = dict(fit_indices=split["fit_indices"],
                      eval_indices=split["eval_indices"], p=11, top_k=2,
                      null_seed=51, n_nulls=2)
        baseline = gr.evaluate_fourier_probes(features, sums, **kwargs)
        shifted = gr.evaluate_fourier_probes(features * -3.25 +
                                              torch.tensor([2.0, -7.0, 4.0, 9.0]),
                                              sums, **kwargs)
        for left, right in zip(baseline["observed"]["per_frequency"],
                               shifted["observed"]["per_frequency"]):
            self.assertAlmostEqual(left["fit_r2"], right["fit_r2"], places=12)
            self.assertAlmostEqual(left["eval_r2"], right["eval_r2"], places=12)
        for baseline_null, shifted_null in zip(baseline["null"]["runs"],
                                               shifted["null"]["runs"]):
            self.assertEqual(baseline_null["selected_frequencies"],
                             shifted_null["selected_frequencies"])
            for left, right in zip(baseline_null["per_frequency"],
                                   shifted_null["per_frequency"]):
                self.assertAlmostEqual(left["fit_r2"], right["fit_r2"], places=12)
                self.assertAlmostEqual(left["eval_r2"], right["eval_r2"], places=12)
        zeros = gr.evaluate_fourier_probes(torch.zeros(len(sums), 3), sums, **kwargs)
        self.assertTrue(zeros["fit_transform"]["degenerate_feature_scale"])
        self.assertTrue(all(abs(row["fit_r2"]) < 1e-12
                            for row in zeros["observed"]["per_frequency"]))
        assert_finite_json(self, zeros)
        json.dumps(zeros, allow_nan=False)

    def test_matches_reference_normal_equations_with_intercept(self):
        pairs, sums = modular_grid(7)
        split = gr.make_probe_split(sums, seed=55)
        fit = torch.tensor(split["fit_indices"])
        evaluation = torch.tensor(split["eval_indices"])
        features = torch.stack((pairs[:, 0].double(), pairs[:, 1].double(),
                                (pairs[:, 0] * pairs[:, 1]).double()), dim=1)
        ridge = 0.003
        result = gr.evaluate_fourier_probes(
            features, sums, fit, evaluation, p=7, ridge=ridge, top_k=1,
            null_seed=56, n_nulls=1,
        )

        mean = features[fit].mean(0)
        rms = ((features[fit] - mean).square().mean()).sqrt()
        design_fit = torch.cat((torch.ones(len(fit), 1),
                                (features[fit] - mean) / rms), dim=1)
        design_eval = torch.cat((torch.ones(len(evaluation), 1),
                                 (features[evaluation] - mean) / rms), dim=1)
        angle = (2 * math.pi / 7) * sums.double()
        targets = torch.stack((angle.cos(), angle.sin()), dim=1)
        target_mean = targets[fit].mean(0)
        penalty = torch.diag(torch.tensor(
            [0.0, ridge, ridge, ridge], dtype=torch.float64))
        coefficient = torch.linalg.solve(
            design_fit.T @ design_fit / len(fit) + penalty,
            design_fit.T @ (targets[fit] - target_mean) / len(fit),
        )
        predicted_fit = design_fit @ coefficient + target_mean
        predicted_eval = design_eval @ coefficient + target_mean
        reference_fit = 1 - ((targets[fit] - predicted_fit) ** 2).sum() / (
            (targets[fit] - target_mean).square().sum())
        reference_eval = 1 - ((targets[evaluation] - predicted_eval) ** 2).sum() / (
            (targets[evaluation] - target_mean).square().sum())
        measured = result["observed"]["per_frequency"][0]
        self.assertAlmostEqual(measured["fit_r2"], float(reference_fit), delta=1e-10)
        self.assertAlmostEqual(measured["eval_r2"], float(reference_eval), delta=1e-10)

    def test_hook_extraction_matches_forward_and_preserves_model(self):
        torch.manual_seed(60)
        model = GrokkingTransformer(p=11, d_model=8, n_heads=2, d_mlp=16)
        model.train()
        model.W_Q.eval()  # heterogeneous modes must be restored exactly
        pairs, _ = modular_grid(11)
        pairs = pairs[:13]
        for index, parameter in enumerate(model.parameters()):
            parameter.grad = torch.full_like(parameter, (index + 1) / 1000)
        state_before = copy.deepcopy(model.state_dict())
        grads_before = [parameter.grad.clone() for parameter in model.parameters()]
        modes_before = [module.training for module in model.modules()]

        for module in model.modules():
            module.training = False
        with torch.no_grad():
            expected_logits = model(pairs).detach().clone()
            embedded = model.tok_embed(pairs) + model.pos_embed(torch.arange(2))
            expected_pre = embedded.reshape(len(pairs), -1)
        for module, mode in zip(model.modules(), modes_before):
            module.training = mode

        extracted = gr.extract_activations(model, pairs, batch_size=len(pairs))
        self.assertTrue(torch.equal(extracted["logits"], expected_logits))
        self.assertTrue(torch.equal(extracted["pre_attention"], expected_pre))
        with torch.no_grad():
            reconstructed_logits = model.unembed(extracted["final_hidden"])
        self.assertTrue(torch.equal(reconstructed_logits, extracted["logits"]))
        self.assertEqual(extracted["final_hidden"].shape, (len(pairs), 8))
        self.assertEqual(extracted["pre_attention"].shape, (len(pairs), 16))

        chunked = gr.extract_activations(model, pairs, batch_size=4)
        chunked_direct = []
        for start in range(0, len(pairs), 4):
            for module in model.modules():
                module.training = False
            with torch.no_grad():
                chunked_direct.append(model(pairs[start:start + 4]).detach().clone())
            for module, mode in zip(model.modules(), modes_before):
                module.training = mode
        self.assertTrue(torch.equal(chunked["logits"], torch.cat(chunked_direct)))
        torch.testing.assert_close(chunked["logits"], expected_logits,
                                   rtol=1e-6, atol=1e-7)
        self.assertEqual(modes_before, [module.training for module in model.modules()])
        for key, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, state_before[key]))
        for parameter, gradient in zip(model.parameters(), grads_before):
            self.assertTrue(torch.equal(parameter.grad, gradient))


if __name__ == "__main__":
    unittest.main()
