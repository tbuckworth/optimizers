"""Tiny synthetic fixtures only; no real MNIST data or trained model."""
import json
import math
import unittest
import copy

import numpy as np

import check_saved as audit


class TestArithmetic(unittest.TestCase):
    def test_uniform_logits_and_empty_mask(self):
        result = audit.logits_stats(np.zeros((3, 10)), np.array([0, 1, 2]))
        self.assertEqual((result["count"], result["correct"]), (3, 1))
        self.assertAlmostEqual(result["ce_sum"], 3 * math.log(10))
        self.assertEqual(audit.logits_stats(np.empty((0, 10)), np.array([], dtype=int)),
                         {"count": 0, "correct": 0, "ce_sum": 0.0})

    def test_large_logits_and_constant_shift(self):
        logits = np.full((2, 10), -1000.0)
        logits[0, 3] = 1000
        logits[1, 4] = 1000
        target = np.array([3, 5])
        result = audit.logits_stats(logits, target)
        self.assertEqual(result["correct"], 1)
        self.assertEqual(result["ce_sum"], 2000)
        self.assertEqual(result, audit.logits_stats(logits + 1234, target))

    def test_counts_and_ce_tampering_rejected(self):
        with self.assertRaises(ValueError):
            audit.close(0.5, 0.6, "accuracy")
        with self.assertRaises(ValueError):
            audit.close(float("nan"), 0.6, "ce")
        with self.assertRaises(ValueError):
            audit.logits_stats(np.zeros((2, 10)), np.array([0, 10]))

    def test_three_seed_summary(self):
        result = audit.sample_summary([-1, 0, 1])
        self.assertEqual(result["mean"], 0)
        self.assertAlmostEqual(result["sample_se"], 1 / math.sqrt(3))
        self.assertEqual([result[x] for x in ("negative_count", "zero_count", "positive_count")], [1, 1, 1])

    def test_pairing_and_json_reject_duplicates(self):
        audit.same_pair_hashes([{"init": "a"}, {"init": "a"}], ["init"])
        with self.assertRaises(ValueError):
            audit.same_pair_hashes([{"init": "a"}, {"init": "b"}], ["init"])
        with self.assertRaises(ValueError):
            json.loads('{"seed": 1, "seed": 2}', object_pairs_hook=audit.unique_object)
        with self.assertRaises(ValueError):
            audit.finite_tree({"steps": [0, float("inf")]})

    def test_wrong_target_mask_and_null_clean_semantics(self):
        train = np.zeros((3, 10))
        train[np.arange(3), [0, 2, 2]] = 2
        clean, fixed = np.array([0, 1, 2]), np.array([0, 2, 3])
        stats, metrics = audit.statistics_and_metrics(train, train, clean, fixed, clean)
        self.assertEqual(stats['train_corrupted_count'], 2)
        self.assertEqual(stats['train_corrupted_correct'], 1)
        self.assertEqual(metrics['train_corrupted_accuracy'], .5)
        self.assertEqual(metrics['train_clean_accuracy'], 2/3)
        self.assertEqual(metrics['train_fixed_accuracy'], 2/3)
        _, clean_metrics = audit.statistics_and_metrics(train, train, clean, clean, clean)
        self.assertIsNone(clean_metrics['train_corrupted_accuracy'])
        altered = dict(stats, train_corrupted_count=3)
        with self.assertRaises(ValueError):
            audit.compare_fields(altered, stats, 'wrong replacement denominator')

    def test_all_frozen_endpoint_contrasts(self):
        curves = []
        for seed_index, seed in enumerate(audit.SEEDS):
            for condition in audit.CONDITIONS:
                for arm_index, arm in enumerate(audit.ARMS):
                    metrics = {metric: .1 * seed_index + .2 * arm_index for metric in audit.METRICS}
                    if condition == 'clean':
                        metrics['train_corrupted_accuracy'] = None
                    curves.append({'seed': seed, 'condition': condition, 'arm': arm,
                                   'curve': [{'step': 2000, 'metrics': metrics}],
                                   'warmup_plus_branch_seconds': 10 + arm_index})
        result = audit.summarize(curves)
        self.assertEqual(len(result['per_arm']), 8)
        self.assertEqual(len(result['paired_differences']), 10)
        difference = result['paired_differences']['clean/mixed32_minus_hard32']['heldout_clean_accuracy']
        self.assertAlmostEqual(difference['mean'], .4)
        self.assertEqual(difference['positive_count'], 3)
        self.assertIsNone(result['paired_differences']['clean/mixed32_minus_hard32']['train_corrupted_accuracy'])
        with self.assertRaises(ValueError):
            audit.summarize(curves[:-1])

    def test_action_roster_algebra_and_contraction(self):
        rows = [{'step': step, 'training_batch_ce': 1., 'raw_squared_norm': 4.,
                 'applied_squared_norm': 1., 'applied_to_raw_norm_ratio': .5}
                for step in range(101, 2001)]
        audit.check_actions(rows, 'mixed32')
        broken = copy.deepcopy(rows)
        broken[100]['applied_to_raw_norm_ratio'] = .6
        with self.assertRaises(ValueError):
            audit.check_actions(broken, 'mixed32')
        with self.assertRaises(ValueError):
            audit.check_actions(rows[1:], 'cluster32')
        with self.assertRaises(ValueError):
            audit.check_actions(rows, 'adamw')

    def test_cluster_rank_and_isolate_arithmetic(self):
        rows = []
        for index, step in enumerate(range(101, 2001, 100)):
            rows.append({'step': step, 'cluster_sizes': [25440, 25440], 'isolates': 10,
                         'nonempty_clusters': 2, 'effective_projector_rank': 12,
                         'isolate_fraction': 10/50890, 'largest_cluster_fraction': 25440/50890,
                         'factor_rank': 32, 'actual_anchors': 64, 'kmeans_iterations': 3,
                         'retained_array_bytes': 100, 'graph_eigenvalues': [1., .3],
                         'refresh_seconds': .1, 'labels_sha256': 'a' * 64,
                         'stability_ari_common_assigned': None if index == 0 else 1.,
                         'common_assigned_count': 0 if index == 0 else 50880,
                         'isolate_status_changed_fraction': None if index == 0 else 0.})
        audit.check_clusters(rows, 'cluster32')
        rows[-1]['effective_projector_rank'] = 2
        with self.assertRaises(ValueError):
            audit.check_clusters(rows, 'cluster32')

    def test_membership_transition_ignores_isolates_not_as_one_cluster(self):
        before = np.array([-1, -1, 0, 0, 1, 1])
        after = np.array([-1, 2, 7, 7, 8, 8])
        count, changed, ari = audit.label_transition(before, after)
        self.assertEqual(count, 4)
        self.assertEqual(changed, 1/6)
        self.assertEqual(ari, 1)
        count, changed, ari = audit.label_transition(np.array([-1, -1, 0]), np.array([-1, 0, -1]))
        self.assertEqual(count, 0)
        self.assertIsNone(ari)
        self.assertEqual(changed, 2/3)
        _, _, ari = audit.label_transition(np.array([0, 0, 1, 1]), np.array([0, 1, 0, 1]))
        self.assertAlmostEqual(ari, -.5)


if __name__ == "__main__":
    unittest.main()
