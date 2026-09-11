"""Pure tiny dense references; not part of the large-P memory measurements."""
import unittest
import numpy as np
from experiments.anchor_graph_pilot import (
    AnchorGraph, anchor_graph, cluster_graph, cluster_mean_action,
    moment_error, rank_one_update, recovery_metrics, truncate_factor,
)


class AnchorGraphTests(unittest.TestCase):
    def test_rank_one_and_truncation_match_dense(self):
        rng = np.random.default_rng(5)
        f = rng.normal(size=(31, 4))
        h = rng.normal(size=31)
        target = .9 * (f @ f.T) + .1*np.outer(h, h)
        values, vectors = np.linalg.eigh(target)
        expected = (vectors[:, -3:] * values[-3:]) @ vectors[:, -3:].T
        updated = rank_one_update(f, h, .9, 3)
        np.testing.assert_allclose(updated @ updated.T, expected, atol=1e-12)

    def test_first_observation_and_untruncated_history(self):
        rng = np.random.default_rng(6)
        f, exact = np.empty((17, 0)), np.zeros((17, 17))
        for _ in range(8):
            h = rng.normal(size=17)
            exact = .8*exact + .2*np.outer(h, h)
            f = rank_one_update(f, h, .8, 17)
            np.testing.assert_allclose(f @ f.T, exact, atol=1e-12)

    def test_graph_and_eigenvectors_dense(self):
        f = np.random.default_rng(7).normal(size=(61, 5))
        graph = anchor_graph(f, anchors=10)
        a = (graph.z / graph.mass) @ graph.z.T
        np.testing.assert_allclose(a, a.T, atol=1e-14)
        np.testing.assert_allclose(a.sum(1), 1., atol=1e-13)
        self.assertGreaterEqual(a.min(), 0.)
        self.assertGreaterEqual(np.linalg.eigvalsh(a).min(), -1e-12)
        self.assertTrue(np.all(np.diag(a) > 0))
        x = np.random.default_rng(8).normal(size=61)
        np.testing.assert_allclose(graph.matvec(x), a @ x, atol=1e-13)
        values, u = graph.eigenvectors(7)
        np.testing.assert_allclose(a @ u, u * values, atol=1e-12)
        np.testing.assert_allclose(u.T @ u, np.eye(len(values)), atol=1e-11)
        np.testing.assert_allclose(values, np.linalg.eigvalsh(a)[::-1][:7], atol=1e-12)
        self.assertGreaterEqual(np.linalg.eigvalsh(np.eye(61)-a).min(), -1e-12)

    def test_zero_rows_and_unused_anchors(self):
        f = np.array([[1., 0], [0, 0], [0, 1], [0, 0]])
        graph = anchor_graph(f, anchors=8)
        np.testing.assert_array_equal(graph.active, [0, 2])
        np.testing.assert_allclose(graph.matvec(np.ones(4)), [1, 0, 1, 0])
        labels, _, _ = cluster_graph(graph, 2)
        np.testing.assert_array_equal(labels[[1, 3]], [-1, -1])
        manual = AnchorGraph.from_weights(2, [0, 1], [0, 1], [[1, 0], [1, 0]])
        self.assertEqual(manual.r.shape, (2, 1))
        np.testing.assert_allclose(manual.matvec(np.ones(2)), 1.)
        empty = anchor_graph(np.zeros((5, 2)))
        np.testing.assert_array_equal(empty.matvec(np.ones(5)), 0.)
        np.testing.assert_array_equal(cluster_graph(empty, 2)[0], -1)

    def test_rotation_invariance_with_recomputed_anchor_ids(self):
        rng = np.random.default_rng(9)
        f = rng.normal(size=(43, 6))
        q, _ = np.linalg.qr(rng.normal(size=(6, 6)))
        first = anchor_graph(f, anchors=9)
        rotated = anchor_graph(f @ q, anchor_indices=first.anchor_indices)
        x = rng.normal(size=43)
        np.testing.assert_allclose(first.matvec(x), rotated.matvec(x), atol=1e-13)

    def test_group_action_dense_projector_and_isolate(self):
        labels = np.array([0, 1, 0, -1, 1, 2, 2])
        x = np.arange(7, dtype=float)
        b = np.zeros((7, 4))
        for group in range(3):
            b[labels == group, group] = 1 / np.sqrt(np.sum(labels == group))
        b[3, 3] = 1
        result = cluster_mean_action(x, labels)
        np.testing.assert_allclose(result, b @ b.T @ x, atol=1e-14)
        np.testing.assert_allclose(cluster_mean_action(result, labels), result)
        np.testing.assert_allclose(result.sum(), x.sum())

    def test_identical_and_extremely_narrow_profiles(self):
        same = anchor_graph(np.ones((15, 3)), anchors=6)
        values, _ = same.eigenvectors(5)
        self.assertEqual(len(values), 1)
        labels, _, _ = cluster_graph(same, 5)
        self.assertEqual(np.unique(labels).size, 1)
        narrow = anchor_graph(np.eye(5), anchors=5, sigma=1e-200)
        np.testing.assert_allclose(narrow.matvec(np.ones(5)), 1.)

    def test_clipping_can_increase_rank(self):
        theta = np.arange(8) * 2*np.pi/8
        f = np.column_stack((np.cos(theta), np.sin(theta)))
        cosine = f @ f.T
        self.assertEqual(np.linalg.matrix_rank(cosine, tol=1e-10), 2)
        self.assertGreater(np.linalg.matrix_rank(np.maximum(cosine, 0), tol=1e-10), 2)

    def test_small_gram_error_matches_dense(self):
        rng = np.random.default_rng(10)
        exact = rng.normal(size=(27, 8))
        low = truncate_factor(exact, 3)
        target = exact @ exact.T
        expected = np.linalg.norm(low @ low.T-target) / np.linalg.norm(target)
        self.assertAlmostEqual(moment_error(low, exact), expected, places=12)

    def test_metrics_known_partitions(self):
        truth = np.array([0, 0, 1, 1, 2, 2])
        result = recovery_metrics(truth, np.array([2, 2, 0, 0, 1, 1]))
        self.assertEqual(result["ari"], 1.)
        self.assertEqual(result["group_best_f1"], [1., 1., 1.])
        self.assertEqual(recovery_metrics(truth, np.zeros(6, dtype=int))["ari"], 0.)

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            anchor_graph(np.full((3, 2), np.nan))
        with self.assertRaises(ValueError):
            anchor_graph(np.ones((3, 2)), sigma=0)
        with self.assertRaises(ValueError):
            cluster_mean_action(np.ones(3), np.array([0, 0, 999999]))
        with self.assertRaises(ValueError):
            AnchorGraph.from_weights(2, [0, 1], [0], [[0], [1]])


if __name__ == "__main__":
    unittest.main()
