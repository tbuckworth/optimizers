"""Tiny synthetic-only fixtures; never read MNIST or invoke acquisition/main."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from experiments import anchor_graph_mnist as run


class AnchorGraphMNISTTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_plan_pairing_and_fixed_corruption(self):
        first = run.make_plan(7, 30, 10, 10, 4, 3)
        second = run.make_plan(7, 30, 10, 10, 4, 3)
        for key in first:
            np.testing.assert_array_equal(first[key], second[key])
        self.assertEqual(len(set(first["train_indices"]) & set(first["heldout_indices"])), 0)
        clean = np.arange(10)
        np.testing.assert_array_equal(run.fixed_targets(clean, first, "clean"), clean)
        fixed = run.fixed_targets(clean, first, "fixed_uniform_0p9")
        np.testing.assert_array_equal(fixed, np.where(first["replacement_mask"], first["replacement_digits"], clean))
        with self.assertRaises(RuntimeError):
            run.fixed_targets(clean, first, "unknown")

    def test_covariance_factor_uses_singular_values_not_square_root(self):
        tracker = SimpleNamespace(V=torch.tensor([[1., 0.], [0., 1.], [0., 0.]]),
                                  S=torch.tensor([3., 2.], dtype=torch.float64), n_params=3)
        factor = run.covariance_factor(tracker)
        np.testing.assert_array_equal(factor @ factor.T, np.diag([9., 4., 0.]))
        tracker.V = None
        self.assertEqual(run.covariance_factor(tracker).shape, (3, 0))

    def test_cluster_projector_and_fixed_residual_mix(self):
        labels = np.array([0, 0, 1, 1, -1])
        matrix = np.column_stack([run.graph_core.cluster_mean_action(x, labels) for x in np.eye(5)])
        np.testing.assert_allclose(matrix.T, matrix)
        np.testing.assert_allclose(matrix @ matrix, matrix)
        x = np.array([2., -2., 3., 5., -7.])
        y = matrix @ x
        self.assertLessEqual(np.linalg.norm(y), np.linalg.norm(x))
        self.assertEqual(y[-1], x[-1])
        mixed = .5 * (x + y)
        np.testing.assert_allclose((np.eye(5) - matrix) @ mixed, .5 * (x - y))

    def test_refresh_stability_ignores_cluster_names_and_tracks_isolates(self):
        old = np.array([0, 0, 1, 1, -1])
        labels = np.array([8, 8, 3, 3, -1])
        stats = run.cluster_diagnostics(labels, old)
        self.assertEqual(stats["effective_projector_rank"], 3)
        self.assertEqual(stats["stability_ari_common_assigned"], 1.)
        self.assertEqual(stats["isolate_fraction"], .2)
        self.assertEqual(stats["isolate_status_changed_fraction"], 0.)

    def test_sufficient_statistics_actual_changed_mask_and_null(self):
        logits = np.array([[3., 0.], [0., 2.], [2., 0.]], np.float32)
        clean, fixed = np.array([0, 0, 1]), np.array([0, 1, 0])
        stats = run.sufficient_statistics(logits, clean, fixed, logits, clean)
        self.assertEqual(stats["train_corrupted_count"], 2)
        self.assertEqual(stats["train_corrupted_correct"], 2)
        metrics = run.metrics_from_statistics(stats)
        self.assertEqual(metrics["train_corrupted_accuracy"], 1.)
        self.assertEqual(metrics["heldout_clean_accuracy"], 1/3)
        stats = run.sufficient_statistics(logits, clean, clean, logits, clean)
        self.assertIsNone(run.metrics_from_statistics(stats)["train_corrupted_accuracy"])
        logits[0, 0] = np.nan
        with self.assertRaises(RuntimeError):
            run.sufficient_statistics(logits, clean, clean, logits, clean)

    def test_hard_step_matches_existing_canonical_and_common_adam_restore(self):
        model = run.core.make_model(11, "cpu", input_dim=4, width=3, classes=2)
        optimizer = run.core.make_optimizer(model)
        tracker = run.core.make_tracker(model, optimizer)
        generator = torch.Generator().manual_seed(11)
        x, y = torch.rand((8, 4), generator=generator), torch.arange(8) % 2
        for step in range(1, 101):
            run.training_step(model, optimizer, tracker, x, y, "adamw", step)
        snapshot = run.core.snapshot(model, optimizer, tracker)
        first = run.core.restore(snapshot, "cpu")
        second = run.core.restore(snapshot, "cpu")
        self.assertEqual(run.core.tree_digest(run.core.snapshot(*first)), run.core.tree_digest(snapshot))
        run.training_step(*first, x, y, "hard32", 101)
        run.core.train_step(*second, x, y, "current32")
        self.assertTrue(run.core.equal_tree(first[0].state_dict()["0.weight"], second[0].state_dict()["0.weight"]))
        self.assertTrue(run.core.equal_tree(first[1].state_dict(), second[1].state_dict()))
        self.assertTrue(run.core.equal_tree(snapshot, run.core.snapshot(*run.core.restore(snapshot, "cpu"))))

    def test_cluster_refresh_schedule_and_current_factor(self):
        factor = torch.tensor([[1., 0.], [1., .01], [0., 1.], [.01, 1.], [0., 0.]])
        tracker = SimpleNamespace(V=factor, S=torch.ones(2, dtype=torch.float64), n_params=5)
        policy = run.ClusterAction(13)
        raw = torch.arange(5, dtype=torch.float32)
        for step in (101, 102, 200, 201):
            applied = policy.apply(tracker, raw, step, "mixed32")
            self.assertTrue(torch.isfinite(applied).all())
            self.assertEqual(applied[-1], raw[-1])
        self.assertEqual([row["step"] for row in policy.refreshes], [101, 201])
        self.assertEqual(np.stack(policy.label_history).shape, (2, 5))
        self.assertEqual(run.array_digest(policy.label_history[-1]), policy.refreshes[-1]["labels_sha256"])

    def test_first_gradient_and_post_observer_pairing_across_all_arms(self):
        model = run.core.make_model(29, "cpu", input_dim=4, width=3, classes=2)
        optimizer = run.core.make_optimizer(model)
        tracker = run.core.make_tracker(model, optimizer)
        x = torch.arange(24, dtype=torch.float32).reshape(6, 4) / 24
        y = torch.arange(6) % 2
        for step in range(1, 101):
            run.training_step(model, optimizer, tracker, x, y, "adamw", step)
        state = run.core.snapshot(model, optimizer, tracker)
        bindings = []
        for arm in run.ARMS:
            m, o, t = run.core.restore(state, "cpu")
            bindings.append(run.training_step(m, o, None if arm == "adamw" else t,
                                              x, y, arm, 101, run.ClusterAction(29)))
        self.assertEqual(len({row["raw_gradient_sha256"] for row in bindings}), 1)
        self.assertEqual(len({row["post_observe_tracker_sha256"] for row in bindings[1:]}), 1)

    def test_exclusive_capped_json_npz_and_tensor_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            runner = run.Run(Path(directory), "cpu")
            for kind, name, value in (("json", "tiny.json", {"a": 3}),
                                      ("npz", "tiny.npz", {"x": np.arange(5)}),
                                      ("tensor", "tiny.pt", {"x": torch.arange(5)})):
                receipt = runner.save(name, value, kind)
                self.assertEqual(receipt["sha256"], run.digest(Path(directory) / name))
                with self.assertRaises(FileExistsError):
                    runner.save(name, value, kind)
            with np.load(Path(directory) / "tiny.npz", allow_pickle=False) as data:
                np.testing.assert_array_equal(data["x"], np.arange(5))
            runner.used = run.MAX_BYTES
            with self.assertRaises(RuntimeError):
                runner.save("over-cap.json", {"a": 1})


if __name__ == "__main__":
    unittest.main()
