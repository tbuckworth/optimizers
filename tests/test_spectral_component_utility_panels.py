"""Pure fabricated metadata fixtures, no original plan, images or model."""

import copy
import importlib
import unittest
from unittest.mock import patch

import numpy as np

from experiments import spectral_component_utility_panels as panels


def fabricated_roles():
    # Nontrivial deterministic order, not a real train/reporting split.
    ids = np.arange(59999, -1, -1, dtype=np.int64)
    result = {}
    for role, indices in (("train", ids[:50000]), ("validation", ids[50000:55000]),
                          ("reporting", ids[55000:])):
        result[role + "_ids"] = indices.copy()
        result[role + "_labels"] = (indices % 10).copy()
    result["assigned_labels"] = (result["train_labels"] + 3) % 10
    return result


class PanelTests(unittest.TestCase):
    def test_import_does_not_draw_or_read(self):
        with patch.object(np, "load", side_effect=AssertionError("file access")), \
                patch.object(np.random, "Generator", side_effect=AssertionError("RNG on import")):
            importlib.reload(panels)

    def test_literal_streams_and_all_mappings(self):
        roles = fabricated_roles()
        for seed in panels.SEEDS:
            actual = panels.select_panels(roles, seed)
            self.assertEqual(tuple(actual), panels.OUTPUT_KEYS)
            generator = np.random.Generator(np.random.PCG64(np.random.SeedSequence([40, seed])))
            order = generator.permutation(np.arange(50000, dtype=np.int64))
            action, evaluation = order[:128].reshape(2, 64), order[128:384]
            np.testing.assert_array_equal(actual["action_positions"], action)
            np.testing.assert_array_equal(actual["evaluation_positions"], evaluation)
            for prefix, positions in (("action", action), ("evaluation", evaluation)):
                for suffix, source in (("ids", "train_ids"), ("true", "train_labels"),
                                       ("assigned", "assigned_labels")):
                    np.testing.assert_array_equal(actual[prefix + "_" + suffix], roles[source][positions])
            independent = np.random.Generator(np.random.PCG64(np.random.SeedSequence([41, seed])))
            np.testing.assert_array_equal(actual["action_shifts"],
                independent.integers(-2, 3, size=(2, 64, 2), dtype=np.int8))
            np.testing.assert_array_equal(actual["evaluation_wrong"],
                roles["assigned_labels"][evaluation] != roles["train_labels"][evaluation])
            for prefix, count in (("reporting", 128), ("baseline", 500)):
                np.testing.assert_array_equal(actual[prefix + "_ids"], roles["reporting_ids"][:count])
                np.testing.assert_array_equal(actual[prefix + "_true"], roles["reporting_labels"][:count])
            self.assertEqual(len(np.unique(np.concatenate((action.ravel(), evaluation)))), 384)
            self.assertEqual(len(set(actual["reporting_ids"]) & set(actual["action_ids"].ravel())), 0)
            self.assertEqual(len(set(actual["reporting_ids"]) & set(actual["evaluation_ids"])), 0)

    def test_exact_view_order_and_freshness(self):
        first, second = panels.view_shifts(), panels.view_shifts()
        self.assertEqual(first.shape, (25, 2))
        self.assertEqual(first.dtype, np.int8)
        self.assertEqual(first.tolist(), [[dy, dx] for dy in (-2, -1, 0, 1, 2)
                                        for dx in (-2, -1, 0, 1, 2)])
        self.assertEqual(first[panels.ORIGINAL_VIEW_INDEX].tolist(), [0, 0])
        first.fill(1)
        self.assertNotEqual(first.tolist(), second.tolist())

    def test_immutable_inputs_independent_outputs_and_global_rng(self):
        roles = fabricated_roles()
        before = copy.deepcopy(roles)
        for value in roles.values():
            value.flags.writeable = False
        rng_before = copy.deepcopy(np.random.get_state())
        output = panels.select_panels(roles, panels.SEEDS[0])
        rng_after = np.random.get_state()
        self.assertEqual(rng_before[0], rng_after[0])
        np.testing.assert_array_equal(rng_before[1], rng_after[1])
        self.assertEqual(rng_before[2:], rng_after[2:])
        for value in output.values():
            self.assertTrue(value.flags.c_contiguous and value.flags.writeable)
            value.fill(0)
        for key in roles:
            np.testing.assert_array_equal(roles[key], before[key])

    def test_labels_do_not_choose_positions_or_views(self):
        roles = fabricated_roles()
        changed = copy.deepcopy(roles)
        for key in ("train_labels", "validation_labels", "reporting_labels", "assigned_labels"):
            changed[key].fill(0)
        first = panels.select_panels(roles, panels.SEEDS[0])
        second = panels.select_panels(changed, panels.SEEDS[0])
        for key in ("action_positions", "action_ids", "action_shifts", "evaluation_positions",
                    "evaluation_ids", "reporting_ids", "baseline_ids", "view_shifts"):
            np.testing.assert_array_equal(first[key], second[key])
        self.assertFalse(second["evaluation_wrong"].any())

    def test_invalid_seeds_roles_ids_labels_rejected(self):
        for seed in (True, 0, 202609174, float(panels.SEEDS[0]), np.int64(panels.SEEDS[0])):
            with self.assertRaises(ValueError):
                panels.select_panels(fabricated_roles(), seed)
        mutations = (
            lambda r: r.pop("validation_labels"),
            lambda r: r.update(extra=np.zeros(1)),
            lambda r: r.update(train_ids=r["train_ids"].astype(np.int32)),
            lambda r: r.update(reporting_ids=r["reporting_ids"][:-1]),
            lambda r: r["reporting_ids"].__setitem__(0, r["train_ids"][0]),
            lambda r: r["validation_ids"].__setitem__(0, 60000),
            lambda r: r["train_labels"].__setitem__(0, -1),
            lambda r: r["assigned_labels"].__setitem__(0, 10),
        )
        for mutate in mutations:
            roles = fabricated_roles()
            mutate(roles)
            with self.assertRaises(ValueError):
                panels.select_panels(roles, panels.SEEDS[0])


if __name__ == "__main__":
    unittest.main()
