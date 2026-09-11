"""Parent checks exercise the independent auditor's actual pairing function."""
import unittest
import audit_completed_results as a


def fixture():
    names = [a.PRIMARY_METRIC, "geometry"] + [f"synthetic.{i}" for i in range(169)]
    return [{"seed": seed, "replacement_probability": noise, "arm": arm,
             "metrics": {name: float(seed + a.ARMS.index(arm)) for name in names},
             "counts": {"geometry": {"null_steps": []}}}
            for seed, noise, arm in sorted(a.EXPECTED_CELLS)]


def target(groups):
    return next(row for row in groups["paired_contrasts"] if row["replacement_probability"] == 0
                and row["treatment"] == "lagged32" and row["control"] == "current32"
                and row["metric"] == "geometry")


class ParentPairingTests(unittest.TestCase):
    def test_different_masks_suppress_pair_and_three_seed_statistics(self):
        rows = fixture()
        row = next(row for row in rows if (row["seed"], row["replacement_probability"], row["arm"]) == (6, 0, "lagged32"))
        row["counts"]["geometry"]["null_steps"] = [101]
        result = target(a.aggregate_groups(rows))
        self.assertEqual(result["unavailable_seeds"], [6])
        self.assertIsNone(result["mean"])
        self.assertEqual(result["paired_differences"][0]["unavailable_reason"], "different_null_step_masks")

    def test_matched_masks_allow_seed_pair(self):
        rows = fixture()
        for row in rows:
            row["counts"]["geometry"]["null_steps"] = [101]
        result = target(a.aggregate_groups(rows))
        self.assertEqual(result["unavailable_seeds"], [])
        self.assertEqual(result["mean"], 1.)

    def test_missing_metric_does_not_become_available_case_mean(self):
        rows = fixture()
        row = next(row for row in rows if (row["seed"], row["replacement_probability"], row["arm"]) == (7, 0, "current32"))
        row["metrics"]["geometry"] = None
        result = target(a.aggregate_groups(rows))
        self.assertEqual(result["unavailable_seeds"], [7])
        self.assertIsNone(result["sample_sd_descriptive"])
        self.assertEqual(result["paired_differences"][1]["unavailable_reason"], "missing_seed_metric")


if __name__ == "__main__":
    unittest.main()
