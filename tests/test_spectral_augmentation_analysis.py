"""Fabricated scalar fixtures, not observations or model runs."""
import copy
import unittest
from experiments import spectral_augmentation_analysis as a


def evaluation(step, accuracy=.4, excess=.1, changed=True):
    stats = {"accuracy": accuracy, "ce": 2. - accuracy}
    group = {key: dict(stats) for key in ("rare", "majority_macro", "balanced_total")}
    group["per_class"] = {str(d): dict(stats) for d in range(10)}
    return {"step": step, "heldout_unpatched": copy.deepcopy(group),
            "heldout_patched": copy.deepcopy(group), "train_true": copy.deepcopy(group),
            "train_assigned": dict(stats),
            "train_actually_changed": {"count": 2 if changed else 0,
                "accuracy": accuracy if changed else None, "ce": 2. - accuracy if changed else None},
            "cue": {g: {"unpatched_target0_rate": .05, "patched_target0_rate": .05 + excess,
                         "patch_excess": excess} for g in ("majority_nonzero", "rare", "all_nonzero")}}


def rows():
    return [{"seed": s, "cell": c, "policy": p, "augmentation": m,
             "endpoint": evaluation(2000, changed=c != "clean"),
             "warmup": evaluation(100, changed=c != "clean")}
            for s in a.SEEDS for c in a.CELLS for p in a.POLICIES for m in a.MODES]


class AnalysisTests(unittest.TestCase):
    def test_absolute_and_interaction_not_confused(self):
        data = rows()
        for row in data:
            if row["augmentation"] == "random":
                # Both improve, native by .2 and raw by .1.
                acc = .6 if row["policy"] == "native32" else .5
                row["endpoint"] = evaluation(2000, acc, changed=row["cell"] != "clean")
        result = a.summarize(data)
        key = "heldout_unpatched/rare/accuracy"
        self.assertAlmostEqual(result["augmentation_contrasts"]["shared/native32/random"][key]["mean"], .2)
        self.assertAlmostEqual(result["interactions"]["shared/random"][key]["mean"], .1)
        self.assertEqual(result["groups"]["clean/raw/none"]["train_actually_changed/ce"], None)
        self.assertAlmostEqual(result["location_contrasts"]["shared/native32"][key]["mean"], 0.)

    def test_favorable_relative_cue_change_can_be_absolute_harm(self):
        data = rows()
        for row in data:
            if row["cell"] == "shared" and row["augmentation"] == "random":
                excess = .2 if row["policy"] == "native32" else .3
                row["endpoint"] = evaluation(2000, excess=excess)
        result = a.summarize(data)["cue_association"]
        self.assertAlmostEqual(result["delta_Q"]["native32/random"]["mean"], .1)
        self.assertAlmostEqual(result["I_Q"]["random"]["mean"], -.1)
        self.assertAlmostEqual(result["Q"]["native32/random"]["mean"], .1)

    def test_roster_warmup_and_finite_validation(self):
        data = rows()
        for bad in (data[:-1], data + [data[0]]):
            with self.assertRaises(ValueError):
                a.summarize(bad)
        bad = copy.deepcopy(data)
        bad[0]["endpoint"]["step"] = 1900
        with self.assertRaises(ValueError):
            a.summarize(bad)
        bad = copy.deepcopy(data)
        bad[0]["warmup"]["train_assigned"]["accuracy"] = .3
        with self.assertRaises(ValueError):
            a.summarize(bad)
        for value in (float("nan"), float("inf"), True, "0.2"):
            bad = copy.deepcopy(data)
            bad[0]["endpoint"]["train_assigned"]["accuracy"] = value
            with self.assertRaises(ValueError):
                a.summarize(bad)

    def test_pairing_by_seed_not_input_order_and_no_mutation(self):
        data = rows()
        for row in data:
            if row["policy"] == "native32":
                row["endpoint"] = evaluation(2000, .5 + .1*a.SEEDS.index(row["seed"]),
                                             changed=row["cell"] != "clean")
        before = copy.deepcopy(data)
        result = a.summarize(list(reversed(data)))
        values = result["policy_contrasts"]["clean/none"]["heldout_unpatched/rare/accuracy"]
        for actual, expected in zip(values["values"], (.1, .2, .3)):
            self.assertAlmostEqual(actual, expected)
        self.assertEqual(data, before)


if __name__ == "__main__":
    unittest.main()
