import json
import unittest

import numpy as np

from experiments.grokking_symmetry import symmetry_diagnostics


def partition(p):
    ids = np.arange(p * p, dtype=np.int64)
    a, b = ids // p, ids % p
    train = ids[((3 * a + 5 * b) % 7) < 2]
    test = np.setdiff1d(ids, train)
    return train, test


def cosine_logits(p, class_offset=0, temperatures=None):
    ids = np.arange(p * p, dtype=np.int64)
    a, b = ids // p, ids % p
    classes = np.arange(p, dtype=np.float64)
    phase = classes[None, :] - a[:, None] - b[:, None] - class_offset
    logits = np.cos(2 * np.pi * phase / p)
    if temperatures is not None:
        logits = logits * np.asarray(temperatures, dtype=np.float64)[:, None]
    return logits


class GrokkingSymmetryTest(unittest.TestCase):
    P = 11
    DELTAS = (1, 2, 4)
    WRONG = 3

    def diagnose(self, logits, train=None, test=None):
        if train is None or test is None:
            train, test = partition(self.P)
        return symmetry_diagnostics(
            logits, train, test, p=self.P, deltas=self.DELTAS,
            wrong_shift_offset=self.WRONG,
        )

    def test_ideal_orientation_exchange_and_row_centering(self):
        train, test = partition(self.P)
        logits = cosine_logits(self.P)
        original = logits.copy()
        result = self.diagnose(logits, train, test)
        np.testing.assert_array_equal(logits, original)

        for axis in ("a", "b"):
            for delta in self.DELTAS:
                record = result["heldout_symmetry"][axis][str(delta)]
                self.assertGreater(record["correct"]["count"], 0)
                self.assertLess(record["correct"]["value"], 1e-28)
                self.assertGreater(record["wrong_shift"]["value"], 0.1)
                self.assertEqual(
                    record["correct"]["count"], record["wrong_shift"]["count"]
                )
        self.assertLess(result["exchange"]["correct"]["value"], 1e-28)
        self.assertGreater(result["exchange"]["wrong_shift"]["value"], 0.1)

        offsets = (np.arange(self.P * self.P) % 13 - 6.0)[:, None]
        shifted = self.diagnose(logits + offsets, train, test)
        for axis in ("a", "b"):
            for delta in self.DELTAS:
                first = result["heldout_symmetry"][axis][str(delta)]["correct"]
                second = shifted["heldout_symmetry"][axis][str(delta)]["correct"]
                self.assertAlmostEqual(first["numerator"], second["numerator"], places=25)
                self.assertAlmostEqual(first["denominator"], second["denominator"], places=11)
                self.assertAlmostEqual(first["value"], second["value"], places=25)

    def test_equivariance_is_neither_necessary_nor_sufficient_for_accuracy(self):
        ids = np.arange(self.P * self.P)
        a, b = ids // self.P, ids % self.P
        temperature = 1.0 + 0.07 * a + 0.03 * b
        variable = cosine_logits(self.P, temperatures=temperature)
        labels = (a + b) % self.P
        self.assertTrue(np.array_equal(variable.argmax(axis=1), labels))
        variable_result = self.diagnose(variable)
        self.assertGreater(
            variable_result["heldout_symmetry"]["a"]["1"]["correct"]["value"],
            1e-5,
        )

        wrong = cosine_logits(self.P, class_offset=2)
        self.assertFalse(np.any(wrong.argmax(axis=1) == labels))
        wrong_result = self.diagnose(wrong)
        self.assertLess(
            wrong_result["heldout_symmetry"]["a"]["1"]["correct"]["value"],
            1e-28,
        )

        uniform_result = self.diagnose(np.full((self.P * self.P, self.P), 4.25))
        metric = uniform_result["heldout_symmetry"]["a"]["1"]["correct"]
        self.assertFalse(metric["energy_defined"])
        self.assertIsNone(metric["value"])
        self.assertEqual(metric["numerator"], 0.0)
        self.assertEqual(metric["denominator"], 0.0)

    def test_train_specific_perturbation_and_temperature_confound(self):
        train, test = partition(self.P)
        base = cosine_logits(self.P)
        base_result = self.diagnose(base, train, test)
        self.assertAlmostEqual(base_result["cleanup"]["pooled"]["excess"], 0.0, places=14)

        ids = np.arange(self.P * self.P)
        a, b = ids // self.P, ids % self.P
        classes = np.arange(self.P, dtype=np.float64)
        perturbation = np.cos(
            2 * np.pi * (classes[None, :] - 3 * a[:, None] + 2 * b[:, None]) / self.P
        )
        planted = base.copy()
        planted[train] += 0.8 * perturbation[train]
        planted_result = self.diagnose(planted, train, test)
        self.assertGreater(planted_result["cleanup"]["pooled"]["excess"], 0.01)

        temperatures = np.ones(self.P * self.P)
        temperatures[train] = 2.0
        scaled_result = self.diagnose(
            cosine_logits(self.P, temperatures=temperatures), train, test
        )
        self.assertGreater(scaled_result["cleanup"]["pooled"]["excess"], 0.01)

    def test_matching_is_exact_balanced_outcome_independent_and_json_safe(self):
        train, test = partition(self.P)
        first = self.diagnose(cosine_logits(self.P), train, test)
        changed_logits = np.arange(self.P * self.P * self.P, dtype=np.float64).reshape(
            self.P * self.P, self.P
        )
        second = self.diagnose(changed_logits, train, test)
        self.assertEqual(first["matching"], second["matching"])
        self.assertEqual(first["partition"], second["partition"])
        self.assertEqual(
            first["cleanup"]["pooled"]["th_edge_ids_sha256"],
            second["cleanup"]["pooled"]["th_edge_ids_sha256"],
        )
        self.assertEqual(
            first["cleanup"]["pooled"]["hh_edge_ids_sha256"],
            second["cleanup"]["pooled"]["hh_edge_ids_sha256"],
        )
        pooled = first["cleanup"]["pooled"]
        self.assertEqual(
            pooled["train_to_heldout"]["count"],
            pooled["heldout_to_heldout"]["count"],
        )
        self.assertGreater(pooled["matched_count"], 0)
        for axis in ("a", "b"):
            for delta in self.DELTAS:
                record = first["cleanup"][axis][str(delta)]
                self.assertEqual(
                    record["train_to_heldout"]["count"],
                    record["heldout_to_heldout"]["count"],
                )
                self.assertEqual(record["matched_count"], sum(
                    group["matched_count"] for group in record["groups"]
                ))
        json.dumps(first, allow_nan=False)

        # These constants pin the exact partition and selected-edge identities.
        self.assertEqual(
            first["partition"]["train_ids_sha256"],
            "9b58a5c0f74b38179554758be565afea26afcc0d6ec53017eff558ff4ddb5d0b",
        )
        self.assertEqual(
            first["partition"]["test_ids_sha256"],
            "1605f3f71067b08c4b025009a660e259d4ce856dad87f993038b942b8d148c0b",
        )
        self.assertEqual(
            pooled["th_edge_ids_sha256"],
            "11869fe9487cfc176836af67901f7610ec7627e2438e51283ddd4e3f211fd9f7",
        )
        self.assertEqual(
            pooled["hh_edge_ids_sha256"],
            "192f66ce084426244e94c0e575ab2441ab0a1d2bbafacef757fa29701a02e801",
        )

    def test_validation_rejects_bad_inputs_without_mutating_arguments(self):
        train, test = partition(self.P)
        logits = cosine_logits(self.P)
        train_before, test_before = train.copy(), test.copy()
        self.diagnose(logits, train, test)
        np.testing.assert_array_equal(train, train_before)
        np.testing.assert_array_equal(test, test_before)

        with self.assertRaisesRegex(ValueError, "partition"):
            self.diagnose(logits, train[:-1], test)
        with self.assertRaisesRegex(ValueError, "overlap"):
            self.diagnose(logits, train, np.append(test, train[0]))
        with self.assertRaisesRegex(ValueError, "shape"):
            self.diagnose(logits[:, :-1], train, test)
        corrupted = logits.copy()
        corrupted[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            self.diagnose(corrupted, train, test)


if __name__ == "__main__":
    unittest.main()
