"""Dataset-free CPU fixtures, not neural effect estimates."""
import importlib.util
import itertools
from pathlib import Path
import unittest

import torch

spec = importlib.util.spec_from_file_location("i7_response_math", Path(__file__).with_name("response_math.py"))
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)


class ResponseTests(unittest.TestCase):
    def setUp(self):
        self.g = torch.tensor([3., 4.])
        self.c = torch.tensor([3., 0.])
        self.l = torch.tensor([0., 4.])

    def test_six_known_vectors_and_owned_storage(self):
        result = r.construct(self.g, self.c, self.l)
        expected = {"raw": [3., 4.], "current": [3., 0.], "lagged": [0., 4.],
                    "restored": [0., 3.], "reciprocal": [4., 0.], "zero": [0., 0.]}
        pointers = set()
        for key, row in result["branches"].items():
            self.assertTrue(torch.equal(row["gradient"], torch.tensor(expected[key])))
            self.assertEqual(row["status"], "defined")
            pointers.add(row["gradient"].untyped_storage().data_ptr())
        self.assertEqual(len(pointers), 6)
        result["branches"]["current"]["gradient"].add_(100)
        self.assertTrue(torch.equal(self.c, torch.tensor([3., 0.])))
        self.assertEqual(result["branches"]["raw"]["gradient"][0].item(), 3.)

    def test_four_zero_cases(self):
        for c_zero, l_zero in itertools.product((False, True), repeat=2):
            with self.subTest(c_zero=c_zero, l_zero=l_zero):
                c = torch.zeros(2) if c_zero else self.c
                l = torch.zeros(2) if l_zero else self.l
                rows = r.construct(self.g, c, l)["branches"]
                expected_undefined = ({"reciprocal"} if c_zero and not l_zero else
                                      {"restored"} if l_zero and not c_zero else set())
                self.assertEqual({key for key, row in rows.items() if row["status"] == "undefined"},
                                 expected_undefined)
                if c_zero:
                    self.assertTrue(torch.equal(rows["restored"]["gradient"], torch.zeros(2)))
                if l_zero:
                    self.assertTrue(torch.equal(rows["reciprocal"]["gradient"], torch.zeros(2)))

    def test_tiny_positive_is_not_zero(self):
        tiny = torch.nextafter(torch.tensor(0.), torch.tensor(1.)).item()
        c = torch.tensor([tiny, 0.])
        result = r.construct(self.g, c, self.l)
        self.assertGreater(result["leverage"]["current_norm"], 0.)
        self.assertEqual(result["branches"]["restored"]["gradient"][1].item(), tiny)

    def test_tiny_positive_quantization_is_not_domain_null(self):
        tiny = torch.nextafter(torch.tensor(0.), torch.tensor(1.)).item()
        with self.assertRaisesRegex(ValueError, "gate"):
            r.construct(torch.ones(2), torch.tensor([tiny, 0.]), torch.ones(2))

    def test_complete_native_underflow_is_not_domain_null(self):
        tiny = torch.nextafter(torch.tensor(0.), torch.tensor(1.)).item()
        c = torch.zeros(8)
        c[0] = tiny
        with self.assertRaisesRegex(ValueError, "underflow"):
            r.construct(torch.ones(8), c, torch.ones(8))

    def test_native_zero_signed_bits_are_preserved(self):
        raw = torch.tensor([-0., 0.])
        result = r.construct(raw, raw, raw)
        for key in ("raw", "current", "lagged"):
            self.assertTrue(torch.equal(result["branches"][key]["gradient"].view(torch.uint8),
                                        raw.view(torch.uint8)))

    def test_bad_inputs(self):
        invalid = (torch.tensor([float("nan"), 1.]), torch.ones(3), torch.ones(2, dtype=torch.float64),
                   torch.ones(1, 2), torch.ones(2, requires_grad=True))
        for item in invalid:
            with self.subTest(item=item), self.assertRaises(ValueError):
                r.construct(self.g, item, self.l)

    def test_leverage_and_unclipped_geometry(self):
        result = r.construct(self.g, self.c, self.l)
        self.assertTrue(result["leverage"]["norm_leverage"])
        self.assertTrue(result["leverage"]["direction_leverage"])
        self.assertEqual(len(result["delivered_pairs"]), 15)
        self.assertEqual(result["leverage"]["unit_direction_cosine"], 0.)
        same = r.construct(self.g, self.c, self.c)
        self.assertFalse(same["leverage"]["norm_leverage"])
        self.assertFalse(same["leverage"]["direction_leverage"])
        self.assertIsNone(same["delivered_pairs"]["raw__zero"]["cosine"])

    def test_known_factorial_values(self):
        values = dict(zip(r.BRANCHES, (5., 2., 7., 11., 3., 1.)))
        expected = (5., 9., 4., -1., 4., 5., -3., 2., 6., -2., 4., 1., 6., 10., 2.)
        actual = r.contrasts(values)
        self.assertEqual(tuple(row["value"] for row in actual.values()), expected)
        self.assertEqual(len(actual), 15)
        self.assertEqual(actual["interaction"]["value"],
                         actual["direction_at_current_norm"]["value"] - actual["direction_at_lagged_norm"]["value"])
        self.assertEqual(actual["interaction"]["value"],
                         actual["norm_at_lagged_direction"]["value"] - actual["norm_at_current_direction"]["value"])

    def test_shared_baseline_and_all_mask_patterns(self):
        base = dict(zip(r.BRANCHES, (5., 2., 7., 11., 3., 1.)))
        shifted = {key: value + 1000000 for key, value in base.items()}
        self.assertEqual([v["value"] for v in r.contrasts(base).values()],
                         [v["value"] for v in r.contrasts(shifted).values()])
        for mask in itertools.product((False, True), repeat=6):
            values = {key: None if missing else base[key] for key, missing in zip(r.BRANCHES, mask)}
            actual = r.contrasts(values)
            for key, coefficients in r.COEFFICIENTS.items():
                expected_missing = any(values[branch] is None for branch in coefficients)
                self.assertEqual(actual[key]["value"] is None, expected_missing)
                self.assertEqual(actual[key]["reason"] is not None, expected_missing)

    def test_missing_branch_and_nonfinite_rejected(self):
        values = {key: 0. for key in r.BRANCHES}
        values.pop("zero")
        with self.assertRaises(ValueError):
            r.contrasts(values)
        for bad in (float("inf"), float("nan"), True, "1", 10**400):
            values = {key: 0. for key in r.BRANCHES}
            values["raw"] = bad
            with self.assertRaises(ValueError):
                r.contrasts(values)

    def test_summary_requires_complete_declared_set(self):
        for mask in itertools.product((False, True), repeat=3):
            values = {bundle: None if missing else float(i) for i, (bundle, missing) in enumerate(zip(r.PRIMARY, mask))}
            result = r.primary_summary(values)
            self.assertEqual(result["mean"] is None, any(mask))
            self.assertEqual(result["defined_mask"], [not item for item in mask])
        result = r.primary_summary(dict(zip(r.PRIMARY, (1., 2., 3.))))
        self.assertEqual((result["mean"], result["min"], result["max"]), (2., 1., 3.))
        for invalid in ({71001: 1., 71002: 2.}, {71901: 1.}, {71001: 1., 71002: 2., 71003: 3., 71901: 4.}):
            with self.assertRaises(ValueError):
                r.primary_summary(invalid)

    def test_calls_preserve_rng_and_do_not_initialize_cuda(self):
        before = torch.random.get_rng_state().clone()
        r.construct(self.g, self.c, self.l)
        r.contrasts({key: 0. for key in r.BRANCHES})
        r.primary_summary({key: 0. for key in r.PRIMARY})
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == "__main__":
    unittest.main()
