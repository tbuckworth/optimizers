"""Fabricated metadata only; no actual inventory or candidate manifest stage."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('pattern_manifest', Path(__file__).with_name('plan_candidates.py'))
selection = importlib.util.module_from_spec(spec)
spec.loader.exec_module(selection)


class SelectionTests(unittest.TestCase):
    def groups(self):
        return [{'topic': topic, 'remaining_pageids': list(range(i*100+1, i*100+1+n))}
                for i, (topic, n) in enumerate(zip(selection.TOPICS, [5, 4, 0, 9]))]

    def test_exact_once_coverage_and_tails(self):
        pairs, tails = selection.ordered_pairs(self.groups())
        used = [p[k] for p in pairs for k in ['left_pageid', 'right_pageid']]
        all_ids = [i for g in self.groups() for i in g['remaining_pageids']]
        self.assertEqual(len(pairs), 8)
        self.assertEqual(len(used), len(set(used)))
        self.assertEqual(sorted(used+[t['pageid'] for t in tails]), sorted(all_ids))
        for p in pairs:
            self.assertEqual(p['left_pageid']//100, p['right_pageid']//100)

    def test_role_assignment(self):
        pairs, _ = selection.ordered_pairs(self.groups())
        self.assertEqual([p['role'] for p in pairs], ['calibration', 'calibration', 'evaluation']*2+['calibration']*2)
        self.assertEqual([p['candidate_id'] for p in pairs], [f'K{i:03d}' for i in range(1, 9)])

    def test_permuting_input_ids_does_not_change_manifest(self):
        groups = self.groups()
        expected = selection.ordered_pairs(groups)
        for g in groups: g['remaining_pageids'].reverse()
        self.assertEqual(selection.ordered_pairs(groups), expected)

    def test_exact_hash_recipe(self):
        pairs, _ = selection.ordered_pairs(self.groups())
        for p in pairs:
            text = f"20260921|pair|{p['topic']}|{p['left_pageid']}|{p['right_pageid']}"
            self.assertEqual(p['order_hash'], selection.digest(text.encode()))
        self.assertEqual([p['order_hash'] for p in pairs], sorted(p['order_hash'] for p in pairs))

    def test_ownership_before_exclusion(self):
        catalogue = {'groups': [
            {'topic': topic, 'candidates': [{'pageid': pid, 'ns': 0} for pid in ids]}
            for topic, ids in zip(selection.TOPICS, [[1, 2], [2, 3], [4], [3, 5]])]}
        groups, excluded = selection.available_from_raw(catalogue, [{'params': {'pageids': 2}}, {'params': {'pageids': 2}}])
        self.assertEqual([g['remaining_pageids'] for g in groups], [[1], [3], [4], [5]])
        self.assertEqual(excluded, [2])

    def test_bad_group_order_or_duplicate(self):
        with self.assertRaises(ValueError): selection.ordered_pairs(self.groups()[::-1])
        groups = self.groups(); groups[1]['remaining_pageids'].append(1)
        with self.assertRaises(ValueError): selection.ordered_pairs(groups)

    def test_strict_metadata_json(self):
        for raw in ['{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}', '{"x":1e999}', '{"x":1.5}']:
            with self.assertRaises(ValueError): selection.strict_json(raw)
        self.assertEqual(selection.strict_json('{"x":1,"ok":true}'), {'x': 1, 'ok': True})


if __name__ == '__main__':
    unittest.main()
