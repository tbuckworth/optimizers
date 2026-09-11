"""Fabricated-only tests; never loads the study's actual inputs or outcomes."""

import copy
import importlib.util
import json
from pathlib import Path
import random
import subprocess
import tempfile
import unittest
from unittest import mock


SOURCE = Path(__file__).with_name("judging.py")
SPEC = importlib.util.spec_from_file_location("within_topic_judging_fixture_module", SOURCE)
j = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(j)


def fabricated():
    dataset = [{"id": ident, "topic": ident.split("-")[0],
                "prefix": f"The synthetic {ident} subject patiently waited"} for ident in j.IDS]
    edges = [(i, i + 1) for i in range(0, 24, 2)]
    pairs = [{"id": f"P{i + 1:02}", "left": j.IDS[a], "right": j.IDS[b]}
             for i, (a, b) in enumerate(edges)]
    refs = {"schema": "jlens_fresh_references_v1", "axes": []}
    for index, axis in enumerate(j.AXES):
        row = {"axis": axis, "C": {"positive": f"Intact fitted high example {index}",
                                    "negative": f"Intact fitted low example {index}"}}
        for ai, arm in enumerate(("A", "B")):
            row[arm] = {sign: [" leading", "\n", "\t", "\\", "空", "", *[
                f"word{index}{ai}{sign}{i}" for i in range(6)]] for sign in ("positive", "negative")}
        refs["axes"].append(row)
    scores = {"schema": "jlens_fresh_scores_v1", "scores": [
        {"axis": axis, "values": {ident: float(n + 1) for n, ident in enumerate(j.IDS)}} for axis in j.AXES]}
    scores["scores"][0]["values"][j.IDS[0]] = 0.0
    scores["scores"][0]["values"][j.IDS[1]] = 0.0
    scores["scores"][1]["values"][j.IDS[2]] = 1e-12
    scores["scores"][1]["values"][j.IDS[3]] = 0.0
    return refs, dataset, pairs, scores


class PureTests(unittest.TestCase):
    def setUp(self):
        self.refs, self.data, self.pairs, self.scores = fabricated()
        self.public, self.mapping = j.make_packets(self.refs, self.data, self.pairs)

    def test_import_inert_and_local_rng(self):
        state = random.getstate()
        with mock.patch.object(Path, "read_bytes", side_effect=AssertionError("import I/O")), \
             mock.patch.object(j.os, "open", side_effect=AssertionError("import I/O")), \
             mock.patch.object(subprocess, "run", side_effect=AssertionError("import subprocess")):
            spec = importlib.util.spec_from_file_location("another_inert_fixture", SOURCE)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        self.assertEqual(state, random.getstate())
        j.make_packets(self.refs, self.data, self.pairs)
        self.assertEqual(state, random.getstate())

    def test_exact_roster_allocation_swaps_and_anonymous_structure(self):
        self.assertEqual(20260913, j.SEED)
        self.assertEqual({"rater1": ("B", "C", "A", "B"),
                          "rater2": ("C", "A", "B", "C"),
                          "rater3": ("A", "B", "C", "A")}, j.ALLOCATION)
        self.assertEqual("Using only the two reference descriptions, which of the two prefixes "
                         "should have the higher value on this direction? Choose FIRST or SECOND "
                         "even if uncertain.", j.PROMPT)
        self.assertEqual("jlens_within_topic_private_map_v1", self.mapping["schema"])
        self.assertEqual(144, len(self.mapping["rows"]))
        seen, cells = set(), {}
        for row in self.mapping["rows"]:
            self.assertNotIn(row["item_id"], seen)
            seen.add(row["item_id"])
            ai = j.AXES.index(row["axis"])
            self.assertEqual(j.ALLOCATION[row["rater"]][ai], row["arm"])
            cells.setdefault((row["axis"], row["pair_id"]), []).append(row)
        self.assertEqual(48, len(cells))
        for rows in cells.values():
            self.assertEqual(set(j.ARMS), {r["arm"] for r in rows})
            self.assertEqual(1, len({(r["first_id"], r["second_id"]) for r in rows}))
        for packet in self.public.values():
            self.assertEqual("jlens_within_topic_public_v1", packet["schema"])
            self.assertEqual(48, len(j._public_ids(packet)))
            serialized = json.dumps(packet)
            for forbidden in ('"axis"', '"arm"', '"score"', '"truth"', '"source_id"', '"pair_id"'):
                self.assertNotIn(forbidden, serialized)
            for axis in j.AXES:
                self.assertNotIn(axis, serialized)
        self.assertEqual(self.public, j.make_packets(self.refs, self.data, self.pairs)[0])

    def test_only_fixed_adjacent_within_topic_roster_allowed(self):
        expected_edges = [(f"{topic}-new-{i}", f"{topic}-new-{i+1}")
                          for topic in ("astronomy", "cooking", "football", "programming")
                          for i in (0, 2, 4)]
        self.assertEqual(expected_edges, [(p["left"], p["right"]) for p in self.pairs])
        # Still disjoint and same-topic, but a different matching: (0,3),(2,1).
        alternative = copy.deepcopy(self.pairs)
        alternative[0]["right"], alternative[1]["right"] = alternative[1]["right"], alternative[0]["right"]
        with self.assertRaises(ValueError):
            j.make_packets(self.refs, self.data, alternative)
        # Still covers every ID once, but now includes cross-topic comparisons.
        cross_topic = copy.deepcopy(self.pairs)
        cross_topic[0]["right"], cross_topic[3]["right"] = cross_topic[3]["right"], cross_topic[0]["right"]
        with self.assertRaises(ValueError):
            j.make_packets(self.refs, self.data, cross_topic)
        reversed_pair = copy.deepcopy(self.pairs)
        reversed_pair[0]["left"], reversed_pair[0]["right"] = reversed_pair[0]["right"], reversed_pair[0]["left"]
        with self.assertRaises(ValueError):
            j.make_packets(self.refs, self.data, reversed_pair)

    def test_references_and_query_strings_are_unchanged(self):
        lookup = {r["id"]: r["prefix"] for r in self.data}
        rows = {r["item_id"]: r for r in self.mapping["rows"]}
        for rater, packet in self.public.items():
            for block in packet["blocks"]:
                first_row = rows[block["comparisons"][0]["item_id"]]
                reference = self.refs["axes"][j.AXES.index(first_row["axis"])][first_row["arm"]]
                self.assertEqual(reference["positive"], block["positive_reference"])
                self.assertEqual(reference["negative"], block["negative_reference"])
                for item in block["comparisons"]:
                    row = rows[item["item_id"]]
                    self.assertEqual(rater, row["rater"])
                    self.assertEqual(lookup[row["first_id"]], item["first"])
                    self.assertEqual(lookup[row["second_id"]], item["second"])

    def test_mutation_isolation_and_score_independent_packaging(self):
        old = copy.deepcopy(self.refs)
        block = next(b for p in self.public.values() for b in p["blocks"] if isinstance(b["positive_reference"], list))
        block["positive_reference"][0] = "modified output"
        self.assertEqual(old, self.refs)
        before = j.make_packets(self.refs, self.data, self.pairs)
        for row in self.scores["scores"]:
            row["values"] = {k: -v for k, v in row["values"].items()}
        self.assertEqual(before, j.make_packets(self.refs, self.data, self.pairs))

    def test_bad_input_rosters_and_reference_metadata_rejected(self):
        for mutation in (lambda r, d, p: d.reverse(),
                         lambda r, d, p: p[0].update(left=p[0]["right"]),
                         lambda r, d, p: r["axes"].reverse(),
                         lambda r, d, p: r["axes"][0]["A"]["positive"].pop(),
                         lambda r, d, p: r["axes"][0]["A"].update(score=1),
                         lambda r, d, p: r["axes"][0]["C"].update(positive=[])):
            refs, data, pairs = copy.deepcopy((self.refs, self.data, self.pairs))
            mutation(refs, data, pairs)
            with self.assertRaises(ValueError):
                j.make_packets(refs, data, pairs)

    def test_response_rejection_and_strict_json(self):
        packet = self.public["rater1"]
        good = {"schema": "jlens_within_topic_responses_v1", "packet_id": packet["packet_id"],
                "responses": [{"item_id": x, "choice": "FIRST"} for x in sorted(j._public_ids(packet))]}
        self.assertEqual(48, len(j.validate_response(good, packet)))
        for mutate in (lambda v: v["responses"].pop(),
                       lambda v: v.update(schema="jlens_fresh_responses_v1"),
                       lambda v: v["responses"].append(v["responses"][0]),
                       lambda v: v["responses"][0].update(choice="first"),
                       lambda v: v["responses"][0].update(confidence="high"),
                       lambda v: v.update(packet_id=self.public["rater2"]["packet_id"]),
                       lambda v: v["responses"][1].update(item_id=v["responses"][0]["item_id"])):
            value = copy.deepcopy(good)
            mutate(value)
            with self.assertRaises(ValueError):
                j.validate_response(value, packet)
        for raw in (b'{"a":1,"a":2}', b'{"n":NaN}', b'{"n":Infinity}', b'{"n":1e999}'):
            with self.assertRaises(ValueError):
                j.decode(raw)

    def choices(self):
        scalar = j._validate_scores(self.scores)
        choices = {r: {} for r in j.RATERS}
        for row in self.mapping["rows"]:
            gap = scalar[row["axis"]][row["first_id"]] - scalar[row["axis"]][row["second_id"]]
            correct = "FIRST" if gap >= 0 else "SECOND"
            choice = correct if row["arm"] == "A" else ("SECOND" if correct == "FIRST" else "FIRST")
            choices[row["rater"]][row["item_id"]] = "FIRST" if row["arm"] == "C" else choice
        return choices

    def test_exact_ties_small_gaps_and_all_grading_reductions(self):
        result = j._grade_values(self.mapping, self.choices(), self.scores)
        self.assertEqual("jlens_within_topic_grades_v1", result["schema"])
        self.assertEqual(144, len(result["items"]))
        self.assertEqual(47.5, result["arms"]["A"]["credit"])
        self.assertEqual(0.5, result["arms"]["B"]["credit"])
        self.assertEqual(47, result["paired_differences"]["A_minus_B"]["total_credit"])
        for arm in j.ARMS:
            summary = result["arms"][arm]
            self.assertEqual(48, summary["total"])
            self.assertEqual(1, summary["exact_ties"])
            self.assertEqual(48, summary["constant_FIRST_credit"] + summary["constant_SECOND_credit"])
            self.assertEqual(summary["credit"], sum(result["per_axis"][x][arm]["credit"] for x in j.AXES))
            for axis in j.AXES:
                self.assertEqual(12, result["per_axis"][axis][arm]["total"])
        tiny = [r for r in result["items"] if r["axis"] == "PC2" and r["pair_id"] == "P02"]
        self.assertEqual(3, len(tiny))
        self.assertTrue(all(r["absolute_gap"] == 1e-12 and r["truth"] != "TIE" for r in tiny))
        self.assertEqual(result["arms"]["C"]["credit"], result["arms"]["C"]["constant_FIRST_credit"])

    def test_score_schema_nonfinite_bool_and_overflow_rejection(self):
        for bad in (True, float("inf"), float("nan"), "1.0"):
            score = copy.deepcopy(self.scores)
            score["scores"][0]["values"][j.IDS[0]] = bad
            with self.assertRaises(ValueError):
                j._validate_scores(score)
        score = copy.deepcopy(self.scores)
        score["scores"][0]["values"][j.IDS[0]] = 1e308
        score["scores"][0]["values"][j.IDS[1]] = -1e308
        with self.assertRaises(ValueError):
            j._grade_values(self.mapping, self.choices(), score)


class FileStages(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="jlens-judging-fabricated-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.refs, self.data, self.pairs, self.scores = fabricated()
        self.paths = {}
        for name, value in (("references", self.refs), ("dataset", self.data),
                            ("pairs", self.pairs), ("scores", self.scores)):
            self.paths[name] = self.root / f"{name}.json"
            self.paths[name].write_bytes(j.encode(value))
        self.paths["protocol"] = self.root / "protocol.md"
        self.paths["protocol"].write_bytes(b"Fabricated protocol fixture, no scientific inputs.\n")
        self.pins = {name: j.sha(filename.read_bytes()) for name, filename in self.paths.items()}
        patch = mock.patch.dict(j.FROZEN, {name: self.pins[name] for name in j.FROZEN})
        patch.start()
        self.addCleanup(patch.stop)

    def package(self, name="packets"):
        self.packet_dir = self.root / name
        self.manifest = j.package(self.packet_dir, references=self.paths["references"],
            references_sha256=self.pins["references"], dataset=self.paths["dataset"],
            pairs=self.paths["pairs"], protocol=self.paths["protocol"])
        return self.manifest

    def seal(self):
        self.package()
        response_inputs = {}
        for rater in j.RATERS:
            packet = j.decode((self.packet_dir / f"public/{rater}.json").read_bytes())
            response = {"schema": "jlens_within_topic_responses_v1", "packet_id": packet["packet_id"],
                        "responses": [{"item_id": x, "choice": "FIRST"} for x in sorted(j._public_ids(packet))]}
            filename = self.root / f"original-{rater}.json"
            # Noncanonical spacing tests exact-byte preservation, not rewriting.
            filename.write_bytes(json.dumps(response, separators=(", ", ": ")).encode())
            response_inputs[rater] = {"path": str(filename), "sha256": j.sha(filename.read_bytes())}
        self.lock_dir = self.root / "responses"
        self.lock = j.seal_responses(self.lock_dir, packets=self.packet_dir,
            manifest_sha256=self.manifest["sha256"], responses=response_inputs)
        self.response_inputs = response_inputs

    def grade(self, commit, name="graded"):
        return j.grade(self.root / name, packets=self.packet_dir,
            manifest_sha256=self.manifest["sha256"], responses=self.lock_dir,
            lock_sha256=self.lock["sha256"], lock_commit=commit,
            scores=self.paths["scores"], scores_sha256=self.pins["scores"])

    def test_package_has_no_score_access_and_no_overwrite(self):
        self.paths["scores"].unlink()
        self.package()
        before = (self.packet_dir / "manifest.json").read_bytes()
        with mock.patch.object(j, "read_bytes", side_effect=AssertionError("input read before guard")):
            with self.assertRaises(FileExistsError):
                self.package()
        self.assertEqual(before, (self.packet_dir / "manifest.json").read_bytes())

    def test_failed_package_consumes_attempt_and_preserves_files(self):
        self.paths["references"].write_bytes(b"{}")
        with self.assertRaises(ValueError):
            self.package()
        self.assertTrue((self.packet_dir / "attempt.json").is_file())
        self.assertTrue((self.packet_dir / "failure.json").is_file())
        self.assertFalse((self.packet_dir / "manifest.json").exists())
        with self.assertRaises(FileExistsError):
            self.package()

    def test_seal_complete_exact_bytes_and_no_score_io(self):
        self.paths["scores"].unlink()
        self.seal()
        for rater in j.RATERS:
            self.assertEqual(Path(self.response_inputs[rater]["path"]).read_bytes(),
                             (self.lock_dir / f"{rater}.json").read_bytes())
        lock = j.decode((self.lock_dir / "lock.json").read_bytes())
        self.assertEqual(144, lock["count"])
        self.assertEqual(self.manifest["sha256"], lock["packet_manifest_sha256"])
        with self.assertRaises(FileExistsError):
            j.seal_responses(self.lock_dir, packets=self.packet_dir,
                manifest_sha256=self.manifest["sha256"], responses=self.response_inputs)

    def test_uncommitted_lock_blocks_before_scores(self):
        self.seal()
        original = j.read_bytes
        accessed_scores = []
        def guarded(filename, expected):
            if Path(filename) == self.paths["scores"]:
                accessed_scores.append(filename)
            return original(filename, expected)
        with mock.patch.object(j, "read_bytes", side_effect=guarded):
            with self.assertRaises((ValueError, subprocess.CalledProcessError)):
                self.grade("0" * 40)
        self.assertEqual([], accessed_scores)
        self.assertTrue((self.root / "graded/failure.json").is_file())

    def test_real_fabricated_git_lock_and_grade_once(self):
        self.seal()
        j._git(self.root, "init", "--quiet")
        j._git(self.root, "add", "responses/lock.json", *[f"responses/{r}.json" for r in j.RATERS])
        j._git(self.root, "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
               "commit", "--quiet", "-m", "Lock fabricated responses")
        commit = j._git(self.root, "rev-parse", "HEAD").decode().strip()
        receipt = self.grade(commit)
        result = j.decode((self.root / "graded/grades.json").read_bytes())
        self.assertEqual(j.sha((self.root / "graded/grades.json").read_bytes()), receipt["sha256"])
        self.assertEqual(commit, result["provenance"]["lock_commit"])
        self.assertEqual(144, len(result["items"]))
        for arm in j.ARMS:
            self.assertEqual(result["arms"][arm]["constant_FIRST_credit"], result["arms"][arm]["credit"])
        with self.assertRaises(FileExistsError):
            self.grade(commit)
        # Same-content bytes, different whitespace: current hash can be truthful,
        # but the new lock bytes are not those committed before grading.
        filename = self.lock_dir / "lock.json"
        filename.write_bytes(filename.read_bytes() + b" ")
        self.lock["sha256"] = j.sha(filename.read_bytes())
        with self.assertRaises(ValueError):
            self.grade(commit, name="uncommitted-mutation")
        self.assertFalse((self.root / "uncommitted-mutation/grades.json").exists())

    def test_missing_response_cannot_create_complete_lock(self):
        self.seal()
        response = copy.deepcopy(self.response_inputs)
        response.pop("rater3")
        with self.assertRaises(ValueError):
            j.seal_responses(self.root / "incomplete", packets=self.packet_dir,
                manifest_sha256=self.manifest["sha256"], responses=response)
        self.assertFalse((self.root / "incomplete/lock.json").exists())
        self.assertFalse((self.root / "incomplete/rater1.json").exists())

    def test_file_caps_symlink_and_receipt_path_rejection(self):
        big = self.root / "big.json"
        big.write_bytes(b" " * (j.MAX_JSON + 1))
        with self.assertRaises(ValueError):
            j.read_bytes(big, j.sha(big.read_bytes()))
        link = self.root / "linked.json"
        link.symlink_to(self.paths["references"])
        with self.assertRaises(OSError):
            j.read_bytes(link, self.pins["references"])
        attempt = j.Attempt(self.root / "writer", "fixture")
        attempt.raw("grades.json", b"{}")
        with self.assertRaises(FileExistsError):
            attempt.raw("grades.json", b"[]")
        with self.assertRaises(ValueError):
            attempt.raw("../escape.json", b"{}")
        attempt.used = j.MAX_OUTPUT
        with self.assertRaises(ValueError):
            attempt.raw("manifest.json", b"{}")
        attempt.used = j.MAX_OUTPUT - 4096
        attempt.fail(ValueError("fabricated budget failure"))
        self.assertTrue((attempt.root / "failure.json").is_file())
        with self.assertRaises(ValueError):
            j._member(self.root, {"path": "../escape", "size_bytes": 0, "sha256": "0" * 64}, "lock.json")


if __name__ == "__main__":
    unittest.main()
