"""Fabricated-only tests: temporary text/JSON/git fixtures, no scientific inputs."""

import copy
import importlib.util
import json
from pathlib import Path
import random
import re
import subprocess
import tempfile
import unittest
from unittest import mock

SOURCE = Path(__file__).with_name("judging.py")
SPEC = importlib.util.spec_from_file_location("natural_addon_fixture", SOURCE)
j = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(j)


def fabricated():
    rows, records, pairs = [], [], []
    for n, ident in enumerate(j.IDS):
        topic = ("astronomy", "cooking", "programming")[n//2 % 3]
        text = " ".join([f"fixture{n}", *[f"word{i}" for i in range(1, 16)]])
        row = {"id": ident, "topic": topic, "prefix": text}
        offsets = [[m.start(), m.end()] for m in re.finditer(r"\S+", text)]
        pos = {"prefix_end": len(offsets)-1}
        records.append({**row, "input_ids": list(range(len(offsets))), "attention_mask": [1]*len(offsets),
                        "offset_mapping": offsets, "captured_positions": pos, "layer": 11,
                        "selected_substrings": {k: text[slice(*offsets[v])] for k, v in pos.items()}})
        rows.append(row)
    for n, pair_id in enumerate(j.PAIR_IDS):
        pairs.append({"id": pair_id, "left": j.IDS[2*n], "right": j.IDS[2*n+1],
                      "topic": rows[2*n]["topic"], "candidate_id": f"K{n+1:03}"})
    refs = {"schema": "jlens_fresh_references_v1", "axes": []}
    for axis in j.AXES:
        refs["axes"].append({"axis": axis,
            **{a: {s: [" lead", "\n", "\t", "空", "", "same", "same", "x", "y", "z", "q", "r"]
                   for s in ("positive", "negative")} for a in ("A", "B")},
            "C": {"positive": "Unchanged fitted high prefix", "negative": "Unchanged fitted low prefix"}})
    artifacts = {name: {"path": "/fabricated/"+name, "size_bytes": 10,
                         "sha256": j.FROZEN.get(name, "1"*64)} for name in j.ARTIFACT_NAMES}
    tokens = {"schema": "jlens_natural_addon_tokens_v1", "records": records}
    preflight = {"schema": "jlens_natural_addon_token_preflight_receipt_v1", "status": "complete",
                 "source_sha256": j.BASE_PRODUCER_SHA, "count": 32, "token_limit": 96,
                 "runtime": {"python": j.PREFLIGHT_PYTHON, "packages": dict(j.RUNTIME_PACKAGES)},
                 "model_loaded": False, "scientific_array_loaded": False,
                 "outputs": [{"path": "tokens.json", "sha256": artifacts["tokens"]["sha256"]}],
                 "input_pins": {artifacts[name]["path"]: artifacts[name]["sha256"] for name in ("dataset", "pairs", "protocol")}}
    values = [{"axis": a, "values": {ident: float(n) for n, ident in enumerate(j.IDS)}} for a in j.AXES]
    values[0]["values"][j.IDS[1]] = 0.0
    values[1]["values"][j.IDS[0]], values[1]["values"][j.IDS[1]] = 1e-12, 0.0
    scores = {"schema": "jlens_natural_addon_scores_v1", "locations": {"prefix_end": values}}
    return refs, rows, pairs, tokens, preflight, artifacts, scores


class PureTests(unittest.TestCase):
    def setUp(self):
        *self.args, self.scores = fabricated()
        self.public, self.mapping = j.make_packets(*self.args)

    def choices(self):
        scalar = j._validate_scores(self.scores)
        result = {r: {} for r in j.RATERS}
        for row in self.mapping["rows"]:
            gap = scalar[row["axis"]][row["first_id"]] - scalar[row["axis"]][row["second_id"]]
            result[row["rater"]][row["item_id"]] = ("FIRST" if gap >= 0 else "SECOND") if row["arm"] == "EA" else "FIRST"
        return result

    def test_import_inert_local_rng_and_deterministic(self):
        state = random.getstate()
        with mock.patch.object(Path, "read_bytes", side_effect=AssertionError("I/O")), \
             mock.patch.object(j.os, "open", side_effect=AssertionError("I/O")), \
             mock.patch.object(subprocess, "run", side_effect=AssertionError("subprocess")):
            spec = importlib.util.spec_from_file_location("inert_again", SOURCE)
            spec.loader.exec_module(importlib.util.module_from_spec(spec))
            self.assertEqual((self.public, self.mapping), j.make_packets(*self.args))
        self.assertEqual(state, random.getstate())

    def test_allocation_256_choices_and_opposite_cohorts(self):
        self.assertEqual(j.SEED, 20260920)
        self.assertEqual(j.ALLOCATION, {"rater1": ("EA",)*4, "rater2": ("E",)*4,
                                       "rater3": ("EA",)*4, "rater4": ("E",)*4})
        self.assertEqual(j.COHORT, dict(zip(j.RATERS, (1, 1, 2, 2))))
        self.assertEqual(len(self.mapping["rows"]), 256)
        cells = {}
        for row in self.mapping["rows"]:
            cells.setdefault((row["axis"], row["pair_id"], row["cohort"]), []).append(row)
        self.assertEqual(len(cells), 128)
        for (axis, pair, cohort), rows in cells.items():
            self.assertEqual({r["arm"] for r in rows}, {"EA", "E"})
            self.assertEqual(len({(r["first_id"], r["second_id"]) for r in rows}), 1)
            other = cells[axis, pair, 3-cohort][0]
            self.assertEqual((rows[0]["first_id"], rows[0]["second_id"]), (other["second_id"], other["first_id"]))
        self.assertEqual({r["item_id"] for r in self.mapping["rows"]}, {f"Q{i:03}" for i in range(1, 257)})
        for packet in self.public.values():
            self.assertEqual(len(j._public_ids(packet)), 64)
            self.assertEqual(len({type(b["positive_reference"]) for b in packet["blocks"]}), 1)
            for key in ('"axis"', '"arm"', '"cohort"', '"pole"', '"pair_id"', '"score"', '"truth"', '"verb_span"'):
                self.assertNotIn(key, json.dumps(packet))

    def test_full_prefixes_identical_examples_poles_tokens_and_layouts(self):
        refs, data, _, tokens, _, _ = self.args
        lookup = {r["id"]: r for r in data}
        mapping = {r["item_id"]: r for r in self.mapping["rows"]}
        layouts = []
        for rater, packet in self.public.items():
            layout = []
            for block in packet["blocks"]:
                m = mapping[block["comparisons"][0]["item_id"]]
                ref = refs["axes"][j.AXES.index(m["axis"])]
                for pole in ("positive", "negative"):
                    got = block[pole+"_reference"]
                    self.assertEqual(got["example_prefix"], ref["C"][pole])
                    self.assertEqual(set(got), {"example_prefix", "direction_tokens"} if m["arm"] == "EA" else {"example_prefix"})
                    if m["arm"] == "EA":
                        self.assertEqual(got["direction_tokens"], ref["A"][pole])
                layout.append((m["axis"], [mapping[v["item_id"]]["pair_id"] for v in block["comparisons"]]))
                for item in block["comparisons"]:
                    m = mapping[item["item_id"]]
                    for side in ("first", "second"):
                        self.assertEqual(item[side], lookup[m[side+"_id"]]["prefix"])
                        self.assertEqual(len(item[side].split()), 16)
            layouts.append(layout)
        self.assertTrue(all(v == layouts[0] for v in layouts))
        for ea, e in (("rater1", "rater2"), ("rater3", "rater4")):
            for ba, be in zip(self.public[ea]["blocks"], self.public[e]["blocks"]):
                self.assertEqual([(x["first"], x["second"]) for x in ba["comparisons"]],
                                 [(x["first"], x["second"]) for x in be["comparisons"]])
        for b1, b2 in zip(self.public["rater1"]["blocks"], self.public["rater3"]["blocks"]):
            self.assertEqual(b1["positive_reference"], b2["positive_reference"])
            self.assertEqual(b1["negative_reference"], b2["negative_reference"])
            self.assertEqual([(x["first"], x["second"]) for x in b1["comparisons"]],
                             [(x["second"], x["first"]) for x in b2["comparisons"]])

    def test_bad_rosters_offsets_and_receipts(self):
        mutations = [lambda a: a[1].reverse(), lambda a: a[2].pop(),
            lambda a: a[2][0].update(right=j.IDS[3]), lambda a: a[2][0].update(topic="football"),
            lambda a: a[2][1].update(candidate_id="K001"), lambda a: a[2][0].update(candidate_id="K000"),
            lambda a: a[1][0].update(prefix="short"),
            lambda a: a[3]["records"][0].update(id=j.IDS[1]),
            lambda a: a[3]["records"][0]["captured_positions"].update(prefix_end=0),
            lambda a: a[3]["records"][0]["captured_positions"].update(prefix_end=True),
            lambda a: a[3]["records"][0]["selected_substrings"].update(prefix_end="whole"),
            lambda a: a[3]["records"][0]["offset_mapping"][3].__setitem__(1, 999),
            lambda a: a[3]["records"][0]["offset_mapping"][0].__setitem__(0, 1),
            lambda a: a[3]["records"][0]["offset_mapping"][-1].__setitem__(1, 1),
            lambda a: a[3]["records"][0]["attention_mask"].__setitem__(0, 0),
            lambda a: a[4].update(count=24), lambda a: a[4].update(source_sha256="0"*64),
            lambda a: a[4]["outputs"][0].update(sha256="0"*64),
            lambda a: a[4].update(input_pins={}),
            lambda a: a[0]["axes"][0]["A"]["positive"].pop()]
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                args = copy.deepcopy(self.args); mutate(args)
                with self.assertRaises((ValueError, TypeError)): j.make_packets(*args)
        for k, wrapper in ((1, "rows"), (2, "pairs")):
            args = copy.deepcopy(self.args); args[k] = {wrapper: args[k]}
            with self.assertRaises(ValueError): j.make_packets(*args)

    def test_ordered_overlapping_unicode_offsets_allowed_but_gaps_rejected(self):
        args = copy.deepcopy(self.args)
        record = args[3]["records"][0]
        # Two byte-level subtokens may share a character span.
        record["offset_mapping"].insert(1, list(record["offset_mapping"][0]))
        record["input_ids"].insert(1, 100)
        record["attention_mask"].insert(1, 1)
        record["captured_positions"]["prefix_end"] += 1
        j.make_packets(*args)
        record["offset_mapping"][3][0] += 1
        with self.assertRaisesRegex(ValueError, "coverage"): j.make_packets(*args)

    def test_reference_copy_score_independence_and_homogeneous_public(self):
        before = copy.deepcopy(self.args)
        block = self.public["rater1"]["blocks"][0]
        block["positive_reference"]["direction_tokens"][0] = "changed"
        self.assertEqual(before, self.args)
        first = j.make_packets(*self.args); self.scores.clear()
        self.assertEqual(first, j.make_packets(*self.args))
        packet = copy.deepcopy(self.public["rater2"])
        packet["blocks"][0]["positive_reference"]["direction_tokens"] = ["x"]*12
        packet["blocks"][0]["negative_reference"]["direction_tokens"] = ["x"]*12
        with self.assertRaisesRegex(ValueError, "mixed reference"): j._public_ids(packet)

    def test_strict_response_64_and_json(self):
        packet = self.public["rater1"]
        response = {"schema": "jlens_natural_addon_responses_v1", "packet_id": packet["packet_id"],
                    "responses": [{"item_id": i, "choice": "FIRST"} for i in sorted(j._public_ids(packet))]}
        self.assertEqual(len(j.validate_response(response, packet)), 64)
        for mutate in (lambda x: x["responses"].pop(), lambda x: x["responses"].append(x["responses"][0]),
                       lambda x: x["responses"][0].update(confidence=1),
                       lambda x: x["responses"][0].update(choice="first"),
                       lambda x: x.update(schema="jlens_independent_content_responses_v1"),
                       lambda x: x.update(packet_id=self.public["rater2"]["packet_id"]),
                       lambda x: x["responses"][1].update(item_id=x["responses"][0]["item_id"])):
            bad = copy.deepcopy(response); mutate(bad)
            with self.assertRaises(ValueError): j.validate_response(bad, packet)
        for raw in (b'{"a":1,"a":2}', b'{"n":NaN}', b'{"n":Infinity}', b'{"n":1e999}'):
            with self.assertRaises(ValueError): j.decode(raw)

    def test_ties_tiny_gaps_all_reductions_and_primary(self):
        result = j._grade_values(self.mapping, self.choices(), self.scores)
        self.assertEqual(len(result["items"]), 256)
        self.assertEqual(result["arms"]["EA"]["credit"], 127)
        self.assertEqual(result["arms"]["E"]["credit"], 64)
        self.assertEqual(result["paired_differences"]["EA_minus_E"]["total_credit"], 63)
        for arm in j.ARMS:
            self.assertEqual(result["arms"][arm]["total"], 128)
            self.assertEqual(result["arms"][arm]["constant_FIRST_credit"], 64)
            self.assertEqual(result["arms"][arm]["constant_SECOND_credit"], 64)
            for axis in j.AXES:
                self.assertEqual(result["per_axis"][axis][arm]["total"], 32)
                self.assertEqual(result["per_axis"][axis][arm]["constant_FIRST_credit"], 16)
                self.assertEqual(sum(result["category_axis"][t][axis][arm]["total"] for t in j.TOPICS), 32)
                self.assertEqual(result["category_axis"]["football"][axis][arm]["total"], 0)
                for c in ("1", "2"):
                    self.assertEqual(result["cohort_axis"][c][axis][arm]["total"], 16)
        for r in j.RATERS:
            self.assertEqual(result["per_reader"][r]["summary"]["total"], 64)
            self.assertTrue(all(v["total"] == 16 for v in result["per_reader"][r]["per_axis"].values()))
        diff = result["paired_differences"]["EA_minus_E"]
        self.assertEqual(sum(v["total_credit"] for v in diff["per_cohort"].values()), 63)
        self.assertEqual(sum(v["total_credit"] for v in diff["per_category"].values()), 63)
        self.assertEqual(result["primary"]["EA_minus_E_credit_out_of_128"], 63)
        self.assertTrue(result["primary"]["each_EA_reader_exceeds_stronger_constant"])
        self.assertTrue(result["primary"]["EA_minus_E_positive_each_cohort"])
        self.assertTrue(result["primary"]["incremental_pilot_criterion_met"])
        for r in j.RATERS:
            v = result["per_reader"][r]["summary"]
            self.assertEqual(result["primary"]["reader_controls"][r]["stronger_constant_credit"],
                             max(v["constant_FIRST_credit"], v["constant_SECOND_credit"]))
        self.assertEqual(result["category_counts"]["football"], {"pairs": 0, "texts": 0})
        self.assertEqual(sum(v["pairs"] for v in result["category_counts"].values()), 16)
        self.assertEqual(len(result["gains_harm"]), 128)
        self.assertEqual(sum(v["EA_minus_E_credit"] for v in result["gains_harm"]), 63)
        tiny = [r for r in result["items"] if r["axis"] == "PC2" and r["pair_id"] == "N01"]
        self.assertEqual(len(tiny), 4)
        self.assertTrue(all(r["absolute_gap"] == 1e-12 and r["truth"] != "TIE" for r in tiny))

    def test_agreement_by_text_and_extra_role_rejected(self):
        result = j._grade_values(self.mapping, self.choices(), self.scores)
        self.assertEqual(result["agreement"]["arms"]["E"], {"agree": 0, "disagree": 64, "total": 64})
        self.assertEqual(result["agreement"]["arms"]["EA"]["agree"], 63)
        changed = copy.deepcopy(self.scores)
        changed["locations"]["sentence_end"] = copy.deepcopy(changed["locations"]["prefix_end"])
        with self.assertRaises(ValueError): j._grade_values(self.mapping, self.choices(), changed)

    def test_score_schema_whole_export_and_bad_roster(self):
        for location in ("prefix_end",):
            for bad in (True, float("inf"), float("nan"), "1"):
                scores = copy.deepcopy(self.scores)
                scores["locations"][location][0]["values"][j.IDS[0]] = bad
                with self.assertRaises(ValueError): j._validate_scores(scores)
        for mutate in (lambda s: s.update(schema="jlens_independent_content_scores_v1"),
                       lambda s: s["locations"]["prefix_end"].reverse(),
                       lambda s: s["locations"].pop("prefix_end"),
                       lambda s: s["locations"]["prefix_end"][0]["values"].pop(j.IDS[0])):
            bad = copy.deepcopy(self.scores); mutate(bad)
            with self.assertRaises(ValueError): j._validate_scores(bad)
        for mutate in (lambda m: m["rows"].pop(), lambda m: m["rows"][0].update(arm="B"),
                       lambda m: m["rows"][0].update(topic="not-a-category"),
                       lambda m: m["rows"][0].update(rater="rater6"),
                       lambda m: m["rows"][0].update(pair_id="P01")):
            bad = copy.deepcopy(self.mapping); mutate(bad)
            with self.assertRaises(ValueError): j._grade_values(bad, self.choices(), self.scores)

    def test_primary_strict_controls_and_harm_not_hidden(self):
        choices = {r: {row["item_id"]: "FIRST" for row in self.mapping["rows"] if row["rater"] == r} for r in j.RATERS}
        result = j._grade_values(self.mapping, choices, self.scores)
        self.assertFalse(result["primary"]["incremental_pilot_criterion_met"])
        self.assertFalse(result["primary"]["each_EA_reader_exceeds_stronger_constant"])
        self.assertEqual(result["primary"]["EA_minus_E_by_cohort_out_of_64"], {"1": 0, "2": 0})
        self.assertTrue(all(v["outcome"] == "tie" for v in result["gains_harm"]))
        better = self.choices()
        for row in self.mapping["rows"]:
            if row["arm"] == "EA":
                old = better[row["rater"]][row["item_id"]]
                better[row["rater"]][row["item_id"]] = "SECOND" if old == "FIRST" else "FIRST"
        harmed = j._grade_values(self.mapping, better, self.scores)
        self.assertEqual(harmed["primary"]["EA_minus_E_credit_out_of_128"], -63)
        self.assertTrue(any(v["outcome"] == "harm" for v in harmed["gains_harm"]))

    def test_frozen_literals_future_scores_not_in_source(self):
        self.assertEqual(j.FROZEN, {
    "protocol": "a510feaf2dfdd5cf6541d974d2c305188e24d04a4e9e185ac73675970b8a6246",
    "references": "0a678ad8d58a6b17fc2f0c3ac197d47be83e163750b2408f52759e23a0e6c44c",
    "dataset": "d57300cb8511e409ff78e6ddc9fb6e1cf11b91bc629bf5492db8932cf585b129",
    "pairs": "0ae85708e0b5da2942e31aa96af4ffa642c1294ff4c8b78612ec2dfc0ee54725",
    "producer": "894f465e36d50f6df15229754367614d3ffc26f20301db3eb57f9535189bbc57",
    "tokens": "e87f996149ea1eb80b3fb3a6225257859f047aefaf7727beb1e792b863e0ef43",  # gitleaks:allow (sha256 of tokens artifact)
    "preflight": "675411e8428e5a8797101b0bed85a4604a4e59d21fe41b40c498a3e330c492fb",
})
        self.assertFalse(hasattr(j, "SCORES_SHA"))


class FileStages(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="jlens-natural-addon-fabricated-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        refs, data, pairs, tokens, preflight, _, scores = fabricated()
        self.paths, self.artifacts = {}, {}
        for name, value in dict(references=refs, dataset=data, pairs=pairs, tokens=tokens,
                                scores=scores).items():
            self.store(name, j.encode(value))
        self.store("protocol", b"Fabricated protocol.\n")
        self.store("producer", b"# Fabricated producer source; no model.\n")
        pins = {self.artifacts[k]["path"]: self.artifacts[k]["sha256"]
                for k in ("dataset", "pairs", "protocol")}
        preflight["source_sha256"] = j.BASE_PRODUCER_SHA
        preflight["input_pins"] = pins
        preflight["outputs"][0]["sha256"] = self.artifacts["tokens"]["sha256"]
        self.store("preflight", j.encode(preflight))
        forward = {"schema": "jlens_natural_addon_forwards_receipt_v1", "status": "complete",
            "source_sha256": self.artifacts["producer"]["sha256"], "input_pins": pins,
            "base_producer_sha256": j.BASE_PRODUCER_SHA,
            "runtime_amendment_sha256": j.AMENDMENT_SHA,
            "failed_admission_sha256s": dict(j.FAILED_ADMISSION),
            "runtime": {"python": j.MEASUREMENT_PYTHON, "packages": dict(j.RUNTIME_PACKAGES)},
            "preflight_receipt_sha256": self.artifacts["preflight"]["sha256"],
            "forward_count": 32, "capture_locations": ["prefix_end"], "parameters_unchanged": True,
            "pca_refit": False, "reference_decodes": 0, "tokenizer_loaded": False,
            "outputs": [{"path": name, "sha256": self.artifacts["scores"]["sha256"] if name == "scores.json" else "1"*64}
                        for name in ("features.npz", "inputs.json", "scores.json", "gaps.json")]}
        self.store("forward_receipt", j.encode(forward))
        patch = mock.patch.dict(j.FROZEN, {k: self.artifacts[k]["sha256"] for k in j.FROZEN})
        patch.start(); self.addCleanup(patch.stop)
        j._git(self.root, "init", "--quiet")
        self.release = self.root/"release.json"
        self.freeze_release()

    def store(self, name, raw):
        filename = self.paths.setdefault(name, self.root/(name+".json"))
        filename.write_bytes(raw)
        self.artifacts[name] = {"path": str(filename), "size_bytes": len(raw), "sha256": j.sha(raw)}

    def commit_paths(self, names):
        j._git(self.root, "add", *names)
        j._git(self.root, "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
               "commit", "--quiet", "--allow-empty", "-m", "Freeze fabricated bytes")
        return j._git(self.root, "rev-parse", "HEAD").decode().strip()

    def freeze_release(self):
        self.release.write_bytes(j.encode({"schema": "jlens_natural_addon_release_v1", "artifacts": self.artifacts}))
        self.release_sha = j.sha(self.release.read_bytes())
        self.release_commit = self.commit_paths(["release.json"])

    def package(self, name="packets"):
        self.packet_dir = self.root/name
        self.manifest = j.package(self.packet_dir, release=self.release,
            release_sha256=self.release_sha, release_commit=self.release_commit)

    def seal(self):
        self.package()
        self.response_inputs = {}
        for r in j.RATERS:
            packet = j.decode((self.packet_dir/f"public/{r}.json").read_bytes())
            response = {"schema": "jlens_natural_addon_responses_v1", "packet_id": packet["packet_id"],
                        "responses": [{"item_id": i, "choice": "FIRST"} for i in sorted(j._public_ids(packet))]}
            filename = self.root/f"raw-{r}.json"
            filename.write_bytes(json.dumps(response, separators=(", ", ": ")).encode())
            self.response_inputs[r] = {"path": str(filename), "sha256": j.sha(filename.read_bytes())}
        self.lock_dir = self.root/"responses"
        self.lock = j.seal_responses(self.lock_dir, packets=self.packet_dir,
            manifest_sha256=self.manifest["sha256"], responses=self.response_inputs)

    def commit(self, omit_last=False):
        names = ["responses/lock.json", *[f"responses/{r}.json" for r in (j.RATERS[:-1] if omit_last else j.RATERS)]]
        return self.commit_paths(names)

    def lock_args(self, commit):
        return dict(packets=self.packet_dir, manifest_sha256=self.manifest["sha256"], responses=self.lock_dir,
                    lock_sha256=self.lock["sha256"], lock_commit=commit)

    def grade(self, commit, name="graded"):
        return j.grade(self.root/name, **self.lock_args(commit))

    def test_package_and_seal_no_key_or_forward_access_exact_blobs_and_guards(self):
        for name in ("scores", "forward_receipt", "producer"):
            self.paths[name].unlink()
        self.seal()
        for r, rec in self.response_inputs.items():
            self.assertEqual(Path(rec["path"]).read_bytes(), (self.lock_dir/f"{r}.json").read_bytes())
        self.assertEqual(j.decode((self.lock_dir/"lock.json").read_bytes())["count"], 256)
        with mock.patch.object(j, "read_bytes", side_effect=AssertionError("input before guard")):
            with self.assertRaises(FileExistsError): self.package()
            with self.assertRaises(FileExistsError):
                j.seal_responses(self.lock_dir, packets=self.packet_dir,
                    manifest_sha256=self.manifest["sha256"], responses=self.response_inputs)

    def test_failed_stage_consumed_and_frozen_input_pins(self):
        self.artifacts["dataset"]["sha256"] = "0"*64
        self.freeze_release()
        with self.assertRaises(ValueError): self.package()
        self.assertTrue((self.packet_dir/"failure.json").exists())
        self.assertTrue((self.packet_dir/"attempt.json").exists())
        with self.assertRaises(FileExistsError): self.package()

    def test_invalid_lock_blocks_before_key_forward_and_private_map(self):
        self.seal(); commit = self.commit(omit_last=True)
        original = j.read_bytes
        def guard(filename, expected):
            if Path(filename) in (self.paths["scores"], self.paths["forward_receipt"], self.packet_dir/"private-map.json"):
                raise AssertionError("key/map accessed before complete committed lock")
            return original(filename, expected)
        with mock.patch.object(j, "read_bytes", side_effect=guard):
            with self.assertRaises(subprocess.CalledProcessError): self.grade(commit)
        self.assertTrue((self.root/"graded/failure.json").exists())

    def test_five_lock_blobs_then_release_then_one_score_read(self):
        self.seal(); commit = self.commit()
        original_git, original_read = j._git, j.read_bytes
        checked, key_reads = [], []
        def spy(cwd, *args):
            result = original_git(cwd, *args)
            if args[:2] == ("cat-file", "blob"): checked.append(args[2])
            return result
        def read(filename, expected):
            if Path(filename) in (self.paths["scores"], self.paths["forward_receipt"]):
                self.assertEqual(len(checked), 6)
                self.assertTrue(all("responses/" in p for p in checked[:5]))
                self.assertTrue(checked[-1].endswith(":release.json"))
            if Path(filename) == self.paths["scores"]: key_reads.append(filename)
            return original_read(filename, expected)
        with mock.patch.object(j, "_git", side_effect=spy), mock.patch.object(j, "read_bytes", side_effect=read):
            receipt = self.grade(commit)
        self.assertEqual(len(key_reads), 1)
        result = j.decode((self.root/"graded/grades.json").read_bytes())
        self.assertEqual(len(result["items"]), 256)
        self.assertEqual(receipt["sha256"], j.sha((self.root/"graded/grades.json").read_bytes()))
        self.assertEqual(result["provenance"]["scores"], self.artifacts["scores"])
        self.assertEqual(result["provenance"]["inputs"], {k:self.artifacts[k] for k in j.INPUT_NAMES})
        self.assertEqual(result["provenance"]["graded_location"], "prefix_end")
        with self.assertRaises(FileExistsError): self.grade(commit)
        raw = self.lock_dir/"lock.json"; raw.write_bytes(raw.read_bytes()+b" ")
        self.lock["sha256"] = j.sha(raw.read_bytes())
        with self.assertRaises(ValueError): self.grade(commit, "mutated-lock")

    def test_public_checker_gate_reads_no_private_or_scientific_file(self):
        self.seal(); commit = self.commit()
        for filename in [self.packet_dir/"private-map.json", *self.paths.values(), self.release]:
            filename.unlink()
        verified = j.verify_response_lock(**self.lock_args(commit))
        self.assertEqual(set(verified), {"manifest", "public", "lock", "choices"})
        self.assertEqual(sum(len(v) for v in verified["choices"].values()), 256)
        for name, pin in j.FROZEN.items():
            self.assertIn(name, verified["manifest"]["inputs"])
            self.assertEqual(verified["manifest"]["inputs"][name]["sha256"], pin)

    def test_release_commit_required_before_any_input_reads(self):
        self.release.write_bytes(self.release.read_bytes()+b" ")
        self.release_sha = j.sha(self.release.read_bytes())
        original = j.read_bytes
        def guard(filename, expected):
            self.assertEqual(Path(filename), self.release)
            return original(filename, expected)
        with mock.patch.object(j, "read_bytes", side_effect=guard):
            with self.assertRaises(ValueError): self.package()
        self.assertTrue((self.packet_dir/"failure.json").exists())

    def test_release_strict_schema_and_receipt_sizes(self):
        value = j.decode(self.release.read_bytes())
        for mutate in (lambda x:x.update(extra=1), lambda x:x["artifacts"].pop("scores"),
                       lambda x:x["artifacts"]["producer"].update(size_bytes=True),
                       lambda x:x["artifacts"]["scores"].update(sha256="bad"),
                       lambda x:x["artifacts"]["scores"].update(path="relative/path")):
            bad=copy.deepcopy(value); mutate(bad)
            self.release.write_bytes(j.encode(bad)); self.release_sha=j.sha(self.release.read_bytes())
            self.release_commit=self.commit_paths(["release.json"])
            with self.assertRaises(ValueError):
                j._release(self.release,self.release_sha,self.release_commit)
        self.artifacts["tokens"]["size_bytes"] += 1
        self.freeze_release()
        with self.assertRaisesRegex(ValueError,"artifact size mismatch"): self.package()

    def test_forward_scope_before_key(self):
        f=j.decode(self.paths["forward_receipt"].read_bytes())
        f["outputs"][2]["sha256"]="0"*64
        self.store("forward_receipt",j.encode(f)); self.freeze_release()
        self.seal(); commit=self.commit()
        original=j.read_bytes
        def guard(filename,expected):
            if Path(filename)==self.paths["scores"]: raise AssertionError("score before scope")
            return original(filename,expected)
        with mock.patch.object(j,"read_bytes",side_effect=guard):
            with self.assertRaisesRegex(ValueError,"score-output binding"): self.grade(commit)

    def test_reconstruction_before_forward_or_score(self):
        self.seal(); commit=self.commit()
        with mock.patch.object(j,"make_packets",return_value=({},{})), \
             mock.patch.object(j,"_forward_scope",side_effect=AssertionError("scope before reconstruction")):
            with self.assertRaisesRegex(ValueError,"packet/source/map mismatch"): self.grade(commit)

    def test_only_explicit_runtime_amendment_accepted(self):
        original = j.decode(self.paths["forward_receipt"].read_bytes())
        j._forward_scope(original, self.artifacts)
        for mutate in (lambda r: r.update(base_producer_sha256="0"*64),
                       lambda r: r.update(runtime_amendment_sha256="0"*64),
                       lambda r: r.update(failed_admission_sha256s={}),
                       lambda r: r["runtime"].update(python=j.PREFLIGHT_PYTHON),
                       lambda r: r["runtime"]["packages"].update(torch="other")):
            bad = copy.deepcopy(original); mutate(bad)
            with self.assertRaises(ValueError): j._forward_scope(bad, self.artifacts)

    def test_missing_response_no_lock_and_changed_score_bytes(self):
        self.seal()
        missing = dict(self.response_inputs); missing.pop("rater4")
        with self.assertRaises(ValueError):
            j.seal_responses(self.root/"missing", packets=self.packet_dir,
                manifest_sha256=self.manifest["sha256"], responses=missing)
        self.assertFalse((self.root/"missing/lock.json").exists())
        commit = self.commit(); self.paths["scores"].write_bytes(b"{}")
        with self.assertRaisesRegex(ValueError, "input SHA256 mismatch"): self.grade(commit)

    def test_caps_symlinks_and_exclusive_attempt(self):
        big = self.root/"big.json"; big.write_bytes(b" "*(j.MAX_JSON+1))
        with self.assertRaises(ValueError): j.read_bytes(big, j.sha(big.read_bytes()))
        link = self.root/"link.json"; link.symlink_to(self.paths["references"])
        with self.assertRaises(OSError): j.read_bytes(link, self.artifacts["references"]["sha256"])
        dangling = self.root/"dangling"; dangling.symlink_to(self.root/"absent")
        with self.assertRaises(FileExistsError): j.Attempt(dangling, "fixture")
        attempt = j.Attempt(self.root/"writer", "fixture")
        attempt.raw("grades.json", b"{}")
        with self.assertRaises(FileExistsError): attempt.raw("grades.json", b"[]")
        with self.assertRaises(ValueError): attempt.raw("../escape", b"{}")
        attempt.used = j.MAX_OUTPUT
        with self.assertRaises(ValueError): attempt.raw("manifest.json", b"{}")


if __name__ == "__main__":
    unittest.main()
