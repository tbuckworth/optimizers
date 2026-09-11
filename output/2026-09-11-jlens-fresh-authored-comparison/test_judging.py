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
SPEC = importlib.util.spec_from_file_location("verb_reader_fixture", SOURCE)
j = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(j)


def fabricated():
    rows, records = [], []
    for n, ident in enumerate(j.IDS):
        pair, template, pole = ident.split("-")
        suffix = chr(65 + int(pair[2:]))
        verb = "observed" if pole == "O" else "prepared"
        text = (f"The actor{suffix} {verb} the samples{suffix}." if template == "active" else
                f"The samples{suffix} were {verb} by the actor{suffix}.")
        start = text.index(verb)
        row = {"id": ident, "pair_id": pair, "template": template, "pole": pole,
               "text": text, "verb": verb, "verb_span": [start, start + len(verb)]}
        offsets = []
        for m in re.finditer(r"\S+", text):
            if m.group() == verb:
                offsets.extend([[m.start(), m.end()-3], [m.end()-3, m.end()]])
            else:
                offsets.append([m.start(), m.end()])
        pos = {"verb": next(i for i, p in enumerate(offsets) if p[1] == row["verb_span"][1])}
        records.append({**row, "input_ids": list(range(len(offsets))), "attention_mask": [1]*len(offsets),
                        "offset_mapping": offsets, "captured_positions": pos, "layer": 11,
                        "selected_substrings": {k: text[slice(*offsets[v])] for k, v in pos.items()}})
        rows.append(row)
    pairs = [{"id": f"{p}-{t}", "content_pair": p, "template": t,
              "observation_id": f"{p}-{t}-O", "provision_id": f"{p}-{t}-P"} for p, t in j.CELLS]
    refs = {"schema": "jlens_fresh_references_v1", "axes": []}
    for axis in j.AXES:
        refs["axes"].append({"axis": axis,
            **{a: {s: [" lead", "\n", "\t", "空", "", "same", "same", "x", "y", "z", "q", "r"]
                   for s in ("positive", "negative")} for a in ("A", "B")},
            "C": {"positive": "Unchanged fitted high prefix", "negative": "Unchanged fitted low prefix"}})
    artifacts = {name: {"path": "/fabricated/"+name, "size_bytes": 10,
                         "sha256": j.FROZEN.get(name, "1"*64)} for name in j.ARTIFACT_NAMES}
    tokens = {"schema": "jlens_fresh_authored_tokens_v1", "records": records}
    preflight = {"schema": "jlens_fresh_authored_token_preflight_receipt_v1", "status": "complete",
                 "source_sha256": artifacts["producer"]["sha256"], "count": 32, "token_limit": 96,
                 "model_loaded": False, "scientific_array_loaded": False,
                 "outputs": [{"path": "tokens.json", "sha256": artifacts["tokens"]["sha256"]}],
                 "input_pins": {name: j.FROZEN[name] for name in ("dataset", "pairs", "protocol")}}
    values = [{"axis": a, "values": {ident: float(n) for n, ident in enumerate(j.IDS)}} for a in j.AXES]
    values[0]["values"][j.IDS[1]] = 0.0
    values[1]["values"][j.IDS[0]], values[1]["values"][j.IDS[1]] = 1e-12, 0.0
    scores = {"schema": "jlens_fresh_authored_scores_v1", "locations": {"verb": values}}
    return refs, {"rows": rows}, {"pairs": pairs}, tokens, preflight, artifacts, scores


class PureTests(unittest.TestCase):
    def setUp(self):
        *self.args, self.scores = fabricated()
        self.public, self.mapping = j.make_packets(*self.args)

    def choices(self):
        scalar = j._validate_scores(self.scores)
        result = {r: {} for r in j.RATERS}
        for row in self.mapping["rows"]:
            gap = scalar[row["axis"]][row["first_id"]] - scalar[row["axis"]][row["second_id"]]
            result[row["rater"]][row["item_id"]] = ("FIRST" if gap >= 0 else "SECOND") if row["arm"] == "A" else "FIRST"
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
        self.assertEqual(j.SEED, 20260917)
        self.assertEqual(j.ALLOCATION, {"rater1": ("A", "A", "A", "A"), "rater2": ("C", "C", "C", "C"),
                                       "rater3": ("A", "A", "A", "A"), "rater4": ("C", "C", "C", "C")})
        self.assertEqual(j.COHORT, dict(zip(j.RATERS, (1, 1, 2, 2))))
        self.assertEqual(len(self.mapping["rows"]), 256)
        cells = {}
        for row in self.mapping["rows"]:
            cells.setdefault((row["axis"], row["pair_id"], row["cohort"]), []).append(row)
        self.assertEqual(len(cells), 128)
        for (axis, pair, cohort), rows in cells.items():
            self.assertEqual({r["arm"] for r in rows}, {"A", "C"})
            self.assertEqual(len({(r["first_id"], r["second_id"]) for r in rows}), 1)
            other = cells[axis, pair, 3-cohort][0]
            self.assertEqual((rows[0]["first_id"], rows[0]["second_id"]), (other["second_id"], other["first_id"]))
        self.assertEqual({r["item_id"] for r in self.mapping["rows"]}, {f"Q{i:03}" for i in range(1, 257)})
        for packet in self.public.values():
            self.assertEqual(len(j._public_ids(packet)), 64)
            self.assertEqual(len({type(b["positive_reference"]) for b in packet["blocks"]}), 1)
            for key in ('"axis"', '"arm"', '"cohort"', '"pole"', '"pair_id"', '"score"', '"truth"', '"verb_span"'):
                self.assertNotIn(key, json.dumps(packet))

    def test_complete_verb_not_last_fragment_and_no_future_words(self):
        refs, data, _, tokens, _, _ = self.args
        lookup = {r["id"]: r for r in data["rows"]}
        mapping = {r["item_id"]: r for r in self.mapping["rows"]}
        self.assertTrue(all(len(r["selected_substrings"]["verb"]) == 3 for r in tokens["records"]))
        for packet in self.public.values():
            for block in packet["blocks"]:
                m = mapping[block["comparisons"][0]["item_id"]]
                reference = refs["axes"][j.AXES.index(m["axis"])][m["arm"]]
                self.assertEqual(block["positive_reference"], reference["positive"])
                self.assertEqual(block["negative_reference"], reference["negative"])
                for item in block["comparisons"]:
                    m = mapping[item["item_id"]]
                    for side in ("first", "second"):
                        row = lookup[m[side+"_id"]]
                        self.assertEqual(item[side], row["text"][:row["verb_span"][1]])
                        self.assertTrue(item[side].endswith(row["verb"]))
                        self.assertEqual(len(item[side].split()), 3 if row["template"] == "active" else 4)

    def test_bad_wrappers_rosters_spans_offsets_and_receipts(self):
        mutations = [lambda a: a[1]["rows"].reverse(), lambda a: a[2]["pairs"].pop(),
            lambda a: a[2]["pairs"][0].update(provision_id=j.IDS[3]),
            lambda a: a[1]["rows"][0]["verb_span"].__setitem__(1, 1),
            lambda a: a[3]["records"][0].update(id=j.IDS[1]),
            lambda a: a[3]["records"][0]["captured_positions"].update(verb=0),
            lambda a: a[3]["records"][0]["selected_substrings"].update(verb="whole"),
            lambda a: a[3]["records"][0]["offset_mapping"][3].__setitem__(1, 99),
            lambda a: a[3]["records"][0]["attention_mask"].__setitem__(0, 0),
            lambda a: a[4].update(count=24), lambda a: a[4].update(source_sha256="0"*64),
            lambda a: a[4]["outputs"][0].update(sha256="0"*64),
            lambda a: a[4].update(input_pins={}),
            lambda a: a[0]["axes"][0]["A"]["positive"].pop()]
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                args = copy.deepcopy(self.args)
                mutate(args)
                with self.assertRaises((ValueError, TypeError)):
                    j.make_packets(*args)
        args = copy.deepcopy(self.args)
        args[1] = args[1]["rows"]
        with self.assertRaises(ValueError):
            j.make_packets(*args)

    def test_reference_copy_and_score_independence(self):
        before = copy.deepcopy(self.args)
        block = next(b for p in self.public.values() for b in p["blocks"] if isinstance(b["positive_reference"], list))
        block["positive_reference"][0] = "changed"
        self.assertEqual(before, self.args)
        first = j.make_packets(*self.args)
        self.scores.clear()
        self.assertEqual(first, j.make_packets(*self.args))

    def test_strict_response_64_and_json(self):
        packet = self.public["rater1"]
        response = {"schema": "jlens_fresh_authored_responses_v1", "packet_id": packet["packet_id"],
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
        self.assertEqual(result["arms"]["A"]["credit"], 127)
        self.assertEqual(result["arms"]["C"]["credit"], 64)
        self.assertEqual(result["paired_differences"]["A_minus_C"]["total_credit"], 63)
        for arm in j.ARMS:
            self.assertEqual(result["arms"][arm]["total"], 128)
            self.assertEqual(result["arms"][arm]["constant_FIRST_credit"], 64)
            self.assertEqual(result["arms"][arm]["constant_SECOND_credit"], 64)
            for axis in j.AXES:
                self.assertEqual(result["per_axis"][axis][arm]["total"], 32)
                self.assertEqual(result["per_axis"][axis][arm]["constant_FIRST_credit"], 16)
                for t in ("active", "passive"):
                    self.assertEqual(result["template_axis"][t][axis][arm]["total"], 16)
                for c in ("1", "2"):
                    self.assertEqual(result["cohort_axis"][c][axis][arm]["total"], 16)
        for r in j.RATERS:
            self.assertEqual(result["per_reader"][r]["summary"]["total"], 64)
            self.assertTrue(all(v["total"] == 16 for v in result["per_reader"][r]["per_axis"].values()))
        diff = result["paired_differences"]["A_minus_C"]
        self.assertEqual(sum(v["total_credit"] for v in diff["per_cohort"].values()), 63)
        self.assertEqual(sum(v["total_credit"] for v in diff["per_template"].values()), 63)
        self.assertEqual(result["primary"]["A_reader_credit_out_of_16"], {"rater1": 16, "rater3": 16})
        self.assertTrue(result["primary"]["each_A_reader_exceeds_8"])
        tiny = [r for r in result["items"] if r["axis"] == "PC2" and r["pair_id"] == "fa01-active"]
        self.assertEqual(len(tiny), 4)
        self.assertTrue(all(r["absolute_gap"] == 1e-12 and r["truth"] != "TIE" for r in tiny))

    def test_agreement_by_text_and_extra_role_rejected(self):
        result = j._grade_values(self.mapping, self.choices(), self.scores)
        self.assertEqual(result["agreement"]["arms"]["C"], {"agree": 0, "disagree": 64, "total": 64})
        self.assertEqual(result["agreement"]["arms"]["A"]["agree"], 63)
        changed = copy.deepcopy(self.scores)
        changed["locations"]["sentence_end"] = copy.deepcopy(changed["locations"]["verb"])
        with self.assertRaises(ValueError): j._grade_values(self.mapping, self.choices(), changed)

    def test_score_schema_whole_export_and_bad_roster(self):
        for location in ("verb",):
            for bad in (True, float("inf"), float("nan"), "1"):
                scores = copy.deepcopy(self.scores)
                scores["locations"][location][0]["values"][j.IDS[0]] = bad
                with self.assertRaises(ValueError): j._validate_scores(scores)
        for mutate in (lambda s: s.update(schema="jlens_independent_content_scores_v1"),
                       lambda s: s["locations"]["verb"].reverse(),
                       lambda s: s["locations"].pop("verb"),
                       lambda s: s["locations"]["verb"][0]["values"].pop(j.IDS[0])):
            bad = copy.deepcopy(self.scores); mutate(bad)
            with self.assertRaises(ValueError): j._validate_scores(bad)
        for mutate in (lambda m: m["rows"].pop(), lambda m: m["rows"][0].update(arm="B"),
                       lambda m: m["rows"][0].update(template="passive"),
                       lambda m: m["rows"][0].update(rater="rater6"),
                       lambda m: m["rows"][0].update(pair_id="P01")):
            bad = copy.deepcopy(self.mapping); mutate(bad)
            with self.assertRaises(ValueError): j._grade_values(bad, self.choices(), self.scores)

    def test_frozen_design_literals_future_hashes_not_in_source(self):
        self.assertEqual(j.FROZEN, {
            "protocol": "a66353d5838df1ab5d9e0a5ff034edcda12bbd4db1a80ba6a256b4760172b71b",
            "references": "0a678ad8d58a6b17fc2f0c3ac197d47be83e163750b2408f52759e23a0e6c44c",
            "dataset": "80749619a5bdbfd9506b8d453cc7f54d8f50c6eff73729f6a148152c0c0b35bf",
            "pairs": "ec361bee2a15ec796230824c2e9125f46cebe76316a6b50ee45904cb2be02912"})
        self.assertFalse(hasattr(j, "SCORES_SHA"))
        self.assertFalse(hasattr(j, "PRODUCER_SHA"))


class FileStages(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="jlens-fresh-authored-fabricated-")
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
        preflight["source_sha256"] = self.artifacts["producer"]["sha256"]
        preflight["input_pins"] = pins
        preflight["outputs"][0]["sha256"] = self.artifacts["tokens"]["sha256"]
        self.store("preflight", j.encode(preflight))
        forward = {"schema": "jlens_fresh_authored_forwards_receipt_v1", "status": "complete",
            "source_sha256": self.artifacts["producer"]["sha256"], "input_pins": pins,
            "preflight_receipt_sha256": self.artifacts["preflight"]["sha256"],
            "forward_count": 32, "capture_locations": ["verb"], "parameters_unchanged": True,
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
        self.release.write_bytes(j.encode({"schema": "jlens_fresh_authored_release_v1", "artifacts": self.artifacts}))
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
            response = {"schema": "jlens_fresh_authored_responses_v1", "packet_id": packet["packet_id"],
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
        self.assertEqual(result["provenance"]["graded_location"], "verb")
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
