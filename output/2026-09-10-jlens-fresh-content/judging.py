"""Inert, standard-library-only J-Lens packets, response lock and grading.

No import-time I/O; no model, NumPy, archive, judge or network dependency.
Acquisition JSON contracts (all keys exact; axes in PC1..PC4 order):
 references: {"schema":"jlens_fresh_references_v1","axes":[
   {"axis":"PC1","A":{"positive":[12 strings],"negative":[12 strings]},
    "B":{"positive":[12 strings],"negative":[12 strings]},
    "C":{"positive":"fit maximum prefix","negative":"fit minimum prefix"}}, ...]}
 scores: {"schema":"jlens_fresh_scores_v1","scores":[
   {"axis":"PC1","values":{"astronomy-new-0": finite_number, ...24 IDs}}, ...]}
Scores are acquisition's canonical U32/FP64 scalar exports, never recomputed.

API stages, each claiming a fresh exclusive output directory before input I/O:
 package(output, references=path, references_sha256=..., dataset=path,
         pairs=path, protocol=path) -> manifest receipt.
 seal_responses(output, packets=path, manifest_sha256=...,
                responses={"rater1":{"path":path,"sha256":...}, ...}) -> lock receipt.
 grade(output, packets=path, manifest_sha256=..., responses=lock_directory,
       lock_sha256=..., lock_commit=immutable_git_commit,
       scores=path, scores_sha256=...) -> grades receipt.

Give raters ONLY their public/raterN.json, not this source/private manifests.
Responses have exactly {"schema":"jlens_fresh_responses_v1","packet_id":...,
 "responses":[{"item_id":...,"choice":"FIRST" or "SECOND"}, ...48]}.
Before grade, main must commit lock.json and all three exact-byte response
copies. grade verifies those four blobs against lock_commit before score I/O.
Commit binds packets through the manifest hash. This is local file provenance,
not verification of actual rater transport/isolation or adversarial tamperproofing.

Official Python 3.12 contracts consulted: /library/json.html (duplicate keys,
finite numbers and size limits), /library/functions.html#open (exclusive mode),
/library/random.html#notes-on-reproducibility, /library/os.html#os.open.
Use under a bounded CPU-only invocation. All failures consume their directory;
partial files remain. There is no automatic retry, repair or overwrite path.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import stat
import subprocess
import sys


SEED = 20260912
AXES = tuple(f"PC{i}" for i in range(1, 5))
ARMS = ("A", "B", "C")
RATERS = ("rater1", "rater2", "rater3")
TOPICS = ("astronomy", "cooking", "football", "programming")
IDS = tuple(f"{t}-new-{i}" for t in TOPICS for i in range(6))
ALLOCATION = {"rater1": ("A", "B", "C", "A"),
              "rater2": ("B", "C", "A", "B"),
              "rater3": ("C", "A", "B", "C")}
PROMPT = ("Using only the two reference descriptions, which of the two new prefixes "
          "should have the higher value on this direction? Choose FIRST or SECOND "
          "even if uncertain.")
FROZEN = {
    "protocol": "56b7d85fa14438f06cd6a23996c858f4f3cd908f45eed3a44c96cf7b5a5f9980",
    "dataset": "c655a5531af3215a29e332ddc6b23ac4854b0bffdf91b8ad5e8a045748ff7cd9",
    "pairs": "8fa981cd7700b1385502e5b96ce451eebb9c34f6b93e05037203e9f314c1347d",
}
MAX_JSON = 1024 * 1024
MAX_OUTPUT = 8 * 1024 * 1024


def require(condition, message):
    if not condition:
        raise ValueError(message)


def keys(value, expected):
    require(type(value) is dict and set(value) == set(expected), "unexpected JSON keys")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def digest(value):
    require(type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value), "invalid SHA256")
    return value


def string(value):
    require(type(value) is str and 0 < len(value) <= 4096, "invalid reference/text string")


def encode(value):
    return (json.dumps(value, ensure_ascii=True, allow_nan=False, indent=2) + "\n").encode()


def _pairs_hook(items):
    out = {}
    for key, value in items:
        require(key not in out, "duplicate JSON key")
        out[key] = value
    return out


def _invalid_number(_):
    raise ValueError("nonfinite JSON number")


def decode(data):
    require(len(data) <= MAX_JSON, "JSON exceeds byte cap")
    value = json.loads(data.decode("utf-8"), object_pairs_hook=_pairs_hook,
                       parse_constant=_invalid_number)
    def finite(v):
        if type(v) is float:
            require(math.isfinite(v), "nonfinite JSON number")
        elif type(v) is dict:
            for child in v.values():
                finite(child)
        elif type(v) is list:
            for child in v:
                finite(child)
    finite(value)
    return value


def path(value):
    p = Path(value)
    require(p.is_absolute() and p.name not in ("", ".", ".."), "absolute file path required")
    return p.parent.resolve(strict=True) / p.name


def read_bytes(filename, expected):
    digest(expected)
    p = path(filename)
    fd = os.open(p, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, "rb") as handle:
        before = os.fstat(handle.fileno())
        require(stat.S_ISREG(before.st_mode) and before.st_size <= MAX_JSON,
                "input is not a bounded regular file")
        raw = handle.read(MAX_JSON + 1)
        after = os.fstat(handle.fileno())
    require(len(raw) == before.st_size == after.st_size and
            before.st_mtime_ns == after.st_mtime_ns, "input changed during read")
    require(sha(raw) == expected, "input SHA256 mismatch")
    return raw, {"path": str(p), "size_bytes": len(raw), "sha256": expected}


def read_json(filename, expected):
    raw, receipt = read_bytes(filename, expected)
    return decode(raw), receipt


def source_sha():
    return sha(Path(__file__).read_bytes())


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


class Attempt:
    def __init__(self, directory, stage):
        self.root = path(directory)
        self.root.mkdir(mode=0o700, exist_ok=False)
        self.used = 0
        self.write("attempt.json", {"stage": stage, "started_utc": utc(),
                                  "pid": os.getpid(), "python": sys.version,
                                  "source_sha256": source_sha()})

    def raw(self, name, data):
        require(name in {"attempt.json", "failure.json", "manifest.json", "private-map.json",
                         "lock.json", "grades.json", *[f"{r}.json" for r in RATERS],
                         *[f"public/{r}.json" for r in RATERS]}, "unexpected output name")
        cap = MAX_OUTPUT if name == "failure.json" else MAX_OUTPUT - 4096
        require(len(data) <= MAX_JSON and self.used + len(data) <= cap,
                "output byte cap exceeded")
        target = self.root / name
        if target.parent != self.root:
            if not target.parent.exists():
                target.parent.mkdir(mode=0o700)
            require(not target.parent.is_symlink(), "symlink output parent")
        fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        self.used += len(data)
        return {"path": name, "size_bytes": len(data), "sha256": sha(data)}

    def write(self, name, value):
        return self.raw(name, encode(value))

    def fail(self, error):
        self.write("failure.json", {"status": "FAILED", "error_type": type(error).__name__,
                                    "failed_utc": utc()})


def validate_inputs(references, dataset, pairs):
    require(type(dataset) is list and len(dataset) == 24, "dataset roster")
    for row, expected in zip(dataset, IDS):
        keys(row, ("id", "topic", "prefix"))
        require(row["id"] == expected and row["topic"] == expected.split("-")[0], "dataset ID/order")
        string(row["prefix"])
    require(len({r["prefix"] for r in dataset}) == 24, "duplicate fresh prefix")
    require(type(pairs) is list and len(pairs) == 12, "pair roster")
    used, contrasts = [], {}
    for i, row in enumerate(pairs):
        keys(row, ("id", "left", "right"))
        require(row["id"] == f"P{i + 1:02}", "pair ID/order")
        require(row["left"] in IDS and row["right"] in IDS, "unknown pair endpoint")
        a, b = sorted((row["left"].split("-")[0], row["right"].split("-")[0]))
        require(a != b, "within-topic pair")
        contrasts[a, b] = contrasts.get((a, b), 0) + 1
        used.extend((row["left"], row["right"]))
    require(sorted(used) == sorted(IDS) and len(contrasts) == 6 and
            set(contrasts.values()) == {2}, "unbalanced/disjoint pair roster")
    keys(references, ("schema", "axes"))
    require(references["schema"] == "jlens_fresh_references_v1", "reference schema")
    require(type(references["axes"]) is list and len(references["axes"]) == 4, "reference axes")
    for row, axis in zip(references["axes"], AXES):
        keys(row, ("axis", *ARMS))
        require(row["axis"] == axis, "reference axis/order")
        for arm in ARMS:
            keys(row[arm], ("positive", "negative"))
            for val in row[arm].values():
                if arm == "C":
                    string(val)
                else:
                    require(type(val) is list and len(val) == 12, "reference token count")
                    for token in val:
                        # Empty decoded fragments are valid strings and must not be removed.
                        require(type(token) is str and len(token) <= 4096, "reference token string")


def make_packets(references, dataset, pairs):
    """Pure construction for fixtures and package; accepts no score/key input."""
    validate_inputs(references, dataset, pairs)
    rng = random.Random(SEED)
    swaps = {(a, p["id"]): rng.choice((False, True)) for a in AXES for p in pairs}
    block_ids, item_ids = list(range(1, 13)), list(range(1, 145))
    rng.shuffle(block_ids)
    rng.shuffle(item_ids)
    text = {r["id"]: r["prefix"] for r in dataset}
    packets, private = {}, []
    for rater in RATERS:
        blocks = []
        for ai, axis in enumerate(AXES):
            arm = ALLOCATION[rater][ai]
            block_id = f"B{block_ids.pop():03}"
            refs = references["axes"][ai][arm]
            block = {"block_id": block_id,
                     "positive_reference": json.loads(json.dumps(refs["positive"])),
                     "negative_reference": json.loads(json.dumps(refs["negative"])),
                     "comparisons": []}
            for pair in pairs:
                first, second = pair["left"], pair["right"]
                if swaps[axis, pair["id"]]:
                    first, second = second, first
                item = f"Q{item_ids.pop():03}"
                block["comparisons"].append({"item_id": item, "first": text[first], "second": text[second]})
                private.append({"item_id": item, "block_id": block_id, "rater": rater,
                                "axis": axis, "arm": arm, "pair_id": pair["id"],
                                "first_id": first, "second_id": second})
            rng.shuffle(block["comparisons"])
            blocks.append(block)
        rng.shuffle(blocks)
        packets[rater] = {"schema": "jlens_fresh_public_v1", "packet_id": f"R{rng.getrandbits(64):016x}",
                          "prompt": PROMPT, "blocks": blocks}
    return packets, {"schema": "jlens_fresh_private_map_v1", "rows": private}


def _public_ids(packet):
    keys(packet, ("schema", "packet_id", "prompt", "blocks"))
    require(packet["schema"] == "jlens_fresh_public_v1" and packet["prompt"] == PROMPT,
            "public schema/prompt")
    require(type(packet["packet_id"]) is str and re.fullmatch(r"R[0-9a-f]{16}", packet["packet_id"]),
            "public packet ID")
    require(type(packet["blocks"]) is list and len(packet["blocks"]) == 4, "public block count")
    blocks, items = [], []
    for block in packet["blocks"]:
        keys(block, ("block_id", "positive_reference", "negative_reference", "comparisons"))
        require(type(block["block_id"]) is str and re.fullmatch(r"B\d{3}", block["block_id"]), "block ID")
        blocks.append(block["block_id"])
        a, b = block["positive_reference"], block["negative_reference"]
        if type(a) is list:
            require(type(b) is list and len(a) == len(b) == 12 and
                    all(type(x) is str and len(x) <= 4096 for x in a + b), "public tokens")
        else:
            string(a)
            string(b)
        require(type(block["comparisons"]) is list and len(block["comparisons"]) == 12, "public pair count")
        for item in block["comparisons"]:
            keys(item, ("item_id", "first", "second"))
            require(type(item["item_id"]) is str and re.fullmatch(r"Q\d{3}", item["item_id"]), "item ID")
            string(item["first"])
            string(item["second"])
            items.append(item["item_id"])
    require(len(set(blocks)) == 4 and len(set(items)) == 48, "duplicate public IDs")
    return set(items)


def validate_response(response, packet):
    ids = _public_ids(packet)
    keys(response, ("schema", "packet_id", "responses"))
    require(response["schema"] == "jlens_fresh_responses_v1" and
            response["packet_id"] == packet["packet_id"], "response packet/schema")
    require(type(response["responses"]) is list and len(response["responses"]) == 48, "response count")
    choices = {}
    for row in response["responses"]:
        keys(row, ("item_id", "choice"))
        require(type(row["item_id"]) is str and row["item_id"] in ids and
                row["item_id"] not in choices, "response ID/allocation")
        require(type(row["choice"]) is str and row["choice"] in ("FIRST", "SECOND"), "response choice")
        choices[row["item_id"]] = row["choice"]
    require(set(choices) == ids, "response completeness")
    return choices


def package(output, *, references, references_sha256, dataset, pairs, protocol):
    attempt = Attempt(output, "package")
    try:
        refs, rr = read_json(references, references_sha256)
        data, dr = read_json(dataset, FROZEN["dataset"])
        pair_data, pr = read_json(pairs, FROZEN["pairs"])
        _, tr = read_bytes(protocol, FROZEN["protocol"])
        packets, mapping = make_packets(refs, data, pair_data)
        public = {r: attempt.write(f"public/{r}.json", packets[r]) for r in RATERS}
        private = attempt.write("private-map.json", mapping)
        return attempt.write("manifest.json", {"schema": "jlens_fresh_packets_v1", "seed": SEED,
            "python": sys.version, "source_sha256": source_sha(),
            "inputs": {"references": rr, "dataset": dr, "pairs": pr, "protocol": tr},
            "public": public, "private_map": private})
    except Exception as error:
        attempt.fail(error)
        raise


def _member(root, receipt, expected_name):
    keys(receipt, ("path", "size_bytes", "sha256"))
    require(receipt["path"] == expected_name and type(receipt["size_bytes"]) is int and
            0 <= receipt["size_bytes"] <= MAX_JSON, "invalid member receipt")
    raw, actual = read_bytes(Path(root) / expected_name, receipt["sha256"])
    require(actual["size_bytes"] == receipt["size_bytes"], "member byte size")
    return raw


def _bundle(directory, expected):
    manifest, _ = read_json(Path(directory) / "manifest.json", expected)
    keys(manifest, ("schema", "seed", "python", "source_sha256", "inputs", "public", "private_map"))
    require(manifest["schema"] == "jlens_fresh_packets_v1" and type(manifest["seed"]) is int and
            manifest["seed"] == SEED and manifest["python"] == sys.version and
            manifest["source_sha256"] == source_sha(), "packet environment/source/schema")
    keys(manifest["inputs"], ("references", *FROZEN))
    keys(manifest["public"], RATERS)
    packets = {r: decode(_member(directory, manifest["public"][r], f"public/{r}.json")) for r in RATERS}
    ids = [_public_ids(packets[r]) for r in RATERS]
    require(len(set.union(*ids)) == 144 and len({p["packet_id"] for p in packets.values()}) == 3,
            "cross-packet duplicate IDs")
    require(len({b["block_id"] for p in packets.values() for b in p["blocks"]}) == 12,
            "cross-packet duplicate block IDs")
    return manifest, packets


def seal_responses(output, *, packets, manifest_sha256, responses):
    attempt = Attempt(output, "seal_responses")
    try:
        _, public = _bundle(packets, manifest_sha256)
        keys(responses, RATERS)
        raw_responses = {}
        for rater in RATERS:
            keys(responses[rater], ("path", "sha256"))
            raw, _ = read_bytes(responses[rater]["path"], responses[rater]["sha256"])
            validate_response(decode(raw), public[rater])
            raw_responses[rater] = raw
        copies = {r: attempt.raw(f"{r}.json", raw_responses[r]) for r in RATERS}
        return attempt.write("lock.json", {"schema": "jlens_fresh_response_lock_v1",
            "sealed_utc": utc(), "source_sha256": source_sha(), "count": 144,
            "packet_manifest_sha256": digest(manifest_sha256), "responses": copies})
    except Exception as error:
        attempt.fail(error)
        raise


def _git(cwd, *args):
    return subprocess.run(["git", "-C", str(cwd), *args], check=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10).stdout


def _committed_lock(directory, commit, files):
    require(type(commit) is str and re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", commit),
            "immutable lock commit required")
    directory = Path(directory).resolve(strict=True)
    root = Path(_git(directory, "rev-parse", "--show-toplevel").decode().strip()).resolve(strict=True)
    require(_git(root, "cat-file", "-t", commit).strip() == b"commit", "lock object is not a commit")
    for name, raw in files.items():
        rel = (directory / name).relative_to(root).as_posix()
        spec = f"{commit}:{rel}"
        size = int(_git(root, "cat-file", "-s", spec).strip())
        require(size == len(raw) and size <= MAX_JSON, "committed lock size mismatch")
        require(_git(root, "cat-file", "blob", spec) == raw, "committed lock bytes mismatch")


def _validate_scores(value):
    keys(value, ("schema", "scores"))
    require(value["schema"] == "jlens_fresh_scores_v1", "score schema")
    require(type(value["scores"]) is list and len(value["scores"]) == 4, "score axes")
    out = {}
    for row, axis in zip(value["scores"], AXES):
        keys(row, ("axis", "values"))
        require(row["axis"] == axis, "score axis/order")
        keys(row["values"], IDS)
        for number in row["values"].values():
            require(type(number) in (int, float) and math.isfinite(number), "invalid scalar score")
        out[axis] = {k: float(v) for k, v in row["values"].items()}
    return out


def _grade_values(mapping, choices, score_export):
    scores = _validate_scores(score_export)
    items = []
    for row in mapping["rows"]:
        first, second = scores[row["axis"]][row["first_id"]], scores[row["axis"]][row["second_id"]]
        gap = first - second
        require(math.isfinite(gap), "nonfinite scalar gap")
        truth = "FIRST" if gap > 0 else "SECOND" if gap < 0 else "TIE"
        choice = choices[row["rater"]][row["item_id"]]
        items.append({**row, "choice": choice, "truth": truth,
                      "score_first": first, "score_second": second, "gap": gap,
                      "absolute_gap": abs(gap), "credit": 0.5 if gap == 0 else float(choice == truth)})
    def summary(rows):
        return {"credit": sum(r["credit"] for r in rows), "total": len(rows),
                "correct": sum(r["credit"] == 1 for r in rows),
                "incorrect": sum(r["credit"] == 0 for r in rows),
                "exact_ties": sum(r["truth"] == "TIE" for r in rows),
                "constant_FIRST_credit": sum(0.5 if r["truth"] == "TIE" else float(r["truth"] == "FIRST") for r in rows),
                "constant_SECOND_credit": sum(0.5 if r["truth"] == "TIE" else float(r["truth"] == "SECOND") for r in rows)}
    arms = {a: summary([r for r in items if r["arm"] == a]) for a in ARMS}
    per_axis = {x: {a: summary([r for r in items if r["axis"] == x and r["arm"] == a]) for a in ARMS} for x in AXES}
    differences = {f"{a}_minus_{b}": {"total_credit": arms[a]["credit"] - arms[b]["credit"],
        "per_axis_credit": {x: per_axis[x][a]["credit"] - per_axis[x][b]["credit"] for x in AXES}}
        for a, b in (("A", "B"), ("A", "C"), ("B", "C"))}
    return {"schema": "jlens_fresh_grades_v1", "arms": arms, "per_axis": per_axis,
            "paired_differences": differences, "items": items}


def grade(output, *, packets, manifest_sha256, responses, lock_sha256,
          lock_commit, scores, scores_sha256):
    attempt = Attempt(output, "grade")
    try:
        manifest, public = _bundle(packets, manifest_sha256)
        raw_lock, _ = read_bytes(Path(responses) / "lock.json", lock_sha256)
        lock = decode(raw_lock)
        keys(lock, ("schema", "sealed_utc", "source_sha256", "count", "packet_manifest_sha256", "responses"))
        require(lock["schema"] == "jlens_fresh_response_lock_v1" and type(lock["count"]) is int and
                lock["count"] == 144 and lock["source_sha256"] == source_sha() and
                lock["packet_manifest_sha256"] == manifest_sha256, "response lock binding")
        keys(lock["responses"], RATERS)
        choices, committed = {}, {"lock.json": raw_lock}
        for rater in RATERS:
            raw = _member(responses, lock["responses"][rater], f"{rater}.json")
            choices[rater] = validate_response(decode(raw), public[rater])
            committed[f"{rater}.json"] = raw
        _committed_lock(responses, lock_commit, committed)
        # Validate source-bound public/private construction BEFORE any score I/O.
        loaded = {}
        for kind, receipt in manifest["inputs"].items():
            keys(receipt, ("path", "size_bytes", "sha256"))
            if kind in FROZEN:
                require(receipt["sha256"] == FROZEN[kind], "frozen input binding")
            raw, actual = read_bytes(receipt["path"], receipt["sha256"])
            require(actual["size_bytes"] == receipt["size_bytes"], "input byte size")
            if kind != "protocol":
                loaded[kind] = decode(raw)
        expected_public, expected_map = make_packets(loaded["references"], loaded["dataset"], loaded["pairs"])
        mapping = decode(_member(packets, manifest["private_map"], "private-map.json"))
        require(public == expected_public and mapping == expected_map, "packet/source/map mismatch")
        scalar_export, sr = read_json(scores, scores_sha256)
        result = _grade_values(mapping, choices, scalar_export)
        result["provenance"] = {"source_sha256": source_sha(), "packet_manifest_sha256": manifest_sha256,
                                "lock_sha256": lock_sha256, "lock_commit": lock_commit,
                                "responses": lock["responses"], "scores": sr,
                                "inputs": manifest["inputs"], "python": sys.version}
        return attempt.write("grades.json", result)
    except Exception as error:
        attempt.fail(error)
        raise
