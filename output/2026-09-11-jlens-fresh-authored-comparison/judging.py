"""Inert JSON-only fresh-authored packet, response-lock and grader.

Four readers x four axes x sixteen pairs = 256 choices; arms A/C only.
Package reads a committed metadata release, but never its score/forward files.
Grade validates all five committed response-lock blobs before score JSON.
Future acquisition hashes live in the release, not source edits. No old producer
imports, model, tokenizer, NumPy, archive, network or import-time I/O.
All stages consume exclusive directories even on failure; no automatic retry.
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


SEED = 20260917
AXES = tuple(f"PC{i}" for i in range(1, 5))
ARMS = ("A", "C")
RATERS = tuple(f"rater{i}" for i in range(1, 5))
CELLS = tuple((f"fa{i:02}", t) for i in range(1, 9) for t in ("active", "passive"))
IDS = tuple(f"{p}-{t}-{pole}" for p, t in CELLS for pole in ("O", "P"))
PAIR_IDS = tuple(f"{p}-{t}" for p, t in CELLS)
ALLOCATION = {"rater1": ("A", "A", "A", "A"), "rater2": ("C", "C", "C", "C"),
              "rater3": ("A", "A", "A", "A"), "rater4": ("C", "C", "C", "C")}
COHORT = {r: 1 if i < 2 else 2 for i, r in enumerate(RATERS)}
PROMPT = ("Using only the two reference descriptions, which of the two prefixes "
          "should have the higher value on this direction? Choose FIRST or SECOND "
          "even if uncertain.")
FROZEN = {
    "protocol": "a66353d5838df1ab5d9e0a5ff034edcda12bbd4db1a80ba6a256b4760172b71b",
    "references": "0a678ad8d58a6b17fc2f0c3ac197d47be83e163750b2408f52759e23a0e6c44c",
    "dataset": "80749619a5bdbfd9506b8d453cc7f54d8f50c6eff73729f6a148152c0c0b35bf",
    "pairs": "ec361bee2a15ec796230824c2e9125f46cebe76316a6b50ee45904cb2be02912",
}
INPUT_NAMES = ("references", "dataset", "pairs", "protocol", "tokens", "preflight")
ARTIFACT_NAMES = (*INPUT_NAMES, "producer", "forward_receipt", "scores")
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


def validate_receipt(value):
    keys(value, ("path", "size_bytes", "sha256"))
    require(type(value["path"]) is str and Path(value["path"]).is_absolute() and
            ".." not in Path(value["path"]).parts, "absolute artifact path required")
    require(type(value["size_bytes"]) is int and 0 < value["size_bytes"] <= MAX_JSON,
            "bounded positive artifact size required")
    digest(value["sha256"])


def validate_artifacts(artifacts):
    keys(artifacts, ARTIFACT_NAMES)
    for receipt in artifacts.values():
        validate_receipt(receipt)
    require(len({r["path"] for r in artifacts.values()}) == len(ARTIFACT_NAMES),
            "duplicate release artifact path")
    for name, pin in FROZEN.items():
        require(artifacts[name]["sha256"] == pin, "frozen design/reference pin required")


def _declared(receipt):
    validate_receipt(receipt)
    raw, actual = read_bytes(receipt["path"], receipt["sha256"])
    require(actual["size_bytes"] == receipt["size_bytes"], "released artifact size mismatch")
    return raw


def _release(filename, expected, commit):
    raw, receipt = read_bytes(filename, expected)
    _committed_files({Path(receipt["path"]): raw}, commit)
    value = decode(raw)
    keys(value, ("schema", "artifacts"))
    require(value["schema"] == "jlens_fresh_authored_release_v1", "release schema")
    validate_artifacts(value["artifacts"])
    return value["artifacts"], {"receipt": receipt, "commit": commit}


def _inputs(artifacts):
    # Deliberately no producer, forward receipt or score file access here.
    loaded = {}
    for name in INPUT_NAMES:
        raw = _declared(artifacts[name])
        if name != "protocol":
            loaded[name] = decode(raw)
    return loaded


def _scope_inputs(receipt, artifacts):
    pins = receipt.get("input_pins")
    require(type(pins) is dict and all(type(k) is str for k in pins), "input pin mapping")
    for value in pins.values():
        digest(value)
    require(all(artifacts[k]["sha256"] in pins.values() for k in ("protocol", "dataset", "pairs")),
            "producer design-input scope mismatch")


def _forward_scope(receipt, artifacts):
    require(type(receipt) is dict and receipt.get("schema") == "jlens_fresh_authored_forwards_receipt_v1" and
            receipt.get("status") == "complete" and
            receipt.get("source_sha256") == artifacts["producer"]["sha256"] and
            receipt.get("preflight_receipt_sha256") == artifacts["preflight"]["sha256"],
            "forward source/preflight binding")
    require(type(receipt.get("forward_count")) is int and receipt["forward_count"] == 32 and
            receipt.get("capture_locations") == ["verb"] and receipt.get("parameters_unchanged") is True and
            receipt.get("pca_refit") is False and receipt.get("tokenizer_loaded") is False and
            type(receipt.get("reference_decodes")) is int and receipt["reference_decodes"] == 0,
            "forward measurement scope")
    _scope_inputs(receipt, artifacts)
    outputs = receipt.get("outputs")
    require(type(outputs) is list and len(outputs) == 4, "forward output roster")
    by_name = {}
    for output in outputs:
        keys(output, ("path", "sha256"))
        require(output["path"] in ("features.npz", "inputs.json", "scores.json", "gaps.json") and
                output["path"] not in by_name, "forward output name/duplicate")
        by_name[output["path"]] = digest(output["sha256"])
    require(by_name["scores.json"] == artifacts["scores"]["sha256"], "forward score-output binding")


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


def validate_inputs(references, dataset, pairs, tokens, preflight, artifacts):
    validate_artifacts(artifacts)
    keys(dataset, ("rows",))
    keys(pairs, ("pairs",))
    rows, edges = dataset["rows"], pairs["pairs"]
    require(type(rows) is list and len(rows) == 32, "dataset roster")
    expected_pairs = [{"id": f"{p}-{t}", "content_pair": p, "template": t,
                      "observation_id": f"{p}-{t}-O", "provision_id": f"{p}-{t}-P"}
                     for p, t in CELLS]
    require(edges == expected_pairs, "exact sixteen content/template pairs required")
    keys(tokens, ("schema", "records"))
    require(tokens["schema"] == "jlens_fresh_authored_tokens_v1" and
            type(tokens["records"]) is list and len(tokens["records"]) == 32, "saved token roster")
    require(type(preflight) is dict and
            preflight.get("schema") == "jlens_fresh_authored_token_preflight_receipt_v1" and
            preflight.get("status") == "complete" and
            preflight.get("source_sha256") == artifacts["producer"]["sha256"] and
            type(preflight.get("count")) is int and preflight["count"] == 32 and
            preflight.get("token_limit") == 96 and preflight.get("model_loaded") is False and
            preflight.get("scientific_array_loaded") is False, "preflight receipt scope/source")
    require(preflight.get("outputs") == [{"path": "tokens.json", "sha256": artifacts["tokens"]["sha256"]}],
            "preflight token-output binding")
    _scope_inputs(preflight, artifacts)
    for row, record, ident in zip(rows, tokens["records"], IDS):
        keys(row, ("id", "pair_id", "template", "pole", "text", "verb", "verb_span"))
        pair, template, pole = ident.split("-")
        require((row["id"], row["pair_id"], row["template"], row["pole"]) ==
                (ident, pair, template, pole), "row identity/order")
        text, span, verb = row["text"], row["verb_span"], row["verb"]
        string(text)
        require(text.isascii() and text == " ".join(text.split()) and text.endswith(".") and
                len(text.split()) == (5 if template == "active" else 7), "full text template")
        require(type(span) is list and len(span) == 2 and all(type(x) is int for x in span) and
                0 <= span[0] < span[1] < len(text) and type(verb) is str and verb.isalpha() and
                text[slice(*span)] == verb, "verb character span")
        require(len(text[:span[1]].split()) == (3 if template == "active" else 4) and
                text[:span[0]].endswith(" "), "exact verb-truncated target")
        keys(record, (*row.keys(), "input_ids", "attention_mask", "offset_mapping",
                      "captured_positions", "selected_substrings", "layer"))
        require(all(record[k] == v for k, v in row.items()) and
                type(record["layer"]) is int and record["layer"] == 11, "saved row/layer binding")
        ids, mask, offsets = record["input_ids"], record["attention_mask"], record["offset_mapping"]
        require(type(ids) is list and 1 <= len(ids) <= 96 and
                all(type(v) is int and 0 <= v < 248320 for v in ids), "saved token IDs")
        require(type(mask) is list and len(mask) == len(ids) and
                all(type(v) is int and v == 1 for v in mask), "saved all-valid mask")
        require(type(offsets) is list and len(offsets) == len(ids) and
                all(type(v) is list and len(v) == 2 and all(type(x) is int for x in v) and
                    0 <= v[0] < v[1] <= len(text) for v in offsets) and
                all(a[1] <= b[0] for a, b in zip(offsets, offsets[1:])), "saved offsets")
        overlap = [i for i, (a, b) in enumerate(offsets) if a < span[1] and b > span[0]]
        require(overlap and overlap == list(range(overlap[0], overlap[-1]+1)), "verb token coverage")
        first, last = offsets[overlap[0]], offsets[overlap[-1]]
        require(first[0] <= span[0] and
                (first[0] == span[0] or text[first[0]:span[0]].isspace()) and last[1] == span[1] and
                all(offsets[i][1] == offsets[i+1][0] for i in overlap[:-1]), "exact final verb subtoken")
        positions = {"verb": overlap[-1]}
        keys(record["captured_positions"], ("verb",))
        require(all(type(v) is int for v in record["captured_positions"].values()) and
                record["captured_positions"] == positions and positions["verb"] < len(ids)-1 and
                offsets[-1][0] <= len(text)-1 and offsets[-1][1] == len(text), "saved capture positions")
        require(record["selected_substrings"] ==
                {k: text[slice(*offsets[v])] for k, v in positions.items()}, "selected substring binding")
    require(len({r["text"] for r in rows}) == 32, "duplicate full texts")
    keys(references, ("schema", "axes"))
    require(references["schema"] == "jlens_fresh_references_v1", "reference schema")
    require(type(references["axes"]) is list and len(references["axes"]) == 4, "reference axes")
    for row, axis in zip(references["axes"], AXES):
        keys(row, ("axis", "A", "B", "C"))
        require(row["axis"] == axis, "reference axis/order")
        for arm in ("A", "B", "C"):
            keys(row[arm], ("positive", "negative"))
            for val in row[arm].values():
                if arm == "C":
                    string(val)
                else:
                    require(type(val) is list and len(val) == 12, "reference token count")
                    for token in val:
                        # Empty decoded fragments are valid strings and must not be removed.
                        require(type(token) is str and len(token) <= 4096, "reference token string")


def make_packets(references, dataset, pairs, tokens, preflight, artifacts):
    """Pure construction for fixtures and package; accepts no score/key input."""
    validate_inputs(references, dataset, pairs, tokens, preflight, artifacts)
    rows, pairs = dataset["rows"], pairs["pairs"]
    rng = random.Random(SEED)
    swaps = {(a, p["id"]): rng.choice((False, True)) for a in AXES for p in pairs}
    block_ids, item_ids = list(range(1, 17)), list(range(1, 257))
    rng.shuffle(block_ids)
    rng.shuffle(item_ids)
    text = {r["id"]: r["text"][:r["verb_span"][1]] for r in rows}
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
                first, second = pair["observation_id"], pair["provision_id"]
                if swaps[axis, pair["id"]] != (COHORT[rater] == 2):
                    first, second = second, first
                item = f"Q{item_ids.pop():03}"
                block["comparisons"].append({"item_id": item, "first": text[first], "second": text[second]})
                private.append({"item_id": item, "block_id": block_id, "rater": rater,
                                "axis": axis, "arm": arm, "cohort": COHORT[rater], "pair_id": pair["id"], "content_pair": pair["content_pair"],
                                "template": pair["template"],
                                "first_id": first, "second_id": second})
            rng.shuffle(block["comparisons"])
            blocks.append(block)
        rng.shuffle(blocks)
        packets[rater] = {"schema": "jlens_fresh_authored_public_v1", "packet_id": f"R{rng.getrandbits(64):016x}",
                          "prompt": PROMPT, "blocks": blocks}
    return packets, {"schema": "jlens_fresh_authored_private_map_v1", "rows": private}


def _public_ids(packet):
    keys(packet, ("schema", "packet_id", "prompt", "blocks"))
    require(packet["schema"] == "jlens_fresh_authored_public_v1" and packet["prompt"] == PROMPT,
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
        require(type(block["comparisons"]) is list and len(block["comparisons"]) == 16, "public pair count")
        for item in block["comparisons"]:
            keys(item, ("item_id", "first", "second"))
            require(type(item["item_id"]) is str and re.fullmatch(r"Q\d{3}", item["item_id"]), "item ID")
            string(item["first"])
            string(item["second"])
            items.append(item["item_id"])
    require(len(set(blocks)) == 4 and len(set(items)) == 64, "duplicate public IDs")
    return set(items)


def validate_response(response, packet):
    ids = _public_ids(packet)
    keys(response, ("schema", "packet_id", "responses"))
    require(response["schema"] == "jlens_fresh_authored_responses_v1" and
            response["packet_id"] == packet["packet_id"], "response packet/schema")
    require(type(response["responses"]) is list and len(response["responses"]) == 64, "response count")
    choices = {}
    for row in response["responses"]:
        keys(row, ("item_id", "choice"))
        require(type(row["item_id"]) is str and row["item_id"] in ids and
                row["item_id"] not in choices, "response ID/allocation")
        require(type(row["choice"]) is str and row["choice"] in ("FIRST", "SECOND"), "response choice")
        choices[row["item_id"]] = row["choice"]
    require(set(choices) == ids, "response completeness")
    return choices


def package(output, *, release, release_sha256, release_commit):
    """Build packets from a committed metadata release; never open its score file."""
    attempt = Attempt(output, "package")
    try:
        artifacts, release_binding = _release(release, release_sha256, release_commit)
        loaded = _inputs(artifacts)
        packets, mapping = make_packets(loaded["references"], loaded["dataset"], loaded["pairs"],
                                         loaded["tokens"], loaded["preflight"], artifacts)
        public = {r: attempt.write(f"public/{r}.json", packets[r]) for r in RATERS}
        private = attempt.write("private-map.json", mapping)
        return attempt.write("manifest.json", {"schema": "jlens_fresh_authored_packets_v1", "seed": SEED,
            "python": sys.version, "source_sha256": source_sha(),
            "inputs": {name: artifacts[name] for name in INPUT_NAMES}, "release": release_binding,
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
    keys(manifest, ("schema", "seed", "python", "source_sha256", "inputs", "release", "public", "private_map"))
    require(manifest["schema"] == "jlens_fresh_authored_packets_v1" and type(manifest["seed"]) is int and
            manifest["seed"] == SEED and manifest["python"] == sys.version and
            manifest["source_sha256"] == source_sha(), "packet environment/source/schema")
    keys(manifest["inputs"], INPUT_NAMES)
    keys(manifest["release"], ("receipt", "commit"))
    validate_receipt(manifest["release"]["receipt"])
    keys(manifest["public"], RATERS)
    packets = {r: decode(_member(directory, manifest["public"][r], f"public/{r}.json")) for r in RATERS}
    ids = [_public_ids(packets[r]) for r in RATERS]
    require(len(set.union(*ids)) == 256 and len({p["packet_id"] for p in packets.values()}) == 4,
            "cross-packet duplicate IDs")
    require(len({b["block_id"] for p in packets.values() for b in p["blocks"]}) == 16,
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
        return attempt.write("lock.json", {"schema": "jlens_fresh_authored_response_lock_v1",
            "sealed_utc": utc(), "source_sha256": source_sha(), "count": 256,
            "packet_manifest_sha256": digest(manifest_sha256), "responses": copies})
    except Exception as error:
        attempt.fail(error)
        raise


def _git(cwd, *args):
    return subprocess.run(["git", "-C", str(cwd), *args], check=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10).stdout


def _committed_files(files, commit):
    require(type(commit) is str and re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", commit),
            "immutable commit required")
    require(type(files) is dict and files, "committed files required")
    directory = next(iter(files)).parent.resolve(strict=True)
    root = Path(_git(directory, "rev-parse", "--show-toplevel").decode().strip()).resolve(strict=True)
    require(_git(root, "cat-file", "-t", commit).strip() == b"commit", "object is not a commit")
    for filename, raw in files.items():
        rel = filename.resolve(strict=True).relative_to(root).as_posix()
        spec = f"{commit}:{rel}"
        size = int(_git(root, "cat-file", "-s", spec).strip())
        require(size == len(raw) and size <= MAX_JSON, "committed lock size mismatch")
        require(_git(root, "cat-file", "blob", spec) == raw, "committed lock bytes mismatch")


def _committed_lock(directory, commit, files):
    require(set(files) == {"lock.json", *[f"{r}.json" for r in RATERS]},
            "lock and all four response blobs required")
    _committed_files({Path(directory) / name: raw for name, raw in files.items()}, commit)


def _validate_scores(value):
    keys(value, ("schema", "locations"))
    require(value["schema"] == "jlens_fresh_authored_scores_v1", "score schema")
    keys(value["locations"], ("verb",))
    out = {}
    for location, axes in value["locations"].items():
        require(type(axes) is list and len(axes) == 4, "score axes")
        out[location] = {}
        for row, axis in zip(axes, AXES):
            keys(row, ("axis", "values"))
            require(row["axis"] == axis, "score axis/order")
            keys(row["values"], IDS)
            require(list(row["values"]) == list(IDS), "score row order")
            for number in row["values"].values():
                require(type(number) in (int, float) and math.isfinite(number), "invalid scalar score")
            out[location][axis] = {k: float(v) for k, v in row["values"].items()}
    return out["verb"]


def _grade_roster(mapping, choices):
    """Defend complete cohort/axis/arm/pair coverage even in pure fixtures."""
    keys(mapping, ("schema", "rows"))
    require(mapping["schema"] == "jlens_fresh_authored_private_map_v1" and
            type(mapping["rows"]) is list and len(mapping["rows"]) == 256, "grade map roster")
    keys(choices, RATERS)
    cells, orientations, item_ids, by_reader = set(), {}, set(), {r: set() for r in RATERS}
    for row in mapping["rows"]:
        keys(row, ("item_id", "block_id", "rater", "axis", "arm", "cohort",
                   "pair_id", "content_pair", "template", "first_id", "second_id"))
        r, axis, arm, cohort = row["rater"], row["axis"], row["arm"], row["cohort"]
        require(r in RATERS and axis in AXES and arm == ALLOCATION[r][AXES.index(axis)] and
                type(cohort) is int and cohort == COHORT[r], "grade allocation")
        require(type(row["pair_id"]) is str and row["pair_id"] in PAIR_IDS, "grade pair ID")
        pi = PAIR_IDS.index(row["pair_id"])
        require((row["content_pair"], row["template"]) == CELLS[pi], "grade content/template")
        require({row["first_id"], row["second_id"]} == {IDS[2*pi], IDS[2*pi+1]}, "grade pair")
        cell = (axis, arm, cohort, row["pair_id"])
        require(cell not in cells and row["item_id"] not in item_ids, "duplicate grade cell/ID")
        cells.add(cell)
        item_ids.add(row["item_id"])
        by_reader[r].add(row["item_id"])
        target = (axis, cohort, row["pair_id"])
        orientation = (row["first_id"], row["second_id"])
        require(orientations.setdefault(target, orientation) == orientation, "common cohort orientation")
    require(item_ids == {f"Q{i:03}" for i in range(1, 257)}, "exact 256 grade IDs")
    for rater in RATERS:
        keys(choices[rater], by_reader[rater])
        require(len(by_reader[rater]) == 64 and
                all(type(v) is str and v in ("FIRST", "SECOND") for v in choices[rater].values()),
                "grade choices")
    for axis in AXES:
        for pair_id in PAIR_IDS:
            require(orientations[axis, 1, pair_id] ==
                    orientations[axis, 2, pair_id][::-1], "opposite cohort orientation")


def _grade_values(mapping, choices, score_export):
    _grade_roster(mapping, choices)
    scores = _validate_scores(score_export)
    items = []
    for row in mapping["rows"]:
        first, second = scores[row["axis"]][row["first_id"]], scores[row["axis"]][row["second_id"]]
        gap = first - second
        require(math.isfinite(gap), "nonfinite scalar gap")
        truth = "FIRST" if gap > 0 else "SECOND" if gap < 0 else "TIE"
        choice = choices[row["rater"]][row["item_id"]]
        items.append({**row, "choice": choice, "truth": truth,
                      "chosen_id": row["first_id"] if choice == "FIRST" else row["second_id"],
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
    per_reader = {r: {
        "cohort": COHORT[r], "summary": summary([v for v in items if v["rater"] == r]),
        "per_axis": {x: {"arm": ALLOCATION[r][ai],
                         **summary([v for v in items if v["rater"] == r and v["axis"] == x])}
                     for ai, x in enumerate(AXES)}} for r in RATERS}
    differences = {f"{a}_minus_{b}": {"total_credit": arms[a]["credit"] - arms[b]["credit"],
        "per_axis_credit": {x: per_axis[x][a]["credit"] - per_axis[x][b]["credit"] for x in AXES}}
        for a, b in (("A", "C"),)}
    by_cell = {(v["axis"], v["arm"], v["cohort"], v["pair_id"]): v for v in items}
    agreement = {"per_axis": {}, "arms": {}}
    for axis in AXES:
        agreement["per_axis"][axis] = {}
        for arm in ARMS:
            rows = []
            for pair_id in PAIR_IDS:
                v1, v2 = by_cell[axis, arm, 1, pair_id], by_cell[axis, arm, 2, pair_id]
                rows.append({"pair_id": pair_id, "cohort1_rater": v1["rater"],
                             "cohort2_rater": v2["rater"], "cohort1_chosen_id": v1["chosen_id"],
                             "cohort2_chosen_id": v2["chosen_id"],
                             "agree": v1["chosen_id"] == v2["chosen_id"]})
            agree = sum(v["agree"] for v in rows)
            agreement["per_axis"][axis][arm] = {"agree": agree, "disagree": 16-agree,
                                                "total": 16, "items": rows}
            require(per_axis[axis][arm]["total"] == 32 and
                    per_axis[axis][arm]["constant_FIRST_credit"] ==
                    per_axis[axis][arm]["constant_SECOND_credit"] == 16, "axis balance")
    for arm in ARMS:
        agree = sum(agreement["per_axis"][axis][arm]["agree"] for axis in AXES)
        agreement["arms"][arm] = {"agree": agree, "disagree": 64-agree, "total": 64}
        require(arms[arm]["total"] == 128 and arms[arm]["constant_FIRST_credit"] ==
                arms[arm]["constant_SECOND_credit"] == 64, "arm balance")
    def breakdown(field, values):
        return {str(v): {a: summary([r for r in items if r[field] == v and r["arm"] == a])
                        for a in ARMS} for v in values}
    per_cohort = breakdown("cohort", (1, 2))
    per_template = breakdown("template", ("active", "passive"))
    cohort_axis = {str(c): {x: {a: summary([r for r in items if r["cohort"] == c and
                    r["axis"] == x and r["arm"] == a]) for a in ARMS} for x in AXES} for c in (1, 2)}
    template_axis = {t: {x: {a: summary([r for r in items if r["template"] == t and
                    r["axis"] == x and r["arm"] == a]) for a in ARMS} for x in AXES} for t in ("active", "passive")}
    diff = differences["A_minus_C"]
    diff["per_cohort"] = {c: {"total_credit": v["A"]["credit"]-v["C"]["credit"],
        "per_axis_credit": {x: cohort_axis[c][x]["A"]["credit"]-cohort_axis[c][x]["C"]["credit"]
                            for x in AXES}} for c, v in per_cohort.items()}
    diff["per_template"] = {t: {"total_credit": v["A"]["credit"]-v["C"]["credit"],
        "per_axis_credit": {x: template_axis[t][x]["A"]["credit"]-template_axis[t][x]["C"]["credit"]
                            for x in AXES}} for t, v in per_template.items()}
    primary = {"axis": "PC4", "A_minus_C_credit_out_of_32": diff["per_axis_credit"]["PC4"],
               "A_reader_credit_out_of_16": {r: per_reader[r]["per_axis"]["PC4"]["credit"]
                   for r in RATERS if ALLOCATION[r][3] == "A"},
               "A_minus_C_by_cohort_out_of_16": {c: d["per_axis_credit"]["PC4"]
                   for c, d in diff["per_cohort"].items()}}
    primary["each_A_reader_exceeds_8"] = all(v > 8 for v in primary["A_reader_credit_out_of_16"].values())
    primary["A_minus_C_positive_each_cohort"] = all(v > 0 for v in primary["A_minus_C_by_cohort_out_of_16"].values())
    return {"schema": "jlens_fresh_authored_grades_v1", "primary": primary,
            "per_cohort": per_cohort, "per_template": per_template,
            "cohort_axis": cohort_axis, "template_axis": template_axis, "arms": arms, "per_axis": per_axis,
            "per_reader": per_reader, "paired_differences": differences, "agreement": agreement,
            "items": items}


def verify_response_lock(*, packets, manifest_sha256, responses, lock_sha256, lock_commit):
    """Read-only checker gate: validates 256 public choices and five committed blobs.

    No private map, release inputs, forward receipt or scalar contents are opened.
    This is a response/source lock check, not independent scientific validation.
    """
    manifest, public = _bundle(packets, manifest_sha256)
    raw_lock, _ = read_bytes(Path(responses) / "lock.json", lock_sha256)
    lock = decode(raw_lock)
    keys(lock, ("schema", "sealed_utc", "source_sha256", "count", "packet_manifest_sha256", "responses"))
    require(lock["schema"] == "jlens_fresh_authored_response_lock_v1" and type(lock["count"]) is int and
            lock["count"] == 256 and lock["source_sha256"] == source_sha() and
            lock["packet_manifest_sha256"] == manifest_sha256, "response lock binding")
    keys(lock["responses"], RATERS)
    choices, committed = {}, {"lock.json": raw_lock}
    for rater in RATERS:
        raw = _member(responses, lock["responses"][rater], f"{rater}.json")
        choices[rater] = validate_response(decode(raw), public[rater])
        committed[f"{rater}.json"] = raw
    _committed_lock(responses, lock_commit, committed)
    return {"manifest": manifest, "public": public, "lock": lock, "choices": choices}


def grade(output, *, packets, manifest_sha256, responses, lock_sha256, lock_commit):
    attempt = Attempt(output, "grade")
    try:
        verified = verify_response_lock(packets=packets, manifest_sha256=manifest_sha256,
            responses=responses, lock_sha256=lock_sha256, lock_commit=lock_commit)
        manifest, public, lock, choices = (verified[k] for k in ("manifest", "public", "lock", "choices"))
        # Validate source-bound public/private construction BEFORE any score I/O.
        binding = manifest["release"]
        artifacts, actual_release = _release(binding["receipt"]["path"],
                                             binding["receipt"]["sha256"], binding["commit"])
        require(actual_release == binding and
                manifest["inputs"] == {name: artifacts[name] for name in INPUT_NAMES}, "release/input binding")
        loaded = _inputs(artifacts)
        expected_public, expected_map = make_packets(loaded["references"], loaded["dataset"], loaded["pairs"],
                                                     loaded["tokens"], loaded["preflight"], artifacts)
        mapping = decode(_member(packets, manifest["private_map"], "private-map.json"))
        require(public == expected_public and mapping == expected_map, "packet/source/map mismatch")
        forward = decode(_declared(artifacts["forward_receipt"]))
        _forward_scope(forward, artifacts)
        scalar_export = decode(_declared(artifacts["scores"]))
        result = _grade_values(mapping, choices, scalar_export)
        result["provenance"] = {"source_sha256": source_sha(), "packet_manifest_sha256": manifest_sha256,
                                "lock_sha256": lock_sha256, "lock_commit": lock_commit,
                                "responses": lock["responses"], "scores": artifacts["scores"],
                                "inputs": manifest["inputs"], "python": sys.version,
                                "release": binding, "producer": artifacts["producer"],
                                "forward_receipt": artifacts["forward_receipt"], "graded_location": "verb"}
        return attempt.write("grades.json", result)
    except Exception as error:
        attempt.fail(error)
        raise
