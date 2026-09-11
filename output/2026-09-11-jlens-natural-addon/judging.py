"""Inert JSON-only natural add-on packet, response-lock and grader.

Four readers x four axes x sixteen pairs = 256 choices; arms EA/E only.
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


SEED = 20260920
AXES = tuple(f"PC{i}" for i in range(1, 5))
ARMS = ("EA", "E")
TOPICS = ("astronomy", "cooking", "football", "programming")
RATERS = tuple(f"rater{i}" for i in range(1, 5))
PAIR_IDS = tuple(f"N{i:02}" for i in range(1, 17))
IDS = tuple(f"{p}-{side}" for p in PAIR_IDS for side in ("L", "R"))
ALLOCATION = {"rater1": ("EA",)*4, "rater2": ("E",)*4,
              "rater3": ("EA",)*4, "rater4": ("E",)*4}
COHORT = {r: 1 if i < 2 else 2 for i, r in enumerate(RATERS)}
PROMPT = ("example_prefix is a reference example for that pole. When supplied, "
          "direction_tokens are additional vocabulary-associated descriptions of that same pole. "
          "Using only the two reference descriptions, which of the two prefixes "
          "should have the higher value on this direction? Choose FIRST or SECOND "
          "even if uncertain.")
FROZEN = {
    "protocol": "a510feaf2dfdd5cf6541d974d2c305188e24d04a4e9e185ac73675970b8a6246",
    "references": "0a678ad8d58a6b17fc2f0c3ac197d47be83e163750b2408f52759e23a0e6c44c",
    "dataset": "d57300cb8511e409ff78e6ddc9fb6e1cf11b91bc629bf5492db8932cf585b129",
    "pairs": "0ae85708e0b5da2942e31aa96af4ffa642c1294ff4c8b78612ec2dfc0ee54725",
    "producer": "894f465e36d50f6df15229754367614d3ffc26f20301db3eb57f9535189bbc57",
    "tokens": "e87f996149ea1eb80b3fb3a6225257859f047aefaf7727beb1e792b863e0ef43",  # gitleaks:allow (sha256 of tokens artifact)
    "preflight": "675411e8428e5a8797101b0bed85a4604a4e59d21fe41b40c498a3e330c492fb",
}
INPUT_NAMES = ("references", "dataset", "pairs", "protocol", "tokens", "preflight", "producer")
ARTIFACT_NAMES = (*INPUT_NAMES, "forward_receipt", "scores")
MAX_JSON = 1024 * 1024
MAX_OUTPUT = 8 * 1024 * 1024
BASE_PRODUCER_SHA = "d2557036a3ebdd6ecbc61be89adf3582f16f084fda091e86f3faabe9ff39f1d0"
AMENDMENT_SHA = "5bcc78c413c319e1ccf329066d01699a9c56bfdbb7a8bb0aa4ecc5abb9a8a72c"
PREFLIGHT_PYTHON = "3.12.3 (main, Jun 19 2026, 12:46:00) [GCC 13.3.0]"
MEASUREMENT_PYTHON = "3.12.3 (main, Aug 31 2026, 10:18:26) [GCC 13.3.0]"
RUNTIME_PACKAGES = {"numpy": "1.26.4", "torch": "2.11.0+cu128",
                    "transformers": "5.5.0", "huggingface_hub": "1.8.0"}
FAILED_ADMISSION = {
    "attempt": "c313921dcdb033cd973dbbd8884d80feef1ca80e79362d02b453c8e9df2bf6ba",
    "failure": "7f81cc5fad6f71b7d8ee901a642100cce1040dae9ad8fc1c2d440db2616fe1c6"}


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
    require(value["schema"] == "jlens_natural_addon_release_v1", "release schema")
    validate_artifacts(value["artifacts"])
    return value["artifacts"], {"receipt": receipt, "commit": commit}


def _inputs(artifacts):
    # Deliberately no producer, forward receipt or score file access here.
    loaded = {}
    for name in INPUT_NAMES:
        if name == "producer":
            continue  # Metadata-only manifest binding for the public checker gate.
        raw = _declared(artifacts[name])
        if name != "protocol":
            loaded[name] = decode(raw)
    return loaded


def _scope_inputs(receipt, artifacts):
    pins = receipt.get("input_pins")
    require(type(pins) is dict and all(type(k) is str for k in pins), "input pin mapping")
    for value in pins.values():
        digest(value)
    require(all(pins.get(artifacts[k]["path"]) == artifacts[k]["sha256"]
                for k in ("protocol", "dataset", "pairs")),
            "producer design-input scope mismatch")


def _forward_scope(receipt, artifacts):
    require(type(receipt) is dict and receipt.get("schema") == "jlens_natural_addon_forwards_receipt_v1" and
            receipt.get("status") == "complete" and
            receipt.get("source_sha256") == artifacts["producer"]["sha256"] and
            receipt.get("preflight_receipt_sha256") == artifacts["preflight"]["sha256"],
            "forward source/preflight binding")
    require(receipt.get("base_producer_sha256") == BASE_PRODUCER_SHA and
            receipt.get("runtime_amendment_sha256") == AMENDMENT_SHA and
            receipt.get("failed_admission_sha256s") == FAILED_ADMISSION,
            "runtime amendment provenance")
    runtime = receipt.get("runtime", {})
    require(runtime.get("python") == MEASUREMENT_PYTHON and
            runtime.get("packages") == RUNTIME_PACKAGES, "amended measurement runtime")
    require(type(receipt.get("forward_count")) is int and receipt["forward_count"] == 32 and
            receipt.get("capture_locations") == ["prefix_end"] and receipt.get("parameters_unchanged") is True and
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
    rows, edges = dataset, pairs
    require(type(rows) is list and len(rows) == 32, "dataset roster")
    require(type(edges) is list and len(edges) == 16, "pair roster")
    for row, ident in zip(rows, IDS):
        keys(row, ("id", "topic", "prefix"))
        require(row["id"] == ident and row["topic"] in TOPICS, "row identity/order/topic")
        string(row["prefix"])
        require(row["prefix"] == " ".join(row["prefix"].split()) and
                len(row["prefix"].split()) == 16, "exact normalized sixteen-word prefix")
    require(len({r["prefix"] for r in rows}) == 32, "duplicate prefixes")
    ranks = []
    for i, pair in enumerate(edges):
        keys(pair, ("id", "left", "right", "topic", "candidate_id"))
        require(pair["id"] == PAIR_IDS[i] and pair["left"] == IDS[2*i] and pair["right"] == IDS[2*i+1]
                and pair["topic"] == rows[2*i]["topic"] == rows[2*i+1]["topic"], "fixed within-category pair")
        require(type(pair["candidate_id"]) is str and re.fullmatch(r"K[0-9]{3}", pair["candidate_id"]),
                "candidate ID format")
        ranks.append(int(pair["candidate_id"][1:]))
    require(all(1 <= n <= 32 for n in ranks) and ranks == sorted(set(ranks)), "candidate order/cap")
    keys(tokens, ("schema", "records"))
    require(tokens["schema"] == "jlens_natural_addon_tokens_v1" and
            type(tokens["records"]) is list and len(tokens["records"]) == 32, "saved token roster")
    require(type(preflight) is dict and
            preflight.get("schema") == "jlens_natural_addon_token_preflight_receipt_v1" and
            preflight.get("status") == "complete" and
            preflight.get("source_sha256") == BASE_PRODUCER_SHA and
            type(preflight.get("count")) is int and preflight["count"] == 32 and
            preflight.get("token_limit") == 96 and preflight.get("model_loaded") is False and
            preflight.get("scientific_array_loaded") is False, "preflight receipt scope/source")
    runtime = preflight.get("runtime", {})
    require(runtime.get("python") == PREFLIGHT_PYTHON and runtime.get("packages") == RUNTIME_PACKAGES,
            "original preflight runtime")
    require(preflight.get("outputs") == [{"path": "tokens.json", "sha256": artifacts["tokens"]["sha256"]}],
            "preflight token-output binding")
    _scope_inputs(preflight, artifacts)
    for row, record, ident in zip(rows, tokens["records"], IDS):
        text = row["prefix"]
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
                all(a[0] <= b[0] and a[1] <= b[1] for a, b in zip(offsets, offsets[1:])),
                "saved ordered offsets")
        require(offsets[0][0] == 0 and offsets[-1][1] == len(text) and
                all(a[1] >= b[0] or text[a[1]:b[0]].isspace()
                    for a, b in zip(offsets, offsets[1:])), "complete prefix coverage")
        positions = {"prefix_end": len(ids)-1}
        keys(record["captured_positions"], ("prefix_end",))
        require(all(type(v) is int for v in record["captured_positions"].values()) and
                record["captured_positions"] == positions, "saved final input position")
        require(record["selected_substrings"] ==
                {k: text[slice(*offsets[v])] for k, v in positions.items()}, "selected substring binding")
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
    rows = dataset
    rng = random.Random(SEED)
    swaps = {(a, p["id"]): rng.choice((False, True)) for a in AXES for p in pairs}
    block_ids, item_ids = list(range(1, 17)), list(range(1, 257))
    rng.shuffle(block_ids)
    rng.shuffle(item_ids)
    # Draw display order ONCE and reuse it in all four packets. Only target
    # sides reverse in cohort 2; anonymous IDs remain unique across packets.
    axis_order = list(AXES)
    rng.shuffle(axis_order)
    pair_order = {}
    for axis in AXES:
        pair_order[axis] = list(pairs)
        rng.shuffle(pair_order[axis])
    text = {r["id"]: r["prefix"] for r in rows}
    packets, private = {}, []
    for rater in RATERS:
        blocks = []
        for axis in axis_order:
            ai = AXES.index(axis)
            arm = ALLOCATION[rater][ai]
            block_id = f"B{block_ids.pop():03}"
            refs = references["axes"][ai]
            def reference(pole):
                result = {"example_prefix": refs["C"][pole]}
                if arm == "EA":
                    result["direction_tokens"] = list(refs["A"][pole])
                return result
            block = {"block_id": block_id, "positive_reference": reference("positive"),
                     "negative_reference": reference("negative"), "comparisons": []}
            for pair in pair_order[axis]:
                first, second = pair["left"], pair["right"]
                if swaps[axis, pair["id"]] != (COHORT[rater] == 2):
                    first, second = second, first
                item = f"Q{item_ids.pop():03}"
                block["comparisons"].append({"item_id": item, "first": text[first], "second": text[second]})
                private.append({"item_id": item, "block_id": block_id, "rater": rater,
                                "axis": axis, "arm": arm, "cohort": COHORT[rater], "pair_id": pair["id"],
                                "topic": pair["topic"], "candidate_id": pair["candidate_id"],
                                "first_id": first, "second_id": second})
            blocks.append(block)
        packets[rater] = {"schema": "jlens_natural_addon_public_v1", "packet_id": f"R{rng.getrandbits(64):016x}",
                          "prompt": PROMPT, "blocks": blocks}
    return packets, {"schema": "jlens_natural_addon_private_map_v1", "rows": private}


def _public_ids(packet):
    keys(packet, ("schema", "packet_id", "prompt", "blocks"))
    require(packet["schema"] == "jlens_natural_addon_public_v1" and packet["prompt"] == PROMPT,
            "public schema/prompt")
    require(type(packet["packet_id"]) is str and re.fullmatch(r"R[0-9a-f]{16}", packet["packet_id"]),
            "public packet ID")
    require(type(packet["blocks"]) is list and len(packet["blocks"]) == 4, "public block count")
    blocks, items, layouts = [], [], set()
    for block in packet["blocks"]:
        keys(block, ("block_id", "positive_reference", "negative_reference", "comparisons"))
        require(type(block["block_id"]) is str and re.fullmatch(r"B\d{3}", block["block_id"]), "block ID")
        blocks.append(block["block_id"])
        a, b = block["positive_reference"], block["negative_reference"]
        require(type(a) is dict and type(b) is dict and set(a) == set(b) and
                set(a) in ({"example_prefix"}, {"example_prefix", "direction_tokens"}), "public reference layout")
        layouts.add(tuple(sorted(a)))
        for ref in (a, b):
            string(ref["example_prefix"])
            if "direction_tokens" in ref:
                val = ref["direction_tokens"]
                require(type(val) is list and len(val) == 12 and
                        all(type(x) is str and len(x) <= 4096 for x in val), "public tokens")
        require(type(block["comparisons"]) is list and len(block["comparisons"]) == 16, "public pair count")
        for item in block["comparisons"]:
            keys(item, ("item_id", "first", "second"))
            require(type(item["item_id"]) is str and re.fullmatch(r"Q\d{3}", item["item_id"]), "item ID")
            string(item["first"])
            string(item["second"])
            items.append(item["item_id"])
    require(len(set(blocks)) == 4 and len(set(items)) == 64 and len(layouts) == 1,
            "duplicate public IDs or mixed reference layout")
    return set(items)


def validate_response(response, packet):
    ids = _public_ids(packet)
    keys(response, ("schema", "packet_id", "responses"))
    require(response["schema"] == "jlens_natural_addon_responses_v1" and
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
        return attempt.write("manifest.json", {"schema": "jlens_natural_addon_packets_v1", "seed": SEED,
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
    require(manifest["schema"] == "jlens_natural_addon_packets_v1" and type(manifest["seed"]) is int and
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
        return attempt.write("lock.json", {"schema": "jlens_natural_addon_response_lock_v1",
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
    require(value["schema"] == "jlens_natural_addon_scores_v1", "score schema")
    keys(value["locations"], ("prefix_end",))
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
    return out["prefix_end"]


def _grade_roster(mapping, choices):
    """Defend complete cohort/axis/arm/pair coverage even in pure fixtures."""
    keys(mapping, ("schema", "rows"))
    require(mapping["schema"] == "jlens_natural_addon_private_map_v1" and
            type(mapping["rows"]) is list and len(mapping["rows"]) == 256, "grade map roster")
    keys(choices, RATERS)
    cells, orientations, item_ids, by_reader = set(), {}, set(), {r: set() for r in RATERS}
    pair_metadata = {}
    for row in mapping["rows"]:
        keys(row, ("item_id", "block_id", "rater", "axis", "arm", "cohort",
                   "pair_id", "topic", "candidate_id", "first_id", "second_id"))
        r, axis, arm, cohort = row["rater"], row["axis"], row["arm"], row["cohort"]
        require(r in RATERS and axis in AXES and arm == ALLOCATION[r][AXES.index(axis)] and
                type(cohort) is int and cohort == COHORT[r], "grade allocation")
        require(type(row["pair_id"]) is str and row["pair_id"] in PAIR_IDS, "grade pair ID")
        pi = PAIR_IDS.index(row["pair_id"])
        require(row["topic"] in TOPICS and type(row["candidate_id"]) is str and
                re.fullmatch(r"K[0-9]{3}", row["candidate_id"]) and
                1 <= int(row["candidate_id"][1:]) <= 32, "grade category/candidate")
        metadata = (row["topic"], row["candidate_id"])
        require(pair_metadata.setdefault(row["pair_id"], metadata) == metadata, "pair metadata binding")
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
        for a, b in (("EA", "E"),)}
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
    per_category = breakdown("topic", TOPICS)
    cohort_axis = {str(c): {x: {a: summary([r for r in items if r["cohort"] == c and
                    r["axis"] == x and r["arm"] == a]) for a in ARMS} for x in AXES} for c in (1, 2)}
    category_axis = {t: {x: {a: summary([r for r in items if r["topic"] == t and
                    r["axis"] == x and r["arm"] == a]) for a in ARMS} for x in AXES} for t in TOPICS}
    diff = differences["EA_minus_E"]
    diff["per_cohort"] = {c: {"total_credit": v["EA"]["credit"]-v["E"]["credit"],
        "per_axis_credit": {x: cohort_axis[c][x]["EA"]["credit"]-cohort_axis[c][x]["E"]["credit"]
                            for x in AXES}} for c, v in per_cohort.items()}
    diff["per_category"] = {t: {"total_credit": v["EA"]["credit"]-v["E"]["credit"],
        "per_axis_credit": {x: category_axis[t][x]["EA"]["credit"]-category_axis[t][x]["E"]["credit"]
                            for x in AXES}} for t, v in per_category.items()}
    reader_controls = {}
    for r in RATERS:
        v = per_reader[r]["summary"]
        control = max(v["constant_FIRST_credit"], v["constant_SECOND_credit"])
        reader_controls[r] = {"arm": ALLOCATION[r][0], "credit_out_of_64": v["credit"],
                              "constant_FIRST_credit": v["constant_FIRST_credit"],
                              "constant_SECOND_credit": v["constant_SECOND_credit"],
                              "stronger_constant_credit": control, "exceeds_stronger_constant": v["credit"] > control}
    primary = {"scope": "all_four_axes", "EA_minus_E_credit_out_of_128": diff["total_credit"],
               "EA_minus_E_by_cohort_out_of_64": {c: d["total_credit"] for c, d in diff["per_cohort"].items()},
               "reader_controls": reader_controls}
    primary["EA_minus_E_positive_each_cohort"] = all(v > 0 for v in primary["EA_minus_E_by_cohort_out_of_64"].values())
    primary["each_EA_reader_exceeds_stronger_constant"] = all(
        v["exceeds_stronger_constant"] for v in reader_controls.values() if v["arm"] == "EA")
    primary["incremental_pilot_criterion_met"] = (
        primary["EA_minus_E_positive_each_cohort"] and primary["each_EA_reader_exceeds_stronger_constant"])
    gains_harm = []
    for cohort in (1, 2):
        for axis in AXES:
            for pair_id in PAIR_IDS:
                ea, e = by_cell[axis, "EA", cohort, pair_id], by_cell[axis, "E", cohort, pair_id]
                delta = ea["credit"] - e["credit"]
                gains_harm.append({"cohort": cohort, "axis": axis, "pair_id": pair_id, "topic": ea["topic"],
                                   "EA_item_id": ea["item_id"], "E_item_id": e["item_id"],
                                   "EA_credit": ea["credit"], "E_credit": e["credit"], "EA_minus_E_credit": delta,
                                   "outcome": "gain" if delta > 0 else "harm" if delta < 0 else "tie"})
    counts = {t: {"pairs": len({r["pair_id"] for r in items if r["topic"] == t}),
                  "texts": len({ident for r in items if r["topic"] == t for ident in (r["first_id"], r["second_id"])})}
              for t in TOPICS}
    return {"schema": "jlens_natural_addon_grades_v1", "primary": primary,
            "secondary_PC4": {"per_arm": per_axis["PC4"], "EA_minus_E_credit_out_of_32": diff["per_axis_credit"]["PC4"],
                              "EA_minus_E_by_cohort_out_of_16": {c: d["per_axis_credit"]["PC4"] for c, d in diff["per_cohort"].items()}},
            "per_cohort": per_cohort, "per_category": per_category, "category_counts": counts,
            "cohort_axis": cohort_axis, "category_axis": category_axis, "arms": arms, "per_axis": per_axis,
            "per_reader": per_reader, "paired_differences": differences, "agreement": agreement,
            "gains_harm": gains_harm, "items": items}


def verify_response_lock(*, packets, manifest_sha256, responses, lock_sha256, lock_commit):
    """Read-only checker gate: validates 256 public choices and five committed blobs.

    No private map, release inputs, forward receipt or scalar contents are opened.
    This is a response/source lock check, not independent scientific validation.
    """
    manifest, public = _bundle(packets, manifest_sha256)
    raw_lock, _ = read_bytes(Path(responses) / "lock.json", lock_sha256)
    lock = decode(raw_lock)
    keys(lock, ("schema", "sealed_utc", "source_sha256", "count", "packet_manifest_sha256", "responses"))
    require(lock["schema"] == "jlens_natural_addon_response_lock_v1" and type(lock["count"]) is int and
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
                                "forward_receipt": artifacts["forward_receipt"], "graded_location": "prefix_end"}
        return attempt.write("grades.json", result)
    except Exception as error:
        attempt.fail(error)
        raise
