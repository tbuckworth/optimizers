"""Import-inert saved-readout packaging; fixture is synthetic, build is explicit.

No model imports, model execution, grading, or external calls. Each judge sees
one method per axis; never send private-key.json or the source files.
"""

import argparse
import hashlib
import json
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "output/2026-09-10-j-lens-completions"
DEST = ROOT / "output/2026-09-10-j-lens-simple-comparison/packet"
SEED = 20260911
AXES = [f"PC{i}" for i in range(1, 5)] + [f"random{i}" for i in range(1, 5)]
METHODS = ["jlens", "plain"]
INPUT_PINS = {
    "output/2026-09-10-j-lens-completions/dataset.json": "b41b91381c1bfd6521ceb0b5d15c35ea28e0929d8eba57527b232a933a85cdd6",
    "output/2026-09-10-j-lens-completions/readouts.json": "fc7d4c8d9322132d78513f91cc9c7a101ed70c7df6472cd275bf29d202e48864",
    "output/2026-09-10-j-lens-completions/analysis_arrays.npz": "cbbe651ed084b36ed468a5932afcbc7e488c81ccdcc0fcd04bccf2fe9f6e7033",
    "output/2026-09-10-j-lens-completions/features.npz": "96e936942fd5cde63c9c482ef19a004ba3fee5adbc46676373772fa5e4123b61",
    "experiments/j_lens_completions.py": "ae04db8770538d278fca32b58542e8bb653f42785de39c2f2bf76d7a8fffe238",
}
EXPECTED_IDS = [f"{topic}-{i}-{style}" for topic in
                ["astronomy", "cooking", "football", "programming"]
                for i in [4, 5] for style in ["plain", "note"]]
PROMPT = (
    "Match the two token lists to the two prefixes. Which assignment best "
    "reflects their meaning? Choose A1_B2 or A2_B1 even if uncertain; report "
    "confidence as low, medium or high. A1_B2 means panel A matches prefix 1 "
    "and panel B matches prefix 2. A2_B1 means the reverse. Use the complete "
    "lists, including non-English tokens; no browsing or external tools. "
    "Return only choice and confidence."
)


def extrema(rows, values):
    """Use all supplied held-out rows. Ties use ascending opaque source ID."""
    if len(rows) != len(values) or len(rows) < 2:
        raise ValueError("Expected at least two aligned candidate rows")
    import math
    if not all(math.isfinite(float(v)) for v in values):
        raise ValueError("Nonfinite signed scores")
    lo = min(range(len(rows)), key=lambda i: (float(values[i]), rows[i]["id"]))
    hi = min(range(len(rows)), key=lambda i: (-float(values[i]), rows[i]["id"]))
    if float(values[lo]) == float(values[hi]):
        raise ValueError("Degenerate axis has no distinct extrema; do not replace")
    return hi, lo


def assigned_rater(axis, method):
    """Counterbalance method within each axis family; no counterpart exposure."""
    odd = int(axis[-1]) % 2 == 1
    first_method = "jlens" if (axis.startswith("PC") == odd) else "plain"
    return "rater1" if method == first_method else "rater2"


def verify_input_pins(pins=INPUT_PINS):
    checked = []
    for relative, expected in pins.items():
        actual = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Input hash mismatch: {relative}")
        checked.append({"path": relative, "sha256": actual})
    return checked


def make_packet(rows, axis_scores, readouts, axes=AXES, seed=SEED):
    """Pure transformation returning disjoint public and private dictionaries."""
    rng = random.Random(seed)
    endpoints = {axis: extrema(rows, axis_scores[axis]) for axis in axes}
    roster = [(axis, method) for axis in axes for method in METHODS]
    rng.shuffle(roster)
    cards, keys = [], []
    for number, (axis, method) in enumerate(roster, 1):
        group = readouts[f"activation_11_{method}"]
        if len(group["names"]) != len(set(group["names"])):
            raise ValueError("Duplicate readout names")
        lookup = dict(zip(group["names"], group["tokens"], strict=True))
        signs = ["+", "-"]
        rng.shuffle(signs)
        endpoint_signs = ["+", "-"]
        rng.shuffle(endpoint_signs)
        ix = dict(zip(["+", "-"], endpoints[axis], strict=True))
        panel_tokens = [lookup[axis + sign] for sign in signs]
        if any(len(t) != 12 or any(not isinstance(s, str) for s in t)
               for t in panel_tokens):
            raise ValueError("Every panel must preserve 12 decoded strings")
        candidates = [rows[ix[sign]] for sign in endpoint_signs]
        card_id = f"C{number:02d}"
        cards.append({
            "card_id": card_id,
            "instruction": PROMPT,
            "panels": dict(zip(["A", "B"], panel_tokens, strict=True)),
            "prefixes": dict(zip(["1", "2"], [r["prefix"] for r in candidates], strict=True)),
        })
        keys.append({
            "card_id": card_id, "axis": axis, "method": method,
            "assigned_rater": assigned_rater(axis, method),
            "correct_choice": "A1_B2" if signs == endpoint_signs else "A2_B1",
            "panel_signs": signs, "prefix_signs": endpoint_signs,
            "candidate_ids": [r["id"] for r in candidates],
            "same_content_extrema": candidates[0]["pair"] == candidates[1]["pair"],
            "endpoint_scores": [float(axis_scores[axis][ix[s]]) for s in endpoint_signs],
        })
    return {"cards": cards}, {"seed": seed, "keys": keys}


def load_saved_inputs():
    """Scientific extraction, reached only by the explicit build subcommand."""
    import numpy as np
    rows = json.loads((SOURCE / "dataset.json").read_text())
    selected = [i for i, row in enumerate(rows) if row["split"] == "heldout"]
    heldout = [rows[i] for i in selected]
    assert len(heldout) == 16 and len({r["pair"] for r in heldout}) == 8
    assert [r["id"] for r in heldout] == EXPECTED_IDS
    assert len({r["id"] for r in rows}) == 48
    assert {r["style"] for r in heldout} == {"plain", "note"}
    with np.load(SOURCE / "analysis_arrays.npz", allow_pickle=False) as archive:
        scores = archive["activation_11_scores"]
        mean = archive["activation_11_mean"]
    assert scores.shape == (48, 4) and mean.shape == (1024,)
    assert np.isfinite(scores).all() and np.isfinite(mean).all()
    with np.load(SOURCE / "features.npz", allow_pickle=False) as archive:
        activations = archive["activation_11"].astype(np.float64)
    assert activations.shape == (48, 1024)
    assert np.isfinite(activations).all()
    q = np.linalg.qr(np.random.default_rng(20260910).normal(size=(1024, 4)))[0]
    random_scores = (activations[selected] - mean) @ q
    assert np.isfinite(random_scores).all()
    by_axis = {f"PC{i+1}": scores[selected, i].tolist() for i in range(4)}
    by_axis.update({f"random{i+1}": random_scores[:, i].tolist() for i in range(4)})
    readouts = json.loads((SOURCE / "readouts.json").read_text())
    expected_names = EXPECTED_IDS + ["mean"] + [
        f"{kind}{i}{sign}" for kind in ["PC", "random"]
        for sign in ["+", "-"] for i in range(1, 5)]
    for method in METHODS:
        names = readouts[f"activation_11_{method}"]["names"]
        assert names == expected_names and len(set(names)) == 33
    return heldout, by_axis, readouts


def build():
    if DEST.exists() or DEST.is_symlink():
        raise FileExistsError(f"Packet target already exists: {DEST}")
    before_pins = verify_input_pins()
    rows, scores, readouts = load_saved_inputs()
    after_pins = verify_input_pins()
    assert before_pins == after_pins
    public, private = make_packet(rows, scores, readouts)
    assert len(public["cards"]) == 16
    sources = [Path(__file__), DEST.parent / "comparison-decision.md"]
    private["input_pins_verified_before_and_after_load"] = after_pins
    private["source_pins"] = after_pins + [
        {"path": str(p.relative_to(ROOT)), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
        for p in sources
    ]
    import contextlib
    import io
    import platform
    import sys
    import numpy as np
    configuration = io.StringIO()
    with contextlib.redirect_stdout(configuration):
        np.show_config()
    private["numeric_environment"] = {
        "interpreter": sys.executable, "python_version": sys.version,
        "platform": platform.platform(), "numpy_version": np.__version__,
        "numpy_build": configuration.getvalue(),
    }
    private["random_control_provenance"] = (
        "Reconstructs seed 20260910, default_rng.normal((1024,4)), np.linalg.qr. "
        "Original random vectors were not saved; acquisition.json, acquisition-start.json "
        "and launch-receipt.json do not record original interpreter/NumPy/build. "
        "Same recipe is verified, identical original basis is not certified. "
        "Random-axis correspondence is secondary and conditional on recipe reproduction; "
        "PC primary uses stored scores and is unaffected."
    )
    private["judging_requirement"] = (
        "Two fresh Astra raters, eight cards each, in card-ID order. Each sees "
        "one method per axis: rater1 odd PC J-Lens/even PC plain, odd random "
        "plain/even random J-Lens; rater2 complement. Lock all responses before "
        "releasing answer key. No judge receives the counterpart for an axis, "
        "private metadata, source, context, tools, or files. Same-model raters; "
        "no replication or human validation."
    )
    DEST.mkdir(exist_ok=False)
    (DEST / "public").mkdir()
    for card in public["cards"]:
        with (DEST / "public" / f"{card['card_id']}.json").open("x") as f:
            json.dump(card, f, indent=2, ensure_ascii=False, allow_nan=False)
            f.write("\n")
    with (DEST / "private-key.json").open("x") as f:
        json.dump(private, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")
    print("Prepared 16 cards and separate private key; no judgments or grades.")


def fixture():
    from unittest.mock import patch
    # Intentionally different framings of one content at the two extremes.
    rows = [
        {"id": "secret-z", "pair": "same", "prefix": "prefix low"},
        {"id": "secret-a", "pair": "same", "prefix": "prefix high"},
        {"id": "secret-b", "pair": "middle", "prefix": "prefix middle"},
    ]
    axes = ["PC1", "random1"]
    scores = {axis: [-4.0, 5.0, 1.0] for axis in axes}
    names = [axis + sign for axis in axes for sign in ["+", "-"]]
    readouts = {f"activation_11_{method}": {
        "names": names,
        "tokens": [[("high" if name.endswith("+") else "low")] * 12 for name in names],
    } for method in METHODS}
    choices = set()
    for seed in range(12):
        public, private = make_packet(rows, scores, readouts, axes=axes, seed=seed)
        assert (public, private) == make_packet(rows, scores, readouts, axes=axes, seed=seed)
        assert len(public["cards"]) == 4
        for card, key in zip(public["cards"], private["keys"], strict=True):
            assert set(card) == {"card_id", "instruction", "panels", "prefixes"}
            assert set(card["panels"]) == {"A", "B"}
            assert set(card["prefixes"]) == {"1", "2"}
            assert key["same_content_extrema"]
            inferred = "A1_B2" if card["panels"]["A"][0] in card["prefixes"]["1"] else "A2_B1"
            assert inferred == key["correct_choice"]
            choices.add(inferred)
        encoded = json.dumps(public)
        assert not any(hidden in encoded for hidden in ["secret-", "PC1", "random1", "jlens", "plain", "correct_choice"])
    assert choices == {"A1_B2", "A2_B1"}
    assignments = {(axis, method): assigned_rater(axis, method) for axis in AXES for method in METHODS}
    for rater in ["rater1", "rater2"]:
        own = [item for item, who in assignments.items() if who == rater]
        assert len(own) == 8 and len({axis for axis, method in own}) == 8
        for family in ["PC", "random"]:
            assert sum(axis.startswith(family) and method == "jlens" for axis, method in own) == 2
            assert sum(axis.startswith(family) and method == "plain" for axis, method in own) == 2
    with patch.object(Path, "exists", return_value=True), patch(__name__ + ".load_saved_inputs") as load:
        try:
            build()
        except FileExistsError:
            pass
        else:
            raise AssertionError("Existing packet target accepted")
        load.assert_not_called()
    with patch.object(Path, "read_bytes", return_value=b"fabricated fixture"):
        expected = hashlib.sha256(b"fabricated fixture").hexdigest()
        assert verify_input_pins({"fixture": expected}) == [{"path": "fixture", "sha256": expected}]
        try:
            verify_input_pins({"fixture": "wrong"})
        except ValueError:
            pass
        else:
            raise AssertionError("Wrong input pin accepted")
    assert extrema(rows, [0, 2, 2]) == (1, 0)
    for invalid in [[1, 1, 1], [1, float("nan"), 2], [1, 2]]:
        try:
            extrema(rows, invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid or degenerate axis was accepted")
    print("PASS synthetic fixture: hidden truth, polarity, all panels, ties, same-content retention, determinism, balanced raters, target guard, input pins.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["fixture", "build"])
    args = parser.parse_args()
    {"fixture": fixture, "build": build}[args.stage]()
