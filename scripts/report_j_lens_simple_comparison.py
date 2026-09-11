"""Grade locked matching responses and plot saved cards; never open feature archives."""

import argparse
from collections import Counter
import hashlib
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/2026-09-10-j-lens-simple-comparison"
LOCK_COMMIT = "fc185c7210b5cd4c611a524c4490a4c204056aaa"
METHODS = ["jlens", "plain"]
CHOICES = {"A1_B2", "A2_B1"}
CONFIDENCES = {"low", "medium", "high"}


def read(path):
    return json.loads(path.read_text())


def pin(path):
    return {"path": str(path.relative_to(ROOT)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def grade_responses(keys, raters):
    expected = {(f"{family}{i}", method) for family in ["PC", "random"]
                for i in range(1, 5) for method in METHODS}
    assert len(keys) == 16
    assert {(k["axis"], k["method"]) for k in keys} == expected
    by_id = {k["card_id"]: k for k in keys}
    assert set(by_id) == {f"C{i:02d}" for i in range(1, 17)}
    responses = {}
    for name in ["rater1", "rater2"]:
        items = raters[name]["responses"]
        assigned = {k["card_id"] for k in keys if k["assigned_rater"] == name}
        assert len(items) == len(assigned) == 8
        assert len({r["card_id"] for r in items}) == 8
        assert {r["card_id"] for r in items} == assigned
        assert [r["card_id"] for r in items] == sorted(assigned)
        for r in items:
            assert set(r) == {"card_id", "choice", "confidence"}
            assert r["choice"] in CHOICES and r["confidence"] in CONFIDENCES
            assert r["card_id"] not in responses
            responses[r["card_id"]] = r
    graded = []
    for card_id, k in sorted(by_id.items()):
        r = responses[card_id]
        assert k["correct_choice"] in CHOICES
        assert sorted(k["panel_signs"]) == sorted(k["prefix_signs"]) == ["+", "-"]
        assert k["correct_choice"] == ("A1_B2" if k["panel_signs"] == k["prefix_signs"] else "A2_B1")
        graded.append({**k, "choice": r["choice"], "confidence": r["confidence"],
                       "correct": r["choice"] == k["correct_choice"]})
    counts, axes = {}, []
    for family in ["PC", "random"]:
        subset = [r for r in graded if r["axis"].startswith(family)]
        counts[family] = {}
        for method in METHODS:
            items = [r for r in subset if r["method"] == method]
            assert len(items) == 4
            counts[family][method] = {"correct": sum(r["correct"] for r in items), "total": 4}
        counts[family]["paired_difference_correct"] = counts[family]["jlens"]["correct"] - counts[family]["plain"]["correct"]
        for i in range(1, 5):
            axis = f"{family}{i}"
            items = {r["method"]: r for r in subset if r["axis"] == axis}
            endpoint_sets = [{s: (r["candidate_ids"][j], r["endpoint_scores"][j])
                              for j, s in enumerate(r["prefix_signs"])} for r in items.values()]
            assert endpoint_sets[0] == endpoint_sets[1]
            axes.append({"axis": axis, "family": family,
                         "methods": {m: {field: items[m][field] for field in
                           ["card_id", "assigned_rater", "correct", "confidence"]} for m in METHODS},
                         "endpoints": {s: {"id": v[0], "score": v[1]} for s, v in endpoint_sets[0].items()},
                         "same_content_extrema": items["jlens"]["same_content_extrema"]})
    return {"counts": counts, "per_axis": axes, "per_card": graded}


def grade():
    target = OUT / "grades.json"
    if target.exists():
        raise FileExistsError("Locked grades already exist; plotting is a separate stage")
    import package_j_lens_simple_comparison as protocol
    keypath = OUT / "packet/private-key.json"
    private = read(keypath)
    declared = {r["path"]: r["sha256"] for r in private["input_pins_verified_before_and_after_load"]}
    assert declared == protocol.INPUT_PINS
    verified, archive_records = [], []
    for record in private["source_pins"]:
        path = ROOT / record["path"]
        if path.suffix == ".npz":
            assert record["sha256"] == protocol.INPUT_PINS[record["path"]]
            archive_records.append(record)
        else:
            actual = pin(path)
            assert actual == record
            verified.append(actual)
    assert len(archive_records) == 2
    sources = [keypath, Path(__file__)]
    raters = {}
    for name in ["rater1", "rater2"]:
        path = OUT / f"{name}-locked.json"
        raters[name] = read(path)
        sources.append(path)
    result = grade_responses(private["keys"], raters)
    readouts = read(ROOT / "output/2026-09-10-j-lens-completions/readouts.json")
    rows = {r["id"]: r for r in read(ROOT / "output/2026-09-10-j-lens-completions/dataset.json")}
    for k in private["keys"]:
        assert protocol.assigned_rater(k["axis"], k["method"]) == k["assigned_rater"]
        path = OUT / f"packet/public/{k['card_id']}.json"
        card = read(path)
        sources.append(path)
        assert set(card) == {"card_id", "instruction", "panels", "prefixes"}
        assert card["card_id"] == k["card_id"] and card["instruction"] == protocol.PROMPT
        group = readouts[f"activation_11_{k['method']}"]
        lookup = dict(zip(group["names"], group["tokens"], strict=True))
        for label, sign in zip(["A", "B"], k["panel_signs"], strict=True):
            assert card["panels"][label] == lookup[k["axis"] + sign]
            assert len(card["panels"][label]) == 12
        for label, row_id in zip(["1", "2"], k["candidate_ids"], strict=True):
            assert card["prefixes"][label] == rows[row_id]["prefix"]
        assert k["same_content_extrema"] == (rows[k["candidate_ids"][0]]["pair"] == rows[k["candidate_ids"][1]]["pair"])
    result["provenance"] = {
        "responses_and_packets_locked_at_commit": LOCK_COMMIT,
        "verified_json_and_source_pins": verified,
        "frozen_archive_pins_checked_as_records_only": archive_records,
        "archives_opened": False,
        "grading_inputs": [pin(p) for p in sources],
        "rater_models": {name: raters[name]["model"] for name in raters},
        "random_control_provenance": private["random_control_provenance"],
    }
    result["validation"] = {"fixed_roster": "PASS", "rater_allocations": "PASS",
        "choices_and_confidence": "PASS", "public_card_source_correspondence": "PASS",
        "declared_answer_key_polarity": "PASS", "no_rejudging": True}
    with target.open("x") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")
    print(json.dumps(result["counts"], indent=2))


def visible(value):
    return value.replace("\\", "\\\\").replace(" ", "␠").replace("\t", "\\t").replace("\n", "\\n").replace("\r", "\\r")


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    plt.rcParams.update({"font.family": ["DejaVu Sans", "Noto Sans CJK JP", "Noto Sans Thai", "Noto Sans Symbols2"],
                         "font.size": 13, "text.usetex": False})
    grades = read(OUT / "grades.json")
    figures = OUT / "figures"
    figures.mkdir(exist_ok=True)
    by_axis = {r["axis"]: r for r in grades["per_axis"]}
    green, red, ink, muted = "#d7ebe1", "#f5dfda", "#203431", "#66736f"
    fig, panels = plt.subplots(1, 2, figsize=(13, 6.1))
    fig.subplots_adjust(left=.06, right=.97, top=.72, bottom=.17, wspace=.22)
    fig.text(.045, .95, "One extra PC match with J-Lens", fontsize=24, weight="bold", color=ink)
    fig.text(.045, .89, "Blinded token-list → held-out endpoint matching · activation layer 11", fontsize=14, color=muted)
    for ax, family, title in zip(panels, ["PC", "random"], ["Four covariance eigenvectors", "Four random axes · secondary*"]):
        ax.set_xlim(0, 2)
        ax.set_ylim(0, 4)
        ax.axis("off")
        ax.set_title(title, loc="left", fontsize=17, pad=42, color=ink)
        for col, method, label in zip(range(2), METHODS, ["J-Lens", "Plain lens"]):
            count = grades["counts"][family][method]["correct"]
            ax.text(col + .5, 4.13, f"{label}  {count}/4", ha="center", fontsize=16, weight="bold", color=ink)
            for i in range(1, 5):
                result = by_axis[f"{family}{i}"]["methods"][method]
                y = 4 - i
                ax.add_patch(FancyBboxPatch((col + .025, y + .04), .95, .9,
                    boxstyle="round,pad=0.012,rounding_size=0.045", linewidth=0,
                    facecolor=green if result["correct"] else red))
                label_axis = f"PC{i}" if family == "PC" else f"Random {i}"
                ax.text(col + .5, y + .58, f"{label_axis} · {'match' if result['correct'] else 'miss'}",
                        fontsize=15, ha="center", va="center", color=ink)
                ax.text(col + .5, y + .27, f"{result['confidence']} confidence", fontsize=11, ha="center", color=muted)
    fig.text(.045, .09, "PC3 accounts for the extra match. PC4 is missed by both. The random comparison has the same 3/4 vs 2/4 tally.", fontsize=12, color=ink)
    fig.text(.045, .035, "Exploratory: 4 dependent axes, 8 contents, 2 same-model raters. *Random basis reconstructed from the saved recipe.", fontsize=11, color=muted)
    fig.savefig(figures / "summary.png", dpi=160, facecolor="white")
    plt.close(fig)
    raw_cards = {k["card_id"]: read(OUT / f"packet/public/{k['card_id']}.json") for k in grades["per_card"]}
    keys = {k["card_id"]: k for k in grades["per_card"]}
    for i in range(1, 5):
        name, item = f"PC{i}", by_axis[f"PC{i}"]
        fig = plt.figure(figsize=(14, 9.5))
        fig.text(.035, .95, name, fontsize=26, weight="bold", color=ink)
        outcomes = "  ·  ".join(f"{label}: {'match' if item['methods'][method]['correct'] else 'miss'} ({item['methods'][method]['confidence']})"
                                for method, label in [("jlens", "J-Lens"), ("plain", "Plain")])
        fig.text(.15, .953, outcomes, fontsize=17, color=ink)
        fig.text(.035, .908, "Held-out geometric endpoints · all 12 saved tokens per sign and method", fontsize=14, color=muted)
        token_lists, prefixes = {}, {}
        for method in METHODS:
            key = keys[item["methods"][method]["card_id"]]
            card = raw_cards[key["card_id"]]
            for panel, sign in zip(["A", "B"], key["panel_signs"], strict=True):
                token_lists[(method, sign)] = card["panels"][panel]
            for label, sign in zip(["1", "2"], key["prefix_signs"], strict=True):
                prefixes[sign] = card["prefixes"][label]
        for x, sign, pole in [(.035, "+", "POSITIVE END"), (.525, "-", "NEGATIVE END")]:
            fig.text(x, .846, pole, fontsize=13, color=muted, weight="bold")
            fig.text(x, .802, prefixes[sign], fontsize=17, color=ink, va="top", linespacing=1.4)
            fig.text(x, .700, f"Signed loading {item['endpoints'][sign]['score']:+.3f}", fontsize=12, color=muted)
        for x, sign, method, label in [(.035, "+", "jlens", "J-Lens +"), (.275, "+", "plain", "Plain +"),
                                      (.525, "-", "jlens", "J-Lens −"), (.765, "-", "plain", "Plain −")]:
            fig.text(x, .638, label, fontsize=17, weight="bold", color=ink)
            for rank, token in enumerate(token_lists[(method, sign)], 1):
                fig.text(x, .590 - (rank - 1) * .038, f"{rank:02d}  {visible(token)}", fontsize=16, color=ink)
        fig.text(.035, .078, "␠ = space; \\n / \\t = newline / tab; \\\\ = literal backslash. Rank order and multilingual fragments are preserved.", fontsize=11, color=muted)
        fig.text(.035, .041, "Matching these two examples is a limited correspondence check; it does not establish a distinct semantic concept.", fontsize=12, color=muted)
        fig.savefig(figures / f"pc{i}.png", dpi=160, facecolor="white")
        plt.close(fig)
    write_reports(grades)
    print("Saved summary, four complete PC panels and report drafts; grades unchanged.")


def write_reports(grades):
    headline = "J-Lens matched 3 of 4 PC endpoint pairs; the plain lens matched 2."
    caveat = ("This small exploratory comparison does not establish an advantage: random axes also scored 3/4 versus 2/4. "
              "PC3 supplies the extra PC match; both methods miss PC4.")
    limits = ("Four dependent PCs; eight authored contents in two framings; two fresh Astra raters of the same model, "
              "one judgment per item, no human validation. The task matches geometric extrema, not independently established concepts. "
              "Random-axis correspondence is conditional on reconstructing the original seed/QR basis; its original numeric environment was not recorded. "
              "No new model inference or classification experiment.")
    body = f"""<div style="max-width:1100px;margin:0 auto;padding:24px;color:#203431;font:16px/1.5 Arial,sans-serif">
<p style="font-size:12px;letter-spacing:1px;color:#66736f">Codex — Spectral Optimizer Investigation</p>
<h1 style="font-size:28px;line-height:1.25">{headline}</h1>
<p>{caveat}</p>
<img src="figures/summary.png" alt="Four PC comparisons and four secondary random comparisons, with confidence." style="width:100%;height:auto">
<p>Each rater saw two anonymous token lists and two held-out prefixes, then matched their meanings. J-Lens and plain used the same directions and endpoint pairs.</p>
{''.join(f'<img src="figures/pc{i}.png" alt="PC{i}: complete signed readouts from both lenses and held-out endpoint prefixes." style="width:100%;height:auto;margin:12px 0">' for i in range(1,5))}
<p style="font-size:13px;color:#66736f">{limits}</p>
</div>"""
    (OUT / "report.html").write_text("<!doctype html><html><head><meta charset=\"utf-8\"><title>J-Lens simple comparison · Codex</title></head><body>" + body + "</body></html>\n")
    email = body.replace('src="figures/summary.png"', 'src="cid:jlens-simple-summary"')
    for i in range(1, 5):
        email = email.replace(f'src="figures/pc{i}.png"', f'src="cid:jlens-simple-pc{i}"')
    (OUT / "email-draft.html").write_text("<!doctype html><html><head><meta charset=\"utf-8\"></head><body>" + email + "</body></html>\n")
    table = "\n".join(f"| {r['axis']} | {'Match' if r['methods']['jlens']['correct'] else 'Miss'} ({r['methods']['jlens']['confidence']}) | {'Match' if r['methods']['plain']['correct'] else 'Miss'} ({r['methods']['plain']['confidence']}) |" for r in grades["per_axis"])
    report = f"""# J-Lens simple comparison

{headline} {caveat}

The fixed layer-11 activation comparison used four PCs, both signs, all 12
saved tokens per sign, and the same held-out extrema under both lenses.
Two fresh blinded Astra raters judged one method per axis, counterbalanced
within PC/random families. Responses were locked in `{LOCK_COMMIT}` before grading.

| Axis | J-Lens | Plain lens |
|---|---|---|
{table}

PC: 3/4 versus 2/4; paired difference +1 correct axis. Random: 3/4 versus 2/4;
paired difference +1. Confidence is descriptive and does not change grading.
No same-content/different-framing extrema occurred; none were replaced.

{limits}

This tests limited readout-to-example correspondence. It does not demonstrate
that PCs beat individual vectors, identify four distinct concepts, or improve
classification. Recognizable tokens alone were not the outcome criterion.

[Exact per-card grades and validation](grades.json) · [Fixed protocol](comparison-decision.md) ·
[Rater 1 locked responses](rater1-locked.json) · [Rater 2 locked responses](rater2-locked.json) ·
[Full visual report](report.html) · [Grader/plotter](../../scripts/report_j_lens_simple_comparison.py).
The grader checked all 16 cards, rater allocations, allowed responses,
polarity keys and correspondence to pinned saved readouts/prefixes. It did not
open feature/analysis archives. Figures decode no new directions.
"""
    (OUT / "results.md").write_text(report)


def fixture():
    import package_j_lens_simple_comparison as protocol
    rows = [{"id": "one", "pair": "one", "prefix": "one"},
            {"id": "two", "pair": "two", "prefix": "two"}]
    scores = {a: [1., -1.] for a in protocol.AXES}
    names = [a + s for a in protocol.AXES for s in ["+", "-"]]
    readouts = {f"activation_11_{m}": {"names": names, "tokens": [[n] * 12 for n in names]} for m in METHODS}
    _, private = protocol.make_packet(rows, scores, readouts)
    raters = {name: {"responses": []} for name in ["rater1", "rater2"]}
    for k in private["keys"]:
        raters[k["assigned_rater"]]["responses"].append({"card_id": k["card_id"], "choice": k["correct_choice"], "confidence": "low"})
    result = grade_responses(private["keys"], raters)
    assert all(result["counts"][f][m]["correct"] == 4 for f in ["PC", "random"] for m in METHODS)
    r = raters["rater1"]["responses"][0]
    r["choice"] = next(c for c in CHOICES if c != r["choice"])
    changed = grade_responses(private["keys"], raters)
    assert sum(x["correct"] for x in changed["per_card"]) == 15
    for bad in ["duplicate", "wrong-rater", "confidence"]:
        copy = json.loads(json.dumps(raters))
        if bad == "duplicate":
            copy["rater1"]["responses"][-1] = copy["rater1"]["responses"][0]
        elif bad == "wrong-rater":
            copy["rater1"]["responses"][0], copy["rater2"]["responses"][0] = copy["rater2"]["responses"][0], copy["rater1"]["responses"][0]
        else:
            copy["rater1"]["responses"][0]["confidence"] = "certain"
        try:
            grade_responses(private["keys"], copy)
        except AssertionError:
            pass
        else:
            raise AssertionError(f"Accepted {bad}")
    assert visible(" a\n\t\\") == "␠a\\n\\t\\\\"
    print("PASS synthetic grading: known truth, flipped choice, duplicate/allocation/confidence rejection, visible whitespace.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["fixture", "grade", "plot"])
    {"fixture": fixture, "grade": grade, "plot": plot}[parser.parse_args().stage]()
