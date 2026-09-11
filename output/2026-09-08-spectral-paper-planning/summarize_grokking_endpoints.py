"""Supplement the frozen collection with sustained endpoints and threshold timing.

Read-only with respect to scientific runs and the original summary. Output is
exclusive-create. Times exclude evaluation/checkpointing and are not a dedicated
wall-clock benchmark. Sustained means at every later observed point through 6000.
"""
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent / "grokking-confirmation-results"


def describe(values):
    return {"values": values, "mean": statistics.mean(values),
            "standard_error": statistics.stdev(values) / math.sqrt(len(values))}


def main():
    summary = json.loads((ROOT / "summary.json").read_text())
    records = {}
    for source in summary["evidence"]:
        path = ROOT / source["local_copy"]
        content = path.read_bytes()
        assert hashlib.sha256(content).hexdigest() == source["sha256"]
        run = json.loads(content)
        rows = run["evaluation_rows"]
        first = next(r for r in rows if r["test"]["accuracy"] >= .9)
        last_below = max(i for i, r in enumerate(rows) if r["test"]["accuracy"] < .9)
        sustained = rows[last_below + 1]
        records[source["seed"], source["arm"]] = {
            "first_step": first["step"], "sustained_step": sustained["step"],
            "training_seconds_to_first": first["cumulative_training_seconds"],
            "training_seconds_to_sustained": sustained["cumulative_training_seconds"],
            "final_test_loss": rows[-1]["test"]["loss"]}
    seeds, arms = list(range(100, 105)), ("adamw", "legacy", "stable")
    assert set(records) == {(s, a) for s in seeds for a in arms}
    out = {"seeds": seeds, "n": 5,
           "timing_scope": "cumulative synchronized training only; excludes evaluation/checkpoints; shared desktop",
           "sustained_scope": "all observed 50-step-grid points through 6000; not continuous-time or future permanence",
           "arms": {}, "paired_differences": {}}
    fields = tuple(records[100, "adamw"])
    for arm in arms:
        out["arms"][arm] = {field: describe([records[s, arm][field] for s in seeds])
                            for field in fields}
    for treatment, control in (("stable", "adamw"), ("legacy", "adamw"), ("stable", "legacy")):
        result = {}
        for field in fields:
            values = [records[s, treatment][field] - records[s, control][field] for s in seeds]
            result[field] = {**describe(values), "negative_count": sum(v < 0 for v in values)}
        out["paired_differences"][treatment + "_minus_" + control] = result
    with (ROOT / "endpoint-supplement.json").open("x") as handle:
        json.dump(out, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
