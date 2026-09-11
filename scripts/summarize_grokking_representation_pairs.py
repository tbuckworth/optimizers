#!/usr/bin/env python3
"""Descriptive paired contrasts at every saved step; no selected thresholds."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from experiments.analyze_grokking_representations import AGGREGATE_PATHS, ARMS, SEEDS, STEPS


def value(row, path):
    for key in path:
        row = row[key]
    return row


def summarize(differences):
    vals = np.array([row["difference"] for row in differences if row["difference"] is not None])
    return {"seed_differences": differences, "defined_count": len(vals),
            "mean": float(vals.mean()) if len(vals) else None,
            "se": float(vals.std(ddof=1) / math.sqrt(len(vals))) if len(vals)>1 else None,
            "positive_count": int((vals>0).sum()), "negative_count": int((vals<0).sum())}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source",type=Path)
    parser.add_argument("output",type=Path)
    args=parser.parse_args()
    source=args.source.read_bytes()
    rows=json.loads(source)["rows"]
    lookup={(r["seed"],r["arm"],r["step"]):r for r in rows}
    expected={(s,a,t) for s in SEEDS for a in ARMS for t in STEPS}
    if len(rows)!=150 or set(lookup)!=expected:
        raise ValueError("Require all 150 states.")
    fixture=summarize([{"seed":i,"difference":float(i)} for i in (-2,-1,0,1,2)])
    assert fixture["mean"]==0 and abs(fixture["se"]-math.sqrt(.5))<1e-14
    results=[]
    for left,right in (("legacy","adamw"),("stable","adamw"),("stable","legacy")):
        for step in STEPS:
            metrics={}
            for name,path in AGGREGATE_PATHS.items():
                diffs=[]
                for seed in SEEDS:
                    a,b=value(lookup[seed,left,step],path),value(lookup[seed,right,step],path)
                    diffs.append({"seed":seed,"difference":None if a is None or b is None else a-b})
                metrics[name]=summarize(diffs)
            results.append({"left":left,"right":right,"step":step,"metrics":metrics})
    result={"source_sha256":hashlib.sha256(source).hexdigest(),
            "script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "description":"Post-hoc descriptive left-minus-right contrasts at all fixed checkpoints; seed is the replicate; no confirmatory p-values.",
            "contrasts":results}
    with args.output.open("x") as handle:
        json.dump(result,handle,indent=2,allow_nan=False)
        handle.write("\n")


if __name__=="__main__":
    main()
