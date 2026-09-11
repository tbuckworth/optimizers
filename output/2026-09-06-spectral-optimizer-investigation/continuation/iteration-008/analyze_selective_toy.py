#!/usr/bin/env python3
"""Read-only evidence audit plus exclusive derived summary; never runs training."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def stats(values):
    values = np.array(values, dtype=float)
    return {"mean": float(values.mean()), "min": float(values.min()),
            "max": float(values.max())}


def main():
    data = HERE / "artifacts"
    complete = json.loads((data / "completion.json").read_text())
    assert complete["status"] == "complete" and complete["trajectories"] == 512
    for name, digest in complete["artifact_sha256"].items():
        assert hashlib.sha256((data / name).read_bytes()).hexdigest() == digest, name
    records = [json.loads(line) for line in (data / "trajectories.jsonl").read_text().splitlines()]
    by_id = {r["id"]: r for r in records}
    assert len(by_id) == len(records) == 512
    groups = defaultdict(list)
    largest_risk_reconstruction_error = 0.0
    with np.load(data / "clean_risk_curves.npz", allow_pickle=False) as curves:
        assert set(curves.files) == set(by_id)
        for record in records:
            curve = curves[record["id"]]
            assert curve.shape == (2001,) and np.isfinite(curve).all()
            assert curve.min() == record["minimum_clean_risk"]
            assert int(curve.argmin()) == record["minimum_step"]
            assert curve[101:].min() == record["post_warmup_minimum_clean_risk"]
            assert int(curve[101:].argmin()) + 101 == record["post_warmup_minimum_step"]
            assert curve[-1] == record["endpoint_clean_risk"]
            for snap in record["snapshots"]:
                assert snap["clean_risk"] == curve[snap["step"]]
                u, v = snap["theta_uv"]
                error = abs(.5 * ((u - 1)**2 + v**2) - snap["clean_risk"])
                largest_risk_reconstruction_error = max(largest_risk_reconstruction_error, error)
                assert error < 1e-13
            raw_id = record["id"].rsplit("-", 1)[0] + "-raw"
            raw = by_id[raw_id]
            assert record["batch_sha256"] == raw["batch_sha256"]
            np.testing.assert_array_equal(curve[:101], curves[raw_id][:101])
            if record["arm"] == "frozen":
                aligns = [s["basis_useful_alignment"] for s in record["snapshots"] if s["step"] >= 100]
                assert min(aligns) == max(aligns)
            record["primary_delta_vs_raw_best"] = record["post_warmup_minimum_clean_risk"] - raw["minimum_clean_risk"]
            record["endpoint_delta_vs_raw_best"] = record["endpoint_clean_risk"] - raw["minimum_clean_risk"]
            key = (record["optimizer"], record["design"], record["angle_degrees"], record["arm"])
            groups[key].append(record)
    cells = []
    for (optimizer, design, angle, arm), rows in groups.items():
        assert sorted(r["seed"] for r in rows) == list(range(8))
        primary = [r["primary_delta_vs_raw_best"] for r in rows]
        cell = {"optimizer": optimizer, "design": design, "angle_degrees": angle,
                "arm": arm, "n_batch_order_seeds": 8,
                "primary_experiment_cell": optimizer == "sgd" and design == "useful" and arm == "live",
                "deterministic_repeats": design == "none",
                "endpoint_risk": stats([r["endpoint_clean_risk"] for r in rows]),
                "best_risk": stats([r["minimum_clean_risk"] for r in rows]),
                "primary_delta_vs_raw_best": stats(primary),
                "primary_wins_tolerance_1e-10": sum(x < -1e-10 for x in primary),
                "endpoint_delta_vs_raw_best": stats([r["endpoint_delta_vs_raw_best"] for r in rows]),
                "endpoint_wins_tolerance_1e-10": sum(r["endpoint_delta_vs_raw_best"] < -1e-10 for r in rows),
                "endpoint_train_risk": stats([r["snapshots"][-1]["train_risk"] for r in rows]),
                "endpoint_useful_coordinate": stats([r["snapshots"][-1]["theta_uv"][0] for r in rows]),
                "endpoint_nuisance_coordinate": stats([r["snapshots"][-1]["theta_uv"][1] for r in rows]),
                "snapshots": {},
        }
        leaks = [r["out_of_applied_subspace_step_energy_fraction"] for r in rows]
        cell["outside_step_energy_fraction"] = None if leaks[0] is None else stats(leaks)
        for step in [100, 101, 150, 250, 500, 1000, 2000]:
            snaps = [next(s for s in r["snapshots"] if s["step"] == step) for r in rows]
            aligns = [s["basis_useful_alignment"] for s in snaps]
            cell["snapshots"][str(step)] = {
                "clean_risk": stats([s["clean_risk"] for s in snaps]),
                "useful_coordinate": stats([s["theta_uv"][0] for s in snaps]),
                "nuisance_coordinate": stats([s["theta_uv"][1] for s in snaps]),
                "useful_alignment": None if aligns[0] is None else stats(aligns),
            }
        cells.append(cell)
    summary = {
        "audit": {"trajectories": 512, "cells": len(cells), "hashes_verified": True,
                  "all_curve_minima_verified": True, "all_paired_warmups_exact": True,
                  "frozen_alignments_constant": True,
                  "largest_snapshot_risk_reconstruction_error": largest_risk_reconstruction_error},
        "interpretation": "Clean fixed-design toy risk; oracle-stopped, not held-out selection. Seeds vary order only.",
        "cells": cells,
    }
    with (HERE / "summary.json").open("x") as f:
        json.dump(summary, f, indent=2, allow_nan=False)
        f.write("\n")
    print(json.dumps(summary["audit"]))
    print("optimizer design angle arm endpoint best primary_delta endpoint_delta wins alignment leakage")
    for c in cells:
        alignment = c["snapshots"]["2000"]["useful_alignment"]
        leakage = c["outside_step_energy_fraction"]
        print(c["optimizer"], c["design"], c["angle_degrees"], c["arm"],
              *(f'{c[k]["mean"]:.6f}' for k in ["endpoint_risk", "best_risk", "primary_delta_vs_raw_best", "endpoint_delta_vs_raw_best"]),
              c["primary_wins_tolerance_1e-10"],
              None if alignment is None else round(alignment["mean"], 6),
              None if leakage is None else round(leakage["mean"], 6))


if __name__ == "__main__":
    main()
