#!/usr/bin/env python3
"""CPU-only check of common decay, realized reciprocal norms and warmup identity."""
import argparse
import json
import math
from pathlib import Path
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    checks, maximum_relative_norm_error, maximum_reapplication_error = 0, 0., 0.
    for path in sorted(args.artifacts.glob("probe-s*.json")):
        record = json.loads(path.read_text())
        binding = torch.load(args.artifacts / record["anchor_artifact"]["name"],
                             weights_only=True, map_location="cpu")
        payload = torch.load(args.artifacts / record["tensor_artifact"]["name"],
                             weights_only=True, map_location="cpu")
        theta = torch.cat([v.reshape(-1) for v in binding["state"]["model_state"].values()])
        decay = .001 * .01 * theta
        rows = payload["interventions"]
        for row in rows.values():
            assert torch.equal(row["data_delta"], row["total_delta"] + decay)
            checks += 1
        for name, direction, target in (
                ("raw_data_delta_to_native_norm", "raw_gradient", "native_gradient"),
                ("native_data_delta_to_raw_norm", "native_gradient", "raw_gradient")):
            vector, reference = rows[direction]["data_delta"], rows[target]["data_delta"]
            target_norm = torch.linalg.vector_norm(reference.double()).item()
            realized = torch.linalg.vector_norm(rows[name]["data_delta"].double()).item()
            relative = abs(realized - target_norm) / target_norm
            assert math.isclose(realized, target_norm, rel_tol=5e-5, abs_tol=1e-10)
            scale = target_norm / torch.linalg.vector_norm(vector.double()).item()
            expected = (theta + (vector * scale - decay)) - theta
            error = (expected - rows[name]["total_delta"]).abs().max().item()
            assert error <= 1e-7
            maximum_relative_norm_error = max(maximum_relative_norm_error, relative)
            maximum_reapplication_error = max(maximum_reapplication_error, error)
            checks += 2
    for seed in (100, 101, 102):
        rows = [json.loads((args.artifacts / f"probe-s{seed}-{s}-t100.json").read_text())
                for s in ("raw", "current32")]
        assert rows[0]["complete_state_digest"] == rows[1]["complete_state_digest"]
        checks += 1
    assert checks == 243
    result = {"status": "pass", "checks": checks,
              "maximum_relative_realized_norm_error": maximum_relative_norm_error,
              "maximum_absolute_float32_reapplication_error": maximum_reapplication_error,
              "scope": "144 exact common-decay identities,48 reciprocal norm and48 direct-reapplication checks,3 exact warmup digests; no optimizer or model step executed."}
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
