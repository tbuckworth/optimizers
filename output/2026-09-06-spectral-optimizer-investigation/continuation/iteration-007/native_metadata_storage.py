"""CPU-only paired anchor sizing using a completed engineering inspection.

The native metadata is inserted into an UNREGISTERED storage specimen. Its
fixture state and envelope hashes deliberately remain nonsemantic. This is
neither a native anchor nor a whole-attempt fit certificate. Default CLI inert.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import stat
import time


def load_inspection(path, expected_sha256):
    """Read one bounded retained result without following a final symlink."""
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_size > 2 << 20:
            raise ValueError("inspection must be a regular file of at most 2 MiB")
        with os.fdopen(fd, "rb", closefd=False) as handle:
            raw = handle.read((2 << 20) + 1)
        after = os.fstat(fd)
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns,
                before.st_ctime_ns) != (after.st_dev, after.st_ino, after.st_size,
                                       after.st_mtime_ns, after.st_ctime_ns):
            raise ValueError("inspection changed during read")
    finally:
        os.close(fd)
    if len(raw) > 2 << 20 or hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("inspection byte cap or expected SHA-256 mismatch")
    def pairs(rows):
        result = {}
        for key, value in rows:
            if key in result:
                raise ValueError("duplicate inspection key")
            result[key] = value
        return result
    result = json.loads(raw, object_pairs_hook=pairs)
    canonical = json.dumps(result, ensure_ascii=True, sort_keys=False,
                           separators=(",", ":"), allow_nan=False).encode("ascii") + b"\n"
    if canonical != raw:
        raise ValueError("noncanonical inspection JSON")
    expected_certificates = {
        "scientific_execution": False, "pilot": False, "full_fit": False,
        "capture_neutrality": False, "continuation_restore": False}
    certificates = result.get("certificates") if type(result) is dict else None
    if (type(result) is not dict or
            result.get("schema") != "i7_native_layout_inspection_v2" or
            result.get("status") != "pass" or
            result.get("scope") != "engineering_native_layout_only" or
            result.get("worker_gpu_process_absent_after_exit") is not True or
            type(certificates) is not dict or
            tuple(certificates) != tuple(expected_certificates) or
            any(certificates[key] is not False for key in expected_certificates)):
        raise ValueError("inspection is not a completed engineering-only observation")
    return result, len(raw)


def measure(inspection_path, expected_sha256):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("explicitly hidden CUDA required")
    # Lazy imports keep the default CLI inert; no native collector is invoked.
    import torch
    import artifact_store as storage
    import full_envelope_storage_fixture as fixture
    import source_environment as provenance

    fixture._cpu_only()
    started = time.monotonic()
    result, result_bytes = load_inspection(inspection_path, expected_sha256)
    sources = result["source_binding"]
    environment = result["native_environment_binding"]
    provenance.validate_sources(sources, profile=storage.SCIENTIFIC)
    provenance.validate_environment(environment, profile=storage.SCIENTIFIC)
    if (sources["repository_revision"] != result["expected_commit"] or
            sources["repository_root_realpath"] != environment["repository_root_realpath"] or
            environment["runtime_role"] != "native_source" or
            environment["cuda"]["devices"][0]["uuid"] != result["expected_gpu_uuid"]):
        raise ValueError("inspection source/environment cross-binding mismatch")
    before = torch.get_rng_state().clone()
    actual = fixture.actual_tiny_records()
    measured = {}
    for name, steps, count in (("pilot", 220, 2), ("long", 2_000, 16)):
        baseline = fixture.lift_tree("anchor", actual["anchor"], plan_steps=steps)["tree"]
        replaced = copy.deepcopy(baseline)
        replaced["payload"]["bindings"]["sources"] = copy.deepcopy(sources)
        replaced["payload"]["bindings"]["environment"] = copy.deepcopy(environment)
        base_row = fixture.serialized_measurement(baseline)
        replaced_row = fixture.serialized_measurement(replaced)
        if fixture.inventory(baseline) != fixture.inventory(replaced):
            raise ValueError("metadata replacement changed tensor layout")
        for (left_path, left), (right_path, right) in zip(
                fixture.tensor_leaves(baseline), fixture.tensor_leaves(replaced)):
            if left_path != right_path or not fixture._same_exact(left, right):
                raise ValueError("metadata replacement changed tensor bytes")
        measured[name] = {
            "count": count, "baseline": base_row, "native_metadata": replaced_row,
            "serialized_delta_bytes": replaced_row["serialized_bytes"] - base_row["serialized_bytes"],
            "tensor_bytes_unchanged": True,
        }
    layout = environment["rng_layout"]
    fixture_cpu_bytes = actual["anchor"]["payload"]["rng"]["torch_cpu"].numel()
    native_cuda_bytes = sum(row["state_length"] for row in layout["torch_cuda"])
    if not torch.equal(before, torch.get_rng_state()) or torch.cuda.is_initialized():
        raise ValueError("CPU comparison changed caller Torch RNG or initialized CUDA")
    compact = lambda value: len(json.dumps(value, ensure_ascii=True, separators=(",", ":"),
                                          allow_nan=False).encode("ascii") + b"\n")
    with open(__file__, "rb") as handle:
        script_sha256 = hashlib.sha256(handle.read()).hexdigest()
    return {
        "schema": "i7_native_metadata_storage_comparison_v1",
        "evidence_role": "dataset_free_cpu_engineering_paired_metadata_only",
        "inspection_path": os.path.abspath(inspection_path),
        "inspection_sha256": expected_sha256, "inspection_bytes": result_bytes,
        "inspected_commit": result["expected_commit"],
        "comparison_script_sha256": script_sha256,
        "actual_source_files": len(sources["files"]),
        "source_compact_json_bytes": compact(sources),
        "native_environment_compact_json_bytes": compact(environment),
        "components": measured,
        "aggregate_paired_anchor_metadata_delta_bytes": sum(
            row["count"] * row["serialized_delta_bytes"] for row in measured.values()),
        "rng_layout_observed": layout,
        "rng_tensor_arithmetic": {
            "full_cores": 24, "fixture_torch_cpu_bytes_per_core": fixture_cpu_bytes,
            "native_torch_cpu_bytes_per_core": layout["torch_cpu"]["state_length"],
            "native_torch_cuda_bytes_per_core": native_cuda_bytes,
            "native_state_tensor_delta_all_cores": 24 * (
                layout["torch_cpu"]["state_length"] - fixture_cpu_bytes + native_cuda_bytes),
            "schema_cuda_continuation_witness_bytes_all_cores": 24 * 16,
            "cuda_rng_values_instantiated_in_specimens": False,
            "cuda_serialization_overhead_measured": False,
        },
        "specimen_profile": fixture.SPECIMEN_PROFILE,
        "cpu_fixture_draws_performed_and_restored": True,
        "cpu_fixture_rng_restoration_scope": "existing actual_tiny_records implementation; wrapper checks Torch CPU only",
        "specimens_are_semantically_valid_native_anchors": False,
        "complete_attempt_fit_certificate": False, "scientific_launch_approved": False,
        "exclusions": ["native CUDA state/witness serialization overhead",
            "CPU auditor environment metadata", "remaining native adapter fields",
            "whole-root writer behavior, receipts, controller files and failures",
            "all-phase timing and peak memory", "actual scientific scalar/string values"],
        "cuda_initialized": False, "torch_cpu_rng_preserved": True,
        "wall_seconds": time.monotonic() - started,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measure", action="store_true")
    parser.add_argument("--inspection")
    parser.add_argument("--expected-sha256")
    args = parser.parse_args(argv)
    if not args.measure:
        parser.print_help()
        return 0
    if not args.inspection or not args.expected_sha256:
        parser.error("measurement requires --inspection and --expected-sha256")
    print(json.dumps(measure(args.inspection, args.expected_sha256), ensure_ascii=True,
                     separators=(",", ":"), allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
