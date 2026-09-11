"""In-memory sizing of required full-width tensor fields, never a study artifact.

No data, training, CUDA initialization, filesystem writes or scientific profile.
This deliberately omits unresolved proof/binding/audit/scalar fields. Measured
ZIP overhead belongs only to this specimen, not to a complete study envelope.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import time
import zipfile

import torch

import artifact_store as storage

PROFILE = "fixture_storage_shape_only_v1"
P, RANK = 50890, 32
PARAMETERS = (("0.weight", (64, 784)), ("0.bias", (64,)),
              ("2.weight", (10, 64)), ("2.bias", (10,)))
BRANCHES = ("raw", "current", "lagged", "restored", "reciprocal", "zero")
CORE_FIXED_BYTES = {"anchor": 7328432, "source_witness": 7735552, "branch_results": 32977072}
INDEX_BYTES = (64 + 256 + 5000) * 8
TENSOR_BYTES = {**CORE_FIXED_BYTES, "anchor": CORE_FIXED_BYTES["anchor"] + INDEX_BYTES}
EXCLUSIONS = ["RNG states and continuation witnesses", "separate complete plan artifacts",
              "complete identities and source/data/environment bindings",
              "capture proofs and on/off pilot evidence", "measurement/comparison scalar maps",
              "independent audit records", "receipts, logs and bounded failures",
              "filesystem allocation and simultaneous copies"]


def cpu_only():
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or torch.cuda.is_initialized():
        raise ValueError("explicitly hide CUDA and keep it uninitialized")


def specimens(pattern="zero"):
    """All-defined maximum-rank fixed tensors; values are not actual endpoints."""
    cpu_only()
    if type(pattern) is not str or pattern not in ("zero", "ones"):
        raise ValueError("pattern must be zero or ones")
    fill = 0. if pattern == "zero" else 1.

    def tensor(shape, dtype=torch.float32):
        return torch.full(shape, fill, dtype=dtype, device="cpu")

    def record(shape, dtype=torch.float32):
        return dict(value=tensor(shape, dtype), shape=list(shape),
                    native_dtype=str(dtype).removeprefix("torch."), native_device="cpu")

    def parameters():
        return [dict(index=i, name=name, shape=list(shape), requires_grad=True,
                     native_dtype="float32", native_device="cpu", value=tensor(shape))
                for i, (name, shape) in enumerate(PARAMETERS)]

    def optimizer():
        return dict(state=[dict(parameter_index=i, parameter_name=name,
            step=torch.tensor(100., dtype=torch.float32), exp_avg=record(shape),
            exp_avg_sq=record(shape)) for i, (name, shape) in enumerate(PARAMETERS)])

    def observer():
        return dict(V=record((P, RANK)), S=record((RANK,), torch.float64),
                    grad_mean=record((P,)))

    def envelope(kind, payload):
        return dict(schema_name="i7_storage_shape_specimen", profile=PROFILE,
                    kind=kind, fill_value=int(fill), synthetic_values=True,
                    scientific_envelope=False, payload=payload)

    anchor = dict(parameters=parameters(), optimizer=optimizer(), observer=observer())
    anchor["bindings"] = dict(plan=dict(next_batch_indices=torch.zeros(64, dtype=torch.int64)),
        probes=dict(training_probe_indices=torch.zeros(256, dtype=torch.int64),
                    auxiliary_indices=torch.zeros(5000, dtype=torch.int64)))
    witness = dict(parameters_after=parameters(), optimizer=optimizer(), observer=observer(),
                   raw_gradient=record((P,)), delivered_current_gradient=record((P,)))
    candidate = dict(g=record((P,)), c=record((P,)), l=record((P,)),
        previous_basis=record((P, RANK)), post_ingest_basis=record((P, RANK)),
        post_ingest_observer=observer(), measurement_before={key: dict(q=tensor((P,), torch.float64))
        for key in ("batch_noisy", "train_probe_noisy", "train_probe_clean", "auxiliary_clean")})
    branches = {name: dict(delivered_gradient=record((P,)), parameters_after=parameters(),
        parameters_after_flat=tensor((P,)), optimizer=optimizer(),
        delta=tensor((P,), torch.float64), delta_data=tensor((P,), torch.float64))
        for name in BRANCHES}
    result = {"anchor": envelope("anchor", anchor),
              "source_witness": envelope("source_witness", witness),
              "branch_results": envelope("branch_results", dict(candidate_state=candidate, branches=branches))}
    # Across all roots as well as within each artifact: duplicates must own storage.
    storage._validate_tree(result)
    return result


def tensor_leaves(tree, path="root"):
    if type(tree) is torch.Tensor:
        yield path, tree
    elif type(tree) is dict:
        for key, value in tree.items():
            yield from tensor_leaves(value, f"{path}.{key}")
    elif type(tree) in (tuple, list):
        for i, value in enumerate(tree):
            yield from tensor_leaves(value, f"{path}.{i}")


def inventory(tree):
    storage._validate_tree(tree)
    return [dict(path=path, shape=list(value.shape), dtype=str(value.dtype),
                 tensor_bytes=value.numel()*value.element_size())
            for path, value in tensor_leaves(tree)]


def serialized_measurement(tree, *, buffer_limit=64 << 20):
    """Own output bytes only in bounded memory; decode our own checked bytes."""
    cpu_only()
    if type(buffer_limit) is not int or not 0 < buffer_limit <= 64 << 20:
        raise ValueError("buffer limit must be a positive integer at most 64 MiB")
    if type(tree) is not dict or tree.get("profile") != PROFILE or tree.get("scientific_envelope") is not False:
        raise ValueError("only explicitly non-scientific sizing specimens are accepted")
    leaves = inventory(tree)
    logical = sum(row["tensor_bytes"] for row in leaves)
    with storage._BoundedBuffer(buffer_limit) as buffer:
        torch.save(tree, buffer)
        if buffer.exceeded:
            raise storage.StoreError("serialization exceeded its fixed memory bound")
        view = buffer.getbuffer()
        try:
            size, digest = len(view), hashlib.sha256(view).hexdigest()
        finally:
            view.release()
        buffer.seek(0)
        with zipfile.ZipFile(buffer) as archive:
            entries = archive.infolist()
            if any(row.compress_type != zipfile.ZIP_STORED or row.compress_size != row.file_size for row in entries):
                raise ValueError("compressed archive cannot validate uncompressed tensor sizing")
            storages = [row for row in entries if "/data/" in row.filename]
            if len(storages) != len(leaves) or sum(row.file_size for row in storages) != logical:
                raise ValueError("serialized storages do not match owned tensor inventory")
        buffer.seek(0)
        loaded = torch.load(buffer, weights_only=True, map_location="cpu")
        if inventory(loaded) != leaves:
            raise ValueError("restricted roundtrip tensor inventory differs")
        for (path, old), (new_path, new) in zip(tensor_leaves(tree), tensor_leaves(loaded)):
            if path != new_path or not torch.equal(old.reshape(-1).view(torch.uint8), new.reshape(-1).view(torch.uint8)):
                raise ValueError("restricted roundtrip tensor bytes differ")
        return dict(tensor_bytes=logical, tensor_count=len(leaves), serialized_bytes=size,
                    specimen_overhead_bytes=size-logical, serialized_sha256=digest,
                    zip_entries=len(entries), all_zip_entries_uncompressed=True,
                    restricted_tensor_roundtrip=True, tensor_inventory=leaves)


def measure(pattern="zero"):
    """Return exact specimen measurements, explicitly not full-attempt feasibility."""
    cpu_only()
    started, cpu_started = time.monotonic(), time.process_time()
    rng = torch.get_rng_state().clone()
    objects = specimens(pattern)
    records = {name: serialized_measurement(tree) for name, tree in objects.items()}
    if {key: row["tensor_bytes"] for key, row in records.items()} != TENSOR_BYTES:
        raise ValueError("specimen no longer matches the independently counted fixed-tensor ledger")
    if not torch.equal(rng, torch.get_rng_state()) or torch.cuda.is_initialized():
        raise ValueError("sizing mutated Torch RNG or initialized CUDA")
    serialized = sum(row["serialized_bytes"] for row in records.values())
    subtotal = 18*serialized + (128 << 20)
    return dict(schema="i7_storage_shape_measurement_v1", evidence_role="dataset_free_cpu_engineering",
        profile=PROFILE, pattern=pattern, artifacts=records,
        core_fixed_tensor_bytes=sum(CORE_FIXED_BYTES.values()), required_index_tensor_bytes=INDEX_BYTES,
        measured_tensor_bytes=sum(TENSOR_BYTES.values()), specimen_serialized_bytes=serialized,
        arithmetic_only_projection=dict(groups=18, inherited_allowance_bytes=128 << 20,
            subtotal_bytes=subtotal, cap_bytes=1 << 30, residual_bytes=(1 << 30)-subtotal),
        scientific_launch_approved=False, complete_attempt_feasibility="unresolved",
        exclusions=list(EXCLUSIONS), in_memory_only=True, rng_preserved=True, cuda_initialized=False,
        runtime=dict(python=platform.python_version(), torch=str(torch.__version__),
            cpu_threads=torch.get_num_threads(), wall_seconds=time.monotonic()-started,
            cpu_seconds=time.process_time()-cpu_started,
            lifetime_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measure", action="store_true", help="explicitly run the in-memory CPU sizing specimen")
    parser.add_argument("--pattern", choices=("zero", "ones"), default="zero")
    args = parser.parse_args()
    if not args.measure:
        parser.print_help()
        return
    print(json.dumps(measure(args.pattern), sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
