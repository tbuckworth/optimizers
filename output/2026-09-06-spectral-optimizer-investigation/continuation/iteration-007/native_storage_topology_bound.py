"""Combine prospective native topology bounds; never validate or authorize a run.

Pure integer arithmetic. No Torch, files, artifacts, plans, IDX, RNG or CUDA.
This computes conditional ceilings, NOT authenticated runtime admission.
"""
from __future__ import annotations

from math import prod

import core_primitive_bound as core
import native_tensor_inventory as inventory
import pickle_storage_bound as pickle_bound
import zip_storage_bound as zip_bound

JSON_COMPARISON_BYTES = 308940
# Independent complete ledger (not just ArtifactStore root). Rechecked against
# final_storage_accounting.project in the focused integration test.
NONBODY_COMPLETE_LEDGER_BYTES = 18449521
STORE_BUDGET_BYTES = 1 << 30


def order_independent_zip_ceiling(pickle_bytes, storage_sizes):
    """Use<=63 alignment bytes per record, independent of encounter order.

    Lower-width/omitted storage rows only reduce raw payload, count, numbered
    filenames and all overheads. Every actual archive must separately satisfy
    the same pinned writer settings,<=64MiB domain and primitive/tensor bounds.
    """
    if type(pickle_bytes) is not int or pickle_bytes < 1:
        raise ValueError("positive exact pickle ceiling required")
    if type(storage_sizes) is not tuple or any(type(n) is not int or n < 0 for n in storage_sizes):
        raise ValueError("exact nonnegative storage-size tuple required")
    count = len(storage_sizes)
    if count + 6 >= 65535:
        raise ValueError("record count outside admitted ZIP32 domain")
    names = ("data.pkl", *(name for name, _ in zip_bound.FIXED_RECORDS),
             *(f"data/{i}" for i in range(count)), *(name for name, _ in zip_bound.FINAL_RECORDS))
    filename_bytes = sum(len((zip_bound.ARCHIVE_PREFIX + name).encode("ascii")) for name in names)
    records = len(names)
    payload = pickle_bytes + sum(storage_sizes) + zip_bound.FIXED_RECORD_PAYLOAD_BYTES
    overhead = records * (zip_bound.LOCAL_HEADER_BYTES + zip_bound.FB_EXTRA_HEADER_BYTES
        + zip_bound.ALIGNMENT_BYTES - 1 + zip_bound.DATA_DESCRIPTOR32_BYTES
        + zip_bound.CENTRAL_HEADER_BYTES) + 2 * filename_bytes + zip_bound.ZIP64_END_BYTES
    total = payload + overhead
    if total >= zip_bound.BUFFER_LIMIT_BYTES:
        raise ValueError("archive ceiling leaves admitted64MiB domain")
    return dict(archive_bytes_upper=total, pickle_bytes_upper=pickle_bytes,
        storage_count=count, record_count=records, raw_tensor_bytes=sum(storage_sizes),
        fixed_record_payload_bytes=zip_bound.FIXED_RECORD_PAYLOAD_BYTES,
        zip_overhead_bytes_upper=overhead, alignment_padding_bytes_upper=63*records,
        first_encounter_order_assumed=False)


def _contiguous_stride(shape):
    return tuple(prod(shape[index+1:]) for index in range(len(shape)))


def tensor_reduction_ceiling(rows):
    """Sum one no-alias descriptor cost per maximal-width inventory row.

    The admitted actual shapes/strides/storage sizes must be bounded by these
    producer-derived descriptors, not merely pass Tensor.is_contiguous().
    """
    return sum(pickle_bound.protocol2_tensor_bytes(dtype_name="torch." + row["dtype"],
        storage_nbytes=row["nbytes"], shape=row["shape"], stride=_contiguous_stride(row["shape"]),
        tensor_count_cap=len(rows)) for row in rows)


def combine(primitive_component_costs):
    """Combine supplied reviewed primitive maxima (PROTO/STOP excluded).

    This interface validates costs/membership, NOT their evidence. A caller's
    invented smaller numbers must never be treated as authenticated admission.
    """
    layouts = inventory.component_layouts()
    expected = set(layouts) - {"capture_comparison_pilot"}
    if (type(primitive_component_costs) is not dict or set(primitive_component_costs) != expected
            or any(type(k) is not str for k in primitive_component_costs)
            or any(type(n) is not int or n <= 0 for n in primitive_component_costs.values())):
        raise ValueError("exact10component positive primitive-cost mapping required")
    components = {}
    for key, rows in layouts.items():
        count = inventory.COMPONENT_COUNTS[key]
        if key == "capture_comparison_pilot":
            row = dict(encoding="bytes", payload_count=count, body_bytes_upper=JSON_COMPARISON_BYTES,
                       aggregate_body_bytes_upper=count*JSON_COMPARISON_BYTES)
        else:
            tensor_cost = tensor_reduction_ceiling(rows)
            primitive = primitive_component_costs[key]
            archive = order_independent_zip_ceiling(3 + primitive + tensor_cost,
                                                    tuple(row["nbytes"] for row in rows))
            row = dict(encoding="torch_weights_only", payload_count=count,
                primitive_subtree_bytes_upper=primitive, pickle_proto_stop_bytes=3,
                tensor_reduction_bytes_upper=tensor_cost, **archive,
                body_bytes_upper=archive["archive_bytes_upper"],
                aggregate_body_bytes_upper=count*archive["archive_bytes_upper"])
        components[key] = row
    bodies = sum(row["aggregate_body_bytes_upper"] for row in components.values())
    total = bodies + NONBODY_COMPLETE_LEDGER_BYTES
    return dict(schema="i7_prospective_native_storage_topology_bound_v1", components=components,
        payload_count=82, torch_payload_count=81, json_payload_count=1,
        raw_tensor_bytes_upper=inventory.summary()["aggregate_raw_tensor_bytes"],
        scientific_body_bytes_upper=bodies, nonbody_complete_ledger_bytes=NONBODY_COMPLETE_LEDGER_BYTES,
        complete_logical_bytes_upper=total, store_budget_bytes=STORE_BUDGET_BYTES,
        conditional_residual_bytes=STORE_BUDGET_BYTES-total,
        conditional_arithmetic_fits=total<=STORE_BUDGET_BYTES,
        independent_per_artifact_audit_maxima=True,
        actual_runtime_domains_validated=False, source_manifest_authenticated=False,
        native_layout_observed=False, storage_fit_proven=False, execution_authorized=False,
        scientific_execution_certified=False)


def compute():
    """Closed calculation from the reviewed three domain-specific calculators."""
    import audit_primitive_bound as audit
    import branch_primitive_bound as branch
    primitive = core.core_component_bounds()
    branch_payload = branch.compute_branch_payload_bound()["maxima"]["payload_primitive_pickle_bytes_upper"]
    primitive["branch_results"] = max(core.envelope_cost(branch_payload, role=r, bundle=b, update=u,
        kind="branch-results") for r,b,u in core.fixed_identities())
    # Each audit helper already includes its exact envelope and three root bytes.
    # Taking max for all16 is deliberately stronger than the fail-stop prefix sum.
    primitive["independent_audit_all_failure"] = max(audit.audit_artifact_pickle_bytes(
        role=r, bundle=b, update=u)-3 for r,b,u in core.fixed_identities() if r != "pilot")
    return combine(primitive)


if __name__ == "__main__":
    print("I7 conditional topology arithmetic only; no payload, runtime or execution admission.")
