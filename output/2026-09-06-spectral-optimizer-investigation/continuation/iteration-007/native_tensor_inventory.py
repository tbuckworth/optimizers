"""Pure maximum-width tensor inventory, not a serialized-size or native-fit proof.

These are prospective descriptors for the unchanged scientific schemas. Import,
default CLI and all functions are Torch-free and perform no filesystem work.
Actual runtime layout and every primitive/container byte need separate admission.
"""
from __future__ import annotations

from math import prod


P, RANK = 50_890, 32
CPU_RNG_BYTES, CUDA_RNG_BYTES = 5_056, 16
PARAMETERS = ((0, "0.weight", (64, 784)), (1, "0.bias", (64,)),
              (2, "2.weight", (10, 64)), (3, "2.bias", (10,)))
DTYPE_BYTES = {"float32": 4, "float64": 8, "int64": 8, "uint8": 1, "uint32": 4}
BRANCHES = ("raw", "current", "lagged", "restored", "reciprocal", "zero")
PROBES = ("batch_noisy", "train_probe_noisy", "train_probe_clean", "auxiliary_clean")
COMPONENT_COUNTS = {
    "anchor_pilot": 2, "anchor_long": 16, "source_witness": 18,
    "branch_results": 18, "independent_audit_all_failure": 16,
    "source_completion_pilot_on": 1, "source_completion_pilot_off": 1,
    "capture_comparison_pilot": 1, "source_completion_long_on": 4,
    "plan_pilot": 1, "plan_long": 4,
}


def component_layouts() -> dict[str, tuple[dict, ...]]:
    """Return owned full-rank path/dtype/shape maxima for all eleven components.

    Order is descriptor order, NOT a claim about pickle's first-visit storage
    order. ZIP entry order must come from the bound traversal of the actual tree.
    Undefined branches or absent observer bases can reduce the tensor inventory;
    their primitive failure/domain fields are not bounded by this function.
    """
    def add(rows, path, dtype, shape):
        rows.append({"path": path, "dtype": dtype, "shape": shape,
                     "nbytes": prod(shape) * DTYPE_BYTES[dtype]})

    def parameters(rows, prefix):
        for index, _, shape in PARAMETERS:
            add(rows, prefix + (index, "value"), "float32", shape)

    def optimizer(rows, prefix):
        for index, _, shape in PARAMETERS:
            add(rows, prefix + ("state", index, "step"), "float32", ())
            for moment in ("exp_avg", "exp_avg_sq"):
                add(rows, prefix + ("state", index, moment, "value"), "float32", shape)

    def observer(rows, prefix):
        for field, dtype, shape in (("V", "float32", (P, RANK)),
                                     ("S", "float64", (RANK,)),
                                     ("grad_mean", "float32", (P,))):
            add(rows, prefix + ("state", field, "value"), dtype, shape)

    def core(prefix):
        rows = []
        parameters(rows, prefix + ("model", "parameters"))
        optimizer(rows, prefix + ("optimizer",))
        observer(rows, prefix + ("observer",))
        for path, dtype, shape in (
            (("numpy", "keys"), "uint32", (624,)),
            (("torch_cpu",), "uint8", (CPU_RNG_BYTES,)),
            (("torch_cuda", 0, "state"), "uint8", (CUDA_RNG_BYTES,)),
            (("continuation_witness", "torch_cpu"), "float64", (4,)),
            (("continuation_witness", "torch_cuda", 0), "float32", (4,)),
        ):
            add(rows, prefix + ("rng",) + path, dtype, shape)
        return rows

    def anchor():
        rows = core(("payload",))
        for path, count in ((("plan", "next_batch_indices"), 64),
                            (("probes", "training_probe_indices"), 256),
                            (("probes", "auxiliary_indices"), 5_000)):
            add(rows, ("payload", "bindings") + path, "int64", (count,))
        return rows

    witness = []
    for key in ("raw_gradient", "delivered_current_gradient"):
        add(witness, ("payload", key, "value"), "float32", (P,))
    parameters(witness, ("payload", "parameters_after"))
    optimizer(witness, ("payload", "optimizer_after"))
    observer(witness, ("payload", "observer_after"))

    branches = []
    candidate = ("payload", "candidate_state")
    for key in ("g", "c", "l"):
        add(branches, candidate + (key, "value"), "float32", (P,))
    for key in ("previous_basis", "post_ingest_basis"):
        add(branches, candidate + (key, "value"), "float32", (P, RANK))
    observer(branches, candidate + ("observer_after",))
    for probe in PROBES:
        add(branches, candidate + ("measurement_before", "probes", probe, "cpu64", "q", "value"),
            "float64", (P,))
    for branch in BRANCHES:
        prefix = ("payload", "branches", branch)
        add(branches, prefix + ("delivered_gradient", "value"), "float32", (P,))
        add(branches, prefix + ("parameters_after_flat",), "float32", (P,))
        parameters(branches, prefix + ("parameters_after",))
        optimizer(branches, prefix + ("optimizer_after",))
        for key in ("delta", "delta_data"):
            add(branches, prefix + ("measurement", "displacement", key, "value"), "float64", (P,))

    def plan(steps):
        rows = []
        for key, dtype, shape in (
            ("permutation", "int64", (60_000,)),
            ("train_indices", "int64", (5_000,)),
            ("validation_indices", "int64", (5_000,)),
            ("auxiliary_indices", "int64", (5_000,)),
            ("replacement_uniforms", "float64", (5_000,)),
            ("replacement_digits", "int64", (5_000,)),
            ("training_batches", "int64", (steps, 64)),
            ("training_probe_indices", "int64", (256,)),
        ):
            add(rows, (key,), dtype, shape)
        return rows

    layouts = {
        "anchor_pilot": anchor(), "anchor_long": anchor(),
        "source_witness": witness, "branch_results": branches,
        "independent_audit_all_failure": [],
        "source_completion_pilot_on": core(("final_state_core",)),
        "source_completion_pilot_off": core(("final_state_core",)),
        "capture_comparison_pilot": [],
        "source_completion_long_on": core(("final_state_core",)),
        "plan_pilot": plan(220), "plan_long": plan(2_000),
    }
    assert tuple(layouts) == tuple(COMPONENT_COUNTS)
    assert sum(prod(shape) for _, _, shape in PARAMETERS) == P
    for rows in layouts.values():
        assert len(rows) == len({row["path"] for row in rows})
    return {key: tuple(rows) for key, rows in layouts.items()}


def summary() -> dict:
    """Report only conditional raw tensor arithmetic; never storage admission."""
    components = {
        key: {"payload_count": COMPONENT_COUNTS[key], "tensor_count": len(rows),
              "raw_tensor_bytes": sum(row["nbytes"] for row in rows)}
        for key, rows in component_layouts().items()
    }
    return {
        "schema": "i7_prospective_native_tensor_inventory_v1",
        "components": components,
        "payload_count": sum(COMPONENT_COUNTS.values()),
        "torch_payload_count": sum(COMPONENT_COUNTS.values()) - 1,
        "json_payload_count": 1,
        "aggregate_tensor_count": sum(row["payload_count"] * row["tensor_count"]
                                      for row in components.values()),
        "aggregate_raw_tensor_bytes": sum(row["payload_count"] * row["raw_tensor_bytes"]
                                          for row in components.values()),
        "full_rng_core_count": 24,
        "fixture_absent_cuda_storage_count": 48,
        "fixture_absent_cuda_raw_bytes": 24 * (CUDA_RNG_BYTES + 4 * DTYPE_BYTES["float32"]),
        "native_layout_observed": False,
        "serialized_size_bounded": False,
        "storage_fit_proven": False,
        "execution_authorized": False,
        "scientific_execution_certified": False,
    }


if __name__ == "__main__":
    print("I7 prospective tensor inventory only; no runtime observation or storage admission.")
