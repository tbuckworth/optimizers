"""Torch-free primitive/container bound for the 18 I7 branch-result payloads.

The arithmetic covers only the value below the common envelope's ``payload``
key. Tensor reductions and raw storages have zero cost here and must be added
by the tensor-pickle, raw-storage and ZIP bounds. Import and the default CLI
perform no filesystem reads, construct no scientific object, and authorize no
execution.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import stat

import primitive_storage_bound as primitive


SCHEMA = "i7_branch_payload_primitive_bound_v1"
PAYLOAD_COUNT = 18
BRANCHES = ("raw", "current", "lagged", "restored", "reciprocal", "zero")
PROBES = ("batch_noisy", "train_probe_noisy", "train_probe_clean", "auxiliary_clean")
PARAMETERS = (
    ("0.weight", (64, 784)), ("0.bias", (64,)),
    ("2.weight", (10, 64)), ("2.bias", (10,)),
)
COEFFICIENTS = {
    "ordering": {"lagged": 1, "current": -1},
    "direction_at_current_norm": {"restored": 1, "current": -1},
    "direction_at_lagged_norm": {"lagged": 1, "reciprocal": -1},
    "norm_at_current_direction": {"current": 1, "reciprocal": -1},
    "norm_at_lagged_direction": {"restored": 1, "lagged": -1},
    "interaction": {"restored": 1, "lagged": -1, "current": -1, "reciprocal": 1},
    "current_minus_raw": {"current": 1, "raw": -1},
    "lagged_minus_raw": {"lagged": 1, "raw": -1},
    "restored_minus_raw": {"restored": 1, "raw": -1},
    "reciprocal_minus_raw": {"reciprocal": 1, "raw": -1},
    "raw_minus_zero": {"raw": 1, "zero": -1},
    "current_minus_zero": {"current": 1, "zero": -1},
    "lagged_minus_zero": {"lagged": 1, "zero": -1},
    "restored_minus_zero": {"restored": 1, "zero": -1},
    "reciprocal_minus_zero": {"reciprocal": 1, "zero": -1},
}
PAIR_KEYS = tuple(
    f"{left}__{right}" for index, left in enumerate(BRANCHES)
    for right in BRANCHES[index + 1:]
)
VARIANTS = ("all_defined", "restored_undefined", "reciprocal_undefined")
UNDEFINED_REASON = {
    "restored": "positive_current_norm_zero_lagged_direction",
    "reciprocal": "positive_lagged_norm_zero_current_direction",
}
SOURCE_SHA256 = {
    "output/2026-09-06-spectral-optimizer-investigation/continuation/iteration-007/artifact_envelopes.py":
        "dfe3558f5930acfcf24f2458c7186ab62f283076854b8743959e8ef7925bb441",
    "output/2026-09-06-spectral-optimizer-investigation/continuation/iteration-007/branch_execution.py":
        "9f09bd255ce1edde5203fbaace7c80858bebea3e28ad54d56d69688e27196c1d",
    "output/2026-09-06-spectral-optimizer-investigation/continuation/iteration-007/response_math.py":
        "7645ca1b65d29c4278ccd1795e817ec6e8438ed8e6fa418e1d02e3d1d5b2276b",
    "output/2026-09-06-spectral-optimizer-investigation/continuation/iteration-007/loss_measurements.py":
        "eca1a3b7c0eb2366b6db7b3c2fb124c0fb89a8b0cd8844219cd679e7d5b15b71",
    "output/2026-09-06-spectral-optimizer-investigation/continuation/iteration-007/measurement_assembly.py":
        "0b8c305e60a97eb5a0483588be3d0962b8afeed512c13ac495e9ed487d49fba8",
    "output/2026-09-06-spectral-optimizer-investigation/continuation/iteration-007/state_core.py":
        "a727ce516d6d3e283bdbb75162a8b1701198d6d9cffd40a5a70a005c83a43c7e",
    "output/2026-09-06-spectral-optimizer-investigation/continuation/iteration-007/anchor_envelope.py":
        "d2290e07044d5ef26cef3b72e5427b00216848680f36b709fa547d944081fc17",
    "output/2026-09-06-spectral-optimizer-investigation/continuation/iteration-007/source_capture.py":
        "b4fd71331040c82e1d28574e1df34f5da2eb78e9de9fd890877ff65ce2866254",
    "output/2026-09-06-spectral-optimizer-investigation/continuation/iteration-007/identity_codec.py":
        "e47c5de57e3bccb47fd96cc5183e35f614577a16867f7caf04819e562df15f5d",
    "output/2026-09-06-spectral-optimizer-investigation/continuation/iteration-007/native_tensor_inventory.py":
        "48266196d4837f146001b2b4e67fb61efcfa7331d0ad895342a56ce466d5b38a",
    "output/2026-09-06-spectral-optimizer-investigation/continuation/iteration-007/pickle_storage_bound.py":
        "df3843821ceb7eeeae69b1a36d81ba231d1d1b920286c753f257f4d4d9dbddf0",
    "output/2026-09-06-spectral-optimizer-investigation/continuation/iteration-007/zip_storage_bound.py":
        "77f0e0136bfaad3ac1dc826f8a1db0389316f3408cd758cf9b26640db191bbb3",
    "output/2026-09-06-spectral-optimizer-investigation/continuation/iteration-007/primitive_storage_bound.py":
        "6414824202152ed6cae710007e0e55564657ef9f406ede09ede3c5ac84a50709",
    "spectral_filter.py":
        "9280c7d360aef4c775baf0d781a5afd56e016e9cfb93ef829c3ae4b8cc3df943",
}

_HASH = primitive.text(64)
_ARTIFACT_ID = primitive.text(128)
_N_PARAMS = 50_890
_UPDATE = primitive.integer(101, 2_000)


def _shape(*dimensions: int | tuple[int, int]) -> int:
    costs = []
    for dimension in dimensions:
        costs.append(primitive.integer(*dimension) if type(dimension) is tuple
                     else primitive.literal(dimension))
    return primitive.plist(costs)


def _native(shape: tuple[int | tuple[int, int], ...], dtype: str = "torch.float32",
            device: str = "cuda:0") -> int:
    return primitive.pdict({
        "value": 0,
        "shape": _shape(*shape),
        "native_dtype": primitive.literal(dtype),
        "native_device": primitive.literal(device),
    })


def _vector_record() -> int:
    return primitive.pdict({
        "value": 0, "shape": _shape(_N_PARAMS),
        "dtype": primitive.literal("float64"), "device": primitive.literal("cpu"),
        "sha256": _HASH,
    })


def _concordance() -> int:
    status = max(primitive.literal("within_scale"), primitive.literal("discordant"))
    return primitive.pdict({
        "abs_discrepancy": primitive.FLOAT,
        "descriptive_ceiling": primitive.FLOAT,
        "status": status,
    })


def _geometry(mask_names: tuple[str, ...], defined: bool) -> int:
    mask = primitive.pdict({name: primitive.BOOL for name in mask_names})
    if not defined:
        return primitive.pdict({
            "defined": primitive.BOOL, "defined_mask": mask,
            "reason": primitive.literal("domain_undefined_required_branch"),
            "distance": primitive.NULL, "cosine": primitive.NULL,
            "cosine_reason": primitive.NULL, "left_norm": primitive.NULL,
            "right_norm": primitive.NULL,
        })
    nonzero = primitive.pdict({
        "defined": primitive.BOOL, "defined_mask": mask, "reason": primitive.NULL,
        "distance": primitive.FLOAT, "cosine": primitive.FLOAT,
        "cosine_reason": primitive.NULL, "left_norm": primitive.FLOAT,
        "right_norm": primitive.FLOAT,
    })
    zero = primitive.pdict({
        "defined": primitive.BOOL, "defined_mask": mask, "reason": primitive.NULL,
        "distance": primitive.FLOAT, "cosine": primitive.NULL,
        "cosine_reason": primitive.literal("zero_norm"), "left_norm": primitive.FLOAT,
        "right_norm": primitive.FLOAT,
    })
    return max(nonzero, zero)


def _parameter_entry(index: int, name: str, shape: tuple[int, ...]) -> int:
    return primitive.pdict({
        "index": primitive.literal(index), "name": primitive.literal(name),
        "shape": _shape(*shape), "requires_grad": primitive.BOOL,
        "native_dtype": primitive.literal("torch.float32"),
        "native_device": primitive.literal("cuda:0"), "value": 0,
    })


def _parameters() -> int:
    return primitive.plist([
        _parameter_entry(index, name, shape)
        for index, (name, shape) in enumerate(PARAMETERS)
    ])


def _optimizer() -> int:
    names = [name for name, _ in PARAMETERS]
    group = primitive.literal({
        "lr": 0.001, "betas": (0.9, 0.999), "eps": 1e-8,
        "weight_decay": 0.01, "amsgrad": False, "maximize": False,
        "foreach": False, "capturable": False, "differentiable": False,
        "fused": False, "decoupled_weight_decay": True,
        "param_indices": [0, 1, 2, 3],
    })
    states = []
    for index, (name, shape) in enumerate(PARAMETERS):
        states.append(primitive.pdict({
            "parameter_index": primitive.literal(index),
            "parameter_name": primitive.literal(name), "step": 0,
            "exp_avg": _native(shape), "exp_avg_sq": _native(shape),
        }))
    return primitive.pdict({
        "class_name": primitive.literal("torch.optim.AdamW"),
        "state_completed_updates": _UPDATE,
        "parameter_order": primitive.literal(names),
        "param_groups": primitive.plist([group]), "state": primitive.plist(states),
    })


def _observer() -> int:
    config = primitive.literal({
        "rank": 32, "decay": 0.99, "warmup": 100, "filter_strength": 1.0,
        "energy_threshold": None, "adaptive": "none", "normalize": "none",
        "weighting": "hard", "alpha": 1.0, "soft_residual": True,
        "stable_update": True, "relative_eig_tol": 1e-8,
        "absolute_eig_floor": 0.0, "stabilize_every": 100, "n_params": _N_PARAMS,
    })
    state = primitive.pdict({
        "V": _native((_N_PARAMS, (1, 32))),
        "S": _native(((1, 32),), "torch.float64", "cpu"),
        "proj_k": primitive.NULL, "step_count": _UPDATE,
        "grad_mean": _native((_N_PARAMS,)),
        "stabilization_count": primitive.integer(0, 4_000),
        "max_orthogonality_error": primitive.FLOAT,
    })
    return primitive.pdict({
        "class_name": primitive.literal("SpectralGradientFilter"),
        "state_completed_observations": _UPDATE,
        "excluded_live_aliases": primitive.literal(["model", "base_optimizer", "param_list"]),
        "config": config, "state": state,
    })


def _before_chunk(index: int) -> int:
    start, end = index * 500, (index + 1) * 500
    return primitive.pdict({
        "chunk_index": primitive.literal(index), "start": primitive.literal(start),
        "end": primitive.literal(end), "count": primitive.literal(500),
        "input_sha256": _HASH, "label_sha256": _HASH,
        "cpu64_before_ce": primitive.FLOAT, "native32_before_ce": primitive.FLOAT,
        "concordance": _concordance(),
    })


def _before_probe(probe: str, sample_count: int) -> int:
    chunks = (primitive.plist([_before_chunk(index) for index in range(10)])
              if probe == "auxiliary_clean" else primitive.plist([]))
    return primitive.pdict({
        "sample_count": primitive.literal(sample_count),
        "sample_identity_sha256": _HASH,
        "cpu64": primitive.pdict({
            "before_ce": primitive.FLOAT, "q": _vector_record(), "q_norm": primitive.FLOAT,
        }),
        "native32": primitive.pdict({"before_ce": primitive.FLOAT}),
        "concordance": _concordance(), "auxiliary_chunks": chunks,
    })


def _measurement_before() -> int:
    counts = (64, 256, 256, 5_000)
    return primitive.pdict({
        "probe_order": primitive.literal(list(PROBES)),
        "probes": primitive.pdict({
            probe: _before_probe(probe, count) for probe, count in zip(PROBES, counts)
        }),
    })


def _branch_chunk(index: int) -> int:
    start, end = index * 500, (index + 1) * 500
    return primitive.pdict({
        "chunk_index": primitive.literal(index), "start": primitive.literal(start),
        "end": primitive.literal(end), "count": primitive.literal(500),
        "input_sha256": _HASH, "label_sha256": _HASH,
        "cpu64_after_ce": primitive.FLOAT, "cpu64_Y": primitive.FLOAT,
        "native32_after_ce": primitive.FLOAT, "native32_Y": primitive.FLOAT,
        "concordance": primitive.pdict({"after": _concordance(), "Y": _concordance()}),
    })


def _branch_probe(probe: str) -> int:
    chunks = (primitive.plist([_branch_chunk(index) for index in range(10)])
              if probe == "auxiliary_clean" else primitive.plist([]))
    return primitive.pdict({
        "before_binding": primitive.pdict({
            "measurement_before_artifact_id": _ARTIFACT_ID,
            "probe_key": primitive.literal(probe),
            "before_ce_cpu64_sha256": _HASH, "q_sha256": _HASH,
        }),
        "cpu64": primitive.pdict({
            "after_ce": primitive.FLOAT, "Y": primitive.FLOAT, "D": primitive.FLOAT,
            "Ddata": primitive.FLOAT, "R": primitive.FLOAT,
        }),
        "native32": primitive.pdict({"after_ce": primitive.FLOAT, "Y": primitive.FLOAT}),
        "concordance": primitive.pdict({"after": _concordance(), "Y": _concordance()}),
        "auxiliary_chunks": chunks,
    })


def _branch_measurement(branch: str) -> int:
    displacement = primitive.pdict({
        "delta": _vector_record(), "delta_data": _vector_record(),
        "delta_norm": primitive.FLOAT, "delta_squared_norm": primitive.FLOAT,
        "delta_data_norm": primitive.FLOAT, "delta_data_squared_norm": primitive.FLOAT,
        "delta_delta_data": _geometry((branch,), True),
    })
    return primitive.pdict({
        "displacement": displacement,
        "probes": primitive.pdict({probe: _branch_probe(probe) for probe in PROBES}),
    })


def _defined_branch(branch: str) -> int:
    return primitive.pdict({
        "status": primitive.literal("defined"), "reason": primitive.NULL,
        "delivered_gradient": _native((_N_PARAMS,)),
        "assigned_gradient_null_mask": primitive.repeat_list(4, primitive.BOOL),
        "parameters_after": _parameters(), "parameters_after_flat": 0,
        "optimizer_after": _optimizer(), "measurement": _branch_measurement(branch),
    })


def _undefined_branch(branch: str) -> int:
    return primitive.pdict({
        "status": primitive.literal("undefined"),
        "reason": primitive.literal(UNDEFINED_REASON[branch]),
        "delivered_gradient": primitive.NULL,
        "assigned_gradient_null_mask": primitive.NULL,
        "parameters_after": primitive.NULL, "parameters_after_flat": primitive.NULL,
        "optimizer_after": primitive.NULL, "measurement": primitive.NULL,
    })


def _leverage(domain: str) -> int:
    if domain == "both_positive":
        nullable = (primitive.FLOAT, primitive.FLOAT, primitive.NULL,
                    primitive.FLOAT, primitive.FLOAT, primitive.NULL)
    elif domain == "both_zero":
        nullable = (primitive.NULL, primitive.NULL, primitive.literal("zero_lagged_norm"),
                    primitive.NULL, primitive.NULL, primitive.literal("zero_norm"))
    elif domain == "restored_undefined":
        nullable = (primitive.FLOAT, primitive.NULL, primitive.literal("zero_lagged_norm"),
                    primitive.NULL, primitive.NULL, primitive.literal("zero_norm"))
    elif domain == "reciprocal_undefined":
        nullable = (primitive.FLOAT, primitive.FLOAT, primitive.NULL,
                    primitive.NULL, primitive.NULL, primitive.literal("zero_norm"))
    else:  # internal closed call set
        raise ValueError("unknown leverage domain")
    separation, ratio, ratio_reason, direction, cosine, direction_reason = nullable
    return primitive.pdict({
        "current_norm": primitive.FLOAT, "lagged_norm": primitive.FLOAT,
        "signed_norm_difference": primitive.FLOAT,
        "relative_norm_separation": separation,
        "current_lagged_norm_ratio": ratio, "ratio_reason": ratio_reason,
        "unit_direction_distance": direction, "unit_direction_cosine": cosine,
        "direction_reason": direction_reason,
        "norm_leverage": primitive.BOOL, "direction_leverage": primitive.BOOL,
    })


def _candidate(undefined_branch: str | None) -> int:
    domain = ("all_defined" if undefined_branch is None else undefined_branch + "_undefined")
    leverage = (max(_leverage("both_positive"), _leverage("both_zero"))
                if domain == "all_defined" else _leverage(domain))
    mask = primitive.pdict({name: primitive.BOOL for name in BRANCHES})
    reasons = primitive.pdict({
        name: (primitive.literal(UNDEFINED_REASON[name]) if name == undefined_branch
               else primitive.NULL) for name in BRANCHES
    })
    return primitive.pdict({
        "g": _native((_N_PARAMS,)), "c": _native((_N_PARAMS,)),
        "l": _native((_N_PARAMS,)),
        "previous_basis": _native((_N_PARAMS, (1, 32))),
        "post_ingest_basis": _native((_N_PARAMS, (1, 32))),
        "nc": primitive.FLOAT, "nl": primitive.FLOAT,
        "observer_after": _observer(), "measurement_before": _measurement_before(),
        "leverage": leverage,
        "zero_cases": primitive.pdict({
            "current_zero": primitive.BOOL, "lagged_zero": primitive.BOOL,
            "branch_defined_mask": mask, "branch_reasons": reasons,
        }),
    })


def _pair(pair_key: str, defined: bool) -> int:
    left, right = pair_key.split("__")
    mask_names = (left, right)
    if not defined:
        return primitive.pdict({
            "defined": primitive.BOOL,
            "defined_mask": primitive.pdict({name: primitive.BOOL for name in mask_names}),
            "reason": primitive.literal("domain_undefined_required_branch"),
            "delivered_gradient": _geometry(mask_names, False),
            "delta": _geometry(mask_names, False), "delta_data": _geometry(mask_names, False),
            "full_data_distance_discrepancy": primitive.NULL,
            "full_data_rounding_ceiling": primitive.NULL,
            "rounding_status": primitive.literal("domain_undefined"),
        })
    return primitive.pdict({
        "defined": primitive.BOOL,
        "defined_mask": primitive.pdict({name: primitive.BOOL for name in mask_names}),
        "reason": primitive.NULL, "delivered_gradient": _geometry(mask_names, True),
        "delta": _geometry(mask_names, True), "delta_data": _geometry(mask_names, True),
        "full_data_distance_discrepancy": primitive.FLOAT,
        "full_data_rounding_ceiling": primitive.FLOAT,
        "rounding_status": primitive.literal("pass"),
    })


def _scalar_record(coefficients: dict[str, int]) -> int:
    return primitive.pdict({
        "component_values": primitive.pdict({name: primitive.FLOAT for name in coefficients}),
        "direct_value": primitive.FLOAT, "component_sum_value": primitive.FLOAT,
        "identity_abs_discrepancy": primitive.FLOAT,
        "identity_rounding_ceiling": primitive.FLOAT,
        "identity_status": primitive.literal("pass"),
    })


def _contrast_chunk(index: int, coefficients: dict[str, int]) -> int:
    start, end = index * 500, (index + 1) * 500
    scalar = _scalar_record(coefficients)
    return primitive.pdict({
        "chunk_index": primitive.literal(index), "start": primitive.literal(start),
        "end": primitive.literal(end), "count": primitive.literal(500),
        "input_sha256": _HASH, "label_sha256": _HASH,
        "cpu64": primitive.pdict({"Y": scalar}),
        "native32": primitive.pdict({"Y": scalar}),
        "concordance": primitive.pdict({"Y": _concordance()}),
    })


def _contrast_probe(probe: str, coefficients: dict[str, int]) -> int:
    scalar = _scalar_record(coefficients)
    chunks = (primitive.plist([_contrast_chunk(index, coefficients) for index in range(10)])
              if probe == "auxiliary_clean" else primitive.plist([]))
    return primitive.pdict({
        "cpu64": primitive.pdict({
            "Y": scalar, "D": scalar, "Ddata": scalar, "R": scalar,
        }),
        "native32": primitive.pdict({"Y": scalar}),
        "concordance": primitive.pdict({"Y": _concordance()}),
        "auxiliary_chunks": chunks,
    })


def _factor_requirements(name: str) -> tuple[str, ...]:
    if name in ("direction_at_current_norm", "direction_at_lagged_norm"):
        return ("direction",)
    if name in ("norm_at_current_direction", "norm_at_lagged_direction"):
        return ("norm",)
    if name == "interaction":
        return ("direction", "norm")
    return ()


def _factor_status(kind: str, requirements: tuple[str, ...], defined: bool,
                   undefined_branch: str | None) -> int:
    if kind not in requirements:
        return primitive.literal("not_required")
    if not defined or undefined_branch is not None and kind == "direction":
        return primitive.literal("unavailable")
    if undefined_branch is not None:  # one-zero domains have exact norm leverage
        return primitive.literal("qualified")
    return max(primitive.literal("unavailable"), primitive.literal("qualified"),
               primitive.literal("weak"))


def _contrast(name: str, coefficients: dict[str, int], defined: bool,
              undefined_branch: str | None) -> int:
    requirements = _factor_requirements(name)
    base = {
        "coefficients": primitive.literal(coefficients),
        "branch_defined_mask": primitive.pdict({branch: primitive.BOOL for branch in BRANCHES}),
        "defined": primitive.BOOL,
        "reason": (primitive.NULL if defined else
                   primitive.literal("domain_undefined_required_branch")),
        "factor_requirements": primitive.literal(list(requirements)),
        "factor_leverage": primitive.pdict({
            kind: _factor_status(kind, requirements, defined, undefined_branch)
            for kind in ("direction", "norm")
        }),
    }
    if not defined:
        return primitive.pdict({**base, "vector": primitive.NULL, "probes": primitive.NULL})
    summary = primitive.pdict({
        "representation": primitive.literal("derived_from_bound_branch_vectors"),
        "canonical_sha256": _HASH, "norm": primitive.FLOAT,
        "squared_norm": primitive.FLOAT, "component_count": primitive.literal(_N_PARAMS),
    })
    vector = primitive.pdict({
        "delta": summary, "delta_data": summary,
        "full_data_agreement": primitive.pdict({
            "difference_norm": primitive.FLOAT, "rounding_ceiling": primitive.FLOAT,
            "status": primitive.literal("pass"),
        }),
    })
    return primitive.pdict({
        **base, "vector": vector,
        "probes": primitive.pdict({
            probe: _contrast_probe(probe, coefficients) for probe in PROBES
        }),
    })


def _reference(schema_name: str) -> int:
    return primitive.pdict({
        "artifact_id": _ARTIFACT_ID, "schema_name": primitive.literal(schema_name),
        "name": primitive.text(131), "size_bytes": primitive.integer(1, 1 << 30),
        "sha256": _HASH, "status": primitive.literal("complete"),
        "encoding": primitive.literal("torch_weights_only"),
        "receipt_name": primitive.text(77),
        "receipt_size_bytes": primitive.integer(1, 4_096),
        "receipt_sha256": _HASH,
    })


def _clone_row(defined: bool) -> int:
    return primitive.pdict({
        "executed": primitive.BOOL,
        "start_core_sha256": _HASH if defined else primitive.NULL,
        "start_directly_equal": primitive.BOOL,
        "storage_disjoint": primitive.BOOL,
        "checked_live_tensor_count": primitive.literal(19 if defined else 0),
    })


def _execution_row(defined: bool) -> int:
    if defined:
        return primitive.pdict({
            "parameters_sha256": _HASH, "optimizer_sha256": _HASH,
            "assigned_gradient_sha256": _HASH, "gradient_null_mask_sha256": _HASH,
            "observer_unchanged_sha256": _HASH, "post_step_null_mask_sha256": _HASH,
            "rng_sha256": _HASH, "optimizer_updates_after": _UPDATE,
        })
    return primitive.pdict({name: primitive.NULL for name in (
        "parameters_sha256", "optimizer_sha256", "assigned_gradient_sha256",
        "gradient_null_mask_sha256", "observer_unchanged_sha256",
        "post_step_null_mask_sha256", "rng_sha256", "optimizer_updates_after",
    )})


def _audit(undefined_branch: str | None) -> int:
    defined = {name: name != undefined_branch for name in BRANCHES}
    clone_order = primitive.pdict({name: _clone_row(defined[name]) for name in BRANCHES})
    execution_order = primitive.pdict({name: _execution_row(defined[name]) for name in BRANCHES})
    match = primitive.pdict({
        "witness_sha256": _HASH, "replay_sha256": _HASH,
        "directly_equal": primitive.BOOL,
    })
    replay_checks = primitive.pdict({
        "raw_gradient": match, "current_gradient": match, "delivered_gradient": match,
        "parameters_after": match, "optimizer_after": match, "observer_after": match,
        "rng": primitive.pdict({
            "anchor_rng_sha256": _HASH, "witness_rng_sha256": _HASH,
            "replay_rng_sha256": _HASH, "replay_directly_equals_anchor": primitive.BOOL,
        }),
        "live_loss": match,
    })
    return primitive.pdict({
        "producer_bindings": primitive.pdict({
            "anchor_ref": _reference("i7_anchor"),
            "source_witness_ref": _reference("i7_source_step_witness"),
            "sources_tree_sha256": _HASH, "environment_tree_sha256": _HASH,
            "formula_ids": primitive.literal(["i7_response_math_v1", "i7_assembly_roundoff_v1"]),
            "tolerance_ids": primitive.literal([
                "native_exact_v1", "delivery_relative_1e-6_v1", "i7_numerical_contract_v1",
            ]),
        }),
        "candidate_proof": primitive.pdict({
            "observer_updates_before": primitive.integer(100, 1_999),
            "observer_ingests": primitive.literal(1), "observer_updates_after": _UPDATE,
            "candidate_core_sha256": _HASH,
            "candidate_core_after_canonical_sha256": _HASH,
            "candidate_core_after_reverse_sha256": _HASH,
            "candidate_core_directly_unchanged": primitive.BOOL,
        }),
        "clone_independence_proof": primitive.pdict({
            "anchor_core_sha256": _HASH,
            "per_order": primitive.pdict({"canonical": clone_order, "reverse": clone_order}),
            "anchor_core_after_sha256": _HASH, "anchor_directly_unchanged": primitive.BOOL,
        }),
        "branch_execution_proof": primitive.pdict({
            "canonical_order": primitive.literal(list(BRANCHES)),
            "verification_order": primitive.literal(list(reversed(BRANCHES))),
            "observer_ingests": primitive.literal(0),
            "per_order": primitive.pdict({
                "canonical": execution_order, "reverse": execution_order,
            }),
            "per_branch_directly_equal": primitive.pdict({
                name: primitive.BOOL for name in BRANCHES
            }),
            "all_defined_order_invariant": primitive.BOOL,
        }),
        "current_replay_proof": primitive.pdict({
            "source_witness_artifact_id": _ARTIFACT_ID,
            "checks": replay_checks, "overall_status": primitive.literal("exact"),
        }),
        "completion": primitive.pdict({
            "defined_branch_count": primitive.literal(5 if undefined_branch else 6),
            "undefined_branch_count": primitive.literal(1 if undefined_branch else 0),
            "required_branch_keys_present": primitive.BOOL,
            "required_comparison_keys_present": primitive.BOOL,
            "exact_replay_complete": primitive.BOOL,
            "producer_artifact_status": primitive.literal("complete"),
        }),
    })


def _payload(undefined_branch: str | None) -> int:
    defined = {name: name != undefined_branch for name in BRANCHES}
    branches = primitive.pdict({
        name: (_defined_branch(name) if defined[name] else _undefined_branch(name))
        for name in BRANCHES
    })
    pairs = primitive.pdict({
        key: _pair(key, undefined_branch is None or undefined_branch not in key.split("__"))
        for key in PAIR_KEYS
    })
    contrasts = primitive.pdict({
        name: _contrast(name, coefficients,
                        undefined_branch is None or undefined_branch not in coefficients,
                        undefined_branch)
        for name, coefficients in COEFFICIENTS.items()
    })
    return primitive.pdict({
        "candidate_state": _candidate(undefined_branch), "branches": branches,
        "comparisons": primitive.pdict({"branch_pairs": pairs, "contrasts": contrasts}),
        "audit_metadata": _audit(undefined_branch),
    })


def _compute_branch_payload_bound() -> dict:
    rows = {}
    for variant in VARIANTS:
        missing = None if variant == "all_defined" else variant.removesuffix("_undefined")
        defined_count = 6 if missing is None else 5
        defined_pairs = 15 if missing is None else 10
        defined_contrasts = sum(missing is None or missing not in coefficients
                                for coefficients in COEFFICIENTS.values())
        chunks = 10 + 10 * defined_count + 10 * defined_contrasts
        rows[variant] = {
            "undefined_branch": missing,
            "defined_branch_count": defined_count,
            "undefined_branch_count": 6 - defined_count,
            "branch_pair_count": 15,
            "defined_branch_pair_count": defined_pairs,
            "undefined_branch_pair_count": 15 - defined_pairs,
            "contrast_count": 15,
            "defined_contrast_count": defined_contrasts,
            "undefined_contrast_count": 15 - defined_contrasts,
            "auxiliary_chunk_row_count": chunks,
            "tensor_count": 12 + 20 * defined_count,
            "payload_primitive_pickle_bytes_upper": _payload(missing),
        }
    maximum = max(row["payload_primitive_pickle_bytes_upper"] for row in rows.values())
    maximum_tensors = max(row["tensor_count"] for row in rows.values())
    result = {
        "schema": SCHEMA, "payload_count": PAYLOAD_COUNT,
        "source_sha256": dict(SOURCE_SHA256),
        "assumptions": {
            "profile": "scientific_mnist_current32_v1", "native_device": "cuda:0",
            "anchor_update_min": 101, "anchor_update_max": 2_000,
            "observer_rank_max": 32, "stabilization_count_max": 4_000,
            "artifact_id_utf8_bytes_max": 128, "artifact_name_utf8_bytes_max": 131,
            "artifact_payload_bytes_max": 1 << 30, "receipt_bytes_max": 4_096,
            "tensor_leaf_primitive_cost": 0, "alias_savings_assumed": False,
        },
        "variants": rows,
        "maxima": {
            "payload_primitive_pickle_bytes_upper": maximum,
            "all_payloads_primitive_pickle_bytes_upper": PAYLOAD_COUNT * maximum,
            "tensor_count_per_payload_upper": maximum_tensors,
            "tensor_count_all_payloads_upper": PAYLOAD_COUNT * maximum_tensors,
        },
        "exclusions": {
            "common_envelope_and_identity": True, "tensor_pickle_descriptors": True,
            "raw_storage_bytes": True, "zip_container_bytes": True,
            "terminal_failure_metadata": True,
        },
        "storage_fit_proven": False, "execution_authorized": False,
        "scientific_execution_certified": False,
    }
    return result


def compute_branch_payload_bound() -> dict:
    """Return the deterministic payload-only bound for the closed 18-result set."""
    result = _compute_branch_payload_bound()
    validate_branch_payload_bound(result)
    return result


def validate_branch_payload_bound(value: object) -> dict:
    """Reject any altered, ill-typed, missing or reordered bound record."""
    expected = _compute_branch_payload_bound()
    if type(value) is not dict or value != expected or tuple(value.keys()) != tuple(expected.keys()):
        raise ValueError("branch payload primitive bound differs from closed calculation")
    # Python equality conflates bool/int; exact recursive type binding closes it.
    def exact(left: object, right: object) -> bool:
        if type(left) is not type(right):
            return False
        if type(left) is dict:
            return (tuple(left.keys()) == tuple(right.keys()) and
                    all(exact(left[key], right[key]) for key in left))
        if type(left) is list:
            return len(left) == len(right) and all(exact(a, b) for a, b in zip(left, right))
        return left == right
    if not exact(value, expected):
        raise ValueError("branch payload primitive bound has wrong exact leaf types")
    return value


def validate_source_bindings(repo_root: os.PathLike[str] | str) -> dict[str, str]:
    """Authenticate the small pinned source set used by the arithmetic proof."""
    root = Path(repo_root)
    root_stat = root.lstat()
    if not stat.S_ISDIR(root_stat.st_mode) or root.is_symlink():
        raise ValueError("source root must be a real directory")
    observed = {}
    for relative, expected in SOURCE_SHA256.items():
        path = root / relative
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or path.is_symlink() or info.st_nlink != 1:
            raise ValueError("pinned source must be a singly-linked regular file")
        if info.st_size < 1 or info.st_size > (1 << 20):
            raise ValueError("pinned source size outside bounded proof domain")
        raw = path.read_bytes()
        if len(raw) != info.st_size or hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError("pinned source bytes differ")
        observed[relative] = expected
    return observed


if __name__ == "__main__":
    print("I7 branch primitive arithmetic only; no payload, data, model, CUDA or execution.")
