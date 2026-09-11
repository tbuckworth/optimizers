"""Pure strict plan-value bindings, not a plan generator or provenance audit.

See identity-binding-contract.md. Callers must supply independently verified
file receipts. No data/file loading, RNG calls, source execution or device moves.
"""
from __future__ import annotations

import re

import torch

import artifact_store as storage
import identity_codec as codec


class BindingError(ValueError):
    pass


ARRAY_KEYS = (
    "permutation", "train_indices", "validation_indices", "auxiliary_indices",
    "replacement_uniforms", "replacement_digits", "training_batches",
    "training_probe_indices",
)
PLAN_KEYS = (
    "schema", "profile", "bundle", "steps_total", "stream_roles", *ARRAY_KEYS[:6],
    "initialization_seed", *ARRAY_KEYS[6:],
)
BINDING_KEYS = ("artifact", "arrays", "initialization_seed", "next_batch_row",
                "next_batch_indices")
DIMENSIONS = {
    storage.SCIENTIFIC: (60000, 5000, 10, 64, 256),
    storage.FIXTURE: (12, 4, 2, 2, 3),
    storage.MLP_FIXTURE: (30, 10, 2, 4, 4),
}
SCHEMAS = {
    storage.SCIENTIFIC: "i7_frozen_plan_v1",
    storage.FIXTURE: "i7_frozen_plan_v1",
    storage.MLP_FIXTURE: "i7_frozen_plan_v2",
}
_SHA = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)


def _keys(value, expected, label):
    if (type(value) is not dict or tuple(value) != tuple(expected)
            or any(type(key) is not str for key in value)):
        raise BindingError(f"{label}: exact ordered keys required")


def _int(value, label, low, high):
    if type(value) is not int or not low <= value < high:
        raise BindingError(f"{label}: invalid integer")


def _tree(value):
    try:
        storage._validate_tree(value)
    except storage.StoreError as exc:
        raise BindingError(str(exc)) from exc
    except RuntimeError:
        raise BindingError("unsupported tensor backend or representation") from None


def _same_exact(left, right):
    """Typed structural comparison; never use a digest as schema validation."""
    if type(left) is not type(right):
        return False
    if type(left) is dict:
        return (len(left) == len(right)
                and all(_same_exact(lk, rk) and _same_exact(lv, rv)
                        for (lk, lv), (rk, rv) in zip(left.items(), right.items())))
    if type(left) in (list, tuple):
        return len(left) == len(right) and all(_same_exact(a, b) for a, b in zip(left, right))
    if type(left) is torch.Tensor:
        return left.dtype == right.dtype and left.shape == right.shape and torch.equal(left, right)
    return left == right


def _receipt(value):
    _keys(value, ("sha256", "size_bytes"), "artifact")
    if type(value["sha256"]) is not str or not _SHA.fullmatch(value["sha256"]):
        raise BindingError("artifact: invalid sha256")
    if type(value["size_bytes"]) is not int or value["size_bytes"] <= 0:
        raise BindingError("artifact: invalid size_bytes")
    return value


def _array(value, *, shape, dtype, label, low, high):
    if (type(value) is not torch.Tensor or tuple(value.shape) != shape or value.dtype != dtype
            or value.is_neg() or value.is_conj()):
        raise BindingError(f"{label}: wrong tensor shape/dtype")
    if not bool(((value >= low) & (value < high)).all()):
        raise BindingError(f"{label}: out-of-range values")


def validate_plan(plan, *, identity, profile):
    """Check exact values/layout; this does not regenerate the seeded plan."""
    try:
        codec.validate_identity(identity, profile=profile)
    except codec.CodecError as exc:
        raise BindingError(str(exc)) from exc
    _keys(plan, PLAN_KEYS, "plan")
    _tree(plan)
    for field, expected in (("schema", SCHEMAS[profile]), ("profile", profile),
                            ("bundle", identity["bundle"]),
                            ("steps_total", identity["steps_total"])):
        if type(plan[field]) is not type(expected) or plan[field] != expected:
            raise BindingError(f"plan: mismatched {field}")
    if not _same_exact(plan["stream_roles"], identity["stream_roles"]):
        raise BindingError("plan: mismatched stream_roles")
    total, count, classes, batch_size, probe_size = DIMENSIONS[profile]
    specifications = {
        "permutation": ((total,), torch.int64, 0, total),
        "train_indices": ((count,), torch.int64, 0, total),
        "validation_indices": ((count,), torch.int64, 0, total),
        "auxiliary_indices": ((count,), torch.int64, 0, total),
        "replacement_uniforms": ((count,), torch.float64, 0, 1),
        "replacement_digits": ((count,), torch.int64, 0, classes),
        "training_batches": ((identity["steps_total"], batch_size), torch.int64, 0, count),
        "training_probe_indices": ((probe_size,), torch.int64, 0, count),
    }
    for field, (shape, dtype, low, high) in specifications.items():
        _array(plan[field], shape=shape, dtype=dtype, label=field, low=low, high=high)
    if not torch.equal(torch.sort(plan["permutation"]).values, torch.arange(total, dtype=torch.int64)):
        raise BindingError("permutation: must contain every original training row exactly once")
    for offset, field in enumerate(("train_indices", "validation_indices", "auxiliary_indices")):
        if not torch.equal(plan[field], plan["permutation"][offset * count:(offset + 1) * count]):
            raise BindingError(f"{field}: differs from exact permutation slice")
    _int(plan["initialization_seed"], "initialization_seed", 0, 2**32)
    return plan


def _make_validated_binding(plan, identity, artifact):
    arrays = {
        field: {
            "sha256": codec.tensor_digest(plan[field]),
            "dtype": str(plan[field].dtype),
            "shape": list(plan[field].shape),
            "semantic_role": field,
        }
        for field in ARRAY_KEYS
    }
    return {
        "artifact": dict(artifact),
        "arrays": arrays,
        "initialization_seed": {
            "value": plan["initialization_seed"],
            "sha256": codec.tree_digest(plan["initialization_seed"]),
        },
        "next_batch_row": identity["anchor_update"] - 1,
        "next_batch_indices": plan["training_batches"][identity["anchor_update"] - 1].clone(),
    }


def make_plan_binding(plan, *, identity, profile, artifact):
    """Bind already validated plan values without repairing/casting any input."""
    validate_plan(plan, identity=identity, profile=profile)
    _receipt(artifact)
    return _make_validated_binding(plan, identity, artifact)


def validate_plan_binding(binding, *, plan, identity, profile, artifact):
    """Require exact binding to caller-supplied plan and verified file receipt."""
    validate_plan(plan, identity=identity, profile=profile)
    _receipt(artifact)
    _keys(binding, BINDING_KEYS, "binding")
    _tree(binding)
    _receipt(binding["artifact"])
    _keys(binding["arrays"], ARRAY_KEYS, "arrays")
    for field in ARRAY_KEYS:
        reference = binding["arrays"][field]
        _keys(reference, ("sha256", "dtype", "shape", "semantic_role"), field)
        if type(reference["shape"]) is not list or any(type(n) is not int for n in reference["shape"]):
            raise BindingError(f"{field}: invalid shape metadata")
    _keys(binding["initialization_seed"], ("value", "sha256"), "initialization_seed")
    _int(binding["initialization_seed"]["value"], "initialization_seed.value", 0, 2**32)
    _int(binding["next_batch_row"], "next_batch_row", 0, identity["steps_total"])
    _, count, _, batch_size, _ = DIMENSIONS[profile]
    _array(binding["next_batch_indices"], shape=(batch_size,), dtype=torch.int64,
           label="next_batch_indices", low=0, high=count)
    expected = _make_validated_binding(plan, identity, artifact)
    if not _same_exact(binding, expected):
        raise BindingError("binding differs from exact plan values, identity or artifact receipt")
    return binding
