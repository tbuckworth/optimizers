#!/usr/bin/env python3
"""Strict identity validation and canonical hashes for iteration 007.

Public API: ``CodecError``, ``validate_identity``, ``artifact_id``,
``validate_created_utc``, ``tensor_digest``, ``tree_digest``, ``json_bytes``
and ``json_loads``. The codec performs no filesystem, data, RNG, model, or
optimizer operation. Typed-tree hashes are content identities, not file hashes
or execution proofs.
"""
from __future__ import annotations

from datetime import datetime
import hashlib
import json
import math
import re
from typing import Any

import numpy as np
import torch

import artifact_store as _storage


SCIENTIFIC_PROFILE = "scientific_mnist_current32_v1"
FIXTURE_PROFILES = ("fixture_tiny_cpu_v1", "fixture_tiny_mlp_cpu_v1")
PROFILES = (SCIENTIFIC_PROFILE, *FIXTURE_PROFILES)
IDENTITY_KEYS = ("run_id", "iteration", "execution_role", "evidence_role",
                 "bundle", "anchor_update", "source_policy", "condition",
                 "steps_total", "anchor_phase", "anchor_completed_updates",
                 "anchor_completed_observations", "rng_namespace_prefix", "stream_roles")
STREAM_ROLES = {0: "permutation", 1: "replacement_uniforms", 2: "replacement_digits",
                3: "initialization_seed", 4: "training_batches", 5: "training_probe",
                6: "unused"}
ARTIFACT_KINDS = ("anchor", "source-witness", "branch-results", "independent-audit")
_RUN_ID = "2026-09-06-spectral-optimizer-investigation"
_PHASE = "pre_forward_pre_observe"
_UTC = re.compile(r"([0-9]{4})-([0-9]{2})-([0-9]{2})T([0-9]{2}):([0-9]{2}):([0-9]{2})Z\Z")
_TREE_DOMAIN = b"i7_typed_tree_sha256_v1\n"
_DTYPES = {
    torch.uint8: "<u1",
    torch.uint32: "<u4",
    torch.int64: "<i8",
    torch.float32: "<f4",
    torch.float64: "<f8",
    torch.bool: "?",
}


class CodecError(ValueError):
    """The supplied identity, tensor, tree, or JSON value is malformed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CodecError(message)


def _exact_keys(value: Any, keys: tuple[Any, ...], where: str) -> None:
    _require(type(value) is dict and len(value) == len(keys), f"{where} keys/order mismatch")
    actual = tuple(value.keys())
    _require(all(type(got) is type(expected) and got == expected
                 for got, expected in zip(actual, keys)), f"{where} keys/order mismatch")


def _exact_int(value: Any, expected: int | None, where: str) -> int:
    _require(type(value) is int and (expected is None or value == expected), f"{where} invalid")
    return value


def _validate_string(value: str, where: str) -> str:
    _require(not any(0xD800 <= ord(character) <= 0xDFFF for character in value),
             f"{where} contains a surrogate code point")
    return value


def validate_identity(identity: Any, *, profile: str) -> dict[str, Any]:
    """Validate and return one exact scientific or registered-fixture identity."""
    _require(type(profile) is str and profile in PROFILES, "unknown identity profile")
    _exact_keys(identity, IDENTITY_KEYS, "identity")
    _require(type(identity["run_id"]) is str and identity["run_id"] == _RUN_ID, "run_id invalid")
    _exact_int(identity["iteration"], 7, "iteration")
    for key in ("execution_role", "evidence_role", "source_policy", "condition", "anchor_phase"):
        _require(type(identity[key]) is str, f"{key} must be a string")
    _exact_int(identity["bundle"], None, "bundle")
    update = _exact_int(identity["anchor_update"], None, "anchor_update")
    steps = _exact_int(identity["steps_total"], None, "steps_total")
    _exact_int(identity["anchor_completed_updates"], update - 1, "anchor_completed_updates")
    _exact_int(identity["anchor_completed_observations"], update - 1,
               "anchor_completed_observations")
    _require(identity["anchor_phase"] == _PHASE, "anchor_phase invalid")

    if profile in FIXTURE_PROFILES:
        expected = ("primary", "primary", 0, 5, "fixture_current2", "fixture_only", 8)
        actual = (identity["execution_role"], identity["evidence_role"], identity["bundle"],
                  update, identity["source_policy"], identity["condition"], steps)
        _require(actual == expected, "fixture identity membership invalid")
    else:
        _require(identity["source_policy"] == "current32" and
                 identity["condition"] == "noise_0.9", "scientific policy/condition invalid")
        role = identity["execution_role"]
        if role == "primary":
            _require(identity["evidence_role"] == "primary" and
                     identity["bundle"] in (71001, 71002, 71003) and
                     update in (101, 500, 1000, 2000) and steps == 2000,
                     "primary identity membership invalid")
        elif role == "sensitivity":
            _require(identity["evidence_role"] == "sensitivity" and
                     identity["bundle"] == 71901 and update in (101, 500, 1000, 2000) and
                     steps == 2000, "sensitivity identity membership invalid")
        elif role == "pilot":
            _require(identity["evidence_role"] == "development" and
                     identity["bundle"] == 71990 and update in (101, 200) and steps == 220,
                     "pilot identity membership invalid")
        else:
            raise CodecError("execution_role invalid")

    prefix = identity["rng_namespace_prefix"]
    _require(type(prefix) is list and len(prefix) == 2 and
             type(prefix[0]) is int and type(prefix[1]) is int and
             prefix == [20260906, identity["bundle"]], "rng_namespace_prefix invalid")
    _exact_keys(identity["stream_roles"], tuple(STREAM_ROLES), "stream_roles")
    _require(all(type(key) is int and type(value) is str and STREAM_ROLES[key] == value
                 for key, value in identity["stream_roles"].items()), "stream_roles invalid")
    return identity


def artifact_id(identity: Any, *, profile: str, kind: str) -> str:
    """Return the canonical artifact ID after validating all inputs."""
    validated = validate_identity(identity, profile=profile)
    _require(type(kind) is str and kind in ARTIFACT_KINDS, "artifact kind invalid")
    return (f"{profile}--{validated['execution_role']}--b{validated['bundle']}--"
            f"u{validated['anchor_update']}--{kind}")


def validate_created_utc(value: Any) -> str:
    """Require the exact second-resolution UTC provenance representation."""
    _require(type(value) is str, "created_utc must be a string")
    match = _UTC.fullmatch(value)
    _require(match is not None, "created_utc format invalid")
    parts = tuple(int(item) for item in match.groups())
    try:
        datetime(*parts)
    except ValueError as exc:
        raise CodecError("created_utc Gregorian date/time invalid") from exc
    return value


def _validate_tensor(value: Any) -> torch.Tensor:
    _require(type(value) is torch.Tensor, "expected an exact torch.Tensor")
    _require(value.device.type == "cpu" and value.layout == torch.strided and
             not value.requires_grad and not value.is_nested,
             "tensor must be detached dense non-nested CPU storage")
    _require(value.dtype in _DTYPES, "unsupported tensor dtype")
    _require(not value.is_neg() and not value.is_conj(),
             "lazy negative/conjugate tensor views are unsupported")
    _require(value.is_contiguous() and value.storage_offset() == 0 and value._base is None and
             value.untyped_storage().nbytes() == value.numel() * value.element_size(),
             "tensor must own compact contiguous storage")
    if value.is_floating_point():
        _require(bool(torch.isfinite(value).all()), "nonfinite tensor")
    return value


def _raw_tensor_bytes(value: torch.Tensor) -> bytes:
    array = value.detach().numpy()
    return array.astype(np.dtype(_DTYPES[value.dtype]), copy=False).tobytes(order="C")


def tensor_digest(value: Any) -> str:
    """Hash canonical row-major little-endian bytes without shape/dtype prefixes."""
    tensor = _validate_tensor(value)
    return hashlib.sha256(_raw_tensor_bytes(tensor)).hexdigest()


def _typed_node(value: Any) -> list[Any]:
    if value is None:
        return ["none"]
    if type(value) is bool:
        return ["bool", value]
    if type(value) is int:
        return ["int", str(value)]
    if type(value) is float:
        _require(math.isfinite(value), "nonfinite typed-tree float")
        return ["float", value.hex()]
    if type(value) is str:
        return ["str", _validate_string(value, "typed-tree string")]
    if type(value) is list:
        return ["list", [_typed_node(item) for item in value]]
    if type(value) is tuple:
        return ["tuple", [_typed_node(item) for item in value]]
    if type(value) is dict:
        return ["dict", [[_typed_node(key), _typed_node(item)] for key, item in value.items()]]
    if type(value) is torch.Tensor:
        tensor = _validate_tensor(value)
        return ["tensor", str(tensor.dtype), list(tensor.shape), tensor_digest(tensor)]
    raise CodecError("unsupported typed-tree value")


def tree_digest(value: Any) -> str:
    """Hash the canonical insertion-ordered typed-tree representation."""
    try:
        _storage._validate_tree(value)
        node = _typed_node(value)
        encoded = json.dumps(node, ensure_ascii=True, allow_nan=False,
                             separators=(",", ":"), sort_keys=False).encode("ascii")
    except CodecError:
        raise
    except _storage.StoreError as exc:
        raise CodecError(f"invalid tensor/primitive tree: {exc}") from exc
    except RuntimeError:
        # Unsupported backend representations can emit huge dispatcher messages.
        raise CodecError("unsupported tensor backend or representation") from None
    except (RecursionError, TypeError, ValueError, OverflowError) as exc:
        raise CodecError("typed-tree encoding failed") from exc
    return hashlib.sha256(_TREE_DOMAIN + encoded).hexdigest()


def _validate_json(value: Any, active: set[int] | None = None) -> None:
    active = set() if active is None else active
    if value is None or type(value) in (bool, int):
        return
    if type(value) is str:
        _validate_string(value, "JSON string")
        return
    if type(value) is float:
        _require(math.isfinite(value), "nonfinite JSON float")
        return
    _require(type(value) in (list, dict), "unsupported JSON value")
    identity = id(value)
    _require(identity not in active, "cyclic JSON container")
    active.add(identity)
    if type(value) is list:
        for item in value:
            _validate_json(item, active)
    else:
        for key, item in value.items():
            _require(type(key) is str, "JSON object keys must be exact strings")
            _validate_string(key, "JSON object key")
            _validate_json(item, active)
    active.remove(identity)


def json_bytes(value: Any) -> bytes:
    """Encode strict JSON primitives to compact ASCII plus one newline."""
    try:
        _validate_json(value)
        text = json.dumps(value, ensure_ascii=True, allow_nan=False,
                          separators=(",", ":"), sort_keys=False)
    except CodecError:
        raise
    except (TypeError, ValueError, RecursionError) as exc:  # defensive after validation
        raise CodecError("JSON encoding failed") from exc
    return (text + "\n").encode("ascii")


def json_loads(data: Any, *, max_bytes: int) -> Any:
    """Decode one bounded ASCII JSON document with duplicate/finite checks."""
    _require(type(max_bytes) is int and max_bytes > 0, "max_bytes must be a positive exact int")
    _require(type(data) is bytes and len(data) <= max_bytes, "JSON input must be bounded bytes")
    try:
        text = data.decode("ascii")
    except UnicodeDecodeError as exc:
        raise CodecError("JSON input must be ASCII") from exc

    def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise CodecError("duplicate JSON object key")
            result[key] = value
        return result

    def finite_float(text_value: str) -> float:
        value = float(text_value)
        if not math.isfinite(value):
            raise CodecError("decoded nonfinite or overflowing JSON float")
        return value

    def reject_constant(_: str) -> None:
        raise CodecError("decoded non-standard JSON constant")

    try:
        value = json.loads(text, object_pairs_hook=pairs_hook, parse_float=finite_float,
                           parse_constant=reject_constant)
        _validate_json(value)
    except CodecError:
        raise
    except (json.JSONDecodeError, TypeError, ValueError, OverflowError, RecursionError) as exc:
        raise CodecError("malformed JSON input") from exc
    return value
