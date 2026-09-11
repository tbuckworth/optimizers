#!/usr/bin/env python3
"""Assemble the inert iteration-007 loss and displacement measurements.

Public API::

    assemble(before_parameters, endpoints, candidates, probes, *, artifact_id,
             profile, native_device="cpu", guard=None)

Inputs are plain, exact-order mappings. ``before_parameters`` and every defined
endpoint contain the four native-float32 MLP parameters. ``endpoints`` has the
six branch keys and uses ``None`` only for a validated response-domain null.
``candidates`` is the complete result of ``response_math.construct``. ``probes``
has the four fixed keys, each mapping to owning CPU float32 ``inputs`` and CPU
int64 ``labels``. The optional guard is called as ``guard(label: str)``; labels
include before/after markers, and it is forwarded to every loss evaluation.

The exact result root is ``{measurement_before, branches, comparisons}``.
Branch values are either ``{displacement, probes}`` or ``None``. Comparison
values implement the layouts in measurement-schema.md. This module owns no
optimizer/artifact envelope, loads no data, and imports no independent auditor.
The tiny profile is an engineering fixture and cannot certify scientific data.
"""
from __future__ import annotations

import hashlib
import importlib.util
import itertools
import math
from pathlib import Path
import re
import struct
from typing import Any, Callable

import numpy as np
import torch


_HERE = Path(__file__).resolve().parent
_SPEC = importlib.util.spec_from_file_location("i7_measurement_loss", _HERE / "loss_measurements.py")
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover - import machinery failure
    raise ImportError("cannot load loss_measurements.py")
_LOSS = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_LOSS)
_RESPONSE_SPEC = importlib.util.spec_from_file_location("i7_measurement_response",
                                                        _HERE / "response_math.py")
if _RESPONSE_SPEC is None or _RESPONSE_SPEC.loader is None:  # pragma: no cover
    raise ImportError("cannot load response_math.py")
_RESPONSE = importlib.util.module_from_spec(_RESPONSE_SPEC)
_RESPONSE_SPEC.loader.exec_module(_RESPONSE)

BRANCHES = ("raw", "current", "lagged", "restored", "reciprocal", "zero")
PROBES = ("batch_noisy", "train_probe_noisy", "train_probe_clean", "auxiliary_clean")
PARAMETERS = ("0.weight", "0.bias", "2.weight", "2.bias")
SCIENTIFIC_PROFILE = "scientific_mnist_current32_v1"
FIXTURE_PROFILE = "fixture_tiny_mlp_cpu_v1"
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
PAIR_KEYS = tuple(f"{a}__{b}" for a, b in itertools.combinations(BRANCHES, 2))
LEVERAGE_KEYS = ("current_norm", "lagged_norm", "signed_norm_difference",
                 "relative_norm_separation", "current_lagged_norm_ratio", "ratio_reason",
                 "unit_direction_distance", "unit_direction_cosine", "direction_reason",
                 "norm_leverage", "direction_leverage")
PROFILE = {
    SCIENTIFIC_PROFILE: {
        "shapes": ((64, 784), (64,), (10, 64), (10,)),
        "counts": (64, 256, 256, 5000), "auxiliary_chunk": 500,
        "shape_mode": "scientific_784_64_10", "device_kind": "cuda",
    },
    FIXTURE_PROFILE: {
        "shapes": ((4, 3), (4,), (2, 4), (2,)),
        "counts": (4, 4, 4, 10), "auxiliary_chunk": 1,
        "shape_mode": "generic_tiny_mlp", "device_kind": "cpu",
    },
}
_U64 = 2.0 ** -53
_MIN64 = 2.0 ** -1022
_DECAY = 0.001 * 0.01
_ARTIFACT_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


class AssemblyFailure(RuntimeError):
    """Fatal producer validation with the bounded failed record retained."""

    def __init__(self, location: str, record: dict[str, Any]):
        self.location = location
        self.record = _clone_plain(record)
        super().__init__(f"fatal assembly validation at {location}")


def _fail(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _keys(value: Any, expected: tuple[str, ...], where: str) -> None:
    _fail(type(value) is dict and tuple(value.keys()) == expected, f"{where} keys/order mismatch")


def _finite(value: Any, where: str) -> float:
    _fail(type(value) is float and math.isfinite(value), f"{where} must be a finite Python float")
    return value


def _owned(value: torch.Tensor) -> bool:
    return (value.is_contiguous() and value.storage_offset() == 0 and value._base is None and
            value.untyped_storage().nbytes() == value.numel() * value.element_size())


def _guard(guard: Callable[[str], None] | None, label: str) -> None:
    if guard is not None:
        guard(label)


def _operation(guard: Callable[[str], None] | None, label: str, function: Callable[[], Any]) -> Any:
    _guard(guard, f"before:{label}")
    result = function()
    _guard(guard, f"after:{label}")
    return result


def _clone_plain(value: Any) -> Any:
    if type(value) is dict:
        return {key: _clone_plain(item) for key, item in value.items()}
    if type(value) is list:
        return [_clone_plain(item) for item in value]
    if type(value) is torch.Tensor:
        return value.detach().clone()
    return value


def _tensor_bytes_equal(left: torch.Tensor, right: torch.Tensor) -> bool:
    if (type(left) is not torch.Tensor or type(right) is not torch.Tensor or
            left.shape != right.shape or left.dtype != right.dtype or left.device != right.device):
        return False
    a = left.detach().to("cpu").contiguous().view(torch.uint8)
    b = right.detach().to("cpu").contiguous().view(torch.uint8)
    return bool(torch.equal(a, b))


def _tree_exact(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    if type(left) is torch.Tensor:
        return _tensor_bytes_equal(left, right)
    if type(left) is dict:
        return (tuple(left.keys()) == tuple(right.keys()) and
                all(_tree_exact(left[key], right[key]) for key in left))
    if type(left) in (list, tuple):
        return len(left) == len(right) and all(_tree_exact(a, b) for a, b in zip(left, right))
    if type(left) is float:
        return struct.pack("<d", left) == struct.pack("<d", right)
    return left == right


def _require_pass(location: str, record: dict[str, Any], status_key: str) -> None:
    if record[status_key] != "pass":
        raise AssemblyFailure(location, record)


def _gamma(operations: int) -> float:
    product = operations * _U64
    _fail(product < 1.0, "roundoff operation count invalid")
    return product / (1.0 - product)


def _rsum(values: list[float]) -> float:
    count = len(values)
    _fail(count > 0 and all(math.isfinite(x) for x in values), "invalid rounding-bound terms")
    return float(2.0 * _gamma(2 * count + 4) * math.fsum(abs(x) for x in values) +
                 4.0 * count * _MIN64)


def _linear(values: dict[str, float], coefficients: dict[str, int]) -> float:
    return _finite(float(math.fsum(coefficients[key] * values[key] for key in coefficients)),
                   "linear combination")


def _tensor64(value: torch.Tensor, where: str) -> torch.Tensor:
    _fail(type(value) is torch.Tensor and value.layout == torch.strided and
          value.dtype in (torch.float32, torch.float64) and value.ndim == 1 and
          value.numel() > 0 and not value.requires_grad and bool(torch.isfinite(value).all()),
          f"{where} must be a finite detached flat float tensor")
    return value.detach().to(device="cpu", dtype=torch.float64).contiguous().clone()


def _norm(value: torch.Tensor) -> float:
    return _finite(float(torch.linalg.vector_norm(value).item()), "vector norm")


def _squared_norm(value: torch.Tensor) -> float:
    result = _finite(float(torch.dot(value, value).item()), "squared norm")
    _fail(result >= 0.0, "negative squared norm")
    return result


def _hash_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _tensor_bytes(value: torch.Tensor, dtype: str) -> bytes:
    array = value.detach().to("cpu").contiguous().numpy()
    return array.astype(dtype, copy=False).tobytes(order="C")


def _sample_hash(inputs: torch.Tensor, labels: torch.Tensor) -> str:
    return _hash_bytes(_tensor_bytes(inputs, "<f4") + _tensor_bytes(labels, "<i8"))


def _vector_record(value: torch.Tensor) -> dict[str, Any]:
    owned = _tensor64(value, "vector record")
    return {"value": owned, "shape": [owned.numel()], "dtype": "float64", "device": "cpu",
            "sha256": _hash_bytes(_tensor_bytes(owned, "<f8"))}


def _scalar_hash(value: float) -> str:
    return _hash_bytes(struct.pack("<d", _finite(value, "bound scalar")))


def _vector_summary(value: torch.Tensor) -> dict[str, Any]:
    owned = _tensor64(value, "vector summary")
    return {"representation": "derived_from_bound_branch_vectors",
            "canonical_sha256": _hash_bytes(_tensor_bytes(owned, "<f8")),
            "norm": _norm(owned), "squared_norm": _squared_norm(owned),
            "component_count": owned.numel()}


def _concordance(native: float, cpu: float, ceiling: float | None = None) -> dict[str, Any]:
    native, cpu = _finite(native, "native CE"), _finite(cpu, "CPU64 CE")
    if ceiling is None:
        ceiling = 5e-6 * max(1.0, abs(cpu))
    ceiling = _finite(float(ceiling), "concordance ceiling")
    _fail(ceiling >= 0.0, "negative concordance ceiling")
    discrepancy = _finite(abs(native - cpu), "concordance discrepancy")
    return {"abs_discrepancy": discrepancy, "descriptive_ceiling": ceiling,
            "status": "within_scale" if discrepancy <= ceiling else "discordant"}


def _geometry(left: torch.Tensor | None, right: torch.Tensor | None,
              mask: dict[str, bool]) -> dict[str, Any]:
    if left is None or right is None:
        return {"defined": False, "defined_mask": dict(mask),
                "reason": "domain_undefined_required_branch", "distance": None,
                "cosine": None, "cosine_reason": None, "left_norm": None, "right_norm": None}
    a, b = _tensor64(left, "geometry left"), _tensor64(right, "geometry right")
    _fail(a.shape == b.shape, "geometry shape mismatch")
    left_norm, right_norm = _norm(a), _norm(b)
    distance = _norm(a - b)
    if left_norm == 0.0 or right_norm == 0.0:
        cosine, cosine_reason = None, "zero_norm"
    else:
        cosine = _finite(float(torch.dot(a / left_norm, b / right_norm).item()), "cosine")
        cosine_reason = None
    return {"defined": True, "defined_mask": dict(mask), "reason": None,
            "distance": distance, "cosine": cosine, "cosine_reason": cosine_reason,
            "left_norm": left_norm, "right_norm": right_norm}


def _component_rsum(vectors: list[torch.Tensor]) -> torch.Tensor:
    _fail(len(vectors) > 0, "empty vector sum")
    count = len(vectors)
    total = torch.zeros_like(vectors[0])
    for value in vectors:
        _fail(value.shape == total.shape and value.dtype == torch.float64 and value.device.type == "cpu",
              "roundoff vector mismatch")
        total = total + value.abs()
    return total * (2.0 * _gamma(2 * count + 4)) + (4.0 * count * _MIN64)


def _data_construction_error(delta: torch.Tensor, decay: torch.Tensor) -> torch.Tensor:
    """Edata from i7_assembly_roundoff_v1, including multiply/add screening."""
    _fail(delta.shape == decay.shape and delta.dtype == decay.dtype == torch.float64,
          "data construction vectors mismatch")
    return (delta.abs() + decay.abs()) * (2.0 * _gamma(4)) + (8.0 * _MIN64)


def _norm_roundoff(value: torch.Tensor) -> float:
    """Nround from i7_assembly_roundoff_v1."""
    value_norm = _norm(value)
    if value_norm == 0.0:
        return 0.0
    return _finite(math.fsum([_dot_roundoff(value, value) / (2.0 * value_norm),
                              8.0 * _U64 * value_norm]), "norm rounding ceiling")


def _linear_vector(values: dict[str, torch.Tensor], coefficients: dict[str, int]) -> tuple[torch.Tensor, torch.Tensor]:
    terms = [values[key] * coefficients[key] for key in coefficients]
    result = torch.zeros_like(terms[0])
    for term in terms:
        result = result + term
    _fail(bool(torch.isfinite(result).all()), "nonfinite vector contrast")
    return result.contiguous().clone(), _component_rsum(terms)


def _dot_roundoff(left: torch.Tensor, right: torch.Tensor) -> float:
    _fail(left.shape == right.shape and left.dtype == right.dtype == torch.float64,
          "dot roundoff vector mismatch")
    count = left.numel()
    absolute_sum = float(torch.sum(torch.abs(left * right)).item())
    return _finite(float(2.0 * _gamma(2 * count + 4) * absolute_sum +
                         4.0 * count * _MIN64), "dot rounding ceiling")


def _validate_parameter_map(value: Any, spec: dict[str, Any], device: torch.device,
                            where: str) -> None:
    _keys(value, PARAMETERS, where)
    seen: set[tuple[int, int]] = set()
    for name, shape in zip(PARAMETERS, spec["shapes"]):
        tensor = value[name]
        _fail(type(tensor) is torch.Tensor and tensor.layout == torch.strided and
              tensor.dtype == torch.float32 and tuple(tensor.shape) == shape and
              tensor.device == device and not tensor.requires_grad and _owned(tensor) and
              bool(torch.isfinite(tensor).all()), f"{where}.{name} invalid")
        identity = (tensor.untyped_storage().data_ptr(), tensor.untyped_storage().nbytes())
        _fail(identity not in seen, f"{where} parameter alias")
        seen.add(identity)


def _validate_probe(value: Any, count: int, input_width: int, classes: int, where: str) -> None:
    _keys(value, ("inputs", "labels"), where)
    inputs, labels = value["inputs"], value["labels"]
    _fail(type(inputs) is torch.Tensor and inputs.device.type == "cpu" and
          inputs.dtype == torch.float32 and tuple(inputs.shape) == (count, input_width) and
          not inputs.requires_grad and _owned(inputs) and bool(torch.isfinite(inputs).all()),
          f"{where}.inputs invalid")
    _fail(type(labels) is torch.Tensor and labels.device.type == "cpu" and
          labels.dtype == torch.int64 and tuple(labels.shape) == (count,) and
          not labels.requires_grad and _owned(labels), f"{where}.labels invalid")
    _fail(bool(((labels >= 0) & (labels < classes)).all()), f"{where}.labels outside class range")


def _validate_candidate(candidate: Any, name: str, components: int,
                        device: torch.device) -> tuple[bool, torch.Tensor | None]:
    _keys(candidate, ("status", "reason", "gradient", "target_norm", "actual_norm",
                      "norm_error", "direction_error"), f"candidate {name}")
    _fail(candidate["status"] in ("defined", "undefined"), f"candidate {name} status invalid")
    if candidate["status"] == "undefined":
        permitted = {
            "restored": "positive_current_norm_zero_lagged_direction",
            "reciprocal": "positive_lagged_norm_zero_current_direction",
        }
        _fail(name in permitted and candidate["reason"] == permitted[name] and
              candidate["gradient"] is None and candidate["actual_norm"] is None and
              candidate["norm_error"] is None and candidate["direction_error"] is None,
              f"candidate {name} malformed domain null")
        _fail(type(candidate["target_norm"]) is float and math.isfinite(candidate["target_norm"]) and
              candidate["target_norm"] > 0.0, f"candidate {name} null target invalid")
        return False, None
    _fail(candidate["reason"] is None, f"candidate {name} defined reason must be null")
    gradient = candidate["gradient"]
    _fail(type(gradient) is torch.Tensor and gradient.layout == torch.strided and
          gradient.dtype == torch.float32 and gradient.device == device and
          gradient.ndim == 1 and gradient.numel() == components and not gradient.requires_grad and
          _owned(gradient) and bool(torch.isfinite(gradient).all()),
          f"candidate {name} gradient invalid")
    for key in ("target_norm", "actual_norm", "norm_error"):
        _fail(type(candidate[key]) is float and math.isfinite(candidate[key]) and candidate[key] >= 0.0,
              f"candidate {name}.{key} invalid")
    direction_error = candidate["direction_error"]
    _fail(direction_error is None or
          (type(direction_error) is float and math.isfinite(direction_error) and direction_error >= 0.0),
          f"candidate {name}.direction_error invalid")
    return True, gradient


def _evaluate(parameters: dict[str, torch.Tensor], probe: dict[str, torch.Tensor], *,
              chunk_size: int, precision: str, with_gradient: bool,
              device: torch.device, guard: Callable[[str], None] | None,
              label: str) -> dict[str, Any]:
    inputs = probe["inputs"]
    labels = probe["labels"]
    if precision == "native32":
        inputs = _operation(guard, f"{label}:native_input_copy",
                            lambda: inputs.detach().to(device=device).contiguous().clone())
        labels = _operation(guard, f"{label}:native_label_copy",
                            lambda: labels.detach().to(device=device).contiguous().clone())
    result = _operation(
        guard, f"{label}:{precision}:evaluate",
        lambda: _LOSS.evaluate(parameters, inputs, labels, chunk_size=chunk_size,
                               with_gradient=with_gradient, precision=precision, guard=guard))
    return result


def _probe_evaluations(parameters: dict[str, torch.Tensor], probe: dict[str, torch.Tensor], *,
                       chunk_size: int, need_gradient: bool, device: torch.device,
                       guard: Callable[[str], None] | None, label: str) -> tuple[dict[str, Any], dict[str, Any]]:
    cpu = _evaluate(parameters, probe, chunk_size=chunk_size, precision="cpu64",
                    with_gradient=need_gradient, device=device, guard=guard, label=label)
    native = _evaluate(parameters, probe, chunk_size=chunk_size, precision="native32",
                       with_gradient=False, device=device, guard=guard, label=label)
    _fail(cpu["count"] == native["count"] and len(cpu["chunks"]) == len(native["chunks"]),
          "precision measurement membership mismatch")
    _fail(all((a["start"], a["end"], a["count"]) == (b["start"], b["end"], b["count"])
              for a, b in zip(cpu["chunks"], native["chunks"])), "precision chunk mismatch")
    return cpu, native


def _before_probe(probe_key: str, probe: dict[str, torch.Tensor],
                  parameters: dict[str, torch.Tensor], spec: dict[str, Any],
                  device: torch.device, guard: Callable[[str], None] | None) -> tuple[dict[str, Any], dict[str, Any]]:
    auxiliary = probe_key == "auxiliary_clean"
    chunk_size = spec["auxiliary_chunk"] if auxiliary else probe["inputs"].shape[0]
    cpu, native = _probe_evaluations(parameters, probe, chunk_size=chunk_size,
                                     need_gradient=True, device=device, guard=guard,
                                     label=f"before:{probe_key}")
    _fail(cpu["shape_mode"] == native["shape_mode"] == spec["shape_mode"],
          "loss shape mode/profile mismatch")
    before_ce = _finite(cpu["mean_cross_entropy"], "before CPU64 CE")
    native_ce = _finite(native["mean_cross_entropy"], "before native CE")
    q = _tensor64(cpu["mean_gradient"], "before gradient")
    chunks = []
    for index, (cr, nr) in enumerate(zip(cpu["chunks"], native["chunks"])):
        start, end, count = cr["start"], cr["end"], cr["count"]
        xs, ys = probe["inputs"][start:end], probe["labels"][start:end]
        cpu_mean, native_mean = cr["mean_cross_entropy"], nr["mean_cross_entropy"]
        chunks.append({"chunk_index": index, "start": start, "end": end, "count": count,
                       "input_sha256": _hash_bytes(_tensor_bytes(xs, "<f4")),
                       "label_sha256": _hash_bytes(_tensor_bytes(ys, "<i8")),
                       "cpu64_before_ce": _finite(cpu_mean, "chunk CPU64 before CE"),
                       "native32_before_ce": _finite(native_mean, "chunk native before CE"),
                       "concordance": _concordance(native_mean, cpu_mean)})
    if not auxiliary:
        _fail(len(chunks) == 1, "non-auxiliary probe must use one loss chunk")
        chunks = []
    else:
        _fail(len(chunks) == 10, "auxiliary probe must have ten chunks")
    record = {"sample_count": cpu["count"],
              "sample_identity_sha256": _sample_hash(probe["inputs"], probe["labels"]),
              "cpu64": {"before_ce": before_ce, "q": _vector_record(q), "q_norm": _norm(q)},
              "native32": {"before_ce": native_ce},
              "concordance": _concordance(native_ce, before_ce),
              "auxiliary_chunks": chunks}
    internal = {"cpu": cpu, "native": native, "q": q}
    return record, internal


def _displacement(before_flat: torch.Tensor, after: dict[str, torch.Tensor],
                  branch: str) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    after_flat = torch.cat([after[key].detach().to("cpu", torch.float64).reshape(-1)
                            for key in PARAMETERS]).contiguous()
    delta = (after_flat - before_flat).contiguous()
    decay = (before_flat * _DECAY).contiguous()
    delta_data = (delta + decay).contiguous()
    _fail(bool(torch.isfinite(delta).all()) and bool(torch.isfinite(delta_data).all()),
          f"branch {branch} displacement nonfinite")
    geometry = _geometry(delta, delta_data, {branch: True})
    record = {"delta": _vector_record(delta), "delta_data": _vector_record(delta_data),
              "delta_norm": _norm(delta), "delta_squared_norm": _squared_norm(delta),
              "delta_data_norm": _norm(delta_data),
              "delta_data_squared_norm": _squared_norm(delta_data),
              "delta_delta_data": geometry}
    internal = {"delta": delta.clone(), "delta_data": delta_data.clone(), "decay": decay.clone(),
                "delta_data_error": _data_construction_error(delta, decay)}
    return record, internal


def _branch_probe(probe_key: str, probe: dict[str, torch.Tensor], endpoint: dict[str, torch.Tensor],
                  before: dict[str, Any], before_internal: dict[str, Any], displacement: dict[str, torch.Tensor],
                  spec: dict[str, Any], device: torch.device, artifact_id: str,
                  guard: Callable[[str], None] | None, branch: str) -> dict[str, Any]:
    auxiliary = probe_key == "auxiliary_clean"
    chunk_size = spec["auxiliary_chunk"] if auxiliary else probe["inputs"].shape[0]
    cpu, native = _probe_evaluations(endpoint, probe, chunk_size=chunk_size,
                                     need_gradient=False, device=device, guard=guard,
                                     label=f"branch:{branch}:{probe_key}")
    _fail(cpu["shape_mode"] == native["shape_mode"] == spec["shape_mode"],
          "endpoint loss shape mode/profile mismatch")
    after, after_native = cpu["mean_cross_entropy"], native["mean_cross_entropy"]
    before_ce = before["cpu64"]["before_ce"]
    before_native = before["native32"]["before_ce"]
    y = _finite(after - before_ce, "branch Y")
    y_native = _finite(after_native - before_native, "native branch Y")
    q = before_internal["q"]
    d = _finite(float(torch.dot(q, displacement["delta"]).item()), "branch D")
    ddata = _finite(float(torch.dot(q, displacement["delta_data"]).item()), "branch Ddata")
    residual = _finite(y - d, "branch residual")
    y_ceiling = math.fsum([
        before["concordance"]["descriptive_ceiling"],
        _concordance(after_native, after)["descriptive_ceiling"],
        max(_rsum([after_native, -before_native]), _rsum([after, -before_ce])),
    ])
    chunk_rows = []
    for index, (cr, nr, br) in enumerate(zip(cpu["chunks"], native["chunks"],
                                             before["auxiliary_chunks"] if auxiliary else [])):
        cpu_after, native_after = cr["mean_cross_entropy"], nr["mean_cross_entropy"]
        cpu_y = _finite(cpu_after - br["cpu64_before_ce"], "chunk Y")
        native_y = _finite(native_after - br["native32_before_ce"], "native chunk Y")
        chunk_y_ceiling = math.fsum([
            br["concordance"]["descriptive_ceiling"],
            _concordance(native_after, cpu_after)["descriptive_ceiling"],
            max(_rsum([native_after, -br["native32_before_ce"]]),
                _rsum([cpu_after, -br["cpu64_before_ce"]])),
        ])
        chunk_rows.append({"chunk_index": index, "start": cr["start"], "end": cr["end"],
                           "count": cr["count"], "input_sha256": br["input_sha256"],
                           "label_sha256": br["label_sha256"],
                           "cpu64_after_ce": _finite(cpu_after, "chunk after CE"), "cpu64_Y": cpu_y,
                           "native32_after_ce": _finite(native_after, "native chunk after CE"),
                           "native32_Y": native_y,
                           "concordance": {"after": _concordance(native_after, cpu_after),
                                           "Y": _concordance(native_y, cpu_y, chunk_y_ceiling)}})
    _fail((auxiliary and len(chunk_rows) == 10) or (not auxiliary and not chunk_rows),
          "branch auxiliary chunk membership mismatch")
    return {"before_binding": {"measurement_before_artifact_id": artifact_id,
                               "probe_key": probe_key,
                               "before_ce_cpu64_sha256": _scalar_hash(before_ce),
                               "q_sha256": before["cpu64"]["q"]["sha256"]},
            "cpu64": {"after_ce": _finite(after, "after CE"), "Y": y, "D": d,
                      "Ddata": ddata, "R": residual},
            "native32": {"after_ce": _finite(after_native, "native after CE"), "Y": y_native},
            "concordance": {"after": _concordance(after_native, after),
                            "Y": _concordance(y_native, y, y_ceiling)},
            "auxiliary_chunks": chunk_rows}


def _scalar_record(component_values: dict[str, float], direct: float, component: float,
                   ceiling: float, location: str) -> dict[str, Any]:
    direct, component = _finite(direct, "direct scalar"), _finite(component, "component scalar")
    ceiling = _finite(float(ceiling), "identity ceiling")
    _fail(ceiling >= 0.0, "negative identity ceiling")
    discrepancy = _finite(abs(direct - component), "identity discrepancy")
    record = {"component_values": dict(component_values), "direct_value": direct,
              "component_sum_value": component, "identity_abs_discrepancy": discrepancy,
              "identity_rounding_ceiling": ceiling,
              "identity_status": "pass" if discrepancy <= ceiling else "fatal_validation"}
    _require_pass(location, record, "identity_status")
    return record


def _y_scalar(after: dict[str, float], branch_values: dict[str, float],
              coefficients: dict[str, int], before: float, location: str) -> dict[str, Any]:
    components = {key: branch_values[key] for key in coefficients}
    direct = _linear(after, coefficients)
    component = _linear(components, coefficients)
    ceiling = math.fsum([
        _rsum([coefficients[key] * after[key] for key in coefficients]),
        _rsum([coefficients[key] * components[key] for key in coefficients]),
        math.fsum(abs(coefficients[key]) * _rsum([after[key], -before])
                  for key in coefficients),
    ])
    return _scalar_record(components, direct, component, ceiling, location)


def _d_scalar(q: torch.Tensor, vector: torch.Tensor, vector_error: torch.Tensor,
              deltas: dict[str, torch.Tensor], branch_values: dict[str, float],
              coefficients: dict[str, int], location: str,
              extra_errors: dict[str, torch.Tensor] | None = None) -> dict[str, Any]:
    components = {key: branch_values[key] for key in coefficients}
    direct = _finite(float(torch.dot(q, vector).item()), "direct D")
    component = _linear(components, coefficients)
    ceiling_terms = [
        _dot_roundoff(q, vector),
        math.fsum(abs(coefficients[key]) * _dot_roundoff(q, deltas[key])
                  for key in coefficients),
        _rsum([coefficients[key] * components[key] for key in coefficients]),
        float(torch.dot(q.abs(), vector_error).item()),
    ]
    if extra_errors is not None:
        ceiling_terms.append(float(torch.dot(q.abs(), sum(
            (extra_errors[key] * abs(coefficients[key]) for key in coefficients),
            torch.zeros_like(q))).item()))
    ceiling = math.fsum(ceiling_terms)
    return _scalar_record(components, direct, component, ceiling, location)


def _r_scalar(y_record: dict[str, Any], d_record: dict[str, Any],
              branch_y: dict[str, float], branch_d: dict[str, float],
              branch_r: dict[str, float], coefficients: dict[str, int], location: str) -> dict[str, Any]:
    components = {key: branch_r[key] for key in coefficients}
    direct = _finite(y_record["direct_value"] - d_record["direct_value"], "direct R")
    component = _linear(components, coefficients)
    ceiling = math.fsum([
        y_record["identity_rounding_ceiling"], d_record["identity_rounding_ceiling"],
        _rsum([y_record["direct_value"], -d_record["direct_value"]]),
        math.fsum(abs(coefficients[key]) * _rsum([branch_y[key], -branch_d[key]])
                  for key in coefficients),
        _rsum([coefficients[key] * components[key] for key in coefficients]),
    ])
    return _scalar_record(components, direct, component, ceiling, location)


def _factor_requirements(name: str) -> list[str]:
    if name in ("direction_at_current_norm", "direction_at_lagged_norm"):
        return ["direction"]
    if name in ("norm_at_current_direction", "norm_at_lagged_direction"):
        return ["norm"]
    if name == "interaction":
        return ["direction", "norm"]
    return []


def _factor_status(kind: str, required: list[str], leverage: dict[str, Any], defined: bool) -> str:
    if kind not in required:
        return "not_required"
    field = "unit_direction_distance" if kind == "direction" else "relative_norm_separation"
    flag = "direction_leverage" if kind == "direction" else "norm_leverage"
    if not defined or leverage[field] is None:
        return "unavailable"
    return "qualified" if leverage[flag] else "weak"


def _assemble_impl(before_parameters: dict[str, torch.Tensor], endpoints: dict[str, Any],
                   candidates: dict[str, Any], probes: dict[str, dict[str, torch.Tensor]], *,
                   artifact_id: str, profile: str, native_device: str,
                   guard: Callable[[str], None] | None) -> dict[str, Any]:
    _fail(guard is None or callable(guard), "guard must be callable or None")
    _guard(guard, "before:assembly:validation")
    _fail(type(artifact_id) is str and _ARTIFACT_ID.fullmatch(artifact_id) is not None,
          "artifact_id must be 1-128 canonical ASCII characters")
    _fail(type(profile) is str and profile in PROFILE, "unknown measurement profile")
    _fail(type(native_device) is str and bool(native_device), "native_device must be a string")
    spec = PROFILE[profile]
    try:
        device = torch.device(native_device)
    except (RuntimeError, ValueError) as exc:
        raise ValueError("invalid native_device") from exc
    _fail(device.type == spec["device_kind"], "profile/native device-kind mismatch")
    _fail(profile != FIXTURE_PROFILE or str(device) == "cpu", "fixture requires canonical CPU device")
    _validate_parameter_map(before_parameters, spec, device, "before_parameters")
    _keys(endpoints, BRANCHES, "endpoints")
    _keys(probes, PROBES, "probes")
    input_width = spec["shapes"][0][1]
    for probe_key, count in zip(PROBES, spec["counts"]):
        _validate_probe(probes[probe_key], count, input_width, spec["shapes"][2][0],
                        f"probe {probe_key}")
    _fail(_tensor_bytes(probes["train_probe_noisy"]["inputs"], "<f4") ==
          _tensor_bytes(probes["train_probe_clean"]["inputs"], "<f4"),
          "training probes must have byte-identical ordered inputs")

    _keys(candidates, ("branches", "leverage", "delivered_pairs"), "candidates")
    _keys(candidates["branches"], BRANCHES, "candidate branches")
    _keys(candidates["leverage"], LEVERAGE_KEYS, "candidate leverage")
    leverage = candidates["leverage"]
    for key in ("current_norm", "lagged_norm"):
        _fail(type(leverage[key]) is float and math.isfinite(leverage[key]) and leverage[key] >= 0.0,
              f"candidate leverage {key} invalid")
    _fail(type(leverage["signed_norm_difference"]) is float and
          math.isfinite(leverage["signed_norm_difference"]), "signed norm difference invalid")
    for key in ("relative_norm_separation", "current_lagged_norm_ratio",
                "unit_direction_distance"):
        _fail(leverage[key] is None or
              (type(leverage[key]) is float and math.isfinite(leverage[key]) and leverage[key] >= 0.0),
              f"candidate leverage {key} invalid")
    _fail(leverage["unit_direction_cosine"] is None or
          (type(leverage["unit_direction_cosine"]) is float and
           math.isfinite(leverage["unit_direction_cosine"])), "candidate leverage cosine invalid")
    _fail(leverage["ratio_reason"] in (None, "zero_lagged_norm") and
          leverage["direction_reason"] in (None, "zero_norm"), "candidate leverage reason invalid")
    for key in ("norm_leverage", "direction_leverage"):
        _fail(type(leverage[key]) is bool, f"candidate leverage {key} invalid")
    _keys(candidates["delivered_pairs"], PAIR_KEYS, "candidate delivered pairs")
    for key, pair in candidates["delivered_pairs"].items():
        _keys(pair, ("distance", "cosine", "cosine_reason"), f"candidate pair {key}")
        _fail(pair["distance"] is None or
              (type(pair["distance"]) is float and math.isfinite(pair["distance"]) and
               pair["distance"] >= 0.0), f"candidate pair {key} distance invalid")
        _fail(pair["cosine"] is None or
              (type(pair["cosine"]) is float and math.isfinite(pair["cosine"])),
              f"candidate pair {key} cosine invalid")
        _fail(pair["cosine_reason"] in (None, "zero_norm", "undefined_branch"),
              f"candidate pair {key} reason invalid")

    components = sum(math.prod(shape) for shape in spec["shapes"])
    defined: dict[str, bool] = {}
    delivered: dict[str, torch.Tensor | None] = {}
    for branch in BRANCHES:
        defined[branch], delivered[branch] = _validate_candidate(
            candidates["branches"][branch], branch, components, device)
        _fail((defined[branch] and endpoints[branch] is not None) or
              (not defined[branch] and endpoints[branch] is None),
              f"endpoint/candidate domain mismatch for {branch}")
        if defined[branch]:
            _validate_parameter_map(endpoints[branch], spec, device, f"endpoint {branch}")
    rebuilt = _RESPONSE.construct(delivered["raw"], delivered["current"], delivered["lagged"])
    _fail(_tree_exact(candidates, rebuilt), "candidate metadata is not bound to supplied gradients")
    _guard(guard, "after:assembly:validation")

    before_records: dict[str, Any] = {}
    before_private: dict[str, Any] = {}
    for probe_key in PROBES:
        record, private = _before_probe(probe_key, probes[probe_key], before_parameters,
                                        spec, device, guard)
        before_records[probe_key], before_private[probe_key] = record, private
    measurement_before = {"probe_order": list(PROBES), "probes": before_records}

    before_flat = _operation(
        guard, "assembly:before_flat_copy",
        lambda: torch.cat([before_parameters[key].detach().to("cpu", torch.float64).reshape(-1)
                           for key in PARAMETERS]).contiguous().clone())
    branch_records: dict[str, Any] = {}
    branch_private: dict[str, Any] = {}
    for branch in BRANCHES:
        if not defined[branch]:
            branch_records[branch] = None
            branch_private[branch] = None
            continue
        displacement_record, displacement_private = _operation(
            guard, f"branch:{branch}:displacement",
            lambda b=branch: _displacement(before_flat, endpoints[b], b))
        probe_records = {}
        for probe_key in PROBES:
            probe_records[probe_key] = _branch_probe(
                probe_key, probes[probe_key], endpoints[branch], before_records[probe_key],
                before_private[probe_key], displacement_private, spec, device, artifact_id,
                guard, branch)
        branch_records[branch] = {"displacement": displacement_record, "probes": probe_records}
        branch_private[branch] = displacement_private

    pairs: dict[str, Any] = {}
    for left, right in itertools.combinations(BRANCHES, 2):
        key = f"{left}__{right}"
        def make_pair(left: str = left, right: str = right) -> dict[str, Any]:
            mask = {left: defined[left], right: defined[right]}
            if not all(mask.values()):
                return {"defined": False, "defined_mask": mask,
                        "reason": "domain_undefined_required_branch",
                        "delivered_gradient": _geometry(None, None, mask),
                        "delta": _geometry(None, None, mask),
                        "delta_data": _geometry(None, None, mask),
                        "full_data_distance_discrepancy": None,
                        "full_data_rounding_ceiling": None,
                        "rounding_status": "domain_undefined"}
            full_left, full_right = branch_private[left]["delta"], branch_private[right]["delta"]
            data_left, data_right = branch_private[left]["delta_data"], branch_private[right]["delta_data"]
            _, full_error = _linear_vector(
                {left: full_left, right: full_right}, {left: 1, right: -1})
            _, data_error = _linear_vector(
                {left: data_left, right: data_right}, {left: 1, right: -1})
            full_difference = (full_left - full_right).contiguous()
            data_difference = (data_left - data_right).contiguous()
            construction = (full_error + data_error + branch_private[left]["delta_data_error"] +
                            branch_private[right]["delta_data_error"])
            full_geometry = _geometry(full_left, full_right, mask)
            data_geometry = _geometry(data_left, data_right, mask)
            discrepancy = abs(full_geometry["distance"] - data_geometry["distance"])
            ceiling = math.fsum([_norm(construction), _norm_roundoff(full_difference),
                                 _norm_roundoff(data_difference)])
            record = {"defined": True, "defined_mask": mask, "reason": None,
                      "delivered_gradient": _geometry(delivered[left], delivered[right], mask),
                      "delta": full_geometry, "delta_data": data_geometry,
                      "full_data_distance_discrepancy": discrepancy,
                      "full_data_rounding_ceiling": ceiling,
                      "rounding_status": "pass" if discrepancy <= ceiling else "fatal_validation"}
            _require_pass(f"branch_pairs.{key}.full_data", record, "rounding_status")
            return record
        pairs[key] = _operation(guard, f"pair:{key}", make_pair)

    contrast_records: dict[str, Any] = {}
    for contrast_name, coefficients in COEFFICIENTS.items():
        def make_contrast(name: str = contrast_name,
                          coefficients: dict[str, int] = coefficients) -> dict[str, Any]:
            mask = {branch: defined[branch] for branch in BRANCHES}
            contrast_defined = all(defined[key] for key in coefficients)
            requirements = _factor_requirements(name)
            factor = {kind: _factor_status(kind, requirements, candidates["leverage"], contrast_defined)
                      for kind in ("direction", "norm")}
            base = {"coefficients": dict(coefficients), "branch_defined_mask": mask,
                    "defined": contrast_defined,
                    "reason": None if contrast_defined else "domain_undefined_required_branch",
                    "factor_requirements": requirements, "factor_leverage": factor}
            if not contrast_defined:
                return {**base, "vector": None, "probes": None}

            deltas = {key: branch_private[key]["delta"] for key in coefficients}
            data = {key: branch_private[key]["delta_data"] for key in coefficients}
            vector, vector_error = _linear_vector(deltas, coefficients)
            data_vector, data_vector_error = _linear_vector(data, coefficients)
            data_construction = sum(
                (branch_private[key]["delta_data_error"] * abs(coefficients[key])
                 for key in coefficients), torch.zeros_like(vector))
            agreement_difference_vector, agreement_difference_error = _linear_vector(
                {"full": vector, "data": data_vector}, {"full": 1, "data": -1})
            agreement_ceiling = math.fsum([
                _norm(vector_error + data_vector_error + data_construction +
                      agreement_difference_error),
                _norm_roundoff(agreement_difference_vector),
            ])
            agreement_difference = _norm(agreement_difference_vector)
            agreement = {"difference_norm": agreement_difference,
                         "rounding_ceiling": agreement_ceiling,
                         "status": "pass" if agreement_difference <= agreement_ceiling
                         else "fatal_validation"}
            _require_pass(f"contrasts.{name}.vector.full_data_agreement", agreement, "status")
            vector_record = {"delta": _vector_summary(vector),
                             "delta_data": _vector_summary(data_vector),
                             "full_data_agreement": agreement}
            probe_records = {}
            for probe_key in PROBES:
                before = before_records[probe_key]
                q = before_private[probe_key]["q"]
                cpu_after = {key: branch_records[key]["probes"][probe_key]["cpu64"]["after_ce"]
                             for key in coefficients}
                cpu_y = {key: branch_records[key]["probes"][probe_key]["cpu64"]["Y"]
                         for key in coefficients}
                cpu_d = {key: branch_records[key]["probes"][probe_key]["cpu64"]["D"]
                         for key in coefficients}
                cpu_ddata = {key: branch_records[key]["probes"][probe_key]["cpu64"]["Ddata"]
                             for key in coefficients}
                cpu_r = {key: branch_records[key]["probes"][probe_key]["cpu64"]["R"]
                         for key in coefficients}
                prefix = f"contrasts.{name}.probes.{probe_key}"
                y_record = _y_scalar(cpu_after, cpu_y, coefficients,
                                     before["cpu64"]["before_ce"], f"{prefix}.cpu64.Y")
                d_record = _d_scalar(q, vector, vector_error, deltas, cpu_d, coefficients,
                                     f"{prefix}.cpu64.D")
                ddata_record = _d_scalar(
                    q, data_vector, data_vector_error, data, cpu_ddata, coefficients,
                    f"{prefix}.cpu64.Ddata",
                    {key: branch_private[key]["delta_data_error"] for key in coefficients})
                r_record = _r_scalar(y_record, d_record, cpu_y, cpu_d, cpu_r, coefficients,
                                     f"{prefix}.cpu64.R")

                native_after = {
                    key: branch_records[key]["probes"][probe_key]["native32"]["after_ce"]
                    for key in coefficients}
                native_y = {key: branch_records[key]["probes"][probe_key]["native32"]["Y"]
                            for key in coefficients}
                native_y_record = _y_scalar(native_after, native_y, coefficients,
                                             before["native32"]["before_ce"],
                                             f"{prefix}.native32.Y")
                native_direct = native_y_record["direct_value"]
                cpu_direct = y_record["direct_value"]
                native_ceiling = math.fsum([math.fsum(
                    abs(coefficients[key]) *
                    branch_records[key]["probes"][probe_key]["concordance"]["after"]
                    ["descriptive_ceiling"] for key in coefficients),
                    max(_rsum([coefficients[key] * native_after[key] for key in coefficients]),
                        _rsum([coefficients[key] * cpu_after[key] for key in coefficients])),
                ])

                auxiliary_rows = []
                if probe_key == "auxiliary_clean":
                    for index, before_chunk in enumerate(before["auxiliary_chunks"]):
                        cpu_chunk_after = {key: branch_records[key]["probes"][probe_key]
                                           ["auxiliary_chunks"][index]["cpu64_after_ce"]
                                           for key in coefficients}
                        cpu_chunk_y = {key: branch_records[key]["probes"][probe_key]
                                      ["auxiliary_chunks"][index]["cpu64_Y"]
                                      for key in coefficients}
                        native_chunk_after = {key: branch_records[key]["probes"][probe_key]
                                              ["auxiliary_chunks"][index]["native32_after_ce"]
                                              for key in coefficients}
                        native_chunk_y = {key: branch_records[key]["probes"][probe_key]
                                          ["auxiliary_chunks"][index]["native32_Y"]
                                          for key in coefficients}
                        cy = _y_scalar(cpu_chunk_after, cpu_chunk_y, coefficients,
                                       before_chunk["cpu64_before_ce"],
                                       f"{prefix}.auxiliary_chunks.{index}.cpu64.Y")
                        ny = _y_scalar(native_chunk_after, native_chunk_y, coefficients,
                                       before_chunk["native32_before_ce"],
                                       f"{prefix}.auxiliary_chunks.{index}.native32.Y")
                        chunk_ceiling = math.fsum([math.fsum(
                            abs(coefficients[key]) * branch_records[key]["probes"][probe_key]
                            ["auxiliary_chunks"][index]["concordance"]["after"]
                            ["descriptive_ceiling"] for key in coefficients),
                            max(_rsum([coefficients[key] * native_chunk_after[key]
                                       for key in coefficients]),
                                _rsum([coefficients[key] * cpu_chunk_after[key]
                                       for key in coefficients])),
                        ])
                        auxiliary_rows.append({"chunk_index": index,
                                               "start": before_chunk["start"],
                                               "end": before_chunk["end"],
                                               "count": before_chunk["count"],
                                               "input_sha256": before_chunk["input_sha256"],
                                               "label_sha256": before_chunk["label_sha256"],
                                               "cpu64": {"Y": cy}, "native32": {"Y": ny},
                                               "concordance": {"Y": _concordance(
                                                   ny["direct_value"], cy["direct_value"],
                                                   chunk_ceiling)}})
                probe_records[probe_key] = {
                    "cpu64": {"Y": y_record, "D": d_record, "Ddata": ddata_record,
                              "R": r_record},
                    "native32": {"Y": native_y_record},
                    "concordance": {"Y": _concordance(native_direct, cpu_direct, native_ceiling)},
                    "auxiliary_chunks": auxiliary_rows}
            return {**base, "vector": vector_record, "probes": probe_records}
        contrast_records[contrast_name] = _operation(guard, f"contrast:{contrast_name}", make_contrast)

    return {"measurement_before": measurement_before, "branches": branch_records,
            "comparisons": {"branch_pairs": pairs, "contrasts": contrast_records}}


def assemble(before_parameters: dict[str, torch.Tensor], endpoints: dict[str, Any],
             candidates: dict[str, Any], probes: dict[str, dict[str, torch.Tensor]], *,
             artifact_id: str, profile: str, native_device: str = "cpu",
             guard: Callable[[str], None] | None = None) -> dict[str, Any]:
    """Validate inputs and return the complete dataset-free measurement mapping."""
    _fail(not torch.is_inference_mode_enabled(), "assemble is unavailable under torch.inference_mode")
    with torch.autocast(device_type="cpu", enabled=False):
        return _assemble_impl(before_parameters, endpoints, candidates, probes,
                              artifact_id=artifact_id, profile=profile,
                              native_device=native_device, guard=guard)
