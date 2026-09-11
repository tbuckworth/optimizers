#!/usr/bin/env python3
"""Pure functional loss/gradient measurements for iteration 007.

Public API: ``evaluate(parameters, inputs, labels, *, chunk_size,
with_gradient=True, precision='cpu64', guard=None)``. ``parameters`` is an owning plain
dict in exact order ``0.weight, 0.bias, 2.weight, 2.bias`` with native float32
tensors. The function performs no loading, normalization, deduplication, model
mutation, RNG operation, or global-setting change.

The returned plain dict has exact keys ``precision``, ``shape_mode``,
``parameter_order``, ``count``, ``chunks``, ``mean_cross_entropy``, and
``mean_gradient``. Each chunk has ``start``, ``end``, ``count``,
``sum_cross_entropy``, and ``mean_cross_entropy``. Scalars are Python
primitives. The optional gradient is an owning contiguous tensor in canonical
flat order, on CPU float64 for ``cpu64`` or the supplied device in float32 for
``native32``. Tiny-shape acceptance is an engineering-fixture mode, not
scientific-shape certification. An optional ``guard(label)`` callback brackets
validation, copies, every chunk, and final reduction; its exceptions propagate.
The caller is responsible for an observational, RNG-neutral resource callback.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


PARAMETER_ORDER = ("0.weight", "0.bias", "2.weight", "2.bias")
SCIENTIFIC_SHAPES = ((64, 784), (64,), (10, 64), (10,))
RESULT_KEYS = ("precision", "shape_mode", "parameter_order", "count", "chunks",
               "mean_cross_entropy", "mean_gradient")
CHUNK_KEYS = ("start", "end", "count", "sum_cross_entropy", "mean_cross_entropy")


def _fail(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _owns_contiguous_storage(value: torch.Tensor) -> bool:
    return (value.is_contiguous() and value.storage_offset() == 0 and value._base is None and
            value.untyped_storage().nbytes() == value.numel() * value.element_size())


def _validate_parameters(parameters: Any) -> tuple[list[torch.Tensor], str, int, int]:
    _fail(type(parameters) is dict, "parameters must be a plain dict")
    _fail(tuple(parameters.keys()) == PARAMETER_ORDER, "parameter keys/order mismatch")
    values = list(parameters.values())
    _fail(all(type(x) is torch.Tensor for x in values), "all parameters must be tensors")
    devices = [str(x.device) for x in values]
    _fail(len(set(devices)) == 1, "parameters must share one native device")
    _fail(values[0].device.type in ("cpu", "cuda"), "only CPU/CUDA native devices are supported")
    seen_storage = set()
    for name, value in zip(PARAMETER_ORDER, values):
        _fail(value.dtype == torch.float32, f"{name} must be native float32")
        _fail(_owns_contiguous_storage(value), f"{name} must own contiguous storage")
        _fail(bool(torch.isfinite(value).all()), f"{name} contains nonfinite values")
        storage = value.untyped_storage()
        identity = (storage.data_ptr(), storage.nbytes())
        _fail(identity not in seen_storage, "parameter tensors must not alias")
        seen_storage.add(identity)
    w1, b1, w2, b2 = values
    shapes = tuple(tuple(x.shape) for x in values)
    hidden, input_width = w1.shape if w1.ndim == 2 else (-1, -1)
    classes, hidden2 = w2.shape if w2.ndim == 2 else (-1, -1)
    if shapes == SCIENTIFIC_SHAPES:
        mode = "scientific_784_64_10"
    else:
        _fail(w1.ndim == 2 and b1.ndim == 1 and w2.ndim == 2 and b2.ndim == 1,
              "generic tiny parameters have wrong ranks")
        _fail(hidden > 0 and input_width > 0 and classes > 1 and hidden2 == hidden and
              b1.shape == (hidden,) and b2.shape == (classes,), "generic tiny shapes mismatch")
        _fail(max(hidden, input_width, classes) <= 16 and
              sum(x.numel() for x in values) <= 512, "non-scientific shape exceeds fixture bounds")
        mode = "generic_tiny_mlp"
    return values, mode, input_width, classes


def _validate_inputs(inputs: Any, labels: Any, input_width: int, classes: int,
                     parameter_device: torch.device, precision: str) -> None:
    _fail(type(inputs) is torch.Tensor and inputs.dtype == torch.float32 and inputs.ndim == 2,
          "inputs must be a float32 rank-two tensor")
    _fail(_owns_contiguous_storage(inputs), "inputs must own contiguous storage")
    _fail(inputs.shape[0] > 0 and inputs.shape[1] == input_width, "input shape mismatch or empty input")
    _fail(bool(torch.isfinite(inputs).all()), "inputs contain nonfinite values")
    _fail(type(labels) is torch.Tensor and labels.dtype == torch.int64 and labels.ndim == 1,
          "labels must be an int64 vector")
    _fail(_owns_contiguous_storage(labels), "labels must own contiguous storage")
    _fail(labels.shape[0] == inputs.shape[0], "label count mismatch")
    _fail(bool(((labels >= 0) & (labels < classes)).all()), "label outside class range")
    if precision == "native32":
        _fail(inputs.device == parameter_device and labels.device == parameter_device,
              "native32 inputs, labels, and parameters must share a device")


def _functional_logits(parameters: list[torch.Tensor], inputs: torch.Tensor) -> torch.Tensor:
    w1, b1, w2, b2 = parameters
    return F.linear(F.relu(F.linear(inputs, w1, b1)), w2, b2)


def evaluate(parameters: dict[str, torch.Tensor], inputs: torch.Tensor, labels: torch.Tensor,
             *, chunk_size: int, with_gradient: bool = True,
             precision: str = "cpu64", guard=None) -> dict[str, Any]:
    """Evaluate ordered chunk sums and the corresponding mean-loss gradient."""
    _fail(not torch.is_inference_mode_enabled(), "evaluate is unavailable under torch.inference_mode")
    _fail(type(chunk_size) is int and chunk_size > 0, "chunk_size must be a positive int")
    _fail(type(with_gradient) is bool, "with_gradient must be bool")
    _fail(type(precision) is str and precision in ("cpu64", "native32"), "unknown precision")
    _fail(guard is None or callable(guard), "guard must be callable or None")
    def check(label):
        if guard is not None:
            guard(label)
    check("loss.before_validation")
    native, shape_mode, input_width, classes = _validate_parameters(parameters)
    native_device = native[0].device
    _validate_inputs(inputs, labels, input_width, classes, native_device, precision)
    check("loss.after_validation")

    if precision == "cpu64":
        work_device, work_dtype = torch.device("cpu"), torch.float64
    else:
        work_device, work_dtype = native_device, torch.float32
    grad_context = torch.enable_grad() if with_gradient else torch.no_grad()
    with torch.autocast(device_type=work_device.type, enabled=False), grad_context:
        check("loss.before_copy")
        work_parameters = [x.detach().to(device=work_device, dtype=work_dtype).contiguous().clone()
                           .requires_grad_(with_gradient) for x in native]
        work_inputs = inputs.detach().to(device=work_device, dtype=work_dtype).contiguous().clone()
        work_labels = labels.detach().to(device=work_device).contiguous().clone()
        check("loss.after_copy")

        total_sum = torch.zeros((), device=work_device, dtype=work_dtype)
        gradient_sum = [torch.zeros_like(x) for x in work_parameters] if with_gradient else None
        chunks = []
        count = int(work_inputs.shape[0])
        for start in range(0, count, chunk_size):
            end = min(start + chunk_size, count)
            check(f"loss.before_chunk.{start}.{end}")
            logits = _functional_logits(work_parameters, work_inputs[start:end])
            _fail(logits.dtype == work_dtype, "logits dtype changed from requested precision")
            _fail(bool(torch.isfinite(logits).all()), "nonfinite logits")
            loss_sum = F.cross_entropy(logits, work_labels[start:end], weight=None,
                                       reduction="sum", label_smoothing=0.0)
            _fail(loss_sum.dtype == work_dtype, "loss dtype changed from requested precision")
            _fail(bool(torch.isfinite(loss_sum)), "nonfinite cross entropy")
            total_sum = total_sum + loss_sum.detach()
            if with_gradient:
                gradient = torch.autograd.grad(loss_sum, tuple(work_parameters), create_graph=False,
                                               retain_graph=False, allow_unused=False)
                _fail(all(x.dtype == work_dtype for x in gradient), "gradient dtype mismatch")
                _fail(all(bool(torch.isfinite(x).all()) for x in gradient), "nonfinite loss gradient")
                gradient_sum = [old + new.detach() for old, new in zip(gradient_sum, gradient)]
            chunk_count = end - start
            chunk_sum = float(loss_sum.detach().item())
            chunks.append({"start": start, "end": end, "count": chunk_count,
                           "sum_cross_entropy": chunk_sum,
                           "mean_cross_entropy": float((loss_sum.detach() / chunk_count).item())})
            check(f"loss.after_chunk.{start}.{end}")

        check("loss.before_reduction")
        _fail(total_sum.dtype == work_dtype and bool(torch.isfinite(total_sum)),
              "invalid accumulated cross entropy")
        mean_loss = float((total_sum / count).item())
        if with_gradient:
            flat = torch.cat([(x / count).reshape(-1) for x in gradient_sum])
            mean_gradient = flat.detach().contiguous().clone()
            _fail(mean_gradient._base is None and mean_gradient.dtype == work_dtype and
                  bool(torch.isfinite(mean_gradient).all()), "invalid mean gradient")
        else:
            mean_gradient = None
    result = {"precision": precision, "shape_mode": shape_mode,
              "parameter_order": list(PARAMETER_ORDER), "count": count,
              "chunks": chunks, "mean_cross_entropy": mean_loss,
              "mean_gradient": mean_gradient}
    _fail(tuple(result.keys()) == RESULT_KEYS and
          all(tuple(chunk.keys()) == CHUNK_KEYS for chunk in chunks), "internal result schema error")
    check("loss.after_reduction")
    return result
