"""Pure supplied-IDX-bytes to exact owned runtime data/probe bindings.

No filesystem, RNG, plan generation, model, validation/test evaluation or device
selection. Only the scientific profile and versioned MLP fixture are accepted.
Callers supply immutable bytes and independently frozen expected file identities;
matching them does not establish acquisition history or source execution.
"""
from __future__ import annotations

import hashlib
import re
import struct

import numpy as np
import torch

import identity_codec as codec
import plan_bindings as plans

SCIENTIFIC = "scientific_mnist_current32_v1"
MLP_FIXTURE = "fixture_tiny_mlp_cpu_v1"
DATASETS = ("train_inputs", "train_clean_labels", "train_noisy_labels",
            "auxiliary_inputs", "auxiliary_clean_labels")
PROBES = ("batch_noisy", "train_probe_noisy", "train_probe_clean", "auxiliary_clean")
_FILES = ("training_images", "training_labels")
_SHA = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)
_SPEC = {SCIENTIFIC: (60000, 28, 28, 10, 500), MLP_FIXTURE: (30, 1, 3, 2, 1)}


class DataBindingError(ValueError):
    """Malformed bytes/metadata or a mismatch with recomputed data/probe values."""


def _require(condition, message):
    if not condition:
        raise DataBindingError(message)


def _keys(value, expected, where):
    _require(type(value) is dict and tuple(value) == tuple(expected)
             and all(type(key) is str for key in value), f"{where}: exact ordered keys required")


def _check_files(images, labels, expected):
    _keys(expected, _FILES, "expected_files")
    for key, data in zip(_FILES, (images, labels)):
        _require(type(data) is bytes, f"{key}: exact immutable bytes required")
        receipt = expected[key]
        _keys(receipt, ("sha256", "size_bytes"), key)
        _require(type(receipt["sha256"]) is str and _SHA.fullmatch(receipt["sha256"]) is not None,
                 f"{key}: invalid lowercase SHA-256")
        _require(type(receipt["size_bytes"]) is int and receipt["size_bytes"] > 0,
                 f"{key}: invalid positive byte count")
        _require(len(data) == receipt["size_bytes"], f"{key}: actual byte count differs")
        _require(hashlib.sha256(data).hexdigest() == receipt["sha256"], f"{key}: actual SHA-256 differs")


def _parse_idx(images, labels, profile):
    """Headers and exact lengths are checked before creating buffer views."""
    count, rows, columns, classes, _ = _SPEC[profile]
    _require(len(images) >= 16 and len(labels) >= 8, "truncated IDX header")
    _require(struct.unpack_from(">IIII", images) == (2051, count, rows, columns),
             "image IDX header/profile differs")
    _require(struct.unpack_from(">II", labels) == (2049, count), "label IDX header/profile differs")
    _require(len(images) == 16+count*rows*columns and len(labels) == 8+count,
             "IDX length differs: trailing or truncated bytes")
    pixels = np.frombuffer(images, dtype=np.uint8, offset=16).reshape(count, rows*columns)
    targets = np.frombuffer(labels, dtype=np.uint8, offset=8)
    _require(bool((targets < classes).all()), "IDX label outside profile class range")
    return pixels, targets


def _reference(value, role):
    return dict(sha256=codec.tensor_digest(value), dtype=str(value.dtype),
                shape=list(value.shape), semantic_role=role)


def _materialize(images, labels, *, plan, identity, profile, expected_files):
    _require(type(profile) is str and profile in _SPEC, "unsupported data/probe profile")
    plans.validate_plan(plan, identity=identity, profile=profile)
    _check_files(images, labels, expected_files)
    pixels, targets = _parse_idx(images, labels, profile)
    train_indices, auxiliary_indices = plan["train_indices"], plan["auxiliary_indices"]

    def selected_inputs(indices):
        # Explicit CPU float32 conversion precedes exactly the canonical /255.
        return torch.tensor(pixels[indices.numpy()], dtype=torch.float32, device="cpu") / 255

    def selected_labels(indices):
        return torch.tensor(targets[indices.numpy()], dtype=torch.int64, device="cpu")

    clean = selected_labels(train_indices)
    noisy = clean.clone()
    mask = plan["replacement_uniforms"] < .9
    noisy[mask] = plan["replacement_digits"][mask]
    datasets = dict(train_inputs=selected_inputs(train_indices), train_clean_labels=clean,
                    train_noisy_labels=noisy, auxiliary_inputs=selected_inputs(auxiliary_indices),
                    auxiliary_clean_labels=selected_labels(auxiliary_indices))
    row = identity["anchor_update"]-1
    batch_indices, probe_indices = plan["training_batches"][row], plan["training_probe_indices"]
    probes = {
        "batch_noisy": dict(inputs=datasets["train_inputs"].index_select(0, batch_indices),
                            labels=noisy.index_select(0, batch_indices)),
        "train_probe_noisy": dict(inputs=datasets["train_inputs"].index_select(0, probe_indices),
                                  labels=noisy.index_select(0, probe_indices)),
        "train_probe_clean": dict(inputs=datasets["train_inputs"].index_select(0, probe_indices),
                                  labels=clean.index_select(0, probe_indices)),
        "auxiliary_clean": dict(inputs=datasets["auxiliary_inputs"].clone(),
                                labels=datasets["auxiliary_clean_labels"].clone()),
    }
    prefix = "mnist_training" if profile == SCIENTIFIC else "fixture"
    data_binding = dict(idx_files={key: dict(expected_files[key]) for key in _FILES},
        preprocessing=prefix+"_idx_uint8_to_torch_float32_div255_v1",
        arrays={key: _reference(value, key) for key, value in datasets.items()},
        replacement_count=int(mask.sum().item()), incorrect_label_count=int((clean != noisy).sum().item()))
    probe_binding = dict(
        next_batch=dict(row=row, coordinate_space="train_subset_position_v1",
                        inputs=_reference(probes["batch_noisy"]["inputs"], "batch_noisy_inputs"),
                        noisy_labels=_reference(probes["batch_noisy"]["labels"], "batch_noisy_labels")),
        training_probe_indices=probe_indices.clone(),
        training_probe_coordinate_space="train_subset_position_v1",
        training_probe_inputs=_reference(probes["train_probe_clean"]["inputs"], "training_probe_inputs"),
        training_probe_clean_labels=_reference(probes["train_probe_clean"]["labels"], "training_probe_clean_labels"),
        training_probe_noisy_labels=_reference(probes["train_probe_noisy"]["labels"], "training_probe_noisy_labels"),
        auxiliary_indices=auxiliary_indices.clone(),
        auxiliary_coordinate_space="original_training_idx_row_v1",
        auxiliary_inputs=_reference(probes["auxiliary_clean"]["inputs"], "auxiliary_inputs"),
        auxiliary_clean_labels=_reference(probes["auxiliary_clean"]["labels"], "auxiliary_clean_labels"),
        auxiliary_chunks=[], stream6="unused")
    chunk = _SPEC[profile][-1]
    for index in range(10):
        start, end = chunk*index, chunk*(index+1)
        probe_binding["auxiliary_chunks"].append(dict(chunk_index=index, start=start, end=end,
            inputs=_reference(probes["auxiliary_clean"]["inputs"][start:end].clone(), "auxiliary_chunk_inputs"),
            labels=_reference(probes["auxiliary_clean"]["labels"][start:end].clone(), "auxiliary_chunk_labels")))
    return dict(datasets=datasets, probes=probes, bindings=dict(data=data_binding, probes=probe_binding))


def materialize(images_bytes, labels_bytes, *, plan, identity, profile, expected_files):
    """Authenticate supplied buffers, parse exact IDX, and return owned CPU values.

    This neither reads files nor regenerates the plan. Invalid supplied bytes,
    metadata, schema or tensor representations raise bounded DataBindingError.
    Memory/resource failures remain failures, never permitted scientific nulls.
    """
    try:
        return _materialize(images_bytes, labels_bytes, plan=plan, identity=identity,
                            profile=profile, expected_files=expected_files)
    except DataBindingError:
        raise
    except (plans.BindingError, codec.CodecError) as error:
        raise DataBindingError(str(error)) from error
    except (RuntimeError, TypeError, ValueError, OverflowError, struct.error):
        raise DataBindingError("unsupported data/probe tensor or byte representation") from None


def _tensor_range(value):
    lower = value.untyped_storage().data_ptr()
    return lower, lower+value.untyped_storage().nbytes()


def _overlap(first, second):
    return first[0] < second[1] and second[0] < first[1]


def _compare(value, expected, *, ranges, plan_ranges, where):
    _require(type(value) is type(expected), f"{where}: exact type differs")
    if type(expected) is dict:
        _keys(value, tuple(expected), where)
        for key in expected:
            _compare(value[key], expected[key], ranges=ranges, plan_ranges=plan_ranges, where=where+"."+key)
    elif type(expected) is list:
        _require(len(value) == len(expected), f"{where}: sequence length differs")
        for index, (left, right) in enumerate(zip(value, expected)):
            _compare(left, right, ranges=ranges, plan_ranges=plan_ranges, where=where+f"[{index}]")
    elif type(expected) is torch.Tensor:
        _require(value.device.type == "cpu" and value.layout == torch.strided and not value.is_nested
                 and not value.requires_grad and not value.is_neg() and not value.is_conj(),
                 f"{where}: detached ordinary CPU tensor required")
        _require(value.dtype == expected.dtype and value.shape == expected.shape,
                 f"{where}: tensor dtype/shape differs")
        _require(value.is_contiguous() and value.storage_offset() == 0 and value._base is None
                 and value.untyped_storage().nbytes() == value.numel()*value.element_size(),
                 f"{where}: owning compact tensor required")
        if value.is_floating_point():
            _require(bool(torch.isfinite(value).all()), f"{where}: nonfinite tensor")
        interval = _tensor_range(value)
        _require(not any(_overlap(interval, prior) for prior in ranges+plan_ranges),
                 f"{where}: tensor storage overlaps output or plan")
        ranges.append(interval)
        # Shape and dtype already match; compare native bytes, including signed
        # zero. A matching digest is deliberately insufficient for this check.
        _require(value.numpy().tobytes(order="C") == expected.numpy().tobytes(order="C"),
                 f"{where}: exact tensor bytes differ")
    else:
        _require(value == expected, f"{where}: primitive value differs")


def validate_materialization(value, images_bytes, labels_bytes, *, plan, identity, profile, expected_files):
    """Recompute, validate exact ordered structure/ownership/bytes, return value.

    The returned indices and all runtime tensors must be distinct from both
    one another and the supplied plan's tensor storage. No hash-only shortcut,
    tolerance, casting, repair, field dropping or signed-zero normalization.
    """
    expected = materialize(images_bytes, labels_bytes, plan=plan, identity=identity,
                           profile=profile, expected_files=expected_files)
    try:
        plan_ranges = [_tensor_range(plan[key]) for key in plans.ARRAY_KEYS]
        _compare(value, expected, ranges=[], plan_ranges=plan_ranges, where="materialization")
    except DataBindingError:
        raise
    except (RuntimeError, TypeError, ValueError, OverflowError):
        raise DataBindingError("unsupported materialization structure or tensor representation") from None
    return value
