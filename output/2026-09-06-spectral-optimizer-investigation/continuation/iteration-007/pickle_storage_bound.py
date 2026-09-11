"""Conservative protocol-2 ``data.pkl`` bounds for the admitted I7 tree.

This module does not bound Torch's ZIP container or the bytes in storage
records.  ``PickleStorageBound.storage_nbytes`` is the hand-off to that proof.
"""
from __future__ import annotations

from dataclasses import dataclass
import collections
import copyreg
import hashlib
import inspect
import math
import os
from pathlib import Path
import pickle
import _compat_pickle
import _pickle
import stat
import sys


class PickleBoundError(RuntimeError):
    """The runtime or value is outside the proved serializer domain."""


@dataclass(frozen=True)
class PickleStorageBound:
    pickle_bytes: int
    storage_nbytes: tuple[int, ...]
    tensor_count: int
    node_count: int
    memo_slot_upper: int
    memoized_reference_count: int
    unique_string_count: int
    max_depth_seen: int


@dataclass(frozen=True)
class _DtypeSpec:
    dtype_attribute: str
    itemsize: int
    reduction: str
    storage_module: str
    storage_name: str
    dtype_module: str | None = None
    dtype_name: str | None = None


# Exact admitted dtype/reduction table for Torch 2.11.0+cu128 at the pinned
# revision.  uint32 is intentionally v3; it is not a legacy TypedStorage dtype.
_DTYPES = {
    "torch.uint8": _DtypeSpec("uint8", 1, "v2", "torch", "ByteStorage"),
    "torch.uint32": _DtypeSpec(
        "uint32", 4, "v3", "torch.storage", "UntypedStorage", "torch", "uint32"
    ),
    "torch.float32": _DtypeSpec("float32", 4, "v2", "torch", "FloatStorage"),
    "torch.float64": _DtypeSpec("float64", 8, "v2", "torch", "DoubleStorage"),
    "torch.int64": _DtypeSpec("int64", 8, "v2", "torch", "LongStorage"),
}
SUPPORTED_DTYPES = frozenset(_DTYPES)

_PINNED_PYTHON = (3, 12, 3)
_PINNED_TORCH_VERSION = "2.11.0+cu128"
_PINNED_TORCH_REVISION = "70d99e998b4955e0049d13a98d77ae1b14db1f45"
_PINNED_HASHES = {
    "python_binary": "1643dacd9feaedc58f3cc581e4d22577dfe25c09b10282936186ccf0f2e61118",
    "pickle.py": "865b5788a1e35433f89d047187a514057e15ddc2a301b06b5f85da62b4259c04",
    "_compat_pickle.py": "12c8356a3d40bd0a336f13d7c6e2bed50d5c1a876563766a3175a6b328b5855e",
    "copyreg.py": "c8eda41f05c6bf95a4da4726a530409d2485ae060b8d019b3a8034389a15d3e9",
    "collections/__init__.py": "0af967cd58036507b3d0fbc33b7a996c61dbb52a94b0b738c8bef12cd4cc7dd4",
    "torch/__init__.py": "0387d8b811b289287479c8bfdf4e1dac3a71b246f938d82da1331cf2dc8bf001",
    "torch/serialization.py": "ce5bc5ca6a8faa2b5aee2c72ff1f1d4b02c2a2d82e96886de74ae9fc436a7f67",
    "torch/_tensor.py": "cf98e9fccc99e34655fe7ac97d1e7bc0486bc9b0a8590472fdc96a148106f28f",
    "torch/_utils.py": "360bb474e47c671467520928b2e5509bd4bfa0298a13dae6727c358f85850594",
    "torch/storage.py": "faeb7a1021c8f01ba017afa281bf0e94deaf55fc9ee5fe742f2f09cc16467fe0",
    "torch/version.py": "c5a512fcd13f6195c117da82802740abd5eafbec7b21709ed8d9babec6c9f5d9",
}
_MAX_RECURSIVE_DEPTH = 256
_MAX_MEMO_SLOTS = 1 << 32
_BATCH_SIZE = 1000
_MEMO_OPCODE_MAX = 5


def _sha256_file(path: Path, maximum: int) -> str:
    if type(maximum) is not int or maximum < 0:
        raise PickleBoundError("invalid source-file read bound")
    digest = hashlib.sha256()
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        before = os.fstat(fd)
        signature = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        if (not stat.S_ISREG(before.st_mode) or before.st_nlink != 1
                or before.st_size > maximum):
            raise PickleBoundError("unsafe or oversized pinned runtime source")
        remaining = before.st_size
        while True:
            chunk = os.read(fd, min(1 << 20, remaining + 1))
            if not chunk:
                break
            digest.update(chunk)
            remaining -= len(chunk)
            if remaining < 0:
                raise PickleBoundError("pinned runtime source grew during read")
        after = os.fstat(fd)
        after_signature = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        if remaining != 0 or signature != after_signature:
            raise PickleBoundError("pinned runtime source changed during read")
        return digest.hexdigest()
    finally:
        os.close(fd)


def _runtime_source_paths() -> dict[str, Path]:
    import torch
    import torch._tensor
    import torch._utils
    import torch.storage
    import torch.version

    return {
        "python_binary": Path(sys.executable).resolve(),
        "pickle.py": Path(pickle.__file__).resolve(),
        "_compat_pickle.py": Path(_compat_pickle.__file__).resolve(),
        "copyreg.py": Path(copyreg.__file__).resolve(),
        "collections/__init__.py": Path(collections.__file__).resolve(),
        "torch/__init__.py": Path(torch.__file__).resolve(),
        "torch/serialization.py": Path(torch.serialization.__file__).resolve(),
        "torch/_tensor.py": Path(torch._tensor.__file__).resolve(),
        "torch/_utils.py": Path(torch._utils.__file__).resolve(),
        "torch/storage.py": Path(torch.storage.__file__).resolve(),
        "torch/version.py": Path(torch.version.__file__).resolve(),
    }


def _assert_runtime_objects():
    import torch

    if sys.implementation.name != "cpython" or sys.version_info[:3] != _PINNED_PYTHON:
        raise PickleBoundError("unproved Python runtime")
    if pickle.Pickler is not _pickle.Pickler:
        raise PickleBoundError("protocol-2 proof requires the C _pickle.Pickler")
    if str(torch.__version__) != _PINNED_TORCH_VERSION:
        raise PickleBoundError("unproved Torch version")
    if torch.version.git_version != _PINNED_TORCH_REVISION:
        raise PickleBoundError("unproved Torch source revision")
    if torch.save is not torch.serialization.save or torch.serialization.DEFAULT_PROTOCOL != 2:
        raise PickleBoundError("unproved torch.save binding or default protocol")

    parameters = inspect.signature(torch.serialization.save).parameters
    expected_defaults = {
        "pickle_module": pickle,
        "pickle_protocol": 2,
        "_use_new_zipfile_serialization": True,
        "_disable_byteorder_record": False,
    }
    if any(parameters[name].default is not expected
           for name, expected in expected_defaults.items()):
        raise PickleBoundError("unproved torch.save defaults")
    tls = torch.serialization._serialization_tls
    if tls.skip_data is not False or tls.materialize_fake_tensors is not False:
        raise PickleBoundError("skip-data serialization context is outside the proof")

    rebuilds = (
        (torch._utils._rebuild_tensor_v2, "_rebuild_tensor_v2"),
        (torch._utils._rebuild_tensor_v3, "_rebuild_tensor_v3"),
    )
    if any(fn.__module__ != "torch._utils" or fn.__name__ != name for fn, name in rebuilds):
        raise PickleBoundError("unproved Tensor rebuild globals")
    if (torch.Tensor.__reduce_ex__.__module__ != "torch._tensor"
            or torch.Tensor.__reduce_ex__.__name__ != "__reduce_ex__"):
        raise PickleBoundError("unproved Tensor reducer")
    if (collections.OrderedDict.__module__ != "collections"
            or collections.OrderedDict.__name__ != "OrderedDict"):
        raise PickleBoundError("unproved OrderedDict reducer global")
    dispatched_types = (torch.Tensor, collections.OrderedDict, type(torch.uint32))
    if any(value_type in copyreg.dispatch_table for value_type in dispatched_types):
        raise PickleBoundError("ambient copyreg reducer overrides the proved reduction")
    new_dtypes = torch.storage._new_dtypes()
    if torch.uint32 not in new_dtypes:
        raise PickleBoundError("uint32 no longer follows the proved v3 reduction")
    for name, spec in _DTYPES.items():
        dtype = getattr(torch, spec.dtype_attribute)
        if str(dtype) != name or ((spec.reduction == "v3") != (dtype in new_dtypes)):
            raise PickleBoundError("dtype reduction table differs from the proof")
        storage_class = getattr(torch, spec.storage_name)
        if (storage_class.__module__ != spec.storage_module
                or storage_class.__name__ != spec.storage_name):
            raise PickleBoundError("storage reducer global differs from the proof")
    return torch


def assert_pinned_pickle_runtime() -> None:
    """Check observable runtime/source pins used by the proof.

    This cannot establish the trusted-process assumptions: loaded objects may
    still have been maliciously monkeypatched, and values may still be mutated
    concurrently after this call.  Callers must enforce those separately.
    """
    _assert_runtime_objects()
    paths = _runtime_source_paths()
    if set(paths) != set(_PINNED_HASHES):
        raise PickleBoundError("incomplete pinned-source manifest")
    for name, path in paths.items():
        expected = _PINNED_HASHES[name]
        maximum = 32 << 20 if name == "python_binary" else 2 << 20
        if len(expected) != 64 or _sha256_file(path, maximum) != expected:
            raise PickleBoundError(f"unproved runtime source: {name}")


def protocol2_int_bytes(value: int) -> int:
    """Return the exact protocol-2 opcode/payload length for an exact int."""
    if type(value) is not int:
        raise PickleBoundError("integer bound requires an exact int")
    if 0 <= value <= 0xFF:
        return 2  # BININT1
    if 0 <= value <= 0xFFFF:
        return 3  # BININT2
    if -0x80000000 <= value <= 0x7FFFFFFF:
        return 5  # BININT

    # Minimal signed two's-complement byte count.  Positive values need a sign
    # bit; for negatives, ~value measures the non-sign-extension bits.
    magnitude_bits = value.bit_length() if value >= 0 else (~value).bit_length()
    payload_bytes = (magnitude_bits + 1 + 7) // 8
    if payload_bytes < 256:
        return 2 + payload_bytes  # LONG1 + uint8 length
    if payload_bytes <= 0x7FFFFFFF:
        return 5 + payload_bytes  # LONG4 + signed uint32-sized length field
    raise PickleBoundError("integer is too wide for the proved protocol-2 encoder")


def protocol2_tuple_of_ints_bytes(values: tuple[int, ...]) -> int:
    """Bound a fresh exact tuple of exact ints, including a worst-case PUT."""
    if type(values) is not tuple or any(type(value) is not int for value in values):
        raise PickleBoundError("tuple bound requires an exact tuple of exact ints")
    contents = sum(protocol2_int_bytes(value) for value in values)
    if not values:
        return 1  # EMPTY_TUPLE is not memoized by the protocol-2 pickler.
    if len(values) <= 3:
        return contents + 1 + _MEMO_OPCODE_MAX  # TUPLE1/2/3 + PUT
    return contents + 2 + _MEMO_OPCODE_MAX  # MARK + TUPLE + PUT


def _global_bytes(module: str, name: str) -> int:
    # GLOBAL opcode + two newline-terminated ASCII identifiers + worst PUT.
    try:
        module_bytes = module.encode("ascii")
        name_bytes = name.encode("ascii")
    except UnicodeEncodeError as exc:
        raise PickleBoundError("protocol-2 global names must be ASCII") from exc
    return 3 + len(module_bytes) + len(name_bytes) + _MEMO_OPCODE_MAX


def _full_string_bytes(utf8_bytes: int) -> int:
    # BINUNICODE + uint32 length + payload + worst PUT.
    if type(utf8_bytes) is not int or utf8_bytes < 0 or utf8_bytes > 0xFFFFFFFF:
        raise PickleBoundError("string is outside protocol-2 BINUNICODE bounds")
    return 5 + utf8_bytes + _MEMO_OPCODE_MAX


def _decimal_digits(value: int) -> int:
    if type(value) is not int or value < 0:
        raise PickleBoundError("decimal digit bound requires a nonnegative int")
    return 1 if value == 0 else len(str(value))


def protocol2_tensor_bytes(
    *,
    dtype_name: str,
    storage_nbytes: int,
    shape: tuple[int, ...],
    stride: tuple[int, ...],
    tensor_count_cap: int,
) -> int:
    """Pure prospective bound for one admitted plain Tensor reduction.

    ``tensor_count_cap`` includes every storage in this torch.save call,
    including a CPU tensor carrying a captured CUDA RNG state.
    """
    if type(dtype_name) is not str or dtype_name not in _DTYPES:
        raise PickleBoundError("unsupported Tensor dtype")
    if (type(storage_nbytes) is not int or storage_nbytes < 0
            or type(tensor_count_cap) is not int or not 1 <= tensor_count_cap <= _MAX_MEMO_SLOTS):
        raise PickleBoundError("invalid Tensor storage/count bound")
    if (type(shape) is not tuple or type(stride) is not tuple
            or len(shape) != len(stride)
            or any(type(value) is not int or value < 0 for value in shape + stride)):
        raise PickleBoundError("Tensor shape/stride must be nonnegative exact-int tuples")

    spec = _DTYPES[dtype_name]
    numel = math.prod(shape)
    if storage_nbytes != numel * spec.itemsize:
        raise PickleBoundError("Tensor storage is not exact for dtype and shape")
    persistent_numel = storage_nbytes if spec.reduction == "v3" else numel
    key_digits = _decimal_digits(tensor_count_cap - 1)

    rebuild = "_rebuild_tensor_v3" if spec.reduction == "v3" else "_rebuild_tensor_v2"
    rebuild_global = _global_bytes("torch._utils", rebuild)  # 38
    persistent_id = (
        _full_string_bytes(7)  # "storage": 17
        + _global_bytes(spec.storage_module, spec.storage_name)
        + _full_string_bytes(key_digits)
        + _full_string_bytes(3)  # "cpu": 13
        + protocol2_int_bytes(persistent_numel)
        + 7  # MARK + TUPLE + worst PUT for the five-item persistent tuple
        + 1  # BINPERSID
    )
    empty_ordered_dict = (
        _global_bytes("collections", "OrderedDict") + 1 + 1 + _MEMO_OPCODE_MAX
    )  # global, EMPTY_TUPLE, REDUCE, object PUT = 37
    dtype_global = 0
    if spec.reduction == "v3":
        assert spec.dtype_module is not None and spec.dtype_name is not None
        dtype_global = _global_bytes(spec.dtype_module, spec.dtype_name)

    return (
        rebuild_global
        + persistent_id
        + protocol2_int_bytes(0)  # storage offset
        + protocol2_tuple_of_ints_bytes(shape)
        + protocol2_tuple_of_ints_bytes(stride)
        + 1  # NEWFALSE for requires_grad
        + empty_ordered_dict
        + dtype_global
        + 7  # MARK + TUPLE + worst PUT for six (v2) or seven (v3) args
        + 1  # REDUCE
        + _MEMO_OPCODE_MAX  # Tensor object PUT
    )


def _list_batch_bytes(item_count: int) -> int:
    if item_count == 0:
        return 0
    if item_count == 1:
        return 1
    # The C exact-list fast path uses MARK+APPENDS even for a one-item tail.
    return 2 * ((item_count + _BATCH_SIZE - 1) // _BATCH_SIZE)


def _dict_batch_bytes(item_count: int) -> int:
    if item_count == 0:
        return 0
    if item_count == 1:
        return 1
    # The C exact-dict fast path repeats while the prior batch had exactly
    # BATCHSIZE items.  At an exact multiple it emits one final empty
    # MARK+SETITEMS pair.
    return 2 * (item_count // _BATCH_SIZE + 1)


def _utf8_surrogatepass_length(value: str, maximum: int) -> int:
    # Do not materialize value.encode(...).  The count stops as soon as the
    # caller's admitted limit is crossed.
    if len(value) > maximum:
        raise PickleBoundError("string exceeds admitted UTF-8 length")
    total = 0
    for character in value:
        codepoint = ord(character)
        total += (1 if codepoint <= 0x7F else 2 if codepoint <= 0x7FF
                  else 3 if codepoint <= 0xFFFF else 4)
        if total > maximum:
            raise PickleBoundError("string exceeds admitted UTF-8 length")
    return total


def _tensor_memo_slots(reduction: str, rank: int) -> int:
    # Rebuild global (1); PID strings/globals/tuple (5); size/stride tuples
    # when nonempty (2); hooks global/object (2); optional dtype global (1);
    # outer args tuple (1); Tensor (1).
    return 10 + (2 if rank else 0) + (1 if reduction == "v3" else 0)


def bound_protocol2_tree(
    tree,
    *,
    tensor_count_cap: int,
    allowed_dtypes: frozenset[str],
    max_depth: int,
    max_nodes: int,
    max_utf8_bytes: int,
    max_tensor_rank: int,
) -> PickleStorageBound:
    """Validate and bound one actual admitted tree without serializing it.

    The result covers only protocol-2 ``data.pkl``.  The caller must separately
    bind the full prospective topology/ranges and Torch ZIP accounting.
    """
    torch = _assert_runtime_objects()
    integer_limits = (tensor_count_cap, max_depth, max_nodes, max_utf8_bytes, max_tensor_rank)
    if any(type(value) is not int or value < 0 for value in integer_limits):
        raise PickleBoundError("bounds must be nonnegative exact ints")
    if (max_nodes < 1 or max_nodes > _MAX_MEMO_SLOTS
            or tensor_count_cap > max_nodes or max_utf8_bytes > 0xFFFFFFFF
            or max_depth > _MAX_RECURSIVE_DEPTH
            or max_tensor_rank > _MAX_RECURSIVE_DEPTH):
        raise PickleBoundError("unsafe traversal depth/node admission")
    if type(allowed_dtypes) is not frozenset or not allowed_dtypes <= SUPPORTED_DTYPES:
        raise PickleBoundError("allowed_dtypes must be a supported frozenset")

    seen: set[int] = set()
    active: set[int] = set()
    tensor_objects: set[int] = set()
    tensor_storages: set[int] = set()
    storage_nbytes: list[int] = []
    node_count = 0
    memo_slots = 0
    references = 0
    unique_strings = 0
    max_depth_seen = 0

    def add_memo_slots(count: int) -> None:
        nonlocal memo_slots
        memo_slots += count
        if memo_slots > _MAX_MEMO_SLOTS:
            raise PickleBoundError("memo index exceeds protocol-2 LONG_BINPUT domain")

    def visit(value, depth: int) -> int:
        nonlocal node_count, references, unique_strings, max_depth_seen
        node_count += 1
        if node_count > max_nodes:
            raise PickleBoundError("tree exceeds admitted node count")
        if depth > max_depth:
            raise PickleBoundError("tree exceeds admitted depth")
        max_depth_seen = max(max_depth_seen, depth)

        value_type = type(value)
        if value is None or value_type is bool:
            return 1
        if value_type is int:
            return protocol2_int_bytes(value)
        if value_type is float:
            if not math.isfinite(value):
                raise PickleBoundError("nonfinite scalar")
            return 9
        if value_type is str:
            identity = id(value)
            if identity in seen:
                references += 1
                return _MEMO_OPCODE_MAX
            seen.add(identity)
            add_memo_slots(1)
            unique_strings += 1
            return _full_string_bytes(_utf8_surrogatepass_length(value, max_utf8_bytes))
        if value_type is torch.Tensor:
            identity = id(value)
            if identity in tensor_objects:
                raise PickleBoundError("repeated Tensor identity is outside the admitted topology")
            tensor_objects.add(identity)
            if len(storage_nbytes) >= tensor_count_cap:
                raise PickleBoundError("tree exceeds admitted Tensor count")
            if (value.device.type != "cpu" or value.layout is not torch.strided
                    or value.is_quantized):
                raise PickleBoundError("Tensor must be ordinary dense strided CPU storage")
            if (value.requires_grad or value.grad is not None or not value.is_contiguous()
                    or value.storage_offset() != 0 or value._base is not None):
                raise PickleBoundError("Tensor must be detached, compact, and independently owned")
            if value.is_conj() or value.is_neg():
                raise PickleBoundError("Tensor view metadata is outside the proof")
            if torch._utils._get_obj_state(value) is not None:
                raise PickleBoundError("Tensor Python state is outside the proof")
            if torch._utils.get_tensor_metadata(value) != {}:
                raise PickleBoundError("Tensor metadata is outside the proof")

            rank = value.dim()
            if type(rank) is not int or rank > max_tensor_rank:
                raise PickleBoundError("Tensor rank exceeds admission")
            shape = tuple(value.size())
            stride = tuple(value.stride())
            if (len(shape) != rank or len(stride) != rank
                    or any(type(item) is not int or item < 0 for item in shape + stride)):
                raise PickleBoundError("Tensor has a symbolic or negative shape/stride")
            if tuple(value.names) != (None,) * rank:
                raise PickleBoundError("named Tensor is outside the proof")

            dtype_name = str(value.dtype)
            if dtype_name not in allowed_dtypes:
                raise PickleBoundError("Tensor dtype is not admitted")
            spec = _DTYPES[dtype_name]
            if value.dtype is not getattr(torch, spec.dtype_attribute):
                raise PickleBoundError("Tensor dtype global differs from the proof")
            raw_storage = value.untyped_storage()
            logical_bytes = value.numel() * value.element_size()
            if logical_bytes <= 0 or raw_storage.nbytes() != logical_bytes:
                raise PickleBoundError(
                    "zero-sized or non-exact Tensor storage is outside the proof"
                )
            storage_identity = raw_storage._cdata
            if storage_identity in tensor_storages:
                raise PickleBoundError("shared Tensor storage is outside the proof")
            tensor_storages.add(storage_identity)
            if torch.serialization.location_tag(raw_storage) != "cpu":
                raise PickleBoundError("Tensor location tag is outside the proof")

            if spec.reduction == "v2":
                if value.dtype in torch.storage._new_dtypes():
                    raise PickleBoundError("Tensor unexpectedly follows v3 reduction")
                if value._typed_storage()._pickle_storage_type() != spec.storage_name:
                    raise PickleBoundError("TypedStorage global differs from the proof")
            else:
                if (value.dtype not in torch.storage._new_dtypes()
                        or type(raw_storage) is not torch.storage.UntypedStorage):
                    raise PickleBoundError("Tensor unexpectedly differs from v3 reduction")

            storage_nbytes.append(logical_bytes)
            add_memo_slots(_tensor_memo_slots(spec.reduction, rank))
            return protocol2_tensor_bytes(
                dtype_name=dtype_name,
                storage_nbytes=logical_bytes,
                shape=shape,
                stride=stride,
                tensor_count_cap=tensor_count_cap,
            )

        if value_type not in (dict, list, tuple):
            raise PickleBoundError("tree contains an unsupported value")
        identity = id(value)
        if identity in active:
            raise PickleBoundError("cyclic container")
        if value_type is tuple and not value:
            return 1
        if identity in seen:
            references += 1
            return _MEMO_OPCODE_MAX
        seen.add(identity)
        add_memo_slots(1)
        active.add(identity)
        try:
            if value_type is dict:
                subtotal = 1 + _MEMO_OPCODE_MAX + _dict_batch_bytes(len(value))
                for key, item in value.items():
                    if type(key) not in (str, int) or type(key) is bool:
                        raise PickleBoundError("mapping keys must be exact strings or integers")
                    subtotal += visit(key, depth + 1)
                    subtotal += visit(item, depth + 1)
                return subtotal
            if value_type is list:
                return (1 + _MEMO_OPCODE_MAX + _list_batch_bytes(len(value))
                        + sum(visit(item, depth + 1) for item in value))
            wrapper = 1 + _MEMO_OPCODE_MAX if len(value) <= 3 else 2 + _MEMO_OPCODE_MAX
            return wrapper + sum(visit(item, depth + 1) for item in value)
        finally:
            active.remove(identity)

    pickle_bytes = 3 + visit(tree, 0)  # PROTO 2 + STOP
    return PickleStorageBound(
        pickle_bytes=pickle_bytes,
        storage_nbytes=tuple(storage_nbytes),
        tensor_count=len(storage_nbytes),
        node_count=node_count,
        memo_slot_upper=memo_slots,
        memoized_reference_count=references,
        unique_string_count=unique_strings,
        max_depth_seen=max_depth_seen,
    )
