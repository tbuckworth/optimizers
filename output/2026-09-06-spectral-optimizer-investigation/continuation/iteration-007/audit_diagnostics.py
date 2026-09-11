"""Strict bounded diagnostics for independent I7 numerical audits.

The coordinate codec is lossless: bit ``i % 8`` of byte ``i // 8`` records
flat coordinate ``i``.  The byte string is always wide enough for the declared
dimension, including when no coordinates fail.  This module performs no I/O,
array conversion, numerical audit, or tolerance selection.
"""

from __future__ import annotations

import math
import re
from typing import Any


ENCODING = "flat_bitset_hex_lsb0_v1"
MAX_DIMENSION = 50_890
INDEX_RECORD_KEYS = ("encoding", "dimension", "count", "bits_hex")
_LOWER_HEX = re.compile(r"[0-9a-f]*\Z", re.ASCII)


class AuditDiagnosticError(ValueError):
    """A diagnostic record or primitive report tree is malformed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditDiagnosticError(message)


def _dimension(value: Any) -> int:
    _require(type(value) is int and 1 <= value <= MAX_DIMENSION,
             f"dimension must be an exact int in [1,{MAX_DIMENSION}]")
    return value


def pack_indices(indices: Any, dimension: Any) -> dict[str, Any]:
    """Pack one canonical increasing list of flat indices into fixed-width hex."""
    size = _dimension(dimension)
    _require(type(indices) is list, "indices must be an exact list")
    bits = bytearray((size + 7) // 8)
    previous = -1
    for position, index in enumerate(indices):
        _require(type(index) is int, f"indices[{position}] must be an exact int")
        _require(previous < index < size,
                 f"indices[{position}] is out of range, repeated, or non-increasing")
        bits[index // 8] |= 1 << (index % 8)
        previous = index
    return {
        "encoding": ENCODING,
        "dimension": size,
        "count": len(indices),
        "bits_hex": bytes(bits).hex(),
    }


def unpack_indices(record: Any) -> list[int]:
    """Validate and decode one canonical fixed-width flat-coordinate record."""
    _require(type(record) is dict and tuple(record) == INDEX_RECORD_KEYS and
             all(type(key) is str for key in record),
             "index record requires exact ordered fields")
    _require(type(record["encoding"]) is str and record["encoding"] == ENCODING,
             "unsupported index encoding")
    size = _dimension(record["dimension"])
    count = record["count"]
    _require(type(count) is int and 0 <= count <= size, "count is invalid")
    encoded = record["bits_hex"]
    expected_chars = 2 * ((size + 7) // 8)
    _require(type(encoded) is str and len(encoded) == expected_chars and
             _LOWER_HEX.fullmatch(encoded) is not None,
             "bits_hex must be canonical lowercase fixed-width hex")
    try:
        bits = bytes.fromhex(encoded)
    except ValueError as error:  # Defensive after the canonical regular expression.
        raise AuditDiagnosticError("bits_hex is malformed") from error

    remainder = size % 8
    if remainder:
        _require(bits[-1] & ~((1 << remainder) - 1) == 0,
                 "bits_hex has nonzero padding bits")
    _require(sum(byte.bit_count() for byte in bits) == count,
             "count differs from encoded coordinate membership")
    return [index for index in range(size)
            if bits[index // 8] & (1 << (index % 8))]


def primitive_tree(value: Any, *, max_nodes: Any, max_depth: Any,
                   max_string_bytes: Any) -> Any:
    """Validate a bounded JSON-primitive tree and return it unchanged.

    Root depth is zero.  The node budget counts containers, values, and mapping
    keys; the string-byte budget is the aggregate UTF-8 size of keys and string
    values.  Only exact ``dict``/``list`` containers, exact string keys, and
    ``None``/``bool``/``int``/finite ``float``/``str`` leaves are supported.
    Cycles are rejected.  Shared aliases are traversed at every occurrence, so
    their node and string costs match their expanded serialized form.  Nothing
    is truncated, converted, reordered, or copied.
    """
    _require(type(max_nodes) is int and max_nodes > 0,
             "max_nodes must be a positive exact int")
    _require(type(max_depth) is int and max_depth >= 0,
             "max_depth must be a nonnegative exact int")
    _require(type(max_string_bytes) is int and max_string_bytes >= 0,
             "max_string_bytes must be a nonnegative exact int")

    nodes = 0
    string_bytes = 0
    active_containers: set[int] = set()
    stack: list[tuple[Any, int, bool, bool]] = [(value, 0, False, False)]
    while stack:
        item, depth, mapping_key, exiting = stack.pop()
        if exiting:
            active_containers.remove(id(item))
            continue
        _require(depth <= max_depth, "primitive tree exceeds max_depth")
        nodes += 1
        _require(nodes <= max_nodes, "primitive tree exceeds max_nodes")

        if mapping_key:
            _require(type(item) is str, "primitive tree mapping keys must be exact strings")
        item_type = type(item)
        if item_type is str:
            try:
                string_bytes += len(item.encode("utf-8"))
            except UnicodeEncodeError as error:
                raise AuditDiagnosticError("primitive tree string is not valid Unicode") from error
            _require(string_bytes <= max_string_bytes,
                     "primitive tree exceeds max_string_bytes")
            continue
        if mapping_key:
            raise AuditDiagnosticError("primitive tree mapping keys must be exact strings")
        if item is None or item_type in (bool, int):
            continue
        if item_type is float:
            _require(math.isfinite(item), "primitive tree contains a nonfinite float")
            continue
        _require(item_type in (list, dict), "primitive tree contains an unsupported value")
        identity = id(item)
        _require(identity not in active_containers, "primitive tree contains a cycle")
        active_containers.add(identity)
        stack.append((item, depth, False, True))
        if item_type is list:
            for child in reversed(item):
                stack.append((child, depth + 1, False, False))
        else:
            for key in reversed(item):
                stack.append((item[key], depth + 1, False, False))
                stack.append((key, depth + 1, True, False))
    return value
