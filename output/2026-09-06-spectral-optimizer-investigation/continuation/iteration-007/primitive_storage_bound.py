"""Torch-free arithmetic for prospective protocol2 primitive subtrees.

Costs exclude the root PROTO/STOP three bytes and every Tensor reduction/raw
storage. No object alias saving is used. A cost is not validation of an actual
payload: constructors/domains, runtime checks, ZIP and global accounting remain
separate proof obligations. Import and default CLI perform no filesystem work.
"""
from __future__ import annotations

import math

from pickle_storage_bound import protocol2_int_bytes


FLOAT, BOOL, NULL = 9, 1, 1


def _natural(value: object, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(label + " must be a nonnegative exact integer")
    return value


def text(max_utf8_bytes: int) -> int:
    """BINUNICODE header/content and worst five-byte memo PUT."""
    length = _natural(max_utf8_bytes, "UTF8 ceiling")
    if length > 0xFFFFFFFF:
        raise ValueError("UTF8 ceiling leaves protocol2 BINUNICODE domain")
    return 10 + length


def integer(low: int, high: int) -> int:
    """Maximum opcode width over a closed integer interval.

    Protocol2 width is nondecreasing with magnitude on each side of zero;
    interval endpoints therefore dominate. There is no value memo for ints.
    """
    if type(low) is not int or type(high) is not int or low > high:
        raise ValueError("integer interval must have exact ordered endpoints")
    return max(protocol2_int_bytes(low), protocol2_int_bytes(high))


def _costs(values) -> tuple[int, ...]:
    if type(values) not in (list, tuple):
        raise ValueError("cost sequence must be an exact list or tuple")
    return tuple(_natural(value, "child cost") for value in values)


def _list_frame(count: int) -> int:
    # Exact C list path: EMPTY_LIST + PUT, then APPEND or batched MARK/APPENDS.
    return 6 + (0 if count == 0 else 1 if count == 1 else 2 * ((count + 999) // 1000))


def plist(child_costs) -> int:
    values = _costs(child_costs)
    return _list_frame(len(values)) + sum(values)


def repeat_list(count: int, child_cost: int) -> int:
    count = _natural(count, "repeat count")
    cost = _natural(child_cost, "child cost")
    return _list_frame(count) + count * cost


def ptuple(child_costs) -> int:
    values = _costs(child_costs)
    if not values:
        return 1
    return (6 if len(values) <= 3 else 7) + sum(values)


def pdict(field_costs: dict) -> int:
    if type(field_costs) is not dict:
        raise ValueError("field costs must be an exact dict")
    count = len(field_costs)
    # Exact C dict path emits a final empty batch at positive multiples1000.
    frame = 6 + (0 if count == 0 else 1 if count == 1 else 2 * (count // 1000 + 1))
    total = frame
    for key, value in field_costs.items():
        if type(key) not in (str, int):
            raise ValueError("field key must be an exact string or integer")
        total += literal(key) + _natural(value, "field cost")
    return total


def literal(value) -> int:
    """Cost of supplied fixed primitive constants; never serialize or load them.

    Call only on small static schema constants. Runtime untrusted trees need
    independent bounded validation; this helper is deliberately not a decoder.
    """
    kind = type(value)
    if value is None:
        return NULL
    if kind is bool:
        return BOOL
    if kind is int:
        return protocol2_int_bytes(value)
    if kind is float and math.isfinite(value):
        return FLOAT
    if kind is str:
        return text(len(value.encode("utf-8", "surrogatepass")))
    if kind is dict:
        return pdict({key: literal(item) for key, item in value.items()})
    if kind is list:
        return plist([literal(item) for item in value])
    if kind is tuple:
        return ptuple([literal(item) for item in value])
    raise ValueError("unsupported fixed primitive literal")


def json_subtree_bytes(max_canonical_json_bytes: int) -> int:
    """Conservative no-alias subtree cost <=5*compact ASCII JSON length.

    Requires an exact-type acyclic primitive JSON tree with str/int dict keys
    (never bool), finite floats and successful unmodified canonical encoding.
    The actual encoding must fit the supplied cap. A cap including its trailing
    newline is also conservative. Not for tensors, custom objects or raw JSON
    text that has not been schema-validated. Root PROTO/STOP remain separate.
    """
    cap = _natural(max_canonical_json_bytes, "canonical JSON ceiling")
    if cap < 1:
        raise ValueError("canonical JSON ceiling must be positive")
    return 5 * cap


if __name__ == "__main__":
    print("I7 primitive bound arithmetic only; no payload, runtime or execution admission.")
