"""Bounded symbolic recipe helpers; not scientific objects or run authority.

Only explicit materialize() imports Torch/allocates tensors. All other helpers
work on ordinary containers and a nonserializable TensorSlot marker. No files,
RNG sampling, producer calls or mutations of imported calculators occur here.
"""
from __future__ import annotations

import math
import os

PROFILE = "diagnostic_mnist_current32_v1"
MAX_NODES = 200_000
MAX_DEPTH = 32
MAX_STRING_BYTES = 4 << 20
MAX_TOTAL_STRING_BYTES = 8 << 20


class RecipeError(ValueError):
    pass


class TensorSlot:
    """A symbolic position only. Never pass this object to a serializer."""
    __slots__ = ()


SLOT = TensorSlot()


def _need(condition, message):
    if not condition:
        raise RecipeError(message)


def _transform(value, scalar, sequence, mapping):
    """Bounded exact-type traversal; repeated containers copied per occurrence."""
    budget = [MAX_NODES, MAX_TOTAL_STRING_BYTES]
    active = set()
    def text_bound(item):
        _need(len(item) <= MAX_STRING_BYTES, "recipe string too long")
        _need(not any(0xD800 <= ord(char) <= 0xDFFF for char in item), "recipe string surrogate")
        encoded = item.encode("utf-8")
        budget[1] -= len(encoded)
        _need(len(encoded) <= MAX_STRING_BYTES and budget[1] >= 0,
              "recipe string budget exceeded")
    def visit(item, path):
        budget[0] -= 1
        _need(len(path) <= MAX_DEPTH and budget[0] >= 0, "recipe traversal budget exceeded")
        kind = type(item)
        if kind in (dict, list, tuple):
            _need(id(item) not in active, "cyclic recipe")
            _need(len(item) <= MAX_NODES, "recipe container too large")
            active.add(id(item))
            try:
                if kind is dict:
                    children = {}
                    for key, child in item.items():
                        _need(type(key) in (str, int), "recipe key must be exact str/int")
                        if type(key) is str:
                            text_bound(key)
                        else:
                            _need(-(1 << 64) <= key < 1 << 64, "recipe key integer too large")
                        children[key] = visit(child, path + (key,))
                    return mapping(children)
                return sequence(kind, [visit(child, path + (index,))
                                       for index, child in enumerate(item)])
            finally:
                active.remove(id(item))
        if kind is str:
            text_bound(item)
        elif kind is int:
            _need(-(1 << 64) <= item < 1 << 64, "recipe integer too large")
        elif kind is float:
            _need(math.isfinite(item), "nonfinite recipe scalar")
        else:
            _need(item is None or kind in (bool, TensorSlot), "unsupported recipe leaf")
        return scalar(item, path)
    return visit(value, ())


def copy_primitive(value):
    """Own every container occurrence, retaining immutable values/slot marker."""
    return _transform(value, lambda item, _path: SLOT if type(item) is TensorSlot else item,
                      lambda kind, children: kind(children), dict)


def primitive_cost(template):
    """Prospective primitive pickle cost only; excludes tensors and PROTO/STOP."""
    import primitive_storage_bound as primitive
    return _transform(template,
        lambda item, _path: 0 if type(item) is TensorSlot else primitive.literal(item),
        lambda kind, children: primitive.plist(children) if kind is list else primitive.ptuple(children),
        primitive.pdict)


def slot_paths(template):
    """Return actual structural traversal order, not ZIP storage encounter proof."""
    result = []
    def scalar(item, path):
        if type(item) is TensorSlot:
            result.append(path)
        return None
    _transform(template, scalar, lambda _kind, _children: None, lambda _children: None)
    return tuple(result)


def validate_template(component, template):
    """Bind symbolic positions and primitive ceiling to the closed inventory.

    Completeness of non-tensor key/container shapes is an additional reviewed
    builder obligation, not something a primitive byte ceiling proves.
    """
    import native_tensor_inventory as inventory
    import native_storage_topology_bound as topology
    _need(type(component) is str and component in inventory.COMPONENT_COUNTS,
          "unknown recipe component")
    _need(type(template) is dict and template.get("profile") == PROFILE,
          "recipe must have unregistered diagnostic root profile")
    rows = inventory.component_layouts()[component]
    paths = slot_paths(template)
    _need(len(paths) == len(rows) and set(paths) == {row["path"] for row in rows},
          "recipe tensor-slot inventory differs")
    bound = topology.compute()["components"][component]
    primitive = primitive_cost(template)
    if bound["encoding"] == "torch_weights_only":
        _need(primitive <= bound["primitive_subtree_bytes_upper"],
              "recipe primitive cost exceeds independent bound")
    return {"component": component, "slot_count": len(paths),
            "raw_storage_bytes": sum(row["nbytes"] for row in rows),
            "primitive_pickle_bytes_upper": primitive,
            "body_bytes_upper": bound["body_bytes_upper"],
            "scientific_artifact": False, "execution_authorized": False}


def materialize(component, template):
    """Explicit full-width CPU allocation boundary, NOT enabled by default CLI.

    Requires a separately reviewed external resource/consumption supervisor.
    Callers must not invoke this on real components during symbolic unit tests.
    Returns owned zero storages; values are nonsemantic, not sampled RNG/state.
    """
    import native_tensor_inventory as inventory
    validate_template(component, template)
    rows = {row["path"]: row for row in inventory.component_layouts()[component]}
    if not rows:
        return copy_primitive(template)
    _need(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "materialization must hide CUDA")
    import torch
    _need(not torch.cuda.is_initialized(), "materialization requires uninitialized CUDA")
    def scalar(item, path):
        if type(item) is not TensorSlot:
            return item
        row = rows[path]
        tensor = torch.zeros(row["shape"], dtype=getattr(torch, row["dtype"]), device="cpu")
        stride = tuple(math.prod(row["shape"][index + 1:]) for index in range(len(row["shape"])))
        _need(tensor.is_contiguous() and tensor.storage_offset() == 0
              and tuple(tensor.stride()) == stride
              and tensor.untyped_storage().nbytes() == row["nbytes"]
              and not tensor.requires_grad, "materialized storage differs")
        return tensor
    result = _transform(template, scalar, lambda kind, children: kind(children), dict)
    _need(not torch.cuda.is_initialized(), "materialization changed CUDA state")
    return result


def main(argv=None):
    print("native_storage_recipe_common: inert; no recipe allocation or execution")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
