"""Pure representation conversion; never query or initialize a CUDA device.

Pinned Torch exposes properties.uuid as torch._C._CUuuid, whose string is bare
8-4-4-4-12 hexadecimal. NVIDIA's query uses GPU- plus that text. Both retained
environment and RNG-core identities use the canonical GPU-prefixed form here.
"""
from __future__ import annotations

import re


_UUID_BODY = re.compile(
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\Z", re.ASCII)


def canonical_cuda_uuid(value) -> str:
    """Normalize supported observed UUID representations, not arbitrary objects."""
    if type(value) is str:
        text = value
    elif type(value) is bytes:
        if len(value) not in (36, 40):
            raise ValueError("CUDA UUID has invalid encoded length")
        try:
            text = value.decode("ascii")
        except UnicodeDecodeError:
            raise ValueError("CUDA UUID bytes must be ASCII") from None
    else:
        # Reading the already loaded extension's type does not query a device.
        # No caller-supplied duck type or arbitrary __str__ is accepted.
        import torch
        native_type = getattr(torch._C, "_CUuuid", None)
        if native_type is None or type(value) is not native_type:
            raise ValueError("unsupported CUDA UUID representation")
        text = str(value)
    if len(text) not in (36, 40) or not text.isascii():
        raise ValueError("CUDA UUID has invalid text length/encoding")
    body = text[4:] if len(text) == 40 and text[:4].upper() == "GPU-" else text
    if _UUID_BODY.fullmatch(body) is None:
        raise ValueError("CUDA UUID must be a complete 128-bit GPU identifier")
    return "GPU-" + body.lower()


def is_canonical_cuda_uuid(value) -> bool:
    """Persisted identifiers must already be canonical exact built-in strings."""
    if type(value) is not str:
        return False
    try:
        return canonical_cuda_uuid(value) == value
    except ValueError:
        return False
