"""Fixed I7 plan and pinned training-IDX adapters; import/default CLI are inert.

Explicit calls perform work. They grant no GO authority: the native controller
must authorize the fixed operation, supply its cumulative resource checkpoint,
and persist the plan once through ArtifactStore before using it. No discovery,
download, official-test loading, global seeding or device selection is provided.
"""
from __future__ import annotations

import fcntl
import hashlib
import os
import re
import stat
import struct


SCIENTIFIC = "scientific_mnist_current32_v1"
MLP_FIXTURE = "fixture_tiny_mlp_cpu_v1"
FILE_KEYS = ("training_images", "training_labels")
FILE_NAMES = ("train-images-idx3-ubyte", "train-labels-idx1-ubyte")
_IDX_SHAPES = {SCIENTIFIC: (60000, 28, 28), MLP_FIXTURE: (30, 1, 3)}
_SHA = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)
_DIRECTORY_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
_FILE_FLAGS = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC
_CHUNK = 1 << 20


class InputError(ValueError):
    """Closed, bounded input validation/read failure; no automatic retry."""


def _need(condition, message):
    if not condition:
        raise InputError(message)


def _check_profile(profile):
    _need(type(profile) is str and profile in _IDX_SHAPES, "unsupported input profile")


def _checkpoint(checkpoint, stage):
    _need(callable(checkpoint), "cumulative resource checkpoint required")
    checkpoint(stage)


def _rng(bundle, stream):
    import numpy as np
    return np.random.default_rng(np.random.SeedSequence([20260906, bundle, stream]))


def _draw_component(field, *, identity, profile):
    """One fixed private stream. Scientific tests replace _rng before any draw."""
    import numpy as np
    import identity_codec as codec
    import plan_bindings as plans
    _check_profile(profile)
    codec.validate_identity(identity, profile=profile)
    total, count, classes, batch, probe = plans.DIMENSIONS[profile]
    bundle = identity["bundle"]
    if field == "permutation":
        return _rng(bundle, 0).permutation(total)
    if field == "replacement_uniforms":
        return _rng(bundle, 1).random(count)
    if field == "replacement_digits":
        return _rng(bundle, 2).integers(0, classes, size=count)
    if field == "initialization_seed":
        return int(_rng(bundle, 3).integers(0, 2**32, dtype=np.uint32))
    if field == "training_batches":
        return _rng(bundle, 4).integers(0, count, size=(identity["steps_total"], batch))
    if field == "training_probe_indices":
        return _rng(bundle, 5).integers(0, count, size=probe)
    raise InputError("unknown frozen plan component")


def generate_plan(*, identity, profile, checkpoint):
    """Generate only this validated member's frozen plan, using six local streams.

Scientific invocation requires the upstream scoped GO. The registered MLP
fixture uses the same algorithm at its existing dimensions and bundle zero.
No optional seed, dimensions, probe distribution or generator override exists.
"""
    _check_profile(profile)
    _checkpoint(checkpoint, "adapter.plan.import.pre")
    import torch
    import identity_codec as codec
    import plan_bindings as plans
    codec.validate_identity(identity, profile=profile)
    _, count, _, _, _ = plans.DIMENSIONS[profile]

    def draw(field, dtype=None):
        _checkpoint(checkpoint, "adapter.plan."+field+".pre")
        value = _draw_component(field, identity=identity, profile=profile)
        if dtype is not None:
            value = torch.tensor(value, dtype=dtype, device="cpu")
        _checkpoint(checkpoint, "adapter.plan."+field+".post")
        return value

    permutation = draw("permutation", torch.int64)
    plan = dict(schema=plans.SCHEMAS[profile], profile=profile,
        bundle=identity["bundle"], steps_total=identity["steps_total"],
        stream_roles=dict(identity["stream_roles"]), permutation=permutation,
        train_indices=permutation[:count].clone(),
        validation_indices=permutation[count:2*count].clone(),
        auxiliary_indices=permutation[2*count:3*count].clone(),
        replacement_uniforms=draw("replacement_uniforms", torch.float64),
        replacement_digits=draw("replacement_digits", torch.int64),
        initialization_seed=draw("initialization_seed"),
        training_batches=draw("training_batches", torch.int64),
        training_probe_indices=draw("training_probe_indices", torch.int64))
    plans.validate_plan(plan, identity=identity, profile=profile)
    _checkpoint(checkpoint, "adapter.plan.final")
    return plan


def _signature(info):
    return (info.st_dev, info.st_ino, info.st_mode, info.st_nlink, info.st_size,
            info.st_mtime_ns, info.st_ctime_ns)


def _open_directory(directory):
    path = os.fspath(directory)
    _need(type(path) is str and path.isascii() and len(path) <= 4096 and path.startswith("/")
          and path != "/" and all(part not in ("", ".", "..") for part in path.split("/")[1:]),
          "IDX directory must be canonical absolute ASCII path")
    fd = os.open("/", _DIRECTORY_FLAGS)
    try:
        for part in path.split("/")[1:]:
            next_fd = os.open(part, _DIRECTORY_FLAGS, dir_fd=fd)
            os.close(fd)
            fd = next_fd
        info = os.fstat(fd)
        return path, fd, (info.st_dev, info.st_ino)
    except BaseException:
        os.close(fd)
        raise


def _pins(expected_files, profile):
    _check_profile(profile)
    _need(type(expected_files) is dict and tuple(expected_files) == FILE_KEYS,
          "exact training image/label pins required")
    count, rows, columns = _IDX_SHAPES[profile]
    result = {}
    for key, size in zip(FILE_KEYS, (16+count*rows*columns, 8+count)):
        pin = expected_files[key]
        _need(type(pin) is dict and tuple(pin) == ("sha256", "size_bytes"), "exact IDX pin keys required")
        _need(type(pin["sha256"]) is str and _SHA.fullmatch(pin["sha256"]) is not None,
              "invalid IDX SHA-256")
        _need(type(pin["size_bytes"]) is int and pin["size_bytes"] == size,
              "IDX size differs from fixed profile")
        result[key] = dict(pin)
    return result


def read_training_idx(*, directory, expected_files, profile, checkpoint):
    """Read exactly two caller-pinned training files, bounded by profile size.

Every path segment is opened without following symlinks. Nonblocking file opens
prevent a substituted FIFO from hanging before regular-file checks. Held fds,
path identities and exact hashes are rechecked; this is observed consistency,
not a claim of a hostile-same-user filesystem snapshot. Returned bytes are the
verified buffers: materialize them directly, never reopen through another path.
"""
    expected = _pins(expected_files, profile)
    _checkpoint(checkpoint, "adapter.idx.open.pre")
    rootfd = None
    opened = []
    try:
        path, rootfd, root_identity = _open_directory(directory)
        # Authenticate both descriptors before reading either payload.
        for key, name in zip(FILE_KEYS, FILE_NAMES):
            _checkpoint(checkpoint, "adapter.idx.open."+key)
            fd = os.open(name, _FILE_FLAGS, dir_fd=rootfd)
            opened.append((key, name, fd, None))
            info = os.fstat(fd)
            _need(stat.S_ISREG(info.st_mode) and info.st_nlink == 1,
                  "IDX input must be a singly linked regular file")
            _need(info.st_size == expected[key]["size_bytes"], "actual IDX size differs from pin")
            fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
            opened[-1] = (key, name, fd, info)
        buffers = {}
        for key, name, fd, info in opened:
            digest = hashlib.sha256()
            body = bytearray()
            remaining = info.st_size
            while remaining:
                _checkpoint(checkpoint, "adapter.idx.read."+key)
                chunk = os.read(fd, min(remaining, _CHUNK))
                _need(bool(chunk), "truncated IDX read")
                digest.update(chunk)
                body.extend(chunk)
                remaining -= len(chunk)
            _need(os.read(fd, 1) == b"", "IDX grew during read")
            _need(digest.hexdigest() == expected[key]["sha256"], "actual IDX SHA-256 differs from pin")
            buffers[key] = bytes(body)
            del body
            _checkpoint(checkpoint, "adapter.idx.verified."+key)
        for key, name, fd, info in opened:
            _need(_signature(os.fstat(fd)) == _signature(info)
                  and _signature(os.stat(name, dir_fd=rootfd, follow_symlinks=False)) == _signature(info),
                  "IDX file changed during read")
        _, finalfd, final_identity = _open_directory(path)
        try:
            _need(final_identity == root_identity, "IDX directory changed during read")
        finally:
            os.close(finalfd)
        count, rows, columns = _IDX_SHAPES[profile]
        _need(struct.unpack_from(">IIII", buffers["training_images"]) == (2051, count, rows, columns)
              and struct.unpack_from(">II", buffers["training_labels"]) == (2049, count),
              "IDX header differs from fixed training profile")
        _checkpoint(checkpoint, "adapter.idx.final")
        return dict(images_bytes=buffers["training_images"], labels_bytes=buffers["training_labels"],
                    expected_files=expected)
    except OSError:
        raise InputError("IDX descriptor/path operation failed") from None
    finally:
        for _, _, fd, _ in reversed(opened):
            os.close(fd)
        if rootfd is not None:
            os.close(rootfd)


if __name__ == "__main__":
    print("native_inputs: inert; no plan generation, data read or execution action")
