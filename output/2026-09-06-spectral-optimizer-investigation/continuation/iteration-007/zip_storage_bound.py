"""Pure, inert accounting for the admitted ``torch.save`` ZIP envelope.

Importing this module and invoking its default CLI do not import Torch.  The
explicit runtime validator imports Torch on CPU and only admits the pinned
source/configuration; it does not serialize an object or attest the compiled
writer binary's provenance.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import stat
import sys
from typing import Final


SCHEMA: Final = "i7_torch_save_zip_bound_v1"
RUNTIME_SCHEMA: Final = "i7_torch_save_zip_runtime_v1"
TORCH_VERSION: Final = "2.11.0+cu128"
TORCH_GIT_REVISION: Final = "70d99e998b4955e0049d13a98d77ae1b14db1f45"

BUFFER_LIMIT_BYTES: Final = 64 << 20
ZIP32_LIMIT: Final = 0xFFFFFFFF
ALIGNMENT_BYTES: Final = 64
ARCHIVE_PREFIX: Final = "archive/"

LOCAL_HEADER_BYTES: Final = 30
CENTRAL_HEADER_BYTES: Final = 46
FB_EXTRA_HEADER_BYTES: Final = 4
DATA_DESCRIPTOR32_BYTES: Final = 16
ZIP64_EOCD_BYTES: Final = 56
ZIP64_LOCATOR_BYTES: Final = 20
EOCD_BYTES: Final = 22
ZIP64_END_BYTES: Final = ZIP64_EOCD_BYTES + ZIP64_LOCATOR_BYTES + EOCD_BYTES

# These payload lengths follow from the pinned serializer/writer sources.
FIXED_RECORDS: Final = (
    (".format_version", 1),
    (".storage_alignment", 2),
    ("byteorder", 6),
)
FINAL_RECORDS: Final = (
    ("version", 2),
    (".data/serialization_id", 40),
)
FIXED_RECORD_PAYLOAD_BYTES: Final = 51

INSTALLED_SOURCE_HASHES: Final = {
    "serialization.py": "ce5bc5ca6a8faa2b5aee2c72ff1f1d4b02c2a2d82e96886de74ae9fc436a7f67",
    "utils/serialization/config.py": "d353610111287c3aade11046d0d4b64d1a1d297504c91c29998c347670442dd9",
    "include/caffe2/serialize/inline_container.h": "647b2691ccf54900397e2b7f53a8831240aba0645a42aba0d1358e3a3fa3b4c6",
    "include/caffe2/serialize/versions.h": "9e737340277e1185704307cb7e08647be8833395f614a21a552f70c9e266a4a4",
}
UPSTREAM_SOURCE_HASHES: Final = {
    "caffe2/serialize/inline_container.cc": "aa3320b4f3e905dc80cb6f9ac0a58d42941d0f74b00c4292783660361c08392e",
    "third_party/miniz-3.0.2/miniz.c": "88d42a713e8ef5396f5bbaab3d9150f9e98478cfd4d71b02334971243515e76d",
    "third_party/miniz-3.0.2/miniz.h": "f959f5dfb5c5d3ed0f55f3e7e455afbe1e924d64d74cd2dd374740b9d87abfd0",
}

BOUND_FIELDS: Final = (
    "schema",
    "archive_prefix",
    "alignment_bytes",
    "pickle_bytes_upper",
    "storage_nbytes_upper",
    "storage_count",
    "record_count",
    "record_payload_bytes_upper",
    "local_header_bytes",
    "local_filename_bytes",
    "fb_extra_header_bytes",
    "alignment_padding_bytes",
    "data_descriptor_bytes_upper",
    "per_entry_zip64_extra_bytes",
    "local_and_data_bytes_upper",
    "central_directory_bytes",
    "zip64_end_bytes",
    "archive_bytes_upper",
    "buffer_limit_bytes",
    "compression_assumed",
    "within_buffer_limit",
)

RUNTIME_FIELDS: Final = (
    "schema",
    "torch_version",
    "torch_git_revision",
    "pickle_protocol",
    "new_zipfile_serialization_required",
    "byteorder_record_required",
    "byteorder",
    "compute_crc32",
    "storage_alignment_bytes",
    "use_pinned_memory_for_d2h",
    "skip_data",
    "archive_prefix",
    "package_registry_verified",
    "save_call_defaults_verified",
    "installed_source_hashes_verified",
    "upstream_source_hashes_bound",
    "require_cuda_uninitialized",
    "cuda_initialized",
    "cuda_initialization_unchanged",
    "source_revision_and_configuration_admitted",
    "compiled_binary_provenance_attested",
)


class ZipBoundError(RuntimeError):
    """A closed ZIP-domain or runtime-admission failure."""


def _integer(value: object, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ZipBoundError(f"{label} must be an integer >= {minimum}")
    return value


def _keys(value: object, fields: tuple[str, ...], label: str) -> dict:
    if type(value) is not dict or tuple(value) != fields:
        raise ZipBoundError(f"{label} has invalid fields")
    return value


def _align_up(value: int, alignment: int = ALIGNMENT_BYTES) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def zip64_extra_bytes(record_size: int, local_header_cursor: int) -> int:
    """Return miniz's ZIP64 extra-field length at the two ZIP32 thresholds.

    This helper documents the excluded branches.  The admitted 64-MiB domain
    always returns zero; the main bound refuses to extrapolate into these cases.
    """

    size = _integer(record_size, "record_size")
    cursor = _integer(local_header_cursor, "local_header_cursor")
    size_large = size >= ZIP32_LIMIT
    cursor_large = cursor >= ZIP32_LIMIT
    if not size_large and not cursor_large:
        return 0
    return 4 + (16 if size_large else 0) + (8 if cursor_large else 0)


def _record_rows(pickle_bytes: int, storages: tuple[int, ...]) -> tuple[tuple[str, int], ...]:
    return (
        (("data.pkl", pickle_bytes),)
        + FIXED_RECORDS
        + tuple((f"data/{index}", size) for index, size in enumerate(storages))
        + FINAL_RECORDS
    )


def _compute(
    pickle_bytes_upper: int,
    storage_nbytes_upper: tuple[int, ...],
    buffer_limit_bytes: int,
) -> dict:
    pickle_size = _integer(pickle_bytes_upper, "pickle_bytes_upper", minimum=1)
    if type(storage_nbytes_upper) is not tuple:
        raise ZipBoundError("storage_nbytes_upper must be a tuple")
    if len(storage_nbytes_upper) + 6 >= 65535:
        raise ZipBoundError("record count leaves the admitted ZIP32 domain")
    storages = tuple(
        _integer(value, f"storage_nbytes_upper[{index}]")
        for index, value in enumerate(storage_nbytes_upper)
    )
    limit = _integer(buffer_limit_bytes, "buffer_limit_bytes", minimum=1)
    if limit > BUFFER_LIMIT_BYTES:
        raise ZipBoundError("buffer_limit_bytes exceeds the admitted 64-MiB cap")

    rows = _record_rows(pickle_size, storages)

    cursor = 0
    filename_bytes = 0
    padding_bytes = 0
    for name, size in rows:
        if size >= BUFFER_LIMIT_BYTES:
            raise ZipBoundError("record size leaves the admitted 64-MiB domain")
        full_name = (ARCHIVE_PREFIX + name).encode("ascii")
        name_size = len(full_name)
        if name_size > 65535:
            raise ZipBoundError("record filename leaves the ZIP32 domain")
        if cursor >= BUFFER_LIMIT_BYTES:
            raise ZipBoundError("record cursor leaves the admitted 64-MiB domain")
        if zip64_extra_bytes(size, cursor) != 0:
            raise ZipBoundError("per-entry ZIP64 extras are outside the admitted domain")

        unaligned_data = cursor + LOCAL_HEADER_BYTES + name_size + FB_EXTRA_HEADER_BYTES
        data_start = _align_up(unaligned_data)
        padding_bytes += data_start - unaligned_data
        filename_bytes += name_size
        # Charging a 16-byte descriptor for every record is conservative.  The
        # pinned miniz omits it for zero-sized add_mem records.
        cursor = data_start + size + DATA_DESCRIPTOR32_BYTES
        if cursor >= BUFFER_LIMIT_BYTES:
            raise ZipBoundError("local records leave the admitted 64-MiB domain")

    record_count = len(rows)
    central_bytes = sum(CENTRAL_HEADER_BYTES + len((ARCHIVE_PREFIX + name).encode("ascii")) for name, _ in rows)
    if central_bytes > BUFFER_LIMIT_BYTES:
        raise ZipBoundError("central directory leaves the admitted 64-MiB domain")
    archive_bytes = cursor + central_bytes + ZIP64_END_BYTES
    if archive_bytes >= BUFFER_LIMIT_BYTES:
        raise ZipBoundError("ZIP upper bound leaves the admitted 64-MiB domain")
    if archive_bytes > limit:
        raise ZipBoundError("ZIP upper bound exceeds the supplied buffer limit")

    return {
        "schema": SCHEMA,
        "archive_prefix": ARCHIVE_PREFIX,
        "alignment_bytes": ALIGNMENT_BYTES,
        "pickle_bytes_upper": pickle_size,
        "storage_nbytes_upper": list(storages),
        "storage_count": len(storages),
        "record_count": record_count,
        "record_payload_bytes_upper": pickle_size + sum(storages) + FIXED_RECORD_PAYLOAD_BYTES,
        "local_header_bytes": LOCAL_HEADER_BYTES * record_count,
        "local_filename_bytes": filename_bytes,
        "fb_extra_header_bytes": FB_EXTRA_HEADER_BYTES * record_count,
        "alignment_padding_bytes": padding_bytes,
        "data_descriptor_bytes_upper": DATA_DESCRIPTOR32_BYTES * record_count,
        "per_entry_zip64_extra_bytes": 0,
        "local_and_data_bytes_upper": cursor,
        "central_directory_bytes": central_bytes,
        "zip64_end_bytes": ZIP64_END_BYTES,
        "archive_bytes_upper": archive_bytes,
        "buffer_limit_bytes": limit,
        "compression_assumed": False,
        "within_buffer_limit": True,
    }


def torch_save_zip_ceiling(
    pickle_bytes_upper: int,
    storage_nbytes_upper: tuple[int, ...],
    *,
    buffer_limit_bytes: int = BUFFER_LIMIT_BYTES,
) -> dict:
    """Return the closed conservative ZIP envelope for supplied payload maxima.

    Storage entries are unique storage records in exact first-encounter order.
    Their count and decimal indices affect filename overhead, so callers must
    supply every record (including both retained-core storage records).
    """

    result = _compute(pickle_bytes_upper, storage_nbytes_upper, buffer_limit_bytes)
    return validate_bound(result)


def validate_bound(value: object) -> dict:
    """Strictly decode and independently recompute a bound record."""

    row = _keys(value, BOUND_FIELDS, "ZIP bound")
    if row["schema"] != SCHEMA:
        raise ZipBoundError("ZIP bound schema mismatch")
    if type(row["archive_prefix"]) is not str or row["archive_prefix"] != ARCHIVE_PREFIX:
        raise ZipBoundError("archive prefix mismatch")
    if type(row["storage_nbytes_upper"]) is not list:
        raise ZipBoundError("storage_nbytes_upper must be a list")
    storages = tuple(
        _integer(value, f"storage_nbytes_upper[{index}]")
        for index, value in enumerate(row["storage_nbytes_upper"])
    )
    integer_fields = (
        "alignment_bytes",
        "pickle_bytes_upper",
        "storage_count",
        "record_count",
        "record_payload_bytes_upper",
        "local_header_bytes",
        "local_filename_bytes",
        "fb_extra_header_bytes",
        "alignment_padding_bytes",
        "data_descriptor_bytes_upper",
        "per_entry_zip64_extra_bytes",
        "local_and_data_bytes_upper",
        "central_directory_bytes",
        "zip64_end_bytes",
        "archive_bytes_upper",
        "buffer_limit_bytes",
    )
    for field in integer_fields:
        _integer(row[field], field)
    expected = _compute(
        _integer(row["pickle_bytes_upper"], "pickle_bytes_upper", minimum=1),
        storages,
        _integer(row["buffer_limit_bytes"], "buffer_limit_bytes", minimum=1),
    )
    if row != expected:
        raise ZipBoundError("ZIP bound arithmetic mismatch")
    # Dict equality conflates bool and int; require all fixed booleans exactly.
    if type(row["compression_assumed"]) is not bool or type(row["within_buffer_limit"]) is not bool:
        raise ZipBoundError("ZIP bound flags must be booleans")
    return expected


def required_torch_save_kwargs() -> dict:
    """Exact explicit keyword settings required at the eventual bounded save."""

    return {
        "pickle_protocol": 2,
        "_use_new_zipfile_serialization": True,
        "_disable_byteorder_record": False,
    }


def _hash_regular(path: Path, maximum: int = 2 << 20) -> str:
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ZipBoundError("installed Torch source is unavailable") from exc
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or not 0 <= info.st_size <= maximum:
            raise ZipBoundError("installed Torch source has invalid file identity")
        digest = hashlib.sha256()
        while chunk := os.read(descriptor, 64 << 10):
            digest.update(chunk)
        after = path.stat(follow_symlinks=False)
        descriptor_identity = (info.st_dev, info.st_ino, info.st_mode, info.st_nlink, info.st_size)
        path_identity = (after.st_dev, after.st_ino, after.st_mode, after.st_nlink, after.st_size)
        if path_identity != descriptor_identity:
            raise ZipBoundError("installed Torch source identity changed while reading")
    except OSError as exc:
        raise ZipBoundError("installed Torch source cannot be read") from exc
    finally:
        os.close(descriptor)
    return digest.hexdigest()


def validate_runtime_save_configuration(*, require_cuda_uninitialized: bool = True) -> dict:
    """Admit the pinned CPU-visible Torch source/configuration.

    This explicit call imports Torch.  It verifies installed Python sources,
    installed public headers, revision metadata and mutable save settings, but
    cannot prove that the loaded compiled writer binary came from those sources.
    """

    import functools
    import inspect
    import torch  # Lazy by design: import/default CLI stay Torch-free.
    import torch.serialization as serialization
    from torch.utils.serialization import config

    if type(require_cuda_uninitialized) is not bool:
        raise ZipBoundError("require_cuda_uninitialized must be a boolean")
    cuda_initialized_before = torch.cuda.is_initialized()
    if require_cuda_uninitialized and cuda_initialized_before:
        raise ZipBoundError("CUDA was initialized before ZIP runtime validation")
    if torch.__version__ != TORCH_VERSION or torch.version.git_version != TORCH_GIT_REVISION:
        raise ZipBoundError("Torch version or revision mismatch")
    if serialization.DEFAULT_PROTOCOL != 2:
        raise ZipBoundError("Torch default pickle protocol mismatch")
    if torch.save is not serialization.save:
        raise ZipBoundError("torch.save binding mismatch")
    save_parameters = inspect.signature(serialization.save).parameters
    if (
        type(save_parameters["pickle_protocol"].default) is not int
        or save_parameters["pickle_protocol"].default != 2
        or type(save_parameters["_use_new_zipfile_serialization"].default) is not bool
        or save_parameters["_use_new_zipfile_serialization"].default is not True
        or type(save_parameters["_disable_byteorder_record"].default) is not bool
        or save_parameters["_disable_byteorder_record"].default is not False
    ):
        raise ZipBoundError("torch.save call defaults mismatch")
    if sys.byteorder != "little":
        raise ZipBoundError("runtime byteorder is not the admitted little-endian value")
    if type(config.save.compute_crc32) is not bool or config.save.compute_crc32 is not True:
        raise ZipBoundError("Torch CRC configuration mismatch")
    if type(config.save.use_pinned_memory_for_d2h) is not bool or config.save.use_pinned_memory_for_d2h is not False:
        raise ZipBoundError("Torch pinned D2H configuration mismatch")
    if type(config.save.storage_alignment) is not int or config.save.storage_alignment != ALIGNMENT_BYTES:
        raise ZipBoundError("Torch storage alignment mismatch")
    if type(serialization._serialization_tls.skip_data) is not bool or serialization._serialization_tls.skip_data is not False:
        raise ZipBoundError("Torch skip_data context is active")
    expected_registry = (
        (10, serialization._cpu_tag, (), serialization._cpu_deserialize, ()),
        (20, serialization._backend_tag, ("cuda",), serialization._deserialize, ("cuda",)),
        (21, serialization._mps_tag, (), serialization._mps_deserialize, ()),
        (22, serialization._meta_tag, (), serialization._meta_deserialize, ()),
        (23, serialization._backend_tag, ("privateuse1",), serialization._deserialize, ("privateuse1",)),
        (24, serialization._backend_tag, ("hpu",), serialization._deserialize, ("hpu",)),
        (25, serialization._backend_tag, ("xpu",), serialization._deserialize, ("xpu",)),
        (26, serialization._backend_tag, ("mtia",), serialization._deserialize, ("mtia",)),
    )
    observed_registry = []
    for priority, tagger, deserializer in serialization._package_registry:
        if type(priority) is not int:
            raise ZipBoundError("Torch storage package registry priority is invalid")
        if isinstance(tagger, functools.partial):
            tag_function, tag_arguments = tagger.func, tagger.args
            tag_keywords = tuple(sorted((tagger.keywords or {}).items()))
        else:
            tag_function, tag_arguments, tag_keywords = tagger, (), ()
        if isinstance(deserializer, functools.partial):
            deserialize_function, deserialize_arguments = deserializer.func, deserializer.args
            deserialize_keywords = tuple(sorted((deserializer.keywords or {}).items()))
        else:
            deserialize_function, deserialize_arguments, deserialize_keywords = deserializer, (), ()
        observed_registry.append(
            (
                priority,
                tag_function,
                tag_arguments,
                tag_keywords,
                deserialize_function,
                deserialize_arguments,
                deserialize_keywords,
            )
        )
    expected_registry_with_keywords = tuple(
        (priority, tagger, tag_args, (), deserializer, deserialize_args, ())
        for priority, tagger, tag_args, deserializer, deserialize_args in expected_registry
    )
    if tuple(observed_registry) != expected_registry_with_keywords:
        raise ZipBoundError("Torch storage package registry mismatch")
    cpu_probe = type("_CPUStorageProbe", (), {"device": type("_CPUDevice", (), {"type": "cpu"})()})()
    if serialization.location_tag(cpu_probe) != "cpu":
        raise ZipBoundError("Torch CPU storage location tag mismatch")

    package = Path(torch.__file__).resolve().parent
    paths = {
        "serialization.py": package / "serialization.py",
        "utils/serialization/config.py": package / "utils" / "serialization" / "config.py",
        "include/caffe2/serialize/inline_container.h": package / "include" / "caffe2" / "serialize" / "inline_container.h",
        "include/caffe2/serialize/versions.h": package / "include" / "caffe2" / "serialize" / "versions.h",
    }
    observed = {name: _hash_regular(path) for name, path in paths.items()}
    if observed != INSTALLED_SOURCE_HASHES:
        raise ZipBoundError("installed Torch serializer source hash mismatch")
    cuda_initialized_after = torch.cuda.is_initialized()
    if cuda_initialized_after != cuda_initialized_before:
        raise ZipBoundError("CUDA initialization state changed during ZIP runtime validation")
    if require_cuda_uninitialized and cuda_initialized_after:
        raise ZipBoundError("CUDA initialized during ZIP runtime validation")

    result = {
        "schema": RUNTIME_SCHEMA,
        "torch_version": TORCH_VERSION,
        "torch_git_revision": TORCH_GIT_REVISION,
        "pickle_protocol": 2,
        "new_zipfile_serialization_required": True,
        "byteorder_record_required": True,
        "byteorder": "little",
        "compute_crc32": True,
        "storage_alignment_bytes": ALIGNMENT_BYTES,
        "use_pinned_memory_for_d2h": False,
        "skip_data": False,
        "archive_prefix": ARCHIVE_PREFIX,
        "package_registry_verified": True,
        "save_call_defaults_verified": True,
        "installed_source_hashes_verified": True,
        "upstream_source_hashes_bound": True,
        "require_cuda_uninitialized": require_cuda_uninitialized,
        "cuda_initialized": cuda_initialized_after,
        "cuda_initialization_unchanged": True,
        "source_revision_and_configuration_admitted": True,
        "compiled_binary_provenance_attested": False,
    }
    return validate_runtime_record(result)


def validate_runtime_record(value: object) -> dict:
    """Strictly decode the closed runtime-admission record."""

    row = _keys(value, RUNTIME_FIELDS, "ZIP runtime record")
    expected = {
        "schema": RUNTIME_SCHEMA,
        "torch_version": TORCH_VERSION,
        "torch_git_revision": TORCH_GIT_REVISION,
        "pickle_protocol": 2,
        "new_zipfile_serialization_required": True,
        "byteorder_record_required": True,
        "byteorder": "little",
        "compute_crc32": True,
        "storage_alignment_bytes": ALIGNMENT_BYTES,
        "use_pinned_memory_for_d2h": False,
        "skip_data": False,
        "archive_prefix": ARCHIVE_PREFIX,
        "package_registry_verified": True,
        "save_call_defaults_verified": True,
        "installed_source_hashes_verified": True,
        "upstream_source_hashes_bound": True,
        "require_cuda_uninitialized": row["require_cuda_uninitialized"],
        "cuda_initialized": row["cuda_initialized"],
        "cuda_initialization_unchanged": True,
        "source_revision_and_configuration_admitted": True,
        "compiled_binary_provenance_attested": False,
    }
    for field in (
        "new_zipfile_serialization_required",
        "byteorder_record_required",
        "compute_crc32",
        "use_pinned_memory_for_d2h",
        "skip_data",
        "package_registry_verified",
        "save_call_defaults_verified",
        "installed_source_hashes_verified",
        "upstream_source_hashes_bound",
        "require_cuda_uninitialized",
        "cuda_initialized",
        "cuda_initialization_unchanged",
        "source_revision_and_configuration_admitted",
        "compiled_binary_provenance_attested",
    ):
        if type(row[field]) is not bool:
            raise ZipBoundError("ZIP runtime flags must be booleans")
    if row["require_cuda_uninitialized"] and row["cuda_initialized"]:
        raise ZipBoundError("ZIP runtime record violates its CUDA requirement")
    if type(row["pickle_protocol"]) is not int or type(row["storage_alignment_bytes"]) is not int:
        raise ZipBoundError("ZIP runtime numeric fields must be integers")
    for field in ("schema", "torch_version", "torch_git_revision", "byteorder", "archive_prefix"):
        if type(row[field]) is not str:
            raise ZipBoundError("ZIP runtime text fields must be strings")
    if row != expected:
        raise ZipBoundError("ZIP runtime record mismatch")
    return dict(row)


def main(argv: list[str] | None = None) -> int:
    """Remain inert; there is deliberately no serialization CLI."""

    arguments = sys.argv[1:] if argv is None else argv
    if arguments:
        raise SystemExit("zip_storage_bound has no command-line execution mode")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
