"""One-descriptor immutable-store plan loading; not an execution/hostile-file sandbox."""
from __future__ import annotations

import fcntl
import hashlib
import io
import os
import re
import stat
import sys
import zipfile

import torch

import artifact_store as storage
import identity_codec as codec
import plan_bindings as plans
import runtime_guard as runtime


class VerifiedPlanLoadError(RuntimeError):
    pass


SCI_FILE_MAX = 4 << 20
FIXTURE_FILE_MAX = 64 << 10
_SHA = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)
_DIRECTORY_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
_BIG_TMP = "/tmp/spectral-experiment-artifacts"
_RECEIPT_KEYS = ("encoding", "name", "schema", "sha256", "size", "status")


def _require(condition, message):
    if not condition:
        raise VerifiedPlanLoadError(message)


def _signature(info):
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns,
            info.st_ctime_ns, info.st_mode, info.st_nlink)


def _snapshot(scanned):
    return {name: _signature(info) for name, info in scanned.items()}


def _root_identity(info):
    return info.st_dev, info.st_ino


def _open_root(path, profile):
    root = os.fspath(path)
    _require(type(root) is str and root.isascii() and root.startswith("/") and root != "/"
             and all(part not in ("", ".", "..") for part in root.split("/")[1:]),
             "root must be a canonical absolute ASCII directory path")
    volume = None
    if profile == storage.SCIENTIFIC:
        _require(root.startswith(_BIG_TMP + "/"), "scientific root must be below big/tmp")
        volume = runtime.verify_big_volume(root)
    initial = os.lstat(root)
    _require(stat.S_ISDIR(initial.st_mode) and not stat.S_ISLNK(initial.st_mode),
             "root is not a non-symlink directory")
    fd = os.open("/", _DIRECTORY_FLAGS)
    current = ""
    try:
        for part in root.split("/")[1:]:
            next_fd = os.open(part, _DIRECTORY_FLAGS, dir_fd=fd)
            os.close(fd)
            fd = next_fd
            current += "/" + part
            if volume is not None and (current == runtime.BIG_VOLUME["target"]
                                       or current.startswith(runtime.BIG_VOLUME["target"] + "/")):
                _require(os.fstat(fd).st_dev == os.makedev(volume["major"], volume["minor"]),
                         "scientific directory crossed the verified device")
        opened = os.fstat(fd)
        _require(_root_identity(initial) == _root_identity(opened), "root changed during open")
        return root, fd, opened, volume
    except BaseException:
        os.close(fd)
        raise


def _strict_metadata(dirfd, scanned):
    for name, info in scanned.items():
        _require(not name.startswith("failure-"), "terminal or partial failure artifact exists")
        if name == "store-header.json" or name.startswith("receipt-"):
            data, _ = storage._read_regular(dirfd, name, maximum=storage._RECEIPT_MAX, expected=info)
            value = codec.json_loads(data, max_bytes=storage._RECEIPT_MAX)
            _require(type(value) is dict and tuple(value) == tuple(sorted(value)),
                     "metadata is not the writer's canonical ordered mapping")


def _preflight(data, *, profile, steps):
    maximum = SCI_FILE_MAX if profile == storage.SCIENTIFIC else FIXTURE_FILE_MAX
    total, count, _, batch, probe = plans.DIMENSIONS[profile]
    expected_tensor_bytes = 8 * (total + 5 * count + steps * batch + probe)
    with zipfile.ZipFile(io.BytesIO(data), "r") as archive:
        entries = archive.infolist()
        _require(0 < len(entries) <= 32 and len({entry.filename for entry in entries}) == len(entries),
                 "invalid or duplicate ZIP entry membership")
        _require(sum(entry.file_size for entry in entries) <= maximum, "ZIP expansion exceeds file cap")
        prefixes = set()
        stores = []
        for entry in entries:
            parts = entry.filename.split("/")
            _require(entry.filename.isascii() and all(part not in ("", ".", "..") for part in parts),
                     "invalid ZIP entry path")
            _require(entry.compress_type == zipfile.ZIP_STORED and not (entry.flag_bits & 1)
                     and entry.file_size <= maximum and entry.compress_size == entry.file_size,
                     "compressed, encrypted or oversized ZIP entry")
            prefixes.add(parts[0])
            if len(parts) == 3 and parts[1] == "data":
                _require(parts[2].isdecimal() and str(int(parts[2])) == parts[2],
                         "invalid storage entry identifier")
                stores.append(entry)
        _require(len(prefixes) == 1, "multiple ZIP roots")
        _require(len(stores) == 8 and {entry.filename.rsplit("/", 1)[1] for entry in stores}
                 == {str(i) for i in range(8)}, "plan must have eight distinct storages")
        _require(sum(entry.file_size for entry in stores) == expected_tensor_bytes,
                 "plan storage bytes differ from exact profile")


def _check_safe_globals():
    # Torch 2.11 registers three at import and nine through AdamW's lazy imports.
    # Match exact objects in already-loaded modules; never clear/add globals.
    sources = (("torch.distributed._mesh_layout", "_MeshLayout"),
               ("torch.nested._internal.nested_tensor", "NestedTensor"),
               ("torch.nested._internal.nested_tensor", "_rebuild_njt"),
               ("torch._dynamo.decorators", "_DimRange"),
               ("torch.distributed.device_mesh", "DeviceMesh"),
               ("torch.distributed.tensor", "DTensor"),
               ("torch.distributed.tensor._dtensor_spec", "DTensorSpec"),
               ("torch.distributed.tensor._dtensor_spec", "ShardOrderEntry"),
               ("torch.distributed.tensor._dtensor_spec", "TensorMeta"),
               ("torch.distributed.tensor.placement_types", "Partial"),
               ("torch.distributed.tensor.placement_types", "Replicate"),
               ("torch.distributed.tensor.placement_types", "Shard"))
    known = [getattr(sys.modules.get(module), name, None) for module, name in sources]
    _require(all(any(value is default for default in known if default is not None)
                 for value in torch.serialization.get_safe_globals()),
             "custom safe globals beyond frozen Torch defaults are forbidden")


def _load_verified_plan_held(root, dirfd, lockfd, root_info, volume, name, *,
                             identity, profile, expected_sha256):
    """Shared verifier for caller-owned root and locked store descriptors."""
    codec.validate_identity(identity, profile=profile)
    storage._validate_name(name)
    _require(type(expected_sha256) is str and _SHA.fullmatch(expected_sha256),
             "expected manifest SHA-256 must be lowercase hexadecimal")
    lock_info = os.fstat(lockfd)
    _require(stat.S_ISREG(lock_info.st_mode) and lock_info.st_nlink == 1 and lock_info.st_size == 0,
             "malformed private store lock")
    initial, _, _ = storage._scan_regular(dirfd)
    _require("store.lock" in initial and _signature(lock_info) == _signature(initial["store.lock"]),
             "store lock replaced before inspection")
    maximum = SCI_FILE_MAX if profile == storage.SCIENTIFIC else FIXTURE_FILE_MAX
    _require(name in initial and 0 < initial[name].st_size <= maximum,
             "plan file exceeds fixed cap or is absent")
    _strict_metadata(dirfd, initial)
    report, scanned = storage._inspect_dirfd(dirfd)
    _require(_snapshot(initial) == _snapshot(scanned), "inventory changed before inspection")
    _require(report["header"]["profile"] == profile, "wrong store profile")
    _require(not report["terminal"] and not report["failures"], "terminal store cannot supply a plan")
    receipts = [row for row in report["receipts"] if row["name"] == name]
    _require(len(receipts) == 1, "missing or ambiguous complete receipt")
    receipt_name = "receipt-" + hashlib.sha256(name.encode("ascii")).hexdigest() + ".json"
    receipt_bytes, _ = storage._read_regular(dirfd, receipt_name,
        maximum=storage._RECEIPT_MAX, expected=scanned[receipt_name])
    receipt = codec.json_loads(receipt_bytes, max_bytes=storage._RECEIPT_MAX)
    _require(tuple(receipt) == _RECEIPT_KEYS and receipt == receipts[0], "receipt key/order or reread mismatch")
    _require(type(receipt["size"]) is int and 0 < receipt["size"] <= maximum
             and receipt["schema"] == "i7_artifact_receipt_v1" and receipt["name"] == name
             and receipt["status"] == "complete" and receipt["encoding"] == "torch_weights_only"
             and receipt["sha256"] == expected_sha256, "receipt differs from required file identity")
    data, _ = storage._read_regular(dirfd, name, maximum=receipt["size"], expected=scanned[name])
    digest = hashlib.sha256(data).hexdigest()
    _require(len(data) == receipt["size"] and digest == expected_sha256, "plan byte/hash mismatch")
    _preflight(data, profile=profile, steps=identity["steps_total"])
    _check_safe_globals()
    plan = torch.load(io.BytesIO(data), weights_only=True, map_location="cpu")
    plans.validate_plan(plan, identity=identity, profile=profile)
    artifact = {"sha256": digest, "size_bytes": len(data)}
    binding = plans.make_plan_binding(plan, identity=identity, profile=profile, artifact=artifact)
    plans.validate_plan_binding(binding, plan=plan, identity=identity, profile=profile, artifact=artifact)
    content_hash = codec.tree_digest(plan)
    final, logical, allocated = storage._scan_regular(dirfd)
    _require(_snapshot(scanned) == _snapshot(final), "inventory changed during verified decoding")
    _require(_signature(os.fstat(lockfd)) == _signature(scanned["store.lock"]), "store lock changed")
    _require(_root_identity(os.fstat(dirfd)) == _root_identity(root_info), "root descriptor changed")
    final_info = os.lstat(root)
    _require(stat.S_ISDIR(final_info.st_mode) and not stat.S_ISLNK(final_info.st_mode)
             and _root_identity(final_info) == _root_identity(root_info), "root path replaced during load")
    _, final_fd, reopened, final_volume = _open_root(root, profile)
    try:
        _require(_root_identity(reopened) == _root_identity(root_info), "root changed during final walk")
        _require(final_volume == volume, "scientific mount changed during load")
    finally:
        os.close(final_fd)
    return {
        "store": {"profile": profile, "root_device": root_info.st_dev, "root_inode": root_info.st_ino,
                  "logical_bytes": logical, "allocated_bytes": allocated},
        "artifact": {"name": name, "status": "complete", "encoding": "torch_weights_only",
                     "size_bytes": len(data), "sha256": digest, "receipt_name": receipt_name,
                     "receipt_size_bytes": len(receipt_bytes),
                     "receipt_sha256": hashlib.sha256(receipt_bytes).hexdigest()},
        "plan": plan, "binding": binding, "plan_content_sha256": content_hash,
    }


def load_verified_plan(root, name, *, identity, profile, expected_sha256):
    """Bind verified bytes and decoded plan at return; never accepts a caller plan."""
    dirfd = lockfd = None
    try:
        # Validate caller-controlled values before opening or locking the store.
        codec.validate_identity(identity, profile=profile)
        storage._validate_name(name)
        _require(type(expected_sha256) is str and _SHA.fullmatch(expected_sha256),
                 "expected manifest SHA-256 must be lowercase hexadecimal")
        root, dirfd, root_info, volume = _open_root(root, profile)
        lockfd = os.open("store.lock", os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=dirfd)
        lock_info = os.fstat(lockfd)
        _require(stat.S_ISREG(lock_info.st_mode) and lock_info.st_nlink == 1 and lock_info.st_size == 0,
                 "malformed private store lock")
        fcntl.flock(lockfd, fcntl.LOCK_SH | fcntl.LOCK_NB)
        return _load_verified_plan_held(root, dirfd, lockfd, root_info, volume, name,
            identity=identity, profile=profile, expected_sha256=expected_sha256)
    except VerifiedPlanLoadError:
        raise
    except Exception as exc:
        # Fail closed without dumping potentially unbounded decoder/backend text.
        raise VerifiedPlanLoadError("verified plan load failed: " + type(exc).__name__) from None
    finally:
        if lockfd is not None:
            os.close(lockfd)
        if dirfd is not None:
            os.close(dirfd)


def load_verified_plan_from_store(store, name, *, identity, profile, expected_sha256):
    """Verify a plan through an already exclusively locked exact ArtifactStore."""
    try:
        _require(type(store) is storage.ArtifactStore, "exact ArtifactStore required")
        _require(not store._closed and not store._terminal, "store is closed or terminal")
        _require(store.profile == profile, "wrong store profile")
        root, checkfd, root_info, volume = _open_root(store.root, profile)
        try:
            held_info = os.fstat(store._dirfd)
            _require(_root_identity(held_info) == _root_identity(root_info),
                     "held store root differs from canonical path")
        finally:
            os.close(checkfd)
        # Do not reopen or relock store.lock: separate flock open descriptions
        # conflict even within one process.  The exact store owns this descriptor.
        return _load_verified_plan_held(root, store._dirfd, store._lockfd, held_info, volume, name,
            identity=identity, profile=profile, expected_sha256=expected_sha256)
    except VerifiedPlanLoadError:
        raise
    except Exception as exc:
        raise VerifiedPlanLoadError("verified plan load failed: " + type(exc).__name__) from None
