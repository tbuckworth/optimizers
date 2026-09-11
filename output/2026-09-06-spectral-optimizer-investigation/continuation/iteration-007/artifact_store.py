"""Small immutable store; flock is advisory and this is not the I7 phase/mount controller."""
from __future__ import annotations

import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
import tempfile

import torch

SCIENTIFIC = "scientific_mnist_current32_v1"
FIXTURE = "fixture_tiny_cpu_v1"
MLP_FIXTURE = "fixture_tiny_mlp_cpu_v1"
FIXTURES = (FIXTURE, MLP_FIXTURE)
DEFAULT_BUDGET = 1 << 30
DEFAULT_FAILURE_RESERVE = 1 << 20
DEFAULT_MIN_FREE = 1 << 30
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}\Z")
_RESERVED = ("store-header.json", "store.lock")
_RECEIPT_MAX = 4096
_FAILURE_MAX = 16384


class StoreError(RuntimeError):
    pass


class StoreInitializationError(StoreError):
    """Initialization failed after zero or one actual root was created."""

    def __init__(self, partial_root):
        super().__init__("store initialization failed")
        self._partial_root = None if partial_root is None else dict(partial_root)

    @property
    def partial_root(self):
        return None if self._partial_root is None else dict(self._partial_root)


class _FileWriteError(StoreError):
    def __init__(self, name, cause):
        super().__init__(f"write of {name} failed: {cause}")
        self.name = name


class _BoundedBuffer(io.BytesIO):
    def __init__(self, limit):
        super().__init__()
        self.limit = limit
        self.exceeded = False

    def write(self, data):
        if self.tell() + len(data) > self.limit:
            self.exceeded = True
            raise StoreError("bounded serialization limit exceeded")
        return super().write(data)


def _json_bytes(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("ascii")


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _signature(info):
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)


def _scan_regular(dirfd):
    found, logical, allocated = {}, 0, 0
    with os.scandir(dirfd) as entries:
        for entry in entries:
            info = entry.stat(follow_symlinks=False)
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise StoreError("symlink, nonregular, or hardlinked entry")
            found[entry.name] = info
            logical += info.st_size
            allocated += info.st_blocks * 512
    return found, logical, allocated


def _read_regular(dirfd, name, *, maximum, expected=None):
    _validate_name(name, internal=True)
    if type(maximum) is not int or type(maximum) is bool or maximum < 0:
        raise StoreError("invalid bounded-read maximum")
    fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=dirfd)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise StoreError("unsafe regular-file read")
        if expected is not None and _signature(before) != _signature(expected):
            raise StoreError("file identity changed before read")
        if before.st_size > maximum:
            raise StoreError("bounded read size exceeded")
        buffer, remaining = io.BytesIO(), before.st_size + 1
        while remaining:
            part = os.read(fd, min(1 << 20, remaining))
            if not part:
                break
            buffer.write(part)
            remaining -= len(part)
        after = os.fstat(fd)
        if (not stat.S_ISREG(after.st_mode) or after.st_nlink != 1
                or _signature(after) != _signature(before) or buffer.tell() != before.st_size):
            raise StoreError("file changed during bounded read")
        return buffer.getvalue(), after
    finally:
        os.close(fd)


def _validate_name(name, *, internal=False):
    if type(name) is not str or not _NAME.fullmatch(name) or name in (".", ".."):
        raise StoreError("artifact name must be flat validated ASCII")
    if not internal and (name in _RESERVED or name.startswith(("receipt-", "failure-"))):
        raise StoreError("reserved artifact name")
    return name


def _native_context(profile, sources, environment):
    """Copy bounded metadata, not a source/runtime authentication or GO token.

    The native worker independently collects/authenticates these values before
    the still-closed runtime-admission gate. Private copies prevent later caller
    mutation from silently changing this writer's representation constraints.
    """
    if profile != SCIENTIFIC:
        if sources is not None or environment is not None:
            raise StoreError("native runtime metadata is scientific-profile only")
        return None, None
    import identity_codec as codec
    import native_payload_guard as guard
    try:
        guard.validate_runtime_metadata(sources, environment)
        return (codec.json_loads(codec.json_bytes(sources), max_bytes=32768),
                codec.json_loads(codec.json_bytes(environment), max_bytes=8192))
    except Exception as exc:
        raise StoreError("invalid scientific runtime metadata: " + type(exc).__name__) from exc


def _native_failure_reserve(scanned, prospective):
    """Bound all complete/partial root failure files inside the shared reserve.

    The terminal root ceiling alone protects 1 GiB, but permits failures to
    consume unused normal-output capacity. This additional inequality enforces
    the tighter failure allowance charged in the complete topology proof.
    """
    import native_write_ledger as ledger
    existing = 0
    for name, info in scanned.items():
        if not name.startswith("failure-"):
            continue
        if (re.fullmatch(ledger.FAILURE_NAME_PATTERN, name) is None
                or int(name[len("failure-"):-len(".json")]) < 1
                or not 0 <= info.st_size <= _FAILURE_MAX):
            raise StoreError("existing failure file is outside the admitted domain")
        # An exclusive create whose first write failed can leave zero bytes.
        existing += info.st_size
    reserve = ledger.SHARED_FAILURE_RESERVE_BYTES - ledger.EXTERNAL_FAILURE_BYTES
    if existing + prospective > reserve:
        raise StoreError("aggregate root failure allowance exhausted")


def _initialization_root(root, dirfd, expected_header):
    """Best-effort closed evidence about an actual retained initialization root."""
    if root is None:
        return None
    opened = None
    try:
        opened = dirfd
        if opened is None:
            opened = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        descriptor = os.fstat(opened)
        pathname = os.lstat(root)
        if (not stat.S_ISDIR(descriptor.st_mode)
                or not stat.S_ISDIR(pathname.st_mode) or stat.S_ISLNK(pathname.st_mode)
                or (descriptor.st_dev, descriptor.st_ino)
                    != (pathname.st_dev, pathname.st_ino)):
            return None
        header_status, header_sha = "absent", None
        try:
            header_info = os.stat("store-header.json", dir_fd=opened,
                                  follow_symlinks=False)
        except FileNotFoundError:
            pass
        except OSError:
            header_status = "unobserved"
        else:
            header_status = "present_unverified"
            if (expected_header is not None and stat.S_ISREG(header_info.st_mode)
                    and header_info.st_nlink == 1
                    and header_info.st_size == len(expected_header)):
                try:
                    raw, _ = _read_regular(opened, "store-header.json",
                                           maximum=_RECEIPT_MAX,
                                           expected=header_info)
                except (OSError, StoreError):
                    pass
                else:
                    if raw == expected_header:
                        header_status = "verified_complete"
                        header_sha = _sha(raw)
        return {"path":str(root), "device":descriptor.st_dev,
                "inode":descriptor.st_ino, "header_status":header_status,
                "header_sha256":header_sha}
    except (OSError, StoreError):
        return None
    finally:
        if opened is not None and opened != dirfd:
            os.close(opened)


def _validated_header(dirfd, scanned):
    if "store-header.json" not in scanned:
        raise StoreError("missing store header")
    data, _ = _read_regular(dirfd, "store-header.json", maximum=_RECEIPT_MAX,
                            expected=scanned["store-header.json"])
    try:
        row = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise StoreError("malformed store header") from exc
    keys = {"schema", "profile", "budget_bytes", "failure_reserve_bytes",
            "min_filesystem_free_bytes", "cap_semantics"}
    if set(row) != keys or row["schema"] != "i7_artifact_store_v1":
        raise StoreError("malformed store header")
    if row["profile"] not in (SCIENTIFIC, *FIXTURES):
        raise StoreError("malformed store profile")
    limits = tuple(row[key] for key in
                   ("budget_bytes", "failure_reserve_bytes", "min_filesystem_free_bytes"))
    if any(type(x) is not int or type(x) is bool or x < 0 for x in limits):
        raise StoreError("malformed store limits")
    if limits[1] < _FAILURE_MAX or limits[0] <= limits[1] + _RECEIPT_MAX:
        raise StoreError("inconsistent store limits")
    if row["profile"] == SCIENTIFIC and limits != (DEFAULT_BUDGET, DEFAULT_FAILURE_RESERVE,
                                                     DEFAULT_MIN_FREE):
        raise StoreError("scientific store limits differ from defaults")
    if row["cap_semantics"] != "sum_regular_file_logical_bytes":
        raise StoreError("wrong cap semantics")
    return row


def _validate_tree(value, active=None, storages=None):
    active = set() if active is None else active
    storages = set() if storages is None else storages
    if type(value) is torch.Tensor:
        if value.device.type != "cpu" or value.layout != torch.strided or value.requires_grad:
            raise StoreError("tensor must be detached dense CPU storage")
        if not value.is_contiguous() or value.storage_offset() != 0 or value._base is not None:
            raise StoreError("tensor must be contiguous and compact")
        logical = value.numel() * value.element_size()
        if value.untyped_storage().nbytes() != logical:
            raise StoreError("tensor must own compact storage")
        if (value.is_floating_point() or value.is_complex()) and not bool(torch.isfinite(value).all()):
            raise StoreError("nonfinite tensor")
        if logical:
            identity = (value.untyped_storage().data_ptr(), logical)
            if identity in storages:
                raise StoreError("shared tensor storage")
            storages.add(identity)
        return
    if value is None or type(value) in (bool, int, str):
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise StoreError("nonfinite scalar")
        return
    if type(value) not in (dict, list, tuple):
        raise StoreError("tree contains a non tensor/primitive value")
    if id(value) in active:
        raise StoreError("cyclic container")
    active.add(id(value))
    if type(value) is dict:
        for key, item in value.items():
            if type(key) not in (str, int) or type(key) is bool:
                raise StoreError("mapping keys must be strings or integers")
            _validate_tree(item, active, storages)
    else:
        for item in value:
            _validate_tree(item, active, storages)
    active.remove(id(value))


def _inspect_dirfd(dirfd):
    """Inspect one held directory descriptor; return report and validated snapshot."""
    scanned, logical, allocated = _scan_regular(dirfd)
    header = _validated_header(dirfd, scanned)
    if "store.lock" not in scanned or scanned["store.lock"].st_size != 0:
        raise StoreError("missing or malformed store lock")
    if logical > header["budget_bytes"]:
        raise StoreError("store exceeds logical byte cap")
    receipts, failures, indexed = [], [], {"store.lock", "store-header.json"}
    failure_items = {}
    for name, info in scanned.items():
        if not name.startswith("failure-"):
            continue
        data, _ = _read_regular(dirfd, name, maximum=_FAILURE_MAX, expected=info)
        try:
            row = json.loads(data)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise StoreError("malformed metadata JSON") from exc
        keys = {"schema", "code", "detail", "files", "terminal"}
        if (not re.fullmatch(r"failure-[0-9]{6}\.json", name) or set(row) != keys
                or row["schema"] != "i7_store_failure_v1" or row["terminal"] is not True
                or type(row["code"]) is not str or type(row["detail"]) is not str
                or type(row["files"]) is not list or len(row["files"]) > 2):
            raise StoreError("malformed failure record")
        failures.append(row)
        indexed.add(name)
        for item in row["files"]:
            if type(item) is dict and type(item.get("name")) is str:
                failure_items[item["name"]] = item.get("status")
    for name, info in scanned.items():
        if not name.startswith("receipt-") or failure_items.get(name) == "partial_or_complete":
            continue
        data, _ = _read_regular(dirfd, name, maximum=_RECEIPT_MAX, expected=info)
        try:
            row = json.loads(data)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise StoreError("malformed receipt JSON") from exc
        keys = {"schema", "name", "size", "sha256", "status", "encoding"}
        if set(row) != keys or row["schema"] != "i7_artifact_receipt_v1":
            raise StoreError("malformed receipt")
        artifact = _validate_name(row["name"])
        expected_name = "receipt-" + _sha(artifact.encode("ascii")) + ".json"
        if name != expected_name or type(row["size"]) is not int or type(row["size"]) is bool:
            raise StoreError("receipt identity/type mismatch")
        if (row["size"] < 0 or row["size"] > header["budget_bytes"]
                or row["status"] != "complete"
                or row["encoding"] not in ("bytes", "torch_weights_only")
                or not re.fullmatch(r"[0-9a-f]{64}", row["sha256"])):
            raise StoreError("malformed receipt values")
        receipts.append(row)
        indexed.add(name)
    for row in receipts:
        name = row["name"]
        if name not in scanned:
            raise StoreError("receipt payload missing")
        data, _ = _read_regular(dirfd, name, maximum=row["size"], expected=scanned[name])
        if len(data) != row["size"] or _sha(data) != row["sha256"]:
            raise StoreError("receipt size/hash mismatch")
        indexed.add(name)
    allowed_status = {"complete_without_receipt", "complete_receipt", "partial_or_complete"}
    for row in failures:
        for item in row["files"]:
            if (type(item) is not dict or set(item) != {"name", "status", "size", "sha256"}
                    or item["status"] not in allowed_status or type(item["size"]) is not int
                    or type(item["size"]) is bool or item["size"] < 0
                    or item["size"] > header["budget_bytes"]
                    or not re.fullmatch(r"[0-9a-f]{64}", item["sha256"])):
                raise StoreError("malformed failure file row")
            name = _validate_name(item["name"], internal=True)
            if name not in scanned:
                raise StoreError("failure-indexed file missing")
            data, _ = _read_regular(dirfd, name, maximum=item["size"], expected=scanned[name])
            if len(data) != item["size"] or _sha(data) != item["sha256"]:
                raise StoreError("failure-file size/hash mismatch")
            indexed.add(name)
    if set(scanned) != indexed:
        raise StoreError("unindexed regular file")
    final, final_logical, final_allocated = _scan_regular(dirfd)
    if {n:_signature(v) for n,v in final.items()} != {n:_signature(v) for n,v in scanned.items()}:
        raise StoreError("inventory changed during inspection")
    files = {name:{"size":info.st_size, "allocated_bytes":info.st_blocks * 512}
             for name,info in final.items()}
    return ({"header":header, "files":files, "receipts":receipts, "failures":failures,
            "terminal":bool(failures), "logical_bytes":final_logical,
            "allocated_bytes":final_allocated}, scanned)


class ArtifactStore:
    """One-process writer for a new or explicitly pinned immutable flat directory."""

    def __init__(self, parent, *, profile=SCIENTIFIC, budget_bytes=DEFAULT_BUDGET,
                 failure_reserve_bytes=DEFAULT_FAILURE_RESERVE,
                 min_filesystem_free_bytes=DEFAULT_MIN_FREE,
                 runtime_sources=None, runtime_environment=None):
        if profile not in (SCIENTIFIC, *FIXTURES):
            raise StoreError("unknown profile")
        limits = (budget_bytes, failure_reserve_bytes, min_filesystem_free_bytes)
        if any(type(x) is not int or type(x) is bool or x < 0 for x in limits):
            raise StoreError("limits must be nonnegative integers")
        if profile not in FIXTURES and limits != (DEFAULT_BUDGET, DEFAULT_FAILURE_RESERVE, DEFAULT_MIN_FREE):
            raise StoreError("custom limits are fixture-profile only")
        if failure_reserve_bytes < _FAILURE_MAX or budget_bytes <= failure_reserve_bytes + _RECEIPT_MAX:
            raise StoreError("insufficient budget or failure reserve")
        self._runtime_sources, self._runtime_environment = _native_context(
            profile, runtime_sources, runtime_environment)
        parent = Path(os.path.abspath(os.fspath(parent)))
        info = os.lstat(parent)
        if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
            raise StoreError("parent must be an existing non-symlink directory")
        self.profile, self.budget = profile, budget_bytes
        self.failure_reserve, self.min_free = failure_reserve_bytes, min_filesystem_free_bytes
        self._dirfd = self._lockfd = None
        self._indexed, self._terminal, self._closed, self._op = {}, False, False, 0
        self.failure_metadata_status = None
        self.root = None
        header = None
        try:
            self.root = Path(tempfile.mkdtemp(prefix="i7-artifacts-", dir=parent))
            self._dirfd = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            if self.profile == SCIENTIFIC:
                import native_write_ledger as ledger
                ledger.initialization_reservation("store.lock", 0, 0)
            self._lockfd, lock_meta = self._create_file("store.lock", b"")
            fcntl.flock(self._lockfd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self._indexed["store.lock"] = lock_meta
            header = _json_bytes({"schema":"i7_artifact_store_v1", "profile":profile,
                                  "budget_bytes":budget_bytes, "failure_reserve_bytes":failure_reserve_bytes,
                                  "min_filesystem_free_bytes":min_filesystem_free_bytes,
                                  "cap_semantics":"sum_regular_file_logical_bytes"})
            if self.profile == SCIENTIFIC:
                logical, _ = self._inventory(self._indexed)
                ledger.initialization_reservation("store-header.json", len(header), logical)
            self._precheck(len(header), reserve_failure=False)
            fd, meta = self._create_file("store-header.json", header)
            os.close(fd)
            self._indexed["store-header.json"] = meta
            os.fsync(self._dirfd)
        except BaseException as exc:
            partial_root = _initialization_root(self.root, self._dirfd, header)
            if self._lockfd is not None:
                try:
                    fcntl.flock(self._lockfd, fcntl.LOCK_UN)
                finally:
                    os.close(self._lockfd)
            if self._dirfd is not None:
                os.close(self._dirfd)
            self._closed = True
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            raise StoreInitializationError(partial_root) from exc

    @property
    def root_identity(self):
        """Device/inode pin for a later bounded reopen; capture before close."""
        if self._closed:
            raise StoreError("store is closed")
        info = os.fstat(self._dirfd)
        return info.st_dev, info.st_ino

    @property
    def header_sha256(self):
        """Exact header pin for a later bounded reopen."""
        return self._indexed["store-header.json"]["sha256"]

    @classmethod
    def reopen(cls, root, *, expected_profile, expected_header_sha256,
               expected_root_identity, runtime_sources=None, runtime_environment=None):
        """Reattach one writer to a closed, pinned, nonterminal store.

        This only restores the immutable-store writer and its byte accounting.  A
        phase controller must separately establish any sealed-boundary protocol;
        reopening is not evidence that a live input or source tree resumed.
        """
        if cls is not ArtifactStore:
            raise StoreError("exact ArtifactStore class required")
        if expected_profile not in (SCIENTIFIC, *FIXTURES):
            raise StoreError("unknown expected profile")
        if (type(expected_header_sha256) is not str
                or not re.fullmatch(r"[0-9a-f]{64}", expected_header_sha256)):
            raise StoreError("expected header SHA-256 must be lowercase hexadecimal")
        if (type(expected_root_identity) is not tuple or len(expected_root_identity) != 2
                or any(type(value) is not int or type(value) is bool or value < 0
                       for value in expected_root_identity)):
            raise StoreError("expected root identity must be an exact (device, inode) tuple")

        native_sources, native_environment = _native_context(
            expected_profile, runtime_sources, runtime_environment)

        dirfd = lockfd = None
        try:
            # Lazy import avoids artifact_store <-> verified_plan_load import setup cycles.
            import verified_plan_load as verified

            canonical, dirfd, root_info, volume = verified._open_root(root, expected_profile)
            if (root_info.st_dev, root_info.st_ino) != expected_root_identity:
                raise StoreError("store root identity differs from pinned expectation")
            lockfd = os.open("store.lock", os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                             dir_fd=dirfd)
            lock_info = os.fstat(lockfd)
            if (not stat.S_ISREG(lock_info.st_mode) or lock_info.st_nlink != 1
                    or lock_info.st_size != 0):
                raise StoreError("malformed private store lock")
            fcntl.flock(lockfd, fcntl.LOCK_EX | fcntl.LOCK_NB)

            initial, _, _ = _scan_regular(dirfd)
            if ("store.lock" not in initial
                    or verified._signature(lock_info) != verified._signature(initial["store.lock"])):
                raise StoreError("store lock replaced before inspection")
            verified._strict_metadata(dirfd, initial)
            report, scanned = _inspect_dirfd(dirfd)
            if verified._snapshot(initial) != verified._snapshot(scanned):
                raise StoreError("inventory changed before inspection")
            header = report["header"]
            if header["profile"] != expected_profile:
                raise StoreError("store profile differs from pinned expectation")
            if report["terminal"] or report["failures"]:
                raise StoreError("terminal store cannot be reopened")
            if expected_profile == SCIENTIFIC:
                import native_write_ledger as ledger
                # Inspection has already authenticated every body/receipt pair.
                # Also reject unknown names, encodings and per-name size excess;
                # a loose old root-only header is not a complete ledger.
                ledger.initialization_reservation(
                    "store-header.json", scanned["store-header.json"].st_size, 0)
                for receipt in report["receipts"]:
                    receipt_name = "receipt-" + _sha(receipt["name"].encode("ascii")) + ".json"
                    ledger.admit_normal_write(
                        receipt["name"], receipt["encoding"], receipt["size"],
                        scanned[receipt_name].st_size, 0)
            header_bytes, _ = _read_regular(
                dirfd, "store-header.json", maximum=_RECEIPT_MAX,
                expected=scanned["store-header.json"])
            if _sha(header_bytes) != expected_header_sha256:
                raise StoreError("store header differs from pinned expectation")

            indexed = {}
            for name, info in scanned.items():
                data, final_info = _read_regular(dirfd, name, maximum=header["budget_bytes"],
                                                 expected=info)
                indexed[name] = cls._meta(final_info, data)
            final, _, _ = _scan_regular(dirfd)
            if verified._snapshot(scanned) != verified._snapshot(final):
                raise StoreError("inventory changed while rebuilding indexed state")
            if verified._signature(os.fstat(lockfd)) != verified._signature(scanned["store.lock"]):
                raise StoreError("store lock changed during reopen")
            if (os.fstat(dirfd).st_dev, os.fstat(dirfd).st_ino) != expected_root_identity:
                raise StoreError("root descriptor changed during reopen")
            final_path = os.lstat(canonical)
            if (not stat.S_ISDIR(final_path.st_mode) or stat.S_ISLNK(final_path.st_mode)
                    or (final_path.st_dev, final_path.st_ino) != expected_root_identity):
                raise StoreError("root path replaced during reopen")
            _, final_fd, reopened_info, final_volume = verified._open_root(
                canonical, expected_profile)
            try:
                if (reopened_info.st_dev, reopened_info.st_ino) != expected_root_identity:
                    raise StoreError("root changed during final reopen walk")
                if final_volume != volume:
                    raise StoreError("scientific mount changed during reopen")
            finally:
                os.close(final_fd)

            self = cls.__new__(cls)
            self.root = Path(canonical)
            self.profile, self.budget = expected_profile, header["budget_bytes"]
            self.failure_reserve = header["failure_reserve_bytes"]
            self.min_free = header["min_filesystem_free_bytes"]
            self._dirfd, self._lockfd = dirfd, lockfd
            self._indexed, self._terminal, self._closed, self._op = indexed, False, False, 0
            self._runtime_sources, self._runtime_environment = native_sources, native_environment
            self.failure_metadata_status = None
            self._precheck(0)
            dirfd = lockfd = None
            return self
        except (KeyboardInterrupt, SystemExit):
            raise
        except StoreError:
            raise
        except Exception as exc:
            raise StoreError("store reopen failed: " + type(exc).__name__) from None
        finally:
            if lockfd is not None:
                os.close(lockfd)
            if dirfd is not None:
                os.close(dirfd)

    def _write_all(self, fd, payload):
        view = memoryview(payload)
        while view:
            count = os.write(fd, view)
            if count <= 0:
                raise OSError("short write")
            view = view[count:]

    def _create_file(self, name, payload):
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
        fd = os.open(name, flags, 0o600, dir_fd=self._dirfd)
        try:
            self._write_all(fd, payload)
            os.fsync(fd)
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise StoreError("new artifact is not a private regular file")
            if info.st_size != len(payload):
                raise StoreError("new artifact size changed during write")
            return fd, self._meta(info, payload)
        except BaseException as exc:
            os.close(fd)
            raise _FileWriteError(name, exc) from exc

    @staticmethod
    def _meta(info, payload):
        return {"size":info.st_size, "allocated_bytes":info.st_blocks * 512,
                "sha256":_sha(payload), "inode":info.st_ino, "mtime_ns":info.st_mtime_ns}

    def _inventory(self, expected=None):
        scanned, logical, allocated = _scan_regular(self._dirfd)
        found = {name:_signature(info) for name,info in scanned.items()}
        if expected is not None:
            wanted = {n:(os.fstat(self._dirfd).st_dev,m["inode"],m["size"],m["mtime_ns"])
                      for n,m in expected.items()}
            if found != wanted:
                raise StoreError("unindexed file or indexed-file inventory change")
        return logical, allocated

    def _precheck(self, prospective, *, reserve_failure=True):
        logical, _ = self._inventory(self._indexed)
        ceiling = self.budget - (self.failure_reserve if reserve_failure else 0)
        if self.profile == SCIENTIFIC:
            import native_write_ledger as ledger
            # Header initialization is normal output too; it cannot borrow the
            # shared failure allowance via reserve_failure=False.
            ceiling = ledger.NORMAL_ROOT_CEILING_BYTES
        if logical + prospective > ceiling:
            raise StoreError("logical byte budget gate")
        # Physical allocation is filesystem-dependent; this is only a conservative free-space precheck.
        if shutil.disk_usage(self.root).free - prospective < self.min_free:
            raise StoreError("minimum filesystem free-space gate")

    def _file_snapshot(self, name, status):
        data, info = _read_regular(self._dirfd, name, maximum=self.budget)
        meta = {"size":info.st_size, "allocated_bytes":info.st_blocks * 512,
                "sha256":_sha(data), "inode":info.st_ino, "mtime_ns":info.st_mtime_ns}
        return {"name":name, "status":status, "size":meta["size"], "sha256":meta["sha256"]}, meta

    def _fail(self, code, detail, affected=()):
        self._terminal = True
        files = []
        for name, status in affected:
            try:
                row, meta = self._file_snapshot(name, status)
                files.append(row)
                self._indexed[name] = meta
            except (FileNotFoundError, StoreError):
                pass
        self._op += 1
        name = f"failure-{self._op:06d}.json"
        record = _json_bytes({"schema":"i7_store_failure_v1", "code":code,
                              "detail":str(detail)[:512], "files":files, "terminal":True})
        if len(record) > _FAILURE_MAX:
            self.failure_metadata_status = "not_retained:record_too_large"
            return self.failure_metadata_status
        try:
            scanned, logical, _ = _scan_regular(self._dirfd)
            if self.profile == SCIENTIFIC:
                import native_write_ledger as ledger
                _native_failure_reserve(scanned, len(record))
                try:
                    ledger.admit_failure_write(name, len(record), logical)
                except ValueError as exc:
                    raise StoreError(str(exc)) from exc
            if logical + len(record) > self.budget:
                raise StoreError("failure record would exceed logical byte cap")
            if shutil.disk_usage(self.root).free - len(record) < self.min_free:
                raise StoreError("failure record would violate free-space floor")
            fd, meta = self._create_file(name, record)
            os.close(fd)
            self._indexed[name] = meta
            os.fsync(self._dirfd)
            try:
                self._inventory(self._indexed)
                self.failure_metadata_status = "retained"
            except StoreError as exc:
                self.failure_metadata_status = "retained_but_inventory_invalid:" + str(exc)[:192]
        except (OSError, StoreError) as exc:
            self.failure_metadata_status = "not_retained:" + str(exc)[:256]
        return self.failure_metadata_status

    def _ensure_writable(self):
        if self._closed or self._terminal:
            raise StoreError("store is closed or terminal")

    def _commit(self, name, payload, encoding):
        self._ensure_writable()
        created = []
        try:
            _validate_name(name)
            digest = _sha(payload)
            receipt_name = "receipt-" + _sha(name.encode("ascii")) + ".json"
            receipt = {"schema":"i7_artifact_receipt_v1", "name":name, "size":len(payload),
                       "sha256":digest, "status":"complete", "encoding":encoding}
            receipt_bytes = _json_bytes(receipt)
            if name in self._indexed or receipt_name in self._indexed:
                raise StoreError("immutable target already exists")
            reserved = len(payload) + len(receipt_bytes)
            if self.profile == SCIENTIFIC:
                import native_write_ledger as ledger
                logical, _ = self._inventory(self._indexed)
                ledger.admit_normal_write(name, encoding, len(payload), len(receipt_bytes), logical)
                allowance = ledger.normal_write_budget(name, encoding, logical)
                reserved = len(payload) + allowance["receipt_bytes_upper"]
            self._precheck(reserved)
            fd, payload_meta = self._create_file(name, payload)
            os.close(fd)
            created.append((name, "complete_without_receipt"))
            fd, receipt_meta = self._create_file(receipt_name, receipt_bytes)
            os.close(fd)
            created.append((receipt_name, "complete_receipt"))
            proposed = dict(self._indexed)
            proposed[name], proposed[receipt_name] = payload_meta, receipt_meta
            self._inventory(proposed)
            self._indexed = proposed
            self._precheck(0)
            os.fsync(self._dirfd)
            return dict(receipt, receipt_name=receipt_name)
        except BaseException as exc:
            if isinstance(exc, _FileWriteError):
                created.append((exc.name, "partial_or_complete"))
            failure_status = self._fail("write_failed", exc, created)
            raise StoreError(f"immutable write failed: {exc}; failure_metadata={failure_status}") from exc

    def write_bytes(self, name, payload):
        self._ensure_writable()
        try:
            if type(payload) is not bytes:
                raise StoreError("payload must be bytes")
            if self.profile == SCIENTIFIC:
                import identity_codec as codec
                import native_payload_guard as guard
                import native_write_ledger as ledger
                logical, _ = self._inventory(self._indexed)
                allowance = ledger.normal_write_budget(name, "bytes", logical)
                if len(payload) > allowance["max_payload_bytes_now"]:
                    raise StoreError("scientific byte payload exceeds admitted capacity")
                if allowance["component"] == "capture_comparison_pilot":
                    value = codec.json_loads(payload, max_bytes=allowance["body_bytes_upper"])
                    guard.admit_json_payload(name, value, sources=self._runtime_sources,
                                             environment=self._runtime_environment)
                elif name == "native-sources.json":
                    if payload != codec.json_bytes(self._runtime_sources):
                        raise StoreError("source metadata differs from writer binding")
                elif name.startswith("native-") and name.endswith("-environment.json"):
                    if payload != codec.json_bytes(self._runtime_environment):
                        raise StoreError("environment metadata differs from writer binding")
        except Exception as exc:
            failure_status = self._fail("input_validation_failed", exc)
            raise StoreError(f"{exc}; failure_metadata={failure_status}") from exc
        return self._commit(name, payload, "bytes")

    def write_tensor_tree(self, name, tree):
        self._ensure_writable()
        try:
            _validate_name(name)
            logical, _ = self._inventory(self._indexed)
            if self.profile == SCIENTIFIC:
                import native_payload_guard as guard
                import native_write_ledger as ledger
                import pickle_storage_bound as pickle_bound
                import zip_storage_bound as zip_bound
                allowance = ledger.normal_write_budget(name, "torch_weights_only", logical)
                admitted = guard.admit_torch_payload(name, tree, sources=self._runtime_sources,
                                                    environment=self._runtime_environment)
                # Preserve the pre-existing finite tensor-value contract. The
                # specialized traversal first bounds the representation/tree;
                # its size proof alone is not a numeric finiteness check.
                _validate_tree(tree)
                pickle_bound.assert_pinned_pickle_runtime()
                zip_bound.validate_runtime_save_configuration(
                    require_cuda_uninitialized=self._runtime_environment["runtime_role"] == "cpu_audit")
                save_kwargs = zip_bound.required_torch_save_kwargs()
                bound = min(admitted["zip_bytes_upper"], allowance["max_payload_bytes_now"],
                            shutil.disk_usage(self.root).free - self.min_free
                            - allowance["receipt_bytes_upper"])
            else:
                _validate_tree(tree)
                save_kwargs = {}
                bound = min(self.budget - self.failure_reserve - logical - _RECEIPT_MAX,
                            shutil.disk_usage(self.root).free - self.min_free - _RECEIPT_MAX)
            if bound < 0:
                raise StoreError("no bounded serialization capacity")
            buffer = _BoundedBuffer(bound)
            torch.save(tree, buffer, **save_kwargs)
            return self._commit(name, buffer.getvalue(), "torch_weights_only")
        except BaseException as exc:
            if not self._terminal:
                failure_status = self._fail("serialization_failed", exc)
            else:
                failure_status = self.failure_metadata_status
            if "buffer" in locals() and buffer.exceeded:
                raise StoreError(f"bounded serialization limit exceeded; failure_metadata={failure_status}") from exc
            if isinstance(exc, StoreError):
                raise StoreError(f"{exc}; failure_metadata={failure_status}") from exc
            raise StoreError(f"serialization failed: {exc}; failure_metadata={failure_status}") from exc

    def close(self):
        if self._closed:
            return
        fcntl.flock(self._lockfd, fcntl.LOCK_UN)
        os.close(self._lockfd)
        os.close(self._dirfd)
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    @staticmethod
    def inspect(root):
        root = Path(root)
        info = os.lstat(root)
        if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
            raise StoreError("store root must be a non-symlink directory")
        dirfd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            return _inspect_dirfd(dirfd)[0]
        finally:
            os.close(dirfd)

    @staticmethod
    def load_tensor_tree(root, name, *, expected_size, expected_sha256):
        """Verify bytes before weights-only decode; not safe for hostile memory metadata."""
        _validate_name(name)
        if type(expected_size) is not int or type(expected_size) is bool or expected_size < 0:
            raise StoreError("expected size must be a nonnegative exact integer")
        if type(expected_sha256) is not str or not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
            raise StoreError("expected SHA-256 must be lowercase hexadecimal")
        dirfd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            scanned, logical, _ = _scan_regular(dirfd)
            header = _validated_header(dirfd, scanned)
            if logical > header["budget_bytes"]:
                raise StoreError("store exceeds logical byte cap")
            if expected_size > header["budget_bytes"]:
                raise StoreError("expected size exceeds store budget ceiling")
            if name not in scanned or scanned[name].st_size != expected_size:
                raise StoreError("size/type/link mismatch before decode")
            data, _ = _read_regular(dirfd, name, maximum=expected_size, expected=scanned[name])
        finally:
            os.close(dirfd)
        if _sha(data) != expected_sha256:
            raise StoreError("hash mismatch before decode")
        value = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
        _validate_tree(value)
        return value
