"""One-shot reservation for the fixed I7 storage-measurement slot.

Import and the default CLI are inert and Torch-free.  The public reservation
has no path parameter: it can consume only the fixed, pre-accounted M slot.
Any created zero or partial file is terminal evidence and is never removed,
reopened, truncated, overwritten, or retried by this module.
"""
from __future__ import annotations

import hashlib
import os
import re
import stat
import threading


MEASUREMENT_BASENAME = "native-max-layout-measurement-attempt-002.json"
ADMISSION_BASENAME = "native-storage-admission.json"
MEASUREMENT_MAX_BYTES = 64 << 10
MIN_FREE_BYTES = 1 << 30
BIG_TMP = "/tmp/spectral-experiment-artifacts"

_COMMIT = re.compile(r"[0-9a-f]{40}\Z", re.ASCII)
_SHA256 = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)


class MeasurementSlotError(RuntimeError):
    """Closed failure classification; arbitrary underlying text is omitted."""

    def __init__(self, code):
        self.code = code
        super().__init__(code)


def _raise(code):
    raise MeasurementSlotError(code) from None


def _owner_token():
    return os.getpid(), threading.get_ident()


def _signature(info):
    return info.st_dev, info.st_ino, info.st_mode, info.st_nlink


def _valid_private_regular(info, *, size):
    return (stat.S_ISREG(info.st_mode) and stat.S_IMODE(info.st_mode) == 0o600
            and info.st_nlink == 1 and info.st_size == size)


def _open_parent(path):
    if (type(path) is not str or not path.startswith("/") or path.startswith("//")
            or os.path.normpath(path) != path or "\x00" in path):
        _raise("invalid_parent")
    parts = path.split("/")[1:]
    if not parts or not all(parts):
        _raise("invalid_parent")
    try:
        fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            for part in parts[:-1]:
                next_fd = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                                  dir_fd=fd)
                os.close(fd)
                fd = next_fd
            return fd, parts[-1]
        except BaseException:
            os.close(fd)
            raise
    except MeasurementSlotError:
        raise
    except OSError:
        _raise("invalid_parent")


def _entry(parent_fd, name):
    try:
        return os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return None
    except OSError:
        _raise("invalid_parent")


def _pwrite(fd, data, offset):
    return os.pwrite(fd, data, offset)


class MeasurementReservation:
    """Held-descriptor capability for exactly one finalize attempt."""

    __slots__ = ("_admission_name", "_closed", "_expected_commit",
                 "_expected_source_set_sha256", "_fd", "_finalize_attempted",
                 "_measurement_name", "_measurement_path", "_owner",
                 "_parent_fd", "_parent_signature",
                 "_slot_signature")

    def __init__(self, *, parent_fd, measurement_name,
                 admission_name, measurement_path, expected_commit,
                 expected_source_set_sha256, slot_info):
        self._parent_fd = parent_fd
        self._fd = None
        self._measurement_name = measurement_name
        self._admission_name = admission_name
        self._measurement_path = measurement_path
        self._expected_commit = expected_commit
        self._expected_source_set_sha256 = expected_source_set_sha256
        self._parent_signature = _signature(os.fstat(parent_fd))
        self._slot_signature = _signature(slot_info)
        self._owner = _owner_token()
        self._finalize_attempted = False
        self._closed = False

    def _require_owner_open(self):
        if _owner_token() != self._owner:
            _raise("owner_mismatch")
        if self._closed or self._fd is None or self._parent_fd is None:
            _raise("closed")

    def _check_identity(self, *, size):
        try:
            parent = os.fstat(self._parent_fd)
            slot = os.fstat(self._fd)
            current = os.stat(self._measurement_name, dir_fd=self._parent_fd,
                              follow_symlinks=False)
            reopened, reopened_name = _open_parent(self._measurement_path)
            try:
                reopened_parent = os.fstat(reopened)
            finally:
                os.close(reopened)
        except MeasurementSlotError:
            _raise("slot_identity_changed")
        except OSError:
            _raise("slot_identity_changed")
        if (reopened_name != self._measurement_name
                or _signature(parent) != self._parent_signature
                or _signature(reopened_parent) != self._parent_signature
                or _signature(slot) != self._slot_signature
                or _signature(current) != self._slot_signature
                or not _valid_private_regular(slot, size=size)
                or current.st_size != size):
            _raise("slot_identity_changed")

    def _require_admission_absent(self):
        if _entry(self._parent_fd, self._admission_name) is not None:
            _raise("admission_appeared")

    def finalize(self, measurement):
        """Validate and durably write one bounded report through the held fd."""
        self._require_owner_open()
        if self._finalize_attempted:
            _raise("finalize_already_attempted")
        # Set first: validation errors and interrupts also consume this capability.
        self._finalize_attempted = True
        self._check_identity(size=0)
        self._require_admission_absent()

        try:
            import native_storage_authority as authority
            authority.validate_measurement(
                measurement, expected_commit=self._expected_commit,
                expected_source_set_sha256=self._expected_source_set_sha256,
                require_complete=False, recompute=True)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            _raise("measurement_invalid")
        try:
            raw = authority.encode_bounded(measurement, maximum=MEASUREMENT_MAX_BYTES)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            _raise("encode_failed")

        # Recheck after the potentially expensive recomputation.  No pathname
        # change or premature A may be accepted in the validation-to-write gap.
        self._check_identity(size=0)
        self._require_admission_absent()
        try:
            offset = 0
            while offset < len(raw):
                written = _pwrite(self._fd, memoryview(raw)[offset:], offset)
                if type(written) is not int or not 0 < written <= len(raw) - offset:
                    raise OSError("invalid bounded write")
                offset += written
            os.fsync(self._fd)
            os.fsync(self._parent_fd)
        except (KeyboardInterrupt, SystemExit):
            self._sync_partial()
            raise
        except Exception:
            self._sync_partial()
            _raise("write_failed")

        try:
            self._check_identity(size=len(raw))
            self._require_admission_absent()
            chunks = []
            offset = 0
            while offset < len(raw):
                chunk = os.pread(self._fd, len(raw) - offset, offset)
                if not chunk:
                    _raise("verification_failed")
                chunks.append(chunk)
                offset += len(chunk)
            if b"".join(chunks) != raw:
                _raise("verification_failed")
            self._check_identity(size=len(raw))
            self._require_admission_absent()
        except (KeyboardInterrupt, SystemExit, MeasurementSlotError):
            raise
        except Exception:
            _raise("verification_failed")
        return {"path": self._measurement_path, "size_bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest()}

    def _sync_partial(self):
        for fd in (self._fd, self._parent_fd):
            try:
                os.fsync(fd)
            except OSError:
                pass

    def close(self):
        self._require_owner_open()
        fd, parent_fd = self._fd, self._parent_fd
        self._fd = None
        self._parent_fd = None
        self._closed = True
        try:
            os.close(fd)
        finally:
            os.close(parent_fd)

    def __enter__(self):
        self._require_owner_open()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
        return False


def _validate_binding(expected_commit, expected_source_set_sha256):
    if (type(expected_commit) is not str or _COMMIT.fullmatch(expected_commit) is None
            or type(expected_source_set_sha256) is not str
            or _SHA256.fullmatch(expected_source_set_sha256) is None):
        _raise("invalid_binding")


def _reserve_at(measurement_path, admission_path, *, expected_commit,
                expected_source_set_sha256):
    _validate_binding(expected_commit, expected_source_set_sha256)
    if (os.path.dirname(measurement_path) != os.path.dirname(admission_path)
            or os.path.basename(measurement_path) != MEASUREMENT_BASENAME
            or os.path.basename(admission_path) != ADMISSION_BASENAME):
        _raise("invalid_parent")
    parent_fd, measurement_name = _open_parent(measurement_path)
    fd = None
    try:
        if _entry(parent_fd, ADMISSION_BASENAME) is not None:
            _raise("admission_present")
        if _entry(parent_fd, MEASUREMENT_BASENAME) is not None:
            _raise("measurement_present")
        try:
            space = os.fstatvfs(parent_fd)
        except OSError:
            _raise("invalid_parent")
        if space.f_bavail * space.f_frsize < MIN_FREE_BYTES + MEASUREMENT_MAX_BYTES:
            _raise("free_space_floor")
        try:
            fd = os.open(measurement_name,
                         os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                         0o600, dir_fd=parent_fd)
            info = os.fstat(fd)
            if not _valid_private_regular(info, size=0):
                _raise("slot_identity_changed")
            os.fsync(fd)
            os.fsync(parent_fd)
            current = os.stat(measurement_name, dir_fd=parent_fd, follow_symlinks=False)
            if _signature(current) != _signature(info) or current.st_size != 0:
                _raise("slot_identity_changed")
        except (KeyboardInterrupt, SystemExit, MeasurementSlotError):
            raise
        except FileExistsError:
            _raise("measurement_present")
        except Exception:
            _raise("verification_failed")
        try:
            reservation = MeasurementReservation(
                parent_fd=parent_fd, measurement_name=measurement_name,
                admission_name=ADMISSION_BASENAME, measurement_path=measurement_path,
                expected_commit=expected_commit,
                expected_source_set_sha256=expected_source_set_sha256, slot_info=info)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            _raise("verification_failed")
        reservation._fd = fd
        reservation._check_identity(size=0)
        reservation._require_admission_absent()
        return reservation
    except BaseException:
        if fd is not None:
            os.close(fd)
        os.close(parent_fd)
        # A created zero/partial slot is deliberately never unlinked.
        raise


def reserve_measurement(expected_commit, expected_source_set_sha256):
    """Consume the one fixed production M slot; never accepts another path."""
    import native_storage_authority as authority
    import process_supervision as supervision
    if (authority.EVIDENCE_MAX != MEASUREMENT_MAX_BYTES
            or authority.MIN_FREE_BYTES != MIN_FREE_BYTES
            or os.path.basename(supervision.LAYOUT_EVIDENCE_PATH) != MEASUREMENT_BASENAME
            or os.path.basename(supervision.STORAGE_ADMISSION_PATH) != ADMISSION_BASENAME):
        _raise("invalid_parent")
    return _reserve_at(
        supervision.LAYOUT_EVIDENCE_PATH, supervision.STORAGE_ADMISSION_PATH,
        expected_commit=expected_commit,
        expected_source_set_sha256=expected_source_set_sha256)


def _reserve_fixture(parent, expected_commit, expected_source_set_sha256):
    """Private tiny-test primitive; production parents and alternate names forbidden."""
    if type(parent) not in (str, os.PathLike):
        _raise("invalid_parent")
    parent = os.fspath(parent)
    try:
        actual = os.path.realpath(parent)
        big = os.path.realpath(BIG_TMP)
        production = None
        import process_supervision as supervision
        production = os.path.dirname(supervision.LAYOUT_EVIDENCE_PATH)
        if (not os.path.isabs(parent) or os.path.normpath(parent) != parent
                or actual != parent or actual == big
                or os.path.commonpath((big, actual)) != big
                or actual == production
                or os.stat(actual).st_dev != os.stat(big).st_dev):
            _raise("invalid_parent")
    except MeasurementSlotError:
        raise
    except (OSError, ValueError):
        _raise("invalid_parent")
    return _reserve_at(
        actual + "/" + MEASUREMENT_BASENAME,
        actual + "/" + ADMISSION_BASENAME,
        expected_commit=expected_commit,
        expected_source_set_sha256=expected_source_set_sha256)


def main(argv=None):
    print("native_storage_measurement_slot: inert; no slot reserved or written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
