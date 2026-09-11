"""Cooperative I7 process guard; no launch, hard preemption, child, or phase-controller claims."""
from __future__ import annotations

import copy
import math
import os
from pathlib import Path
import resource
import stat
import subprocess
import sys
import time
from types import MappingProxyType

import torch

from cuda_identity import canonical_cuda_uuid

SCIENTIFIC = "scientific_mnist_current32_v1"
FIXTURE = "fixture_tiny_mlp_cpu_v1"
GIB = 1 << 30
PHASE_CAPS = MappingProxyType({
    "development": MappingProxyType({"wall_seconds":180.0, "cpu_seconds":None}),
    "primary": MappingProxyType({"wall_seconds":600.0, "cpu_seconds":None}),
    "sensitivity": MappingProxyType({"wall_seconds":240.0, "cpu_seconds":None}),
    "audit": MappingProxyType({"wall_seconds":600.0, "cpu_seconds":600.0}),
})
RSS_CAP = 12 * GIB
CUDA_ALLOCATED_CAP = 8 * GIB
TORCH_VERSION = "2.11.0+cu128"
CUDA_DEVICE_INDEX = 0
CUDA_DEVICE_NAME = "NVIDIA GeForce RTX 3090"
BIG_VOLUME = MappingProxyType({"target":"/private-artifacts/storage", "source":"/dev/RECONFIGURE_FOR_LOCAL_STORAGE",
                               "uuid":"00000000-0000-4000-8000-000000000000"})


class GuardError(RuntimeError):
    pass


class GuardTerminal(GuardError):
    def __init__(self, record):
        super().__init__("runtime guard entered terminal failure state")
        self.record = copy.deepcopy(record)


def _finite_number(value, name):
    if type(value) not in (int, float) or type(value) is bool:
        raise GuardError(f"{name} must be a finite number")
    try:
        converted = float(value)
    except (OverflowError, ValueError) as exc:
        raise GuardError(f"{name} must be a finite number") from exc
    if not math.isfinite(converted):
        raise GuardError(f"{name} must be a finite number")
    return converted


def _byte_count(value, name):
    if type(value) is not int or type(value) is bool or value < 0:
        raise GuardError(f"{name} must be a nonnegative exact integer")
    return value


def _native_rss_probe():
    if not sys.platform.startswith("linux"):
        raise GuardError("scientific RSS conversion requires Linux ru_maxrss KiB semantics")
    return _byte_count(int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024), "peak RSS")


def _native_cuda_identity():
    if not torch.cuda.is_initialized():
        raise GuardError("scientific CUDA context must already be initialized")
    if str(torch.__version__) != TORCH_VERSION:
        raise GuardError("scientific Torch version mismatch")
    if torch.cuda.current_device() != CUDA_DEVICE_INDEX:
        raise GuardError("scientific CUDA device index mismatch")
    props = torch.cuda.get_device_properties(CUDA_DEVICE_INDEX)
    if props.name != CUDA_DEVICE_NAME:
        raise GuardError("scientific CUDA device identity mismatch")
    try:
        stable_identity = canonical_cuda_uuid(getattr(props, "uuid", None))
    except ValueError as exc:
        raise GuardError("scientific CUDA UUID is missing or malformed") from exc
    return {"index":CUDA_DEVICE_INDEX, "name":props.name,
            "total_memory":int(props.total_memory),
            "stable_identity":stable_identity,
            "stable_identity_verified":True}


def _native_cuda_probe():
    identity = _native_cuda_identity()
    return {"allocated_bytes":int(torch.cuda.max_memory_allocated(CUDA_DEVICE_INDEX)),
            "reserved_bytes":int(torch.cuda.max_memory_reserved(CUDA_DEVICE_INDEX)),
            "device":identity}


def _zero_cuda_probe():
    return {"allocated_bytes":0, "reserved_bytes":0, "device":None}


def _scientific_audit_cuda_probe():
    if torch.cuda.is_initialized():
        raise GuardError("CPU-only scientific audit observed an initialized CUDA context")
    return _zero_cuda_probe()


def _limits(phase, overrides=None):
    if phase not in PHASE_CAPS:
        raise GuardError("unknown runtime phase")
    row = dict(PHASE_CAPS[phase], rss_bytes=RSS_CAP,
               cuda_allocated_bytes=CUDA_ALLOCATED_CAP)
    if overrides is not None:
        if type(overrides) is not dict or set(overrides) != set(row):
            raise GuardError("fixture limits must have the exact limit keys")
        row = dict(overrides)
    for key in ("wall_seconds", "rss_bytes", "cuda_allocated_bytes"):
        if _finite_number(row[key], key) < 0:
            raise GuardError(f"negative {key}")
    if row["cpu_seconds"] is not None and _finite_number(row["cpu_seconds"], "cpu_seconds") < 0:
        raise GuardError("negative cpu_seconds")
    row["wall_seconds"] = float(row["wall_seconds"])
    row["cpu_seconds"] = None if row["cpu_seconds"] is None else float(row["cpu_seconds"])
    row["rss_bytes"] = _byte_count(row["rss_bytes"], "rss_bytes")
    row["cuda_allocated_bytes"] = _byte_count(row["cuda_allocated_bytes"],
                                                "cuda_allocated_bytes")
    return row


class RuntimeGuard:
    """Caller-invoked checks; one blocking operation may overrun before the next check.

    Optional entry origins and their process ID are trusted readings captured
    together earlier by the same launcher process from ``time.monotonic`` and
    ``time.process_time``.  They extend accounting to launcher-entry work but
    are neither serialized nor a cross-process CPU-time offset.
    """

    def __init__(self, phase, *, profile=SCIENTIFIC, wall_clock=None, cpu_clock=None,
                 rss_probe=None, cuda_probe=None, limits=None,
                 entry_wall_origin=None, entry_cpu_origin=None,
                 entry_process_id=None):
        if profile not in (SCIENTIFIC, FIXTURE):
            raise GuardError("unknown guard profile")
        has_wall_origin = entry_wall_origin is not None
        has_cpu_origin = entry_cpu_origin is not None
        if has_wall_origin != has_cpu_origin:
            raise GuardError("entry wall and CPU origins must be supplied together")
        if has_wall_origin != (entry_process_id is not None):
            raise GuardError("entry clock origins and process ID must be supplied together")
        process_id = os.getpid()
        if type(process_id) is not int or type(process_id) is bool or process_id <= 0:
            raise GuardError("current process ID is invalid")
        entry_origins = None
        if has_wall_origin:
            if (type(entry_process_id) is not int or type(entry_process_id) is bool
                    or entry_process_id <= 0):
                raise GuardError("entry process ID must be a positive exact integer")
            if entry_process_id != process_id:
                raise GuardError("entry clock origins were captured in a different process")
            wall_origin = _finite_number(entry_wall_origin, "entry wall origin")
            cpu_origin = _finite_number(entry_cpu_origin, "entry CPU origin")
            if wall_origin < 0 or cpu_origin < 0:
                raise GuardError("entry clock origins must be nonnegative")
            entry_origins = (wall_origin, cpu_origin)
        supplied = (wall_clock, cpu_clock, rss_probe, cuda_probe, limits)
        if profile == SCIENTIFIC and any(item is not None for item in supplied):
            raise GuardError("scientific guard forbids fixture probes or limit overrides")
        self.profile, self.phase = profile, phase
        self._phase_limits = MappingProxyType(
            _limits(phase, limits if profile == FIXTURE else None))
        self.runtime = {"torch_version":str(torch.__version__),
                        "cuda_build":None if torch.version.cuda is None else str(torch.version.cuda)}
        if profile == SCIENTIFIC:
            if str(torch.__version__) != TORCH_VERSION:
                raise GuardError("scientific Torch version mismatch")
            self._wall, self._cpu, self._rss = time.monotonic, time.process_time, _native_rss_probe
            if phase == "audit":
                if torch.cuda.is_initialized():
                    raise GuardError("CPU-only scientific audit requires CUDA uninitialized")
                self._cuda, self.device = _scientific_audit_cuda_probe, None
                self.cuda_initialized = False
            else:
                self._cuda, self.device = _native_cuda_probe, _native_cuda_identity()
                self.cuda_initialized = True
        else:
            self._wall = time.monotonic if wall_clock is None else wall_clock
            self._cpu = time.process_time if cpu_clock is None else cpu_clock
            self._rss = _native_rss_probe if rss_probe is None else rss_probe
            self._cuda = _zero_cuda_probe if cuda_probe is None else cuda_probe
            self.device = None
            self.cuda_initialized = False
        if not all(callable(probe) for probe in (self._wall, self._cpu, self._rss, self._cuda)):
            raise GuardError("runtime clocks and resource probes must be callable")
        wall_observed = _finite_number(self._wall(), "wall clock")
        cpu_observed = _finite_number(self._cpu(), "CPU clock")
        if entry_origins is not None:
            if entry_origins[0] > wall_observed or entry_origins[1] > cpu_observed:
                raise GuardError("entry clock origin is later than constructor observation")
        self._supplied_entry_origins = entry_origins
        self._process_id = process_id
        self._wall_start, self._cpu_start = (
            entry_origins if entry_origins is not None else (wall_observed, cpu_observed))
        self._wall_last, self._cpu_last = wall_observed, cpu_observed
        self._peak_rss = self._peak_allocated = self._peak_reserved = 0
        self._checks, self._last, self._failure = 0, None, None

    def _terminal(self, record):
        record["status"] = "failed"
        self._last = self._failure = copy.deepcopy(record)
        raise GuardTerminal(record)

    @property
    def limits(self):
        return dict(self._phase_limits)

    @property
    def entry_origins(self):
        if self._supplied_entry_origins is None:
            return None
        return {"wall_origin":self._supplied_entry_origins[0],
                "cpu_origin":self._supplied_entry_origins[1]}

    def check(self, stage):
        if self._failure is not None:
            raise GuardTerminal(self._failure)
        if type(stage) is not str or not stage or len(stage) > 96 or not stage.isascii():
            self._checks += 1
            return self._terminal({
                "check_index":self._checks, "stage":"<invalid-stage>",
                "wall_seconds":None, "cpu_seconds":None,
                "peak_rss_bytes":self._peak_rss,
                "peak_cuda_allocated_bytes":self._peak_allocated,
                "peak_cuda_reserved_bytes":self._peak_reserved,
                "device":copy.deepcopy(self.device),
                "runtime":copy.deepcopy(self.runtime),
                "cuda_initialized":self.cuda_initialized, "status":"checking",
                "violations":["invalid_stage"], "probe_error":"invalid_stage",
            })
        self._checks += 1
        record = {"check_index":self._checks, "stage":stage, "wall_seconds":None,
                  "cpu_seconds":None, "peak_rss_bytes":self._peak_rss,
                  "peak_cuda_allocated_bytes":self._peak_allocated,
                  "peak_cuda_reserved_bytes":self._peak_reserved,
                  "device":copy.deepcopy(self.device),
                  "runtime":copy.deepcopy(self.runtime),
                  "cuda_initialized":self.cuda_initialized, "status":"checking",
                  "violations":[], "probe_error":None}
        try:
            if os.getpid() != self._process_id:
                record["probe_error"] = "process_identity_changed"
                return self._terminal(record)
            wall_now = _finite_number(self._wall(), "wall clock")
            cpu_now = _finite_number(self._cpu(), "CPU clock")
            record["wall_seconds"] = wall_now - self._wall_start
            record["cpu_seconds"] = cpu_now - self._cpu_start
            if wall_now < self._wall_last or cpu_now < self._cpu_last:
                record["probe_error"] = "clock_regression"
                return self._terminal(record)
            self._wall_last, self._cpu_last = wall_now, cpu_now
            self._peak_rss = max(self._peak_rss, _byte_count(self._rss(), "peak RSS"))
            cuda = self._cuda()
            if type(cuda) is not dict or set(cuda) != {"allocated_bytes", "reserved_bytes", "device"}:
                raise GuardError("CUDA probe returned wrong shape")
            allocated = _byte_count(cuda["allocated_bytes"], "CUDA allocation")
            reserved = _byte_count(cuda["reserved_bytes"], "CUDA reservation")
            if allocated > reserved:
                raise GuardError("CUDA allocation exceeds reservation")
            if self.profile == SCIENTIFIC and cuda["device"] != self.device:
                raise GuardError("scientific CUDA identity changed")
            if self.profile == FIXTURE and cuda["device"] is not None:
                if type(cuda["device"]) is not dict:
                    raise GuardError("fixture CUDA identity must be a mapping or None")
                self.device = copy.deepcopy(cuda["device"])
            self._peak_allocated = max(self._peak_allocated, allocated)
            self._peak_reserved = max(self._peak_reserved, reserved)
            record.update(peak_rss_bytes=self._peak_rss,
                          peak_cuda_allocated_bytes=self._peak_allocated,
                          peak_cuda_reserved_bytes=self._peak_reserved,
                          device=copy.deepcopy(self.device))
        except GuardTerminal:
            raise
        except BaseException as exc:
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            record["probe_error"] = f"{type(exc).__name__}:{exc}"[:256]
            return self._terminal(record)
        if record["wall_seconds"] > self._phase_limits["wall_seconds"]:
            record["violations"].append("wall_seconds")
        cpu_cap = self._phase_limits["cpu_seconds"]
        if cpu_cap is not None and record["cpu_seconds"] > cpu_cap:
            record["violations"].append("cpu_seconds")
        if self._peak_rss > self._phase_limits["rss_bytes"]:
            record["violations"].append("peak_rss_bytes")
        if self._peak_allocated > self._phase_limits["cuda_allocated_bytes"]:
            record["violations"].append("peak_cuda_allocated_bytes")
        if record["violations"]:
            return self._terminal(record)
        record["status"] = "pass"
        self._last = copy.deepcopy(record)
        return copy.deepcopy(record)

    __call__ = check

    def failure_info(self):
        return copy.deepcopy(self._failure)

    def summary(self):
        return {"profile":self.profile, "phase":self.phase, "limits":dict(self._phase_limits),
                "check_count":self._checks, "last_record":copy.deepcopy(self._last),
                "terminal":self._failure is not None,
                "cuda_initialized":self.cuda_initialized,
                "runtime":copy.deepcopy(self.runtime),
                "rss_scope":"RUSAGE_SELF process-lifetime peak, Linux KiB, all process threads",
                "cpu_scope":"process_time for this process; child processes excluded",
                "preemption":"cooperative checks only"}


def _path_without_symlinks(path):
    absolute = Path(os.path.abspath(os.fspath(path)))
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        try:
            info = os.lstat(current)
        except OSError as exc:
            raise GuardError(f"volume path inspection failed: {exc}") from exc
        if stat.S_ISLNK(info.st_mode):
            raise GuardError("verified volume path contains a symlink")
    try:
        final_info = os.stat(absolute)
    except OSError as exc:
        raise GuardError(f"volume path inspection failed: {exc}") from exc
    if not stat.S_ISDIR(final_info.st_mode):
        raise GuardError("verified volume path is not a directory")
    return absolute


def verify_big_volume(path, *, profile=SCIENTIFIC, runner=None, stat_probe=None, expected=None):
    """Read-only mount identity check; does not create a study root or authorize launch."""
    if profile not in (SCIENTIFIC, FIXTURE):
        raise GuardError("unknown volume-check profile")
    if profile == SCIENTIFIC and any(item is not None for item in (runner, stat_probe, expected)):
        raise GuardError("scientific volume check forbids fixture overrides")
    runner = subprocess.run if runner is None else runner
    stat_probe = os.stat if stat_probe is None else stat_probe
    expected = dict(BIG_VOLUME if expected is None else expected)
    if set(expected) != {"target", "source", "uuid"} or any(type(x) is not str for x in expected.values()):
        raise GuardError("invalid expected mount identity")
    absolute = _path_without_symlinks(path)
    target = Path(os.path.abspath(expected["target"]))
    try:
        absolute.relative_to(target)
    except ValueError as exc:
        raise GuardError("path is outside the expected mount root") from exc
    try:
        result = runner(["findmnt", "--noheadings", "--target", str(absolute),
                         "--output", "TARGET,SOURCE,UUID,MAJ:MIN"],
                        capture_output=True, text=True, check=True, timeout=5)
    except subprocess.TimeoutExpired as exc:
        raise GuardError("findmnt probe timed out") from exc
    except BaseException as exc:
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        raise GuardError(f"findmnt probe failed: {exc}") from exc
    if type(result.stdout) is not str:
        raise GuardError("findmnt returned non-text output")
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    fields = lines[0].split() if len(lines) == 1 else []
    if len(fields) != 4:
        raise GuardError("findmnt did not return one exact four-field row")
    found_target, found_source, found_uuid, found_device = fields
    if (found_target, found_source, found_uuid) != (str(target), expected["source"], expected["uuid"]):
        raise GuardError("wrong mount target/source/UUID")
    try:
        major_text, minor_text = found_device.split(":", 1)
        if not all(text.isascii() and text.isdecimal() and str(int(text)) == text
                   for text in (major_text, minor_text)):
            raise ValueError("noncanonical device number")
        mount_device = (int(major_text), int(minor_text))
    except (ValueError, TypeError) as exc:
        raise GuardError("invalid findmnt device number") from exc
    try:
        info = stat_probe(absolute)
    except OSError as exc:
        raise GuardError(f"stat device probe failed: {exc}") from exc
    if (not hasattr(info, "st_dev") or type(info.st_dev) is not int
            or info.st_dev < 0):
        raise GuardError("stat probe returned invalid device metadata")
    stat_device = (os.major(info.st_dev), os.minor(info.st_dev))
    if stat_device != mount_device:
        raise GuardError("stat device disagrees with findmnt")
    return {"path":str(absolute), "target":found_target, "source":found_source,
            "uuid":found_uuid, "major":stat_device[0], "minor":stat_device[1],
            "read_only_verification":True}
