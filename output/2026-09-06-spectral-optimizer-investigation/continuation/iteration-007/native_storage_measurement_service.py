"""Fixed CPU-measurement service controls; import/default CLI are inert.

Kernel charged memory is NOT process RSS. Observations describe both, and do
not authenticate history against an actor controlling the review permission.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import re
import selectors
import signal
import stat
import subprocess
import time

UNIT = "i7-native-storage-measurement-002.service"
CGROUP = "/user.slice/user-1000.slice/user@1000.service/app.slice/" + UNIT
CGROUP_ROOT = "/sys/fs/cgroup"
DEPENDENCY_DIRECTORY = "/private-artifacts/.local/lib/python3.12/site-packages"
DEPENDENCY_PACKAGES = ("numpy", "torch")
MEMORY_MAX = 2 << 30
WALL_NS = 120_000_000_000
STDOUT_MAX = 56 << 10  # Leaves 8KiB for the bounded observed-supervision record.
STDERR_MAX = 4 << 10
SAMPLE_NS = 50_000_000
PROPERTIES = {
    "Type": "exec", "ExitType": "main", "Restart": "no",
    "RuntimeMaxUSec": "2min", "RuntimeRandomizedExtraUSec": "0",
    "TimeoutStopUSec": "1s", "RemainAfterExit": "no", "OOMPolicy": "kill",
    "Delegate": "no", "MemoryAccounting": "yes", "MemoryMax": str(MEMORY_MAX),
    "MemorySwapMax": "0", "MemoryZSwapMax": "0", "TasksAccounting": "yes",
    "TasksMax": "128", "LimitCORE": "0",
    "StandardInput": "null", "StandardOutput": "null", "StandardError": "null",
    "KillMode": "control-group", "KillSignal": "15", "FinalKillSignal": "9",
    "SendSIGKILL": "yes", "Transient": "yes", "Slice": "app.slice",
}
RUN_PROPERTIES = (
    "Type=exec", "ExitType=main", "Restart=no", "RuntimeMaxSec=120s",
    "RuntimeRandomizedExtraSec=0", "TimeoutStopSec=1s", "RemainAfterExit=no",
    "OOMPolicy=kill", "Delegate=no", "MemoryAccounting=yes",
    "MemoryMax=2147483648", "MemorySwapMax=0", "MemoryZSwapMax=0",
    "TasksAccounting=yes", "TasksMax=128", "LimitCORE=0",
    "StandardInput=null", "StandardOutput=null", "StandardError=null",
    "KillMode=control-group", "KillSignal=SIGTERM", "FinalKillSignal=SIGKILL",
    "SendSIGKILL=yes", "Slice=app.slice",
)
EVENT_FIELDS = ("low", "high", "max", "oom", "oom_kill", "oom_group_kill")
SNAPSHOT_FIELDS = ("device", "inode", "memory_max_bytes", "memory_swap_max_bytes",
                   "tasks_max", "memory_peak_bytes", "memory_events", "member_pids",
                   "sampled_rss_bytes", "memory_zswap_max_bytes", "memory_oom_group",
                   "pids_max_events", "nr_descendants")
SUPERVISION_FIELDS = (
    "schema", "mode", "unit", "control_group", "applied_properties", "process_environment",
    "controller_pid", "invocation_id", "service_active_enter_ns",
    "worker_pid", "worker_start_ticks", "worker_process_group",
    "worker_exit_code", "candidate_size_bytes", "candidate_sha256", "stderr_bytes",
    "elapsed_wall_ns", "max_poll_wait_ns", "sampled_aggregate_peak_rss_bytes",
    "rss_scope", "wall_scope", "before", "after", "worker_group_gone",
)


class MeasurementServiceError(RuntimeError):
    pass


def _need(condition, code):
    if not condition:
        raise MeasurementServiceError(code)


def _read(path, cap=8192):
    with open(path, "rb") as handle:
        raw = handle.read(cap + 1)
    _need(len(raw) <= cap, "observation_cap")
    return raw.decode("ascii")


def _membership(pid):
    _need(type(pid) is int and pid > 0, "invalid_pid")
    lines = _read(f"/proc/{pid}/cgroup").splitlines()
    _need(len(lines) == 1 and lines[0].startswith("0::/"), "unified_cgroup_required")
    path = lines[0][3:]
    _need(os.path.normpath(path) == path and not path.startswith("//"), "cgroup_path")
    return path


def process_environment():
    """Exact public nonsecret CPU/Git/DBus environment; no inherited settings."""
    return {
        "PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C", "TZ": "UTC",
        "XDG_RUNTIME_DIR": "/run/user/1000",
        "DBUS_SESSION_BUS_ADDRESS": "unix:path=/run/user/1000/bus",
        "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_SYSTEM": "/dev/null", "GIT_OPTIONAL_LOCKS": "0",
        "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "TMPDIR": "/tmp/spectral-experiment-artifacts",
        "XDG_CACHE_HOME": "/tmp/spectral-experiment-artifacts", "PYTHONDONTWRITEBYTECODE": "1",
    }


def validate_dependency_directory(path=DEPENDENCY_DIRECTORY):
    """Validate the one explicit import directory without importing its packages.

    The isolated interpreter intentionally does not discover the user site.  The
    worker appends this reviewed directory explicitly; appending a path does not
    process its ``.pth`` files.
    """
    _need(type(path) is str and path == DEPENDENCY_DIRECTORY
          and path.startswith("/") and os.path.normpath(path) == path
          and os.path.realpath(path) == path, "dependency_directory")
    try:
        directory = os.stat(path, follow_symlinks=False)
        _need(stat.S_ISDIR(directory.st_mode) and directory.st_uid == os.getuid()
              and directory.st_gid == os.getgid()
              and not directory.st_mode & stat.S_IWOTH, "dependency_directory")
        for package in DEPENDENCY_PACKAGES:
            package_path = path + "/" + package
            init_path = package_path + "/__init__.py"
            package_info = os.stat(package_path, follow_symlinks=False)
            init_info = os.stat(init_path, follow_symlinks=False)
            _need(os.path.realpath(package_path) == package_path
                  and os.path.realpath(init_path) == init_path
                  and stat.S_ISDIR(package_info.st_mode)
                  and stat.S_ISREG(init_info.st_mode)
                  and package_info.st_uid == init_info.st_uid == os.getuid()
                  and package_info.st_gid == init_info.st_gid == os.getgid()
                  and not package_info.st_mode & stat.S_IWOTH
                  and not init_info.st_mode & stat.S_IWOTH
                  and init_info.st_nlink == 1 and 0 < init_info.st_size <= 2 << 20,
                  "dependency_directory")
    except MeasurementServiceError:
        raise
    except OSError:
        raise MeasurementServiceError("dependency_directory") from None
    return path


def _run_command(command, *, cap=8192, timeout=5):
    """Torch-free bounded OS-supervision command, no shell/inherited env."""
    _need(type(command) is list and command and command[0].startswith("/")
          and type(cap) is int and 0 < cap <= 8192, "os_command")
    process = subprocess.Popen(command, cwd="/", env=process_environment(),
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        close_fds=True, start_new_session=True)
    selector = selectors.DefaultSelector()
    buffers = {process.stdout: bytearray(), process.stderr: bytearray()}
    deadline = time.monotonic() + timeout
    try:
        for stream in buffers:
            os.set_blocking(stream.fileno(), False)
            selector.register(stream, selectors.EVENT_READ)
        while selector.get_map():
            _need(time.monotonic() < deadline, "os_command_timeout")
            for key, _ in selector.select(min(.05, max(0, deadline-time.monotonic()))):
                data = os.read(key.fileobj.fileno(), min(4096, cap-len(buffers[key.fileobj])+1))
                if not data:
                    selector.unregister(key.fileobj)
                else:
                    buffers[key.fileobj].extend(data)
                    _need(len(buffers[key.fileobj]) <= cap, "os_command_output")
        _need(process.wait(timeout=max(.001, deadline-time.monotonic())) == 0, "os_command_exit")
        return bytes(buffers[process.stdout])
    finally:
        try:
            if process.returncode is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except OSError:
                    pass
                finally:
                    try:
                        process.wait(timeout=1)
                    except subprocess.TimeoutExpired:
                        pass
        finally:
            selector.close()
            for stream in buffers:
                stream.close()


def assert_service_member(*, controller_pid, expected_cgroup):
    _need(expected_cgroup == CGROUP and os.getuid() == 1000, "fixed_cgroup_required")
    _need(_membership(os.getpid()) == expected_cgroup
          and _membership(controller_pid) == expected_cgroup, "cgroup_membership")
    return expected_cgroup


def _show(unit=UNIT):
    # During execution this bounded helper also inherits the service cgroup.
    names = (*PROPERTIES, "Id", "ControlGroup", "MainPID", "ActiveState", "SubState",
             "InvocationID", "ActiveEnterTimestampMonotonic")
    raw = _run_command(["/usr/bin/systemctl", "--user", "show", unit,
                       "--no-pager", "--property=" + ",".join(names)], cap=8192)
    result = {}
    for line in raw.decode("utf-8").splitlines():
        key, sep, value = line.partition("=")
        _need(sep and key not in result and key in names, "service_properties")
        result[key] = value
    _need(set(result) == set(names), "service_properties")
    return result


def validate_live_properties(value, *, controller_pid, cgroup=CGROUP, unit=UNIT):
    _need(type(value) is dict, "service_properties")
    _need(all(value.get(k) == v for k, v in PROPERTIES.items()), "service_controls")
    _need(value.get("Id") == unit and value.get("ControlGroup") == cgroup
          and value.get("MainPID") == str(controller_pid)
          and value.get("ActiveState") == "active"
          and value.get("SubState") == "running", "service_identity")
    return dict(PROPERTIES)


class ServiceContext:
    """Held cgroup identity; caller owns close, no process work at construction."""

    def __init__(self, cgroup=CGROUP):
        self.cgroup = cgroup
        self.fd = os.open(CGROUP_ROOT + cgroup, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        info = os.fstat(self.fd)
        self.identity = info.st_dev, info.st_ino

    def snapshot(self):
        current = os.stat(CGROUP_ROOT + self.cgroup, follow_symlinks=False)
        _need((current.st_dev, current.st_ino) == self.identity, "cgroup_replaced")
        return snapshot(self.cgroup, dir_fd=self.fd)

    def close(self):
        os.close(self.fd)


def assert_controller():
    import time
    _need(dict(os.environ) == process_environment(), "exact_process_environment")
    assert_service_member(controller_pid=os.getpid(), expected_cgroup=CGROUP)
    live = _show()
    properties = validate_live_properties(live, controller_pid=os.getpid())
    origin = int(live["ActiveEnterTimestampMonotonic"]) * 1000
    _need(0 < origin <= time.monotonic_ns(), "service_origin")
    _need(re.fullmatch("[0-9a-f]{32}", live["InvocationID"]) is not None, "invocation_id")
    context = ServiceContext()
    try:
        before = context.snapshot()
        _need(before["member_pids"] == [os.getpid()], "unexpected_service_member")
        _need(not any(before["memory_events"].values())
              and before["pids_max_events"] == 0, "prior_resource_events")
        return context, properties, before, live["InvocationID"], origin
    except BaseException:
        context.close()
        raise


def verify_big_tmp():
    """Read-only bounded mount/free-floor check; no cache/directory creation."""
    import native_control as control
    path = Path(process_environment()["TMPDIR"])
    _need(str(path.resolve()) == str(path), "big_tmp_symlink")
    raw = _run_command(["/usr/bin/findmnt", "--noheadings", "--target", str(path),
                        "--output", "TARGET,SOURCE,UUID,MAJ:MIN"], cap=1024)
    rows = [line.split() for line in raw.decode("ascii").splitlines() if line.strip()]
    _need(len(rows) == 1 and len(rows[0]) == 4, "big_tmp_mount")
    target, source, uuid, device = rows[0]
    _need((target, source, uuid) == tuple(control.BIG_VOLUME[key] for key in ("target", "source", "uuid")),
          "big_tmp_mount")
    info = path.stat()
    _need(device == f"{os.major(info.st_dev)}:{os.minor(info.st_dev)}"
          and path.is_dir(), "big_tmp_device")
    space = os.statvfs(path)
    _need(space.f_bavail * space.f_frsize >= 1 << 30, "big_tmp_free_floor")


def _rss(pid):
    try:
        fields = dict(line.split(":", 1) for line in _read(f"/proc/{pid}/status").splitlines()
                      if ":" in line)
        if fields.get("State", "").strip().startswith(("Z", "X")):
            return 0
        parts = fields["VmRSS"].split()
        _need(len(parts) == 2 and parts[1] == "kB", "rss_units")
        return int(parts[0]) * 1024
    except FileNotFoundError:
        return 0  # Exited between the membership and status reads, not a live sample.


def snapshot(cgroup=CGROUP, *, dir_fd):
    _need(type(cgroup) is str and cgroup.startswith("/user.slice/")
          and os.path.normpath(cgroup) == cgroup, "cgroup_path")
    root = Path(CGROUP_ROOT + cgroup)
    info = os.fstat(dir_fd)
    def read(name):
        fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=dir_fd)
        try:
            data = os.read(fd, 8193)
            _need(len(data) <= 8192, "cgroup_observation_cap")
            return data.decode("ascii")
        finally:
            os.close(fd)
    _need(not root.is_symlink() and read("cgroup.type").strip() == "domain",
          "cgroup_domain")
    stats = dict(line.split() for line in read("cgroup.stat").splitlines())
    _need(int(stats["nr_descendants"]) == 0, "nested_cgroup")
    events = {}
    for line in read("memory.events").splitlines():
        key, value = line.split()
        _need(key in EVENT_FIELDS and key not in events, "memory_events")
        events[key] = int(value)
    _need(set(events) == set(EVENT_FIELDS), "memory_events")
    pids = sorted(set(int(pid) for pid in read("cgroup.procs").split()))
    _need(len(pids) <= 128 and all(pid > 0 for pid in pids), "cgroup_tasks")
    result = dict(device=info.st_dev, inode=info.st_ino,
        memory_max_bytes=int(read("memory.max")),
        memory_swap_max_bytes=int(read("memory.swap.max")),
        tasks_max=int(read("pids.max")),
        memory_peak_bytes=int(read("memory.peak")),
        memory_events={key: events[key] for key in EVENT_FIELDS},
        member_pids=pids, sampled_rss_bytes=sum(_rss(pid) for pid in pids),
        memory_zswap_max_bytes=int(read("memory.zswap.max")),
        memory_oom_group=int(read("memory.oom.group")),
        pids_max_events=int(dict(line.split() for line in read("pids.events").splitlines())["max"]),
        nr_descendants=int(stats["nr_descendants"]))
    _need(result["memory_max_bytes"] == MEMORY_MAX
          and result["memory_swap_max_bytes"] == 0 and result["tasks_max"] == 128
          and result["memory_zswap_max_bytes"] == 0 and result["memory_oom_group"] == 1,
          "kernel_controls")
    return result


def validate_record(record, candidate):
    """Strict historical shape/cross-binding only; no live OS queries."""
    import native_storage_authority as authority
    need, keys, integer = authority._need, authority._keys, authority._integer
    keys(record, SUPERVISION_FIELDS, "measurement supervision")
    need(record["schema"] == "i7_storage_service_observation_v1"
         and record["mode"] == "systemd_user_service_cgroup_v2"
         and record["unit"] == UNIT and record["control_group"] == CGROUP,
         "supervision identity differs")
    keys(record["applied_properties"], PROPERTIES, "applied service properties")
    need(record["applied_properties"] == PROPERTIES, "service controls differ")
    keys(record["process_environment"], process_environment(), "process environment")
    need(record["process_environment"] == process_environment(), "process environment differs")
    need(type(record["invocation_id"]) is str
         and re.fullmatch("[0-9a-f]{32}", record["invocation_id"]) is not None,
         "service invocation differs")
    integer(record["service_active_enter_ns"], "service origin", minimum=1)
    for field in ("controller_pid", "worker_pid", "worker_start_ticks", "worker_process_group",
                  "candidate_size_bytes", "elapsed_wall_ns", "max_poll_wait_ns",
                  "sampled_aggregate_peak_rss_bytes"):
        integer(record[field], field, minimum=1)
    integer(record["stderr_bytes"], "stderr bytes", maximum=STDERR_MAX)
    integer(record["worker_exit_code"], "worker exit", maximum=1)
    need(record["worker_pid"] != record["controller_pid"]
         and record["worker_process_group"] == record["worker_pid"]
         and record["worker_pid"] == candidate["observed_resources"]["runner_pid"]
         and record["worker_exit_code"] == (0 if candidate["status"] == "complete" else 1),
         "worker/exit binding differs")
    need(record["worker_group_gone"] is True
         and record["max_poll_wait_ns"] == SAMPLE_NS
         and record["rss_scope"] == "sum_cgroup_member_VmRSS_sampled_not_kernel_charge"
         and record["wall_scope"] == "service_activation_through_validation_before_slot_finalize",
         "supervision scope/cleanup differs")
    for label in ("before", "after"):
        snap = record[label]
        keys(snap, SNAPSHOT_FIELDS, "cgroup snapshot")
        for field in SNAPSHOT_FIELDS[:6]:
            integer(snap[field], field)
        integer(snap["sampled_rss_bytes"], "sampled RSS", minimum=1)
        for field in SNAPSHOT_FIELDS[9:]:
            integer(snap[field], field)
        keys(snap["memory_events"], EVENT_FIELDS, "memory events")
        for number in snap["memory_events"].values():
            integer(number, "memory event")
        need(snap["memory_max_bytes"] == MEMORY_MAX
             and snap["memory_swap_max_bytes"] == 0 and snap["tasks_max"] == 128
             and snap["memory_zswap_max_bytes"] == 0 and snap["memory_oom_group"] == 1
             and snap["nr_descendants"] == 0
             and type(snap["member_pids"]) is list
             and snap["member_pids"] == [record["controller_pid"]]
             and all(type(pid) is int for pid in snap["member_pids"]), "kernel controls/cleanup differ")
    before, after = record["before"], record["after"]
    need((before["device"], before["inode"]) == (after["device"], after["inode"])
         and 0 < before["memory_peak_bytes"] <= after["memory_peak_bytes"]
         and not any(before["memory_events"].values()) and before["pids_max_events"] == 0
         and all(after["memory_events"][k] >= before["memory_events"][k] for k in EVENT_FIELDS),
         "cgroup lifetime differs")
    need(record["elapsed_wall_ns"] >= candidate["observed_resources"]["elapsed_wall_ns"],
         "elapsed observation differs")
    need(record["sampled_aggregate_peak_rss_bytes"] >=
         max(before["sampled_rss_bytes"], after["sampled_rss_bytes"]), "RSS peak differs")
    if candidate["status"] == "complete":
        need(record["elapsed_wall_ns"] <= WALL_NS
             and after["memory_peak_bytes"] <= MEMORY_MAX
             and record["sampled_aggregate_peak_rss_bytes"] <= MEMORY_MAX
             and not any(after["memory_events"].values()) and after["pids_max_events"] == 0
             and record["stderr_bytes"] == 0, "supervised resource conditions failed")
    raw = authority.encode_bounded(candidate, maximum=STDOUT_MAX)
    authority.encode_bounded(record, maximum=8 << 10)
    need(record["candidate_size_bytes"] == len(raw) <= STDOUT_MAX
         and type(record["candidate_sha256"]) is str
         and re.fullmatch("[0-9a-f]{64}", record["candidate_sha256"]) is not None
         and record["candidate_sha256"] == hashlib.sha256(raw).hexdigest(),
         "supervised candidate bytes differ")
    return record


def main(argv=None):
    print("native_storage_measurement_service: inert; no service launch")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
