"""One-shot CPU serializer controller. Default CLI is inert; no native GO.

The explicit service entry works only in the fixed frozen checkout and verified
transient service. It reserves M before spawning work and never creates A.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import selectors
import signal
import subprocess
import sys
import time

import native_storage_measurement_service as service


class MeasurementControllerError(RuntimeError):
    pass


_WORKER_DIAGNOSTIC_PREFIX = b"i7_measurement_worker_failure_v1:"
_WORKER_DIAGNOSTIC_EXITS = {
    "dependency_directory": 20,
    "dependency_import": 21,
    "preflight_unobserved": 22,
    "runtime_unobserved": 23,
    "rows_unobserved": 24,
    "report_unobserved": 25,
    "worker_contract": 26,
    "worker_unobserved": 27,
}


class MeasurementWorkerFailure(MeasurementControllerError):
    def __init__(self, code):
        self.code = code if code in _WORKER_DIAGNOSTIC_EXITS else "worker_unobserved"
        self.exit_status = _WORKER_DIAGNOSTIC_EXITS[self.code]
        super().__init__(self.code)


def _need(condition, code):
    if not condition:
        raise MeasurementControllerError(code)


def _capture(process, context, *, deadline_ns):
    """Bounded dual pipes; retain leader pidfd/PGID ownership until cleanup.

Failure propagates without fabricated candidate facts. The containing service
is the final whole-tree cleanup authority even if this controller is killed.
"""
    import process_supervision as supervision
    pidfd = None
    owned = False
    stdout, stderr, rss_peak = bytearray(), bytearray(), 0
    selector = selectors.DefaultSelector()
    try:
        pidfd = os.pidfd_open(process.pid)
        identity = supervision._proc_identity(process.pid)
        _need(identity is not None and identity["parent"] == os.getpid()
              and identity["group"] == identity["session"] == process.pid, "worker_ownership")
        owned = True
        for stream, label in ((process.stdout, "stdout"), (process.stderr, "stderr")):
            os.set_blocking(stream.fileno(), False)
            selector.register(stream, selectors.EVENT_READ, label)
        while True:
            now = time.monotonic_ns()
            _need(now < deadline_ns, "worker_wall_limit")
            snap = context.snapshot()
            rss_peak = max(rss_peak, snap["sampled_rss_bytes"])
            _need(rss_peak <= service.MEMORY_MAX
                  and snap["memory_peak_bytes"] <= service.MEMORY_MAX
                  and not any(snap["memory_events"].values())
                  and snap["pids_max_events"] == 0, "worker_resource_limit")
            exited = supervision._waitid_exited(pidfd)
            if exited and not selector.get_map():
                break
            for key, _ in selector.select(min(0.05, max(0, (deadline_ns - now) / 1e9))):
                cap = service.STDOUT_MAX if key.data == "stdout" else service.STDERR_MAX
                current = len(stdout) if key.data == "stdout" else len(stderr)
                chunk = os.read(key.fileobj.fileno(), min(4096, cap - current + 1))
                if not chunk:
                    selector.unregister(key.fileobj)
                elif key.data == "stdout":
                    stdout.extend(chunk)
                    _need(len(stdout) <= cap, "candidate_output_limit")
                else:
                    stderr.extend(chunk)
                    _need(len(stderr) <= cap, "stderr_output_limit")
        # Leader remains unreaped, so its PID/PGID cannot be reused.
        _need(not supervision._live_group_members(process.pid), "worker_descendants_remain")
        returncode = process.wait(timeout=1)
        return dict(raw=bytes(stdout), stderr=bytes(stderr), stderr_bytes=len(stderr),
                    worker_pid=process.pid, worker_start_ticks=identity["start"],
                    worker_process_group=process.pid, worker_exit_code=returncode,
                    sampled_aggregate_peak_rss_bytes=rss_peak, worker_group_gone=True)
    finally:
        selector.close()
        # On all exceptional exits terminate only this still-owned child group.
        # Escaped sessions are still in the service cgroup and killed on main exit.
        try:
            if process.returncode is None:
                try:
                    if owned:
                        os.killpg(process.pid, signal.SIGKILL)
                    elif pidfd is not None:
                        signal.pidfd_send_signal(pidfd, signal.SIGKILL)
                    else:
                        process.kill()  # Owned unreaped Popen child.
                except OSError:
                    pass  # Do not mask the original error or skip fd cleanup.
                finally:
                    try:
                        process.wait(timeout=1)
                    except (OSError, subprocess.TimeoutExpired):
                        pass  # Main exits; systemd owns whole-cgroup cleanup.
        finally:
            try:
                for stream in (process.stdout, process.stderr):
                    if stream is not None:
                        try:
                            stream.close()
                        except OSError:
                            pass
            finally:
                if pidfd is not None:
                    os.close(pidfd)


def _worker_command(*, expected_commit, expected_source_set_sha256, controller_pid):
    import process_supervision as supervision
    root = supervision.REPOSITORY_ROOT
    script_dir = root + "/" + supervision.I7_RELATIVE.rstrip("/")
    code = """import os,sys,time
class _DependencyPreflightError(Exception):
    pass
entry_wall_ns=time.monotonic_ns()
entry_cpu_ns=time.process_time_ns()
sys.path.insert(0,sys.argv[1])
try:
    import native_storage_measurement_service as service
    dependency_directory=service.validate_dependency_directory(sys.argv[2])
    if sys.flags.isolated != 1 or dependency_directory in sys.path:
        raise _DependencyPreflightError
    sys.path.append(dependency_directory)
    import native_storage_measurement_worker as worker
    import native_storage_authority as authority
    result=worker.build_measurement_candidate(
        expected_commit=sys.argv[3],expected_source_set_sha256=sys.argv[4],
        repository_root=sys.argv[5],entry_wall_ns=entry_wall_ns,entry_cpu_ns=entry_cpu_ns,
        runner_pid=os.getpid(),controller_pid=int(sys.argv[6]),expected_cgroup=sys.argv[7])
    raw=authority.encode_bounded(result,maximum=56<<10)
    offset=0
    while offset<len(raw):
        offset+=os.write(1,raw[offset:])
    sys.exit(0 if result['status']=='complete' else 1)
except Exception as exc:
    name=type(exc).__name__
    if name=='MeasurementWorkerBootstrapError':
        failure=getattr(exc,'code','worker_unobserved')
    elif name in ('ModuleNotFoundError','ImportError'):
        failure='dependency_import'
    elif name=='MeasurementServiceError':
        failure='dependency_directory'
    elif name=='_DependencyPreflightError':
        failure='preflight_unobserved'
    elif name=='MeasurementWorkerError':
        failure='worker_contract'
    else:
        failure='worker_unobserved'
    allowed=('dependency_directory','dependency_import','preflight_unobserved',
             'runtime_unobserved','rows_unobserved','report_unobserved',
             'worker_contract','worker_unobserved')
    if failure not in allowed:
        failure='worker_unobserved'
    diagnostic=('i7_measurement_worker_failure_v1:'+failure+'\\n').encode('ascii')
    offset=0
    while offset<len(diagnostic):
        offset+=os.write(2,diagnostic[offset:])
    sys.exit(2)
"""
    return [supervision.PYTHON_EXECUTABLE, "-I", "-B", "-c", code, script_dir,
            service.DEPENDENCY_DIRECTORY,
            expected_commit, expected_source_set_sha256, root, str(controller_pid), service.CGROUP]


def _worker_diagnostic(outcome):
    _need(type(outcome) is dict and type(outcome.get("stderr")) is bytes,
          "worker_diagnostic")
    raw = outcome["stderr"]
    if raw.startswith(_WORKER_DIAGNOSTIC_PREFIX) and raw.endswith(b"\n"):
        payload = raw[len(_WORKER_DIAGNOSTIC_PREFIX):-1]
        try:
            code = payload.decode("ascii")
        except UnicodeDecodeError:
            code = "worker_unobserved"
        if code in _WORKER_DIAGNOSTIC_EXITS:
            return code
    return "worker_unobserved"


def _attach_observation(candidate, outcome, *, properties, before, after,
                        invocation_id, service_origin_ns):
    import native_storage_authority as authority
    _need(candidate["supervision"] is None, "candidate_supervision")
    _need(authority.encode_bounded(candidate, maximum=service.STDOUT_MAX) == outcome["raw"],
          "candidate_bytes_changed")
    record = dict(schema="i7_storage_service_observation_v1",
        mode="systemd_user_service_cgroup_v2", unit=service.UNIT, control_group=service.CGROUP,
        applied_properties=properties, process_environment=service.process_environment(),
        controller_pid=os.getpid(), invocation_id=invocation_id,
        service_active_enter_ns=service_origin_ns, worker_pid=outcome["worker_pid"],
        worker_start_ticks=outcome["worker_start_ticks"],
        worker_process_group=outcome["worker_process_group"],
        worker_exit_code=outcome["worker_exit_code"], candidate_size_bytes=len(outcome["raw"]),
        candidate_sha256=hashlib.sha256(outcome["raw"]).hexdigest(),
        stderr_bytes=outcome["stderr_bytes"], elapsed_wall_ns=time.monotonic_ns()-service_origin_ns,
        max_poll_wait_ns=service.SAMPLE_NS,
        sampled_aggregate_peak_rss_bytes=max(outcome["sampled_aggregate_peak_rss_bytes"],
                                            before["sampled_rss_bytes"], after["sampled_rss_bytes"]),
        rss_scope="sum_cgroup_member_VmRSS_sampled_not_kernel_charge",
        wall_scope="service_activation_through_validation_before_slot_finalize",
        before=before, after=after, worker_group_gone=outcome["worker_group_gone"])
    service.validate_record(record, candidate)
    final = dict(candidate)
    final["supervision"] = record
    authority.encode_bounded(final)
    return final


def run_one_shot(*, expected_commit, expected_source_set_sha256):
    import native_storage_authority as authority
    import native_storage_measurement_slot as slot
    import process_supervision as supervision
    slot._validate_binding(expected_commit, expected_source_set_sha256)
    root = supervision.REPOSITORY_ROOT
    expected_file = root + "/" + supervision.I7_RELATIVE + "native_storage_measurement_controller.py"
    _need(str(Path(__file__).resolve()) == expected_file and os.getcwd() == root, "fixed_frozen_root")
    _need("torch" not in sys.modules and "numpy" not in sys.modules, "fresh_controller_required")
    _need(os.environ.get("CUDA_VISIBLE_DEVICES") == ""
          and all(os.environ.get(k) == v for k, v in authority.THREAD_SETTINGS.items())
          and os.environ.get("TMPDIR") == slot.BIG_TMP, "cpu_environment")
    service.validate_dependency_directory()
    context, properties, before, invocation, origin = service.assert_controller()
    try:
        service.verify_big_tmp()
        _need(time.monotonic_ns() - origin < 10_000_000_000, "preflight_time")
        with slot.reserve_measurement(expected_commit=expected_commit,
                expected_source_set_sha256=expected_source_set_sha256) as reservation:
            process = subprocess.Popen(_worker_command(expected_commit=expected_commit,
                expected_source_set_sha256=expected_source_set_sha256, controller_pid=os.getpid()),
                cwd=root, env=service.process_environment(), stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, close_fds=True,
                start_new_session=True)
            outcome = _capture(process, context, deadline_ns=origin + 110_000_000_000)
            if outcome["worker_exit_code"] == 2:
                raise MeasurementWorkerFailure(_worker_diagnostic(outcome))
            candidate = authority.decode_bounded(outcome["raw"], maximum=service.STDOUT_MAX)
            authority.validate_measurement_candidate(candidate, expected_commit=expected_commit,
                expected_source_set_sha256=expected_source_set_sha256)
            _need(outcome["worker_exit_code"] == (0 if candidate["status"] == "complete" else 1),
                  "worker_exit")
            # Re-check actual manager controls and held cgroup identity after all
            # worker descendants are gone. The short show helper has already exited.
            live = service._show()
            service.validate_live_properties(live, controller_pid=os.getpid())
            _need(live["InvocationID"] == invocation and
                  int(live["ActiveEnterTimestampMonotonic"]) * 1000 == origin, "service_replaced")
            after = context.snapshot()
            _need(after["member_pids"] == [os.getpid()], "service_descendants_remain")
            final = _attach_observation(candidate, outcome, properties=properties,
                before=before, after=after, invocation_id=invocation, service_origin_ns=origin)
            # Manager enforces remaining wall time even in a blocked final fsync.
            # A later independent review must also inspect the terminal unit result.
            _need(time.monotonic_ns() - origin < 115_000_000_000, "finalization_time")
            pin = reservation.finalize(final)
            _need(time.monotonic_ns() - origin < service.WALL_NS, "finalization_overrun")
            return pin
    finally:
        context.close()


def launch_one_shot(*, expected_commit, expected_source_set_sha256):
    """Explicit later-reviewed launch only. No auto-retry or direct fallback."""
    import native_storage_measurement_slot as slot
    import process_supervision as supervision
    slot._validate_binding(expected_commit, expected_source_set_sha256)
    service.validate_dependency_directory()
    for path in (supervision.LAYOUT_EVIDENCE_PATH, supervision.STORAGE_ADMISSION_PATH):
        _need(not os.path.lexists(path), "production_slot_present")
    raw = service._run_command(["/usr/bin/systemctl", "--user", "show", service.UNIT,
        "--property=LoadState", "--value"], cap=64)
    _need(raw == b"not-found\n", "measurement_service_already_loaded")
    root = supervision.REPOSITORY_ROOT
    path = root + "/" + supervision.I7_RELATIVE + "native_storage_measurement_controller.py"
    _need(Path(path).is_file() and str(Path(path).resolve()) == path, "fixed_controller_absent")
    environment = service.process_environment()
    code = ("import sys;sys.path.insert(0,sys.argv.pop(1));"
            "import native_storage_measurement_controller as c;raise SystemExit(c.main(sys.argv[1:]))")
    command = ["/usr/bin/systemd-run", "--user", "--unit=" + service.UNIT, "--quiet",
        "--working-directory=" + root,
        *("--property=" + p for p in service.RUN_PROPERTIES),
        "--", "/usr/bin/env", "-i", *(k + "=" + v for k, v in environment.items()),
        supervision.PYTHON_EXECUTABLE, "-I", "-B", "-c", code,
        str(Path(path).parent), "--service-entry", expected_commit, expected_source_set_sha256]
    # The manager owns the service after submission, even if this caller exits.
    service._run_command(command, cap=4096)
    return service.UNIT


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        print("native_storage_measurement_controller: inert; no measurement or service launch")
        return 0
    if len(argv) == 3 and argv[0] == "--service-entry":
        try:
            run_one_shot(expected_commit=argv[1], expected_source_set_sha256=argv[2])
            return 0
        except MeasurementWorkerFailure as exc:
            return exc.exit_status
        except service.MeasurementServiceError as exc:
            return (_WORKER_DIAGNOSTIC_EXITS["dependency_directory"]
                    if str(exc) == "dependency_directory" else 2)
        except Exception:
            return 2  # Fixed exit; no unbounded exception/error payload.
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
