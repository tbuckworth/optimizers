"""Fresh, single-phase worker; import/default CLI are inert and Torch-free.

Native entry requires the supervisor's consumed pinned request and the existing
inner phase permission. No callbacks, replacement data, retry or CPU training
fallback are exposed. Storage admission is a separate fail-closed prerequisite.
"""
from __future__ import annotations

# These readings belong to this interpreter, before project/numerical imports.
import time
import os
_ENTRY_WALL = time.monotonic()
_ENTRY_CPU = time.process_time()
_ENTRY_PID = os.getpid()

import argparse
from datetime import datetime, timezone
from pathlib import Path
import resource
import sys


class WorkerError(RuntimeError):
    pass


class WorkerResourceError(WorkerError):
    pass


def _need(condition, message):
    if not condition:
        raise WorkerError(message)


def _origins():
    _need(os.getpid() == _ENTRY_PID, "worker must not reuse forked entry origins")
    return dict(entry_wall_origin=_ENTRY_WALL, entry_cpu_origin=_ENTRY_CPU,
                entry_process_id=_ENTRY_PID)


def _utc():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _resources():
    _origins()
    torch = sys.modules.get("torch")
    initialized = torch is not None and torch.cuda.is_initialized()
    return {
        "worker_entry_wall_origin":_ENTRY_WALL,
        "worker_entry_cpu_origin":_ENTRY_CPU,
        "worker_entry_process_id":_ENTRY_PID,
        "wall_seconds":time.monotonic() - _ENTRY_WALL,
        "cpu_seconds":time.process_time() - _ENTRY_CPU,
        "peak_rss_bytes":int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
        "peak_cuda_allocated_bytes":int(torch.cuda.max_memory_allocated(0)) if initialized else 0,
        "peak_cuda_reserved_bytes":int(torch.cuda.max_memory_reserved(0)) if initialized else 0,
        "cuda_initialized":bool(initialized),
    }


def _check_resources(request):
    values = _resources()
    limits = request["limits"]
    if not (time.monotonic() <= request["deadline_monotonic"]
          and values["wall_seconds"] <= limits["wall_seconds"]
          and (limits["cpu_seconds"] is None
               or values["cpu_seconds"] <= limits["cpu_seconds"])
          and values["peak_rss_bytes"] <= limits["peak_rss_bytes"]
          and values["peak_cuda_allocated_bytes"] <= limits["peak_cuda_allocated_bytes"]):
        raise WorkerResourceError("worker resource limit exceeded")
    if request["phase"] == "audit":
        _need(not values["cuda_initialized"], "CPU audit initialized CUDA")
    return values


def _gpu_preflight(request):
    """Read-only occupancy snapshot; helpers inherit this owned worker session."""
    import source_environment as provenance
    import native_layout_inspection as layout
    root = request["worker_binding"]["repository_root"]
    uuid = request["expected_gpu_uuid"]
    _check_resources(request)
    raw = provenance._run(["nvidia-smi",
        "--query-gpu=index,uuid,name,memory.free,driver_version",
        "--format=csv,noheader,nounits"], cwd=root, cap=64 << 10)
    rows = [line for line in raw.decode("ascii").splitlines() if line.strip()]
    fields = [value.strip() for value in rows[0].split(",")] if len(rows) == 1 else []
    _need(len(fields) == 5 and fields[0] == "0" and fields[1] == uuid
          and fields[2] == layout.EXPECTED_GPU_NAME and fields[3].isdecimal()
          and int(fields[3]) >= layout.GPU_FREE_MINIMUM_MIB and bool(fields[4]),
          "GPU preflight identity/free-memory mismatch")
    _check_resources(request)
    raw = provenance._run(["nvidia-smi",
        "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
        "--format=csv,noheader,nounits"], cwd=root, cap=64 << 10)
    for found_uuid, pid, reported_name, _ in layout._parse_compute_rows(raw):
        _need(found_uuid == uuid, "unexpected GPU occupancy")
        identity = layout._proc_identity(pid)
        if identity is None:
            continue
        executable, comm, uid = identity
        allowed = layout.ALLOWED_COMPUTE_PROCESSES.get(executable)
        _need(allowed is not None and comm in allowed["comms"]
              and reported_name in allowed["reported_names"] and uid == os.getuid(),
              "unknown or research process occupies GPU")
    _check_resources(request)


def _initialize_runtime(request, permission):
    # Caller has durably consumed the inner permission before this function.
    import torch
    import native_layout_inspection as layout
    _need(request["phase"] != "audit"
          and permission["expected_gpu_uuid"] == request["expected_gpu_uuid"],
          "initializer permission differs")
    _gpu_preflight(request)
    layout._configure_and_assert_torch(torch)
    _check_resources(request)
    torch.cuda.init()
    _need(torch.cuda.device_count() == 1 and torch.cuda.current_device() == 0,
          "exactly one native CUDA device required")
    _check_resources(request)
    # No scientific payload, no returned authority. Inner caller verifies UUID.


def _collect_metadata(request, permission, progress):
    import source_environment as provenance
    import source_environment_schema as schema
    import identity_codec as codec
    import final_storage_accounting as accounting
    root = request["worker_binding"]["repository_root"]
    role = "cpu_audit" if request["phase"] == "audit" else "native_source"
    if role == "cpu_audit":
        import torch
        import native_layout_inspection as layout
        _need(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "audit CUDA must be hidden")
        layout._configure_and_assert_torch(torch)
    _check_resources(request)
    sources = provenance.collect_verified_sources(root, profile=schema.SCIENTIFIC)
    _need(sources["repository_revision"] == permission["expected_commit"]
          and codec.tree_digest(sources) == permission["expected_source_set_sha256"],
          "actual frozen sources differ from permission")
    environment = provenance.collect_runtime_environment(
        root, profile=schema.SCIENTIFIC, runtime_role=role)
    _check_resources(request)
    # This gate must validate the actual runtime against a reviewed encoded-size
    # upper bound, before bootstrap creates a root/GO or acquisition writes data.
    progress["stage"] = "storage_admission"
    accounting.validate_runtime_admission(request["storage_admission_pin"],
        sources=sources, environment=environment)
    _check_resources(request)
    return sources, environment


def _execute_request(request, progress):
    import native_control as control  # stdlib-only until consumed bootstrap
    phase = request["phase"]
    binding = request["worker_binding"]
    origins = _origins()
    controller = None
    opened = None
    collected_metadata = None
    def collect(permission):
        nonlocal collected_metadata
        progress["stage"] = "metadata_collection"
        result = _collect_metadata(request, permission, progress)
        collected_metadata = result
        progress["stage"] = "bootstrap" if phase == "development" else "acquire"
        return result
    def initialize(permission):
        progress["stage"] = "runtime_initialization"
        _initialize_runtime(request, permission)
    try:
        if phase == "development":
            progress["stage"] = "bootstrap"
            def create_store():
                import artifact_store as storage
                _need(collected_metadata is not None,
                      "store requires completed metadata/admission collection")
                sources, environment = collected_metadata
                return storage.ArtifactStore("/tmp/spectral-experiment-artifacts", profile=storage.SCIENTIFIC,
                    runtime_sources=sources, runtime_environment=environment)
            opened = control.bootstrap_development(
                request["development_permission_ascii"].encode("ascii"),
                pins=request["development_pins"], **origins,
                initialize_runtime=initialize, collect_metadata=collect,
                create_store=create_store, utc_now=_utc)
            from native_controller import NativeController
            progress["stage"] = "controller"
            controller = NativeController(opened, entry_process_id=_ENTRY_PID,
                data_directory=binding["data_directory"], expected_files=binding["expected_files"],
                utc_now=_utc)
            controller.run_development()
            progress["stage"] = "boundary"
            return controller.close_development()
        else:
            import phase_transition as transition
            progress["stage"] = "acquire"
            opened = transition.acquire_phase(request["phase_permission_pin"],
                phase=phase, prior_boundary=request["prior_boundary"], **origins,
                initialize_runtime=None if phase == "audit" else initialize,
                collect_metadata=collect)
            from scientific_controller import ScientificController
            progress["stage"] = "controller"
            controller = ScientificController(opened, entry_process_id=_ENTRY_PID,
                data_directory=binding["data_directory"], expected_files=binding["expected_files"],
                utc_now=_utc)
            controller.run_phase()
            progress["stage"] = "boundary"
            return controller.close_phase()
    finally:
        # No retry/reopen. Inner controller owns its detailed failure evidence.
        if opened is not None:
            store = opened["store"]
            if not store._closed:
                store.close()


def _report(request, digest, *, handoff, failure, resources):
    import process_supervision as supervision
    return {"schema":supervision.WORKER_REPORT_SCHEMA,
        "status":"boundary_closed" if failure is None else "worker_failed",
        "attempt_id":request["attempt_id"], "phase":request["phase"],
        "request_sha256":digest, "worker_pid":_ENTRY_PID,
        "handoff":handoff, "failure":failure, "resources":resources,
        "execution_authorized":False, "scientific_execution_certified":False}


def _native_main(args):
    import process_supervision as supervision
    _need("torch" not in sys.modules, "native worker must enter before Torch import")
    _, request = supervision.load_pinned_request(
        args.request_path, args.request_size, args.request_sha256)
    binding = request["worker_binding"]
    _need(request["supervisor_pid"] == args.supervisor_pid == os.getppid(),
          "request is not from the live parent supervisor")
    _need(os.getpid() == os.getpgrp() == os.getsid(0), "worker must own a fresh session")
    _need(os.getcwd() == binding["repository_root"]
          and str(Path(__file__).resolve()) == binding["worker_path"]
          and os.path.realpath(sys.executable) == binding["python_executable"],
          "worker interpreter/source/cwd differs")
    _need(os.environ.get("CUDA_VISIBLE_DEVICES") ==
          ("" if request["phase"] == "audit" else request["expected_gpu_uuid"]),
          "worker CUDA visibility differs")
    report = _execute_and_report(request, args.request_sha256)
    sys.stdout.buffer.write(supervision.json_bytes(report, maximum=supervision.WORKER_OUTPUT_MAX))
    sys.stdout.buffer.flush()
    return 0 if report["failure"] is None else 1


def _execute_and_report(request, digest):
    """Internal reporting boundary; inner permission APIs still authenticate."""
    progress = {"stage":"request_validation"}
    handoff = failure = None
    try:
        _check_resources(request)
        handoff = _execute_request(request, progress)
        from scientific_controller import prior_boundary
        handoff = prior_boundary(request["phase"], handoff)
        progress["stage"] = "worker_final"
        resources = _check_resources(request)
    except BaseException as exc:
        # Never serialize raw exceptions, credentials, RNG, source bytes or logs.
        handoff = None
        failure = {"stage":progress["stage"],
                   "reason":"resource_limit" if isinstance(exc, WorkerResourceError)
                       else "worker_stage_failed",
                   "retention":"consumed_request_no_retry"}
        try:
            resources = _resources()
        except BaseException:
            resources = None
    return _report(request, digest, handoff=handoff, failure=failure, resources=resources)


def _fixture_audit_attestation():
    """Registered tiny CPU process probe; no request, file, plan or data path."""
    _need(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "fixture requires hidden CUDA")
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        _need(os.environ.get(key) == "1", "fixture requires single numerical threads")
    import torch
    import native_layout_inspection as layout
    layout._configure_and_assert_torch(torch)
    _need(not torch.cuda.is_initialized(), "fixture must not initialize CUDA")
    values = _resources()
    _need(values["wall_seconds"] < 120.0 and values["peak_rss_bytes"] < 2 << 30,
          "fixture resource cap exceeded")
    import json
    print(json.dumps({"schema":"i7_worker_cpu_fixture_v1", "fixture_only":True,
        "resources":values, "execution_authorized":False,
        "scientific_execution_certified":False}, separators=(",", ":")))
    return 0


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        print('{"status":"inert"}')
        return 0
    if argv == ["--fixture-audit-attestation"]:
        return _fixture_audit_attestation()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-native-phase", action="store_true", required=True)
    parser.add_argument("--request-path", required=True)
    parser.add_argument("--request-size", required=True, type=int)
    parser.add_argument("--request-sha256", required=True)
    parser.add_argument("--supervisor-pid", required=True, type=int)
    args = parser.parse_args(argv)
    try:
        return _native_main(args)
    except Exception:
        print("worker request rejected", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
