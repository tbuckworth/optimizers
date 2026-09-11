"""One-shot, sequential CPU structural-storage measurement candidate.

Import and default CLI are inert and Torch-free.  The public entry point is for
the separately supervised measurement child only; it neither reserves/writes M
nor creates A, permissions, specimens, plans, data, or scientific artifacts.
"""
from __future__ import annotations

import os
import re
import sys


COMPONENTS = (
    ("anchor_pilot", "torch_weights_only", 2),
    ("anchor_long", "torch_weights_only", 16),
    ("source_witness", "torch_weights_only", 18),
    ("branch_results", "torch_weights_only", 18),
    ("independent_audit_all_failure", "torch_weights_only", 16),
    ("source_completion_pilot_on", "torch_weights_only", 1),
    ("source_completion_pilot_off", "torch_weights_only", 1),
    ("capture_comparison_pilot", "bytes", 1),
    ("source_completion_long_on", "torch_weights_only", 4),
    ("plan_pilot", "torch_weights_only", 1),
    ("plan_long", "torch_weights_only", 4),
)
THREAD_NAMES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS")
_COMMIT = re.compile(r"[0-9a-f]{40}\Z", re.ASCII)
_SHA256 = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)
_STAGES = frozenset(("source", "runtime", "recipe", "serialize", "roundtrip",
                     "resource", "report"))
_DEPENDENCY_KEYS = (
    "fixed_root", "attempt_id", "measurement_schema", "measurement_protocol",
    "measurement_role", "resource_limits", "thread_settings", "monotonic_ns",
    "process_time_ns", "peak_rss_bytes", "current_pid", "parent_pid",
    "assert_service_member", "configure_runtime", "python_rng", "capture_rng",
    "rng_equal", "cuda_initialized", "cuda_visible_devices", "observed_threads",
    "collect_sources", "source_digest", "collect_environment", "validate_metadata",
    "component_specs", "build_template", "validate_template", "materialize",
    "json_roundtrip", "serialize_roundtrip", "collect_garbage", "validate_candidate",
)


class MeasurementWorkerError(ValueError):
    """An exact worker input/dependency or measurement contract was violated."""


class MeasurementWorkerBootstrapError(RuntimeError):
    """No truthful schema-valid M candidate could be constructed."""

    def __init__(self, code):
        if type(code) is not str or code not in (
                "preflight_unobserved", "runtime_unobserved", "rows_unobserved",
                "report_unobserved"):
            code = "report_unobserved"
        self.code = code
        super().__init__(code)


class _ResourceLimit(Exception):
    pass


def _need(condition, message):
    if not condition:
        raise MeasurementWorkerError(message)


def _dependencies(value):
    _need(type(value) is dict and tuple(value) == _DEPENDENCY_KEYS,
          "exact ordered worker dependencies required")
    for key in _DEPENDENCY_KEYS:
        if key not in {"fixed_root", "attempt_id", "measurement_schema",
                       "measurement_protocol", "measurement_role", "resource_limits",
                       "thread_settings"}:
            _need(callable(value[key]), "worker dependency must be callable")
    _need(type(value["fixed_root"]) is str and value["fixed_root"].startswith("/")
          and os.path.normpath(value["fixed_root"]) == value["fixed_root"],
          "fixed root dependency invalid")
    for key in ("attempt_id", "measurement_schema", "measurement_protocol", "measurement_role"):
        _need(type(value[key]) is str and value[key], "worker literal dependency invalid")
    limits = value["resource_limits"]
    _need(type(limits) is dict and tuple(limits) == ("wall_ns", "rss_bytes", "buffer_bytes")
          and all(type(item) is int and item > 0 for item in limits.values()),
          "worker resource limits invalid")
    threads = value["thread_settings"]
    _need(type(threads) is dict and tuple(threads) == THREAD_NAMES
          and all(type(k) is str and type(v) is str for k, v in threads.items()),
          "worker thread settings invalid")
    return value


def _inputs(*, expected_commit, expected_source_set_sha256, repository_root,
            entry_wall_ns, entry_cpu_ns, runner_pid, controller_pid, expected_cgroup,
            dependencies):
    _need(type(expected_commit) is str and _COMMIT.fullmatch(expected_commit) is not None,
          "invalid expected commit")
    _need(type(expected_source_set_sha256) is str
          and _SHA256.fullmatch(expected_source_set_sha256) is not None,
          "invalid expected source digest")
    _need(type(repository_root) is str and repository_root == dependencies["fixed_root"],
          "measurement requires fixed frozen repository root")
    _need(type(entry_wall_ns) is int and type(entry_cpu_ns) is int
          and entry_wall_ns >= 0 and entry_cpu_ns >= 0, "invalid entry origins")
    _need(type(runner_pid) is int and runner_pid > 0
          and type(controller_pid) is int and controller_pid > 0
          and runner_pid != controller_pid, "invalid process identities")
    _need(type(expected_cgroup) is str and expected_cgroup.isascii()
          and expected_cgroup.startswith("/") and len(expected_cgroup) <= 4096
          and os.path.normpath(expected_cgroup) == expected_cgroup,
          "invalid expected service cgroup")


class _Tracker:
    def __init__(self, dependencies, entry_wall_ns, entry_cpu_ns):
        self.dependencies = dependencies
        self.entry_wall_ns = entry_wall_ns
        self.entry_cpu_ns = entry_cpu_ns
        self.peak_buffer_bytes = 0

    def observe_buffer(self, value):
        _need(type(value) is int and 0 <= value <= self.dependencies["resource_limits"]["buffer_bytes"],
              "invalid serializer buffer observation")
        self.peak_buffer_bytes = max(self.peak_buffer_bytes, value)

    def sample(self):
        wall = self.dependencies["monotonic_ns"]() - self.entry_wall_ns
        cpu = self.dependencies["process_time_ns"]() - self.entry_cpu_ns
        rss = self.dependencies["peak_rss_bytes"]()
        _need(type(wall) is int and wall >= 0 and type(cpu) is int and cpu >= 0
              and type(rss) is int and rss >= 0, "invalid resource observation")
        return wall, cpu, rss

    def check(self):
        wall, _, rss = self.sample()
        limits = self.dependencies["resource_limits"]
        if (wall > limits["wall_ns"] or rss > limits["rss_bytes"]
                or self.peak_buffer_bytes > limits["buffer_bytes"]):
            raise _ResourceLimit


def _specifications(dependencies):
    raw = dependencies["component_specs"]()
    _need(type(raw) is tuple and len(raw) == len(COMPONENTS),
          "exact eleven component specifications required")
    result = []
    expected_fields = ("component", "encoding", "payload_count",
                       "analytic_pickle_bytes_upper", "analytic_body_bytes_upper", "tensor_count")
    for value, fixed in zip(raw, COMPONENTS):
        _need(type(value) is dict and tuple(value) == expected_fields,
              "component specification fields differ")
        name, encoding, count = fixed
        _need((value["component"], value["encoding"], value["payload_count"])
              == (name, encoding, count), "component specification order differs")
        pickle_upper = value["analytic_pickle_bytes_upper"]
        _need((pickle_upper is None if encoding == "bytes" else
               type(pickle_upper) is int and 0 < pickle_upper <= 64 << 20),
              "component pickle ceiling invalid")
        _need(type(value["analytic_body_bytes_upper"]) is int
              and 0 < value["analytic_body_bytes_upper"] <= 64 << 20
              and type(value["tensor_count"]) is int and 0 <= value["tensor_count"] <= 132
              and (encoding != "bytes" or value["tensor_count"] == 0),
              "component body/count ceiling invalid")
        result.append(dict(value))
    return tuple(result)


def _rows(specifications):
    return [dict(component=spec["component"], encoding=spec["encoding"],
                 payload_count=spec["payload_count"],
                 analytic_pickle_bytes_upper=spec["analytic_pickle_bytes_upper"],
                 analytic_body_bytes_upper=spec["analytic_body_bytes_upper"],
                 status="not_run", observation=None)
            for spec in specifications]


def _measure_component(spec, *, sources, environment, dependencies, tracker, stage):
    """Keep the only full tree in this frame and discard it before returning."""
    template = tree = None
    try:
        stage[0] = "recipe"
        template = dependencies["build_template"](
            spec["component"], sources=sources, environment=environment)
        check = dependencies["validate_template"](spec["component"], template)
        _need(type(check) is dict and check.get("slot_count") == spec["tensor_count"]
              and check.get("body_bytes_upper") == spec["analytic_body_bytes_upper"],
              "template validation result differs")
        tree = dependencies["materialize"](spec["component"], template)
        template = None
        dependencies["collect_garbage"]()
        tracker.check()

        stage[0] = "serialize"
        def on_stage(value):
            _need(type(value) is str and value in ("serialize", "roundtrip"),
                  "invalid serializer stage callback")
            stage[0] = value
        if spec["encoding"] == "bytes":
            observation = dependencies["json_roundtrip"](
                tree, body_ceiling=spec["analytic_body_bytes_upper"],
                on_buffer_peak=tracker.observe_buffer, on_stage=on_stage)
            runtime = None
        else:
            observation, runtime = dependencies["serialize_roundtrip"](
                tree, pickle_ceiling=spec["analytic_pickle_bytes_upper"],
                body_ceiling=spec["analytic_body_bytes_upper"],
                tensor_count=spec["tensor_count"], on_buffer_peak=tracker.observe_buffer,
                on_stage=on_stage)
        tree = None
        dependencies["collect_garbage"]()
        tracker.check()
        return observation, runtime
    finally:
        tree = None
        template = None
        dependencies["collect_garbage"]()


def _resource_record(dependencies, tracker, *, cuda_before, rng_preserved):
    wall, cpu, rss = tracker.sample()
    cuda_after = dependencies["cuda_initialized"]()
    visible = dependencies["cuda_visible_devices"]()
    threads = dependencies["observed_threads"]()
    _need(type(cuda_before) is bool and type(cuda_after) is bool
          and type(rng_preserved) is bool and type(visible) is str,
          "invalid runtime observation")
    _need(type(threads) is dict and tuple(threads) == tuple(dependencies["thread_settings"])
          and all(type(v) is str for v in threads.values()), "invalid observed thread settings")
    return dict(elapsed_wall_ns=wall, elapsed_cpu_ns=cpu, peak_rss_bytes=rss,
                peak_buffer_bytes=tracker.peak_buffer_bytes,
                runner_pid=dependencies["current_pid"](), process_scope="runner_self",
                cuda_visible_devices=visible, cuda_initialized_before=cuda_before,
                cuda_initialized_after=cuda_after, thread_settings=dict(threads),
                rng_preserved=rng_preserved)


def _candidate(*, dependencies, expected_commit, expected_source_set_sha256,
               environment, rows, serializer_runtime, resources, failure):
    return dict(schema=dependencies["measurement_schema"],
        status="complete" if failure is None else "failed",
        evidence_role=dependencies["measurement_role"], attempt_id=dependencies["attempt_id"],
        repository_revision=expected_commit, source_set_sha256=expected_source_set_sha256,
        diagnostic_environment=environment,
        measurement_protocol=dependencies["measurement_protocol"], component_rows=rows,
        serializer_runtime=dict(zip_runtime=serializer_runtime,
            pickle_runtime_checked=serializer_runtime is not None,
            compiled_binary_provenance_attested=False),
        resource_limits=dict(dependencies["resource_limits"]), observed_resources=resources,
        supervision=None, failure=failure, execution_authorized=False,
        scientific_execution_certified=False)


def _finish_candidate(candidate, *, dependencies, expected_commit, expected_source_set_sha256):
    try:
        dependencies["validate_candidate"](candidate, expected_commit=expected_commit,
                                             expected_source_set_sha256=expected_source_set_sha256)
        return candidate
    except Exception:
        if candidate["status"] != "complete":
            raise MeasurementWorkerBootstrapError("report_unobserved") from None
        failed = dict(candidate)
        failed["status"] = "failed"
        failed["failure"] = dict(stage="report", component=None, reason="contract_failure")
        try:
            dependencies["validate_candidate"](
                failed, expected_commit=expected_commit,
                expected_source_set_sha256=expected_source_set_sha256)
            return failed
        except Exception:
            raise MeasurementWorkerBootstrapError("report_unobserved") from None


def _build_with_dependencies(*, expected_commit, expected_source_set_sha256,
                             repository_root, entry_wall_ns, entry_cpu_ns, runner_pid,
                             controller_pid, expected_cgroup, python_rng_before,
                             dependencies):
    """Explicit tiny-dependency seam.  Production callers use the public API."""
    dependencies = _dependencies(dependencies)
    _inputs(expected_commit=expected_commit,
            expected_source_set_sha256=expected_source_set_sha256,
            repository_root=repository_root, entry_wall_ns=entry_wall_ns,
            entry_cpu_ns=entry_cpu_ns, runner_pid=runner_pid,
            controller_pid=controller_pid, expected_cgroup=expected_cgroup,
            dependencies=dependencies)
    _need(dependencies["current_pid"]() == runner_pid
          and dependencies["parent_pid"]() == controller_pid,
          "worker process identity differs")
    try:
        member = dependencies["assert_service_member"](
            controller_pid=controller_pid, expected_cgroup=expected_cgroup)
    except Exception:
        raise MeasurementWorkerError("worker service cgroup differs") from None
    _need(member == expected_cgroup, "worker service cgroup differs")
    tracker = _Tracker(dependencies, entry_wall_ns, entry_cpu_ns)
    try:
        dependencies["configure_runtime"]()
        cuda_before = dependencies["cuda_initialized"]()
        _need(cuda_before is False, "CUDA initialized before measurement")
        rng_before = dependencies["capture_rng"]()
        _need(dependencies["python_rng"]() == python_rng_before,
              "Python RNG changed during runtime import/configuration")
    except Exception:
        raise MeasurementWorkerBootstrapError("runtime_unobserved") from None
    try:
        specifications = _specifications(dependencies)
        rows = _rows(specifications)
    except Exception:
        raise MeasurementWorkerBootstrapError("rows_unobserved") from None

    environment = None
    serializer_runtime = None
    failure = None
    stage = ["source"]
    current_index = None
    try:
        tracker.check()
        sources = dependencies["collect_sources"](repository_root)
        _need(type(sources) is dict
              and sources.get("repository_root_realpath") == repository_root
              and sources.get("repository_revision") == expected_commit
              and dependencies["source_digest"](sources) == expected_source_set_sha256,
              "fresh source binding differs")
        stage[0] = "runtime"
        environment = dependencies["collect_environment"](repository_root)
        dependencies["validate_metadata"](sources, environment)
        _need(dependencies["parent_pid"]() == controller_pid,
              "measurement controller disappeared")

        for current_index, spec in enumerate(specifications):
            tracker.check()
            observation, runtime = _measure_component(
                spec, sources=sources, environment=environment,
                dependencies=dependencies, tracker=tracker, stage=stage)
            stage[0] = "runtime"
            component_rng = dependencies["capture_rng"]()
            _need(dependencies["python_rng"]() == python_rng_before
                  and dependencies["rng_equal"](rng_before, component_rng)
                  and dependencies["cuda_initialized"]() is False,
                  "component changed RNG/CUDA state")
            component_rng = None
            if runtime is not None:
                _need(type(runtime) is dict, "serializer runtime record invalid")
                if serializer_runtime is None:
                    serializer_runtime = runtime
                else:
                    _need(runtime == serializer_runtime, "serializer runtime changed")
            rows[current_index]["status"] = "measured"
            rows[current_index]["observation"] = observation
            current_index = None
            _need(dependencies["parent_pid"]() == controller_pid,
                  "measurement controller disappeared")

        stage[0] = "source"
        sources_after = dependencies["collect_sources"](repository_root)
        _need(sources_after == sources
              and dependencies["source_digest"](sources_after) == expected_source_set_sha256,
              "source binding changed during measurement")
        stage[0] = "runtime"
        environment_after = dependencies["collect_environment"](repository_root)
        dependencies["validate_metadata"](sources_after, environment_after)
        _need(environment_after == environment, "runtime environment changed during measurement")
        tracker.check()
    except _ResourceLimit:
        failure = dict(stage="runtime" if environment is None else "resource",
                       component=None if current_index is None else specifications[current_index]["component"],
                       reason="resource_limit")
    except Exception:
        failure = dict(stage=stage[0],
                       component=None if current_index is None else specifications[current_index]["component"],
                       reason="contract_failure")

    # An exception traceback can retain serializer frames until the handler has
    # exited.  Collect again here, after that implicit exception reference was
    # cleared, before observing final runtime state or constructing the report.
    dependencies["collect_garbage"]()
    if failure is not None and current_index is not None:
        rows[current_index]["status"] = "failed"
        rows[current_index]["observation"] = None
    try:
        rng_after = dependencies["capture_rng"]()
        rng_preserved = (dependencies["python_rng"]() == python_rng_before
                         and dependencies["rng_equal"](rng_before, rng_after))
    except Exception:
        rng_preserved = False
        if failure is None:
            failure = dict(stage="runtime", component=None, reason="contract_failure")
    try:
        resources = _resource_record(dependencies, tracker, cuda_before=cuda_before,
                                     rng_preserved=rng_preserved)
    except Exception:
        raise MeasurementWorkerBootstrapError("runtime_unobserved") from None
    if failure is None:
        limits = dependencies["resource_limits"]
        if (resources["elapsed_wall_ns"] > limits["wall_ns"]
                or resources["peak_rss_bytes"] > limits["rss_bytes"]
                or resources["peak_buffer_bytes"] > limits["buffer_bytes"]):
            failure = dict(stage="resource", component=None, reason="resource_limit")
        elif (not rng_preserved or resources["cuda_initialized_after"]
              or resources["cuda_visible_devices"] != ""
              or resources["thread_settings"] != dependencies["thread_settings"]):
            failure = dict(stage="runtime", component=None, reason="contract_failure")
    value = _candidate(dependencies=dependencies, expected_commit=expected_commit,
        expected_source_set_sha256=expected_source_set_sha256, environment=environment,
        rows=rows, serializer_runtime=serializer_runtime, resources=resources, failure=failure)
    return _finish_candidate(value, dependencies=dependencies, expected_commit=expected_commit,
                             expected_source_set_sha256=expected_source_set_sha256)


def _production_dependencies():
    """Load the reviewed runtime closure only after the fresh-process preflight."""
    import gc
    import random
    import resource
    import time

    import identity_codec
    import native_storage_authority as authority
    import native_storage_crosscheck as crosscheck
    import native_storage_measurement_service as service
    import native_storage_recipe_branch_audit as branch_audit
    import native_storage_recipe_common as common
    import native_storage_recipe_core as core
    import native_storage_topology_bound as topology
    import native_tensor_inventory as inventory
    import process_supervision as supervision
    import source_environment
    import source_environment_schema as environment_schema
    import torch

    def configure_runtime():
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.set_default_dtype(torch.float32)
        torch.set_default_device("cpu")
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.set_float32_matmul_precision("highest")
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    def capture_rng():
        state = __import__("numpy").random.get_state()
        copied = (state[0], state[1].copy(), *state[2:])
        return random.getstate(), copied, torch.get_rng_state().detach().cpu().clone()

    def rng_equal(left, right):
        numpy = __import__("numpy")
        return (left[0] == right[0] and left[1][0] == right[1][0]
                and numpy.array_equal(left[1][1], right[1][1])
                and left[1][2:] == right[1][2:]
                and torch.equal(left[2], right[2]))

    def component_specs():
        bounds = topology.compute()["components"]
        layouts = inventory.component_layouts()
        _need(tuple(bounds) == tuple(layouts) == tuple(name for name, _, _ in COMPONENTS),
              "calculated component order differs")
        return tuple(dict(component=name, encoding=bound["encoding"],
            payload_count=bound["payload_count"],
            analytic_pickle_bytes_upper=bound.get("pickle_bytes_upper"),
            analytic_body_bytes_upper=bound["body_bytes_upper"],
            tensor_count=len(layouts[name])) for name, bound in bounds.items())

    def build_template(component, *, sources, environment):
        builder = core if component in core.COMPONENTS else branch_audit
        return builder.build_template(component, sources=sources, environment=environment)

    return {
        "fixed_root": supervision.REPOSITORY_ROOT,
        "attempt_id": supervision.ATTEMPT_ID,
        "measurement_schema": authority.MEASUREMENT_SCHEMA,
        "measurement_protocol": authority.MEASUREMENT_PROTOCOL,
        "measurement_role": authority.MEASUREMENT_ROLE,
        "resource_limits": dict(authority.RESOURCE_LIMITS),
        "thread_settings": dict(authority.THREAD_SETTINGS),
        "monotonic_ns": time.monotonic_ns,
        "process_time_ns": time.process_time_ns,
        "peak_rss_bytes": lambda: int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
        "current_pid": os.getpid,
        "parent_pid": os.getppid,
        "assert_service_member": service.assert_service_member,
        "configure_runtime": configure_runtime,
        "python_rng": random.getstate,
        "capture_rng": capture_rng,
        "rng_equal": rng_equal,
        "cuda_initialized": torch.cuda.is_initialized,
        "cuda_visible_devices": lambda: os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
        "observed_threads": lambda: {key: os.environ.get(key, "<unset>")
                                      for key in authority.THREAD_SETTINGS},
        "collect_sources": lambda root: source_environment.collect_verified_sources(
            root, profile=environment_schema.SCIENTIFIC),
        "source_digest": identity_codec.tree_digest,
        "collect_environment": lambda root: source_environment.collect_runtime_environment(
            root, profile=environment_schema.SCIENTIFIC,
            runtime_role="storage_crosscheck_cpu"),
        "validate_metadata": crosscheck.validate_metadata,
        "component_specs": component_specs,
        "build_template": build_template,
        "validate_template": common.validate_template,
        "materialize": common.materialize,
        "json_roundtrip": crosscheck._json_roundtrip,
        "serialize_roundtrip": crosscheck._serialize_roundtrip,
        "collect_garbage": gc.collect,
        "validate_candidate": authority.validate_measurement_candidate,
    }


def build_measurement_candidate(*, expected_commit, expected_source_set_sha256,
                                repository_root, entry_wall_ns, entry_cpu_ns,
                                runner_pid, controller_pid, expected_cgroup):
    """Run the CPU recipe once and return an unwritten Mv3 candidate.

    The controller must already have exclusively consumed M and verified the
    systemd service before calling.  This function cannot attest either fact.
    """
    if any(name == "torch" or name.startswith("torch.") or name == "numpy"
           or name.startswith("numpy.") for name in sys.modules):
        raise MeasurementWorkerBootstrapError("preflight_unobserved")
    import process_supervision as supervision
    import random
    import time
    import native_storage_measurement_service as service
    now_wall, now_cpu = time.monotonic_ns(), time.process_time_ns()
    valid_cgroup = (type(expected_cgroup) is str and expected_cgroup.isascii()
                    and expected_cgroup.startswith("/") and len(expected_cgroup) <= 4096
                    and os.path.normpath(expected_cgroup) == expected_cgroup)
    if (type(expected_commit) is not str or _COMMIT.fullmatch(expected_commit) is None
            or type(expected_source_set_sha256) is not str
            or _SHA256.fullmatch(expected_source_set_sha256) is None
            or repository_root != supervision.REPOSITORY_ROOT or runner_pid != os.getpid()
            or controller_pid != os.getppid() or type(entry_wall_ns) is not int
            or type(entry_cpu_ns) is not int or entry_wall_ns < 0 or entry_cpu_ns < 0
            or entry_wall_ns > now_wall or now_wall - entry_wall_ns > 120_000_000_000
            or entry_cpu_ns > now_cpu or not valid_cgroup):
        raise MeasurementWorkerBootstrapError("preflight_unobserved")
    required_environment = service.process_environment()
    if dict(os.environ) != required_environment:
        raise MeasurementWorkerBootstrapError("preflight_unobserved")
    try:
        member = service.assert_service_member(
            controller_pid=controller_pid, expected_cgroup=expected_cgroup)
    except Exception:
        raise MeasurementWorkerBootstrapError("preflight_unobserved") from None
    if member != expected_cgroup:
        raise MeasurementWorkerBootstrapError("preflight_unobserved")
    python_rng_before = random.getstate()
    try:
        dependencies = _production_dependencies()
    except Exception:
        raise MeasurementWorkerBootstrapError("runtime_unobserved") from None
    return _build_with_dependencies(expected_commit=expected_commit,
        expected_source_set_sha256=expected_source_set_sha256,
        repository_root=repository_root, entry_wall_ns=entry_wall_ns,
        entry_cpu_ns=entry_cpu_ns, runner_pid=runner_pid, controller_pid=controller_pid,
        expected_cgroup=expected_cgroup, python_rng_before=python_rng_before,
        dependencies=dependencies)


def main(argv=None):
    print("native_storage_measurement_worker: inert; controller-only API, no M/A write")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
