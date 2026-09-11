"""Actual-tree storage preflight for the fixed I7 native payload schedule.

Import and default CLI are Torch-free and perform no filesystem work. Explicit
calls validate actual trees and compute bounded protocol-2/ZIP sizes, but do not
serialize, write, execute science, or confer GO authority. Exact upstream
semantic validation remains an immediate caller obligation.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
from typing import Any


SCHEMA = "i7_actual_payload_storage_preflight_v1"
PROFILE = "scientific_mnist_current32_v1"
SOURCE_JSON_CAP = 32_768
ENVIRONMENT_JSON_CAP = 8_192
MAX_NODES = 200_000
MAX_DEPTH = 32
AUDIT_MAX_UTF8_BYTES = 4 << 20
OTHER_MAX_UTF8_BYTES = SOURCE_JSON_CAP
COMPARISON_MAX_NODES = 10_000
COMPARISON_MAX_DEPTH = 16
COMPARISON_MAX_STRING_BYTES = 4_096

COMPONENT_BODY_CEILINGS = {
    "anchor_pilot": 7_625_293,
    "anchor_long": 7_625_312,
    "source_witness": 7_753_753,
    "branch_results": 33_452_716,
    "independent_audit_all_failure": 7_051_883,
    "source_completion_pilot_on": 7_520_543,
    "source_completion_pilot_off": 7_517_718,
    "capture_comparison_pilot": 308_940,
    "source_completion_long_on": 8_744_584,
    "plan_pilot": 799_641,
    "plan_long": 1_711_005,
}

SUMMARY_FIELDS = (
    "schema", "name", "component", "encoding", "component_body_bytes_upper",
    "pickle_bytes_upper", "storage_count", "raw_storage_bytes",
    "zip_bytes_upper", "json_bytes", "remaining_component_bytes",
    "tensor_layout_verified", "source_environment_caps_verified",
    "component_bound_satisfied", "semantic_validation_external",
    "storage_fit_proven", "execution_authorized", "scientific_execution_certified",
)


class PayloadAdmissionError(RuntimeError):
    """The name, actual tree, or computed size is outside the admitted domain."""


@dataclass(frozen=True)
class _Spec:
    name: str
    component: str
    kind: str
    role: str | None = None
    bundle: int | None = None
    update: int | None = None
    steps: int | None = None
    capture_mode: str | None = None


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise PayloadAdmissionError(message)


def _same_exact(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    if type(left) is dict:
        return (tuple(left) == tuple(right)
                and all(_same_exact(left[key], right[key]) for key in left))
    if type(left) in (list, tuple):
        return len(left) == len(right) and all(_same_exact(a, b) for a, b in zip(left, right))
    return left == right


@lru_cache(maxsize=1)
def _schedule() -> tuple[_Spec, ...]:
    import core_primitive_bound as core
    import native_phase_policy as policy
    import root_storage_accounting as accounting

    rows: dict[str, _Spec] = {}

    def add(spec: _Spec) -> None:
        _need(spec.name not in rows, "fixed schedule contains duplicate payload")
        rows[spec.name] = spec

    add(_Spec(policy.plan_name(71990), "plan_pilot", "plan",
              role="pilot", bundle=71990, steps=220))
    for bundle in (*policy.PRIMARY, 71901):
        role = "primary" if bundle in policy.PRIMARY else "sensitivity"
        add(_Spec(policy.plan_name(bundle), "plan_long", "plan",
                  role=role, bundle=bundle, steps=2_000))

    for role, bundle, update in core.fixed_identities():
        pilot = role == "pilot"
        for kind, component in (
            ("anchor", "anchor_pilot" if pilot else "anchor_long"),
            ("source-witness", "source_witness"),
            ("branch-results", "branch_results"),
        ):
            name = core.artifact_id(role, bundle, update, kind) + ".pt"
            add(_Spec(name, component, kind, role, bundle, update))
        if not pilot:
            name = core.artifact_id(role, bundle, update, "independent-audit") + ".pt"
            add(_Spec(name, "independent_audit_all_failure", "independent-audit",
                      role, bundle, update))

    add(_Spec(policy.source_name("pilot", 71990), "source_completion_pilot_on",
              "source-completion", "pilot", 71990, steps=220, capture_mode="capture_on"))
    add(_Spec(policy.source_name("pilot", 71990, "capture_off"),
              "source_completion_pilot_off", "source-completion", "pilot", 71990,
              steps=220, capture_mode="capture_off"))
    add(_Spec("i7-native-pilot-b71990-capture-comparison.json",
              "capture_comparison_pilot", "capture-comparison", "pilot", 71990, steps=220))
    for role, bundles in (("primary", policy.PRIMARY), ("sensitivity", (71901,))):
        for bundle in bundles:
            add(_Spec(policy.source_name(role, bundle), "source_completion_long_on",
                      "source-completion", role, bundle, steps=2_000,
                      capture_mode="capture_on"))

    scheduled = accounting.scheduled_payloads()
    _need(len(scheduled) == 82 and set(rows) == {name for name, _ in scheduled},
          "fixed payload schedule differs from accounting")
    _need(all(rows[name].component == component for name, component in scheduled),
          "fixed component classification differs from accounting")
    return tuple(rows[name] for name, _ in scheduled)


def _spec(name: object) -> _Spec:
    _need(type(name) is str, "scheduled payload name must be an exact string")
    found = tuple(row for row in _schedule() if row.name == name)
    _need(len(found) == 1, "payload name is outside the fixed82-member schedule")
    return found[0]


def scheduled_component(name: str) -> str:
    """Return the single fixed component for one exact scheduled payload name."""
    return _spec(name).component


@lru_cache(maxsize=1)
def _fixed_admission() -> tuple[tuple[str, int, int], ...]:
    import native_storage_topology_bound as topology
    import native_tensor_inventory as inventory

    result = topology.compute()
    _need(type(result) is dict
          and result.get("schema") == "i7_prospective_native_storage_topology_bound_v1"
          and result.get("payload_count") == 82
          and result.get("conditional_arithmetic_fits") is True
          and result.get("storage_fit_proven") is False
          and result.get("execution_authorized") is False,
          "prospective topology result is outside the reviewed contract")
    components = result.get("components")
    layouts = inventory.component_layouts()
    _need(type(components) is dict and tuple(components) == tuple(COMPONENT_BODY_CEILINGS)
          and tuple(layouts) == tuple(COMPONENT_BODY_CEILINGS),
          "prospective component membership differs")
    for component, ceiling in COMPONENT_BODY_CEILINGS.items():
        row = components[component]
        _need(type(row) is dict and row.get("body_bytes_upper") == ceiling
              and type(ceiling) is int and ceiling > 0,
              "prospective component ceiling differs from reviewed value")
    return tuple((name, COMPONENT_BODY_CEILINGS[component], len(layouts[component]))
                 for name, component in ((row.name, row.component) for row in _schedule()))


def _ceiling(spec: _Spec) -> tuple[int, int]:
    matches = tuple(row for row in _fixed_admission() if row[0] == spec.name)
    _need(len(matches) == 1, "payload is absent from immutable admission table")
    return matches[0][1], matches[0][2]


def _bounded_primitive(value: Any, *, max_nodes: int, max_depth: int,
                       max_string_bytes: int, allow_tensors: bool = False) -> None:
    torch_type = None
    if allow_tensors:
        import torch
        torch_type = torch.Tensor
    nodes = 0
    active: set[int] = set()

    def visit(item: Any, depth: int) -> None:
        nonlocal nodes
        nodes += 1
        _need(nodes <= max_nodes and depth <= max_depth, "primitive metadata exceeds traversal cap")
        kind = type(item)
        if torch_type is not None and kind is torch_type:
            return
        if item is None or kind in (bool, int):
            return
        if kind is float:
            _need(math.isfinite(item), "primitive metadata contains nonfinite float")
            return
        if kind is str:
            total = 0
            for character in item:
                code = ord(character)
                total += 1 if code <= 0x7F else 2 if code <= 0x7FF else 3 if code <= 0xFFFF else 4
                _need(total <= max_string_bytes, "primitive metadata string exceeds cap")
            return
        _need(kind in (dict, list, tuple), "primitive metadata contains unsupported object")
        identity = id(item)
        _need(identity not in active, "primitive metadata is cyclic")
        active.add(identity)
        try:
            if kind is dict:
                for key, child in item.items():
                    _need(type(key) in (str, int) and type(key) is not bool,
                          "primitive metadata has unsupported key")
                    visit(key, depth + 1)
                    visit(child, depth + 1)
            else:
                for child in item:
                    visit(child, depth + 1)
        finally:
            active.remove(identity)
    visit(value, 0)


def _source_environment(sources: object, environment: object, *, audit: bool) -> tuple[str, str]:
    import identity_codec as codec
    import source_environment_schema as schema

    try:
        schema.validate_sources(sources, profile=PROFILE)
        schema.validate_environment(environment, profile=PROFILE)
    except Exception as exc:
        raise PayloadAdmissionError("source/environment schema rejected") from exc
    _need(sources["repository_root_realpath"] == environment["repository_root_realpath"],
          "source/environment repository roots differ")
    expected_role = "cpu_audit" if audit else "native_source"
    _need(environment["runtime_role"] == expected_role, "runtime role differs from payload phase")
    _bounded_primitive(sources, max_nodes=10_000, max_depth=16,
                       max_string_bytes=SOURCE_JSON_CAP)
    _bounded_primitive(environment, max_nodes=10_000, max_depth=16,
                       max_string_bytes=ENVIRONMENT_JSON_CAP)
    try:
        source_bytes = codec.json_bytes(sources)
        environment_bytes = codec.json_bytes(environment)
    except Exception as exc:
        raise PayloadAdmissionError("source/environment canonical encoding failed") from exc
    _need(0 < len(source_bytes) <= SOURCE_JSON_CAP,
          "source canonical JSON exceeds admitted32768-byte cap")
    _need(0 < len(environment_bytes) <= ENVIRONMENT_JSON_CAP,
          "environment canonical JSON exceeds admitted8192-byte cap")
    if not audit:
        layout = environment["rng_layout"]
        devices = environment["cuda"]["devices"]
        _need(layout["torch_cpu"] == {"dtype": "torch.uint8", "state_length": 5_056}
              and len(layout["torch_cuda"]) == len(devices) == 1
              and layout["torch_cuda"][0]["state_length"] == 16
              and devices[0]["name"] == "NVIDIA GeForce RTX 3090",
              "native RNG/device layout differs from admitted topology")
    else:
        _need(environment["rng_layout"]["torch_cpu"] ==
              {"dtype": "torch.uint8", "state_length": 5_056},
              "audit CPU RNG layout differs from admitted environment")
    return codec.tree_digest(sources), codec.tree_digest(environment)


def validate_runtime_metadata(sources: object, environment: object) -> None:
    """Validate scientific source/environment shape and the admitted byte caps.

    This checks supplied values only. It does not authenticate their collection,
    repository bytes, runtime history, or association with a future payload.
    """
    audit = type(environment) is dict and environment.get("runtime_role") == "cpu_audit"
    _source_environment(sources, environment, audit=audit)


def _metadata_binding(tree: Any, spec: _Spec, sources: Any, environment: Any,
                      source_digest: str, environment_digest: str) -> None:
    import core_primitive_bound as core

    _need(type(tree) is dict and tree.get("profile") == PROFILE,
          "payload root/profile differs")
    if spec.kind in ("anchor", "source-witness", "branch-results", "independent-audit"):
        schemas = {"anchor": "i7_anchor", "source-witness": "i7_source_step_witness",
                   "branch-results": "i7_branch_results",
                   "independent-audit": "i7_anchor_numerical_audit"}
        expected_identity = core.identity(spec.role, spec.bundle, spec.update)
        _need(tree.get("schema_name") == schemas[spec.kind]
              and tree.get("schema_version") == 1
              and tree.get("artifact_id") + ".pt" == spec.name
              and _same_exact(tree.get("identity"), expected_identity),
              "envelope name/schema/identity differs from schedule")
    elif spec.kind == "plan":
        _need(tree.get("schema") == "i7_frozen_plan_v1"
              and tree.get("bundle") == spec.bundle and tree.get("steps_total") == spec.steps,
              "plan metadata differs from schedule")
    elif spec.kind == "source-completion":
        trajectory = tree.get("trajectory")
        _need(tree.get("schema_name") == "i7_source_completion"
              and tree.get("schema_version") == 1 and tree.get("artifact_name") == spec.name
              and tree.get("capture_mode") == spec.capture_mode and type(trajectory) is dict
              and trajectory.get("execution_role") == spec.role
              and trajectory.get("bundle") == spec.bundle
              and trajectory.get("steps_total") == spec.steps,
              "source-completion metadata differs from schedule")
    else:
        raise PayloadAdmissionError("Torch payload kind is unsupported")

    if spec.kind == "anchor":
        bindings = tree.get("payload", {}).get("bindings")
        _need(type(bindings) is dict and _same_exact(bindings.get("sources"), sources)
              and _same_exact(bindings.get("environment"), environment),
              "anchor source/environment binding differs")
    elif spec.kind == "branch-results":
        producer = tree.get("payload", {}).get("audit_metadata", {}).get("producer_bindings")
        _need(type(producer) is dict and producer.get("sources_tree_sha256") == source_digest
              and producer.get("environment_tree_sha256") == environment_digest,
              "branch source/environment digest binding differs")
    elif spec.kind == "independent-audit":
        payload = tree.get("payload", {})
        exact = payload.get("exact_validation") if type(payload) is dict else None
        inputs = payload.get("input_bindings") if type(payload) is dict else None
        _need(type(exact) is dict and type(inputs) is dict
              and _same_exact(exact.get("auditor_environment"), environment)
              and inputs.get("sources_tree_sha256") == source_digest,
              "audit source/environment binding differs")
    elif spec.kind == "source-completion":
        provenance = tree.get("provenance")
        _need(type(provenance) is dict and provenance.get("sources_sha256") == source_digest
              and provenance.get("environment_sha256") == environment_digest,
              "source-completion source/environment binding differs")


def _counter(value: Any, expected: int, label: str) -> None:
    _need(type(value) is int and value == expected, label + " differs")


def _core_counters(core: Any, completed: int, label: str) -> None:
    _need(type(core) is dict, label + " core is absent")
    _counter(core.get("anchor_update"), completed + 1, label + ".anchor_update")
    _counter(core.get("state_completed_updates"), completed, label + ".state_completed_updates")
    optimizer, observer = core.get("optimizer"), core.get("observer")
    _need(type(optimizer) is dict and type(observer) is dict, label + " state metadata absent")
    _counter(optimizer.get("state_completed_updates"), completed, label + ".optimizer")
    _counter(observer.get("state_completed_observations"), completed, label + ".observer")
    state = observer.get("state")
    _need(type(state) is dict, label + " observer state absent")
    _counter(state.get("step_count"), completed, label + ".observer.step_count")
    count = state.get("stabilization_count")
    _need(type(count) is int and 0 <= count <= 2 * completed <= 4_000,
          label + ".stabilization_count exceeds source-derived bound")


def _anchor_counters(payload: Any, completed: int) -> None:
    _need(type(payload) is dict, "anchor payload is absent")
    optimizer, observer = payload.get("optimizer"), payload.get("observer")
    _need(type(optimizer) is dict and type(observer) is dict,
          "anchor state metadata absent")
    _counter(optimizer.get("state_completed_updates"), completed, "anchor.optimizer")
    _counter(observer.get("state_completed_observations"), completed, "anchor.observer")
    state = observer.get("state")
    _need(type(state) is dict, "anchor observer state absent")
    _counter(state.get("step_count"), completed, "anchor.observer.step_count")
    count = state.get("stabilization_count")
    _need(type(count) is int and 0 <= count <= 2 * completed,
          "anchor.stabilization_count exceeds source-derived bound")


def _counter_bindings(tree: dict, spec: _Spec) -> None:
    if spec.kind == "anchor":
        payload = tree["payload"]
        completed = spec.update - 1
        _anchor_counters(payload, completed)
    elif spec.kind == "source-witness":
        payload = tree["payload"]
        step = payload.get("source_step")
        _need(type(step) is dict, "source-step counters absent")
        for key, expected in (("anchor_update", spec.update),
                              ("completed_updates_before", spec.update - 1),
                              ("completed_observations_before", spec.update - 1),
                              ("completed_updates_after", spec.update),
                              ("completed_observations_after", spec.update)):
            _counter(step.get(key), expected, "source_step." + key)
        _counter(payload.get("optimizer_after", {}).get("state_completed_updates"),
                 spec.update, "source witness optimizer")
        observer = payload.get("observer_after", {})
        _counter(observer.get("state_completed_observations"), spec.update,
                 "source witness observer")
        _counter(observer.get("state", {}).get("step_count"), spec.update,
                 "source witness observer step")
        count = observer.get("state", {}).get("stabilization_count")
        _need(type(count) is int and 0 <= count <= 2 * spec.update,
              "source witness stabilization counter differs")
    elif spec.kind == "branch-results":
        payload = tree["payload"]
        observer = payload.get("candidate_state", {}).get("observer_after", {})
        _counter(observer.get("state_completed_observations"), spec.update,
                 "branch candidate observer")
        _counter(observer.get("state", {}).get("step_count"), spec.update,
                 "branch candidate observer step")
        count = observer.get("state", {}).get("stabilization_count")
        _need(type(count) is int and 0 <= count <= 2 * spec.update,
              "branch candidate stabilization counter differs")
        branches = payload.get("branches")
        _need(type(branches) is dict and tuple(branches) == (
            "raw", "current", "lagged", "restored", "reciprocal", "zero"),
            "branch membership differs")
        statuses = {name: row.get("status") if type(row) is dict else None
                    for name, row in branches.items()}
        missing = tuple(name for name, status in statuses.items() if status == "undefined")
        _need(missing in ((), ("restored",), ("reciprocal",))
              and all(status in ("defined", "undefined") for status in statuses.values()),
              "branch domain is outside the three admitted variants")
        expected_reasons = {name: None for name in branches}
        if missing == ("restored",):
            expected_reasons["restored"] = "positive_current_norm_zero_lagged_direction"
        elif missing == ("reciprocal",):
            expected_reasons["reciprocal"] = "positive_lagged_norm_zero_current_direction"
        _need(all(type(row) is dict and row.get("reason") == expected_reasons[name]
                  for name, row in branches.items()),
              "branch domain reason differs from admitted source construction")
        for name, row in branches.items():
            if statuses[name] == "defined":
                _counter(row.get("optimizer_after", {}).get("state_completed_updates"),
                         spec.update, "branch optimizer")
    elif spec.kind == "source-completion":
        _core_counters(tree.get("final_state_core"), spec.steps, "source completion")


def _get(tree: Any, path: tuple[Any, ...]) -> Any:
    value = tree
    for key in path:
        if type(key) is int:
            _need(type(value) in (list, tuple) and 0 <= key < len(value),
                  "tensor path index is absent")
        else:
            _need(type(value) is dict and key in value, "tensor path key is absent")
        value = value[key]
    return value


def _tensor_paths(tree: Any) -> dict[tuple[Any, ...], Any]:
    import torch

    found: dict[tuple[Any, ...], Any] = {}
    active: set[int] = set()

    def walk(value: Any, path: tuple[Any, ...], depth: int) -> None:
        _need(depth <= MAX_DEPTH, "tensor path traversal exceeds depth cap")
        if type(value) is torch.Tensor:
            _need(path not in found, "duplicate tensor path")
            found[path] = value
            return
        if type(value) not in (dict, list, tuple):
            return
        identity = id(value)
        _need(identity not in active, "cyclic tensor container")
        active.add(identity)
        try:
            values = value.items() if type(value) is dict else enumerate(value)
            for key, child in values:
                walk(child, path + (key,), depth + 1)
        finally:
            active.remove(identity)
    walk(tree, (), 0)
    return found


def _canonical_stride(shape: tuple[int, ...]) -> tuple[int, ...]:
    result = []
    for index in range(len(shape)):
        product = 1
        for dimension in shape[index + 1:]:
            product *= dimension
        result.append(product)
    return tuple(result)


def _optional_inventory_path(tree: Any, path: tuple[Any, ...]) -> bool:
    if path[-1] != "value":
        return False
    record_path = path[:-1]
    if record_path[-1] in ("previous_basis", "post_ingest_basis"):
        return _get(tree, record_path) is None
    if len(record_path) >= 2 and record_path[-2:] in (("state", "V"), ("state", "S")):
        return _get(tree, record_path) is None
    return False


def _active_inventory(tree: dict, component: str) -> tuple[dict, ...]:
    import native_tensor_inventory as inventory

    maximum = inventory.component_layouts()[component]
    active = []
    for row in maximum:
        path = row["path"]
        if component == "branch_results" and len(path) >= 4 and path[:2] == (
                "payload", "branches"):
            branch = path[2]
            branch_row = tree["payload"]["branches"][branch]
            if branch_row["status"] == "undefined":
                continue
        try:
            value = _get(tree, path)
        except PayloadAdmissionError:
            if _optional_inventory_path(tree, path):
                continue
            raise
        if value is None and _optional_inventory_path(tree, path):
            continue
        active.append(row)
    return tuple(active)


def _variable_rank_shape(path: tuple[Any, ...], maximum: tuple[int, ...],
                         actual: tuple[int, ...]) -> bool:
    if path[-1] != "value":
        return actual == maximum
    field = path[-2]
    if field in ("V", "previous_basis", "post_ingest_basis"):
        return len(actual) == 2 and actual[0] == 50_890 and 1 <= actual[1] <= 32
    if field == "S":
        return len(actual) == 1 and 1 <= actual[0] <= 32
    return actual == maximum


def _tensor_metadata(path: tuple[Any, ...], tensor: Any, tree: dict, dtype: str,
                     shape: tuple[int, ...]) -> None:
    if not path or path[-1] != "value":
        return
    parent = _get(tree, path[:-1])
    if type(parent) is not dict:
        return
    if all(key in parent for key in ("shape", "native_dtype", "native_device")):
        expected_device = "cpu" if dtype == "float64" else "cuda:0"
        _need(parent["shape"] == list(shape)
              and parent["native_dtype"] == "torch." + dtype
              and parent["native_device"] == expected_device,
              "native tensor metadata differs from admitted device/dtype/shape")
    if all(key in parent for key in ("shape", "dtype", "device", "sha256")):
        _need(parent["shape"] == list(shape) and parent["dtype"] == dtype
              and parent["device"] == "cpu",
              "CPU vector metadata differs from admitted dtype/shape")


def _validate_tensor_layout(tree: dict, component: str) -> tuple[dict[tuple[Any, ...], Any], tuple[dict, ...]]:
    found = _tensor_paths(tree)
    expected = _active_inventory(tree, component)
    rows = {row["path"]: row for row in expected}
    _need(set(found) == set(rows), "actual tensor path membership differs from component topology")
    for path, tensor in found.items():
        row = rows[path]
        shape, maximum = tuple(tensor.shape), tuple(row["shape"])
        _need(str(tensor.dtype) == "torch." + row["dtype"], "actual tensor dtype differs")
        _need(_variable_rank_shape(path, maximum, shape), "actual tensor shape exceeds topology")
        _need(tuple(tensor.stride()) == _canonical_stride(shape),
              "actual tensor stride is not the exact canonical stride")
        _tensor_metadata(path, tensor, tree, row["dtype"], shape)
    return found, expected


def _rng_metadata(tree: dict, environment: dict, spec: _Spec) -> None:
    if spec.kind in ("plan", "independent-audit"):
        return
    cores = []
    if spec.kind == "anchor":
        cores.append(tree["payload"])
    elif spec.kind == "source-completion":
        cores.append(tree["final_state_core"])
    for core in cores:
        rng = core.get("rng") if type(core) is dict else None
        _need(type(rng) is dict and type(rng.get("torch_cuda")) is list
              and len(rng["torch_cuda"]) == 1,
              "retained core CUDA RNG membership differs")
        row = rng["torch_cuda"][0]
        device = environment["cuda"]["devices"][0]
        _need(type(row) is dict and row.get("device_index") == 0
              and row.get("name") == device["name"] and row.get("uuid") == device["uuid"],
              "retained core CUDA RNG identity differs from environment")


def _summary(spec: _Spec, ceiling: int, *, pickle_bytes: int | None,
             storage_count: int | None, raw_bytes: int | None,
             zip_bytes: int | None, json_bytes: int | None) -> dict:
    body = zip_bytes if zip_bytes is not None else json_bytes
    _need(type(body) is int and 0 < body <= ceiling, "actual body bound exceeds component ceiling")
    result = {
        "schema": SCHEMA, "name": spec.name, "component": spec.component,
        "encoding": "bytes" if spec.kind == "capture-comparison" else "torch_weights_only",
        "component_body_bytes_upper": ceiling, "pickle_bytes_upper": pickle_bytes,
        "storage_count": storage_count, "raw_storage_bytes": raw_bytes,
        "zip_bytes_upper": zip_bytes, "json_bytes": json_bytes,
        "remaining_component_bytes": ceiling - body,
        "tensor_layout_verified": True, "source_environment_caps_verified": True,
        "component_bound_satisfied": True,
        "semantic_validation_external": "required_immediately_before_guard_not_attested",
        "storage_fit_proven": False, "execution_authorized": False,
        "scientific_execution_certified": False,
    }
    _need(tuple(result) == SUMMARY_FIELDS, "internal summary field order differs")
    return result


def admit_torch_payload(name: str, tree: Any, *, sources: Any, environment: Any) -> dict:
    """Bound one actual scheduled Torch tree after its upstream semantic validator.

    The caller must exclude mutation from semantic validation through this call
    and the immediately following explicitly configured ``torch.save``.
    """
    import pickle_storage_bound as pickle_bound
    import zip_storage_bound as zip_bound

    spec = _spec(name)
    _need(spec.kind != "capture-comparison", "JSON comparison requires admit_json_payload")
    source_digest, environment_digest = _source_environment(
        sources, environment, audit=spec.kind == "independent-audit")
    _bounded_primitive(tree, max_nodes=MAX_NODES, max_depth=MAX_DEPTH,
                       max_string_bytes=(AUDIT_MAX_UTF8_BYTES
                                         if spec.kind == "independent-audit"
                                         else OTHER_MAX_UTF8_BYTES), allow_tensors=True)
    _metadata_binding(tree, spec, sources, environment, source_digest, environment_digest)
    _counter_bindings(tree, spec)
    found, expected = _validate_tensor_layout(tree, spec.component)
    _rng_metadata(tree, environment, spec)
    ceiling, tensor_cap = _ceiling(spec)
    _need(len(expected) <= tensor_cap, "active tensor count exceeds maximum inventory")
    allowed = frozenset("torch." + row["dtype"] for row in expected)
    try:
        bounded = pickle_bound.bound_protocol2_tree(
            tree, tensor_count_cap=tensor_cap, allowed_dtypes=allowed,
            max_depth=MAX_DEPTH, max_nodes=MAX_NODES,
            max_utf8_bytes=(AUDIT_MAX_UTF8_BYTES if spec.kind == "independent-audit"
                            else OTHER_MAX_UTF8_BYTES), max_tensor_rank=2,
        )
    except Exception as exc:
        raise PayloadAdmissionError("actual protocol2 tree is outside admitted domain") from exc
    actual_storage_order = tuple(tensor.untyped_storage().nbytes() for tensor in found.values())
    _need(bounded.tensor_count == len(found)
          and bounded.storage_nbytes == actual_storage_order,
          "protocol2 traversal differs from path inventory")
    if spec.kind in ("anchor", "source-witness", "branch-results", "independent-audit"):
        _need(tree.get("payload_tensor_bytes") == sum(bounded.storage_nbytes),
              "envelope payload_tensor_bytes differs from actual storages")
    try:
        zipped = zip_bound.torch_save_zip_ceiling(
            bounded.pickle_bytes, bounded.storage_nbytes,
            buffer_limit_bytes=zip_bound.BUFFER_LIMIT_BYTES,
        )
    except Exception as exc:
        raise PayloadAdmissionError("actual ZIP prediction is outside admitted domain") from exc
    _need(zipped["archive_bytes_upper"] <= ceiling,
          "actual ZIP upper bound exceeds scheduled component ceiling")
    return _summary(spec, ceiling, pickle_bytes=bounded.pickle_bytes,
                    storage_count=bounded.tensor_count,
                    raw_bytes=sum(bounded.storage_nbytes),
                    zip_bytes=zipped["archive_bytes_upper"], json_bytes=None)


def admit_json_payload(name: str, value: Any, *, sources: Any, environment: Any) -> dict:
    """Validate and exactly encode the one bounded canonical JSON comparison."""
    import identity_codec as codec
    import source_history as history

    spec = _spec(name)
    _need(spec.kind == "capture-comparison", "Torch payload requires admit_torch_payload")
    _source_environment(sources, environment, audit=False)
    _bounded_primitive(value, max_nodes=COMPARISON_MAX_NODES, max_depth=COMPARISON_MAX_DEPTH,
                       max_string_bytes=COMPARISON_MAX_STRING_BYTES)
    try:
        history.validate_capture_comparison(value)
        identity_bytes = codec.json_bytes(value)
    except Exception as exc:
        raise PayloadAdmissionError("capture comparison validation/encoding failed") from exc
    _need(value.get("profile") == PROFILE and value.get("artifact_name") == spec.name,
          "capture comparison name/profile differs from schedule")
    ceiling, tensor_cap = _ceiling(spec)
    _need(tensor_cap == 0 and 0 < len(identity_bytes) <= ceiling,
          "capture comparison exceeds scheduled component ceiling")
    return _summary(spec, ceiling, pickle_bytes=None, storage_count=None, raw_bytes=None,
                    zip_bytes=None, json_bytes=len(identity_bytes))


if __name__ == "__main__":
    print("I7 actual-payload storage guard is library-only; no payload, write, CUDA or execution.")
