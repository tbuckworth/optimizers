"""Torch-free exact source/environment schemas shared by collector and inspector.

These validate supplied values only: no process, device, RNG or Git observation.
Profile literals mirror artifact_store; tests check that they remain equal.
"""
from __future__ import annotations

import os
import re
from typing import Any

from cuda_identity import is_canonical_cuda_uuid

SCIENTIFIC = "scientific_mnist_current32_v1"
FIXTURE = "fixture_tiny_cpu_v1"
MLP_FIXTURE = "fixture_tiny_mlp_cpu_v1"


class SourceEnvironmentError(ValueError):
    pass


I7 = "output/2026-09-06-spectral-optimizer-investigation/continuation/iteration-007/"
SOURCE_FILE_CAP = 2 << 20
SOURCE_TOTAL_CAP = 16 << 20
METADATA_CAP = 1 << 20
COMMAND_TIMEOUT = 5.0
SOURCE_KEYS = ("schema", "profile", "repository_revision", "git_object_format",
               "repository_root_realpath", "worktree_status", "files")
SOURCE_FILE_KEYS = ("path", "role", "git_mode", "size_bytes", "sha256", "git_blob_oid")
ENVIRONMENT_KEYS = ("schema", "profile", "runtime_role", "repository_root_realpath",
                    "python", "libraries", "operating_system", "safe_environment",
                    "torch_settings", "cuda", "rng_layout")
PYTHON_KEYS = ("implementation", "version", "version_info", "cache_tag", "executable_realpath")
LIBRARY_KEYS = ("numpy_version", "torch_version", "torch_git_revision",
                "torch_cuda_build", "cudnn_version")
OS_KEYS = ("sys_platform", "system", "release", "machine", "libc_name", "libc_version")
SAFE_ENV_KEYS = ("CUBLAS_WORKSPACE_CONFIG", "CUDA_VISIBLE_DEVICES", "MKL_NUM_THREADS",
                 "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "PYTHONHASHSEED")
TORCH_KEYS = ("num_threads", "num_interop_threads", "default_dtype", "default_device",
              "deterministic_algorithms", "deterministic_warn_only", "grad_enabled",
              "inference_mode_enabled", "autocast_cpu_enabled", "autocast_cuda_enabled",
              "autocast_cpu_dtype", "autocast_cuda_dtype", "float32_matmul_precision",
              "cudnn_enabled", "cudnn_benchmark", "cudnn_deterministic",
              "cuda_matmul_allow_tf32", "cudnn_allow_tf32",
              "allow_fp16_reduced_precision_reduction",
              "allow_bf16_reduced_precision_reduction")
CUDA_KEYS = ("initialized", "visible_device_count", "current_device", "driver_version", "devices")
CUDA_DEVICE_KEYS = ("index", "name", "uuid", "total_memory_bytes",
                    "capability_major", "capability_minor")
RNG_KEYS = ("python", "numpy", "torch_cpu", "torch_cuda")
PY_RNG_KEYS = ("state_version", "internal_length")
NP_RNG_KEYS = ("algorithm", "keys_dtype", "keys_length")
TORCH_RNG_KEYS = ("dtype", "state_length")
CUDA_RNG_KEYS = ("index", "name", "uuid", "dtype", "state_length")
PROFILES = (SCIENTIFIC, FIXTURE, MLP_FIXTURE)
FIXTURE_ROLES = (("producer", "fixture_source.py"), ("contract", "fixture_contract.md"))
SCIENTIFIC_ROLES = (
    ("canonical_filter", "spectral_filter.py"),
    ("binding_codec", I7 + "artifact_store.py"),
    ("binding_codec", I7 + "identity_codec.py"),
    ("binding_codec", I7 + "plan_bindings.py"),
    ("binding_codec", I7 + "verified_plan_load.py"),
    ("binding_codec", I7 + "data_probe_bindings.py"),
    ("state_codec", I7 + "state_core.py"),
    ("binding_codec", I7 + "cuda_identity.py"),
    ("source_binding", I7 + "source_environment_schema.py"),
    ("producer", I7 + "response_math.py"),
    ("producer", I7 + "loss_measurements.py"),
    ("producer", I7 + "measurement_assembly.py"),
    ("independent_auditor", I7 + "independent_numerics.py"),
    ("independent_auditor", I7 + "measurement_audit.py"),
    ("independent_auditor", I7 + "audit_diagnostics.py"),
    ("independent_auditor", I7 + "audit_envelope.py"),
    ("runtime_guard", I7 + "runtime_guard.py"),
    ("source_binding", I7 + "source_environment.py"),
    ("producer", I7 + "source_capture.py"),
    ("producer", I7 + "branch_execution.py"),
    ("envelope", I7 + "anchor_envelope.py"),
    ("envelope", I7 + "artifact_envelopes.py"),
    ("phase_controller", I7 + "phase_controller.py"),
    ("phase_controller", I7 + "native_phase_policy.py"),
    ("producer", I7 + "source_history.py"),
    ("scientific_entrypoint", I7 + "scientific_runner.py"),
    ("audit_entrypoint", I7 + "audit_runner.py"),
    ("phase_controller", I7 + "native_control.py"),
    ("producer", I7 + "native_inputs.py"),
    ("producer", I7 + "source_streaming.py"),
    ("producer", I7 + "native_source.py"),
    ("producer", I7 + "native_branches.py"),
    ("phase_controller", I7 + "native_controller.py"),
    ("phase_controller", I7 + "phase_transition.py"),
    ("phase_controller", I7 + "scientific_controller.py"),
    ("independent_auditor", I7 + "native_audit.py"),
    ("producer", I7 + "result_collection.py"),
    ("scientific_entrypoint", I7 + "phase_worker.py"),
    ("phase_controller", I7 + "process_supervision.py"),
    ("runtime_guard", I7 + "final_storage_accounting.py"),
    ("runtime_guard", I7 + "native_layout_inspection.py"),
    ("runtime_guard", I7 + "pickle_storage_bound.py"),
    ("runtime_guard", I7 + "zip_storage_bound.py"),
    ("runtime_guard", I7 + "native_tensor_inventory.py"),
    ("runtime_guard", I7 + "comparison_storage_bound.py"),
    ("runtime_guard", I7 + "primitive_storage_bound.py"),
    ("runtime_guard", I7 + "core_primitive_bound.py"),
    ("runtime_guard", I7 + "branch_primitive_bound.py"),
    ("runtime_guard", I7 + "audit_primitive_bound.py"),
    ("runtime_guard", I7 + "native_storage_topology_bound.py"),
    ("runtime_guard", I7 + "native_payload_guard.py"),
    ("runtime_guard", I7 + "native_write_ledger.py"),
    ("runtime_guard", I7 + "root_storage_accounting.py"),
    ("runtime_guard", I7 + "native_storage_authority.py"),
    ("runtime_guard", I7 + "native_storage_recipe_common.py"),
    ("runtime_guard", I7 + "native_storage_recipe_core.py"),
    ("runtime_guard", I7 + "native_storage_recipe_branch_audit.py"),
    ("runtime_guard", I7 + "native_storage_crosscheck.py"),
    ("runtime_guard", I7 + "native_storage_measurement_service.py"),
    ("runtime_guard", I7 + "native_storage_measurement_controller.py"),
    ("runtime_guard", I7 + "native_storage_measurement_slot.py"),
    ("runtime_guard", I7 + "native_storage_measurement_worker.py"),
    ("contract", I7 + "common-state-design.md"),
    ("contract", I7 + "analysis-spec.md"),
    ("contract", I7 + "anchor-schema.md"),
    ("contract", I7 + "measurement-schema.md"),
    ("contract", I7 + "numerical-contract.md"),
    ("contract", I7 + "assembly-roundoff.md"),
    ("contract", I7 + "resource-contract.md"),
    ("contract", I7 + "identity-binding-contract.md"),
    ("contract", I7 + "data-probe-contract.md"),
    ("contract", I7 + "verified-load-decision.md"),
    ("contract", I7 + "source-environment-proposal.md"),
    ("contract", I7 + "source-environment-decision.md"),
    ("contract", I7 + "anchor-envelope-contract.md"),
    ("contract", I7 + "branch-envelope-contract.md"),
    ("contract", I7 + "envelope-contract.md"),
    ("contract", I7 + "phase-controller-contract.md"),
    ("contract", I7 + "native-records-contract.md"),
    ("contract", I7 + "audit-envelope-contract.md"),
)
_ROLES = frozenset(role for role, _ in SCIENTIFIC_ROLES + FIXTURE_ROLES)
_SHA256 = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)
_OID = {"sha1": re.compile(r"[0-9a-f]{40}\Z", re.ASCII),
        "sha256": re.compile(r"[0-9a-f]{64}\Z", re.ASCII)}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SourceEnvironmentError(message)


def _keys(value: Any, expected: tuple[str, ...], where: str) -> None:
    _require(type(value) is dict and tuple(value) == expected and
             all(type(key) is str for key in value), f"{where}: exact ordered keys required")


def _string(value: Any, where: str, *, ascii_only: bool = False, nonempty: bool = True) -> str:
    _require(type(value) is str and (not nonempty or bool(value)), f"{where}: invalid string")
    _require(not any(0xD800 <= ord(char) <= 0xDFFF for char in value),
             f"{where}: surrogate code point")
    _require(not ascii_only or value.isascii(), f"{where}: ASCII required")
    return value


def _integer(value: Any, where: str, minimum: int = 0) -> int:
    _require(type(value) is int and value >= minimum, f"{where}: invalid integer")
    return value


def _canonical_absolute(value: Any, where: str) -> str:
    path = _string(value, where, ascii_only=True)
    _require(path.startswith("/") and not path.startswith("//") and path != "/" and
             os.path.normpath(path) == path,
             f"{where}: noncanonical absolute path")
    return path


def _membership(profile: str) -> tuple[tuple[str, str], ...]:
    _require(type(profile) is str and profile in PROFILES, "unknown profile")
    return SCIENTIFIC_ROLES if profile == SCIENTIFIC else FIXTURE_ROLES


def _source_path(value: Any) -> str:
    path = _string(value, "source path", ascii_only=True)
    parts = path.split("/")
    _require(not path.startswith("/") and "\\" not in path and
             all(part not in ("", ".", "..") for part in parts), "invalid source path")
    return path


def validate_sources(value, *, profile):
    """Pure exact schema/policy validation; does not inspect Git or files."""
    membership = _membership(profile)
    _keys(value, SOURCE_KEYS, "sources")
    _require(type(value["schema"]) is str and value["schema"] == "i7_source_binding_v1" and
             type(value["profile"]) is str and value["profile"] == profile,
             "source schema/profile mismatch")
    object_format = value["git_object_format"]
    _require(type(object_format) is str and object_format in _OID, "unsupported Git object format")
    revision = _string(value["repository_revision"], "repository revision", ascii_only=True)
    _require(_OID[object_format].fullmatch(revision) is not None, "invalid repository revision")
    _canonical_absolute(value["repository_root_realpath"], "repository root")
    _require(type(value["worktree_status"]) is str and
             value["worktree_status"] == "clean_including_untracked_v1",
             "source worktree status mismatch")
    files = value["files"]
    _require(type(files) is list and len(files) == len(membership), "source membership count mismatch")
    total = 0
    for index, (record, expected) in enumerate(zip(files, membership)):
        _keys(record, SOURCE_FILE_KEYS, f"source file {index}")
        role, path = expected
        _require(type(record["role"]) is str and record["role"] == role and
                 record["path"] == path and role in _ROLES,
                 f"source file {index}: membership mismatch")
        _source_path(record["path"])
        _require(type(record["git_mode"]) is str and record["git_mode"] in ("100644", "100755"),
                 "invalid source Git mode")
        size = _integer(record["size_bytes"], "source size")
        _require(size <= SOURCE_FILE_CAP, "source file exceeds cap")
        total += size
        sha = _string(record["sha256"], "source SHA-256", ascii_only=True)
        oid = _string(record["git_blob_oid"], "source Git blob", ascii_only=True)
        _require(_SHA256.fullmatch(sha) is not None and _OID[object_format].fullmatch(oid) is not None,
                 "invalid source digest")
    _require(total <= SOURCE_TOTAL_CAP, "source set exceeds total cap")
    return value


def _optional_string(value: Any, where: str) -> None:
    _require(value is None or type(value) is str, f"{where}: string or None required")
    if type(value) is str:
        _string(value, where, nonempty=False)


def _bools(mapping: dict[str, Any], names: tuple[str, ...], where: str) -> None:
    _require(all(type(mapping[name]) is bool for name in names), f"{where}: exact booleans required")


def validate_environment(value, *, profile):
    """Pure exact schema/policy validation; does not inspect this process."""
    _membership(profile)
    _keys(value, ENVIRONMENT_KEYS, "environment")
    _require(type(value["schema"]) is str and value["schema"] == "i7_environment_binding_v1" and
             type(value["profile"]) is str and value["profile"] == profile,
             "environment schema/profile mismatch")
    role = value["runtime_role"]
    allowed = ("native_source", "cpu_audit", "storage_crosscheck_cpu") if profile == SCIENTIFIC else ("fixture_cpu",)
    _require(type(role) is str and role in allowed, "runtime role/profile mismatch")
    root = _canonical_absolute(value["repository_root_realpath"], "environment repository root")

    py = value["python"]
    _keys(py, PYTHON_KEYS, "python")
    for name in ("implementation", "version", "cache_tag"):
        _string(py[name], "python." + name)
    _canonical_absolute(py["executable_realpath"], "Python executable")
    _require(type(py["version_info"]) is list and len(py["version_info"]) == 3 and
             all(type(item) is int and item >= 0 for item in py["version_info"]),
             "invalid Python version_info")

    libraries = value["libraries"]
    _keys(libraries, LIBRARY_KEYS, "libraries")
    for name in ("numpy_version", "torch_version", "torch_git_revision"):
        _string(libraries[name], "libraries." + name)
    _optional_string(libraries["torch_cuda_build"], "libraries.torch_cuda_build")
    _require(libraries["cudnn_version"] is None or
             (type(libraries["cudnn_version"]) is int and libraries["cudnn_version"] >= 0),
             "invalid cuDNN version")

    os_row = value["operating_system"]
    _keys(os_row, OS_KEYS, "operating_system")
    for name in OS_KEYS:
        _string(os_row[name], "operating_system." + name, nonempty=False)

    safe = value["safe_environment"]
    _keys(safe, SAFE_ENV_KEYS, "safe_environment")
    for name in SAFE_ENV_KEYS:
        _optional_string(safe[name], "safe_environment." + name)

    settings = value["torch_settings"]
    _keys(settings, TORCH_KEYS, "torch_settings")
    _integer(settings["num_threads"], "num_threads", 1)
    _integer(settings["num_interop_threads"], "num_interop_threads", 1)
    for name in ("default_dtype", "default_device", "autocast_cpu_dtype",
                 "autocast_cuda_dtype", "float32_matmul_precision"):
        _string(settings[name], "torch_settings." + name)
    bool_fields = tuple(name for name in TORCH_KEYS if name not in {
        "num_threads", "num_interop_threads", "default_dtype", "default_device",
        "autocast_cpu_dtype", "autocast_cuda_dtype", "float32_matmul_precision"})
    _bools(settings, bool_fields, "torch_settings")

    cuda = value["cuda"]
    _keys(cuda, CUDA_KEYS, "cuda")
    _require(type(cuda["initialized"]) is bool and type(cuda["devices"]) is list,
             "invalid CUDA root fields")
    for name in ("visible_device_count", "current_device"):
        _require(cuda[name] is None or (type(cuda[name]) is int and cuda[name] >= 0),
                 "invalid CUDA index/count")
    _optional_string(cuda["driver_version"], "CUDA driver version")
    for index, device in enumerate(cuda["devices"]):
        _keys(device, CUDA_DEVICE_KEYS, f"CUDA device {index}")
        _require(type(device["index"]) is int and device["index"] == index,
                 "CUDA device order mismatch")
        _string(device["name"], "CUDA device name")
        _string(device["uuid"], "CUDA device UUID")
        _require(is_canonical_cuda_uuid(device["uuid"]), "noncanonical CUDA device UUID")
        for name in ("total_memory_bytes", "capability_major", "capability_minor"):
            _integer(device[name], "CUDA device " + name)
        _require(device["total_memory_bytes"] > 0, "CUDA memory must be positive")

    layout = value["rng_layout"]
    _keys(layout, RNG_KEYS, "rng_layout")
    _keys(layout["python"], PY_RNG_KEYS, "Python RNG layout")
    _integer(layout["python"]["state_version"], "Python RNG state version")
    _integer(layout["python"]["internal_length"], "Python RNG internal length", 1)
    _keys(layout["numpy"], NP_RNG_KEYS, "NumPy RNG layout")
    _string(layout["numpy"]["algorithm"], "NumPy RNG algorithm")
    _string(layout["numpy"]["keys_dtype"], "NumPy RNG dtype")
    _integer(layout["numpy"]["keys_length"], "NumPy RNG key length", 1)
    _keys(layout["torch_cpu"], TORCH_RNG_KEYS, "Torch CPU RNG layout")
    _string(layout["torch_cpu"]["dtype"], "Torch CPU RNG dtype")
    _integer(layout["torch_cpu"]["state_length"], "Torch CPU RNG length", 1)
    _require(type(layout["torch_cuda"]) is list, "Torch CUDA RNG layouts must be a list")
    for index, rng in enumerate(layout["torch_cuda"]):
        _keys(rng, CUDA_RNG_KEYS, f"CUDA RNG layout {index}")
        _require(type(rng["index"]) is int and rng["index"] == index,
                 "CUDA RNG layout order mismatch")
        _string(rng["name"], "CUDA RNG name")
        _string(rng["uuid"], "CUDA RNG UUID")
        _require(is_canonical_cuda_uuid(rng["uuid"]), "noncanonical CUDA RNG UUID")
        _require(rng["dtype"] == "torch.uint8", "CUDA RNG dtype mismatch")
        _integer(rng["state_length"], "CUDA RNG length", 1)

    _require(layout["python"] == {"state_version": 3, "internal_length": 625} and
             layout["numpy"] == {"algorithm": "MT19937", "keys_dtype": "uint32",
                                  "keys_length": 624} and
             layout["torch_cpu"]["dtype"] == "torch.uint8", "unsupported RNG layout")
    if profile == SCIENTIFIC:
        _require(py["implementation"] == "CPython" and py["version"] == "3.12.3" and
                 py["version_info"] == [3, 12, 3] and
                 libraries["numpy_version"] == "1.26.4" and
                 libraries["torch_version"] == "2.11.0+cu128" and
                 libraries["torch_git_revision"] == "70d99e998b4955e0049d13a98d77ae1b14db1f45" and
                 libraries["torch_cuda_build"] == "12.8" and
                 libraries["cudnn_version"] == 91900 and os_row["system"] == "Linux" and
                 os_row["machine"] == "x86_64", "scientific software/platform mismatch")
        _require(safe["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8" and
                 safe["OMP_NUM_THREADS"] == safe["OPENBLAS_NUM_THREADS"] ==
                 safe["MKL_NUM_THREADS"] == "1", "scientific numerical environment mismatch")
        _require(settings["num_threads"] == 1 and settings["default_dtype"] == "torch.float32" and
                 settings["default_device"] == "cpu" and
                 settings["deterministic_algorithms"] and not settings["deterministic_warn_only"] and
                 settings["grad_enabled"] and not settings["inference_mode_enabled"] and
                 not settings["autocast_cpu_enabled"] and not settings["autocast_cuda_enabled"] and
                 settings["float32_matmul_precision"] == "highest" and
                 not settings["cudnn_benchmark"] and not settings["cuda_matmul_allow_tf32"] and
                 not settings["cudnn_allow_tf32"], "scientific Torch settings mismatch")
    if role == "native_source":
        _require(cuda["initialized"] and cuda["visible_device_count"] == 1 and
                 cuda["current_device"] == 0 and type(cuda["driver_version"]) is str and
                 len(cuda["devices"]) == len(layout["torch_cuda"]) == 1,
                 "native source requires one initialized CUDA device")
        device, rng = cuda["devices"][0], layout["torch_cuda"][0]
        _require(device["name"] == "NVIDIA GeForce RTX 3090" and bool(device["uuid"]) and
                 (rng["index"], rng["name"], rng["uuid"]) ==
                 (device["index"], device["name"], device["uuid"]),
                 "native CUDA identity mismatch")
    else:
        _require(not cuda["initialized"] and cuda["visible_device_count"] is None and
                 cuda["current_device"] is None and cuda["driver_version"] is None and
                 cuda["devices"] == [] and layout["torch_cuda"] == [],
                 "CPU role must not claim CUDA inspection")
    if role == "storage_crosscheck_cpu":
        _require(safe["CUDA_VISIBLE_DEVICES"] == "" and settings["num_interop_threads"] == 1
                 and layout["torch_cpu"]["state_length"] == 5056,
                 "storage crosscheck requires hidden CUDA and fixed CPU settings/layout")
    _require(root == value["repository_root_realpath"], "repository root mismatch")
    return value
