#!/usr/bin/env python3
"""Read-only measurement of the completed 35-state grokking action corpus."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import resource
import shutil
import subprocess
import sys
import time
from typing import Any

# Pin CPU math libraries before importing NumPy/Torch-backed analysis helpers.
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from experiments.analyze_grokking_representations import (  # noqa: E402
    RAW_KEYS, analysis_sources as prior_analysis_sources, analyze_state,
)
from experiments.grokking_action_intervention import (  # noqa: E402
    CAPTURE_STEPS, POLICIES, SCHEMA as ACTION_SCHEMA, source_pins as action_source_pins,
)
from experiments.grokking_confirmation import (  # noqa: E402
    CHECKPOINT_SCHEMA, FILTER_MUTABLE, MODEL, atomic_exclusive, canonical_hash,
    environment as confirmation_environment, file_hash, finite_tree, load_checkpoint,
    parameter_identity, tensor_set_identity,
)
from experiments.grokking_model import (  # noqa: E402
    GrokkingTransformer, get_modular_addition_data,
)
from experiments.grokking_representation import (  # noqa: E402
    evaluate_fourier_probes, extract_activations, make_probe_split,
    make_row_permutations,
)
from experiments.measure_grokking_representations import (  # noqa: E402
    RECIPE, behavior, corpus, sources as prior_measurement_sources,
)

SEEDS = tuple(range(100, 105))
FORK_STEP = 1500
FIXED_PANEL = {"frequencies": [9, 33, 32, 49, 11]}
OUTPUT_LIMIT = 3 * 1024**3
OUTPUT_RESERVE = 1024**3
MEMORY_LIMIT = 16 * 1024**3
COOPERATIVE_SECONDS = 19 * 60
PRIOR_SUMMARY = (REPO / "output/2026-09-09-spectral-grokking-mechanism/results/summary.json")
PRIOR_SUMMARY_SHA256 = "9ac881c1bbca72a9dfd5525c226e05938ea4b0553e1e924bafb59ffd8a9a7c0d"
ACTION_CHECKPOINT_SCHEMA = "grokking_action_checkpoint_v1"
INNER_KEYS = {
    "schema", "step", "model_state", "optimizer_state", "filter_state",
    "filter_configuration", "torch_cpu_rng_state", "torch_cuda_rng_states",
    "device_type", "cuda_device_count", "config", "config_sha256",
    "source_identity", "source_identity_sha256", "split_identity",
    "parameter_identity", "evaluation_rows", "timing",
}
OUTER_KEYS = {
    "schema", "policy", "seed", "step", "parent_checkpoint", "source_sha256",
    "shared_first_action", "native_compatible_state", "note",
}


def expected_roster() -> list[tuple[int, str, int]]:
    """The only admissible new-state roster, in acquisition order."""
    return [
        (seed, policy, step)
        for seed in SEEDS
        for policy in POLICIES
        for step in ((1501,) if policy == "native" else CAPTURE_STEPS)
    ]


def _read_json(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"not a regular JSON file: {path}")
    info = path.stat()
    if info.st_nlink != 1:
        raise ValueError(f"JSON must be singly linked: {path}")
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError(f"JSON root is not an object: {path}")
    return value, {"path": str(path.resolve()), "sha256": file_hash(path),
                   "size_bytes": info.st_size}


def _verify_receipt(receipt: dict[str, Any], parent: Path,
                    *, exact_name: str | None = None) -> dict[str, Any]:
    required = {"path", "sha256"}
    if not isinstance(receipt, dict) or not required.issubset(receipt):
        raise ValueError("malformed file receipt")
    path = Path(receipt["path"])
    if (path.is_symlink() or not path.is_file() or path.stat().st_nlink != 1
            or path.resolve().parent != parent.resolve()
            or (exact_name is not None and path.name != exact_name)):
        raise ValueError("receipt path/type/parent mismatch")
    if "size_bytes" in receipt and path.stat().st_size != receipt["size_bytes"]:
        raise ValueError("receipt size mismatch")
    if file_hash(path) != receipt["sha256"]:
        raise ValueError("receipt hash mismatch")
    return {"path": str(path.resolve()), "sha256": receipt["sha256"],
            "size_bytes": path.stat().st_size}


def _parent_receipt(record: dict[str, Any]) -> dict[str, Any]:
    matches = [row for row in record["checkpoints"]
               if Path(row["path"]).name == "checkpoint-step-001500.pt"]
    if len(matches) != 1:
        raise ValueError("legacy source lacks one step-1500 checkpoint")
    return matches[0]


def verify_action_batch(parent: Path, records: dict,
                        expected_sources: dict[str, str]) -> dict[str, Any]:
    """Admit only a fully completed, immutable five-seed action batch."""
    parent = Path(parent).resolve()
    storage = Path("/tmp/spectral-experiment-artifacts").resolve()
    if parent.parent != storage or not parent.is_dir():
        raise ValueError("action batch must be one directory directly under large-volume tmp")
    manifest, manifest_receipt = _read_json(parent / "batch-manifest.json")
    complete, complete_receipt = _read_json(parent / "batch-complete.json")
    if (manifest.get("schema") != "grokking_action_batch_v1"
            or manifest.get("seeds") != list(SEEDS)
            or manifest.get("first_seed_is_resource_only_admission") is not True
            or manifest.get("paid_spend_usd") != 0
            or complete.get("status") != "complete"
            or complete.get("seeds") != list(SEEDS)
            or complete.get("paid_spend_usd") != 0
            or not isinstance(manifest.get("source_commit"), str)
            or len(complete.get("accepted", [])) != len(SEEDS)):
        raise ValueError("batch manifest/completion contract mismatch")
    expected_seed_roster = [[policy, step] for policy in POLICIES
                            for step in ((1501,) if policy == "native" else CAPTURE_STEPS)]
    if re.fullmatch(r"[0-9a-f]{40}", manifest["source_commit"]) is None:
        raise ValueError("batch source commit is not a lowercase Git identity")
    states, input_receipts, environments = [], [manifest_receipt, complete_receipt], []
    seed_commits = {}
    for index, seed in enumerate(SEEDS):
        seed_dir = parent / f"seed{seed}"
        accepted = complete["accepted"][index]
        if accepted.get("seed") != seed:
            raise ValueError("batch accepted seed order/identity mismatch")
        completion_receipt = _verify_receipt(
            accepted, seed_dir, exact_name="complete.json")
        seed_complete, reread_receipt = _read_json(seed_dir / "complete.json")
        if completion_receipt["sha256"] != reread_receipt["sha256"]:
            raise ValueError("seed completion changed while admitted")
        seed_manifest, seed_manifest_receipt = _read_json(seed_dir / "manifest.json")
        common_keys = (
            "schema", "seed", "parent_checkpoint", "parent_metrics_sha256",
            "source_sha256", "source_commit", "environment",
            "fork_scientific_state_sha256", "capture_steps", "policies",
            "native_policy_is_one_step_diagnostic_only",
        )
        if (seed_complete.get("status") != "complete"
                or any(seed_complete.get(key) != seed_manifest.get(key) for key in common_keys)
                or seed_complete.get("schema") != ACTION_SCHEMA
                or seed_complete.get("seed") != seed
                or seed_complete.get("capture_steps") != list(CAPTURE_STEPS)
                or seed_complete.get("policies") != list(POLICIES)
                or seed_complete.get("native_policy_is_one_step_diagnostic_only") is not True
                or seed_complete.get("accepted_roster") != expected_seed_roster
                or seed_complete.get("source_sha256") != expected_sources):
            raise ValueError("seed manifest/completion/source contract mismatch")
        seed_commit = seed_complete.get("source_commit")
        if not isinstance(seed_commit, str) or re.fullmatch(r"[0-9a-f]{40}", seed_commit) is None:
            raise ValueError("seed source commit is not a lowercase Git identity")
        seed_commits[seed] = seed_commit
        record, record_hash = records[seed, "legacy"]
        if (seed_complete.get("parent_checkpoint") != _parent_receipt(record)
                or seed_complete.get("parent_metrics_sha256") != record_hash):
            raise ValueError("action seed is not bound to the accepted legacy fork")
        environments.append(seed_complete.get("environment"))
        input_receipts.extend((seed_manifest_receipt, completion_receipt))
        for key, name in (("shared_first_action", "shared-first-action.pt"),
                          ("first_step_adam_diagnostics", "first-step-adam-diagnostics.pt")):
            input_receipts.append(_verify_receipt(seed_complete[key], seed_dir,
                                                  exact_name=name))
        checkpoints = seed_complete.get("checkpoints", [])
        if len(checkpoints) != 7:
            raise ValueError("seed does not contain exactly seven action checkpoints")
        checkpoint_roster = []
        for receipt in checkpoints:
            identity = (receipt.get("policy"), receipt.get("step"))
            checkpoint_roster.append(list(identity))
            expected_name = f"{identity[0]}-step-{identity[1]:06d}.pt"
            verified = _verify_receipt(receipt, seed_dir, exact_name=expected_name)
            input_receipts.append(verified)
            states.append({"seed": seed, "policy": identity[0], "step": identity[1],
                           "checkpoint": verified, "seed_completion": completion_receipt,
                           "seed_manifest": seed_manifest_receipt,
                           "completion": seed_complete, "record": record,
                           "parent_metrics_sha256": record_hash})
        if checkpoint_roster != expected_seed_roster:
            raise ValueError("checkpoint roster/order differs from completion roster")
        disk_names = {path.name for path in seed_dir.iterdir()
                      if re.fullmatch(r"(?:native|orthogonal|norm_matched)-step-\d{6}\.pt",
                                      path.name)}
        expected_names = {f"{policy}-step-{step:06d}.pt"
                          for policy, step in expected_seed_roster}
        if disk_names != expected_names:
            raise ValueError("listed and on-disk action checkpoint sets differ")
        if (seed_dir / "failure.json").exists():
            raise ValueError("complete seed also contains a failure marker")
    if any(environment != environments[0] for environment in environments[1:]):
        raise ValueError("action seed environments differ")
    if (parent / "batch-failure.json").exists():
        raise ValueError("complete batch also contains a failure marker")
    identities = [(row["seed"], row["policy"], row["step"]) for row in states]
    if identities != expected_roster() or len(set(identities)) != 35:
        raise ValueError("action input is not the exact ordered 35-state roster")
    return {"path": str(parent), "manifest": manifest,
            "manifest_receipt": manifest_receipt,
            "completion_receipt": complete_receipt,
            "states": states, "input_receipts": input_receipts,
            "environment": environments[0], "source_sha256": expected_sources,
            "source_commits": {"batch_start": manifest["source_commit"],
                               "seeds": seed_commits}}


def validate_envelope(envelope: dict[str, Any], item: dict[str, Any],
                      split_identity: dict[str, Any]) -> dict[str, Any]:
    """Validate outer intervention truth separately from inherited native truth."""
    seed, policy, step = item["seed"], item["policy"], item["step"]
    completion, record = item["completion"], item["record"]
    if not isinstance(envelope, dict) or set(envelope) != OUTER_KEYS:
        raise ValueError("intervention checkpoint envelope schema mismatch")
    if (envelope["schema"] != ACTION_CHECKPOINT_SCHEMA
            or (envelope["seed"], envelope["policy"], envelope["step"])
               != (seed, policy, step)
            or envelope["source_sha256"] != completion["source_sha256"]
            or envelope["parent_checkpoint"] != completion["parent_checkpoint"]
            or envelope["shared_first_action"] != completion["shared_first_action"]):
        raise ValueError("outer intervention policy/seed/step/source mismatch")
    inner = envelope["native_compatible_state"]
    if not isinstance(inner, dict) or set(inner) != INNER_KEYS or not finite_tree(inner):
        raise ValueError("invalid inherited checkpoint state")
    if (inner["schema"] != CHECKPOINT_SCHEMA or inner["step"] != step
            or inner["config"] != record["config"]
            or inner["config_sha256"] != canonical_hash(record["config"])
            or inner["source_identity"] != record["source_identity"]
            or inner["source_identity_sha256"] != canonical_hash(record["source_identity"])
            or inner["split_identity"] != split_identity
            or inner["split_identity"] != record["split_identity"]
            or inner["parameter_identity"] != record["parameter_identity"]
            or (inner["config"]["seed"], inner["config"]["arm"]) != (seed, "legacy")):
        raise ValueError("inherited config/source/split/parameter identity mismatch")
    filter_state, configuration = inner["filter_state"], inner["filter_configuration"]
    if (not isinstance(filter_state, dict) or set(filter_state) != set(FILTER_MUTABLE)
            or filter_state["step_count"] != step or not isinstance(configuration, dict)
            or configuration.get("stable_update") is not False):
        raise ValueError("inherited legacy filter identity/counter mismatch")
    expected_evaluations = [] if step == 1501 else list(range(1550, step + 1, 50))
    rows = inner["evaluation_rows"]
    if (policy == "native" and step != 1501) or [row.get("step") for row in rows] != expected_evaluations:
        raise ValueError("intervention evaluation prefix mismatch")
    return inner


def _array_hash(name: str, value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(name.encode() + b"\0" + str(array.dtype).encode() + b"\0")
    digest.update(json.dumps(list(array.shape)).encode() + b"\0")
    digest.update(array.tobytes())
    return digest.hexdigest()


def _structural_arrays(seed: int, all_pairs: torch.Tensor,
                       all_sums: torch.Tensor) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    data = get_modular_addition_data(p=RECIPE["p"], train_frac=.3, seed=seed)
    train_ids = torch.sort(data[0][:, 0] * RECIPE["p"] + data[0][:, 1]).values
    test_ids = torch.sort(data[2][:, 0] * RECIPE["p"] + data[2][:, 1]).values
    y = all_sums[test_ids]
    split = make_probe_split(y, RECIPE["split_seed_offset"] + seed)
    permutations = make_row_permutations(
        len(y), RECIPE["null_seed_offset"] + seed, RECIPE["n_nulls"])
    arrays = {
        "pairs": all_pairs.numpy(), "sums": all_sums.numpy(),
        "train_ids": train_ids.numpy(), "test_ids": test_ids.numpy(),
        "probe_fit_indices": np.asarray(split["fit_indices"], dtype=np.int64),
        "probe_eval_indices": np.asarray(split["eval_indices"], dtype=np.int64),
        "null_permutations": np.asarray(
            [row["permutation"] for row in permutations], dtype=np.int64),
    }
    return arrays, {"data": data, "train_ids": train_ids, "test_ids": test_ids,
                    "y": y, "split": split, "permutations": permutations}


def validate_original_source_bindings(
        acquisition_sources: dict[str, str], scalar_sources: list[dict[str, str]],
        analysis_sources: dict[str, str], current_acquisition_sources: dict[str, str],
        current_analysis_sources: dict[str, str]) -> None:
    """Bind reused numerical helpers to both accepted source generations."""
    if acquisition_sources != current_acquisition_sources:
        raise ValueError("current original acquisition sources differ from accepted pins")
    if not scalar_sources or any(value != acquisition_sources for value in scalar_sources):
        raise ValueError("prior native scalars do not share accepted acquisition sources")
    if analysis_sources != current_analysis_sources:
        raise ValueError("current analyzer/symmetry sources differ from accepted analysis pins")


def current_cuda_environments() -> tuple[dict[str, Any], dict[str, Any]]:
    """Return the exact old-readout fields and observable action-run fields."""
    if not torch.cuda.is_available():
        raise RuntimeError("accepted readout requires CUDA")
    prior = {"python": platform.python_version(), "torch": str(torch.__version__),
             "numpy": np.__version__, "device": "cuda",
             "gpu": torch.cuda.get_device_name()}
    batch = confirmation_environment(torch.device("cuda"), 1)
    batch["torch"] = str(batch["torch"])
    batch["current_backend_flags"] = {
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
    }
    return prior, batch


def validate_environment_contract(calibration: dict[str, Any], remaining: dict[str, Any],
                                  batch: dict[str, Any], current_prior: dict[str, Any],
                                  current_batch: dict[str, Any]) -> None:
    expected_prior_keys = {"python", "torch", "numpy", "device", "gpu"}
    if (set(calibration) != expected_prior_keys or calibration != remaining
            or calibration.get("device") != "cuda" or current_prior != calibration):
        raise ValueError("current CUDA readout environment differs from accepted prior environment")
    batch_keys = ("python", "platform", "torch", "device", "cuda_runtime", "cudnn",
                  "cuda_device_name", "cuda_capability", "torch_cpu_threads",
                  "current_backend_flags")
    if any(key not in batch or current_batch.get(key) != batch.get(key) for key in batch_keys):
        raise ValueError("current environment differs from accepted action-batch environment")
    crosswalk = (("python", "python"), ("torch", "torch"), ("device", "device"),
                 ("gpu", "cuda_device_name"))
    if any(calibration[left] != batch[right] for left, right in crosswalk):
        raise ValueError("accepted prior-readout and action-batch environments disagree")


def load_prior_recipe_contracts(all_pairs: torch.Tensor,
                                all_sums: torch.Tensor) -> tuple[dict, list[dict]]:
    """Prove the generated split/null arrays equal the archived native-1500 arrays."""
    summary, summary_receipt = _read_json(PRIOR_SUMMARY)
    if summary_receipt["sha256"] != PRIOR_SUMMARY_SHA256:
        raise ValueError("prior accepted analysis summary hash changed")
    if summary.get("fixed_frequency_panel", {}).get("frequencies") != FIXED_PANEL["frequencies"]:
        raise ValueError("prior fixed frequency panel changed")
    input_receipts = summary.get("input_receipts", {})
    manifests, manifest_receipts = {}, {}
    for stage in ("calibration", "remaining"):
        supplied = input_receipts.get(stage, {}).get("manifest_receipt")
        if not isinstance(supplied, dict) or "path" not in supplied:
            raise ValueError("accepted summary lacks a prior measurement manifest receipt")
        path = Path(supplied["path"])
        verified = _verify_receipt(supplied, path.resolve().parent,
                                   exact_name="manifest.json")
        manifest, reread = _read_json(path)
        if (verified["sha256"] != reread["sha256"]
                or manifest.get("schema") != "grokking_representation_measurement_v1"
                or manifest.get("stage") != stage or manifest.get("recipe") != RECIPE):
            raise ValueError("accepted prior measurement manifest contract changed")
        manifests[stage], manifest_receipts[stage] = manifest, verified
    accepted_acquisition_sources = manifests["remaining"].get("source_sha256")
    current_acquisition_sources = prior_measurement_sources()
    current_analysis = prior_analysis_sources()
    if (manifests["calibration"].get("source_sha256") != accepted_acquisition_sources
            or accepted_acquisition_sources != current_acquisition_sources
            or summary.get("analysis_source_sha256") != current_analysis):
        raise ValueError("accepted original readout/analysis source binding changed")
    raw_rows = summary.get("input_receipts", {}).get("remaining", {}).get("raw_receipts", [])
    result_rows = summary.get("input_receipts", {}).get("remaining", {}).get("result_receipts", [])
    raw_by_key = {(row["seed"], row["arm"], row["step"]): row for row in raw_rows}
    result_by_key = {(row["seed"], row["arm"], row["step"]): row for row in result_rows}
    contracts = {}
    receipts = [summary_receipt, manifest_receipts["calibration"],
                manifest_receipts["remaining"]]
    scalar_source_pins = []
    for seed in SEEDS:
        key = (seed, "legacy", FORK_STEP)
        if key not in raw_by_key or key not in result_by_key:
            raise ValueError("prior summary lacks one legacy step-1500 recipe source")
        raw_receipt, scalar_receipt = raw_by_key[key], result_by_key[key]
        raw_path, scalar_path = Path(raw_receipt["path"]), Path(scalar_receipt["path"])
        raw_verified = _verify_receipt(raw_receipt, raw_path.resolve().parent)
        scalar_verified = _verify_receipt(scalar_receipt, scalar_path.resolve().parent)
        scalar, reread = _read_json(scalar_path)
        if (reread["sha256"] != scalar_verified["sha256"]
                or (scalar.get("seed"), scalar.get("arm"), scalar.get("step")) != key
                or scalar.get("raw_activations", {}).get("sha256") != raw_verified["sha256"]
                or scalar.get("source_sha256") != accepted_acquisition_sources):
            raise ValueError("prior scalar/raw receipt binding mismatch")
        scalar_source_pins.append(scalar["source_sha256"])
        expected, recipe = _structural_arrays(seed, all_pairs, all_sums)
        with np.load(raw_path, allow_pickle=False) as archive:
            if set(archive.files) != RAW_KEYS:
                raise ValueError("prior raw member schema changed")
            hashes = {}
            for name, wanted in expected.items():
                actual = archive[name]
                if (actual.dtype != wanted.dtype or actual.shape != wanted.shape
                        or not np.array_equal(actual, wanted)):
                    raise ValueError(f"generated {name} differs from prior seed-{seed} recipe")
                hashes[name] = _array_hash(name, actual)
        if scalar.get("probe_split_sha256") != recipe["split"]["sha256"]:
            raise ValueError("generated probe split hash differs from prior scalar")
        contracts[seed] = {"raw": raw_verified, "scalar": scalar_verified,
                           "probe_split_sha256": recipe["split"]["sha256"],
                           "structural_array_sha256": hashes}
        receipts.extend((raw_verified, scalar_verified))
    validate_original_source_bindings(
        accepted_acquisition_sources, scalar_source_pins,
        summary["analysis_source_sha256"], current_acquisition_sources, current_analysis)
    return {"summary": summary_receipt, "manifests": manifest_receipts,
            "environments": {stage: manifests[stage]["environment"]
                             for stage in ("calibration", "remaining")},
            "acquisition_source_sha256": accepted_acquisition_sources,
            "analysis_source_sha256": summary["analysis_source_sha256"],
            "seeds": contracts}, receipts


def measurement_sources() -> dict[str, str]:
    paths = (
        "experiments/measure_grokking_action_states.py",
        "tests/test_grokking_action_measurement.py",
        "experiments/grokking_representation.py",
        "experiments/measure_grokking_representations.py",
        "experiments/analyze_grokking_representations.py",
        "experiments/grokking_symmetry.py", "experiments/grokking_model.py",
        "experiments/grokking_confirmation.py",
        "experiments/grokking_action_intervention.py",
        "experiments/run_grokking_action_batch.py",
        "experiments/grokking_action_policy.py",
        "output/2026-09-09-spectral-grokking-action/protocol.md",
        "output/2026-09-09-spectral-grokking-action/measurement-protocol.md",
        "output/2026-09-09-spectral-grokking-mechanism/analysis-protocol.md",
    )
    return {path: file_hash(REPO / path) for path in paths}


def _peak_rss_bytes() -> int:
    # Linux reports ru_maxrss in KiB.
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


class Output:
    def __init__(self, path: Path):
        storage = Path("/tmp/spectral-experiment-artifacts").resolve()
        target = Path(path).resolve()
        if (target == storage or not target.is_relative_to(storage) or target.exists()
                or not target.parent.is_dir()):
            raise ValueError("output must be a new descendant of large-volume tmp")
        if shutil.disk_usage(target.parent).free < OUTPUT_LIMIT + OUTPUT_RESERVE:
            raise ValueError("insufficient space for output bound plus reserve")
        target.mkdir(mode=0o700)
        for name in ("raw", "scalars", "states"):
            (target / name).mkdir()
        self.path, self.bytes = target, 0

    def _admit(self, upper_bound: int) -> None:
        if self.bytes + upper_bound > OUTPUT_LIMIT:
            raise RuntimeError("measurement output bound exhausted")
        if shutil.disk_usage(self.path).free < upper_bound + OUTPUT_RESERVE:
            raise RuntimeError("large-volume free-space reserve exhausted")

    def write_bytes(self, relative: Path | str, raw: bytes) -> dict[str, Any]:
        self._admit(len(raw))
        path = self.path / relative
        atomic_exclusive(path, lambda handle: handle.write(raw))
        self.bytes += path.stat().st_size
        return {"path": str(path.resolve()), "sha256": file_hash(path),
                "size_bytes": path.stat().st_size}

    def write_json(self, relative: Path | str, value: Any) -> dict[str, Any]:
        raw = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
        return self.write_bytes(relative, raw)

    def write_npz(self, relative: Path | str,
                  arrays: dict[str, np.ndarray]) -> dict[str, Any]:
        upper = sum(array.nbytes for array in arrays.values()) + 1024**2
        self._admit(upper)
        path = self.path / relative
        atomic_exclusive(path, lambda handle: np.savez_compressed(handle, **arrays))
        self.bytes += path.stat().st_size
        return {"path": str(path.resolve()), "sha256": file_hash(path),
                "size_bytes": path.stat().st_size}


def _check_resources(started: float, output: Output | None = None) -> None:
    if time.monotonic() - started > COOPERATIVE_SECONDS:
        raise TimeoutError("20-minute service deadline approached; preserve partial output")
    if _peak_rss_bytes() > MEMORY_LIMIT:
        raise MemoryError("16-GiB process memory bound exceeded")
    if output is not None:
        output._admit(0)


def _recheck(receipts: list[dict[str, Any]]) -> None:
    for receipt in receipts:
        path = Path(receipt["path"])
        if (path.is_symlink() or not path.is_file() or path.stat().st_nlink != 1
                or path.stat().st_size != receipt["size_bytes"]
                or file_hash(path) != receipt["sha256"]):
            raise ValueError("an admitted input changed during measurement")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action-batch-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError("accepted CUDA readout environment unavailable")
    started = time.monotonic()
    records = corpus()
    action_pins = action_source_pins()
    batch = verify_action_batch(args.action_batch_dir, records, action_pins)
    all_pairs = torch.cartesian_prod(torch.arange(RECIPE["p"]), torch.arange(RECIPE["p"]))
    all_sums = all_pairs.sum(-1) % RECIPE["p"]
    prior, prior_receipts = load_prior_recipe_contracts(all_pairs, all_sums)
    current_prior_environment, current_batch_environment = current_cuda_environments()
    validate_environment_contract(
        prior["environments"]["calibration"], prior["environments"]["remaining"],
        batch["environment"], current_prior_environment, current_batch_environment)
    pins = measurement_sources()
    output = Output(args.output_dir)
    manifest = {
        "schema": "grokking_action_measurement_v1", "roster": expected_roster(),
        "action_batch": {"path": batch["path"],
                         "manifest": batch["manifest_receipt"],
                         "completion": batch["completion_receipt"],
                         "source_commits": batch["source_commits"]},
        "action_source_sha256": action_pins, "measurement_source_sha256": pins,
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "prior_recipe": prior, "recipe": RECIPE, "fixed_frequency_panel": FIXED_PANEL,
        "environment": current_prior_environment,
        "action_environment": current_batch_environment,
        "bounds": {"cooperative_seconds": COOPERATIVE_SECONDS,
                   "cpu_threads": 1, "memory_bytes": MEMORY_LIMIT,
                   "output_bytes": OUTPUT_LIMIT, "free_reserve_bytes": OUTPUT_RESERVE},
        "interpretation": "read-only per-state measurements; no causal aggregation",
    }
    manifest_receipt = output.write_json("manifest.json", manifest)
    accepted = []
    try:
        for item in batch["states"]:
            _check_resources(started, output)
            seed, policy, step = item["seed"], item["policy"], item["step"]
            structural, recipe = _structural_arrays(seed, all_pairs, all_sums)
            prior_seed = prior["seeds"][seed]
            if (recipe["split"]["sha256"] != prior_seed["probe_split_sha256"]
                    or any(_array_hash(name, array)
                           != prior_seed["structural_array_sha256"][name]
                           for name, array in structural.items())):
                raise ValueError("in-memory recipe differs from admitted prior arrays")
            envelope = load_checkpoint(Path(item["checkpoint"]["path"]),
                                       item["checkpoint"]["sha256"])
            inner = validate_envelope(
                envelope, item, tensor_set_identity(recipe["data"]))
            model = GrokkingTransformer(**MODEL).to(args.device)
            model.load_state_dict(inner["model_state"], strict=True)
            if parameter_identity(model) != inner["parameter_identity"]:
                raise ValueError("loaded intervention model parameter layout changed")
            if any(not bool(torch.isfinite(value).all())
                   for value in model.state_dict().values()
                   if value.is_floating_point()):
                raise ValueError("nonfinite intervention model state")
            rng_before = {"cpu": torch.get_rng_state().clone(),
                          "cuda": [value.clone() for value in torch.cuda.get_rng_state_all()]}
            activations = extract_activations(
                model, all_pairs, batch_size=RECIPE["batch_size"])
            if (not torch.equal(rng_before["cpu"], torch.get_rng_state())
                    or any(not torch.equal(left, right) for left, right in zip(
                        rng_before["cuda"], torch.cuda.get_rng_state_all()))):
                raise ValueError("activation extraction changed global RNG")
            if any(not bool(torch.isfinite(value).all()) for value in activations.values()):
                raise ValueError("nonfinite extracted action-state activation")
            arrays = {name: value.numpy() for name, value in activations.items()}
            arrays.update(structural)
            stem = f"seed{seed}-{policy}-step{step:06d}"
            raw_receipt = output.write_npz(Path("raw") / f"{stem}.npz", arrays)
            behavior_record = {
                name: behavior(activations["logits"], all_sums, ids)
                for name, ids in (("train", recipe["train_ids"]),
                                  ("test", recipe["test_ids"]))
            }
            matching_eval = [row for row in inner["evaluation_rows"] if row["step"] == step]
            if matching_eval:
                for name in ("train", "test"):
                    if (abs(behavior_record[name]["loss"] - matching_eval[0][name]["loss"]) > 1e-4
                            or abs(behavior_record[name]["accuracy"]
                                   - matching_eval[0][name]["accuracy"]) > 1e-6):
                        raise ValueError("saved state does not reproduce its training-grid behavior")
            probes = {
                feature: evaluate_fourier_probes(
                    activations[feature][recipe["test_ids"]], recipe["y"],
                    recipe["split"]["fit_indices"], recipe["split"]["eval_indices"],
                    p=RECIPE["p"], ridge=RECIPE["ridge"], top_k=RECIPE["top_k"],
                    null_seed=RECIPE["null_seed_offset"] + seed,
                    n_nulls=RECIPE["n_nulls"],
                    null_permutations=recipe["permutations"])
                for feature in RECIPE["features"]
            }
            checkpoint_provenance = {
                "action_checkpoint": item["checkpoint"],
                "action_seed_completion": item["seed_completion"],
                "action_seed_manifest": item["seed_manifest"],
                "parent_legacy_checkpoint": envelope["parent_checkpoint"],
                "parent_metrics_sha256": item["parent_metrics_sha256"],
                "action_source_sha256": envelope["source_sha256"],
                "outer_policy": policy, "outer_seed": seed, "outer_step": step,
            }
            result = {
                "schema": "grokking_action_measurement_state_v1",
                "seed": seed, "arm": policy, "policy": policy, "step": step,
                "checkpoint_provenance": checkpoint_provenance,
                "source_sha256": pins, "probe_split_sha256": recipe["split"]["sha256"],
                "prior_recipe_contract": prior_seed,
                "raw_activations": raw_receipt, "behavior": behavior_record,
                "probes": probes, "elapsed_seconds": time.monotonic() - started,
            }
            scalar_receipt = output.write_json(Path("scalars") / f"{stem}.json", result)
            record = {"stage": "action", "result": result,
                      "result_path": Path(scalar_receipt["path"]),
                      "result_sha256": scalar_receipt["sha256"],
                      "result_bytes": Path(scalar_receipt["path"]).read_bytes(),
                      "raw_path": Path(raw_receipt["path"]),
                      "raw_sha256": raw_receipt["sha256"],
                      "raw_size_bytes": raw_receipt["size_bytes"]}
            analyzed = analyze_state(record, FIXED_PANEL)
            analyzed["policy"] = policy
            analyzed["checkpoint_provenance"] = checkpoint_provenance
            analyzed["prior_recipe_contract"] = prior_seed
            analyzed["source"]["action_checkpoint_sha256"] = item["checkpoint"]["sha256"]
            state_receipt = output.write_json(Path("states") / f"{stem}.json", analyzed)
            accepted.append({"seed": seed, "policy": policy, "step": step,
                             "checkpoint": item["checkpoint"], "raw": raw_receipt,
                             "scalar": scalar_receipt, "analyzed_state": state_receipt})
            print(json.dumps({"measured": [seed, policy, step],
                              "output_bytes": output.bytes}), flush=True)
            del envelope, inner, model, activations, arrays, probes, result, analyzed
        if [(row["seed"], row["policy"], row["step"]) for row in accepted] != expected_roster():
            raise ValueError("completed measurement roster differs from fixed 35 states")
        _check_resources(started, output)
        if (measurement_sources() != pins or action_source_pins() != action_pins
                or prior_measurement_sources() != prior["acquisition_source_sha256"]
                or prior_analysis_sources() != prior["analysis_source_sha256"]):
            raise ValueError("source changed during action-state measurement")
        _recheck(batch["input_receipts"] + prior_receipts)
        complete = {
            "schema": "grokking_action_measurement_complete_v1", "status": "complete",
            "state_count": 35, "accepted_results": accepted,
            "manifest": manifest_receipt, "measurement_source_sha256": pins,
            "action_source_sha256": action_pins,
            "input_batch_completion": batch["completion_receipt"],
            "prior_summary": prior["summary"],
            "elapsed_seconds": time.monotonic() - started,
            "peak_rss_bytes": _peak_rss_bytes(),
            "output_bytes_before_completion": output.bytes,
            "interpretation": "per-state measurements only; no aggregate causal conclusion",
        }
        output.write_json("complete.json", complete)
    except BaseException as error:
        try:
            output.write_json("failure.json", {
                "schema": "grokking_action_measurement_failure_v1",
                "status": "failed_preserved", "type": type(error).__name__,
                "message": str(error), "accepted_results": accepted,
                "manifest": manifest_receipt,
                "elapsed_seconds": time.monotonic() - started,
                "peak_rss_bytes": _peak_rss_bytes(),
                "output_bytes_before_failure": output.bytes,
            })
        except BaseException:
            pass
        raise


if __name__ == "__main__":
    main()
