#!/usr/bin/env python3
"""Three new saved-action AdamW counterfactuals per seed; never replay raw.

No backward, gradient/observer/SVD recomputation, probe fitting or continuation.
Before/raw logits and the raw after-state are reused from accepted artifacts.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import re
import resource
import shutil
import subprocess
import sys
import time

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from experiments.grokking_action_intervention import scientific_hash, tensor_bytes, tree_hash
from experiments.grokking_confirmation import (
    ADAMW, CHECKPOINT_SCHEMA, MODEL, atomic_exclusive, clone_cpu, file_hash, finite_tree,
    load_checkpoint, parameter_identity, save_checkpoint,
)
from experiments.grokking_model import GrokkingTransformer
from experiments.grokking_representation import extract_activations
from experiments.grokking_function_response import analyze_seed, paired_summary
from experiments.measure_grokking_action_states import (
    _array_hash, _read_json, _verify_receipt, current_cuda_environments,
    validate_environment_contract,
)
from experiments.measure_grokking_raw_direction_states import (
    measurement_sources as accepted_measurement_sources,
)

STORAGE = Path("/tmp/spectral-experiment-artifacts")
BATCH = STORAGE / "spectral-grokking-raw-direction-20260909.KzGtkl"
BATCH_SHA = "8c0040c193a1f7d755004b002a3da70f315db3e41a959a5789d15844ad5b6f3b"
MEASUREMENT = BATCH / "measurement-001"
MEASUREMENT_SHA = "ecbb06431733302f6d6b61b8770c781076676b9ca65f244e2bd9bc467d681d46"
PRIOR_SHA = "9ac881c1bbca72a9dfd5525c226e05938ea4b0553e1e924bafb59ffd8a9a7c0d"
SEEDS = tuple(range(100, 105))
POLICY = "raw_norm_matched"
NEW_ACTIONS = ("trunc", "projected", "zero")
STRUCTURAL = ("pairs", "sums", "train_ids", "test_ids", "probe_fit_indices",
              "probe_eval_indices", "null_permutations")
MAX_BYTES, FREE_RESERVE, MEMORY_BYTES, SECONDS = 2 * 1024**3, 1024**3, 16 * 1024**3, 480
SCHEMA = "grokking_function_response_measurement_v1"


def source_pins():
    paths = ("experiments/measure_grokking_function_response.py",
             "experiments/grokking_function_response.py",
             "tests/test_grokking_function_response_measurement.py",
             "tests/test_grokking_function_response.py",
             "tests/test_grokking_function_response_guard.py",
             "output/2026-09-09-spectral-function-response/protocol.md",
             "output/2026-09-09-spectral-function-response/guarded_launch.py")
    return {**accepted_measurement_sources(), **{name: file_hash(REPO / name) for name in paths}}


def _identity(receipt):
    return {key: receipt[key] for key in ("path", "sha256", "size_bytes")}


def _bound_json(receipt, parent, name, receipts):
    verified = _verify_receipt(receipt, parent, exact_name=name)
    value, reread = _read_json(Path(verified["path"]))
    if reread != verified:
        raise ValueError("JSON changed during admission")
    receipts.append(verified)
    return value


def admit_inputs():
    """Read only accepted receipt JSON here; tensor/array loading is per-seed."""
    batch, batch_receipt = _read_json(BATCH / "batch-complete.json")
    measured, measured_receipt = _read_json(MEASUREMENT / "complete.json")
    if (batch_receipt["sha256"] != BATCH_SHA or measured_receipt["sha256"] != MEASUREMENT_SHA
            or batch.get("schema") != "grokking_raw_direction_batch_v1" or batch.get("status") != "complete"
            or batch.get("seeds") != list(SEEDS) or batch.get("policy") != POLICY
            or [entry.get("seed") for entry in batch.get("accepted", [])] != list(SEEDS)
            or measured.get("schema") != "grokking_raw_direction_measurement_complete_v1"
            or measured.get("status") != "complete" or measured.get("state_count") != 15
            or measured.get("input_batch_completion") != batch_receipt
            or (BATCH / "batch-failure.json").exists() or (MEASUREMENT / "failure.json").exists()):
        raise ValueError("pinned completed batch/measurement identity mismatch")
    receipts = [batch_receipt, measured_receipt]
    batch_manifest = _bound_json(batch["batch_manifest"], BATCH, "batch-manifest.json", receipts)
    manifest = _bound_json(measured["manifest"], MEASUREMENT, "manifest.json", receipts)
    if (any(batch.get(key) != value for key, value in batch_manifest.items())
            or measured.get("measurement_source_sha256") != accepted_measurement_sources()
            or manifest.get("measurement_source_sha256") != measured["measurement_source_sha256"]
            or manifest.get("raw_batch", {}).get("completion") != batch_receipt
            or manifest.get("acquisition_source_sha256") != batch["source_sha256"]):
        raise ValueError("accepted source/batch manifest binding mismatch")
    prior = manifest["prior_recipe"]
    prior_receipt = prior["summary"]
    if prior_receipt["sha256"] != PRIOR_SHA or measured["prior_summary"] != prior_receipt:
        raise ValueError("accepted before-readout summary mismatch")
    _bound_json(prior_receipt, Path(prior_receipt["path"]).parent, "summary.json", receipts)
    for stage in ("calibration", "remaining"):
        item = prior["manifests"][stage]
        prior_manifest = _bound_json(item, Path(item["path"]).parent, "manifest.json", receipts)
        if prior_manifest.get("source_sha256") != prior["acquisition_source_sha256"]:
            raise ValueError("before-readout source binding mismatch")
    expected = [(seed, POLICY, step) for seed in SEEDS for step in (1501, 2000, 2500)]
    if [(r["seed"], r["policy"], r["step"]) for r in measured["accepted_results"]] != expected:
        raise ValueError("accepted measurement roster changed")
    by_seed = {r["seed"]: r for r in measured["accepted_results"] if r["step"] == 1501}
    items = []
    for seed, completion in zip(SEEDS, batch["accepted"]):
        directory = BATCH / f"seed{seed}"
        seed_complete = _bound_json(completion, directory, "complete.json", receipts)
        if (seed_complete.get("seed") != seed or seed_complete.get("policy") != POLICY
                or seed_complete.get("status") != "complete" or seed_complete.get("source_sha256") != batch["source_sha256"]
                or seed_complete.get("completed_updates") != 1000 or (directory / "failure.json").exists()):
            raise ValueError("accepted seed completion mismatch")
        artifacts = {Path(r["path"]).name: r for r in seed_complete["artifact_receipts"]}
        first = _verify_receipt(artifacts["first-step-tensors.pt"], directory, exact_name="first-step-tensors.pt")
        checkpoint = _verify_receipt(artifacts[f"{POLICY}-step-001501.pt"], directory,
                                      exact_name=f"{POLICY}-step-001501.pt")
        accepted = by_seed[seed]
        if accepted["checkpoint"] != checkpoint:
            raise ValueError("raw-after measurement/checkpoint mismatch")
        stem = f"seed{seed}-{POLICY}-step001501"
        scalar = _bound_json(accepted["scalar"], MEASUREMENT / "scalars", stem + ".json", receipts)
        state = _bound_json(accepted["analyzed_state"], MEASUREMENT / "states", stem + ".json", receipts)
        raw = _verify_receipt(accepted["raw"], MEASUREMENT / "raw", exact_name=stem + ".npz")
        contract = prior["seeds"][str(seed)]
        before_scalar = _bound_json(contract["scalar"], Path(contract["scalar"]["path"]).parent,
                                    f"seed{seed}-legacy-step001500.json", receipts)
        before_raw = _verify_receipt(contract["raw"], Path(contract["raw"]["path"]).parent,
                                     exact_name=f"seed{seed}-legacy-step001500.npz")
        provenance = scalar.get("checkpoint_provenance", {})
        if ((scalar.get("seed"), scalar.get("policy"), scalar.get("step")) != (seed, POLICY, 1501)
                or scalar.get("source_sha256") != measured["measurement_source_sha256"]
                or scalar.get("raw_activations") != raw or scalar.get("prior_recipe_contract") != contract
                or provenance != state.get("checkpoint_provenance")
                or provenance.get("raw_direction_checkpoint") != checkpoint
                or provenance.get("parent_legacy_checkpoint") != seed_complete["parent_checkpoint"]
                or state.get("source", {}).get("source_scalar_sha256") != accepted["scalar"]["sha256"]
                or state.get("source", {}).get("source_npz_sha256") != raw["sha256"]
                or (before_scalar.get("seed"), before_scalar.get("arm"), before_scalar.get("step")) != (seed, "legacy", 1500)
                or before_scalar.get("checkpoint") != seed_complete["parent_checkpoint"]
                or before_scalar.get("source_sha256") != prior["acquisition_source_sha256"]
                or before_scalar.get("raw_activations") != before_raw):
            raise ValueError("before/raw archived-logit/checkpoint provenance mismatch")
        receipts.extend((first, checkpoint, raw, before_raw))
        items.append({"seed": seed, "completion": seed_complete, "first": first, "checkpoint": checkpoint,
                      "before_raw": before_raw, "raw_after": raw, "recipe": contract})
    current_prior, current_batch = current_cuda_environments()
    validate_environment_contract(prior["environments"]["calibration"], prior["environments"]["remaining"],
                                  batch["environment"], current_prior, current_batch)
    if manifest["environment"] != current_prior:
        raise ValueError("raw-after archived readout environment mismatch")
    return {"items": items, "receipts": list({r["path"]: r for r in receipts}.values()),
            "environment": current_prior, "acquisition_environment": current_batch}


def archived_logits(before_path, raw_path, contract, p=113):
    results, structure = {}, None
    for name, path in (("before", before_path), ("raw", raw_path)):
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != set(STRUCTURAL) | {"logits", "final_hidden", "pre_attention"}:
                raise ValueError("archived activation member schema mismatch")
            current = {key: archive[key] for key in STRUCTURAL}
            logits = archive["logits"]
            if (logits.shape != (p * p, p) or logits.dtype != np.float32 or not np.isfinite(logits).all()
                    or any(_array_hash(key, value) != contract["structural_array_sha256"][key]
                           for key, value in current.items())):
                raise ValueError("archived logits/structural hash mismatch")
            if structure is not None and any(not np.array_equal(current[k], structure[k]) for k in STRUCTURAL):
                raise ValueError("before/raw archived grids differ")
            results[name], structure = logits.copy(), current
    pairs = np.column_stack((np.repeat(np.arange(p), p), np.tile(np.arange(p), p)))
    if (not np.array_equal(structure["pairs"], pairs)
            or not np.array_equal(structure["sums"], pairs.sum(axis=1) % p)
            or not np.array_equal(np.sort(np.concatenate((structure["train_ids"], structure["test_ids"]))), np.arange(p * p))):
        raise ValueError("archived full-grid/split contract mismatch")
    return results, structure


def actions_from_saved(first, device):
    result = first["actions"]
    raw = result["actions"][POLICY].detach().clone().to(device)
    projected = result["actions"]["norm_matched"].detach().clone().to(device)
    q = result["Q"].detach().to(device)
    gradient = first["raw_gradient"].detach().to(device)
    if (raw.ndim != 1 or projected.shape != raw.shape or projected.dtype != raw.dtype
            or raw.dtype != torch.float32 or gradient.shape != raw.shape or gradient.dtype != raw.dtype
            or q.ndim != 2 or q.shape[0] != raw.numel() or q.shape[1] == 0
            or q.dtype != torch.float64 or not finite_tree((raw, projected, q, gradient))):
        raise ValueError("saved action/basis shape/dtype/finiteness mismatch")
    projected_raw64 = q @ (q.T @ raw.double())
    trunc = projected_raw64.to(raw.dtype)
    actions = {"raw": raw, "trunc": trunc, "projected": projected, "zero": torch.zeros_like(raw)}
    if not finite_tree(actions):
        raise ValueError("nonfinite saved-action construction")
    norms = {key: float(value.double().norm()) for key, value in actions.items()}
    gradient_norm = float(gradient.double().norm())
    retained_gradient_norm = float((q @ (q.T @ gradient.double())).norm())
    if min(gradient_norm, retained_gradient_norm, *(norms[key] for key in ("raw", "trunc", "projected"))) <= 0:
        raise ValueError("degenerate saved positive-norm action")
    rho = retained_gradient_norm / gradient_norm
    tolerance = 32 * torch.finfo(raw.dtype).eps
    orthogonality = float((q.T @ q - torch.eye(q.shape[1], dtype=q.dtype, device=q.device)).abs().max())
    residuals = {
        "raw_projected_norm_relative_mismatch": abs(norms["raw"] - norms["projected"]) / norms["raw"],
        "trunc_rho_projected_relative_error": float((trunc.double() - rho * projected.double()).norm()) / norms["raw"],
        "trunc_projection_relative_error": float((trunc.double() - projected_raw64).norm()) / norms["raw"],
        "trunc_projected_collinearity_error": float((trunc.double() / norms["trunc"] - projected.double() / norms["projected"]).norm()),
    }
    if orthogonality > 1e-10 or rho > 1 + tolerance or any(value > tolerance for value in residuals.values()):
        raise ValueError("saved-action float32-scaled algebra check failed: " + json.dumps(
            {"rho": rho, "Q_orthogonality_max_abs": orthogonality, "tolerance": tolerance, **residuals}))
    return actions, {"norms": norms, "gradient_norm": gradient_norm, "rho": rho,
                     "Q_orthogonality_max_abs": orthogonality, "Q_orthogonality_tolerance": 1e-10,
                     "relative_tolerance": tolerance, **residuals}


def parameter_state(model, optimizer):
    return {"model_state": clone_cpu(dict(model.state_dict())),
            "optimizer_state": clone_cpu(optimizer.state_dict()),
            "parameter_identity": parameter_identity(model)}


def adam_checks(before, after, action):
    """Saved-state scalar corroboration; not an additional optimizer execution.

    Max-absolute residual is scaled by max(|expected|, |old|, dtype.tiny).
    The fixed bound is 32 float32 eps, with all unrounded residuals recorded.
    """
    old_opt, new_opt = before["optimizer_state"], after["optimizer_state"]
    if len(old_opt["param_groups"]) != 1 or new_opt["param_groups"] != old_opt["param_groups"]:
        raise ValueError("Adam groups/execution flags differ from the saved single-group optimizer")
    group = old_opt["param_groups"][0]
    if group.get("amsgrad") or group.get("maximize"):
        raise ValueError("unsupported non-original AdamW mode")
    tolerance = 32 * torch.finfo(torch.float32).eps
    maxima = {name: 0. for name in ("m_relative_residual", "v_relative_residual", "parameter_relative_residual")}
    offset = 0
    for pid, identity in zip(group["params"], before["parameter_identity"], strict=True):
        name = identity["name"]
        weight = before["model_state"][name].double()
        count = weight.numel()
        gradient = action[offset:offset + count].detach().cpu().double().reshape_as(weight)
        offset += count
        old, new = old_opt["state"][pid], new_opt["state"][pid]
        step = float(new["step"])
        if step != float(old["step"]) + 1:
            raise ValueError("new action did not take exactly one Adam step")
        beta1, beta2 = group["betas"]
        m = beta1 * old["exp_avg"].double() + (1 - beta1) * gradient
        v = beta2 * old["exp_avg_sq"].double() + (1 - beta2) * gradient.square()
        expected_weight = (1 - group["lr"] * group["weight_decay"]) * weight - group["lr"] * (
            new["exp_avg"].double() / (1 - beta1**step)) / (
            (new["exp_avg_sq"].double() / (1 - beta2**step)).sqrt() + group["eps"])
        for key, expected, observed, previous in (
                ("m_relative_residual", m, new["exp_avg"].double(), old["exp_avg"].double()),
                ("v_relative_residual", v, new["exp_avg_sq"].double(), old["exp_avg_sq"].double()),
                ("parameter_relative_residual", expected_weight, after["model_state"][name].double(), weight)):
            scale = max(float(expected.abs().max()), float(previous.abs().max()), torch.finfo(torch.float32).tiny)
            maxima[key] = max(maxima[key], float((observed - expected).abs().max()) / scale)
    if any(value > tolerance for value in maxima.values()):
        raise ValueError("saved AdamW recurrence exceeds float32-scaled tolerance: " + json.dumps(
            {"tolerance": tolerance, **maxima}))
    return {"relative_tolerance": tolerance, **maxima}


@torch.no_grad()
def apply_once(before, action, name, device, model_factory=None):
    """Only the three new actions: fresh model/Adam, exact saved state, one step."""
    if name not in NEW_ACTIONS:
        raise ValueError("raw/before replay is forbidden")
    model = (GrokkingTransformer(**MODEL) if model_factory is None else model_factory()).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), **ADAMW)
    model.load_state_dict(clone_cpu(before["model_state"]), strict=True)
    optimizer.load_state_dict(clone_cpu(before["optimizer_state"]))
    expected = {key: before[key] for key in ("model_state", "optimizer_state", "parameter_identity")}
    expected_hash = tree_hash(expected)
    restored = parameter_state(model, optimizer)
    if tree_hash(restored) != expected_hash:
        raise ValueError("model/Adam/order restoration differs from saved before state")
    torch.set_rng_state(before["torch_cpu_rng_state"].clone())
    if torch.device(device).type == "cuda":
        if len(before["torch_cuda_rng_states"]) != torch.cuda.device_count():
            raise ValueError("saved CUDA RNG/device count mismatch")
        torch.cuda.set_rng_state_all([value.clone() for value in before["torch_cuda_rng_states"]])
    parameters = list(model.parameters())
    if action.ndim != 1 or action.numel() != sum(parameter.numel() for parameter in parameters):
        raise ValueError("action does not match complete parameter order")
    if action.dtype != torch.float32 or not finite_tree(action):
        raise ValueError("action must be finite original float32")
    offset = 0
    for parameter in parameters:
        if parameter.dtype != action.dtype or parameter.device != action.device:
            raise ValueError("action dtype/device differs from saved parameters")
        parameter.grad = action[offset:offset + parameter.numel()].reshape_as(parameter).clone()
        offset += parameter.numel()
    optimizer.step()  # explicit zero gradients still carry ordinary moments and decay
    after = parameter_state(model, optimizer)
    if not finite_tree(after):
        raise ValueError("nonfinite one-step model/Adam state")
    arithmetic = adam_checks(before, after, action)
    if tree_hash({key: before[key] for key in expected}) != expected_hash:
        raise ValueError("saved before state mutated")
    return model, after, {"before_state_sha256": tree_hash(restored), "after_state_sha256": tree_hash(after),
                          "adam_arithmetic": arithmetic}


class Output:
    def __init__(self, path):
        path = Path(path).resolve()
        if (path.name != "diagnostic-001" or path.parent.parent != STORAGE.resolve()
                or re.fullmatch(r"spectral-grokking-function-response-20260909\.[A-Za-z0-9]+", path.parent.name) is None
                or path.exists() or not path.parent.is_dir()):
            raise ValueError("need a new diagnostic-001 child of the exclusive large-volume parent")
        self.path, self.bytes = path, 0
        self.reserve(MAX_BYTES)
        path.mkdir(mode=0o700)

    def reserve(self, estimate):
        if self.bytes + estimate > MAX_BYTES or shutil.disk_usage(self.path.parent).free < estimate + FREE_RESERVE:
            raise RuntimeError("prospective two-GiB output/free-space budget exhausted")

    def write(self, name, value, kind):
        path = self.path / name
        if Path(name).name != name or path.exists() or path.is_symlink():
            raise ValueError("output path must be new and directly contained")
        if kind == "json":
            raw = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
            self.reserve(len(raw)); atomic_exclusive(path, lambda handle: handle.write(raw))
        elif kind == "npz":
            self.reserve(sum(array.nbytes for array in value.values()) + 1024**2)
            atomic_exclusive(path, lambda handle: np.savez_compressed(handle, **value))
        elif kind == "tensor":
            self.reserve(tensor_bytes(value) + 1024**2)
            save_checkpoint(path, value)
        else:
            raise ValueError("unknown output kind")
        self.bytes += path.stat().st_size
        self.reserve(0)
        return {"path": str(path), "sha256": file_hash(path), "size_bytes": path.stat().st_size}


def check_resources(started):
    if time.monotonic() - started > SECONDS:
        raise TimeoutError("eight-minute cooperative diagnostic deadline")
    if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 > MEMORY_BYTES:
        raise MemoryError("16-GiB diagnostic bound exceeded")
    swap = [line for line in Path("/proc/self/status").read_text().splitlines() if line.startswith("VmSwap:")]
    if len(swap) != 1 or int(swap[0].split()[1]) != 0:
        raise MemoryError("diagnostic has nonzero/unreadable swap")


def measure_seed(item, output, pins, started):
    seed, device = item["seed"], torch.device("cuda")
    first = load_checkpoint(Path(item["first"]["path"]), item["first"]["sha256"])
    raw_checkpoint = load_checkpoint(Path(item["checkpoint"]["path"]), item["checkpoint"]["sha256"])
    before, raw_after = first["full_before"], first["full_after"]
    completion = item["completion"]
    if ((first.get("seed"), first.get("policy"), first.get("step")) != (seed, POLICY, 1501)
            or first.get("schema") != "grokking_raw_direction_acquisition_v1"
            or before.get("schema") != CHECKPOINT_SCHEMA or raw_after.get("schema") != CHECKPOINT_SCHEMA
            or before.get("step") != 1500 or raw_after.get("step") != 1501
            or before.get("config", {}).get("model") != MODEL
            or (before.get("config", {}).get("seed"), before.get("config", {}).get("arm")) != (seed, "legacy")
            or first.get("source_sha256") != completion["source_sha256"]
            or first.get("parent_checkpoint") != completion["parent_checkpoint"]
            or scientific_hash(before) != completion["fork_scientific_state_sha256"]
            or scientific_hash(before) != first.get("restored_scientific_state_sha256")
            or scientific_hash(raw_after) != scientific_hash(raw_checkpoint["native_compatible_state"])
            or (raw_checkpoint.get("seed"), raw_checkpoint.get("policy"), raw_checkpoint.get("step")) != (seed, POLICY, 1501)
            or raw_checkpoint.get("source_sha256") != completion["source_sha256"]
            or raw_checkpoint.get("parent_checkpoint") != completion["parent_checkpoint"]
            or _identity(raw_checkpoint["first_step_tensors"]) != item["first"]):
        raise ValueError("saved common before/raw-after complete-state identity mismatch")
    del raw_checkpoint
    logits, structural = archived_logits(Path(item["before_raw"]["path"]), Path(item["raw_after"]["path"]), item["recipe"])
    actions, algebra = actions_from_saved(first, device)
    states = {name: {key: source[key] for key in ("model_state", "optimizer_state", "parameter_identity")}
              for name, source in (("before", before), ("raw", raw_after))}
    identities = {}
    pairs = torch.from_numpy(structural["pairs"])
    for name in NEW_ACTIONS:
        check_resources(started)
        model, states[name], identities[name] = apply_once(before, actions[name], name, device)
        rng = (torch.get_rng_state().clone(), [value.clone() for value in torch.cuda.get_rng_state_all()])
        activations = extract_activations(model, pairs, batch_size=1024)
        logits[name] = activations["logits"].numpy()
        if (not torch.equal(rng[0], torch.get_rng_state())
                or any(not torch.equal(a, b) for a, b in zip(rng[1], torch.cuda.get_rng_state_all(), strict=True))
                or tree_hash(states[name]["model_state"]) != tree_hash(dict(model.state_dict()))):
            raise ValueError("new-state inference changed model/RNG")
        del activations, model
    state_receipt = output.write(f"seed{seed}-states.pt", {
        "schema": SCHEMA, "seed": seed, "states": states, "actions": clone_cpu(actions),
        "identities": identities, "action_algebra": algebra, "input_first_step": item["first"], "source_sha256": pins}, "tensor")
    logit_receipt = output.write(f"seed{seed}-logits.npz", {
        **logits, "pairs": structural["pairs"], "sums": structural["sums"],
        "train_indices": structural["train_ids"], "test_indices": structural["test_ids"]}, "npz")
    metrics = analyze_seed(logits, structural["sums"], structural["train_ids"], structural["test_ids"])
    result = {"schema": SCHEMA, "seed": seed, "metrics": metrics, "identities": identities,
              "action_algebra": algebra,
              "logits": logit_receipt, "states": state_receipt,
              "archived_before_logits": item["before_raw"], "archived_raw_logits": item["raw_after"],
              "new_one_step_actions": list(NEW_ACTIONS), "new_full_grid_extractions": 3,
              "raw_update_replayed": False, "source_sha256": pins}
    receipt = output.write(f"seed{seed}-result.json", result, "json")
    return result, receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    started = time.monotonic()
    check_resources(started)
    inputs = admit_inputs()
    pins = source_pins()
    output = Output(args.output_dir)
    manifest = output.write("manifest.json", {
        "schema": SCHEMA, "seeds": list(SEEDS), "actions": list(NEW_ACTIONS),
        "input_receipts": inputs["receipts"], "source_sha256": pins,
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "environment": inputs["environment"], "acquisition_environment": inputs["acquisition_environment"],
        "bounds": {"memory_bytes": MEMORY_BYTES, "swap_bytes": 0, "cpu_threads": 1,
                   "cooperative_seconds": SECONDS, "output_bytes": MAX_BYTES, "free_reserve_bytes": FREE_RESERVE},
        "limitations": "Archived common-state before/raw logits have CUDA-sensitivity limits; no raw replay or new probes."}, "json")
    results, receipts = [], []
    try:
        for item in inputs["items"]:
            check_resources(started)
            result, receipt = measure_seed(item, output, pins, started)
            results.append(result); receipts.append(receipt)
            print(json.dumps({"completed_seed": item["seed"], "new_actions": 3, "output_bytes": output.bytes}), flush=True)
            gc.collect()
        if [result["seed"] for result in results] != list(SEEDS):
            raise ValueError("diagnostic did not complete all five fixed seeds")
        summary = output.write("summary.json", {"schema": SCHEMA, "seeds": list(SEEDS),
            "seed_results": results, "paired": paired_summary([result["metrics"] for result in results]),
            "source_sha256": pins, "manifest": manifest}, "json")
        for receipt in inputs["receipts"]:
            check_resources(started)
            _verify_receipt(receipt, Path(receipt["path"]).parent)
        if source_pins() != pins:
            raise ValueError("frozen diagnostic source changed")
        check_resources(started)
        output.write("complete.json", {"schema": SCHEMA, "status": "complete", "seeds": list(SEEDS),
            "new_actions": 15, "new_full_grid_extractions": 15, "manifest": manifest, "summary": summary,
            "accepted_results": receipts, "source_sha256": pins, "elapsed_seconds": time.monotonic() - started,
            "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024}, "json")
    except BaseException as error:
        failure = {"schema": SCHEMA, "status": "failed_preserved", "error_type": type(error).__name__,
                   "error": str(error), "accepted_results": receipts, "manifest": manifest}
        print(json.dumps(failure), flush=True)
        try:
            output.write("failure.json", failure, "json")
        except Exception:
            pass
        raise


if __name__ == "__main__":
    main()
