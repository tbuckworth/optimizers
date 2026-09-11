#!/usr/bin/env python3
"""Independent iteration006 fresh-bundle re-execution.

Import and synthetic CPU tests are inert. Dataset access and execution require
the explicit, separately approved ``--audit-reexecution --audit-go`` pair.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import traceback

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]


def _load(filename, name):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


producer = _load("delivery_order_harness.py", "iteration006_frozen_producer_for_audit")
storage = producer.storage
torch, np = producer.torch, producer.np

BUNDLE, STEPS = 60006, 2000
NOISES = (0.0, 0.9)
ARMS = ("current32", "lagged32", "lagged32_current_norm")
CHECKPOINTS = producer.CHECKPOINTS
ARTIFACT_CAP = 256 * 1024 ** 2
AUDIT_SOURCE_NAMES = ("audit_reexecution.py", "audit-reexecution-plan.md")
METADATA_DIRECTORY = HERE / "audit-reexecution"
OUTER_LOG = HERE / "audit-reexecution-run.log"
PRIMARY_EXECUTION = HERE / "results" / "execution.json"
IMPORTED_AUDIT_SCRIPT_SHA256 = storage.sha256(Path(__file__))


def require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def audit_rng(stream):
    require(type(stream) is int and 0 <= stream <= 6, "Invalid audit RNG stream")
    return np.random.default_rng(np.random.SeedSequence([20260906, BUNDLE, stream]))


def make_audit_plan(steps=STEPS):
    """Construct the audit plan without importing or calling producer.make_plan."""
    require(type(steps) is int and steps >= producer.WARMUP, "Invalid audit plan length")
    permutation = audit_rng(0).permutation(60000)
    return {
        "seed": BUNDLE,
        "initialization_seed": int(audit_rng(3).integers(0, 2**32, dtype=np.uint32)),
        "train_indices": permutation[:5000],
        "validation_indices": permutation[5000:10000],
        "auxiliary_indices": permutation[10000:15000],
        "replacement_uniforms": audit_rng(1).random(5000),
        "replacement_digits": audit_rng(2).integers(0, 10, size=5000),
        "training_batches": audit_rng(4).integers(0, 5000, size=(steps, 64)),
        "primary_probe_batches": audit_rng(5).integers(0, 5000, size=(steps, 256)),
        "auxiliary_probe_batches": audit_rng(6).integers(0, 5000, size=(steps, 256)),
    }


def validate_audit_plan(plan, steps=STEPS):
    require(set(plan) == set(make_audit_plan(steps)), "Audit plan keys changed")
    require(type(plan["seed"]) is int and plan["seed"] == BUNDLE, "Wrong audit bundle")
    require(type(plan["initialization_seed"]) is int, "Initialization seed must be an integer")
    producer.validate_plan(plan, steps)
    expected = make_audit_plan(steps)
    for key in expected:
        if isinstance(expected[key], np.ndarray):
            require(isinstance(plan[key], np.ndarray) and np.array_equal(plan[key], expected[key]),
                    f"Audit plan does not match frozen SeedSequence streams: {key}")
        else:
            require(plan[key] == expected[key], f"Audit plan scalar mismatch: {key}")


def audit_cell(noise, arm):
    require(noise in NOISES and arm in ARMS, "Unexpected audit cell")
    return producer.cell(BUNDLE, noise, arm)


def expected_audit_cells():
    return [audit_cell(noise, arm) for noise in NOISES for arm in ARMS]


def audit_source_paths():
    return [HERE / name for name in AUDIT_SOURCE_NAMES]


def audit_source_hashes():
    return {str(path.relative_to(REPO)): storage.sha256(path) for path in audit_source_paths()}


def _committed_hashes(paths):
    result = {}
    for path in paths:
        relative = str(path.resolve().relative_to(REPO))
        current = storage.sha256(path)
        committed = subprocess.check_output(["git", "show", f"HEAD:{relative}"], cwd=REPO)
        require(hashlib.sha256(committed).hexdigest() == current,
                f"Audit scientific source is uncommitted: {relative}")
        result[relative] = current
    return result


def committed_source_gate():
    primary = producer.committed_source_gate()
    require(len(primary) == 15 and set(primary) == {str(p.relative_to(REPO)) for p in producer.source_paths()},
            "Primary scientific-source map changed")
    audit = _committed_hashes(audit_source_paths())
    require(len(audit) == 2 and set(primary).isdisjoint(audit), "Audit sources must be separately bound")
    script_key = str(Path(__file__).resolve().relative_to(REPO))
    require(audit[script_key] == IMPORTED_AUDIT_SCRIPT_SHA256,
            "Audit script changed after import")
    return primary, audit


def source_stability_gate(primary, audit):
    require(producer.source_hashes() == primary, "Primary scientific sources changed during audit")
    require(audit_source_hashes() == audit, "Audit scientific sources changed during audit")


class AuditGuard(storage.Guard):
    def __init__(self, device="cuda"):
        super().__init__(False, device)
        self.limit = 300


class AuditStore(storage.Store):
    def capacity(self, additional=0):
        require(type(additional) is int and additional >= 0, "Invalid pending artifact size")
        if self.bytes_used() + additional > ARTIFACT_CAP:
            raise RuntimeError("256-MiB audit artifact budget exceeded")
        # Retain the frozen store's one-GiB free-space reserve.
        import shutil
        if shutil.disk_usage(self.root).free < storage.GIB + additional:
            raise RuntimeError("Bulk free-space headroom failed")


def primary_completion_gate(primary_hashes, current_environment, training_bindings):
    primary = json.loads(PRIMARY_EXECUTION.read_text())
    require(primary["mode"] == "full" and primary["status"] == "complete"
            and primary["completed_runs"] == 36 and primary["test_evaluations"] == 144
            and primary["all_gates_passed"] and primary["official_test_opened"],
            "Primary GPU process is not terminal and complete")
    require(primary["source_sha256"] == primary_hashes, "Primary source map differs from audit launch")
    require(primary["environment"] == current_environment, "Numerical environment differs from primary")
    require(primary["training_data_artifacts"] == training_bindings, "Training data differs from primary")
    require(len(primary["test_data_artifacts"]) == 2, "Primary test-data binding is incomplete")
    # Do not open or hash official-test files here. Their expected bindings may
    # be carried forward, but bytes are checked only after all six audit cells.
    for item in primary["training_data_artifacts"]:
        storage.verify_artifact(item)
    return storage.artifact(PRIMARY_EXECUTION), primary["test_data_artifacts"]


def complete_training_gate(results, checkpoint_bindings, plan_sha256):
    expected = expected_audit_cells()
    producer.complete_training_gate(results, checkpoint_bindings, expected)
    require(len(results) == 6, "Audit re-execution requires exactly six cells")
    rates = {}
    for result in results:
        require(result.get("mode") == "audit_reexecution" and result.get("audit_bundle") == BUNDLE,
                "Run is not identified as an audit re-execution")
        require(result["instrumented"] is True and result["measurement_state_checks"] == 22,
                "Audit cell lacks full instrumentation/state checks")
        require(result["plan_sha256"] == plan_sha256 and "test" not in result,
                "Plan binding mismatch or early test access")
        require(len(result["trajectory_parameter_sha256"]) == STEPS
                and len(result["estimation_rank_by_step"]) == STEPS
                and len(result["repair_count_by_step"]) == STEPS
                and len(result["step_elapsed_seconds"]) == STEPS,
                "Incomplete audit trajectory history")
        require(len(result["warmup_trajectory_hashes"]) == producer.WARMUP
                and len(result["warmup_observer_hashes"]) == producer.WARMUP,
                "Incomplete audit warmup history")
        noise = result["replacement_probability"]
        pair = (result["realized_replacement_fraction"], result["realized_incorrect_fraction"])
        require(all(math.isfinite(value) and 0 <= value <= 1 for value in pair),
                "Invalid realized corruption rate")
        require(noise not in rates or rates[noise] == pair, "Within-condition corruption differs across arms")
        rates[noise] = pair
    require(rates[0.0] == (0.0, 0.0) and set(rates) == set(NOISES),
            "Audit corruption conditions are incomplete or invalid")


def primary_contrasts(results):
    by_cell = {(row["replacement_probability"], row["arm"]): row for row in results}
    require(len(by_cell) == 6, "Cannot derive audit contrasts from incomplete cells")
    contrasts = []
    for noise in NOISES:
        reference = by_cell[(noise, "current32")]
        reference_metric = reference["test"]["max_val_accuracy"]
        producer.valid_evaluation(reference_metric, 10000)
        for treatment in ("lagged32", "lagged32_current_norm"):
            treatment_row = by_cell[(noise, treatment)]
            treatment_metric = treatment_row["test"]["max_val_accuracy"]
            producer.valid_evaluation(treatment_metric, 10000)
            contrasts.append({
                "audit_bundle": BUNDLE,
                "replacement_probability": noise,
                "comparison": f"{treatment}-current32",
                "checkpoint": "max_val_accuracy",
                "current32_selected_step": reference["checkpoint_steps"]["max_val_accuracy"],
                "treatment_selected_step": treatment_row["checkpoint_steps"]["max_val_accuracy"],
                "current32_test_accuracy": reference_metric["accuracy"],
                "treatment_test_accuracy": treatment_metric["accuracy"],
                "accuracy_difference": treatment_metric["accuracy"] - reference_metric["accuracy"],
            })
    require(len(contrasts) == 4, "Expected exactly four audit primary contrasts")
    storage.json_bytes(contrasts)
    return contrasts


def execute(store, metadata, manifest, guard, context, primary_test_bindings):
    plan = make_audit_plan()
    validate_audit_plan(plan)
    plan_binding = store.plan("plan-bundle60006.npz", plan)
    storage.verify_artifact(plan_binding)
    manifest["plans"] = [plan_binding]
    metadata.write("execution.json", manifest)

    x, y = producer.h.load_training()
    results = []
    for noise in NOISES:
        data = producer.h.dataset_for_plan(x, y, plan, noise, "cuda")
        producer.validate_data(data, noise, plan)
        warmup_reference = None
        for arm in ARMS:
            source_stability_gate(manifest["source_sha256"], manifest["audit_source_sha256"])
            identity = audit_cell(noise, arm)
            result, checkpoints, _, warm = producer.train_cell(
                plan, data, noise, arm, STEPS, True, False, guard, context)
            if warmup_reference is None:
                warmup_reference = (result, warm)
            else:
                producer.verify_warmup(*warmup_reference, result, warm, observer=True)
            result.update(mode="audit_reexecution", audit_bundle=BUNDLE,
                          warmup_checks_passed=True, plan_sha256=plan_binding["sha256"])
            binding = store.checkpoints(identity["run_key"] + "-checkpoints.pt", checkpoints, **identity)
            result["checkpoint_path"] = binding["path"]
            manifest["checkpoints"].append(binding)
            manifest["training_runs"].append(
                store.json(identity["run_key"] + ".training.json", result, **identity))
            manifest["completed_cells"].append(identity)
            results.append(result)
            manifest["completed_runs"] = len(results)
            manifest["resources"] = guard.check()
            metadata.write("execution.json", manifest)

    complete_training_gate(results, manifest["checkpoints"], plan_binding["sha256"])
    source_stability_gate(manifest["source_sha256"], manifest["audit_source_sha256"])
    manifest["all_training_completed_utc"] = producer.utc()
    context.clear()
    context.update(phase="audit_test_loading_after_all_six_training_cells")

    def loader():
        metadata.write("execution.json", manifest)
        return producer.h.load_test_after_training()

    tx, ty = producer.load_test_with_gate(
        results, manifest["checkpoints"], manifest, loader, expected_audit_cells())
    require(len(tx) == len(ty) == 10000, "Wrong official test cardinality")
    manifest["test_data_artifacts"] = [storage.artifact(producer.h.DATA / name) for name in
                                       ("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte")]
    require(manifest["test_data_artifacts"] == primary_test_bindings, "Test data differs from primary")
    tx, ty = tx.cuda(), ty.cuda()
    model = producer.h.make_model(0, "cuda")
    for result in results:
        identity = audit_cell(result["replacement_probability"], result["arm"])
        bundle = torch.load(result["checkpoint_path"], map_location="cpu", weights_only=True)
        test = {}
        for name in CHECKPOINTS:
            context.update(identity, phase="audit_test_evaluation", checkpoint=name)
            model.load_state_dict(bundle[name])
            outcome = producer.h.evaluate(model, tx, ty)
            producer.valid_evaluation(outcome, 10000)
            test[name] = outcome
            manifest["test_evaluations"] += 1
            guard.check()
        result["test"] = test
        manifest["runs"].append(store.json(identity["run_key"] + ".json", result, **identity))
        metadata.write("execution.json", manifest)

    require(manifest["test_evaluations"] == 24 and len(manifest["runs"]) == 6,
            "Incomplete audit checkpoint evaluation")
    contrasts = primary_contrasts(results)
    manifest["fresh_bundle_primary_contrasts"] = store.json(
        "fresh-bundle-primary-contrasts.json", contrasts, audit_bundle=BUNDLE)
    source_stability_gate(manifest["source_sha256"], manifest["audit_source_sha256"])
    manifest.update(status="complete", all_gates_passed=True,
                    warmup_checks_passed=True, official_test_opened=True)


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, value):
        for stream in self.streams:
            stream.write(value)
        return len(value)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def launch():
    guard = AuditGuard()
    metadata = storage.Metadata(METADATA_DIRECTORY)
    manifest = {
        "schema_version": 1, "mode": "audit_reexecution", "audit_bundle": BUNDLE,
        "status": "training", "started_utc": producer.utc(), "all_gates_passed": False,
        "official_test_opened": False, "completed_cells": [], "completed_runs": 0,
        "test_evaluations": 0, "plans": [], "checkpoints": [], "training_runs": [], "runs": [],
    }
    context, store, log_path, log_binding = {"phase": "launch_gates"}, None, None, None
    metadata.write("execution.json", manifest)
    try:
        producer.h.configure()
        require(torch.cuda.is_available(), "Local GPU required; no CPU-training fallback")
        manifest["occupancy_before"] = producer.occupancy_gate()
        manifest["source_sha256"], manifest["audit_source_sha256"] = committed_source_gate()
        manifest["repository_revision"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
        manifest["environment"] = producer.environment()
        manifest["training_data_artifacts"] = [storage.artifact(producer.h.DATA / name)
                                                for name in producer.h.TRAINING_FILES]
        primary_binding, primary_test_bindings = primary_completion_gate(
            manifest["source_sha256"], manifest["environment"], manifest["training_data_artifacts"])
        manifest["primary_execution"] = primary_binding
        store = AuditStore("audit-reexecution")
        manifest.update(bulk_root=str(store.root), bulk_mount=store.mount)
        log_path = store.path("run.log")
        with log_path.open("x", buffering=1) as log_stream, \
                contextlib.redirect_stdout(_Tee(sys.stdout, log_stream)), \
                contextlib.redirect_stderr(_Tee(sys.stderr, log_stream)):
            print(json.dumps({"event": "audit_reexecution_started", "bundle": BUNDLE}), flush=True)
            metadata.write("execution.json", manifest)
            torch.cuda.reset_peak_memory_stats()
            execute(store, metadata, manifest, guard, context, primary_test_bindings)
            print(json.dumps({"event": "audit_reexecution_complete", "test_evaluations": 24}), flush=True)
            log_stream.flush()
            os.fsync(log_stream.fileno())
        log_binding = store.finish(log_path)
        # Bind the closed log before any later guard/capacity operation can fail.
        manifest["run_log"] = log_binding
        manifest.update(resources=guard.check(), completed_utc=producer.utc())
        manifest["artifact_total_bytes"] = store.bytes_used()
        store.capacity()
        metadata.write("execution.json", manifest)
    except BaseException as error:
        if store is not None and log_path is not None and log_path.exists() and log_binding is None:
            try:
                log_binding = store.finish(log_path)
                manifest["run_log"] = log_binding
            except Exception as log_error:
                context["log_preservation_error"] = repr(log_error)
        manifest["resources"] = guard.observe()
        storage.preserve_failure(metadata, manifest, context, error, store)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-reexecution", action="store_true")
    parser.add_argument("--audit-go", action="store_true")
    args = parser.parse_args()
    if not (args.audit_reexecution and args.audit_go):
        parser.error("Independent execution requires separate parent approval and both --audit-reexecution --audit-go")
    # This exclusive outer log covers launch gates and an explicit traceback;
    # the separately bound bulk run.log covers the attempt after Store exists.
    with OUTER_LOG.open("x", buffering=1) as outer_stream, \
            contextlib.redirect_stdout(_Tee(sys.stdout, outer_stream)), \
            contextlib.redirect_stderr(_Tee(sys.stderr, outer_stream)):
        try:
            launch()
        except BaseException:
            traceback.print_exc()
            raise
        finally:
            outer_stream.flush()
            os.fsync(outer_stream.fileno())


if __name__ == "__main__":
    main()
