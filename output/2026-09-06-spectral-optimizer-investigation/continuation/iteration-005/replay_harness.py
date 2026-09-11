#!/usr/bin/env python3
"""Separately gated common-AdamW replay and finite-history covariance reference."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / "iteration-004" / "norm_control_harness.py"
spec = importlib.util.spec_from_file_location("iteration004_immutable_helpers", PREVIOUS)
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)
h, torch, np = old.h, old.torch, old.np
REPO = h.REPO
import artifact_store as disk
import reference_math as mathref

SEEDS, STEPS, SNAPSHOTS = (3, 4, 5), 2000, (200, 500, 1000, 2000)
WIDTHS = (32, 128)
HISTORICAL_COMMIT = "c52d2ceb56fab7fcf88497738c6aae28e7d26e4c"
PROTOCOL_SHA = "a5d5065947a1f65657c96abac4e5e2ba21e188efb9a7f00f8612fff43dceed1b"
HISTORY = HERE.parent / "iteration-004"
DOCUMENTS = ("protocol.md", "design-intent.md", "best-practices-check.md", "reference-identities.md",
             "check_reference_identities.py", "reference-identity-checks.json", "analysis-intent.md",
             "result-schema.md", "implementation-decision.md", "test_harness.py",
             "analysis-plan.md", "summarize_results.py", "test_summary.py")


def sources():
    return [Path(__file__), HERE / "reference_math.py", HERE / "artifact_store.py",
            *(HERE / name for name in DOCUMENTS), PREVIOUS, Path(h.__file__), REPO / "spectral_filter.py"]


def source_hashes(committed=False):
    for loaded, expected in ((mathref.__file__, HERE / "reference_math.py"),
                             (disk.__file__, HERE / "artifact_store.py"),
                             (old.__file__, PREVIOUS),
                             (h.__file__, HERE.parent / "iteration-003" / "neural_harness.py"),
                             (h.SpectralGradientFilter.__init__.__code__.co_filename, REPO / "spectral_filter.py")):
        mathref.require(Path(loaded).resolve() == expected.resolve(), f"Unexpected imported source: {loaded}")
    result = {}
    for path in sources():
        relative = str(path.relative_to(REPO))
        result[relative] = disk.sha256(path)
        if committed:
            raw = subprocess.check_output(["git", "show", f"HEAD:{relative}"], cwd=REPO)
            mathref.require(hashlib.sha256(raw).hexdigest() == result[relative], f"Uncommitted source: {relative}")
    mathref.require(disk.sha256(HERE / "protocol.md") == PROTOCOL_SHA, "Protocol differs from implementation approval")
    return result


def verify_artifact(item):
    path = Path(item["path"])
    mathref.require(path.stat().st_size == item["size_bytes"] and disk.sha256(path) == item["sha256"],
                    f"Artifact changed: {path}")


def historical_bindings():
    """Read/hash immutable provenance; no evaluation and no plan regeneration."""
    archive_path = HISTORY / "raw-results" / "manifest.json"
    committed = subprocess.check_output(["git", "show", f"{HISTORICAL_COMMIT}:{archive_path.relative_to(REPO)}"], cwd=REPO)
    mathref.require(hashlib.sha256(committed).hexdigest() == disk.sha256(archive_path), "Historical archive manifest changed")
    archive = json.loads(committed)
    selected = [item for item in archive["files"] if item["original_path"] == "results/execution.json"
                or item["original_path"] in {f"results/seed{s}-adamw.json" for s in SEEDS}]
    mathref.require(len(selected) == 4, "Missing historical AdamW evidence")
    artifacts = [disk.artifact(archive_path)]
    for item in selected:
        path = HISTORY / item["original_path"]
        mathref.require(disk.sha256(path) == item["original_sha256"], "Historical raw JSON changed")
        artifacts.append(disk.artifact(path))
    execution = json.loads((HISTORY / "results" / "execution.json").read_text())
    mathref.require(execution["status"] == "complete" and execution["completed_runs"] == 12, "Historical run incomplete")
    for relative, expected in execution["source_sha256"].items():
        mathref.require(disk.sha256(REPO / relative) == expected, f"Historical source changed: {relative}")
    bound = [*execution["training_data_artifacts"], *execution["plans"],
             *(item for item in execution["checkpoints"] if "-adamw-checkpoints.pt" in item["path"])]
    for item in bound:
        verify_artifact(item)
    artifacts.extend(bound)
    for name, current in (("python", sys.version), ("numpy", np.__version__), ("torch", torch.__version__),
                          ("cuda", torch.version.cuda)):
        mathref.require(execution[name] == current, f"Historical numerical environment changed: {name}")
    return {"audited_results_commit": HISTORICAL_COMMIT, "execution_commit": execution["repository_revision"],
            "artifacts": artifacts}, execution


def check_compute_processes(process_csv, own_pid=None):
    own_pid = os.getpid() if own_pid is None else own_pid
    allowed = {"/usr/libexec/gnome-remote-desktop-daemon", "stremio", "/usr/bin/gnome-shell", "/usr/lib/xorg/Xorg"}
    for row in csv.reader(process_csv.splitlines()):
        if not row:
            continue
        mathref.require(len(row) == 2, "Malformed GPU process identity")
        pid, name = int(row[0].strip()), row[1].strip()
        mathref.require(pid == own_pid or name in allowed, f"Another GPU compute job is present: PID {pid}, {name}")


def resource_start_gate():
    mathref.require(torch.cuda.is_available(), "Local GPU required; no fallback compute")
    mathref.require(torch.cuda.get_device_name(0) == "NVIDIA GeForce RTX 3090", "Unexpected GPU")
    occupancy = h.occupancy()
    text = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"], text=True)
    # get_device_name may have initialized our own context; exclude only its PID.
    check_compute_processes(text)
    free, _ = torch.cuda.mem_get_info()
    mathref.require(free >= 8 * disk.GIB, "Insufficient free GPU memory")
    memory = {line.split(":")[0]: int(line.split()[1]) * 1024
              for line in Path("/proc/meminfo").read_text().splitlines() if ":" in line}
    mathref.require(memory["MemAvailable"] >= 16 * disk.GIB, "Insufficient available RAM")
    return occupancy


def core_snapshot(model, optimizer):
    value = h.snapshot(model, optimizer, None)
    # The immutable helper omits NumPy's legacy global RNG; include it here.
    numpy_state = np.random.get_state()
    value["numpy_rng"] = (numpy_state[0], torch.from_numpy(numpy_state[1].copy()), *numpy_state[2:])
    return value


def complete_snapshot(model, optimizer, observers):
    value = core_snapshot(model, optimizer)
    value["observers"] = {str(width): h.tracker_state(tracker) for width, tracker in observers.items()}
    return value


def checkpoint_gate(model, step, historical, checkpoints):
    checked = []
    if historical is not None:
        for name, selected in historical["checkpoint_steps"].items():
            if selected == step:
                mathref.require(h.equal_tree(old.cpu_tree(model.state_dict()), checkpoints[name]),
                                f"Historical checkpoint mismatch: {name} at {step}")
                checked.append({"name": name, "step": step, "bitwise_equal": True})
    return checked


def check_historical_scalars(raw, theta_before, theta_after, historical_row):
    delta = theta_after - theta_before
    data_delta = h.decay_subtracted_update(delta, theta_before)
    measured = {"raw_squared_norm": h.dot(raw, raw),
                "total_squared_norm": h.dot(delta, delta),
                "total_raw_dot": h.dot(raw, delta),
                "decay_subtracted_squared_norm": h.dot(data_delta, data_delta),
                "decay_subtracted_raw_dot": h.dot(raw, data_delta)}
    expected = {"raw_squared_norm": historical_row["raw_squared_norm"],
                "total_squared_norm": historical_row["total"]["squared_norm"],
                "total_raw_dot": historical_row["total"]["raw_gradient_dot_update"]["value"],
                "decay_subtracted_squared_norm": historical_row["decay_subtracted"]["squared_norm"],
                "decay_subtracted_raw_dot": historical_row["decay_subtracted"]["raw_gradient_dot_update"]["value"]}
    mathref.require(measured == expected, f"Historical per-step scalar mismatch: {measured} != {expected}")


def observe_common(observers, raw, independent_mean, first, step):
    mathref.finite(raw)
    if independent_mean is None:
        independent_mean = raw.detach().clone()
    else:
        independent_mean.mul_(.99).add_(raw.detach(), alpha=1 - .99)
    innovation = raw.detach() - independent_mean
    mathref.finite(independent_mean, innovation)
    accepted = []
    for width, tracker in observers.items():
        existed = tracker.V is not None
        tracker.step_count += 1
        tracker._update_svd(raw)
        mathref.require(torch.equal(tracker.grad_mean, independent_mean), "Observer/independent means differ")
        mathref.require(tracker.step_count == step, "Observer step mismatch")
        mathref.require(not existed or tracker.V is not None, "Observer reset after initialization")
        if not existed and tracker.V is not None:
            mathref.require(first is None, "Observer reinitialized")
            accepted.append(width)
    if accepted:
        mathref.require(len(accepted) == len(observers) and innovation.norm() > 0, "Observer first-acceptance mismatch")
        first = step
    if first is None:
        mathref.require(not bool(torch.count_nonzero(innovation)), "Nonzero innovation before accepted initialization")
    return independent_mean, innovation, first


def make_probes(model, optimizer, observers, data, primary, auxiliary):
    before = complete_snapshot(model, optimizer, observers)
    clean = h.gradient(model, data["x"][primary], data["clean"][primary])
    noisy = h.gradient(model, data["x"][primary], data["noisy"][primary])
    heldout = h.gradient(model, data["ax"][auxiliary], data["ay"][auxiliary])
    values = {"clean": clean, "noisy": noisy, "corruption_residual": noisy - clean, "auxiliary_clean": heldout}
    mathref.require(h.equal_tree(before, complete_snapshot(model, optimizer, observers)), "Probes mutated training state")
    return values


def load_historical_seed(seed, plan):
    result = json.loads((HISTORY / "results" / f"seed{seed}-adamw.json").read_text())
    with np.load(HISTORY / "results" / f"plan-seed{seed}.npz", allow_pickle=False) as saved:
        mathref.require(set(saved.files) == set(plan), "Historical plan keys differ")
        for key, value in plan.items():
            mathref.require(np.array_equal(saved[key], value), f"Historical plan mismatch: {key}")
    checkpoints = torch.load(result["checkpoint_path"], map_location="cpu", weights_only=True)
    return result, checkpoints


def run_trajectory(plan, data, instrumented, pilot, store, guard, context, output, historical=None, checkpoints=None,
                   failure_capture=None):
    steps = len(plan["training_batches"])
    model = h.make_model(plan["initialization_seed"], "cuda")
    optimizer = h.make_optimizer(model)
    observers = {width: h.make_tracker(model, optimizer, width) for width in WIDTHS} if instrumented else {}
    independent_mean, first = None, None
    hashes, raw_hashes, innovation_hashes, warmup, timings, ranks, repairs, anchors, saved_snapshots = [], [], [], [], [], [], [], [], []
    tag = f"seed{plan['seed']}"
    scheduled = (200, 220) if pilot else SNAPSHOTS
    context.update(seed=plan["seed"], step=0, instrumented=instrumented, phase="initialization")
    context["partial_replay"] = {"seed": plan["seed"], "trajectory_parameter_sha256": hashes,
                                 "raw_gradient_sha256": raw_hashes, "innovation_sha256": innovation_hashes,
                                 "warmup_trajectory_hashes": warmup, "step_elapsed_seconds": timings,
                                 "ranks_by_step": ranks, "repair_counts_by_step": repairs,
                                 "checkpoint_comparisons": anchors, "snapshot_artifacts": saved_snapshots}
    if failure_capture is not None and not pilot:
        failure_capture["get"] = lambda: old.cpu_tree({"training": complete_snapshot(model, optimizer, observers),
                                                       "independent_mean": independent_mean})
    anchors.extend(checkpoint_gate(model, 0, historical, checkpoints))
    if not pilot:
        store.stream(f"{tag}-raw.npy", (steps, 50890))
        store.stream(f"{tag}-innovations.npy", (steps, 50890))
    batches = torch.as_tensor(plan["training_batches"], device="cuda")
    primary = torch.as_tensor(plan["primary_probe_batches"], device="cuda")
    auxiliary = torch.as_tensor(plan["auxiliary_probe_batches"], device="cuda")
    warm_hash = None
    for step in range(1, steps + 1):
        guard.check()
        context.update(step=step, phase="training_gradient")
        h.synchronize("cuda")
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        batch = batches[step - 1]
        loss = h.F.cross_entropy(model(data["x"][batch]), data["noisy"][batch])
        mathref.finite(loss.detach())
        loss.backward()
        raw, theta = h.flat_grad(model), h.flat_params(model)
        previous = {width: None if tracker.V is None else tracker.V[:, :32].clone()
                    for width, tracker in observers.items()} if step in scheduled else {}
        core_before = core_snapshot(model, optimizer) if instrumented and step in scheduled else None
        if instrumented:
            context["phase"] = "common_observation"
            independent_mean, innovation, first = observe_common(observers, raw, independent_mean, first, step)
            if core_before is not None:
                mathref.require(h.equal_tree(core_before, core_snapshot(model, optimizer)), "Observation changed training core")
            ranks.append({str(width): 0 if tracker.V is None else tracker.V.shape[1] for width, tracker in observers.items()})
            repairs.append({str(width): tracker.stabilization_count for width, tracker in observers.items()})
            for tracker in observers.values():
                if tracker.V is not None:
                    mathref.finite(tracker.V, tracker.S)
        if not pilot:
            store.append(f"{tag}-raw.npy", step - 1, raw.cpu().numpy())
            store.append(f"{tag}-innovations.npy", step - 1, innovation.cpu().numpy())
            raw_hashes.append(h.tensor_hash(raw))
            innovation_hashes.append(h.tensor_hash(innovation))
        if instrumented and step in scheduled:
            context["phase"] = "pre_adam_snapshot_probes"
            protected = complete_snapshot(model, optimizer, observers)
            probes = make_probes(model, optimizer, observers, data, primary[step - 1], auxiliary[step - 1])
            for tracker in observers.values():
                if tracker.V is not None:
                    mathref.require(tracker.orthogonality_error() <= 5e-3, "Native snapshot basis not orthogonal")
            # Pilot validates measurement arithmetic but does not store its outcomes.
            for tracker in observers.values():
                mathref.probe_metrics(probes, None if tracker.V is None else tracker.V[:, :32])
            if not pilot:
                snapshot = {"seed": plan["seed"], "step": step, "first": first,
                            "state_timing": "pre_adam_after_current_gradient_observation",
                            "parameter_hash": h.tensor_hash(theta), "raw": raw, "innovation": innovation,
                            "mean": independent_mean, "probes": probes,
                            "previous_bases": {str(k): v for k, v in previous.items()},
                            "observers": {str(k): h.tracker_state(v) for k, v in observers.items()}}
                saved_snapshots.append(store.tensor_bundle(f"{tag}-step{step}-state.pt", old.cpu_tree(snapshot),
                                                           expected_bytes=40 * 1024 ** 2, seed=plan["seed"], step=step,
                                                           state_timing=snapshot["state_timing"]))
            mathref.require(h.equal_tree(protected, complete_snapshot(model, optimizer, observers)), "Snapshot/metrics changed training state")
        mathref.require(torch.equal(h.flat_grad(model), raw), "Observer or probe changed delivered gradient")
        context["phase"] = "unchanged_adamw_step"
        optimizer.step()
        after = h.flat_params(model)
        mathref.finite(after)
        hashes.append(h.tensor_hash(after))
        if step <= 100:
            item = {"step": step, "parameters": hashes[-1], "raw_gradient": h.tensor_hash(raw), "applied_gradient": h.tensor_hash(h.flat_grad(model))}
            warmup.append(item)
            if historical is not None:
                mathref.require(item == historical["warmup_trajectory_hashes"][step - 1], "Historical warmup hash mismatch")
        if step == 100:
            warm_hash = old.tree_hash(old.warmup_snapshot(model, optimizer, None)["core"])
            if historical is not None:
                mathref.require(warm_hash == historical["warmup_core_sha256"], "Historical full warmup core differs")
        if historical is not None:
            check_historical_scalars(raw, theta, after, historical["steps_raw"][step - 1])
            anchors.extend(checkpoint_gate(model, step, historical, checkpoints))
        h.synchronize("cuda")
        timings.append(time.perf_counter() - started)
        if step % 50 == 0:
            if not pilot:
                store.flush()
                disk.write_json(output / "progress.json", {**context, "completed_stream_rows": dict(store.durable_rows),
                                                            "bulk_root": str(store.root), "resources": guard.check()})
            print(json.dumps({"seed": plan["seed"], "step": step, "instrumented": instrumented,
                              "phase": "replay", "elapsed_seconds": guard.check()["elapsed_seconds"]}), flush=True)
    if instrumented:
        mathref.require(ranks[159]["128"] == 128, "Wide observer did not reach128 by160")
    if historical is not None:
        mathref.require(len(anchors) == 4, "Not all historical checkpoints checked")
    result = {"schema_version": 1, "seed": plan["seed"], "steps": steps, "instrumented": instrumented,
              "first_accepted_innovation_step": first, "trajectory_parameter_sha256": hashes,
              "warmup_trajectory_hashes": warmup, "warmup_core_sha256": warm_hash,
              "raw_gradient_sha256": raw_hashes, "innovation_sha256": innovation_hashes,
              "ranks_by_step": ranks, "repair_counts_by_step": repairs, "checkpoint_comparisons": anchors,
              "historical_scalar_comparison_steps": steps if historical is not None else 0,
              "all_historical_gates_passed": historical is not None,
              "historical_warmup_hash_steps": 100 if historical is not None else 0,
              "observer_means_bitwise_equal_steps": steps if instrumented else 0,
              "delivered_raw_equal_steps": steps, "no_observer_resets": True,
              "snapshot_count": len(saved_snapshots),
              "probe_state_check_count": len(scheduled) if instrumented else 0,
              "step_elapsed_seconds": timings, "steady129_to220_mean_seconds": float(np.mean(timings[128:220])),
              "repair_step200_seconds": timings[199], "snapshot_artifacts": saved_snapshots,
              "all_state_gates_passed": True}
    if not pilot:
        result["stream_artifacts"] = [item for item in store.stream_artifacts() if Path(item["path"]).name.startswith(tag + "-")]
        result["realized_replacement_fraction"] = data["realized_replacement_fraction"]
        result["realized_incorrect_fraction"] = data["realized_incorrect_fraction"]
    return result, old.cpu_tree(core_snapshot(model, optimizer))


def reference_snapshot(replay, item, store, guard, context):
    verify_artifact(item)
    saved = torch.load(item["path"], map_location="cuda", weights_only=True)
    seed, step = saved["seed"], saved["step"]
    context.clear()
    context.update(seed=seed, step=step, snapshot=step, phase="reference_input")
    context["source_snapshot_artifact"] = item
    innovation_item = next(v for v in replay["stream_artifacts"] if v["path"].endswith("-innovations.npy"))
    verify_artifact(innovation_item)
    require_rows = innovation_item["completed_rows"] == STEPS
    mathref.require(require_rows, "Incomplete innovation stream")
    mapped = np.load(innovation_item["path"], mmap_mode="r", allow_pickle=False)
    innovation = torch.from_numpy(np.array(mapped[:step], copy=True))
    x, weights = mathref.weighted_columns(innovation, saved["first"], device="cuda")
    del innovation
    guard.check()
    context["phase"] = "dual_reference"
    ref, tensors = mathref.reference(x, saved["first"])
    context["partial_reference"] = ref
    mathref.add_reference_probes(ref, tensors, saved["probes"])
    guard.check()
    context["phase"] = "observer_reference_comparison"
    observers = {f"width{width}": mathref.observer_metrics(x, ref, tensors, saved["observers"][str(width)],
                                                         saved["previous_bases"][str(width)], saved["probes"], saved["raw"], width)
                 for width in WIDTHS}
    guard.check()
    context["phase"] = "reference_artifact_write"
    tag = f"seed{seed}-step{step}"
    artifacts = [item, innovation_item]
    artifacts.append(store.array(f"{tag}-weights.npy", weights, seed=seed, step=step, axes=["innovation_index"],
                                 first_innovation_step=saved["first"], last_innovation_step=step, beta=.99))
    axes = {"gram": ["innovation_index", "innovation_index"], "eigenvalues": ["descending_eigenvalue"],
            "dual_vectors": ["innovation_index", "descending_mode"], "mapped_vectors": ["flat_parameter", "descending_mode"]}
    for name in ("gram", "eigenvalues", "dual_vectors", "mapped_vectors"):
        artifacts.append(store.array(f"{tag}-{name}.npy", tensors[name], seed=seed, step=step, role=name, axes=axes[name]))
    return {"schema_version": 1, "seed": seed, "step": step, "state_timing": saved["state_timing"],
            "parameter_hash": saved["parameter_hash"], "reference": ref, "observers": observers,
            "artifacts": artifacts, "all_numerical_gates_passed": True}


def worst_size_resource_pilot(store, guard, context):
    context.update(seed=9879, phase="synthetic_worst_size_write", step=0)
    generator = np.random.default_rng(9879)
    shape = (2000, 50890)
    store.stream("synthetic-raw.npy", shape)
    store.stream("synthetic-innovations.npy", shape)
    start = time.perf_counter()
    for i in range(shape[0]):
        row = np.zeros(shape[1], np.float32) if i == 0 else generator.standard_normal(shape[1], dtype=np.float32)
        for name in ("synthetic-raw.npy", "synthetic-innovations.npy"):
            store.append(name, i, row)
        if i % 100 == 0:
            guard.check()
            context["step"] = i + 1
    store.flush()
    write_seconds = time.perf_counter() - start
    start = time.perf_counter()
    artifacts = store.stream_artifacts()
    hash_seconds = time.perf_counter() - start
    context["phase"] = "synthetic_worst_size_reference"
    x, _ = mathref.weighted_columns(torch.from_numpy(store.maps["synthetic-innovations.npy"]), 2, device="cuda")
    result, tensors = mathref.reference(x, 2)
    resources = guard.check()
    timing = {key: result["diagnostics"][key] for key in ("gram_seconds", "solve_seconds", "mapping_seconds")}
    # Deliberately do not save reference spectra, capture or retention outcomes.
    return {"seed": 9879, "dimension": 50890, "columns": 2000, "all_reference_gates_passed": True,
            "timing": {**timing, "write_seconds": write_seconds, "hash_seconds": hash_seconds},
            "resources": resources, "artifacts": artifacts}


def full_gate():
    hashes = source_hashes(committed=True)
    pilot = json.loads((HERE / "pilot" / "execution.json").read_text())
    mathref.require(pilot["status"] == "complete" and pilot["mode"] == "pilot", "Missing passing pilot")
    mathref.require(pilot.get("all_gates_passed") is True, "Pilot did not pass every required gate")
    mathref.require(pilot["completed_development_traces"] == 2 and pilot["worst_size_resource_check_passed"], "Incomplete pilot")
    mathref.require(pilot["source_sha256"] == hashes, "Sources changed since pilot")
    return pilot


def run(mode):
    output = HERE / ("pilot" if mode == "pilot" else "results")
    mathref.require(not output.exists(), "Output already exists; preserve attempt before retry")
    h.configure()
    output.mkdir(exist_ok=False)
    guard = disk.Guard(mode == "pilot")
    record = {"schema_version": 1, "mode": mode, "status": "replay", "started_utc": h.utc(),
              "repository_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
              "python": sys.version, "numpy": np.__version__, "torch": torch.__version__, "cuda": torch.version.cuda,
              "cpu_threads": torch.get_num_threads(),
              "completed_replay_seeds": [], "completed_snapshots": [], "official_test_opened": False,
              "new_accuracy_or_checkpoint_selection_computed": False}
    context = {"phase": "source_and_launch_gates", "mode": mode}
    store, failure_capture = None, {}
    disk.write_json(output / "execution.json", record)
    previous_handler = signal.getsignal(signal.SIGALRM)
    def timeout_handler(signum, frame):
        raise TimeoutError("Prospective wall-time cap reached")
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(guard.limit)
    try:
        hashes = source_hashes(committed=mode == "full")
        pilot = full_gate() if mode == "full" else None
        context["phase"] = "occupancy_resource_gate"
        occupancy = resource_start_gate()
        torch.cuda.reset_peak_memory_stats()
        context["phase"] = "historical_provenance_gate"
        bindings, history = historical_bindings()
        if pilot is not None:
            mathref.require(pilot["historical_bindings"] == bindings, "Historical/data artifacts changed after pilot")
        store = disk.Store(mode)
        record.update(source_sha256=hashes, historical_bindings=bindings, bulk_root=str(store.root),
                      bulk_mount=store.mount, occupancy_before=occupancy, gpu=torch.cuda.get_device_name(0))
        disk.write_json(output / "execution.json", record)
        context["phase"] = "loading_training_data"
        x, y = h.load_training()
        if mode == "pilot":
            plan = h.make_plan(9878, 220)
            data = h.dataset_for_plan(x, y, plan, .9, "cuda")
            reports, finals = [], []
            for instrumented in (False, True):
                result, final = run_trajectory(plan, data, instrumented, True, store, guard, context, output)
                reports.append(result)
                finals.append(final)
            mathref.require(reports[0]["trajectory_parameter_sha256"] == reports[1]["trajectory_parameter_sha256"], "Pilot instrumentation changed trajectory")
            mathref.require(h.equal_tree(finals[0], finals[1]), "Pilot instrumentation changed complete core state")
            del data, finals
            disk.write_json(output / "timing-and-invariants.json", {"traces": reports, "bitwise_core_trajectory_equal": True})
            record["completed_development_traces"] = 2
            disk.write_json(output / "execution.json", record)
            resource_report = worst_size_resource_pilot(store, guard, context)
            disk.write_json(output / "worst-size-resource.json", resource_report)
            record["worst_size_resource_check_passed"] = True
        else:
            replays = []
            for seed in SEEDS:
                plan = h.make_plan(seed, STEPS)
                historical, checkpoints = load_historical_seed(seed, plan)
                data = h.dataset_for_plan(x, y, plan, .9, "cuda")
                replay, _ = run_trajectory(plan, data, True, False, store, guard, context, output, historical, checkpoints,
                                           failure_capture=failure_capture)
                disk.write_json(output / f"replay-seed{seed}.json", replay)
                replays.append(replay)
                record["completed_replay_seeds"].append(seed)
                disk.write_json(output / "execution.json", record)
                failure_capture.clear()
                del data, checkpoints
            record.update(status="reference", all_replay_completed_utc=h.utc())
            disk.write_json(output / "execution.json", record)
            for replay in replays:
                for item in replay["snapshot_artifacts"]:
                    row = reference_snapshot(replay, item, store, guard, context)
                    disk.write_json(output / f"snapshot-seed{row['seed']}-step{row['step']}.json", row)
                    record["completed_snapshots"].append({"seed": row["seed"], "step": row["step"]})
                    disk.write_json(output / "execution.json", record)
                    print(json.dumps({"phase": "reference", "seed": row["seed"], "step": row["step"],
                                      "completed_snapshots": len(record["completed_snapshots"]),
                                      "elapsed_seconds": guard.check()["elapsed_seconds"]}), flush=True)
            mathref.require(record["completed_replay_seeds"] == list(SEEDS) and len(record["completed_snapshots"]) == 12, "Incomplete confirmation")
        record.update(status="complete", completed_utc=h.utc(), resources=guard.check(), all_gates_passed=True)
        disk.write_json(output / "execution.json", record)
    except BaseException as error:
        record["failed_utc"] = h.utc()
        if "get" in failure_capture and store is not None:
            try:
                context["failure_state_artifact"] = store.tensor_bundle("failure-state.pt", failure_capture["get"]())
            except BaseException as capture_error:
                context["failure_state_capture_error"] = repr(capture_error)
        disk.preserve_failure(output, record, context, error, store)
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pilot", action="store_true")
    group.add_argument("--full", action="store_true")
    parser.add_argument("--development-go", action="store_true")
    parser.add_argument("--confirmatory-go", action="store_true")
    args = parser.parse_args()
    if args.pilot and not args.development_go:
        parser.error("Separate parent development-pilot GO required")
    if args.full and not args.confirmatory_go:
        parser.error("Separate parent confirmatory GO required")
    run("pilot" if args.pilot else "full")


if __name__ == "__main__":
    main()
