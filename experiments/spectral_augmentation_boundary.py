#!/usr/bin/env python3
"""Fixed augmentation study. Import is inert; only --execute admits acquisition.

No resume, seed/step overrides, diagnostics, adaptive search, or auto-retry.
The base scientific functions and constants are never monkeypatched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments import spectral_selectivity_boundary as base
from experiments import spectral_augmentation_masks as masks

core = base.core
need, digest, array_digest = base.need, base.digest, base.array_digest
DOCS = ROOT / 'output/2026-09-10-spectral-augmentation'
SCHEMA = 'spectral_augmentation_boundary_v1'
UNIT = 'spectral-augmentation-001.service'
SEEDS = masks.EXPERIMENT_SEEDS
CELLS = ('clean', 'shared', 'sham')
POLICIES = ('raw', 'native32')
AUGMENTATIONS = masks.MODES
WARMUP, STEPS, BATCH = 100, 2000, 64
EVAL_STEPS = tuple(range(0, 2001, 100))
MAX_BYTES, RESERVE_BYTES = 2 * 1024**3, 1024**2
DEADLINE_SECONDS = 25 * 60
BASE_SHA256 = 'cd8e1da355e3090951428ba72d0e062281c2891bfba326f367bf6845293f1ade'
# These are existing desktop clients, not a blanket allowance for other jobs.
GPU_CLIENT_MAX_MIB = {2101: 512, 8861: 128}


def branch_roster():
    """Frozen seed/cell/mode order; alternate policy-first position by indices."""
    rows = []
    for si, seed in enumerate(SEEDS):
        for ci, cell in enumerate(CELLS):
            for ai, augmentation in enumerate(AUGMENTATIONS):
                offset = (si + ci + ai) % 2
                for policy in POLICIES[offset:] + POLICIES[:offset]:
                    rows.append({'seed': seed, 'cell': cell, 'augmentation': augmentation,
                                 'policy': policy})
    return rows


def source_pins():
    pins = base.source_pins()
    need(digest(Path(base.__file__)) == BASE_SHA256, 'accepted base source changed')
    paths = [Path(__file__), Path(masks.__file__), DOCS / 'protocol.md',
             ROOT / 'tests/test_spectral_augmentation_masks.py',
             ROOT / 'tests/test_spectral_augmentation_boundary.py',
             ROOT / 'experiments/spectral_augmentation_analysis.py',
             ROOT / 'tests/test_spectral_augmentation_analysis.py',
             ROOT / 'experiments/spectral_augmentation_audit.py',
             ROOT / 'tests/test_spectral_augmentation_audit.py']
    pins.update({str(path.relative_to(ROOT)): digest(path) for path in paths})
    return pins


def byte_inventory():
    # Each state overcounts cleared gradients and raw endpoints without observers.
    full_state = 37 * base.P * 4 + 32 * 8 + 4 * 4 + 64 * 1024
    parts = {
        '78_initial_warmup_final_full_states': 78 * full_state,
        '72_full_logit_histories': 72 * 21 * 3 * 5000 * 10 * 4,
        'shared_initial_warmup_logits': 3 * 2 * 5 * 5000 * 10 * 4,
        'indexed_plans_labels_and_metadata_allowance': 16 * 1024**2,
        'gates_centers_four_rectangle_plans': 3 * 1900 * 64 * (1 + 4 + 4 * 8),
        '137100_action_rows_1024_bytes_each': 137100 * 1024,
        'curves_results_bindings_coverage_receipts_zip_overhead': 64 * 1024**2,
    }
    total = sum(parts.values())
    need(total + RESERVE_BYTES < MAX_BYTES, 'analytic artifact inventory exceeds 2 GiB')
    return {'component_upper_bytes': parts, 'total_upper_bytes': total,
            'cap_bytes': MAX_BYTES, 'cap_headroom_bytes': MAX_BYTES - total,
            'failure_metadata_reserve_bytes': RESERVE_BYTES}


def occurrence_coverage(rectangles, gates, batches, patch, true):
    """Counts use occurrence denominators, including repeated example IDs."""
    need(rectangles.shape == batches.shape + (4,) and gates.shape == batches.shape,
         'coverage occurrence shapes differ')
    need(batches.dtype.kind in 'iu' and batches.ndim == 2 and patch.dtype == np.bool_
         and patch.shape == true.shape and true.ndim == 1
         and ((batches >= 0) & (batches < len(true))).all(), 'invalid occurrence membership')
    selected = {'all': np.ones(batches.shape, dtype=bool),
                'actually_cued': patch[batches], 'rare': true[batches] == 8}
    result = {}
    for name, mask in selected.items():
        summary = masks.coverage_summary(rectangles[mask])
        summary['erased_area_counts'] = summary['erased_area_counts'].tolist()
        summary['gate_active'] = int(gates[mask].sum())
        summary['denominator'] = int(mask.sum())
        result[name] = summary
    return result


def augmented_batch(cell_cpu, batch, rectangles, device):
    """Cell cue already inserted on CPU. Complete FP32 erasure BEFORE transfer."""
    need(isinstance(cell_cpu, np.ndarray) and cell_cpu.dtype == np.float32,
         'canonical CPU float32 cell required')
    need(isinstance(batch, np.ndarray) and batch.dtype.kind in 'iu' and batch.ndim == 1
         and ((batch >= 0) & (batch < len(cell_cpu))).all(), 'invalid batch positions')
    values = masks.apply_masks(cell_cpu[batch], rectangles)
    return torch.from_numpy(values).to(device)


def continuation_update(model, optimizer, tracker, cell_cpu, assigned, batch,
                        rectangles, policy, step, device, check):
    need(policy in POLICIES and WARMUP < step <= STEPS, 'invalid continuation action')
    x = augmented_batch(cell_cpu, batch, rectangles, device)
    targets = torch.as_tensor(assigned[batch], device=device)
    row, diagnostic = base.training_update(model, optimizer, tracker, x, targets,
        policy, step, definitions=None, identity=None, check=check)
    need(diagnostic is None, 'diagnostics forbidden in augmentation acquisition')
    # Bound the JSON action inventory explicitly, rather than guessing later.
    need(len(json.dumps(row, allow_nan=False).encode()) + 2 <= 1024,
         'action row exceeded inventoried size')
    return row


def verified_restore(snapshot, expected_sha, device):
    model, optimizer, tracker = core.restore(snapshot, device)
    need(core.tree_digest(core.snapshot(model, optimizer, tracker)) == expected_sha,
         'full-state fork identity failed')
    return model, optimizer, tracker


class CappedWriter:
    def __init__(self, handle, allowance, check=lambda: None):
        self.handle, self.allowance, self.check = handle, allowance, check

    def write(self, value):
        self.check()
        size = memoryview(value).nbytes
        need(size <= self.allowance, 'artifact byte cap exhausted')
        count = self.handle.write(value)
        need(count == size, 'short artifact write')
        self.allowance -= count
        return count

    def flush(self):
        self.handle.flush()

    def read(self, *args):
        return self.handle.read(*args)


class Run:
    def __init__(self, path, device='cuda'):
        self.path, self.device = Path(path), device
        self.started, self.last_resource_check = time.monotonic(), 0.
        self.used, self.receipts = 0, []

    def check(self):
        now = time.monotonic()
        need(now - self.started < DEADLINE_SECONDS, '25-minute cooperative deadline exceeded')
        if now - self.last_resource_check > 1:
            need(shutil.disk_usage(self.path).free > 1024**3, 'less than 1 GiB free reserve')
            need(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 <= 16 * 1024**3,
                 '16 GiB host-memory guard exceeded')
            if self.device == 'cuda':
                need(torch.cuda.max_memory_allocated() <= 8 * 1024**3, '8 GiB GPU guard exceeded')
            self.last_resource_check = now

    def save(self, name, value, kind='json'):
        self.check()
        need(Path(name).name == name and name not in ('', '.', '..'), 'direct-child artifact required')
        need(kind in ('json', 'npz', 'tensor'), 'unknown artifact type')
        target = self.path / name
        try:
            with target.open('xb') as handle:
                writer = CappedWriter(handle, MAX_BYTES - self.used - RESERVE_BYTES, self.check)
                if kind == 'tensor':
                    torch.save(value, writer)
                elif kind == 'npz':
                    np.savez(writer, **value)
                else:
                    writer.write((json.dumps(value, allow_nan=False, separators=(',', ':')) + '\n').encode())
        except BaseException:
            # Include partial bytes in the failure budget; no retry in this Run.
            self.used = sum(p.stat().st_size for p in self.path.iterdir() if p.is_file())
            raise
        item = {'path': name, 'size_bytes': target.stat().st_size, 'sha256': digest(target)}
        self.used += item['size_bytes']
        self.receipts.append(item)
        self.check()
        return item


def acquire(run):
    images, labels = base.read_training()
    roster, results = branch_roster(), []
    for seed in SEEDS:
        run.check()
        plan, allocations = base.make_plan(labels, seed)
        need(plan['warmup_batches'].shape == (100, 64)
             and plan['continuation_batches'].shape == (1900, 64), 'fixed plan shape changed')
        mask_plan = masks.make_mask_plan(seed)
        rectangles = {mode: masks.rectangles_for_mode(mask_plan, mode) for mode in AUGMENTATIONS}
        mask_arrays = {'gates': mask_plan.gates, 'centers': mask_plan.centers,
                       **{mode + '_rectangles': value for mode, value in rectangles.items()}}
        plan_receipt = run.save(f'plan-s{seed}.npz', plan, 'npz')
        mask_receipt = run.save(f'masks-s{seed}.npz', mask_arrays, 'npz')
        plan_hashes = {name: array_digest(value) for name, value in plan.items()}
        mask_hashes = {name: array_digest(value) for name, value in mask_arrays.items()}
        metadata = run.save(f'plan-s{seed}.json', {'seed': seed, 'array_hashes': plan_hashes,
            'mask_array_hashes': mask_hashes, 'numpy_version': np.__version__,
            'rng': 'PCG64/SeedSequence([seed,stream_id]); gates float64 stream6; centers int16 stream7',
            'sham_allocation': allocations, 'unused_base_diffuse_fields': 'preserved, never trained'})
        clean_cpu = base.normalized_inputs(images, plan['train_ids'], 'cpu')
        clean_x = clean_cpu.to(run.device)
        heldout_x = base.normalized_inputs(images, plan['heldout_ids'], run.device)
        patched_heldout_x = base.patch_images(heldout_x)
        cell_data = {}
        for cell in CELLS:
            assigned, patch, counts = base.cell_definition(plan, cell)
            cpu = base.patch_images(clean_cpu, patch).numpy()
            need(cpu.dtype == np.float32, 'cell CPU dtype changed')
            cell_data[cell] = {'cpu': cpu, 'x': torch.from_numpy(cpu).to(run.device),
                               'assigned': assigned, 'patch': patch, 'counts': counts}
        model = core.make_model(seed, run.device)
        need(sum(p.numel() for p in model.parameters()) == base.P, 'model specification changed')
        optimizer = core.make_optimizer(model)
        tracker = core.make_tracker(model, optimizer)
        initial = core.snapshot(model, optimizer, tracker)
        initial_sha = core.tree_digest(initial)
        initial_receipt = run.save(f'initial-s{seed}.pt', initial, 'tensor')
        del model, optimizer, tracker
        model, optimizer, tracker = verified_restore(initial, initial_sha, run.device)
        initial_heldout = (base.predict(model, heldout_x), base.predict(model, patched_heldout_x))
        initial_train = {c: base.predict(model, data['x']) for c, data in cell_data.items()}
        need(core.equal_tree(initial, core.snapshot(model, optimizer, tracker)), 'initial evaluation changed state')
        base.sync(run.device)
        started = time.monotonic()
        warmup_actions = []
        for step, batch in enumerate(plan['warmup_batches'], 1):
            need(not (plan['train_true'][batch] == 8).any(), 'rare example in warmup')
            ids = torch.as_tensor(batch, device=run.device)
            target = torch.as_tensor(plan['train_true'][batch], device=run.device)
            row, diagnostic = base.training_update(model, optimizer, tracker, clean_x[ids],
                target, 'raw', step, definitions=None, check=run.check)
            need(diagnostic is None, 'warmup diagnostics forbidden')
            need(len(json.dumps(row, allow_nan=False).encode()) + 2 <= 1024, 'warmup action size')
            warmup_actions.append(row)
        base.sync(run.device)
        warmup_seconds = time.monotonic() - started
        warmup = core.snapshot(model, optimizer, tracker)
        warmup_sha = core.tree_digest(warmup)
        warmup_receipt = run.save(f'warmup-s{seed}.pt', warmup, 'tensor')
        run.save(f'warmup-actions-s{seed}.json', warmup_actions)
        warmup_heldout = (base.predict(model, heldout_x), base.predict(model, patched_heldout_x))
        warmup_train = {c: base.predict(model, data['x']) for c, data in cell_data.items()}
        need(core.equal_tree(warmup, core.snapshot(model, optimizer, tracker)), 'warmup evaluation changed state')
        run.save(f'common-logits-s{seed}.npz', {
            'steps': np.array([0, 100], dtype=np.int64),
            'heldout_unpatched': np.stack([initial_heldout[0], warmup_heldout[0]]),
            'heldout_patched': np.stack([initial_heldout[1], warmup_heldout[1]]),
            **{c + '_train': np.stack([initial_train[c], warmup_train[c]]) for c in CELLS}}, 'npz')
        del model, optimizer, tracker, initial
        first_raw = {}
        for identity in (r for r in roster if r['seed'] == seed):
            run.check()
            cell, mode, policy = identity['cell'], identity['augmentation'], identity['policy']
            data = cell_data[cell]
            identifier = f's{seed}-{cell}-{mode}-{policy}'
            model, optimizer, tracker = verified_restore(warmup, warmup_sha, run.device)
            if policy == 'raw':
                tracker = None  # After verified full-state restore, as in accepted base.
            tr_history = [initial_train[cell], warmup_train[cell]]
            va_history = [initial_heldout[0], warmup_heldout[0]]
            pa_history = [initial_heldout[1], warmup_heldout[1]]
            curve = [base.evaluation_row(step, tr, va, pa, plan['train_true'], data['assigned'],
                plan['heldout_true']) for step, tr, va, pa in zip((0, 100), tr_history, va_history, pa_history)]
            base.sync(run.device)
            started, actions = time.monotonic(), []
            for index, batch in enumerate(plan['continuation_batches']):
                step = index + 101
                row = continuation_update(model, optimizer, tracker, data['cpu'], data['assigned'],
                    batch, rectangles[mode][index], policy, step, run.device, run.check)
                actions.append(row)
                if step == 101:
                    key = (cell, mode)
                    sha = row['raw_gradient_sha256']
                    need(key not in first_raw or first_raw[key] == sha, 'paired first raw gradient differs')
                    first_raw[key] = sha
                if step in EVAL_STEPS:
                    tr, va, pa = (base.predict(model, data['x']), base.predict(model, heldout_x),
                                  base.predict(model, patched_heldout_x))
                    tr_history.append(tr)
                    va_history.append(va)
                    pa_history.append(pa)
                    curve.append(base.evaluation_row(step, tr, va, pa, plan['train_true'],
                        data['assigned'], plan['heldout_true']))
                    print(json.dumps({**identity, 'step': step,
                        'elapsed_seconds': time.monotonic() - run.started}), flush=True)
            base.sync(run.device)
            branch_seconds = time.monotonic() - started
            need(len(actions) == 1900 and [r['step'] for r in curve] == list(EVAL_STEPS), 'incomplete branch')
            final = core.snapshot(model, optimizer, tracker)
            coverage = occurrence_coverage(rectangles[mode], mask_plan.gates,
                plan['continuation_batches'], data['patch'], plan['train_true'])
            record = {**identity, 'schema': SCHEMA, 'plan': plan_receipt, 'mask_plan': mask_receipt,
                'plan_metadata': metadata, 'plan_array_hashes': plan_hashes, 'mask_array_hashes': mask_hashes,
                'initial_state': initial_receipt, 'initial_state_sha256': initial_sha,
                'warmup_state': warmup_receipt, 'warmup_state_sha256': warmup_sha,
                'restored_state_sha256': warmup_sha,
                'full_state_fork_check': 'PASS', 'coverage': coverage, 'cell_counts': data['counts'],
                'cpu_cell_inputs_sha256': array_digest(data['cpu']),
                'assigned_targets_sha256': array_digest(data['assigned']),
                'cell_patch_mask_sha256': array_digest(data['patch']),
                'input_convention': 'CPU NumPy float32 / float32(255); canonical cue then occurrence erasure then transfer',
                'warmup_seconds': warmup_seconds, 'branch_seconds_including_evaluation': branch_seconds,
                'warmup_plus_branch_seconds': warmup_seconds + branch_seconds,
                'final_state_sha256': core.tree_digest(final),
                'final_state': run.save(f'final-{identifier}.pt', final, 'tensor'),
                'logits': run.save(f'logits-{identifier}.npz', {'steps': np.array(EVAL_STEPS, dtype=np.int64),
                    'train': np.stack(tr_history), 'heldout_unpatched': np.stack(va_history),
                    'heldout_patched': np.stack(pa_history)}, 'npz'),
                'actions': run.save(f'actions-{identifier}.json', actions),
                'first_action_binding': {k: v for k, v in actions[0].items() if k.endswith('sha256')},
                'curve': curve, 'diagnostics': []}
            receipt = run.save(f'curve-{identifier}.json', record)
            results.append({**identity, 'curve': receipt, 'endpoint': curve[-1], 'warmup': curve[1],
                            'warmup_plus_branch_seconds': warmup_seconds + branch_seconds})
            del model, optimizer, tracker, final, actions, tr_history, va_history, pa_history
        need(len(first_raw) == 12, 'incomplete cell/mode gradient pairing')
        run.save(f'pairing-s{seed}.json', {'status': 'PASS', 'full_state_forks': 24,
            'first_raw_gradient_sha256': {c + '/' + a: v for (c, a), v in first_raw.items()}})
        need({k: array_digest(v) for k, v in mask_arrays.items()} == mask_hashes, 'mask plan mutated')
        need({k: array_digest(v) for k, v in plan.items()} == plan_hashes, 'data plan mutated')
        del warmup, clean_cpu, clean_x, heldout_x, patched_heldout_x, cell_data
    need([{k: r[k] for k in ('seed', 'cell', 'augmentation', 'policy')} for r in results] == roster,
         'incomplete ordered 72-branch roster')
    return run.save('results.json', {'schema': SCHEMA, 'rows': results})


def validate_gpu_clients(text):
    clients = []
    for line in text.splitlines():
        if not line.strip():
            continue
        fields = [v.strip() for v in line.split(',')]
        need(len(fields) == 2, 'unrecognized GPU client inventory')
        pid, mib = map(int, fields)
        need(pid in GPU_CLIENT_MAX_MIB and 0 <= mib <= GPU_CLIENT_MAX_MIB[pid],
             'GPU occupied beyond pre-existing desktop clients')
        clients.append({'pid': pid, 'used_gpu_memory_mib': mib})
    return clients


def configure():
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        need(os.environ.get(name) == '1', 'set ' + name + '=1 before Python')
    need(os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8', 'deterministic CUDA workspace required')
    group = next(line.split(':', 2)[2] for line in Path('/proc/self/cgroup').read_text().splitlines()
                 if line.startswith('0::'))
    need(Path(group).name == UNIT, 'unexpected acquisition unit')
    root = Path('/sys/fs/cgroup') / group.lstrip('/')
    effective = {k: (root / k).read_text().strip() for k in ('memory.max', 'memory.swap.max', 'cpu.max')}
    properties = subprocess.check_output(['systemctl', '--user', 'show', UNIT, '--property=Type',
        '--property=RuntimeMaxUSec', '--property=Restart', '--property=KillMode'], text=True)
    service = dict(line.split('=', 1) for line in properties.splitlines())
    base.validate_bounds(effective, service)
    clients = validate_gpu_clients(subprocess.check_output(['nvidia-smi',
        '--query-compute-apps=pid,used_gpu_memory', '--format=csv,noheader,nounits'], text=True))
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    need(torch.__version__ == '2.11.0+cu128', 'unexpected torch version')
    need(torch.cuda.is_available() and torch.cuda.get_device_name() == 'NVIDIA GeForce RTX 3090',
         'local RTX3090 required')
    free, total = torch.cuda.mem_get_info()
    need(free >= 8 * 1024**3, 'at least 8 GiB free GPU required')
    torch.cuda.set_per_process_memory_fraction(8 * 1024**3 / total)
    return {'cgroup': group, 'effective': effective, 'service': service, 'preexisting_gpu_clients': clients}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--output-dir', type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    need(args.execute, 'No acquisition without explicit --execute')
    parent = args.output_dir.parent.resolve(strict=True)
    need(parent.is_relative_to(Path('/tmp/spectral-experiment-artifacts')) and not list(parent.iterdir()),
         'unused large-volume parent required')
    need(args.output_dir == parent / 'acquisition-001' and not args.output_dir.exists()
         and not args.output_dir.is_symlink(), 'new exclusive acquisition-001 required')
    mount = subprocess.check_output(['findmnt', '-n', '-o', 'TARGET,SOURCE', '--target', str(parent)], text=True).split()
    need(mount == ['/private-artifacts/storage', '/dev/RECONFIGURE_FOR_LOCAL_STORAGE'], 'wrong output mount identity')
    inventory, pins = byte_inventory(), source_pins()
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    for name, sha in pins.items():
        committed = subprocess.check_output(['git', 'show', commit + ':' + name], cwd=ROOT)
        need(hashlib.sha256(committed).hexdigest() == sha, 'source not committed: ' + name)
    need(shutil.disk_usage(parent).free > MAX_BYTES + 1024**3, '2 GiB cap plus 1 GiB reserve required')
    # Cross-directory, exclusive attempt guard. A failed attempt is terminal too.
    # Written ONLY under --execute after committed-source and output admission.
    with (DOCS / 'acquisition-attempt.json').open('x') as handle:
        json.dump({'unit': UNIT, 'output': str(args.output_dir), 'pid': os.getpid(),
                   'started_unix': time.time(), 'git_commit': commit}, handle)
    args.output_dir.mkdir(exist_ok=False)
    run = Run(args.output_dir)
    try:
        admission = configure()
        data_pins = {name: digest(base.DATA / name) for name in base.DATA_SHA256}
        need(data_pins == base.DATA_SHA256, 'accepted original training IDX hashes differ')
        manifest = {'schema': SCHEMA, 'source_pins': pins, 'git_commit': commit,
            'data_pins': data_pins, 'data_directory': str(base.DATA), 'resource_admission': admission,
            'pid': os.getpid(), 'invocation_id': os.environ.get('INVOCATION_ID'),
            'started_unix': time.time(), 'branch_roster': branch_roster(), 'seeds': list(SEEDS),
            'cells': list(CELLS), 'policies': list(POLICIES), 'augmentations': list(AUGMENTATIONS),
            'warmup': WARMUP, 'steps': STEPS, 'batch_size': BATCH, 'eval_steps': list(EVAL_STEPS),
            'rank': 32, 'cooperative_seconds': DEADLINE_SECONDS, 'byte_inventory': inventory,
            'numpy': np.__version__, 'torch': torch.__version__, 'python': sys.version,
            'cloud_spend_usd': 0, 'diagnostics': [], 'timing_scope': 'training and evaluation; not a speed comparison'}
        run.save('manifest.json', manifest)
        print(json.dumps({'resource_guard': 'PASS', 'unit': UNIT, 'pid': os.getpid(),
                          'output': str(args.output_dir), 'inventory': inventory}), flush=True)
        result = acquire(run)
        need(source_pins() == pins and {n: digest(base.DATA / n) for n in data_pins} == data_pins,
             'sources or data changed during acquisition')
        run.save('complete.json', {'schema': SCHEMA, 'status': 'complete', 'results': result,
            'source_pins': pins, 'data_pins': data_pins, 'finished_unix': time.time(),
            'wall_seconds': time.monotonic() - run.started, 'completed_trajectories': 72,
            'completed_diagnostics': 0, 'gpu_max_allocated_bytes': torch.cuda.max_memory_allocated(),
            'process_high_water_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'artifact_bytes_before_completion': run.used, 'receipts': list(run.receipts)})
    except BaseException as exc:
        failure = {'schema': SCHEMA, 'status': 'failed', 'type': type(exc).__name__, 'message': str(exc),
            'finished_unix': time.time(), 'wall_seconds': time.monotonic() - run.started,
            'partial_artifact_bytes': run.used, 'completed_receipts': run.receipts}
        payload = json.dumps(failure, allow_nan=False).encode()
        need(len(payload) <= RESERVE_BYTES, 'failure metadata reserve exceeded')
        with (args.output_dir / 'failed.json').open('xb') as handle:
            handle.write(payload)
        raise


if __name__ == '__main__':
    main()
