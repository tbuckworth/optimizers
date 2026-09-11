#!/usr/bin/env python3
"""Once-only fixed-state diagnostic; import inert, execution explicitly gated."""
from __future__ import annotations

import argparse
import hashlib
import io
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
sys.path.insert(0, str(ROOT))
from experiments import spectral_general_augmentation as base
from experiments import spectral_general_augmentation_data as data

core, need, digest = base.core, base.need, base.digest
DOCS = ROOT / 'output/2026-09-10-spectral-augmentation-state'
SOURCE_DOCS = base.DOCS
SCHEMA = 'spectral_augmentation_state_v1'
UNIT = 'spectral-augmentation-state-001.service'
SEEDS = (202609141, 202609142, 202609143)
WARMUPS = ('none', 'translate')
MAX_BYTES = 1024**3
P = 50890


def diagnostic_plan(seed, source, labels):
    """Pure fixed-size PCG plans; source IDs are global IDX row numbers."""
    need(type(seed) is int and seed in SEEDS, 'fixed seed required')
    need(isinstance(labels, np.ndarray) and labels.dtype == np.int64
         and labels.ndim == 1 and np.all((labels >= 0) & (labels < 10)), 'label schema')
    for key in ('train_ids', 'eval_ids'):
        a = source[key]
        need(a.dtype == np.int64 and a.shape == (5000,) and len(np.unique(a)) == 5000
             and np.all((a >= 0) & (a < len(labels))), 'source IDs')
    need(not np.intersect1d(source['train_ids'], source['eval_ids']).size, 'source split overlap')
    rng = lambda stream: np.random.Generator(np.random.PCG64(np.random.SeedSequence([stream, seed])))
    train_local = rng(10).permutation(5000)[:64].astype(np.int64)
    eval_local = rng(12).permutation(5000)[:256].astype(np.int64)
    train_ids, eval_ids = source['train_ids'][train_local], source['eval_ids'][eval_local]
    return {'train_positions': train_local, 'eval_positions': eval_local,
            'train_ids': train_ids.copy(), 'eval_ids': eval_ids.copy(),
            'train_labels': labels[train_ids].copy(), 'eval_labels': labels[eval_ids].copy(),
            'train_shifts': rng(11).integers(-2, 3, size=(4, 64, 2), dtype=np.int8),
            'eval_shifts': rng(13).integers(-2, 3, size=(256, 2), dtype=np.int8)}


def receipt_bytes(root, receipt, cap=16 * 1024**2):
    name = receipt['path']
    need(type(name) is str and Path(name).name == name, 'receipt must name direct child')
    target = Path(root) / name
    need(not target.is_symlink() and target.is_file(), 'receipt path is not regular file')
    need(type(receipt['size_bytes']) is int and 0 < receipt['size_bytes'] <= cap
         and target.stat().st_size == receipt['size_bytes'], 'receipt size mismatch')
    payload = target.read_bytes()
    need(hashlib.sha256(payload).hexdigest() == receipt['sha256'], 'receipt hash mismatch')
    return payload


def validate_snapshot(value):
    need(type(value) is dict and tuple(value) == ('schema', 'model_spec', 'model_state',
         'gradients', 'model_modes', 'optimizer', 'tracker', 'rng')
         and value['schema'] == 'i9_neural_snapshot_v1', 'snapshot schema')
    need(value['model_spec'] == {'input_dim': 784, 'width': 64, 'classes': 10}, 'model specification')
    core.tree_digest(value)  # Recursively rejects unsupported/nonfinite scalar values.
    def finite_tree(item):
        if type(item) is torch.Tensor:
            need(item.device.type == 'cpu' and bool(torch.isfinite(item).all()), 'nonfinite/non-CPU snapshot tensor')
        elif type(item) is dict:
            for child in item.values():
                finite_tree(child)
        elif type(item) in (list, tuple):
            for child in item:
                finite_tree(child)
    finite_tree(value)
    shapes = {'0.weight': (64, 784), '0.bias': (64,), '2.weight': (10, 64), '2.bias': (10,)}
    need(tuple(value['model_state']) == tuple(shapes), 'parameter names')
    for key, shape in shapes.items():
        a = value['model_state'][key]
        need(type(a) is torch.Tensor and a.device.type == 'cpu' and a.dtype == torch.float32
             and tuple(a.shape) == shape and bool(torch.isfinite(a).all()), 'parameter schema')
    gradients = value['gradients']
    need(type(gradients) is list and len(gradients) == 4, 'gradient topology')
    for a, shape in zip(gradients, shapes.values()):
        need(a is None or (type(a) is torch.Tensor and a.dtype == torch.float32
                          and tuple(a.shape) == shape), 'gradient schema')
    modes = value['model_modes']
    need(type(modes) is list and [row[0] for row in modes] == ['', '0', '1', '2']
         and all(type(row) is tuple and len(row) == 2 and type(row[1]) is bool for row in modes), 'module mode schema')
    rng = value['rng']
    need(type(rng) is dict and tuple(rng) == ('python', 'numpy', 'torch_cpu', 'torch_cuda')
         and type(rng['python']) is tuple and type(rng['torch_cuda']) is list
         and type(rng['torch_cpu']) is torch.Tensor and rng['torch_cpu'].dtype == torch.uint8,
         'RNG schema')
    opt = value['optimizer']
    need(set(opt) == {'state', 'param_groups'} and set(opt['state']) == set(range(4))
         and len(opt['param_groups']) == 1, 'Adam topology')
    group = opt['param_groups'][0]
    for key, expected in {'lr': .001, 'weight_decay': .01, 'betas': (.9, .999),
                          'eps': 1e-8, 'amsgrad': False, 'maximize': False,
                          'foreach': False, 'fused': False, 'params': list(range(4))}.items():
        need(group[key] == expected, 'Adam configuration: ' + key)
    for i, shape in enumerate(shapes.values()):
        row = opt['state'][i]
        need(set(row) == {'step', 'exp_avg', 'exp_avg_sq'} and row['step'].numel() == 1
             and float(row['step']) == 100, 'Adam update must be 100')
        for key in ('exp_avg', 'exp_avg_sq'):
            a = row[key]
            need(a.dtype == torch.float32 and tuple(a.shape) == shape and bool(torch.isfinite(a).all()), 'Adam moment schema')
        need(bool((row['exp_avg_sq'] >= 0).all()), 'negative second moment')
    tracker = value['tracker']
    need(type(tracker) is dict and tracker['step_count'] == 100 and tracker['warmup'] == 100
         and tracker['rank'] == 32 and tracker['stable_update'] is True, 'native observer configuration')
    for key, expected in {'decay': .99, 'filter_strength': 1., 'energy_threshold': None,
                          'adaptive': 'none', 'normalize': 'none', 'weighting': 'hard',
                          'alpha': 1., 'soft_residual': True, 'relative_eig_tol': 1e-8,
                          'absolute_eig_floor': 0., 'stabilize_every': 100}.items():
        need(tracker[key] == expected, 'observer configuration: '+key)
    for key in ('V', 'S', 'grad_mean'):
        a = tracker[key]
        need(type(a) is torch.Tensor and bool(torch.isfinite(a).all()), 'observer tensor schema')
    need(tuple(tracker['V'].shape) == (P, 32) and tracker['grad_mean'].numel() == P, 'observer dimensions')


def load_parent(root, receipt):
    payload = receipt_bytes(root, receipt)
    value = torch.load(io.BytesIO(payload), weights_only=True, map_location='cpu')
    validate_snapshot(value)
    return value


def byte_inventory():
    # Includes all full-size arrays, ZIP headers, plans, extra scalar arrays and JSON.
    floats_per_parent = P * (5*64 + 5 + 3 + 32 + 5*32 + 5*2 + 5*5 + 5*5*2 + 2)
    logits_per_parent = 2*256*10 + 5*5*2*2*256*10
    per_parent = 4*(floats_per_parent + logits_per_parent) + 1024**2
    total = 6*per_parent + 16*1024**2
    need(per_parent < 128*1024**2 and total + base.RESERVE_BYTES < MAX_BYTES, 'artifact inventory')
    return {'per_parent_upper_bytes': per_parent, 'total_upper_bytes': total, 'cap_bytes': MAX_BYTES}


def source_pins():
    names = ['spectral_filter.py', str(Path(core.__file__).relative_to(ROOT)),
             'experiments/spectral_general_augmentation.py', 'experiments/spectral_general_augmentation_data.py',
             'experiments/spectral_general_augmentation_audit.py',
             'experiments/spectral_augmentation_state.py', 'experiments/spectral_augmentation_state_core.py',
             'experiments/spectral_augmentation_state_audit.py',
             'tests/test_spectral_augmentation_state.py', 'tests/test_spectral_augmentation_state_core.py',
             'tests/test_spectral_augmentation_state_audit.py']
    names += [str((DOCS / n).relative_to(ROOT)) for n in ('protocol.md', 'parent-pins.json', 'implementation-check.md')]
    pins = {n: digest(ROOT / n) for n in names}
    need(pins['spectral_filter.py'] == core.FILTER_SHA256 and pins[str(Path(core.__file__).relative_to(ROOT))] == base.CORE_SHA, 'canonical source changed')
    need(pins['experiments/spectral_general_augmentation.py'] == 'abffd85aa30f350cc2a6c857758a315d878c27de170c1a4a2adee2f39101538e'
         and pins['experiments/spectral_general_augmentation_data.py'] == '4404d59fe79e294abc946cd3a58963d1b6e84d65e47d29c33bcebbde7003396a', 'frozen reused helper changed')
    need(pins['experiments/spectral_general_augmentation_audit.py'] == '9f2c38c48a9d0dc14c733930a1022fe094caf623cb10a524bf1a262612d2cb15', 'frozen reused audit helper changed')
    return pins


def configure():
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        need(os.environ.get(key) == '1', 'set ' + key + '=1')
    need(os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8', 'deterministic workspace required')
    group = next(x.split(':', 2)[2] for x in Path('/proc/self/cgroup').read_text().splitlines() if x.startswith('0::'))
    need(Path(group).name == UNIT, 'unexpected acquisition unit')
    root = Path('/sys/fs/cgroup') / group.lstrip('/')
    effective = {k: (root/k).read_text().strip() for k in ('memory.max', 'memory.swap.max', 'cpu.max')}
    props = subprocess.check_output(['systemctl', '--user', 'show', UNIT, '--property=Type', '--property=RuntimeMaxUSec', '--property=Restart', '--property=KillMode'], text=True)
    service = dict(x.split('=', 1) for x in props.splitlines())
    base.validate_bounds(effective, service)
    clients = base.validate_gpu_clients(subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_gpu_memory', '--format=csv,noheader,nounits'], text=True))
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    need(torch.__version__ == '2.11.0+cu128' and np.__version__ == '1.26.4', 'framework versions')
    need(torch.cuda.is_available() and torch.cuda.get_device_name() == 'NVIDIA GeForce RTX 3090', 'RTX3090 required')
    free, total = torch.cuda.mem_get_info()
    need(free >= 8*1024**3, '8GiB free GPU required')
    torch.cuda.set_per_process_memory_fraction(8*1024**3/total)
    return {'cgroup': group, 'effective': effective, 'service': service, 'preexisting_gpu_clients': clients}


def validate_source_metadata(registry, results, audit, attempt):
    need(results['schema'] == base.SCHEMA and results['status'] == 'complete', 'source acquisition incomplete')
    need(audit['status'] == 'PASS' and audit['input_results_sha256'] == registry['source_results_sha256']
         and audit['input_dir'] == registry['input_dir'], 'source audit not passed/bound')
    # The immutable attempt records launch provenance, not terminal status.
    need(attempt['schema'] == base.SCHEMA and attempt['output_dir'] == registry['input_dir']
         and attempt['source_pins'] == results['source_pins'], 'source attempt binding')
    need(all(r in results['receipts'] for r in registry['receipts']), 'parent receipt not in completed source')


def acquire(run, registry):
    from experiments import spectral_augmentation_state_core as helper
    archive = Path(registry['input_dir'])
    need(digest(archive/'results.json') == registry['source_results_sha256']
         and digest(SOURCE_DOCS/'audit.json') == registry['source_audit_sha256']
         and digest(SOURCE_DOCS/'attempt.json') == registry['source_attempt_sha256'], 'source provenance binding')
    validate_source_metadata(registry, json.loads((archive/'results.json').read_text()),
                             json.loads((SOURCE_DOCS/'audit.json').read_text()),
                             json.loads((SOURCE_DOCS/'attempt.json').read_text()))
    receipts = {r['path']: r for r in registry['receipts']}
    expected = {f'plan-s{s}.npz' for s in SEEDS} | {f'warmup-s{s}-native32-{w}.pt' for s in SEEDS for w in WARMUPS}
    need(set(receipts) == expected and len(registry['receipts']) == 9, 'parent roster')
    images, labels = base.read_training()
    rows = []
    for seed in SEEDS:
        source_receipt = receipts[f'plan-s{seed}.npz']
        with np.load(io.BytesIO(receipt_bytes(archive, source_receipt)), allow_pickle=False) as z:
            source = {k: z[k] for k in z.files}
        regenerated = data.make_plan(labels, seed)
        need(all(np.array_equal(source[k], v) for k, v in regenerated.items()), 'source plan reconstruction')
        need(np.array_equal(source['train_labels'], labels[source['train_ids']]) and np.array_equal(source['eval_labels'], labels[source['eval_ids']]), 'source label binding')
        plan = diagnostic_plan(seed, source, labels)
        plan_receipt = run.save(f'plan-s{seed}.npz', plan, 'npz')
        x = images[plan['train_ids']].astype(np.float32)/np.float32(255)
        e = images[plan['eval_ids']].astype(np.float32)/np.float32(255)
        views = np.stack([x] + [data.translate(x, shifts) for shifts in plan['train_shifts']], axis=1)
        objectives = np.stack([e, data.translate(e, plan['eval_shifts'])])
        for warmup in WARMUPS:
            run.check()
            name = f's{seed}-native32-{warmup}'
            parent_receipt = receipts['warmup-'+name+'.pt']
            snapshot = load_parent(archive, parent_receipt)
            before = core.tree_digest(snapshot)
            model, opt, tracker = core.restore(snapshot, run.device)
            live_before = core.tree_digest(core.snapshot(model, opt, tracker))
            need(live_before == before, 'restore changed snapshot content')
            g = helper.collect_gradients(model, torch.from_numpy(views).to(run.device), torch.from_numpy(plan['train_labels']).to(run.device), check=run.check)
            proposals = helper.propose_actions(model, opt, tracker, g['batch'], check=run.check)
            measured = helper.measure_actions(model, proposals, torch.from_numpy(objectives).to(run.device), torch.from_numpy(plan['eval_labels']).to(run.device), scales=(1., .1), check=run.check)
            keys = ('theta', 'adam_m', 'adam_v', 'adam_steps', 'param_sizes', 'basis_before',
                    'basis_before_rank', 'basis_after', 'basis_after_ranks', 'applied_grads',
                    'planned_data', 'after_parameters', 'valid', 'scales')
            arrays = {k: proposals[k] for k in keys}
            arrays.update({k: measured[k] for k in ('after_logits', 'objective_grads', 'baseline_logits')})
            arrays.update(per_example_grads=g['per_example'].transpose(1, 0, 2).copy(), batch_grads=g['batch'])
            need(all(isinstance(a, np.ndarray) and a.dtype.kind in 'fbiu' and np.isfinite(a).all() for a in arrays.values()), 'finite numeric arrays only')
            need(sum(a.nbytes for a in arrays.values()) + 65536 < 128*1024**2, 'parent archive cap')
            need(core.tree_digest(snapshot) == before and core.tree_digest(core.snapshot(model, opt, tracker)) == live_before, 'parent mutated in memory')
            need(digest(archive/parent_receipt['path']) == parent_receipt['sha256'], 'parent changed on disk')
            output = run.save('parent-'+name+'.npz', arrays, 'npz')
            rows.append({'name': name, 'seed': seed, 'warmup': warmup, 'plan_receipt': plan_receipt,
                         'source_plan_receipt': source_receipt, 'parent_receipt': parent_receipt,
                         'arrays_receipt': output, 'parent_hash_before': parent_receipt['sha256'],
                         'parent_hash_after': digest(archive/parent_receipt['path']),
                         'parent_tree_sha256': before, 'restored_tree_sha256': live_before,
                         'mean_error_norm': g['mean_error_norm'].tolist(),
                         'mean_relative_error': g['mean_relative_error'].tolist(), 'parent_unchanged': True})
            del arrays, g, proposals, measured, model, opt, tracker, snapshot
            print(json.dumps({'completed_parent': name}), flush=True)
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    need(args.execute, 'No acquisition without explicit --execute')
    parent = args.output_dir.parent.resolve(strict=True)
    need(parent.is_relative_to(Path('/tmp/spectral-experiment-artifacts')) and not list(parent.iterdir()), 'unused large-volume parent required')
    need(args.output_dir == parent/'acquisition-001' and not args.output_dir.exists() and not args.output_dir.is_symlink(), 'exclusive acquisition required')
    mount = subprocess.check_output(['findmnt', '-n', '-o', 'TARGET,SOURCE', '--target', str(parent)], text=True).split()
    need(mount == ['/private-artifacts/storage', '/dev/RECONFIGURE_FOR_LOCAL_STORAGE'] and shutil.disk_usage(parent).free > MAX_BYTES+1024**3, 'output mount/free space')
    pins, inventory = source_pins(), byte_inventory()
    commit = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    for path, sha in pins.items():
        need(hashlib.sha256(subprocess.check_output(['git', '-C', str(ROOT), 'show', commit+':'+path])).hexdigest() == sha, 'uncommitted source: '+path)
    registry = json.loads((DOCS/'parent-pins.json').read_text())
    attempt = {'schema': SCHEMA, 'commit': commit, 'source_pins': pins, 'data_pins': base.DATA_PINS,
               'output_dir': str(args.output_dir), 'unit': UNIT, 'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    with (DOCS/'attempt.json').open('x') as f:
        json.dump(attempt, f, indent=2)
    args.output_dir.mkdir()
    run = base.Run(args.output_dir)
    try:
        guards = configure()
        run.save('provenance.json', {**attempt, 'guards': guards, 'inventory': inventory, 'parents': registry})
        records = acquire(run, registry)
        need(source_pins() == pins and {n: digest(base.DATA/n) for n in base.DATA_PINS} == base.DATA_PINS, 'source/data changed')
        result = {'schema': SCHEMA, 'status': 'complete', 'source_pins': pins, 'data_pins': base.DATA_PINS,
                  'parent_pins': registry, 'parents': records,
                  'roster': [{'seed': s, 'warmup': w, 'name': f's{s}-native32-{w}'} for s in SEEDS for w in WARMUPS],
                  'receipts': list(run.receipts),
                  'seconds': time.monotonic()-run.started, 'gpu_max_allocated_bytes': torch.cuda.max_memory_allocated(),
                  'host_high_water_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
        receipt = run.save('results.json', result)
        print(json.dumps({'status': 'complete', 'result': receipt, 'bytes': run.used}), flush=True)
    except BaseException as exc:
        with (args.output_dir/'failure.json').open('x') as f:
            json.dump({'status': 'failed', 'error_type': type(exc).__name__, 'message': str(exc)[:2000]}, f)
        raise


if __name__ == '__main__':
    main()
