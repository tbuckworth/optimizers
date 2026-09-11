#!/usr/bin/env python3
"""Fixed clean multiview experiment. Import inert; one explicit guarded launch."""
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
sys.path.insert(0, str(ROOT))
from experiments import spectral_general_augmentation as base

core, need, digest = base.core, base.need, base.digest
DOCS = ROOT / 'output/2026-09-10-spectral-multiview-clean'
SCHEMA = 'spectral_multiview_clean_v1'
UNIT = 'spectral-multiview-clean-001.service'
SEEDS = (202609161, 202609162, 202609163)
POLICIES = ('raw1', 'native1', 'observer4', 'raw4')
STEPS, BATCH = 4000, 64
EVAL_STEPS = (0, 100) + tuple(range(200, 4001, 200))
PANELS = ('train', 'heldout', 'heldout_translated')
FLOAT_STREAMS = ('observation_norm', 'delivery_norm', 'applied_norm', 'data_step_norm')
INT_STREAMS = ('gradient_evaluations', 'observer_steps', 'adam_steps', 'basis_rank')
MAX_BYTES = 1024**3
OLD_PINS = {
    'spectral_filter.py': '9280c7d360aef4c775baf0d781a5afd56e016e9cfb93ef829c3ae4b8cc3df943',
    'experiments/spectral_general_augmentation.py': 'abffd85aa30f350cc2a6c857758a315d878c27de170c1a4a2adee2f39101538e',
    'experiments/spectral_general_augmentation_data.py': '4404d59fe79e294abc946cd3a58963d1b6e84d65e47d29c33bcebbde7003396a',
    'experiments/spectral_general_augmentation_audit.py': '9f2c38c48a9d0dc14c733930a1022fe094caf623cb10a524bf1a262612d2cb15',
    str(Path(core.__file__).relative_to(ROOT)): base.CORE_SHA,
}


def branch_roster():
    return [{'seed': seed, 'policy': policy}
            for index, seed in enumerate(SEEDS)
            for policy in POLICIES[index:] + POLICIES[:index]]


def empty_stream():
    return {'losses': np.zeros((STEPS, 4), dtype=np.float64),
            **{k: np.zeros(STEPS, dtype=np.float64) for k in FLOAT_STREAMS},
            **{k: np.zeros(STEPS, dtype=np.int64) for k in INT_STREAMS}}


def record_step(stream, index, values, policy):
    """Validate the helper's scalar contract before writing each fixed record."""
    need(type(index) is int and 0 <= index < STEPS and policy in POLICIES, 'step/policy')
    need(set(values) == {'losses', *FLOAT_STREAMS, *INT_STREAMS}, 'step result schema')
    losses = values['losses']
    views = 4 if policy in ('observer4', 'raw4') else 1
    need(isinstance(losses, np.ndarray) and losses.dtype == np.float64 and losses.shape == (4,)
         and np.isfinite(losses).all() and np.all(losses >= 0)
         and np.all(losses[views:] == 0), 'loss schema')
    need(values['gradient_evaluations'] == views and values['adam_steps'] == index+1
         and values['observer_steps'] == (index+1 if policy in ('native1','observer4') else 0), 'step counters')
    stream['losses'][index] = losses
    for key in FLOAT_STREAMS:
        need(isinstance(values[key], (float, np.floating)) and np.isfinite(values[key])
             and values[key] >= 0, 'norm schema: '+key)
        stream[key][index] = values[key]
    for key in INT_STREAMS:
        need(type(values[key]) is int and values[key] >= 0, 'integer counter schema: '+key)
        stream[key][index] = values[key]
    need(values['basis_rank'] <= 32, 'basis rank cap')
    if policy in ('raw1', 'raw4'):
        need(values['observation_norm'] == 0 and values['basis_rank'] == 0, 'raw observer metadata')


def byte_inventory():
    parts = {'27_full_states_upper': 27*(37*50890*4 + 128*1024),
             '12_three_panel_logit_histories': 12*22*3*5000*10*4,
             'plans_streams_JSON_ZIP_allowance': 64*1024**2}
    total = sum(parts.values())
    need(total + base.RESERVE_BYTES < MAX_BYTES, 'artifact inventory exceeds cap')
    return {'component_upper_bytes': parts, 'total_upper_bytes': total, 'cap_bytes': MAX_BYTES}


def source_pins():
    names = list(OLD_PINS)
    names += [f'experiments/spectral_multiview{suffix}.py' for suffix in ('', '_core', '_data', '_audit')]
    names += [f'tests/test_spectral_multiview{suffix}.py' for suffix in ('', '_core', '_data', '_audit')]
    names += [str((DOCS/name).relative_to(ROOT)) for name in ('protocol.md', 'implementation-check.md')]
    pins = {name: digest(ROOT/name) for name in names}
    need(all(pins[name] == sha for name, sha in OLD_PINS.items()), 'frozen dependency changed')
    return pins


def configure():
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        need(os.environ.get(key) == '1', 'set '+key+'=1 before Python')
    need(os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8', 'deterministic workspace required')
    group = next(x.split(':', 2)[2] for x in Path('/proc/self/cgroup').read_text().splitlines() if x.startswith('0::'))
    need(Path(group).name == UNIT, 'unexpected acquisition unit')
    root = Path('/sys/fs/cgroup')/group.lstrip('/')
    effective = {k: (root/k).read_text().strip() for k in ('memory.max', 'memory.swap.max', 'cpu.max')}
    props = subprocess.check_output(['systemctl', '--user', 'show', UNIT, '--property=Type',
        '--property=RuntimeMaxUSec', '--property=Restart', '--property=KillMode'], text=True)
    service = dict(x.split('=', 1) for x in props.splitlines())
    base.validate_bounds(effective, service)
    clients = base.validate_gpu_clients(subprocess.check_output(['nvidia-smi',
        '--query-compute-apps=pid,used_gpu_memory', '--format=csv,noheader,nounits'], text=True))
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    need(torch.__version__ == '2.11.0+cu128' and np.__version__ == '1.26.4', 'framework versions')
    need(torch.cuda.is_available() and torch.cuda.get_device_name() == 'NVIDIA GeForce RTX 3090', 'local RTX3090 required')
    free, total = torch.cuda.mem_get_info()
    need(free >= 8*1024**3, '8GiB free GPU required')
    torch.cuda.set_per_process_memory_fraction(8*1024**3/total)
    return {'cgroup': group, 'effective': effective, 'service': service, 'preexisting_gpu_clients': clients}


def acquire(run):
    from experiments import spectral_multiview_data as data
    from experiments import spectral_multiview_core as helper
    images, labels = base.read_training()
    branches, initial_hashes, warmup_hashes, paired_predictions = [], {}, {}, {}
    for seed in SEEDS:
        plan = data.make_plan(labels, seed)
        run.save(f'plan-s{seed}.npz', plan, 'npz')
        run.save(f'plan-s{seed}.json', {'seed': seed, 'numpy_version': np.__version__,
            'array_hashes': {k: base.array_digest(v) for k, v in plan.items()}})
        cpu = images[plan['train_ids']].astype(np.float32)/np.float32(255)
        heldout_cpu = images[plan['eval_ids']].astype(np.float32)/np.float32(255)
        eval_inputs = {'train': torch.from_numpy(cpu).to(run.device),
                       'heldout': torch.from_numpy(heldout_cpu).to(run.device),
                       'heldout_translated': torch.from_numpy(data.translate(heldout_cpu, plan['eval_shifts'])).to(run.device)}
        for row in (r for r in branch_roster() if r['seed'] == seed):
            run.check()
            name, policy = f"s{seed}-{row['policy']}", row['policy']
            base.sync(run.device)
            started = time.monotonic()
            model = core.make_model(seed, run.device)
            optimizer = core.make_optimizer(model)
            tracker = core.make_tracker(model, optimizer) if policy in ('native1','observer4') else None
            need(sum(p.numel() for p in model.parameters()) == 50890 and not optimizer.state, 'fresh canonical model/Adam')
            model_sha = core.tree_digest(dict(model.state_dict()))
            if seed not in initial_hashes:
                initial_hashes[seed] = model_sha
                run.save(f'initial-s{seed}.pt', core.snapshot(model, optimizer, tracker), 'tensor')
            need(model_sha == initial_hashes[seed], 'paired initialization mismatch')
            metrics, logits, stream = [], {k: [] for k in PANELS}, empty_stream()
            training_seconds = augmentation_seconds = evaluation_seconds = 0.
            warmup_sha = warmup_receipt = None

            def evaluate(step):
                nonlocal evaluation_seconds
                base.sync(run.device)
                before = time.monotonic()
                record = {'step': step}
                for panel in PANELS:
                    values = base.predict(model, eval_inputs[panel])
                    if step == 0 or (step == 100 and policy != 'raw4'):
                        key = (seed, step, panel)
                        sha = base.array_digest(values)
                        if key in paired_predictions:
                            need(sha == paired_predictions[key], 'paired initial/warmup predictions differ')
                        paired_predictions[key] = sha
                    logits[panel].append(values)
                    target = plan['train_labels' if panel == 'train' else 'eval_labels']
                    record[panel] = base.classification_stats(values, target)
                metrics.append(record)
                base.sync(run.device)
                evaluation_seconds += time.monotonic()-before

            evaluate(0)
            for index, batch in enumerate(plan['occurrences']):
                run.check()
                base.sync(run.device)
                before = time.monotonic()
                source = cpu[batch]
                views = [data.translate(source, plan['shifts'][index])]
                if policy in ('observer4', 'raw4'):
                    views += [data.translate(source, shifts) for shifts in plan['extra_shifts'][index]]
                # All augmentation is CPU float32 before tensor transfer; timed separately.
                values = np.stack(views)
                augmentation_seconds += time.monotonic()-before
                before = time.monotonic()
                x = torch.from_numpy(values).to(run.device)
                y = torch.from_numpy(plan['train_labels'][batch]).to(run.device)
                record_step(stream, index, helper.training_update(model, optimizer, tracker, x, y, policy), policy)
                base.sync(run.device)
                training_seconds += time.monotonic()-before
                step = index+1
                if step == 100:
                    warmup_sha = base.learning_digest(model, optimizer)
                    if policy != 'raw4':
                        if seed in warmup_hashes:
                            need(warmup_sha == warmup_hashes[seed], 'first-view warmup model/Adam mismatch')
                        warmup_hashes[seed] = warmup_sha
                    warmup_receipt = run.save('warmup-'+name+'.pt', core.snapshot(model, optimizer, tracker), 'tensor')
                if step in EVAL_STEPS:
                    evaluate(step)
            need(all(bool(torch.isfinite(p).all()) for p in model.parameters()), 'nonfinite endpoint')
            gradient_count = int(stream['gradient_evaluations'].sum())
            need(gradient_count == STEPS*(4 if policy in ('observer4','raw4') else 1), 'gradient evaluation count')
            final_receipt = run.save('final-'+name+'.pt', core.snapshot(model, optimizer, tracker), 'tensor')
            logits_receipt = run.save('logits-'+name+'.npz', {'steps': np.array(EVAL_STEPS,dtype=np.int64),
                **{k: np.stack(v) for k,v in logits.items()}}, 'npz')
            stream_receipt = run.save('stream-'+name+'.npz', stream, 'npz')
            branch = {**row, 'name': name, 'initial_model_sha256': model_sha,
                'warmup_learning_sha256': warmup_sha, 'warmup_receipt': warmup_receipt,
                'final_receipt': final_receipt, 'logits_receipt': logits_receipt, 'stream_receipt': stream_receipt,
                'metrics': metrics, 'gradient_evaluation_count': gradient_count,
                'training_seconds': training_seconds, 'augmentation_seconds': augmentation_seconds,
                'evaluation_seconds': evaluation_seconds, 'wall_seconds': time.monotonic()-started}
            branches.append(branch)
            print(json.dumps({'completed': name, 'count': len(branches), 'of': 12,
                              'heldout': metrics[-1]['heldout'], 'gradient_evaluation_count': gradient_count}), flush=True)
            del model, optimizer, tracker, logits, stream
    need(sum(b['gradient_evaluation_count'] for b in branches) == 120000, 'total gradient count')
    return branches


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    need(args.execute, 'No acquisition without explicit --execute')
    parent = args.output_dir.parent.resolve(strict=True)
    need(parent.is_relative_to(Path('/tmp/spectral-experiment-artifacts')) and not list(parent.iterdir()), 'unused large-volume parent required')
    need(args.output_dir == parent/'acquisition-001' and not args.output_dir.exists() and not args.output_dir.is_symlink(), 'exclusive acquisition-001 required')
    mount = subprocess.check_output(['findmnt', '-n', '-o', 'TARGET,SOURCE', '--target', str(parent)], text=True).split()
    need(mount == ['/private-artifacts/storage', '/dev/RECONFIGURE_FOR_LOCAL_STORAGE'] and shutil.disk_usage(parent).free > MAX_BYTES+1024**3, 'mount/free-space identity')
    pins, inventory = source_pins(), byte_inventory()
    commit = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    for path, sha in pins.items():
        need(hashlib.sha256(subprocess.check_output(['git', '-C', str(ROOT), 'show', commit+':'+path])).hexdigest() == sha, 'uncommitted source: '+path)
    attempt = {'schema': SCHEMA, 'commit': commit, 'source_pins': pins, 'data_pins': base.DATA_PINS,
               'unit': UNIT, 'output_dir': str(args.output_dir), 'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    with (DOCS/'attempt.json').open('x') as f:
        json.dump(attempt, f, indent=2)
    args.output_dir.mkdir()
    run = base.Run(args.output_dir)
    try:
        guards = configure()
        run.save('provenance.json', {**attempt, 'guards': guards, 'inventory': inventory,
                 'python': sys.version, 'torch': torch.__version__, 'numpy': np.__version__})
        branches = acquire(run)
        need(source_pins() == pins and {n: digest(base.DATA/n) for n in base.DATA_PINS} == base.DATA_PINS, 'source/data changed')
        result = {'schema': SCHEMA, 'status': 'complete', 'source_pins': pins, 'data_pins': base.DATA_PINS,
                  'roster': branch_roster(), 'eval_steps': EVAL_STEPS, 'branches': branches,
                  'gradient_evaluation_count': sum(b['gradient_evaluation_count'] for b in branches),
                  'receipts': list(run.receipts), 'seconds': time.monotonic()-run.started,
                  'gpu_max_allocated_bytes': torch.cuda.max_memory_allocated(),
                  'host_high_water_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
        receipt = run.save('results.json', result)
        print(json.dumps({'status': 'complete', 'result': receipt, 'bytes': run.used}), flush=True)
    except BaseException as exc:
        with (args.output_dir/'failure.json').open('x') as f:
            json.dump({'status': 'failed', 'error_type': type(exc).__name__, 'message': str(exc)[:2000]}, f)
        raise


if __name__ == '__main__':
    main()
