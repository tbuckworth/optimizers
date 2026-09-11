#!/usr/bin/env python3
"""One fixed clean-data augmentation factorial. Import is inert; no auto-retry."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import resource
import shutil
import struct
import subprocess
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
I9 = ROOT / 'output/2026-09-06-spectral-optimizer-investigation/continuation/iteration-009'
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(I9))
import neural_core as core
from experiments import spectral_general_augmentation_data as data

DOCS = ROOT / 'output/2026-09-10-spectral-general-augmentation'
DATA = Path('data/MNIST/raw')
DATA_PINS = {
    'train-images-idx3-ubyte': 'ba891046e6505d7aadcbbe25680a0738ad16aec93bde7f9b65e87a2fc25776db',
    'train-labels-idx1-ubyte': '65a50cbbf4e906d70832878ad85ccda5333a97f0f4c3dd2ef09a8a9eef7101c5',
}
CORE_SHA = '706851828b7120bea0d7a41bd0fa75474ac800b2df71087941d0140c8d1970ec'
SCHEMA = 'spectral_general_augmentation_v1'
UNIT = 'spectral-general-augmentation-001.service'
SEEDS, STEPS, BATCH = data.SEEDS, data.STEPS, data.BATCH
POLICIES, AUGMENTATIONS = ('raw', 'native32'), ('none', 'translate')
EVAL_STEPS = (0, 100) + tuple(range(200, 4001, 200))
MAX_BYTES, RESERVE_BYTES, DEADLINE_SECONDS = 1024**3, 1024**2, 900
GPU_CLIENT_MAX_MIB = {2101: 512, 8861: 128}


def need(condition, message):
    if not condition:
        raise RuntimeError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024**2), b''):
            h.update(block)
    return h.hexdigest()


def array_digest(value):
    value = np.ascontiguousarray(value)
    head = json.dumps([value.dtype.str, list(value.shape)], separators=(',', ':'))
    return hashlib.sha256(head.encode() + b'\n' + value.tobytes()).hexdigest()


def branch_roster():
    rows = []
    for si, seed in enumerate(SEEDS):
        for ai, augmentation in enumerate(AUGMENTATIONS):
            offset = (si + ai) % 2
            for policy in POLICIES[offset:] + POLICIES[:offset]:
                rows.append({'seed': seed, 'policy': policy, 'augmentation': augmentation})
    return rows


def source_pins():
    paths = [ROOT / 'spectral_filter.py', Path(core.__file__), Path(__file__), Path(data.__file__),
             DOCS / 'protocol.md', DOCS / 'implementation-check.md', DOCS / 'visual-review.md',
             DOCS / 'preview.json', *(DOCS / f'preview-{i}.png' for i in (1, 2, 3)),
             ROOT / 'experiments/spectral_general_augmentation_audit.py',
             ROOT / 'experiments/spectral_general_augmentation_preview.py',
             ROOT / 'tests/test_spectral_general_augmentation.py',
             ROOT / 'tests/test_spectral_general_augmentation_data.py',
             ROOT / 'tests/test_spectral_general_augmentation_audit.py']
    pins = {str(p.resolve().relative_to(ROOT)): digest(p) for p in paths}
    need(pins['spectral_filter.py'] == core.FILTER_SHA256, 'canonical filter changed')
    need(pins[str(Path(core.__file__).resolve().relative_to(ROOT))] == CORE_SHA, 'accepted I9 core changed')
    return pins


def byte_inventory():
    parts = {'27_full_states_upper': 27 * (37 * 50890 * 4 + 128 * 1024),
             '12_logit_histories': 12 * 22 * 2 * 5000 * 10 * 4,
             'plans_streams_metadata_receipts_allowance': 64 * 1024**2}
    total = sum(parts.values())
    need(total + RESERVE_BYTES < MAX_BYTES, 'artifact inventory exceeds cap')
    return {'component_upper_bytes': parts, 'total_upper_bytes': total, 'cap_bytes': MAX_BYTES}


class CappedWriter:
    def __init__(self, handle, allowance, check=lambda: None):
        self.handle, self.allowance, self.check = handle, allowance, check

    def write(self, value):
        self.check()
        size = memoryview(value).nbytes
        need(size <= self.allowance, 'artifact byte cap exhausted')
        count = self.handle.write(value)
        need(count == size, 'short write')
        self.allowance -= count
        return count

    def flush(self):
        self.handle.flush()

    def read(self, *args):
        # NumPy's zipfile_factory recognizes file-like objects by this attribute.
        return self.handle.read(*args)


class Run:
    def __init__(self, path, device='cuda'):
        self.path, self.device = Path(path), device
        self.started, self.last_check = time.monotonic(), 0.
        self.used, self.receipts = 0, []

    def check(self):
        now = time.monotonic()
        need(now - self.started < DEADLINE_SECONDS, '15-minute cooperative deadline exceeded')
        if now - self.last_check > 1:
            need(shutil.disk_usage(self.path).free > 1024**3, 'less than 1 GiB free disk')
            need(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 <= 16 * 1024**3,
                 'host memory guard exceeded')
            if self.device == 'cuda':
                need(torch.cuda.max_memory_allocated() <= 8 * 1024**3, 'GPU memory guard exceeded')
            self.last_check = now

    def save(self, name, value, kind='json'):
        self.check()
        need(Path(name).name == name and name not in ('', '.', '..'), 'direct child required')
        need(kind in ('json', 'npz', 'tensor'), 'unknown artifact type')
        target = self.path / name
        try:
            with target.open('xb') as handle:
                writer = CappedWriter(handle, MAX_BYTES - self.used - RESERVE_BYTES, self.check)
                if kind == 'npz':
                    np.savez(writer, **value)
                elif kind == 'tensor':
                    torch.save(value, writer)
                else:
                    writer.write((json.dumps(value, allow_nan=False, separators=(',', ':')) + '\n').encode())
        except BaseException:
            self.used = sum(p.stat().st_size for p in self.path.iterdir() if p.is_file())
            raise
        receipt = {'path': name, 'size_bytes': target.stat().st_size, 'sha256': digest(target)}
        self.used += receipt['size_bytes']
        self.receipts.append(receipt)
        return receipt


def read_training():
    for name, sha in DATA_PINS.items():
        need(digest(DATA / name) == sha, 'IDX source hash mismatch: ' + name)
    raw = (DATA / 'train-images-idx3-ubyte').read_bytes()
    need(len(raw) == 16 + 60000 * 784 and struct.unpack('>IIII', raw[:16]) == (2051, 60000, 28, 28),
         'image IDX schema')
    images = np.frombuffer(raw, dtype=np.uint8, offset=16).reshape(60000, 784).copy()
    raw = (DATA / 'train-labels-idx1-ubyte').read_bytes()
    need(len(raw) == 60008 and struct.unpack('>II', raw[:8]) == (2049, 60000), 'label IDX schema')
    return images, np.frombuffer(raw, dtype=np.uint8, offset=8).astype(np.int64)


def classification_stats(logits, labels):
    values = torch.as_tensor(logits, dtype=torch.float64, device='cpu')
    target = torch.as_tensor(labels, dtype=torch.int64, device='cpu')
    need(values.shape == (len(target), 10) and bool(torch.isfinite(values).all()), 'finite logits required')
    correct = int((values.argmax(1) == target).sum())
    ce_sum = float(F.cross_entropy(values, target, reduction='sum'))
    return {'count': len(target), 'correct': correct, 'accuracy': correct / len(target),
            'ce_sum': ce_sum, 'ce': ce_sum / len(target)}


@torch.no_grad()
def predict(model, inputs):
    # The frozen MLP has no train/eval-dependent modules; do not mutate its modes.
    values = torch.cat([model(x).detach().cpu() for x in inputs.split(500)]).numpy()
    need(values.dtype == np.float32 and np.isfinite(values).all(), 'prediction schema')
    return values


def sync(device):
    if device == 'cuda':
        torch.cuda.synchronize()


def learning_digest(model, optimizer):
    return core.tree_digest({'model': dict(model.state_dict()), 'optimizer': optimizer.state_dict()})


def training_update(model, optimizer, tracker, x, targets, policy):
    need(policy in POLICIES and ((policy == 'raw') == (tracker is None)), 'tracker/policy mismatch')
    optimizer.zero_grad(set_to_none=True)
    loss = F.cross_entropy(model(x), targets)
    need(bool(torch.isfinite(loss)), 'nonfinite training loss')
    loss.backward()
    diagnostic = tracker.filter_grad() if tracker is not None else None
    optimizer.step()
    return float(loss.detach()), diagnostic


def acquire(run):
    images, labels = read_training()
    branches, identities, warmups = [], {}, {}
    roster = branch_roster()
    for seed in SEEDS:
        plan = data.make_plan(labels, seed)
        plan['train_labels'], plan['eval_labels'] = labels[plan['train_ids']], labels[plan['eval_ids']]
        run.save(f'plan-s{seed}.npz', plan, 'npz')
        run.save(f'plan-s{seed}.json', {'seed': seed, 'array_hashes': {k: array_digest(v) for k, v in plan.items()},
                                     'numpy_version': np.__version__})
        cpu = images[plan['train_ids']].astype(np.float32) / np.float32(255)
        train_x = torch.from_numpy(cpu).to(run.device)
        eval_x = torch.from_numpy(images[plan['eval_ids']].astype(np.float32) / np.float32(255)).to(run.device)
        for row in (r for r in roster if r['seed'] == seed):
            run.check()
            name = f"s{seed}-{row['policy']}-{row['augmentation']}"
            sync(run.device)
            started = time.monotonic()
            model = core.make_model(seed, run.device)
            optimizer = core.make_optimizer(model)
            tracker = core.make_tracker(model, optimizer) if row['policy'] == 'native32' else None
            need(sum(p.numel() for p in model.parameters()) == 50890, 'model parameter count')
            model_sha = core.tree_digest(dict(model.state_dict()))
            if seed not in identities:
                identities[seed] = model_sha
                run.save(f'initial-s{seed}.pt', core.snapshot(model, optimizer, tracker), 'tensor')
            need(model_sha == identities[seed] and len(optimizer.state) == 0, 'fresh paired initialization required')
            metrics, train_logits, heldout_logits = [], [], []
            loss_history = np.empty(STEPS, dtype=np.float64)
            training_seconds = augmentation_seconds = evaluation_seconds = 0.
            warmup_sha, warmup_receipt = None, None

            def evaluate(step):
                nonlocal evaluation_seconds
                sync(run.device)
                before = time.monotonic()
                tr, ev = predict(model, train_x), predict(model, eval_x)
                metrics.append({'step': step, 'train': classification_stats(tr, plan['train_labels']),
                                'heldout': classification_stats(ev, plan['eval_labels'])})
                train_logits.append(tr)
                heldout_logits.append(ev)
                sync(run.device)
                evaluation_seconds += time.monotonic() - before

            evaluate(0)
            for step, batch in enumerate(plan['occurrences'], 1):
                run.check()
                sync(run.device)
                before = time.monotonic()
                values = cpu[batch]
                if row['augmentation'] == 'translate':
                    aug_before = time.monotonic()
                    values = data.translate(values, plan['shifts'][step - 1])
                    augmentation_seconds += time.monotonic() - aug_before
                x = torch.from_numpy(values).to(run.device)
                targets = torch.as_tensor(plan['train_labels'][batch], device=run.device)
                loss_history[step - 1], diagnostic = training_update(model, optimizer, tracker, x, targets, row['policy'])
                sync(run.device)
                training_seconds += time.monotonic() - before
                if step == 100:
                    warmup_sha = learning_digest(model, optimizer)
                    key = (seed, row['augmentation'])
                    if key in warmups:
                        need(warmup_sha == warmups[key], 'within-condition warmup learning state differs')
                    warmups[key] = warmup_sha
                    warmup_receipt = run.save('warmup-' + name + '.pt', core.snapshot(model, optimizer, tracker), 'tensor')
                if step in EVAL_STEPS:
                    evaluate(step)
            need(all(bool(torch.isfinite(p).all()) for p in model.parameters()), 'nonfinite final parameters')
            final_receipt = run.save('final-' + name + '.pt', core.snapshot(model, optimizer, tracker), 'tensor')
            logits_receipt = run.save('logits-' + name + '.npz', {'steps': np.array(EVAL_STEPS, dtype=np.int64),
                'train': np.stack(train_logits), 'heldout': np.stack(heldout_logits)}, 'npz')
            stream_receipt = run.save('stream-' + name + '.npz', {'loss': loss_history}, 'npz')
            rank_summary = None if tracker is None else {
                'basis_rank': 0 if tracker.S is None else int(tracker.S.numel()),
                'stabilization_count': tracker.stabilization_count,
                'max_orthogonality_error': tracker.max_orthogonality_error,
                'step_count': tracker.step_count}
            branch = {**row, 'name': name, 'initial_model_sha256': model_sha,
                'warmup_learning_sha256': warmup_sha, 'warmup_receipt': warmup_receipt,
                'final_receipt': final_receipt, 'logits_receipt': logits_receipt, 'stream_receipt': stream_receipt,
                'metrics': metrics, 'training_seconds': training_seconds, 'augmentation_seconds': augmentation_seconds,
                'evaluation_seconds': evaluation_seconds, 'wall_seconds': time.monotonic() - started,
                'rank_summary': rank_summary}
            branches.append(branch)
            print(json.dumps({'completed': name, 'of': len(roster), 'count': len(branches),
                              'heldout': metrics[-1]['heldout'], 'seconds': branch['wall_seconds']}), flush=True)
            del model, optimizer, tracker
    return branches


def validate_gpu_clients(output):
    rows = []
    for line in output.splitlines():
        if not line.strip():
            continue
        pid, mib = [int(x.strip()) for x in line.split(',')]
        need(pid in GPU_CLIENT_MAX_MIB and 0 <= mib <= GPU_CLIENT_MAX_MIB[pid], 'unapproved GPU client')
        rows.append({'pid': pid, 'memory_mib': mib})
    return rows


def validate_bounds(effective, service):
    quota, period = effective['cpu.max'].split()
    need(effective['memory.max'] == str(16 * 1024**3) and effective['memory.swap.max'] == '0'
         and quota != 'max' and int(quota) == int(period) and int(period) > 0, 'unexpected cgroup bounds')
    need(service == {'Type': 'exec', 'RuntimeMaxUSec': '20min', 'Restart': 'no', 'KillMode': 'control-group'},
         'unexpected service bounds')


def configure():
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        need(os.environ.get(name) == '1', 'set ' + name + '=1 before Python')
    need(os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8', 'deterministic workspace required')
    group = next(line.split(':', 2)[2] for line in Path('/proc/self/cgroup').read_text().splitlines()
                 if line.startswith('0::'))
    need(Path(group).name == UNIT, 'unexpected acquisition unit')
    root = Path('/sys/fs/cgroup') / group.lstrip('/')
    effective = {k: (root / k).read_text().strip() for k in ('memory.max', 'memory.swap.max', 'cpu.max')}
    props = subprocess.check_output(['systemctl', '--user', 'show', UNIT, '--property=Type',
        '--property=RuntimeMaxUSec', '--property=Restart', '--property=KillMode'], text=True)
    service = dict(line.split('=', 1) for line in props.splitlines())
    validate_bounds(effective, service)
    clients = validate_gpu_clients(subprocess.check_output(['nvidia-smi',
        '--query-compute-apps=pid,used_gpu_memory', '--format=csv,noheader,nounits'], text=True))
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    need(torch.__version__ == '2.11.0+cu128' and np.__version__ == '1.26.4', 'unexpected framework version')
    need(torch.cuda.is_available() and torch.cuda.get_device_name() == 'NVIDIA GeForce RTX 3090', 'local RTX3090 required')
    free, total = torch.cuda.mem_get_info()
    need(free >= 8 * 1024**3, '8 GiB free GPU required')
    torch.cuda.set_per_process_memory_fraction(8 * 1024**3 / total)
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    return {'cgroup': group, 'effective': effective, 'service': service, 'preexisting_gpu_clients': clients}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--output-dir', type=Path, required=True)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    need(args.execute, 'No acquisition without explicit --execute')
    parent = args.output_dir.parent.resolve(strict=True)
    need(parent.is_relative_to(Path('/tmp/spectral-experiment-artifacts')) and not list(parent.iterdir()), 'unused large-volume parent required')
    need(args.output_dir == parent / 'acquisition-001' and not args.output_dir.exists()
         and not args.output_dir.is_symlink(), 'exclusive acquisition-001 required')
    mount = subprocess.check_output(['findmnt', '-n', '-o', 'TARGET,SOURCE', '--target', str(parent)], text=True).split()
    need(mount == ['/private-artifacts/storage', '/dev/RECONFIGURE_FOR_LOCAL_STORAGE'], 'wrong output mount identity')
    need(shutil.disk_usage(parent).free > MAX_BYTES + 1024**3, 'cap plus free-disk reserve required')
    pins, inventory = source_pins(), byte_inventory()
    commit = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    for path, sha in pins.items():
        blob = subprocess.check_output(['git', '-C', str(ROOT), 'show', commit + ':' + path])
        need(hashlib.sha256(blob).hexdigest() == sha, 'uncommitted source: ' + path)
    need('PASS' in (DOCS / 'visual-review.md').read_text(), 'visual review not passed')
    attempt = {'schema': SCHEMA, 'commit': commit, 'source_pins': pins, 'data_pins': DATA_PINS,
               'output_dir': str(args.output_dir), 'unit': UNIT, 'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    with (DOCS / 'attempt.json').open('x') as handle:
        json.dump(attempt, handle, indent=2)
        handle.write('\n')
    args.output_dir.mkdir()
    run = Run(args.output_dir)
    try:
        guards = configure()
        run.save('provenance.json', {**attempt, 'guards': guards, 'inventory': inventory,
            'python': sys.version, 'torch': torch.__version__, 'numpy': np.__version__})
        branches = acquire(run)
        need(source_pins() == pins and {n: digest(DATA / n) for n in DATA_PINS} == DATA_PINS,
             'source or data changed during acquisition')
        result = {'schema': SCHEMA, 'status': 'complete', 'source_pins': pins, 'data_pins': DATA_PINS,
                  'roster': branch_roster(), 'eval_steps': EVAL_STEPS, 'branches': branches,
                  'seconds': time.monotonic() - run.started, 'receipts': list(run.receipts),
                  'gpu_max_allocated_bytes': torch.cuda.max_memory_allocated(),
                  'host_high_water_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
        receipt = run.save('results.json', result)
        print(json.dumps({'status': 'complete', 'result': receipt, 'bytes': run.used}), flush=True)
    except BaseException as exc:
        # No retry. This fixed-size footer uses the separately reserved MiB.
        failure = {'status': 'failed', 'error_type': type(exc).__name__, 'message': str(exc)[:2000],
                   'seconds': time.monotonic() - run.started}
        with (args.output_dir / 'failure.json').open('x') as handle:
            json.dump(failure, handle)
        raise


if __name__ == '__main__':
    main()
