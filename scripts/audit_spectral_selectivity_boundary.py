#!/usr/bin/env python3
"""Independent, saved-data-only selectivity audit; inert unless --execute.

No producer imports, models, inference, differentiation, optimizer execution or
covariance-stream replay. NumPy does arithmetic; torch only decodes CPU tensors.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import stat
import struct
import subprocess
import time
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DOCS = 'output/2026-09-09-spectral-selectivity-boundary'
DATA = Path('data/MNIST/raw')
DATA_PINS = {
    'train-images-idx3-ubyte': 'ba891046e6505d7aadcbbe25680a0738ad16aec93bde7f9b65e87a2fc25776db',
    'train-labels-idx1-ubyte': '65a50cbbf4e906d70832878ad85ccda5333a97f0f4c3dd2ef09a8a9eef7101c5',
}
SOURCE_NAMES = {'spectral_filter.py', 'experiments/spectral_selectivity_boundary.py',
                'tests/test_spectral_selectivity_boundary.py', DOCS + '/protocol.md',
                'output/2026-09-06-spectral-optimizer-investigation/continuation/iteration-009/neural_core.py'}
SEEDS = (202609111, 202609112, 202609113)
CELLS = ('clean', 'diffuse', 'shared', 'sham')
POLICIES = ('raw', 'native32', 'norm_raw')
EVAL = tuple(range(0, 2001, 100))
ANCHORS = (101, 500, 2000)
MAJORITY = (0, 1, 2, 3, 4, 5, 6, 7, 9)
SCHEMA = 'spectral_selectivity_boundary_v1'
MODEL_SHAPES = {'0.weight': (64, 784), '0.bias': (64,), '2.weight': (10, 64), '2.bias': (10,)}
EPS32 = float(np.finfo(np.float32).eps)
TOLERANCES = {'metric_atol': 1e-10, 'metric_rtol': 1e-10,
              'geometry_atol': 1e-7, 'geometry_rtol': 5e-4,
              'action_l2_relative_to_raw': 3e-5, 'action_l2_atol': 1e-12,
              'adam_roundoff_epsilon_multiplier': 16,
              'span_atol': 1e-10, 'span_rtol': 1e-8}
UNIT = 'spectral-selectivity-boundary-audit-001.service'


class AuditError(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise AuditError(message)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def array_hash(value):
    value = np.ascontiguousarray(value)
    header = json.dumps([value.dtype.str, list(value.shape)], separators=(',', ':')).encode()
    return hashlib.sha256(header + b'\n' + value.tobytes()).hexdigest()


def read_json(path):
    require(Path(path).stat().st_size <= 64 * 1024**2, 'oversized JSON')
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'duplicate JSON key: ' + key)
            result[key] = value
        return result
    def bad_constant(value):
        raise AuditError('nonfinite JSON constant: ' + value)
    return json.loads(Path(path).read_text(), object_pairs_hook=pairs, parse_constant=bad_constant)


def contained_file(root, name):
    require(isinstance(name, str), 'artifact path must be a string')
    relative = Path(name)
    require(isinstance(name, str) and relative.name == name and name not in ('', '.', '..'),
            'artifact path is not a direct child')
    path = Path(root) / name
    require(stat.S_ISREG(path.lstat().st_mode) and path.resolve().parent == Path(root).resolve(),
            'artifact is not a contained regular file: ' + name)
    return path


def verify_receipt(root, item):
    require(type(item) is dict and set(item) == {'path', 'size_bytes', 'sha256'}, 'receipt schema')
    require(type(item['size_bytes']) is int and 0 <= item['size_bytes'] <= 256 * 1024**2, 'receipt size')
    require(type(item['sha256']) is str and len(item['sha256']) == 64, 'receipt hash')
    path = contained_file(root, item['path'])
    require(path.stat().st_size == item['size_bytes'] and sha256(path) == item['sha256'],
            'receipt mismatch: ' + item['path'])
    return path


def load_npz(path):
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        require(len(members) <= 32 and len({m.filename for m in members}) == len(members), 'NPZ member roster')
        require(all(m.filename.endswith('.npy') and Path(m.filename).name == m.filename for m in members), 'NPZ member escape')
        require(sum(m.file_size for m in members) <= 128 * 1024**2, 'expanded NPZ cap')
    with np.load(path, allow_pickle=False, max_header_size=10000) as archive:
        result = {key: archive[key] for key in archive.files}
    for key, value in result.items():
        require(value.dtype.kind in 'biuf' and np.isfinite(value).all(), 'finite numeric NPZ: ' + key)
    return result


def cpu_tensor_file(path):
    import torch
    with zipfile.ZipFile(path) as archive:
        require(sum(m.file_size for m in archive.infolist()) <= 128 * 1024**2, 'expanded tensor cap')
    return torch.load(path, weights_only=True, map_location='cpu')


def arr(value, dtype=None, shape=None):
    if hasattr(value, 'detach'):
        require(str(value.device) == 'cpu' and not value.requires_grad, 'CPU detached saved tensor required')
        value = value.numpy()
    value = np.asarray(value)
    require(value.dtype.kind in 'biuf' and np.isfinite(value).all(), 'finite array required')
    if dtype is not None:
        require(value.dtype == np.dtype(dtype), 'array dtype mismatch')
    if shape is not None:
        require(value.shape == shape, 'array shape mismatch')
    return value


def tree_hash(value):
    """Independent implementation of the documented i9_neural_tree_v1 encoding."""
    def node(v):
        if hasattr(v, 'detach'):
            a = arr(v)
            return ['tensor', str(v.dtype), list(a.shape), hashlib.sha256(a.tobytes(order='C')).hexdigest()]
        if type(v) is dict:
            return ['dict', [[node(k), node(x)] for k, x in v.items()]]
        if type(v) in (tuple, list):
            return [type(v).__name__, [node(x) for x in v]]
        if v is None or type(v) in (bool, int, str):
            return [type(v).__name__, v]
        if type(v) is float:
            require(math.isfinite(v), 'nonfinite tree float')
            return ['float', v.hex()]
        raise AuditError('unsupported saved-tree type: ' + str(type(v)))
    encoded = json.dumps(node(value), ensure_ascii=True, allow_nan=False, separators=(',', ':')).encode('ascii')
    return hashlib.sha256(b'i9_neural_tree_v1\n' + encoded).hexdigest()


class Checks:
    def __init__(self):
        self.started = time.monotonic()
        self.count = 0
        self.max_absolute = {}

    def tick(self):
        require(time.monotonic() - self.started < 250, '250-second cooperative audit deadline')

    def equal(self, observed, expected, label, atol=1e-10, rtol=1e-10):
        self.tick()
        self.count += 1
        if isinstance(expected, dict):
            require(type(observed) is dict and set(observed) == set(expected), label + ': dictionary keys')
            for key in expected:
                self.equal(observed[key], expected[key], label + '/' + str(key), atol, rtol)
        elif isinstance(expected, (list, tuple)):
            require(isinstance(observed, (list, tuple)) and len(observed) == len(expected), label + ': length')
            for i, (left, right) in enumerate(zip(observed, expected)):
                self.equal(left, right, label + '/' + str(i), atol, rtol)
        elif isinstance(expected, bool):
            require(type(observed) is bool and observed == expected, label + ': exact Boolean')
        elif expected is None or isinstance(expected, (str, int, np.integer)):
            require(observed == expected and not isinstance(observed, bool), label + ': exact value')
        else:
            require(isinstance(observed, (float, int)) and math.isfinite(observed), label + ': finite scalar')
            error = abs(float(observed) - float(expected))
            category = label.split('/')[0]
            self.max_absolute[category] = max(error, self.max_absolute.get(category, 0.))
            require(error <= atol + rtol * abs(expected), label + ': numerical mismatch')

    def array(self, observed, expected, label):
        observed = arr(observed)
        require(observed.dtype == expected.dtype and observed.shape == expected.shape
                and np.array_equal(observed, expected), label + ': exact array mismatch')
        self.count += 1


def rebuild_plan(labels, seed, majority_count=550, rare_count=50, heldout_count=500,
                 poison_count=500, warmup_steps=100, branch_steps=1900, batch=64):
    """Small dimensions are usable by fabricated fixtures, never a CLI option."""
    rng = {s: np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, s]))) for s in range(6)}
    train, heldout = [], []
    for digit in range(10):
        order = rng[0].permutation(np.where(labels == digit)[0])
        n = rare_count if digit == 8 else majority_count
        require(len(order) >= n + heldout_count, 'insufficient label population')
        train += order[:n].tolist()
        heldout += order[n:n + heldout_count].tolist()
    train, heldout = np.asarray(train, dtype=np.int64), np.asarray(heldout, dtype=np.int64)
    true = labels[train].astype(np.int64)
    majority_positions = np.where(true != 8)[0]
    warm = rng[1].choice(majority_positions, size=(warmup_steps, batch), replace=True)
    continuation = rng[2].integers(0, len(train), size=(branch_steps, batch))
    selected = np.zeros(len(train), dtype=bool)
    selected[majority_positions] = rng[3].random(len(majority_positions)) < .9
    replacement = np.full(len(train), -1, dtype=np.int64)
    replacement[majority_positions] = rng[3].choice(np.asarray(MAJORITY), size=len(majority_positions), replace=True)
    diffuse = true.copy()
    diffuse[selected] = replacement[selected]
    poison = np.zeros(len(train), dtype=bool)
    poison[rng[4].choice(np.where((true != 0) & (true != 8))[0], size=poison_count, replace=False)] = True
    wrong = np.where(poison, 0, true).astype(np.int64)
    sham, allocations = np.zeros(len(train), dtype=bool), []
    for digit in (1, 2, 3, 4, 5, 6, 7, 9):
        pools = [np.where((true == digit) & (poison == bool(z)))[0] for z in (0, 1)]
        populations = [len(pool) for pool in pools]
        k, n = populations[1], sum(populations)
        quotas = [k * size // n for size in populations]
        priority = sorted((0, 1), key=lambda z: (-(k * populations[z] % n), z))
        for z in priority[:k - sum(quotas)]:
            quotas[z] += 1
        for z, pool in enumerate(pools):
            if quotas[z] > 0:
                sham[rng[5].choice(pool, size=quotas[z], replace=False)] = True
            ideal = k * populations[z] / n
            allocations.append({'true_digit': digit, 'poison': z, 'population': populations[z],
                'shared_patch_count_for_digit': k, 'sham_patch_count': quotas[z],
                'ideal_count': ideal, 'rounding_deviation': quotas[z] - ideal})
    return {'train_ids': train, 'heldout_ids': heldout, 'train_true': true,
        'heldout_true': labels[heldout].astype(np.int64), 'warmup_batches': warm,
        'continuation_batches': continuation, 'diffuse_selected': selected,
        'diffuse_replacement_labels': replacement, 'diffuse_targets': diffuse,
        'poison_mask': poison, 'shared_sham_targets': wrong, 'sham_patch_mask': sham}, allocations


def cell_data(plan, cell):
    true = plan['train_true']
    target = {'clean': true, 'diffuse': plan['diffuse_targets'],
              'shared': plan['shared_sham_targets'], 'sham': plan['shared_sham_targets']}[cell]
    patch = {'clean': np.zeros(len(true), bool), 'diffuse': np.zeros(len(true), bool),
             'shared': plan['poison_mask'], 'sham': plan['sham_patch_mask']}[cell]
    cube = np.zeros((2, 2, 10, 10), np.int64)
    for a, b, c, d in zip(patch, plan['poison_mask'], true, target):
        cube[int(a), int(b), int(c), int(d)] += 1
    return target, patch, {'contingency_axes': ['patch', 'shared_poison_id', 'true_label', 'assigned_label'],
        'contingency_counts': cube.tolist(), 'patch_count': int(patch.sum()),
        'patch_poison_overlap': int(np.logical_and(patch, plan['poison_mask']).sum()),
        'actually_changed_count': int(np.count_nonzero(target != true)),
        'diffuse_selected_count': int(plan['diffuse_selected'].sum()) if cell == 'diffuse' else 0}


def patched_images(images, mask=None):
    result = images.copy().reshape(-1, 28, 28)
    result[np.ones(len(images), bool) if mask is None else mask, :3, :3] = np.float32(1)
    return result.reshape(-1, 784)


def classification(logits, labels, mask=None):
    x, y = np.asarray(logits, np.float64), np.asarray(labels)
    require(x.shape == (len(y), 10) and np.isfinite(x).all(), 'logit shape/finiteness')
    require(y.dtype.kind in 'iu' and ((y >= 0) & (y < 10)).all(), 'target domain')
    if mask is not None:
        x, y = x[mask], y[mask]
    n = len(y)
    if n == 0:
        return {'count': 0, 'correct': 0, 'ce_sum': 0., 'accuracy': None, 'ce': None}
    shifted = x - x.max(axis=1, keepdims=True)
    ce_sum = float(np.sum(np.log(np.exp(shifted).sum(axis=1)) - shifted[np.arange(n), y]))
    correct = int(np.count_nonzero(x.argmax(axis=1) == y))
    return {'count': n, 'correct': correct, 'ce_sum': ce_sum, 'accuracy': correct / n, 'ce': ce_sum / n}


def grouped(logits, labels):
    per = {str(d): classification(logits, labels, labels == d) for d in range(10)}
    def macro(digits):
        require(all(per[str(d)]['count'] > 0 for d in digits), 'empty class for macro mean')
        return {m: math.fsum(per[str(d)][m] for d in digits) / len(digits) for m in ('accuracy', 'ce')}
    return {'per_class': per, 'rare': per['8'], 'majority_macro': macro(MAJORITY),
            'balanced_total': macro(tuple(range(10))), 'micro': classification(logits, labels)}


def metrics(step, train, unpatched, patched, true, assigned, heldout_true):
    cue = {}
    old, new = unpatched.argmax(axis=1), patched.argmax(axis=1)
    for name, mask in {'majority_nonzero': (heldout_true != 0) & (heldout_true != 8),
                       'rare': heldout_true == 8, 'all_nonzero': heldout_true != 0}.items():
        n = int(mask.sum())
        a, b = int(np.count_nonzero(old[mask] == 0)), int(np.count_nonzero(new[mask] == 0))
        cue[name] = {'count': n, 'unpatched_target0_count': a, 'patched_target0_count': b,
                     'unpatched_target0_rate': a / n if n else None,
                     'patched_target0_rate': b / n if n else None, 'patch_excess': (b - a) / n if n else None}
    return {'step': step, 'train_true': grouped(train, true), 'train_assigned': classification(train, assigned),
            'train_actually_changed': classification(train, assigned, true != assigned),
            'heldout_unpatched': grouped(unpatched, heldout_true),
            'heldout_patched': grouped(patched, heldout_true), 'cue': cue}


def endpoint(row):
    result = {g + '_' + m: row['heldout_unpatched'][g][m]
              for g in ('rare', 'majority_macro', 'balanced_total') for m in ('accuracy', 'ce')}
    for prefix, source in (('patched_', 'heldout_patched'), ('train_true_', 'train_true')):
        result.update({prefix + group + '_' + metric: row[source][group][metric]
                       for group in ('rare', 'majority_macro', 'balanced_total')
                       for metric in ('accuracy', 'ce')})
    for prefix, source in (('unpatched', 'heldout_unpatched'), ('patched', 'heldout_patched'), ('train_true', 'train_true')):
        for digit in range(10):
            for metric in ('accuracy', 'ce'):
                result[f'{prefix}_class{digit}_{metric}'] = row[source]['per_class'][str(digit)][metric]
    result.update({'train_assigned_' + metric: row['train_assigned'][metric] for metric in ('accuracy', 'ce')})
    result.update({g + '_patch_excess': row['cue'][g]['patch_excess']
                   for g in ('majority_nonzero', 'rare', 'all_nonzero')})
    result['majority_nonzero_patched_asr'] = row['cue']['majority_nonzero']['patched_target0_rate']
    result['train_wrong_accuracy'] = row['train_actually_changed']['accuracy']
    result['train_wrong_ce'] = row['train_actually_changed']['ce']
    return result


def three_seed(values):
    require(len(values) == 3, 'three paired seed values required')
    if all(x is None for x in values):
        return None
    require(all(x is not None and math.isfinite(x) for x in values), 'partly undefined seed aggregation')
    mean = math.fsum(values) / 3
    sd = math.sqrt(math.fsum((x - mean)**2 for x in values) / 2)
    return {'values': values, 'mean': mean, 'sample_sd': sd, 'sample_se': sd / math.sqrt(3),
            'positive_count': sum(x > 0 for x in values), 'negative_count': sum(x < 0 for x in values),
            'zero_count': sum(x == 0 for x in values)}


def aggregate(rows):
    roster = {(r['seed'], r['cell'], r['policy']): r for r in rows}
    require(len(rows) == len(roster) == 36 and set(roster) == {(s, c, p) for s in SEEDS for c in CELLS for p in POLICIES}, 'complete aggregate roster')
    def value(seed, cell, policy, key, phase='endpoint'):
        return endpoint(roster[seed, cell, policy][phase])[key]
    keys = list(endpoint(rows[0]['endpoint']))
    groups, changes, contrasts = {}, {}, {}
    for cell in CELLS:
        for policy in POLICIES:
            groups[cell + '/' + policy] = {k: three_seed([value(s, cell, policy, k) for s in SEEDS]) for k in keys}
            changes[cell + '/' + policy] = {k: three_seed([
                None if value(s, cell, policy, k) is None or value(s, cell, policy, k, 'warmup') is None
                else value(s, cell, policy, k) - value(s, cell, policy, k, 'warmup') for s in SEEDS]) for k in keys}
        for control in ('raw', 'norm_raw'):
            contrasts[cell + '/native32_minus_' + control] = {k: three_seed([
                None if value(s, cell, 'native32', k) is None else value(s, cell, 'native32', k) - value(s, cell, control, k)
                for s in SEEDS]) for k in keys}
    interactions = {}
    cue_keys = ('majority_nonzero_patch_excess', 'majority_nonzero_patched_asr')
    for policy in ('native32', 'norm_raw'):
        for k in cue_keys:
            interactions[policy + '_minus_raw/' + k] = three_seed([
                (value(s, 'shared', policy, k) - value(s, 'sham', policy, k))
                - (value(s, 'shared', 'raw', k) - value(s, 'sham', 'raw', k)) for s in SEEDS])
    return {'seeds': list(SEEDS), 'endpoint_step': 2000, 'per_group': groups, 'policy_contrasts': contrasts,
            'change_from_warmup': changes, 'cue_interactions': interactions,
            'continuous_raw_cue_assay': {k: three_seed([value(s, 'shared', 'raw', k) - value(s, 'sham', 'raw', k) for s in SEEDS]) for k in cue_keys},
            'selection': 'fixed_step_2000_no_checkpoint_selection'}


def snapshot_arrays(state, step, tracker_expected, shapes=None, gradients_present=False):
    shapes = MODEL_SHAPES if shapes is None else shapes
    require(state['schema'] == 'i9_neural_snapshot_v1', 'snapshot schema')
    require(list(state['model_state']) == list(shapes), 'model parameter order')
    require(len(state['gradients']) == len(shapes), 'gradient parameter order')
    parameters = []
    for i, (name, shape) in enumerate(shapes.items()):
        parameters.append(arr(state['model_state'][name], np.float32, shape).reshape(-1))
        if gradients_present:
            arr(state['gradients'][i], np.float32, shape)
        else:
            require(state['gradients'][i] is None, 'cleared gradients expected')
    optimizer = state['optimizer']
    require(len(optimizer['param_groups']) == 1, 'single AdamW group')
    group = optimizer['param_groups'][0]
    fixed = {'lr': .001, 'weight_decay': .01, 'betas': (.9, .999), 'eps': 1e-8,
             'foreach': False, 'fused': False, 'amsgrad': False, 'maximize': False,
             'capturable': False, 'differentiable': False}
    for key, value in fixed.items():
        require(group[key] == value, 'AdamW contract: ' + key)
    ids = group['params']
    require(len(ids) == len(shapes) and len(set(ids)) == len(ids) and set(ids) == set(optimizer['state']), 'Adam parameter IDs')
    first, second = [], []
    for ident, shape in zip(ids, shapes.values()):
        row = optimizer['state'][ident]
        require(set(row) == {'step', 'exp_avg', 'exp_avg_sq'}, 'Adam moment fields')
        require(float(arr(row['step'])) == step, 'Adam step counter')
        first.append(arr(row['exp_avg'], np.float32, shape).reshape(-1))
        v = arr(row['exp_avg_sq'], np.float32, shape).reshape(-1)
        require((v >= 0).all(), 'nonnegative Adam second moment')
        second.append(v)
    tracker = state['tracker']
    if tracker_expected:
        require(tracker is not None and tracker['step_count'] == step + int(gradients_present), 'post-observe tracker timing')
        fixed_tracker = {'rank': 32, 'decay': .99, 'warmup': 100, 'stable_update': True,
                         'stabilize_every': 100, 'relative_eig_tol': 1e-8, 'absolute_eig_floor': 0.,
                         'weighting': 'hard', 'normalize': 'none', 'adaptive': 'none', 'proj_k': None,
                         'filter_strength': 1., 'energy_threshold': None}
        for key, value in fixed_tracker.items():
            require(tracker[key] == value, 'tracker contract: ' + key)
        p = sum(int(np.prod(shape)) for shape in shapes.values())
        require(tracker['n_params'] == p, 'tracker parameter count')
        arr(tracker['grad_mean'], np.float32, (p,))
        if tracker['V'] is not None:
            v = arr(tracker['V'], np.float32)
            require(v.ndim == 2 and v.shape[0] == p and 0 < v.shape[1] <= 32, 'native V shape')
            singular = arr(tracker['S'], shape=(v.shape[1],))
            require((singular >= 0).all(), 'nonnegative covariance singular values')
    else:
        require(tracker is None, 'raw branch must not carry observer')
    return tuple(np.concatenate(parts) for parts in (parameters, first, second))


def action(rows, v):
    rows = np.asarray(rows, np.float32)
    if v is None:
        return rows.copy()
    v = np.asarray(v, np.float32)
    return ((rows @ v) @ v.T).astype(np.float32)


def geometry(gradients, v):
    rows = arr(gradients, np.float32)
    x = rows.astype(np.float64)
    energy = np.einsum('ij,ij->i', x, x)
    mean64 = x.mean(axis=0)
    average = float(energy.mean())
    mean_energy = float(mean64 @ mean64)
    applied = action(rows, v).astype(np.float64)
    applied_energy = np.einsum('ij,ij->i', applied, applied)
    mean32 = rows.mean(axis=0, dtype=np.float32)
    applied_mean = action(mean32, v).astype(np.float64)
    denominator = float(mean32.astype(np.float64) @ mean32.astype(np.float64))
    numerator = float(applied_mean @ applied_mean)
    return {'count': len(rows), 'mean_gradient_squared_norm_float64': mean_energy,
        'mean_individual_squared_norm': average, 'coherence': mean_energy / average if average else None,
        'zero_energy_reason': None if average else 'zero_individual_energy',
        'individual_squared_norms': energy.tolist(), 'individual_native_squared_norms': applied_energy.tolist(),
        'individual_native_retention': [float(a / e) if e else None for a, e in zip(applied_energy, energy)],
        'energy_weighted_native_retention': float(applied_energy.sum() / energy.sum()) if average else None,
        'native_action_of_fp32_mean_squared_norm': numerator, 'fp32_mean_squared_norm': denominator,
        'native_mean_retention': numerator / denominator if denominator else None,
        'basis_identity_fallback': v is None}


def check_geometry(observed, gradients, v, checks, label):
    expected = geometry(gradients, v)
    checks.equal(observed, expected, label, atol=1e-7, rtol=5e-4)
    exact_arithmetic = ('mean_gradient_squared_norm_float64', 'mean_individual_squared_norm',
                        'coherence', 'individual_squared_norms')
    for key in exact_arithmetic:
        checks.equal(observed[key], expected[key], label + ' FP64/' + key)


def span_basis(v):
    if v is None:
        return None, {'rank': 0, 'reason': 'basis_unavailable', 'singular_values': [], 'tolerance': None}
    x = v.astype(np.float64)
    u, s, _ = np.linalg.svd(x, full_matrices=False)
    tolerance = float(np.finfo(np.float64).eps * max(x.shape) * s[0])
    keep = s > tolerance
    return u[:, keep], {'rank': int(keep.sum()), 'reason': None, 'singular_values': s.tolist(),
        'tolerance': tolerance, 'native_gram_max_abs_error': float(np.max(np.abs(x.T @ x - np.eye(x.shape[1]))))}


def movements(old, new, q):
    decayed = old * np.float32(1 - .001 * .01)
    vectors = {'total': new.astype(np.float64) - old.astype(np.float64),
               'decay': decayed.astype(np.float64) - old.astype(np.float64),
               'adaptive': new.astype(np.float64) - decayed.astype(np.float64)}
    summary = {}
    for name, d in vectors.items():
        energy = float(d @ d)
        outside = None if q is None else float(np.sum((d - q @ (q.T @ d))**2))
        summary[name] = {'squared_norm': energy, 'norm': math.sqrt(energy),
                         'outside_span_squared_norm': outside,
                         'outside_span_fraction': outside / energy if outside is not None and energy else None}
    return vectors, summary


def verify_adam(old, new, m0, v0, m1, v1, g, update, checks):
    """Saved-array recurrence identities only, not an optimizer.step/replay."""
    old, new, m0, v0, m1, v1, g = [np.asarray(x, np.float64) for x in (old, new, m0, v0, m1, v1, g)]
    expected_m, expected_v = .9 * m0 + .1 * g, .999 * v0 + .001 * g * g
    bounds = {'first_moment': 16 * EPS32 * (.9 * np.abs(m0) + .1 * np.abs(g)) + 1e-30,
              'second_moment': 16 * EPS32 * (.999 * np.abs(v0) + .001 * g * g) + 1e-30}
    records = {}
    for name, got, target in (('first_moment', m1, expected_m), ('second_moment', v1, expected_v)):
        error = np.abs(got - target)
        require(np.all(error <= bounds[name]), name + ': recurrence exceeds FP32 roundoff bound')
        records[name + '_max_abs_error'] = float(error.max())
    adaptive = -.001 * (m1 / (1 - .9**update)) / (np.sqrt(v1 / (1 - .999**update)) + 1e-8)
    ideal_new = old * (1 - .001 * .01) + adaptive
    error = np.abs(new - ideal_new)
    bound = 16 * EPS32 * (np.abs(old) + np.abs(adaptive)) + 1e-30
    require(np.all(error <= bound), 'AdamW parameter recurrence exceeds FP32 roundoff bound')
    records['parameter_max_abs_error'] = float(error.max())
    records['parameter_max_error_over_bound'] = float(np.max(error / bound))
    checks.count += 3
    return records


def verify_diagnostic(payload, report, plan, clean_x, cell_x, assigned, identity, warmup, checks):
    seed, cell, update = identity
    for obj in (payload, report):
        require(obj['schema'] == SCHEMA and obj['seed'] == seed and obj['cell'] == cell
                and obj['policy'] == 'native32' and obj['update'] == update, 'diagnostic identity')
        require(obj['warmup_state_sha256'] == tree_hash(warmup), 'diagnostic warmup binding')
    require(payload['parameter_order'] == list(MODEL_SHAPES), 'diagnostic parameter order')
    require(payload['probe_parameter_state'] == 'pre_update_after_one_training_observer_update', 'probe timing')
    before, after = payload['before'], payload['after']
    require(tree_hash(before) == report['before_sha256'] and tree_hash(after) == report['after_sha256'], 'diagnostic state hash')
    # Before: Adam counter u-1, observer u. After: both counters u.
    old, m0, v0 = snapshot_arrays(before, update - 1, True, gradients_present=True)
    # After gradients still exist but the observer is not advanced a second time.
    after_without_gradients = {**after, 'gradients': [None] * len(MODEL_SHAPES)}
    new, m1, v1 = snapshot_arrays(after_without_gradients, update, True)
    for key in ('tracker', 'rng', 'model_modes', 'model_spec', 'gradients'):
        require(tree_hash(before[key]) == tree_hash(after[key]), 'unexpected one-step mutation: ' + key)
    if update == 101:
        for key in ('model_state', 'optimizer', 'rng', 'model_modes', 'model_spec'):
            require(tree_hash(before[key]) == tree_hash(warmup[key]), 'step101 common-state mismatch: ' + key)
    p = len(old)
    raw = arr(payload['raw_training_gradient'], np.float32, (p,))
    delivered = arr(payload['applied_training_gradient'], np.float32, (p,))
    stored_grad = np.concatenate([arr(g).reshape(-1) for g in before['gradients']])
    checks.array(stored_grad, delivered, 'delivered training .grad')
    v = None if before['tracker']['V'] is None else arr(before['tracker']['V'], np.float32)
    ideal = raw.astype(np.float64) if v is None else v.astype(np.float64) @ (v.astype(np.float64).T @ raw.astype(np.float64))
    error = float(np.linalg.norm(delivered.astype(np.float64) - ideal))
    bound = 3e-5 * float(np.linalg.norm(raw.astype(np.float64))) + 1e-12
    require(error <= bound, 'native training action mismatch')
    q, basis = span_basis(v)
    checks.equal(report['basis'], basis, 'basis', atol=1e-10, rtol=1e-8)
    vectors, movement = movements(old, new, q)
    checks.equal(report['movement'], movement, 'movement', atol=1e-10, rtol=1e-8)
    adam = verify_adam(old, new, m0, v0, m1, v1, delivered, update, checks)
    true = plan['train_true']
    wrong = np.where(true != assigned)[0][:32]
    positions = {'common3': np.where(true == 3)[0][:32], 'rare8': np.where(true == 8)[0][:32],
                 'wrong_assigned': wrong, 'wrong_corrected': wrong}
    require(set(payload['groups']) == set(report['groups']) == set(positions), 'probe group roster')
    for name, selected in positions.items():
        group, measured = payload['groups'][name], report['groups'][name]
        if len(selected) == 0:
            require(group is None, 'absent probe must be null')
            checks.equal(measured, {'status': 'absent', 'reason': 'no_actually_changed_examples', 'count': 0}, 'absent probe')
            continue
        require(group is not None and measured['status'] == 'defined', 'defined probe required')
        for key, expected in {'training_positions': selected, 'source_ids': plan['train_ids'][selected],
                              'true_targets': true[selected],
                              'targets': assigned[selected] if name == 'wrong_assigned' else true[selected]}.items():
            checks.array(group[key], expected, 'probe ' + name + '/' + key)
        checks.array(group['inputs'], (cell_x if name.startswith('wrong_') else clean_x)[selected], 'probe inputs')
        gradients = arr(group['per_example_gradients'], np.float32, (len(selected), p))
        pre_logits = arr(group['pre_logits'], np.float32, (len(selected), 10))
        post_logits = arr(group['post_logits'], np.float32, (len(selected), 10))
        target = arr(group['targets'])
        pre, post = classification(pre_logits, target), classification(post_logits, target)
        checks.equal(measured['true_label_counts'], np.bincount(true[selected], minlength=10).tolist(), 'probe true-label composition')
        checks.equal(measured['pre'], pre, 'probe pre metrics')
        checks.equal(measured['post'], post, 'probe post metrics')
        checks.equal(measured['finite_ce_improvement'], pre['ce'] - post['ce'], 'probe finite CE improvement')
        check_geometry(measured['geometry'], gradients, v, checks, 'probe geometry')
        utilities = {key: -float(gradients.astype(np.float64).mean(axis=0) @ delta) for key, delta in vectors.items()}
        checks.equal(measured['signed_first_order_utilities'], utilities, 'probe utilities')
    residual = None
    if len(wrong):
        residual = (arr(payload['groups']['wrong_assigned']['per_example_gradients'])
                    - arr(payload['groups']['wrong_corrected']['per_example_gradients']))
        check_geometry(report['wrong_minus_true_gradient_geometry'], residual, v, checks, 'residual geometry')
    else:
        checks.equal(report['wrong_minus_true_gradient_geometry'], None, 'absent residual')
    require(report['residual_arithmetic_dtype'] == 'float32' and report['probe_side_effect_checks'] == 'PASS', 'probe neutrality metadata')
    return {'seed': seed, 'cell': cell, 'update': update, 'action_l2_error': error,
            'action_l2_bound': bound, 'adam': adam, 'basis': basis, 'movement': movement,
            'after_without_gradients_sha256': tree_hash(after_without_gradients)}


def verify_actions(rows, policy, checks, warmup=False):
    expected_steps = list(range(1, 101)) if warmup else list(range(101, 2001))
    require([r['step'] for r in rows] == expected_steps, 'complete action history')
    for row in rows:
        checks.tick()
        for key in ('training_batch_ce', 'raw_norm', 'native_norm', 'applied_norm'):
            require(type(row[key]) in (float, int) and math.isfinite(row[key]) and row[key] >= 0, 'finite action scalar: ' + key)
        tracked = warmup or policy != 'raw'
        require(row['observer_step'] == (row['step'] if tracked else None), 'action observer counter')
        require((type(row['basis_rank']) is int and 0 <= row['basis_rank'] <= 32) if tracked
                else row['basis_rank'] is None, 'action basis rank')
        require(row['native_identity_fallback'] == bool(tracked and row['basis_rank'] == 0), 'native identity fallback flag')
        law = row['norm_law']
        if policy != 'norm_raw' or warmup:
            require(law is None, 'unexpected norm-law record')
            target = row['raw_norm'] if warmup or policy == 'raw' else row['native_norm']
            checks.equal(row['applied_norm'], target, 'action norm', atol=1e-12, rtol=1e-10)
        else:
            require(type(law) is dict, 'missing norm-law record')
            raw, target, achieved = row['raw_norm'], row['native_norm'], row['applied_norm']
            require(raw > 0 or target == 0, 'zero raw with nonzero native target')
            error = abs(achieved - target) / target if target else 0.
            require(error <= 10 * EPS32 and (target > 0 or achieved == 0), 'norm-law tolerance')
            expected = {'raw_norm': raw, 'target_norm': target, 'delivered_norm': achieved,
                'scale': target / raw if raw else None, 'relative_postcast_mismatch': error,
                'relative_tolerance': 10 * EPS32, 'zero_raw': raw == 0, 'zero_target': target == 0}
            checks.equal(law, expected, 'norm law', atol=1e-12, rtol=1e-10)
        checks.count += 1


def expected_artifacts():
    names = {'manifest.json', 'results.json'}
    for seed in SEEDS:
        names.update({f'plan-s{seed}.npz', f'plan-s{seed}.json', f'warmup-s{seed}.pt',
                      f'warmup-actions-s{seed}.json', f'common-heldout-s{seed}.npz'})
        for cell in CELLS:
            names.add(f'binding-s{seed}-{cell}.json')
            for policy in POLICIES:
                identity = f's{seed}-{cell}-{policy}'
                names.update({f'final-{identity}.pt', f'logits-{identity}.npz',
                              f'actions-{identity}.json', f'curve-{identity}.json'})
                if policy == 'native32':
                    for update in ANCHORS:
                        names.update({f'diagnostic-{identity}-u{update:04d}.pt',
                                      f'diagnostic-{identity}-u{update:04d}.json'})
    return names


def verify_sources(pins):
    require(type(pins) is dict and set(pins) == SOURCE_NAMES, 'exact producer source pin roster')
    for name, expected in pins.items():
        path = ROOT / name
        require(path.resolve().is_relative_to(ROOT), 'source path escape')
        cursor = path
        while cursor != ROOT:
            require(not cursor.is_symlink(), 'source symlink')
            cursor = cursor.parent
        require(path.is_file() and sha256(path) == expected, 'source pin mismatch: ' + name)


class Bundle:
    def __init__(self, root, complete_sha, manifest_sha, source_pins, checks):
        self.root = Path(root).resolve(strict=True)
        self.checks = checks
        require(self.root.is_dir(), 'acquisition directory missing')
        self.complete_path = contained_file(self.root, 'complete.json')
        require(sha256(self.complete_path) == complete_sha, 'main-supplied completion pin')
        self.complete = read_json(self.complete_path)
        c = self.complete
        require(c['schema'] == SCHEMA and c['status'] == 'complete' and c['completed_trajectories'] == 36
                and c['completed_diagnostics'] == 36, 'complete acquisition required')
        require(c['source_pins'] == source_pins and c['data_pins'] == DATA_PINS, 'completion input pins')
        self.index = {}
        for item in c['receipts']:
            checks.tick()
            path = verify_receipt(self.root, item)
            require(path.name not in self.index, 'duplicate artifact receipt')
            self.index[path.name] = item
        require(set(self.index) == expected_artifacts(), 'exact scientific artifact roster')
        require({p.name for p in self.root.iterdir()} == expected_artifacts() | {'complete.json'}, 'unexpected/unreceipted acquisition file')
        require(sum(r['size_bytes'] for r in self.index.values()) == c['artifact_bytes_before_completion'], 'artifact byte accounting')
        require(c['artifact_bytes_before_completion'] + self.complete_path.stat().st_size <= 3 * 1024**3, 'acquisition output cap')
        require(0 < c['wall_seconds'] <= 1800 and c['gpu_max_allocated_bytes'] <= 8 * 1024**3
                and c['process_high_water_rss_kib'] <= 16 * 1024**2, 'acquisition resource completion')
        require(self.index['manifest.json']['sha256'] == manifest_sha, 'main-supplied manifest pin')
        self.manifest = self.json('manifest.json')
        m = self.manifest
        checks.equal({k: m[k] for k in ('schema', 'seeds', 'cells', 'policies', 'steps', 'warmup', 'batch_size', 'eval_steps',
            'probe_steps', 'rank', 'rare_class', 'wrong_target', 'poison_count', 'majority_train_per_class',
            'rare_train_count', 'heldout_per_class', 'cloud_spend_usd')},
            {'schema': SCHEMA, 'seeds': list(SEEDS), 'cells': list(CELLS), 'policies': list(POLICIES),
             'steps': 2000, 'warmup': 100, 'batch_size': 64, 'eval_steps': list(EVAL), 'probe_steps': list(ANCHORS),
             'rank': 32, 'rare_class': 8, 'wrong_target': 0, 'poison_count': 500, 'majority_train_per_class': 550,
             'rare_train_count': 50, 'heldout_per_class': 500, 'cloud_spend_usd': 0}, 'manifest contract')
        require(m['source_pins'] == source_pins and m['data_pins'] == DATA_PINS, 'manifest source/data binding')
        require(m['data_directory'] == str(DATA) and m['numpy'] == '1.26.4' and m['torch'] == '2.11.0+cu128', 'manifest environment')
        require(m['cooperative_seconds'] == 1500 and m['byte_inventory']['cap_bytes'] == 3 * 1024**3, 'manifest limits')
        admission = m['resource_admission']
        require(admission['effective']['memory.max'] == str(16 * 1024**3)
                and admission['effective']['memory.swap.max'] == '0', 'acquisition memory guard')
        quota, period = admission['effective']['cpu.max'].split()
        require(quota != 'max' and int(quota) == int(period) > 0, 'acquisition CPU guard')
        require(admission['service'] == {'Type': 'exec', 'RuntimeMaxUSec': '30min', 'Restart': 'no', 'KillMode': 'control-group'}, 'acquisition service guard')
        self.bound(c['results'], 'results.json')
        verify_sources(source_pins)
        self.source_pins = source_pins
        self.complete_sha = complete_sha

    def bound(self, item, name):
        require(name in self.index and item == self.index[name], 'cross-receipt binding: ' + name)
        return self.root / name

    def json(self, name):
        require(name in self.index, 'unreceipted JSON')
        return read_json(self.root / name)

    def npz(self, name):
        require(name in self.index, 'unreceipted NPZ')
        return load_npz(self.root / name)

    def tensor(self, name):
        require(name in self.index, 'unreceipted tensor')
        return cpu_tensor_file(self.root / name)

    def recheck(self):
        for item in self.index.values():
            self.checks.tick()
            verify_receipt(self.root, item)
        require(sha256(self.complete_path) == self.complete_sha, 'completion changed during audit')
        verify_sources(self.source_pins)
        for name, expected in DATA_PINS.items():
            require(sha256(DATA / name) == expected, 'data changed during audit')


def read_pinned_data():
    for name, expected in DATA_PINS.items():
        require((DATA / name).is_file() and not (DATA / name).is_symlink() and sha256(DATA / name) == expected, 'accepted data pin: ' + name)
    data = (DATA / 'train-images-idx3-ubyte').read_bytes()
    require(len(data) == 16 + 60000 * 784 and struct.unpack('>4I', data[:16]) == (2051, 60000, 28, 28), 'image IDX header')
    images = np.frombuffer(data, np.uint8, offset=16).reshape(60000, 784)
    labels = (DATA / 'train-labels-idx1-ubyte').read_bytes()
    require(len(labels) == 60008 and struct.unpack('>2I', labels[:8]) == (2049, 60000), 'label IDX header')
    return images, np.frombuffer(labels, np.uint8, offset=8).astype(np.int64)


def audit(bundle, checks):
    images, labels = read_pinned_data()
    result = bundle.json('results.json')
    require(result['schema'] == SCHEMA and len(result['rows']) == 36, 'results schema/roster')
    rows_by_id = {(r['seed'], r['cell'], r['policy']): r for r in result['rows']}
    require(set(rows_by_id) == {(s, c, p) for s in SEEDS for c in CELLS for p in POLICIES}, 'results IDs')
    rebuilt_rows, diagnostic_rows = [], []
    for seed in SEEDS:
        checks.tick()
        plan, allocations = rebuild_plan(labels, seed)
        saved_plan = bundle.npz(f'plan-s{seed}.npz')
        require(set(saved_plan) == set(plan), 'plan member roster')
        for key, expected in plan.items():
            checks.array(saved_plan[key], expected, 'plan/' + key)
        meta = bundle.json(f'plan-s{seed}.json')
        plan_hashes = {key: array_hash(value) for key, value in plan.items()}
        checks.equal(meta, {'seed': seed, 'array_hashes': plan_hashes, 'sham_allocation': allocations,
            'rng': 'PCG64/SeedSequence([seed, stream_id])', 'numpy_version': '1.26.4', 'initialization_seed': seed}, 'plan metadata')
        warmup = bundle.tensor(f'warmup-s{seed}.pt')
        snapshot_arrays(warmup, 100, True)
        warmup_hash = tree_hash(warmup)
        verify_actions(bundle.json(f'warmup-actions-s{seed}.json'), 'raw', checks, warmup=True)
        common = bundle.npz(f'common-heldout-s{seed}.npz')
        require(set(common) == {'steps', 'unpatched', 'patched'}, 'common logits members')
        checks.array(common['steps'], np.array([0, 100], np.int64), 'common eval steps')
        for key in ('unpatched', 'patched'):
            arr(common[key], np.float32, (2, 5000, 10))
        clean_x = np.ascontiguousarray(images[plan['train_ids']].astype(np.float32) / np.float32(255))
        heldout_x = np.ascontiguousarray(images[plan['heldout_ids']].astype(np.float32) / np.float32(255))
        input_hashes = {'train_clean_inputs_sha256': array_hash(clean_x),
                        'heldout_unpatched_inputs_sha256': array_hash(heldout_x),
                        'heldout_patched_inputs_sha256': array_hash(patched_images(heldout_x))}
        initial_hash = None
        for cell in CELLS:
            assigned, patch, counts = cell_data(plan, cell)
            cell_x = patched_images(clean_x, patch)
            binding = bundle.json(f'binding-s{seed}-{cell}.json')
            for key, expected in {'schema': SCHEMA, 'seed': seed, 'cell': cell, 'plan_array_hashes': plan_hashes,
                'warmup_state_sha256': warmup_hash, 'counts': counts, 'assigned_targets_sha256': array_hash(assigned),
                'patch_mask_sha256': array_hash(patch), 'train_inputs_sha256': array_hash(cell_x), **input_hashes}.items():
                checks.equal(binding[key], expected, 'cell binding/' + key)
            require(binding['input_hash_convention'] == 'array_digest of contiguous CPU float32 [N,784]; NumPy uint8.astype(float32)/float32(255) before transfer; plan row order', 'input hash convention')
            for field, name in {'plan': f'plan-s{seed}.npz', 'plan_metadata': f'plan-s{seed}.json',
                                'warmup': f'warmup-s{seed}.pt', 'common_heldout': f'common-heldout-s{seed}.npz'}.items():
                bundle.bound(binding[field], name)
            require(initial_hash is None or binding['initial_model_sha256'] == initial_hash, 'shared initial model hash')
            initial_hash = binding['initial_model_sha256']
            first_raw, first_observer, first_train = None, None, None
            for policy in POLICIES:
                checks.tick()
                identifier = f's{seed}-{cell}-{policy}'
                row = rows_by_id[seed, cell, policy]
                bundle.bound(row['curve'], f'curve-{identifier}.json')
                record = bundle.json(f'curve-{identifier}.json')
                for key in binding:
                    checks.equal(record[key], binding[key], 'trajectory binding/' + key)
                require(record['policy'] == policy, 'trajectory policy')
                offset = (SEEDS.index(seed) + CELLS.index(cell)) % 3
                require(record['policy_order'] == list(POLICIES[offset:] + POLICIES[:offset]), 'fixed policy order')
                for field, name in {'logits': f'logits-{identifier}.npz', 'actions': f'actions-{identifier}.json',
                                    'final_state': f'final-{identifier}.pt'}.items():
                    bundle.bound(record[field], name)
                prediction = bundle.npz(f'logits-{identifier}.npz')
                require(set(prediction) == {'steps', 'train', 'heldout_unpatched', 'heldout_patched'}, 'trajectory NPZ members')
                checks.array(prediction['steps'], np.asarray(EVAL, np.int64), 'eval steps')
                for key in ('train', 'heldout_unpatched', 'heldout_patched'):
                    arr(prediction[key], np.float32, (21, 5000, 10))
                checks.array(prediction['heldout_unpatched'][:2], common['unpatched'], 'common unpatched reuse')
                checks.array(prediction['heldout_patched'][:2], common['patched'], 'common patched reuse')
                if first_train is None:
                    first_train = prediction['train'][:2].copy()
                checks.array(prediction['train'][:2], first_train, 'within-cell common train reuse')
                require(len(record['curve']) == 21, 'all evaluation rows required')
                independently_derived = []
                for i, step in enumerate(EVAL):
                    expected = metrics(step, prediction['train'][i], prediction['heldout_unpatched'][i],
                        prediction['heldout_patched'][i], plan['train_true'], assigned, plan['heldout_true'])
                    checks.equal(record['curve'][i], expected, 'evaluation')
                    independently_derived.append(expected)
                checks.equal(row['endpoint'], independently_derived[-1], 'results endpoint')
                checks.equal(row['warmup'], independently_derived[1], 'results warmup')
                require(record['branch_seconds_including_eval_and_diagnostics'] > 0 and binding['warmup_seconds'] > 0, 'positive runtime')
                checks.equal(row['warmup_plus_branch_seconds'], record['warmup_plus_branch_seconds'], 'runtime row')
                checks.equal(record['warmup_plus_branch_seconds'], binding['warmup_seconds'] + record['branch_seconds_including_eval_and_diagnostics'], 'runtime arithmetic')
                actions = bundle.json(f'actions-{identifier}.json')
                verify_actions(actions, policy, checks)
                first = {key: value for key, value in actions[0].items() if key.endswith('sha256')}
                checks.equal(record['first_action_binding'], first, 'first action binding')
                require(first_raw is None or first_raw == first['raw_gradient_sha256'], 'first raw gradient differs across policies')
                first_raw = first['raw_gradient_sha256']
                if policy != 'raw':
                    require(first_observer is None or first_observer == first['post_observe_tracker_sha256'], 'first observer differs')
                    first_observer = first['post_observe_tracker_sha256']
                require(len(record['diagnostics']) == (3 if policy == 'native32' else 0), 'diagnostic count')
                diagnostic_after = None
                for update, receipt in zip(ANCHORS, record['diagnostics']):
                    stem = f'diagnostic-{identifier}-u{update:04d}'
                    bundle.bound(receipt, stem + '.json')
                    report = bundle.json(stem + '.json')
                    bundle.bound(report['tensor'], stem + '.pt')
                    payload = bundle.tensor(stem + '.pt')
                    diagnostic = verify_diagnostic(payload, report, plan, clean_x, cell_x, assigned, (seed, cell, update), warmup, checks)
                    if update == 101:
                        require(tree_hash(payload['raw_training_gradient']) == first['raw_gradient_sha256']
                                and tree_hash(payload['before']['tracker']) == first['post_observe_tracker_sha256'], 'first diagnostic binding')
                    action_row = actions[update - 101]
                    for field, tensor in (('raw_norm', payload['raw_training_gradient']), ('applied_norm', payload['applied_training_gradient'])):
                        checks.equal(action_row[field], float(np.linalg.norm(arr(tensor).astype(np.float64))), 'diagnostic action scalar')
                    diagnostic_after = diagnostic['after_without_gradients_sha256']
                    diagnostic_rows.append(diagnostic)
                    del payload
                final = bundle.tensor(f'final-{identifier}.pt')
                snapshot_arrays(final, 2000, policy != 'raw')
                require(tree_hash(final) == record['final_state_sha256'], 'final scientific hash')
                if policy == 'native32':
                    require(tree_hash(final) == diagnostic_after, 'final native state equals actual update2000 after gradient clearing')
                rebuilt_rows.append({'seed': seed, 'cell': cell, 'policy': policy,
                                     'endpoint': independently_derived[-1], 'warmup': independently_derived[1]})
                del final, prediction, actions, record
    summary = aggregate(rebuilt_rows)
    checks.equal(result['summary'], summary, 'aggregate')
    require(len(diagnostic_rows) == 36, 'complete independent diagnostics')
    bundle.recheck()
    return {'summary': summary, 'diagnostics': diagnostic_rows,
            'evaluations_recomputed': 756, 'native_events_checked': 36,
            'paired_rows': rebuilt_rows}


def resource_guard():
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        require(os.environ.get(name) == '1', 'one CPU math thread required: ' + name)
    group = next(line.split(':', 2)[2] for line in Path('/proc/self/cgroup').read_text().splitlines() if line.startswith('0::'))
    require(Path(group).name == UNIT, 'unexpected audit unit')
    directory = Path('/sys/fs/cgroup') / group.lstrip('/')
    limits = {name: (directory / name).read_text().strip() for name in ('memory.max', 'memory.swap.max', 'cpu.max')}
    quota, period = limits['cpu.max'].split()
    require(limits['memory.max'] == str(4 * 1024**3) and limits['memory.swap.max'] == '0'
            and quota != 'max' and int(quota) == int(period) > 0, 'audit cgroup limits')
    output = subprocess.check_output(['systemctl', '--user', 'show', UNIT, '--property=Type',
        '--property=RuntimeMaxUSec', '--property=Restart', '--property=KillMode'], text=True)
    service = dict(line.split('=', 1) for line in output.splitlines())
    require(service == {'Type': 'exec', 'RuntimeMaxUSec': '5min', 'Restart': 'no', 'KillMode': 'control-group'}, 'audit service limits')
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    require(torch.__version__ == '2.11.0+cu128' and np.__version__ == '1.26.4', 'audit arithmetic environment')
    return {'cgroup': group, 'effective': limits, 'service': service}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--acquisition-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--expected-complete-sha256', required=True)
    parser.add_argument('--expected-manifest-sha256', required=True)
    parser.add_argument('--expected-sources-json', type=Path, required=True)
    args = parser.parse_args()
    require(args.execute, 'explicit --execute is required')
    require(not args.acquisition_dir.is_symlink(), 'acquisition symlink forbidden')
    parent = args.output_dir.parent.resolve(strict=True)
    require(args.output_dir == parent / 'audit-001' and not args.output_dir.exists()
            and not args.output_dir.is_symlink() and not parent.is_relative_to(args.acquisition_dir.resolve()), 'exclusive audit-001 outside acquisition required')
    resource_admission = resource_guard()
    expected_pins = read_json(args.expected_sources_json)
    paths = [Path(__file__), ROOT / 'tests/test_audit_spectral_selectivity_boundary.py',
             ROOT / DOCS / 'audit-plan.md', ROOT / DOCS / 'audit-implementation.md']
    checker_pins = {str(p.relative_to(ROOT)): sha256(p) for p in paths}
    expected_pins_file_sha = sha256(args.expected_sources_json)
    checks = Checks()
    args.output_dir.mkdir(exist_ok=False)
    status, error, numerical = 'PASS', None, None
    bundle = None
    try:
        bundle = Bundle(args.acquisition_dir, args.expected_complete_sha256, args.expected_manifest_sha256, expected_pins, checks)
        numerical = audit(bundle, checks)
        require({str(p.relative_to(ROOT)): sha256(p) for p in paths} == checker_pins,
                'checker sources changed during audit')
        require(sha256(args.expected_sources_json) == expected_pins_file_sha, 'main source-pin file changed during audit')
    except Exception as exc:
        status, error = 'FAIL', {'type': type(exc).__name__, 'message': str(exc)}
    payload = {'schema': 'spectral_selectivity_boundary_independent_audit_v1', 'status': status,
        'error': error, 'checks': checks.count, 'max_absolute_discrepancies': checks.max_absolute,
        'tolerances': TOLERANCES, 'elapsed_seconds': time.monotonic() - checks.started,
        'resource_admission': resource_admission, 'process_high_water_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'acquisition_dir': str(args.acquisition_dir.resolve()), 'expected_completion_sha256': args.expected_complete_sha256,
        'expected_manifest_sha256': args.expected_manifest_sha256, 'expected_sources': expected_pins,
        'checker_sources': checker_pins, 'expected_sources_file_sha256': expected_pins_file_sha,
        'input_receipts': None if bundle is None else list(bundle.index.values()), 'independent': numerical,
        'scope_limit': 'Saved-array arithmetic and provenance only; no model inference, differentiation, optimizer execution or fresh-seed replication.'}
    data = (json.dumps(payload, indent=2, allow_nan=False) + '\n').encode()
    require(len(data) <= 100 * 1024**2, 'audit output byte cap')
    with (args.output_dir / 'result.json').open('xb') as handle:
        handle.write(data)
    print(json.dumps({'status': status, 'checks': checks.count, 'error': error,
                      'result_sha256': sha256(args.output_dir / 'result.json')}), flush=True)
    if status != 'PASS':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
