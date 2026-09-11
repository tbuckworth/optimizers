#!/usr/bin/env python3
"""One-pass saved-logit audit; no model, torch, IDX, GPU, or optimizer execution.

Import is inert. CLI requires --execute and a new exclusive output directory.
Checkpoint bytes are SHA256-checked without deserialization. State tree hashes
and first-gradient hashes are producer assertions checked for paired binding,
not independently recomputed mathematical states.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import resource
import stat
import sys
import time
import zipfile

import numpy as np

if __package__:
    from .spectral_augmentation_analysis import summarize
else:
    from spectral_augmentation_analysis import summarize

SEEDS = (202609131, 202609132, 202609133)
CELLS = ('clean', 'shared', 'sham')
POLICIES = ('raw', 'native32')
MODES = ('none', 'random', 'targeted', 'opposite')
EVAL = tuple(range(0, 2001, 100))
SCHEMA = 'spectral_augmentation_boundary_v1'
AUDIT_SCHEMA = 'spectral_augmentation_audit_v1'
ATOL, RTOL = 1e-10, 1e-10
INPUT_CAP, OUTPUT_CAP = 2 * 1024**3, 16 * 1024**2
DATA_PINS = {
    'train-images-idx3-ubyte': 'ba891046e6505d7aadcbbe25680a0738ad16aec93bde7f9b65e87a2fc25776db',
    'train-labels-idx1-ubyte': '65a50cbbf4e906d70832878ad85ccda5333a97f0f4c3dd2ef09a8a9eef7101c5',
}


class AuditError(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise AuditError(message)


@dataclass(frozen=True)
class Dimensions:
    """Nondefault sizes exist only for fabricated tests; never a CLI option."""
    common_train: int = 550
    rare_train: int = 50
    heldout_per_class: int = 500
    poison_count: int = 500
    batch: int = 64

    @property
    def train(self):
        return 9 * self.common_train + self.rare_train

    @property
    def heldout(self):
        return 10 * self.heldout_per_class


def array_hash(value):
    value = np.ascontiguousarray(value)
    header = json.dumps([value.dtype.str, list(value.shape)], separators=(',', ':')).encode()
    return hashlib.sha256(header + b'\n' + value.tobytes()).hexdigest()


def is_hash(value):
    return type(value) is str and re.fullmatch('[0-9a-f]{64}', value) is not None


def array(value, dtype, shape, name):
    require(isinstance(value, np.ndarray) and value.dtype == np.dtype(dtype)
            and value.shape == shape, 'array schema: ' + name)
    require(np.isfinite(value).all(), 'nonfinite array: ' + name)
    return value


def classification(logits, labels, mask=None):
    labels = np.asarray(labels)
    require(labels.ndim == 1 and labels.dtype.kind in 'iu'
            and np.all((labels >= 0) & (labels < 10)), 'label domain')
    array(logits, np.float32, (len(labels), 10), 'classification logits')
    chosen = np.ones(len(labels), dtype=bool) if mask is None else np.asarray(mask)
    require(chosen.dtype == np.bool_ and chosen.shape == labels.shape, 'metric mask')
    y, z = labels[chosen], logits[chosen].astype(np.float64)
    count = len(y)
    if not count:
        return {'count': 0, 'correct': 0, 'ce_sum': 0., 'accuracy': None, 'ce': None}
    shifted = z - z.max(axis=1, keepdims=True)
    losses = np.log(np.exp(shifted).sum(axis=1)) - shifted[np.arange(count), y]
    total = math.fsum(float(v) for v in losses)
    correct = int(np.count_nonzero(z.argmax(axis=1) == y))
    return {'count': count, 'correct': correct, 'ce_sum': total,
            'accuracy': correct / count, 'ce': total / count}


def grouped(logits, labels):
    per_class = {str(d): classification(logits, labels, labels == d) for d in range(10)}
    require(all(v['count'] for v in per_class.values()), 'missing evaluation class')
    def macro(digits):
        return {key: math.fsum(per_class[str(d)][key] for d in digits) / len(digits)
                for key in ('accuracy', 'ce')}
    return {'per_class': per_class, 'rare': per_class['8'],
            'majority_macro': macro([d for d in range(10) if d != 8]),
            'balanced_total': macro(list(range(10))), 'micro': classification(logits, labels)}


def evaluation_row(step, train, unpatched, patched, true, assigned, heldout_true):
    for name, value in (('unpatched', unpatched), ('patched', patched)):
        array(value, np.float32, (len(heldout_true), 10), name)
    before, after = unpatched.argmax(1), patched.argmax(1)
    cue = {}
    for name, selected in (('majority_nonzero', ~np.isin(heldout_true, [0, 8])),
                           ('rare', heldout_true == 8), ('all_nonzero', heldout_true != 0)):
        count = int(selected.sum())
        left = int(np.count_nonzero(before[selected] == 0))
        right = int(np.count_nonzero(after[selected] == 0))
        cue[name] = {'count': count, 'unpatched_target0_count': left,
            'patched_target0_count': right, 'unpatched_target0_rate': left / count if count else None,
            'patched_target0_rate': right / count if count else None,
            'patch_excess': (right - left) / count if count else None}
    return {'step': step, 'train_true': grouped(train, true),
            'train_assigned': classification(train, assigned),
            'train_actually_changed': classification(train, assigned, assigned != true),
            'heldout_unpatched': grouped(unpatched, heldout_true),
            'heldout_patched': grouped(patched, heldout_true), 'cue': cue}


class Checks:
    def __init__(self):
        self.count, self.max_error = 0, 0.

    def equal(self, observed, expected, path='value'):
        self.count += 1
        if type(expected) is dict:
            require(type(observed) is dict and observed.keys() == expected.keys(), 'keys: ' + path)
            for key in expected:
                self.equal(observed[key], expected[key], path + '/' + str(key))
        elif type(expected) is list:
            require(type(observed) is list and len(observed) == len(expected), 'list: ' + path)
            for i, (left, right) in enumerate(zip(observed, expected)):
                self.equal(left, right, path + '/' + str(i))
        elif type(expected) is float:
            require(type(observed) in (float, int) and math.isfinite(observed)
                    and math.isfinite(expected), 'finite metric: ' + path)
            error = abs(observed - expected)
            self.max_error = max(error, self.max_error)
            require(error <= ATOL + RTOL * abs(expected), 'numerical mismatch: ' + path)
        else:
            require(type(observed) is type(expected) and observed == expected, 'exact mismatch: ' + path)


def parse_json(payload):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'duplicate JSON key')
            result[key] = value
        return result
    def finite(value):
        result = float(value)
        require(math.isfinite(result), 'nonfinite JSON number')
        return result
    def reject(value):
        raise AuditError('nonfinite JSON constant: ' + value)
    return json.loads(payload, object_pairs_hook=pairs, parse_float=finite, parse_constant=reject)


def parse_npz(payload):
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        names = archive.namelist()
        require(0 < len(names) <= 32 and len(names) == len(set(names)), 'NPZ member roster')
        require(all(Path(n).name == n and n.endswith('.npy') for n in names), 'NPZ member escape')
        require(sum(v.file_size for v in archive.infolist()) <= 128 * 1024**2, 'expanded NPZ cap')
    try:
        with np.load(io.BytesIO(payload), allow_pickle=False, max_header_size=10000) as archive:
            result = {key: archive[key] for key in archive.files}
    except (ValueError, OSError) as error:
        raise AuditError('unsafe or invalid NPZ') from error
    require(all(v.dtype.kind in 'biuf' and np.isfinite(v).all() for v in result.values()),
            'NPZ must contain finite numeric arrays')
    return result


def contained(root, name):
    require(type(name) is str and Path(name).name == name and name not in ('', '.', '..'),
            'artifact must be a direct child')
    path = root / name
    require(stat.S_ISREG(path.lstat().st_mode) and path.resolve().parent == root,
            'artifact must be a contained nonsymlink regular file: ' + name)
    return path


class Reader:
    """Every artifact is physically opened at most once, for hash and parsing."""
    def __init__(self, root, check=lambda: None):
        self.root, self.check = Path(root).resolve(strict=True), check
        self.receipts, self.read, self.json_cache = {}, {}, {}
        self.total_bytes = 0

    def receipt(self, item):
        require(type(item) is dict and set(item) == {'path', 'size_bytes', 'sha256'}, 'receipt schema')
        require(type(item['size_bytes']) is int and 0 <= item['size_bytes'] <= 256 * 1024**2
                and is_hash(item['sha256']), 'receipt values')
        path = contained(self.root, item['path'])
        require(path.stat().st_size == item['size_bytes'], 'receipt size: ' + item['path'])
        require(item['path'] not in self.receipts or self.receipts[item['path']] == item,
                'conflicting receipt: ' + item['path'])
        self.receipts[item['path']] = item
        if item['path'] in self.read:
            require(self.read[item['path']] == item, 'already-read receipt mismatch')
        return path

    def payload(self, name, *, retain=True):
        self.check()
        require(name not in self.read, 'attempted second physical read: ' + name)
        path = contained(self.root, name)
        require(path.suffix in ('.json', '.npz', '.pt'), 'unsupported artifact')
        require(retain or path.suffix == '.pt', 'stream-only read must be opaque checkpoint')
        require(not (retain and path.suffix == '.pt'), 'checkpoint deserialization forbidden')
        size = path.stat().st_size
        require(size <= (32 if path.suffix == '.json' else 128) * 1024**2, 'file cap')
        require(self.total_bytes + size <= INPUT_CAP, 'cumulative input cap')
        digest, pieces, count = hashlib.sha256(), [], 0
        with path.open('rb') as handle:
            for block in iter(lambda: handle.read(1024**2), b''):
                self.check()
                digest.update(block)
                count += len(block)
                require(count <= size, 'artifact grew during read')
                if retain:
                    pieces.append(block)
        require(count == size, 'artifact size changed during read')
        item = {'path': name, 'size_bytes': size, 'sha256': digest.hexdigest()}
        self.read[name] = item
        self.total_bytes += size
        require(name == 'complete.json' or self.receipts.get(name) == item,
                'artifact hash mismatch: ' + name)
        return b''.join(pieces) if retain else None

    def json(self, name):
        require(Path(name).suffix == '.json', 'JSON suffix required')
        if name not in self.json_cache:
            self.json_cache[name] = parse_json(self.payload(name))
        return self.json_cache[name]

    def npz(self, name):
        require(Path(name).suffix == '.npz', 'NPZ suffix required')
        return parse_npz(self.payload(name))


def expected_roster():
    rows = []
    for si, seed in enumerate(SEEDS):
        for ci, cell in enumerate(CELLS):
            for ai, mode in enumerate(MODES):
                offset = (si + ci + ai) % 2
                for policy in POLICIES[offset:] + POLICIES[:offset]:
                    rows.append(dict(seed=seed, cell=cell, augmentation=mode, policy=policy))
    return rows


def identifier(identity):
    return 's{seed}-{cell}-{augmentation}-{policy}'.format(**identity)


def expected_files():
    files = {'manifest.json', 'results.json', 'complete.json'}
    for seed in SEEDS:
        files.update(f'{prefix}-s{seed}.{suffix}' for prefix, suffix in
                     [('plan', 'npz'), ('masks', 'npz'), ('plan', 'json'),
                      ('initial', 'pt'), ('warmup', 'pt'), ('warmup-actions', 'json'),
                      ('common-logits', 'npz'), ('pairing', 'json')])
    for identity in expected_roster():
        files.update(f'{prefix}-{identifier(identity)}.{suffix}' for prefix, suffix in
                     [('curve', 'json'), ('actions', 'json'), ('logits', 'npz'), ('final', 'pt')])
    return files


def validate_plan(plan, dimensions=Dimensions()):
    d, n = dimensions, dimensions.train
    integer_shapes = {'train_ids': (n,), 'heldout_ids': (d.heldout,), 'train_true': (n,),
        'heldout_true': (d.heldout,), 'warmup_batches': (100, d.batch),
        'continuation_batches': (1900, d.batch), 'diffuse_replacement_labels': (n,),
        'diffuse_targets': (n,), 'shared_sham_targets': (n,)}
    boolean_names = {'diffuse_selected', 'poison_mask', 'sham_patch_mask'}
    require(set(plan) == set(integer_shapes) | boolean_names, 'plan keys')
    for name, shape in integer_shapes.items():
        array(plan[name], np.int64, shape, name)
    for name in boolean_names:
        array(plan[name], np.bool_, (n,), name)
    train_ids, held_ids = plan['train_ids'], plan['heldout_ids']
    for ids in (train_ids, held_ids):
        require(np.all((ids >= 0) & (ids < 60000)) and len(np.unique(ids)) == len(ids), 'unique IDX IDs')
    require(not np.intersect1d(train_ids, held_ids).size, 'train/heldout overlap')
    true, held = plan['train_true'], plan['heldout_true']
    wanted = np.repeat(np.arange(10), [d.rare_train if z == 8 else d.common_train for z in range(10)])
    require(np.array_equal(true, wanted) and np.array_equal(held, np.repeat(np.arange(10), d.heldout_per_class)),
            'ordered true-label populations')
    for name in ('warmup_batches', 'continuation_batches'):
        require(np.all((plan[name] >= 0) & (plan[name] < n)), 'batch positions')
    require(not np.any(true[plan['warmup_batches']] == 8), 'rare warmup occurrence')
    poison, sham = plan['poison_mask'], plan['sham_patch_mask']
    require(int(poison.sum()) == d.poison_count and not poison[np.isin(true, [0, 8])].any(), 'poison membership')
    expected = np.where(poison, 0, true)
    require(np.array_equal(plan['shared_sham_targets'], expected), 'assigned labels')
    require(not sham[np.isin(true, [0, 8])].any(), 'sham eligibility')
    allocations = []
    for digit in (1, 2, 3, 4, 5, 6, 7, 9):
        group = true == digit
        count = int((group & poison).sum())
        require(int((group & sham).sum()) == count, 'sham class patch count')
        populations = [int((group & (poison == z)).sum()) for z in (0, 1)]
        qr = [divmod(count * v, sum(populations)) for v in populations]
        quotas = [q for q, _ in qr]
        for z in sorted(range(2), key=lambda z: (-qr[z][1], z))[:count - sum(quotas)]:
            quotas[z] += 1
        for z in (0, 1):
            require(int((group & sham & (poison == z)).sum()) == quotas[z], 'sham integer allocation')
            ideal = count * populations[z] / sum(populations)
            allocations.append({'true_digit': digit, 'poison': z, 'population': populations[z],
                'shared_patch_count_for_digit': count, 'sham_patch_count': quotas[z],
                'ideal_count': ideal, 'rounding_deviation': quotas[z] - ideal})
    require(np.all((plan['diffuse_targets'] >= 0) & (plan['diffuse_targets'] < 10))
            and np.array_equal(plan['diffuse_targets'][true == 8], true[true == 8]), 'unused diffuse labels')
    return allocations


def cell_values(plan, cell):
    true, poison = plan['train_true'], plan['poison_mask']
    assigned = true.copy() if cell == 'clean' else plan['shared_sham_targets'].copy()
    patch = np.zeros(len(true), dtype=bool) if cell == 'clean' else (
        poison.copy() if cell == 'shared' else plan['sham_patch_mask'].copy())
    contingency = np.zeros((2, 2, 10, 10), np.int64)
    np.add.at(contingency, (patch.astype(int), poison.astype(int), true, assigned), 1)
    counts = {'contingency_axes': ['patch', 'shared_poison_id', 'true_label', 'assigned_label'],
        'contingency_counts': contingency.tolist(), 'patch_count': int(patch.sum()),
        'patch_poison_overlap': int((patch & poison).sum()),
        'actually_changed_count': int((assigned != true).sum()), 'diffuse_selected_count': 0}
    return assigned, patch, counts


def validate_masks(values, dimensions=Dimensions()):
    shape = (1900, dimensions.batch)
    require(set(values) == {'gates', 'centers'} | {m + '_rectangles' for m in MODES}, 'mask keys')
    gates = array(values['gates'], np.bool_, shape, 'gates')
    centers = array(values['centers'], np.int16, shape + (2,), 'centers')
    require(np.all((centers >= 0) & (centers < 28)), 'center range')
    for mode in MODES:
        actual = array(values[mode + '_rectangles'], np.int16, shape + (4,), 'rectangles')
        expected = np.full(shape + (4,), -1, np.int16)
        if mode == 'random':
            r, c = centers[..., 0], centers[..., 1]
            bounds = np.stack((np.maximum(r - 4, 0), np.minimum(r + 4, 28),
                               np.maximum(c - 4, 0), np.minimum(c + 4, 28)), axis=-1)
            expected[gates] = bounds[gates]
        elif mode == 'targeted':
            expected[gates] = (0, 8, 0, 8)
        elif mode == 'opposite':
            expected[gates] = (20, 28, 20, 28)
        require(np.array_equal(actual, expected), 'mask geometry: ' + mode)


def coverage(rectangles, gates, batches, patch, true):
    result = {}
    for name, chosen in (('all', np.ones(batches.shape, dtype=bool)),
                          ('actually_cued', patch[batches]), ('rare', true[batches] == 8)):
        rect = rectangles[chosen].astype(np.int64)
        active = np.any(rect != -1, axis=1)
        r0, r1, c0, c1 = rect.T
        area = np.where(active, (r1 - r0) * (c1 - c0), 0)
        count = int(chosen.sum())
        result[name] = {'occurrences': count, 'active': int(active.sum()),
            'any_cue_coverage': int(np.sum(active & (r0 < 3) & (c0 < 3))),
            'full_cue_coverage': int(np.sum(active & (r0 == 0) & (c0 == 0) & (r1 >= 3) & (c1 >= 3))),
            'erased_area_total': int(area.sum()),
            'erased_area_counts': np.bincount(area, minlength=785).tolist(),
            'gate_active': int(gates[chosen].sum()), 'denominator': count}
    return result


def validate_actions(actions, policy, *, warmup=False):
    steps = list(range(1, 101)) if warmup else list(range(101, 2001))
    require(type(actions) is list and len(actions) == len(steps), 'action count')
    for row, step in zip(actions, steps):
        require(type(row['step']) is int and row['step'] == step, 'action step')
        observing = warmup or policy == 'native32'
        require(row['observer_step'] == (step if observing else None), 'observer counter')
        require(row['norm_law'] is None, 'unexpected norm control')
        for key in ('training_batch_ce', 'raw_norm', 'native_norm', 'applied_norm'):
            require(type(row[key]) in (float, int) and math.isfinite(row[key]) and row[key] >= 0,
                    'finite action scalar')
        source = 'raw_norm' if policy == 'raw' else 'native_norm'
        require(row['applied_norm'] == row[source], 'delivery norm identity')
        if observing:
            require(type(row['basis_rank']) is int and 0 <= row['basis_rank'] <= 32, 'observer rank')
        else:
            require(row['basis_rank'] is None, 'raw observer rank')
    if not warmup:
        binding = {key: value for key, value in actions[0].items() if key.endswith('sha256')}
        wanted = {'raw_gradient_sha256'} | ({'post_observe_tracker_sha256'} if policy == 'native32' else set())
        require(set(binding) == wanted and all(is_hash(v) for v in binding.values()), 'first-action hashes')
        return binding


def audit_saved(source_dir, *, dimensions=Dimensions(), check=lambda: None):
    """Audit a future completed acquisition; caller controls explicit execution."""
    reader, checks = Reader(source_dir, check), Checks()
    complete = reader.json('complete.json')
    require(complete['schema'] == SCHEMA and complete['status'] == 'complete', 'completion status')
    checks.equal(complete['completed_trajectories'], 72, 'completed trajectories')
    checks.equal(complete['completed_diagnostics'], 0, 'no diagnostics')
    require(type(complete['receipts']) is list, 'completion receipts')
    for item in complete['receipts']:
        require(item['path'] not in reader.receipts, 'duplicate completion receipt')
        reader.receipt(item)
    require(set(reader.receipts) == expected_files() - {'complete.json'}, 'exact receipt inventory')
    require({p.name for p in reader.root.iterdir()} == expected_files(), 'exact artifact inventory')
    checks.equal(sum(r['size_bytes'] for r in reader.receipts.values()),
                 complete['artifact_bytes_before_completion'], 'artifact byte accounting')
    manifest, results = reader.json('manifest.json'), reader.json('results.json')
    checks.equal(complete['results'], reader.receipts['results.json'], 'result binding')
    fixed = {'schema': SCHEMA, 'seeds': list(SEEDS), 'cells': list(CELLS), 'policies': list(POLICIES),
        'augmentations': list(MODES), 'warmup': 100, 'steps': 2000, 'batch_size': dimensions.batch,
        'eval_steps': list(EVAL), 'rank': 32, 'cloud_spend_usd': 0, 'diagnostics': [],
        'branch_roster': expected_roster()}
    for key, value in fixed.items():
        checks.equal(manifest[key], value, 'manifest/' + key)
    checks.equal(manifest['source_pins'], complete['source_pins'], 'source-pin bindings')
    require(bool(manifest['source_pins']) and all(is_hash(v) for v in manifest['source_pins'].values()), 'source hashes')
    checks.equal(manifest['data_pins'], DATA_PINS, 'declared IDX pins')
    checks.equal(complete['data_pins'], DATA_PINS, 'final IDX pins')
    require(results['schema'] == SCHEMA, 'result schema')
    rows = results['rows']
    require(type(rows) is list and len(rows) == 72, 'result row count')
    checks.equal([{k: row[k] for k in ('seed', 'cell', 'augmentation', 'policy')} for row in rows],
                 expected_roster(), 'result roster')
    rebuilt, evaluated = [], 0
    for seed in SEEDS:
        check()
        plan = reader.npz(f'plan-s{seed}.npz')
        masks = reader.npz(f'masks-s{seed}.npz')
        metadata = reader.json(f'plan-s{seed}.json')
        checks.equal(metadata['seed'], seed, 'plan seed')
        checks.equal(metadata['numpy_version'], manifest['numpy'], 'saved NumPy version')
        checks.equal(metadata['sham_allocation'], validate_plan(plan, dimensions), 'sham allocation')
        validate_masks(masks, dimensions)
        plan_hashes = {k: array_hash(v) for k, v in plan.items()}
        mask_hashes = {k: array_hash(v) for k, v in masks.items()}
        checks.equal(metadata['array_hashes'], plan_hashes, 'plan array hashes')
        checks.equal(metadata['mask_array_hashes'], mask_hashes, 'mask array hashes')
        validate_actions(reader.json(f'warmup-actions-s{seed}.json'), 'raw', warmup=True)
        common = reader.npz(f'common-logits-s{seed}.npz')
        require(set(common) == {'steps', 'heldout_unpatched', 'heldout_patched'} | {c + '_train' for c in CELLS},
                'common logits keys')
        array(common['steps'], np.int64, (2,), 'common steps')
        require(common['steps'].tolist() == [0, 100], 'common evaluation steps')
        for key in ('heldout_unpatched', 'heldout_patched'):
            array(common[key], np.float32, (2, dimensions.heldout, 10), key)
        for cell in CELLS:
            array(common[cell + '_train'], np.float32, (2, dimensions.train, 10), cell)
        shared_state, cell_input_hashes, first_raw = None, {}, {}
        for row in (r for r in rows if r['seed'] == seed):
            check()
            identity = {k: row[k] for k in ('seed', 'cell', 'augmentation', 'policy')}
            cell, mode, policy = row['cell'], row['augmentation'], row['policy']
            stem = identifier(identity)
            checks.equal(row['curve'], reader.receipts[f'curve-{stem}.json'], 'curve receipt')
            record = reader.json(row['curve']['path'])
            for key, value in {'schema': SCHEMA, **identity}.items():
                checks.equal(record[key], value, 'curve identity')
            for key, name in [('plan', f'plan-s{seed}.npz'), ('mask_plan', f'masks-s{seed}.npz'),
                              ('plan_metadata', f'plan-s{seed}.json'), ('initial_state', f'initial-s{seed}.pt'),
                              ('warmup_state', f'warmup-s{seed}.pt'), ('final_state', f'final-{stem}.pt'),
                              ('logits', f'logits-{stem}.npz'), ('actions', f'actions-{stem}.json')]:
                checks.equal(record[key], reader.receipts[name], 'record artifact binding/' + key)
            for key, hashes in [('plan_array_hashes', plan_hashes), ('mask_array_hashes', mask_hashes)]:
                checks.equal(record[key], hashes, key)
            require(record['full_state_fork_check'] == 'PASS' and record['diagnostics'] == [], 'fork/diagnostic claims')
            checks.equal(record['restored_state_sha256'], record['warmup_state_sha256'], 'restored state hash claim')
            states = [record['initial_state_sha256'], record['warmup_state_sha256']]
            require(all(is_hash(v) for v in states + [record['final_state_sha256'], record['cpu_cell_inputs_sha256']]),
                    'state/input hash schema')
            if shared_state is None:
                shared_state = states
            checks.equal(states, shared_state, 'paired state hash claims')
            cell_input_hashes.setdefault(cell, record['cpu_cell_inputs_sha256'])
            checks.equal(record['cpu_cell_inputs_sha256'], cell_input_hashes[cell], 'paired cell input hash claims')
            assigned, patch, counts = cell_values(plan, cell)
            checks.equal(record['assigned_targets_sha256'], array_hash(assigned), 'assigned target hash')
            checks.equal(record['cell_patch_mask_sha256'], array_hash(patch), 'cell patch hash')
            checks.equal(record['cell_counts'], counts, 'cell counts')
            checks.equal(record['coverage'], coverage(masks[mode + '_rectangles'], masks['gates'],
                plan['continuation_batches'], patch, plan['train_true']), 'occurrence coverage')
            actions = reader.json(record['actions']['path'])
            binding = validate_actions(actions, policy)
            checks.equal(record['first_action_binding'], binding, 'first action binding')
            pair = cell + '/' + mode
            first_raw.setdefault(pair, binding['raw_gradient_sha256'])
            checks.equal(binding['raw_gradient_sha256'], first_raw[pair], 'paired first raw gradient claims')
            logits = reader.npz(record['logits']['path'])
            require(set(logits) == {'steps', 'train', 'heldout_unpatched', 'heldout_patched'}, 'logit keys')
            array(logits['steps'], np.int64, (21,), 'logit steps')
            require(logits['steps'].tolist() == list(EVAL), 'logit step sequence')
            array(logits['train'], np.float32, (21, dimensions.train, 10), 'train logits')
            for key in ('heldout_unpatched', 'heldout_patched'):
                array(logits[key], np.float32, (21, dimensions.heldout, 10), key)
                require(np.array_equal(logits[key][:2], common[key]), 'shared initial/warmup heldout bytes')
            require(np.array_equal(logits['train'][:2], common[cell + '_train']), 'shared initial/warmup training bytes')
            require(type(record['curve']) is list and len(record['curve']) == 21, 'curve evaluation count')
            curve = []
            for index, step in enumerate(EVAL):
                check()
                value = evaluation_row(step, logits['train'][index], logits['heldout_unpatched'][index],
                    logits['heldout_patched'][index], plan['train_true'], assigned, plan['heldout_true'])
                checks.equal(record['curve'][index], value, f'{stem}/step{step}')
                curve.append(value)
                evaluated += 1
            checks.equal(row['endpoint'], curve[-1], 'result endpoint')
            checks.equal(row['warmup'], curve[1], 'result warmup')
            checks.equal(row['warmup_plus_branch_seconds'], record['warmup_plus_branch_seconds'], 'timing binding')
            rebuilt.append({**identity, 'endpoint': curve[-1], 'warmup': curve[1]})
            del logits, curve
        checks.equal(reader.json(f'pairing-s{seed}.json'), {'status': 'PASS', 'full_state_forks': 24,
                     'first_raw_gradient_sha256': first_raw}, 'seed pairing receipt')
    # Hash checkpoint bytes once without importing torch or decoding tensor data.
    for name in sorted(reader.receipts):
        if name.endswith('.pt'):
            reader.payload(name, retain=False)
    require(set(reader.read) == expected_files() and evaluated == 72 * 21, 'complete one-pass coverage')
    summary = summarize(rebuilt)
    if 'summary' in results:
        checks.equal(results['summary'], summary, 'saved summary')
    report = {'schema': AUDIT_SCHEMA, 'status': 'PASS', 'source_schema': SCHEMA,
        'source_completion_sha256': reader.read['complete.json']['sha256'],
        'source_results_sha256': reader.read['results.json']['sha256'],
        'source_pins': manifest['source_pins'], 'verified_artifacts': len(reader.read),
        'input_bytes_read_once': reader.total_bytes, 'logical_evaluation_records': evaluated,
        'trajectories': 72, 'paired_seeds': list(SEEDS), 'checks': checks.count,
        'metric_atol': ATOL, 'metric_rtol': RTOL, 'max_absolute_scalar_error': checks.max_error,
        'limits': ['No model, optimizer, gradient, or observer execution.',
                   'Checkpoint file bytes hashed; internal state tree hashes not independently recomputed.',
                   'State restores and first gradients checked as producer hash bindings.',
                   'Training IDX not read; true-label/source-ID association relies on acquisition pins.',
                   'Source pin equality checked between acquisition manifest and completion; no source replay.',
                   'Fixed contrasts use the shared pure analysis function; raw metrics are independently rebuilt.']}
    return report, summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--source-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    require(args.execute, 'No saved-array audit without explicit --execute')
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        require(os.environ.get(key) == '1', 'set ' + key + '=1 before Python')
    source = args.source_dir.resolve(strict=True)
    parent = args.output_dir.parent.resolve(strict=True)
    target = parent / args.output_dir.name
    require(not target.exists() and not target.is_symlink(), 'exclusive new audit output required')
    require(not target.is_relative_to(source) and not source.is_relative_to(target), 'source/output must be separate')
    target.mkdir(exist_ok=False)
    started = time.monotonic()
    def check():
        require(time.monotonic() - started < 540, '9-minute cooperative audit deadline')
        require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 <= 8 * 1024**3, '8 GiB audit RSS cap')
    try:
        report, summary = audit_saved(source, check=check)
        report['wall_seconds'] = time.monotonic() - started
        report['process_high_water_rss_kib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # The PASS receipt is last, so a failed summary write cannot look complete.
        outputs = {'summary.json': summary, 'audit.json': report}
        encoded = {name: (json.dumps(v, allow_nan=False, separators=(',', ':')) + '\n').encode()
                   for name, v in outputs.items()}
        require(sum(map(len, encoded.values())) <= OUTPUT_CAP, 'audit artifact cap')
        for name, payload in encoded.items():
            with (target / name).open('xb') as handle:
                handle.write(payload)
        print(json.dumps({'status': 'PASS', 'output': str(target), 'evaluations': 1512}), flush=True)
    except BaseException as error:
        payload = json.dumps({'schema': AUDIT_SCHEMA, 'status': 'FAIL', 'type': type(error).__name__,
            'message': str(error), 'wall_seconds': time.monotonic() - started}, allow_nan=False).encode()
        require(len(payload) <= 65536, 'failure metadata cap')
        with (target / 'failed.json').open('xb') as handle:
            handle.write(payload)
        raise


if __name__ == '__main__':
    main()
