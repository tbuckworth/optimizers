#!/usr/bin/env python3
"""Fixed rare-truth/shared-wrong-cue study. Import is inert; --execute is required.

Only the explicit entry point reads MNIST or runs the scientific acquisition.
There is no resume, sweep, adaptive selection or automatic retry path.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
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

DOCS = ROOT / 'output/2026-09-09-spectral-selectivity-boundary'
DATA = Path('data/MNIST/raw')
DATA_SHA256 = {
    'train-images-idx3-ubyte': 'ba891046e6505d7aadcbbe25680a0738ad16aec93bde7f9b65e87a2fc25776db',
    'train-labels-idx1-ubyte': '65a50cbbf4e906d70832878ad85ccda5333a97f0f4c3dd2ef09a8a9eef7101c5',
}
CORE_SHA = '706851828b7120bea0d7a41bd0fa75474ac800b2df71087941d0140c8d1970ec'
SEEDS = (202609111, 202609112, 202609113)
CELLS = ('clean', 'diffuse', 'shared', 'sham')
POLICIES = ('raw', 'native32', 'norm_raw')
MAJORITY = np.array([0, 1, 2, 3, 4, 5, 6, 7, 9], dtype=np.int64)
EVAL_STEPS = tuple(range(0, 2001, 100))
PROBE_STEPS = (101, 500, 2000)
WARMUP, STEPS, BATCH, P, RANK = 100, 2000, 64, 50890, 32
MAX_BYTES = 3 * 1024**3
RESERVE_BYTES = 1024**2  # Failure/footer only; free-disk reserve is separately 1 GiB.
DEADLINE_SECONDS = 1500
UNIT = 'spectral-selectivity-boundary-001.service'
SCHEMA = 'spectral_selectivity_boundary_v1'


def need(condition, message):
    if not condition:
        raise RuntimeError(message)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024**2), b''):
            value.update(chunk)
    return value.hexdigest()


def array_digest(value):
    value = np.ascontiguousarray(value)
    header = json.dumps([value.dtype.str, list(value.shape)], separators=(',', ':'))
    return hashlib.sha256(header.encode() + b'\n' + value.tobytes()).hexdigest()


def source_pins():
    paths = [ROOT / 'spectral_filter.py', Path(core.__file__), Path(__file__),
             ROOT / 'tests/test_spectral_selectivity_boundary.py', DOCS / 'protocol.md']
    pins = {str(path.relative_to(ROOT)): digest(path) for path in paths}
    need(pins['spectral_filter.py'] == core.FILTER_SHA256, 'canonical filter changed')
    need(pins[str(Path(core.__file__).relative_to(ROOT))] == CORE_SHA, 'accepted I9 core changed')
    return pins


def largest_remainder(total, populations):
    """Exact integer remainder ordering; ties favor ascending stratum index."""
    populations = [int(n) for n in populations]
    denominator = sum(populations)
    need(all(n >= 0 for n in populations) and denominator > 0 and 0 <= total <= denominator,
         'invalid allocation population')
    qr = [divmod(total * n, denominator) for n in populations]
    counts = [q for q, _ in qr]
    order = sorted(range(len(qr)), key=lambda z: (-qr[z][1], z))
    for z in order[:total - sum(counts)]:
        counts[z] += 1
    need(sum(counts) == total and all(0 <= k <= n for k, n in zip(counts, populations)), 'allocation failed')
    return counts


def balanced_sham(labels, poison, rng):
    patch = np.zeros(len(labels), dtype=bool)
    allocations = []
    for digit in (1, 2, 3, 4, 5, 6, 7, 9):
        group = labels == digit
        count = int((group & poison).sum())
        populations = [int((group & (poison == z)).sum()) for z in (0, 1)]
        need(sum(populations) > 0, 'missing eligible class')
        quotas = largest_remainder(count, populations)
        for z in (0, 1):
            pool = np.flatnonzero(group & (poison == z))
            if quotas[z]:
                patch[rng.choice(pool, size=quotas[z], replace=False)] = True
            ideal = count * populations[z] / sum(populations)
            allocations.append({'true_digit': digit, 'poison': z, 'population': populations[z],
                                'shared_patch_count_for_digit': count, 'sham_patch_count': quotas[z],
                                'ideal_count': ideal, 'rounding_deviation': quotas[z] - ideal})
        need(int((patch & group).sum()) == count, 'true-class patch count changed')
    need(int(patch.sum()) == int(poison.sum()) and not patch[np.isin(labels, [0, 8])].any(), 'sham eligibility')
    return patch, allocations


def make_plan(all_labels, seed, majority_train=550, rare_train=50, heldout_per_class=500,
              poison_count=500, warmup_steps=WARMUP, branch_steps=STEPS-WARMUP, batch_size=BATCH):
    """Shape overrides are for fabricated fixtures, never exposed by the CLI."""
    labels = np.asarray(all_labels)
    need(labels.ndim == 1 and labels.dtype.kind in 'iu' and ((labels >= 0) & (labels < 10)).all(), 'label domain')
    streams = [np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, s]))) for s in range(6)]
    train, heldout = [], []
    for digit in range(10):
        pool = streams[0].permutation(np.flatnonzero(labels == digit))
        count = rare_train if digit == 8 else majority_train
        need(len(pool) >= count + heldout_per_class, 'insufficient per-class examples')
        train.extend(pool[:count].tolist())
        heldout.extend(pool[count:count + heldout_per_class].tolist())
    train, heldout = np.array(train, dtype=np.int64), np.array(heldout, dtype=np.int64)
    clean = labels[train].astype(np.int64)
    majority = np.flatnonzero(clean != 8)
    warmup_batches = streams[1].choice(majority, size=(warmup_steps, batch_size), replace=True)
    batches = streams[2].integers(0, len(train), size=(branch_steps, batch_size))
    diffuse_selected = np.zeros(len(train), dtype=bool)
    diffuse_selected[majority] = streams[3].random(len(majority)) < .9
    replacement = np.full(len(train), -1, dtype=np.int64)
    replacement[majority] = streams[3].choice(MAJORITY, size=len(majority), replace=True)
    diffuse = clean.copy()
    diffuse[diffuse_selected] = replacement[diffuse_selected]
    eligible = np.flatnonzero(~np.isin(clean, [0, 8]))
    need(0 <= poison_count <= len(eligible), 'poison count')
    poison = np.zeros(len(train), dtype=bool)
    poison[streams[4].choice(eligible, size=poison_count, replace=False)] = True
    wrong = clean.copy()
    wrong[poison] = 0
    sham, allocations = balanced_sham(clean, poison, streams[5])
    arrays = {'train_ids': train, 'heldout_ids': heldout, 'train_true': clean,
              'heldout_true': labels[heldout].astype(np.int64), 'warmup_batches': warmup_batches,
              'continuation_batches': batches, 'diffuse_selected': diffuse_selected,
              'diffuse_replacement_labels': replacement, 'diffuse_targets': diffuse,
              'poison_mask': poison, 'shared_sham_targets': wrong, 'sham_patch_mask': sham}
    need(np.intersect1d(train, heldout).size == 0 and np.unique(train).size == len(train), 'split overlap')
    need(not (clean[warmup_batches] == 8).any(), 'rare example entered warmup')
    return arrays, allocations


def cell_definition(plan, cell):
    need(cell in CELLS, 'unknown cell')
    clean = plan['train_true']
    targets = (plan['diffuse_targets'] if cell == 'diffuse' else
               plan['shared_sham_targets'] if cell in ('shared', 'sham') else clean).copy()
    patch = (plan['poison_mask'] if cell == 'shared' else
             plan['sham_patch_mask'] if cell == 'sham' else np.zeros(len(clean), dtype=bool)).copy()
    need(np.array_equal(targets[clean == 8], clean[clean == 8]) and not patch[clean == 8].any(), 'rare truth altered')
    if cell in ('shared', 'sham'):
        need(not patch[clean == 0].any(), 'zero-class patch in training')
    contingency = np.zeros((2, 2, 10, 10), dtype=np.int64)
    np.add.at(contingency, (patch.astype(int), plan['poison_mask'].astype(int), clean, targets), 1)
    return targets, patch, {'contingency_axes': ['patch', 'shared_poison_id', 'true_label', 'assigned_label'],
                            'contingency_counts': contingency.tolist(), 'patch_count': int(patch.sum()),
                            'patch_poison_overlap': int((patch & plan['poison_mask']).sum()),
                            'actually_changed_count': int((targets != clean).sum()),
                            'diffuse_selected_count': int(plan['diffuse_selected'].sum()) if cell == 'diffuse' else 0}


def patch_images(images, mask=None):
    need(images.ndim == 2 and images.shape[1] == 784, 'patch image shape')
    before = images.clone()
    result = images.clone()
    selected = torch.ones(len(images), dtype=torch.bool, device=images.device) if mask is None else torch.as_tensor(mask, device=images.device)
    need(selected.dtype == torch.bool and selected.shape == (len(images),), 'patch mask')
    view = result.reshape(-1, 28, 28)
    view[selected, :3, :3] = 1.
    need(torch.equal(images, before), 'patch construction mutated its source')
    return result


def normalized_inputs(images, ids, device):
    """Canonical pixel conversion is CPU NumPy FP32 division before transfer."""
    values = np.ascontiguousarray(images[ids].astype(np.float32) / np.float32(255))
    need(values.ndim == 2 and values.shape[1] == 784 and values.dtype == np.float32,
         'normalized input schema')
    return torch.from_numpy(values).to(device)


def classification_stats(logits, labels, mask=None):
    logits = torch.as_tensor(logits, dtype=torch.float64, device='cpu')
    labels = torch.as_tensor(labels, dtype=torch.long, device='cpu')
    need(logits.ndim == 2 and logits.shape == (len(labels), 10) and bool(torch.isfinite(logits).all()), 'finite eval logits')
    if mask is not None:
        chosen = torch.as_tensor(mask, dtype=torch.bool, device='cpu')
        logits, labels = logits[chosen], labels[chosen]
    count = len(labels)
    if not count:
        return {'count': 0, 'correct': 0, 'ce_sum': 0., 'accuracy': None, 'ce': None}
    correct = int((logits.argmax(1) == labels).sum())
    loss = float(F.cross_entropy(logits, labels, reduction='sum'))
    return {'count': count, 'correct': correct, 'ce_sum': loss, 'accuracy': correct/count, 'ce': loss/count}


def group_stats(logits, labels):
    labels = np.asarray(labels)
    per_class = {str(d): classification_stats(logits, labels, labels == d) for d in range(10)}
    def macro(digits):
        rows = [per_class[str(d)] for d in digits]
        need(all(row['count'] > 0 for row in rows), 'missing evaluation class')
        return {metric: math.fsum(row[metric] for row in rows)/len(rows) for metric in ('accuracy', 'ce')}
    return {'per_class': per_class, 'rare': per_class['8'], 'majority_macro': macro(MAJORITY),
            'balanced_total': macro(range(10)), 'micro': classification_stats(logits, labels)}


def cue_stats(unpatched, patched, labels):
    labels = np.asarray(labels)
    before, after = np.asarray(unpatched).argmax(1), np.asarray(patched).argmax(1)
    result = {}
    masks = {'majority_nonzero': ~np.isin(labels, [0, 8]), 'rare': labels == 8, 'all_nonzero': labels != 0}
    for name, mask in masks.items():
        count = int(mask.sum())
        clean_zero, patch_zero = int((before[mask] == 0).sum()), int((after[mask] == 0).sum())
        result[name] = {'count': count, 'unpatched_target0_count': clean_zero, 'patched_target0_count': patch_zero,
                        'unpatched_target0_rate': clean_zero/count if count else None,
                        'patched_target0_rate': patch_zero/count if count else None,
                        'patch_excess': (patch_zero-clean_zero)/count if count else None}
    return result


def evaluation_row(step, train_logits, unpatched, patched, true, assigned, heldout_true):
    return {'step': step, 'train_true': group_stats(train_logits, true),
            'train_assigned': classification_stats(train_logits, assigned),
            'train_actually_changed': classification_stats(train_logits, assigned, assigned != true),
            'heldout_unpatched': group_stats(unpatched, heldout_true),
            'heldout_patched': group_stats(patched, heldout_true),
            'cue': cue_stats(unpatched, patched, heldout_true)}


@torch.no_grad()
def predict(model, images):
    logits = model(images).detach().cpu()
    need(logits.dtype == torch.float32 and bool(torch.isfinite(logits).all()), 'finite FP32 predictions')
    return logits.numpy()


def norm_direction(raw, native):
    need(raw.dtype == native.dtype and raw.shape == native.shape and bool(torch.isfinite(raw).all())
         and bool(torch.isfinite(native).all()), 'finite matching action inputs')
    raw_norm = float(raw.double().norm())
    target = float(native.double().norm())
    if raw_norm == 0:
        need(target == 0, 'nonzero native target with zero raw gradient')
        applied, scale = torch.zeros_like(raw), None
    elif target == 0:
        applied, scale = torch.zeros_like(raw), 0.
    else:
        scale = target / raw_norm
        applied = (raw.double() * scale).to(raw.dtype)
    need(bool(torch.isfinite(applied).all()), 'nonfinite norm-law delivery')
    achieved = float(applied.double().norm())
    mismatch = abs(achieved-target)/target if target else 0.
    tolerance = 10 * torch.finfo(raw.dtype).eps
    need((target > 0 and mismatch <= tolerance) or (target == 0 and achieved == 0), 'post-cast norm law failed')
    return applied, {'raw_norm': raw_norm, 'target_norm': target, 'delivered_norm': achieved,
                     'scale': scale, 'relative_postcast_mismatch': mismatch,
                     'relative_tolerance': tolerance, 'zero_raw': raw_norm == 0, 'zero_target': target == 0}


def observe(tracker, raw, step):
    need(tracker.step_count == step-1, 'observer must advance exactly once')
    tracker.step_count += 1
    tracker._update_svd(raw)


def probe_inputs(plan, assigned, cell_images, clean_images):
    true = plan['train_true']
    definitions = {'common3': (np.flatnonzero(true == 3)[:32], clean_images, true),
                   'rare8': (np.flatnonzero(true == 8)[:32], clean_images, true)}
    wrong = np.flatnonzero(assigned != true)[:32]
    definitions['wrong_assigned'] = (wrong, cell_images, assigned) if len(wrong) else None
    definitions['wrong_corrected'] = (wrong, cell_images, true) if len(wrong) else None
    result = {}
    for name, definition in definitions.items():
        if definition is None:
            result[name] = None
            continue
        positions, images, target = definition
        result[name] = {'training_positions': positions.copy(), 'source_ids': plan['train_ids'][positions].copy(),
                        'x': images[torch.as_tensor(positions, device=images.device)],
                        'targets': np.asarray(target[positions], dtype=np.int64),
                        'true_targets': true[positions].copy()}
    return result


def per_example_probe(model, x, targets, check=lambda: None):
    """autograd.grad never accumulates into training .grad; caller verifies state."""
    target = torch.as_tensor(targets, dtype=torch.long, device=x.device)
    need(len(x) == len(target) and len(x) > 0, 'nonempty probe')
    logits = model(x)
    losses = F.cross_entropy(logits, target, reduction='none')
    rows = []
    for index in range(len(x)):
        check()
        parts = torch.autograd.grad(losses[index], tuple(model.parameters()),
                                    retain_graph=index < len(x)-1, create_graph=False, allow_unused=False)
        rows.append(torch.cat([part.detach().reshape(-1) for part in parts]).cpu())
    gradients = torch.stack(rows)
    need(gradients.dtype == torch.float32 and bool(torch.isfinite(gradients).all())
         and bool(torch.isfinite(logits).all()), 'finite probe tensors')
    return logits.detach().cpu(), gradients


def collect_pre_probes(model, optimizer, tracker, definitions, check=lambda: None):
    before = core.snapshot(model, optimizer, tracker)
    groups = {}
    for name, definition in definitions.items():
        if definition is None:
            groups[name] = None
            continue
        logits, gradients = per_example_probe(model, definition['x'], definition['targets'], check)
        groups[name] = {'training_positions': torch.from_numpy(definition['training_positions'].copy()),
                        'source_ids': torch.from_numpy(definition['source_ids'].copy()),
                        'inputs': definition['x'].detach().cpu().clone(),
                        'targets': torch.from_numpy(definition['targets'].copy()),
                        'true_targets': torch.from_numpy(definition['true_targets'].copy()),
                        'pre_logits': logits, 'per_example_gradients': gradients}
    after = core.snapshot(model, optimizer, tracker)
    need(core.equal_tree(before, after), 'pre-update probes changed model/grad/Adam/observer/RNG')
    return before, groups


def numerical_basis(tracker):
    if tracker.V is None:
        return None, {'rank': 0, 'reason': 'basis_unavailable', 'singular_values': [], 'tolerance': None}
    v = tracker.V.detach().double().cpu()
    u, singular, _ = torch.linalg.svd(v, full_matrices=False)
    tolerance = torch.finfo(torch.float64).eps * max(v.shape) * float(singular[0])
    keep = singular > tolerance
    q = u[:, keep]
    return q, {'rank': int(keep.sum()), 'reason': None, 'singular_values': singular.tolist(),
               'tolerance': tolerance, 'native_gram_max_abs_error': float((v.T @ v-torch.eye(v.shape[1])).abs().max())}


def gradient_geometry(rows, tracker):
    rows64 = rows.double()
    energies = rows64.square().sum(1)
    mean = rows.mean(0)
    mean_energy = float(rows64.mean(0).square().sum())
    average_energy = float(energies.mean())
    device = tracker.param_list[0].device
    delivered = torch.stack([tracker._project_gradient(row.to(device)).detach().cpu() for row in rows])
    applied_energy = delivered.double().square().sum(1)
    applied_mean = tracker._project_gradient(mean.to(device)).detach().double().cpu()
    mean_native_denominator = float(mean.double().square().sum())
    return {'count': len(rows), 'mean_gradient_squared_norm_float64': mean_energy,
            'mean_individual_squared_norm': average_energy,
            'coherence': mean_energy/average_energy if average_energy else None,
            'zero_energy_reason': None if average_energy else 'zero_individual_energy',
            'individual_squared_norms': energies.tolist(),
            'individual_native_squared_norms': applied_energy.tolist(),
            'individual_native_retention': [float(a/e) if e else None for a, e in zip(applied_energy, energies)],
            'energy_weighted_native_retention': float(applied_energy.sum()/energies.sum()) if average_energy else None,
            'native_action_of_fp32_mean_squared_norm': float(applied_mean.square().sum()),
            'fp32_mean_squared_norm': mean_native_denominator,
            'native_mean_retention': float(applied_mean.square().sum())/mean_native_denominator if mean_native_denominator else None,
            'basis_identity_fallback': tracker.V is None}


def movement_summary(before, after, q):
    names = list(before['model_state'])
    old = torch.cat([before['model_state'][name].reshape(-1) for name in names])
    new = torch.cat([after['model_state'][name].reshape(-1) for name in names])
    groups = before['optimizer']['param_groups']
    need(len(groups) == 1 and groups[0]['foreach'] is False and groups[0]['fused'] is False, 'fixed AdamW group required')
    factor = 1-groups[0]['lr']*groups[0]['weight_decay']
    decayed = old * factor  # FP32 multiplication, matching the first AdamW operation.
    vectors = {'total': new.double()-old.double(), 'decay': decayed.double()-old.double(),
               'adaptive': new.double()-decayed.double()}
    summary = {}
    for name, delta in vectors.items():
        energy = float(delta.square().sum())
        outside = None if q is None else float((delta-q @ (q.T @ delta)).square().sum())
        summary[name] = {'squared_norm': energy, 'norm': math.sqrt(energy),
                         'outside_span_squared_norm': outside,
                         'outside_span_fraction': outside/energy if outside is not None and energy else None}
    return vectors, summary


def finish_diagnostic(model, optimizer, tracker, definitions, before, groups, raw, applied,
                      step, identity, check=lambda: None):
    after = core.snapshot(model, optimizer, tracker)
    for name, group in groups.items():
        if group is not None:
            check()
            group['post_logits'] = torch.from_numpy(predict(model, definitions[name]['x']).copy())
    need(core.equal_tree(after, core.snapshot(model, optimizer, tracker)), 'post-update probes changed live state')
    q, basis_info = numerical_basis(tracker)
    vectors, movement = movement_summary(before, after, q)
    summaries = {}
    for name, group in groups.items():
        if group is None:
            summaries[name] = {'status': 'absent', 'reason': 'no_actually_changed_examples', 'count': 0}
            continue
        gradient = group['per_example_gradients']
        pre = classification_stats(group['pre_logits'], group['targets'])
        post = classification_stats(group['post_logits'], group['targets'])
        summaries[name] = {'status': 'defined', 'geometry': gradient_geometry(gradient, tracker),
                           'true_label_counts': torch.bincount(group['true_targets'], minlength=10).tolist(),
                           'pre': pre, 'post': post, 'finite_ce_improvement': pre['ce']-post['ce'],
                           'signed_first_order_utilities': {key: -float(gradient.double().mean(0) @ value)
                                                            for key, value in vectors.items()}}
    residual = None
    if groups['wrong_assigned'] is not None:
        residual = gradient_geometry(groups['wrong_assigned']['per_example_gradients']
                                     - groups['wrong_corrected']['per_example_gradients'], tracker)
    need(core.equal_tree(after, core.snapshot(model, optimizer, tracker)), 'diagnostic geometry changed live state')
    payload = {'schema': SCHEMA, **identity, 'update': step,
               'probe_parameter_state': 'pre_update_after_one_training_observer_update',
               'before': before, 'after': after, 'raw_training_gradient': raw.detach().cpu().clone(),
               'applied_training_gradient': applied.detach().cpu().clone(), 'groups': groups,
               'parameter_order': [name for name, _ in model.named_parameters()]}
    report = {'schema': SCHEMA, **identity, 'update': step, 'before_sha256': core.tree_digest(before),
              'after_sha256': core.tree_digest(after), 'basis': basis_info, 'movement': movement,
              'groups': summaries, 'wrong_minus_true_gradient_geometry': residual,
              'residual_arithmetic_dtype': 'float32', 'probe_side_effect_checks': 'PASS'}
    return payload, report


def training_update(model, optimizer, tracker, x, targets, policy, step, definitions=None,
                    identity=None, check=lambda: None):
    need(policy in POLICIES, 'unknown policy')
    check()
    optimizer.zero_grad(set_to_none=True)
    loss = F.cross_entropy(model(x), targets)
    need(bool(torch.isfinite(loss)), 'nonfinite training loss')
    loss.backward()
    raw = core.flat_grad(model)
    if tracker is not None:
        observe(tracker, raw, step)
    native = raw if tracker is None else tracker._project_gradient(raw)
    applied, norm_record = raw, None
    if step > WARMUP and policy == 'native32':
        applied = native
    elif step > WARMUP and policy == 'norm_raw':
        applied, norm_record = norm_direction(raw, native)
    core.set_grad(model, applied)  # Explicit zero remains a real gradient.
    first = {}
    if step == 101:
        first['raw_gradient_sha256'] = core.tree_digest(raw.detach().cpu())
        if tracker is not None:
            first['post_observe_tracker_sha256'] = core.tree_digest(core._tracker_state(tracker))
    diagnostic = None
    before = groups = None
    if definitions is not None:
        need(policy == 'native32' and step in PROBE_STEPS, 'unexpected diagnostic')
        before, groups = collect_pre_probes(model, optimizer, tracker, definitions, check)
    optimizer.step()
    need(bool(torch.isfinite(core.flat_params(model)).all()), 'nonfinite updated parameters')
    if definitions is not None:
        diagnostic = finish_diagnostic(model, optimizer, tracker, definitions, before, groups, raw, applied,
                                       step, identity, check)
    optimizer.zero_grad(set_to_none=True)
    norm = lambda value: float(value.double().norm())
    row = {'step': step, 'training_batch_ce': float(loss.detach()), 'raw_norm': norm(raw),
           'native_norm': norm(native), 'applied_norm': norm(applied), 'norm_law': norm_record,
           'observer_step': None if tracker is None else tracker.step_count,
           'basis_rank': None if tracker is None else (0 if tracker.V is None else tracker.V.shape[1]),
           'native_identity_fallback': tracker is not None and tracker.V is None, **first}
    json.dumps(row, allow_nan=False)
    check()
    return row, diagnostic


def sample_summary(values):
    if all(value is None for value in values):
        return None
    need(len(values) == 3 and all(value is not None and math.isfinite(value) for value in values), 'three finite seed values')
    mean = math.fsum(values)/3
    sd = math.sqrt(math.fsum((value-mean)**2 for value in values)/2)
    return {'values': values, 'mean': mean, 'sample_sd': sd, 'sample_se': sd/math.sqrt(3),
            'positive_count': sum(value > 0 for value in values), 'negative_count': sum(value < 0 for value in values),
            'zero_count': sum(value == 0 for value in values)}


def endpoint_metrics(row):
    values = {group+'_'+metric: row['heldout_unpatched'][group][metric]
              for group in ('rare', 'majority_macro', 'balanced_total') for metric in ('accuracy', 'ce')}
    values.update({'patched_'+group+'_'+metric: row['heldout_patched'][group][metric]
                   for group in ('rare', 'majority_macro', 'balanced_total') for metric in ('accuracy', 'ce')})
    for prefix, source in (('unpatched', 'heldout_unpatched'), ('patched', 'heldout_patched'),
                           ('train_true', 'train_true')):
        values.update({prefix+'_class'+str(digit)+'_'+metric: row[source]['per_class'][str(digit)][metric]
                       for digit in range(10) for metric in ('accuracy', 'ce')})
    values.update({'train_true_'+group+'_'+metric: row['train_true'][group][metric]
                   for group in ('rare', 'majority_macro', 'balanced_total') for metric in ('accuracy', 'ce')})
    values.update({'train_assigned_'+metric: row['train_assigned'][metric] for metric in ('accuracy', 'ce')})
    values.update({group+'_patch_excess': row['cue'][group]['patch_excess']
                   for group in ('majority_nonzero', 'rare', 'all_nonzero')})
    values['majority_nonzero_patched_asr'] = row['cue']['majority_nonzero']['patched_target0_rate']
    values['train_wrong_accuracy'] = row['train_actually_changed']['accuracy']
    values['train_wrong_ce'] = row['train_actually_changed']['ce']
    return values


def summarize_results(rows):
    roster = {(row['seed'], row['cell'], row['policy']): row for row in rows}
    need(len(roster) == len(rows) == 36 and set(roster) == {(s, c, p) for s in SEEDS for c in CELLS for p in POLICIES}, 'exact complete result roster')
    groups, contrasts, acquisition = {}, {}, {}
    for cell in CELLS:
        for policy in POLICIES:
            values = [endpoint_metrics(roster[seed, cell, policy]['endpoint']) for seed in SEEDS]
            groups[cell+'/'+policy] = {key: sample_summary([value[key] for value in values]) for key in values[0]}
            warmup = [endpoint_metrics(roster[seed, cell, policy]['warmup']) for seed in SEEDS]
            acquisition[cell+'/'+policy] = {key: sample_summary([
                value[key]-start[key] if value[key] is not None and start[key] is not None else None
                for value, start in zip(values, warmup)]) for key in values[0]}
        for control in ('raw', 'norm_raw'):
            values = []
            for seed in SEEDS:
                left = endpoint_metrics(roster[seed, cell, 'native32']['endpoint'])
                right = endpoint_metrics(roster[seed, cell, control]['endpoint'])
                values.append({key: left[key]-right[key] if left[key] is not None else None for key in left})
            contrasts[cell+'/native32_minus_'+control] = {key: sample_summary([value[key] for value in values]) for key in values[0]}
    interactions = {}
    for policy in ('native32', 'norm_raw'):
        for metric in ('majority_nonzero_patch_excess', 'majority_nonzero_patched_asr'):
            deltas = []
            for seed in SEEDS:
                e = lambda p, c: endpoint_metrics(roster[seed, c, p]['endpoint'])[metric]
                deltas.append((e(policy, 'shared')-e(policy, 'sham'))-(e('raw', 'shared')-e('raw', 'sham')))
            interactions[policy+'_minus_raw/'+metric] = sample_summary(deltas)
    raw_assay = {metric: sample_summary([endpoint_metrics(roster[seed, 'shared', 'raw']['endpoint'])[metric]
                                        - endpoint_metrics(roster[seed, 'sham', 'raw']['endpoint'])[metric]
                                        for seed in SEEDS])
                 for metric in ('majority_nonzero_patch_excess', 'majority_nonzero_patched_asr')}
    return {'seeds': list(SEEDS), 'endpoint_step': 2000, 'per_group': groups, 'policy_contrasts': contrasts,
            'change_from_warmup': acquisition, 'cue_interactions': interactions,
            'continuous_raw_cue_assay': raw_assay, 'selection': 'fixed_step_2000_no_checkpoint_selection'}


def byte_inventory():
    # Deliberately assumes every snapshot has a full rank-32 observer and gradients,
    # including raw endpoints and cleared-gradient warmups where that overcounts.
    full_state = 37 * P * 4 + 32 * 8 + 4 * 4 + 64 * 1024
    parts = {'39_warmup_and_final_full_states': 39 * full_state,
             '72_diagnostic_pre_post_full_states': 72 * full_state,
             '4032_per_example_fp32_gradients': 4032 * P * 4,
             'diagnostic_raw_and_applied_gradients': 36 * 2 * P * 4,
             'diagnostic_inputs_targets_pre_post_logits': 4032 * (784*4 + 2*8 + 2*10*4),
             '36_full_logit_histories': 36 * 21 * 3 * 5000 * 10 * 4,
             'shared_initial_warmup_heldout_logits': 3 * 2 * 2 * 5000 * 10 * 4,
             'plans_scalar_json_zip_headers_and_metadata_allowance': 256 * 1024**2}
    total = sum(parts.values())
    need(total + RESERVE_BYTES < MAX_BYTES, 'analytic output inventory exceeds cap')
    return {'component_upper_bytes': parts, 'total_upper_bytes': total, 'cap_bytes': MAX_BYTES,
            'cap_headroom_bytes': MAX_BYTES-total, 'failure_metadata_reserve_bytes': RESERVE_BYTES}


class CappedWriter:
    def __init__(self, handle, allowance, check=lambda: None):
        self.handle, self.allowance, self.check = handle, allowance, check

    def write(self, value):
        self.check()
        size = memoryview(value).nbytes
        need(size <= self.allowance, 'output byte cap exhausted')
        written = self.handle.write(value)
        self.allowance -= written
        return written

    def flush(self):
        self.handle.flush()

    def read(self, *args):
        # NumPy identifies file-like objects via read; writes remain capped.
        return self.handle.read(*args)


class Run:
    def __init__(self, path, device='cuda'):
        self.path, self.device = Path(path), device
        self.started, self.last_resource_check = time.monotonic(), 0.
        self.used, self.receipts = 0, []

    def check(self):
        now = time.monotonic()
        need(now-self.started < DEADLINE_SECONDS, '25-minute cooperative deadline exceeded')
        if now-self.last_resource_check > 1:
            need(shutil.disk_usage(self.path).free > 1024**3, 'less than 1 GiB free reserve')
            if self.device == 'cuda':
                need(torch.cuda.max_memory_allocated() <= 8*1024**3, 'GPU allocation cap exceeded')
            self.last_resource_check = now

    def save(self, name, value, kind='json'):
        self.check()
        need(Path(name).name == name and name not in ('', '.', '..'), 'direct-child artifact required')
        target = self.path/name
        with target.open('xb') as handle:
            capped = CappedWriter(handle, MAX_BYTES-self.used-RESERVE_BYTES, self.check)
            if kind == 'tensor':
                torch.save(value, capped)
            elif kind == 'npz':
                np.savez(capped, **value)
            elif kind == 'json':
                capped.write((json.dumps(value, indent=2, allow_nan=False)+'\n').encode())
            else:
                raise ValueError('unknown artifact type')
        item = {'path': name, 'size_bytes': target.stat().st_size, 'sha256': digest(target)}
        self.used += item['size_bytes']
        self.receipts.append(item)
        self.check()
        return item


def read_training():
    raw = (DATA/'train-images-idx3-ubyte').read_bytes()
    need(len(raw) == 16+60000*784 and struct.unpack('>IIII', raw[:16]) == (2051, 60000, 28, 28), 'training image IDX identity')
    images = np.frombuffer(raw, dtype=np.uint8, offset=16).reshape(60000, 784).copy()
    raw = (DATA/'train-labels-idx1-ubyte').read_bytes()
    need(len(raw) == 60008 and struct.unpack('>II', raw[:8]) == (2049, 60000), 'training label IDX identity')
    labels = np.frombuffer(raw, dtype=np.uint8, offset=8).astype(np.int64)
    need(((labels >= 0) & (labels < 10)).all(), 'MNIST label domain')
    return images, labels


def sync(device):
    if device == 'cuda':
        torch.cuda.synchronize()


def acquire(run):
    images, all_labels = read_training()
    results = []
    for seed in SEEDS:
        run.check()
        plan, allocations = make_plan(all_labels, seed)
        plan_receipt = run.save(f'plan-s{seed}.npz', plan, 'npz')
        plan_hashes = {name: array_digest(value) for name, value in plan.items()}
        plan_metadata = run.save(f'plan-s{seed}.json', {'seed': seed, 'array_hashes': plan_hashes,
            'sham_allocation': allocations, 'rng': 'PCG64/SeedSequence([seed, stream_id])',
            'numpy_version': np.__version__, 'initialization_seed': seed})
        clean_x = normalized_inputs(images, plan['train_ids'], run.device)
        heldout_x = normalized_inputs(images, plan['heldout_ids'], run.device)
        patched_heldout_x = patch_images(heldout_x)
        cell_data = {}
        for cell in CELLS:
            assigned, patch, counts = cell_definition(plan, cell)
            x = patch_images(clean_x, patch) if patch.any() else clean_x
            cell_data[cell] = {'x': x, 'assigned': assigned, 'patch': patch, 'counts': counts}
        model = core.make_model(seed, run.device)
        need(sum(parameter.numel() for parameter in model.parameters()) == P, 'model specification changed')
        optimizer, tracker = core.make_optimizer(model), None
        tracker = core.make_tracker(model, optimizer)
        initial_sha = core.tree_digest(dict(model.state_dict()))
        initial_heldout = (predict(model, heldout_x), predict(model, patched_heldout_x))
        initial_train = {cell: predict(model, data['x']) for cell, data in cell_data.items()}
        sync(run.device)
        start = time.monotonic()
        warmup_actions = []
        for index, batch in enumerate(plan['warmup_batches'], 1):
            need(not (plan['train_true'][batch] == 8).any(), 'rare example in warmup')
            ids = torch.as_tensor(batch, device=run.device)
            target = torch.as_tensor(plan['train_true'][batch], device=run.device)
            row, diagnostic = training_update(model, optimizer, tracker, clean_x[ids], target,
                                               'raw', index, check=run.check)
            need(diagnostic is None, 'warmup diagnostic forbidden')
            warmup_actions.append(row)
        sync(run.device)
        warmup_seconds = time.monotonic()-start
        warmup = core.snapshot(model, optimizer, tracker)
        warmup_sha = core.tree_digest(warmup)
        warmup_receipt = run.save(f'warmup-s{seed}.pt', warmup, 'tensor')
        run.save(f'warmup-actions-s{seed}.json', warmup_actions)
        warmup_heldout = (predict(model, heldout_x), predict(model, patched_heldout_x))
        warmup_train = {cell: predict(model, data['x']) for cell, data in cell_data.items()}
        common_predictions = run.save(f'common-heldout-s{seed}.npz', {
            'steps': np.array([0, 100], dtype=np.int64),
            'unpatched': np.stack([initial_heldout[0], warmup_heldout[0]]),
            'patched': np.stack([initial_heldout[1], warmup_heldout[1]])}, 'npz')
        need(core.equal_tree(warmup, core.snapshot(model, optimizer, tracker)), 'warmup predictions altered state')
        del model, optimizer, tracker
        for cell in CELLS:
            data = cell_data[cell]
            target_all = torch.as_tensor(data['assigned'], device=run.device)
            binding = {'schema': SCHEMA, 'seed': seed, 'cell': cell, 'plan': plan_receipt,
                       'plan_metadata': plan_metadata, 'plan_array_hashes': plan_hashes,
                       'warmup': warmup_receipt, 'warmup_state_sha256': warmup_sha,
                       'initial_model_sha256': initial_sha, 'common_heldout': common_predictions,
                       'input_hash_convention': 'array_digest of contiguous CPU float32 [N,784]; NumPy uint8.astype(float32)/float32(255) before transfer; plan row order',
                       'train_inputs_sha256': array_digest(data['x'].detach().cpu().numpy()),
                       'train_clean_inputs_sha256': array_digest(clean_x.detach().cpu().numpy()),
                       'heldout_unpatched_inputs_sha256': array_digest(heldout_x.detach().cpu().numpy()),
                       'heldout_patched_inputs_sha256': array_digest(patched_heldout_x.detach().cpu().numpy()),
                       'assigned_targets_sha256': array_digest(data['assigned']),
                       'patch_mask_sha256': array_digest(data['patch']), 'counts': data['counts'],
                       'warmup_seconds': warmup_seconds}
            run.save(f'binding-s{seed}-{cell}.json', binding)
            offset = (SEEDS.index(seed)+CELLS.index(cell)) % len(POLICIES)
            order = POLICIES[offset:]+POLICIES[:offset]
            first_raw = first_observer = None
            for policy in order:
                run.check()
                model, optimizer, tracker = core.restore(warmup, run.device)
                need(core.tree_digest(core.snapshot(model, optimizer, tracker)) == warmup_sha, 'full-state fork identity failed')
                if policy == 'raw':
                    tracker = None
                definitions = probe_inputs(plan, data['assigned'], data['x'], clean_x) if policy == 'native32' else None
                train_history = [initial_train[cell], warmup_train[cell]]
                heldout_history = [initial_heldout[0], warmup_heldout[0]]
                patched_history = [initial_heldout[1], warmup_heldout[1]]
                curve = [evaluation_row(step, tr, va, pa, plan['train_true'], data['assigned'], plan['heldout_true'])
                         for step, tr, va, pa in zip((0, 100), train_history, heldout_history, patched_history)]
                actions, diagnostics = [], []
                identifier = f's{seed}-{cell}-{policy}'
                identity = {'seed': seed, 'cell': cell, 'policy': policy, 'warmup_state_sha256': warmup_sha}
                sync(run.device)
                started = time.monotonic()
                for step, batch in enumerate(plan['continuation_batches'], 101):
                    ids = torch.as_tensor(batch, device=run.device)
                    selected = definitions if policy == 'native32' and step in PROBE_STEPS else None
                    row, diagnostic = training_update(model, optimizer, tracker, data['x'][ids], target_all[ids],
                        policy, step, selected, identity, run.check)
                    actions.append(row)
                    if step == 101:
                        need(first_raw is None or first_raw == row['raw_gradient_sha256'], 'first branch raw gradient differs across policies')
                        first_raw = row['raw_gradient_sha256']
                        if tracker is not None:
                            need(first_observer is None or first_observer == row['post_observe_tracker_sha256'], 'first branch observer differs')
                            first_observer = row['post_observe_tracker_sha256']
                    if diagnostic is not None:
                        tensor, report = diagnostic
                        item = run.save(f'diagnostic-{identifier}-u{step:04d}.pt', tensor, 'tensor')
                        report['tensor'] = item
                        diagnostics.append(run.save(f'diagnostic-{identifier}-u{step:04d}.json', report))
                        del tensor, report, diagnostic
                    if step % 100 == 0:
                        train_logits, val, patched = predict(model, data['x']), predict(model, heldout_x), predict(model, patched_heldout_x)
                        train_history.append(train_logits)
                        heldout_history.append(val)
                        patched_history.append(patched)
                        curve.append(evaluation_row(step, train_logits, val, patched, plan['train_true'], data['assigned'], plan['heldout_true']))
                        print(json.dumps({**identity, 'step': step, 'metrics': endpoint_metrics(curve[-1]),
                                          'batch_elapsed_seconds': time.monotonic()-run.started}), flush=True)
                sync(run.device)
                branch_seconds = time.monotonic()-started
                final = core.snapshot(model, optimizer, tracker)
                need([row['step'] for row in curve] == list(EVAL_STEPS), 'complete evaluation roster')
                need(len(diagnostics) == (3 if policy == 'native32' else 0), 'complete diagnostic roster')
                record = {**binding, 'policy': policy, 'policy_order': list(order), 'curve': curve,
                          'branch_seconds_including_eval_and_diagnostics': branch_seconds,
                          'warmup_plus_branch_seconds': warmup_seconds+branch_seconds,
                          'final_state_sha256': core.tree_digest(final),
                          'final_state': run.save(f'final-{identifier}.pt', final, 'tensor'),
                          'logits': run.save(f'logits-{identifier}.npz', {'steps': np.array(EVAL_STEPS),
                              'train': np.stack(train_history), 'heldout_unpatched': np.stack(heldout_history),
                              'heldout_patched': np.stack(patched_history)}, 'npz'),
                          'actions': run.save(f'actions-{identifier}.json', actions), 'diagnostics': diagnostics,
                          'first_action_binding': {key: value for key, value in actions[0].items() if key.endswith('sha256')}}
                receipt = run.save(f'curve-{identifier}.json', record)
                results.append({'seed': seed, 'cell': cell, 'policy': policy, 'curve': receipt,
                                'endpoint': curve[-1], 'warmup': curve[1],
                                'warmup_plus_branch_seconds': warmup_seconds+branch_seconds})
                del model, optimizer, tracker, final, definitions, train_history, heldout_history, patched_history, actions
        del warmup, clean_x, heldout_x, patched_heldout_x, cell_data
    need(len(results) == 36, 'incomplete scientific roster')
    return run.save('results.json', {'schema': SCHEMA, 'rows': results, 'summary': summarize_results(results)})


def validate_bounds(effective, service):
    quota, period = effective['cpu.max'].split()
    need(effective['memory.max'] == str(16*1024**3) and effective['memory.swap.max'] == '0'
         and quota != 'max' and int(quota) == int(period) and int(period) > 0, 'unexpected cgroup resource limits')
    need(service == {'Type': 'exec', 'RuntimeMaxUSec': '30min', 'Restart': 'no', 'KillMode': 'control-group'}, 'unexpected service limits')


def configure():
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        need(os.environ.get(name) == '1', 'set '+name+'=1 before Python')
    need(os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8', 'deterministic CUDA workspace required')
    group = next(line.split(':', 2)[2] for line in Path('/proc/self/cgroup').read_text().splitlines() if line.startswith('0::'))
    need(Path(group).name == UNIT, 'unexpected experiment unit')
    root = Path('/sys/fs/cgroup')/group.lstrip('/')
    effective = {key: (root/key).read_text().strip() for key in ('memory.max', 'memory.swap.max', 'cpu.max')}
    properties = subprocess.check_output(['systemctl', '--user', 'show', UNIT,
        '--property=Type', '--property=RuntimeMaxUSec', '--property=Restart', '--property=KillMode'], text=True)
    service = dict(line.split('=', 1) for line in properties.splitlines())
    validate_bounds(effective, service)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    need(torch.__version__ == '2.11.0+cu128', 'unexpected torch version')
    need(torch.cuda.is_available() and torch.cuda.get_device_name() == 'NVIDIA GeForce RTX 3090', 'local RTX3090 required')
    need(torch.cuda.mem_get_info()[0] >= 8*1024**3, 'at least 8 GiB free GPU required')
    torch.cuda.set_per_process_memory_fraction(8*1024**3/torch.cuda.mem_get_info()[1])
    return {'cgroup': group, 'effective': effective, 'service': service}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true', help='explicit one-shot scientific authorization')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    need(args.execute, 'No scientific work without explicit --execute')
    parent = args.output_dir.parent.resolve(strict=True)
    need(parent.is_relative_to(Path('/tmp/spectral-experiment-artifacts')) and not parent.is_symlink()
         and not list(parent.iterdir()), 'unused large-volume parent required')
    need(args.output_dir == parent/'acquisition-001' and not args.output_dir.exists()
         and not args.output_dir.is_symlink(), 'new exclusive acquisition-001 required')
    mount = subprocess.check_output(['findmnt', '-n', '-o', 'TARGET,SOURCE', '--target', str(parent)], text=True).split()
    need(mount == ['/private-artifacts/storage', '/dev/RECONFIGURE_FOR_LOCAL_STORAGE'], 'wrong output mount identity')
    admission = configure()
    inventory, pins = byte_inventory(), source_pins()
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    for name, expected in pins.items():
        committed = subprocess.check_output(['git', 'show', commit+':'+name], cwd=ROOT)
        need(hashlib.sha256(committed).hexdigest() == expected, 'source not committed: '+name)
    data_pins = {name: digest(DATA/name) for name in DATA_SHA256}
    need(data_pins == DATA_SHA256, 'accepted original training IDX hashes differ')
    need(shutil.disk_usage(parent).free > 1024**3+inventory['total_upper_bytes'], 'insufficient admitted output reserve')
    args.output_dir.mkdir(exist_ok=False)
    run = Run(args.output_dir)
    try:
        manifest = {'schema': SCHEMA, 'source_pins': pins, 'data_pins': data_pins,
            'data_directory': str(DATA), 'git_commit': commit, 'resource_admission': admission,
            'pid': os.getpid(), 'invocation_id': os.environ.get('INVOCATION_ID'), 'started_unix': time.time(),
            'seeds': list(SEEDS), 'cells': list(CELLS), 'policies': list(POLICIES),
            'steps': STEPS, 'warmup': WARMUP, 'batch_size': BATCH, 'eval_steps': list(EVAL_STEPS),
            'probe_steps': list(PROBE_STEPS), 'rank': RANK, 'byte_inventory': inventory,
            'torch': torch.__version__, 'numpy': np.__version__, 'python': sys.version,
            'rng': 'numpy.random.PCG64 via SeedSequence([seed, stream_id])',
            'cooperative_seconds': DEADLINE_SECONDS, 'cloud_spend_usd': 0,
            'rare_class': 8, 'wrong_target': 0, 'poison_count': 500, 'majority_train_per_class': 550,
            'rare_train_count': 50, 'heldout_per_class': 500, 'observer': 'unchanged I9 current stable hard32',
            'timing_scope': 'branch includes evaluation, native-only diagnostics and diagnostic serialization; not speed comparison'}
        run.save('manifest.json', manifest)
        print(json.dumps({'resource_guard': 'PASS', 'unit': UNIT, 'output': str(args.output_dir),
                          'source_pins': pins, 'data_pins': data_pins, 'inventory': inventory,
                          'invocation_id': os.environ.get('INVOCATION_ID'), **admission}), flush=True)
        result = acquire(run)
        need(source_pins() == pins and {name: digest(DATA/name) for name in DATA_SHA256} == data_pins, 'sources/data changed during acquisition')
        run.save('complete.json', {'schema': SCHEMA, 'status': 'complete', 'results': result,
            'source_pins': pins, 'data_pins': data_pins, 'finished_unix': time.time(),
            'wall_seconds': time.monotonic()-run.started, 'process_high_water_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'gpu_max_allocated_bytes': torch.cuda.max_memory_allocated(), 'receipts': list(run.receipts),
            'artifact_bytes_before_completion': run.used, 'completed_trajectories': 36, 'completed_diagnostics': 36})
    except BaseException as exc:
        failure = {'schema': SCHEMA, 'status': 'failed', 'type': type(exc).__name__, 'message': str(exc),
                   'finished_unix': time.time(), 'wall_seconds': time.monotonic()-run.started,
                   'completed_receipts': run.receipts}
        payload = json.dumps(failure, indent=2, allow_nan=False).encode()
        need(len(payload) <= RESERVE_BYTES, 'failure metadata reserve exceeded')
        with (args.output_dir/'failed.json').open('xb') as handle:
            handle.write(payload)
        raise


if __name__ == '__main__':
    main()
