#!/usr/bin/env python3
"""Fixed common-state diagnostic. Import/default CLI perform no experiment.

Execution consumes one exclusive attempt under the separately admitted systemd
unit. It reuses twelve frozen native parents, never resumes their training.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import resource
import shutil
import stat
import subprocess
import sys
import time
import zipfile

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments import spectral_general_augmentation as base
from experiments import spectral_strong_augmentation_core as core
from experiments import spectral_strong_augmentation_data as data
from experiments import spectral_component_utility_actions as action_helper
from experiments import spectral_component_utility_objectives as objective
from experiments import spectral_component_utility_panels as panel_helper
from experiments import spectral_component_utility_restore as restore_helper

DOCS = ROOT / 'output/2026-09-10-spectral-component-utility'
INVENTORY = DOCS / 'parent-inventory.json'
OLD_AUDIT = ROOT / 'output/2026-09-10-spectral-strong-augmentation/audit.json'
SCHEMA = 'spectral_component_utility_v1'
UNIT = 'spectral-component-utility-001.service'
SEEDS = (202609171, 202609172, 202609173)
STEPS, AUGMENTATIONS = (100, 56304), ('none', 'translate')
FRACTIONS = ((1.0, 'full'), (0.1, 'tenth'))
KEYS = ('S', 'F', 'C', 'L', 'H_O', 'H_T')
ROLE_KEYS = ('train_ids', 'validation_ids', 'reporting_ids', 'train_labels',
             'validation_labels', 'reporting_labels', 'assigned_labels')
P = 235146
MAX_INPUT = 256 * 1024**2


def need(condition, message):
    if not condition:
        raise ValueError(message)


def roster():
    return [{'seed': seed, 'augmentation': aug, 'step': step,
             'parent_id': f's{seed}-{aug}-h{step:05d}'}
            for seed in SEEDS for aug in AUGMENTATIONS for step in STEPS]


def byte_inventory():
    parts = {'48_action_tensor_records': 48 * (9 * P * 4 + 2 * 6 * 8),
             '24_post_native_bases': 24 * P * 200 * 4,
             '24_post_native_singular_values': 24 * 200 * 8,
             '12_baseline_gradient_theta_sets': 12 * 7 * P * 4,
             '12_baseline_check_sets': 12 * (2 * 500 * 10 * 4 + 500 * 8),
             '144_path_records': 144 * P * 32,
             '156_logit_sets': 156 * (256 * 25 * 10 + 128 * 25 * 10 + 128 * 10) * 4,
             'metadata_container_panel_reserve': 256 * 1024**2}
    total = sum(parts.values())
    need(total + 1024**2 < 8 * 1024**3, 'archive budget exceeds hard cap')
    return {'component_upper_bytes': parts, 'total_upper_bytes': total,
            'failure_reserve_bytes': 1024**2, 'cap_bytes': 8 * 1024**3,
            'expected_receipted_artifacts': 376}


def read_pinned(path, expected_sha, *, expected_size=None, cap=16 * 1024**2):
    """Bounded read/hash on one regular non-symlink descriptor, no deserialization."""
    path = Path(path)
    need(path.is_absolute() and re.fullmatch(r'[0-9a-f]{64}', expected_sha), 'absolute pinned input required')
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, 'rb') as handle:
        before = os.fstat(handle.fileno())
        need(stat.S_ISREG(before.st_mode) and 0 < before.st_size <= cap, 'input type/byte cap')
        need(expected_size is None or before.st_size == expected_size, 'input receipt size differs')
        payload = handle.read(cap + 1)
        after = os.fstat(handle.fileno())
        need((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) ==
             (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns), 'input changed during read')
    need(len(payload) == before.st_size and hashlib.sha256(payload).hexdigest() == expected_sha,
         'input byte hash differs')
    return payload


def receipt_bytes(directory, receipt, cap=MAX_INPUT):
    need(type(receipt) is dict and set(receipt) == {'path', 'size_bytes', 'sha256'}, 'receipt schema')
    name = receipt['path']
    need(type(name) is str and Path(name).name == name and name not in ('', '.', '..'), 'direct input child required')
    need(type(receipt['size_bytes']) is int and 0 < receipt['size_bytes'] <= cap, 'receipt byte range')
    return read_pinned(Path(directory) / name, receipt['sha256'],
                       expected_size=receipt['size_bytes'], cap=cap)


def numeric_npz(payload, *, cap=MAX_INPUT):
    """Reject object data, compressed inflation and oversized declared members."""
    with zipfile.ZipFile(io.BytesIO(payload)) as z:
        entries = z.infolist()
        names = [i.filename for i in entries]
        need(0 < len(entries) <= 32 and len(set(names)) == len(names), 'NPZ member count/duplicates')
        need(all(Path(n).name == n and n.endswith('.npy') for n in names), 'NPZ direct numeric members')
        need(sum(i.file_size for i in entries) <= cap, 'NPZ expanded byte cap')
        for entry in entries:
            with z.open(entry) as member:
                version = np.lib.format.read_magic(member)
                need(version in ((1, 0), (2, 0)), 'bounded known NPY header version')
                reader = (np.lib.format.read_array_header_1_0 if version == (1, 0)
                          else np.lib.format.read_array_header_2_0)
                shape, _, dtype = reader(member, max_header_size=10000)
                need(dtype.kind in 'bifu' and dtype.fields is None and dtype.subdtype is None,
                     'numeric NPY header required')
                size = math.prod(shape) * dtype.itemsize
                need(len(shape) <= 8 and size <= cap and member.tell() + size == entry.file_size,
                     'NPY declared shape/payload bound')
    with np.load(io.BytesIO(payload), allow_pickle=False, max_header_size=10000) as source:
        values = {key: source[key] for key in source.files}
    need(all(v.dtype.kind in 'bifu' and v.dtype.fields is None and np.isfinite(v).all()
             for v in values.values()), 'plain finite numeric NPZ required')
    return values


def required_sources(inventory):
    suffixes = ('', '_actions', '_objectives', '_panels', '_restore', '_guard', '_audit')
    sources = set(inventory['source_pins'])
    sources.update(f'experiments/spectral_component_utility{s}.py' for s in suffixes)
    sources.update(f'tests/test_spectral_component_utility{s}.py' for s in suffixes)
    sources.update(str((DOCS / name).relative_to(ROOT)) for name in
                   ('protocol.md', 'archive-contract.md', 'parent-inventory.json', 'implementation-check.md'))
    return sources


def verify_manifest(path, expected_sha):
    manifest = json.loads(read_pinned(path, expected_sha))
    need(set(manifest) == {'schema', 'commit', 'source_pins', 'input_inventory_sha256'} and
         manifest['schema'] == 'spectral_component_utility_source_manifest_v1', 'source manifest schema')
    need(re.fullmatch(r'[0-9a-f]{40}', manifest['commit']), 'committed source identity required')
    inventory = json.loads(read_pinned(INVENTORY, manifest['input_inventory_sha256']))
    need(set(manifest['source_pins']) == required_sources(inventory), 'source set differs')
    need(manifest['source_pins'][str(INVENTORY.relative_to(ROOT))] == manifest['input_inventory_sha256'],
         'inventory/source manifest binding')
    for name, sha in manifest['source_pins'].items():
        need(not Path(name).is_absolute() and '..' not in Path(name).parts, 'source path escapes worktree')
        read_pinned(ROOT / name, sha)
        committed = subprocess.check_output(['git', 'show', f"{manifest['commit']}:{name}"], cwd=ROOT)
        need(hashlib.sha256(committed).hexdigest() == sha, 'source differs from frozen commit: ' + name)
    need(all(manifest['source_pins'].get(k) == v for k, v in inventory['source_pins'].items()),
         'old source changed')
    return manifest, inventory


def verify_input_metadata(inventory):
    """Bind exact parents/plans/readouts through original producer and PASS audit."""
    need(inventory['schema'] == 'spectral_component_utility_parent_inventory_v1', 'inventory schema')
    archive = Path(inventory['archive'])
    need(archive.is_absolute() and archive.resolve() == archive, 'original archive identity')
    old = json.loads(read_pinned(archive / 'results.json', inventory['source_results_sha256']))
    audit = json.loads(read_pinned(OLD_AUDIT, inventory['source_audit_sha256']))
    need(old['schema'] == 'spectral_strong_augmentation_v1' and old['status'] == 'complete', 'original producer incomplete')
    need(audit['schema'] == 'spectral_strong_augmentation_audit_v1' and audit['status'] == 'PASS' and
         audit['input_results_sha256'] == inventory['source_results_sha256'] and
         audit['input_dir'] == str(archive), 'original audit/results/archive binding')
    for value in (old, audit):
        need(value['source_pins'] == inventory['source_pins'] and value['data_pins'] == inventory['data_pins'],
             'original source/data pins differ')
    need(inventory['data_pins'] == base.DATA_PINS, 'canonical data pins differ')
    receipts = {r['path']: r for r in old['receipts']}
    need(len(receipts) == len(old['receipts']), 'duplicate original receipts')
    expected = [(seed, aug) for seed in SEEDS for aug in AUGMENTATIONS]
    need([(r['seed'], r['augmentation']) for r in inventory['parents']] == expected, 'native inventory roster differs')
    branches = {(b['seed'], b['augmentation']): b for b in old['branches'] if b['policy'] == 'native200'}
    need(set(branches) == set(expected), 'original native branch roster differs')
    for row in inventory['parents']:
        branch = branches[row['seed'], row['augmentation']]
        for stage, step in zip(('warmup', 'final'), STEPS):
            need(row[stage] == branch[stage + '_receipt'], 'parent checkpoint binding differs')
            match = [m['readout_receipt'] for m in branch['metrics'] if m['step'] == step]
            need(match == [row[stage + '_readout']], 'parent readout binding differs')
            for receipt in (row[stage], row[stage + '_readout']):
                need(receipts.get(receipt['path']) == receipt, 'original artifact receipt differs')
        need(row['plan_npz'] == f"plan-s{row['seed']}.npz" and
             row['plan_json'] == f"plan-s{row['seed']}.json", 'seed/plan binding')
    expected_plans = [f'plan-s{s}.{ext}' for s in SEEDS for ext in ('npz', 'json')]
    need([r['path'] for r in inventory['plan_receipts']] == expected_plans, 'plan receipt roster')
    need(all(receipts.get(r['path']) == r for r in inventory['plan_receipts']), 'plan receipts differ')
    return inventory


def load_plan(inventory, seed, truth):
    by_name = {r['path']: r for r in inventory['plan_receipts']}
    plan = numeric_npz(receipt_bytes(inventory['archive'], by_name[f'plan-s{seed}.npz']))
    header = json.loads(receipt_bytes(inventory['archive'], by_name[f'plan-s{seed}.json'], 1024**2))
    need(set(plan) == set(data.PLAN_KEYS) and header['seed'] == seed and
         header['numpy_version'] == '1.26.4', 'source plan schema/version')
    need(header['array_hashes'] == {k: base.array_digest(v) for k, v in plan.items()}, 'source plan array hashes')
    for role in ('train', 'validation', 'reporting'):
        need(np.array_equal(plan[role + '_labels'], truth[plan[role + '_ids']]), 'role labels differ from pinned IDX')
    need(np.array_equal(plan['assigned_labels'], np.where(plan['corruption_mask'],
              plan['replacement_labels'], plan['train_labels'])) and
         np.array_equal(plan['changed_mask'], plan['assigned_labels'] != plan['train_labels']), 'fixed assignment law')
    need(header['corruption_selected'] == int(plan['corruption_mask'].sum()) and
         header['actually_wrong'] == int(plan['changed_mask'].sum()) and
         header['true_class_counts'] == {role: np.bincount(plan[role + '_labels'], minlength=10).tolist()
                                       for role in ('train', 'validation', 'reporting')}, 'source plan header counts')
    return panel_helper.select_panels({k: plan[k] for k in ROLE_KEYS}, seed)


def make_inputs(images, panels, device):
    """Image-major grids, raw-pixel zero padding, then original normalization."""
    def original(ids):
        return images[ids].astype(np.float32) / np.float32(255)
    def grid(ids):
        raw = np.repeat(original(ids), 25, axis=0)
        shifts = np.tile(panels['view_shifts'], (len(ids), 1))
        return torch.from_numpy(data.normalize(data.translate(raw, shifts))).to(device)
    action = []
    for batch in range(2):
        raw = data.translate(original(panels['action_ids'][batch]), panels['action_shifts'][batch])
        action.append(torch.from_numpy(data.normalize(raw)).to(device))
    return {'I': grid(panels['evaluation_ids']), 'R': grid(panels['reporting_ids']),
            'R_original': torch.from_numpy(data.normalize(original(panels['reporting_ids']))).to(device),
            'baseline': torch.from_numpy(data.normalize(original(panels['baseline_ids']))).to(device),
            'actions': action}


def forward_grids(model, inputs, check):
    values = {}
    for key, count in (('I', 256), ('R', 128), ('R_original', 128)):
        parts = []
        for chunk in inputs[key].split(500):
            check()
            parts.append(model(chunk))
        logits = torch.cat(parts)
        need(logits.dtype == torch.float32 and bool(torch.isfinite(logits).all()), 'finite FP32 logits')
        values[key] = logits.reshape(count, 25, 10) if key != 'R_original' else logits
    return values


def report(logits, panels):
    values = {k: torch.as_tensor(v, device='cpu') for k, v in logits.items()}
    return objective.metric_report(values['I'], torch.from_numpy(panels['evaluation_true']),
              torch.from_numpy(panels['evaluation_assigned']), original_view_index=12,
              r_original=values['R_original'], r_grid=values['R'],
              r_true=torch.from_numpy(panels['reporting_true']))


def check_identity(metrics):
    v = metrics['objectives']
    need(math.isclose(v['L'], math.fsum(v[k] for k in ('S', 'F', 'C')), rel_tol=1e-10, abs_tol=1e-10), 'S+F+C identity')
    need(math.isclose(v['S'], .1 * v['S_true'] + .9 * v['S_uniform'], rel_tol=1e-10, abs_tol=1e-10), 'S split identity')


def baseline_gradients(model, inputs, panels, check):
    """Graph references die on function return, before any private action."""
    logits = forward_grids(model, inputs, check)
    device = next(model.parameters()).device
    values = objective.objective_tensors(logits['I'], torch.from_numpy(panels['evaluation_true']).to(device),
                 torch.from_numpy(panels['evaluation_assigned']).to(device), logits['R_original'], logits['R'],
                 torch.from_numpy(panels['reporting_true']).to(device))
    gradients = objective.flat_objective_gradients({k: values[k] for k in KEYS}, model.parameters())
    cpu_gradients = {k: v.detach().cpu() for k, v in gradients.items()}
    residual = sum(cpu_gradients[k].double() for k in ('S', 'F', 'C')) - cpu_gradients['L'].double()
    need(float(residual.norm()) <= 1e-6 + 5e-5 * float(cpu_gradients['L'].double().norm()), 'gradient decomposition identity')
    arrays = {k: v.detach().cpu().numpy() for k, v in logits.items()}
    metrics = report(arrays, panels)
    check_identity(metrics)
    return arrays, cpu_gradients, metrics


@torch.no_grad()
def install_point(model, point):
    offset = 0
    for parameter in model.parameters():
        size = parameter.numel()
        parameter.copy_(point[offset:offset + size].reshape(parameter.shape))
        offset += size
    need(offset == point.numel(), 'point parameter order/length')


def finish_effects(baseline, endpoints):
    for row in endpoints:
        decay = next(r for r in endpoints if r['batch'] == row['batch'] and
                     r['policy'] == 'decay' and r['fraction'] == row['fraction'])
        for key in KEYS:
            effect = row['effects'][key]
            effect['finite'] = baseline['objectives'][key] - row['metrics']['objectives'][key]
            effect['residual'] = effect['finite'] - effect['linear']
            effect['finite_decay'] = baseline['objectives'][key] - decay['metrics']['objectives'][key]
            effect['finite_data'] = decay['metrics']['objectives'][key] - row['metrics']['objectives'][key]
    contrasts = []
    for batch in range(2):
        for fraction, name in FRACTIONS:
            pair = {r['policy']: r for r in endpoints if r['batch'] == batch and r['fraction'] == fraction}
            contrasts.append({'batch': batch, 'fraction': fraction, 'fraction_id': name,
                              'objectives': {k: {kind: pair['native']['effects'][k][kind] -
                                                pair['raw']['effects'][k][kind] for kind in ('finite', 'linear')}
                                             for k in KEYS}})
    return contrasts


def summarize(parents):
    need([{k: row[k] for k in ('seed', 'augmentation', 'step', 'parent_id')} for row in parents] == roster(), 'summary roster')
    cells = []
    for augmentation in AUGMENTATIONS:
        for step in STEPS:
            for fraction, name in FRACTIONS:
                rows = []
                for seed in SEEDS:
                    parent = next(p for p in parents if (p['seed'], p['augmentation'], p['step']) == (seed, augmentation, step))
                    values = {}
                    for key in KEYS:
                        row = {}
                        for kind in ('finite', 'linear'):
                            for policy in ('raw', 'native'):
                                selected = [e['effects'][key][kind] for e in parent['endpoints']
                                            if e['policy'] == policy and e['fraction'] == fraction]
                                need(len(selected) == 2, 'two draws required in summary')
                                row[policy + '_' + kind] = math.fsum(selected) / 2
                            contrasts = [c['objectives'][key][kind] for c in parent['contrasts'] if c['fraction'] == fraction]
                            need(len(contrasts) == 2, 'two paired contrasts required')
                            row['contrast_' + kind] = math.fsum(contrasts) / 2
                        values[key] = row
                    rows.append({'seed': seed, 'objectives': values})
                cells.append({'augmentation': augmentation, 'step': step, 'fraction': fraction,
                              'fraction_id': name, 'seed_rows': rows})
    primary = copy.deepcopy(next(c for c in cells if (c['augmentation'], c['step'], c['fraction']) == ('translate', 56304, 1.0)))
    for row in primary['seed_rows']:
        row['objectives'] = {key: row['objectives'][key] for key in ('H_O', 'C')}
    return {'primary': primary, 'cells': cells}


def acquire_parent(run, row, source, inventory, panels, inputs):
    run.check()
    name, stage = row['parent_id'], 'warmup' if row['step'] == 100 else 'final'
    receipt, readout = source[stage], source[stage + '_readout']
    saved = restore_helper.load_snapshot(Path(inventory['archive']) / receipt['path'],
                expected_size=receipt['size_bytes'], expected_sha256=receipt['sha256'], expected_step=row['step'])
    before = core.neural_core.tree_digest(saved)
    rng = saved['rng']
    parent = restore_helper.restore(saved, device='cuda:0', expected_step=row['step'])
    need(core.neural_core.tree_digest(restore_helper.snapshot_with_rng(*parent, rng)) == before, 'restored parent round trip')
    del saved
    model = parent[0]
    source_logits = numeric_npz(receipt_bytes(inventory['archive'], readout))
    need(set(source_logits) == {'step', 'train', 'validation', 'reporting'} and
         source_logits['step'].dtype == np.int64 and source_logits['step'].shape == (1,) and
         int(source_logits['step'][0]) == row['step'] and source_logits['reporting'].dtype == np.float32 and
         source_logits['reporting'].shape == (5000, 10), 'source readout schema')
    actual = base.predict(model, inputs['baseline'])
    expected = source_logits['reporting'][:500].copy()
    check_receipt = run.save(f'baseline-check-{name}.npz',
                   {'actual': actual, 'expected': expected, 'ids': panels['baseline_ids']}, 'npz')
    need(np.array_equal(actual, expected), 'original reporting prediction mismatch; no action admitted')
    del source_logits, actual, expected
    logits, gradients, metrics = baseline_gradients(model, inputs, panels, run.check)
    baseline_receipt = run.save(f'baseline-{name}.npz', logits, 'npz')
    theta = core.flat_params(model).cpu()
    gradient_receipt = run.save(f'gradients-{name}.npz',
                       {**{k: v.numpy() for k, v in gradients.items()}, 'theta': theta.numpy()}, 'npz')
    del logits
    result = {**row, 'source_checkpoint': receipt, 'source_readout': readout,
              'baseline_check_receipt': check_receipt, 'baseline_receipt': baseline_receipt,
              'gradient_receipt': gradient_receipt, 'baseline_metrics': metrics,
              'parent_digest_before': before, 'actions': [], 'endpoints': []}
    for batch in range(2):
        run.check()
        target = torch.from_numpy(panels['action_assigned'][batch]).to(run.device)
        action_logits = model(inputs['actions'][batch])
        loss = F.cross_entropy(action_logits, target, reduction='mean')
        parts = torch.autograd.grad(loss, tuple(model.parameters()), create_graph=False)
        raw_gradient = torch.cat([p.detach().reshape(-1) for p in parts])
        del parts, loss, action_logits, target
        pair = action_helper.paired_actions(parent, raw_gradient)
        del raw_gradient
        for record in pair:
            policy = record['policy']
            arrays = {k: v.numpy() for k, v in record.items() if isinstance(v, torch.Tensor)}
            rank = 0 if record['post_basis'] is None else record['post_basis'].shape[1]
            if record['post_basis'] is None:
                arrays['post_basis'] = np.empty((P, 0), np.float32)
                arrays['post_singular_values'] = np.empty((0,), np.float64)
            result['actions'].append({'batch': batch, 'policy': policy,
                       'receipt': run.save(f'action-{name}-b{batch}-{policy}.npz', arrays, 'npz'),
                       'observer_steps_before': record['observer_steps_before'],
                       'observer_steps_after': record['observer_steps_after'], 'basis_rank': rank})
            del arrays
        candidates = [('raw', pair[0]['theta_after']), ('native', pair[1]['theta_after']),
                      ('decay', pair[0]['decay_endpoint'])]
        decay_endpoint = pair[0]['decay_endpoint']
        need(torch.equal(decay_endpoint, pair[1]['decay_endpoint']), 'paired decay differs')
        # Basis/moment arrays are archived; retain only candidate vectors now.
        del record, pair
        for policy, endpoint in candidates:
            for fraction, fraction_id in FRACTIONS:
                run.check()
                path = objective.path_accounting(theta, endpoint, decay_endpoint, fraction)
                path_receipt = run.save(f'path-{name}-b{batch}-{policy}-{fraction_id}.npz',
                               {k: v.numpy() for k, v in path.items() if isinstance(v, torch.Tensor)}, 'npz')
                private_model = copy.deepcopy(model)
                with torch.no_grad():
                    install_point(private_model, path['point'])
                    evaluated = forward_grids(private_model, inputs, run.check)
                    arrays = {k: v.detach().cpu().numpy() for k, v in evaluated.items()}
                del evaluated, private_model
                endpoint_metrics = report(arrays, panels)
                check_identity(endpoint_metrics)
                logit_receipt = run.save(f'logits-{name}-b{batch}-{policy}-{fraction_id}.npz', arrays, 'npz')
                effects = {key: {label: objective.signed_utility(gradients[key], path[delta])
                                for label, delta in (('linear', 'total_delta'), ('linear_decay', 'decay_delta'),
                                                     ('linear_data', 'data_delta'))} for key in KEYS}
                result['endpoints'].append({'batch': batch, 'policy': policy, 'fraction': fraction,
                         'fraction_id': fraction_id, 'logit_receipt': logit_receipt, 'path_receipt': path_receipt,
                         'metrics': endpoint_metrics, 'effects': effects,
                         'norms': {k: path[k] for k in ('total_norm', 'decay_norm', 'data_norm')}})
                del arrays, path
        del candidates, endpoint, decay_endpoint
    result['contrasts'] = finish_effects(metrics, result['endpoints'])
    result['parent_digest_after'] = core.neural_core.tree_digest(restore_helper.snapshot_with_rng(*parent, rng))
    need(result['parent_digest_after'] == before, 'parent mutated during private diagnostic')
    need(len(result['actions']) == 4 and len(result['endpoints']) == 12, 'parent incomplete')
    return result


def acquire(run, inventory, manifest):
    verify_input_metadata(inventory)
    images, truth = base.read_training()
    # Fix all seed panels before any checkpoint/model is loaded.
    selected, panel_rows = {}, []
    for seed in SEEDS:
        run.check()
        selected[seed] = load_plan(inventory, seed, truth)
        panel_rows.append({'seed': seed, 'receipt': run.save(f'panels-s{seed}.npz', selected[seed], 'npz')})
    parents = []
    for seed in SEEDS:
        panels = selected[seed]
        inputs = make_inputs(images, panels, run.device)
        for row in (r for r in roster() if r['seed'] == seed):
            source = next(p for p in inventory['parents'] if p['seed'] == seed and p['augmentation'] == row['augmentation'])
            parents.append(acquire_parent(run, row, source, inventory, panels, inputs))
        del inputs
    torch.cuda.synchronize('cuda:0')
    run.check()
    need(len(run.receipts) == byte_inventory()['expected_receipted_artifacts'], 'artifact inventory incomplete')
    for name, sha in manifest['source_pins'].items():
        read_pinned(ROOT / name, sha)
    for name, sha in inventory['data_pins'].items():
        need(base.digest(base.DATA / name) == sha, 'data changed during diagnostic')
    return {'schema': SCHEMA, 'status': 'complete', 'source_pins': manifest['source_pins'],
            'data_pins': inventory['data_pins'], 'input_inventory_sha256': manifest['input_inventory_sha256'],
            'roster': roster(), 'panels': panel_rows, 'parents': parents, 'summary': summarize(parents),
            'receipts': list(run.receipts), 'seconds': time.monotonic() - run.started,
            'serialization_seconds': run.serialization_seconds,
            'gpu_max_allocated_bytes': torch.cuda.max_memory_allocated('cuda:0'),
            'host_high_water_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}


def check_output_directory(output):
    output = Path(output)
    need(output.is_absolute() and output.resolve() == output and output.is_dir() and not any(output.iterdir()),
         'exclusive existing empty absolute output directory required')
    need(output.name == 'acquisition-001' and output.parent.parent == Path('/tmp/spectral-experiment-artifacts') and
         output.parent.name.startswith('spectral-component-utility-'), 'fixed large-volume output identity')
    mounted = subprocess.check_output(['findmnt', '-n', '-o', 'SOURCE,TARGET', '-T', str(output)], text=True).split()
    need(mounted == ['/dev/RECONFIGURE_FOR_LOCAL_STORAGE', '/private-artifacts/storage'], 'required /dev/RECONFIGURE_FOR_LOCAL_STORAGE mount missing')
    need(shutil.disk_usage(output).free >= 16 * 1024**3, '16GiB admission disk reserve')
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--expected-sources', type=Path)
    parser.add_argument('--expected-sources-sha256')
    args = parser.parse_args(argv)
    if not args.execute:
        print(json.dumps({'status': 'inert', 'schema': SCHEMA, 'roster': roster(), 'inventory': byte_inventory()}))
        return 0
    need(args.output_dir is not None and args.expected_sources is not None and
         args.expected_sources_sha256 is not None, 'execution requires output and committed manifest pins')
    manifest, inventory = verify_manifest(args.expected_sources, args.expected_sources_sha256)
    output = check_output_directory(args.output_dir)
    need(not (DOCS / 'attempt.json').exists(), 'one attempt already consumed; never retry automatically')
    invocation = os.environ.get('INVOCATION_ID', '')
    need(re.fullmatch(r'[0-9a-f]{32}', invocation), 'systemd invocation identity required')
    attempt = {'schema': SCHEMA, 'commit': manifest['commit'],
               'worktree_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
               'source_pins': manifest['source_pins'], 'data_pins': inventory['data_pins'],
               'input_inventory_sha256': manifest['input_inventory_sha256'],
               'manifest_sha256': args.expected_sources_sha256, 'unit': UNIT, 'output_dir': str(output),
               'started_utc': datetime.now(timezone.utc).isoformat(), 'pid': os.getpid(),
               'invocation_id': invocation, 'argv': sys.argv if argv is None else [str(Path(__file__)), *argv]}
    from experiments import spectral_component_utility_guard as guard
    run = guard.Run(output, device='cuda:0')
    try:
        with (DOCS / 'attempt.json').open('x') as handle:
            json.dump(attempt, handle, allow_nan=False, separators=(',', ':'))
            handle.write('\n')
        configured = guard.configure(device='cuda:0')
        run.save('provenance.json', {**attempt, 'guards': configured, 'inventory': byte_inventory(),
                                   'python': sys.version, 'numpy': np.__version__, 'torch': str(torch.__version__)})
        results = acquire(run, inventory, manifest)
        receipt = run.save('results.json', results)
        print(json.dumps({'status': 'complete', 'results': receipt}), flush=True)
    except BaseException as exc:
        run.write_failure(exc)
        raise
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
