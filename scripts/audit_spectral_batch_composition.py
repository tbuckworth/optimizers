#!/usr/bin/env python3
"""Independent fixed-exposure saved-data audit; explicit --execute required.

Only inert arithmetic from the pinned previous independent checker is reused.
No producer import, model, inference, gradient estimation or optimizer replay.
"""
from __future__ import annotations

import hashlib
import importlib.util
import argparse
import json
import os
from pathlib import Path
import re
import resource
import subprocess
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HELPER_PATH = ROOT / 'scripts/audit_spectral_selectivity_boundary.py'
HELPER_SHA = '93b99a093d8641670c7b3eb77f91f8438dfadee34fbd28939d8d5944c51ab5da'
if hashlib.sha256(HELPER_PATH.read_bytes()).hexdigest() != HELPER_SHA:
    raise RuntimeError('independent arithmetic helper source changed')
_spec = importlib.util.spec_from_file_location('selectivity_independent_arithmetic', HELPER_PATH)
h = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(h)

SCHEMA = 'spectral_batch_composition_v1'
SEEDS = (202609121, 202609122, 202609123)
CELLS = ('clean', 'diffuse')
SCHEDULES = ('interleaved', 'grouped')
POLICIES = ('raw', 'native32', 'norm_raw')
ANCHOR_BLOCKS = (0, 7, 37)
DOCS = 'output/2026-09-10-spectral-batch-composition'
UNIT = 'spectral-batch-composition-audit-001.service'
SOURCE_NAMES = {'spectral_filter.py', 'experiments/spectral_selectivity_boundary.py',
                'experiments/spectral_batch_composition.py', 'tests/test_spectral_batch_composition.py',
                DOCS + '/protocol.md',
                'output/2026-09-06-spectral-optimizer-investigation/continuation/iteration-009/neural_core.py'}
require = h.require


def stream(seed, *parts):
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, *parts])))


def rebuild_schedules(original, true, seed, block_size=50):
    """Rebuild exact occurrence routing; small arrays are for fabricated tests."""
    original, true = np.asarray(original), np.asarray(true)
    require(original.ndim == 2 and original.dtype == np.int64, 'original batches must be int64 matrix')
    require(len(original) % block_size == 0 and block_size >= 2, 'whole nontrivial blocks required')
    require(((original >= 0) & (original < len(true))).all(), 'batch index outside training pool')
    nblocks, batch_size = len(original) // block_size, original.shape[1]
    result = {schedule + '_batches': np.empty_like(original) for schedule in SCHEDULES}
    result.update({schedule + '_rare_counts': np.empty(len(original), np.int64) for schedule in SCHEDULES})
    result['occurrence_permutations'] = np.empty((nblocks, block_size * batch_size), np.int64)
    result['count_position_permutations'] = np.empty((nblocks, block_size), np.int64)
    result['within_batch_permutations'] = np.empty((nblocks, block_size, batch_size), np.int64)
    for block in range(nblocks):
        start = block * block_size
        draws = original[start:start + block_size].reshape(-1)
        occurrence_order = stream(seed, 6, block).permutation(len(draws))
        result['occurrence_permutations'][block] = occurrence_order
        ordered = draws[occurrence_order]
        rare = ordered[true[ordered] == 8]
        common = ordered[true[ordered] != 8]
        position_order = stream(seed, 7, block).permutation(block_size)
        result['count_position_permutations'][block] = position_order
        counts = {'interleaved': np.full(block_size, len(rare) // block_size, np.int64),
                  'grouped': np.zeros(block_size, np.int64)}
        counts['interleaved'][position_order[:len(rare) % block_size]] += 1
        remaining = len(rare)
        for position in position_order:
            assigned = min(batch_size, remaining)
            counts['grouped'][position] = assigned
            remaining -= assigned
        slot_orders = [stream(seed, 8, block, t).permutation(batch_size) for t in range(block_size)]
        result['within_batch_permutations'][block] = np.stack(slot_orders)
        for schedule in SCHEDULES:
            rare_cursor, common_cursor = 0, 0
            for position, count in enumerate(counts[schedule]):
                common_count = batch_size - int(count)
                unshuffled = np.concatenate((rare[rare_cursor:rare_cursor + count],
                                            common[common_cursor:common_cursor + common_count]))
                require(len(unshuffled) == batch_size, 'schedule exhausted its occurrence lists')
                result[schedule + '_batches'][start + position] = unshuffled[slot_orders[position]]
                rare_cursor += int(count)
                common_cursor += common_count
            require(rare_cursor == len(rare) and common_cursor == len(common), 'schedule did not consume every occurrence')
            result[schedule + '_rare_counts'][start:start + block_size] = counts[schedule]
            require(np.array_equal(np.sort(result[schedule + '_batches'][start:start + block_size].reshape(-1)),
                                   np.sort(draws)), 'block occurrence multiset differs')
    return result


def selected_anchors(counts, blocks=ANCHOR_BLOCKS, block_size=50):
    counts = np.asarray(counts)
    anchors = []
    for block in blocks:
        values = counts[block * block_size:(block + 1) * block_size]
        require(len(values) == block_size, 'missing anchor block')
        high = int(np.argmax(values))
        candidates = [p for p in range(block_size) if p != high]
        low = min(candidates, key=lambda p: (int(values[p]), p))
        for label, position in (('high', high), ('low', low)):
            anchors.append({'block': block, 'position': position, 'update': 101 + block_size * block + position,
                            'label': label, 'rare_count': int(values[position]),
                            'equal_counts_in_block': bool(np.all(values == values[0]))})
    return sorted(anchors, key=lambda row: row['update'])


def probe_positions(plan, assigned, seed):
    true = plan['train_true']
    common_rng = stream(seed, 9, 0)
    common = np.concatenate([common_rng.permutation(np.where(true == d)[0])[:6] for d in h.MAJORITY])
    rare = stream(seed, 9, 1).permutation(np.where(true == 8)[0])[:32]
    wrong = stream(seed, 9, 2).permutation(np.where(true != assigned)[0])[:32]
    return {'majority': common, 'rare8': rare, 'wrong_assigned': wrong, 'wrong_corrected': wrong.copy()}


def mean_geometry(gradient, v, q=None):
    gradient = h.arr(gradient, np.float32)
    require(gradient.ndim == 1, 'group-mean gradient is one vector, not per-example rows')
    x = gradient.astype(np.float64)
    applied = h.action(gradient, v).astype(np.float64)
    energy, native_energy = float(x @ x), float(applied @ applied)
    if v is not None and q is None:
        q, _ = h.span_basis(v)
    span_energy = None if q is None else float(np.sum((q.T @ x)**2))
    return {'squared_norm': energy, 'native_squared_norm': native_energy,
            'native_retention': native_energy / energy if energy else None,
            'numerical_span_squared_norm': span_energy,
            'numerical_span_retention': span_energy / energy if span_energy is not None and energy else None,
            'zero_denominator': energy == 0, 'identity_fallback': v is None,
            'numerical_basis_rank': 0 if q is None else q.shape[1]}


def evaluation(step, train, heldout, true, assigned, heldout_true):
    return {'step': step, 'train_true': h.grouped(train, true),
            'train_assigned': h.classification(train, assigned),
            'train_actually_changed': h.classification(train, assigned, true != assigned),
            'heldout': h.grouped(heldout, heldout_true)}


def endpoint(row):
    result = {group + '_' + metric: row['heldout'][group][metric]
              for group in ('rare', 'majority_macro', 'balanced_total') for metric in ('accuracy', 'ce')}
    for prefix in ('heldout', 'train_true'):
        for digit in range(10):
            for metric in ('accuracy', 'ce'):
                result[f'{prefix}_class{digit}_{metric}'] = row[prefix]['per_class'][str(digit)][metric]
    for group in ('rare', 'majority_macro', 'balanced_total'):
        for metric in ('accuracy', 'ce'):
            result[f'train_true_{group}_{metric}'] = row['train_true'][group][metric]
    for prefix, field in (('train_assigned', 'train_assigned'), ('train_wrong', 'train_actually_changed')):
        for metric in ('accuracy', 'ce'):
            result[prefix + '_' + metric] = row[field][metric]
    return result


def difference(left, right):
    require((left is None) == (right is None), 'asymmetric missing paired outcome')
    return None if left is None else left - right


def aggregate(rows):
    roster = {(row['seed'], row['cell'], row['schedule'], row['policy']): row for row in rows}
    expected = {(s, c, t, p) for s in SEEDS for c in CELLS for t in SCHEDULES for p in POLICIES}
    require(len(roster) == len(rows) == 36 and set(roster) == expected, 'all 36 unique paired trajectories required')
    keys = list(endpoint(rows[0]['endpoint']))
    def value(s, c, t, p, k, phase='endpoint'):
        return endpoint(roster[s, c, t, p][phase])[k]
    groups, changes, schedules, policies, interactions = {}, {}, {}, {}, {}
    for cell in CELLS:
        for schedule in SCHEDULES:
            for policy in POLICIES:
                name = '/'.join((cell, schedule, policy))
                groups[name] = {k: h.three_seed([value(s, cell, schedule, policy, k) for s in SEEDS]) for k in keys}
                changes[name] = {k: h.three_seed([difference(value(s, cell, schedule, policy, k),
                    value(s, cell, schedule, policy, k, 'warmup')) for s in SEEDS]) for k in keys}
            for control in ('raw', 'norm_raw'):
                name = '/'.join((cell, schedule, 'native32_minus_' + control))
                policies[name] = {k: h.three_seed([difference(value(s, cell, schedule, 'native32', k),
                    value(s, cell, schedule, control, k)) for s in SEEDS]) for k in keys}
        for policy in POLICIES:
            name = '/'.join((cell, policy, 'grouped_minus_interleaved'))
            schedules[name] = {k: h.three_seed([difference(value(s, cell, 'grouped', policy, k),
                value(s, cell, 'interleaved', policy, k)) for s in SEEDS]) for k in keys}
        for policy in ('native32', 'norm_raw'):
            name = cell + '/' + policy + '_minus_raw'
            interactions[name] = {k: h.three_seed([difference(
                difference(value(s, cell, 'grouped', policy, k), value(s, cell, 'interleaved', policy, k)),
                difference(value(s, cell, 'grouped', 'raw', k), value(s, cell, 'interleaved', 'raw', k)))
                for s in SEEDS]) for k in keys}
    return {'seeds': list(SEEDS), 'endpoint_step': 2000, 'per_group': groups,
            'change_from_warmup': changes, 'schedule_contrasts': schedules, 'policy_contrasts': policies,
            'schedule_policy_interactions': interactions, 'selection': 'fixed_step_2000_no_checkpoint_selection'}


def block_metadata(original, schedules, block_size=50):
    result = []
    for block, start in enumerate(range(0, len(original), block_size)):
        raw = original[start:start + block_size]
        sorted_hash = h.array_hash(np.sort(raw.reshape(-1)))
        counts = {s: schedules[s + '_rare_counts'][start:start + block_size] for s in SCHEDULES}
        result.append({'block': block, 'draws': raw.size, 'rare_occurrences': int(counts['interleaved'].sum()),
                       'original_sorted_multiset_sha256': sorted_hash,
                       'interleaved_sorted_multiset_sha256': sorted_hash, 'grouped_sorted_multiset_sha256': sorted_hash,
                       'rare_fraction_mean': float(counts['interleaved'].mean() / raw.shape[1]),
                       'rare_fraction_population_variances': {s: float(np.var(counts[s] / raw.shape[1])) for s in SCHEDULES},
                       'multiset_checks': 'PASS'})
    return result


def verify_diagnostic(payload, report, plan, assigned, train_x, warmup, anchor, identity, checks):
    seed, cell, schedule = identity
    update = anchor['update']
    for obj in (payload, report):
        checks.equal({key: obj[key] for key in ('schema', 'seed', 'cell', 'schedule', 'policy', 'update', 'anchor')},
                     {'schema': SCHEMA, 'seed': seed, 'cell': cell, 'schedule': schedule,
                      'policy': 'native32', 'update': update, 'anchor': anchor}, 'diagnostic identity')
        require(obj['warmup_state_sha256'] == h.tree_hash(warmup), 'diagnostic warmup hash')
    require(payload['probe_parameter_state'] == 'pre_update_after_one_training_observer_update', 'probe timing')
    require(payload['parameter_order'] == list(h.MODEL_SHAPES), 'diagnostic parameter order')
    before, after = payload['before'], payload['after']
    require(h.tree_hash(before) == report['before_sha256'] and h.tree_hash(after) == report['after_sha256'], 'saved diagnostic state hash')
    old, m0, v0 = h.snapshot_arrays(before, update - 1, True, gradients_present=True)
    cleared = {**after, 'gradients': [None] * len(h.MODEL_SHAPES)}
    new, m1, v1 = h.snapshot_arrays(cleared, update, True)
    for key in ('tracker', 'rng', 'model_modes', 'model_spec', 'gradients'):
        require(h.tree_hash(before[key]) == h.tree_hash(after[key]), 'unexpected diagnostic mutation: ' + key)
    if update == 101:
        for key in ('model_state', 'optimizer', 'rng', 'model_modes', 'model_spec'):
            require(h.tree_hash(before[key]) == h.tree_hash(warmup[key]), 'first-update warmup identity: ' + key)
    raw = h.arr(payload['raw_training_gradient'], np.float32, (len(old),))
    applied = h.arr(payload['applied_training_gradient'], np.float32, (len(old),))
    checks.array(np.concatenate([h.arr(g).reshape(-1) for g in before['gradients']]), applied, 'actual delivered gradient')
    v = None if before['tracker']['V'] is None else h.arr(before['tracker']['V'], np.float32)
    q, basis = h.span_basis(v)
    checks.equal(report['basis'], basis, 'numerical basis', atol=1e-10, rtol=1e-8)
    expected_action = raw.astype(np.float64) if v is None else v.astype(np.float64) @ (v.astype(np.float64).T @ raw.astype(np.float64))
    action_error = float(np.linalg.norm(applied.astype(np.float64) - expected_action))
    action_bound = 3e-5 * float(np.linalg.norm(raw.astype(np.float64))) + 1e-12
    require(action_error <= action_bound, 'native action exceeds fixed roundoff bound')
    vectors, movement = h.movements(old, new, q)
    checks.equal(report['movement'], movement, 'actual movement', atol=1e-10, rtol=1e-8)
    adam = h.verify_adam(old, new, m0, v0, m1, v1, applied, update, checks)
    positions = probe_positions(plan, assigned, seed)
    require(set(payload['groups']) == set(report['groups']) == set(positions), 'mean probe group roster')
    means = {}
    for name, indices in positions.items():
        group, recorded = payload['groups'][name], report['groups'][name]
        if not len(indices):
            require(group is None, 'absent wrong group must be null')
            checks.equal(recorded, {'status': 'absent', 'reason': 'no_actually_changed_examples', 'count': 0}, 'absent group')
            means[name] = None
            continue
        require(group is not None and recorded['status'] == 'defined', 'defined mean group required')
        require(set(group) == {'training_positions', 'source_ids', 'inputs', 'targets', 'true_targets',
                               'assigned_targets', 'pre_logits', 'post_logits', 'mean_gradient'}, 'mean probe tensor schema')
        true = plan['train_true'][indices]
        targets = assigned[indices] if name == 'wrong_assigned' else true
        for key, value in {'training_positions': indices, 'source_ids': plan['train_ids'][indices],
                           'inputs': train_x[indices], 'targets': targets, 'true_targets': true,
                           'assigned_targets': assigned[indices]}.items():
            checks.array(group[key], value, 'probe membership/' + name + '/' + key)
        checks.equal(recorded['count'], len(indices), 'probe count')
        checks.equal(recorded['true_label_counts'], np.bincount(true, minlength=10).tolist(), 'probe label composition')
        gradient = h.arr(group['mean_gradient'], np.float32, (len(old),))
        means[name] = gradient
        expected_geometry = mean_geometry(gradient, v, q)
        checks.equal(recorded['geometry'], expected_geometry, 'mean native geometry', atol=1e-7, rtol=5e-4)
        for key in ('squared_norm', 'numerical_span_squared_norm', 'numerical_span_retention'):
            checks.equal(recorded['geometry'][key], expected_geometry[key], 'mean exact geometry/' + key, atol=1e-10, rtol=1e-8)
        pre = h.classification(h.arr(group['pre_logits'], np.float32, (len(indices), 10)), targets)
        post = h.classification(h.arr(group['post_logits'], np.float32, (len(indices), 10)), targets)
        checks.equal(recorded['pre'], pre, 'mean probe pre metrics')
        checks.equal(recorded['post'], post, 'mean probe post metrics')
        checks.equal(recorded['finite_ce_improvement'], pre['ce'] - post['ce'], 'finite group CE improvement')
        utilities = {key: -float(gradient.astype(np.float64) @ delta) for key, delta in vectors.items()}
        checks.equal(recorded['signed_first_order_utilities'], utilities, 'mean signed utilities')
    differences = {'rare_minus_majority': means['rare8'] - means['majority'],
                   'wrong_assigned_minus_corrected': None if means['wrong_assigned'] is None else
                       means['wrong_assigned'] - means['wrong_corrected']}
    require(set(payload['mean_differences']) == set(report['mean_difference_geometry']) == set(differences), 'mean difference roster')
    for name, expected in differences.items():
        if expected is None:
            require(payload['mean_differences'][name] is None and report['mean_difference_geometry'][name] is None, 'absent mean difference')
        else:
            checks.array(payload['mean_differences'][name], expected, 'exact FP32 mean difference')
            expected_geometry = mean_geometry(expected, v, q)
            checks.equal(report['mean_difference_geometry'][name], expected_geometry, 'mean difference geometry', atol=1e-7, rtol=5e-4)
            checks.equal(report['mean_difference_geometry'][name]['squared_norm'], expected_geometry['squared_norm'], 'mean difference squared norm')
    require(report['difference_arithmetic_dtype'] == 'float32'
            and report['probe_gradient_definition'] == 'autograd.grad of FP32 group mean cross_entropy'
            and report['probe_side_effect_checks'] == 'PASS' and report['per_example_gradients_or_coherence'] is False,
            'mean-only diagnostic contract')
    return {'seed': seed, 'cell': cell, 'schedule': schedule, 'update': update, 'anchor': anchor,
            'action_l2_error': action_error, 'action_l2_bound': action_bound, 'adam': adam,
            'movement': movement, 'basis': basis, 'after_cleared_sha256': h.tree_hash(cleared)}


def fixed_artifact_names():
    names = {'manifest.json', 'results.json'}
    for seed in SEEDS:
        names.update({f'plan-s{seed}.npz', f'plan-s{seed}.json', f'schedules-s{seed}.npz', f'probes-s{seed}.npz',
                      f'warmup-s{seed}.pt', f'warmup-actions-s{seed}.json', f'common-predictions-s{seed}.npz'})
        for cell in CELLS:
            for schedule in SCHEDULES:
                names.add(f'binding-s{seed}-{cell}-{schedule}.json')
                for policy in POLICIES:
                    identity = f's{seed}-{cell}-{schedule}-{policy}'
                    names.update({f'final-{identity}.pt', f'logits-{identity}.npz', f'actions-{identity}.json', f'curve-{identity}.json'})
    return names


def verify_sources(pins):
    require(type(pins) is dict and set(pins) == SOURCE_NAMES, 'exact new producer source pin roster')
    for name, expected in pins.items():
        path = ROOT / name
        require(path.resolve().is_relative_to(ROOT), 'producer source escape')
        cursor = path
        while cursor != ROOT:
            require(not cursor.is_symlink(), 'producer source symlink')
            cursor = cursor.parent
        require(path.is_file() and h.sha256(path) == expected, 'producer source pin: ' + name)


class Bundle:
    def __init__(self, root, complete_sha, manifest_sha, pins, checks):
        self.root = Path(root).resolve(strict=True)
        self.checks, self.complete_sha, self.pins = checks, complete_sha, pins
        self.complete_path = h.contained_file(self.root, 'complete.json')
        require(h.sha256(self.complete_path) == complete_sha, 'main-supplied completion hash')
        c = h.read_json(self.complete_path)
        self.complete = c
        require(c['schema'] == SCHEMA and c['status'] == 'complete' and c['completed_trajectories'] == 36
                and c['completed_diagnostics'] == 72, 'complete 36-trajectory/72-event batch required')
        require(c['source_pins'] == pins and c['data_pins'] == h.DATA_PINS, 'completion source/data pins')
        self.index = {}
        for item in c['receipts']:
            checks.tick()
            path = h.verify_receipt(self.root, item)
            require(path.name not in self.index, 'duplicate artifact receipt')
            self.index[path.name] = item
        fixed = fixed_artifact_names()
        require(fixed <= set(self.index) and len(self.index) == 323, 'fixed scientific artifact roster')
        dynamic = set(self.index) - fixed
        pattern = r'diagnostic-s(202609121|202609122|202609123)-(clean|diffuse)-(interleaved|grouped)-native32-u[0-9]{4}\.(json|pt)'
        require(len(dynamic) == 144 and all(re.fullmatch(pattern, name) for name in dynamic), 'diagnostic filename roster')
        require({p.name for p in self.root.iterdir()} == set(self.index) | {'complete.json'}, 'unreceipted acquisition file')
        size = sum(item['size_bytes'] for item in self.index.values())
        require(size == c['artifact_bytes_before_completion'] and size + self.complete_path.stat().st_size <= 3 * 1024**3, 'acquisition output inventory')
        require(0 < c['wall_seconds'] <= 1800 and c['gpu_max_allocated_bytes'] <= 8 * 1024**3
                and c['process_high_water_rss_kib'] <= 16 * 1024**2, 'acquisition resource completion')
        require(self.index['manifest.json']['sha256'] == manifest_sha, 'main-supplied manifest hash')
        self.manifest = self.json('manifest.json')
        require(self.manifest['schema'] == SCHEMA and self.manifest['source_pins'] == pins
                and self.manifest['data_pins'] == h.DATA_PINS, 'manifest source binding')
        m = self.manifest
        fixed_manifest = {'seeds': list(SEEDS), 'cells': list(CELLS), 'schedules': list(SCHEDULES),
            'policies': list(POLICIES), 'warmup': 100, 'steps': 2000, 'batch_size': 64, 'block_size': 50,
            'blocks': 38, 'anchor_blocks': list(ANCHOR_BLOCKS), 'native_events_per_trajectory': 6,
            'eval_steps': list(h.EVAL), 'rank': 32, 'cooperative_seconds': 1500, 'cloud_spend_usd': 0,
            'rare_class': 8, 'majority_train_per_class': 550, 'rare_train_count': 50, 'heldout_per_class': 500,
            'data_directory': str(h.DATA), 'torch': '2.11.0+cu128', 'numpy': '1.26.4'}
        checks.equal({key: m[key] for key in fixed_manifest}, fixed_manifest, 'manifest contract')
        require(m['byte_inventory']['cap_bytes'] == 3 * 1024**3, 'manifest output cap')
        admission = m['resource_admission']
        require(admission['effective']['memory.max'] == str(16 * 1024**3)
                and admission['effective']['memory.swap.max'] == '0', 'acquisition memory limits')
        quota, period = admission['effective']['cpu.max'].split()
        require(quota != 'max' and int(quota) == int(period) > 0, 'acquisition CPU limits')
        require(admission['service'] == {'Type': 'exec', 'RuntimeMaxUSec': '30min', 'Restart': 'no', 'KillMode': 'control-group'}, 'acquisition service limits')
        self.bound(c['results'], 'results.json')
        verify_sources(pins)

    def bound(self, item, name):
        require(name in self.index and item == self.index[name], 'artifact cross-binding: ' + name)
        return self.root / name

    def json(self, name):
        require(name in self.index, 'unreceipted JSON')
        return h.read_json(self.root / name)

    def npz(self, name):
        require(name in self.index, 'unreceipted NPZ')
        return h.load_npz(self.root / name)

    def tensor(self, name):
        require(name in self.index, 'unreceipted CPU tensor archive')
        return h.cpu_tensor_file(self.root / name)

    def recheck(self):
        for item in self.index.values():
            self.checks.tick()
            h.verify_receipt(self.root, item)
        require(h.sha256(self.complete_path) == self.complete_sha, 'completion changed during audit')
        verify_sources(self.pins)
        for name, expected in h.DATA_PINS.items():
            require(h.sha256(h.DATA / name) == expected, 'input data changed during audit')


def verify_actions(rows, policy, counts, checks, warmup=False):
    h.verify_actions(rows, policy, checks, warmup=warmup)
    require(len(rows) == len(counts), 'count-bound action length')
    for row, count in zip(rows, counts):
        require(type(row['rare_count']) is int and row['rare_count'] == int(count)
                and row['batch_size'] == 64, 'action batch size/rare count differs from frozen schedule')


def audit(bundle, checks):
    images, labels = h.read_pinned_data()
    result = bundle.json('results.json')
    require(result['schema'] == SCHEMA and len(result['rows']) == 36, 'results schema/complete roster')
    roster = {(r['seed'], r['cell'], r['schedule'], r['policy']): r for r in result['rows']}
    require(len(roster) == 36 and set(roster) == {(s, c, t, p) for s in SEEDS for c in CELLS for t in SCHEDULES for p in POLICIES}, 'results unique identities')
    scientific_names = fixed_artifact_names()
    rebuilt, diagnostics = [], []
    for seed in SEEDS:
        checks.tick()
        plan, unused_allocations = h.rebuild_plan(labels, seed)
        saved = bundle.npz(f'plan-s{seed}.npz')
        require(set(saved) == set(plan), 'plan array roster')
        for key, value in plan.items():
            checks.array(saved[key], value, 'original plan/' + key)
        schedules = rebuild_schedules(plan['continuation_batches'], plan['train_true'], seed)
        saved = bundle.npz(f'schedules-s{seed}.npz')
        require(set(saved) == set(schedules), 'schedule array roster')
        for key, value in schedules.items():
            checks.array(saved[key], value, 'reconstructed schedules/' + key)
        probe = probe_positions(plan, plan['diffuse_targets'], seed)
        probe_plan = {'majority': probe['majority'], 'rare8': probe['rare8'], 'wrong': probe['wrong_assigned']}
        saved = bundle.npz(f'probes-s{seed}.npz')
        require(set(saved) == set(probe_plan), 'probe plan roster')
        for key, value in probe_plan.items():
            checks.array(saved[key], value, 'randomized probe plan/' + key)
        anchors = {schedule: selected_anchors(schedules[schedule + '_rare_counts']) for schedule in SCHEDULES}
        plan_hashes = {key: h.array_hash(value) for key, value in plan.items()}
        schedule_hashes = {key: h.array_hash(value) for key, value in schedules.items()}
        meta = bundle.json(f'plan-s{seed}.json')
        checks.equal(meta, {'schema': SCHEMA, 'seed': seed, 'plan_array_hashes': plan_hashes,
            'schedule_array_hashes': schedule_hashes, 'probe_array_hashes': {key: h.array_hash(value) for key, value in probe_plan.items()},
            'block_checks': block_metadata(plan['continuation_batches'], schedules), 'anchors': anchors,
            'unused_original_sham_allocation': unused_allocations, 'numpy_version': '1.26.4',
            'rng': 'PCG64 via named SeedSequence streams in protocol', 'initialization_seed': seed}, 'plan metadata')
        warmup = bundle.tensor(f'warmup-s{seed}.pt')
        h.snapshot_arrays(warmup, 100, True)
        warmup_hash = h.tree_hash(warmup)
        verify_actions(bundle.json(f'warmup-actions-s{seed}.json'), 'raw', [0] * 100, checks, warmup=True)
        common = bundle.npz(f'common-predictions-s{seed}.npz')
        require(set(common) == {'steps', 'train', 'heldout'}, 'common prediction members')
        checks.array(common['steps'], np.array([0, 100], np.int64), 'common prediction steps')
        for key in ('train', 'heldout'):
            h.arr(common[key], np.float32, (2, 5000, 10))
        train_x = np.ascontiguousarray(images[plan['train_ids']].astype(np.float32) / np.float32(255))
        heldout_x = np.ascontiguousarray(images[plan['heldout_ids']].astype(np.float32) / np.float32(255))
        input_hashes = {'train_inputs_sha256': h.array_hash(train_x), 'heldout_inputs_sha256': h.array_hash(heldout_x),
                       'input_hash_convention': 'array_digest contiguous CPU float32 [N,784]; NumPy uint8.astype(float32)/float32(255); plan row order'}
        initial_hash = None
        for cell in CELLS:
            assigned = plan['train_true'] if cell == 'clean' else plan['diffuse_targets']
            for schedule in SCHEDULES:
                binding = bundle.json(f'binding-s{seed}-{cell}-{schedule}.json')
                expected_binding = {'schema': SCHEMA, 'seed': seed, 'cell': cell, 'schedule': schedule,
                    'plan_array_hashes': plan_hashes, 'schedule_array_hashes': schedule_hashes,
                    'warmup_state_sha256': warmup_hash, 'assigned_targets_sha256': h.array_hash(assigned),
                    'actually_changed_count': int(np.count_nonzero(assigned != plan['train_true'])),
                    'diffuse_selected_count': int(plan['diffuse_selected'].sum()) if cell == 'diffuse' else 0,
                    'anchors': anchors[schedule], **input_hashes}
                checks.equal({key: binding[key] for key in expected_binding}, expected_binding, 'binding contract')
                for field, name in {'plan': f'plan-s{seed}.npz', 'plan_metadata': f'plan-s{seed}.json',
                    'schedules': f'schedules-s{seed}.npz', 'probe_plan': f'probes-s{seed}.npz',
                    'warmup': f'warmup-s{seed}.pt', 'common_predictions': f'common-predictions-s{seed}.npz'}.items():
                    bundle.bound(binding[field], name)
                require(initial_hash is None or initial_hash == binding['initial_model_sha256'], 'initial model mismatch across new branches')
                initial_hash = binding['initial_model_sha256']
                first_raw = first_observer = None
                for policy in POLICIES:
                    checks.tick()
                    identity = f's{seed}-{cell}-{schedule}-{policy}'
                    result_row = roster[seed, cell, schedule, policy]
                    bundle.bound(result_row['curve'], f'curve-{identity}.json')
                    record = bundle.json(f'curve-{identity}.json')
                    for key, value in binding.items():
                        checks.equal(record[key], value, 'trajectory common binding/' + key)
                    require(record['policy'] == policy, 'trajectory policy identity')
                    offset = (SEEDS.index(seed) + CELLS.index(cell) + SCHEDULES.index(schedule)) % 3
                    require(record['policy_order'] == list(POLICIES[offset:] + POLICIES[:offset]), 'fixed policy execution order')
                    for field, name in {'logits': f'logits-{identity}.npz', 'actions': f'actions-{identity}.json',
                                        'final_state': f'final-{identity}.pt'}.items():
                        bundle.bound(record[field], name)
                    logits = bundle.npz(f'logits-{identity}.npz')
                    require(set(logits) == {'steps', 'train', 'heldout'}, 'full prediction members')
                    checks.array(logits['steps'], np.asarray(h.EVAL, np.int64), 'all eval steps')
                    for key in ('train', 'heldout'):
                        h.arr(logits[key], np.float32, (21, 5000, 10))
                        checks.array(logits[key][:2], common[key], 'exact shared prediction reuse/' + key)
                    require(len(record['curve']) == 21, 'complete evaluation curve')
                    derived = []
                    for i, step in enumerate(h.EVAL):
                        expected = evaluation(step, logits['train'][i], logits['heldout'][i], plan['train_true'], assigned, plan['heldout_true'])
                        checks.equal(record['curve'][i], expected, 'prediction metrics')
                        derived.append(expected)
                    checks.equal(result_row['endpoint'], derived[-1], 'results endpoint')
                    checks.equal(result_row['warmup'], derived[1], 'results warmup')
                    require(binding['warmup_seconds'] > 0 and record['branch_seconds_including_eval_and_diagnostics'] > 0, 'positive descriptive runtime')
                    checks.equal(record['warmup_plus_branch_seconds'], binding['warmup_seconds'] + record['branch_seconds_including_eval_and_diagnostics'], 'runtime sum')
                    checks.equal(result_row['warmup_plus_branch_seconds'], record['warmup_plus_branch_seconds'], 'runtime result binding')
                    actions = bundle.json(f'actions-{identity}.json')
                    verify_actions(actions, policy, schedules[schedule + '_rare_counts'], checks)
                    first = {key: value for key, value in actions[0].items() if key.endswith('sha256')}
                    checks.equal(record['first_action_binding'], first, 'first-action binding')
                    require(first_raw is None or first_raw == first['raw_gradient_sha256'], 'first raw action differs across policies within same schedule')
                    first_raw = first['raw_gradient_sha256']
                    if policy != 'raw':
                        require(first_observer is None or first_observer == first['post_observe_tracker_sha256'], 'first observer differs within schedule')
                        first_observer = first['post_observe_tracker_sha256']
                    require(len(record['diagnostics']) == (6 if policy == 'native32' else 0), 'fixed diagnostic count')
                    endpoint_diagnostic_hash = None
                    for anchor, receipt in zip(anchors[schedule], record['diagnostics']):
                        update = anchor['update']
                        stem = f'diagnostic-{identity}-u{update:04d}'
                        scientific_names.update({stem + '.json', stem + '.pt'})
                        bundle.bound(receipt, stem + '.json')
                        report = bundle.json(stem + '.json')
                        bundle.bound(report['tensor'], stem + '.pt')
                        payload = bundle.tensor(stem + '.pt')
                        diagnostic = verify_diagnostic(payload, report, plan, assigned, train_x, warmup, anchor, (seed, cell, schedule), checks)
                        if update == 101:
                            require(h.tree_hash(payload['raw_training_gradient']) == first['raw_gradient_sha256']
                                    and h.tree_hash(payload['before']['tracker']) == first['post_observe_tracker_sha256'], 'first native diagnostic binding')
                        if update == 2000:
                            endpoint_diagnostic_hash = diagnostic['after_cleared_sha256']
                        for key, vector in (('raw_norm', 'raw_training_gradient'), ('applied_norm', 'applied_training_gradient')):
                            checks.equal(actions[update - 101][key], float(np.linalg.norm(h.arr(payload[vector]).astype(np.float64))), 'diagnostic action scalar')
                        diagnostics.append(diagnostic)
                        del payload
                    final = bundle.tensor(f'final-{identity}.pt')
                    h.snapshot_arrays(final, 2000, policy != 'raw')
                    require(h.tree_hash(final) == record['final_state_sha256'], 'full final state hash')
                    if endpoint_diagnostic_hash is not None:
                        require(h.tree_hash(final) == endpoint_diagnostic_hash, 'actual update2000 diagnostic matches final cleared state')
                    rebuilt.append({'seed': seed, 'cell': cell, 'schedule': schedule, 'policy': policy,
                                    'endpoint': derived[-1], 'warmup': derived[1]})
                    del final, logits, actions, record
    require(set(bundle.index) == scientific_names and len(diagnostics) == 72, 'exact reconstructed diagnostic artifact roster')
    summary = aggregate(rebuilt)
    checks.equal(result['summary'], summary, 'all paired summary fields')
    bundle.recheck()
    return summary, diagnostics


def resource_guard():
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        require(os.environ.get(name) == '1', 'one CPU math thread required: ' + name)
    group = next(line.split(':', 2)[2] for line in Path('/proc/self/cgroup').read_text().splitlines() if line.startswith('0::'))
    require(Path(group).name == UNIT, 'unexpected audit service')
    directory = Path('/sys/fs/cgroup') / group.lstrip('/')
    limits = {name: (directory / name).read_text().strip() for name in ('memory.max', 'memory.swap.max', 'cpu.max')}
    quota, period = limits['cpu.max'].split()
    require(limits['memory.max'] == str(4 * 1024**3) and limits['memory.swap.max'] == '0'
            and quota != 'max' and int(quota) == int(period) > 0, 'audit cgroup envelope')
    output = subprocess.check_output(['systemctl', '--user', 'show', UNIT, '--property=Type',
        '--property=RuntimeMaxUSec', '--property=Restart', '--property=KillMode'], text=True)
    service = dict(line.split('=', 1) for line in output.splitlines())
    require(service == {'Type': 'exec', 'RuntimeMaxUSec': '5min', 'Restart': 'no', 'KillMode': 'control-group'}, 'audit service envelope')
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    require(torch.__version__ == '2.11.0+cu128' and np.__version__ == '1.26.4', 'pinned audit numerical environment')
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
    require(args.execute, 'scientific saved-data audit needs explicit --execute')
    require(not args.acquisition_dir.is_symlink(), 'acquisition symlink forbidden')
    parent = args.output_dir.parent.resolve(strict=True)
    require(args.output_dir == parent / 'audit-001' and not args.output_dir.exists() and not args.output_dir.is_symlink()
            and not parent.is_relative_to(args.acquisition_dir.resolve()), 'exclusive audit output outside acquisition required')
    admission = resource_guard()
    pins = h.read_json(args.expected_sources_json)
    own_paths = [Path(__file__), HELPER_PATH, ROOT / 'tests/test_audit_spectral_batch_composition.py',
                 ROOT / DOCS / 'audit-implementation.md']
    own_pins = {str(p.relative_to(ROOT)): h.sha256(p) for p in own_paths}
    pin_file_sha = h.sha256(args.expected_sources_json)
    args.output_dir.mkdir(exist_ok=False)
    checks, bundle = h.Checks(), None
    status, errors, summary, diagnostics = 'PASS', [], None, None
    try:
        bundle = Bundle(args.acquisition_dir, args.expected_complete_sha256, args.expected_manifest_sha256, pins, checks)
        summary, diagnostics = audit(bundle, checks)
        require({str(p.relative_to(ROOT)): h.sha256(p) for p in own_paths} == own_pins, 'checker sources changed during audit')
        require(h.sha256(args.expected_sources_json) == pin_file_sha, 'main source-pin file changed during audit')
    except Exception as exc:
        status = 'FAIL'
        errors = [{'type': type(exc).__name__, 'message': str(exc)}]
    payload = {'schema': 'spectral_batch_composition_independent_audit_v1', 'status': status, 'errors': errors,
        'checks': checks.count, 'max_absolute_discrepancies': checks.max_absolute, 'tolerances': h.TOLERANCES,
        'acquisition_dir': str(args.acquisition_dir.resolve()), 'expected_completion_sha256': args.expected_complete_sha256,
        'expected_manifest_sha256': args.expected_manifest_sha256, 'expected_sources': pins,
        'expected_sources_file_sha256': pin_file_sha, 'checker_sources': own_pins,
        'input_receipts': None if bundle is None else list(bundle.index.values()),
        'independent_summary': summary, 'diagnostics': diagnostics,
        'trajectories_checked': 36 if status == 'PASS' else None,
        'evaluations_recomputed': 756 if status == 'PASS' else None,
        'native_events_checked': 72 if status == 'PASS' else None,
        'elapsed_seconds': time.monotonic() - checks.started,
        'resource_admission': admission, 'process_high_water_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'scope_limit': 'Saved-data arithmetic/provenance only; no model, inference, gradient estimation, optimizer execution or old audit replay.'}
    data = (json.dumps(payload, indent=2, allow_nan=False) + '\n').encode()
    require(len(data) <= 100 * 1024**2, 'audit output cap exceeded')
    with (args.output_dir / 'result.json').open('xb') as handle:
        handle.write(data)
    print(json.dumps({'status': status, 'checks': checks.count, 'errors': errors,
                      'result_sha256': h.sha256(args.output_dir / 'result.json')}), flush=True)
    if status != 'PASS':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
