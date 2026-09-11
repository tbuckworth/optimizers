#!/usr/bin/env python3
"""Independent saved-array observer-pathway audit; never replays an observer.

Only pinned independent arithmetic helpers are imported. No producer, model,
forward/backward, optimizer object, GPU API or observation-stream execution.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import subprocess
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HELPER_PATH = ROOT / 'scripts/audit_spectral_batch_composition.py'
HELPER_SHA = 'fd3f89d2f730d517ddbcfd682bfe1e7e7a34d1e172e0fdfb40f0d00db812c137'
if hashlib.sha256(HELPER_PATH.read_bytes()).hexdigest() != HELPER_SHA:
    raise RuntimeError('pinned independent helper changed')
_spec = importlib.util.spec_from_file_location('independent_batch_arithmetic', HELPER_PATH)
b = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(b)
h = b.h
require = h.require
SEEDS, CELLS, SCHEDULES = b.SEEDS, b.CELLS, b.SCHEDULES
SCHEMA = 'spectral_observer_pathway_v1'
DOCS = 'output/2026-09-10-spectral-observer-pathway'
UNIT = 'spectral-observer-pathway-audit-001.service'
PRIOR = Path('/tmp/spectral-experiment-artifacts/spectral-batch-composition-20260910.H2Ow40/acquisition-001')
PRIOR_COMPLETE = '39b4ef091e691780b46c1d7f5dcb12e108dbfd5b3b85c6abba612a93a393b60a'
PRIOR_AUDIT = PRIOR.parent / 'audit-001/result.json'
PRIOR_AUDIT_SHA = '3640269898d6c4115adbac925586fb54a6f1d47b0000b7e511a2fa1f1fcdeb6b'
PARAMETERS = 50890
ACTIONS = ('native_interleaved', 'native_grouped', 'raw', 'zero')
TOLERANCES = {**h.TOLERANCES, 'block_mean_epsilon_multiplier': 128,
              'block_mean_absolute_floor': 1e-10, 'final_observer_mean_epsilon_multiplier': 16,
              'saved_mean_atol': 1e-12, 'saved_mean_rtol': 1e-10}
SOURCE_NAMES = {'spectral_filter.py', 'experiments/spectral_selectivity_boundary.py',
    'experiments/spectral_batch_composition.py', 'experiments/spectral_observer_pathway.py',
    'tests/test_spectral_observer_pathway.py', DOCS + '/protocol.md',
    'output/2026-09-10-spectral-batch-composition/next-discriminator.md',
    'output/2026-09-10-spectral-batch-composition/next-design-review.md',
    'output/2026-09-06-spectral-optimizer-investigation/continuation/iteration-009/neural_core.py'}


def observer_arrays(observer, expected_step, n_params=PARAMETERS):
    """Validate the recorded observer clock without modifying any snapshot."""
    require(type(observer) is dict and type(observer['step_count']) is int
            and observer['step_count'] == expected_step, 'separate observer counter')
    fixed = {'rank': 32, 'decay': .99, 'warmup': 100, 'stable_update': True,
             'stabilize_every': 100, 'relative_eig_tol': 1e-8, 'absolute_eig_floor': 0.,
             'weighting': 'hard', 'normalize': 'none', 'adaptive': 'none',
             'proj_k': None, 'filter_strength': 1., 'energy_threshold': None}
    for key, expected in fixed.items():
        require(observer[key] == expected, 'observer configuration: ' + key)
    require(observer['n_params'] == n_params, 'observer parameter count')
    mean = h.arr(observer['grad_mean'], np.float32, (n_params,))
    v, singular = observer['V'], observer['S']
    if v is None:
        require(singular is None, 'absent basis requires absent singular values')
    else:
        v = h.arr(v, np.float32)
        require(v.ndim == 2 and v.shape[0] == n_params and 0 < v.shape[1] <= 32, 'observer V shape')
        singular = h.arr(singular, np.float64, (v.shape[1],))
        require((singular >= 0).all(), 'observer singular values')
    return mean, v


def snapshot_arrays(state, adam_step, observer_step, gradients_present=False, shapes=None):
    """Separate-clock validation: Adam 100/101 is NOT coerced to observer time."""
    shapes = h.MODEL_SHAPES if shapes is None else shapes
    require(state['schema'] == 'i9_neural_snapshot_v1', 'snapshot content schema')
    require(state['model_spec'] == {'input_dim': 784, 'width': 64, 'classes': 10}, 'model specification')
    require(list(state['model_state']) == list(shapes) and len(state['gradients']) == len(shapes), 'parameter/gradient order')
    parameters = []
    for i, (name, shape) in enumerate(shapes.items()):
        parameters.append(h.arr(state['model_state'][name], np.float32, shape).reshape(-1))
        if gradients_present:
            h.arr(state['gradients'][i], np.float32, shape)
        else:
            require(state['gradients'][i] is None, 'explicitly cleared snapshot gradients required')
    optimizer = state['optimizer']
    require(len(optimizer['param_groups']) == 1, 'one saved Adam group')
    group = optimizer['param_groups'][0]
    fixed = {'lr': .001, 'weight_decay': .01, 'betas': (.9, .999), 'eps': 1e-8,
             'foreach': False, 'fused': False, 'amsgrad': False, 'maximize': False,
             'capturable': False, 'differentiable': False}
    for key, expected in fixed.items():
        require(group[key] == expected, 'saved Adam setting: ' + key)
    ids = group['params']
    require(len(ids) == len(shapes) and len(set(ids)) == len(ids)
            and set(ids) == set(optimizer['state']), 'Adam parameter IDs/order')
    first, second = [], []
    for ident, shape in zip(ids, shapes.values()):
        row = optimizer['state'][ident]
        require(set(row) == {'step', 'exp_avg', 'exp_avg_sq'}, 'Adam moment fields')
        require(float(h.arr(row['step'], shape=())) == adam_step, 'separate Adam step counter')
        first.append(h.arr(row['exp_avg'], np.float32, shape).reshape(-1))
        value = h.arr(row['exp_avg_sq'], np.float32, shape).reshape(-1)
        require((value >= 0).all(), 'nonnegative saved Adam second moment')
        second.append(value)
    if observer_step is None:
        require(state['tracker'] is None, 'unexpected observer in action state')
    else:
        observer_arrays(state['tracker'], observer_step, sum(x.size for x in parameters))
    return tuple(np.concatenate(parts) for parts in (parameters, first, second))


def stream_means(interleaved, grouped):
    left, right = h.arr(interleaved, np.float32), h.arr(grouped, np.float32)
    require(left.ndim == right.ndim == 2 and left.shape == right.shape and len(left) == 50, 'two full 50-gradient streams')
    x, y = left.astype(np.float64), right.astype(np.float64)
    means = [x.mean(axis=0), y.mean(axis=0)]
    difference = float(np.linalg.norm(means[0] - means[1]))
    scale = float((np.linalg.norm(x, axis=1).sum() + np.linalg.norm(y, axis=1).sum()) / 100)
    bound = 128 * h.EPS32 * scale + 1e-10
    require(difference <= bound, 'fixed-theta stream-mean equality exceeds prospective tolerance')
    common = ((means[0] + means[1]) / 2).astype(np.float32)
    return means, common, {'difference_l2': difference, 'average_batch_gradient_norm': scale,
        'bound': bound, 'epsilon_multiplier': 128, 'absolute_floor': 1e-10,
        'interleaved_mean_norm': float(np.linalg.norm(means[0])),
        'grouped_mean_norm': float(np.linalg.norm(means[1]))}


def native_action(gradient, observer, checks, recorded=None):
    raw = h.arr(gradient, np.float32)
    require(raw.ndim == 1, 'one common readout gradient')
    _, v = observer_arrays(observer, observer['step_count'], len(raw))
    target = raw.astype(np.float64) if v is None else v.astype(np.float64) @ (v.astype(np.float64).T @ raw.astype(np.float64))
    if recorded is None:
        return target
    got = h.arr(recorded, np.float32, raw.shape)
    error = float(np.linalg.norm(got.astype(np.float64) - target))
    bound = 3e-5 * float(np.linalg.norm(raw.astype(np.float64))) + 1e-12
    require(error <= bound, 'native action exceeds fixed FP32 roundoff bound')
    checks.count += 1
    return {'action_l2_error': error, 'action_l2_bound': bound}


def absolute_utilities(before, after):
    """Signed useful improvement: lower CE, higher accuracy; no effect gating."""
    left, right = b.endpoint(before), b.endpoint(after)
    require(set(left) == set(right), 'readout outcome keys')
    result = {}
    for key in left:
        require((left[key] is None) == (right[key] is None), 'asymmetric absent outcome')
        result[key] = None if left[key] is None else (left[key] - right[key] if key.endswith('_ce') else right[key] - left[key])
    return result


def envelope_arrays(envelope, adam_step, observer_step, gradients_present, shapes=None):
    require(type(envelope) is dict and set(envelope) == {'schema', 'model_adam', 'observer', 'clocks'}, 'diagnostic envelope schema')
    require(envelope['schema'] == SCHEMA and envelope['clocks'] == {'adam': adam_step, 'observer': observer_step}, 'separate envelope clocks')
    require(type(envelope['clocks']['adam']) is int and (observer_step is None or type(envelope['clocks']['observer']) is int), 'integer envelope counters')
    values = snapshot_arrays(envelope['model_adam'], adam_step, None, gradients_present, shapes)
    if observer_step is None:
        require(envelope['observer'] is None, 'raw/zero envelope has no observer')
    else:
        observer_arrays(envelope['observer'], observer_step, len(values[0]))
    return values


def verify_step(before, after, parent, delivered, observer_step, checks, shapes=None):
    shapes = h.MODEL_SHAPES if shapes is None else shapes
    old, m0, v0 = envelope_arrays(before, 100, observer_step, True, shapes)
    new, m1, v1 = envelope_arrays(after, 101, observer_step, True, shapes)
    snapshot_arrays(parent, 100, 100, False, shapes)
    require(h.tree_hash(before['observer']) == h.tree_hash(after['observer']), 'one-step observer mutation')
    before, after = before['model_adam'], after['model_adam']
    for key in ('model_state', 'optimizer', 'rng', 'model_modes', 'model_spec'):
        require(h.tree_hash(before[key]) == h.tree_hash(parent[key]), 'same-parent before identity: ' + key)
    for key in ('tracker', 'rng', 'model_modes', 'model_spec', 'gradients'):
        require(h.tree_hash(before[key]) == h.tree_hash(after[key]), 'one-step unintended mutation: ' + key)
    gradient = h.arr(delivered, np.float32, old.shape)
    checks.array(np.concatenate([h.arr(g).reshape(-1) for g in before['gradients']]), gradient, 'actual delivered gradient')
    return h.verify_adam(old, new, m0, v0, m1, v1, gradient, 101, checks)


def vector_pair(left, right):
    x, y = h.arr(left).astype(np.float64), h.arr(right).astype(np.float64)
    require(x.ndim == y.ndim == 1 and x.shape == y.shape, 'paired vector shape')
    nx, ny = float(np.linalg.norm(x)), float(np.linalg.norm(y))
    return {'left_norm': nx, 'right_norm': ny, 'difference_norm': float(np.linalg.norm(x - y)),
            'cosine': float(x @ y) / (nx * ny) if nx and ny else None, 'zero_norm': nx == 0 or ny == 0}


def verify_history(data, report, parent, common, oracles, checks, shapes=None):
    shapes = h.MODEL_SHAPES if shapes is None else shapes
    require(set(data) == {'at150', 'at151', 'pre_action', 'post_action'}, 'history tensor members')
    result = {}
    for clock, phase, action in ((150, 'pre_inclusion', 'pre_action'), (151, 'post_inclusion', 'post_action')):
        state = data['at' + str(clock)]
        envelope_arrays(state, 100, clock, False, shapes)
        require(h.tree_hash(state) == report['at' + str(clock) + '_sha256'], 'history envelope hash')
        for key in ('schema', 'model_state', 'model_spec', 'model_modes', 'optimizer', 'gradients', 'rng'):
            require(h.tree_hash(state['model_adam'][key]) == h.tree_hash(parent[key]), 'fixed history parent: ' + key)
        observer = state['observer']
        v = None if observer['V'] is None else h.arr(observer['V'], np.float32)
        q, basis = h.span_basis(v)
        checks.equal(report[phase]['basis'], basis, 'history numerical basis', atol=1e-10, rtol=1e-8)
        expected = {name: b.mean_geometry(g, v, q) for name, g in {'block': common, **oracles}.items()}
        checks.equal(report[phase]['geometry'], expected, 'history native geometry', atol=1e-7, rtol=5e-4)
        for name in expected:
            for key in ('squared_norm', 'numerical_span_squared_norm', 'numerical_span_retention'):
                checks.equal(report[phase]['geometry'][name][key], expected[name][key], 'history FP64 geometry', atol=1e-10, rtol=1e-8)
        result[phase] = {'basis': basis, 'geometry': expected,
                         **native_action(common, observer, checks, data[action])}
    old_mean = h.arr(data['at150']['observer']['grad_mean'], np.float32).astype(np.float64)
    new_mean = h.arr(data['at151']['observer']['grad_mean'], np.float32).astype(np.float64)
    input64 = h.arr(common, np.float32).astype(np.float64)
    expected_mean = .99 * old_mean + .01 * input64
    mean_error = np.abs(new_mean - expected_mean)
    mean_bound = 16 * h.EPS32 * (.99 * np.abs(old_mean) + .01 * np.abs(input64)) + 1e-30
    require(np.all(mean_error <= mean_bound), 'final common-observation mean recurrence exceeds fixed FP32 bound')
    result['final_mean_recurrence'] = {'max_absolute_error': float(mean_error.max()),
                                      'max_error_over_bound': float(np.max(mean_error / mean_bound))}
    checks.count += 1
    checks.equal(report['self_inclusion_action_comparison'], vector_pair(data['pre_action'], data['post_action']), 'self inclusion action comparison')
    require(report['observer_only_updates'] == 50 and report['self_inclusion_updates'] == 1
            and report['model_adam_preservation'] == report['nonaliasing'] == 'PASS', 'history execution metadata')
    return result


def verify_readout(data, report, parent, delivered, observer, oracles, checks, shapes=None):
    shapes = h.MODEL_SHAPES if shapes is None else shapes
    require(set(data) == {'inherited', 'before', 'after', 'delivered_gradient', 'parameter_order'}, 'readout tensor members')
    step = None if observer is None else 151
    require(data['parameter_order'] == list(shapes), 'readout parameter order')
    envelope_arrays(data['inherited'], 100, step, False, shapes)
    require(h.tree_hash(data['inherited']['observer']) == h.tree_hash(observer), 'readout observer binding')
    for name in ('inherited', 'before', 'after'):
        require(h.tree_hash(data[name]) == report[name + '_sha256'], 'readout envelope hash')
    checks.array(data['delivered_gradient'], h.arr(delivered, np.float32), 'readout delivered action')
    for key in ('schema', 'model_state', 'model_spec', 'model_modes', 'optimizer', 'rng'):
        require(h.tree_hash(data['inherited']['model_adam'][key]) == h.tree_hash(parent[key]), 'inherited parent field')
        require(h.tree_hash(data['inherited']['model_adam'][key]) == h.tree_hash(data['before']['model_adam'][key]), 'gradient assignment mutation')
    require(h.tree_hash(data['inherited']['observer']) == h.tree_hash(data['before']['observer']), 'gradient assignment observer mutation')
    adam = verify_step(data['before'], data['after'], parent, data['delivered_gradient'], step, checks, shapes)
    before = data['before']['model_adam']
    after = data['after']['model_adam']
    old = np.concatenate([h.arr(before['model_state'][key]).reshape(-1) for key in shapes])
    new = np.concatenate([h.arr(after['model_state'][key]).reshape(-1) for key in shapes])
    v = None if observer is None or observer['V'] is None else h.arr(observer['V'], np.float32)
    q, basis = (None, None) if observer is None else h.span_basis(v)
    checks.equal(report['basis'], basis, 'readout numerical basis', atol=1e-10, rtol=1e-8)
    vectors, movement = h.movements(old, new, q)
    checks.equal(report['movement'], movement, 'readout displacement', atol=1e-10, rtol=1e-8)
    utilities = {name: {part: -float(h.arr(g).astype(np.float64) @ delta) for part, delta in vectors.items()}
                 for name, g in oracles.items()}
    checks.equal(report['oracle_signed_utilities'], utilities, 'oracle signed local utilities')
    require(report['parent_sha256'] == h.tree_hash(parent) and report['physical_optimizer_steps'] == 1
            and report['new_observer_updates'] == 0 and report['preservation'] == report['nonaliasing'] == 'PASS'
            and report['gradient_semantics'] == 'explicit assigned gradient retained in before and after; inherited .grad saved separately', 'readout semantics')
    return {'adam': adam, 'basis': basis, 'movement': movement, 'oracle_signed_utilities': utilities}


def aggregate(cases):
    roster = {(row['seed'], row['cell']): row for row in cases}
    require(len(cases) == len(roster) == 6 and set(roster) == {(s, c) for s in SEEDS for c in CELLS}, 'six unique complete cases')
    keys = set(cases[0]['actions']['raw']['improvements'])
    for row in cases:
        require(set(row['actions']) == set(ACTIONS), 'complete logical action roster')
        require(all(set(action['improvements']) == keys for action in row['actions'].values()), 'uniform metric roster')
    absolute, contrasts = {}, {}
    for cell in CELLS:
        for action in ACTIONS:
            absolute[cell + '/' + action] = {key: h.three_seed([roster[s, cell]['actions'][action]['improvements'][key] for s in SEEDS]) for key in sorted(keys)}
        for left, right in (('native_grouped', 'native_interleaved'), ('native_grouped', 'raw'),
                            ('native_interleaved', 'raw'), ('native_grouped', 'zero'),
                            ('native_interleaved', 'zero'), ('raw', 'zero')):
            contrasts[cell + '/' + left + '_minus_' + right] = {
                key: h.three_seed([b.difference(roster[s, cell]['actions'][left]['improvements'][key],
                                               roster[s, cell]['actions'][right]['improvements'][key]) for s in SEEDS]) for key in sorted(keys)}
    return {'seeds': list(SEEDS), 'absolute_improvements': absolute, 'improvement_contrasts': contrasts,
            'orientation': 'CE:before-minus-after; accuracy:after-minus-before; positive is favorable for both',
            'primary_contrast': 'native_grouped_minus_native_interleaved',
            'primary_metrics': ['rare_ce', 'majority_macro_ce'], 'adam_before': 100, 'adam_after': 101,
            'native_observer_delivery': 151, 'physical_readouts': 21, 'logical_case_readouts': 24}


def parent_envelope(parent):
    model_adam = dict(parent)
    model_adam['tracker'] = None
    return {'schema': SCHEMA, 'model_adam': model_adam, 'observer': parent['tracker'],
            'clocks': {'adam': 100, 'observer': 100}}


def verify_preservation(row, parent, position=None):
    expected = h.tree_hash(parent_envelope(parent))
    require(row['before_sha256'] == row['after_sha256'] == expected
            and row['parent_sha256'] == h.tree_hash(parent)
            and row['nonaliasing'] == row['preservation'] == 'PASS', 'fixed-model gradient neutrality')
    if position is not None:
        require(type(row['position']) is int and row['position'] == position, 'stream preservation order')


def verify_sources(pins):
    require(type(pins) is dict and set(pins) == SOURCE_NAMES, 'exact producer source roster')
    for name, expected in pins.items():
        path = ROOT / name
        require(path.resolve().is_relative_to(ROOT), 'producer source escape')
        cursor = path
        while cursor != ROOT:
            require(not cursor.is_symlink(), 'source symlink forbidden')
            cursor = cursor.parent
        require(path.is_file() and h.sha256(path) == expected, 'source pin mismatch: ' + name)


def physical_ids():
    return {f's{seed}-zero' for seed in SEEDS} | {
        f's{seed}-{cell}-{action}' for seed in SEEDS for cell in CELLS for action in ACTIONS if action != 'zero'}


def artifact_names():
    names = {'manifest.json', 'results.json'} | {f'oracles-s{seed}.pt' for seed in SEEDS}
    for identity in physical_ids():
        names.update({f'readout-{identity}.{suffix}' for suffix in ('pt', 'json', 'npz')})
    for seed in SEEDS:
        for cell in CELLS:
            key = f's{seed}-{cell}'
            names.update({f'common-action-{key}.pt', f'mean-admission-{key}.json', f'case-{key}.json'})
            for schedule in SCHEDULES:
                names.add(f'stream-{key}-{schedule}.pt')
                names.update({f'observer-{key}-{schedule}.{suffix}' for suffix in ('pt', 'json')})
    return names


class PriorInputs:
    """Only accepted parent inputs; old scientific results are never recomputed."""
    def __init__(self, checks):
        self.checks, self.used = checks, {}
        require(h.sha256(PRIOR / 'complete.json') == PRIOR_COMPLETE and h.sha256(PRIOR_AUDIT) == PRIOR_AUDIT_SHA, 'accepted ancestor hashes')
        complete, audit = h.read_json(PRIOR / 'complete.json'), h.read_json(PRIOR_AUDIT)
        require(complete['schema'] == 'spectral_batch_composition_v1' and complete['status'] == 'complete'
                and complete['completed_trajectories'] == 36 and complete['completed_diagnostics'] == 72, 'ancestor completion')
        require(audit['status'] == 'PASS' and audit['errors'] == []
                and audit['expected_completion_sha256'] == PRIOR_COMPLETE
                and audit['acquisition_dir'] == str(PRIOR)
                and (audit['evaluations_recomputed'], audit['trajectories_checked'], audit['native_events_checked']) == (756, 36, 72), 'accepted ancestor audit contract')
        require(complete['receipts'] == audit['input_receipts'], 'ancestor receipt identity')
        self.index = {item['path']: item for item in complete['receipts']}
        require(len(self.index) == len(complete['receipts']) == 323, 'ancestor unique receipt roster')
        self.manifest = self.json('manifest.json')
        require(self.manifest['source_pins'] == complete['source_pins'] == audit['expected_sources']
                and self.manifest['data_pins'] == h.DATA_PINS, 'ancestor source/data binding')
        b.verify_sources(self.manifest['source_pins'])

    def file(self, name):
        require(name in self.index, 'unreceipted parent input')
        path = h.verify_receipt(PRIOR, self.index[name])
        self.used[name] = self.index[name]
        self.checks.tick()
        return path

    def json(self, name):
        return h.read_json(self.file(name))

    def tensor(self, name):
        return h.cpu_tensor_file(self.file(name))

    def npz(self, name):
        return h.load_npz(self.file(name))

    def bound(self, item, name):
        require(name in self.index and item == self.index[name], 'ancestor receipt cross-binding')

    def recheck(self):
        require(h.sha256(PRIOR / 'complete.json') == PRIOR_COMPLETE and h.sha256(PRIOR_AUDIT) == PRIOR_AUDIT_SHA, 'ancestor completion/audit changed')
        for name in list(self.used):
            self.file(name)
        b.verify_sources(self.manifest['source_pins'])


class Bundle:
    def __init__(self, root, complete_sha, manifest_sha, pins, checks):
        self.root = Path(root).resolve(strict=True)
        self.pins, self.checks, self.complete_sha = pins, checks, complete_sha
        self.complete_path = h.contained_file(self.root, 'complete.json')
        require(h.sha256(self.complete_path) == complete_sha, 'main completion hash')
        c = h.read_json(self.complete_path)
        self.complete = c
        fixed = {'schema': SCHEMA, 'status': 'complete', 'completed_cases': 6,
            'completed_stream_gradients': 600, 'completed_oracle_gradients': 6,
            'completed_physical_readouts': 21, 'completed_prediction_pairs': 21,
            'completed_observer_only_updates': 600, 'completed_self_inclusion_updates': 12,
            'parent_completion_sha256': PRIOR_COMPLETE, 'parent_audit_sha256': PRIOR_AUDIT_SHA}
        checks.equal({key: c[key] for key in fixed}, fixed, 'complete fixed roster')
        require(c['source_pins'] == pins and c['data_pins'] == h.DATA_PINS, 'completion source/data binding')
        self.index = {}
        for item in c['receipts']:
            checks.tick()
            path = h.verify_receipt(self.root, item)
            require(path.name not in self.index, 'duplicate receipt')
            self.index[path.name] = item
        require(set(self.index) == artifact_names(), 'exact new artifact roster')
        require({path.name for path in self.root.iterdir()} == set(self.index) | {'complete.json'}, 'unreceipted acquisition file')
        size = sum(item['size_bytes'] for item in self.index.values())
        require(size == c['artifact_bytes_before_completion'] and size + self.complete_path.stat().st_size <= 1024**3, '1 GiB output inventory')
        require(0 < c['wall_seconds'] <= 480 and c['process_high_water_rss_kib'] <= 16 * 1024**2
                and c['gpu_max_allocated_bytes'] <= 8 * 1024**3, 'acquisition resource completion')
        require(self.index['manifest.json']['sha256'] == manifest_sha, 'main manifest hash')
        self.manifest = self.json('manifest.json')
        m = self.manifest
        require(m['schema'] == SCHEMA and m['source_pins'] == pins and m['data_pins'] == h.DATA_PINS, 'manifest binding')
        fixed = {'seeds': list(SEEDS), 'cells': list(CELLS), 'schedules': list(SCHEDULES), 'actions': list(ACTIONS),
            'parent_acquisition': str(PRIOR), 'parent_completion_sha256': PRIOR_COMPLETE, 'parent_audit_sha256': PRIOR_AUDIT_SHA,
            'stream_gradients': 600, 'oracle_gradients': 6, 'physical_readouts': 21, 'new_prediction_pairs': 21,
            'observer_clocks': [100, 150, 151], 'adam_clocks': [100, 101],
            'torch': '2.11.0+cu128', 'numpy': '1.26.4', 'cooperative_seconds': 480,
            'data_directory': str(h.DATA), 'cloud_spend_usd': 0}
        checks.equal({key: m[key] for key in fixed}, fixed, 'manifest scientific contract')
        require(m['byte_inventory']['cap_bytes'] == 1024**3 and m['byte_inventory']['total_upper_bytes'] < 1024**3, 'prospective inventory cap')
        limits, service = m['resource_admission']['effective'], m['resource_admission']['service']
        quota, period = limits['cpu.max'].split()
        require(limits['memory.max'] == str(16 * 1024**3) and limits['memory.swap.max'] == '0'
                and quota != 'max' and int(quota) == int(period) > 0, 'acquisition cgroup limits')
        require(service == {'Type': 'exec', 'RuntimeMaxUSec': '10min', 'Restart': 'no', 'KillMode': 'control-group'}, 'acquisition service limits')
        self.bound(c['results'], 'results.json')
        verify_sources(pins)

    def bound(self, item, name):
        require(name in self.index and item == self.index[name], 'artifact cross-binding: ' + name)

    def json(self, name):
        require(name in self.index, 'unreceipted JSON')
        return h.read_json(self.root / name)

    def tensor(self, name):
        require(name in self.index, 'unreceipted tensor archive')
        return h.cpu_tensor_file(self.root / name)

    def npz(self, name):
        require(name in self.index, 'unreceipted NPZ')
        return h.load_npz(self.root / name)

    def recheck(self):
        for item in self.index.values():
            self.checks.tick()
            h.verify_receipt(self.root, item)
        require(h.sha256(self.complete_path) == self.complete_sha, 'completion changed')
        verify_sources(self.pins)
        for name, expected in h.DATA_PINS.items():
            require(h.sha256(h.DATA / name) == expected, 'training IDX bytes changed')


def identity(obj, seed, cell=None, schedule=None):
    expected = {'schema': SCHEMA, 'seed': seed}
    if cell is not None:
        expected['cell'] = cell
    if schedule is not None:
        expected['schedule'] = schedule
    require(all(obj[key] == value for key, value in expected.items()), 'scientific artifact identity')


def audit(bundle, prior, checks):
    images, labels = h.read_pinned_data()
    for key in ('torch', 'numpy', 'python'):
        require(bundle.manifest[key] == prior.manifest[key], 'archived prediction environment: ' + key)
    recorded_results = bundle.json('results.json')
    require(recorded_results['schema'] == SCHEMA, 'results schema')
    rows = recorded_results['cases']
    roster = {(row['seed'], row['cell']): row for row in rows}
    require(len(rows) == len(roster) == 6 and set(roster) == {(s, c) for s in SEEDS for c in CELLS}, 'six exact result cases')
    checked_cases, checked_histories, checked_readouts, physical = [], [], [], set()
    for seed in SEEDS:
        checks.tick()
        parent = prior.tensor(f'warmup-s{seed}.pt')
        snapshot_arrays(parent, 100, 100)
        parent_sha = h.tree_hash(parent)
        plan = prior.npz(f'plan-s{seed}.npz')
        schedules = prior.npz(f'schedules-s{seed}.npz')
        probes = prior.npz(f'probes-s{seed}.npz')
        common_predictions = prior.npz(f'common-predictions-s{seed}.npz')
        for key, count in (('train_ids', 5000), ('heldout_ids', 5000)):
            h.arr(plan[key], np.int64, (count,))
            require(((plan[key] >= 0) & (plan[key] < len(labels))).all(), 'ancestor example indices')
        checks.array(plan['train_true'], labels[plan['train_ids']].astype(np.int64), 'training true labels')
        checks.array(plan['heldout_true'], labels[plan['heldout_ids']].astype(np.int64), 'heldout true labels')
        require(len(set(plan['train_ids'])) == len(set(plan['heldout_ids'])) == 5000
                and not set(plan['train_ids']) & set(plan['heldout_ids']), 'distinct disjoint source IDs')
        checks.equal(np.bincount(plan['train_true'], minlength=10).tolist(), [550] * 8 + [50, 550], 'fixed training class balance')
        checks.equal(np.bincount(plan['heldout_true'], minlength=10).tolist(), [500] * 10, 'fixed heldout class balance')
        checks.array(plan['diffuse_targets'][plan['train_true'] == 8], plan['train_true'][plan['train_true'] == 8], 'rare true targets unchanged')
        require(set(common_predictions) == {'steps', 'train', 'heldout'}, 'ancestor prediction members')
        checks.array(common_predictions['steps'], np.array([0, 100], np.int64), 'ancestor prediction times')
        for key in ('train', 'heldout'):
            h.arr(common_predictions[key], np.float32, (2, 5000, 10))
        train_x = images[plan['train_ids']].astype(np.float32) / np.float32(255)
        heldout_x = images[plan['heldout_ids']].astype(np.float32) / np.float32(255)
        bindings = {}
        for cell in CELLS:
            binding = prior.json(f'binding-s{seed}-{cell}-interleaved.json')
            for key, name in {'warmup': f'warmup-s{seed}.pt', 'plan': f'plan-s{seed}.npz',
                'schedules': f'schedules-s{seed}.npz', 'probe_plan': f'probes-s{seed}.npz',
                'common_predictions': f'common-predictions-s{seed}.npz'}.items():
                prior.bound(binding[key], name)
            assigned = plan['train_true'] if cell == 'clean' else plan['diffuse_targets']
            require(binding['seed'] == seed and binding['cell'] == cell and binding['schedule'] == 'interleaved'
                    and binding['warmup_state_sha256'] == parent_sha
                    and binding['train_inputs_sha256'] == h.array_hash(train_x)
                    and binding['heldout_inputs_sha256'] == h.array_hash(heldout_x)
                    and binding['assigned_targets_sha256'] == h.array_hash(assigned), 'ancestor prediction and target binding')
            bindings[cell] = binding
        oracle = bundle.tensor(f'oracles-s{seed}.pt')
        identity(oracle, seed)
        require(oracle['parent_sha256'] == parent_sha and oracle['use'] == 'oracle diagnostic only; never delivered or observed'
                and set(oracle['groups']) == {'majority', 'rare8'}, 'oracle-only saved groups')
        expected_probes = b.probe_positions(plan, plan['diffuse_targets'], seed)
        oracles = {}
        for name in ('majority', 'rare8'):
            positions = expected_probes[name]
            checks.array(probes[name], positions, 'accepted randomized oracle membership')
            group = oracle['groups'][name]
            for key, expected in {'training_positions': positions, 'source_ids': plan['train_ids'][positions],
                'inputs': train_x[positions], 'targets': plan['train_true'][positions]}.items():
                checks.array(group[key], expected, 'oracle ' + name + '/' + key)
            oracles[name] = h.arr(group['mean_gradient'], np.float32, (PARAMETERS,))
            verify_preservation(group['preservation'], parent)
        zero_cache = None
        for cell in CELLS:
            checks.tick()
            tag = f's{seed}-{cell}'
            case = bundle.json(f'case-{tag}.json')
            checks.equal(case, roster[seed, cell], 'results/case JSON binding')
            identity(case, seed, cell)
            require(case['parent_sha256'] == parent_sha and set(case['actions']) == set(ACTIONS), 'case parent/action roster')
            expected_parent_inputs = {name: item for name, item in prior.used.items() if str(seed) in name}
            checks.equal(case['parent_inputs'], expected_parent_inputs, 'case ancestor input receipts')
            bundle.bound(case['oracles'], f'oracles-s{seed}.pt')
            prior.bound(case['before_predictions'], f'common-predictions-s{seed}.npz')
            require(set(case['streams']) == set(case['histories']) == set(SCHEDULES), 'two observer schedules')
            assigned = plan['train_true'] if cell == 'clean' else plan['diffuse_targets']
            before = b.evaluation(100, common_predictions['train'][1], common_predictions['heldout'][1], plan['train_true'], assigned, plan['heldout_true'])
            checks.equal(case['before'], before, 'independent before metrics')
            streams = {}
            for schedule in SCHEDULES:
                name = f'stream-{tag}-{schedule}.pt'
                bundle.bound(case['streams'][schedule], name)
                stream = bundle.tensor(name)
                identity(stream, seed, cell, schedule)
                require(stream['parent_sha256'] == parent_sha and stream['clock'] == {'adam': 100, 'observer': 100}
                        and stream['observer_updates_during_measurement'] == 0, 'fixed-model stream provenance')
                expected = h.arr(schedules[schedule + '_batches'], np.int64, (1900, 64))[:50]
                checks.array(stream['memberships'], expected, 'first-block batch membership')
                checks.array(np.sort(expected.reshape(-1)), np.sort(plan['continuation_batches'][:50].reshape(-1)), 'exact occurrence multiset')
                streams[schedule] = h.arr(stream['gradients'], np.float32, (50, PARAMETERS))
                require(len(stream['preservation']) == 50, 'per-gradient preservation roster')
                for position, row in enumerate(stream['preservation']):
                    verify_preservation(row, parent, position)
            means, star, check = stream_means(streams['interleaved'], streams['grouped'])
            name = f'common-action-{tag}.pt'
            bundle.bound(case['common_action'], name)
            common = bundle.tensor(name)
            identity(common, seed, cell)
            checks.array(common['g_star'], star, 'exact common FP32 symmetric input')
            for key, value in zip(('mean_interleaved', 'mean_grouped'), means):
                got = h.arr(common[key], np.float64, (PARAMETERS,))
                require(np.all(np.abs(got - value) <= 1e-12 + 1e-10 * np.abs(value)), 'saved FP64 stream mean')
            mean_report = {'mean_difference_norm': check['difference_l2'], 'average_batch_gradient_norm': check['average_batch_gradient_norm'],
                'bound': check['bound'], 'mean_norms': [check['interleaved_mean_norm'], check['grouped_mean_norm']],
                'passed': True, 'multiplier': 128, 'absolute_floor': 1e-10,
                'g_star_norm': float(np.linalg.norm(star.astype(np.float64))), 'g_star_sha256': h.tree_hash(common['g_star'])}
            checks.equal(case['mean_admission'], mean_report, 'fixed stream-mean admission')
            admission = bundle.json(f'mean-admission-{tag}.json')
            checks.equal(admission, {'schema': SCHEMA, 'seed': seed, 'cell': cell,
                'streams': case['streams'], 'common_action': case['common_action'], **mean_report}, 'mean admission receipt binding')
            histories = {}
            for schedule in SCHEDULES:
                stem = f'observer-{tag}-{schedule}'
                bundle.bound(case['histories'][schedule], stem + '.pt')
                history, report = bundle.tensor(stem + '.pt'), bundle.json(stem + '.json')
                identity(history, seed, cell, schedule)
                identity(report, seed, cell, schedule)
                bundle.bound(report['state'], stem + '.pt')
                require(history['parent_sha256'] == parent_sha and history['g_star_sha256'] == h.tree_hash(common['g_star']), 'history parent/common input')
                data = {key: history[key] for key in ('at150', 'at151', 'pre_action', 'post_action')}
                result = verify_history(data, report, parent, star, oracles, checks)
                histories[schedule] = data
                checked_histories.append({'seed': seed, 'cell': cell, 'schedule': schedule, **result,
                    'self_inclusion_action_comparison': vector_pair(data['pre_action'], data['post_action'])})
            comparison = {phase: vector_pair(histories['interleaved'][phase + '_action'], histories['grouped'][phase + '_action']) for phase in ('pre', 'post')}
            checks.equal(case['history_action_comparisons'], comparison, 'common-input history action contrast')
            action_rows = {}
            for action in ACTIONS:
                row = case['actions'][action]
                require(row['shared_across_label_cells'] is (action == 'zero'), 'shared-zero logical binding')
                physical_id = f's{seed}-zero' if action == 'zero' else tag + '-' + action
                record = row['readout']
                require(record['schema'] == SCHEMA and record['physical_id'] == physical_id, 'physical action identity')
                for field, suffix in (('state', 'pt'), ('logits', 'npz'), ('report', 'json')):
                    bundle.bound(record[field], f'readout-{physical_id}.{suffix}')
                saved_report = bundle.json(f'readout-{physical_id}.json')
                checks.equal({key: value for key, value in record.items() if key != 'report'}, saved_report, 'saved physical readout report')
                delivered = np.zeros(PARAMETERS, np.float32) if action == 'zero' else star if action == 'raw' else h.arr(histories[action.removeprefix('native_')]['post_action'], np.float32)
                observer = histories[action.removeprefix('native_')]['at151']['observer'] if action.startswith('native_') else None
                if action == 'zero' and zero_cache is not None:
                    checks.equal(record, zero_cache['record'], 'same byte-bound physical zero reused')
                    predictions = zero_cache['predictions']
                else:
                    require(physical_id not in physical, 'unexpected physical readout reuse')
                    payload = bundle.tensor(f'readout-{physical_id}.pt')
                    require(payload['schema'] == SCHEMA and payload['physical_id'] == physical_id, 'readout tensor identity')
                    data = {key: payload[key] for key in ('inherited', 'before', 'after', 'delivered_gradient', 'parameter_order')}
                    diagnostic = verify_readout(data, saved_report, parent, delivered, observer, oracles, checks)
                    predictions = bundle.npz(f'readout-{physical_id}.npz')
                    require(set(predictions) == {'train', 'heldout'}, 'new prediction pair members')
                    for key in ('train', 'heldout'):
                        h.arr(predictions[key], np.float32, (5000, 10))
                    physical.add(physical_id)
                    checked_readouts.append({'physical_id': physical_id, **diagnostic})
                    if action == 'zero':
                        zero_cache = {'record': record, 'predictions': predictions}
                    del payload, data
                after = b.evaluation(101, predictions['train'], predictions['heldout'], plan['train_true'], assigned, plan['heldout_true'])
                checks.equal(row['after'], after, 'independent after metrics')
                improvements = absolute_utilities(before, after)
                checks.equal(row['improvements'], improvements, 'signed absolute useful change')
                action_rows[action] = {'after': after, 'improvements': improvements, 'readout': record,
                                       'shared_across_label_cells': action == 'zero'}
            checked_cases.append({'seed': seed, 'cell': cell, 'before': before, 'actions': action_rows,
                                  'mean_admission': mean_report, 'history_action_comparisons': comparison})
            require(h.tree_hash(parent) == parent_sha, 'loaded parent altered during independent audit')
            del streams, histories, common
    require(physical == physical_ids() and len(checked_histories) == 12 and len(checked_cases) == 6, 'complete physical/history/case audit roster')
    summary = aggregate(checked_cases)
    checks.equal(recorded_results['summary'], summary, 'full all-seed summary')
    checks.equal(bundle.complete['parent_input_receipts'], list(prior.used.values()), 'complete scoped ancestor receipts')
    prior.recheck()
    bundle.recheck()
    return summary, checked_cases, checked_histories, checked_readouts


def resource_guard():
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'GPU must be explicitly hidden from saved-data audit')
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        require(os.environ.get(name) == '1', 'one math thread required: ' + name)
    group = next(line.split(':', 2)[2] for line in Path('/proc/self/cgroup').read_text().splitlines() if line.startswith('0::'))
    require(Path(group).name == UNIT, 'unexpected saved-data audit service')
    folder = Path('/sys/fs/cgroup') / group.lstrip('/')
    limits = {name: (folder / name).read_text().strip() for name in ('memory.max', 'memory.swap.max', 'cpu.max')}
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
    require(torch.__version__ == '2.11.0+cu128' and np.__version__ == '1.26.4', 'pinned audit environment')
    return {'cgroup': group, 'effective': limits, 'service': service}


def main():
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--acquisition-dir', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--expected-complete-sha256', required=True)
    parser.add_argument('--expected-manifest-sha256', required=True)
    parser.add_argument('--expected-sources-json', required=True, type=Path)
    args = parser.parse_args()
    require(args.execute, 'explicit --execute required for scientific saved-data audit')
    require(not args.acquisition_dir.is_symlink(), 'acquisition symlink forbidden')
    output_parent = args.output_dir.parent.resolve(strict=True)
    require(args.output_dir == output_parent / 'audit-001' and not args.output_dir.exists()
            and not args.output_dir.is_symlink() and output_parent == args.acquisition_dir.resolve().parent,
            'exclusive sibling audit output')
    admission = resource_guard()
    pins = h.read_json(args.expected_sources_json)
    own_paths = [Path(__file__), HELPER_PATH, b.HELPER_PATH,
                 ROOT / 'tests/test_audit_spectral_observer_pathway.py', ROOT / DOCS / 'audit-implementation.md']
    own_pins = {str(path.relative_to(ROOT)): h.sha256(path) for path in own_paths}
    pin_file_sha = h.sha256(args.expected_sources_json)
    args.output_dir.mkdir(exist_ok=False)
    checks, bundle, prior = h.Checks(), None, None
    checks.started = started
    status, errors, summary, cases, histories, readouts = 'PASS', [], None, None, None, None
    try:
        bundle = Bundle(args.acquisition_dir, args.expected_complete_sha256, args.expected_manifest_sha256, pins, checks)
        prior = PriorInputs(checks)
        summary, cases, histories, readouts = audit(bundle, prior, checks)
        require({str(path.relative_to(ROOT)): h.sha256(path) for path in own_paths} == own_pins, 'checker sources changed')
        require(h.sha256(args.expected_sources_json) == pin_file_sha, 'main source map changed')
        checks.tick()
    except Exception as exc:
        status = 'FAIL'
        errors = [{'type': type(exc).__name__, 'message': str(exc)}]
    payload = {'schema': 'spectral_observer_pathway_independent_audit_v1', 'status': status,
        'errors': errors, 'checks': checks.count, 'max_absolute_discrepancies': checks.max_absolute,
        'tolerances': TOLERANCES, 'acquisition_dir': str(args.acquisition_dir.resolve()),
        'expected_completion_sha256': args.expected_complete_sha256,
        'expected_manifest_sha256': args.expected_manifest_sha256,
        'expected_sources': pins, 'expected_sources_file_sha256': pin_file_sha, 'checker_sources': own_pins,
        'parent_completion_sha256': PRIOR_COMPLETE, 'parent_audit_sha256': PRIOR_AUDIT_SHA,
        'input_receipts': None if bundle is None else list(bundle.index.values()),
        'parent_input_receipts': None if prior is None else list(prior.used.values()),
        'independent_summary': summary, 'checked_cases': cases, 'checked_histories': histories,
        'checked_readouts': readouts, 'elapsed_seconds': time.monotonic() - started,
        'resource_admission': admission, 'process_high_water_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'scope_limit': 'Saved-array arithmetic and receipt/provenance checks only. No neural inference, differentiation, optimizer execution, observer-stream replay or old scientific audit.'}
    for key, count in {'cases_checked': 6, 'observer_histories_checked': 12, 'stream_gradients_checked': 600,
        'oracle_gradients_checked': 6, 'physical_readouts_checked': 21, 'logical_readouts_checked': 24,
        'prediction_pairs_checked': 21}.items():
        payload[key] = count if status == 'PASS' else None
    encoded = (json.dumps(payload, indent=2, allow_nan=False) + '\n').encode()
    require(len(encoded) <= 100 * 1024**2, '100 MiB audit output cap')
    with (args.output_dir / 'result.json').open('xb') as file:
        file.write(encoded)
    print(json.dumps({'status': status, 'checks': checks.count, 'errors': errors,
                      'result_sha256': h.sha256(args.output_dir / 'result.json')}), flush=True)
    if status != 'PASS':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
