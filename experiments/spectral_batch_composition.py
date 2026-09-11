#!/usr/bin/env python3
"""Fixed multiset batch-composition study; scientific entry is explicit and one-shot.

The accepted selectivity runner is imported only for immutable numerical and
storage helpers. Its main/acquire entry points and globals are never modified.
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
import subprocess
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import spectral_selectivity_boundary as accepted

core = accepted.core
need, digest, array_digest = accepted.need, accepted.digest, accepted.array_digest
SCHEMA = 'spectral_batch_composition_v1'
DOCS = ROOT/'output/2026-09-10-spectral-batch-composition'
SEEDS = (202609121, 202609122, 202609123)
CELLS = ('clean', 'diffuse')
SCHEDULES = ('interleaved', 'grouped')
POLICIES = accepted.POLICIES
WARMUP, STEPS, BATCH, P, RANK = 100, 2000, 64, 50890, 32
BLOCK_SIZE = 50
BLOCKS = 38
ANCHOR_BLOCKS = (0, 7, 37)
UNIT = 'spectral-batch-composition-001.service'
MAX_BYTES, RESERVE_BYTES = 3*1024**3, 1024**2
DEADLINE_SECONDS = 1500
EVAL_STEPS = tuple(range(0, 2001, 100))
ACCEPTED_SHA = 'cd8e1da355e3090951428ba72d0e062281c2891bfba326f367bf6845293f1ade'


def source_pins():
    paths = [ROOT/'spectral_filter.py', Path(core.__file__), Path(accepted.__file__),
             Path(__file__), ROOT/'tests/test_spectral_batch_composition.py', DOCS/'protocol.md']
    pins = {str(path.relative_to(ROOT)): digest(path) for path in paths}
    need(pins['spectral_filter.py'] == core.FILTER_SHA256, 'canonical filter changed')
    need(pins[str(Path(core.__file__).relative_to(ROOT))] == accepted.CORE_SHA, 'accepted I9 core changed')
    need(pins['experiments/spectral_selectivity_boundary.py'] == ACCEPTED_SHA, 'accepted helper changed')
    return pins


def rare_count_vectors(total, positions, batch_size=BATCH):
    """Two partitions with identical exposure; shape overrides are fixtures only."""
    positions = np.asarray(positions)
    count = len(positions)
    need(count > 1 and batch_size > 0 and 0 <= total <= count*batch_size
         and np.array_equal(np.sort(positions), np.arange(count)), 'count-vector inputs')
    interleaved = np.full(count, total//count, dtype=np.int64)
    interleaved[positions[:total % count]] += 1
    grouped = np.zeros(count, dtype=np.int64)
    remaining = total
    for position in positions:
        grouped[position] = min(batch_size, remaining)
        remaining -= int(grouped[position])
    need(remaining == 0 and int(interleaved.sum()) == int(grouped.sum()) == total,
         'rare count conservation')
    need(int(interleaved.max()-interleaved.min()) <= 1
         and int(((grouped > 0) & (grouped < batch_size)).sum()) <= 1, 'schedule concentration rule')
    return {'interleaved': interleaved, 'grouped': grouped}


def schedule_block(original, true_labels, seed, block):
    original = np.asarray(original)
    need(original.ndim == 2 and original.dtype == np.int64 and len(original) > 1,
         'integer block matrix')
    rows, width = original.shape
    flat = original.reshape(-1)
    need(((flat >= 0) & (flat < len(true_labels))).all(), 'block training-position domain')
    occurrence_order = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, 6, block]))).permutation(len(flat))
    ordered = flat[occurrence_order]
    rare = ordered[np.asarray(true_labels)[ordered] == 8]
    common = ordered[np.asarray(true_labels)[ordered] != 8]
    position_order = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, 7, block]))).permutation(rows)
    counts = rare_count_vectors(len(rare), position_order, width)
    slot_orders = np.stack([np.random.Generator(np.random.PCG64(
        np.random.SeedSequence([seed, 8, block, t]))).permutation(width) for t in range(rows)])
    schedules = {}
    for name in SCHEDULES:
        rare_start = common_start = 0
        batches = []
        for t, rare_count in enumerate(counts[name]):
            rare_count = int(rare_count)
            common_count = width-rare_count
            sequence = np.concatenate([rare[rare_start:rare_start+rare_count],
                                       common[common_start:common_start+common_count]])
            need(sequence.shape == (width,), 'batch occurrence consumption')
            batches.append(sequence[slot_orders[t]])
            rare_start += rare_count
            common_start += common_count
        value = np.stack(batches)
        need(rare_start == len(rare) and common_start == len(common), 'whole occurrence lists consumed')
        need(np.array_equal(np.sort(value.reshape(-1)), np.sort(flat)), 'original block multiset changed')
        need(np.array_equal((np.asarray(true_labels)[value] == 8).sum(1), counts[name]), 'realized rare counts differ')
        schedules[name] = value
    return schedules, counts, {'occurrence': occurrence_order, 'positions': position_order, 'slots': slot_orders}


def make_schedules(plan, seed, block_size=BLOCK_SIZE):
    original = plan['continuation_batches']
    need(len(original) % block_size == 0, 'no short final block allowed')
    arrays = {name+'_batches': [] for name in SCHEDULES}
    arrays.update({name+'_rare_counts': [] for name in SCHEDULES})
    arrays.update({'occurrence_permutations': [], 'count_position_permutations': [], 'within_batch_permutations': []})
    checks = []
    for block, start in enumerate(range(0, len(original), block_size)):
        source = original[start:start+block_size]
        schedules, counts, permutations = schedule_block(source, plan['train_true'], seed, block)
        for name in SCHEDULES:
            arrays[name+'_batches'].append(schedules[name])
            arrays[name+'_rare_counts'].append(counts[name])
        for key, value in [('occurrence_permutations', permutations['occurrence']),
                           ('count_position_permutations', permutations['positions']),
                           ('within_batch_permutations', permutations['slots'])]:
            arrays[key].append(value)
        checks.append({'block': block, 'draws': source.size,
                       'rare_occurrences': int(counts['interleaved'].sum()),
                       'original_sorted_multiset_sha256': array_digest(np.sort(source.reshape(-1))),
                       'interleaved_sorted_multiset_sha256': array_digest(np.sort(schedules['interleaved'].reshape(-1))),
                       'grouped_sorted_multiset_sha256': array_digest(np.sort(schedules['grouped'].reshape(-1))),
                       'rare_fraction_mean': float(counts['interleaved'].mean()/source.shape[1]),
                       'rare_fraction_population_variances': {name: float(np.var(counts[name]/source.shape[1]))
                                                             for name in SCHEDULES}, 'multiset_checks': 'PASS'})
    result = {key: np.concatenate(values, axis=0) if key.endswith(('_batches', '_rare_counts'))
              else np.stack(values) for key, values in arrays.items()}
    return result, checks


def select_anchors(rare_counts, block_size=BLOCK_SIZE, blocks=ANCHOR_BLOCKS):
    counts = np.asarray(rare_counts)
    need(counts.ndim == 1 and len(counts) % block_size == 0, 'anchor count shape')
    events = []
    for block in blocks:
        values = counts[block*block_size:(block+1)*block_size]
        need(len(values) == block_size and block_size > 1, 'anchor block absent')
        high = int(np.argmax(values))
        candidates = np.array([t for t in range(block_size) if t != high])
        low = int(candidates[np.argmin(values[candidates])])
        for label, t in [('high', high), ('low', low)]:
            events.append({'block': int(block), 'position': t, 'label': label,
                           'update': WARMUP+1+block*block_size+t,
                           'rare_count': int(values[t]), 'equal_counts_in_block': bool(values.max() == values.min())})
    return sorted(events, key=lambda row: row['update'])


def make_probe_plan(plan, seed):
    true = plan['train_true']
    common_rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, 9, 0])))
    majority = []
    for digit in accepted.MAJORITY:
        pool = np.flatnonzero(true == digit)
        need(len(pool) >= 6, 'six majority probe examples per stratum required')
        majority.extend(common_rng.permutation(pool)[:6].tolist())
    rare_rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, 9, 1])))
    rare_pool = np.flatnonzero(true == 8)
    need(len(rare_pool) >= 32, '32 rare probe examples required')
    wrong_rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, 9, 2])))
    changed = np.flatnonzero(plan['diffuse_targets'] != true)
    return {'majority': np.array(majority, dtype=np.int64),
            'rare8': rare_rng.permutation(rare_pool)[:32],
            'wrong': wrong_rng.permutation(changed)[:min(32, len(changed))]}


def probe_definitions(plan, probe_plan, images, cell):
    need(cell in CELLS, 'unknown cell')
    true = plan['train_true']
    assigned = plan['diffuse_targets'] if cell == 'diffuse' else true
    result = {}
    for name, key, targets in [('majority', 'majority', true), ('rare8', 'rare8', true),
                               ('wrong_assigned', 'wrong', assigned), ('wrong_corrected', 'wrong', true)]:
        positions = probe_plan[key]
        if name.startswith('wrong') and (cell == 'clean' or not len(positions)):
            result[name] = None
        else:
            result[name] = {'training_positions': positions.copy(), 'source_ids': plan['train_ids'][positions].copy(),
                            'x': images[torch.as_tensor(positions, device=images.device)],
                            'targets': targets[positions].copy(), 'true_targets': true[positions].copy(),
                            'assigned_targets': assigned[positions].copy()}
    return result


def evaluation_row(step, train_logits, heldout_logits, true, assigned, heldout_true):
    return {'step': step, 'train_true': accepted.group_stats(train_logits, true),
            'train_assigned': accepted.classification_stats(train_logits, assigned),
            'train_actually_changed': accepted.classification_stats(train_logits, assigned, assigned != true),
            'heldout': accepted.group_stats(heldout_logits, heldout_true)}


def endpoint_metrics(row):
    metrics = {group+'_'+metric: row['heldout'][group][metric]
               for group in ('rare', 'majority_macro', 'balanced_total') for metric in ('accuracy', 'ce')}
    for prefix, source in [('heldout', 'heldout'), ('train_true', 'train_true')]:
        metrics.update({prefix+'_class'+str(digit)+'_'+metric: row[source]['per_class'][str(digit)][metric]
                        for digit in range(10) for metric in ('accuracy', 'ce')})
    metrics.update({'train_true_'+group+'_'+metric: row['train_true'][group][metric]
                    for group in ('rare', 'majority_macro', 'balanced_total') for metric in ('accuracy', 'ce')})
    for prefix, source in [('train_assigned', 'train_assigned'), ('train_wrong', 'train_actually_changed')]:
        metrics.update({prefix+'_'+metric: row[source][metric] for metric in ('accuracy', 'ce')})
    return metrics


def summarize_results(rows):
    roster = {(r['seed'], r['cell'], r['schedule'], r['policy']): r for r in rows}
    expected = {(s, c, t, p) for s in SEEDS for c in CELLS for t in SCHEDULES for p in POLICIES}
    need(len(roster) == len(rows) == 36 and set(roster) == expected, 'exact complete result roster')
    def value(seed, cell, schedule, policy, key, phase='endpoint'):
        return endpoint_metrics(roster[seed, cell, schedule, policy][phase])[key]
    def difference(left, right):
        need((left is None) == (right is None), 'asymmetric undefined metric')
        return None if left is None else left-right
    keys = list(endpoint_metrics(rows[0]['endpoint']))
    groups, changes, schedule_contrasts, policy_contrasts, interactions = {}, {}, {}, {}, {}
    summarize = accepted.sample_summary
    for cell in CELLS:
        for schedule in SCHEDULES:
            for policy in POLICIES:
                name = '/'.join([cell, schedule, policy])
                groups[name] = {key: summarize([value(s, cell, schedule, policy, key) for s in SEEDS]) for key in keys}
                changes[name] = {key: summarize([difference(value(s, cell, schedule, policy, key),
                    value(s, cell, schedule, policy, key, 'warmup')) for s in SEEDS]) for key in keys}
            for control in ('raw', 'norm_raw'):
                name = '/'.join([cell, schedule, 'native32_minus_'+control])
                policy_contrasts[name] = {key: summarize([difference(value(s, cell, schedule, 'native32', key),
                    value(s, cell, schedule, control, key)) for s in SEEDS]) for key in keys}
        for policy in POLICIES:
            name = '/'.join([cell, policy, 'grouped_minus_interleaved'])
            schedule_contrasts[name] = {key: summarize([difference(value(s, cell, 'grouped', policy, key),
                value(s, cell, 'interleaved', policy, key)) for s in SEEDS]) for key in keys}
        for policy in ('native32', 'norm_raw'):
            name = cell+'/'+policy+'_minus_raw'
            interactions[name] = {key: summarize([difference(
                difference(value(s, cell, 'grouped', policy, key), value(s, cell, 'interleaved', policy, key)),
                difference(value(s, cell, 'grouped', 'raw', key), value(s, cell, 'interleaved', 'raw', key)))
                for s in SEEDS]) for key in keys}
    return {'seeds': list(SEEDS), 'endpoint_step': STEPS, 'per_group': groups,
            'change_from_warmup': changes, 'schedule_contrasts': schedule_contrasts,
            'policy_contrasts': policy_contrasts, 'schedule_policy_interactions': interactions,
            'selection': 'fixed_step_2000_no_checkpoint_selection'}


def mean_probe(model, inputs, targets):
    """One FP32 group-mean CE gradient; never accumulates into training .grad."""
    labels = torch.as_tensor(targets, dtype=torch.long, device=inputs.device)
    need(inputs.ndim == 2 and len(inputs) == len(labels) and len(labels) > 0, 'nonempty mean probe')
    logits = model(inputs)
    loss = F.cross_entropy(logits, labels, reduction='mean')
    gradients = torch.autograd.grad(loss, tuple(model.parameters()), create_graph=False,
                                    retain_graph=False, allow_unused=False)
    gradient = torch.cat([g.detach().reshape(-1) for g in gradients]).cpu()
    need(gradient.dtype == torch.float32 and bool(torch.isfinite(gradient).all())
         and bool(torch.isfinite(logits).all()), 'finite FP32 mean probe')
    return logits.detach().cpu(), gradient


def mean_geometry(gradient, tracker, q=None, basis=None):
    """Canonical native action and diagnostic numerical-span retention, distinct."""
    device = tracker.param_list[0].device
    native = tracker._project_gradient(gradient.to(device)).detach().cpu()
    energy = float(gradient.double().square().sum())
    native_energy = float(native.double().square().sum())
    if basis is None:
        q, basis = accepted.numerical_basis(tracker)
    projected_energy = None if q is None else float((q.T @ gradient.double()).square().sum())
    return {'squared_norm': energy, 'native_squared_norm': native_energy,
            'native_retention': native_energy/energy if energy else None,
            'numerical_span_squared_norm': projected_energy,
            'numerical_span_retention': projected_energy/energy if projected_energy is not None and energy else None,
            'zero_denominator': energy == 0, 'identity_fallback': tracker.V is None,
            'numerical_basis_rank': basis['rank']}


def collect_pre_probes(model, optimizer, tracker, definitions, check=lambda: None):
    before = core.snapshot(model, optimizer, tracker)
    groups = {}
    for name, definition in definitions.items():
        if definition is None:
            groups[name] = None
            continue
        check()
        logits, gradient = mean_probe(model, definition['x'], definition['targets'])
        groups[name] = {'training_positions': torch.from_numpy(definition['training_positions'].copy()),
                        'source_ids': torch.from_numpy(definition['source_ids'].copy()),
                        'inputs': definition['x'].detach().cpu().clone(),
                        'targets': torch.from_numpy(definition['targets'].copy()),
                        'true_targets': torch.from_numpy(definition['true_targets'].copy()),
                        'assigned_targets': torch.from_numpy(definition['assigned_targets'].copy()),
                        'pre_logits': logits, 'mean_gradient': gradient}
    need(core.equal_tree(before, core.snapshot(model, optimizer, tracker)),
         'mean probes changed model/grad/Adam/observer/RNG')
    return before, groups


def finish_diagnostic(model, optimizer, tracker, before, groups, definitions, raw, applied,
                      anchor, identity, check=lambda: None):
    after = core.snapshot(model, optimizer, tracker)
    for name, group in groups.items():
        if group is not None:
            check()
            group['post_logits'] = torch.from_numpy(accepted.predict(model, definitions[name]['x']).copy())
    need(core.equal_tree(after, core.snapshot(model, optimizer, tracker)), 'post probes changed live state')
    q, basis = accepted.numerical_basis(tracker)
    vectors, movement = accepted.movement_summary(before, after, q)
    summaries = {}
    for name, group in groups.items():
        if group is None:
            summaries[name] = {'status': 'absent', 'reason': 'no_actually_changed_examples', 'count': 0}
            continue
        pre = accepted.classification_stats(group['pre_logits'], group['targets'])
        post = accepted.classification_stats(group['post_logits'], group['targets'])
        gradient = group['mean_gradient']
        summaries[name] = {'status': 'defined', 'count': len(group['targets']),
                           'true_label_counts': torch.bincount(group['true_targets'], minlength=10).tolist(),
                           'geometry': mean_geometry(gradient, tracker, q, basis),
                           'pre': pre, 'post': post, 'finite_ce_improvement': pre['ce']-post['ce'],
                           'signed_first_order_utilities': {key: -float(gradient.double() @ delta)
                                                            for key, delta in vectors.items()}}
    differences = {'rare_minus_majority': groups['rare8']['mean_gradient']-groups['majority']['mean_gradient'],
                   'wrong_assigned_minus_corrected': None if groups['wrong_assigned'] is None else
                       groups['wrong_assigned']['mean_gradient']-groups['wrong_corrected']['mean_gradient']}
    geometry = {name: None if gradient is None else mean_geometry(gradient, tracker, q, basis)
                for name, gradient in differences.items()}
    need(core.equal_tree(after, core.snapshot(model, optimizer, tracker)), 'mean geometry changed live state')
    payload = {'schema': SCHEMA, **identity, 'update': anchor['update'], 'anchor': anchor,
               'probe_parameter_state': 'pre_update_after_one_training_observer_update',
               'before': before, 'after': after, 'raw_training_gradient': raw.detach().cpu().clone(),
               'applied_training_gradient': applied.detach().cpu().clone(), 'groups': groups,
               'mean_differences': differences, 'parameter_order': [name for name, _ in model.named_parameters()]}
    report = {'schema': SCHEMA, **identity, 'update': anchor['update'], 'anchor': anchor,
              'before_sha256': core.tree_digest(before), 'after_sha256': core.tree_digest(after),
              'basis': basis, 'movement': movement, 'groups': summaries,
              'mean_difference_geometry': geometry, 'difference_arithmetic_dtype': 'float32',
              'probe_gradient_definition': 'autograd.grad of FP32 group mean cross_entropy',
              'probe_side_effect_checks': 'PASS', 'per_example_gradients_or_coherence': False}
    return payload, report


def training_update(model, optimizer, tracker, x, targets, policy, step, rare_count,
                    definitions=None, anchor=None, identity=None, check=lambda: None):
    need(policy in POLICIES and (policy == 'raw' or tracker is not None), 'policy observer contract')
    check()
    optimizer.zero_grad(set_to_none=True)
    loss = F.cross_entropy(model(x), targets)
    need(bool(torch.isfinite(loss)), 'finite training loss')
    loss.backward()
    raw = core.flat_grad(model)
    if tracker is not None:
        accepted.observe(tracker, raw, step)
    native = raw if tracker is None else tracker._project_gradient(raw)
    applied, norm_law = raw, None
    if step > WARMUP and policy == 'native32':
        applied = native
    elif step > WARMUP and policy == 'norm_raw':
        applied, norm_law = accepted.norm_direction(raw, native)
    core.set_grad(model, applied)
    first = {}
    if step == WARMUP+1:
        first['raw_gradient_sha256'] = core.tree_digest(raw.detach().cpu())
        if tracker is not None:
            first['post_observe_tracker_sha256'] = core.tree_digest(core._tracker_state(tracker))
    before = groups = None
    if anchor is not None:
        need(policy == 'native32' and definitions is not None and anchor['update'] == step
             and anchor['rare_count'] == rare_count, 'fixed diagnostic anchor identity')
        before, groups = collect_pre_probes(model, optimizer, tracker, definitions, check)
    optimizer.step()
    need(bool(torch.isfinite(core.flat_params(model)).all()), 'finite updated parameters')
    diagnostic = None
    if anchor is not None:
        diagnostic = finish_diagnostic(model, optimizer, tracker, before, groups, definitions,
                                       raw, applied, anchor, identity, check)
    optimizer.zero_grad(set_to_none=True)
    norm = lambda value: float(value.double().norm())
    row = {'step': step, 'rare_count': int(rare_count), 'batch_size': len(x),
           'training_batch_ce': float(loss.detach()), 'raw_norm': norm(raw), 'native_norm': norm(native),
           'applied_norm': norm(applied), 'norm_law': norm_law,
           'observer_step': None if tracker is None else tracker.step_count,
           'basis_rank': None if tracker is None else (0 if tracker.V is None else tracker.V.shape[1]),
           'native_identity_fallback': tracker is not None and tracker.V is None, **first}
    json.dumps(row, allow_nan=False)
    check()
    return row, diagnostic


def byte_inventory():
    full_state = 37*P*4+32*8+4*4+64*1024
    # Overcount all endpoint/warmup states as fully tracked with populated gradients.
    # Every event receives four means/two differences in this bound, including Clean.
    parts = {'39_warmup_and_final_full_states': 39*full_state,
             '144_diagnostic_pre_post_full_states': 144*full_state,
             '72_events_4_mean_2_difference_2_action_gradients': 72*8*P*4,
             '72_events_probe_inputs_targets_logits': 72*150*(784*4+3*8+2*10*4+2*8),
             '36_train_and_heldout_logit_histories': 36*21*2*5000*10*4,
             'shared_initial_warmup_predictions': 3*2*2*5000*10*4,
             'plans_schedules_scalar_json_and_metadata_allowance': 256*1024**2}
    total = sum(parts.values())
    need(total+RESERVE_BYTES < MAX_BYTES, 'analytic inventory exceeds output limit')
    return {'component_upper_bytes': parts, 'total_upper_bytes': total, 'cap_bytes': MAX_BYTES,
            'cap_headroom_bytes': MAX_BYTES-total, 'failure_metadata_reserve_bytes': RESERVE_BYTES}


def acquire(run):
    images, labels = accepted.read_training()
    results = []
    for seed in SEEDS:
        run.check()
        plan, unused_sham_allocations = accepted.make_plan(labels, seed)
        plan_receipt = run.save(f'plan-s{seed}.npz', plan, 'npz')
        plan_hashes = {key: array_digest(value) for key, value in plan.items()}
        schedules, block_checks = make_schedules(plan, seed)
        need(schedules['grouped_batches'].shape == schedules['interleaved_batches'].shape == (1900, BATCH), 'fixed schedule dimensions')
        schedule_receipt = run.save(f'schedules-s{seed}.npz', schedules, 'npz')
        schedule_hashes = {key: array_digest(value) for key, value in schedules.items()}
        anchors = {name: select_anchors(schedules[name+'_rare_counts']) for name in SCHEDULES}
        need(all(len(events) == 6 and len({row['update'] for row in events}) == 6 for events in anchors.values()), 'six distinct anchors')
        probe_plan = make_probe_plan(plan, seed)
        probe_receipt = run.save(f'probes-s{seed}.npz', probe_plan, 'npz')
        metadata = run.save(f'plan-s{seed}.json', {'schema': SCHEMA, 'seed': seed, 'plan_array_hashes': plan_hashes,
            'schedule_array_hashes': schedule_hashes, 'probe_array_hashes': {k: array_digest(v) for k, v in probe_plan.items()},
            'block_checks': block_checks, 'anchors': anchors, 'unused_original_sham_allocation': unused_sham_allocations,
            'numpy_version': np.__version__, 'rng': 'PCG64 via named SeedSequence streams in protocol',
            'initialization_seed': seed})
        train_x = accepted.normalized_inputs(images, plan['train_ids'], run.device)
        heldout_x = accepted.normalized_inputs(images, plan['heldout_ids'], run.device)
        input_hashes = {'train_inputs_sha256': array_digest(train_x.detach().cpu().numpy()),
                        'heldout_inputs_sha256': array_digest(heldout_x.detach().cpu().numpy()),
                        'input_hash_convention': 'array_digest contiguous CPU float32 [N,784]; NumPy uint8.astype(float32)/float32(255); plan row order'}
        model = core.make_model(seed, run.device)
        need(sum(p.numel() for p in model.parameters()) == P, 'fixed model parameter count')
        initial_sha = core.tree_digest(dict(model.state_dict()))
        optimizer = core.make_optimizer(model)
        tracker = core.make_tracker(model, optimizer)
        initial_train, initial_heldout = accepted.predict(model, train_x), accepted.predict(model, heldout_x)
        accepted.sync(run.device)
        started = time.monotonic()
        warmup_actions = []
        for step, batch in enumerate(plan['warmup_batches'], 1):
            need(not (plan['train_true'][batch] == 8).any(), 'rare example in warmup')
            index = torch.as_tensor(batch, device=run.device)
            target = torch.as_tensor(plan['train_true'][batch], device=run.device)
            row, _ = training_update(model, optimizer, tracker, train_x[index], target, 'raw', step, 0, check=run.check)
            warmup_actions.append(row)
        accepted.sync(run.device)
        warmup_seconds = time.monotonic()-started
        warmup = core.snapshot(model, optimizer, tracker)
        warmup_sha = core.tree_digest(warmup)
        warmup_receipt = run.save(f'warmup-s{seed}.pt', warmup, 'tensor')
        run.save(f'warmup-actions-s{seed}.json', warmup_actions)
        warmup_train, warmup_heldout = accepted.predict(model, train_x), accepted.predict(model, heldout_x)
        common = run.save(f'common-predictions-s{seed}.npz', {'steps': np.array([0, 100], dtype=np.int64),
            'train': np.stack([initial_train, warmup_train]),
            'heldout': np.stack([initial_heldout, warmup_heldout])}, 'npz')
        need(core.equal_tree(warmup, core.snapshot(model, optimizer, tracker)), 'shared predictions changed warmup state')
        del model, optimizer, tracker
        for cell in CELLS:
            assigned = plan['train_true'] if cell == 'clean' else plan['diffuse_targets']
            target_all = torch.as_tensor(assigned, device=run.device)
            definitions = probe_definitions(plan, probe_plan, train_x, cell)
            for schedule in SCHEDULES:
                batches = schedules[schedule+'_batches']
                counts = schedules[schedule+'_rare_counts']
                event_map = {row['update']: row for row in anchors[schedule]}
                binding = {'schema': SCHEMA, 'seed': seed, 'cell': cell, 'schedule': schedule,
                           'plan': plan_receipt, 'plan_metadata': metadata, 'schedules': schedule_receipt, 'probe_plan': probe_receipt,
                           'plan_array_hashes': plan_hashes, 'schedule_array_hashes': schedule_hashes,
                           'warmup': warmup_receipt, 'warmup_state_sha256': warmup_sha,
                           'initial_model_sha256': initial_sha, 'common_predictions': common,
                           'assigned_targets_sha256': array_digest(assigned),
                           'actually_changed_count': int((assigned != plan['train_true']).sum()),
                           'diffuse_selected_count': int(plan['diffuse_selected'].sum()) if cell == 'diffuse' else 0,
                           'warmup_seconds': warmup_seconds, 'anchors': anchors[schedule], **input_hashes}
                run.save(f'binding-s{seed}-{cell}-{schedule}.json', binding)
                offset = (SEEDS.index(seed)+CELLS.index(cell)+SCHEDULES.index(schedule)) % 3
                order = POLICIES[offset:]+POLICIES[:offset]
                first_raw = first_observer = None
                for policy in order:
                    run.check()
                    model, optimizer, tracker = core.restore(warmup, run.device)
                    need(core.tree_digest(core.snapshot(model, optimizer, tracker)) == warmup_sha, 'full warmup fork identity')
                    if policy == 'raw':
                        tracker = None
                    train_history, heldout_history = [initial_train, warmup_train], [initial_heldout, warmup_heldout]
                    curve = [evaluation_row(step, tr, he, plan['train_true'], assigned, plan['heldout_true'])
                             for step, tr, he in zip((0, 100), train_history, heldout_history)]
                    actions, diagnostics = [], []
                    identity = {'seed': seed, 'cell': cell, 'schedule': schedule, 'policy': policy,
                                'warmup_state_sha256': warmup_sha}
                    identifier = f's{seed}-{cell}-{schedule}-{policy}'
                    accepted.sync(run.device)
                    started = time.monotonic()
                    for step, (batch, count) in enumerate(zip(batches, counts), WARMUP+1):
                        index = torch.as_tensor(batch, device=run.device)
                        anchor = event_map.get(step) if policy == 'native32' else None
                        row, diagnostic = training_update(model, optimizer, tracker, train_x[index], target_all[index],
                            policy, step, int(count), definitions if anchor is not None else None, anchor, identity, run.check)
                        actions.append(row)
                        if step == WARMUP+1:
                            need(first_raw is None or first_raw == row['raw_gradient_sha256'], 'first raw gradients differ across policies')
                            first_raw = row['raw_gradient_sha256']
                            if tracker is not None:
                                need(first_observer is None or first_observer == row['post_observe_tracker_sha256'], 'first observed states differ across policies')
                                first_observer = row['post_observe_tracker_sha256']
                        if diagnostic is not None:
                            payload, report = diagnostic
                            report['tensor'] = run.save(f'diagnostic-{identifier}-u{step:04d}.pt', payload, 'tensor')
                            diagnostics.append(run.save(f'diagnostic-{identifier}-u{step:04d}.json', report))
                            del payload, report, diagnostic
                        if step % 100 == 0:
                            tr, he = accepted.predict(model, train_x), accepted.predict(model, heldout_x)
                            train_history.append(tr)
                            heldout_history.append(he)
                            curve.append(evaluation_row(step, tr, he, plan['train_true'], assigned, plan['heldout_true']))
                            print(json.dumps({**identity, 'step': step,
                                'rare_accuracy': curve[-1]['heldout']['rare']['accuracy'],
                                'majority_macro_accuracy': curve[-1]['heldout']['majority_macro']['accuracy'],
                                'batch_elapsed_seconds': time.monotonic()-run.started}), flush=True)
                    accepted.sync(run.device)
                    seconds = time.monotonic()-started
                    final = core.snapshot(model, optimizer, tracker)
                    need([r['step'] for r in curve] == list(EVAL_STEPS), 'complete evaluation roster')
                    need(len(actions) == 1900 and len(diagnostics) == (6 if policy == 'native32' else 0), 'complete branch roster')
                    record = {**binding, 'policy': policy, 'policy_order': list(order), 'curve': curve,
                              'branch_seconds_including_eval_and_diagnostics': seconds,
                              'warmup_plus_branch_seconds': warmup_seconds+seconds,
                              'final_state_sha256': core.tree_digest(final),
                              'final_state': run.save(f'final-{identifier}.pt', final, 'tensor'),
                              'logits': run.save(f'logits-{identifier}.npz', {'steps': np.array(EVAL_STEPS, dtype=np.int64),
                                  'train': np.stack(train_history), 'heldout': np.stack(heldout_history)}, 'npz'),
                              'actions': run.save(f'actions-{identifier}.json', actions), 'diagnostics': diagnostics,
                              'first_action_binding': {key: value for key, value in actions[0].items() if key.endswith('sha256')}}
                    receipt = run.save(f'curve-{identifier}.json', record)
                    results.append({'seed': seed, 'cell': cell, 'schedule': schedule, 'policy': policy,
                                    'curve': receipt, 'endpoint': curve[-1], 'warmup': curve[1],
                                    'warmup_plus_branch_seconds': warmup_seconds+seconds})
                    del model, optimizer, tracker, final, train_history, heldout_history, actions
        del warmup, train_x, heldout_x, definitions
    need(len(results) == 36, 'complete fixed scientific roster')
    return run.save('results.json', {'schema': SCHEMA, 'rows': results, 'summary': summarize_results(results)})


def configure():
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        need(os.environ.get(key) == '1', 'set '+key+'=1 before Python')
    need(os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8', 'deterministic CUDA workspace required')
    group = next(line.split(':', 2)[2] for line in Path('/proc/self/cgroup').read_text().splitlines() if line.startswith('0::'))
    need(Path(group).name == UNIT, 'unexpected experiment unit')
    directory = Path('/sys/fs/cgroup')/group.lstrip('/')
    effective = {key: (directory/key).read_text().strip() for key in ('memory.max', 'memory.swap.max', 'cpu.max')}
    properties = subprocess.check_output(['systemctl', '--user', 'show', UNIT, '--property=Type',
        '--property=RuntimeMaxUSec', '--property=Restart', '--property=KillMode'], text=True)
    service = dict(line.split('=', 1) for line in properties.splitlines())
    accepted.validate_bounds(effective, service)
    need((MAX_BYTES, RESERVE_BYTES, DEADLINE_SECONDS) ==
         (accepted.MAX_BYTES, accepted.RESERVE_BYTES, accepted.DEADLINE_SECONDS), 'inherited storage/deadline mismatch')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    need(torch.__version__ == '2.11.0+cu128', 'unexpected Torch version')
    need(torch.cuda.is_available() and torch.cuda.get_device_name() == 'NVIDIA GeForce RTX 3090', 'local RTX3090 required')
    free, total = torch.cuda.mem_get_info()
    need(free >= 8*1024**3, 'at least 8 GiB free GPU required')
    torch.cuda.set_per_process_memory_fraction(8*1024**3/total)
    return {'cgroup': group, 'effective': effective, 'service': service}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    need(args.execute, 'No scientific work without explicit --execute')
    parent = args.output_dir.parent.resolve(strict=True)
    need(parent.is_relative_to(Path('/tmp/spectral-experiment-artifacts')) and not list(parent.iterdir()), 'unused large-volume parent required')
    need(args.output_dir == parent/'acquisition-001' and not args.output_dir.exists()
         and not args.output_dir.is_symlink(), 'new exclusive acquisition-001 required')
    mount = subprocess.check_output(['findmnt', '-n', '-o', 'TARGET,SOURCE', '--target', str(parent)], text=True).split()
    need(mount == ['/private-artifacts/storage', '/dev/RECONFIGURE_FOR_LOCAL_STORAGE'], 'wrong output mount')
    admission = configure()
    inventory, pins = byte_inventory(), source_pins()
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    for name, expected in pins.items():
        committed = subprocess.check_output(['git', 'show', commit+':'+name], cwd=ROOT)
        need(hashlib.sha256(committed).hexdigest() == expected, 'source not committed: '+name)
    data_pins = {name: digest(accepted.DATA/name) for name in accepted.DATA_SHA256}
    need(data_pins == accepted.DATA_SHA256, 'accepted training IDX hashes differ')
    need(shutil.disk_usage(parent).free > 1024**3+inventory['total_upper_bytes'], 'insufficient output/free-disk reserve')
    args.output_dir.mkdir(exist_ok=False)
    run = accepted.Run(args.output_dir)  # Immutable inherited capped writer/deadline checks.
    try:
        manifest = {'schema': SCHEMA, 'source_pins': pins, 'data_pins': data_pins,
                    'data_directory': str(accepted.DATA), 'git_commit': commit, 'resource_admission': admission,
                    'pid': os.getpid(), 'invocation_id': os.environ.get('INVOCATION_ID'), 'started_unix': time.time(),
                    'seeds': list(SEEDS), 'cells': list(CELLS), 'schedules': list(SCHEDULES), 'policies': list(POLICIES),
                    'warmup': WARMUP, 'steps': STEPS, 'batch_size': BATCH, 'block_size': BLOCK_SIZE,
                    'blocks': BLOCKS, 'anchor_blocks': list(ANCHOR_BLOCKS), 'native_events_per_trajectory': 6,
                    'eval_steps': list(EVAL_STEPS), 'rank': RANK, 'byte_inventory': inventory,
                    'torch': torch.__version__, 'numpy': np.__version__, 'python': sys.version,
                    'rng': 'PCG64 via named SeedSequence streams in protocol',
                    'cooperative_seconds': DEADLINE_SECONDS, 'cloud_spend_usd': 0,
                    'rare_class': 8, 'majority_train_per_class': 550, 'rare_train_count': 50,
                    'heldout_per_class': 500, 'observer': 'unchanged I9 current stable hard32',
                    'probe_definition': '54 stratified-majority true,32 rare true,up to32 randomized changed assigned/corrected means',
                    'timing_scope': 'branch includes evaluation and native-only diagnostics/serialization; not speed comparison'}
        run.save('manifest.json', manifest)
        print(json.dumps({'resource_guard': 'PASS', 'unit': UNIT, 'output': str(args.output_dir),
                          'source_pins': pins, 'data_pins': data_pins, 'inventory': inventory,
                          'invocation_id': os.environ.get('INVOCATION_ID'), **admission}), flush=True)
        results = acquire(run)
        need(source_pins() == pins and {name: digest(accepted.DATA/name) for name in accepted.DATA_SHA256} == data_pins,
             'source/data changed during acquisition')
        run.save('complete.json', {'schema': SCHEMA, 'status': 'complete', 'results': results,
            'source_pins': pins, 'data_pins': data_pins, 'finished_unix': time.time(),
            'wall_seconds': time.monotonic()-run.started, 'process_high_water_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'gpu_max_allocated_bytes': torch.cuda.max_memory_allocated(), 'receipts': list(run.receipts),
            'artifact_bytes_before_completion': run.used, 'completed_trajectories': 36, 'completed_diagnostics': 72})
    except BaseException as exc:
        payload = json.dumps({'schema': SCHEMA, 'status': 'failed', 'type': type(exc).__name__, 'message': str(exc),
            'finished_unix': time.time(), 'wall_seconds': time.monotonic()-run.started,
            'completed_receipts': run.receipts}, indent=2, allow_nan=False).encode()
        need(len(payload) <= RESERVE_BYTES, 'failure metadata reserve exceeded')
        with (args.output_dir/'failed.json').open('xb') as stream:
            stream.write(payload)
        raise


if __name__ == '__main__':
    main()
