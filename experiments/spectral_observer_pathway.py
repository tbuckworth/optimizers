#!/usr/bin/env python3
"""One-shot matched-parent observer diagnostic. Import never reads scientific data."""
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import spectral_batch_composition as batch

accepted, core = batch.accepted, batch.core
need, digest, array_digest = accepted.need, accepted.digest, accepted.array_digest
SCHEMA = 'spectral_observer_pathway_v1'
DOCS = ROOT/'output/2026-09-10-spectral-observer-pathway'
DESIGN = ROOT/'output/2026-09-10-spectral-batch-composition'
PARENT = Path('/tmp/spectral-experiment-artifacts/spectral-batch-composition-20260910.H2Ow40/acquisition-001')
AUDIT = PARENT.parent/'audit-001/result.json'
COMPLETE_SHA = '39b4ef091e691780b46c1d7f5dcb12e108dbfd5b3b85c6abba612a93a393b60a'
AUDIT_SHA = '3640269898d6c4115adbac925586fb54a6f1d47b0000b7e511a2fa1f1fcdeb6b'
BATCH_SHA = '022d6cf0a469119ce044623345720da8eb5186812576e2fac0ee71b8e7450550'
SEEDS, CELLS, SCHEDULES = batch.SEEDS, batch.CELLS, batch.SCHEDULES
ACTIONS = ('native_interleaved', 'native_grouped', 'raw', 'zero')
P = 50890
UNIT = 'spectral-observer-pathway-001.service'
MAX_BYTES, FOOTER_RESERVE, DISK_RESERVE = 1024**3, 1024**2, 1024**3
COOPERATIVE_SECONDS = 480


def source_pins():
    paths = [ROOT/'spectral_filter.py', Path(core.__file__), Path(accepted.__file__), Path(batch.__file__),
             Path(__file__), ROOT/'tests/test_spectral_observer_pathway.py', DOCS/'protocol.md',
             DESIGN/'next-discriminator.md', DESIGN/'next-design-review.md']
    pins = {str(p.relative_to(ROOT)): digest(p) for p in paths}
    need(pins['spectral_filter.py'] == core.FILTER_SHA256, 'canonical filter changed')
    need(pins[str(Path(core.__file__).relative_to(ROOT))] == accepted.CORE_SHA, 'core changed')
    need(pins['experiments/spectral_selectivity_boundary.py'] == batch.ACCEPTED_SHA, 'selectivity helper changed')
    need(pins['experiments/spectral_batch_composition.py'] == BATCH_SHA, 'batch helper changed')
    return pins


def adam_clock(state):
    clocks = [int(float(row['step'])) for row in state['optimizer']['state'].values()]
    need(clocks and len(set(clocks)) == 1, 'uniform initialized Adam clock required')
    return clocks[0]


def envelope(model, optimizer, observer=None):
    model_adam = core.snapshot(model, optimizer, None)
    return {'schema': SCHEMA, 'model_adam': model_adam,
            'observer': core._tracker_state(observer),
            'clocks': {'adam': adam_clock(model_adam),
                       'observer': None if observer is None else observer.step_count}}


def tensor_leaves(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from tensor_leaves(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from tensor_leaves(item)


def live_tensors(model, optimizer, observer=None):
    return {'parameters': [p.detach() for p in model.parameters()],
            'gradients': [p.grad for p in model.parameters()], 'optimizer': optimizer.state,
            'observer': None if observer is None else {k: v for k, v in vars(observer).items()
                if k not in ('model', 'base_optimizer', 'param_list')}}


def require_nonaliasing(*trees):
    previous = set()
    for tree in trees:
        storages = {(str(t.device), t.untyped_storage().data_ptr()) for t in tensor_leaves(tree) if t.numel()}
        need(not (previous & storages), 'writable tensor storage aliases a sibling/reference')
        previous.update(storages)


def copied_parent(parent, device):
    reference = core.tree_digest(parent)
    model, optimizer, observer = core.restore(parent, device)
    need(core.tree_digest(core.snapshot(model, optimizer, observer)) == reference, 'parent restore identity')
    require_nonaliasing(parent, live_tensors(model, optimizer, observer))
    need(core.tree_digest(parent) == reference and observer.step_count == 100
         and adam_clock(core.snapshot(model, optimizer, None)) == 100, 'parent clocks/immutability')
    return model, optimizer, observer


def protected_gradient(model, optimizer, observer, inputs, targets, parent, check=lambda: None):
    check()
    before = envelope(model, optimizer, observer)
    parent_sha = core.tree_digest(parent)
    require_nonaliasing(parent, live_tensors(model, optimizer, observer))
    _, gradient = batch.mean_probe(model, inputs, targets)
    after = envelope(model, optimizer, observer)
    need(core.equal_tree(before, after) and core.tree_digest(parent) == parent_sha,
         'autograd.grad changed parent/model/Adam/observer/grad/modes/RNG')
    check()
    return gradient, {'before_sha256': core.tree_digest(before), 'after_sha256': core.tree_digest(after),
                      'parent_sha256': parent_sha, 'nonaliasing': 'PASS', 'preservation': 'PASS'}


def common_input(left, right):
    need(left.dtype == right.dtype == torch.float32 and left.shape == right.shape
         and left.ndim == 2 and left.shape[0] == 50 and bool(torch.isfinite(left).all())
         and bool(torch.isfinite(right).all()), 'two finite 50-gradient FP32 streams required')
    # Canonical CPU NumPy reduction/cast order binds exact saved g_star bytes.
    arrays = [x.detach().cpu().numpy().astype(np.float64) for x in (left, right)]
    mean_arrays = [x.mean(axis=0, dtype=np.float64) for x in arrays]
    means = tuple(torch.from_numpy(x.copy()) for x in mean_arrays)
    discrepancy = float((means[0]-means[1]).norm())
    scale = float((left.double().norm(dim=1).sum()+right.double().norm(dim=1).sum())/100)
    bound = 128*torch.finfo(torch.float32).eps*scale+1e-10
    star = torch.from_numpy(((mean_arrays[0]+mean_arrays[1])/np.float64(2)).astype(np.float32).copy())
    need(bool(torch.isfinite(star).all()), 'finite common action input')
    report = {'mean_difference_norm': discrepancy, 'average_batch_gradient_norm': scale,
              'bound': bound, 'mean_norms': [float(x.norm()) for x in means],
              'passed': discrepancy <= bound, 'multiplier': 128, 'absolute_floor': 1e-10,
              'g_star_norm': float(star.double().norm()), 'g_star_sha256': core.tree_digest(star)}
    return star, means, report


def vector_pair(left, right):
    x, y = left.double(), right.double()
    nx, ny = float(x.norm()), float(y.norm())
    return {'left_norm': nx, 'right_norm': ny, 'difference_norm': float((x-y).norm()),
            'cosine': float(x @ y)/(nx*ny) if nx and ny else None,
            'zero_norm': nx == 0 or ny == 0}


def observer_history(parent, gradients, star, oracles, device, check=lambda: None):
    inputs_sha = core.tree_digest({'gradients': gradients, 'star': star, 'oracles': oracles})
    model, optimizer, observer = copied_parent(parent, device)
    fixed = core.snapshot(model, optimizer, None)
    parent_sha = core.tree_digest(parent)
    for step, gradient in enumerate(gradients, 101):
        check()
        accepted.observe(observer, gradient.to(device), step)
        need(core.equal_tree(fixed, core.snapshot(model, optimizer, None))
             and core.tree_digest(parent) == parent_sha, 'observer update changed fixed model/Adam/RNG/parent')
    need(observer.step_count == 150, '50 observer-only updates required')
    at150 = envelope(model, optimizer, observer)
    pre_action = observer._project_gradient(star.to(device)).detach().cpu().clone()
    q, basis150 = accepted.numerical_basis(observer)
    pre_geometry = {name: batch.mean_geometry(g, observer, q, basis150)
                    for name, g in {'block': star, **oracles}.items()}
    # Fresh restored sibling, then copied observer attributes; never mutate O150.
    model2, optimizer2, observer2 = copied_parent(parent, device)
    for key, value in at150['observer'].items():
        setattr(observer2, key, core._move_tracker_value(key, value, torch.device(device)))
    require_nonaliasing(parent, at150, live_tensors(model, optimizer, observer), live_tensors(model2, optimizer2, observer2))
    accepted.observe(observer2, star.to(device), 151)
    need(core.equal_tree(fixed, core.snapshot(model2, optimizer2, None)), 'self inclusion changed model/Adam/RNG')
    at151 = envelope(model2, optimizer2, observer2)
    post_action = observer2._project_gradient(star.to(device)).detach().cpu().clone()
    q, basis151 = accepted.numerical_basis(observer2)
    post_geometry = {name: batch.mean_geometry(g, observer2, q, basis151)
                     for name, g in {'block': star, **oracles}.items()}
    need(core.equal_tree(at150, envelope(model, optimizer, observer))
         and core.equal_tree(at151, envelope(model2, optimizer2, observer2))
         and core.tree_digest(parent) == parent_sha
         and core.tree_digest({'gradients': gradients, 'star': star, 'oracles': oracles}) == inputs_sha,
         'action/geometry changed state or input gradients')
    data = {'at150': at150, 'at151': at151, 'pre_action': pre_action, 'post_action': post_action}
    report = {'at150_sha256': core.tree_digest(at150), 'at151_sha256': core.tree_digest(at151),
              'pre_inclusion': {'basis': basis150, 'geometry': pre_geometry},
              'post_inclusion': {'basis': basis151, 'geometry': post_geometry},
              'self_inclusion_action_comparison': vector_pair(pre_action, post_action),
              'observer_only_updates': 50, 'self_inclusion_updates': 1, 'model_adam_preservation': 'PASS',
              'nonaliasing': 'PASS'}
    return data, report


def readout(parent, gradient, observer_state, oracles, train_x, heldout_x, device, check=lambda: None):
    model, optimizer, observer = copied_parent(parent, device)
    if observer_state is None:
        observer = None
    else:
        for key, value in observer_state.items():
            setattr(observer, key, core._move_tracker_value(key, value, torch.device(device)))
        need(observer.step_count == 151, 'native readout observer must remain151')
    inherited = envelope(model, optimizer, observer)
    parent_sha = core.tree_digest(parent)
    require_nonaliasing(parent, observer_state, live_tensors(model, optimizer, observer))
    need(gradient.dtype == torch.float32 and gradient.numel() == core.flat_params(model).numel()
         and bool(torch.isfinite(gradient).all()), 'finite explicit delivered action')
    core.set_grad(model, gradient.to(device))
    before = envelope(model, optimizer, observer)
    before_without_grad = core._cpu_clone(before)
    before_without_grad['model_adam']['gradients'] = inherited['model_adam']['gradients']
    need(core.equal_tree(before_without_grad, inherited), 'action assignment changed something besides .grad')
    check()
    optimizer.step()  # No backward, tracker observation, normalization or second optimizer step.
    after = envelope(model, optimizer, observer)
    need(before['clocks']['adam'] == 100 and after['clocks']['adam'] == 101
         and before['clocks']['observer'] == after['clocks']['observer'], 'separate readout clocks')
    need(core.equal_tree(before['observer'], after['observer']), 'readout changed observer')
    predictions = {'train': accepted.predict(model, train_x), 'heldout': accepted.predict(model, heldout_x)}
    need(core.equal_tree(after, envelope(model, optimizer, observer))
         and core.tree_digest(parent) == parent_sha, 'predictions changed readout/reference state')
    q, basis = (None, None) if observer is None else accepted.numerical_basis(observer)
    movement, movement_report = accepted.movement_summary(before['model_adam'], after['model_adam'], q)
    utility = {name: {part: -float(g.double() @ delta) for part, delta in movement.items()}
               for name, g in oracles.items()}
    need(core.equal_tree(after, envelope(model, optimizer, observer)), 'movement diagnostics changed state')
    data = {'inherited': inherited, 'before': before, 'after': after,
            'delivered_gradient': gradient.detach().cpu().clone(),
            'parameter_order': [name for name, _ in model.named_parameters()]}
    report = {'inherited_sha256': core.tree_digest(inherited), 'before_sha256': core.tree_digest(before),
              'after_sha256': core.tree_digest(after), 'basis': basis, 'movement': movement_report,
              'oracle_signed_utilities': utility, 'parent_sha256': parent_sha,
              'gradient_semantics': 'explicit assigned gradient retained in before and after; inherited .grad saved separately',
              'physical_optimizer_steps': 1, 'new_observer_updates': 0, 'preservation': 'PASS', 'nonaliasing': 'PASS'}
    return data, predictions, report


def action_improvement(before, after):
    left, right = batch.endpoint_metrics(before), batch.endpoint_metrics(after)
    result = {}
    for key in left:
        need((left[key] is None) == (right[key] is None), 'asymmetric undefined metric')
        result[key] = None if left[key] is None else left[key]-right[key] if key.endswith('_ce') else right[key]-left[key]
    return result


def summarize_cases(cases):
    roster = {(row['seed'], row['cell']): row for row in cases}
    need(len(cases) == len(roster) == 6 and set(roster) == {(s, c) for s in SEEDS for c in CELLS}, 'six fixed cases')
    keys = list(cases[0]['actions']['raw']['improvements'])
    absolute, contrasts = {}, {}
    for cell in CELLS:
        for action in ACTIONS:
            absolute[cell+'/'+action] = {k: accepted.sample_summary([
                roster[s, cell]['actions'][action]['improvements'][k] for s in SEEDS]) for k in keys}
        for left, right in [('native_grouped', 'native_interleaved'),
                             ('native_grouped', 'raw'), ('native_interleaved', 'raw'),
                             ('native_grouped', 'zero'), ('native_interleaved', 'zero'), ('raw', 'zero')]:
            values = {}
            for key in keys:
                rows = [roster[s, cell]['actions'] for s in SEEDS]
                values[key] = accepted.sample_summary([None if row[left]['improvements'][key] is None else
                    row[left]['improvements'][key]-row[right]['improvements'][key] for row in rows])
            contrasts[cell+'/'+left+'_minus_'+right] = values
    return {'seeds': list(SEEDS), 'absolute_improvements': absolute, 'improvement_contrasts': contrasts,
            'orientation': 'CE:before-minus-after; accuracy:after-minus-before; positive is favorable for both',
            'primary_contrast': 'native_grouped_minus_native_interleaved',
            'primary_metrics': ['rare_ce', 'majority_macro_ce'], 'adam_before': 100, 'adam_after': 101,
            'native_observer_delivery': 151, 'physical_readouts': 21, 'logical_case_readouts': 24}


class Inputs:
    """Admission of only accepted parent inputs; never invokes an old audit."""
    def __init__(self):
        need(digest(AUDIT) == AUDIT_SHA and digest(PARENT/'complete.json') == COMPLETE_SHA, 'accepted parent/audit pins')
        audit = json.loads(AUDIT.read_text())
        need(audit['status'] == 'PASS' and not audit['errors']
             and (audit['evaluations_recomputed'], audit['trajectories_checked'], audit['native_events_checked']) == (756, 36, 72)
             and audit['expected_completion_sha256'] == COMPLETE_SHA
             and Path(audit['acquisition_dir']) == PARENT, 'accepted audit contract')
        complete = json.loads((PARENT/'complete.json').read_text())
        need(complete['status'] == 'complete' and complete['completed_trajectories'] == 36
             and complete['completed_diagnostics'] == 72, 'accepted completion roster')
        self.index = {item['path']: item for item in audit['input_receipts']}
        need(len(self.index) == len(audit['input_receipts']) and complete['receipts'] == audit['input_receipts'], 'parent receipt identity')
        self.used = {}
        self.manifest = self.json('manifest.json')
        need(self.manifest['source_pins'] == complete['source_pins'] == audit['expected_sources'], 'parent source binding')
        need(self.manifest['data_pins'] == accepted.DATA_SHA256 and self.manifest['torch'] == torch.__version__
             and self.manifest['numpy'] == np.__version__ and self.manifest['python'] == sys.version,
             'parent input/environment reuse binding')
        for name, expected in self.manifest['source_pins'].items():
            need(digest(ROOT/name) == expected, 'accepted source changed: '+name)

    def file(self, name):
        need(Path(name).name == name and name in self.index, 'unbound parent input')
        path, item = PARENT/name, self.index[name]
        need(not path.is_symlink() and path.stat().st_size == item['size_bytes'] and digest(path) == item['sha256'], 'parent input receipt changed')
        self.used[name] = dict(item)
        return path

    def json(self, name):
        return json.loads(self.file(name).read_text())

    def tensor(self, name):
        return torch.load(self.file(name), map_location='cpu', weights_only=True)

    def npz(self, name):
        with np.load(self.file(name), allow_pickle=False) as file:
            return {key: file[key].copy() for key in file.files}

    def bound(self, item, name):
        need(item == self.index[name], 'parent cross-receipt binding')

    def recheck(self):
        need(digest(AUDIT) == AUDIT_SHA and digest(PARENT/'complete.json') == COMPLETE_SHA, 'parent completion/audit changed')
        for name in list(self.used):
            self.file(name)


class Run:
    def __init__(self, path, device='cuda', started=None):
        self.path, self.device = Path(path), device
        self.started, self.last_check = time.monotonic() if started is None else started, 0.
        self.used, self.receipts = 0, []

    def check(self):
        now = time.monotonic()
        need(now-self.started < COOPERATIVE_SECONDS, 'eight-minute cooperative deadline exceeded')
        if now-self.last_check > 1:
            need(shutil.disk_usage(self.path).free > DISK_RESERVE, '1 GiB free-disk reserve exhausted')
            if self.device == 'cuda':
                need(torch.cuda.max_memory_allocated() <= 8*1024**3, '8 GiB GPU cap exceeded')
            self.last_check = now

    def save(self, name, value, kind='json'):
        self.check()
        need(Path(name).name == name, 'direct-child output required')
        path = self.path/name
        with path.open('xb') as file:
            capped = accepted.CappedWriter(file, MAX_BYTES-self.used-FOOTER_RESERVE, self.check)
            if kind == 'json':
                capped.write((json.dumps(value, indent=2, allow_nan=False)+'\n').encode())
            elif kind == 'tensor':
                torch.save(value, capped)
            elif kind == 'npz':
                np.savez(capped, **value)
            else:
                raise ValueError('unknown artifact kind')
        item = {'path': name, 'sha256': digest(path), 'size_bytes': path.stat().st_size}
        self.used += item['size_bytes']
        self.receipts.append(item)
        self.check()
        return item


def byte_inventory():
    model_adam = 4*P*4+64*1024
    observer_envelope = 37*P*4+64*1024
    parts = {'600_stream_gradients': 600*P*4,
             '24_observer150_151_envelopes': 24*observer_envelope,
             '36_native_readout_inherited_before_after_envelopes': 36*observer_envelope,
             '27_raw_zero_readout_envelopes': 27*model_adam,
             '6_case_stars_and_two_fp64_means': 6*P*(4+2*8),
             '24_pre_post_native_and21_delivered_actions': 45*P*4,
             '6_oracle_gradients_and_probe_inputs': 6*P*4+3*86*(784*4+3*8),
             '21_new_train_heldout_logits': 21*2*5000*10*4,
             'membership_receipts_scalar_metadata_allowance': 128*1024**2}
    total = sum(parts.values())
    need(total+FOOTER_RESERVE < MAX_BYTES, 'prospective output inventory exceeds1GiB')
    return {'component_upper_bytes': parts, 'total_upper_bytes': total, 'cap_bytes': MAX_BYTES,
            'cap_headroom_bytes': MAX_BYTES-total, 'failure_footer_reserve_bytes': FOOTER_RESERVE}


def save_readout(run, identity, parent, gradient, observer, oracles, train_x, heldout_x):
    data, predictions, report = readout(parent, gradient, observer, oracles, train_x, heldout_x, run.device, run.check)
    record = {'schema': SCHEMA, 'physical_id': identity,
              'state': run.save('readout-'+identity+'.pt', {'schema': SCHEMA, 'physical_id': identity, **data}, 'tensor'),
              'logits': run.save('readout-'+identity+'.npz', predictions, 'npz'), **report}
    record['report'] = run.save('readout-'+identity+'.json', record)
    return record, predictions


def acquire(run, inputs):
    images, all_labels = accepted.read_training()
    cases = []
    physical, stream_count, oracle_count = 0, 0, 0
    for seed in SEEDS:
        parent = inputs.tensor(f'warmup-s{seed}.pt')
        parent_sha = core.tree_digest(parent)
        need(parent['model_spec'] == {'input_dim': 784, 'width': 64, 'classes': 10}
             and parent['tracker']['step_count'] == 100 and adam_clock(parent) == 100, 'fixed accepted parent')
        plan = inputs.npz(f'plan-s{seed}.npz')
        schedules = inputs.npz(f'schedules-s{seed}.npz')
        probes = inputs.npz(f'probes-s{seed}.npz')
        common = inputs.npz(f'common-predictions-s{seed}.npz')
        need(np.array_equal(common['steps'], [0, 100]) and common['train'].shape == common['heldout'].shape == (2, 5000, 10)
             and common['train'].dtype == common['heldout'].dtype == np.float32, 'accepted before-logit identity')
        need(np.array_equal(plan['train_true'], all_labels[plan['train_ids']])
             and np.array_equal(plan['heldout_true'], all_labels[plan['heldout_ids']]), 'input label/source identity')
        train_x = accepted.normalized_inputs(images, plan['train_ids'], run.device)
        heldout_x = accepted.normalized_inputs(images, plan['heldout_ids'], run.device)
        bindings = {}
        for cell in CELLS:
            binding = inputs.json(f'binding-s{seed}-{cell}-interleaved.json')
            for key, name in [('warmup', f'warmup-s{seed}.pt'), ('plan', f'plan-s{seed}.npz'),
                              ('schedules', f'schedules-s{seed}.npz'), ('probe_plan', f'probes-s{seed}.npz'),
                              ('common_predictions', f'common-predictions-s{seed}.npz')]:
                inputs.bound(binding[key], name)
            need(binding['seed'] == seed and binding['cell'] == cell
                 and binding['warmup_state_sha256'] == parent_sha
                 and binding['train_inputs_sha256'] == array_digest(train_x.cpu().numpy())
                 and binding['heldout_inputs_sha256'] == array_digest(heldout_x.cpu().numpy()), 'before predictions model/input binding')
            assigned = plan['train_true'] if cell == 'clean' else plan['diffuse_targets']
            need(binding['assigned_targets_sha256'] == array_digest(assigned), 'assigned-label binding')
            bindings[cell] = binding
        model, optimizer, observer = copied_parent(parent, run.device)
        oracle_data, oracles = {}, {}
        for name in ('majority', 'rare8'):
            positions = probes[name]
            need(len(positions) == (54 if name == 'majority' else 32), 'fixed oracle probe count')
            x = train_x[torch.as_tensor(positions, device=run.device)]
            target = plan['train_true'][positions]
            gradient, preservation = protected_gradient(model, optimizer, observer, x, target, parent, run.check)
            oracle_count += 1
            oracle_data[name] = {'training_positions': torch.from_numpy(positions),
                'source_ids': torch.from_numpy(plan['train_ids'][positions]), 'inputs': x.detach().cpu(),
                'targets': torch.from_numpy(target), 'mean_gradient': gradient, 'preservation': preservation}
            oracles[name] = gradient
        oracle_receipt = run.save(f'oracles-s{seed}.pt', {'schema': SCHEMA, 'seed': seed,
            'parent_sha256': parent_sha, 'groups': oracle_data, 'use': 'oracle diagnostic only; never delivered or observed'}, 'tensor')
        del model, optimizer, observer
        zero_record, zero_predictions = save_readout(run, f's{seed}-zero', parent, torch.zeros(P), None, oracles, train_x, heldout_x)
        physical += 1
        for cell in CELLS:
            assigned = plan['train_true'] if cell == 'clean' else plan['diffuse_targets']
            before_metrics = batch.evaluation_row(100, common['train'][1], common['heldout'][1],
                                                 plan['train_true'], assigned, plan['heldout_true'])
            streams, stream_receipts = {}, {}
            for schedule in SCHEDULES:
                model, optimizer, observer = copied_parent(parent, run.device)
                memberships = schedules[schedule+'_batches'][:50].copy()
                need(memberships.shape == (50, 64) and np.array_equal(np.sort(memberships.ravel()),
                     np.sort(plan['continuation_batches'][:50].ravel())), 'fixed first block multiset')
                gradients, preservation = [], []
                for t, positions in enumerate(memberships):
                    x = train_x[torch.as_tensor(positions, device=run.device)]
                    gradient, receipt = protected_gradient(model, optimizer, observer, x, assigned[positions], parent, run.check)
                    gradients.append(gradient)
                    preservation.append({'position': t, **receipt})
                    stream_count += 1
                streams[schedule] = torch.stack(gradients)
                stream_receipts[schedule] = run.save(f'stream-s{seed}-{cell}-{schedule}.pt', {
                    'schema': SCHEMA, 'seed': seed, 'cell': cell, 'schedule': schedule,
                    'parent_sha256': parent_sha, 'memberships': torch.from_numpy(memberships),
                    'gradients': streams[schedule], 'preservation': preservation,
                    'clock': {'adam': 100, 'observer': 100}, 'observer_updates_during_measurement': 0}, 'tensor')
                del model, optimizer, observer
            star, means, admission = common_input(streams['interleaved'], streams['grouped'])
            common_receipt = run.save(f'common-action-s{seed}-{cell}.pt', {'schema': SCHEMA, 'seed': seed, 'cell': cell,
                'g_star': star, 'mean_interleaved': means[0], 'mean_grouped': means[1]}, 'tensor')
            run.save(f'mean-admission-s{seed}-{cell}.json', {'schema': SCHEMA, 'seed': seed, 'cell': cell,
                'streams': stream_receipts, 'common_action': common_receipt, **admission})
            need(admission['passed'], 'fixed mean-identity tolerance failed; saved evidence retained')
            histories, history_receipts, records, predictions = {}, {}, {'zero': zero_record}, {'zero': zero_predictions}
            for schedule in SCHEDULES:
                data, report = observer_history(parent, streams[schedule], star, oracles, run.device, run.check)
                histories[schedule] = data
                history_receipts[schedule] = run.save(f'observer-s{seed}-{cell}-{schedule}.pt', {
                    'schema': SCHEMA, 'seed': seed, 'cell': cell, 'schedule': schedule,
                    'parent_sha256': parent_sha, 'g_star_sha256': core.tree_digest(star), **data}, 'tensor')
                run.save(f'observer-s{seed}-{cell}-{schedule}.json', {'schema': SCHEMA, 'seed': seed, 'cell': cell,
                    'schedule': schedule, 'state': history_receipts[schedule], **report})
                action = 'native_'+schedule
                records[action], predictions[action] = save_readout(run, f's{seed}-{cell}-{action}', parent,
                    data['post_action'], data['at151']['observer'], oracles, train_x, heldout_x)
                physical += 1
            require_nonaliasing(histories['interleaved'], histories['grouped'], parent)
            records['raw'], predictions['raw'] = save_readout(run, f's{seed}-{cell}-raw', parent, star, None,
                                                              oracles, train_x, heldout_x)
            physical += 1
            action_rows = {}
            for action in ACTIONS:
                pred = predictions[action]
                after_metrics = batch.evaluation_row(101, pred['train'], pred['heldout'], plan['train_true'], assigned, plan['heldout_true'])
                action_rows[action] = {'readout': records[action], 'after': after_metrics,
                                       'improvements': action_improvement(before_metrics, after_metrics),
                                       'shared_across_label_cells': action == 'zero'}
            case = {'schema': SCHEMA, 'seed': seed, 'cell': cell, 'parent_sha256': parent_sha,
                    'parent_inputs': {key: value for key, value in inputs.used.items() if str(seed) in key},
                    'oracles': oracle_receipt, 'before_predictions': bindings[cell]['common_predictions'],
                    'streams': stream_receipts, 'common_action': common_receipt, 'mean_admission': admission,
                    'histories': history_receipts, 'history_action_comparisons': {
                        phase: vector_pair(histories['interleaved'][phase+'_action'], histories['grouped'][phase+'_action'])
                        for phase in ('pre', 'post')}, 'before': before_metrics, 'actions': action_rows}
            run.save(f'case-s{seed}-{cell}.json', case)
            cases.append(case)
            need(core.tree_digest(parent) == parent_sha, 'accepted parent mutated')
            print(json.dumps({'seed': seed, 'cell': cell, 'status': 'case_complete',
                              'physical_readouts': physical, 'stream_gradients': stream_count}), flush=True)
        del parent, train_x, heldout_x, oracle_data, oracles, histories, streams
    need((physical, stream_count, oracle_count) == (21, 600, 6), 'fixed physical scientific roster')
    return run.save('results.json', {'schema': SCHEMA, 'cases': cases, 'summary': summarize_cases(cases)})


def validate_bounds(effective, service):
    quota, period = effective['cpu.max'].split()
    need(effective['memory.max'] == str(16*1024**3) and effective['memory.swap.max'] == '0'
         and quota != 'max' and int(quota) == int(period) > 0, 'fixed cgroup bounds required')
    need(service == {'Type': 'exec', 'RuntimeMaxUSec': '10min', 'Restart': 'no', 'KillMode': 'control-group'}, 'fixed service bounds required')


def configure():
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        need(os.environ.get(key) == '1', 'set '+key+'=1 before Python')
    need(os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8', 'deterministic CUDA workspace')
    group = next(line.split(':', 2)[2] for line in Path('/proc/self/cgroup').read_text().splitlines() if line.startswith('0::'))
    need(Path(group).name == UNIT, 'unexpected acquisition service')
    path = Path('/sys/fs/cgroup')/group.lstrip('/')
    effective = {key: (path/key).read_text().strip() for key in ('memory.max', 'memory.swap.max', 'cpu.max')}
    properties = subprocess.check_output(['systemctl', '--user', 'show', UNIT, '--property=Type',
        '--property=RuntimeMaxUSec', '--property=Restart', '--property=KillMode'], text=True)
    service = dict(line.split('=', 1) for line in properties.splitlines())
    validate_bounds(effective, service)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    need(torch.__version__ == '2.11.0+cu128' and np.__version__ == '1.26.4', 'pinned numerical environment')
    need(torch.cuda.is_available() and torch.cuda.get_device_name() == 'NVIDIA GeForce RTX 3090', 'local RTX3090 required')
    free, total = torch.cuda.mem_get_info()
    need(free >= 8*1024**3, 'at least8GiB free GPU required')
    torch.cuda.set_per_process_memory_fraction(8*1024**3/total)
    return {'cgroup': group, 'effective': effective, 'service': service}


def main():
    started = time.monotonic()  # Cooperative deadline includes main-entry admission/preparation.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    need(args.execute, 'No scientific execution without explicit --execute')
    parent = args.output_dir.parent.resolve(strict=True)
    need(parent.is_relative_to(Path('/tmp/spectral-experiment-artifacts')) and not list(parent.iterdir()), 'unused large-volume parent')
    need(args.output_dir == parent/'acquisition-001' and not args.output_dir.exists()
         and not args.output_dir.is_symlink(), 'exclusive new acquisition-001 required')
    mount = subprocess.check_output(['findmnt', '-n', '-o', 'TARGET,SOURCE', '--target', str(parent)], text=True).split()
    need(mount == ['/private-artifacts/storage', '/dev/RECONFIGURE_FOR_LOCAL_STORAGE'], 'wrong mounted output volume')
    admission, pins, inventory = configure(), source_pins(), byte_inventory()
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    for name, expected in pins.items():
        committed = subprocess.check_output(['git', 'show', commit+':'+name], cwd=ROOT)
        need(hashlib.sha256(committed).hexdigest() == expected, 'source not committed: '+name)
    data_pins = {name: digest(accepted.DATA/name) for name in accepted.DATA_SHA256}
    need(data_pins == accepted.DATA_SHA256, 'accepted training dataset pins')
    need(shutil.disk_usage(parent).free > DISK_RESERVE+inventory['total_upper_bytes'], 'insufficient output/free-disk reserve')
    args.output_dir.mkdir(exist_ok=False)
    run = Run(args.output_dir, started=started)
    try:
        manifest = {'schema': SCHEMA, 'source_pins': pins, 'data_pins': data_pins,
            'data_directory': str(accepted.DATA), 'git_commit': commit, 'resource_admission': admission,
            'pid': os.getpid(), 'invocation_id': os.environ.get('INVOCATION_ID'), 'started_unix': time.time(),
            'seeds': list(SEEDS), 'cells': list(CELLS), 'schedules': list(SCHEDULES), 'actions': list(ACTIONS),
            'parent_acquisition': str(PARENT), 'parent_completion_sha256': COMPLETE_SHA, 'parent_audit_sha256': AUDIT_SHA,
            'stream_gradients': 600, 'oracle_gradients': 6, 'physical_readouts': 21, 'new_prediction_pairs': 21,
            'observer_clocks': [100, 150, 151], 'adam_clocks': [100, 101], 'byte_inventory': inventory,
            'torch': torch.__version__, 'numpy': np.__version__, 'python': sys.version,
            'cooperative_seconds': COOPERATIVE_SECONDS, 'cloud_spend_usd': 0,
            'scope': 'outcome-informed matched-parent observer-only histories and one-step copies; no training replay'}
        run.save('manifest.json', manifest)
        print(json.dumps({'resource_guard': 'PASS', 'unit': UNIT, 'output': str(args.output_dir),
                          'invocation_id': os.environ.get('INVOCATION_ID'), 'source_pins': pins,
                          'inventory': inventory, **admission}), flush=True)
        inputs = Inputs()
        result = acquire(run, inputs)
        inputs.recheck()
        need(source_pins() == pins and {name: digest(accepted.DATA/name) for name in accepted.DATA_SHA256} == data_pins,
             'sources/data changed during acquisition')
        run.save('complete.json', {'schema': SCHEMA, 'status': 'complete', 'results': result,
            'source_pins': pins, 'data_pins': data_pins, 'parent_completion_sha256': COMPLETE_SHA,
            'parent_audit_sha256': AUDIT_SHA, 'parent_input_receipts': list(inputs.used.values()),
            'finished_unix': time.time(), 'wall_seconds': time.monotonic()-run.started,
            'process_high_water_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'gpu_max_allocated_bytes': torch.cuda.max_memory_allocated(), 'receipts': list(run.receipts),
            'artifact_bytes_before_completion': run.used, 'completed_cases': 6, 'completed_stream_gradients': 600,
            'completed_oracle_gradients': 6, 'completed_physical_readouts': 21, 'completed_prediction_pairs': 21,
            'completed_observer_only_updates': 600, 'completed_self_inclusion_updates': 12})
    except BaseException as exc:
        data = json.dumps({'schema': SCHEMA, 'status': 'failed', 'type': type(exc).__name__, 'message': str(exc),
            'finished_unix': time.time(), 'wall_seconds': time.monotonic()-run.started,
            'completed_receipts': run.receipts}, indent=2, allow_nan=False).encode()
        need(len(data) <= FOOTER_RESERVE, 'failure metadata allowance exceeded')
        with (args.output_dir/'failed.json').open('xb') as file:
            file.write(data)
        raise


if __name__ == '__main__':
    main()
