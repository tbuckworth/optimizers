#!/usr/bin/env python3
"""Explicit, once-only saved-vector alignment accounting; no neural computation."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import shutil
import stat
import subprocess
import sys
import time

ENTRY_MONOTONIC = time.monotonic()
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DOCS = 'output/2026-09-10-spectral-observer-signal'
OLD_DOCS = 'output/2026-09-10-spectral-observer-pathway'
ACQUISITION = Path('/tmp/spectral-experiment-artifacts/spectral-observer-pathway-20260910.jTwt14/acquisition-001')
AUDIT = ACQUISITION.parent/'audit-001/result.json'
AUDIT_SHA = 'd9b05160c9a58952d4f0899699f79de2ee2d919c5319f2bb96d687ea448d0d35'
COMPLETE_SHA = 'b1c4837438051dc6ed51665879b6c37d7bd35673a86f8c4f363878c3f022464f'
MANIFEST_SHA = '3ffe8f04d4ad0fe37aa35691fd6a19115d3181ce5991aac701147823597ab8b3'
OLD_PINS = {OLD_DOCS+'/next-decision.md': '178184bbab9a0dbbab2d7b0a9a4d197248151db3b0c511127fcff39053d93d0a',
    OLD_DOCS+'/next-discriminator.md': 'a3c6222868de4e9833fce9e48878db08bb1b02128c858fbafd5f5ca747e2dc72',
    OLD_DOCS+'/next-design-review.md': '34f7bd31b8c338b434e2ca41eec8fdccdba4d08979e1531c0efd60dd7a82874a'}
SOURCE_NAMES = set(OLD_PINS) | {'scripts/analyze_spectral_observer_signal.py',
    'tests/test_spectral_observer_signal.py', DOCS+'/protocol.md', DOCS+'/implementation.md'}
SEEDS = (202609121, 202609122, 202609123)
CELLS, PROBES = ('clean', 'diffuse'), ('majority', 'rare8')
SCHEDULES = ('interleaved', 'grouped')
ACTIONS = ('native_interleaved', 'native_grouped', 'raw', 'zero')
PAIRS = (('native_grouped', 'native_interleaved'), ('native_grouped', 'raw'),
         ('native_interleaved', 'raw'), ('native_grouped', 'zero'),
         ('native_interleaved', 'zero'), ('raw', 'zero'))
SCHEMA, PARENT_SCHEMA = 'spectral_observer_signal_v1', 'spectral_observer_pathway_v1'
PARAMETERS, ARCHIVE_BYTES = 50890, 189384513
UNIT = 'spectral-observer-signal-001.service'
MAX_BYTES, FOOTER_RESERVE, DISK_RESERVE = 100*1024**2, 1024**2, 1024**3
COOPERATIVE_SECONDS = 150
EPS64, MULTIPLIER, ABSOLUTE_FLOOR = float(np.finfo(np.float64).eps), 128, 1e-12
TOLERANCES = {'epsilon_multiplier': MULTIPLIER, 'absolute_floor': ABSOLUTE_FLOOR,
    'single_dot': '128*eps64*sum(abs(x*y))+1e-12',
    'paired_identity': '128*eps64*(S_direct+S_left+S_right)+3e-12',
    'status': 'operational consistency thresholds, not arbitrary-reduction error theorems'}


class AnalysisError(RuntimeError):
    def __init__(self, message, evidence=None):
        super().__init__(message)
        self.evidence = evidence


def require(condition, message, evidence=None):
    if not condition:
        raise AnalysisError(message, evidence)


def sha256(path, check=lambda: None):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            check()
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    require(Path(path).stat().st_size <= 64*1024**2, 'oversized JSON')
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'duplicate JSON key')
            result[key] = value
        return result
    def bad(value):
        raise AnalysisError('nonfinite JSON token: '+value)
    return json.loads(Path(path).read_text(), object_pairs_hook=pairs, parse_constant=bad)


def finite(value):
    require(type(value) in (int, float) and math.isfinite(value), 'finite scalar required')
    return float(value)


def vector(value):
    result = np.asarray(value)
    require(result.ndim == 1 and result.size > 0 and result.dtype in (np.float32, np.float64)
            and np.isfinite(result).all(), 'finite floating vector required')
    return result.astype(np.float64, copy=True)


def alignment(probe, action):
    """Two independent reductions; fsum of FP64 products is the reported dot."""
    x, y = vector(probe), vector(action)
    require(x.shape == y.shape, 'matching vector shapes')
    products = x*y
    dot = math.fsum(float(a)*float(b) for a, b in zip(x, y))
    numpy_dot = float(np.dot(x, y))
    absolute_sum = math.fsum(abs(float(a)*float(b)) for a, b in zip(x, y))
    bound = MULTIPLIER*EPS64*absolute_sum+ABSOLUTE_FLOOR
    nx = math.sqrt(math.fsum(float(v)*float(v) for v in x))
    ny = math.sqrt(math.fsum(float(v)*float(v) for v in y))
    cosine = dot/(nx*ny) if nx and ny else None
    row = {'dot': dot, 'numpy_dot': numpy_dot, 'absolute_product_sum': absolute_sum,
           'operational_bound': bound, 'reduction_discrepancy': abs(dot-numpy_dot),
           'left_norm': nx, 'right_norm': ny, 'cosine': cosine,
           'zero_norm': nx == 0 or ny == 0,
           'raw_sign': 'positive' if dot > 0 else 'negative' if dot < 0 else 'zero',
           'sign_status': 'positive' if dot > bound else 'negative' if dot < -bound else 'roundoff_unresolved'}
    require(np.isfinite(products).all() and all(math.isfinite(row[k]) for k in
            ('dot', 'numpy_dot', 'absolute_product_sum', 'operational_bound', 'left_norm', 'right_norm')),
            'nonfinite alignment arithmetic', row)
    require(cosine is None or math.isfinite(cosine) and abs(cosine) <= 1+32*EPS64,
            'invalid signed cosine', row)
    require(row['reduction_discrepancy'] <= bound, 'independent dot reductions disagree', row)
    return row


def difference_alignment(probe, left, right, left_row=None, right_row=None):
    """Subtract after FP64 conversion; bound includes both parent product sums."""
    x, y = vector(left), vector(right)
    direct = alignment(probe, x-y)
    a = alignment(probe, x) if left_row is None else left_row
    b = alignment(probe, y) if right_row is None else right_row
    paired = math.fsum([a['dot'], -b['dot']])
    total_sum = direct['absolute_product_sum']+a['absolute_product_sum']+b['absolute_product_sum']
    bound = MULTIPLIER*EPS64*total_sum+3*ABSOLUTE_FLOOR
    evidence = {'paired_dot_difference': paired, 'discrepancy': abs(direct['dot']-paired),
                'absolute_constituent_sum': total_sum, 'operational_bound': bound}
    require(evidence['discrepancy'] <= bound, 'difference-vector/paired-dot disagreement', evidence)
    direct['paired_identity'] = evidence
    # Sign resolution also respects the cancellation-aware identity threshold.
    direct['sign_status'] = ('positive' if direct['dot'] > bound else
                             'negative' if direct['dot'] < -bound else 'roundoff_unresolved')
    return direct


def sample_summary(values):
    require(len(values) == 3, 'three values in fixed seed order')
    present = [finite(x) for x in values if x is not None]
    row = {'seeds': list(SEEDS), 'values': values, 'present_count': len(present),
           'mean': None, 'sample_sd': None, 'sample_se': None,
           'positive_count': None, 'negative_count': None, 'zero_count': None}
    if len(present) == 3:
        mean = math.fsum(present)/3
        sd = math.sqrt(math.fsum((x-mean)**2 for x in present)/2)
        row.update(mean=mean, sample_sd=sd, sample_se=sd/math.sqrt(3),
                   positive_count=sum(x > 0 for x in present), negative_count=sum(x < 0 for x in present),
                   zero_count=sum(x == 0 for x in present))
    return row


def unique_rows(rows, keys, expected):
    index = {tuple(row[k] for k in keys): row for row in rows}
    require(len(index) == len(rows) == len(expected) and set(index) == set(expected), 'exact row roster: '+str(keys))
    return index


def analyze_case(seed, cell, probes, common, native, accepted_case, checked_readouts, checked_histories):
    require(seed in SEEDS and cell in CELLS and set(probes) == set(PROBES)
            and set(native) == set(SCHEDULES), 'case vector roster')
    require((accepted_case['seed'], accepted_case['cell']) == (seed, cell)
            and set(accepted_case['actions']) == set(ACTIONS), 'accepted scalar case identity')
    g = vector(common)
    actions = {'native_'+s: vector(native[s]) for s in SCHEDULES}
    actions.update(raw=g.copy(), zero=np.zeros_like(g))
    result = {'seed': seed, 'cell': cell, 'probes': {}, 'contrasts': []}
    for probe in PROBES:
        q = vector(probes[probe])
        base = alignment(q, g)
        rows = {}
        metric = 'rare' if probe == 'rare8' else 'majority_macro'
        zero_id = f's{seed}-zero'
        zero_j = finite(checked_readouts[zero_id]['oracle_signed_utilities'][probe]['total'])
        for action in ACTIONS:
            h = actions[action]
            delivered = alignment(q, h)
            changed = difference_alignment(q, h, g, delivered, base)
            if action == 'raw':
                require(delivered['dot'] == base['dot'] and changed['dot'] == 0., 'raw alignment identities')
            if action == 'zero':
                require(delivered['dot'] == 0. and changed['dot'] == -base['dot'], 'zero alignment identities')
            old = accepted_case['actions'][action]
            ident = zero_id if action == 'zero' else f's{seed}-{cell}-{action}'
            require(old['readout']['physical_id'] == ident
                    and old['shared_across_label_cells'] is (action == 'zero'), 'physical readout binding')
            checked = checked_readouts[ident]
            j = {part: finite(checked['oracle_signed_utilities'][probe][part]) for part in ('total', 'decay', 'adaptive')}
            improvements = old['improvements']
            joined = {'J': j, 'E_over_zero': j['total']-zero_j,
                'U_heldout_ce': finite(improvements[metric+'_ce']),
                'U_train_true_ce': finite(improvements['train_true_'+metric+'_ce']),
                'heldout_accuracy_change': finite(improvements[metric+'_accuracy']),
                'train_true_accuracy_change': finite(improvements['train_true_'+metric+'_accuracy']),
                'U_train_assigned_ce': finite(improvements['train_assigned_ce']),
                'U_train_wrong_ce': None if improvements['train_wrong_ce'] is None else finite(improvements['train_wrong_ce'])}
            history = None if not action.startswith('native_') else checked_histories[(seed, cell, action.removeprefix('native_'))]
            rows[action] = {'F': delivered, 'K': changed, 'accepted_join': joined,
                'accepted_probe_accessibility': None if history is None else history['post_inclusion']['geometry'][probe],
                'accepted_movement': checked['movement'], 'physical_id': ident,
                'accepted_state_receipt': old['readout']['state'],
                'accepted_logit_receipt': old['readout']['logits'],
                'control_identity': {'raw': 'F=B; K=0', 'zero': 'F=0; K=-B'}.get(action)}
        result['probes'][probe] = {'B': base, 'actions': rows}
        for left, right in PAIRS:
            a, b = rows[left]['accepted_join'], rows[right]['accepted_join']
            result['contrasts'].append({'seed': seed, 'cell': cell, 'probe': probe,
                'contrast': left+'_minus_'+right, 'left': left, 'right': right,
                'primary': (left, right) == PAIRS[0],
                'D_filter': difference_alignment(q, actions[left], actions[right], rows[left]['F'], rows[right]['F']),
                'D_Adam': {part: a['J'][part]-b['J'][part] for part in ('total', 'decay', 'adaptive')},
                'D_U_heldout_ce': a['U_heldout_ce']-b['U_heldout_ce'],
                'D_U_train_true_ce': a['U_train_true_ce']-b['U_train_true_ce'],
                'D_heldout_accuracy': a['heldout_accuracy_change']-b['heldout_accuracy_change']})
    return result


def summarize(cases, labels):
    indexed = unique_rows(cases, ('seed', 'cell'), [(s, c) for s in SEEDS for c in CELLS])
    label_index = unique_rows(labels, ('seed', 'probe'), [(s, p) for s in SEEDS for p in PROBES])
    out = {'input': {}, 'actions': {}, 'contrasts': {}, 'label_intervention': {}}
    def align(rows):
        summary = {key: sample_summary([r[key] for r in rows]) for key in ('dot', 'left_norm', 'right_norm', 'cosine')}
        summary['sign_status_counts'] = {status: sum(r['sign_status'] == status for r in rows)
            for status in ('positive', 'negative', 'roundoff_unresolved')}
        return summary
    for cell in CELLS:
        for probe in PROBES:
            panels = [indexed[s, cell]['probes'][probe] for s in SEEDS]
            out['input'][cell+'/'+probe] = align([p['B'] for p in panels])
            for action in ACTIONS:
                rows = [p['actions'][action] for p in panels]
                out['actions'][cell+'/'+probe+'/'+action] = {
                    'F': align([r['F'] for r in rows]), 'K': align([r['K'] for r in rows]),
                    'J': {part: sample_summary([r['accepted_join']['J'][part] for r in rows]) for part in ('total', 'decay', 'adaptive')},
                    **{key: sample_summary([r['accepted_join'][key] for r in rows]) for key in
                       ('E_over_zero', 'U_heldout_ce', 'U_train_true_ce', 'heldout_accuracy_change',
                        'train_true_accuracy_change', 'U_train_assigned_ce', 'U_train_wrong_ce')}}
            for left, right in PAIRS:
                name = left+'_minus_'+right
                rows = [next(r for r in indexed[s, cell]['contrasts'] if r['probe'] == probe and r['contrast'] == name) for s in SEEDS]
                out['contrasts'][cell+'/'+probe+'/'+name] = {'D_filter': align([r['D_filter'] for r in rows]),
                    'D_Adam': {part: sample_summary([r['D_Adam'][part] for r in rows]) for part in ('total', 'decay', 'adaptive')},
                    **{key: sample_summary([r[key] for r in rows]) for key in
                       ('D_U_heldout_ce', 'D_U_train_true_ce', 'D_heldout_accuracy')}}
    for probe in PROBES:
        out['label_intervention'][probe] = align([label_index[s, probe]['alignment'] for s in SEEDS])
    return out


def selected_names():
    return [name for seed in SEEDS for name in [f'oracles-s{seed}.pt', *[
        name for cell in CELLS for name in [f'common-action-s{seed}-{cell}.pt',
            *[f'observer-s{seed}-{cell}-{schedule}.pt' for schedule in SCHEDULES]]]]]


def receipt_index(rows):
    index = {}
    for row in rows:
        require(type(row) is dict and set(row) == {'path', 'size_bytes', 'sha256'}, 'receipt schema')
        name = row['path']
        require(type(name) is str and Path(name).name == name and name not in ('', '.', '..')
                and name not in index, 'unique contained receipt name')
        require(type(row['size_bytes']) is int and 0 <= row['size_bytes'] <= 100*1024**2
                and type(row['sha256']) is str and len(row['sha256']) == 64
                and all(c in '0123456789abcdef' for c in row['sha256']), 'receipt size/hash')
        index[name] = row
    return index


def verify_file(root, receipt, check=lambda: None):
    path = root/receipt['path']
    require(stat.S_ISREG(path.lstat().st_mode) and path.resolve().parent == root.resolve(), 'regular direct-child input')
    require(path.stat().st_size == receipt['size_bytes'] and sha256(path, check) == receipt['sha256'], 'input receipt mismatch')
    return path


def vector_hash32(value):
    """Documented scalar-tensor I9 encoding, not an imported producer helper."""
    x = np.ascontiguousarray(value, dtype=np.float32)
    node = ['tensor', 'torch.float32', list(x.shape), hashlib.sha256(x.tobytes()).hexdigest()]
    encoded = json.dumps(node, ensure_ascii=True, separators=(',', ':')).encode('ascii')
    return hashlib.sha256(b'i9_neural_tree_v1\n'+encoded).hexdigest()


def restricted_vectors(path, kind, seed, cell=None, schedule=None):
    """Only called after explicit execution/admission. No model or GPU APIs."""
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'restricted loader requires hidden CUDA')
    import torch
    require(torch.__version__ == '2.11.0+cu128', 'pinned restricted-loader version')
    torch.set_num_threads(1)
    payload = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    require(type(payload) is dict and payload['schema'] == PARENT_SCHEMA and payload['seed'] == seed,
            'tensor archive identity')
    if cell is not None:
        require(payload['cell'] == cell, 'tensor label cell')
    if schedule is not None:
        require(payload['schedule'] == schedule, 'tensor observer schedule')
    def extract(value):
        require(type(value) is torch.Tensor and value.device.type == 'cpu' and value.dtype == torch.float32
                and tuple(value.shape) == (PARAMETERS,), 'saved FP32 CPU vector shape')
        result = value.detach().numpy().copy()
        require(np.isfinite(result).all(), 'finite saved vector')
        return result
    if kind == 'oracles':
        require(set(payload['groups']) == set(PROBES) and payload['use'] == 'oracle diagnostic only; never delivered or observed', 'oracle groups/use')
        vectors = {q: extract(payload['groups'][q]['mean_gradient']) for q in PROBES}
        metadata = {'parent_sha256': payload['parent_sha256']}
    elif kind == 'common':
        vectors = {'g_star': extract(payload['g_star'])}
        metadata = {}
    else:
        require(kind == 'observer', 'permitted archive kind')
        vectors = {'post_action': extract(payload['post_action'])}
        metadata = {'parent_sha256': payload['parent_sha256'], 'g_star_sha256': payload['g_star_sha256']}
    del payload
    return vectors, metadata


class Inputs:
    def __init__(self, check):
        self.check, self.loaded = check, []
        for path, expected in ((AUDIT, AUDIT_SHA), (ACQUISITION/'complete.json', COMPLETE_SHA),
                               (ACQUISITION/'manifest.json', MANIFEST_SHA)):
            require(stat.S_ISREG(path.lstat().st_mode) and sha256(path, check) == expected, 'accepted JSON hash')
        self.audit = read_json(AUDIT)
        completion, manifest = read_json(ACQUISITION/'complete.json'), read_json(ACQUISITION/'manifest.json')
        a = self.audit
        require(a['schema'] == 'spectral_observer_pathway_independent_audit_v1' and a['status'] == 'PASS'
                and a['errors'] == [] and a['checks'] == 33032 and a['acquisition_dir'] == str(ACQUISITION)
                and a['expected_completion_sha256'] == COMPLETE_SHA and a['expected_manifest_sha256'] == MANIFEST_SHA,
                'accepted complete audit identity')
        for key, count in {'cases_checked': 6, 'observer_histories_checked': 12, 'stream_gradients_checked': 600,
                'oracle_gradients_checked': 6, 'physical_readouts_checked': 21, 'logical_readouts_checked': 24,
                'prediction_pairs_checked': 21}.items():
            require(a[key] == count, 'accepted audit full roster')
        require(completion['schema'] == manifest['schema'] == PARENT_SCHEMA and completion['status'] == 'complete'
                and completion['source_pins'] == manifest['source_pins'] == a['expected_sources']
                and completion['receipts'] == a['input_receipts'], 'completion/manifest/audit bindings')
        self.index = receipt_index(completion['receipts'])
        require(len(self.index) == 122 and all(name in self.index for name in selected_names()), 'selected input archive inventory')
        self.selected = [self.index[name] for name in selected_names()]
        require(len(self.selected) == 21 and sum(r['size_bytes'] for r in self.selected) == ARCHIVE_BYTES,
                'fixed 21-archive byte inventory')
        self.cases = unique_rows(a['checked_cases'], ('seed', 'cell'), [(s, c) for s in SEEDS for c in CELLS])
        self.histories = unique_rows(a['checked_histories'], ('seed', 'cell', 'schedule'),
                                    [(s, c, t) for s in SEEDS for c in CELLS for t in SCHEDULES])
        physical = [f's{s}-zero' for s in SEEDS]+[f's{s}-{c}-{x}' for s in SEEDS for c in CELLS for x in ACTIONS[:-1]]
        indexed = unique_rows(a['checked_readouts'], ('physical_id',), [(x,) for x in physical])
        self.readouts = {key[0]: value for key, value in indexed.items()}
        for seed in SEEDS:
            left, right = [self.cases[seed, c]['actions']['zero']['readout'] for c in CELLS]
            require(left == right, 'same physical zero reference across cells')

    def load(self, name, kind, seed, cell=None, schedule=None):
        require(name in selected_names() and name not in self.loaded, 'one load per selected archive')
        self.check()
        path = verify_file(ACQUISITION, self.index[name], self.check)
        result = restricted_vectors(path, kind, seed, cell, schedule)
        self.loaded.append(name)
        self.check()
        return result

    def recheck(self):
        require(self.loaded == selected_names(), 'all and only the 21 scoped archive loads')
        for row in self.selected:
            verify_file(ACQUISITION, row, self.check)
        for path, expected in ((AUDIT, AUDIT_SHA), (ACQUISITION/'complete.json', COMPLETE_SHA),
                               (ACQUISITION/'manifest.json', MANIFEST_SHA)):
            require(sha256(path, self.check) == expected, 'accepted JSON changed during accounting')


def account(inputs, run):
    cases, labels = [], []
    for seed in SEEDS:
        probes, metadata = inputs.load(f'oracles-s{seed}.pt', 'oracles', seed)
        parent_sha = metadata['parent_sha256']
        commons, bases = {}, {}
        for cell in CELLS:
            old = inputs.cases[seed, cell]
            require(all(r['readout']['parent_sha256'] == parent_sha for r in old['actions'].values()), 'common accepted parent/probe identity')
            data, _ = inputs.load(f'common-action-s{seed}-{cell}.pt', 'common', seed, cell)
            common = data['g_star']
            common_sha = vector_hash32(common)
            require(common_sha == old['mean_admission']['g_star_sha256'], 'accepted common input bytes')
            native = {}
            for schedule in SCHEDULES:
                data, meta = inputs.load(f'observer-s{seed}-{cell}-{schedule}.pt', 'observer', seed, cell, schedule)
                require(meta == {'parent_sha256': parent_sha, 'g_star_sha256': common_sha}, 'observer parent/common identity')
                native[schedule] = data['post_action']
            case = analyze_case(seed, cell, probes, common, native, old, inputs.readouts, inputs.histories)
            case['vector_bindings'] = {'parent_sha256': parent_sha, 'g_star_i9_sha256': common_sha,
                'oracle_i9_sha256': {q: vector_hash32(probes[q]) for q in PROBES},
                'native_i9_sha256': {s: vector_hash32(native[s]) for s in SCHEDULES}}
            run.save(f'case-s{seed}-{cell}.json', case)
            cases.append(case)
            commons[cell] = common
            bases[cell] = {q: case['probes'][q]['B'] for q in PROBES}
        for probe in PROBES:
            labels.append({'seed': seed, 'probe': probe, 'contrast': 'diffuse_minus_clean_input',
                'alignment': difference_alignment(probes[probe], commons['diffuse'], commons['clean'],
                    bases['diffuse'][probe], bases['clean'][probe])})
        run.check()
    require(len(cases) == 6 and sum(len(c['contrasts']) for c in cases) == 72 and len(labels) == 6, 'complete new scalar roster')
    return {'schema': SCHEMA, 'cases': cases, 'label_interventions': labels,
        'summary': summarize(cases, labels),
        'counts': {'cases': 6, 'input_probe_alignments': 12, 'native_delivered_alignments': 24,
                  'native_filter_changes': 24, 'logical_action_probe_joins': 48,
                  'primary_grouping_contrasts': 12, 'all_control_contrasts': 72,
                  'label_input_contrasts': 6, 'accepted_physical_readouts': 21,
                  'accepted_logical_readouts': 24, 'loaded_archives': 21},
        'units': {'B_F_K_D_filter': 'gradient inner products; not finite CE improvements',
                  'J_E_D_Adam': 'probe-gradient dot actual displacement; first-order loss units',
                  'U': 'finite group CE before minus after; positive helpful'},
        'scope': 'Post-hoc three reused parents; new saved-vector cross terms plus accepted scalar joins. No new neural measurements or causal mediation fraction.'}


class Run:
    def __init__(self, output, started):
        self.output, self.started, self.used, self.receipts = output, started, 0, []

    def check(self):
        require(time.monotonic()-self.started <= COOPERATIVE_SECONDS, '150-second cooperative deadline')
        require(shutil.disk_usage(self.output.parent).free >= DISK_RESERVE, '1 GiB free-disk reserve')

    def save(self, name, payload, footer=False):
        require(Path(name).name == name and name not in ('', '.', '..'), 'direct output name')
        if not footer:
            self.check()
        encoded = (json.dumps(payload, indent=2, allow_nan=False)+'\n').encode()
        limit = MAX_BYTES if footer else MAX_BYTES-FOOTER_RESERVE
        require(len(encoded) <= (FOOTER_RESERVE if footer else MAX_BYTES) and self.used+len(encoded) <= limit,
                'bounded JSON output/footer')
        with (self.output/name).open('xb') as stream:
            stream.write(encoded)
        self.used += len(encoded)
        receipt = {'path': name, 'size_bytes': len(encoded), 'sha256': hashlib.sha256(encoded).hexdigest()}
        self.receipts.append(receipt)
        return receipt


def resource_guard():
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CUDA must be hidden')
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        require(os.environ.get(name) == '1', 'one math thread: '+name)
    group = next(line.split(':', 2)[2] for line in Path('/proc/self/cgroup').read_text().splitlines() if line.startswith('0::'))
    require(Path(group).name == UNIT, 'dedicated saved-vector unit required')
    folder = Path('/sys/fs/cgroup')/group.lstrip('/')
    limits = {key: (folder/key).read_text().strip() for key in ('memory.max', 'memory.swap.max', 'cpu.max')}
    quota, period = limits['cpu.max'].split()
    require(limits['memory.max'] == str(4*1024**3) and limits['memory.swap.max'] == '0'
            and quota != 'max' and int(quota) == int(period) > 0, '4 GiB/no-swap/one-CPU envelope')
    text = subprocess.check_output(['systemctl', '--user', 'show', UNIT, '--property=Type',
        '--property=RuntimeMaxUSec', '--property=Restart', '--property=KillMode'], text=True)
    service = dict(line.split('=', 1) for line in text.splitlines())
    require(service == {'Type': 'exec', 'RuntimeMaxUSec': '3min', 'Restart': 'no', 'KillMode': 'control-group'}, '180-second service/no-retry envelope')
    require(np.__version__ == '1.26.4', 'pinned NumPy version')
    return {'cgroup': group, 'effective': limits, 'service': service}


def sources(pin_file, check):
    pins = read_json(pin_file)
    require(set(pins) == SOURCE_NAMES and all(pins[k] == v for k, v in OLD_PINS.items()), 'exact source/protocol/decision map')
    commit = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    for name, expected in pins.items():
        require(sha256(ROOT/name, check) == expected, 'source byte pin: '+name)
        saved = subprocess.check_output(['git', '-C', str(ROOT), 'show', commit+':'+name])
        require(hashlib.sha256(saved).hexdigest() == expected, 'source must match committed bytes: '+name)
    return pins, commit


def main(argv=None):
    started = ENTRY_MONOTONIC
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--expected-sources-json', required=True, type=Path)
    args = parser.parse_args(argv)
    require(args.execute, 'explicit --execute required; import/tests never admit scientific data')
    parent = args.output_dir.parent.resolve(strict=True)
    require(args.output_dir == parent/'analysis-001' and parent.parent == Path('/tmp/spectral-experiment-artifacts')
            and not args.output_dir.exists() and not args.output_dir.is_symlink()
            and not any(parent.iterdir()), 'unused exclusive large-volume output parent')
    mount = subprocess.check_output(['findmnt', '-n', '-o', 'TARGET,SOURCE', '-T', str(parent)], text=True).split()
    require(mount == ['/private-artifacts/storage', '/dev/RECONFIGURE_FOR_LOCAL_STORAGE'], 'mounted large-volume output')
    admission = resource_guard()
    args.output_dir.mkdir(exist_ok=False)
    run = Run(args.output_dir, started)
    pins, inputs, source_map_sha = None, None, None
    try:
        run.check()
        require(shutil.disk_usage(parent).free >= MAX_BYTES+DISK_RESERVE, 'output cap plus disk reserve')
        pins, commit = sources(args.expected_sources_json, run.check)
        source_map_sha = sha256(args.expected_sources_json, run.check)
        inputs = Inputs(run.check)
        run.save('manifest.json', {'schema': SCHEMA, 'status': 'admitted', 'source_pins': pins,
            'source_map_sha256': source_map_sha, 'git_commit': commit, 'seeds': list(SEEDS), 'cells': list(CELLS),
            'probes': list(PROBES), 'actions': list(ACTIONS), 'selected_input_receipts': inputs.selected,
            'selected_input_bytes': ARCHIVE_BYTES, 'acquisition_dir': str(ACQUISITION),
            'audit_sha256': AUDIT_SHA, 'completion_sha256': COMPLETE_SHA, 'manifest_sha256': MANIFEST_SHA,
            'tolerances': TOLERANCES, 'resource_admission': admission, 'numpy': np.__version__,
            'python': sys.version, 'pid': os.getpid(), 'invocation_id': os.environ.get('INVOCATION_ID'),
            'cooperative_seconds': COOPERATIVE_SECONDS, 'output_cap_bytes': MAX_BYTES, 'disk_reserve_bytes': DISK_RESERVE})
        result = account(inputs, run)
        inputs.recheck()
        for name, expected in pins.items():
            require(sha256(ROOT/name, run.check) == expected, 'source changed during accounting')
        require(sha256(args.expected_sources_json, run.check) == source_map_sha, 'source map changed')
        result.update(status='PASS', tolerances=TOLERANCES, input_receipts=inputs.selected,
            source_pins=pins, accepted_audit_sha256=AUDIT_SHA, accepted_completion_sha256=COMPLETE_SHA,
            accepted_manifest_sha256=MANIFEST_SHA, elapsed_seconds=time.monotonic()-started,
            process_high_water_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        result_receipt = run.save('result.json', result)
        run.check()
        run.save('complete.json', {'schema': SCHEMA, 'status': 'complete', 'result': result_receipt,
            'receipts': list(run.receipts), 'input_receipts': inputs.selected, 'source_pins': pins,
            'source_map_sha256': source_map_sha, 'counts': result['counts'], 'accepted_audit_sha256': AUDIT_SHA,
            'accepted_completion_sha256': COMPLETE_SHA, 'resource_admission': admission,
            'elapsed_seconds': time.monotonic()-started,
            'process_high_water_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}, footer=True)
        print(json.dumps({'status': 'complete', 'output': str(args.output_dir), 'result': result_receipt, 'bytes': run.used}))
        return 0
    except Exception as exc:
        failure = {'schema': SCHEMA, 'status': 'failed', 'error_type': type(exc).__name__,
            'error': str(exc), 'evidence': getattr(exc, 'evidence', None), 'receipts': list(run.receipts),
            'source_pins': pins, 'source_map_sha256': source_map_sha,
            'input_receipts': [] if inputs is None else inputs.selected,
            'loaded_archives': [] if inputs is None else inputs.loaded,
            'elapsed_seconds': time.monotonic()-started, 'resource_admission': admission,
            'process_high_water_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'accepted_audit_sha256': AUDIT_SHA, 'accepted_completion_sha256': COMPLETE_SHA,
            'no_retry': True}
        run.save('failed.json', failure, footer=True)
        print(json.dumps({'status': 'failed', 'error': str(exc), 'output': str(args.output_dir)}))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
