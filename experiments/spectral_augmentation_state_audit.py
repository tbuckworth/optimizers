#!/usr/bin/env python3
"""NumPy-only fixed-state audit. No torch, forward pass, or training replay.

Autograd arrays, model logits and observer-history reconstruction remain
producer assertions. Independently check their archived algebra and metrics.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import resource
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import spectral_general_augmentation_audit as common

require, sha = common.require, common.sha
DOCS = ROOT / 'output/2026-09-10-spectral-augmentation-state'
SCHEMA = 'spectral_augmentation_state_v1'
AUDIT_SCHEMA = 'spectral_augmentation_state_audit_v1'
SEEDS = (202609141, 202609142, 202609143)
WARMUPS = ('none', 'translate')
ACTIONS = ('raw', 'native', 'raw_to_native', 'native_to_raw', 'decay')
FRACTIONS = (1., .1)
OUTPUT_CAP = 32 * 1024**2


@dataclass(frozen=True)
class Dimensions:
    """Small overrides for fabricated tests only; never CLI arguments."""
    p: int = 50890
    b: int = 64
    n: int = 256
    k: int = 32
    sizes: tuple = (50176, 64, 640, 10)


def specs(d=Dimensions()):
    f = np.float32
    return {
        'per_example_grads': (f, (5, d.b, d.p)), 'batch_grads': (f, (5, d.p)),
        'theta': (f, (d.p,)), 'adam_m': (f, (d.p,)), 'adam_v': (f, (d.p,)),
        'adam_steps': (np.int64, (4,)), 'param_sizes': (np.int64, (4,)),
        'basis_before': (f, (d.p, d.k)), 'basis_before_rank': (np.int64, ()),
        'basis_after': (f, (5, d.p, d.k)), 'basis_after_ranks': (np.int64, (5,)),
        'applied_grads': (f, (5, 2, d.p)), 'planned_data': (f, (5, 5, d.p)),
        'after_parameters': (f, (5, 5, 2, d.p)), 'valid': (np.bool_, (5, 5)),
        'scales': (np.float64, (5, 5)), 'objective_grads': (f, (2, d.p)),
        'baseline_logits': (f, (2, d.n, 10)),
        'after_logits': (f, (5, 5, 2, 2, d.n, 10)),
    }


def diagnostic_plan(raw_labels, seed, d=Dimensions()):
    original = common.regenerate_plan(raw_labels, seed)
    rng = lambda stream: np.random.Generator(np.random.PCG64(np.random.SeedSequence([stream, seed])))
    train = rng(10).permutation(5000)[:d.b].astype(np.int64)
    evaluation = rng(12).permutation(5000)[:d.n].astype(np.int64)
    tid, eid = original['train_ids'][train], original['eval_ids'][evaluation]
    return {'train_positions': train, 'eval_positions': evaluation,
            'train_ids': tid, 'eval_ids': eid, 'train_labels': raw_labels[tid],
            'eval_labels': raw_labels[eid],
            'train_shifts': rng(11).integers(-2, 3, size=(4, d.b, 2), dtype=np.int8),
            'eval_shifts': rng(13).integers(-2, 3, size=(d.n, 2), dtype=np.int8)}


def close_vector(actual, expected, label, atol=1e-6, rtol=5e-5):
    actual, expected = np.asarray(actual, dtype=np.float64), np.asarray(expected, dtype=np.float64)
    require(actual.shape == expected.shape and np.isfinite(actual).all()
            and np.isfinite(expected).all(), 'vector schema: '+label)
    error = float(np.linalg.norm(actual - expected))
    norm = float(np.linalg.norm(expected))
    require(error <= atol + rtol*norm, 'vector mismatch: '+label)
    return {'l2_error': error, 'reference_norm': norm, 'allowed_error': atol+rtol*norm}


def basis_check(q, rank):
    require(0 <= rank <= q.shape[1], 'basis rank')
    require(np.count_nonzero(q[:, rank:]) == 0, 'basis padding must be zero')
    live = q[:, :rank].astype(np.float64)
    gram = live.T @ live
    error = float(np.max(np.abs(gram - np.eye(rank)), initial=0))
    require(error <= 1e-3, 'basis orthogonality')
    return live, error


def geometry(values, q):
    """[5,B,P] finite-grid trace decomposition under actual QQ^T operator."""
    g = values.astype(np.float64)
    translated = g[1:]
    per_image = translated.mean(axis=0)
    grand = per_image.mean(axis=0)
    original = g[0].mean(axis=0)
    gram = q.T @ q
    p = g.shape[-1]
    def energy(a):
        a = a.reshape(-1, p)
        coordinates = a @ q
        full = float(np.einsum('ij,ij->', a, a) / len(a))
        kept = float(np.einsum('ik,kl,il->', coordinates, gram, coordinates) / len(a))
        return {'trace': full, 'operator_trace': kept,
                'operator_retention': None if full == 0 else kept/full}
    result = {'total': energy(translated-grand), 'between': energy(per_image-grand),
              'within': energy(translated-per_image[None]),
              'original_centered': energy(g[0]-original), 'original_mean': energy(original),
              'translated_mean': energy(grand), 'mean_translation_change': energy(grand-original)}
    errors = {}
    for field in ('trace', 'operator_trace'):
        error = result['total'][field] - result['between'][field] - result['within'][field]
        require(abs(error) <= 1e-10+1e-9*abs(result['total'][field]), 'finite-grid decomposition')
        errors[field] = error
    result['identity_residuals'] = errors
    result['denominators'] = {'total': 4*g.shape[1], 'between': g.shape[1], 'within': 4*g.shape[1]}
    return result


def native_projection(q, gradient):
    # Canonical adaptive=none uses V=None for an empty truncated eigensystem;
    # _project_gradient returns the raw gradient in that case, not zero.
    g = gradient.astype(np.float64)
    return g if q.shape[1] == 0 else q @ (q.T @ g)


def adam_reference(theta, m, v, gradient, steps, sizes):
    """Real-arithmetic carried-Adam reference, checked with fixed FP32 bound."""
    gradient = gradient.astype(np.float64)
    first = .9*m.astype(np.float64) + .1*gradient
    second = .999*v.astype(np.float64) + .001*gradient**2
    result = np.empty_like(gradient)
    offset = 0
    for step, size in zip(steps, sizes):
        sl = slice(offset, offset+int(size))
        t = int(step)+1
        result[sl] = -.001*(first[sl]/(1-.9**t))/(np.sqrt(second[sl]/(1-.999**t))+1e-8)
        offset += int(size)
    require(offset == len(theta), 'Adam parameter coverage')
    return result


def strict_mean(values):
    return None if any(x is None for x in values) else math.fsum(values)/len(values)


def summarize_parent(rows):
    lookup = {(r['view'], r['action'], r['fraction'], r['objective']): r for r in rows}
    contrasts = (('native_to_raw-minus-raw', 'native_to_raw', 'raw'),
                 ('native-minus-raw_to_native', 'native', 'raw_to_native'),
                 ('native_to_raw-minus-native', 'native_to_raw', 'native'),
                 ('native-minus-raw', 'native', 'raw'))
    fields = ('ce_improvement', 'accuracy_change', 'linear_total', 'linear_data', 'ce_data_improvement')
    output = []
    for mode, views in (('original', (0,)), ('translate', (1, 2, 3, 4))):
        for fraction in FRACTIONS:
            for objective in ('original', 'translate'):
                absolute = {}
                for action in ACTIONS:
                    selected = [lookup[v, action, fraction, objective] for v in views]
                    absolute[action] = {field: strict_mean([r[field] for r in selected]) for field in fields}
                    absolute[action]['available_draws'] = sum(r['available'] for r in selected)
                differences = {}
                for name, left, right in contrasts:
                    differences[name] = {}
                    for field in fields:
                        pairs = [(lookup[v, left, fraction, objective][field],
                                  lookup[v, right, fraction, objective][field]) for v in views]
                        vals = [None if a is None or b is None else a-b for a, b in pairs]
                        differences[name][field] = {'draws': vals, 'mean': strict_mean(vals),
                                                   'available_draws': sum(x is not None for x in vals)}
                output.append({'input_mode': mode, 'fraction': fraction, 'objective': objective,
                               'required_draws': len(views), 'actions': absolute, 'contrasts': differences})
    return output


def audit_parent(a, labels, d=Dimensions(), check=lambda: None):
    require(set(a) == set(specs(d)), 'parent array inventory')
    for key, (dtype, shape) in specs(d).items():
        common.checked_array(a[key], dtype, shape, key)
    require(tuple(a['param_sizes']) == d.sizes and sum(d.sizes) == d.p, 'parameter layout')
    require(np.all(a['adam_steps'] == 100) and np.all(a['adam_v'] >= 0), 'carried Adam state')
    require(np.all(a['valid'][:, [0, 1, 4]]), 'raw/native/decay always available')
    q, orthogonal_error = basis_check(a['basis_before'], int(a['basis_before_rank']))
    require(q.shape[1] == d.k, 'warmup rank')
    geo = geometry(a['per_example_grads'], q)
    geo['basis_gram_max_abs_error'] = orthogonal_error
    theta = a['theta']
    decay = np.multiply(theta, np.float32(1-.001*.01), dtype=np.float32)
    errors, norm_rows, rows = [], [], []
    baseline = [common.classification(z, labels) for z in a['baseline_logits']]
    for view in range(5):
        check()
        errors.append(close_vector(a['batch_grads'][view], a['per_example_grads'][view].astype(np.float64).mean(axis=0), 'batch mean'))
        require(np.array_equal(a['applied_grads'][view, 0], a['batch_grads'][view]), 'raw applied gradient')
        current, gram_error = basis_check(a['basis_after'][view], int(a['basis_after_ranks'][view]))
        projected = native_projection(current, a['batch_grads'][view])
        errors.append(close_vector(a['applied_grads'][view, 1], projected, 'native current-basis projection'))
        full = a['after_parameters'][view, :, 0]
        data = [(full[j]-decay).astype(np.float32) for j in (0, 1)]
        norms = [float(np.linalg.norm(x.astype(np.float64))) for x in data]
        for action in (0, 1):
            reference = adam_reference(theta, a['adam_m'], a['adam_v'], a['applied_grads'][view, action], a['adam_steps'], a['param_sizes'])
            errors.append(close_vector(data[action], reference, 'carried Adam data delta'))
            require(np.array_equal(a['planned_data'][view, action], data[action])
                    and a['scales'][view, action] == 1., 'actual data delta/scales')
        require(np.array_equal(full[4], decay) and np.count_nonzero(a['planned_data'][view, 4]) == 0
                and a['scales'][view, 4] == 0, 'rounded decay-only baseline')
        match_errors = []
        for action, src, tgt in ((2, 0, 1), (3, 1, 0)):
            scale = None if norms[src] == 0 or norms[tgt] == 0 else norms[tgt]/norms[src]
            valid = scale is not None and math.isfinite(scale) and scale <= 100
            require(bool(a['valid'][view, action]) == valid, 'matched validity rule')
            if valid:
                require(abs(a['scales'][view, action]-scale) <= 1e-10+1e-10*scale, 'matching scale')
                planned = (data[src].astype(np.float64)*scale).astype(np.float32)
                require(np.array_equal(a['planned_data'][view, action], planned)
                        and np.array_equal(full[action], decay+planned), 'rounded matched proposal')
                realized = float(np.linalg.norm((full[action]-decay).astype(np.float64)))
                require(abs(realized-norms[tgt]) <= 1e-7+5e-5*norms[tgt], 'materialized matched norm')
                match_errors.append(realized-norms[tgt])
            else:
                for name in ('planned_data', 'after_parameters', 'after_logits'):
                    require(np.count_nonzero(a[name][view, action]) == 0, 'invalid zero fill: '+name)
                require(a['scales'][view, action] == 0, 'invalid scale filler')
                match_errors.append(None)
        norm_rows.append({'view': view, 'raw_data_norm': norms[0], 'native_data_norm': norms[1],
                          'native_over_raw': None if norms[0] == 0 else norms[1]/norms[0],
                          'match_errors': match_errors, 'updated_basis_gram_error': gram_error})
        decay_metrics = [[common.classification(a['after_logits'][view, 4, s, o], labels)
                          for o in range(2)] for s in range(2)]
        for action, name in enumerate(ACTIONS):
            available = bool(a['valid'][view, action])
            if available:
                small = theta + np.float32(.1)*(full[action]-theta)
                require(np.array_equal(a['after_parameters'][view, action, 1], small), 'fractional total-displacement path')
            for s, fraction in enumerate(FRACTIONS):
                after = a['after_parameters'][view, action, s].astype(np.float64)
                for o, objective in enumerate(('original', 'translate')):
                    entry = {'view': view, 'action': name, 'fraction': fraction, 'objective': objective, 'available': available}
                    fields = ('ce', 'accuracy', 'ce_improvement', 'accuracy_change', 'linear_total', 'linear_data', 'ce_data_improvement')
                    if available:
                        metric = common.classification(a['after_logits'][view, action, s, o], labels)
                        g = a['objective_grads'][o].astype(np.float64)
                        entry.update(ce=metric['ce'], accuracy=metric['accuracy'],
                                     ce_improvement=baseline[o]['ce']-metric['ce'],
                                     accuracy_change=metric['accuracy']-baseline[o]['accuracy'],
                                     linear_total=-float(g @ (after-theta.astype(np.float64))),
                                     linear_data=-float(g @ (after-a['after_parameters'][view, 4, s].astype(np.float64))),
                                     ce_data_improvement=decay_metrics[s][o]['ce']-metric['ce'])
                    else:
                        entry.update({field: None for field in fields})
                    rows.append(entry)
    return {'geometry': geo, 'baseline': dict(zip(('original', 'translate'), baseline)),
            'vector_checks': errors, 'norms': norm_rows, 'readouts': rows, 'summary': summarize_parent(rows)}


def seed_summary(parents):
    output = []
    for warmup in WARMUPS:
        group = sorted([p for p in parents if p['warmup'] == warmup], key=lambda p: p['seed'])
        require([p['seed'] for p in group] == list(SEEDS), 'three seed parents')
        for index, row in enumerate(group[0]['summary']):
            summaries = [p['summary'][index] for p in group]
            result = {k: row[k] for k in ('input_mode', 'fraction', 'objective', 'required_draws')}
            result.update(warmup=warmup, seeds=list(SEEDS), actions={}, contrasts={})
            for action in ACTIONS:
                result['actions'][action] = {}
                for field in ('ce_improvement', 'accuracy_change', 'linear_total', 'linear_data', 'ce_data_improvement'):
                    vals = [s['actions'][action][field] for s in summaries]
                    result['actions'][action][field] = {'values': vals, 'mean': strict_mean(vals)}
            for name, fields in row['contrasts'].items():
                result['contrasts'][name] = {}
                for field in fields:
                    vals = [s['contrasts'][name][field]['mean'] for s in summaries]
                    result['contrasts'][name][field] = {'values': vals, 'mean': strict_mean(vals)}
            output.append(result)
    return output


def audit_saved(directory, raw_labels, check=lambda: None):
    directory = Path(directory)
    require(directory.is_dir() and not directory.is_symlink(), 'ordinary input directory')
    path = directory/'results.json'
    require(path.is_file() and not path.is_symlink() and path.stat().st_size <= common.RESULTS_CAP, 'result admission')
    payload = path.read_bytes()
    result = common.strict_json(payload)
    require(result['schema'] == SCHEMA and result['status'] == 'complete', 'completed acquisition required')
    require(result['data_pins'] == common.DATA_PINS, 'data pins')
    for name, expected in result['source_pins'].items():
        require(type(name) is str and not Path(name).is_absolute() and '..' not in Path(name).parts
                and common.is_hash(expected), 'source pin schema')
        target = ROOT/name
        require(target.is_file() and not target.is_symlink() and target.resolve().is_relative_to(ROOT)
                and sha(target.read_bytes()) == expected, 'source pin changed: '+name)
    registry = common.strict_json((DOCS/'parent-pins.json').read_bytes())
    require(result['parent_pins'] == registry, 'frozen parent registry')
    source = Path(registry['input_dir'])
    old_docs = ROOT/'output/2026-09-10-spectral-general-augmentation'
    for target, key in ((source/'results.json', 'source_results_sha256'),
                        (old_docs/'audit.json', 'source_audit_sha256'),
                        (old_docs/'attempt.json', 'source_attempt_sha256')):
        require(target.is_file() and not target.is_symlink() and target.stat().st_size <= common.RESULTS_CAP
                and sha(target.read_bytes()) == registry[key], 'source provenance hash')
    parent_table = {r['path']: r for r in registry['receipts']}
    expected_parents = {f'plan-s{s}.npz' for s in SEEDS} | {f'warmup-s{s}-native32-{w}.pt' for s in SEEDS for w in WARMUPS}
    require(set(parent_table) == expected_parents and len(registry['receipts']) == 9,
            'exact source parent inventory')
    for name, receipt in parent_table.items():
        common.direct_name(name)
        target = source/name
        require(target.is_file() and not target.is_symlink() and target.stat().st_size == receipt['size_bytes']
                and 0 < receipt['size_bytes'] <= common.RESULTS_CAP and common.is_hash(receipt['sha256']),
                'ordinary bounded source parent')
    parent_reader = common.Reader(source, parent_table, check)
    for name in parent_table:
        parent_reader.load(name)  # Hash only. Never unpickle/checkpoint-load.
    roster = [{'seed': s, 'warmup': w, 'name': f's{s}-native32-{w}'} for s in SEEDS for w in WARMUPS]
    require(result['roster'] == roster and len(result['parents']) == 6, 'fixed six-parent roster')
    table = common.validate_inventory(directory, result['receipts'])
    expected_names = {'provenance.json'} | {f'plan-s{s}.npz' for s in SEEDS} | {'parent-'+r['name']+'.npz' for r in roster}
    require(set(table) == expected_names and sum(r['size_bytes'] for r in table.values())+len(payload) <= common.INPUT_CAP, 'bounded exact inventory')
    reader = common.Reader(directory, table, check)
    provenance = common.strict_json(reader.load('provenance.json', retain=True))
    attempt = common.strict_json((DOCS/'attempt.json').read_bytes())
    require(all(provenance[k] == v for k, v in attempt.items()) and provenance['source_pins'] == result['source_pins']
            and provenance['parents'] == registry and attempt['output_dir'] == str(directory), 'attempt/provenance binding')
    plans = {}
    for seed in SEEDS:
        expected = diagnostic_plan(raw_labels, seed)
        plan = reader.npz(f'plan-s{seed}.npz', {k: (a.dtype, a.shape) for k, a in expected.items()})
        require(all(np.array_equal(plan[k], v) for k, v in expected.items()), 'independent diagnostic plan')
        plans[seed] = plan
    parents = []
    for entry, identity in zip(result['parents'], roster):
        check()
        require(all(entry[k] == v for k, v in identity.items()), 'ordered parent identity')
        name, seed = entry['name'], entry['seed']
        for key, expected_name, inventory in (
                ('arrays_receipt', 'parent-'+name+'.npz', table),
                ('plan_receipt', f'plan-s{seed}.npz', table),
                ('source_plan_receipt', f'plan-s{seed}.npz', parent_table),
                ('parent_receipt', 'warmup-'+name+'.pt', parent_table)):
            require(entry[key] == inventory[expected_name], 'embedded receipt binding')
        expected_hash = entry['parent_receipt']['sha256']
        require(entry['parent_hash_before'] == entry['parent_hash_after'] == expected_hash
                and entry['parent_unchanged'] is True and common.is_hash(entry['parent_tree_sha256'])
                and entry['parent_tree_sha256'] == entry['restored_tree_sha256'], 'producer immutability assertions')
        arrays = reader.npz(entry['arrays_receipt']['path'], specs())
        report = audit_parent(arrays, plans[seed]['eval_labels'], check=check)
        parents.append({**identity, **report})
        del arrays
    require(reader.read == set(table), 'all artifacts checked')
    return {'schema': AUDIT_SCHEMA, 'status': 'PASS', 'source_results_sha256': sha(payload),
            'source_pins': result['source_pins'], 'artifact_count': len(table),
            'artifact_bytes': reader.bytes, 'parent_bytes_hashed': parent_reader.bytes,
            'parents': parents, 'seed_summary': seed_summary(parents),
            'limitations': ['Six fixed clean warmup states, not six independent training seeds.',
                'Three descriptive seed values; translated four-view means are not extra replicates.',
                'Empirical within-view variation is not population noise or semantic memorization.',
                'Different warmups change model, Adam and observer together.',
                'Autograd, logits, restored-state assertions and SVD history are not independently replayed.',
                'Local finite differences do not identify the full-trajectory causal mechanism.']}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--output-json', type=Path, required=True)
    args = parser.parse_args(argv)
    for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        require(os.environ.get(key) == '1', 'set '+key+'=1')
    source = args.input_dir.resolve(strict=True)
    output = args.output_json.parent.resolve(strict=True)/args.output_json.name
    require(not output.is_relative_to(source) and not output.exists() and not output.is_symlink(), 'exclusive audit outside source')
    started = time.monotonic()
    def check():
        require(time.monotonic()-started < 300, 'audit deadline')
        require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024 < 2*1024**3, 'audit 2GiB memory cap')
    with output.open('xb') as handle:
        try:
            report = audit_saved(source, common.read_fixed_labels(), check)
            check()
            payload = (json.dumps(report, allow_nan=False, separators=(',', ':'))+'\n').encode()
            require(len(payload) <= OUTPUT_CAP, '32MiB audit output cap')
            handle.write(payload)
        except BaseException as exc:
            handle.write((json.dumps({'schema': AUDIT_SCHEMA, 'status': 'FAIL', 'type': type(exc).__name__, 'message': str(exc)})+'\n').encode())
            raise
    print(json.dumps({'status': 'PASS', 'path': str(output), 'size_bytes': len(payload), 'sha256': sha(payload)}))


if __name__ == '__main__':
    main()
