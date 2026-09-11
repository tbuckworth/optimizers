#!/usr/bin/env python3
"""NumPy-only independent audit of fixed wrong-label augmentation artifacts.

No producer/data helper imports, torch, model replay, or training. Reuses only
the earlier independent auditor's bounded I/O and float64 metric primitives.
Corruption, metric grouping and contrasts are reconstructed independently here.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import resource
import struct
import time

import numpy as np

if __package__:
    from . import spectral_general_augmentation_audit as common
else:
    import spectral_general_augmentation_audit as common

require, AuditError = common.require, common.AuditError
SEEDS = (202609151, 202609152, 202609153)
POLICIES, AUGMENTATIONS = ('raw', 'native32'), ('none', 'translate')
EVAL_STEPS = common.EVAL_STEPS
SCHEMA = 'spectral_wrong_label_augmentation_v1'
AUDIT_SCHEMA = 'spectral_wrong_label_augmentation_audit_v1'
MEASUREMENTS = ('train_clean', 'train_assigned', 'wrong_true', 'wrong_target', 'heldout')


@dataclass(frozen=True)
class Dimensions:
    """Size overrides are fabricated-test-only; never exposed in the CLI."""
    per_class: int = 500
    wrong_per_class: int = 400
    updates: int = 4000
    batch: int = 64
    eval_steps: tuple = EVAL_STEPS

    @property
    def examples(self):
        return 10 * self.per_class


def regenerate_plan(raw_labels, seed, dimensions=Dimensions()):
    # Independent base split/occurrence/shift reconstruction, not producer code.
    base_dimensions = common.Dimensions(dimensions.per_class, dimensions.updates,
                                        dimensions.batch, dimensions.eval_steps)
    plan = common.regenerate_plan(raw_labels, seed, base_dimensions)
    require(0 < dimensions.wrong_per_class <= dimensions.per_class, 'corruption count bounds')
    true = plan['train_labels']
    selection = np.random.Generator(np.random.PCG64(np.random.SeedSequence([3, seed])))
    mask = np.zeros(len(true), dtype=bool)
    for digit in range(10):
        positions = np.flatnonzero(true == digit)
        selection.shuffle(positions)
        mask[positions[:dimensions.wrong_per_class]] = True
    replacement = np.random.Generator(np.random.PCG64(np.random.SeedSequence([4, seed])))
    offsets = replacement.integers(1, 10, size=len(true), dtype=np.int64)
    assigned = true.copy()
    assigned[mask] = (true[mask] + offsets[mask]) % 10
    return {**plan, 'assigned_labels': assigned, 'corruption_mask': mask}


def validate_plan(plan, labels, seed, dimensions=Dimensions()):
    expected = regenerate_plan(labels, seed, dimensions)
    require(set(plan) == set(expected), 'exact plan key inventory')
    for key, value in expected.items():
        common.checked_array(plan[key], value.dtype, value.shape, key)
        require(np.array_equal(plan[key], value), 'independent plan differs: ' + key)
    true, assigned, mask = plan['train_labels'], plan['assigned_labels'], plan['corruption_mask']
    require(np.array_equal(assigned != true, mask), 'mask must mark exactly every changed label')
    require(np.all(assigned[mask] != true[mask]) and np.array_equal(assigned[~mask], true[~mask]),
            'guaranteed-wrong/unchanged label contract')
    require(np.all((assigned >= 0) & (assigned < 10)) and int(mask.sum()) == 10 * dimensions.wrong_per_class,
            'wrong-label count/domain')
    require(np.array_equal(np.bincount(true[mask], minlength=10), np.full(10, dimensions.wrong_per_class)),
            'wrong-label class allocation')
    require(np.intersect1d(plan['train_ids'], plan['eval_ids']).size == 0, 'split overlap')


def evaluation_row(step, train, heldout, plan):
    true, assigned, mask = plan['train_labels'], plan['assigned_labels'], plan['corruption_mask']
    return {'step': step, 'train_clean': common.classification(train, true),
            'train_assigned': common.classification(train, assigned),
            'wrong_true': common.classification(train[mask], true[mask]),
            'wrong_target': common.classification(train[mask], assigned[mask]),
            'heldout': common.classification(heldout, plan['eval_labels'])}


def summarize(branches, endpoint_step=4000):
    indexed = {(b['seed'], b['policy'], b['augmentation']): b for b in branches}
    expected = {(s, p, a) for s in SEEDS for p in POLICIES for a in AUGMENTATIONS}
    require(len(branches) == len(indexed) == 12 and set(indexed) == expected, 'summary roster')
    endpoints, warmups = {}, {}
    for key, branch in indexed.items():
        require(branch['metrics'][-1]['step'] == endpoint_step, 'fixed endpoint required')
        rows = [row for row in branch['metrics'] if row['step'] == 100]
        require(len(rows) == 1, 'unique warmup checkpoint required')
        endpoints[key], warmups[key] = branch['metrics'][-1], rows[0]
    def group_table(source):
        return {measurement: {p + '/' + a: {metric: common.estimate([
            source[s, p, a][measurement][metric] for s in SEEDS]) for metric in ('accuracy', 'ce')}
            for p in POLICIES for a in AUGMENTATIONS} for measurement in MEASUREMENTS}
    endpoint_table, warmup_table = group_table(endpoints), group_table(warmups)
    benefits = {p: {metric: common.estimate([sign * (endpoints[s, p, 'translate']['heldout'][metric]
        - endpoints[s, p, 'none']['heldout'][metric]) for s in SEEDS])
        for metric, sign in (('accuracy', 1), ('ce', -1))} for p in POLICIES}
    interaction = {metric: common.estimate([n - r for n, r in zip(
        benefits['native32'][metric]['values'], benefits['raw'][metric]['values'])]) for metric in ('accuracy', 'ce')}
    changes = {measurement: {p + '/' + a: {metric: common.estimate([
        endpoints[s, p, a][measurement][metric] - warmups[s, p, a][measurement][metric]
        for s in SEEDS]) for metric in ('accuracy', 'ce')}
        for p in POLICIES for a in AUGMENTATIONS} for measurement in MEASUREMENTS}
    return {'seeds': list(SEEDS), 'endpoint_step': endpoint_step, 'groups': endpoint_table['heldout'],
        'endpoint_by_measurement': endpoint_table, 'warmup_by_measurement': warmup_table,
        'change_from_warmup': changes, 'augmentation_benefits': benefits, 'interaction': interaction,
        'conventions': {'heldout_accuracy_benefit': 'translate minus none', 'heldout_ce_benefit': 'none minus translate',
            'interaction': 'native32 benefit minus raw benefit', 'change_from_warmup': 'endpoint minus warmup for BOTH metrics',
            'wrong_target': 'fit to assigned wrong targets, not beneficial competence',
            'wrong_true': 'fit to original true labels on exactly the same corrupted subset'},
        'independent_units': 'three paired training seeds; twelve branches are not independent replicates',
        'uncertainty': 'descriptive seed values and means; no significance, confidence interval or best epoch'}


def audit_saved(input_dir, raw_labels, *, dimensions=Dimensions(), source_root=common.ROOT, check=lambda: None):
    started = time.monotonic()
    directory = Path(input_dir)
    require(directory.is_dir() and not directory.is_symlink(), 'ordinary input directory')
    result_path = directory / 'results.json'
    require(result_path.is_file() and not result_path.is_symlink()
            and result_path.stat().st_size <= common.RESULTS_CAP, 'results admission')
    payload = result_path.read_bytes()
    result = common.strict_json(payload)
    require(result['schema'] == SCHEMA and result['status'] == 'complete', 'completed matching source schema')
    require(result['eval_steps'] == list(dimensions.eval_steps) and result['data_pins'] == common.DATA_PINS,
            'fixed evaluation/data identity differs')
    pins = result['source_pins']
    require(type(pins) is dict and pins, 'source pins required')
    for name, expected in pins.items():
        require(type(name) is str and not Path(name).is_absolute() and '..' not in Path(name).parts
                and common.is_hash(expected), 'source pin schema')
        path = Path(source_root) / name
        require(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(Path(source_root).resolve()),
                'source path outside root or missing')
        require(common.sha(path.read_bytes()) == expected, 'source bytes changed: ' + name)
    roster, branches = result['roster'], result['branches']
    expected = {(s, p, a) for s in SEEDS for p in POLICIES for a in AUGMENTATIONS}
    require(type(roster) is list and len(roster) == 12 and all(type(r) is dict
            and set(r) == {'seed', 'policy', 'augmentation'} for r in roster), 'roster schema')
    require({(r['seed'], r['policy'], r['augmentation']) for r in roster} == expected, 'exact branch roster')
    require(type(branches) is list and len(branches) == 12 and
            [(b['seed'], b['policy'], b['augmentation']) for b in branches] ==
            [(r['seed'], r['policy'], r['augmentation']) for r in roster], 'ordered branch roster')
    table = common.validate_inventory(directory, result['receipts'])
    require(sum(r['size_bytes'] for r in table.values()) + len(payload) <= common.INPUT_CAP, 'total input cap')
    common.check_embedded_receipts(branches, table)
    reader = common.Reader(directory, table, check)
    n, updates, batch, neval = dimensions.examples, dimensions.updates, dimensions.batch, len(dimensions.eval_steps)
    plans = {}
    for seed in SEEDS:
        require(f'initial-s{seed}.pt' in table, 'missing initial receipt')
        plan = reader.npz(f'plan-s{seed}.npz', {
            'train_ids': (np.int64, (n,)), 'eval_ids': (np.int64, (n,)),
            'train_labels': (np.int64, (n,)), 'eval_labels': (np.int64, (n,)),
            'assigned_labels': (np.int64, (n,)), 'corruption_mask': (np.bool_, (n,)),
            'occurrences': (np.int64, (updates, batch)), 'shifts': (np.int8, (updates, batch, 2))})
        validate_plan(plan, raw_labels, seed, dimensions)
        plans[seed] = plan
    initial_hash, warmup_hash, initial_logits, warmup_logits = {}, {}, {}, {}
    verified, max_error = [], 0.
    for branch in branches:
        check()
        seed, policy, augmentation = (branch[k] for k in ('seed', 'policy', 'augmentation'))
        name = f's{seed}-{policy}-{augmentation}'
        require(branch['name'] == name, 'branch name')
        for field, filename in (('logits_receipt', f'logits-{name}.npz'), ('stream_receipt', f'stream-{name}.npz'),
                                ('warmup_receipt', f'warmup-{name}.pt'), ('final_receipt', f'final-{name}.pt')):
            require(type(branch.get(field)) is dict and branch[field] == table.get(filename), 'branch receipt: ' + field)
        for key in ('initial_model_sha256', 'warmup_learning_sha256'):
            require(common.is_hash(branch[key]), 'state hash schema')
        require(seed not in initial_hash or initial_hash[seed] == branch['initial_model_sha256'], 'initial model pairing')
        initial_hash[seed] = branch['initial_model_sha256']
        pair = (seed, augmentation)
        require(pair not in warmup_hash or warmup_hash[pair] == branch['warmup_learning_sha256'], 'warmup learning pairing')
        warmup_hash[pair] = branch['warmup_learning_sha256']
        for key in ('training_seconds', 'augmentation_seconds', 'evaluation_seconds', 'wall_seconds'):
            require(common.finite_number(branch[key], key) >= 0, 'negative timing')
        logits = reader.npz(f'logits-{name}.npz', {'steps': (np.int64, (neval,)),
            'train': (np.float32, (neval, n, 10)), 'heldout': (np.float32, (neval, n, 10))})
        require(np.array_equal(logits['steps'], dimensions.eval_steps), 'logit steps differ')
        losses = reader.npz(f'stream-{name}.npz', {'loss': (np.float64, (updates,))})['loss']
        require((losses >= 0).all(), 'negative training loss')
        for split in ('train', 'heldout'):
            ikey, wkey = (seed, split), (seed, augmentation, split)
            require(ikey not in initial_logits or np.array_equal(initial_logits[ikey], logits[split][0]), 'initial logit pairing')
            require(wkey not in warmup_logits or np.array_equal(warmup_logits[wkey], logits[split][1]), 'warmup logit pairing')
            initial_logits[ikey], warmup_logits[wkey] = logits[split][0].copy(), logits[split][1].copy()
        require(type(branch['metrics']) is list and len(branch['metrics']) == neval, 'metric roster size')
        rebuilt = []
        for i, step in enumerate(dimensions.eval_steps):
            actual = branch['metrics'][i]
            require(type(actual) is dict and set(actual) == {'step', *MEASUREMENTS}
                    and type(actual['step']) is int and actual['step'] == step, 'metric row schema/step')
            row = evaluation_row(step, logits['train'][i], logits['heldout'][i], plans[seed])
            for key in MEASUREMENTS:
                max_error = max(max_error, common.compare_metrics(actual[key], row[key]))
            rebuilt.append(row)
        verified.append({'seed': seed, 'policy': policy, 'augmentation': augmentation, 'name': name, 'metrics': rebuilt})
    for name in table:
        if name not in reader.read:
            reader.load(name)  # Hash opaque checkpoint/metadata bytes; never deserialize a model.
    require(reader.read == set(table), 'artifact inventory not fully consumed')
    return {'schema': AUDIT_SCHEMA, 'status': 'PASS', 'input_dir': str(directory.resolve()),
        'input_results_sha256': common.sha(payload), 'source_pins': pins, 'data_pins': result['data_pins'],
        'verified_artifacts': len(reader.read), 'input_bytes_read_once': reader.bytes + len(payload),
        'trajectories': 12, 'logical_evaluation_records': 12 * neval,
        'metric_atol': common.ATOL, 'metric_rtol': common.RTOL, 'max_absolute_scalar_error': max_error,
        'corruption': {'count_per_seed': 10 * dimensions.wrong_per_class,
                       'per_true_class': dimensions.wrong_per_class, 'wrong_targets_exclude_true': True},
        'summary': summarize(verified, dimensions.eval_steps[-1]), 'branches': verified,
        'wall_seconds': time.monotonic() - started,
        'limits': ['No model, optimizer, observer, image transformation or training replay.',
            'State digests are producer assertions checked for paired identity; checkpoint bytes are hash-checked only.',
            'Plans and fixed corruption independently regenerated from supplied source labels; CLI checks both original IDX hashes.',
            'Labels are fixed per example in the saved plan; actual training consumption is not independently replayed.',
            'Assigned-wrong-target fit is not clean competence; three paired seeds are descriptive, not significance evidence.']}


def read_fixed_source_labels():
    labels = common.read_fixed_labels()
    path = common.LABELS_PATH.with_name('train-images-idx3-ubyte')
    digest, size = hashlib.sha256(), 0
    with path.open('rb') as handle:
        header = handle.read(16)
        require(len(header) == 16 and struct.unpack('>IIII', header) == (2051, 60000, 28, 28), 'image IDX header')
        digest.update(header)
        size += len(header)
        for chunk in iter(lambda: handle.read(1024**2), b''):
            digest.update(chunk)
            size += len(chunk)
    require(size == 16 + 60000 * 784 and digest.hexdigest() == common.DATA_PINS[path.name], 'image IDX identity')
    return labels


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--output-json', type=Path, required=True)
    args = parser.parse_args(argv)
    for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        require(os.environ.get(key) == '1', 'set ' + key + '=1 before Python')
    source = args.input_dir.resolve(strict=True)
    output = args.output_json.parent.resolve(strict=True) / args.output_json.name
    require(not output.is_relative_to(source) and not output.exists() and not output.is_symlink(), 'new output outside input')
    started = time.monotonic()
    def check():
        require(time.monotonic() - started < 300, 'five-minute audit deadline')
        require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 < 2 * 1024**3, '2 GiB CPU memory guard')
    with output.open('xb') as handle:
        try:
            report = audit_saved(source, read_fixed_source_labels(), check=check)
            check()
            payload = (json.dumps(report, allow_nan=False, separators=(',', ':')) + '\n').encode()
            require(len(payload) <= common.OUTPUT_CAP, 'audit output cap')
            handle.write(payload)
        except BaseException as exc:
            handle.write((json.dumps({'schema': AUDIT_SCHEMA, 'status': 'FAIL',
                'type': type(exc).__name__, 'message': str(exc)}, allow_nan=False) + '\n').encode())
            raise
    print(json.dumps({'status': 'PASS', 'path': str(output), 'size_bytes': len(payload), 'sha256': common.sha(payload)}))


if __name__ == '__main__':
    main()
