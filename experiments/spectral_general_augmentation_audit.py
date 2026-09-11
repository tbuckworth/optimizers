#!/usr/bin/env python3
"""Independent NumPy audit of saved general-augmentation evidence.

No torch, model/optimizer replay, training, downloads, or retries. Checkpoint
bytes are hashed only; state digests are producer assertions checked for
pairing. CLI reads the existing pinned training-label IDX, never image IDX.
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
import struct
import time
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LABELS_PATH = Path('data/MNIST/raw/train-labels-idx1-ubyte')
SEEDS = (202609141, 202609142, 202609143)
POLICIES, AUGMENTATIONS = ('raw', 'native32'), ('none', 'translate')
EVAL_STEPS = (0, 100) + tuple(range(200, 4001, 200))
SCHEMA = 'spectral_general_augmentation_v1'
AUDIT_SCHEMA = 'spectral_general_augmentation_audit_v1'
DATA_PINS = {
    'train-images-idx3-ubyte': 'ba891046e6505d7aadcbbe25680a0738ad16aec93bde7f9b65e87a2fc25776db',
    'train-labels-idx1-ubyte': '65a50cbbf4e906d70832878ad85ccda5333a97f0f4c3dd2ef09a8a9eef7101c5',
}
ATOL = RTOL = 1e-10
INPUT_CAP, NPZ_CAP, RESULTS_CAP, OUTPUT_CAP = 1024**3, 128 * 1024**2, 16 * 1024**2, 16 * 1024**2


class AuditError(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise AuditError(message)


@dataclass(frozen=True)
class Dimensions:
    """Overrides only support fabricated fixtures; none are exposed by CLI."""
    per_class: int = 500
    updates: int = 4000
    batch: int = 64
    eval_steps: tuple = EVAL_STEPS

    @property
    def examples(self):
        return 10 * self.per_class


def is_hash(value):
    return type(value) is str and re.fullmatch('[0-9a-f]{64}', value) is not None


def sha(payload):
    return hashlib.sha256(payload).hexdigest()


def finite_number(value, label):
    require(type(value) in (int, float) and math.isfinite(value), 'finite numeric ' + label)
    return float(value)


def checked_array(value, dtype, shape, label):
    require(isinstance(value, np.ndarray) and value.dtype == np.dtype(dtype)
            and value.shape == shape, 'array schema: ' + label)
    require(np.isfinite(value).all(), 'nonfinite array: ' + label)
    return value


def classification(logits, labels):
    require(isinstance(labels, np.ndarray) and labels.dtype == np.int64 and labels.ndim == 1
            and len(labels) > 0 and np.all((labels >= 0) & (labels < 10)), 'label schema/domain')
    checked_array(logits, np.float32, (len(labels), 10), 'logits')
    z = logits.astype(np.float64)
    shifted = z - z.max(axis=1, keepdims=True)
    losses = np.log(np.exp(shifted).sum(axis=1)) - shifted[np.arange(len(labels)), labels]
    total = math.fsum(float(value) for value in losses)
    correct = int(np.count_nonzero(z.argmax(axis=1) == labels))  # first index wins ties
    return {'count': len(labels), 'correct': correct, 'accuracy': correct / len(labels),
            'ce_sum': total, 'ce': total / len(labels)}


def compare_metrics(actual, expected):
    require(type(actual) is dict and set(actual) == set(expected), 'metric keys differ')
    errors = []
    for key, value in expected.items():
        if key in ('count', 'correct'):
            require(type(actual[key]) is int and actual[key] == value, 'integer metric differs: ' + key)
        else:
            observed = finite_number(actual[key], key)
            error = abs(observed - value)
            require(error <= ATOL + RTOL * abs(value), 'metric differs: ' + key)
            errors.append(error)
    return max(errors, default=0.)


def regenerate_plan(raw_labels, seed, dimensions=Dimensions()):
    """Independent stream implementation: entropy order is [stream_id,seed]."""
    require(isinstance(raw_labels, np.ndarray) and raw_labels.dtype == np.int64
            and raw_labels.ndim == 1 and np.all((raw_labels >= 0) & (raw_labels < 10)), 'raw labels')
    rngs = [np.random.Generator(np.random.PCG64(np.random.SeedSequence([stream, seed])))
            for stream in range(3)]
    train, evaluation = [], []
    for digit in range(10):
        pool = rngs[0].permutation(np.flatnonzero(raw_labels == digit))
        require(len(pool) >= 2 * dimensions.per_class, 'insufficient class population')
        train.extend(pool[:dimensions.per_class].tolist())
        evaluation.extend(pool[dimensions.per_class:2 * dimensions.per_class].tolist())
    train, evaluation = np.array(train, dtype=np.int64), np.array(evaluation, dtype=np.int64)
    return {'train_ids': train, 'eval_ids': evaluation, 'train_labels': raw_labels[train].copy(),
            'eval_labels': raw_labels[evaluation].copy(),
            'occurrences': rngs[1].integers(0, dimensions.examples,
                size=(dimensions.updates, dimensions.batch), dtype=np.int64),
            'shifts': rngs[2].integers(-2, 3,
                size=(dimensions.updates, dimensions.batch, 2), dtype=np.int8)}


def validate_plan(plan, raw_labels, seed, dimensions=Dimensions()):
    expected = regenerate_plan(raw_labels, seed, dimensions)
    require(set(plan) == set(expected), 'plan array inventory differs')
    for key, reference in expected.items():
        checked_array(plan[key], reference.dtype, reference.shape, key)
        require(np.array_equal(plan[key], reference), 'independent plan differs: ' + key)
    require(np.intersect1d(plan['train_ids'], plan['eval_ids']).size == 0, 'split overlap')
    for key in ('train_labels', 'eval_labels'):
        require(np.array_equal(np.bincount(plan[key], minlength=10), np.full(10, dimensions.per_class)),
                'unbalanced split')


def estimate(values):
    require(len(values) == 3 and all(math.isfinite(v) for v in values), 'three finite seed values required')
    return {'values': [float(v) for v in values], 'mean': math.fsum(values) / 3}


def summarize(branches, endpoint_step=4000):
    indexed = {(b['seed'], b['policy'], b['augmentation']): b for b in branches}
    expected = {(s, p, a) for s in SEEDS for p in POLICIES for a in AUGMENTATIONS}
    require(len(branches) == len(indexed) == 12 and set(indexed) == expected, 'summary branch roster')
    endpoints = {}
    for key, branch in indexed.items():
        row = branch['metrics'][-1]
        require(row['step'] == endpoint_step, 'wrong fixed endpoint')
        endpoints[key] = row['heldout']
    groups, benefits, interaction = {}, {}, {}
    for policy in POLICIES:
        for augmentation in AUGMENTATIONS:
            groups[policy + '/' + augmentation] = {metric: estimate([
                endpoints[seed, policy, augmentation][metric] for seed in SEEDS])
                for metric in ('accuracy', 'ce')}
        benefits[policy] = {}
        for metric, sign in (('accuracy', 1), ('ce', -1)):
            benefits[policy][metric] = estimate([sign * (endpoints[s, policy, 'translate'][metric]
                - endpoints[s, policy, 'none'][metric]) for s in SEEDS])
    for metric in ('accuracy', 'ce'):
        interaction[metric] = estimate([n - r for n, r in zip(
            benefits['native32'][metric]['values'], benefits['raw'][metric]['values'])])
    return {'seeds': list(SEEDS), 'endpoint_step': endpoint_step,
            'independent_units': 'three paired training seeds; branches are not independent replicates',
            'uncertainty': 'descriptive values and arithmetic means; no significance or epoch selection',
            'benefit_conventions': {'accuracy': 'translate minus none', 'ce': 'none minus translate',
                                    'interaction': 'native32 benefit minus raw benefit'},
            'groups': groups, 'augmentation_benefits': benefits, 'interaction': interaction}


def direct_name(name):
    require(type(name) is str and name not in ('', '.', '..') and Path(name).name == name
            and '/' not in name and '\\' not in name, 'direct-child artifact name required')
    return name


def strict_json(payload):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'duplicate JSON key')
            result[key] = value
        return result
    def reject(value):
        raise AuditError('nonfinite JSON constant: ' + value)
    def parse_float(value):
        result = float(value)
        require(math.isfinite(result), 'nonfinite JSON numeric value')
        return result
    return json.loads(payload, object_pairs_hook=pairs, parse_constant=reject, parse_float=parse_float)


def validate_inventory(directory, receipts):
    require(type(receipts) is list, 'receipt list required')
    table = {}
    total = 0
    for record in receipts:
        require(type(record) is dict and set(record) == {'path', 'size_bytes', 'sha256'}, 'receipt schema')
        name = direct_name(record['path'])
        require(name != 'results.json' and name not in table, 'duplicate or self receipt')
        require(type(record['size_bytes']) is int and record['size_bytes'] >= 0
                and is_hash(record['sha256']), 'receipt size/hash')
        table[name] = record
        total += record['size_bytes']
    require(total <= INPUT_CAP, 'input byte cap exceeded')
    children = list(directory.iterdir())
    require(all(p.is_file() and not p.is_symlink() for p in children), 'only ordinary direct files allowed')
    require({p.name for p in children} == set(table) | {'results.json'}, 'exact artifact inventory differs')
    for name, receipt in table.items():
        require((directory / name).stat().st_size == receipt['size_bytes'], 'artifact size differs: ' + name)
    return table


class Reader:
    """Every receipted artifact is read once, including opaque checkpoint bytes."""
    def __init__(self, directory, receipts, check):
        self.directory, self.receipts, self.check = directory, receipts, check
        self.read = set()
        self.bytes = 0

    def load(self, name, retain=False):
        require(name in self.receipts and name not in self.read, 'missing or repeated artifact read: ' + name)
        expected = self.receipts[name]
        require(not retain or expected['size_bytes'] <= NPZ_CAP, 'retained payload too large')
        digest, count, chunks = hashlib.sha256(), 0, []
        with (self.directory / name).open('rb') as handle:
            for chunk in iter(lambda: handle.read(1024**2), b''):
                self.check()
                digest.update(chunk)
                count += len(chunk)
                require(count <= expected['size_bytes'], 'artifact grew while reading')
                if retain:
                    chunks.append(chunk)
        require(count == expected['size_bytes'] and digest.hexdigest() == expected['sha256'],
                'artifact hash/size differs: ' + name)
        self.read.add(name)
        self.bytes += count
        return b''.join(chunks) if retain else None

    def npz(self, name, specs):
        payload = self.load(name, retain=True)
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            members = archive.infolist()
            require(len(members) == len(specs) and {m.filename for m in members} == {k + '.npy' for k in specs},
                    'NPZ member inventory differs')
            require(sum(m.file_size for m in members) <= NPZ_CAP, 'expanded archive cap')
            # Validate public .npy headers before NumPy can allocate array shapes.
            for member in members:
                with archive.open(member) as handle:
                    version = np.lib.format.read_magic(handle)
                    require(version in ((1, 0), (2, 0)), 'unsupported array format')
                    reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
                    shape, fortran, dtype = reader(handle)
                    expected_dtype, expected_shape = specs[member.filename[:-4]]
                    require(dtype == np.dtype(expected_dtype) and shape == expected_shape and not fortran,
                            'array header differs: ' + member.filename)
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            result = {key: checked_array(archive[key], dtype, shape, key)
                      for key, (dtype, shape) in specs.items()}
        return result


def check_embedded_receipts(value, table):
    if type(value) is dict:
        if set(value) == {'path', 'size_bytes', 'sha256'}:
            require(value.get('path') in table and table[value['path']] == value, 'embedded receipt differs')
        else:
            for item in value.values():
                check_embedded_receipts(item, table)
    elif type(value) is list:
        for item in value:
            check_embedded_receipts(item, table)


def audit_saved(input_dir, raw_labels, *, dimensions=Dimensions(), source_root=ROOT, check=lambda: None):
    started = time.monotonic()
    directory = Path(input_dir)
    require(directory.is_dir() and not directory.is_symlink(), 'ordinary input directory required')
    result_path = directory / 'results.json'
    require(result_path.is_file() and not result_path.is_symlink()
            and result_path.stat().st_size <= RESULTS_CAP, 'results file admission')
    payload = result_path.read_bytes()
    results = strict_json(payload)
    require(results['schema'] == SCHEMA and results['status'] == 'complete', 'incomplete or wrong source schema')
    require(results['eval_steps'] == list(dimensions.eval_steps), 'fixed evaluation steps differ')
    require(results['data_pins'] == DATA_PINS, 'original training data pins differ')
    pins = results['source_pins']
    require(type(pins) is dict and len(pins) > 0, 'source pins required')
    for name, expected in pins.items():
        require(type(name) is str and not Path(name).is_absolute() and '..' not in Path(name).parts
                and is_hash(expected), 'invalid source pin')
        path = Path(source_root) / name
        require(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(Path(source_root).resolve()),
                'source path outside root or missing')
        require(sha(path.read_bytes()) == expected, 'source bytes changed: ' + name)
    roster = results['roster']
    expected = {(s, p, a) for s in SEEDS for p in POLICIES for a in AUGMENTATIONS}
    require(type(roster) is list and len(roster) == 12
            and all(type(r) is dict and set(r) == {'seed', 'policy', 'augmentation'} for r in roster), 'roster schema')
    require({(r['seed'], r['policy'], r['augmentation']) for r in roster} == expected, 'exact roster differs')
    branches = results['branches']
    require(type(branches) is list and len(branches) == 12
            and [(b['seed'], b['policy'], b['augmentation']) for b in branches]
            == [(r['seed'], r['policy'], r['augmentation']) for r in roster], 'ordered branch roster differs')
    table = validate_inventory(directory, results['receipts'])
    require(sum(r['size_bytes'] for r in table.values()) + len(payload) <= INPUT_CAP, 'total input byte cap')
    reader = Reader(directory, table, check)
    check_embedded_receipts(branches, table)
    plans = {}
    n, updates, batch, neval = dimensions.examples, dimensions.updates, dimensions.batch, len(dimensions.eval_steps)
    for seed in SEEDS:
        require(f'initial-s{seed}.pt' in table, 'missing initial checkpoint receipt')
        name = f'plan-s{seed}.npz'
        plan = reader.npz(name, {
            'train_ids': (np.int64, (n,)), 'eval_ids': (np.int64, (n,)),
            'train_labels': (np.int64, (n,)), 'eval_labels': (np.int64, (n,)),
            'occurrences': (np.int64, (updates, batch)), 'shifts': (np.int8, (updates, batch, 2))})
        validate_plan(plan, raw_labels, seed, dimensions)
        plans[seed] = plan
    initial_hashes, warmup_hashes, initial_logits, warmup_logits = {}, {}, {}, {}
    verified, max_error = [], 0.
    for branch in branches:
        check()
        seed, policy, augmentation = (branch[k] for k in ('seed', 'policy', 'augmentation'))
        name = f's{seed}-{policy}-{augmentation}'
        require(branch['name'] == name, 'branch name differs')
        for field in ('logits_receipt', 'stream_receipt', 'warmup_receipt', 'final_receipt'):
            record = branch.get(field)
            require(type(record) is dict and set(record) == {'path', 'size_bytes', 'sha256'}
                    and record.get('path') in table and table[record['path']] == record,
                    'required branch receipt differs: ' + field)
        require(branch['logits_receipt']['path'] == f'logits-{name}.npz'
                and branch['stream_receipt']['path'] == f'stream-{name}.npz', 'branch artifact path differs')
        require(branch['warmup_receipt']['path'] == f'warmup-{name}.pt'
                and branch['final_receipt']['path'] == f'final-{name}.pt', 'checkpoint receipt path differs')
        for field in ('initial_model_sha256', 'warmup_learning_sha256'):
            require(is_hash(branch[field]), 'state hash format')
        require(seed not in initial_hashes or initial_hashes[seed] == branch['initial_model_sha256'], 'initial model pairing differs')
        initial_hashes[seed] = branch['initial_model_sha256']
        pair = (seed, augmentation)
        require(pair not in warmup_hashes or warmup_hashes[pair] == branch['warmup_learning_sha256'], 'warmup learning pairing differs')
        warmup_hashes[pair] = branch['warmup_learning_sha256']
        for key in ('training_seconds', 'augmentation_seconds', 'evaluation_seconds', 'wall_seconds'):
            require(finite_number(branch[key], key) >= 0, 'negative timing')
        logits = reader.npz(f'logits-{name}.npz', {'steps': (np.int64, (neval,)),
            'train': (np.float32, (neval, n, 10)), 'heldout': (np.float32, (neval, n, 10))})
        require(np.array_equal(logits['steps'], dimensions.eval_steps), 'logit steps differ')
        stream = reader.npz(f'stream-{name}.npz', {'loss': (np.float64, (updates,))})
        # Float32 cross-entropy can round to exact zero for a confident correct
        # batch. Zero is valid; negative or nonfinite values are not.
        require((stream['loss'] >= 0).all(), 'training stream losses must be nonnegative')
        for split in ('train', 'heldout'):
            initial_key, warmup_key = (seed, split), (seed, augmentation, split)
            require(initial_key not in initial_logits or np.array_equal(initial_logits[initial_key], logits[split][0]),
                    'initial logits differ across policies/augmentations')
            require(warmup_key not in warmup_logits or np.array_equal(warmup_logits[warmup_key], logits[split][1]),
                    'warmup logits differ across policies')
            initial_logits[initial_key] = logits[split][0].copy()
            warmup_logits[warmup_key] = logits[split][1].copy()
        metrics = branch['metrics']
        require(type(metrics) is list and len(metrics) == neval, 'metric roster size')
        rebuilt = []
        for index, step in enumerate(dimensions.eval_steps):
            actual = metrics[index]
            require(type(actual) is dict and set(actual) == {'step', 'train', 'heldout'}
                    and type(actual['step']) is int and actual['step'] == step, 'metric step/schema differs')
            row = {'step': step}
            for split, label_key in (('train', 'train_labels'), ('heldout', 'eval_labels')):
                row[split] = classification(logits[split][index], plans[seed][label_key])
                max_error = max(max_error, compare_metrics(actual[split], row[split]))
            rebuilt.append(row)
        verified.append({'name': name, 'seed': seed, 'policy': policy, 'augmentation': augmentation, 'metrics': rebuilt})
    for name in table:
        if name not in reader.read:
            reader.load(name)  # Opaque artifacts, including snapshots: hash only.
    require(reader.read == set(table), 'artifact read roster incomplete')
    return {'schema': AUDIT_SCHEMA, 'status': 'PASS', 'input_results_sha256': sha(payload),
        'input_dir': str(directory.resolve()), 'source_pins': pins, 'data_pins': results['data_pins'],
        'verified_artifacts': len(reader.read), 'input_bytes_read_once': reader.bytes + len(payload),
        'trajectories': 12, 'logical_evaluation_records': 12 * neval,
        'metric_atol': ATOL, 'metric_rtol': RTOL, 'max_absolute_scalar_error': max_error,
        'summary': summarize(verified, dimensions.eval_steps[-1]), 'branches': verified,
        'wall_seconds': time.monotonic() - started,
        'limits': ['No model, optimizer, observer or training replay.',
                   'State digests are producer assertions checked for paired identity; checkpoint bytes are hash-checked only.',
                   'Existing pinned training-label IDX verifies source IDs/labels; image IDX bytes are not reread.',
                   'Saved shifts and occurrences independently regenerated; their actual use during training is not replayed.',
                   'Three paired seeds; descriptive endpoint comparison, no significance or best-epoch selection.']}


def read_fixed_labels():
    payload = LABELS_PATH.read_bytes()
    require(len(payload) == 60008 and sha(payload) == DATA_PINS[LABELS_PATH.name]
            and struct.unpack('>II', payload[:8]) == (2049, 60000), 'training-label IDX identity differs')
    return np.frombuffer(payload, dtype=np.uint8, offset=8).astype(np.int64)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--output-json', type=Path, required=True)
    args = parser.parse_args(argv)
    for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        require(os.environ.get(key) == '1', 'set ' + key + '=1 before Python')
    source = args.input_dir.resolve(strict=True)
    output = args.output_json.parent.resolve(strict=True) / args.output_json.name
    require(not output.is_relative_to(source) and not output.exists() and not output.is_symlink(),
            'new output outside input directory required')
    started = time.monotonic()
    def check():
        require(time.monotonic() - started < 300, 'five-minute audit deadline')
        require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 < 2 * 1024**3, '2 GiB CPU memory guard')
    # Reserve output once, before reading scientific inputs. Failure stays explicit.
    with output.open('xb') as handle:
        try:
            report = audit_saved(source, read_fixed_labels(), check=check)
            check()
            payload = (json.dumps(report, allow_nan=False, separators=(',', ':')) + '\n').encode()
            require(len(payload) <= OUTPUT_CAP, 'audit output cap')
            handle.write(payload)
        except BaseException as exc:
            failure = {'schema': AUDIT_SCHEMA, 'status': 'FAIL', 'type': type(exc).__name__, 'message': str(exc)}
            handle.write((json.dumps(failure, allow_nan=False) + '\n').encode())
            raise
    print(json.dumps({'status': 'PASS', 'path': str(output), 'size_bytes': len(payload), 'sha256': sha(payload)}))


if __name__ == '__main__':
    main()
