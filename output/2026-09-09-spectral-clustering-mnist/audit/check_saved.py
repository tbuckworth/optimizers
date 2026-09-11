"""Independent saved-evidence arithmetic; no producer imports or model execution."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import statistics
import struct
import subprocess
import time

import numpy as np

WORKTREE = Path('/tmp/spectral-experiment-artifacts/spectral-clustering-pilot-20260909.j2jkfp/worktree')
PARENT = Path('/tmp/spectral-experiment-artifacts/spectral-clustering-mnist-20260909.51ggnu')
SOURCE, OUTPUT = PARENT / 'acquisition-001', PARENT / 'audit-001'
SEEDS = (202609091, 202609092, 202609093)
CONDITIONS = ('clean', 'fixed_uniform_0p9')
ARMS = ('adamw', 'hard32', 'cluster32', 'mixed32')
EVAL_STEPS = tuple(range(0, 2001, 100))
METRICS = ('heldout_clean_accuracy', 'heldout_clean_ce', 'train_clean_accuracy',
           'train_fixed_accuracy', 'train_corrupted_accuracy', 'train_clean_ce', 'train_fixed_ce')
FILTER_SHA = '9280c7d360aef4c775baf0d781a5afd56e016e9cfb93ef829c3ae4b8cc3df943'
GRAPH_SHA = '2581010abe529d1b3da2af0a71106710dcc59b81ec03e58b93b56f2f4790f89e'
DATA_SHA = {'train-images-idx3-ubyte': 'ba891046e6505d7aadcbbe25680a0738ad16aec93bde7f9b65e87a2fc25776db',
            'train-labels-idx1-ubyte': '65a50cbbf4e906d70832878ad85ccda5333a97f0f4c3dd2ef09a8a9eef7101c5'}


def require(condition, label):
    if not condition:
        raise ValueError(label)


def finite_tree(value, path="root"):
    if isinstance(value, dict):
        for key, item in value.items():
            finite_tree(item, path + "." + str(key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            finite_tree(item, path + "." + str(index))
    elif isinstance(value, float):
        require(math.isfinite(value), "nonfinite " + path)


def unique_object(pairs):
    out = {}
    for key, value in pairs:
        require(key not in out, "duplicate JSON key " + key)
        out[key] = value
    return out


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    with Path(path).open() as stream:
        result = json.load(stream, object_pairs_hook=unique_object)
    finite_tree(result)
    return result


def close(actual, expected, label, rtol=1e-10, atol=1e-10):
    require(isinstance(actual, (float, int)) and not isinstance(actual, bool), label + " numeric")
    require(math.isfinite(actual) and math.isfinite(expected), label + " finite")
    require(math.isclose(actual, expected, rel_tol=rtol, abs_tol=atol),
            f"{label}: saved {actual!r}, recomputed {expected!r}")


def logits_stats(logits, labels):
    """Independent float64 log-sum-exp from saved logits, no Torch."""
    values = np.asarray(logits, dtype=np.float64)
    target = np.asarray(labels)
    require(values.ndim == 2 and values.shape[1] == 10, "logits shape")
    require(target.shape == (len(values),) and target.dtype.kind in "iu", "labels shape/type")
    require(np.isfinite(values).all() and ((target >= 0) & (target < 10)).all(), "logits/labels domain")
    if not len(target):
        return {"count": 0, "correct": 0, "ce_sum": 0.0}
    peak = values.max(axis=1)
    losses = peak + np.log(np.exp(values - peak[:, None]).sum(axis=1))
    losses -= values[np.arange(len(target)), target]
    return {"count": len(target), "correct": int((values.argmax(axis=1) == target).sum()),
            "ce_sum": math.fsum(float(x) for x in losses)}


def sample_summary(values):
    values = [float(x) for x in values]
    require(len(values) > 1 and all(math.isfinite(x) for x in values), "finite sample")
    sd = statistics.stdev(values)
    return {"values": values, "mean": math.fsum(values) / len(values),
            "sample_sd": sd, "sample_se": sd / math.sqrt(len(values)),
            "min": min(values), "max": max(values),
            "positive_count": sum(x > 0 for x in values),
            "negative_count": sum(x < 0 for x in values),
            "zero_count": sum(x == 0 for x in values)}


def same_pair_hashes(rows, fields):
    require(bool(rows), "nonempty paired rows")
    for field in fields:
        values = [row[field] for row in rows]
        require(all(value == values[0] for value in values), "paired mismatch " + field)


def array_hash(value):
    contiguous = np.ascontiguousarray(value)
    header = json.dumps([contiguous.dtype.str, list(contiguous.shape)], separators=(',', ':'))
    return hashlib.sha256(header.encode() + b'\n' + contiguous.tobytes()).hexdigest()


def statistics_and_metrics(train, heldout, clean, fixed, heldout_labels):
    clean_stats, fixed_stats = logits_stats(train, clean), logits_stats(train, fixed)
    heldout_stats = logits_stats(heldout, heldout_labels)
    changed = fixed != clean
    changed_stats = logits_stats(train[changed], fixed[changed])
    stats = {'train_count': clean_stats['count'], 'heldout_count': heldout_stats['count'],
             'train_clean_correct': clean_stats['correct'], 'train_fixed_correct': fixed_stats['correct'],
             'train_corrupted_count': changed_stats['count'], 'train_corrupted_correct': changed_stats['correct'],
             'heldout_clean_correct': heldout_stats['correct'], 'train_clean_ce_sum': clean_stats['ce_sum'],
             'train_fixed_ce_sum': fixed_stats['ce_sum'], 'heldout_clean_ce_sum': heldout_stats['ce_sum']}
    metrics = {'heldout_clean_accuracy': heldout_stats['correct'] / heldout_stats['count'],
               'heldout_clean_ce': heldout_stats['ce_sum'] / heldout_stats['count'],
               'train_clean_accuracy': clean_stats['correct'] / clean_stats['count'],
               'train_fixed_accuracy': fixed_stats['correct'] / fixed_stats['count'],
               'train_corrupted_accuracy': changed_stats['correct'] / changed_stats['count'] if changed_stats['count'] else None,
               'train_clean_ce': clean_stats['ce_sum'] / clean_stats['count'],
               'train_fixed_ce': fixed_stats['ce_sum'] / fixed_stats['count']}
    return stats, metrics


def compare_fields(saved, expected, label):
    require(set(saved) == set(expected), label + ' fields')
    for key, value in expected.items():
        if isinstance(value, int):
            require(isinstance(saved[key], int) and not isinstance(saved[key], bool), label + '.' + key + ' integer')
            require(saved[key] == value, label + '.' + key)
        elif value is None:
            require(saved[key] == value, label + '.' + key)
        else:
            close(saved[key], value, label + '.' + key)


def validate_plan(plan, seed, all_labels):
    expected_fields = {'train_indices', 'heldout_indices', 'initialization_seed', 'replacement_mask',
                       'replacement_digits', 'training_batches', 'train_clean_labels', 'heldout_clean_labels'}
    require(set(plan) == expected_fields, 'plan members')
    train, heldout = plan['train_indices'], plan['heldout_indices']
    both = np.concatenate((train, heldout))
    require(train.shape == heldout.shape == (5000,) and both.dtype.kind in 'iu', 'split shape/type')
    require(np.unique(both).size == 10000 and ((both >= 0) & (both < 60000)).all(), 'disjoint split indices')
    require(np.array_equal(plan['train_clean_labels'], all_labels[train])
            and np.array_equal(plan['heldout_clean_labels'], all_labels[heldout]), 'IDX label identities')
    # Independently replay only the documented pseudorandom plan, never model work.
    streams = [np.random.default_rng(np.random.SeedSequence([20260909, 31, seed, stream])) for stream in range(5)]
    order = streams[0].permutation(60000)
    expected = {'train_indices': order[:5000], 'heldout_indices': order[5000:10000],
                'initialization_seed': np.array(int(streams[1].integers(0, 2**31))),
                'replacement_mask': streams[2].random(5000) < .9,
                'replacement_digits': streams[3].integers(0, 10, 5000),
                'training_batches': streams[4].integers(0, 5000, (2000, 64))}
    for key, value in expected.items():
        require(plan[key].dtype == value.dtype and np.array_equal(plan[key], value), 'frozen plan stream ' + key)
    return {key: array_hash(value) for key, value in expected.items()}


def check_actions(rows, arm):
    require([row['step'] for row in rows] == list(range(101, 2001)), 'action step roster')
    ratios = []
    for row in rows:
        require(row['training_batch_ce'] >= 0, 'negative batch CE')
        raw, applied = row['raw_squared_norm'], row['applied_squared_norm']
        require(raw >= 0 and applied >= 0, 'negative action energy')
        if raw:
            ratio = math.sqrt(applied / raw)
            close(row['applied_to_raw_norm_ratio'], ratio, 'action norm ratio')
            if arm == 'adamw':
                require(raw == applied, 'AdamW delivered raw gradient')
            if arm in ('cluster32', 'mixed32'):
                require(ratio <= 1 + 2e-6, 'cluster action contraction')
            if arm == 'mixed32':
                require(ratio >= .5 - 2e-6, 'half identity action norm floor')
            ratios.append(ratio)
        else:
            require(applied == 0 and row['applied_to_raw_norm_ratio'] is None, 'zero action semantics')
    return {'logged_steps': len(rows), 'norm_ratio_summary': sample_summary(ratios) if len(ratios) > 1 else None}


def check_clusters(rows, arm):
    if arm not in ('cluster32', 'mixed32'):
        require(rows == [], 'baseline has no cluster refreshes')
        return
    require([row['step'] for row in rows] == list(range(101, 2001, 100)), 'cluster refresh roster')
    for index, row in enumerate(rows):
        sizes, isolates = row['cluster_sizes'], row['isolates']
        require(all(isinstance(n, int) and n > 0 for n in sizes) and 0 <= isolates <= 50890, 'cluster sizes')
        require(sum(sizes) + isolates == 50890 and len(sizes) <= 32, 'cluster coverage')
        require(row['nonempty_clusters'] == len(sizes)
                and row['effective_projector_rank'] == len(sizes) + isolates, 'cluster rank algebra')
        close(row['isolate_fraction'], isolates / 50890, 'isolate fraction')
        close(row['largest_cluster_fraction'], max(sizes, default=0) / 50890, 'largest cluster fraction')
        require(0 <= row['factor_rank'] <= 32 and 0 <= row['actual_anchors'] <= 64, 'factor/anchor rank')
        require(0 <= row['kmeans_iterations'] <= 30 and row['retained_array_bytes'] > 0, 'cluster cost domain')
        values = row['graph_eigenvalues']
        require(len(values) <= 32 and all(1e-10 < value <= 1 + 1e-8 for value in values), 'graph eigenvalue domain')
        require(values == sorted(values, reverse=True), 'graph eigenvalue order')
        require(row['refresh_seconds'] >= 0 and len(row['labels_sha256']) == 64, 'cluster receipt/runtime')
        if index == 0:
            require(row['stability_ari_common_assigned'] is None and row['common_assigned_count'] == 0
                    and row['isolate_status_changed_fraction'] is None, 'first refresh null stability')
        else:
            require(0 <= row['common_assigned_count'] <= 50890, 'common assigned count')
            require(0 <= row['isolate_status_changed_fraction'] <= 1, 'isolate status fraction')
            ari = row['stability_ari_common_assigned']
            require((ari is None and row['common_assigned_count'] < 2)
                    or (ari is not None and -1 <= ari <= 1), 'ARI domain')


def label_transition(previous, current):
    old_active, active = previous >= 0, current >= 0
    common = old_active & active
    count = int(common.sum())
    changed = int((old_active != active).sum()) / len(current)
    if count < 2:
        return count, changed, None
    table = np.bincount(previous[common] * 32 + current[common], minlength=1024).reshape(32, 32)
    choose2 = lambda counts: sum(int(x) * (int(x) - 1) // 2 for x in counts)
    observed = choose2(table.ravel())
    old_pairs, new_pairs = choose2(table.sum(axis=1)), choose2(table.sum(axis=0))
    expected = old_pairs * new_pairs / (count * (count - 1) / 2)
    maximum = (old_pairs + new_pairs) / 2
    ari = (observed - expected) / (maximum - expected) if maximum != expected else 1.0
    return count, changed, ari


def check_label_history(labels, rows):
    require(labels.shape == (19, 50890) and labels.dtype == np.int64
            and ((labels >= -1) & (labels < 32)).all(), 'cluster labels history domain')
    for index, row in enumerate(rows):
        current = labels[index]
        require(array_hash(current) == row['labels_sha256'], 'cluster label digest')
        counts = np.bincount(current[current >= 0])
        require(counts[counts > 0].tolist() == row['cluster_sizes']
                and int((current == -1).sum()) == row['isolates'], 'cluster label count arithmetic')
        if index:
            count, changed, ari = label_transition(labels[index - 1], current)
            require(row['common_assigned_count'] == count, 'cluster common-assignment arithmetic')
            close(row['isolate_status_changed_fraction'], changed, 'isolate churn arithmetic')
            if ari is None:
                require(row['stability_ari_common_assigned'] is None, 'small common subset has null ARI')
            else:
                close(row['stability_ari_common_assigned'], ari, 'membership ARI arithmetic')


def summarize(curves):
    indexed = {(row['seed'], row['condition'], row['arm']): row for row in curves}
    require(len(indexed) == 24, 'unique complete trajectory roster')
    groups, differences = {}, {}
    for condition in CONDITIONS:
        for arm in ARMS:
            rows = [indexed[seed, condition, arm] for seed in SEEDS]
            metrics = {}
            for metric in METRICS:
                values = [row['curve'][-1]['metrics'][metric] for row in rows]
                metrics[metric] = None if all(value is None for value in values) else sample_summary(values)
            metrics['warmup_plus_branch_seconds'] = sample_summary([row['warmup_plus_branch_seconds'] for row in rows])
            groups[condition + '/' + arm] = metrics
        for left, right in (('adamw', 'hard32'), ('adamw', 'cluster32'), ('adamw', 'mixed32'),
                            ('hard32', 'cluster32'), ('hard32', 'mixed32')):
            metrics = {}
            for metric in METRICS:
                values = []
                for seed in SEEDS:
                    lhs = indexed[seed, condition, left]['curve'][-1]['metrics'][metric]
                    rhs = indexed[seed, condition, right]['curve'][-1]['metrics'][metric]
                    if lhs is not None and rhs is not None:
                        values.append(rhs - lhs)
                metrics[metric] = sample_summary(values) if values else None
            differences[condition + '/' + right + '_minus_' + left] = metrics
    return {'seeds': list(SEEDS), 'endpoint_step': 2000, 'per_arm': groups, 'paired_differences': differences}


def audit(completion_sha):
    started, bound = time.monotonic(), {}
    def bind(path, expected=None):
        path = Path(path)
        require(path.is_file() and not path.is_symlink(), 'regular input ' + str(path))
        actual = sha256(path)
        require(expected is None or expected == actual, 'input hash ' + str(path))
        bound[str(path)] = actual
        require(time.monotonic() - started < 110, '110-second cooperative audit deadline')
        return path
    def receipt(item):
        require(set(item) == {'path', 'size_bytes', 'sha256'}, 'receipt fields')
        require(Path(item['path']).name == item['path'], 'direct-child receipt')
        path = bind(SOURCE / item['path'], item['sha256'])
        require(path.stat().st_size == item['size_bytes'], 'receipt size')
        return path
    complete = read_json(bind(SOURCE / 'complete.json', completion_sha))
    require(complete['schema'] == 'anchor_graph_mnist_completion_v1' and complete['status'] == 'complete', 'terminal success')
    require(not (SOURCE / 'failed.json').exists(), 'no acquisition failure')
    receipts = {item['path']: item for item in complete['receipts']}
    require(len(receipts) == len(complete['receipts']), 'unique completion receipts')
    require(set(receipts) | {'complete.json'} == {path.name for path in SOURCE.iterdir()}, 'exact completion artifact roster')
    for item in receipts.values():
        receipt(item)  # Hash tensors opaquely; never load them.
    require(sum(item['size_bytes'] for item in receipts.values()) == complete['artifact_bytes_before_completion'], 'artifact byte sum')
    manifest = read_json(receipt(receipts['manifest.json']))
    require(manifest['schema'] == 'anchor_graph_mnist_manifest_v1', 'manifest schema')
    expected = {'seeds': list(SEEDS), 'conditions': list(CONDITIONS), 'arms': list(ARMS), 'steps': 2000,
                'warmup': 100, 'eval_steps': list(EVAL_STEPS), 'rank': 32, 'anchors': 64, 'clusters': 32,
                'sigma': .35, 'refresh_steps': list(range(101, 2001, 100)), 'mixed_identity_weight': .5,
                'cloud_spend_usd': 0}
    for key, value in expected.items():
        require(manifest[key] == value, 'frozen manifest ' + key)
    require(manifest['source_pins'] == complete['source_pins'] and manifest['data_pins'] == complete['data_pins'], 'source/data freeze continuity')
    require(manifest['data_pins'] == DATA_SHA, 'accepted original training IDX hashes')
    pins = manifest['source_pins']
    require(set(pins) == {'spectral_filter.py', 'experiments/anchor_graph_pilot.py',
            'output/2026-09-06-spectral-optimizer-investigation/continuation/iteration-009/neural_core.py',
            'experiments/anchor_graph_mnist.py', 'tests/test_anchor_graph_mnist.py',
            'output/2026-09-09-spectral-clustering-mnist/protocol.md'}, 'scientific source roster')
    require(pins['spectral_filter.py'] == FILTER_SHA and pins['experiments/anchor_graph_pilot.py'] == GRAPH_SHA, 'canonical sources')
    commit = manifest['git_commit']
    require(len(commit) == 40 and all(c in '0123456789abcdef' for c in commit), 'source commit identity')
    for name, digest in pins.items():
        bind(WORKTREE / name, digest)
        committed = subprocess.check_output(['git', 'show', commit + ':' + name], cwd=WORKTREE)
        require(hashlib.sha256(committed).hexdigest() == digest, 'committed source freeze ' + name)
    data = Path(manifest['data_directory'])
    require(data == Path('data/MNIST/raw')
            and set(manifest['data_pins']) == {'train-images-idx3-ubyte', 'train-labels-idx1-ubyte'}, 'training data only')
    for name, digest in manifest['data_pins'].items():
        bind(data / name, digest)
    label_bytes = (data / 'train-labels-idx1-ubyte').read_bytes()
    require(len(label_bytes) == 60008 and struct.unpack('>II', label_bytes[:8]) == (2049, 60000), 'label IDX format')
    all_labels = np.frombuffer(label_bytes, dtype=np.uint8, offset=8).astype(np.int64)
    require(((all_labels >= 0) & (all_labels < 10)).all(), 'MNIST label domain')
    results = read_json(receipt(complete['results']))
    require(complete['results'] == receipts['results.json'] and results['schema'] == 'anchor_graph_mnist_results_v1'
            and results['selection'] == 'fixed_step_2000_no_checkpoint_selection', 'results identity and endpoint selector')
    expected_roster = {(seed, condition, arm) for seed in SEEDS for condition in CONDITIONS for arm in ARMS}
    actual_roster = [(row['seed'], row['condition'], row['arm']) for row in results['rows']]
    require(len(actual_roster) == 24 and set(actual_roster) == expected_roster, 'exact trajectory roster')
    curves, action_summaries, common_logits = [], {}, {}
    for seed in SEEDS:
        with np.load(receipt(receipts[f'plan-s{seed}.npz']), allow_pickle=False) as source:
            plan = {key: source[key] for key in source.files}
        plan_hashes = validate_plan(plan, seed, all_labels)
        initial_hash = None
        for condition in CONDITIONS:
            fixed = plan['train_clean_labels'].copy()
            if condition != 'clean':
                fixed[plan['replacement_mask']] = plan['replacement_digits'][plan['replacement_mask']]
            changed = fixed != plan['train_clean_labels']
            stem = f's{seed}-{condition}'
            binding = read_json(receipt(receipts[f'binding-{stem}.json']))
            require(binding['seed'] == seed and binding['condition'] == condition, 'binding identity')
            require(binding['plan'] == receipts[f'plan-s{seed}.npz'] and binding['plan_array_hashes'] == plan_hashes, 'plan binding')
            require(binding['fixed_targets_sha256'] == array_hash(fixed)
                    and binding['actually_changed_mask_sha256'] == array_hash(changed), 'corruption binding')
            require(binding['selected_replacement_count'] == (int(plan['replacement_mask'].sum()) if condition != 'clean' else 0)
                    and binding['actually_changed_count'] == int(changed.sum()), 'replacement/changed distinction')
            require(binding['warmup'] == receipts[f'warmup-{stem}.pt'] and binding['warmup_seconds'] >= 0, 'warmup binding')
            if initial_hash is not None:
                require(binding['initial_model_sha256'] == initial_hash, 'initial model matched across label conditions')
            initial_hash = binding['initial_model_sha256']
            paired_rows = []
            offset = (SEEDS.index(seed) + CONDITIONS.index(condition)) % 4
            order = list(ARMS[offset:] + ARMS[:offset])
            for arm in ARMS:
                identifier = stem + '-' + arm
                row = read_json(receipt(receipts[f'curve-{identifier}.json']))
                require(all(row[key] == value for key, value in binding.items()), 'common binding exact in each arm')
                require(row['arm'] == arm and row['arm_order'] == order, 'arm identity/order')
                require([entry['step'] for entry in row['curve']] == list(EVAL_STEPS), 'evaluation step roster')
                close(row['warmup_plus_branch_seconds'], row['warmup_seconds'] + row['branch_seconds_including_eval_excluding_save'], 'runtime sum')
                require(row['branch_seconds_including_eval_excluding_save'] >= 0, 'nonnegative runtime')
                require(row['final_state'] == receipts[f'final-{identifier}.pt'], 'final state receipt')
                with np.load(receipt(row['logits']), allow_pickle=False) as saved:
                    require(set(saved.files) == {'steps', 'train_logits', 'heldout_logits'}, 'logit members')
                    require(np.array_equal(saved['steps'], EVAL_STEPS), 'logit steps')
                    train, heldout = saved['train_logits'], saved['heldout_logits']
                    require(train.dtype == heldout.dtype == np.float32 and train.shape == heldout.shape == (21, 5000, 10), 'saved logit dtype/shape')
                    for index in (0, 1):
                        key = (seed, 'initial' if index == 0 else condition)
                        hashes = (array_hash(train[index]), array_hash(heldout[index]))
                        require(key not in common_logits or common_logits[key] == hashes, 'common initial/warmup logits')
                        common_logits[key] = hashes
                    derived = []
                    for index, entry in enumerate(row['curve']):
                        stats, metrics = statistics_and_metrics(train[index], heldout[index],
                            plan['train_clean_labels'], fixed, plan['heldout_clean_labels'])
                        compare_fields(entry['sufficient_statistics'], stats, identifier + ' stats')
                        compare_fields(entry['metrics'], metrics, identifier + ' metrics')
                        derived.append({'step': entry['step'], 'metrics': metrics})
                actions = read_json(receipt(row['actions']))
                action_summaries[identifier] = check_actions(actions, arm)
                first = row['first_action_binding']
                first_keys = {'raw_gradient_sha256'} | ({'post_observe_tracker_sha256'} if arm != 'adamw' else set())
                require(set(first) == first_keys and all(len(value) == 64 for value in first.values()), 'first-action binding fields')
                require({key: value for key, value in actions[0].items() if key.endswith('sha256')} == first,
                        'first-action row/binding equality')
                clusters = read_json(receipt(row['clusters']))
                check_clusters(clusters, arm)
                if arm in ('cluster32', 'mixed32'):
                    require(row['cluster_labels'] == receipts[f'labels-{identifier}.npz'], 'label history receipt')
                    with np.load(receipt(row['cluster_labels']), allow_pickle=False) as saved:
                        require(set(saved.files) == {'steps', 'labels'}
                                and np.array_equal(saved['steps'], list(range(101, 2001, 100))), 'cluster label history members/steps')
                        check_label_history(saved['labels'], clusters)
                else:
                    require(row['cluster_labels'] is None, 'baseline has no clustering labels')
                result = next(item for item in results['rows'] if (item['seed'], item['condition'], item['arm']) == (seed, condition, arm))
                require(result['curve_receipt'] == receipts[f'curve-{identifier}.json'], 'summary curve receipt')
                compare_fields(result['endpoint'], derived[-1]['metrics'], 'summary fixed endpoint')
                close(result['warmup_plus_branch_seconds'], row['warmup_plus_branch_seconds'], 'summary runtime')
                curves.append({'seed': seed, 'condition': condition, 'arm': arm,
                               'curve': derived, 'warmup_plus_branch_seconds': row['warmup_plus_branch_seconds']})
                paired_rows.append(row)
            same_pair_hashes(paired_rows, ('plan_array_hashes', 'initial_model_sha256', 'warmup_state_sha256',
                                           'fixed_targets_sha256', 'actually_changed_mask_sha256'))
            same_pair_hashes([row['first_action_binding'] for row in paired_rows], ('raw_gradient_sha256',))
            same_pair_hashes([row['first_action_binding'] for row in paired_rows if row['arm'] != 'adamw'],
                            ('post_observe_tracker_sha256',))
    for path, digest in list(bound.items()):
        require(sha256(path) == digest, 'input unchanged after audit')
    return {'schema': 'anchor_graph_mnist_saved_audit_v1', 'status': 'PASS', 'completion_sha256': completion_sha,
            'scope': 'saved evidence only; no training, inference or optimizer replay', 'input_hashes': bound,
            'source_hashes': {path.name: sha256(path) for path in Path(__file__).parent.glob('*.py')},
            'summary': summarize(curves), 'curves': curves, 'action_summaries': action_summaries,
            'wall_seconds': time.monotonic() - started, 'peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'limitations': ['Three paired seeds; exploratory endpoints, no tuning or significance claim.',
                'Saved model-state identities are matched and files hashed, not independently replayed.',
                'Runtime includes branch evaluation and unequal observer overhead; not a production speed benchmark.',
                'No semantic grouping, causal mediation or safety conclusion follows from arithmetic agreement.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--completion-sha', required=True)
    args = parser.parse_args()
    require(len(args.completion_sha) == 64 and all(c in '0123456789abcdef' for c in args.completion_sha), 'completion SHA-256')
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        require(os.environ.get(name) == '1', 'one-thread environment ' + name)
    require(not OUTPUT.exists() and not OUTPUT.is_symlink(), 'exclusive audit output')
    OUTPUT.mkdir()
    try:
        result = audit(args.completion_sha)
    except BaseException as exc:
        result = {'status': 'FAIL', 'type': type(exc).__name__, 'message': str(exc),
                  'completion_sha256': args.completion_sha, 'scope': 'saved-evidence audit; no automatic retry'}
        with (OUTPUT / 'failed.json').open('x') as handle:
            json.dump(result, handle, indent=2, allow_nan=False)
        raise
    payload = json.dumps(result, indent=2, allow_nan=False) + '\n'
    require(len(payload.encode()) < 100 * 1024**2, 'audit output below 100 MiB')
    with (OUTPUT / 'result.json').open('x') as handle:
        handle.write(payload)
    print(json.dumps({'status': result['status'], 'output': str(OUTPUT / 'result.json'),
                      'sha256': sha256(OUTPUT / 'result.json'), 'wall_seconds': result['wall_seconds']}))


if __name__ == '__main__':
    main()
