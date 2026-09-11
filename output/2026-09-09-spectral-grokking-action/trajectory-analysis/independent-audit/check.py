"""Independent saved-JSON arithmetic audit; never imports production analysis.

Run once under external 1 CPU / 2 GiB / no swap / 300 s service limits.
Only the exclusive result JSON is written. No tensor/checkpoint is opened.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import statistics
import time
import traceback

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
BASE = Path('/tmp/spectral-experiment-artifacts/spectral-grokking-action-20260909.ghvEvD')
SUMMARY = HERE.parent / 'summary.json'
AUDIT = HERE.parent.parent / 'results/tensor-audit-001.json'
SEEDS = list(range(100, 105))
POLICIES = ['orthogonal', 'norm_matched']
WINDOWS = [(1502, 2500), (1502, 2000), (2001, 2500)]
ROSTER = [['native', 1501], ['orthogonal', 1501], ['orthogonal', 2000],
          ['orthogonal', 2500], ['norm_matched', 1501],
          ['norm_matched', 2000], ['norm_matched', 2500]]
T0 = time.monotonic()
CHECKS = 0
ERRORS = []
MAX_ABS_DIFF = 0.0
RECEIPTS = {}


def check(condition, label):
    global CHECKS
    CHECKS += 1
    if not condition:
        ERRORS.append(label)
    if time.monotonic() - T0 > 280:
        raise TimeoutError('Cooperative 280-second limit')


def compare(actual, expected, label):
    """Strict structure/integers; 1e-12 abs+relative floating tolerance."""
    global MAX_ABS_DIFF
    if isinstance(expected, dict):
        check(isinstance(actual, dict), label + '/dict')
        if not isinstance(actual, dict):
            return
        check(set(actual) == set(expected), label + '/keys')
        for key in expected.keys() & actual.keys():
            compare(actual[key], expected[key], label + '/' + str(key))
    elif isinstance(expected, list):
        check(isinstance(actual, list), label + '/list')
        if not isinstance(actual, list):
            return
        check(len(actual) == len(expected), label + '/length')
        for index, (a, e) in enumerate(zip(actual, expected)):
            compare(a, e, label + '/' + str(index))
    elif isinstance(expected, float):
        numeric = isinstance(actual, (int, float)) and not isinstance(actual, bool)
        check(numeric, label + '/numeric')
        if numeric:
            MAX_ABS_DIFF = max(MAX_ABS_DIFF, abs(actual - expected))
            check(math.isfinite(actual) and math.isclose(actual, expected,
                  rel_tol=1e-12, abs_tol=1e-12), label + '/value')
    else:
        check(actual == expected, label + '/value')


def receipt(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    result = {'sha256': digest.hexdigest(), 'size_bytes': path.stat().st_size}
    RECEIPTS[str(path)] = result
    return result


def read(path):
    with Path(path).open() as stream:
        return json.load(stream)


def stats(values):
    defined = [x for x in values if x is not None]
    result = {'count': len(values), 'defined_count': len(defined),
              'undefined_count': len(values) - len(defined)}
    # Undefined values invalidate the whole statistic, not a selected subset.
    if len(defined) != len(values) or not defined:
        result.update(dict.fromkeys(['minimum', 'maximum', 'mean', 'median']))
    else:
        result.update(minimum=min(defined), maximum=max(defined),
                      mean=statistics.mean(defined), median=statistics.median(defined))
    return result


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def values_and_flags(row, policy, label):
    k, rank = row['basis_column_count'], row['numerical_rank']
    s = row['singular_values']
    d = row['diagnostics']
    norms = d['action_norms']
    check(row['P'] == 227313 and row['k'] == k, label + '/dimensions')
    check(len(s) == k and 0 <= rank <= k <= 200, label + '/rank_shape')
    check(all(math.isfinite(x) and x >= 0 for x in s), label + '/finite_singular')
    check(all(a >= b for a, b in zip(s, s[1:])), label + '/singular_order')
    compare(row['sigma_max'], max(s), label + '/sigma_max')
    compare(row['tolerance'], max(row['P'], k) * (2.0 ** -52) * max(s),
            label + '/rank_tolerance')
    check(rank == sum(x > row['tolerance'] for x in s), label + '/rank_threshold')
    check(row['full_column_rank'] == (rank == k), label + '/rank_flag')
    expected_actions = ['orthogonal'] if policy == 'orthogonal' else ['native', 'orthogonal', 'norm_matched']
    check(set(norms) == set(expected_actions), label + '/norm_availability')
    check(set(row['coefficients']) == set(expected_actions), label + '/coefficient_availability')
    for action, coefficients in row['coefficients'].items():
        check(len(coefficients) == rank, label + '/' + action + '/coefficient_length')
        check(all(math.isfinite(x) for x in coefficients), label + '/' + action + '/finite_coefficients')
    check(math.isfinite(d['input_norm']) and d['input_norm'] >= 0, label + '/raw_norm')
    check(all(math.isfinite(x) and x >= 0 for x in norms.values()), label + '/action_norms')
    v = {'k': k, 'rank': rank, 'rank_over_k': ratio(rank, k),
         'singular_max': max(s), 'singular_min': min(s),
         'raw_norm': d['input_norm'], 'delivered_norm': norms[policy],
         'delivered_over_raw': ratio(norms[policy], d['input_norm'])}
    flags = {'full_column_rank': int(rank == k)}
    if policy == 'orthogonal':
        check(d['pairwise_cosines'] == {}, label + '/absent_cosines')
        check(not any(key.startswith('norm_match') for key in d), label + '/absent_norm_match')
    else:
        absolute = abs(norms['norm_matched'] - norms['native'])
        relative = absolute / max(norms['native'], 1e-30)
        scale = norms['native'] / max(norms['orthogonal'], 1e-30)
        compare(d['norm_match_scale'], scale, label + '/scale')
        compare(d['norm_match_absolute_mismatch'], absolute, label + '/absolute_mismatch')
        compare(d['norm_match_relative_mismatch'], relative, label + '/relative_mismatch')
        check(d['norm_match_denominator_clamped'] == (norms['orthogonal'] < 1e-30), label + '/clamp_flag')
        # Actual data are all nondegenerate; do not invent unsupported alternate semantics.
        check(norms['native'] > 0 and norms['orthogonal'] > 0 and
              not d['norm_match_degenerate'], label + '/nondegenerate')
        check(d['norm_match_exact'] == (relative <= d['norm_match_relative_tolerance']), label + '/exact_flag')
        cos = d['pairwise_cosines']['native_orthogonal']
        check(cos is None or (math.isfinite(cos) and -1.000001 <= cos <= 1.000001), label + '/cosine')
        v.update(c=scale, branch_local_native_over_raw=ratio(norms['native'], d['input_norm']),
                 branch_local_orthogonal_over_raw=ratio(norms['orthogonal'], d['input_norm']),
                 native_orthogonal_cosine=cos, absolute_postcast_mismatch=absolute,
                 relative_postcast_mismatch=relative)
        flags.update(clamped=int(d['norm_match_denominator_clamped']),
                     degenerate=int(d['norm_match_degenerate']), nonexact=int(not d['norm_match_exact']),
                     tolerance_exceeded=int(relative > d['norm_match_relative_tolerance']))
    return v, flags


def main():
    summary_receipt = receipt(SUMMARY)
    summary = read(SUMMARY)
    complete = read(HERE.parent / 'complete.json')
    compare(complete['summary_sha256'], summary_receipt['sha256'], 'summary_completion_hash')
    compare(summary['seeds'], SEEDS, 'seed_roster')
    compare(summary['policies'], POLICIES, 'policy_roster')
    compare(summary['total_history_rows'], 9990, 'total_rows')
    compare(summary['history_steps_per_branch'], 999, 'steps_per_branch')
    for path, expected in summary['input_receipts'].items():
        compare(receipt(path), expected, 'input_receipt/' + path)
    for path, digest in summary['analysis_source_sha256'].items():
        compare(receipt(path)['sha256'], digest, 'analysis_source/' + path)
    for path, digest in summary['source_sha256'].items():
        compare(receipt(REPO / path)['sha256'], digest, 'scientific_source/' + path)
    batch = read(BASE / 'batch-complete.json')
    compare(receipt(BASE / 'batch-complete.json')['sha256'],
            '22bb866319fa2fa85771bd156eb406523a66abee4d0bc789cc6104d40ec86e45', 'accepted_batch_hash')
    compare(receipt(AUDIT)['sha256'],
            '07113dba0b319ff64c920f8dba1d55abcd1350b9e50133acd6fd2530f1cb5da7', 'accepted_tensor_audit_hash')
    tensor = read(AUDIT)
    first_keys = ['seed', 'k', 'rank', 'action_cosines', 'action_norms', 'norm_match_scale', 'adam']
    compare(summary['separate_1501_audited_first_step'],
            [{k: row[k] for k in first_keys} for row in tensor['rows']], 'separate_first_step')
    expected_rows = []
    all_values = {}
    for accepted in batch['accepted']:
        seed = accepted['seed']
        check(seed in SEEDS, 'accepted_seed')
        compare(receipt(accepted['path'])['sha256'], accepted['sha256'], 'accepted_seed_hash')
        seed_receipt = read(accepted['path'])
        compare(seed_receipt['accepted_roster'], ROSTER, f'{seed}/accepted_roster')
        compare([[x['policy'], x['step']] for x in seed_receipt['checkpoints']], ROSTER, f'{seed}/checkpoint_roster')
        compare(seed_receipt['source_sha256'], summary['source_sha256'], f'{seed}/source_map')
        accepted_tensor_row = next(row for row in tensor['rows'] if row['seed'] == seed)
        compare(seed_receipt['parent_checkpoint'], accepted_tensor_row['parent_receipt'], f'{seed}/parent_receipt')
        for policy in POLICIES:
            path = BASE / f'seed{seed}/{policy}-through-002500.json'
            before = receipt(path)
            raw = read(path)
            label = f'{seed}/{policy}'
            compare([raw['schema'], raw['seed'], raw['policy'], raw['step']],
                    ['grokking_action_intervention_v1', seed, policy, 2500], label + '/identity')
            compare(raw['source_sha256'], summary['source_sha256'], label + '/source_map')
            history = raw['action_history']
            compare([r['step'] for r in history], list(range(1502, 2501)), label + '/exact_steps')
            checkpoint = next(r for r in seed_receipt['checkpoints'] if r['policy'] == policy and r['step'] == 2500)
            compare(raw['checkpoint'], checkpoint, label + '/checkpoint')
            values = [values_and_flags(row, policy, label + '/' + str(row['step'])) for row in history]
            windows = []
            for start, end in WINDOWS:
                subset = values[start - 1502:end - 1501]
                metric_names, flag_names = set(subset[0][0]), set(subset[0][1])
                metrics = {name: stats([r[0][name] for r in subset]) for name in metric_names}
                counts = {name: sum(r[1][name] for r in subset) for name in flag_names}
                windows.append({'count': end - start + 1, 'counts': counts,
                                'start_step': start, 'end_step': end, 'metrics': metrics})
                all_values[seed, policy, start, end] = subset
            expected_rows.append({'seed': seed, 'policy': policy, 'source_commit': seed_receipt['source_commit'],
                                  'history_receipt': before, 'endpoint_checkpoint_receipt': checkpoint,
                                  'windows': windows})
            compare(receipt(path), before, label + '/unchanged_during_read')
    compare(summary['rows'], expected_rows, 'all_per_seed_windows')
    aggregates = []
    for policy in POLICIES:
        for start, end in WINDOWS:
            selected = [all_values[seed, policy, start, end] for seed in SEEDS]
            metrics, flags = set(selected[0][0][0]), set(selected[0][0][1])
            seed_means = {name: stats([stats([r[0][name] for r in rows])['mean'] for rows in selected]) for name in metrics}
            ranges = {name: {'minimum': min(r[0][name] for rows in selected for r in rows),
                             'maximum': max(r[0][name] for rows in selected for r in rows)} for name in metrics}
            counts = {name: sum(r[1][name] for rows in selected for r in rows) for name in flags}
            aggregates.append({'policy': policy, 'start_step': start, 'end_step': end,
                               'counts': counts, 'seed_mean_summaries': seed_means,
                               'all_observed_ranges': ranges})
    compare(summary['equal_seed_aggregates'], aggregates, 'all_equal_seed_aggregates')
    compare(receipt(SUMMARY), summary_receipt, 'summary_unchanged')


if __name__ == '__main__':
    result_path = HERE / 'result-001.json'
    if result_path.exists():
        raise FileExistsError('Refusing an audit retry or overwrite')
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    try:
        main()
    except Exception:
        ERRORS.append(traceback.format_exc())
    result = {'status': 'PASS' if not ERRORS else 'FAIL', 'checks': CHECKS,
              'errors': ERRORS, 'maximum_absolute_comparison_difference': MAX_ABS_DIFF,
              'elapsed_seconds': time.monotonic() - T0,
              'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
              'scope': 'Independent saved scalar arithmetic and present JSON/source receipts; no tensor truth or retrospective immutable-history proof.',
              'script_sha256': receipt(Path(__file__))['sha256'], 'input_receipts': RECEIPTS}
    encoded = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n'
    if len(encoded.encode()) > 100 * 1024 * 1024:
        raise RuntimeError('100 MiB output limit')
    with result_path.open('x') as stream:
        stream.write(encoded)
    print(json.dumps({k: result[k] for k in ['status', 'checks', 'errors', 'elapsed_seconds', 'peak_rss_bytes']}))
    raise SystemExit(bool(ERRORS))
