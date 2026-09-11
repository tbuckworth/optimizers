#!/usr/bin/env python3
"""One-shot, fixed-roster, post-outcome descriptive analysis of saved JSON only."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import resource
import shutil
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
BATCH = Path('/tmp/spectral-experiment-artifacts/spectral-grokking-action-20260909.ghvEvD')
OUTPUT = BATCH / 'trajectory-analysis-001'
BATCH_SHA = '22bb866319fa2fa85771bd156eb406523a66abee4d0bc789cc6104d40ec86e45'
AUDIT = HERE.parent / 'results/tensor-audit-001.json'
AUDIT_SHA = '07113dba0b319ff64c920f8dba1d55abcd1350b9e50133acd6fd2530f1cb5da7'
SCHEMA = 'grokking_action_intervention_v1'
SEEDS = tuple(range(100, 105))
POLICIES = ('orthogonal', 'norm_matched')
WINDOWS = ((1502, 2500), (1502, 2000), (2001, 2500))
ROSTER = [('native', 1501)] + [(p, s) for p in POLICIES for s in (1501, 2000, 2500)]
FLOOR = 1e-30
RELATIVE_TOLERANCE = 10 * 2**-23  # Original float32 delivered gradients.
MAX_BYTES = 100 * 1024**2
RESERVE = 1024**3
START = time.monotonic()
INPUTS = {}
CHECKS = 0


def check(condition, message):
    global CHECKS
    CHECKS += 1
    if not condition:
        raise ValueError(message)
    if time.monotonic() - START > 280:
        raise TimeoutError('Cooperative runtime limit exceeded')


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def read_json(path, expected=None):
    check(path.is_file() and not path.is_symlink(), f'Not a regular input: {path}')
    check(path.stat().st_size <= 100 * 1024**2, f'Oversize input: {path}')
    before = sha(path)
    if expected is not None:
        check(before == expected, f'Input hash mismatch: {path}')
    value = json.loads(path.read_text(), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
    check(sha(path) == before, f'Input changed while reading: {path}')
    INPUTS[str(path)] = {'sha256': before, 'size_bytes': path.stat().st_size}
    return value


def numeric(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def close(left, right, label):
    check(numeric(left) and numeric(right) and math.isclose(left, right, rel_tol=1e-10, abs_tol=1e-30), label)


def stats(values):
    defined = [v for v in values if v is not None]
    check(all(numeric(v) for v in defined), 'Nonfinite summary input')
    result = {'count': len(values), 'defined_count': len(defined), 'undefined_count': len(values) - len(defined)}
    if len(defined) != len(values):
        return {**result, 'minimum': None, 'maximum': None, 'mean': None, 'median': None}
    return {**result, 'minimum': min(defined), 'maximum': max(defined),
            'mean': statistics.fmean(defined), 'median': statistics.median(defined)}


def ratio(numerator, denominator):
    return numerator / denominator if denominator > 0 else None


def extract(row, policy, expected_step):
    check(type(row['step']) is int and row['step'] == expected_step, 'Noncontiguous action step')
    p, k, rank = row['P'], row['k'], row['numerical_rank']
    check(type(p) is int and p == 227313 and type(k) is int and 0 < k <= 200, 'P/k contract')
    check(type(rank) is int and 0 <= rank <= k, 'Numerical rank bounds')
    check(row['basis_column_count'] == k, 'Basis-column count mismatch')
    check(type(row['full_column_rank']) is bool and row['full_column_rank'] == (rank == k), 'Full-rank flag')
    singular = row['singular_values']
    check(len(singular) == k and all(numeric(x) and x >= 0 for x in singular), 'Singular values')
    check(all(a >= b for a, b in zip(singular, singular[1:])), 'Singular order')
    close(row['sigma_max'], singular[0], 'Largest singular value')
    close(row['tolerance'], max(p, k) * sys.float_info.epsilon * singular[0], 'Rank tolerance')
    kept = [x for x in singular if x > row['tolerance']]
    check(len(kept) == rank, 'Threshold rank')
    total = math.fsum(x*x for x in singular)
    discarded = math.fsum(x*x for x in singular if x <= row['tolerance'])
    close(row['svd_truncation_relative_frobenius_residual'], math.sqrt(discarded / total) if total else 0, 'Truncation residual')
    check(numeric(row['elapsed_seconds']) and row['elapsed_seconds'] >= 0, 'Elapsed field')
    diag = row['diagnostics']
    expected_actions = {'orthogonal'} if policy == 'orthogonal' else {'native', 'orthogonal', 'norm_matched'}
    check(set(diag['action_norms']) == expected_actions, 'Action metric availability')
    check(set(row['coefficients']) == expected_actions, 'Coefficient availability')
    check(all(len(v) == rank and all(numeric(x) for x in v) for v in row['coefficients'].values()), 'Coefficient dimensions/finiteness')
    norms = diag['action_norms']
    check(all(numeric(x) and x >= 0 for x in norms.values()), 'Action norms')
    raw = diag['input_norm']
    check(numeric(raw) and raw >= 0, 'Raw gradient norm')
    check(norms['orthogonal'] <= raw * (1 + RELATIVE_TOLERANCE) + FLOOR, 'Projection norm exceeds raw norm')
    result = {'step': expected_step, 'k': k, 'rank': rank, 'rank_over_k': rank/k,
              'singular_max': singular[0], 'singular_min': singular[-1],
              'raw_norm': raw, 'delivered_norm': norms[policy],
              'delivered_over_raw': ratio(norms[policy], raw),
              'full_column_rank': rank == k}
    if policy == 'orthogonal':
        check(set(diag) == {'action_norms', 'input_norm', 'pairwise_cosines'} and diag['pairwise_cosines'] == {}, 'Orthogonal-only logging contract')
        return result
    native, orth = norms['native'], norms['orthogonal']
    c = native / max(orth, FLOOR)
    close(diag['norm_match_scale'], c, 'Norm-match c')
    absolute = abs(norms['norm_matched'] - native)
    relative = absolute / max(native, FLOOR)
    close(diag['norm_match_absolute_mismatch'], absolute, 'Absolute postcast mismatch')
    close(diag['norm_match_relative_mismatch'], relative, 'Relative postcast mismatch')
    close(diag['norm_match_relative_tolerance'], RELATIVE_TOLERANCE, 'Original fp32 tolerance')
    expected_flags = {'norm_match_degenerate': orth <= FLOOR,
                      'norm_match_denominator_clamped': orth < FLOOR,
                      'norm_match_exact': orth >= FLOOR or (orth == 0 and native == 0)}
    for key, expected in expected_flags.items():
        check(type(diag[key]) is bool and diag[key] == expected, key)
    check(not (orth == 0 and native > 0), 'Invalid zero-denominator successful trajectory')
    cosine = diag['pairwise_cosines']
    check(set(cosine) == {'native_orthogonal', 'native_norm_matched', 'orthogonal_norm_matched'}, 'Cosine availability')
    for key, (left, right) in {'native_orthogonal': ('native', 'orthogonal'), 'native_norm_matched': ('native', 'norm_matched'), 'orthogonal_norm_matched': ('orthogonal', 'norm_matched')}.items():
        value = cosine[key]
        check((value is None) == (norms[left] * norms[right] <= FLOOR), 'Cosine degeneracy')
        if value is not None:
            check(numeric(value) and -1 - 1e-12 <= value <= 1 + 1e-12, 'Cosine bounds')
    result.update({'c': c, 'branch_local_native_over_raw': ratio(native, raw),
                   'branch_local_orthogonal_over_raw': ratio(orth, raw),
                   'native_orthogonal_cosine': cosine['native_orthogonal'],
                   'absolute_postcast_mismatch': absolute, 'relative_postcast_mismatch': relative,
                   'degenerate': diag['norm_match_degenerate'],
                   'clamped': diag['norm_match_denominator_clamped'],
                   'nonexact': not diag['norm_match_exact'],
                   'tolerance_exceeded': relative > RELATIVE_TOLERANCE})
    return result


def summarize(rows, policy, lo, hi):
    window = [row for row in rows if lo <= row['step'] <= hi]
    check(len(window) == hi - lo + 1, 'Window roster')
    flags = {'full_column_rank'} | ({'degenerate', 'clamped', 'nonexact', 'tolerance_exceeded'} if policy == 'norm_matched' else set())
    keys = set(window[0]) - flags - {'step'}
    return {'start_step': lo, 'end_step': hi, 'count': len(window),
            'metrics': {key: stats([row[key] for row in window]) for key in sorted(keys)},
            'counts': {key: sum(row[key] for row in window) for key in sorted(flags)}}


def bounded_write(path, value):
    encoded = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()
    existing = sum(p.stat().st_size for p in OUTPUT.iterdir() if p.is_file())
    check(existing + len(encoded) <= MAX_BYTES and shutil.disk_usage(OUTPUT).free >= RESERVE + len(encoded), 'Output/free-space limit')
    with path.open('xb') as target:
        target.write(encoded)


def main():
    check(len(sys.argv) == 1, 'Fixed analysis accepts no arguments')
    check(not OUTPUT.exists() and not OUTPUT.is_symlink(), 'Output must be new and exclusive')
    check(shutil.disk_usage(BATCH).free >= MAX_BYTES + RESERVE, 'Initial free-space reserve')
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        check(os.environ.get(key) == '1', f'{key} must equal 1')
    # CPU-only analysis does not import any external numerical library.
    source_pins = {str(path): sha(path) for path in (Path(__file__).resolve(), HERE / 'plan.md', HERE / 'test_analyze.py')}
    batch = read_json(BATCH / 'batch-complete.json', BATCH_SHA)
    check(not (BATCH / 'batch-failure.json').exists(), 'Batch failure marker')
    audit = read_json(AUDIT, AUDIT_SHA)
    check(batch['status'] == 'complete' and batch['seeds'] == list(SEEDS), 'Completed fixed batch')
    check([r['seed'] for r in batch['accepted']] == list(SEEDS), 'Five completion receipts')
    check(audit['status'] == 'PASS' and not audit['errors'] and audit['batch_completion_sha256'] == BATCH_SHA, 'Accepted tensor audit')
    check([row['seed'] for row in audit['rows']] == list(SEEDS), 'First-step audit roster')
    OUTPUT.mkdir(mode=0o700)
    summaries, sources, first_steps = [], None, []
    for seed, receipt, first in zip(SEEDS, batch['accepted'], audit['rows']):
        expected_path = BATCH / f'seed{seed}/complete.json'
        check(Path(receipt['path']) == expected_path, 'Completion path')
        complete = read_json(expected_path, receipt['sha256'])
        check(not (expected_path.parent / 'failure.json').exists(), 'Seed failure marker')
        check(complete['schema'] == SCHEMA and complete['status'] == 'complete' and complete['seed'] == seed, 'Seed completion identity')
        check(complete['accepted_roster'] == [list(x) for x in ROSTER], 'Seven-state accepted roster')
        check([(r['policy'], r['step']) for r in complete['checkpoints']] == ROSTER, 'Checkpoint roster')
        check(re.fullmatch('[0-9a-f]{40}', complete['source_commit']) is not None, 'Per-seed commit identity')
        for checkpoint in complete['checkpoints']:
            check(checkpoint['seed'] == seed and Path(checkpoint['path']) == expected_path.parent / f"{checkpoint['policy']}-step-{checkpoint['step']:06d}.pt", 'Accepted checkpoint identity')
            check(re.fullmatch('[0-9a-f]{64}', checkpoint['sha256']) is not None and type(checkpoint['size_bytes']) is int and checkpoint['size_bytes'] > 0, 'Accepted checkpoint receipt format')
        check(first['completion_receipt'] == receipt and first['parent_receipt'] == complete['parent_checkpoint'], 'Accepted parent/audit identity')
        check(complete['parent_checkpoint']['path'].endswith(f'/seed{seed}/legacy/checkpoint-step-001500.pt'), 'Parent seed/arm/step identity')
        check(first['source_sha256'] == complete['source_sha256'], 'First-step scientific source binding')
        if sources is None:
            sources = complete['source_sha256']
            check(len(sources) == 10, 'Original ten-file scientific map')
            for name, expected in sources.items():
                check(sha(ROOT / name) == expected, f'Scientific source hash: {name}')
        check(complete['source_sha256'] == sources, 'Shared scientific source map')
        first_steps.append({key: first[key] for key in ('seed', 'k', 'rank', 'norm_match_scale', 'action_norms', 'action_cosines', 'adam')})
        for policy in POLICIES:
            path = BATCH / f'seed{seed}/{policy}-through-002500.json'
            data = read_json(path)
            check((data['schema'], data['seed'], data['policy'], data['step']) == (SCHEMA, seed, policy, 2500), 'History identity')
            check(data['source_sha256'] == sources, 'History source binding')
            checkpoint = next(r for r in complete['checkpoints'] if r['policy'] == policy and r['step'] == 2500)
            check(data['checkpoint'] == checkpoint, 'History checkpoint receipt equality')
            check(checkpoint['seed'] == seed and Path(checkpoint['path']) == path.parent / f'{policy}-step-002500.pt', 'Checkpoint seed/path identity')
            history = data['action_history']
            check(len(history) == 999, 'Exact history length')
            check([r['step'] for r in data['evaluation_rows']] == list(range(1550, 2501, 50)), 'Original evaluation-grid identity')
            rows = [extract(row, policy, step) for row, step in zip(history, range(1502, 2501))]
            summaries.append({'seed': seed, 'policy': policy, 'history_receipt': INPUTS[str(path)],
                              'source_commit': complete['source_commit'], 'endpoint_checkpoint_receipt': checkpoint,
                              'windows': [summarize(rows, policy, lo, hi) for lo, hi in WINDOWS]})
            del data, history, rows
    # Equal-seed descriptive aggregation retains all five per-seed summaries.
    aggregates = []
    for policy in POLICIES:
        for window_index, (lo, hi) in enumerate(WINDOWS):
            selected = [row['windows'][window_index] for row in summaries if row['policy'] == policy]
            check(len(selected) == 5, 'Aggregate complete seed roster')
            aggregates.append({'policy': policy, 'start_step': lo, 'end_step': hi,
                'seed_mean_summaries': {key: stats([w['metrics'][key]['mean'] for w in selected]) for key in selected[0]['metrics']},
                'all_observed_ranges': {key: {'minimum': min(w['metrics'][key]['minimum'] for w in selected), 'maximum': max(w['metrics'][key]['maximum'] for w in selected)} if all(w['metrics'][key]['undefined_count'] == 0 for w in selected) else None for key in selected[0]['metrics']},
                'counts': {key: sum(w['counts'][key] for w in selected) for key in selected[0]['counts']}})
    for name, receipt in INPUTS.items():
        check(sha(Path(name)) == receipt['sha256'], f'Input changed by completion: {name}')
    for name, expected in sources.items():
        check(sha(ROOT / name) == expected, f'Scientific source changed: {name}')
    for name, expected in source_pins.items():
        check(sha(Path(name)) == expected, f'Analysis source changed: {name}')
    result = {'schema': 'grokking_action_trajectory_summary_v1', 'status': 'complete',
              'analysis_type': 'post_outcome_exploratory_saved_json_only', 'seeds': list(SEEDS),
              'policies': list(POLICIES), 'history_steps_per_branch': 999, 'total_history_rows': 9990,
              'source_sha256': sources, 'analysis_source_sha256': source_pins,
              'source_commit_at_analysis': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              'input_receipts': INPUTS, 'history_provenance_caveat': 'History JSONs were not individually hashed in the original completion receipts; present hashes and metadata consistency are not retrospective immutable-history proof.',
              'checkpoint_verification_scope': 'Accepted receipt metadata only; no checkpoint payload reopened.',
              'availability': {'orthogonal': 'No native action, c, or native-versus-orthogonal cosine recorded.', 'norm_matched': 'All actions are at this branch own current state, not archived-native trajectory.'},
              'rows': summaries, 'equal_seed_aggregates': aggregates,
              'separate_1501_audited_first_step': first_steps,
              'checks': CHECKS, 'elapsed_seconds': time.monotonic()-START,
              'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024}
    bounded_write(OUTPUT / 'summary.json', result)
    bounded_write(OUTPUT / 'complete.json', {'status': 'complete', 'summary_sha256': sha(OUTPUT / 'summary.json'), 'elapsed_seconds': time.monotonic()-START, 'rows': 9990})
    print(json.dumps({'status': 'complete', 'output': str(OUTPUT), 'checks': CHECKS, 'elapsed_seconds': time.monotonic()-START}), flush=True)


if __name__ == '__main__':
    main()
