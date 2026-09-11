#!/usr/bin/env python3
"""Reaggregate located external studies without importing their producer code.

This checks arithmetic from saved metrics; it is not an independent training run
or attack replay. Source hashes and metric granularity are recorded explicitly.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from statistics import mean, stdev


OUT = Path(__file__).resolve().parent
ARCHIVE = Path('/private-artifacts/research-archive')
REPOS = Path('/private-artifacts/pyg')
sources = []


def read(path: Path, granularity: str):
    data = path.read_bytes()
    sources.append({'path': str(path), 'sha256': hashlib.sha256(data).hexdigest(),
                    'bytes': len(data), 'granularity': granularity})
    return json.loads(data)


def curve_study(path: Path):
    data = read(path, 'per-model per-radius aggregate and paired C&W aggregates')
    models = data['models']
    assert len(models) == 10
    arms = {}
    for arm in ('control', 'treatment'):
        selected = [m for m in models if m['arm'] == arm]
        assert len(selected) == 5
        grid = sorted(float(x) for x in selected[0]['robust_accuracy'])
        curves = [[m['robust_accuracy'][f'{r:.3f}'] for r in grid] for m in selected]
        for m, values in zip(selected, curves):
            area = sum((y0 + y1) * (x1 - x0) / 2
                       for x0, x1, y0, y1 in zip(grid, grid[1:], values, values[1:]))
            area /= grid[-1] - grid[0]
            assert abs(area - m['normalized_aurac_0_25']) < 1e-12
            assert abs(values[0] - m['clean_accuracy']) < 1e-12
            assert all(a >= b for a, b in zip(values, values[1:]))
        arms[arm] = {
            'clean_accuracy': mean(m['clean_accuracy'] for m in selected),
            'aurac': mean(m['normalized_aurac_0_25'] for m in selected),
            'radius_grid': grid,
            'robust_accuracy': [mean(v[i] for v in curves) for i in range(len(grid))],
        }
    deltas = []
    for pair in range(5):
        c = next(m for m in models if m['arm'] == 'control' and m['pair_index'] == pair)
        s = next(m for m in models if m['arm'] == 'treatment' and m['pair_index'] == pair)
        deltas.append(s['normalized_aurac_0_25'] - c['normalized_aurac_0_25'])
    return {'arms': arms, 'aurac_paired_deltas': deltas,
            'aurac_delta_mean': mean(deltas), 'aurac_delta_sample_sd': stdev(deltas),
            'clean_gap_pp': 100 * (arms['treatment']['clean_accuracy'] - arms['control']['clean_accuracy']),
            'negative_pairs': sum(d < 0 for d in deltas),
            'paired_cw': data['pairs']}


def main():
    cap = ARCHIVE / '2026-08-25-spectral-global-vs-matrix-capacity-followup-draft'
    rank = REPOS / 'research-using-the-spectral-optimiser-existing/followup-rank-epoch-sweep'
    finance = ARCHIVE / '2026-07-31-spectral-optimizer-for-noise-reduction-on-financial-timeseri/experiments/exp-006/out'
    records = {
        'scope': 'Independent arithmetic reaggregation of saved model/era summaries; no retraining or adversarial-pixel replay.',
        'cifar_matrix512': curve_study(cap / 'audit/screen-confirmation-round1/evidence/robustness-terminal/results.json'),
        'cifar_global200': curve_study(rank / 'canonical-results.json'),
    }
    numerai = read(finance / 'summary.json', 'five seed means and 110 per-era paired differences')['comparison']
    per_seed = numerai['per_seed']
    delta = mean(p['spectral'] - p['adamw'] for p in per_seed)
    assert len(per_seed) == 5
    assert len(numerai['per_era']) == 110
    assert abs(delta - numerai['B_minus_A']) < 1e-12
    assert abs(delta - mean(p['mean_difference'] for p in numerai['per_era'])) < 1e-12
    records['numerai_small_mlp'] = {
        'selected_rank': numerai['selected_rank'],
        'adamw': mean(p['adamw'] for p in per_seed),
        'spectral': mean(p['spectral'] for p in per_seed), 'delta': delta,
        'per_seed': per_seed, 'reported_block_bootstrap': numerai['intervals'],
        'bootstrap_status': 'interval settings and saved values read; not recomputed by this script',
        'loss_summary': read(finance / 'loss-curve-summary.json', 'paired late-training mean MSE'),
    }
    em = REPOS / 'em-and-optimisers/reports/spectral'
    judged = read(em / 'lr_derisk/judge_scores.json', '15 checkpoint judge aggregates')
    # Identify score by its exact stored value, not by assumed directory ordering.
    keep94 = [r for r in judged if abs(r['mean_score'] - 68.77) < .01]
    assert len(keep94) == 1
    keep94 = keep94[0]
    ablate94 = read(em / 'ablation/judge_step94.json', 'one checkpoint judge aggregate')[0]
    ablate564 = read(em / 'ablation/judge_step564.json', 'one checkpoint judge aggregate')[0]
    records['em_top8_ablation'] = {
        'selection_note': 'Checkpoint mapping anchored to the report score; separate audit must verify raw-log checkpoint identity.',
        'keep94': keep94, 'ablate94': ablate94, 'ablate564': ablate564,
        'step94_conditional_alignment_delta': ablate94['mean_score'] - keep94['mean_score'],
        'step94_unconditional_misaligned_fraction_delta': ablate94['misaligned_count'] / ablate94['total'] - keep94['misaligned_count'] / keep94['total'],
        'step94_coherence_exclusion_fraction_delta': ablate94['excluded_coherence'] / ablate94['total'] - keep94['excluded_coherence'] / keep94['total'],
        'denominator_caveat': 'These saved aggregates use their producer scoring conventions; CODE/coherence corrections require a separate metric audit.',
    }
    records['sources'] = sources
    target = OUT / 'external-evidence.json'
    target.write_text(json.dumps(records, indent=2) + '\n')
    print(json.dumps({
        'output': str(target), 'sources': len(sources),
        'matrix512_aurac_delta': records['cifar_matrix512']['aurac_delta_mean'],
        'global200_aurac_delta': records['cifar_global200']['aurac_delta_mean'],
        'numerai_delta': delta,
        'em_step94_delta': records['em_top8_ablation']['step94_conditional_alignment_delta'],
    }, indent=2))


if __name__ == '__main__':
    main()
