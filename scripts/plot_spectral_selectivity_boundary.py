"""Render hash-bound, independently checked selectivity results; no model replay."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

CELLS = ('clean', 'diffuse', 'shared', 'sham')
POLICIES = ('raw', 'native32', 'norm_raw')
LABELS = ('AdamW', 'Spectral', 'Raw direction (own norm rule)')
COLORS = ('#677789', '#137e91', '#bc693a')
SEEDS = (202609111, 202609112, 202609113)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def decorate(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', alpha=.15)
    ax.tick_params(labelsize=9)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--expected-audit-sha256', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    assert digest(args.audit) == args.expected_audit_sha256
    audit = json.loads(args.audit.read_text())
    assert audit['status'] == 'PASS'
    assert audit['independent']['evaluations_recomputed'] == 756
    assert audit['independent']['native_events_checked'] == 36
    acquisition = Path(audit['acquisition_dir'])
    assert digest(acquisition / 'complete.json') == audit['expected_completion_sha256']
    receipts = {r['path']: r for r in audit['input_receipts']}
    curves = {}
    for seed in SEEDS:
        for cell in CELLS:
            for policy in POLICIES:
                name = f'curve-s{seed}-{cell}-{policy}.json'
                path = acquisition / name
                assert digest(path) == receipts[name]['sha256']
                assert path.stat().st_size == receipts[name]['size_bytes']
                record = json.loads(path.read_text())
                assert (record['seed'], record['cell'], record['policy']) == (seed, cell, policy)
                assert [r['step'] for r in record['curve']] == list(range(0, 2001, 100))
                curves[seed, cell, policy] = record['curve']
    args.output_dir.mkdir(exist_ok=False)
    chart_data = {'audit_sha256': args.expected_audit_sha256, 'seeds': list(SEEDS),
                  'learning': {}, 'cue': {}, 'interactions': {}}
    fig, axes = plt.subplots(2, 4, figsize=(13, 7.3), sharex=True, sharey=True)
    fig.patch.set_facecolor('white')
    steps = list(range(0, 2001, 100))
    for column, (cell, title) in enumerate(zip(CELLS, ('Correct labels', 'Scattered wrong labels',
                                                       'Shared wrong cue', 'Matched sham cue'))):
        for row, (group, caption) in enumerate((('majority_macro', 'Common digits'), ('rare', 'Rare digit 8'))):
            ax = axes[row, column]
            ax.axvspan(0, 100, color='#edf0f4')
            for policy, label, color in zip(POLICIES, LABELS, COLORS):
                values = np.array([[100*r['heldout_unpatched'][group]['accuracy']
                                    for r in curves[seed, cell, policy]] for seed in SEEDS])
                mean = values.mean(axis=0)
                chart_data['learning'][f'{cell}/{group}/{policy}'] = {
                    'steps': steps, 'seed_accuracy_percent': values.tolist(), 'mean_accuracy_percent': mean.tolist()}
                for values_seed in values:
                    ax.plot(steps, values_seed, color=color, alpha=.2, linewidth=.8)
                ax.plot(steps, mean, color=color, linewidth=2.1, label=label)
            if row == 0:
                ax.set_title(title, fontsize=10.5, fontweight='bold', pad=10)
            else:
                ax.set_xlabel('Training updates', fontsize=10)
            if column == 0:
                ax.set_ylabel(caption+'\nheld-out accuracy (%)', fontsize=10)
            ax.set_xlim(0, 2000)
            ax.set_ylim(0, 100)
            decorate(ax)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.5, .035), ncol=3, frameon=False, fontsize=10)
    fig.suptitle('Does filtering preserve useful learning—including a new rare class?',
                 x=.06, ha='left', fontsize=16, fontweight='bold')
    fig.text(.06, .015, 'Three paired seeds · thick: mean; faint: every seed · shaded: common warmup without digit 8',
             fontsize=9, color='#5d6876')
    fig.subplots_adjust(left=.075, right=.99, bottom=.16, top=.88, hspace=.17, wspace=.12)
    fig.savefig(args.output_dir/'learning.png', dpi=150, facecolor='white')
    plt.close(fig)

    summary = audit['independent']['summary']
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.8))
    fig.patch.set_facecolor('white')
    for i, (policy, label, color) in enumerate(zip(POLICIES, LABELS, COLORS)):
        xs = np.arange(4) + (i-1)*.21
        for x, cell in zip(xs, CELLS):
            values = np.array(summary['per_group'][cell+'/'+policy]['majority_nonzero_patch_excess']['values'])*100
            chart_data['cue'][cell+'/'+policy] = values.tolist()
            axes[0].scatter(x+np.array([-.035, 0, .035]), values, s=17, color=color, alpha=.5)
            axes[0].plot(x, values.mean(), marker='_', markersize=15, markeredgewidth=2.5, color=color,
                         label=label if cell == 'clean' else None)
    axes[0].set_xticks(range(4), ['Clean', 'Diffuse', 'Shared', 'Sham'])
    axes[0].set_ylabel('Patch-induced target-0 bias (percentage points)')
    axes[0].set_title('A shared cue, beyond ordinary target bias', loc='left', fontsize=11, fontweight='bold')
    axes[0].legend(frameon=False, fontsize=8, loc='best')
    for x, policy, color in zip((0, 1), ('native32', 'norm_raw'), COLORS[1:]):
        key = policy+'_minus_raw/majority_nonzero_patch_excess'
        values = np.array(summary['cue_interactions'][key]['values'])*100
        chart_data['interactions'][key] = values.tolist()
        axes[1].scatter(x+np.array([-.045, 0, .045]), values, s=32, color=color, alpha=.65)
        axes[1].plot(x, values.mean(), marker='_', markersize=22, markeredgewidth=3, color=color)
    axes[1].set_xticks([0, 1], ['Spectral − AdamW', 'Norm-control − AdamW'])
    axes[1].set_xlim(-.5, 1.5)
    axes[1].set_ylabel('Shared-minus-sham interaction (points)')
    axes[1].set_title('Does cue association affect filtering differently?', loc='left', fontsize=11, fontweight='bold')
    for ax in axes:
        decorate(ax)
        ax.axhline(0, color='#7b8590', linewidth=.8, linestyle='--')
    fig.suptitle('Misleading-cue sensitivity at the fixed endpoint', x=.065, ha='left', fontsize=16, fontweight='bold')
    fig.text(.065, .025, '4,000 matched images within each seed · patched minus unpatched · dots: paired seeds; bars: means',
             fontsize=8.6, color='#5d6876')
    fig.subplots_adjust(left=.075, right=.985, bottom=.19, top=.80, wspace=.32)
    fig.savefig(args.output_dir/'cue.png', dpi=150, facecolor='white')
    plt.close(fig)
    write_new(args.output_dir/'chart-data.json', chart_data)
    write_new(args.output_dir/'checked-summary.json', summary)
    write_new(args.output_dir/'render-receipt.json', {
        'audit': str(args.audit), 'audit_sha256': args.expected_audit_sha256,
        'source_sha256': digest(Path(__file__)),
        'outputs': {name: digest(args.output_dir/name) for name in
                    ('learning.png', 'cue.png', 'chart-data.json', 'checked-summary.json')},
        'scope': 'Rendering of fixed audited curves/contrasts only; no models, selector or new experiment.'})
    print(json.dumps({'status': 'rendered', 'output_dir': str(args.output_dir)}))


if __name__ == '__main__':
    main()
