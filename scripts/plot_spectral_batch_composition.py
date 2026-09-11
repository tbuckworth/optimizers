"""Plot the complete independently checked batching study; no model or training."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

SEEDS = (202609121, 202609122, 202609123)
CELLS = ('clean', 'diffuse')
SCHEDULES = ('interleaved', 'grouped')
POLICIES = ('raw', 'native32', 'norm_raw')
LABELS = ('AdamW', 'Spectral', 'Raw direction / own norm rule')
COLORS = ('#68798c', '#087f8c', '#bc6132')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    with path.open('x') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')


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
    if digest(args.audit) != args.expected_audit_sha256:
        raise ValueError('Wrong accepted audit bytes')
    audit = json.loads(args.audit.read_text())
    if (audit['status'] != 'PASS' or audit['errors'] != [] or audit['checks'] <= 0
            or (audit['evaluations_recomputed'], audit['trajectories_checked'],
                audit['native_events_checked']) != (756, 36, 72)):
        raise ValueError('Completed independent audit PASS required')
    acquisition = Path(audit['acquisition_dir'])
    if digest(acquisition/'complete.json') != audit['expected_completion_sha256']:
        raise ValueError('Acquisition completion changed')
    completion = json.loads((acquisition/'complete.json').read_text())
    if (completion['completed_trajectories'], completion['completed_diagnostics']) != (36, 72):
        raise ValueError('Incomplete scientific roster')
    receipts = {r['path']: r for r in audit['input_receipts']}
    steps = list(range(0, 2001, 100))
    curves = {}
    for seed in SEEDS:
        for cell in CELLS:
            for schedule in SCHEDULES:
                for policy in POLICIES:
                    name = f'curve-s{seed}-{cell}-{schedule}-{policy}.json'
                    path = acquisition/name
                    if digest(path) != receipts[name]['sha256'] or path.stat().st_size != receipts[name]['size_bytes']:
                        raise ValueError('Curve provenance mismatch: '+name)
                    record = json.loads(path.read_text())
                    identity = (record['seed'], record['cell'], record['schedule'], record['policy'])
                    if identity != (seed, cell, schedule, policy) or [r['step'] for r in record['curve']] != steps:
                        raise ValueError('Curve identity or step roster mismatch')
                    curves[identity] = record['curve']
    summary = audit['independent_summary']
    if summary['seeds'] != list(SEEDS) or summary['endpoint_step'] != 2000:
        raise ValueError('Summary roster mismatch')
    args.output_dir.mkdir(exist_ok=False)
    data = {'audit_sha256': args.expected_audit_sha256, 'seeds': list(SEEDS),
            'learning': {}, 'schedule_effects': {}}

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.4), sharex=True, sharey=True)
    fig.patch.set_facecolor('white')
    for row, (cell, condition) in enumerate(zip(CELLS, ('Correct labels', 'Scattered wrong labels'))):
        for col, (group, title) in enumerate((('rare', 'Rare digit 8'), ('majority_macro', 'Common digits'))):
            ax = axes[row, col]
            ax.axvspan(0, 100, color='#edf0f4')
            for policy, label, color in zip(POLICIES, LABELS, COLORS):
                for schedule, style in zip(SCHEDULES, ('-', '--')):
                    values = np.array([[100*r['heldout'][group]['accuracy']
                                        for r in curves[seed, cell, schedule, policy]] for seed in SEEDS])
                    if not np.isfinite(values).all():
                        raise ValueError('Nonfinite chart value')
                    key = '/'.join((cell, group, schedule, policy))
                    data['learning'][key] = {'steps': steps, 'seed_accuracy_percent': values.tolist()}
                    for values_seed in values:
                        ax.plot(steps, values_seed, color=color, alpha=.15, linewidth=.7, linestyle=style)
                    ax.plot(steps, values.mean(0), color=color, linewidth=2, linestyle=style,
                            label=label+' · '+schedule)
            ax.set_title(condition+' — '+title, loc='left', fontsize=11, fontweight='bold')
            if col == 0:
                ax.set_ylabel('Held-out accuracy (%)')
            if row == 1:
                ax.set_xlabel('Training updates')
            ax.set_xlim(0, 2000)
            ax.set_ylim(0, 100)
            decorate(ax)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=2, loc='lower center', bbox_to_anchor=(.5, .028), frameon=False, fontsize=9)
    fig.suptitle('Same examples, different batch composition', x=.07, ha='left', fontsize=17, fontweight='bold')
    fig.text(.07, .012, 'Three paired seeds · thick: means; faint: every seed · shaded: common warmup without digit 8',
             fontsize=9, color='#5d6876')
    fig.subplots_adjust(left=.08, right=.98, top=.90, bottom=.20, hspace=.27, wspace=.15)
    fig.savefig(args.output_dir/'learning.png', dpi=150, facecolor='white')
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11.8, 7.7))
    fig.patch.set_facecolor('white')
    for row, (metric, scale, ylabel, direction) in enumerate((
            ('accuracy', 100, 'Accuracy change (percentage points)', 'positive is better'),
            ('ce', 1, 'Cross-entropy change', 'negative is better'))):
        for col, (group, title) in enumerate((('rare', 'Rare digit 8'), ('majority_macro', 'Common digits'))):
            ax = axes[row, col]
            for offset, cell, label, color in zip((-.12, .12), CELLS,
                    ('Correct labels', 'Scattered wrong labels'), ('#167e91', '#b76837')):
                for i, policy in enumerate(POLICIES):
                    key = cell+'/'+policy+'/grouped_minus_interleaved'
                    values = scale*np.array(summary['schedule_contrasts'][key][group+'_'+metric]['values'])
                    data['schedule_effects'][key+'/'+group+'_'+metric] = values.tolist()
                    ax.scatter(i+offset+np.array([-.025, 0, .025]), values, color=color, s=25, alpha=.65,
                               label=label if i == 0 else None)
                    ax.plot(i+offset, values.mean(), marker='_', color=color, markersize=17, markeredgewidth=2.5)
            ax.axhline(0, color='#73818e', linestyle='--', linewidth=.8)
            ax.set_xticks(range(3), ('AdamW', 'Spectral', 'Own-norm\nraw direction'))
            ax.set_xlim(-.5, 2.5)
            ax.set_ylabel(ylabel)
            ax.set_title(title+' — '+direction, loc='left', fontsize=11, fontweight='bold')
            decorate(ax)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=2, loc='lower center', bbox_to_anchor=(.5, .03), frameon=False)
    fig.suptitle('What does grouping change at update 2,000?', x=.07, ha='left', fontsize=17, fontweight='bold')
    fig.text(.07, .012, 'Grouped minus interleaved · dots: every paired seed; bars: means · no selected checkpoint',
             fontsize=9, color='#5d6876')
    fig.subplots_adjust(left=.08, right=.98, top=.89, bottom=.15, hspace=.37, wspace=.28)
    fig.savefig(args.output_dir/'schedule-effects.png', dpi=150, facecolor='white')
    plt.close(fig)

    write_new(args.output_dir/'checked-summary.json', summary)
    write_new(args.output_dir/'chart-data.json', data)
    names = ('learning.png', 'schedule-effects.png', 'checked-summary.json', 'chart-data.json')
    write_new(args.output_dir/'render-receipt.json', {'audit': str(args.audit),
        'audit_sha256': args.expected_audit_sha256, 'source_sha256': digest(Path(__file__)),
        'outputs': {name: digest(args.output_dir/name) for name in names},
        'scope': 'Complete audited curves and fixed endpoint contrasts only; no model, inference or selection.'})
    print(json.dumps({'status': 'rendered', 'output': str(args.output_dir), 'images': 2}))


if __name__ == '__main__':
    main()
