"""Render all-seed useful-loss readouts from the accepted independent audit only."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

SEEDS = [202609121, 202609122, 202609123]
CELLS = ('clean', 'diffuse')
ACTIONS = ('native_interleaved', 'native_grouped', 'raw', 'zero')
LABELS = ('Spectral\ninterleaved', 'Spectral\ngrouped', 'Raw\ngradient', 'Zero-gradient\nAdam')
COLORS = ('#6c8ca4', '#087f8c', '#bf7844', '#8a8793')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    with path.open('x') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')


def values(summary, container, key, metric):
    row = summary[container][key][metric]
    data = np.asarray(row['values'], dtype=np.float64)
    if data.shape != (3,) or not np.isfinite(data).all():
        raise ValueError('Missing or invalid full seed roster: ' + key + '/' + metric)
    if not np.isclose(data.mean(), row['mean'], rtol=1e-12, atol=1e-14):
        raise ValueError('Mean differs from saved seeds')
    return data


def decorate(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.axhline(0, color='#687482', linewidth=.8)
    ax.grid(axis='y', alpha=.15)
    ax.tick_params(labelsize=9)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(-3, 3))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--expected-audit-sha256', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if digest(args.audit) != args.expected_audit_sha256:
        raise ValueError('Wrong accepted audit bytes')
    audit = json.loads(args.audit.read_text())
    if audit['status'] != 'PASS' or audit['errors'] != [] or audit['checks'] <= 0:
        raise ValueError('Independent audit PASS required')
    counters = {'cases_checked': 6, 'observer_histories_checked': 12,
                'stream_gradients_checked': 600, 'oracle_gradients_checked': 6,
                'physical_readouts_checked': 21, 'logical_readouts_checked': 24,
                'prediction_pairs_checked': 21}
    if any(audit[key] != expected for key, expected in counters.items()):
        raise ValueError('Incomplete independently checked roster')
    summary = audit['independent_summary']
    fixed = {'seeds': SEEDS, 'adam_before': 100, 'adam_after': 101,
             'native_observer_delivery': 151, 'physical_readouts': 21,
             'logical_case_readouts': 24,
             'primary_contrast': 'native_grouped_minus_native_interleaved',
             'primary_metrics': ['rare_ce', 'majority_macro_ce']}
    for key, expected in fixed.items():
        if summary[key] != expected:
            raise ValueError('Diagnostic identity differs: ' + key)
    required = {cell + '/' + action for cell in CELLS for action in ACTIONS}
    if set(summary['absolute_improvements']) != required:
        raise ValueError('Incomplete action/cell roster')
    for cell in CELLS:
        for action in ACTIONS:
            for metric in ('rare_ce', 'majority_macro_ce'):
                values(summary, 'absolute_improvements', cell + '/' + action, metric)
    args.output_dir.mkdir(exist_ok=False)
    plotted = {'audit_sha256': args.expected_audit_sha256, 'seeds': SEEDS,
               'absolute_heldout_ce_improvements': {}, 'paired_heldout_ce_improvements': {}}

    fig, axes = plt.subplots(2, 2, figsize=(11.8, 7.8))
    fig.patch.set_facecolor('white')
    for row, cell in enumerate(CELLS):
        for col, (metric, group) in enumerate((('rare_ce', 'Rare digit 8'),
                                             ('majority_macro_ce', 'Common digits'))):
            ax = axes[row, col]
            groups = [values(summary, 'absolute_improvements', cell + '/' + action, metric)
                      for action in ACTIONS]
            for seed_idx, offset in enumerate((-.07, 0, .07)):
                ax.plot([offset, 1 + offset], [groups[0][seed_idx], groups[1][seed_idx]],
                        color='#acb9bf', linewidth=1, alpha=.7, zorder=1)
            for x, (action, color, group_values) in enumerate(zip(ACTIONS, COLORS, groups)):
                ax.scatter(x + np.array([-.07, 0, .07]), group_values,
                           s=34, color=color, zorder=3)
                ax.plot([x - .17, x + .17], [group_values.mean()] * 2,
                        color=color, linewidth=3, zorder=4)
                plotted['absolute_heldout_ce_improvements'][cell + '/' + action + '/' + metric] = group_values.tolist()
            ax.set_xticks(range(4), LABELS)
            ax.set_title(('Correct labels' if cell == 'clean' else 'Scattered wrong labels')
                         + ' — ' + group, loc='left', fontsize=11, fontweight='bold')
            ax.set_ylabel('CE before − after (nats/example)')
            decorate(ax)
    fig.suptitle('Does each single step actually help?', x=.08, ha='left',
                 fontsize=17, fontweight='bold')
    fig.text(.08, .025,
             'Positive = lower held-out loss. Dots: all 3 reused, outcome-informed parents; bars: means.\n'
             'Same parent model/Adam. Native and raw use the common block mean; zero delivers zero. One copied step each.',
             fontsize=9, color='#596673')
    fig.subplots_adjust(left=.085, right=.98, top=.90, bottom=.17, hspace=.42, wspace=.28)
    fig.savefig(args.output_dir / 'absolute-usefulness.png', dpi=150, facecolor='white')
    plt.close(fig)

    comparisons = ('native_grouped_minus_native_interleaved', 'native_grouped_minus_raw',
                   'native_grouped_minus_zero')
    comparison_labels = ('Grouped −\ninterleaved', 'Grouped −\nraw', 'Grouped −\nzero-gradient Adam')
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 7.8))
    fig.patch.set_facecolor('white')
    for row, cell in enumerate(CELLS):
        for col, (metric, group) in enumerate((('rare_ce', 'Rare digit 8'),
                                             ('majority_macro_ce', 'Common digits'))):
            ax = axes[row, col]
            for x, comparison in enumerate(comparisons):
                data = values(summary, 'improvement_contrasts', cell + '/' + comparison, metric)
                color = COLORS[1] if x == 0 else '#7a8895'
                ax.scatter(x + np.array([-.07, 0, .07]), data, color=color, s=34)
                ax.plot([x - .17, x + .17], [data.mean()] * 2, color=color, linewidth=3)
                plotted['paired_heldout_ce_improvements'][cell + '/' + comparison + '/' + metric] = data.tolist()
            ax.set_xticks(range(3), comparison_labels)
            ax.set_title(('Correct labels' if cell == 'clean' else 'Scattered wrong labels')
                         + ' — ' + group, loc='left', fontsize=11, fontweight='bold')
            ax.set_ylabel('Extra CE improvement (nats/example)')
            decorate(ax)
    fig.suptitle('What does grouped history add?', x=.08, ha='left',
                 fontsize=17, fontweight='bold')
    fig.text(.08, .025,
             'Positive = grouped history is relatively better. A relative benefit can still be reduced damage.\n'
             'Three reused, outcome-informed parents; all paired values shown. Local observer-history test, not long-run recognition.',
             fontsize=9, color='#596673')
    fig.subplots_adjust(left=.085, right=.98, top=.90, bottom=.17, hspace=.42, wspace=.28)
    fig.savefig(args.output_dir / 'paired-effects.png', dpi=150, facecolor='white')
    plt.close(fig)
    save_json(args.output_dir / 'checked-summary.json', summary)
    save_json(args.output_dir / 'plotted-data.json', plotted)
    paths = sorted(path for path in args.output_dir.iterdir() if path.is_file())
    save_json(args.output_dir / 'render-receipt.json', {
        'status': 'RENDERED_AWAITING_VISUAL_REVIEW', 'audit_sha256': args.expected_audit_sha256,
        'renderer_sha256': digest(__file__),
        'file_sha256': {path.name: digest(path) for path in paths}})
    print(json.dumps({'status': 'RENDERED_AWAITING_VISUAL_REVIEW', 'images': 2}))


if __name__ == '__main__':
    main()
