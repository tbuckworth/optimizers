#!/usr/bin/env python3
"""Render independently audited general-augmentation scalar results on CPU.

Import is inert. Reads only audit/results JSON; no model, arrays, or training.
Outputs require a new directory and are never overwritten. No epoch selection,
uncertainty bars, significance tests, or smoothing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

SEEDS = (202609141, 202609142, 202609143)
STEPS = (0, 100) + tuple(range(200, 4001, 200))
CONDITIONS = (('raw', 'none'), ('raw', 'translate'),
              ('native32', 'none'), ('native32', 'translate'))
COLORS = {'raw': '#2471A3', 'native32': '#D97722'}
LABELS = {'raw': 'Raw AdamW', 'native32': 'Spectral + AdamW'}
STYLES = {'none': '--', 'translate': '-'}
JSON_CAP = 16 * 1024**2


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(payload):
    return hashlib.sha256(payload).hexdigest()


def load_json(path):
    require(path.is_file() and not path.is_symlink() and path.stat().st_size <= JSON_CAP,
            'ordinary bounded JSON file required')
    payload = path.read_bytes()
    return json.loads(payload), sha(payload)


def series(branches):
    """Validate and collect scalar plot values; usable with fabricated records."""
    require(type(branches) is list and len(branches) == 12, 'twelve audited branches required')
    indexed = {(r['seed'], r['policy'], r['augmentation']): r for r in branches}
    require(set(indexed) == {(s, p, a) for s in SEEDS for p, a in CONDITIONS}, 'exact paired roster required')
    for row in branches:
        require([m['step'] for m in row['metrics']] == list(STEPS), 'all22 fixed evaluations required')
    result = {}
    for split in ('heldout', 'train'):
        for metric in ('accuracy', 'ce'):
            for policy, augmentation in CONDITIONS:
                values = np.array([[r[split][metric] for r in indexed[s, policy, augmentation]['metrics']]
                                   for s in SEEDS], dtype=np.float64)
                require(values.shape == (3, 22) and np.isfinite(values).all(), 'three finite full seed curves required')
                require((values >= 0).all() and (metric != 'accuracy' or (values <= 1).all()), 'metric domain')
                result[split, metric, policy, augmentation] = values * (100 if metric == 'accuracy' else 1)
    return result


def label(policy, augmentation):
    return LABELS[policy] + (' · translation' if augmentation == 'translate' else ' · no augmentation')


def style_axes(ax, accuracy=False):
    ax.set_facecolor('white')
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis='y', color='#DDE3E9', lw=.7)
    if accuracy:
        ax.set_ylim(0, 100)
    else:
        ax.set_ylim(bottom=0)


def save_png(fig, path):
    fig.patch.set_facecolor('white')
    with path.open('xb') as handle:
        fig.savefig(handle, format='png', dpi=150, facecolor='white')
    plt.close(fig)
    payload = path.read_bytes()
    return {'path': path.name, 'size_bytes': len(payload), 'sha256': sha(payload)}


def render(values, output):
    """Create exactly two PNGs in the caller's newly reserved directory."""
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.titlesize': 13, 'axes.labelsize': 10})
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.subplots_adjust(left=.09, right=.97, bottom=.10, top=.79, hspace=.32, wspace=.25)
    fig.suptitle('Ordinary translation augmentation: complete learning curves',
                 x=.06, y=.97, ha='left', fontsize=19, fontweight='bold')
    fig.text(.06, .925, 'Clean, balanced MNIST · all classes learned from initialization · evaluation images are never augmented',
             fontsize=10.5, color='#435164')
    for row, split in enumerate(('heldout', 'train')):
        for col, metric in enumerate(('accuracy', 'ce')):
            ax = axes[row, col]
            maximum = 0.
            for policy, augmentation in CONDITIONS:
                data = values[split, metric, policy, augmentation]
                for seed_values in data:
                    ax.plot(STEPS, seed_values, color=COLORS[policy], ls=STYLES[augmentation],
                            lw=.8, alpha=.22)
                ax.plot(STEPS, data.mean(axis=0), color=COLORS[policy], ls=STYLES[augmentation],
                        lw=2.3, label=label(policy, augmentation))
                maximum = max(maximum, float(data.max()))
            ax.set_title(('Held-out' if split == 'heldout' else 'Training') +
                         (' accuracy' if metric == 'accuracy' else ' cross-entropy'), loc='left')
            ax.set_xlabel('Training update')
            ax.set_ylabel('Accuracy (%)' if metric == 'accuracy' else 'CE (nats; lower is better)')
            ax.set_xlim(0, 4000)
            ax.set_xticks([0, 1000, 2000, 3000, 4000])
            style_axes(ax, metric == 'accuracy')
            if metric == 'ce':
                ax.set_ylim(0, max(.01, maximum * 1.05))
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc='upper left', bbox_to_anchor=(.05, .90),
               ncol=2, frameon=False, fontsize=10)
    fig.text(.06, .036, 'Heavy lines: three-seed means. Faint lines: every seed. All 22 scheduled evaluations; no smoothing or epoch selection.',
             fontsize=10, color='#435164')
    receipts = [save_png(fig, output / 'curves.png')]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.8))
    fig.subplots_adjust(left=.085, right=.97, bottom=.24, top=.75, wspace=.26)
    fig.suptitle('Held-out performance at the fixed 4,000-update endpoint',
                 x=.06, y=.97, ha='left', fontsize=19, fontweight='bold')
    fig.text(.06, .902, 'Absolute outcomes · bars show means; points show all three paired seeds · no error bars or significance claims',
             fontsize=10.5, color='#435164')
    tick_labels = ['Raw AdamW\nNone', 'Raw AdamW\nTranslation',
                   'Spectral +\nAdamW\nNone', 'Spectral +\nAdamW\nTranslation']
    for ax, metric in zip(axes, ('accuracy', 'ce')):
        maximum = 0.
        for x, (policy, augmentation) in enumerate(CONDITIONS):
            data = values['heldout', metric, policy, augmentation][:, -1]
            ax.bar(x, data.mean(), width=.65, color=COLORS[policy], alpha=.20,
                   edgecolor=COLORS[policy], linewidth=1.1,
                   hatch='//' if augmentation == 'none' else None)
            for offset, marker, value in zip((-.14, 0, .14), ('o', 's', '^'), data):
                ax.scatter(x + offset, value, marker=marker, s=35, color=COLORS[policy],
                           edgecolors='white', linewidths=.45, zorder=3)
            maximum = max(maximum, float(data.max()))
        ax.set_xticks(range(4), tick_labels, fontsize=9)
        ax.set_title('Accuracy' if metric == 'accuracy' else 'Cross-entropy', loc='left')
        ax.set_ylabel('Accuracy (%)' if metric == 'accuracy' else 'CE (nats; lower is better)')
        style_axes(ax, metric == 'accuracy')
        if metric == 'ce':
            ax.set_ylim(0, max(.01, maximum * 1.10))
    fig.text(.06, .075, 'Points, left to right within each bar: seeds 202609141 (circle), 202609142 (square), 202609143 (triangle).',
             fontsize=10, color='#435164')
    fig.text(.06, .035, 'Three paired seeds—not 12 independent replicates. Both training and held-out sets contain 500 examples per digit.',
             fontsize=10, color='#435164')
    receipts.append(save_png(fig, output / 'endpoints.png'))
    return receipts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--audit-json', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        require(os.environ.get(key) == '1', 'set ' + key + '=1 before Python')
    require(not args.output_dir.exists() and not args.output_dir.is_symlink(), 'new exclusive output directory required')
    audit, audit_sha = load_json(args.audit_json)
    require(audit.get('schema') == 'spectral_general_augmentation_audit_v1'
            and audit.get('status') == 'PASS', 'independent PASS audit required')
    input_dir = Path(audit['input_dir']).resolve(strict=True)
    results, result_sha = load_json(input_dir / 'results.json')
    require(result_sha == audit['input_results_sha256'], 'audit/result byte binding differs')
    require(results.get('schema') == 'spectral_general_augmentation_v1'
            and results.get('status') == 'complete', 'completed matching acquisition required')
    require(not args.output_dir.resolve().is_relative_to(input_dir), 'plot outputs must be outside acquisition')
    values = series(audit['branches'])
    args.output_dir.mkdir(exist_ok=False)
    receipts = render(values, args.output_dir)
    manifest = {'scope': 'Audited scalar presentation only; no acquisition, tensor reads, or new audit',
        'audit_path': str(args.audit_json.resolve()), 'audit_sha256': audit_sha,
        'results_path': str(input_dir / 'results.json'), 'results_sha256': result_sha,
        'script_sha256': sha(Path(__file__).read_bytes()), 'seeds': list(SEEDS),
        'endpoint_step': 4000, 'evaluation_steps': list(STEPS), 'images': receipts,
        'scalar_ranges': {'/'.join(key): {'min': float(value.min()), 'max': float(value.max())}
                          for key, value in values.items()}}
    with (args.output_dir / 'plot-manifest.json').open('x') as handle:
        json.dump(manifest, handle, indent=2, allow_nan=False)
    print(json.dumps(manifest, allow_nan=False))


if __name__ == '__main__':
    main()
