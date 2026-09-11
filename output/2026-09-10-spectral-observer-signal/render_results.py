"""Archive accepted scalar output and plot it; never reopen scientific tensors."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
SOURCE = Path('/tmp/spectral-experiment-artifacts/spectral-observer-signal-20260910.PXKMpV/analysis-001')
RESULT_SHA = '20d0514e3e6d5b0453d781957e38ac549c4c8269892657c5b76ee594e3d8cb2c'
COMPLETE_SHA = '5951b22e03308117280efde69bd06a0fea1cb15956bc64aaa2fb58c8c7d683af'
SEEDS = [202609121, 202609122, 202609123]
COLORS = ['#386cb0', '#d07822', '#20826e']


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--presentation-revision', action='store_true')
    revision = parser.parse_args().presentation_revision
    complete_raw = (SOURCE / 'complete.json').read_bytes()
    if digest(complete_raw) != COMPLETE_SHA:
        raise ValueError('Completion pin changed')
    complete = json.loads(complete_raw)
    if complete['status'] != 'complete':
        raise ValueError('No complete analysis')
    for item in complete['receipts']:
        raw = (SOURCE / item['path']).read_bytes()
        if len(raw) != item['size_bytes'] or digest(raw) != item['sha256']:
            raise ValueError('Output receipt mismatch')
    raw = (SOURCE / 'result.json').read_bytes()
    if digest(raw) != RESULT_SHA:
        raise ValueError('Result pin changed')
    result = json.loads(raw)
    if result['status'] != 'PASS' or result['counts']['loaded_archives'] != 21:
        raise ValueError('Incomplete result')
    out = HERE / 'results'
    if revision:
        if not out.is_dir() or (out / 'input-signal-v2.png').exists():
            raise ValueError('Expected original presentation and unused revision')
    else:
        out.mkdir(exist_ok=False)
    for name in ['result.json', 'complete.json', 'manifest.json']:
        if revision:
            if (out / name).read_bytes() != (SOURCE / name).read_bytes():
                raise ValueError('Archived source changed')
        else:
            with (out / name).open('xb') as handle:
                handle.write((SOURCE / name).read_bytes())
    cases = {(case['seed'], case['cell']): case for case in result['cases']}
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.6))
    fields = [('Filtered gradient', 'D_filter', 'Gradient dot-product difference'),
              ('Actual Adam step', 'D_Adam', 'Probe linear-loss utility difference'),
              ('Held-out loss', 'D_U_heldout_ce', 'Cross-entropy improvement difference (nats)')]
    for ax, (title, field, ylabel) in zip(axes, fields):
        for i, (seed, color) in enumerate(zip(SEEDS, COLORS)):
            rows = [next(row for row in cases[seed, cell]['contrasts']
                         if row['primary'] and row['probe'] == 'rare8') for cell in ['clean', 'diffuse']]
            values = [row[field]['dot'] if field == 'D_filter' else row[field]['total']
                      if field == 'D_Adam' else row[field] for row in rows]
            x = [0 + (i-1)*.055, 1 + (i-1)*.055]
            ax.plot(x, values, 'o-', color=color, lw=1.2, ms=6, label=str(seed))
        ax.axhline(0, color='#46505b', lw=.9)
        ax.set_xticks([0, 1], ['Clean labels', 'Noisy labels'])
        ax.set_title(title, fontsize=13)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xlim(-.3, 1.3)
        ax.grid(axis='y', alpha=.18)
        ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle('Rare-example contrast: grouped minus spread-out batches', fontsize=16, y=.98)
    fig.legend(*axes[0].get_legend_handles_labels(), loc='lower center', ncol=3,
               title='Same three reused starting models', frameon=False, bbox_to_anchor=(.5, .045))
    fig.text(.5, .013, 'Positive = more favorable. Separate units: compare signs, not panel heights. No new model run.',
             ha='center', fontsize=10, color='#526273')
    fig.subplots_adjust(left=.08, right=.98, top=.85, bottom=.27, wspace=.4)
    if not revision:
        fig.savefig(out / 'rare-signal-chain.png', dpi=160, facecolor='white')
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for row, (probe, name) in enumerate([('rare8', 'Rare digit'), ('majority', 'Common digits')]):
        for col, (field, label) in enumerate([('dot', 'Input–probe dot product'), ('cosine', 'Input–probe cosine')]):
            ax = axes[row, col]
            for i, (seed, color) in enumerate(zip(SEEDS, COLORS)):
                values = [cases[seed, cell]['probes'][probe]['B'][field] for cell in ['clean', 'diffuse']]
                ax.plot([0+(i-1)*.055, 1+(i-1)*.055], values, 'o-', color=color, lw=1.2, ms=6, label=str(seed))
            ax.axhline(0, color='#46505b', lw=.9)
            ax.set_xticks([0, 1], ['Clean labels', 'Noisy labels'])
            ax.set_xlim(-.3, 1.3)
            ax.set_title(name + ': ' + label, fontsize=12)
            ax.set_ylabel('Gradient inner product' if field == 'dot' else 'Cosine (unitless)')
            ax.grid(axis='y', alpha=.18)
            ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle('Input signal: magnitude and orientation tell different stories', fontsize=15, y=.98)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc='upper center',
               bbox_to_anchor=(.5, .94), ncol=3, frameon=False)
    fig.text(.5, .025, 'Each coloured line follows one reused model. Both groups use correct-label diagnostic probes.\n'
             'These are local gradient alignments, not finite loss improvements or rare-example attribution.',
             ha='center', fontsize=10, color='#526273')
    fig.subplots_adjust(left=.1, right=.97, top=.82, bottom=.14, hspace=.47, wspace=.3)
    fig.savefig(out / ('input-signal-v2.png' if revision else 'input-signal.png'), dpi=160, facecolor='white')
    plt.close(fig)
    if revision:
        receipt = {'status': 'PRESENTATION_REVISION_PASS', 'source_result_sha256': RESULT_SHA,
                   'reason': 'Clarify diagnostic-probe wording and identify every seed; no numerical change.',
                   'file': 'input-signal-v2.png', 'sha256': digest((out / 'input-signal-v2.png').read_bytes())}
        with (out / 'presentation-revision.json').open('x') as handle:
            json.dump(receipt, handle, indent=2)
            handle.write('\n')
        print(json.dumps(receipt))
        return

    table = io.StringIO()
    writer = csv.writer(table)
    writer.writerow(['seed', 'cell', 'probe', 'action', 'B', 'B_cosine', 'F', 'F_cosine', 'K',
                     'J_actual', 'J_over_zero', 'U_heldout_ce', 'heldout_accuracy_change'])
    for case in result['cases']:
        for probe, panel in case['probes'].items():
            for action, row in panel['actions'].items():
                joined = row['accepted_join']
                writer.writerow([case['seed'], case['cell'], probe, action, panel['B']['dot'],
                                 panel['B']['cosine'], row['F']['dot'], row['F']['cosine'], row['K']['dot'],
                                 joined['J']['total'], joined['E_over_zero'], joined['U_heldout_ce'],
                                 joined['heldout_accuracy_change']])
    with (out / 'all-actions.csv').open('x') as handle:
        handle.write(table.getvalue())
    receipt = {'status': 'ARCHIVE_AND_RENDER_PASS', 'source_result_sha256': RESULT_SHA,
               'source_complete_sha256': COMPLETE_SHA,
               'files': {path.name: {'size_bytes': path.stat().st_size, 'sha256': digest(path.read_bytes())}
                         for path in sorted(out.iterdir())}}
    with (out / 'render-receipt.json').open('x') as handle:
        json.dump(receipt, handle, indent=2)
        handle.write('\n')
    print(json.dumps({'status': receipt['status'], 'files': len(receipt['files']),
                      'output': str(out), 'tensor_reads': 0}))


if __name__ == '__main__':
    main()
