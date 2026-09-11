"""Render fixed complete-data plots/tables; no inference, decoding or selection."""
import html
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
WORKER = Path('/tmp/spectral-experiment-artifacts/j-lens-pilot-20260909-MMxZIs/worktree/output/2026-09-11-j-lens-aggregate-projection')
COLORS = ['#64748b', '#087f8c', '#7353ba', '#c1c6cc', '#d49535']
ARMS = ('raw', 'centered', 'second_moment', 'zero', 'fit_mean')
LABELS = ('Raw batch', 'Covariance\nfilter', 'Mean-inclusive\nfilter', 'Zero', 'Fit mean')


def read(path): return json.loads(path.read_text())


def token_table(readouts, family, subset):
    by_name = {r['name']: r for r in readouts}
    names = [(subset+'/raw', 'Raw aggregate'), (subset+'/centered', 'Covariance filter'),
             (subset+'/second_moment', 'Mean-inclusive filter'), ('reference_pooled', 'Independent reference')]
    rows = []
    for suffix, label in names:
        row = by_name[family+'/'+suffix]
        tokens = ' · '.join(html.escape(json.dumps(t, ensure_ascii=False)) for t in row['tokens']) if row['defined'] else 'Undefined zero direction'
        rows.append('<tr><th style="padding:10px;border:1px solid #ddd;text-align:left;vertical-align:top">'+label+
                    '</th><td style="padding:10px;border:1px solid #ddd;line-height:1.7"><code>'+tokens+'</code></td></tr>')
    return '<table style="border-collapse:collapse;width:100%;font-size:13px">'+''.join(rows)+'</table>'


def run():
    results = read(ROOT/'analysis/results.json')['results']
    readouts = read(WORKER/'decoding/readouts.json')['readouts']
    assert read(ROOT/'analysis-audit.json')['status'] == 'PASS'
    assert read(WORKER/'decoding/receipt.json')['status'] == 'complete'
    target = ROOT/'plots'
    target.mkdir()
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    fig.patch.set_facecolor('white')
    for axis, family in zip(axes, ('gradient', 'activation')):
        arms = results[family]['arms']
        raw_error = np.array(arms['raw']['mean_squared_errors'])
        errors = np.array([arms[a]['mean_squared_errors'] for a in ARMS])/raw_error[None, :]
        axis.bar(np.arange(5), errors[:, 2], color=COLORS)
        for i in range(5):
            axis.scatter([i-.07, i+.07], errors[i, :2], color='#202020', s=16, zorder=3)
            axis.text(i, errors[i, 2]+.025, f'{errors[i, 2]:.2f}', ha='center', fontsize=9)
        axis.axhline(1, color='#475569', ls='--', lw=.8)
        axis.set_xticks(np.arange(5), LABELS, fontsize=9)
        axis.set_ylim(0, max(1.18, float(errors.max())*1.14))
        axis.set_ylabel('Reference error / raw-batch error — lower is better')
        axis.set_title('Loss gradients (primary)' if family == 'gradient' else 'Activations (secondary)')
    fig.suptitle('Does filtering recover a more representative aggregate?', fontsize=14)
    fig.text(.5, .01, '16 batches of 16 excerpts; bars: pooled reference. Dots: two reference groups, not confidence intervals.', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .04, 1, .93))
    fig.savefig(target/'aggregate-error.png', dpi=150, facecolor='white'); plt.close(fig)

    by_name = {r['name']: r for r in readouts}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3)); fig.patch.set_facecolor('white')
    for axis, family in zip(axes, ('gradient', 'activation')):
        for i, arm in enumerate(ARMS[:3]):
            vector = results[family]['pooled_query_descriptive'][arm]['cosines'][2]
            lens = by_name[family+'/query_pooled/'+arm]['reference_fidelity']['reference_pooled']['full_logit_cosine']
            ys = [np.nan if v is None else v for v in (vector, lens)]
            axis.plot([0, 1], ys, '-o', label=LABELS[i].replace('\n', ' '), color=COLORS[i], lw=2)
        axis.set_xticks([0, 1], ['Actual vector', 'J-Lens full logits'])
        axis.set_ylabel('Cosine with independent reference — higher is closer')
        axis.set_ylim(-1.05, 1.05); axis.set_xlim(-.2, 1.2)
        axis.set_title('Pooled 256-example gradient' if family == 'gradient' else 'Pooled 256-example activation')
        axis.grid(axis='y', alpha=.15)
    axes[0].legend(loc='lower left', fontsize=9)
    fig.suptitle('Does the readout preserve aggregate fidelity?', fontsize=14)
    fig.text(.5, .01, 'Closeness to a reference is not evidence that the words are semantically useful.', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .04, 1, .93))
    fig.savefig(target/'aggregate-readout.png', dpi=150, facecolor='white'); plt.close(fig)

    for family in ('gradient', 'activation'):
        with (target/(family+'-pooled-tokens.html')).open('x') as f:
            f.write(token_table(readouts, family, 'query_pooled'))
    tables = []
    for family in ('gradient', 'activation'):
        for subset in ['query_pooled']+[f'query{i:02}' for i in range(16)]:
            tables.append('<h3>'+family+' · '+subset+'</h3>'+token_table(readouts, family, subset))
    with (target/'all-token-tables.html').open('x') as f: f.write(''.join(tables))
    print(json.dumps({'status': 'complete', 'plots': ['aggregate-error.png', 'aggregate-readout.png'],
                      'selection': 'all batches; pooled query and batch0 fixed for executive displays'}))


if __name__ == '__main__': run()
