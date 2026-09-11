"""Post-outcome explanatory plot; read accepted scalar audit JSON only."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

AUDIT = Path('/tmp/spectral-experiment-artifacts/spectral-observer-pathway-20260910.jTwt14/audit-001/result.json')
EXPECTED = 'd9b05160c9a58952d4f0899699f79de2ee2d919c5319f2bb96d687ea448d0d35'
OUT = Path(__file__).resolve().parents[1] / 'output/2026-09-10-spectral-observer-pathway/results-geometry'
SEEDS = [202609121, 202609122, 202609123]


def main():
    raw = AUDIT.read_bytes()
    if hashlib.sha256(raw).hexdigest() != EXPECTED:
        raise ValueError('Accepted audit bytes changed')
    audit = json.loads(raw)
    if audit['status'] != 'PASS' or audit['errors'] or audit['checks'] != 33032:
        raise ValueError('Complete accepted audit required')
    histories = {(r['seed'], r['cell'], r['schedule']): r for r in audit['checked_histories']}
    expected = {(s, c, h) for s in SEEDS for c in ('clean', 'diffuse')
                for h in ('interleaved', 'grouped')}
    if set(histories) != expected or len(audit['checked_histories']) != 12:
        raise ValueError('Incomplete or duplicated histories')
    data = {}
    for cell in ('clean', 'diffuse'):
        for history in ('interleaved', 'grouped'):
            rows = [histories[(seed, cell, history)]['post_inclusion'] for seed in SEEDS]
            if any(r['basis']['rank'] != 32 or r['geometry']['rare8']['identity_fallback'] for r in rows):
                raise ValueError('Unexpected rank or fallback')
            data[cell + '/' + history] = [100 * r['geometry']['rare8']['native_retention'] for r in rows]
        row = audit['independent_summary']['improvement_contrasts'][cell + '/native_grouped_minus_native_interleaved']['rare_ce']
        if len(row['values']) != 3:
            raise ValueError('Incomplete paired primary')
        data[cell + '/rare_extra_improvement'] = [1000 * v for v in row['values']]

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.9))
    fig.patch.set_facecolor('white')
    colors = ['#087f8c', '#bf7844']
    marks = ['o', 's', '^']
    ax = axes[0]
    for c, cell in enumerate(('clean', 'diffuse')):
        for s, seed in enumerate(SEEDS):
            offset = (s - 1) * .055
            ax.plot([2.6 * c + offset, 2.6 * c + 1 + offset],
                    [data[cell + '/interleaved'][s], data[cell + '/grouped'][s]],
                    color=colors[c], marker=marks[s], markersize=6, linewidth=1.2,
                    label='Parent ' + str(seed)[-3:] if c == 0 else None)
    ax.set_xticks([0, 1, 2.6, 3.6], ['Interleaved', 'Grouped', 'Interleaved', 'Grouped'])
    ax.set_ylim(0, 105)
    ax.set_ylabel('Rare-probe gradient energy passed (%)')
    ax.set_title('Grouping makes a rare direction accessible', loc='left', fontsize=11, fontweight='bold')
    ax.text(.5, 8, 'Correct labels', ha='center', color=colors[0], fontsize=10)
    ax.text(3.1, 8, 'Scattered wrong labels', ha='center', color=colors[1], fontsize=10)
    ax.legend(loc='lower left', bbox_to_anchor=(0, -0.30), ncol=3, frameon=False, fontsize=8)
    ax = axes[1]
    ax.axhline(0, color='#687482', linewidth=1)
    for c, cell in enumerate(('clean', 'diffuse')):
        values = data[cell + '/rare_extra_improvement']
        for s, value in enumerate(values):
            ax.scatter(c + (s - 1) * .06, value, marker=marks[s], color=colors[c], s=45)
        mean = sum(values) / len(values)
        ax.plot([c - .16, c + .16], [mean, mean], color=colors[c], linewidth=3)
    ax.set_xlim(-.45, 1.45)
    ax.set_xticks([0, 1], ['Correct labels', 'Scattered wrong labels'])
    ax.set_ylabel('Extra rare CE improvement (millinats/example)')
    ax.set_title('But does the actual step become more useful?', loc='left', fontsize=11, fontweight='bold')
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=.15)
        ax.tick_params(labelsize=9)
    fig.suptitle('Access to a useful direction is not the same as a useful step',
                 x=.07, ha='left', fontsize=16, fontweight='bold')
    fig.text(.07, .035,
             'Right: grouped minus interleaved; positive favors grouping. All 3 reused, outcome-informed parents.\n'
             'Left: 32-example true-label probe, not the delivered input. One copied Adam step; rare accuracy stays zero.',
             fontsize=9, color='#596673')
    fig.subplots_adjust(left=.075, right=.98, top=.81, bottom=.25, wspace=.32)
    OUT.mkdir(exist_ok=False)
    fig.savefig(OUT / 'access-and-usefulness.png', dpi=150, facecolor='white')
    plt.close(fig)
    with (OUT / 'plotted-data.json').open('x') as f:
        json.dump({'audit_sha256': EXPECTED, 'seeds': SEEDS, 'values': data}, f, indent=2, allow_nan=False)
        f.write('\n')
    files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.iterdir()}
    with (OUT / 'render-receipt.json').open('x') as f:
        json.dump({'status': 'RENDERED_AWAITING_VISUAL_REVIEW', 'audit_sha256': EXPECTED,
                   'renderer_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   'file_sha256': files, 'analysis': 'Post-outcome display of checked scalars only'}, f, indent=2)
        f.write('\n')
    print(json.dumps({'status': 'RENDERED_AWAITING_VISUAL_REVIEW', 'images': 1}))


if __name__ == '__main__':
    main()
