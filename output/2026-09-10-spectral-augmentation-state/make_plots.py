"""Render audited scalars only. No model, checkpoint or NPZ reads."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
payload = (HERE/'audit.json').read_bytes()
assert hashlib.sha256(payload).hexdigest() == '75dcc35ef970d196425b3285004e99504ead1e485fcf5cb520bc425db7bb25d9'
data = json.loads(payload)
assert data['status'] == 'PASS'
OUT = HERE/'plots'
OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.labelcolor': '#26364a', 'text.color': '#26364a'})
manifest = {'audit_sha256': hashlib.sha256(payload).hexdigest(), 'source': 'audited JSON scalars only', 'plots': {}}


def save(fig, name):
    fig.patch.set_facecolor('white')
    path = OUT/(name+'.png')
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    manifest['plots'][name] = {'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'bytes': path.stat().st_size}


pairs = [('original', 'original'), ('original', 'translate'), ('translate', 'original'), ('translate', 'translate')]
labels = ['Original → original', 'Original → shifted', 'Shifted → original', 'Shifted → shifted']
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharex=True, sharey=True)
styles = [('native_to_raw-minus-raw', '#137f78', 'o', -.10, 'At raw update size'),
          ('native-minus-raw_to_native', '#6854a2', 's', .10, 'At native update size')]
for ax, warmup, title in zip(axes, ('none', 'translate'), ('Unaugmented warmup', 'Augmented warmup')):
    for y, (inputs, objective) in enumerate(pairs):
        row = next(r for r in data['seed_summary'] if r['warmup']==warmup and r['input_mode']==inputs
                   and r['objective']==objective and r['fraction']==1)
        for key, color, marker, offset, label in styles:
            values = np.array(row['contrasts'][key]['ce_improvement']['values'])*1000
            ax.scatter(values, y+offset+np.array([-.035, 0, .035]), s=20, alpha=.55, color=color)
            ax.scatter(values.mean(), y+offset, s=80, marker=marker, color=color, edgecolor='white', linewidth=.8,
                       label=label if y==0 and warmup=='none' else None)
    ax.axvline(0, color='#8c96a3', lw=1)
    ax.axvspan(0, .28, color='#e5f4ef', zorder=-1)
    ax.set_title(title, pad=12, weight='bold')
    ax.set_yticks(range(4), labels)
    ax.grid(axis='x', alpha=.18)
    ax.set_xlim(-.84, .28)
    ax.set_xlabel('Native advantage in loss improvement\n(×0.001 nats; right is better)')
axes[0].invert_yaxis()
axes[0].set_ylabel('Gradient input → held-out readout')
fig.suptitle('Equal update size reveals both useful and costly directions', fontsize=14, weight='bold', y=1.03)
fig.legend(loc='lower center', ncol=2, frameon=False, bbox_to_anchor=(.55,-.04))
fig.text(.5, -.075, 'Large markers: 3-seed means. Small dots: every seed. Full-step readout; four shifted draws averaged within each seed.',
         ha='center', fontsize=8)
fig.tight_layout()
save(fig, 'direction')

fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharey=True)
fields = ('between', 'within', 'original_mean', 'translated_mean')
names = ('Between\nimages', 'Within-image\nview changes', 'Original\nmean gradient', 'Shifted\nmean gradient')
colors = ('#137f78', '#c88337', '#486a99', '#6854a2')
for ax, warmup, title in zip(axes, ('none','translate'), ('Unaugmented warmup', 'Augmented warmup')):
    parents = [p for p in data['parents'] if p['warmup']==warmup]
    for x, (field, color) in enumerate(zip(fields, colors)):
        vals = np.array([p['geometry'][field]['operator_retention']*100 for p in parents])
        ax.bar(x, vals.mean(), width=.65, color=color, alpha=.8)
        ax.scatter(x+np.array([-.10,0,.10]), vals, c='#142636', s=23, edgecolor='white', linewidth=.5)
        ax.text(x, max(vals)+2, f'{vals.mean():.1f}%', ha='center', fontsize=10)
    ax.set_xticks(range(4), names, fontsize=8)
    ax.set_title(title, weight='bold')
    ax.set_ylim(0, 103)
    ax.grid(axis='y', alpha=.18)
axes[0].set_ylabel('Gradient energy retained by the recorded span (%)')
fig.suptitle('Between-image variation is retained more than view changes', fontsize=14, weight='bold', y=1.02)
fig.text(.5, -.025, 'Six fixed states, three seeds. This finite four-view decomposition does not label components as semantic signal or memorization.',
         ha='center', fontsize=8)
fig.tight_layout()
save(fig, 'geometry')

with (OUT/'manifest.json').open('w') as f:
    json.dump(manifest, f, indent=2)
print(json.dumps(manifest))
