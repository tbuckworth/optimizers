"""Plot audited scalar outcomes only; no model or array acquisition."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from experiments.spectral_general_augmentation_plot import (
    COLORS, CONDITIONS, LABELS, STYLES, STEPS, label, load_json, require,
    save_png, style_axes,
)

SEEDS = (202609151, 202609152, 202609153)
SPLITS = ('heldout', 'train_clean', 'wrong_target')
TITLES = {'heldout': 'Clean held-out images (5,000)',
          'train_clean': 'True labels on original training images (5,000)',
          'wrong_target': 'Assigned wrong labels on corrupted training subset (4,000)'}


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--audit-json', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output_dir.exists(), 'new output directory required')
    audit, audit_sha = load_json(args.audit_json)
    require(audit['status'] == 'PASS' and audit['schema'] == 'spectral_wrong_label_augmentation_audit_v1',
            'matching independent PASS audit required')
    source = Path(audit['input_dir']) / 'results.json'
    results, result_sha = load_json(source)
    require(result_sha == audit['input_results_sha256'] and results['status'] == 'complete'
            and results['schema'] == 'spectral_wrong_label_augmentation_v1', 'result binding')
    require(not args.output_dir.resolve().is_relative_to(source.parent.resolve()), 'outside acquisition only')
    rows = audit['branches']
    indexed = {(r['seed'], r['policy'], r['augmentation']): r for r in rows}
    require(len(rows) == 12 and set(indexed) == {(s,p,a) for s in SEEDS for p,a in CONDITIONS}, 'paired roster')
    require(all([m['step'] for m in r['metrics']] == list(STEPS) for r in rows), 'all22 states')
    values = {}
    for split in SPLITS:
        for metric in ('accuracy', 'ce'):
            for p,a in CONDITIONS:
                v = np.array([[m[split][metric] for m in indexed[s,p,a]['metrics']] for s in SEEDS])
                require(v.shape == (3,22) and np.isfinite(v).all() and (v>=0).all()
                        and (metric != 'accuracy' or (v<=1).all()), 'finite metric domain')
                values[split,metric,p,a] = v * (100 if metric == 'accuracy' else 1)
    args.output_dir.mkdir()
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})
    fig, axes = plt.subplots(3,2,figsize=(12,12))
    fig.subplots_adjust(left=.085,right=.97,bottom=.08,top=.84,hspace=.40,wspace=.24)
    fig.suptitle('Augmentation under 80% fixed wrong labels',x=.05,y=.98,ha='left',fontsize=20,fontweight='bold')
    fig.text(.05,.945,'Clean signal and wrong-target fitting are separate outcomes. Evaluation uses original, unaugmented images.',fontsize=10)
    for row,split in enumerate(SPLITS):
        for col,metric in enumerate(('accuracy','ce')):
            ax=axes[row,col]
            maximum=0.
            for p,a in CONDITIONS:
                v=values[split,metric,p,a]
                for curve in v:
                    ax.plot(STEPS,curve,color=COLORS[p],ls=STYLES[a],lw=.8,alpha=.22)
                ax.plot(STEPS,v.mean(0),color=COLORS[p],ls=STYLES[a],lw=2.2,label=label(p,a))
                maximum=max(maximum,float(v.max()))
            ax.axvline(100,color='#8A94A3',lw=.8,alpha=.7)
            ax.set_title(TITLES[split],loc='left',fontsize=10)
            ax.set_xlabel('Training update')
            ax.set_ylabel('Accuracy (%)' if metric=='accuracy' else 'Cross-entropy (nats)')
            ax.set_xlim(0,4000)
            ax.set_xticks([0,1000,2000,3000,4000])
            style_axes(ax,metric=='accuracy')
            if metric=='ce':
                ax.set_ylim(0,max(.01,maximum*1.05))
    handles, labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper left',bbox_to_anchor=(.04,.926),ncol=2,frameon=False)
    fig.text(.05,.035,'Heavy: three-seed mean. Faint: every seed. Vertical line: warmup ends. All22 states; no smoothing or checkpoint selection.',fontsize=9.5)
    receipts=[save_png(fig,args.output_dir/'curves.png')]
    fig,axes=plt.subplots(1,3,figsize=(13,5.7))
    fig.subplots_adjust(left=.065,right=.98,bottom=.25,top=.76,wspace=.30)
    fig.suptitle('Fixed 4,000-update outcomes',x=.045,y=.97,ha='left',fontsize=20,fontweight='bold')
    fig.text(.045,.903,'Bars: means. Points: all three seeds. Wrong-target fitting is not useful learning by itself.',fontsize=11)
    ticks=['AdamW\nNone','AdamW\nTranslation','Spectral\n+ AdamW\nNone','Spectral\n+ AdamW\nTranslation']
    for ax,(split,metric,title) in zip(axes,[('heldout','accuracy','Clean held-out accuracy'),
            ('heldout','ce','Clean held-out loss'),('wrong_target','accuracy','Wrong-target training fit')]):
        maximum=0.
        for x,(p,a) in enumerate(CONDITIONS):
            v=values[split,metric,p,a][:,-1]
            ax.bar(x,v.mean(),width=.65,color=COLORS[p],alpha=.2,edgecolor=COLORS[p],hatch='//' if a=='none' else None)
            for offset,marker,y in zip((-.14,0,.14),('o','s','^'),v):
                ax.scatter(x+offset,y,marker=marker,s=32,color=COLORS[p],edgecolors='white',linewidths=.4,zorder=3)
            maximum=max(maximum,float(v.max()))
        ax.set_xticks(range(4),ticks,fontsize=9)
        ax.set_title(title,loc='left',fontsize=12)
        ax.set_ylabel('Accuracy (%)' if metric=='accuracy' else 'CE (nats; lower is better)')
        style_axes(ax,metric=='accuracy')
        if metric=='ce':
            ax.set_ylim(0,max(.01,maximum*1.1))
    fig.text(.045,.073,'400 wrong labels per true digit; 4,000/5,000 training labels guaranteed wrong. Each label persists across augmented views.',fontsize=10)
    fig.text(.045,.035,'Seeds202609151 (circle), 202609152 (square), 202609153 (triangle). Paired within this study; prior clean runs are context only.',fontsize=9.5)
    receipts.append(save_png(fig,args.output_dir/'endpoints.png'))
    manifest={'scope':'audited scalar presentation only; no new science or model reads',
              'audit_sha256':audit_sha,'results_sha256':result_sha,'images':receipts,
              'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'seeds':list(SEEDS),'steps':list(STEPS)}
    with (args.output_dir/'manifest.json').open('x') as f:
        json.dump(manifest,f,indent=2,allow_nan=False)
    print(json.dumps(manifest))


if __name__=='__main__':
    main()
