"""Package fixed completed action results; no model or probe execution."""
import hashlib
import json
from pathlib import Path
import shutil
import statistics as st

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
BATCH = Path('/tmp/spectral-experiment-artifacts/spectral-grokking-action-20260909.ghvEvD')
OUT = REPO / 'output/2026-09-09-spectral-grokking-action/results'
SHA = 'dc79ddbc9261607468c6adb5c8ff48b3e1249803f4651397aca134bcbc02a7a1'
POLICIES = ('native', 'orthogonal', 'norm_matched')
LABELS = ('Native (archived)', 'Orthogonal', 'Norm-matched orthogonal')
COLORS = ('#6553a1', '#bd7733', '#087d78')
METRICS = (
    ('heldout_cross_entropy', ('behavior','test','loss'), 'Held-out cross-entropy ↓'),
    ('heldout_correct_margin_mean', ('behavior','test','correct_class_margin_mean'), 'Correct-class margin ↑'),
    ('final_hidden_selected_five_heldout_r2', ('probes','final_hidden','selected_eval_mean_r2'), 'Rule-information readout R² ↑'),
)

def value(row, path):
    for key in path:
        row = row[key]
    return row

def main():
    source = BATCH / 'analysis-001/summary.json'
    raw = source.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == SHA
    summary = json.loads(raw)
    assert summary['new_state_count'] == 35
    OUT.mkdir(exist_ok=False)
    for name in ('summary.json', 'complete.json', 'manifest.json'):
        shutil.copyfile(BATCH / 'analysis-001' / name, OUT / name)
    rows = summary['archived_native_reference_rows'] + summary['new_state_rows']
    lookup = {(r['seed'],r['policy'],r['step']):r for r in rows}
    assert len(lookup) == 50
    plt.rcParams.update({'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False})
    fig, axes = plt.subplots(1,3,figsize=(12,3.9),constrained_layout=True)
    for ax, (_,path,label) in zip(axes,METRICS):
        for pol, name, color in zip(POLICIES,LABELS,COLORS):
            means, ses = [], []
            for step in (1500,2000,2500):
                vals = [value(lookup[seed,'native' if step==1500 else pol,step],path) for seed in range(100,105)]
                means.append(st.mean(vals)); ses.append(st.stdev(vals)/5**.5)
            ax.errorbar((1500,2000,2500),means,yerr=ses,marker='o',color=color,label=name,capsize=3)
        ax.set_title(label,fontsize=11); ax.set_xlabel('Absolute optimizer update')
        ax.set_xticks((1500,2000,2500)); ax.grid(alpha=.18)
    axes[0].legend(fontsize=8,frameon=False)
    fig.suptitle('Common step-1500 start · five paired seeds · mean ± sample SE',fontsize=13)
    fig.savefig(OUT/'endpoints.png',dpi=150,facecolor='white'); plt.close(fig)

    fig, axes = plt.subplots(3,5,figsize=(14,7),constrained_layout=True)
    for i,(_,path,label) in enumerate(METRICS):
        for j,seed in enumerate(range(100,105)):
            ax=axes[i,j]
            for pol,name,color in zip(POLICIES,LABELS,COLORS):
                vals=[value(lookup[seed,'native' if step==1500 else pol,step],path) for step in (1500,2000,2500)]
                ax.plot((1500,2000,2500),vals,'o-',color=color,label=name,markersize=3)
            if i==0: ax.set_title(f'Seed {seed}')
            if j==0: ax.set_ylabel(label,fontsize=9)
            if i==2: ax.set_xlabel('Update')
            ax.set_xticks((1500,2500));ax.grid(alpha=.18)
    axes[0,0].legend(fontsize=7,frameon=False)
    fig.suptitle('Every seed, both fixed endpoints · lines do not resolve intervening dynamics')
    fig.savefig(OUT/'all-seeds.png',dpi=150,facecolor='white');plt.close(fig)

    fig,axes=plt.subplots(1,3,figsize=(12,4),constrained_layout=True)
    paths=(('behavior','test','accuracy'),('probes','final_hidden','fixed_panel_eval_mean_r2'),('probes','pre_attention','selected_eval_mean_r2'))
    for ax,path,label in zip(axes,paths,('Held-out accuracy (fraction)','Fixed-panel R² (seed-100 modes)','Pre-attention R² (negative control)')):
        for idx,(pol,name,color) in enumerate(zip(POLICIES,LABELS,COLORS)):
            vals=[value(lookup[seed,pol,2500],path) for seed in range(100,105)]
            ax.scatter([idx+(seed-102)*.035 for seed in range(100,105)],vals,color=color,s=28)
            ax.plot([idx-.17,idx+.17],[st.mean(vals)]*2,color=color,lw=2)
        ax.set_xticks(range(3),('Native\narchived','Orthogonal','Norm-matched'))
        ax.set_title(label,fontsize=10);ax.grid(axis='y',alpha=.18)
    fig.suptitle('Step 2,500 secondary diagnostics · five seed dots; bars show means')
    fig.savefig(OUT/'controls.png',dpi=150,facecolor='white');plt.close(fig)

    table=['| Step | Contrast (left − right) | Δ CE ↓ | Δ margin ↑ | Δ selected R² ↑ |',
           '|---|---|---|---|---|']
    for contrast in summary['paired_endpoint_contrasts']:
        cells=[]
        for metric,_,_ in METRICS:
            v=contrast['metrics'][metric]
            good=v['negative_count'] if v['favorable_direction']=='lower' else v['positive_count']
            cells.append(f"{v['mean_difference']:+.4f} ± {v['sample_se']:.4f}; {good}/5")
        table.append('| '+str(contrast['step'])+' | '+contrast['contrast'].replace('_minus_',' − ').replace('_',' ')+' | '+' | '.join(cells)+' |')
    (OUT/'paired-table.md').write_text('\n'.join(table)+'\n')
    exact=[]
    for step in (2000,2500):
        exact += [f'### Step {step}', '', '| Seed | Policy | CE | Correct margin | Selected R² | Accuracy (%) |', '|---|---|---|---|---|---|']
        for seed in range(100,105):
            for pol in POLICIES:
                r=lookup[seed,pol,step]
                vals=[value(r,path) for _,path,_ in METRICS]
                exact.append(f'| {seed} | {pol} | '+ ' | '.join(f'{v:.6f}' for v in vals)+f" | {100*value(r,('behavior','test','accuracy')):.4f} |")
        exact.append('')
    (OUT/'all-seed-table.md').write_text('\n'.join(exact)+'\n')
    receipt={'source_sha256':SHA,'builder_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             'artifacts':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.iterdir() if p.is_file()}}
    (OUT/'packaging-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({'output':str(OUT),'source_sha256':SHA}))

if __name__=='__main__':
    main()
