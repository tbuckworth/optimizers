#!/usr/bin/env python3
"""Scalar presentation only. Default/import inert; never follow audit paths."""
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re

HERE=Path(__file__).resolve().parent
DIRECTORY_NAME='2026-09-10-spectral-strong-augmentation'
AUDIT_SCHEMA='spectral_strong_augmentation_audit_v1'
MATPLOTLIB_VERSION='3.10.8'
SEEDS=(202609171,202609172,202609173)
ARMS=(('raw','none'),('native200','none'),('raw','translate'),('native200','translate'))
STEPS=(0,100)+tuple(782*k for k in range(1,73))
EXPOSURES=(0,6400)+tuple(50000*k for k in range(1,73))
LABELS=('Raw AdamW · none','Native r200 · none','Raw AdamW · translation','Native r200 · translation')
SHORT=('Raw\nnone','Native r200\nnone','Raw\ntranslation','Native r200\ntranslation')
COLORS=('#0072B2','#D55E00','#0072B2','#D55E00')
STYLES=('--','--','-','-')
MARKERS=('o','s','^')
CURVES='reporting-learning-curves.png'
SELECTED='fixed-versus-validation-selected.png'
WRONG='wrong-assignment-fit-curves.png'
MANIFEST='plots-manifest.json'
INPUT_CAP,PNG_CAP=64*1024**2,8*1024**2


def require(condition,message):
    if not condition:
        raise ValueError(message)


def sha(payload):
    return hashlib.sha256(payload).hexdigest()


def number(value,name):
    require(type(value) in (int,float) and math.isfinite(value),'nonfinite/invalid '+name)
    return float(value)


def strict_json(payload):
    def pairs(items):
        result={}
        for k,v in items:
            require(k not in result,'duplicate JSON key')
            result[k]=v
        return result
    def reject(value):
        raise ValueError('nonfinite JSON constant')
    return json.loads(payload,object_pairs_hook=pairs,parse_constant=reject)


def validate_metric(row,expected_count=None):
    require(type(row) is dict and set(row)=={'count','correct','accuracy','ce','ce_sum'},'scalar metric schema')
    count,correct=row['count'],row['correct']
    require(type(count) is int and count>0 and type(correct) is int and 0<=correct<=count,'metric counts')
    require(expected_count is None or count==expected_count,'panel count mismatch')
    accuracy,ce,total=(number(row[k],k) for k in ('accuracy','ce','ce_sum'))
    require(0<=accuracy<=1 and ce>=0 and total>=0,'metric domain')
    require(abs(accuracy-correct/count)<=1e-10 and abs(total/count-ce)<=1e-10+1e-10*ce,'inconsistent metric scalars')
    return count


def validate_audit(audit):
    require(type(audit) is dict and audit.get('schema')==AUDIT_SCHEMA,'unexpected audit schema')
    require(audit.get('status')=='PASS','independent audit PASS required')
    for key,value in {'trajectories':12,'logical_evaluation_records':888,
                      'validation_choices_verified_before_reporting':12,
                      'gradient_evaluation_count':675648,'training_example_count':43200000}.items():
        require(type(audit.get(key)) is int and audit[key]==value,'incomplete audit counter: '+key)
    require(audit.get('summary',{}).get('seeds')==list(SEEDS),'summary seeds')
    branches=audit.get('branches')
    require(type(branches) is list and len(branches)==12,'twelve branches required')
    indexed,wrong_counts={},{}
    for branch in branches:
        require(type(branch) is dict and type(branch.get('seed')) is int,'branch schema')
        branch_id=(branch['seed'],branch.get('policy'),branch.get('augmentation'))
        require(branch_id[0] in SEEDS and branch_id[1:] in ARMS and branch_id not in indexed,'duplicate/unknown branch')
        name=f's{branch_id[0]}-{branch_id[1]}-{branch_id[2]}'
        require(branch.get('name')==name and type(branch.get('gradient_evaluation_count')) is int
                and branch['gradient_evaluation_count']==56304 and type(branch.get('training_example_count')) is int
                and branch['training_example_count']==3600000,'branch identity/work')
        rows=branch.get('metrics')
        require(type(rows) is list and len(rows)==74,'all74 states required')
        for step,row in zip(STEPS,rows):
            require(type(row) is dict and type(row.get('step')) is int and row['step']==step,'readout schedule')
            for panel,count in (('train_true',50000),('train_assigned',50000),('validation',5000),('reporting',5000)):
                validate_metric(row.get(panel),count)
            wrong=validate_metric(row.get('wrong_assigned'))
            validate_metric(row.get('wrong_true'),wrong)
            require(0<wrong<50000,'actual wrong-subset count')
            require(branch_id[0] not in wrong_counts or wrong_counts[branch_id[0]]==wrong,'wrong subset count changed')
            wrong_counts[branch_id[0]]=wrong
        selected=branch.get('selection')
        require(type(selected) is dict and all(selected.get(k)==branch[k] for k in ('name','seed','policy','augmentation')),'selection branch binding')
        index=selected.get('selected_index')
        require(type(index) is int and 0<=index<74 and type(selected.get('selected_step')) is int
                and selected['selected_step']==STEPS[index],'selection index/step mismatch')
        require(type(selected.get('selected_example_count')) is int
                and selected['selected_example_count']==EXPOSURES[index],'selected exposure mismatch')
        ce=number(selected.get('validation_ce'),'selected validation CE')
        reference=rows[index]['validation']['ce']
        require(ce>=0 and abs(ce-reference)<=1e-10+1e-10*abs(reference),'selection validation reference mismatch')
        # Independent audit already applied canonical exact ties. Its rebuilt
        # validation CE uses a different FP64 reduction, so only verify tolerance
        # here; never pick a different index, particularly from reporting values.
        require(all(ce<=r['validation']['ce']+1e-10+1e-10*abs(r['validation']['ce']) for r in rows),'selection not validation minimum')
        receipt=selected.get('readout_receipt',{})
        require(type(receipt) is dict and receipt.get('path')==f'logits-{name}-h{STEPS[index]:05d}.npz'
                and type(receipt.get('size_bytes')) is int and 0<receipt['size_bytes']<=16*1024**2
                and type(receipt.get('sha256')) is str and re.fullmatch('[0-9a-f]{64}',receipt['sha256']),'selection evidence receipt')
        indexed[branch_id]=branch
    require(set(indexed)=={(s,p,a) for s in SEEDS for p,a in ARMS},'incomplete paired roster')
    return indexed


def series(indexed,arm,panel,metric):
    return [[float(r[panel][metric]) for r in indexed[s,*arm]['metrics']] for s in SEEDS]


def comparison_values(indexed,window,metric):
    require(window in ('endpoint','selected') and metric in ('accuracy','ce'),'comparison request')
    return [[float(indexed[s,*arm]['metrics'][-1 if window=='endpoint' else
                 indexed[s,*arm]['selection']['selected_index']]['reporting'][metric])
             for arm in ARMS] for s in SEEDS]


def png_bytes(fig,audit_sha,title):
    target=io.BytesIO()
    fig.savefig(target,format='png',dpi=150,facecolor='white',edgecolor='white',
                metadata={'Title':title,'AuditSHA256':audit_sha,
                          'Description':'Audited scalars only; three paired seeds; no reporting-based checkpoint selection.'})
    payload=target.getvalue()
    require(payload.startswith(b'\x89PNG\r\n\x1a\n') and len(payload)<=PNG_CAP,'PNG format/8MiB cap')
    return payload


def render(indexed,audit_sha):
    import matplotlib
    require(matplotlib.__version__==MATPLOTLIB_VERSION,'matplotlib version changed')
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import numpy as np
    output={}
    style={'font.family':'DejaVu Sans','font.size':11,'axes.titlesize':12,'axes.labelsize':11,
           'figure.facecolor':'white','axes.facecolor':'white','axes.spines.top':False,
           'axes.spines.right':False,'axes.axisbelow':True,'axes.edgecolor':'#A6AFB8'}
    with plt.rc_context(style):
        for panel,filename,title,subtitle in (
            ('reporting',CURVES,'Clean reporting performance across all training exposures',
             'Original clean reporting images · Higher accuracy / lower CE is better'),
            ('wrong_assigned',WRONG,'Fit to the fixed wrong assignments',
             'Actually-wrong training subset · Lower target fit is not uniformly beneficial or a safety measure')):
            fig,axes=plt.subplots(1,2,figsize=(13.5,5.6))
            fig.subplots_adjust(left=.075,right=.985,bottom=.18,top=.70,wspace=.25)
            for axis,metric in zip(axes,('accuracy','ce')):
                for arm,color,linestyle,label in zip(ARMS,COLORS,STYLES,LABELS):
                    values=np.asarray(series(indexed,arm,panel,metric),dtype=np.float64)
                    if metric=='accuracy':
                        values*=100
                    for seedline in values:
                        axis.plot(EXPOSURES,seedline,color=color,linestyle=linestyle,alpha=.20,linewidth=.8)
                    axis.plot(EXPOSURES,values.mean(0),color=color,linestyle=linestyle,label=label,linewidth=2.5)
                axis.axvspan(0,6400,color='#697580',alpha=.10)
                axis.axvline(6400,color='#697580',linestyle=':',linewidth=1)
                axis.set(xlim=(0,3600000),xlabel='Training-example exposures (millions)',
                         ylabel='Accuracy (%)' if metric=='accuracy' else 'Cross-entropy (nats)')
                axis.set_xticks((0,900000,1800000,2700000,3600000),('0','0.9','1.8','2.7','3.6'))
                axis.grid(axis='y',color='#CBD2DA',alpha=.55,linewidth=.65)
                axis.set_ylim(0,100) if metric=='accuracy' else axis.set_ylim(bottom=0)
            handles,labels=axes[0].get_legend_handles_labels()
            fig.legend(handles=handles,labels=labels,loc='upper center',bbox_to_anchor=(.52,.89),ncol=2,frameon=False,fontsize=10)
            fig.suptitle(title,y=.995,fontsize=16)
            fig.text(.5,.93,subtitle,ha='center',fontsize=10,color='#46515D')
            fig.text(.5,.055,'Faint: all 3 paired seeds · Bold: means · All 74 states · Warmup: 6,400 exposures (near origin)',
                     ha='center',fontsize=10,color='#56616D')
            output[filename]=png_bytes(fig,audit_sha,title)
            plt.close(fig)
        fig,axes=plt.subplots(2,2,figsize=(13.5,9.2),sharey='row')
        fig.subplots_adjust(left=.085,right=.985,bottom=.14,top=.84,hspace=.34,wspace=.15)
        x=np.arange(4,dtype=float)
        for ri,metric in enumerate(('accuracy','ce')):
            all_values=[]
            for ci,window in enumerate(('endpoint','selected')):
                values=np.asarray(comparison_values(indexed,window,metric),dtype=np.float64)
                if metric=='accuracy':
                    values*=100
                all_values.extend(values.ravel())
                ax=axes[ri,ci]
                for si,(offset,marker) in enumerate(zip((-.15,-.03,.09),MARKERS)):
                    ax.plot(x+offset,values[si],color='#697580',alpha=.35,linewidth=.8)
                    for j,color in enumerate(COLORS):
                        ax.scatter(x[j]+offset,values[si,j],color=color,marker=marker,s=38,zorder=3)
                ax.scatter(x+.24,values.mean(0),color='#17212B',marker='D',s=42,zorder=4)
                ax.set_xticks(x,SHORT,fontsize=9.5)
                ax.set_xlim(-.4,3.5)
                ax.grid(axis='y',color='#CBD2DA',alpha=.55,linewidth=.65)
                if ri==0:
                    ax.set_title('Fixed final: 3.6M exposures' if ci==0 else 'Own validation-CE-selected checkpoint',pad=12)
                if ci==0:
                    ax.set_ylabel('Reporting accuracy (%)' if metric=='accuracy' else 'Reporting cross-entropy (nats)')
            axes[ri,0].set_ylim(0,100) if metric=='accuracy' else axes[ri,0].set_ylim(0,max(all_values)*1.08+.02)
        handles=[Line2D([],[],marker=m,linestyle='none',color='#697580',label='Seed '+str(s)[-3:]) for s,m in zip(SEEDS,MARKERS)]
        handles.append(Line2D([],[],marker='D',linestyle='none',color='#17212B',label='3-seed mean'))
        fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.52,.93),ncol=4,frameon=False)
        fig.suptitle('Clean reporting outcomes: fixed endpoint and validation-only stopping',y=.99,fontsize=16)
        fig.text(.5,.055,'Thin lines connect the same seed across arms · Selected indices are frozen by validation CE, never reporting performance',
                 ha='center',fontsize=10,color='#56616D')
        output[SELECTED]=png_bytes(fig,audit_sha,'Fixed endpoint versus validation-selected reporting outcomes')
        plt.close(fig)
    return output,matplotlib.__version__


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--audit-sha256',help='required exact accepted audit hash when executing')
    args=parser.parse_args(argv)
    if not args.execute:
        print('Prepared only: no input read or output written.')
        return 0
    require(type(args.audit_sha256) is str and re.fullmatch('[0-9a-f]{64}',args.audit_sha256),'accepted --audit-sha256 required')
    require(HERE.name==DIRECTORY_NAME,'assigned report directory required')
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        require(os.environ.get(key)=='1','set '+key+'=1')
    targets=[HERE/name for name in (CURVES,SELECTED,WRONG,MANIFEST)]
    require(all(not p.exists() and not p.is_symlink() for p in targets),'exclusive new outputs required')
    path=HERE/'audit.json'
    require(path.is_file() and not path.is_symlink() and path.stat().st_size<=INPUT_CAP,'regular audit.json within64MiB required')
    with path.open('rb') as handle:
        payload=handle.read(INPUT_CAP+1)
    require(len(payload)<=INPUT_CAP and sha(payload)==args.audit_sha256,'accepted audit hash/size mismatch')
    audit=strict_json(payload)
    indexed=validate_audit(audit)
    # Lexical comparison only: never stat/resolve/open any referenced archive.
    archive=Path(audit.get('input_dir',''))
    require(archive.is_absolute() and '..' not in archive.parts and not HERE.is_relative_to(archive),'outputs must be outside archive')
    images,version=render(indexed,args.audit_sha256)
    manifest={'schema':'spectral_strong_scalar_plots_v1','status':'complete',
              'created_utc':datetime.now(timezone.utc).isoformat(),
              'input':{'path':'audit.json','sha256':sha(payload),'size_bytes':len(payload),'schema':AUDIT_SCHEMA},
              'source_commit':audit.get('source_commit'),'input_results_sha256':audit.get('input_results_sha256'),
              'script_sha256':sha(Path(__file__).read_bytes()),'matplotlib_version':version,'dpi':150,
              'steps':list(STEPS),'training_example_exposures':list(EXPOSURES),'seeds':list(SEEDS),
              'arms':[list(a) for a in ARMS],
              'selected_steps':{b['name']:b['selection']['selected_step'] for b in indexed.values()},
              'outputs':[{'path':n,'size_bytes':len(v),'sha256':sha(v)} for n,v in images.items()],
              'scope':'Scalar presentation only; all3 seeds/all74 states; no referenced paths opened or reporting-based selection.'}
    encoded=(json.dumps(manifest,indent=2,allow_nan=False)+'\n').encode()
    for name,value in images.items():
        with (HERE/name).open('xb') as handle:
            require(handle.write(value)==len(value),'short PNG write')
    with (HERE/MANIFEST).open('xb') as handle:
        require(handle.write(encoded)==len(encoded),'short manifest write')
    print(json.dumps({'status':'complete','input_sha256':sha(payload),'outputs':manifest['outputs']}))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
