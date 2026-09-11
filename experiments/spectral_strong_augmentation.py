#!/usr/bin/env python3
"""One guarded strong-regime bridge; reporting metrics follow locked selection."""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments import spectral_general_augmentation as base
from experiments import spectral_general_augmentation_audit as scalar

need,digest=base.need,base.digest
DOCS=ROOT/'output/2026-09-10-spectral-strong-augmentation'
SCHEMA='spectral_strong_augmentation_v1'
UNIT='spectral-strong-augmentation-001.service'
SEEDS=(202609171,202609172,202609173)
CELLS=(('raw','none'),('native200','none'),('raw','translate'),('native200','translate'))
EPOCHS,BATCH,TRAIN,VALIDATION,REPORTING,P=72,64,50000,5000,5000,235146
PER_EPOCH,STEPS=782,56304
EVAL_STEPS=(0,100)+tuple(782*e for e in range(1,73))
MAX_BYTES,RESERVE_BYTES,DEADLINE_SECONDS=8*1024**3,1024**2,10800
FLOAT_STREAMS=('loss','raw_norm','applied_norm','data_step_norm')
INT_STREAMS=('observer_steps','adam_steps','basis_rank')
OLD_PINS={
 'spectral_filter.py':'9280c7d360aef4c775baf0d781a5afd56e016e9cfb93ef829c3ae4b8cc3df943',
 'experiments/spectral_general_augmentation.py':'abffd85aa30f350cc2a6c857758a315d878c27de170c1a4a2adee2f39101538e',
 'experiments/spectral_general_augmentation_data.py':'4404d59fe79e294abc946cd3a58963d1b6e84d65e47d29c33bcebbde7003396a',
 'experiments/spectral_general_augmentation_audit.py':'9f2c38c48a9d0dc14c733930a1022fe094caf623cb10a524bf1a262612d2cb15',
 str(Path(base.core.__file__).relative_to(ROOT)):base.CORE_SHA}


def roster():
    return [{'seed':seed,'policy':p,'augmentation':a} for i,seed in enumerate(SEEDS)
            for p,a in CELLS[i:]+CELLS[:i]]


def branch_name(row):
    return f"s{row['seed']}-{row['policy']}-{row['augmentation']}"


def byte_inventory():
    parts={'27_snapshot_upper':27*(205*P*4+1024**2),
           '888_readout_upper':12*74*((TRAIN+VALIDATION+REPORTING)*10*4+4096),
           'three_plans_upper':128*1024**2,'12_streams_upper':64*1024**2,
           'JSON_receipts_metadata_upper':64*1024**2}
    total=sum(parts.values())
    need(total+RESERVE_BYTES<MAX_BYTES,'8GiB inventory exceeded')
    return {'component_upper_bytes':parts,'total_upper_bytes':total,'cap_bytes':MAX_BYTES}


class Run:
    def __init__(self,path,device='cuda'):
        self.path,self.device=Path(path),device
        self.started,self.last_check=time.monotonic(),0.
        self.used,self.receipts=0,[]
        self.serialization_seconds=0.

    def check(self):
        now=time.monotonic()
        need(now-self.started<DEADLINE_SECONDS,'three-hour cooperative deadline')
        if now-self.last_check>1:
            need(shutil.disk_usage(self.path).free>1024**3,'1GiB free-disk reserve')
            need(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<=16*1024**3,'16GiB host cap')
            if self.device=='cuda':
                need(torch.cuda.max_memory_allocated()<=8*1024**3,'8GiB GPU cap')
            self.last_check=now

    def save(self,name,value,kind='json'):
        started=time.monotonic()
        self.check()
        need(type(name) is str and Path(name).name==name and name not in ('','.','..'),'direct child required')
        need(kind in ('json','npz','tensor'),'artifact kind')
        target=self.path/name
        try:
            with target.open('xb') as handle:
                writer=base.CappedWriter(handle,MAX_BYTES-self.used-RESERVE_BYTES,self.check)
                if kind=='npz':
                    np.savez(writer,**value)
                elif kind=='tensor':
                    torch.save(value,writer)
                else:
                    writer.write((json.dumps(value,allow_nan=False,separators=(',',':'))+'\n').encode())
        except BaseException:
            self.used=sum(p.stat().st_size for p in self.path.iterdir() if p.is_file())
            raise
        receipt={'path':name,'size_bytes':target.stat().st_size,'sha256':digest(target)}
        self.used+=receipt['size_bytes']
        self.receipts.append(receipt)
        self.serialization_seconds+=time.monotonic()-started
        return receipt


def read_receipt(directory,receipt,cap=16*1024**2):
    scalar.direct_name(receipt['path'])
    target=Path(directory)/receipt['path']
    need(not target.is_symlink() and target.is_file() and 0<=receipt['size_bytes']<=cap
         and target.stat().st_size==receipt['size_bytes'],'receipt size/path')
    payload=target.read_bytes()
    need(hashlib.sha256(payload).hexdigest()==receipt['sha256'],'receipt hash mismatch')
    return payload


def load_panel(directory,receipt,panel,count):
    need(panel in ('validation','reporting'),'deferred readback panel')
    payload=read_receipt(directory,receipt)
    with np.load(io.BytesIO(payload),allow_pickle=False) as archive:
        need(set(archive.files)=={'step','train','validation','reporting'},'readout keys')
        need(archive['step'].dtype==np.int64 and archive['step'].shape==(1,),'readout step schema')
        expected=int(Path(receipt['path']).stem.rsplit('-h',1)[1])
        need(int(archive['step'][0])==expected,'readout step/name binding')
        value=archive[panel]  # NPZ lazy access does not decode the other panels.
    scalar.checked_array(value,np.float32,(count,10),panel)
    return value


def choose_validation(records):
    """Only validation CE and receipts enter selection; exact ties go earliest."""
    need(type(records) is list and len(records)==len(EVAL_STEPS),'validation roster')
    for step,row in zip(EVAL_STEPS,records):
        need(set(row)=={'step','validation_ce','readout_receipt'} and type(row['step']) is int
             and row['step']==step,'validation-only schema')
        need(type(row['validation_ce']) is float and math.isfinite(row['validation_ce'])
             and row['validation_ce']>=0,'finite validation CE')
    index=min(range(len(records)),key=lambda i:(records[i]['validation_ce'],records[i]['step']))
    step=records[index]['step']
    return {'selected_index':index,'selected_step':records[index]['step'],
            'selected_example_count':BATCH*step if step<=100 else (step//PER_EPOCH)*TRAIN,
            'validation_ce':records[index]['validation_ce'],'readout_receipt':records[index]['readout_receipt']}


def selection_rows(branches):
    need([{k:b[k] for k in ('seed','policy','augmentation')} for b in branches]==roster(),'all12 branches required')
    return [{'name':b['name'],'seed':b['seed'],'policy':b['policy'],'augmentation':b['augmentation'],
             **choose_validation([{'step':r['step'],'validation_ce':r['validation']['ce'],
                                   'readout_receipt':r['readout_receipt']} for r in b['metrics']])} for b in branches]


def verify_selection(branches,choices,validation_labels,loader):
    """Re-derive every candidate CE from validation arrays, before reporting."""
    need(choices==selection_rows(branches),'selection readback mismatch')
    rebuilt=[]
    for branch,choice in zip(branches,choices):
        records=[]
        for row in branch['metrics']:
            values=loader(row['readout_receipt'],validation_labels[branch['seed']].size)
            stats=scalar.classification(values,validation_labels[branch['seed']])
            need(stats==row['validation'],'saved validation metrics differ')
            records.append({'step':row['step'],'validation_ce':stats['ce'],'readout_receipt':row['readout_receipt']})
        rebuilt.append({k:choice[k] for k in ('name','seed','policy','augmentation')}|choose_validation(records))
    need(rebuilt==choices,'validation-logit selection mismatch')
    return True


def finish_reporting(run,branches,labels):
    """Mechanical phase boundary: all choices locked+verified before metrics."""
    started=time.monotonic()
    run.save('validation-complete.json',{'schema':SCHEMA,'branches':branches,
              'reporting_metrics_computed':False})
    choices=selection_rows(branches)
    selection={'schema':'spectral_strong_selection_v1','rule':'minimum_float64_validation_ce_earliest_exact_tie',
               'reporting_metrics_computed':False,'choices':choices,
               'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
    receipt=run.save('selection.json',selection)
    saved=scalar.strict_json(read_receipt(run.path,receipt))
    need(saved==selection,'selection file readback mismatch')
    def validation_loader(artifact,count):
        run.check()
        return load_panel(run.path,artifact,'validation',count)
    verify_selection(branches,saved['choices'],{s:r['validation'] for s,r in labels.items()},validation_loader)
    verification=run.save('selection-verification.json',{'status':'PASS','selection_receipt':receipt,
                  'verified_choices':12,'reporting_metrics_computed':False,
                  'verified_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})
    timing={'selection_seconds':time.monotonic()-started,
            'reporting_started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
    started=time.monotonic()
    # Every reporting metric call is below the successful all-branch boundary.
    for branch,choice in zip(branches,choices):
        for row in branch['metrics']:
            run.check()
            values=load_panel(run.path,row['readout_receipt'],'reporting',len(labels[branch['seed']]['reporting']))
            row['reporting']=scalar.classification(values,labels[branch['seed']]['reporting'])
        branch['selection']=choice
    need(scalar.strict_json(read_receipt(run.path,receipt))==selection,'selection changed during reporting')
    timing['reporting_seconds']=time.monotonic()-started
    return receipt,verification,timing


def summarize(branches):
    indexed={(b['seed'],b['policy'],b['augmentation']):b for b in branches}
    panels=('train_true','train_assigned','wrong_true','wrong_assigned','validation','reporting')
    groups,progress={},{}
    def at(branch,window,panel,metric):
        rows=branch['metrics']
        if window=='selected':
            return rows[branch['selection']['selected_index']][panel][metric]
        if window=='late61_72':
            return math.fsum(r[panel][metric] for r in rows if r['step']>=61*PER_EPOCH)/12
        return rows[-1 if window=='endpoint' else 1][panel][metric]
    for p,a in CELLS:
        key=p+'/'+a
        groups[key]={w:{panel:{m:scalar.estimate([at(indexed[s,p,a],w,panel,m) for s in SEEDS])
                    for m in ('accuracy','ce')} for panel in panels}
                    for w in ('endpoint','warmup','selected','late61_72')}
        progress[key]={m:scalar.estimate([sign*(at(indexed[s,p,a],'endpoint','reporting',m)
                                  -at(indexed[s,p,a],'warmup','reporting',m)) for s in SEEDS])
                       for m,sign in (('accuracy',1),('ce',-1))}
    pairs=((('native200','translate'),('raw','translate')),(('native200','none'),('raw','none')),
           (('raw','translate'),('raw','none')),(('native200','translate'),('native200','none')))
    contrasts={}
    for left,right in pairs:
        contrasts['/'.join(left)+' minus '+'/'.join(right)]={w:{m:scalar.estimate([
            sign*(at(indexed[s,*left],w,'reporting',m)-at(indexed[s,*right],w,'reporting',m)) for s in SEEDS])
            for m,sign in (('accuracy',1),('ce',-1))} for w in ('endpoint','warmup','selected','late61_72')}
    interaction={w:{m:scalar.estimate([sign*((at(indexed[s,'native200','translate'],w,'reporting',m)
                     -at(indexed[s,'native200','none'],w,'reporting',m))
                     -(at(indexed[s,'raw','translate'],w,'reporting',m)
                     -at(indexed[s,'raw','none'],w,'reporting',m))) for s in SEEDS])
                 for m,sign in (('accuracy',1),('ce',-1))} for w in ('endpoint','warmup','selected','late61_72')}
    return {'seeds':list(SEEDS),'groups':groups,'reporting_progress_from100':progress,'reporting_contrasts':contrasts,
            'reporting_augmentation_interaction':interaction,
            'contrast_sign':'accuracy left-right; CE right-left, positive favorable to left',
            'scope':'three descriptive paired seeds; fixed endpoint and validation-only stopping; no best-reporting selection'}


def record_step(stream,index,row,policy):
    need(set(row)==set(FLOAT_STREAMS+INT_STREAMS),'step schema')
    for key in FLOAT_STREAMS:
        need(type(row[key]) is float and math.isfinite(row[key]) and row[key]>=0,'finite nonnegative step scalar')
        stream[key][index]=row[key]
    for key in INT_STREAMS:
        need(type(row[key]) is int and row[key]>=0,'integer step scalar')
        stream[key][index]=row[key]
    need(row['adam_steps']==index+1 and row['observer_steps']==(index+1 if policy=='native200' else 0),'step clocks')
    need(row['basis_rank']<=200 and (policy!='raw' or row['basis_rank']==0),'basis rank')


def acquire(run):
    from experiments import spectral_strong_augmentation_data as data
    from experiments import spectral_strong_augmentation_core as helper
    images,truth=base.read_training()
    branches,labels,initials,warmups,paired_logits=[], {}, {}, {}, {}
    for seed in SEEDS:
        plan=data.make_plan(truth,seed)
        run.save(f'plan-s{seed}.npz',plan,'npz')
        run.save(f'plan-s{seed}.json',{'seed':seed,'numpy_version':np.__version__,
                 'array_hashes':{k:base.array_digest(v) for k,v in plan.items()},
                 'corruption_selected':int(plan['corruption_mask'].sum()),'actually_wrong':int(plan['changed_mask'].sum()),
                 'true_class_counts':{role:np.bincount(plan[role+'_labels'],minlength=10).tolist()
                                      for role in ('train','validation','reporting')}})
        labels[seed]={'validation':plan['validation_labels'],'reporting':plan['reporting_labels']}
        cpu=images[plan['train_ids']].astype(np.float32)/np.float32(255)
        normalize=data.normalize
        panels={'train':torch.from_numpy(normalize(cpu)).to(run.device),
                'validation':torch.from_numpy(normalize(images[plan['validation_ids']].astype(np.float32)/np.float32(255))).to(run.device),
                'reporting':torch.from_numpy(normalize(images[plan['reporting_ids']].astype(np.float32)/np.float32(255))).to(run.device)}
        for cell in (r for r in roster() if r['seed']==seed):
            run.check()
            name=branch_name(cell)
            base.sync(run.device)
            started=time.monotonic()
            model=helper.make_model(seed,run.device)
            optimizer=helper.make_optimizer(model)
            tracker=helper.make_tracker(model,optimizer) if cell['policy']=='native200' else None
            need(sum(p.numel() for p in model.parameters())==P and not optimizer.state,'fresh canonical model')
            initial=base.core.tree_digest(dict(model.state_dict()))
            if seed not in initials:
                initials[seed]=initial
                run.save(f'initial-s{seed}.pt',helper.snapshot(model,optimizer,tracker),'tensor')
            need(initial==initials[seed],'paired initialization')
            metrics=[]
            stream={**{k:np.zeros(STEPS,dtype=np.float64) for k in FLOAT_STREAMS},
                    **{k:np.zeros(STEPS,dtype=np.int64) for k in INT_STREAMS}}
            training_seconds=augmentation_seconds=evaluation_seconds=0.
            warmup_sha=warmup_receipt=None
            def evaluate(step):
                nonlocal evaluation_seconds
                base.sync(run.device)
                before=time.monotonic()
                values={k:base.predict(model,x) for k,x in panels.items()}
                if step in (0,100):
                    for panel,value in values.items():
                        key=(seed,step,None if step==0 else cell['augmentation'],panel)
                        sha=base.array_digest(value)
                        need(key not in paired_logits or paired_logits[key]==sha,'initial/warmup prediction pairing')
                        paired_logits[key]=sha
                artifact=run.save(f'logits-{name}-h{step:05d}.npz',{'step':np.array([step],dtype=np.int64),**values},'npz')
                mask=plan['changed_mask']
                row={'step':step,'readout_receipt':artifact,
                     'train_true':scalar.classification(values['train'],plan['train_labels']),
                     'train_assigned':scalar.classification(values['train'],plan['assigned_labels']),
                     'wrong_true':scalar.classification(values['train'][mask],plan['train_labels'][mask]),
                     'wrong_assigned':scalar.classification(values['train'][mask],plan['assigned_labels'][mask]),
                     'validation':scalar.classification(values['validation'],plan['validation_labels'])}
                # Reporting logits are archived but never scored in this phase.
                metrics.append(row)
                base.sync(run.device)
                evaluation_seconds+=time.monotonic()-before
            evaluate(0)
            step=0
            for epoch in range(EPOCHS):
                for start,end in zip(plan['batch_boundaries'][:-1],plan['batch_boundaries'][1:]):
                    run.check()
                    ids=plan['occurrences'][epoch,start:end]
                    base.sync(run.device)
                    before=time.monotonic()
                    x=cpu[ids]
                    if cell['augmentation']=='translate':
                        x=data.translate(x,plan['shifts'][epoch,start:end])
                    x=normalize(x)
                    augmentation_seconds+=time.monotonic()-before
                    before=time.monotonic()
                    row=helper.training_update(model,optimizer,tracker,torch.from_numpy(x).to(run.device),
                          torch.from_numpy(plan['assigned_labels'][ids]).to(run.device),policy=cell['policy'])
                    record_step(stream,step,row,cell['policy'])
                    step+=1
                    base.sync(run.device)
                    training_seconds+=time.monotonic()-before
                    if step==100:
                        warmup_sha=base.learning_digest(model,optimizer)
                        key=(seed,cell['augmentation'])
                        need(key not in warmups or warmups[key]==warmup_sha,'paired warmup model/Adam')
                        warmups[key]=warmup_sha
                        warmup_receipt=run.save('warmup-'+name+'.pt',helper.snapshot(model,optimizer,tracker),'tensor')
                    if step in EVAL_STEPS:
                        evaluate(step)
            need(step==STEPS and [r['step'] for r in metrics]==list(EVAL_STEPS),'exact update/readout counts')
            final_receipt=run.save('final-'+name+'.pt',helper.snapshot(model,optimizer,tracker),'tensor')
            stream_receipt=run.save('stream-'+name+'.npz',stream,'npz')
            branches.append({**cell,'name':name,'initial_model_sha256':initial,'warmup_learning_sha256':warmup_sha,
                             'warmup_receipt':warmup_receipt,'final_receipt':final_receipt,'stream_receipt':stream_receipt,
                             'metrics':metrics,'gradient_evaluation_count':step,'training_example_count':EPOCHS*TRAIN,
                             'training_seconds':training_seconds,'augmentation_seconds':augmentation_seconds,
                             'evaluation_seconds':evaluation_seconds,'wall_seconds':time.monotonic()-started})
            print(json.dumps({'completed':name,'count':len(branches),'of':12,'validation':metrics[-1]['validation'],
                              'reporting_metrics_computed':False}),flush=True)
            del model,optimizer,tracker,stream
    return branches,labels


def source_pins():
    paths=list(OLD_PINS)
    paths += [f'experiments/spectral_strong_augmentation{s}.py' for s in ('','_data','_core','_audit')]
    paths += [f'tests/test_spectral_strong_augmentation{s}.py' for s in ('','_data','_core','_audit')]
    paths += [str((DOCS/p).relative_to(ROOT)) for p in ('protocol.md','implementation-check.md')]
    pins={p:digest(ROOT/p) for p in paths}
    need(all(pins[k]==v for k,v in OLD_PINS.items()),'frozen dependency changed')
    return pins


def validate_bounds(effective,service):
    need(effective=={'memory.max':str(16*1024**3),'memory.swap.max':'0','cpu.max':'100000 100000'},'exact cgroup caps')
    need(service=={'Type':'exec','RuntimeMaxUSec':'3h 30min','Restart':'no','KillMode':'control-group'},'exact systemd caps')


def configure():
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        need(os.environ.get(key)=='1','set '+key+'=1')
    need(os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8','deterministic workspace')
    group=next(x.split(':',2)[2] for x in Path('/proc/self/cgroup').read_text().splitlines() if x.startswith('0::'))
    need(Path(group).name==UNIT,'exact service identity')
    cgroup=Path('/sys/fs/cgroup')/group.lstrip('/')
    effective={k:(cgroup/k).read_text().strip() for k in ('memory.max','memory.swap.max','cpu.max')}
    props=subprocess.check_output(['systemctl','--user','show',UNIT,'--property=Type','--property=RuntimeMaxUSec','--property=Restart','--property=KillMode'],text=True)
    service=dict(x.split('=',1) for x in props.splitlines())
    validate_bounds(effective,service)
    clients=base.validate_gpu_clients(subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True))
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    need(torch.__version__=='2.11.0+cu128' and np.__version__=='1.26.4','framework versions')
    need(torch.cuda.is_available() and torch.cuda.get_device_name()=='NVIDIA GeForce RTX 3090','local3090 required')
    free,total=torch.cuda.mem_get_info()
    need(free>=8*1024**3,'8GiB free GPU')
    torch.cuda.set_per_process_memory_fraction(8*1024**3/total)
    return {'cgroup':group,'effective':effective,'service':service,'preexisting_gpu_clients':clients}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args(argv)
    need(args.execute,'No acquisition without explicit --execute')
    parent=args.output_dir.parent.resolve(strict=True)
    need(parent.is_relative_to(Path('/tmp/spectral-experiment-artifacts')) and not list(parent.iterdir()),'unused large-volume parent')
    need(args.output_dir==parent/'acquisition-001' and not args.output_dir.exists() and not args.output_dir.is_symlink(),'exclusive acquisition')
    mount=subprocess.check_output(['findmnt','-n','-o','TARGET,SOURCE','--target',str(parent)],text=True).split()
    need(mount==['/private-artifacts/storage','/dev/RECONFIGURE_FOR_LOCAL_STORAGE'] and shutil.disk_usage(parent).free>=16*1024**3,'mount/16GiB free space')
    pins,inventory=source_pins(),byte_inventory()
    commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    for path,sha in pins.items():
        need(hashlib.sha256(subprocess.check_output(['git','-C',str(ROOT),'show',commit+':'+path])).hexdigest()==sha,'uncommitted source: '+path)
    attempt={'schema':SCHEMA,'commit':commit,'source_pins':pins,'data_pins':base.DATA_PINS,'unit':UNIT,
             'output_dir':str(args.output_dir),'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
             'pid':os.getpid(),'argv':sys.argv}
    with (DOCS/'attempt.json').open('x') as handle:
        json.dump(attempt,handle,indent=2)
    args.output_dir.mkdir()
    run=Run(args.output_dir)
    try:
        run.save('provenance.json',{**attempt,'guards':configure(),'inventory':inventory,
                 'python':sys.version,'torch':torch.__version__,'numpy':np.__version__})
        branches,labels=acquire(run)
        selection,verification,timing=finish_reporting(run,branches,labels)
        summary=summarize(branches)
        need(source_pins()==pins and {k:digest(base.DATA/k) for k in base.DATA_PINS}==base.DATA_PINS,'source/data changed')
        results={'schema':SCHEMA,'status':'complete','source_pins':pins,'data_pins':base.DATA_PINS,'roster':roster(),
                 'eval_steps':EVAL_STEPS,'branches':branches,'selection_receipt':selection,'selection_verification_receipt':verification,
                 'summary':summary,'gradient_evaluation_count':12*STEPS,'training_example_count':12*EPOCHS*TRAIN,
                 'receipts':list(run.receipts),'seconds':time.monotonic()-run.started,
                 'phase_timings':timing,'serialization_seconds':run.serialization_seconds,
                 'gpu_max_allocated_bytes':torch.cuda.max_memory_allocated(),
                 'host_high_water_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
        receipt=run.save('results.json',results)
        print(json.dumps({'status':'complete','result':receipt,'bytes':run.used}),flush=True)
    except BaseException as exc:
        with (args.output_dir/'failure.json').open('x') as handle:
            json.dump({'status':'failed','error_type':type(exc).__name__,'message':str(exc)[:2000]},handle)
        raise


if __name__=='__main__':
    main()
