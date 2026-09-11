#!/usr/bin/env python3
"""Independent saved-array audit for the fixed strong augmentation bridge.

Import is inert. No torch, training, image IDX, model replay or unpickling.
Two bounded logit passes keep validation choice verification before reporting
arithmetic without retaining the entire 2.13GB history. Opaque states are
stream-hashed, never retained. No acquisition or automatic retry.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import re
import resource
import time

import numpy as np
from experiments import spectral_general_augmentation_audit as base

ROOT = Path(__file__).resolve().parents[1]
DOCS_REL = Path('output/2026-09-10-spectral-strong-augmentation')
SCHEMA = 'spectral_strong_augmentation_v1'
AUDIT_SCHEMA = 'spectral_strong_augmentation_audit_v1'
UNIT = 'spectral-strong-augmentation-001.service'
SEEDS = (202609171, 202609172, 202609173)
ARMS = (('raw', 'none'), ('native200', 'none'), ('raw', 'translate'), ('native200', 'translate'))
PANELS = ('train', 'validation', 'reporting')
FLOATS = ('loss', 'raw_norm', 'applied_norm', 'data_step_norm')
INTS = ('observer_steps', 'adam_steps', 'basis_rank')
INPUT_CAP, OUTPUT_CAP = 8*1024**3, 64*1024**2
BASE_SHA = '9f2c38c48a9d0dc14c733930a1022fe094caf623cb10a524bf1a262612d2cb15'
OLD_PINS = {
    'spectral_filter.py': '9280c7d360aef4c775baf0d781a5afd56e016e9cfb93ef829c3ae4b8cc3df943',
    'experiments/spectral_general_augmentation.py': 'abffd85aa30f350cc2a6c857758a315d878c27de170c1a4a2adee2f39101538e',
    'experiments/spectral_general_augmentation_data.py': '4404d59fe79e294abc946cd3a58963d1b6e84d65e47d29c33bcebbde7003396a',
    'experiments/spectral_general_augmentation_audit.py': BASE_SHA,
    'output/2026-09-06-spectral-optimizer-investigation/continuation/iteration-009/neural_core.py':
        '706851828b7120bea0d7a41bd0fa75474ac800b2df71087941d0140c8d1970ec',
}
require = base.require


@dataclass(frozen=True)
class Dimensions:
    """Small overrides are for fabricated fixtures only, never CLI options."""
    train: int = 50000
    validation: int = 5000
    reporting: int = 5000
    epochs: int = 72
    batch: int = 64
    warmup: int = 100

    @property
    def batches(self):
        return (self.train+self.batch-1)//self.batch

    @property
    def updates(self):
        return self.batches*self.epochs

    @property
    def eval_steps(self):
        return (0, self.warmup)+tuple(self.batches*i for i in range(1,self.epochs+1))


def required_sources():
    return {*OLD_PINS,
            *{f'experiments/spectral_strong_augmentation{s}.py' for s in ('','_core','_data','_audit')},
            *{f'tests/test_spectral_strong_augmentation{s}.py' for s in ('','_core','_data','_audit')},
            str(DOCS_REL/'protocol.md'), str(DOCS_REL/'implementation-check.md')}


def roster():
    return [{'seed':seed,'policy':p,'augmentation':a} for i,seed in enumerate(SEEDS)
            for p,a in ARMS[i:]+ARMS[:i]]


def array_digest(value):
    value=np.ascontiguousarray(value)
    header=json.dumps([value.dtype.str,list(value.shape)],separators=(',',':'))
    return base.sha(header.encode()+b'\n'+value.tobytes())


def regenerate_plan(raw_labels, seed, dimensions=Dimensions()):
    """Independent implementation of frozen draw order, including unused replacements."""
    d=dimensions
    require(raw_labels.dtype==np.int64 and raw_labels.shape==(d.train+d.validation+d.reporting,)
            and np.all((raw_labels>=0)&(raw_labels<10)), 'source labels schema')
    rngs=[np.random.Generator(np.random.PCG64(np.random.SeedSequence([s,seed]))) for s in range(4)]
    ids=rngs[0].permutation(len(raw_labels)).astype(np.int64)
    train=ids[:d.train].copy()
    validation=ids[d.train:d.train+d.validation].copy()
    reporting=ids[d.train+d.validation:].copy()
    truth=raw_labels[train].copy()
    mask=rngs[1].random(d.train)<.9
    replacement=rngs[1].integers(0,10,size=d.train,dtype=np.int64)
    assigned=np.where(mask,replacement,truth)
    occurrences=np.stack([rngs[2].permutation(d.train).astype(np.int32) for _ in range(d.epochs)])
    shifts=rngs[3].integers(-2,3,size=(d.epochs,d.train,2),dtype=np.int8)
    return {'train_ids':train,'validation_ids':validation,'reporting_ids':reporting,
            'train_labels':truth,'validation_labels':raw_labels[validation].copy(),
            'reporting_labels':raw_labels[reporting].copy(),'corruption_mask':mask,
            'replacement_labels':replacement,'assigned_labels':assigned,
            'changed_mask':assigned!=truth,'occurrences':occurrences,'shifts':shifts,
            'batch_boundaries':np.append(np.arange(0,d.train,d.batch,dtype=np.int64),d.train)}


def validate_inventory(directory, receipts):
    """Separate 8GiB admission; never modify the frozen reader's 1GiB global."""
    require(type(receipts) is list,'receipt list')
    table={}
    for r in receipts:
        require(type(r) is dict and set(r)=={'path','size_bytes','sha256'},'receipt schema')
        name=base.direct_name(r['path'])
        require(name!='results.json' and name not in table,'duplicate/self receipt')
        require(type(r['size_bytes']) is int and r['size_bytes']>=0 and base.is_hash(r['sha256']), 'receipt size/hash')
        table[name]=r
    require(sum(r['size_bytes'] for r in table.values())<=INPUT_CAP,'8GiB input cap')
    children=list(directory.iterdir())
    require(all(p.is_file() and not p.is_symlink() for p in children),'ordinary direct files only')
    require({p.name for p in children}==set(table)|{'results.json'},'exact artifact inventory')
    require(all((directory/n).stat().st_size==r['size_bytes'] for n,r in table.items()),'artifact size')
    return table


def validation_choice(rows):
    """Take ONLY (step,validation-CE) data, never a reporting field."""
    require(type(rows) is list and len(rows)>0,'validation choice rows')
    last=-1
    for row in rows:
        require(type(row) is dict and set(row)=={'step','ce'} and type(row['step']) is int
                and row['step']>last,'ordered validation-only schema')
        require(base.finite_number(row['ce'],'validation CE')>=0,'negative validation CE')
        last=row['step']
    return min(rows,key=lambda r:(r['ce'],r['step']))['step']


def classification(logits, labels):
    """Independent CE route: logaddexp reduction, not producer's shifted exp sum."""
    require(labels.dtype==np.int64 and labels.ndim==1 and len(labels)>0
            and np.all((labels>=0)&(labels<10)), 'metric labels')
    base.checked_array(logits,np.float32,(len(labels),10),'logits')
    values=logits.astype(np.float64)
    losses=np.logaddexp.reduce(values,axis=1)-values[np.arange(len(labels)),labels]
    total=math.fsum(losses.tolist())
    correct=int(np.count_nonzero(np.argmax(values,axis=1)==labels))
    return {'count':len(labels),'correct':correct,'accuracy':correct/len(labels),
            'ce_sum':total,'ce':total/len(labels)}


def canonical_selection_ce(logits, labels):
    """Reconstruct the frozen selector's exact floating-point reduction order.

    This deliberately mirrors the specified shifted-exp/ordered-fsum selector,
    independently of producer code. Alternative equivalent loss formulas can
    reorder exact ties; they remain useful for metric checks, not this argmin.
    """
    base.checked_array(logits,np.float32,(len(labels),10),'selection logits')
    values=logits.astype(np.float64)
    shifted=values-np.max(values,axis=1,keepdims=True)
    losses=np.log(np.exp(shifted).sum(axis=1))-shifted[np.arange(len(labels)),labels]
    return math.fsum(float(v) for v in losses)/len(labels)


def validate_stream(stream, policy, dimensions=Dimensions()):
    d=dimensions
    steps=np.arange(1,d.updates+1,dtype=np.int64)
    native=policy=='native200'
    require(policy in ('raw','native200'),'policy')
    require(np.array_equal(stream['adam_steps'],steps),'Adam counters')
    require(np.array_equal(stream['observer_steps'],steps if native else np.zeros_like(steps)),'observer counters')
    require(np.all((stream['basis_rank']>=0)&(stream['basis_rank']<=200)),'rank cap')
    for key in FLOATS:
        require(np.all(stream[key]>=0),'negative norm')
    if not native:
        require(np.all(stream['basis_rank']==0),'raw observer fields')
    stop=d.warmup if native else d.updates
    require(np.array_equal(stream['raw_norm'][:stop],stream['applied_norm'][:stop]),'unfiltered norm mismatch')
    return len(steps)


def summarize(branches, dimensions=Dimensions()):
    d=dimensions
    indexed={(b['seed'],b['policy'],b['augmentation']):b for b in branches}
    require(len(indexed)==12 and set(indexed)=={(s,p,a) for s in SEEDS for p,a in ARMS},'summary roster')
    groups,contrasts={},{ }
    views=('final','selected','late_mean','warmup_progress')
    for view in views:
        groups[view]={}
        def value(s,p,a,m):
            b=indexed[s,p,a]
            rows={r['step']:r['reporting'] for r in b['metrics']}
            if view=='final':
                return rows[d.updates][m]
            if view=='selected':
                return rows[b['selection']['selected_step']][m]
            if view=='late_mean':
                steps=[d.batches*e for e in range(max(1,d.epochs-11),d.epochs+1)]
                return math.fsum(rows[t][m] for t in steps)/len(steps)
            return (1 if m=='accuracy' else -1)*(rows[d.updates][m]-rows[d.warmup][m])
        for p,a in ARMS:
            groups[view][p+'/'+a]={m:base.estimate([value(s,p,a,m) for s in SEEDS]) for m in ('accuracy','ce')}
        if view=='warmup_progress':
            continue
        pairs=(('native200','translate','raw','translate'),('native200','none','raw','none'),
               ('raw','translate','raw','none'),('native200','translate','native200','none'))
        contrasts[view]={}
        for lp,la,rp,ra in pairs:
            contrasts[view][f'{lp}/{la}-{rp}/{ra}']={m:base.estimate([
                sign*(value(s,lp,la,m)-value(s,rp,ra,m)) for s in SEEDS])
                for m,sign in (('accuracy',1),('ce',-1))}
        contrasts[view]['augmentation_interaction']={m:base.estimate([
            sign*(value(s,'native200','translate',m)-value(s,'native200','none',m)
                  -value(s,'raw','translate',m)+value(s,'raw','none',m)) for s in SEEDS])
            for m,sign in (('accuracy',1),('ce',-1))}
    return {'seeds':list(SEEDS),'groups':groups,'contrasts':contrasts,
            'primary':'fixed final native200/translate versus raw/translate, accuracy and CE',
            'benefit_convention':'accuracy left-right, CE right-left; warmup CE progress warmup-final',
            'selection':'one minimum clean-validation-CE step per trajectory, earliest exact tie',
            'late_epochs':list(range(max(1,d.epochs-11),d.epochs+1)),
            'limits':'three paired seeds, no significance/equivalence/optimality or safety claim'}


def expected_files(dimensions=Dimensions()):
    names={'provenance.json','validation-complete.json','selection.json','selection-verification.json'}
    for seed in SEEDS:
        names.update((f'initial-s{seed}.pt',f'plan-s{seed}.npz',f'plan-s{seed}.json'))
        for p,a in ARMS:
            name=f's{seed}-{p}-{a}'
            names.update((f'warmup-{name}.pt',f'final-{name}.pt',f'stream-{name}.npz'))
            names.update(f'logits-{name}-h{step:05d}.npz' for step in dimensions.eval_steps)
    return names


def readout_specs(d):
    return {'step':(np.int64,(1,)),'train':(np.float32,(d.train,10)),
            'validation':(np.float32,(d.validation,10)),'reporting':(np.float32,(d.reporting,10))}


def check_summary(produced, rebuilt, branches, dimensions):
    """Check producer summary arithmetic without importing the producer."""
    max_error=0.
    def same(left,right):
        nonlocal max_error
        require(type(left) is dict and set(left)=={'values','mean'},'summary estimate schema')
        require(type(left['values']) is list and len(left['values'])==3,'three summary seed values')
        for a,b in zip(left['values']+[left['mean']],right['values']+[right['mean']]):
            error=abs(base.finite_number(a,'summary')-b)
            require(error<=base.ATOL+base.RTOL*abs(b),'summary scalar differs')
            max_error=max(max_error,error)
    require(produced['seeds']==list(SEEDS),'summary seed order')
    indexed={(b['seed'],b['policy'],b['augmentation']):b for b in branches}
    panels=('train_true','train_assigned','wrong_true','wrong_assigned','validation','reporting')
    require(set(produced['groups'])=={p+'/'+a for p,a in ARMS},'summary group roster')
    def metric(b,w,p,m):
        rows={r['step']:r for r in b['metrics']}
        if w=='late61_72':
            steps=[dimensions.batches*e for e in range(max(1,dimensions.epochs-11),dimensions.epochs+1)]
            return math.fsum(rows[t][p][m] for t in steps)/len(steps)
        step={'endpoint':dimensions.updates,'warmup':dimensions.warmup,
              'selected':b['selection']['selected_step']}[w]
        return rows[step][p][m]
    for p,a in ARMS:
        group=produced['groups'][p+'/'+a]
        require(set(group)=={'endpoint','warmup','selected','late61_72'},'summary window roster')
        for w in group:
            require(set(group[w])==set(panels),'summary panels')
            for panel in panels:
                require(set(group[w][panel])=={'accuracy','ce'},'summary metrics')
                for m in ('accuracy','ce'):
                    same(group[w][panel][m],base.estimate([metric(indexed[s,p,a],w,panel,m) for s in SEEDS]))
        for m in ('accuracy','ce'):
            same(produced['reporting_progress_from100'][p+'/'+a][m],rebuilt['groups']['warmup_progress'][p+'/'+a][m])
    pairs=(('native200','translate','raw','translate'),('native200','none','raw','none'),
           ('raw','translate','raw','none'),('native200','translate','native200','none'))
    for lp,la,rp,ra in pairs:
        field=f'{lp}/{la} minus {rp}/{ra}'
        for w in ('endpoint','warmup','selected','late61_72'):
            for m,sign in (('accuracy',1),('ce',-1)):
                expected=base.estimate([sign*(metric(indexed[s,lp,la],w,'reporting',m)
                    -metric(indexed[s,rp,ra],w,'reporting',m)) for s in SEEDS])
                same(produced['reporting_contrasts'][field][w][m],expected)
    for w in ('endpoint','warmup','selected','late61_72'):
        for m,sign in (('accuracy',1),('ce',-1)):
            expected=base.estimate([sign*(metric(indexed[s,'native200','translate'],w,'reporting',m)
                -metric(indexed[s,'native200','none'],w,'reporting',m)
                -metric(indexed[s,'raw','translate'],w,'reporting',m)
                +metric(indexed[s,'raw','none'],w,'reporting',m)) for s in SEEDS])
            same(produced['reporting_augmentation_interaction'][w][m],expected)
    return max_error


def audit_saved(input_dir, raw_labels, *, dimensions=Dimensions(), source_root=ROOT, check=lambda:None):
    started=time.monotonic()
    d=dimensions
    require(0<d.warmup<d.batches and len(set(d.eval_steps))==len(d.eval_steps),'dimension schedule')
    require(base.sha(Path(base.__file__).read_bytes())==BASE_SHA,'frozen reader dependency changed')
    directory=Path(input_dir)
    require(directory.is_dir() and not directory.is_symlink(),'ordinary archive')
    path=directory/'results.json'
    require(path.is_file() and not path.is_symlink() and path.stat().st_size<=base.RESULTS_CAP,'results admission')
    payload=path.read_bytes()
    result=base.strict_json(payload)
    require(result['schema']==SCHEMA and result['status']=='complete','completed source schema')
    require(result['eval_steps']==list(d.eval_steps) and result['roster']==roster(),'fixed schedule/roster')
    require(result['data_pins']==base.DATA_PINS,'pinned data')
    pins=result['source_pins']
    require(type(pins) is dict and set(pins)==required_sources(),'exact source inventory')
    require(all(pins[k]==v for k,v in OLD_PINS.items()),'old source pins')
    for name,expected in pins.items():
        check()
        source=Path(source_root)/name
        require(base.is_hash(expected) and source.is_file() and not source.is_symlink()
                and source.resolve().is_relative_to(Path(source_root).resolve()),'source confinement')
        require(base.sha(source.read_bytes())==expected,'source hash mismatch: '+name)
    table=validate_inventory(directory,result['receipts'])
    require(set(table)==expected_files(d),'fixed artifact inventory')
    require(sum(r['size_bytes'] for r in table.values())+len(payload)<=INPUT_CAP,'total input cap')
    base.check_embedded_receipts(result,table)
    reader=base.Reader(directory,table,check)
    provenance=base.strict_json(reader.load('provenance.json',retain=True))
    require(provenance['schema']==SCHEMA and provenance['source_pins']==pins
            and provenance['data_pins']==base.DATA_PINS and provenance['unit']==UNIT
            and provenance['output_dir']==str(directory.resolve()),'provenance binding')
    require(type(provenance['commit']) is str and re.fullmatch('[0-9a-f]{40}',provenance['commit']) is not None,'commit schema')
    attempt_path=Path(source_root)/DOCS_REL/'attempt.json'
    require(attempt_path.is_file() and not attempt_path.is_symlink()
            and attempt_path.stat().st_size<=base.RESULTS_CAP,'attempt admission')
    attempt_payload=attempt_path.read_bytes()
    attempt=base.strict_json(attempt_payload)
    require(set(attempt)=={'schema','commit','source_pins','data_pins','unit','output_dir','started_utc','pid','argv'}
            and all(provenance[k]==v for k,v in attempt.items()),'attempt/provenance identity')
    require(type(attempt['pid']) is int and attempt['pid']>0 and type(attempt['argv']) is list,'attempt process receipt')
    branches=result['branches']
    require(type(branches) is list and len(branches)==12
            and [{k:b[k] for k in ('seed','policy','augmentation')} for b in branches]==roster(),'ordered branch roster')
    before=base.strict_json(reader.load('validation-complete.json',retain=True))
    require(before['schema']==SCHEMA and before['reporting_metrics_computed'] is False,'validation phase receipt')
    stripped=[]
    for b in branches:
        stripped.append({k:([{q:v for q,v in row.items() if q!='reporting'} for row in value] if k=='metrics' else value)
                         for k,value in b.items() if k!='selection'})
    require(before['branches']==stripped,'pre-report snapshot differs from final branches')
    selection=base.strict_json(reader.load('selection.json',retain=True))
    require(set(selection)=={'schema','rule','reporting_metrics_computed','choices','created_utc'}
            and selection['schema']=='spectral_strong_selection_v1'
            and selection['rule']=='minimum_float64_validation_ce_earliest_exact_tie'
            and selection['reporting_metrics_computed'] is False,'selection schema/rule')
    choices=selection['choices']
    require(type(choices) is list and len(choices)==12,'all12 choices')
    verification=base.strict_json(reader.load('selection-verification.json',retain=True))
    require(verification['status']=='PASS' and verification['selection_receipt']==table['selection.json']
            and verification['verified_choices']==12 and verification['reporting_metrics_computed'] is False,'selection verification binding')
    require(result['selection_receipt']==table['selection.json']
            and result['selection_verification_receipt']==table['selection-verification.json'],'result selection receipts')
    for stamp in (selection['created_utc'],verification['verified_utc'],result['phase_timings']['reporting_started_utc']):
        require(type(stamp) is str and re.fullmatch(r'\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ',stamp) is not None,'phase timestamp')
    require(selection['created_utc']<=verification['verified_utc']<=result['phase_timings']['reporting_started_utc'],'phase timestamp order')
    plans={}
    for seed in SEEDS:
        expected=regenerate_plan(raw_labels,seed,d)
        plan=reader.npz(f'plan-s{seed}.npz',{k:(v.dtype,v.shape) for k,v in expected.items()})
        require(all(np.array_equal(plan[k],v) for k,v in expected.items()),'independent plan reconstruction')
        metadata=base.strict_json(reader.load(f'plan-s{seed}.json',retain=True))
        require(metadata['seed']==seed and metadata['numpy_version']==np.__version__
                and metadata['array_hashes']=={k:array_digest(v) for k,v in plan.items()},'plan metadata binding')
        require(metadata['corruption_selected']==int(plan['corruption_mask'].sum())
                and metadata['actually_wrong']==int(plan['changed_mask'].sum()),'realized corruption counts')
        require(metadata['true_class_counts']=={role:np.bincount(plan[role+'_labels'],minlength=10).tolist()
                for role in PANELS},'true role class counts')
        require(0<int(plan['changed_mask'].sum())<d.train,'nonempty wrong/correct subsets')
        plans[seed]=plan

    max_error=0.
    logit_names=set()
    # FIRST pass: validation-only numerical work; no train/report metric calls.
    for b,choice in zip(branches,choices):
        name=f"s{b['seed']}-{b['policy']}-{b['augmentation']}"
        require(b['name']==name and b['selection']==choice,'branch identity/selection')
        require(type(b['metrics']) is list and len(b['metrics'])==len(d.eval_steps),'readout count')
        validation_rows=[]
        for step,row in zip(d.eval_steps,b['metrics']):
            require(set(row)=={'step','readout_receipt','train_true','train_assigned','wrong_true','wrong_assigned','validation','reporting'}
                    and type(row['step']) is int and row['step']==step,'metric row schema')
            filename=f'logits-{name}-h{step:05d}.npz'
            require(row['readout_receipt']==table[filename],'readout receipt')
            logit_names.add(filename)
            arrays=reader.npz(filename,readout_specs(d))
            require(arrays['step'].tolist()==[step],'readout step')
            metric=classification(arrays['validation'],plans[b['seed']]['validation_labels'])
            max_error=max(max_error,base.compare_metrics(row['validation'],metric))
            choice_ce=canonical_selection_ce(arrays['validation'],plans[b['seed']]['validation_labels'])
            require(choice_ce==row['validation']['ce'],'canonical selection CE differs')
            validation_rows.append({'step':step,'ce':choice_ce})
        selected=validation_choice(validation_rows)
        require(set(choice)=={'name','seed','policy','augmentation','selected_index','selected_step','selected_example_count','validation_ce','readout_receipt'}
                and all(choice[k]==b[k] for k in ('name','seed','policy','augmentation')),'choice branch fields')
        require(type(choice['selected_step']) is int and choice['selected_step']==selected
                and type(choice['selected_index']) is int and choice['selected_index']==d.eval_steps.index(selected),'validation-only minimum/tie differs')
        chosen=b['metrics'][choice['selected_index']]
        exposures=d.batch*selected if selected<=d.warmup else (selected//d.batches)*d.train
        require(type(choice['selected_example_count']) is int and choice['selected_example_count']==exposures,'selected example exposure')
        require(choice['validation_ce']==chosen['validation']['ce']
                and choice['readout_receipt']==chosen['readout_receipt'],'choice evidence binding')
    del arrays
    validation_verified_seconds=time.monotonic()-started

    # SECOND pass: choices already independently validated for all12 arms.
    second=base.Reader(directory,{n:table[n] for n in logit_names},check)
    initial,warmup,initial_predictions,warmup_predictions={},{},{},{}
    rebuilt=[]
    for b in branches:
        check()
        seed,policy,augmentation=b['seed'],b['policy'],b['augmentation']
        name=b['name']
        for field,prefix,suffix in (('warmup_receipt','warmup','.pt'),('final_receipt','final','.pt'),('stream_receipt','stream','.npz')):
            require(b[field]==table[f'{prefix}-{name}{suffix}'],'state/stream receipt binding')
        require(base.is_hash(b['initial_model_sha256']) and base.is_hash(b['warmup_learning_sha256']),'learning digest format')
        require(seed not in initial or initial[seed]==b['initial_model_sha256'],'initial learning pairing')
        initial[seed]=b['initial_model_sha256']
        pair=(seed,augmentation)
        require(pair not in warmup or warmup[pair]==b['warmup_learning_sha256'],'warmup learning pairing')
        warmup[pair]=b['warmup_learning_sha256']
        stream=reader.npz(f'stream-{name}.npz',{**{k:(np.float64,(d.updates,)) for k in FLOATS},
                                              **{k:(np.int64,(d.updates,)) for k in INTS}})
        count=validate_stream(stream,policy,d)
        require(type(b['gradient_evaluation_count']) is int and b['gradient_evaluation_count']==count
                and type(b['training_example_count']) is int and b['training_example_count']==d.train*d.epochs,'branch work counts')
        timing={k:base.finite_number(b[k],k) for k in ('training_seconds','augmentation_seconds','evaluation_seconds','wall_seconds')}
        require(all(v>=0 for v in timing.values()) and sum(timing[k] for k in ('training_seconds','augmentation_seconds','evaluation_seconds'))
                <=timing['wall_seconds']+1e-6,'branch timing accounting')
        rows=[]
        plan=plans[seed]
        mask=plan['changed_mask']
        for step,original in zip(d.eval_steps,b['metrics']):
            arrays=second.npz(f'logits-{name}-h{step:05d}.npz',readout_specs(d))
            for panel in PANELS:
                sha=array_digest(arrays[panel])
                if step==0:
                    key=(seed,panel)
                    require(key not in initial_predictions or initial_predictions[key]==sha,'initial prediction pairing')
                    initial_predictions[key]=sha
                if step==d.warmup:
                    key=(seed,augmentation,panel)
                    require(key not in warmup_predictions or warmup_predictions[key]==sha,'warmup prediction pairing')
                    warmup_predictions[key]=sha
            row={'step':step,'train_true':classification(arrays['train'],plan['train_labels']),
                 'train_assigned':classification(arrays['train'],plan['assigned_labels']),
                 'wrong_true':classification(arrays['train'][mask],plan['train_labels'][mask]),
                 'wrong_assigned':classification(arrays['train'][mask],plan['assigned_labels'][mask]),
                 'validation':classification(arrays['validation'],plan['validation_labels']),
                 'reporting':classification(arrays['reporting'],plan['reporting_labels'])}
            for panel in ('train_true','train_assigned','wrong_true','wrong_assigned','validation','reporting'):
                max_error=max(max_error,base.compare_metrics(original[panel],row[panel]))
            rows.append(row)
        rebuilt.append({'seed':seed,'policy':policy,'augmentation':augmentation,'name':name,'metrics':rows,
                        'selection':b['selection'],'gradient_evaluation_count':count,
                        'training_example_count':d.train*d.epochs,**timing})
    require(type(result['gradient_evaluation_count']) is int and result['gradient_evaluation_count']==12*d.updates
            and type(result['training_example_count']) is int and result['training_example_count']==12*d.epochs*d.train,'total work counts')
    for name in table:
        if name not in reader.read:
            reader.load(name)  # Checkpoint bytes are opaque; never np.load or torch.load.
    require(reader.read==set(table) and second.read==logit_names,'complete bounded read passes')
    summary=summarize(rebuilt,d)
    max_error=max(max_error,check_summary(result['summary'],summary,rebuilt,d))
    return {'schema':AUDIT_SCHEMA,'status':'PASS','input_dir':str(directory.resolve()),
            'input_results_sha256':base.sha(payload),'input_attempt_sha256':base.sha(attempt_payload),
            'source_pins':pins,'data_pins':base.DATA_PINS,'source_commit':provenance['commit'],
            'verified_artifacts':len(reader.read),'logit_artifacts_read_twice':len(second.read),
            'total_artifact_bytes_read':reader.bytes+second.bytes+len(payload),
            'validation_choices_verified_before_reporting':12,'validation_phase_seconds':validation_verified_seconds,
            'trajectories':12,'logical_evaluation_records':12*len(d.eval_steps),
            'gradient_evaluation_count':12*d.updates,'training_example_count':12*d.epochs*d.train,
            'metric_atol':base.ATOL,'metric_rtol':base.RTOL,'max_absolute_scalar_error':max_error,
            'summary':summary,'branches':rebuilt,'wall_seconds':time.monotonic()-started,
            'limits':['Saved-array arithmetic and integrity audit, not a fresh training replication.',
                      'No checkpoint deserialization, model/optimizer/observer replay, or image/test IDX reads.',
                      'Model/Adam digests are producer assertions checked for pairing, not rebuilt states.',
                      'All validation choices verified before reporting arithmetic; stored reporting arrays were not cryptographically blinded.',
                      'Two hash-verified logit passes preserve the2GiB audit memory cap; opaque checkpoints are streamed once.',
                      'Three paired seeds; no semantic selection, optimality, transfer or alignment claim.']}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument('--input-dir',type=Path,required=True)
    parser.add_argument('--output-json',type=Path,required=True)
    args=parser.parse_args(argv)
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        require(os.environ.get(key)=='1','set '+key+'=1')
    source=args.input_dir.resolve(strict=True)
    output=args.output_json.parent.resolve(strict=True)/args.output_json.name
    require(not output.is_relative_to(source) and not output.exists() and not output.is_symlink(),'exclusive audit output outside archive')
    started=time.monotonic()
    def check():
        require(time.monotonic()-started<900,'900s cooperative audit deadline')
        require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<2*1024**3,'2GiB audit memory cap')
    with output.open('xb') as handle:
        try:
            report=audit_saved(source,base.read_fixed_labels(),check=check)
            check()
            payload=(json.dumps(report,allow_nan=False,separators=(',',':'))+'\n').encode()
            require(len(payload)<=OUTPUT_CAP,'64MiB audit output cap')
            handle.write(payload)
        except BaseException as exc:
            handle.write((json.dumps({'schema':AUDIT_SCHEMA,'status':'FAIL','type':type(exc).__name__,
                                     'message':str(exc)[:2000]},allow_nan=False)+'\n').encode())
            raise
    print(json.dumps({'status':'PASS','path':str(output),'size_bytes':len(payload),'sha256':base.sha(payload)}))


if __name__=='__main__':
    main()
