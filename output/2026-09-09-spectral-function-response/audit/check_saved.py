#!/usr/bin/env python3
"""Independent saved arrays/state equations only; never creates a model."""
import argparse
import gc
import hashlib
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

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2]
PARENT=Path('/tmp/spectral-experiment-artifacts/spectral-grokking-function-response-20260909.PegeVa')
SOURCE=PARENT/'diagnostic-001';OUT=PARENT/'audit-001'
RAW=Path('/tmp/spectral-experiment-artifacts/spectral-grokking-raw-direction-20260909.KzGtkl')
RAW_SHA='8c0040c193a1f7d755004b002a3da70f315db3e41a959a5789d15844ad5b6f3b'
MEASURE_SHA='ecbb06431733302f6d6b61b8770c781076676b9ca65f244e2bd9bc467d681d46'
SCHEMA='grokking_function_response_measurement_v1'
SEEDS=tuple(range(100,105));NAMES=('before','raw','trunc','projected','zero');NEW=('trunc','projected','zero')
CONTRASTS=(('raw','trunc'),('trunc','projected'),('raw','projected'),('zero','raw'),('zero','trunc'),('zero','projected'))
EPS32=2**-23;TINY32=2**-126


def sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024**2),b''):value.update(block)
    return value.hexdigest()


class Check:
    def __init__(self):self.start=time.monotonic();self.count=0;self.errors=[];self.inputs={};self.max_error=0.
    def require(self,ok,label):
        self.count+=1
        if not ok:self.errors.append(label);raise ValueError(label)
        if time.monotonic()-self.start>280:raise TimeoutError('280-second audit limit')
    def close(self,a,b,label,rtol=1e-9,atol=1e-10):
        a,b=float(a),float(b);self.require(math.isfinite(a) and math.isfinite(b),label+' finite')
        error=abs(a-b);self.max_error=max(error,self.max_error)
        self.count+=1
        if not math.isclose(a,b,rel_tol=rtol,abs_tol=atol):self.errors.append(f'{label}: {a} != {b}')
    def receipt(self,r,expected=None):
        path=Path(r['path'])
        self.require(path.is_file() and not path.is_symlink() and path.stat().st_nlink==1,'Regular singly-linked '+str(path))
        if expected is not None:self.require(path.resolve()==Path(expected).resolve(),'Exact receipt path')
        self.require(sha(path)==r['sha256'] and ('size_bytes' not in r or path.stat().st_size==r['size_bytes']),'Receipt SHA/size '+str(path))
        self.inputs[str(path)]={'path':str(path),'sha256':r['sha256'],'size_bytes':path.stat().st_size}
        return path
    def read(self,path,expected=None):
        path=Path(path);self.receipt({'path':str(path),'sha256':expected or sha(path)})
        return json.loads(path.read_text())


def centered(x):
    values=np.asarray(x,dtype=np.float64)
    if values.ndim!=2 or not np.isfinite(values).all():raise ValueError('Finite2D logits required')
    return values-np.mean(values,axis=1)[:,None]


def projection(x,sums):
    p=x.shape[1]
    if x.shape!=(p*p,p) or not np.array_equal(np.bincount(sums,minlength=p),np.full(p,p)):
        raise ValueError('Uniform full grid required')
    means=np.stack([np.mean(x[sums==k],axis=0) for k in range(p)])
    return means[sums]


def finite_behavior(x,sums,ids):
    f=x[ids];labels=sums[ids];position=np.arange(len(ids))
    good=f[position,labels];peak=np.max(f,axis=1)
    ce=np.mean(peak+np.log(np.sum(np.exp(f-peak[:,None]),axis=1))-good)
    other=f.copy();other[position,labels]=-np.inf
    return {'ce':float(ce),'margin':float(np.mean(good-np.max(other,axis=1))),
            'accuracy':float(np.mean(np.argmax(f,axis=1)==labels))}


def response(change,sums,weight,splits):
    total=centered(change);rule=projection(total,sums);residual=total-rule
    components={'total':total,'sum_consistent':rule,'within_sum':residual}
    energy={key:float(np.einsum('ij,ij->',value,value)/len(value)) for key,value in components.items()}
    utility={split:{key:float(np.einsum('ij,ij->',weight[ids],value[ids])/len(ids)) for key,value in components.items()} for split,ids in splits.items()}
    return {'energy':energy,'cross_inner_product':float(np.einsum('ij,ij->',rule,residual)/len(total)),
            'energy_additivity_residual':energy['total']-energy['sum_consistent']-energy['within_sum'],
            'sum_projection_idempotence_max_abs':float(np.max(np.abs(projection(rule,sums)-rule))),
            'within_sum_mean_max_abs':float(np.max(np.abs(projection(residual,sums)))),
            'class_mean_max_abs':float(np.max(np.abs(np.mean(total,axis=1)))),
            'linear_ce_utility':utility,
            'utility_additivity_residual':{key:value['total']-value['sum_consistent']-value['within_sum'] for key,value in utility.items()}}


def recompute(arrays):
    f={name:centered(arrays[name]) for name in NAMES};base=f['before'];sums=arrays['sums']
    splits={name:arrays[name+'_indices'] for name in ('train','test')}
    exp=np.exp(base-np.max(base,axis=1)[:,None]);weight=-exp/np.sum(exp,axis=1)[:,None]
    weight[np.arange(len(base)),sums]+=1
    metrics={name:{split:finite_behavior(x,sums,ids) for split,ids in splits.items()} for name,x in f.items()}
    arms={};contrasts={}
    for name in NAMES[1:]:
        value=response(f[name]-base,sums,weight,splits)
        value['finite_improvement']={split:{metric:(metrics['before'][split][metric]-metrics[name][split][metric] if metric=='ce' else metrics[name][split][metric]-metrics['before'][split][metric]) for metric in ('ce','margin','accuracy')} for split in splits}
        value['ce_convex_remainder']={split:value['linear_ce_utility'][split]['total']-value['finite_improvement'][split]['ce'] for split in splits}
        arms[name]=value
    for left,right in CONTRASTS:
        value=response(f[right]-f[left],sums,weight,splits)
        value['finite_difference_right_minus_left']={split:{metric:metrics[right][split][metric]-metrics[left][split][metric] for metric in ('ce','margin','accuracy')} for split in splits}
        contrasts[left+'_to_'+right]=value
    finite={split:{metric:contrasts['raw_to_trunc']['finite_difference_right_minus_left'][split][metric]+contrasts['trunc_to_projected']['finite_difference_right_minus_left'][split][metric]-contrasts['raw_to_projected']['finite_difference_right_minus_left'][split][metric] for metric in ('ce','margin','accuracy')} for split in splits}
    linear={split:{key:contrasts['raw_to_trunc']['linear_ce_utility'][split][key]+contrasts['trunc_to_projected']['linear_ce_utility'][split][key]-contrasts['raw_to_projected']['linear_ce_utility'][split][key] for key in ('total','sum_consistent','within_sum')} for split in splits}
    return {'behavior':metrics,'responses_from_before':arms,'contrasts':contrasts,
            'finite_scalar_telescoping_residual':finite,'linear_utility_telescoping_residual':linear}


def compare_tree(check,actual,expected,label='metrics'):
    if isinstance(expected,dict):
        check.require(set(actual)==set(expected),label+' key schema')
        for key,value in expected.items():compare_tree(check,actual[key],value,label+'.'+key)
    else:check.close(actual,expected,label)


def aggregate_check(check,actual,values,label='paired'):
    first=values[0]
    if isinstance(first,dict):
        check.require(set(actual)==set(first) and all(set(v)==set(first) for v in values),label+' schemas')
        for key in first:aggregate_check(check,actual[key],[v[key] for v in values],label+'.'+key)
    elif isinstance(first,str):check.require(all(v==first for v in values) and actual==first,label+' semantics')
    else:
        x=np.asarray(values,dtype=np.float64);check.require(x.shape==(5,) and np.isfinite(x).all(),label+' five finite seeds')
        check.require(set(actual)=={'values','mean','sample_se','positive_count','negative_count','zero_count'},label+' scalar-summary schema')
        check.require(actual['values']==values,label+' exact seed values/order')
        mean=math.fsum(values)/5;se=math.sqrt(math.fsum((v-mean)**2 for v in values)/4)/math.sqrt(5)
        check.close(actual['mean'],mean,label+'.mean',rtol=1e-12,atol=1e-12)
        check.close(actual['sample_se'],se,label+'.sample_se',rtol=1e-12,atol=1e-12)
        for name,predicate in (('positive_count',lambda v:v>0),('negative_count',lambda v:v<0),('zero_count',lambda v:v==0)):
            check.require(actual[name]==sum(predicate(v) for v in values),label+'.'+name)


def tree_hash(value):
    result=hashlib.sha256()
    def visit(x):
        if isinstance(x,torch.Tensor):
            x=x.detach().cpu().contiguous();result.update(b'tensor\0'+str(x.dtype).encode()+b'\0'+json.dumps(list(x.shape)).encode()+b'\0'+x.numpy().tobytes())
        elif isinstance(x,dict):
            result.update(b'dict\0')
            for key in sorted(x,key=lambda key:(type(key).__name__,str(key))):visit(key);visit(x[key])
            result.update(b'enddict\0')
        elif isinstance(x,(tuple,list)):
            result.update(type(x).__name__.encode()+b'\0')
            for child in x:visit(child)
            result.update(b'endsequence\0')
        else:result.update(type(x).__name__.encode()+b'\0'+json.dumps(x,allow_nan=False).encode()+b'\0')
    visit(value);return result.hexdigest()


def adam_residuals(before,after,action):
    group=before['optimizer_state']['param_groups'][0]
    if after['optimizer_state']['param_groups']!=before['optimizer_state']['param_groups'] or len(before['optimizer_state']['param_groups'])!=1:
        raise ValueError('Common Adam flags/order not preserved')
    if after['parameter_identity']!=before['parameter_identity']:raise ValueError('Parameter layout changed')
    if set(after['model_state'])!=set(before['model_state']) or set(after['optimizer_state']['state'])!=set(before['optimizer_state']['state']):raise ValueError('Model/moment key roster changed')
    if group.get('amsgrad') or group.get('maximize'):raise ValueError('Non-original Adam mode')
    maxima={name:0. for name in ('m_relative_residual','v_relative_residual','parameter_relative_residual')}
    offset=0
    for identity,pid in zip(before['parameter_identity'],group['params'],strict=True):
        name=identity['name'];old=before['optimizer_state']['state'][pid];new=after['optimizer_state']['state'][pid]
        theta=before['model_state'][name].double();g=action[offset:offset+theta.numel()].double().reshape(theta.shape);offset+=theta.numel()
        if int(new['step'])!=1501 or int(old['step'])!=1500:raise ValueError('Not exactly one common Adam step')
        b1,b2=group['betas'];expected_m=b1*old['exp_avg'].double()+(1-b1)*g;expected_v=b2*old['exp_avg_sq'].double()+(1-b2)*g*g
        direction=new['exp_avg'].double()/(1-b1**1501)/((new['exp_avg_sq'].double()/(1-b2**1501)).sqrt()+group['eps'])
        theta_expected=theta-group['lr']*group['weight_decay']*theta-group['lr']*direction
        for key,expected,actual,previous in (
            ('m_relative_residual',expected_m,new['exp_avg'].double(),old['exp_avg'].double()),
            ('v_relative_residual',expected_v,new['exp_avg_sq'].double(),old['exp_avg_sq'].double()),
            ('parameter_relative_residual',theta_expected,after['model_state'][name].double(),theta)):
            denominator=max(float(expected.abs().max()),float(previous.abs().max()),TINY32)
            maxima[key]=max(maxima[key],float((actual-expected).abs().max())/denominator)
    if offset!=action.numel():raise ValueError('Unconsumed gradient coordinates')
    return maxima


def tensor_check(check,saved,first,seed):
    check.require(saved['schema']==SCHEMA and saved['seed']==seed and set(saved['states'])==set(NAMES),'Saved five-state schema')
    states=saved['states'];actions=saved['actions'];keys=('model_state','optimizer_state','parameter_identity')
    before={key:first['full_before'][key] for key in keys};raw_after={key:first['full_after'][key] for key in keys}
    check.require(tree_hash(states['before'])==tree_hash(before) and tree_hash(states['raw'])==tree_hash(raw_after),'Exact inherited before/raw copies')
    check.require(set(actions)=={'raw','trunc','projected','zero'} and set(saved['identities'])==set(NEW),'Exactly three new action identities')
    check.require(torch.equal(actions['raw'],first['actions']['actions']['raw_norm_matched']) and torch.equal(actions['projected'],first['actions']['actions']['norm_matched']),'Saved raw/projected action copy identity')
    check.require(bool((actions['zero']==0).all()),'Explicit all-coordinate zero action')
    q=first['actions']['Q'].double();g=first['raw_gradient'].double();u=actions['raw'].double();v=actions['projected'].double();t=actions['trunc'].double()
    check.require(all(a.dtype==torch.float32 and a.shape==g.shape and bool(torch.isfinite(a).all()) for a in actions.values()),'Finite complete float32 actions')
    pu=q@(q.T@u);pg=q@(q.T@g);rho=float(pg.norm()/g.norm())
    norms={name:float(a.double().norm()) for name,a in actions.items()}
    algebra={'norms':norms,'gradient_norm':float(g.norm()),'rho':rho,
             'Q_orthogonality_max_abs':float((q.T@q-torch.eye(q.shape[1],dtype=torch.float64)).abs().max()),
             'Q_orthogonality_tolerance':1e-10,'relative_tolerance':32*EPS32,
             'raw_projected_norm_relative_mismatch':abs(norms['raw']-norms['projected'])/norms['raw'],
             'trunc_rho_projected_relative_error':float((t-rho*v).norm())/norms['raw'],
             'trunc_projection_relative_error':float((t-pu).norm())/norms['raw'],
             'trunc_projected_collinearity_error':float((t/norms['trunc']-v/norms['projected']).norm())}
    compare_tree(check,saved['action_algebra'],algebra,'action_algebra')
    check.require(algebra['Q_orthogonality_max_abs']<=1e-10 and rho<=1+32*EPS32,'Action geometry tolerance')
    check.require(all(algebra[key]<=32*EPS32 for key in ('raw_projected_norm_relative_mismatch','trunc_rho_projected_relative_error','trunc_projection_relative_error','trunc_projected_collinearity_error')),'Explicit float32 action bounds')
    result={}
    for name in NEW:
        identities=saved['identities'][name]
        check.require(identities['before_state_sha256']==tree_hash(before) and identities['after_state_sha256']==tree_hash(states[name]),'Common new-state identity '+name)
        residuals=adam_residuals(before,states[name],actions[name]);result[name]=residuals
        check.require(max(residuals.values())<=32*EPS32,'FP32-aware Adam recurrence '+name)
        compare_tree(check,identities['adam_arithmetic'],{'relative_tolerance':32*EPS32,**residuals},'Adam '+name)
    return {'seed':seed,'action_algebra':algebra,'new_adam_residuals':result}


def acceptance(check,metrics,arrays):
    centered_logits={name:centered(arrays[name]) for name in NAMES}
    entries=[(value,centered_logits[name]-centered_logits['before']) for name,value in metrics['responses_from_before'].items()]
    entries += [(metrics['contrasts'][left+'_to_'+right],centered_logits[right]-centered_logits[left]) for left,right in CONTRASTS]
    for value,change in entries:
        energy_tol=1e-10*(1+sum(abs(x) for x in value['energy'].values()))
        check.require(abs(value['energy_additivity_residual'])<=energy_tol and abs(value['cross_inner_product'])<=energy_tol,'Energy orthogonality/Pythagorean acceptance')
        point_tol=1e-10*(1+float(np.max(np.abs(centered(change)))))
        check.require(all(value[key]<=point_tol for key in ('sum_projection_idempotence_max_abs','within_sum_mean_max_abs','class_mean_max_abs')),'Pointwise projection/class mean acceptance')
        for split in ('train','test'):
            check.require(abs(value['utility_additivity_residual'][split])<=1e-10*(1+sum(abs(x) for x in value['linear_ce_utility'][split].values())),'Utility additivity acceptance')
    for value in metrics['responses_from_before'].values():
        check.require(all(x>=-1e-10 for x in value['ce_convex_remainder'].values()),'CE convex remainder acceptance')
    check.require(all(abs(x)<=1e-12 for split in metrics['finite_scalar_telescoping_residual'].values() for x in split.values()),'Finite scalar telescoping acceptance')
    for split in ('train','test'):
        for component,value in metrics['linear_utility_telescoping_residual'][split].items():
            pieces=[metrics['contrasts'][key]['linear_ce_utility'][split][component] for key in ('raw_to_trunc','trunc_to_projected','raw_to_projected')]
            check.require(abs(value)<=1e-10*(1+sum(abs(x) for x in pieces)),'Linear utility telescoping acceptance')


def admission(check,completion_sha):
    check.require(not OUT.exists() and not OUT.is_symlink(),'Exclusive audit output')
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        check.require(os.environ.get(key)=='1','Single-thread environment '+key)
    group=Path('/sys/fs/cgroup')/Path('/proc/self/cgroup').read_text().strip().split(':',2)[2].lstrip('/')
    check.require(int((group/'memory.max').read_text())==16*1024**3 and int((group/'memory.swap.max').read_text())==0,'16GiB no-swap cgroup')
    quota,period=map(int,(group/'cpu.max').read_text().split());check.require(quota==period,'One CPU cgroup')
    own=next(part for part in reversed(group.parts) if part.endswith('.service'))
    def properties(unit,keys):
        value=subprocess.check_output(['systemctl','--user','show',unit,*['--property='+key for key in keys]],text=True)
        return dict(line.split('=',1) for line in value.splitlines())
    check.require(properties(own,('Type','RuntimeMaxUSec','Restart','KillMode'))=={'Type':'exec','RuntimeMaxUSec':'5min','Restart':'no','KillMode':'control-group'},'Five-minute bounded service')
    check.require(properties('spectral-base-grokking-function-response-001.service',('MainPID','ActiveState','Result'))=={'MainPID':'0','ActiveState':'inactive','Result':'success'},'Source diagnostic successfully terminal')
    check.require(shutil.disk_usage(PARENT).free>1024**3+100*1024**2,'Audit output reserve')
    OUT.mkdir(mode=0o700)
    for path in sorted(HERE.glob('*.py'))+[HERE/'plan.md']:check.receipt({'path':str(path),'sha256':sha(path)})
    done=check.read(SOURCE/'complete.json',completion_sha)
    check.require(done['schema']==SCHEMA and done['status']=='complete' and done['seeds']==list(SEEDS) and done['new_actions']==done['new_full_grid_extractions']==15,'Full five-seed15-state diagnostic')
    check.require(not (SOURCE/'failure.json').exists(),'No source failure marker')
    manifest=check.read(check.receipt(done['manifest'],SOURCE/'manifest.json'))
    summary=check.read(check.receipt(done['summary'],SOURCE/'summary.json'))
    check.require(manifest['schema']==summary['schema']==SCHEMA and manifest['seeds']==summary['seeds']==list(SEEDS),'Diagnostic schemas/seeds')
    check.require(summary['manifest']==done['manifest'],'Summary/complete manifest identity')
    check.require(manifest['actions']==list(NEW) and done['source_sha256']==manifest['source_sha256']==summary['source_sha256'],'Fixed actions/source binding')
    for name,value in manifest['source_sha256'].items():
        check.require(not Path(name).is_absolute() and '..' not in Path(name).parts,'Relative source path')
        check.receipt({'path':str(REPO/name),'sha256':value})
    for rec in manifest['input_receipts']:check.receipt(rec)
    batch=check.read(RAW/'batch-complete.json',RAW_SHA)
    measure=check.read(RAW/'measurement-001/complete.json',MEASURE_SHA)
    old_manifest=check.read(check.receipt(measure['manifest'],RAW/'measurement-001/manifest.json'))
    check.require(batch['status']==measure['status']=='complete' and batch['seeds']==list(SEEDS),'Pinned original completed raw inputs')
    check.require([r['seed'] for r in batch['accepted']]==list(SEEDS),'Original accepted seed order')
    check.require(len(done['accepted_results'])==5 and [r['seed'] for r in summary['seed_results']]==list(SEEDS),'Exact result roster')
    return done,summary,batch,measure,old_manifest


def run(check,completion_sha):
    done,summary,batch,measure,old_manifest=admission(check,completion_sha)
    reports=[];saved_metrics=[]
    for seed,result_receipt,summary_row,seed_receipt in zip(SEEDS,done['accepted_results'],summary['seed_results'],batch['accepted'],strict=True):
        result=check.read(check.receipt(result_receipt,SOURCE/f'seed{seed}-result.json'))
        check.require(result==summary_row and result['seed']==seed and result['schema']==SCHEMA,'Per-seed result/summary identity')
        check.require(result['new_one_step_actions']==list(NEW) and result['new_full_grid_extractions']==3 and result['raw_update_replayed'] is False,'Fixed new actions/no raw replay declaration')
        check.require(result['source_sha256']==summary['source_sha256'],'Per-seed source binding')
        old=check.read(check.receipt(seed_receipt,RAW/f'seed{seed}/complete.json'))
        check.require(old['seed']==seed and old['status']=='complete' and old['source_sha256']==batch['source_sha256'],'Inherited completed seed source identity')
        first_receipt=next(r for r in old['artifact_receipts'] if Path(r['path']).name=='first-step-tensors.pt')
        first=torch.load(check.receipt(first_receipt),map_location='cpu',weights_only=True)
        saved=torch.load(check.receipt(result['states'],SOURCE/f'seed{seed}-states.pt'),map_location='cpu',weights_only=True)
        check.require(saved['source_sha256']==result['source_sha256'] and saved['input_first_step']=={key:first_receipt[key] for key in ('path','sha256','size_bytes')},'New tensor input/source binding')
        check.require(first['seed']==seed and first['policy']=='raw_norm_matched' and first['step']==1501 and first['parent_checkpoint']==old['parent_checkpoint'],'Inherited bundle identity')
        tensor_report=tensor_check(check,saved,first,seed)
        check.require(saved['action_algebra']==result['action_algebra'] and saved['identities']==result['identities'],'Scalar/tensor diagnostic identity')
        contract=old_manifest['prior_recipe']['seeds'][str(seed)]
        raw_row=next(r for r in measure['accepted_results'] if (r['seed'],r['step'])==(seed,1501))
        check.require(result['archived_before_logits']==contract['raw'] and result['archived_raw_logits']==raw_row['raw'],'Original before/raw logit receipts')
        with np.load(check.receipt(result['logits'],SOURCE/f'seed{seed}-logits.npz'),allow_pickle=False) as archive:arrays={key:archive[key] for key in archive.files}
        check.require(set(arrays)==set(NAMES)|{'pairs','sums','train_indices','test_indices'},'Exact diagnostic NPZ members')
        for name in NAMES:check.require(arrays[name].shape==(12769,113) and arrays[name].dtype==np.float32 and np.isfinite(arrays[name]).all(),'Finite full-grid action logits '+name)
        pairs=np.column_stack((np.repeat(np.arange(113),113),np.tile(np.arange(113),113)))
        check.require(np.array_equal(arrays['pairs'],pairs) and np.array_equal(arrays['sums'],pairs.sum(1)%113),'Ordered modular full grid')
        check.require(np.array_equal(np.sort(np.r_[arrays['train_indices'],arrays['test_indices']]),np.arange(12769)),'Complete train/test partition')
        for name,rec in (('before',contract['raw']),('raw',raw_row['raw'])):
            with np.load(check.receipt(rec),allow_pickle=False) as archive:
                check.require(np.array_equal(arrays[name],archive['logits']),'Byte-value unchanged archived logits '+name)
                for target,key in (('pairs','pairs'),('sums','sums'),('train_indices','train_ids'),('test_indices','test_ids')):
                    check.require(np.array_equal(arrays[target],archive[key]),'Common archived structural identity '+key)
        expected=recompute(arrays);actual={key:value for key,value in result['metrics'].items() if key!='semantics'}
        compare_tree(check,actual,expected)
        acceptance(check,expected,arrays);acceptance(check,actual,arrays)
        reports.append(tensor_report);saved_metrics.append(result['metrics'])
        del saved,first,arrays,expected;gc.collect()
        print(json.dumps({'audited_seed':seed,'checks':check.count,'errors':len(check.errors)}),flush=True)
    aggregate_check(check,summary['paired'],saved_metrics)
    return {'scope':'Five common-state saved diagnostics:15 new state recurrences, function responses and six contrasts; no inference/optimizer replay',
            'seed_tensor_checks':reports,'numpy_version':np.__version__,'torch_version':torch.__version__,
            'completion_sha256':completion_sha,'limitations':['Only common-state local effects, not a1000-step mediation analysis.',
               'Archived/new logits originate from separate CUDA invocations.',
               'Roundoff-scale diagnostic signs are aggregated from verified saved scalar values, not claimed reduction-order invariant.']}


def finish(check,payload):
    for rec in list(check.inputs.values()):
        try:check.receipt(rec)
        except Exception as error:check.errors.append('Final receipt recheck '+str(error))
    result={'status':'FAIL' if check.errors else 'PASS','checks':check.count,'errors':check.errors,
            'maximum_absolute_scalar_discrepancy':check.max_error,'input_receipts':list(check.inputs.values()),
            'elapsed_seconds':time.monotonic()-check.start,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,**payload}
    raw=(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
    if len(raw)>100*1024**2 or shutil.disk_usage(PARENT).free<len(raw)+1024**3:raise RuntimeError('Audit output budget')
    if not OUT.is_dir():raise RuntimeError('Audit failed before exclusive-output admission')
    with (OUT/'result.json').open('xb') as handle:handle.write(raw)
    print(json.dumps({'status':result['status'],'checks':check.count,'result_sha256':sha(OUT/'result.json')}),flush=True)
    return not check.errors


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--completion-sha',required=True)
    args=parser.parse_args();check=Check()
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    try:payload=run(check,args.completion_sha)
    except Exception as error:check.errors.append(f'{type(error).__name__}: {error}');payload={'scope':'Incomplete audit; preserved failure'}
    sys.exit(0 if finish(check,payload) else 1)
