#!/usr/bin/env python3
"""Independent NumPy rederivation of all 35 saved action-state readouts."""
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import statistics
import time

for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    assert os.environ.get(key) == '1'
import numpy as np
import torch
import audit_action_tensors as common

BATCH = common.BATCH
MEASURE = BATCH/'measurement-001'
ANALYSIS = BATCH/'analysis-001'
OUT = BATCH/'readout-audit-001.json'
P = 113
FIXED = [9,33,32,49,11]
ROWS = []
STRUCTURES = {}
MAX_ERRORS = {}


def check(ok, name):
    common.check(ok, name)


def compare(a, b, name, tol=1e-8):
    if a is None or b is None:
        check(a is b, name+' undefined handling'); return
    a, b = np.asarray(a,dtype=np.float64), np.asarray(b,dtype=np.float64)
    error = float(np.max(np.abs(a-b))) if a.size else 0.0
    MAX_ERRORS[name.split(':')[0]] = max(MAX_ERRORS.get(name.split(':')[0],0.0),error)
    check(np.allclose(a,b,rtol=tol,atol=tol), f'{name}: maximum absolute error={error}')


def hash_ids(a):
    a = np.ascontiguousarray(a,dtype='<i8')
    return hashlib.sha256(b'little-endian-int64-v1\0'+np.asarray(a.shape,dtype='<i8').tobytes()+a.tobytes()).hexdigest()


def structures(arrays, seed):
    pair = np.column_stack((np.repeat(np.arange(P),P),np.tile(np.arange(P),P)))
    check(np.array_equal(arrays['pairs'],pair),'lexicographic pairs')
    check(np.array_equal(arrays['sums'],pair.sum(1)%P),'modular labels')
    train,test = arrays['train_ids'], arrays['test_ids']
    check(np.array_equal(np.sort(np.r_[train,test]),np.arange(P*P)), 'train/test partition')
    fit,evaluation = arrays['probe_fit_indices'],arrays['probe_eval_indices']
    check(np.array_equal(np.sort(np.r_[fit,evaluation]),np.arange(len(test))), 'probe partition')
    # Independent stratified split regeneration; no model or training operation.
    rng = torch.Generator().manual_seed(20260909+seed)
    expected_fit, expected_eval = [],[]
    labels = arrays['sums'][test]
    for label in range(P):
        indices = np.flatnonzero(labels == label)
        shuffled = indices[torch.randperm(len(indices),generator=rng).numpy()]
        expected_fit.extend(shuffled[:len(indices)//2]); expected_eval.extend(shuffled[len(indices)//2:])
    check(np.array_equal(fit,np.sort(expected_fit)), 'stratified fit seed')
    check(np.array_equal(evaluation,np.sort(expected_eval)), 'stratified eval seed')
    rng = torch.Generator().manual_seed(20261909+seed)
    for permutation in arrays['null_permutations']:
        expected = torch.randperm(len(test),generator=rng).numpy()
        if np.array_equal(expected,np.arange(len(test))):
            expected = np.roll(expected,1)
        check(np.array_equal(permutation,expected), 'null permutation seed')
    if seed in STRUCTURES:
        for key in ('pairs','sums','train_ids','test_ids','probe_fit_indices','probe_eval_indices','null_permutations'):
            check(np.array_equal(arrays[key],STRUCTURES[seed][key]), 'fixed structural arrays '+key)
        return STRUCTURES[seed]['edges']
    mask = np.zeros(P*P,dtype=bool); mask[test]=True
    edges=[]
    for axis in ('a','b'):
        for delta in (1,2,4,8,16,32):
            def destination(source):
                a,b=source//P,source%P
                return ((a+delta)%P)*P+b if axis=='a' else a*P+(b+delta)%P
            dst = destination(test); keep=mask[dst]
            hh_all = np.column_stack((test[keep],dst[keep]))
            matched={}
            groups=[]
            for label in range(P):
                blocks={}
                for role,source in (('TH',train),('HH',test)):
                    target=destination(source)
                    eligible=mask[target] & ((source//P+source%P)%P==label)
                    candidates=list(zip(source[eligible].tolist(),target[eligible].tolist()))
                    candidates.sort(key=lambda edge:(hashlib.sha256(
                        f'grokking-symmetry-match-v1|{role}|{axis}|{delta}|{label}|{edge[0]}|{edge[1]}'.encode()).digest(),*edge))
                    blocks[role]=candidates
                count=min(map(len,blocks.values()))
                group={'source_sum':label,'th_candidates':len(blocks['TH']), 'hh_candidates':len(blocks['HH']), 'matched_count':count}
                for role in ('TH','HH'):
                    selected=np.asarray(blocks[role][:count],dtype=np.int64).reshape(-1,2)
                    matched.setdefault(role,[]).append(selected)
                    group[role.lower()+'_edge_ids_sha256']=hash_ids(selected)
                groups.append(group)
            edges.append((axis,delta,hh_all,{role:np.concatenate(parts) for role,parts in matched.items()},groups))
    STRUCTURES[seed]={key:arrays[key].copy() for key in ('pairs','sums','train_ids','test_ids','probe_fit_indices','probe_eval_indices','null_permutations')}
    STRUCTURES[seed]['edges']=edges
    return edges


def metric(logits,centered,edges,shift):
    left,right=edges.T
    a,b=centered[left],centered[right]
    residual=b-a[:,(np.arange(P)-shift)%P]
    numerator=float(np.sum(residual**2)); denominator=float(np.sum(a*a)+np.sum(b*b))
    count=len(edges); rms=math.sqrt(denominator/(2*count*P)) if count else 0.0
    raw=float(np.sum(logits[left]**2)+np.sum(logits[right]**2))
    defined=rms>1e-12 and denominator>0
    return {'count':count,'numerator':numerator,'denominator':denominator,
            'centered_rms':rms,'raw_rms':math.sqrt(raw/(2*count*P)) if count else 0.0,
            'energy_defined':defined,'value':numerator/denominator if defined else None}


def metric_compare(observed,expected,label):
    for key,val in expected.items():
        if isinstance(val,bool):
            check(observed[key] is val,label+key)
        else:
            compare(observed[key],val,label+key,1e-9)


def pooled(items):
    count=sum(x['count'] for x in items); num=sum(x['numerator'] for x in items); den=sum(x['denominator'] for x in items)
    rms=math.sqrt(den/(2*count*P)) if count else 0.0
    raw=sum(x['raw_rms']**2*2*x['count']*P for x in items)
    defined=rms>1e-12 and den>0
    return {'count':count,'numerator':num,'denominator':den,'centered_rms':rms,
            'raw_rms':math.sqrt(raw/(2*count*P)) if count else 0.0,
            'energy_defined':defined,'value':num/den if defined else None}


def symmetry(arrays,row,edges):
    logits=arrays['logits'].astype(np.float64); centered=logits-logits.mean(1,keepdims=True)
    full=row['full_symmetry']; corrects=[]; wrongs=[]; ths=[]; hhs=[]; th_edges=[]; hh_edges=[]
    for axis,delta,heldout,matched,groups in edges:
        observed=full['heldout_symmetry'][axis][str(delta)]
        check(observed['edge_ids_sha256']==hash_ids(heldout),'heldout edge hash')
        for name,shift,collector in (('correct',delta,corrects),('wrong_shift',(delta+37)%P,wrongs)):
            value=metric(logits,centered,heldout,shift); metric_compare(observed[name],value,'symmetry:heldout '+name); collector.append(value)
        observed=full['cleanup'][axis][str(delta)]
        check(observed['groups']==groups,'matched group roster/hashes')
        values={}
        for role,key,collector,edgecollector in (('TH','train_to_heldout',ths,th_edges),('HH','heldout_to_heldout',hhs,hh_edges)):
            values[role]=metric(logits,centered,matched[role],delta)
            metric_compare(observed[key],values[role],'symmetry:matched '+role)
            check(observed[role.lower()+'_edge_ids_sha256']==hash_ids(matched[role]),'matched edge hash')
            collector.append(values[role]); edgecollector.append(matched[role])
        expected=None if values['TH']['value'] is None or values['HH']['value'] is None else values['TH']['value']-values['HH']['value']
        compare(observed['excess'],expected,'symmetry:excess')
    for name,items in (('correct',corrects),('wrong_shift',wrongs)):
        metric_compare(row['symmetry']['heldout_shift_pooled'][name],pooled(items),'symmetry:pooled heldout '+name)
    for key,items,edgeblocks,role in (('train_to_heldout',ths,th_edges,'th'),('heldout_to_heldout',hhs,hh_edges,'hh')):
        value=pooled(items)
        metric_compare(full['cleanup']['pooled'][key],value,'symmetry:pooled '+key)
        check(full['cleanup']['pooled'][role+'_edge_ids_sha256']==hash_ids(np.concatenate(edgeblocks)), 'pooled edge hash')
    th,hh=pooled(ths),pooled(hhs)
    excess=None if th['value'] is None or hh['value'] is None else th['value']-hh['value']
    compare(row['symmetry']['training_membership_pooled']['excess'],excess,'symmetry:pooled excess')
    test=np.sort(arrays['test_ids']); mask=np.zeros(P*P,dtype=bool); mask[test]=True
    swapped=test%P*P+test//P; keep=(test//P<test%P)&mask[swapped]
    exchange=np.column_stack((test[keep],swapped[keep]))
    check(full['exchange']['edge_ids_sha256']==hash_ids(exchange),'exchange edge hash')
    for name,shift in (('correct',0),('wrong_shift',37)):
        metric_compare(full['exchange'][name],metric(logits,centered,exchange,shift),'symmetry:exchange '+name)
    for split,ids in (('all',np.arange(P*P)),('train',arrays['train_ids']),('test',arrays['test_ids'])):
        compare(row['symmetry']['centered_logit_rms'][split],np.sqrt(np.mean(centered[ids]**2)),'rms:'+split)


def probes(arrays,scalar,row):
    ids=arrays['test_ids']; fit=arrays['probe_fit_indices']; ev=arrays['probe_eval_indices']
    labels=arrays['sums'][ids]; angles=(2*math.pi/P)*labels[:,None]*np.arange(1,57)[None,:]
    targets=np.stack((np.cos(angles),np.sin(angles)),axis=2)
    for feature in ('final_hidden','pre_attention'):
        original=scalar['probes'][feature]; compact=row['probes'][feature]
        values=arrays[feature][ids].astype(np.float64)
        average=values[fit].mean(0); centered=values-average
        scale=float(np.sqrt(np.mean(centered[fit]**2))); scale=scale or 1.0
        compare(original['fit_transform']['feature_mean'],average,'probe-transform:mean',1e-9)
        compare(original['fit_transform']['feature_rms_scalar'],scale,'probe-transform:scale',1e-9)
        x=centered/scale; xf,xe=x[fit],x[ev]
        matrix=xf.T@xf/len(fit)+np.eye(x.shape[1])*.001
        inverse=np.linalg.inv(matrix)
        null_scores=[]; observed_eval=None
        for index,perm in enumerate([None,*arrays['null_permutations']]):
            y=targets if perm is None else targets[perm]
            yf,ye=y[fit],y[ev]; mean=yf.mean(0)
            rhs=xf.T@(yf-mean).reshape(len(fit),-1)/len(fit)
            coefficients=inverse@rhs
            pf=(xf@coefficients).reshape(yf.shape)+mean
            pe=(xe@coefficients).reshape(ye.shape)+mean
            fs=1-((yf-pf)**2).sum((0,2))/((yf-mean)**2).sum((0,2))
            es=1-((ye-pe)**2).sum((0,2))/((ye-mean)**2).sum((0,2))
            selected=sorted(range(1,57),key=lambda k:(-fs[k-1],k))[:5]
            score=float(es[np.asarray(selected)-1].mean())
            record=original['observed'] if index==0 else original['null']['runs'][index-1]
            compare([v['fit_r2'] for v in record['per_frequency']],fs,'probe-fit:all 56',1e-8)
            compare([v['eval_r2'] for v in record['per_frequency']],es,'probe-eval:all 56',1e-8)
            compare([[v['fit_target_mean_cos'],v['fit_target_mean_sin']] for v in record['per_frequency']],mean,'probe-target:mean',1e-10)
            check(record['selected_frequencies']==selected, 'fit-selected frequencies '+feature+str(index))
            compare(record['selected_eval_mean_r2'],score,'probe-selected:mean',1e-8)
            if index==0:
                observed_eval=es
                compare(compact['selected_eval_mean_r2'],score,'compact:selected')
                compare(compact['fixed_panel_eval_mean_r2'],es[np.asarray(FIXED)-1].mean(),'compact:fixed')
                check(compact['selected_frequencies']==selected and compact['fixed_panel_frequencies']==FIXED,'compact panels')
            else:
                null_scores.append(score)
        for key,value in (('mean',np.mean(null_scores)),('std_population',np.std(null_scores)),('max',max(null_scores))):
            compare(original['null']['selected_eval_mean_r2_'+key],value,'null:'+key)
        compare(compact['null_max_eval_mean_r2'],max(null_scores),'compact:null max')


def audit_state(item):
    for key in ('checkpoint','raw','scalar','analyzed_state'):
        common.receipt(item[key])
    scalar=json.loads(Path(item['scalar']['path']).read_text()); row=json.loads(Path(item['analyzed_state']['path']).read_text())
    check((scalar['seed'],scalar['policy'],scalar['step'])==(item['seed'],item['policy'],item['step']),'state identity')
    with np.load(item['raw']['path'],allow_pickle=False) as archive:
        arrays={key:archive[key] for key in archive.files}
    edges=structures(arrays,item['seed'])
    logits=arrays['logits'].astype(np.float64); labels=arrays['sums']
    for split,ids in (('train',arrays['train_ids']),('test',arrays['test_ids'])):
        value=logits[ids]; target=labels[ids]
        correct=value[np.arange(len(ids)),target]
        maximum=value.max(1)
        ce=float(np.mean(maximum+np.log(np.exp(value-maximum[:,None]).sum(1))-correct))
        accuracy=float(np.mean(value.argmax(1)==target))
        competitors=value.copy(); competitors[np.arange(len(ids)),target]=-np.inf
        margin=float(np.mean(correct-competitors.max(1)))
        compare(scalar['behavior'][split]['loss'],ce,'behavior:CE',2e-6)
        compare(row['behavior'][split]['loss'],ce,'behavior:analyzed CE',2e-6)
        compare(row['behavior'][split]['accuracy'],accuracy,'behavior:accuracy',1e-7)
        compare(row['behavior'][split]['correct_class_margin_mean'],margin,'behavior:margin',1e-10)
        check(row['behavior'][split]['count']==len(ids),'behavior count')
    probes(arrays,scalar,row); symmetry(arrays,row,edges)
    return row


def paired(summary,manifest,rows):
    check(summary['new_state_rows']==rows,'analysis rows exactly match measured states')
    source=summary['input_receipts']['archived_summary']; common.receipt(source)
    check(source['sha256']=='9ac881c1bbca72a9dfd5525c226e05938ea4b0553e1e924bafb59ffd8a9a7c0d','accepted archived source')
    refs=summary['archived_native_reference_rows']
    check([(x['seed'],x['step']) for x in refs]==[(s,t) for s in range(100,105) for t in (1500,2000,2500)],'15 archived references')
    data={(x['seed'],x['policy'],x['step']):x for x in rows}
    for x in refs:
        data[x['seed'],'native',x['step']]=x
    def nested(row,path):
        for key in path:
            row=row[key]
        return row
    seen=[]
    for contrast in summary['paired_endpoint_contrasts']:
        seen.append((contrast['step'],contrast['contrast']))
        for metric,record in contrast['metrics'].items():
            definition=manifest['metric_definitions'][metric]
            differences=[]
            for item,seed in zip(record['paired'],range(100,105),strict=True):
                left=nested(data[seed,contrast['left_policy'],contrast['step']],definition['path'])
                right=nested(data[seed,contrast['right_policy'],contrast['step']],definition['path'])
                difference=None if left is None or right is None else left-right
                check(item=={'seed':seed,'left':left,'right':right,'difference':difference},'paired source/difference')
                if difference is not None:
                    differences.append(difference)
            complete=len(differences)==5
            check(record['defined_count']==len(differences) and record['complete_five_seed_aggregate']==complete,'paired count')
            mean=statistics.mean(differences) if complete else None
            sd=statistics.stdev(differences) if complete else None
            compare(record['mean_difference'],mean,'paired:mean',1e-12)
            compare(record['sample_sd'],sd,'paired:SD',1e-12)
            compare(record['sample_se'],sd/math.sqrt(5) if complete else None,'paired:SE',1e-12)
            for key,predicate in (('positive_count',lambda x:x>0),('negative_count',lambda x:x<0),('zero_count',lambda x:x==0)):
                check(record[key]==(sum(map(predicate,differences)) if complete else None),'paired signs')
    check(seen==[(step,name) for step in (2000,2500) for name in ('orthogonal_minus_native','norm_matched_minus_native','norm_matched_minus_orthogonal')],'six fixed contrasts')


def main():
    started=time.monotonic(); assert not OUT.exists()
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    cgroup=Path('/sys/fs/cgroup')/Path('/proc/self/cgroup').read_text().strip().split(':',2)[2].lstrip('/')
    assert int((cgroup/'memory.max').read_text())==4*1024**3 and int((cgroup/'memory.swap.max').read_text())==0
    quota,period=map(int,(cgroup/'cpu.max').read_text().split()); assert quota==period
    complete=json.loads((MEASURE/'complete.json').read_text()); assert complete['status']=='complete'
    analysis_complete=json.loads((ANALYSIS/'complete.json').read_text()); assert analysis_complete['status']=='complete'
    common.receipt(analysis_complete['summary']); common.receipt(analysis_complete['manifest'])
    summary=json.loads((ANALYSIS/'summary.json').read_text()); manifest=json.loads((ANALYSIS/'manifest.json').read_text())
    expected=[(seed,policy,step) for seed in range(100,105) for policy in ('native','orthogonal','norm_matched') for step in ((1501,) if policy=='native' else (1501,2000,2500))]
    check([(r['seed'],r['policy'],r['step']) for r in complete['accepted_results']]==expected,'35 fixed-state roster')
    for name,value in summary['analysis_source_sha256'].items():
        check(common.sha(common.REPO/name)==value,'analysis source hash '+name)
    for item in complete['accepted_results']:
        ROWS.append(audit_state(item))
        print(json.dumps({'audited_state':[item['seed'],item['policy'],item['step']], 'checks':common.CHECKS,'errors':len(common.ERRORS)}),flush=True)
        assert time.monotonic()-started<580
    paired(summary,manifest,ROWS)
    result={'status':'PASS' if not common.ERRORS else 'FAIL','scope':'independent NumPy raw-array and paired-JSON corroboration, not training/inference replication',
            'states':expected,'checks':common.CHECKS,'errors':common.ERRORS,'maximum_absolute_errors_by_family':MAX_ERRORS,
            'script_sha256':common.sha(__file__),'helper_sha256':common.sha(common.__file__),
            'measurement_completion_sha256':common.sha(MEASURE/'complete.json'),
            'analysis_completion_sha256':common.sha(ANALYSIS/'complete.json'),
            'analysis_summary_sha256':common.sha(ANALYSIS/'summary.json'),
            'input_receipts':complete['accepted_results'],'source_sha256':summary['analysis_source_sha256'],
            'elapsed_seconds':time.monotonic()-started,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}
    with OUT.open('x') as handle:
        json.dump(result,handle,indent=2,sort_keys=True,allow_nan=False); handle.write('\n')
    print(json.dumps({key:value for key,value in result.items() if key not in ('input_receipts','states')}),flush=True)
    assert not common.ERRORS


if __name__=='__main__':
    main()
