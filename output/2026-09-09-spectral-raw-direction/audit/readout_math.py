"""Independent NumPy equations; no model, producer imports or artifact IO."""
import hashlib
import math
import numpy as np


def behavior(logits, labels, ids):
    values=logits[ids].astype(np.float64); targets=labels[ids]
    correct=values[np.arange(len(ids)),targets]
    peak=values.max(axis=1)
    ce=np.mean(peak+np.log(np.exp(values-peak[:,None]).sum(axis=1))-correct)
    competitors=values.copy(); competitors[np.arange(len(ids)),targets]=-np.inf
    return {'count':len(ids),'loss':float(ce),'accuracy':float(np.mean(values.argmax(axis=1)==targets)),
            'correct_class_margin_mean':float(np.mean(correct-competitors.max(axis=1)))}


def probe(values,labels,fit,ev,permutations,p=113,top_k=5):
    values=values.astype(np.float64)
    mean_x=values[fit].mean(axis=0)
    x=values-mean_x
    rms=float(np.sqrt(np.mean(x[fit]**2))) or 1.0
    x=x/rms; xf,xe=x[fit],x[ev]
    angles=labels[:,None]*(2*math.pi/p)*np.arange(1,(p-1)//2+1)[None,:]
    target=np.stack((np.cos(angles),np.sin(angles)),axis=2)
    gram=xf.T@xf/len(fit)+.001*np.eye(x.shape[1])
    inverse=np.linalg.inv(gram)
    runs=[]
    for permutation in (None,*permutations):
        y=target if permutation is None else target[permutation]
        yf,ye=y[fit],y[ev]; mean_y=yf.mean(axis=0)
        coef=inverse@(xf.T@(yf-mean_y).reshape(len(fit),-1)/len(fit))
        predicted_fit=(xf@coef).reshape(yf.shape)+mean_y
        predicted_eval=(xe@coef).reshape(ye.shape)+mean_y
        fit_den=((yf-mean_y)**2).sum(axis=(0,2)); eval_den=((ye-mean_y)**2).sum(axis=(0,2))
        if np.any(fit_den<=0) or np.any(eval_den<=0):
            raise ValueError('Undefined Fourier target variance')
        fs=1-((yf-predicted_fit)**2).sum(axis=(0,2))/fit_den
        es=1-((ye-predicted_eval)**2).sum(axis=(0,2))/eval_den
        selected=sorted(range(1,len(fs)+1),key=lambda k:(-fs[k-1],k))[:top_k]
        runs.append({'fit_r2':fs,'eval_r2':es,'target_mean':mean_y,
                     'selected_frequencies':selected,'selected_eval_mean_r2':float(es[np.asarray(selected)-1].mean())})
    return {'feature_mean':mean_x,'feature_rms_scalar':rms,'runs':runs}


def hash_ids(array):
    array=np.ascontiguousarray(array,dtype='<i8')
    return hashlib.sha256(b'little-endian-int64-v1\0'+np.asarray(array.shape,dtype='<i8').tobytes()+array.tobytes()).hexdigest()


def edges(train,test,p=113):
    """Independent deterministic split-matched edge enumeration."""
    mask=np.zeros(p*p,dtype=bool);mask[test]=True
    blocks=[]
    for axis in ('a','b'):
        for delta in (1,2,4,8,16,32):
            def dest(ids):
                a,b=ids//p,ids%p
                return ((a+delta)%p)*p+b if axis=='a' else a*p+(b+delta)%p
            target=dest(test); hh=np.column_stack((test[mask[target]],target[mask[target]]))
            chosen={'TH':[],'HH':[]}; groups=[]
            for label in range(p):
                candidates={}
                for role,source in (('TH',train),('HH',test)):
                    target=dest(source)
                    selected=mask[target]&((source//p+source%p)%p==label)
                    pairs=list(zip(source[selected].tolist(),target[selected].tolist()))
                    pairs.sort(key=lambda edge:(hashlib.sha256(f'grokking-symmetry-match-v1|{role}|{axis}|{delta}|{label}|{edge[0]}|{edge[1]}'.encode()).digest(),*edge))
                    candidates[role]=pairs
                count=min(len(v) for v in candidates.values())
                group={'source_sum':label,'th_candidates':len(candidates['TH']),'hh_candidates':len(candidates['HH']),'matched_count':count}
                for role in ('TH','HH'):
                    values=np.asarray(candidates[role][:count],dtype=np.int64).reshape(-1,2)
                    chosen[role].append(values);group[role.lower()+'_edge_ids_sha256']=hash_ids(values)
                groups.append(group)
            blocks.append((axis,delta,hh,{role:np.concatenate(values) for role,values in chosen.items()},groups))
    return blocks


def defect(logits,centered,edge,shift,p=113):
    left,right=edge.T; a,b=centered[left],centered[right]
    residual=b-a[:,(np.arange(p)-shift)%p]
    numerator=float(np.sum(residual*residual));denominator=float(np.sum(a*a)+np.sum(b*b))
    count=len(edge); rms=math.sqrt(denominator/(2*count*p)) if count else 0
    raw=float(np.sum(logits[left]**2)+np.sum(logits[right]**2))
    defined=rms>1e-12 and denominator>0
    return {'count':count,'numerator':numerator,'denominator':denominator,'centered_rms':rms,
            'raw_rms':math.sqrt(raw/(2*count*p)) if count else 0,'energy_defined':defined,
            'value':numerator/denominator if defined else None}


def pooled(records,p=113):
    count=sum(r['count'] for r in records); num=sum(r['numerator'] for r in records); den=sum(r['denominator'] for r in records)
    rms=math.sqrt(den/(2*count*p)) if count else 0
    raw=sum(r['raw_rms']**2*2*r['count']*p for r in records)
    defined=rms>1e-12 and den>0
    return {'count':count,'numerator':num,'denominator':den,'centered_rms':rms,
            'raw_rms':math.sqrt(raw/(2*count*p)) if count else 0,'energy_defined':defined,'value':num/den if defined else None}


def check_array(audit,actual,expected,label,rtol=1e-8,atol=1e-8):
    actual,expected=np.asarray(actual,dtype=np.float64),np.asarray(expected,dtype=np.float64)
    audit.require(actual.shape==expected.shape and np.isfinite(actual).all() and np.isfinite(expected).all(),label+' finite shape')
    error=float(np.max(np.abs(actual-expected))) if actual.size else 0.0
    family=label.split(':')[0];audit.max_error[family]=max(audit.max_error.get(family,0),error)
    audit.check(np.allclose(actual,expected,rtol=rtol,atol=atol),f'{label}: max discrepancy {error}')


def check_defect(audit,actual,expected,label):
    for key,value in expected.items():
        if type(value) is bool:
            audit.check(actual[key] is value,label+key)
        else:
            audit.close(actual[key],value,label+key,rtol=1e-9,atol=1e-9)


def audit_symmetry(audit,arrays,row,edge_blocks,p=113):
    logits=arrays['logits'].astype(np.float64);centered=logits-logits.mean(axis=1,keepdims=True)
    full=row['full_symmetry'];corrects=[];wrongs=[];ths=[];hhs=[];te=[];he=[]
    for axis,delta,hh,matched,groups in edge_blocks:
        observed=full['heldout_symmetry'][axis][str(delta)]
        audit.check(observed['edge_ids_sha256']==hash_ids(hh),'Symmetry HH edge hash')
        for name,shift,target in (('correct',delta,corrects),('wrong_shift',(delta+37)%p,wrongs)):
            value=defect(logits,centered,hh,shift,p);check_defect(audit,observed[name],value,'Symmetry:heldout '+name);target.append(value)
        observed=full['cleanup'][axis][str(delta)]
        audit.check(observed['groups']==groups,'Symmetry matched group IDs/counts')
        values={}
        for role,key,target,edge_list in (('TH','train_to_heldout',ths,te),('HH','heldout_to_heldout',hhs,he)):
            value=defect(logits,centered,matched[role],delta,p);values[role]=value
            check_defect(audit,observed[key],value,'Symmetry:matched '+role)
            audit.check(observed[role.lower()+'_edge_ids_sha256']==hash_ids(matched[role]),'Symmetry matched edge hash')
            target.append(value);edge_list.append(matched[role])
        excess=None if any(values[r]['value'] is None for r in ('TH','HH')) else values['TH']['value']-values['HH']['value']
        audit.close(observed['excess'],excess,'Symmetry:excess')
    for key,values in (('correct',corrects),('wrong_shift',wrongs)):
        check_defect(audit,row['symmetry']['heldout_shift_pooled'][key],pooled(values,p),'Symmetry:heldout pooled')
    for role,key,values,blocks in (('th','train_to_heldout',ths,te),('hh','heldout_to_heldout',hhs,he)):
        expected=pooled(values,p)
        check_defect(audit,full['cleanup']['pooled'][key],expected,'Symmetry:membership pooled')
        audit.check(full['cleanup']['pooled'][role+'_edge_ids_sha256']==hash_ids(np.concatenate(blocks)),'Symmetry pooled edge hash')
    th,hh=pooled(ths,p),pooled(hhs,p)
    excess=None if th['value'] is None or hh['value'] is None else th['value']-hh['value']
    audit.close(row['symmetry']['training_membership_pooled']['excess'],excess,'Symmetry:pooled excess')
    test=np.sort(arrays['test_ids']);mask=np.zeros(p*p,dtype=bool);mask[test]=True
    swapped=test%p*p+test//p;keep=(test//p<test%p)&mask[swapped]
    exchange=np.column_stack((test[keep],swapped[keep]))
    audit.check(full['exchange']['edge_ids_sha256']==hash_ids(exchange),'Symmetry exchange edge hash')
    for name,shift in (('correct',0),('wrong_shift',37)):
        expected=defect(logits,centered,exchange,shift,p)
        check_defect(audit,full['exchange'][name],expected,'Symmetry:exchange')
        check_defect(audit,row['symmetry']['exchange'][name],expected,'Symmetry:compact exchange')
    for split,ids in (('all',np.arange(p*p)),('train',arrays['train_ids']),('test',test)):
        audit.close(row['symmetry']['centered_logit_rms'][split],float(np.sqrt(np.mean(centered[ids]**2))),'RMS:'+split)


def paired_stats(left,right,direction):
    differences=[None if a is None or b is None else a-b for a,b in zip(left,right,strict=True)]
    defined=[x for x in differences if x is not None]
    if len(left)!=5 or len(right)!=5:
        raise ValueError('Exactly five paired seeds required')
    complete=len(defined)==5
    mean=math.fsum(defined)/5 if complete else None
    sd=math.sqrt(math.fsum((x-mean)**2 for x in defined)/4) if complete else None
    positive=sum(x>0 for x in defined) if complete else None
    negative=sum(x<0 for x in defined) if complete else None
    return {'differences':differences,'defined_count':len(defined),'complete_five_seed_aggregate':complete,
            'mean_difference':mean,'sample_sd':sd,'sample_se':sd/math.sqrt(5) if complete else None,
            'positive_count':positive,'negative_count':negative,
            'zero_count':sum(x==0 for x in defined) if complete else None,
            'favorable_count':negative if direction=='lower' else positive if direction=='higher' else None}
