#!/usr/bin/env python3
"""Independent saved new-first-step tensors and bound scalar-history audit."""
import argparse
import gc
import hashlib
import json
import math
import sys

import torch

from common import Audit, BATCH, FLOOR, POLICY, SEEDS, STEPS, batch_contract, finite_tree, numeric

STATE_KEYS = ('model_state','optimizer_state','filter_state','filter_configuration',
              'torch_cpu_rng_state','torch_cuda_rng_states','device_type','cuda_device_count',
              'config','config_sha256','split_identity','parameter_identity')
OLD_ACTIONS = ('native','orthogonal','norm_matched')
ACTIONS = (*OLD_ACTIONS, POLICY)


def norm(value):
    return float(torch.linalg.vector_norm(value.double()))


def cosine(a, b):
    denom = norm(a)*norm(b)
    return float(a.double().dot(b.double()))/denom if denom>FLOOR else None


def tree_hash(value):
    digest = hashlib.sha256()
    def visit(x):
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().contiguous()
            digest.update(b'tensor\0'+str(x.dtype).encode()+b'\0')
            digest.update(json.dumps(list(x.shape)).encode()+b'\0')
            digest.update(x.numpy().tobytes())
        elif isinstance(x, dict):
            digest.update(b'dict\0')
            for k in sorted(x, key=lambda k:(type(k).__name__,str(k))):
                visit(k)
                visit(x[k])
            digest.update(b'enddict\0')
        elif isinstance(x, (list,tuple)):
            digest.update(type(x).__name__.encode()+b'\0')
            for v in x:
                visit(v)
            digest.update(b'endsequence\0')
        else:
            digest.update(type(x).__name__.encode()+b'\0')
            digest.update(json.dumps(x,allow_nan=False).encode()+b'\0')
    visit(value)
    return digest.hexdigest()


def vector(audit, actual, expected, label, relative=1e-9, absolute=1e-13):
    audit.require(actual.shape==expected.shape and bool(torch.isfinite(actual).all()), label+' shape/finiteness')
    error = norm(actual.double()-expected.double())
    reference = norm(expected)
    audit.check(error <= absolute+relative*reference, f'{label}: norm error {error}, relative {error/max(reference,FLOOR)}')
    return error/max(reference,FLOOR)


def norm_law(audit, diagnostics, prefix, direction_norm, target_norm, delivered_norm):
    """Domain flags and realized cast agreement are independently checked."""
    lead = prefix+'_norm_match_'
    audit.require(not (direction_norm==0 and target_norm>0), lead+'invalid zero direction')
    scale = target_norm/max(direction_norm,FLOOR)
    absolute = abs(delivered_norm-target_norm)
    relative = absolute/max(target_norm,FLOOR)
    for key, expected in {'scale':scale, 'absolute_mismatch':absolute,
                          'relative_mismatch':relative, 'relative_tolerance':10*2**-23}.items():
        audit.close(diagnostics[lead+key], expected, lead+key, atol=1e-30)
    for key, expected in {'degenerate':direction_norm<=FLOOR,
                          'denominator_clamped':direction_norm<FLOOR,
                          'exact':direction_norm>=FLOOR or (direction_norm==0 and target_norm==0)}.items():
        audit.check(type(diagnostics[lead+key]) is bool and diagnostics[lead+key]==expected, lead+key)
    if direction_norm>=FLOOR:
        audit.check(relative<=10*2**-23, lead+'postcast tolerance')
    return scale


def history_row(audit, row, step):
    audit.require(row['step']==step and finite_tree(row), 'History step/finiteness')
    d = row['diagnostics']
    p,k,rank = row['P'],row['k'],row['numerical_rank']
    audit.require(p==227313 and type(k) is int and 0<=rank<=k<=200, 'History dimensions')
    audit_adam_scalars(audit,row['adam'],p)
    if d.get('no_basis_identity'):
        audit.require(k==rank==0 and row['singular_values']==[], 'Explicit no-basis identity')
        return {'no_basis_identity':True, 'rank_deficient':False, 'raw_clamped':False}
    singular=row['singular_values']
    audit.require(len(singular)==k and all(numeric(x) and x>=0 for x in singular), 'History singular dimensions')
    audit.check(all(a>=b for a,b in zip(singular,singular[1:])), 'Descending singular spectrum')
    audit.close(row['sigma_max'], singular[0] if singular else 0, 'History sigma max')
    tolerance=max(p,k)*sys.float_info.epsilon*row['sigma_max']
    audit.close(row['tolerance'],tolerance,'History rank tolerance',atol=1e-30)
    audit.check(sum(x>tolerance for x in singular)==rank and row['basis_column_count']==k, 'History threshold rank')
    audit.check(type(row['full_column_rank']) is bool and row['full_column_rank']==(rank==k), 'History full-column flag')
    all_energy=math.fsum(x*x for x in singular)
    lost=math.fsum(x*x for x in singular if x<=tolerance)
    audit.close(row['svd_truncation_relative_frobenius_residual'], math.sqrt(lost/all_energy) if all_energy else 0,'History truncated energy')
    norms=d['action_norms']
    audit.require(set(norms)==set(ACTIONS) and all(numeric(x) and x>=0 for x in norms.values()) and numeric(d['input_norm']) and d['input_norm']>=0,'History action norm roster')
    expected_coefficients=set(ACTIONS if step==1501 else OLD_ACTIONS)
    audit.require(set(row['coefficients'])==expected_coefficients and all(len(v)==rank for v in row['coefficients'].values()),'History coefficient scope')
    raw=d['input_norm']
    audit.close(d['projected_over_raw_norm'], norms['orthogonal']/raw if raw else None,'History rho')
    audit.check(norms['orthogonal']<=raw*(1+10*2**-23)+FLOOR, 'History projected norm bound')
    norm_law(audit,d,'raw',raw,norms['native'],norms[POLICY])
    audit.close(d['raw_norm_match_target_norm'], norms['native'],'History raw target norm')
    # Old projected matching has the same keys without the raw_ prefix.
    renamed={'projected_'+key:value for key,value in d.items() if key.startswith('norm_match_')}
    norm_law(audit,renamed,'projected',norms['orthogonal'],norms['native'],norms['norm_matched'])
    pairs={f'{a}_{b}':(a,b) for i,a in enumerate(ACTIONS) for b in ACTIONS[i+1:]}
    pairs.update({'raw_'+a:('raw',a) for a in ACTIONS})
    audit.require(set(d['pairwise_cosines'])==set(pairs),'History cosine roster')
    for key,(a,b) in pairs.items():
        left=raw if a=='raw' else norms[a]
        value=d['pairwise_cosines'][key]
        audit.check((value is None)==(left*norms[b]<=FLOOR),'History cosine undefined domain')
        if value is not None:
            audit.check(-1-1e-12<=value<=1+1e-12,'History cosine range')
    return {'no_basis_identity':False, 'rank_deficient':rank<k,
            'raw_clamped':d['raw_norm_match_denominator_clamped']}


def audit_adam_scalars(audit,adam,p):
    expected={'m_norm','v_norm','adaptive_direction_norm','adaptive_movement_norm','decay_movement_norm',
              'actual_displacement_norm','decomposition_residual_norm','decomposition_residual_max_abs'}
    audit.require(set(adam)==expected and all(numeric(value) and value>=0 for value in adam.values()),'History Adam norm schema/finiteness')
    audit.close(adam['adaptive_movement_norm'], .001*adam['adaptive_direction_norm'],'History adaptive movement',atol=1e-12)
    residual=adam['decomposition_residual_norm']; maximum=adam['decomposition_residual_max_abs']
    audit.check(maximum<=residual+1e-13 and residual<=math.sqrt(p)*maximum+1e-13,'History residual norm/max bounds')
    actual,adaptive,decay=(adam[key+'_norm'] for key in ('actual_displacement','adaptive_movement','decay_movement'))
    audit.check(actual<=adaptive+decay+residual+1e-12 and actual+residual+1e-12>=abs(adaptive-decay),'History displacement triangle bounds')


def audit_first(audit, completion, by_name):
    seed=completion['seed']
    first=torch.load(audit.receipt(by_name['first-step-tensors.pt']),map_location='cpu',weights_only=True)
    checkpoint=torch.load(audit.receipt(by_name[f'{POLICY}-step-001501.pt']),map_location='cpu',weights_only=True)
    audit.require((first['seed'],first['policy'],first['step'])==(seed,POLICY,1501),'First-tensor identity')
    audit.require(first['schema']=='grokking_raw_direction_acquisition_v1' and first['source_sha256']==completion['source_sha256'],'First-tensor source/schema')
    audit.require(first['parent_checkpoint']==completion['parent_checkpoint'],'First-tensor parent receipt')
    before,after=first['full_before'],first['full_after']
    state_hash=tree_hash({key:before[key] for key in STATE_KEYS})
    audit.check(state_hash==first['restored_scientific_state_sha256']==completion['fork_scientific_state_sha256'],'Scientific restored parent identity')
    audit.require(checkpoint['schema']=='grokking_raw_direction_checkpoint_v1' and checkpoint['first_step_tensors']==by_name['first-step-tensors.pt'],'1501 outer envelope')
    audit.check(tree_hash({key:after[key] for key in STATE_KEYS})==tree_hash({key:checkpoint['native_compatible_state'][key] for key in STATE_KEYS}),'Saved first-after versus1501 checkpoint')
    audit.require(before['step']==1500 and after['step']==1501 and after['filter_state']['step_count']==1501,'First-step counters')
    audit.require(before['config']['seed']==seed and before['config']['arm']=='legacy' and before['config']==after['config'],'Inherited original config')
    g=first['raw_gradient']; result=first['actions']; actions=result['actions']; Q=result['Q'].double()
    V=after['filter_state']['V'].double(); singular=result['singular_values'].double()
    audit.require(g.ndim==1 and len(g)==227313 and g.dtype==torch.float32 and set(actions)==set(ACTIONS),'Separate action tensors')
    audit.require(Q.shape==(len(g),result['numerical_rank']) and V.shape==(len(g),result['k']),'Q/V dimensions')
    gram=Q.T@Q
    gram_error=float((gram-torch.eye(Q.shape[1],dtype=torch.float64)).abs().max()) if Q.shape[1] else 0
    audit.check(gram_error<1e-10,'Q orthogonality')
    compressed=Q.T@V
    vector(audit,Q@compressed,V,'V span reconstruction',relative=1e-9)
    inferred=torch.linalg.svdvals(compressed)
    vector(audit,inferred,singular[:len(inferred)],'Singular values',relative=1e-9)
    native_error=vector(audit,actions['native'],V@(V.T@g.double()),'Native fp64 cross-check',relative=2e-5)
    projected=(Q@(Q.T@g.double())).to(g.dtype)
    vector(audit,actions['orthogonal'],projected,'Cast orthogonal action',relative=1e-8)
    projected_scale=norm(actions['native'])/max(norm(actions['orthogonal']),FLOOR)
    vector(audit,actions['norm_matched'],(actions['orthogonal'].double()*projected_scale).to(g.dtype),'Cast projected-match action',relative=1e-8)
    raw_scale=norm(actions['native'])/max(norm(g),FLOOR)
    expected=(g.double()*raw_scale).to(g.dtype)
    vector(audit,actions[POLICY],expected,'Cast raw-match action',relative=1e-8)
    vector(audit,result['raw_gradient'],g,'Explicit raw-gradient identity',relative=0,absolute=0)
    d=result['diagnostics']; action_norms={name:norm(value) for name,value in actions.items()}
    for name,value in actions.items():
        audit.require(value.shape==g.shape and value.dtype==g.dtype, 'Action dtype/shape '+name)
        audit.close(d['action_norms'][name],action_norms[name],'Action norm '+name)
        vector(audit,result['coefficients'][name],Q.T@value.double(),'Action coefficients '+name,relative=1e-9)
    for i,name in enumerate(ACTIONS):
        audit.close(d['pairwise_cosines']['raw_'+name],cosine(g,actions[name]),'Raw cosine '+name)
        for other in ACTIONS[i+1:]:
            audit.close(d['pairwise_cosines'][name+'_'+other],cosine(actions[name],actions[other]),'Action cosine '+name+other)
    audit.close(d['input_norm'],norm(g),'Raw norm')
    rho=norm(actions['orthogonal'])/norm(g) if norm(g)>0 else None
    audit.close(d['projected_over_raw_norm'],rho,'Projected/raw rho')
    if rho is not None and norm(actions['orthogonal'])*norm(g)>FLOOR:
        audit.close(cosine(g,actions['orthogonal']),rho,'Local orthogonal geometry',rtol=2e-7,atol=2e-7)
    names=[x['name'] for x in before['parameter_identity']]
    before_opt,after_opt=before['optimizer_state'],after['optimizer_state']
    audit.require(len(before_opt['param_groups'])==len(after_opt['param_groups'])==1,'One original Adam group')
    group=before_opt['param_groups'][0]
    audit.require(group==after_opt['param_groups'][0] and not group['amsgrad'] and not group.get('maximize',False),'Unchanged supported Adam group')
    audit.require(group['lr']==.001 and group['weight_decay']==1 and list(group['betas'])==[.9,.98] and group['eps']==1e-8,'Frozen Adam parameters')
    ids=group['params']; audit.require(len(ids)==len(names),'Adam parameter count/order')
    def flat_model(state):
        return torch.cat([state['model_state'][name].flatten().double() for name in names])
    def moment(state,key):
        return torch.cat([state['state'][i][key].flatten().double() for i in ids])
    old,new=flat_model(before),flat_model(after)
    m0,v0=moment(before_opt,'exp_avg'),moment(before_opt,'exp_avg_sq')
    m1,v1=moment(after_opt,'exp_avg'),moment(after_opt,'exp_avg_sq')
    audit.require(all(int(before_opt['state'][i]['step'])==1500 and int(after_opt['state'][i]['step'])==1501 for i in ids),'Adam executes one step including zero semantics')
    beta1,beta2=group['betas']; delivered=actions[POLICY].double()
    m_error=vector(audit,m1,beta1*m0+(1-beta1)*delivered,'Adam first-moment recurrence',relative=5e-7)
    v_error=vector(audit,v1,beta2*v0+(1-beta2)*delivered.square(),'Adam second-moment recurrence',relative=5e-7)
    direction=(m1/(1-beta1**1501))/((v1/(1-beta2**1501)).sqrt()+group['eps'])
    expected={'m':m1,'v':v1,'adaptive_direction':direction,'adaptive_movement':-group['lr']*direction,
              'decay_movement':-group['lr']*group['weight_decay']*old,'actual_displacement':new-old}
    diagnostic=first['adam']
    for name,value in expected.items():
        vector(audit,diagnostic[name],value,'First-step '+name,relative=1e-12)
        audit.close(diagnostic['summary'][name+'_norm'],norm(value),'First-step norm '+name)
    residual=expected['actual_displacement']-expected['adaptive_movement']-expected['decay_movement']
    audit.close(diagnostic['summary']['decomposition_residual_norm'],norm(residual),'First-step residual norm')
    audit.close(diagnostic['summary']['decomposition_residual_max_abs'],float(residual.abs().max()),'First-step residual max')
    summary=audit.read(audit.receipt(by_name['first-step-summary.json']))
    audit.check(summary['first_step_tensors']==by_name['first-step-tensors.pt'] and summary['step_diagnostic']['adam']==diagnostic['summary'],'First tensor/scalar summary identity')
    history_row(audit,summary['step_diagnostic'],1501)
    return {'seed':seed,'rho':rho,'raw_scale':raw_scale,'action_norms':action_norms,
            'raw_vs_projected_cosine':cosine(g,actions['orthogonal']), 'rank':result['numerical_rank'],
            'k':result['k'],'native_fp64_relative_error':native_error,'Q_orthogonality_max_abs':gram_error,
            'm_recurrence_relative_error':m_error,'v_recurrence_relative_error':v_error,
            'adam':diagnostic['summary']}


def run(audit,args):
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    _,completed=batch_contract(audit,args.batch_sha)
    results=[]
    for complete,by_name in completed:
        seed=complete['seed']
        histories=[]
        for step in STEPS:
            value=audit.read(audit.receipt(by_name[f'{POLICY}-through-{step:06d}.json']))
            audit.require((value['seed'],value['policy'],value['step'])==(seed,POLICY,step),'Bound history identity')
            audit.require(value['source_sha256']==complete['source_sha256'] and value['checkpoint']==by_name[f'{POLICY}-step-{step:06d}.pt'],'Bound history/checkpoint/source')
            audit.require([r['step'] for r in value['action_history']]==list(range(1501,step+1)),'Exact bound history prefix')
            audit.require([r['step'] for r in value['evaluation_rows']]==list(range(1550,step+1,50)),'Original evaluation prefix')
            histories.append(value)
        audit.require(histories[0]['action_history']==histories[2]['action_history'][:1] and histories[1]['action_history']==histories[2]['action_history'][:500],'Unchanged history prefixes')
        counts={'no_basis_identity':0,'rank_deficient':0,'raw_clamped':0}
        for step,row in zip(range(1501,2501),histories[2]['action_history'],strict=True):
            for key,value in history_row(audit,row,step).items(): counts[key]+=int(value)
        first=audit_first(audit,complete,by_name)
        stored=audit.read(audit.receipt(by_name['first-step-summary.json']))
        audit.check(stored['step_diagnostic']==histories[0]['action_history'][0],'First-row identity across scalar/history')
        results.append({**first,'history_counts':counts,'history_rows':1000})
        del histories,first
        gc.collect()
        print(json.dumps({'audited_seed':seed,'checks':audit.checks,'errors':len(audit.errors)}),flush=True)
    return {'scope':'New saved first-step tensor and5000 bound scalar-history audit; no training/inference/old tensor replay',
            'batch_completion_sha256':args.batch_sha,'rows':results,'torch_version':torch.__version__,
            'limitations':['Later histories support scalar identities/inequalities only, not dense moment recurrences.',
                           'Raw gradient not regenerated by backpropagation and not compared with old raw tensor.',
                           'No recurrence replay or fresh-seed replication.']}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch-sha',required=True)
    args=parser.parse_args(); audit=Audit(); audit.start('tensor-history-audit-001')
    try:
        payload=run(audit,args)
    except Exception as error:
        audit.errors.append(f'{type(error).__name__}: {error}')
        payload={'scope':'Incomplete saved-data audit; preserved failure'}
    sys.exit(0 if audit.finish(payload) else 1)
