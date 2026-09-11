#!/usr/bin/env python3
"""Independent outcome reaggregation and CPU checkpoint replay after the gate.

Imports neither the training harness nor its summary implementation. All
outcome/test reads follow a complete/18-run execution-record gate. Generated
audit JSON is new and never overwrites existing evidence. No training occurs.
CPU/GPU replay tolerances frozen before execution: CE absolute 5e-5, accuracy
at most one example per evaluated set. These are comparison tolerances, not a
proof that any discrepancy is numerical. Every discrepancy remains recorded.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import struct
import subprocess
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
RESULTS = HERE / 'results'
FREEZE = 'd3e93cce20325e84a0ab98dafb9fa62c9289cf80'
DATA = Path('data/MNIST/raw')
ARMS = ['adamw', 'estimate32_project32', 'estimate128_project32']
WINDOWS = {'all': (101, 2000), 'early': (101, 500), 'late': (1501, 2000)}
SOURCE_MAP = {'neural_harness.py': 'source_sha256', 'protocol.md': 'protocol_sha256',
              'test_harness.py': 'tests_sha256'}


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(test, description):
    if not test:
        raise AssertionError(description)


def approx(a, b, atol=1e-11):
    if a is None or b is None:
        return a is b
    return bool(np.isclose(a, b, rtol=1e-11, atol=atol))


def sign(entry):
    x, t = entry['value'], entry['tolerance']
    require(np.isfinite(x) and np.isfinite(t) and t >= 0, 'finite sign input')
    expected = int(x > t) - int(x < -t)
    require(expected == entry['sign'], 'recomputed signed derivative')
    return expected


def independent_means(run):
    expected_probes = [1, *range(50, 2001, 50), 101]
    expected_probes.sort()
    require([r['step'] for r in run['steps_raw']] == list(range(1, 2001)), 'complete ordered training rows')
    require([r['step'] for r in run['probes_raw']] == expected_probes, 'complete ordered probe rows')
    require([r['step'] for r in run['validation_trajectory']] == list(range(0, 2001, 100)), 'validation schedule')
    require(run['steps'] == 2000 and run['probe_state_checks'] == 42, 'run endpoint/probe check count')
    require(run['all_invariant_gates_passed'] is True and run['instrumented'] is True, 'recorded runtime gates')
    width = 0 if run['arm'] == 'adamw' else (32 if run['arm'] == ARMS[1] else 128)
    for row in run['steps_raw']:
        require(0 <= row['estimation_rank'] <= width, 'estimation cap')
        require(row['estimation_rank'] == run['estimation_rank_by_step'][row['step'] - 1], 'rank trajectory agreement')
        identity = run['arm'] == 'adamw' or row['step'] <= 100 or row['estimation_rank'] == 0
        require(row['filtering_active'] == (not identity), 'identity/filter warmup policy')
        require(row['applied_rank'] == (50890 if identity else min(32, row['estimation_rank'])), 'applied rank separation')
        for kind in ['total', 'decay_subtracted']:
            m = row[kind]
            t, i, o = [m[k]['value'] for k in ['raw_gradient_dot_update', 'in_subspace_contribution', 'outside_contribution']]
            require(approx(t-i-o, m['identity_closure_residual']), 'signed decomposition arithmetic')
            require(m['identity_relative_closure'] <= 1e-4, 'recorded closure limit')
            for k in ['raw_gradient_dot_update', 'in_subspace_contribution', 'outside_contribution']:
                sign(m[k])
            if identity:
                require(m['leakage_squared_fraction'] in (0, None), 'identity leakage')
    for row in run['probes_raw']:
        for component, entry in row['retention'].items():
            denominator = entry['squared_norm']
            numerator = entry['projected_squared_norm']
            require(denominator >= 0 and numerator >= 0, 'nonnegative energies')
            expected = numerator / denominator if denominator else None
            require(approx(expected, entry['squared_norm_retention']), 'retention rederived from energies')
        ret = row['retention']
        c, r = ret['clean']['squared_norm_retention'], ret['corruption_residual']['squared_norm_retention']
        require(approx(None if c is None or r is None else c-r, row['clean_minus_corruption_retention']), 'selectivity difference')
        if run['replacement_probability'] == 0:
            require(ret['corruption_residual']['squared_norm'] == 0 and r is None, 'clean-condition null residual')
        joint = row['joint_geometry']
        require(approx(joint['noisy_energy'], ret['noisy']['squared_norm']), 'joint noisy energy')
        require(approx(joint['projected_noisy_energy'], ret['noisy']['projected_squared_norm']), 'joint projected noisy energy')
        for projected in [False, True]:
            key = 'projected_squared_norm' if projected else 'squared_norm'
            prefix = 'projected_' if projected else ''
            residual = ret['noisy'][key] - ret['clean'][key] - ret['corruption_residual'][key] - 2*joint[prefix+'clean_dot_corruption']
            require(approx(residual, joint[prefix+'energy_closure_residual']), 'joint energy closure arithmetic')
        loss = row['same_training_batch_loss']
        require(approx(loss['after']-loss['before'], loss['change']), 'finite batch loss difference')
        require(approx(loss['tolerance'], 1e-6*max(1, abs(loss['before']))), 'finite loss threshold')
        require(loss['sign'] == int(loss['change'] > loss['tolerance']) - int(loss['change'] < -loss['tolerance']), 'finite loss sign')
    metrics, masks = {}, {}
    def add(name, values, steps):
        finite = [v for v in values if v is not None]
        require(all(np.isfinite(v) for v in finite), 'finite summary values')
        metrics[name] = float(np.mean(finite)) if finite else None
        masks[name] = [step for step, value in zip(steps, values) if value is None]
    for window, (lo, hi) in WINDOWS.items():
        probes = [r for r in run['probes_raw'] if lo <= r['step'] <= hi]
        updates = [r for r in run['steps_raw'] if lo <= r['step'] <= hi]
        ps, us = [r['step'] for r in probes], [r['step'] for r in updates]
        require(len(updates) == hi-lo+1, 'update denominator')
        require(len(probes) == {'all':39,'early':9,'late':10}[window], 'probe denominator')
        for component in ['clean','corruption_residual','auxiliary_clean','noisy']:
            name = f'probe.{window}.{component}_retention'
            entries = [r['retention'][component] for r in probes]
            add(name, [r['projected_squared_norm']/r['squared_norm'] if r['squared_norm'] else None for r in entries], ps)
            denom = sum(r['squared_norm'] for r in entries)
            metrics[name+'.secondary_energy_weighted'] = sum(r['projected_squared_norm'] for r in entries)/denom if denom else None
            masks[name+'.secondary_energy_weighted'] = masks[name]
        add(f'probe.{window}.clean_minus_corruption_retention', [r['clean_minus_corruption_retention'] for r in probes], ps)
        for k in ['clean_dot_corruption','projected_clean_dot_corruption','noisy_energy','projected_noisy_energy']:
            add(f'probe.{window}.joint.{k}', [r['joint_geometry'][k] for r in probes], ps)
        for kind in ['total','decay_subtracted']:
            prefix = f'update.{window}.{kind}'
            entries = [r[kind] for r in updates]
            for k in ['leakage_squared_fraction','cosine_raw_gradient_update']:
                add(prefix+'.'+k, [r[k] for r in entries], us)
            for k in ['raw_gradient_dot_update','in_subspace_contribution','outside_contribution']:
                add(prefix+'.'+k, [r[k]['value'] for r in entries], us)
            event_lists = {k: [] for k in ['current_gradient_ascent','positive_outside_contribution','nominal_leakage_reversal','closure_robust_leakage_reversal','reversal_unresolved_by_closure']}
            for r in entries:
                total, inside, outside = [r[k] for k in ['raw_gradient_dot_update','in_subspace_contribution','outside_contribution']]
                t, i, o = sign(total), sign(inside), sign(outside)
                closure = abs(r['identity_closure_residual'])
                robust = total['value'] > total['tolerance']+closure and inside['value'] < -inside['tolerance']-closure and outside['value'] > outside['tolerance']+closure
                unresolved = t == 1 and i == -1 and not robust
                require(r['closure_robust_leakage_reversal'] == robust and r['reversal_unresolved_by_closure'] == unresolved, 'recomputed closure-aware events')
                for name, event in zip(event_lists, [t == 1, o == 1, t == 1 and i == -1 and o == 1, robust, unresolved]):
                    event_lists[name].append(int(event))
            for name, values in event_lists.items():
                add(prefix+'.'+name+'_frequency', values, us)
            metrics[prefix+'.max_identity_relative_closure'] = max(r['identity_relative_closure'] for r in entries)
        add(f'probe.{window}.finite_training_batch_loss_increase_frequency', [int(r['same_training_batch_loss']['sign']==1) for r in probes], ps)
        add(f'probe.{window}.finite_training_batch_loss_change', [r['same_training_batch_loss']['change'] for r in probes], ps)
    for ck in ['final','validation_selected']:
        for k in ['accuracy','cross_entropy']:
            metrics[f'test.{ck}.{k}'] = run['test'][ck][k]
    for phase in ['final_training_clean','final_training_noisy','final_validation']:
        for k in ['accuracy','cross_entropy']:
            metrics[f'learning.{phase}.{k}'] = run[phase][k]
    metrics['learning.validation_selected_step'] = run['validation_selected_step']
    return metrics, masks


def idx(path, images):
    data = path.read_bytes()
    if images:
        magic, n, rows, cols = struct.unpack('>4I', data[:16])
        require(magic == 2051 and rows == cols == 28 and len(data) == 16+n*784, 'image IDX schema')
        return torch.tensor(np.frombuffer(data, dtype=np.uint8, offset=16).copy().reshape(n,784), dtype=torch.float32)/255
    magic, n = struct.unpack('>2I', data[:8])
    require(magic == 2049 and len(data) == 8+n, 'label IDX schema')
    return torch.tensor(np.frombuffer(data, dtype=np.uint8, offset=8).copy(), dtype=torch.long)


@torch.no_grad()
def evaluate(state, x, y):
    # Direct tensor forward pass avoids importing model or evaluation functions.
    loss, correct = 0., 0
    for start in range(0,len(y),512):
        hidden = torch.relu(x[start:start+512] @ state['0.weight'].T + state['0.bias'])
        logits = hidden @ state['2.weight'].T + state['2.bias']
        labels = y[start:start+512]
        loss += float(torch.nn.functional.cross_entropy(logits, labels, reduction='sum'))
        correct += int((logits.argmax(1)==labels).sum())
    return {'cross_entropy': loss/len(y), 'accuracy': correct/len(y), 'count':len(y), 'correct':correct}


def main():
    execution_path = RESULTS/'execution.json'
    execution = read(execution_path)
    require((execution.get('mode'), execution.get('status'), execution.get('completed_runs')) == ('confirmatory','complete',18), 'OUTCOME GATE CLOSED: do not read raw outcomes/test data')
    output = HERE/'audit-results.json'
    require(not output.exists(), 'refuse existing audit output')
    started = datetime.now(timezone.utc).isoformat()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    require(execution['repository_revision'] == FREEZE, 'frozen pre-execution revision')
    git = lambda *args: subprocess.check_output(['git','-C',str(ROOT),*args])
    commit_time = git('show','-s','--format=%cI',FREEZE).decode().strip()
    require(datetime.fromisoformat(commit_time) <= datetime.fromisoformat(execution['started_utc']), 'commit before execution')
    require(execution['started_utc'] <= execution['all_training_completed_utc'] <= execution['test_first_loaded_utc'] <= execution['completed_utc'], 'recorded train/test chronology')
    hashes = {}
    for name, field in SOURCE_MAP.items():
        digest = sha(HERE/name)
        require(digest == execution[field], 'execution/current source hash '+name)
        require(hashlib.sha256(git('show',FREEZE+':'+(HERE/name).relative_to(ROOT).as_posix())).hexdigest() == digest, 'freeze commit source '+name)
        hashes[name] = digest
    for name,digest in execution['analysis_sources_sha256'].items():
        require(sha(HERE/name)==digest, 'analysis source hash')
        require(hashlib.sha256(git('show',FREEZE+':'+(HERE/name).relative_to(ROOT).as_posix())).hexdigest()==digest, 'analysis pre-execution commit')
        hashes[name] = digest
    require(sha(ROOT/'spectral_filter.py') == execution['canonical_source_sha256'], 'canonical source unchanged')
    for artifact in execution['data_artifacts']+execution['plans']+execution['checkpoints']:
        path=Path(artifact['path'])
        require(path.stat().st_size==artifact['size_bytes'] and sha(path)==artifact['sha256'], 'recorded artifact hash/size '+str(path))
    train_x = idx(DATA/'train-images-idx3-ubyte', True)
    train_y = idx(DATA/'train-labels-idx1-ubyte', False)
    test_x = idx(DATA/'t10k-images-idx3-ubyte', True)
    test_y = idx(DATA/'t10k-labels-idx1-ubyte', False)
    plan_checks, plans = [], {}
    for seed in [0,1,2]:
        random = lambda stream: np.random.default_rng(np.random.SeedSequence([20260906,3,seed,stream]))
        permutation=random(0).permutation(60000)
        expected={'seed':seed, 'initialization_seed':int(random(3).integers(0,2**32,dtype=np.uint32)),
                  'train_indices':permutation[:5000],'validation_indices':permutation[5000:10000],
                  'auxiliary_indices':permutation[10000:15000], 'replacement_uniforms':random(1).random(5000),
                  'replacement_digits':random(2).integers(0,10,size=5000),
                  'training_batches':random(4).integers(0,5000,size=(2000,64)),
                  'primary_probe_batches':random(5).integers(0,5000,size=(2000,256)),
                  'auxiliary_probe_batches':random(6).integers(0,5000,size=(2000,256))}
        with np.load(RESULTS/f'plan-seed{seed}.npz',allow_pickle=False) as saved:
            require(set(saved.files)==set(expected), 'plan schema')
            for name,value in expected.items():
                require(np.array_equal(saved[name],value), 'independent RNG reconstruction '+name)
        plans[seed]=expected
        plan_checks.append({'seed':seed,'all_arrays_exactly_reconstructed':True})
    summaries, replay, raw_hashes = [], [], []
    for seed in [0,1,2]:
        plan=plans[seed]
        clean=train_y[plan['train_indices']]
        vx,vy=train_x[plan['validation_indices']],train_y[plan['validation_indices']]
        for noise in [0.,.9]:
            replaced=torch.tensor(plan['replacement_uniforms']<noise)
            noisy=torch.where(replaced,torch.tensor(plan['replacement_digits']),clean)
            replacement=float(replaced.float().mean())
            incorrect=float((noisy!=clean).float().mean())
            for arm in ARMS:
                tag=f'seed{seed}-noise{noise:g}-{arm}'
                path=RESULTS/(tag+'.json')
                raw_hashes.append({'path':str(path),'sha256':sha(path)})
                run=read(path)
                require((run['seed'],run['replacement_probability'],run['arm'])==(seed,noise,arm), 'all condition identities')
                require(run['realized_replacement_fraction']==replacement and run['realized_incorrect_fraction']==incorrect,'realized corruption reconstruction')
                chosen=min(run['validation_trajectory'],key=lambda r:(r['cross_entropy'],r['step']))
                require(chosen['step']==run['validation_selected_step'], 'earliest minimum validation selection')
                metrics,masks=independent_means(run)
                summaries.append({'seed':seed,'replacement_probability':noise,'arm':arm,'metrics':metrics,'null_steps':masks,'realized_replacement_fraction':replacement,'realized_incorrect_fraction':incorrect})
                ckpath=Path(run['checkpoint_path'])
                require(ckpath.resolve()==(RESULTS/(tag+'-checkpoints.pt')).resolve(),'checkpoint condition path')
                states=torch.load(ckpath,map_location='cpu',weights_only=True)
                require(set(states)=={'final','validation_selected'},'both checkpoint choices')
                for which,state in states.items():
                    require(sum(x.numel() for x in state.values())==50890,'checkpoint parameter count')
                    require(all(torch.isfinite(x).all() for x in state.values()),'finite checkpoint')
                    for dataset,x,y,reference in [('test',test_x,test_y,run['test'][which]),('validation',vx,vy,run['final_validation'] if which=='final' else chosen)]:
                        observed=evaluate(state,x,y)
                        loss_error=abs(observed['cross_entropy']-reference['cross_entropy'])
                        accuracy_error=abs(observed['accuracy']-reference['accuracy'])
                        passed=loss_error<=5e-5 and accuracy_error<=1/len(y)+1e-12 and reference['count']==len(y)
                        replay.append({'tag':tag,'checkpoint':which,'dataset':dataset,'observed':observed,'recorded':reference,'loss_abs_error':loss_error,'accuracy_abs_error':accuracy_error,'passed_tolerance':passed})
    indexed={(r['seed'],r['replacement_probability'],r['arm']):r for r in summaries}
    paired=[]
    for noise in [0.,.9]:
        for treatment,control in [(ARMS[2],ARMS[1]),(ARMS[1],ARMS[0]),(ARMS[2],ARMS[0])]:
            for metric in summaries[0]['metrics']:
                values=[]
                unavailable=[]
                for seed in [0,1,2]:
                    a,b=indexed[seed,noise,treatment],indexed[seed,noise,control]
                    av,bv=a['metrics'][metric],b['metrics'][metric]
                    if av is None or bv is None or a['null_steps'].get(metric)!=b['null_steps'].get(metric):
                        unavailable.append(seed)
                    else:
                        values.append({'seed':seed,'difference':av-bv})
                ds=[r['difference'] for r in values]
                paired.append({'replacement_probability':noise,'treatment':treatment,'control':control,'metric':metric,
                               'paired_differences':values,'unavailable_seeds':unavailable,'mean':float(np.mean(ds)) if len(ds)==3 else None,
                               'median':float(np.median(ds)) if len(ds)==3 else None,'sample_sd_descriptive':float(np.std(ds,ddof=1)) if len(ds)==3 else None})
    summary_check={'available':False}
    if (HERE/'summary.json').exists():
        official=read(HERE/'summary.json')
        offidx={(r['seed'],r['replacement_probability'],r['arm']):r for r in official['seed_summaries']}
        count=0
        for row in summaries:
            other=offidx[row['seed'],row['replacement_probability'],row['arm']]
            require(set(row['metrics'])==set(other['metrics']),'complete independent summary schema')
            for key,value in row['metrics'].items():
                require(approx(value,other['metrics'][key]),'independent seed metric '+key)
                count+=1
        pairs={(r['replacement_probability'],r['treatment'],r['control'],r['metric']):r for r in official['paired_contrasts']}
        for row in paired:
            other=pairs[row['replacement_probability'],row['treatment'],row['control'],row['metric']]
            for key in ['mean','median','sample_sd_descriptive']:
                require(approx(row[key],other[key]),'independent paired '+key)
            require(row['unavailable_seeds']==other['unavailable_seeds'],'independent unavailable pairs')
        summary_check={'available':True,'sha256':sha(HERE/'summary.json'),'seed_metrics_compared':count,'paired_metric_contrasts':len(paired),'all_agree':True}
    audit={'status':'passed' if all(r['passed_tolerance'] for r in replay) else 'checkpoint_replay_discrepancy',
           'started_utc':started,'completed_utc':datetime.now(timezone.utc).isoformat(),
           'audit_source_sha256':sha(Path(__file__)),'execution_sha256':sha(execution_path),'source_freeze':FREEZE,
           'freeze_commit_time':commit_time,'recorded_training_start':execution['started_utc'],'recorded_test_open':execution['test_first_loaded_utc'],
           'source_hashes':hashes,'raw_result_hashes':raw_hashes,'plan_checks':plan_checks,'seed_summaries':summaries,'paired_contrasts':paired,
           'checkpoint_replays':replay,'producer_summary_comparison':summary_check,
           'replay_scope':{'device':'cpu','torch_threads':1,'precision':'float32','ce_abs_tolerance':5e-5,'accuracy_tolerance':'one example per evaluated set','test_replays':36,'validation_replays':36,
                           'limits':'Reaggregated stored scalar/norm/probe records; full per-step vectors/bases and intermediate checkpoints absent, so no independent replay of gradient geometry or entire training. Chronology is verified from frozen source, commit timestamps and recorded execution timestamps, not external access telemetry.'}}
    output.write_text(json.dumps(audit,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'output':str(output),'sha256':sha(output),'status':audit['status'],'seed_summaries':len(summaries),'paired_contrasts':len(paired),'checkpoint_replays':len(replay),'summary_comparison':summary_check},indent=2))


if __name__=='__main__':
    main()
